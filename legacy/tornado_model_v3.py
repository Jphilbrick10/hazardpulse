#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TORNADO PREDICTION MODEL v3 - MULTI-SCALE ENSEMBLE SYSTEM
==========================================================

Building on v2 (Formation AUC=0.935, EF4+ AUC=0.992):

v3 Improvements:
  1. Multi-scale spatial features (1/2/4-degree grids, 100/200/400km radii, directional)
  2. Temporal evolution features (acceleration, time-of-day, outbreak position)
  3. Intensity escalation features (EF trend, width trend, killer proximity, path length)
  4. Enhanced seasonal/climatological features (doy encoding, month*lat, anomaly, trend)
  5. Width-lifetime coherence scaling (L ~ W^0.97 as feature)
  6. Ensemble stacking (L2 logistic + gradient boosted stumps + bagged logistic -> meta)
  7. Severity prediction upgrade (outbreak context, time-of-day, coherence scaling)

No sklearn. No pandas. Manual implementations of all ML.
Data: SPC 1950-2023 tornado database.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import urllib.request
import csv
import io
import os
import sys
import warnings
import math
import time as time_module
warnings.filterwarnings('ignore')

# Force UTF-8
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'coherence_field_results')
os.makedirs(OUT_DIR, exist_ok=True)


# =============================================================================
# UTILITIES
# =============================================================================

def sigmoid(z):
    """Numerically stable sigmoid."""
    z = np.clip(z, -500, 500)
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def compute_auc(y_true, y_score):
    """Manual AUC via trapezoidal rule on ROC."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(-y_score)
    y_true = y_true[order]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tpr_prev = 0.0
    fpr_prev = 0.0
    tp = fp = 0
    auc = 0.0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
        tpr_prev = tpr
        fpr_prev = fpr
    return auc


def compute_brier(y_true, y_score):
    return np.mean((np.asarray(y_score) - np.asarray(y_true, dtype=np.float64)) ** 2)


def compute_bss(y_true, y_score, clim_prob):
    bs_model = compute_brier(y_true, y_score)
    bs_clim = compute_brier(y_true, np.full(len(y_true), clim_prob))
    if bs_clim < 1e-12:
        return 0.0
    return 1.0 - bs_model / bs_clim


def compute_roc_curve(y_true, y_score, n_points=200):
    """Return (fpr, tpr) arrays for plotting."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    thresholds = np.linspace(1.0, 0.0, n_points)
    fpr_list = [0.0]
    tpr_list = [0.0]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.array([0, 1]), np.array([0, 1])
    for th in thresholds:
        pred = (y_score >= th)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)
    fpr_list.append(1.0)
    tpr_list.append(1.0)
    return np.array(fpr_list), np.array(tpr_list)


# =============================================================================
# ML: LOGISTIC REGRESSION WITH L2 (Adam optimizer)
# =============================================================================

def logistic_train(X, y, lr=0.01, lam=1.0, epochs=500, class_weight=None, verbose=False):
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Xs = (X - mu) / sd
    N, D = Xs.shape
    w = np.zeros(D)
    b = 0.0
    sample_w = np.ones(N)
    if class_weight == 'balanced':
        n_pos = y.sum()
        n_neg = N - n_pos
        if n_pos > 0 and n_neg > 0:
            sample_w = np.where(y == 1, N / (2 * n_pos), N / (2 * n_neg))
    m_w, v_w = np.zeros(D), np.zeros(D)
    m_b, v_b = 0.0, 0.0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for epoch in range(epochs):
        z = Xs @ w + b
        p = sigmoid(z)
        residual = (p - y) * sample_w
        dw = (Xs.T @ residual) / N + (lam / N) * w
        db = residual.mean()
        m_w = beta1 * m_w + (1 - beta1) * dw
        v_w = beta2 * v_w + (1 - beta2) * dw ** 2
        m_b = beta1 * m_b + (1 - beta1) * db
        v_b = beta2 * v_b + (1 - beta2) * db ** 2
        t = epoch + 1
        w -= lr * (m_w / (1 - beta1**t)) / (np.sqrt(v_w / (1 - beta2**t)) + eps)
        b -= lr * (m_b / (1 - beta1**t)) / (np.sqrt(v_b / (1 - beta2**t)) + eps)
        if verbose and epoch % 500 == 0:
            loss = -np.mean(sample_w * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))
            print(f"      Epoch {epoch}: loss={loss:.4f}")
    return w, b, mu, sd


def logistic_predict(X, w, b, mu, sd):
    X = np.array(X, dtype=np.float64)
    return sigmoid(((X - mu) / sd) @ w + b)


# =============================================================================
# ML: GRADIENT BOOSTED STUMPS (decision stumps boosted with gradient)
# =============================================================================

class GradientBoostedStumps:
    """
    Gradient-boosted decision stumps for binary classification.
    Each stump splits on one feature at one threshold.
    Uses logistic loss (gradient boosting for classification).
    """
    def __init__(self, n_stumps=300, learning_rate=0.1, min_samples_leaf=20):
        self.n_stumps = n_stumps
        self.lr = learning_rate
        self.min_leaf = min_samples_leaf
        self.stumps = []
        self.init_pred = 0.0

    def _find_best_stump(self, X, residuals, sample_w):
        """Find the single best feature+threshold split."""
        N, D = X.shape
        best_gain = -1e30
        best_feat = 0
        best_thresh = 0.0
        best_left_val = 0.0
        best_right_val = 0.0

        # Try a subset of features for speed (sqrt(D))
        n_try = max(5, int(np.sqrt(D)))
        feat_idx = np.random.choice(D, size=min(n_try, D), replace=False)

        for f in feat_idx:
            col = X[:, f]
            # Try ~20 quantile thresholds
            unique_vals = np.unique(col)
            if len(unique_vals) <= 1:
                continue
            if len(unique_vals) > 20:
                percentiles = np.linspace(5, 95, 20)
                thresholds = np.percentile(col, percentiles)
            else:
                thresholds = unique_vals[:-1]

            for thresh in thresholds:
                left_mask = col <= thresh
                right_mask = ~left_mask
                n_left = left_mask.sum()
                n_right = right_mask.sum()
                if n_left < self.min_leaf or n_right < self.min_leaf:
                    continue

                # Weighted mean of residuals
                wl = sample_w[left_mask]
                wr = sample_w[right_mask]
                left_val = np.sum(residuals[left_mask] * wl) / (np.sum(wl) + 1e-12)
                right_val = np.sum(residuals[right_mask] * wr) / (np.sum(wr) + 1e-12)

                # Gain = reduction in squared error
                gain = (np.sum(wl) * left_val**2 + np.sum(wr) * right_val**2)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = f
                    best_thresh = thresh
                    best_left_val = left_val
                    best_right_val = right_val

        return best_feat, best_thresh, best_left_val, best_right_val

    def fit(self, X, y, verbose=False):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        N = len(y)

        # Class balance weights
        n_pos = y.sum()
        n_neg = N - n_pos
        sample_w = np.where(y == 1, N / (2 * n_pos + 1e-12), N / (2 * n_neg + 1e-12))

        # Initialize with log-odds
        p_avg = np.clip(y.mean(), 0.01, 0.99)
        self.init_pred = np.log(p_avg / (1 - p_avg))
        F = np.full(N, self.init_pred)

        for i in range(self.n_stumps):
            p = sigmoid(F)
            residuals = y - p  # negative gradient of log-loss

            feat, thresh, left_val, right_val = self._find_best_stump(X, residuals, sample_w)
            self.stumps.append((feat, thresh, left_val, right_val))

            # Update predictions
            mask = X[:, feat] <= thresh
            F[mask] += self.lr * left_val
            F[~mask] += self.lr * right_val

            if verbose and (i + 1) % 100 == 0:
                loss = -np.mean(y * np.log(sigmoid(F) + 1e-12) + (1 - y) * np.log(1 - sigmoid(F) + 1e-12))
                print(f"      Stump {i+1}/{self.n_stumps}: loss={loss:.4f}")

    def predict_proba(self, X):
        X = np.array(X, dtype=np.float64)
        F = np.full(X.shape[0], self.init_pred)
        for feat, thresh, left_val, right_val in self.stumps:
            mask = X[:, feat] <= thresh
            F[mask] += self.lr * left_val
            F[~mask] += self.lr * right_val
        return sigmoid(F)


# =============================================================================
# ML: BAGGED LOGISTIC REGRESSION (feature subspace)
# =============================================================================

class BaggedLogistic:
    """Logistic regression bagged over random feature subsets."""
    def __init__(self, n_bags=10, feature_fraction=0.5, lam=1.0, epochs=800):
        self.n_bags = n_bags
        self.ff = feature_fraction
        self.lam = lam
        self.epochs = epochs
        self.models = []

    def fit(self, X, y, verbose=False):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        D = X.shape[1]
        n_feat = max(3, int(D * self.ff))
        rng = np.random.RandomState(123)
        for i in range(self.n_bags):
            feat_idx = rng.choice(D, size=n_feat, replace=False)
            feat_idx.sort()
            X_sub = X[:, feat_idx]
            w, b, mu, sd = logistic_train(X_sub, y, lr=0.01, lam=self.lam,
                                           epochs=self.epochs, class_weight='balanced')
            self.models.append((feat_idx, w, b, mu, sd))
            if verbose and (i + 1) % 5 == 0:
                print(f"      Bag {i+1}/{self.n_bags} done")

    def predict_proba(self, X):
        X = np.array(X, dtype=np.float64)
        preds = np.zeros(X.shape[0])
        for feat_idx, w, b, mu, sd in self.models:
            X_sub = X[:, feat_idx]
            preds += logistic_predict(X_sub, w, b, mu, sd)
        return preds / len(self.models)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_spc_tornadoes():
    url = "https://www.spc.noaa.gov/wcm/data/1950-2023_actual_tornadoes.csv"
    print("  Loading SPC tornado data...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode('utf-8', errors='replace')
    except Exception as e:
        print(f"  Download failed: {e}")
        return None

    reader = csv.DictReader(io.StringIO(raw))
    tornadoes = []
    for row in reader:
        try:
            yr = int(row.get('yr', 0))
            mo = int(row.get('mo', 0))
            dy = int(row.get('dy', 0))
            mag = int(row.get('mag', -9))
            wid = float(row.get('wid', 0))
            plen = float(row.get('len', 0))
            slat = float(row.get('slat', 0))
            slon = float(row.get('slon', 0))
            # Try to get time
            time_str = row.get('time', '').strip()
            hour = -1
            if time_str and ':' in time_str:
                try:
                    parts = time_str.split(':')
                    hour = int(parts[0])
                    minute = int(parts[1]) if len(parts) > 1 else 0
                    hour = hour + minute / 60.0
                except:
                    hour = -1
            # Fatalities / injuries
            fat = 0
            try:
                fat = int(row.get('fat', 0))
            except:
                pass

            if yr < 1980:
                continue
            if mag < 0:
                continue
            if slat < 24 or slat > 50 or slon > -65 or slon < -125:
                continue
            if slat == 0 or slon == 0:
                continue

            tornadoes.append({
                'yr': yr, 'mo': mo, 'dy': dy,
                'mag': mag,
                'width_yd': wid,
                'width_m': wid * 0.9144 if wid > 0 else 0,
                'path_len_mi': plen,
                'path_len_km': plen * 1.60934 if plen > 0 else 0,
                'slat': slat, 'slon': slon,
                'hour': hour,
                'fat': fat,
            })
        except (ValueError, KeyError):
            continue

    print(f"    {len(tornadoes)} tornadoes loaded (1980-2023, CONUS)")
    return tornadoes


# =============================================================================
# GRID AND DATE UTILITIES
# =============================================================================

def date_to_day_number(yr, mo, dy):
    days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    d = (yr - 2000) * 365 + (yr - 2000) // 4
    for m in range(1, mo):
        d += days_in_month[m]
    d += dy
    return d


def day_number_to_approx_doy(dn):
    """Approximate day-of-year from day number."""
    # Rough: dn mod 365
    return ((dn % 365) + 365) % 365


def latlon_to_cell(lat, lon, cell_size=2.0):
    lat_bin = int((lat - 25.0) / cell_size)
    lon_bin = int((lon - (-125.0)) / cell_size)
    n_lat = int((50 - 25) / cell_size)
    n_lon = int(60 / cell_size)
    lat_bin = max(0, min(lat_bin, n_lat - 1))
    lon_bin = max(0, min(lon_bin, n_lon - 1))
    return lat_bin, lon_bin


def haversine_approx_km(lat1, lon1, lat2, lon2):
    """Approximate distance in km."""
    dlat = (lat2 - lat1) * 111.0
    dlon = (lon2 - lon1) * 111.0 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.sqrt(dlat**2 + dlon**2)


# =============================================================================
# INDEX BUILDING (multi-scale)
# =============================================================================

def build_indices(tornadoes):
    """Build all spatial-temporal indices needed for feature engineering."""
    print("  Building multi-scale indices...")

    # Multi-scale cell indices
    indices = {}
    for cell_size in [1.0, 2.0, 4.0]:
        cell_day = defaultdict(list)
        cell_month = defaultdict(list)
        for t in tornadoes:
            dn = date_to_day_number(t['yr'], t['mo'], t['dy'])
            lb, lnb = latlon_to_cell(t['slat'], t['slon'], cell_size)
            cell_day[(lb, lnb, dn)].append(t)
            cell_month[(lb, lnb, t['mo'])].append(t)
        indices[cell_size] = {'cell_day': cell_day, 'cell_month': cell_month}
        print(f"    {cell_size}-degree grid: {len(cell_day)} cell-day events")

    # Day-level index
    day_events = defaultdict(list)
    for t in tornadoes:
        dn = date_to_day_number(t['yr'], t['mo'], t['dy'])
        lb, lnb = latlon_to_cell(t['slat'], t['slon'], 2.0)
        day_events[dn].append((lb, lnb, t))
    indices['day_events'] = day_events

    # Lat/lon level index for radius-based queries
    # Bucket by 1-degree for fast radius lookup
    latlon_day = defaultdict(list)
    for t in tornadoes:
        dn = date_to_day_number(t['yr'], t['mo'], t['dy'])
        lat_i = int(t['slat'])
        lon_i = int(t['slon'])
        latlon_day[(lat_i, lon_i, dn)].append(t)
    indices['latlon_day'] = latlon_day

    return indices


# =============================================================================
# FORMATION FEATURE ENGINEERING (v3)
# =============================================================================

def build_formation_features(tornadoes, indices, train_years, test_years):
    """
    Build formation prediction dataset with v3 features.
    """
    print("\n" + "=" * 70)
    print("PART 1: FORMATION PREDICTION (v3)")
    print("=" * 70)

    cell_size = 2.0
    n_lat = int((50 - 25) / cell_size)
    n_lon = int(60 / cell_size)

    cd2 = indices[2.0]['cell_day']
    cm2 = indices[2.0]['cell_month']
    cd1 = indices[1.0]['cell_day']
    cd4 = indices[4.0]['cell_day']
    day_events = indices['day_events']
    latlon_day = indices['latlon_day']

    days_per_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # Precompute climatological rates (training period)
    print("  Computing climatological rates...")
    train_start, train_end = train_years
    clim_counts = defaultdict(int)
    for t in tornadoes:
        if train_start <= t['yr'] <= train_end:
            lb, lnb = latlon_to_cell(t['slat'], t['slon'], cell_size)
            clim_counts[(lb, lnb, t['mo'])] += 1

    clim_rate = {}
    for lb in range(n_lat):
        for lnb in range(n_lon):
            for mo in range(1, 13):
                n_years = train_end - train_start + 1
                n_days = n_years * days_per_month[mo] if mo <= 12 else 30
                cnt = clim_counts.get((lb, lnb, mo), 0)
                clim_rate[(lb, lnb, mo)] = cnt / max(n_days, 1)

    # Precompute multi-year trend: slope of annual tornado count per cell
    print("  Computing multi-year trends...")
    cell_year_count = defaultdict(lambda: defaultdict(int))
    for t in tornadoes:
        if train_start <= t['yr'] <= train_end:
            lb, lnb = latlon_to_cell(t['slat'], t['slon'], cell_size)
            cell_year_count[(lb, lnb)][t['yr']] += 1

    cell_trend = {}
    for (lb, lnb), yr_counts in cell_year_count.items():
        years = sorted(yr_counts.keys())
        if len(years) >= 5:
            x = np.array(years, dtype=np.float64)
            y = np.array([yr_counts[yr] for yr in years], dtype=np.float64)
            x_mean = x.mean()
            y_mean = y.mean()
            slope = np.sum((x - x_mean) * (y - y_mean)) / (np.sum((x - x_mean)**2) + 1e-12)
            cell_trend[(lb, lnb)] = slope
        else:
            cell_trend[(lb, lnb)] = 0.0

    # -------------------------------------------------------------------------
    # Feature function
    # -------------------------------------------------------------------------
    def get_features(lb, lnb, dn, mo, yr):
        feat = []
        center_lat = 25.0 + lb * cell_size + cell_size / 2
        center_lon = -125.0 + lnb * cell_size + cell_size / 2

        # ===== 1. SEASONAL / CLIMATOLOGICAL (12 features) =====
        doy = (mo - 1) * 30.4 + 15
        feat.append(math.sin(2 * math.pi * doy / 365.25))      # 0
        feat.append(math.cos(2 * math.pi * doy / 365.25))      # 1
        feat.append(math.sin(4 * math.pi * doy / 365.25))      # 2
        feat.append(math.cos(4 * math.pi * doy / 365.25))      # 3

        cr = clim_rate.get((lb, lnb, mo), 0.0)
        feat.append(cr)                                          # 4 clim rate
        feat.append(cr * cr)                                     # 5 clim rate squared

        # Neighbor climatological rate
        nb_cr = 0.0
        n_nb = 0
        for dlb in [-1, 0, 1]:
            for dlnb in [-1, 0, 1]:
                if dlb == 0 and dlnb == 0:
                    continue
                if 0 <= lb + dlb < n_lat and 0 <= lnb + dlnb < n_lon:
                    nb_cr += clim_rate.get((lb + dlb, lnb + dlnb, mo), 0.0)
                    n_nb += 1
        feat.append(nb_cr / max(n_nb, 1))                       # 6 neighbor clim

        # Month * latitude interaction
        feat.append(math.sin(2 * math.pi * doy / 365.25) * lb / n_lat)  # 7
        feat.append(math.cos(2 * math.pi * doy / 365.25) * lb / n_lat)  # 8

        # Multi-year trend
        feat.append(cell_trend.get((lb, lnb), 0.0))             # 9

        # Geographic
        feat.append(lb / n_lat)                                  # 10 lat
        feat.append(lnb / n_lon)                                 # 11 lon

        # ===== 2. MULTI-SCALE TEMPORAL ACTIVITY (18 features) =====
        # Activity in THIS cell at 3 grid scales, multiple lookbacks
        for cs, cd in [(1.0, cd1), (2.0, cd2), (4.0, cd4)]:
            lb_s, lnb_s = latlon_to_cell(center_lat, center_lon, cs)
            for lookback in [1, 3, 7, 14, 30, 90]:
                count = 0
                for d_off in range(1, lookback + 1):
                    count += len(cd.get((lb_s, lnb_s, dn - d_off), []))
                feat.append(count)                               # 12-29

        # ===== 3. SPATIAL PROPAGATION AT MULTIPLE DISTANCES (18 features) =====
        # Using 2-degree grid, distance rings 1/2/3 cells, time lags 1/3/7
        for max_dist in [1, 2, 3]:
            for max_lag in [1, 3, 7]:
                count = 0
                for dlb in range(-max_dist, max_dist + 1):
                    for dlnb in range(-max_dist, max_dist + 1):
                        if dlb == 0 and dlnb == 0:
                            continue
                        if max(abs(dlb), abs(dlnb)) > max_dist:
                            continue
                        nlb = lb + dlb
                        nlnb = lnb + dlnb
                        if 0 <= nlb < n_lat and 0 <= nlnb < n_lon:
                            for d_off in range(0, max_lag + 1):
                                count += len(cd2.get((nlb, nlnb, dn - d_off), []))
                feat.append(count)                               # 30-38

        # ===== 4. DIRECTIONAL PROPAGATION (4 features) =====
        # NE quadrant (typical front approach) vs SW, past 1-2 days
        ne_count = 0
        sw_count = 0
        nw_count = 0
        se_count = 0
        for dlb in range(-3, 4):
            for dlnb in range(-3, 4):
                if dlb == 0 and dlnb == 0:
                    continue
                nlb = lb + dlb
                nlnb = lnb + dlnb
                if 0 <= nlb < n_lat and 0 <= nlnb < n_lon:
                    cnt = 0
                    for d_off in [1, 2]:
                        cnt += len(cd2.get((nlb, nlnb, dn - d_off), []))
                    if cnt > 0:
                        if dlb > 0 and dlnb > 0:
                            ne_count += cnt
                        elif dlb < 0 and dlnb < 0:
                            sw_count += cnt
                        elif dlb > 0 and dlnb < 0:
                            nw_count += cnt
                        else:
                            se_count += cnt
        feat.append(sw_count)                                    # 39 SW approach (typical)
        feat.append(ne_count)                                    # 40
        feat.append(sw_count - ne_count)                         # 41 directional bias
        # Front speed proxy: SW yesterday vs this cell today pattern
        feat.append(sw_count + se_count)                         # 42 southern approach

        # ===== 5. OUTBREAK DETECTION (6 features) =====
        for lookback in [0, 1, 2, 3]:
            active_cells = set()
            total_tor = 0
            for d_off in range(0, lookback + 1):
                for (alb, alnb, t_rec) in day_events.get(dn - d_off, []):
                    active_cells.add((alb, alnb))
                    total_tor += 1
            if lookback == 0:
                feat.append(len(active_cells))                   # 43 today's outbreak extent
            else:
                feat.append(len(active_cells))                   # 44-46
        feat.append(np.log1p(total_tor))                         # 47 log total recent

        # Day-of-outbreak position
        # Look backward: how many consecutive days had tornadoes in region?
        outbreak_len = 0
        for d_off in range(1, 8):
            if len(day_events.get(dn - d_off, [])) > 0:
                outbreak_len += 1
            else:
                break
        feat.append(outbreak_len)                                # 48

        # ===== 6. TORNADO ACCELERATION (3 features) =====
        counts_by_day = []
        for d_off in range(4):
            cnt = 0
            for dlb in range(-2, 3):
                for dlnb in range(-2, 3):
                    nlb, nlnb = lb + dlb, lnb + dlnb
                    if 0 <= nlb < n_lat and 0 <= nlnb < n_lon:
                        cnt += len(cd2.get((nlb, nlnb, dn - d_off), []))
            counts_by_day.append(cnt)
        # Acceleration: change from 2 days ago to yesterday
        feat.append(counts_by_day[1] - counts_by_day[2])        # 49
        feat.append(counts_by_day[0] - counts_by_day[1])        # 50 today vs yesterday
        feat.append(np.log1p(counts_by_day[1]))                 # 51 log yesterday

        # ===== 7. INTENSITY ESCALATION (8 features) =====
        # Are nearby tornadoes getting STRONGER?
        ef_vals_recent = []  # last 1 day
        ef_vals_older = []   # 2-3 days ago
        widths_recent = []
        widths_older = []
        path_len_recent = []
        killer_nearby = 0
        for dlb in range(-2, 3):
            for dlnb in range(-2, 3):
                nlb, nlnb = lb + dlb, lnb + dlnb
                if 0 <= nlb < n_lat and 0 <= nlnb < n_lon:
                    for d_off in [0, 1]:
                        for evt in cd2.get((nlb, nlnb, dn - d_off), []):
                            ef_vals_recent.append(evt['mag'])
                            if evt['width_yd'] > 0:
                                widths_recent.append(evt['width_yd'])
                            if evt['path_len_mi'] > 0:
                                path_len_recent.append(evt['path_len_mi'])
                            if evt['fat'] > 0:
                                killer_nearby = 1
                    for d_off in [2, 3]:
                        for evt in cd2.get((nlb, nlnb, dn - d_off), []):
                            ef_vals_older.append(evt['mag'])
                            if evt['width_yd'] > 0:
                                widths_older.append(evt['width_yd'])

        mean_ef_recent = np.mean(ef_vals_recent) if ef_vals_recent else 0.0
        mean_ef_older = np.mean(ef_vals_older) if ef_vals_older else 0.0
        feat.append(mean_ef_recent)                              # 52
        feat.append(mean_ef_recent - mean_ef_older)              # 53 EF escalation

        mean_w_recent = np.mean(widths_recent) if widths_recent else 0.0
        mean_w_older = np.mean(widths_older) if widths_older else 0.0
        feat.append(np.log1p(mean_w_recent))                    # 54
        feat.append(mean_w_recent - mean_w_older)               # 55 width escalation

        feat.append(killer_nearby)                               # 56
        feat.append(np.sum(path_len_recent))                     # 57 total path length nearby

        # Max EF in cell history
        max_ef_7d = 0
        max_ef_30d = 0
        for d_off in range(1, 31):
            for evt in cd2.get((lb, lnb, dn - d_off), []):
                if d_off <= 7 and evt['mag'] > max_ef_7d:
                    max_ef_7d = evt['mag']
                if evt['mag'] > max_ef_30d:
                    max_ef_30d = evt['mag']
        feat.append(max_ef_7d)                                   # 58
        feat.append(max_ef_30d)                                  # 59

        # ===== 8. WIDTH-LIFETIME COHERENCE SCALING (3 features) =====
        # L ~ W^0.97 scaling: predict path length from width
        if widths_recent:
            max_w_m = max(widths_recent) * 0.9144  # to meters
            # Predicted path length (km) from coherence scaling
            # L_predicted = a * W^0.97 where a ~ 0.015 (empirical)
            pred_path_km = 0.015 * (max_w_m ** 0.97) if max_w_m > 0 else 0.0
            feat.append(pred_path_km)                            # 60
            # Width-duration anomaly
            if path_len_recent:
                actual_mean_path = np.mean(path_len_recent) * 1.60934  # mi to km
                feat.append(actual_mean_path - pred_path_km)     # 61 anomaly
            else:
                feat.append(0.0)                                 # 61
            feat.append(np.log1p(max_w_m))                       # 62
        else:
            feat.append(0.0)                                     # 60
            feat.append(0.0)                                     # 61
            feat.append(0.0)                                     # 62

        # ===== 9. CELL-MONTH ANOMALY (2 features) =====
        this_yr_count = 0
        for t_rec in cm2.get((lb, lnb, mo), []):
            if t_rec['yr'] == yr:
                t_dn = date_to_day_number(t_rec['yr'], t_rec['mo'], t_rec['dy'])
                if t_dn < dn:
                    this_yr_count += 1
        avg_count = cr * days_per_month[mo] if 1 <= mo <= 12 else 0
        feat.append(this_yr_count - avg_count)                   # 63 anomaly
        feat.append(this_yr_count)                               # 64

        # ===== 10. INTERACTION TERMS (4 features) =====
        prop_signal = feat[30]  # 1-cell 1-day propagation
        feat.append(cr * prop_signal)                            # 65 clim * prop
        feat.append(cr * mean_ef_recent)                         # 66 clim * severity
        feat.append(feat[10] * feat[0])                          # 67 lat * sin(doy) interaction
        feat.append(float(outbreak_len > 0) * prop_signal)       # 68 outbreak * prop

        # Total: 69 features
        return feat

    # -------------------------------------------------------------------------
    # Generate samples
    # -------------------------------------------------------------------------
    rng = np.random.RandomState(42)

    def generate_samples(start_yr, end_yr, max_neg_ratio=5):
        X_list = []
        y_list = []
        n_pos = 0
        n_neg = 0

        positive_cell_days = set()
        for (lb, lnb, dn), evts in cd2.items():
            if not evts:
                continue
            if start_yr <= evts[0]['yr'] <= end_yr:
                positive_cell_days.add((lb, lnb, dn))

        print(f"    Positive cell-days: {len(positive_cell_days)}")

        pos_by_cell_month = defaultdict(list)
        for (lb, lnb, dn) in positive_cell_days:
            evts = cd2[(lb, lnb, dn)]
            mo = evts[0]['mo']
            pos_by_cell_month[(lb, lnb, mo)].append(dn)

        for (lb, lnb, mo), pos_dns in pos_by_cell_month.items():
            for dn in pos_dns:
                evts = cd2[(lb, lnb, dn)]
                yr = evts[0]['yr']
                feat = get_features(lb, lnb, dn, mo, yr)
                X_list.append(feat)
                y_list.append(1)
                n_pos += 1

            pos_dn_set = set(pos_dns)
            neg_candidates = []
            for yr in range(start_yr, end_yr + 1):
                for dy in range(1, 29, 2):
                    dn_cand = date_to_day_number(yr, mo, dy)
                    if dn_cand not in pos_dn_set and (lb, lnb, dn_cand) not in positive_cell_days:
                        neg_candidates.append((dn_cand, yr))

            n_neg_wanted = min(len(neg_candidates), len(pos_dns) * max_neg_ratio)
            if n_neg_wanted > 0:
                chosen = rng.choice(len(neg_candidates), size=min(n_neg_wanted, len(neg_candidates)), replace=False)
                for idx in chosen:
                    dn_neg, yr_neg = neg_candidates[idx]
                    feat = get_features(lb, lnb, dn_neg, mo, yr_neg)
                    X_list.append(feat)
                    y_list.append(0)
                    n_neg += 1

        print(f"    Total: {n_pos} positive, {n_neg} negative ({n_pos + n_neg} samples)")
        return np.array(X_list, dtype=np.float64), np.array(y_list, dtype=np.float64)

    print("  Generating training samples...")
    X_train, y_train = generate_samples(train_years[0], train_years[1])
    print("  Generating test samples...")
    X_test, y_test = generate_samples(test_years[0], test_years[1])

    # Clean
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    return X_train, y_train, X_test, y_test


# =============================================================================
# ENSEMBLE TRAINING AND PREDICTION
# =============================================================================

def train_ensemble(X_train, y_train, X_test, y_test):
    """Train 3-model ensemble + meta-learner."""
    print(f"\n  Training ensemble (features={X_train.shape[1]}, "
          f"train={len(y_train)}, test={len(y_test)})")
    print(f"    Train pos rate: {y_train.mean():.3f}, Test pos rate: {y_test.mean():.3f}")

    # --- Model A: L2-regularized logistic regression ---
    print("\n  Model A: L2 Logistic Regression...")
    t0 = time_module.time()
    wA, bA, muA, sdA = logistic_train(X_train, y_train, lr=0.01, lam=0.5,
                                        epochs=2000, class_weight='balanced', verbose=True)
    pA_train = logistic_predict(X_train, wA, bA, muA, sdA)
    pA_test = logistic_predict(X_test, wA, bA, muA, sdA)
    aucA = compute_auc(y_test, pA_test)
    print(f"    Model A Test AUC: {aucA:.4f} ({time_module.time()-t0:.1f}s)")

    # --- Model B: Gradient Boosted Stumps ---
    print("\n  Model B: Gradient Boosted Stumps (300)...")
    t0 = time_module.time()
    gbm = GradientBoostedStumps(n_stumps=300, learning_rate=0.1, min_samples_leaf=20)
    gbm.fit(X_train, y_train, verbose=True)
    pB_train = gbm.predict_proba(X_train)
    pB_test = gbm.predict_proba(X_test)
    aucB = compute_auc(y_test, pB_test)
    print(f"    Model B Test AUC: {aucB:.4f} ({time_module.time()-t0:.1f}s)")

    # --- Model C: Bagged Logistic ---
    print("\n  Model C: Bagged Logistic (10 bags, 50% features)...")
    t0 = time_module.time()
    bag = BaggedLogistic(n_bags=10, feature_fraction=0.5, lam=1.0, epochs=800)
    bag.fit(X_train, y_train, verbose=True)
    pC_train = bag.predict_proba(X_train)
    pC_test = bag.predict_proba(X_test)
    aucC = compute_auc(y_test, pC_test)
    print(f"    Model C Test AUC: {aucC:.4f} ({time_module.time()-t0:.1f}s)")

    # --- Meta-learner: logistic on 3 outputs ---
    print("\n  Meta-learner: stacking...")
    X_meta_train = np.column_stack([pA_train, pB_train, pC_train])
    X_meta_test = np.column_stack([pA_test, pB_test, pC_test])

    wM, bM, muM, sdM = logistic_train(X_meta_train, y_train, lr=0.05, lam=0.01,
                                        epochs=1000, class_weight='balanced')
    p_ensemble_train = logistic_predict(X_meta_train, wM, bM, muM, sdM)
    p_ensemble_test = logistic_predict(X_meta_test, wM, bM, muM, sdM)
    auc_ensemble = compute_auc(y_test, p_ensemble_test)

    clim_prob = y_test.mean()
    bss = compute_bss(y_test, p_ensemble_test, clim_prob)

    print(f"\n  ENSEMBLE RESULTS:")
    print(f"  {'='*50}")
    print(f"    Model A (L2 Logistic)    AUC: {aucA:.4f}")
    print(f"    Model B (GBM Stumps)     AUC: {aucB:.4f}")
    print(f"    Model C (Bagged Logistic) AUC: {aucC:.4f}")
    print(f"    Ensemble (meta-stacked)   AUC: {auc_ensemble:.4f}")
    print(f"    Brier Skill Score (vs clim): {bss:+.4f}")

    return {
        'auc_A': aucA, 'auc_B': aucB, 'auc_C': aucC,
        'auc_ensemble': auc_ensemble,
        'bss': bss,
        'p_test': p_ensemble_test,
        'y_test': y_test,
        'p_A': pA_test, 'p_B': pB_test, 'p_C': pC_test,
        'w': wA, 'b': bA, 'mu': muA, 'sd': sdA,  # for feature importance
    }


# =============================================================================
# SEVERITY PREDICTION (v3)
# =============================================================================

def build_severity_model(tornadoes, indices):
    """Severity prediction with v3 features: outbreak context, time-of-day, coherence."""
    print("\n" + "=" * 70)
    print("PART 2: TORNADO SEVERITY PREDICTION v3")
    print("=" * 70)

    cd2 = indices[2.0]['cell_day']
    day_events = indices['day_events']

    valid = [t for t in tornadoes if t['yr'] >= 2000 and t['width_yd'] > 0 and t['path_len_mi'] > 0]
    print(f"  Valid tornadoes: {len(valid)}")

    # Precompute day counts
    day_key_counts = defaultdict(int)
    day_key_max_ef = defaultdict(int)
    for t in valid:
        key = (t['yr'], t['mo'], t['dy'])
        day_key_counts[key] += 1
        if t['mag'] > day_key_max_ef[key]:
            day_key_max_ef[key] = t['mag']

    def severity_features(t):
        feat = []

        # 1. Width (the star feature)
        log_w = np.log1p(t['width_yd'])
        feat.append(log_w)                                # 0
        feat.append(log_w ** 2)                           # 1
        feat.append(log_w ** 3)                           # 2 cubic for threshold effects

        # 2. Path length
        log_pl = np.log1p(t['path_len_mi'])
        feat.append(log_pl)                               # 3
        feat.append(log_pl ** 2)                          # 4

        # 3. Season
        mo = t['mo']
        doy = (mo - 1) * 30.4 + 15
        feat.append(math.sin(2 * math.pi * doy / 365.25))  # 5
        feat.append(math.cos(2 * math.pi * doy / 365.25))  # 6

        # 4. Location
        feat.append(t['slat'])                            # 7
        feat.append(t['slon'])                            # 8

        # 5. Outbreak context
        key = (t['yr'], t['mo'], t['dy'])
        dc = day_key_counts[key]
        feat.append(dc)                                   # 9
        feat.append(np.log1p(dc))                         # 10
        feat.append(day_key_max_ef[key])                  # 11 max EF today

        # 6. Width * outbreak interaction
        feat.append(log_w * np.log1p(dc))                 # 12

        # 7. Time-of-day features
        if t['hour'] >= 0:
            h = t['hour']
            feat.append(math.sin(2 * math.pi * h / 24))  # 13
            feat.append(math.cos(2 * math.pi * h / 24))  # 14
            # Afternoon flag (peak severity 3-8 PM = 15-20)
            feat.append(1.0 if 15 <= h <= 20 else 0.0)   # 15
        else:
            feat.append(0.0)                              # 13
            feat.append(0.0)                              # 14
            feat.append(0.5)                              # 15 unknown -> average

        # 8. Coherence scaling: L ~ W^0.97
        w_m = t['width_m']
        if w_m > 0:
            pred_path_km = 0.015 * (w_m ** 0.97)
            actual_path_km = t['path_len_km']
            feat.append(pred_path_km)                     # 16
            feat.append(actual_path_km - pred_path_km)    # 17 anomaly
            # Width-duration ratio (path/width dimensionless)
            feat.append(np.log1p(actual_path_km / (w_m / 1000 + 0.01)))  # 18
        else:
            feat.append(0.0)
            feat.append(0.0)
            feat.append(0.0)

        # 9. Nearby intensity context
        dn = date_to_day_number(t['yr'], t['mo'], t['dy'])
        lb, lnb = latlon_to_cell(t['slat'], t['slon'], 2.0)
        nearby_ef = []
        nearby_widths = []
        n_lat = int((50 - 25) / 2.0)
        n_lon = int(60 / 2.0)
        for dlb in range(-1, 2):
            for dlnb in range(-1, 2):
                nlb, nlnb = lb + dlb, lnb + dlnb
                if 0 <= nlb < n_lat and 0 <= nlnb < n_lon:
                    for d_off in [0, 1]:
                        for evt in cd2.get((nlb, nlnb, dn - d_off), []):
                            nearby_ef.append(evt['mag'])
                            if evt['width_yd'] > 0:
                                nearby_widths.append(evt['width_yd'])

        feat.append(np.mean(nearby_ef) if nearby_ef else 0.0)   # 19
        feat.append(max(nearby_ef) if nearby_ef else 0.0)       # 20
        feat.append(np.mean(nearby_widths) if nearby_widths else 0.0)  # 21

        # 10. Width * location interactions
        feat.append(log_w * t['slat'] / 50.0)            # 22
        feat.append(log_w * math.sin(2 * math.pi * doy / 365.25))  # 23

        return feat

    # Split
    train_t = [t for t in valid if 2000 <= t['yr'] <= 2016]
    test_t = [t for t in valid if 2017 <= t['yr'] <= 2023]

    print(f"  Train: {len(train_t)}, Test: {len(test_t)}")

    for label, subset in [("Train", train_t), ("Test", test_t)]:
        ef_counts = defaultdict(int)
        for t in subset:
            ef_counts[t['mag']] += 1
        dist_str = ", ".join(f"EF{k}:{v}" for k, v in sorted(ef_counts.items()))
        print(f"    {label} EF dist: {dist_str}")

    results = {}
    for threshold, label in [(2, 'EF2+'), (3, 'EF3+'), (4, 'EF4+')]:
        print(f"\n  --- {label} Severity Model ---")

        X_tr = np.array([severity_features(t) for t in train_t], dtype=np.float64)
        y_tr = np.array([1 if t['mag'] >= threshold else 0 for t in train_t], dtype=np.float64)
        X_te = np.array([severity_features(t) for t in test_t], dtype=np.float64)
        y_te = np.array([1 if t['mag'] >= threshold else 0 for t in test_t], dtype=np.float64)

        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
        X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

        n_pos_tr = y_tr.sum()
        n_pos_te = y_te.sum()
        print(f"    Train: {n_pos_tr:.0f}/{len(y_tr)} pos ({100*n_pos_tr/len(y_tr):.1f}%)")
        print(f"    Test:  {n_pos_te:.0f}/{len(y_te)} pos ({100*n_pos_te/len(y_te):.1f}%)")

        if n_pos_tr < 5 or n_pos_te < 3:
            print(f"    Skipping {label}: too few positive samples")
            continue

        # Ensemble for severity too: logistic + GBM
        lam = 0.3 if threshold <= 2 else 0.05
        w, b_val, mu, sd = logistic_train(X_tr, y_tr, lr=0.05, lam=lam,
                                           epochs=1500, class_weight='balanced')
        pA_te = logistic_predict(X_te, w, b_val, mu, sd)
        pA_tr = logistic_predict(X_tr, w, b_val, mu, sd)

        # GBM for severity
        n_stumps = 200 if threshold <= 3 else 100
        gbm = GradientBoostedStumps(n_stumps=n_stumps, learning_rate=0.05, min_samples_leaf=10)
        gbm.fit(X_tr, y_tr)
        pB_te = gbm.predict_proba(X_te)
        pB_tr = gbm.predict_proba(X_tr)

        # Ensemble average (simple for severity since meta-learner may overfit with few pos)
        p_te = 0.5 * pA_te + 0.5 * pB_te
        p_tr = 0.5 * pA_tr + 0.5 * pB_tr

        auc_tr = compute_auc(y_tr, p_tr)
        auc_te = compute_auc(y_te, p_te)
        clim_rate = y_te.mean()
        bss = compute_bss(y_te, p_te, clim_rate)

        print(f"    Logistic AUC: {compute_auc(y_te, pA_te):.4f}")
        print(f"    GBM AUC:      {compute_auc(y_te, pB_te):.4f}")
        print(f"    Ensemble AUC: {auc_te:.4f}")
        print(f"    BSS:          {bss:+.4f}")

        results[label] = {
            'auc_train': auc_tr, 'auc_test': auc_te,
            'bss': bss, 'p_test': p_te, 'y_test': y_te,
            'clim_rate': clim_rate,
        }

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_figure(formation_results, severity_results):
    print("\n" + "=" * 70)
    print("CREATING FIGURE")
    print("=" * 70)

    fig = plt.figure(figsize=(22, 18))
    fig.suptitle('Tornado Prediction System v3 - Multi-Scale Ensemble',
                 fontsize=16, fontweight='bold', y=0.98)

    # --- Panel 1: Formation ROC ---
    ax1 = fig.add_subplot(2, 3, 1)
    fpr, tpr = compute_roc_curve(formation_results['y_test'], formation_results['p_test'])
    ax1.plot(fpr, tpr, 'b-', linewidth=2,
             label=f'v3 Ensemble (AUC={formation_results["auc_ensemble"]:.3f})')
    # Individual models
    for key, color, name in [('p_A', 'g--', 'L2 Logistic'),
                              ('p_B', 'r--', 'GBM Stumps'),
                              ('p_C', 'm--', 'Bagged LR')]:
        if key in formation_results:
            auc_i = compute_auc(formation_results['y_test'], formation_results[key])
            fpr_i, tpr_i = compute_roc_curve(formation_results['y_test'], formation_results[key])
            ax1.plot(fpr_i, tpr_i, color, linewidth=1, alpha=0.7,
                     label=f'{name} ({auc_i:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Formation Prediction ROC')
    ax1.legend(fontsize=8, loc='lower right')
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Severity ROC ---
    ax2 = fig.add_subplot(2, 3, 2)
    colors = {'EF2+': 'green', 'EF3+': 'orange', 'EF4+': 'red'}
    for label in ['EF2+', 'EF3+', 'EF4+']:
        if label in severity_results:
            sr = severity_results[label]
            fpr_s, tpr_s = compute_roc_curve(sr['y_test'], sr['p_test'])
            ax2.plot(fpr_s, tpr_s, color=colors[label], linewidth=2,
                     label=f'{label} (AUC={sr["auc_test"]:.3f})')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Severity Prediction ROC')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: v2 vs v3 comparison ---
    ax3 = fig.add_subplot(2, 3, 3)
    v2_metrics = {
        'Formation': 0.935,
        'EF2+': 0.873,
        'EF3+': 0.940,
        'EF4+': 0.992,
    }
    v3_metrics = {
        'Formation': formation_results['auc_ensemble'],
    }
    for label in ['EF2+', 'EF3+', 'EF4+']:
        if label in severity_results:
            v3_metrics[label] = severity_results[label]['auc_test']

    labels = list(v2_metrics.keys())
    v2_vals = [v2_metrics[k] for k in labels]
    v3_vals = [v3_metrics.get(k, 0) for k in labels]

    x = np.arange(len(labels))
    width = 0.35
    bars1 = ax3.bar(x - width/2, v2_vals, width, label='v2', color='steelblue', alpha=0.8)
    bars2 = ax3.bar(x + width/2, v3_vals, width, label='v3', color='firebrick', alpha=0.8)

    for bar, val in zip(bars1, v2_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', fontsize=8)
    for bar, val in zip(bars2, v3_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', fontsize=8)

    ax3.set_ylabel('AUC')
    ax3.set_title('v2 vs v3 Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_ylim(0.82, 1.02)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    # --- Panel 4: Brier Skill Scores ---
    ax4 = fig.add_subplot(2, 3, 4)
    bss_labels = ['Formation']
    bss_vals = [formation_results['bss']]
    bss_colors = ['steelblue']
    for label in ['EF2+', 'EF3+', 'EF4+']:
        if label in severity_results:
            bss_labels.append(label)
            bss_vals.append(severity_results[label]['bss'])
            bss_colors.append(colors.get(label, 'gray'))

    ax4.barh(range(len(bss_labels)), bss_vals, color=bss_colors, alpha=0.8)
    ax4.axvline(x=0, color='black', linewidth=1)
    for i, (lbl, val) in enumerate(zip(bss_labels, bss_vals)):
        ax4.text(val + 0.01, i, f'{val:+.3f}', va='center', fontsize=10)
    ax4.set_yticks(range(len(bss_labels)))
    ax4.set_yticklabels(bss_labels)
    ax4.set_xlabel('Brier Skill Score (vs climatology)')
    ax4.set_title('Skill vs Climatology')
    ax4.grid(True, alpha=0.3, axis='x')

    # --- Panel 5: Feature importance (Model A) ---
    ax5 = fig.add_subplot(2, 3, 5)
    if 'w' in formation_results:
        importance = np.abs(formation_results['w'])
        feature_names = [
            'sin(doy)', 'cos(doy)', 'sin2(doy)', 'cos2(doy)',
            'clim_rate', 'clim_sq', 'nb_clim', 'lat*sin', 'lat*cos',
            'trend', 'lat', 'lon',
            # 1-deg activity
            'act1d_1', 'act1d_3', 'act1d_7', 'act1d_14', 'act1d_30', 'act1d_90',
            # 2-deg activity
            'act2d_1', 'act2d_3', 'act2d_7', 'act2d_14', 'act2d_30', 'act2d_90',
            # 4-deg activity
            'act4d_1', 'act4d_3', 'act4d_7', 'act4d_14', 'act4d_30', 'act4d_90',
            # propagation
            'prop1_1', 'prop1_3', 'prop1_7',
            'prop2_1', 'prop2_3', 'prop2_7',
            'prop3_1', 'prop3_3', 'prop3_7',
            # directional
            'SW', 'NE', 'dir_bias', 'south_app',
            # outbreak
            'ob_today', 'ob_1d', 'ob_2d', 'ob_3d', 'log_recent', 'ob_len',
            # acceleration
            'accel_1', 'accel_0', 'log_yest',
            # intensity
            'mean_ef', 'ef_escal', 'log_w_near', 'w_escal',
            'killer', 'path_sum', 'maxef_7d', 'maxef_30d',
            # coherence
            'pred_path', 'path_anom', 'log_maxw',
            # anomaly
            'cell_anom', 'cell_count',
            # interactions
            'clim*prop', 'clim*ef', 'lat*sin2', 'ob*prop',
        ]
        top_k = min(15, len(importance))
        top_idx = np.argsort(-importance)[:top_k]
        top_names = [feature_names[i] if i < len(feature_names) else f'f{i}' for i in top_idx]
        top_vals = importance[top_idx]
        ax5.barh(range(top_k), top_vals[::-1], color='steelblue', alpha=0.8)
        ax5.set_yticks(range(top_k))
        ax5.set_yticklabels(top_names[::-1], fontsize=8)
        ax5.set_xlabel('|Weight| (standardized)')
        ax5.set_title('Top 15 Features (Model A)')
        ax5.grid(True, alpha=0.3, axis='x')

    # --- Panel 6: Summary text ---
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')
    summary_lines = [
        "TORNADO PREDICTION v3 SUMMARY",
        "=" * 40,
        "",
        "Architecture:",
        "  Model A: L2 Logistic Regression",
        "  Model B: 300 Gradient Boosted Stumps",
        "  Model C: 10-bag Feature-Subspace LR",
        "  Meta: Logistic on 3 model outputs",
        "",
        f"Formation AUC: {formation_results['auc_ensemble']:.4f}",
        f"  (v2 was 0.935)",
        f"Formation BSS: {formation_results['bss']:+.4f}",
        "",
    ]
    for label in ['EF2+', 'EF3+', 'EF4+']:
        if label in severity_results:
            sr = severity_results[label]
            summary_lines.append(f"{label} AUC: {sr['auc_test']:.4f}")
    summary_lines += [
        "",
        "v3 New Features:",
        "  Multi-scale grids (1/2/4-deg)",
        "  Directional propagation",
        "  Intensity escalation tracking",
        "  Width-lifetime coherence scaling",
        "  Ensemble stacking",
    ]
    ax6.text(0.05, 0.95, '\n'.join(summary_lines), transform=ax6.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    out_path = os.path.join(OUT_DIR, 'tornado_model_v3.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("TORNADO PREDICTION MODEL v3")
    print("Multi-Scale Ensemble with Coherence Scaling")
    print("=" * 70)
    t_start = time_module.time()

    # Load data
    tornadoes = load_spc_tornadoes()
    if tornadoes is None:
        print("FATAL: Could not load tornado data")
        return

    # Build indices
    indices = build_indices(tornadoes)

    # Formation prediction
    train_years = (2000, 2015)
    test_years = (2016, 2023)
    X_train, y_train, X_test, y_test = build_formation_features(
        tornadoes, indices, train_years, test_years)

    # Train ensemble
    formation_results = train_ensemble(X_train, y_train, X_test, y_test)

    # Severity prediction
    severity_results = build_severity_model(tornadoes, indices)

    # Create figure
    create_figure(formation_results, severity_results)

    # Final summary
    elapsed = time_module.time() - t_start
    print("\n" + "=" * 70)
    print("FINAL RESULTS: v2 vs v3")
    print("=" * 70)
    print(f"{'Metric':<25} {'v2':>10} {'v3':>10} {'Delta':>10}")
    print("-" * 55)

    v2_vals = {'Formation AUC': 0.935, 'Formation BSS': 0.295,
               'EF2+ AUC': 0.873, 'EF3+ AUC': 0.940, 'EF4+ AUC': 0.992}

    v3_vals = {
        'Formation AUC': formation_results['auc_ensemble'],
        'Formation BSS': formation_results['bss'],
    }
    for label in ['EF2+', 'EF3+', 'EF4+']:
        if label in severity_results:
            v3_vals[f'{label} AUC'] = severity_results[label]['auc_test']

    for metric in v2_vals:
        v2 = v2_vals[metric]
        v3 = v3_vals.get(metric, 0.0)
        delta = v3 - v2
        marker = " <-- IMPROVED" if delta > 0.001 else ""
        print(f"  {metric:<23} {v2:>10.4f} {v3:>10.4f} {delta:>+10.4f}{marker}")

    print(f"\n  Total runtime: {elapsed:.1f}s")
    print("  Done.")


if __name__ == '__main__':
    main()
