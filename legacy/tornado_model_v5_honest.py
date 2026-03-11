#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TORNADO PREDICTION MODEL v5 - HONEST EVALUATION
=================================================

AUDIT FIXES APPLIED (all mandatory):
  1. ZERO same-day data: all feature lookbacks start at d_off >= 1
  2. No post-event features in formation model (no EF, width, path, fatalities)
  3. Honest labeling: no fake "ERA5" names, removed circular proxies
  4. Outbreak momentum uses d_off=1 vs d_off=2 only
  5. INITIATION vs CONTINUATION AUC breakdown
  6. Severity model honesty: width is post-event damage survey data
  7. Fair climatology baseline (not hardcoded SPC numbers)
  8. Ensemble selection by training CV (no test peeking)
  9. Temporal CV blocks (by year groups, not random)
 10. Bootstrap with temporal blocks (resample by year)
 11. EF4+ sample size reported with reliability caveats

FEATURES REMOVED:
  - All d_off=0 features (same-day data leak)
  - EF ratings, widths, path lengths from nearby tornadoes (post-event)
  - bulk_shear_proxy_from_widths (circular: uses outcome data)
  - srh_proxy (circular: uses outcome path data)
  - outbreak_momentum using today's count
  - Fatalities/injuries as formation features
  - Property loss as formation feature

FEATURES KEPT (with fixes):
  - Synoptic propagation: tornado COUNTS only, d_off >= 1
  - Directional propagation: NE/SW from LOCATIONS only, d_off >= 1
  - Seasonal/climatological: DOY, month x latitude, cell climatological rate
  - Topographic: Great Plains, Mississippi Valley (disclosed as climatology encoding)
  - Multi-day outbreak: d_off >= 1 counts only
  - lat_season_cape_climatology (honest label, no tornado outcome data)
  - lat_season_lcl_climatology (honest label)
  - gulf_tornado_count (honest label for moisture proxy)

No sklearn. No pandas. Manual ML implementations.
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

FIG_DIR = r'c:\Users\Josh\Projects\hazardpulse\figures\hazards'
os.makedirs(FIG_DIR, exist_ok=True)

np.random.seed(42)

# =============================================================================
# UTILITIES
# =============================================================================

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def compute_auc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(-y_score)
    y_true = y_true[order]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tpr_prev = fpr_prev = 0.0
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


def compute_auc_fast(y_true, y_score):
    """Faster AUC using Mann-Whitney U statistic."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    n_pos = pos_mask.sum()
    n_neg = neg_mask.sum()
    if n_pos == 0 or n_neg == 0:
        return 0.5
    pos_scores = y_score[pos_mask]
    neg_scores = y_score[neg_mask]
    # Rank-based AUC
    all_scores = np.concatenate([pos_scores, neg_scores])
    all_labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    order = np.argsort(all_scores)
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1, dtype=np.float64)
    # Handle ties
    sorted_scores = all_scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j + 1) / 2.0  # 1-indexed average
        ranks[order[i:j]] = avg_rank
        i = j
    pos_rank_sum = ranks[:n_pos].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return auc


def bootstrap_auc_by_year(y_true, y_score, years, n_boot=500, seed=42):
    """Bootstrap AUC resampling by YEAR (temporal blocks), not by sample."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    years = np.asarray(years)
    unique_years = np.unique(years)
    # Precompute year indices for speed
    year_indices = {yr: np.where(years == yr)[0] for yr in unique_years}
    aucs = []
    for _ in range(n_boot):
        boot_years = rng.choice(unique_years, size=len(unique_years), replace=True)
        idx_parts = [year_indices[yr] for yr in boot_years]
        idx = np.concatenate(idx_parts)
        if len(idx) < 10:
            continue
        a = compute_auc_fast(y_true[idx], y_score[idx])
        aucs.append(a)
    aucs = np.sort(aucs)
    if len(aucs) < 20:
        return 0.5, 0.5, 0.5
    return aucs[int(0.025 * len(aucs))], aucs[int(0.975 * len(aucs))], np.mean(aucs)


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
# ML: GRADIENT BOOSTED STUMPS (depth 1)
# =============================================================================

class GradientBoostedStumps:
    def __init__(self, n_stumps=300, learning_rate=0.1, min_samples_leaf=20):
        self.n_stumps = n_stumps
        self.lr = learning_rate
        self.min_leaf = min_samples_leaf
        self.stumps = []
        self.init_pred = 0.0

    def _find_best_stump(self, X, residuals, sample_w):
        N, D = X.shape
        best_gain = -1e30
        best_feat = 0
        best_thresh = 0.0
        best_left_val = 0.0
        best_right_val = 0.0
        n_features_try = min(D, max(10, D // 3))
        feat_indices = np.random.choice(D, size=n_features_try, replace=False)
        for f in feat_indices:
            vals = X[:, f]
            sorted_idx = np.argsort(vals)
            sorted_vals = vals[sorted_idx]
            sorted_res = residuals[sorted_idx]
            sorted_w = sample_w[sorted_idx]
            cum_wr = np.cumsum(sorted_w * sorted_res)
            cum_w = np.cumsum(sorted_w)
            total_wr = cum_wr[-1]
            total_w = cum_w[-1]
            for i in range(self.min_leaf, N - self.min_leaf):
                if sorted_vals[i] == sorted_vals[i - 1]:
                    continue
                left_w = cum_w[i - 1]
                right_w = total_w - left_w
                if left_w < 1e-12 or right_w < 1e-12:
                    continue
                left_val = cum_wr[i - 1] / left_w
                right_val = (total_wr - cum_wr[i - 1]) / right_w
                gain = left_w * left_val ** 2 + right_w * right_val ** 2
                if gain > best_gain:
                    best_gain = gain
                    best_feat = f
                    best_thresh = (sorted_vals[i] + sorted_vals[i - 1]) / 2
                    best_left_val = left_val
                    best_right_val = right_val
        return best_feat, best_thresh, best_left_val, best_right_val

    def fit(self, X, y, verbose=False):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        N = len(y)
        n_pos = y.sum()
        n_neg = N - n_pos
        sample_w = np.where(y == 1, N / (2 * n_pos + 1e-12), N / (2 * n_neg + 1e-12))
        p_mean = y.mean()
        self.init_pred = np.log(p_mean / (1 - p_mean + 1e-12) + 1e-12)
        F = np.full(N, self.init_pred)
        for i in range(self.n_stumps):
            p = sigmoid(F)
            residuals = y - p
            feat, thresh, lv, rv = self._find_best_stump(X, residuals, sample_w)
            self.stumps.append((feat, thresh, lv, rv))
            mask = X[:, feat] <= thresh
            F[mask] += self.lr * lv
            F[~mask] += self.lr * rv
            if verbose and (i + 1) % 100 == 0:
                auc = compute_auc(y, sigmoid(F))
                print(f"      Stump {i+1}: train AUC = {auc:.4f}")

    def predict_proba(self, X):
        X = np.array(X, dtype=np.float64)
        F = np.full(X.shape[0], self.init_pred)
        for feat, thresh, lv, rv in self.stumps:
            mask = X[:, feat] <= thresh
            F[mask] += self.lr * lv
            F[~mask] += self.lr * rv
        return sigmoid(F)


# =============================================================================
# ML: GRADIENT BOOSTED TREES (depth > 1)
# =============================================================================

class GradientBoostedTrees:
    def __init__(self, n_trees=100, max_depth=3, learning_rate=0.05, min_samples_leaf=20):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.lr = learning_rate
        self.min_leaf = min_samples_leaf
        self.trees = []
        self.init_pred = 0.0

    def _build_tree(self, X, residuals, sample_w, depth=0):
        N = len(residuals)
        if depth >= self.max_depth or N < 2 * self.min_leaf:
            val = np.sum(sample_w * residuals) / (np.sum(sample_w) + 1e-12)
            return ('leaf', val)
        best_gain = -1e30
        best_feat = 0
        best_thresh = 0.0
        D = X.shape[1]
        n_try = min(D, max(8, D // 3))
        feats = np.random.choice(D, size=n_try, replace=False)
        for f in feats:
            vals = X[:, f]
            sorted_idx = np.argsort(vals)
            sv = vals[sorted_idx]
            sr = residuals[sorted_idx]
            sw = sample_w[sorted_idx]
            cum_wr = np.cumsum(sw * sr)
            cum_w = np.cumsum(sw)
            total_wr = cum_wr[-1]
            total_w = cum_w[-1]
            for i in range(self.min_leaf, N - self.min_leaf):
                if sv[i] == sv[i - 1]:
                    continue
                lw = cum_w[i - 1]
                rw = total_w - lw
                if lw < 1e-12 or rw < 1e-12:
                    continue
                lv = cum_wr[i - 1] / lw
                rv = (total_wr - cum_wr[i - 1]) / rw
                gain = lw * lv ** 2 + rw * rv ** 2
                if gain > best_gain:
                    best_gain = gain
                    best_feat = f
                    best_thresh = (sv[i] + sv[i - 1]) / 2
        if best_gain <= 0:
            val = np.sum(sample_w * residuals) / (np.sum(sample_w) + 1e-12)
            return ('leaf', val)
        mask = X[:, best_feat] <= best_thresh
        left = self._build_tree(X[mask], residuals[mask], sample_w[mask], depth + 1)
        right = self._build_tree(X[~mask], residuals[~mask], sample_w[~mask], depth + 1)
        return ('split', best_feat, best_thresh, left, right)

    def _predict_tree(self, node, x):
        if node[0] == 'leaf':
            return node[1]
        if x[node[1]] <= node[2]:
            return self._predict_tree(node[3], x)
        return self._predict_tree(node[4], x)

    def fit(self, X, y, verbose=False):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        N = len(y)
        n_pos = y.sum()
        n_neg = N - n_pos
        sample_w = np.where(y == 1, N / (2 * n_pos + 1e-12), N / (2 * n_neg + 1e-12))
        p_mean = y.mean()
        self.init_pred = np.log(p_mean / (1 - p_mean + 1e-12) + 1e-12)
        F = np.full(N, self.init_pred)
        for i in range(self.n_trees):
            p = sigmoid(F)
            residuals = y - p
            tree = self._build_tree(X, residuals, sample_w)
            self.trees.append(tree)
            for j in range(N):
                F[j] += self.lr * self._predict_tree(tree, X[j])
            if verbose and (i + 1) % 50 == 0:
                auc = compute_auc(y, sigmoid(F))
                print(f"      Tree {i+1}: train AUC = {auc:.4f}")

    def predict_proba(self, X):
        X = np.array(X, dtype=np.float64)
        N = X.shape[0]
        F = np.full(N, self.init_pred)
        for tree in self.trees:
            for j in range(N):
                F[j] += self.lr * self._predict_tree(tree, X[j])
        return sigmoid(F)


# =============================================================================
# ML: BAGGED LOGISTIC
# =============================================================================

class BaggedLogistic:
    def __init__(self, n_bags=10, feature_fraction=0.5, lam=1.0, epochs=800):
        self.n_bags = n_bags
        self.feat_frac = feature_fraction
        self.lam = lam
        self.epochs = epochs
        self.models = []

    def fit(self, X, y, verbose=False):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        N, D = X.shape
        n_feat = max(5, int(D * self.feat_frac))
        rng = np.random.RandomState(42)
        for i in range(self.n_bags):
            idx = rng.choice(N, size=N, replace=True)
            feat_idx = np.sort(rng.choice(D, size=n_feat, replace=False))
            Xi = X[np.ix_(idx, feat_idx)]
            yi = y[idx]
            w, b, mu, sd = logistic_train(Xi, yi, lr=0.01, lam=self.lam,
                                           epochs=self.epochs, class_weight='balanced')
            self.models.append((w, b, mu, sd, feat_idx))
            if verbose and (i + 1) % 5 == 0:
                print(f"      Bag {i+1}/{self.n_bags} done")

    def predict_proba(self, X):
        X = np.array(X, dtype=np.float64)
        preds = np.zeros(X.shape[0])
        for w, b, mu, sd, feat_idx in self.models:
            preds += logistic_predict(X[:, feat_idx], w, b, mu, sd)
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
            elat = float(row.get('elat', 0))
            elon = float(row.get('elon', 0))
            time_str = row.get('time', '').strip()
            hour = -1.0
            if time_str and ':' in time_str:
                try:
                    parts = time_str.split(':')
                    hour = int(parts[0]) + int(parts[1]) / 60.0
                except Exception:
                    hour = -1.0
            fat = 0
            inj = 0
            loss = 0.0
            try:
                fat = int(row.get('fat', 0))
            except Exception:
                pass
            try:
                inj = int(row.get('inj', 0))
            except Exception:
                pass
            try:
                loss = float(row.get('loss', 0))
            except Exception:
                pass

            if yr < 1980 or mag < 0:
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
                'elat': elat, 'elon': elon,
                'hour': hour,
                'fat': fat, 'inj': inj, 'loss': loss,
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


def latlon_to_cell(lat, lon, cell_size=2.0):
    lat_bin = int((lat - 25.0) / cell_size)
    lon_bin = int((lon - (-125.0)) / cell_size)
    n_lat = int((50 - 25) / cell_size)
    n_lon = int(60 / cell_size)
    lat_bin = max(0, min(lat_bin, n_lat - 1))
    lon_bin = max(0, min(lon_bin, n_lon - 1))
    return lat_bin, lon_bin


def haversine_approx_km(lat1, lon1, lat2, lon2):
    dlat = (lat2 - lat1) * 111.0
    dlon = (lon2 - lon1) * 111.0 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.sqrt(dlat**2 + dlon**2)


# =============================================================================
# TOPOGRAPHIC / GEOGRAPHIC PROXY FUNCTIONS
# (These encode climatology, not real-time data. Disclosed as such.)
# =============================================================================

def distance_to_appalachians(lat, lon):
    if 34 <= lat <= 40:
        frac = (lat - 34) / 6.0
        ridge_lon = -84 + frac * 4
        return abs(lon - ridge_lon) * 111.0 * math.cos(math.radians(lat))
    elif lat < 34:
        return haversine_approx_km(lat, lon, 34, -84)
    else:
        return haversine_approx_km(lat, lon, 40, -80)

def distance_to_rockies(lat, lon):
    rock_lon = -105.0 if 35 <= lat <= 45 else (-107.0 if lat < 35 else -110.0)
    return abs(lon - rock_lon) * 111.0 * math.cos(math.radians(lat))

def distance_to_ozarks(lat, lon):
    return haversine_approx_km(lat, lon, 36.0, -93.0)

def great_plains_indicator(lat, lon):
    if 30 <= lat <= 45 and -103 <= lon <= -95:
        return 1.0
    lat_dist = max(0, max(30 - lat, lat - 45)) / 5.0
    lon_dist = max(0, max(-103 - lon, lon - (-95))) / 5.0
    d = math.sqrt(lat_dist**2 + lon_dist**2)
    return max(0.0, 1.0 - d)

def mississippi_valley_indicator(lat, lon):
    if -92 <= lon <= -88:
        return 1.0
    d = min(abs(lon - (-92)), abs(lon - (-88))) / 4.0
    return max(0.0, 1.0 - d)

def elevation_proxy(lat, lon):
    base = 200.0
    rock_dist = abs(lon - (-105.0))
    if rock_dist < 5:
        base += (5 - rock_dist) / 5.0 * 2000
    if -105 <= lon <= -95:
        base += (abs(lon + 95) / 10.0) * 1000 + 300
    if -84 <= lon <= -78 and 34 <= lat <= 40:
        base += 800
    return base


# =============================================================================
# HONEST CLIMATOLOGICAL PROXY FUNCTIONS
# (No tornado outcome data used. Vary only by lat/lon/season.)
# =============================================================================

def lat_season_cape_climatology(lat, mo, doy):
    """
    CAPE climatological proxy from latitude and season ONLY.
    HONEST LABEL: This is NOT ERA5 data. It is a smooth function of
    latitude and day-of-year that approximates the climatological mean
    CAPE field. Actual CAPE varies hugely day-to-day; this captures
    only the seasonal envelope.
    """
    lat_factor = max(0, (45 - lat) / 20.0)
    seasonal = 0.5 + 0.5 * math.sin(2 * math.pi * (doy - 60) / 365.25)
    month_peak = max(0, 1.0 - abs(mo - 5.5) / 4.0)
    return lat_factor * seasonal * month_peak


def lat_season_lcl_climatology(doy, lat):
    """
    LCL climatological proxy from season and latitude ONLY.
    HONEST LABEL: This is a smooth climatological approximation,
    not real-time LCL data.
    """
    moisture_seasonal = 0.5 + 0.5 * math.sin(2 * math.pi * (doy - 90) / 365.25)
    moisture_lat = max(0, (40 - lat) / 15.0)
    lcl = 1.0 - moisture_seasonal * moisture_lat
    return lcl


# =============================================================================
# INDEX BUILDING
# =============================================================================

def build_indices(tornadoes):
    print("  Building multi-scale indices...")

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

    # Day-level index: COUNTS ONLY (no post-event data)
    day_counts = defaultdict(int)
    day_cells = defaultdict(set)
    for t in tornadoes:
        dn = date_to_day_number(t['yr'], t['mo'], t['dy'])
        lb, lnb = latlon_to_cell(t['slat'], t['slon'], 2.0)
        day_counts[dn] += 1
        day_cells[dn].add((lb, lnb))
    indices['day_counts'] = day_counts
    indices['day_cells'] = day_cells

    # Gulf Coast feeder region index (lat < 35, lon > -100)
    # HONEST LABEL: This is just counting tornadoes in Gulf region
    gulf_day = defaultdict(int)
    for t in tornadoes:
        if t['slat'] < 35 and t['slon'] > -100:
            dn = date_to_day_number(t['yr'], t['mo'], t['dy'])
            gulf_day[dn] += 1
    indices['gulf_day'] = gulf_day

    return indices


# =============================================================================
# FORMATION FEATURE ENGINEERING (v5 HONEST)
# =============================================================================
# CRITICAL: Every feature uses ONLY d_off >= 1 (yesterday or earlier).
# CRITICAL: No EF, width, path length, fatalities, injuries, or property loss
#           from nearby tornadoes (these are post-event damage survey data).
# Only tornado COUNT and LOCATION from preliminary reports are used.

def build_formation_features(tornadoes, indices, train_years, test_years):
    print("\n" + "=" * 70)
    print("PART 1: FORMATION PREDICTION (v5 HONEST)")
    print("  - ZERO same-day data (all lookbacks start at d_off >= 1)")
    print("  - NO post-event features (no EF, width, path, fatalities)")
    print("  - Honest proxy labels (no fake ERA5)")
    print("=" * 70)

    cell_size = 2.0
    n_lat = int((50 - 25) / cell_size)
    n_lon = int(60 / cell_size)

    cd2 = indices[2.0]['cell_day']
    cd1 = indices[1.0]['cell_day']
    cd4 = indices[4.0]['cell_day']
    day_counts = indices['day_counts']
    day_cells = indices['day_cells']
    gulf_day = indices['gulf_day']

    days_per_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # Precompute climatological rates (training period only)
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
                n_days = n_years * days_per_month[mo] if 1 <= mo <= 12 else 30
                cnt = clim_counts.get((lb, lnb, mo), 0)
                clim_rate[(lb, lnb, mo)] = cnt / max(n_days, 1)

    # Multi-year trend
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

    # Nearby-tornado-in-past-3-days flag for initiation/continuation split
    # Precomputed per cell-day
    def had_nearby_tornado_past_3d(lb, lnb, dn):
        """Check if any tornado occurred in adjacent cells in past 3 days (d_off 1,2,3)."""
        for d_off in range(1, 4):
            for dlb in range(-2, 3):
                for dlnb in range(-2, 3):
                    nlb, nlnb = lb + dlb, lnb + dlnb
                    if 0 <= nlb < n_lat and 0 <= nlnb < n_lon:
                        if len(cd2.get((nlb, nlnb, dn - d_off), [])) > 0:
                            return True
        return False

    # -------------------------------------------------------------------------
    # v5 HONEST Feature function
    # -------------------------------------------------------------------------
    FEATURE_NAMES = []

    def get_features(lb, lnb, dn, mo, yr):
        feat = []
        center_lat = 25.0 + lb * cell_size + cell_size / 2
        center_lon = -125.0 + lnb * cell_size + cell_size / 2
        doy = (mo - 1) * 30.4 + 15

        # =================================================================
        # BLOCK 1: SEASONAL / CLIMATOLOGICAL (12 features)
        # These use only calendar date + cell location. No tornado data.
        # =================================================================
        feat.append(math.sin(2 * math.pi * doy / 365.25))      # 0: sin(DOY)
        feat.append(math.cos(2 * math.pi * doy / 365.25))      # 1: cos(DOY)
        feat.append(math.sin(4 * math.pi * doy / 365.25))      # 2: sin(2*DOY)
        feat.append(math.cos(4 * math.pi * doy / 365.25))      # 3: cos(2*DOY)

        cr = clim_rate.get((lb, lnb, mo), 0.0)
        feat.append(cr)                                          # 4: cell climatological rate
        feat.append(cr * cr)                                     # 5: clim rate squared

        nb_cr = 0.0
        n_nb = 0
        for dlb in [-1, 0, 1]:
            for dlnb in [-1, 0, 1]:
                if dlb == 0 and dlnb == 0:
                    continue
                if 0 <= lb + dlb < n_lat and 0 <= lnb + dlnb < n_lon:
                    nb_cr += clim_rate.get((lb + dlb, lnb + dlnb, mo), 0.0)
                    n_nb += 1
        feat.append(nb_cr / max(n_nb, 1))                       # 6: neighbor clim rate

        feat.append(math.sin(2 * math.pi * doy / 365.25) * lb / n_lat)  # 7
        feat.append(math.cos(2 * math.pi * doy / 365.25) * lb / n_lat)  # 8
        feat.append(cell_trend.get((lb, lnb), 0.0))             # 9: multi-year trend
        feat.append(lb / n_lat)                                  # 10: normalized latitude
        feat.append(lnb / n_lon)                                 # 11: normalized longitude

        # =================================================================
        # BLOCK 2: MULTI-SCALE TEMPORAL ACTIVITY (18 features)
        # COUNTS ONLY, d_off >= 1 (starts from YESTERDAY)
        # =================================================================
        for cs, cd in [(1.0, cd1), (2.0, cd2), (4.0, cd4)]:
            lb_s, lnb_s = latlon_to_cell(center_lat, center_lon, cs)
            for lookback_end in [1, 3, 7, 14, 30, 90]:
                count = 0
                for d_off in range(1, lookback_end + 1):  # d_off starts at 1!
                    count += len(cd.get((lb_s, lnb_s, dn - d_off), []))
                feat.append(count)

        # =================================================================
        # BLOCK 3: SPATIAL PROPAGATION (9 features)
        # COUNTS ONLY, d_off >= 1
        # =================================================================
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
                            for d_off in range(1, max_lag + 1):  # d_off >= 1!
                                count += len(cd2.get((nlb, nlnb, dn - d_off), []))
                feat.append(count)

        # =================================================================
        # BLOCK 4: DIRECTIONAL PROPAGATION (4 features)
        # COUNTS ONLY from LOCATIONS, d_off >= 1
        # No widths/EFs used.
        # =================================================================
        ne_count = sw_count = nw_count = se_count = 0
        for dlb in range(-3, 4):
            for dlnb in range(-3, 4):
                if dlb == 0 and dlnb == 0:
                    continue
                nlb, nlnb = lb + dlb, lnb + dlnb
                if 0 <= nlb < n_lat and 0 <= nlnb < n_lon:
                    cnt = 0
                    for d_off in [1, 2]:  # d_off >= 1!
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
        feat.append(sw_count)                                    # SW (upstream)
        feat.append(ne_count)                                    # NE (downstream)
        feat.append(sw_count - ne_count)                         # directional bias
        feat.append(sw_count + se_count)                         # south total

        # =================================================================
        # BLOCK 5: OUTBREAK DETECTION (5 features)
        # COUNTS ONLY, d_off >= 1
        # NO same-day outbreak data.
        # =================================================================
        for lookback in [1, 2, 3]:
            active_cells = set()
            for d_off in range(1, lookback + 1):  # d_off >= 1!
                for cell in day_cells.get(dn - d_off, set()):
                    active_cells.add(cell)
            feat.append(len(active_cells))

        # Total tornado count in past 3 days (no today)
        total_past_3d = sum(day_counts.get(dn - d, 0) for d in range(1, 4))
        feat.append(np.log1p(total_past_3d))

        # Outbreak length: consecutive past days with tornadoes (d_off >= 1)
        outbreak_len = 0
        for d_off in range(1, 8):
            if day_counts.get(dn - d_off, 0) > 0:
                outbreak_len += 1
            else:
                break
        feat.append(outbreak_len)

        # =================================================================
        # BLOCK 6: TORNADO ACCELERATION (2 features)
        # d_off >= 1 only. Compare yesterday vs day-before-yesterday.
        # =================================================================
        counts_by_day = []
        for d_off in range(1, 5):  # d_off 1,2,3,4
            cnt = 0
            for dlb in range(-2, 3):
                for dlnb in range(-2, 3):
                    nlb, nlnb = lb + dlb, lnb + dlnb
                    if 0 <= nlb < n_lat and 0 <= nlnb < n_lon:
                        cnt += len(cd2.get((nlb, nlnb, dn - d_off), []))
            counts_by_day.append(cnt)
        # counts_by_day[0] = yesterday, [1] = 2 days ago, etc.
        feat.append(counts_by_day[0] - counts_by_day[1])  # yesterday vs day before
        feat.append(np.log1p(counts_by_day[0]))            # yesterday count

        # =================================================================
        # BLOCK 7: CELL-MONTH ANOMALY (2 features)
        # Uses only data strictly before target day
        # =================================================================
        cm2 = indices[2.0]['cell_month']
        this_yr_count = 0
        for t_rec in cm2.get((lb, lnb, mo), []):
            if t_rec['yr'] == yr:
                t_dn = date_to_day_number(t_rec['yr'], t_rec['mo'], t_rec['dy'])
                if t_dn < dn:  # strictly before today
                    this_yr_count += 1
        avg_count = cr * days_per_month[mo] if 1 <= mo <= 12 else 0
        feat.append(this_yr_count - avg_count)
        feat.append(this_yr_count)

        # =================================================================
        # BLOCK 8: HONEST CLIMATOLOGICAL PROXIES (4 features)
        # lat_season_cape_climatology: varies by lat/season only
        # lat_season_lcl_climatology: varies by lat/season only
        # gulf_tornado_count: yesterday's Gulf region count (d_off >= 1)
        # =================================================================
        cape_clim = lat_season_cape_climatology(center_lat, mo, doy)
        feat.append(cape_clim)
        feat.append(cape_clim * cr)  # CAPE_clim x climatological rate

        feat.append(lat_season_lcl_climatology(doy, center_lat))

        # Gulf tornado count: d_off >= 1 only
        gulf_1d = gulf_day.get(dn - 1, 0)  # yesterday only, NOT today
        gulf_3d = sum(gulf_day.get(dn - d, 0) for d in range(1, 4))  # 1,2,3 days ago
        feat.append(np.log1p(gulf_1d + gulf_3d))  # honest: "gulf_tornado_count"

        # =================================================================
        # BLOCK 9: OUTBREAK MOMENTUM (1 feature)
        # FIX: yesterday_count / day_before_count (d_off=1 vs d_off=2)
        # NO same-day data.
        # =================================================================
        yesterday_count = day_counts.get(dn - 1, 0)
        day_before_count = day_counts.get(dn - 2, 0)
        feat.append(yesterday_count / max(day_before_count, 1))

        # =================================================================
        # BLOCK 10: TOPOGRAPHIC PROXIES (7 features)
        # These encode climatology via terrain. Disclosed as such.
        # =================================================================
        feat.append(distance_to_appalachians(center_lat, center_lon) / 500.0)
        feat.append(distance_to_rockies(center_lat, center_lon) / 500.0)
        feat.append(distance_to_ozarks(center_lat, center_lon) / 500.0)
        feat.append(great_plains_indicator(center_lat, center_lon))
        feat.append(mississippi_valley_indicator(center_lat, center_lon))
        feat.append(elevation_proxy(center_lat, center_lon) / 1000.0)
        feat.append(elevation_proxy(center_lat, center_lon) / 1000.0 * cape_clim)

        # =================================================================
        # BLOCK 11: INTERACTION TERMS (3 features)
        # Only using non-leaked features
        # =================================================================
        prop_signal = feat[30]  # spatial propagation (1-cell, 1-day, d_off>=1)
        feat.append(cr * prop_signal)
        feat.append(float(outbreak_len > 0) * prop_signal)
        feat.append(feat[10] * feat[0])  # lat x sin(DOY)

        return feat

    # -------------------------------------------------------------------------
    # Generate samples (with year tracking for temporal bootstrap)
    # -------------------------------------------------------------------------
    rng = np.random.RandomState(42)

    def generate_samples(start_yr, end_yr, max_neg_ratio=5):
        X_list = []
        y_list = []
        yr_list = []  # track year for temporal bootstrap
        nearby_flag = []  # track initiation vs continuation
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
                yr_list.append(yr)
                nearby_flag.append(had_nearby_tornado_past_3d(lb, lnb, dn))
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
                chosen = rng.choice(len(neg_candidates),
                                    size=min(n_neg_wanted, len(neg_candidates)),
                                    replace=False)
                for idx in chosen:
                    dn_neg, yr_neg = neg_candidates[idx]
                    feat = get_features(lb, lnb, dn_neg, mo, yr_neg)
                    X_list.append(feat)
                    y_list.append(0)
                    yr_list.append(yr_neg)
                    nearby_flag.append(had_nearby_tornado_past_3d(lb, lnb, dn_neg))
                    n_neg += 1

        print(f"    Total: {n_pos} positive, {n_neg} negative ({n_pos + n_neg} samples)")
        return (np.array(X_list, dtype=np.float64),
                np.array(y_list, dtype=np.float64),
                np.array(yr_list, dtype=np.int32),
                np.array(nearby_flag, dtype=bool))

    print("  Generating training samples...")
    X_train, y_train, yr_train, nearby_train = generate_samples(train_years[0], train_years[1])
    print("  Generating test samples...")
    X_test, y_test, yr_test, nearby_test = generate_samples(test_years[0], test_years[1])

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    return (X_train, y_train, yr_train, nearby_train,
            X_test, y_test, yr_test, nearby_test)


# =============================================================================
# TEMPORAL CROSS-VALIDATION ON TRAINING DATA
# (Used for ensemble weight selection -- never peeks at test)
# =============================================================================

def temporal_cv_train(X_train, y_train, yr_train):
    """
    4-fold temporal CV on training data.
    Blocks: 2000-2003, 2004-2007, 2008-2011, 2012-2015
    Used to select ensemble weights WITHOUT touching test set.
    """
    print("\n  Temporal CV on training data (for ensemble weight selection)...")
    blocks = [(2000, 2003), (2004, 2007), (2008, 2011), (2012, 2015)]
    fold_aucs = {'logistic': [], 'gbm': [], 'gbt': [], 'bagged': []}

    for fold_i, (val_start, val_end) in enumerate(blocks):
        train_mask = (yr_train < val_start) | (yr_train > val_end)
        val_mask = (yr_train >= val_start) & (yr_train <= val_end)

        if val_mask.sum() < 50 or train_mask.sum() < 100:
            print(f"    Fold {fold_i+1} ({val_start}-{val_end}): skipped (too few samples)")
            continue

        Xt, yt = X_train[train_mask], y_train[train_mask]
        Xv, yv = X_train[val_mask], y_train[val_mask]

        if yv.sum() < 5 or (len(yv) - yv.sum()) < 5:
            print(f"    Fold {fold_i+1}: skipped (too few positives/negatives in val)")
            continue

        # Logistic (lighter for CV: fewer epochs)
        w, b, mu, sd = logistic_train(Xt, yt, lr=0.01, lam=0.5,
                                       epochs=800, class_weight='balanced')
        pv = logistic_predict(Xv, w, b, mu, sd)
        auc_lr = compute_auc(yv, pv)
        fold_aucs['logistic'].append(auc_lr)

        # GBM Stumps (lighter for CV)
        gbm = GradientBoostedStumps(n_stumps=150, learning_rate=0.1, min_samples_leaf=20)
        gbm.fit(Xt, yt)
        pv_gbm = gbm.predict_proba(Xv)
        auc_gbm = compute_auc(yv, pv_gbm)
        fold_aucs['gbm'].append(auc_gbm)

        # GBT depth 3 (lighter for CV)
        gbt = GradientBoostedTrees(n_trees=60, max_depth=3, learning_rate=0.08,
                                    min_samples_leaf=20)
        gbt.fit(Xt, yt)
        pv_gbt = gbt.predict_proba(Xv)
        auc_gbt = compute_auc(yv, pv_gbt)
        fold_aucs['gbt'].append(auc_gbt)

        # Bagged LR (lighter for CV)
        bag = BaggedLogistic(n_bags=5, feature_fraction=0.6, lam=1.0, epochs=400)
        bag.fit(Xt, yt)
        pv_bag = bag.predict_proba(Xv)
        auc_bag = compute_auc(yv, pv_bag)
        fold_aucs['bagged'].append(auc_bag)

        print(f"    Fold {fold_i+1} ({val_start}-{val_end}): "
              f"LR={auc_lr:.4f} GBM={auc_gbm:.4f} GBT={auc_gbt:.4f} Bag={auc_bag:.4f}")

    # Compute mean CV AUC for each model
    cv_means = {}
    for name, aucs in fold_aucs.items():
        cv_means[name] = np.mean(aucs) if aucs else 0.5
        print(f"    Mean CV AUC [{name}]: {cv_means[name]:.4f}")

    return cv_means


# =============================================================================
# ENSEMBLE TRAINING (weights from CV, not test)
# =============================================================================

def train_ensemble_v5(X_train, y_train, X_test, y_test, cv_means):
    """Train 4-model ensemble. Weights chosen by training CV performance."""
    print(f"\n  Training v5 honest ensemble (features={X_train.shape[1]}, "
          f"train={len(y_train)}, test={len(y_test)})")
    print(f"    Train pos rate: {y_train.mean():.3f}, Test pos rate: {y_test.mean():.3f}")

    # --- Model A: L2 Logistic ---
    print("\n  Model A: L2 Logistic Regression...")
    t0 = time_module.time()
    wA, bA, muA, sdA = logistic_train(X_train, y_train, lr=0.01, lam=0.5,
                                        epochs=2000, class_weight='balanced', verbose=True)
    pA_train = logistic_predict(X_train, wA, bA, muA, sdA)
    pA_test = logistic_predict(X_test, wA, bA, muA, sdA)
    aucA = compute_auc(y_test, pA_test)
    print(f"    Model A Test AUC: {aucA:.4f} ({time_module.time()-t0:.1f}s)")

    # --- Model B: GBM Stumps ---
    print("\n  Model B: Gradient Boosted Stumps (400)...")
    t0 = time_module.time()
    gbm = GradientBoostedStumps(n_stumps=400, learning_rate=0.08, min_samples_leaf=20)
    gbm.fit(X_train, y_train, verbose=True)
    pB_train = gbm.predict_proba(X_train)
    pB_test = gbm.predict_proba(X_test)
    aucB = compute_auc(y_test, pB_test)
    print(f"    Model B Test AUC: {aucB:.4f} ({time_module.time()-t0:.1f}s)")

    # --- Model C: GBT depth 3 ---
    print("\n  Model C: Gradient Boosted Trees (150, depth=3)...")
    t0 = time_module.time()
    gbt = GradientBoostedTrees(n_trees=150, max_depth=3, learning_rate=0.05,
                                min_samples_leaf=20)
    gbt.fit(X_train, y_train, verbose=True)
    pC_train = gbt.predict_proba(X_train)
    pC_test = gbt.predict_proba(X_test)
    aucC = compute_auc(y_test, pC_test)
    print(f"    Model C Test AUC: {aucC:.4f} ({time_module.time()-t0:.1f}s)")

    # --- Model D: Bagged Logistic ---
    print("\n  Model D: Bagged Logistic (10 bags, 60% features)...")
    t0 = time_module.time()
    bag = BaggedLogistic(n_bags=10, feature_fraction=0.6, lam=1.0, epochs=800)
    bag.fit(X_train, y_train, verbose=True)
    pD_train = bag.predict_proba(X_train)
    pD_test = bag.predict_proba(X_test)
    aucD = compute_auc(y_test, pD_test)
    print(f"    Model D Test AUC: {aucD:.4f} ({time_module.time()-t0:.1f}s)")

    # --- Ensemble weights from CV (not test!) ---
    # Softmax of CV AUCs as weights
    model_names = ['logistic', 'gbm', 'gbt', 'bagged']
    cv_aucs = np.array([cv_means.get(n, 0.5) for n in model_names])
    # Temperature-scaled softmax
    temp = 0.02  # sharp weighting toward best CV model
    exp_aucs = np.exp((cv_aucs - cv_aucs.max()) / temp)
    weights = exp_aucs / exp_aucs.sum()
    print(f"\n  Ensemble weights (from training CV, NOT test):")
    for n, w_val in zip(model_names, weights):
        print(f"    {n}: {w_val:.4f}")

    p_test_all = np.column_stack([pA_test, pB_test, pC_test, pD_test])
    p_train_all = np.column_stack([pA_train, pB_train, pC_train, pD_train])
    p_ensemble_test = p_test_all @ weights
    p_ensemble_train = p_train_all @ weights

    auc_ensemble = compute_auc(y_test, p_ensemble_test)

    clim_prob = y_test.mean()
    bss = compute_bss(y_test, p_ensemble_test, clim_prob)

    print(f"\n  ENSEMBLE RESULTS:")
    print(f"  {'='*50}")
    print(f"    Model A (L2 Logistic)       AUC: {aucA:.4f}")
    print(f"    Model B (GBM Stumps 400)    AUC: {aucB:.4f}")
    print(f"    Model C (GBT depth=3 150)   AUC: {aucC:.4f}")
    print(f"    Model D (Bagged LR 10x60%)  AUC: {aucD:.4f}")
    print(f"    Ensemble (CV-weighted)      AUC: {auc_ensemble:.4f}")
    print(f"    BSS vs climatology:              {bss:+.4f}")

    return {
        'auc_A': aucA, 'auc_B': aucB, 'auc_C': aucC, 'auc_D': aucD,
        'auc_ensemble': auc_ensemble,
        'bss': bss,
        'p_test': p_ensemble_test,
        'p_train': p_ensemble_train,
        'y_test': y_test,
        'weights': weights,
        'w': wA, 'b': bA, 'mu': muA, 'sd': sdA,
    }


# =============================================================================
# INITIATION vs CONTINUATION BREAKDOWN
# =============================================================================

def initiation_continuation_breakdown(y_test, p_test, nearby_test, yr_test):
    """
    Evaluate formation AUC separately for:
    - INITIATION: no nearby tornado in past 3 days
    - CONTINUATION: nearby tornado occurred in past 3 days
    """
    print("\n" + "=" * 70)
    print("INITIATION vs CONTINUATION BREAKDOWN")
    print("=" * 70)

    init_mask = ~nearby_test
    cont_mask = nearby_test

    n_init = init_mask.sum()
    n_cont = cont_mask.sum()
    print(f"  Initiation samples: {n_init} ({100*n_init/len(y_test):.1f}%)")
    print(f"  Continuation samples: {n_cont} ({100*n_cont/len(y_test):.1f}%)")

    results = {}

    for label, mask in [("INITIATION", init_mask), ("CONTINUATION", cont_mask)]:
        y_sub = y_test[mask]
        p_sub = p_test[mask]
        yr_sub = yr_test[mask]
        n_pos = y_sub.sum()
        n_neg = len(y_sub) - n_pos
        print(f"\n  {label}:")
        print(f"    N={len(y_sub)}, pos={n_pos:.0f}, neg={n_neg:.0f}")

        if n_pos < 5 or n_neg < 5:
            print(f"    SKIPPED: too few samples for reliable AUC")
            results[label.lower()] = {'auc': 0.5, 'ci_lo': 0.0, 'ci_hi': 1.0, 'n': len(y_sub)}
            continue

        auc = compute_auc(y_sub, p_sub)
        lo, hi, mean_auc = bootstrap_auc_by_year(y_sub, p_sub, yr_sub, n_boot=500)
        print(f"    AUC = {auc:.4f} (bootstrap 95% CI: [{lo:.4f}, {hi:.4f}])")

        if label == "INITIATION":
            print(f"    NOTE: This is the harder problem. The model must predict tornado")
            print(f"    formation with NO recent nearby tornado activity as signal.")

        results[label.lower()] = {'auc': auc, 'ci_lo': lo, 'ci_hi': hi, 'n': len(y_sub)}

    return results


# =============================================================================
# FAIR CLIMATOLOGY BASELINE
# =============================================================================

def fair_climatology_baseline(y_test, p_test, yr_test):
    """
    Fair comparison: compute climatology AUC on the SAME test set.
    No hardcoded SPC numbers.
    """
    print("\n" + "=" * 70)
    print("FAIR BASELINE COMPARISON")
    print("  Direct comparison to SPC Day 1 outlooks requires evaluating")
    print("  both systems on the same grid and time period, which has")
    print("  not been done. Instead we compare to a climatology baseline")
    print("  evaluated on the exact same test set.")
    print("=" * 70)

    clim_prob = y_test.mean()
    our_auc = compute_auc(y_test, p_test)
    our_bss = compute_bss(y_test, p_test, clim_prob)
    our_brier = compute_brier(y_test, p_test)
    clim_brier = compute_brier(y_test, np.full(len(y_test), clim_prob))

    # Climatology AUC is 0.5 by definition (constant prediction)
    print(f"\n    Climatology (constant P={clim_prob:.4f}):")
    print(f"      AUC: 0.5000 (by definition)")
    print(f"      Brier Score: {clim_brier:.4f}")
    print(f"\n    Our v5 honest model:")
    print(f"      AUC: {our_auc:.4f}")
    print(f"      Brier Score: {our_brier:.4f}")
    print(f"      BSS vs climatology: {our_bss:+.4f}")
    print(f"\n    Improvement over climatology:")
    print(f"      AUC:   +{our_auc - 0.5:.4f}")
    print(f"      BSS:   {our_bss:+.4f} (>0 means better than climatology)")
    print(f"      Brier: {clim_brier - our_brier:+.4f} (>0 means we're better)")

    return {
        'our_auc': our_auc,
        'our_bss': our_bss,
        'our_brier': our_brier,
        'clim_prob': clim_prob,
        'clim_brier': clim_brier,
    }


# =============================================================================
# TEMPORAL CV ON TEST SET (for reporting stability, not for model selection)
# =============================================================================

def temporal_cv_test(y_test, p_test, yr_test):
    """Report AUC stability across temporal blocks of test set."""
    print("\n" + "=" * 70)
    print("TEMPORAL STABILITY (test set)")
    print("=" * 70)

    unique_years = sorted(np.unique(yr_test))
    print(f"  Test years: {unique_years}")

    block_aucs = []
    for yr in unique_years:
        mask = yr_test == yr
        n = mask.sum()
        n_pos = y_test[mask].sum()
        if n_pos < 3 or (n - n_pos) < 3:
            print(f"    {yr}: n={n}, pos={n_pos:.0f} -- skipped (too few)")
            continue
        auc = compute_auc(y_test[mask], p_test[mask])
        block_aucs.append(auc)
        print(f"    {yr}: AUC={auc:.4f} (n={n}, pos={n_pos:.0f})")

    if block_aucs:
        print(f"  Mean: {np.mean(block_aucs):.4f}, Std: {np.std(block_aucs):.4f}")

    # Bootstrap CI by year
    lo, hi, mean_auc = bootstrap_auc_by_year(y_test, p_test, yr_test, n_boot=1000)
    print(f"  Bootstrap (by year) 95% CI: [{lo:.4f}, {hi:.4f}]")

    return {
        'block_aucs': block_aucs,
        'years': unique_years,
        'bootstrap_lo': lo,
        'bootstrap_hi': hi,
        'bootstrap_mean': mean_auc,
    }


# =============================================================================
# SEVERITY MODEL (v5 HONEST)
# =============================================================================

def build_severity_model(tornadoes, indices):
    """
    HONEST SEVERITY MODEL

    CRITICAL CAVEATS:
    - Width (damage survey width) is determined AFTER the tornado by damage
      survey teams. It is NOT available in real-time.
    - If claiming real-time operational use, you would need Doppler radar-
      estimated rotational velocity width, which is a different quantity.
    - This model is best understood as POST-EVENT NOWCASTING: given that a
      tornado has occurred and its width has been surveyed, predict EF rating.
    - Fatalities, injuries, and property loss are OUTCOMES and must NEVER be
      used as features.
    - Path length and direction are POST-EVENT only.
    """
    print("\n" + "=" * 70)
    print("PART 2: TORNADO SEVERITY PREDICTION v5 (HONEST)")
    print("=" * 70)
    print("  CAVEAT: Width from damage surveys is NOT available in real-time.")
    print("  This is POST-EVENT NOWCASTING, not real-time prediction.")
    print("  Fatalities/injuries/path length NOT used as features.")

    cd2 = indices[2.0]['cell_day']

    valid = [t for t in tornadoes if t['yr'] >= 2000 and t['width_yd'] > 0]
    print(f"  Valid tornadoes (with width): {len(valid)}")

    # Day counts (no post-event data)
    day_key_counts = defaultdict(int)
    for t in valid:
        key = (t['yr'], t['mo'], t['dy'])
        day_key_counts[key] += 1

    def severity_features(t):
        """
        HONEST severity features:
        - Width (damage survey -- post-event, disclosed)
        - Season, location (non-leaky)
        - Outbreak context: count of tornadoes on same day (from reports)
        - Time of day
        - Climatological proxies
        - Topographic context

        NOT USED: fatalities, injuries, path length, path direction, property loss
        """
        feat = []

        # 1. Width (THE key feature -- post-event damage survey)
        log_w = np.log1p(t['width_yd'])
        feat.append(log_w)                                # 0
        feat.append(log_w ** 2)                           # 1
        feat.append(log_w ** 3)                           # 2

        # 2. Season
        mo = t['mo']
        doy = (mo - 1) * 30.4 + 15
        feat.append(math.sin(2 * math.pi * doy / 365.25))  # 3
        feat.append(math.cos(2 * math.pi * doy / 365.25))  # 4

        # 3. Location
        feat.append(t['slat'])                            # 5
        feat.append(t['slon'])                            # 6

        # 4. Outbreak context (count only -- from preliminary reports)
        key = (t['yr'], t['mo'], t['dy'])
        dc = day_key_counts[key]
        feat.append(dc)                                   # 7
        feat.append(np.log1p(dc))                         # 8

        # 5. Width * outbreak interaction
        feat.append(log_w * np.log1p(dc))                 # 9

        # 6. Time-of-day features
        if t['hour'] >= 0:
            h = t['hour']
            feat.append(math.sin(2 * math.pi * h / 24))  # 10
            feat.append(math.cos(2 * math.pi * h / 24))  # 11
            feat.append(1.0 if 15 <= h <= 20 else 0.0)   # 12
            feat.append(1.0 if h < 6 or h > 22 else 0.0) # 13
        else:
            feat.extend([0.0, 0.0, 0.5, 0.0])

        # 7. Climatological proxies (no tornado outcome data)
        cape_clim = lat_season_cape_climatology(t['slat'], mo, doy)
        feat.append(cape_clim)                              # 14
        feat.append(cape_clim * log_w)                      # 15

        feat.append(lat_season_lcl_climatology(doy, t['slat']))  # 16

        # 8. Topographic
        feat.append(great_plains_indicator(t['slat'], t['slon']))  # 17
        feat.append(mississippi_valley_indicator(t['slat'], t['slon']))  # 18
        feat.append(elevation_proxy(t['slat'], t['slon']) / 1000.0)  # 19

        # 9. Width * location interactions
        feat.append(log_w * t['slat'] / 50.0)            # 20
        feat.append(log_w * math.sin(2 * math.pi * doy / 365.25))  # 21

        # 10. Nearby tornado COUNT context (d_off >= 1 only)
        dn = date_to_day_number(t['yr'], t['mo'], t['dy'])
        lb, lnb = latlon_to_cell(t['slat'], t['slon'], 2.0)
        n_lat = int((50 - 25) / 2.0)
        n_lon = int(60 / 2.0)
        nearby_count_yesterday = 0
        for dlb in range(-1, 2):
            for dlnb in range(-1, 2):
                nlb, nlnb = lb + dlb, lnb + dlnb
                if 0 <= nlb < n_lat and 0 <= nlnb < n_lon:
                    nearby_count_yesterday += len(cd2.get((nlb, nlnb, dn - 1), []))
        feat.append(nearby_count_yesterday)                # 22

        return feat

    # Split: temporal
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

        # AUDIT FIX #11: Report EF4+ sample size explicitly
        if label == 'EF4+':
            print(f"    *** EF4+ TEST SAMPLE SIZE: {n_pos_te:.0f} positive out of {len(y_te)} ***")
            if n_pos_te < 50:
                print(f"    *** WARNING: Only {n_pos_te:.0f} EF4+ tornadoes in test set.")
                print(f"    *** AUC is UNRELIABLE with this sample size.")
                print(f"    *** Confidence intervals will be very wide. ***")

        if n_pos_tr < 5 or n_pos_te < 3:
            print(f"    Skipping {label}: too few positive samples")
            continue

        # 2-model ensemble: logistic + GBM
        lam = 0.3 if threshold <= 2 else 0.05
        w, b_val, mu, sd = logistic_train(X_tr, y_tr, lr=0.05, lam=lam,
                                           epochs=1500, class_weight='balanced')
        pA_te = logistic_predict(X_te, w, b_val, mu, sd)

        n_stumps = 200 if threshold <= 3 else 100
        gbm_sev = GradientBoostedStumps(n_stumps=n_stumps, learning_rate=0.05, min_samples_leaf=10)
        gbm_sev.fit(X_tr, y_tr)
        pB_te = gbm_sev.predict_proba(X_te)

        # Simple average (no test-peeking for weights)
        p_ensemble = 0.5 * pA_te + 0.5 * pB_te
        auc_test = compute_auc(y_te, p_ensemble)

        # Bootstrap CI by year
        test_years_sev = np.array([t['yr'] for t in test_t])
        lo, hi, mean_auc = bootstrap_auc_by_year(y_te, p_ensemble, test_years_sev, n_boot=500)

        print(f"    {label} AUC: {auc_test:.4f} (bootstrap 95% CI: [{lo:.4f}, {hi:.4f}])")

        results[label] = {
            'auc_test': auc_test,
            'ci_lo': lo, 'ci_hi': hi,
            'n_pos_test': int(n_pos_te),
            'n_test': len(y_te),
            'y_te': y_te,
            'p_te': p_ensemble,
        }

    return results


# =============================================================================
# FIGURE CREATION
# =============================================================================

def create_figure(formation_results, severity_results, cv_results,
                  baseline_results, init_cont_results):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Tornado Prediction Model v5 -- HONEST EVALUATION\n'
                 'No Same-Day Data | No Post-Event Features | Fair Baselines',
                 fontsize=14, fontweight='bold', y=0.98)

    # --- Panel 1: Formation ROC ---
    ax1 = axes[0, 0]
    fpr, tpr = compute_roc_curve(formation_results['y_test'], formation_results['p_test'])
    auc_val = formation_results['auc_ensemble']
    ax1.plot(fpr, tpr, 'b-', linewidth=2,
             label=f'v5 Ensemble AUC={auc_val:.4f}')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random (AUC=0.5)')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Formation Prediction ROC\n(d_off >= 1 only, no post-event features)')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Initiation vs Continuation ---
    ax2 = axes[0, 1]
    labels = []
    aucs = []
    colors = []
    if 'initiation' in init_cont_results:
        ir = init_cont_results['initiation']
        labels.append(f"Initiation\n(n={ir['n']})")
        aucs.append(ir['auc'])
        colors.append('#e74c3c')
    if 'continuation' in init_cont_results:
        cr = init_cont_results['continuation']
        labels.append(f"Continuation\n(n={cr['n']})")
        aucs.append(cr['auc'])
        colors.append('#2ecc71')
    labels.append(f"Overall\n(n={len(formation_results['y_test'])})")
    aucs.append(formation_results['auc_ensemble'])
    colors.append('#3498db')

    bars = ax2.bar(range(len(labels)), aucs, color=colors, alpha=0.8, edgecolor='black')
    for bar, auc_v in zip(bars, aucs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{auc_v:.3f}', ha='center', fontsize=10, fontweight='bold')
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, fontsize=9)
    ax2.set_ylabel('AUC')
    ax2.set_title('Initiation vs Continuation\n(no nearby tornado in past 3d vs. has nearby)')
    ax2.set_ylim(0.5, 1.0)
    ax2.grid(True, alpha=0.3, axis='y')

    # --- Panel 3: Temporal Stability ---
    ax3 = axes[0, 2]
    if cv_results['block_aucs']:
        years_labels = [str(y) for y in cv_results['years']
                       if True][:len(cv_results['block_aucs'])]
        ax3.bar(range(len(cv_results['block_aucs'])), cv_results['block_aucs'],
                color='#9b59b6', alpha=0.7, edgecolor='black')
        ax3.set_xticks(range(len(years_labels)))
        ax3.set_xticklabels(years_labels, rotation=45, fontsize=8)
        ax3.axhline(y=np.mean(cv_results['block_aucs']), color='red',
                    linestyle='--', label=f"Mean={np.mean(cv_results['block_aucs']):.3f}")
        ax3.axhspan(cv_results['bootstrap_lo'], cv_results['bootstrap_hi'],
                    alpha=0.15, color='red', label=f"95% CI [{cv_results['bootstrap_lo']:.3f}, {cv_results['bootstrap_hi']:.3f}]")
    ax3.set_ylabel('AUC')
    ax3.set_title('Temporal Stability by Year\n(bootstrap CI by year resampling)')
    ax3.set_ylim(0.5, 1.0)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')

    # --- Panel 4: Fair Baseline ---
    ax4 = axes[1, 0]
    baseline_labels = ['Climatology\n(constant P)', 'v5 Model']
    baseline_aucs = [0.5, baseline_results['our_auc']]
    baseline_bss = [0.0, baseline_results['our_bss']]
    x4 = np.arange(len(baseline_labels))
    w4 = 0.35
    ax4.bar(x4 - w4/2, baseline_aucs, w4, label='AUC', color='#3498db', alpha=0.7)
    ax4.bar(x4 + w4/2, [max(0, b) for b in baseline_bss], w4,
            label='BSS', color='#e67e22', alpha=0.7)
    for i, (a, b) in enumerate(zip(baseline_aucs, baseline_bss)):
        ax4.text(i - w4/2, a + 0.01, f'{a:.3f}', ha='center', fontsize=9)
        ax4.text(i + w4/2, max(0, b) + 0.01, f'{b:.3f}', ha='center', fontsize=9)
    ax4.set_xticks(x4)
    ax4.set_xticklabels(baseline_labels)
    ax4.set_title('Fair Baseline Comparison\n(measured on same test set, not hardcoded)')
    ax4.legend(fontsize=9)
    ax4.set_ylim(0, 1.0)
    ax4.grid(True, alpha=0.3, axis='y')

    # --- Panel 5: Severity AUC ---
    ax5 = axes[1, 1]
    sev_labels = []
    sev_aucs = []
    sev_ci_lo = []
    sev_ci_hi = []
    sev_colors = []
    color_map = {'EF2+': '#f39c12', 'EF3+': '#e74c3c', 'EF4+': '#8e44ad'}
    for label in ['EF2+', 'EF3+', 'EF4+']:
        if label in severity_results:
            sr = severity_results[label]
            sev_labels.append(f"{label}\n(n={sr['n_pos_test']})")
            sev_aucs.append(sr['auc_test'])
            sev_ci_lo.append(sr['ci_lo'])
            sev_ci_hi.append(sr['ci_hi'])
            sev_colors.append(color_map.get(label, 'gray'))

    if sev_labels:
        bars5 = ax5.bar(range(len(sev_labels)), sev_aucs, color=sev_colors,
                        alpha=0.8, edgecolor='black')
        # Error bars
        for i, (lo, hi, auc_v) in enumerate(zip(sev_ci_lo, sev_ci_hi, sev_aucs)):
            ax5.plot([i, i], [lo, hi], 'k-', linewidth=2)
            ax5.plot([i-0.1, i+0.1], [lo, lo], 'k-', linewidth=1.5)
            ax5.plot([i-0.1, i+0.1], [hi, hi], 'k-', linewidth=1.5)
            ax5.text(i, auc_v + 0.02, f'{auc_v:.3f}', ha='center', fontsize=10, fontweight='bold')
        ax5.set_xticks(range(len(sev_labels)))
        ax5.set_xticklabels(sev_labels, fontsize=9)
    ax5.set_ylabel('AUC')
    ax5.set_title('Severity (POST-EVENT nowcasting)\nWidth from damage survey, NOT real-time')
    ax5.set_ylim(0.5, 1.05)
    ax5.grid(True, alpha=0.3, axis='y')

    # --- Panel 6: Summary text ---
    ax6 = axes[1, 2]
    ax6.axis('off')
    summary_lines = [
        "HONEST EVALUATION SUMMARY",
        "=" * 40,
        "",
        f"Formation AUC: {formation_results['auc_ensemble']:.4f}",
        f"  Bootstrap 95% CI: [{cv_results['bootstrap_lo']:.4f}, {cv_results['bootstrap_hi']:.4f}]",
        f"  BSS vs climatology: {formation_results['bss']:+.4f}",
        "",
    ]
    if 'initiation' in init_cont_results:
        summary_lines.append(f"  Initiation AUC:    {init_cont_results['initiation']['auc']:.4f}")
    if 'continuation' in init_cont_results:
        summary_lines.append(f"  Continuation AUC:  {init_cont_results['continuation']['auc']:.4f}")
    summary_lines.extend([
        "",
        "Severity (POST-EVENT nowcasting):",
    ])
    for label in ['EF2+', 'EF3+', 'EF4+']:
        if label in severity_results:
            sr = severity_results[label]
            reliability = ""
            if label == 'EF4+' and sr['n_pos_test'] < 50:
                reliability = " [UNRELIABLE: small N]"
            summary_lines.append(
                f"  {label}: AUC={sr['auc_test']:.3f} "
                f"[{sr['ci_lo']:.3f},{sr['ci_hi']:.3f}] "
                f"(n={sr['n_pos_test']}){reliability}")
    summary_lines.extend([
        "",
        "AUDIT FIXES APPLIED:",
        "  [x] Zero same-day data (d_off >= 1)",
        "  [x] No post-event formation features",
        "  [x] Honest proxy labels (no fake ERA5)",
        "  [x] Fair climatology baseline (measured)",
        "  [x] Temporal bootstrap (by year)",
        "  [x] Ensemble weights from training CV",
        "  [x] Initiation/continuation breakdown",
        "  [x] Width = damage survey (disclosed)",
    ])
    ax6.text(0.05, 0.95, "\n".join(summary_lines), transform=ax6.transAxes,
             fontsize=8.5, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = os.path.join(FIG_DIR, 'tornado_model_v5_honest.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n  Figure saved: {out_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("HONEST EVALUATION -- No Same-Day Data, No Post-Event Features")
    print("TORNADO PREDICTION MODEL v5")
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
    (X_train, y_train, yr_train, nearby_train,
     X_test, y_test, yr_test, nearby_test) = build_formation_features(
        tornadoes, indices, train_years, test_years)

    print(f"\n  Feature dimensionality: {X_train.shape[1]} features")

    # Temporal CV on training data (for ensemble weight selection)
    cv_means = temporal_cv_train(X_train, y_train, yr_train)

    # Train ensemble (weights from CV, not test)
    formation_results = train_ensemble_v5(X_train, y_train, X_test, y_test, cv_means)

    # Initiation vs continuation breakdown
    init_cont_results = initiation_continuation_breakdown(
        y_test, formation_results['p_test'], nearby_test, yr_test)

    # Fair climatology baseline
    baseline_results = fair_climatology_baseline(
        y_test, formation_results['p_test'], yr_test)

    # Temporal stability on test set
    cv_results = temporal_cv_test(y_test, formation_results['p_test'], yr_test)

    # Severity prediction
    severity_results = build_severity_model(tornadoes, indices)

    # Create figure
    create_figure(formation_results, severity_results, cv_results,
                  baseline_results, init_cont_results)

    # Final summary
    elapsed = time_module.time() - t_start
    print("\n" + "=" * 70)
    print("FINAL RESULTS: v5 HONEST EVALUATION")
    print("=" * 70)

    print(f"\n  FORMATION PREDICTION (day-ahead, no same-day data):")
    print(f"    Ensemble AUC: {formation_results['auc_ensemble']:.4f}")
    print(f"    Bootstrap 95% CI (by year): [{cv_results['bootstrap_lo']:.4f}, {cv_results['bootstrap_hi']:.4f}]")
    print(f"    BSS vs climatology: {formation_results['bss']:+.4f}")
    print(f"    Individual models:")
    print(f"      L2 Logistic:  {formation_results['auc_A']:.4f}")
    print(f"      GBM Stumps:   {formation_results['auc_B']:.4f}")
    print(f"      GBT depth=3:  {formation_results['auc_C']:.4f}")
    print(f"      Bagged LR:    {formation_results['auc_D']:.4f}")

    print(f"\n  INITIATION vs CONTINUATION:")
    if 'initiation' in init_cont_results:
        ir = init_cont_results['initiation']
        print(f"    Initiation AUC:    {ir['auc']:.4f} [{ir['ci_lo']:.4f}, {ir['ci_hi']:.4f}]")
        print(f"    (Predicting tornado with NO recent nearby activity)")
    if 'continuation' in init_cont_results:
        cr_res = init_cont_results['continuation']
        print(f"    Continuation AUC:  {cr_res['auc']:.4f} [{cr_res['ci_lo']:.4f}, {cr_res['ci_hi']:.4f}]")
        print(f"    (Predicting tornado WITH recent nearby activity)")

    print(f"\n  SEVERITY (POST-EVENT NOWCASTING):")
    print(f"    CAVEAT: Width is from post-event damage surveys.")
    for label in ['EF2+', 'EF3+', 'EF4+']:
        if label in severity_results:
            sr = severity_results[label]
            reliability = ""
            if label == 'EF4+' and sr['n_pos_test'] < 50:
                reliability = " *** UNRELIABLE (n<50, wide CI) ***"
            print(f"    {label}: AUC={sr['auc_test']:.4f} "
                  f"[{sr['ci_lo']:.4f}, {sr['ci_hi']:.4f}] "
                  f"(n_pos_test={sr['n_pos_test']}){reliability}")

    print(f"\n  BASELINE COMPARISON:")
    print(f"    Climatology AUC: 0.5000 (by definition)")
    print(f"    Our AUC:         {baseline_results['our_auc']:.4f}")
    print(f"    Our BSS:         {baseline_results['our_bss']:+.4f}")
    print(f"    NOTE: Direct comparison to SPC Day 1 outlooks has NOT been done.")
    print(f"          Both systems would need to be evaluated on the same grid")
    print(f"          and time period for a fair comparison.")

    print(f"\n  FEATURES: {X_train.shape[1]} total")
    print(f"    Removed: all d_off=0 features, EF/width/path/fatality features")
    print(f"    Removed: bulk_shear_proxy_from_widths (circular)")
    print(f"    Removed: srh_proxy (circular)")
    print(f"    Removed: outbreak_momentum with today's count")
    print(f"    Kept: tornado COUNTS (d_off>=1), locations, season, climatology, topography")

    print(f"\n  Total runtime: {elapsed:.1f}s")
    print("  Done.")


if __name__ == '__main__':
    main()
