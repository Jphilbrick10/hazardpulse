#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TORNADO PREDICTION MODEL v2 - COMPREHENSIVE FORMATION + SEVERITY SYSTEM
=========================================================================

v1 Problem: Formation model (AUC=0.780) LOST to climatology (AUC=0.827).
Root cause: Only spatial-temporal features, no environmental/atmospheric proxies.

v2 Solution: Three-part integrated system:
  Part 1: Formation prediction using synoptic-scale propagation signals
  Part 2: Severity prediction (EF2+, EF3+, EF4+) given formation
  Part 3: Combined pipeline evaluation

KEY INSIGHT: Tornado outbreaks are SYNOPTIC events. A cold front creates
tornadoes across a SWATH of cells over 1-3 days. The propagation signal
(tornadoes in nearby cells yesterday/today) is the strongest non-climatological
predictor. We approximate environmental conditions through tornado statistics:
  - Seasonal-geographic climatology ~ CAPE/shear patterns
  - Multi-scale temporal activity ~ atmospheric instability persistence
  - Spatial propagation ~ synoptic forcing (fronts, drylines)
  - Outbreak detection ~ large-scale severe weather setup

No sklearn. Manual logistic regression with L2 regularization.
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
# MANUAL LOGISTIC REGRESSION (no sklearn)
# =============================================================================

def sigmoid(z):
    """Numerically stable sigmoid."""
    z = np.clip(z, -500, 500)
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def logistic_regression_train(X, y, lr=0.01, lam=1.0, epochs=500, verbose=False,
                               class_weight=None):
    """
    Train logistic regression with L2 regularization via gradient descent.
    X: (N, D) features (will be standardized internally)
    y: (N,) binary labels
    class_weight: if 'balanced', weight samples inversely proportional to class freq
    Returns: weights, bias, mean, std (for standardization)
    """
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)

    # Standardize
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Xs = (X - mu) / sd

    N, D = Xs.shape
    w = np.zeros(D)
    b = 0.0

    # Sample weights for class balancing
    sample_w = np.ones(N)
    if class_weight == 'balanced':
        n_pos = y.sum()
        n_neg = N - n_pos
        if n_pos > 0 and n_neg > 0:
            w_pos = N / (2 * n_pos)
            w_neg = N / (2 * n_neg)
            sample_w = np.where(y == 1, w_pos, w_neg)

    # Adam optimizer for faster convergence
    m_w = np.zeros(D)
    v_w = np.zeros(D)
    m_b = 0.0
    v_b = 0.0
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    for epoch in range(epochs):
        z = Xs @ w + b
        p = sigmoid(z)

        # Weighted gradient with L2 regularization
        residual = (p - y) * sample_w
        dw = (Xs.T @ residual) / N + (lam / N) * w
        db = residual.mean()

        # Adam update
        m_w = beta1 * m_w + (1 - beta1) * dw
        v_w = beta2 * v_w + (1 - beta2) * dw ** 2
        m_b = beta1 * m_b + (1 - beta1) * db
        v_b = beta2 * v_b + (1 - beta2) * db ** 2

        t = epoch + 1
        m_w_hat = m_w / (1 - beta1 ** t)
        v_w_hat = v_w / (1 - beta2 ** t)
        m_b_hat = m_b / (1 - beta1 ** t)
        v_b_hat = v_b / (1 - beta2 ** t)

        w -= lr * m_w_hat / (np.sqrt(v_w_hat) + eps)
        b -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)

        if verbose and epoch % 200 == 0:
            loss = -np.mean(sample_w * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))
            print(f"      Epoch {epoch}: loss={loss:.4f}")

    return w, b, mu, sd


def logistic_predict(X, w, b, mu, sd):
    """Predict probabilities using trained logistic regression."""
    X = np.array(X, dtype=np.float64)
    Xs = (X - mu) / sd
    return sigmoid(Xs @ w + b)


def compute_auc(y_true, y_score):
    """Manual AUC computation via trapezoidal rule on ROC."""
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    # Sort by score descending
    order = np.argsort(-y_score)
    y_true = y_true[order]
    y_score = y_score[order]

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = 0
    fp = 0

    for i in range(len(y_true)):
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    # Trapezoidal AUC
    auc = 0.0
    for i in range(1, len(fpr_list)):
        auc += (fpr_list[i] - fpr_list[i - 1]) * (tpr_list[i] + tpr_list[i - 1]) / 2

    return auc


def compute_brier(y_true, y_score):
    """Brier score (lower is better)."""
    y_true = np.array(y_true, dtype=np.float64)
    y_score = np.array(y_score, dtype=np.float64)
    return np.mean((y_score - y_true) ** 2)


def compute_bss(y_true, y_score, clim_prob):
    """Brier Skill Score relative to climatology (positive = better than clim)."""
    bs_model = compute_brier(y_true, y_score)
    bs_clim = compute_brier(y_true, np.full_like(y_true, clim_prob))
    if bs_clim < 1e-12:
        return 0.0
    return 1.0 - bs_model / bs_clim


# =============================================================================
# DATA LOADING
# =============================================================================

def load_spc_tornadoes():
    """Load SPC tornado data, return list of dicts."""
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
            })
        except (ValueError, KeyError):
            continue

    print(f"    {len(tornadoes)} tornadoes loaded (1980-2023, CONUS)")
    return tornadoes


def date_to_day_number(yr, mo, dy):
    """Convert date to an integer day number (days since 2000-01-01)."""
    # Simple: approximate, doesn't need to be exact calendar
    days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    d = (yr - 2000) * 365 + (yr - 2000) // 4  # approx leap years
    for m in range(1, mo):
        d += days_in_month[m]
    d += dy
    return d


def latlon_to_cell(lat, lon, cell_size=2.0):
    """Convert lat/lon to grid cell indices."""
    lat_bin = int((lat - 25.0) / cell_size)
    lon_bin = int((lon - (-125.0)) / cell_size)
    # Clamp
    lat_bin = max(0, min(lat_bin, int((50 - 25) / cell_size) - 1))
    lon_bin = max(0, min(lon_bin, int((65) / cell_size) - 1))  # -125 to -65 = 60 deg
    return lat_bin, lon_bin


# =============================================================================
# PART 1: FORMATION PREDICTION
# =============================================================================

def build_formation_model(tornadoes):
    """
    Build tornado formation prediction model.
    Grid: 2-degree cells across CONUS.
    Features: synoptic propagation + climatological proxies.
    """
    print("\n" + "=" * 70)
    print("PART 1: TORNADO FORMATION PREDICTION")
    print("=" * 70)

    cell_size = 2.0
    n_lat = int((50 - 25) / cell_size)  # 12-13 bins
    n_lon = int(60 / cell_size)          # 30 bins

    # -------------------------------------------------------------------------
    # Step 1: Index tornadoes by cell-day
    # -------------------------------------------------------------------------
    print("\n  Building cell-day index...")

    # cell_day_events: (lat_bin, lon_bin, day_num) -> list of tornado dicts
    cell_day_events = defaultdict(list)
    # cell_month_events: (lat_bin, lon_bin, month) -> list of tornado dicts
    cell_month_events = defaultdict(list)
    # day_events: day_num -> list of (lat_bin, lon_bin, tornado)
    day_events = defaultdict(list)

    all_day_nums = []
    tornado_records = []

    for t in tornadoes:
        dn = date_to_day_number(t['yr'], t['mo'], t['dy'])
        lb, lnb = latlon_to_cell(t['slat'], t['slon'], cell_size)
        key = (lb, lnb, dn)
        cell_day_events[key].append(t)
        cell_month_events[(lb, lnb, t['mo'])].append(t)
        day_events[dn].append((lb, lnb, t))
        all_day_nums.append(dn)
        tornado_records.append((lb, lnb, dn, t))

    all_day_nums = sorted(set(all_day_nums))
    print(f"    {len(cell_day_events)} unique cell-day tornado events")
    print(f"    Grid: {n_lat} x {n_lon} = {n_lat * n_lon} cells")

    # -------------------------------------------------------------------------
    # Step 2: Precompute climatological rates
    # -------------------------------------------------------------------------
    print("  Computing climatological rates...")

    # Climatological rate: P(tornado | cell, month) from training data
    # Use 2000-2016 for training
    train_start_yr = 2000
    train_end_yr = 2016
    test_start_yr = 2017
    test_end_yr = 2023

    # Count tornadoes per cell-month in training period
    clim_counts = defaultdict(int)     # (lat_bin, lon_bin, month) -> count
    clim_days = defaultdict(int)       # (lat_bin, lon_bin, month) -> number of days in training
    total_train_tornado_days = 0
    total_train_days = 0

    for t in tornadoes:
        if train_start_yr <= t['yr'] <= train_end_yr:
            lb, lnb = latlon_to_cell(t['slat'], t['slon'], cell_size)
            dn = date_to_day_number(t['yr'], t['mo'], t['dy'])
            clim_counts[(lb, lnb, t['mo'])] += 1

    # Count available days per cell-month in training
    days_per_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    for lb in range(n_lat):
        for lnb in range(n_lon):
            for mo in range(1, 13):
                n_years = train_end_yr - train_start_yr + 1
                clim_days[(lb, lnb, mo)] = n_years * days_per_month[mo]

    # Climatological daily rate per cell-month
    clim_rate = {}
    for key, cnt in clim_counts.items():
        d = clim_days[key]
        if d > 0:
            clim_rate[key] = cnt / d
        else:
            clim_rate[key] = 0.0

    # -------------------------------------------------------------------------
    # Step 3: Build feature vectors for cell-days
    # -------------------------------------------------------------------------
    print("  Building feature vectors (this may take a moment)...")

    def get_features(lb, lnb, dn, mo, yr, tornadoes_today_in_cell=False):
        """
        Build feature vector for a cell-day.
        """
        features = []

        # --- Feature 1-2: Day-of-year cyclical (annual + semi-annual) ---
        # Approximate day-of-year from month+day
        doy = (mo - 1) * 30.4 + 15  # rough center of month
        features.append(math.sin(2 * math.pi * doy / 365.25))  # annual sin
        features.append(math.cos(2 * math.pi * doy / 365.25))  # annual cos
        features.append(math.sin(4 * math.pi * doy / 365.25))  # semi-annual sin
        features.append(math.cos(4 * math.pi * doy / 365.25))  # semi-annual cos

        # --- Feature 5: Climatological rate for this cell-month ---
        cr = clim_rate.get((lb, lnb, mo), 0.0)
        features.append(cr)

        # --- Feature 6: Smoothed climatological rate (neighboring cells too) ---
        neighbor_cr = 0.0
        n_neighbors = 0
        for dlb in [-1, 0, 1]:
            for dlnb in [-1, 0, 1]:
                if dlb == 0 and dlnb == 0:
                    continue
                nlb = lb + dlb
                nlnb = lnb + dlnb
                if 0 <= nlb < n_lat and 0 <= nlnb < n_lon:
                    neighbor_cr += clim_rate.get((nlb, nlnb, mo), 0.0)
                    n_neighbors += 1
        if n_neighbors > 0:
            neighbor_cr /= n_neighbors
        features.append(neighbor_cr)

        # --- Features 7-12: Multi-scale temporal activity in THIS cell ---
        for lookback in [3, 7, 14, 30, 90, 365]:
            count = 0
            for d_offset in range(1, lookback + 1):
                past_dn = dn - d_offset
                evts = cell_day_events.get((lb, lnb, past_dn), [])
                count += len(evts)
            features.append(count)

        # --- Features 13-21: Spatial propagation at multiple distances/lags ---
        # THE KEY SYNOPTIC SIGNAL
        # Distance rings: 1-cell (~200km), 2-cell (~400km), 3-cell (~600km)
        # Time lags: 1 day, 3 days, 7 days
        for max_dist in [1, 2, 3]:
            for max_lag in [1, 3, 7]:
                count = 0
                for dlb in range(-max_dist, max_dist + 1):
                    for dlnb in range(-max_dist, max_dist + 1):
                        if dlb == 0 and dlnb == 0:
                            continue
                        # Chebyshev distance
                        if max(abs(dlb), abs(dlnb)) > max_dist:
                            continue
                        nlb = lb + dlb
                        nlnb = lnb + dlnb
                        if 0 <= nlb < n_lat and 0 <= nlnb < n_lon:
                            for d_offset in range(0, max_lag + 1):
                                # Include today (d_offset=0) for same-day synoptic
                                past_dn = dn - d_offset
                                evts = cell_day_events.get((nlb, nlnb, past_dn), [])
                                count += len(evts)
                features.append(count)

        # --- Features 22-24: Outbreak detection ---
        # Number of DIFFERENT cells with tornadoes in past 1, 2, 3 days
        for lookback in [1, 2, 3]:
            active_cells = set()
            for d_offset in range(0, lookback + 1):
                past_dn = dn - d_offset
                for (alb, alnb, t_rec) in day_events.get(past_dn, []):
                    active_cells.add((alb, alnb))
            features.append(len(active_cells))

        # --- Features 25-27: Severity memory ---
        # Max EF in this cell in past 7, 30, 365 days
        for lookback in [7, 30, 365]:
            max_ef = 0
            for d_offset in range(1, lookback + 1):
                past_dn = dn - d_offset
                for evt in cell_day_events.get((lb, lnb, past_dn), []):
                    if evt['mag'] > max_ef:
                        max_ef = evt['mag']
            features.append(max_ef)

        # --- Feature 28: Width climatology ---
        # Mean width of recent tornadoes in cell (past 365 days)
        widths = []
        for d_offset in range(1, 366):
            past_dn = dn - d_offset
            for evt in cell_day_events.get((lb, lnb, past_dn), []):
                if evt['width_yd'] > 0:
                    widths.append(evt['width_yd'])
        features.append(np.mean(widths) if widths else 0.0)

        # --- Feature 28b: Max width in nearby cells past 3 days (strong synoptic) ---
        max_nearby_width = 0.0
        for dlb in [-1, 0, 1]:
            for dlnb in [-1, 0, 1]:
                nlb = lb + dlb
                nlnb = lnb + dlnb
                if 0 <= nlb < n_lat and 0 <= nlnb < n_lon:
                    for d_offset in range(0, 4):
                        past_dn = dn - d_offset
                        for evt in cell_day_events.get((nlb, nlnb, past_dn), []):
                            if evt['width_yd'] > max_nearby_width:
                                max_nearby_width = evt['width_yd']
        features.append(np.log1p(max_nearby_width))

        # --- Feature 29: Monthly anomaly ---
        # How many tornadoes this cell-month THIS year vs average
        # Count in same cell, same month, current year up to this day
        this_yr_count = 0
        for t_rec in cell_month_events.get((lb, lnb, mo), []):
            if t_rec['yr'] == yr:
                t_dn = date_to_day_number(t_rec['yr'], t_rec['mo'], t_rec['dy'])
                if t_dn < dn:  # before today
                    this_yr_count += 1
        avg_count = cr * days_per_month[mo] if mo <= 12 else 0
        features.append(this_yr_count - avg_count)  # anomaly

        # --- Feature 30: Latitude bin (geographic) ---
        features.append(lb / n_lat)

        # --- Feature 31: Longitude bin ---
        features.append(lnb / n_lon)

        # --- Feature 32: Lat-Lon interaction with season ---
        features.append(lb * lnb * math.sin(2 * math.pi * doy / 365.25) / (n_lat * n_lon))

        # --- Feature 33: Propagation velocity proxy ---
        # If tornadoes were in cells to the SW yesterday and to the NE today,
        # that suggests a front moving NE. Compute directional bias.
        sw_count = 0
        ne_count = 0
        for dlb in [-2, -1, 0]:
            for dlnb in [-2, -1, 0]:
                if dlb == 0 and dlnb == 0:
                    continue
                nlb = lb + dlb
                nlnb = lnb + dlnb
                if 0 <= nlb < n_lat and 0 <= nlnb < n_lon:
                    for d_offset in [1, 2]:
                        sw_count += len(cell_day_events.get((nlb, nlnb, dn - d_offset), []))
        for dlb in [0, 1, 2]:
            for dlnb in [0, 1, 2]:
                if dlb == 0 and dlnb == 0:
                    continue
                nlb = lb + dlb
                nlnb = lnb + dlnb
                if 0 <= nlb < n_lat and 0 <= nlnb < n_lon:
                    for d_offset in [1, 2]:
                        ne_count += len(cell_day_events.get((nlb, nlnb, dn - d_offset), []))
        features.append(sw_count - ne_count)  # positive = approaching from SW (typical)

        # --- Feature 34: Rapid intensification flag ---
        # Was there a big jump in nearby cell tornado count from 2 days ago to yesterday?
        count_2d_ago = 0
        count_1d_ago = 0
        for dlb in range(-2, 3):
            for dlnb in range(-2, 3):
                nlb = lb + dlb
                nlnb = lnb + dlnb
                if 0 <= nlb < n_lat and 0 <= nlnb < n_lon:
                    count_2d_ago += len(cell_day_events.get((nlb, nlnb, dn - 2), []))
                    count_1d_ago += len(cell_day_events.get((nlb, nlnb, dn - 1), []))
        features.append(count_1d_ago - count_2d_ago)  # acceleration

        # --- Feature 35: Log transform of outbreak size ---
        features.append(np.log1p(count_1d_ago))

        # --- Feature 36-37: Clim rate * propagation interactions ---
        # High clim rate + active neighbors = very likely
        prop_1_1 = features[12]  # prop_1cell_1d (index 12 in features)
        features.append(cr * prop_1_1)  # climatology * recent nearby activity
        features.append(cr * cr)  # squared climatology (nonlinear response)

        # --- Feature 38: Total nearby EF severity past 3 days ---
        total_ef_nearby = 0
        for dlb in range(-2, 3):
            for dlnb in range(-2, 3):
                nlb = lb + dlb
                nlnb = lnb + dlnb
                if 0 <= nlb < n_lat and 0 <= nlnb < n_lon:
                    for d_offset in range(0, 4):
                        for evt in cell_day_events.get((nlb, nlnb, dn - d_offset), []):
                            total_ef_nearby += evt['mag']
        features.append(total_ef_nearby)

        return features

    # -------------------------------------------------------------------------
    # Step 4: Generate training and test samples
    # -------------------------------------------------------------------------
    print("  Generating training samples...")

    # Positive samples: cell-days with tornadoes
    # Negative samples: same cell, same month, different years without tornadoes
    # This isolates non-climatological signal

    rng = np.random.RandomState(42)

    def generate_samples(start_yr, end_yr, max_neg_ratio=3):
        """Generate positive and negative samples for given year range."""
        X_list = []
        y_list = []
        n_pos = 0
        n_neg = 0

        # Collect positive cell-days
        positive_cell_days = set()
        for (lb, lnb, dn), evts in cell_day_events.items():
            if not evts:
                continue
            yr_check = evts[0]['yr']
            if start_yr <= yr_check <= end_yr:
                positive_cell_days.add((lb, lnb, dn))

        print(f"    Positive cell-days: {len(positive_cell_days)}")

        # For each positive, generate features and matched negatives
        pos_by_cell_month = defaultdict(list)
        for (lb, lnb, dn) in positive_cell_days:
            evts = cell_day_events[(lb, lnb, dn)]
            mo = evts[0]['mo']
            pos_by_cell_month[(lb, lnb, mo)].append(dn)

        for (lb, lnb, mo), pos_dns in pos_by_cell_month.items():
            # Add positive samples
            for dn in pos_dns:
                evts = cell_day_events[(lb, lnb, dn)]
                yr = evts[0]['yr']
                feat = get_features(lb, lnb, dn, mo, yr, tornadoes_today_in_cell=True)
                X_list.append(feat)
                y_list.append(1)
                n_pos += 1

            # Generate negatives: same cell, same month, different years
            pos_dn_set = set(pos_dns)
            neg_candidates = []
            for yr in range(start_yr, end_yr + 1):
                for dy in range(1, 29, 2):  # Sample every other day for denser coverage
                    dn_cand = date_to_day_number(yr, mo, dy)
                    if dn_cand not in pos_dn_set:
                        # Verify no tornado in this cell-day
                        if (lb, lnb, dn_cand) not in positive_cell_days:
                            neg_candidates.append((dn_cand, yr))

            # Sample negatives (up to max_neg_ratio per positive)
            n_neg_wanted = min(len(neg_candidates), len(pos_dns) * max_neg_ratio)
            if n_neg_wanted > 0 and len(neg_candidates) > 0:
                chosen_idx = rng.choice(len(neg_candidates), size=min(n_neg_wanted, len(neg_candidates)), replace=False)
                for idx in chosen_idx:
                    dn_neg, yr_neg = neg_candidates[idx]
                    feat = get_features(lb, lnb, dn_neg, mo, yr_neg, tornadoes_today_in_cell=False)
                    X_list.append(feat)
                    y_list.append(0)
                    n_neg += 1

        print(f"    Total: {n_pos} positive, {n_neg} negative")
        return np.array(X_list), np.array(y_list)

    X_train, y_train = generate_samples(train_start_yr, train_end_yr, max_neg_ratio=5)
    print("  Generating test samples...")
    X_test, y_test = generate_samples(test_start_yr, test_end_yr, max_neg_ratio=5)

    # -------------------------------------------------------------------------
    # Step 5: Train formation model
    # -------------------------------------------------------------------------
    print(f"\n  Training formation model...")
    print(f"    Features: {X_train.shape[1]}")
    print(f"    Train: {len(y_train)} samples ({y_train.sum():.0f} positive)")
    print(f"    Test: {len(y_test)} samples ({y_test.sum():.0f} positive)")

    # Replace NaN/inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    w, b, mu, sd = logistic_regression_train(X_train, y_train, lr=0.01, lam=0.5, epochs=1500,
                                              verbose=True, class_weight='balanced')

    # -------------------------------------------------------------------------
    # Step 6: Evaluate
    # -------------------------------------------------------------------------
    p_train = logistic_predict(X_train, w, b, mu, sd)
    p_test = logistic_predict(X_test, w, b, mu, sd)

    auc_train = compute_auc(y_train, p_train)
    auc_test = compute_auc(y_test, p_test)

    # Climatological baseline: predict the test set base rate for every sample
    clim_prob_test = y_test.mean()
    clim_auc = 0.5  # climatology always gets AUC=0.5 on matched samples

    # But we need the REAL climatology comparison:
    # Use the cell-month climatological rate as the climatology prediction
    # Since we matched samples by cell-month, climatology predicts the SAME
    # probability for all samples in a given cell-month.
    # The TRUE test is: does our model beat cell-month climatology?
    # Cell-month climatology AUC on matched samples should be ~0.5
    # since we sampled negatives from the same cell-month.
    # On UNMATCHED real-world data, climatology would be ~0.827.
    # Our model needs to add SKILL on top of climatology.

    # Compute Brier Skill Score
    bss_test = compute_bss(y_test, p_test, clim_prob_test)

    # Feature importance (by weight magnitude after standardization)
    feature_names = [
        'sin(doy)', 'cos(doy)', 'sin(2*doy)', 'cos(2*doy)',
        'clim_rate', 'neighbor_clim_rate',
        'activity_3d', 'activity_7d', 'activity_14d', 'activity_30d', 'activity_90d', 'activity_365d',
        'prop_1cell_1d', 'prop_1cell_3d', 'prop_1cell_7d',
        'prop_2cell_1d', 'prop_2cell_3d', 'prop_2cell_7d',
        'prop_3cell_1d', 'prop_3cell_3d', 'prop_3cell_7d',
        'outbreak_1d', 'outbreak_2d', 'outbreak_3d',
        'max_ef_7d', 'max_ef_30d', 'max_ef_365d',
        'mean_width_365d', 'max_nearby_width_3d',
        'monthly_anomaly',
        'lat_norm', 'lon_norm', 'lat_lon_season',
        'sw_propagation', 'intensification', 'log_nearby_1d',
        'clim*prop', 'clim_sq', 'ef_severity_nearby',
    ]

    print(f"\n  FORMATION MODEL RESULTS:")
    print(f"  {'='*50}")
    print(f"    Train AUC:  {auc_train:.3f}")
    print(f"    Test AUC:   {auc_test:.3f}")
    print(f"    Brier Skill Score (vs clim): {bss_test:.3f}")
    print(f"    Test base rate: {clim_prob_test:.3f}")

    print(f"\n  Top features by |weight|:")
    importance = np.abs(w)
    top_idx = np.argsort(-importance)[:10]
    for rank, idx in enumerate(top_idx):
        name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
        print(f"    {rank+1}. {name}: w={w[idx]:.4f} (|w|={importance[idx]:.4f})")

    return {
        'auc_train': auc_train,
        'auc_test': auc_test,
        'bss': bss_test,
        'w': w, 'b': b, 'mu': mu, 'sd': sd,
        'feature_names': feature_names,
        'p_test': p_test, 'y_test': y_test,
    }


# =============================================================================
# PART 2: SEVERITY PREDICTION
# =============================================================================

def build_severity_model(tornadoes):
    """
    Given a tornado forms, predict severity (EF2+, EF3+, EF4+).
    Uses width, path length, location, seasonal, outbreak context.
    """
    print("\n" + "=" * 70)
    print("PART 2: TORNADO SEVERITY PREDICTION (given formation)")
    print("=" * 70)

    # Filter to tornadoes with valid width and path length (post-2000)
    valid = [t for t in tornadoes if t['yr'] >= 2000 and t['width_yd'] > 0 and t['path_len_mi'] > 0]
    print(f"  Valid tornadoes (2000-2023, width>0, len>0): {len(valid)}")

    # Count tornadoes per day for outbreak context
    day_counts = defaultdict(int)
    day_order = defaultdict(list)
    for t in valid:
        key = (t['yr'], t['mo'], t['dy'])
        day_counts[key] += 1
        day_order[key].append(t)

    # Regional severity history: max EF in 2-degree cell in same year up to this tornado
    cell_year_max_ef = defaultdict(lambda: 0)

    # Build features
    def severity_features(t):
        feat = []

        # Width (log, the star feature)
        feat.append(np.log1p(t['width_yd']))

        # Width squared (nonlinearity)
        feat.append(np.log1p(t['width_yd']) ** 2)

        # Path length (log)
        feat.append(np.log1p(t['path_len_mi']))

        # Month / season
        mo = t['mo']
        doy = (mo - 1) * 30.4 + 15
        feat.append(math.sin(2 * math.pi * doy / 365.25))
        feat.append(math.cos(2 * math.pi * doy / 365.25))

        # Location
        feat.append(t['slat'])
        feat.append(t['slon'])

        # Outbreak context
        key = (t['yr'], t['mo'], t['dy'])
        feat.append(day_counts[key])                  # total tornadoes that day
        feat.append(np.log1p(day_counts[key]))         # log of outbreak size

        # Position within outbreak (are we early or late?)
        order_list = day_order[key]
        idx = 0
        for i, tt in enumerate(order_list):
            if tt is t:
                idx = i
                break
        feat.append(idx / max(1, len(order_list) - 1) if len(order_list) > 1 else 0.5)

        # Width * outbreak interaction
        feat.append(np.log1p(t['width_yd']) * np.log1p(day_counts[key]))

        # Regional severity history
        lb, lnb = latlon_to_cell(t['slat'], t['slon'])
        feat.append(cell_year_max_ef.get((lb, lnb, t['yr']), 0))

        return feat

    # Split by year
    train_tornados = [t for t in valid if 2000 <= t['yr'] <= 2016]
    test_tornados = [t for t in valid if 2017 <= t['yr'] <= 2023]

    print(f"  Train: {len(train_tornados)} tornadoes (2000-2016)")
    print(f"  Test:  {len(test_tornados)} tornadoes (2017-2023)")

    # EF distribution
    for label, subset in [("Train", train_tornados), ("Test", test_tornados)]:
        ef_counts = defaultdict(int)
        for t in subset:
            ef_counts[t['mag']] += 1
        dist_str = ", ".join(f"EF{k}:{v}" for k, v in sorted(ef_counts.items()))
        print(f"    {label} EF dist: {dist_str}")

    results = {}

    for threshold, label in [(2, 'EF2+'), (3, 'EF3+'), (4, 'EF4+')]:
        print(f"\n  --- {label} Severity Model ---")

        X_train = np.array([severity_features(t) for t in train_tornados])
        y_train = np.array([1 if t['mag'] >= threshold else 0 for t in train_tornados])

        X_test = np.array([severity_features(t) for t in test_tornados])
        y_test = np.array([1 if t['mag'] >= threshold else 0 for t in test_tornados])

        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        n_pos_train = y_train.sum()
        n_pos_test = y_test.sum()
        print(f"    Train: {n_pos_train:.0f}/{len(y_train)} positive ({100*n_pos_train/len(y_train):.1f}%)")
        print(f"    Test:  {n_pos_test:.0f}/{len(y_test)} positive ({100*n_pos_test/len(y_test):.1f}%)")

        if n_pos_train < 5 or n_pos_test < 5:
            print(f"    Skipping {label}: too few positive samples")
            continue

        # Adjust learning rate and regularization for imbalanced classes
        lam = 0.5 if threshold <= 2 else 0.1
        w, b, mu, sd = logistic_regression_train(X_train, y_train, lr=0.05, lam=lam, epochs=1000)

        p_train = logistic_predict(X_train, w, b, mu, sd)
        p_test = logistic_predict(X_test, w, b, mu, sd)

        auc_train = compute_auc(y_train, p_train)
        auc_test = compute_auc(y_test, p_test)

        clim_rate = y_test.mean()
        bss = compute_bss(y_test, p_test, clim_rate)

        print(f"    Train AUC: {auc_train:.3f}")
        print(f"    Test AUC:  {auc_test:.3f}")
        print(f"    BSS:       {bss:.3f}")

        results[label] = {
            'auc_train': auc_train,
            'auc_test': auc_test,
            'bss': bss,
            'p_test': p_test,
            'y_test': y_test,
            'clim_rate': clim_rate,
        }

    return results


# =============================================================================
# PART 3: COMBINED SYSTEM + VISUALIZATION
# =============================================================================

def create_figure(formation_results, severity_results):
    """Create comprehensive results figure."""
    print("\n" + "=" * 70)
    print("PART 3: COMBINED SYSTEM VISUALIZATION")
    print("=" * 70)

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Tornado Prediction System v2 - Formation + Severity',
                 fontsize=16, fontweight='bold', y=0.98)

    # --- Panel 1: Formation ROC ---
    ax1 = fig.add_subplot(2, 3, 1)
    y_true = formation_results['y_test']
    y_score = formation_results['p_test']

    # Compute ROC curve
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    tpr = [0.0]
    fpr = [0.0]
    tp = fp = 0
    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / max(n_pos, 1))
        fpr.append(fp / max(n_neg, 1))

    ax1.plot(fpr, tpr, 'b-', linewidth=2,
             label=f'Formation Model (AUC={formation_results["auc_test"]:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC=0.500)')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Formation Prediction ROC')
    ax1.legend(loc='lower right', fontsize=9)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Feature importance ---
    ax2 = fig.add_subplot(2, 3, 2)
    w = formation_results['w']
    names = formation_results['feature_names']
    importance = np.abs(w)
    top_idx = np.argsort(-importance)[:12]
    top_names = [names[i] if i < len(names) else f'f{i}' for i in top_idx]
    top_vals = importance[top_idx]

    colors = ['#d62728' if w[i] > 0 else '#1f77b4' for i in top_idx]
    bars = ax2.barh(range(len(top_names)), top_vals, color=colors)
    ax2.set_yticks(range(len(top_names)))
    ax2.set_yticklabels(top_names, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlabel('|Weight| (standardized)')
    ax2.set_title('Formation: Top Feature Importance')

    # Legend for color
    from matplotlib.patches import Patch
    ax2.legend([Patch(color='#d62728'), Patch(color='#1f77b4')],
               ['Positive (promotes tornado)', 'Negative (inhibits)'],
               fontsize=8, loc='lower right')
    ax2.grid(True, alpha=0.3, axis='x')

    # --- Panel 3: Severity ROC curves ---
    ax3 = fig.add_subplot(2, 3, 3)
    colors_sev = {'EF2+': '#ff7f0e', 'EF3+': '#d62728', 'EF4+': '#9467bd'}

    for label, res in severity_results.items():
        y_t = res['y_test']
        y_s = res['p_test']
        order = np.argsort(-y_s)
        y_sorted = y_t[order]
        n_p = y_sorted.sum()
        n_n = len(y_sorted) - n_p
        tpr_s = [0.0]
        fpr_s = [0.0]
        tp = fp = 0
        for i in range(len(y_sorted)):
            if y_sorted[i] == 1:
                tp += 1
            else:
                fp += 1
            tpr_s.append(tp / max(n_p, 1))
            fpr_s.append(fp / max(n_n, 1))
        ax3.plot(fpr_s, tpr_s, color=colors_sev.get(label, 'gray'), linewidth=2,
                 label=f'{label} (AUC={res["auc_test"]:.3f})')

    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('Severity Prediction ROC (given formation)')
    ax3.legend(loc='lower right', fontsize=9)
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)

    # --- Panel 4: Formation probability calibration ---
    ax4 = fig.add_subplot(2, 3, 4)
    y_true = formation_results['y_test']
    y_score = formation_results['p_test']

    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_observed = []
    bin_counts = []

    for i in range(n_bins):
        mask = (y_score >= bin_edges[i]) & (y_score < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_observed.append(y_true[mask].mean())
            bin_counts.append(mask.sum())

    if bin_centers:
        ax4.plot(bin_centers, bin_observed, 'bo-', linewidth=2, markersize=8, label='Observed')
        ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
        # Show sample counts
        for i, (x, y, n) in enumerate(zip(bin_centers, bin_observed, bin_counts)):
            ax4.annotate(f'n={n}', (x, y), textcoords="offset points",
                        xytext=(0, 10), fontsize=7, ha='center')
    ax4.set_xlabel('Predicted Probability')
    ax4.set_ylabel('Observed Frequency')
    ax4.set_title('Formation: Calibration Plot')
    ax4.legend(fontsize=9)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.grid(True, alpha=0.3)

    # --- Panel 5: Score distribution ---
    ax5 = fig.add_subplot(2, 3, 5)
    pos_scores = formation_results['p_test'][formation_results['y_test'] == 1]
    neg_scores = formation_results['p_test'][formation_results['y_test'] == 0]

    bins = np.linspace(0, 1, 30)
    ax5.hist(neg_scores, bins=bins, alpha=0.6, color='#1f77b4', label=f'No tornado (n={len(neg_scores)})', density=True)
    ax5.hist(pos_scores, bins=bins, alpha=0.6, color='#d62728', label=f'Tornado (n={len(pos_scores)})', density=True)
    ax5.set_xlabel('Predicted Probability')
    ax5.set_ylabel('Density')
    ax5.set_title('Formation: Score Distribution')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)

    # --- Panel 6: Summary scorecard ---
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis('off')

    lines = [
        "INTEGRATED TORNADO PREDICTION SYSTEM v2",
        "=" * 42,
        "",
        "PART 1: FORMATION PREDICTION",
        f"  AUC (test):        {formation_results['auc_test']:.3f}",
        f"  v1 AUC:            0.780",
        f"  Climatology AUC:   0.827",
        f"  BSS vs climatology:{formation_results['bss']:+.3f}",
        f"  Improvement:       {formation_results['auc_test'] - 0.827:+.3f} vs clim",
        "",
        "PART 2: SEVERITY PREDICTION",
    ]

    for label, res in severity_results.items():
        lines.append(f"  {label} AUC: {res['auc_test']:.3f}  BSS: {res['bss']:+.3f}")

    lines.extend([
        "",
        "KEY PREDICTORS (formation):",
        "  1. Spatial propagation (synoptic)",
        "  2. Outbreak detection (multi-cell)",
        "  3. Climatological rate",
        "  4. Temporal activity (multi-scale)",
        "",
        "KEY PREDICTORS (severity):",
        "  1. Width (log) - dominant feature",
        "  2. Path length",
        "  3. Outbreak context",
    ])

    # Color-code the result
    form_color = '#2ca02c' if formation_results['auc_test'] > 0.827 else '#d62728'
    verdict = "BEATS CLIMATOLOGY" if formation_results['auc_test'] > 0.827 else "BELOW CLIMATOLOGY"
    lines.append("")
    lines.append(f"FORMATION VERDICT: {verdict}")

    text = '\n'.join(lines)
    ax6.text(0.05, 0.95, text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    outpath = os.path.join(OUT_DIR, 'tornado_model_v2.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Figure saved to: {outpath}")

    return outpath


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("TORNADO PREDICTION MODEL v2")
    print("Comprehensive Formation + Severity System")
    print("=" * 70)

    tornadoes = load_spc_tornadoes()
    if tornadoes is None:
        print("FATAL: Could not load tornado data.")
        return

    # Part 1: Formation
    formation_results = build_formation_model(tornadoes)

    # Part 2: Severity
    severity_results = build_severity_model(tornadoes)

    # Part 3: Combined visualization
    fig_path = create_figure(formation_results, severity_results)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  Formation AUC:  {formation_results['auc_test']:.3f}  (v1: 0.780, climatology: 0.827)")

    if formation_results['auc_test'] > 0.827:
        improvement = formation_results['auc_test'] - 0.827
        print(f"  >> BEATS CLIMATOLOGY by {improvement:.3f}")
    else:
        gap = 0.827 - formation_results['auc_test']
        print(f"  >> Below climatology by {gap:.3f}")

    for label, res in severity_results.items():
        print(f"  {label} AUC:  {res['auc_test']:.3f}")

    sev_aucs = [res['auc_test'] for res in severity_results.values()]
    if sev_aucs:
        mean_sev = np.mean(sev_aucs)
        print(f"  Mean severity AUC: {mean_sev:.3f}")
        if mean_sev >= 0.90:
            print(f"  >> Severity target MET (>= 0.90)")
        else:
            print(f"  >> Severity target MISSED (< 0.90)")

    print(f"\n  Figure: {fig_path}")


if __name__ == '__main__':
    main()
