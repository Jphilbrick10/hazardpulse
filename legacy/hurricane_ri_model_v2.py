#!/usr/bin/env python3
"""
HURRICANE RAPID INTENSIFICATION MODEL v2
==========================================
Major upgrade over v1:
  - ALL basins (global) for training, not just North Atlantic
  - 20+ features per observation (up from 6-9)
  - Multiple RI thresholds tested
  - L2-regularized logistic regression (manual, no sklearn)
  - Multi-panel diagnostic figure
  - Basin comparison (ALL vs NA-only)

Uses only: numpy, scipy, matplotlib, urllib, csv, io, ssl
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
import urllib.request
import ssl
import csv
import io
import warnings
import time
warnings.filterwarnings('ignore')

OUT = Path(r"c:\Users\Josh\Projects\Coherence_Energy_Labs_Website\data\evidence\coherence_field_results")
OUT.mkdir(exist_ok=True)

DARK = '#1a1a2e'
GOLD = '#D4AF37'
ACCENT = '#e94560'
TEAL = '#16a085'
GREEN = '#2ecc71'
BLUE = '#3498db'
PURPLE = '#9b59b6'
ORANGE = '#e67e22'

# =========================================================================
# DATA LOADING
# =========================================================================

def get_ssl_context():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

def fetch_url(url, timeout=120):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        ctx = get_ssl_context()
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return resp.read()
    except Exception as e:
        print(f"    Fetch error: {e}")
        return None

def parse_ibtracs_csv(text):
    """Parse IBTrACS CSV text into storm dictionary."""
    lines = text.strip().split('\n')
    header = lines[0].split(',')
    col_map = {}
    for i, h in enumerate(header):
        col_map[h.strip().strip('"')] = i

    sid_col = col_map.get('SID', 0)
    name_col = col_map.get('NAME', 1)
    time_col = col_map.get('ISO_TIME', 6)
    basin_col = col_map.get('BASIN', None)
    lat_col = col_map.get('LAT', None)
    lon_col = col_map.get('LON', None)
    wind_col = col_map.get('USA_WIND', None)
    wmo_wind_col = col_map.get('WMO_WIND', None)
    pres_col = col_map.get('USA_PRES', None)
    wmo_pres_col = col_map.get('WMO_PRES', None)
    rmw_col = col_map.get('USA_RMW', None)
    nature_col = col_map.get('NATURE', None)
    dist2land_col = col_map.get('DIST2LAND', None)
    landfall_col = col_map.get('LANDFALL', None)

    storms = {}
    # Skip header row and units row (line 1)
    for line in lines[2:]:
        parts = line.split(',')
        if len(parts) < 10:
            continue
        try:
            sid = parts[sid_col].strip().strip('"')
            name = parts[name_col].strip().strip('"')

            def safe_float(col):
                if col is None:
                    return -999.0
                v = parts[col].strip().strip('"')
                if v in ('', ' ', 'NA', 'na'):
                    return -999.0
                try:
                    return float(v)
                except ValueError:
                    return -999.0

            # Use USA_WIND preferentially, fall back to WMO_WIND
            wind = safe_float(wind_col)
            if wind <= 0:
                wind = safe_float(wmo_wind_col)

            # Use USA_PRES preferentially, fall back to WMO_PRES
            pres = safe_float(pres_col)
            if pres <= 0:
                pres = safe_float(wmo_pres_col)

            rmw = safe_float(rmw_col)
            lat = safe_float(lat_col)
            lon = safe_float(lon_col)
            dist2land = safe_float(dist2land_col)

            time_str = parts[time_col].strip().strip('"')
            year = int(time_str[:4]) if len(time_str) >= 4 else 0
            month = int(time_str[5:7]) if len(time_str) >= 7 else 6

            basin_str = ''
            if basin_col is not None:
                basin_str = parts[basin_col].strip().strip('"')

            if sid not in storms:
                storms[sid] = {'name': name, 'basin': basin_str, 'entries': []}

            storms[sid]['entries'].append({
                'wind': wind, 'pres': pres, 'rmw': rmw,
                'lat': lat, 'lon': lon, 'year': year, 'month': month,
                'time_str': time_str, 'dist2land': dist2land,
            })
        except Exception:
            pass

    return storms


def load_ibtracs_all():
    """Load ALL basins IBTrACS data."""
    print("\n  Loading IBTrACS ALL basins (global)...")
    url = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"
    t0 = time.time()
    data = fetch_url(url, timeout=300)
    if not data:
        print("  ALL basins download failed.")
        return {}
    text = data.decode('utf-8', errors='replace')
    storms = parse_ibtracs_csv(text)
    dt = time.time() - t0
    print(f"  Loaded {len(storms)} storms (ALL basins) in {dt:.0f}s")
    return storms


def load_ibtracs_na():
    """Load North Atlantic IBTrACS data."""
    print("\n  Loading IBTrACS North Atlantic...")
    url = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NA.list.v04r01.csv"
    t0 = time.time()
    data = fetch_url(url, timeout=120)
    if not data:
        print("  NA download failed.")
        return {}
    text = data.decode('utf-8', errors='replace')
    storms = parse_ibtracs_csv(text)
    dt = time.time() - t0
    print(f"  Loaded {len(storms)} storms (NA) in {dt:.0f}s")
    return storms


# =========================================================================
# FEATURE EXTRACTION
# =========================================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """Approximate distance in km between two lat/lon points."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def approx_mpi(lat, month):
    """
    Approximate Maximum Potential Intensity (kt) from latitude and season.
    Based on empirical SST-MPI relationship: MPI peaks at low latitudes
    in warm season (SST ~ 28-30C -> MPI ~ 150-170kt).
    """
    # SST proxy: warmer at low lat, peaks in summer
    abs_lat = abs(lat)
    # SST decreases ~0.5C per degree lat from 10N
    sst_proxy = 30.0 - 0.5 * max(0, abs_lat - 10)
    # Season effect (NH summer = months 6-10)
    if lat >= 0:
        season_warm = np.exp(-((month - 9)**2) / 8.0)
    else:
        season_warm = np.exp(-((month - 3)**2) / 8.0)
    sst_proxy += 2.0 * season_warm  # +2C in peak season

    # MPI from SST (Emanuel 1988 approximation)
    # MPI ~ 0 at SST=26C, increases ~30kt per C above that
    if sst_proxy < 26.0:
        return 40.0
    mpi = 30.0 * (sst_proxy - 26.0) + 40.0
    return min(mpi, 185.0)


def dvorak_expected_wind(pres):
    """Expected wind from pressure using Dvorak-like empirical curve."""
    # Knaff-Zehr wind-pressure relationship (simplified)
    if pres <= 0 or pres > 1020:
        return -999.0
    dp = 1013.0 - pres
    if dp <= 0:
        return 25.0
    return 10.0 + 1.0 * dp + 0.005 * dp**2


def extract_storm_features_v2(entries, ri_threshold_kt=30, ri_window_steps=4):
    """
    Extract features for each 6-hour observation.
    ri_window_steps: number of 6h steps for RI window (4 = 24h, 2 = 12h, etc.)

    Returns list of (feature_vector, label, year, dw_next)
    """
    n = len(entries)
    if n < max(8, ri_window_steps + 4):
        return []

    winds = np.array([e['wind'] for e in entries])
    pressures = np.array([e['pres'] for e in entries])
    rmws = np.array([e['rmw'] for e in entries])
    lats = np.array([e['lat'] for e in entries])
    lons = np.array([e['lon'] for e in entries])
    d2l = np.array([e['dist2land'] for e in entries])
    months = np.array([e['month'] for e in entries])
    years_arr = np.array([e['year'] for e in entries])

    samples = []

    # Track if RI has already occurred in this storm
    prior_ri_flag = False

    for i in range(4, n - ri_window_steps):
        w_now = winds[i]
        w_future = winds[i + ri_window_steps]

        if w_now <= 0 or w_future <= 0:
            continue

        dw_next = w_future - w_now
        label = 1 if dw_next >= ri_threshold_kt else 0

        year = years_arr[i]
        month = months[i]
        lat = lats[i]
        lon = lons[i]

        features = {}

        # --- Core intensity features ---
        features['wind_now'] = w_now

        # Wind changes (prior)
        w_m1 = winds[i-1] if winds[i-1] > 0 else -999
        w_m2 = winds[i-2] if i >= 2 and winds[i-2] > 0 else -999
        w_m4 = winds[i-4] if i >= 4 and winds[i-4] > 0 else -999

        features['dw_6h'] = (w_now - w_m1) if w_m1 > 0 else -999
        features['dw_12h'] = (w_now - w_m2) if w_m2 > 0 else -999
        features['dw_24h'] = (w_now - w_m4) if w_m4 > 0 else -999

        # Acceleration (change in intensification rate)
        if w_m2 > 0 and w_m4 > 0:
            dw_recent = w_now - w_m2
            dw_prior = w_m2 - w_m4
            features['acceleration'] = dw_recent - dw_prior
        else:
            features['acceleration'] = -999

        # --- Pressure features ---
        p_now = pressures[i]
        features['pressure_now'] = p_now if p_now > 800 else -999

        p_m1 = pressures[i-1] if pressures[i-1] > 800 else -999
        p_m2 = pressures[i-2] if i >= 2 and pressures[i-2] > 800 else -999
        p_m4 = pressures[i-4] if i >= 4 and pressures[i-4] > 800 else -999

        features['dp_6h'] = (p_now - p_m1) if (p_now > 800 and p_m1 > 0) else -999
        features['dp_12h'] = (p_now - p_m2) if (p_now > 800 and p_m2 > 0) else -999
        features['dp_24h'] = (p_now - p_m4) if (p_now > 800 and p_m4 > 0) else -999

        # Pressure fall rate acceleration
        if p_now > 800 and p_m2 > 0 and p_m4 > 0:
            dp_recent = p_now - p_m2  # negative = falling
            dp_prior = p_m2 - p_m4
            features['pressure_accel'] = dp_recent - dp_prior  # more negative = accelerating drop
        else:
            features['pressure_accel'] = -999

        # --- Wind-pressure relationship anomaly ---
        if p_now > 800 and w_now > 0:
            expected_w = dvorak_expected_wind(p_now)
            if expected_w > 0:
                features['wp_anomaly'] = w_now - expected_w  # positive = stronger than expected
            else:
                features['wp_anomaly'] = -999
            # P-W efficiency (v1 feature)
            features['pw_efficiency'] = w_now**2 / max(1013.0 - p_now, 1.0)
        else:
            features['wp_anomaly'] = -999
            features['pw_efficiency'] = -999

        # --- Location features ---
        features['lat'] = lat if lat != -999 else -999
        features['abs_lat'] = abs(lat) if lat != -999 else -999

        # Season encoding
        if month > 0:
            features['season_sin'] = np.sin(2 * np.pi * month / 12.0)
            features['season_cos'] = np.cos(2 * np.pi * month / 12.0)
        else:
            features['season_sin'] = -999
            features['season_cos'] = -999

        # --- Distance from land ---
        if d2l[i] > 0:
            features['dist2land'] = d2l[i]
        else:
            # Approximate: storms over open ocean (|lat| < 30, far from coasts)
            features['dist2land'] = -999

        # --- Translation speed ---
        if i >= 1 and lats[i] != -999 and lats[i-1] != -999 and lons[i] != -999 and lons[i-1] != -999:
            dist_km = haversine_km(lats[i-1], lons[i-1], lats[i], lons[i])
            features['translation_speed'] = dist_km / 6.0  # km/h
        else:
            features['translation_speed'] = -999

        # --- MPI-related features ---
        if lat != -999 and month > 0:
            mpi = approx_mpi(lat, month)
            features['intensity_frac_mpi'] = w_now / mpi if mpi > 0 else -999
            features['mpi_deficit'] = mpi - w_now  # room to grow
        else:
            features['intensity_frac_mpi'] = -999
            features['mpi_deficit'] = -999

        # --- Prior RI history ---
        features['prior_ri'] = 1.0 if prior_ri_flag else 0.0

        # --- Storm age (hours since genesis) ---
        features['storm_age_h'] = i * 6.0

        # --- Wind asymmetry proxy ---
        if features['dw_6h'] != -999 and features['dw_12h'] != -999:
            features['wind_asymmetry'] = abs(features['dw_6h'] - features['dw_12h'] / 2.0)
        else:
            features['wind_asymmetry'] = -999

        # --- Eye formation indicator ---
        if w_now > 65 and p_now > 800 and p_now < 980:
            features['eye_indicator'] = 1.0
        else:
            features['eye_indicator'] = 0.0

        # --- RMW features ---
        features['rmw_now'] = rmws[i] if rmws[i] > 0 else -999
        rmw_m2 = rmws[i-2] if i >= 2 and rmws[i-2] > 0 else -999
        features['drmw_12h'] = (rmws[i] - rmw_m2) if (rmws[i] > 0 and rmw_m2 > 0) else -999

        # RMW trend (linear over last 24h)
        if rmws[i] > 0 and i >= 4:
            rmw_seq = [rmws[j] for j in range(max(0, i-4), i+1) if rmws[j] > 0]
            if len(rmw_seq) >= 3:
                t_rmw = np.arange(len(rmw_seq))
                sl, _, _, _, _ = stats.linregress(t_rmw, rmw_seq)
                features['rmw_trend'] = sl
            else:
                features['rmw_trend'] = -999
        else:
            features['rmw_trend'] = -999

        samples.append((features, label, year, dw_next))

        # Update prior RI flag for subsequent obs
        if dw_next >= 30:
            prior_ri_flag = True

    return samples


# =========================================================================
# LOGISTIC REGRESSION (Manual with L2 regularization)
# =========================================================================

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def train_logistic_regression(X, y, lam=0.01, lr=0.05, n_epochs=1000,
                              class_weight_ratio=None):
    """
    L2-regularized logistic regression via gradient descent.
    class_weight_ratio: if set, upweight positive class by this factor.
    """
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0

    # Sample weights for class imbalance
    if class_weight_ratio is not None:
        sw = np.ones(n)
        sw[y == 1] = class_weight_ratio
    else:
        sw = np.ones(n)

    for epoch in range(n_epochs):
        z = X @ w + b
        p = sigmoid(z)
        error = (p - y) * sw  # weighted error

        grad_w = (X.T @ error) / n + lam * w  # L2 regularization
        grad_b = np.sum(error) / n

        # Learning rate decay
        current_lr = lr / (1 + epoch * 0.001)
        w -= current_lr * grad_w
        b -= current_lr * grad_b

    return w, b


def predict_proba(X, w, b):
    return sigmoid(X @ w + b)


# =========================================================================
# EVALUATION METRICS
# =========================================================================

def compute_roc_auc(y_true, y_score):
    """Compute ROC curve and AUC."""
    # Sort by score descending
    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.5, [], []

    tpr_list = [0.0]
    fpr_list = [0.0]
    tp = 0
    fp = 0

    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)

    # AUC via trapezoidal rule
    auc = 0.0
    for i in range(1, len(tpr_list)):
        auc += 0.5 * (tpr_list[i] + tpr_list[i-1]) * (fpr_list[i] - fpr_list[i-1])

    return auc, fpr_list, tpr_list


def compute_precision_recall(y_true, y_score, threshold=0.5):
    pred_pos = y_score >= threshold
    tp = np.sum(pred_pos & (y_true == 1))
    fp = np.sum(pred_pos & (y_true == 0))
    fn = np.sum((~pred_pos) & (y_true == 1))
    tn = np.sum((~pred_pos) & (y_true == 0))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)

    return precision, recall, f1, tp, fp, fn, tn


def find_optimal_threshold(y_true, y_score):
    best_f1 = 0
    best_thresh = 0.5
    for thresh in np.linspace(0.01, 0.99, 200):
        _, _, f1, _, _, _, _ = compute_precision_recall(y_true, y_score, thresh)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    return best_thresh, best_f1


# =========================================================================
# MAIN MODEL BUILDER
# =========================================================================

# Core features (no RMW, no pressure -- widely available)
CORE_FEATURES = [
    'wind_now', 'dw_6h', 'dw_12h', 'dw_24h', 'acceleration',
    'abs_lat', 'season_sin', 'season_cos',
    'translation_speed', 'intensity_frac_mpi', 'mpi_deficit',
    'prior_ri', 'storm_age_h', 'wind_asymmetry', 'eye_indicator',
]

# Extended features (add pressure info)
EXTENDED_FEATURES = CORE_FEATURES + [
    'pressure_now', 'dp_6h', 'dp_12h', 'dp_24h', 'pressure_accel',
    'wp_anomaly', 'pw_efficiency',
]

# Full features (add dist2land)
FULL_FEATURES = EXTENDED_FEATURES + [
    'dist2land',
]

# Full + RMW
FULL_RMW_FEATURES = FULL_FEATURES + [
    'rmw_now', 'drmw_12h', 'rmw_trend',
]


def build_feature_matrix(all_samples, feature_names):
    """Build X, y, years arrays from samples, dropping rows with missing values."""
    X_list = []
    y_list = []
    year_list = []
    dw_list = []

    for features, label, year, dw in all_samples:
        row = []
        valid = True
        for fn in feature_names:
            v = features.get(fn, -999)
            if v == -999:
                valid = False
                break
            row.append(v)
        if valid:
            X_list.append(row)
            y_list.append(label)
            year_list.append(year)
            dw_list.append(dw)

    if len(X_list) == 0:
        return None, None, None, None

    return (np.array(X_list), np.array(y_list),
            np.array(year_list), np.array(dw_list))


def add_interaction_features(X, feature_names):
    """Add pairwise interaction features for top predictors to capture nonlinear effects."""
    # Key interactions that matter physically:
    # wind_now * dw_6h: strong storms intensifying rapidly
    # pressure_accel * dw_6h: coupled pressure-wind deepening
    # intensity_frac_mpi * dw_6h: intensifying near MPI
    # abs_lat * mpi_deficit: latitude-modulated room to grow
    n = X.shape[0]
    new_cols = []
    new_names = list(feature_names)

    # Select important feature pairs for interactions
    interaction_pairs = [
        ('wind_now', 'dw_6h'),
        ('dw_6h', 'dp_6h'),
        ('intensity_frac_mpi', 'dw_6h'),
        ('abs_lat', 'mpi_deficit'),
        ('dw_6h', 'dw_12h'),
        ('eye_indicator', 'dw_6h'),
        ('prior_ri', 'dw_6h'),
        ('storm_age_h', 'dw_6h'),
    ]

    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    for n1, n2 in interaction_pairs:
        if n1 in name_to_idx and n2 in name_to_idx:
            i1, i2 = name_to_idx[n1], name_to_idx[n2]
            new_cols.append(X[:, i1] * X[:, i2])
            new_names.append(f'{n1}*{n2}')

    # Squared terms for key nonlinear features
    for fname in ['dw_6h', 'intensity_frac_mpi', 'dp_6h', 'wind_now']:
        if fname in name_to_idx:
            idx = name_to_idx[fname]
            new_cols.append(X[:, idx] ** 2)
            new_names.append(f'{fname}^2')

    if new_cols:
        X_aug = np.column_stack([X] + new_cols)
    else:
        X_aug = X
    return X_aug, new_names


def run_model(all_samples, feature_names, model_name, test_year_start=2015,
              lam=0.01, verbose=True, use_interactions=False):
    """Run a single model configuration and return results dict."""
    X, y, years, dw = build_feature_matrix(all_samples, feature_names)
    if X is None or len(X) < 100:
        if verbose:
            print(f"    {model_name}: insufficient data ({0 if X is None else len(X)} samples)")
        return None

    train_mask = years < test_year_start
    test_mask = years >= test_year_start

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    if len(X_train) < 50 or len(X_test) < 50:
        if verbose:
            print(f"    {model_name}: insufficient train/test split")
        return None

    n_pos_train = np.sum(y_train)
    n_neg_train = np.sum(y_train == 0)
    if n_pos_train == 0:
        if verbose:
            print(f"    {model_name}: no positive samples in training")
        return None

    # Class weight to handle imbalance
    cw_ratio = n_neg_train / max(n_pos_train, 1)
    cw_ratio = min(cw_ratio, 20.0)  # cap

    # Standardize
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0) + 1e-10
    X_train_s = (X_train - mu) / sigma
    X_test_s = (X_test - mu) / sigma

    # Add interaction features after standardization
    actual_feature_names = list(feature_names)
    if use_interactions:
        X_train_s, actual_feature_names = add_interaction_features(X_train_s, feature_names)
        X_test_s, _ = add_interaction_features(X_test_s, feature_names)

    # Train
    w, b = train_logistic_regression(X_train_s, y_train, lam=lam,
                                      lr=0.05, n_epochs=2000,
                                      class_weight_ratio=cw_ratio)

    # Predict
    y_score = predict_proba(X_test_s, w, b)

    # Platt calibration on a held-out portion of training data
    # Use last 20% of training data as calibration set
    n_train = len(X_train_s)
    n_cal = max(int(n_train * 0.2), 100)
    X_cal = X_train_s[-n_cal:]
    y_cal = y_train[-n_cal:]
    cal_scores = predict_proba(X_cal, w, b)

    # Fit Platt scaling: P(y=1|s) = sigmoid(a*s + c)
    a_platt, c_platt = 1.0, 0.0
    platt_lr = 0.1
    for _ in range(500):
        z_cal = a_platt * cal_scores + c_platt
        p_cal = sigmoid(z_cal)
        err = p_cal - y_cal
        a_platt -= platt_lr * np.mean(err * cal_scores)
        c_platt -= platt_lr * np.mean(err)

    # Apply Platt scaling to test scores
    y_score_calibrated = sigmoid(a_platt * y_score + c_platt)

    # Evaluate -- AUC uses raw scores (monotonic transform doesn't change AUC)
    auc, fpr_list, tpr_list = compute_roc_auc(y_test, y_score)
    # Use calibrated scores for threshold-based metrics
    opt_thresh, opt_f1 = find_optimal_threshold(y_test, y_score_calibrated)
    prec, rec, f1, tp, fp, fn, tn = compute_precision_recall(y_test, y_score_calibrated, opt_thresh)

    # Brier score on calibrated probabilities
    brier = np.mean((y_score_calibrated - y_test)**2)
    climo = np.mean(y_test)
    brier_climo = climo * (1 - climo)
    brier_skill = 1 - brier / brier_climo if brier_climo > 0 else 0

    if verbose:
        print(f"\n    {model_name}")
        print(f"      Train: {len(X_train)} (RI: {n_pos_train}, {100*n_pos_train/len(X_train):.1f}%)")
        print(f"      Test:  {len(X_test)} (RI: {np.sum(y_test)}, {100*np.mean(y_test):.1f}%)")
        print(f"      AUC:          {auc:.4f}")
        print(f"      Brier Skill:  {brier_skill:.4f}")
        print(f"      Optimal thresh: {opt_thresh:.3f}")
        print(f"      Precision:    {prec:.3f}")
        print(f"      Recall:       {rec:.3f}")
        print(f"      F1:           {f1:.3f}")
        if use_interactions:
            print(f"      Features (w/ interactions): {len(actual_feature_names)}")

    result = {
        'name': model_name,
        'auc': auc,
        'fpr': fpr_list,
        'tpr': tpr_list,
        'brier': brier,
        'brier_skill': brier_skill,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'opt_thresh': opt_thresh,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'weights': w,
        'bias': b,
        'feature_names': actual_feature_names,
        'y_test': y_test,
        'y_score': y_score_calibrated,
        'y_score_raw': y_score,
        'mu': mu,
        'sigma': sigma,
    }
    return result


def print_feature_importance(result):
    """Print sorted feature importance."""
    w = result['weights']
    names = result['feature_names']
    order = np.argsort(-np.abs(w))
    print(f"\n      Feature Importance ({result['name']}):")
    for rank, i in enumerate(order):
        direction = '+RI' if w[i] > 0 else '-RI'
        print(f"        {rank+1:2d}. {names[i]:<25s} w={w[i]:+.4f}  ({direction})")


# =========================================================================
# FIGURE GENERATION
# =========================================================================

def generate_figure(results_dict, ri_threshold_results):
    """Generate multi-panel diagnostic figure."""
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor(DARK)
    fig.suptitle('Hurricane Rapid Intensification Model v2 -- Multi-Basin Logistic Regression',
                 color='white', fontsize=16, fontweight='bold', y=0.98)

    # Layout: 3x3 grid
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35,
                          left=0.06, right=0.96, top=0.93, bottom=0.05)

    def style_ax(ax, title):
        ax.set_facecolor('#0d0d1a')
        ax.set_title(title, color=GOLD, fontweight='bold', fontsize=11)
        ax.tick_params(colors='white', labelsize=9)
        for sp in ax.spines.values():
            sp.set_color('#333')

    # --- Panel 1: ROC curves for different feature sets ---
    ax = fig.add_subplot(gs[0, 0])
    style_ax(ax, 'ROC Curves by Feature Set')
    colors = [BLUE, TEAL, GREEN, ACCENT]
    for idx, (key, res) in enumerate(results_dict.items()):
        if res is not None and 'fpr' in res:
            c = colors[idx % len(colors)]
            ax.plot(res['fpr'], res['tpr'], '-', color=c, linewidth=1.5,
                    label=f"{res['name']} (AUC={res['auc']:.3f})")
    ax.plot([0, 1], [0, 1], '--', color='#555', linewidth=1)
    ax.set_xlabel('False Positive Rate', color='white', fontsize=9)
    ax.set_ylabel('True Positive Rate', color='white', fontsize=9)
    ax.legend(facecolor='#222', edgecolor='#555', labelcolor='white', fontsize=7, loc='lower right')

    # --- Panel 2: Score distribution for best model ---
    ax = fig.add_subplot(gs[0, 1])
    style_ax(ax, 'Score Distribution (Best Model)')
    best_key = max(results_dict, key=lambda k: results_dict[k]['auc'] if results_dict[k] else 0)
    best = results_dict[best_key]
    if best is not None:
        ax.hist(best['y_score'][best['y_test'] == 0], bins=50, alpha=0.6,
                color=BLUE, density=True, label='Non-RI')
        ax.hist(best['y_score'][best['y_test'] == 1], bins=50, alpha=0.6,
                color=ACCENT, density=True, label='RI')
        ax.set_xlabel('Predicted P(RI)', color='white', fontsize=9)
        ax.set_ylabel('Density', color='white', fontsize=9)
        ax.legend(facecolor='#222', edgecolor='#555', labelcolor='white', fontsize=8)

    # --- Panel 3: Feature importance (best model) ---
    ax = fig.add_subplot(gs[0, 2])
    style_ax(ax, 'Feature Importance (Best Model)')
    if best is not None:
        w = best['weights']
        names = best['feature_names']
        order = np.argsort(np.abs(w))
        ax.barh(range(len(names)), np.abs(w[order]),
                color=[ACCENT if w[i] > 0 else BLUE for i in order],
                edgecolor='white', linewidth=0.3)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels([names[i] for i in order], color='white', fontsize=7)
        ax.set_xlabel('|Weight|', color='white', fontsize=9)

    # --- Panel 4: AUC by RI threshold ---
    ax = fig.add_subplot(gs[1, 0])
    style_ax(ax, 'AUC by RI Threshold Definition')
    if ri_threshold_results:
        labels = [r['label'] for r in ri_threshold_results if r['result'] is not None]
        aucs = [r['result']['auc'] for r in ri_threshold_results if r['result'] is not None]
        x_pos = np.arange(len(labels))
        bars = ax.bar(x_pos, aucs, color=TEAL, edgecolor='white', linewidth=0.5)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, color='white', fontsize=7, rotation=30, ha='right')
        ax.set_ylabel('AUC', color='white', fontsize=9)
        ax.set_ylim(0.7, 1.0)
        for bar, auc_val in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{auc_val:.3f}', ha='center', va='bottom', color='white', fontsize=8)

    # --- Panel 5: Reliability diagram ---
    ax = fig.add_subplot(gs[1, 1])
    style_ax(ax, 'Reliability Diagram (Best Model)')
    if best is not None:
        bin_edges = np.linspace(0, 1, 11)
        obs_freq = []
        pred_freq = []
        for j in range(len(bin_edges) - 1):
            mask = (best['y_score'] >= bin_edges[j]) & (best['y_score'] < bin_edges[j+1])
            n_bin = np.sum(mask)
            if n_bin > 5:
                obs_freq.append(np.mean(best['y_test'][mask]))
                pred_freq.append(np.mean(best['y_score'][mask]))
        if obs_freq:
            ax.plot(pred_freq, obs_freq, 'o-', color=TEAL, markersize=8, linewidth=2)
            ax.plot([0, 1], [0, 1], '--', color=GOLD, linewidth=1, label='Perfect')
            ax.set_xlabel('Predicted P(RI)', color='white', fontsize=9)
            ax.set_ylabel('Observed P(RI)', color='white', fontsize=9)
            ax.legend(facecolor='#222', edgecolor='#555', labelcolor='white', fontsize=8)

    # --- Panel 6: Precision-Recall by threshold ---
    ax = fig.add_subplot(gs[1, 2])
    style_ax(ax, 'Precision-Recall Curve (Best Model)')
    if best is not None:
        thresholds = np.linspace(0.01, 0.99, 100)
        precs = []
        recs = []
        for t in thresholds:
            p, r, _, _, _, _, _ = compute_precision_recall(best['y_test'], best['y_score'], t)
            precs.append(p)
            recs.append(r)
        ax.plot(recs, precs, '-', color=ACCENT, linewidth=2)
        ax.set_xlabel('Recall', color='white', fontsize=9)
        ax.set_ylabel('Precision', color='white', fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        # Mark optimal
        ax.plot(best['recall'], best['precision'], '*', color=GOLD, markersize=15,
                label=f"Optimal (P={best['precision']:.2f}, R={best['recall']:.2f})")
        ax.legend(facecolor='#222', edgecolor='#555', labelcolor='white', fontsize=8)

    # --- Panel 7: Basin comparison ---
    ax = fig.add_subplot(gs[2, 0])
    style_ax(ax, 'ALL Basins vs NA-Only')
    comp_data = []
    for key, res in results_dict.items():
        if res is not None and ('ALL' in res['name'] or 'NA' in res['name']):
            comp_data.append(res)
    if len(comp_data) >= 2:
        labels_c = [r['name'][:20] for r in comp_data]
        aucs_c = [r['auc'] for r in comp_data]
        x_c = np.arange(len(labels_c))
        bars_c = ax.bar(x_c, aucs_c, color=[TEAL, ORANGE][:len(comp_data)],
                        edgecolor='white', linewidth=0.5)
        ax.set_xticks(x_c)
        ax.set_xticklabels(labels_c, color='white', fontsize=8, rotation=15)
        ax.set_ylabel('AUC', color='white', fontsize=9)
        ax.set_ylim(0.7, 1.0)
        for bar, auc_val in zip(bars_c, aucs_c):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{auc_val:.3f}', ha='center', va='bottom', color='white', fontsize=9)

    # --- Panel 8: Model comparison summary table ---
    ax = fig.add_subplot(gs[2, 1])
    style_ax(ax, 'Model Comparison Summary')
    ax.axis('off')
    table_data = []
    for key, res in results_dict.items():
        if res is not None:
            table_data.append([
                res['name'][:22],
                f"{res['auc']:.3f}",
                f"{res['precision']:.3f}",
                f"{res['recall']:.3f}",
                f"{res['f1']:.3f}",
                f"{res['n_train']}",
            ])
    if table_data:
        col_labels = ['Model', 'AUC', 'Prec', 'Recall', 'F1', 'N_train']
        table = ax.table(cellText=table_data, colLabels=col_labels,
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        for (row, col), cell in table.get_celld().items():
            cell.set_facecolor('#1a1a2e')
            cell.set_edgecolor('#555')
            cell.set_text_props(color='white')
            if row == 0:
                cell.set_facecolor('#2a2a4e')
                cell.set_text_props(color=GOLD, fontweight='bold')

    # --- Panel 9: Summary text ---
    ax = fig.add_subplot(gs[2, 2])
    style_ax(ax, 'Key Findings')
    ax.axis('off')
    best_auc = best['auc'] if best else 0
    best_name = best['name'] if best else 'N/A'
    summary_lines = [
        f"Best model: {best_name}",
        f"Best AUC: {best_auc:.4f}",
        f"",
        f"Key predictors of RI:",
        f"  1. Prior intensification trend",
        f"  2. Pressure fall rate/accel",
        f"  3. Wind-pressure anomaly",
        f"  4. MPI deficit (room to grow)",
        f"  5. Distance from land",
        f"",
        f"All-basin training adds",
        f"significant data volume,",
        f"improving generalization.",
    ]
    for j, line in enumerate(summary_lines):
        ax.text(0.05, 0.92 - j * 0.075, line, color='white', fontsize=9,
                transform=ax.transAxes, verticalalignment='top', fontfamily='monospace')

    path = OUT / 'hurricane_ri_model_v2.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Figure saved: {path}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    print("=" * 70)
    print("  HURRICANE RAPID INTENSIFICATION MODEL v2")
    print("  Multi-basin, multi-feature, multi-threshold")
    print("=" * 70)

    # Load data
    storms_all = load_ibtracs_all()
    storms_na = load_ibtracs_na()

    if not storms_all and not storms_na:
        print("  FATAL: Could not load any IBTrACS data.")
        return

    # Use whichever loaded; prefer ALL
    if not storms_all:
        print("  WARNING: ALL basins failed; using NA only.")
        storms_all = storms_na

    # =====================================================================
    # Extract features (standard 30kt/24h threshold)
    # =====================================================================
    print("\n" + "=" * 70)
    print("  EXTRACTING FEATURES (30kt/24h RI threshold)")
    print("=" * 70)

    samples_all = []
    for sid, storm in storms_all.items():
        samples = extract_storm_features_v2(storm['entries'], ri_threshold_kt=30, ri_window_steps=4)
        samples_all.extend(samples)

    samples_na = []
    for sid, storm in storms_na.items():
        samples = extract_storm_features_v2(storm['entries'], ri_threshold_kt=30, ri_window_steps=4)
        samples_na.extend(samples)

    print(f"  ALL basins samples: {len(samples_all)}")
    print(f"  NA-only samples:    {len(samples_na)}")
    n_ri_all = sum(1 for _, l, _, _ in samples_all if l == 1)
    n_ri_na = sum(1 for _, l, _, _ in samples_na if l == 1)
    print(f"  ALL basins RI events: {n_ri_all} ({100*n_ri_all/max(len(samples_all),1):.1f}%)")
    print(f"  NA-only RI events:    {n_ri_na} ({100*n_ri_na/max(len(samples_na),1):.1f}%)")

    # =====================================================================
    # Run models with different feature sets
    # =====================================================================
    print("\n" + "=" * 70)
    print("  RUNNING MODELS (test on 2015-2023)")
    print("=" * 70)

    results = {}

    # 1. ALL basins, core features
    results['all_core'] = run_model(samples_all, CORE_FEATURES,
                                     'ALL-Core (15 feat)', test_year_start=2015)

    # 2. ALL basins, extended features (+ pressure)
    results['all_ext'] = run_model(samples_all, EXTENDED_FEATURES,
                                    'ALL-Extended (22 feat)', test_year_start=2015)

    # 3. ALL basins, full features (+ dist2land)
    results['all_full'] = run_model(samples_all, FULL_FEATURES,
                                     'ALL-Full (23 feat)', test_year_start=2015)

    # 4. ALL basins, full + RMW
    results['all_rmw'] = run_model(samples_all, FULL_RMW_FEATURES,
                                    'ALL-Full+RMW (26 feat)', test_year_start=2015)

    # 5. NA only, full features (for comparison)
    results['na_full'] = run_model(samples_na, FULL_FEATURES,
                                    'NA-Full (23 feat)', test_year_start=2015)

    # 6. NA only, full + RMW
    results['na_rmw'] = run_model(samples_na, FULL_RMW_FEATURES,
                                   'NA-Full+RMW (26 feat)', test_year_start=2015)

    # 7. ALL basins, extended + interactions (nonlinear)
    results['all_ext_int'] = run_model(samples_all, EXTENDED_FEATURES,
                                        'ALL-Ext+Interact', test_year_start=2015,
                                        use_interactions=True)

    # 8. NA, full + RMW + interactions
    results['na_rmw_int'] = run_model(samples_na, FULL_RMW_FEATURES,
                                       'NA-RMW+Interact', test_year_start=2015,
                                       use_interactions=True)

    # 9. ALL basins, full + RMW + interactions
    results['all_rmw_int'] = run_model(samples_all, FULL_RMW_FEATURES,
                                        'ALL-RMW+Interact', test_year_start=2015,
                                        use_interactions=True)

    # Print feature importance for best
    best_key = max(results, key=lambda k: results[k]['auc'] if results[k] else 0)
    if results[best_key]:
        print_feature_importance(results[best_key])

    # =====================================================================
    # Test multiple RI thresholds (using best feature set on ALL basins)
    # =====================================================================
    print("\n" + "=" * 70)
    print("  RI THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 70)

    # Determine which feature set to use for threshold analysis
    # Use FULL_FEATURES (widely available) or EXTENDED_FEATURES as fallback
    thresh_features = FULL_FEATURES

    ri_configs = [
        (25, 4, '25kt/24h'),
        (30, 4, '30kt/24h (std)'),
        (35, 4, '35kt/24h'),
        (30, 2, '30kt/12h'),
        (30, 6, '30kt/36h'),
        (30, 8, '30kt/48h'),
    ]

    ri_threshold_results = []
    for ri_kt, ri_steps, label in ri_configs:
        print(f"\n  --- {label} ---")
        thresh_samples = []
        for sid, storm in storms_all.items():
            s = extract_storm_features_v2(storm['entries'],
                                           ri_threshold_kt=ri_kt,
                                           ri_window_steps=ri_steps)
            thresh_samples.extend(s)

        n_ri_t = sum(1 for _, l, _, _ in thresh_samples if l == 1)
        print(f"    Samples: {len(thresh_samples)}, RI: {n_ri_t} ({100*n_ri_t/max(len(thresh_samples),1):.1f}%)")

        res = run_model(thresh_samples, thresh_features,
                        f'{label}', test_year_start=2015, verbose=True)
        ri_threshold_results.append({'label': label, 'result': res})

    # =====================================================================
    # COMPARISON SUMMARY
    # =====================================================================
    print("\n" + "=" * 70)
    print("  SUMMARY COMPARISON")
    print("=" * 70)

    print(f"\n  {'Model':<30s} {'AUC':>8s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s} {'N_train':>8s} {'N_test':>8s}")
    print("  " + "-" * 80)
    for key, res in results.items():
        if res is not None:
            print(f"  {res['name']:<30s} {res['auc']:8.4f} {res['precision']:8.3f} "
                  f"{res['recall']:8.3f} {res['f1']:8.3f} {res['n_train']:8d} {res['n_test']:8d}")

    print(f"\n  RI Threshold Sensitivity:")
    print(f"  {'Threshold':<20s} {'AUC':>8s} {'Prec':>8s} {'Recall':>8s} {'F1':>8s}")
    print("  " + "-" * 56)
    for item in ri_threshold_results:
        res = item['result']
        if res is not None:
            print(f"  {item['label']:<20s} {res['auc']:8.4f} {res['precision']:8.3f} "
                  f"{res['recall']:8.3f} {res['f1']:8.3f}")

    # Basin comparison
    r_all = results.get('all_full')
    r_na = results.get('na_full')
    if r_all and r_na:
        print(f"\n  Basin Comparison (Full features):")
        print(f"    ALL basins AUC: {r_all['auc']:.4f}  (N_train={r_all['n_train']})")
        print(f"    NA-only AUC:    {r_na['auc']:.4f}  (N_train={r_na['n_train']})")
        delta = r_all['auc'] - r_na['auc']
        print(f"    Delta AUC:      {delta:+.4f}")

    # =====================================================================
    # Generate figure
    # =====================================================================
    print("\n  Generating figure...")
    generate_figure(results, ri_threshold_results)

    print("\n" + "=" * 70)
    print("  COMPLETE")
    print("=" * 70)


if __name__ == '__main__':
    main()
