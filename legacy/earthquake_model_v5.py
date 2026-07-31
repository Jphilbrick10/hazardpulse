#!/usr/bin/env python3
"""
EARTHQUAKE PREDICTION MODEL v5
==============================
Target: AUC > 0.83 on M6+ test events (2015-2023)
Baseline: v4 AUC = 0.796

Key improvements over v4:
  1. Multi-window temporal acceleration (1m/3m/6m/12m rates + acceleration + jerk)
  2. Spatial clustering evolution (NN distances, centroid migration, concentration)
  3. Magnitude distribution shape (b-trend, mag gap, entropy, M5+ fraction)
  4. Interevent time features (CV, burstiness, quiet period ratio)
  5. Depth distribution changes (shallowing, subduction flag)
  6. Depth-2 gradient boosted trees (not stumps) for interaction capture
  7. 3-model stacked ensemble with cross-validated meta-learner
  8. 40+ features, all as CHANGE from background, same-zone controls
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import urllib.request
import ssl
import csv
import io
import json
import warnings
import time as time_module
from datetime import datetime
from collections import defaultdict
import os
import pathlib as _pathlib
# Output/cache roots resolve from THIS repo and are overridable via
# $HAZARDPULSE_OUT / $HAZARDPULSE_CACHE. They used to be absolute paths on one
# workstation, which made this file unrunnable anywhere else and published that
# machine's layout -- plus the name of a PRIVATE sibling repository -- from a
# PUBLIC repo.
_REPO = _pathlib.Path(__file__).resolve().parents[1]
_OUT_ROOT = _pathlib.Path(os.environ.get('HAZARDPULSE_OUT', _REPO / 'figures' / 'hazards'))
_CACHE_ROOT = _pathlib.Path(os.environ.get('HAZARDPULSE_CACHE', _REPO / '.cache' / 'legacy'))


warnings.filterwarnings('ignore')

OUT = _OUT_ROOT / "coherence_field_results"
OUT.mkdir(parents=True, exist_ok=True)
CACHE_DIR = _OUT_ROOT / "evidence" / ".eq_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DARK = '#1a1a2e'
GOLD = '#D4AF37'
ACCENT = '#e94560'
TEAL = '#16a085'
GREEN = '#2ecc71'
BLUE = '#3498db'

# ─── Utility ──────────────────────────────────────────────────────────────────

def get_ssl_context():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

def fetch_url(url, timeout=90):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 EQ-Model-v5'})
        ctx = get_ssl_context()
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return resp.read()
    except Exception as e:
        print(f"    Fetch error: {e}")
        return None

def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

def compute_auc(y_true, y_score):
    """Manual AUC via sorting (Wilcoxon-Mann-Whitney)."""
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Efficient: sort-based
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    n_pos = len(pos)
    n_neg = len(neg)
    # Count concordant pairs
    cum_neg = 0
    auc_sum = 0.0
    for i in range(len(y_sorted)):
        if y_sorted[i] == 0:
            cum_neg += 1
        else:
            auc_sum += cum_neg
    # AUC = 1 - (sum of neg ranks above pos) / (n_pos * n_neg)
    return 1.0 - auc_sum / (n_pos * n_neg)

def safe_ratio(a, b, default=0.0):
    return a / b if abs(b) > 1e-10 else default

# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_global_catalog(start_year=2000, end_year=2024, minmag=4.0):
    """Load global M4+ catalog from USGS, year by year, with disk caching."""
    all_events = []
    for year in range(start_year, end_year):
        cache_file = CACHE_DIR / f"usgs_m{minmag}_{year}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                events = json.load(f)
            print(f"  {year}: {len(events)} events (cached)")
            all_events.extend(events)
            continue

        print(f"  Fetching {year}...", end=" ", flush=True)
        url = (f"https://earthquake.usgs.gov/fdsnws/event/1/query?"
               f"format=csv&starttime={year}-01-01&endtime={year+1}-01-01"
               f"&minmagnitude={minmag}&orderby=time&limit=20000")
        data = fetch_url(url)
        events = []
        if data:
            text = data.decode('utf-8', errors='replace')
            reader = csv.DictReader(io.StringIO(text))
            for row in reader:
                try:
                    t = row.get('time', '')
                    if not t:
                        continue
                    dt = datetime.fromisoformat(t.replace('Z', '+00:00'))
                    epoch_days = (dt - datetime(2000, 1, 1, tzinfo=dt.tzinfo)).total_seconds() / 86400.0
                    lat = float(row.get('latitude', 0))
                    lon = float(row.get('longitude', 0))
                    mag = float(row.get('mag', 0))
                    depth = float(row.get('depth', 10) or 10)
                    events.append([epoch_days, lat, lon, mag, depth])
                except:
                    continue
        print(f"{len(events)} events")
        with open(cache_file, 'w') as f:
            json.dump(events, f)
        all_events.extend(events)
        time_module.sleep(0.3)

    arr = np.array(all_events, dtype=np.float64)
    arr = arr[np.argsort(arr[:, 0])]
    print(f"\n  Total catalog: {len(arr)} events, M{minmag}+, {start_year}-{end_year-1}")
    return arr

# ─── Spatial Utilities ────────────────────────────────────────────────────────

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def assign_zones(catalog, cell_size=3.0):
    """Assign each event to a spatial zone using grid cells."""
    lat_cells = np.floor(catalog[:, 1] / cell_size).astype(int)
    lon_cells = np.floor(catalog[:, 2] / cell_size).astype(int)
    return lat_cells * 10000 + lon_cells

# ─── Feature Engineering ─────────────────────────────────────────────────────

def compute_b_value(mags, mc=4.0):
    m = mags[mags >= mc]
    if len(m) < 5:
        return 1.0
    return 1.0 / (np.mean(m) - mc + 0.05) * np.log10(np.e)

def compute_b_trend(mags, times, mc=4.0, n_windows=5):
    """Linear trend in b-value over time windows."""
    mask = mags >= mc
    m, t = mags[mask], times[mask]
    if len(m) < 20:
        return 0.0
    edges = np.linspace(t.min(), t.max(), n_windows + 1)
    b_vals = []
    t_centers = []
    for i in range(n_windows):
        idx = (t >= edges[i]) & (t < edges[i+1])
        if idx.sum() >= 5:
            b_vals.append(compute_b_value(m[idx], mc))
            t_centers.append((edges[i] + edges[i+1]) / 2)
    if len(b_vals) < 3:
        return 0.0
    tc = np.array(t_centers) - np.mean(t_centers)
    bv = np.array(b_vals) - np.mean(b_vals)
    denom = np.sum(tc**2)
    if denom < 1e-10:
        return 0.0
    return np.sum(tc * bv) / denom

def compute_nn_mean(lats, lons, max_n=150):
    """Mean nearest-neighbor distance."""
    n = len(lats)
    if n < 3:
        return 500.0
    if n > max_n:
        idx = np.random.choice(n, max_n, replace=False)
        lats, lons = lats[idx], lons[idx]
        n = max_n
    nn_min = np.full(n, np.inf)
    for i in range(n):
        d = haversine_km(lats[i], lons[i], np.delete(lats, i), np.delete(lons, i))
        if len(d) > 0:
            nn_min[i] = d.min()
    return np.median(nn_min[nn_min < np.inf])

def compute_centroid_migration(lats, lons, times):
    n = len(lats)
    if n < 6:
        return 0.0
    mid = n // 2
    c1_lat, c1_lon = lats[:mid].mean(), lons[:mid].mean()
    c2_lat, c2_lon = lats[mid:].mean(), lons[mid:].mean()
    dist = haversine_km(c1_lat, c1_lon, c2_lat, c2_lon)
    dt_years = (times[-1] - times[0]) / 365.25
    return safe_ratio(dist, dt_years)

def compute_spatial_concentration(lats, lons, n_grid=4):
    """Fraction of events in densest 25% of grid cells."""
    n = len(lats)
    if n < 5:
        return 0.25
    lat_bins = np.linspace(lats.min() - 0.01, lats.max() + 0.01, n_grid + 1)
    lon_bins = np.linspace(lons.min() - 0.01, lons.max() + 0.01, n_grid + 1)
    counts = np.zeros(n_grid * n_grid)
    for i in range(n_grid):
        for j in range(n_grid):
            mask = ((lats >= lat_bins[i]) & (lats < lat_bins[i+1]) &
                    (lons >= lon_bins[j]) & (lons < lon_bins[j+1]))
            counts[i * n_grid + j] = mask.sum()
    counts.sort()
    total = counts.sum()
    if total == 0:
        return 0.25
    top_quarter = int(max(1, len(counts) // 4))
    return counts[-top_quarter:].sum() / total

def compute_mag_entropy(mags, mc=4.0):
    m = mags[mags >= mc]
    if len(m) < 5:
        return 0.0
    bins = np.arange(mc, 9.0, 0.3)
    hist, _ = np.histogram(m, bins=bins)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return 0.0
    p = hist / hist.sum()
    return -np.sum(p * np.log2(p + 1e-12))

def extract_features(zone_events, t_center, window_days=365, bg_window_days=365*5):
    """Extract all features for a zone at time t_center."""
    f = {}

    t_end = t_center
    t_start = t_center - window_days
    bg_end = t_start
    bg_start = bg_end - bg_window_days

    fg_mask = (zone_events[:, 0] >= t_start) & (zone_events[:, 0] < t_end)
    bg_mask = (zone_events[:, 0] >= bg_start) & (zone_events[:, 0] < bg_end)

    fg = zone_events[fg_mask]
    bg = zone_events[bg_mask]

    if len(fg) < 5 or len(bg) < 10:
        return None

    fg_t, fg_lat, fg_lon, fg_mag, fg_dep = fg.T
    bg_t, bg_lat, bg_lon, bg_mag, bg_dep = bg.T

    fg_years = window_days / 365.25
    bg_years = bg_window_days / 365.25

    fg_rate = len(fg) / fg_years
    bg_rate = len(bg) / bg_years
    if bg_rate < 1.0:  # Too sparse for meaningful features
        return None

    # ══════════════════════════════════════════════════════════════════════
    # 1. MULTI-WINDOW RATE FEATURES (key new addition)
    # ══════════════════════════════════════════════════════════════════════
    windows = [30, 90, 180, 365]
    rates = []
    for w in windows:
        m = (zone_events[:, 0] >= t_center - w) & (zone_events[:, 0] < t_center)
        rates.append(m.sum() / (w / 365.25))

    # Normalize all to background rate
    nr = [r / bg_rate for r in rates]
    f['rate_1m'] = nr[0] - 1.0
    f['rate_3m'] = nr[1] - 1.0
    f['rate_6m'] = nr[2] - 1.0
    f['rate_12m'] = nr[3] - 1.0

    # Acceleration: is short-term rate increasing relative to longer-term?
    f['accel_1m_vs_12m'] = safe_ratio(nr[0], max(nr[3], 0.01)) - 1.0
    f['accel_3m_vs_12m'] = safe_ratio(nr[1], max(nr[3], 0.01)) - 1.0
    f['accel_1m_vs_6m'] = safe_ratio(nr[0], max(nr[2], 0.01)) - 1.0

    # Second derivative (acceleration of acceleration)
    d1 = nr[1] - nr[3]  # 3m rate change vs 12m
    d2 = nr[0] - nr[1]  # 1m rate change vs 3m
    f['rate_accel2'] = d2 - d1  # is acceleration itself accelerating?

    # Jerk: third derivative proxy
    # Use 4 rate windows to compute discrete 3rd derivative
    if len(nr) == 4:
        dd1 = nr[0] - 2*nr[1] + nr[2]
        dd2 = nr[1] - 2*nr[2] + nr[3]
        f['rate_jerk'] = dd1 - dd2
    else:
        f['rate_jerk'] = 0.0

    # ══════════════════════════════════════════════════════════════════════
    # 2. B-VALUE FEATURES
    # ══════════════════════════════════════════════════════════════════════
    fg_b = compute_b_value(fg_mag)
    bg_b = compute_b_value(bg_mag)
    f['b_change'] = fg_b - bg_b
    f['b_trend'] = compute_b_trend(fg_mag, fg_t)

    # b-value in recent 3 months vs prior 9 months
    t_3m = t_center - 90
    recent_mag = fg_mag[fg_t >= t_3m]
    prior_mag = fg_mag[fg_t < t_3m]
    if len(recent_mag) >= 5 and len(prior_mag) >= 5:
        f['b_recent_change'] = compute_b_value(recent_mag) - compute_b_value(prior_mag)
    else:
        f['b_recent_change'] = 0.0

    # ══════════════════════════════════════════════════════════════════════
    # 3. MAGNITUDE DISTRIBUTION FEATURES
    # ══════════════════════════════════════════════════════════════════════
    f['mean_mag_change'] = fg_mag.mean() - bg_mag.mean()
    f['max_mag_anomaly'] = fg_mag.max() - bg_mag.max()
    f['mag_var_change'] = fg_mag.var() - bg_mag.var()

    # Magnitude gap
    sorted_fg = np.sort(fg_mag)[::-1]
    f['mag_gap'] = (sorted_fg[0] - sorted_fg[1]) if len(sorted_fg) >= 2 else 0.0

    # Magnitude entropy change
    f['mag_entropy_change'] = compute_mag_entropy(fg_mag) - compute_mag_entropy(bg_mag)

    # M5+ fraction change
    fg_m5 = (fg_mag >= 5.0).sum() / len(fg_mag)
    bg_m5 = (bg_mag >= 5.0).sum() / len(bg_mag)
    f['m5_frac_change'] = fg_m5 - bg_m5

    # M4.5+ fraction (intermediate)
    fg_m45 = (fg_mag >= 4.5).sum() / len(fg_mag)
    bg_m45 = (bg_mag >= 4.5).sum() / len(bg_mag)
    f['m45_frac_change'] = fg_m45 - bg_m45

    # ══════════════════════════════════════════════════════════════════════
    # 4. INTEREVENT TIME FEATURES
    # ══════════════════════════════════════════════════════════════════════
    fg_sorted = np.sort(fg_t)
    bg_sorted = np.sort(bg_t)
    fg_dt = np.diff(fg_sorted)
    bg_dt = np.diff(bg_sorted)
    fg_dt = fg_dt[fg_dt > 0]
    bg_dt = bg_dt[bg_dt > 0]

    if len(fg_dt) >= 3 and len(bg_dt) >= 3:
        fg_mu, fg_sig = fg_dt.mean(), fg_dt.std()
        bg_mu, bg_sig = bg_dt.mean(), bg_dt.std()

        fg_cv = safe_ratio(fg_sig, fg_mu, 1.0)
        bg_cv = safe_ratio(bg_sig, bg_mu, 1.0)
        f['cv_change'] = fg_cv - bg_cv

        fg_burst = safe_ratio(fg_sig - fg_mu, fg_sig + fg_mu, 0.0)
        bg_burst = safe_ratio(bg_sig - bg_mu, bg_sig + bg_mu, 0.0)
        f['burstiness_change'] = fg_burst - bg_burst

        f['max_quiet_ratio'] = safe_ratio(fg_dt.max(), bg_dt.max(), 1.0) - 1.0
        f['median_iet_ratio'] = safe_ratio(np.median(fg_dt), np.median(bg_dt), 1.0) - 1.0
    else:
        f['cv_change'] = 0.0
        f['burstiness_change'] = 0.0
        f['max_quiet_ratio'] = 0.0
        f['median_iet_ratio'] = 0.0

    # Recent acceleration: last 3 months vs prior 9 months event count
    recent = (fg_t >= t_3m).sum()
    prior = (fg_t < t_3m).sum()
    f['recent_3m_9m_ratio'] = safe_ratio(recent / (90/365.25), prior / max((window_days - 90)/365.25, 0.01)) - 1.0

    # ══════════════════════════════════════════════════════════════════════
    # 5. SPATIAL FEATURES
    # ══════════════════════════════════════════════════════════════════════
    fg_nn = compute_nn_mean(fg_lat, fg_lon)
    bg_nn = compute_nn_mean(bg_lat, bg_lon)
    f['nn_mean_ratio'] = safe_ratio(fg_nn, bg_nn, 1.0) - 1.0

    f['centroid_migration'] = compute_centroid_migration(fg_lat, fg_lon, fg_t)

    fg_conc = compute_spatial_concentration(fg_lat, fg_lon)
    bg_conc = compute_spatial_concentration(bg_lat, bg_lon)
    f['spatial_conc_change'] = fg_conc - bg_conc

    # Spatial extent change
    fg_ell = np.sqrt(fg_lat.std()**2 + fg_lon.std()**2) if len(fg_lat) > 1 else 0
    bg_ell = np.sqrt(bg_lat.std()**2 + bg_lon.std()**2) if len(bg_lat) > 1 else 0
    f['ell_change'] = fg_ell - bg_ell

    # ══════════════════════════════════════════════════════════════════════
    # 6. DEPTH FEATURES
    # ══════════════════════════════════════════════════════════════════════
    f['depth_mean_change'] = fg_dep.mean() - bg_dep.mean()
    f['depth_std_change'] = fg_dep.std() - bg_dep.std() if len(fg_dep) > 1 else 0.0
    f['is_subduction'] = 1.0 if (fg_dep > 70).sum() >= 3 else 0.0

    # Shallow fraction change (< 20 km)
    fg_shallow = (fg_dep < 20).sum() / len(fg_dep)
    bg_shallow = (bg_dep < 20).sum() / len(bg_dep)
    f['shallow_frac_change'] = fg_shallow - bg_shallow

    # ══════════════════════════════════════════════════════════════════════
    # 7. INTERACTION FEATURES (products of top raw features)
    # ══════════════════════════════════════════════════════════════════════
    f['rate_x_btrend'] = f['rate_1m'] * f['b_trend']
    f['accel_x_bchange'] = f['accel_1m_vs_12m'] * abs(f['b_change'])
    f['rate_x_conc'] = f['rate_1m'] * f['spatial_conc_change']
    f['accel_x_m5frac'] = f['accel_1m_vs_12m'] * f['m5_frac_change']

    return f

# ─── Dataset Construction ─────────────────────────────────────────────────────

def build_dataset(catalog, train_end_day):
    """Build positive (pre-M6+) and negative (same-zone control) samples."""
    print("\n=== Building Dataset ===")

    m6_mask = catalog[:, 3] >= 6.0
    m6_events = catalog[m6_mask]
    print(f"  Total M6+ events: {len(m6_events)}")

    zone_ids = assign_zones(catalog)
    m6_zone_ids = zone_ids[m6_mask]

    unique_zones = np.unique(zone_ids)
    zone_event_map = {}
    for z in unique_zones:
        mask = zone_ids == z
        zone_event_map[z] = catalog[mask]

    m6_times_per_zone = defaultdict(list)
    for i in range(len(m6_events)):
        m6_times_per_zone[m6_zone_ids[i]].append(m6_events[i, 0])

    feature_names = None
    X_list, y_list, t_list, mag_list = [], [], [], []

    # ── Positive samples ──
    print("  Extracting positive samples...", flush=True)
    skipped = 0
    for i in range(len(m6_events)):
        ev = m6_events[i]
        t_eq = ev[0]
        z = m6_zone_ids[i]
        zone_ev = zone_event_map.get(z)
        if zone_ev is None or len(zone_ev) < 30:
            skipped += 1
            continue
        if t_eq < 365.25 * 6:
            skipped += 1
            continue

        feats = extract_features(zone_ev, t_eq)
        if feats is None:
            skipped += 1
            continue

        if feature_names is None:
            feature_names = sorted(feats.keys())

        X_list.append([feats[fn] for fn in feature_names])
        y_list.append(1)
        t_list.append(t_eq)
        mag_list.append(ev[3])

        if len(X_list) % 500 == 0:
            print(f"    {len(X_list)} positive...", flush=True)

    n_pos = len(X_list)
    print(f"  Positive: {n_pos} (skipped {skipped})")

    # ── Negative samples (same-zone controls) ──
    print("  Extracting negative samples...", flush=True)
    neg_count = 0
    target_neg = n_pos * 3
    np.random.seed(42)
    m6_zones = list(m6_times_per_zone.keys())
    attempts = 0

    while neg_count < target_neg and attempts < target_neg * 15:
        attempts += 1
        z = m6_zones[np.random.randint(len(m6_zones))]
        zone_ev = zone_event_map.get(z)
        if zone_ev is None or len(zone_ev) < 30:
            continue

        t_min = zone_ev[:, 0].min() + 365.25 * 6
        t_max = zone_ev[:, 0].max()
        if t_max <= t_min:
            continue
        t_sample = np.random.uniform(t_min, t_max)

        # Require NO M6+ within 2 years after
        m6_ts = np.array(m6_times_per_zone[z])
        if np.any((m6_ts >= t_sample) & (m6_ts < t_sample + 730)):
            continue
        # Also require NO M6+ within 6 months before (avoid aftershock contamination)
        if np.any((m6_ts >= t_sample - 180) & (m6_ts < t_sample)):
            continue

        feats = extract_features(zone_ev, t_sample)
        if feats is None:
            continue

        X_list.append([feats[fn] for fn in feature_names])
        y_list.append(0)
        t_list.append(t_sample)
        mag_list.append(0.0)
        neg_count += 1

        if neg_count % 500 == 0:
            print(f"    {neg_count} negative...", flush=True)

    print(f"  Negative: {neg_count}")

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)
    t = np.array(t_list, dtype=np.float64)
    mags = np.array(mag_list, dtype=np.float64)

    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)

    # Clip extreme outliers (>5 sigma) to reduce noise
    for j in range(X.shape[1]):
        mu = np.median(X[:, j])
        mad = np.median(np.abs(X[:, j] - mu)) + 1e-8
        sigma_est = mad * 1.4826  # robust sigma
        X[:, j] = np.clip(X[:, j], mu - 6*sigma_est, mu + 6*sigma_est)

    train_mask = t < train_end_day
    test_mask = t >= train_end_day

    print(f"\n  Features: {len(feature_names)}")
    print(f"  Names: {feature_names}")
    print(f"  Train: {train_mask.sum()} ({(y[train_mask]==1).sum()} pos / {(y[train_mask]==0).sum()} neg)")
    print(f"  Test:  {test_mask.sum()} ({(y[test_mask]==1).sum()} pos / {(y[test_mask]==0).sum()} neg)")

    return (X[train_mask], y[train_mask], X[test_mask], y[test_mask],
            mags[test_mask], feature_names)

# ─── ML Models ────────────────────────────────────────────────────────────────

class LogisticRegression:
    """L2-regularized logistic regression via mini-batch gradient descent."""

    def __init__(self, lr=0.01, lam=1.0, n_iter=3000, batch_size=256):
        self.lr = lr
        self.lam = lam
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.w = None
        self.b = 0.0

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        bs = min(self.batch_size, n)

        for it in range(self.n_iter):
            idx = np.random.choice(n, bs, replace=False)
            Xi, yi = X[idx], y[idx]
            z = Xi @ self.w + self.b
            p = sigmoid(z)
            grad_w = (Xi.T @ (p - yi)) / bs + self.lam * self.w
            grad_b = (p - yi).mean()
            # Learning rate decay
            lr_t = self.lr / (1 + it * 0.0001)
            self.w -= lr_t * grad_w
            self.b -= lr_t * grad_b

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

class GBMDepth2:
    """Gradient boosted depth-2 trees for binary classification.
    Depth-2 trees capture pairwise feature interactions."""

    def __init__(self, n_trees=500, learning_rate=0.03, min_leaf=10, max_features=None):
        self.n_trees = n_trees
        self.learning_rate = learning_rate
        self.min_leaf = min_leaf
        self.max_features = max_features
        self.trees = []
        self.init_pred = 0.0

    def _find_best_split(self, X, residuals, feature_subset=None):
        """Find best feature and threshold for a split."""
        n, d = X.shape
        best_gain = -np.inf
        best_feat = 0
        best_thresh = 0.0

        features = feature_subset if feature_subset is not None else range(d)

        for j in features:
            col = X[:, j]
            thresholds = np.percentile(col, [10, 20, 30, 40, 50, 60, 70, 80, 90])
            thresholds = np.unique(thresholds)

            for thresh in thresholds:
                left = col <= thresh
                right = ~left
                nl = left.sum()
                nr_ = right.sum()
                if nl < self.min_leaf or nr_ < self.min_leaf:
                    continue
                gain = nl * residuals[left].mean()**2 + nr_ * residuals[right].mean()**2
                if gain > best_gain:
                    best_gain = gain
                    best_feat = j
                    best_thresh = thresh

        return best_feat, best_thresh, best_gain

    def _build_depth2_tree(self, X, residuals, feature_subset=None):
        """Build a depth-2 tree (root split + 2 child splits = up to 4 leaves)."""
        n = X.shape[0]

        # Root split
        feat1, thresh1, gain1 = self._find_best_split(X, residuals, feature_subset)
        if gain1 <= 0:
            return {'leaf': True, 'val': residuals.mean()}

        left_mask = X[:, feat1] <= thresh1
        right_mask = ~left_mask

        # Left child split
        left_node = self._build_leaf_or_split(X[left_mask], residuals[left_mask], feature_subset)
        right_node = self._build_leaf_or_split(X[right_mask], residuals[right_mask], feature_subset)

        return {
            'leaf': False,
            'feat': feat1, 'thresh': thresh1,
            'left': left_node, 'right': right_node
        }

    def _build_leaf_or_split(self, X, residuals, feature_subset):
        if len(residuals) < self.min_leaf * 2:
            return {'leaf': True, 'val': residuals.mean()}
        feat, thresh, gain = self._find_best_split(X, residuals, feature_subset)
        if gain <= 0:
            return {'leaf': True, 'val': residuals.mean()}
        left = X[:, feat] <= thresh
        right = ~left
        if left.sum() < self.min_leaf or right.sum() < self.min_leaf:
            return {'leaf': True, 'val': residuals.mean()}
        return {
            'leaf': False,
            'feat': feat, 'thresh': thresh,
            'left': {'leaf': True, 'val': residuals[left].mean()},
            'right': {'leaf': True, 'val': residuals[right].mean()}
        }

    def _predict_tree(self, tree, X):
        n = X.shape[0]
        preds = np.zeros(n)
        if tree['leaf']:
            return np.full(n, tree['val'])

        left = X[:, tree['feat']] <= tree['thresh']
        right = ~left

        if tree['left']['leaf']:
            preds[left] = tree['left']['val']
        else:
            l = tree['left']
            left_idx = np.where(left)[0]
            Xl = X[left]
            ll = Xl[:, l['feat']] <= l['thresh']
            preds[left_idx[ll]] = l['left']['val']
            preds[left_idx[~ll]] = l['right']['val']

        if tree['right']['leaf']:
            preds[right] = tree['right']['val']
        else:
            r = tree['right']
            right_idx = np.where(right)[0]
            Xr = X[right]
            rr = Xr[:, r['feat']] <= r['thresh']
            preds[right_idx[rr]] = r['left']['val']
            preds[right_idx[~rr]] = r['right']['val']

        return preds

    def fit(self, X, y):
        n, d = X.shape
        pos = y.sum()
        neg = n - pos
        self.init_pred = np.log(max(pos, 1) / max(neg, 1))
        F = np.full(n, self.init_pred)

        for i in range(self.n_trees):
            p = sigmoid(F)
            residuals = y - p

            if self.max_features and self.max_features < d:
                feat_subset = np.random.choice(d, self.max_features, replace=False)
            else:
                feat_subset = None

            tree = self._build_depth2_tree(X, residuals, feat_subset)
            self.trees.append(tree)
            F += self.learning_rate * self._predict_tree(tree, X)

    def predict_proba(self, X):
        n = X.shape[0]
        F = np.full(n, self.init_pred)
        for tree in self.trees:
            F += self.learning_rate * self._predict_tree(tree, X)
        return sigmoid(F)

    def feature_importance(self, n_features):
        imp = np.zeros(n_features)
        def _count(tree, weight=1.0):
            if tree['leaf']:
                return
            imp[tree['feat']] += weight
            _count(tree['left'], weight * 0.5)
            _count(tree['right'], weight * 0.5)
        for tree in self.trees:
            _count(tree)
        s = imp.sum()
        if s > 0:
            imp /= s
        return imp

class EnsembleModel:
    """Stack 3 models with cross-validated meta-learner."""

    def __init__(self):
        self.lr_model = LogisticRegression(lr=0.005, lam=0.3, n_iter=4000)
        self.gbm_model = GBMDepth2(n_trees=700, learning_rate=0.03, min_leaf=10)
        self.rs_gbm = GBMDepth2(n_trees=700, learning_rate=0.03, min_leaf=10)
        self.meta = LogisticRegression(lr=0.01, lam=0.1, n_iter=3000)
        self.mu = None
        self.sigma = None

    def fit(self, X_train, y_train):
        n, d = X_train.shape
        self.mu = X_train.mean(axis=0)
        self.sigma = X_train.std(axis=0) + 1e-8
        X_std = (X_train - self.mu) / self.sigma

        self.rs_gbm.max_features = max(d * 2 // 3, 8)

        print("  Training L2 logistic regression...", flush=True)
        self.lr_model.fit(X_std, y_train)
        print(f"    LR train AUC: {compute_auc(y_train, self.lr_model.predict_proba(X_std)):.4f}")

        print("  Training GBM depth-2 (700 trees)...", flush=True)
        self.gbm_model.fit(X_train, y_train)
        print(f"    GBM train AUC: {compute_auc(y_train, self.gbm_model.predict_proba(X_train)):.4f}")

        print("  Training Random-Subspace GBM...", flush=True)
        self.rs_gbm.fit(X_train, y_train)
        print(f"    RS-GBM train AUC: {compute_auc(y_train, self.rs_gbm.predict_proba(X_train)):.4f}")

        # Cross-validated meta features
        print("  Training meta-learner (5-fold CV stacking)...", flush=True)
        meta_X = self._cv_meta_features(X_train, y_train, n_folds=5)
        self.meta.fit(meta_X, y_train)
        print(f"    Meta train AUC: {compute_auc(y_train, self.meta.predict_proba(meta_X)):.4f}")

    def _cv_meta_features(self, X, y, n_folds=5):
        n = len(y)
        indices = np.arange(n)
        np.random.seed(99)
        np.random.shuffle(indices)
        fold_size = n // n_folds
        meta = np.zeros((n, 3))

        for fold in range(n_folds):
            vs = fold * fold_size
            ve = vs + fold_size if fold < n_folds - 1 else n
            val_idx = indices[vs:ve]
            tr_idx = np.concatenate([indices[:vs], indices[ve:]])
            Xtr, ytr = X[tr_idx], y[tr_idx]
            Xval = X[val_idx]
            Xtr_std = (Xtr - self.mu) / self.sigma
            Xval_std = (Xval - self.mu) / self.sigma

            lr = LogisticRegression(lr=0.005, lam=0.3, n_iter=2500)
            lr.fit(Xtr_std, ytr)
            meta[val_idx, 0] = lr.predict_proba(Xval_std)

            gbm = GBMDepth2(n_trees=500, learning_rate=0.03, min_leaf=10)
            gbm.fit(Xtr, ytr)
            meta[val_idx, 1] = gbm.predict_proba(Xval)

            d = X.shape[1]
            rs = GBMDepth2(n_trees=500, learning_rate=0.03, min_leaf=10,
                           max_features=max(d*2//3, 8))
            rs.fit(Xtr, ytr)
            meta[val_idx, 2] = rs.predict_proba(Xval)

        return meta

    def predict_proba(self, X):
        X_std = (X - self.mu) / self.sigma
        p1 = self.lr_model.predict_proba(X_std)
        p2 = self.gbm_model.predict_proba(X)
        p3 = self.rs_gbm.predict_proba(X)
        return self.meta.predict_proba(np.column_stack([p1, p2, p3]))

    def feature_importance(self, n_features):
        imp1 = self.gbm_model.feature_importance(n_features)
        imp2 = self.rs_gbm.feature_importance(n_features)
        return 0.5 * imp1 + 0.5 * imp2

# ─── Visualization ────────────────────────────────────────────────────────────

def plot_results(y_test, y_proba, mags_test, feature_names, importances, component_aucs):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor(DARK)
    for ax in axes.flat:
        ax.set_facecolor(DARK)
        for spine in ax.spines.values():
            spine.set_color('#444')
        ax.tick_params(colors='white')

    auc_val = compute_auc(y_test, y_proba)

    # 1. ROC
    ax = axes[0, 0]
    thresholds = np.linspace(0, 1, 300)
    tprs, fprs = [], []
    for th in thresholds:
        pred = (y_proba >= th).astype(int)
        tp = ((pred == 1) & (y_test == 1)).sum()
        fp = ((pred == 1) & (y_test == 0)).sum()
        fn = ((pred == 0) & (y_test == 1)).sum()
        tn = ((pred == 0) & (y_test == 0)).sum()
        tprs.append(tp / max(tp + fn, 1))
        fprs.append(fp / max(fp + tn, 1))
    ax.plot(fprs, tprs, color=GOLD, linewidth=2.5, label=f'v5 Ensemble AUC={auc_val:.3f}')
    ax.plot([0, 1], [0, 1], '--', color='#666')
    ax.set_xlabel('False Positive Rate', color='white')
    ax.set_ylabel('True Positive Rate', color='white')
    ax.set_title('ROC Curve — M6+ Test', color=GOLD, fontsize=13, fontweight='bold')
    ax.legend(facecolor='#222', edgecolor=GOLD, labelcolor='white')

    # 2. AUC by mag threshold
    ax = axes[0, 1]
    mag_thresholds = [6.0, 6.5, 7.0, 7.5]
    aucs_by_mag, counts = [], []
    for mt in mag_thresholds:
        pos_m = (y_test == 1) & (mags_test >= mt)
        neg_m = (y_test == 0)
        comb = pos_m | neg_m
        c = pos_m.sum()
        counts.append(c)
        aucs_by_mag.append(compute_auc(y_test[comb], y_proba[comb]) if c >= 5 else 0.5)

    bars = ax.bar([f'M{mt}+\n(n={c})' for mt, c in zip(mag_thresholds, counts)],
                  aucs_by_mag, color=[GOLD, TEAL, GREEN, ACCENT], alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, aucs_by_mag):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', color='white', fontsize=11, fontweight='bold')
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.796, color=ACCENT, linestyle=':', linewidth=1.5, label='v4 (0.796)')
    ax.set_ylabel('AUC', color='white')
    ax.set_title('AUC by Magnitude', color=GOLD, fontsize=13, fontweight='bold')
    ax.legend(facecolor='#222', edgecolor=GOLD, labelcolor='white', fontsize=9)

    # 3. Feature importance top 15
    ax = axes[0, 2]
    idx = np.argsort(importances)[::-1][:15]
    ax.barh(range(len(idx)), importances[idx][::-1], color=TEAL, alpha=0.85, edgecolor='white')
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx][::-1], color='white', fontsize=9)
    ax.set_xlabel('Importance', color='white')
    ax.set_title('Top 15 Features', color=GOLD, fontsize=13, fontweight='bold')

    # 4. Score distribution
    ax = axes[1, 0]
    bins = np.linspace(0, 1, 40)
    ax.hist(y_proba[y_test == 0], bins=bins, alpha=0.6, color=BLUE, label='No M6+', density=True)
    ax.hist(y_proba[y_test == 1], bins=bins, alpha=0.6, color=ACCENT, label='Pre-M6+', density=True)
    ax.set_xlabel('Predicted Probability', color='white')
    ax.set_ylabel('Density', color='white')
    ax.set_title('Score Distributions', color=GOLD, fontsize=13, fontweight='bold')
    ax.legend(facecolor='#222', edgecolor=GOLD, labelcolor='white')

    # 5. Component AUCs
    ax = axes[1, 1]
    names = list(component_aucs.keys())
    vals = list(component_aucs.values())
    bars = ax.bar(names, vals, color=[BLUE, TEAL, GREEN, GOLD], alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', color='white', fontsize=11, fontweight='bold')
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.796, color=ACCENT, linestyle=':', linewidth=1.5, label='v4 (0.796)')
    ax.set_ylabel('Test AUC', color='white')
    ax.set_title('Component Models', color=GOLD, fontsize=13, fontweight='bold')
    ax.legend(facecolor='#222', edgecolor=GOLD, labelcolor='white', fontsize=9)

    # 6. Detection vs False Alarm
    ax = axes[1, 2]
    far_targets = [0.01, 0.05, 0.10, 0.20, 0.30]
    det_rates = []
    for ft in far_targets:
        best_t, best_d = 0.5, 0.0
        for th in np.linspace(0, 1, 500):
            pred = (y_proba >= th).astype(int)
            fp = ((pred == 1) & (y_test == 0)).sum()
            tn = ((pred == 0) & (y_test == 0)).sum()
            far = fp / max(fp + tn, 1)
            if abs(far - ft) < 0.015:
                tp = ((pred == 1) & (y_test == 1)).sum()
                fn = ((pred == 0) & (y_test == 1)).sum()
                dr = tp / max(tp + fn, 1)
                if dr > best_d:
                    best_d = dr
                    best_t = th
        det_rates.append(best_d)

    ax.plot([f*100 for f in far_targets], [d*100 for d in det_rates],
            'o-', color=GOLD, markersize=8, linewidth=2)
    for f, d in zip(far_targets, det_rates):
        ax.annotate(f'{d*100:.0f}%', (f*100, d*100+2), color='white', fontsize=10, ha='center')
    ax.set_xlabel('False Alarm Rate (%)', color='white')
    ax.set_ylabel('Detection Rate (%)', color='white')
    ax.set_title('Detection vs False Alarm', color=GOLD, fontsize=13, fontweight='bold')
    ax.set_xlim(-1, 35)
    ax.set_ylim(0, 105)

    fig.suptitle('Earthquake Prediction Model v5 — Depth-2 GBM Ensemble + Temporal Acceleration',
                 color=GOLD, fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = OUT / 'earthquake_model_v5.png'
    plt.savefig(path, dpi=150, facecolor=DARK, bbox_inches='tight')
    plt.close()
    print(f"\n  Figure saved: {path}")

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("EARTHQUAKE PREDICTION MODEL v5")
    print("Depth-2 GBM Ensemble + Temporal Acceleration + Spatial Clustering")
    print("=" * 70)

    print("\n=== Loading Global M4+ Catalog ===")
    catalog = load_global_catalog(2000, 2024, minmag=4.0)
    print(f"  Shape: {catalog.shape}")
    print(f"  M6+: {(catalog[:, 3] >= 6.0).sum()}")
    print(f"  M7+: {(catalog[:, 3] >= 7.0).sum()}")

    train_end_day = (datetime(2015, 1, 1) - datetime(2000, 1, 1)).days

    X_train, y_train, X_test, y_test, mags_test, fnames = \
        build_dataset(catalog, train_end_day)

    if len(X_train) < 50 or len(X_test) < 20:
        print("ERROR: Not enough samples.")
        return

    # ── Train ──
    print("\n=== Training Ensemble ===")
    model = EnsembleModel()
    model.fit(X_train, y_train)

    # ── Evaluate ──
    print("\n=== Test Evaluation ===")
    y_proba = model.predict_proba(X_test)

    X_test_std = (X_test - model.mu) / model.sigma
    lr_auc = compute_auc(y_test, model.lr_model.predict_proba(X_test_std))
    gbm_auc = compute_auc(y_test, model.gbm_model.predict_proba(X_test))
    rs_auc = compute_auc(y_test, model.rs_gbm.predict_proba(X_test))
    ens_auc = compute_auc(y_test, y_proba)

    comp_aucs = {'L2-LR': lr_auc, 'GBM-D2': gbm_auc, 'RS-GBM': rs_auc, 'Ensemble': ens_auc}

    print(f"\n  L2 Logistic Regression: {lr_auc:.4f}")
    print(f"  GBM depth-2 (700):     {gbm_auc:.4f}")
    print(f"  Random-Subspace GBM:    {rs_auc:.4f}")
    print(f"  ENSEMBLE:               {ens_auc:.4f}")
    print(f"\n  v4 baseline:            0.7960")
    print(f"  Improvement:            {(ens_auc - 0.796)*100:+.2f} pp")

    print("\n  AUC by magnitude:")
    for mt in [6.0, 6.5, 7.0, 7.5]:
        pm = (y_test == 1) & (mags_test >= mt)
        nm = (y_test == 0)
        comb = pm | nm
        np_ = pm.sum()
        if np_ >= 5:
            a = compute_auc(y_test[comb], y_proba[comb])
            print(f"    M{mt}+ : AUC={a:.4f} (n={np_})")
        else:
            print(f"    M{mt}+ : too few (n={np_})")

    imp = model.feature_importance(len(fnames))
    print("\n  Top 15 features:")
    idx = np.argsort(imp)[::-1][:15]
    for r, i in enumerate(idx):
        print(f"    {r+1:2d}. {fnames[i]:30s} {imp[i]:.4f}")

    print(f"\n  Test: {int((y_test==1).sum())} pos, {int((y_test==0).sum())} neg")

    print("\n=== Generating Figure ===")
    plot_results(y_test, y_proba, mags_test, fnames, imp, comp_aucs)

    print("\n" + "=" * 70)
    print(f"MODEL v5 FINAL AUC:  {ens_auc:.4f}")
    print(f"v4 baseline:         0.7960")
    print(f"Improvement:         {(ens_auc - 0.796)*100:+.2f} pp")
    print("=" * 70)

if __name__ == "__main__":
    main()
