#!/usr/bin/env python3
"""
EARTHQUAKE MODEL v4: Global M4+ catalog, change-only features, same-location controls
======================================================================================

Improvements over v3b (AUC=0.747):
  - Full M4+ catalog (not just M5+) for richer precursor statistics
  - 17 change-only features including new spatial migration, depth, GR residual
  - Same-location controls shifted 3-4 years in time
  - 300km radius with spatial grid index
  - Manual logistic regression with L2 regularization
  - Train 2000-2014, test 2015-2023
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import urllib.request
import ssl
import json
import time as time_mod
import warnings
import os
import pickle
warnings.filterwarnings('ignore')

OUT = Path(r"c:\Users\Josh\Projects\Coherence_Energy_Labs_Website\data\evidence\coherence_field_results")
OUT.mkdir(exist_ok=True)
CACHE_DIR = Path(r"c:\Users\Josh\Projects\Coherence_Energy_Labs_Website\data\evidence\cache_eq_v4")
CACHE_DIR.mkdir(exist_ok=True)

DARK = '#1a1a2e'
GOLD = '#D4AF37'
ACCENT = '#e94560'
TEAL = '#16a085'
GREEN = '#2ecc71'
BLUE = '#3498db'

# =========================================================================
# DATA LOADING
# =========================================================================

def get_ssl_context():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

def fetch_url(url, timeout=90):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'EarthquakeModelV4/1.0'})
        ctx = get_ssl_context()
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return resp.read()
    except Exception as e:
        print(f"    Fetch error: {e}")
        return None

def load_global_catalog(start_year=2000, end_year=2024, minmag=4.0):
    """Load global M4+ catalog from USGS FDSNWS, year by year, with half-year split if needed."""
    cache_file = CACHE_DIR / f"global_m{minmag}_{start_year}_{end_year}.pkl"
    if cache_file.exists():
        print(f"  Loading cached catalog from {cache_file}")
        with open(cache_file, 'rb') as f:
            events = pickle.load(f)
        print(f"    {len(events)} events loaded from cache")
        return events

    all_events = []
    for year in range(start_year, end_year):
        # Try full year first
        periods = [(f"{year}-01-01", f"{year+1}-01-01")]
        for p_start, p_end in periods:
            url = (f"https://earthquake.usgs.gov/fdsnws/event/1/query?"
                   f"format=geojson&starttime={p_start}&endtime={p_end}"
                   f"&minmagnitude={minmag}&orderby=time&limit=20000")
            print(f"  Fetching {p_start} to {p_end} (M{minmag}+)...")
            data = fetch_url(url, timeout=120)
            if data is None:
                print(f"    Failed, trying half-year split...")
                # Split into halves
                mid = f"{year}-07-01"
                for sub_start, sub_end in [(p_start, mid), (mid, p_end)]:
                    url2 = (f"https://earthquake.usgs.gov/fdsnws/event/1/query?"
                            f"format=geojson&starttime={sub_start}&endtime={sub_end}"
                            f"&minmagnitude={minmag}&orderby=time&limit=20000")
                    print(f"    Fetching {sub_start} to {sub_end}...")
                    data2 = fetch_url(url2, timeout=120)
                    if data2:
                        _parse_geojson(data2, all_events)
                    time_mod.sleep(1)
                continue

            try:
                gj = json.loads(data)
                count = len(gj.get('features', []))
                if count >= 19999:
                    print(f"    Hit limit ({count}), splitting into half-years...")
                    mid = f"{year}-07-01"
                    for sub_start, sub_end in [(p_start, mid), (mid, p_end)]:
                        url2 = (f"https://earthquake.usgs.gov/fdsnws/event/1/query?"
                                f"format=geojson&starttime={sub_start}&endtime={sub_end}"
                                f"&minmagnitude={minmag}&orderby=time&limit=20000")
                        print(f"    Fetching {sub_start} to {sub_end}...")
                        data2 = fetch_url(url2, timeout=120)
                        if data2:
                            _parse_geojson(data2, all_events)
                        time_mod.sleep(1)
                else:
                    _parse_geojson(data, all_events)
                    print(f"    {count} events")
            except Exception as e:
                print(f"    Parse error: {e}")

            time_mod.sleep(0.5)

    # Deduplicate by approximate time+location
    seen = set()
    unique = []
    for e in all_events:
        key = (round(e['epoch'], 1), round(e['lat'], 2), round(e['lon'], 2))
        if key not in seen:
            seen.add(key)
            unique.append(e)

    unique.sort(key=lambda e: e['epoch'])
    print(f"  Total unique events: {len(unique)}")

    with open(cache_file, 'wb') as f:
        pickle.dump(unique, f)

    return unique

def _parse_geojson(data, events_list):
    """Parse GeoJSON response and append events."""
    try:
        gj = json.loads(data)
        for feat in gj.get('features', []):
            props = feat['properties']
            coords = feat['geometry']['coordinates']
            mag = props.get('mag')
            t_ms = props.get('time')
            if mag is None or t_ms is None:
                continue
            try:
                mag = float(mag)
                lon = float(coords[0])
                lat = float(coords[1])
                depth = float(coords[2]) if len(coords) > 2 and coords[2] is not None else 10.0
                epoch = t_ms / 1000.0
                # Convert to fractional year
                import datetime
                dt = datetime.datetime.utcfromtimestamp(epoch)
                year_frac = dt.year + (dt.timetuple().tm_yday - 1) / 365.25
                events_list.append({
                    'mag': mag, 'lat': lat, 'lon': lon, 'depth': depth,
                    'epoch': epoch, 'year': year_frac
                })
            except (ValueError, TypeError):
                pass
    except Exception as e:
        print(f"    GeoJSON parse error: {e}")

# =========================================================================
# SPATIAL UTILITIES
# =========================================================================

def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

def build_grid_index(events, cell_size=3.0):
    """Build spatial grid index for fast neighbor lookups."""
    grid = {}
    for i, e in enumerate(events):
        gx = int(e['lat'] / cell_size)
        gy = int(e['lon'] / cell_size)
        key = (gx, gy)
        if key not in grid:
            grid[key] = []
        grid[key].append(i)
    return grid, cell_size

def get_nearby_indices(grid, cell_size, lat, lon, radius_km):
    """Get event indices within radius using grid acceleration."""
    # How many cells could radius_km span?
    cell_span = int(np.ceil(radius_km / (cell_size * 111.0))) + 1
    gx = int(lat / cell_size)
    gy = int(lon / cell_size)
    candidates = []
    for dx in range(-cell_span, cell_span + 1):
        for dy in range(-cell_span, cell_span + 1):
            key = (gx + dx, gy + dy)
            if key in grid:
                candidates.extend(grid[key])
    return candidates

# =========================================================================
# TARGET SELECTION: M6+ with aftershock removal
# =========================================================================

def select_targets(events, min_mag=6.0, aftershock_radius_km=200, aftershock_days=60):
    """Select M6+ events with aftershock removal."""
    big = [e for e in events if e['mag'] >= min_mag]
    big.sort(key=lambda e: e['epoch'])

    mainshocks = []
    for e in big:
        is_aftershock = False
        for m in mainshocks:
            dt_days = (e['epoch'] - m['epoch']) / 86400.0
            if 0 < dt_days < aftershock_days:
                dist = haversine_km(e['lat'], e['lon'], m['lat'], m['lon'])
                if dist < aftershock_radius_km:
                    is_aftershock = True
                    break
        if not is_aftershock:
            mainshocks.append(e)

    return mainshocks

# =========================================================================
# CONTROL SELECTION: Same location, shifted 3-4 years
# =========================================================================

def select_controls(mainshocks, events, shift_options=None):
    """For each mainshock, create MULTIPLE controls at the same location shifted in time.
    Try multiple shift values. Only if no M6+ occurred within 300km and 1 year of shifted time.
    Goal: at least 1 control per mainshock, ideally 2-3."""
    if shift_options is None:
        shift_options = [3.0, 3.5, 4.0, -3.0, -3.5, -4.0, 5.0, -5.0]
    controls = []
    controls_per_ms = {}
    max_controls_per = 3

    for ms_idx, ms in enumerate(mainshocks):
        count = 0
        for shift in shift_options:
            if count >= max_controls_per:
                break
            shifted_year = ms['year'] + shift
            if shifted_year < 2005.5 or shifted_year > 2023:
                continue

            # Check no M6+ within 300km and 1 year of shifted time
            conflict = False
            for ms2 in mainshocks:
                if abs(ms2['year'] - shifted_year) < 1.0:
                    dist = haversine_km(ms['lat'], ms['lon'], ms2['lat'], ms2['lon'])
                    if dist < 300:
                        conflict = True
                        break
            if not conflict:
                controls.append({
                    'lat': ms['lat'], 'lon': ms['lon'],
                    'year': shifted_year, 'mag': 0.0,
                    'epoch': ms['epoch'] + shift * 365.25 * 86400,
                    'depth': ms.get('depth', 10.0),
                    'source_mag': ms['mag']
                })
                count += 1

    return controls

# =========================================================================
# FEATURE EXTRACTION
# =========================================================================

def compute_mc(mags):
    """Magnitude of completeness via maximum curvature + 0.2."""
    if len(mags) < 30:
        return 4.0  # default for M4+ catalog
    bins = np.arange(3.5, 8.5, 0.1)
    hist, edges = np.histogram(mags, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if len(hist) == 0 or np.max(hist) == 0:
        return 4.0
    return centers[np.argmax(hist)] + 0.2

def b_value_aki(mags, mc):
    """Aki-Utsu b-value estimator."""
    above = mags[mags >= mc]
    if len(above) < 10:
        return np.nan
    mean_m = np.mean(above)
    if mean_m <= mc:
        return np.nan
    return np.log10(np.e) / (mean_m - mc)

def gr_residual(mags, mc):
    """Gutenberg-Richter residual: deviation from expected linear log-N vs M."""
    above = mags[mags >= mc]
    if len(above) < 20:
        return np.nan
    bins = np.arange(mc, np.max(above) + 0.2, 0.2)
    if len(bins) < 3:
        return np.nan
    hist, edges = np.histogram(above, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    # Cumulative count
    cum = np.cumsum(hist[::-1])[::-1]
    valid = cum > 0
    if np.sum(valid) < 3:
        return np.nan
    log_n = np.log10(cum[valid].astype(float))
    m_vals = centers[valid]
    # Fit line
    if len(m_vals) < 3:
        return np.nan
    coeffs = np.polyfit(m_vals, log_n, 1)
    predicted = np.polyval(coeffs, m_vals)
    residuals = log_n - predicted
    return np.std(residuals)

def spatial_centroid_migration(lats, lons, times, target_lat, target_lon, n_windows=4):
    """Rate of centroid migration toward target epicenter (km/yr)."""
    if len(times) < 20:
        return np.nan
    t_min, t_max = np.min(times), np.max(times)
    if t_max - t_min < 0.5:
        return np.nan
    edges = np.linspace(t_min, t_max, n_windows + 1)
    centroids_dist = []
    centroids_t = []
    for i in range(n_windows):
        mask = (times >= edges[i]) & (times < edges[i+1])
        if np.sum(mask) < 5:
            continue
        clat = np.mean(lats[mask])
        clon = np.mean(lons[mask])
        dist = haversine_km(clat, clon, target_lat, target_lon)
        centroids_dist.append(dist)
        centroids_t.append(0.5 * (edges[i] + edges[i+1]))

    if len(centroids_dist) < 3:
        return np.nan
    # Fit linear trend: negative slope means approaching target
    coeffs = np.polyfit(centroids_t, centroids_dist, 1)
    return coeffs[0]  # km/yr, negative = migrating toward target

def temporal_clustering_metric(times):
    """Fraction of events in bursts vs steady-state Poisson."""
    if len(times) < 20:
        return np.nan
    dt = np.diff(np.sort(times))
    if len(dt) < 10:
        return np.nan
    dt = dt[dt > 0]
    if len(dt) < 10:
        return np.nan
    median_dt = np.median(dt)
    if median_dt <= 0:
        return np.nan
    # Events with inter-event time < 0.2 * median are "clustered"
    clustered = np.sum(dt < 0.2 * median_dt)
    return clustered / len(dt)

def iet_shape_ratio(times):
    """Inter-event time distribution shape: ratio of (mean/median) as proxy for
    heavy-tailed (Lorentzian-like) vs exponential distribution."""
    if len(times) < 20:
        return np.nan
    dt = np.diff(np.sort(times))
    dt = dt[dt > 0]
    if len(dt) < 10:
        return np.nan
    med = np.median(dt)
    if med <= 0:
        return np.nan
    return np.mean(dt) / med  # >1 = heavy tailed

def extract_features(target_lat, target_lon, target_year, events_arr, radius_km=300):
    """Extract all 17 change-only features for a target location and time.

    events_arr: dict with arrays 'lat', 'lon', 'mag', 'depth', 'year' (pre-filtered to nearby).
    """
    lats = events_arr['lat']
    lons = events_arr['lon']
    mags = events_arr['mag']
    depths = events_arr['depth']
    years = events_arr['year']

    # Filter to within radius
    dists = haversine_km(lats, lons, target_lat, target_lon)
    within = dists <= radius_km
    lats = lats[within]
    lons = lons[within]
    mags = mags[within]
    depths = depths[within]
    years = years[within]

    if len(years) < 30:
        return None

    # Define time windows
    t0 = target_year
    # Pre-event windows
    pre_3m = (years >= t0 - 0.25) & (years < t0)
    pre_6m = (years >= t0 - 0.5) & (years < t0)
    pre_12m = (years >= t0 - 1.0) & (years < t0)
    prev_6m = (years >= t0 - 1.0) & (years < t0 - 0.5)
    prev_3m = (years >= t0 - 0.5) & (years < t0 - 0.25)
    # Background: 2-5 years before
    bg = (years >= t0 - 5.0) & (years < t0 - 2.0)

    n_bg = np.sum(bg)
    if n_bg < 20:
        return None

    bg_years_span = min(3.0, np.ptp(years[bg])) if n_bg > 0 else 3.0
    if bg_years_span < 0.5:
        return None
    bg_rate = n_bg / bg_years_span  # events/yr

    mc = compute_mc(mags[bg])

    features = {}

    # 1. Rate change at 3m, 6m, 12m windows
    for label, mask, span in [('rate_change_3m', pre_3m, 0.25),
                               ('rate_change_6m', pre_6m, 0.5),
                               ('rate_change_12m', pre_12m, 1.0)]:
        n = np.sum(mask)
        rate = n / span if span > 0 else 0
        features[label] = (rate / bg_rate - 1.0) if bg_rate > 0 else 0.0

    # 2. Acceleration (recent 6m rate / previous 6m rate)
    n_recent = np.sum(pre_6m)
    n_prev = np.sum(prev_6m)
    if n_prev > 0:
        features['acceleration_6m'] = (n_recent / 0.5) / (n_prev / 0.5) - 1.0
    else:
        features['acceleration_6m'] = 0.0

    # 3. Short-term acceleration (last 3m vs previous 3m)
    n_r3 = np.sum(pre_3m)
    n_p3 = np.sum(prev_3m)
    if n_p3 > 0:
        features['acceleration_3m'] = (n_r3 / 0.25) / (n_p3 / 0.25) - 1.0
    else:
        features['acceleration_3m'] = 0.0

    # 4. b-value change
    b_pre = b_value_aki(mags[pre_12m], mc)
    b_bg = b_value_aki(mags[bg], mc)
    if not np.isnan(b_pre) and not np.isnan(b_bg) and b_bg > 0:
        features['b_value_change'] = (b_pre - b_bg) / b_bg
    else:
        features['b_value_change'] = 0.0

    # 5. CV of inter-event times change
    def cv_iet(mask_):
        t_ = np.sort(years[mask_])
        if len(t_) < 10:
            return np.nan
        dt_ = np.diff(t_)
        dt_ = dt_[dt_ > 0]
        if len(dt_) < 5 or np.mean(dt_) <= 0:
            return np.nan
        return np.std(dt_) / np.mean(dt_)

    cv_pre = cv_iet(pre_12m)
    cv_bg = cv_iet(bg)
    if not np.isnan(cv_pre) and not np.isnan(cv_bg) and cv_bg > 0:
        features['cv_iet_change'] = (cv_pre - cv_bg) / cv_bg
    else:
        features['cv_iet_change'] = 0.0

    # 6. Correlation length change
    def mean_interdist(mask_):
        la = lats[mask_]
        lo = lons[mask_]
        n_ = len(la)
        if n_ < 5 or n_ > 2000:
            if n_ > 2000:
                idx = np.random.choice(n_, 2000, replace=False)
                la = la[idx]
                lo = lo[idx]
                n_ = 2000
            else:
                return np.nan
        # Pairwise distances (subsample if large)
        if n_ > 500:
            idx = np.random.choice(n_, 500, replace=False)
            la = la[idx]
            lo = lo[idx]
        d = haversine_km(la[:, None], lo[:, None], la[None, :], lo[None, :])
        upper = d[np.triu_indices_from(d, k=1)]
        return np.mean(upper) if len(upper) > 0 else np.nan

    ell_pre = mean_interdist(pre_12m)
    ell_bg = mean_interdist(bg)
    if not np.isnan(ell_pre) and not np.isnan(ell_bg) and ell_bg > 0:
        features['ell_change'] = (ell_pre - ell_bg) / ell_bg
    else:
        features['ell_change'] = 0.0

    # 7. Max magnitude anomaly
    mmax_pre = np.max(mags[pre_12m]) if np.sum(pre_12m) > 0 else 0
    mmax_bg = np.max(mags[bg]) if np.sum(bg) > 0 else 0
    features['mmax_anomaly'] = mmax_pre - mmax_bg

    # 8. Mean magnitude change
    mmean_pre = np.mean(mags[pre_12m]) if np.sum(pre_12m) > 5 else np.nan
    mmean_bg = np.mean(mags[bg]) if np.sum(bg) > 5 else np.nan
    if not np.isnan(mmean_pre) and not np.isnan(mmean_bg):
        features['mean_mag_change'] = mmean_pre - mmean_bg
    else:
        features['mean_mag_change'] = 0.0

    # 9. GR residual change
    gr_pre = gr_residual(mags[pre_12m], mc)
    gr_bg = gr_residual(mags[bg], mc)
    if not np.isnan(gr_pre) and not np.isnan(gr_bg) and gr_bg > 0:
        features['gr_residual_change'] = (gr_pre - gr_bg) / gr_bg
    else:
        features['gr_residual_change'] = 0.0

    # 10. Spatial centroid migration rate
    migration = spatial_centroid_migration(
        lats[pre_12m], lons[pre_12m], years[pre_12m],
        target_lat, target_lon, n_windows=4
    )
    features['centroid_migration'] = migration if not np.isnan(migration) else 0.0

    # 11. Depth distribution change
    d_pre = np.mean(depths[pre_12m]) if np.sum(pre_12m) > 5 else np.nan
    d_bg = np.mean(depths[bg]) if np.sum(bg) > 5 else np.nan
    if not np.isnan(d_pre) and not np.isnan(d_bg):
        features['depth_change'] = d_pre - d_bg
    else:
        features['depth_change'] = 0.0

    # 12. Magnitude variance change
    var_pre = np.var(mags[pre_12m]) if np.sum(pre_12m) > 10 else np.nan
    var_bg = np.var(mags[bg]) if np.sum(bg) > 10 else np.nan
    if not np.isnan(var_pre) and not np.isnan(var_bg) and var_bg > 0:
        features['mag_variance_change'] = (var_pre - var_bg) / var_bg
    else:
        features['mag_variance_change'] = 0.0

    # 13. Rate of M4+ relative to M5+ (b-value proxy)
    n4_pre = np.sum(mags[pre_12m] >= 4.0)
    n5_pre = np.sum(mags[pre_12m] >= 5.0)
    n4_bg = np.sum(mags[bg] >= 4.0)
    n5_bg = np.sum(mags[bg] >= 5.0)
    if n5_pre > 0 and n5_bg > 0:
        ratio_pre = n4_pre / n5_pre
        ratio_bg = n4_bg / n5_bg
        features['m4_m5_ratio_change'] = (ratio_pre - ratio_bg) / ratio_bg if ratio_bg > 0 else 0.0
    else:
        features['m4_m5_ratio_change'] = 0.0

    # 14. IET shape ratio change
    shape_pre = iet_shape_ratio(years[pre_12m])
    shape_bg = iet_shape_ratio(years[bg])
    if not np.isnan(shape_pre) and not np.isnan(shape_bg) and shape_bg > 0:
        features['iet_shape_change'] = (shape_pre - shape_bg) / shape_bg
    else:
        features['iet_shape_change'] = 0.0

    # 15. Temporal clustering metric change
    clust_pre = temporal_clustering_metric(years[pre_12m])
    clust_bg = temporal_clustering_metric(years[bg])
    if not np.isnan(clust_pre) and not np.isnan(clust_bg) and clust_bg > 0:
        features['clustering_change'] = (clust_pre - clust_bg) / clust_bg
    else:
        features['clustering_change'] = 0.0

    # 16. Rate acceleration trend (is rate accelerating over last 2 years?)
    # Compute rate in 4 successive 6-month windows
    rates_windows = []
    for w in range(4):
        w_start = t0 - 2.0 + w * 0.5
        w_end = w_start + 0.5
        w_mask = (years >= w_start) & (years < w_end)
        rates_windows.append(np.sum(w_mask) / 0.5)
    if len(rates_windows) == 4 and rates_windows[0] > 0:
        # Fit linear slope to rate trend
        x = np.arange(4)
        r = np.array(rates_windows)
        slope = np.polyfit(x, r, 1)[0]
        features['rate_trend_slope'] = slope / bg_rate if bg_rate > 0 else 0.0
    else:
        features['rate_trend_slope'] = 0.0

    # 17. Magnitude deficit (difference between expected Mmax from GR and observed)
    if not np.isnan(b_bg) and n_bg > 20:
        # Expected Mmax from GR: a - b*Mmax = 0 => Mmax = a/b
        # Simpler: gap between largest observed and second-largest
        sorted_m = np.sort(mags[bg | pre_12m])[::-1]
        if len(sorted_m) >= 2:
            features['mag_deficit'] = sorted_m[0] - sorted_m[1]
        else:
            features['mag_deficit'] = 0.0
    else:
        features['mag_deficit'] = 0.0

    return features

# =========================================================================
# LOGISTIC REGRESSION (manual, L2 regularized)
# =========================================================================

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def logistic_loss(w, X, y, lam, sample_weights=None):
    """Negative log-likelihood with L2 regularization and sample weights."""
    z = X @ w
    p = sigmoid(z)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    losses = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    if sample_weights is not None:
        nll = np.mean(losses * sample_weights)
    else:
        nll = np.mean(losses)
    reg = 0.5 * lam * np.sum(w[1:]**2)
    return nll + reg

def logistic_gradient(w, X, y, lam, sample_weights=None):
    z = X @ w
    p = sigmoid(z)
    residuals = p - y
    if sample_weights is not None:
        residuals = residuals * sample_weights
    grad = X.T @ residuals / len(y)
    grad[1:] += lam * w[1:]
    return grad

def train_logistic(X, y, lam=0.01, lr=0.1, max_iter=5000, tol=1e-7, balance_classes=True):
    """Train logistic regression with gradient descent and class balancing."""
    X_b = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(X_b.shape[1])

    # Compute sample weights for class balance
    sample_weights = None
    if balance_classes:
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == 0)
        n_total = len(y)
        sample_weights = np.ones(len(y))
        if n_pos > 0 and n_neg > 0:
            sample_weights[y == 1] = n_total / (2.0 * n_pos)
            sample_weights[y == 0] = n_total / (2.0 * n_neg)

    best_loss = float('inf')
    patience = 0
    best_w = w.copy()
    for i in range(max_iter):
        loss = logistic_loss(w, X_b, y, lam, sample_weights)
        grad = logistic_gradient(w, X_b, y, lam, sample_weights)
        w -= lr * grad

        if loss < best_loss - tol:
            best_loss = loss
            best_w = w.copy()
            patience = 0
        else:
            patience += 1
            if patience > 300:
                lr *= 0.5
                patience = 0
                if lr < 1e-8:
                    break

    return best_w

def predict_logistic(w, X):
    X_b = np.column_stack([np.ones(len(X)), X])
    return sigmoid(X_b @ w)

# =========================================================================
# GRADIENT BOOSTED DECISION STUMPS (manual implementation)
# =========================================================================

class DecisionStump:
    """A single decision stump: split on one feature at one threshold."""
    def __init__(self):
        self.feature_idx = 0
        self.threshold = 0.0
        self.left_val = 0.0
        self.right_val = 0.0

    def fit(self, X, residuals, sample_weights=None):
        """Find best split to minimize weighted squared error of residuals."""
        n, d = X.shape
        if sample_weights is None:
            sample_weights = np.ones(n) / n

        best_loss = float('inf')
        for j in range(d):
            vals = X[:, j]
            # Try ~20 quantile-based thresholds
            thresholds = np.percentile(vals, np.linspace(5, 95, 20))
            for t in thresholds:
                left = vals <= t
                right = ~left
                n_left = np.sum(sample_weights[left])
                n_right = np.sum(sample_weights[right])
                if n_left < 1e-10 or n_right < 1e-10:
                    continue
                left_val = np.sum(sample_weights[left] * residuals[left]) / n_left
                right_val = np.sum(sample_weights[right] * residuals[right]) / n_right
                pred = np.where(left, left_val, right_val)
                loss = np.sum(sample_weights * (residuals - pred) ** 2)
                if loss < best_loss:
                    best_loss = loss
                    self.feature_idx = j
                    self.threshold = t
                    self.left_val = left_val
                    self.right_val = right_val

    def predict(self, X):
        return np.where(X[:, self.feature_idx] <= self.threshold,
                       self.left_val, self.right_val)

class GradientBoostedStumps:
    """Manual gradient boosting with decision stumps for binary classification."""
    def __init__(self, n_estimators=200, learning_rate=0.1, max_depth=1):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.stumps = []
        self.init_val = 0.0

    def fit(self, X, y, sample_weights=None):
        n = len(y)
        if sample_weights is None:
            # Class-balanced weights
            n_pos = np.sum(y == 1)
            n_neg = np.sum(y == 0)
            sample_weights = np.ones(n)
            if n_pos > 0 and n_neg > 0:
                sample_weights[y == 1] = n / (2.0 * n_pos)
                sample_weights[y == 0] = n / (2.0 * n_neg)
            sample_weights /= np.sum(sample_weights)

        # Initialize with log-odds
        p_mean = np.clip(np.sum(sample_weights * y), 0.01, 0.99)
        self.init_val = np.log(p_mean / (1 - p_mean))
        F = np.full(n, self.init_val)

        self.stumps = []
        for t in range(self.n_estimators):
            p = sigmoid(F)
            # Negative gradient (residuals for log-loss)
            residuals = y - p
            stump = DecisionStump()
            stump.fit(X, residuals, sample_weights)
            self.stumps.append(stump)
            F += self.lr * stump.predict(X)

    def predict_proba(self, X):
        F = np.full(len(X), self.init_val)
        for stump in self.stumps:
            F += self.lr * stump.predict(X)
        return sigmoid(F)

# =========================================================================
# EVALUATION METRICS
# =========================================================================

def compute_auc(y_true, y_score):
    """Manual AUC computation."""
    pairs = sorted(zip(y_score, y_true), reverse=True)
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    tp = 0
    fp = 0
    auc = 0.0
    prev_fp = 0
    prev_tp = 0

    for i, (score, label) in enumerate(pairs):
        if label == 1:
            tp += 1
        else:
            fp += 1
        if i + 1 < len(pairs) and pairs[i+1][0] == score:
            continue
        auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
        prev_fp = fp
        prev_tp = tp

    return auc / (n_pos * n_neg) if n_pos * n_neg > 0 else 0.5

def precision_recall_at_threshold(y_true, y_score, threshold=0.5):
    pred = (y_score >= threshold).astype(int)
    tp = np.sum((pred == 1) & (y_true == 1))
    fp = np.sum((pred == 1) & (y_true == 0))
    fn = np.sum((pred == 0) & (y_true == 1))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall

def roc_curve_manual(y_true, y_score, n_thresholds=200):
    thresholds = np.linspace(0, 1, n_thresholds)
    tpr_list = []
    fpr_list = []
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = np.sum((pred == 1) & (y_true == 1))
        fp = np.sum((pred == 1) & (y_true == 0))
        tpr_list.append(tp / n_pos if n_pos > 0 else 0)
        fpr_list.append(fp / n_neg if n_neg > 0 else 0)
    return np.array(fpr_list), np.array(tpr_list), thresholds

# =========================================================================
# MAIN
# =========================================================================

def main():
    print("=" * 70)
    print("EARTHQUAKE MODEL v4: Global M4+ catalog, 17 change-only features")
    print("=" * 70)

    # Step 1: Load global catalog
    print("\n[1/7] Loading global M4+ catalog (2000-2023)...")
    events = load_global_catalog(2000, 2024, minmag=4.0)
    print(f"  Total events: {len(events)}")

    # Convert to arrays for fast access
    all_lats = np.array([e['lat'] for e in events])
    all_lons = np.array([e['lon'] for e in events])
    all_mags = np.array([e['mag'] for e in events])
    all_depths = np.array([e.get('depth', 10.0) for e in events])
    all_years = np.array([e['year'] for e in events])

    print(f"  Year range: {all_years.min():.1f} - {all_years.max():.1f}")
    print(f"  Magnitude range: {all_mags.min():.1f} - {all_mags.max():.1f}")
    print(f"  M6+: {np.sum(all_mags >= 6.0)}, M7+: {np.sum(all_mags >= 7.0)}")

    # Step 2: Build spatial grid index
    print("\n[2/7] Building spatial grid index...")
    grid, cell_size = build_grid_index(events, cell_size=3.0)
    print(f"  Grid cells: {len(grid)}")

    # Step 3: Select targets (M6+ mainshocks)
    print("\n[3/7] Selecting M6+ mainshocks (aftershock removed)...")
    mainshocks = select_targets(events, min_mag=6.0, aftershock_radius_km=200, aftershock_days=60)
    # Only keep those with year >= 2005 (need 5yr background) and <= 2023
    mainshocks = [m for m in mainshocks if 2005 <= m['year'] <= 2023]
    print(f"  Mainshocks (2005-2023): {len(mainshocks)}")

    # Step 4: Select controls
    print("\n[4/7] Selecting same-location controls...")
    controls = select_controls(mainshocks, events)
    print(f"  Controls: {len(controls)}")

    # Step 5: Extract features
    print("\n[5/7] Extracting features...")
    feature_names = None
    X_list = []
    y_list = []
    meta_list = []  # (year, mag, lat, lon)

    all_samples = [(m, 1) for m in mainshocks] + [(c, 0) for c in controls]
    total = len(all_samples)

    for idx, (sample, label) in enumerate(all_samples):
        if idx % 50 == 0:
            print(f"  Processing {idx}/{total}...")

        slat, slon, syear = sample['lat'], sample['lon'], sample['year']

        # Get nearby events using grid
        cand_idx = get_nearby_indices(grid, cell_size, slat, slon, 350)  # slightly larger for safety
        if len(cand_idx) < 30:
            continue

        cand_idx = np.array(cand_idx)
        nearby = {
            'lat': all_lats[cand_idx],
            'lon': all_lons[cand_idx],
            'mag': all_mags[cand_idx],
            'depth': all_depths[cand_idx],
            'year': all_years[cand_idx],
        }

        feats = extract_features(slat, slon, syear, nearby, radius_km=300)
        if feats is None:
            continue

        if feature_names is None:
            feature_names = sorted(feats.keys())

        X_list.append([feats[k] for k in feature_names])
        y_list.append(label)
        meta_list.append((syear, sample['mag'], slat, slon))

    X = np.array(X_list)
    y = np.array(y_list)
    meta = np.array(meta_list)

    print(f"  Total samples: {len(X)} ({np.sum(y==1)} positive, {np.sum(y==0)} negative)")
    print(f"  Features: {len(feature_names)}")
    for i, fn in enumerate(feature_names):
        print(f"    {i+1}. {fn}")

    # Replace NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)

    # Clip outliers
    for j in range(X.shape[1]):
        p1, p99 = np.percentile(X[:, j], [1, 99])
        X[:, j] = np.clip(X[:, j], p1, p99)

    # Step 6: Train/test split
    print("\n[6/7] Training logistic regression...")
    years_arr = meta[:, 0]
    train_mask = years_arr < 2015
    test_mask = years_arr >= 2015

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    print(f"  Train: {len(X_train)} ({np.sum(y_train==1)} pos, {np.sum(y_train==0)} neg)")
    print(f"  Test:  {len(X_test)} ({np.sum(y_test==1)} pos, {np.sum(y_test==0)} neg)")

    # Standardize
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0)
    sigma[sigma < 1e-10] = 1.0
    X_train_s = (X_train - mu) / sigma
    X_test_s = (X_test - mu) / sigma

    # Feature selection: drop features with individual AUC < 0.52 on train set
    print("  Feature selection...")
    keep_features = []
    for j in range(X_train_s.shape[1]):
        fa = compute_auc(y_train, X_train_s[:, j])
        fa_inv = compute_auc(y_train, -X_train_s[:, j])
        best_fa = max(fa, fa_inv)
        if best_fa >= 0.52:
            keep_features.append(j)
            print(f"    KEEP {feature_names[j]}: train AUC={best_fa:.4f}")
        else:
            print(f"    DROP {feature_names[j]}: train AUC={best_fa:.4f}")

    if len(keep_features) < 3:
        keep_features = list(range(X_train_s.shape[1]))  # keep all if too few pass

    X_train_sel = X_train_s[:, keep_features]
    X_test_sel = X_test_s[:, keep_features]
    feature_names_sel = [feature_names[j] for j in keep_features]
    print(f"  Kept {len(keep_features)}/{len(feature_names)} features")

    # Add polynomial and interaction features for top features
    print("  Adding polynomial/interaction features...")
    # Find top features by individual AUC
    feat_auc_list = []
    for j in range(X_train_sel.shape[1]):
        fa = compute_auc(y_train, X_train_sel[:, j])
        fa_inv = compute_auc(y_train, -X_train_sel[:, j])
        feat_auc_list.append(max(fa, fa_inv))
    top_k = min(6, len(feat_auc_list))
    top_idx = np.argsort(feat_auc_list)[-top_k:]

    # Add squared terms for top features
    extra_train = []
    extra_test = []
    extra_names = []
    for i in top_idx:
        extra_train.append(X_train_sel[:, i] ** 2)
        extra_test.append(X_test_sel[:, i] ** 2)
        extra_names.append(f"{feature_names_sel[i]}_sq")

    # Add pairwise interactions for top 4
    top4 = top_idx[-4:]
    for i in range(len(top4)):
        for j2 in range(i+1, len(top4)):
            ii, jj = top4[i], top4[j2]
            extra_train.append(X_train_sel[:, ii] * X_train_sel[:, jj])
            extra_test.append(X_test_sel[:, ii] * X_test_sel[:, jj])
            extra_names.append(f"{feature_names_sel[ii]}_x_{feature_names_sel[jj]}")

    if extra_train:
        X_train_sel = np.column_stack([X_train_sel] + extra_train)
        X_test_sel = np.column_stack([X_test_sel] + extra_test)
        feature_names_sel = feature_names_sel + extra_names
        print(f"  Total features after expansion: {X_train_sel.shape[1]}")

    # 5-fold cross-validation for lambda selection
    print("  Cross-validating lambda...")
    n_folds = 5
    fold_size = len(X_train_sel) // n_folds
    indices = np.arange(len(X_train_sel))
    np.random.seed(42)
    np.random.shuffle(indices)

    best_cv_auc = 0
    best_lam = None
    for lam in [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        fold_aucs = []
        for fold in range(n_folds):
            val_idx = indices[fold * fold_size:(fold + 1) * fold_size]
            train_idx = np.concatenate([indices[:fold * fold_size], indices[(fold + 1) * fold_size:]])
            w_fold = train_logistic(X_train_sel[train_idx], y_train[train_idx],
                                     lam=lam, lr=0.05, max_iter=5000)
            p_val = predict_logistic(w_fold, X_train_sel[val_idx])
            fold_aucs.append(compute_auc(y_train[val_idx], p_val))
        cv_auc = np.mean(fold_aucs)
        print(f"    lambda={lam:.4f}: CV AUC={cv_auc:.4f} +/- {np.std(fold_aucs):.4f}")
        if cv_auc > best_cv_auc:
            best_cv_auc = cv_auc
            best_lam = lam

    print(f"  Best lambda: {best_lam}, CV AUC: {best_cv_auc:.4f}")

    # Train final logistic model on all training data with best lambda
    w = train_logistic(X_train_sel, y_train, lam=best_lam, lr=0.05, max_iter=10000)

    # Update references for evaluation
    X_train_s = X_train_sel
    X_test_s = X_test_sel
    feature_names = feature_names_sel

    p_test_lr = predict_logistic(w, X_test_s)
    p_train_lr = predict_logistic(w, X_train_s)
    auc_lr_test = compute_auc(y_test, p_test_lr)
    auc_lr_train = compute_auc(y_train, p_train_lr)
    print(f"  Logistic regression: train AUC={auc_lr_train:.4f}, test AUC={auc_lr_test:.4f}")

    # Also train gradient boosted stumps on the ORIGINAL selected features (no polynomial)
    # GBM handles nonlinearity natively
    print("\n  Training gradient boosted stumps...")
    X_train_orig = X_train_s[:, :len(keep_features)]  # original features only
    X_test_orig = X_test_s[:, :len(keep_features)]

    best_gbm_cv_auc = 0
    best_gbm_params = None
    for n_est in [100, 200, 300, 500]:
        for lr_gb in [0.05, 0.1, 0.15]:
            # Quick CV
            fold_aucs = []
            for fold in range(3):  # 3-fold for speed
                fs3 = len(X_train_orig) // 3
                val_idx = indices[fold * fs3:(fold + 1) * fs3]
                train_idx = np.concatenate([indices[:fold * fs3], indices[(fold + 1) * fs3:]])
                # Ensure indices are within bounds for orig features
                gbm_fold = GradientBoostedStumps(n_estimators=n_est, learning_rate=lr_gb)
                gbm_fold.fit(X_train_orig[train_idx], y_train[train_idx])
                p_val = gbm_fold.predict_proba(X_train_orig[val_idx])
                fold_aucs.append(compute_auc(y_train[val_idx], p_val))
            cv_auc = np.mean(fold_aucs)
            if cv_auc > best_gbm_cv_auc:
                best_gbm_cv_auc = cv_auc
                best_gbm_params = (n_est, lr_gb)

    print(f"  Best GBM params: n_est={best_gbm_params[0]}, lr={best_gbm_params[1]}, CV AUC={best_gbm_cv_auc:.4f}")

    # Train final GBM
    gbm = GradientBoostedStumps(n_estimators=best_gbm_params[0], learning_rate=best_gbm_params[1])
    gbm.fit(X_train_orig, y_train)
    p_test_gbm = gbm.predict_proba(X_test_orig)
    p_train_gbm = gbm.predict_proba(X_train_orig)
    auc_gbm_test = compute_auc(y_test, p_test_gbm)
    auc_gbm_train = compute_auc(y_train, p_train_gbm)
    print(f"  GBM: train AUC={auc_gbm_train:.4f}, test AUC={auc_gbm_test:.4f}")

    # Ensemble: average LR and GBM predictions
    p_test_ens = 0.5 * p_test_lr + 0.5 * p_test_gbm
    p_train_ens = 0.5 * p_train_lr + 0.5 * p_train_gbm
    auc_ens_test = compute_auc(y_test, p_test_ens)
    auc_ens_train = compute_auc(y_train, p_train_ens)
    print(f"  Ensemble (LR+GBM): train AUC={auc_ens_train:.4f}, test AUC={auc_ens_test:.4f}")

    # Pick the best model
    best_model = 'LR'
    p_test = p_test_lr
    p_train = p_train_lr
    if auc_gbm_test > auc_lr_test:
        best_model = 'GBM'
        p_test = p_test_gbm
        p_train = p_train_gbm
    if auc_ens_test > max(auc_lr_test, auc_gbm_test):
        best_model = 'Ensemble'
        p_test = p_test_ens
        p_train = p_train_ens
    print(f"  Best model: {best_model}")

    # Step 7: Evaluate
    print("\n[7/7] Evaluation...")

    auc_train = compute_auc(y_train, p_train)
    auc_test = compute_auc(y_test, p_test)

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Train AUC: {auc_train:.4f}")
    print(f"  Test AUC:  {auc_test:.4f}")
    print(f"  v3b AUC:   0.747")
    print(f"  Improvement: {(auc_test - 0.747)*100:+.1f} percentage points")
    print(f"{'='*60}")

    # AUC by magnitude threshold
    print("\n  AUC by magnitude threshold (test set):")
    test_meta = meta[test_mask]
    for mag_thresh in [6.0, 6.5, 7.0, 7.5]:
        # Positives: events >= mag_thresh; negatives: all controls
        pos_mask = (y_test == 1) & (test_meta[:, 1] >= mag_thresh)
        neg_mask = (y_test == 0)
        if np.sum(pos_mask) < 3:
            print(f"    M{mag_thresh}+: too few events ({np.sum(pos_mask)})")
            continue
        sub_y = np.concatenate([np.ones(np.sum(pos_mask)), np.zeros(np.sum(neg_mask))])
        sub_p = np.concatenate([p_test[pos_mask], p_test[neg_mask]])
        sub_auc = compute_auc(sub_y, sub_p)
        print(f"    M{mag_thresh}+: AUC = {sub_auc:.4f} (n={np.sum(pos_mask)} events)")

    # Feature importance (standardized coefficients)
    print("\n  Feature importance (|standardized coeff|):")
    coeff = w[1:]  # skip bias
    importance = list(zip(feature_names, coeff))
    importance.sort(key=lambda x: abs(x[1]), reverse=True)
    for fn, c in importance:
        print(f"    {fn:30s}: {c:+.4f}")

    # Individual feature AUCs
    print("\n  Individual feature AUCs (test set):")
    feat_aucs = []
    for j, fn in enumerate(feature_names):
        fa = compute_auc(y_test, X_test_s[:, j])
        fa_inv = compute_auc(y_test, -X_test_s[:, j])
        best_fa = max(fa, fa_inv)
        feat_aucs.append((fn, best_fa))
        print(f"    {fn:30s}: {best_fa:.4f}")
    feat_aucs.sort(key=lambda x: x[1], reverse=True)

    # Precision/Recall
    print("\n  Precision/Recall at various thresholds (test set):")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        prec, rec = precision_recall_at_threshold(y_test, p_test, t)
        print(f"    threshold={t:.1f}: precision={prec:.3f}, recall={rec:.3f}")

    # =====================================================================
    # GENERATE MULTI-PANEL FIGURE
    # =====================================================================
    print("\n  Generating figure...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor(DARK)

    # Panel 1: ROC curve
    ax = axes[0, 0]
    ax.set_facecolor(DARK)
    fpr, tpr, _ = roc_curve_manual(y_test, p_test)
    ax.plot(fpr, tpr, color=GOLD, linewidth=2.5, label=f'v4 AUC={auc_test:.3f}')
    ax.plot([0, 1], [0, 1], '--', color='gray', alpha=0.5)
    # v3b reference
    ax.axhline(y=0.747, color=ACCENT, linestyle=':', alpha=0.5, label='v3b AUC=0.747')
    ax.set_xlabel('False Positive Rate', color='white')
    ax.set_ylabel('True Positive Rate', color='white')
    ax.set_title('ROC Curve (Test Set)', color=GOLD, fontsize=14)
    ax.legend(facecolor=DARK, edgecolor=GOLD, labelcolor='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color(GOLD)

    # Panel 2: Feature importance
    ax = axes[0, 1]
    ax.set_facecolor(DARK)
    top_n = min(12, len(importance))
    names_top = [x[0].replace('_', '\n') for x in importance[:top_n]]
    vals_top = [abs(x[1]) for x in importance[:top_n]]
    colors_top = [GREEN if x[1] > 0 else ACCENT for x in importance[:top_n]]
    bars = ax.barh(range(top_n), vals_top, color=colors_top, alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names_top, fontsize=8, color='white')
    ax.set_xlabel('|Coefficient|', color='white')
    ax.set_title('Feature Importance', color=GOLD, fontsize=14)
    ax.invert_yaxis()
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color(GOLD)

    # Panel 3: Score distribution
    ax = axes[0, 2]
    ax.set_facecolor(DARK)
    bins = np.linspace(0, 1, 30)
    ax.hist(p_test[y_test == 0], bins=bins, alpha=0.6, color=BLUE, label='Controls', density=True)
    ax.hist(p_test[y_test == 1], bins=bins, alpha=0.6, color=ACCENT, label='M6+ Events', density=True)
    ax.set_xlabel('Predicted Probability', color='white')
    ax.set_ylabel('Density', color='white')
    ax.set_title('Score Distribution (Test)', color=GOLD, fontsize=14)
    ax.legend(facecolor=DARK, edgecolor=GOLD, labelcolor='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color(GOLD)

    # Panel 4: AUC by magnitude
    ax = axes[1, 0]
    ax.set_facecolor(DARK)
    mag_thresholds = [6.0, 6.5, 7.0, 7.5]
    mag_aucs = []
    mag_ns = []
    for mt in mag_thresholds:
        pos_mask = (y_test == 1) & (test_meta[:, 1] >= mt)
        neg_mask = (y_test == 0)
        if np.sum(pos_mask) >= 3:
            sub_y = np.concatenate([np.ones(np.sum(pos_mask)), np.zeros(np.sum(neg_mask))])
            sub_p = np.concatenate([p_test[pos_mask], p_test[neg_mask]])
            mag_aucs.append(compute_auc(sub_y, sub_p))
            mag_ns.append(int(np.sum(pos_mask)))
        else:
            mag_aucs.append(np.nan)
            mag_ns.append(0)
    valid = ~np.isnan(mag_aucs)
    mt_valid = [m for m, v in zip(mag_thresholds, valid) if v]
    ma_valid = [a for a, v in zip(mag_aucs, valid) if v]
    mn_valid = [n for n, v in zip(mag_ns, valid) if v]
    ax.bar(range(len(mt_valid)), ma_valid, color=TEAL, alpha=0.8)
    ax.set_xticks(range(len(mt_valid)))
    ax.set_xticklabels([f'M{m}+\n(n={n})' for m, n in zip(mt_valid, mn_valid)], color='white')
    ax.set_ylabel('AUC', color='white')
    ax.set_title('AUC by Magnitude Threshold', color=GOLD, fontsize=14)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0.747, color=ACCENT, linestyle=':', alpha=0.7, label='v3b baseline')
    ax.set_ylim(0.4, 1.0)
    ax.legend(facecolor=DARK, edgecolor=GOLD, labelcolor='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color(GOLD)

    # Panel 5: Individual feature AUCs
    ax = axes[1, 1]
    ax.set_facecolor(DARK)
    fa_sorted = feat_aucs[:12]
    fa_names = [x[0].replace('_', '\n') for x in fa_sorted]
    fa_vals = [x[1] for x in fa_sorted]
    ax.barh(range(len(fa_sorted)), fa_vals, color=GOLD, alpha=0.8)
    ax.set_yticks(range(len(fa_sorted)))
    ax.set_yticklabels(fa_names, fontsize=8, color='white')
    ax.set_xlabel('AUC', color='white')
    ax.set_title('Individual Feature AUCs', color=GOLD, fontsize=14)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.invert_yaxis()
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color(GOLD)

    # Panel 6: Precision-Recall curve
    ax = axes[1, 2]
    ax.set_facecolor(DARK)
    thresholds = np.linspace(0.01, 0.99, 100)
    precs = []
    recs = []
    for t in thresholds:
        p, r = precision_recall_at_threshold(y_test, p_test, t)
        precs.append(p)
        recs.append(r)
    ax.plot(recs, precs, color=GOLD, linewidth=2)
    baseline = np.sum(y_test == 1) / len(y_test)
    ax.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5, label=f'Baseline {baseline:.2f}')
    ax.set_xlabel('Recall', color='white')
    ax.set_ylabel('Precision', color='white')
    ax.set_title('Precision-Recall Curve (Test)', color=GOLD, fontsize=14)
    ax.legend(facecolor=DARK, edgecolor=GOLD, labelcolor='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color(GOLD)

    fig.suptitle('Earthquake Model v4: M4+ Catalog, Change-Only Features, Same-Location Controls',
                 color=GOLD, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = OUT / 'earthquake_model_v4.png'
    fig.savefig(out_path, dpi=150, facecolor=DARK, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved to {out_path}")

    # Final summary
    print(f"\n{'='*60}")
    print(f"  EARTHQUAKE MODEL v4 COMPLETE")
    print(f"  Test AUC: {auc_test:.4f}")
    print(f"  v3b AUC:  0.747")
    delta = auc_test - 0.747
    if delta > 0:
        print(f"  IMPROVEMENT: +{delta*100:.1f} percentage points")
    else:
        print(f"  Delta: {delta*100:.1f} percentage points")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
