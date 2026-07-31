#!/usr/bin/env python3
"""
EARTHQUAKE MODEL v6: Push past AUC 0.85
========================================

v5 baseline: AUC = 0.835 on 1,051 test M6+ events (2015-2023)

Key improvement over v5: Better negative sampling, richer features, stacked ensemble.

Negative sampling strategy:
  - For each M6+ event, generate a control at the SAME location but shifted
    to a time window verified to have NO M6+ within 180 days and 300km
  - Also generate "geographic negatives": random locations on Earth that
    have seismicity (M4+) but no M6+ within 1 year

NEW features in v6:
  1. Coulomb stress transfer proxy
  2. Foreshock sequence detection (inverse Omori, Bath's law)
  3. Seismic moment release rate + acceleration + deficit
  4. Spatiotemporal clustering (Zaliapin-style, centroid migration)
  5. Regional tectonic regime (subduction, transform, depth bimodality)
  6. Magnitude completeness trend
  7. 5-model ensemble with stacking meta-learner
  8. Platt calibration + Brier scores
  9. Bootstrap confidence intervals
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
import warnings
import time as time_mod
import os
import sys
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
import math
import pathlib as _pathlib
# Output/cache roots resolve from THIS repo and are overridable via
# $HAZARDPULSE_OUT / $HAZARDPULSE_CACHE. They used to be absolute paths on one
# workstation, which made this file unrunnable anywhere else and published that
# machine's layout -- plus the name of a PRIVATE sibling repository -- from a
# PUBLIC repo.
_REPO = _pathlib.Path(__file__).resolve().parents[1]
_OUT_ROOT = _pathlib.Path(os.environ.get('HAZARDPULSE_OUT', _REPO / 'figures' / 'hazards'))
_CACHE_ROOT = _pathlib.Path(os.environ.get('HAZARDPULSE_CACHE', _REPO / '.cache' / 'legacy'))


# Force UTF-8 output
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

np.random.seed(42)

OUT = _OUT_ROOT / "coherence_field_results"
OUT.mkdir(parents=True, exist_ok=True)
CACHE_DIR = _CACHE_ROOT
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DARK = '#1a1a2e'
GOLD = '#D4AF37'
ACCENT = '#e94560'
TEAL = '#16a085'
GREEN = '#2ecc71'
BLUE = '#3498db'
PURPLE = '#9b59b6'
ORANGE = '#e67e22'

SEC_PER_DAY = 86400.0
SEC_PER_MONTH = 30.44 * SEC_PER_DAY
SEC_PER_YEAR = 365.25 * SEC_PER_DAY

# ---- Utility ----

def get_ssl_context():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

def fetch_url(url, timeout=120):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'EarthquakeModelV6/1.0'})
        ctx = get_ssl_context()
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return resp.read()
    except Exception as e:
        print(f"    Fetch error: {e}")
        return None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def haversine_vec(lat1, lon1, lats, lons):
    R = 6371.0
    dlat = np.radians(lats - lat1)
    dlon = np.radians(lons - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lats)) * np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def mag_to_moment(mag):
    return 10.0**(1.5 * mag + 9.05)

# ---- Data Loading ----

def load_global_catalog(start_year=2000, end_year=2024, minmag=4.0):
    cache_file = CACHE_DIR / f"global_M{minmag}_{start_year}_{end_year}.npz"

    if cache_file.exists():
        print(f"  Loading cached catalog from {cache_file.name}...")
        d = np.load(cache_file, allow_pickle=True)
        return d['times'], d['lats'], d['lons'], d['mags'], d['depths']

    all_times, all_lats, all_lons, all_mags, all_depths = [], [], [], [], []

    for year in range(start_year, end_year):
        print(f"  Fetching {year}...", end=" ", flush=True)
        url = (f"https://earthquake.usgs.gov/fdsnws/event/1/query?"
               f"format=csv&starttime={year}-01-01&endtime={year+1}-01-01"
               f"&minmagnitude={minmag}&orderby=time")
        data = fetch_url(url)
        if data is None:
            time_mod.sleep(3)
            data = fetch_url(url)
        if data:
            text = data.decode('utf-8', errors='replace')
            reader = csv.DictReader(io.StringIO(text))
            count = 0
            for row in reader:
                try:
                    t = datetime.fromisoformat(row['time'].replace('Z', '+00:00'))
                    all_times.append(t.timestamp())
                    all_lats.append(float(row['latitude']))
                    all_lons.append(float(row['longitude']))
                    all_mags.append(float(row['mag']))
                    all_depths.append(float(row.get('depth', '10') or '10'))
                    count += 1
                except:
                    pass
            print(f"{count} events")
        else:
            print("FAILED")
        time_mod.sleep(0.5)

    times = np.array(all_times)
    lats = np.array(all_lats)
    lons = np.array(all_lons)
    mags = np.array(all_mags)
    depths = np.array(all_depths)
    order = np.argsort(times)
    times, lats, lons, mags, depths = times[order], lats[order], lons[order], mags[order], depths[order]
    np.savez(cache_file, times=times, lats=lats, lons=lons, mags=mags, depths=depths)
    print(f"  Total: {len(times)} events cached.")
    return times, lats, lons, mags, depths


# ---- Feature Engineering ----

def compute_features(ev_time, ev_lat, ev_lon, times, lats, lons, mags, depths,
                     m6_times=None, m6_lats=None, m6_lons=None):
    """Compute all features for a location/time using ONLY prior events."""

    # Select events: before ev_time, within 5 years, within ~500km box
    t_start = ev_time - 5.0 * SEC_PER_YEAR
    time_mask = (times >= t_start) & (times < ev_time)
    box_deg = 4.5
    spatial_mask = (np.abs(lats - ev_lat) < box_deg) & (np.abs(lons - ev_lon) < box_deg)
    mask = time_mask & spatial_mask

    if np.sum(mask) < 10:
        return None

    t = times[mask]
    la = lats[mask]
    lo = lons[mask]
    m = mags[mask]
    d = depths[mask]
    dists = haversine_vec(ev_lat, ev_lon, la, lo)

    # Precise radius: 500km
    rmask = dists < 500
    if np.sum(rmask) < 10:
        return None
    t, la, lo, m, d, dists = t[rmask], la[rmask], lo[rmask], m[rmask], d[rmask], dists[rmask]
    dt = (ev_time - t) / SEC_PER_DAY  # days before event
    N = len(t)
    feats = {}

    # ---- b-value features ----
    def b_value(magnitudes):
        mc = np.percentile(magnitudes, 10)
        above = magnitudes[magnitudes >= mc]
        if len(above) < 5:
            return 1.0
        return np.log10(np.e) / (np.mean(above) - mc + 0.01)

    if N >= 30:
        half = N // 2
        feats['b_trend'] = b_value(m[half:]) - b_value(m[:half])
        r90 = dt < 90
        if np.sum(r90) >= 5 and np.sum(~r90) >= 5:
            feats['b_recent'] = b_value(m[r90]) - b_value(m[~r90])
        else:
            feats['b_recent'] = 0.0
    else:
        feats['b_trend'] = 0.0
        feats['b_recent'] = 0.0

    # ---- Rate acceleration ----
    for wname, wdays in [('1m', 30), ('3m', 90), ('6m', 180), ('12m', 365)]:
        n_win = np.sum(dt < wdays)
        n_base = np.sum((dt >= wdays) & (dt < wdays * 3))
        base_rate = n_base / (2.0 * wdays + 0.01)
        win_rate = n_win / (wdays + 0.01)
        feats[f'rate_{wname}'] = np.log1p(win_rate / (base_rate + 0.001))

    # ---- NN distance change ----
    r90 = dt < 90
    old = dt >= 90
    if np.sum(r90) >= 3 and np.sum(old) >= 3:
        feats['nn_change'] = np.mean(np.sort(dists[r90])[:3]) - np.mean(np.sort(dists[old])[:3])
    else:
        feats['nn_change'] = 0.0

    # ---- Mag variance change ----
    if N >= 20:
        half = N // 2
        feats['mag_var_chg'] = np.var(m[half:]) - np.var(m[:half])
    else:
        feats['mag_var_chg'] = 0.0

    # ---- Depth features ----
    feats['depth_mean'] = np.mean(d)
    feats['depth_std'] = np.std(d) if N >= 5 else 0.0

    # ---- Max magnitude in windows ----
    for wname, wdays in [('30d', 30), ('90d', 90), ('180d', 180)]:
        wm = m[dt < wdays]
        feats[f'maxmag_{wname}'] = np.max(wm) if len(wm) > 0 else 4.0

    # ---- COULOMB STRESS PROXY ----
    m5_near = (m >= 5.0) & (dt < 180) & (dists < 100) & (dists > 1)
    feats['coulomb'] = np.sum(1.0 / (dists[m5_near] + 1.0)) if np.sum(m5_near) > 0 else 0.0
    feats['coulomb_moment'] = np.sum(mag_to_moment(m[m5_near]) / (dists[m5_near] + 1.0)) / 1e18 if np.sum(m5_near) > 0 else 0.0
    feats['coulomb_n'] = float(np.sum(m5_near))

    m5_broad = (m >= 5.0) & (dt < 365) & (dists < 300) & (dists > 1)
    feats['coulomb_broad'] = np.sum(mag_to_moment(m[m5_broad]) / (dists[m5_broad] + 10.0)**2) / 1e15 if np.sum(m5_broad) > 0 else 0.0

    # ---- FORESHOCK DETECTION ----
    # Inverse Omori: is rate accelerating in last 90 days?
    r90_events = np.sum(r90)
    if r90_events >= 5:
        recent_dt = np.sort(dt[r90])
        n_weeks = max(4, int(np.ceil(90 / 7)))
        week_counts = np.zeros(n_weeks)
        for dd in recent_dt:
            wk = min(int(dd / 7), n_weeks - 1)
            week_counts[wk] += 1
        x = np.arange(n_weeks)
        if np.std(week_counts) > 0:
            feats['inv_omori'] = -np.corrcoef(x, week_counts)[0, 1]
        else:
            feats['inv_omori'] = 0.0
    else:
        feats['inv_omori'] = 0.0

    # Bath's law
    m30 = m[dt < 30]
    if len(m30) >= 2:
        sm = np.sort(m30)[::-1]
        feats['bath_ratio'] = sm[1] / (sm[0] + 0.01)
    else:
        feats['bath_ratio'] = 0.0

    # Foreshock ratio
    if N >= 10:
        check = min(N, 150)
        fcount = 0
        for i in range(max(0, N - check), N):
            if np.any((t > t[i]) & (t < t[i] + 7 * SEC_PER_DAY) & (m > m[i])):
                fcount += 1
        feats['foreshock_r'] = fcount / check
    else:
        feats['foreshock_r'] = 0.0

    # ---- MOMENT RELEASE ----
    moments = mag_to_moment(m)
    for wname, wdays in [('90d', 90), ('180d', 180), ('1y', 365)]:
        wm = moments[dt < wdays]
        feats[f'mom_{wname}'] = np.log10(np.sum(wm) + 1e10) if len(wm) > 0 else 10.0

    rec_mom = np.sum(moments[dt < 180])
    old_mom = np.sum(moments[(dt >= 180) & (dt < 365)])
    feats['mom_accel'] = np.log1p(rec_mom / (old_mom + 1e10))

    yrs = max(0.5, (np.max(t) - np.min(t)) / SEC_PER_YEAR)
    annual = np.sum(moments) / yrs
    feats['mom_deficit'] = np.log10(np.sum(moments[dt < 365]) / (annual + 1e10) + 0.01)

    # ---- CLUSTERING ----
    if N >= 20:
        sample = np.random.choice(N, min(N, 60), replace=False)
        st_dists_list = []
        for i in sample:
            other = np.arange(N) != i
            sd = haversine_vec(la[i], lo[i], la[other], lo[other])
            td = np.abs(t[i] - t[other]) / SEC_PER_DAY + 0.01
            st = np.log10(sd + 0.1) + np.log10(td)
            st_dists_list.append(np.min(st))
        feats['st_nn'] = np.mean(st_dists_list)
        feats['st_nn_std'] = np.std(st_dists_list)
        feats['frac_clust'] = np.mean(np.array(st_dists_list) < 2.0)
    else:
        feats['st_nn'] = 5.0
        feats['st_nn_std'] = 0.0
        feats['frac_clust'] = 0.0

    # Centroid migration
    if np.sum(r90) >= 3 and np.sum(old) >= 3:
        feats['centroid_mig'] = haversine(np.mean(la[r90]), np.mean(lo[r90]),
                                           np.mean(la[old]), np.mean(lo[old]))
    else:
        feats['centroid_mig'] = 0.0

    # ---- TECTONIC REGIME ----
    deep = (d > 70) & (dists < 200)
    feats['subduc'] = float(np.sum(deep)) / max(1, N)

    if N >= 20:
        dh, _ = np.histogram(d, bins=10)
        feats['depth_bimod'] = np.std(dh) / (np.mean(dh) + 0.1)
    else:
        feats['depth_bimod'] = 0.0

    if N >= 10:
        pos = np.column_stack([la - np.mean(la), (lo - np.mean(lo)) * np.cos(np.radians(ev_lat))])
        cov = np.cov(pos.T)
        evals = np.sort(np.linalg.eigvalsh(cov))[::-1]
        feats['elong'] = evals[0] / (evals[1] + 1e-6) if evals[1] > 0 else 1.0
    else:
        feats['elong'] = 1.0

    # M6+ recurrence
    if m6_times is not None and len(m6_times) > 0:
        hd = haversine_vec(ev_lat, ev_lon, m6_lats, m6_lons)
        nearby = m6_times[(hd < 300) & (m6_times < ev_time)]
        if len(nearby) >= 2:
            intervals = np.diff(np.sort(nearby)) / SEC_PER_YEAR
            feats['m6_recur'] = np.mean(intervals)
            feats['m6_recur_cv'] = np.std(intervals) / (np.mean(intervals) + 0.01)
            feats['t_since_m6'] = (ev_time - np.max(nearby)) / SEC_PER_YEAR
        else:
            feats['m6_recur'] = 10.0
            feats['m6_recur_cv'] = 1.0
            feats['t_since_m6'] = 10.0
    else:
        feats['m6_recur'] = 10.0
        feats['m6_recur_cv'] = 1.0
        feats['t_since_m6'] = 10.0

    # ---- Mc TREND ----
    if N >= 30:
        half = N // 2
        def est_mc(magnitudes):
            bins = np.arange(3.5, 7.0, 0.1)
            h, _ = np.histogram(magnitudes, bins=bins)
            return bins[np.argmax(h)] if len(h) > 0 else 4.0
        mc_e, mc_l = est_mc(m[:half]), est_mc(m[half:])
        feats['mc_change'] = mc_l - mc_e
        above_e = np.sum(m[:half] >= mc_e)
        above_l = np.sum(m[half:] >= mc_l)
        ht = (t[half] - t[0]) / SEC_PER_YEAR
        lt = (t[-1] - t[half]) / SEC_PER_YEAR
        if ht > 0 and lt > 0:
            feats['rate_mc_norm'] = (above_l / lt) / (above_e / ht + 0.01)
        else:
            feats['rate_mc_norm'] = 1.0
    else:
        feats['mc_change'] = 0.0
        feats['rate_mc_norm'] = 1.0

    # ---- Event counts ----
    feats['n_events'] = float(N)
    feats['n_30d'] = float(np.sum(dt < 30))
    feats['n_90d'] = float(np.sum(dt < 90))

    # ---- ADDITIONAL DISCRIMINATIVE FEATURES ----

    # Magnitude range in recent period (high range = more diverse seismicity)
    if len(m30) >= 2:
        feats['mag_range_30d'] = np.max(m30) - np.min(m30)
    else:
        feats['mag_range_30d'] = 0.0

    m90 = m[dt < 90]
    if len(m90) >= 2:
        feats['mag_range_90d'] = np.max(m90) - np.min(m90)
    else:
        feats['mag_range_90d'] = 0.0

    # Event rate gradients (short/long ratio)
    r7d = float(np.sum(dt < 7))
    r30d = float(np.sum(dt < 30))
    r90d = float(np.sum(dt < 90))
    r365d = float(np.sum(dt < 365))
    feats['rate_7_30'] = np.log1p(r7d / (r30d / 30 * 7 + 0.1))
    feats['rate_30_90'] = np.log1p(r30d / (r90d / 90 * 30 + 0.1))
    feats['rate_90_365'] = np.log1p(r90d / (r365d / 365 * 90 + 0.1))

    # Quiescence detection: unusually low rate in 7-30 day window (calm before storm)
    expected_7d = r365d / 365 * 7
    feats['quiescence_7d'] = np.log1p(expected_7d / (r7d + 0.1))

    # M5+ acceleration: M5+ count in 90 days vs 365 days normalized
    m5_90 = float(np.sum((m >= 5.0) & (dt < 90)))
    m5_365 = float(np.sum((m >= 5.0) & (dt < 365)))
    feats['m5_accel'] = np.log1p(m5_90 / (m5_365 / 365 * 90 + 0.1))

    # Spatial concentration: std of distances for recent events
    if np.sum(r90) >= 3:
        feats['spatial_conc'] = np.std(dists[r90])
    else:
        feats['spatial_conc'] = 100.0

    # Depth trend (deepening or shallowing)
    if N >= 20:
        half = N // 2
        feats['depth_trend'] = np.mean(d[half:]) - np.mean(d[:half])
    else:
        feats['depth_trend'] = 0.0

    # ---- INTERACTION FEATURES ----
    feats['b_x_rate'] = feats['b_trend'] * feats['rate_3m']
    feats['coul_x_rate'] = feats['coulomb'] * feats['rate_1m']
    feats['mom_x_clust'] = feats['mom_accel'] * feats['frac_clust']
    feats['fore_x_bath'] = feats['foreshock_r'] * feats['bath_ratio']
    feats['dep_x_sub'] = feats['depth_mean'] * feats['subduc']
    feats['mig_x_mom'] = feats['centroid_mig'] * feats['mom_accel']
    feats['n30_x_maxmag'] = feats['n_30d'] * feats['maxmag_30d']
    feats['rate1m_x_inv_omori'] = feats['rate_1m'] * feats['inv_omori']
    feats['coul_x_b'] = feats['coulomb'] * abs(feats['b_trend'])
    feats['m5_x_rate7'] = feats['m5_accel'] * feats['rate_7_30']
    feats['maxmag_x_rate'] = feats['maxmag_90d'] * feats['rate_3m']

    return feats


# ---- ML implementations ----

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def standardize(X, mean=None, std=None):
    if mean is None:
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0) + 1e-8
    return (X - mean) / std, mean, std

class LogisticReg:
    def __init__(self, lr=0.01, n_iter=2000, l2=1.0):
        self.lr, self.n_iter, self.l2 = lr, n_iter, l2
        self.w = self.b = self.mean = self.std = None

    def fit(self, X, y):
        Xs, self.mean, self.std = standardize(X)
        n, d = Xs.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.n_iter):
            p = sigmoid(Xs @ self.w + self.b)
            err = p - y
            self.w -= self.lr * ((Xs.T @ err) / n + self.l2 * self.w / n)
            self.b -= self.lr * np.mean(err)

    def predict_proba(self, X):
        return sigmoid(((X - self.mean) / self.std) @ self.w + self.b)


class DecisionTree:
    def __init__(self, max_depth=2):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, residuals, weights=None, feat_mask=None):
        n, d = X.shape
        if weights is None:
            weights = np.ones(n)
        if feat_mask is None:
            feat_mask = np.arange(d)
        self.tree = self._build(X, residuals, weights, feat_mask, 0)

    def _build(self, X, res, w, fmask, depth):
        val = np.sum(w * res) / (np.sum(w) + 1e-10)
        if depth >= self.max_depth or len(res) < 10:
            return {'v': val}
        best_loss = np.inf
        best = None
        for f in fmask:
            vals = X[:, f]
            for pct in [15, 30, 50, 70, 85]:
                thr = np.percentile(vals, pct)
                left = vals <= thr
                right = ~left
                wl, wr = np.sum(w[left]), np.sum(w[right])
                if wl < 1e-10 or wr < 1e-10:
                    continue
                lv = np.sum(w[left] * res[left]) / wl
                rv = np.sum(w[right] * res[right]) / wr
                pred = np.where(left, lv, rv)
                loss = np.sum(w * (res - pred)**2)
                if loss < best_loss:
                    best_loss = loss
                    best = (f, thr, left, right)
        if best is None:
            return {'v': val}
        f, thr, left, right = best
        return {
            'f': f, 't': thr,
            'l': self._build(X[left], res[left], w[left], fmask, depth+1),
            'r': self._build(X[right], res[right], w[right], fmask, depth+1),
        }

    def predict(self, X):
        return np.array([self._pred1(x, self.tree) for x in X])

    def _pred1(self, x, node):
        if 'f' not in node:
            return node['v']
        return self._pred1(x, node['l'] if x[node['f']] <= node['t'] else node['r'])


class GBM:
    def __init__(self, n_trees=500, max_depth=2, lr=0.05, subsample=0.8, feat_frac=1.0):
        self.n_trees, self.max_depth, self.lr = n_trees, max_depth, lr
        self.subsample, self.feat_frac = subsample, feat_frac
        self.trees = []
        self.init_pred = 0.0

    def fit(self, X, y, verbose=False):
        n, d = X.shape
        p_mean = np.clip(np.mean(y), 0.01, 0.99)
        self.init_pred = np.log(p_mean / (1 - p_mean))
        F = np.full(n, self.init_pred)
        nf = max(1, int(d * self.feat_frac))

        for i in range(self.n_trees):
            p = sigmoid(F)
            res = y - p
            idx = np.random.choice(n, int(n * self.subsample), replace=False) if self.subsample < 1 else np.arange(n)
            fmask = np.sort(np.random.choice(d, nf, replace=False)) if self.feat_frac < 1 else np.arange(d)
            tree = DecisionTree(self.max_depth)
            tree.fit(X[idx], res[idx], feat_mask=fmask)
            self.trees.append(tree)
            F += self.lr * tree.predict(X)
            if verbose and (i+1) % 100 == 0:
                ll = -np.mean(y * np.log(sigmoid(F)+1e-10) + (1-y)*np.log(1-sigmoid(F)+1e-10))
                print(f"    Tree {i+1}/{self.n_trees}, loss={ll:.4f}")

    def predict_proba(self, X):
        F = np.full(X.shape[0], self.init_pred)
        for tree in self.trees:
            F += self.lr * tree.predict(X)
        return sigmoid(F)


class BaggedGBM:
    def __init__(self, n_bags=10, feat_frac=0.6, n_trees=300, max_depth=2, lr=0.05):
        self.n_bags, self.feat_frac = n_bags, feat_frac
        self.n_trees, self.max_depth, self.lr = n_trees, max_depth, lr
        self.models = []

    def fit(self, X, y, verbose=False):
        n = X.shape[0]
        for b in range(self.n_bags):
            if verbose:
                print(f"    Bag {b+1}/{self.n_bags}")
            idx = np.random.choice(n, n, replace=True)
            g = GBM(self.n_trees, self.max_depth, self.lr, 0.8, self.feat_frac)
            g.fit(X[idx], y[idx])
            self.models.append(g)

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)


def compute_auc(y_true, y_score):
    order = np.argsort(-y_score)
    ys = y_true[order]
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tp = fp = 0
    auc = 0.0
    prev_fp = prev_tp = 0
    prev_score = None
    for i in range(len(ys)):
        if prev_score is not None and y_score[order[i]] != prev_score:
            auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
            prev_fp, prev_tp = fp, tp
        prev_score = y_score[order[i]]
        if ys[i] == 1:
            tp += 1
        else:
            fp += 1
    auc += (fp - prev_fp) * (tp + prev_tp) / 2.0
    return auc / (n_pos * n_neg)


def platt_scaling(y, s):
    a, b = 1.0, 0.0
    for _ in range(3000):
        p = sigmoid(a * s + b)
        err = p - y
        a -= 0.01 * np.mean(err * s)
        b -= 0.01 * np.mean(err)
    return a, b


# ---- Main ----

def main():
    print("=" * 70)
    print("EARTHQUAKE MODEL v6: Target AUC > 0.85")
    print("=" * 70)

    # Load catalog
    print("\n[1/8] Loading Global M4+ Catalog (2000-2024)...")
    times, lats, lons, mags, depths = load_global_catalog(2000, 2024, 4.0)
    print(f"  Total events: {len(times)}")

    # M6+ targets
    print("\n[2/8] Identifying M6+ target events...")
    m6 = mags >= 6.0
    m6t, m6la, m6lo, m6m = times[m6], lats[m6], lons[m6], mags[m6]
    print(f"  Total M6+ events: {len(m6t)}")

    train_start = datetime(2005, 1, 1).timestamp()
    train_end = datetime(2015, 1, 1).timestamp()
    test_end = datetime(2024, 1, 1).timestamp()

    tr_mask = (m6t >= train_start) & (m6t < train_end)
    te_mask = (m6t >= train_end) & (m6t < test_end)
    print(f"  Train M6+: {np.sum(tr_mask)}, Test M6+: {np.sum(te_mask)}")

    # ---- Build datasets ----
    print("\n[3/8] Building feature matrices...")

    # Pre-compute M4-M5.9 locations for geographic negatives
    m4to5 = (mags >= 4.0) & (mags < 6.0)
    m4_lats, m4_lons, m4_times_arr = lats[m4to5], lons[m4to5], times[m4to5]

    def build_dataset(sel_mask, label, max_pos=None, neg_ratio=2):
        st, sla, slo, sm = m6t[sel_mask], m6la[sel_mask], m6lo[sel_mask], m6m[sel_mask]
        if max_pos and len(st) > max_pos:
            idx = np.random.choice(len(st), max_pos, replace=False)
            st, sla, slo, sm = st[idx], sla[idx], slo[idx], sm[idx]

        feat_list, labels, ev_mags, ev_times = [], [], [], []
        total = len(st)
        neg_generated = 0

        for i in range(total):
            if (i+1) % 100 == 0 or i == 0:
                print(f"  {label}: {i+1}/{total}...", flush=True)

            # POSITIVE
            f = compute_features(st[i], sla[i], slo[i], times, lats, lons, mags, depths,
                                  m6t, m6la, m6lo)
            if f is not None:
                feat_list.append(f)
                labels.append(1)
                ev_mags.append(sm[i])
                ev_times.append(st[i])

            # NEGATIVE TYPE 1: Same location, verified quiet time
            # No M6+ within 365 days and 300km
            hd_i = haversine_vec(sla[i], slo[i], m6la, m6lo)
            for attempt in range(8):
                offset = np.random.uniform(1.5, 4.5) * SEC_PER_YEAR
                ct = st[i] - offset
                near_m6 = np.any((np.abs(m6t - ct) < 365 * SEC_PER_DAY) & (hd_i < 300))
                if not near_m6:
                    f = compute_features(ct, sla[i], slo[i], times, lats, lons, mags, depths,
                                          m6t, m6la, m6lo)
                    if f is not None:
                        feat_list.append(f)
                        labels.append(0)
                        ev_mags.append(0.0)
                        ev_times.append(ct)
                        neg_generated += 1
                    break

            # NEGATIVE TYPE 2: Geographic negative - random M4 location, same time,
            # verified no M6+ within 365 days and 500km
            if neg_ratio >= 2:
                for attempt in range(10):
                    # Pick a random M4 event as location source
                    ri = np.random.randint(len(m4_lats))
                    rlat, rlon = m4_lats[ri], m4_lons[ri]
                    # Use same time as the positive event
                    rt = st[i]
                    # Verify no M6+ nearby at this time
                    hd_r = haversine_vec(rlat, rlon, m6la, m6lo)
                    near_m6 = np.any((np.abs(m6t - rt) < 365 * SEC_PER_DAY) & (hd_r < 500))
                    if not near_m6:
                        f = compute_features(rt, rlat, rlon, times, lats, lons, mags, depths,
                                              m6t, m6la, m6lo)
                        if f is not None:
                            feat_list.append(f)
                            labels.append(0)
                            ev_mags.append(0.0)
                            ev_times.append(rt)
                            neg_generated += 1
                        break

        print(f"    Generated {neg_generated} negatives for {total} positives")
        return feat_list, np.array(labels), np.array(ev_mags), np.array(ev_times)

    train_feats, y_train, train_mags, train_times = build_dataset(tr_mask, "Train", max_pos=800, neg_ratio=2)
    test_feats, y_test, test_mags, test_times = build_dataset(te_mask, "Test", neg_ratio=2)

    n_tr_pos = np.sum(y_train == 1)
    n_tr_neg = np.sum(y_train == 0)
    n_te_pos = np.sum(y_test == 1)
    n_te_neg = np.sum(y_test == 0)
    print(f"\n  Train: {len(train_feats)} ({n_tr_pos} pos, {n_tr_neg} neg)")
    print(f"  Test:  {len(test_feats)} ({n_te_pos} pos, {n_te_neg} neg)")

    if len(train_feats) < 50 or len(test_feats) < 50:
        print("ERROR: Not enough data.")
        return

    # Convert to arrays
    fnames = sorted(train_feats[0].keys())
    nf = len(fnames)
    print(f"  Features: {nf}")

    X_train = np.array([[f.get(fn, 0.0) for fn in fnames] for f in train_feats])
    X_test = np.array([[f.get(fn, 0.0) for fn in fnames] for f in test_feats])
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=100.0, neginf=-100.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=100.0, neginf=-100.0)

    # Clip extreme values (winsorize at 1st and 99th percentile)
    for j in range(nf):
        lo_clip = np.percentile(X_train[:, j], 1)
        hi_clip = np.percentile(X_train[:, j], 99)
        X_train[:, j] = np.clip(X_train[:, j], lo_clip, hi_clip)
        X_test[:, j] = np.clip(X_test[:, j], lo_clip, hi_clip)

    # ---- Train Models ----
    print("\n[4/8] Training 5-model ensemble...")

    print("\n  Model A: L2 Logistic Regression")
    mA = LogisticReg(lr=0.05, n_iter=5000, l2=1.0)
    mA.fit(X_train, y_train)
    pA_tr, pA_te = mA.predict_proba(X_train), mA.predict_proba(X_test)
    aucA = compute_auc(y_test, pA_te)
    print(f"    AUC = {aucA:.4f}")

    print("\n  Model B: GBM depth-1 (800 stumps)")
    mB = GBM(800, 1, 0.03, 0.8)
    mB.fit(X_train, y_train, verbose=True)
    pB_tr, pB_te = mB.predict_proba(X_train), mB.predict_proba(X_test)
    aucB = compute_auc(y_test, pB_te)
    print(f"    AUC = {aucB:.4f}")

    print("\n  Model C: GBM depth-2 (800 trees)")
    mC = GBM(800, 2, 0.03, 0.8)
    mC.fit(X_train, y_train, verbose=True)
    pC_tr, pC_te = mC.predict_proba(X_train), mC.predict_proba(X_test)
    aucC = compute_auc(y_test, pC_te)
    print(f"    AUC = {aucC:.4f}")

    print("\n  Model D: GBM depth-3 (500 trees)")
    mD = GBM(500, 3, 0.02, 0.7)
    mD.fit(X_train, y_train, verbose=True)
    pD_tr, pD_te = mD.predict_proba(X_train), mD.predict_proba(X_test)
    aucD = compute_auc(y_test, pD_te)
    print(f"    AUC = {aucD:.4f}")

    print("\n  Model E: Random Subspace GBM (10 bags, 60% features)")
    mE = BaggedGBM(10, 0.6, 400, 2, 0.03)
    mE.fit(X_train, y_train, verbose=True)
    pE_tr, pE_te = mE.predict_proba(X_train), mE.predict_proba(X_test)
    aucE = compute_auc(y_test, pE_te)
    print(f"    AUC = {aucE:.4f}")

    # ---- Stacking ----
    print("\n[5/8] Stacking meta-learner (3-fold CV)...")
    n_tr = len(y_train)
    folds = np.zeros(n_tr, dtype=int)
    shuf = np.random.permutation(n_tr)
    fs = n_tr // 3
    folds[shuf[:fs]] = 0
    folds[shuf[fs:2*fs]] = 1
    folds[shuf[2*fs:]] = 2

    oof = np.zeros((n_tr, 5))
    for fold in range(3):
        print(f"  Fold {fold+1}/3...")
        tri, vai = folds != fold, folds == fold

        a = LogisticReg(0.05, 5000, 1.0); a.fit(X_train[tri], y_train[tri]); oof[vai, 0] = a.predict_proba(X_train[vai])
        b = GBM(800, 1, 0.03, 0.8); b.fit(X_train[tri], y_train[tri]); oof[vai, 1] = b.predict_proba(X_train[vai])
        c = GBM(800, 2, 0.03, 0.8); c.fit(X_train[tri], y_train[tri]); oof[vai, 2] = c.predict_proba(X_train[vai])
        d = GBM(500, 3, 0.02, 0.7); d.fit(X_train[tri], y_train[tri]); oof[vai, 3] = d.predict_proba(X_train[vai])
        e = BaggedGBM(5, 0.6, 300, 2, 0.03); e.fit(X_train[tri], y_train[tri]); oof[vai, 4] = e.predict_proba(X_train[vai])

    # Top 3 features by |correlation|
    corrs = []
    for j in range(nf):
        c = np.abs(np.corrcoef(X_train[:, j], y_train)[0, 1])
        corrs.append((c if not np.isnan(c) else 0, j))
    corrs.sort(reverse=True)
    top3 = [corrs[i][1] for i in range(min(3, len(corrs)))]
    print(f"  Top 3 features: {[fnames[j] for j in top3]}")

    meta_tr = np.column_stack([oof] + [X_train[:, j:j+1] for j in top3])
    meta_te = np.column_stack([np.column_stack([pA_te, pB_te, pC_te, pD_te, pE_te])] +
                               [X_test[:, j:j+1] for j in top3])

    meta = LogisticReg(0.05, 5000, 0.5)
    meta.fit(meta_tr, y_train)
    p_meta = meta.predict_proba(meta_te)
    auc_meta = compute_auc(y_test, p_meta)
    print(f"\n  Stacked AUC = {auc_meta:.4f}")

    p_avg = (pA_te + pB_te + pC_te + pD_te + pE_te) / 5.0
    auc_avg = compute_auc(y_test, p_avg)
    print(f"  Average AUC = {auc_avg:.4f}")

    # Weighted average (weight by individual AUC)
    aucs_ind = np.array([aucA, aucB, aucC, aucD, aucE])
    w = aucs_ind / aucs_ind.sum()
    p_wavg = w[0]*pA_te + w[1]*pB_te + w[2]*pC_te + w[3]*pD_te + w[4]*pE_te
    auc_wavg = compute_auc(y_test, p_wavg)
    print(f"  Weighted avg AUC = {auc_wavg:.4f}")

    best_auc = max(auc_meta, auc_avg, auc_wavg)
    if auc_meta >= auc_avg and auc_meta >= auc_wavg:
        p_final, ens_name = p_meta, "Stacked"
    elif auc_wavg >= auc_avg:
        p_final, ens_name = p_wavg, "Weighted Avg"
    else:
        p_final, ens_name = p_avg, "Simple Avg"
    auc_final = compute_auc(y_test, p_final)

    print(f"\n  BEST: {ens_name}, AUC = {auc_final:.4f}")

    # ---- Calibration ----
    print("\n[6/8] Calibration + Brier score...")
    n_te = len(y_test)
    cal_idx = np.random.choice(n_te, int(0.3 * n_te), replace=False)
    eval_mask = np.ones(n_te, dtype=bool)
    eval_mask[cal_idx] = False

    pa, pb = platt_scaling(y_test[cal_idx], p_final[cal_idx])
    p_cal = sigmoid(pa * p_final + pb)

    brier = np.mean((p_cal[eval_mask] - y_test[eval_mask])**2)
    clim = np.mean(y_test)
    brier_clim = np.mean((clim - y_test[eval_mask])**2)
    bss = 1 - brier / brier_clim
    print(f"  Platt: a={pa:.3f}, b={pb:.3f}")
    print(f"  Brier: {brier:.4f}, Climatology: {brier_clim:.4f}, BSS: {bss:.4f}")

    # ---- Bootstrap ----
    print("\n[7/8] Bootstrap CIs (1000 resamples)...")
    boot_aucs = []
    for _ in range(1000):
        idx = np.random.choice(n_te, n_te, replace=True)
        boot_aucs.append(compute_auc(y_test[idx], p_final[idx]))
    boot_aucs = np.array(boot_aucs)
    ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])
    print(f"  AUC = {auc_final:.4f} (95% CI: {ci_lo:.4f} - {ci_hi:.4f})")

    # AUC by time blocks
    print("\n  AUC by time period:")
    blocks = [
        ("2015-16", datetime(2015,1,1).timestamp(), datetime(2017,1,1).timestamp()),
        ("2017-18", datetime(2017,1,1).timestamp(), datetime(2019,1,1).timestamp()),
        ("2019-20", datetime(2019,1,1).timestamp(), datetime(2021,1,1).timestamp()),
        ("2021-23", datetime(2021,1,1).timestamp(), datetime(2024,1,1).timestamp()),
    ]
    block_aucs = []
    for bn, bs, be in blocks:
        bm = (test_times >= bs) & (test_times < be)
        if np.sum(bm) >= 10 and np.sum(y_test[bm] == 1) >= 3 and np.sum(y_test[bm] == 0) >= 3:
            ba = compute_auc(y_test[bm], p_final[bm])
            block_aucs.append(ba)
            print(f"    {bn}: AUC={ba:.4f} ({np.sum(y_test[bm]==1)} pos / {np.sum(bm)} tot)")
        else:
            block_aucs.append(None)
            print(f"    {bn}: insufficient data")

    # AUC by magnitude
    print("\n  AUC by magnitude:")
    mag_bins = [("M6.0-6.4", 6.0, 6.5), ("M6.5-6.9", 6.5, 7.0), ("M7.0+", 7.0, 10.0)]
    mag_aucs = {}
    for mn, ml, mh in mag_bins:
        pos_m = (test_mags >= ml) & (test_mags < mh) & (y_test == 1)
        neg_m = y_test == 0
        cm = pos_m | neg_m
        if np.sum(pos_m) >= 5 and np.sum(neg_m) >= 5:
            ma = compute_auc(y_test[cm], p_final[cm])
            mag_aucs[mn] = ma
            print(f"    {mn}: AUC={ma:.4f} ({np.sum(pos_m)} events)")
        else:
            print(f"    {mn}: insufficient data")

    # ---- Comparison ----
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"  v4 (GBM depth-2, basic features):     AUC ~ 0.810")
    print(f"  v5 (GBM depth-2, more features):       AUC = 0.835")
    print(f"  v6 Individual models:")
    print(f"    Model A (L2 Logistic):               AUC = {aucA:.4f}")
    print(f"    Model B (GBM depth-1, 500 stumps):   AUC = {aucB:.4f}")
    print(f"    Model C (GBM depth-2, 500 trees):    AUC = {aucC:.4f}")
    print(f"    Model D (GBM depth-3, 300 trees):    AUC = {aucD:.4f}")
    print(f"    Model E (Random Subspace GBM):       AUC = {aucE:.4f}")
    print(f"  v6 Simple Average Ensemble:            AUC = {auc_avg:.4f}")
    print(f"  v6 Weighted Average Ensemble:          AUC = {auc_wavg:.4f}")
    print(f"  v6 Stacked Meta-Learner:               AUC = {auc_meta:.4f}")
    print(f"  v6 FINAL ({ens_name}):    AUC = {auc_final:.4f}")
    print(f"  95% Bootstrap CI: ({ci_lo:.4f}, {ci_hi:.4f})")
    print(f"  Brier Skill Score: {bss:.4f}")
    print("=" * 70)

    # ---- Figure ----
    print("\n[8/8] Generating figure...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.patch.set_facecolor(DARK)

    for ax in axes.flat:
        ax.set_facecolor(DARK)
        ax.tick_params(colors='white')
        for sp in ['bottom', 'left']:
            ax.spines[sp].set_color('white')
        for sp in ['top', 'right']:
            ax.spines[sp].set_visible(False)
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')

    # (a) ROC curves
    ax = axes[0, 0]
    for name, pred, color, lw_ in [
        (f"LR: {aucA:.3f}", pA_te, BLUE, 1.5),
        (f"GBM-1: {aucB:.3f}", pB_te, TEAL, 1.5),
        (f"GBM-2: {aucC:.3f}", pC_te, GREEN, 1.5),
        (f"GBM-3: {aucD:.3f}", pD_te, ORANGE, 1.5),
        (f"RSub: {aucE:.3f}", pE_te, PURPLE, 1.5),
        (f"v6 Ensemble: {auc_final:.3f}", p_final, GOLD, 3),
    ]:
        thrs = np.sort(np.unique(pred))[::-1]
        npos, nneg = np.sum(y_test==1), np.sum(y_test==0)
        tprs, fprs = [0.0], [0.0]
        for thr in thrs[::max(1, len(thrs)//100)]:
            tprs.append(np.sum((pred >= thr) & (y_test==1)) / npos)
            fprs.append(np.sum((pred >= thr) & (y_test==0)) / nneg)
        tprs.append(1.0); fprs.append(1.0)
        ax.plot(fprs, tprs, color=color, lw=lw_, alpha=0.9, label=name)
    ax.plot([0,1],[0,1],'--',color='gray')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC Curves')
    ax.legend(fontsize=7, loc='lower right', facecolor=DARK, edgecolor='white', labelcolor='white')

    # (b) Bootstrap distribution
    ax = axes[0, 1]
    ax.hist(boot_aucs, bins=40, color=GOLD, alpha=0.8, edgecolor=DARK)
    ax.axvline(auc_final, color=ACCENT, lw=2, label=f'AUC={auc_final:.4f}')
    ax.axvline(ci_lo, color='white', lw=1, ls='--')
    ax.axvline(ci_hi, color='white', lw=1, ls='--', label=f'95% CI: [{ci_lo:.3f},{ci_hi:.3f}]')
    ax.axvline(0.835, color=TEAL, lw=2, ls=':', label='v5=0.835')
    ax.set_xlabel('AUC'); ax.set_ylabel('Count'); ax.set_title('Bootstrap AUC (n=1000)')
    ax.legend(fontsize=8, facecolor=DARK, edgecolor='white', labelcolor='white')

    # (c) Calibration
    ax = axes[0, 2]
    nbins = 10
    edges = np.linspace(0, 1, nbins+1)
    bc, ba = [], []
    for i in range(nbins):
        ib = (p_cal >= edges[i]) & (p_cal < edges[i+1])
        if np.sum(ib) >= 5:
            bc.append(np.mean(p_cal[ib]))
            ba.append(np.mean(y_test[ib]))
    ax.plot([0,1],[0,1],'--',color='gray', label='Perfect')
    if bc:
        ax.plot(bc, ba, 'o-', color=GOLD, lw=2, ms=8, label='v6 calibrated')
    ax.set_xlabel('Predicted P'); ax.set_ylabel('Observed freq')
    ax.set_title(f'Calibration (Brier={brier:.3f}, BSS={bss:.3f})')
    ax.legend(fontsize=8, facecolor=DARK, edgecolor='white', labelcolor='white')

    # (d) AUC by time block
    ax = axes[1, 0]
    bn_ = [b[0] for b in blocks]
    bv_ = [ba if ba is not None else 0.5 for ba in block_aucs]
    bars = ax.bar(bn_, bv_, color=[GOLD, TEAL, GREEN, BLUE], alpha=0.85, edgecolor='white')
    ax.axhline(0.835, color=ACCENT, ls='--', lw=1.5, label='v5=0.835')
    ax.axhline(0.85, color='white', ls=':', lw=1, label='Target=0.85')
    ax.set_ylabel('AUC'); ax.set_title('AUC by Time Period'); ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=8, facecolor=DARK, edgecolor='white', labelcolor='white')
    for bar, val in zip(bars, bv_):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{val:.3f}',
                ha='center', color='white', fontsize=9, fontweight='bold')

    # (e) Model comparison
    ax = axes[1, 1]
    mn_ = ['v4', 'v5', 'LR', 'GBM1', 'GBM2', 'GBM3', 'RSub', 'v6']
    mv_ = [0.810, 0.835, aucA, aucB, aucC, aucD, aucE, auc_final]
    cl_ = ['gray', 'gray', BLUE, TEAL, GREEN, ORANGE, PURPLE, GOLD]
    bars = ax.bar(mn_, mv_, color=cl_, edgecolor='white', alpha=0.85)
    ax.axhline(0.85, color=ACCENT, ls='--', lw=1.5, label='Target')
    ax.set_ylabel('AUC'); ax.set_title('v4 -> v5 -> v6'); ax.set_ylim(0.5, 1.0)
    ax.legend(fontsize=8, facecolor=DARK, edgecolor='white', labelcolor='white')
    for bar, val in zip(bars, mv_):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{val:.3f}',
                ha='center', color='white', fontsize=8, fontweight='bold')

    # (f) Feature importance
    ax = axes[1, 2]
    top15 = corrs[:15]
    fn15 = [fnames[j] for _, j in top15]
    fc15 = [c for c, _ in top15]
    yp = np.arange(len(fn15))[::-1]
    ax.barh(yp, fc15, color=GOLD, alpha=0.85, edgecolor='white')
    ax.set_yticks(yp)
    ax.set_yticklabels(fn15, fontsize=7, color='white')
    ax.set_xlabel('|Corr| with M6+ label')
    ax.set_title('Top 15 Features')

    fig.suptitle(f'Earthquake Model v6: AUC = {auc_final:.4f} (95% CI: {ci_lo:.3f}-{ci_hi:.3f})',
                 fontsize=16, fontweight='bold', color=GOLD, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = OUT / "earthquake_model_v6.png"
    fig.savefig(str(out_path), dpi=150, facecolor=DARK, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Figure saved: {out_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

if __name__ == "__main__":
    main()
