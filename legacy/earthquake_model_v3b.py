#!/usr/bin/env python3
"""
EARTHQUAKE MODEL v3b: PROPER CONTROLS + CHANGE-ONLY FEATURES
=============================================================
Fix two issues from v3:
1. Controls are now SAME LOCATION, different time (no M6+ event)
   - This tests whether the model detects CHANGES, not just active regions
2. Features measure CHANGE from background, not absolute levels
   - Rate CHANGE, b-value CHANGE, CV CHANGE, ell CHANGE
   - Removes trivial "seismic zones have earthquakes" signal

Train 2000-2014, Test 2015-2023.
"""

import numpy as np
from scipy import stats
import urllib.request
import json
from datetime import datetime
from collections import defaultdict
import os
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), 'coherence_field_results')
os.makedirs(OUT_DIR, exist_ok=True)


def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def load_all_events():
    all_events = []
    for yr in range(2000, 2024):
        params = {
            'format': 'geojson',
            'starttime': f'{yr}-01-01',
            'endtime': f'{yr}-12-31',
            'minmagnitude': 5.0,
            'orderby': 'time',
            'limit': 20000,
        }
        url = 'https://earthquake.usgs.gov/fdsnws/event/1/query?' + '&'.join(
            f'{k}={v}' for k, v in params.items()
        )
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode())
            for feat in data['features']:
                p = feat['properties']
                c = feat['geometry']['coordinates']
                all_events.append({
                    'time': p['time'] / 1000,
                    'lat': c[1],
                    'lon': c[0],
                    'mag': p['mag'],
                })
            print(f'    {yr}: {len(data["features"])}', end='  ')
            if yr % 6 == 5:
                print()
        except Exception as e:
            print(f'    {yr}: ERROR')

    all_events.sort(key=lambda x: x['time'])
    print(f'\n    TOTAL: {len(all_events)}')
    return all_events


def main():
    print("=" * 70)
    print("EARTHQUAKE MODEL v3b: CHANGE-ONLY FEATURES, SAME-LOCATION CONTROLS")
    print("=" * 70)

    print("\n  Loading M5+ events globally...")
    all_events = load_all_events()

    # Build grid
    grid = defaultdict(list)
    for e in all_events:
        key = (int(e['lat'] / 5) * 5, int(e['lon'] / 5) * 5)
        grid[key].append(e)

    def get_nearby(lat, lon, R, t_start, t_end):
        clat = int(lat / 5) * 5
        clon = int(lon / 5) * 5
        cands = []
        for dl in range(-3, 4):
            for dn in range(-3, 4):
                cands.extend(grid.get((clat + dl * 5, clon + dn * 5), []))
        return [e for e in cands
                if t_start <= e['time'] < t_end
                and haversine(lat, lon, e['lat'], e['lon']) <= R]

    # Get M6+ targets
    targets_raw = [e for e in all_events if e['mag'] >= 6.0]
    targets_raw.sort(key=lambda x: x['time'])

    # Remove aftershocks
    targets = []
    for t in targets_raw:
        is_after = False
        for prev in targets[-50:]:
            dt = t['time'] - prev['time']
            if 0 < dt < 60 * 86400:
                dist = haversine(t['lat'], t['lon'], prev['lat'], prev['lon'])
                if dist < 200 and prev['mag'] >= t['mag'] - 0.5:
                    is_after = True
                    break
        if not is_after:
            targets.append(t)

    print(f"\n  {len(targets)} M6+ targets (aftershocks removed)")

    R = 300  # km

    def compute_change_features(lat, lon, t_event):
        """Features that measure CHANGE from background, not absolute level."""
        pre_3m = get_nearby(lat, lon, R, t_event - 90 * 86400, t_event)
        pre_6m = get_nearby(lat, lon, R, t_event - 180 * 86400, t_event)
        pre_12m = get_nearby(lat, lon, R, t_event - 365 * 86400, t_event)
        prev_6m = get_nearby(lat, lon, R, t_event - 365 * 86400, t_event - 180 * 86400)
        bg = get_nearby(lat, lon, R, t_event - 5 * 365 * 86400, t_event - 2 * 365 * 86400)

        if len(bg) < 5 or len(pre_12m) < 3:
            return None

        f = {}
        rate_bg_yr = len(bg) / 3.0

        # Rate CHANGE (normalized to own background)
        f['rate_change_6m'] = (len(pre_6m) / 0.5) / max(1, rate_bg_yr) - 1
        f['rate_change_12m'] = (len(pre_12m) / 1.0) / max(1, rate_bg_yr) - 1

        # Acceleration
        f['acceleration'] = len(pre_6m) / max(1, len(prev_6m)) - 1

        # b-value CHANGE
        def bval(evts):
            if len(evts) < 10:
                return None
            mags = np.array([e['mag'] for e in evts])
            mc = max(5.0, np.percentile(mags, 10))
            above = mags[mags >= mc]
            if len(above) < 5 or np.mean(above) <= mc:
                return None
            return np.log10(np.e) / (np.mean(above) - mc)

        b_pre = bval(pre_12m)
        b_bg = bval(bg)
        f['b_change'] = (b_pre - b_bg) if (b_pre and b_bg) else 0

        # CV change (inter-event time clustering)
        def cv_iet(evts):
            if len(evts) < 5:
                return None
            times = sorted([e['time'] for e in evts])
            iets = np.diff(times)
            return np.std(iets) / np.mean(iets) if np.mean(iets) > 0 else None

        cv_pre = cv_iet(pre_12m)
        cv_bg = cv_iet(bg)
        f['cv_change'] = (cv_pre - cv_bg) if (cv_pre is not None and cv_bg is not None) else 0

        # Correlation length CHANGE
        def ell_mean(evts):
            if len(evts) < 5:
                return None
            lats = [e['lat'] for e in evts[:40]]
            lons = [e['lon'] for e in evts[:40]]
            dists = []
            for i in range(len(lats)):
                for j in range(i + 1, len(lats)):
                    dists.append(haversine(lats[i], lons[i], lats[j], lons[j]))
            return np.mean(dists) if dists else None

        ell_pre = ell_mean(pre_12m)
        ell_bg = ell_mean(bg)
        f['ell_change'] = (ell_pre / ell_bg - 1) if (ell_pre and ell_bg and ell_bg > 0) else 0

        # Max magnitude anomaly
        max_pre = max(e['mag'] for e in pre_12m)
        max_bg = max(e['mag'] for e in bg)
        f['max_mag_anomaly'] = max_pre - max_bg

        # Mean magnitude change
        f['mean_mag_change'] = np.mean([e['mag'] for e in pre_12m]) - np.mean([e['mag'] for e in bg])

        # Rate in last 3 months vs previous 3 months (short-term acceleration)
        pre_prev_3m = get_nearby(lat, lon, R, t_event - 180 * 86400, t_event - 90 * 86400)
        f['short_accel'] = len(pre_3m) / max(1, len(pre_prev_3m)) - 1

        return f

    # Compute TARGET features
    print(f"\n  Computing target features...")
    target_data = []
    for i, t in enumerate(targets):
        if i % 200 == 0:
            print(f"    {i}/{len(targets)}")

        feat = compute_change_features(t['lat'], t['lon'], t['time'])
        if feat is not None:
            target_data.append({
                'features': feat,
                'mag': t['mag'],
                'year': datetime.utcfromtimestamp(t['time']).year,
                'lat': t['lat'],
                'lon': t['lon'],
                'label': 1,
            })

    print(f"  {len(target_data)} valid targets")

    # CONTROLS: Same location, different time (no M6+ nearby)
    print(f"\n  Computing control features (same location, shifted time)...")
    np.random.seed(42)
    control_data = []

    for i, t in enumerate(targets):
        if i % 200 == 0:
            print(f"    {i}/{len(targets)}")

        for shift_years in [-4, -3, 3, 4]:
            ctrl_time = t['time'] + shift_years * 365.25 * 86400

            # Check time bounds
            if ctrl_time < datetime(2002, 1, 1).timestamp():
                continue
            if ctrl_time > datetime(2022, 1, 1).timestamp():
                continue

            # Check no M6+ within 300km and 1 year of control time
            has_m6 = False
            for t2 in targets:
                if abs(t2['time'] - ctrl_time) < 365 * 86400:
                    if haversine(t['lat'], t['lon'], t2['lat'], t2['lon']) < 300:
                        has_m6 = True
                        break
            if has_m6:
                continue

            feat = compute_change_features(t['lat'], t['lon'], ctrl_time)
            if feat is not None:
                control_data.append({
                    'features': feat,
                    'mag': 0,
                    'year': datetime.utcfromtimestamp(ctrl_time).year,
                    'lat': t['lat'],
                    'lon': t['lon'],
                    'label': 0,
                })

    print(f"  {len(control_data)} valid controls")

    # Build matrices
    feat_names = [
        'rate_change_6m', 'rate_change_12m', 'acceleration', 'b_change',
        'cv_change', 'ell_change', 'max_mag_anomaly', 'mean_mag_change', 'short_accel',
    ]

    all_data = target_data + control_data
    X = np.array([[d['features'][k] for k in feat_names] for d in all_data])
    y = np.array([d['label'] for d in all_data])
    years = np.array([d['year'] for d in all_data])
    mags = np.array([d['mag'] for d in all_data])

    # Clean
    X = np.nan_to_num(X, nan=0, posinf=5, neginf=-5)
    for j in range(X.shape[1]):
        p1, p99 = np.percentile(X[:, j], [1, 99])
        X[:, j] = np.clip(X[:, j], p1, p99)

    # Train/test split
    train_mask = years <= 2014
    test_mask = years > 2014

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    mags_test = mags[test_mask]

    print(f"\n  Train (2000-2014): {len(X_train)} ({y_train.sum():.0f} targets, {(1 - y_train).sum():.0f} controls)")
    print(f"  Test  (2015-2023): {len(X_test)} ({y_test.sum():.0f} targets, {(1 - y_test).sum():.0f} controls)")

    # Standardize
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0)
    sd[sd == 0] = 1
    X_train_s = (X_train - mu) / sd
    X_test_s = (X_test - mu) / sd

    # Train logistic regression
    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    w = np.zeros(X_train_s.shape[1])
    b = 0
    lr = 0.05
    reg = 0.01

    for epoch in range(3000):
        p = sigmoid(X_train_s @ w + b)
        w -= lr * ((1 / len(y_train)) * X_train_s.T @ (p - y_train) + reg * w)
        b -= lr * (1 / len(y_train)) * np.sum(p - y_train)

    scores_train = sigmoid(X_train_s @ w + b)
    scores_test = sigmoid(X_test_s @ w + b)

    def compute_auc(scores, labels):
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        if len(pos) == 0 or len(neg) == 0:
            return 0.5
        if len(pos) * len(neg) > 2000000:
            n_s = 5000
            ip = np.random.choice(len(pos), min(n_s, len(pos)), replace=False)
            ine = np.random.choice(len(neg), min(n_s, len(neg)), replace=False)
            nc = sum(np.sum(pos[i] > neg[ine]) + 0.5 * np.sum(pos[i] == neg[ine]) for i in ip)
            return nc / (len(ip) * len(ine))
        nc = 0
        for pv in pos:
            nc += np.sum(pv > neg) + 0.5 * np.sum(pv == neg)
        return nc / (len(pos) * len(neg))

    auc_train = compute_auc(scores_train, y_train)
    auc_test = compute_auc(scores_test, y_test)

    print(f"\n  {'=' * 50}")
    print(f"  RESULTS (CHANGE-ONLY FEATURES, SAME-LOCATION CONTROLS)")
    print(f"  {'=' * 50}")
    print(f"  Train AUC: {auc_train:.3f}")
    print(f"  Test AUC:  {auc_test:.3f}")

    # Feature importance
    print(f"\n  Feature importance (standardized coefficients):")
    for name, coef in sorted(zip(feat_names, w), key=lambda x: -abs(x[1])):
        print(f"    {name:>20}: {coef:+.4f}")

    # AUC by magnitude
    print(f"\n  AUC by target magnitude (test):")
    for mt in [6.0, 6.5, 7.0, 7.5]:
        tm = (y_test == 1) & (mags_test >= mt)
        cm = (y_test == 0)
        if tm.sum() >= 5:
            cs = np.concatenate([scores_test[tm], scores_test[cm]])
            cl = np.concatenate([np.ones(tm.sum()), np.zeros(cm.sum())])
            a = compute_auc(cs, cl)
            print(f"    M>={mt}: AUC={a:.3f} (N_targets={tm.sum()})")

    # Individual feature AUCs
    print(f"\n  Individual feature AUCs (test):")
    for idx, name in enumerate(feat_names):
        a = compute_auc(X_test[:, idx], y_test)
        if a < 0.5:
            a = 1 - a  # flip if inverted
        print(f"    {name:>20}: AUC={a:.3f}")
    print(f"    {'Random':>20}: AUC={compute_auc(np.random.rand(len(y_test)), y_test):.3f}")

    # Precision/recall
    print(f"\n  Precision/Recall (test):")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        pred_pos = scores_test >= thresh
        if pred_pos.sum() > 0:
            prec = y_test[pred_pos].mean()
            rec = y_test[pred_pos].sum() / y_test.sum() if y_test.sum() > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            print(f"    P>={thresh:.1f}: precision={prec:.3f}, recall={rec:.3f}, F1={f1:.3f}")

    # Figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Earthquake Model v3b: Change-Only Features\n'
                 f'Same-Location Controls | Test AUC = {auc_test:.3f}',
                 fontsize=14, fontweight='bold')

    # Score distributions
    ax = axes[0, 0]
    ax.hist(scores_test[y_test == 1], bins=30, alpha=0.7, color='red', label='Pre-M6+ events', density=True)
    ax.hist(scores_test[y_test == 0], bins=30, alpha=0.7, color='blue', label='No-event controls', density=True)
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ROC
    ax = axes[0, 1]
    thresholds = np.linspace(0, 1, 200)
    tprs, fprs = [], []
    for th in thresholds:
        tp = np.sum((scores_test >= th) & (y_test == 1))
        fn = np.sum((scores_test < th) & (y_test == 1))
        fp = np.sum((scores_test >= th) & (y_test == 0))
        tn = np.sum((scores_test < th) & (y_test == 0))
        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
    ax.plot(fprs, tprs, 'r-', lw=2, label=f'Model (AUC={auc_test:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Feature importance
    ax = axes[1, 0]
    sorted_feats = sorted(zip(feat_names, w), key=lambda x: abs(x[1]))
    names = [f[0] for f in sorted_feats]
    coefs = [f[1] for f in sorted_feats]
    colors = ['red' if c > 0 else 'blue' for c in coefs]
    ax.barh(range(len(names)), coefs, color=colors, alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Coefficient')
    ax.set_title('Feature Importance (Change Features Only)')
    ax.grid(True, alpha=0.3)

    # AUC by magnitude
    ax = axes[1, 1]
    mag_thresholds = [6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5]
    aucs_mag = []
    ns_mag = []
    for mt in mag_thresholds:
        tm = (y_test == 1) & (mags_test >= mt)
        cm = (y_test == 0)
        if tm.sum() >= 5:
            cs = np.concatenate([scores_test[tm], scores_test[cm]])
            cl = np.concatenate([np.ones(tm.sum()), np.zeros(cm.sum())])
            aucs_mag.append(compute_auc(cs, cl))
            ns_mag.append(tm.sum())
        else:
            aucs_mag.append(np.nan)
            ns_mag.append(0)

    ax.plot(mag_thresholds, aucs_mag, 'ro-', markersize=8)
    ax.axhline(0.5, color='gray', ls='--')
    ax.set_xlabel('Minimum Target Magnitude')
    ax.set_ylabel('AUC')
    ax.set_title('Discrimination by Magnitude')
    for mt, a, n in zip(mag_thresholds, aucs_mag, ns_mag):
        if not np.isnan(a):
            ax.annotate(f'N={n}', (mt, a), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'earthquake_model_v3b_change_features.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")

    print(f"\n{'=' * 70}")
    print("COMPLETE")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
