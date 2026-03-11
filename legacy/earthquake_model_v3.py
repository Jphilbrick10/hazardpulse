#!/usr/bin/env python3
"""
EARTHQUAKE PREDICTION MODEL v3: FULL-SCALE VALIDATION
=====================================================
Test on ALL M6+ earthquakes globally (2000-2023), not just 3-4 famous ones.
Train on 2000-2014, test on 2015-2023.

For each M6+ event: compute precursors from surrounding M5+ seismicity.
Create matched controls: random locations/times with no M6+ events.

This is how real earthquake prediction research works.
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
    """Load ALL M5+ events globally, 2000-2023."""
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
                    'depth': c[2],
                })
            n = len([f for f in data['features']])
            print(f'    {yr}: {n}', end='  ')
            if yr % 6 == 5:
                print()
        except Exception as e:
            print(f'    {yr}: ERROR {e}')

    all_events.sort(key=lambda x: x['time'])
    print(f'\n    TOTAL: {len(all_events)} M5+ events')
    return all_events


def build_spatial_index(events):
    """Grid-based spatial index for fast neighbor lookups."""
    grid = defaultdict(list)
    for e in events:
        key = (int(e['lat'] / 5) * 5, int(e['lon'] / 5) * 5)
        grid[key].append(e)
    return grid


def get_nearby(grid, lat, lon, radius_km, t_start, t_end):
    """Get events within radius and time window."""
    clat = int(lat / 5) * 5
    clon = int(lon / 5) * 5
    candidates = []
    for dlat in range(-3, 4):
        for dlon in range(-3, 4):
            candidates.extend(grid.get((clat + dlat * 5, clon + dlon * 5), []))

    result = []
    for e in candidates:
        if t_start <= e['time'] < t_end:
            if haversine(lat, lon, e['lat'], e['lon']) <= radius_km:
                result.append(e)
    return result


def compute_features(grid, lat, lon, t_event, R=300):
    """Compute precursor features for a location/time."""
    # Pre-event windows
    pre_3m = get_nearby(grid, lat, lon, R, t_event - 90 * 86400, t_event)
    pre_6m = get_nearby(grid, lat, lon, R, t_event - 180 * 86400, t_event)
    pre_12m = get_nearby(grid, lat, lon, R, t_event - 365 * 86400, t_event)
    pre_24m = get_nearby(grid, lat, lon, R, t_event - 730 * 86400, t_event)

    # Background: 2-5 years before
    bg = get_nearby(grid, lat, lon, R, t_event - 5 * 365 * 86400, t_event - 2 * 365 * 86400)

    if len(bg) < 3:
        return None

    features = {}

    # Rate ratios at different windows
    rate_bg = len(bg) / 3.0  # per year
    features['rate_ratio_3m'] = (len(pre_3m) / 0.25) / rate_bg if rate_bg > 0 else 0
    features['rate_ratio_6m'] = (len(pre_6m) / 0.5) / rate_bg if rate_bg > 0 else 0
    features['rate_ratio_12m'] = (len(pre_12m) / 1.0) / rate_bg if rate_bg > 0 else 0
    features['rate_ratio_24m'] = (len(pre_24m) / 2.0) / rate_bg if rate_bg > 0 else 0

    # Acceleration: rate in last 6m / rate in previous 6m
    pre_early_6m = get_nearby(grid, lat, lon, R, t_event - 365 * 86400, t_event - 180 * 86400)
    features['acceleration'] = len(pre_6m) / max(1, len(pre_early_6m))

    # b-value change
    def compute_b(events_list):
        if len(events_list) < 10:
            return None
        mags = np.array([e['mag'] for e in events_list])
        mc = max(5.0, np.percentile(mags, 10))
        above = mags[mags >= mc]
        if len(above) < 5 or np.mean(above) <= mc:
            return None
        return np.log10(np.e) / (np.mean(above) - mc)

    b_pre = compute_b(pre_12m)
    b_bg = compute_b(bg)
    features['b_pre'] = b_pre if b_pre is not None else 1.0
    features['b_bg'] = b_bg if b_bg is not None else 1.0
    features['b_change'] = features['b_pre'] - features['b_bg']

    # Max magnitude in pre-event window
    features['max_mag_pre_6m'] = max((e['mag'] for e in pre_6m), default=0)
    features['max_mag_pre_12m'] = max((e['mag'] for e in pre_12m), default=0)
    features['max_mag_bg'] = max((e['mag'] for e in bg), default=0)
    features['max_mag_ratio'] = features['max_mag_pre_12m'] / features['max_mag_bg'] if features['max_mag_bg'] > 0 else 0

    # Mean magnitude
    features['mean_mag_pre'] = np.mean([e['mag'] for e in pre_12m]) if pre_12m else 5.0
    features['mean_mag_bg'] = np.mean([e['mag'] for e in bg]) if bg else 5.0

    # Inter-event time statistics
    if len(pre_12m) >= 5:
        times = sorted([e['time'] for e in pre_12m])
        iets = np.diff(times)
        features['cv_pre'] = np.std(iets) / np.mean(iets) if np.mean(iets) > 0 else 1.0
        features['median_iet_pre'] = np.median(iets) / 86400  # days
    else:
        features['cv_pre'] = 1.0
        features['median_iet_pre'] = 30.0

    if len(bg) >= 5:
        times_bg = sorted([e['time'] for e in bg])
        iets_bg = np.diff(times_bg)
        features['cv_bg'] = np.std(iets_bg) / np.mean(iets_bg) if np.mean(iets_bg) > 0 else 1.0
    else:
        features['cv_bg'] = 1.0

    features['cv_change'] = features['cv_pre'] - features['cv_bg']

    # Spatial clustering: mean distance between pre-event pairs
    if len(pre_12m) >= 5:
        lats = [e['lat'] for e in pre_12m[:50]]  # sample for speed
        lons = [e['lon'] for e in pre_12m[:50]]
        dists = []
        for i in range(min(30, len(lats))):
            for j in range(i + 1, min(30, len(lats))):
                dists.append(haversine(lats[i], lons[i], lats[j], lons[j]))
        features['ell_pre'] = np.mean(dists) if dists else R
    else:
        features['ell_pre'] = R

    if len(bg) >= 5:
        lats_bg = [e['lat'] for e in bg[:50]]
        lons_bg = [e['lon'] for e in bg[:50]]
        dists_bg = []
        for i in range(min(30, len(lats_bg))):
            for j in range(i + 1, min(30, len(lats_bg))):
                dists_bg.append(haversine(lats_bg[i], lons_bg[i], lats_bg[j], lons_bg[j]))
        features['ell_bg'] = np.mean(dists_bg) if dists_bg else R
    else:
        features['ell_bg'] = R

    features['ell_ratio'] = features['ell_pre'] / features['ell_bg'] if features['ell_bg'] > 0 else 1.0

    # Absolute counts
    features['n_pre_6m'] = len(pre_6m)
    features['n_pre_12m'] = len(pre_12m)
    features['n_bg_annual'] = len(bg) / 3.0

    return features


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def train_logistic(X, y, lr=0.01, epochs=2000, reg=0.01):
    """Logistic regression with L2 regularization."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0

    for epoch in range(epochs):
        z = X @ w + b
        p = sigmoid(z)

        grad_w = (1 / n) * X.T @ (p - y) + reg * w
        grad_b = (1 / n) * np.sum(p - y)

        w -= lr * grad_w
        b -= lr * grad_b

    return w, b


def compute_auc(scores, labels):
    """Compute AUC with sampling for large datasets."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5

    # Sample for speed if needed
    if len(pos) * len(neg) > 2000000:
        n_sample = 5000
        idx_p = np.random.choice(len(pos), min(n_sample, len(pos)), replace=False)
        idx_n = np.random.choice(len(neg), min(n_sample, len(neg)), replace=False)
        n_conc = 0
        for i in idx_p:
            n_conc += np.sum(pos[i] > neg[idx_n]) + 0.5 * np.sum(pos[i] == neg[idx_n])
        return n_conc / (len(idx_p) * len(idx_n))

    n_conc = 0
    for p in pos:
        n_conc += np.sum(p > neg) + 0.5 * np.sum(p == neg)
    return n_conc / (len(pos) * len(neg))


def main():
    print("=" * 70)
    print("EARTHQUAKE PREDICTION MODEL v3: FULL-SCALE VALIDATION")
    print("ALL M6+ events globally, train 2000-2014, test 2015-2023")
    print("=" * 70)

    # Load data
    print("\n  Loading ALL M5+ events globally (2000-2023)...")
    all_events = load_all_events()

    print("\n  Building spatial index...")
    grid = build_spatial_index(all_events)

    # Get M6+ targets
    targets = [e for e in all_events if e['mag'] >= 6.0]
    print(f"\n  M6+ events (raw): {len(targets)}")

    # Remove aftershocks (within 200km and 60 days of a larger or earlier event)
    targets.sort(key=lambda x: x['time'])
    filtered = []
    for t in targets:
        is_aftershock = False
        for prev in filtered[-50:]:  # check recent
            dt = t['time'] - prev['time']
            if 0 < dt < 60 * 86400:
                dist = haversine(t['lat'], t['lon'], prev['lat'], prev['lon'])
                if dist < 200 and prev['mag'] >= t['mag'] - 0.5:
                    is_aftershock = True
                    break
        if not is_aftershock:
            filtered.append(t)

    targets = filtered
    print(f"  After aftershock removal: {len(targets)}")
    print(f"    M6.0-6.4: {sum(1 for t in targets if 6.0 <= t['mag'] < 6.5)}")
    print(f"    M6.5-6.9: {sum(1 for t in targets if 6.5 <= t['mag'] < 7.0)}")
    print(f"    M7.0-7.4: {sum(1 for t in targets if 7.0 <= t['mag'] < 7.5)}")
    print(f"    M7.5+:    {sum(1 for t in targets if t['mag'] >= 7.5)}")

    # Compute features for targets
    print(f"\n  Computing features for {len(targets)} target events...")
    target_data = []
    for i, t in enumerate(targets):
        if i % 100 == 0:
            print(f"    {i}/{len(targets)}...", end=" " if i % 500 != 0 else "\n    ")
        feat = compute_features(grid, t['lat'], t['lon'], t['time'])
        if feat is not None:
            target_data.append({
                'features': feat,
                'mag': t['mag'],
                'year': datetime.utcfromtimestamp(t['time']).year,
                'lat': t['lat'],
                'lon': t['lon'],
                'label': 1,
            })

    print(f"\n  {len(target_data)} targets with valid features")

    # Create controls: random locations with no M6+ event
    print(f"\n  Creating control samples...")
    np.random.seed(42)
    control_data = []

    for i, t in enumerate(targets):
        if i % 100 == 0:
            print(f"    {i}/{len(targets)}...", end=" " if i % 500 != 0 else "\n    ")

        for _ in range(3):  # 3 controls per target for class balance
            # Random shift 10-40 degrees
            shift_lat = np.random.uniform(-30, 30)
            shift_lon = np.random.uniform(-30, 30)
            ctrl_lat = np.clip(t['lat'] + shift_lat, -80, 80)
            ctrl_lon = ((t['lon'] + shift_lon + 180) % 360) - 180

            # Verify no M6+ nearby
            has_m6 = any(
                haversine(ctrl_lat, ctrl_lon, t2['lat'], t2['lon']) < 300
                and abs(t2['time'] - t['time']) < 365 * 86400
                for t2 in targets
            )
            if has_m6:
                continue

            feat = compute_features(grid, ctrl_lat, ctrl_lon, t['time'])
            if feat is not None:
                control_data.append({
                    'features': feat,
                    'mag': 0,
                    'year': datetime.utcfromtimestamp(t['time']).year,
                    'lat': ctrl_lat,
                    'lon': ctrl_lon,
                    'label': 0,
                })

    print(f"\n  {len(control_data)} controls with valid features")

    # Combine
    all_data = target_data + control_data
    feat_names = [
        'rate_ratio_3m', 'rate_ratio_6m', 'rate_ratio_12m', 'rate_ratio_24m',
        'acceleration', 'b_change', 'max_mag_ratio',
        'mean_mag_pre', 'cv_pre', 'cv_change',
        'ell_ratio', 'n_pre_12m', 'n_bg_annual',
    ]

    X = np.array([[d['features'][k] for k in feat_names] for d in all_data])
    y = np.array([d['label'] for d in all_data])
    years = np.array([d['year'] for d in all_data])
    mags = np.array([d['mag'] for d in all_data])

    # Clean
    X = np.nan_to_num(X, nan=0, posinf=10, neginf=-10)
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
    print(f"\n  Training logistic regression...")
    w, b = train_logistic(X_train_s, y_train, lr=0.05, epochs=3000, reg=0.01)

    # Predict
    scores_train = sigmoid(X_train_s @ w + b)
    scores_test = sigmoid(X_test_s @ w + b)

    auc_train = compute_auc(scores_train, y_train)
    auc_test = compute_auc(scores_test, y_test)

    print(f"\n  {'=' * 50}")
    print(f"  RESULTS")
    print(f"  {'=' * 50}")
    print(f"  Train AUC: {auc_train:.3f}")
    print(f"  Test AUC:  {auc_test:.3f}")

    # Feature importance
    print(f"\n  Feature importance (standardized logistic coefficients):")
    for name, coef in sorted(zip(feat_names, w), key=lambda x: -abs(x[1])):
        direction = "+" if coef > 0 else "-"
        print(f"    {name:>20}: {coef:+.4f}")

    # AUC by magnitude
    print(f"\n  AUC by target magnitude (test set):")
    for mag_thresh in [6.0, 6.5, 7.0, 7.5]:
        target_mask = (y_test == 1) & (mags_test >= mag_thresh)
        control_mask = (y_test == 0)

        if target_mask.sum() >= 5:
            combined_scores = np.concatenate([scores_test[target_mask], scores_test[control_mask]])
            combined_labels = np.concatenate([np.ones(target_mask.sum()), np.zeros(control_mask.sum())])
            auc_mag = compute_auc(combined_scores, combined_labels)
            print(f"    M>={mag_thresh}: AUC={auc_mag:.3f} (N_targets={target_mask.sum()}, N_controls={control_mask.sum()})")

    # Precision-recall at thresholds
    print(f"\n  Precision/Recall at probability thresholds (test):")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pred_pos = scores_test >= thresh
        if pred_pos.sum() > 0:
            precision = y_test[pred_pos].mean()
            recall = (y_test[pred_pos] == 1).sum() / y_test.sum() if y_test.sum() > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            print(f"    P>={thresh:.1f}: precision={precision:.3f}, recall={recall:.3f}, F1={f1:.3f}, N={pred_pos.sum()}")

    # Compare to baselines
    print(f"\n  BASELINE COMPARISONS (test set):")

    # Baseline 1: rate only
    rate_auc = compute_auc(X_test[:, feat_names.index('rate_ratio_12m')], y_test)
    print(f"    Rate ratio (12m) only: AUC={rate_auc:.3f}")

    # Baseline 2: max magnitude only
    maxmag_auc = compute_auc(X_test[:, feat_names.index('max_mag_ratio')], y_test)
    print(f"    Max mag ratio only:    AUC={maxmag_auc:.3f}")

    # Baseline 3: event count only
    ncount_auc = compute_auc(X_test[:, feat_names.index('n_pre_12m')], y_test)
    print(f"    N events (12m) only:   AUC={ncount_auc:.3f}")

    # Baseline 4: random
    rand_auc = compute_auc(np.random.rand(len(y_test)), y_test)
    print(f"    Random:                AUC={rand_auc:.3f}")

    # Figure
    print(f"\n  Generating figure...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Earthquake Prediction Model v3: Full-Scale Validation\n'
                 f'Train 2000-2014, Test 2015-2023 | Test AUC = {auc_test:.3f}',
                 fontsize=14, fontweight='bold')

    # Panel 1: Score distribution
    ax = axes[0, 0]
    ax.hist(scores_test[y_test == 1], bins=30, alpha=0.7, color='red', label='M6+ events', density=True)
    ax.hist(scores_test[y_test == 0], bins=30, alpha=0.7, color='blue', label='Controls', density=True)
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('Density')
    ax.set_title('Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: ROC curve (approximate)
    ax = axes[0, 1]
    thresholds = np.linspace(0, 1, 100)
    tprs, fprs = [], []
    for t in thresholds:
        tp = np.sum((scores_test >= t) & (y_test == 1))
        fn = np.sum((scores_test < t) & (y_test == 1))
        fp = np.sum((scores_test >= t) & (y_test == 0))
        tn = np.sum((scores_test < t) & (y_test == 0))
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tprs.append(tpr)
        fprs.append(fpr)
    ax.plot(fprs, tprs, 'r-', lw=2, label=f'Model (AUC={auc_test:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve (Test Set)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Feature importance
    ax = axes[0, 2]
    sorted_feats = sorted(zip(feat_names, w), key=lambda x: abs(x[1]))
    names = [f[0] for f in sorted_feats]
    coefs = [f[1] for f in sorted_feats]
    colors = ['red' if c > 0 else 'blue' for c in coefs]
    ax.barh(range(len(names)), coefs, color=colors, alpha=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel('Coefficient')
    ax.set_title('Feature Importance')
    ax.grid(True, alpha=0.3)

    # Panel 4: AUC by magnitude
    ax = axes[1, 0]
    mag_thresholds = [6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5]
    aucs_by_mag = []
    ns_by_mag = []
    for mt in mag_thresholds:
        tm = (y_test == 1) & (mags_test >= mt)
        cm = (y_test == 0)
        if tm.sum() >= 5:
            cs = np.concatenate([scores_test[tm], scores_test[cm]])
            cl = np.concatenate([np.ones(tm.sum()), np.zeros(cm.sum())])
            aucs_by_mag.append(compute_auc(cs, cl))
            ns_by_mag.append(tm.sum())
        else:
            aucs_by_mag.append(np.nan)
            ns_by_mag.append(tm.sum())

    ax.plot(mag_thresholds, aucs_by_mag, 'ro-', markersize=8)
    ax.axhline(0.5, color='gray', ls='--')
    ax.set_xlabel('Minimum Magnitude')
    ax.set_ylabel('AUC')
    ax.set_title('AUC by Target Magnitude')
    for mt, a, n in zip(mag_thresholds, aucs_by_mag, ns_by_mag):
        if not np.isnan(a):
            ax.annotate(f'N={n}', (mt, a), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 5: Precision-recall
    ax = axes[1, 1]
    precisions, recalls = [], []
    for t in thresholds:
        pred_pos = scores_test >= t
        if pred_pos.sum() > 0:
            prec = y_test[pred_pos].mean()
            rec = (y_test[pred_pos] == 1).sum() / y_test.sum() if y_test.sum() > 0 else 0
        else:
            prec = 1.0
            rec = 0.0
        precisions.append(prec)
        recalls.append(rec)
    ax.plot(recalls, precisions, 'b-', lw=2)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.grid(True, alpha=0.3)

    # Panel 6: Score vs time
    ax = axes[1, 2]
    test_years = years[test_mask]
    for yr in sorted(set(test_years)):
        mask_yr = test_years == yr
        if mask_yr.sum() > 0:
            mean_target = np.mean(scores_test[(y_test == 1) & mask_yr]) if ((y_test == 1) & mask_yr).sum() > 0 else np.nan
            mean_control = np.mean(scores_test[(y_test == 0) & mask_yr]) if ((y_test == 0) & mask_yr).sum() > 0 else np.nan
            ax.scatter(yr, mean_target, c='red', s=30, alpha=0.7)
            ax.scatter(yr, mean_control, c='blue', s=30, alpha=0.7)

    ax.set_xlabel('Year')
    ax.set_ylabel('Mean predicted probability')
    ax.set_title('Annual Score Stability')
    ax.legend(['Targets', 'Controls'], fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, 'earthquake_model_v3_fullscale.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    print(f"\n{'=' * 70}")
    print(f"COMPLETE")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
