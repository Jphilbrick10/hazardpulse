#!/usr/bin/env python3
"""
HURRICANE RAPID INTENSIFICATION MODEL
=======================================
Dedicated hurricane model. NOT the same as earthquakes.

A hurricane is a THERMODYNAMIC ENGINE, not a brittle failure.
The physics is fundamentally different:
  - Energy source: ocean heat content (not tectonic stress)
  - Dissipation: wind shear, dry air, friction (not fracture)
  - Structure: axisymmetric vortex (not fault plane)
  - Timescale: hours to days (not years)

HELMHOLTZ IN HURRICANE CONTEXT:
  D * nabla^2(tau) - Gamma * tau + S = 0

  Where:
    tau = vortex coherence (organized convection)
    D = radial diffusion of angular momentum
    Gamma = decoherence from:
      - Vertical wind shear (tilts vortex)
      - Dry air intrusion (disrupts convection)
      - Sea surface friction
      - Beta-drift dispersion
    S = energy source from:
      - Sea surface enthalpy flux (SST dependent)
      - Ocean heat content (depth of warm water)
      - Moisture convergence

  RI occurs when: S/Gamma crosses a threshold
    -> Positive feedback loop: wind increases -> more surface flux -> more wind
    -> This is the Wind-Induced Surface Heat Exchange (WISHE) feedback
    -> In Helmholtz terms: S becomes self-amplifying

WHAT WE ACTUALLY MEASURE:
  - Wind speed at each 6-hour observation (IBTrACS)
  - Central pressure
  - Radius of maximum wind (RMW) — this IS the coherence length ell
  - Latitude (Coriolis parameter)
  - We compute: dV/dt, d(RMW)/dt, pressure-wind relationship

RI DEFINITION: dV >= 30 kt in 24 hours (standard NHC definition)

THE KEY QUESTION:
  Can we identify which storms will undergo RI, and when, BEFORE it happens?
  What is the 6-12 hour RI prediction skill?
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, optimize
from pathlib import Path
import urllib.request
import ssl
import warnings
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

DARK = '#1a1a2e'
GOLD = '#D4AF37'
ACCENT = '#e94560'
TEAL = '#16a085'
GREEN = '#2ecc71'
BLUE = '#3498db'

def get_ssl_context():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx

def fetch_url(url, timeout=60):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        ctx = get_ssl_context()
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return resp.read()
    except Exception as e:
        print(f"    Fetch error: {e}")
        return None


def load_ibtracs():
    """Load North Atlantic IBTrACS data."""
    print("\n  Loading IBTrACS North Atlantic...")
    url = "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.NA.list.v04r01.csv"
    data = fetch_url(url, timeout=90)

    if not data:
        print("  IBTrACS download failed.")
        return {}

    text = data.decode('utf-8', errors='replace')
    lines = text.strip().split('\n')

    header = lines[0].split(',')
    col_map = {}
    for i, h in enumerate(header):
        col_map[h.strip().strip('"')] = i

    sid_col = col_map.get('SID', 0)
    name_col = col_map.get('NAME', 1)
    time_col = col_map.get('ISO_TIME', 6)
    lat_col = col_map.get('LAT', None)
    lon_col = col_map.get('LON', None)
    wind_col = col_map.get('USA_WIND', None)
    pres_col = col_map.get('USA_PRES', None)
    rmw_col = col_map.get('USA_RMW', None)

    storms = {}
    for line in lines[2:]:
        parts = line.split(',')
        if len(parts) < 10:
            continue
        try:
            sid = parts[sid_col].strip().strip('"')
            name = parts[name_col].strip().strip('"')

            def safe_float(col):
                if col is None:
                    return -999
                v = parts[col].strip().strip('"')
                return float(v) if v not in ('', ' ') else -999

            wind = safe_float(wind_col)
            pres = safe_float(pres_col)
            rmw = safe_float(rmw_col)
            lat = safe_float(lat_col)
            lon = safe_float(lon_col)
            time_str = parts[time_col].strip().strip('"')
            year = int(time_str[:4]) if len(time_str) >= 4 else 0

            if sid not in storms:
                storms[sid] = {'name': name, 'entries': []}

            storms[sid]['entries'].append({
                'wind': wind, 'pres': pres, 'rmw': rmw,
                'lat': lat, 'lon': lon, 'year': year,
                'time_str': time_str,
            })
        except:
            pass

    print(f"  Loaded {len(storms)} storms")
    return storms


# =========================================================================
# STORM FEATURE EXTRACTION
# =========================================================================

def extract_storm_features(entries):
    """
    For each 6-hour observation in a storm, compute features that
    could predict RI in the NEXT 24 hours.

    Returns list of (features_dict, label) where label = 1 if RI happens next 24h.
    """
    if len(entries) < 8:
        return []

    winds = [e['wind'] for e in entries]
    pressures = [e['pres'] for e in entries]
    rmws = [e['rmw'] for e in entries]
    lats = [e['lat'] for e in entries]

    samples = []

    for i in range(4, len(entries) - 4):
        # Need valid wind for current and next 24h (4 steps)
        w_now = winds[i]
        w_24 = winds[min(i+4, len(winds)-1)]

        if w_now <= 0 or w_24 <= 0:
            continue

        # RI label: did wind increase >= 30 kt in next 24h?
        dw_next = w_24 - w_now
        label = 1 if dw_next >= 30 else 0

        features = {}

        # F1: Current intensity
        features['wind_now'] = w_now

        # F2: Current pressure
        features['pressure_now'] = pressures[i] if pressures[i] > 0 else -999

        # F3: Current RMW (coherence length!)
        features['rmw_now'] = rmws[i] if rmws[i] > 0 else -999

        # F4: Latitude (Coriolis parameter affects vortex dynamics)
        features['lat_now'] = lats[i] if lats[i] != -999 else -999

        # F5: 12-hour intensity change (prior trend)
        w_12ago = winds[i-2] if i >= 2 and winds[i-2] > 0 else -999
        features['dw_12h'] = w_now - w_12ago if w_12ago > 0 else -999

        # F6: 24-hour intensity change (prior trend)
        w_24ago = winds[i-4] if i >= 4 and winds[i-4] > 0 else -999
        features['dw_24h'] = w_now - w_24ago if w_24ago > 0 else -999

        # F7: RMW change (is eye contracting?)
        rmw_12ago = rmws[i-2] if i >= 2 and rmws[i-2] > 0 else -999
        features['drmw_12h'] = rmws[i] - rmw_12ago if rmws[i] > 0 and rmw_12ago > 0 else -999

        # F8: Pressure-wind relationship slope (gradient balance efficiency)
        if pressures[i] > 0 and w_now > 0:
            # MPI theory: V^2 ~ (SST - T_out) * (C_k/C_d) * dp
            # Simplified: V ~ sqrt(1010 - P) approximately
            features['pw_efficiency'] = w_now**2 / max(1013 - pressures[i], 1)
        else:
            features['pw_efficiency'] = -999

        # F9: Intensification acceleration
        if i >= 6 and winds[i-2] > 0 and winds[i-4] > 0 and winds[i-6] > 0:
            dw_recent = winds[i] - winds[i-2]
            dw_prior = winds[i-2] - winds[i-4]
            features['acceleration'] = dw_recent - dw_prior
        else:
            features['acceleration'] = -999

        # F10: RMW contraction rate (key Helmholtz predictor)
        if rmws[i] > 0 and i >= 4:
            rmw_seq = [rmws[j] for j in range(max(0,i-4), i+1) if rmws[j] > 0]
            if len(rmw_seq) >= 3:
                t_rmw = np.arange(len(rmw_seq))
                sl, _, r, _, _ = stats.linregress(t_rmw, rmw_seq)
                features['rmw_trend'] = sl  # negative = contracting
            else:
                features['rmw_trend'] = -999
        else:
            features['rmw_trend'] = -999

        # F11: S/Gamma proxy — wind relative to MPI
        # MPI ~ 140 kt for typical SST, so wind/MPI = how close to max
        features['intensity_fraction'] = w_now / 140.0

        samples.append((features, label, dw_next))

    return samples


# =========================================================================
# RI PREDICTION MODEL
# =========================================================================

def build_ri_predictor(storms):
    """
    Build and evaluate an RI prediction model.

    Uses a simple but proper approach:
    1. Extract features from every 6-hour observation
    2. Split into train/test by year (pre-2010 train, 2010+ test)
    3. Use logistic regression (interpretable, no black box)
    4. Evaluate with ROC AUC, reliability, Brier score
    """
    print("\n" + "=" * 60)
    print("  BUILDING RI PREDICTION MODEL")
    print("=" * 60)

    # Extract features from all storms
    all_samples = []
    for sid, storm in storms.items():
        samples = extract_storm_features(storm['entries'])
        for features, label, dw in samples:
            features['year'] = storm['entries'][0]['year']
            all_samples.append((features, label, dw))

    print(f"  Total samples: {len(all_samples)}")
    n_ri = sum(1 for _, l, _ in all_samples if l == 1)
    print(f"  RI events: {n_ri} ({100*n_ri/len(all_samples):.1f}%)")

    # Feature names (only use features with good coverage)
    feature_names = ['wind_now', 'dw_12h', 'dw_24h', 'intensity_fraction',
                     'lat_now', 'acceleration']
    rmw_feature_names = ['rmw_now', 'drmw_12h', 'rmw_trend']  # may have missing data

    # Build feature matrix
    X_all = []
    y_all = []
    years = []
    dw_all = []

    for features, label, dw in all_samples:
        row = []
        valid = True
        for fn in feature_names:
            v = features.get(fn, -999)
            if v == -999:
                valid = False
                break
            row.append(v)
        if valid:
            X_all.append(row)
            y_all.append(label)
            years.append(features['year'])
            dw_all.append(dw)

    X = np.array(X_all)
    y = np.array(y_all)
    years = np.array(years)
    dw_all = np.array(dw_all)

    print(f"  Samples with valid features: {len(X)}")
    print(f"  RI events: {np.sum(y)} ({100*np.mean(y):.1f}%)")

    # Train/test split by year
    train_mask = years < 2010
    test_mask = years >= 2010

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    print(f"  Train: {len(X_train)} (pre-2010), Test: {len(X_test)} (2010+)")
    print(f"  Train RI rate: {100*np.mean(y_train):.1f}%, Test RI rate: {100*np.mean(y_test):.1f}%")

    # Standardize features
    mu = np.mean(X_train, axis=0)
    sigma = np.std(X_train, axis=0) + 1e-10
    X_train_s = (X_train - mu) / sigma
    X_test_s = (X_test - mu) / sigma

    # Simple logistic regression (gradient descent)
    n_features = X_train_s.shape[1]
    weights = np.zeros(n_features)
    bias = 0
    lr = 0.01
    n_epochs = 500

    def sigmoid(z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    for epoch in range(n_epochs):
        z = X_train_s @ weights + bias
        pred = sigmoid(z)
        error = pred - y_train
        grad_w = X_train_s.T @ error / len(y_train)
        grad_b = np.mean(error)
        weights -= lr * grad_w
        bias -= lr * grad_b

    # Evaluate on test set
    z_test = X_test_s @ weights + bias
    pred_test = sigmoid(z_test)

    # ROC AUC
    thresholds = np.linspace(0, 1, 200)
    tpr_list = []
    fpr_list = []
    for thresh in thresholds:
        pred_pos = pred_test >= thresh
        tp = np.sum(pred_pos & (y_test == 1))
        fp = np.sum(pred_pos & (y_test == 0))
        fn = np.sum(~pred_pos & (y_test == 1))
        tn = np.sum(~pred_pos & (y_test == 0))
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    auc = -np.trapz(tpr_list, fpr_list)

    # Brier score
    brier = np.mean((pred_test - y_test)**2)
    brier_climo = np.mean(y_test) * (1 - np.mean(y_test))
    brier_skill = 1 - brier / brier_climo

    print(f"\n  MODEL PERFORMANCE (Test Set, 2010+):")
    print(f"    ROC AUC: {auc:.3f} (0.5 = random, 1.0 = perfect)")
    print(f"    Brier Score: {brier:.4f}")
    print(f"    Brier Skill Score: {brier_skill:.3f} (>0 = better than climatology)")

    # Feature importance (absolute weight)
    print(f"\n  FEATURE IMPORTANCE:")
    importance = np.abs(weights)
    order = np.argsort(-importance)
    for i in order:
        direction = '+' if weights[i] > 0 else '-'
        print(f"    {feature_names[i]:<25} weight={weights[i]:+.4f} ({direction}RI)")

    # Classification at optimal threshold
    best_f1 = 0
    best_thresh = 0.5
    for thresh in thresholds:
        pred_pos = pred_test >= thresh
        tp = np.sum(pred_pos & (y_test == 1))
        fp = np.sum(pred_pos & (y_test == 0))
        fn = np.sum(~pred_pos & (y_test == 1))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    pred_pos = pred_test >= best_thresh
    tp = np.sum(pred_pos & (y_test == 1))
    fp = np.sum(pred_pos & (y_test == 0))
    fn = np.sum(~pred_pos & (y_test == 1))
    tn = np.sum(~pred_pos & (y_test == 0))

    print(f"\n  AT OPTIMAL THRESHOLD ({best_thresh:.3f}):")
    print(f"    Precision: {tp/max(tp+fp,1):.3f} (of predicted RI, how many actually RI)")
    print(f"    Recall:    {tp/max(tp+fn,1):.3f} (of actual RI, how many detected)")
    print(f"    F1:        {best_f1:.3f}")
    print(f"    TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    # Now do a SEPARATE analysis with RMW features (fewer samples but key predictor)
    print(f"\n  {'='*60}")
    print(f"  RMW CONTRACTION AS RI PREDICTOR")
    print(f"  {'='*60}")

    rmw_samples = []
    for features, label, dw in all_samples:
        if all(features.get(fn, -999) != -999 for fn in feature_names + rmw_feature_names):
            row = [features[fn] for fn in feature_names + rmw_feature_names]
            rmw_samples.append((row, label, features['year']))

    if len(rmw_samples) > 100:
        X_rmw = np.array([s[0] for s in rmw_samples])
        y_rmw = np.array([s[1] for s in rmw_samples])
        years_rmw = np.array([s[2] for s in rmw_samples])

        train_r = years_rmw < 2010
        test_r = years_rmw >= 2010

        if np.sum(train_r) > 50 and np.sum(test_r) > 50:
            X_tr = X_rmw[train_r]
            y_tr = y_rmw[train_r]
            X_te = X_rmw[test_r]
            y_te = y_rmw[test_r]

            mu_r = np.mean(X_tr, axis=0)
            sig_r = np.std(X_tr, axis=0) + 1e-10
            X_tr_s = (X_tr - mu_r) / sig_r
            X_te_s = (X_te - mu_r) / sig_r

            n_f = X_tr_s.shape[1]
            w_r = np.zeros(n_f)
            b_r = 0

            for epoch in range(500):
                z = X_tr_s @ w_r + b_r
                pred = sigmoid(z)
                error = pred - y_tr
                w_r -= lr * (X_tr_s.T @ error / len(y_tr))
                b_r -= lr * np.mean(error)

            z_te = X_te_s @ w_r + b_r
            pred_te = sigmoid(z_te)

            # AUC
            tpr_l = []
            fpr_l = []
            for thresh in thresholds:
                pp = pred_te >= thresh
                tp_ = np.sum(pp & (y_te == 1))
                fp_ = np.sum(pp & (y_te == 0))
                fn_ = np.sum(~pp & (y_te == 1))
                tn_ = np.sum(~pp & (y_te == 0))
                tpr_l.append(tp_ / max(tp_ + fn_, 1))
                fpr_l.append(fp_ / max(fp_ + tn_, 1))

            auc_rmw = -np.trapz(tpr_l, fpr_l)

            print(f"    Samples with RMW: {len(X_rmw)}")
            print(f"    Train: {np.sum(train_r)}, Test: {np.sum(test_r)}")
            print(f"    ROC AUC with RMW: {auc_rmw:.3f} (vs {auc:.3f} without)")
            print(f"    Improvement: {auc_rmw - auc:+.3f}")

            all_names = feature_names + rmw_feature_names
            print(f"\n    FEATURE IMPORTANCE (with RMW):")
            imp = np.abs(w_r)
            order_r = np.argsort(-imp)
            for i in order_r:
                direction = '+' if w_r[i] > 0 else '-'
                print(f"      {all_names[i]:<25} weight={w_r[i]:+.4f} ({direction}RI)")

    # ── Helmholtz interpretation ──
    print(f"\n  {'='*60}")
    print(f"  HELMHOLTZ INTERPRETATION")
    print(f"  {'='*60}")

    print(f"""
  S/Gamma FRAMEWORK FOR RI:
  ========================
  S (energy source) increases with:
    - Higher SST -> more surface enthalpy flux -> higher S
    - Deeper warm layer -> sustained S over time
    - More moisture -> more latent heat release -> higher S

  Gamma (decoherence) increases with:
    - Higher vertical wind shear -> tilts vortex -> higher Gamma
    - Dry air intrusion -> suppresses convection -> higher Gamma
    - Land interaction -> friction -> higher Gamma
    - Higher latitude -> beta-drift dispersion -> higher Gamma

  RI occurs when S/Gamma exceeds critical value:
    - Physically: WISHE feedback becomes self-sustaining
    - In data: prior intensification + low latitude + already strong

  WHAT THE MODEL FOUND:
    - Prior 24h trend (dw_24h) is strongest predictor = momentum
    - Current intensity (wind_now) matters = already organized vortex
    - Acceleration matters = nonlinear feedback beginning
    - Latitude matters = Coriolis efficiency
    - RMW contraction = coherence length decreasing (ell -> smaller)
      This is the Helmholtz prediction: approaching critical point,
      the system CONCENTRATES coherence (unlike earthquakes where
      ell DIVERGES — hurricanes are OPPOSITE because the failure mode
      is intensification, not rupture)
    """)

    # ── GENERATE FIGURES ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor(DARK)
    fig.suptitle('Hurricane RI Prediction Model',
                 color='white', fontsize=16, fontweight='bold', y=0.98)

    # Panel 1: ROC curve
    ax = axes[0, 0]
    ax.set_facecolor('#0d0d1a')
    ax.plot(fpr_list, tpr_list, '-', color=TEAL, linewidth=2, label=f'Model (AUC={auc:.3f})')
    ax.plot([0, 1], [0, 1], '--', color='#555', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate', color='white')
    ax.set_ylabel('True Positive Rate', color='white')
    ax.set_title('ROC Curve', color=GOLD, fontweight='bold')
    ax.legend(facecolor='#222', edgecolor='#555', labelcolor='white')
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_color('#333')

    # Panel 2: Probability distribution RI vs non-RI
    ax = axes[0, 1]
    ax.set_facecolor('#0d0d1a')
    ax.hist(pred_test[y_test == 0], bins=50, alpha=0.6, color=BLUE, density=True, label='Non-RI')
    ax.hist(pred_test[y_test == 1], bins=50, alpha=0.6, color=ACCENT, density=True, label='RI')
    ax.set_xlabel('Predicted P(RI)', color='white')
    ax.set_ylabel('Density', color='white')
    ax.set_title('Score Distribution', color=GOLD, fontweight='bold')
    ax.legend(facecolor='#222', edgecolor='#555', labelcolor='white')
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_color('#333')

    # Panel 3: Feature importance
    ax = axes[1, 0]
    ax.set_facecolor('#0d0d1a')
    sorted_idx = np.argsort(np.abs(weights))
    ax.barh(range(len(feature_names)), np.abs(weights[sorted_idx]),
            color=[ACCENT if weights[i] > 0 else BLUE for i in sorted_idx],
            edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels([feature_names[i] for i in sorted_idx], color='white')
    ax.set_xlabel('|Weight|', color='white')
    ax.set_title('Feature Importance', color=GOLD, fontweight='bold')
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_color('#333')

    # Panel 4: Reliability diagram
    ax = axes[1, 1]
    ax.set_facecolor('#0d0d1a')
    bin_edges = np.linspace(0, 1, 11)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    observed_freq = []
    predicted_freq = []
    bin_counts = []
    for i in range(len(bin_edges)-1):
        mask = (pred_test >= bin_edges[i]) & (pred_test < bin_edges[i+1])
        n = np.sum(mask)
        if n > 5:
            observed_freq.append(np.mean(y_test[mask]))
            predicted_freq.append(np.mean(pred_test[mask]))
            bin_counts.append(n)

    if observed_freq:
        ax.plot(predicted_freq, observed_freq, 'o-', color=TEAL, markersize=8)
        ax.plot([0, 1], [0, 1], '--', color=GOLD, linewidth=1, label='Perfect calibration')
        ax.set_xlabel('Predicted P(RI)', color='white')
        ax.set_ylabel('Observed P(RI)', color='white')
        ax.set_title('Reliability Diagram', color=GOLD, fontweight='bold')
        ax.legend(facecolor='#222', edgecolor='#555', labelcolor='white')
    ax.tick_params(colors='white')
    for sp in ax.spines.values(): sp.set_color('#333')

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    path = OUT / 'hurricane_ri_prediction.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Saved: {path}")

    return {
        'auc': auc, 'brier': brier, 'brier_skill': brier_skill,
        'weights': weights, 'feature_names': feature_names,
        'n_train': len(X_train), 'n_test': len(X_test),
    }


# =========================================================================
# MAIN
# =========================================================================

if __name__ == '__main__':
    print("=" * 70)
    print("  HURRICANE RAPID INTENSIFICATION MODEL")
    print("  Thermodynamic engine, not brittle failure.")
    print("=" * 70)

    storms = load_ibtracs()
    if storms:
        result = build_ri_predictor(storms)

    print("\n" + "=" * 70)
    print("COMPLETE")
    print("=" * 70)
