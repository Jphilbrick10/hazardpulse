#!/usr/bin/env python3
"""Test whether NEW signal sources add real, significant skill to the EQ nowcast.

The controls are same-location/different-time, so only TIME-VARYING-at-location signals
can help. This harness reconstructs each cached sample's (lat, lon, epoch) -- replicating
the deterministic split + downsample -- then bolts candidate features onto the existing
73 and measures the held-out AUC delta with a PAIRED bootstrap (is the lift real?).

Candidates (100% coverage, global, cheap):
  * tidal   -- solid-earth tidal forcing phases at the event time (fortnightly/monthly/
               semidiurnal). Tidal triggering is a known ~1% effect; test if it shows.
  * natclock-- "natural time" seismic clock: time + small-quake count since the last
               M>=5 within 150 km (a recurrence clock), strictly causal.

    python scripts/backtest_augment_earthquake.py --candidates tidal natclock
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
_spec = importlib.util.spec_from_file_location("tbt", REPO / "scripts" / "train_best_tabular.py")
_tbt = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_tbt)
roc_auc, brier = _tbt.roc_auc, _tbt.brier

_SEC_DAY = 86400.0
_LUNAR_SYNODIC = 29.530589 * _SEC_DAY     # new-moon to new-moon
_LUNAR_ANOM = 27.554550 * _SEC_DAY        # perigee cycle
# reference new moon 2000-01-06 18:14 UTC (epoch seconds)
_REF_NEW_MOON = 947182440.0


def _fit_xgb(X, y, seed):
    from xgboost import XGBClassifier
    kw = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.7,
              colsample_bytree=0.7, reg_lambda=1.0, eval_metric="logloss",
              tree_method="hist", random_state=int(seed))
    X = np.asarray(X, float); y = np.asarray(y)
    try:
        m = XGBClassifier(device="cuda", **kw); m.fit(X, y); return m
    except Exception:
        m = XGBClassifier(**kw); m.fit(X, y); return m


def _all_meta(max_year):
    """(lat, lon, epoch) for train(downsampled)+val+test in EXACT cached order."""
    from hazardpulse.earthquake.definitive_model import (
        load_usgs_catalog, decluster_gardner_knopoff, build_samples, CatalogArrays,
        TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END)
    catalog = load_usgs_catalog(min_year=2000, max_year=max_year, min_mag=2.5)
    cat_arrays = CatalogArrays(catalog, verbose=False)   # for fast causal queries
    mainshocks, _ = decluster_gardner_knopoff(catalog)
    samples = build_samples(mainshocks, catalog, verbose=False)
    test_end = max(TEST_END, int(max_year))
    tr = [s for s in samples if TRAIN_START <= s["year"] <= TRAIN_END]
    va = [s for s in samples if VAL_START <= s["year"] <= VAL_END]
    te = [s for s in samples if TEST_START <= s["year"] <= test_end]
    # replicate the 5:1 train downsample (RandomState(42)) so train aligns with the cache
    ytr = np.array([s["label"] for s in tr])
    n_pos = int(ytr.sum()); n_neg = int(len(ytr) - n_pos)
    if n_neg > 5 * n_pos:
        rng = np.random.RandomState(42)
        pos_idx = np.where(ytr == 1)[0]; neg_idx = np.where(ytr == 0)[0]
        keep = np.sort(np.concatenate([pos_idx, rng.choice(neg_idx, 5 * n_pos, replace=False)]))
        tr = [tr[i] for i in keep]

    def arr(ss):
        return (np.array([s["latitude"] for s in ss], float),
                np.array([s["longitude"] for s in ss], float),
                np.array([s["ref_epoch"] for s in ss], float),
                np.array([s["label"] for s in ss], int))
    return arr(tr), arr(va), arr(te), cat_arrays


def _tidal_features(lat, lon, epoch):
    ph_syn = 2 * math.pi * ((epoch - _REF_NEW_MOON) % _LUNAR_SYNODIC) / _LUNAR_SYNODIC
    ph_fort = 2 * ph_syn                                   # fortnightly (spring/neap)
    ph_anom = 2 * math.pi * ((epoch - _REF_NEW_MOON) % _LUNAR_ANOM) / _LUNAR_ANOM
    ph_sd = 2 * math.pi * ((epoch % _SEC_DAY) / _SEC_DAY)  # ~semidiurnal proxy
    return [math.cos(ph_fort), math.sin(ph_fort), math.cos(ph_anom),
            math.sin(ph_anom), math.cos(ph_sd), math.sin(ph_sd)]


def _natclock_features(cat, lat, lon, epoch):
    """Causal seismic clock: days + small-quake count since the last M>=5 within 150 km."""
    from hazardpulse.earthquake.definitive_model import haversine_vec
    box = 2.0
    m = ((cat.times < epoch) & (np.abs(cat.lats - lat) < box) & (np.abs(cat.lons - lon) < box))
    if not m.any():
        return [9999.0, 0.0, 0.0]
    d = haversine_vec(lat, lon, cat.lats[m], cat.lons[m])
    near = d < 150.0
    if not near.any():
        return [9999.0, 0.0, 0.0]
    t = cat.times[m][near]; mg = cat.mags[m][near]
    big = t[mg >= 5.0]
    last_big = big.max() if big.size else t.min()
    days_since = (epoch - last_big) / _SEC_DAY
    count_since = int((t > last_big).sum())               # natural-time count
    return [min(days_since, 9999.0), float(count_since), float(np.log1p(count_since))]


def _teleseismic_features(cat, epoch):
    """Global dynamic-triggering proxy: causal global big-quake activity before ev_time.
    days since last global M>=7; counts of global M>=6 / M>=7 in the prior 30 days."""
    m = cat.times < epoch
    if not m.any():
        return [9999.0, 0.0, 0.0]
    t = cat.times[m]; mg = cat.mags[m]
    big = t[mg >= 7.0]
    days_since_m7 = (epoch - big.max()) / _SEC_DAY if big.size else 9999.0
    w30 = t > (epoch - 30 * _SEC_DAY)
    return [min(days_since_m7, 9999.0),
            float((w30 & (mg >= 6.0)).sum()), float((w30 & (mg >= 7.0)).sum())]


def _candidate_matrix(name, metas, cat):
    rows = []
    for (lat, lon, ep, _y) in metas:
        for i in range(len(lat)):
            if name == "tidal":
                rows.append(_tidal_features(lat[i], lon[i], ep[i]))
            elif name == "teleseismic":
                rows.append(_teleseismic_features(cat, ep[i]))
            else:
                rows.append(_natclock_features(cat, lat[i], lon[i], ep[i]))
    return np.array(rows, float)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", nargs="+", default=["tidal", "natclock"])
    ap.add_argument("--max-year", type=int, default=2024)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", default="results/calibration/earthquake_augment.json")
    args = ap.parse_args(argv)

    Xtr, Xval, Xte, ytr, yval, yte = _tbt._load_all_eq_cached(verbose=False, max_year=args.max_year)
    enh = "enhanced"
    Xt = np.vstack([Xtr[enh], Xval[enh]]); yt = np.concatenate([ytr, yval])
    Xe, ye = np.asarray(Xte[enh], float), np.asarray(yte).astype(int)
    print(f"cache: train={len(yt)} test={len(ye)} feat={Xt.shape[1]}")

    print("reconstructing per-sample metadata (declustering, no extraction)...")
    (mtr, mva, mte, cat) = _all_meta(args.max_year)
    n_meta = len(mtr[0]) + len(mva[0])
    if n_meta != len(yt) or len(mte[0]) != len(ye):
        print(f"  META MISALIGNED: train+val meta {n_meta} vs cache {len(yt)}, "
              f"test {len(mte[0])} vs {len(ye)}. Aborting (would corrupt the test).")
        return 1
    # sanity: reconstructed labels must match the cached labels exactly
    if not (np.array_equal(np.concatenate([mtr[3], mva[3]]), yt) and np.array_equal(mte[3], ye)):
        print("  LABELS MISMATCH between reconstruction and cache. Aborting.")
        return 1
    print("  metadata aligned + labels verified.")

    base = [_fit_xgb(Xt, yt, s) for s in range(3)]
    base_p = np.mean([np.asarray(m.predict_proba(Xe))[:, 1] for m in base], axis=0)
    base_auc = roc_auc(ye, base_p)
    print(f"  baseline (73 feat) AUC {base_auc:.4f}")

    rep = {"max_year": args.max_year, "baseline_auc": round(base_auc, 4), "candidates": {}}
    for name in args.candidates:
        print(f"  computing '{name}' features...")
        Ctr = _candidate_matrix(name, [mtr, mva], cat)
        Cte = _candidate_matrix(name, [mte], cat)
        Xt2 = np.hstack([Xt, Ctr]); Xe2 = np.hstack([Xe, Cte])
        aug = [_fit_xgb(Xt2, yt, s) for s in range(3)]
        aug_p = np.mean([np.asarray(m.predict_proba(Xe2))[:, 1] for m in aug], axis=0)
        aug_auc = roc_auc(ye, aug_p)
        # paired bootstrap on (augmented - baseline)
        rng = np.random.RandomState(0); n = len(ye); diffs = []
        for _ in range(args.boot):
            idx = rng.randint(0, n, n); yb = ye[idx]
            if 0 < yb.sum() < len(yb):
                diffs.append(roc_auc(yb, aug_p[idx]) - roc_auc(yb, base_p[idx]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        rep["candidates"][name] = {"aug_auc": round(aug_auc, 4),
                                   "delta_auc": round(aug_auc - base_auc, 4),
                                   "ci95": [round(float(lo), 4), round(float(hi), 4)],
                                   "significant": bool(lo > 0), "n_new_feats": Ctr.shape[1]}
        verdict = "ADDS SIGNAL" if lo > 0 else ("hurts" if hi < 0 else "no significant effect")
        print(f"    {name}: AUC {base_auc:.4f} -> {aug_auc:.4f} "
              f"(delta {aug_auc-base_auc:+.4f}, CI [{lo:+.4f},{hi:+.4f}]) -> {verdict}")

    out = REPO / args.out; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
