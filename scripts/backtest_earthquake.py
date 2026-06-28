#!/usr/bin/env python3
"""Rigorous backtest of the earthquake nowcast: is the skill LEGIT and is it an EDGE?

In earthquake forecasting the honest bar is NOT 0.5 -- "it shakes where it recently
shook" (clustering / smoothed seismicity) is a strong baseline. A model that doesn't
clearly beat that has no edge. This harness measures, on the held-out test set:

  * climatology (base rate) -- the floor
  * persistence -- the single best recent-rate feature used directly as the score
    (the smoothed-seismicity baseline)
  * best single feature overall -- the strongest one-variable predictor
  * the ML model, multi-seed, with a bootstrap CI
  * a PAIRED bootstrap CI on (model - persistence) -- the actual edge, with significance
  * an ablation: Block S (seismicity) vs +Block C (coherence) vs +Block X
  * calibration (reliability) on the holdout

Loads the cached features, so it runs in a couple of minutes.

    python scripts/backtest_earthquake.py --seeds 5 --max-year 2024
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
_spec = importlib.util.spec_from_file_location("tbt", REPO / "scripts" / "train_best_tabular.py")
_tbt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tbt)
roc_auc, brier = _tbt.roc_auc, _tbt.brier


def _auc_oriented(y, s):
    a = roc_auc(y, s)
    return a if a >= 0.5 else 1.0 - a   # |AUC| for a single raw feature (orientation-free)


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


def _reliability(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum() >= 5:
            out.append({"p_pred": round(float(p[m].mean()), 3),
                        "p_obs": round(float(y[m].mean()), 3), "n": int(m.sum())})
    return out


def _test_meta(max_year):
    """Re-derive (year, lat, lon) for each TEST sample in the SAME order as the cached
    features. Mirrors load_all_data's deterministic construction (Gardner-Knopoff +
    build_samples with a fixed seed; test is NOT downsampled), so it aligns row-for-row
    with the cached X_test -- without re-running the expensive feature extraction.
    """
    from hazardpulse.earthquake.definitive_model import (
        load_usgs_catalog, decluster_gardner_knopoff, build_samples,
        TEST_START, TEST_END)
    catalog = load_usgs_catalog(min_year=2000, max_year=max_year, min_mag=2.5)
    mainshocks, _ = decluster_gardner_knopoff(catalog)
    samples = build_samples(mainshocks, catalog, verbose=False)
    test_end = max(TEST_END, int(max_year))
    test = [s for s in samples if TEST_START <= s["year"] <= test_end]
    return (np.array([s["year"] for s in test]),
            np.array([s["latitude"] for s in test], float),
            np.array([s["longitude"] for s in test], float))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--max-year", type=int, default=2024)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--breakdown", action="store_true",
                    help="per-year (walk-forward) + per-region AUC (re-derives sample "
                         "metadata; ~6 min for declustering, no feature extraction)")
    ap.add_argument("--out", default="results/calibration/earthquake_backtest.json")
    args = ap.parse_args(argv)

    from hazardpulse.earthquake.definitive_model import BLOCK_S_NAMES, BLOCK_C_NAMES
    Xtr, Xval, Xte, ytr, yval, yte = _tbt._load_all_eq_cached(verbose=True, max_year=args.max_year)
    enh = "enhanced"
    Xt = np.vstack([Xtr[enh], Xval[enh]]); yt = np.concatenate([ytr, yval])
    Xe, ye = Xte[enh], np.asarray(yte).astype(int)
    names = list(BLOCK_S_NAMES) + list(BLOCK_C_NAMES)
    print(f"train={len(yt)} test={len(ye)} ({int(ye.sum())} pos, base rate {ye.mean():.3f})")

    rep = {"max_year": args.max_year, "n_train": int(len(yt)), "n_test": int(len(ye)),
           "test_base_rate": round(float(ye.mean()), 4)}

    # --- baselines on the test set --------------------------------------------- #
    rate_feats = [n for n in BLOCK_S_NAMES if any(k in n.lower()
                  for k in ("rate", "n_7d", "n_14d", "n_30d", "n_90d", "n_events"))]
    rate_idx = [names.index(n) for n in rate_feats]
    pers = [(n, _auc_oriented(ye, Xe[:, names.index(n)])) for n in rate_feats]
    pers.sort(key=lambda kv: kv[1], reverse=True)
    best_pers_name, best_pers_auc = pers[0]
    single = [(n, _auc_oriented(ye, Xe[:, i])) for i, n in enumerate(names)]
    single.sort(key=lambda kv: kv[1], reverse=True)
    rep["persistence"] = {"best_feature": best_pers_name, "auc": round(best_pers_auc, 4),
                          "top5_rate_feats": [(n, round(a, 4)) for n, a in pers[:5]]}
    rep["best_single_feature"] = {"feature": single[0][0], "auc": round(single[0][1], 4),
                                  "top5": [(n, round(a, 4)) for n, a in single[:5]]}
    print(f"  persistence (best rate feat '{best_pers_name}'): AUC {best_pers_auc:.4f}")
    print(f"  best single feature      '{single[0][0]}': AUC {single[0][1]:.4f}")

    # --- model, multi-seed ----------------------------------------------------- #
    seed_aucs, model_proba = [], None
    for s in range(args.seeds):
        m = _fit_xgb(Xt, yt, s)
        p = np.asarray(m.predict_proba(np.asarray(Xe, float)))[:, 1]
        seed_aucs.append(roc_auc(ye, p))
        if model_proba is None:
            model_proba = p
    rep["model"] = {"auc_mean": round(float(np.mean(seed_aucs)), 4),
                    "auc_std": round(float(np.std(seed_aucs)), 4),
                    "auc_seeds": [round(a, 4) for a in seed_aucs],
                    "brier": round(brier(ye, model_proba), 4)}
    print(f"  MODEL (xgboost x{args.seeds} seeds): AUC {rep['model']['auc_mean']:.4f} "
          f"+/- {rep['model']['auc_std']:.4f}")

    # --- ablation: does coherence (Block C) / Block X add over seismicity? ------ #
    abl = {}
    for variant in ("baseline", "enhanced", "full"):
        Xtv = np.vstack([Xtr[variant], Xval[variant]])
        m = _fit_xgb(Xtv, yt, 0)
        p = np.asarray(m.predict_proba(np.asarray(Xte[variant], float)))[:, 1]
        abl[variant] = round(roc_auc(ye, p), 4)
    rep["ablation"] = {"S_only": abl["baseline"], "S+C": abl["enhanced"], "S+C+X": abl["full"]}
    print(f"  ablation: S={abl['baseline']:.4f}  S+C={abl['enhanced']:.4f}  S+C+X={abl['full']:.4f}")

    # --- paired bootstrap: model - persistence (is the edge real?) -------------- #
    pers_score = Xe[:, names.index(best_pers_name)].astype(float)
    if roc_auc(ye, pers_score) < 0.5:
        pers_score = -pers_score   # orient
    rng = np.random.RandomState(0); n = len(ye); diffs = []; m_aucs = []
    for _ in range(args.boot):
        idx = rng.randint(0, n, n)
        yb = ye[idx]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        ma = roc_auc(yb, model_proba[idx]); pa = roc_auc(yb, pers_score[idx])
        diffs.append(ma - pa); m_aucs.append(ma)
    diffs = np.array(diffs); m_aucs = np.array(m_aucs)
    d_lo, d_hi = np.percentile(diffs, [2.5, 97.5])
    rep["model_auc_ci95"] = [round(float(np.percentile(m_aucs, 2.5)), 4),
                             round(float(np.percentile(m_aucs, 97.5)), 4)]
    rep["edge_over_persistence"] = {
        "delta_auc": round(float(np.mean(diffs)), 4),
        "ci95": [round(float(d_lo), 4), round(float(d_hi), 4)],
        "significant": bool(d_lo > 0)}
    print(f"  model AUC 95% CI: {rep['model_auc_ci95']}")
    print(f"  EDGE over persistence: {rep['edge_over_persistence']['delta_auc']:+.4f} "
          f"CI {rep['edge_over_persistence']['ci95']} -> "
          f"{'SIGNIFICANT' if rep['edge_over_persistence']['significant'] else 'NOT significant'}")

    rep["calibration_reliability"] = _reliability(ye, model_proba)

    # --- walk-forward (per-year) + per-region breakdown ------------------------ #
    if args.breakdown:
        print("  deriving test-sample metadata (declustering, no feature extraction)...")
        years, lats, lons = _test_meta(args.max_year)
        if len(years) == len(ye):
            from hazardpulse.trust.group_conformal import geo_region
            by_year = {}
            for y in sorted(set(int(v) for v in years)):
                m = years == y
                if m.sum() >= 20 and 0 < ye[m].sum() < m.sum():
                    by_year[int(y)] = {"auc": round(roc_auc(ye[m], model_proba[m]), 4),
                                       "n": int(m.sum()), "pos": int(ye[m].sum())}
            regions = np.array([geo_region(la, lo) for la, lo in zip(lats, lons)])
            by_region = {}
            for r in sorted(set(regions)):
                m = regions == r
                if m.sum() >= 20 and 0 < ye[m].sum() < m.sum():
                    pers = Xe[:, names.index(best_pers_name)].astype(float)
                    if roc_auc(ye, pers) < 0.5:
                        pers = -pers
                    by_region[r] = {"auc": round(roc_auc(ye[m], model_proba[m]), 4),
                                    "persistence_auc": round(roc_auc(ye[m], pers[m]), 4),
                                    "n": int(m.sum()), "pos": int(ye[m].sum())}
            rep["walk_forward_by_year"] = by_year
            rep["by_region"] = by_region
            print("  walk-forward (per test year): " +
                  "  ".join(f"{y}:{d['auc']:.3f}" for y, d in by_year.items()))
            print("  per-region AUC (model vs persistence):")
            for r, d in by_region.items():
                print(f"    {r:16s} model {d['auc']:.3f}  pers {d['persistence_auc']:.3f}  (n={d['n']})")
        else:
            print(f"  WARNING: meta length {len(years)} != test {len(ye)}; skipping breakdown.")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
