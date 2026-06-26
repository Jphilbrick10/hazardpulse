#!/usr/bin/env python3
"""Measure tornado-environment skill from the self-contained HRRR dataset.

Loads the .npz built by build_tornado_hrrr_dataset.py, does an HONEST temporal
holdout (train on earlier dates, test on later -- no leakage across the same
outbreak), and reports AUC + Brier for the incumbent-style GBT vs the SOTA
candidates. The bar is the live tornado ceiling (~0.58-0.64): does real,
peak-pooled HRRR environment data discriminate tornadic from non-tornadic
convective cells better than that?

    python scripts/train_tornado_hrrr.py --data .cache/tornado/hrrr_env_dataset.npz
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
sys.path.insert(0, str(REPO.parent / "Coherence" / "omega_one"))

_tbt_spec = importlib.util.spec_from_file_location("tbt", REPO / "scripts" / "train_best_tabular.py")
_tbt = importlib.util.module_from_spec(_tbt_spec)
_tbt_spec.loader.exec_module(_tbt)
roc_auc, brier = _tbt.roc_auc, _tbt.brier


def _temporal_split(X, y, dates, test_frac=0.25):
    uniq = np.array(sorted(set(int(d) for d in dates)))
    cut = uniq[int((1.0 - test_frac) * len(uniq))]
    tr = dates < cut
    te = dates >= cut
    return X[tr], y[tr], X[te], y[te], int(cut)


def _impute(Xtr, Xte):
    mean = np.nanmean(Xtr, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    Xtr = np.where(np.isfinite(Xtr), Xtr, mean)
    Xte = np.where(np.isfinite(Xte), Xte, mean)
    return Xtr.astype(float), Xte.astype(float)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=".cache/tornado/hrrr_env_dataset.npz")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--out", default="results/calibration/tornado_hrrr_report.json")
    args = ap.parse_args(argv)

    d = np.load(REPO / args.data if not Path(args.data).is_absolute() else args.data,
                allow_pickle=True)
    X, y, dates = d["X"], d["y"].astype(int), d["dates"]
    names = list(d["feature_names"])
    print(f"loaded {len(y)} cells  ({int(y.sum())} tornado / {len(y)-int(y.sum())} null)  "
          f"feat={X.shape[1]}  dates={len(set(int(x) for x in dates))}")

    Xtr, ytr, Xte, yte, cut = _temporal_split(X, y, dates, args.test_frac)
    Xtr, Xte = _impute(Xtr, Xte)
    print(f"temporal split @ {cut}: train={len(ytr)} ({int(ytr.sum())} tor)  "
          f"test={len(yte)} ({int(yte.sum())} tor)")
    if ytr.sum() < 5 or yte.sum() < 5 or len(set(yte)) < 2:
        print("  not enough tornado samples in a split for an honest AUC yet.")
        return 0

    report = {"n": int(len(y)), "n_pos": int(y.sum()), "n_features": int(X.shape[1]),
              "split_cut": cut, "n_test": int(len(yte)), "live_ceiling": 0.64}

    # incumbent-style GBD baseline (xgboost CPU/GPU via the shared fitters)
    base = _tbt._fit_xgboost(Xtr, ytr, 0)
    pbase = np.asarray(base.predict_proba(Xte))[:, 1]
    report["xgboost"] = {"auc": roc_auc(yte, pbase), "brier": brier(yte, pbase)}
    print(f"  xgboost      AUC {report['xgboost']['auc']:.4f}  Brier {report['xgboost']['brier']:.4f}")

    # servable + signed VerifiableForest (best of xgb/lgbm, honest selection)
    try:
        vf = _tbt._verifiable_forest(Xtr, ytr, Xte, seed=0)
        report["verifiable_forest"] = {
            "auc": roc_auc(yte, vf["proba"]), "brier": brier(yte, vf["proba"]),
            "booster": vf["booster"], "self_reproduce": vf["self_reproduce"],
            "bit_exact": vf["bit_exact"], "n_trees": vf["n_trees"],
        }
        print(f"  forest[{vf['booster']}] AUC {report['verifiable_forest']['auc']:.4f}  "
              f"Brier {report['verifiable_forest']['brier']:.4f}  (servable+signed)")
    except Exception as exc:
        print(f"  verifiable_forest skipped: {exc}")

    # SOTA ceiling
    try:
        from omega.super_ensemble import BestTabular
        bt = BestTabular(); bt.fit(Xtr, ytr)
        pbt = np.asarray(bt.predict_proba(Xte))[:, 1]
        report["best_tabular"] = {"auc": roc_auc(yte, pbt), "brier": brier(yte, pbt)}
        print(f"  BestTabular  AUC {report['best_tabular']['auc']:.4f}  "
              f"Brier {report['best_tabular']['brier']:.4f}  (ceiling)")
    except Exception as exc:
        print(f"  best_tabular skipped: {exc}")

    best_auc = max(v["auc"] for k, v in report.items()
                   if isinstance(v, dict) and "auc" in v)
    report["best_auc"] = best_auc
    report["beats_live_ceiling"] = bool(best_auc > 0.64)
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    verdict = "BEATS" if best_auc > 0.64 else "does NOT beat"
    print(f"\n  best AUC {best_auc:.4f} -> {verdict} the live ceiling (~0.64)")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
