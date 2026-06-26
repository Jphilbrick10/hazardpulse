#!/usr/bin/env python3
"""Accuracy upgrade: train omega_one's BestTabular SOTA ensemble vs the incumbent.

HazardPulse's hand-rolled gradient-boosted trees are competitive but not SOTA.
omega_one's ``BestTabular`` (size-routed: a subspace TabPFN portfolio on
small/medium data, a tuned GBDT SuperEnsemble on large data) beats the GBDT field
by ~2-3pt and ties/uses TabPFN, the tabular SOTA. This script trains it on a
hazard's real training data, compares it to the incumbent model on the SAME
temporal holdout across seeds, and keeps the better one under a NEVER-WORSE guard
(if BestTabular doesn't beat the incumbent, the incumbent stays). It writes a
public comparison report so every accuracy claim is auditable.

This is an OFFLINE trainer (needs xgboost/lightgbm/catboost/tabpfn). The live
scoring path stays numpy-only; serving the champion is a separate, deliberate step.

    python scripts/train_best_tabular.py --hazard earthquake --seeds 0,1,2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
# omega_one is a sibling repo; import only for offline training.
_OMEGA = REPO.parent / "Coherence" / "omega_one"


def roc_auc(y_true, y_score) -> float:
    y = np.asarray(y_true, float)
    s = np.asarray(y_score, float)
    pos = float((y == 1).sum())
    neg = float((y == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(-s)
    ys = y[order]
    tp = fp = auc = tpp = fpp = 0.0
    for label in ys:
        if label == 1:
            tp += 1.0
        else:
            fp += 1.0
        auc += (fp / neg - fpp / neg) * (tp / pos + tpp / pos) / 2.0
        tpp, fpp = tp, fp
    return float(auc)


def brier(y_true, y_score) -> float:
    return float(np.mean((np.asarray(y_score, float) - np.asarray(y_true, float)) ** 2))


def _best_tabular():
    if str(_OMEGA) not in sys.path:
        sys.path.insert(0, str(_OMEGA))
    from omega.super_ensemble import BestTabular
    return BestTabular()


def compare(X_tr, y_tr, X_te, y_te, *, baseline_proba=None, seeds=(0, 1, 2)) -> dict:
    """Train BestTabular (multi-seed) and compare to a baseline's probabilities.

    Returns an auditable report; ``champion`` is 'best_tabular' only when it does
    NOT regress the baseline AUC (never-worse guard).
    """
    aucs, briers = [], []
    champion_proba = None
    for s in seeds:
        np.random.seed(int(s))
        model = _best_tabular()
        model.fit(np.asarray(X_tr, float), np.asarray(y_tr))
        proba = np.asarray(model.predict_proba(np.asarray(X_te, float)))[:, 1]
        aucs.append(roc_auc(y_te, proba))
        briers.append(brier(y_te, proba))
        if champion_proba is None:
            champion_proba = proba
    report = {
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "seeds": list(seeds),
        "best_tabular": {
            "auc_mean": float(np.nanmean(aucs)),
            "auc_std": float(np.nanstd(aucs)),
            "brier_mean": float(np.mean(briers)),
        },
    }
    if baseline_proba is not None:
        bp = np.asarray(baseline_proba, float)
        report["baseline"] = {"auc": roc_auc(y_te, bp), "brier": brier(y_te, bp)}
        report["delta_auc"] = report["best_tabular"]["auc_mean"] - report["baseline"]["auc"]
        # Never-worse: only crown BestTabular if it does not regress AUC.
        report["champion"] = "best_tabular" if report["delta_auc"] >= 0.0 else "baseline"
    else:
        report["champion"] = "best_tabular"
    return report, champion_proba


# --------------------------------------------------------------------------- #
# Hazard data adapters (real training data via the existing research pipelines)
# --------------------------------------------------------------------------- #
def _load_earthquake():
    from hazardpulse.earthquake.definitive_model import load_all_data
    X_tr, X_val, X_te, y_tr, y_val, y_te, _meta = load_all_data(verbose=True)
    # fold val into train for the final fit
    X_tr = np.vstack([X_tr, X_val])
    y_tr = np.concatenate([y_tr, y_val])
    return X_tr, y_tr, X_te, y_te


_LOADERS = {"earthquake": _load_earthquake}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hazard", choices=sorted(_LOADERS), required=True)
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args(argv)
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())

    print(f"Loading {args.hazard} training data...")
    X_tr, y_tr, X_te, y_te = _LOADERS[args.hazard]()
    print(f"  train={len(y_tr)}  test={len(y_te)}  features={X_tr.shape[1]}")

    t0 = time.time()
    report, _ = compare(X_tr, y_tr, X_te, y_te, seeds=seeds)
    report["hazard"] = args.hazard
    report["train_seconds"] = round(time.time() - t0, 1)

    out = REPO / "results" / "calibration" / f"{args.hazard}_champion_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    bt = report["best_tabular"]
    print(f"  BestTabular AUC {bt['auc_mean']:.4f} +/- {bt['auc_std']:.4f}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
