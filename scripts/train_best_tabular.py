#!/usr/bin/env python3
"""Accuracy upgrade: omega_one SOTA candidates vs HazardPulse's incumbent model.

HazardPulse's hand-rolled gradient-boosted trees are competitive but not SOTA. This
trainer measures two omega_one candidates against the incumbent on the SAME temporal
holdout, multi-seed, under a NEVER-WORSE guard, and writes an auditable report:

  * ``BestTabular`` -- the accuracy CEILING (size-routed TabPFN-subspace / tuned GBDT
    SuperEnsemble; beats the GBDT field ~2-3pt). Needs the heavy SOTA libs at
    inference, so it is a yardstick, not directly servable.
  * ``VerifiableForest`` -- the DEPLOYABLE champion: an XGBoost fit FROZEN into a
    portable integer forest whose JSON ``fp_constants`` a tiny numpy reproducer
    re-runs bit-for-bit. It serves in HazardPulse's numpy-only live path AND emits a
    0-ULP Ed25519-signed receipt. When it beats the incumbent it is exported to
    ``results/calibration/<hazard>_forest_fp.json`` for the live scorer to load.

This is an OFFLINE trainer (needs xgboost/lightgbm/catboost/tabpfn). The live scoring
path stays numpy-only; it loads only the frozen forest constants.

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


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


def _forest_proba(constants, F):
    """P(event) from the frozen forest's OWN integer decision function (binary).

    ``F[:, 1]`` is base + summed class-1 margins x scale; the served probability is
    sigmoid(margin / scale). This is the probability the live path actually emits --
    we measure and serve the frozen forest itself, not its xgboost origin.
    """
    scale = float(constants["scale"])
    margin = (np.asarray(F, float)[:, 1] - np.asarray(F, float)[:, 0]) / scale
    return _sigmoid(margin)


def _verifiable_forest(X_tr, y_tr, X_te, *, seed=0):
    """Fit XGBoost, then FREEZE it into a portable integer forest (omega's
    VerifiableForest). The frozen ``fp_constants`` is a pure-JSON model a numpy
    reproducer re-runs bit-for-bit -- so THIS frozen forest is the champion that
    serves in HazardPulse's numpy-only live path AND emits a 0-ULP signed receipt,
    unlike BestTabular (which needs the heavy SOTA libs at inference).

    Accuracy is measured on the FROZEN FOREST's own probabilities (what serves).
    ``self_reproduce`` proves a third party holding ONLY the round-tripped JSON
    constants reproduces the decision exactly (the servability guarantee).
    ``xgb_agreement`` (how often the frozen forest's argmax matches xgboost's
    float32 path) is an informational diagnostic, not a correctness gate -- the two
    differ only at float32-vs-float64 split boundaries.
    """
    if str(_OMEGA) not in sys.path:
        sys.path.insert(0, str(_OMEGA))
    from xgboost import XGBClassifier
    from omega.verifiable_forest import VerifiableForest, predict_fp_from_forest_constants

    Xtr = np.asarray(X_tr, float); Xte = np.asarray(X_te, float)
    clf = XGBClassifier(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.7, reg_lambda=1.0,
        eval_metric="logloss", tree_method="hist", random_state=int(seed),
    )
    clf.fit(Xtr, np.asarray(y_tr))

    vf = VerifiableForest.from_xgboost(clf)
    constants = vf.fp_constants()

    # The served probability comes from the FROZEN forest's own decision function.
    labels_self, F = predict_fp_from_forest_constants(constants, Xte)
    proba = _forest_proba(constants, F)

    # Servability proof: a third party with ONLY the JSON file reproduces the decision.
    constants_roundtrip = json.loads(json.dumps(constants))
    labels_3p, F_3p = predict_fp_from_forest_constants(constants_roundtrip, Xte)
    self_reproduce = float(np.mean(np.asarray(labels_3p) == np.asarray(labels_self))
                           ) if len(labels_self) else 1.0
    bit_exact = bool(np.array_equal(F_3p, F))

    # Informational: agreement with xgboost's own float32 prediction path.
    xgb_agreement = float(np.mean(np.asarray(labels_self) == np.asarray(clf.predict(Xte))))
    return {
        "proba": proba,
        "self_reproduce": self_reproduce,
        "bit_exact": bit_exact,
        "xgb_agreement": xgb_agreement,
        "fp_sha256": vf.fp_model_sha256(),
        "constants": constants,
        "n_trees": int(len(constants["tree_root"])),
    }


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
def _load_earthquake(variant: str = "full"):
    from hazardpulse.earthquake.definitive_model import load_all_data
    X_tr, X_val, X_te, y_tr, y_val, y_te, _meta = load_all_data(verbose=True)
    # load_all_data returns {"baseline","enhanced","full"} -> pick the full feature set
    Xtr = np.vstack([X_tr[variant], X_val[variant]])    # fold val into train for the final fit
    y_tr = np.concatenate([y_tr, y_val])
    return Xtr, y_tr, X_te[variant], y_te


def _baseline_earthquake(X_tr, y_tr, X_te):
    """Train the incumbent pure-numpy GBT on the SAME split -> its test probabilities.

    This is the honest never-worse baseline: BestTabular only deploys if it beats the
    model HazardPulse ships today, measured on the identical temporal holdout.
    """
    from hazardpulse.earthquake import definitive_model as dm
    gbt = dm.GradientBoostedTrees(
        n_trees=dm.GBT_N_TREES, max_depth=dm.GBT_MAX_DEPTH,
        learning_rate=dm.GBT_LEARNING_RATE, min_samples_leaf=dm.GBT_MIN_SAMPLES_LEAF,
        subsample=dm.GBT_SUBSAMPLE, colsample=dm.GBT_COLSAMPLE,
        l2_reg=dm.GBT_L2_REG, gamma=dm.GBT_GAMMA,
    )
    gbt.fit(np.asarray(X_tr, np.float32), np.asarray(y_tr), verbose=False)
    return np.asarray(gbt.predict_proba(np.asarray(X_te, np.float32))).ravel()


_LOADERS = {"earthquake": _load_earthquake}
_BASELINES = {"earthquake": _baseline_earthquake}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hazard", choices=sorted(_LOADERS), required=True)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--no-forest", action="store_true",
                        help="skip the VerifiableForest (servable) candidate")
    args = parser.parse_args(argv)
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())

    print(f"Loading {args.hazard} training data...")
    X_tr, y_tr, X_te, y_te = _LOADERS[args.hazard]()
    print(f"  train={len(y_tr)}  test={len(y_te)}  features={X_tr.shape[1]}")

    baseline_proba = None
    if args.hazard in _BASELINES:
        print("Training incumbent baseline on the same holdout...")
        tb = time.time()
        baseline_proba = _BASELINES[args.hazard](X_tr, y_tr, X_te)
        print(f"  incumbent AUC {roc_auc(y_te, baseline_proba):.4f}  ({time.time()-tb:.0f}s)")
    base_auc = roc_auc(y_te, baseline_proba) if baseline_proba is not None else None

    t0 = time.time()
    report, _bt_proba = compare(X_tr, y_tr, X_te, y_te,
                                baseline_proba=baseline_proba, seeds=seeds)
    report["hazard"] = args.hazard

    # VerifiableForest: the SERVABLE, SIGNABLE candidate (frozen integer forest).
    forest = None
    if not args.no_forest:
        print("Training VerifiableForest (XGBoost -> frozen integer forest)...")
        tf = time.time()
        forest = _verifiable_forest(X_tr, y_tr, X_te, seed=seeds[0])
        vf_auc = roc_auc(y_te, forest["proba"])
        report["verifiable_forest"] = {
            "auc": vf_auc,
            "brier": brier(y_te, forest["proba"]),
            "self_reproduce": forest["self_reproduce"],   # JSON-only third party reproduces it
            "bit_exact": forest["bit_exact"],
            "xgb_agreement": forest["xgb_agreement"],     # informational diagnostic
            "fp_sha256": forest["fp_sha256"],
            "n_trees": forest["n_trees"],
            "delta_auc_vs_baseline": (vf_auc - base_auc) if base_auc is not None else None,
            "servable": True,   # re-runs from JSON constants in the numpy live path
        }
        print(f"  VerifiableForest AUC {vf_auc:.4f}  self_reproduce {forest['self_reproduce']:.4f}  "
              f"bit_exact {forest['bit_exact']}  ({time.time()-tf:.0f}s)")

    # Deployable champion = the best candidate that BEATS the incumbent AND can serve.
    # VerifiableForest serves + signs natively; BestTabular is the (non-servable) ceiling.
    deployable = "baseline"
    if forest is not None and (base_auc is None or report["verifiable_forest"]["auc"] >= base_auc):
        deployable = "verifiable_forest"
    report["deployable_champion"] = deployable
    report["train_seconds"] = round(time.time() - t0, 1)

    out = REPO / "results" / "calibration" / f"{args.hazard}_champion_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    # Export the frozen forest constants when it is the deployable champion (the
    # live scorer loads these + re-runs them in numpy; the receipt is signed at serve time).
    if deployable == "verifiable_forest":
        fp_out = REPO / "results" / "calibration" / f"{args.hazard}_forest_fp.json"
        fp_out.write_text(json.dumps(forest["constants"]) + "\n", encoding="utf-8")
        print(f"  exported servable frozen forest -> {fp_out}")

    bt = report["best_tabular"]
    print(f"  BestTabular AUC {bt['auc_mean']:.4f} +/- {bt['auc_std']:.4f} (ceiling, not servable)")
    if "delta_auc" in report:
        print(f"  baseline AUC {report['baseline']['auc']:.4f}  "
              f"BestTabular delta {report['delta_auc']:+.4f}")
    print(f"  DEPLOYABLE CHAMPION = {deployable}")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
