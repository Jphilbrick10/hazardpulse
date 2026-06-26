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
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
# The earthquake feature extraction scans the full catalog per sample (~hours for a
# full split). Cache the extracted matrices so the cost is paid ONCE; delete the file
# (or set HAZARDPULSE_EQ_FEATURE_REBUILD=1) to rebuild after a feature-code change.
_EQ_FEATURE_CACHE = REPO / ".cache" / "earthquake" / "features_v1.npz"
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


# GPU is opt-in via env (the boosters fit in seconds either way; the real cost is
# the feature extraction, which is CPU-bound). Falls back to CPU automatically.
_USE_GPU = os.environ.get("HAZARDPULSE_GPU", "1") != "0"


def _fit_xgboost(Xtr, ytr, seed):
    from xgboost import XGBClassifier
    kw = dict(
        n_estimators=400, max_depth=4, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.7, reg_lambda=1.0,
        eval_metric="logloss", tree_method="hist", random_state=int(seed),
    )
    X = np.asarray(Xtr, float); y = np.asarray(ytr)
    if _USE_GPU:
        try:
            clf = XGBClassifier(device="cuda", **kw); clf.fit(X, y); return clf
        except Exception as exc:
            print(f"    [xgboost] GPU unavailable ({exc}); using CPU.")
    clf = XGBClassifier(**kw); clf.fit(X, y); return clf


def _fit_lightgbm(Xtr, ytr, seed):
    from lightgbm import LGBMClassifier
    kw = dict(
        n_estimators=400, max_depth=4, learning_rate=0.05, num_leaves=15,
        subsample=0.7, colsample_bytree=0.7, reg_lambda=1.0,
        random_state=int(seed), verbose=-1,
    )
    X = np.asarray(Xtr, float); y = np.asarray(ytr)
    if _USE_GPU:
        try:
            clf = LGBMClassifier(device="gpu", **kw); clf.fit(X, y); return clf
        except Exception as exc:
            print(f"    [lightgbm] GPU unavailable ({exc}); using CPU.")
    clf = LGBMClassifier(**kw); clf.fit(X, y); return clf


# Each booster: (name matching VerifiableForest.from_<name>, fit fn).
_FOREST_BOOSTERS = (("xgboost", _fit_xgboost), ("lightgbm", _fit_lightgbm))


def _freeze(name, model):
    """Freeze a fitted booster into a VerifiableForest + its JSON constants."""
    from omega.verifiable_forest import VerifiableForest
    vf = getattr(VerifiableForest, f"from_{name}")(model)
    return vf, vf.fp_constants()


def _forest_proba_from_constants(constants, X):
    from omega.verifiable_forest import predict_fp_from_forest_constants
    labels, F = predict_fp_from_forest_constants(constants, np.asarray(X, float))
    return labels, F, _forest_proba(constants, F)


def _verifiable_forest(X_tr, y_tr, X_te, *, seed=0):
    """Fit the best available booster (XGBoost / LightGBM) and FREEZE it into a
    portable integer forest (omega's VerifiableForest). The frozen ``fp_constants``
    is a pure-JSON model a numpy reproducer re-runs bit-for-bit -- so THIS frozen
    forest is the champion that serves in HazardPulse's numpy-only live path AND
    emits a 0-ULP signed receipt, unlike BestTabular (heavy libs at inference).

    The booster is chosen HONESTLY on an internal validation split carved from the
    training data (never the test holdout), each candidate scored by its FROZEN
    forest's own probability so the selection reflects what actually serves. The
    winner is refit on the full training set.

    Accuracy is then measured on the frozen forest's own test probabilities (what
    serves). ``self_reproduce`` proves a third party holding ONLY the round-tripped
    JSON reproduces the decision exactly (the servability guarantee).
    ``origin_agreement`` (frozen-forest argmax vs the source booster's own predict)
    is informational -- it can be <1.0 only for float32-`<` libraries (xgboost) at
    split boundaries, not for `<=` libraries (lightgbm).
    """
    if str(_OMEGA) not in sys.path:
        sys.path.insert(0, str(_OMEGA))

    Xtr = np.asarray(X_tr, float); Xte = np.asarray(X_te, float); ytr = np.asarray(y_tr)
    # internal validation split for booster SELECTION (last 20% of train; no test leak)
    cut = max(1, int(0.8 * len(ytr)))
    Xfit, yfit, Xval, yval = Xtr[:cut], ytr[:cut], Xtr[cut:], ytr[cut:]

    candidates = []
    for name, fit_fn in _FOREST_BOOSTERS:
        try:
            model = fit_fn(Xfit, yfit, seed)
            _, c = _freeze(name, model)
            _, _, pval = _forest_proba_from_constants(c, Xval)
            candidates.append((name, roc_auc(yval, pval)))
        except Exception as exc:  # booster not installed / freeze failed -> skip
            print(f"    [{name}] unavailable or failed: {exc}")
    if not candidates:
        raise RuntimeError("no forest booster available (need xgboost or lightgbm)")
    best_name, best_val_auc = max(candidates, key=lambda kv: kv[1])

    # Refit the winner on the FULL training set, then freeze the served model.
    model = dict(_FOREST_BOOSTERS)[best_name](Xtr, ytr, seed)
    vf, constants = _freeze(best_name, model)

    labels_self, F, proba = _forest_proba_from_constants(constants, Xte)

    # Servability proof: a third party with ONLY the JSON file reproduces the decision.
    labels_3p, F_3p, _ = _forest_proba_from_constants(json.loads(json.dumps(constants)), Xte)
    self_reproduce = float(np.mean(np.asarray(labels_3p) == np.asarray(labels_self))
                           ) if len(labels_self) else 1.0
    bit_exact = bool(np.array_equal(F_3p, F))
    origin_agreement = float(np.mean(np.asarray(labels_self) == np.asarray(model.predict(Xte))))
    return {
        "proba": proba,
        "booster": best_name,
        "val_auc": best_val_auc,
        "candidates": {n: round(a, 4) for n, a in candidates},
        "self_reproduce": self_reproduce,
        "bit_exact": bit_exact,
        "origin_agreement": origin_agreement,
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
_EQ_VARIANTS = ("baseline", "enhanced", "full")


def _load_all_eq_cached(verbose: bool = True):
    """``load_all_data`` with its expensive (~hours) feature extraction cached to .npz.

    All three variants are cached in one pass (extraction is variant-independent), so
    switching --variant never re-extracts. Rebuild with HAZARDPULSE_EQ_FEATURE_REBUILD=1.
    """
    if _EQ_FEATURE_CACHE.exists() and os.environ.get("HAZARDPULSE_EQ_FEATURE_REBUILD") != "1":
        d = np.load(_EQ_FEATURE_CACHE, allow_pickle=False)
        X_tr = {v: d[f"Xtr_{v}"] for v in _EQ_VARIANTS}
        X_val = {v: d[f"Xval_{v}"] for v in _EQ_VARIANTS}
        X_te = {v: d[f"Xte_{v}"] for v in _EQ_VARIANTS}
        print(f"  [cache] loaded EQ features from {_EQ_FEATURE_CACHE.name} "
              f"(train={len(d['y_tr'])}, val={len(d['y_val'])}, test={len(d['y_te'])})")
        return X_tr, X_val, X_te, d["y_tr"], d["y_val"], d["y_te"]

    from hazardpulse.earthquake.definitive_model import load_all_data
    # Parallel extraction across cores (deterministic now -> safe to fan out + cache).
    X_tr, X_val, X_te, y_tr, y_val, y_te, _meta = load_all_data(verbose=verbose, parallel=True)
    _EQ_FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for v in _EQ_VARIANTS:
        arrays[f"Xtr_{v}"] = X_tr[v]; arrays[f"Xval_{v}"] = X_val[v]; arrays[f"Xte_{v}"] = X_te[v]
    arrays.update(y_tr=y_tr, y_val=y_val, y_te=y_te)
    np.savez(_EQ_FEATURE_CACHE, **arrays)
    print(f"  [cache] saved EQ features -> {_EQ_FEATURE_CACHE.name} (future runs load in seconds)")
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def _load_earthquake(variant: str = "enhanced"):
    # IMPORTANT: the live scorer serves the "enhanced" variant (Block S + Block C =
    # 73 features = ALL_FEATURE_NAMES_ENHANCED). Training/comparing on any other
    # variant would be train/serve skew and the exported forest would reference
    # features the live path cannot build. Default MUST match serve.
    X_tr, X_val, X_te, y_tr, y_val, y_te = _load_all_eq_cached(verbose=True)
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
    parser.add_argument("--variant", default="enhanced",
                        help="earthquake feature variant; MUST match the live scorer "
                             "(default 'enhanced' = Block S+C = the served 73 features)")
    parser.add_argument("--no-forest", action="store_true",
                        help="skip the VerifiableForest (servable) candidate")
    args = parser.parse_args(argv)
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())

    print(f"Loading {args.hazard} training data (variant={args.variant})...")
    load_kw = {"variant": args.variant} if args.hazard == "earthquake" else {}
    X_tr, y_tr, X_te, y_te = _LOADERS[args.hazard](**load_kw)
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
    report["variant"] = args.variant
    report["n_features"] = int(X_tr.shape[1])

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
            "booster": forest["booster"],                 # best of {xgboost, lightgbm}
            "val_auc": forest["val_auc"],                 # internal-validation selection score
            "candidates": forest["candidates"],
            "self_reproduce": forest["self_reproduce"],   # JSON-only third party reproduces it
            "bit_exact": forest["bit_exact"],
            "origin_agreement": forest["origin_agreement"],   # informational diagnostic
            "fp_sha256": forest["fp_sha256"],
            "n_trees": forest["n_trees"],
            "delta_auc_vs_baseline": (vf_auc - base_auc) if base_auc is not None else None,
            "servable": True,   # re-runs from JSON constants in the numpy live path
        }
        print(f"  VerifiableForest[{forest['booster']}] AUC {vf_auc:.4f}  "
              f"self_reproduce {forest['self_reproduce']:.4f}  bit_exact {forest['bit_exact']}  "
              f"candidates={forest['candidates']}  ({time.time()-tf:.0f}s)")

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
