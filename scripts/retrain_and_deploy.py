#!/usr/bin/env python3
"""Continuous-learning retrain + deploy for the earthquake model.

Turns "we capture everything" into "we LEARN from everything". Today the deployed
model is frozen on a single historical split; new earthquakes only flow into scoring
and calibration. This pipeline:

  1. HONEST TUNE  -- pick the forest hyperparameters on a validation split carved
     from training (never the test holdout).
  2. GATE         -- evaluate the tuned forest vs the incumbent GBT on the temporal
     TEST holdout. Deploy only if it does not regress (never-worse guard).
  3. TRAIN-ON-ALL -- if it passes, refit the SAME config on EVERY year we have
     (folding the held-out years back in) so the shipped model knows about all the
     data we have captured, not just the training era.
  4. SHIP         -- freeze to a portable, 0-ULP, Ed25519-signable VerifiableForest
     and export results/calibration/earthquake_forest_fp.json, which the live scorer
     already loads (and refuses on feature mismatch).

Run monthly (workflow) as the catalog grows. Loads the cached feature matrices, so
after the first extraction this is minutes, not hours.

    python scripts/retrain_and_deploy.py --hazard earthquake --deploy
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO.parent / "Coherence" / "omega_one"))  # VerifiableForest
_spec = importlib.util.spec_from_file_location("tbt", REPO / "scripts" / "train_best_tabular.py")
_tbt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tbt)
roc_auc, brier = _tbt.roc_auc, _tbt.brier

# Honest hyperparameter grid (selected on validation, never on test). Includes
# configs that MIRROR the incumbent GBT's regularization (subsample 0.6, colsample
# 0.7, strong leaf/gamma/L2) so a SIGNABLE forest has a fair shot at matching it.
_GRID = [
    # incumbent-mirroring (300t, depth4, lr0.03, heavy leaf reg, min-split-gain)
    ("xgboost", dict(n_estimators=300, max_depth=4, learning_rate=0.03, reg_lambda=1.0,
                     min_child_weight=15, gamma=0.05, subsample=0.6, colsample_bytree=0.7)),
    ("xgboost", dict(n_estimators=500, max_depth=4, learning_rate=0.03, reg_lambda=2.0,
                     min_child_weight=20, gamma=0.1, subsample=0.6, colsample_bytree=0.7)),
    ("lightgbm", dict(n_estimators=300, num_leaves=15, max_depth=4, learning_rate=0.03,
                      reg_lambda=1.0, min_child_samples=15, subsample=0.6, colsample_bytree=0.7)),
    ("lightgbm", dict(n_estimators=500, num_leaves=15, max_depth=4, learning_rate=0.02,
                      reg_lambda=3.0, min_child_samples=50, subsample=0.6, colsample_bytree=0.7)),
    # exploratory
    ("xgboost", dict(n_estimators=800, max_depth=4, learning_rate=0.03, reg_lambda=3.0, min_child_weight=5)),
    ("xgboost", dict(n_estimators=1200, max_depth=3, learning_rate=0.02, reg_lambda=5.0, min_child_weight=10)),
    ("lightgbm", dict(n_estimators=600, num_leaves=15, max_depth=4, learning_rate=0.03, reg_lambda=2.0, min_child_samples=30)),
    ("lightgbm", dict(n_estimators=1000, num_leaves=31, max_depth=6, learning_rate=0.02, reg_lambda=5.0, min_child_samples=40)),
]

_USE_GPU = _tbt._USE_GPU


def _fit(name, X, y, kw, seed=0):
    X = np.asarray(X, float); y = np.asarray(y)
    if name == "xgboost":
        from xgboost import XGBClassifier
        params = {"tree_method": "hist", "eval_metric": "logloss", "random_state": seed,
                  "subsample": 0.7, "colsample_bytree": 0.7, **kw}   # config overrides base
        if _USE_GPU:
            try:
                m = XGBClassifier(device="cuda", **params); m.fit(X, y); return m
            except Exception:
                pass
        m = XGBClassifier(**params); m.fit(X, y); return m
    from lightgbm import LGBMClassifier
    params = {"random_state": seed, "subsample": 0.7, "colsample_bytree": 0.7,
              "verbose": -1, **kw}
    if _USE_GPU:
        try:
            m = LGBMClassifier(device="gpu", **params); m.fit(X, y); return m
        except Exception:
            pass
    m = LGBMClassifier(**params); m.fit(X, y); return m


def _forest_proba(name, model, X):
    vf, c = _tbt._freeze(name, model)
    _, _, p = _tbt._forest_proba_from_constants(c, np.asarray(X, float))
    return p, vf, c


def _train_and_serialize_gbt(Xall, yall):
    """Train the incumbent GBT architecture on ALL data and serialize to the live
    `hazardpulse_gbt_v1` JSON the scorer loads. Trees are scale-invariant, so we
    z-score with stored means/stds purely so the scorer's nan->0 == train-time
    impute-with-mean (keeping train/serve consistent)."""
    from hazardpulse.earthquake import definitive_model as dm
    Xall = np.asarray(Xall, np.float32)
    means = Xall.mean(0); stds = Xall.std(0); stds = np.where(stds == 0, 1.0, stds)
    Xz = ((Xall - means) / stds).astype(np.float32)
    gbt = dm.GradientBoostedTrees(
        n_trees=dm.GBT_N_TREES, max_depth=dm.GBT_MAX_DEPTH,
        learning_rate=dm.GBT_LEARNING_RATE, min_samples_leaf=dm.GBT_MIN_SAMPLES_LEAF,
        subsample=dm.GBT_SUBSAMPLE, colsample=dm.GBT_COLSAMPLE,
        l2_reg=dm.GBT_L2_REG, gamma=dm.GBT_GAMMA,
    )
    gbt.fit(Xz, np.asarray(yall), verbose=False)
    return {
        "model_format": "hazardpulse_gbt_v1",
        "model_version": "eq_coherence_v1_retrain_all",
        "n_trees": len(gbt.trees),
        "init_pred": float(gbt.init_pred),
        "learning_rate": float(gbt.lr),
        "trees": gbt.trees,
        "normalization": {"means": means.tolist(), "stds": stds.tolist()},
        "feature_names": list(dm.ALL_FEATURE_NAMES_ENHANCED),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hazard", default="earthquake", choices=["earthquake"])
    ap.add_argument("--deploy", action="store_true",
                    help="export the champion if it passes the never-worse gate")
    ap.add_argument("--out", default="results/calibration/earthquake_retrain_report.json")
    ap.add_argument("--floor", type=float, default=0.70,
                    help="GBT-only mode: deploy only if holdout AUC >= floor")
    args = ap.parse_args(argv)

    # The VerifiableForest comparison needs omega_one (a sibling repo). When it is
    # absent (e.g. GitHub-hosted CI), fall back to a GBT-only retrain -- the GBT wins
    # earthquakes anyway, and its train-on-all redeploy is the real improvement.
    try:
        import omega.verifiable_forest  # noqa: F401
        have_forest = True
    except Exception:
        have_forest = False
        print("  omega VerifiableForest unavailable -> GBT-only retrain mode")

    print("Loading cached features...")
    Xtr, Xval, Xte, ytr, yval, yte = _tbt._load_all_eq_cached(verbose=False)
    v = "enhanced"   # the served variant
    Xfit, Xv, Xtest = Xtr[v].astype(float), Xval[v].astype(float), Xte[v].astype(float)
    yfit, yv, ytest = np.asarray(ytr), np.asarray(yval), np.asarray(yte)
    print(f"  fit={len(yfit)} val={len(yv)} test={len(ytest)} feat={Xfit.shape[1]}")

    t0 = time.time()
    Xtv = np.vstack([Xfit, Xv]); ytv = np.concatenate([yfit, yv])
    base_proba = _tbt._baseline_earthquake(Xtv, ytv, Xtest)
    inc_auc, inc_brier = roc_auc(ytest, base_proba), brier(ytest, base_proba)

    report = {
        "hazard": args.hazard, "variant": v, "n_features": int(Xfit.shape[1]),
        "incumbent_test_auc": inc_auc, "incumbent_test_brier": inc_brier,
        "deployed": False,
    }

    if have_forest:
        # 1) honest tune on validation; 2) gate tuned forest vs incumbent on test
        tuned = []
        for name, kw in _GRID:
            m = _fit(name, Xfit, yfit, kw)
            p, _, _ = _forest_proba(name, m, Xv)
            tuned.append((name, kw, roc_auc(yv, p)))
            print(f"  tune {name:8s} {kw.get('n_estimators')}t d{kw.get('max_depth')} "
                  f"lr{kw.get('learning_rate')}: val AUC {tuned[-1][2]:.4f}")
        best_name, best_kw, best_val = max(tuned, key=lambda t: t[2])
        print(f"  -> best on val: {best_name} {best_kw} (val AUC {best_val:.4f})")
        m_tv = _fit(best_name, Xtv, ytv, best_kw)
        p_test, _, _ = _forest_proba(best_name, m_tv, Xtest)
        forest_auc, forest_brier = roc_auc(ytest, p_test), brier(ytest, p_test)
        gate_pass = bool(forest_auc >= inc_auc)
        print(f"\n  GATE (test holdout): forest {forest_auc:.4f} vs incumbent {inc_auc:.4f}"
              f"  -> {'forest wins' if gate_pass else 'GBT wins (keep incumbent arch)'}")
        report.update(best_config={"booster": best_name, **best_kw}, val_auc=best_val,
                      forest_test_auc=forest_auc, forest_test_brier=forest_brier,
                      delta_auc=forest_auc - inc_auc)
        winner = "verifiable_forest" if gate_pass else "gbt"
    else:
        # GBT-only: gate the incumbent architecture on the holdout vs a skill floor.
        gate_pass = bool(inc_auc >= args.floor)
        print(f"  GATE (test holdout): GBT {inc_auc:.4f} vs floor {args.floor:.2f}"
              f"  -> {'PASS' if gate_pass else 'FAIL (do not deploy)'}")
        winner = "gbt" if gate_pass else None
        best_name = best_kw = None

    report["winner"] = winner
    report["tune_seconds"] = round(time.time() - t0, 1)

    # 3-4) TRAIN-ON-ALL + ship the winner (fold the held-out years back in so the
    # deployed model knows every earthquake we have captured, not just 2005-2017).
    Xall = np.vstack([Xfit, Xv, Xtest]); yall = np.concatenate([yfit, yv, ytest])
    if args.deploy and winner is None:
        print("  gate FAILED -> not deploying (keeping the current model).")
    elif args.deploy:
        print(f"  TRAIN-ON-ALL: refit the gate winner ({winner}) on every year "
              f"({len(yall)} samples)...")
        if winner == "verifiable_forest":
            m_all = _fit(best_name, Xall, yall, best_kw)
            vf, constants = _tbt._freeze(best_name, m_all)
            fp = REPO / "results" / "calibration" / "earthquake_forest_fp.json"
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(json.dumps(constants) + "\n", encoding="utf-8")
            report.update(deployed=True, n_train_all=int(len(yall)),
                          model_sha256=vf.fp_model_sha256(),
                          n_trees=int(len(constants["tree_root"])),
                          deployed_path=str(fp.relative_to(REPO)))
            print(f"  SHIPPED signed forest ({report['n_trees']} trees, "
                  f"sha {report['model_sha256'][:12]}) -> {fp.name}")
        else:
            payload = _train_and_serialize_gbt(Xall, yall)
            mp = REPO / "results" / "models" / "earthquake_gbt_v1.json"
            mp.parent.mkdir(parents=True, exist_ok=True)
            mp.write_text(json.dumps(payload), encoding="utf-8")
            report.update(deployed=True, n_train_all=int(len(yall)),
                          n_trees=int(payload["n_trees"]),
                          deployed_path=str(mp.relative_to(REPO)),
                          model_version=payload["model_version"])
            print(f"  SHIPPED retrained GBT ({payload['n_trees']} trees) -> {mp.name}")
        print(f"    skill estimate (conservative, pre-fold) AUC "
              f"{forest_auc if winner=='verifiable_forest' else inc_auc:.4f}")
    else:
        print(f"  winner = {winner}; rerun with --deploy to ship it trained on all data.")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
