#!/usr/bin/env python3
"""Aggressive operational earthquake model bakeoff.

This script is intentionally broader than the production-ish ranker:

* classifier families (CatBoost/LightGBM/ExtraTrees);
* monthly listwise rankers (LightGBM LambdaRank);
* hard-gated regime experts;
* ETAS-residual stacking;
* a small neural MLP baseline when PyTorch is available.

All families use the same cached causal feature matrix from
``operational_tabular_ranker.py`` and the same temporal split:
train < 2018, validation 2018-2019, held-out test >= 2020.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "research"))

import operational_tabular_ranker as otr  # noqa: E402


def _predict_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.predict(X)


def _sorted_group(mask, T):
    idx = np.where(mask)[0]
    order = np.lexsort((np.arange(len(idx)), T[idx]))
    idx = idx[order]
    _, counts = np.unique(T[idx], return_counts=True)
    return idx, counts.tolist()


def _feature_index(names, name):
    try:
        return names.index(name)
    except ValueError:
        return None


def _regime_labels(A, names):
    """Coarse tectonic regimes from static priors; no labels are used."""
    n = A.shape[0]
    regime = np.full(n, "remote", dtype=object)
    sub = _feature_index(names, "plate_subduction_lt200")
    other = _feature_index(names, "plate_other_lt200")
    hist6 = _feature_index(names, "hist_m6_500km_n")
    lat = np.abs(A[:, _feature_index(names, "context_0")] * 90.0)

    if hist6 is not None:
        regime[A[:, hist6] > 0] = "historic_m6"
    if other is not None:
        regime[A[:, other] > 0.5] = "plate_other"
    if sub is not None:
        regime[A[:, sub] > 0.5] = "subduction"
    regime[(regime == "remote") & (lat > 55.0)] = "high_lat_remote"
    return regime


def _evaluate(name, val_score, test_score, Y, T, val, test, extra=None):
    row = {
        "name": name,
        "val_auc": round(otr._auc(Y[val], val_score), 4),
        "test_auc": round(otr._auc(Y[test], test_score), 4),
        "test_grouped_by_month_auc": round(otr._grouped_auc(Y[test], test_score, T[test]), 4),
    }
    if extra:
        row.update(extra)
    return row


def _run_lightgbm(A, Y, T, train, val, test):
    rows = []
    try:
        import lightgbm as lgb
    except Exception as exc:
        return [{"name": "lightgbm_unavailable", "error": repr(exc)}]

    configs = [
        ("lgb_cls_15", dict(n_estimators=900, learning_rate=0.025, num_leaves=15,
                            min_child_samples=20, subsample=0.9, colsample_bytree=0.85,
                            reg_lambda=1.0)),
        ("lgb_cls_31", dict(n_estimators=1000, learning_rate=0.020, num_leaves=31,
                            min_child_samples=30, subsample=0.9, colsample_bytree=0.8,
                            reg_lambda=3.0)),
        ("lgb_cls_63", dict(n_estimators=1200, learning_rate=0.015, num_leaves=63,
                            min_child_samples=40, subsample=0.85, colsample_bytree=0.75,
                            reg_lambda=5.0)),
    ]
    for seed, (name, cfg) in enumerate(configs, start=710):
        model = lgb.LGBMClassifier(
            objective="binary",
            class_weight="balanced",
            random_state=seed,
            verbose=-1,
            n_jobs=-1,
            **cfg,
        )
        model.fit(
            A[train],
            Y[train],
            eval_set=[(A[val], Y[val])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        rows.append(_evaluate(name, model.predict_proba(A[val])[:, 1],
                              model.predict_proba(A[test])[:, 1], Y, T, val, test))

    tr_idx, tr_group = _sorted_group(train, T)
    va_idx, va_group = _sorted_group(val, T)
    for seed, leaves in enumerate([15, 31, 63], start=750):
        ranker = lgb.LGBMRanker(
            objective="lambdarank",
            metric="auc",
            n_estimators=800,
            learning_rate=0.025,
            num_leaves=leaves,
            min_child_samples=20,
            subsample=0.9,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=seed,
            verbose=-1,
            n_jobs=-1,
        )
        ranker.fit(
            A[tr_idx],
            Y[tr_idx],
            group=tr_group,
            eval_set=[(A[va_idx], Y[va_idx])],
            eval_group=[va_group],
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        rows.append(_evaluate(f"lgb_lambdarank_{leaves}", ranker.predict(A[val]),
                              ranker.predict(A[test]), Y, T, val, test))
    return rows


def _run_xgboost_classifiers(A, Y, T, train, val, test):
    rows = []
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        return [{"name": "xgboost_unavailable", "error": repr(exc)}]

    scale_pos_weight = float((Y[train] == 0).sum() / max(int(Y[train].sum()), 1))
    configs = [
        ("xgb_cls_d3", dict(n_estimators=900, learning_rate=0.025, max_depth=3,
                            min_child_weight=3, subsample=0.9, colsample_bytree=0.9,
                            reg_lambda=5.0)),
        ("xgb_cls_d4", dict(n_estimators=1100, learning_rate=0.020, max_depth=4,
                            min_child_weight=5, subsample=0.85, colsample_bytree=0.85,
                            reg_lambda=5.0)),
        ("xgb_cls_d5", dict(n_estimators=1200, learning_rate=0.018, max_depth=5,
                            min_child_weight=7, subsample=0.85, colsample_bytree=0.8,
                            reg_lambda=5.0)),
    ]
    for seed, (name, cfg) in enumerate(configs, start=1300):
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            tree_method="hist",
            n_jobs=-1,
            **cfg,
        )
        model.fit(A[train], Y[train], eval_set=[(A[val], Y[val])], verbose=False)
        rows.append(_evaluate(name, model.predict_proba(A[val])[:, 1],
                              model.predict_proba(A[test])[:, 1], Y, T, val, test))
    return rows


def _run_tree_baggers(A, Y, T, train, val, test):
    rows = []
    try:
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
    except Exception as exc:
        return [{"name": "sklearn_baggers_unavailable", "error": repr(exc)}]
    models = [
        ("extra_trees", ExtraTreesClassifier(
            n_estimators=700,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=810,
        )),
        ("random_forest", RandomForestClassifier(
            n_estimators=700,
            min_samples_leaf=20,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=811,
        )),
    ]
    for name, model in models:
        model.fit(A[train], Y[train])
        rows.append(_evaluate(name, model.predict_proba(A[val])[:, 1],
                              model.predict_proba(A[test])[:, 1], Y, T, val, test))
    return rows


def _run_etas_residual(A, names, Y, T, train, val, test):
    """Base ETAS-like feature + residual correction."""
    rows = []
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:
        return [{"name": "etas_residual_unavailable", "error": repr(exc)}]

    etas_cols = [i for i, name in enumerate(names) if "etas" in str(name)]
    if not etas_cols:
        return [{"name": "etas_residual_missing_etas_columns"}]
    etas = A[:, etas_cols].max(axis=1)
    base = LogisticRegression(class_weight="balanced", max_iter=1000)
    base.fit(etas[train, None], Y[train])
    base_val = base.predict_proba(etas[val, None])[:, 1]
    base_test = base.predict_proba(etas[test, None])[:, 1]
    rows.append(_evaluate("etas_only_logistic", base_val, base_test, Y, T, val, test))

    residual_features = np.column_stack([
        A,
        base.predict_proba(etas[:, None])[:, 1],
    ]).astype(np.float32)
    model = HistGradientBoostingClassifier(
        max_iter=700,
        learning_rate=0.03,
        max_leaf_nodes=31,
        l2_regularization=0.2,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=820,
    )
    model.fit(residual_features[train], Y[train])
    rows.append(_evaluate("etas_plus_hgb_residual",
                          model.predict_proba(residual_features[val])[:, 1],
                          model.predict_proba(residual_features[test])[:, 1],
                          Y, T, val, test))
    return rows


def _run_regime_experts(A, names, Y, T, train, val, test):
    rows = []
    try:
        from catboost import CatBoostClassifier
    except Exception as exc:
        return [{"name": "regime_experts_unavailable", "error": repr(exc)}]

    regimes = _regime_labels(A, names)
    global_model = CatBoostClassifier(
        iterations=800,
        learning_rate=0.02,
        depth=5,
        l2_leaf_reg=6,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=830,
        verbose=False,
        allow_writing_files=False,
    )
    global_model.fit(A[train], Y[train], eval_set=(A[val], Y[val]), use_best_model=True)
    val_score = global_model.predict_proba(A[val])[:, 1]
    test_score = global_model.predict_proba(A[test])[:, 1]

    expert_info = {}
    for seed, regime in enumerate(sorted(set(regimes[train])), start=831):
        mask = train & (regimes == regime)
        if mask.sum() < 800 or Y[mask].sum() < 20:
            expert_info[regime] = {"status": "fallback", "n_train": int(mask.sum())}
            continue
        model = CatBoostClassifier(
            iterations=700,
            learning_rate=0.025,
            depth=4,
            l2_leaf_reg=6,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(A[mask], Y[mask])
        vmask = regimes[val] == regime
        tmask = regimes[test] == regime
        if vmask.any():
            val_score[vmask] = model.predict_proba(A[val][vmask])[:, 1]
        if tmask.any():
            test_score[tmask] = model.predict_proba(A[test][tmask])[:, 1]
        expert_info[regime] = {
            "status": "trained",
            "n_train": int(mask.sum()),
            "pos_train": int(Y[mask].sum()),
        }

    rows.append(_evaluate("hard_regime_catboost_experts", val_score, test_score,
                          Y, T, val, test, extra={"experts": expert_info}))
    return rows


def _run_mlp(A, Y, T, train, val, test, epochs):
    rows = []
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:
        return [{"name": "torch_mlp_unavailable", "error": repr(exc)}]

    rng = np.random.RandomState(0)
    mu = A[train].mean(axis=0)
    sd = A[train].std(axis=0) + 1e-6
    X = ((A - mu) / sd).astype(np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    class MLP(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.25),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.25),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    Xtr = torch.tensor(X[train], device=device)
    ytr = torch.tensor(Y[train].astype(np.float32), device=device)
    Xva = torch.tensor(X[val], device=device)
    Xte = torch.tensor(X[test], device=device)
    pos_weight = torch.tensor([(Y[train] == 0).sum() / max(Y[train].sum(), 1)], device=device)
    model = MLP(A.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best = None
    batch = 1024
    for _ in range(epochs):
        model.train()
        perm = torch.tensor(rng.permutation(len(Xtr)), device=device)
        for start in range(0, len(perm), batch):
            idx = perm[start:start + batch]
            opt.zero_grad()
            loss = lossf(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_score = torch.sigmoid(model(Xva)).detach().cpu().numpy()
        val_auc = otr._auc(Y[val], val_score)
        if best is None or val_auc > best[0]:
            best = (val_auc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    model.load_state_dict(best[1])
    model.eval()
    with torch.no_grad():
        val_score = torch.sigmoid(model(Xva)).detach().cpu().numpy()
        test_score = torch.sigmoid(model(Xte)).detach().cpu().numpy()
    rows.append(_evaluate("torch_mlp_bce", val_score, test_score, Y, T, val, test,
                          extra={"device": device, "epochs": epochs}))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default=str(otr.DEFAULT_NPZ))
    ap.add_argument("--rebuild-features", action="store_true")
    ap.add_argument("--skip-neural", action="store_true")
    ap.add_argument("--mlp-epochs", type=int, default=8)
    ap.add_argument("--out", default="results/calibration/earthquake_operational_mega_bakeoff.json")
    args = ap.parse_args(argv)

    npz_path = Path(args.npz)
    z = np.load(npz_path)
    Y = z["Y"].astype(int)
    T = z["T"]
    train = T < otr.VAL0
    val = (T >= otr.VAL0) & (T < otr.TEST0)
    test = T >= otr.TEST0
    A, names = otr.build_feature_matrix(
        npz_path,
        label_days=30.0,
        historical_m5_csv=otr.DEFAULT_HISTORICAL_M5_CSV,
        gsrm_principal=otr.DEFAULT_GSRM_PRINCIPAL,
        rebuild=args.rebuild_features,
    )
    names = [str(n) for n in names]

    families = {}
    families["lightgbm"] = _run_lightgbm(A, Y, T, train, val, test)
    families["xgboost_classifiers"] = _run_xgboost_classifiers(A, Y, T, train, val, test)
    families["baggers"] = _run_tree_baggers(A, Y, T, train, val, test)
    families["etas_residual"] = _run_etas_residual(A, names, Y, T, train, val, test)
    families["regime_experts"] = _run_regime_experts(A, names, Y, T, train, val, test)
    if not args.skip_neural:
        families["neural_mlp"] = _run_mlp(A, Y, T, train, val, test, args.mlp_epochs)

    flat = [
        row
        for rows in families.values()
        for row in rows
        if "val_auc" in row and "test_auc" in row
    ]
    best_by_val = max(flat, key=lambda r: r["val_auc"]) if flat else None
    best_by_test_audit = max(flat, key=lambda r: r["test_auc"]) if flat else None
    report = {
        "label": "M5.0+/100km/30d mega bakeoff",
        "selection_rule": "Compare families by 2018-2019 validation AUC; 2020+ test is held out.",
        "n_features": int(A.shape[1]),
        "n_train": int(train.sum()),
        "n_val": int(val.sum()),
        "n_test": int(test.sum()),
        "test_pos": int(Y[test].sum()),
        "families": families,
        "best_by_validation": best_by_val,
        "best_by_test_audit_not_for_selection": best_by_test_audit,
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
