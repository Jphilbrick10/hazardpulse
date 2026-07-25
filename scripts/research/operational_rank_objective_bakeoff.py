#!/usr/bin/env python3
"""Rank-objective bakeoff for the operational M5+/100km/30d earthquake task.

The broad classifier plateau is around 0.73 test AUC. This script attacks the part of the
problem the requested metric actually cares about: cross-location ranking inside each
forecast month. It reuses the causal feature matrix from ``operational_tabular_ranker.py`` and
adds three objective families:

* XGBoost learning-to-rank objectives;
* CatBoost ranking objectives;
* a small PyTorch discrete point-process/ranking network.

Selection is still by 2018-2019 validation AUC. The 2020+ test split is reported for the
validation-selected champion, not used for tuning.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "research"))

import operational_tabular_ranker as otr  # noqa: E402


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


def _sorted_group_indices(mask, T):
    idx = np.where(mask)[0]
    order = np.lexsort((np.arange(len(idx)), T[idx]))
    idx = idx[order]
    group_values, counts = np.unique(T[idx], return_counts=True)
    return idx, counts.astype(np.int32), group_values


def _qid_for_mask(mask, T):
    idx, counts, _ = _sorted_group_indices(mask, T)
    qid = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
    return idx, qid, counts


def _standardize(A, train):
    mu = A[train].mean(axis=0)
    sd = A[train].std(axis=0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    return ((A - mu) / sd).astype(np.float32)


def _run_xgboost_rankers(A, Y, T, train, val, test, quick=False):
    rows = []
    try:
        from xgboost import XGBRanker
    except Exception as exc:
        return [{"name": "xgboost_unavailable", "error": repr(exc)}]

    tr_idx, tr_qid, _ = _qid_for_mask(train, T)
    va_idx, va_qid, _ = _qid_for_mask(val, T)

    n_estimators = 500 if quick else 900
    configs = [
        ("xgb_rank_pairwise_d4", "rank:pairwise", 4, 0.035, 2.0),
        ("xgb_rank_pairwise_d6", "rank:pairwise", 6, 0.025, 4.0),
        ("xgb_rank_ndcg_d4", "rank:ndcg", 4, 0.035, 2.0),
        ("xgb_rank_map_d4", "rank:map", 4, 0.035, 2.0),
    ]
    for seed, (name, objective, depth, lr, reg_lambda) in enumerate(configs, start=910):
        model = XGBRanker(
            objective=objective,
            n_estimators=n_estimators,
            learning_rate=lr,
            max_depth=depth,
            min_child_weight=8,
            subsample=0.9,
            colsample_bytree=0.85,
            reg_lambda=reg_lambda,
            random_state=seed,
            tree_method="hist",
            n_jobs=-1,
        )
        try:
            model.fit(
                A[tr_idx],
                Y[tr_idx],
                qid=tr_qid,
                eval_set=[(A[va_idx], Y[va_idx])],
                qid_eval_set=[va_qid],
                verbose=False,
            )
        except TypeError:
            # Older xgboost accepts group arrays rather than qid.
            _, tr_counts, _ = _sorted_group_indices(train, T)
            _, va_counts, _ = _sorted_group_indices(val, T)
            model.fit(
                A[tr_idx],
                Y[tr_idx],
                group=tr_counts.tolist(),
                eval_set=[(A[va_idx], Y[va_idx])],
                eval_group=[va_counts.tolist()],
                verbose=False,
            )
        rows.append(_evaluate(name, model.predict(A[val]), model.predict(A[test]), Y, T, val, test))
    return rows


def _run_catboost_rankers(A, Y, T, train, val, test, quick=False):
    rows = []
    try:
        from catboost import CatBoostRanker, Pool
    except Exception as exc:
        return [{"name": "catboost_ranker_unavailable", "error": repr(exc)}]

    tr_idx, tr_counts, _ = _sorted_group_indices(train, T)
    va_idx, va_counts, _ = _sorted_group_indices(val, T)
    tr_gid = np.repeat(np.arange(len(tr_counts)), tr_counts)
    va_gid = np.repeat(np.arange(len(va_counts)), va_counts)
    train_pool = Pool(A[tr_idx], Y[tr_idx], group_id=tr_gid)
    val_pool = Pool(A[va_idx], Y[va_idx], group_id=va_gid)

    iters = 550 if quick else 900
    configs = [
        ("cat_rank_yetirank_d5", "YetiRank", 5, 0.035, 8.0),
        ("cat_rank_pairlogit_d5", "PairLogit", 5, 0.035, 8.0),
        ("cat_rank_pairlogit_d6", "PairLogit", 6, 0.025, 10.0),
    ]
    for seed, (name, loss, depth, lr, l2) in enumerate(configs, start=950):
        model = CatBoostRanker(
            iterations=iters,
            learning_rate=lr,
            depth=depth,
            l2_leaf_reg=l2,
            loss_function=loss,
            eval_metric="AUC",
            random_seed=seed,
            verbose=False,
            allow_writing_files=False,
        )
        try:
            model.fit(train_pool, eval_set=val_pool, use_best_model=True)
            rows.append(_evaluate(name, model.predict(A[val]), model.predict(A[test]), Y, T, val, test))
        except Exception as exc:
            rows.append({"name": name, "error": repr(exc)})
    return rows


def _run_torch_point_process(A, Y, T, train, val, test, quick=False, epochs=60):
    rows = []
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except Exception as exc:
        return [{"name": "torch_point_process_unavailable", "error": repr(exc)}]

    X = _standardize(A, train)
    train_groups = []
    for ref in np.unique(T[train]):
        idx = np.where(train & (T == ref))[0]
        if len(idx):
            train_groups.append(idx)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    X_all = torch.tensor(X, device=device)
    y_all = torch.tensor(Y.astype(np.float32), device=device)
    X_val = X_all[val]
    X_test = X_all[test]
    pos_weight = float((Y[train] == 0).sum() / max(int(Y[train].sum()), 1))

    class RankNet(nn.Module):
        def __init__(self, d, width, dropout):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d, width),
                nn.SiLU(),
                nn.LayerNorm(width),
                nn.Dropout(dropout),
                nn.Linear(width, width // 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(width // 2, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(-1)

    def listwise_loss(scores, labels, temp):
        pos = labels > 0.5
        if not bool(pos.any()):
            return scores.new_tensor(0.0)
        scaled = scores / temp
        return (torch.logsumexp(scaled, dim=0) - scaled[pos].mean()) * temp

    def pairwise_loss(scores, labels, temp, max_neg=256):
        pos = scores[labels > 0.5]
        neg = scores[labels < 0.5]
        if pos.numel() == 0 or neg.numel() == 0:
            return scores.new_tensor(0.0)
        if neg.numel() > max_neg:
            neg = neg[torch.randperm(neg.numel(), device=device)[:max_neg]]
        return F.softplus(-(pos[:, None] - neg[None, :]) / temp).mean() * temp

    def poisson_process_loss(scores, labels):
        # Discrete cell approximation to a point-process likelihood:
        # total intensity penalty plus log intensity at observed positive cells.
        intensity = F.softplus(scores - 2.0) + 1e-5
        pos = labels > 0.5
        if not bool(pos.any()):
            return intensity.sum() * 0.02
        return (intensity.sum() - torch.log(intensity[pos]).sum()) / labels.numel()

    configs = [
        {
            "name": "torch_monthly_listwise",
            "width": 192,
            "dropout": 0.20,
            "lr": 8e-4,
            "bce": 0.05,
            "list": 1.00,
            "pair": 0.00,
            "poisson": 0.00,
            "temp": 1.0,
        },
        {
            "name": "torch_pairwise_plus_bce",
            "width": 192,
            "dropout": 0.25,
            "lr": 7e-4,
            "bce": 0.35,
            "list": 0.00,
            "pair": 1.00,
            "poisson": 0.00,
            "temp": 0.75,
        },
        {
            "name": "torch_point_process_hybrid",
            "width": 256,
            "dropout": 0.25,
            "lr": 6e-4,
            "bce": 0.20,
            "list": 0.65,
            "pair": 0.35,
            "poisson": 0.10,
            "temp": 0.9,
        },
    ]
    if quick:
        configs = configs[:2]
        epochs = min(epochs, 20)

    rng = random.Random(20260629)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    y_train = y_all[train]
    train_index_tensor = torch.tensor(np.where(train)[0], device=device)

    for cfg_seed, cfg in enumerate(configs, start=980):
        torch.manual_seed(cfg_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg_seed)
        model = RankNet(A.shape[1], cfg["width"], cfg["dropout"]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=2e-4)
        best = None
        stale = 0
        max_epochs = int(epochs)
        for epoch in range(max_epochs):
            model.train()
            rng.shuffle(train_groups)
            total_loss = 0.0
            for idx in train_groups:
                ti = torch.tensor(idx, device=device)
                scores = model(X_all[ti])
                labels = y_all[ti]
                loss = scores.new_tensor(0.0)
                if cfg["list"]:
                    loss = loss + cfg["list"] * listwise_loss(scores, labels, cfg["temp"])
                if cfg["pair"]:
                    loss = loss + cfg["pair"] * pairwise_loss(scores, labels, cfg["temp"])
                if cfg["poisson"]:
                    loss = loss + cfg["poisson"] * poisson_process_loss(scores, labels)
                if cfg["bce"]:
                    # A small global BCE term prevents pathological month-only score shifts.
                    sample = train_index_tensor[torch.randint(0, len(train_index_tensor), (1024,), device=device)]
                    loss = loss + cfg["bce"] * bce_loss(model(X_all[sample]), y_all[sample])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                total_loss += float(loss.detach().cpu())

            model.eval()
            with torch.no_grad():
                val_score = model(X_val).detach().cpu().numpy()
            val_auc = otr._auc(Y[val], val_score)
            if best is None or val_auc > best["val_auc"]:
                best = {
                    "epoch": epoch + 1,
                    "val_auc": val_auc,
                    "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "loss": total_loss / max(len(train_groups), 1),
                }
                stale = 0
            else:
                stale += 1
            if stale >= 12 and epoch >= 18:
                break

        model.load_state_dict(best["state"])
        model.eval()
        with torch.no_grad():
            val_score = model(X_val).detach().cpu().numpy()
            test_score = model(X_test).detach().cpu().numpy()
        rows.append(_evaluate(
            cfg["name"],
            val_score,
            test_score,
            Y,
            T,
            val,
            test,
            extra={
                "device": device,
                "best_epoch": int(best["epoch"]),
                "best_train_loss": round(float(best["loss"]), 6),
            },
        ))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default=str(otr.DEFAULT_NPZ))
    ap.add_argument("--rebuild-features", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-xgboost", action="store_true")
    ap.add_argument("--skip-catboost-rank", action="store_true")
    ap.add_argument("--skip-torch", action="store_true")
    ap.add_argument("--torch-epochs", type=int, default=60)
    ap.add_argument(
        "--out",
        default="results/calibration/earthquake_operational_rank_objective_bakeoff.json",
    )
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

    families = {}
    if not args.skip_xgboost:
        families["xgboost_rankers"] = _run_xgboost_rankers(A, Y, T, train, val, test, args.quick)
    if not args.skip_catboost_rank:
        families["catboost_rankers"] = _run_catboost_rankers(A, Y, T, train, val, test, args.quick)
    if not args.skip_torch:
        families["torch_point_process"] = _run_torch_point_process(
            A,
            Y,
            T,
            train,
            val,
            test,
            quick=args.quick,
            epochs=args.torch_epochs,
        )

    flat = [
        row
        for rows in families.values()
        for row in rows
        if "val_auc" in row and "test_auc" in row
    ]
    best_by_val = max(flat, key=lambda r: r["val_auc"]) if flat else None
    best_by_test_audit = max(flat, key=lambda r: r["test_auc"]) if flat else None
    report = {
        "label": "M5.0+/100km/30d rank-objective bakeoff",
        "selection_rule": "Highest validation AUC on 2018-2019; 2020+ test is held out.",
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
