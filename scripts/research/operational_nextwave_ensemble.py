#!/usr/bin/env python3
"""Validation-first ensemble audit adding nextwave-data models to the existing stack."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "research"))

import operational_nextwave_ranker as nwr  # noqa: E402
import operational_tabular_ranker as otr  # noqa: E402

DEFAULT_STACK_NPZ = REPO / ".cache" / "earthquake" / "operational_stack_preds_hist_v1.npz"
OUT_PREDS_DEFAULT = REPO / ".cache" / "earthquake" / "operational_nextwave_stack_preds_v1.npz"
OUT_PREDS_HEAVY = REPO / ".cache" / "earthquake" / "operational_nextwave_stack_preds_v2_heavy.npz"


def _rank_average(scores):
    out = np.zeros(scores.shape[0], dtype=np.float64)
    for j in range(scores.shape[1]):
        order = np.argsort(scores[:, j])
        ranks = np.empty(scores.shape[0], dtype=np.float64)
        ranks[order] = np.arange(scores.shape[0], dtype=np.float64)
        out += ranks / max(scores.shape[0] - 1, 1)
    return out / scores.shape[1]


def _evaluate(name, val_score, test_score, yval, ytest, ttest, extra=None):
    row = {
        "name": name,
        "val_auc": round(otr._auc(yval, val_score), 6),
        "test_auc": round(otr._auc(ytest, test_score), 6),
        "test_grouped_by_month_auc": round(otr._grouped_auc(ytest, test_score, ttest), 6),
    }
    if extra:
        row.update(extra)
    return row


def _cat_model(depth, seed):
    from catboost import CatBoostClassifier

    params = {
        5: dict(iterations=850, learning_rate=0.022, depth=5, l2_leaf_reg=7),
        6: dict(iterations=1000, learning_rate=0.018, depth=6, l2_leaf_reg=10),
        7: dict(iterations=1100, learning_rate=0.014, depth=7, l2_leaf_reg=12),
    }[depth]
    return CatBoostClassifier(
        **params,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )


def build_nextwave_predictions(rebuild=False, include_heavy_data=False):
    out_preds = OUT_PREDS_HEAVY if include_heavy_data else OUT_PREDS_DEFAULT
    if out_preds.exists() and not rebuild:
        z = np.load(out_preds, allow_pickle=True)
        return z["names"], z["val"], z["test"]

    base_npz = np.load(otr.DEFAULT_NPZ)
    Y = base_npz["Y"].astype(int)
    T = base_npz["T"]
    train = T < otr.VAL0
    val = (T >= otr.VAL0) & (T < otr.TEST0)
    test = T >= otr.TEST0
    A, _ = otr.build_feature_matrix(
        otr.DEFAULT_NPZ,
        label_days=30.0,
        historical_m5_csv=otr.DEFAULT_HISTORICAL_M5_CSV,
        gsrm_principal=otr.DEFAULT_GSRM_PRINCIPAL,
        rebuild=False,
    )
    B, names = nwr.build_nextwave_features(
        otr.DEFAULT_NPZ,
        rebuild=False,
        include_heavy_data=include_heavy_data,
    )
    families = {
        "nw_tremor_d7": ([i for i, name in enumerate(names) if name.startswith("tremor_")], 7),
        "nw_gnss_d7": (
            [
                i
                for i, name in enumerate(names)
                if name.startswith("gnss_") and not name.startswith("gnss_crescent_field_")
            ],
            7,
        ),
        "nw_gnss_field_d7": (
            [i for i, name in enumerate(names) if name.startswith("gnss_crescent_field_")],
            7,
        ),
        "nw_regional_d7": (
            [i for i, name in enumerate(names) if name.startswith("regional_micro_")],
            7,
        ),
        "nw_insar_coverage_d7": (
            [i for i, name in enumerate(names) if name.startswith("insar_aria_coverage_")],
            7,
        ),
        "nw_waveform_d7": ([i for i, name in enumerate(names) if name.startswith("waveform_noise_")], 7),
        "nw_slab2_d7": ([i for i, name in enumerate(names) if name.startswith("slab2_")], 7),
        "nw_coupling_d7": ([i for i, name in enumerate(names) if name.startswith("coupling_")], 7),
        "nw_tremor_regional_d7": (
            [
                i
                for i, name in enumerate(names)
                if name.startswith("tremor_") or name.startswith("regional_micro_")
            ],
            7,
        ),
        "nw_all_d6": (list(range(B.shape[1])), 6),
    }
    val_preds = []
    test_preds = []
    pred_names = []
    for seed, (name, (cols, depth)) in enumerate(families.items(), start=1900):
        if not cols:
            continue
        Z = np.hstack([A, B[:, cols]]).astype(np.float32)
        model = _cat_model(depth, seed)
        model.fit(Z[train], Y[train], eval_set=(Z[val], Y[val]), use_best_model=True)
        val_preds.append(model.predict_proba(Z[val])[:, 1])
        test_preds.append(model.predict_proba(Z[test])[:, 1])
        pred_names.append(name)

    val_arr = np.column_stack(val_preds)
    test_arr = np.column_stack(test_preds)
    out_preds.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_preds,
        names=np.asarray(pred_names, dtype=object),
        val=val_arr,
        test=test_arr,
    )
    return np.asarray(pred_names, dtype=object), val_arr, test_arr


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild-nextwave-preds", action="store_true")
    ap.add_argument("--include-heavy-data", action="store_true")
    ap.add_argument("--out", default="results/calibration/earthquake_operational_nextwave_ensemble.json")
    args = ap.parse_args(argv)

    base_npz = np.load(otr.DEFAULT_NPZ)
    Y = base_npz["Y"].astype(int)
    T = base_npz["T"]
    val = (T >= otr.VAL0) & (T < otr.TEST0)
    test = T >= otr.TEST0
    yval = Y[val]
    ytest = Y[test]
    ttest = T[test]

    old = np.load(DEFAULT_STACK_NPZ, allow_pickle=True)
    old_names = [str(x) for x in old["names"]]
    nw_names, nw_val, nw_test = build_nextwave_predictions(
        rebuild=args.rebuild_nextwave_preds,
        include_heavy_data=args.include_heavy_data,
    )
    names = old_names + [str(x) for x in nw_names]
    Pval = np.column_stack([old["val"], nw_val])
    Ptest = np.column_stack([old["test"], nw_test])

    rows = []
    for j, name in enumerate(names):
        rows.append(_evaluate(name, Pval[:, j], Ptest[:, j], yval, ytest, ttest))

    val_aucs = np.array([otr._auc(yval, Pval[:, j]) for j in range(Pval.shape[1])])
    order = np.argsort(val_aucs)[::-1]
    for k in range(1, Pval.shape[1] + 1):
        cols = order[:k]
        rows.append(_evaluate(
            f"rank_avg_top{k}_by_val",
            _rank_average(Pval[:, cols]),
            _rank_average(Ptest[:, cols]),
            yval,
            ytest,
            ttest,
            extra={"members": [names[i] for i in cols]},
        ))
    rows.append(_evaluate(
        "rank_avg_all_old_plus_nextwave",
        _rank_average(Pval),
        _rank_average(Ptest),
        yval,
        ytest,
        ttest,
        extra={"members": names},
    ))

    selectable = [row for row in rows if "val_auc" in row]
    report = {
        "label": "M5.0+/100km/30d nextwave ensemble audit",
        "selection_rule": "Rank-average candidate selected by 2018-2019 validation AUC; 2020+ test held out.",
        "n_old_models": int(old["val"].shape[1]),
        "n_nextwave_models": int(nw_val.shape[1]),
        "n_total_models": int(Pval.shape[1]),
        "include_heavy_data": bool(args.include_heavy_data),
        "rows": rows,
        "selected_by_validation": max(selectable, key=lambda r: r["val_auc"]),
        "best_by_test_audit_not_for_selection": max(selectable, key=lambda r: r["test_auc"]),
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
