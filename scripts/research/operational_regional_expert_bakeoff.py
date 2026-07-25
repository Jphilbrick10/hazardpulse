#!/usr/bin/env python3
"""Validation-gated regional experts for the M5+/100km/30d operational task.

This tests the "one global model may be too blunt" hypothesis without peeking at test results:

1. Train a global CatBoost on the full causal feature matrix.
2. Train region-specific experts on high-quality tectonic/network regions.
3. On 2018-2019 validation only, allow an expert to replace the global score inside its region
   if it improves regional validation AUC.
4. Report the resulting broad 2020+ AUC once.
"""
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


def _cat_model(depth=5, seed=2300, iterations=700):
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=0.024 if depth <= 5 else 0.018,
        depth=depth,
        l2_leaf_reg=9,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )


def _lat_lon_from_npz(z):
    ctx = z["X"][:, -1, 6:20].astype(np.float32)
    return ctx[:, 0] * 90.0, ctx[:, 1] * 180.0


def _region_masks(lat, lon, next_names, B):
    masks = {
        "california": (lat >= 32.0) & (lat <= 42.5) & (lon >= -125.0) & (lon <= -114.0),
        "cascadia": (lat >= 40.0) & (lat <= 52.5) & (lon >= -130.0) & (lon <= -118.0),
        "alaska": (lat >= 50.0) & (lat <= 72.0) & (lon >= -170.0) & (lon <= -130.0),
        "hawaii": (lat >= 18.0) & (lat <= 23.0) & (lon >= -161.5) & (lon <= -154.0),
        "puerto_rico": (lat >= 16.0) & (lat <= 20.5) & (lon >= -68.5) & (lon <= -62.0),
        "japan_kuril": (lat >= 29.0) & (lat <= 47.0) & (lon >= 128.0) & (lon <= 150.0),
        "chile": (lat >= -46.0) & (lat <= -17.0) & (lon >= -76.0) & (lon <= -66.0),
        "new_zealand": (lat >= -48.0) & (lat <= -33.0) & (lon >= 165.0) & (lon <= 180.0),
    }
    if "slab2_nearest_logdist" in next_names:
        j = next_names.index("slab2_nearest_logdist")
        masks["slab2_within300km"] = np.expm1(B[:, j]) <= 300.0
    return masks


def _eval_row(name, yval, val_score, ytest, test_score, ttest, extra=None):
    row = {
        "name": name,
        "val_auc": round(otr._auc(yval, val_score), 6),
        "test_auc": round(otr._auc(ytest, test_score), 6),
        "test_grouped_by_month_auc": round(otr._grouped_auc(ytest, test_score, ttest), 6),
    }
    if extra:
        row.update(extra)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default=str(otr.DEFAULT_NPZ))
    ap.add_argument("--rebuild-nextwave", action="store_true")
    ap.add_argument("--include-heavy-data", action="store_true")
    ap.add_argument("--min-train-pos", type=int, default=40)
    ap.add_argument("--min-val-pos", type=int, default=8)
    ap.add_argument("--out", default="results/calibration/earthquake_operational_regional_experts.json")
    args = ap.parse_args(argv)

    npz_path = Path(args.npz)
    z = np.load(npz_path)
    Y = z["Y"].astype(int)
    T = z["T"]
    lat, lon = _lat_lon_from_npz(z)
    train = T < otr.VAL0
    val = (T >= otr.VAL0) & (T < otr.TEST0)
    test = T >= otr.TEST0

    A, _ = otr.build_feature_matrix(
        npz_path,
        label_days=30.0,
        historical_m5_csv=otr.DEFAULT_HISTORICAL_M5_CSV,
        gsrm_principal=otr.DEFAULT_GSRM_PRINCIPAL,
        rebuild=False,
    )
    B, next_names = nwr.build_nextwave_features(
        npz_path,
        rebuild=args.rebuild_nextwave,
        include_heavy_data=args.include_heavy_data,
    )
    Z = np.hstack([A, B]).astype(np.float32)

    global_model = _cat_model(depth=5, seed=2301, iterations=850)
    global_model.fit(Z[train], Y[train], eval_set=(Z[val], Y[val]), use_best_model=True)
    global_val = global_model.predict_proba(Z[val])[:, 1]
    global_test = global_model.predict_proba(Z[test])[:, 1]

    val_score = global_val.copy()
    test_score = global_test.copy()
    masks = _region_masks(lat, lon, next_names, B)
    expert_rows = []
    selected_regions = []
    for k, (name, mask) in enumerate(masks.items(), start=1):
        tr = train & mask
        va = val & mask
        te = test & mask
        stats = {
            "train_n": int(tr.sum()),
            "train_pos": int(Y[tr].sum()),
            "val_n": int(va.sum()),
            "val_pos": int(Y[va].sum()),
            "test_n": int(te.sum()),
            "test_pos": int(Y[te].sum()),
        }
        if stats["train_pos"] < args.min_train_pos or stats["val_pos"] < args.min_val_pos:
            expert_rows.append({"name": name, "status": "skipped_sparse", **stats})
            continue
        if Y[va].sum() == 0 or Y[va].sum() == va.sum():
            expert_rows.append({"name": name, "status": "skipped_one_class_val", **stats})
            continue

        model = _cat_model(depth=5, seed=2400 + k, iterations=650)
        model.fit(Z[tr], Y[tr], eval_set=(Z[va], Y[va]), use_best_model=True)
        ev = model.predict_proba(Z[va])[:, 1]
        et = model.predict_proba(Z[te])[:, 1] if te.any() else np.array([], dtype=float)
        region_global_val_auc = otr._auc(Y[va], global_val[va[val]])
        region_expert_val_auc = otr._auc(Y[va], ev)
        region_row = {
            "name": name,
            "status": "trained",
            **stats,
            "global_region_val_auc": round(region_global_val_auc, 6),
            "expert_region_val_auc": round(region_expert_val_auc, 6),
        }
        if te.any() and Y[te].sum() not in (0, te.sum()):
            region_row["global_region_test_auc"] = round(otr._auc(Y[te], global_test[te[test]]), 6)
            region_row["expert_region_test_auc"] = round(otr._auc(Y[te], et), 6)

        if region_expert_val_auc > region_global_val_auc:
            val_score[va[val]] = ev
            if te.any():
                test_score[te[test]] = et
            selected_regions.append(name)
            region_row["selected_by_validation"] = True
        else:
            region_row["selected_by_validation"] = False
        expert_rows.append(region_row)

    rows = [
        _eval_row("global_cat_d5", Y[val], global_val, Y[test], global_test, T[test]),
        _eval_row(
            "validation_gated_regional_experts",
            Y[val],
            val_score,
            Y[test],
            test_score,
            T[test],
            extra={"selected_regions": selected_regions},
        ),
    ]
    report = {
        "label": "M5.0+/100km/30d validation-gated regional experts",
        "selection_rule": "Experts replace global model only when regional 2018-2019 validation AUC improves.",
        "features": int(Z.shape[1]),
        "include_heavy_data": bool(args.include_heavy_data),
        "n_train": int(train.sum()),
        "n_val": int(val.sum()),
        "n_test": int(test.sum()),
        "expert_rows": expert_rows,
        "rows": rows,
        "selected_by_validation": rows[1],
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
