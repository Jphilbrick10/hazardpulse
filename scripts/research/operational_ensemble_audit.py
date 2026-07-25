#!/usr/bin/env python3
"""Validation-first ensemble audit for operational earthquake stack predictions.

This script consumes a base-model prediction cache with validation/test columns and reports
what can be selected without looking at the held-out 2020+ period. It is deliberately an audit
layer: it does not claim a new broad champion unless validation selection chose it.
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

DEFAULT_STACK_NPZ = REPO / ".cache" / "earthquake" / "operational_stack_preds_hist_v1.npz"


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


def _rank_average(scores):
    out = np.zeros(scores.shape[0], dtype=np.float64)
    for j in range(scores.shape[1]):
        order = np.argsort(scores[:, j])
        ranks = np.empty(scores.shape[0], dtype=np.float64)
        ranks[order] = np.arange(scores.shape[0], dtype=np.float64)
        out += ranks / max(scores.shape[0] - 1, 1)
    return out / scores.shape[1]


def _standardize(fit, val, test):
    mu = fit.mean(axis=0)
    sd = fit.std(axis=0)
    sd = np.where(sd > 1e-9, sd, 1.0)
    return (fit - mu) / sd, (val - mu) / sd, (test - mu) / sd


def _split_meta_stack(Pval, Ptest, yval, tval, ytest, ttest, names):
    """Fit meta-models on 2018, select by 2019, then refit on 2018-2019 for test."""
    try:
        from sklearn.linear_model import LogisticRegression, RidgeClassifier
    except Exception as exc:
        return [{"name": "meta_stack_unavailable", "error": repr(exc)}]

    split = tval < otr.dt.datetime(2019, 1, 1, tzinfo=otr.dt.timezone.utc).timestamp()
    fit = split
    select = ~split
    rows = []
    candidates = []

    Xfit, Xsel, Xtest = _standardize(Pval[fit], Pval[select], Ptest)
    for C in [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]:
        model = LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=1200,
        )
        model.fit(Xfit, yval[fit])
        sel_score = model.predict_proba(Xsel)[:, 1]
        candidates.append(("logistic", C, otr._auc(yval[select], sel_score)))

    for alpha in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
        model = RidgeClassifier(alpha=alpha, class_weight="balanced")
        model.fit(Xfit, yval[fit])
        sel_score = model.decision_function(Xsel)
        candidates.append(("ridge", alpha, otr._auc(yval[select], sel_score)))

    best_kind, best_param, best_select_auc = max(candidates, key=lambda x: x[2])
    Xfull, _, Xtest_final = _standardize(Pval, Pval, Ptest)
    if best_kind == "logistic":
        model = LogisticRegression(
            C=best_param,
            class_weight="balanced",
            max_iter=2000,
            solver="lbfgs",
            random_state=1201,
        )
        model.fit(Xfull, yval)
        val_score = model.predict_proba(Xfull)[:, 1]
        test_score = model.predict_proba(Xtest_final)[:, 1]
    else:
        model = RidgeClassifier(alpha=best_param, class_weight="balanced")
        model.fit(Xfull, yval)
        val_score = model.decision_function(Xfull)
        test_score = model.decision_function(Xtest_final)

    rows.append(_evaluate(
        "meta_stack_refit_after_2019_selection",
        val_score,
        test_score,
        yval,
        ytest,
        ttest,
        extra={
            "meta_model": best_kind,
            "selected_param": best_param,
            "selection_2019_auc": round(float(best_select_auc), 6),
            "meta_fit_for_selection": "2018",
            "meta_select": "2019",
            "final_refit": "2018-2019 validation rows only",
            "inputs": list(names),
        },
    ))
    rows.append({
        "name": "meta_stack_selection_candidates",
        "candidates": [
            {"kind": kind, "param": param, "selection_2019_auc": round(float(auc), 6)}
            for kind, param, auc in candidates
        ],
    })
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stack-npz", default=str(DEFAULT_STACK_NPZ))
    ap.add_argument("--out", default="results/calibration/earthquake_operational_ensemble_audit.json")
    args = ap.parse_args(argv)

    stack_path = Path(args.stack_npz)
    if not stack_path.exists():
        raise FileNotFoundError(f"missing stack prediction cache: {stack_path}")

    stack = np.load(stack_path, allow_pickle=True)
    names = [str(x) for x in stack["names"]]
    Pval = stack["val"].astype(np.float64)
    Ptest = stack["test"].astype(np.float64)

    z = np.load(otr.DEFAULT_NPZ)
    Y = z["Y"].astype(int)
    T = z["T"]
    val = (T >= otr.VAL0) & (T < otr.TEST0)
    test = T >= otr.TEST0
    yval = Y[val]
    ytest = Y[test]
    tval = T[val]
    ttest = T[test]

    rows = []
    for j, name in enumerate(names):
        rows.append(_evaluate(name, Pval[:, j], Ptest[:, j], yval, ytest, ttest))

    val_aucs = np.array([otr._auc(yval, Pval[:, j]) for j in range(Pval.shape[1])])
    order = np.argsort(val_aucs)[::-1]
    topk_rows = []
    for k in range(1, Pval.shape[1] + 1):
        cols = order[:k]
        row = _evaluate(
            f"avg_top{k}_by_val",
            Pval[:, cols].mean(axis=1),
            Ptest[:, cols].mean(axis=1),
            yval,
            ytest,
            ttest,
            extra={"members": [names[i] for i in cols]},
        )
        topk_rows.append(row)
    rows.extend(topk_rows)
    rows.append(_evaluate(
        "avg_all_models_fixed",
        Pval.mean(axis=1),
        Ptest.mean(axis=1),
        yval,
        ytest,
        ttest,
        extra={"members": names},
    ))
    rows.append(_evaluate(
        "rank_avg_all_models_fixed",
        _rank_average(Pval),
        _rank_average(Ptest),
        yval,
        ytest,
        ttest,
        extra={"members": names},
    ))

    meta_rows = _split_meta_stack(Pval, Ptest, yval, tval, ytest, ttest, names)
    rows.extend(meta_rows)

    selectable = [row for row in rows if "val_auc" in row and row["name"] != "meta_stack_refit_after_2019_selection"]
    best_by_val = max(selectable, key=lambda r: r["val_auc"])
    testable = [row for row in rows if "test_auc" in row]
    best_by_test = max(testable, key=lambda r: r["test_auc"])
    report = {
        "label": "M5.0+/100km/30d ensemble audit",
        "contract": "Uses validation/test prediction cache only; no 2020+ labels are used for selection.",
        "stack_npz": str(stack_path),
        "n_models": int(Pval.shape[1]),
        "n_val": int(len(yval)),
        "n_test": int(len(ytest)),
        "test_pos": int(ytest.sum()),
        "base_model_names": names,
        "rows": rows,
        "selected_by_validation": best_by_val,
        "meta_selected_by_2019_then_refit": next(
            (row for row in rows if row.get("name") == "meta_stack_refit_after_2019_selection"),
            None,
        ),
        "best_by_test_audit_not_for_selection": best_by_test,
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
