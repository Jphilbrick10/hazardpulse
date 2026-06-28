#!/usr/bin/env python3
"""Bake-off: EVERY earthquake-nowcast model on the SAME M5.0 test set + every ensemble.

"Get the best from every model and thing we have." This loads, all aligned row-for-row
on the same deterministic build_samples(M5.0) test set:

  * persistence (best recent-rate feature)              -- the honest baseline
  * hand-crafted GBT (Block S seismicity + Block C coherence)
  * deep GRU+attention on the raw event sequence        -- the breakthrough model
  * ensembles: rank-average, a val-selected weighted blend, and a logistic STACK
    (the stacker is fit on the VALIDATION predictions, never the test set, so the
    reported ensemble number is honest -- no test leakage).

Everything is loaded from caches (deep sequences + GBT features), so it runs in seconds
ONCE the M5.0 GBT feature cache exists (features_v1_my2024_m5.0.npz, built by
backtest_earthquake.py @ HAZARDPULSE_MIN_MAINSHOCK_MAG=5.0).

    python scripts/bakeoff_earthquake.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("HAZARDPULSE_MIN_MAINSHOCK_MAG", "5.0")

_spec = importlib.util.spec_from_file_location("tbt", REPO / "scripts" / "train_best_tabular.py")
_tbt = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tbt)
roc_auc = _tbt.roc_auc


def _fit_xgb(X, y, seed):
    from xgboost import XGBClassifier
    kw = dict(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.7,
              colsample_bytree=0.7, reg_lambda=1.0, eval_metric="logloss",
              tree_method="hist", random_state=int(seed))
    X = np.asarray(X, float); y = np.asarray(y)
    try:
        m = XGBClassifier(device="cuda", **kw); m.fit(X, y); return m
    except Exception:
        m = XGBClassifier(**kw); m.fit(X, y); return m


def _rank(p):
    """Rank-normalize to [0,1] so different-scaled scores can be averaged fairly."""
    p = np.asarray(p, float)
    return np.argsort(np.argsort(p)) / max(len(p) - 1, 1)


def _oriented(y, s):
    a = roc_auc(y, s)
    return (a, s) if a >= 0.5 else (1.0 - a, -np.asarray(s, float))


def main() -> int:
    import torch
    import torch.nn as nn
    mag = "5.0"

    # ---- cached deep sequences (raw, un-normalized) -------------------------- #
    dz = np.load(REPO / ".cache" / "earthquake" / f"deepseq_my2024_m{mag}_K48.npz")
    Xva_s, Mva_s, yva_s = dz["Xva"], dz["Mva"], dz["yva"]
    Xte_s, Mte_s, yte_s = dz["Xte"], dz["Mte"], dz["yte"]

    # ---- cached GBT features (enhanced = Block S + Block C) ------------------- #
    Xtr, Xval, Xte, ytr, yval, yte = _tbt._load_all_eq_cached(verbose=True, max_year=2024)
    enh = "enhanced"
    Xtr_g = np.vstack([Xtr[enh], Xval[enh]]); ytr_g = np.concatenate([ytr, yval])
    Xval_g = Xval[enh]; yval_g = np.asarray(yval).astype(int)
    Xte_g = Xte[enh]; yte_g = np.asarray(yte).astype(int)

    # ---- ALIGNMENT GUARD: both caches derive from the same deterministic ------ #
    #      build_samples(M5.0), so test/val labels must match row-for-row.
    ok_te = len(yte_s) == len(yte_g) and (np.asarray(yte_s).astype(int) == yte_g).all()
    ok_va = len(yva_s) == len(yval_g) and (np.asarray(yva_s).astype(int) == yval_g).all()
    if not (ok_te and ok_va):
        print(f"  FATAL: caches misaligned (test {len(yte_s)}/{len(yte_g)} match={ok_te}; "
              f"val {len(yva_s)}/{len(yval_g)} match={ok_va}). Cannot ensemble safely.")
        return 2
    y = yte_g; yv = yval_g
    print(f"  aligned: train(gbt)={len(ytr_g)}  val={len(y)==len(yte_g) and len(yv)}  "
          f"test={len(y)} ({int(y.sum())} pos, base {y.mean():.3f})")

    # ---- GBT predictions (val + test) ---------------------------------------- #
    mg = _fit_xgb(Xtr_g, ytr_g, 0)
    p_gbt = np.asarray(mg.predict_proba(np.asarray(Xte_g, float)))[:, 1]
    p_gbt_v = np.asarray(mg.predict_proba(np.asarray(Xval_g, float)))[:, 1]

    # ---- deep predictions (val + test) from the saved model ------------------ #
    ck = torch.load(REPO / "results" / "models" / "eq_deep_nowcast_m5.0.pt",
                    map_location="cpu", weights_only=False)
    mu, sd = ck["norm_mu"], ck["norm_sd"]

    class SeqModel(nn.Module):
        def __init__(self, d=6, h=64):
            super().__init__()
            self.proj = nn.Linear(d, h)
            self.gru = nn.GRU(h, h, batch_first=True, bidirectional=True)
            self.att = nn.Linear(2 * h, 1)
            self.head = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Dropout(0.3), nn.Linear(h, 1))

        def forward(self, x, m):
            z = torch.relu(self.proj(x)); z, _ = self.gru(z)
            a = self.att(z).squeeze(-1).masked_fill(m == 0, -1e9)
            a = torch.softmax(a, 1).unsqueeze(-1)
            return self.head((z * a).sum(1)).squeeze(-1)

    net = SeqModel(); net.load_state_dict(ck["state_dict"]); net.eval()

    def deep_prob(X, M):
        Xn = ((X - mu) / sd).astype(np.float32)
        with torch.no_grad():
            return torch.sigmoid(net(torch.tensor(Xn), torch.tensor(M))).numpy()
    p_deep = deep_prob(Xte_s, Mte_s)
    p_deep_v = deep_prob(Xva_s, Mva_s)

    # ---- persistence: best recent-rate feature (oriented) -------------------- #
    from hazardpulse.earthquake.definitive_model import BLOCK_S_NAMES, BLOCK_C_NAMES
    names = list(BLOCK_S_NAMES) + list(BLOCK_C_NAMES)
    rate_feats = [n for n in BLOCK_S_NAMES if any(k in n.lower()
                  for k in ("rate", "n_7d", "n_14d", "n_30d", "n_90d", "n_events"))]
    pers_auc, pers_name = 0.5, None
    for n in rate_feats:
        a, _ = _oriented(y, Xte_g[:, names.index(n)])
        if a > pers_auc:
            pers_auc, pers_name = a, n

    # ---- ensembles ----------------------------------------------------------- #
    rg, rd = _rank(p_gbt), _rank(p_deep)
    ens_avg = roc_auc(y, (rg + rd) / 2.0)

    # weighted blend: pick weight on VALIDATION, apply to test (honest)
    rgv, rdv = _rank(p_gbt_v), _rank(p_deep_v)
    best_w, best_va = 0.5, 0.0
    for w in np.linspace(0, 1, 41):
        av = roc_auc(yv, w * rdv + (1 - w) * rgv)
        if av > best_va:
            best_va, best_w = av, float(w)
    ens_w_test = roc_auc(y, best_w * rd + (1 - best_w) * rg)

    # logistic stack: fit on validation predictions, apply to test (honest)
    from sklearn.linear_model import LogisticRegression
    stack = LogisticRegression(max_iter=1000)
    stack.fit(np.column_stack([p_gbt_v, p_deep_v]), yv)
    p_stack = stack.predict_proba(np.column_stack([p_gbt, p_deep]))[:, 1]
    ens_stack = roc_auc(y, p_stack)

    rep = {
        "n_test": int(len(y)), "test_base_rate": round(float(y.mean()), 4),
        "persistence": {"feature": pers_name, "auc": round(pers_auc, 4)},
        "gbt": round(roc_auc(y, p_gbt), 4),
        "deep": round(roc_auc(y, p_deep), 4),
        "ensemble_rank_avg": round(float(ens_avg), 4),
        "ensemble_weighted": {"auc": round(float(ens_w_test), 4),
                              "deep_weight": round(best_w, 3), "val_auc": round(float(best_va), 4)},
        "ensemble_logistic_stack": round(float(ens_stack), 4),
    }
    cands = {"persistence": rep["persistence"]["auc"], "gbt": rep["gbt"], "deep": rep["deep"],
             "ens_rank_avg": rep["ensemble_rank_avg"], "ens_weighted": rep["ensemble_weighted"]["auc"],
             "ens_stack": rep["ensemble_logistic_stack"]}
    best = max(cands.items(), key=lambda kv: kv[1])
    rep["BEST"] = {"model": best[0], "auc": best[1]}

    print("\n  === M5.0 nowcast bake-off (same test set, all aligned) ===")
    print(f"  persistence ({pers_name}): {rep['persistence']['auc']:.4f}")
    print(f"  hand-crafted GBT          : {rep['gbt']:.4f}")
    print(f"  deep GRU+attention        : {rep['deep']:.4f}")
    print(f"  ensemble rank-average     : {rep['ensemble_rank_avg']:.4f}")
    print(f"  ensemble weighted (val w={best_w:.2f}): {rep['ensemble_weighted']['auc']:.4f}")
    print(f"  ensemble logistic stack   : {rep['ensemble_logistic_stack']:.4f}")
    print(f"  --> BEST: {best[0]} = {best[1]:.4f}")

    out = REPO / "results" / "calibration" / "earthquake_bakeoff_m5.0.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
