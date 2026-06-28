#!/usr/bin/env python3
"""Calibrate the deep nowcast so its probabilities are HONEST ("0.7 means 70%").

A high-AUC model can still be badly calibrated (over/under-confident). Before this goes
on a public-safety site, the probability must mean what it says. We fit a calibrator on
the VALIDATION predictions (never test), apply to test, and measure ECE + Brier + a
reliability table BEFORE and AFTER. Two standard methods are compared (temperature scaling
= 1 parameter, robust; isotonic = non-parametric, flexible); the one with the lower
VALIDATION ECE is selected, and its honest TEST ECE is reported. The calibrator is saved
alongside the model.

    python scripts/calibrate_deep_earthquake.py --model results/models/eq_deep_nowcast_m5.0_2025.pt --mag 5.0 --max-year 2025
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _ece(y, p, bins=10):
    """Expected Calibration Error + reliability table (equal-width bins)."""
    y = np.asarray(y); p = np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0; table = []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum() == 0:
            continue
        conf = float(p[m].mean()); acc = float(y[m].mean()); w = m.sum() / len(p)
        ece += w * abs(conf - acc)
        table.append({"bin": [round(float(edges[i]), 2), round(float(edges[i + 1]), 2)],
                      "p_pred": round(conf, 3), "p_obs": round(acc, 3), "n": int(m.sum())})
    return float(ece), table


def _brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y)) ** 2))


def _fit_temperature(logits, y):
    """1-param temperature scaling: minimize NLL over T (binary)."""
    import torch
    z = torch.tensor(logits, dtype=torch.float64); yt = torch.tensor(y, dtype=torch.float64)
    T = torch.ones(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=100)
    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad(); loss = bce(z / T.clamp_min(1e-3), yt); loss.backward(); return loss
    opt.step(closure)
    return float(T.detach().clamp_min(1e-3))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="results/models/eq_deep_nowcast_m5.0_2025.pt")
    ap.add_argument("--mag", default="5.0")
    ap.add_argument("--max-year", type=int, default=2025)
    ap.add_argument("--K", type=int, default=48, help="must match the model's sequence length")
    args = ap.parse_args(argv)

    import torch
    import torch.nn as nn
    from sklearn.isotonic import IsotonicRegression

    ck = torch.load(REPO / args.model, map_location="cpu", weights_only=False)
    mu, sd = ck["norm_mu"], ck["norm_sd"]
    dz = np.load(REPO / ".cache" / "earthquake" / f"deepseq_my{args.max_year}_m{args.mag}_K{args.K}.npz")
    Xva, Mva, yva = dz["Xva"], dz["Mva"], np.asarray(dz["yva"]).astype(int)
    Xte, Mte, yte = dz["Xte"], dz["Mte"], np.asarray(dz["yte"]).astype(int)

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

    def logits(X, M):
        Xn = ((X - mu) / sd).astype(np.float32)
        with torch.no_grad():
            return net(torch.tensor(Xn), torch.tensor(M)).numpy()
    zv, zt = logits(Xva, Mva), logits(Xte, Mte)
    pv_raw = 1 / (1 + np.exp(-zv)); pt_raw = 1 / (1 + np.exp(-zt))

    # --- temperature scaling (fit on val) ---
    T = _fit_temperature(zv, yva)
    pv_temp = 1 / (1 + np.exp(-zv / T)); pt_temp = 1 / (1 + np.exp(-zt / T))

    # --- isotonic (fit on val) ---
    iso = IsotonicRegression(out_of_bounds="clip"); iso.fit(pv_raw, yva)
    pv_iso = iso.predict(pv_raw); pt_iso = iso.predict(pt_raw)

    # Honest method selection: isotonic overfits its own fit set (ECE ~0 there), so
    # picking on full-val ECE is biased toward it. Select on a held-out 30% of val
    # (fit calibrators on 70%, score the held-out 30%); the FINAL calibrator for the
    # chosen method is still the full-val fit applied to test.
    rng = np.random.RandomState(0); perm = rng.permutation(len(yva)); cut = int(0.7 * len(yva))
    fi, si = perm[:cut], perm[cut:]
    Ts = _fit_temperature(zv[fi], yva[fi])
    isos = IsotonicRegression(out_of_bounds="clip"); isos.fit(pv_raw[fi], yva[fi])
    ece_v = {"raw": _ece(yva[si], pv_raw[si])[0],
             "temp": _ece(yva[si], 1 / (1 + np.exp(-zv[si] / Ts)))[0],
             "iso": _ece(yva[si], isos.predict(pv_raw[si]))[0]}
    best = min(ece_v, key=ece_v.get)                 # picked on held-out val (no test leakage)
    pt_best = {"raw": pt_raw, "temp": pt_temp, "iso": pt_iso}[best]

    ece_raw, tbl_raw = _ece(yte, pt_raw)
    ece_best, tbl_best = _ece(yte, pt_best)
    rep = {
        "model": args.model, "n_test": int(len(yte)), "test_base_rate": round(float(yte.mean()), 4),
        "selected_method": best, "temperature": round(T, 4),
        "val_ece": {k: round(v, 4) for k, v in ece_v.items()},
        "test_ece_raw": round(ece_raw, 4), "test_ece_calibrated": round(ece_best, 4),
        "test_brier_raw": round(_brier(yte, pt_raw), 4),
        "test_brier_calibrated": round(_brier(yte, pt_best), 4),
        "reliability_raw": tbl_raw, "reliability_calibrated": tbl_best,
    }
    print(f"  deep model {args.model}")
    print(f"  held-out val ECE: raw {ece_v['raw']:.4f}  temp(T={T:.2f}) {ece_v['temp']:.4f}  iso {ece_v['iso']:.4f}")
    print(f"  --> selected '{best}' (lowest held-out val ECE)")
    print(f"  TEST ECE:   raw {ece_raw:.4f}  ->  calibrated {ece_best:.4f}")
    print(f"  TEST Brier: raw {rep['test_brier_raw']:.4f} -> calibrated {rep['test_brier_calibrated']:.4f}")

    # persist the calibrator constants alongside the model (for serving)
    calib = {"method": best, "temperature": T}
    if best == "iso":
        calib["iso_x"] = iso.f_.x.tolist(); calib["iso_y"] = iso.f_.y.tolist()
    cp = (REPO / args.model).with_suffix(".calib.json")
    cp.write_text(json.dumps(calib) + "\n", encoding="utf-8")
    out = REPO / "results" / "calibration" / "earthquake_deep_calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(f"  saved calibrator -> {cp.name}; report -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
