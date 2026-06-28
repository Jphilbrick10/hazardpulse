#!/usr/bin/env python3
"""Export a trained deep EQ nowcast (.pt) to a pure-numpy .npz for torch-free serving,
and VERIFY the numpy forward (hazardpulse.earthquake.deep_serve) matches PyTorch on the
held-out test sequences. Refuses to write if max|diff| exceeds 1e-4.

    python scripts/export_deep_eq_serve.py --model results/models/eq_deep_nowcast_m5.0_2025_K192.pt --mag 5.0 --max-year 2025 --K 192
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--mag", default="5.0")
    ap.add_argument("--max-year", type=int, default=2025)
    ap.add_argument("--K", type=int, default=192)
    ap.add_argument("--seq-cache", default="",
                    help="explicit sequence cache .npz for verification (short-term products "
                         "have label/input-radius suffixes the default name omits)")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    import torch
    import torch.nn as nn
    from hazardpulse.earthquake.deep_serve import DeepEQScorer

    ck = torch.load(REPO / args.model, map_location="cpu", weights_only=False)
    h = 64

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

    net = SeqModel(h=h); net.load_state_dict(ck["state_dict"]); net.eval()
    g = net.gru

    def npy(t):
        return t.detach().cpu().numpy().astype(np.float64)

    arrays = dict(
        proj_w=npy(net.proj.weight), proj_b=npy(net.proj.bias),
        gru_ih=npy(g.weight_ih_l0), gru_hh=npy(g.weight_hh_l0),
        gru_bih=npy(g.bias_ih_l0), gru_bhh=npy(g.bias_hh_l0),
        gru_ih_r=npy(g.weight_ih_l0_reverse), gru_hh_r=npy(g.weight_hh_l0_reverse),
        gru_bih_r=npy(g.bias_ih_l0_reverse), gru_bhh_r=npy(g.bias_hh_l0_reverse),
        att_w=npy(net.att.weight), att_b=npy(net.att.bias),
        head_w1=npy(net.head[0].weight), head_b1=npy(net.head[0].bias),
        head_w2=npy(net.head[3].weight), head_b2=npy(net.head[3].bias),
        norm_mu=np.asarray(ck["norm_mu"], np.float64), norm_sd=np.asarray(ck["norm_sd"], np.float64),
        K=np.int64(ck.get("K", args.K)), radius_km=np.float64(ck.get("radius_km", 500.0)),
    )
    out = REPO / (args.out or args.model.replace(".pt", ".serve.npz"))
    np.savez(out, **arrays)
    print(f"  wrote {out.name}")

    # --- verify numpy forward == torch on the test sequences ---
    seq_path = (REPO / args.seq_cache) if args.seq_cache else (
        REPO / ".cache" / "earthquake" / f"deepseq_my{args.max_year}_m{args.mag}_K{args.K}.npz")
    dz = np.load(seq_path)
    Xte, Mte = dz["Xte"], dz["Mte"]
    mu, sd = ck["norm_mu"], ck["norm_sd"]
    n = min(500, len(Xte))
    Xn = ((Xte[:n] - mu) / sd).astype(np.float32)
    with torch.no_grad():
        p_torch = torch.sigmoid(net(torch.tensor(Xn), torch.tensor(Mte[:n]))).numpy()
    scorer = DeepEQScorer(out)
    p_np = np.array([scorer.forward(Xte[i], Mte[i]) for i in range(n)])  # raw (no calib) -- calib not loaded
    md = float(np.max(np.abs(p_torch - p_np)))
    print(f"  numpy-vs-torch max|prob diff| over {n} test seqs: {md:.3e}")
    if md > 1e-4:
        print("  FAIL: exceeds 1e-4 tolerance -- not deploying this export.")
        out.unlink(missing_ok=True)
        return 2
    print("  OK: numpy serve matches torch within tolerance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
