#!/usr/bin/env python3
"""Let the ML discover precursors from the RAW event stream (no hand-crafted features).

Everything else feeds a GBT 73 human-designed features. This instead feeds a sequence
model the RAW sequence of the K most recent events before each sample -- each event just
[time-before, magnitude, distance, depth, azimuth], no aggregation -- and lets it learn
whatever precursory pattern exists. If it beats / complements the 0.77 hand-crafted
model, there is structure we never thought to measure. Strictly causal (events < ev_time).
Trains on the GPU.

    python scripts/deep_sequence_earthquake.py --max-year 2024 --epochs 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

SEC_DAY = 86400.0


def _auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    P = float((y == 1).sum()); N = float((y == 0).sum())
    if P == 0 or N == 0:
        return float("nan")
    order = np.argsort(-s); ys = y[order]
    tp = fp = a = tpp = fpp = 0.0
    for l in ys:
        tp += (l == 1); fp += (l == 0)
        a += (fp / N - fpp / N) * (tp / P + tpp / P) / 2; tpp, fpp = tp, fp
    return a


def _meta(max_year):
    """(lat,lon,epoch,label) for train(downsampled)+val+test in cached order."""
    from hazardpulse.earthquake.definitive_model import (
        load_usgs_catalog, decluster_gardner_knopoff, build_samples, CatalogArrays,
        TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, TEST_END)
    catalog = load_usgs_catalog(min_year=2000, max_year=max_year, min_mag=2.5)
    cat = CatalogArrays(catalog, verbose=False)
    ms, _ = decluster_gardner_knopoff(catalog)
    samples = build_samples(ms, catalog, verbose=False)
    te_end = max(TEST_END, int(max_year))
    tr = [s for s in samples if TRAIN_START <= s["year"] <= TRAIN_END]
    va = [s for s in samples if VAL_START <= s["year"] <= VAL_END]
    teh = [s for s in samples if TEST_START <= s["year"] <= te_end]
    ytr = np.array([s["label"] for s in tr])
    npos = int(ytr.sum())
    if len(ytr) - npos > 5 * npos:
        rng = np.random.RandomState(42)
        keep = np.sort(np.concatenate([np.where(ytr == 1)[0],
                       rng.choice(np.where(ytr == 0)[0], 5 * npos, replace=False)]))
        tr = [tr[i] for i in keep]

    def pack(ss):
        return (np.array([s["latitude"] for s in ss], float),
                np.array([s["longitude"] for s in ss], float),
                np.array([s["ref_epoch"] for s in ss], float),
                np.array([s["label"] for s in ss], int))
    return pack(tr), pack(va), pack(teh), cat


def _haversine(lat, lon, lats, lons):
    lat, lon = np.radians(lat), np.radians(lon)
    lats, lons = np.radians(lats), np.radians(lons)
    dlat, dlon = lats - lat, lons - lon
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lats) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a)), np.arctan2(
        np.sin(dlon) * np.cos(lats), np.cos(lat) * np.sin(lats) - np.sin(lat) * np.cos(lats) * np.cos(dlon))


def _sequences(meta, cat, K, radius_km):
    """For each sample, the K most-recent events < ev_time within radius -> (N,K,6) + mask."""
    lat, lon, ep, y = meta
    n = len(lat)
    X = np.zeros((n, K, 6), np.float32); M = np.zeros((n, K), np.float32)
    for i in range(n):
        t0 = ep[i] - 5 * 365 * SEC_DAY
        sel = (cat.times >= t0) & (cat.times < ep[i]) & \
              (np.abs(cat.lats - lat[i]) < 6) & (np.abs(cat.lons - lon[i]) < 6)
        idx = np.where(sel)[0]
        if idx.size == 0:
            continue
        d, az = _haversine(lat[i], lon[i], cat.lats[idx], cat.lons[idx])
        near = d < radius_km
        idx, d, az = idx[near], d[near], az[near]
        if idx.size == 0:
            continue
        order = np.argsort(cat.times[idx])[-K:]            # most recent K, chronological
        idx, d, az = idx[order], d[order], az[order]
        dt = (ep[i] - cat.times[idx]) / SEC_DAY
        seq = np.stack([np.log1p(dt), cat.mags[idx], d / radius_km,
                        np.clip(cat.depths[idx], 0, 700) / 700.0,
                        np.sin(az), np.cos(az)], axis=1)
        X[i, K - len(idx):] = seq; M[i, K - len(idx):] = 1.0
    return X, M, y


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-year", type=int, default=2024)
    ap.add_argument("--K", type=int, default=48)
    ap.add_argument("--radius-km", type=float, default=500.0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=1, help="train N seeds -> AUC mean+/-std + ensemble")
    ap.add_argument("--arch", default="gru", choices=["gru", "transformer"])
    ap.add_argument("--out", default="results/calibration/earthquake_deep_sequence.json")
    args = ap.parse_args(argv)

    import torch
    import torch.nn as nn
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")
    import os
    mag = os.environ.get("HAZARDPULSE_MIN_MAINSHOCK_MAG", "6.0")
    seq_cache = REPO / ".cache" / "earthquake" / f"deepseq_my{args.max_year}_m{mag}_K{args.K}.npz"
    if seq_cache.exists() and os.environ.get("HAZARDPULSE_DEEPSEQ_REBUILD") != "1":
        z = np.load(seq_cache)
        Xtr, Mtr, ytr = z["Xtr"], z["Mtr"], z["ytr"]
        Xva, Mva, yva = z["Xva"], z["Mva"], z["yva"]
        Xte, Mte, yte = z["Xte"], z["Mte"], z["yte"]
        print(f"  [cache] loaded sequences from {seq_cache.name}")
    else:
        print("reconstructing metadata + building raw sequences (causal)...")
        mtr, mva, mte, cat = _meta(args.max_year)
        Xtr, Mtr, ytr = _sequences(mtr, cat, args.K, args.radius_km)
        Xva, Mva, yva = _sequences(mva, cat, args.K, args.radius_km)
        Xte, Mte, yte = _sequences(mte, cat, args.K, args.radius_km)
        seq_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(seq_cache, Xtr=Xtr, Mtr=Mtr, ytr=ytr, Xva=Xva, Mva=Mva, yva=yva,
                            Xte=Xte, Mte=Mte, yte=yte)
        print(f"  [cache] saved sequences -> {seq_cache.name}")
    print(f"  train {len(ytr)} ({ytr.sum()} pos)  val {len(yva)}  test {len(yte)} ({yte.sum()} pos)")
    # normalize features by train stats (per channel)
    flat = Xtr[Mtr.astype(bool)]
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    norm = lambda X: ((X - mu) / sd).astype(np.float32)
    Xtr, Xva, Xte = norm(Xtr), norm(Xva), norm(Xte)

    class SeqModel(nn.Module):
        def __init__(self, d=6, h=64):
            super().__init__()
            self.proj = nn.Linear(d, h)
            self.gru = nn.GRU(h, h, batch_first=True, bidirectional=True)
            self.att = nn.Linear(2 * h, 1)
            self.head = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Dropout(0.3), nn.Linear(h, 1))

        def forward(self, x, m):
            z = torch.relu(self.proj(x))
            z, _ = self.gru(z)
            a = self.att(z).squeeze(-1).masked_fill(m == 0, -1e9)
            a = torch.softmax(a, 1).unsqueeze(-1)
            return self.head((z * a).sum(1)).squeeze(-1)

    class TransformerSeq(nn.Module):
        def __init__(self, d=6, h=64, nlayers=2, nhead=4):
            super().__init__()
            self.proj = nn.Linear(d, h)
            self.pos = nn.Parameter(torch.randn(1, args.K, h) * 0.02)   # learned positional
            enc = nn.TransformerEncoderLayer(h, nhead, 4 * h, dropout=0.3, batch_first=True)
            self.tf = nn.TransformerEncoder(enc, nlayers)
            self.att = nn.Linear(h, 1)
            self.head = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Dropout(0.3), nn.Linear(h, 1))

        def forward(self, x, m):
            z = self.proj(x) + self.pos
            z = self.tf(z, src_key_padding_mask=(m == 0))
            a = self.att(z).squeeze(-1).masked_fill(m == 0, -1e9)
            a = torch.softmax(a, 1).unsqueeze(-1)
            return self.head((z * a).sum(1)).squeeze(-1)

    Arch = TransformerSeq if args.arch == "transformer" else SeqModel

    def tens(X, M, y):
        return (torch.tensor(X, device=dev), torch.tensor(M, device=dev),
                torch.tensor(y, dtype=torch.float32, device=dev))
    Xtr_t, Mtr_t, ytr_t = tens(Xtr, Mtr, ytr)
    Xva_t, Mva_t, _ = tens(Xva, Mva, yva)
    Xte_t, Mte_t, _ = tens(Xte, Mte, yte)

    pos_w = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)], device=dev, dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    bs = 256
    seed_aucs, ens = [], np.zeros(len(yte))
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        model = Arch().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best_va, best_state = 0.0, None
        for ep in range(args.epochs):
            model.train(); perm = torch.randperm(len(ytr_t), device=dev)
            for j in range(0, len(perm), bs):
                b = perm[j:j + bs]
                opt.zero_grad()
                loss = lossf(model(Xtr_t[b], Mtr_t[b]), ytr_t[b]); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                va = _auc(yva, torch.sigmoid(model(Xva_t, Mva_t)).cpu().numpy())
            if va > best_va:
                best_va = va; best_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            pte = torch.sigmoid(model(Xte_t, Mte_t)).cpu().numpy()
        a = _auc(yte, pte); seed_aucs.append(a); ens += pte
        print(f"  seed {seed}: test AUC {a:.4f} (best val {best_va:.4f})")
    deep_auc = float(np.mean(seed_aucs)); ens_auc = _auc(yte, ens / args.seeds)
    print(f"\n  DEEP sequence model (raw events, GRU+attention): "
          f"test AUC {deep_auc:.4f} +/- {np.std(seed_aucs):.4f}  (ensemble {ens_auc:.4f})")
    print(f"  hand-crafted GBT reference on same task: ~0.77")

    import json
    rep = {"max_year": args.max_year, "K": args.K, "seeds": args.seeds,
           "deep_auc_mean": round(deep_auc, 4), "deep_auc_std": round(float(np.std(seed_aucs)), 4),
           "deep_auc_seeds": [round(a, 4) for a in seed_aucs],
           "ensemble_auc": round(float(ens_auc), 4), "n_test": int(len(yte))}
    out = REPO / args.out; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
