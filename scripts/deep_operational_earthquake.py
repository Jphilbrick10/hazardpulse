#!/usr/bin/env python3
"""Train the deep model DIRECTLY on the OPERATIONAL task: "among all active cells right now,
which one actually ruptures in the forward window?" -- the real life-saving question.

The case-control model is trained against SAME-LOCATION controls, so it learns location-
*relative* criticality and cannot rank one place against another (operational AUC ~0.60).
Here the negatives are OTHER ACTIVE CELLS (cross-location), which forces cross-location
ranking, and we feed the model LOCATION + BASELINE context (lat/lon, long-run rate, recent
anomaly) so "a normally-quiet fault now lit up" can outrank "an always-noisy zone."

Sampling: monthly reference times; at each, every active 2-deg cell (>= --active-min-ev
events within --active-radius in the prior --active-days). Per (cell, time): input = K
most-recent events within --input-radius before t; label = declustered M>=--label-mag
within --label-radius / --label-days FORWARD. Temporal split (train<=2017, val 18-19,
test 20-25). Reports the POOLED OPERATIONAL AUC on held-out test times -- the honest metric.

    python scripts/deep_operational_earthquake.py --label-mag 5.0 --label-radius 100 --label-days 30 --seeds 3
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
SEC_DAY = 86400.0
# train/val/test boundaries (epoch seconds, UTC) -- match definitive_model's year split
_VAL0 = 1514764800.0   # 2018-01-01
_TEST0 = 1577836800.0  # 2020-01-01


def _auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


_CAT = None
_MS = None
_CFG = None


def _winit(max_year, cfg):
    global _CAT, _MS, _CFG
    from hazardpulse.earthquake.definitive_model import (
        load_usgs_catalog, CatalogArrays, decluster_gardner_knopoff)
    cl = load_usgs_catalog(min_year=2000, max_year=max_year, min_mag=2.5)
    _CAT = CatalogArrays(cl, verbose=False)
    ms, _ = decluster_gardner_knopoff(cl)
    import datetime as dt
    _MS = np.array([[e["latitude"], e["longitude"],
                     dt.datetime.fromisoformat(e["time"].replace("Z", "+00:00")).timestamp()]
                    for e in ms if e.get("mag", 0) >= cfg["label_mag"]])
    _CFG = cfg


def _haz(lat, lon, lats, lons):
    rlat, rlon = np.radians(lat), np.radians(lon)
    rla, rlo = np.radians(lats), np.radians(lons)
    dlon = rlo - rlon
    a = np.sin((rla - rlat) / 2) ** 2 + np.cos(rlat) * np.cos(rla) * np.sin(dlon / 2) ** 2
    dist = 6371.0 * 2 * np.arcsin(np.sqrt(a))
    az = np.arctan2(np.sin(dlon) * np.cos(rla),
                    np.cos(rlat) * np.sin(rla) - np.sin(rlat) * np.cos(rla) * np.cos(dlon))
    return dist, az


def _build_one(arg):
    """arg=(lat,lon,ref). Returns (X[K,9], mask[K], label, ref) or None."""
    lat, lon, ref = arg
    cat, ms, cfg = _CAT, _MS, _CFG
    K = cfg["K"]; R = cfg["input_radius"]
    X = np.zeros((K, 9), np.float32); m = np.zeros(K, np.float32)
    t0 = ref - 5 * 365 * SEC_DAY
    sel = ((cat.times >= t0) & (cat.times < ref)
           & (np.abs(cat.lats - lat) < 6) & (np.abs(cat.lons - lon) < 6))
    idx = np.where(sel)[0]
    if idx.size == 0:
        return None
    dist, az = _haz(lat, lon, cat.lats[idx], cat.lons[idx])
    near = dist < R
    idx, dist, az = idx[near], dist[near], az[near]
    if idx.size == 0:
        return None
    order = np.argsort(cat.times[idx])[-K:]
    idx, dist, az = idx[order], dist[order], az[order]
    dd = (ref - cat.times[idx]) / SEC_DAY
    # per-event seismicity channels (6) + cross-location context channels (3):
    #   lat/90, lon/180 (absolute location), and log1p(recent rate) -- the same value on
    #   every step so the GRU/attention can read "where am I + how active am I overall".
    n_1yr = float(((ref - cat.times[idx]) < 365 * SEC_DAY).sum())
    loc_lat = lat / 90.0; loc_lon = lon / 180.0; rate = np.log1p(n_1yr) / 6.0
    seq = np.stack([np.log1p(dd), cat.mags[idx], dist / R,
                    np.clip(cat.depths[idx], 0, 700) / 700.0, np.sin(az), np.cos(az),
                    np.full(len(idx), loc_lat, np.float32),
                    np.full(len(idx), loc_lon, np.float32),
                    np.full(len(idx), rate, np.float32)], axis=1)
    X[K - len(idx):] = seq; m[K - len(idx):] = 1.0
    # forward label
    lab = 0
    if len(ms):
        d = _haz(lat, lon, ms[:, 0], ms[:, 1])[0]
        fwd = (ms[:, 2] > ref) & (ms[:, 2] <= ref + cfg["label_days"] * SEC_DAY)
        lab = int(((d < cfg["label_radius"]) & fwd).any())
    return X, m, np.int8(lab), np.float64(ref)


def _active_cells(cat, t, g, active_radius, active_days, active_min):
    """Cells with >= active_min events in the prior active_days (cheap 2-deg-bin proxy)."""
    recent = (cat.times > t - active_days * SEC_DAY) & (cat.times < t)
    cells = {}
    for la, lo in zip(cat.lats[recent], cat.lons[recent]):
        k = (round(la / g) * g, round(lo / g) * g)
        cells[k] = cells.get(k, 0) + 1
    return [c for c, n in cells.items() if n >= active_min]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-year", type=int, default=2025)
    ap.add_argument("--K", type=int, default=192)
    ap.add_argument("--input-radius", type=float, default=100.0)
    ap.add_argument("--label-mag", type=float, default=5.0)
    ap.add_argument("--label-radius", type=float, default=100.0)
    ap.add_argument("--label-days", type=float, default=30.0)
    ap.add_argument("--active-min-ev", type=int, default=8)
    ap.add_argument("--active-days", type=float, default=180.0)
    ap.add_argument("--grid", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results/calibration/earthquake_deep_operational.json")
    args = ap.parse_args(argv)

    import torch
    import torch.nn as nn
    from concurrent.futures import ProcessPoolExecutor
    from hazardpulse.earthquake.definitive_model import load_usgs_catalog, CatalogArrays
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")

    cfg = dict(K=args.K, input_radius=args.input_radius, label_mag=args.label_mag,
               label_radius=args.label_radius, label_days=args.label_days)
    tag = f"op_my{args.max_year}_m{args.label_mag}_lr{args.label_radius:.0f}_ld{args.label_days:.0f}_K{args.K}_ir{args.input_radius:.0f}"
    cache = REPO / ".cache" / "earthquake" / f"deep{tag}.npz"
    if cache.exists() and os.environ.get("HAZARDPULSE_OP_REBUILD") != "1":
        z = np.load(cache)
        X, M, Y, T = z["X"], z["M"], z["Y"], z["T"]
        print(f"  [cache] loaded {len(Y)} operational samples from {cache.name}")
    else:
        cl = load_usgs_catalog(min_year=2000, max_year=args.max_year, min_mag=2.5)
        cat = CatalogArrays(cl, verbose=False)
        # monthly reference times 2005-01 .. (max_year)-?, label window must fit in catalog
        cat_end = float(cat.times.max())
        refs = []
        import datetime as dt
        for yr in range(2005, args.max_year + 1):
            for mo in range(1, 13):
                t = dt.datetime(yr, mo, 1, tzinfo=dt.timezone.utc).timestamp()
                if t + args.label_days * SEC_DAY <= cat_end:
                    refs.append(t)
        print(f"  building operational samples over {len(refs)} monthly snapshots "
              f"(active cells x time, x{args.workers} workers)...", flush=True)
        tasks = []
        for t in refs:
            for (la, lo) in _active_cells(cat, t, args.grid, args.input_radius,
                                          args.active_days, args.active_min_ev):
                tasks.append((la, lo, t))
        print(f"  {len(tasks)} (cell,time) candidate samples", flush=True)
        del cat, cl
        Xs, Ms, Ys, Ts = [], [], [], []
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=args.workers, initializer=_winit,
                                 initargs=(args.max_year, cfg)) as ex:
            for i, r in enumerate(ex.map(_build_one, tasks, chunksize=64)):
                if r is not None:
                    Xs.append(r[0]); Ms.append(r[1]); Ys.append(r[2]); Ts.append(r[3])
                if (i + 1) % 20000 == 0 or (i + 1) == len(tasks):
                    rate = (i + 1) / max(time.time() - t0, 1e-9)
                    print(f"      {i+1}/{len(tasks)} ({100*(i+1)//len(tasks)}%, {rate:.0f}/s, "
                          f"ETA {(len(tasks)-i-1)/rate/60:.1f}min)", flush=True)
        X = np.stack(Xs); M = np.stack(Ms); Y = np.array(Ys, np.int8); T = np.array(Ts)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, X=X, M=M, Y=Y, T=T)
        print(f"  [cache] saved {len(Y)} samples -> {cache.name}")

    tr = T < _VAL0; va = (T >= _VAL0) & (T < _TEST0); te = T >= _TEST0
    print(f"  train {tr.sum()} ({Y[tr].sum()} pos, {Y[tr].mean():.3%}) | "
          f"val {va.sum()} ({Y[va].sum()} pos) | test {te.sum()} ({Y[te].sum()} pos, {Y[te].mean():.3%})")
    if Y[te].sum() == 0:
        print("  no positive test samples -- adjust label window."); return 2

    # normalize per-channel by train stats
    flat = X[tr][M[tr].astype(bool)]
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)

    class OpModel(nn.Module):
        def __init__(self, d=9, h=96):
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

    def T_(a):
        return torch.tensor(a, device=dev)
    Xtr, Mtr, ytr = T_(Xn[tr]), T_(M[tr]), T_(Y[tr].astype(np.float32))
    Xva, Mva = T_(Xn[va]), T_(M[va])
    Xte, Mte = T_(Xn[te]), T_(M[te])
    yte = Y[te]; yva = Y[va]

    def predict(model, Xt, Mt, bs=4096):
        out = []
        with torch.no_grad():
            for j in range(0, len(Xt), bs):
                out.append(torch.sigmoid(model(Xt[j:j + bs], Mt[j:j + bs])))
        return torch.cat(out).cpu().numpy()

    pos_w = T_(np.array([(Y[tr] == 0).sum() / max((Y[tr] == 1).sum(), 1)], np.float32))
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    seed_aucs = []; ens = np.zeros(te.sum())
    for seed in range(args.seeds):
        torch.manual_seed(seed)
        model = OpModel().to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best_va, best_state = 0.0, None
        bs = 512
        for ep in range(args.epochs):
            model.train(); perm = torch.randperm(len(ytr), device=dev)
            for j in range(0, len(perm), bs):
                b = perm[j:j + bs]; opt.zero_grad()
                loss = lossf(model(Xtr[b], Mtr[b]), ytr[b]); loss.backward(); opt.step()
            model.eval()
            a = _auc(yva, predict(model, Xva, Mva))
            if a > best_va:
                best_va = a; best_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state); model.eval()
        p = predict(model, Xte, Mte)
        au = _auc(yte, p); seed_aucs.append(au); ens += p
        print(f"  seed {seed}: TEST operational AUC {au:.4f} (best val {best_va:.4f})")
    import json
    op_auc = float(np.mean(seed_aucs)); ens_auc = _auc(yte, ens / args.seeds)
    print(f"\n  OPERATIONAL deep model (cross-location ranking, +location context): "
          f"test AUC {op_auc:.4f} +/- {np.std(seed_aucs):.4f}  (ensemble {ens_auc:.4f})")
    print(f"  baseline to beat: case-control model scored ~0.60 operationally (M6+) / 0.64 (M4.5+).")
    rep = {"label": f"M{args.label_mag}+/{args.label_radius:.0f}km/{args.label_days:.0f}d",
           "n_train": int(tr.sum()), "n_test": int(te.sum()), "test_pos": int(yte.sum()),
           "test_pos_rate": round(float(yte.mean()), 4),
           "operational_auc_mean": round(op_auc, 4), "operational_auc_std": round(float(np.std(seed_aucs)), 4),
           "operational_auc_seeds": [round(a, 4) for a in seed_aucs], "ensemble_auc": round(float(ens_auc), 4)}
    out = REPO / args.out; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
