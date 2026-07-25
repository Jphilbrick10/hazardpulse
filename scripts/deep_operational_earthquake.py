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
_GLA = _GLO = _GM0 = _GT = None   # causal GCMT arrays (lat, lon, scalar moment, epoch)


def _extra_feats(lat, lon, ref, cat, bi, bd, bdays):
    """The 4 levers that showed real operational signal (ETAS intensity +0.011, b-value,
    AMR curvature, Coulomb-from-GCMT). Computed from the already-selected box events `bi`."""
    mags = cat.mags[bi]
    # ETAS conditional intensity (Ogata), 100 km, 30-day expected count
    inr = bd < 100
    if inr.any():
        dd = np.maximum(bdays[inr], 0.0); mg = mags[inr]
        aft = (0.018 * 10 ** (0.9 * (mg - 2.5)) / ((dd + 0.01) ** 1.10)).sum()
        bg = ((bdays >= 365) & (bdays < 5 * 365) & inr).sum() / (4 * 365.0)
        etas = np.log1p((bg + aft) * 30.0)
    else:
        etas = 0.0
    # Gutenberg-Richter b-value (Aki MLE), 150 km / 5 yr
    sb = (bd < 150) & (bdays < 5 * 365); mm = mags[sb]
    bval = (np.log10(np.e) / (mm.mean() - (mm.min() - 0.05))) if (mm.size >= 25 and mm.mean() > mm.min()) else 1.0
    # accelerating moment release (Benioff strain curvature), 250 km / 3 yr
    sa = (bd < 250) & (bdays < 3 * 365)
    if sa.sum() >= 15:
        ee = np.sqrt(10.0 ** (1.5 * mags[sa] + 4.8)); tt = bdays[sa]
        o = np.argsort(-tt); S = np.cumsum(ee[o]); x = -tt[o]
        A = np.vstack([x, np.ones_like(x)]).T
        cf = np.linalg.lstsq(A, S, rcond=None)[0]; resid = S - A @ cf
        amr = float(resid[-1] / (S[-1] + 1e-9))
    else:
        amr = 0.0
    # Coulomb stress proxy from prior nearby GCMT M6+ events (sum of M0 / r^3), 500 km.
    # Rows without origin time are excluded at load time; using the full GCMT catalog here would
    # leak future focal mechanisms into old forecasts.
    if _GLA is not None and _GLA.size:
        causal = _GT < ref
        gm0 = _GM0[causal]
        gd = _haz(lat, lon, _GLA[causal], _GLO[causal])[0] if causal.any() else np.array([])
        ing = gd < 500
        coul = float(np.log1p((gm0[ing] / np.maximum(gd[ing] * 1000.0, 5000.0) ** 3).sum())) if ing.any() else 0.0
    else:
        coul = 0.0
    return [float(etas) / 5.0, float(bval) / 2.0, float(np.clip(amr, -1, 1)), coul / 5.0]


def _epoch_from_iso_z(value):
    if not value:
        return None
    import datetime as dt
    try:
        return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _winit(max_year, cfg):
    global _CAT, _MS, _CFG
    from hazardpulse.earthquake.definitive_model import (
        load_usgs_catalog, CatalogArrays, decluster_gardner_knopoff)
    mim = cfg.get("min_input_mag", 2.5)
    # input feature catalog at min_input_mag (M2.0 = richer foreshock detail). Labels are
    # declustered M5+ mainshocks, identical for M2.0 vs M2.5 input (GK breaks at M<5), so we
    # decluster a M2.5-filtered view to keep that step cheap and the label set comparable.
    cl = load_usgs_catalog(min_year=2000, max_year=max_year, min_mag=mim)
    _CAT = CatalogArrays(cl, verbose=False)
    cl_dec = [e for e in cl if (e.get("mag") or 0) >= 2.5] if mim < 2.5 else cl
    ms, _ = decluster_gardner_knopoff(cl_dec)
    import datetime as dt
    _MS = np.array([[e["latitude"], e["longitude"],
                     dt.datetime.fromisoformat(e["time"].replace("Z", "+00:00")).timestamp()]
                    for e in ms if e.get("mag", 0) >= cfg["label_mag"]])
    _CFG = cfg
    global _GLA, _GLO, _GM0, _GT
    _GLA = _GLO = _GM0 = _GT = None
    if cfg.get("extra_features"):
        try:
            from hazardpulse.data.earthquake import load_gcmt_catalog
            gla, glo, gm0, gt = [], [], [], []
            for r in load_gcmt_catalog():
                try:
                    la = float(r["lat"]); lo = float(r["lon"]); mw = float(r["Mw"])
                except Exception:
                    continue
                if mw < 6.0:
                    continue
                epoch = _epoch_from_iso_z(r.get("time"))
                if epoch is None:
                    continue
                try:
                    moment = float(r.get("scalar_moment") or 10.0 ** (1.5 * mw + 9.1))
                except Exception:
                    moment = 10.0 ** (1.5 * mw + 9.1)
                gla.append(la); glo.append(lo); gm0.append(moment); gt.append(epoch)
            _GLA = np.array(gla); _GLO = np.array(glo); _GM0 = np.array(gm0); _GT = np.array(gt)
        except Exception:
            _GLA = np.array([]); _GLO = np.array([]); _GM0 = np.array([]); _GT = np.array([])


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
    # --- MULTI-SCALE cross-location context: catch quiescence (a drop) + the regional field
    # + stress transfer -- the tells a local-active-only model misses. Broad query first. ---
    bsel = ((cat.times < ref) & (np.abs(cat.lats - lat) < 10) & (np.abs(cat.lons - lon) < 10))
    bi = np.where(bsel)[0]
    if bi.size == 0:
        return None
    bd = _haz(lat, lon, cat.lats[bi], cat.lons[bi])[0]
    bdays = (ref - cat.times[bi]) / SEC_DAY
    ctx = [lat / 90.0, lon / 180.0]
    for rad, rec_n, base_n in [(100, 5.0, 8.0), (300, 6.0, 9.0), (500, 7.0, 10.0)]:
        inr = bd < rad
        rec = float(((bdays < 180) & inr).sum())            # recent (180d) rate at this scale
        base = float(((bdays >= 365) & (bdays < 5 * 365) & inr).sum()) / 4.0 / 2.0  # 180d-equiv baseline
        ctx.append(np.log1p(rec) / rec_n)
        ctx.append(np.log1p(((bdays < 5 * 365) & inr).sum()) / base_n)
        ctx.append(np.clip(np.log1p(rec) - np.log1p(base), -3, 3) / 3.0)  # anomaly: <0 quiescence, >0 ramp
    maxmag = float(cat.mags[bi[bd < 300]].max()) if (bd < 300).any() else 0.0
    ctx.append(maxmag / 9.0)
    big = (bd < 1000) & (cat.mags[bi] >= 6.5) & (bdays < 2 * 365)   # stress transfer from a recent big quake
    if big.any():
        nb = np.argmin(bdays[big]); ctx.append((1000 - bd[big][nb]) / 1000.0)
        ctx.append(max(0.0, 1 - bdays[big][nb] / 730.0))
    else:
        ctx.append(0.0); ctx.append(0.0)
    if cfg.get("extra_features"):
        ctx = ctx + _extra_feats(lat, lon, ref, cat, bi, bd, bdays)
    NCH = 6 + len(ctx)
    X = np.zeros((K, NCH), np.float32); m = np.zeros(K, np.float32)
    # --- per-event local sequence (most-recent K within R / 5yr) ---
    t0 = ref - 5 * 365 * SEC_DAY
    sel = ((cat.times >= t0) & (cat.times < ref)
           & (np.abs(cat.lats - lat) < 6) & (np.abs(cat.lons - lon) < 6))
    idx = np.where(sel)[0]
    if idx.size:
        dist, az = _haz(lat, lon, cat.lats[idx], cat.lons[idx])
        near = dist < R
        idx, dist, az = idx[near], dist[near], az[near]
    if idx.size:
        order = np.argsort(cat.times[idx])[-K:]
        idx, dist, az = idx[order], dist[order], az[order]
        dd = (ref - cat.times[idx]) / SEC_DAY
        seq = np.stack([np.log1p(dd), cat.mags[idx], dist / R,
                        np.clip(cat.depths[idx], 0, 700) / 700.0, np.sin(az), np.cos(az)]
                       + [np.full(len(idx), c, np.float32) for c in ctx], axis=1)
        X[K - len(idx):] = seq; m[K - len(idx):] = 1.0
    else:
        X[K - 1] = np.array([0, 0, 0, 0, 0, 0] + ctx, np.float32); m[K - 1] = 1.0  # context-only (quiescent cell)
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
    ap.add_argument("--min-input-mag", type=float, default=2.5,
                    help="min magnitude for the INPUT feature catalog (2.0 w/ HAZARDPULSE_USGS_FULL=1 "
                         "= richer foreshock sequences; labels/active-cells stay M2.5-comparable)")
    ap.add_argument("--extra-features", action="store_true",
                    help="append 4 physics channels (ETAS intensity, b-value, AMR curvature, "
                         "Coulomb-from-GCMT) that showed real operational signal")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--save-model", default="", help="save the best-seed model (.pt) for serving")
    ap.add_argument("--out", default="results/calibration/earthquake_deep_operational.json")
    args = ap.parse_args(argv)

    import torch
    import torch.nn as nn
    from concurrent.futures import ProcessPoolExecutor
    from hazardpulse.earthquake.definitive_model import load_usgs_catalog, CatalogArrays
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {dev}")

    cfg = dict(K=args.K, input_radius=args.input_radius, label_mag=args.label_mag,
               label_radius=args.label_radius, label_days=args.label_days,
               min_input_mag=args.min_input_mag, extra_features=args.extra_features)
    _mimtag = "" if args.min_input_mag == 2.5 else f"_mim{args.min_input_mag}"
    _xftag = "_xf" if args.extra_features else ""
    tag = f"op_v3_my{args.max_year}_m{args.label_mag}_lr{args.label_radius:.0f}_ld{args.label_days:.0f}_K{args.K}_ir{args.input_radius:.0f}_am{args.active_min_ev}_g{args.grid:.0f}{_mimtag}{_xftag}"
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

    # walk-forward: --test-start-year re-splits the SAME cached samples (no rebuild) so we
    # can confirm the result holds across multiple temporal splits, not one lucky cut.
    tsy = int(os.environ.get("HAZARDPULSE_TEST_START_YEAR", "0"))
    if tsy:
        import datetime as _dt
        test0 = _dt.datetime(tsy, 1, 1, tzinfo=_dt.timezone.utc).timestamp()
        val0 = _dt.datetime(tsy - 2, 1, 1, tzinfo=_dt.timezone.utc).timestamp()
        print(f"  [walk-forward] split @ test-start {tsy}: train<{tsy-2} val[{tsy-2},{tsy}) test>={tsy}")
    else:
        val0, test0 = _VAL0, _TEST0
    tr = T < val0; va = (T >= val0) & (T < test0); te = T >= test0
    if os.environ.get("HAZARDPULSE_SHUFFLE_LABELS") == "1":
        # NULL TEST: break the input->label link by permuting train+val labels. A real
        # signal must collapse to ~0.5 here; if it doesn't, the model is exploiting a leak.
        Y = Y.copy(); rng = np.random.RandomState(0)
        Y[tr] = rng.permutation(Y[tr]); Y[va] = rng.permutation(Y[va])
        print("  [NULL TEST] train+val labels SHUFFLED -- test AUC must collapse to ~0.5")
    print(f"  train {tr.sum()} ({Y[tr].sum()} pos, {Y[tr].mean():.3%}) | "
          f"val {va.sum()} ({Y[va].sum()} pos) | test {te.sum()} ({Y[te].sum()} pos, {Y[te].mean():.3%})")
    if Y[te].sum() == 0:
        print("  no positive test samples -- adjust label window."); return 2

    # normalize per-channel by train stats
    flat = X[tr][M[tr].astype(bool)]
    mu, sd = flat.mean(0), flat.std(0) + 1e-6
    Xn = ((X - mu) / sd).astype(np.float32)

    class OpModel(nn.Module):
        def __init__(self, d=X.shape[-1], h=96):
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
    best_overall = (None, -1.0, float("nan"))
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
        if best_va > best_overall[1]:
            best_overall = ({k: v.cpu().clone() for k, v in best_state.items()}, best_va, au)
        print(f"  seed {seed}: TEST operational AUC {au:.4f} (best val {best_va:.4f})")
    if args.save_model:
        mp = REPO / args.save_model
        mp.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": best_overall[0], "val_op_auc": best_overall[1],
                    "test_op_auc": best_overall[2],
                    "norm_mu": mu, "norm_sd": sd, "K": args.K, "radius_km": args.input_radius,
                    "n_channels": int(X.shape[-1]), "label_mag": args.label_mag,
                    "label_radius_km": args.label_radius, "label_days": args.label_days,
                    "spec": "hazardpulse/eq-operational-forecaster/v1"}, mp)
        print(f"  saved operational forecaster (val {best_overall[1]:.4f}, "
              f"test {best_overall[2]:.4f}) -> {mp}")
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
