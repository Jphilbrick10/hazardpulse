#!/usr/bin/env python3
"""Operational skill of the DEEP nowcast: can the GRU pick which active region gets the
next big quake? The honest "helps people" metric -- same framing as
backtest_operational_grid.py (which scored the GBT at pooled AUC 0.509), but scoring the
deep sequence model instead, so the two are directly comparable.

At each reference time we score a global grid of seismically-active cells with the saved
deep model (strictly-causal raw-event sequence), label each cell by its real forward
outcome (declustered M6+ within radius/horizon -- a doublet counts once), and pool the
AUC. Also reports M5+ forward labels (the model's native target) for context.

    python scripts/backtest_operational_deep.py --ref 2023-06-01 2024-06-01 2024-12-01
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
SEC_DAY = 86400.0


def _hav(lat, lon, lats, lons):
    rlat, rlon = np.radians(lat), np.radians(lon)
    rla, rlo = np.radians(lats), np.radians(lons)
    d = (np.sin((rla - rlat) / 2) ** 2
         + np.cos(rlat) * np.cos(rla) * np.sin((rlo - rlon) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(d))


def _hav_az(lat, lon, lats, lons):
    """Distance (km) + azimuth (rad) from (lat,lon) to arrays -- matches deep _haversine."""
    rlat, rlon = np.radians(lat), np.radians(lon)
    rla, rlo = np.radians(lats), np.radians(lons)
    dlon = rlo - rlon
    d = (np.sin((rla - rlat) / 2) ** 2
         + np.cos(rlat) * np.cos(rla) * np.sin(dlon / 2) ** 2)
    dist = 6371.0 * 2 * np.arcsin(np.sqrt(d))
    az = np.arctan2(np.sin(dlon) * np.cos(rla),
                    np.cos(rlat) * np.sin(rla) - np.sin(rlat) * np.cos(rla) * np.cos(dlon))
    return dist, az


def _auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(s); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", nargs="+", default=["2023-06-01", "2024-06-01", "2024-12-01"])
    ap.add_argument("--grid", type=float, default=2.5)
    ap.add_argument("--horizon-days", type=int, default=365)
    ap.add_argument("--radius-km", type=float, default=300.0)
    ap.add_argument("--max-cells", type=int, default=180)
    ap.add_argument("--model", default="results/models/eq_deep_nowcast_m5.0.pt")
    ap.add_argument("--max-year", type=int, default=2025)
    ap.add_argument("--label-mags", type=float, nargs="+", default=[6.0, 5.0],
                    help="forward-label magnitudes to score (short-term product uses 4.5)")
    args = ap.parse_args(argv)

    import torch
    import torch.nn as nn
    from hazardpulse.earthquake.definitive_model import (
        load_usgs_catalog, CatalogArrays, decluster_gardner_knopoff)

    ck = torch.load(REPO / args.model, map_location="cpu", weights_only=False)
    mu, sd = ck["norm_mu"], ck["norm_sd"]
    K = int(ck.get("K", 48)); radius = float(ck.get("radius_km", 500.0))

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

    cat_list = load_usgs_catalog(min_year=2000, max_year=args.max_year, min_mag=2.5)
    cat = CatalogArrays(cat_list, verbose=False)
    mainshocks, _ = decluster_gardner_knopoff(cat_list)

    def _ms_array(min_mag):
        return np.array([[e["latitude"], e["longitude"],
                          dt.datetime.fromisoformat(e["time"].replace("Z", "+00:00")).timestamp()]
                         for e in mainshocks if e.get("mag", 0) >= min_mag])
    tags = [f"M{mg:g}" for mg in args.label_mags]
    ms_by_tag = {f"M{mg:g}": _ms_array(mg) for mg in args.label_mags}
    print(f"catalog {len(cat_list)} events, {len(m6)} M6+ / {len(m5)} M5+ declustered mainshocks")
    print(f"deep model {args.model} (K={K}, radius={radius:.0f}km)")

    def seq_score(lat, lon, ep):
        X = np.zeros((K, 6), np.float32); m = np.zeros(K, np.float32)
        t0 = ep - 5 * 365 * SEC_DAY
        sel = ((cat.times >= t0) & (cat.times < ep)
               & (np.abs(cat.lats - lat) < 6) & (np.abs(cat.lons - lon) < 6))
        idx = np.where(sel)[0]
        if idx.size:
            d, az = _hav_az(lat, lon, cat.lats[idx], cat.lons[idx])
            near = d < radius
            idx, d, az = idx[near], d[near], az[near]
            if idx.size:
                order = np.argsort(cat.times[idx])[-K:]
                idx, d, az = idx[order], d[order], az[order]
                dd = (ep - cat.times[idx]) / SEC_DAY
                seq = np.stack([np.log1p(dd), cat.mags[idx], d / radius,
                                np.clip(cat.depths[idx], 0, 700) / 700.0, np.sin(az), np.cos(az)], axis=1)
                X[K - len(idx):] = seq; m[K - len(idx):] = 1.0
        if m.sum() == 0:
            return None
        Xn = ((X - mu) / sd).astype(np.float32)
        with torch.no_grad():
            return float(torch.sigmoid(net(torch.tensor(Xn[None]), torch.tensor(m[None])))[0])

    g = args.grid
    pools = {tag: ([], []) for tag in tags}
    for ref in args.ref:
        t = dt.datetime.fromisoformat(ref + "T00:00:00+00:00").timestamp()
        recent = (cat.times > t - 2 * 365 * SEC_DAY) & (cat.times < t)
        cells = {}
        for la, lo in zip(cat.lats[recent], cat.lons[recent]):
            key = (round(la / g) * g, round(lo / g) * g)
            cells[key] = cells.get(key, 0) + 1
        active = [c for c, n in cells.items() if n >= 15]
        rng = np.random.RandomState(0)
        if len(active) > args.max_cells:
            active = [active[i] for i in rng.choice(len(active), args.max_cells, replace=False)]
        rowy = {tag: [] for tag in tags}; rows = []
        for (la, lo) in active:
            sc = seq_score(la, lo, t)
            if sc is None:
                continue
            rows.append(sc)
            for tag in tags:
                arr = ms_by_tag[tag]
                if len(arr):
                    d = _hav(la, lo, arr[:, 0], arr[:, 1])
                    fwd = (arr[:, 2] > t) & (arr[:, 2] <= t + args.horizon_days * SEC_DAY)
                    rowy[tag].append(int(((d < args.radius_km) & fwd).any()))
                else:
                    rowy[tag].append(0)
        for tag in tags:
            pools[tag][0].extend(rowy[tag]); pools[tag][1].extend(rows)
            print(f"  {ref} [{tag}+]: {len(rows)} active cells, {int(np.sum(rowy[tag]))} positive, "
                  f"operational AUC {_auc(rowy[tag], rows):.4f}")

    print()
    for tag in tags:
        y, s = pools[tag]
        print(f"POOLED operational AUC [{tag}+ forward, {args.horizon_days}d/{args.radius_km:.0f}km] "
              f"({len(y)} forecasts, {int(np.sum(y))} positive): {_auc(y, s):.4f}")
    print("\nThe honest 'which active cell sees the next event in-window' operational skill, "
          "distinct from the case-control NOWCAST AUC.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
