#!/usr/bin/env python3
"""Honest OPERATIONAL forecast skill: can the model pick which active region has the
next M6+ -- evaluated against the model's ACTUAL label (M6+ within 300 km, 365 days
forward), not a contaminated 'rank vs random active cells' proxy.

At each reference time we score a global grid of seismically-active cells with the
deployed model (strictly causal features), then label each by its real forward outcome
(did an M6+ mainshock occur within 300 km in the next 365 days?) and compute the AUC of
positives vs negatives. THIS is the operational question, framed the way the model was
actually trained. Forward outcomes use the catalog, declustered so a doublet/aftershock
(e.g. the 38-second-apart Venezuela M7.5/M7.2) counts once.

    python scripts/backtest_operational_grid.py --ref 2023-06-01 2024-06-01 2024-12-01
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
    lat, lon = np.radians(lat), np.radians(lon)
    lats, lons = np.radians(lats), np.radians(lons)
    dlat, dlon = lats - lat, lons - lon
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lats) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def _auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    P = float((y == 1).sum()); N = float((y == 0).sum())
    if P == 0 or N == 0:
        return float("nan")
    order = np.argsort(-s); ys = y[order]
    tp = fp = a = tpp = fpp = 0.0
    for l in ys:
        if l == 1: tp += 1
        else: fp += 1
        a += (fp / N - fpp / N) * (tp / P + tpp / P) / 2; tpp, fpp = tp, fp
    return a


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", nargs="+", default=["2023-06-01", "2024-06-01", "2024-12-01"])
    ap.add_argument("--grid", type=float, default=2.5, help="cell size (deg)")
    ap.add_argument("--horizon-days", type=int, default=365)
    ap.add_argument("--radius-km", type=float, default=300.0)
    ap.add_argument("--max-cells", type=int, default=180, help="active cells per ref time (cost cap)")
    ap.add_argument("--today", default="2026-06-26")
    args = ap.parse_args(argv)

    from hazardpulse.earthquake.definitive_model import (
        load_usgs_catalog, CatalogArrays, compute_block_s, compute_block_c,
        decluster_gardner_knopoff, ALL_FEATURE_NAMES_ENHANCED)
    import importlib.util
    spec = importlib.util.spec_from_file_location("fse", REPO / "scripts" / "fetch_and_score_earthquake.py")
    fse = importlib.util.module_from_spec(spec); spec.loader.exec_module(fse)
    gbt = fse.load_pretrained_eq_gbt()
    if gbt is None:
        print("No deployed model."); return 1

    today = dt.date.fromisoformat(args.today)
    cat_list = load_usgs_catalog(min_year=2000, max_year=today.year, min_mag=2.5)
    # current year top-up
    import urllib.request, csv, io
    try:
        url = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv"
               f"&starttime={today.year}-01-01&endtime={today.isoformat()}&minmagnitude=2.5")
        with urllib.request.urlopen(url, timeout=90) as r:
            for e in csv.DictReader(io.StringIO(r.read().decode())):
                cat_list.append({"time": e["time"], "latitude": float(e["latitude"]),
                                 "longitude": float(e["longitude"]),
                                 "mag": float(e["mag"] or 0), "depth": float(e["depth"] or 10)})
    except Exception as exc:
        print(f"  (current-year top-up failed: {exc})")
    cat = CatalogArrays(cat_list, verbose=False)
    # declustered M6+ mainshocks for the forward LABEL (count a doublet once)
    mainshocks, _ = decluster_gardner_knopoff(cat_list)
    m6 = np.array([[e["latitude"], e["longitude"],
                    dt.datetime.fromisoformat(e["time"].replace("Z", "+00:00")).timestamp()]
                   for e in mainshocks if e.get("mag", 0) >= 6.0])
    print(f"catalog {len(cat_list)} events, {len(m6)} declustered M6+ mainshocks")

    def score(lat, lon, epoch):
        bs = compute_block_s(lat, lon, epoch, cat)
        if bs is None:
            return None
        bc = compute_block_c(cat_list, lat, lon, epoch)
        vec = np.concatenate([np.asarray(bs), np.nan_to_num(np.asarray(bc), nan=0.0)])
        return float(fse._predict_eq_with_gbt(gbt, vec)) if vec.shape[0] == len(ALL_FEATURE_NAMES_ENHANCED) else None

    g = args.grid
    all_y, all_s = [], []
    for ref in args.ref:
        t = dt.datetime.fromisoformat(ref + "T00:00:00+00:00").timestamp()
        # active cells: where there was recent seismicity (>=15 events in prior 2 yr)
        recent = (cat.times > t - 2 * 365 * SEC_DAY) & (cat.times < t)
        rl, ro = cat.lats[recent], cat.lons[recent]
        cells = {}
        for la, lo in zip(rl, ro):
            cells[(round(la / g) * g, round(lo / g) * g)] = cells.get((round(la / g) * g, round(lo / g) * g), 0) + 1
        active = [c for c, n in cells.items() if n >= 15]
        rng = np.random.RandomState(0)
        if len(active) > args.max_cells:
            active = [active[i] for i in rng.choice(len(active), args.max_cells, replace=False)]
        y, s, npos = [], [], 0
        for (la, lo) in active:
            sc = score(la, lo, t)
            if sc is None:
                continue
            # forward label: declustered M6+ within radius in (t, t+horizon]
            if len(m6):
                d = _hav(la, lo, m6[:, 0], m6[:, 1])
                fwd = (m6[:, 2] > t) & (m6[:, 2] <= t + args.horizon_days * SEC_DAY)
                lab = int(((d < args.radius_km) & fwd).any())
            else:
                lab = 0
            y.append(lab); s.append(sc); npos += lab
        auc = _auc(y, s)
        all_y += y; all_s += s
        print(f"  {ref}: {len(y)} active cells, {npos} positive (M6+ in next {args.horizon_days}d "
              f"within {args.radius_km:.0f}km), operational AUC {auc:.4f}")

    print(f"\nPOOLED operational AUC ({len(all_y)} active-cell forecasts, "
          f"{int(np.sum(all_y))} positive): {_auc(all_y, all_s):.4f}")
    print("This is the honest 'pick the next-year M6+ region among active areas' skill,\n"
          "framed exactly as the model was trained (M6+ within 300km, 365 days forward),\n"
          "with declustered forward labels (a doublet counts once).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
