#!/usr/bin/env python3
"""Operational backtest: would the deployed nowcast have flagged the RECENT M6+ quakes?

For each recent M6+ event we score its location AT ITS TIME using only data available
BEFORE it (the features are strictly causal), with the deployed model. Honesty built in:

  * epicenter score -- the model's P(mainshock-setting) at the actual location/time
  * RANK among all globally-active cells at that moment -- the false-alarm context
    (catching the epicenter only matters if the model didn't also light up everywhere)
  * spatial localization -- where the regional probability peaks vs the true epicenter
  * a same-location control score (a random earlier time) -- did the event-time setting
    actually look more critical than a quiet time at the same place?

Prints a per-event table + an honest aggregate. Nothing here is cherry-picked: every
recent M6+ is scored, hits and misses alike.

    python scripts/backtest_recent_earthquakes.py --days 30 --min-mag 6.0
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
import urllib.request
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

_FDSN = ("https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv"
         "&starttime={start}&endtime={end}&minmagnitude={mag}")


def _fetch(start, end, mag):
    url = _FDSN.format(start=start, end=end, mag=mag)
    with urllib.request.urlopen(url, timeout=90) as r:
        return list(csv.DictReader(io.StringIO(r.read().decode())))


def _haversine(lat, lon, lats, lons):
    lat, lon = np.radians(lat), np.radians(lon)
    lats, lons = np.radians(lats), np.radians(lons)
    dlat, dlon = lats - lat, lons - lon
    a = np.sin(dlat / 2) ** 2 + np.cos(lat) * np.cos(lats) * np.sin(dlon / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--min-mag", type=float, default=6.0)
    ap.add_argument("--today", default="2026-06-26", help="reference 'now' (YYYY-MM-DD)")
    ap.add_argument("--n-active", type=int, default=20, help="globally-active cells sampled for the rank baseline")
    ap.add_argument("--spatial", action="store_true", help="also localize (slow: +25 scores/event)")
    args = ap.parse_args(argv)

    import datetime as dt
    from hazardpulse.earthquake.definitive_model import (
        load_usgs_catalog, CatalogArrays, compute_block_s, compute_block_c,
        ALL_FEATURE_NAMES_ENHANCED)
    import importlib.util
    spec = importlib.util.spec_from_file_location("fse", REPO / "scripts" / "fetch_and_score_earthquake.py")
    fse = importlib.util.module_from_spec(spec); spec.loader.exec_module(fse)
    gbt = fse.load_pretrained_eq_gbt()
    if gbt is None:
        print("No deployed earthquake model found."); return 1

    today = dt.date.fromisoformat(args.today)
    start = (today - dt.timedelta(days=args.days)).isoformat()
    print(f"Fetching recent M>={args.min_mag} events {start}..{today}...")
    events = _fetch(start, today.isoformat(), args.min_mag)
    events = [e for e in events if e.get("type") == "earthquake"]
    print(f"  {len(events)} recent M>={args.min_mag} events")

    # Build the seismicity catalog (history for features). Cached 2000-2025 + fetch the
    # current year so the lead-up to each recent event is present.
    print("Loading catalog history (cached + current year)...")
    cat_list = load_usgs_catalog(min_year=2000, max_year=today.year, min_mag=2.5)
    cur = _fetch(f"{today.year}-01-01", today.isoformat(), 2.5)
    for e in cur:
        cat_list.append({"time": e["time"], "latitude": float(e["latitude"]),
                         "longitude": float(e["longitude"]), "mag": float(e["mag"] or 0),
                         "depth": float(e["depth"] or 10)})
    cat = CatalogArrays(cat_list, verbose=False)
    print(f"  catalog events: {len(cat_list)}")

    def score(lat, lon, epoch):
        bs = compute_block_s(lat, lon, epoch, cat)
        if bs is None:
            return None
        bc = compute_block_c(cat_list, lat, lon, epoch)
        vec = np.concatenate([np.asarray(bs), np.nan_to_num(np.asarray(bc), nan=0.0)])
        if vec.shape[0] != len(ALL_FEATURE_NAMES_ENHANCED):
            return None
        return float(fse._predict_eq_with_gbt(gbt, vec))

    # globally-active cell sample for the rank baseline: random recent-seismicity sites
    rng = np.random.RandomState(0)
    recent_mask = cat.times > (dt.datetime(today.year - 1, today.month, today.day,
                                           tzinfo=dt.timezone.utc).timestamp())
    act_idx = np.where(recent_mask & (cat.mags >= 4.0))[0]

    rows = []
    for e in events:
        lat, lon = float(e["latitude"]), float(e["longitude"])
        epoch = dt.datetime.fromisoformat(e["time"].replace("Z", "+00:00")).timestamp()
        p = score(lat, lon, epoch)
        if p is None:
            rows.append((e, None, None, None, None, None)); continue
        # rank vs globally-active cells scored at the SAME time (false-alarm context)
        samp = rng.choice(act_idx, min(args.n_active, len(act_idx)), replace=False)
        active_scores = []
        for j in samp:
            ps = score(cat.lats[j], cat.lons[j], epoch)
            if ps is not None:
                active_scores.append(ps)
        active_scores = np.array(active_scores)
        pct = float((active_scores < p).mean() * 100) if active_scores.size else float("nan")
        # same-location control at a random earlier time
        ctrl_epoch = epoch - rng.uniform(1.0, 3.0) * 365.25 * 86400.0
        pc = score(lat, lon, ctrl_epoch)
        # spatial localization (optional): best-scoring cell within +/-6 deg
        best_d = None
        if args.spatial:
            gl, go, gp = [], [], []
            for dla in range(-6, 7, 3):
                for dlo in range(-6, 7, 3):
                    ps = score(lat + dla, lon + dlo, epoch)
                    if ps is not None:
                        gl.append(lat + dla); go.append(lon + dlo); gp.append(ps)
            if gp:
                k = int(np.argmax(gp))
                best_d = float(_haversine(lat, lon, np.array([gl[k]]), np.array([go[k]]))[0])
        rows.append((e, p, pct, pc, best_d, len(active_scores)))

    # --- report ---------------------------------------------------------------- #
    print(f"\n{'place':38s} {'M':>4s} {'P(epi)':>7s} {'rank%':>6s} {'ctrl':>6s} {'peak_km':>8s}")
    hits = pcts = total = 0
    beat_ctrl = ctrl_n = 0
    for (e, p, pct, pc, bd, na) in rows:
        place = (e.get("place", "")[:36])
        m = float(e["mag"] or 0)
        if p is None:
            print(f"{place:38s} {m:4.1f}   (insufficient local seismicity to score)")
            continue
        total += 1
        pcts += pct if pct == pct else 0
        if pct >= 80:
            hits += 1
        if pc is not None:
            ctrl_n += 1
            if p > pc:
                beat_ctrl += 1
        cs = f"{pc:.3f}" if pc is not None else "  -  "
        ds = f"{bd:.0f}" if bd is not None else "  -  "
        print(f"{place:38s} {m:4.1f} {p:7.3f} {pct:5.0f}% {cs:>6s} {ds:>8s}")

    print(f"\nScored {total}/{len(rows)} events (others lacked enough local seismicity).")
    if total:
        print(f"  in the top 20% of globally-active cells at event time: {hits}/{total} "
              f"({100*hits/total:.0f}%)")
        print(f"  mean rank-percentile of the true epicenter: {pcts/total:.0f}th")
    if ctrl_n:
        print(f"  event-time score > same-location quiet-time control: {beat_ctrl}/{ctrl_n} "
              f"({100*beat_ctrl/ctrl_n:.0f}%)")
    print("\nHONEST READ: a high rank-percentile means the model prioritized the true location\n"
          "over other active areas; ~50th would mean no localization. 'peak_km' is how far the\n"
          "regional probability peak sat from the true epicenter (grid resolution ~220 km).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
