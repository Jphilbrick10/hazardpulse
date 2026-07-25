#!/usr/bin/env python3
"""Bulk-grade CONTROLLED multi-event dv/v with the VALIDATED cross-correlation method.

Survives the data-access wall that blocked the session-grade version:
  - dvv_lib now THROTTLES per node + RETRIES transient 429/5xx/timeout (no more rate-limit holes)
  - data-driven event selection: PROBE 2-3 sample days per candidate station (node-agnostic),
    keep only stations that actually return data, require >=2 within pair distance
  - resumable: disk-cached fetch + partial JSON written per event

Per qualifying M6.0+ event: network-average dv/v (corr-weighted across station pairs),
reference = baseline-year CCF stack.
  PRE-quake anomaly  : mean dv/v [q-90d, q-7d]  vs baseline scatter        (sigma, foreshock-safe)
  POSITIVE control   : co-seismic [q, q+45d]    vs baseline (must DROP -> pipeline fires)
  NEGATIVE control   : same windows 3y earlier (quiet: no M6+ within 100km/120d)
Paired across events: is PRE-quake systematically below its NEGATIVE control? (Wilcoxon + sign).

Usage: python scripts/research/dvv_multi_event.py [--max-events 30] [--min-mag 6.0]
"""
import sys, os, json, argparse, itertools, datetime as dt
import concurrent.futures as cf
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import numpy as np
from dvv_lib import (series_day, find_stations, haversine, cross_correlate, dvv_stretch, CACHE)
try:
    from scipy.stats import wilcoxon
except Exception:
    wilcoxon = None
from hazardpulse.data.earthquake import load_usgs_catalog

FS, FMIN, FMAX, MAXLAG = 20.0, 0.5, 2.0, 40.0
MAXR_DEG = 0.8            # event<->station ~88 km
PAIR_MAX_KM = 100.0
Y = 365
# networks reachable at the open FDSN nodes dvv_lib routes to (probe still decides per station)
ROUTABLE = {"CI","AZ","CE","NP","SB","BK","NC","US","IU","II","IM","N4","GT","AK","AV","AT","AV",
            "GE","NZ","IV","MN","HL","HT","HA","HP","HC","C","C1","CX","G","PB","UW","UO","CC"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                   "results", "calibration", "dvv_multi_event_results.json")


def parse_iso(s):
    return dt.datetime.fromisoformat(s.replace("Z", "").split(".")[0])


def daterange(t0, t1, step):
    out = []; d = t0
    while d < t1:
        out.append(d.date().isoformat()); d += dt.timedelta(days=step)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-mag", type=float, default=6.0)
    ap.add_argument("--max-events", type=int, default=30)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--y0", type=int, default=2013)
    ap.add_argument("--y1", type=int, default=2022)
    args = ap.parse_args()

    print("loading catalog ...", flush=True)
    cat = load_usgs_catalog(2007, 2024, 6.0)
    rec = []
    for e in cat:
        try:
            rec.append((parse_iso(e["time"]), float(e["latitude"]), float(e["longitude"]), float(e["mag"])))
        except Exception:
            continue
    rec.sort()
    ct = np.array([r[0].timestamp() for r in rec]); cla = np.array([r[1] for r in rec])
    clo = np.array([r[2] for r in rec]); cmg = np.array([r[3] for r in rec])
    cands = [(r[0], r[1], r[2], r[3]) for r in rec if r[3] >= args.min_mag
             and dt.datetime(args.y0, 1, 1) <= r[0] <= dt.datetime(args.y1, 6, 1)]
    print(f"  {len(cands)} M{args.min_mag}+ candidates {args.y0}..{args.y1}", flush=True)

    def quiet(lat, lon, tc, days=120, radius=100):
        lo = tc.timestamp() - days * 86400; hi = tc.timestamp() + days * 86400
        m = (ct >= lo) & (ct <= hi)
        return (not m.any()) or (not (haversine(lat, lon, cla[m], clo[m]) < radius).any())

    def probe_station(s, qt):
        """True if the station returns data on >=2 of 3 probe days spanning the windows."""
        probe_days = [(qt - dt.timedelta(days=int(1.5 * Y))).date().isoformat(),
                      (qt - dt.timedelta(days=45)).date().isoformat(),
                      (qt - dt.timedelta(days=int(3 * Y))).date().isoformat()]
        ok = sum(series_day(s["net"], s["sta"], s["cha"], d, 4, FMIN, FMAX) is not None
                 for d in probe_days)
        return ok >= 2

    def select_stations(c):
        qt, lat, lon, mag = c
        ns = (qt - dt.timedelta(days=int(1.5 * Y))).date().isoformat()
        ne = (qt + dt.timedelta(days=60)).date().isoformat()
        stas = find_stations(lat, lon, MAXR_DEG, ns, ne, "HHZ,BHZ")
        picked, seen = [], set()
        for s in stas:
            if s["sta"] in seen or s["net"] not in ROUTABLE:
                continue
            sstart = (s.get("start") or "")[:10]; send = (s.get("end") or "")[:10]
            if sstart and sstart <= ns and (not send or send >= qt.date().isoformat()):
                seen.add(s["sta"]); picked.append(s)
            if len(picked) >= 5:
                break
        return picked

    def fetch_series(stas, days):
        jobs = [(s["net"], s["sta"], s["cha"], d) for s in stas for d in days]
        out = {(s["net"], s["sta"], s["cha"]): {} for s in stas}
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(series_day, n, st, ch, d, 4, FMIN, FMAX): (n, st, ch, d) for (n, st, ch, d) in jobs}
            for fu in cf.as_completed(futs):
                n, st, ch, d = futs[fu]; a = fu.result()
                if a is not None:
                    out[(n, st, ch)][d] = a
        return out

    def network_anom(stas, qt):
        # 1-year baseline far from the quake, 6-day step (~61 days -> stable stacked reference,
        # ~3x fewer fetches than 4-day). pre/co stay dense (3-day).
        base_d = daterange(qt - dt.timedelta(days=int(1.5 * Y)), qt - dt.timedelta(days=int(0.5 * Y)), 6)
        pre_d = daterange(qt - dt.timedelta(days=90), qt, 3)
        co_d = daterange(qt, qt + dt.timedelta(days=45), 3)
        ser = fetch_series(stas, base_d + pre_d + co_d)
        keys = [k for k in ser if len(ser[k]) >= 20]
        if len(keys) < 2:
            return None, f"stations>=20d: {[(k[1], len(ser[k])) for k in ser]}"
        coord = {(s["net"], s["sta"], s["cha"]): (s["lat"], s["lon"]) for s in stas}
        base_set = set(base_d)
        pair_dvv = {}
        for a, b in itertools.combinations(keys, 2):
            if haversine(*coord[a], *coord[b]) > PAIR_MAX_KM:
                continue
            common = sorted(set(ser[a]) & set(ser[b]))
            ccf = {d: cross_correlate(ser[a][d], ser[b][d], FS, MAXLAG, FMIN, FMAX) for d in common}
            bc = [ccf[d] for d in common if d in base_set]
            if len(bc) < 20:
                continue
            ref = np.mean(bc, axis=0)
            for d in ccf:
                di = dt.date.fromisoformat(d)
                mem = [ccf[x] for x in ccf if abs((dt.date.fromisoformat(x) - di).days) <= 12]
                if len(mem) < 3:
                    continue
                v, cc = dvv_stretch(np.mean(mem, axis=0), ref, FS, MAXLAG)
                if cc > 0.55:
                    pair_dvv.setdefault(d, []).append((v, cc))
        if not pair_dvv:
            return None, "no pair survived (baseline CCF<20 or corr<.55)"
        netv = {}
        for d, vs in pair_dvv.items():
            w = np.array([c for v, c in vs]); val = np.array([v for v, c in vs])
            netv[d] = float(np.average(val, weights=w))
        D = lambda s: dt.date.fromisoformat(s)
        base = np.array([netv[d] for d in netv if d in base_set])
        pre = np.array([netv[d] for d in netv if (qt - dt.timedelta(days=90)).date() <= D(d) < (qt - dt.timedelta(days=7)).date()])
        co = np.array([netv[d] for d in netv if qt.date() <= D(d) < (qt + dt.timedelta(days=45)).date()])
        if len(base) < 8 or len(pre) < 2 or len(co) < 2:
            return None, f"counts base={len(base)} pre={len(pre)} co={len(co)}"
        bsd = max(base.std(), 0.02); bm = base.mean()
        clip = lambda v: float(np.clip((v - bm) / bsd, -15, 15))
        return dict(pre_sigma=clip(pre.mean()), co_sigma=clip(co.mean()),
                    pre_n=int(len(pre)), co_n=int(len(co)), base_n=int(len(base)),
                    base_sd=round(float(base.std()), 4), npairs=len({k for k in pair_dvv})), "ok"

    # ---- select events by probing ----
    print("selecting events (find stations + probe for real data) ...", flush=True)
    selected = []
    for c in cands:
        qt, lat, lon, mag = c
        stas = select_stations(c)
        if len(stas) < 2:
            continue
        good = [s for s in stas if probe_station(s, qt)]
        # require >=2 good stations within pair distance
        pair_ok = any(haversine(good[i]["lat"], good[i]["lon"], good[j]["lat"], good[j]["lon"]) <= PAIR_MAX_KM
                      for i in range(len(good)) for j in range(i + 1, len(good)))
        if len(good) >= 2 and pair_ok:
            selected.append((c, good))
            print(f"  + M{mag:.1f} {qt.date()} {len(good)} stations "
                  f"({','.join(s['net']+'.'+s['sta'] for s in good[:4])})", flush=True)
        if len(selected) >= args.max_events:
            break
    print(f"\n{len(selected)} events with verified continuous station pairs", flush=True)

    # ---- analyze ----
    results = []
    for i, (c, stas) in enumerate(selected):
        qt, lat, lon, mag = c
        print(f"\n[{i+1}/{len(selected)}] M{mag:.1f} {qt.date()} ({len(stas)} sta)", flush=True)
        real, why = network_anom(stas, qt)
        if real is None:
            print(f"   real skip: {why}", flush=True); continue
        cqt = qt - dt.timedelta(days=3 * Y); ctrl = None
        if quiet(lat, lon, cqt):
            ctrl, _ = network_anom(stas, cqt)
        results.append(dict(mag=mag, date=qt.date().isoformat(), nsta=len(stas),
                            real=real, ctrl=ctrl))
        print(f"   pre {real['pre_sigma']:+.2f}sig | co-seismic {real['co_sigma']:+.2f}sig "
              f"(base n={real['base_n']}, {real['npairs']} pairs)"
              + (f" | ctrl pre {ctrl['pre_sigma']:+.2f}sig" if ctrl else ""), flush=True)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)

    # ---- aggregate ----
    print("\n" + "=" * 68, flush=True)
    if not results:
        print("no events scored -- data-access wall persists at this magnitude/region cut"); return
    pre = np.array([r["real"]["pre_sigma"] for r in results])
    co = np.array([r["real"]["co_sigma"] for r in results])
    print(f"N events scored: {len(results)}", flush=True)
    print(f"POSITIVE CONTROL (co-seismic, must DROP): median {np.median(co):+.2f}sig "
          f"({(co < -1).mean()*100:.0f}% < -1sig) -> "
          f"{'method FIRES' if np.median(co) < -0.5 else 'WEAK/absent - interpret with care'}", flush=True)
    print(f"MAIN pre-quake: median {np.median(pre):+.2f}sig ({(pre < -1.5).mean()*100:.0f}% < -1.5sig)", flush=True)
    cr = [r for r in results if r["ctrl"]]
    if cr:
        rp = np.array([r["real"]["pre_sigma"] for r in cr]); cp = np.array([r["ctrl"]["pre_sigma"] for r in cr])
        print(f"paired vs control (n={len(cr)}): real median {np.median(rp):+.2f}sig vs "
              f"control {np.median(cp):+.2f}sig; real<control in {(rp < cp).mean()*100:.0f}%", flush=True)
        if wilcoxon and len(cr) >= 6:
            try:
                _, p = wilcoxon(rp, cp, alternative="less")
                print(f"Wilcoxon (real<control): p={p:.3f} -> "
                      f"{'SIGNAL' if p < 0.05 else 'NULL (no pre-quake dv/v beyond chance)'}", flush=True)
            except Exception as ex:
                print(f"wilcoxon: {ex}", flush=True)
    print(f"\nsaved -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
