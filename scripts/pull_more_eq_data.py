#!/usr/bin/env python3
"""Pull a FULLER USGS catalog: global M2.0+ (vs the M2.5+ we had), monthly chunks to
beat the 20k/query cap. More small events = more foreshocks = better precursor signal,
especially for the 'silent' big quakes. Polite rate-limiting + resumable (skips done months).
"""
import csv, io, sys, time, urllib.request, urllib.error
from pathlib import Path
import datetime as dt

OUT = Path(".cache/earthquake/usgs_full"); OUT.mkdir(parents=True, exist_ok=True)
API = "https://earthquake.usgs.gov/fdsnws/event/1/query"
UA = "hazardpulse/0.1 (research; +https://github.com/Jphilbrick10/hazardpulse)"
MINMAG = 2.0

def fetch_month(y, mo):
    start = dt.date(y, mo, 1)
    end = dt.date(y + (mo == 12), (mo % 12) + 1, 1)
    url = (f"{API}?format=csv&starttime={start}&endtime={end}"
           f"&minmagnitude={MINMAG}&orderby=time-asc&limit=20000")
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                return r.read().decode("utf-8", errors="replace")
        except Exception as exc:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))
    return ""

def main():
    y0, y1 = 2000, 2026
    total = 0
    for y in range(y0, y1 + 1):
        rows = []
        hdr = None
        for mo in range(1, 13):
            if y == 2026 and mo > 6:  # stop at present
                break
            t0 = time.time()
            try:
                txt = fetch_month(y, mo)
            except Exception as exc:
                print(f"  {y}-{mo:02d} FAILED: {exc}", flush=True); continue
            lines = txt.splitlines()
            if not lines:
                continue
            hdr = lines[0]
            n = len(lines) - 1
            rows.extend(lines[1:])
            if n >= 19999:
                print(f"  WARN {y}-{mo:02d} hit 20k cap -- need finer chunking", flush=True)
            print(f"  {y}-{mo:02d}: {n:6d} events ({time.time()-t0:.1f}s)", flush=True)
            time.sleep(1.0)  # polite
        if hdr and rows:
            dest = OUT / f"usgs_M{MINMAG}_{y}.csv"
            dest.write_text(hdr + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
            total += len(rows)
            print(f"  [{y}] wrote {len(rows)} events -> {dest.name}  (running total {total})", flush=True)
    print(f"DONE. {total} M{MINMAG}+ events pulled to {OUT}")

if __name__ == "__main__":
    raise SystemExit(main())
