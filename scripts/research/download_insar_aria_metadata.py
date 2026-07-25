#!/usr/bin/env python3
"""Download ARIA Sentinel-1 GUNW metadata for operational InSAR coverage features.

This intentionally caches metadata, not displacement rasters. The full ARIA GUNW products are
large and Earthdata/ASF-access dependent; the metadata still gives a causal coverage signal and a
stable path for later displacement-summary ingestion.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_CSV = REPO / ".cache" / "earthquake" / "insar" / "aria_gunw_metadata_v1.csv"
CMR_GRANULES = "https://cmr.earthdata.nasa.gov/search/granules.json"
USER_AGENT = "hazardpulse-insar-aria-metadata/0.1"

REGIONS = {
    "california": (-125.0, 32.0, -114.0, 42.5),
    "cascadia": (-129.5, 40.0, -120.0, 52.5),
    "alaska": (-170.0, 50.0, -130.0, 72.0),
    "hawaii": (-161.5, 18.0, -154.0, 23.0),
    "puerto_rico": (-68.5, 16.0, -62.0, 20.5),
    "japan_kuril": (128.0, 29.0, 150.0, 47.0),
    "chile": (-76.0, -46.0, -66.0, -17.0),
    "new_zealand": (165.0, -48.0, 180.0, -33.0),
}


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def _polygon_centroid(entry: dict) -> tuple[float, float] | None:
    points = []
    for poly in entry.get("polygons") or []:
        for text in poly:
            parts = text.replace(",", " ").split()
            vals = []
            for item in parts:
                try:
                    vals.append(float(item))
                except ValueError:
                    pass
            for i in range(0, len(vals) - 1, 2):
                # CMR polygon strings are "lat lon lat lon ..."
                la, lo = vals[i], vals[i + 1]
                if math.isfinite(la) and math.isfinite(lo):
                    points.append((la, ((lo + 180.0) % 360.0) - 180.0))
    if not points:
        boxes = entry.get("boxes") or []
        for text in boxes:
            vals = [float(x) for x in text.replace(",", " ").split()]
            if len(vals) >= 4:
                south, west, north, east = vals[:4]
                points.append(((south + north) / 2.0, ((west + east) / 2.0 + 180.0) % 360.0 - 180.0))
    if not points:
        return None
    lat = float(sum(p[0] for p in points) / len(points))
    # Averaging lon directly is acceptable for these bounded regional boxes.
    lon = float(sum(p[1] for p in points) / len(points))
    return lat, lon


def _query_region_year(
    region: str,
    bbox: tuple[float, float, float, float],
    year: int,
    page_size: int,
    max_pages: int,
    sleep_s: float,
):
    start = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc).isoformat().replace("+00:00", "Z")
    end = dt.datetime(year + 1, 1, 1, tzinfo=dt.timezone.utc).isoformat().replace("+00:00", "Z")
    rows = []
    for page in range(1, max_pages + 1):
        params = {
            "short_name": "ARIA_S1_GUNW",
            "temporal": f"{start},{end}",
            "bounding_box": ",".join(f"{x:g}" for x in bbox),
            "page_size": str(page_size),
            "page_num": str(page),
            "sort_key[]": "start_date",
        }
        url = CMR_GRANULES + "?" + urllib.parse.urlencode(params, doseq=True)
        with urllib.request.urlopen(_request(url), timeout=90) as resp:
            payload = json.load(resp)
        entries = payload.get("feed", {}).get("entry", [])
        if not entries:
            break
        for entry in entries:
            centroid = _polygon_centroid(entry)
            if centroid is None:
                continue
            rows.append({
                "region": region,
                "granule_id": entry.get("id", ""),
                "producer_granule_id": entry.get("producer_granule_id", ""),
                "title": entry.get("title", ""),
                "time_start": entry.get("time_start", ""),
                "time_end": entry.get("time_end", ""),
                "updated": entry.get("updated", ""),
                "centroid_lat": f"{centroid[0]:.6f}",
                "centroid_lon": f"{centroid[1]:.6f}",
                "granule_size_gb": entry.get("granule_size", "0") or "0",
            })
        if len(entries) < page_size:
            break
        time.sleep(sleep_s)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start-year", type=int, default=2017)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--regions", default="california,cascadia,alaska,hawaii,puerto_rico")
    ap.add_argument("--page-size", type=int, default=2000)
    ap.add_argument("--max-pages-per-query", type=int, default=10)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    selected = [x.strip() for x in args.regions.split(",") if x.strip()]
    existing = {}
    if OUT_CSV.exists() and not args.force:
        with OUT_CSV.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                existing[row.get("granule_id") or row.get("producer_granule_id")] = row

    rows = dict(existing)
    for region in selected:
        if region not in REGIONS:
            raise SystemExit(f"unknown region {region!r}; choices={','.join(REGIONS)}")
        for year in range(args.start_year, args.end_year + 1):
            try:
                got = _query_region_year(
                    region,
                    REGIONS[region],
                    year,
                    args.page_size,
                    args.max_pages_per_query,
                    args.sleep,
                )
            except Exception as exc:
                print(f"miss {region} {year}: {exc!r}", flush=True)
                continue
            for row in got:
                key = row.get("granule_id") or row.get("producer_granule_id")
                rows[key] = row
            print(f"ok {region} {year}: {len(got)}", flush=True)
            time.sleep(args.sleep)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "region",
        "granule_id",
        "producer_granule_id",
        "title",
        "time_start",
        "time_end",
        "updated",
        "centroid_lat",
        "centroid_lon",
        "granule_size_gb",
    ]
    ordered = sorted(rows.values(), key=lambda r: (r.get("time_start", ""), r.get("granule_id", "")))
    with OUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ordered)
    print(f"wrote {OUT_CSV} rows={len(ordered)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
