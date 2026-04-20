#!/usr/bin/env python3
"""Build historical RI training dataset from IBTrACS best-track data.

Downloads IBTrACS CSV for multiple basins, extracts 6-hourly observations,
labels each with RI (rapid intensification = 30+ kt in 24h), and writes
a JSONL file that the operational scoring pipeline uses for training.

Run once (or periodically to update with latest seasons):
  python scripts/build_hurricane_training_data.py
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import ssl
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / ".cache" / "ibtracs"
OUTPUT_PATH = PROJECT_ROOT / "results" / "hurricane_operational_ri_2000_2024_al_sst.jsonl"

BASINS = {
    "NA": "Atlantic",
    "EP": "East Pacific",
    "WP": "West Pacific",
    "NI": "North Indian",
    "SI": "South Indian",
    "SP": "South Pacific",
}

IBTRACS_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs"
    "/v04r01/access/csv/ibtracs.{basin}.list.v04r01.csv"
)

RI_THRESHOLD_KT = 30
RI_WINDOW_STEPS = 4  # 4 x 6h = 24h
MIN_YEAR = 2000
MAX_YEAR = 2024


def fetch_cached(url: str) -> bytes | None:
    digest = hashlib.md5(url.encode()).hexdigest()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{digest}.csv"
    if cache_path.exists():
        print(f"    Using cached: {cache_path.name}")
        return cache_path.read_bytes()
    try:
        ctx = ssl.create_default_context()
        req = urllib.request.Request(url, headers={"User-Agent": "HazardPulse/1.0"})
        with urllib.request.urlopen(req, timeout=300, context=ctx) as resp:
            data = resp.read()
        cache_path.write_bytes(data)
        return data
    except Exception as e:
        print(f"    Download failed: {e}")
        return None


def parse_ibtracs(text: str) -> dict[str, dict]:
    """Parse IBTrACS CSV into storms keyed by SID."""
    reader = csv.reader(io.StringIO(text))
    header = next(reader)
    col = {h.strip().strip('"'): i for i, h in enumerate(header)}
    next(reader, None)  # skip units row

    storms: dict[str, dict] = {}
    for row in reader:
        if len(row) < 10:
            continue
        try:
            sid = row[col.get("SID", 0)].strip().strip('"')
            name = row[col.get("NAME", 1)].strip().strip('"')

            def sf(key):
                if key not in col:
                    return -999.0
                v = row[col[key]].strip().strip('"')
                if v in ("", " ", "NA", "na"):
                    return -999.0
                return float(v)

            wind = sf("USA_WIND")
            if wind <= 0:
                wind = sf("WMO_WIND")
            pres = sf("USA_PRES")
            if pres <= 0:
                pres = sf("WMO_PRES")

            time_str = row[col.get("ISO_TIME", 6)].strip().strip('"')
            year = int(time_str[:4]) if len(time_str) >= 4 else 0
            month = int(time_str[5:7]) if len(time_str) >= 7 else 0

            basin_str = ""
            if "BASIN" in col:
                basin_str = row[col["BASIN"]].strip().strip('"')

            if sid not in storms:
                storms[sid] = {"name": name, "basin": basin_str, "entries": []}

            storms[sid]["entries"].append({
                "wind": wind, "pres": pres,
                "lat": sf("LAT"), "lon": sf("LON"),
                "year": year, "month": month,
                "time_str": time_str,
            })
        except Exception:
            continue
    return storms


def extract_ri_cases(storms: dict[str, dict]) -> list[dict]:
    """Extract RI training cases from parsed IBTrACS storms."""
    cases: list[dict] = []

    for sid, sdata in storms.items():
        entries = sdata["entries"]
        n = len(entries)
        if n < RI_WINDOW_STEPS + 6:
            continue

        for i in range(4, n - RI_WINDOW_STEPS):
            e = entries[i]
            w_now = e["wind"]
            if w_now <= 0 or e["year"] < MIN_YEAR or e["year"] > MAX_YEAR:
                continue

            w_future = entries[i + RI_WINDOW_STEPS]["wind"]
            if w_future <= 0:
                continue

            dv_24h = w_future - w_now
            ri_label = 1 if dv_24h >= RI_THRESHOLD_KT else 0

            # Build feature dict with available best-track data
            lat = e["lat"]
            lon = e["lon"]
            if lat == -999 or lon == -999:
                continue

            case: dict = {
                "storm_id": sid,
                "season_year": e["year"],
                "issue_time": e["time_str"],
                "basin": sdata["basin"],
                "storm_name": sdata["name"],
                "analysis_model": "BEST",
                "analysis_lat": lat,
                "analysis_lon": lon,
                "analysis_vmax_kt": w_now,
                "analysis_mslp_hpa": e["pres"] if e["pres"] > 800 else None,
                "ri_label_30kt": ri_label,
            }

            # Wind change features
            for steps, suffix in [(1, "6h"), (2, "12h"), (4, "24h")]:
                if i >= steps and entries[i - steps]["wind"] > 0:
                    case[f"analysis_dv_{suffix}"] = w_now - entries[i - steps]["wind"]
                else:
                    case[f"analysis_dv_{suffix}"] = None

            # Pressure changes
            for steps, suffix in [(1, "6h"), (2, "12h"), (4, "24h")]:
                p_now = e["pres"]
                p_prev = entries[i - steps]["pres"] if i >= steps else -999
                if p_now > 800 and p_prev > 800:
                    case[f"analysis_dp_{suffix}"] = p_now - p_prev
                else:
                    case[f"analysis_dp_{suffix}"] = None

            # Location features
            case["abs_lat"] = abs(lat)
            case["issue_month_sin"] = float(np.sin(2 * np.pi * e["month"] / 12.0))
            case["issue_month_cos"] = float(np.cos(2 * np.pi * e["month"] / 12.0))

            # MPI estimate
            abs_lat = abs(lat)
            sst_est = 30.0 - 0.5 * max(0, abs_lat - 10)
            if lat >= 0:
                sst_est += 2.0 * np.exp(-((e["month"] - 9) ** 2) / 8.0)
            else:
                sst_est += 2.0 * np.exp(-((e["month"] - 3) ** 2) / 8.0)
            mpi = min(30.0 * max(sst_est - 26.0, 0) + 40.0, 185.0)
            case["mpi_deficit"] = mpi - w_now
            case["intensity_frac_mpi"] = w_now / mpi if mpi > 0 else None

            # Translation speed
            if i >= 1 and entries[i - 1]["lat"] != -999:
                dlat = math.radians(lat - entries[i - 1]["lat"])
                dlon = math.radians(lon - entries[i - 1]["lon"])
                a = (math.sin(dlat / 2) ** 2
                     + math.cos(math.radians(entries[i - 1]["lat"]))
                     * math.cos(math.radians(lat))
                     * math.sin(dlon / 2) ** 2)
                dist_km = 6371.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
                case["translation_speed_kmh"] = dist_km / 6.0
            else:
                case["translation_speed_kmh"] = None

            # Storm age
            case["storm_age_h"] = i * 6.0

            cases.append(case)

    return cases


def main() -> int:
    print("Building hurricane RI training dataset from IBTrACS")
    print(f"  Years: {MIN_YEAR}-{MAX_YEAR}")
    print(f"  RI threshold: {RI_THRESHOLD_KT} kt / 24h")
    print()

    all_cases: list[dict] = []

    for basin_code, basin_name in BASINS.items():
        print(f"  Loading IBTrACS {basin_name} ({basin_code})...")
        t0 = time.time()
        url = IBTRACS_URL.format(basin=basin_code)
        data = fetch_cached(url)
        if not data:
            continue
        text = data.decode("utf-8", errors="replace")
        storms = parse_ibtracs(text)
        cases = extract_ri_cases(storms)
        elapsed = time.time() - t0
        n_pos = sum(1 for c in cases if c["ri_label_30kt"] == 1)
        print(f"    {len(storms)} storms, {len(cases)} cases ({n_pos} RI+) in {elapsed:.1f}s")
        all_cases.extend(cases)

    n_pos = sum(1 for c in all_cases if c["ri_label_30kt"] == 1)
    n_neg = len(all_cases) - n_pos
    print()
    print(f"  Total: {len(all_cases)} cases ({n_pos} RI+, {n_neg} RI-, rate={n_pos / max(len(all_cases), 1):.1%})")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        for case in all_cases:
            fh.write(json.dumps(case) + "\n")

    print(f"  Wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
