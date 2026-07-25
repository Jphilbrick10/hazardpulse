#!/usr/bin/env python3
"""Download high-effort external data for the operational earthquake ranker.

The goal is to make the "new data, not another classifier" path reproducible:

* CRESCENT/Zenodo Cascadia tremor and GNSS benchmark files.
* EarthScope station inventory for waveform/noise-availability proxy features.
* Regional low-magnitude USGS/ComCat catalogs for microseismicity below the global M2.5 floor.

Downloads are resumable where possible and all files land in ``.cache/earthquake``.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EQ_CACHE = REPO / ".cache" / "earthquake"
ZENODO_RECORD = "https://zenodo.org/api/records/20276956"
USER_AGENT = "hazardpulse-nextwave-earthquake/0.1"

CRESCENT_KEYS = [
    "tremor_events-2010-2025.csv",
    "gnss_SOPAC_2010_2025.nc",
    "gnss_PANGA_2010_2025.nc",
    "gnss_unr_2010_2025.nc",
]

STATION_URL = (
    "https://service.earthscope.org/fdsnws/station/1/query?"
    "format=text&level=station"
    "&network=IU,II,IC,CU,GE,G,US,TA,CI,NC,UW,BK,PB,AK,AV,HV,PR,NN,CN"
    "&startbefore=2026-01-01&endafter=2005-01-01"
)

REGIONS = {
    "usgs_cascadia_m1": {
        "minlatitude": 40,
        "maxlatitude": 52,
        "minlongitude": -130,
        "maxlongitude": -118,
        "minmagnitude": 1.0,
    },
    "usgs_california_m1": {
        "minlatitude": 32,
        "maxlatitude": 42.5,
        "minlongitude": -125,
        "maxlongitude": -114,
        "minmagnitude": 1.0,
    },
    "usgs_alaska_m1": {
        "minlatitude": 50,
        "maxlatitude": 72,
        "minlongitude": -170,
        "maxlongitude": -130,
        "minmagnitude": 1.0,
    },
    "usgs_hawaii_m1": {
        "minlatitude": 18,
        "maxlatitude": 23,
        "minlongitude": -161.5,
        "maxlongitude": -154,
        "minmagnitude": 1.0,
    },
    "usgs_puerto_rico_m1": {
        "minlatitude": 16,
        "maxlatitude": 20.5,
        "minlongitude": -68.5,
        "maxlongitude": -62,
        "minmagnitude": 1.0,
    },
}


def _request(url: str, headers: dict[str, str] | None = None):
    merged = {"User-Agent": USER_AGENT}
    if headers:
        merged.update(headers)
    return urllib.request.Request(url, headers=merged)


def _download_resumable(url: str, out: Path, expected_size: int | None = None, force=False) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    if force and out.exists():
        out.unlink()
    for _ in range(30):
        got = out.stat().st_size if out.exists() else 0
        if expected_size and got >= expected_size:
            return out
        headers = {}
        mode = "wb"
        if got > 0:
            headers["Range"] = f"bytes={got}-"
            mode = "ab"
        with urllib.request.urlopen(_request(url, headers), timeout=300) as resp:
            with out.open(mode) as fh:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
        if not expected_size or out.stat().st_size >= expected_size:
            return out
        time.sleep(1.0)
    raise RuntimeError(f"incomplete download: {out} ({out.stat().st_size} of {expected_size})")


def download_crescent(force=False, include_gnss=True) -> list[Path]:
    with urllib.request.urlopen(_request(ZENODO_RECORD), timeout=60) as resp:
        record = json.load(resp)
    files = {item["key"]: item for item in record["files"]}
    wanted = CRESCENT_KEYS if include_gnss else [CRESCENT_KEYS[0]]
    paths = []
    for key in wanted:
        item = files[key]
        url = item["links"]["self"]
        out = EQ_CACHE / "crescent" / key
        paths.append(_download_resumable(url, out, expected_size=int(item["size"]), force=force))
    return paths


def download_station_inventory(force=False) -> Path:
    out = EQ_CACHE / "stations" / "earthscope_selected_stations_2005_2026.txt"
    if out.exists() and out.stat().st_size > 0 and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(_request(STATION_URL), timeout=180) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    out.write_text(text, encoding="utf-8")
    return out


def download_regional_catalogs(force=False, start_year=2005, end_year=2025) -> list[Path]:
    outdir = EQ_CACHE / "regional_catalogs"
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, region in REGIONS.items():
        for year in range(start_year, end_year + 1):
            out = outdir / f"{name}_{year}.csv"
            if out.exists() and out.stat().st_size > 0 and not force:
                paths.append(out)
                continue
            params = {
                "format": "csv",
                "starttime": f"{year}-01-01",
                "endtime": f"{year + 1}-01-01",
                "orderby": "time-asc",
                "limit": 20000,
                **region,
            }
            url = "https://earthquake.usgs.gov/fdsnws/event/1/query?" + urllib.parse.urlencode(params)
            try:
                with urllib.request.urlopen(_request(url), timeout=180) as resp:
                    text = resp.read().decode("utf-8", errors="replace")
            except Exception as exc:
                print(f"regional catalog failed: {name} {year}: {exc!r}", flush=True)
                continue
            out.write_text(text, encoding="utf-8")
            paths.append(out)
            time.sleep(0.25)
    return paths


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-crescent-gnss", action="store_true")
    ap.add_argument("--skip-regional-catalogs", action="store_true")
    ap.add_argument("--skip-stations", action="store_true")
    args = ap.parse_args(argv)

    paths = []
    paths.extend(download_crescent(force=args.force, include_gnss=not args.skip_crescent_gnss))
    if not args.skip_stations:
        paths.append(download_station_inventory(force=args.force))
    if not args.skip_regional_catalogs:
        paths.extend(download_regional_catalogs(force=args.force))
    for path in paths:
        print(f"{path} ({path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
