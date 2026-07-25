#!/usr/bin/env python3
"""Download optional static priors for the earthquake operational ranker.

These are cache files, not source-controlled model weights:

* USGS M5+ 1900-1999 historical catalog for a pre-sample seismicity prior.
* GSRM v1.2 principal strain-rate grid from the primary UNAVCO/GSRM endpoint.
* GEM global active faults harmonized GeoJSON for static active-fault proximity priors.
"""
from __future__ import annotations

import argparse
import ssl
import time
import urllib.request
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
EQ_CACHE = REPO / ".cache" / "earthquake"
USER_AGENT = "hazardpulse-operational-priors/0.1"
USGS_API = "https://earthquake.usgs.gov/fdsnws/event/1/query"
GSRM_BASE = "https://gsrm.unavco.org/model/files/1.2"
GEM_ACTIVE_FAULTS = (
    "https://raw.githubusercontent.com/GEMScienceTools/gem-global-active-faults/"
    "master/geojson/gem_active_faults_harmonized.geojson"
)


def _fetch_text(url: str, timeout: int = 180) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout, context=ssl.create_default_context()) as resp:
        return resp.read().decode("utf-8", errors="replace")


def download_usgs_historical_m5(force=False) -> Path:
    out = EQ_CACHE / "usgs_historical_m5_1900_1999.csv"
    if out.exists() and not force:
        return out
    rows = []
    header = None
    for start in range(1900, 2000, 10):
        end = start + 10
        url = (
            f"{USGS_API}?format=csv&starttime={start}-01-01&endtime={end}-01-01"
            f"&minmagnitude=5.0&orderby=time-asc&limit=20000"
        )
        text = _fetch_text(url)
        lines = text.splitlines()
        if not lines:
            continue
        header = header or lines[0]
        rows.extend(lines[1:])
        time.sleep(0.5)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text((header or "") + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return out


def download_gsrm_principal(force=False) -> Path:
    out = EQ_CACHE / "gsrm" / "principal_strain_rate.dat"
    if out.exists() and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_fetch_text(f"{GSRM_BASE}/principal_strain_rate.dat", timeout=300), encoding="utf-8")
    return out


def download_gem_active_faults(force=False) -> Path:
    out = EQ_CACHE / "gem" / "gem_active_faults_harmonized.geojson"
    if out.exists() and out.stat().st_size > 0 and not force:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(GEM_ACTIVE_FAULTS, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120, context=ssl.create_default_context()) as resp:
        with out.open("wb") as fh:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                fh.write(chunk)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    usgs = download_usgs_historical_m5(force=args.force)
    gsrm = download_gsrm_principal(force=args.force)
    gem = download_gem_active_faults(force=args.force)
    print(f"USGS historical M5 prior: {usgs} ({usgs.stat().st_size:,} bytes)")
    print(f"GSRM principal strain prior: {gsrm} ({gsrm.stat().st_size:,} bytes)")
    print(f"GEM active faults prior: {gem} ({gem.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
