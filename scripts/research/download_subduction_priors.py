#!/usr/bin/env python3
"""Download subduction-zone physics priors for earthquake operational research."""
from __future__ import annotations

import argparse
import json
import subprocess
import zipfile
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EQ_CACHE = REPO / ".cache" / "earthquake"
SLAB2_ITEM = "https://www.sciencebase.gov/catalog/item/5aa1b00ee4b0b1c392e86467?format=json"
SLAB2_TAR = EQ_CACHE / "slab2" / "Slab2Distribute_Mar2018.tar.gz"
SLAB2_DIR = EQ_CACHE / "slab2" / "Slab2Distribute_Mar2018"
COUPLING_URL = "https://couplingcloud.ucsd.edu/download/all_models"
COUPLING_ZIP = EQ_CACHE / "coupling_cloud" / "allCouplingModels.zip"
COUPLING_DIR = EQ_CACHE / "coupling_cloud" / "extracted"


def _slab2_download_url() -> tuple[str, int]:
    req = urllib.request.Request(SLAB2_ITEM, headers={"User-Agent": "hazardpulse-slab2"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    item = next(f for f in data["files"] if f["name"] == "Slab2Distribute_Mar2018.tar.gz")
    return item["downloadUri"], int(item["size"])


def download_slab2(force=False) -> Path:
    url, expected_size = _slab2_download_url()
    SLAB2_TAR.parent.mkdir(parents=True, exist_ok=True)
    if force and SLAB2_TAR.exists():
        SLAB2_TAR.unlink()
    if not SLAB2_TAR.exists() or SLAB2_TAR.stat().st_size != expected_size:
        subprocess.run(
            [
                "curl.exe",
                "-L",
                "--retry",
                "10",
                "--retry-all-errors",
                "--connect-timeout",
                "60",
                "-A",
                "hazardpulse-slab2",
                "-o",
                str(SLAB2_TAR),
                url,
            ],
            cwd=REPO,
            check=True,
        )
    if SLAB2_TAR.stat().st_size != expected_size:
        raise RuntimeError(f"Slab2 tar size mismatch: {SLAB2_TAR.stat().st_size} != {expected_size}")
    subprocess.run(["tar", "-tzf", str(SLAB2_TAR)], cwd=REPO, check=True, stdout=subprocess.DEVNULL)
    if force and SLAB2_DIR.exists():
        # Keep deletion scoped to the known cache directory.
        for path in sorted(SLAB2_DIR.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        SLAB2_DIR.rmdir()
    if not SLAB2_DIR.exists():
        subprocess.run(["tar", "-xzf", str(SLAB2_TAR), "-C", str(SLAB2_TAR.parent)], cwd=REPO, check=True)
    return SLAB2_DIR


def download_coupling_cloud(force=False) -> Path:
    COUPLING_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if force and COUPLING_ZIP.exists():
        COUPLING_ZIP.unlink()
    if not COUPLING_ZIP.exists() or COUPLING_ZIP.stat().st_size == 0:
        subprocess.run(
            [
                "curl.exe",
                "-L",
                "--retry",
                "10",
                "--retry-all-errors",
                "--connect-timeout",
                "60",
                "-A",
                "hazardpulse-coupling",
                "-o",
                str(COUPLING_ZIP),
                COUPLING_URL,
            ],
            cwd=REPO,
            check=True,
        )
    with zipfile.ZipFile(COUPLING_ZIP) as zf:
        for info in zf.infolist():
            if not (info.filename.endswith(".nc") or info.filename.endswith("metadata.yaml")):
                continue
            target = COUPLING_DIR / info.filename
            if force and target.exists():
                target.unlink()
            if target.exists() and target.stat().st_size == info.file_size:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, target.open("wb") as dst:
                dst.write(src.read())
    return COUPLING_DIR


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-coupling-cloud", action="store_true")
    args = ap.parse_args(argv)
    slab2 = download_slab2(force=args.force)
    n_grids = len(list(slab2.glob("*.grd")))
    print(f"Slab2 grids: {slab2} ({n_grids} grids)")
    if not args.skip_coupling_cloud:
        coupling = download_coupling_cloud(force=args.force)
        n_nc = len(list(coupling.rglob("*.nc")))
        print(f"Coupling Cloud models: {coupling} ({n_nc} NetCDFs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
