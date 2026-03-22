"""ProbSevere v3 storm-object data access with on-disk caching.

Downloads ProbSevere v3 JSON storm-object files from AWS S3 at 15-minute
intervals during convective hours (12 Z -- 06 Z) and caches as compressed
JSON.

Each time step contains storm objects with:
  - Storm ID, lat/lon, motion vectors
  - ProbSevere scores (PS, PStor, PShail, PSwind)
  - Atmospheric proxies (MUCAPE, MLCAPE, MLCIN, EBSHEAR, SRH, etc.)
  - Lightning (flash rate, flash density, MaxLLAz, etc.)
  - Radar-derived (MESH, VIL density, etc.)
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

from hazardpulse.data.http import fetch_bytes

# ---------------------------------------------------------------------------
# Project paths — check env var HAZARDPULSE_PROBSEVERE_CACHE first
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT = Path(os.environ.get(
    "HAZARDPULSE_PROBSEVERE_CACHE",
    str(PROJECT_ROOT / ".cache" / "probsevere"),
))

# ---------------------------------------------------------------------------
# AWS ProbSevere v3 endpoints
# ---------------------------------------------------------------------------

# ProbSevere v3 is available via MRMS/ProbSevere on AWS Open Data
PS_S3_ROOT = (
    "https://mrms-cp-probsevere.s3.amazonaws.com/ProbSevere/v3/"
    "{year}/{month}/{day}/"
)

# Convective hours to scan (12 Z to 06 Z next day, every 15 min)
CONVECTIVE_HOURS: list[int] = list(range(12, 24)) + list(range(0, 7))
SCAN_MINUTES: list[int] = [0, 15, 30, 45]

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_path(
    date_str: str,
    *,
    cache_dir: Path | None = None,
) -> Path:
    """Return the local cache path for a given date."""
    root = cache_dir or CACHE_ROOT
    return root / f"{date_str}.json.gz"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_cached_probsevere(
    date_str: str,
    *,
    cache_dir: Path | None = None,
) -> list[dict] | None:
    """Load ProbSevere storm objects from local cache only (no network).

    Parameters
    ----------
    date_str : str
        Date in ``YYYYMMDD`` format.
    cache_dir : Path, optional
        Override the default cache directory.

    Returns
    -------
    list[dict] or None
        List of time-step dicts with storm objects, or *None* if not cached.
    """
    path = _cache_path(date_str, cache_dir=cache_dir)
    if not path.exists():
        return None
    try:
        with gzip.open(path, "rb") as fh:
            data = json.loads(fh.read().decode("utf-8"))
        return data.get("time_steps", data) if isinstance(data, dict) else data
    except Exception:
        return None


def scan_probsevere_cache(
    cache_dir: Path | None = None,
) -> list[str]:
    """List available cached dates.

    Returns
    -------
    list[str]
        Sorted list of date strings (``YYYYMMDD``) present in cache.
    """
    root = cache_dir or CACHE_ROOT
    if not root.is_dir():
        return []
    dates: list[str] = []
    for path in root.iterdir():
        if path.name.endswith(".json.gz") and len(path.stem) >= 8:
            dates.append(path.stem[:8])
    dates.sort()
    return dates


def fetch_probsevere_day(
    date_str: str,
    *,
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> list[dict]:
    """Fetch all ProbSevere v3 storm objects for a day.

    Downloads from the AWS S3 bucket at 15-minute intervals during
    convective hours (12 Z -- 06 Z).  Results are cached as compressed
    JSON for subsequent calls.

    Parameters
    ----------
    date_str : str
        Date in ``YYYYMMDD`` format.
    cache_dir : Path, optional
        Override the default cache directory.
    refresh : bool
        Force re-download even if cached.

    Returns
    -------
    list[dict]
        List of time-step dicts.  Each dict contains:
        ``valid_time`` (ISO str) and ``storms`` (list of storm-object dicts).
    """
    if not refresh:
        cached = load_cached_probsevere(date_str, cache_dir=cache_dir)
        if cached is not None:
            return cached

    year = date_str[:4]
    month = date_str[4:6]
    day = date_str[6:8]

    time_steps: list[dict] = []
    for hour in CONVECTIVE_HOURS:
        for minute in SCAN_MINUTES:
            ts = _fetch_single_timestep(year, month, day, hour, minute)
            if ts is not None:
                time_steps.append(ts)

    # Persist to cache
    out_path = _cache_path(date_str, cache_dir=cache_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"date": date_str, "time_steps": time_steps},
        separators=(",", ":"),
    ).encode("utf-8")
    with gzip.open(out_path, "wb") as fh:
        fh.write(payload)

    return time_steps


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fetch_single_timestep(
    year: str,
    month: str,
    day: str,
    hour: int,
    minute: int,
) -> dict | None:
    """Fetch a single 15-minute ProbSevere v3 time step.

    Returns a dict with ``valid_time`` and ``storms`` or *None* on failure.
    """
    # ProbSevere v3 filename convention
    timestamp = f"{year}{month}{day}_{hour:02d}{minute:02d}00"
    filename = f"MRMS_ProbSevere_{timestamp}.json"
    url = (
        f"https://mrms-cp-probsevere.s3.amazonaws.com/"
        f"ProbSevere/v3/{year}/{month}/{day}/{filename}"
    )

    try:
        raw = fetch_bytes(url, namespace="probsevere", timeout=30)
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        return None

    valid_time = f"{year}-{month}-{day}T{hour:02d}:{minute:02d}:00Z"

    # ProbSevere v3 GeoJSON: features are storm objects
    storms: list[dict] = []
    features = data.get("features", [])
    for feat in features:
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        storm: dict = {
            "id": props.get("ID", 0),
            "lat": props.get("LAT", 0.0),
            "lon": props.get("LON", 0.0),
            "ps": props.get("PROB", 0.0),
            "ps_tor": props.get("PROBTOR", 0.0),
            "ps_hail": props.get("PROBHAIL", 0.0),
            "ps_wind": props.get("PROBWIND", 0.0),
            "mucape": props.get("MUCAPE", 0.0),
            "mlcape": props.get("MLCAPE", 0.0),
            "mlcin": props.get("MLCIN", 0.0),
            "ebshear": props.get("EBSHEAR", 0.0),
            "srh01": props.get("SRH01", 0.0),
            "mesh": props.get("MESH", 0.0),
            "vil_density": props.get("VILD", 0.0),
            "flash_rate": props.get("FLASHRATE", 0.0),
            "flash_density": props.get("FLASHDENSITY", 0.0),
            "maxllaz": props.get("MAXLLAZ", 0.0),
            "p98llaz": props.get("P98LLAZ", 0.0),
            "p98mlaz": props.get("P98MLAZ", 0.0),
            "lja": props.get("LJA", 0.0),
            "size": props.get("SIZE", 0.0),
            "motion_east": props.get("MOTION_EAST", 0.0),
            "motion_south": props.get("MOTION_SOUTH", 0.0),
        }
        # Attach geometry for frontend polygon rendering
        if geom:
            storm["geometry"] = geom
        storms.append(storm)

    return {"valid_time": valid_time, "storms": storms}
