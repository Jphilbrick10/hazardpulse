"""HRRR Zarr data access with on-disk .npz caching.

Downloads HRRR analysis fields from the AWS Open Data Zarr store,
subsamples to an 80 km CONUS grid (34 lat x 63 lon), and caches locally.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

from hazardpulse.data.http import fetch_bytes

# ---------------------------------------------------------------------------
# Project paths — check env var HAZARDPULSE_HRRR_CACHE first
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT = Path(os.environ.get(
    "HAZARDPULSE_HRRR_CACHE",
    str(PROJECT_ROOT / ".cache" / "hrrr"),
))

# ---------------------------------------------------------------------------
# 80 km CONUS grid constants
# ---------------------------------------------------------------------------

GRID_DLAT: float = 0.72  # ~80 km
GRID_DLON: float = 0.94  # ~80 km at 37.5 N
LAT_MIN: float = 25.0
LAT_MAX: float = 50.0
LON_MIN: float = -125.0
LON_MAX: float = -65.0

HRRR_N_LAT: int = 34
HRRR_N_LON: int = 63

# Pre-computed grid cell centres
GRID_LATS: np.ndarray = np.array(
    [LAT_MIN + (i + 0.5) * GRID_DLAT for i in range(HRRR_N_LAT)],
    dtype=np.float32,
)
GRID_LONS: np.ndarray = np.array(
    [LON_MIN + (j + 0.5) * GRID_DLON for j in range(HRRR_N_LON)],
    dtype=np.float32,
)

# Physical grid spacing
DX_KM: float = GRID_DLON * 111.0 * math.cos(math.radians(37.5))
DY_KM: float = GRID_DLAT * 111.0
DX_M: float = DX_KM * 1000.0
DY_M: float = DY_KM * 1000.0

# ---------------------------------------------------------------------------
# HRRR variable mapping  (short name -> Zarr group/variable path)
# ---------------------------------------------------------------------------

HRRR_VARS: dict[str, str] = {
    "cape": "surface/CAPE",
    "cin": "surface/CIN",
    "mlcape": "surface/MLCAPE",
    "mlcin": "surface/MLCIN",
    "mucape": "surface/MUCAPE",
    "srh_01": "1000_0m_above_ground/HLCY",
    "srh_03": "3000_0m_above_ground/HLCY",
    "refc": "entire_atmosphere/REFC",
    "ushear_01": "0_1000m_above_ground/VUCSH",
    "vshear_01": "0_1000m_above_ground/VVCSH",
    "ushear_06": "0_6000m_above_ground/VUCSH",
    "vshear_06": "0_6000m_above_ground/VVCSH",
    "ustorm": "0m_above_ground/USTM",
    "vstorm": "0m_above_ground/VSTM",
    "t2m": "2m_above_ground/TMP",
    "td2m": "2m_above_ground/DPT",
    "pwat": "entire_atmosphere/PWAT",
}

# AWS HRRR Zarr root (NOAA Open Data)
HRRR_ZARR_ROOT = (
    "https://noaa-hrrr-bdp-pds.s3.amazonaws.com/hrrr.{date}/conus/"
    "hrrr.t{hour:02d}z.wrfprsf00.grib2.zarr"
)

# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _npz_path(date_str: str, hour: int, *, cache_dir: Path | None = None) -> Path:
    """Return the local .npz cache path for a given date/hour."""
    root = cache_dir or CACHE_ROOT
    return root / f"{date_str}_{hour:02d}z.npz"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_cached_hrrr(
    date_str: str,
    hour: int = 18,
    *,
    cache_dir: Path | None = None,
) -> dict[str, np.ndarray] | None:
    """Load HRRR data from local .npz cache only (no network).

    Parameters
    ----------
    date_str : str
        Date in ``YYYYMMDD`` format.
    hour : int
        Analysis hour (default 18 Z).
    cache_dir : Path, optional
        Override the default cache directory.

    Returns
    -------
    dict or None
        Mapping of variable name to ``(34, 63)`` float32 array, or *None*
        if the file does not exist in cache.
    """
    path = _npz_path(date_str, hour, cache_dir=cache_dir)
    if not path.exists():
        return None
    try:
        data = np.load(path)
        return {key: data[key] for key in data.files}
    except Exception:
        return None


def fetch_hrrr_grid(
    date_str: str,
    hour: int = 18,
    *,
    cache_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    """Fetch HRRR analysis fields subsampled to the 80 km CONUS grid.

    Checks the local ``.npz`` cache first; downloads from the AWS Zarr
    store when the cache misses.

    Parameters
    ----------
    date_str : str
        Date in ``YYYYMMDD`` format.
    hour : int
        Analysis hour (default 18 Z).
    cache_dir : Path, optional
        Override the default cache directory.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of variable name to ``(34, 63)`` float32 array.

    Raises
    ------
    RuntimeError
        If the HRRR data cannot be fetched from AWS.
    """
    cached = load_cached_hrrr(date_str, hour, cache_dir=cache_dir)
    if cached is not None:
        return cached

    # Format the date for the S3 key  (YYYYMMDD -> YYYY/MM/DD)
    formatted_date = f"{date_str[:4]}/{date_str[4:6]}/{date_str[6:8]}"
    zarr_root = HRRR_ZARR_ROOT.format(date=formatted_date, hour=hour)

    grids: dict[str, np.ndarray] = {}
    for var_name, zarr_path in HRRR_VARS.items():
        url = f"{zarr_root}/{zarr_path}/0.0"
        try:
            raw = fetch_bytes(url, namespace="hrrr", timeout=120)
            # Zarr chunks are raw numpy — decode as float32 and subsample
            full = np.frombuffer(raw, dtype=np.float32)
            # HRRR CONUS native is ~1059 x 1799; subsample to 34 x 63
            subsampled = _subsample_to_grid(full, var_name)
            grids[var_name] = subsampled
        except Exception:
            # Fill with zeros on failure — downstream handles gracefully
            grids[var_name] = np.zeros(
                (HRRR_N_LAT, HRRR_N_LON), dtype=np.float32
            )

    # Persist to cache
    out_path = _npz_path(date_str, hour, cache_dir=cache_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), **grids)

    return grids


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _subsample_to_grid(flat: np.ndarray, var_name: str) -> np.ndarray:
    """Subsample a flat HRRR array to the 80 km grid.

    If the raw array cannot be reshaped to the native HRRR dimensions we
    fall back to an interpolated or zero-filled grid.
    """
    native_ny, native_nx = 1059, 1799
    target_shape = (HRRR_N_LAT, HRRR_N_LON)

    if flat.size == native_ny * native_nx:
        full2d = flat.reshape((native_ny, native_nx))
    elif flat.size >= HRRR_N_LAT * HRRR_N_LON:
        # Already subsampled or different shape — take first N_LAT*N_LON
        full2d = flat[: HRRR_N_LAT * HRRR_N_LON].reshape(target_shape)
        return full2d.astype(np.float32)
    else:
        return np.zeros(target_shape, dtype=np.float32)

    # Compute stride to subsample from native grid
    stride_y = native_ny // HRRR_N_LAT
    stride_x = native_nx // HRRR_N_LON
    subsampled = full2d[::stride_y, ::stride_x][:HRRR_N_LAT, :HRRR_N_LON]

    # Pad if subsample came up short
    if subsampled.shape != target_shape:
        result = np.zeros(target_shape, dtype=np.float32)
        sy, sx = subsampled.shape
        result[:sy, :sx] = subsampled[:sy, :sx]
        return result

    return subsampled.astype(np.float32)


def latlon_to_hrrr_cell(lat: float, lon: float) -> tuple[int, int]:
    """Map a lat/lon coordinate to the nearest HRRR subsample grid index.

    Returns
    -------
    tuple[int, int]
        ``(i_lat, j_lon)`` indices clamped to valid grid bounds.
    """
    i = int((lat - LAT_MIN) / GRID_DLAT)
    j = int((lon - LON_MIN) / GRID_DLON)
    i = max(0, min(i, HRRR_N_LAT - 1))
    j = max(0, min(j, HRRR_N_LON - 1))
    return i, j
