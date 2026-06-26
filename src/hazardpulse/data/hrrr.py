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

# Variable name -> relative path inside the hrrrzarr store.
# The Utah hrrrzarr bucket nests paths as "{level}/{var}/{level}/{var}".
# Mixed-layer CAPE is approximated by the 180_0mb layer; MU CAPE by 255_0mb.
#
# IMPORTANT: hrrrzarr does NOT publish named MLCAPE/MUCAPE/SBCAPE arrays the
# way the deprecated noaa-hrrr-bdp-pds wrfprsf bucket did. It exposes raw
# layer-CAPE values at fixed pressure-thickness layers (90/180/255 mb above
# ground). Standard NWS conventions:
#   MLCAPE  ≈ 90 mb mixed-layer CAPE  ->  90_0mb_above_ground/CAPE
#   MUCAPE  ≈ 255 mb most-unstable    ->  255_0mb_above_ground/CAPE
#   MLCIN   ≈ 90 mb mixed-layer CIN   ->  90_0mb_above_ground/CIN
# These are the canonical equivalents; the trained tornado GBT was trained
# against surface/MLCAPE which was NCEP's name for the same 0-90 mb
# mixed-layer integration, so they should be numerically close.
HRRR_VARS: dict[str, str] = {
    "cape": "surface/CAPE/surface/CAPE",
    "cin": "surface/CIN/surface/CIN",
    "mlcape": "90_0mb_above_ground/CAPE/90_0mb_above_ground/CAPE",
    "mlcin": "90_0mb_above_ground/CIN/90_0mb_above_ground/CIN",
    "mucape": "255_0mb_above_ground/CAPE/255_0mb_above_ground/CAPE",
    "srh_01": "1000_0m_above_ground/HLCY/1000_0m_above_ground/HLCY",
    "srh_03": "3000_0m_above_ground/HLCY/3000_0m_above_ground/HLCY",
    "refc": "entire_atmosphere/REFC/entire_atmosphere/REFC",
    "ushear_01": "0_1000m_above_ground/VUCSH/0_1000m_above_ground/VUCSH",
    "vshear_01": "0_1000m_above_ground/VVCSH/0_1000m_above_ground/VVCSH",
    "ushear_06": "0_6000m_above_ground/VUCSH/0_6000m_above_ground/VUCSH",
    "vshear_06": "0_6000m_above_ground/VVCSH/0_6000m_above_ground/VVCSH",
    "ustorm": "0_6000m_above_ground/USTM/0_6000m_above_ground/USTM",
    "vstorm": "0_6000m_above_ground/VSTM/0_6000m_above_ground/VSTM",
    "t2m": "2m_above_ground/TMP/2m_above_ground/TMP",
    "td2m": "2m_above_ground/DPT/2m_above_ground/DPT",
    "pwat": "entire_atmosphere_single_layer/PWAT/entire_atmosphere_single_layer/PWAT",
}

# Utah hrrrzarr bucket (active, public HRRR analysis Zarr mirror)
HRRR_ZARR_ROOT = (
    "https://hrrrzarr.s3.amazonaws.com/sfc/{date}/{date}_{hour:02d}z_anl.zarr"
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
) -> dict[str, np.ndarray] | None:
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
        # Treat all-NaN cache as missing so we re-fetch
        n_nan = sum(
            float(np.isnan(a).mean()) for a in cached.values() if isinstance(a, np.ndarray)
        )
        if n_nan / max(len(cached), 1) < 0.9:
            return cached

    import zarr
    import fsspec

    zarr_root = HRRR_ZARR_ROOT.format(date=date_str, hour=hour)
    try:
        store = fsspec.get_mapper(zarr_root)
        root = zarr.open(store, mode="r")
    except Exception as exc:
        # Don't return an all-NaN dict silently — callers cannot distinguish
        # that from "store populated but every variable is nan". Callers get
        # None and must decide to fallback or fail.
        print(f"  HRRR zarr store unreachable ({zarr_root}): {exc}")
        return None  # type: ignore[return-value]

    grids: dict[str, np.ndarray] = {}
    n_failed = 0
    for var_name, zarr_path in HRRR_VARS.items():
        try:
            full = np.asarray(root[zarr_path], dtype=np.float32)
            grids[var_name] = _subsample_to_grid(full.ravel(), var_name)
        except Exception as exc:
            print(f"  HRRR fetch failed for {var_name}: {exc}")
            grids[var_name] = np.full(
                (HRRR_N_LAT, HRRR_N_LON), np.nan, dtype=np.float32
            )
            n_failed += 1

    # If more than half the variables failed to fetch, treat the whole pull
    # as failed — partial data produces misleading ML output.
    if n_failed > len(HRRR_VARS) // 2:
        print(
            f"  HRRR pull failed: {n_failed}/{len(HRRR_VARS)} variables unavailable. "
            "Discarding partial results."
        )
        return None  # type: ignore[return-value]

    out_path = _npz_path(date_str, hour, cache_dir=cache_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out_path), **grids)
    return grids


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


# Fields where the 80 km cell's PEAK is the meaningful tornado signal (instability,
# storm-relative helicity, reflectivity) -> block-MAX. Everything else (winds,
# temperature, moisture, inhibition) -> block-MEAN. Striding (one native point per
# 80 km cell) discards 99.9% of the field and misses these peaks; pooling keeps them.
_HRRR_MAX_FIELDS = frozenset({
    "cape", "mlcape", "mucape", "srh_01", "srh_03", "refc",
})

# Extraction mode. "stride" reproduces the legacy single-point subsample (kept as
# the default so the CURRENTLY-TRAINED live model never sees a shifted feature
# distribution). The retrain pipeline sets HAZARDPULSE_HRRR_POOL=max to use the
# physically-correct block pooling; once a model is retrained on it, flip the default.
HRRR_POOL_MODE: str = os.environ.get("HAZARDPULSE_HRRR_POOL", "stride").strip().lower()

NATIVE_NY, NATIVE_NX = 1059, 1799


def _block_pool(full2d: np.ndarray, var_name: str) -> np.ndarray:
    """Reduce the native HRRR grid to the 80 km grid by NaN-aware block pooling.

    Each 80 km cell aggregates the native ~3 km points it contains: MAX for
    instability/helicity/reflectivity (capture the convective peak), MEAN
    otherwise.
    """
    ny, nx = full2d.shape
    by, bx = ny // HRRR_N_LAT, nx // HRRR_N_LON
    if by < 1 or bx < 1:
        return np.zeros((HRRR_N_LAT, HRRR_N_LON), dtype=np.float32)
    cropped = full2d[: by * HRRR_N_LAT, : bx * HRRR_N_LON]
    blocks = cropped.reshape(HRRR_N_LAT, by, HRRR_N_LON, bx)
    with np.errstate(invalid="ignore"):
        if var_name in _HRRR_MAX_FIELDS:
            pooled = np.nanmax(blocks, axis=(1, 3))
        else:
            pooled = np.nanmean(blocks, axis=(1, 3))
    return np.asarray(pooled, dtype=np.float32)


def _subsample_to_grid(flat: np.ndarray, var_name: str, *, mode: str | None = None) -> np.ndarray:
    """Reduce a flat native HRRR array to the 80 km grid (pooling or legacy stride)."""
    target_shape = (HRRR_N_LAT, HRRR_N_LON)
    mode = (mode or HRRR_POOL_MODE)

    if flat.size == NATIVE_NY * NATIVE_NX:
        full2d = flat.reshape((NATIVE_NY, NATIVE_NX))
    elif flat.size >= HRRR_N_LAT * HRRR_N_LON:
        # Already subsampled / different shape — take first N_LAT*N_LON.
        return flat[: HRRR_N_LAT * HRRR_N_LON].reshape(target_shape).astype(np.float32)
    else:
        return np.zeros(target_shape, dtype=np.float32)

    if mode == "max" or mode == "pool":
        return _block_pool(full2d, var_name)

    # Legacy striding (one native point per 80 km cell).
    stride_y = NATIVE_NY // HRRR_N_LAT
    stride_x = NATIVE_NX // HRRR_N_LON
    subsampled = full2d[::stride_y, ::stride_x][:HRRR_N_LAT, :HRRR_N_LON]
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
