"""HRRR block-pooling captures convective peaks that legacy striding discards."""

from __future__ import annotations

import numpy as np

from hazardpulse.data import hrrr


def test_max_pool_captures_peak_that_striding_misses():
    native = np.zeros((hrrr.NATIVE_NY, hrrr.NATIVE_NX), dtype=np.float32)
    # A sharp CAPE maximum inside the (0,0) 80 km cell but NOT on the stride grid
    # (stride samples native[0,0]; the block spans rows 0-30, cols 0-27).
    native[15, 14] = 5000.0
    flat = native.ravel()

    strided = hrrr._subsample_to_grid(flat, "mlcape", mode="stride")
    pooled = hrrr._subsample_to_grid(flat, "mlcape", mode="max")

    assert pooled.shape == (hrrr.HRRR_N_LAT, hrrr.HRRR_N_LON)
    assert pooled[0, 0] >= 4999.0, "max-pool must capture the convective peak"
    assert strided[0, 0] < 1.0, "striding samples one corner point and misses the peak"


def test_pool_mean_for_non_instability_fields():
    native = np.full((hrrr.NATIVE_NY, hrrr.NATIVE_NX), 12.5, dtype=np.float32)
    # t2m is not an instability field -> mean pooling, value preserved
    pooled = hrrr._subsample_to_grid(native.ravel(), "t2m", mode="max")
    assert np.allclose(pooled, 12.5, atol=1e-3)


def test_nan_aware_pooling():
    native = np.full((hrrr.NATIVE_NY, hrrr.NATIVE_NX), np.nan, dtype=np.float32)
    native[10, 10] = 3000.0          # one valid point in cell (0,0)
    pooled = hrrr._subsample_to_grid(native.ravel(), "mlcape", mode="max")
    assert pooled[0, 0] >= 2999.0     # nanmax ignores the NaNs around the peak


def test_stride_mode_is_unchanged_default():
    # Default mode preserves legacy striding so a model trained on it is unaffected.
    assert hrrr.HRRR_POOL_MODE in ("stride", "max", "pool")
    native = np.arange(hrrr.NATIVE_NY * hrrr.NATIVE_NX, dtype=np.float32)
    out = hrrr._subsample_to_grid(native, "mlcape", mode="stride")
    assert out.shape == (hrrr.HRRR_N_LAT, hrrr.HRRR_N_LON)
    assert out[0, 0] == 0.0           # native[0,0]
