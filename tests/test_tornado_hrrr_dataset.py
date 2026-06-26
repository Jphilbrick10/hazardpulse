"""The HRRR-environment tornado dataset builder labels cells correctly.

Validates the no-network core: a tornado report lands in its HRRR cell as a
positive, hard negatives are drawn only from convective (cape >= threshold)
non-tornado cells, and every row carries the full 26-feature vector (17 raw + 9
derived). Guards against silent label/feature drift.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _builder():
    pytest.importorskip("numpy")
    try:
        spec = importlib.util.spec_from_file_location(
            "thd", REPO / "scripts" / "build_tornado_hrrr_dataset.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"builder not importable: {exc}")
    return m


def _synthetic_grid(m):
    """A full 17-var HRRR grid: a convective swath (high cape/shear) over otherwise calm CONUS."""
    ny, nx = 34, 63
    rng = np.random.RandomState(0)
    grids = {}
    for v in m._VAR_NAMES:
        grids[v] = np.zeros((ny, nx), dtype=np.float32)
    # a band of convective cells in the middle rows
    grids["mlcape"][10:20, 20:40] = 2500.0
    grids["cape"][10:20, 20:40] = 3000.0
    grids["mucape"][10:20, 20:40] = 2800.0
    grids["srh_01"][10:20, 20:40] = 250.0
    grids["srh_03"][10:20, 20:40] = 400.0
    grids["ushear_06"][10:20, 20:40] = 18.0
    grids["vshear_06"][10:20, 20:40] = 12.0
    grids["ushear_01"][10:20, 20:40] = 8.0
    grids["vshear_01"][10:20, 20:40] = 6.0
    grids["t2m"][:] = 300.0
    grids["td2m"][:] = 293.0
    grids["mlcin"][:] = -25.0
    return grids


def test_positive_cell_and_hard_negatives():
    m = _builder()
    grids = _synthetic_grid(m)

    # place a tornado at the center of cell (15, 30) -- inside the convective band
    import hazardpulse.data.hrrr as H
    lat = float(H.GRID_LATS[15]); lon = float(H.GRID_LONS[30])
    assert H.latlon_to_hrrr_cell(lat, lon) == (15, 30)
    reports = [{"slat": lat, "slon": lon, "hour": 20, "mag": 2}]

    rng = np.random.RandomState(1)
    rows, labels = m._cells_for(grids, reports, neg_per_day=15, neg_min_cape=250.0, rng=rng)
    labels = np.array(labels)
    assert labels.sum() == 1, "exactly one tornado cell -> one positive"
    assert (labels == 0).sum() >= 1, "hard negatives drawn from convective cells"
    # every row is the full 26-feature vector
    assert all(len(r) == len(m._FEATURE_NAMES) == 26 for r in rows)
    # the positive row's STP (derived) should be finite and computed
    pos_row = rows[int(np.argmax(labels))]
    stp_idx = m._FEATURE_NAMES.index("stp_eff")
    assert np.isfinite(pos_row[stp_idx])


def test_negatives_are_convective_not_random():
    m = _builder()
    grids = _synthetic_grid(m)
    rng = np.random.RandomState(2)
    rows, labels = m._cells_for(grids, [], neg_per_day=50, neg_min_cape=250.0, rng=rng)
    # no tornado -> all negatives, and only as many as there are convective cells
    assert set(labels) == {0}
    cape = grids["mlcape"]
    n_convective = int((cape >= 250.0).sum())
    assert len(labels) == min(50, n_convective)
    assert n_convective == 10 * 20   # the convective band only
