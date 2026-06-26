"""Live tornado scoring features == training features (no train/serve skew).

The dataset builder extracts features one cell at a time (cell_features); the live
scorer extracts the whole grid at once (grid_feature_matrix). If those two ever
disagree, the deployed forest sees different inputs than it trained on. This pins
them together, byte-for-byte, plus the 26-feature contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from hazardpulse.data.hrrr import HRRR_VARS, HRRR_N_LAT, HRRR_N_LON  # noqa: E402
from hazardpulse.tornado import hrrr_env  # noqa: E402
from hazardpulse.tornado.coherence_engine import compute_derived_hrrr  # noqa: E402


def _synthetic_grid():
    rng = np.random.RandomState(3)
    grids = {}
    for v in HRRR_VARS:
        grids[v] = (rng.uniform(-5, 5, size=(HRRR_N_LAT, HRRR_N_LON))).astype(np.float32)
    # plausible instability/moisture so derived params are well-defined
    grids["mlcape"] = rng.uniform(0, 4000, (HRRR_N_LAT, HRRR_N_LON)).astype(np.float32)
    grids["cape"] = grids["mlcape"] * 1.2
    grids["srh_01"] = rng.uniform(-50, 400, (HRRR_N_LAT, HRRR_N_LON)).astype(np.float32)
    grids["t2m"] = np.full((HRRR_N_LAT, HRRR_N_LON), 298.0, np.float32)
    grids["td2m"] = np.full((HRRR_N_LAT, HRRR_N_LON), 290.0, np.float32)
    grids["mlcin"] = np.full((HRRR_N_LAT, HRRR_N_LON), -30.0, np.float32)
    return grids


def test_feature_contract():
    assert hrrr_env.N_FEATURES == 26
    assert hrrr_env.FEATURE_NAMES[: len(HRRR_VARS)] == list(HRRR_VARS)
    assert "stp_eff" in hrrr_env.DERIVED_NAMES and "srh_05_est" in hrrr_env.DERIVED_NAMES


def test_grid_matrix_matches_per_cell():
    grids = _synthetic_grid()
    derived = compute_derived_hrrr(grids)
    X, cape = hrrr_env.grid_feature_matrix(grids)
    assert X.shape == (HRRR_N_LAT * HRRR_N_LON, 26)
    # check a spread of cells: vectorized row == per-cell vector, byte for byte
    for (i, j) in [(0, 0), (15, 30), (33, 62), (7, 50), (20, 5)]:
        per_cell = hrrr_env.cell_features(grids, derived, i, j)
        row = X[i * HRRR_N_LON + j]
        assert np.array_equal(row, per_cell), f"mismatch at ({i},{j})"
    assert np.array_equal(np.asarray(cape, np.float32), grids["mlcape"])
