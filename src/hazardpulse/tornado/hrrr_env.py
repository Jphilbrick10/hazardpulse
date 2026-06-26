"""Shared HRRR-environment tornado feature extraction (train == serve).

ONE definition of the 26-feature vector (17 raw HRRR analysis variables + 9 derived
tornado discriminators) used by BOTH the offline dataset builder and the live scorer,
so the served features are byte-identical to the trained ones -- no train/serve skew.
The derived block (bulk shear, LCL, 0-500 m SRH, Significant Tornado Parameter, RFD
warmth, streamwise vorticity) is what actually separates tornadic from non-tornadic
storm environments.
"""

from __future__ import annotations

import numpy as np

from hazardpulse.data.hrrr import HRRR_VARS, HRRR_N_LAT, HRRR_N_LON
from hazardpulse.tornado.coherence_engine import compute_derived_hrrr

RAW_NAMES = list(HRRR_VARS)
DERIVED_NAMES = [
    "shear_01", "shear_06", "storm_speed", "td_depression", "lcl_est",
    "srh_05_est", "stp_eff", "rfd_warmth", "streamwise_vort",
]
FEATURE_NAMES = RAW_NAMES + DERIVED_NAMES
N_FEATURES = len(FEATURE_NAMES)


def cell_features(grids: dict, derived: dict, i: int, j: int) -> np.ndarray:
    """The 26-feature vector for HRRR cell (i, j)."""
    raw = [float(grids[v][i, j]) for v in RAW_NAMES]
    der = [float(derived[v][i, j]) for v in DERIVED_NAMES]
    return np.array(raw + der, dtype=np.float32)


def grid_feature_matrix(grids: dict):
    """Vectorized (N_cells, 26) feature matrix + the cape grid, for whole-grid scoring.

    Returns (X, cape) where X[i*HRRR_N_LON + j] is the feature vector for cell (i, j)
    in row-major order, and cape is the (34, 63) instability grid for masking.
    """
    derived = compute_derived_hrrr(grids)
    layers = [np.asarray(grids[v], np.float32) for v in RAW_NAMES]
    layers += [np.asarray(derived[v], np.float32) for v in DERIVED_NAMES]
    # stack to (34, 63, 26) then flatten cells -> (34*63, 26)
    cube = np.stack(layers, axis=-1)
    X = cube.reshape(HRRR_N_LAT * HRRR_N_LON, N_FEATURES)
    cape = np.asarray(grids.get("mlcape", grids.get("cape")), np.float32)
    return X, cape
