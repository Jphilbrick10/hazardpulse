import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str, relative_path: str):
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_ndk_parser_preserves_origin_time():
    downloader = _load_script_module(
        "hazardpulse_download_earthquake_data_gcmt_test",
        "scripts/download_earthquake_data.py",
    )
    block = [
        "MLI  1976/01/01 01:29:39.6 -28.61 -177.64  59.0 6.2 0.0 KERMADEC ISLANDS REGION ",
        "M010176A         B:  0    0   0 S:  0    0   0 M: 12   30 135 CMT: 1 BOXHD:  9.4",
        "CENTROID:     13.8 0.2 -29.25 0.02 -176.96 0.01  47.8  0.6 FREE O-00000000000000",
        "26  7.680 0.090  0.090 0.060 -7.770 0.070  1.390 0.160  4.520 0.160 -3.260 0.060",
        "V10   8.940 75 283   1.260  2  19 -10.190 15 110   9.560 202 30   93  18 60   88",
    ]

    rec = downloader._parse_ndk_block(block)

    assert rec is not None
    assert rec["time"] == "1976-01-01T01:29:39.600Z"
    assert rec["event_id"] == "M010176A"


def test_gcmt_coulomb_feature_is_causal():
    op = _load_script_module(
        "hazardpulse_deep_operational_earthquake_gcmt_test",
        "scripts/deep_operational_earthquake.py",
    )
    ref = 1_600_000_000.0
    cat = SimpleNamespace(mags=np.array([6.2], dtype=float))
    bi = np.array([0])
    bd = np.array([10.0])
    bdays = np.array([30.0])

    op._GLA = np.array([0.0])
    op._GLO = np.array([0.0])
    op._GM0 = np.array([1.0e28])
    op._GT = np.array([ref + 86_400.0])
    future_only = op._extra_feats(0.0, 0.0, ref, cat, bi, bd, bdays)[-1]

    op._GLA = np.array([0.0, 0.0])
    op._GLO = np.array([0.0, 0.0])
    op._GM0 = np.array([1.0e20, 1.0e28])
    op._GT = np.array([ref - 86_400.0, ref + 86_400.0])
    past_small_future_huge = op._extra_feats(0.0, 0.0, ref, cat, bi, bd, bdays)[-1]

    op._GLA = np.array([0.0])
    op._GLO = np.array([0.0])
    op._GM0 = np.array([1.0e20])
    op._GT = np.array([ref - 86_400.0])
    past_available = op._extra_feats(0.0, 0.0, ref, cat, bi, bd, bdays)[-1]

    assert future_only == 0.0
    assert past_available > 0.0
    assert past_small_future_huge == past_available
