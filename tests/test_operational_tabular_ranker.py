import importlib.util
import json
import sys
from pathlib import Path

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


def test_causal_cell_history_uses_only_prior_rows():
    ranker = _load_script_module(
        "hazardpulse_operational_tabular_ranker_test",
        "scripts/research/operational_tabular_ranker.py",
    )
    Y = np.array([1, 0, 1], dtype=int)
    T = np.array([100.0, 110.0, 200.0])
    lat = np.array([1.0, 1.0, 1.0])
    lon = np.array([2.0, 2.0, 2.0])

    H, names = ranker._causal_cell_history(Y, T, lat, lon)
    rate_s2 = H[:, names.index("cell_prev_rate_s2")]

    assert rate_s2[0] == 0.25
    assert np.isclose(rate_s2[1], 1.5 / 3.0)
    assert np.isclose(rate_s2[2], 1.5 / 4.0)


def test_causal_neighbor_labels_wait_for_matured_windows():
    ranker = _load_script_module(
        "hazardpulse_operational_tabular_ranker_test2",
        "scripts/research/operational_tabular_ranker.py",
    )
    Y = np.array([1, 0, 0], dtype=int)
    T = np.array([0.0, 20.0 * ranker.SEC_DAY, 40.0 * ranker.SEC_DAY])
    lat = np.array([0.0, 0.0, 0.0])
    lon = np.array([0.0, 0.0, 0.0])

    H, names = ranker._causal_neighbor_label_features(Y, T, lat, lon, label_days=30.0)
    n_col = names.index("neighbor_all_100km_n")
    rate_col = names.index("neighbor_all_100km_rate_s5")

    assert H[1, n_col] == 0.0
    assert H[2, n_col] > 0.0
    assert H[2, rate_col] > H[1, rate_col]


def test_gem_active_fault_features_parse_distance_and_slip(tmp_path):
    ranker = _load_script_module(
        "hazardpulse_operational_tabular_ranker_test3",
        "scripts/research/operational_tabular_ranker.py",
    )
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "net_slip_rate": "(6.5,4.0,8.0)",
                    "slip_type": "Reverse",
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[0.0, 0.0], [0.1, 0.0]],
                },
            }
        ],
    }
    path = tmp_path / "faults.geojson"
    path.write_text(json.dumps(geojson), encoding="utf-8")

    F, names = ranker._gem_active_fault_features(
        np.array([0.0, 10.0]),
        np.array([0.0, 10.0]),
        path,
    )

    assert F.shape[0] == 2
    assert "gemfault_all_logdist" in names
    assert "gemfault_reverse_nearest_log_slip_rate" in names
    assert F[0, names.index("gemfault_all_logdist")] < F[1, names.index("gemfault_all_logdist")]
    assert np.isclose(
        F[0, names.index("gemfault_nearest_log_slip_rate")],
        np.log1p(6.5),
    )
