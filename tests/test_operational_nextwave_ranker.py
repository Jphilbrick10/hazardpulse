import importlib.util
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


def test_event_cloud_features_use_only_prior_events():
    mod = _load_script_module(
        "hazardpulse_operational_nextwave_test",
        "scripts/research/operational_nextwave_ranker.py",
    )
    day = mod.otr.SEC_DAY
    T = np.array([100 * day, 200 * day], dtype=float)
    lat = np.array([0.0, 0.0])
    lon = np.array([0.0, 0.0])
    events = (
        np.array([50 * day, 150 * day], dtype=float),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([2.0, 4.0]),
    )

    F, names = mod._event_cloud_features(
        T,
        lat,
        lon,
        events,
        "synthetic",
        windows=[90],
        radii=[100],
        weights="sum",
    )

    assert names == ["synthetic_90d_100km_n", "synthetic_90d_100km_sum", "synthetic_90d_100km_max"]
    assert np.isclose(F[0, 0], np.log1p(1))
    assert np.isclose(F[0, 1], np.log1p(2.0))
    assert np.isclose(F[1, 0], np.log1p(1))
    assert np.isclose(F[1, 1], np.log1p(4.0))


def test_station_inventory_features_respect_station_lifetime(tmp_path):
    mod = _load_script_module(
        "hazardpulse_operational_nextwave_test2",
        "scripts/research/operational_nextwave_ranker.py",
    )
    path = tmp_path / "stations.txt"
    path.write_text(
        "#Network | Station | Latitude | Longitude | Elevation | SiteName | StartTime | EndTime\n"
        "XX|AAA|0.0|0.0|0|Test|2000-01-01T00:00:00.0000|2000-06-01T00:00:00.0000\n",
        encoding="utf-8",
    )
    T = np.array(
        [
            mod.dt.datetime(2000, 3, 1, tzinfo=mod.dt.timezone.utc).timestamp(),
            mod.dt.datetime(2000, 9, 1, tzinfo=mod.dt.timezone.utc).timestamp(),
        ],
        dtype=float,
    )
    F, names = mod.station_inventory_features(T, np.array([0.0, 0.0]), np.array([0.0, 0.0]), path)

    col = names.index("station_100km_n")
    assert np.isclose(F[0, col], np.log1p(1))
    assert F[1, col] == 0.0


def test_decimal_year_tenv3_metric_uses_past_only(tmp_path):
    mod = _load_script_module(
        "hazardpulse_operational_nextwave_test3",
        "scripts/research/operational_nextwave_ranker.py",
    )
    path = tmp_path / "AAA.tenv3"
    header = (
        "site YYMMMDD yyyy.yyyy __MJD week d reflon _e0(m) __east(m) ____n0(m) "
        "_north(m) u0(m) ____up(m) _ant(m) sig_e(m) sig_n(m) sig_u(m) __corr_en "
        "__corr_eu __corr_nu _latitude(deg) _longitude(deg) __height(m)\n"
    )
    rows = [header]
    for i in range(900):
        year = 2000.0 + i / 365.25
        east = i * 0.001
        rows.append(
            f"AAA 00JAN01 {year:.6f} 0 0 0 0 0 {east:.6f} 0 0.0 0 0.0 0 "
            "0.001 0.001 0.002 0 0 0 0.0 0.0 0.0\n"
        )
    path.write_text("".join(rows), encoding="utf-8")
    station = mod._load_tenv3_station(path)

    ref = mod.dt.datetime(2002, 7, 1, tzinfo=mod.dt.timezone.utc).timestamp()
    metric = mod._station_metrics_at_ref(station, ref)

    assert metric is not None
    assert metric[0] > 100.0


def test_slab2_geometry_features_from_points():
    mod = _load_script_module(
        "hazardpulse_operational_nextwave_test4",
        "scripts/research/operational_nextwave_ranker.py",
    )
    slab = {
        "lat": np.array([0.0, 5.0], dtype=np.float32),
        "lon": np.array([0.0, 5.0], dtype=np.float32),
        "depth": np.array([30.0, 150.0], dtype=np.float32),
        "dip": np.array([15.0, 60.0], dtype=np.float32),
        "strike": np.array([90.0, 180.0], dtype=np.float32),
        "thickness": np.array([10.0, 20.0], dtype=np.float32),
        "uncertainty": np.array([5.0, 8.0], dtype=np.float32),
    }

    F, names = mod._slab2_geometry_from_points(
        np.array([0.0, 5.0]),
        np.array([0.0, 5.0]),
        slab,
    )

    assert F.shape[0] == 2
    assert "slab2_nearest_logdist" in names
    assert "slab2_shallow70_logdist" in names
    assert F[0, names.index("slab2_nearest_depth")] < F[1, names.index("slab2_nearest_depth")]
    assert F[0, names.index("slab2_shallow70_logdist")] < F[1, names.index("slab2_shallow70_logdist")]


def test_coupling_cloud_features_from_points():
    mod = _load_script_module(
        "hazardpulse_operational_nextwave_test5",
        "scripts/research/operational_nextwave_ranker.py",
    )
    cloud = {
        "lat": np.array([0.0, 5.0], dtype=np.float32),
        "lon": np.array([0.0, 5.0], dtype=np.float32),
        "coupling": np.array([0.9, 0.1], dtype=np.float32),
        "std": np.array([0.2, 0.4], dtype=np.float32),
        "slip_def": np.array([3.0, 0.0], dtype=np.float32),
        "depth": np.array([25.0, 80.0], dtype=np.float32),
    }

    F, names = mod._coupling_cloud_features_from_points(
        np.array([0.0, 5.0]),
        np.array([0.0, 5.0]),
        cloud,
    )

    assert F.shape[0] == 2
    assert "coupling_nearest_value" in names
    assert "coupling_100km_mean" in names
    assert F[0, names.index("coupling_nearest_value")] > F[1, names.index("coupling_nearest_value")]
    assert F[0, names.index("coupling_nearest_log_slip_def")] > F[1, names.index("coupling_nearest_log_slip_def")]


def test_waveform_noise_features_use_latest_prior_embedding_only():
    mod = _load_script_module(
        "hazardpulse_operational_nextwave_test6",
        "scripts/research/operational_nextwave_ranker.py",
    )
    day = mod.otr.SEC_DAY
    emb = {
        "station": np.array([0, 0], dtype=np.int16),
        "time": np.array([10 * day, 30 * day], dtype=float),
        "lat": np.array([0.0, 0.0], dtype=np.float32),
        "lon": np.array([0.0, 0.0], dtype=np.float32),
        "values": np.array(
            [
                [1.0, 2.0, 0, 0, 0, 0, 3.0, 4.0],
                [10.0, 20.0, 0, 0, 0, 0, 30.0, 40.0],
            ],
            dtype=np.float32,
        ),
    }

    F, names = mod._waveform_noise_features_from_embeddings(
        np.array([20 * day, 40 * day], dtype=float),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        emb,
        max_age_days=100,
    )

    rms_col = names.index("waveform_noise_300km_rms")
    assert np.isclose(F[0, rms_col], 1.0)
    assert np.isclose(F[1, rms_col], 10.0)


def test_insar_aria_coverage_features_use_prior_granules(tmp_path):
    mod = _load_script_module(
        "hazardpulse_operational_nextwave_test7",
        "scripts/research/operational_nextwave_ranker.py",
    )
    path = tmp_path / "aria.csv"
    path.write_text(
        "region,granule_id,producer_granule_id,title,time_start,time_end,updated,"
        "centroid_lat,centroid_lon,granule_size_gb\n"
        "x,g1,p1,t,2000-01-01T00:00:00Z,2000-01-01T01:00:00Z,2000-01-02T00:00:00Z,0,0,2\n"
        "x,g2,p2,t,2000-03-01T00:00:00Z,2000-03-01T01:00:00Z,2000-03-02T00:00:00Z,0,0,5\n",
        encoding="utf-8",
    )
    T = np.array(
        [
            mod.dt.datetime(2000, 2, 1, tzinfo=mod.dt.timezone.utc).timestamp(),
            mod.dt.datetime(2000, 4, 1, tzinfo=mod.dt.timezone.utc).timestamp(),
        ],
        dtype=float,
    )

    F, names = mod.insar_aria_coverage_features(T, np.array([0.0, 0.0]), np.array([0.0, 0.0]), path)

    n_90_col = names.index("insar_aria_coverage_90d_100km_n")
    sum_90_col = names.index("insar_aria_coverage_90d_100km_sum")
    n_long_col = names.index("insar_aria_coverage_1095d_100km_n")
    sum_long_col = names.index("insar_aria_coverage_1095d_100km_sum")
    assert np.isclose(F[0, n_90_col], np.log1p(1))
    assert np.isclose(F[0, sum_90_col], np.log1p(2.0))
    assert np.isclose(F[1, n_long_col], np.log1p(2))
    assert np.isclose(F[1, sum_long_col], np.log1p(7.0))


def test_gnss_field_features_aggregate_directional_slopes():
    mod = _load_script_module(
        "hazardpulse_operational_nextwave_test8",
        "scripts/research/operational_nextwave_ranker.py",
    )
    day = mod.otr.SEC_DAY
    t = np.arange(0, 800, dtype=float) * day
    stations = [
        {
            "time": t,
            "east": t / (365.25 * day) * 0.01,
            "north": np.zeros_like(t),
            "up": np.zeros_like(t),
            "lat": 0.0,
            "lon": 0.0,
        }
    ]

    F, names = mod._gnss_field_features_from_stations(
        np.array([760 * day], dtype=float),
        np.array([0.0]),
        np.array([0.1]),
        stations,
        "gnss_test_field",
    )

    n_col = names.index("gnss_test_field_100km_n")
    speed_col = names.index("gnss_test_field_100km_mean_hspeed_1y_mmyr")
    radial_col = names.index("gnss_test_field_100km_radial_mean_mmyr")
    assert np.isclose(F[0, n_col], np.log1p(1))
    assert F[0, speed_col] > np.log1p(9.0)
    assert F[0, radial_col] > 0.05
