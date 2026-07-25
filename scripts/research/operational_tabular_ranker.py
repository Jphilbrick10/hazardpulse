#!/usr/bin/env python3
"""Causal tabular ranker for the M5+/100km/30d operational earthquake task.

This is the high-signal companion to ``deep_operational_earthquake.py``. It reuses the exact
operational sample cache and adds only causal, forecast-time-available features:

* multi-scale context channels already present in the deep cache;
* sequence summaries over the K prior local events;
* prior, matured M5 outcome rates for the same/nearby cells;
* static geography/plate-boundary distance features;
* optional causal precursor features produced by ``catalog_features_incremental.py``.

Model selection is by 2018-2019 validation AUC. The 2020+ test AUC is reported once for the
validation-selected champion.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SEC_DAY = 86400.0
EARTH_KM = 6371.0
VAL0 = dt.datetime(2018, 1, 1, tzinfo=dt.timezone.utc).timestamp()
TEST0 = dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc).timestamp()
DEFAULT_NPZ = REPO / ".cache" / "earthquake" / (
    "deepop_v3_my2025_m5.0_lr100_ld30_K192_ir100_am8_g2.npz"
)
DEFAULT_PRECURSOR_NPZ = (
    REPO / ".cache" / "earthquake" / "catalog_features_incremental_causal.npz"
)
DEFAULT_HISTORICAL_M5_CSV = (
    REPO / ".cache" / "earthquake" / "usgs_historical_m5_1900_1999.csv"
)
DEFAULT_GSRM_PRINCIPAL = (
    REPO / ".cache" / "earthquake" / "gsrm" / "principal_strain_rate.dat"
)
DEFAULT_GEM_ACTIVE_FAULTS = (
    REPO / ".cache" / "earthquake" / "gem" / "gem_active_faults_harmonized.geojson"
)


def _auc(y, score):
    y = np.asarray(y).astype(int)
    score = np.asarray(score, float)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty(len(score), float)
    ranks[order] = np.arange(1, len(score) + 1)
    n1 = y.sum()
    n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _grouped_auc(y, score, groups):
    vals = []
    weights = []
    for group in np.unique(groups):
        idx = groups == group
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        vals.append(_auc(yy, score[idx]))
        weights.append(int(yy.sum()) * int(len(yy) - yy.sum()))
    return float(np.average(vals, weights=weights)) if vals else float("nan")


def _fill_nan_by_train(features, train_mask):
    out = np.asarray(features, np.float32).copy()
    med = np.nanmedian(out[train_mask], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    for j in range(out.shape[1]):
        col = out[:, j]
        col[~np.isfinite(col)] = med[j]
    return out


def _sequence_summary_features(X, M):
    days = np.expm1(X[:, :, 0].astype(np.float64))
    mag = X[:, :, 1].astype(np.float64)
    dist = X[:, :, 2].astype(np.float64)
    sin_az = X[:, :, 4].astype(np.float64)
    cos_az = X[:, :, 5].astype(np.float64)
    valid = M.astype(bool) & (mag > 0)
    cols = [np.log1p(valid.sum(1))[:, None]]
    names = ["seq_count"]

    for window in [3, 7, 14, 30, 60, 90, 180, 365, 730, 1095, 1825]:
        in_window = valid & (days <= window)
        cols.append(np.log1p(in_window.sum(1))[:, None])
        names.append(f"seq_n_{window}d")
        for threshold in [3.0, 4.0, 4.5, 5.0, 6.0]:
            cols.append(np.log1p((in_window & (mag >= threshold)).sum(1))[:, None])
            names.append(f"seq_n_{window}d_m{threshold:g}")
        energy = np.where(
            in_window,
            np.power(10.0, 0.75 * np.clip(mag - 2.5, 0, 7)),
            0.0,
        )
        cols.append(np.log1p(energy.sum(1))[:, None])
        names.append(f"seq_energy_{window}d")
        max_mag = np.where(in_window, mag, -99).max(1)
        max_mag[max_mag < 0] = 0
        cols.append((max_mag / 9.0)[:, None])
        names.append(f"seq_maxmag_{window}d")
        min_days = np.where(in_window, days, 1e9).min(1)
        min_days[min_days > 1e8] = 3650
        cols.append(np.log1p(min_days)[:, None])
        names.append(f"seq_mindt_{window}d")

    for p in [0.6, 0.8, 1.0, 1.1, 1.3, 1.5]:
        etas_like = np.where(
            valid,
            np.power(10.0, 0.9 * (mag - 2.5)) / np.power(days + 0.01, p),
            0.0,
        )
        cols.append(np.log1p(etas_like.sum(1))[:, None])
        names.append(f"seq_etas_p{p:g}")

    for radius_frac in [0.1, 0.25, 0.5, 0.75, 1.0]:
        in_radius = valid & (dist <= radius_frac)
        cols.append(np.log1p(in_radius.sum(1))[:, None])
        names.append(f"seq_n_r{radius_frac:g}")
        etas_like = np.where(
            in_radius,
            np.power(10.0, 0.9 * (mag - 2.5)) / np.power(days + 0.01, 1.1),
            0.0,
        )
        cols.append(np.log1p(etas_like.sum(1))[:, None])
        names.append(f"seq_etas_r{radius_frac:g}")

    for window in [30, 90, 180, 365]:
        in_window = valid & (days <= window)
        s = np.where(in_window, sin_az, 0).sum(1)
        c = np.where(in_window, cos_az, 0).sum(1)
        denom = np.maximum(in_window.sum(1), 1)
        cols.append((np.sqrt(s * s + c * c) / denom)[:, None])
        names.append(f"seq_az_conc_{window}d")

    return np.hstack(cols).astype(np.float32), names


def _causal_cell_history(Y, T, lat, lon):
    keys = list(zip(np.round(lat, 6), np.round(lon, 6)))
    order = np.argsort(T)
    prev_n = np.zeros(len(Y), np.float32)
    prev_pos = np.zeros(len(Y), np.float32)
    last_pos_days = np.full(len(Y), 3650.0, np.float32)
    stats = {}
    last_positive = {}
    for i in order:
        key = keys[i]
        n, p = stats.get(key, (0, 0))
        prev_n[i] = n
        prev_pos[i] = p
        if key in last_positive:
            last_pos_days[i] = (T[i] - last_positive[key]) / SEC_DAY
        stats[key] = (n + 1, p + int(Y[i]))
        if Y[i]:
            last_positive[key] = T[i]

    cols = [np.log1p(prev_n)[:, None], np.log1p(last_pos_days)[:, None]]
    names = ["cell_prev_n", "cell_last_positive_days"]
    for smooth in [2, 5, 10, 20]:
        cols.append(((prev_pos + 0.5) / (prev_n + smooth))[:, None])
        names.append(f"cell_prev_rate_s{smooth}")
    return np.hstack(cols).astype(np.float32), names


def _causal_neighbor_label_features(Y, T, lat, lon, label_days):
    from sklearn.neighbors import BallTree

    coords = np.radians(np.c_[lat, lon])
    cols = {}
    for years in [0, 2, 5, 10]:
        for radius_km in [100, 200, 300, 500, 800]:
            cols[(years, radius_km, "n")] = np.zeros(len(Y), np.float32)
            cols[(years, radius_km, "p")] = np.zeros(len(Y), np.float32)

    for ref in np.unique(T):
        current = np.where(T == ref)[0]
        matured = T <= ref - label_days * SEC_DAY
        for years in [0, 2, 5, 10]:
            if years:
                previous = matured & (T >= ref - years * 365 * SEC_DAY)
            else:
                previous = matured
            prev_idx = np.where(previous)[0]
            if len(prev_idx) == 0:
                continue
            tree = BallTree(coords[prev_idx], metric="haversine")
            for radius_km in [100, 200, 300, 500, 800]:
                neighbors = tree.query_radius(coords[current], r=radius_km / EARTH_KM)
                counts = np.fromiter((len(v) for v in neighbors), np.float32, len(current))
                positives = np.fromiter(
                    (Y[prev_idx[v]].sum() if len(v) else 0 for v in neighbors),
                    np.float32,
                    len(current),
                )
                cols[(years, radius_km, "n")][current] = counts
                cols[(years, radius_km, "p")][current] = positives

    out = []
    names = []
    for years in [0, 2, 5, 10]:
        for radius_km in [100, 200, 300, 500, 800]:
            n = cols[(years, radius_km, "n")]
            p = cols[(years, radius_km, "p")]
            years_label = "all" if years == 0 else f"{years}y"
            stem = f"neighbor_{years_label}_{radius_km}km"
            out.append(np.log1p(n)[:, None])
            names.append(f"{stem}_n")
            for smooth in [5, 20, 100]:
                out.append(((p + 0.5) / (n + smooth))[:, None])
                names.append(f"{stem}_rate_s{smooth}")
    return np.hstack(out).astype(np.float32), names


def _extract_geojson_coords(geometry):
    coords = []

    def walk(node):
        if not node:
            return
        if isinstance(node[0], (int, float)) and len(node) >= 2:
            coords.append((node[1], node[0]))
            return
        for child in node:
            walk(child)

    walk(geometry.get("coordinates", []))
    return coords


def _static_geo_features(lat, lon):
    from sklearn.neighbors import BallTree

    cols = [
        np.sin(np.radians(lat))[:, None],
        np.cos(np.radians(lat))[:, None],
        np.sin(np.radians(lon))[:, None],
        np.cos(np.radians(lon))[:, None],
        (np.abs(lat) / 90.0)[:, None],
    ]
    names = ["geo_sin_lat", "geo_cos_lat", "geo_sin_lon", "geo_cos_lon", "geo_abs_lat"]

    path = REPO / ".cache" / "earthquake" / "plates" / "pb2002_boundaries.json"
    if not path.exists():
        return np.hstack(cols).astype(np.float32), names

    groups = {"all": [], "subduction": [], "other": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    for feature in data.get("features", []):
        typ = (feature.get("properties", {}).get("Type") or "").lower()
        coords = _extract_geojson_coords(feature.get("geometry", {}))
        groups["all"].extend(coords)
        if "subduction" in typ:
            groups["subduction"].extend(coords)
        else:
            groups["other"].extend(coords)

    query = np.radians(np.c_[lat, lon])
    for name, coords in groups.items():
        if not coords:
            continue
        tree = BallTree(np.radians(np.asarray(coords, float)), metric="haversine")
        dist, _ = tree.query(query, k=1)
        km = dist[:, 0] * EARTH_KM
        cols.append(np.log1p(km)[:, None])
        cols.append((km < 50).astype(np.float32)[:, None])
        cols.append((km < 200).astype(np.float32)[:, None])
        names.extend([f"plate_{name}_logdist", f"plate_{name}_lt50", f"plate_{name}_lt200"])

    return np.hstack(cols).astype(np.float32), names


def _historical_m5_prior_features(lat, lon, csv_path):
    """Leakage-safe long-horizon M5+ spatial prior from pre-sample USGS history.

    The default CSV is 1900-1999 only, so every row predates the 2005+ operational samples. This
    strengthens the spatial background without using future labels from the held-out years.
    """
    from sklearn.neighbors import BallTree

    path = Path(csv_path)
    if not path.exists():
        return np.zeros((len(lat), 0), np.float32), []

    hlat = []
    hlon = []
    hmag = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for row in csv.DictReader(fh):
            try:
                mag = float(row.get("mag") or row.get("magnitude") or "nan")
                la = float(row["latitude"])
                lo = float(row["longitude"])
            except Exception:
                continue
            if mag >= 5.0 and math.isfinite(la) and math.isfinite(lo):
                hlat.append(la)
                hlon.append(lo)
                hmag.append(mag)

    if not hmag:
        return np.zeros((len(lat), 0), np.float32), []

    hlat = np.asarray(hlat, float)
    hlon = np.asarray(hlon, float)
    hmag = np.asarray(hmag, float)
    query = np.radians(np.c_[lat, lon])
    tree = BallTree(np.radians(np.c_[hlat, hlon]), metric="haversine")
    cols = []
    names = []

    for radius_km in [50, 100, 200, 300, 500, 800, 1000, 1500, 2000]:
        neighbors = tree.query_radius(query, r=radius_km / EARTH_KM)
        for threshold in [5.0, 6.0, 7.0, 8.0]:
            counts = np.fromiter(
                ((hmag[idx] >= threshold).sum() if len(idx) else 0 for idx in neighbors),
                np.float32,
                len(neighbors),
            )
            cols.append(np.log1p(counts)[:, None])
            names.append(f"hist_m{threshold:g}_{radius_km}km_n")
        moment = np.fromiter(
            ((10.0 ** (1.5 * hmag[idx] + 4.8)).sum() if len(idx) else 0 for idx in neighbors),
            np.float64,
            len(neighbors),
        )
        cols.append(np.log1p(moment / 1e12)[:, None].astype(np.float32))
        names.append(f"hist_moment_{radius_km}km")

    for k in [1, 3, 5, 10]:
        dist, idx = tree.query(query, k=k)
        cols.append(np.log1p(dist[:, -1] * EARTH_KM)[:, None].astype(np.float32))
        cols.append((hmag[idx].max(1) / 9.0)[:, None].astype(np.float32))
        names.extend([f"hist_knn{k}_logdist", f"hist_knn{k}_maxmag"])

    for threshold in [5.0, 6.0, 7.0, 8.0]:
        keep = hmag >= threshold
        if not keep.any():
            continue
        sub_tree = BallTree(np.radians(np.c_[hlat[keep], hlon[keep]]), metric="haversine")
        dist, _ = sub_tree.query(query, k=1)
        cols.append(np.log1p(dist[:, 0] * EARTH_KM)[:, None].astype(np.float32))
        names.append(f"hist_m{threshold:g}_nearest_logdist")

    return np.hstack(cols).astype(np.float32), names


def _gsrm_principal_strain_features(lat, lon, gsrm_path):
    """Nearest-cell GSRM v1.2 principal strain-rate features.

    Source format is lon, lat, azimuth of largest principal axis, largest principal strain,
    smallest principal strain. Units are nanostrain/year in the original file.
    """
    from sklearn.neighbors import BallTree

    path = Path(gsrm_path)
    if not path.exists():
        return np.zeros((len(lat), 0), np.float32), []

    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                lo, la, az, emax, emin = [float(x) for x in parts[:5]]
            except ValueError:
                continue
            if all(math.isfinite(x) for x in [lo, la, az, emax, emin]):
                rows.append((la, lo, az, emax, emin))
    if not rows:
        return np.zeros((len(lat), 0), np.float32), []

    arr = np.asarray(rows, float)
    tree = BallTree(np.radians(arr[:, :2]), metric="haversine")
    dist, idx = tree.query(np.radians(np.c_[lat, lon]), k=1)
    near = arr[idx[:, 0]]
    emax = near[:, 3]
    emin = near[:, 4]
    az = np.radians(near[:, 2])
    dilatation = emax + emin
    differential = emax - emin
    second_invariant = np.sqrt(0.5 * (emax * emax + emin * emin))
    cols = [
        np.log1p(np.abs(emax))[:, None],
        np.log1p(np.abs(emin))[:, None],
        np.log1p(np.abs(dilatation))[:, None],
        np.log1p(np.abs(differential))[:, None],
        np.log1p(second_invariant)[:, None],
        np.sign(dilatation)[:, None],
        np.sin(az)[:, None],
        np.cos(az)[:, None],
        np.log1p(dist[:, 0] * EARTH_KM)[:, None],
    ]
    names = [
        "gsrm_log_abs_emax",
        "gsrm_log_abs_emin",
        "gsrm_log_abs_dilatation",
        "gsrm_log_differential",
        "gsrm_log_second_invariant",
        "gsrm_dilatation_sign",
        "gsrm_axis_sin",
        "gsrm_axis_cos",
        "gsrm_nearest_logdist",
    ]
    return np.hstack(cols).astype(np.float32), names


def _parse_fault_slip_rate(value):
    if value is None:
        return 0.0
    nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", str(value))
    vals = []
    for num in nums:
        try:
            val = float(num)
        except ValueError:
            continue
        if math.isfinite(val) and val >= 0:
            vals.append(val)
    return float(vals[0]) if vals else 0.0


def _fault_family(slip_type):
    text = (slip_type or "").lower()
    if "reverse" in text or "thrust" in text:
        return "reverse"
    if "normal" in text:
        return "normal"
    if "strike" in text or "dextral" in text or "sinistral" in text:
        return "strike_slip"
    return "other"


def _gem_active_fault_features(lat, lon, faults_path):
    """Static GEM active-fault proximity and nearest slip-rate features."""
    from sklearn.neighbors import BallTree

    path = Path(faults_path)
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((len(lat), 0), np.float32), []

    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    rows = []
    for feature in data.get("features", []):
        props = feature.get("properties", {}) or {}
        slip = _parse_fault_slip_rate(props.get("net_slip_rate"))
        family = _fault_family(props.get("slip_type"))
        for la, lo in _extract_geojson_coords(feature.get("geometry", {}) or {}):
            if math.isfinite(la) and math.isfinite(lo):
                rows.append((la, lo, slip, family))
    if not rows:
        return np.zeros((len(lat), 0), np.float32), []

    query = np.radians(np.c_[lat, lon])
    cols = []
    names = []

    arr = np.asarray([(r[0], r[1], r[2]) for r in rows], dtype=object)
    coords = np.asarray(arr[:, :2], dtype=float)
    slip = np.asarray(arr[:, 2], dtype=float)
    tree = BallTree(np.radians(coords), metric="haversine")
    dist, idx = tree.query(query, k=1)
    km = dist[:, 0] * EARTH_KM
    nearest_slip = slip[idx[:, 0]]
    cols.extend([
        np.log1p(km)[:, None],
        (km < 25).astype(np.float32)[:, None],
        (km < 50).astype(np.float32)[:, None],
        (km < 100).astype(np.float32)[:, None],
        (km < 200).astype(np.float32)[:, None],
        np.log1p(nearest_slip)[:, None],
        (nearest_slip >= 1.0).astype(np.float32)[:, None],
        (nearest_slip >= 5.0).astype(np.float32)[:, None],
    ])
    names.extend([
        "gemfault_all_logdist",
        "gemfault_all_lt25",
        "gemfault_all_lt50",
        "gemfault_all_lt100",
        "gemfault_all_lt200",
        "gemfault_nearest_log_slip_rate",
        "gemfault_nearest_slip_ge1",
        "gemfault_nearest_slip_ge5",
    ])

    families = np.asarray([r[3] for r in rows], dtype=object)
    for family in ["reverse", "normal", "strike_slip", "other"]:
        keep = families == family
        if not keep.any():
            continue
        sub_tree = BallTree(np.radians(coords[keep]), metric="haversine")
        dist, sub_idx = sub_tree.query(query, k=1)
        km = dist[:, 0] * EARTH_KM
        sub_slip = slip[keep][sub_idx[:, 0]]
        cols.extend([
            np.log1p(km)[:, None],
            (km < 50).astype(np.float32)[:, None],
            (km < 200).astype(np.float32)[:, None],
            np.log1p(sub_slip)[:, None],
        ])
        names.extend([
            f"gemfault_{family}_logdist",
            f"gemfault_{family}_lt50",
            f"gemfault_{family}_lt200",
            f"gemfault_{family}_nearest_log_slip_rate",
        ])

    return np.hstack(cols).astype(np.float32), names


def _load_precursor_features(train_mask, n_rows):
    if not DEFAULT_PRECURSOR_NPZ.exists():
        return np.zeros((n_rows, 0), np.float32), []
    raw = np.load(DEFAULT_PRECURSOR_NPZ)["F"]
    if raw.shape[0] != n_rows:
        return np.zeros((n_rows, 0), np.float32), []
    names = ["bval", "amr_ratio", "amr_curv", "nt_k1", "gcmt_coulomb", "tidal_c", "tidal_fn"]
    return _fill_nan_by_train(raw, train_mask), names[: raw.shape[1]]


def build_feature_matrix(
    npz_path,
    label_days,
    historical_m5_csv,
    gsrm_principal,
    gem_active_faults=None,
    rebuild=False,
):
    cache_name = (
        "operational_tabular_ranker_features_v4_gem.npz"
        if gem_active_faults
        else "operational_tabular_ranker_features_v3.npz"
    )
    cache = REPO / ".cache" / "earthquake" / cache_name
    if cache.exists() and not rebuild:
        z = np.load(cache, allow_pickle=True)
        return z["A"], list(z["names"])

    z = np.load(npz_path)
    X = z["X"]
    M = z["M"]
    Y = z["Y"].astype(int)
    T = z["T"]
    ctx = X[:, -1, 6:20].astype(np.float32)
    train = T < VAL0
    lat = ctx[:, 0] * 90.0
    lon = ctx[:, 1] * 180.0

    blocks = [ctx]
    names = [f"context_{i}" for i in range(ctx.shape[1])]
    feature_blocks = [
        _sequence_summary_features(X, M),
        _causal_cell_history(Y, T, lat, lon),
        _causal_neighbor_label_features(Y, T, lat, lon, label_days),
        _static_geo_features(lat, lon),
        _historical_m5_prior_features(lat, lon, historical_m5_csv),
        _gsrm_principal_strain_features(lat, lon, gsrm_principal),
        _load_precursor_features(train, len(Y)),
    ]
    if gem_active_faults:
        # Keep GEM active faults opt-in: the 2026-06-29 audit showed they improve
        # plausibility as a prior but hurt validation-selected broad AUC.
        feature_blocks.insert(-1, _gem_active_fault_features(lat, lon, gem_active_faults))

    for block, block_names in feature_blocks:
        blocks.append(block)
        names.extend(block_names)

    A = np.hstack(blocks).astype(np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, A=A, names=np.asarray(names, dtype=object))
    return A, names


def _catboost_candidates():
    from catboost import CatBoostClassifier

    configs = [
        ("cat_d5", dict(iterations=800, learning_rate=0.025, depth=5, l2_leaf_reg=5)),
        ("cat_d6", dict(iterations=1000, learning_rate=0.020, depth=6, l2_leaf_reg=8)),
        ("cat_d7", dict(iterations=1200, learning_rate=0.015, depth=7, l2_leaf_reg=10)),
    ]
    out = []
    for seed, (name, cfg) in enumerate(configs, start=300):
        out.append((
            name,
            CatBoostClassifier(
                **cfg,
                loss_function="Logloss",
                eval_metric="AUC",
                auto_class_weights="Balanced",
                random_seed=seed,
                verbose=False,
                allow_writing_files=False,
            ),
        ))
    return out


def _fallback_candidates():
    from sklearn.ensemble import HistGradientBoostingClassifier

    return [(
        "sklearn_hgb",
        HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.04,
            max_leaf_nodes=31,
            l2_regularization=0.2,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=0,
        ),
    )]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default=str(DEFAULT_NPZ))
    ap.add_argument("--label-days", type=float, default=30.0)
    ap.add_argument("--historical-m5-csv", default=str(DEFAULT_HISTORICAL_M5_CSV))
    ap.add_argument("--gsrm-principal", default=str(DEFAULT_GSRM_PRINCIPAL))
    ap.add_argument(
        "--gem-active-faults",
        default="",
        help="Optional GEM active-fault GeoJSON; opt-in because it hurt validation-selected AUC.",
    )
    ap.add_argument("--rebuild-features", action="store_true")
    ap.add_argument("--out", default="results/calibration/earthquake_operational_tabular_ranker.json")
    args = ap.parse_args(argv)

    npz_path = Path(args.npz)
    z = np.load(npz_path)
    Y = z["Y"].astype(int)
    T = z["T"]
    train = T < VAL0
    val = (T >= VAL0) & (T < TEST0)
    test = T >= TEST0

    A, names = build_feature_matrix(
        npz_path,
        args.label_days,
        args.historical_m5_csv,
        args.gsrm_principal,
        args.gem_active_faults or None,
        rebuild=args.rebuild_features,
    )
    try:
        candidates = _catboost_candidates()
        model_family = "catboost"
    except Exception:
        candidates = _fallback_candidates()
        model_family = "sklearn_fallback"

    results = []
    best = None
    for name, model in candidates:
        if model_family == "catboost":
            model.fit(A[train], Y[train], eval_set=(A[val], Y[val]), use_best_model=True)
        else:
            model.fit(A[train], Y[train])
        val_score = model.predict_proba(A[val])[:, 1]
        test_score = model.predict_proba(A[test])[:, 1]
        row = {
            "name": name,
            "val_auc": round(_auc(Y[val], val_score), 4),
            "test_auc": round(_auc(Y[test], test_score), 4),
            "test_grouped_by_month_auc": round(_grouped_auc(Y[test], test_score, T[test]), 4),
        }
        results.append(row)
        if best is None or row["val_auc"] > best["val_auc"]:
            best = row

    report = {
        "label": f"M5.0+/100km/{args.label_days:.0f}d",
        "model_family": model_family,
        "selection_rule": "highest validation AUC on 2018-2019; test reported for selected model",
        "n_features": int(A.shape[1]),
        "n_train": int(train.sum()),
        "n_val": int(val.sum()),
        "n_test": int(test.sum()),
        "test_pos": int(Y[test].sum()),
        "test_pos_rate": round(float(Y[test].mean()), 4),
        "candidates": results,
        "selected": best,
        "feature_count_by_family": {
            "context": 14,
            "total": int(len(names)),
        },
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
