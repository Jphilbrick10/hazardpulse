#!/usr/bin/env python3
"""Operational earthquake ranker with new-data feature families.

This is the "next real levers" experiment:

* Cascadia tremor/slow-slip proxy features from the CRESCENT/PNSN tremor catalog.
* Local GNSS transient features from cached ``.tenv3`` station time series.
* FDSN station-inventory density as a waveform/noise availability proxy.
* Regional low-magnitude catalog features below the global M2.5 input floor.
* Slab2/Coupling Cloud subduction priors, CRESCENT dense GNSS vector fields, and
  ARIA Sentinel-1 GUNW coverage metadata when cached.

All dynamic features are causal: rows at or after the forecast reference timestamp are ignored.
The script appends these features to the current 307-feature causal tabular matrix and evaluates
CatBoost candidates using the same 2018-2019 validation / 2020+ held-out test split.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "research"))

import operational_tabular_ranker as otr  # noqa: E402

DEFAULT_TREMOR_CSV = REPO / ".cache" / "earthquake" / "crescent" / "tremor_events-2010-2025.csv"
DEFAULT_CRESCENT_DIR = REPO / ".cache" / "earthquake" / "crescent"
DEFAULT_GNSS_DIR = REPO / ".cache" / "earthquake" / "gnss"
DEFAULT_STATION_TXT = (
    REPO / ".cache" / "earthquake" / "stations" / "earthscope_selected_stations_2005_2026.txt"
)
DEFAULT_REGIONAL_DIR = REPO / ".cache" / "earthquake" / "regional_catalogs"
DEFAULT_SLAB2_DIR = REPO / ".cache" / "earthquake" / "slab2" / "Slab2Distribute_Mar2018"
DEFAULT_COUPLING_DIR = REPO / ".cache" / "earthquake" / "coupling_cloud" / "extracted"
DEFAULT_WAVEFORM_NOISE_CSV = (
    REPO / ".cache" / "earthquake" / "waveform_noise" / "waveform_noise_embeddings_v2.csv"
)
LEGACY_WAVEFORM_NOISE_CSV = (
    REPO / ".cache" / "earthquake" / "waveform_noise" / "waveform_noise_embeddings_v1.csv"
)
DEFAULT_INSAR_ARIA_CSV = REPO / ".cache" / "earthquake" / "insar" / "aria_gunw_metadata_v1.csv"
SLAB2_POINTS_CACHE = REPO / ".cache" / "earthquake" / "slab2" / "slab2_points_v1.npz"
COUPLING_POINTS_CACHE = REPO / ".cache" / "earthquake" / "coupling_cloud" / "coupling_points_v1.npz"


def _parse_epoch(value):
    text = str(value or "").strip()
    if not text:
        return float("nan")
    text = text.replace("Z", "+00:00")
    for fmt in [None, "%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S"]:
        try:
            if fmt is None:
                parsed = dt.datetime.fromisoformat(text)
            else:
                parsed = dt.datetime.strptime(text, fmt)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=dt.timezone.utc)
            return parsed.timestamp()
        except Exception:
            continue
    return float("nan")


def _decimal_year_to_epoch(year_value):
    year = int(math.floor(float(year_value)))
    frac = float(year_value) - year
    start = dt.datetime(year, 1, 1, tzinfo=dt.timezone.utc)
    end = dt.datetime(year + 1, 1, 1, tzinfo=dt.timezone.utc)
    return start.timestamp() + frac * (end - start).total_seconds()


def _load_tremor(path):
    if not Path(path).exists():
        return None
    lat, lon, t, energy, duration = [], [], [], [], []
    with Path(path).open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                la = float((row.get("lat") or "").strip())
                lo = float((row.get("lon") or "").strip())
                ts = _parse_epoch(row.get("starttime"))
                en = float((row.get("energy") or "0").strip())
                du = float((row.get("duration ") or row.get("duration") or "0").strip())
            except Exception:
                continue
            if all(math.isfinite(v) for v in [la, lo, ts]):
                lat.append(la)
                lon.append(lo)
                t.append(ts)
                energy.append(en if math.isfinite(en) and en > 0 else 0.0)
                duration.append(du if math.isfinite(du) and du > 0 else 0.0)
    if not t:
        return None
    order = np.argsort(t)
    return (
        np.asarray(t, np.float64)[order],
        np.asarray(lat, np.float64)[order],
        np.asarray(lon, np.float64)[order],
        np.asarray(energy, np.float64)[order],
        np.asarray(duration, np.float64)[order],
    )


def _load_regional_catalogs(path):
    path = Path(path)
    if not path.exists():
        return None
    seen = set()
    t, lat, lon, mag = [], [], [], []
    for csv_path in sorted(path.glob("*.csv")):
        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                eid = row.get("id") or row.get("eventid") or ""
                if eid and eid in seen:
                    continue
                try:
                    ts = _parse_epoch(row.get("time"))
                    la = float(row["latitude"])
                    lo = float(row["longitude"])
                    ma = float(row.get("mag") or row.get("magnitude") or "nan")
                except Exception:
                    continue
                if all(math.isfinite(v) for v in [ts, la, lo, ma]):
                    if eid:
                        seen.add(eid)
                    t.append(ts)
                    lat.append(la)
                    lon.append(lo)
                    mag.append(ma)
    if not t:
        return None
    order = np.argsort(t)
    return (
        np.asarray(t, np.float64)[order],
        np.asarray(lat, np.float64)[order],
        np.asarray(lon, np.float64)[order],
        np.asarray(mag, np.float64)[order],
    )


def _load_station_inventory(path):
    path = Path(path)
    if not path.exists():
        return None
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 8:
                continue
            try:
                la = float(parts[2])
                lo = float(parts[3])
                start = _parse_epoch(parts[6])
                end = _parse_epoch(parts[7]) if parts[7] else float("inf")
            except Exception:
                continue
            if all(math.isfinite(v) for v in [la, lo, start]):
                rows.append((la, lo, start, end, parts[0]))
    if not rows:
        return None
    lat = np.asarray([r[0] for r in rows], np.float64)
    lon = np.asarray([r[1] for r in rows], np.float64)
    start = np.asarray([r[2] for r in rows], np.float64)
    end = np.asarray([r[3] for r in rows], np.float64)
    network = np.asarray([r[4] for r in rows], dtype=object)
    return lat, lon, start, end, network


def _load_tenv3_station(path):
    rows = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as fh:
        header = fh.readline()
        if "yyyy.yyyy" not in header or "__east" not in header:
            return None
        for line in fh:
            parts = line.split()
            if len(parts) < 22:
                continue
            try:
                ts = _decimal_year_to_epoch(float(parts[2]))
                east = float(parts[8])
                north = float(parts[10])
                up = float(parts[12])
                la = float(parts[20])
                lo = float(parts[21])
            except Exception:
                continue
            if all(math.isfinite(v) for v in [ts, east, north, up, la, lo]):
                rows.append((ts, east, north, up, la, lo))
    if len(rows) < 30:
        return None
    arr = np.asarray(rows, np.float64)
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    return {
        "time": arr[:, 0],
        "east": arr[:, 1],
        "north": arr[:, 2],
        "up": arr[:, 3],
        "lat": float(np.nanmedian(arr[:, 4])),
        "lon": float(np.nanmedian(arr[:, 5])),
    }


def _load_tenv3_stations(path):
    path = Path(path)
    if not path.exists():
        return []
    stations = []
    for item in sorted(path.glob("*.tenv3")):
        station = _load_tenv3_station(item)
        if station is not None:
            stations.append(station)
    return stations


def _load_crescent_nc_stations(path=DEFAULT_CRESCENT_DIR):
    path = Path(path)
    if not path.exists():
        return []
    try:
        import h5py
    except Exception:
        return []

    stations = []
    seen = set()
    base = dt.datetime(2010, 1, 1, tzinfo=dt.timezone.utc).timestamp()
    # Prefer UNR first, then PANGA, then SOPAC for duplicate station names.
    files = [
        path / "gnss_unr_2010_2025.nc",
        path / "gnss_PANGA_2010_2025.nc",
        path / "gnss_SOPAC_2010_2025.nc",
    ]
    for item in files:
        if not item.exists() or item.stat().st_size == 0:
            continue
        with h5py.File(item, "r") as h:
            days = np.asarray(h["time"][:], np.float64)
            time = base + days * otr.SEC_DAY
            names = [
                x.decode("utf-8", errors="replace").lower() if isinstance(x, bytes) else str(x).lower()
                for x in h["station"][:]
            ]
            lat = np.asarray(h["lat"][:], np.float64)
            lon = np.asarray(h["lon"][:], np.float64)
            east = np.asarray(h["east_m"][:], np.float32)
            north = np.asarray(h["north_m"][:], np.float32)
            up = np.asarray(h["up_m"][:], np.float32)
            for j, station_name in enumerate(names):
                if station_name in seen:
                    continue
                if not (math.isfinite(lat[j]) and math.isfinite(lon[j])):
                    continue
                valid = np.isfinite(east[j]) & np.isfinite(north[j]) & np.isfinite(up[j])
                if valid.sum() < 60:
                    continue
                seen.add(station_name)
                stations.append({
                    "time": time,
                    "east": east[j],
                    "north": north[j],
                    "up": up[j],
                    "lat": float(lat[j]),
                    "lon": float(lon[j]),
                })
    return stations


def _event_cloud_features(T, lat, lon, events, prefix, windows, radii, weights=None, thresholds=None):
    from sklearn.neighbors import BallTree

    n = len(T)
    out = []
    names = []
    if events is None:
        return np.zeros((n, 0), np.float32), []

    ev_t, ev_lat, ev_lon, ev_val = events
    coords_query = np.radians(np.c_[lat, lon])
    for window in windows:
        for radius in radii:
            count_col = np.zeros(n, np.float32)
            sum_col = np.zeros(n, np.float32)
            max_col = np.zeros(n, np.float32)
            for ref in np.unique(T):
                current = np.where(T == ref)[0]
                keep = (ev_t < ref) & (ev_t >= ref - window * otr.SEC_DAY)
                if not keep.any():
                    continue
                sub_val = ev_val[keep]
                tree = BallTree(np.radians(np.c_[ev_lat[keep], ev_lon[keep]]), metric="haversine")
                neigh = tree.query_radius(coords_query[current], r=radius / otr.EARTH_KM)
                count_col[current] = np.fromiter((len(v) for v in neigh), np.float32, len(current))
                if weights == "sum":
                    sum_col[current] = np.fromiter(
                        (sub_val[v].sum() if len(v) else 0.0 for v in neigh),
                        np.float32,
                        len(current),
                    )
                    max_col[current] = np.fromiter(
                        (sub_val[v].max() if len(v) else 0.0 for v in neigh),
                        np.float32,
                        len(current),
                    )
            stem = f"{prefix}_{window}d_{radius}km"
            out.append(np.log1p(count_col)[:, None])
            names.append(f"{stem}_n")
            if weights == "sum":
                out.append(np.log1p(sum_col)[:, None])
                out.append(np.log1p(max_col)[:, None])
                names.extend([f"{stem}_sum", f"{stem}_max"])

            if thresholds:
                for threshold in thresholds:
                    thr_col = np.zeros(n, np.float32)
                    for ref in np.unique(T):
                        current = np.where(T == ref)[0]
                        keep = (
                            (ev_t < ref)
                            & (ev_t >= ref - window * otr.SEC_DAY)
                            & (ev_val >= threshold)
                        )
                        if not keep.any():
                            continue
                        tree = BallTree(
                            np.radians(np.c_[ev_lat[keep], ev_lon[keep]]),
                            metric="haversine",
                        )
                        neigh = tree.query_radius(coords_query[current], r=radius / otr.EARTH_KM)
                        thr_col[current] = np.fromiter(
                            (len(v) for v in neigh),
                            np.float32,
                            len(current),
                        )
                    out.append(np.log1p(thr_col)[:, None])
                    names.append(f"{stem}_ge{threshold:g}_n")
    return np.hstack(out).astype(np.float32), names


def tremor_features(T, lat, lon, tremor_csv=DEFAULT_TREMOR_CSV):
    tremor = _load_tremor(tremor_csv)
    if tremor is None:
        return np.zeros((len(T), 0), np.float32), []
    ev_t, ev_lat, ev_lon, energy, duration = tremor
    F1, n1 = _event_cloud_features(
        T,
        lat,
        lon,
        (ev_t, ev_lat, ev_lon, energy),
        "tremor_energy",
        windows=[30, 90, 365, 1095],
        radii=[100, 200, 300],
        weights="sum",
    )
    F2, n2 = _event_cloud_features(
        T,
        lat,
        lon,
        (ev_t, ev_lat, ev_lon, duration),
        "tremor_duration",
        windows=[30, 365],
        radii=[100, 300],
        weights="sum",
    )
    return np.hstack([F1, F2]).astype(np.float32), n1 + n2


def regional_micro_catalog_features(T, lat, lon, regional_dir=DEFAULT_REGIONAL_DIR):
    events = _load_regional_catalogs(regional_dir)
    if events is None:
        return np.zeros((len(T), 0), np.float32), []
    ev_t, ev_lat, ev_lon, mag = events
    return _event_cloud_features(
        T,
        lat,
        lon,
        (ev_t, ev_lat, ev_lon, mag),
        "regional_micro",
        windows=[30, 90, 365, 1095],
        radii=[50, 100, 200],
        thresholds=[1.0, 2.0, 3.0, 4.0],
    )


def station_inventory_features(T, lat, lon, station_txt=DEFAULT_STATION_TXT):
    from sklearn.neighbors import BallTree

    inv = _load_station_inventory(station_txt)
    if inv is None:
        return np.zeros((len(T), 0), np.float32), []
    slat, slon, start, end, network = inv
    query = np.radians(np.c_[lat, lon])
    out = []
    names = []
    for radius in [100, 300, 500, 1000]:
        count = np.zeros(len(T), np.float32)
        net_count = np.zeros(len(T), np.float32)
        nearest = np.full(len(T), 5000.0, np.float32)
        for ref in np.unique(T):
            current = np.where(T == ref)[0]
            active = (start <= ref) & (end >= ref)
            if not active.any():
                continue
            tree = BallTree(np.radians(np.c_[slat[active], slon[active]]), metric="haversine")
            dist, _ = tree.query(query[current], k=1)
            nearest[current] = np.minimum(nearest[current], dist[:, 0] * otr.EARTH_KM)
            neigh = tree.query_radius(query[current], r=radius / otr.EARTH_KM)
            nets = network[active]
            count[current] = np.fromiter((len(v) for v in neigh), np.float32, len(current))
            net_count[current] = np.fromiter(
                (len(set(nets[v])) if len(v) else 0 for v in neigh),
                np.float32,
                len(current),
            )
        out.append(np.log1p(count)[:, None])
        out.append(np.log1p(net_count)[:, None])
        out.append(np.log1p(nearest)[:, None])
        names.extend([
            f"station_{radius}km_n",
            f"station_{radius}km_network_n",
            f"station_{radius}km_nearest_logdist",
        ])
    return np.hstack(out).astype(np.float32), names


def _load_waveform_noise_embeddings(path=DEFAULT_WAVEFORM_NOISE_CSV):
    path = Path(path)
    if not path.exists():
        if path == DEFAULT_WAVEFORM_NOISE_CSV and LEGACY_WAVEFORM_NOISE_CSV.exists():
            path = LEGACY_WAVEFORM_NOISE_CSV
        else:
            return None
    rows = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                ts = _parse_epoch(row["time"])
                la = float(row["lat"])
                lo = float(row["lon"])
                vals = [
                    float(row["rms"]),
                    float(row["mad"]),
                    float(row["band_005_05"]),
                    float(row["band_05_2"]),
                    float(row["band_2_8"]),
                    float(row["band_8_20"]),
                    float(row["low_high"]),
                    float(row["entropy"]),
                ]
            except Exception:
                continue
            if all(math.isfinite(x) for x in [ts, la, lo, *vals]):
                key = "|".join([row["client"], row["network"], row["station"], row["channel"]])
                rows.append((key, ts, la, lo, vals))
    if not rows:
        return None
    stations = sorted({r[0] for r in rows})
    station_id = {name: i for i, name in enumerate(stations)}
    rows.sort(key=lambda r: r[1])
    return {
        "station": np.asarray([station_id[r[0]] for r in rows], np.int16),
        "time": np.asarray([r[1] for r in rows], np.float64),
        "lat": np.asarray([r[2] for r in rows], np.float32),
        "lon": np.asarray([r[3] for r in rows], np.float32),
        "values": np.asarray([r[4] for r in rows], np.float32),
        "station_names": np.asarray(stations, dtype=object),
    }


def _waveform_noise_features_from_embeddings(T, lat, lon, emb, max_age_days=420):
    from sklearn.neighbors import BallTree

    if emb is None:
        return np.zeros((len(T), 0), np.float32), []
    query = np.radians(np.c_[lat, lon])
    out = []
    names = []
    feature_names = ["rms", "mad", "low_high", "entropy"]
    for radius in [300, 800, 1200]:
        count = np.zeros(len(T), np.float32)
        means = np.zeros((len(T), len(feature_names)), np.float32)
        age = np.zeros(len(T), np.float32)
        for ref in np.unique(T):
            current = np.where(T == ref)[0]
            latest_rows = []
            for station in np.unique(emb["station"]):
                rows = np.where(
                    (emb["station"] == station)
                    & (emb["time"] < ref)
                    & (emb["time"] >= ref - max_age_days * otr.SEC_DAY)
                )[0]
                if len(rows):
                    latest_rows.append(rows[-1])
            if not latest_rows:
                continue
            latest_rows = np.asarray(latest_rows, dtype=int)
            tree = BallTree(
                np.radians(np.c_[emb["lat"][latest_rows], emb["lon"][latest_rows]]),
                metric="haversine",
            )
            neigh = tree.query_radius(query[current], r=radius / otr.EARTH_KM)
            vals = emb["values"][latest_rows]
            times = emb["time"][latest_rows]
            for local_i, ids in enumerate(neigh):
                if len(ids) == 0:
                    continue
                i = current[local_i]
                count[i] = len(ids)
                means[i, 0] = float(np.nanmean(vals[ids, 0]))
                means[i, 1] = float(np.nanmean(vals[ids, 1]))
                means[i, 2] = float(np.nanmean(vals[ids, 6]))
                means[i, 3] = float(np.nanmean(vals[ids, 7]))
                age[i] = float(np.nanmean((ref - times[ids]) / otr.SEC_DAY))
        out.append(np.log1p(count)[:, None])
        for j, fname in enumerate(feature_names):
            out.append(means[:, j:j + 1])
        out.append(np.log1p(age)[:, None])
        names.extend([
            f"waveform_noise_{radius}km_n",
            f"waveform_noise_{radius}km_rms",
            f"waveform_noise_{radius}km_mad",
            f"waveform_noise_{radius}km_low_high",
            f"waveform_noise_{radius}km_entropy",
            f"waveform_noise_{radius}km_age_days",
        ])
    return np.hstack(out).astype(np.float32), names


def waveform_noise_features(T, lat, lon, path=DEFAULT_WAVEFORM_NOISE_CSV):
    return _waveform_noise_features_from_embeddings(T, lat, lon, _load_waveform_noise_embeddings(path))


def _load_insar_aria_metadata(path=DEFAULT_INSAR_ARIA_CSV):
    path = Path(path)
    if not path.exists():
        return None
    t, lat, lon, size = [], [], [], []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                ts = _parse_epoch(row.get("time_start") or row.get("time"))
                la = float(row.get("centroid_lat") or "nan")
                lo = float(row.get("centroid_lon") or "nan")
                gb = float(row.get("granule_size_gb") or "0")
            except Exception:
                continue
            if all(math.isfinite(x) for x in [ts, la, lo]):
                t.append(ts)
                lat.append(la)
                lon.append(lo)
                size.append(gb if math.isfinite(gb) and gb > 0 else 0.0)
    if not t:
        return None
    order = np.argsort(t)
    return (
        np.asarray(t, np.float64)[order],
        np.asarray(lat, np.float64)[order],
        np.asarray(lon, np.float64)[order],
        np.asarray(size, np.float64)[order],
    )


def insar_aria_coverage_features(T, lat, lon, path=DEFAULT_INSAR_ARIA_CSV):
    events = _load_insar_aria_metadata(path)
    if events is None:
        return np.zeros((len(T), 0), np.float32), []
    return _event_cloud_features(
        T,
        lat,
        lon,
        events,
        "insar_aria_coverage",
        windows=[30, 90, 365, 1095],
        radii=[100, 300, 800],
        weights="sum",
    )


def _station_metrics_at_ref(station, ref):
    t = station["time"]
    i_ref = int(np.searchsorted(t, ref, side="left"))
    i_recent0 = int(np.searchsorted(t, ref - 180 * otr.SEC_DAY, side="left"))
    i_base0 = int(np.searchsorted(t, ref - 730 * otr.SEC_DAY, side="left"))
    i_base1 = int(np.searchsorted(t, ref - 180 * otr.SEC_DAY, side="left"))
    i_year0 = int(np.searchsorted(t, ref - 365 * otr.SEC_DAY, side="left"))

    def valid_slice(a, b):
        e = station["east"][a:b]
        n = station["north"][a:b]
        u = station["up"][a:b]
        keep = np.isfinite(e) & np.isfinite(n) & np.isfinite(u)
        return e[keep], n[keep], u[keep], t[a:b][keep]

    e_recent, n_recent, u_recent, _ = valid_slice(i_recent0, i_ref)
    e_base, n_base, u_base, _ = valid_slice(i_base0, i_base1)
    if len(e_recent) < 10 or len(e_base) < 20:
        return None
    e_recent_med = np.nanmedian(e_recent)
    n_recent_med = np.nanmedian(n_recent)
    u_recent_med = np.nanmedian(u_recent)
    e_base_med = np.nanmedian(e_base)
    n_base_med = np.nanmedian(n_base)
    u_base_med = np.nanmedian(u_base)
    horiz_anom = math.hypot(e_recent_med - e_base_med, n_recent_med - n_base_med) * 1000.0
    up_anom = abs(u_recent_med - u_base_med) * 1000.0
    horiz_slope = 0.0
    e_year, n_year, _, t_year = valid_slice(i_year0, i_ref)
    if len(e_year) >= 20:
        x = (t_year - t_year.mean()) / (365.25 * otr.SEC_DAY)
        e_slope = np.polyfit(x, e_year, 1)[0]
        n_slope = np.polyfit(x, n_year, 1)[0]
        horiz_slope = math.hypot(e_slope, n_slope) * 1000.0
    return horiz_anom, up_anom, horiz_slope


def _linear_slope_mm_per_year(values, times, min_points=20):
    values = np.asarray(values, np.float64)
    times = np.asarray(times, np.float64)
    keep = np.isfinite(values) & np.isfinite(times)
    if keep.sum() < min_points:
        return float("nan")
    yy = values[keep]
    xx = (times[keep] - times[keep].mean()) / (365.25 * otr.SEC_DAY)
    if np.nanmax(xx) - np.nanmin(xx) < 0.05:
        return float("nan")
    return float(np.polyfit(xx, yy, 1)[0] * 1000.0)


def _station_vector_metrics_at_ref(station, ref):
    t = station["time"]
    i_ref = int(np.searchsorted(t, ref, side="left"))
    if i_ref <= 0:
        return None

    def valid_slice(a, b):
        e = station["east"][a:b]
        n = station["north"][a:b]
        u = station["up"][a:b]
        tt = t[a:b]
        keep = np.isfinite(e) & np.isfinite(n) & np.isfinite(u) & np.isfinite(tt)
        return e[keep], n[keep], u[keep], tt[keep]

    i_1y = int(np.searchsorted(t, ref - 365 * otr.SEC_DAY, side="left"))
    i_2y = int(np.searchsorted(t, ref - 730 * otr.SEC_DAY, side="left"))
    e1, n1, u1, t1 = valid_slice(i_1y, i_ref)
    e2, n2, u2, t2 = valid_slice(i_2y, i_ref)
    if len(e1) < 20:
        return None

    e1_slope = _linear_slope_mm_per_year(e1, t1)
    n1_slope = _linear_slope_mm_per_year(n1, t1)
    u1_slope = _linear_slope_mm_per_year(u1, t1)
    e2_slope = _linear_slope_mm_per_year(e2, t2)
    n2_slope = _linear_slope_mm_per_year(n2, t2)
    u2_slope = _linear_slope_mm_per_year(u2, t2)
    if not (math.isfinite(e1_slope) and math.isfinite(n1_slope)):
        return None

    i_recent0 = int(np.searchsorted(t, ref - 180 * otr.SEC_DAY, side="left"))
    i_base0 = int(np.searchsorted(t, ref - 730 * otr.SEC_DAY, side="left"))
    i_base1 = int(np.searchsorted(t, ref - 180 * otr.SEC_DAY, side="left"))
    e_recent, n_recent, u_recent, _ = valid_slice(i_recent0, i_ref)
    e_base, n_base, u_base, _ = valid_slice(i_base0, i_base1)
    horiz_anom = 0.0
    up_anom = 0.0
    if len(e_recent) >= 10 and len(e_base) >= 20:
        horiz_anom = math.hypot(
            np.nanmedian(e_recent) - np.nanmedian(e_base),
            np.nanmedian(n_recent) - np.nanmedian(n_base),
        ) * 1000.0
        up_anom = abs(np.nanmedian(u_recent) - np.nanmedian(u_base)) * 1000.0

    fit_e = e1_slope / 1000.0 * ((t1 - t1.mean()) / (365.25 * otr.SEC_DAY)) + np.nanmedian(e1)
    fit_n = n1_slope / 1000.0 * ((t1 - t1.mean()) / (365.25 * otr.SEC_DAY)) + np.nanmedian(n1)
    residual_mad = float(
        np.nanmedian(np.hypot((e1 - fit_e) * 1000.0, (n1 - fit_n) * 1000.0))
    )

    vals = [
        e1_slope,
        n1_slope,
        u1_slope if math.isfinite(u1_slope) else 0.0,
        e2_slope if math.isfinite(e2_slope) else e1_slope,
        n2_slope if math.isfinite(n2_slope) else n1_slope,
        u2_slope if math.isfinite(u2_slope) else 0.0,
        horiz_anom,
        up_anom,
        residual_mad if math.isfinite(residual_mad) else 0.0,
    ]
    return tuple(vals)


def _gnss_field_features_from_stations(T, lat, lon, stations, prefix):
    from sklearn.neighbors import BallTree

    if not stations:
        return np.zeros((len(T), 0), np.float32), []
    slat = np.asarray([s["lat"] for s in stations], np.float64)
    slon = np.asarray([s["lon"] for s in stations], np.float64)
    tree = BallTree(np.radians(np.c_[slat, slon]), metric="haversine")
    query = np.radians(np.c_[lat, lon])
    ref_metrics = {}
    for ref in np.unique(T):
        metrics = np.full((len(stations), 9), np.nan, np.float32)
        for j, station in enumerate(stations):
            row = _station_vector_metrics_at_ref(station, ref)
            if row is not None:
                metrics[j] = row
        ref_metrics[ref] = metrics

    out = []
    names = []
    for radius in [100, 300, 500, 800]:
        cols = {key: np.zeros(len(T), np.float32) for key in [
            "n",
            "mean_hspeed_1y",
            "max_hspeed_1y",
            "mean_hspeed_2y",
            "radial_mean",
            "radial_abs_mean",
            "tangent_abs_mean",
            "coherence",
            "mean_up_abs",
            "mean_recent_hanom",
            "mean_residual_mad",
        ]}
        for ref in np.unique(T):
            current = np.where(T == ref)[0]
            metrics = ref_metrics[ref]
            neigh = tree.query_radius(query[current], r=radius / otr.EARTH_KM)
            for local_i, station_idx in enumerate(neigh):
                vals = metrics[station_idx]
                keep = np.isfinite(vals[:, 0]) & np.isfinite(vals[:, 1])
                if not keep.any():
                    continue
                ids = station_idx[keep]
                vals = vals[keep]
                i = current[local_i]
                speed1 = np.hypot(vals[:, 0], vals[:, 1])
                speed2 = np.hypot(vals[:, 3], vals[:, 4])
                dlat = np.radians(lat[i] - slat[ids])
                dlon = np.radians(lon[i] - slon[ids]) * np.cos(np.radians(lat[i]))
                norm = np.hypot(dlon, dlat)
                norm[norm == 0] = 1.0
                ux = dlon / norm
                uy = dlat / norm
                radial = vals[:, 0] * ux + vals[:, 1] * uy
                tangent = -vals[:, 0] * uy + vals[:, 1] * ux
                cols["n"][i] = len(vals)
                cols["mean_hspeed_1y"][i] = float(np.nanmean(speed1))
                cols["max_hspeed_1y"][i] = float(np.nanmax(speed1))
                cols["mean_hspeed_2y"][i] = float(np.nanmean(speed2))
                cols["radial_mean"][i] = float(np.nanmean(radial))
                cols["radial_abs_mean"][i] = float(np.nanmean(np.abs(radial)))
                cols["tangent_abs_mean"][i] = float(np.nanmean(np.abs(tangent)))
                cols["coherence"][i] = float(
                    np.hypot(np.nansum(vals[:, 0]), np.nansum(vals[:, 1]))
                    / (np.nansum(speed1) + 1e-6)
                )
                cols["mean_up_abs"][i] = float(np.nanmean(np.abs(vals[:, 2])))
                cols["mean_recent_hanom"][i] = float(np.nanmean(vals[:, 6]))
                cols["mean_residual_mad"][i] = float(np.nanmean(vals[:, 8]))

        out.extend([
            np.log1p(cols["n"])[:, None],
            np.log1p(cols["mean_hspeed_1y"])[:, None],
            np.log1p(cols["max_hspeed_1y"])[:, None],
            np.log1p(cols["mean_hspeed_2y"])[:, None],
            np.clip(cols["radial_mean"], -200.0, 200.0)[:, None] / 100.0,
            np.log1p(cols["radial_abs_mean"])[:, None],
            np.log1p(cols["tangent_abs_mean"])[:, None],
            np.clip(cols["coherence"], 0.0, 1.0)[:, None],
            np.log1p(cols["mean_up_abs"])[:, None],
            np.log1p(cols["mean_recent_hanom"])[:, None],
            np.log1p(cols["mean_residual_mad"])[:, None],
        ])
        names.extend([
            f"{prefix}_{radius}km_n",
            f"{prefix}_{radius}km_mean_hspeed_1y_mmyr",
            f"{prefix}_{radius}km_max_hspeed_1y_mmyr",
            f"{prefix}_{radius}km_mean_hspeed_2y_mmyr",
            f"{prefix}_{radius}km_radial_mean_mmyr",
            f"{prefix}_{radius}km_radial_abs_mean_mmyr",
            f"{prefix}_{radius}km_tangent_abs_mean_mmyr",
            f"{prefix}_{radius}km_vector_coherence",
            f"{prefix}_{radius}km_mean_up_abs_mmyr",
            f"{prefix}_{radius}km_mean_recent_hanom_mm",
            f"{prefix}_{radius}km_mean_residual_mad_mm",
        ])
    return np.hstack(out).astype(np.float32), names


def gnss_tenv3_features(T, lat, lon, gnss_dir=DEFAULT_GNSS_DIR):
    from sklearn.neighbors import BallTree

    stations = _load_tenv3_stations(gnss_dir)
    if not stations:
        return np.zeros((len(T), 0), np.float32), []
    slat = np.asarray([s["lat"] for s in stations], np.float64)
    slon = np.asarray([s["lon"] for s in stations], np.float64)
    tree = BallTree(np.radians(np.c_[slat, slon]), metric="haversine")
    query = np.radians(np.c_[lat, lon])
    out = []
    names = []
    for radius in [100, 300, 500]:
        n_col = np.zeros(len(T), np.float32)
        mean_h = np.zeros(len(T), np.float32)
        max_h = np.zeros(len(T), np.float32)
        mean_u = np.zeros(len(T), np.float32)
        max_slope = np.zeros(len(T), np.float32)
        for ref in np.unique(T):
            current = np.where(T == ref)[0]
            metrics = np.full((len(stations), 3), np.nan, np.float32)
            for j, station in enumerate(stations):
                row = _station_metrics_at_ref(station, ref)
                if row is not None:
                    metrics[j] = row
            neigh = tree.query_radius(query[current], r=radius / otr.EARTH_KM)
            for local_i, station_idx in enumerate(neigh):
                vals = metrics[station_idx]
                vals = vals[np.isfinite(vals[:, 0])]
                i = current[local_i]
                if len(vals) == 0:
                    continue
                n_col[i] = len(vals)
                mean_h[i] = float(np.nanmean(vals[:, 0]))
                max_h[i] = float(np.nanmax(vals[:, 0]))
                mean_u[i] = float(np.nanmean(vals[:, 1]))
                max_slope[i] = float(np.nanmax(vals[:, 2]))
        out.extend([
            np.log1p(n_col)[:, None],
            np.log1p(mean_h)[:, None],
            np.log1p(max_h)[:, None],
            np.log1p(mean_u)[:, None],
            np.log1p(max_slope)[:, None],
        ])
        names.extend([
            f"gnss_tenv3_{radius}km_n",
            f"gnss_tenv3_{radius}km_mean_horiz_anom_mm",
            f"gnss_tenv3_{radius}km_max_horiz_anom_mm",
            f"gnss_tenv3_{radius}km_mean_up_anom_mm",
            f"gnss_tenv3_{radius}km_max_horiz_slope_mmyr",
        ])
    return np.hstack(out).astype(np.float32), names


def _gnss_station_collection_features(T, lat, lon, stations, prefix):
    from sklearn.neighbors import BallTree

    if not stations:
        return np.zeros((len(T), 0), np.float32), []
    slat = np.asarray([s["lat"] for s in stations], np.float64)
    slon = np.asarray([s["lon"] for s in stations], np.float64)
    tree = BallTree(np.radians(np.c_[slat, slon]), metric="haversine")
    query = np.radians(np.c_[lat, lon])
    ref_metrics = {}
    for ref in np.unique(T):
        metrics = np.full((len(stations), 3), np.nan, np.float32)
        for j, station in enumerate(stations):
            row = _station_metrics_at_ref(station, ref)
            if row is not None:
                metrics[j] = row
        ref_metrics[ref] = metrics

    out = []
    names = []
    for radius in [100, 300, 500]:
        n_col = np.zeros(len(T), np.float32)
        mean_h = np.zeros(len(T), np.float32)
        max_h = np.zeros(len(T), np.float32)
        mean_u = np.zeros(len(T), np.float32)
        max_slope = np.zeros(len(T), np.float32)
        for ref in np.unique(T):
            current = np.where(T == ref)[0]
            metrics = ref_metrics[ref]
            neigh = tree.query_radius(query[current], r=radius / otr.EARTH_KM)
            for local_i, station_idx in enumerate(neigh):
                vals = metrics[station_idx]
                vals = vals[np.isfinite(vals[:, 0])]
                i = current[local_i]
                if len(vals) == 0:
                    continue
                n_col[i] = len(vals)
                mean_h[i] = float(np.nanmean(vals[:, 0]))
                max_h[i] = float(np.nanmax(vals[:, 0]))
                mean_u[i] = float(np.nanmean(vals[:, 1]))
                max_slope[i] = float(np.nanmax(vals[:, 2]))
        out.extend([
            np.log1p(n_col)[:, None],
            np.log1p(mean_h)[:, None],
            np.log1p(max_h)[:, None],
            np.log1p(mean_u)[:, None],
            np.log1p(max_slope)[:, None],
        ])
        names.extend([
            f"{prefix}_{radius}km_n",
            f"{prefix}_{radius}km_mean_horiz_anom_mm",
            f"{prefix}_{radius}km_max_horiz_anom_mm",
            f"{prefix}_{radius}km_mean_up_anom_mm",
            f"{prefix}_{radius}km_max_horiz_slope_mmyr",
        ])
    return np.hstack(out).astype(np.float32), names


def gnss_crescent_nc_features(T, lat, lon, crescent_dir=DEFAULT_CRESCENT_DIR):
    return _gnss_station_collection_features(
        T,
        lat,
        lon,
        _load_crescent_nc_stations(crescent_dir),
        "gnss_crescent",
    )


def gnss_crescent_field_features(T, lat, lon, crescent_dir=DEFAULT_CRESCENT_DIR):
    return _gnss_field_features_from_stations(
        T,
        lat,
        lon,
        _load_crescent_nc_stations(crescent_dir),
        "gnss_crescent_field",
    )


def _read_slab2_grid(path):
    import h5py

    with h5py.File(path, "r") as h:
        x = np.asarray(h["x"][:], np.float64)
        y = np.asarray(h["y"][:], np.float64)
        z = np.asarray(h["z"][:], np.float32)
    lon = ((x + 180.0) % 360.0) - 180.0
    lat = y
    return lat, lon, z


def _slab2_region_files(root):
    root = Path(root)
    regions = {}
    for path in root.glob("*.grd"):
        parts = path.name.split("_")
        if len(parts) < 3:
            continue
        region = parts[0]
        kind = parts[2]
        regions.setdefault(region, {})[kind] = path
    return regions


def build_slab2_point_cache(root=DEFAULT_SLAB2_DIR, rebuild=False):
    if SLAB2_POINTS_CACHE.exists() and not rebuild:
        z = np.load(SLAB2_POINTS_CACHE, allow_pickle=True)
        return {key: z[key] for key in z.files}

    root = Path(root)
    if not root.exists():
        return None

    lat_cols = []
    lon_cols = []
    depth_cols = []
    dip_cols = []
    strike_cols = []
    thk_cols = []
    unc_cols = []
    region_cols = []
    region_names = []

    for region_id, (region, files) in enumerate(sorted(_slab2_region_files(root).items())):
        if "dep" not in files:
            continue
        lat_axis, lon_axis, dep = _read_slab2_grid(files["dep"])
        valid = np.isfinite(dep)
        if not valid.any():
            continue
        yy, xx = np.where(valid)
        lat_vals = lat_axis[yy]
        lon_vals = lon_axis[xx]
        depth_vals = np.abs(dep[valid]).astype(np.float32)

        def aligned(kind, default=0.0):
            if kind not in files:
                return np.full(len(depth_vals), default, np.float32)
            _, _, arr = _read_slab2_grid(files[kind])
            if arr.shape != dep.shape:
                return np.full(len(depth_vals), default, np.float32)
            vals = arr[valid].astype(np.float32)
            vals[~np.isfinite(vals)] = default
            return vals

        lat_cols.append(lat_vals.astype(np.float32))
        lon_cols.append(lon_vals.astype(np.float32))
        depth_cols.append(depth_vals)
        dip_cols.append(aligned("dip"))
        strike_cols.append(aligned("str"))
        thk_cols.append(aligned("thk"))
        unc_cols.append(aligned("unc"))
        region_cols.append(np.full(len(depth_vals), region_id, np.int16))
        region_names.append(region)

    if not lat_cols:
        return None

    out = {
        "lat": np.concatenate(lat_cols).astype(np.float32),
        "lon": np.concatenate(lon_cols).astype(np.float32),
        "depth": np.concatenate(depth_cols).astype(np.float32),
        "dip": np.concatenate(dip_cols).astype(np.float32),
        "strike": np.concatenate(strike_cols).astype(np.float32),
        "thickness": np.concatenate(thk_cols).astype(np.float32),
        "uncertainty": np.concatenate(unc_cols).astype(np.float32),
        "region_id": np.concatenate(region_cols).astype(np.int16),
        "region_names": np.asarray(region_names, dtype=object),
    }
    SLAB2_POINTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SLAB2_POINTS_CACHE, **out)
    return out


def _slab2_geometry_from_points(lat, lon, slab):
    from sklearn.neighbors import BallTree

    if slab is None or len(slab.get("lat", [])) == 0:
        return np.zeros((len(lat), 0), np.float32), []

    query = np.radians(np.c_[lat, lon])
    coords = np.radians(np.c_[slab["lat"], slab["lon"]])
    tree = BallTree(coords, metric="haversine")
    k = min(5, len(slab["lat"]))
    dist, idx = tree.query(query, k=k)
    dist_km = dist * otr.EARTH_KM
    nearest = idx[:, 0]
    strike = np.radians(slab["strike"][nearest])
    cols = [
        np.log1p(dist_km[:, 0])[:, None],
        (dist_km[:, 0] < 100).astype(np.float32)[:, None],
        (dist_km[:, 0] < 300).astype(np.float32)[:, None],
        (slab["depth"][nearest] / 700.0)[:, None],
        (slab["dip"][nearest] / 90.0)[:, None],
        np.sin(strike)[:, None],
        np.cos(strike)[:, None],
        np.log1p(np.abs(slab["thickness"][nearest]))[:, None],
        np.log1p(np.abs(slab["uncertainty"][nearest]))[:, None],
        (slab["depth"][idx].mean(axis=1) / 700.0)[:, None],
        (slab["dip"][idx].mean(axis=1) / 90.0)[:, None],
        np.log1p(dist_km[:, -1])[:, None],
    ]
    names = [
        "slab2_nearest_logdist",
        "slab2_nearest_lt100",
        "slab2_nearest_lt300",
        "slab2_nearest_depth",
        "slab2_nearest_dip",
        "slab2_nearest_strike_sin",
        "slab2_nearest_strike_cos",
        "slab2_nearest_log_thickness",
        "slab2_nearest_log_uncertainty",
        "slab2_knn5_depth_mean",
        "slab2_knn5_dip_mean",
        "slab2_knn5_logdist",
    ]

    depth = slab["depth"]
    for label, keep in [
        ("shallow70", depth <= 70.0),
        ("interface20_80", (depth >= 20.0) & (depth <= 80.0)),
        ("deep80", depth > 80.0),
    ]:
        if not keep.any():
            continue
        sub_tree = BallTree(np.radians(np.c_[slab["lat"][keep], slab["lon"][keep]]), metric="haversine")
        sub_dist, sub_idx = sub_tree.query(query, k=1)
        km = sub_dist[:, 0] * otr.EARTH_KM
        sub_depth = depth[keep][sub_idx[:, 0]]
        cols.extend([
            np.log1p(km)[:, None],
            (km < 100).astype(np.float32)[:, None],
            (km < 300).astype(np.float32)[:, None],
            (sub_depth / 700.0)[:, None],
        ])
        names.extend([
            f"slab2_{label}_logdist",
            f"slab2_{label}_lt100",
            f"slab2_{label}_lt300",
            f"slab2_{label}_depth",
        ])
    return np.hstack(cols).astype(np.float32), names


def slab2_geometry_features(lat, lon, slab2_dir=DEFAULT_SLAB2_DIR, rebuild=False):
    return _slab2_geometry_from_points(lat, lon, build_slab2_point_cache(slab2_dir, rebuild=rebuild))


def _is_coupling_var(name, obj):
    lower = name.lower()
    if "std" in lower or "sigma" in lower or lower in {"lat", "lon", "depth"}:
        return False
    text = lower
    try:
        attrs = dict(obj.attrs)
        text += " " + str(attrs).lower()
    except Exception:
        pass
    return "coupling" in text or lower.startswith("ps_") or lower.startswith("ssa_")


def _is_std_var(name):
    lower = name.lower()
    return "std" in lower or "sigma" in lower or "unc" in lower


def build_coupling_cloud_point_cache(root=DEFAULT_COUPLING_DIR, rebuild=False, max_per_model=50000):
    if COUPLING_POINTS_CACHE.exists() and not rebuild:
        z = np.load(COUPLING_POINTS_CACHE, allow_pickle=True)
        return {key: z[key] for key in z.files}
    root = Path(root)
    if not root.exists():
        return None
    try:
        import h5py
    except Exception:
        return None

    lat_cols = []
    lon_cols = []
    coupling_cols = []
    std_cols = []
    slip_cols = []
    depth_cols = []
    model_cols = []
    model_names = []

    for model_id, path in enumerate(sorted(root.rglob("*.nc"))):
        try:
            with h5py.File(path, "r") as h:
                if "lat" not in h or "lon" not in h:
                    continue
                lat_axis = np.asarray(h["lat"][:], np.float32)
                lon_axis = ((np.asarray(h["lon"][:], np.float32) + 180.0) % 360.0) - 180.0
                shape = (len(lat_axis), len(lon_axis))
                coupling_vars = [
                    name
                    for name in h.keys()
                    if getattr(h[name], "shape", None) == shape and _is_coupling_var(name, h[name])
                ]
                if not coupling_vars:
                    continue
                coupling_stack = [np.asarray(h[name][:], np.float32) for name in coupling_vars]
                coupling = np.nanmean(np.stack(coupling_stack), axis=0)
                valid = np.isfinite(coupling)
                if not valid.any():
                    continue

                std_vars = [
                    name
                    for name in h.keys()
                    if getattr(h[name], "shape", None) == shape and _is_std_var(name)
                ]
                std = (
                    np.nanmean(np.stack([np.asarray(h[name][:], np.float32) for name in std_vars]), axis=0)
                    if std_vars
                    else np.full(shape, np.nan, np.float32)
                )
                slip_vars = [
                    name
                    for name in h.keys()
                    if getattr(h[name], "shape", None) == shape and "slip" in name.lower()
                ]
                slip = (
                    np.nanmean(np.stack([np.asarray(h[name][:], np.float32) for name in slip_vars]), axis=0)
                    if slip_vars
                    else np.full(shape, np.nan, np.float32)
                )
                depth = (
                    np.asarray(h["depth"][:], np.float32)
                    if "depth" in h and getattr(h["depth"], "shape", None) == shape
                    else np.full(shape, np.nan, np.float32)
                )
        except Exception:
            continue

        yy, xx = np.where(valid)
        if len(yy) > max_per_model:
            pick = np.linspace(0, len(yy) - 1, max_per_model).astype(int)
            yy = yy[pick]
            xx = xx[pick]
        vals = coupling[yy, xx]
        std_vals = std[yy, xx]
        slip_vals = slip[yy, xx]
        depth_vals = depth[yy, xx]
        std_vals[~np.isfinite(std_vals)] = 0.0
        slip_vals[~np.isfinite(slip_vals)] = 0.0
        depth_vals[~np.isfinite(depth_vals)] = 0.0
        lat_cols.append(lat_axis[yy].astype(np.float32))
        lon_cols.append(lon_axis[xx].astype(np.float32))
        coupling_cols.append(np.clip(vals, -0.5, 1.5).astype(np.float32))
        std_cols.append(np.clip(std_vals, 0.0, 2.0).astype(np.float32))
        slip_cols.append(np.clip(slip_vals, 0.0, 100.0).astype(np.float32))
        depth_cols.append(np.clip(depth_vals, 0.0, 700.0).astype(np.float32))
        model_cols.append(np.full(len(vals), model_id, np.int16))
        model_names.append(str(path.relative_to(root)))

    if not lat_cols:
        return None
    out = {
        "lat": np.concatenate(lat_cols).astype(np.float32),
        "lon": np.concatenate(lon_cols).astype(np.float32),
        "coupling": np.concatenate(coupling_cols).astype(np.float32),
        "std": np.concatenate(std_cols).astype(np.float32),
        "slip_def": np.concatenate(slip_cols).astype(np.float32),
        "depth": np.concatenate(depth_cols).astype(np.float32),
        "model_id": np.concatenate(model_cols).astype(np.int16),
        "model_names": np.asarray(model_names, dtype=object),
    }
    COUPLING_POINTS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(COUPLING_POINTS_CACHE, **out)
    return out


def _coupling_cloud_features_from_points(lat, lon, cloud):
    from sklearn.neighbors import BallTree

    if cloud is None or len(cloud.get("lat", [])) == 0:
        return np.zeros((len(lat), 0), np.float32), []
    query = np.radians(np.c_[lat, lon])
    coords = np.radians(np.c_[cloud["lat"], cloud["lon"]])
    tree = BallTree(coords, metric="haversine")
    k = min(5, len(cloud["lat"]))
    dist, idx = tree.query(query, k=k)
    km = dist * otr.EARTH_KM
    nearest = idx[:, 0]
    coupling_knn = cloud["coupling"][idx]
    cols = [
        np.log1p(km[:, 0])[:, None],
        (km[:, 0] < 100).astype(np.float32)[:, None],
        (km[:, 0] < 300).astype(np.float32)[:, None],
        cloud["coupling"][nearest][:, None],
        cloud["std"][nearest][:, None],
        np.log1p(cloud["slip_def"][nearest])[:, None],
        (cloud["depth"][nearest] / 700.0)[:, None],
        coupling_knn.mean(axis=1)[:, None],
        coupling_knn.max(axis=1)[:, None],
        np.log1p(km[:, -1])[:, None],
    ]
    names = [
        "coupling_nearest_logdist",
        "coupling_nearest_lt100",
        "coupling_nearest_lt300",
        "coupling_nearest_value",
        "coupling_nearest_std",
        "coupling_nearest_log_slip_def",
        "coupling_nearest_depth",
        "coupling_knn5_mean",
        "coupling_knn5_max",
        "coupling_knn5_logdist",
    ]
    for radius in [100, 300, 500]:
        neigh = tree.query_radius(query, r=radius / otr.EARTH_KM)
        count = np.zeros(len(lat), np.float32)
        mean = np.zeros(len(lat), np.float32)
        maxv = np.zeros(len(lat), np.float32)
        slip = np.zeros(len(lat), np.float32)
        depth = np.zeros(len(lat), np.float32)
        for i, ids in enumerate(neigh):
            if len(ids) == 0:
                continue
            vals = cloud["coupling"][ids]
            count[i] = len(ids)
            mean[i] = float(np.nanmean(vals))
            maxv[i] = float(np.nanmax(vals))
            slip[i] = float(np.nanmean(cloud["slip_def"][ids]))
            depth[i] = float(np.nanmean(cloud["depth"][ids]))
        cols.extend([
            np.log1p(count)[:, None],
            mean[:, None],
            maxv[:, None],
            np.log1p(slip)[:, None],
            (depth / 700.0)[:, None],
        ])
        names.extend([
            f"coupling_{radius}km_n",
            f"coupling_{radius}km_mean",
            f"coupling_{radius}km_max",
            f"coupling_{radius}km_log_slip_def_mean",
            f"coupling_{radius}km_depth_mean",
        ])
    return np.hstack(cols).astype(np.float32), names


def coupling_cloud_features(lat, lon, coupling_dir=DEFAULT_COUPLING_DIR, rebuild=False):
    return _coupling_cloud_features_from_points(
        lat,
        lon,
        build_coupling_cloud_point_cache(coupling_dir, rebuild=rebuild),
    )


def build_nextwave_features(
    npz_path=otr.DEFAULT_NPZ,
    rebuild=False,
    include_waveform_noise=False,
    include_heavy_data=False,
):
    if include_heavy_data and include_waveform_noise:
        cache_name = "operational_nextwave_features_v7_heavy_waveform.npz"
    elif include_heavy_data:
        cache_name = "operational_nextwave_features_v6_heavy.npz"
    elif include_waveform_noise:
        cache_name = "operational_nextwave_features_v5_waveform.npz"
    else:
        cache_name = "operational_nextwave_features_v4_subduction.npz"
    cache = REPO / ".cache" / "earthquake" / cache_name
    if cache.exists() and not rebuild:
        z = np.load(cache, allow_pickle=True)
        return z["B"], [str(x) for x in z["names"]]

    z = np.load(npz_path)
    X = z["X"]
    T = z["T"]
    ctx = X[:, -1, 6:20].astype(np.float32)
    lat = ctx[:, 0] * 90.0
    lon = ctx[:, 1] * 180.0

    blocks = []
    names = []
    feature_blocks = [
        tremor_features(T, lat, lon),
        gnss_tenv3_features(T, lat, lon),
        gnss_crescent_nc_features(T, lat, lon),
        station_inventory_features(T, lat, lon),
        regional_micro_catalog_features(T, lat, lon),
        slab2_geometry_features(lat, lon, rebuild=rebuild),
        coupling_cloud_features(lat, lon, rebuild=rebuild),
    ]
    if include_heavy_data:
        feature_blocks.insert(3, gnss_crescent_field_features(T, lat, lon))
        feature_blocks.insert(-2, insar_aria_coverage_features(T, lat, lon))
    if include_waveform_noise:
        feature_blocks.insert(4, waveform_noise_features(T, lat, lon))

    for block, block_names in feature_blocks:
        if block.shape[1]:
            blocks.append(block)
            names.extend(block_names)
    B = np.hstack(blocks).astype(np.float32) if blocks else np.zeros((len(T), 0), np.float32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, B=B, names=np.asarray(names, dtype=object))
    return B, names


def _catboost_candidates():
    from catboost import CatBoostClassifier

    configs = [
        ("nextwave_cat_d5", dict(iterations=850, learning_rate=0.022, depth=5, l2_leaf_reg=7)),
        ("nextwave_cat_d6", dict(iterations=1000, learning_rate=0.018, depth=6, l2_leaf_reg=10)),
        ("nextwave_cat_d7", dict(iterations=1100, learning_rate=0.014, depth=7, l2_leaf_reg=12)),
    ]
    out = []
    for seed, (name, cfg) in enumerate(configs, start=1600):
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", default=str(otr.DEFAULT_NPZ))
    ap.add_argument("--rebuild-nextwave", action="store_true")
    ap.add_argument(
        "--include-waveform-noise",
        action="store_true",
        help="Opt-in raw waveform-noise pilot features; off by default because they hurt validation selection.",
    )
    ap.add_argument(
        "--include-heavy-data",
        action="store_true",
        help="Opt-in dense GNSS vector-field and ARIA InSAR coverage features; measured separately.",
    )
    ap.add_argument("--out", default="results/calibration/earthquake_operational_nextwave_ranker.json")
    args = ap.parse_args(argv)

    npz_path = Path(args.npz)
    z = np.load(npz_path)
    Y = z["Y"].astype(int)
    T = z["T"]
    train = T < otr.VAL0
    val = (T >= otr.VAL0) & (T < otr.TEST0)
    test = T >= otr.TEST0
    A, base_names = otr.build_feature_matrix(
        npz_path,
        label_days=30.0,
        historical_m5_csv=otr.DEFAULT_HISTORICAL_M5_CSV,
        gsrm_principal=otr.DEFAULT_GSRM_PRINCIPAL,
        rebuild=False,
    )
    B, next_names = build_nextwave_features(
        npz_path,
        rebuild=args.rebuild_nextwave,
        include_waveform_noise=args.include_waveform_noise,
        include_heavy_data=args.include_heavy_data,
    )
    Z = np.hstack([A, B]).astype(np.float32)

    rows = []
    best = None
    for name, model in _catboost_candidates():
        model.fit(Z[train], Y[train], eval_set=(Z[val], Y[val]), use_best_model=True)
        val_score = model.predict_proba(Z[val])[:, 1]
        test_score = model.predict_proba(Z[test])[:, 1]
        row = {
            "name": name,
            "val_auc": round(otr._auc(Y[val], val_score), 4),
            "test_auc": round(otr._auc(Y[test], test_score), 4),
            "test_grouped_by_month_auc": round(otr._grouped_auc(Y[test], test_score, T[test]), 4),
        }
        rows.append(row)
        if best is None or row["val_auc"] > best["val_auc"]:
            best = row

    report = {
        "label": "M5.0+/100km/30d nextwave data ranker",
        "selection_rule": "highest validation AUC on 2018-2019; test reported for selected model",
        "base_features": int(A.shape[1]),
        "nextwave_features": int(B.shape[1]),
        "total_features": int(Z.shape[1]),
        "n_train": int(train.sum()),
        "n_val": int(val.sum()),
        "n_test": int(test.sum()),
        "test_pos": int(Y[test].sum()),
        "feature_families": {
            "tremor": int(sum(name.startswith("tremor_") for name in next_names)),
            "gnss_tenv3": int(sum(name.startswith("gnss_tenv3_") for name in next_names)),
            "gnss_crescent": int(
                sum(
                    name.startswith("gnss_crescent_")
                    and not name.startswith("gnss_crescent_field_")
                    for name in next_names
                )
            ),
            "gnss_crescent_field": int(
                sum(name.startswith("gnss_crescent_field_") for name in next_names)
            ),
            "station_inventory": int(sum(name.startswith("station_") for name in next_names)),
            "waveform_noise": int(sum(name.startswith("waveform_noise_") for name in next_names)),
            "regional_micro": int(sum(name.startswith("regional_micro_") for name in next_names)),
            "insar_aria_coverage": int(
                sum(name.startswith("insar_aria_coverage_") for name in next_names)
            ),
            "slab2": int(sum(name.startswith("slab2_") for name in next_names)),
            "coupling_cloud": int(sum(name.startswith("coupling_") for name in next_names)),
        },
        "candidates": rows,
        "selected": best,
    }
    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
