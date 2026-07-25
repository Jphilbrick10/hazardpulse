#!/usr/bin/env python3
"""Build a small raw-waveform noise embedding cache for regional earthquake features.

This downloads short continuous waveform snippets from selected Cascadia/California broadband
stations and writes compact spectral/RMS embeddings. It is intentionally modest: enough to test
whether real waveform-derived noise summaries add broad operational signal without starting a
terabyte-scale continuous-waveform project. The default station plan is still bounded, but the
CLI can expand months/stations and resume an existing cache.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from obspy import UTCDateTime
from obspy.clients.fdsn import Client

REPO = Path(__file__).resolve().parents[2]
DEFAULT_OUT_CSV = REPO / ".cache" / "earthquake" / "waveform_noise" / "waveform_noise_embeddings_v2.csv"

STATIONS = [
    ("EARTHSCOPE", "UW", "GNW", "HHZ"),
    ("EARTHSCOPE", "UW", "LON", "HHZ"),
    ("NCEDC", "NC", "KCPB", "HHZ"),
    ("NCEDC", "NC", "JCC", "BHZ"),
    ("EARTHSCOPE", "BK", "CMB", "HHZ"),
    ("EARTHSCOPE", "BK", "WDC", "BHZ"),
    ("SCEDC", "CI", "PASC", "BHZ"),
    ("SCEDC", "CI", "SBC", "BHZ"),
    ("SCEDC", "CI", "RVR", "BHZ"),
    ("EARTHSCOPE", "AK", "COLA", "BHZ"),
    ("EARTHSCOPE", "AK", "PAX", "BHZ"),
    ("EARTHSCOPE", "HV", "HSSD", "BHZ"),
    ("EARTHSCOPE", "PR", "SJG", "BHZ"),
]


def _station_location(client_name, network, station, channel, year):
    client = Client(client_name, timeout=30)
    inv = client.get_stations(
        network=network,
        station=station,
        channel=channel,
        level="channel",
        starttime=UTCDateTime(f"{year}-01-01"),
        endtime=UTCDateTime(f"{year}-12-31"),
    )
    for net in inv:
        for sta in net.stations:
            return float(sta.latitude), float(sta.longitude)
    raise RuntimeError(f"no station location for {network}.{station}.{channel}")


def _band_power(freq, power, lo, hi):
    keep = (freq >= lo) & (freq < hi)
    if not keep.any():
        return 0.0
    return float(np.nanmean(power[keep]))


def _embedding_from_trace(trace):
    data = trace.data.astype(np.float64)
    data = data[np.isfinite(data)]
    if len(data) < 1000:
        return None
    data = data - np.nanmedian(data)
    data = data - np.linspace(data[0], data[-1], len(data))
    rms = float(np.sqrt(np.nanmean(data * data)))
    mad = float(np.nanmedian(np.abs(data - np.nanmedian(data))))
    win = np.hanning(len(data))
    spec = np.fft.rfft(data * win)
    power = (np.abs(spec) ** 2) / max(len(data), 1)
    freq = np.fft.rfftfreq(len(data), d=1.0 / float(trace.stats.sampling_rate))
    bands = [
        _band_power(freq, power, 0.05, 0.5),
        _band_power(freq, power, 0.5, 2.0),
        _band_power(freq, power, 2.0, 8.0),
        _band_power(freq, power, 8.0, min(20.0, 0.45 * float(trace.stats.sampling_rate))),
    ]
    total = sum(bands) + 1e-12
    probs = np.asarray(bands, dtype=np.float64) / total
    entropy = float(-(probs * np.log(probs + 1e-12)).sum())
    low_high = float(math.log1p(bands[0] + bands[1]) - math.log1p(bands[2] + bands[3]))
    return {
        "rms": math.log1p(rms),
        "mad": math.log1p(mad),
        "band_005_05": math.log1p(bands[0]),
        "band_05_2": math.log1p(bands[1]),
        "band_2_8": math.log1p(bands[2]),
        "band_8_20": math.log1p(bands[3]),
        "low_high": low_high,
        "entropy": entropy,
    }


def _fetch_embedding(client_name, network, station, channel, when, minutes):
    client = Client(client_name, timeout=60)
    start = UTCDateTime(when)
    end = start + minutes * 60
    stream = client.get_waveforms(network, station, "*", channel, start, end)
    if len(stream) == 0:
        return None
    stream = stream.merge(method=1, fill_value="interpolate")
    trace = max(stream, key=lambda tr: tr.stats.npts)
    return _embedding_from_trace(trace)


def _parse_station_plan(text):
    if not text:
        return STATIONS
    out = []
    for item in text.split(","):
        parts = [p.strip() for p in item.split(":")]
        if len(parts) != 4:
            raise SystemExit(f"station must be CLIENT:NETWORK:STATION:CHANNEL, got {item!r}")
        out.append(tuple(parts))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start-year", type=int, default=2010)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--months", default="1,7")
    ap.add_argument("--minutes", type=int, default=10)
    ap.add_argument("--stations", default="", help="Comma-separated CLIENT:NET:STA:CHAN override.")
    ap.add_argument("--station-limit", type=int, default=0)
    ap.add_argument("--out", default=str(DEFAULT_OUT_CSV))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    out_csv = Path(args.out)
    months = [int(x) for x in args.months.split(",") if x.strip()]
    station_plan = _parse_station_plan(args.stations)
    if args.station_limit > 0:
        station_plan = station_plan[: args.station_limit]
    existing = set()
    rows = []
    if out_csv.exists() and not args.force:
        with out_csv.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                key = (row["client"], row["network"], row["station"], row["channel"], row["time"])
                existing.add(key)
                rows.append(row)

    locations = {}
    for client_name, network, station, channel in station_plan:
        for year in range(args.start_year, args.end_year + 1):
            for month in months:
                when = f"{year}-{month:02d}-15T00:00:00"
                key = (client_name, network, station, channel, when)
                if key in existing:
                    continue
                try:
                    latlon = locations.get((client_name, network, station, channel))
                    if latlon is None:
                        latlon = _station_location(client_name, network, station, channel, year)
                        locations[(client_name, network, station, channel)] = latlon
                    emb = _fetch_embedding(client_name, network, station, channel, when, args.minutes)
                    if emb is None:
                        continue
                    row = {
                        "client": client_name,
                        "network": network,
                        "station": station,
                        "channel": channel,
                        "time": when,
                        "lat": latlon[0],
                        "lon": latlon[1],
                        **emb,
                    }
                    rows.append(row)
                    print("ok", client_name, network, station, channel, when, flush=True)
                except Exception as exc:
                    print("miss", client_name, network, station, channel, when, repr(exc), flush=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "client",
        "network",
        "station",
        "channel",
        "time",
        "lat",
        "lon",
        "rms",
        "mad",
        "band_005_05",
        "band_05_2",
        "band_2_8",
        "band_8_20",
        "low_high",
        "entropy",
    ]
    with out_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote {out_csv} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
