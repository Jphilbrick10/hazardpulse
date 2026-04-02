"""Shared helpers for prospective earthquake forecast logging and scoring."""

from __future__ import annotations

import csv
import datetime as dt
import io
from typing import Iterator
from urllib.parse import urlencode

from hazardpulse.data.http import fetch_text


USGS_API = "https://earthquake.usgs.gov/fdsnws/event/1/query"


def parse_utc_datetime(text: str) -> dt.datetime:
    """Parse an ISO-8601 timestamp and return a UTC-aware datetime."""
    parsed = dt.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def format_utc_z(value: dt.datetime) -> str:
    """Format a datetime as UTC with a trailing Z."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def forecast_id_for_time(
    when: dt.datetime,
    *,
    prefix: str = "eq_fcst",
) -> str:
    """Return a stable forecast identifier for a UTC issue time."""
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    when = when.astimezone(dt.timezone.utc)
    return f"{prefix}_{when.strftime('%Y%m%d_%H00')}"


def iter_monthly_windows(
    start: dt.datetime,
    end: dt.datetime,
) -> Iterator[tuple[dt.datetime, dt.datetime]]:
    """Yield contiguous month-aligned windows spanning [start, end)."""
    if start >= end:
        return

    cursor = start
    while cursor < end:
        if cursor.month == 12:
            next_month = dt.datetime(
                cursor.year + 1,
                1,
                1,
                tzinfo=dt.timezone.utc,
            )
        else:
            next_month = dt.datetime(
                cursor.year,
                cursor.month + 1,
                1,
                tzinfo=dt.timezone.utc,
            )
        yield cursor, min(next_month, end)
        cursor = next_month


def fetch_usgs_catalog_range(
    start: dt.datetime,
    end: dt.datetime,
    *,
    min_magnitude: float = 2.5,
    namespace: str = "usgs_prospective",
    verbose: bool = False,
) -> list[dict]:
    """Fetch a USGS earthquake catalog over a time range.

    The USGS CSV endpoint can truncate large queries, so this helper pulls
    month-by-month and deduplicates by event id.
    """
    if start.tzinfo is None:
        start = start.replace(tzinfo=dt.timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=dt.timezone.utc)
    start = start.astimezone(dt.timezone.utc)
    end = end.astimezone(dt.timezone.utc)

    events_by_id: dict[str, dict] = {}

    for window_start, window_end in iter_monthly_windows(start, end):
        params = urlencode(
            {
                "format": "csv",
                "starttime": format_utc_z(window_start),
                "endtime": format_utc_z(window_end),
                "minmagnitude": f"{min_magnitude:.1f}",
                "orderby": "time-asc",
                "limit": 20000,
            }
        )
        url = f"{USGS_API}?{params}"
        if verbose:
            print(
                "  USGS fetch",
                format_utc_z(window_start),
                "to",
                format_utc_z(window_end),
            )
        raw = fetch_text(
            url,
            namespace=namespace,
            use_cache=False,
            refresh=True,
        )
        reader = csv.DictReader(io.StringIO(raw))
        for row in reader:
            try:
                event_id = row.get("id", "")
                event = {
                    "time": row.get("time", ""),
                    "latitude": float(row["latitude"]),
                    "longitude": float(row["longitude"]),
                    "depth": float(row.get("depth", 0) or 0),
                    "mag": float(row["mag"]),
                    "magType": row.get("magType", ""),
                    "place": row.get("place", ""),
                    "id": event_id,
                }
            except (KeyError, ValueError):
                continue

            key = event_id or (
                f"{event['time']}|{event['latitude']:.4f}|"
                f"{event['longitude']:.4f}|{event['mag']:.2f}"
            )
            events_by_id[key] = event

    return sorted(events_by_id.values(), key=lambda event: event.get("time", ""))
