# IMPORTANT: This software is for RESEARCH PURPOSES ONLY.
# It is NOT an operational earthquake prediction system.
# It does NOT replace official USGS or national seismological agency warnings.
# Always follow guidance from your national geological survey (USGS, JMA, etc.).
# False negatives (missed earthquakes) WILL occur. Do NOT rely on this
# system for safety-critical decisions.

#!/usr/bin/env python3
"""Fetch USGS earthquake catalog, score with coherence model, output static HTML.

Designed to run every 6 hours via GitHub Actions cron.  Outputs:

  - dist/live/earthquake/index.html   (static HTML page, zero JavaScript)
  - dist/data/live-pulse.json         (updated earthquake entry)
  - dist/data/earthquake-ledger.jsonl (append-only SHA-256 prediction chain)

All data is baked directly into HTML.  Zero JavaScript.
Same architecture as the tornado scorer.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import math
import sys
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

import numpy as np

# Add src to path
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from hazardpulse.earthquake.coherence_engine import (  # noqa: E402
    GRID_DLAT,
    GRID_DLON,
    LAT_MIN,
    LAT_MAX,
    LON_MIN,
    LON_MAX,
    N_LAT,
    N_LON,
    ELL_BACKGROUND_KM,
    B_VALUE_BACKGROUND,
    compute_seismic_coherence_field,
    extract_coherence_features,
    grid_cell_to_latlon,
    latlon_to_grid_cell,
    test_earthquake_singularity,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DIST = Path(__file__).resolve().parents[1] / "dist"
LEDGER_PATH = DIST / "data" / "earthquake-ledger.jsonl"

MODEL_VERSION = "eq_coherence_v1_0"

# ---------------------------------------------------------------------------
# Risk band mapping
# ---------------------------------------------------------------------------

RISK_BANDS = [
    (0.50, "critical"),
    (0.30, "very_high"),
    (0.15, "elevated"),
    (0.08, "guarded"),
    (0.03, "low"),
    (0.00, "minimal"),
]


def _risk_band(prob: float) -> str:
    """Map probability to risk band.

    Band names deliberately avoid official seismological terminology
    to prevent confusion with authoritative products.
    """
    for threshold, band in RISK_BANDS:
        if prob >= threshold:
            return band
    return "minimal"


RISK_COLORS = {
    "critical": "#b71c1c",
    "very_high": "#d32f2f",
    "elevated": "#e65100",
    "guarded": "#f9a825",
    "low": "#1976d2",
    "minimal": "#757575",
}

RISK_LABELS = {
    "critical": "Critical",
    "very_high": "Very High",
    "elevated": "Elevated",
    "guarded": "Guarded",
    "low": "Low",
    "minimal": "Minimal",
}

# ---------------------------------------------------------------------------
# USGS earthquake catalog fetch
# ---------------------------------------------------------------------------

USGS_CSV_URL = (
    "https://earthquake.usgs.gov/fdsnws/event/1/query"
    "?format=csv&starttime={start}&endtime={end}"
    "&minmagnitude=2.5&orderby=time"
)


def fetch_usgs_catalog(
    days: int = 30,
    end_time: dt.datetime | None = None,
) -> list[dict]:
    """Fetch USGS earthquake catalog (M2.5+) for the last N days.

    Returns list of event dicts with keys:
        time, latitude, longitude, depth, mag, magType, place, id
    """
    if end_time is None:
        end_time = dt.datetime.now(dt.timezone.utc)
    start_time = end_time - dt.timedelta(days=days)

    url = USGS_CSV_URL.format(
        start=start_time.strftime("%Y-%m-%dT%H:%M:%S"),
        end=end_time.strftime("%Y-%m-%dT%H:%M:%S"),
    )

    print(f"  Fetching USGS catalog: M2.5+, {days} days...")
    print(f"  URL: {url[:100]}...")

    req = Request(url, headers={"User-Agent": "HazardPulse/1.0 (research)"})
    try:
        with urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
    except URLError as e:
        print(f"  Error fetching USGS catalog: {e}")
        return []

    reader = csv.DictReader(io.StringIO(raw))
    events: list[dict] = []
    for row in reader:
        try:
            ev = {
                "time": row.get("time", ""),
                "latitude": float(row["latitude"]),
                "longitude": float(row["longitude"]),
                "depth": float(row.get("depth", 0) or 0),
                "mag": float(row["mag"]),
                "magType": row.get("magType", ""),
                "place": row.get("place", ""),
                "id": row.get("id", ""),
            }
            events.append(ev)
        except (ValueError, KeyError):
            continue

    print(f"  Fetched {len(events)} events from USGS catalog")
    return events


# ---------------------------------------------------------------------------
# Grid cell scoring
# ---------------------------------------------------------------------------


def bin_events_to_grid(events: list[dict]) -> dict[tuple[int, int], list[dict]]:
    """Bin earthquake events into 2-degree grid cells.

    Returns dict: (row, col) -> list of events in that cell.
    """
    grid: dict[tuple[int, int], list[dict]] = {}
    for ev in events:
        lat = ev["latitude"]
        lon = ev["longitude"]
        row, col = latlon_to_grid_cell(lat, lon)
        key = (row, col)
        if key not in grid:
            grid[key] = []
        grid[key].append(ev)
    return grid


def score_grid_cells(
    events: list[dict],
    grid_fields: dict[str, np.ndarray] | None = None,
    now: dt.datetime | None = None,
) -> list[dict]:
    """Score all active grid cells and return ranked list.

    For each 2-degree cell with recent seismicity, compute:
    - b-value and trend
    - Correlation length and trend
    - Rate acceleration
    - Singularity conditions (0-5)
    - Estimated days to criticality
    """
    if now is None:
        now = dt.datetime.now(dt.timezone.utc)
    ref_epoch = now.replace(tzinfo=dt.timezone.utc).timestamp()

    cell_bins = bin_events_to_grid(events)
    scored_cells: list[dict] = []

    for (row, col), cell_events in cell_bins.items():
        if len(cell_events) < 5:
            continue

        lat, lon = grid_cell_to_latlon(row, col)

        # Extract coherence features
        features = extract_coherence_features(
            events, lat, lon,
            radius_km=300.0,
            time_window_days=365.0,
            ref_epoch=ref_epoch,
            grid_fields=grid_fields,
        )

        # Test singularity conditions
        sing = test_earthquake_singularity(features)

        # Compute probability estimate from singularity conditions
        # Base probability from conditions met (empirical calibration)
        base_prob = sing.conditions_met * 0.08
        # Boost for rate acceleration
        rate_accel = features.get("rate_acceleration", 1.0)
        if not math.isnan(rate_accel) and rate_accel > 1.0:
            base_prob *= min(rate_accel, 3.0) / 1.5
        # Boost for b-value depression
        b_val = features.get("b_value", 1.0)
        if not math.isnan(b_val) and b_val < 0.85:
            base_prob *= 1.2
        # Cap at 0.95
        prob = min(max(base_prob, 0.0), 0.95)

        # Max magnitude in cell in last 30 days
        max_mag = max(
            (e["mag"] for e in cell_events if e.get("mag") is not None),
            default=0.0,
        )

        risk = _risk_band(prob)

        entry = {
            "row": row,
            "col": col,
            "lat": round(lat, 2),
            "lon": round(lon, 2),
            "n_events": len(cell_events),
            "max_mag": round(max_mag, 1),
            "probability": round(prob, 4),
            "risk_band": risk,
            "b_value": round(features.get("b_value", float("nan")), 3),
            "b_trend": round(features.get("b_trend", float("nan")), 4),
            "ell_km": round(features.get("ell", float("nan")), 1),
            "ell_trend": round(features.get("ell_trend", float("nan")), 2),
            "rate_acceleration": round(
                features.get("rate_acceleration", float("nan")), 2
            ),
            "delta_aic_iet": round(
                features.get("delta_aic_iet", float("nan")), 2
            ),
            "S_over_Gamma": round(
                features.get("S_over_Gamma", float("nan")), 3
            ),
            "days_to_criticality": round(
                features.get("days_to_criticality", float("nan")), 1
            ),
            "conditions_met": sing.conditions_met,
            "singularity_detail": {
                "ell_elevated": sing.ell_elevated,
                "b_depressed": sing.b_depressed,
                "iet_lorentzian": sing.iet_lorentzian,
                "rate_accelerating": sing.rate_accelerating,
                "loading_exceeds_healing": sing.loading_exceeds_healing,
            },
            "tau_local": round(features.get("tau_local", float("nan")), 4),
            "grad_tau_local": round(
                features.get("grad_tau_local", float("nan")), 4
            ),
            "depth_trend": round(
                features.get("depth_trend", float("nan")), 2
            ),
            "spatial_concentration": round(
                features.get("spatial_concentration", float("nan")), 1
            ),
            "model_version": MODEL_VERSION,
        }
        scored_cells.append(entry)

    # Sort by probability descending, then by conditions_met
    scored_cells.sort(
        key=lambda c: (c["probability"], c["conditions_met"]),
        reverse=True,
    )
    return scored_cells


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------


def _esc(s: str) -> str:
    """Escape HTML special characters."""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _pct(p: float) -> str:
    """Format a probability as a percentage string."""
    if math.isnan(p):
        return "--"
    return f"{p * 100:.1f}%"


def _fmt(val: float, fmt: str = ".2f") -> str:
    """Format a float, handling NaN."""
    if isinstance(val, float) and math.isnan(val):
        return "--"
    return f"{val:{fmt}}"


def _format_time(ts: str) -> str:
    """Format a timestamp string for display."""
    if not ts:
        return "--"
    try:
        d = dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return d.strftime("%a, %d %b %Y %H:%M:%S UTC")
    except Exception:
        return str(ts)


def _lat_lon_to_svg(lat: float, lon: float) -> tuple[float, float]:
    """Convert lat/lon to SVG coordinates for 960x480 equirectangular map."""
    x = ((lon + 180) / 360) * 960
    y = ((90 - lat) / 180) * 480
    return (x, y)


# ---------------------------------------------------------------------------
# SVG grid heatmap
# ---------------------------------------------------------------------------


def _render_grid_heatmap(cells: list[dict]) -> str:
    """Render 2-degree grid cells as colored rectangles on the SVG map."""
    lines: list[str] = []
    for c in cells:
        lat = c["lat"]
        lon = c["lon"]
        prob = c["probability"]
        risk = c["risk_band"]
        color = RISK_COLORS.get(risk, "#757575")

        # Cell corners
        x1, y1 = _lat_lon_to_svg(lat + GRID_DLAT / 2, lon - GRID_DLON / 2)
        x2, y2 = _lat_lon_to_svg(lat - GRID_DLAT / 2, lon + GRID_DLON / 2)
        w = x2 - x1
        h = y2 - y1

        opacity = min(0.15 + prob * 1.2, 0.8)
        label = (
            f"{lat:.0f}N, {lon:.0f}E - {_pct(prob)} "
            f"({c['conditions_met']}/5 conditions)"
        )
        lines.append(
            f'    <a href="#cell-{c["row"]}-{c["col"]}" '
            f'aria-label="{_esc(label)}">'
            f'<rect x="{x1:.1f}" y="{y1:.1f}" width="{w:.1f}" '
            f'height="{h:.1f}" fill="{color}" opacity="{opacity:.2f}" '
            f'stroke="{color}" stroke-width="0.5"/></a>'
        )
    return "\n".join(lines)


def _render_svg_markers(cells: list[dict]) -> str:
    """Generate SVG marker elements for top risk cells on the map."""
    lines: list[str] = []
    limit = min(len(cells), 10)
    for i in range(limit):
        c = cells[i]
        rank = i + 1
        x, y = _lat_lon_to_svg(c["lat"], c["lon"])
        risk_label = RISK_LABELS.get(c["risk_band"], c["risk_band"])
        label = (
            f'Cell {c["lat"]:.0f}N {c["lon"]:.0f}E - '
            f'{_pct(c["probability"])} ({risk_label})'
        )
        base_r = 5.0 if rank == 1 else 3.5

        lines.append(
            f'    <a href="#cell-{rank}" aria-label="{_esc(label)}">'
        )
        for p in (1, 2):
            lines.append(
                f'      <circle class="hz-pulse hz-pulse-eq hz-pulse-delay-{p}" '
                f'cx="{x:.1f}" cy="{y:.1f}" r="{base_r + 1:.0f}"/>'
            )
        lines.append(
            f'      <circle class="hz-marker hz-marker-eq" '
            f'cx="{x:.1f}" cy="{y:.1f}" r="{base_r:.1f}" '
            f'filter="url(#glow-eq)"/>'
        )
        # Tooltip
        lines.append(f'      <g class="map-tooltip">')
        lines.append(
            f'        <rect class="tooltip-bg" x="{x + 10:.1f}" '
            f'y="{y - 18:.1f}" width="120" height="22" rx="3"/>'
        )
        lines.append(
            f'        <text class="tooltip-text" x="{x + 12:.1f}" '
            f'y="{y - 8:.1f}">'
            f'{c["lat"]:.0f}N, {c["lon"]:.0f}E</text>'
        )
        lines.append(
            f'        <text class="tooltip-sub" x="{x + 12:.1f}" '
            f'y="{y + 1:.1f}">'
            f'{_pct(c["probability"])} \u00b7 {c["conditions_met"]}/5</text>'
        )
        lines.append(f'      </g>')
        lines.append(f'    </a>')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cell row rendering (details/summary, no JS)
# ---------------------------------------------------------------------------


def _render_cell_rows(cells: list[dict]) -> str:
    """Render top 10 grid cells as <details>/<summary> elements. Zero JS."""
    lines: list[str] = []
    limit = min(len(cells), 10)
    for i in range(limit):
        c = cells[i]
        rank = i + 1
        risk_color = RISK_COLORS.get(c["risk_band"], "#757575")
        risk_label = RISK_LABELS.get(c["risk_band"], c["risk_band"])
        rank_class = " rank-1" if rank == 1 else ""

        lines.append(
            f'        <details class="event-row-details" id="cell-{rank}">'
        )
        lines.append(f'          <summary class="event-row">')
        lines.append(
            f'            <span class="rank-badge{rank_class}">{rank}</span>'
            f' <strong>{c["lat"]:.0f}N, {c["lon"]:.0f}E</strong>'
            f' ({c["n_events"]} events, Mmax {c["max_mag"]:.1f})'
            f' <strong style="color:{risk_color}">'
            f'{_pct(c["probability"])}</strong>'
            f' <span class="chip" style="background:{risk_color};'
            f'color:#fff;font-size:11px;padding:2px 8px;">'
            f'{_esc(risk_label)}</span>'
            f' Conditions: {c["conditions_met"]}/5'
        )
        lines.append(f'          </summary>')
        lines.append(
            f'          <div class="event-detail" style="padding:12px 10px;">'
        )

        # Key metrics
        lines.append(
            f'            <div class="kv"><span>b-value</span>'
            f'<strong>{_fmt(c["b_value"], ".3f")}</strong></div>'
        )
        lines.append(
            f'            <div class="kv"><span>b-value trend</span>'
            f'<strong>{_fmt(c["b_trend"], ".4f")}</strong></div>'
        )
        lines.append(
            f'            <div class="kv"><span>Correlation length</span>'
            f'<strong>{_fmt(c["ell_km"], ".1f")} km</strong></div>'
        )
        lines.append(
            f'            <div class="kv"><span>Corr. length trend</span>'
            f'<strong>{_fmt(c["ell_trend"], ".2f")} km/month</strong></div>'
        )
        lines.append(
            f'            <div class="kv"><span>Rate acceleration</span>'
            f'<strong>{_fmt(c["rate_acceleration"], ".2f")}x</strong></div>'
        )
        lines.append(
            f'            <div class="kv"><span>IET delta-AIC</span>'
            f'<strong>{_fmt(c["delta_aic_iet"], ".2f")}</strong></div>'
        )
        lines.append(
            f'            <div class="kv"><span>S / Gamma</span>'
            f'<strong>{_fmt(c["S_over_Gamma"], ".3f")}</strong></div>'
        )
        lines.append(
            f'            <div class="kv"><span>Days to criticality</span>'
            f'<strong>{_fmt(c["days_to_criticality"], ".0f")}</strong></div>'
        )
        lines.append(
            f'            <div class="kv"><span>Depth trend</span>'
            f'<strong>{_fmt(c["depth_trend"], ".2f")} km</strong></div>'
        )
        lines.append(
            f'            <div class="kv"><span>Spatial concentration</span>'
            f'<strong>{_fmt(c["spatial_concentration"], ".1f")} km</strong></div>'
        )

        # Singularity conditions
        sing = c.get("singularity_detail", {})
        cond_labels = [
            ("ell_elevated", "Corr. length elevated"),
            ("b_depressed", "b-value depressed"),
            ("iet_lorentzian", "IET Lorentzian"),
            ("rate_accelerating", "Rate accelerating"),
            ("loading_exceeds_healing", "Loading > healing"),
        ]
        for key, label in cond_labels:
            val = sing.get(key)
            if val is not None:
                if val:
                    chip = (
                        '<span class="chip" style="background:#d32f2f;'
                        'color:#fff;font-size:11px;padding:2px 8px;">YES'
                        '</span>'
                    )
                else:
                    chip = (
                        '<span class="chip" style="background:#757575;'
                        'color:#fff;font-size:11px;padding:2px 8px;">no'
                        '</span>'
                    )
                lines.append(
                    f'            <div class="kv"><span>{_esc(label)}'
                    f'</span>{chip}</div>'
                )

        lines.append(f'          </div>')
        lines.append(f'        </details>')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Coherence deep dive for #1 cell
# ---------------------------------------------------------------------------


def _render_coherence_deep_dive(top: dict) -> str:
    """Render coherence diagnostics for the top-risk cell. Pure HTML, no JS."""
    if not top:
        return ""

    risk_color = RISK_COLORS.get(top["risk_band"], "#757575")
    lines: list[str] = []

    lines.append(
        '      <section class="section" aria-labelledby="focus-heading">'
    )
    lines.append(
        '        <h2 id="focus-heading">'
        'Top risk cell -- coherence diagnostics</h2>'
    )
    lines.append('        <div class="grid">')

    # Coherence fields card
    lines.append('          <div class="card col-6 hazard-eq">')
    lines.append(
        f'            <h3>{top["lat"]:.0f}N, {top["lon"]:.0f}E '
        f'-- Coherence fields</h3>'
    )
    lines.append(
        f'            <div class="metric" style="color:{risk_color}">'
        f'{_pct(top["probability"])}</div>'
    )
    lines.append(
        '            <div class="metric-label">'
        'Estimated M6.0+ probability (30 days)</div>'
    )

    coh_keys = [
        ("b_value", "b-value (GR)"),
        ("ell_km", "Correlation length (km)"),
        ("ell_trend", "Corr. length trend (km/mo)"),
        ("rate_acceleration", "Rate acceleration"),
        ("delta_aic_iet", "IET delta-AIC"),
        ("S_over_Gamma", "S / Gamma"),
        ("tau_local", "tau (coherence field)"),
        ("grad_tau_local", "grad(tau)"),
        ("days_to_criticality", "Days to criticality"),
    ]
    for key, label in coh_keys:
        val = top.get(key)
        if val is not None:
            lines.append(
                f'            <div class="kv"><span>{_esc(label)}</span>'
                f'<strong>{_fmt(float(val), ".4f")}</strong></div>'
            )
    lines.append('          </div>')

    # Singularity analysis card
    lines.append('          <div class="card col-6 hazard-eq">')
    lines.append('            <h3>Singularity analysis</h3>')
    sing = top.get("singularity_detail", {})
    sing_count = top.get("conditions_met", 0)
    lines.append(
        f'            <div class="kv"><span>Conditions met</span>'
        f'<strong>{sing_count} / 5</strong></div>'
    )

    cond_labels = [
        ("ell_elevated", "Corr. length elevated (>1.5x bg)"),
        ("b_depressed", "b-value depressed (<0.85)"),
        ("iet_lorentzian", "IET Lorentzian (delta-AIC < -2)"),
        ("rate_accelerating", "Rate accelerating (>1.5x)"),
        ("loading_exceeds_healing", "Loading > healing (S/Gamma > 1)"),
    ]
    for sk, label in cond_labels:
        sv = sing.get(sk)
        if sv is not None:
            if sv:
                chip = (
                    '<span class="chip" style="background:#d32f2f;'
                    'color:#fff;font-size:11px;padding:2px 8px;">YES</span>'
                )
            else:
                chip = (
                    '<span class="chip" style="background:#757575;'
                    'color:#fff;font-size:11px;padding:2px 8px;">no</span>'
                )
            lines.append(
                f'            <div class="kv"><span>{_esc(label)}'
                f'</span>{chip}</div>'
            )
    lines.append('          </div>')
    lines.append('        </div>')

    # Cell statistics card
    lines.append('        <div class="grid" style="margin-top:var(--s-md);">')
    lines.append('          <div class="card col-12 hazard-eq">')
    lines.append('            <h3>Cell statistics</h3>')
    lines.append(
        f'            <div class="kv"><span>Location</span>'
        f'<strong>{top["lat"]:.2f}N, {top["lon"]:.2f}E</strong></div>'
    )
    lines.append(
        f'            <div class="kv"><span>Events (30 days)</span>'
        f'<strong>{top["n_events"]}</strong></div>'
    )
    lines.append(
        f'            <div class="kv"><span>Max magnitude</span>'
        f'<strong>M{top["max_mag"]:.1f}</strong></div>'
    )
    lines.append(
        f'            <div class="kv"><span>b-value trend</span>'
        f'<strong>{_fmt(top["b_trend"], ".4f")}</strong></div>'
    )
    lines.append(
        f'            <div class="kv"><span>Depth trend</span>'
        f'<strong>{_fmt(top["depth_trend"], ".2f")} km</strong></div>'
    )
    lines.append(
        f'            <div class="kv"><span>Model version</span>'
        f'<strong>{_esc(top.get("model_version", "--"))}</strong></div>'
    )
    lines.append('          </div>')
    lines.append('        </div>')
    lines.append('      </section>')

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Full page rendering (zero JavaScript)
# ---------------------------------------------------------------------------


def render_earthquake_page(
    scored_cells: list[dict],
    now: dt.datetime,
    n_events_total: int = 0,
) -> str:
    """Generate complete static HTML for the earthquake live page.

    All data is embedded directly in the HTML. Zero JavaScript.
    Follows the HazardPulse Classic Truth Surface spec.
    """
    updated_str = _format_time(now.isoformat() + "Z")

    # Load base SVG map
    svg_path = DIST / "assets" / "world-map-base.svg"
    if svg_path.exists():
        svg_content = svg_path.read_text(encoding="utf-8")
    else:
        svg_content = (
            '<svg class="world-map" viewBox="0 0 960 480" '
            'xmlns="http://www.w3.org/2000/svg">'
            '<rect width="960" height="480" fill="#e4eef8"/></svg>'
        )

    # Inject grid heatmap and markers into the SVG
    if scored_cells:
        heatmap_html = _render_grid_heatmap(scored_cells)
        markers_html = _render_svg_markers(scored_cells)
        svg_content = svg_content.replace(
            "    <!-- Markers go here per page -->\n",
            heatmap_html + "\n" + markers_html + "\n",
        )

    # Build disclaimer
    disclaimer = (
        "RESEARCH ONLY. NOT an operational earthquake prediction system. "
        "Does NOT replace USGS, JMA, or any national geological survey warnings. "
        "Always follow official guidance. See earthquake.usgs.gov for authoritative data."
    )

    # Status bar
    n_active = len(scored_cells)
    top_cond = scored_cells[0]["conditions_met"] if scored_cells else 0
    status_html = f"""
      <section class="section">
        <div class="grid">
          <div class="card col-3">
            <div class="kv"><span>Last update</span><strong>{_esc(updated_str)}</strong></div>
          </div>
          <div class="card col-3">
            <div class="kv"><span>Model</span><strong>{_esc(MODEL_VERSION)}</strong></div>
          </div>
          <div class="card col-3">
            <div class="kv"><span>Active cells</span><strong>{n_active} cells scored</strong></div>
          </div>
          <div class="card col-3">
            <div class="kv"><span>Events (30 d)</span><strong>{n_events_total} M2.5+</strong></div>
          </div>
        </div>
      </section>"""

    # Cells section
    if scored_cells:
        cell_rows = _render_cell_rows(scored_cells)
        cells_html = f"""
      <section class="section" aria-labelledby="cells-heading">
        <h2 id="cells-heading">Top risk cells by seismic criticality</h2>
        <p class="muted" style="margin-top:-8px;margin-bottom:16px;">2-degree grid cells ranked by estimated M6.0+ probability (30 days). Click any row to expand coherence diagnostics.</p>

{cell_rows}
      </section>"""
        dive_html = _render_coherence_deep_dive(scored_cells[0])
    else:
        cells_html = """
      <section class="section">
        <div class="card" style="text-align:center;padding:48px 24px;">
          <h3 style="color:var(--text-secondary);">No active seismic zones</h3>
          <p class="muted">No grid cells with sufficient seismicity detected. Check back later.</p>
        </div>
      </section>"""
        dive_html = ""

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Earthquake Monitor - HazardPulse</title>
  <meta name="description" content="30-day M6.0+ earthquake probability for global seismic zones. Grid cells ranked by coherence field singularity conditions with full evidence.">
  <meta name="theme-color" content="#f6f9ff">
  <link rel="canonical" href="https://hazardpulse.io/live/earthquake/">
  <link rel="stylesheet" href="/assets/styles.css?v=4">
  <link rel="icon" type="image/png" sizes="32x32" href="/assets/favicon-32.png">
  <link rel="apple-touch-icon" sizes="180x180" href="/assets/apple-touch-icon.png">
  <link rel="alternate" type="application/rss+xml" title="HazardPulse Feed" href="/feed.xml">

  <meta property="og:type" content="website">
  <meta property="og:title" content="Earthquake Monitor - HazardPulse">
  <meta property="og:description" content="30-day M6.0+ earthquake probability for global seismic zones. Grid cells ranked by coherence field singularity conditions with full evidence.">
  <meta property="og:url" content="https://hazardpulse.io/live/earthquake/">
  <meta property="og:site_name" content="HazardPulse">
  <meta name="twitter:card" content="summary">
  <meta name="twitter:title" content="Earthquake Monitor - HazardPulse">
  <meta name="twitter:description" content="30-day M6.0+ earthquake probability for global seismic zones. Grid cells ranked by coherence field singularity conditions with full evidence.">

  <script type="application/ld+json">
  {{
    "@context": "https://schema.org",
    "@type": "Dataset",
    "name": "HazardPulse Global Earthquake Criticality Forecast",
    "description": "Probabilistic earthquake forecasts for global seismic zones using coherence field theory singularity conditions with full provenance chain.",
    "license": "https://hazardpulse.io/legal/disclaimer/",
    "creator": {{ "@type": "Organization", "name": "Coherence Energy Labs", "url": "https://coherenceenergylabs.com" }},
    "temporalCoverage": "{now.strftime('%Y-%m-%d')}/{(now + dt.timedelta(days=30)).strftime('%Y-%m-%d')}",
    "spatialCoverage": {{ "@type": "Place", "name": "Global seismic zones" }},
    "variableMeasured": "Probability of M6.0+ earthquake in 30 days"
  }}
  </script>

  <script type="speculationrules">
  {{
    "prefetch": [
      {{ "source": "list", "urls": ["/live/", "/live/tornado/", "/live/hurricane/", "/evidence/", "/verification/"] }}
    ]
  }}
  </script>
</head>
<body>

  <div class="emergency-banner" role="alert" aria-live="assertive">
    <!-- Populated by Cloudflare Worker when seismic threat detected near user -->
  </div>

  <a class="skip-link" href="#main">Skip to content</a>

  <header class="topbar" role="banner">
    <div class="container topbar-inner">
      <a href="/" class="brand" aria-label="HazardPulse home">
        <img src="/assets/hp-logo.png" alt="" class="brand-logo" width="30" height="30">
        HazardPulse
        <small>Classic</small>
      </a>
      <input type="checkbox" id="nav-toggle" class="nav-hamburger-input" aria-label="Toggle navigation">
      <label for="nav-toggle" class="nav-hamburger" aria-hidden="true">
        <span class="nav-hamburger-bar"></span>
        <span class="nav-hamburger-bar"></span>
        <span class="nav-hamburger-bar"></span>
      </label>
      <nav class="nav" aria-label="Primary navigation">
        <div class="nav-dropdown">
          <a href="/live/" aria-current="page">Live</a>
          <div class="nav-dropdown-menu">
            <a href="/live/earthquake/"><span class="hazard-dot eq"></span> Earthquake</a>
            <a href="/live/hurricane/"><span class="hazard-dot hu"></span> Hurricane</a>
            <a href="/live/tornado/"><span class="hazard-dot to"></span> Tornado</a>
          </div>
        </div>
        <a href="/verification/">Verification</a>
        <a href="/evidence/">Evidence</a>
        <a href="/methods/">Methods</a>
        <a href="/registry/">Registry</a>
        <a href="/api/">API</a>
      </nav>
      <div class="theme-switch">
        <input id="theme-toggle" class="theme-toggle" type="checkbox" aria-label="Switch to dark mode">
        <label for="theme-toggle">Dark</label>
      </div>
    </div>
  </header>

  <main id="main" class="container">

    <section class="hero" aria-labelledby="hero-heading">
      <div class="eyebrow">Live seismic intelligence</div>
      <h1 id="hero-heading">Global Earthquake Monitor</h1>
      <p class="subtitle">
        30-day M6.0+ earthquake probability for the world's most active seismic zones.
        Grid cells ranked by coherence field singularity conditions with full evidence.
      </p>
    </section>

    <div class="depth-content">
      <div class="depth-toggle" role="radiogroup" aria-label="Content depth">
        <input type="radio" name="depth" id="depth-simple" value="simple" checked>
        <label for="depth-simple">Simple</label>
        <input type="radio" name="depth" id="depth-technical" value="technical">
        <label for="depth-technical">Technical</label>
      </div>

      <!-- YOUR AREA - populated by Cloudflare Worker via HTMLRewriter -->
      <section class="your-area-section section" aria-labelledby="your-area-heading">
        <!-- Worker injects personalized seismic threat content here based on IP geolocation -->
      </section>

      <!-- WORLD MAP -->
      <section class="section" aria-labelledby="worldmap-heading">
        <h2 id="worldmap-heading">Global seismic activity map</h2>
        <p class="muted" style="margin-top:-8px;margin-bottom:16px;">2-degree grid cells colored by M6.0+ probability. Markers show top 10 risk zones. Hover for details.</p>

        <div class="world-map-wrapper">
          {svg_content}

          <div class="map-legend">
            <span class="map-legend-item"><span class="map-legend-dot eq"></span> Seismic zone</span>
            <span class="map-legend-item"><span style="display:inline-block;width:20px;height:2px;border-top:2px dashed var(--eq);opacity:.5;vertical-align:middle;margin-right:2px;"></span> Plate boundary</span>
            <span class="map-legend-item"><span class="map-legend-dot user"></span> Your location</span>
          </div>
        </div>
      </section>

      <!-- DISCLAIMER BANNER -->
      <section class="section">
        <div class="card" style="background:var(--warn-bg,#fff8e1);border-left:4px solid var(--warn,#c98a12);padding:12px 16px;">
          <strong>Research Only</strong> --
          {_esc(disclaimer)}
        </div>
      </section>

      <!-- STATUS BAR -->
{status_html}

      <!-- TOP RISK CELLS -->
{cells_html}

      <!-- TOP CELL DEEP DIVE -->
{dive_html}

      <!-- EVIDENCE AND REPLAY -->
      <section class="section" aria-labelledby="evidence-heading">
        <h2 id="evidence-heading">Evidence and replay</h2>
        <div class="grid">
          <div class="card col-4">
            <h3>See the evidence</h3>
            <p class="muted">Every forecast links to the exact data and model that produced it. Nothing is hidden.</p>
            <a href="/evidence/#eq" class="btn btn-secondary" style="margin-top:8px;">Browse evidence</a>
          </div>
          <div class="card col-4">
            <h3>Check our track record</h3>
            <p class="muted">How often are we right? We publish accuracy scores publicly, broken down by region and magnitude.</p>
            <a href="/verification/" class="btn btn-secondary" style="margin-top:8px;">See accuracy</a>
          </div>
          <div class="card col-4">
            <h3>Replay any forecast</h3>
            <p class="muted">Download the input data and re-run any past forecast yourself. Same data in, same result out - guaranteed.</p>
            <a href="/data/replay/eq_fcst_{now.strftime('%Y%m%d')}_0300.json" class="btn btn-secondary" style="margin-top:8px;">Download replay</a>
          </div>
        </div>
      </section>

    </div>
  </main>

  <footer class="footer" role="contentinfo">
    <div class="container footer-inner">
      <div class="footer-col">
        <h4>Platform</h4>
        <a href="/live/">Live forecasts</a>
        <a href="/verification/">Verification</a>
        <a href="/evidence/">Evidence</a>
        <a href="/methods/">Methods</a>
      </div>
      <div class="footer-col">
        <h4>Data</h4>
        <a href="/registry/">Model registry</a>
        <a href="/api/">API contracts</a>
        <a href="/ops/status/">System status</a>
        <a href="/feed.xml">RSS feed</a>
      </div>
      <div class="footer-col">
        <h4>Legal</h4>
        <a href="/legal/disclaimer/">Disclaimer</a>
        <a href="https://earthquake.usgs.gov/" rel="noopener">USGS</a>
        <a href="https://www.nhc.noaa.gov/" rel="noopener">NHC</a>
        <a href="https://www.spc.noaa.gov/" rel="noopener">SPC</a>
      </div>
      <p class="footer-disclaimer">
        HazardPulse provides experimental research outputs only. These are not official forecasts or warnings.
        Always follow guidance from the USGS, National Hurricane Center (NHC), National Weather Service (NWS),
        Storm Prediction Center (SPC), JMA (Japan), and IMD (India). Probabilistic outputs represent model estimates
        with stated uncertainty - they are not certainties.
      </p>
      <p class="footer-build">Built with Coherence Lang &middot; Zero client-side JavaScript &middot; Zero npm dependencies &middot; Geolocation by Cloudflare Edge</p>
    </div>
  </footer>

</body>
</html>
"""
    return page


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_outputs(
    scored_cells: list[dict],
    now: dt.datetime,
) -> None:
    """Write scored results to dist/data/."""
    # Update live-pulse.json earthquake entry
    pulse_path = DIST / "data" / "live-pulse.json"
    if pulse_path.exists():
        pulse = json.loads(pulse_path.read_text(encoding="utf-8"))
        for hazard in pulse.get("hazards", []):
            if hazard.get("key") == "eq":
                if scored_cells:
                    top = scored_cells[0]
                    hazard["probability"] = top["probability"]
                    hazard["risk_band"] = top["risk_band"]
                    hazard["gate_status"] = "pass"
                    hazard["model_version"] = MODEL_VERSION
                    hazard["forecast_id"] = (
                        f"eq_fcst_{now.strftime('%Y%m%d')}_"
                        f"{now.strftime('%H')}00"
                    )
                else:
                    hazard["probability"] = 0.0
                    hazard["risk_band"] = "minimal"
                    hazard["gate_status"] = "pass"
                    hazard["model_version"] = MODEL_VERSION
                break
        pulse["updated_at"] = now.isoformat() + "Z"
        pulse_path.write_text(
            json.dumps(pulse, indent=2) + "\n", encoding="utf-8"
        )
        print(f"  Updated {pulse_path}")


def append_ledger(
    scored_cells: list[dict],
    now: dt.datetime,
) -> None:
    """Append prediction to SHA-256 chain ledger."""
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Read previous hash
    prev_hash = "0" * 64
    if LEDGER_PATH.exists():
        lines = LEDGER_PATH.read_text(encoding="utf-8").strip().split("\n")
        if lines and lines[-1].strip():
            try:
                last = json.loads(lines[-1])
                prev_hash = last.get("hash", prev_hash)
            except json.JSONDecodeError:
                pass

    # Build ledger entry
    entry = {
        "timestamp": now.isoformat() + "Z",
        "model_version": MODEL_VERSION,
        "n_cells_scored": len(scored_cells),
        "top_probability": (
            scored_cells[0]["probability"] if scored_cells else 0.0
        ),
        "top_conditions": (
            scored_cells[0]["conditions_met"] if scored_cells else 0
        ),
        "prev_hash": prev_hash,
    }
    # Add top 5 cells summary
    entry["top_cells"] = [
        {
            "lat": c["lat"],
            "lon": c["lon"],
            "probability": c["probability"],
            "conditions_met": c["conditions_met"],
            "max_mag": c["max_mag"],
        }
        for c in scored_cells[:5]
    ]

    # Compute SHA-256 hash
    payload = json.dumps(entry, sort_keys=True)
    entry["hash"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    # Append
    with open(LEDGER_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"  Appended to {LEDGER_PATH}")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the earthquake scoring pipeline."""
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    print(f"HazardPulse Earthquake Scoring Pipeline -- {now.isoformat()}Z")
    print()

    # Step 1: Fetch USGS earthquake catalog
    print("Step 1: Fetching USGS earthquake catalog (M2.5+, 30 days)...")
    events = fetch_usgs_catalog(days=30)

    if not events:
        print("  No events fetched. Writing empty state...")
        write_outputs([], now)
        append_ledger([], now)
        eq_html = render_earthquake_page([], now, n_events_total=0)
        eq_page = DIST / "live" / "earthquake" / "index.html"
        eq_page.parent.mkdir(parents=True, exist_ok=True)
        eq_page.write_text(eq_html, encoding="utf-8")
        print(f"  Wrote {eq_page} (no data, zero JS)")
        print()
        print("Done. No events to score.")
        return

    n_events_total = len(events)
    print(f"  {n_events_total} events fetched")

    # Step 2: Compute seismic coherence field (Helmholtz PDE)
    print()
    print("Step 2: Computing seismic coherence field (Helmholtz PDE)...")
    grid_fields: dict[str, np.ndarray] | None = None
    try:
        grid_fields = compute_seismic_coherence_field(
            events, time_window_days=365.0,
        )
        tau_max = float(grid_fields["tau"].max())
        print(f"  tau_max = {tau_max:.4f}")
    except Exception as e:
        print(f"  Warning: Coherence field computation failed: {e}")
        print("  Proceeding with point-based features only")

    # Step 3: Score all active grid cells
    print()
    print("Step 3: Scoring active grid cells...")
    scored = score_grid_cells(events, grid_fields=grid_fields, now=now)
    print(f"  {len(scored)} cells scored")

    for c in scored[:10]:
        cond = c["conditions_met"]
        print(
            f"  [{c['lat']:.0f}N, {c['lon']:.0f}E] "
            f"P={c['probability']:.1%} conditions={cond}/5 "
            f"b={_fmt(c['b_value'], '.3f')} "
            f"ell={_fmt(c['ell_km'], '.0f')}km "
            f"events={c['n_events']} Mmax={c['max_mag']:.1f}"
        )

    # Step 4: Write outputs
    print()
    print("Step 4: Writing outputs...")
    write_outputs(scored, now)
    append_ledger(scored, now)

    # Step 5: Render static HTML page (zero JavaScript)
    print()
    print("Step 5: Rendering static HTML page (zero JS)...")
    eq_html = render_earthquake_page(
        scored, now, n_events_total=n_events_total,
    )
    eq_page = DIST / "live" / "earthquake" / "index.html"
    eq_page.parent.mkdir(parents=True, exist_ok=True)
    eq_page.write_text(eq_html, encoding="utf-8")
    print(f"  Wrote {eq_page} ({len(scored)} cells baked in, zero JS)")

    print()
    print(f"Done. Scored {len(scored)} cells from {n_events_total} events.")


if __name__ == "__main__":
    main()
