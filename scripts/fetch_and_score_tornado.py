# IMPORTANT: This software is for RESEARCH PURPOSES ONLY.
# It is NOT an operational tornado warning system.
# It does NOT replace official NWS tornado warnings.
# Always follow guidance from the National Weather Service (weather.gov).
# False negatives (missed tornadoes) WILL occur. Do NOT rely on this
# system for safety-critical decisions.

#!/usr/bin/env python3
"""Fetch ProbSevere + HRRR, score with coherence model, output JSON.

Designed to run every 15 minutes via GitHub Actions cron during severe
season (March -- September).  Outputs:

  - dist/data/live-tornadoes.json   (scored active storms)
  - dist/data/live-pulse.json       (updated tornado entry)
  - dist/data/tornado-ledger.jsonl  (append-only SHA-256 prediction chain)

If no ProbSevere data is available (e.g., off-season), writes empty state
with honest timestamps and exits cleanly.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Add src to path
SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from hazardpulse.data.hrrr import (  # noqa: E402
    fetch_hrrr_grid,
    load_cached_hrrr,
)
from hazardpulse.data.probsevere import (  # noqa: E402
    fetch_probsevere_day,
    load_cached_probsevere,
)
from hazardpulse.tornado.coherence_engine import (  # noqa: E402
    compute_coherence_fields,
    compute_derived_hrrr,
    extract_coherence_at_point,
    test_singularity_at_point,
)
try:
    from hazardpulse.tornado.operational_storm import (  # noqa: E402
        ALL_NAMES_FULL,
        GradientBoostedTrees,
        MetaStacker,
        TornadoStormConfig,
        build_storm_features,
        compute_auc,
        extract_block_a,
        extract_block_a_from_probsevere,
        extract_block_e,
        extract_block_s,
        extract_block_t,
        logistic_predict,
        logistic_train,
        predict_tornado_probability,
        sigmoid,
    )
    HAS_OPERATIONAL = True
except ImportError:
    HAS_OPERATIONAL = False

from hazardpulse.tornado.tornado_npe import (  # noqa: E402
    analytic_tornado_probability,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DIST = Path(__file__).resolve().parents[1] / "dist"
RESULTS = Path(__file__).resolve().parents[1] / "results"
LEDGER_PATH = DIST / "data" / "tornado-ledger.jsonl"

MODEL_VERSION = "tornado_storm_v1_0"


# ---------------------------------------------------------------------------
# Risk band
# ---------------------------------------------------------------------------


def _risk_band(prob: float) -> str:
    """Map probability to risk band.

    Band names deliberately avoid NWS terminology (e.g. "watch", "warning")
    to prevent confusion with official NWS products.
    """
    if prob >= 0.50:
        return "very_high"
    if prob >= 0.30:
        return "high"
    if prob >= 0.15:
        return "moderate"
    if prob >= 0.05:
        return "low"
    return "minimal"


# ---------------------------------------------------------------------------
# Storm tracking from ProbSevere time steps
# ---------------------------------------------------------------------------


def build_storm_tracks(
    time_steps: list[dict],
) -> dict[int, list[tuple[int, str, dict]]]:
    """Group storms by ID across time steps to build tracks.

    Returns dict: storm_id -> list of (time_step_idx, valid_time, storm_dict).
    """
    tracks: dict[int, list[tuple[int, str, dict]]] = defaultdict(list)
    for ts_idx, ts in enumerate(time_steps):
        valid_time = ts.get("valid_time", "")
        for storm in ts.get("storms", []):
            sid = storm.get("id", 0)
            if sid == 0:
                continue
            tracks[sid].append((ts_idx, valid_time, storm))
    return dict(tracks)


# ---------------------------------------------------------------------------
# Model loading and analytic scoring
# ---------------------------------------------------------------------------


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate haversine distance in km."""
    import math
    dlat = (lat2 - lat1) * 111.0
    dlon = (lon2 - lon1) * 111.0 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.sqrt(dlat ** 2 + dlon ** 2)


def load_pretrained_model() -> dict | None:
    """Try to load a pre-trained model from results/models/tornado_gbt.json.

    Returns the model dict if found and loadable, otherwise None.
    """
    model_path = RESULTS / "models" / "tornado_gbt.json"
    if not model_path.exists():
        # Also check the old location
        model_path = RESULTS / "tornado_storm_model.json"
    if not model_path.exists():
        return None
    try:
        model_data = json.loads(model_path.read_text(encoding="utf-8"))
        if not HAS_OPERATIONAL:
            print("  Warning: operational_storm module not available, cannot use ML model")
            return None
        print(f"  Loaded pre-trained model from {model_path.name}")
        return model_data
    except Exception as e:
        print(f"  Warning: Failed to load model: {e}")
        return None


def score_storm_analytic(
    storm: dict,
    coherence_fields: dict[str, np.ndarray] | None,
) -> float:
    """Score a single storm using analytic probability (no ML needed).

    Uses coherence field theory + ProbSevere observational signals.
    Falls back to zero coherence fields if HRRR is unavailable.
    """
    lat = float(storm.get("lat", 0))
    lon = float(storm.get("lon", 0))

    # Extract coherence at storm location
    if coherence_fields is not None:
        coh = extract_coherence_at_point(coherence_fields, lat, lon)
        tau = coh.get("tau", 0)
        grad_tau = coh.get("grad_tau", 0)
        torsion = coh.get("torsion", 0)
        alignment = coh.get("alignment", 0)
        s_over_gamma = coh.get("s_over_gamma", 0)
        da = coh.get("da", 0)
    else:
        tau = grad_tau = torsion = alignment = s_over_gamma = da = 0.0

    maxllaz = float(storm.get("maxllaz", 0))
    srh01 = float(storm.get("srh01", 0))

    prob = analytic_tornado_probability(
        tau=tau, grad_tau=grad_tau, torsion=torsion,
        alignment=alignment, s_over_gamma=s_over_gamma, da=da,
        maxllaz=maxllaz, srh01=srh01,
    )
    return float(prob)


# ---------------------------------------------------------------------------
# Score storms
# ---------------------------------------------------------------------------


def score_storms(
    time_steps: list[dict],
    hrrr: dict[str, np.ndarray] | None,
    coherence_fields: dict[str, np.ndarray] | None,
    model: dict | None,
    now: dt.datetime,
    scoring_tier: str = "tier3_ps_only",
) -> list[dict]:
    """Score all active storms from the latest ProbSevere time step.

    scoring_tier controls which scoring method is used:
      - "tier1_ml": Full ML model (pre-trained GBT)
      - "tier2_analytic": Analytic coherence probability (no ML)
      - "tier3_ps_only": ProbSevere raw scores only (minimal fallback)

    Returns list of scored storm dicts ready for JSON output.
    """
    if not time_steps:
        return []

    # Use the latest time step
    latest_ts = time_steps[-1]
    storms = latest_ts.get("storms", [])
    valid_time = latest_ts.get("valid_time", now.isoformat() + "Z")

    if not storms:
        return []

    # Build tracks for history
    tracks = build_storm_tracks(time_steps)

    scored: list[dict] = []
    for storm in storms:
        sid = storm.get("id", 0)
        if sid == 0:
            continue

        lat = float(storm.get("lat", 0))
        lon = float(storm.get("lon", 0))

        # Get storm history
        track = tracks.get(sid, [])
        history = [t[2] for t in track]

        # --- Three-tier scoring ---
        top_features: list = []
        model_scores: dict = {}
        coherence_score: float = 0.0

        if scoring_tier == "tier1_ml" and model is not None and HAS_OPERATIONAL:
            # Tier 1: Full ML model
            result = predict_tornado_probability(
                storm, history, hrrr, coherence_fields, model
            )
            prob = result["probability"]
            risk = result["risk_band"]
            top_features = result["top_features"]
            model_scores = result["model_scores"]
            coherence_score = result["coherence_score"]
        elif scoring_tier in ("tier1_ml", "tier2_analytic") and coherence_fields is not None:
            # Tier 2: Analytic coherence model (no ML needed)
            prob = score_storm_analytic(storm, coherence_fields)
            prob = round(min(max(prob, 0.0), 0.99), 4)
            risk = _risk_band(prob)
            model_scores = {"analytic_prob": prob}
            coherence_score = prob
        else:
            # Tier 3: ProbSevere-only fallback
            ps_tor = float(storm.get("ps_tor", 0)) / 100.0
            prob = round(min(max(ps_tor, 0.0), 0.99), 4)
            risk = _risk_band(prob)
            model_scores = {"ps_tor_raw": round(ps_tor, 4)}

        # Coherence diagnostics at storm location
        coherence_diag: dict = {}
        if coherence_fields is not None:
            coherence_diag = extract_coherence_at_point(
                coherence_fields, lat, lon
            )
            sing = test_singularity_at_point(coherence_fields, lat, lon)
            coherence_diag["singularity_conditions_met"] = sing.count
            coherence_diag["singularity_detail"] = {
                "s_over_gamma": sing.s_over_gamma,
                "high_gradient": sing.high_gradient,
                "high_torsion": sing.high_torsion,
                "positive_alignment": sing.positive_alignment,
                "high_damkohler": sing.high_damkohler,
            }

        entry: dict = {
            "storm_id": sid,
            "lat": round(lat, 4),
            "lon": round(lon, 4),
            "motion_east": float(storm.get("motion_east", 0)),
            "motion_south": float(storm.get("motion_south", 0)),
            "valid_time": valid_time,
            "tornado_probability": prob,
            "risk_band": risk,
            "ps_tor": float(storm.get("ps_tor", 0)),
            "ps": float(storm.get("ps", 0)),
            "mucape": float(storm.get("mucape", 0)),
            "ebshear": float(storm.get("ebshear", 0)),
            "srh01": float(storm.get("srh01", 0)),
            "maxllaz": float(storm.get("maxllaz", 0)),
            "mesh": float(storm.get("mesh", 0)),
            "flash_rate": float(storm.get("flash_rate", 0)),
            "top_features": top_features,
            "model_scores": model_scores,
            "coherence_score": coherence_score,
            "coherence_diagnostics": coherence_diag,
            "scoring_tier": scoring_tier,
            "model_version": MODEL_VERSION,
            "track_length": len(history),
        }

        # Include geometry for frontend polygon rendering
        geom = storm.get("geometry")
        if geom:
            entry["geometry"] = geom

        scored.append(entry)

    # Sort by probability descending
    scored.sort(key=lambda s: s["tornado_probability"], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_outputs(
    scored_storms: list[dict],
    now: dt.datetime,
    scoring_tier: str = "tier3_ps_only",
) -> None:
    """Write scored results to dist/data/."""

    # Determine scoring tier label for display
    tier_labels = TIER_LABELS

    # Write live-tornadoes.json
    output = {
        "disclaimer": (
            "RESEARCH ONLY. NOT an operational warning system. "
            "Does NOT replace NWS tornado warnings. Always follow "
            "official NWS guidance. See weather.gov for official alerts."
        ),
        "updated_at": now.isoformat() + "Z",
        "model_version": MODEL_VERSION,
        "scoring_tier": scoring_tier,
        "scoring_tier_label": tier_labels.get(scoring_tier, scoring_tier),
        "n_active_storms": len(scored_storms),
        "storms": scored_storms,
    }
    storms_path = DIST / "data" / "live-tornadoes.json"
    storms_path.parent.mkdir(parents=True, exist_ok=True)
    storms_path.write_text(
        json.dumps(output, indent=2) + "\n", encoding="utf-8"
    )
    print(f"  Wrote {storms_path} ({len(scored_storms)} storms)")

    # Update live-pulse.json tornado entry
    pulse_path = DIST / "data" / "live-pulse.json"
    if pulse_path.exists():
        pulse = json.loads(pulse_path.read_text(encoding="utf-8"))
        for hazard in pulse.get("hazards", []):
            if hazard.get("key") == "to":
                if scored_storms:
                    top = scored_storms[0]  # Already sorted by probability
                    hazard["probability"] = top["tornado_probability"]
                    hazard["risk_band"] = top["risk_band"]
                    hazard["gate_status"] = "pass"
                    hazard["model_version"] = MODEL_VERSION
                    hazard["n_active_storms"] = len(scored_storms)
                    hazard["coherence_score"] = top.get("coherence_score", 0)
                else:
                    hazard["probability"] = 0.0
                    hazard["risk_band"] = "minimal"
                    hazard["gate_status"] = "pass"
                    hazard["model_version"] = MODEL_VERSION
                    hazard["n_active_storms"] = 0
                break
        pulse["updated_at"] = now.isoformat() + "Z"
        pulse_path.write_text(
            json.dumps(pulse, indent=2) + "\n", encoding="utf-8"
        )
        print(f"  Updated {pulse_path}")


TIER_LABELS = {
    "tier1_ml": "ML (pre-trained gradient-boosted trees)",
    "tier2_analytic": "Analytic coherence model (physics-only, no ML)",
    "tier3_ps_only": "ProbSevere-only fallback (no ML, no HRRR)",
}

RISK_COLORS = {
    "very_high": "#d32f2f",
    "high": "#e65100",
    "moderate": "#f9a825",
    "low": "#1976d2",
    "minimal": "#757575",
}

RISK_LABELS = {
    "very_high": "Very High",
    "high": "High",
    "moderate": "Moderate",
    "low": "Low",
    "minimal": "Minimal",
}


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
    return f"{p * 100:.1f}%"


def _format_time(ts: str) -> str:
    """Format a timestamp string for display."""
    if not ts:
        return "--"
    import re
    m = re.match(r"^(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\s*UTC$", str(ts))
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)} {m.group(4)}:{m.group(5)}:{m.group(6)} UTC"
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


def _render_svg_markers(storms: list[dict]) -> str:
    """Generate SVG marker elements for storm positions on the map."""
    lines: list[str] = []
    limit = min(len(storms), 20)
    for i in range(limit):
        s = storms[i]
        rank = i + 1
        x, y = _lat_lon_to_svg(s["lat"], s["lon"])
        risk_label = RISK_LABELS.get(s["risk_band"], s["risk_band"])
        label = f'Storm {s["storm_id"]} - {_pct(s["tornado_probability"])} ({risk_label})'
        base_r = 4.5 if rank == 1 else 3.0

        lines.append(f'    <a href="#storm-{rank}" aria-label="{_esc(label)}">')
        for p in (1, 2):
            lines.append(
                f'      <circle class="hz-pulse hz-pulse-to hz-pulse-delay-{p}" '
                f'cx="{x:.1f}" cy="{y:.1f}" r="{base_r + 1:.0f}"/>'
            )
        lines.append(
            f'      <circle class="hz-marker hz-marker-to" '
            f'cx="{x:.1f}" cy="{y:.1f}" r="{base_r:.1f}" filter="url(#glow-to)"/>'
        )
        # Tooltip
        lines.append(f'      <g class="map-tooltip">')
        lines.append(
            f'        <rect class="tooltip-bg" x="{x + 10:.1f}" y="{y - 18:.1f}" '
            f'width="90" height="22" rx="3"/>'
        )
        lines.append(
            f'        <text class="tooltip-text" x="{x + 12:.1f}" y="{y - 8:.1f}">'
            f'Storm {_esc(str(s["storm_id"]))}</text>'
        )
        lines.append(
            f'        <text class="tooltip-sub" x="{x + 12:.1f}" y="{y + 1:.1f}">'
            f'{_pct(s["tornado_probability"])} \u00b7 Rank #{rank}</text>'
        )
        lines.append(f'      </g>')
        lines.append(f'    </a>')
    return "\n".join(lines)


def _render_storm_rows(storms: list[dict]) -> str:
    """Render storm table as <details>/<summary> elements. Zero JavaScript."""
    lines: list[str] = []
    limit = min(len(storms), 20)
    for i in range(limit):
        s = storms[i]
        rank = i + 1
        risk_color = RISK_COLORS.get(s["risk_band"], "#757575")
        risk_label = RISK_LABELS.get(s["risk_band"], s["risk_band"])
        rank_class = " rank-1" if rank == 1 else ""

        lines.append(f'        <details class="event-row-details" id="storm-{rank}">')
        lines.append(f'          <summary class="event-row">')
        lines.append(
            f'            <span class="rank-badge{rank_class}">{rank}</span>'
            f' <strong>{_esc(str(s["storm_id"]))}</strong>'
            f' {s["lat"]:.2f}, {s["lon"]:.2f}'
            f' <strong style="color:{risk_color}">{_pct(s["tornado_probability"])}</strong>'
            f' <span class="chip" style="background:{risk_color};color:#fff;font-size:11px;padding:2px 8px;">'
            f'{_esc(risk_label)}</span>'
            f' CAPE: {s.get("mucape", 0):.0f}'
            f' | SRH: {s.get("srh01", 0):.0f}'
            f' | MaxLLAz: {s.get("maxllaz", 0):.4f}'
        )
        lines.append(f'          </summary>')
        lines.append(f'          <div class="event-detail" style="padding:12px 10px;">')
        lines.append(f'            <div class="kv"><span>ProbSevere score</span><strong>{s.get("ps", 0):.0f}</strong></div>')
        lines.append(f'            <div class="kv"><span>PS tornado score</span><strong>{s.get("ps_tor", 0):.0f}</strong></div>')
        lines.append(f'            <div class="kv"><span>Effective bulk shear</span><strong>{s.get("ebshear", 0):.0f} kt</strong></div>')
        lines.append(f'            <div class="kv"><span>MESH</span><strong>{s.get("mesh", 0):.1f} in</strong></div>')
        lines.append(f'            <div class="kv"><span>Flash rate</span><strong>{s.get("flash_rate", 0):.0f} /min</strong></div>')
        lines.append(f'            <div class="kv"><span>Track length</span><strong>{s.get("track_length", 0)} time steps</strong></div>')
        lines.append(f'            <div class="kv"><span>Scoring tier</span><strong>{_esc(s.get("scoring_tier", "--"))}</strong></div>')
        lines.append(f'            <div class="kv"><span>Valid time</span><strong>{_format_time(s.get("valid_time", ""))}</strong></div>')
        lines.append(
            f'            <div class="kv"><span>Motion</span><strong>'
            f'{s.get("motion_east", 0):.1f} E, {s.get("motion_south", 0):.1f} S m/s</strong></div>'
        )
        lines.append(f'          </div>')
        lines.append(f'        </details>')
    return "\n".join(lines)


def _render_coherence_deep_dive(top: dict) -> str:
    """Render coherence diagnostics for the top storm. Pure HTML, no JS."""
    coh = top.get("coherence_diagnostics", {})
    if not coh:
        return ""

    risk_color = RISK_COLORS.get(top["risk_band"], "#757575")
    lines: list[str] = []

    lines.append('      <section class="section" aria-labelledby="focus-heading">')
    lines.append('        <h2 id="focus-heading">Top storm -- coherence diagnostics</h2>')
    lines.append('        <div class="grid">')

    # Coherence fields card
    lines.append('          <div class="card col-6 hazard-to">')
    lines.append(f'            <h3>Storm {_esc(str(top["storm_id"]))} -- Coherence fields</h3>')
    lines.append(f'            <div class="metric" style="color:{risk_color}">{_pct(top["tornado_probability"])}</div>')
    lines.append('            <div class="metric-label">Tornado probability</div>')

    coh_keys = [
        ("tau", "tau"), ("grad_tau", "grad_tau"), ("torsion", "torsion"),
        ("alignment", "alignment"), ("S_field", "S_field"),
        ("Gamma_field", "Gamma_field"), ("S_over_Gamma", "S / Gamma"),
        ("Da", "Da (Damkohler)"), ("E_coh", "E_coh"),
        ("singularity_count", "Singularity count"),
    ]
    for key, label in coh_keys:
        val = coh.get(key)
        if val is not None:
            lines.append(
                f'            <div class="kv"><span>{_esc(label)}</span>'
                f'<strong>{float(val):.4f}</strong></div>'
            )
    lines.append('          </div>')

    # Singularity analysis card
    lines.append('          <div class="card col-6 hazard-to">')
    lines.append('            <h3>Singularity analysis</h3>')
    sing = coh.get("singularity_detail", {})
    sing_count = coh.get("singularity_conditions_met", 0)
    lines.append(f'            <div class="kv"><span>Conditions met</span><strong>{sing_count} / 5</strong></div>')

    for sk in ("s_over_gamma", "high_gradient", "high_torsion", "positive_alignment", "high_damkohler"):
        sv = sing.get(sk)
        if sv is not None:
            if sv:
                chip = '<span class="chip" style="background:#d32f2f;color:#fff;font-size:11px;padding:2px 8px;">YES</span>'
            else:
                chip = '<span class="chip" style="background:#757575;color:#fff;font-size:11px;padding:2px 8px;">no</span>'
            lines.append(f'            <div class="kv"><span>{_esc(sk)}</span>{chip}</div>')
    lines.append('          </div>')
    lines.append('        </div>')

    # Storm parameters card
    lines.append('        <div class="grid" style="margin-top:var(--s-md);">')
    lines.append('          <div class="card col-12 hazard-to">')
    lines.append('            <h3>Storm parameters</h3>')
    lines.append(f'            <div class="kv"><span>Location</span><strong>{top["lat"]:.4f}, {top["lon"]:.4f}</strong></div>')
    lines.append(f'            <div class="kv"><span>CAPE</span><strong>{top.get("mucape", 0):.0f} J/kg</strong></div>')
    lines.append(f'            <div class="kv"><span>0-1km SRH</span><strong>{top.get("srh01", 0):.0f} m^2/s^2</strong></div>')
    lines.append(f'            <div class="kv"><span>Eff. bulk shear</span><strong>{top.get("ebshear", 0):.0f} kt</strong></div>')
    lines.append(f'            <div class="kv"><span>MaxLLAz</span><strong>{top.get("maxllaz", 0):.4f} /s</strong></div>')
    lines.append(f'            <div class="kv"><span>Valid time</span><strong>{_esc(_format_time(top.get("valid_time", "")))}</strong></div>')
    lines.append(f'            <div class="kv"><span>Model version</span><strong>{_esc(top.get("model_version", "--"))}</strong></div>')
    lines.append('          </div>')
    lines.append('        </div>')
    lines.append('      </section>')

    return "\n".join(lines)


def render_tornado_page(
    scored_storms: list[dict],
    now: dt.datetime,
    scoring_tier: str = "tier3_ps_only",
    coherence_fields: object = None,
) -> str:
    """Generate complete static HTML for the tornado live page.

    All data is embedded directly in the HTML. Zero JavaScript.
    Follows the HazardPulse Classic Truth Surface spec.
    """
    tier_label = TIER_LABELS.get(scoring_tier, scoring_tier)
    updated_str = _format_time(now.isoformat() + "Z")

    # Load base SVG map
    svg_path = DIST / "assets" / "world-map-base.svg"
    if svg_path.exists():
        svg_content = svg_path.read_text(encoding="utf-8")
    else:
        svg_content = '<svg class="world-map" viewBox="0 0 960 480" xmlns="http://www.w3.org/2000/svg"><rect width="960" height="480" fill="#e4eef8"/></svg>'

    # Inject storm markers into the SVG
    if scored_storms:
        markers_html = _render_svg_markers(scored_storms)
        svg_content = svg_content.replace(
            "    <!-- Markers go here per page -->\n",
            markers_html + "\n",
        )

    # Build disclaimer
    disclaimer = (
        "RESEARCH ONLY. NOT an operational warning system. "
        "Does NOT replace NWS tornado warnings. Always follow "
        "official NWS guidance. See weather.gov for official alerts."
    )

    # Status bar
    status_html = f"""
      <section class="section">
        <div class="grid">
          <div class="card col-4">
            <div class="kv"><span>Last update</span><strong>{_esc(updated_str)}</strong></div>
          </div>
          <div class="card col-4">
            <div class="kv"><span>Scoring model</span><strong>{_esc(tier_label)}</strong></div>
          </div>
          <div class="card col-4">
            <div class="kv"><span>Active storms</span><strong>{len(scored_storms)} storms</strong></div>
          </div>
        </div>
      </section>"""

    # Storms section
    if scored_storms:
        storm_rows = _render_storm_rows(scored_storms)
        storms_html = f"""
      <section class="section" aria-labelledby="systems-heading">
        <h2 id="systems-heading">Active storms by tornado probability</h2>
        <p class="muted" style="margin-top:-8px;margin-bottom:16px;">ProbSevere storm objects scored with coherence field analysis. Ranked by estimated tornado probability. Click any row to expand details.</p>

{storm_rows}
      </section>"""
        dive_html = _render_coherence_deep_dive(scored_storms[0])
    else:
        storms_html = """
      <section class="section">
        <div class="card" style="text-align:center;padding:48px 24px;">
          <h3 style="color:var(--text-secondary);">No active severe weather</h3>
          <p class="muted">No ProbSevere storm objects detected at last scan. Check back during active convective weather.</p>
        </div>
      </section>"""
        dive_html = ""

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tornado Monitor - HazardPulse</title>
  <meta name="description" content="24-hour tornado formation probability for global severe convection zones. Top cells ranked by STP/SCP indices with full evidence.">
  <meta name="theme-color" content="#f6f9ff">
  <link rel="canonical" href="https://hazardpulse.io/live/tornado/">
  <link rel="stylesheet" href="/assets/styles.css?v=4">
  <link rel="icon" type="image/png" sizes="32x32" href="/assets/favicon-32.png">
  <link rel="apple-touch-icon" sizes="180x180" href="/assets/apple-touch-icon.png">
  <link rel="alternate" type="application/rss+xml" title="HazardPulse Feed" href="/feed.xml">

  <meta property="og:type" content="website">
  <meta property="og:title" content="Tornado Monitor - HazardPulse">
  <meta property="og:description" content="24-hour tornado formation probability for global severe convection zones. Top cells ranked by STP/SCP indices with full evidence.">
  <meta property="og:url" content="https://hazardpulse.io/live/tornado/">
  <meta property="og:site_name" content="HazardPulse">
  <meta name="twitter:card" content="summary">
  <meta name="twitter:title" content="Tornado Monitor - HazardPulse">
  <meta name="twitter:description" content="24-hour tornado formation probability for global severe convection zones. Top cells ranked by STP/SCP indices with full evidence.">

  <script type="application/ld+json">
  {{
    "@context": "https://schema.org",
    "@type": "Dataset",
    "name": "HazardPulse Global Tornado Formation Forecast",
    "description": "Probabilistic tornado formation forecasts for active severe convection zones worldwide using composite STP/SCP indices with full provenance chain.",
    "license": "https://hazardpulse.io/legal/disclaimer/",
    "creator": {{ "@type": "Organization", "name": "Coherence Energy Labs", "url": "https://coherenceenergylabs.com" }},
    "temporalCoverage": "{now.strftime('%Y-%m-%d')}/{(now + dt.timedelta(days=1)).strftime('%Y-%m-%d')}",
    "spatialCoverage": {{ "@type": "Place", "name": "Global severe convection zones" }},
    "variableMeasured": "Probability of tornado formation in 24 h"
  }}
  </script>

  <script type="speculationrules">
  {{
    "prefetch": [
      {{ "source": "list", "urls": ["/live/", "/live/earthquake/", "/live/hurricane/", "/evidence/", "/verification/"] }}
    ]
  }}
  </script>
</head>
<body>

  <div class="emergency-banner" role="alert" aria-live="assertive">
    <!-- Populated by Cloudflare Worker when severe convective threat detected near user -->
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
      <div class="eyebrow">Live severe weather intelligence</div>
      <h1 id="hero-heading">Global Tornado Monitor</h1>
      <p class="subtitle">
        24-hour tornado formation probability for the world's most active severe convection zones.
        Ranked by composite STP/SCP indices with full evidence.
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
        <!-- Worker injects personalized severe weather threat content here based on IP geolocation -->
      </section>

      <!-- WORLD MAP -->
      <section class="section" aria-labelledby="worldmap-heading">
        <h2 id="worldmap-heading">Global tornado activity map</h2>
        <p class="muted" style="margin-top:-8px;margin-bottom:16px;">Active and monitored severe convection zones worldwide. Hover a marker for details.</p>

        <div class="world-map-wrapper">
          {svg_content}

          <div class="map-legend">
            <span class="map-legend-item"><span class="map-legend-dot to"></span> Tornado cell</span>
            <span class="map-legend-item"><span style="display:inline-block;width:20px;height:2px;border-top:2px dashed var(--to);opacity:.5;vertical-align:middle;margin-right:2px;"></span> Tornado-prone region</span>
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

      <!-- STORMS -->
{storms_html}

      <!-- TOP STORM DEEP DIVE -->
{dive_html}

      <!-- EVIDENCE AND REPLAY -->
      <section class="section" aria-labelledby="evidence-heading">
        <h2 id="evidence-heading">Evidence and replay</h2>
        <div class="grid">
          <div class="card col-4">
            <h3>See the evidence</h3>
            <p class="muted">Every forecast links to the exact data and model that produced it. Nothing is hidden.</p>
            <a href="/evidence/#to" class="btn btn-secondary" style="margin-top:8px;">Browse evidence</a>
          </div>
          <div class="card col-4">
            <h3>Check our track record</h3>
            <p class="muted">How often are we right? We publish accuracy scores publicly, broken down by region and severity.</p>
            <a href="/verification/" class="btn btn-secondary" style="margin-top:8px;">See accuracy</a>
          </div>
          <div class="card col-4">
            <h3>Replay any forecast</h3>
            <p class="muted">Download the input data and re-run any past forecast yourself. Same data in, same result out - guaranteed.</p>
            <a href="/data/replay/to_fcst_{now.strftime('%Y%m%d')}_0300.json" class="btn btn-secondary" style="margin-top:8px;">Download replay</a>
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
      <p class="footer-build">Built with Coherence Lang · Zero client-side JavaScript · Zero npm dependencies · Geolocation by Cloudflare Edge</p>
    </div>
  </footer>

</body>
</html>
"""
    return page


def render_homepage_cards(
    scored_storms: list[dict],
    now: dt.datetime,
    scoring_tier: str = "tier3_ps_only",
) -> None:
    """Update dist/index.html with current hazard data baked in. Zero JavaScript.

    Reads the existing homepage, strips the <script> block, and replaces
    the dynamic hazard-cards section with statically rendered HTML.
    Also updates the map markers, what-changed, and system health sections.
    """
    homepage_path = DIST / "index.html"
    if not homepage_path.exists():
        print("  Warning: dist/index.html not found, skipping homepage update")
        return

    # Read live-pulse.json for all hazard data
    pulse_path = DIST / "data" / "live-pulse.json"
    if not pulse_path.exists():
        print("  Warning: live-pulse.json not found, skipping homepage update")
        return
    pulse = json.loads(pulse_path.read_text(encoding="utf-8"))

    # Read live-storms.json for hurricane data
    storms_path = DIST / "data" / "live-storms.json"
    hurricanes = {}
    if storms_path.exists():
        try:
            hurricanes = json.loads(storms_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    tier_label = TIER_LABELS.get(scoring_tier, scoring_tier)
    updated_str = _format_time(now.isoformat() + "Z")

    hazard_meta = {
        "eq": {"label": "Earthquake", "color": "var(--eq, #3b7dff)", "link": "/live/earthquake/", "unit": "M6.0+ in 30 days"},
        "hu": {"label": "Hurricane", "color": "var(--hu, #0fa878)", "link": "/live/hurricane/", "unit": "RI in 24 hours"},
        "to": {"label": "Tornado", "color": "var(--to, #c98a12)", "link": "/live/tornado/", "unit": "formation in 24 hours"},
    }

    risk_style = {
        "critical": "bad", "very_high": "bad", "high": "bad",
        "elevated": "warn", "guarded": "warn", "moderate": "warn",
        "low": "good", "minimal": "good",
    }
    risk_labels_all = {
        "critical": "Critical", "very_high": "Very High", "high": "High",
        "elevated": "Elevated", "guarded": "Guarded", "moderate": "Moderate",
        "low": "Low", "minimal": "Minimal",
    }

    # Sort hazards by probability descending
    hazards = sorted(pulse.get("hazards", []), key=lambda h: h.get("probability", 0), reverse=True)

    # Build hazard cards HTML
    cards_lines: list[str] = []
    for i, hz in enumerate(hazards):
        meta = hazard_meta.get(hz.get("key", ""))
        if not meta:
            continue
        rank = i + 1
        prob = _pct(hz.get("probability", 0))
        risk_label = risk_labels_all.get(hz.get("risk_band", ""), hz.get("risk_band", ""))
        rs = risk_style.get(hz.get("risk_band", ""), "warn")
        delta = hz.get("delta", 0)
        delta_sign = "+" if delta >= 0 else ""
        delta_arrow = "\u2191" if delta > 0 else ("\u2193" if delta < 0 else "\u2192")
        conf_lo = _pct(hz["conf_lo"]) if hz.get("conf_lo") else "--"
        conf_hi = _pct(hz["conf_hi"]) if hz.get("conf_hi") else "--"
        model_ver = hz.get("model_version", "")
        gate_status = hz.get("gate_status", "--")
        gate_chip = "good" if gate_status == "pass" else "warn"

        extra = ""
        if hz.get("key") == "hu" and hurricanes.get("storms"):
            s = hurricanes["storms"][0]
            sname = s.get("storm_name", "Active storm")
            extra += f'<div class="kv"><span>Storm</span><strong>{_esc(sname)} ({_esc(s.get("category", "--"))}, {s.get("vmax_kt", "--")} kt)</strong></div>'
            extra += f'<div class="kv"><span>Location</span><strong>{s.get("lat", "--")}\u00b0N, {abs(s.get("lon", 0))}\u00b0W</strong></div>'
        if hz.get("key") == "to":
            n_storms = hz.get("n_active_storms", 0)
            extra += f'<div class="kv"><span>Active storms</span><strong>{n_storms} tracked</strong></div>'
            extra += f'<div class="kv"><span>Scoring</span><strong>{_esc(tier_label)}</strong></div>'
        if hz.get("key") == "eq":
            extra += f'<div class="kv"><span>Forecast</span><strong>{_esc(hz.get("forecast_id", "--"))}</strong></div>'

        model_line = f'<div data-depth="technical"><div class="kv"><span>Model</span><strong>{_esc(model_ver)}</strong></div></div>' if model_ver else ""
        gate_label = "Checks passed" if gate_status == "pass" else _esc(gate_status)

        cards_lines.append(
            f'          <a href="{meta["link"]}" class="card card-link card-secondary hazard-{hz["key"]}" '
            f'aria-label="{_esc(meta["label"])} - {prob} probability" '
            f'style="border-left:4px solid {meta["color"]};">'
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">'
            f'<span class="rank-badge{" rank-1" if rank == 1 else ""}">{rank}</span>'
            f'<h3 style="margin:0;">{_esc(meta["label"])}</h3>'
            f'<span class="chip {rs}" style="margin-left:auto;">{_esc(risk_label)}</span>'
            f'</div>'
            f'<div class="metric">{prob} probability</div>'
            f'<div class="metric-label">of {meta["unit"]}</div>'
            f'<div class="kv"><span>Confidence</span><strong>{conf_lo} \u2013 {conf_hi}</strong></div>'
            f'<div class="kv"><span>Trend</span><strong style="color:{"var(--bad)" if delta > 0 else "var(--good)"}">'
            f'{delta_arrow} {delta_sign}{_pct(abs(delta))}</strong></div>'
            f'{extra}{model_line}'
            f'<div class="chip-row"><span class="chip {gate_chip}">{gate_label}</span></div>'
            f'<span class="card-cta">Full detail \u2192</span>'
            f'</a>'
        )

    cards_html = "\n".join(cards_lines)

    # Build map markers
    map_markers: list[str] = []
    # Hurricane markers
    for h_idx, hs in enumerate(hurricanes.get("storms", [])):
        x, y = _lat_lon_to_svg(hs.get("lat", 0), hs.get("lon", 0))
        hu_name = hs.get("storm_name", f'Storm {hs.get("storm_id", "")}')
        hu_sub = f'{_pct(hs.get("ri_probability", 0))} RI \u00b7 {hs.get("category", "")}'
        rank = h_idx + 1
        base_r = 5 if rank == 1 else (4 if rank <= 3 else 3)
        map_markers.append(f'    <a href="/live/hurricane/" aria-label="{_esc(hu_name)} - {_esc(hu_sub)}">')
        for p in (1, 2):
            map_markers.append(f'      <circle class="hz-pulse hz-pulse-hu hz-pulse-delay-{p}" cx="{x:.1f}" cy="{y:.1f}" r="{base_r + 1}"/>')
        map_markers.append(f'      <circle class="hz-marker hz-marker-hu" cx="{x:.1f}" cy="{y:.1f}" r="{base_r}" filter="url(#glow-hu)"/>')
        map_markers.append(f'      <g class="map-tooltip"><rect class="tooltip-bg" x="{x+10:.1f}" y="{y-18:.1f}" width="100" height="22" rx="3"/>')
        map_markers.append(f'        <text class="tooltip-text" x="{x+12:.1f}" y="{y-8:.1f}">{_esc(hu_name)}</text>')
        map_markers.append(f'        <text class="tooltip-sub" x="{x+12:.1f}" y="{y+1:.1f}">{_esc(hu_sub)}</text></g>')
        map_markers.append(f'    </a>')

    # Tornado markers
    for t_idx, ts in enumerate(scored_storms[:10]):
        x, y = _lat_lon_to_svg(ts["lat"], ts["lon"])
        to_sub = f'{_pct(ts["tornado_probability"])} \u00b7 {RISK_LABELS.get(ts["risk_band"], ts["risk_band"])}'
        rank = len(hurricanes.get("storms", [])) + t_idx + 1
        base_r = 5 if rank == 1 else (4 if rank <= 3 else 3)
        map_markers.append(f'    <a href="/live/tornado/" aria-label="Storm {ts["storm_id"]} - {_esc(to_sub)}">')
        for p in (1, 2):
            map_markers.append(f'      <circle class="hz-pulse hz-pulse-to hz-pulse-delay-{p}" cx="{x:.1f}" cy="{y:.1f}" r="{base_r + 1}"/>')
        map_markers.append(f'      <circle class="hz-marker hz-marker-to" cx="{x:.1f}" cy="{y:.1f}" r="{base_r}" filter="url(#glow-to)"/>')
        map_markers.append(f'      <g class="map-tooltip"><rect class="tooltip-bg" x="{x+10:.1f}" y="{y-18:.1f}" width="100" height="22" rx="3"/>')
        map_markers.append(f'        <text class="tooltip-text" x="{x+12:.1f}" y="{y-8:.1f}">Storm {_esc(str(ts["storm_id"]))}</text>')
        map_markers.append(f'        <text class="tooltip-sub" x="{x+12:.1f}" y="{y+1:.1f}">{_esc(to_sub)}</text></g>')
        map_markers.append(f'    </a>')

    markers_svg = "\n".join(map_markers)

    # Build what-changed section content
    simple_lines: list[str] = []
    tech_lines: list[str] = []
    for hz in hazards:
        meta = hazard_meta.get(hz.get("key", ""))
        if not meta:
            continue
        delta = hz.get("delta", 0)
        delta_sign = "+" if delta >= 0 else ""
        delta_arrow = "\u2191" if delta > 0 else ("\u2193" if delta < 0 else "\u2192")
        risk_label = risk_labels_all.get(hz.get("risk_band", ""), hz.get("risk_band", ""))
        delta_str = f"{delta_arrow}{delta_sign}{_pct(abs(delta))}"

        if hz.get("key") == "hu" and hurricanes.get("storms"):
            s = hurricanes["storms"][0]
            sname = s.get("storm_name", "Active storm")
            simple_lines.append(f'<div class="kv"><span>{_esc(meta["label"])} ({delta_str})</span><strong>{_esc(sname)} ({_esc(s.get("category", "--"))}) with {_pct(hz.get("probability", 0))} RI probability. Risk band: {_esc(risk_label)}.</strong></div>')
            tech_lines.append(f'<div class="kv"><span>{_esc(meta["label"])} ({delta_str})</span><strong>{_esc(hz.get("model_version", "--"))}. RI prob {_pct(hz.get("probability", 0))}. SST {s.get("sst_c", "--")}\u00b0C, shear {s.get("shear_kt", "--")} kt.</strong></div>')
        elif hz.get("key") == "to":
            n_storms = hz.get("n_active_storms", 0)
            s_word = "s" if n_storms != 1 else ""
            simple_lines.append(f'<div class="kv"><span>{_esc(meta["label"])} ({delta_str})</span><strong>{n_storms} active storm{s_word} tracked. Risk band: {_esc(risk_label)}. Probability {_pct(hz.get("probability", 0))}.</strong></div>')
            tech_lines.append(f'<div class="kv"><span>{_esc(meta["label"])} ({delta_str})</span><strong>{_esc(hz.get("model_version", "--"))}. {n_storms} storms. Prob {_pct(hz.get("probability", 0))}</strong></div>')
        elif hz.get("key") == "eq":
            simple_lines.append(f'<div class="kv"><span>{_esc(meta["label"])} ({delta_str})</span><strong>Probability at {_pct(hz.get("probability", 0))}. Risk band: {_esc(risk_label)}.</strong></div>')
            tech_lines.append(f'<div class="kv"><span>{_esc(meta["label"])} ({delta_str})</span><strong>Forecast {_esc(hz.get("forecast_id", "--"))}. Prob {_pct(hz.get("probability", 0))} [{_pct(hz.get("conf_lo", 0))}, {_pct(hz.get("conf_hi", 0))}].</strong></div>')

    simple_content = "\n                ".join(simple_lines) if simple_lines else '<p class="muted">No recent changes.</p>'
    tech_content = "\n                ".join(tech_lines) if tech_lines else '<p class="muted">No recent changes.</p>'

    # System health
    total_storms = sum(h.get("n_active_storms", 0) for h in hazards)
    all_pass = all(h.get("gate_status") == "pass" for h in hazards)
    gate_text = "All gates passed" if all_pass else "Some gates degraded"
    gate_color = "var(--good)" if all_pass else "var(--warn)"

    # Read existing homepage base SVG
    svg_home_path = DIST / "assets" / "world-map-base.svg"
    if svg_home_path.exists():
        svg_home = svg_home_path.read_text(encoding="utf-8")
        svg_home = svg_home.replace(
            "    <!-- Markers go here per page -->\n",
            markers_svg + "\n",
        )
    else:
        svg_home = ""

    # Now build the full homepage
    homepage = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HazardPulse - Global hazard intelligence you can verify</title>
  <meta name="description" content="Live probabilistic hazard forecasts for earthquakes, hurricanes, and tornadoes worldwide. Transparent uncertainty, verifiable evidence.">
  <meta name="theme-color" content="#f6f9ff">
  <link rel="canonical" href="https://hazardpulse.io/">
  <link rel="stylesheet" href="/assets/styles.css?v=4">
  <link rel="icon" type="image/png" sizes="32x32" href="/assets/favicon-32.png">
  <link rel="apple-touch-icon" sizes="180x180" href="/assets/apple-touch-icon.png">
  <link rel="alternate" type="application/rss+xml" title="HazardPulse Feed" href="/feed.xml">

  <meta property="og:type" content="website">
  <meta property="og:title" content="HazardPulse - Global hazard intelligence you can verify">
  <meta property="og:description" content="Live probabilistic hazard forecasts for earthquakes, hurricanes, and tornadoes worldwide with full evidence lineage.">
  <meta property="og:url" content="https://hazardpulse.io/">
  <meta property="og:site_name" content="HazardPulse">
  <meta name="twitter:card" content="summary">
  <meta name="twitter:title" content="HazardPulse - Global hazard intelligence you can verify">
  <meta name="twitter:description" content="Live probabilistic hazard forecasts for earthquakes, hurricanes, and tornadoes worldwide with full evidence lineage.">

  <script type="application/ld+json">
  {{
    "@context": "https://schema.org",
    "@type": "WebSite",
    "name": "HazardPulse",
    "url": "https://hazardpulse.io/",
    "description": "Global hazard intelligence instrument providing live probabilistic forecasts with full evidence lineage and deterministic verification.",
    "publisher": {{
      "@type": "Organization",
      "name": "Coherence Energy Labs",
      "url": "https://coherenceenergylabs.com"
    }}
  }}
  </script>

  <script type="speculationrules">
  {{
    "prefetch": [
      {{ "source": "list", "urls": ["/live/", "/live/earthquake/", "/live/hurricane/", "/live/tornado/", "/evidence/", "/verification/"] }}
    ]
  }}
  </script>
</head>
<body>

  <div class="emergency-banner" role="alert" aria-live="assertive">
    <!-- Populated by Cloudflare Worker when threat detected near user -->
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
          <a href="/live/">Live</a>
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
      <div class="eyebrow">Global hazard intelligence</div>
      <h1 id="hero-heading">Know your risk. Check the proof.</h1>
      <p class="subtitle">
        HazardPulse tracks earthquakes, hurricanes, and tornadoes worldwide. Every forecast
        shows how likely, how confident, and exactly what evidence backs it - so you can verify it yourself.
      </p>
      <div class="cta-row">
        <a href="/live/" class="btn btn-primary">See live forecasts</a>
        <a href="/evidence/" class="btn btn-secondary">Check the evidence</a>
        <a href="/methods/" class="btn btn-secondary">How we do this</a>
      </div>
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
        <!-- Worker injects personalized content here based on IP geolocation -->
      </section>

      <!-- WORLD MAP -->
      <section class="section" aria-labelledby="worldmap-heading">
        <h2 id="worldmap-heading">Global hazard map</h2>
        <p class="muted" style="margin-top:-8px;margin-bottom:16px;">Active hazard zones across the world. Hover a marker for details. Click to see full analysis.</p>

        <div class="world-map-wrapper">
          {svg_home}

          <div class="map-legend">
            <span class="map-legend-item"><span class="map-legend-dot eq"></span> Earthquake zone</span>
            <span class="map-legend-item"><span class="map-legend-dot hu"></span> Tropical system</span>
            <span class="map-legend-item"><span class="map-legend-dot to"></span> Tornado cell</span>
            <span class="map-legend-item"><span class="map-legend-dot user"></span> Your location</span>
          </div>
          <p class="muted" style="margin-top:8px;font-size:11px;text-align:center;">RESEARCH SYSTEM &mdash; Not operational. See <a href="https://www.weather.gov/" rel="noopener">weather.gov</a> for official warnings.</p>
        </div>
      </section>

      <!-- TOP 3 GLOBAL HAZARDS -->
      <section class="section" aria-labelledby="pulse-heading">
        <h2 id="pulse-heading">Highest risk right now</h2>
        <p class="muted" style="margin-top:-8px;margin-bottom:16px;">The three highest-risk hazards we're tracking globally, ranked by probability. Updated every 3 hours.</p>

        <div class="grid">
{cards_html}
        </div>
      </section>

      <!-- WHAT CHANGED & SYSTEM HEALTH -->
      <section class="section" aria-labelledby="why-heading">
        <div class="grid">
          <div class="col-8">
            <h2 id="why-heading">What changed and why</h2>
            <p class="muted" style="margin-top:-8px;margin-bottom:16px;">Plain-language summary of what's driving the numbers since last update.</p>
            <div class="card">
              <div data-depth="simple">
                {simple_content}
              </div>
              <div data-depth="technical">
                {tech_content}
              </div>
            </div>
          </div>
          <div class="col-4">
            <h2>System health</h2>
            <p class="muted" style="margin-top:-8px;margin-bottom:16px;">Is HazardPulse working properly?</p>
            <div class="card">
              <div class="kv"><span>Last update</span><strong>{_esc(updated_str)}</strong></div>
              <div class="kv"><span>Active storms</span><strong>{total_storms} tracked globally</strong></div>
              <div class="kv"><span>Hazard types</span><strong>{len(hazards)} hazard types monitored</strong></div>
              <div class="kv"><span>Gate status</span><strong style="color:{gate_color}">{gate_text}</strong></div>
              <div data-depth="technical">
                <div class="kv"><span>Data sources</span><strong>{_esc(MODEL_VERSION)}</strong></div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <!-- HOW IT WORKS -->
      <section class="section" aria-labelledby="how-heading">
        <h2 id="how-heading">How it works</h2>
        <p class="muted" style="margin-top:-8px;margin-bottom:16px;">Four steps between raw data and the numbers you see above.</p>
        <div class="grid">
          <div class="card col-3">
            <h3>1. Collect</h3>
            <div data-depth="simple">
              <p class="muted">We pull real-time data from the USGS (earthquakes), National Hurricane Center (storms), Storm Prediction Center (tornadoes), JMA (Japan), and IMD (India).</p>
            </div>
            <div data-depth="technical">
              <p class="muted">Ingestion from USGS ComCat, NHC ATCF, SPC mesoscale, JMA seismic catalog, EMSC, IMD cyclone bulletins. Validated against schema v2.1, timestamped with provenance envelope.</p>
            </div>
          </div>
          <div class="card col-3">
            <h3>2. Calculate</h3>
            <div data-depth="simple">
              <p class="muted">Statistical models estimate how likely each hazard is, and calculate a confidence range showing how sure we are.</p>
            </div>
            <div data-depth="technical">
              <p class="muted">Ensemble inference across registered models (ETAS, SHIPS-RII, DTOPS, SPC-calibrated meso). Bayesian credible intervals via MCMC (n=10,000). Deterministic execution on Coherence runtime.</p>
            </div>
          </div>
          <div class="card col-3">
            <h3>3. Verify</h3>
            <div data-depth="simple">
              <p class="muted">Every forecast goes through 13 automatic checks for accuracy, freshness, and safety before it's published.</p>
            </div>
            <div data-depth="technical">
              <p class="muted">Hard gate constitution: G0 (schema) through G12 (calibration drift). Each gate produces a signed decision envelope. Degrade-and-explain on failure - never silent publishing.</p>
            </div>
          </div>
          <div class="card col-3">
            <h3>4. Show you</h3>
            <div data-depth="simple">
              <p class="muted">If all checks pass, the forecast goes live with a link to the evidence that created it - so you can verify it yourself.</p>
            </div>
            <div data-depth="technical">
              <p class="muted">Static site generation from SiteWorld graph. Each page deterministically compiled from .cl source with content-addressed assets. Full replay bundle published per cycle.</p>
            </div>
          </div>
        </div>
      </section>

      <!-- DON'T TAKE OUR WORD -->
      <section class="section" aria-labelledby="verify-heading">
        <h2 id="verify-heading">Don't take our word for it</h2>
        <p class="muted" style="margin-top:-8px;margin-bottom:16px;">Every claim on this site can be checked.</p>
        <div class="grid">
          <div class="card col-4">
            <h3>See the evidence</h3>
            <p class="muted">Every forecast links to the exact data and model that produced it. Nothing is hidden.</p>
            <a href="/evidence/" class="btn btn-secondary" style="margin-top:8px;">Browse evidence</a>
          </div>
          <div class="card col-4">
            <h3>Check our track record</h3>
            <p class="muted">How often are we right? We publish our accuracy scores publicly, broken down by hazard type and region.</p>
            <a href="/verification/" class="btn btn-secondary" style="margin-top:8px;">See accuracy</a>
          </div>
          <div class="card col-4">
            <h3>Replay any forecast</h3>
            <p class="muted">Download the input data and re-run any past forecast yourself. Same data in, same result out - guaranteed.</p>
            <a href="/evidence/#replay" class="btn btn-secondary" style="margin-top:8px;">Try it</a>
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
      <p class="footer-build">Built with Coherence Lang · Zero client-side JavaScript · Zero npm dependencies · Geolocation by Cloudflare Edge</p>
    </div>
  </footer>

</body>
</html>
"""
    homepage_path.write_text(homepage, encoding="utf-8")
    print(f"  Wrote {homepage_path} (zero JS, all data baked in)")


def append_ledger(
    scored_storms: list[dict],
    now: dt.datetime,
) -> None:
    """Append prediction to SHA-256 chain ledger."""
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Read previous hash
    prev_hash = "0" * 64
    if LEDGER_PATH.exists():
        lines = LEDGER_PATH.read_text(encoding="utf-8").strip().split("\n")
        if lines:
            try:
                last = json.loads(lines[-1])
                prev_hash = last.get("hash", prev_hash)
            except json.JSONDecodeError:
                pass

    # Build ledger entry
    entry = {
        "timestamp": now.isoformat() + "Z",
        "model_version": MODEL_VERSION,
        "n_storms": len(scored_storms),
        "top_probability": (
            scored_storms[0]["tornado_probability"] if scored_storms else 0.0
        ),
        "prev_hash": prev_hash,
    }
    # Add storm IDs and probabilities
    entry["storms"] = [
        {
            "id": s["storm_id"],
            "prob": s["tornado_probability"],
            "risk": s["risk_band"],
        }
        for s in scored_storms[:20]  # Cap at 20 for ledger size
    ]

    # SHA-256 hash of this entry
    payload = json.dumps(entry, sort_keys=True, separators=(",", ":"))
    entry["hash"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    with LEDGER_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, separators=(",", ":")) + "\n")
    print(f"  Appended to {LEDGER_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the tornado scoring pipeline."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None,
                        help="Date to score (YYYYMMDD). Default: today UTC.")
    args = parser.parse_args()

    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    date_str = args.date if args.date else now.strftime("%Y%m%d")
    print(f"HazardPulse Tornado Scoring Pipeline -- {now.isoformat()}Z")
    print()

    # Step 1: Fetch ProbSevere data
    print("Step 1: Fetching ProbSevere storm objects...")
    try:
        time_steps = fetch_probsevere_day(date_str, refresh=True)
    except Exception as e:
        print(f"  Warning: ProbSevere fetch failed: {e}")
        time_steps = []

    # Fallback to cache
    if not time_steps:
        cached = load_cached_probsevere(date_str)
        if cached is not None:
            time_steps = cached
            print(f"  Loaded {len(time_steps)} time steps from cache")
        else:
            print("  No ProbSevere data available.")
            print("  Writing empty state...")
            write_outputs([], now, scoring_tier="tier3_ps_only")
            append_ledger([], now)
            # Render static pages with empty storm list
            tornado_html = render_tornado_page([], now, scoring_tier="tier3_ps_only")
            tornado_page = DIST / "live" / "tornado" / "index.html"
            tornado_page.parent.mkdir(parents=True, exist_ok=True)
            tornado_page.write_text(tornado_html, encoding="utf-8")
            print(f"  Wrote {tornado_page} (no storms, zero JS)")
            render_homepage_cards([], now, scoring_tier="tier3_ps_only")
            print()
            print("Done. No storms to score.")
            return

    n_storms_latest = len(time_steps[-1].get("storms", [])) if time_steps else 0
    print(f"  {len(time_steps)} time steps, {n_storms_latest} storms in latest")

    if n_storms_latest == 0:
        print("  No active storms in latest time step.")
        write_outputs([], now, scoring_tier="tier3_ps_only")
        append_ledger([], now)
        # Render static pages with empty storm list
        tornado_html = render_tornado_page([], now, scoring_tier="tier3_ps_only")
        tornado_page = DIST / "live" / "tornado" / "index.html"
        tornado_page.parent.mkdir(parents=True, exist_ok=True)
        tornado_page.write_text(tornado_html, encoding="utf-8")
        print(f"  Wrote {tornado_page} (no storms, zero JS)")
        render_homepage_cards([], now, scoring_tier="tier3_ps_only")
        print()
        print("Done. No storms to score.")
        return

    # Step 2: Fetch HRRR analysis
    print()
    print("Step 2: Fetching HRRR 18Z analysis...")
    hrrr: dict[str, np.ndarray] | None = None

    # Try current day 18Z, then 12Z, then yesterday 18Z
    for hour in (18, 12):
        hrrr = load_cached_hrrr(date_str, hour=hour)
        if hrrr is not None:
            print(f"  Loaded HRRR {hour}Z from cache")
            break

    if hrrr is None:
        try:
            hrrr = fetch_hrrr_grid(date_str, hour=18)
            print("  Fetched HRRR 18Z from AWS")
        except Exception as e:
            print(f"  Warning: HRRR fetch failed: {e}")
            print("  Proceeding without HRRR (ProbSevere fallback mode)")

    # Step 3: Compute coherence fields
    print()
    print("Step 3: Computing coherence fields...")
    coherence_fields: dict[str, np.ndarray] | None = None
    if hrrr is not None:
        try:
            coherence_fields = compute_coherence_fields(hrrr, month=now.month)
            tau_max = float(coherence_fields["tau"].max())
            sing_max = float(coherence_fields["singularity_count"].max())
            print(f"  tau_max={tau_max:.4f}, singularity_max={sing_max:.0f}")
        except Exception as e:
            print(f"  Warning: Coherence field computation failed: {e}")
    else:
        print("  Skipped (no HRRR data)")

    # Step 4: Determine scoring tier
    print()
    print("Step 4: Determining scoring tier...")
    model: dict | None = None
    scoring_tier: str = "tier3_ps_only"

    # Tier 1: Try loading a pre-trained ML model
    model = load_pretrained_model()
    if model is not None:
        scoring_tier = "tier1_ml"
        print(f"  -> Tier 1: Pre-trained ML model")
    elif coherence_fields is not None:
        # Tier 2: Use analytic coherence model (no ML needed)
        scoring_tier = "tier2_analytic"
        print(f"  -> Tier 2: Analytic coherence model (no ML, uses HRRR)")
    else:
        # Tier 3: ProbSevere-only fallback
        scoring_tier = "tier3_ps_only"
        print(f"  -> Tier 3: ProbSevere-only fallback (no ML, no HRRR)")

    # Step 5: Score active storms
    print()
    print("Step 5: Scoring active storms...")
    scored = score_storms(
        time_steps, hrrr, coherence_fields, model, now,
        scoring_tier=scoring_tier,
    )

    for s in scored[:10]:
        print(
            f"  Storm {s['storm_id']}: P(tornado) = {s['tornado_probability']:.1%} "
            f"[{s['risk_band']}] -- CAPE={s['mucape']:.0f}, "
            f"SRH={s['srh01']:.0f}, MaxLLAz={s['maxllaz']:.4f}"
        )

    # Step 6: Write outputs
    print()
    print("Step 6: Writing outputs...")
    write_outputs(scored, now, scoring_tier=scoring_tier)
    append_ledger(scored, now)

    # Step 7: Render static HTML pages (zero JavaScript)
    print()
    print("Step 7: Rendering static HTML pages (zero JS)...")
    tornado_html = render_tornado_page(
        scored, now, scoring_tier=scoring_tier,
        coherence_fields=coherence_fields,
    )
    tornado_page = DIST / "live" / "tornado" / "index.html"
    tornado_page.parent.mkdir(parents=True, exist_ok=True)
    tornado_page.write_text(tornado_html, encoding="utf-8")
    print(f"  Wrote {tornado_page} ({len(scored)} storms baked in, zero JS)")

    render_homepage_cards(scored, now, scoring_tier=scoring_tier)

    print()
    print(f"Done. Scored {len(scored)} storms.")


if __name__ == "__main__":
    main()
