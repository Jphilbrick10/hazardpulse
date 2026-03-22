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
    tier_labels = {
        "tier1_ml": "ML (pre-trained gradient-boosted trees)",
        "tier2_analytic": "Analytic coherence model (physics-only, no ML)",
        "tier3_ps_only": "ProbSevere-only fallback (no ML, no HRRR)",
    }

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
        time_steps = fetch_probsevere_day(date_str)
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
            print()
            print("Done. No storms to score.")
            return

    n_storms_latest = len(time_steps[-1].get("storms", [])) if time_steps else 0
    print(f"  {len(time_steps)} time steps, {n_storms_latest} storms in latest")

    if n_storms_latest == 0:
        print("  No active storms in latest time step.")
        write_outputs([], now, scoring_tier="tier3_ps_only")
        append_ledger([], now)
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

    print()
    print(f"Done. Scored {len(scored)} storms.")


if __name__ == "__main__":
    main()
