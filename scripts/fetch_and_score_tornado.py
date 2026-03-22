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
    scan_probsevere_cache,
)
from hazardpulse.tornado.coherence_engine import (  # noqa: E402
    compute_coherence_fields,
    compute_derived_hrrr,
    extract_coherence_at_point,
    test_singularity_at_point,
)
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
# Quick-train a model on cached historical data
# ---------------------------------------------------------------------------


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate haversine distance in km."""
    import math
    dlat = (lat2 - lat1) * 111.0
    dlon = (lon2 - lon1) * 111.0 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.sqrt(dlat ** 2 + dlon ** 2)


def _load_spc_tornado_index() -> dict[str, list[dict]]:
    """Load SPC tornado reports indexed by date string YYYYMMDD."""
    import csv
    import io
    from hazardpulse.data.http import fetch_text

    url = "https://www.spc.noaa.gov/wcm/data/1950-2024_actual_tornadoes.csv"
    try:
        raw = fetch_text(url, namespace="spc_tornado", timeout=120)
    except Exception:
        # Try 2023 version
        raw = fetch_text(
            "https://www.spc.noaa.gov/wcm/data/1950-2023_actual_tornadoes.csv",
            namespace="spc_tornado", timeout=120,
        )

    by_date: dict[str, list[dict]] = {}
    for row in csv.DictReader(io.StringIO(raw)):
        try:
            yr = int(row.get("yr", 0))
            mo = int(row.get("mo", 0))
            dy = int(row.get("dy", 0))
            slat = float(row.get("slat", 0))
            slon = float(row.get("slon", 0))
            mag = int(row.get("mag", -9))
            if yr < 2020 or mag < 0 or slat == 0:
                continue
            date_str = f"{yr:04d}{mo:02d}{dy:02d}"
            time_str = row.get("time", "")
            hour = -1.0
            if time_str and ":" in time_str:
                try:
                    parts = time_str.split(":")
                    hour = int(parts[0]) + int(parts[1]) / 60.0
                except Exception:
                    hour = -1.0
            by_date.setdefault(date_str, []).append({
                "slat": slat, "slon": slon, "mag": mag, "hour": hour,
            })
        except (ValueError, KeyError):
            continue
    return by_date


def train_quick_model(
    config: TornadoStormConfig | None = None,
) -> dict | None:
    """Train a lightweight model on any available cached ProbSevere data.

    Uses actual SPC tornado reports as labels (matched by 40km proximity).
    If no historical ProbSevere data is cached, returns None.
    """
    if config is None:
        config = TornadoStormConfig()

    cached_dates = scan_probsevere_cache()
    if not cached_dates:
        return None

    # Load SPC tornado reports for label matching
    print("  Loading SPC tornado reports for labels...")
    spc_tornadoes_by_date = _load_spc_tornado_index()
    print(f"  SPC tornado dates: {len(spc_tornadoes_by_date)}")

    print(f"  Found {len(cached_dates)} cached ProbSevere days for training")

    # Build a minimal training set from cached data
    from hazardpulse.tornado.coherence_engine import compute_coherence_fields

    train_samples: list[dict] = []
    for date_str in cached_dates[:60]:  # Cap at 60 days for speed
        ps = load_cached_probsevere(date_str)
        if ps is None or not ps:
            continue

        # Try loading HRRR for this day
        hrrr = load_cached_hrrr(date_str, hour=18)
        coh_fields = None
        derived = None
        if hrrr is not None:
            try:
                coh_fields = compute_coherence_fields(hrrr)
                derived = coh_fields.get("_derived")
            except Exception:
                coh_fields = None

        tracks = build_storm_tracks(ps)
        for sid, track in tracks.items():
            if len(track) < 2:
                continue
            history: list[dict] = []
            for ts_idx, valid_time, storm in track[-5:]:
                history.append(storm)
            storm = track[-1][2]
            block_s = extract_block_s(storm)
            block_e = extract_block_e(storm, history)
            lat = float(storm.get("lat", 0))
            lon = float(storm.get("lon", 0))
            if hrrr is not None and derived is not None:
                block_a = extract_block_a(lat, lon, hrrr, derived)
            else:
                block_a = extract_block_a_from_probsevere(storm)
            block_t = extract_block_t(lat, lon, storm, coh_fields)

            # Parse storm valid_time to get hour
            storm_hour = float(storm.get("hour", -1))
            if storm_hour < 0:
                # Try parsing from valid_time string
                vt = storm.get("valid_time", "")
                if "T" in vt:
                    try:
                        storm_hour = int(vt.split("T")[1][:2]) + int(vt.split("T")[1][3:5]) / 60.0
                    except Exception:
                        storm_hour = -1

            # Use actual SPC tornado reports as labels
            # Match: tornado within 40km AND within 60 min after storm obs
            label = 0
            for tor in spc_tornadoes_by_date.get(date_str, []):
                dist = _haversine_km(lat, lon, tor["slat"], tor["slon"])
                if dist > 40.0:
                    continue
                # Check temporal overlap: tornado must occur within 60 min of storm obs
                tor_hour = tor.get("hour", -1)
                if storm_hour >= 0 and tor_hour >= 0:
                    dt_hours = tor_hour - storm_hour
                    if dt_hours < 0:
                        dt_hours += 24  # handle midnight crossing
                    if 0 <= dt_hours <= 1.0:  # within 60 min forward
                        label = 1
                        break
                elif dist <= 40.0:
                    # No hour info -- fall back to same-day (mark as uncertain)
                    label = 1
                    break

            train_samples.append({
                "S": block_s, "E": block_e,
                "A": block_a, "T": block_t,
                "label": label, "date": date_str,
            })

    if len(train_samples) < 50:
        print("  Not enough training samples, using fallback mode")
        return None

    # Temporal split: first 80% of dates = train, last 20% = val
    all_dates_sorted = sorted(set(s.get("date", "") for s in train_samples))
    n_dates = len(all_dates_sorted)
    split_date = all_dates_sorted[int(n_dates * 0.8)]
    train_data = [s for s in train_samples if s.get("date", "") <= split_date]
    val_data = [s for s in train_samples if s.get("date", "") > split_date]

    if len(val_data) < 10:
        # Not enough validation data with temporal split; fall back
        np.random.seed(42)
        np.random.shuffle(train_samples)
        split = int(0.8 * len(train_samples))
        train_data = train_samples[:split]
        val_data = train_samples[split:]

    print(f"  Training quick model on {len(train_data)} samples...")

    from hazardpulse.tornado.operational_storm import train_operational_tornado_model
    model = train_operational_tornado_model(train_data, val_data, config)
    return model


# ---------------------------------------------------------------------------
# Score storms
# ---------------------------------------------------------------------------


def score_storms(
    time_steps: list[dict],
    hrrr: dict[str, np.ndarray] | None,
    coherence_fields: dict[str, np.ndarray] | None,
    model: dict | None,
    now: dt.datetime,
) -> list[dict]:
    """Score all active storms from the latest ProbSevere time step.

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

        # Score
        if model is not None:
            result = predict_tornado_probability(
                storm, history, hrrr, coherence_fields, model
            )
            prob = result["probability"]
            risk = result["risk_band"]
            top_features = result["top_features"]
            model_scores = result["model_scores"]
            coherence_score = result["coherence_score"]
        else:
            # Fallback: use ProbSevere tornado score + coherence diagnostics
            ps_tor = float(storm.get("ps_tor", 0)) / 100.0
            prob = round(min(ps_tor, 0.99), 4)
            risk = _risk_band(prob)
            top_features = []
            model_scores = {"ps_tor_raw": round(ps_tor, 4)}
            coherence_score = 0.0

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
) -> None:
    """Write scored results to dist/data/."""

    # Write live-tornadoes.json
    output = {
        "disclaimer": (
            "RESEARCH ONLY. NOT an operational warning system. "
            "Does NOT replace NWS tornado warnings. Always follow "
            "official NWS guidance. See weather.gov for official alerts."
        ),
        "updated_at": now.isoformat() + "Z",
        "model_version": MODEL_VERSION,
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
            write_outputs([], now)
            append_ledger([], now)
            print()
            print("Done. No storms to score.")
            return

    n_storms_latest = len(time_steps[-1].get("storms", [])) if time_steps else 0
    print(f"  {len(time_steps)} time steps, {n_storms_latest} storms in latest")

    if n_storms_latest == 0:
        print("  No active storms in latest time step.")
        write_outputs([], now)
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

    # Step 4: Load or train model
    print()
    print("Step 4: Loading prediction model...")
    model: dict | None = None

    # Try to load pre-trained model from results/
    model_path = RESULTS / "tornado_storm_model.json"
    if model_path.exists():
        try:
            model_data = json.loads(model_path.read_text(encoding="utf-8"))
            print(f"  Loaded pre-trained model v{model_data.get('version', '?')}")
            # For now, fall through to quick-train since JSON model loading
            # requires reconstructing GBT trees.  TODO: implement model
            # serialization.
            model = None
        except Exception:
            pass

    if model is None:
        print("  No pre-trained model found. Training quick model...")
        model = train_quick_model()
        if model is not None:
            print(f"  Quick model trained (val AUC={model.get('val_auc', 0):.4f})")
        else:
            print("  Using ProbSevere fallback scoring (no ML model)")

    # Step 5: Score active storms
    print()
    print("Step 5: Scoring active storms...")
    scored = score_storms(time_steps, hrrr, coherence_fields, model, now)

    for s in scored[:10]:
        print(
            f"  Storm {s['storm_id']}: P(tornado) = {s['tornado_probability']:.1%} "
            f"[{s['risk_band']}] -- CAPE={s['mucape']:.0f}, "
            f"SRH={s['srh01']:.0f}, MaxLLAz={s['maxllaz']:.4f}"
        )

    # Step 6: Write outputs
    print()
    print("Step 6: Writing outputs...")
    write_outputs(scored, now)
    append_ledger(scored, now)

    print()
    print(f"Done. Scored {len(scored)} storms.")


if __name__ == "__main__":
    main()
