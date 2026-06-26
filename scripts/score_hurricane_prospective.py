#!/usr/bin/env python3
"""Score frozen hurricane replay artifacts against realized intensity changes.

Reads replay artifacts from dist/data/replay/hu_fcst_*.json, evaluates
only forecasts whose windows have fully matured, fetches NHC ATCF
best-track data to determine if rapid intensification actually occurred,
and writes:

- results/hurricane_prospective/prospective_summary.json
- results/hurricane_prospective/per_forecast_scores.jsonl

For forecasts with no active storms (pre-season), scoring is trivial:
prediction = 0.0, outcome = no RI → correct null forecast.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from hazardpulse.data.http import fetch_bytes  # noqa: E402
from hazardpulse.hurricane.atcf import (  # noqa: E402
    ATCF_ROOT,
    ATCFRecord,
    DEFAULT_ANALYSIS_PRIORITY,
    parse_atcf_deck,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY_DIR = PROJECT_ROOT / "dist" / "data" / "replay"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "hurricane_prospective"

# RI definition: ≥30 kt increase in 24 hours (NHC standard)
RI_THRESHOLD_KT = 30.0
RI_WINDOW_HOURS = 24

# ATCF best-track archive
BDECK_URL = "{root}/btk/b{storm_id}.dat"


def parse_utc(text: str) -> dt.datetime:
    return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)


def format_utc_z(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_best_track(storm_id: str) -> list[ATCFRecord]:
    """Fetch best-track b-deck data for a storm from NHC archives."""
    url = BDECK_URL.format(root=ATCF_ROOT, storm_id=storm_id.lower())
    try:
        data = fetch_bytes(url, namespace="atcf_bdeck", timeout=30)
        text = data.decode("utf-8", errors="replace")
        return parse_atcf_deck(text)
    except Exception:
        pass

    # Try gzipped version
    try:
        data = fetch_bytes(url + ".gz", namespace="atcf_bdeck", timeout=30)
        text = gzip.decompress(data).decode("utf-8", errors="replace")
        return parse_atcf_deck(text)
    except Exception as exc:
        print(f"  Warning: Could not fetch best-track for {storm_id}: {exc}")
        return []


def check_ri_occurred(
    records: list[ATCFRecord],
    issue_time: dt.datetime,
    window_hours: int = RI_WINDOW_HOURS,
    threshold_kt: float = RI_THRESHOLD_KT,
) -> tuple[bool, float | None, float | None]:
    """Check if rapid intensification occurred after issue_time.

    Returns (ri_occurred, vmax_at_issue, vmax_at_issue_plus_window).
    """
    window_end = issue_time + dt.timedelta(hours=window_hours)

    # Find analysis records (BEST track, tau=0)
    best_records = [
        r for r in records
        if r.model in ("BEST", "CARQ", "OFCL")
        and r.tau_hours == 0
        and r.vmax_kt is not None
    ]
    if not best_records:
        return False, None, None

    # Find vmax closest to issue time
    best_at_issue = min(
        best_records,
        key=lambda r: abs((r.cycle - issue_time).total_seconds()),
    )
    dt_issue = abs((best_at_issue.cycle - issue_time).total_seconds()) / 3600.0
    if dt_issue > 12:  # too far from issue time
        return False, None, None

    vmax_issue = best_at_issue.vmax_kt

    # Find vmax closest to issue + 24h
    best_at_end = min(
        best_records,
        key=lambda r: abs((r.cycle - window_end).total_seconds()),
    )
    dt_end = abs((best_at_end.cycle - window_end).total_seconds()) / 3600.0
    if dt_end > 12:
        return False, vmax_issue, None

    vmax_end = best_at_end.vmax_kt

    dv = vmax_end - vmax_issue
    return dv >= threshold_kt, vmax_issue, vmax_end


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = np.sum(y_true == 1)
    neg = np.sum(y_true == 0)
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    tp = fp = 0.0
    tp_prev = fp_prev = 0.0
    auc = 0.0
    for label in y_sorted:
        if label == 1:
            tp += 1.0
        else:
            fp += 1.0
        tpr = tp / pos
        fpr = fp / neg
        auc += (fpr - fp_prev / neg) * (tpr + tp_prev / pos) / 2.0
        tp_prev = tp
        fp_prev = fp
    return float(auc)


def brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_score) - np.asarray(y_true)) ** 2))


def load_replay_artifacts(replay_dir: Path) -> list[dict]:
    artifacts: list[dict] = []
    for path in sorted(replay_dir.glob("hu_fcst_*.json")):
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        artifact["_path"] = str(path)
        artifacts.append(artifact)
    artifacts.sort(key=lambda a: a.get("issued_at", ""))
    return artifacts


def matured_artifacts(
    artifacts: list[dict], score_as_of: dt.datetime
) -> list[dict]:
    matured: list[dict] = []
    for artifact in artifacts:
        issued_at = parse_utc(artifact["issued_at"])
        # Hurricane forecasts use 24-hour RI window + 24h buffer for best-track
        mature_time = issued_at + dt.timedelta(hours=48)
        if mature_time <= score_as_of:
            matured.append(artifact)
    return matured


def _accumulate_calibration(calib_acc: dict, scores: np.ndarray, y_true: np.ndarray) -> None:
    """Pool (storm RI probability -> #positive, #total) into a histogram."""
    rscore = np.round(scores, 6)
    uniq, inv = np.unique(rscore, return_inverse=True)
    tot = np.bincount(inv)
    pos = np.bincount(inv, weights=y_true).astype(np.int64)
    for u, t, p in zip(uniq, tot, pos):
        key = float(u)
        slot = calib_acc.get(key)
        if slot is None:
            calib_acc[key] = [int(t), int(p)]
        else:
            slot[0] += int(t)
            slot[1] += int(p)


def write_calibration_dataset(output_dir: Path, calib_acc: dict, hazard: str = "hurricane") -> Path:
    keys = sorted(calib_acc.keys())
    total = [int(calib_acc[k][0]) for k in keys]
    pos = [int(calib_acc[k][1]) for k in keys]
    n = int(sum(total))
    payload = {
        "hazard": hazard,
        "n": n,
        "n_groups": len(keys),
        "base_rate": (sum(pos) / n) if n else 0.0,
        "scores": [round(float(k), 6) for k in keys],
        "pos": pos,
        "total": total,
    }
    path = output_dir / "calibration_dataset.json"
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return path


def score_single_forecast(artifact: dict, calib_acc: dict | None = None) -> dict:
    """Score a single hurricane forecast.

    For forecasts with storms: fetch best-track, check if RI occurred.
    For forecasts with no storms: correct null forecast (pred=0, label=0).
    """
    issued_at = parse_utc(artifact["issued_at"])
    storms = artifact.get("storms", [])
    forecast_id = artifact["forecast_id"]

    if not storms:
        # No active TCs — correct null forecast
        return {
            "forecast_id": forecast_id,
            "issued_at": artifact["issued_at"],
            "n_storms": 0,
            "n_ri_events": 0,
            "predictions": [],
            "auc": float("nan"),
            "brier": 0.0,  # predicted 0, observed 0 → perfect
            "null_forecast": True,
        }

    predictions: list[dict] = []

    for storm in storms:
        storm_id = storm.get("storm_id", "")
        pred_prob = float(storm.get("ri_probability", storm.get("result_probability", 0.0)))

        # Fetch best-track for this storm
        records = fetch_best_track(storm_id)
        ri_occurred, vmax_issue, vmax_end = check_ri_occurred(records, issued_at)

        predictions.append({
            "storm_id": storm_id,
            "storm_name": storm.get("storm_name", storm_id),
            "predicted_ri_probability": round(pred_prob, 4),
            "raw_ri_probability": round(float(storm.get("raw_probability", pred_prob)), 4),
            "ri_occurred": ri_occurred,
            "vmax_at_issue": vmax_issue,
            "vmax_at_end": vmax_end,
            "intensity_change_kt": (
                round(vmax_end - vmax_issue, 1)
                if vmax_issue is not None and vmax_end is not None
                else None
            ),
        })

    y_true = np.array([1.0 if p["ri_occurred"] else 0.0 for p in predictions])
    y_score = np.array([p["predicted_ri_probability"] for p in predictions])

    if calib_acc is not None:
        raw = np.array([p["raw_ri_probability"] for p in predictions], dtype=np.float64)
        _accumulate_calibration(calib_acc, raw, y_true)

    auc = compute_auc(y_true, y_score)
    bs = brier_score(y_true, y_score)

    return {
        "forecast_id": forecast_id,
        "issued_at": artifact["issued_at"],
        "n_storms": len(storms),
        "n_ri_events": int(np.sum(y_true)),
        "predictions": predictions,
        "auc": auc,
        "brier": bs,
        "null_forecast": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score matured hurricane replay artifacts against best-track.",
    )
    parser.add_argument(
        "--replay-dir", type=Path, default=DEFAULT_REPLAY_DIR,
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--score-as-of", default=None,
        help="UTC timestamp for determining maturity. Defaults to now.",
    )
    parser.add_argument(
        "--emit-calibration", action="store_true",
        help="Pool per-storm (probability -> outcome) into calibration_dataset.json "
        "for the calibrator fitter (scripts/fit_calibration.py).",
    )
    args = parser.parse_args(argv)

    score_as_of = (
        parse_utc(args.score_as_of)
        if args.score_as_of
        else dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    )

    artifacts = load_replay_artifacts(args.replay_dir)
    matured = matured_artifacts(artifacts, score_as_of)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "scored_as_of": format_utc_z(score_as_of),
        "replay_dir": str(args.replay_dir.resolve()),
        "n_replay_artifacts": len(artifacts),
        "n_matured_forecasts": len(matured),
        "ri_threshold_kt": RI_THRESHOLD_KT,
        "ri_window_hours": RI_WINDOW_HOURS,
        "outcome_source": "NHC ATCF best-track",
        "status": "ok",
    }

    calib_acc: dict | None = {} if args.emit_calibration else None

    per_forecast_path = output_dir / "per_forecast_scores.jsonl"
    per_forecast_results: list[dict] = []

    if matured:
        with open(per_forecast_path, "w", encoding="utf-8") as handle:
            for artifact in matured:
                result = score_single_forecast(artifact, calib_acc=calib_acc)
                per_forecast_results.append(result)
                handle.write(json.dumps(result) + "\n")

        if calib_acc is not None:
            calib_path = write_calibration_dataset(output_dir, calib_acc, hazard="hurricane")
            summary["calibration_dataset"] = str(calib_path)
            summary["calibration_n"] = int(sum(slot[0] for slot in calib_acc.values()))

        n_null = sum(1 for r in per_forecast_results if r.get("null_forecast"))
        n_with_storms = sum(1 for r in per_forecast_results if not r.get("null_forecast"))
        aucs = [r["auc"] for r in per_forecast_results if math.isfinite(r["auc"])]
        briers = [r["brier"] for r in per_forecast_results if math.isfinite(r["brier"])]
        total_storms = sum(r["n_storms"] for r in per_forecast_results)
        total_ri = sum(r["n_ri_events"] for r in per_forecast_results)

        summary.update({
            "n_null_forecasts": n_null,
            "n_forecasts_with_storms": n_with_storms,
            "total_storms_scored": total_storms,
            "total_ri_events": total_ri,
            "ri_rate": round(total_ri / max(total_storms, 1), 4),
            "mean_auc": round(float(np.mean(aucs)), 4) if aucs else None,
            "median_auc": round(float(np.median(aucs)), 4) if aucs else None,
            "mean_brier": round(float(np.mean(briers)), 4) if briers else None,
            "n_forecasts_with_valid_auc": len(aucs),
        })
    else:
        summary["status"] = "waiting_for_matured_forecasts"
        summary["message"] = (
            "No hurricane replay artifacts have fully matured yet."
        )
        per_forecast_path.write_text("", encoding="utf-8")

    summary_path = output_dir / "prospective_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print()
    print("Hurricane prospective scoring")
    print(f"  Replay dir:          {args.replay_dir.resolve()}")
    print(f"  Matured forecasts:   {len(matured)} / {len(artifacts)}")
    print(f"  Summary:             {summary_path}")
    if matured:
        print(f"  Per-forecast scores: {per_forecast_path}")
        print(f"  Null forecasts:      {summary.get('n_null_forecasts')}")
        print(f"  With storms:         {summary.get('n_forecasts_with_storms')}")
        print(f"  Mean AUC:            {summary.get('mean_auc')}")
        print(f"  Mean Brier:          {summary.get('mean_brier')}")
    else:
        print("  Waiting for matured forecast windows.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
