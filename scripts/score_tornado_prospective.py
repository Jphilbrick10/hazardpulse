#!/usr/bin/env python3
"""Score frozen tornado replay artifacts against observed SPC tornado reports.

Reads replay artifacts from dist/data/replay/to_fcst_*.json, evaluates
only forecasts whose 24-hour windows have fully matured, fetches actual
tornado reports from the SPC storm reports CSV feed, and writes:

- results/tornado_prospective/prospective_summary.json
- results/tornado_prospective/per_forecast_scores.jsonl
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import math
import ssl
import sys
import urllib.request
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY_DIR = PROJECT_ROOT / "dist" / "data" / "replay"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "tornado_prospective"

# SPC storm reports CSV — daily archives
SPC_REPORTS_URL = "https://www.spc.noaa.gov/climo/reports/{date}_rpts_filtered_torn.csv"

# Matching criteria
MATCH_RADIUS_KM = 40.0  # spatial proximity threshold
MATCH_WINDOW_HOURS = 4.0  # temporal proximity threshold


def parse_utc(text: str) -> dt.datetime:
    """Parse timestamp to naive UTC datetime (handles multiple formats)."""
    text = text.strip()
    # ISO-8601: "2026-04-03T18:41:00Z"
    if text[:4].isdigit() and len(text) > 4 and text[4] == "-":
        return dt.datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    # SPC-style: "20260403_183040 UTC"
    text = text.replace(" UTC", "").replace(" utc", "")
    if "_" in text and len(text) >= 15:
        return dt.datetime.strptime(text[:15], "%Y%m%d_%H%M%S")
    if "_" in text and len(text) >= 13:
        return dt.datetime.strptime(text[:13], "%Y%m%d_%H%M")
    # Fallback
    return dt.datetime.strptime(text[:14], "%Y%m%d%H%M%S")


def format_utc_z(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometers."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def fetch_spc_reports_for_date(date: dt.date) -> list[dict]:
    """Fetch SPC filtered tornado reports for a single date.

    Returns list of dicts with keys: time, lat, lon, mag, location, state.
    SPC CSV format: Time,F_Scale,Location,County,State,Lat,Lon,Comments
    """
    date_str = date.strftime("%y%m%d")
    url = SPC_REPORTS_URL.format(date=date_str)

    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "HazardPulse/1.0 (research)"}
        )
        ctx = ssl.create_default_context()
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        print(f"  Warning: Could not fetch SPC reports for {date}: {exc}")
        return []

    reports: list[dict] = []
    reader = csv.reader(io.StringIO(raw))
    header = None
    for row in reader:
        if header is None:
            header = [col.strip().lower() for col in row]
            continue
        if len(row) < len(header):
            continue
        rec = dict(zip(header, row))
        try:
            lat = float(rec.get("lat", 0))
            lon = float(rec.get("lon", 0))
            time_str = rec.get("time", "0000").strip()
            mag_str = rec.get("f_scale", rec.get("mag", "-1")).strip()
            # Handle EF-scale strings like "EF1" or just "1"
            mag_str = mag_str.replace("EF", "").replace("ef", "")
            mag = int(mag_str) if mag_str.lstrip("-").isdigit() else -1

            # Parse time (HHMM format)
            if len(time_str) >= 4:
                hour = int(time_str[:2])
                minute = int(time_str[2:4])
            else:
                hour, minute = 0, 0

            report_time = dt.datetime(date.year, date.month, date.day, hour, minute)

            if abs(lat) < 0.01 and abs(lon) < 0.01:
                continue

            reports.append({
                "time": format_utc_z(report_time),
                "lat": lat,
                "lon": lon,
                "mag": mag,
                "location": rec.get("location", ""),
                "state": rec.get("state", ""),
            })
        except (ValueError, TypeError):
            continue

    return reports


def fetch_spc_reports_range(
    start: dt.datetime, end: dt.datetime
) -> list[dict]:
    """Fetch SPC tornado reports for a date range."""
    all_reports: list[dict] = []
    current = start.date()
    end_date = end.date()

    while current <= end_date:
        reports = fetch_spc_reports_for_date(current)
        all_reports.extend(reports)
        current += dt.timedelta(days=1)

    print(f"  Fetched {len(all_reports)} tornado reports from SPC ({start.date()} to {end_date})")
    return all_reports


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


def brier_skill_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    bs = brier_score(y_true, y_score)
    clim = float(np.mean(y_true))
    bs_clim = float(np.mean((clim - y_true) ** 2))
    if bs_clim < 1e-12:
        return 0.0
    return 1.0 - bs / bs_clim


def load_replay_artifacts(replay_dir: Path) -> list[dict]:
    artifacts: list[dict] = []
    for path in sorted(replay_dir.glob("to_fcst_*.json")):
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
        horizon_hours = int(artifact.get("forecast_horizon_hours", 24))
        mature_time = issued_at + dt.timedelta(hours=horizon_hours)
        if mature_time <= score_as_of:
            matured.append(artifact)
    return matured


def _accumulate_calibration(calib_acc: dict, scores: np.ndarray, y_true: np.ndarray) -> None:
    """Pool (storm probability -> #positive, #total) into a histogram."""
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


def write_calibration_dataset(output_dir: Path, calib_acc: dict, hazard: str = "tornado") -> Path:
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


def score_single_forecast(
    artifact: dict,
    tornado_reports: list[dict],
    calib_acc: dict | None = None,
) -> dict:
    """Score a single tornado forecast against observed reports.

    Each storm in the forecast is labeled: did a tornado report occur
    within MATCH_RADIUS_KM and MATCH_WINDOW_HOURS of the storm?
    """
    issued_at = parse_utc(artifact["issued_at"])
    horizon_hours = int(artifact.get("forecast_horizon_hours", 24))
    window_end = issued_at + dt.timedelta(hours=horizon_hours)

    storms = artifact.get("storms", [])
    if not storms:
        return {
            "forecast_id": artifact["forecast_id"],
            "issued_at": artifact["issued_at"],
            "window_end": format_utc_z(window_end),
            "n_storms": 0,
            "n_reports_in_window": 0,
            "n_matched_storms": 0,
            "auc": float("nan"),
            "brier": float("nan"),
            "brier_skill_score": float("nan"),
        }

    # Filter reports to forecast window
    reports_in_window = []
    for report in tornado_reports:
        rtime = parse_utc(report["time"])
        if issued_at <= rtime <= window_end:
            reports_in_window.append(report)

    # For each storm, check if any report matches
    y_true = np.zeros(len(storms), dtype=np.float64)
    y_score = np.zeros(len(storms), dtype=np.float64)

    for i, storm in enumerate(storms):
        prob = float(storm.get("tornado_probability", 0.0))
        y_score[i] = prob
        storm_lat = float(storm.get("lat", 0))
        storm_lon = float(storm.get("lon", 0))
        storm_time = parse_utc(storm.get("valid_time", artifact["issued_at"]))

        for report in reports_in_window:
            dist = haversine_km(storm_lat, storm_lon, report["lat"], report["lon"])
            dt_hours = abs((parse_utc(report["time"]) - storm_time).total_seconds()) / 3600.0
            if dist <= MATCH_RADIUS_KM and dt_hours <= MATCH_WINDOW_HOURS:
                y_true[i] = 1.0
                break

    if calib_acc is not None:
        # Pool the RAW model score (not the deployed/calibrated one) so re-fitting
        # never double-calibrates. Before any calibrator exists raw == probability.
        raw = np.array(
            [float(s.get("raw_probability", s.get("tornado_probability", 0.0))) for s in storms],
            dtype=np.float64,
        )
        _accumulate_calibration(calib_acc, raw, y_true)

    n_matched = int(np.sum(y_true))
    auc = compute_auc(y_true, y_score)
    bs = brier_score(y_true, y_score)
    bss = brier_skill_score(y_true, y_score)

    return {
        "forecast_id": artifact["forecast_id"],
        "issued_at": artifact["issued_at"],
        "window_end": format_utc_z(window_end),
        "n_storms": len(storms),
        "n_reports_in_window": len(reports_in_window),
        "n_matched_storms": n_matched,
        "match_rate": round(n_matched / len(storms), 4) if storms else 0.0,
        "base_rate": round(n_matched / len(storms), 4) if storms else 0.0,
        "auc": auc,
        "brier": bs,
        "brier_skill_score": bss,
        "top_probability": float(np.max(y_score)) if len(y_score) > 0 else 0.0,
        "scoring_tier": artifact.get("scoring_tier", "unknown"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score matured tornado replay artifacts against SPC reports.",
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
        "--issued-after", default=None,
        help=("Only include forecasts issued AT or AFTER this UTC timestamp. "
              "Used to isolate metrics from a specific deployment / fix. "
              "Example: --issued-after 2026-04-28T13:00:00Z"),
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

    issued_after_dt: dt.datetime | None = None
    if args.issued_after:
        issued_after_dt = parse_utc(args.issued_after)
        before = len(matured)
        matured = [
            a for a in matured
            if parse_utc(a["issued_at"]) >= issued_after_dt
        ]
        print(f"  Filter --issued-after {args.issued_after}: "
              f"kept {len(matured)} of {before} matured forecasts")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "scored_as_of": format_utc_z(score_as_of),
        "replay_dir": str(args.replay_dir.resolve()),
        "n_replay_artifacts": len(artifacts),
        "n_matured_forecasts": len(matured),
        "forecast_horizon_hours": 24,
        "match_radius_km": MATCH_RADIUS_KM,
        "match_window_hours": MATCH_WINDOW_HOURS,
        "outcome_source": "SPC filtered tornado reports",
        "status": "ok",
    }

    calib_acc: dict | None = {} if args.emit_calibration else None

    per_forecast_path = output_dir / "per_forecast_scores.jsonl"
    per_forecast_results: list[dict] = []

    if matured:
        # Determine date range to fetch SPC reports
        earliest = min(parse_utc(a["issued_at"]) for a in matured)
        latest = max(
            parse_utc(a["issued_at"])
            + dt.timedelta(hours=int(a.get("forecast_horizon_hours", 24)))
            for a in matured
        )
        print(f"Fetching SPC tornado reports for {earliest.date()} to {latest.date()}...")
        all_reports = fetch_spc_reports_range(earliest, latest)

        with open(per_forecast_path, "w", encoding="utf-8") as handle:
            for artifact in matured:
                result = score_single_forecast(artifact, all_reports, calib_acc=calib_acc)
                per_forecast_results.append(result)
                handle.write(json.dumps(result) + "\n")

        if calib_acc is not None:
            calib_path = write_calibration_dataset(output_dir, calib_acc, hazard="tornado")
            summary["calibration_dataset"] = str(calib_path)
            summary["calibration_n"] = int(sum(slot[0] for slot in calib_acc.values()))

        aucs = [r["auc"] for r in per_forecast_results if math.isfinite(r["auc"])]
        briers = [r["brier"] for r in per_forecast_results if math.isfinite(r["brier"])]
        bss_vals = [r["brier_skill_score"] for r in per_forecast_results if math.isfinite(r["brier_skill_score"])]
        total_storms = sum(r["n_storms"] for r in per_forecast_results)
        total_matched = sum(r["n_matched_storms"] for r in per_forecast_results)
        total_reports = sum(r["n_reports_in_window"] for r in per_forecast_results)

        # Split metrics by scoring tier so tier2 physics-only runs don't
        # contaminate the tier1 ML-model's measured accuracy.
        by_tier: dict[str, list[dict]] = {}
        for r in per_forecast_results:
            by_tier.setdefault(r.get("scoring_tier", "unknown"), []).append(r)

        per_tier_metrics: dict[str, dict] = {}
        for tier, rs in by_tier.items():
            t_aucs = [r["auc"] for r in rs if math.isfinite(r["auc"])]
            t_briers = [r["brier"] for r in rs if math.isfinite(r["brier"])]
            t_bss = [r["brier_skill_score"] for r in rs if math.isfinite(r["brier_skill_score"])]
            t_storms = sum(r["n_storms"] for r in rs)
            t_matched = sum(r["n_matched_storms"] for r in rs)
            per_tier_metrics[tier] = {
                "n_forecasts": len(rs),
                "n_forecasts_with_valid_auc": len(t_aucs),
                "total_storms_scored": t_storms,
                "total_matched_storms": t_matched,
                "match_rate": round(t_matched / max(t_storms, 1), 4),
                "mean_auc": round(float(np.mean(t_aucs)), 4) if t_aucs else None,
                "median_auc": round(float(np.median(t_aucs)), 4) if t_aucs else None,
                "mean_brier": round(float(np.mean(t_briers)), 4) if t_briers else None,
                "mean_brier_skill_score": round(float(np.mean(t_bss)), 4) if t_bss else None,
            }

        # Recovery-curve buckets: roll up per-tier metrics by recency window.
        # Each bucket is keyed by the cutoff timestamp; "all_time" includes
        # everything matured. The "last_7d" / "last_3d" / "last_24h" windows
        # let us watch the AUC trend after a model fix lands.
        now_dt = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
        recency_windows = [
            ("last_24h", dt.timedelta(hours=24)),
            ("last_3d", dt.timedelta(days=3)),
            ("last_7d", dt.timedelta(days=7)),
            ("last_14d", dt.timedelta(days=14)),
        ]
        per_tier_recovery: dict[str, dict[str, dict]] = {}
        for tier, rs in by_tier.items():
            tier_recovery: dict[str, dict] = {}
            for label, delta in recency_windows:
                cutoff = now_dt - delta
                window_rs = [
                    r for r in rs
                    if parse_utc(r["issued_at"]) >= cutoff
                ]
                w_aucs = [r["auc"] for r in window_rs if math.isfinite(r["auc"])]
                w_briers = [r["brier"] for r in window_rs if math.isfinite(r["brier"])]
                tier_recovery[label] = {
                    "n_forecasts": len(window_rs),
                    "mean_auc": round(float(np.mean(w_aucs)), 4) if w_aucs else None,
                    "mean_brier": round(float(np.mean(w_briers)), 4) if w_briers else None,
                }
            per_tier_recovery[tier] = tier_recovery

        summary.update({
            "observed_window": {
                "start": format_utc_z(earliest),
                "end": format_utc_z(latest),
                "n_tornado_reports": len(all_reports),
            },
            "total_storms_scored": total_storms,
            "total_matched_storms": total_matched,
            "total_tornado_reports_in_windows": total_reports,
            "overall_match_rate": round(total_matched / max(total_storms, 1), 4),
            # Aggregated metrics — kept for backwards compatibility. Prefer
            # by_tier for anything driving decisions, since tier1_ml (GBT)
            # and tier2_analytic (physics) have very different calibration.
            "mean_auc": round(float(np.mean(aucs)), 4) if aucs else None,
            "median_auc": round(float(np.median(aucs)), 4) if aucs else None,
            "mean_brier": round(float(np.mean(briers)), 4) if briers else None,
            "mean_brier_skill_score": round(float(np.mean(bss_vals)), 4) if bss_vals else None,
            "n_forecasts_with_valid_auc": len(aucs),
            "n_forecasts_without_matches": sum(
                1 for r in per_forecast_results if r["n_matched_storms"] == 0
            ),
            "by_tier": per_tier_metrics,
            "by_tier_recovery": per_tier_recovery,
            "issued_after_filter": (
                args.issued_after if issued_after_dt is not None else None
            ),
        })
    else:
        summary["status"] = "waiting_for_matured_forecasts"
        summary["message"] = (
            "No tornado replay artifacts have fully matured yet."
        )
        per_forecast_path.write_text("", encoding="utf-8")

    summary_path = output_dir / "prospective_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    # Also publish a worker-served subset focused on the recovery curve so
    # /verification/tornado/ can render it without loading the full payload.
    recovery_subset = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).replace(tzinfo=None).isoformat() + "Z",
        "scored_as_of": summary.get("scored_as_of"),
        "n_matured_forecasts": summary.get("n_matured_forecasts", 0),
        "issued_after_filter": summary.get("issued_after_filter"),
        "by_tier": summary.get("by_tier", {}),
        "by_tier_recovery": summary.get("by_tier_recovery", {}),
        "fix_landed_at": "2026-04-28T13:50:00Z",  # HRRR + mlcape fix commit
    }
    worker_path = (
        Path(__file__).resolve().parents[1] / "dist" / "data" / "tornado-recovery.json"
    )
    worker_path.parent.mkdir(parents=True, exist_ok=True)
    worker_path.write_text(
        json.dumps(recovery_subset, indent=2) + "\n", encoding="utf-8"
    )

    print()
    print("Tornado prospective scoring")
    print(f"  Replay dir:          {args.replay_dir.resolve()}")
    print(f"  Matured forecasts:   {len(matured)} / {len(artifacts)}")
    print(f"  Summary:             {summary_path}")
    if matured:
        print(f"  Per-forecast scores: {per_forecast_path}")
        print(f"  Mean AUC:            {summary.get('mean_auc')}")
        print(f"  Mean Brier:          {summary.get('mean_brier')}")
        print(f"  Mean BSS:            {summary.get('mean_brier_skill_score')}")
        print(f"  Match rate:          {summary.get('overall_match_rate')}")
    else:
        print("  Waiting for matured forecast windows.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
