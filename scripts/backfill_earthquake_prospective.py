#!/usr/bin/env python3
"""Generate a batch of historical earthquake forecasts and score them.

This is a convenience driver around:
- `fetch_and_score_earthquake.py`
- `score_earthquake_prospective.py`
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fetch_and_score_earthquake import run_pipeline
from score_earthquake_prospective import main as score_main
from hazardpulse.earthquake.prospective import (
    forecast_id_for_time,
    format_utc_z,
    parse_utc_datetime,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPLAY_DIR = PROJECT_ROOT / "results" / "earthquake_prospective" / "replay"
DEFAULT_LEDGER_PATH = PROJECT_ROOT / "results" / "earthquake_prospective" / "backfill-ledger.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "earthquake_prospective" / "backfill_scores"


def iter_issue_times(
    start: dt.datetime,
    end: dt.datetime,
    *,
    step_hours: int,
) -> list[dt.datetime]:
    """Return regularly spaced UTC issue times on [start, end]."""
    step = dt.timedelta(hours=step_hours)
    current = start
    issue_times: list[dt.datetime] = []
    while current <= end:
        issue_times.append(current)
        current += step
    return issue_times


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backfill historical earthquake forecasts and score matured windows.",
    )
    parser.add_argument("--start", required=True, help="UTC start issue time (ISO-8601).")
    parser.add_argument("--end", required=True, help="UTC end issue time (ISO-8601).")
    parser.add_argument(
        "--step-hours",
        type=int,
        default=168,
        help="Spacing between issue times in hours (default: 168, weekly).",
    )
    parser.add_argument(
        "--replay-dir",
        type=Path,
        default=DEFAULT_REPLAY_DIR,
        help=f"Replay output directory (default: {DEFAULT_REPLAY_DIR})",
    )
    parser.add_argument(
        "--ledger-path",
        type=Path,
        default=DEFAULT_LEDGER_PATH,
        help=f"Ledger path (default: {DEFAULT_LEDGER_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Scoring output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--score-as-of",
        default=None,
        help="UTC timestamp used to decide which windows have matured. Defaults to now.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of generated forecasts.",
    )
    parser.add_argument(
        "--skip-score",
        action="store_true",
        help="Generate backfill forecasts only and skip the scoring pass.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip issue times whose replay artifacts already exist.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    start = parse_utc_datetime(args.start).replace(minute=0, second=0, microsecond=0)
    end = parse_utc_datetime(args.end).replace(minute=0, second=0, microsecond=0)
    if end < start:
        parser.error("--end must be on or after --start")
    if args.step_hours <= 0:
        parser.error("--step-hours must be positive")

    issue_times = iter_issue_times(start, end, step_hours=args.step_hours)
    if args.limit is not None:
        issue_times = issue_times[: args.limit]

    args.replay_dir.mkdir(parents=True, exist_ok=True)
    args.ledger_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generated: list[dict] = []
    n_skipped_existing = 0

    print("Earthquake backfill generation")
    print(f"  Start:        {format_utc_z(start)}")
    print(f"  End:          {format_utc_z(end)}")
    print(f"  Step hours:   {args.step_hours}")
    print(f"  Issue count:  {len(issue_times)}")
    print(f"  Replay dir:   {args.replay_dir.resolve()}")
    print()

    for index, issue_time in enumerate(issue_times, start=1):
        forecast_id = forecast_id_for_time(issue_time)
        replay_path = args.replay_dir / f"{forecast_id}.json"
        print(f"[{index}/{len(issue_times)}] {format_utc_z(issue_time)}")

        if args.skip_existing and replay_path.exists():
            print(f"  Skipping existing replay: {replay_path}")
            result = {
                "forecast_id": forecast_id,
                "issued_at": format_utc_z(issue_time),
                "replay_path": str(replay_path),
                "skipped_existing": True,
            }
            try:
                artifact = json.loads(replay_path.read_text(encoding="utf-8"))
                result["n_active_cells"] = artifact.get("n_active_cells")
                result["top_probability"] = artifact.get("top_probability")
            except Exception:
                pass
            generated.append(result)
            n_skipped_existing += 1
            print()
            continue

        result = run_pipeline(
            issue_time=issue_time,
            replay_dir=args.replay_dir,
            ledger_path=args.ledger_path,
            skip_site=True,
            skip_live_pulse=True,
            skip_replay_index=True,
        )
        generated.append(result)
        print()

    manifest = {
        "generated_at": format_utc_z(dt.datetime.now(dt.timezone.utc)),
        "start": format_utc_z(start),
        "end": format_utc_z(end),
        "step_hours": args.step_hours,
        "n_generated": len(generated),
        "n_skipped_existing": n_skipped_existing,
        "replay_dir": str(args.replay_dir.resolve()),
        "ledger_path": str(args.ledger_path.resolve()),
        "forecasts": generated,
    }
    manifest_path = args.output_dir / "backfill_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")

    if not args.skip_score:
        score_argv = [
            "--replay-dir",
            str(args.replay_dir),
            "--output-dir",
            str(args.output_dir),
        ]
        if args.score_as_of:
            score_argv.extend(["--score-as-of", args.score_as_of])
        score_main(score_argv)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
