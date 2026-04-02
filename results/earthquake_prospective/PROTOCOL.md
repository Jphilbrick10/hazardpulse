# Earthquake Prospective Protocol

## Purpose

This directory tracks the forward-looking benchmark for the live earthquake
forecast pipeline in `scripts/fetch_and_score_earthquake.py`.

The goal is not to claim operational readiness. The goal is to freeze each
forecast, wait for the forecast window to mature, and then score it against
observed `M6.0+` earthquakes with no hindsight edits.

## Forecast Definition

- hazard: earthquake
- model: `eq_coherence_v1_0`
- domain: global `2 deg x 2 deg` grid
- target: `M6.0+`
- horizon: `30 days`
- default probability for unscored cells: `0.0`
- active cells: cells with at least `5` recent `M2.5+` events in the last `30`
  days, scored using a `400` day history window for feature computation

## Freeze / Replay Artifacts

Each live run writes:

- `dist/data/replay/eq_fcst_YYYYMMDD_HH00.json`
- `dist/data/earthquake-ledger.jsonl`

The replay artifact contains:

- issue time
- model version
- forecast domain metadata
- source catalog window metadata
- all active cell probabilities for that forecast

That is the frozen object later used for scoring.

## Scoring

Run:

```bash
python scripts/score_earthquake_prospective.py
```

Outputs:

- `results/earthquake_prospective/prospective_summary.json`
- `results/earthquake_prospective/per_forecast_scores.jsonl`
- `results/earthquake_prospective/grid_forecasts/*_forecast.csv`
- `results/earthquake_prospective/observed_events/*_observed.csv`

Only forecasts whose full 30-day windows have matured are scored.

## Current Metrics

Per matured forecast the scorer computes:

- cell-wise AUC
- cell-wise PR-AUC
- Brier score
- Poisson log-likelihood
- information gain per event relative to a uniform-rate baseline
- top-1 / top-5 / top-10 / top-20 hit flags

## Backfill Mode

Historical pseudo-prospective backfills can be created by issuing the live
forecaster at an explicit past timestamp:

```bash
python scripts/fetch_and_score_earthquake.py ^
  --issue-time 2026-02-01T00:00:00Z ^
  --replay-dir results/earthquake_prospective/replay ^
  --ledger-path results/earthquake_prospective/backfill-ledger.jsonl ^
  --skip-site ^
  --skip-live-pulse ^
  --skip-replay-index
```

Then score that replay directory:

```bash
python scripts/score_earthquake_prospective.py ^
  --replay-dir results/earthquake_prospective/replay ^
  --output-dir results/earthquake_prospective/backfill_scores
```

For multi-forecast archives, use the batch driver:

```bash
python scripts/backfill_earthquake_prospective.py ^
  --start 2026-01-01T00:00:00Z ^
  --end 2026-01-22T00:00:00Z ^
  --step-hours 168 ^
  --replay-dir results/earthquake_prospective/sample_replay ^
  --ledger-path results/earthquake_prospective/sample-ledger.jsonl ^
  --output-dir results/earthquake_prospective/sample_scores ^
  --score-as-of 2026-04-02T00:00:00Z
```

Add `--skip-existing` to resume a partially built archive without regenerating
existing replay artifacts.

## Not Yet CSEP-Complete

This protocol is a serious step toward prospective evaluation, but it is not a
full CSEP submission yet.

Missing pieces:

- direct ETAS / STEP side-by-side baselines in the same files
- pyCSEP-native forecast packaging and official N/L/S/M tests
- fixed magnitude-bin rate forecasts beyond the single `M6.0+` target
- public governance around forecast lock, release cadence, and threshold use

## Early Signal From The First Weekly Backfill Sample

The first small January 2026 weekly sample is intentionally not presented as a
claim of skill. It is a workflow proof plus an early warning that calibration
needs work:

- ranking AUC was respectable on the small sample
- Poisson information gain versus a uniform-rate baseline was negative
- top-1 and top-5 hit rates were zero in that sample

That means the archive/scoring machinery is working and is already useful:
it can reject overconfident forecast formulations before any public-facing
claims are made.

## Safety Note

This is still research-only. A frozen forecast ledger is necessary for trust,
but not sufficient for public warning use.
