# Earthquake Nowcast — Validation & Honest Assessment

*Auditable record of the rigorous validation of the HazardPulse earthquake nowcast.
Every number here is reproducible from the committed harnesses + reports.*

## What the model does (and does not) claim

- **Task:** a *nowcast* — given the seismicity history up to time *t* at a location,
  estimate whether that location/time is an M6+ mainshock setting vs a control
  (same location, different time). It is **not** a deterministic prediction and
  **not** an official forecast.
- **It does not beat USGS.** USGS does not make short-term earthquake predictions
  (they state quakes can't be predicted) — they publish probabilistic hazard maps and
  aftershock (ETAS) forecasts. This is a *different task* nobody operationally does
  well. The honest claim is "research-grade nowcast that significantly beats the
  baselines a seismologist would demand," not "better than USGS."

## 1. No leakage — features are strictly causal

Both feature blocks were audited line-by-line:

- **Block S** (`compute_block_s`): `time_mask = (cat.times >= t_start) & (cat.times < ev_time)`
  — only events strictly *before* the forecast time, within 5 yr / 500 km.
- **Block C** (`compute_block_c` → `extract_coherence_features`): `if t < t_min or t > ref_epoch: skip`
  — only events ≤ ref_epoch.

The features physically cannot see past the forecast time. Labels are defined by
declustering the full catalog, which is correct (labels are ground truth, not inputs).

## 2. It beats the honest baseline — significantly

The real bar in earthquake forecasting is **not 0.5** — "it shakes where it recently
shook" (smoothed seismicity / clustering) is strong. On the held-out test set
(`scripts/backtest_earthquake.py`):

| Predictor | Test AUC |
|---|---|
| Persistence (best recent-rate feature, `rate_7_30`) | 0.670 |
| Best single feature (`quiescence_7d`) | 0.675 |
| **Model (xgboost, 5 seeds)** | **0.751 ± 0.003** |

- **Edge over persistence: +0.079, 95% bootstrap CI [0.040, 0.121]** — excludes zero.
  The model is **not** a relabelled clustering baseline; it adds real precursory signal
  (quiescence, b-value trends, accelerating moment release, coherence).
- **Seed std 0.003** — the result is not a lucky split.
- **Ablation:** seismicity-only 0.742 → +coherence (CFT) 0.749 → +cross-terms 0.749.
  Coherence-field features add a small, real lift on top of seismicity.

## 3. More data helps (signature of real signal)

Retraining with the 2025 catalog folded in raised the holdout AUC from **0.757 → 0.774**.
Overfit noise does not improve with more data; real signal does.

## 4. New-signal search — what was tested and what it showed

Key constraint: the controls share the positive's *location*, so only
**time-varying-at-location** signals can help. That rules out static fields.

| Candidate | Result | Notes |
|---|---|---|
| GSRM strain rate, fault distance (static) | cannot help | identical for a quake and its same-place control |
| GNSS crustal deformation (2.5 GB cached) | coverage-limited | only ~5–10% of M6+ events have a land GPS station within 50–100 km (most are offshore subduction); a *regional* lever, not global |
| Tidal stress (fortnightly/anomalistic/semidiurnal) | ΔAUC −0.003, CI [−0.014,+0.009] | null — the ~1% tidal triggering washes out |
| Teleseismic dynamic triggering | ΔAUC +0.007, CI [−0.005,+0.019] | small, not significant |
| Natural-time seismic clock | ΔAUC +0.011, CI [−0.003,+0.025] | best lead; physically motivated; borderline (re-tested on the larger 2025 holdout) |

**Conclusion of the search:** the seismicity catalog is the only globally-complete
signal, which is why the model is seismicity-driven. Exotic instrument data is either
coverage-starved (GNSS, geomagnetic, ionospheric — sparse networks vs offshore quakes)
or too weak (tidal). The realistic levers are **more catalog data** (proven) and
**better seismicity features** (the natural-time clock is the live candidate).

## How it compares to the field

- **vs USGS:** different task; USGS doesn't claim short-term prediction. Not comparable
  as "better/worse" — it's a nowcast of something USGS doesn't operationally forecast.
- **vs research seismicity nowcasting (ETAS, pattern-informatics, natural-time):**
  competitive — a 0.75–0.77 case-control AUC that significantly beats smoothed
  seismicity is in line with the better research results on this hard problem.
- **Honest caveat:** M6+ events are *rare per cell*, so even good ranking yields low
  precision at operational thresholds. The value is in honest, calibrated, *signed and
  replayable* probabilities — the trust layer — not in claiming certainty.

## Reproduce

```
python scripts/backtest_earthquake.py --seeds 5 --breakdown        # baselines, edge, ablation
python scripts/backtest_augment_earthquake.py --candidates tidal natclock teleseismic
python scripts/retrain_and_deploy.py --hazard earthquake --max-year 2025 --deploy
```
Reports land in `results/calibration/earthquake_backtest.json`,
`earthquake_augment*.json`, `earthquake_retrain_report.json`.
