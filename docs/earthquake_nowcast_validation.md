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
| Teleseismic dynamic triggering | ΔAUC +0.007 (2024) / +0.002 (2025), ns | faded on the larger holdout |
| Natural-time seismic clock | ΔAUC **+0.011 (2024) / +0.010 (2025)**, CI lower bound ≈ 0 | **consistent ~+0.01 across two independent holdouts** — real but marginal, borderline significant. Worth folding into Block S at the next retrain (the re-extraction happens there anyway). |

**Conclusion of the search:** the seismicity catalog is the only globally-complete
signal, which is why the model is seismicity-driven. Exotic instrument data is either
coverage-starved (GNSS, geomagnetic, ionospheric — sparse networks vs offshore quakes)
or too weak (tidal). The realistic levers are **more catalog data** (proven) and
**better seismicity features** (the natural-time clock is the live candidate).

## Operational reality check — recent real M6+ events (the hard truth)

`scripts/backtest_recent_earthquakes.py` scored every M6+ event of the last 30 days
(June 2026) at its true location/time with the deployed model, using only prior data.
This is the "would we have predicted them, and where?" test, and it is humbling:

- **Mean epicenter rank: 47th percentile** among globally-active cells — essentially
  random. The model does **not** reliably pick which active region ruptures next.
- Only **4/19 (21%)** epicenters landed in the top quintile.
- **But 74% beat their own quiet-time control** — the precursory signal is real: a
  location's pre-mainshock setting does look more critical than a random earlier time.
- Spread of outcomes: the **M7.5 Venezuela was nailed** (top 1%, regional peak 0 km),
  while the **M7.8 Philippines (17th pct) and M6.9 Japan (17th pct) would have been
  missed.**

**Reconciliation:** the case-control AUC (0.77, +0.079 over persistence) is a real
*statistical* skill on balanced same-location pairs. The operational task — localizing
the next rupture among *all* active regions — is far harder, and there the model is
near random. Both are true. The precursory signal exists but is not specific enough to
localize.

**Definitive operational number** (`backtest_operational_grid.py`, declustered forward
labels, 3 reference times, 450 active-cell forecasts): **pooled operational AUC = 0.509
(random).** Per-snapshot: 0.54 / 0.54 / 0.33. As a "which active region gets the next
M6+ within 300 km in the next 365 days" FORECASTER, the model has **no skill**. The
0.77 nowcast scores positives AT the mainshock moment (precursors peak); the operational
forecast scores at arbitrary times months ahead, where the short-lived signal is absent.
The model is a short-term NOWCAST, not an operational forecaster -- and is presented as
such.

**Data-lever result** (M5.5, 2.7x more samples): the ~0.76 nowcast ceiling holds, but
the edge over persistence *strengthens* (+0.087, CI [0.064,0.109], 3x more seed-stable).
**Deep representation learning** on raw event sequences (GRU+attention, incl. raw depth)
learns real signal (0.73) but loses to the hand-crafted GBT (0.77) -- domain knowledge
beats raw learning at this data size. **External forces** (tidal/celestial/moon, +0.007;
teleseismic; seasonality) are all non-significant nulls. The catalog seismicity is the
signal; the nowcast ceiling is ~0.76; operational forecasting is unsolved here as
everywhere.

**Consequence:** the earthquake nowcast is presented as an honest, calibrated,
research-grade nowcast with a real statistical edge — **explicitly NOT operational
prediction**, and **not** a front-and-center "we predict earthquakes" claim. The
recent-events table is published as honest evidence, hits and misses alike.

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
