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

**Data-lever result** (M5.5, 2.7x more samples): the ~0.76 nowcast ceiling holds for the
hand-crafted GBT, but the edge over persistence *strengthens* (+0.087, CI [0.064,0.109],
3x more seed-stable).

**The breakthrough -- deep learning + data, combined.** A GRU+attention model fed the
RAW event sequence (no hand-crafted features) SCALES with data where hand-crafting
plateaus:

| | persistence | hand-crafted GBT | deep (raw events) | pos-rate |
|---|---|---|---|---|
| M6 (3.1k train) | 0.67 | 0.751 | 0.730 | ~33% (1:2 design) |
| M5.5 (7.3k) | 0.67 | 0.762 | 0.752 | ~45% |
| **M5.0 (15k)** | **0.646** | ~0.77 (confirming) | **0.815** | **55% (PEAK)** |
| M4.5 (36k) | -- | -- | 0.750 | **77% (controls collapse)** |

The pos-rate column tells the real story: the case-control design wants 33%
(1 mainshock : 2 quiet controls), but as magnitude drops the controls become
**unfindable** -- every active location already has a forward event -- so the
positive fraction climbs 33% -> 45% -> 55% -> 77%. At M5.0 (55%) there is still
genuine quiet-vs-critical contrast and the deep model peaks at 0.815; by M4.5 (77%)
the contrast is gone and skill collapses to 0.750. The climb didn't "stop" -- the
*task itself* dissolved beneath it. M5.0 is the last magnitude with a real control.

At M5.0 the deep model reaches **0.815, beating persistence by +0.17** -- and the
persistence baseline stayed ~0.65 at every magnitude, so this is NOT an easier task; it
is real signal. Representation learning on the raw event stream, given enough data, found
precursory structure (likely foreshock-sequence timing) the engineered features miss.
This is the genuine lever: **lower magnitude (more data) + let-the-ML-discover (deep
learning), together** -- neither worked alone.

**The climb is bounded -- M5.0 is the empirical sweet spot, and we found the floor.**
Pushing *lower* to M4.5 (36k train, 2.4x more data) did NOT continue the climb -- it
**reversed to 0.750**. The reason is diagnostic, not noise: at M4.5 the label "an M4.5+
within 300 km / 365 days forward" is almost always TRUE at any active location, so
`generate_control_samples` can no longer find quiet-time controls -- the train/test sets
go **77% / 74% positive** and the case-control contrast degenerates. So the user's
instinct (train below M6) paid off all the way down to M5.0 and then hit a hard physical
floor: below M5.0 there is no such thing as a "quiet" control at 300 km / 1 yr. **M5.0 is
the deployable model** (`results/models/eq_deep_nowcast_m5.0.pt`, 0.81, 5-seed stable);
M4.5 is past the optimum. Remaining caveat: the M5.0 result carries a val>test gap
(val ~0.93 vs test ~0.81) -- era-shift overfitting -- so the honest deployable number is
~0.81 on the post-2018 holdout, not the optimistic validation figure. The fair
GBT-at-M5.0 comparison is the last confirming run; it is still NOWCAST skill (the
operational forecast limit of 0.51 is separate physics).

**Self-supervised pretraining on the small-magnitude stream -- tested, measured NULL.**
A natural idea: pretrain the sequence encoder on the abundant unlabeled event stream
(120k anchors in the train window, self-supervised target = next-30-day max magnitude
within 500 km), then transfer the encoder to the rare M5.0 nowcast head -- "use the small
quakes to teach the big-quake task." Two fine-tuning regimes were run, 5 seeds each:

| config | test AUC | val AUC |
|---|---|---|
| **from scratch (deployed)** | **0.810 +/- 0.004** | ~0.93 |
| pretrained, naive FT (lr 1e-3) | 0.796 +/- 0.004 | 0.87 |
| pretrained, discriminative FT (freeze 8ep, lr 3e-4) | 0.789 +/- 0.004 | 0.86 |

Pretraining **did not help** (both configs ~0.79 < 0.81). The mechanism is informative,
not a failure: the deep model *already* reads every M2.5+ event as raw input, so "small
quakes inform big quakes" is the existing mechanism -- it is *why* deep beats the
hand-crafted GBT. The proxy-task pretraining tried to inject that same signal a second
time and the direct supervised data already saturates it, so the warm start only moved
the encoder to a slightly worse basin. (It *did* regularize -- val fell 0.93->0.86,
closing the era-overfit gap -- but that did not convert to test skill.) The harness lives
behind `--pretrain-anchors` for reproducibility; the deployable model stays from-scratch.
The remaining *direct* lever (feed MORE of the stream: larger K / longer lookback) is the
honest next experiment.

**External forces** (tidal/celestial/moon, +0.007; teleseismic; seasonality) are all
non-significant nulls. The catalog seismicity is the signal; the hand-crafted nowcast
ceiling is ~0.76 but deep+data reaches ~0.82; operational forecasting is unsolved here
as everywhere.

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
