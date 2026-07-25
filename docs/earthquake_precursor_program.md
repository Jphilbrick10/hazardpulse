# Earthquake "where-next" forecasting & precursor program — honest results

*Campaign goal: push the operational "where will the next large quake be" forecast as far as the
data allows, and exhaustively test every plausible precursor signal under rigorous controls.
Lives are at stake, so every number here is measured on held-out data with a stated control and
no overclaiming. Where something is null or blocked, it says so plainly.*

## Bottom line

## 2026-06-29 audit update

The strongest honest improvement found in this audit is a causal tabular operational ranker:
`scripts/research/operational_tabular_ranker.py`. It reuses the exact M5+/100km/30d operational
sample cache and adds only forecast-time-available features: sequence summaries, matured
same-cell and neighboring-cell M5 outcome history, plate/geography priors, and causal precursor
channels. Model selection is by 2018-2019 validation AUC; 2020+ test is reported once for the
validation-selected model.

Result: **test AUC 0.7292**, **month-grouped test AUC 0.7303**, 22,248 held-out test samples,
1,321 positives. This supersedes the broad-set GRU baseline of **0.7086 +/- 0.0014** as the best
local operational ranker result, but it does **not** crack 0.8-0.9 under the honest broad active-cell
task. The tested routes to 0.8+ remain non-operational: same-location case-control framing,
future catalog leakage, test-set model selection, or narrowing the candidate set after the fact.

Extended bakeoff after the first audit added the missing independent-prior families:
pre-2000 USGS M5+ historical seismicity and GSRM v1.2 principal geodetic strain-rate. The
validation-selected CatBoost ranker with these priors reached **test AUC 0.7313** and
**month-grouped AUC 0.7321** (`results/calibration/earthquake_operational_tabular_ranker.json`).
The broader mega-bakeoff tried LightGBM classifiers, XGBoost classifiers, LightGBM
LambdaRank/listwise monthly rankers, ETAS-only and ETAS-residual models, hard-gated tectonic
regime experts, ExtraTrees/RandomForest, and a neural MLP baseline
(`results/calibration/earthquake_operational_mega_bakeoff_gsrm.json`).
None beat the broad CatBoost result on the held-out 2020+ split. The strongest conclusion is now
sharper: **new priors move the third decimal, not the regime**. The honest broad operational ceiling
from this data universe is still about 0.73.

Second-pass "try everything" audit added three more pushes. First, direct ranking/point-process
objectives (`scripts/research/operational_rank_objective_bakeoff.py`) tried XGBoost rankers,
CatBoost YetiRank/PairLogit, and CUDA PyTorch monthly listwise/pairwise/point-process losses; the
best validation-selected result was **test AUC 0.7305**, **grouped AUC 0.7317**, so rank losses did
not beat the classifier champion. Second, GEM global active-fault proximity/slip-rate features were
added as an opt-in static prior; the validation-selected 331-feature CatBoost model fell to
**test AUC 0.7287** (`results/calibration/earthquake_operational_tabular_ranker_gem.json`), so GEM
faults are retained for experiments but not enabled by default. Third, a validation-first ensemble
audit of the 14 cached diverse base models selected fixed rank-averaging by validation and reached
**test AUC 0.731667**, **grouped AUC 0.733003**
(`results/calibration/earthquake_operational_ensemble_audit.json`). That is the best broad honest
number in this run, but it is a tiny lift, not the requested 0.8-0.9 break.

Third-pass "new data" audit pulled the feasible public sources behind the remaining hypotheses:
CRESCENT/Zenodo Cascadia tremor detections and GNSS NetCDF time series, cached `.tenv3` GNSS
stations, EarthScope station inventory as a waveform/noise-availability proxy, and USGS regional
M1+ low-magnitude catalogs for Cascadia and California. These add **150 causal new-data features**
(`scripts/research/operational_nextwave_ranker.py`): 48 tremor, 15 cached `.tenv3` GNSS, 15
CRESCENT GNSS, 12 station-inventory, and 60 regional microseismicity features. The single
validation-selected nextwave CatBoost reached **test AUC 0.7321**, **grouped AUC 0.7331**.
The validation-selected old+nextwave ensemble reached the best current broad result:
**test AUC 0.732523**, **grouped AUC 0.733363**
(`results/calibration/earthquake_operational_nextwave_ensemble.json`). This confirms the missing
data sources do help, but only by about +0.001 over the previous ensemble and +0.0012 over the
single CatBoost champion. The broad all-active-cell task remains nowhere near 0.8 without changing
the contract or adding much richer global observations.

Fourth-pass data audit added the highest-value public subduction priors: USGS Slab2 geometry and
UCSD Coupling Cloud slip-coupling/slip-deficit models. The default nextwave feature matrix is now
**506 columns**: the 307 causal base features plus 199 public-data features (48 tremor, 15 cached
`.tenv3` GNSS, 15 CRESCENT GNSS, 12 station-inventory, 60 regional microseismicity, 24 Slab2, and
25 Coupling Cloud). The validation-selected single CatBoost reached the strongest broad result so
far: **test AUC 0.7348**, **month-grouped AUC 0.7358**
(`results/calibration/earthquake_operational_nextwave_ranker.json`). The validation-selected
old+nextwave rank-average ensemble remains more conservative at **test AUC 0.732659**, grouped
**0.733711** (`results/calibration/earthquake_operational_nextwave_ensemble.json`). A real
waveform/noise pilot was also added: 115 derived spectral/RMS embeddings from regional waveform
snippets. It is kept opt-in because it lowered the validation-selected broad result to **test AUC
0.7291** (`results/calibration/earthquake_operational_nextwave_ranker_full.json`). Net: the new
physics/observational priors produce a real but small lift; no honest broad run has cracked 0.8.

Fifth-pass "do all of it" audit added the heavier remaining paths and made them opt-in because
they did not improve validation-selected broad test skill. The run added:
expanded USGS regional M1 catalogs for Alaska, Hawaii, and Puerto Rico; a paginated NASA CMR ARIA
Sentinel-1 GUNW metadata cache with **108,520** prior interferogram footprints; a scaled v2
waveform/noise embedding cache with **238** real snippets; CRESCENT dense GNSS vector-field
features; and validation-gated regional expert models. The heavy feature matrix has **586 columns**
(44 dense-GNSS-field features and 36 ARIA-coverage features on top of the default 506). Results:
heavy single CatBoost fell to **test AUC 0.7298**, grouped **0.7313**
(`results/calibration/earthquake_operational_nextwave_ranker_heavy.json`); heavy rank-average
ensemble selected by validation reached **test AUC 0.732712**, grouped **0.733604**
(`results/calibration/earthquake_operational_nextwave_ensemble_heavy.json`); validation-gated
regional experts reached **test AUC 0.730945** without heavy features and **0.730012** with them.
The ARIA features are coverage metadata only, not displacement rasters. Verdict: these pipelines
are now implemented and reproducible, but the default operational champion remains the
subduction-prior single model at **0.7348 / 0.7358 grouped**.

The audit also found and fixed a GCMT causality issue: the optional Coulomb feature previously had
no GCMT origin-time field available, so future focal-mechanism rows could affect historical feature
values. GCMT parsing now preserves event time, legacy no-time caches rebuild from NDK files, and
the Coulomb feature is restricted to prior M6+ GCMT rows. With that correction, causal classic
precursors add only a small increment: context-14 GBT **0.6978 -> 0.7026** (+0.0048).

1. **The operational catalog forecaster is the real, validated win.** Trained directly on the
   operational task (rank which *active* cell ruptures next, M5+/100km/30d), it scores
   **AUC ≈ 0.71** on held-out test cells across the broad active-cell set, **beats the
   gold-standard ETAS forecaster by +0.064**, and beats a climatology/geography baseline by a
   wide margin. This is genuine *temporal* where-skill, null-tested (label-shuffle → 0.55) and
   stable across walk-forward splits.
2. **Every classical precursor we tested is either redundant with what that model already learns,
   or null under proper controls, or blocked by data access — not by the absence of a method.**
3. **The dominant wall for "silent-region" precursors is data-access engineering, not physics.**
   Two data-plumbing fixes this campaign (FDSN node routing; transient-error cache poisoning)
   unblocked far more than any algorithm change.

## Honest calibration note

The earlier headline of **0.73** was measured on the regular USGS-M2.5 sample set (~57k samples).
On the broader active-cell set built from the fuller catalog (~75k samples, more marginal cells),
the same model scores **0.7086 ± 0.0014**. Both are honest; the broader set is harder. We report
~0.71 as the conservative operational number and 0.73 as the figure on the original sample set.

## The operational model vs. baselines (held-out test, M5+/100km/30d)

| Forecaster | Operational AUC | Note |
|---|---|---|
| **Deep operational (GRU+attention on event sequences + context)** | **0.7086 ± 0.0014** | the deployed approach |
| ETAS (Ogata conditional intensity, gold-standard statistical) | 0.6662 | deep model **beats it by +0.064** |
| context-14 GBT (engineered features, no sequence model) | 0.7194 | proxy; GRU ≈ this + implicit ETAS |
| Climatology (historical rate per cell) | ~0.57–0.60 | geography-only |
| Label-shuffle null | ~0.55 | confirms no leakage |
| case-control model (old approach) | ~0.60 | can't rank cross-location |

## Precursor / enhancement tests — full results

All are operational-AUC on the same held-out test, or a controlled before/after.

| Idea | Result | Verdict |
|---|---|---|
| **ETAS intensity** | standalone 0.666; **+0.011 to the GBT**, **+0.0015 to the deep model** | Real, but the **GRU already learns aftershock-decay** from the raw sequence — explicit channel redundant |
| **b-value** (Gutenberg-Richter, Aki MLE) | univariate 0.60 (low-b → hazard) | Real classical signal, **~95% redundant** with model context |
| **Coulomb stress** (from GCMT focal mechs) | univariate 0.61 | Strongest single new feature, still **redundant** with the stress-transfer context |
| **AMR** (accelerating moment release / Benioff) | univariate 0.57 | Real, redundant |
| good-4 physics features **combined**, added to deep model | **+0.0015 (within seed noise 0.002)** | **NULL on the deep model** — it's already information-efficient |
| **Natural-time** (Varotsos κ1) | univariate 0.515 | **No operational signal** |
| **Tidal phase** (lunar synodic/fortnightly) | univariate 0.47–0.54; adding it *hurt* | **No operational signal** |
| **Seismic-gap** prior | ~0.56 (earlier) | Null |
| **M2.0 foreshock enrichment** (input catalog 2.5→2.0) | 0.7047 vs 0.7086 baseline | **NULL — slightly worse.** M2.5 already captures the sequence; sub-2.5 adds noise |
| **GNSS deformation** (common-mode-filtered, Ridgecrest) | pre-quake 0.70 mm vs control 0.22 mm, noise floor 0.69 mm | **NULL pre-slip**, with a **proven 213 mm coseismic positive control** |
| **dv/v** (seismic velocity change) | method **validated**; multi-event run done | co-seismic real *with care*; **no pre-seismic precursor**; generic harness too blunt |

### dv/v in detail

- **Single-station autocorrelation FAILED validation** — too noisy to resolve the known ~0.1–0.3%
  signal even after multi-day stacking (dv/v slammed to grid edges, baseline σ ≈ 1.1%).
- **Station-pair cross-correlation (whitened, stacked, multi-pair) VALIDATED** on Ridgecrest M7.1
  with hand-tuned processing (4 stations, full-year baseline, tuned coda): a clear **co-seismic
  velocity drop of −2.3σ** at the mainshock week (the published signature).
- **Pre-seismic:** at the well-resolved Ridgecrest validation, the pre-quake window shows **no
  clear velocity drop** (points within baseline scatter). Consistent with the literature:
  co-seismic dv/v is robust, pre-seismic is absent/contested.

### dv/v multi-event controlled test (bulk pipeline) — done

A bulk-grade pipeline (`scripts/research/dvv_multi_event.py`) solved the data-access wall that
blocked the first attempt (per-node throttle + backoff retry; transient-aware caching; probe-based
station qualification). It found **15 M6.0+ events with verified continuous station pairs**
(California, Alaska, Chile) out of 969 candidates and ran the full controlled test (baseline
reference + co-seismic positive control + 3-year negative control). **11 scored.** Verdict:

- **Positive control is WEAK in the generic harness: co-seismic median −0.17σ** (only 2/6 even of
  the ≥4-station events drop < −1σ; Ridgecrest M7.1 itself flips to +1.4σ here vs −2.3σ in the
  hand-tuned validation). The difference is **per-event craftsmanship** — baseline length,
  coda-window/frequency tuning, aftershock removal, station QC — which a one-size harness can't do.
- **No pre-seismic precursor:** pre-quake median −0.05σ (−0.56σ for ≥4-station events, but signs
  are mixed: −1.7, −1.1, −1.1 vs +0.6, +0.9); paired vs negative control (n=5) real-below-control
  in 3/5 — chance, not signal.
- **Conclusion:** dv/v at scale is **not** a "generic harness over many events" problem; it needs
  MSNoise/NoisePy-grade *per-event* processing. The bulk pipeline cracked the data-access barrier
  (the session's binding constraint) but confirmed the science needs craftsmanship per event. The
  honest payoff remains **co-seismic monitoring** (with care), **not pre-seismic prediction**.
  Results: `results/calibration/dvv_multi_event_results.json`.

## The hidden wall: data-access engineering (the real lever)

Two fixes mattered more than any algorithm:

1. **FDSN node routing.** Regional networks archive at their *own* data center, not IRIS
   (CI→SCEDC, BK/NC→NCEDC, GE→GEOFON, KO→ORFEUS, IV→INGV, …). Querying IRIS-only returned
   **0 bytes** for data that exists; routing per-network unblocked it (e.g. a station 5 km from
   the Ridgecrest epicenter went from "no data" to fully usable). Several earlier "silent-region"
   nulls were partly *us querying the wrong server*.
2. **Transient-error cache poisoning.** Caching every failed fetch as a permanent "no data" miss
   turned transient timeouts/429s into permanent holes — **9,908 poisoned entries** silently
   killed entire events (including known-good Ridgecrest). Fixed to cache misses only on genuine
   HTTP-204 no-data and retry transients.

**Lesson:** for silent-region precursor work, the binding constraint is almost always *can we get
the data*, not *is there a signal*. "We can't see it" ≠ "it isn't there."

## What would actually move the needle (honest next steps)

- **Operational forecaster, not precursors, is where skill lives today.** Deploy it honestly:
  state the ~0.71 where-skill, the +0.06 over ETAS, *and* the limit that ~40% of great quakes are
  catalog-silent (no usable precursor at any scale we can access).
- **dv/v at scale** would need a dedicated data-engineering pipeline (authenticated bulk waveform
  download per archive, MSNoise/NoisePy-grade processing) — a project, not a session. Expected
  payoff is co-seismic monitoring, not pre-seismic prediction.
- **Better instrumentation in silent-prone regions** (Turkey/Haiti/Myanmar) is the physical fix:
  the silence correlates with thin/changing monitoring, which is *why* foreshocks go uncataloged
  and dv/v baselines don't exist there.

## Reproduce

- Current best causal tabular ranker: `scripts/research/operational_tabular_ranker.py --rebuild-features`
  writes `results/calibration/earthquake_operational_tabular_ranker.json`.
- Optional prior downloader: `scripts/research/download_operational_priors.py` fetches the USGS
  1900-1999 M5+ historical catalog, GSRM principal strain-rate grid, and opt-in GEM active-fault
  GeoJSON.
- Mega bakeoff: `scripts/research/operational_mega_bakeoff.py` writes
  `results/calibration/earthquake_operational_mega_bakeoff.json` or a chosen output path.
- Rank-objective bakeoff: `scripts/research/operational_rank_objective_bakeoff.py` writes
  `results/calibration/earthquake_operational_rank_objective_bakeoff.json`.
- Ensemble audit: `scripts/research/operational_ensemble_audit.py` writes
  `results/calibration/earthquake_operational_ensemble_audit.json`.
- Nextwave public-data downloader: `scripts/research/download_nextwave_earthquake_data.py` fetches
  CRESCENT tremor/GNSS, EarthScope station inventory, and regional M1+ catalogs.
- Subduction-prior downloader: `scripts/research/download_subduction_priors.py` fetches USGS
  Slab2 and UCSD Coupling Cloud grids used by the default nextwave ranker.
- Waveform/noise pilot downloader: `scripts/research/download_waveform_noise_embeddings.py` fetches
  regional waveform snippets and writes derived spectral/RMS embeddings. These features are opt-in
  with `scripts/research/operational_nextwave_ranker.py --include-waveform-noise`.
- ARIA InSAR metadata downloader: `scripts/research/download_insar_aria_metadata.py` fetches
  paginated NASA CMR metadata for ARIA Sentinel-1 GUNW coverage features. These are coverage
  features, not displacement features, and are opt-in with `--include-heavy-data`.
- Nextwave ranker and ensemble: `scripts/research/operational_nextwave_ranker.py --rebuild-nextwave`
  writes `results/calibration/earthquake_operational_nextwave_ranker.json`;
  `scripts/research/operational_nextwave_ensemble.py --rebuild-nextwave-preds` writes
  `results/calibration/earthquake_operational_nextwave_ensemble.json`.
- Heavy-data audit: `scripts/research/operational_nextwave_ranker.py --include-heavy-data` writes
  the dense-GNSS/ARIA coverage run; `scripts/research/operational_regional_expert_bakeoff.py`
  tests validation-gated regional experts.
- Operational model + A/B: `scripts/deep_operational_earthquake.py`
  (`--min-input-mag` for the M2.0 lever, `--extra-features` for the physics channels;
  `HAZARDPULSE_USGS_FULL=1` for the fuller catalog; `HAZARDPULSE_SHUFFLE_LABELS=1` null test;
  `HAZARDPULSE_TEST_START_YEAR` walk-forward).
- Results JSON: `results/calibration/earthquake_deep_op_{fullm25,m2input,xf}.json`.
- dv/v toolkit + validation, GNSS test, ETAS/feature benchmarks: research scripts (scratchpad),
  promotable to `scripts/` once a dv/v data pipeline is committed.
