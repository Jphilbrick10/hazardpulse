# Earthquake Prediction Model - Complete Verification Package

## CLAIM

Same-location temporal discrimination AUC of 0.80 for M6.0+ earthquakes
within 100km and 90 days, using seismicity precursor features. This isolates
temporal prediction skill from spatial knowledge.

Global AUC (including spatial discrimination): 0.907 [0.892, 0.923]
Same-location AUC (temporal only): 0.799 (macro-average across 237 locations)

## HOW TO REPRODUCE

### Step 1: Clone the repository

```bash
git clone https://github.com/Jphilbrick10/hazardpulse.git
cd hazardpulse
pip install -e .
pip install numpy scipy
```

### Step 2: Run the honest evaluation

```bash
python -m hazardpulse.earthquake.v4_regional --honest --output-dir results/earthquake_honest
```

This will:
- Download USGS earthquake catalog (2000-2024, M2.5+, 493,963 events)
- Aftershock decluster using Gardner-Knopoff (1974) windows
- Build samples: M6.0+ mainshocks as positives, same-location time-offset controls as negatives
- Extract 55 features per sample across 4 blocks
- Train GBT models per region and globally
- Evaluate on held-out test set (2020-2024)
- Save results to results/earthquake_honest/v4_regional_honest_results.json

Parameters (hardcoded in honest mode):
- Magnitude threshold: M6.0+ (no M5.5 fallback)
- Label radius: 100 km
- Forward window: 90 days
- Control ratio: 5:1 (negatives per positive)
- Train: 2005-2017, Val: 2018-2019, Test: 2020-2024

Expected runtime: ~1 hour (with checkpointing)

### Step 3: Run the same-location AUC test

```bash
python scripts/same_location_auc.py
```

This will:
- Load the same data and train a GBT on rest_of_world region
- Predict on test samples while tracking each sample's source location
- Group test samples by (lat, lon)
- Compute AUC within each location group (positive vs control at same site)
- Report macro-averaged same-location AUC

Expected runtime: ~3-4 hours (feature extraction is the bottleneck)

### Step 4: Verify results

Results are saved to:
- results/earthquake_honest/v4_regional_honest_results.json (global evaluation)
- results/earthquake_honest/same_location_auc.json (same-location temporal test)

---

## FILES INVOLVED

### Model Code

| File | Lines | Purpose |
|------|-------|---------|
| `src/hazardpulse/earthquake/v4_regional.py` | ~2700 | Main model: feature extraction, GBT training, evaluation |
| `src/hazardpulse/earthquake/coherence_engine.py` | ~1400 | Coherence field theory feature computation |
| `src/hazardpulse/data/earthquake.py` | ~200 | USGS catalog download and caching |
| `scripts/same_location_auc.py` | ~360 | Same-location AUC computation |
| `scripts/download_gnss.py` | ~530 | GNSS station data downloader |
| `scripts/download_geophysical.py` | ~1290 | Multi-source geophysical data downloader |

### Results Files

| File | Content |
|------|---------|
| `results/earthquake_honest/v4_regional_honest_results.json` | Full honest evaluation results |
| `results/earthquake_honest/same_location_auc.json` | Same-location temporal AUC results |
| `results/earthquake_definitive/v4_regional_results.json` | Original (generous) evaluation for comparison |
| `results/earthquake_definitive/definitive_results_v2.json` | Previous v2 model results |

### Data Sources

| Source | URL | What it provides |
|--------|-----|-----------------|
| USGS FDSNWS | https://earthquake.usgs.gov/fdsnws/event/1/query | Global earthquake catalog, M2.5+, 2000-2024 |
| NGL GPS | https://geodesy.unr.edu/gps_timeseries/IGS14/tenv3/IGS14/ | GNSS station time series (3,432 stations cached) |

All data is publicly available and free. No API keys required for USGS.

---

## EXACT EXPERIMENTAL SETUP

### Data

- **Source**: USGS FDSNWS earthquake catalog
- **Period**: 2000-2024 (for feature computation), test on 2020-2024
- **Magnitude range**: M2.5+ for catalog, M6.0+ for targets
- **Geographic scope**: Global
- **Total events**: 493,963
- **After declustering**: 256,018 mainshocks (Gardner-Knopoff 1974)
- **M6.0+ mainshocks**: 1,488 in rest_of_world region

### Positive samples

Each M6.0+ declustered mainshock becomes a positive sample. The label means:
"An M6.0+ earthquake occurs within 100km of this location in the next 90 days."

### Negative samples (controls)

For each positive at (lat, lon, t_event):
- Generate 5 controls at the SAME (lat, lon)
- Time offset: random 1.5-4.5 years forward or backward
- Exclusion: no M6.0+ within 100km and 90 days forward of the control time
- This ensures the control represents a genuinely quiet period at the same location

### Temporal split

- **Train**: 2005-2017 (events whose ref_epoch falls in this range)
- **Validation**: 2018-2019 (for early stopping only)
- **Test**: 2020-2024 (evaluated ONCE, no hyperparameter tuning)
- Integrity enforced with assertions in code

### Feature blocks (55 total)

**Block S: Seismicity (30 features)**
All computed from events BEFORE ref_epoch (strict < ref_epoch filter, line 1297).
- rate_7d, rate_14d, rate_30d, rate_90d, rate_180d, rate_365d (6)
- b_overall, b_shallow (<30km), b_deep (>80km) (3)
- nn_spatial, nn_temporal, st_nn, nn_change (4)
- mom_30d, mom_90d, mom_accel, mom_deficit (4)
- inv_omori, bath_ratio, foreshock_frac (3)
- maxmag_30d, maxmag_90d, mag_var_chg (3)
- depth_mean, depth_trend, frac_shallow (3)
- spatial_conc, elong, quiescence_7d (3)
- iet_delta_aic (1)

**Block G: Geodetic/GNSS (12 features)**
From cached NGL station data. Currently static (same for pos/control at same location).
Not contributing to temporal discrimination. Included for completeness.

**Block C: Coherence Field Theory (8 features)**
tau, grad_tau, S_over_Gamma, Da, divergence_rate, t_c_estimate, singularity_count, tau_x_strain.
Bug fix applied: CatalogArrays now correctly converted to list[dict] for coherence engine.

**Block T: Tectonic context (5 features)**
Tectonic type (3 one-hot), plate_boundary_dist, regional_m6_rate.
Spatial features - contribute to global AUC but not same-location AUC.

### Model

- **Algorithm**: Gradient Boosted Trees (custom implementation, pure NumPy)
- **Hyperparameters**: 200 trees, depth 4, learning rate 0.03, subsample 0.6
- **Early stopping**: patience 40 (on validation loss)
- **Class weighting**: Balanced (inversely proportional to class frequency)
- **No meta-stacker**: Single GBT per variant, no blending
- **Feature normalization**: Zero-mean, unit-variance (fit on train, applied to val/test)
- **NaN handling**: Imputed with training mean per feature

### Evaluation

- **AUC**: Trapezoidal ROC integration (manual implementation)
- **Bootstrap CI**: 1000 resamples, 2.5th/97.5th percentiles
- **Same-location AUC**: Group test samples by (lat, lon), compute AUC within each group (requires ≥1 positive and ≥1 negative), macro-average across groups

---

## RESULTS

### Global evaluation (rest_of_world, honest mode)

| Metric | Value |
|--------|-------|
| Test AUC | 0.907 [0.892, 0.923] |
| Test PR-AUC | 0.721 |
| Brier Score | 0.116 |
| BSS | 0.342 |
| N positive | 258 |
| N negative | 868 |
| Base rate | 22.9% |

### Same-location evaluation (temporal skill only)

| Metric | Value |
|--------|-------|
| Same-location macro AUC | **0.799** |
| Same-location weighted AUC | 0.797 |
| Same-location median AUC | 1.000 |
| Locations evaluated | 237 (with both pos+neg) |
| Locations positive-only | 21 |
| Locations negative-only | 167 |

### Distribution of per-location AUC

| Percentile | AUC |
|-----------|-----|
| P10 | 0.000 |
| P25 | 0.667 |
| P50 | 1.000 |
| P75 | 1.000 |
| P90 | 1.000 |

Note: Many locations have only 1 positive and 1-5 negatives, so individual
location AUCs are binary (0 or 1). The macro-average of 0.799 across 237
locations is the meaningful number.

### Comparison to published systems

| System | AUC | Target | Method |
|--------|-----|--------|--------|
| ETAS (operational) | 0.60-0.75 | M6+, grid cells | Rate-based statistical |
| OEF-Italy | 0.65-0.70 | Regional | ETAS-based |
| Pattern Informatics | 0.60-0.65 | M6+, 5-10yr | Long-range patterns |
| **Ours (global)** | **0.907** | M6+, 100km, 90d | GBT, 55 features |
| **Ours (temporal only)** | **0.799** | Same-location | Temporal discrimination |

IMPORTANT: Direct comparison to ETAS is approximate. ETAS evaluates on fixed
spatial grids with Poisson likelihood tests (N-test, L-test). Our evaluation
uses same-location controls with AUC. The problem definitions differ.

---

## KNOWN LIMITATIONS AND CAVEATS

1. **Retrospective, not prospective.** All evaluation is on held-out historical
   data. True validation requires prospective predictions logged before events.

2. **Same-location controls are easier than grid-based evaluation.** The model
   only needs to distinguish "active time" from "quiet time" at known-active
   locations. It never has to predict in locations that have never had M6+.

3. **Small per-location samples.** Most locations have 1 positive and 1-5
   negatives. Individual location AUCs are noisy. The macro-average is the
   reliable number.

4. **GNSS features are static.** The current GNSS integration provides location-
   level features (velocity, strain rate) but not time-resolved features.
   GNSS does not contribute to temporal discrimination.

5. **Block C (coherence) contribution is unproven.** The Block C type mismatch
   bug was fixed but we have not run an ablation to verify coherence features
   add value above seismicity-only.

6. **No CSEP-format evaluation.** A CSEP submission would require reformulating
   as a rate model on a fixed grid with N/L/S/M consistency tests.

7. **Base rate of 22.9% is still relatively high** compared to operational
   forecasting (typically <5%). The 5:1 control ratio is standard for
   ML evaluation but not representative of operational false alarm rates.

---

## TOP FEATURES (seismologically meaningful)

From the global model (rest_of_world):

1. **quiescence_7d** - Seismic quiescence in the 7 days before the reference
   time. High values = recent quiet period relative to annual average.
   Known precursor phenomenon (Mogi doughnut hypothesis).

2. **rate_30d / rate_90d** - Seismicity rate acceleration in recent weeks/months.
   Foreshock sequences and rate increases before large events.

3. **nn_change** - Change in nearest-neighbor distance. Decreasing NN distance
   = spatial clustering = stress concentration.

4. **mom_accel** - Seismic moment release acceleration. Energy release
   increasing faster than background rate.

5. **bath_ratio** - Ratio of largest to second-largest event in recent window.
   Large Bath ratio = one dominant event = possible foreshock.

6. **plate_boundary_dist** - Distance to nearest historical M6+. This is a
   SPATIAL feature, not temporal. Contributes to global AUC but not
   same-location AUC.

---

## WHAT A VERIFIER SHOULD CHECK

1. **Temporal integrity**: In `compute_block_s` (line 1297), verify that
   `cat.times < ev_time` strictly excludes the target event. This is the
   single most important line in the model.

2. **Control generation**: In `generate_control_samples` (line 1098), verify
   that controls are at the same (lat, lon) and are validated to have no
   M6.0+ within 100km/90d forward.

3. **Temporal split**: Verify assertions at lines 1834-1839 that enforce
   train ≤ 2017, val in 2018-2019, test in 2020-2024.

4. **No meta-stacker**: Verify there is a single GBT per model variant,
   no blending or stacking of predictions.

5. **Same-location AUC computation**: In `same_location_auc.py`, verify
   that samples are grouped by (lat, lon) and AUC is computed within groups.

6. **Feature computation reproducibility**: Run the model twice and verify
   identical results (random seed is fixed at 42).

7. **Data reproducibility**: The USGS catalog is the same for everyone
   (public API, deterministic query). Cache files can be deleted to force
   fresh downloads.

---

## CONTACT

Josh Philbrick
Coherence Energy Labs
josh@coherenceenergylabs.com
https://github.com/Jphilbrick10/hazardpulse

This is RESEARCH ONLY. Not an operational earthquake prediction system.
