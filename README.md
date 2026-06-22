# HazardPulse

> Part of **[Coherence Energy Labs](https://coherenceenergylabs.com)** — an independent research lab developing coherence field theory and the Coherence Lang language (founded 2024 by Josh Philbrick).

**Three natural hazard prediction systems derived from a single partial differential equation.**

All models are pure Python + NumPy. Every algorithm — logistic regression, gradient boosted trees, ensemble stacking, bootstrap confidence intervals — implemented from scratch. No sklearn. No TensorFlow. No PyTorch.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)

---

## Results

| Model | AUC | 95% CI | Test Set | Baseline | Margin |
|---|---|---|---|---|---|
| Hurricane RI (Config C, full) | **0.967** | [0.955, 0.977] | NA storms 2015+ | Persistence: 0.940 | +0.027 |
| Hurricane RI (Config A, met only) | **0.956** | [0.937, 0.971] | NA storms 2015+ | Wind+Lat: 0.868 | +0.088 |
| Tornado severity EF4+ | 0.987 | [0.984, 0.992] | 20 EF4+ tornadoes | — | **unreliable (n=20)** |
| Tornado severity EF3+ | **0.917** | [0.901, 0.934] | 166 EF3+ tornadoes | — | — |
| Tornado severity EF2+ | **0.851** | [0.843, 0.860] | 932 EF2+ tornadoes | — | — |
| Earthquake M6+ global | **0.733** | [0.704, 0.750] | 980 mainshocks (2015-2023) | Rate-only: 0.597 | +0.137 |
| Tornado formation (day-ahead) | **0.644** | [0.623, 0.658] | 23,912 cell-days | Climatology: 0.500 | +0.144 |

All models use strict temporal train/test splits. No data leakage. No test-set peeking for ensemble selection or calibration. Bootstrap confidence intervals on every claim.

---

## The Equation

One partial differential equation, three hazard domains:

```
D · ∇²(τ_c) - Γ · τ_c + S = 0
```

The Helmholtz coherence PDE — a damped wave equation with source. Same structure, different physics:

| Parameter | Earthquake | Hurricane | Tornado |
|---|---|---|---|
| **τ_c** (coherence field) | Seismic moment accumulation | Vortex organization | Mesocyclone intensity |
| **D** (diffusion) | Stress transfer between faults | Lateral mixing in vortex | Storm-relative flow |
| **Γ** (damping) | Friction and healing | Environmental wind shear | Downdraft disruption |
| **S** (source) | Tectonic loading rate | Ocean heat flux (WISHE) | Shear + buoyancy forcing |

Each system follows the same pattern: coherence accumulates → precursors emerge → critical threshold is crossed → coherence releases. The model detects the precursor phase.

---

## Quick Start

```bash
pip install hazardpulse
```

### Earthquake prediction (all M6+ events globally)
```bash
hazardpulse earthquake evaluate --min-magnitude 6.0
```

### Hurricane rapid intensification
```bash
hazardpulse hurricane evaluate --basin NA --threshold 30kt/24h
```

### Tornado formation + severity
```bash
hazardpulse tornado evaluate --mode formation
hazardpulse tornado evaluate --mode severity --threshold EF3+
```

### Python API
```python
from hazardpulse.earthquake import model as eq_model

# Train and evaluate on USGS catalog
results = eq_model.train_and_evaluate(
    train_years=(2005, 2014),
    test_years=(2015, 2023),
    min_magnitude=6.0
)
print(f"Test AUC: {results['test_auc']:.3f}")  # 0.733
```

---

## How It Works

### Earthquake (AUC = 0.733 [0.704, 0.750])

62 features extracted from the M4+ USGS catalog for each potential M6+ location. Key signals: **maximum magnitude in 6-month window** (r=0.30), **90-day event counts** (r=0.27), **Coulomb stress proxy × rate** (r=0.27), **b-value trends** (Gutenberg-Richter distribution shifting). 5-model ensemble: L2 logistic + 3 gradient boosted tree depths + random subspace GBM → weighted average (selected by temporal CV). Tested on 980 M6+ mainshocks (Gardner-Knopoff aftershock declustering) with same-location controls only. Block bootstrap 95% CI: [0.704, 0.750]. Brier Skill Score: 0.102. Stable across all 2-year test blocks (0.688-0.752).

### Hurricane Rapid Intensification (AUC = 0.967 [0.955, 0.977])

Three feature configurations tested via ablation:
- **Config A** (15 standard meteorological features): AUC 0.956 — just intensity, change rates, latitude, translational speed
- **Config B** (+climatological SST/shear estimates): AUC 0.966 — parametric functions of lat/lon/month, NOT actual observations
- **Config C** (full + interactions): AUC 0.967 — wind², lat×MPI_deficit, prior RI history

Fair baselines on the same test set: persistence (AUC 0.940), wind+latitude logistic (AUC 0.868). Cross-basin generalization: NA→WP AUC 0.939, WP→NA AUC 0.949.

**Important caveat**: This uses IBTrACS best-track data (post-season reanalysis), not real-time operational data. Comparison to SHIPS-RII or other operational models requires evaluation on real-time data, which has NOT been done.

### Tornado Formation (AUC = 0.644 [0.623, 0.658])

67 features predicting which 2° grid cells produce tornadoes on which days, using **only information available the day before** (all lookbacks start at d_off ≥ 1). No same-day atmospheric data. No post-event features. Key signals: tornado counts in nearby cells from previous days (synoptic propagation), seasonal/climatological rates, topographic encoding.

- **Continuation** (nearby tornadoes yesterday): AUC 0.680 — real signal from outbreak propagation
- **Initiation** (no recent nearby activity): AUC 0.578 — much harder without atmospheric model data

This is the weakest model because tornado prediction fundamentally requires real-time atmospheric data (CAPE, wind shear, helicity) which we don't currently ingest. The model demonstrates that historical tornado patterns carry modest predictive signal, but cannot compete with NWP-based systems like SPC Day 1 outlooks without atmospheric inputs.

**Severity** (post-event nowcasting, NOT real-time): EF2+ AUC 0.851, EF3+ AUC 0.917. Uses width from damage surveys — these numbers describe post-event analysis capability only.

---

## Replication

Every result is reproducible:

```bash
git clone https://github.com/Jphilbrick10/hazardpulse.git
cd hazardpulse
pip install -e ".[viz]"

# Reproduce all figures and AUC numbers
python legacy/earthquake_model_v7_honest.py
python legacy/hurricane_ri_model_v4_honest.py
python legacy/tornado_model_v5_honest.py
```

The scripts download data directly from USGS, IBTrACS, and SPC (free, no API key needed). First run takes ~15-30 minutes to fetch catalogs and train models.

**If you get a different AUC number, [open an issue](https://github.com/Jphilbrick10/hazardpulse/issues/new?template=replication_report.yml)**. Reproducibility is non-negotiable.

---

## Live Predictions (Coming Soon)

The public prediction platform at **hazardpulse.com** includes:

- **Real-time earthquake risk map** — USGS data feed, predictions every 6 hours, 3° global grid
- **Hurricane RI tracker** — NHC advisory data, RI probability for every active tropical cyclone
- **Tornado risk grid** — atmospheric model data integration for genuine day-ahead prediction
- **Public prediction ledger** — Every prediction SHA-256 hashed and timestamped before events occur
- **Running accuracy scores** — AUC, Brier Score, reliability diagrams updated live

No hiding. No cherry-picking. Every prediction logged, every outcome tracked.

[Platform architecture →](docs/live_platform.md)

---

## Data Sources

All data is freely available from US government agencies:

| Source | Agency | URL | Update Frequency |
|---|---|---|---|
| Earthquake catalog | USGS | [earthquake.usgs.gov](https://earthquake.usgs.gov/fdsnws/event/1/) | Real-time (1 min) |
| Tropical cyclone tracks | NOAA/NCEI | [IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive) | Seasonal |
| Tornado reports | SPC/NOAA | [spc.noaa.gov](https://www.spc.noaa.gov/wcm/) | Daily |

---

## Honest Caveats

We believe in transparency. Key limitations:

1. **Hurricane models use best-track data** (retrospective reanalysis), not real-time intensity estimates which carry ~10 kt uncertainty. No comparison to operational models (SHIPS-RII, etc.) is claimed — that requires evaluation on real-time data.
2. **Tornado formation AUC 0.644 is modest.** Without real-time atmospheric data (CAPE, shear, helicity from NWP models), the model relies on historical tornado patterns and seasonal climatology. Direct comparison to SPC Day 1 outlooks requires evaluating both systems on the same grid/period, which has not been done.
3. **Tornado severity uses post-event data** (damage survey widths). These are nowcasting metrics, not real-time predictions.
4. **Earthquake negatives are same-location controls** (same zone, different time). This is more honest than geographic negatives but may still understate the difficulty of real-world deployment.
5. **EF4+ tornado severity** has only 20 test events — the AUC (0.987) is statistically unreliable.
6. **All models are retrospective**. True validation requires prospective real-time testing with timestamped predictions — which is exactly what the live platform will provide.
7. **"Climatological estimates"** in hurricane Config B are parametric functions of lat/lon/month — they are NOT actual SST, ocean heat content, or wind shear observations.

---

## The Unified Framework

The fact that one equation — with different physical parameters — produces useful predictions across three completely different natural hazards is noteworthy:

- Earthquakes: Solid earth, tectonic plates, years of stress accumulation → AUC 0.733
- Hurricanes: Tropical ocean-atmosphere, days of thermodynamic feedback → AUC 0.967
- Tornadoes: Mesoscale convective storms, hours of severe weather dynamics → AUC 0.644

The hurricane model is genuinely strong. The earthquake model extracts real signal beyond seismicity rates. The tornado model needs atmospheric data to compete with operational systems. We report what the numbers say, not what we wish they said.

---

## Citation

```bibtex
@software{hazardpulse_2026,
  author = {Philbrick, Josh},
  title = {HazardPulse: Unified Natural Hazard Prediction from Coherence Field Theory},
  year = {2026},
  url = {https://github.com/Jphilbrick10/hazardpulse}
}
```

---

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

---

**HazardPulse** | [hazardpulse.com](https://hazardpulse.com) | [OneUnity.earth](https://oneunity.earth)
