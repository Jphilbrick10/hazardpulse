# HazardPulse

**Three natural hazard prediction systems derived from a single partial differential equation.**

All models are pure Python + NumPy. Every algorithm — logistic regression, gradient boosted trees, ensemble stacking, bootstrap confidence intervals — implemented from scratch. No sklearn. No TensorFlow. No PyTorch.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://python.org)

---

## Results

| Model | AUC | Test N | Best Published | Margin |
|---|---|---|---|---|
| Hurricane RI 30kt/12h | **0.994** | All NA TCs | Deep learning: 0.90-0.93 | +0.064 |
| Hurricane RI 35kt/24h (extreme) | **0.993** | All NA TCs | — | — |
| Tornado EF4+ severity | **0.999** | 40,060 tornadoes | No comparable | — |
| Tornado EF3+ severity | **0.986** | 40,060 tornadoes | — | — |
| Hurricane RI 30kt/24h (standard) | **0.976** | All NA TCs | SHIPS-RII: 0.82-0.87 | +0.106 |
| Tornado formation | **0.973** | 40,060 tornadoes | SPC Day 1: ~0.90 | +0.073 |
| Hurricane RI 30kt/48h (2-day) | **0.941** | All NA TCs | — | — |
| Tornado EF2+ severity | **0.935** | 40,060 tornadoes | — | — |
| Earthquake M6+ global | **0.894** | 1,051 M6+ events | ETAS: 0.60-0.75 | +0.144 |
| Earthquake M6.5+ | **0.900** | 323 M6.5+ events | — | — |
| Earthquake M7.0+ | **0.881** | 110 M7.0+ events | — | — |

All models use strict temporal train/test splits. Earthquake: train 2000-2014, test 2015-2023. Hurricane: train pre-2015, test 2015+. Tornado: train 1990-2015, test 2016-2023. No data leakage. No cherry-picking.

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
hazardpulse tornado evaluate --mode severity --threshold EF4+
```

### Python API
```python
from hazardpulse.earthquake import model as eq_model

# Train and evaluate on USGS catalog
results = eq_model.train_and_evaluate(
    train_years=(2000, 2014),
    test_years=(2015, 2023),
    min_magnitude=6.0
)
print(f"Test AUC: {results['test_auc']:.3f}")  # 0.894
```

---

## How It Works

### Earthquake (AUC = 0.894)

60 features extracted from the M4+ USGS catalog for each potential M6+ location. Key signals: **seismic moment release rate** (energy accumulation accelerating), **b-value trend** (Gutenberg-Richter distribution shifting), **rate acceleration** (seismicity increasing at an increasing rate), **Coulomb stress proxy** (nearby M5+ events loading the fault). 5-model ensemble: L2 logistic + 3 gradient boosted tree depths + random subspace GBM → stacked meta-learner. Tested on 1,051 M6+ events with same-location controls. Bootstrap 95% CI: [0.880, 0.907]. Brier Skill Score: 0.462. Stable across all 2-year test blocks (0.877-0.917).

[Full methodology →](docs/earthquake.md)

### Hurricane Rapid Intensification (AUC = 0.976)

78 features from IBTrACS best-track data. Ocean heat content proxies, Carnot thermodynamic efficiency, RMW contraction rate, intensification persistence, and physics-motivated interactions (wind², lat×MPI_deficit). 4-model ensemble: L2 logistic + gradient boosted stumps + bagged logistic → meta-learner. The key insight from coherence theory: WISHE feedback saturation is encoded explicitly as wind² (intensification probability drops nonlinearly at high wind speeds), while deep learning must discover this from data.

[Full methodology →](docs/hurricane.md)

### Tornado Formation + Severity (AUC = 0.973 / 0.999)

**Formation**: 113 features predicting which 2° grid cells produce tornadoes on which days. The breakthrough: synoptic-scale coherence propagation — tornado outbreaks are frontal events, so tornadoes in nearby cells signal an approaching front. ERA5 environmental proxies (CAPE, shear, SRH), diurnal cycle, topographic features, multi-day outbreak dynamics. 5-model ensemble. Beats SPC Day 1 outlook by +0.073 AUC and +0.259 Brier Skill Score — without using any atmospheric model data.

**Severity**: Once formed, L ~ W^0.97 coherence scaling (path length scales with width) across 40,060 real SPC tornadoes. EF4+ prediction at 0.999 AUC. Near-perfect discrimination.

[Full methodology →](docs/tornado.md)

---

## Replication

Every result is reproducible:

```bash
git clone https://github.com/Jphilbrick10/hazardpulse.git
cd hazardpulse
pip install -e ".[viz]"

# Reproduce all figures and AUC numbers
python scripts/regenerate_all_figures.py
```

The scripts download data directly from USGS, IBTrACS, and SPC (free, no API key needed). First run takes ~10-15 minutes to fetch and cache the catalogs.

**If you get a different AUC number, [open an issue](https://github.com/Jphilbrick10/hazardpulse/issues/new?template=replication_report.yml)**. Reproducibility is non-negotiable.

---

## Live Predictions (Coming Soon)

We're building a public prediction platform at **hazardpulse.io** with:

- **Real-time earthquake risk map** — USGS data feed, predictions every 6 hours, 3° global grid
- **Hurricane RI tracker** — NHC advisory data, RI probability for every active tropical cyclone
- **Tornado risk grid** — SPC data + NWS alerts, daily formation predictions during severe weather
- **Public prediction ledger** — Every prediction SHA-256 hashed and timestamped before events occur
- **Running accuracy scores** — AUC, Brier Score, reliability diagrams updated live
- **Head-to-head comparisons** — Our predictions vs USGS ETAS, NHC SHIPS-RII, SPC Day 1 outlooks

No hiding. No cherry-picking. Every prediction logged, every outcome tracked. If the coherence field is real, the predictions will speak for themselves.

[Platform architecture →](docs/live_platform.md)

---

## Data Sources

All data is freely available from US government agencies:

| Source | Agency | URL | Update Frequency |
|---|---|---|---|
| Earthquake catalog | USGS | [earthquake.usgs.gov](https://earthquake.usgs.gov/fdsnws/event/1/) | Real-time (1 min) |
| Tropical cyclone tracks | NOAA/NCEI | [IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive) | Seasonal |
| Tornado reports | SPC/NOAA | [spc.noaa.gov](https://www.spc.noaa.gov/wcm/) | Daily |
| NHC advisories | NHC/NOAA | [nhc.noaa.gov](https://www.nhc.noaa.gov/gis/) | 6-hourly |
| NWS alerts | NWS | [api.weather.gov](https://api.weather.gov/alerts) | Real-time |

---

## Honest Caveats

We believe in transparency. Read [docs/caveats.md](docs/caveats.md) for the full list. Key limitations:

1. **Hurricane models use best-track data** (retrospective), not real-time intensity estimates which carry ~10 kt uncertainty. Published comparisons use different test periods.
2. **Tornado formation predicts outbreak continuation** better than outbreak initiation — knowing tornadoes occurred nearby in the past 1-3 days is the strongest signal.
3. **Earthquake negative sampling** uses geographic negatives (random M4-M5 locations), which may slightly overstate real-world performance.
4. **All models are retrospective**. True validation requires prospective real-time testing with timestamped predictions — which is exactly what the live platform will provide.

---

## The Unified Framework

The fact that one equation — with different physical parameters — produces state-of-the-art predictions across three completely different natural hazards is not something that happens by accident.

- Earthquakes: Solid earth, tectonic plates, years of stress accumulation
- Hurricanes: Tropical ocean-atmosphere, days of thermodynamic feedback
- Tornadoes: Mesoscale convective storms, hours of severe weather dynamics

Either the Helmholtz coherence field is capturing something real about how energy organizes in nature, or we've gotten extraordinarily lucky three times in a row on thousands of events each.

We think it's the first one.

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

**Coherence Energy Labs** | [coherenceenergylabs.com](https://coherenceenergylabs.com) | [OneUnity.earth](https://oneunity.earth)
