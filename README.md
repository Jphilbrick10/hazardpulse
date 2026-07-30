# HazardPulse

> Part of **[Coherence Energy Labs](https://coherenceenergylabs.com)** — an independent research lab developing coherence field theory and the Coherence Lang language (founded 2024 by Josh Philbrick).

**Three natural hazard prediction systems derived from a single partial differential equation.**

All models are pure Python + NumPy. Every algorithm — logistic regression, gradient boosted trees, ensemble stacking, bootstrap confidence intervals — implemented from scratch. No sklearn. No TensorFlow. No PyTorch.

[![License](https://img.shields.io/badge/License-AGPL--3.0-blue.svg)](LICENSE)
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

> **Provenance:** the table above reports *retrospective* results from the research model line (v7/v4/v5 scripts, since superseded; preserved in git history). The primary evidence surface going forward is the **live prospective scorecard** — forecasts frozen, signed, and timestamped *before* outcomes, then scored as outcomes mature (see [Live Predictions](#live-predictions--operating-now) and `dist/data/`). Retrospective and prospective numbers are not comparable and are never mixed.

---

## The Equation

One partial differential equation, three hazard domains:

```
D · ∇²(τ_c) - Γ · τ_c + S = 0
```

The screened-Poisson (static Helmholtz-type) coherence PDE — the steady state of damped diffusion with a source term. Same structure, different physics:

| Parameter | Earthquake | Hurricane | Tornado |
|---|---|---|---|
| **τ_c** (coherence field) | Seismic moment accumulation | Vortex organization | Mesocyclone intensity |
| **D** (diffusion) | Stress transfer between faults | Lateral mixing in vortex | Storm-relative flow |
| **Γ** (damping) | Friction and healing | Environmental wind shear | Downdraft disruption |
| **S** (source) | Tectonic loading rate | Ocean heat flux (WISHE) | Shear + buoyancy forcing |

Each system follows the same pattern: coherence accumulates → precursors emerge → critical threshold is crossed → coherence releases. The model detects the precursor phase.

---

## Quick Start

Not yet on PyPI — install from source:

```bash
git clone https://github.com/coherence-energy-labs/hazardpulse.git
cd hazardpulse
pip install -e .
```

### Earthquake model (regional v4, honest mode)
```bash
hazardpulse earthquake-v4 --honest
```

### Hurricane operational-RI utilities
```bash
hazardpulse hurricane build-operational-ri-dataset \
    --start-year 2018 --end-year 2024 --out ri_cases.json
hazardpulse hurricane benchmark-operational-ri \
    --dataset ri_cases.json --test-start-year 2022
```

### Verify a published forecast — with zero trust in our code
Every live forecast carries an Ed25519-signed receipt. The verifier is
self-contained (stdlib + `cryptography` only) and re-implements the published
receipt spec, so it proves integrity and authenticity without running any
HazardPulse code:

```bash
python scripts/verify_forecast.py \
    --artifact "$(ls dist/data/replay/eq_fcst_*.json | tail -n 1)"
```

This checks receipt integrity (any tampered field is detected). Pass `--pubkey <hex>`
with the published signing key to additionally verify authenticity.

### Python API
```python
from hazardpulse.earthquake.v4_regional import main as run_earthquake_v4

run_earthquake_v4(honest=True)  # M6+, 100 km, 90 days, 5:1 controls
```

The CLI entry points and the receipt verifier above are exercised in CI on every push — if a documented command rots, the build goes red.

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

Reproducibility here is enforced, not promised:

- **Training is deterministic by construction.** Every stochastic operation in the model line draws from an explicitly seeded generator — never from NumPy's global stream. This is guarded by a CI gate ([`tests/test_determinism.py`](tests/test_determinism.py)) that trains every model twice under *different* global RNG states and fails unless the resulting trees are identical, value for value. The gate was proven against the bug it prevents: it goes red on the pre-fix code.
- **Live forecasts are replayable.** Each forecast cycle freezes its full input snapshot and model fingerprint into a replay artifact (`dist/data/replay/`), and its decision is bound into an Ed25519-signed receipt you can verify independently with [`scripts/verify_forecast.py`](scripts/verify_forecast.py) — no HazardPulse code in the loop.
- **The current model line is in-tree.** The definitive training protocols live in `src/hazardpulse/earthquake/` and `src/hazardpulse/tornado/`; the retrospective research scripts behind the historical results table are superseded and preserved in git history, labeled as such above.

**If anything fails to reproduce, [open an issue](https://github.com/coherence-energy-labs/hazardpulse/issues/new?template=replication_report.yml)**. Reproducibility is non-negotiable — a replication failure is treated as a defect in us, not in you.

---

## Live Predictions — operating now

The public platform at **[hazardpulse.com](https://hazardpulse.com)** runs prospective, timestamped forecasting on a schedule — this is the project's primary truth surface:

- **Scheduled forecast cycles** (GitHub Actions) fetch live agency feeds, freeze each forecast *before* the outcome window, and score it when outcomes mature
- **Publication gates** — every cycle passes a gate engine (schema validity, source freshness, model provenance, calibration health, spatiotemporal sanity, uncertainty, replayability) with **pass / degrade / block** authority: a failing gate really does stop publication
- **Signed prediction ledger** — every forecast is SHA-256 hashed, Ed25519-signed, and committed before events occur (`dist/data/earthquake-ledger.jsonl`, `dist/data/evidence/`)
- **Replay artifacts** — full input snapshots per cycle (`dist/data/replay/`), independently verifiable with `scripts/verify_forecast.py`
- **Honest calibration** — Venn-Abers calibrated probabilities, reliability diagrams, distribution-drift monitoring, and abstention when the model shouldn't speak

No hiding. No cherry-picking. Every prediction logged, every outcome tracked — and the receipts are checkable by someone who doesn't trust us.

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
6. **The headline table is retrospective.** True validation requires prospective real-time testing with timestamped predictions — which the live platform now performs on every cycle. Prospective skill so far is materially weaker than the retrospective table (early prospective earthquake AUC ≈ 0.70 with calibrated Brier skill near climatology), which is exactly why the prospective scorecard, not the retrospective table, is the number that counts.
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
  url = {https://github.com/coherence-energy-labs/hazardpulse}
}
```

---

## License

Dual-licensed:

- **AGPL-3.0** for open-source use — see [LICENSE](LICENSE). If you run a modified HazardPulse as a network service, the AGPL requires you to publish your modifications.
- **Commercial license** for closed-source or proprietary-service use — see [COMMERCIAL_LICENSE.md](COMMERCIAL_LICENSE.md) or contact [info@coherenceenergylabs.com](mailto:info@coherenceenergylabs.com).

---

**HazardPulse** | [hazardpulse.com](https://hazardpulse.com) | [OneUnity.earth](https://oneunity.earth)
