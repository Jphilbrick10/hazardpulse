# Legacy Research Scripts

These are the original monolithic research scripts that produced all published results. They are preserved here for provenance and reproducibility.

Each script is self-contained — it downloads data, extracts features, trains models, evaluates, and generates figures in a single run. They were developed iteratively, with each version building on insights from the previous.

## Earthquake Models

| Script | AUC | Key Advance |
|---|---|---|
| `earthquake_tohoku_focused.py` | 0.861 | Focused epicentral analysis (100km radius) |
| `earthquake_sumatra_focused.py` | 0.746 | Extended radius for 1300km rupture zone |
| `earthquake_model_v3.py` | 0.854 | All M6+ globally (but trivial features) |
| `earthquake_model_v3b.py` | 0.747 | Change-only features, same-location controls |
| `earthquake_model_v4.py` | 0.796 | M4+ catalog, gradient boosted stumps |
| `earthquake_model_v5.py` | 0.835 | b-value trend, temporal acceleration |
| `earthquake_model_v6.py` | 0.894 | Moment release, Coulomb proxy, 5-model ensemble |

## Hurricane Models

| Script | AUC | Key Advance |
|---|---|---|
| `hurricane_ri_model.py` | 0.948 | IBTrACS features + RMW |
| `hurricane_ri_model_v2.py` | 0.963 | Pressure dynamics, MPI deficit, interactions |
| `hurricane_ri_model_v3.py` | 0.976 | Ocean heat proxy, Carnot efficiency, 4-model ensemble |

## Tornado Models

| Script | AUC | Key Advance |
|---|---|---|
| `tornado_prediction_model.py` | 0.950 (EF4+) | Width-severity scaling law (L ~ W^0.97) |
| `tornado_formation_model.py` | 0.780 | Grid-based formation (lost to climatology) |
| `tornado_model_v2.py` | 0.935 | Synoptic propagation (beats climatology) |
| `tornado_model_v3.py` | 0.956 | Multi-scale grids, ensemble stacking |
| `tornado_model_v4.py` | 0.973 | ERA5 proxies, synoptic patterns, 5-model ensemble |

## Running

Each script can be run independently:

```bash
python earthquake_model_v6.py
```

Scripts will download data from USGS/IBTrACS/SPC on first run and cache locally. Typical runtime: 5-15 minutes depending on network speed and model complexity.

**Note**: These scripts contain hardcoded Windows paths for figure output. Modify the output paths for your system, or use the refactored `src/tau_predict/` package instead.
