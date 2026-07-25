# Earthquake precursor research scripts

Reproducibility scripts behind [`docs/earthquake_precursor_program.md`](../../docs/earthquake_precursor_program.md).
Research-grade (read the program doc for the honest verdicts). They read cached catalog/waveform
artifacts under `.cache/` built by the operational trainer and the dv/v toolkit.

| Script | What it does | Headline result |
|---|---|---|
| `dvv_lib.py` | Ambient-noise dv/v toolkit: FDSN node routing (CI→SCEDC etc.), disk-cached fetch, whitened station-pair cross-correlation, stretch-method dv/v, FDSN station finder | reusable library |
| `dvv_validate_ridgecrest.py` | Validates the **cross-correlation dv/v method** on Ridgecrest M7.1 (CLC/TOW2/SRT/WRC2) | co-seismic drop **−2.3σ** (method works) |
| `etas_benchmark.py` | Ogata ETAS conditional-intensity operational forecast vs the deep model | deep model **beats ETAS 0.666 by +0.064** |
| `catalog_features_incremental.py` | b-value / AMR / natural-time / Coulomb(GCMT) / tidal: univariate + incremental over model context | real but **~95% redundant** (+0.004) |
| `gnss_ridgecrest_test.py` | Common-mode-filtered GNSS deformation, near-vs-far, 3yr negative control | **null pre-slip**; proven 213 mm coseismic positive control |

Key reusable infrastructure: per-network FDSN routing (regional data lives at SCEDC/NCEDC/GEOFON/
ORFEUS/INGV, **not** IRIS) and transient-error-aware caching (only cache genuine HTTP-204 no-data,
retry timeouts/429 — never poison the cache with transient failures).

Notes: `dvv_lib` cache dir overridable via `DVV_CACHE`. The benchmark scripts read the cached
operational sample set `.cache/earthquake/deepop_v3_*.npz` produced by
`scripts/deep_operational_earthquake.py`.
