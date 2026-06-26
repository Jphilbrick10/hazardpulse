# HazardPulse → a platform that truly helps people: full build program

> Status: active program (started 2026-06-25, branch `platform-trust-program`).
> This is the canonical engineering program for turning HazardPulse into a genuinely
> trustworthy public-safety instrument. Full rationale + phase detail below.

## Context (re-baselined to current `main`, 2026-06-25)

HazardPulse is a live, self-verifying natural-hazard forecaster (earthquake / hurricane / tornado)
built on the Helmholtz coherence PDE, with a Cloudflare-Worker "truth surface" and a GitHub-Actions
scoring loop. The science and the product *spec* are strong, but the live, prospectively-scored
evidence shows the platform is **not yet trustworthy enough to help anyone**:

1. **It is miscalibrated to the point of being misleading.** Re-measured on current `main`
   (867 matured tornado forecasts vs 892 real tornadoes): tornado **AUC 0.66 but Brier-Skill = −0.71**,
   and the live ML tier (`tier1_ml`) is **BSS −0.88** — its *ranking* is okay but its *probabilities*
   are catastrophically miscalibrated, far worse than climatology. Earthquake (213 matured): **AUC 0.73
   but information-gain/event = −15.4 (negative)** and **top-cell hit-rate ≈ 0%** (top-10 = 3%,
   top-20 = 24%) — the highest-risk cells users look at almost never contain the real M6+. A hazard
   tool that says "20%" when it isn't really 20% is worse than no tool.
2. **The "truth surface" is scaffolded but hollow.** Every `gate-decisions.json` entry is `"pass"`
   carrying the same warning on *every* forecast: `"confidence_interval_unavailable"`. The platform
   literally admits, on each prediction, that it has **no uncertainty bands**. Gates never
   block/degrade; provenance/replay are emitted but the cryptographic-trust promise (signed,
   independently re-runnable) is only a SHA-256 hash chain.
3. **It has no idea when it's out of its depth.** No OOD detection, no abstention — it emits a
   confident number even for unprecedented setups or when input data is missing.
4. **It has been running unimproved.** The loop is alive (1193 automated data commits since
   2026-04-28) but **zero source-code changes in ~2 months** — it has simply been publishing
   increasingly-miscalibrated forecasts, not getting better.

The fix is not "another algorithm." Our sibling project **omega_one** is a mature, measured
**trust/audit ML layer** whose documented strengths — calibration (ECE ~0.04), OOD (AUROC 0.77–0.81),
conformal coverage guarantees, abstention, and Ed25519-signed re-runnable receipts — map one-to-one
onto HazardPulse's exact failures. Symmetrically, omega_one's own roadmap names its #1 highest-impact
gap as "validate the whole stack on a real, consequential dataset where abstention/coverage/audit
change an outcome." **HazardPulse is that dataset.** This is a two-way unlock, not a bolt-on.

**Intended outcome:** HazardPulse becomes a genuinely excellent public-safety instrument — every
forecast honestly calibrated, carrying real uncertainty bands, abstaining when it should,
cryptographically signed and independently replayable, continuously and visibly verified, and
communicated clearly. No thin modules, nothing decorative — every component load-bearing and measured.

## Guiding principles (non-negotiable)

- **Help-people-first.** A miscalibrated or overconfident hazard forecast can cause harm. Calibration,
  abstention, and honest uncertainty come *before* accuracy chasing.
- **No thin modules / no garbage.** Every module is load-bearing, tested, and measured against a
  baseline. We reuse the real omega_one engine; we do not reimplement a toy version of it.
- **Truth before polish, evidence before narrative** — the project's own Truth-Surface principles,
  now actually enforced.
- **Measure before/after, multi-seed, on the real prospective data** — never claim a win on one run.
- **Honest labels.** "exists today" vs "we are building." Gate fails → go harder, never descope.

## The integration thesis (how omega_one is consumed)

omega_one's trust layer (`conformal.py`, `ood_selector.py`, `trusted.py`, `regression.py`,
`super_ensemble.py`, `verifiable_forest.py`, `selective.py`, `guardian.py`, `compliance.py`,
`validation_pack.py`, `monitoring.py`) is **pure-numpy + `cryptography` (Ed25519)** — verified import
closure: `conformal`→`trusted`→`regression`, all numpy + stdlib only; `import omega` does not eagerly
load torch or the `.cl` toolchain. The `.cl`/0-ULP/CUDA cross-substrate receipt path *does* need the
vendored 88 MB toolchain — that's deferred to Phase 4.

**Decision:** For Phases 1–3, **vendor omega_one's numpy-only trust subset** into
`src/hazardpulse/trust/_vendor_omega/` with a documented re-vendor script (mirroring how omega_one
itself vendored `coherence_ml`). Rationale: HazardPulse CI is its own repo running `pip install -e .`
on GH Actions; vendoring gives deterministic, version-pinned CI with no heavy toolchain and no
cross-repo checkout. Add `cryptography` to `pyproject.toml`. A single `scripts/vendor_omega_trust.py`
keeps the subset in sync and pins a source commit hash so drift is visible.

---

## Phase 0 — Resurrect-baseline, diagnose, and build the iteration harness (foundation)

*The loop is alive but unimproved; the burning issue is calibration. This phase makes the pipeline
runnable + testable locally so we can iterate, and root-causes the research-vs-live skew.*

- **0.1 ✅ Re-baseline + re-measure (done 2026-06-25).** Loop confirmed alive; current numbers measured
  (above). No source changes in 2 months. Findings confirm calibration is the core problem.
- **0.2 Local end-to-end harness.** `scripts/run_local.py` runs each scorer against a **frozen sample
  of real input data** (USGS/SPC/HRRR/ATCF snapshots under `tests/_fixtures/`) so the full
  fetch→score→render→gate→ledger path runs offline, deterministically, in seconds. The inner loop.
- **0.3 Train/serve parity audit.** Dump the exact feature vector the live earthquake & tornado
  scorers build at serve time; compare feature-by-feature vs the research `definitive_model.py`
  training pipeline on the same event. Find the divergence (HRRR nulls → `mlcape` zeros; feature-order/
  normalization; live `compute_block_*` vs training path). Fix at root.
- **0.4 HRRR/ProbSevere ingest reality check** + hard ingest-health assertions: a forecast that
  silently lost atmospheric inputs must **degrade**, not emit a confident number.
- **0.5 Test + CI spine** (scoring math, gate emission, ledger hash-chain, HTML integrity; extend
  `tests/test_site_integrity.py`); make `liveness-check.yml` actually page and assert tier1_ml fires.
- **0.6 Lift training perf/row caps** (numba the hot path or let omega `super_ensemble` carry big data).

**Exit:** full pipeline runs locally on fixtures in one command; train/serve skew root-caused; CI runs
the new tests.

## Phase 1 — The trust spine (the core fix: calibration + uncertainty + abstention + signed receipts)

*The heart. Removes the `confidence_interval_unavailable` warning and makes "20% mean 20%."*

- **1.1 Vendor the omega trust subset** (`scripts/vendor_omega_trust.py` → `src/hazardpulse/trust/
  _vendor_omega/`), add `cryptography`, CI import smoke test.
- **1.2 One `TrustedForecast` wrapper** (`src/hazardpulse/trust/forecast.py`): any head's probability +
  feature vector → calibrated probability, **conformal interval** (real `confidence_lo/hi` populating
  `HazardForecastV1`), **OOD/novelty score** (`MahalanobisOOD`), **abstention** (`selective`
  risk-coverage / Chow + `CoherenceGuardian` modes), and an **Ed25519-signed re-runnable receipt**
  (`fast_trusted_decision`/`verify_trusted_receipt`). One wrapper, all three scorers — no per-hazard copies.
- **1.3 Calibrate each head honestly** on a held-out calibration split inside `definitive_model.py`'s
  temporal-split harness; persist calibration constants beside the model JSONs; measure ECE/reliability
  before/after.
- **1.4 Real gate engine** (`src/hazardpulse/gates/engine.py`): extract the inline gate-stamping from
  the four `fetch_and_score*.py` + `build_site_artifacts.py` into one engine that *actually evaluates*
  `G1_SOURCE_FRESHNESS`, `G3_PROVENANCE_COMPLETE`, `G4_CALIBRATION_FLOOR`, `G5_SPATIOTEMPORAL_SANITY`,
  `G6_ALERT_HARM_GUARD`, `G8_REPLAYABILITY_REQUIRED`; returns `degrade`/`block`; kill the universal
  `confidence_interval_unavailable` warning by *providing* the interval.
- **1.5 Wire abstention into the public surface.** OOD-high / DEGRADED → "insufficient signal /
  out-of-distribution — defer to official sources," not a confident number. The single most important
  public-safety behavior.
- **1.6 Signed independently-verifiable receipts.** Replace SHA-256-only with omega receipts
  (`verify_gbt_receipt`/`verify_trusted_receipt`); publish the public key; ship a tiny standalone
  verifier so a stranger can re-run a forecast and confirm it; wire worker
  `/api/v1/evidence|gates|forecast` to the real artifacts.

**Exit:** every published forecast carries a real conformal interval, OOD/abstention status, an
*evaluated* gate decision, and an Ed25519 receipt an independent script verifies; ECE improved on all
hazards; tornado BSS no longer negative.

## Phase 2 — Accuracy & data completeness (recover the gap, then push)

- **2.1** Swap hand-rolled GBTs for `BestTabular`/`SuperEnsemble` where it wins, under a never-worse
  guard; use `VerifiableForest` so the predictor emits a fixed-point signed receipt.
- **2.2** Close the tornado data gap (the real lever): robust, complete real-time HRRR
  CAPE/shear/SRH/helicity ingest (+ SPC mesoanalysis/RAP if needed).
- **2.3** Per-region / per-tier coverage via `MondrianConformalPredictor`/`group_coverage`
  (tectonic regions, EF tiers, basins).
- **2.4** Calibrated continuous-target severity/intensity heads (`TrustedRegression` +
  `AdaptiveConformalRegressor`).
- **2.5** Multi-seed temporal-block re-validation (`bootstrap_auc_ci`/`paired_bootstrap_test`); update
  `model_weights_registry.json` + `/methods` to honest re-measured numbers.

## Phase 3 — Verification & the public scoreboard (truth made visible)

- **3.1** Prospective scoring real + continuous; keep earthquake maturation flowing.
- **3.2** Honest scoreboard: reliability diagrams, Brier/Brier-skill, ROC-over-time, sharpness,
  hit/miss + false-alarm ledger — per hazard / region / tier, on `/verification`.
- **3.3** Drift monitoring (`ProductionMonitor`/`psi`) → auto-degrade + alert on input shift.
- **3.4** Calibration-in-the-loop: refresh calibration from verified outcomes (`AdaptiveConformal`).

## Phase 4 — 0-ULP cross-substrate receipts + federation (the uncontested moat)

- **4.1** Upgrade receipts to full 0-ULP via omega's `.cl` path: a decision re-runs bit-identically
  CPU == native == (optionally) CUDA, Ed25519-signed.
- **4.2** Wire the `signalbook` federation the worker already references (Ed25519 peer mesh + federated
  event-catalog backend).
- **4.3** (Optional, separate repo) `coherence_proof_fabric` Tier-0/1 ZK on high-stakes forecasts —
  consume its verifier pattern; do not build inside that repo.

## Phase 5 — Product & UX excellence

Full top-to-bottom rendered review of every route; real data in every panel (why-changed, Forecast-DNA,
evidence drawer, replay); WCAG AA, reduced-motion, non-color encodings, perf budgets, always-visible
disclaimers + official-agency links, never "warning/imminent"; public ledger + working in-browser
hash/receipt verify tool.

## Phase 6 — Hazard expansion (committed, last)

Flood / wildfire / extreme-heat — **only** once each meets the same bar (calibrated, abstaining, signed,
verified) the original three now hold. Deepen before widen.

---

## Cross-cutting (every phase)

- A test for every new module (scoring math, gate evaluation, conformal coverage guarantees, receipt
  verify/tamper, HTML integrity). No module ships without a test that fails if it breaks.
- Before/after measurements committed as JSON under `results/` (ECE, coverage, AUC, BSS).
- Honest docs updated in lockstep (`README`, `docs/results.md`, `/methods`) — re-measured numbers only.

## Key files

- Live scorers: `scripts/fetch_and_score_earthquake.py`, `scripts/fetch_and_score_tornado.py`,
  `scripts/fetch_and_score.py`, `scripts/build_site_artifacts.py`.
- Models / calibration: `src/hazardpulse/{earthquake,tornado}/definitive_model.py` (training + metrics:
  `compute_auc`/`compute_bss`/`bootstrap_auc_ci`/`paired_bootstrap_test`), `results/models/*.json`,
  `results/models/model_weights_registry.json`.
- New: `src/hazardpulse/trust/forecast.py`, `src/hazardpulse/trust/_vendor_omega/`,
  `src/hazardpulse/gates/engine.py`, `scripts/vendor_omega_trust.py`, `scripts/run_local.py`.
- Verification: `scripts/score_*_prospective.py`, `results/verification/*/live_rollup.json`.
- Worker/UX: `src/worker.js`, `dist/` pages + `dist/data/evidence/*`.
- omega_one source of truth: `C:\Users\Josh\Projects\Coherence\omega_one\omega\`.

## Honest caveats

- omega_one is **not** an accuracy moonshot — it ties XGBoost, beats the rest of the GBDT field ~2.9pt.
  Its win here is the **trust/calibration/abstention/signed** axes (HazardPulse's actual failures).
- Tornado raw skill is gated by **data completeness (HRRR)** — Phase 2 — not the model class.
- Earthquake prediction is genuinely hard; the goal is **honest, calibrated, abstaining** forecasts,
  not certainty.
- The 0-ULP `.cl` path needs the vendored toolchain (Phase 4); the numpy receipt (Phase 1) is the 80%.
- `coherence_proof_fabric` is a separate repo — consume its pattern, don't build there.
