# HazardPulse Classic Truth Surface Specification

Version: 1.0.0  
Status: Draft for immediate implementation  
Scope: Public website + live prediction platform surfaces for `hazardpulse.io`  
Mode policy: Single mode only (`classic`), no future-mode split

---

## 1. Mission

HazardPulse is not a marketing website.  
HazardPulse is a public hazard intelligence instrument that renders:

- live probabilistic forecasts,
- immutable prediction evidence,
- deterministic verification,
- transparent uncertainty,
- and replayable decision history.

The product must feel modern and futuristic without visual clutter.  
The experience must be calm, precise, and deeply trustworthy.

---

## 2. Product North Star

### 2.1 Primary promise

For every published hazard output, users can answer:

1. What is the current probability and confidence?
2. Why did this number change?
3. What exact model/data produced it?
4. Did it pass governance gates?
5. Can this decision be replayed exactly?

### 2.2 Success definition

The platform is successful when:

- users can understand risk in under 30 seconds,
- advanced users can audit any claim end-to-end in under 2 minutes,
- no publishable forecast exists without full provenance,
- and public trust improves through visible verification, not claims.

---

## 3. Non-Negotiable Design Principles

1. Truth before polish.
2. Evidence before narrative.
3. Determinism before convenience.
4. Clarity before feature density.
5. Confidence and uncertainty must always co-exist.
6. All key claims must be one-click auditable.
7. No decorative complexity that does not improve comprehension.
8. No hidden model state behind unexplained scores.
9. No "official warning" language; only probabilistic risk framing.
10. Every page is a trust surface.

---

## 4. Experience Direction (Modern, Futuristic, Uncluttered)

### 4.1 Visual intent

- Minimal geometric layout, high information density with strict hierarchy.
- Dark-neutral base, hazard-accent colors used sparingly for salience.
- Motion is purposeful and subtle (state transitions, focus shifts), never flashy.
- Grid-based composition with strong whitespace discipline.

### 4.2 Tone

- Scientific, calm, operational.
- Never alarmist, never sensational.
- Language is specific and bounded ("elevated probability", "confidence band", "data freshness").

### 4.3 Information rhythm

Each page uses the same top-to-bottom reading order:

1. Current state
2. Delta from previous cycle
3. Why it changed
4. Confidence + uncertainty
5. Gate status
6. Evidence and replay

---

## 5. Information Architecture

## 5.1 Top-level routes

- `/` Home instrument overview
- `/live` Unified live map + hazard pulse strip
- `/live/earthquake`
- `/live/hurricane`
- `/live/tornado`
- `/evidence` Public ledgers and verification artifacts
- `/verification` Calibration, Brier, reliability, log score, historical rollups
- `/methods` Methodology, data contracts, caveats
- `/registry` Model registry and promotion/rejection ledger
- `/api` API docs and contract explorer
- `/ops/status` Build/publish/gate status and incident notes
- `/legal/disclaimer`

### 5.2 Utility routes

- `/search`
- `/feed/rss`
- `/sitemap.xml`
- `/robots.txt`
- `/_headers`

---

## 6. Core User Journeys

### 6.1 Public user (fast decision)

1. Open `/live`.
2. Search location.
3. Read hazard cards with probability + confidence.
4. Expand "Why changed" summary.
5. Follow official agency links.

Target: useful risk comprehension in less than 30 seconds.

### 6.2 Analyst/researcher (deep audit)

1. Open hazard detail page.
2. Inspect model version and feature digest.
3. Open gate report.
4. Open evidence packet.
5. Run replay in browser or fetch replay artifact.

Target: full claim audit in less than 2 minutes.

### 6.3 Partner/API user

1. Open `/api`.
2. Inspect endpoint contracts + schema versions.
3. Validate signature fields and freshness contracts.
4. Integrate signed forecast stream.

Target: first successful integration in less than 1 day.

---

## 7. Page Specifications

## 7.1 Home (`/`)

Must include:

- hero statement: "Hazard intelligence you can verify."
- global pulse strip: EQ/HU/TO current state cards
- "What changed since last cycle" module
- trust strip: freshness, gate pass rate, replayability rate
- direct CTA to `/live`, `/evidence`, `/verification`

Must not include:

- inflated headline claims,
- marketing animation walls,
- ambiguous "AI predicts disasters" messaging.

### 7.2 Live (`/live`)

Must include:

- map panel with hazard layer toggles
- location search
- time-window selector per hazard
- hazard pulse cards with:
  - probability,
  - uncertainty band,
  - confidence quality,
  - data freshness,
  - gate badge,
  - evidence quick-link

### 7.3 Hazard detail pages

Each hazard page must include:

- current forecast state
- trend over recent windows
- "why changed" decomposition panel
- uncertainty calibration panel
- model/version lineage panel
- gate decision summary
- evidence packet link
- replay link

### 7.4 Evidence (`/evidence`)

Must include:

- immutable prediction ledger browser
- provenance envelopes
- gate decision ledger
- model promotion/rejection ledger
- export options (JSON, NDJSON, CSV)

### 7.5 Verification (`/verification`)

Must include:

- rolling AUC
- Brier score + Brier skill
- reliability diagrams
- sharpness distribution
- hit/miss and false alarm debt
- hazard- and region-specific breakdowns

### 7.6 Methods (`/methods`)

Must include:

- model caveats and boundaries
- evaluation protocol
- data source descriptions and update latencies
- legal and interpretation guidance

---

## 8. SiteWorld Model (Classic-Only)

SiteWorld remains core even in single classic mode.

### 8.1 Required node types

- `route`
- `hazard`
- `forecast_artifact`
- `verification_artifact`
- `evidence_artifact`
- `model_version`
- `gate`
- `data_source`
- `policy`

### 8.2 Required edge types

- `explains`
- `derived_from`
- `verified_by`
- `gated_by`
- `supersedes`
- `references`
- `constrained_by`

### 8.3 Lens definitions

- `public`: plain-language risk + official guidance
- `analyst`: full decomposition + metrics
- `audit`: complete provenance and gate trace
- `ops`: freshness/latency/incident state

---

## 9. Data Contracts (Canonical)

All public contracts must be versioned and immutable.

### 9.1 `HazardForecastV1`

Fields:

- `forecast_id: String`
- `hazard_type: String` (`earthquake|hurricane|tornado`)
- `issued_at: Timestamp`
- `valid_from: Timestamp`
- `valid_to: Timestamp`
- `scope: Object` (grid cell / storm id / region)
- `probability: Float` (0.0-1.0)
- `confidence_lo: Float`
- `confidence_hi: Float`
- `uncertainty_class: String`
- `delta_from_prev: Float`
- `model_version: String`
- `features_digest: String`
- `inference_digest: String`
- `provenance_id: String`
- `gate_decision_id: String`

### 9.2 `ProvenanceEnvelopeV1`

Fields:

- `provenance_id: String`
- `trace_id: String`
- `parent_trace_ids: List[String]`
- `input_hash: String`
- `output_hash: String`
- `source_refs: List[String]`
- `transform_refs: List[String]`
- `signer: String`
- `signed_at: Timestamp`

### 9.3 `GateDecisionV1`

Fields:

- `gate_decision_id: String`
- `gate_set_version: String`
- `decision: String` (`pass|block|degrade`)
- `blocking_reasons: List[String]`
- `warnings: List[String]`
- `emitted_at: Timestamp`
- `artifact_refs: List[String]`

### 9.4 `VerificationRecordV1`

Fields:

- `verification_id: String`
- `forecast_id: String`
- `outcome_observed: Bool`
- `observed_at: Timestamp`
- `brier_contrib: Float`
- `log_score: Float`
- `calibration_bin: String`
- `verifier_version: String`

---

## 10. Gate Constitution (Hard Publish Policy)

All hazard publishes must pass the full gate spine.

### 10.1 Required gates

1. `G0_SCHEMA_VALIDITY`
2. `G1_SOURCE_FRESHNESS`
3. `G2_MODEL_LINEAGE_PINNED`
4. `G3_PROVENANCE_COMPLETE`
5. `G4_CALIBRATION_FLOOR`
6. `G5_SPATIOTEMPORAL_SANITY`
7. `G6_ALERT_HARM_GUARD`
8. `G7_EXPLANATION_MINIMUM`
9. `G8_REPLAYABILITY_REQUIRED`
10. `G9_PUBLIC_PROJECTION_POLICY`
11. `G10_SECURITY_POLICY`
12. `G11_PERFORMANCE_BUDGETS`
13. `G12_TRUST_SURFACE_SYNC`

### 10.2 Gate outcomes

- `pass`: fully publish
- `degrade`: publish with constrained presentation + explicit warning banner
- `block`: do not publish forecast artifacts to public surfaces

### 10.3 Gate transparency rule

Every user-facing forecast must expose:

- latest gate decision,
- failing/warning reasons,
- time of evaluation,
- evidence links.

---

## 11. `.cl` Module Architecture

## 11.1 Top-level package layout

- `site/build.cl`
- `site/router.cl`
- `site/render/layout.cl`
- `site/render/hazard_cards.cl`
- `site/render/evidence_drawer.cl`
- `site/render/verification_panels.cl`
- `site/render/disclaimer_panels.cl`
- `site/siteworld_loader.cl`
- `site/json_ld_hazard.cl`
- `site/search_index.cl`
- `pipeline/content_pipeline.cl`
- `pipeline/forecast_publish.cl`
- `pipeline/verification_rollup.cl`
- `pipeline/replay_artifacts.cl`
- `data/contracts.cl`
- `data/provenance_chain.cl`
- `data/model_registry.cl`
- `gates/hard_gates.cl`
- `gates/seo_security_perf.cl`
- `ops/status_report.cl`

### 11.2 Build orchestration (`build.cl`)

`build.cl` stages:

1. Validate input contracts
2. Compile SiteWorld
3. Build content artifacts
4. Build verification summaries
5. Execute gate spine
6. Render public pages
7. Emit headers/sitemap/rss/search index
8. Emit build + gate report

---

## 12. Runtime and API Surfaces (Classic-Compatible)

Platform can be static-first with selective dynamic endpoints.

### 12.1 Core endpoints

- `GET /api/v1/live/pulse`
- `GET /api/v1/live/{hazard}`
- `GET /api/v1/forecast/{forecast_id}`
- `GET /api/v1/evidence/{provenance_id}`
- `GET /api/v1/gates/{gate_decision_id}`
- `GET /api/v1/verification/summary`
- `GET /api/v1/registry/models`
- `GET /api/v1/replay/{forecast_id}`

### 12.2 Streaming endpoints

- `GET /stream/live/pulse` (SSE)
- `GET /stream/ops/status` (SSE)

### 12.3 API principles

- strict schema versioning
- deterministic response envelopes
- full cache directives
- explicit deprecation windows

---

## 13. Trust Surface Components

Every hazard card supports:

- probability + uncertainty
- freshness chip
- gate status chip
- model version chip
- evidence link
- replay link

### 13.1 Forecast DNA panel

Compact panel showing:

- model lineage id,
- feature digest hash,
- inference digest hash,
- calibration bucket,
- gate outcome summary.

### 13.2 Why-changed panel

Human-readable delta summary with bounded claims:

- top contributing factors,
- data update references,
- model/version changes (if any),
- confidence movement.

---

## 14. Performance and Delivery Budgets

### 14.1 Core budgets

- HTML route payload: <= 120 KB compressed
- CSS total critical + deferred: <= 80 KB compressed
- above-the-fold render target: <= 1.8 s on mid-tier mobile
- TTFB target: <= 400 ms for cached routes
- LCP target: <= 2.2 s
- INP target: <= 160 ms
- CLS target: <= 0.05

### 14.2 Build determinism

Given same input artifacts + config:

- generated HTML must be byte-stable,
- siteworld projection must be stable,
- gate output ids must be reproducible.

---

## 15. Security and Policy

### 15.1 Mandatory headers

- strict transport security
- content security policy
- x-content-type-options
- x-frame-options
- referrer-policy
- permissions-policy

### 15.2 Public language policy

Always display:

- experimental research disclaimer,
- links to official agencies (USGS/NHC/NWS/SPC),
- uncertainty bounds,
- no official warning terminology.

---

## 16. Observability

### 16.1 Event classes

- `page_view`
- `search_query`
- `layer_toggle`
- `evidence_open`
- `replay_open`
- `gate_warning_viewed`
- `api_contract_viewed`

### 16.2 Ops metrics

- data freshness lag by source
- publish cadence adherence
- gate pass/block/degrade rates
- replayability coverage %
- evidence completeness %
- per-hazard calibration drift

---

## 17. Accessibility

Requirements:

- keyboard-complete navigation
- visible focus states
- WCAG AA contrast minimum
- semantic landmarks and heading order
- reduced-motion mode
- hazard color encodings with non-color redundancy

---

## 18. Content and Narrative System

### 18.1 Copy style

- short declarative sentences
- no hype adjectives
- no certainty language where uncertainty exists
- every claim accompanied by one evidence affordance

### 18.2 Disclosure pattern

Use progressive disclosure:

1. Essential risk signal
2. Basic explanation
3. Deep evidence
4. Full replay artifact

---

## 19. Implementation Phases

### Phase A: Spec spine and contracts (2 weeks)

Deliver:

- this spec accepted as canonical
- contract definitions in `.cl`
- initial gate interfaces

Exit criteria:

- all contract tests passing
- zero undocumented payload fields

### Phase B: Core classic surfaces (3 weeks)

Deliver:

- `/`, `/live`, per-hazard detail pages
- evidence drawer on all hazard cards
- initial verification page

Exit criteria:

- all pages valid and navigable
- baseline SEO/security/perf gates passing

### Phase C: Gate constitution and ledgering (3 weeks)

Deliver:

- hard gates `G0..G12`
- public gate and provenance ledgers
- degrade/block public behavior

Exit criteria:

- blocked publishes never reach public routes
- 100% public forecasts have gate + provenance ids

### Phase D: Replayability and trust acceleration (3 weeks)

Deliver:

- replay artifacts for all forecasts
- replay viewer route + API
- trust scoreboards by hazard and region

Exit criteria:

- 100% of sampled forecasts replay successfully
- trust panel on all live hazard pages

---

## 20. Acceptance Criteria (Release Gate)

Release is allowed only if all are true:

1. all hard publish gates green,
2. no missing provenance on public forecasts,
3. verification pipeline emits daily rollups,
4. replayability coverage >= 99.9% of public forecasts,
5. performance budgets green on top routes,
6. policy/disclaimer checks green,
7. deterministic rebuild check green.

---

## 21. Immediate Implementation Backlog

1. Create `site/`, `pipeline/`, `gates/`, `data/`, `ops/` module shells with concrete interfaces.
2. Implement canonical contract types (`HazardForecastV1`, `ProvenanceEnvelopeV1`, `GateDecisionV1`, `VerificationRecordV1`).
3. Implement initial gate runner with stubbed but executable checks.
4. Build first route set (`/`, `/live`, `/live/{hazard}`, `/evidence`, `/verification`).
5. Integrate evidence drawer and Forecast DNA panel on live hazard cards.
6. Emit `_headers`, `sitemap.xml`, `rss`, and deterministic build report.
7. Add contract + gate + rendering tests for release confidence.

---

## 22. Governance and Change Control

- This file is the canonical website/platform spec.
- Changes require:
  - rationale,
  - backward-compatibility impact,
  - contract version notes,
  - gate impact notes.
- Any change weakening provenance, replayability, or gate visibility is disallowed unless explicitly approved and time-bounded.

---

## 23. Final Product Statement

HazardPulse classic mode must look simple and feel futuristic because the complexity is in the truth spine, not in visual clutter.

The user should experience:

- immediate clarity,
- deep transparency,
- and confidence that every number can be audited.

If a feature does not improve trust, understanding, or response quality, it does not ship.

