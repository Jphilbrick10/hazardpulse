# Governance

HazardPulse is developed by [Coherence Energy Labs](https://coherenceenergylabs.com).

## Roles

- **Maintainer:** Josh Philbrick ([@Jphilbrick10](https://github.com/Jphilbrick10)) - final
  authority on merges, releases, and public claims.
- **Contributors:** anyone, via pull request. The most valuable external
  contribution is an independent replication report (pass or fail).

## Change policy

- `main` history is protected by an active ruleset: force pushes and branch
  deletion are blocked for everyone, with no bypass actors. Code changes land by
  pull request with passing status checks; the scheduled scoring workflows
  commit forecast data directly (writing only under `dist/`), which is why
  PR-only enforcement is not yet mechanical - it becomes so when forecast data
  moves out of the code history (planned; see the hardening roadmap in PR #3).
- **Trust-critical surfaces** (model training, calibration, the gate engine,
  signed receipts, the verifier, CI workflows) additionally require maintainer
  review via CODEOWNERS - including for changes authored by the maintainer's own
  tooling.
- Every public performance claim must trace to a committed artifact. A claim
  that cannot be regenerated or independently verified is a defect, handled like
  any other bug.

## Honest-single-maintainer disclosure

This is presently a single-maintainer project that makes heavy, disclosed use of
AI-assisted engineering. We compensate with mechanical review surfaces (protected
branches, required checks, CODEOWNERS), determinism gates that cannot be
argued with, externally verifiable receipts, and a standing invitation for
independent replication. External reviewers - meteorological, seismological, and
numerical - are actively sought; contact
[info@coherenceenergylabs.com](mailto:info@coherenceenergylabs.com).
