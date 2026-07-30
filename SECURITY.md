# Security Policy

## Reporting a vulnerability

Email **security@coherenceenergylabs.com**. We aim to acknowledge reports within
72 hours. Please include a reproduction path and, if relevant, the affected
forecast artifacts or receipts.

Please do **not** open a public issue for a vulnerability before we have had a
chance to respond — coordinated disclosure protects users of the live
forecasting surfaces.

## Scope

- The `hazardpulse` Python package (`src/hazardpulse/`)
- The scoring/publication pipeline (`scripts/`, `.github/workflows/`)
- The signed-receipt scheme and its verifier (`scripts/verify_forecast.py`)
- The published site artifacts (`dist/`)

## What counts as a security issue here

Beyond conventional vulnerabilities (code execution, credential exposure,
supply-chain injection), this project treats **integrity failures as security
issues**: anything that would let a forecast be altered after issuance, a
receipt to verify for data it does not bind, a publication gate to be bypassed,
or a replay artifact to diverge from what was actually scored.

## Supply chain

- Workflow actions are pinned; dependencies are version-bounded.
- The receipt verifier is deliberately self-contained (stdlib + `cryptography`)
  so verification does not inherit this repository's dependency surface.

See also: the organization-wide policy at
[coherenceenergylabs.com/security](https://coherenceenergylabs.com/security/) and
our RFC 9116 [security.txt](https://coherenceenergylabs.com/.well-known/security.txt).
