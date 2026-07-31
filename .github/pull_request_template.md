# Summary

<!-- What changes, and why. -->

## Evidence

<!-- Every claim in this PR needs an artifact behind it. Delete rows that don't apply. -->

- [ ] Tests pass locally (`python -m pytest -q tests/`)
- [ ] Determinism gate passes (`python -m pytest -q tests/test_determinism.py`)
- [ ] No new unseeded randomness (all stochastic ops draw from an injected/seeded generator)
- [ ] Documented commands were actually run as written
- [ ] Public numbers changed? → the producing artifact/protocol is linked here

## Trust-critical surfaces touched?

<!-- model training / calibration / gate engine / receipts / verifier / workflows -->
<!-- If yes: describe the blast radius and how a reviewer can independently check it. -->
