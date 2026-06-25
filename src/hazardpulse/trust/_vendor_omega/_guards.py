"""
_guards.py - shared input-validation guards (the ROOT fix).

Two adversarial reviews surfaced the SAME small set of failure classes scattered across modules:
  - FAIL-OPEN GATES: `if x < thresh: reject` silently PASSES when x is NaN (NaN < thresh is
    False), so a NaN evidence/confidence defeats the gate. The safe default is fail-CLOSED.
  - NON-FINITE INPUT: NaN / inf reaching a comparison, a threshold, or a signed receipt.
  - MALFORMED / EMPTY INPUT: an empty array into argmin/quantile/mean, float(None), etc.

Rather than re-derive (or miss) the guard at each site, every module imports these. The rule is
fail-closed: a non-finite confidence/evidence collapses to 0.0 (the LEAST-trusted value), so a
NaN can never pass a `>=` gate and can never be the most-confident option.
"""
from __future__ import annotations

import math

import numpy as np

__all__ = ["is_finite", "finite_or", "clamp01", "all_finite", "require_finite_2d", "safe_argmin"]


def is_finite(x) -> bool:
    """True iff x is a finite real scalar (not NaN/inf/None/non-numeric)."""
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def finite_or(x, default=0.0) -> float:
    """x as a float if finite, else `default`. Use when consuming an evidence/confidence/threshold
    so a NaN/inf/None fails CLOSED instead of slipping through a `<`/`>=` comparison."""
    return float(x) if is_finite(x) else float(default)


def clamp01(x, default=0.0) -> float:
    """Finite-guarded clamp to [0,1] for a confidence/evidence score. NaN/inf -> `default`
    (0.0 = least trusted), so it can never pass a confidence gate or win an argmax."""
    return float(np.clip(finite_or(x, default), 0.0, 1.0))


def all_finite(arr) -> bool:
    """True iff `arr` is non-empty AND every element is finite. Empty is False (you cannot
    certify finiteness of nothing - callers that want vacuous-true should check size first)."""
    a = np.asarray(arr, float)
    return bool(a.size) and bool(np.all(np.isfinite(a)))


def require_finite_2d(X, name="X"):
    """Return X as a finite 2-D float array, or raise ValueError. For fit paths that must reject
    non-finite features at the boundary rather than let a NaN propagate into a covariance."""
    a = np.asarray(X, float)
    if a.ndim != 2 or a.size == 0:
        raise ValueError(f"{name} must be a non-empty 2-D array, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values (NaN/inf)")
    return a


def safe_argmin(values):
    """argmin among the FINITE entries; None if the input is empty or every entry is non-finite
    (so a caller abstains instead of crashing the batch or selecting a NaN/inf). A non-finite
    entry is never selected."""
    v = np.asarray(values, float)
    finite = np.isfinite(v)
    if not finite.any():                    # empty, or every entry non-finite -> nothing to pick
        return None
    v = np.where(finite, v, np.inf)         # non-finite entries can never be the min
    return int(v.argmin())
