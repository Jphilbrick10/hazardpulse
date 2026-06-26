"""Forecast drift monitoring.

A deployed calibrator is only valid while the live forecast distribution looks
like the one it was fit on. When the input/forecast distribution shifts (a new
seismic sequence, a different storm regime), the calibration can go stale and the
published probabilities drift out of true. The Population Stability Index (PSI)
quantifies that shift; this wraps omega_one's measured ``psi`` into a small,
status-bearing helper the scoreboard and live monitor use.

Thresholds follow the standard model-risk rule: PSI < 0.1 stable, 0.1-0.25
moderate drift, > 0.25 major drift.
"""

from __future__ import annotations

import numpy as np

from ._vendor_omega.monitoring import ProductionMonitor, psi

__all__ = ["ProductionMonitor", "psi", "forecast_drift_status", "expand_histogram"]


def expand_histogram(scores, total, *, cap: int = 5000):
    """Turn a (score, count) histogram into a bounded representative sample so it
    can be used as a PSI reference without materialising millions of points."""
    s = np.asarray(scores, dtype=np.float64).ravel()
    t = np.asarray(total, dtype=np.float64).ravel()
    n = float(t.sum())
    if n <= 0 or s.size == 0:
        return np.asarray([], dtype=np.float64)
    reps = np.maximum(1, np.round(t / n * cap)).astype(int)
    return np.repeat(s, reps)


def forecast_drift_status(reference_scores, current_scores, *, bins: int = 10) -> dict:
    """PSI between a reference and current forecast distribution + a status label."""
    ref = np.asarray(reference_scores, dtype=np.float64).ravel()
    cur = np.asarray(current_scores, dtype=np.float64).ravel()
    ref = ref[np.isfinite(ref)]
    cur = cur[np.isfinite(cur)]
    if ref.size < 2 or cur.size < 2:
        return {"psi": 0.0, "status": "insufficient_data"}
    value = psi(ref, cur, bins=bins)
    if value > 0.25:
        status = "major"
    elif value > 0.10:
        status = "moderate"
    else:
        status = "stable"
    return {"psi": round(float(value), 4), "status": status}
