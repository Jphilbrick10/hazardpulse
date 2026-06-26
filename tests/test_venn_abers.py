"""Tests for Venn-Abers calibration (hazardpulse.trust.venn_abers).

The load-bearing property: it turns a miscalibrated forecaster into a calibrated
one (lower ECE), while emitting an honest [p0, p1] band around each probability.
"""

from __future__ import annotations

import numpy as np

from hazardpulse.trust.calibration import expected_calibration_error
from hazardpulse.trust.venn_abers import VennAbersCalibrator, pav_isotonic


def test_pav_isotonic_basic():
    fit = pav_isotonic(np.array([3.0, 1.0, 2.0, 4.0]), np.ones(4))
    np.testing.assert_allclose(fit, [2.0, 2.0, 2.0, 4.0])
    # already-monotone input is unchanged
    mono = np.array([0.1, 0.2, 0.2, 0.9])
    np.testing.assert_allclose(pav_isotonic(mono, np.ones(4)), mono)


def _overconfident(t):
    """Push a true probability toward 0/1 (a classic miscalibration)."""
    eps = 1e-6
    logit = np.log(np.clip(t, eps, 1 - eps) / np.clip(1 - t, eps, 1 - eps))
    return 1.0 / (1.0 + np.exp(-2.5 * logit))


def test_venn_abers_fixes_miscalibration():
    rng = np.random.RandomState(0)
    t = rng.uniform(0, 1, size=40000)
    y = (rng.uniform(0, 1, size=t.size) < t).astype(float)
    raw = _overconfident(t)                      # badly overconfident scores
    cut = t.size // 2

    va = VennAbersCalibrator().fit(raw[:cut], y[:cut])
    p_cal, lo, hi = va.predict(raw[cut:])

    raw_ece = expected_calibration_error(raw[cut:], y[cut:])
    cal_ece = expected_calibration_error(p_cal, y[cut:])
    assert raw_ece > 0.08, f"raw should be miscalibrated, got ECE {raw_ece:.3f}"
    assert cal_ece < raw_ece * 0.5, f"Venn-Abers should roughly halve ECE: {raw_ece:.3f} -> {cal_ece:.3f}"
    assert not va.inflated


def test_interval_brackets_point_and_is_valid():
    rng = np.random.RandomState(1)
    t = rng.uniform(0, 1, size=20000)
    y = (rng.uniform(0, 1, size=t.size) < t).astype(float)
    va = VennAbersCalibrator().fit(t[:10000], y[:10000])
    p, lo, hi = va.predict(t[10000:])
    assert np.all(lo <= p + 1e-9) and np.all(p <= hi + 1e-9)
    assert np.all(lo >= 0) and np.all(hi <= 1)


def test_calibrated_input_stays_calibrated():
    rng = np.random.RandomState(2)
    t = rng.uniform(0, 1, size=30000)
    y = (rng.uniform(0, 1, size=t.size) < t).astype(float)
    va = VennAbersCalibrator().fit(t[:15000], y[:15000])
    p, _, _ = va.predict(t[15000:])
    assert expected_calibration_error(p, y[15000:]) < 0.03


def test_monotone_in_score():
    rng = np.random.RandomState(3)
    t = rng.uniform(0, 1, size=20000)
    y = (rng.uniform(0, 1, size=t.size) < t).astype(float)
    va = VennAbersCalibrator().fit(t, y)
    grid = np.linspace(0.01, 0.99, 50)
    p, _, _ = va.predict(grid)
    assert np.all(np.diff(p) >= -1e-9), "calibrated probability must be non-decreasing in score"


def test_too_few_points_flags_inflated_passthrough():
    va = VennAbersCalibrator(min_calibration=50).fit(np.array([0.2, 0.8]), np.array([0.0, 1.0]))
    assert va.inflated
    r = va.predict_one(0.4)
    assert abs(r.probability - 0.4) < 1e-9 and r.upper > r.lower  # wide honest band
