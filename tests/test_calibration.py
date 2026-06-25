"""Tests for the binary-forecast calibration module (hazardpulse.trust.calibration)."""

from __future__ import annotations

import numpy as np

from hazardpulse.trust.calibration import (
    brier_decomposition,
    brier_score,
    brier_skill_score,
    expected_calibration_error,
    reliability_curve,
)


def test_perfect_calibration_has_near_zero_ece():
    """If outcomes are drawn Bernoulli(p), the forecaster p is calibrated -> ECE ~ 0."""
    rng = np.random.RandomState(0)
    p = rng.uniform(0, 1, size=60000)
    y = (rng.uniform(0, 1, size=p.size) < p).astype(float)
    ece = expected_calibration_error(p, y, n_bins=10)
    assert ece < 0.02, f"calibrated forecaster should have small ECE, got {ece:.4f}"
    decomp = brier_decomposition(p, y, n_bins=10)
    assert decomp.reliability < 0.005


def test_overconfident_forecaster_is_penalized():
    """Saying 0.85 when the event happens ~15% of the time is badly miscalibrated."""
    rng = np.random.RandomState(1)
    y = (rng.uniform(0, 1, size=20000) < 0.15).astype(float)
    p = np.full_like(y, 0.85)
    ece = expected_calibration_error(p, y)
    assert ece > 0.5, f"overconfident forecaster should have large ECE, got {ece:.3f}"
    assert brier_skill_score(p, y) < 0.0, "miscalibrated forecast should be worse than climatology"


def test_brier_decomposition_identity():
    """Murphy: Brier == reliability - resolution + uncertainty (exact for discrete forecasts)."""
    rng = np.random.RandomState(2)
    levels = np.linspace(0.05, 0.95, 10)        # one value per decile bin -> zero within-bin variance
    p = rng.choice(levels, size=40000)
    # outcomes correlated with p but not perfectly (gives nonzero reliability + resolution)
    true = np.clip(0.7 * p + 0.1, 0, 1)
    y = (rng.uniform(0, 1, size=p.size) < true).astype(float)
    d = brier_decomposition(p, y, n_bins=10)
    recon = d.reliability - d.resolution + d.uncertainty
    assert abs(d.brier - recon) < 1e-9, f"decomposition identity off by {d.brier - recon:.2e}"


def test_climatology_forecast_has_zero_skill():
    """Forecasting the constant base rate gives Brier == uncertainty and BSS == 0."""
    rng = np.random.RandomState(3)
    y = (rng.uniform(0, 1, size=10000) < 0.27).astype(float)
    base = float(y.mean())
    p = np.full_like(y, base)
    assert abs(brier_skill_score(p, y)) < 1e-9
    d = brier_decomposition(p, y)
    assert abs(brier_score(p, y) - d.uncertainty) < 1e-9


def test_reliability_curve_diagonal_for_calibrated():
    rng = np.random.RandomState(4)
    p = rng.uniform(0, 1, size=40000)
    y = (rng.uniform(0, 1, size=p.size) < p).astype(float)
    curve = reliability_curve(p, y, n_bins=10)
    for b in curve.bins:
        if b.count > 100:
            assert abs(b.mean_predicted - b.observed_freq) < 0.05
