"""Forecast drift monitoring (PSI) tests."""

from __future__ import annotations

import numpy as np

from hazardpulse.trust import ProductionMonitor, forecast_drift_status, psi
from hazardpulse.trust.monitor import expand_histogram


def test_exports_available():
    assert callable(psi) and callable(forecast_drift_status) and ProductionMonitor is not None


def test_stable_when_distribution_unchanged():
    rng = np.random.RandomState(0)
    ref = rng.uniform(0, 1, 4000)
    cur = rng.uniform(0, 1, 4000)
    out = forecast_drift_status(ref, cur)
    assert out["status"] == "stable" and out["psi"] < 0.1


def test_major_when_distribution_shifts():
    rng = np.random.RandomState(1)
    ref = rng.uniform(0.0, 0.3, 4000)
    cur = rng.uniform(0.6, 1.0, 4000)
    out = forecast_drift_status(ref, cur)
    assert out["status"] == "major" and out["psi"] > 0.25


def test_insufficient_data():
    assert forecast_drift_status([0.1], [0.2, 0.3])["status"] == "insufficient_data"


def test_expand_histogram_preserves_proportions():
    scores = np.array([0.1, 0.9])
    total = np.array([900, 100])          # 90% low, 10% high
    sample = expand_histogram(scores, total, cap=1000)
    frac_low = float(np.mean(sample < 0.5))
    assert 0.85 < frac_low < 0.95
