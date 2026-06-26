"""End-to-end tests for the TrustedForecaster wrapper (the heart of the trust spine).

Proves the four behaviours every scorer will rely on: calibration (lower ECE),
honest intervals, abstention (OOD / unhealthy data), and signed verifiable
receipts.
"""

from __future__ import annotations

import numpy as np
import pytest

from hazardpulse.trust import (
    TrustedForecaster,
    expected_calibration_error,
    verify_forecast_receipt,
)


def _overconfident(t):
    eps = 1e-6
    logit = np.log(np.clip(t, eps, 1 - eps) / np.clip(1 - t, eps, 1 - eps))
    return 1.0 / (1.0 + np.exp(-2.5 * logit))


def _make(seed=0, n=30000, d=6):
    rng = np.random.RandomState(seed)
    feats = rng.randn(n, d)
    # a true event probability driven by the features, then an overconfident head
    lin = feats @ rng.randn(d)
    t = 1.0 / (1.0 + np.exp(-lin))
    y = (rng.uniform(0, 1, size=n) < t).astype(float)
    raw = _overconfident(t)
    return raw, y, feats, rng


def _fit_forecaster(signer=None):
    raw, y, feats, rng = _make()
    cut = len(y) // 2
    tf = TrustedForecaster(model_version="test_v1", signer=signer).fit(
        raw[:cut], y[:cut], feats[:cut])
    return tf, raw[cut:], y[cut:], feats[cut:], rng


def test_calibration_improves_ece():
    tf, raw_te, y_te, feats_te, _ = _fit_forecaster()
    results = tf.forecast(raw_te, feats_te)
    keep = [(r.probability, yi) for r, yi in zip(results, y_te) if not r.abstained]
    p_cal = np.array([p for p, _ in keep])
    y_kept = np.array([yi for _, yi in keep])
    raw_kept = np.array([raw_te[i] for i, r in enumerate(results) if not r.abstained])
    raw_ece = expected_calibration_error(raw_kept, y_kept)
    cal_ece = expected_calibration_error(p_cal, y_kept)
    assert raw_ece > 0.05
    assert cal_ece < raw_ece * 0.6, f"calibration should cut ECE: {raw_ece:.3f} -> {cal_ece:.3f}"


def test_non_abstained_forecasts_have_intervals():
    tf, raw_te, y_te, feats_te, _ = _fit_forecaster()
    results = tf.forecast(raw_te[:200], feats_te[:200])
    for r in results:
        if not r.abstained:
            assert r.probability is not None
            assert r.confidence_lo is not None and r.confidence_hi is not None
            assert r.confidence_lo <= r.probability <= r.confidence_hi + 1e-9
            assert r.uncertainty_class in {"tight", "moderate", "wide"}
        c = r.as_contract()
        assert "confidence_lo" in c and "receipt_sha256" in c


def test_ood_input_abstains():
    tf, raw_te, _, _, rng = _fit_forecaster()
    far = rng.randn(20, 6) * 5.0 + 12.0          # far outside the calibration manifold
    results = tf.forecast([0.5] * 20, far)
    assert sum(r.abstained for r in results) >= 18
    abst = next(r for r in results if r.abstained)
    assert abst.abstain_reason == "out_of_distribution"
    assert abst.probability is None
    assert abst.gateway_mode == "DEGRADED"


def test_unhealthy_data_abstains():
    tf, raw_te, _, feats_te, _ = _fit_forecaster()
    r = tf.forecast_one(0.7, feats_te[0], data_health_ok=False)
    assert r.abstained and r.abstain_reason == "data_unhealthy"
    assert r.probability is None and r.confidence_lo is None
    c = r.as_contract()
    assert c["probability"] is None and c["abstained"] is True


def test_signed_receipt_verifies_and_tamper_is_caught():
    crypto = pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from hazardpulse.trust import load_ed25519_pubkey

    signer = Ed25519PrivateKey.generate()
    pubkey = load_ed25519_pubkey(signer.public_key().public_bytes_raw())

    tf, raw_te, _, feats_te, _ = _fit_forecaster(signer=signer)
    results = tf.forecast(raw_te[:30], feats_te[:30], issued_at="2026-06-25T00:00:00Z")
    for r in results:
        assert verify_forecast_receipt(r.receipt, pubkey=pubkey) is True

    tampered = dict(results[0].receipt)
    tampered["probability"] = (tampered["probability"] or 0.0) + 0.3
    assert verify_forecast_receipt(tampered, pubkey=pubkey) is False
    assert verify_forecast_receipt(tampered) is False           # integrity alone catches it


def test_receipt_is_reproducible_for_same_input():
    """Same input + same fitted model -> byte-identical receipt hash (re-runnable)."""
    tf, raw_te, _, feats_te, _ = _fit_forecaster()
    a = tf.forecast_one(raw_te[3], feats_te[3], issued_at="2026-06-25T00:00:00Z")
    b = tf.forecast_one(raw_te[3], feats_te[3], issued_at="2026-06-25T00:00:00Z")
    assert a.receipt["receipt_sha256"] == b.receipt["receipt_sha256"]
