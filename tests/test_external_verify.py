"""The standalone zero-trust verifier (scripts/verify_forecast.py).

Proves (1) the verifier's independently re-implemented receipt hash matches the
spec our signer uses, (2) it verifies a real signed receipt, (3) it catches
tampering, and (4) the CLI exits 0 on a clean artifact and 1 on a tampered one.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from hazardpulse.trust.forecast import _RECEIPT_CORE_FIELDS, _canon_sha
from hazardpulse.trust.scoring import enrich_cells
from hazardpulse.trust.forecast import TrustedForecaster
from hazardpulse.trust.venn_abers import VennAbersCalibrator

REPO = Path(__file__).resolve().parents[1]


def _load_verifier():
    spec = importlib.util.spec_from_file_location("verify_forecast", REPO / "scripts" / "verify_forecast.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _forecaster(signer=None):
    rng = np.random.RandomState(0)
    p = rng.uniform(0, 1, 6000)
    y = (rng.uniform(0, 1, p.size) < p).astype(float)
    return TrustedForecaster.from_calibration_dict(
        {"model_version": "eq_coherence_v1_0", "calibrator": VennAbersCalibrator().fit(p, y).to_dict()},
        signer=signer)


def test_verifier_field_set_and_hash_match_the_spec():
    vf = _load_verifier()
    # the standalone verifier must use the exact same core field set as the signer
    assert set(vf.RECEIPT_CORE_FIELDS) == set(_RECEIPT_CORE_FIELDS)
    claim = {k: None for k in _RECEIPT_CORE_FIELDS}
    claim.update({"spec": "hazardpulse/forecast/v1", "probability": 0.12,
                  "confidence_lo": 0.08, "confidence_hi": 0.18, "ood_flag": False,
                  "abstained": False, "coverage_target": 0.9})
    assert vf.canonical_sha256(claim) == _canon_sha(claim)


def test_verifies_real_signed_receipt_and_catches_tamper():
    crypto = pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    vf = _load_verifier()
    signer = Ed25519PrivateKey.generate()
    pub_hex = signer.public_key().public_bytes_raw().hex()
    tf = _forecaster(signer=signer)
    r = tf.forecast_one(0.4, issued_at="2026-06-25T03:00:00Z").receipt

    ok, detail = vf.verify_receipt(r, pub_hex)
    assert ok and "verified" in detail

    tampered = dict(r)
    tampered["probability"] = (tampered["probability"] or 0.0) + 0.2
    ok2, _ = vf.verify_receipt(tampered, pub_hex)
    assert ok2 is False

    # wrong key fails authenticity
    other = Ed25519PrivateKey.generate().public_key().public_bytes_raw().hex()
    ok3, _ = vf.verify_receipt(r, other)
    assert ok3 is False


def test_cli_on_artifact(tmp_path):
    vf = _load_verifier()
    tf = _forecaster()  # unsigned -> integrity-only path
    cells = enrich_cells([{"probability": 0.6, "lat": 35, "lon": -97},
                          {"probability": 0.1, "lat": 33, "lon": -98}], tf,
                         issued_at="2026-06-25T03:00:00Z")
    artifact = {"forecast_id": "eq_fcst_X", "hazard": "earthquake", "active_cells": cells}
    apath = tmp_path / "art.json"
    apath.write_text(json.dumps(artifact), encoding="utf-8")
    assert vf.main(["--artifact", str(apath)]) == 0     # integrity verifies

    # tamper one cell's receipt -> CLI fails
    artifact["active_cells"][0]["receipt"]["probability"] = 0.99
    apath.write_text(json.dumps(artifact), encoding="utf-8")
    assert vf.main(["--artifact", str(apath)]) == 1


# ------------------------------------------------------------------------------
# A DISCOVERED key is a convenience. A SUPPLIED key is an assertion. The verifier
# stopped distinguishing them and reported [FAIL] on every unsigned artifact,
# purely because a published key sits in dist/. On a zero-trust verifier a
# stranger runs to check our claims, FAIL reads as TAMPERED -- so this is a
# credibility bug, not a usability one.

def test_auto_discovered_key_does_not_fail_an_unsigned_receipt(tmp_path):
    vf = _load_verifier()
    tf = _forecaster()                                   # unsigned by construction
    cells = enrich_cells([{"probability": 0.6, "lat": 35, "lon": -97}], tf,
                         issued_at="2026-06-25T03:00:00Z")
    receipt = cells[0]["receipt"]
    assert not (receipt.get("signature") or {}).get("sig"), "fixture must be unsigned"

    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    key = Ed25519PrivateKey.generate().public_key().public_bytes_raw().hex()
    ok, detail = vf.verify_receipt(receipt, key, key_is_explicit=False)
    assert ok is True, detail
    assert "integrity OK" in detail


def test_an_EXPLICIT_key_still_requires_a_signature(tmp_path):
    """The paired control. If the fallback above were applied unconditionally,
    --pubkey would be silently ignored and an unsigned artifact would pass an
    authenticity check it never underwent."""
    vf = _load_verifier()
    tf = _forecaster()
    cells = enrich_cells([{"probability": 0.6, "lat": 35, "lon": -97}], tf,
                         issued_at="2026-06-25T03:00:00Z")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    key = Ed25519PrivateKey.generate().public_key().public_bytes_raw().hex()
    ok, detail = vf.verify_receipt(cells[0]["receipt"], key, key_is_explicit=True)
    assert ok is False
    assert "no signature present" in detail


def test_tampering_still_fails_even_with_the_unsigned_fallback(tmp_path):
    """The fallback must relax AUTHENTICITY only. Integrity is never optional."""
    vf = _load_verifier()
    tf = _forecaster()
    cells = enrich_cells([{"probability": 0.6, "lat": 35, "lon": -97}], tf,
                         issued_at="2026-06-25T03:00:00Z")
    receipt = dict(cells[0]["receipt"])
    receipt["probability"] = 0.99                        # claim no longer matches its hash
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    key = Ed25519PrivateKey.generate().public_key().public_bytes_raw().hex()
    ok, detail = vf.verify_receipt(receipt, key, key_is_explicit=False)
    assert ok is False
    assert "INTEGRITY FAIL" in detail
