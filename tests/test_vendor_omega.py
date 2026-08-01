"""P1.1 — the vendored omega_one trust subset works and its guarantees hold.

This is a load-bearing test: HazardPulse's whole trust spine (conformal
intervals, OOD/abstention, signed receipts) is built on this subset. The test
proves (a) the vendored bytes match the pinned manifest (no silent drift),
(b) the conformal coverage guarantee actually holds, (c) Mahalanobis OOD
separates in- from out-of-distribution, and (d) a signed decision receipt
verifies and tampering is caught.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hazardpulse.trust import (
    MahalanobisOOD,
    coverage,
    fast_trusted_decision,
    load_ed25519_pubkey,
    verify_trusted_receipt,
)
from hazardpulse.trust._vendor_omega.conformal import calibrate_aps, predict_sets_aps

VENDOR_DIR = Path(__file__).resolve().parents[1] / "src" / "hazardpulse" / "trust" / "_vendor_omega"


# --------------------------------------------------------------------------- #
# A minimal sklearn-style classifier so the test has no sklearn dependency.
# --------------------------------------------------------------------------- #
class _TinyLogReg:
    """Multinomial logistic regression in pure numpy (fit / predict / predict_proba)."""

    def __init__(self, n_iter: int = 300, lr: float = 0.5):
        self.n_iter = n_iter
        self.lr = lr
        self.W = None
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        k = len(self.classes_)
        idx = {c: i for i, c in enumerate(self.classes_)}
        Y = np.zeros((len(y), k))
        for i, yi in enumerate(y):
            Y[i, idx[yi]] = 1.0
        Xb = np.hstack([X, np.ones((len(X), 1))])
        self.W = np.zeros((Xb.shape[1], k))
        for _ in range(self.n_iter):
            P = self._softmax(Xb @ self.W)
            self.W -= self.lr * (Xb.T @ (P - Y)) / len(X)
        return self

    @staticmethod
    def _softmax(Z):
        Z = Z - Z.max(1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(1, keepdims=True)

    def predict_proba(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        Xb = np.hstack([X, np.ones((len(X), 1))])
        return self._softmax(Xb @ self.W)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


def _synth(seed=0, n=6000, d=6, k=3):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    w = rng.randn(d, k)
    P = np.exp(X @ w)
    P /= P.sum(1, keepdims=True)
    y = np.array([rng.choice(k, p=p) for p in P])
    return X, y, P


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_vendor_manifest_no_drift():
    """The vendored bytes match the pinned sha256 manifest (catches silent edits)."""
    manifest = json.loads((VENDOR_DIR / "VENDOR_MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["vendored_from"] == "omega_one"
    assert manifest.get("source_commit")
    for module, expected in manifest["sha256"].items():
        got = hashlib.sha256((VENDOR_DIR / module).read_bytes()).hexdigest()
        assert got == expected, f"vendor drift in {module}: re-run scripts/vendor_omega_trust.py"


def test_trust_public_api_imports():
    import hazardpulse.trust as T

    for name in ("ConformalPredictor", "MahalanobisOOD", "fast_trusted_decision",
                 "verify_trusted_receipt", "ConformalRegressor", "CoherenceGuardian"):
        assert name in T.__all__ and hasattr(T, name)


def test_conformal_coverage_guarantee():
    """Split-conformal (APS) on raw probabilities meets its >= 1-alpha coverage target."""
    _, y, P = _synth(seed=1)
    cut = len(y) // 2
    alpha = 0.1
    qhat = calibrate_aps(P[:cut], y[:cut], alpha=alpha)
    sets = predict_sets_aps(P[cut:], qhat)
    cov = coverage(sets, y[cut:])
    # Marginal guarantee is >= 1-alpha in expectation; allow a small finite-sample slack.
    assert cov >= (1.0 - alpha) - 0.03, f"coverage {cov:.3f} below target {1 - alpha:.2f}"


def test_mahalanobis_ood_separates():
    """Far-shifted inputs score much higher (more OOD) than in-distribution inputs."""
    X, y, _ = _synth(seed=2)
    cut = len(y) // 2
    ood = MahalanobisOOD().fit(X[:cut], y[:cut])
    rng = np.random.RandomState(3)
    s_in = ood.score(X[cut:])
    s_out = ood.score(rng.randn(500, X.shape[1]) * 4.0 + 9.0)
    assert np.median(s_out) > 5.0 * np.median(s_in)


def test_signed_receipt_roundtrip_and_tamper():
    """A signed decision receipt verifies with the public key; tampering is caught."""
    crypto = pytest.importorskip("cryptography")  # declared dependency; skip cleanly if absent
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    X, y, _ = _synth(seed=4)
    n = len(y)
    a, b = n // 3, 2 * n // 3
    Xtr, ytr = X[:a], y[:a]
    Xcal, ycal = X[a:b], y[a:b]
    Xte = X[b:b + 25]

    signer = Ed25519PrivateKey.generate()
    pub_raw = signer.public_key().public_bytes_raw()
    pubkey = load_ed25519_pubkey(pub_raw)

    td = fast_trusted_decision(_TinyLogReg().fit(Xtr, ytr), Xtr, ytr, Xcal, ycal,
                               alpha=0.1, signer=signer)
    decisions = td.decide(Xte)
    assert decisions, "no decisions produced"

    for d in decisions:
        receipt = d["receipt"]
        assert receipt["spec"].startswith("omega-one/trusted/")
        assert "signature" in receipt
        # genuine receipt: integrity + authenticity both verify
        assert verify_trusted_receipt(receipt, pubkey=pubkey) is True

    # tamper: flip a field without re-signing -> integrity (and authenticity) must fail
    tampered = dict(decisions[0]["receipt"])
    tampered["ood_rejected"] = not tampered["ood_rejected"]
    assert verify_trusted_receipt(tampered, pubkey=pubkey) is False
    # even without the pubkey, the self-consistent hash check catches the tamper
    assert verify_trusted_receipt(tampered) is False


def test_pinned_vendor_bytes_contain_no_crlf():
    """A byte-pin is only as good as the bytes surviving checkout.

    Git's line-ending normalisation rewrote all ten vendored modules to CRLF on Windows, so every
    SHA-256 in VENDOR_MANIFEST.json mismatched and the drift test above failed -- on Windows only,
    while Linux CI stayed green on the same commit. The vendored bytes were never wrong; the
    checkout changed them underneath the pin.

    `.gitattributes` marks these paths `-text`; this asserts the bytes actually are what that
    promises, so a drift detector cannot go back to firing on the platform instead of on drift.
    A detector that cries wolf gets written off as pre-existing -- and this one was.
    """
    crlf = bytes([13, 10])                    # built by code point; heredocs eat escapes
    pinned = sorted(VENDOR_DIR.glob("*.py")) + [VENDOR_DIR / "VENDOR_MANIFEST.json"]
    assert len(pinned) > 1, "no pinned artifacts found -- this assertion would be vacuous"
    offenders = [p.name for p in pinned if p.is_file() and crlf in p.read_bytes()]
    assert not offenders, (
        f"CRLF in byte-pinned vendor artifacts {offenders}; their manifest hashes cannot match "
        f"after a checkout on this platform")
