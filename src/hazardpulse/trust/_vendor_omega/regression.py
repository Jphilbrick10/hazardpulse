"""
regression.py - TRUSTED REGRESSION: distribution-free prediction INTERVALS + OOD + signed receipts.

The regression twin of TrustedDecision. SR 11-7 model risk (pricing, PD/LGD, forecasting) is
regression-heavy, so this roughly doubles the addressable regulated surface. The pieces:

  - ConformalRegressor  : split / locally-normalized conformal intervals with a finite-sample
                          coverage GUARANTEE  P(y in [lo, hi]) >= 1 - alpha, for ANY regressor.
  - TrustedRegression   : best regressor (accuracy) + the interval (guarantee) + OOD-reject
                          (Mahalanobis on inputs) + a signed, VERIFIABLE receipt per prediction.
  - verify_regression_receipt : the offline check (integrity + authenticity + smuggle-reject), the
                          regression analogue of verify_trusted_receipt.

numpy + the omega stack; scikit-learn used lazily (the normalized difficulty model).
"""
from __future__ import annotations

import hashlib
import os

import numpy as np

from .trusted import _sha

__all__ = ["ConformalRegressor", "CQRRegressor", "AdaptiveConformalRegressor", "TrustedRegression",
           "verify_regression_receipt", "regression_coverage", "interval_width"]

_REG_FIELDS = ("spec", "model_sha256", "input_sha256", "prediction", "interval_lo", "interval_hi",
               "ood_rejected", "ood_score", "coverage_target", "nonce")
_REG_ALLOWED = set(_REG_FIELDS) | {"receipt_sha256", "signature"}


def _conformal_q(scores, alpha) -> float:
    """Finite-sample-corrected conformal quantile: the ceil((n+1)(1-alpha))-th smallest score."""
    s = np.sort(np.asarray(scores, float)); n = len(s)
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    return float(s[min(k, n) - 1])


class ConformalRegressor:
    """Conformal prediction intervals with a finite-sample coverage guarantee for any fitted `reg`.
    method='split' -> constant-width; 'normalized' -> width adapts to a residual-magnitude model
    (tighter where the model is confident), exchangeability preserved by an internal cal sub-split."""

    def __init__(self, reg, *, alpha=0.1, method="normalized", random_state=0):
        self.reg = reg
        self.alpha = float(alpha)
        self.method = method
        self.random_state = int(random_state)
        self.qhat = None
        self.sigma_ = None

    def fit_calibrate(self, Xcal, ycal):
        X = np.asarray(Xcal, float); y = np.asarray(ycal, float)
        if self.method == "normalized" and len(y) >= 8:
            from sklearn.ensemble import RandomForestRegressor
            rng = np.random.default_rng(self.random_state); idx = rng.permutation(len(y))
            a, b = idx[: len(y) // 2], idx[len(y) // 2:]                 # A fits sigma, B sets qhat
            res_a = np.abs(y[a] - np.asarray(self.reg.predict(X[a]), float))
            self.sigma_ = RandomForestRegressor(n_estimators=80, random_state=self.random_state).fit(X[a], res_a + 1e-6)
            s_b = np.maximum(self.sigma_.predict(X[b]), 1e-6)
            res_b = np.abs(y[b] - np.asarray(self.reg.predict(X[b]), float))
            self.qhat = _conformal_q(res_b / s_b, self.alpha)
        else:
            self.method = "split"
            res = np.abs(y - np.asarray(self.reg.predict(X), float))
            self.qhat = _conformal_q(res, self.alpha)
        return self

    def predict_interval(self, X):
        X = np.asarray(X, float); pred = np.asarray(self.reg.predict(X), float)
        if self.method == "normalized":
            w = self.qhat * np.maximum(self.sigma_.predict(X), 1e-6)
        else:
            w = self.qhat
        return pred - w, pred + w


class CQRRegressor:
    """Conformalized Quantile Regression (Romano et al. 2019) - ADAPTIVE intervals (narrow where the
    model is confident, wide where it is not) with the SAME finite-sample coverage guarantee. Fits a
    LOWER and UPPER quantile regressor, then conformalizes their residuals. Default = gradient-boosted
    quantile regressors; pass `lo_factory`/`hi_factory` for your own. fit(Xtr, ytr) then
    fit_calibrate(Xcal, ycal)."""

    def __init__(self, *, alpha=0.1, lo_factory=None, hi_factory=None, random_state=0):
        self.alpha = float(alpha)
        self.random_state = int(random_state)
        self.lo_factory = lo_factory
        self.hi_factory = hi_factory
        self.lo_ = self.hi_ = None
        self.qhat = None

    def _default(self, q):
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(loss="quantile", alpha=q, n_estimators=200,
                                         max_depth=3, random_state=self.random_state)

    def fit(self, Xtr, ytr):
        Xtr = np.asarray(Xtr, float); ytr = np.asarray(ytr, float)
        self.lo_ = (self.lo_factory or (lambda: self._default(self.alpha / 2)))().fit(Xtr, ytr)
        self.hi_ = (self.hi_factory or (lambda: self._default(1 - self.alpha / 2)))().fit(Xtr, ytr)
        return self

    def fit_calibrate(self, Xcal, ycal):
        X = np.asarray(Xcal, float); y = np.asarray(ycal, float)
        lo, hi = self.lo_.predict(X), self.hi_.predict(X)
        e = np.maximum(lo - y, y - hi)                        # CQR conformity score (signed slack)
        self.qhat = _conformal_q(e, self.alpha)
        return self

    def predict_interval(self, X):
        X = np.asarray(X, float)
        return self.lo_.predict(X) - self.qhat, self.hi_.predict(X) + self.qhat


class AdaptiveConformalRegressor:
    """Adaptive Conformal Inference (Gibbs & Candes 2021) for TIME-SERIES / online data, where the
    EXCHANGEABILITY that split conformal assumes is broken by temporal drift (split conformal then
    UNDER-covers under a regime shift). Maintains a running alpha_t - widen the interval after a miss,
    tighten after a hit - so long-run coverage tracks 1-alpha even under distribution shift. `reg` is
    a fitted point regressor; `stream(X, y)` processes the sequence IN ORDER (no shuffling)."""

    def __init__(self, reg, *, alpha=0.1, gamma=0.05, window=250):
        self.reg = reg
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.window = int(window)

    def stream(self, X, y):
        """Process (X, y) in temporal order. Returns (lo, hi, realized_coverage). Each interval uses
        the adaptive quantile of the RECENT residual window; alpha_t is updated after seeing y_t."""
        from collections import deque
        X = np.asarray(X, float); y = np.asarray(y, float)
        pred = np.asarray(self.reg.predict(X), float)
        resid = deque(maxlen=self.window)
        a_t = self.alpha; lo = np.empty(len(X)); hi = np.empty(len(X)); covered = 0
        for t in range(len(X)):
            q = np.quantile(np.asarray(resid), min(max(1.0 - a_t, 0.0), 1.0)) if resid else 0.0
            lo[t], hi[t] = pred[t] - q, pred[t] + q
            inside = lo[t] <= y[t] <= hi[t]; covered += int(inside)
            a_t = min(max(a_t + self.gamma * (self.alpha - (0.0 if inside else 1.0)), 0.0), 1.0)
            resid.append(abs(y[t] - pred[t]))
        return lo, hi, covered / len(X)


def regression_coverage(y, lo, hi) -> float:
    y, lo, hi = np.asarray(y, float), np.asarray(lo, float), np.asarray(hi, float)
    return float(((y >= lo) & (y <= hi)).mean()) if len(y) else 0.0


def interval_width(lo, hi) -> float:
    return float(np.mean(np.asarray(hi, float) - np.asarray(lo, float)))


class _MahaOOD:
    """Default regression OOD: squared Mahalanobis distance of x to the training inputs (high = OOD)."""
    def fit(self, X):
        X = np.asarray(X, float); self.mu = X.mean(0)
        cov = np.cov(X.T) + 1e-6 * np.eye(X.shape[1]); self.prec = np.linalg.pinv(cov)
        return self

    def score(self, X):
        d = np.asarray(X, float) - self.mu
        return np.einsum("nd,de,ne->n", d, self.prec, d)


class TrustedRegression:
    """Best regressor + conformal interval (guaranteed coverage) + OOD-reject + signed receipt.
    `reg` predicts; pass `ood_scorer` (object with .score / a callable) or rely on the built-in
    Mahalanobis-on-inputs detector. Receipt fields are regression-native (point + interval)."""

    def __init__(self, reg, *, alpha=0.1, method="normalized", ood_scorer=None,
                 ood_reject_quantile=0.99, signer=None, stamp=False, random_state=0):
        self.reg = reg
        self.alpha = float(alpha)
        self.method = method
        self.ood_scorer = ood_scorer
        self.ood_reject_quantile = float(ood_reject_quantile)
        self.signer = signer
        self.stamp = bool(stamp)
        self.random_state = int(random_state)
        self.cp_ = None
        self.ood_threshold_ = None
        self.model_sha256_ = None

    def _ood(self, X):
        s = self.ood_scorer
        if s is None:
            return self._maha.score(X)
        return np.asarray(s.score(X) if hasattr(s, "score") else s(X), float)

    def fit_calibrate(self, Xcal, ycal):
        Xcal = np.asarray(Xcal, float)
        if self.ood_scorer is None:
            self._maha = _MahaOOD().fit(Xcal)
        self.cp_ = ConformalRegressor(self.reg, alpha=self.alpha, method=self.method,
                                      random_state=self.random_state).fit_calibrate(Xcal, ycal)
        self.ood_threshold_ = float(np.quantile(self._ood(Xcal), self.ood_reject_quantile))
        import pickle
        self.model_sha256_ = hashlib.sha256(pickle.dumps(self.reg)).hexdigest()
        return self

    def decide(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        pred = np.asarray(self.reg.predict(X), float)
        lo, hi = self.cp_.predict_interval(X)
        energy = self._ood(X); finite = np.isfinite(energy)
        rejected = (energy > self.ood_threshold_) | ~finite
        out = []
        for i in range(len(X)):
            claim = {
                "spec": "omega-one/trusted-regression/v1",
                "model_sha256": self.model_sha256_,
                "input_sha256": hashlib.sha256(np.ascontiguousarray(X[i], np.float64).tobytes()).hexdigest(),
                "prediction": None if rejected[i] else round(float(pred[i]), 8),
                "interval_lo": None if rejected[i] else round(float(lo[i]), 8),
                "interval_hi": None if rejected[i] else round(float(hi[i]), 8),
                "ood_rejected": bool(rejected[i]),
                "ood_score": round(float(energy[i]), 6) if finite[i] else None,
                "coverage_target": round(1.0 - self.alpha, 4),
                "nonce": os.urandom(16).hex() if self.stamp else None,
            }
            receipt = dict(claim, receipt_sha256=_sha(claim))
            if self.signer is not None:
                receipt["signature"] = {"alg": "ed25519",
                                        "sig": self.signer.sign(receipt["receipt_sha256"].encode()).hex()}
            out.append({"prediction": claim["prediction"], "abstained": bool(rejected[i]),
                        "interval": None if rejected[i] else (claim["interval_lo"], claim["interval_hi"]),
                        "ood_score": claim["ood_score"], "receipt": receipt})
        return out

    def verify(self, receipt, *, pubkey=None) -> bool:
        return verify_regression_receipt(receipt, pubkey=pubkey)


def verify_regression_receipt(receipt, *, pubkey=None) -> bool:
    """Offline verification of a regression receipt: only canonical fields (no smuggling), integrity
    (hash), and - with a pubkey - authenticity (Ed25519). Mirrors verify_trusted_receipt."""
    if not isinstance(receipt, dict) or (set(receipt) - _REG_ALLOWED):
        return False
    claim = {k: receipt.get(k) for k in _REG_FIELDS}
    try:
        if _sha(claim) != receipt.get("receipt_sha256"):
            return False
    except ValueError:
        return False
    if pubkey is not None:
        sig = (receipt.get("signature") or {}).get("sig")
        try:
            pubkey.verify(bytes.fromhex(sig or ""), receipt["receipt_sha256"].encode())
        except Exception:
            return False
    return True
