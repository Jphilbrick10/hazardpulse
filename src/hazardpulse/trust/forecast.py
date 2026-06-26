"""TrustedForecaster — one wrapper that makes a hazard forecast trustworthy.

Every HazardPulse scorer (earthquake / hurricane / tornado) produces a raw
probability of an event per cell or storm. On its own that number is
miscalibrated (live Brier skill < 0), carries no uncertainty band, never
abstains, and isn't independently verifiable. This wrapper fixes all four in one
place — used by all three scorers, no per-hazard copies:

  1. CALIBRATION + INTERVAL — a fitted Venn-Abers calibrator maps the raw score to
     a calibrated probability with an honest [confidence_lo, confidence_hi] band
     (this is what finally populates HazardForecastV1's interval and removes the
     universal ``confidence_interval_unavailable`` gate warning).
  2. OUT-OF-DISTRIBUTION — a model-free Mahalanobis detector on the feature vector
     flags inputs unlike anything the model was calibrated on.
  3. ABSTENTION — when the input is OOD or its source data is unhealthy, the
     forecast defers ("insufficient signal — see official sources") instead of
     emitting a confident wrong number. The single most important public-safety
     behavior.
  4. SIGNED, RE-RUNNABLE RECEIPT — an Ed25519-signed, tamper-evident receipt binds
     the exact input + model fingerprint + decision, so a third party can verify
     the forecast without trusting us.

Fit once per (model, calibration window); call ``forecast`` each scoring cycle.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Sequence

import numpy as np

from ._vendor_omega.guardian import GatewayMode
from ._vendor_omega.ood_selector import MahalanobisOOD
from .venn_abers import VennAbersCalibrator

__all__ = [
    "TrustedForecastResult",
    "TrustedForecaster",
    "sign_forecast_receipt",
    "verify_forecast_receipt",
]

RECEIPT_SPEC = "hazardpulse/forecast/v1"
# Canonical fields that are hashed into receipt_sha256 (order-independent: sorted).
_RECEIPT_CORE_FIELDS = (
    "spec", "model_version", "model_sha256", "input_sha256", "issued_at",
    "raw_probability", "probability", "confidence_lo", "confidence_hi",
    "uncertainty_class", "ood_score", "ood_flag", "abstained", "abstain_reason",
    "gateway_mode", "coverage_target",
)


def _canon_sha(obj) -> str:
    """SHA-256 of a canonical JSON encoding (rejects NaN/Inf -> tamper-safe)."""
    encoded = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _round(x, n=6):
    if x is None:
        return None
    xf = float(x)
    return None if not np.isfinite(xf) else round(xf, n)


def sign_forecast_receipt(core: dict, signer) -> dict:
    """Stamp a forecast-core dict with its receipt hash and (optional) Ed25519 sig.

    ``signer`` is any object with ``.sign(bytes) -> bytes`` (e.g. a cryptography
    Ed25519PrivateKey), or None for an unsigned (integrity-only) receipt.
    """
    claim = {k: core.get(k) for k in _RECEIPT_CORE_FIELDS}
    receipt = dict(claim)
    receipt["receipt_sha256"] = _canon_sha(claim)
    if signer is not None:
        sig = signer.sign(receipt["receipt_sha256"].encode())
        receipt["signature"] = {"alg": "ed25519", "sig": sig.hex()}
    return receipt


def verify_forecast_receipt(receipt: dict, *, pubkey=None, require_signature: bool = False) -> bool:
    """A receipt verifies iff its core fields hash to receipt_sha256 (integrity)
    and, if ``pubkey`` is given, the Ed25519 signature checks out (authenticity).
    The model is NOT needed to verify — a third party can audit standalone.
    """
    if not isinstance(receipt, dict):
        return False
    claim = {k: receipt.get(k) for k in _RECEIPT_CORE_FIELDS}
    try:
        if _canon_sha(claim) != receipt.get("receipt_sha256"):
            return False
    except (ValueError, TypeError):
        return False  # NaN/Inf or unencodable -> reject
    sig = (receipt.get("signature") or {}).get("sig")
    if (pubkey is not None or require_signature) and not sig:
        return False
    if pubkey is not None and sig:
        try:
            pubkey.verify(bytes.fromhex(sig), receipt["receipt_sha256"].encode())
        except Exception:
            return False
    return True


@dataclass(frozen=True)
class TrustedForecastResult:
    raw_probability: float
    probability: float | None         # calibrated; None when abstained
    confidence_lo: float | None
    confidence_hi: float | None
    uncertainty_class: str            # tight | moderate | wide | abstain
    ood_score: float | None
    ood_flag: bool
    abstained: bool
    abstain_reason: str | None
    gateway_mode: str
    receipt: dict = field(default_factory=dict)

    def as_contract(self, **extra) -> dict:
        """Project onto the public HazardForecastV1-style fields."""
        out = {
            "probability": self.probability,
            "confidence_lo": self.confidence_lo,
            "confidence_hi": self.confidence_hi,
            "uncertainty_class": self.uncertainty_class,
            "abstained": self.abstained,
            "abstain_reason": self.abstain_reason,
            "gateway_mode": self.gateway_mode,
            "ood_flag": self.ood_flag,
            "raw_probability": self.raw_probability,
            "receipt_sha256": self.receipt.get("receipt_sha256"),
        }
        out.update(extra)
        return out


def _uncertainty_class(lo: float, hi: float) -> str:
    width = hi - lo
    if width <= 0.10:
        return "tight"
    if width <= 0.25:
        return "moderate"
    return "wide"


class TrustedForecaster:
    """Calibrate + OOD-gate + sign a hazard head's raw probabilities.

    Parameters
    ----------
    model_version : str
        Bound into every receipt (provenance).
    alpha : float
        Conformal/coverage miss-rate target (coverage_target = 1 - alpha),
        recorded for the contract; the Venn-Abers band is the operative interval.
    ood_reject_quantile : float
        A feature scores OOD if its Mahalanobis distance exceeds this quantile of
        the calibration features' distances. None disables OOD abstention.
    signer : optional
        Ed25519 private key (``.sign(bytes)``) for receipt authenticity.
    """

    def __init__(self, *, model_version: str, alpha: float = 0.1,
                 ood_reject_quantile: float | None = 0.99,
                 min_calibration: int = 50, max_groups: int = 512, signer=None):
        self.model_version = str(model_version)
        self.alpha = float(alpha)
        self.ood_reject_quantile = ood_reject_quantile
        self.signer = signer
        self.calibrator = VennAbersCalibrator(min_calibration=min_calibration,
                                              max_groups=max_groups)
        self.ood: MahalanobisOOD | None = None
        self.ood_threshold_: float | None = None
        self.model_sha256_: str | None = None
        self.fitted = False

    @classmethod
    def from_calibration_dict(cls, d: dict, *, signer=None, alpha: float = 0.1) -> "TrustedForecaster":
        """Build a forecaster from a persisted calibration record (results/models/
        <hazard>_calibration.json). Loads the fitted probability calibrator; OOD is
        left unset (no per-cell feature manifold persisted) so abstention is driven
        by data-health until OOD is fit at scoring time."""
        from .venn_abers import VennAbersCalibrator
        tf = cls(model_version=d.get("model_version", "unknown"), alpha=alpha,
                 ood_reject_quantile=None, signer=signer)
        tf.calibrator = VennAbersCalibrator.from_dict(d["calibrator"])
        tf.ood = None
        tf.model_sha256_ = tf._fingerprint()
        tf.fitted = True
        return tf

    # -- fitting ----------------------------------------------------------- #
    def fit(self, cal_scores, cal_outcomes, cal_features=None) -> "TrustedForecaster":
        """Fit the calibrator on (raw_score, outcome) and, if given, the OOD
        detector on the calibration feature vectors."""
        self.calibrator.fit(cal_scores, cal_outcomes)
        if cal_features is not None and self.ood_reject_quantile is not None:
            X = np.atleast_2d(np.asarray(cal_features, dtype=np.float64))
            X = X[np.isfinite(X).all(axis=1)]
            if len(X) >= 10:
                self.ood = MahalanobisOOD().fit(X, np.zeros(len(X)))
                self.ood_threshold_ = float(np.quantile(self.ood.score(X),
                                                        self.ood_reject_quantile))
        self.model_sha256_ = self._fingerprint()
        self.fitted = True
        return self

    def _fingerprint(self) -> str:
        cal = self.calibrator
        cal_fp = {
            "inflated": bool(cal.inflated),
            "reps": None if cal._reps is None else np.round(cal._reps, 6).tolist(),
            "p0": None if cal._P0 is None else np.round(cal._P0, 6).tolist(),
            "p1": None if cal._P1 is None else np.round(cal._P1, 6).tolist(),
        }
        ood_fp = None
        if self.ood is not None:
            ood_fp = {"threshold": _round(self.ood_threshold_),
                      "means": np.round(np.asarray(self.ood.means_), 6).tolist()
                      if getattr(self.ood, "means_", None) is not None else None}
        return _canon_sha({"model_version": self.model_version, "alpha": self.alpha,
                           "calibrator": cal_fp, "ood": ood_fp})

    # -- prediction -------------------------------------------------------- #
    def forecast_one(self, raw_prob: float, feature_vec: Sequence[float] | None = None, *,
                     data_health_ok: bool = True, issued_at: str | None = None,
                     signer=None) -> TrustedForecastResult:
        if not self.fitted:
            raise RuntimeError("TrustedForecaster not fitted")
        raw = float(np.clip(raw_prob, 0.0, 1.0))
        p, lo, hi = (float(v[0]) for v in self.calibrator.predict([raw]))

        ood_score = None
        ood_flag = False
        if self.ood is not None and feature_vec is not None:
            fv = np.asarray(feature_vec, dtype=np.float64).reshape(1, -1)
            if np.isfinite(fv).all():
                ood_score = float(self.ood.score(fv)[0])
                ood_flag = bool(ood_score > self.ood_threshold_)

        abstained = False
        abstain_reason: str | None = None
        if not data_health_ok:
            abstained, abstain_reason = True, "data_unhealthy"
        elif ood_flag:
            abstained, abstain_reason = True, "out_of_distribution"

        if abstained:
            prob_out: float | None = None
            lo_out = hi_out = None
            uclass = "abstain"
            mode = GatewayMode.DEGRADED.value
        else:
            prob_out, lo_out, hi_out = p, lo, hi
            if self.calibrator.inflated:
                uclass = "wide"
                mode = GatewayMode.CAUTIOUS.value
            else:
                uclass = _uncertainty_class(lo, hi)
                mode = GatewayMode.CAUTIOUS.value if uclass == "wide" else GatewayMode.NORMAL.value

        input_payload = (np.asarray(feature_vec, dtype=np.float64).tobytes()
                         if feature_vec is not None
                         else np.asarray([raw], dtype=np.float64).tobytes())
        core = {
            "spec": RECEIPT_SPEC,
            "model_version": self.model_version,
            "model_sha256": self.model_sha256_,
            "input_sha256": hashlib.sha256(input_payload).hexdigest(),
            "issued_at": issued_at,
            "raw_probability": _round(raw),
            "probability": _round(prob_out),
            "confidence_lo": _round(lo_out),
            "confidence_hi": _round(hi_out),
            "uncertainty_class": uclass,
            "ood_score": _round(ood_score),
            "ood_flag": ood_flag,
            "abstained": abstained,
            "abstain_reason": abstain_reason,
            "gateway_mode": mode,
            "coverage_target": _round(1.0 - self.alpha, 4),
        }
        receipt = sign_forecast_receipt(core, signer if signer is not None else self.signer)
        return TrustedForecastResult(
            raw_probability=raw, probability=prob_out,
            confidence_lo=lo_out, confidence_hi=hi_out, uncertainty_class=uclass,
            ood_score=ood_score, ood_flag=ood_flag, abstained=abstained,
            abstain_reason=abstain_reason, gateway_mode=mode, receipt=receipt,
        )

    def forecast(self, raw_probs, feature_matrix=None, *, data_health=None,
                 issued_at: str | None = None) -> list[TrustedForecastResult]:
        raw_probs = list(raw_probs)
        n = len(raw_probs)
        feats = [None] * n if feature_matrix is None else list(feature_matrix)
        health = [True] * n if data_health is None else list(data_health)
        return [
            self.forecast_one(raw_probs[i], feats[i],
                              data_health_ok=bool(health[i]), issued_at=issued_at)
            for i in range(n)
        ]
