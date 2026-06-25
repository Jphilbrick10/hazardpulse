"""HazardPulse trust layer.

HazardPulse's measured failure is *calibration and trust*, not raw ranking:
live forecasts have a negative Brier skill score (worse-calibrated than
climatology), carry no uncertainty bands, never abstain, and the "truth surface"
is only a SHA-256 hash chain. This package is the fix.

It composes the vendored, measured omega_one trust primitives
(``_vendor_omega``: split-conformal prediction, Mahalanobis OOD, selective
prediction / abstention, the immune-mode guardian, and Ed25519-signed
re-runnable decision receipts) into one HazardPulse-facing surface used by all
three scorers.

Re-vendor / verify the omega_one subset:
    python scripts/vendor_omega_trust.py
    python scripts/vendor_omega_trust.py --verify
"""

from __future__ import annotations

from ._vendor_omega import (
    AdaptiveConformalRegressor,
    CoherenceGuardian,
    ConformalPredictor,
    ConformalRegressor,
    CQRRegressor,
    GatewayMode,
    MahalanobisOOD,
    MondrianConformalPredictor,
    OODSelector,
    TrustedDecision,
    TrustedRegression,
    aurc,
    coverage,
    ece,
    fast_trusted_decision,
    group_coverage,
    load_ed25519_pubkey,
    merkle_proof,
    merkle_root,
    regression_coverage,
    risk_coverage,
    selective_report,
    sign_batch,
    sign_batch_full,
    threshold_for_coverage,
    threshold_for_risk,
    verify_batch_signature,
    verify_merkle_proof,
    verify_regression_receipt,
    verify_trusted_receipt,
    verify_trusted_receipt_modes,
)
from .calibration import (
    brier_decomposition,
    brier_score,
    brier_skill_score,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_curve,
)
from .forecast import (
    TrustedForecaster,
    TrustedForecastResult,
    sign_forecast_receipt,
    verify_forecast_receipt,
)
from .venn_abers import VennAbersCalibrator, VennAbersResult

__all__ = [
    # HazardPulse trust surface (the wrapper all scorers use)
    "TrustedForecaster",
    "TrustedForecastResult",
    "sign_forecast_receipt",
    "verify_forecast_receipt",
    # binary probability calibration + measurement
    "VennAbersCalibrator",
    "VennAbersResult",
    "expected_calibration_error",
    "maximum_calibration_error",
    "reliability_curve",
    "brier_score",
    "brier_skill_score",
    "brier_decomposition",
    # conformal prediction (real uncertainty bands)
    "ConformalPredictor",
    "MondrianConformalPredictor",
    "coverage",
    "group_coverage",
    "ConformalRegressor",
    "CQRRegressor",
    "AdaptiveConformalRegressor",
    "TrustedRegression",
    "regression_coverage",
    "verify_regression_receipt",
    # out-of-distribution / novelty
    "MahalanobisOOD",
    "OODSelector",
    # selective prediction / abstention quality
    "aurc",
    "ece",
    "risk_coverage",
    "selective_report",
    "threshold_for_coverage",
    "threshold_for_risk",
    # immune-mode guardian
    "CoherenceGuardian",
    "GatewayMode",
    # signed, re-runnable decision receipts
    "TrustedDecision",
    "fast_trusted_decision",
    "load_ed25519_pubkey",
    "verify_trusted_receipt",
    "verify_trusted_receipt_modes",
    "sign_batch",
    "sign_batch_full",
    "verify_batch_signature",
    "merkle_root",
    "merkle_proof",
    "verify_merkle_proof",
]
