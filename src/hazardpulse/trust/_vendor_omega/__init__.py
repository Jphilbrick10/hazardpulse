"""VENDORED from omega_one (do NOT edit by hand).

Source: omega_one @ 7becb4069b30a6007d5f57731b2ebeb77111dfb5
Re-vendor with: python scripts/vendor_omega_trust.py
Verify drift with: python scripts/vendor_omega_trust.py --verify

The numpy-only trust subset: split-conformal prediction, Mahalanobis OOD,
selective prediction, the immune-mode guardian, and Ed25519-signed
re-runnable decision receipts. Pure numpy + stdlib (cryptography is imported
lazily, only for signature verification).
"""
from __future__ import annotations

from .conformal import ConformalPredictor, MondrianConformalPredictor, coverage, group_coverage
from .ood_selector import MahalanobisOOD, OODSelector
from .selective import aurc, ece, risk_coverage, selective_report, threshold_for_coverage, threshold_for_risk
from .trusted import TrustedDecision, fast_trusted_decision, load_ed25519_pubkey, verify_trusted_receipt, verify_trusted_receipt_modes
from .regression import ConformalRegressor, CQRRegressor, AdaptiveConformalRegressor, TrustedRegression, regression_coverage, verify_regression_receipt
from .guardian import CoherenceGuardian, GatewayMode
from .batchsign import sign_batch, sign_batch_full, verify_batch_signature, merkle_root, merkle_proof, verify_merkle_proof
from .monitoring import ProductionMonitor, psi

__all__ = [
    "ConformalPredictor",
    "MondrianConformalPredictor",
    "coverage",
    "group_coverage",
    "MahalanobisOOD",
    "OODSelector",
    "aurc",
    "ece",
    "risk_coverage",
    "selective_report",
    "threshold_for_coverage",
    "threshold_for_risk",
    "TrustedDecision",
    "fast_trusted_decision",
    "load_ed25519_pubkey",
    "verify_trusted_receipt",
    "verify_trusted_receipt_modes",
    "ConformalRegressor",
    "CQRRegressor",
    "AdaptiveConformalRegressor",
    "TrustedRegression",
    "regression_coverage",
    "verify_regression_receipt",
    "CoherenceGuardian",
    "GatewayMode",
    "sign_batch",
    "sign_batch_full",
    "verify_batch_signature",
    "merkle_root",
    "merkle_proof",
    "verify_merkle_proof",
    "ProductionMonitor",
    "psi",
]
