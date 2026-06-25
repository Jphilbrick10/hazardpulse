"""HazardPulse publish gates.

The Truth-Surface spec (docs/HAZARDPULSE_CLASSIC_TRUTH_SURFACE_SPEC.md, §10)
defines a hard publish-gate spine, but today the live pipeline only *stamps*
``decision: pass`` with a single ``confidence_interval_unavailable`` warning on
every forecast — the gates never actually evaluate anything and never block or
degrade a publish.

This package is the real gate engine: each gate evaluates a concrete,
safety-relevant property of a forecast (source freshness, complete provenance,
calibration floor, spatiotemporal sanity, alert-harm guard, replayability) and
returns ``pass`` / ``degrade`` / ``block``. The engine aggregates to the worst
outcome and emits a GateDecisionV1-shaped record the worker and UX render.
"""

from __future__ import annotations

from .engine import (
    BLOCK,
    DEGRADE,
    PASS,
    GateConfig,
    GateContext,
    GateDecision,
    GateEngine,
    GateResult,
    default_config_for,
)

__all__ = [
    "PASS",
    "DEGRADE",
    "BLOCK",
    "GateConfig",
    "GateContext",
    "GateResult",
    "GateDecision",
    "GateEngine",
    "default_config_for",
]
