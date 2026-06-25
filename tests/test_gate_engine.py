"""Tests for the publish-gate engine (hazardpulse.gates).

Proves the gates actually evaluate (the old pipeline hardcoded ``pass``): a clean
forecast passes; stale data, missing provenance, miscalibration, invalid
geometry, alarmist-on-thin-evidence, missing replay, and reserved terminology
each degrade or block; and the engine aggregates to the worst outcome.
"""

from __future__ import annotations

from dataclasses import replace

from hazardpulse.gates import BLOCK, DEGRADE, PASS, GateContext, GateEngine

ENGINE = GateEngine()


def _good() -> GateContext:
    return GateContext(
        hazard="earthquake",
        forecast_id="eq_fcst_20260625_0300",
        model_version="eq_coherence_v1_0",
        model_sha256="a" * 64,
        input_sha256="b" * 64,
        receipt_sha256="c" * 64,
        replay_artifact="/data/replay/eq_fcst_20260625_0300.json",
        probability=0.12,
        confidence_lo=0.08,
        confidence_hi=0.18,
        abstained=False,
        ood_flag=False,
        uncertainty_class="moderate",
        lat=38.0,
        lon=-122.0,
        cell_size_deg=2.0,
        data_age_seconds=3600.0,
        ece=0.04,
        brier_skill_score=0.2,
        calibration_known=True,
        risk_label="Elevated",
    )


def _decide(ctx):
    return ENGINE.evaluate(ctx).decision


def test_clean_forecast_passes():
    d = ENGINE.evaluate(_good())
    assert d.decision == PASS
    assert not d.blocking_reasons and not d.warnings
    assert d.gate_decision_id == "gdec_eq_fcst_20260625_0300"
    # confidence interval present -> the old universal warning is gone
    assert all("confidence_interval_unavailable" not in (w or "") for w in d.warnings)


def test_stale_source_blocks_aging_degrades():
    assert _decide(replace(_good(), data_age_seconds=40 * 3600)) == BLOCK
    assert _decide(replace(_good(), data_age_seconds=20 * 3600)) == DEGRADE
    assert _decide(replace(_good(), data_age_seconds=None)) == DEGRADE


def test_missing_provenance_blocks():
    assert _decide(replace(_good(), receipt_sha256=None)) == BLOCK
    assert _decide(replace(_good(), model_sha256=None)) == BLOCK


def test_calibration_floor():
    assert _decide(replace(_good(), ece=0.30)) == BLOCK            # ECE above block
    assert _decide(replace(_good(), ece=0.15)) == DEGRADE          # ECE elevated
    assert _decide(replace(_good(), brier_skill_score=-0.71)) == BLOCK   # the live tornado number
    assert _decide(replace(_good(), brier_skill_score=-0.05)) == DEGRADE
    assert _decide(replace(_good(), calibration_known=False)) == DEGRADE


def test_spatiotemporal_sanity_blocks_bad_geometry():
    assert _decide(replace(_good(), lat=200.0)) == BLOCK
    assert _decide(replace(_good(), probability=1.4)) == BLOCK
    assert _decide(replace(_good(), confidence_lo=0.4, confidence_hi=0.1)) == BLOCK
    # finer-than-floor cell -> degrade, not block
    assert _decide(replace(_good(), cell_size_deg=0.5)) == DEGRADE


def test_alert_harm_guard():
    # high probability is fine when in-distribution and not wide
    assert _decide(replace(_good(), probability=0.7)) == PASS
    # high probability on OOD input -> degrade (cap the alarm)
    assert _decide(replace(_good(), probability=0.7, ood_flag=True)) == DEGRADE
    assert _decide(replace(_good(), probability=0.7, uncertainty_class="wide")) == DEGRADE


def test_missing_replay_blocks():
    assert _decide(replace(_good(), replay_artifact=None)) == BLOCK


def test_reserved_terminology_blocks():
    assert _decide(replace(_good(), risk_label="Tornado Warning")) == BLOCK
    assert _decide(replace(_good(), risk_label="imminent danger")) == BLOCK


def test_clean_abstained_forecast_passes():
    ctx = replace(_good(), abstained=True, probability=None,
                  confidence_lo=None, confidence_hi=None, uncertainty_class="abstain")
    assert ENGINE.evaluate(ctx).decision == PASS


def test_engine_aggregates_to_worst_and_lists_reasons():
    ctx = replace(_good(), data_age_seconds=20 * 3600, replay_artifact=None)  # degrade + block
    d = ENGINE.evaluate(ctx)
    assert d.decision == BLOCK
    assert any("replay" in r for r in d.blocking_reasons)
    assert any("aging" in w for w in d.warnings)
    payload = d.as_dict()
    assert payload["decision"] == "block" and payload["gates"]
