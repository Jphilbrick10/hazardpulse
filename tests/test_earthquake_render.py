"""P5a: the calibrated uncertainty band renders into the earthquake page HTML."""

from __future__ import annotations

import importlib.util
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _load_scorer():
    spec = importlib.util.spec_from_file_location(
        "fetch_and_score_earthquake", REPO / "scripts" / "fetch_and_score_earthquake.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_band_text_formatting():
    m = _load_scorer()
    assert m._band_text(0.08, 0.18) == " (90% band: 8.0%-18.0%)"
    assert m._band_text(None, 0.18) == ""          # raw forecast -> no band
    assert m._band_text(float("nan"), 0.18) == ""


def _cell(**over):
    c = {
        "row": 10, "col": 20, "lat": 38.0, "lon": -122.0, "n_events": 14,
        "max_mag": 4.6, "probability": 0.12, "risk_band": "elevated",
        "scoring_tier": "tier1_ml", "b_value": 0.82, "b_trend": -0.01,
        "ell_km": 120.0, "ell_trend": 2.0, "rate_acceleration": 1.6,
        "delta_aic_iet": -3.0, "S_over_Gamma": 1.2, "days_to_criticality": 40.0,
        "conditions_met": 3,
        "singularity_detail": {"ell_elevated": True, "b_depressed": True,
                               "iet_lorentzian": False, "rate_accelerating": True,
                               "loading_exceeds_healing": True},
        "tau_local": 0.3, "grad_tau_local": 0.1, "depth_trend": 0.0,
        "spatial_concentration": 50.0, "model_version": "eq_coherence_v1_0",
        "confidence_lo": 0.08, "confidence_hi": 0.18, "uncertainty_class": "moderate",
        "abstained": False,
    }
    c.update(over)
    return c


def test_calibrated_band_appears_in_cell_rows():
    m = _load_scorer()
    html = m._render_cell_rows([_cell()])
    assert "90% band: 8.0%-18.0%" in html


def test_raw_cell_has_no_band():
    m = _load_scorer()
    html = m._render_cell_rows([_cell(confidence_lo=None, confidence_hi=None)])
    assert "90% band" not in html
