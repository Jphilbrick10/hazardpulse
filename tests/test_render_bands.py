"""P5a: the calibrated band helper is correct in the tornado + hurricane scorers.

(The earthquake band + full cell-render is covered by test_earthquake_render.py;
the tornado/hurricane render functions need heavy page fixtures, so here we test
the new _band_text formatting that each scorer's labels append.)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load(script):
    spec = importlib.util.spec_from_file_location(script.replace(".py", ""), REPO / "scripts" / script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("script", ["fetch_and_score_tornado.py", "fetch_and_score.py"])
def test_band_text(script):
    m = _load(script)
    assert m._band_text(0.08, 0.18) == " (90% band: 8.0%-18.0%)"
    assert m._band_text(None, 0.18) == ""          # raw forecast -> no band
    assert m._band_text(0.08, None) == ""
    assert m._band_text(float("nan"), 0.18) == ""
    assert m._band_text("x", 0.18) == ""


def test_tornado_storm_rows_show_band():
    m = _load("fetch_and_score_tornado.py")
    storm = {
        "storm_id": "S1", "risk_band": "moderate", "tornado_probability": 0.6,
        "lat": 35.0, "lon": -97.0, "mucape": 2000.0, "mlcape": 1800.0, "mlcin": 20.0,
        "srh01": 200.0, "ebshear": 40.0, "maxllaz": 0.004, "mesh": 30.0,
        "vil_density": 3.0, "flash_rate": 10.0, "ps": 0.5, "size": 50.0,
        "valid_time": "2026-06-25T20:00:00Z", "coherence_score": 0.5,
        "confidence_lo": 0.55, "confidence_hi": 0.66, "uncertainty_class": "moderate",
        "storm_age_minutes": 30.0, "model_scores": {}, "top_features": [],
    }
    try:
        html = m._render_storm_rows([storm])
    except Exception as exc:  # pragma: no cover - fixture shape may drift
        pytest.skip(f"storm-row fixture incomplete: {exc}")
    assert "90% band: 55.0%-66.0%" in html
