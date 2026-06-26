"""The /verification calibration scoreboard renders from real calibration data."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from hazardpulse.trust.venn_abers import VennAbersCalibrator

REPO = Path(__file__).resolve().parents[1]


def _bsa():
    spec = importlib.util.spec_from_file_location("bsa", REPO / "scripts" / "build_site_artifacts.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_scoreboard_empty_without_calibration(tmp_path):
    bsa = _bsa()
    bsa.ROOT = tmp_path
    assert bsa._render_calibration_scoreboard() == ""     # nothing until a calibrator exists


def test_scoreboard_renders_with_calibration(tmp_path):
    bsa = _bsa()
    bsa.ROOT = tmp_path
    (tmp_path / "results" / "models").mkdir(parents=True)
    (tmp_path / "results" / "earthquake_prospective").mkdir(parents=True)

    rng = np.random.RandomState(0)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, p.size) < p).astype(float)
    cal = VennAbersCalibrator().fit(p, y)
    rec = {
        "model_version": "eq_coherence_v1_0",
        "calibrator": cal.to_dict(),
        "metrics_before": {"ece": 0.12, "brier_skill_score": -0.1},
        "metrics_after": {"ece": 0.02, "brier_skill_score": 0.34},
        "n_calibration": 5000,
    }
    (tmp_path / "results" / "models" / "earthquake_calibration.json").write_text(json.dumps(rec))
    g = np.round(np.linspace(0.05, 0.95, 10), 6)
    tot = [500] * 10
    pos = [int(round(t * gi)) for t, gi in zip(tot, g)]
    (tmp_path / "results" / "earthquake_prospective" / "calibration_dataset.json").write_text(
        json.dumps({"hazard": "earthquake", "scores": g.tolist(), "pos": pos, "total": tot}))

    html = bsa._render_calibration_scoreboard()
    assert "Calibration scoreboard" in html
    assert "<svg" in html and "stroke-dasharray" in html      # a reliability diagram
    assert "0.120" in html and "0.020" in html                # ECE raw -> calibrated
    assert "Earthquake" in html
