"""Tests for the calibration-fitting pipeline (histogram fit + fit_calibration.py).

Proves the counts-based helpers match the row-based ones, that fitting from a
pooled histogram equals fitting from expanded rows, and that the end-to-end
fit script turns a miscalibrated histogram into a calibrated one (ECE down,
Brier-skill up).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from hazardpulse.trust.calibration import (
    brier_score,
    ece_from_counts,
    expected_calibration_error,
)
from hazardpulse.trust.venn_abers import VennAbersCalibrator

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_fit_calibration():
    spec = importlib.util.spec_from_file_location(
        "fit_calibration", REPO_ROOT / "scripts" / "fit_calibration.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _overconfident(t):
    eps = 1e-6
    logit = np.log(np.clip(t, eps, 1 - eps) / np.clip(1 - t, eps, 1 - eps))
    return 1.0 / (1.0 + np.exp(-2.5 * logit))


def test_ece_from_counts_matches_expanded():
    rng = np.random.RandomState(0)
    p = rng.uniform(0, 1, 5000)
    y = (rng.uniform(0, 1, p.size) < p).astype(float)
    # group into a histogram by rounding scores
    keys = np.round(p, 2)
    reps = np.unique(keys)
    pos = np.array([y[keys == k].sum() for k in reps])
    tot = np.array([(keys == k).sum() for k in reps])
    direct = expected_calibration_error(keys, y, n_bins=10)   # use rounded scores both ways
    counts = ece_from_counts(reps, pos, tot, n_bins=10)
    assert abs(direct - counts) < 1e-9


def test_fit_grouped_equals_fit_on_expanded():
    rng = np.random.RandomState(1)
    p = rng.uniform(0, 1, 8000)
    y = (rng.uniform(0, 1, p.size) < p).astype(float)
    keys = np.round(p, 2)
    reps = np.unique(keys)
    pos = np.array([y[keys == k].sum() for k in reps])
    tot = np.array([(keys == k).sum() for k in reps])

    a = VennAbersCalibrator(max_groups=512).fit(keys, y)
    b = VennAbersCalibrator(max_groups=512).fit_grouped(reps, pos, tot)
    grid = np.linspace(0.02, 0.98, 40)
    pa, _, _ = a.predict(grid)
    pb, _, _ = b.predict(grid)
    np.testing.assert_allclose(pa, pb, atol=1e-9)


def test_roundtrip_to_dict_from_dict():
    rng = np.random.RandomState(2)
    p = rng.uniform(0, 1, 4000)
    y = (rng.uniform(0, 1, p.size) < p).astype(float)
    cal = VennAbersCalibrator().fit(p, y)
    restored = VennAbersCalibrator.from_dict(cal.to_dict())
    grid = np.linspace(0.05, 0.95, 30)
    np.testing.assert_allclose(cal.predict(grid)[0], restored.predict(grid)[0], atol=1e-7)


def test_fit_calibration_improves_metrics(tmp_path):
    fc = _load_fit_calibration()
    # a miscalibrated (overconfident) histogram: rep score = overconfident(true_rate)
    g = np.linspace(0.02, 0.98, 40)
    total = np.full(g.size, 3000.0)
    pos = np.round(total * g)
    scores = _overconfident(g)
    ds = tmp_path / "calibration_dataset.json"
    ds.write_text(json.dumps({
        "hazard": "tornado",
        "scores": scores.tolist(), "pos": pos.tolist(), "total": total.tolist(),
    }), encoding="utf-8")
    out = tmp_path / "tornado_calibration.json"
    payload = fc.fit_one(ds, out, model_version="tornado_storm_v1_0", min_calibration=200)

    assert not payload["inflated"]
    b, a = payload["metrics_before"], payload["metrics_after"]
    assert a["ece"] < b["ece"] * 0.5, f"calibration should cut ECE: {b['ece']} -> {a['ece']}"
    assert a["brier_skill_score"] > b["brier_skill_score"]
    assert a["brier_skill_score"] > 0.0     # calibrated forecast beats climatology
    assert out.is_file()
    # the persisted calibrator reloads and predicts
    reloaded = VennAbersCalibrator.from_dict(json.loads(out.read_text())["calibrator"])
    assert reloaded.fitted and not reloaded.inflated
