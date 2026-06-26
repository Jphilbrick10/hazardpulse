"""Tests for the reliability curve-from-counts + the SVG scoreboard renderer."""

from __future__ import annotations

import numpy as np

from hazardpulse.trust.calibration import (
    ReliabilityBin,
    ReliabilityCurve,
    reliability_curve,
    reliability_curve_from_counts,
)
from hazardpulse.trust.viz import reliability_diagram_svg


def test_curve_from_counts_matches_expanded():
    scores = np.array([0.05, 0.25, 0.55, 0.85])
    pos = np.array([10, 40, 120, 360])
    total = np.array([100, 100, 200, 400])
    c = reliability_curve_from_counts(scores, pos, total, n_bins=10)

    es, ey = [], []
    for s, p, t in zip(scores, pos, total):
        es += [s] * int(t)
        ey += [1.0] * int(p) + [0.0] * int(t - p)
    c2 = reliability_curve(np.array(es), np.array(ey), n_bins=10)

    assert c.n == c2.n
    for b1, b2 in zip(c.bins, c2.bins):
        assert b1.count == b2.count
        if b1.count:
            assert abs(b1.mean_predicted - b2.mean_predicted) < 1e-9
            assert abs(b1.observed_freq - b2.observed_freq) < 1e-9


def test_svg_renders_with_points():
    g = np.linspace(0.05, 0.95, 10)
    total = np.full(10, 1000.0)
    pos = np.round(total * g)
    curve = reliability_curve_from_counts(g, pos, total, n_bins=10)
    svg = reliability_diagram_svg(curve, title="Earthquake calibration")

    assert svg.startswith("<svg") and svg.rstrip().endswith("</svg>")
    assert "stroke-dasharray" in svg                # perfect-calibration diagonal
    assert svg.count("<circle") >= 8                # one marker per populated bin
    assert "Predicted probability" in svg and "Observed frequency" in svg
    assert "Earthquake calibration" in svg


def test_svg_empty_curve_is_still_valid():
    empty = ReliabilityCurve(
        bins=[ReliabilityBin(0.0, 0.1, 0, float("nan"), float("nan"))],
        n=0, base_rate=float("nan"))
    svg = reliability_diagram_svg(empty)
    assert svg.startswith("<svg") and "<circle" not in svg     # no points, still a frame
    assert "stroke-dasharray" in svg
