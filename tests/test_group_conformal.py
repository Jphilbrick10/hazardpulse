"""Per-group (Mondrian) conformal gives each group its guarantee where marginal hides a failure.

The scenario: an easy majority group (model confident and right) and a hard minority
group (model confident and WRONG). Pooled/marginal conformal looks fine on average yet
systematically under-covers the minority -- exactly the public-safety failure mode. The
Mondrian predictor restores per-group coverage.
"""

from __future__ import annotations

import numpy as np

from hazardpulse.trust._vendor_omega.conformal import ConformalPredictor, group_coverage
from hazardpulse.trust.group_conformal import GroupConformal, _PackedProba, geo_region


def _scenario(seed: int):
    rng = np.random.RandomState(seed)
    nA, nB = 900, 60
    # easy majority A: P(class1) ~ 0.95/0.05 aligned with y -> tiny nonconformity
    yA = (rng.uniform(size=nA) < 0.3).astype(int)
    pA = np.where(yA == 1, 0.95, 0.05)
    # hard minority B: confident but WRONG -> the true class gets only ~0.4
    yB = (rng.uniform(size=nB) < 0.5).astype(int)
    pB = np.where(yB == 1, 0.40, 0.60)
    prob = np.concatenate([pA, pB]).astype(float)
    y = np.concatenate([yA, yB]).astype(int)
    grp = np.array(["A"] * nA + ["B"] * nB)
    return prob, y, grp


def test_mondrian_covers_a_group_marginal_underserves():
    prob_cal, y_cal, grp_cal = _scenario(0)
    prob_te, y_te, grp_te = _scenario(1)

    # --- marginal (pooled) conformal: one threshold for everyone ---------------- #
    Xcal = np.column_stack([prob_cal, np.zeros_like(prob_cal)])
    Xte = np.column_stack([prob_te, np.zeros_like(prob_te)])
    cp = ConformalPredictor(_PackedProba(), alpha=0.1, method="lac").fit_calibrate(Xcal, y_cal)
    marg = group_coverage(cp.predict_set(Xte), y_te, grp_te)
    assert marg["B"] < 0.5, "marginal conformal should under-cover the hard minority"

    # --- Mondrian (group-conditional) conformal --------------------------------- #
    gc = GroupConformal(alpha=0.1, method="lac").fit(prob_cal, y_cal, grp_cal)
    rep = gc.coverage_report(prob_te, y_te, grp_te)

    assert rep["per_group"]["B"]["coverage"] >= 0.83, "Mondrian must restore the minority's guarantee"
    assert rep["per_group"]["A"]["coverage"] >= 0.83
    assert rep["per_group"]["B"]["coverage"] > marg["B"] + 0.3
    assert rep["target"] == 0.9 and rep["method"] == "lac" and rep["n"] == len(y_te)
    assert rep["worst_group_coverage"] >= 0.83


def test_unseen_group_falls_back_to_pooled():
    prob_cal, y_cal, grp_cal = _scenario(0)
    gc = GroupConformal(alpha=0.1).fit(prob_cal, y_cal, grp_cal)
    # a group label never seen at fit time must not crash -> pooled threshold used
    sets = gc.predict_set(np.array([0.7, 0.2]), ["Z_new", "A"])
    assert sets.shape == (2, 2)


def test_coverage_report_flags_a_failing_group():
    prob_cal, y_cal, grp_cal = _scenario(0)
    gc = GroupConformal(alpha=0.1).fit(prob_cal, y_cal, grp_cal)
    rep = gc.coverage_report(*(_scenario(2)[::1]))
    assert set(rep["per_group"]) == {"A", "B"}
    assert isinstance(rep["all_groups_meet_target"], bool)
    assert 0.0 <= rep["marginal_coverage"] <= 1.0


def test_geo_region_is_deterministic_and_partitions():
    assert geo_region(37.7, -122.4) == "us_west_coast"     # San Francisco
    assert geo_region(38.0, 142.0) == "pacific_ring_nw"    # off Tohoku
    assert geo_region(-33.4, -70.6) == "andes"             # Santiago
    assert geo_region(80.0, 10.0) == "high_lat"
    # longitude wrap-around is handled (e.g. 200 -> -160)
    assert geo_region(20.0, 200.0) == geo_region(20.0, -160.0)
