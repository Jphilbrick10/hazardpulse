"""Per-group (Mondrian) conformal coverage for HazardPulse's natural groups.

Marginal conformal coverage is an *average*: it can sit comfortably at the target
while systematically UNDER-covering a hard subpopulation -- a quiet tectonic
region, a rare EF tier, one ocean basin -- and over-covering an easy one. For a
public-safety instrument that is the dangerous failure: the group that most needs
an honest interval is the one the marginal number hides.

omega_one's ``MondrianConformalPredictor`` gives a SEPARATE coverage guarantee per
group: P(true outcome in set | group = g) >= 1 - alpha for *every* g, each group
getting its own conformal threshold (an unseen group falls back to the pooled one).
``group_coverage`` then audits the realized per-group coverage so the published
number is honest per region / tier / basin, not just on average.

HazardPulse forecasts are binary (event / no-event) per cell, so each cell's
calibrated probability is wrapped as a 2-class ``predict_proba`` and partitioned by
its group label. This module is group-AGNOSTIC -- the caller supplies the grouping
(``geo_region`` below is the concrete one for earthquakes); the engine underneath is
the real, tested omega conformal code, not a reimplementation.
"""

from __future__ import annotations

import numpy as np

from ._vendor_omega.conformal import (
    MondrianConformalPredictor,
    coverage,
    group_coverage,
)


class _PackedProba:
    """A ``predict_proba`` shim over packed ``[prob, group_code]`` rows.

    Column 0 is P(event); column 1 is the group code (read by ``group_fn``, ignored
    here). ``classes_ = [0, 1]`` makes the binary hazard outcome a 2-class problem so
    the Mondrian set predictor applies unchanged.
    """

    classes_ = [0, 1]

    def predict_proba(self, X):
        p = np.clip(np.asarray(X, float)[:, 0], 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


def _group_codes_from(X):
    return np.asarray(X, float)[:, 1]


class GroupConformal:
    """Group-conditional conformal coverage over binary hazard outcomes.

    ``fit(prob, y, group)`` learns a per-group conformal threshold on a held-out
    calibration split; ``predict_set`` returns the conformal membership masks; and
    ``coverage_report`` audits realized per-group coverage against the 1 - alpha
    target. Group labels are preserved (the report keys are the original labels, not
    internal codes); an unseen group at predict time falls back to the pooled
    threshold instead of crashing.
    """

    def __init__(self, *, alpha: float = 0.1, method: str = "lac"):
        self.alpha = float(alpha)
        self.method = method
        self._mcp: MondrianConformalPredictor | None = None
        self._code_of: dict = {}        # label -> float code (frozen at fit)
        self._label_of: dict = {}        # float code -> label
        self._unseen_code: float = -1.0

    # -- label <-> code bookkeeping so codes survive the float X matrix ---------- #
    def _freeze_codes(self, labels) -> None:
        uniq = sorted({_hashable(v) for v in labels})
        self._code_of = {lab: float(i) for i, lab in enumerate(uniq)}
        self._label_of = {float(i): lab for lab, i in self._code_of.items()}
        self._unseen_code = float(len(uniq))

    def _encode(self, labels) -> np.ndarray:
        return np.array(
            [self._code_of.get(_hashable(v), self._unseen_code) for v in labels],
            dtype=float,
        )

    def _pack(self, prob, group) -> np.ndarray:
        prob = np.asarray(prob, float).ravel()
        codes = self._encode(group)
        return np.column_stack([prob, codes])

    # -- public API ------------------------------------------------------------- #
    def fit(self, prob, y, group) -> "GroupConformal":
        group = list(group)
        self._freeze_codes(group)
        X = self._pack(prob, group)
        y = np.asarray(y).astype(int)
        self._mcp = MondrianConformalPredictor(
            _PackedProba(),
            group_fn=_group_codes_from,
            alpha=self.alpha,
            method=self.method,
        )
        self._mcp.fit_calibrate(X, y)
        return self

    def predict_set(self, prob, group) -> np.ndarray:
        if self._mcp is None:
            raise RuntimeError("GroupConformal.fit must be called before predict_set")
        return self._mcp.predict_set(self._pack(prob, group))

    def coverage_report(self, prob, y, group) -> dict:
        """Audit realized coverage marginally and per group against 1 - alpha."""
        group = list(group)
        y = np.asarray(y).astype(int)
        sets = self.predict_set(prob, group)
        codes = self._encode(group)
        target = 1.0 - self.alpha

        per_code = group_coverage(sets, y, codes)
        per_group = {}
        for code, cov in per_code.items():
            label = self._label_of.get(float(code), "__unseen__")
            n = int((codes == code).sum())
            per_group[str(label)] = {
                "coverage": round(float(cov), 4),
                "n": n,
                "meets_target": bool(cov >= target - 1e-9),
            }
        worst = min(per_group.items(), key=lambda kv: kv[1]["coverage"], default=None)
        return {
            "alpha": self.alpha,
            "target": round(target, 4),
            "method": self.method,
            "n": int(len(y)),
            "marginal_coverage": round(float(coverage(sets, y)), 4),
            "per_group": per_group,
            "all_groups_meet_target": all(v["meets_target"] for v in per_group.values()),
            "worst_group": (worst[0] if worst else None),
            "worst_group_coverage": (worst[1]["coverage"] if worst else None),
        }


def _hashable(v):
    """Make group labels usable as dict keys / sortable (arrays -> tuples)."""
    if isinstance(v, np.ndarray):
        return tuple(v.tolist())
    return v


# --------------------------------------------------------------------------- #
# Concrete grouping for earthquakes: coarse tectonic / geographic bands.
# Cells carry lat/lon but no pre-baked region field, so derive a stable region
# label from position. Bands are intentionally coarse (seismotectonic provinces)
# so each has enough calibration mass for its own conformal threshold.
# --------------------------------------------------------------------------- #
def geo_region(lat: float, lon: float) -> str:
    lon = ((float(lon) + 180.0) % 360.0) - 180.0   # normalize to [-180, 180)
    lat = float(lat)
    if 30.0 <= lat <= 50.0 and -130.0 <= lon <= -110.0:
        return "us_west_coast"
    if -10.0 <= lat <= 60.0 and (lon >= 120.0 or lon <= -150.0):
        return "pacific_ring_nw"
    if -45.0 <= lat <= 15.0 and -85.0 <= lon <= -60.0:
        return "andes"
    if 25.0 <= lat <= 45.0 and 25.0 <= lon <= 100.0:
        return "alpide_belt"
    if -40.0 <= lat <= 0.0 and 90.0 <= lon <= 160.0:
        return "indonesia_png"
    if lat >= 50.0:
        return "high_lat"
    if abs(lat) <= 30.0:
        return "tropics_other"
    return "midlat_other"
