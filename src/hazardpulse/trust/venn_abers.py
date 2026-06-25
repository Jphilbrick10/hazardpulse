"""Venn-Abers calibration: a calibrated binary probability WITH a validity interval.

HazardPulse emits single-event probabilities (P(M6+ in 30d), P(RI in 24h),
P(tornado in 24h)) and they are badly miscalibrated (live Brier skill < 0). A
raw model score needs to be turned into a *calibrated* probability, and the
public/contract surface needs an honest band (``confidence_lo``/``confidence_hi``)
around it — not the always-``None`` that produces the universal
``confidence_interval_unavailable`` gate warning.

The Inductive Venn-Abers Predictor (IVAP; Vovk, Petej & Fedorova 2015) is the
principled tool: from a held-out calibration set of (score, outcome) pairs it
produces, for a new score s, a probability interval [p0, p1] that is
*automatically calibrated* (a Venn multiprobability is calibrated by
construction). The log-loss-optimal point estimate is p = p1 / (1 - p0 + p1),
and [p0, p1] is the honest band on that probability.

Implementation (fast + exact-on-the-binned-calibration):
  - the calibration set is reduced to <= ``max_groups`` quantile groups (exact
    when there are already few distinct scores);
  - the two calibrated step-functions p0(.) and p1(.) — the IVAP value when the
    hypothetical label is 0 / 1 — are PRECOMPUTED once at every insertion gap
    between groups (each gap's value is constant in s, since it depends only on
    the inserted point's rank, not its exact value);
  - prediction is then a vectorized ``searchsorted`` lookup: O(m log G), so
    scoring a whole forecast cycle is milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["VennAbersCalibrator", "VennAbersResult", "pav_isotonic"]


def pav_isotonic(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Weighted pool-adjacent-violators: nearest non-decreasing fit (L2).

    ``values`` must already be ordered by the covariate (score) ascending.
    Returns the isotonic (non-decreasing) fitted values, same length.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    n = values.size
    if n == 0:
        return values.copy()
    block_mean: list[float] = []
    block_w: list[float] = []
    block_n: list[int] = []
    for i in range(n):
        cur_mean = float(values[i])
        cur_w = float(weights[i])
        cur_n = 1
        while block_mean and block_mean[-1] > cur_mean:
            pm = block_mean.pop()
            pw = block_w.pop()
            pn = block_n.pop()
            cur_mean = (pm * pw + cur_mean * cur_w) / (pw + cur_w)
            cur_w = pw + cur_w
            cur_n = pn + cur_n
        block_mean.append(cur_mean)
        block_w.append(cur_w)
        block_n.append(cur_n)
    fitted = np.empty(n, dtype=np.float64)
    idx = 0
    for m, c in zip(block_mean, block_n):
        fitted[idx:idx + c] = m
        idx += c
    return fitted


@dataclass(frozen=True)
class VennAbersResult:
    probability: float   # calibrated point estimate p1/(1-p0+p1)
    lower: float         # p0
    upper: float         # p1


class VennAbersCalibrator:
    """Inductive Venn-Abers calibrator for binary outcomes.

    Fit on a HELD-OUT calibration split of (raw_score, outcome in {0,1}); predict
    a calibrated probability and a [p0, p1] validity interval for new scores.
    """

    def __init__(self, *, min_calibration: int = 50, max_groups: int = 512):
        self.min_calibration = int(min_calibration)
        self.max_groups = int(max_groups)
        self._reps: np.ndarray | None = None   # group representative scores (ascending)
        self._P0: np.ndarray | None = None     # precomputed p0 per insertion gap (len G+1)
        self._P1: np.ndarray | None = None     # precomputed p1 per insertion gap (len G+1)
        self.n_calibration: int = 0
        self.fitted: bool = False
        self.inflated: bool = False            # too few points -> passthrough + wide band

    # -- fitting ----------------------------------------------------------- #
    def _grouped(self, s: np.ndarray, y: np.ndarray):
        """Reduce calibration to (rep_score, sum_y, count) groups, <= max_groups."""
        order = np.argsort(s, kind="mergesort")
        s, y = s[order], y[order]
        uniq = np.unique(s)
        if uniq.size <= self.max_groups:
            reps, inv = uniq, np.searchsorted(uniq, s)
            gsum = np.zeros(uniq.size)
            gcnt = np.zeros(uniq.size)
            np.add.at(gsum, inv, y)
            np.add.at(gcnt, inv, 1.0)
            return reps, gsum, gcnt
        # quantile-bin into <= max_groups bins
        edges = np.unique(np.quantile(s, np.linspace(0.0, 1.0, self.max_groups + 1)))
        bin_idx = np.clip(np.searchsorted(edges[1:-1], s, side="right"), 0, edges.size - 2)
        nb = edges.size - 1
        ssum = np.zeros(nb)
        ysum = np.zeros(nb)
        cnt = np.zeros(nb)
        np.add.at(ssum, bin_idx, s)
        np.add.at(ysum, bin_idx, y)
        np.add.at(cnt, bin_idx, 1.0)
        keep = cnt > 0
        reps = ssum[keep] / cnt[keep]            # mean score per bin (ascending)
        gsum, gcnt = ysum[keep], cnt[keep]
        # merge any reps that collide numerically (keeps reps strictly increasing)
        reps, inv = np.unique(reps, return_inverse=True)
        msum = np.zeros(reps.size)
        mcnt = np.zeros(reps.size)
        np.add.at(msum, inv, gsum)
        np.add.at(mcnt, inv, gcnt)
        return reps, msum, mcnt

    def _precompute_gaps(self, reps_mean: np.ndarray, reps_cnt: np.ndarray):
        """p0[g], p1[g] = IVAP value of a singleton (label 0/1) inserted at gap g."""
        G = reps_mean.size
        P0 = np.empty(G + 1)
        P1 = np.empty(G + 1)
        for g in range(G + 1):
            for label, P in ((0.0, P0), (1.0, P1)):
                means = np.empty(G + 1)
                cnts = np.empty(G + 1)
                means[:g] = reps_mean[:g]
                cnts[:g] = reps_cnt[:g]
                means[g] = label
                cnts[g] = 1.0
                means[g + 1:] = reps_mean[g:]
                cnts[g + 1:] = reps_cnt[g:]
                P[g] = pav_isotonic(means, cnts)[g]
        return np.clip(P0, 0.0, 1.0), np.clip(P1, 0.0, 1.0)

    def fit(self, scores, outcomes) -> "VennAbersCalibrator":
        s = np.asarray(scores, dtype=np.float64).ravel()
        y = np.asarray(outcomes, dtype=np.float64).ravel()
        mask = np.isfinite(s) & np.isfinite(y)
        s, y = s[mask], y[mask]
        if s.shape != y.shape:
            raise ValueError("scores/outcomes length mismatch")
        if y.size and not np.all((y == 0) | (y == 1)):
            raise ValueError("outcomes must be binary {0, 1}")
        self.n_calibration = int(s.size)
        if s.size < self.min_calibration:
            self.inflated = True
            self.fitted = True
            self._reps = self._P0 = self._P1 = None
            return self
        reps, gsum, gcnt = self._grouped(s, y)
        self._reps = reps
        self._P0, self._P1 = self._precompute_gaps(gsum / gcnt, gcnt)
        self.inflated = False
        self.fitted = True
        return self

    # -- prediction -------------------------------------------------------- #
    def predict(self, scores) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized predict -> (probability, lower=p0, upper=p1) arrays."""
        if not self.fitted:
            raise RuntimeError("VennAbersCalibrator not fitted")
        s = np.asarray(scores, dtype=np.float64).ravel()
        if self.inflated:
            p = np.clip(s, 0.0, 1.0)
            return p, np.clip(p - 0.25, 0.0, 1.0), np.clip(p + 0.25, 0.0, 1.0)
        gap = np.searchsorted(self._reps, s, side="right")     # 0..G
        p0 = self._P0[gap]
        p1 = self._P1[gap]
        denom = 1.0 - p0 + p1
        p = np.where(denom > 0, p1 / denom, 0.5 * (p0 + p1))
        return p, p0, p1

    def predict_one(self, s: float) -> VennAbersResult:
        p, lo, hi = self.predict([s])
        return VennAbersResult(float(p[0]), float(lo[0]), float(hi[0]))
