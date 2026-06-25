"""Binary-forecast calibration measurement.

HazardPulse forecasts are binary probabilities: P(M6+ in 30 days), P(RI in 24h),
P(tornado in 24h). The platform's measured failure is that these probabilities
are *miscalibrated* — live Brier Skill Score is negative (worse than
climatology). This module measures calibration directly so we can (a) gate on it
(``G4_CALIBRATION_FLOOR``), (b) show honest reliability diagrams on the public
scoreboard, and (c) prove before/after that the trust spine fixes it.

The omega_one ``selective.ece`` we vendored is a top-label *classification*
confidence ECE; hazard forecasting needs *binary reliability* (a single
probability of one event), which is what this module provides — the standard
reliability ECE and Murphy's Brier decomposition (reliability / resolution /
uncertainty). Pure numpy, no external deps.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

__all__ = [
    "ReliabilityBin",
    "ReliabilityCurve",
    "reliability_curve",
    "expected_calibration_error",
    "maximum_calibration_error",
    "brier_score",
    "brier_skill_score",
    "brier_decomposition",
    "BrierDecomposition",
]


def _clean(probs, outcomes):
    p = np.asarray(probs, dtype=np.float64).ravel()
    y = np.asarray(outcomes, dtype=np.float64).ravel()
    if p.shape != y.shape:
        raise ValueError(f"probs/outcomes length mismatch: {p.shape} vs {y.shape}")
    mask = np.isfinite(p) & np.isfinite(y)
    p, y = p[mask], y[mask]
    if p.size and (p.min() < -1e-9 or p.max() > 1 + 1e-9):
        raise ValueError("probabilities must lie in [0, 1]")
    return np.clip(p, 0.0, 1.0), y


@dataclass(frozen=True)
class ReliabilityBin:
    lo: float
    hi: float
    count: int
    mean_predicted: float   # mean forecast probability in the bin
    observed_freq: float    # observed event frequency in the bin


@dataclass(frozen=True)
class ReliabilityCurve:
    bins: list[ReliabilityBin]
    n: int
    base_rate: float

    def as_dict(self) -> dict:
        return {
            "n": self.n,
            "base_rate": self.base_rate,
            "bins": [asdict(b) for b in self.bins],
        }


def reliability_curve(probs, outcomes, n_bins: int = 10) -> ReliabilityCurve:
    """Bin forecasts by predicted probability; report predicted vs observed per bin.

    A perfectly calibrated forecaster sits on the diagonal (mean_predicted ==
    observed_freq in every populated bin).
    """
    p, y = _clean(probs, outcomes)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # rightmost edge inclusive so p == 1.0 lands in the last bin
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    bins: list[ReliabilityBin] = []
    for b in range(n_bins):
        in_bin = idx == b
        cnt = int(in_bin.sum())
        if cnt:
            mp = float(p[in_bin].mean())
            of = float(y[in_bin].mean())
        else:
            mp = of = float("nan")
        bins.append(ReliabilityBin(float(edges[b]), float(edges[b + 1]), cnt, mp, of))
    return ReliabilityCurve(bins=bins, n=int(p.size),
                            base_rate=float(y.mean()) if p.size else float("nan"))


def expected_calibration_error(probs, outcomes, n_bins: int = 10) -> float:
    """ECE: count-weighted mean |mean_predicted - observed_freq| over populated bins."""
    curve = reliability_curve(probs, outcomes, n_bins=n_bins)
    if not curve.n:
        return float("nan")
    total = 0.0
    for b in curve.bins:
        if b.count:
            total += b.count * abs(b.mean_predicted - b.observed_freq)
    return total / curve.n


def maximum_calibration_error(probs, outcomes, n_bins: int = 10) -> float:
    """MCE: the worst single-bin calibration gap."""
    curve = reliability_curve(probs, outcomes, n_bins=n_bins)
    gaps = [abs(b.mean_predicted - b.observed_freq) for b in curve.bins if b.count]
    return max(gaps) if gaps else float("nan")


def brier_score(probs, outcomes) -> float:
    """Mean squared error of the probabilistic forecast."""
    p, y = _clean(probs, outcomes)
    return float(np.mean((p - y) ** 2)) if p.size else float("nan")


def brier_skill_score(probs, outcomes) -> float:
    """BSS vs the climatology (base-rate) forecast. >0 beats climatology; <0 is worse.

    This is the headline calibration+refinement skill score; HazardPulse's live
    tornado BSS is currently negative (the problem this program fixes).
    """
    p, y = _clean(probs, outcomes)
    if not p.size:
        return float("nan")
    base = float(y.mean())
    bs_clim = base * (1.0 - base)
    if bs_clim <= 0.0:
        return float("nan")  # degenerate: all-events or no-events window
    return 1.0 - brier_score(p, y) / bs_clim


@dataclass(frozen=True)
class BrierDecomposition:
    brier: float
    reliability: float   # lower is better (0 = perfectly calibrated)
    resolution: float    # higher is better (forecasts separate outcomes)
    uncertainty: float   # base_rate*(1-base_rate); intrinsic difficulty
    n_bins: int

    def as_dict(self) -> dict:
        return asdict(self)


def brier_decomposition(probs, outcomes, n_bins: int = 10) -> BrierDecomposition:
    """Murphy's decomposition: Brier = reliability - resolution + uncertainty.

    reliability = sum_k n_k/N (p_k - o_k)^2   (calibration error; want ~0)
    resolution  = sum_k n_k/N (o_k - obar)^2  (sharpness/discrimination; want high)
    uncertainty = obar*(1-obar)               (irreducible)
    """
    curve = reliability_curve(probs, outcomes, n_bins=n_bins)
    p, y = _clean(probs, outcomes)
    n = curve.n
    if not n:
        nan = float("nan")
        return BrierDecomposition(nan, nan, nan, nan, n_bins)
    obar = float(y.mean())
    reliability = 0.0
    resolution = 0.0
    for b in curve.bins:
        if b.count:
            wk = b.count / n
            reliability += wk * (b.mean_predicted - b.observed_freq) ** 2
            resolution += wk * (b.observed_freq - obar) ** 2
    uncertainty = obar * (1.0 - obar)
    return BrierDecomposition(
        brier=brier_score(p, y),
        reliability=float(reliability),
        resolution=float(resolution),
        uncertainty=float(uncertainty),
        n_bins=n_bins,
    )
