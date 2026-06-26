"""
selective.py - selective prediction (risk-coverage) for the abstaining engines.

Every coherence surface here can ABSTAIN: the control plane on OOD, the selector on an
incoherent fusion pattern, the OmegaOne loop on novelty. Abstention is only useful if you
can MEASURE and TUNE it. An abstaining classifier trades COVERAGE (the fraction it answers)
for RISK (the error rate ON the answers). This module makes that trade-off first-class:

  - risk_coverage(confidence, correct) : the full risk-vs-coverage curve.
  - aurc(confidence, correct)          : Area Under the Risk-Coverage curve - the standard
                                         SCALAR for abstention QUALITY. LOWER is better: it
                                         measures whether the confidence ordering puts the
                                         errors LAST (does the engine know what it doesn't
                                         know?). A random confidence gives aurc ~ overall
                                         error rate; a perfect ranker drives it toward 0.
  - threshold_for_coverage(...)        : answer ~target_coverage of inputs.
  - threshold_for_risk(...)            : the MAX-coverage threshold whose answered-set risk
                                         stays <= a target (Chow's rule) - the operational
                                         knob: "abstain just enough to guarantee <= X% error".

Works on ANY scalar confidence signal: the selector's max candidate score, the control
plane's margin, or -(min basin energy). No coherence magic - this is honest selective
classification, the measurable core of "the engine knows when to abstain".
"""
from __future__ import annotations

import numpy as np


def risk_coverage(confidence, correct):
    """Sort predictions by DESCENDING confidence and admit them progressively. Returns
    (coverage, risk, threshold) arrays: coverage[i] = (i+1)/n answered; risk[i] = error rate
    over the i+1 most-confident; threshold[i] = the confidence of the i-th admitted (answer
    iff confidence >= threshold[i])."""
    confidence = np.asarray(confidence, float)
    correct = np.asarray(correct, bool)
    if len(confidence) != len(correct):
        raise ValueError("confidence and correct must be the same length")
    n = len(confidence)
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    order = np.argsort(-confidence, kind="stable")
    c, t = correct[order], confidence[order]
    coverage = np.arange(1, n + 1) / n
    risk = np.cumsum(~c) / np.arange(1, n + 1)
    return coverage, risk, t


def _trapz(yv, xv) -> float:
    """Trapezoidal integral of y over x (NumPy-2-safe; np.trapz was removed in 2.x)."""
    yv, xv = np.asarray(yv, float), np.asarray(xv, float)
    return float(np.sum(np.diff(xv) * (yv[1:] + yv[:-1]) * 0.5))


def aurc(confidence, correct) -> float:
    """Area Under the Risk-Coverage curve (trapezoid over coverage in [1/n, 1]). LOWER is
    better - the confidence ordering concentrates errors at low confidence."""
    coverage, risk, _ = risk_coverage(confidence, correct)
    if len(coverage) < 2:
        return float(risk[0]) if len(risk) else 0.0
    return _trapz(risk, coverage)


def ece(confidence, correct, bins=15) -> float:
    """Expected Calibration Error: the gap between confidence and accuracy, averaged over
    confidence bins (population-weighted). LOWER is better - a model is calibrated when, among
    the inputs it calls 80%-confident, ~80% are actually correct. A confidently-wrong model
    (the hallucination failure) has high ECE; this is where a generative basin's likelihood
    often beats a softmax's over-confident scores."""
    confidence = np.asarray(confidence, float)
    correct = np.asarray(correct, bool)
    n = len(confidence)
    if n == 0:
        return 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (confidence > lo) & (confidence <= hi) if i > 0 else (confidence >= lo) & (confidence <= hi)
        c = int(m.sum())
        if c:
            total += (c / n) * abs(correct[m].mean() - confidence[m].mean())
    return float(total)


def coverage_at_threshold(confidence, threshold) -> float:
    """Fraction of inputs answered if we answer iff confidence >= threshold."""
    confidence = np.asarray(confidence, float)
    return float((confidence >= threshold).mean()) if len(confidence) else 0.0


def threshold_for_coverage(confidence, target_coverage) -> float:
    """The confidence threshold that answers ~target_coverage of inputs (answer iff
    confidence >= threshold)."""
    confidence = np.asarray(confidence, float)
    if len(confidence) == 0:
        return float("inf")
    q = float(np.clip(1.0 - target_coverage, 0.0, 1.0))
    return float(np.quantile(confidence, q))


def threshold_for_risk(confidence, correct, target_risk) -> float:
    """Chow's rule: the LOWEST confidence threshold (MAX coverage) whose answered-set risk is
    still <= target_risk. Returns +inf if no non-empty answered set achieves it (abstain on
    everything). Answer iff confidence >= the returned threshold."""
    coverage, risk, thr = risk_coverage(confidence, correct)
    ok = np.where(risk <= target_risk)[0]
    if len(ok) == 0:
        return float("inf")
    return float(thr[ok[-1]])      # largest admitted index still under target risk == max coverage


def selective_report(confidence, correct, coverages=(0.5, 0.7, 0.9, 1.0)) -> dict:
    """A compact selective-prediction summary: overall accuracy, AURC, and the risk at a few
    target coverages - the numbers you operate an abstaining model by."""
    correct = np.asarray(correct, bool)
    out = {"n": int(len(correct)), "accuracy": float(correct.mean()) if len(correct) else 0.0,
           "aurc": aurc(confidence, correct), "risk_at_coverage": {}}
    cov, risk, _ = risk_coverage(confidence, correct)
    for tc in coverages:
        if len(cov) == 0:
            out["risk_at_coverage"][tc] = None
            continue
        i = int(np.clip(round(tc * len(cov)) - 1, 0, len(cov) - 1))
        out["risk_at_coverage"][tc] = float(risk[i])
    return out
