"""
conformal.py - conformal prediction SETS with a finite-sample coverage GUARANTEE.

Abstention says "I don't know." A conformal SET says something stronger and rigorous: "the true
class is in {A, B} with >= 90% probability" - a distribution-free, finite-sample marginal coverage
promise that holds for ANY underlying model. It is coherence's "honest about uncertainty" thesis
taken to its provable form, and it composes with any probability source here (the basin
classifier's predict_proba, the deep coherence head, the selector, OmegaOne).

Two standard split-conformal methods:
  - LAC (Least-Ambiguous set Classifier): smallest sets at a target coverage. Threshold on the
    true-class probability. Tight but can under-cover the hard examples.
  - APS (Adaptive Prediction Sets): set size ADAPTS to difficulty (bigger where the model is
    unsure) and gives better conditional coverage.

Both guarantee: P(true class in the set) >= 1 - alpha on fresh data drawn like the calibration
set. Calibrate on a HELD-OUT split (never the training or test data).
"""
from __future__ import annotations

import numpy as np


def _conformal_quantile(scores, alpha):
    """The finite-sample-corrected quantile: ceil((n+1)(1-alpha)) / n, clipped to [0,1], taken
    with the 'higher' rule. This correction is what turns an empirical quantile into a guarantee."""
    s = np.asarray(scores, float)
    n = len(s)
    if n == 0:
        return float("inf")
    level = min(np.ceil((n + 1) * (1.0 - alpha)) / n, 1.0)
    return float(np.quantile(s, level, method="higher"))


def calibrate_lac(proba_cal, y_cal, alpha=0.1) -> float:
    """LAC threshold q: the conformal quantile of the non-conformity s = 1 - p(true class). The
    set {y : p(y) >= 1 - q} then covers the true class with probability >= 1 - alpha on fresh data."""
    p = np.asarray(proba_cal, float)
    y = np.asarray(y_cal)
    s = 1.0 - p[np.arange(len(y)), y]
    return _conformal_quantile(s, alpha)


def predict_sets_lac(proba, qhat):
    """Boolean (n, k) membership mask: class y is in the set iff p(y) >= 1 - qhat."""
    return np.asarray(proba, float) >= (1.0 - qhat)


def calibrate_aps(proba_cal, y_cal, alpha=0.1) -> float:
    """APS threshold: non-conformity = the cumulative probability (sorted high->low) up to AND
    including the true class. The conformal quantile of that is the set-inclusion budget."""
    p = np.asarray(proba_cal, float)
    y = np.asarray(y_cal)
    n = len(y)
    if n == 0:
        return float("inf")
    order = np.argsort(-p, axis=1)                        # classes high->low prob, per row
    csum = np.cumsum(np.take_along_axis(p, order, axis=1), axis=1)   # cumulative prob in that order
    rank_of_true = (order == y[:, None]).argmax(1)        # where the true class sits in the order
    s = csum[np.arange(n), rank_of_true]                  # cumprob up to + INCLUDING the true class
    return _conformal_quantile(s, alpha)                  # vectorized; identical to the per-row loop


def predict_sets_aps(proba, qhat):
    """Boolean (n, k) mask: include classes in descending-probability order until the cumulative
    probability first reaches qhat (adaptive set sizes - bigger where the model is unsure)."""
    p = np.asarray(proba, float)
    n, k = p.shape
    order = np.argsort(-p, axis=1)
    csum = np.cumsum(np.take_along_axis(p, order, axis=1), axis=1)
    take = np.minimum((csum < qhat).sum(1) + 1, k)        # #classes to include per row (>= qhat, inclusive)
    include_sorted = np.arange(k)[None, :] < take[:, None]  # first `take` positions in sorted order
    sets = np.zeros((n, k), bool)
    np.put_along_axis(sets, order, include_sorted, axis=1)  # scatter back to original class order
    return sets                                           # vectorized; identical to the per-row loop


def calibrate_raps(proba_cal, y_cal, alpha=0.1, *, lam=0.01, kreg=None, seed=0):
    """RAPS (Regularized APS, Angelopoulos et al. 2020): APS plus a penalty `lam` on including
    classes past rank `kreg`, which BOUNDS the set size in the heavy tail (where plain APS bloats)
    while keeping the coverage guarantee. Non-conformity for the true class at sorted position r
    (0-based): cumprob_before_r + U*p_r + lam*max(0, (r+1) - kreg); kreg auto = the (1-alpha)
    quantile of true-class ranks. Returns (qhat, lam, kreg). Randomization is SEEDED (reproducible)."""
    p = np.asarray(proba_cal, float)
    y = np.asarray(y_cal)
    n = len(y)
    if n == 0:
        return float("inf"), lam, (1 if kreg is None else kreg)
    rng = np.random.default_rng(seed)
    order = np.argsort(-p, axis=1)
    ps = np.take_along_axis(p, order, axis=1)
    rank_true = (order == y[:, None]).argmax(1)
    if kreg is None:
        kreg = max(int(np.quantile(rank_true + 1, 1.0 - alpha)), 1)
    csum_excl = np.cumsum(ps, axis=1) - ps                     # cumprob strictly before each position
    scores = (csum_excl[np.arange(n), rank_true]
              + rng.uniform(size=n) * ps[np.arange(n), rank_true]
              + lam * np.maximum(0, (rank_true + 1) - kreg))
    return _conformal_quantile(scores, alpha), lam, kreg


def predict_sets_raps(proba, qhat, lam, kreg, *, seed=1):
    """RAPS membership mask: include classes in descending-prob order while the regularized score
    (cumprob_before + U*p_r + lam*max(0,(r+1)-kreg)) stays <= qhat; the top class is always kept
    (never-empty). Sets are prefix-closed in the sorted order."""
    p = np.asarray(proba, float)
    n, k = p.shape
    rng = np.random.default_rng(seed)
    order = np.argsort(-p, axis=1)
    ps = np.take_along_axis(p, order, axis=1)
    csum_excl = np.cumsum(ps, axis=1) - ps
    pos = np.arange(k)[None, :]
    score_at = csum_excl + rng.uniform(size=(n, 1)) * ps + lam * np.maximum(0, (pos + 1) - kreg)
    inc = score_at <= qhat
    inc = np.cumsum(inc[:, ::-1], axis=1)[:, ::-1] > 0         # prefix-closed (include up to the last kept)
    inc[:, 0] = True                                          # never empty
    sets = np.zeros((n, k), bool)
    np.put_along_axis(sets, order, inc, axis=1)
    return sets


def coverage(sets, y) -> float:
    """Empirical coverage: the fraction of samples whose TRUE class is in the predicted set
    (should be >= 1 - alpha by the guarantee)."""
    sets = np.asarray(sets, bool)
    y = np.asarray(y)
    return float(sets[np.arange(len(y)), y].mean()) if len(y) else 0.0


def avg_set_size(sets) -> float:
    """Mean number of classes per set (efficiency - smaller is better at the same coverage)."""
    sets = np.asarray(sets, bool)
    return float(sets.sum(1).mean()) if len(sets) else 0.0


class ConformalPredictor:
    """Wrap any classifier with a `predict_proba` into a conformal SET predictor with a coverage
    guarantee. `fit_calibrate(Xcal, ycal)` sets the threshold; `predict_set(X)` returns the
    boolean class-membership masks; `predict(X)` returns lists of class labels."""

    def __init__(self, clf, *, alpha=0.1, method="aps", lam=0.01, kreg=None):
        self.clf = clf
        self.alpha = float(alpha)
        self.method = method                       # "aps" | "lac" | "raps"
        self.lam, self.kreg = lam, kreg            # RAPS regularization (size penalty + cutoff)
        self.qhat = None
        self.classes_ = None

    def fit_calibrate(self, Xcal, ycal):
        proba = self.clf.predict_proba(Xcal)
        self.classes_ = list(getattr(self.clf, "classes_", range(np.asarray(proba).shape[1])))
        y_idx = np.array([self.classes_.index(t) for t in np.asarray(ycal)])
        if self.method == "raps":
            self.qhat, self.lam, self.kreg = calibrate_raps(proba, y_idx, self.alpha,
                                                            lam=self.lam, kreg=self.kreg)
        elif self.method == "aps":
            self.qhat = calibrate_aps(proba, y_idx, self.alpha)
        else:
            self.qhat = calibrate_lac(proba, y_idx, self.alpha)
        return self

    def predict_set(self, X):
        return self.predict_set_proba(self.clf.predict_proba(X))

    def predict_set_proba(self, proba):
        """Conformal set from PRECOMPUTED probabilities - lets a caller that already ran the model (e.g.
        a fused single-forward decode) skip a redundant predict_proba. Identical result to predict_set."""
        proba = np.asarray(proba, float)
        if self.method == "raps":
            return predict_sets_raps(proba, self.qhat, self.lam, self.kreg)
        if self.method == "aps":
            return predict_sets_aps(proba, self.qhat)
        return predict_sets_lac(proba, self.qhat)

    def predict(self, X):
        sets = self.predict_set(X)
        return [[self.classes_[j] for j in np.where(row)[0]] for row in sets]


class MondrianConformalPredictor:
    """Group-conditional (Mondrian) conformal: a SEPARATE coverage guarantee per group, so
    P(true in set | group=g) >= 1 - alpha for EVERY group g - not just marginally. This is the
    coverage promise regulated / fairness use cases actually require (each subpopulation covered),
    which plain marginal conformal does NOT give (it can systematically under-cover a hard group).

    `group_fn(X) -> array of group ids` partitions both calibration and test (e.g. a sensitive
    attribute, a region, or the predicted class). Each group gets its own conformal threshold; an
    unseen test group falls back to the pooled threshold."""

    def __init__(self, clf, group_fn, *, alpha=0.1, method="lac", lam=0.01, kreg=None):
        self.clf = clf
        self.group_fn = group_fn
        self.alpha = float(alpha)
        self.method = method
        self.lam, self.kreg = lam, kreg
        self.qhat_: dict = {}
        self.q_pooled_ = None
        self.kreg_: dict = {}
        self.classes_ = None

    def _calibrate(self, proba, y_idx):
        if self.method == "raps":
            q, _, kreg = calibrate_raps(proba, y_idx, self.alpha, lam=self.lam, kreg=self.kreg)
            return q, kreg
        if self.method == "aps":
            return calibrate_aps(proba, y_idx, self.alpha), None
        return calibrate_lac(proba, y_idx, self.alpha), None

    def _sets(self, proba, q, kreg):
        if self.method == "raps":
            return predict_sets_raps(proba, q, self.lam, kreg)
        if self.method == "aps":
            return predict_sets_aps(proba, q)
        return predict_sets_lac(proba, q)

    def fit_calibrate(self, Xcal, ycal):
        proba = np.asarray(self.clf.predict_proba(Xcal), float)
        self.classes_ = list(getattr(self.clf, "classes_", range(proba.shape[1])))
        y_idx = np.array([self.classes_.index(t) for t in np.asarray(ycal)])
        groups = np.asarray(self.group_fn(Xcal))
        self.q_pooled_, kreg_pooled = self._calibrate(proba, y_idx)     # fallback for unseen groups
        self.kreg_["__pooled__"] = kreg_pooled
        for g in np.unique(groups):
            m = groups == g
            if m.sum() >= 2:
                self.qhat_[g], self.kreg_[g] = self._calibrate(proba[m], y_idx[m])
        return self

    def predict_set(self, X):
        proba = np.asarray(self.clf.predict_proba(X), float)
        groups = np.asarray(self.group_fn(X))
        sets = np.zeros(proba.shape, bool)
        for g in np.unique(groups):
            m = groups == g
            q = self.qhat_.get(g, self.q_pooled_)
            kreg = self.kreg_.get(g, self.kreg_["__pooled__"])
            sets[m] = self._sets(proba[m], q, kreg)
        return sets

    def predict(self, X):
        return [[self.classes_[j] for j in np.where(row)[0]] for row in self.predict_set(X)]


def group_coverage(sets, y, groups) -> dict:
    """Per-group empirical coverage - the audit a marginal coverage number hides. Returns
    {group: coverage}; a fair system has all >= 1 - alpha, not just the mean."""
    sets = np.asarray(sets, bool); y = np.asarray(y); groups = np.asarray(groups)
    out = {}
    for g in np.unique(groups):
        m = groups == g
        out[g] = float(sets[m][np.arange(m.sum()), y[m]].mean()) if m.any() else 0.0
    return out
