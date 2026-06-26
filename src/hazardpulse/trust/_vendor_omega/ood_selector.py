"""
ood_selector.py - pick the BEST OOD detector for the current representation, by measurement.

Measured fact: no single OOD score wins everywhere. The coherence basin ties-or-beats on tabular
features; plain Mahalanobis beats it on some image-backbone features; near-OOD is hard for all. So
rather than ASSUME the basin (the keystone mistake in reverse), this selects among a panel
- basin energy, energy-logsumexp (Liu 2020), Mahalanobis (Lee 2018), k-NN distance (Sun 2022),
max-softmax (Hendrycks 2017) - by AUROC on a validation OOD set, and uses the winner. It is the
champion idea applied to the trust signal: never assume, always measure.

If real OOD validation examples are available, select on them (best). If not, select on synthetic
pseudo-OOD (heavy noise + covariate permutation) - a general-novelty proxy. All scores are oriented
HIGH = more out-of-distribution, so the selected scorer drops into TrustedDecision as the OOD source
(TrustedDecision(..., ood_scorer=selector)). The coherence scores come from `clf` (ood_score /
decision_energy / predict_proba); the distance scores are computed on the raw input representation.
"""
from __future__ import annotations

import numpy as np

__all__ = ["OODSelector", "MahalanobisOOD"]


class MahalanobisOOD:
    """A MODEL-FREE OOD detector: min Mahalanobis distance to the class means under a shared (pooled)
    precision, on standardized inputs (Lee 2018). Pure numpy - no neural forward - so it scores in
    microseconds and drops into TrustedDecision(..., ood_scorer=...) for the LOW-LATENCY served path
    (where the deep head's basin energy is too slow per-request). HIGH = more out-of-distribution.

    Honest scope: a distance detector on the RAW input space. Strong on tabular / backbone features
    (the regime the low-latency path serves); the coherence basin can still win on representations
    where energy beats distance - use the deep-head TrustedDecision there. Reg-floored so the precision
    is always PD."""

    def __init__(self, *, reg=1e-3):
        self.reg = float(reg)
        self.means_ = None
        self.prec_ = None
        self._mu = None
        self._sd = None

    def _z(self, X):
        Z = (np.asarray(X, np.float64) - self._mu) / self._sd
        return np.clip(np.nan_to_num(Z, nan=0.0), -1e4, 1e4)

    def fit(self, X, y):
        X = np.asarray(X, np.float64); y = np.asarray(y)
        self._mu = X.mean(0); self._sd = X.std(0) + 1e-6
        F = self._z(X); classes = np.unique(y)
        self.means_ = np.stack([F[y == c].mean(0) for c in classes])
        cov = sum((F[y == c] - self.means_[i]).T @ (F[y == c] - self.means_[i])
                  for i, c in enumerate(classes)) / len(F)
        self.prec_ = np.linalg.pinv(cov + self.reg * np.eye(cov.shape[0]))
        return self

    def score(self, X):
        F = self._z(X); d = F[:, None, :] - self.means_[None]
        return np.einsum("ncd,de,nce->nc", d, self.prec_, d).min(1)   # min distance over classes


def _auroc(s_in, s_out):
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.zeros(len(s_in)), np.ones(len(s_out))]
    s = np.clip(np.nan_to_num(np.r_[s_in, s_out], nan=0.0), -1e9, 1e9)
    return float(roc_auc_score(y, s))


class OODSelector:
    """Select and apply the best-measured OOD detector for `clf`'s representation."""

    def __init__(self, clf, *, knn_k=10):
        self.clf = clf
        self.knn_k = int(knn_k)
        self.selected_ = None
        self.report_: dict = {}
        self._maha = None
        self._nn = None
        self._mu = None
        self._sd = None

    def _features(self, X):
        # the distance detectors operate on the RAW input representation (standardized) - e.g. the
        # backbone features for images - which is where Mahalanobis/kNN are strong (measured).
        Z = (np.asarray(X, np.float64) - self._mu) / self._sd
        return np.clip(np.nan_to_num(Z, nan=0.0), -1e4, 1e4)

    def _fit_refs(self, F, y):
        classes = np.unique(y)
        means = np.stack([F[y == c].mean(0) for c in classes])
        cov = sum((F[y == c] - means[i]).T @ (F[y == c] - means[i]) for i, c in enumerate(classes)) / len(F)
        self._maha = (means, np.linalg.pinv(cov + 1e-3 * np.eye(len(cov))))
        from sklearn.neighbors import NearestNeighbors
        self._nn = NearestNeighbors(n_neighbors=self.knn_k).fit(F)

    def _score_all(self, X):
        """Every candidate OOD score on X, oriented HIGH = more out-of-distribution."""
        X = np.asarray(X, float); F = self._features(X)
        p = np.asarray(self.clf.predict_proba(X), float)
        e = np.asarray(self.clf.decision_energy(X), float)          # per-class free energy (low = fits)
        a = -e; lse = a.max(1) + np.log(np.exp(a - a.max(1, keepdims=True)).sum(1))   # logsumexp(-energy)
        means, prec = self._maha
        d = F[:, None, :] - means[None]
        maha = np.einsum("ncd,de,nce->nc", d, prec, d).min(1)
        knn = self._nn.kneighbors(F)[0].mean(1)
        return {
            "basin": np.asarray(self.clf.ood_score(X), float),      # min basin free energy
            "energy_lse": -lse,                                     # low logsumexp = OOD
            "max_softmax": -p.max(1),                              # low confidence = OOD
            "mahalanobis": maha,                                   # far from class means = OOD
            "knn": knn,                                            # far from training points = OOD
        }

    def fit(self, Xin, yin, *, Xood=None, val_frac=0.3, seed=0):
        """Build the Mahalanobis/kNN references on an in-distribution split, then score each detector
        on (held-out in-dist vs OOD) and keep the AUROC winner. `Xood` = real OOD examples if you have
        them (best); otherwise synthetic pseudo-OOD (noise + covariate permutation) is used."""
        Xin = np.asarray(Xin, float); yin = np.asarray(yin)
        rng = np.random.default_rng(seed); n = len(Xin); idx = rng.permutation(n)
        nv = max(1, int(val_frac * n)); vi, ri = idx[:nv], idx[nv:]
        Xref = Xin[ri]
        self._mu = Xref.mean(0); self._sd = Xref.std(0) + 1e-6
        self._fit_refs(self._features(Xref), yin[ri])
        Xval = Xin[vi]
        if Xood is None:
            r2 = np.random.default_rng(seed + 1)
            noise = Xval + r2.normal(0, 1.0, Xval.shape)
            cov = Xval[:, r2.permutation(Xval.shape[1])]
            Xood = np.vstack([noise, cov]).astype(float)
        sin, sout = self._score_all(Xval), self._score_all(np.asarray(Xood, float))
        self.report_ = {m: _auroc(sin[m], sout[m]) for m in sin}
        self.selected_ = max(self.report_, key=self.report_.get)
        return self

    def score(self, X):
        """The selected detector's OOD score for X (HIGH = more out-of-distribution)."""
        return self._score_all(X)[self.selected_]
