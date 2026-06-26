"""
monitoring.py - ProductionMonitor: watch a STREAM of served trusted-decisions over time.

Deployed models decay; this is the operational layer that catches it. Per served decision it tracks
abstention rate, OOD-score DRIFT (population-stability index vs a reference window), and - as ground
truth arrives - realized accuracy and conformal COVERAGE (overall + per group), plus model-version
LINEAGE from the receipt (alerts if the deployed model_sha256 changes mid-stream). `report()` is the
live health an operator watches; `alerts()` is the threshold breaches that should page someone.
stdlib + numpy.
"""
from __future__ import annotations

from collections import Counter, deque

import numpy as np

__all__ = ["ProductionMonitor", "psi"]


def psi(reference, current, *, bins=10) -> float:
    """Population Stability Index between two score samples (how much the OOD-score distribution has
    shifted). ~0 = stable; > 0.1 = moderate drift; > 0.25 = major drift (industry model-risk rule)."""
    ref, cur = np.asarray(reference, float), np.asarray(current, float)
    if len(ref) < 2 or len(cur) < 2:
        return 0.0
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1)); edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref) + 1e-6
    c = np.histogram(cur, edges)[0] / len(cur) + 1e-6
    return float(np.sum((c - r) * np.log(c / r)))


class ProductionMonitor:
    def __init__(self, *, coverage_target=0.9, window=500, drift_alert=0.25,
                 coverage_slack=0.05, abstain_alert=0.5):
        self.coverage_target = float(coverage_target)
        self.window = int(window)
        self.drift_alert = float(drift_alert)
        self.coverage_slack = float(coverage_slack)
        self.abstain_alert = float(abstain_alert)
        self.n = 0
        self._abstained = 0
        self._ood = deque(maxlen=self.window)
        self._ood_ref = None
        self._correct = 0
        self._answered = 0
        self._covered = 0
        self._labeled = 0
        self._grp = {}                                       # group -> [covered, total]
        self._models = Counter()
        self._model_changes = 0
        self._last_model = None

    def set_reference(self, ood_scores):
        """Pin the in-distribution OOD-score reference (e.g. the calibration set) for drift PSI."""
        self._ood_ref = np.asarray([s for s in ood_scores if s is not None], float)
        return self

    def observe(self, decision, *, y_true=None, group=None):
        """Ingest one served decision dict (prediction / abstained / ood_score / conformal_set /
        receipt). Pass y_true once it is known to score realized accuracy + coverage."""
        self.n += 1
        self._abstained += int(bool(decision.get("abstained")))
        s = decision.get("ood_score")
        if s is not None:
            self._ood.append(float(s))
        model = (decision.get("receipt") or {}).get("model_sha256")
        if model is not None:
            self._models[model] += 1
            if self._last_model is not None and model != self._last_model:
                self._model_changes += 1
            self._last_model = model
        if y_true is not None:
            self._labeled += 1
            if not decision.get("abstained"):
                self._answered += 1
                self._correct += int(decision.get("prediction") == y_true)
            cset = decision.get("conformal_set")
            if cset is not None:
                covered = int(y_true in cset)
                self._covered += covered
                if group is not None:
                    g = self._grp.setdefault(group, [0, 0]); g[0] += covered; g[1] += 1
        return self

    def report(self) -> dict:
        drift = psi(self._ood_ref, self._ood) if self._ood_ref is not None and self._ood else 0.0
        return {
            "decisions": self.n,
            "abstention_rate": round(self._abstained / self.n, 4) if self.n else 0.0,
            "ood_drift_psi": round(float(drift), 4),
            "accuracy": round(self._correct / self._answered, 4) if self._answered else None,
            "coverage": round(self._covered / self._labeled, 4) if self._labeled else None,
            "coverage_by_group": {g: round(c / t, 4) for g, (c, t) in self._grp.items() if t},
            "model_versions": len(self._models),
            "model_changes": self._model_changes,
            "alerts": self.alerts(),
        }

    def alerts(self) -> list:
        a = []
        if self.n and self._abstained / self.n > self.abstain_alert:
            a.append(f"abstention_rate {self._abstained / self.n:.2f} > {self.abstain_alert}")
        if self._ood_ref is not None and self._ood:
            d = psi(self._ood_ref, self._ood)
            if d > self.drift_alert:
                a.append(f"ood_drift_psi {d:.2f} > {self.drift_alert} (input distribution shifted)")
        if self._labeled and self._covered / self._labeled < self.coverage_target - self.coverage_slack:
            a.append(f"coverage {self._covered / self._labeled:.3f} < target {self.coverage_target} (calibration decayed)")
        for g, (c, t) in self._grp.items():
            if t >= 30 and c / t < self.coverage_target - self.coverage_slack:
                a.append(f"group {g} coverage {c / t:.3f} below target (per-group fairness breach)")
        if self._model_changes:
            a.append(f"model_sha256 changed {self._model_changes}x mid-stream (lineage)")
        return a
