"""Serve a frozen VerifiableForest in HazardPulse's numpy-only live path.

The offline trainer (``scripts/train_best_tabular.py``) exports the deployable
champion as ``results/calibration/<hazard>_forest_fp.json`` -- a portable integer
forest. This module loads it and produces, in pure numpy with NO heavy ML
dependency, the same probabilities the trainer measured. Because the underlying
traversal is the byte-faithful vendored reproducer, the served score is identical
to the offline one and re-runnable bit-for-bit by anyone holding the JSON.

The raw forest probability is intentionally an *uncalibrated* model score: the
trust spine's Venn-Abers calibrator + conformal interval still wrap it, so the
public number stays honest. ``ForestScorer`` is the drop-in the scorer calls to
get that raw score.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ._vendor_omega.forest_fp import predict_fp_from_forest_constants


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, float)))


class ForestScorer:
    """Pure-numpy evaluator for an exported frozen forest (binary hazard head)."""

    def __init__(self, constants: dict):
        if int(len(constants.get("classes", []))) != 2:
            raise ValueError("ForestScorer supports binary forests (2 classes) only")
        self.constants = constants
        self.scale = float(constants["scale"])
        self.classes_ = list(constants["classes"])
        self.model_sha256 = constants.get("model_sha256")

    @classmethod
    def from_file(cls, path: str | Path) -> "ForestScorer":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def decision_function(self, X) -> np.ndarray:
        """The frozen forest's int64 per-class decision function (N, 2)."""
        _, F = predict_fp_from_forest_constants(self.constants, np.atleast_2d(np.asarray(X, float)))
        return F

    def raw_proba(self, X) -> np.ndarray:
        """P(event) as the served forest computes it: sigmoid(margin / scale).

        This is the model score the Venn-Abers calibrator then maps to a calibrated,
        honest probability -- never published raw.
        """
        F = self.decision_function(X)
        margin = (F[:, 1].astype(np.float64) - F[:, 0].astype(np.float64)) / self.scale
        return _sigmoid(margin)

    def raw_proba_one(self, x) -> float:
        return float(self.raw_proba(np.atleast_2d(np.asarray(x, float)))[0])


def load_forest_scorer(hazard: str, directory: str | Path = "results/calibration") -> ForestScorer | None:
    """Load ``<hazard>_forest_fp.json`` if a deployable forest has been exported, else None."""
    path = Path(directory) / f"{hazard}_forest_fp.json"
    if not path.exists():
        return None
    return ForestScorer.from_file(path)
