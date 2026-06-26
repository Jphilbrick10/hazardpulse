"""The numpy-only serve path reproduces the trained forest's probabilities exactly.

This closes the loop on the VerifiableForest swap: whatever the offline trainer
measured, the live scorer re-runs bit-for-bit from the exported JSON with no heavy
ML dependency. The trainer-side build is xgboost-gated (skips where absent); the
hand-built-constants tests always run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from hazardpulse.trust.forest_serve import ForestScorer, load_forest_scorer

REPO = Path(__file__).resolve().parents[1]
_OMEGA = REPO.parent / "Coherence" / "omega_one"
if _OMEGA.is_dir() and str(_OMEGA) not in sys.path:
    sys.path.insert(0, str(_OMEGA))


def _hand_forest():
    """A tiny 1-tree binary forest: if feature0 <= 0.5 -> class1 margin +scale, else -scale."""
    scale = 1 << 20
    constants = {
        "spec": "omega-one/verifiable-forest/v1",
        "scale": scale,
        "op": "le",
        "classes": [0, 1],
        # node 0: split feat0 <= 0.5 -> left(1) / right(2); nodes 1,2 leaves
        "feat": [0, -1, -1],
        "thr": [0.5, 0.0, 0.0],
        "left": [1, -1, -1],
        "right": [2, -1, -1],
        "value_fp": [[0, 0], [0, scale], [0, -scale]],
        "tree_root": [0],
        "default_left": [0, 0, 0],
        "base_fp": [0, 0],
    }
    return constants, scale


def test_serve_reproduces_handbuilt_forest():
    constants, scale = _hand_forest()
    sc = ForestScorer(constants)
    X = np.array([[0.2], [0.9], [0.5], [0.51]])
    p = sc.raw_proba(X)
    # feat0 <= 0.5 -> margin +1 -> sigmoid(1); else sigmoid(-1)
    expected = 1.0 / (1.0 + np.exp(-np.array([1.0, -1.0, 1.0, -1.0])))
    assert np.allclose(p, expected, atol=1e-9)
    assert sc.raw_proba_one([0.2]) > 0.7 and sc.raw_proba_one([0.9]) < 0.3


def test_from_file_and_loader(tmp_path):
    constants, _ = _hand_forest()
    path = tmp_path / "earthquake_forest_fp.json"
    path.write_text(json.dumps(constants), encoding="utf-8")
    sc = ForestScorer.from_file(path)
    assert sc.classes_ == [0, 1]
    assert load_forest_scorer("earthquake", tmp_path) is not None
    assert load_forest_scorer("tornado", tmp_path) is None     # not exported


def test_rejects_non_binary_forest():
    constants, _ = _hand_forest()
    constants["classes"] = [0, 1, 2]
    with pytest.raises(ValueError):
        ForestScorer(constants)


def test_serve_matches_real_xgboost_export():
    """End-to-end: train -> export constants -> serve path == trainer's own proba."""
    pytest.importorskip("xgboost")
    pytest.importorskip("omega.verifiable_forest", reason="omega_one not importable")
    import importlib.util
    spec = importlib.util.spec_from_file_location("tbt", REPO / "scripts" / "train_best_tabular.py")
    tbt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tbt)

    rng = np.random.RandomState(0)
    X = rng.randn(400, 6); w = rng.randn(6)
    y = (rng.uniform(size=400) < 1 / (1 + np.exp(-(X @ w)))).astype(int)
    Xtr, ytr, Xte = X[:300], y[:300], X[300:]
    vf = tbt._verifiable_forest(Xtr, ytr, Xte, seed=0)

    served = ForestScorer(vf["constants"]).raw_proba(Xte)
    assert np.allclose(served, vf["proba"], atol=1e-12)    # serve == train, bit-for-bit
