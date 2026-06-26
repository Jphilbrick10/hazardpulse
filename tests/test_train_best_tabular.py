"""Tests for the BestTabular accuracy-upgrade trainer.

The metric helpers always run. The BestTabular comparison + never-worse guard run
where the SOTA libs are installed (skipped in the lightweight CI test job).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
# Make omega_one importable when present (sibling repo); the comparison tests
# importorskip on it, so they run here and skip wherever omega_one is absent.
_OMEGA = REPO.parent / "Coherence" / "omega_one"
if _OMEGA.is_dir() and str(_OMEGA) not in sys.path:
    sys.path.insert(0, str(_OMEGA))


def _load():
    spec = importlib.util.spec_from_file_location("tbt", REPO / "scripts" / "train_best_tabular.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_roc_auc_and_brier():
    m = _load()
    y = np.array([0, 0, 1, 1])
    assert m.roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0    # perfect ranking
    assert m.roc_auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0    # inverted
    assert abs(m.brier(y, y.astype(float))) < 1e-12               # perfect probs


def _synth(seed=0, n=900, d=8):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d)
    w = rng.randn(d)
    logit = X @ w + 0.7 * X[:, 0] * X[:, 1]
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X[:600], y[:600], X[600:], y[600:]


def test_never_worse_guard_keeps_better_baseline():
    pytest.importorskip("xgboost")
    pytest.importorskip("omega.super_ensemble", reason="omega_one not importable")
    m = _load()
    Xtr, ytr, Xte, yte = _synth()
    # a PERFECT baseline -> BestTabular cannot beat it -> guard keeps the baseline
    report, proba = m.compare(Xtr, ytr, Xte, yte, baseline_proba=yte.astype(float), seeds=(0,))
    assert report["baseline"]["auc"] == 1.0
    assert report["champion"] == "baseline"
    assert proba.shape == (len(yte),)
    assert "auc_mean" in report["best_tabular"]


def test_best_tabular_beats_a_weak_baseline():
    pytest.importorskip("xgboost")
    pytest.importorskip("omega.super_ensemble", reason="omega_one not importable")
    m = _load()
    Xtr, ytr, Xte, yte = _synth(seed=1)
    weak = np.full(len(yte), float(ytr.mean()))     # constant base-rate predictor
    report, _ = m.compare(Xtr, ytr, Xte, yte, baseline_proba=weak, seeds=(0,))
    assert report["delta_auc"] > 0.0                 # BestTabular beats climatology
    assert report["champion"] == "best_tabular"
