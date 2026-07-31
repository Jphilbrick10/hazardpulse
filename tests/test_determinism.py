"""Determinism gate: training must be a pure function of (data, hyperparameters).

Two independent trainings on identical inputs must produce IDENTICAL models --
tree for tree, value for value -- regardless of the state of NumPy's *global*
RNG. This is a regression gate for a real shipped bug class: row subsampling
used a seeded ``RandomState`` while feature subsampling read the global
``np.random`` stream, so repeated runs could grow different trees and the
repository's core reproducibility claim was false.

The adversarial part of this gate is the global reseed between the two
trainings: under the old code, different global seeds produced different
feature subsets and therefore different models, so this test goes red on the
bug it exists to prevent.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from hazardpulse.earthquake import definitive_model as eq_definitive
from hazardpulse.earthquake import v4_regional as eq_v4
from hazardpulse.tornado import definitive_model as tor_definitive
from hazardpulse.tornado import operational_storm as tor_operational

MODULES = [eq_definitive, eq_v4, tor_definitive, tor_operational]


def _synthetic(n: int = 240, d: int = 12, seed: int = 7):
    """Fixed synthetic classification data (seeded, never global)."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, d).astype(np.float32)
    logits = X[:, 0] * 1.5 - X[:, 3] + 0.5 * X[:, 7]
    y = (logits + rng.randn(n) * 0.5 > 0).astype(np.float32)
    return X, y


def _train(module, global_seed: int):
    """Train one GBT with the global RNG deliberately set to a given state."""
    X, y = _synthetic()
    # Adversarial: the global stream must be irrelevant to the result.
    np.random.seed(global_seed)
    model = module.GradientBoostedTrees(
        n_trees=12,
        max_depth=3,
        subsample=0.7,   # exercise the row-subsample RNG path
        colsample=0.5,   # exercise the feature-subsample RNG path
    )
    model.fit(X, y)
    return model


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_training_is_identical_across_runs_and_global_rng(module):
    a = _train(module, global_seed=0)
    b = _train(module, global_seed=987654321)

    assert a.init_pred == b.init_pred
    assert json.dumps(a.trees, sort_keys=True) == json.dumps(
        b.trees, sort_keys=True
    ), f"{module.__name__}: two identical trainings grew different trees"

    X, _ = _synthetic()
    assert np.array_equal(a.predict_proba(X), b.predict_proba(X)), (
        f"{module.__name__}: identical models produced different predictions"
    )


@pytest.mark.parametrize("module", MODULES, ids=lambda m: m.__name__)
def test_training_does_not_consume_global_rng(module):
    """Training must not advance the global stream (no hidden global draws)."""
    np.random.seed(1234)
    before = np.random.random_sample(4)

    np.random.seed(1234)
    _train(module, global_seed=1234)
    # _train reseeded to 1234 itself; if fit() drew nothing from the global
    # stream, the next draws continue exactly where seed 1234 begins.
    after = np.random.random_sample(4)

    assert np.array_equal(before, after), (
        f"{module.__name__}: fit() consumed the global np.random stream"
    )
