"""Vendored: omega_one's frozen-integer-forest reproducer (pure numpy).

This is the SERVE half of the VerifiableForest moat -- the function a third party
(here, HazardPulse's numpy-only live scorer) executes from the portable JSON
constants alone to RE-RUN a forest decision bit-for-bit, with no model object and
no heavy ML libraries. Kept byte-faithful to the source so the served decision is
identical to the one the offline trainer measured and signed.

Source: omega_one/omega/verifiable_forest.py (_decision_fp,
predict_fp_from_forest_constants). Re-vendor via scripts/vendor_omega_trust.py.
"""

from __future__ import annotations

import numpy as np


def _decision_fp(feat, thr, left, right, value_fp, root, base_fp, X, *, op, default_left):
    """Deterministic float-threshold traversal + int64 per-class accumulation. ``op`` is the
    source library's split rule ('le' = x<=thr -> left, 'lt' = x<thr -> left);
    ``default_left[node]`` routes a NaN/missing feature. Pure integer => the argmax (the
    decision) is bit-identical across substrates."""
    X = np.asarray(X, float)
    n = len(X)
    F = np.tile(base_fp.astype(np.int64), (n, 1))
    for r in root:
        nd = np.full(n, r, np.int64); active = feat[nd] >= 0
        while active.any():
            a = np.where(active)[0]; cur = nd[a]
            xv = X[a, feat[cur]]
            base = (xv < thr[cur]) if op == "lt" else (xv <= thr[cur])
            go_left = np.where(np.isnan(xv), default_left[cur].astype(bool), base)
            nd[a] = np.where(go_left, left[cur], right[cur])
            active = feat[nd] >= 0
        F += value_fp[nd]
    return F


def predict_fp_from_forest_constants(constants, X):
    """RE-RUN the decision from the portable integer constants alone (no model object). Returns
    ``(labels, F_int64)`` - exactly what a third party executes to REPRODUCE a signed decision."""
    feat = np.asarray(constants["feat"], np.int32); thr = np.asarray(constants["thr"], np.float64)
    left = np.asarray(constants["left"], np.int32); right = np.asarray(constants["right"], np.int32)
    value_fp = np.asarray(constants["value_fp"], np.int64)
    root = np.asarray(constants["tree_root"], np.int64)
    base_fp = np.asarray(constants.get("base_fp", [0] * len(constants["classes"])), np.int64)
    default_left = np.asarray(constants.get("default_left", [0] * len(feat)), np.int8)
    F = _decision_fp(feat, thr, left, right, value_fp, root, base_fp, X,
                     op=constants.get("op", "le"), default_left=default_left)
    classes = constants["classes"]
    return np.array([classes[i] for i in F.argmax(1)], dtype=object), F
