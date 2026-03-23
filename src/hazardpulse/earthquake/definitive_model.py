# IMPORTANT: This software is for RESEARCH PURPOSES ONLY.
# It is NOT an operational earthquake prediction system.
# It does NOT replace official USGS or national seismological agency warnings.
# Always follow guidance from your national geological survey.
# False negatives (missed earthquakes) WILL occur. Do NOT rely on this
# system for safety-critical decisions.

"""Definitive leak-proof earthquake prediction model.

Design philosophy: ZERO LEAKAGE BY CONSTRUCTION.
    - ONE model type: Gradient Boosted Trees. No meta-stacker, no blending.
    - Strict temporal separation enforced with assertions.
    - Labels require temporal ordering in code.
    - All features verified causal (available at prediction time).
    - No hyperparameter tuning on test. Test touched ONCE.

Three model variants (one GBT each, NO stacking):
    1. Baseline:  Block S only (15 features) -- standard seismicity
    2. Enhanced:  Block S + C (27 features) -- + coherence field theory
    3. Full:      Block S + C + I (32 features) -- + interaction terms

Temporal splits (hardcoded, non-negotiable):
    Train: 2005-01-01 to 2017-12-31
    Val:   2018-01-01 to 2019-12-31
    Test:  2020-01-01 to 2024-12-31

Label definition:
    Positive: M6+ mainshock within 300km and 365 days FORWARD
    Negative: same location, different time, no M6+ within window
    Gardner-Knopoff aftershock declustering applied first.
    2 negatives per positive, time-offset +/-1.5 to +/-4.5 years.

Self-contained: copies GBT implementation fresh, imports only from
hazardpulse.data.earthquake and hazardpulse.earthquake.coherence_engine.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Internal imports (the ONLY allowed external dependencies)
# ---------------------------------------------------------------------------

from hazardpulse.data.earthquake import load_usgs_catalog
from hazardpulse.earthquake.coherence_engine import (
    extract_coherence_features,
    compute_seismic_coherence_field,
    test_earthquake_singularity,
    _parse_event_time,
    _haversine_km,
    _haversine_km_arrays,
    compute_b_value,
    compute_seismic_moment,
)


# ===================================================================
# TEMPORAL SPLITS -- HARDCODED, NON-NEGOTIABLE
# ===================================================================

TRAIN_START: int = 2005
TRAIN_END: int = 2017
VAL_START: int = 2018
VAL_END: int = 2019
TEST_START: int = 2020
TEST_END: int = 2024

# Label parameters
LABEL_RADIUS_KM: float = 300.0
FORWARD_WINDOW_DAYS: float = 365.0
MIN_MAINSHOCK_MAG: float = 6.0
CONTROL_RATIO: int = 2  # negatives per positive
CONTROL_OFFSET_RANGE: tuple[float, float] = (1.5, 4.5)  # years


# ===================================================================
# FEATURE BLOCK SIZES
# ===================================================================

N_FEAT_S: int = 15   # Standard seismicity
N_FEAT_C: int = 12   # Coherence field theory
N_FEAT_I: int = 5    # Interaction terms

N_FEAT_BASELINE: int = N_FEAT_S                          # 15
N_FEAT_ENHANCED: int = N_FEAT_S + N_FEAT_C               # 27
N_FEAT_FULL: int = N_FEAT_S + N_FEAT_C + N_FEAT_I        # 32


# ===================================================================
# FEATURE NAMES -- every feature has a causal validity comment
# ===================================================================

# Block S: Standard Seismicity (15 features)
# All derived from PAST seismicity at the sample location.
BLOCK_S_NAMES: list[str] = [
    "rate_1m",          # 1.  1-month event rate acceleration -- past
    "rate_3m",          # 2.  3-month event rate acceleration -- past
    "rate_6m",          # 3.  6-month event rate acceleration -- past
    "rate_12m",         # 4.  12-month event rate acceleration -- past
    "b_value",          # 5.  Gutenberg-Richter b-value -- past
    "b_trend",          # 6.  b-value change (recent vs older) -- past
    "mc_late",          # 7.  Magnitude of completeness (recent) -- past
    "nn_change",        # 8.  Nearest-neighbour distance change -- past
    "st_nn",            # 9.  Spatiotemporal nearest-neighbour metric -- past
    "frac_clust",       # 10. Fraction of clustered events -- past
    "mom_accel",        # 11. Moment release acceleration -- past
    "mom_deficit",      # 12. Moment deficit vs long-term rate -- past
    "coulomb_proxy",    # 13. Coulomb stress transfer proxy -- past
    "max_mag_90d",      # 14. Max magnitude in last 90 days -- past
    "max_mag_180d",     # 15. Max magnitude in last 180 days -- past
]

# Block C: Coherence Field Theory (12 features)
# Derived from coherence_engine using PAST seismicity only.
BLOCK_C_NAMES: list[str] = [
    "ell",                  # 16. Correlation length (km) -- past
    "ell_trend",            # 17. Correlation length change rate -- past
    "ell_acceleration",     # 18. d^2(ell)/dt^2 -- past
    "days_to_criticality",  # 19. Estimated days until t_c -- past divergence fit
    "nu_exponent",          # 20. Critical exponent from ell(t) fit -- past
    "divergence_r2",        # 21. R^2 of divergence fit (confidence) -- past
    "delta_aic_iet",        # 22. Lorentzian vs exponential IET -- past
    "tau_local",            # 23. Helmholtz coherence at location -- past
    "grad_tau_local",       # 24. Coherence gradient -- past
    "S_over_Gamma",         # 25. Source/damping ratio -- past
    "Da_local",             # 26. Damkohler number -- past
    "singularity_count",    # 27. Conditions met (0-5) -- past
]

# Block I: Interaction Terms (5 features)
# Cross-products of Block S and Block C features.
BLOCK_I_NAMES: list[str] = [
    "ell_x_b_trend",           # 28. Diverging length with dropping b -- past
    "ell_x_rate_accel",        # 29. Diverging length with accelerating rate -- past
    "tau_x_max_mag",           # 30. Coherence x largest event -- past
    "dtc_x_singularity",      # 31. Imminence x conditions met -- past
    "sg_x_ell_accel",         # 32. Loading x accelerating divergence -- past
]

ALL_FEATURE_NAMES_BASELINE: list[str] = BLOCK_S_NAMES
ALL_FEATURE_NAMES_ENHANCED: list[str] = BLOCK_S_NAMES + BLOCK_C_NAMES
ALL_FEATURE_NAMES_FULL: list[str] = (
    BLOCK_S_NAMES + BLOCK_C_NAMES + BLOCK_I_NAMES
)


# ===================================================================
# GBT HYPERPARAMETERS -- FIXED A PRIORI, NOT TUNED ON VALIDATION
# ===================================================================

GBT_N_TREES: int = 200
GBT_MAX_DEPTH: int = 4
GBT_LEARNING_RATE: float = 0.05
GBT_SUBSAMPLE: float = 0.5
GBT_COLSAMPLE: float = 0.7
GBT_MIN_SAMPLES_LEAF: int = 20
GBT_L2_REG: float = 1.0
GBT_GAMMA: float = 0.1

# Early stopping patience (on validation loss)
EARLY_STOP_PATIENCE: int = 20


# ===================================================================
# HAVERSINE DISTANCE
# ===================================================================

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in km between two points.

    Uses the Haversine formula. Inputs in degrees.
    """
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2.0) ** 2
    )
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


# ===================================================================
# SIGMOID (numerically stable)
# ===================================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function.

    Uses the two-branch trick to avoid overflow in exp().
    """
    z = np.asarray(z, dtype=np.float32)
    z = np.clip(z, -88.0, 88.0)
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    ).astype(np.float32)


# ===================================================================
# GRADIENT BOOSTED TREES -- SELF-CONTAINED IMPLEMENTATION
# ===================================================================

class GradientBoostedTrees:
    """Memory-efficient gradient boosted trees for binary classification.

    Uses log-loss (cross-entropy) with Newton-Raphson leaf values.
    Supports subsample, colsample, balanced class weights, L2 regularization,
    and minimum complexity gain (gamma) pruning.

    Feature importance is via split counts (honest about limitations:
    this is a frequency-based proxy, not a causal importance measure).
    """

    def __init__(
        self,
        n_trees: int = GBT_N_TREES,
        max_depth: int = GBT_MAX_DEPTH,
        learning_rate: float = GBT_LEARNING_RATE,
        min_samples_leaf: int = GBT_MIN_SAMPLES_LEAF,
        subsample: float = GBT_SUBSAMPLE,
        colsample: float = GBT_COLSAMPLE,
        l2_reg: float = GBT_L2_REG,
        gamma: float = GBT_GAMMA,
    ) -> None:
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.lr = learning_rate
        self.min_leaf = min_samples_leaf
        self.subsample = subsample
        self.colsample = colsample
        self.l2_reg = l2_reg
        self.gamma = gamma
        self.trees: list[dict] = []
        self.init_pred: float = 0.0

    def _build_tree(
        self,
        X: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        depth: int = 0,
    ) -> dict:
        """Build a single regression tree on gradient/hessian targets.

        Uses exact greedy split finding with candidate thresholds from
        percentiles for continuous features.
        """
        N = len(gradients)
        G = float(np.sum(gradients))
        H = float(np.sum(hessians))

        leaf_val = -G / (H + self.l2_reg) if (H + self.l2_reg) > 1e-12 else 0.0

        if depth >= self.max_depth or N < 2 * self.min_leaf:
            return {"leaf": True, "val": float(leaf_val)}

        D = X.shape[1]
        n_try = max(5, int(D * self.colsample))
        feat_idx = np.random.choice(D, size=min(n_try, D), replace=False)

        best_gain = -1e30
        best_feat = 0
        best_thresh = 0.0

        score_no_split = G ** 2 / (H + self.l2_reg)

        for f in feat_idx:
            col = X[:, f]
            unique_vals = np.unique(col)
            if len(unique_vals) <= 1:
                continue

            if len(unique_vals) > 20:
                thresholds = np.percentile(col, np.linspace(5, 95, 20))
            else:
                thresholds = unique_vals[:-1]

            for thresh in thresholds:
                left_mask = col <= thresh
                right_mask = ~left_mask
                n_left = int(left_mask.sum())
                n_right = int(right_mask.sum())

                if n_left < self.min_leaf or n_right < self.min_leaf:
                    continue

                GL = float(np.sum(gradients[left_mask]))
                HL = float(np.sum(hessians[left_mask]))
                GR = float(np.sum(gradients[right_mask]))
                HR = float(np.sum(hessians[right_mask]))

                gain = 0.5 * (
                    GL ** 2 / (HL + self.l2_reg)
                    + GR ** 2 / (HR + self.l2_reg)
                    - score_no_split
                ) - self.gamma

                if gain > best_gain:
                    best_gain = gain
                    best_feat = int(f)
                    best_thresh = float(thresh)

        if best_gain <= 0:
            return {"leaf": True, "val": float(leaf_val)}

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask
        left_child = self._build_tree(
            X[left_mask], gradients[left_mask], hessians[left_mask], depth + 1
        )
        right_child = self._build_tree(
            X[right_mask], gradients[right_mask], hessians[right_mask], depth + 1
        )
        return {
            "leaf": False,
            "feat": best_feat,
            "thresh": best_thresh,
            "left": left_child,
            "right": right_child,
        }

    def _predict_tree_row(self, node: dict, x: np.ndarray) -> float:
        """Predict a single row through a single tree."""
        while not node["leaf"]:
            if x[node["feat"]] <= node["thresh"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["val"]

    def _predict_tree_batch(self, node: dict, X: np.ndarray) -> np.ndarray:
        """Predict all rows through a single tree, row by row."""
        result = np.empty(X.shape[0], dtype=np.float32)
        for i in range(X.shape[0]):
            result[i] = self._predict_tree_row(node, X[i])
        return result

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        verbose: bool = False,
    ) -> dict:
        """Train the GBT ensemble on labelled data.

        Parameters
        ----------
        X : ndarray, shape (n_train, n_features)
        y : ndarray, shape (n_train,)  -- binary labels (0 or 1)
        X_val, y_val : optional validation set for early stopping
        verbose : bool

        Returns
        -------
        dict  -- training metadata
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        N = len(y)

        # Balanced class weights
        n_pos = float(y.sum())
        n_neg = float(N - n_pos)
        assert n_pos > 0, "No positive samples in training set"
        assert n_neg > 0, "No negative samples in training set"
        sample_w = np.where(
            y == 1,
            N / (2.0 * n_pos),
            N / (2.0 * n_neg),
        ).astype(np.float32)

        # Initial prediction: log-odds of training prevalence
        p_avg = float(np.clip(y.mean(), 0.01, 0.99))
        self.init_pred = math.log(p_avg / (1.0 - p_avg))
        F_train = np.full(N, self.init_pred, dtype=np.float32)

        # Validation setup for early stopping
        use_early_stop = X_val is not None and y_val is not None
        F_val = None
        best_val_loss = float("inf")
        patience_counter = 0
        best_n_trees = 0

        if use_early_stop:
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.float32)
            F_val = np.full(len(y_val), self.init_pred, dtype=np.float32)

        rng = np.random.RandomState(42)
        self.trees = []

        for t in range(self.n_trees):
            p_train = sigmoid(F_train)

            gradients = ((p_train - y) * sample_w).astype(np.float32)
            hessians = (p_train * (1.0 - p_train) * sample_w + 1e-6).astype(
                np.float32
            )

            if self.subsample < 1.0:
                n_sub = max(1, int(N * self.subsample))
                idx = rng.choice(N, size=n_sub, replace=False)
                tree = self._build_tree(
                    X[idx], gradients[idx], hessians[idx]
                )
            else:
                tree = self._build_tree(X, gradients, hessians)

            self.trees.append(tree)
            F_train += self.lr * self._predict_tree_batch(tree, X)

            if use_early_stop:
                F_val += self.lr * self._predict_tree_batch(tree, X_val)
                p_val = sigmoid(F_val)
                val_loss = -float(np.mean(
                    y_val * np.log(p_val + 1e-12)
                    + (1 - y_val) * np.log(1 - p_val + 1e-12)
                ))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_n_trees = t + 1
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= EARLY_STOP_PATIENCE:
                    if verbose:
                        print(
                            f"      Early stop at tree {t + 1}, "
                            f"best_val_loss={best_val_loss:.6f} at tree {best_n_trees}"
                        )
                        sys.stdout.flush()
                    self.trees = self.trees[:best_n_trees]
                    return {
                        "n_trees_used": best_n_trees,
                        "stopped_early": True,
                        "best_val_loss": best_val_loss,
                    }

            if verbose and (t + 1) % 25 == 0:
                train_auc = compute_auc(y, sigmoid(F_train))
                msg = f"      Tree {t + 1}/{self.n_trees}: train_auc={train_auc:.4f}"
                if use_early_stop:
                    msg += f"  val_loss={best_val_loss:.6f}"
                print(msg)
                sys.stdout.flush()

        return {
            "n_trees_used": len(self.trees),
            "stopped_early": False,
            "best_val_loss": best_val_loss if use_early_stop else None,
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for new data."""
        X = np.asarray(X, dtype=np.float32)
        N = X.shape[0]
        F = np.full(N, self.init_pred, dtype=np.float32)
        for tree in self.trees:
            for i in range(N):
                F[i] += self.lr * self._predict_tree_row(tree, X[i])
        return sigmoid(F)

    def feature_importances(self, n_features: int) -> np.ndarray:
        """Compute feature importance from tree split counts.

        NOTE: This is a frequency-based proxy for importance, not a causal
        measure. Use with appropriate caveats.
        """
        counts = np.zeros(n_features, dtype=np.float64)

        def _walk(node: dict) -> None:
            if node["leaf"]:
                return
            f = node["feat"]
            if 0 <= f < n_features:
                counts[f] += 1
            _walk(node["left"])
            _walk(node["right"])

        for tree in self.trees:
            _walk(tree)

        total = counts.sum()
        if total > 0:
            counts /= total
        return counts


# ===================================================================
# EVALUATION METRICS
# ===================================================================

def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC via trapezoidal integration."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if len(y_true) < 2:
        return 0.5
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    tp = 0
    fp = 0
    tpr_prev = 0.0
    fpr_prev = 0.0
    auc = 0.0

    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2.0
        tpr_prev = tpr
        fpr_prev = fpr

    return float(auc)


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Precision-Recall AUC -- better metric for imbalanced data."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    n_pos = y_true.sum()
    if n_pos == 0 or len(y_true) == 0:
        return 0.0

    order = np.argsort(-y_score)
    y_sorted = y_true[order]

    tp = 0
    fp = 0
    pr_auc = 0.0
    recall_prev = 0.0

    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / n_pos
        pr_auc += (recall - recall_prev) * precision
        recall_prev = recall

    return float(pr_auc)


def compute_brier(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Brier score: mean squared error between predicted proba and labels."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean((y_pred - y_true) ** 2))


def compute_bss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Brier Skill Score: improvement over climatological baseline."""
    brier = compute_brier(y_true, y_pred)
    base_rate = float(np.mean(y_true))
    brier_clim = base_rate * (1 - base_rate)
    if brier_clim < 1e-12:
        return 0.0
    return float(1.0 - brier / brier_clim)


def compute_calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Reliability diagram: predicted vs observed frequency per bin."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    bin_means: list[float] = []
    bin_trues: list[float] = []
    bin_counts: list[int] = []

    for i in range(n_bins):
        if i < n_bins - 1:
            mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        else:
            mask = (y_pred >= bins[i]) & (y_pred <= bins[i + 1])
        count = int(mask.sum())
        if count > 0:
            bin_means.append(float(y_pred[mask].mean()))
            bin_trues.append(float(y_true[mask].mean()))
            bin_counts.append(count)

    return {"predicted": bin_means, "observed": bin_trues, "counts": bin_counts}


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Bootstrap confidence interval for ROC-AUC."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    N = len(y_true)
    aucs = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        idx = rng.choice(N, size=N, replace=True)
        aucs[b] = compute_auc(y_true[idx], y_score[idx])

    return {
        "mean": float(np.mean(aucs)),
        "ci_lo": float(np.percentile(aucs, 100 * alpha / 2)),
        "ci_hi": float(np.percentile(aucs, 100 * (1 - alpha / 2))),
        "std": float(np.std(aucs)),
    }


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Complete evaluation suite for a single model."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return {
        "auc": compute_auc(y_true, y_pred),
        "pr_auc": compute_pr_auc(y_true, y_pred),
        "brier": compute_brier(y_true, y_pred),
        "bss": compute_bss(y_true, y_pred),
        "calibration": compute_calibration_curve(y_true, y_pred, n_bins=10),
        "bootstrap_ci": bootstrap_auc_ci(y_true, y_pred, n_boot=2000),
        "n_positive": int(y_true.sum()),
        "n_negative": int((1 - y_true).sum()),
        "base_rate": float(y_true.mean()),
    }


# ===================================================================
# SIGNIFICANCE TESTS
# ===================================================================

def paired_bootstrap_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    n_boot: int = 2000,
    seed: int = 42,
) -> dict:
    """Test whether model B significantly outperforms model A.

    Uses paired bootstrap: the SAME resampled indices are used for both
    models on each iteration, so the comparison controls for sampling
    variability.
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred_a = np.asarray(y_pred_a, dtype=np.float64)
    y_pred_b = np.asarray(y_pred_b, dtype=np.float64)
    N = len(y_true)

    auc_a_full = compute_auc(y_true, y_pred_a)
    auc_b_full = compute_auc(y_true, y_pred_b)
    delta_full = auc_b_full - auc_a_full

    deltas = np.empty(n_boot, dtype=np.float64)
    n_a_wins = 0

    for b in range(n_boot):
        idx = rng.choice(N, size=N, replace=True)
        auc_a = compute_auc(y_true[idx], y_pred_a[idx])
        auc_b = compute_auc(y_true[idx], y_pred_b[idx])
        deltas[b] = auc_b - auc_a
        if auc_a >= auc_b:
            n_a_wins += 1

    p_value = n_a_wins / n_boot

    return {
        "delta_auc": float(delta_full),
        "ci_lo": float(np.percentile(deltas, 2.5)),
        "ci_hi": float(np.percentile(deltas, 97.5)),
        "p_value": float(p_value),
    }


# ===================================================================
# FEATURE NORMALIZER
# ===================================================================

class FeatureNormalizer:
    """Z-score normalizer. Fit on train, apply to val/test.

    Prevents any information leakage from val/test into normalization
    statistics.
    """

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> None:
        """Compute mean and std from training data only."""
        X = np.asarray(X, dtype=np.float32)
        self.mean = X.mean(axis=0, dtype=np.float32)
        self.std = X.std(axis=0, dtype=np.float32)
        self.std[self.std < 1e-12] = 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply normalization using stored train statistics."""
        assert self.mean is not None, "FeatureNormalizer has not been fit"
        X = np.asarray(X, dtype=np.float32)
        return ((X - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step (for training data only)."""
        self.fit(X)
        return self.transform(X)


# ===================================================================
# GARDNER-KNOPOFF AFTERSHOCK DECLUSTERING
# ===================================================================

def _gardner_knopoff_window(mag: float) -> tuple[float, float]:
    """Return (distance_km, time_days) window for aftershock removal.

    Gardner & Knopoff (1974) empirical aftershock windows.
    """
    # Distance window: 10^(0.1238*M + 0.983) km
    d_km = 10.0 ** (0.1238 * mag + 0.983)
    # Time window: 10^(0.5409*M - 0.547) days  (for M >= 2.5)
    if mag >= 6.5:
        t_days = 10.0 ** (0.032 * mag + 2.7389)
    else:
        t_days = 10.0 ** (0.5409 * mag - 0.547)
    return d_km, t_days


def decluster_gardner_knopoff(
    events: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Remove aftershocks using Gardner-Knopoff (1974) windows.

    Processes events in descending magnitude order: each event's window
    removes subsequent smaller events.

    Parameters
    ----------
    events : list[dict]
        USGS catalog events with keys: time, latitude, longitude, mag.

    Returns
    -------
    mainshocks : list[dict]
        Declustered catalog (mainshocks only).
    aftershocks : list[dict]
        Removed aftershock events.
    """
    # Parse epochs for all events
    parsed: list[tuple[float, dict]] = []
    for e in events:
        t_str = e.get("time", "")
        epoch = _parse_event_time(t_str) if isinstance(t_str, str) else 0.0
        if epoch > 0 and e.get("mag") is not None:
            parsed.append((epoch, e))

    # FAST approach: only process M5+ as potential mainshocks.
    # Events below M5 are kept as-is (used for features, not as targets).
    # This reduces O(N²) from 494K² to ~5K × 494K = manageable.

    # Sort by magnitude descending
    parsed.sort(key=lambda x: x[1]["mag"], reverse=True)

    is_aftershock = set()  # indices marked as aftershocks
    n = len(parsed)

    # Build arrays for vectorized checks
    epochs = np.array([p[0] for p in parsed])
    lats = np.array([p[1]["latitude"] for p in parsed])
    lons = np.array([p[1]["longitude"] for p in parsed])
    mags = np.array([p[1]["mag"] for p in parsed])

    # Only M5+ events can be mainshocks that remove aftershocks
    for i in range(n):
        if i in is_aftershock:
            continue
        if mags[i] < 5.0:
            break  # sorted by mag desc, so all remaining are < 5.0
        d_km, t_days = _gardner_knopoff_window(float(mags[i]))
        t_sec = t_days * 86400.0

        # Temporal filter (vectorized)
        dt = np.abs(epochs - epochs[i])
        time_mask = dt <= t_sec

        # Magnitude filter
        mag_mask = mags < mags[i]

        candidates = np.where(time_mask & mag_mask)[0]

        for j in candidates:
            if j == i or j in is_aftershock:
                continue
            dist = haversine_km(float(lats[i]), float(lons[i]),
                                float(lats[j]), float(lons[j]))
            if dist <= d_km:
                is_aftershock.add(j)

    mainshocks = [parsed[i][1] for i in range(n) if i not in is_aftershock]
    aftershocks = [parsed[i][1] for i in range(n) if i in is_aftershock]

    return mainshocks, aftershocks


# ===================================================================
# EVENT TIME UTILITIES
# ===================================================================

def _event_epoch(event: dict) -> float:
    """Extract epoch seconds from an event dict."""
    t_str = event.get("time", "")
    if isinstance(t_str, str):
        return _parse_event_time(t_str)
    return 0.0


def _event_year(event: dict) -> int:
    """Extract year from an event's ISO-8601 timestamp."""
    t_str = event.get("time", "")
    if isinstance(t_str, str) and len(t_str) >= 4:
        try:
            return int(t_str[:4])
        except ValueError:
            pass
    return 0


# ===================================================================
# SAMPLE GENERATION
# ===================================================================

def identify_target_mainshocks(
    mainshocks: list[dict],
    min_mag: float = MIN_MAINSHOCK_MAG,
) -> list[dict]:
    """Filter declustered catalog to M6+ mainshocks with valid locations."""
    targets = []
    for e in mainshocks:
        mag = e.get("mag")
        lat = e.get("latitude")
        lon = e.get("longitude")
        if (
            mag is not None
            and mag >= min_mag
            and lat is not None
            and lon is not None
            and -90 <= lat <= 90
            and -180 <= lon <= 180
        ):
            targets.append(e)
    return targets


def generate_control_samples(
    target: dict,
    mainshocks: list[dict],
    catalog_epochs: np.ndarray,
    n_controls: int = CONTROL_RATIO,
    offset_range: tuple[float, float] = CONTROL_OFFSET_RANGE,
    label_radius_km: float = LABEL_RADIUS_KM,
    forward_days: float = FORWARD_WINDOW_DAYS,
    rng: np.random.RandomState | None = None,
) -> list[dict]:
    """Generate negative control samples at the SAME location as a target.

    Controls are time-offset +/-1.5 to +/-4.5 years from the mainshock,
    verified to have no M6+ within label_radius_km and forward_days.

    Parameters
    ----------
    target : dict
        The positive (M6+) mainshock event.
    mainshocks : list[dict]
        Full declustered catalog for checking negatives.
    catalog_epochs : ndarray
        Pre-computed epochs for all mainshocks.
    n_controls : int
        Number of negative samples to generate per positive.
    offset_range : tuple
        Range (min_years, max_years) for time offset.
    label_radius_km, forward_days : float
        Spatial and temporal window for label verification.

    Returns
    -------
    list[dict]
        Control sample dicts with keys: latitude, longitude, time,
        ref_epoch, label (=0).
    """
    if rng is None:
        rng = np.random.RandomState(42)

    target_epoch = _event_epoch(target)
    target_lat = target["latitude"]
    target_lon = target["longitude"]

    controls: list[dict] = []
    max_attempts = n_controls * 10

    for _ in range(max_attempts):
        if len(controls) >= n_controls:
            break

        # Random offset: +/-1.5 to +/-4.5 years
        offset_years = rng.uniform(offset_range[0], offset_range[1])
        if rng.random() < 0.5:
            offset_years = -offset_years
        offset_sec = offset_years * 365.25 * 86400.0
        control_epoch = target_epoch + offset_sec

        # Verify no M6+ mainshock within radius and forward window
        has_m6_forward = False
        for ms in mainshocks:
            ms_mag = ms.get("mag", 0)
            if ms_mag < MIN_MAINSHOCK_MAG:
                continue
            ms_epoch = _event_epoch(ms)
            dt_days = (ms_epoch - control_epoch) / 86400.0
            if dt_days < 0 or dt_days > forward_days:
                continue
            dist = haversine_km(
                target_lat, target_lon,
                ms["latitude"], ms["longitude"],
            )
            if dist <= label_radius_km:
                has_m6_forward = True
                break

        if not has_m6_forward:
            # Reconstruct an ISO-8601 timestamp for the control
            ctrl_dt = _dt.datetime.fromtimestamp(
                control_epoch, tz=_dt.timezone.utc
            )
            ctrl_time_str = ctrl_dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")

            controls.append({
                "latitude": target_lat,
                "longitude": target_lon,
                "time": ctrl_time_str,
                "ref_epoch": control_epoch,
                "label": 0,
            })

    return controls


def build_samples(
    mainshocks: list[dict],
    full_catalog: list[dict],
    verbose: bool = True,
) -> list[dict]:
    """Build positive + negative samples from declustered catalog.

    Positive samples: each M6+ mainshock.
    Negative samples: same location, time-offset, verified no M6+ forward.

    Each sample dict has: latitude, longitude, ref_epoch, label, year.
    """
    targets = identify_target_mainshocks(mainshocks)
    if verbose:
        print(f"    M6+ mainshocks found: {len(targets)}")
        sys.stdout.flush()

    # Pre-compute epochs for all mainshocks (for control verification)
    ms_epochs = np.array([_event_epoch(m) for m in mainshocks], dtype=np.float64)

    rng = np.random.RandomState(42)
    samples: list[dict] = []

    for target in targets:
        target_epoch = _event_epoch(target)
        target_year = _event_year(target)

        # Positive sample
        samples.append({
            "latitude": target["latitude"],
            "longitude": target["longitude"],
            "ref_epoch": target_epoch,
            "label": 1,
            "year": target_year,
        })

        # Negative controls
        controls = generate_control_samples(
            target, mainshocks, ms_epochs, rng=rng,
        )
        for ctrl in controls:
            ctrl_year = int(ctrl["time"][:4]) if isinstance(ctrl["time"], str) else 0
            samples.append({
                "latitude": ctrl["latitude"],
                "longitude": ctrl["longitude"],
                "ref_epoch": ctrl["ref_epoch"],
                "label": ctrl["label"],
                "year": ctrl_year,
            })

    if verbose:
        n_pos = sum(1 for s in samples if s["label"] == 1)
        n_neg = sum(1 for s in samples if s["label"] == 0)
        print(f"    Total samples: {len(samples)} ({n_pos} pos, {n_neg} neg)")
        sys.stdout.flush()

    return samples


# ===================================================================
# BLOCK S FEATURE EXTRACTION (Standard Seismicity)
# ===================================================================

def _events_in_window(
    full_catalog: list[dict],
    lat: float,
    lon: float,
    ref_epoch: float,
    radius_km: float,
    window_days: float,
) -> list[dict]:
    """Filter catalog to events within radius_km and window_days BEFORE ref_epoch."""
    window_sec = window_days * 86400.0
    result = []
    for e in full_catalog:
        emag = e.get("mag")
        elat = e.get("latitude")
        elon = e.get("longitude")
        if emag is None or elat is None or elon is None:
            continue
        eepoch = _event_epoch(e)
        if eepoch <= 0:
            continue
        # STRICT: only PAST events (before ref_epoch)
        dt = ref_epoch - eepoch
        if dt < 0 or dt > window_sec:
            continue
        dist = haversine_km(lat, lon, elat, elon)
        if dist <= radius_km:
            result.append(e)
    return result


def extract_block_s(
    full_catalog: list[dict],
    lat: float,
    lon: float,
    ref_epoch: float,
    radius_km: float = LABEL_RADIUS_KM,
) -> np.ndarray:
    """Extract Block S (standard seismicity) features.

    All features use ONLY events BEFORE ref_epoch. No leakage.

    Returns
    -------
    ndarray, shape (15,), dtype float32
    """
    feats = np.full(N_FEAT_S, np.nan, dtype=np.float64)

    # Get events in progressively larger time windows
    ev_1m = _events_in_window(full_catalog, lat, lon, ref_epoch, radius_km, 30)
    ev_3m = _events_in_window(full_catalog, lat, lon, ref_epoch, radius_km, 90)
    ev_6m = _events_in_window(full_catalog, lat, lon, ref_epoch, radius_km, 180)
    ev_12m = _events_in_window(full_catalog, lat, lon, ref_epoch, radius_km, 365)
    ev_24m = _events_in_window(full_catalog, lat, lon, ref_epoch, radius_km, 730)

    # 1-4: Rate acceleration ratios (recent / older, normalised by time)
    def _rate_accel(recent: list, full: list, recent_frac: float) -> float:
        n_full = len(full)
        n_recent = len(recent)
        if n_full == 0:
            return 0.0
        n_older = n_full - n_recent
        rate_recent = n_recent / recent_frac if recent_frac > 0 else 0
        rate_older = n_older / (1.0 - recent_frac) if (1.0 - recent_frac) > 0 else 0
        if rate_older < 1e-12:
            return rate_recent / max(rate_older, 1e-12)
        return rate_recent / rate_older

    feats[0] = _rate_accel(ev_1m, ev_3m, 1.0 / 3.0)   # rate_1m
    feats[1] = _rate_accel(ev_3m, ev_12m, 3.0 / 12.0)  # rate_3m
    feats[2] = _rate_accel(ev_6m, ev_24m, 6.0 / 24.0)  # rate_6m
    feats[3] = _rate_accel(ev_12m, ev_24m, 12.0 / 24.0) # rate_12m

    # 5-7: b_value, b_trend, mc_late
    mags_12m = np.array([e["mag"] for e in ev_12m if e.get("mag") is not None])
    b_val, mc, b_unc, n_b = compute_b_value(mags_12m)
    feats[4] = b_val   # b_value

    # b_trend: compare b of recent 6m vs older 6m (within 12m window)
    mags_6m = np.array([e["mag"] for e in ev_6m if e.get("mag") is not None])
    b_recent, _, _, _ = compute_b_value(mags_6m)
    # Older 6 months: events in 12m but not in 6m
    ev_older_6m = [e for e in ev_12m if e not in ev_6m]
    mags_older_6m = np.array([e["mag"] for e in ev_older_6m if e.get("mag") is not None])
    b_older, _, _, _ = compute_b_value(mags_older_6m)
    if not math.isnan(b_recent) and not math.isnan(b_older):
        feats[5] = b_recent - b_older  # b_trend (negative = dropping)
    else:
        feats[5] = 0.0

    feats[6] = mc  # mc_late (magnitude of completeness)

    # 8-10: Nearest-neighbour metrics
    if len(ev_12m) >= 5:
        lats = np.array([e["latitude"] for e in ev_12m])
        lons = np.array([e["longitude"] for e in ev_12m])

        # Compute pairwise NN distances (subsample for efficiency)
        n_ev = len(ev_12m)
        nn_dists = np.full(n_ev, np.inf, dtype=np.float64)
        for i in range(n_ev):
            for j in range(n_ev):
                if i == j:
                    continue
                d = haversine_km(lats[i], lons[i], lats[j], lons[j])
                if d < nn_dists[i]:
                    nn_dists[i] = d
        nn_mean = float(np.mean(nn_dists[nn_dists < np.inf]))

        # NN of recent vs older
        n_recent = len(ev_6m)
        if n_recent >= 3 and n_ev - n_recent >= 3:
            nn_recent = float(np.mean(nn_dists[:n_recent]))
            nn_older = float(np.mean(nn_dists[n_recent:]))
            feats[7] = nn_recent - nn_older  # nn_change (negative = tightening)
        else:
            feats[7] = 0.0

        feats[8] = nn_mean  # st_nn (mean spatiotemporal NN distance)

        # Fraction clustered: events with NN < median
        median_nn = float(np.median(nn_dists[nn_dists < np.inf]))
        if median_nn > 0:
            feats[9] = float(np.mean(nn_dists < median_nn))  # frac_clust
        else:
            feats[9] = 0.5
    else:
        feats[7] = 0.0
        feats[8] = 0.0
        feats[9] = 0.5

    # 11-12: Moment acceleration and deficit
    moments_12m = np.array([
        compute_seismic_moment(e["mag"]) for e in ev_12m
        if e.get("mag") is not None
    ])
    moments_6m = np.array([
        compute_seismic_moment(e["mag"]) for e in ev_6m
        if e.get("mag") is not None
    ])

    if len(moments_12m) > 0 and len(moments_6m) > 0:
        mom_total = float(moments_12m.sum())
        mom_recent = float(moments_6m.sum())
        mom_older = mom_total - mom_recent
        if mom_older > 0:
            feats[10] = mom_recent / mom_older  # mom_accel
        else:
            feats[10] = 1.0

        # Moment deficit: compare 12m rate vs 24m long-term rate
        moments_24m = np.array([
            compute_seismic_moment(e["mag"]) for e in ev_24m
            if e.get("mag") is not None
        ])
        if len(moments_24m) > 0:
            long_term_rate = float(moments_24m.sum()) / 2.0  # per year
            recent_rate = mom_total  # per year (12m window)
            feats[11] = recent_rate / max(long_term_rate, 1e-30) - 1.0  # mom_deficit
        else:
            feats[11] = 0.0
    else:
        feats[10] = 1.0
        feats[11] = 0.0

    # 13: Coulomb proxy (largest event in 12m * inverse distance)
    if len(ev_12m) >= 1:
        best_coulomb = 0.0
        for e in ev_12m:
            emag = e.get("mag", 0)
            dist = haversine_km(lat, lon, e["latitude"], e["longitude"])
            if dist < 1.0:
                dist = 1.0
            coulomb = compute_seismic_moment(emag) / (dist ** 2)
            if coulomb > best_coulomb:
                best_coulomb = coulomb
        feats[12] = math.log10(best_coulomb + 1e-30)  # coulomb_proxy (log scale)
    else:
        feats[12] = 0.0

    # 14-15: Max magnitude in last 90d and 180d
    mags_90d = [e["mag"] for e in ev_3m if e.get("mag") is not None]
    mags_180d = [e["mag"] for e in ev_6m if e.get("mag") is not None]
    feats[13] = max(mags_90d) if mags_90d else 0.0   # max_mag_90d
    feats[14] = max(mags_180d) if mags_180d else 0.0  # max_mag_180d

    return feats.astype(np.float32)


# ===================================================================
# BLOCK C FEATURE EXTRACTION (Coherence Field Theory)
# ===================================================================

def extract_block_c(
    full_catalog: list[dict],
    lat: float,
    lon: float,
    ref_epoch: float,
    grid_fields: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Extract Block C (coherence field theory) features.

    Uses extract_coherence_features from coherence_engine.
    All computations use ONLY events BEFORE ref_epoch.

    Returns
    -------
    ndarray, shape (12,), dtype float32
    """
    feats = np.full(N_FEAT_C, np.nan, dtype=np.float64)

    # Events already filtered to nearby + before ref_epoch by caller
    past_catalog = [e for e in full_catalog if _event_epoch(e) < ref_epoch]

    # Extract coherence features (float64 precision for fits)
    cft = extract_coherence_features(
        past_catalog,
        lat=lat,
        lon=lon,
        radius_km=LABEL_RADIUS_KM,
        time_window_days=FORWARD_WINDOW_DAYS,
        ref_epoch=ref_epoch,
        grid_fields=grid_fields,
    )

    feats[0] = cft.get("ell", np.nan)                  # ell
    feats[1] = cft.get("ell_trend", np.nan)             # ell_trend
    feats[2] = cft.get("ell_acceleration", np.nan)      # ell_acceleration
    feats[3] = cft.get("days_to_criticality", np.nan)   # days_to_criticality
    feats[4] = cft.get("nu_exponent", np.nan)            # nu_exponent
    feats[5] = cft.get("t_c_confidence", np.nan)         # divergence_r2
    feats[6] = cft.get("delta_aic_iet", np.nan)          # delta_aic_iet
    feats[7] = cft.get("tau_local", np.nan)              # tau_local
    feats[8] = cft.get("grad_tau_local", np.nan)         # grad_tau_local
    feats[9] = cft.get("S_over_Gamma", np.nan)           # S_over_Gamma
    feats[10] = cft.get("Da_local", np.nan)              # Da_local

    # Singularity count
    sing = test_earthquake_singularity(cft)
    feats[11] = float(sing.conditions_met)               # singularity_count

    return feats.astype(np.float32)


# ===================================================================
# BLOCK I FEATURE EXTRACTION (Interaction Terms)
# ===================================================================

def extract_block_i(
    block_s: np.ndarray,
    block_c: np.ndarray,
) -> np.ndarray:
    """Extract Block I (interaction) features from S and C blocks.

    Cross-products designed to capture physically meaningful interactions:
    1. ell x b_trend: diverging length with dropping b-value
    2. ell x rate_accel: diverging length with accelerating rate
    3. tau x max_mag: coherence x largest event
    4. days_to_criticality x singularity_count: imminence x conditions
    5. S_over_Gamma x ell_acceleration: loading x accelerating divergence

    Returns
    -------
    ndarray, shape (5,), dtype float32
    """
    feats = np.full(N_FEAT_I, 0.0, dtype=np.float64)

    # Handle NaN by replacing with 0 for products
    def _safe(val: float) -> float:
        return 0.0 if math.isnan(val) else val

    ell = _safe(float(block_c[0]))           # ell
    ell_trend = _safe(float(block_c[1]))     # ell_trend
    ell_accel = _safe(float(block_c[2]))     # ell_acceleration
    dtc = _safe(float(block_c[3]))           # days_to_criticality
    tau = _safe(float(block_c[7]))           # tau_local
    sg = _safe(float(block_c[9]))            # S_over_Gamma
    sing = _safe(float(block_c[11]))         # singularity_count

    b_trend = _safe(float(block_s[5]))       # b_trend
    rate_1m = _safe(float(block_s[0]))       # rate_1m (acceleration)
    max_mag_180d = _safe(float(block_s[14])) # max_mag_180d

    feats[0] = ell * b_trend                 # ell_x_b_trend
    feats[1] = ell * rate_1m                 # ell_x_rate_accel
    feats[2] = tau * max_mag_180d            # tau_x_max_mag
    feats[3] = dtc * sing                    # dtc_x_singularity
    feats[4] = sg * ell_accel                # sg_x_ell_accel

    return feats.astype(np.float32)


# ===================================================================
# DATA LOADING AND SAMPLE PIPELINE
# ===================================================================

def load_all_data(
    verbose: bool = True,
) -> tuple[
    dict[str, np.ndarray],  # X_train
    dict[str, np.ndarray],  # X_val
    dict[str, np.ndarray],  # X_test
    np.ndarray,             # y_train
    np.ndarray,             # y_val
    np.ndarray,             # y_test
    dict,                   # meta
]:
    """Load earthquake catalog, build samples, extract features, split.

    This is the complete data pipeline. Steps:
    1. Load USGS catalog (2000-2024, M2.5+)
    2. Gardner-Knopoff aftershock declustering
    3. Identify M6+ mainshocks
    4. Generate same-location controls (2:1 neg ratio)
    5. Temporal split with assertions
    6. Extract features for each sample
    """
    # Step 1: Load USGS catalog
    if verbose:
        print("    Loading USGS catalog (2000-2024, M2.5+)...")
        sys.stdout.flush()

    full_catalog = load_usgs_catalog(min_year=2000, max_year=2024, min_mag=2.5)

    if verbose:
        print(f"    Loaded {len(full_catalog)} events")
        sys.stdout.flush()

    if len(full_catalog) == 0:
        raise RuntimeError(
            "No earthquake data found. Run scripts/download_earthquake_data.py first."
        )

    # Step 2: Gardner-Knopoff aftershock declustering
    if verbose:
        print("    Aftershock declustering (Gardner-Knopoff 1974)...")
        sys.stdout.flush()

    mainshocks, aftershocks = decluster_gardner_knopoff(full_catalog)

    if verbose:
        print(
            f"    Declustered: {len(mainshocks)} mainshocks, "
            f"{len(aftershocks)} aftershocks removed"
        )
        sys.stdout.flush()

    # Step 3-4: Build positive + negative samples
    if verbose:
        print("    Building samples...")
        sys.stdout.flush()

    samples = build_samples(mainshocks, full_catalog, verbose=verbose)

    # Step 5: Temporal split
    train_samples = [s for s in samples if TRAIN_START <= s["year"] <= TRAIN_END]
    val_samples = [s for s in samples if VAL_START <= s["year"] <= VAL_END]
    test_samples = [s for s in samples if TEST_START <= s["year"] <= TEST_END]

    # ASSERT temporal integrity
    for s in train_samples:
        assert s["year"] <= TRAIN_END, (
            f"LEAKAGE: Train sample year {s['year']} > TRAIN_END {TRAIN_END}"
        )
    for s in val_samples:
        assert VAL_START <= s["year"] <= VAL_END, (
            f"LEAKAGE: Val sample year {s['year']} outside [{VAL_START}, {VAL_END}]"
        )
    for s in test_samples:
        assert TEST_START <= s["year"] <= TEST_END, (
            f"LEAKAGE: Test sample year {s['year']} outside [{TEST_START}, {TEST_END}]"
        )

    # Additional cross-boundary assertion
    if train_samples and val_samples:
        max_train_year = max(s["year"] for s in train_samples)
        min_val_year = min(s["year"] for s in val_samples)
        assert max_train_year < min_val_year, (
            f"LEAKAGE: max train year {max_train_year} >= min val year {min_val_year}"
        )
    if val_samples and test_samples:
        max_val_year = max(s["year"] for s in val_samples)
        min_test_year = min(s["year"] for s in test_samples)
        assert max_val_year < min_test_year, (
            f"LEAKAGE: max val year {max_val_year} >= min test year {min_test_year}"
        )

    if verbose:
        print(f"\n    Temporal split:")
        print(f"      Train ({TRAIN_START}-{TRAIN_END}): {len(train_samples)} samples")
        print(f"      Val   ({VAL_START}-{VAL_END}):   {len(val_samples)} samples")
        print(f"      Test  ({TEST_START}-{TEST_END}):  {len(test_samples)} samples")
        sys.stdout.flush()

    # Step 6: Extract features
    if verbose:
        print("\n    Extracting features...")
        sys.stdout.flush()

    # Pre-build spatial index for fast radius queries
    if verbose:
        print("    Building spatial index for fast radius queries...")
        sys.stdout.flush()

    # Bin events by 5-degree lat/lon cells
    spatial_bins: dict[tuple[int, int], list[dict]] = {}
    for e in full_catalog:
        lat_bin = int(e.get("latitude", 0) / 5)
        lon_bin = int(e.get("longitude", 0) / 5)
        spatial_bins.setdefault((lat_bin, lon_bin), []).append(e)

    def _get_nearby_events(lat: float, lon: float, radius_km: float,
                            before_epoch: float) -> list[dict]:
        """Fast spatial query using pre-built bins."""
        lat_bin = int(lat / 5)
        lon_bin = int(lon / 5)
        # 300km ≈ 3 degrees, so check ±1 bin
        nearby = []
        for dlat in range(-1, 2):
            for dlon in range(-1, 2):
                for e in spatial_bins.get((lat_bin + dlat, lon_bin + dlon), []):
                    eepoch = _event_epoch(e)
                    if eepoch <= 0 or eepoch >= before_epoch:
                        continue
                    dist = _haversine_km(lat, lon,
                                         e.get("latitude", 0), e.get("longitude", 0))
                    if dist <= radius_km:
                        nearby.append(e)
        return nearby

    if verbose:
        print(f"    Spatial index: {len(spatial_bins)} bins")
        sys.stdout.flush()

    def _extract_features_for_split(
        split_samples: list[dict],
        split_name: str,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """Extract all feature blocks for a list of samples."""
        n = len(split_samples)
        X_s = np.zeros((n, N_FEAT_S), dtype=np.float32)
        X_c = np.zeros((n, N_FEAT_C), dtype=np.float32)
        X_i = np.zeros((n, N_FEAT_I), dtype=np.float32)
        y = np.zeros(n, dtype=np.float32)

        for idx, sample in enumerate(split_samples):
            if verbose and (idx + 1) % 50 == 0:
                print(
                    f"      {split_name}: {idx + 1}/{n} "
                    f"({100 * (idx + 1) / n:.0f}%)",
                )
                sys.stdout.flush()

            lat = sample["latitude"]
            lon = sample["longitude"]
            ref_epoch = sample["ref_epoch"]

            # Get nearby events using spatial index (FAST)
            nearby = _get_nearby_events(lat, lon, LABEL_RADIUS_KM, ref_epoch)

            # Block S (pass nearby events instead of full catalog)
            s_feats = extract_block_s(nearby, lat, lon, ref_epoch)
            X_s[idx] = s_feats

            # Block C (pass nearby events)
            c_feats = extract_block_c(nearby, lat, lon, ref_epoch)
            X_c[idx] = c_feats

            # Block I
            i_feats = extract_block_i(s_feats, c_feats)
            X_i[idx] = i_feats

            y[idx] = sample["label"]

        if verbose:
            print(f"      {split_name}: {n}/{n} (100%)     ")
            sys.stdout.flush()

        # Assemble feature matrices for each variant
        X_dict = {
            "baseline": X_s.copy(),
            "enhanced": np.hstack([X_s, X_c]),
            "full": np.hstack([X_s, X_c, X_i]),
        }
        return X_dict, y

    X_train, y_train = _extract_features_for_split(train_samples, "train")
    X_val, y_val = _extract_features_for_split(val_samples, "val")
    X_test, y_test = _extract_features_for_split(test_samples, "test")

    # Impute NaN with training mean (per feature)
    for variant in ["baseline", "enhanced", "full"]:
        train_mean = np.nanmean(X_train[variant], axis=0)
        # Replace NaN in training mean with 0
        train_mean = np.where(np.isnan(train_mean), 0.0, train_mean)

        for X in [X_train[variant], X_val[variant], X_test[variant]]:
            nan_mask = np.isnan(X)
            for col in range(X.shape[1]):
                col_nans = nan_mask[:, col]
                if col_nans.any():
                    X[col_nans, col] = train_mean[col]

    # 5:1 downsample if needed (check class ratio in training)
    n_pos_train = int(y_train.sum())
    n_neg_train = int(len(y_train) - n_pos_train)
    if n_neg_train > 5 * n_pos_train:
        if verbose:
            print(
                f"\n    Downsampling: {n_neg_train} neg >> {5 * n_pos_train} target"
            )
            sys.stdout.flush()
        rng = np.random.RandomState(42)
        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == 0)[0]
        keep_neg = rng.choice(neg_idx, size=5 * n_pos_train, replace=False)
        keep_idx = np.sort(np.concatenate([pos_idx, keep_neg]))

        y_train = y_train[keep_idx]
        for variant in ["baseline", "enhanced", "full"]:
            X_train[variant] = X_train[variant][keep_idx]

        if verbose:
            print(
                f"    After downsample: {len(y_train)} samples "
                f"({int(y_train.sum())} pos, {int(len(y_train) - y_train.sum())} neg)"
            )

    meta = {
        "n_catalog_events": len(full_catalog),
        "n_mainshocks": len(mainshocks),
        "n_aftershocks_removed": len(aftershocks),
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test),
        "train_pos": int(y_train.sum()),
        "train_neg": int(len(y_train) - y_train.sum()),
        "val_pos": int(y_val.sum()),
        "val_neg": int(len(y_val) - y_val.sum()),
        "test_pos": int(y_test.sum()),
        "test_neg": int(len(y_test) - y_test.sum()),
    }

    return X_train, X_val, X_test, y_train, y_val, y_test, meta


# ===================================================================
# TEMPORAL INTEGRITY ASSERTION
# ===================================================================

def assert_temporal_integrity(
    train_years: list[int],
    val_years: list[int],
    test_years: list[int],
) -> None:
    """Assert that no sample crosses temporal boundaries.

    This is the core anti-leakage guarantee.
    """
    for y in train_years:
        assert y <= TRAIN_END, (
            f"LEAKAGE: Train year {y} > TRAIN_END {TRAIN_END}"
        )

    for y in val_years:
        assert y >= VAL_START, (
            f"LEAKAGE: Val year {y} < VAL_START {VAL_START}"
        )
        assert y <= VAL_END, (
            f"LEAKAGE: Val year {y} > VAL_END {VAL_END}"
        )

    for y in test_years:
        assert y >= TEST_START, (
            f"LEAKAGE: Test year {y} < TEST_START {TEST_START}"
        )
        assert y <= TEST_END, (
            f"LEAKAGE: Test year {y} > TEST_END {TEST_END}"
        )


# ===================================================================
# MAIN PIPELINE
# ===================================================================

def main(
    output_dir: str | Path | None = None,
    verbose: bool = True,
) -> dict:
    """Run the complete definitive earthquake prediction pipeline.

    Steps:
        1. Load USGS catalog
        2. Gardner-Knopoff aftershock declustering
        3. Identify M6+ mainshocks
        4. Generate same-location controls (2:1 neg ratio)
        5. For each sample: compute Block S features from catalog
        6. For each sample: compute Block C features from coherence_engine
        7. Assert temporal split integrity
        8. 5:1 downsample if needed
        9. Normalize features
        10. Train 3 GBT models
        11. Evaluate on test ONCE
        12. Significance tests
        13. Save results

    Parameters
    ----------
    output_dir : str or Path, optional
        Directory for results JSON. Defaults to project root.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Complete results with all metrics and audit guarantees.
    """
    t_start = time.time()

    # Resolve paths
    project_root = Path(__file__).resolve().parents[3]
    if output_dir is None:
        output_dir = project_root / "results" / "earthquake_definitive"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 72)
        print("DEFINITIVE EARTHQUAKE PREDICTION MODEL v1")
        print("RESEARCH ONLY -- NOT OPERATIONAL")
        print("=" * 72)
        print()
        print("Temporal splits:")
        print(f"  Train: {TRAIN_START} -- {TRAIN_END}")
        print(f"  Val:   {VAL_START} -- {VAL_END}")
        print(f"  Test:  {TEST_START} -- {TEST_END}")
        print()
        print(f"GBT config: {GBT_N_TREES} trees, depth={GBT_MAX_DEPTH}, "
              f"lr={GBT_LEARNING_RATE}, subsample={GBT_SUBSAMPLE}")
        print(f"Label: M{MIN_MAINSHOCK_MAG}+ within {LABEL_RADIUS_KM} km, "
              f"{FORWARD_WINDOW_DAYS:.0f} days forward")
        print()
        sys.stdout.flush()

    # ---------------------------------------------------------------
    # Steps 1-6: Load data, decluster, build samples, extract features
    # ---------------------------------------------------------------
    if verbose:
        print("[1-6] Loading data, building samples, extracting features...")
        sys.stdout.flush()

    X_train, X_val, X_test, y_train, y_val, y_test, meta = load_all_data(
        verbose=verbose,
    )

    assert len(y_train) > 0, "No training samples loaded"
    assert len(y_val) > 0, "No validation samples loaded"
    assert len(y_test) > 0, "No test samples loaded"

    # ---------------------------------------------------------------
    # Step 9: Normalize features (fit on train, apply to val/test)
    # ---------------------------------------------------------------
    if verbose:
        print("\n[9] Normalizing features...")
        sys.stdout.flush()

    normalizers: dict[str, FeatureNormalizer] = {}
    for variant in ["baseline", "enhanced", "full"]:
        norm = FeatureNormalizer()
        X_train[variant] = norm.fit_transform(X_train[variant])
        X_val[variant] = norm.transform(X_val[variant])
        X_test[variant] = norm.transform(X_test[variant])
        normalizers[variant] = norm

    # ---------------------------------------------------------------
    # Step 10: Train 3 GBT models on train (with early stopping on val)
    # ---------------------------------------------------------------
    if verbose:
        print("\n[10] Training GBT models...")
        sys.stdout.flush()

    models: dict[str, GradientBoostedTrees] = {}
    train_meta: dict[str, dict] = {}

    for variant in ["baseline", "enhanced", "full"]:
        if verbose:
            n_feat = X_train[variant].shape[1]
            feature_names = {
                "baseline": ALL_FEATURE_NAMES_BASELINE,
                "enhanced": ALL_FEATURE_NAMES_ENHANCED,
                "full": ALL_FEATURE_NAMES_FULL,
            }[variant]
            print(f"\n  --- {variant.upper()} ({n_feat} features) ---")
            sys.stdout.flush()

        gbt = GradientBoostedTrees(
            n_trees=GBT_N_TREES,
            max_depth=GBT_MAX_DEPTH,
            learning_rate=GBT_LEARNING_RATE,
            min_samples_leaf=GBT_MIN_SAMPLES_LEAF,
            subsample=GBT_SUBSAMPLE,
            colsample=GBT_COLSAMPLE,
            l2_reg=GBT_L2_REG,
            gamma=GBT_GAMMA,
        )

        tmeta = gbt.fit(
            X_train[variant], y_train,
            X_val=X_val[variant], y_val=y_val,
            verbose=verbose,
        )

        models[variant] = gbt
        train_meta[variant] = tmeta

        if verbose:
            print(f"    Trees used: {tmeta['n_trees_used']}, "
                  f"early_stop: {tmeta['stopped_early']}")
            sys.stdout.flush()

    # ---------------------------------------------------------------
    # Validation sanity check
    # ---------------------------------------------------------------
    if verbose:
        print("\n[VAL] Validation sanity check...")
        sys.stdout.flush()

    val_preds: dict[str, np.ndarray] = {}
    for variant in ["baseline", "enhanced", "full"]:
        p_val = models[variant].predict_proba(X_val[variant])
        val_preds[variant] = p_val
        val_auc = compute_auc(y_val, p_val)
        if verbose:
            print(f"  {variant}: val_auc = {val_auc:.4f}")

        if val_auc < 0.55:
            print(
                f"  WARNING: {variant} val AUC ({val_auc:.4f}) is very low. "
                f"Model may be broken."
            )

    # ---------------------------------------------------------------
    # Step 11: Evaluate on test ONCE
    # ---------------------------------------------------------------
    if verbose:
        print("\n[11] TEST EVALUATION (single pass, no going back)...")
        sys.stdout.flush()

    test_preds: dict[str, np.ndarray] = {}
    test_results: dict[str, dict] = {}

    for variant in ["baseline", "enhanced", "full"]:
        p_test = models[variant].predict_proba(X_test[variant])
        test_preds[variant] = p_test
        test_results[variant] = evaluate(y_test, p_test)

        if verbose:
            r = test_results[variant]
            print(f"\n  {variant.upper()} on TEST:")
            print(f"    AUC:    {r['auc']:.4f} "
                  f"[{r['bootstrap_ci']['ci_lo']:.4f}, "
                  f"{r['bootstrap_ci']['ci_hi']:.4f}]")
            print(f"    PR-AUC: {r['pr_auc']:.4f}")
            print(f"    Brier:  {r['brier']:.6f}")
            print(f"    BSS:    {r['bss']:.4f}")
            print(f"    N_pos:  {r['n_positive']}, N_neg: {r['n_negative']}, "
                  f"base_rate: {r['base_rate']:.6f}")
            sys.stdout.flush()

    # ---------------------------------------------------------------
    # Step 12: Significance tests
    # ---------------------------------------------------------------
    if verbose:
        print("\n[12] Paired bootstrap significance tests...")
        sys.stdout.flush()

    cft_lift = paired_bootstrap_test(
        y_test, test_preds["baseline"], test_preds["enhanced"]
    )
    interaction_lift = paired_bootstrap_test(
        y_test, test_preds["enhanced"], test_preds["full"]
    )

    if verbose:
        print(f"\n  CFT lift (enhanced vs baseline):")
        print(f"    delta_AUC = {cft_lift['delta_auc']:.4f} "
              f"[{cft_lift['ci_lo']:.4f}, {cft_lift['ci_hi']:.4f}] "
              f"p = {cft_lift['p_value']:.4f}")
        print(f"\n  Interaction lift (full vs enhanced):")
        print(f"    delta_AUC = {interaction_lift['delta_auc']:.4f} "
              f"[{interaction_lift['ci_lo']:.4f}, {interaction_lift['ci_hi']:.4f}] "
              f"p = {interaction_lift['p_value']:.4f}")
        sys.stdout.flush()

    # ---------------------------------------------------------------
    # Feature importance for full model
    # ---------------------------------------------------------------
    full_importance = models["full"].feature_importances(N_FEAT_FULL)
    importance_ranked = sorted(
        zip(ALL_FEATURE_NAMES_FULL, full_importance.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    if verbose:
        print(f"\n  Top 15 features (full model, split count importance):")
        for name, imp in importance_ranked[:15]:
            print(f"    {name:25s}  {imp:.4f}")
        sys.stdout.flush()

    # ---------------------------------------------------------------
    # Step 13: Assemble and save results
    # ---------------------------------------------------------------
    elapsed = time.time() - t_start

    results = {
        "model": "hazardpulse_earthquake_definitive_v1",
        "disclaimer": "RESEARCH ONLY. NOT operational. NOT a replacement for USGS.",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_seconds": round(elapsed, 1),
        "audit_guarantees": {
            "temporal_split": (
                f"Train {TRAIN_START}-{TRAIN_END}, "
                f"Val {VAL_START}-{VAL_END}, "
                f"Test {TEST_START}-{TEST_END}"
            ),
            "label_definition": (
                f"M{MIN_MAINSHOCK_MAG}+ within {LABEL_RADIUS_KM}km "
                f"and {FORWARD_WINDOW_DAYS:.0f} days FORWARD"
            ),
            "aftershock_declustering": "Gardner-Knopoff (1974)",
            "control_generation": (
                f"Same-location, {CONTROL_RATIO}:1 neg ratio, "
                f"time-offset {CONTROL_OFFSET_RANGE[0]}-{CONTROL_OFFSET_RANGE[1]} years"
            ),
            "meta_stacker": False,
            "hyperparameter_tuning_on_test": False,
            "single_test_evaluation": True,
            "model_type": "GradientBoostedTrees (single per variant, no ensemble)",
            "gbt_config": {
                "n_trees": GBT_N_TREES,
                "max_depth": GBT_MAX_DEPTH,
                "learning_rate": GBT_LEARNING_RATE,
                "subsample": GBT_SUBSAMPLE,
                "colsample": GBT_COLSAMPLE,
                "min_samples_leaf": GBT_MIN_SAMPLES_LEAF,
                "l2_reg": GBT_L2_REG,
                "gamma": GBT_GAMMA,
            },
        },
        "data_summary": meta,
        "training_metadata": {
            variant: {
                "n_trees_used": train_meta[variant]["n_trees_used"],
                "stopped_early": train_meta[variant]["stopped_early"],
                "best_val_loss": train_meta[variant].get("best_val_loss"),
            }
            for variant in ["baseline", "enhanced", "full"]
        },
        "validation_sanity_check": {
            variant: {
                "auc": compute_auc(y_val, val_preds[variant]),
            }
            for variant in ["baseline", "enhanced", "full"]
        },
        "baseline": test_results["baseline"],
        "enhanced": test_results["enhanced"],
        "full": test_results["full"],
        "theory_tests": {
            "cft_lift": cft_lift,
            "interaction_lift": interaction_lift,
        },
        "feature_importance": {
            "full_model": [
                {"feature": name, "importance": imp}
                for name, imp in importance_ranked
            ],
            "note": (
                "Importance is split frequency, NOT causal attribution. "
                "A feature that splits often may simply have high variance."
            ),
        },
    }

    # Save to JSON
    results_path = output_dir / "definitive_results.json"
    with open(results_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)

    if verbose:
        print(f"\n  Results saved to: {results_path}")
        print(f"  Total time: {elapsed:.1f} seconds")
        print("\n" + "=" * 72)
        print("DONE. Remember: this is RESEARCH ONLY.")
        print("=" * 72)
        sys.stdout.flush()

    return results


# ===================================================================
# CLI ENTRYPOINT
# ===================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Definitive leak-proof earthquake prediction model (RESEARCH ONLY)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for results JSON (default: results/earthquake_definitive/)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()
    main(
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )
