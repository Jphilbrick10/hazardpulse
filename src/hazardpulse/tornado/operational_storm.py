# IMPORTANT: This software is for RESEARCH PURPOSES ONLY.
# It is NOT an operational tornado warning system.
# It does NOT replace official NWS tornado warnings.
# Always follow guidance from the National Weather Service (weather.gov).
# False negatives (missed tornadoes) WILL occur. Do NOT rely on this
# system for safety-critical decisions.

"""Operational storm-object tornado prediction model.

Fuses ProbSevere v3 storm-object data, HRRR atmospheric analysis, and
coherence field theory (Helmholtz PDE) into a 4-model ensemble for
60-minute forward tornado prediction.

Architecture
------------
For each ProbSevere storm object at each time step:
  1. Extract Block S (15): storm-object features from ProbSevere
  2. Extract Block E (8):  storm evolution features (tracking history)
  3. Extract Block A (10): atmospheric context from HRRR at storm location
  4. Extract Block T (15): coherence field theory at storm location
  5. Combine blocks -> logistic + GBT -> meta-stacker -> calibration

Four competing models trained simultaneously:
  1. Storm-only:       S + E       (23 features)
  2. Storm+Atm:        S + E + A   (33 features)
  3. Full (coherence): S + E + A + T (48 features)  [primary]
  4. Gaussian control:  S + E + A + G (48 features)

Pure NumPy.  Zero external ML dependencies.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field as dc_field
from pathlib import Path

import numpy as np

from hazardpulse.data.hrrr import (
    HRRR_N_LAT,
    HRRR_N_LON,
    latlon_to_hrrr_cell,
)
from hazardpulse.tornado.coherence_engine import (
    ROTATION_THRESHOLD,
    compute_coherence_fields,
    compute_derived_hrrr,
    compute_gaussian_fields,
)

# ---------------------------------------------------------------------------
# Feature block sizes
# ---------------------------------------------------------------------------

N_FEAT_S: int = 15  # Storm-object ProbSevere
N_FEAT_E: int = 8   # Storm evolution
N_FEAT_A: int = 10  # Atmospheric context (HRRR)
N_FEAT_T: int = 15  # Coherence theory at storm
N_FEAT_G: int = 15  # Gaussian control (mirrors T)

N_FEAT_STORM_ONLY: int = N_FEAT_S + N_FEAT_E          # 23
N_FEAT_STORM_ATM: int = N_FEAT_S + N_FEAT_E + N_FEAT_A  # 33
N_FEAT_FULL: int = N_FEAT_S + N_FEAT_E + N_FEAT_A + N_FEAT_T  # 48
N_FEAT_GAUSS: int = N_FEAT_S + N_FEAT_E + N_FEAT_A + N_FEAT_G  # 48

# ---------------------------------------------------------------------------
# Feature names
# ---------------------------------------------------------------------------

BLOCK_S_NAMES: list[str] = [
    "mucape", "mlcape", "mlcin", "ebshear", "srh01",
    "mesh", "vil_density", "flash_rate", "flash_density",
    "maxllaz", "p98llaz", "p98mlaz", "lja", "ps", "size",
]

BLOCK_E_NAMES: list[str] = [
    "storm_age_minutes", "maxllaz_trend", "flash_rate_trend",
    "ps_trend", "mesh_trend", "storm_speed",
    "storm_has_rotation", "sustained_rotation_minutes",
]

BLOCK_A_NAMES: list[str] = [
    "hrrr_cape", "hrrr_srh01", "hrrr_shear06", "hrrr_shear01",
    "hrrr_lcl", "hrrr_rfd_warmth", "hrrr_stp",
    "hrrr_srh05_est", "hrrr_streamwise_vort", "hrrr_pwat",
]

BLOCK_T_NAMES: list[str] = [
    "tau_at_storm", "grad_tau_at_storm", "dtau_dt_at_storm",
    "torsion_at_storm", "alignment_at_storm",
    "S_at_storm", "Gamma_at_storm", "S_over_Gamma",
    "Da_at_storm", "E_coh_at_storm", "singularity_count",
    "tau_x_maxllaz", "torsion_x_srh", "alignment_x_cape",
    "E_coh_x_flash_rate",
]

BLOCK_G_NAMES: list[str] = ["G_" + n for n in BLOCK_T_NAMES]

ALL_NAMES_FULL: list[str] = (
    BLOCK_S_NAMES + BLOCK_E_NAMES + BLOCK_A_NAMES + BLOCK_T_NAMES
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TornadoStormConfig:
    """Hyperparameters for storm-object tornado prediction."""

    # GBT
    gbt_n_trees: int = 100
    gbt_max_depth: int = 3
    gbt_learning_rate: float = 0.05
    gbt_subsample: float = 0.4
    gbt_colsample: float = 0.8
    gbt_min_leaf: int = 20
    gbt_l2_reg: float = 1.0
    gbt_gamma: float = 0.0

    # Logistic
    lr_epochs: int = 2000
    lr_lr: float = 0.01
    lr_lam: float = 1.0
    lr_batch: int = 2048

    # Meta-stacker
    stack_lr: float = 0.01
    stack_lam: float = 0.1
    stack_epochs: int = 1000

    # Sampling
    neg_ratio: int = 5
    max_samples: int = 80000
    max_steps_per_storm: int = 10

    # Labeling
    label_radius_km: float = 40.0
    forward_window_min: int = 60


# ---------------------------------------------------------------------------
# ML primitives  (pure NumPy)
# ---------------------------------------------------------------------------


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    z = np.clip(z, -500, 500)
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z)),
    )


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC via trapezoidal integration."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    if len(y_true) < 2:
        return 0.5
    order = np.argsort(-y_score)
    y_true = y_true[order]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tpr_prev = fpr_prev = 0.0
    tp = fp = 0
    auc = 0.0
    for i in range(len(y_true)):
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr = tp / n_pos
        fpr = fp / n_neg
        auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
        tpr_prev = tpr
        fpr_prev = fpr
    return float(auc)


def logistic_train(
    X: np.ndarray,
    y: np.ndarray,
    *,
    lr: float = 0.01,
    lam: float = 1.0,
    epochs: int = 500,
    class_weight: str | None = None,
    batch_size: int = 2048,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Mini-batch Adam logistic regression.  All float32, memory-safe.

    Returns (weights, bias, feature_means, feature_stds).
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    mu = X.mean(axis=0, dtype=np.float32)
    sd = X.std(axis=0, dtype=np.float32)
    sd[sd < 1e-12] = 1.0
    X = (X - mu) / sd
    N, D = X.shape
    w = np.zeros(D, dtype=np.float32)
    b = np.float32(0.0)

    sample_w = np.ones(N, dtype=np.float32)
    if class_weight == "balanced":
        n_pos = float(y.sum())
        n_neg = N - n_pos
        if n_pos > 0 and n_neg > 0:
            sample_w = np.where(
                y == 1, N / (2 * n_pos), N / (2 * n_neg)
            ).astype(np.float32)

    m_w = np.zeros(D, dtype=np.float32)
    v_w = np.zeros(D, dtype=np.float32)
    m_b = np.float32(0.0)
    v_b = np.float32(0.0)
    beta1 = np.float32(0.9)
    beta2 = np.float32(0.999)
    eps = np.float32(1e-8)

    rng = np.random.RandomState(42)
    bs = min(batch_size, N)
    for epoch in range(epochs):
        idx = rng.choice(N, size=bs, replace=False)
        Xb = X[idx]
        yb = y[idx]
        wb = sample_w[idx]
        z = (Xb @ w + b).astype(np.float32)
        p = sigmoid(z).astype(np.float32)
        res = (p - yb) * wb
        dw = (Xb.T @ res).astype(np.float32) / bs + (lam / N) * w
        db = np.float32(res.mean())
        m_w = beta1 * m_w + (1 - beta1) * dw
        v_w = beta2 * v_w + (1 - beta2) * dw ** 2
        m_b = beta1 * m_b + (1 - beta1) * db
        v_b = beta2 * v_b + (1 - beta2) * db ** 2
        t = epoch + 1
        w -= lr * (m_w / (1 - beta1 ** t)) / (
            np.sqrt(v_w / (1 - beta2 ** t)) + eps
        )
        b -= lr * (m_b / (1 - beta1 ** t)) / (
            np.sqrt(v_b / (1 - beta2 ** t)) + eps
        )
    return w, float(b), mu, sd


def logistic_predict(
    X: np.ndarray,
    w: np.ndarray,
    b: float,
    mu: np.ndarray,
    sd: np.ndarray,
) -> np.ndarray:
    """Memory-safe chunked logistic prediction."""
    X = np.asarray(X, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)
    mu = np.asarray(mu, dtype=np.float32)
    sd = np.asarray(sd, dtype=np.float32)
    N = X.shape[0]
    result = np.empty(N, dtype=np.float32)
    chunk = 4096
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        Xc = (X[start:end] - mu) / sd
        result[start:end] = sigmoid(Xc @ w + b).astype(np.float32)
    return result


# ---------------------------------------------------------------------------
# Gradient Boosted Trees
# ---------------------------------------------------------------------------


class GradientBoostedTrees:
    """Memory-efficient GBT with subsample, colsample, and balanced weights."""

    def __init__(
        self,
        n_trees: int = 100,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        min_samples_leaf: int = 20,
        subsample: float = 0.4,
        colsample: float = 0.8,
        l2_reg: float = 1.0,
        gamma: float = 0.0,
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
        residuals: np.ndarray,
        sample_w: np.ndarray,
        depth: int = 0,
    ) -> dict:
        N = len(residuals)
        G = float(np.sum(residuals * sample_w))
        H = float(
            np.sum(
                sample_w * (1.0 - np.abs(residuals)) * np.abs(residuals)
                + 1e-6
            )
        )
        leaf_val = G / (H + self.l2_reg)
        if depth >= self.max_depth or N < 2 * self.min_leaf:
            return {"leaf": True, "val": float(leaf_val)}

        D = X.shape[1]
        n_try = max(5, int(D * self.colsample))
        feat_idx = np.random.choice(D, size=min(n_try, D), replace=False)
        best_gain = -1e30
        best_feat = 0
        best_thresh = 0.0

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
                GL = float(np.sum(residuals[left_mask] * sample_w[left_mask]))
                GR = float(np.sum(residuals[right_mask] * sample_w[right_mask]))
                HL = float(np.sum(sample_w[left_mask])) + 1e-6
                HR = float(np.sum(sample_w[right_mask])) + 1e-6
                gain = (
                    GL ** 2 / (HL + self.l2_reg)
                    + GR ** 2 / (HR + self.l2_reg)
                    - G ** 2 / (H + self.l2_reg)
                    - self.gamma
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feat = int(f)
                    best_thresh = float(thresh)

        if best_gain <= 0:
            return {"leaf": True, "val": float(leaf_val)}

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask
        left_child = self._build_tree(
            X[left_mask], residuals[left_mask], sample_w[left_mask], depth + 1
        )
        right_child = self._build_tree(
            X[right_mask], residuals[right_mask], sample_w[right_mask], depth + 1
        )
        return {
            "leaf": False,
            "feat": best_feat,
            "thresh": best_thresh,
            "left": left_child,
            "right": right_child,
        }

    def _predict_tree_batch(self, node: dict, X: np.ndarray) -> np.ndarray:
        result = np.empty(X.shape[0], dtype=np.float32)
        for i in range(X.shape[0]):
            n = node
            while not n["leaf"]:
                if X[i, n["feat"]] <= n["thresh"]:
                    n = n["left"]
                else:
                    n = n["right"]
            result[i] = n["val"]
        return result

    def fit(self, X: np.ndarray, y: np.ndarray, *, verbose: bool = False) -> None:
        """Train the GBT ensemble on labelled data."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        N = len(y)
        n_pos = float(y.sum())
        n_neg = N - n_pos
        sample_w = np.where(
            y == 1,
            N / (2 * n_pos + 1e-12),
            N / (2 * n_neg + 1e-12),
        ).astype(np.float32)
        p_avg = float(np.clip(y.mean(), 0.01, 0.99))
        self.init_pred = math.log(p_avg / (1 - p_avg))
        F = np.full(N, self.init_pred, dtype=np.float32)
        for i in range(self.n_trees):
            p = sigmoid(F).astype(np.float32)
            residuals = (y - p).astype(np.float32)
            if self.subsample < 1.0:
                idx = np.random.choice(
                    N, size=int(N * self.subsample), replace=False
                )
                tree = self._build_tree(X[idx], residuals[idx], sample_w[idx])
            else:
                tree = self._build_tree(X, residuals, sample_w)
            self.trees.append(tree)
            F += self.lr * self._predict_tree_batch(tree, X)
            if verbose and (i + 1) % 25 == 0:
                auc = compute_auc(y, sigmoid(F))
                print(f"      Tree {i + 1}/{self.n_trees}: train_auc={auc:.4f}")
                sys.stdout.flush()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for new data."""
        X = np.asarray(X, dtype=np.float32)
        F = np.full(X.shape[0], self.init_pred, dtype=np.float32)
        for tree in self.trees:
            F += self.lr * self._predict_tree_batch(tree, X)
        return sigmoid(F)

    def feature_importances(self, n_features: int) -> np.ndarray:
        """Compute feature importance from tree split counts."""
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


# ---------------------------------------------------------------------------
# Platt calibration
# ---------------------------------------------------------------------------


def _platt_calibrate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lr: float = 0.01,
    epochs: int = 500,
) -> tuple[float, float]:
    """Fit Platt sigmoid calibration: P_cal = sigmoid(a * y_pred + b).

    Uses gradient descent to find optimal (a, b) on validation predictions.
    Returns (a, b) parameters.
    """
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    a = np.float32(1.0)
    b = np.float32(0.0)
    N = len(y_true)
    if N == 0:
        return float(a), float(b)
    for _ in range(epochs):
        z = a * y_pred + b
        p = sigmoid(z).astype(np.float32)
        grad_a = float(np.sum((p - y_true) * y_pred)) / N
        grad_b = float(np.sum(p - y_true)) / N
        a -= np.float32(lr * grad_a)
        b -= np.float32(lr * grad_b)
    return float(a), float(b)


# ---------------------------------------------------------------------------
# Meta-stacker
# ---------------------------------------------------------------------------


class MetaStacker:
    """Combines LR + GBT predictions via a learned logistic stacker."""

    def __init__(self) -> None:
        self.w: np.ndarray | None = None
        self.b: float = 0.0
        self.mu: np.ndarray | None = None
        self.sd: np.ndarray | None = None

    def fit(self, preds_list: list[np.ndarray], y: np.ndarray) -> None:
        """Fit the meta-stacker on base-model predictions."""
        X = np.column_stack(preds_list).astype(np.float32)
        self.w, self.b, self.mu, self.sd = logistic_train(
            X, y, lr=0.01, lam=0.1, epochs=1000, class_weight="balanced"
        )

    def predict_proba(self, preds_list: list[np.ndarray]) -> np.ndarray:
        """Predict with the meta-stacker."""
        X = np.column_stack(preds_list).astype(np.float32)
        return logistic_predict(X, self.w, self.b, self.mu, self.sd)


# ---------------------------------------------------------------------------
# Feature extraction per storm object
# ---------------------------------------------------------------------------


def extract_block_s(storm: dict) -> np.ndarray:
    """Extract Block S: storm-object ProbSevere features (15)."""
    feat = np.zeros(N_FEAT_S, dtype=np.float32)
    keys = [
        "mucape", "mlcape", "mlcin", "ebshear", "srh01",
        "mesh", "vil_density", "flash_rate", "flash_density",
        "maxllaz", "p98llaz", "p98mlaz", "lja", "ps", "size",
    ]
    for idx, key in enumerate(keys):
        feat[idx] = float(storm.get(key, 0))
    return feat


def extract_block_e(storm: dict, storm_history: list[dict]) -> np.ndarray:
    """Extract Block E: storm evolution features (8)."""
    feat = np.zeros(N_FEAT_E, dtype=np.float32)
    n_steps = len(storm_history)
    if n_steps < 2:
        feat[6] = float(float(storm.get("maxllaz", 0)) > ROTATION_THRESHOLD)
        return feat

    feat[0] = np.float32((n_steps - 1) * 15.0)  # storm_age_minutes

    current = storm_history[-1]
    prev = storm_history[-2]

    # Trends: ratio or difference
    for idx, key, mode in [
        (1, "maxllaz", "ratio"),
        (2, "flash_rate", "ratio"),
        (3, "ps", "diff"),
        (4, "mesh", "ratio"),
    ]:
        cur_val = float(current.get(key, 0))
        prev_val = float(prev.get(key, 0))
        if mode == "ratio":
            threshold = 1e-6 if key != "flash_rate" else 0.1
            if prev_val > threshold:
                feat[idx] = np.float32(cur_val / prev_val)
            else:
                feat[idx] = np.float32(1.0 if cur_val <= threshold else 2.0)
        else:
            feat[idx] = np.float32(cur_val - prev_val)

    # Storm speed from motion vectors
    me = float(current.get("motion_east", 0))
    ms = float(current.get("motion_south", 0))
    feat[5] = np.float32(math.sqrt(me ** 2 + ms ** 2))

    # Rotation
    cur_maxllaz = float(current.get("maxllaz", 0))
    feat[6] = np.float32(cur_maxllaz > ROTATION_THRESHOLD)

    # Sustained rotation: consecutive time steps with rotation
    sustained = 0
    for s in reversed(storm_history):
        if float(s.get("maxllaz", 0)) > ROTATION_THRESHOLD:
            sustained += 1
        else:
            break
    feat[7] = np.float32(sustained * 15.0)

    return feat


def extract_block_a(
    lat: float,
    lon: float,
    hrrr: dict[str, np.ndarray] | None,
    derived: dict[str, np.ndarray] | None,
) -> np.ndarray:
    """Extract Block A: atmospheric context from HRRR (10 features)."""
    feat = np.zeros(N_FEAT_A, dtype=np.float32)
    if hrrr is None or derived is None:
        return feat

    i, j = latlon_to_hrrr_cell(lat, lon)

    feat[0] = hrrr["mlcape"][i, j]
    feat[1] = hrrr["srh_01"][i, j]
    feat[2] = derived["shear_06"][i, j]
    feat[3] = derived["shear_01"][i, j]
    feat[4] = derived["lcl_est"][i, j]
    feat[5] = derived["rfd_warmth"][i, j]
    feat[6] = derived["stp_eff"][i, j]
    feat[7] = derived["srh_05_est"][i, j]
    feat[8] = derived["streamwise_vort"][i, j]
    feat[9] = hrrr["pwat"][i, j]

    return feat


def extract_block_a_from_probsevere(storm: dict) -> np.ndarray:
    """Fallback: atmospheric context from ProbSevere when HRRR unavailable."""
    feat = np.zeros(N_FEAT_A, dtype=np.float32)
    feat[0] = float(storm.get("mlcape", 0))
    feat[1] = float(storm.get("srh01", 0))
    feat[2] = float(storm.get("ebshear", 0))
    feat[3] = float(storm.get("ebshear", 0)) * 0.5
    return feat


def extract_block_t(
    lat: float,
    lon: float,
    storm: dict,
    coh_fields: dict[str, np.ndarray] | None,
) -> np.ndarray:
    """Extract Block T: coherence theory at storm location (15 features)."""
    feat = np.zeros(N_FEAT_T, dtype=np.float32)
    if coh_fields is None:
        return feat

    i, j = latlon_to_hrrr_cell(lat, lon)

    tau_val = coh_fields["tau"][i, j]
    grad_val = coh_fields["grad_tau"][i, j]
    torsion_val = coh_fields["torsion"][i, j]
    alignment_val = coh_fields["alignment"][i, j]
    S_val = coh_fields["S_field"][i, j]
    Gamma_val = coh_fields["Gamma_field"][i, j]
    SG_val = coh_fields["S_over_Gamma"][i, j]
    Da_val = coh_fields["Da"][i, j]
    E_coh_val = coh_fields["E_coh"][i, j]
    sing_val = coh_fields["singularity_count"][i, j]

    feat[0] = tau_val
    feat[1] = grad_val
    feat[2] = np.float32(0.0)  # dtau_dt requires multi-hour data
    feat[3] = torsion_val
    feat[4] = alignment_val
    feat[5] = S_val
    feat[6] = Gamma_val
    feat[7] = SG_val
    feat[8] = Da_val
    feat[9] = E_coh_val
    feat[10] = sing_val

    # Cross-terms: coherence x storm-object interactions
    maxllaz = float(storm.get("maxllaz", 0))
    srh = float(storm.get("srh01", 0))
    cape = float(storm.get("mucape", 0))
    flash_rate = float(storm.get("flash_rate", 0))

    feat[11] = np.float32(tau_val * maxllaz)
    feat[12] = np.float32(torsion_val * srh)
    feat[13] = np.float32(alignment_val * cape / 1000.0)
    feat[14] = np.float32(E_coh_val * flash_rate)

    return feat


# ---------------------------------------------------------------------------
# Build all feature blocks for a storm
# ---------------------------------------------------------------------------


def build_storm_features(
    storm: dict,
    storm_history: list[dict],
    hrrr: dict[str, np.ndarray] | None,
    coherence_fields: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build feature vectors for a storm object.

    Parameters
    ----------
    storm : dict
        Current ProbSevere storm-object dict.
    storm_history : list[dict]
        Ordered history of this storm's previous time steps.
    hrrr : dict or None
        HRRR grid data (raw variables).
    coherence_fields : dict or None
        Output of ``compute_coherence_fields``.

    Returns
    -------
    tuple of ndarrays
        ``(block_s, block_e, block_a, block_t)`` feature arrays.
    """
    block_s = extract_block_s(storm)
    block_e = extract_block_e(storm, storm_history)

    lat = float(storm.get("lat", 0))
    lon = float(storm.get("lon", 0))

    derived = None
    if coherence_fields is not None:
        derived = coherence_fields.get("_derived")

    if hrrr is not None and derived is not None:
        block_a = extract_block_a(lat, lon, hrrr, derived)
    else:
        block_a = extract_block_a_from_probsevere(storm)

    block_t = extract_block_t(lat, lon, storm, coherence_fields)

    return block_s, block_e, block_a, block_t


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_operational_tornado_model(
    train_data: list[dict],
    val_data: list[dict],
    config: TornadoStormConfig | None = None,
) -> dict:
    """Train the full 4-model ensemble.

    Parameters
    ----------
    train_data : list[dict]
        Training samples. Each dict has keys ``S``, ``E``, ``A``, ``T``
        (numpy arrays) and ``label`` (int).
    val_data : list[dict]
        Validation samples (same format).
    config : TornadoStormConfig, optional
        Hyperparameter configuration.

    Returns
    -------
    dict
        Trained model containing LR weights, GBT trees, and MetaStacker.
    """
    if config is None:
        config = TornadoStormConfig()

    # Assemble feature matrices
    X_train = np.column_stack([
        np.vstack([s["S"] for s in train_data]),
        np.vstack([s["E"] for s in train_data]),
        np.vstack([s["A"] for s in train_data]),
        np.vstack([s["T"] for s in train_data]),
    ]).astype(np.float32)
    y_train = np.array([s["label"] for s in train_data], dtype=np.float32)

    X_val = np.column_stack([
        np.vstack([s["S"] for s in val_data]),
        np.vstack([s["E"] for s in val_data]),
        np.vstack([s["A"] for s in val_data]),
        np.vstack([s["T"] for s in val_data]),
    ]).astype(np.float32)
    y_val = np.array([s["label"] for s in val_data], dtype=np.float32)

    n_features = X_train.shape[1]

    # Train logistic regression
    print("  Training logistic regression...")
    sys.stdout.flush()
    lr_w, lr_b, lr_mu, lr_sd = logistic_train(
        X_train, y_train,
        lr=config.lr_lr,
        lam=config.lr_lam,
        epochs=config.lr_epochs,
        class_weight="balanced",
        batch_size=config.lr_batch,
    )

    # Train GBT
    print("  Training gradient boosted trees...")
    sys.stdout.flush()
    gbt = GradientBoostedTrees(
        n_trees=config.gbt_n_trees,
        max_depth=config.gbt_max_depth,
        learning_rate=config.gbt_learning_rate,
        min_samples_leaf=config.gbt_min_leaf,
        subsample=config.gbt_subsample,
        colsample=config.gbt_colsample,
        l2_reg=config.gbt_l2_reg,
        gamma=config.gbt_gamma,
    )
    gbt.fit(X_train, y_train, verbose=True)

    # Meta-stacker with out-of-fold predictions to prevent data leakage
    print("  Training meta-stacker (out-of-fold)...")
    sys.stdout.flush()
    n = len(y_train)
    folds = 3
    fold_size = n // folds
    oof_lr = np.zeros(n, dtype=np.float32)
    oof_gbt = np.zeros(n, dtype=np.float32)

    for fold in range(folds):
        val_start = fold * fold_size
        val_end = (fold + 1) * fold_size if fold < folds - 1 else n
        val_idx = np.arange(val_start, val_end)
        train_idx = np.concatenate([np.arange(0, val_start), np.arange(val_end, n)])

        # Train LR on fold
        w_f, b_f, mu_f, sd_f = logistic_train(
            X_train[train_idx], y_train[train_idx],
            lr=config.lr_lr, lam=config.lr_lam,
            epochs=config.lr_epochs // 2,  # faster for folds
            class_weight="balanced",
            batch_size=config.lr_batch,
        )
        oof_lr[val_idx] = logistic_predict(X_train[val_idx], w_f, b_f, mu_f, sd_f)

        # Train GBT on fold
        gbt_f = GradientBoostedTrees(
            n_trees=config.gbt_n_trees // 2,  # fewer trees for speed
            max_depth=config.gbt_max_depth,
            learning_rate=config.gbt_learning_rate,
            min_samples_leaf=config.gbt_min_leaf,
            subsample=config.gbt_subsample,
            colsample=config.gbt_colsample,
            l2_reg=config.gbt_l2_reg,
            gamma=config.gbt_gamma,
        )
        gbt_f.fit(X_train[train_idx], y_train[train_idx])
        oof_gbt[val_idx] = gbt_f.predict_proba(X_train[val_idx])

    stacker = MetaStacker()
    stacker.fit([oof_lr, oof_gbt], y_train)

    # Validation metrics
    p_lr_val = logistic_predict(X_val, lr_w, lr_b, lr_mu, lr_sd)
    p_gbt_val = gbt.predict_proba(X_val)
    p_stack_val = stacker.predict_proba([p_lr_val, p_gbt_val])

    # Platt calibration on validation set
    platt_a, platt_b = _platt_calibrate(y_val, p_stack_val)

    p_calibrated_val = sigmoid(platt_a * p_stack_val + platt_b)
    val_auc = compute_auc(y_val, p_calibrated_val)
    print(f"  Validation AUC: {val_auc:.4f}")
    sys.stdout.flush()

    # Feature importances
    importances = gbt.feature_importances(n_features)

    model = {
        "lr_w": lr_w,
        "lr_b": lr_b,
        "lr_mu": lr_mu,
        "lr_sd": lr_sd,
        "gbt": gbt,
        "stacker": stacker,
        "platt_a": platt_a,
        "platt_b": platt_b,
        "n_features": n_features,
        "feature_names": ALL_NAMES_FULL[:n_features],
        "importances": importances,
        "val_auc": val_auc,
        "config": config,
        "model_version": "tornado_storm_v1_0",
    }
    return model


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def _risk_band(prob: float) -> str:
    """Map probability to risk band.

    Band names deliberately avoid NWS terminology (e.g. "watch", "warning")
    to prevent confusion with official NWS products.
    """
    if prob >= 0.50:
        return "very_high"
    if prob >= 0.30:
        return "high"
    if prob >= 0.15:
        return "moderate"
    if prob >= 0.05:
        return "low"
    return "minimal"


def predict_tornado_probability(
    storm: dict,
    storm_history: list[dict],
    hrrr: dict[str, np.ndarray] | None,
    coherence_fields: dict[str, np.ndarray] | None,
    model: dict,
) -> dict:
    # TODO: model deserialization from JSON requires from_dict() methods on
    # GradientBoostedTrees, MetaStacker, and TornadoStormConfig. Currently
    # masked because no pre-trained model file exists yet. When a model file
    # is saved, the dict keys (e.g. model["gbt_full"], model["meta_stacker"])
    # must be deserialized into their respective dataclass/object instances
    # before use. Without from_dict(), this function will raise on attribute
    # access (e.g. .predict()) against raw dicts.
    """Predict tornado probability for a single storm.

    Parameters
    ----------
    storm : dict
        Current ProbSevere storm-object dict.
    storm_history : list[dict]
        Ordered history of previous time steps for this storm.
    hrrr : dict or None
        HRRR grid data.
    coherence_fields : dict or None
        Output of ``compute_coherence_fields``.
    model : dict
        Trained model from ``train_operational_tornado_model``.

    Returns
    -------
    dict
        ``probability``, ``risk_band``, ``top_features``,
        ``coherence_score``, ``model_scores``.
    """
    block_s, block_e, block_a, block_t = build_storm_features(
        storm, storm_history, hrrr, coherence_fields
    )

    X = np.concatenate([block_s, block_e, block_a, block_t]).reshape(1, -1)
    X = X.astype(np.float32)

    # LR prediction
    p_lr = logistic_predict(
        X, model["lr_w"], model["lr_b"], model["lr_mu"], model["lr_sd"]
    )

    # GBT prediction
    p_gbt = model["gbt"].predict_proba(X)

    # Meta-stacker
    p_stack = model["stacker"].predict_proba([p_lr, p_gbt])

    # Platt calibration (if available)
    platt_a = model.get("platt_a", 1.0)
    platt_b = model.get("platt_b", 0.0)
    p_calibrated = sigmoid(np.array([platt_a * p_stack[0] + platt_b], dtype=np.float32))
    prob = float(p_calibrated[0])

    # Top features
    importances = model.get("importances", np.zeros(X.shape[1]))
    feature_names = model.get("feature_names", ALL_NAMES_FULL)
    top_idx = np.argsort(-importances)[:10]
    top_features = [
        {"name": feature_names[i], "importance": round(float(importances[i]), 4)}
        for i in top_idx
        if importances[i] > 0
    ]

    # Coherence score: weighted combination of theory features
    coherence_score = float(
        0.3 * block_t[4]   # alignment
        + 0.2 * block_t[0]   # tau
        + 0.2 * block_t[7]   # S/Gamma
        + 0.15 * block_t[10]  # singularity count / 5
        + 0.15 * block_t[3]   # torsion
    )

    return {
        "probability": round(prob, 4),
        "risk_band": _risk_band(prob),
        "top_features": top_features,
        "coherence_score": round(coherence_score, 4),
        "model_scores": {
            "logistic": round(float(p_lr[0]), 4),
            "gbt": round(float(p_gbt[0]), 4),
            "stacked": round(prob, 4),
        },
    }
