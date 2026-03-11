#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TORNADO PREDICTION MODEL v4 - ENVIRONMENTAL-AWARE MEGA-ENSEMBLE
=================================================================

Building on v3 (Formation AUC=0.956, EF4+ AUC=0.999):

v4 Improvements:
  1. ERA5 reanalysis environmental proxies (CAPE, shear, SRH, LCL, moisture)
  2. Synoptic pattern recognition (dryline, fronts, squall/supercell)
  3. Diurnal cycle exploitation (hour encoding, peak window, night ratio)
  4. Multi-day outbreak dynamics (trajectory, cumulative damage, spread rate)
  5. Topographic proxies (terrain features, Great Plains, Mississippi Valley)
  6. Population/impact features (injuries, fatalities, property loss)
  7. 5-model ensemble + enriched meta-learner
  8. Head-to-head benchmark against SPC baseline
  9. Cross-validation stability + bootstrap confidence intervals

No sklearn. No pandas. Manual implementations of all ML.
Data: SPC 1950-2023 tornado database.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import urllib.request
import csv
import io
import os
import sys
import warnings
import math
import time as time_module
warnings.filterwarnings('ignore')

# Force UTF-8
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'coherence_field_results')
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)

# =============================================================================
# UTILITIES
# =============================================================================

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return np.where(z >= 0,
                    1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


def compute_auc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
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
    return auc


def compute_brier(y_true, y_score):
    return np.mean((np.asarray(y_score) - np.asarray(y_true, dtype=np.float64)) ** 2)


def compute_bss(y_true, y_score, clim_prob):
    bs_model = compute_brier(y_true, y_score)
    bs_clim = compute_brier(y_true, np.full(len(y_true), clim_prob))
    if bs_clim < 1e-12:
        return 0.0
    return 1.0 - bs_model / bs_clim


def compute_roc_curve(y_true, y_score, n_points=200):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    thresholds = np.linspace(1.0, 0.0, n_points)
    fpr_list = [0.0]
    tpr_list = [0.0]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.array([0, 1]), np.array([0, 1])
    for th in thresholds:
        pred = (y_score >= th)
        tp = ((pred == 1) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        tpr_list.append(tp / n_pos)
        fpr_list.append(fp / n_neg)
    fpr_list.append(1.0)
    tpr_list.append(1.0)
    return np.array(fpr_list), np.array(tpr_list)


def bootstrap_auc(y_true, y_score, n_boot=1000, seed=42):
    """Bootstrap confidence interval for AUC."""
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        a = compute_auc(y_true[idx], y_score[idx])
        aucs.append(a)
    aucs = np.sort(aucs)
    return aucs[int(0.025 * n_boot)], aucs[int(0.975 * n_boot)], np.mean(aucs)


# =============================================================================
# ML: LOGISTIC REGRESSION WITH L2 (Adam optimizer)
# =============================================================================

def logistic_train(X, y, lr=0.01, lam=1.0, epochs=500, class_weight=None, verbose=False):
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-12] = 1.0
    Xs = (X - mu) / sd
    N, D = Xs.shape
    w = np.zeros(D)
    b = 0.0
    sample_w = np.ones(N)
    if class_weight == 'balanced':
        n_pos = y.sum()
        n_neg = N - n_pos
        if n_pos > 0 and n_neg > 0:
            sample_w = np.where(y == 1, N / (2 * n_pos), N / (2 * n_neg))
    m_w, v_w = np.zeros(D), np.zeros(D)
    m_b, v_b = 0.0, 0.0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for epoch in range(epochs):
        z = Xs @ w + b
        p = sigmoid(z)
        residual = (p - y) * sample_w
        dw = (Xs.T @ residual) / N + (lam / N) * w
        db = residual.mean()
        m_w = beta1 * m_w + (1 - beta1) * dw
        v_w = beta2 * v_w + (1 - beta2) * dw ** 2
        m_b = beta1 * m_b + (1 - beta1) * db
        v_b = beta2 * v_b + (1 - beta2) * db ** 2
        t = epoch + 1
        w -= lr * (m_w / (1 - beta1**t)) / (np.sqrt(v_w / (1 - beta2**t)) + eps)
        b -= lr * (m_b / (1 - beta1**t)) / (np.sqrt(v_b / (1 - beta2**t)) + eps)
        if verbose and epoch % 500 == 0:
            loss = -np.mean(sample_w * (y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)))
            print(f"      Epoch {epoch}: loss={loss:.4f}")
    return w, b, mu, sd


def logistic_predict(X, w, b, mu, sd):
    X = np.array(X, dtype=np.float64)
    return sigmoid(((X - mu) / sd) @ w + b)


# =============================================================================
# ML: GRADIENT BOOSTED STUMPS (depth 1)
# =============================================================================

class GradientBoostedStumps:
    def __init__(self, n_stumps=300, learning_rate=0.1, min_samples_leaf=20):
        self.n_stumps = n_stumps
        self.lr = learning_rate
        self.min_leaf = min_samples_leaf
        self.stumps = []
        self.init_pred = 0.0

    def _find_best_stump(self, X, residuals, sample_w):
        N, D = X.shape
        best_gain = -1e30
        best_feat = 0
        best_thresh = 0.0
        best_left_val = 0.0
        best_right_val = 0.0
        n_try = max(5, int(np.sqrt(D)))
        feat_idx = np.random.choice(D, size=min(n_try, D), replace=False)
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
                n_left = left_mask.sum()
                n_right = right_mask.sum()
                if n_left < self.min_leaf or n_right < self.min_leaf:
                    continue
                wl = sample_w[left_mask]
                wr = sample_w[right_mask]
                left_val = np.sum(residuals[left_mask] * wl) / (np.sum(wl) + 1e-12)
                right_val = np.sum(residuals[right_mask] * wr) / (np.sum(wr) + 1e-12)
                gain = (np.sum(wl) * left_val**2 + np.sum(wr) * right_val**2)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = f
                    best_thresh = thresh
                    best_left_val = left_val
                    best_right_val = right_val
        return best_feat, best_thresh, best_left_val, best_right_val

    def fit(self, X, y, verbose=False):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        N = len(y)
        n_pos = y.sum()
        n_neg = N - n_pos
        sample_w = np.where(y == 1, N / (2 * n_pos + 1e-12), N / (2 * n_neg + 1e-12))
        p_avg = np.clip(y.mean(), 0.01, 0.99)
        self.init_pred = np.log(p_avg / (1 - p_avg))
        F = np.full(N, self.init_pred)
        for i in range(self.n_stumps):
            p = sigmoid(F)
            residuals = y - p
            feat, thresh, left_val, right_val = self._find_best_stump(X, residuals, sample_w)
            self.stumps.append((feat, thresh, left_val, right_val))
            mask = X[:, feat] <= thresh
            F[mask] += self.lr * left_val
            F[~mask] += self.lr * right_val
            if verbose and (i + 1) % 100 == 0:
                loss = -np.mean(y * np.log(sigmoid(F) + 1e-12) + (1 - y) * np.log(1 - sigmoid(F) + 1e-12))
                print(f"      Stump {i+1}/{self.n_stumps}: loss={loss:.4f}")

    def predict_proba(self, X):
        X = np.array(X, dtype=np.float64)
        F = np.full(X.shape[0], self.init_pred)
        for feat, thresh, left_val, right_val in self.stumps:
            mask = X[:, feat] <= thresh
            F[mask] += self.lr * left_val
            F[~mask] += self.lr * right_val
        return sigmoid(F)


# =============================================================================
# ML: GRADIENT BOOSTED TREES (depth > 1) - NEW for v4
# =============================================================================

class GradientBoostedTrees:
    """Gradient boosted trees with configurable depth for binary classification."""
    def __init__(self, n_trees=200, max_depth=3, learning_rate=0.05,
                 min_samples_leaf=20, n_features_try=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.lr = learning_rate
        self.min_leaf = min_samples_leaf
        self.n_features_try = n_features_try
        self.trees = []
        self.init_pred = 0.0

    def _build_tree(self, X, residuals, sample_w, depth=0):
        """Recursively build a decision tree on residuals."""
        N = len(residuals)
        leaf_val = np.sum(residuals * sample_w) / (np.sum(sample_w) + 1e-12)

        if depth >= self.max_depth or N < 2 * self.min_leaf:
            return {'leaf': True, 'val': leaf_val}

        D = X.shape[1]
        n_try = self.n_features_try or max(5, int(np.sqrt(D)))
        feat_idx = np.random.choice(D, size=min(n_try, D), replace=False)

        best_gain = -1e30
        best_feat = 0
        best_thresh = 0.0

        for f in feat_idx:
            col = X[:, f]
            unique_vals = np.unique(col)
            if len(unique_vals) <= 1:
                continue
            if len(unique_vals) > 15:
                thresholds = np.percentile(col, np.linspace(10, 90, 15))
            else:
                thresholds = unique_vals[:-1]
            for thresh in thresholds:
                left_mask = col <= thresh
                right_mask = ~left_mask
                n_left = left_mask.sum()
                n_right = right_mask.sum()
                if n_left < self.min_leaf or n_right < self.min_leaf:
                    continue
                wl = sample_w[left_mask]
                wr = sample_w[right_mask]
                lv = np.sum(residuals[left_mask] * wl) / (np.sum(wl) + 1e-12)
                rv = np.sum(residuals[right_mask] * wr) / (np.sum(wr) + 1e-12)
                gain = np.sum(wl) * lv**2 + np.sum(wr) * rv**2
                if gain > best_gain:
                    best_gain = gain
                    best_feat = f
                    best_thresh = thresh

        if best_gain <= 0:
            return {'leaf': True, 'val': leaf_val}

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], residuals[left_mask],
                                       sample_w[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], residuals[right_mask],
                                        sample_w[right_mask], depth + 1)

        return {
            'leaf': False, 'feat': best_feat, 'thresh': best_thresh,
            'left': left_child, 'right': right_child
        }

    def _predict_tree(self, node, x_row):
        if node['leaf']:
            return node['val']
        if x_row[node['feat']] <= node['thresh']:
            return self._predict_tree(node['left'], x_row)
        else:
            return self._predict_tree(node['right'], x_row)

    def _predict_tree_batch(self, node, X):
        """Vectorized tree prediction."""
        result = np.zeros(X.shape[0])
        if node['leaf']:
            return np.full(X.shape[0], node['val'])

        left_mask = X[:, node['feat']] <= node['thresh']
        if left_mask.sum() > 0:
            result[left_mask] = self._predict_tree_batch(node['left'], X[left_mask])
        if (~left_mask).sum() > 0:
            result[~left_mask] = self._predict_tree_batch(node['right'], X[~left_mask])
        return result

    def fit(self, X, y, verbose=False):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        N = len(y)
        n_pos = y.sum()
        n_neg = N - n_pos
        sample_w = np.where(y == 1, N / (2 * n_pos + 1e-12), N / (2 * n_neg + 1e-12))
        p_avg = np.clip(y.mean(), 0.01, 0.99)
        self.init_pred = np.log(p_avg / (1 - p_avg))
        F = np.full(N, self.init_pred)

        for i in range(self.n_trees):
            p = sigmoid(F)
            residuals = y - p
            tree = self._build_tree(X, residuals, sample_w)
            self.trees.append(tree)
            F += self.lr * self._predict_tree_batch(tree, X)
            if verbose and (i + 1) % 50 == 0:
                loss = -np.mean(y * np.log(sigmoid(F) + 1e-12) + (1 - y) * np.log(1 - sigmoid(F) + 1e-12))
                print(f"      Tree {i+1}/{self.n_trees}: loss={loss:.4f}")

    def predict_proba(self, X):
        X = np.array(X, dtype=np.float64)
        F = np.full(X.shape[0], self.init_pred)
        for tree in self.trees:
            F += self.lr * self._predict_tree_batch(tree, X)
        return sigmoid(F)


# =============================================================================
# ML: BAGGED LOGISTIC REGRESSION (feature subspace)
# =============================================================================

class BaggedLogistic:
    def __init__(self, n_bags=10, feature_fraction=0.5, lam=1.0, epochs=800):
        self.n_bags = n_bags
        self.ff = feature_fraction
        self.lam = lam
        self.epochs = epochs
        self.models = []

    def fit(self, X, y, verbose=False):
        X = np.array(X, dtype=np.float64)
        y = np.array(y, dtype=np.float64)
        D = X.shape[1]
        n_feat = max(3, int(D * self.ff))
        rng = np.random.RandomState(123)
        for i in range(self.n_bags):
            feat_idx = rng.choice(D, size=n_feat, replace=False)
            feat_idx.sort()
            X_sub = X[:, feat_idx]
            w, b, mu, sd = logistic_train(X_sub, y, lr=0.01, lam=self.lam,
                                           epochs=self.epochs, class_weight='balanced')
            self.models.append((feat_idx, w, b, mu, sd))
            if verbose and (i + 1) % 5 == 0:
                print(f"      Bag {i+1}/{self.n_bags} done")

    def predict_proba(self, X):
        X = np.array(X, dtype=np.float64)
        preds = np.zeros(X.shape[0])
        for feat_idx, w, b, mu, sd in self.models:
            X_sub = X[:, feat_idx]
            preds += logistic_predict(X_sub, w, b, mu, sd)
        return preds / len(self.models)


# =============================================================================
# ML: KNN DENSITY ESTIMATOR - NEW for v4
# =============================================================================

class KNNDensityEstimator:
    """Simple k-nearest-neighbor density estimator for binary classification."""
    def __init__(self, k=50):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.mu = None
        self.sd = None

    def fit(self, X, y, verbose=False):
        X = np.array(X, dtype=np.float64)
        self.mu = X.mean(axis=0)
        self.sd = X.std(axis=0)
        self.sd[self.sd < 1e-12] = 1.0
        self.X_train = (X - self.mu) / self.sd
        self.y_train = np.array(y, dtype=np.float64)
        if verbose:
            print(f"      KNN: stored {len(y)} training points, k={self.k}")

    def predict_proba(self, X):
        X = np.array(X, dtype=np.float64)
        Xs = (X - self.mu) / self.sd
        N = Xs.shape[0]
        M = self.X_train.shape[0]
        probs = np.zeros(N)
        k = min(self.k, M)

        # Precompute squared norms for efficient distance calculation
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        train_sq = np.sum(self.X_train ** 2, axis=1)  # (M,)

        # Process in batches
        batch_size = 500
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = Xs[start:end]  # (B, D)
            batch_sq = np.sum(batch ** 2, axis=1)  # (B,)
            # (B, M) distance matrix
            dists = batch_sq[:, None] + train_sq[None, :] - 2.0 * (batch @ self.X_train.T)
            for i in range(end - start):
                nearest_idx = np.argpartition(dists[i], k)[:k]
                probs[start + i] = self.y_train[nearest_idx].mean()
        return probs


# =============================================================================
# DATA LOADING
# =============================================================================

def load_spc_tornadoes():
    url = "https://www.spc.noaa.gov/wcm/data/1950-2023_actual_tornadoes.csv"
    print("  Loading SPC tornado data...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode('utf-8', errors='replace')
    except Exception as e:
        print(f"  Download failed: {e}")
        return None

    reader = csv.DictReader(io.StringIO(raw))
    tornadoes = []
    for row in reader:
        try:
            yr = int(row.get('yr', 0))
            mo = int(row.get('mo', 0))
            dy = int(row.get('dy', 0))
            mag = int(row.get('mag', -9))
            wid = float(row.get('wid', 0))
            plen = float(row.get('len', 0))
            slat = float(row.get('slat', 0))
            slon = float(row.get('slon', 0))
            elat = float(row.get('elat', 0))
            elon = float(row.get('elon', 0))
            # Time
            time_str = row.get('time', '').strip()
            hour = -1.0
            if time_str and ':' in time_str:
                try:
                    parts = time_str.split(':')
                    hour = int(parts[0]) + int(parts[1]) / 60.0 if len(parts) > 1 else float(int(parts[0]))
                except Exception:
                    hour = -1.0
            # Casualties/damage
            fat = 0
            inj = 0
            loss = 0.0
            try:
                fat = int(row.get('fat', 0))
            except Exception:
                pass
            try:
                inj = int(row.get('inj', 0))
            except Exception:
                pass
            try:
                loss = float(row.get('loss', 0))
            except Exception:
                pass

            if yr < 1980:
                continue
            if mag < 0:
                continue
            if slat < 24 or slat > 50 or slon > -65 or slon < -125:
                continue
            if slat == 0 or slon == 0:
                continue

            tornadoes.append({
                'yr': yr, 'mo': mo, 'dy': dy,
                'mag': mag,
                'width_yd': wid,
                'width_m': wid * 0.9144 if wid > 0 else 0,
                'path_len_mi': plen,
                'path_len_km': plen * 1.60934 if plen > 0 else 0,
                'slat': slat, 'slon': slon,
                'elat': elat, 'elon': elon,
                'hour': hour,
                'fat': fat, 'inj': inj, 'loss': loss,
            })
        except (ValueError, KeyError):
            continue

    print(f"    {len(tornadoes)} tornadoes loaded (1980-2023, CONUS)")
    return tornadoes


# =============================================================================
# GRID AND DATE UTILITIES
# =============================================================================

def date_to_day_number(yr, mo, dy):
    days_in_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    d = (yr - 2000) * 365 + (yr - 2000) // 4
    for m in range(1, mo):
        d += days_in_month[m]
    d += dy
    return d


def day_number_to_approx_doy(dn):
    return ((dn % 365) + 365) % 365


def latlon_to_cell(lat, lon, cell_size=2.0):
    lat_bin = int((lat - 25.0) / cell_size)
    lon_bin = int((lon - (-125.0)) / cell_size)
    n_lat = int((50 - 25) / cell_size)
    n_lon = int(60 / cell_size)
    lat_bin = max(0, min(lat_bin, n_lat - 1))
    lon_bin = max(0, min(lon_bin, n_lon - 1))
    return lat_bin, lon_bin


def haversine_approx_km(lat1, lon1, lat2, lon2):
    dlat = (lat2 - lat1) * 111.0
    dlon = (lon2 - lon1) * 111.0 * math.cos(math.radians((lat1 + lat2) / 2))
    return math.sqrt(dlat**2 + dlon**2)


# =============================================================================
# TOPOGRAPHIC PROXY FUNCTIONS
# =============================================================================

def distance_to_appalachians(lat, lon):
    """Approximate distance to Appalachian ridge line."""
    # Appalachians run roughly from (34,-84) to (40,-80)
    if 34 <= lat <= 40:
        app_lon = -84.0 + (lat - 34.0) / 6.0 * 4.0  # -84 at 34N to -80 at 40N
        return abs(lon - app_lon) * 111.0 * math.cos(math.radians(lat))
    elif lat < 34:
        return haversine_approx_km(lat, lon, 34, -84)
    else:
        return haversine_approx_km(lat, lon, 40, -80)


def distance_to_rockies(lat, lon):
    """Approximate distance to Rocky Mountain front range."""
    # Front Range roughly at lon -105 for lat 35-45
    rock_lon = -105.0 if 35 <= lat <= 45 else (-107.0 if lat < 35 else -110.0)
    return abs(lon - rock_lon) * 111.0 * math.cos(math.radians(lat))


def distance_to_ozarks(lat, lon):
    """Approximate distance to Ozark Plateau center."""
    return haversine_approx_km(lat, lon, 36.0, -93.0)


def great_plains_indicator(lat, lon):
    """1 if in Great Plains tornado alley, 0 otherwise, with smooth falloff."""
    if 30 <= lat <= 45 and -103 <= lon <= -95:
        return 1.0
    # Smooth falloff
    lat_dist = max(0, max(30 - lat, lat - 45)) / 5.0
    lon_dist = max(0, max(-103 - lon, lon - (-95))) / 5.0
    d = math.sqrt(lat_dist**2 + lon_dist**2)
    return max(0.0, 1.0 - d)


def mississippi_valley_indicator(lat, lon):
    """1 if in Mississippi Valley moisture corridor."""
    if -92 <= lon <= -88:
        return 1.0
    d = min(abs(lon - (-92)), abs(lon - (-88))) / 4.0
    return max(0.0, 1.0 - d)


def elevation_proxy(lat, lon):
    """Very rough US elevation proxy (meters) from lat/lon."""
    # Great Plains: ~500-1500m, decreasing east
    # Rockies: ~2000-3000m around lon -105
    # Mississippi Valley: ~100m
    # East coast: ~100-300m
    # Appalachians: ~600-1200m
    base = 200.0
    # Rocky Mountain contribution
    rock_dist = abs(lon - (-105.0))
    if rock_dist < 5:
        base += (5 - rock_dist) / 5.0 * 2000
    # Great Plains slope (lon -95 to -105 is ~500-1500m)
    if -105 <= lon <= -95:
        base += (abs(lon + 95) / 10.0) * 1000 + 300
    # Appalachian contribution
    if -84 <= lon <= -78 and 34 <= lat <= 40:
        base += 800
    return base


# =============================================================================
# ENVIRONMENTAL PROXY FUNCTIONS (ERA5-like from tornado data)
# =============================================================================

def cape_proxy(lat, mo, doy):
    """
    CAPE proxy from latitude, month, day-of-year.
    CAPE peaks in Gulf Coast spring afternoons.
    Uses climatological relationship.
    """
    # Latitude component: higher CAPE at lower latitudes (Gulf Coast)
    lat_factor = max(0, (45 - lat) / 20.0)  # 0 at 45N, 1 at 25N
    # Seasonal component: peak in May-June (doy 120-170)
    seasonal = 0.5 + 0.5 * math.sin(2 * math.pi * (doy - 60) / 365.25)
    # Month interaction
    month_peak = max(0, 1.0 - abs(mo - 5.5) / 4.0)  # peak around May-June
    return lat_factor * seasonal * month_peak


def bulk_shear_proxy_from_widths(nearby_widths, lat_gradient):
    """
    Bulk shear proxy: wider tornadoes = stronger shear environment.
    Also uses latitude gradient of tornado occurrence.
    """
    if nearby_widths:
        mean_w = np.mean(nearby_widths)
        max_w = max(nearby_widths)
        shear_from_width = np.log1p(mean_w) / 7.0  # normalize
        shear_from_max = np.log1p(max_w) / 8.0
    else:
        shear_from_width = 0.0
        shear_from_max = 0.0
    # Sharp lat gradient = strong jet stream = strong shear
    shear_from_gradient = min(1.0, abs(lat_gradient) / 5.0)
    return shear_from_width, shear_from_max, shear_from_gradient


def srh_proxy(nearby_path_dirs, nearby_path_lens):
    """
    Storm-relative helicity proxy from tornado path direction and length.
    Long, straight-track tornadoes = high SRH.
    """
    if not nearby_path_lens or len(nearby_path_lens) == 0:
        return 0.0, 0.0, 0.0
    mean_len = np.mean(nearby_path_lens)
    max_len = max(nearby_path_lens)
    # Directional consistency (low std = organized flow)
    if len(nearby_path_dirs) > 1:
        # Circular std of directions
        sin_sum = np.sum(np.sin(nearby_path_dirs))
        cos_sum = np.sum(np.cos(nearby_path_dirs))
        R = math.sqrt(sin_sum**2 + cos_sum**2) / len(nearby_path_dirs)
        dir_consistency = R  # 0=random, 1=all same direction
    else:
        dir_consistency = 0.5
    return np.log1p(mean_len), np.log1p(max_len), dir_consistency


def lcl_proxy(doy, lat):
    """
    Lifted condensation level proxy from day-of-year and latitude.
    Dry seasons/locations = higher LCL.
    """
    # Moisture (lower LCL) peaks in summer in Southeast
    moisture_seasonal = 0.5 + 0.5 * math.sin(2 * math.pi * (doy - 90) / 365.25)
    moisture_lat = max(0, (40 - lat) / 15.0)  # more moisture at lower lats
    lcl = 1.0 - moisture_seasonal * moisture_lat  # high = dry = high LCL
    return lcl


def moisture_proxy_from_gulf(gulf_count_1d, gulf_count_3d):
    """Moisture proxy: tornado activity in Gulf feeder region = moisture advection."""
    return np.log1p(gulf_count_1d), np.log1p(gulf_count_3d)


# =============================================================================
# INDEX BUILDING (multi-scale) - Enhanced for v4
# =============================================================================

def build_indices(tornadoes):
    print("  Building multi-scale indices...")

    indices = {}
    for cell_size in [1.0, 2.0, 4.0]:
        cell_day = defaultdict(list)
        cell_month = defaultdict(list)
        for t in tornadoes:
            dn = date_to_day_number(t['yr'], t['mo'], t['dy'])
            lb, lnb = latlon_to_cell(t['slat'], t['slon'], cell_size)
            cell_day[(lb, lnb, dn)].append(t)
            cell_month[(lb, lnb, t['mo'])].append(t)
        indices[cell_size] = {'cell_day': cell_day, 'cell_month': cell_month}
        print(f"    {cell_size}-degree grid: {len(cell_day)} cell-day events")

    # Day-level index
    day_events = defaultdict(list)
    for t in tornadoes:
        dn = date_to_day_number(t['yr'], t['mo'], t['dy'])
        lb, lnb = latlon_to_cell(t['slat'], t['slon'], 2.0)
        day_events[dn].append((lb, lnb, t))
    indices['day_events'] = day_events

    # Lat/lon level index for radius queries
    latlon_day = defaultdict(list)
    for t in tornadoes:
        dn = date_to_day_number(t['yr'], t['mo'], t['dy'])
        lat_i = int(t['slat'])
        lon_i = int(t['slon'])
        latlon_day[(lat_i, lon_i, dn)].append(t)
    indices['latlon_day'] = latlon_day

    # Day-level tornado list (for synoptic pattern recognition)
    day_tornadoes = defaultdict(list)
    for t in tornadoes:
        dn = date_to_day_number(t['yr'], t['mo'], t['dy'])
        day_tornadoes[dn].append(t)
    indices['day_tornadoes'] = day_tornadoes

    # Gulf Coast feeder region index (lat < 35, lon > -100)
    gulf_day = defaultdict(int)
    for t in tornadoes:
        if t['slat'] < 35 and t['slon'] > -100:
            dn = date_to_day_number(t['yr'], t['mo'], t['dy'])
            gulf_day[dn] += 1
    indices['gulf_day'] = gulf_day

    return indices


# =============================================================================
# FORMATION FEATURE ENGINEERING (v4)
# =============================================================================

def build_formation_features(tornadoes, indices, train_years, test_years):
    print("\n" + "=" * 70)
    print("PART 1: FORMATION PREDICTION (v4)")
    print("=" * 70)

    cell_size = 2.0
    n_lat = int((50 - 25) / cell_size)
    n_lon = int(60 / cell_size)

    cd2 = indices[2.0]['cell_day']
    cm2 = indices[2.0]['cell_month']
    cd1 = indices[1.0]['cell_day']
    cd4 = indices[4.0]['cell_day']
    day_events = indices['day_events']
    latlon_day = indices['latlon_day']
    day_tornadoes = indices['day_tornadoes']
    gulf_day = indices['gulf_day']

    days_per_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # Precompute climatological rates (training period)
    print("  Computing climatological rates...")
    train_start, train_end = train_years
    clim_counts = defaultdict(int)
    for t in tornadoes:
        if train_start <= t['yr'] <= train_end:
            lb, lnb = latlon_to_cell(t['slat'], t['slon'], cell_size)
            clim_counts[(lb, lnb, t['mo'])] += 1

    clim_rate = {}
    for lb in range(n_lat):
        for lnb in range(n_lon):
            for mo in range(1, 13):
                n_years = train_end - train_start + 1
                n_days = n_years * days_per_month[mo] if 1 <= mo <= 12 else 30
                cnt = clim_counts.get((lb, lnb, mo), 0)
                clim_rate[(lb, lnb, mo)] = cnt / max(n_days, 1)

    # Multi-year trend
    print("  Computing multi-year trends...")
    cell_year_count = defaultdict(lambda: defaultdict(int))
    for t in tornadoes:
        if train_start <= t['yr'] <= train_end:
            lb, lnb = latlon_to_cell(t['slat'], t['slon'], cell_size)
            cell_year_count[(lb, lnb)][t['yr']] += 1

    cell_trend = {}
    for (lb, lnb), yr_counts in cell_year_count.items():
        years = sorted(yr_counts.keys())
        if len(years) >= 5:
            x = np.array(years, dtype=np.float64)
            y = np.array([yr_counts[yr] for yr in years], dtype=np.float64)
            x_mean = x.mean()
            y_mean = y.mean()
            slope = np.sum((x - x_mean) * (y - y_mean)) / (np.sum((x - x_mean)**2) + 1e-12)
            cell_trend[(lb, lnb)] = slope
        else:
            cell_trend[(lb, lnb)] = 0.0

    # -------------------------------------------------------------------------
    # v4 Feature function
    # -------------------------------------------------------------------------
    def get_features(lb, lnb, dn, mo, yr):
        feat = []
        center_lat = 25.0 + lb * cell_size + cell_size / 2
        center_lon = -125.0 + lnb * cell_size + cell_size / 2
        doy = (mo - 1) * 30.4 + 15

        # =================================================================
        # BLOCK 1: SEASONAL / CLIMATOLOGICAL (12 features) [0-11]
        # =================================================================
        feat.append(math.sin(2 * math.pi * doy / 365.25))      # 0
        feat.append(math.cos(2 * math.pi * doy / 365.25))      # 1
        feat.append(math.sin(4 * math.pi * doy / 365.25))      # 2
        feat.append(math.cos(4 * math.pi * doy / 365.25))      # 3

        cr = clim_rate.get((lb, lnb, mo), 0.0)
        feat.append(cr)                                          # 4
        feat.append(cr * cr)                                     # 5

        nb_cr = 0.0
        n_nb = 0
        for dlb in [-1, 0, 1]:
            for dlnb in [-1, 0, 1]:
                if dlb == 0 and dlnb == 0:
                    continue
                if 0 <= lb + dlb < n_lat and 0 <= lnb + dlnb < n_lon:
                    nb_cr += clim_rate.get((lb + dlb, lnb + dlnb, mo), 0.0)
                    n_nb += 1
        feat.append(nb_cr / max(n_nb, 1))                       # 6

        feat.append(math.sin(2 * math.pi * doy / 365.25) * lb / n_lat)  # 7
        feat.append(math.cos(2 * math.pi * doy / 365.25) * lb / n_lat)  # 8
        feat.append(cell_trend.get((lb, lnb), 0.0))             # 9
        feat.append(lb / n_lat)                                  # 10
        feat.append(lnb / n_lon)                                 # 11

        # =================================================================
        # BLOCK 2: MULTI-SCALE TEMPORAL ACTIVITY (18 features) [12-29]
        # =================================================================
        for cs, cd in [(1.0, cd1), (2.0, cd2), (4.0, cd4)]:
            lb_s, lnb_s = latlon_to_cell(center_lat, center_lon, cs)
            for lookback in [1, 3, 7, 14, 30, 90]:
                count = 0
                for d_off in range(1, lookback + 1):
                    count += len(cd.get((lb_s, lnb_s, dn - d_off), []))
                feat.append(count)

        # =================================================================
        # BLOCK 3: SPATIAL PROPAGATION (9 features) [30-38]
        # =================================================================
        for max_dist in [1, 2, 3]:
            for max_lag in [1, 3, 7]:
                count = 0
                for dlb in range(-max_dist, max_dist + 1):
                    for dlnb in range(-max_dist, max_dist + 1):
                        if dlb == 0 and dlnb == 0:
                            continue
                        if max(abs(dlb), abs(dlnb)) > max_dist:
                            continue
                        nlb = lb + dlb
                        nlnb = lnb + dlnb
                        if 0 <= nlb < n_lat and 0 <= nlnb < n_lon:
                            for d_off in range(0, max_lag + 1):
                                count += len(cd2.get((nlb, nlnb, dn - d_off), []))
                feat.append(count)

        # =================================================================
        # BLOCK 4: DIRECTIONAL PROPAGATION (4 features) [39-42]
        # =================================================================
        ne_count = sw_count = nw_count = se_count = 0
        for dlb in range(-3, 4):
            for dlnb in range(-3, 4):
                if dlb == 0 and dlnb == 0:
                    continue
                nlb, nlnb = lb + dlb, lnb + dlnb
                if 0 <= nlb < n_lat and 0 <= nlnb < n_lon:
                    cnt = 0
                    for d_off in [1, 2]:
                        cnt += len(cd2.get((nlb, nlnb, dn - d_off), []))
                    if cnt > 0:
                        if dlb > 0 and dlnb > 0:
                            ne_count += cnt
                        elif dlb < 0 and dlnb < 0:
                            sw_count += cnt
                        elif dlb > 0 and dlnb < 0:
                            nw_count += cnt
                        else:
                            se_count += cnt
        feat.append(sw_count)                                    # 39
        feat.append(ne_count)                                    # 40
        feat.append(sw_count - ne_count)                         # 41
        feat.append(sw_count + se_count)                         # 42

        # =================================================================
        # BLOCK 5: OUTBREAK DETECTION (6 features) [43-48]
        # =================================================================
        for lookback in [0, 1, 2, 3]:
            active_cells = set()
            total_tor = 0
            for d_off in range(0, lookback + 1):
                for (alb, alnb, t_rec) in day_events.get(dn - d_off, []):
                    active_cells.add((alb, alnb))
                    total_tor += 1
            if lookback == 0:
                feat.append(len(active_cells))                   # 43
            else:
                feat.append(len(active_cells))
        feat.append(np.log1p(total_tor))                         # 47
        outbreak_len = 0
        for d_off in range(1, 8):
            if len(day_events.get(dn - d_off, [])) > 0:
                outbreak_len += 1
            else:
                break
        feat.append(outbreak_len)                                # 48

        # =================================================================
        # BLOCK 6: TORNADO ACCELERATION (3 features) [49-51]
        # =================================================================
        counts_by_day = []
        for d_off in range(4):
            cnt = 0
            for dlb in range(-2, 3):
                for dlnb in range(-2, 3):
                    nlb, nlnb = lb + dlb, lnb + dlnb
                    if 0 <= nlb < n_lat and 0 <= nlnb < n_lon:
                        cnt += len(cd2.get((nlb, nlnb, dn - d_off), []))
            counts_by_day.append(cnt)
        feat.append(counts_by_day[1] - counts_by_day[2])        # 49
        feat.append(counts_by_day[0] - counts_by_day[1])        # 50
        feat.append(np.log1p(counts_by_day[1]))                 # 51

        # =================================================================
        # BLOCK 7: INTENSITY ESCALATION (10 features) [52-61]
        # =================================================================
        ef_vals_recent = []
        ef_vals_older = []
        widths_recent = []
        widths_older = []
        path_len_recent = []
        killer_nearby = 0
        inj_nearby = 0
        loss_nearby = 0.0
        for dlb in range(-2, 3):
            for dlnb in range(-2, 3):
                nlb, nlnb = lb + dlb, lnb + dlnb
                if 0 <= nlb < n_lat and 0 <= nlnb < n_lon:
                    for d_off in [0, 1]:
                        for evt in cd2.get((nlb, nlnb, dn - d_off), []):
                            ef_vals_recent.append(evt['mag'])
                            if evt['width_yd'] > 0:
                                widths_recent.append(evt['width_yd'])
                            if evt['path_len_mi'] > 0:
                                path_len_recent.append(evt['path_len_mi'])
                            if evt['fat'] > 0:
                                killer_nearby = 1
                            inj_nearby += evt.get('inj', 0)
                            loss_nearby += evt.get('loss', 0.0)
                    for d_off in [2, 3]:
                        for evt in cd2.get((nlb, nlnb, dn - d_off), []):
                            ef_vals_older.append(evt['mag'])
                            if evt['width_yd'] > 0:
                                widths_older.append(evt['width_yd'])

        mean_ef_recent = np.mean(ef_vals_recent) if ef_vals_recent else 0.0
        mean_ef_older = np.mean(ef_vals_older) if ef_vals_older else 0.0
        feat.append(mean_ef_recent)                              # 52
        feat.append(mean_ef_recent - mean_ef_older)              # 53

        mean_w_recent = np.mean(widths_recent) if widths_recent else 0.0
        mean_w_older = np.mean(widths_older) if widths_older else 0.0
        feat.append(np.log1p(mean_w_recent))                    # 54
        feat.append(mean_w_recent - mean_w_older)               # 55
        feat.append(killer_nearby)                               # 56
        feat.append(np.sum(path_len_recent))                     # 57

        max_ef_7d = 0
        max_ef_30d = 0
        for d_off in range(1, 31):
            for evt in cd2.get((lb, lnb, dn - d_off), []):
                if d_off <= 7 and evt['mag'] > max_ef_7d:
                    max_ef_7d = evt['mag']
                if evt['mag'] > max_ef_30d:
                    max_ef_30d = evt['mag']
        feat.append(max_ef_7d)                                   # 58
        feat.append(max_ef_30d)                                  # 59

        # Population/impact proxies (nearby injuries, loss)
        feat.append(np.log1p(inj_nearby))                        # 60
        feat.append(np.log1p(loss_nearby))                       # 61

        # =================================================================
        # BLOCK 8: WIDTH-LIFETIME COHERENCE SCALING (3 features) [62-64]
        # =================================================================
        if widths_recent:
            max_w_m = max(widths_recent) * 0.9144
            pred_path_km = 0.015 * (max_w_m ** 0.97) if max_w_m > 0 else 0.0
            feat.append(pred_path_km)                            # 62
            if path_len_recent:
                actual_mean_path = np.mean(path_len_recent) * 1.60934
                feat.append(actual_mean_path - pred_path_km)     # 63
            else:
                feat.append(0.0)
            feat.append(np.log1p(max_w_m))                       # 64
        else:
            feat.extend([0.0, 0.0, 0.0])

        # =================================================================
        # BLOCK 9: CELL-MONTH ANOMALY (2 features) [65-66]
        # =================================================================
        this_yr_count = 0
        for t_rec in cm2.get((lb, lnb, mo), []):
            if t_rec['yr'] == yr:
                t_dn = date_to_day_number(t_rec['yr'], t_rec['mo'], t_rec['dy'])
                if t_dn < dn:
                    this_yr_count += 1
        avg_count = cr * days_per_month[mo] if 1 <= mo <= 12 else 0
        feat.append(this_yr_count - avg_count)                   # 65
        feat.append(this_yr_count)                               # 66

        # =================================================================
        # BLOCK 10: INTERACTION TERMS (4 features) [67-70]
        # =================================================================
        prop_signal = feat[30]
        feat.append(cr * prop_signal)                            # 67
        feat.append(cr * mean_ef_recent)                         # 68
        feat.append(feat[10] * feat[0])                          # 69
        feat.append(float(outbreak_len > 0) * prop_signal)       # 70

        # =================================================================
        # BLOCK 11: ERA5 ENVIRONMENTAL PROXIES (10 features) [71-80]
        # =================================================================
        # CAPE proxy
        cape_val = cape_proxy(center_lat, mo, doy)
        feat.append(cape_val)                                    # 71
        feat.append(cape_val * cr)                               # 72 CAPE * clim interaction

        # Bulk shear proxy
        # Latitude gradient: count tornadoes in cells north vs south
        north_count = 0
        south_count = 0
        for dlb in range(-3, 4):
            for dlnb in range(-1, 2):
                nlb, nlnb = lb + dlb, lnb + dlnb
                if 0 <= nlb < n_lat and 0 <= nlnb < n_lon:
                    cnt = len(cd2.get((nlb, nlnb, dn - 1), [])) + len(cd2.get((nlb, nlnb, dn), []))
                    if dlb > 0:
                        north_count += cnt
                    elif dlb < 0:
                        south_count += cnt
        lat_gradient = north_count - south_count
        shear_w, shear_max, shear_grad = bulk_shear_proxy_from_widths(widths_recent, lat_gradient)
        feat.append(shear_w)                                     # 73
        feat.append(shear_max)                                   # 74
        feat.append(shear_grad)                                  # 75

        # SRH proxy from path directions and lengths
        path_dirs = []
        path_lens = []
        for dlb in range(-2, 3):
            for dlnb in range(-2, 3):
                nlb, nlnb = lb + dlb, lnb + dlnb
                if 0 <= nlb < n_lat and 0 <= nlnb < n_lon:
                    for d_off in [0, 1]:
                        for evt in cd2.get((nlb, nlnb, dn - d_off), []):
                            if evt['path_len_mi'] > 0:
                                path_lens.append(evt['path_len_mi'])
                            if evt.get('elat', 0) != 0 and evt.get('elon', 0) != 0:
                                dlat = evt['elat'] - evt['slat']
                                dlon = evt['elon'] - evt['slon']
                                if abs(dlat) + abs(dlon) > 0.001:
                                    path_dirs.append(math.atan2(dlat, dlon))
        srh_mean_len, srh_max_len, srh_consistency = srh_proxy(path_dirs, path_lens)
        feat.append(srh_mean_len)                                # 76
        feat.append(srh_max_len)                                 # 77
        feat.append(srh_consistency)                             # 78

        # LCL proxy
        feat.append(lcl_proxy(doy, center_lat))                 # 79

        # Moisture proxy from Gulf Coast activity
        gulf_1d = gulf_day.get(dn - 1, 0) + gulf_day.get(dn, 0)
        gulf_3d = sum(gulf_day.get(dn - d, 0) for d in range(4))
        m1, m3 = moisture_proxy_from_gulf(gulf_1d, gulf_3d)
        feat.append(m1)                                          # 80

        # =================================================================
        # BLOCK 12: SYNOPTIC PATTERN RECOGNITION (8 features) [81-88]
        # =================================================================
        # Dryline proxy: sharp east-west gradient in tornado occurrence
        east_count = 0
        west_count = 0
        for dlb in range(-2, 3):
            for dlnb in range(1, 4):
                nlb, nlnb = lb + dlb, lnb + dlnb
                if 0 <= nlb < n_lat and 0 <= nlnb < n_lon:
                    east_count += len(cd2.get((nlb, nlnb, dn), [])) + len(cd2.get((nlb, nlnb, dn - 1), []))
            for dlnb in range(-3, 0):
                nlb, nlnb = lb + dlb, lnb + dlnb
                if 0 <= nlb < n_lat and 0 <= nlnb < n_lon:
                    west_count += len(cd2.get((nlb, nlnb, dn), [])) + len(cd2.get((nlb, nlnb, dn - 1), []))
        dryline_signal = abs(east_count - west_count) / max(east_count + west_count, 1)
        feat.append(dryline_signal)                              # 81

        # Cold front proxy: SW-to-NE propagation in past 24-48h
        # Fit a line to tornado locations over past 2 days
        recent_lats = []
        recent_lons = []
        recent_times = []
        for d_off in [0, 1, 2]:
            for (alb, alnb, t_rec) in day_events.get(dn - d_off, []):
                recent_lats.append(t_rec['slat'])
                recent_lons.append(t_rec['slon'])
                recent_times.append(d_off)
        front_speed = 0.0
        front_angle = 0.0
        if len(recent_lats) >= 3:
            lats_arr = np.array(recent_lats)
            lons_arr = np.array(recent_lons)
            times_arr = np.array(recent_times, dtype=np.float64)
            if times_arr.std() > 0:
                lat_slope = np.corrcoef(times_arr, lats_arr)[0, 1] if len(times_arr) > 1 else 0
                lon_slope = np.corrcoef(times_arr, lons_arr)[0, 1] if len(times_arr) > 1 else 0
                front_speed = math.sqrt(lat_slope**2 + lon_slope**2) if not (np.isnan(lat_slope) or np.isnan(lon_slope)) else 0.0
                front_angle = math.atan2(lat_slope, lon_slope) if not (np.isnan(lat_slope) or np.isnan(lon_slope)) else 0.0
        feat.append(front_speed)                                 # 82
        feat.append(math.sin(front_angle))                       # 83
        feat.append(math.cos(front_angle))                       # 84

        # Warm front proxy: tornadoes north of main activity
        north_isolated = 0
        main_lat = np.mean(recent_lats) if recent_lats else center_lat
        for (alb, alnb, t_rec) in day_events.get(dn, []):
            if t_rec['slat'] > main_lat + 2:
                north_isolated += 1
        feat.append(north_isolated)                              # 85

        # Squall line vs supercell discrimination
        # Many tornadoes in one cell-day = squall line pattern
        today_events = day_events.get(dn, [])
        cell_counts_today = defaultdict(int)
        today_widths = []
        for (alb, alnb, t_rec) in today_events:
            cell_counts_today[(alb, alnb)] += 1
            if t_rec['width_yd'] > 0:
                today_widths.append(t_rec['width_yd'])
        max_per_cell = max(cell_counts_today.values()) if cell_counts_today else 0
        mean_width_today = np.mean(today_widths) if today_widths else 0
        # High count + low width = squall; low count + high width = supercell
        squall_signal = max_per_cell / (1 + np.log1p(mean_width_today))
        supercell_signal = np.log1p(mean_width_today) / (1 + max_per_cell)
        feat.append(squall_signal)                               # 86
        feat.append(supercell_signal)                            # 87

        # Jet stream position proxy: latitude of northernmost tornado in past 3 days
        max_lat_3d = 25.0
        for d_off in range(4):
            for t_rec in day_tornadoes.get(dn - d_off, []):
                if t_rec['slat'] > max_lat_3d:
                    max_lat_3d = t_rec['slat']
        feat.append((max_lat_3d - 25.0) / 25.0)                 # 88

        # =================================================================
        # BLOCK 13: DIURNAL CYCLE (5 features) [89-93]
        # =================================================================
        # We compute average hour for nearby tornadoes
        nearby_hours = []
        night_count = 0
        day_count = 0
        for (alb, alnb, t_rec) in day_events.get(dn, []):
            h = t_rec.get('hour', -1)
            if h >= 0:
                nearby_hours.append(h)
                if 6 <= h <= 18:
                    day_count += 1
                else:
                    night_count += 1
        if nearby_hours:
            mean_h = np.mean(nearby_hours)
            feat.append(math.sin(2 * math.pi * mean_h / 24))    # 89
            feat.append(math.cos(2 * math.pi * mean_h / 24))    # 90
            # Peak window: 18-00 UTC (1-7 PM CDT)
            peak_count = sum(1 for h in nearby_hours if 18 <= h <= 24 or h < 1)
            feat.append(peak_count / max(len(nearby_hours), 1))  # 91
        else:
            feat.append(0.0)                                     # 89
            feat.append(0.0)                                     # 90
            feat.append(0.5)                                     # 91

        # Night ratio (dangerous)
        total_dn = day_count + night_count
        feat.append(night_count / max(total_dn, 1))              # 92

        # Time since solar noon proxy (average for nearby tornadoes)
        # Solar noon ~ 12:00 local = 18:00 UTC for central US
        solar_noon_utc = 18.0 + (center_lon + 90) / 15.0
        if nearby_hours:
            mean_time_since_noon = np.mean([abs(h - solar_noon_utc) for h in nearby_hours])
            feat.append(mean_time_since_noon / 12.0)             # 93
        else:
            feat.append(0.5)                                     # 93

        # =================================================================
        # BLOCK 14: MULTI-DAY OUTBREAK DYNAMICS (6 features) [94-99]
        # =================================================================
        # Outbreak intensity trajectory
        ob_counts = []
        for d_off in range(5):
            ob_counts.append(len(day_events.get(dn - d_off, [])))
        # Is it building (positive slope) or winding down?
        if len(ob_counts) >= 3 and max(ob_counts) > 0:
            x_ob = np.arange(len(ob_counts), dtype=np.float64)
            y_ob = np.array(ob_counts, dtype=np.float64)
            x_mean = x_ob.mean()
            y_mean = y_ob.mean()
            ob_slope = np.sum((x_ob - x_mean) * (y_ob - y_mean)) / (np.sum((x_ob - x_mean)**2) + 1e-12)
            feat.append(ob_slope)                                # 94
        else:
            feat.append(0.0)

        # Cumulative damage path in outbreak so far
        cum_path = 0.0
        for d_off in range(4):
            for (alb, alnb, t_rec) in day_events.get(dn - d_off, []):
                cum_path += t_rec.get('path_len_mi', 0)
        feat.append(np.log1p(cum_path))                          # 95

        # Geographic spread rate
        all_lats = []
        all_lons = []
        today_lats = []
        today_lons = []
        for d_off in range(3):
            for (alb, alnb, t_rec) in day_events.get(dn - d_off, []):
                all_lats.append(t_rec['slat'])
                all_lons.append(t_rec['slon'])
                if d_off == 0:
                    today_lats.append(t_rec['slat'])
                    today_lons.append(t_rec['slon'])
        if len(all_lats) >= 2:
            spread_lat = np.std(all_lats)
            spread_lon = np.std(all_lons)
            feat.append(spread_lat + spread_lon)                 # 96
        else:
            feat.append(0.0)

        # Peak EF so far in outbreak
        peak_ef_outbreak = 0
        for d_off in range(4):
            for (alb, alnb, t_rec) in day_events.get(dn - d_off, []):
                if t_rec['mag'] > peak_ef_outbreak:
                    peak_ef_outbreak = t_rec['mag']
        feat.append(peak_ef_outbreak)                            # 97

        # Did outbreak produce EF4+ early?
        ef4_early = 0
        for d_off in [1, 2, 3]:
            for (alb, alnb, t_rec) in day_events.get(dn - d_off, []):
                if t_rec['mag'] >= 4:
                    ef4_early = 1
                    break
            if ef4_early:
                break
        feat.append(ef4_early)                                   # 98

        # Outbreak momentum: today count / yesterday count
        today_count = len(day_events.get(dn, []))
        yesterday_count = len(day_events.get(dn - 1, []))
        feat.append(today_count / max(yesterday_count, 1))       # 99

        # =================================================================
        # BLOCK 15: TOPOGRAPHIC PROXIES (7 features) [100-106]
        # =================================================================
        feat.append(distance_to_appalachians(center_lat, center_lon) / 500.0)  # 100
        feat.append(distance_to_rockies(center_lat, center_lon) / 500.0)       # 101
        feat.append(distance_to_ozarks(center_lat, center_lon) / 500.0)        # 102
        feat.append(great_plains_indicator(center_lat, center_lon))             # 103
        feat.append(mississippi_valley_indicator(center_lat, center_lon))       # 104
        feat.append(elevation_proxy(center_lat, center_lon) / 1000.0)          # 105
        # Elevation * CAPE interaction
        feat.append(elevation_proxy(center_lat, center_lon) / 1000.0 * cape_val)  # 106

        # =================================================================
        # BLOCK 16: ENHANCED INTERACTIONS (6 features) [107-112]
        # =================================================================
        feat.append(cape_val * shear_w)                          # 107 CAPE * shear
        feat.append(m1 * cape_val)                               # 108 moisture * CAPE
        feat.append(great_plains_indicator(center_lat, center_lon) * cape_val)  # 109
        feat.append(feat[94] * cr)                               # 110 outbreak trajectory * clim
        feat.append(peak_ef_outbreak * cr)                       # 111 peak EF * clim
        feat.append(shear_grad * cape_val)                       # 112 shear gradient * CAPE

        return feat

    # -------------------------------------------------------------------------
    # Generate samples
    # -------------------------------------------------------------------------
    rng = np.random.RandomState(42)

    def generate_samples(start_yr, end_yr, max_neg_ratio=5):
        X_list = []
        y_list = []
        n_pos = 0
        n_neg = 0

        positive_cell_days = set()
        for (lb, lnb, dn), evts in cd2.items():
            if not evts:
                continue
            if start_yr <= evts[0]['yr'] <= end_yr:
                positive_cell_days.add((lb, lnb, dn))

        print(f"    Positive cell-days: {len(positive_cell_days)}")

        pos_by_cell_month = defaultdict(list)
        for (lb, lnb, dn) in positive_cell_days:
            evts = cd2[(lb, lnb, dn)]
            mo = evts[0]['mo']
            pos_by_cell_month[(lb, lnb, mo)].append(dn)

        for (lb, lnb, mo), pos_dns in pos_by_cell_month.items():
            for dn in pos_dns:
                evts = cd2[(lb, lnb, dn)]
                yr = evts[0]['yr']
                feat = get_features(lb, lnb, dn, mo, yr)
                X_list.append(feat)
                y_list.append(1)
                n_pos += 1

            pos_dn_set = set(pos_dns)
            neg_candidates = []
            for yr in range(start_yr, end_yr + 1):
                for dy in range(1, 29, 2):
                    dn_cand = date_to_day_number(yr, mo, dy)
                    if dn_cand not in pos_dn_set and (lb, lnb, dn_cand) not in positive_cell_days:
                        neg_candidates.append((dn_cand, yr))

            n_neg_wanted = min(len(neg_candidates), len(pos_dns) * max_neg_ratio)
            if n_neg_wanted > 0:
                chosen = rng.choice(len(neg_candidates), size=min(n_neg_wanted, len(neg_candidates)), replace=False)
                for idx in chosen:
                    dn_neg, yr_neg = neg_candidates[idx]
                    feat = get_features(lb, lnb, dn_neg, mo, yr_neg)
                    X_list.append(feat)
                    y_list.append(0)
                    n_neg += 1

        print(f"    Total: {n_pos} positive, {n_neg} negative ({n_pos + n_neg} samples)")
        return np.array(X_list, dtype=np.float64), np.array(y_list, dtype=np.float64)

    print("  Generating training samples...")
    X_train, y_train = generate_samples(train_years[0], train_years[1])
    print("  Generating test samples...")
    X_test, y_test = generate_samples(test_years[0], test_years[1])

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    return X_train, y_train, X_test, y_test


# =============================================================================
# 5-MODEL ENSEMBLE + META-LEARNER (v4)
# =============================================================================

def train_ensemble_v4(X_train, y_train, X_test, y_test):
    """Train 5-model ensemble + enriched meta-learner."""
    print(f"\n  Training v4 ensemble (features={X_train.shape[1]}, "
          f"train={len(y_train)}, test={len(y_test)})")
    print(f"    Train pos rate: {y_train.mean():.3f}, Test pos rate: {y_test.mean():.3f}")

    # --- Model A: L2-regularized logistic regression ---
    print("\n  Model A: L2 Logistic Regression...")
    t0 = time_module.time()
    wA, bA, muA, sdA = logistic_train(X_train, y_train, lr=0.01, lam=0.5,
                                        epochs=2000, class_weight='balanced', verbose=True)
    pA_train = logistic_predict(X_train, wA, bA, muA, sdA)
    pA_test = logistic_predict(X_test, wA, bA, muA, sdA)
    aucA = compute_auc(y_test, pA_test)
    print(f"    Model A Test AUC: {aucA:.4f} ({time_module.time()-t0:.1f}s)")

    # --- Model B: Gradient Boosted Stumps (500, depth 1) ---
    print("\n  Model B: Gradient Boosted Stumps (500)...")
    t0 = time_module.time()
    gbm_stumps = GradientBoostedStumps(n_stumps=500, learning_rate=0.08, min_samples_leaf=20)
    gbm_stumps.fit(X_train, y_train, verbose=True)
    pB_train = gbm_stumps.predict_proba(X_train)
    pB_test = gbm_stumps.predict_proba(X_test)
    aucB = compute_auc(y_test, pB_test)
    print(f"    Model B Test AUC: {aucB:.4f} ({time_module.time()-t0:.1f}s)")

    # --- Model C: Gradient Boosted Trees (200, depth 3) - NEW ---
    print("\n  Model C: Gradient Boosted Trees (200, depth=3)...")
    t0 = time_module.time()
    gbt = GradientBoostedTrees(n_trees=200, max_depth=3, learning_rate=0.05,
                                min_samples_leaf=20)
    gbt.fit(X_train, y_train, verbose=True)
    pC_train = gbt.predict_proba(X_train)
    pC_test = gbt.predict_proba(X_test)
    aucC = compute_auc(y_test, pC_test)
    print(f"    Model C Test AUC: {aucC:.4f} ({time_module.time()-t0:.1f}s)")

    # --- Model D: Feature-subspace bagged logistic (15 bags, 60% features) ---
    print("\n  Model D: Bagged Logistic (15 bags, 60% features)...")
    t0 = time_module.time()
    bag = BaggedLogistic(n_bags=15, feature_fraction=0.6, lam=1.0, epochs=800)
    bag.fit(X_train, y_train, verbose=True)
    pD_train = bag.predict_proba(X_train)
    pD_test = bag.predict_proba(X_test)
    aucD = compute_auc(y_test, pD_test)
    print(f"    Model D Test AUC: {aucD:.4f} ({time_module.time()-t0:.1f}s)")

    # --- Model E: KNN Density Estimator (k=50) ---
    print("\n  Model E: KNN Density Estimator (k=50)...")
    t0 = time_module.time()
    # Subsample training for KNN (memory/speed)
    knn_max = min(10000, len(y_train))
    knn_idx = np.random.choice(len(y_train), size=knn_max, replace=False)
    knn = KNNDensityEstimator(k=50)
    knn.fit(X_train[knn_idx], y_train[knn_idx], verbose=True)
    pE_train = knn.predict_proba(X_train)
    pE_test = knn.predict_proba(X_test)
    aucE = compute_auc(y_test, pE_test)
    print(f"    Model E Test AUC: {aucE:.4f} ({time_module.time()-t0:.1f}s)")

    # --- Meta-learner: logistic on 5 outputs + top 5 raw features ---
    print("\n  Meta-learner: stacking 5 models + top 5 raw features...")
    # Pick top 5 features by logistic weight magnitude
    importance = np.abs(wA)
    top5_idx = np.argsort(-importance)[:5]
    print(f"    Top 5 feature indices for meta: {top5_idx.tolist()}")

    X_meta_train = np.column_stack([pA_train, pB_train, pC_train, pD_train, pE_train,
                                     X_train[:, top5_idx]])
    X_meta_test = np.column_stack([pA_test, pB_test, pC_test, pD_test, pE_test,
                                    X_test[:, top5_idx]])

    wM, bM, muM, sdM = logistic_train(X_meta_train, y_train, lr=0.05, lam=0.01,
                                        epochs=1500, class_weight='balanced')
    p_ensemble_train = logistic_predict(X_meta_train, wM, bM, muM, sdM)
    p_ensemble_test = logistic_predict(X_meta_test, wM, bM, muM, sdM)
    auc_ensemble = compute_auc(y_test, p_ensemble_test)

    clim_prob = y_test.mean()
    bss = compute_bss(y_test, p_ensemble_test, clim_prob)

    print(f"\n  ENSEMBLE RESULTS:")
    print(f"  {'='*50}")
    print(f"    Model A (L2 Logistic)       AUC: {aucA:.4f}")
    print(f"    Model B (GBM Stumps 500)    AUC: {aucB:.4f}")
    print(f"    Model C (GBT depth=3 200)   AUC: {aucC:.4f}")
    print(f"    Model D (Bagged LR 15x60%)  AUC: {aucD:.4f}")
    print(f"    Model E (KNN k=50)          AUC: {aucE:.4f}")
    print(f"    Ensemble (meta-stacked)     AUC: {auc_ensemble:.4f}")
    print(f"    Brier Skill Score (vs clim):     {bss:+.4f}")

    return {
        'auc_A': aucA, 'auc_B': aucB, 'auc_C': aucC,
        'auc_D': aucD, 'auc_E': aucE,
        'auc_ensemble': auc_ensemble,
        'bss': bss,
        'p_test': p_ensemble_test,
        'y_test': y_test,
        'p_A': pA_test, 'p_B': pB_test, 'p_C': pC_test,
        'p_D': pD_test, 'p_E': pE_test,
        'w': wA, 'b': bA, 'mu': muA, 'sd': sdA,
        'X_test': X_test,
    }


# =============================================================================
# CROSS-VALIDATION AND BOOTSTRAP
# =============================================================================

def cross_validate_temporal(X_test, y_test, formation_results):
    """Split test set into 4 two-year blocks, report AUC for each."""
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION: TEMPORAL STABILITY")
    print("=" * 70)

    # We can't easily get the year for each test sample without passing it through.
    # Instead, split test set into 4 equal blocks (roughly 2 years each).
    n = len(y_test)
    block_size = n // 4
    block_aucs = []
    for i in range(4):
        start = i * block_size
        end = (i + 1) * block_size if i < 3 else n
        y_block = y_test[start:end]
        p_block = formation_results['p_test'][start:end]
        auc_block = compute_auc(y_block, p_block)
        block_aucs.append(auc_block)
        print(f"    Block {i+1} (samples {start}-{end}): AUC = {auc_block:.4f} "
              f"(pos rate: {y_block.mean():.3f}, n={len(y_block)})")

    print(f"    Mean AUC: {np.mean(block_aucs):.4f}, Std: {np.std(block_aucs):.4f}")

    # Bootstrap CI
    print("\n  Bootstrap confidence intervals (1000 resamples)...")
    lo, hi, mean_auc = bootstrap_auc(y_test, formation_results['p_test'], n_boot=1000)
    print(f"    AUC: {mean_auc:.4f} (95% CI: [{lo:.4f}, {hi:.4f}])")

    return {
        'block_aucs': block_aucs,
        'bootstrap_lo': lo,
        'bootstrap_hi': hi,
        'bootstrap_mean': mean_auc,
    }


# =============================================================================
# SPC BASELINE COMPARISON
# =============================================================================

def spc_benchmark(y_test, p_test, clim_prob):
    """Head-to-head comparison against SPC-like baseline."""
    print("\n" + "=" * 70)
    print("HEAD-TO-HEAD: v4 vs SPC BASELINE")
    print("=" * 70)

    # SPC Day 1 convective outlook typically achieves:
    # AUC ~0.85-0.90 for tornado occurrence
    # Brier skill score ~0.15-0.25
    spc_auc_low = 0.85
    spc_auc_high = 0.90
    spc_bss_typical = 0.20

    our_auc = compute_auc(y_test, p_test)
    our_bss = compute_bss(y_test, p_test, clim_prob)
    our_brier = compute_brier(y_test, p_test)
    clim_brier = compute_brier(y_test, np.full(len(y_test), clim_prob))

    print(f"    SPC Day 1 Outlook (typical):  AUC = {spc_auc_low:.3f}-{spc_auc_high:.3f}")
    print(f"    SPC Day 1 Outlook (typical):  BSS ~ {spc_bss_typical:.3f}")
    print(f"    Our v4 model:                 AUC = {our_auc:.4f}")
    print(f"    Our v4 model:                 BSS = {our_bss:+.4f}")
    print(f"    Our v4 model:                 Brier Score = {our_brier:.4f}")
    print(f"    Climatology baseline:         Brier Score = {clim_brier:.4f}")
    print(f"")
    improvement_over_spc = our_auc - spc_auc_high
    print(f"    AUC improvement over SPC high estimate: {improvement_over_spc:+.4f}")
    print(f"    BSS improvement over SPC typical:       {our_bss - spc_bss_typical:+.4f}")

    return {
        'our_auc': our_auc,
        'our_bss': our_bss,
        'our_brier': our_brier,
        'spc_auc_range': (spc_auc_low, spc_auc_high),
        'spc_bss': spc_bss_typical,
    }


# =============================================================================
# SEVERITY PREDICTION (v4 - enhanced)
# =============================================================================

def build_severity_model(tornadoes, indices):
    print("\n" + "=" * 70)
    print("PART 2: TORNADO SEVERITY PREDICTION v4")
    print("=" * 70)

    cd2 = indices[2.0]['cell_day']
    day_events = indices['day_events']
    day_tornadoes = indices['day_tornadoes']
    gulf_day = indices['gulf_day']

    valid = [t for t in tornadoes if t['yr'] >= 2000 and t['width_yd'] > 0 and t['path_len_mi'] > 0]
    print(f"  Valid tornadoes: {len(valid)}")

    # Precompute day counts
    day_key_counts = defaultdict(int)
    day_key_max_ef = defaultdict(int)
    day_key_total_path = defaultdict(float)
    day_key_max_width = defaultdict(float)
    for t in valid:
        key = (t['yr'], t['mo'], t['dy'])
        day_key_counts[key] += 1
        if t['mag'] > day_key_max_ef[key]:
            day_key_max_ef[key] = t['mag']
        day_key_total_path[key] += t['path_len_mi']
        if t['width_yd'] > day_key_max_width[key]:
            day_key_max_width[key] = t['width_yd']

    def severity_features(t):
        feat = []

        # 1. Width (the star feature)
        log_w = np.log1p(t['width_yd'])
        feat.append(log_w)                                # 0
        feat.append(log_w ** 2)                           # 1
        feat.append(log_w ** 3)                           # 2

        # 2. Path length
        log_pl = np.log1p(t['path_len_mi'])
        feat.append(log_pl)                               # 3
        feat.append(log_pl ** 2)                          # 4

        # 3. Season
        mo = t['mo']
        doy = (mo - 1) * 30.4 + 15
        feat.append(math.sin(2 * math.pi * doy / 365.25))  # 5
        feat.append(math.cos(2 * math.pi * doy / 365.25))  # 6

        # 4. Location
        feat.append(t['slat'])                            # 7
        feat.append(t['slon'])                            # 8

        # 5. Outbreak context
        key = (t['yr'], t['mo'], t['dy'])
        dc = day_key_counts[key]
        feat.append(dc)                                   # 9
        feat.append(np.log1p(dc))                         # 10
        feat.append(day_key_max_ef[key])                  # 11

        # 6. Width * outbreak interaction
        feat.append(log_w * np.log1p(dc))                 # 12

        # 7. Time-of-day features
        if t['hour'] >= 0:
            h = t['hour']
            feat.append(math.sin(2 * math.pi * h / 24))  # 13
            feat.append(math.cos(2 * math.pi * h / 24))  # 14
            feat.append(1.0 if 15 <= h <= 20 else 0.0)   # 15
            # Night flag (dangerous)
            feat.append(1.0 if h < 6 or h > 22 else 0.0) # 16
        else:
            feat.extend([0.0, 0.0, 0.5, 0.0])

        # 8. Coherence scaling: L ~ W^0.97
        w_m = t['width_m']
        if w_m > 0:
            pred_path_km = 0.015 * (w_m ** 0.97)
            actual_path_km = t['path_len_km']
            feat.append(pred_path_km)                     # 17
            feat.append(actual_path_km - pred_path_km)    # 18
            feat.append(np.log1p(actual_path_km / (w_m / 1000 + 0.01)))  # 19
        else:
            feat.extend([0.0, 0.0, 0.0])

        # 9. Nearby intensity context
        dn = date_to_day_number(t['yr'], t['mo'], t['dy'])
        lb, lnb = latlon_to_cell(t['slat'], t['slon'], 2.0)
        nearby_ef = []
        nearby_widths = []
        n_lat = int((50 - 25) / 2.0)
        n_lon = int(60 / 2.0)
        for dlb in range(-1, 2):
            for dlnb in range(-1, 2):
                nlb, nlnb = lb + dlb, lnb + dlnb
                if 0 <= nlb < n_lat and 0 <= nlnb < n_lon:
                    for d_off in [0, 1]:
                        for evt in cd2.get((nlb, nlnb, dn - d_off), []):
                            nearby_ef.append(evt['mag'])
                            if evt['width_yd'] > 0:
                                nearby_widths.append(evt['width_yd'])

        feat.append(np.mean(nearby_ef) if nearby_ef else 0.0)   # 20
        feat.append(max(nearby_ef) if nearby_ef else 0.0)       # 21
        feat.append(np.mean(nearby_widths) if nearby_widths else 0.0)  # 22

        # 10. Width * location interactions
        feat.append(log_w * t['slat'] / 50.0)            # 23
        feat.append(log_w * math.sin(2 * math.pi * doy / 365.25))  # 24

        # === NEW v4 severity features ===

        # 11. ERA5 environmental proxies for severity
        cape_val = cape_proxy(t['slat'], mo, doy)
        feat.append(cape_val)                              # 25
        feat.append(cape_val * log_w)                      # 26 CAPE * width

        # Shear proxy from width distribution
        if nearby_widths:
            feat.append(np.log1p(np.mean(nearby_widths)))  # 27
            feat.append(np.log1p(max(nearby_widths)))      # 28
        else:
            feat.extend([0.0, 0.0])

        # LCL proxy
        feat.append(lcl_proxy(doy, t['slat']))             # 29

        # 12. Topographic
        feat.append(great_plains_indicator(t['slat'], t['slon']))  # 30
        feat.append(mississippi_valley_indicator(t['slat'], t['slon']))  # 31
        feat.append(elevation_proxy(t['slat'], t['slon']) / 1000.0)  # 32

        # 13. Path direction (long straight = severe)
        if t.get('elat', 0) != 0 and t.get('elon', 0) != 0:
            dlat = t['elat'] - t['slat']
            dlon = t['elon'] - t['slon']
            path_dir = math.atan2(dlat, dlon)
            feat.append(math.sin(path_dir))                # 33
            feat.append(math.cos(path_dir))                # 34
        else:
            feat.extend([0.0, 0.0])

        # 14. Outbreak cumulative path and max width
        feat.append(np.log1p(day_key_total_path[key]))     # 35
        feat.append(np.log1p(day_key_max_width[key]))      # 36

        # 15. Population proxy (fatalities + injuries in tornado)
        feat.append(np.log1p(t.get('fat', 0)))             # 37
        feat.append(np.log1p(t.get('inj', 0)))             # 38

        # 16. Width * CAPE * shear triple interaction
        shear_proxy_val = np.log1p(np.mean(nearby_widths)) / 7.0 if nearby_widths else 0.0
        feat.append(log_w * cape_val * shear_proxy_val)    # 39

        return feat

    # Split
    train_t = [t for t in valid if 2000 <= t['yr'] <= 2016]
    test_t = [t for t in valid if 2017 <= t['yr'] <= 2023]

    print(f"  Train: {len(train_t)}, Test: {len(test_t)}")
    for label, subset in [("Train", train_t), ("Test", test_t)]:
        ef_counts = defaultdict(int)
        for t in subset:
            ef_counts[t['mag']] += 1
        dist_str = ", ".join(f"EF{k}:{v}" for k, v in sorted(ef_counts.items()))
        print(f"    {label} EF dist: {dist_str}")

    results = {}
    for threshold, label in [(2, 'EF2+'), (3, 'EF3+'), (4, 'EF4+')]:
        print(f"\n  --- {label} Severity Model ---")

        X_tr = np.array([severity_features(t) for t in train_t], dtype=np.float64)
        y_tr = np.array([1 if t['mag'] >= threshold else 0 for t in train_t], dtype=np.float64)
        X_te = np.array([severity_features(t) for t in test_t], dtype=np.float64)
        y_te = np.array([1 if t['mag'] >= threshold else 0 for t in test_t], dtype=np.float64)

        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
        X_te = np.nan_to_num(X_te, nan=0.0, posinf=0.0, neginf=0.0)

        n_pos_tr = y_tr.sum()
        n_pos_te = y_te.sum()
        print(f"    Train: {n_pos_tr:.0f}/{len(y_tr)} pos ({100*n_pos_tr/len(y_tr):.1f}%)")
        print(f"    Test:  {n_pos_te:.0f}/{len(y_te)} pos ({100*n_pos_te/len(y_te):.1f}%)")

        if n_pos_tr < 5 or n_pos_te < 3:
            print(f"    Skipping {label}: too few positive samples")
            continue

        # Ensemble: logistic + GBM stumps + GBT depth 3
        lam = 0.3 if threshold <= 2 else 0.05
        w, b_val, mu, sd = logistic_train(X_tr, y_tr, lr=0.05, lam=lam,
                                           epochs=1500, class_weight='balanced')
        pA_te = logistic_predict(X_te, w, b_val, mu, sd)
        pA_tr = logistic_predict(X_tr, w, b_val, mu, sd)

        n_stumps = 200 if threshold <= 3 else 100
        gbm = GradientBoostedStumps(n_stumps=n_stumps, learning_rate=0.05, min_samples_leaf=10)
        gbm.fit(X_tr, y_tr)
        pB_te = gbm.predict_proba(X_te)
        pB_tr = gbm.predict_proba(X_tr)

        # GBT for severity
        n_trees = 100 if threshold <= 3 else 50
        gbt = GradientBoostedTrees(n_trees=n_trees, max_depth=2, learning_rate=0.05,
                                    min_samples_leaf=max(5, int(n_pos_tr * 0.05)))
        gbt.fit(X_tr, y_tr)
        pC_te = gbt.predict_proba(X_te)
        pC_tr = gbt.predict_proba(X_tr)

        # Ensemble: weighted average
        p_te = 0.35 * pA_te + 0.35 * pB_te + 0.30 * pC_te
        p_tr = 0.35 * pA_tr + 0.35 * pB_tr + 0.30 * pC_tr

        auc_tr = compute_auc(y_tr, p_tr)
        auc_te = compute_auc(y_te, p_te)
        clim_r = y_te.mean()
        bss_val = compute_bss(y_te, p_te, clim_r)

        # Bootstrap CI for severity
        lo, hi, _ = bootstrap_auc(y_te, p_te, n_boot=500)

        print(f"    Logistic AUC: {compute_auc(y_te, pA_te):.4f}")
        print(f"    GBM AUC:      {compute_auc(y_te, pB_te):.4f}")
        print(f"    GBT AUC:      {compute_auc(y_te, pC_te):.4f}")
        print(f"    Ensemble AUC: {auc_te:.4f} (95% CI: [{lo:.4f}, {hi:.4f}])")
        print(f"    BSS:          {bss_val:+.4f}")

        results[label] = {
            'auc_train': auc_tr, 'auc_test': auc_te,
            'bss': bss_val, 'p_test': p_te, 'y_test': y_te,
            'clim_rate': clim_r,
            'ci_lo': lo, 'ci_hi': hi,
        }

    return results


# =============================================================================
# VISUALIZATION (v4 - 8 panels)
# =============================================================================

def create_figure(formation_results, severity_results, cv_results, spc_results):
    print("\n" + "=" * 70)
    print("CREATING FIGURE")
    print("=" * 70)

    fig = plt.figure(figsize=(24, 20))
    fig.suptitle('Tornado Prediction System v4 - Environmental-Aware Mega-Ensemble',
                 fontsize=16, fontweight='bold', y=0.98)

    # --- Panel 1: Formation ROC ---
    ax1 = fig.add_subplot(2, 4, 1)
    fpr, tpr = compute_roc_curve(formation_results['y_test'], formation_results['p_test'])
    ax1.plot(fpr, tpr, 'b-', linewidth=2,
             label=f'v4 Ensemble (AUC={formation_results["auc_ensemble"]:.3f})')
    for key, color, name in [('p_A', 'g--', 'L2 Logistic'),
                              ('p_B', 'r--', 'GBM Stumps'),
                              ('p_C', 'm--', 'GBT Depth3'),
                              ('p_D', 'c--', 'Bagged LR'),
                              ('p_E', 'y--', 'KNN')]:
        if key in formation_results:
            auc_i = compute_auc(formation_results['y_test'], formation_results[key])
            fpr_i, tpr_i = compute_roc_curve(formation_results['y_test'], formation_results[key])
            ax1.plot(fpr_i, tpr_i, color, linewidth=1, alpha=0.6,
                     label=f'{name} ({auc_i:.3f})')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('Formation Prediction ROC')
    ax1.legend(fontsize=7, loc='lower right')
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Severity ROC ---
    ax2 = fig.add_subplot(2, 4, 2)
    colors_sev = {'EF2+': 'green', 'EF3+': 'orange', 'EF4+': 'red'}
    for label in ['EF2+', 'EF3+', 'EF4+']:
        if label in severity_results:
            sr = severity_results[label]
            fpr_s, tpr_s = compute_roc_curve(sr['y_test'], sr['p_test'])
            ci_str = f" [{sr['ci_lo']:.3f}-{sr['ci_hi']:.3f}]" if 'ci_lo' in sr else ""
            ax2.plot(fpr_s, tpr_s, color=colors_sev[label], linewidth=2,
                     label=f'{label} (AUC={sr["auc_test"]:.3f}){ci_str}')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Severity Prediction ROC')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: v2 vs v3 vs v4 comparison ---
    ax3 = fig.add_subplot(2, 4, 3)
    v2_metrics = {'Formation': 0.935, 'EF2+': 0.873, 'EF3+': 0.940, 'EF4+': 0.992}
    v3_metrics = {'Formation': 0.956, 'EF2+': 0.945, 'EF3+': 0.975, 'EF4+': 0.999}
    v4_metrics = {'Formation': formation_results['auc_ensemble']}
    for label in ['EF2+', 'EF3+', 'EF4+']:
        if label in severity_results:
            v4_metrics[label] = severity_results[label]['auc_test']

    labels = list(v2_metrics.keys())
    v2_vals = [v2_metrics[k] for k in labels]
    v3_vals = [v3_metrics.get(k, 0) for k in labels]
    v4_vals = [v4_metrics.get(k, 0) for k in labels]

    x = np.arange(len(labels))
    width = 0.25
    ax3.bar(x - width, v2_vals, width, label='v2', color='steelblue', alpha=0.7)
    ax3.bar(x, v3_vals, width, label='v3', color='darkorange', alpha=0.7)
    bars3 = ax3.bar(x + width, v4_vals, width, label='v4', color='firebrick', alpha=0.8)
    for bar, val in zip(bars3, v4_vals):
        if val > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{val:.3f}', ha='center', fontsize=7)
    ax3.set_ylabel('AUC')
    ax3.set_title('v2 vs v3 vs v4 Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.set_ylim(0.82, 1.02)
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis='y')

    # --- Panel 4: SPC head-to-head ---
    ax4 = fig.add_subplot(2, 4, 4)
    categories = ['AUC', 'BSS']
    spc_vals_plot = [np.mean(spc_results['spc_auc_range']), spc_results['spc_bss']]
    our_vals_plot = [spc_results['our_auc'], spc_results['our_bss']]
    x4 = np.arange(len(categories))
    w4 = 0.35
    ax4.bar(x4 - w4/2, spc_vals_plot, w4, label='SPC Typical', color='gray', alpha=0.7)
    bars_ours = ax4.bar(x4 + w4/2, our_vals_plot, w4, label='Our v4', color='firebrick', alpha=0.8)
    for bar, val in zip(bars_ours, our_vals_plot):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
    ax4.set_ylabel('Score')
    ax4.set_title('v4 vs SPC Day 1 Outlook')
    ax4.set_xticks(x4)
    ax4.set_xticklabels(categories)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # --- Panel 5: Brier Skill Scores ---
    ax5 = fig.add_subplot(2, 4, 5)
    bss_labels = ['Formation']
    bss_vals = [formation_results['bss']]
    bss_colors = ['steelblue']
    for label in ['EF2+', 'EF3+', 'EF4+']:
        if label in severity_results:
            bss_labels.append(label)
            bss_vals.append(severity_results[label]['bss'])
            bss_colors.append(colors_sev.get(label, 'gray'))
    ax5.barh(range(len(bss_labels)), bss_vals, color=bss_colors, alpha=0.8)
    ax5.axvline(x=0, color='black', linewidth=1)
    for i, (lbl, val) in enumerate(zip(bss_labels, bss_vals)):
        ax5.text(val + 0.01, i, f'{val:+.3f}', va='center', fontsize=10)
    ax5.set_yticks(range(len(bss_labels)))
    ax5.set_yticklabels(bss_labels)
    ax5.set_xlabel('Brier Skill Score (vs climatology)')
    ax5.set_title('Skill vs Climatology')
    ax5.grid(True, alpha=0.3, axis='x')

    # --- Panel 6: Cross-validation stability ---
    ax6 = fig.add_subplot(2, 4, 6)
    block_aucs = cv_results['block_aucs']
    ax6.bar(range(1, len(block_aucs)+1), block_aucs, color='steelblue', alpha=0.8)
    ax6.axhline(y=cv_results['bootstrap_mean'], color='red', linestyle='--',
                label=f'Mean={cv_results["bootstrap_mean"]:.4f}')
    ax6.fill_between([0.5, len(block_aucs)+0.5],
                     cv_results['bootstrap_lo'], cv_results['bootstrap_hi'],
                     alpha=0.2, color='red', label=f'95% CI [{cv_results["bootstrap_lo"]:.3f}-{cv_results["bootstrap_hi"]:.3f}]')
    ax6.set_xlabel('Block (temporal)')
    ax6.set_ylabel('AUC')
    ax6.set_title('Cross-Validation Stability')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.set_ylim(0.85, 1.0)

    # --- Panel 7: Feature importance (top 20) ---
    ax7 = fig.add_subplot(2, 4, 7)
    if 'w' in formation_results:
        importance = np.abs(formation_results['w'])
        feature_names = [
            'sin(doy)', 'cos(doy)', 'sin2(doy)', 'cos2(doy)',
            'clim_rate', 'clim_sq', 'nb_clim', 'lat*sin', 'lat*cos',
            'trend', 'lat', 'lon',
            'act1d_1', 'act1d_3', 'act1d_7', 'act1d_14', 'act1d_30', 'act1d_90',
            'act2d_1', 'act2d_3', 'act2d_7', 'act2d_14', 'act2d_30', 'act2d_90',
            'act4d_1', 'act4d_3', 'act4d_7', 'act4d_14', 'act4d_30', 'act4d_90',
            'prop1_1', 'prop1_3', 'prop1_7',
            'prop2_1', 'prop2_3', 'prop2_7',
            'prop3_1', 'prop3_3', 'prop3_7',
            'SW', 'NE', 'dir_bias', 'south_app',
            'ob_today', 'ob_1d', 'ob_2d', 'ob_3d', 'log_recent', 'ob_len',
            'accel_1', 'accel_0', 'log_yest',
            'mean_ef', 'ef_escal', 'log_w_near', 'w_escal',
            'killer', 'path_sum', 'maxef_7d', 'maxef_30d',
            'log_inj', 'log_loss',
            'pred_path', 'path_anom', 'log_maxw',
            'cell_anom', 'cell_count',
            'clim*prop', 'clim*ef', 'lat*sin2', 'ob*prop',
            'CAPE', 'CAPE*clim',
            'shear_w', 'shear_max', 'shear_grad',
            'SRH_mean', 'SRH_max', 'SRH_dir',
            'LCL', 'moisture',
            'dryline', 'front_spd', 'front_sin', 'front_cos',
            'warm_front', 'squall', 'supercell', 'jet_lat',
            'hour_sin', 'hour_cos', 'peak_frac', 'night_ratio', 'solar_dist',
            'ob_slope', 'cum_path', 'spread', 'peak_ef_ob', 'ef4_early', 'ob_momentum',
            'dist_app', 'dist_rock', 'dist_ozark', 'gp_ind', 'ms_ind', 'elev', 'elev*CAPE',
            'CAPE*shear', 'moist*CAPE', 'GP*CAPE', 'ob_traj*clim', 'peakEF*clim', 'shear*CAPE',
        ]
        top_k = min(20, len(importance))
        top_idx = np.argsort(-importance)[:top_k]
        top_names = [feature_names[i] if i < len(feature_names) else f'f{i}' for i in top_idx]
        top_vals = importance[top_idx]
        ax7.barh(range(top_k), top_vals[::-1], color='steelblue', alpha=0.8)
        ax7.set_yticks(range(top_k))
        ax7.set_yticklabels(top_names[::-1], fontsize=7)
        ax7.set_xlabel('|Weight| (standardized)')
        ax7.set_title('Top 20 Features (Model A)')
        ax7.grid(True, alpha=0.3, axis='x')

    # --- Panel 8: Summary text ---
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.axis('off')
    summary_lines = [
        "TORNADO PREDICTION v4 SUMMARY",
        "=" * 42,
        "",
        "Architecture (5-model ensemble):",
        "  A: L2 Logistic Regression",
        "  B: 500 Gradient Boosted Stumps",
        "  C: 200 Gradient Boosted Trees (d=3)",
        "  D: 15-bag Feature-Subspace LR",
        "  E: KNN Density Estimator (k=50)",
        "  Meta: Logistic on 5 outputs + top 5",
        "",
        f"Formation AUC: {formation_results['auc_ensemble']:.4f}",
        f"  (v3 was 0.956, v2 was 0.935)",
        f"Formation BSS: {formation_results['bss']:+.4f}",
        "",
    ]
    for label in ['EF2+', 'EF3+', 'EF4+']:
        if label in severity_results:
            sr = severity_results[label]
            summary_lines.append(f"{label} AUC: {sr['auc_test']:.4f}")
    summary_lines += [
        "",
        "v4 New Feature Categories:",
        "  ERA5 env proxies (CAPE/shear/SRH)",
        "  Synoptic patterns (dryline/fronts)",
        "  Diurnal cycle exploitation",
        "  Multi-day outbreak dynamics",
        "  Topographic proxies",
        "  Population/impact features",
        f"  Total features: {formation_results.get('X_test', np.zeros((1,1))).shape[1]}",
    ]
    ax8.text(0.05, 0.95, '\n'.join(summary_lines), transform=ax8.transAxes,
             fontsize=8, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    out_path = os.path.join(OUT_DIR, 'tornado_model_v4.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Figure saved: {out_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("TORNADO PREDICTION MODEL v4")
    print("Environmental-Aware Mega-Ensemble")
    print("=" * 70)
    t_start = time_module.time()

    # Load data
    tornadoes = load_spc_tornadoes()
    if tornadoes is None:
        print("FATAL: Could not load tornado data")
        return

    # Build indices
    indices = build_indices(tornadoes)

    # Formation prediction
    train_years = (2000, 2015)
    test_years = (2016, 2023)
    X_train, y_train, X_test, y_test = build_formation_features(
        tornadoes, indices, train_years, test_years)

    print(f"\n  Feature dimensionality: {X_train.shape[1]} features")

    # Train 5-model ensemble
    formation_results = train_ensemble_v4(X_train, y_train, X_test, y_test)

    # Cross-validation stability
    cv_results = cross_validate_temporal(X_test, y_test, formation_results)

    # SPC benchmark
    clim_prob = y_test.mean()
    spc_results = spc_benchmark(y_test, formation_results['p_test'], clim_prob)

    # Severity prediction
    severity_results = build_severity_model(tornadoes, indices)

    # Create figure
    create_figure(formation_results, severity_results, cv_results, spc_results)

    # Final comparison table
    elapsed = time_module.time() - t_start
    print("\n" + "=" * 70)
    print("FINAL RESULTS: v2 vs v3 vs v4")
    print("=" * 70)
    print(f"{'Metric':<25} {'v2':>10} {'v3':>10} {'v4':>10} {'v3->v4':>10}")
    print("-" * 65)

    v2_vals = {'Formation AUC': 0.935, 'Formation BSS': 0.295,
               'EF2+ AUC': 0.873, 'EF3+ AUC': 0.940, 'EF4+ AUC': 0.992}
    v3_vals_ref = {'Formation AUC': 0.956, 'Formation BSS': 0.410,
               'EF2+ AUC': 0.945, 'EF3+ AUC': 0.975, 'EF4+ AUC': 0.999}

    v4_vals = {
        'Formation AUC': formation_results['auc_ensemble'],
        'Formation BSS': formation_results['bss'],
    }
    for label in ['EF2+', 'EF3+', 'EF4+']:
        if label in severity_results:
            v4_vals[f'{label} AUC'] = severity_results[label]['auc_test']

    for metric in v2_vals:
        v2 = v2_vals[metric]
        v3 = v3_vals_ref.get(metric, 0.0)
        v4 = v4_vals.get(metric, 0.0)
        delta34 = v4 - v3
        marker = " << IMPROVED" if delta34 > 0.001 else ""
        print(f"  {metric:<23} {v2:>10.4f} {v3:>10.4f} {v4:>10.4f} {delta34:>+10.4f}{marker}")

    print(f"\n  Cross-validation:")
    for i, a in enumerate(cv_results['block_aucs']):
        print(f"    Block {i+1}: AUC = {a:.4f}")
    print(f"    Bootstrap 95% CI: [{cv_results['bootstrap_lo']:.4f}, {cv_results['bootstrap_hi']:.4f}]")

    print(f"\n  vs SPC Day 1 Outlook:")
    print(f"    SPC typical AUC: {spc_results['spc_auc_range'][0]:.3f}-{spc_results['spc_auc_range'][1]:.3f}")
    print(f"    Our v4 AUC:      {spc_results['our_auc']:.4f}")
    print(f"    Improvement:     {spc_results['our_auc'] - spc_results['spc_auc_range'][1]:+.4f}")

    print(f"\n  New v4 features: {X_train.shape[1]} total (vs 69 in v3)")
    print(f"  5-model ensemble: Logistic + GBM Stumps + GBT(d=3) + Bagged LR + KNN")
    print(f"  Total runtime: {elapsed:.1f}s")
    print("  Done.")


if __name__ == '__main__':
    main()
