// Gradient Boosted Tree inference for tornado prediction
// Written in Coherence Lang — compiled to WASM for browser/Cloudflare
//
// This replaces the Python GBT implementation with native .cl code.
// The trained model weights are loaded from JSON (exported by the Python trainer).
//
// Architecture: Binary classification GBT with sigmoid output.
// Each tree is a recursive decision tree. Prediction = sigmoid(init_pred + lr * sum(tree_preds)).

module hazardpulse.tornado.gbt;

import std.core {Option, Result, Ok, Err, Vec, HashMap}
import std.data.json {JsonValue, parse_json}
import std.math.arithmetic {exp, max, min}

/// A single node in a decision tree
enum TreeNode {
    Leaf { value: F64 },
    Split {
        feature_index: USize,
        threshold: F64,
        left: Box[TreeNode],
        right: Box[TreeNode]
    }
}

impl TreeNode {
    /// Predict a single sample by traversing the tree iteratively.
    /// No recursion — no stack overflow on deep trees.
    fn predict(self, features: &Vec[F64]) -> F64 @ L0 {
        let mut node = self;
        loop {
            match node {
                TreeNode::Leaf { value } => return value,
                TreeNode::Split { feature_index, threshold, left, right } => {
                    if features[feature_index] <= threshold {
                        node = *left;
                    } else {
                        node = *right;
                    }
                }
            }
        }
    }

    /// Deserialize a tree node from JSON
    fn from_json(json: &JsonValue) -> Result[Self, String] @ L0 {
        let obj = json.as_object().ok_or("Expected object for tree node")?;

        if let Some(leaf_val) = obj.get("val") {
            // Leaf node
            let value = leaf_val.as_f64().ok_or("Expected float for leaf value")?;
            Ok(TreeNode::Leaf { value })
        } else {
            // Split node
            let feat = obj.get("feat")
                .and_then(|v| v.as_usize())
                .ok_or("Missing 'feat' in split node")?;
            let thresh = obj.get("thresh")
                .and_then(|v| v.as_f64())
                .ok_or("Missing 'thresh' in split node")?;
            let left_json = obj.get("left")
                .ok_or("Missing 'left' in split node")?;
            let right_json = obj.get("right")
                .ok_or("Missing 'right' in split node")?;

            let left = TreeNode::from_json(left_json)?;
            let right = TreeNode::from_json(right_json)?;

            Ok(TreeNode::Split {
                feature_index: feat,
                threshold: thresh,
                left: Box::new(left),
                right: Box::new(right),
            })
        }
    }
}

/// Sigmoid activation function
fn sigmoid(x: F64) -> F64 @ L0 {
    let clamped = max(-10.0, min(10.0, x));
    1.0 / (1.0 + exp(-clamped))
}

/// A trained Gradient Boosted Tree ensemble
struct GBTModel {
    trees: Vec[TreeNode],
    init_pred: F64,
    learning_rate: F64,
    feature_names: Vec[String],
    n_features: USize,
    // Normalization parameters
    means: Vec[F64],
    stds: Vec[F64]
}

impl GBTModel {
    /// Load a trained model from JSON string
    fn from_json_str(json_str: &str) -> Result[Self, String] @ L0 {
        let json = parse_json(json_str)
            .map_err(|e| format!("JSON parse error: {}", e))?;

        let obj = json.as_object()
            .ok_or("Expected root object")?;

        let init_pred = obj.get("init_pred")
            .and_then(|v| v.as_f64())
            .ok_or("Missing init_pred")?;

        let learning_rate = obj.get("learning_rate")
            .and_then(|v| v.as_f64())
            .ok_or("Missing learning_rate")?;

        let feature_names: Vec[String] = obj.get("feature_names")
            .and_then(|v| v.as_array())
            .ok_or("Missing feature_names")?
            .iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();

        let n_features = feature_names.len();

        // Parse trees
        let trees_json = obj.get("trees")
            .and_then(|v| v.as_array())
            .ok_or("Missing trees array")?;

        let mut trees = Vec::with_capacity(trees_json.len());
        for tree_json in trees_json {
            let tree = TreeNode::from_json(tree_json)?;
            trees.push(tree);
        }

        // Parse normalization parameters
        let norm = obj.get("normalization")
            .and_then(|v| v.as_object())
            .ok_or("Missing normalization")?;

        let means: Vec[F64] = norm.get("means")
            .and_then(|v| v.as_array())
            .ok_or("Missing means")?
            .iter()
            .filter_map(|v| v.as_f64())
            .collect();

        let stds: Vec[F64] = norm.get("stds")
            .and_then(|v| v.as_array())
            .ok_or("Missing stds")?
            .iter()
            .filter_map(|v| v.as_f64())
            .collect();

        Ok(GBTModel {
            trees,
            init_pred,
            learning_rate,
            feature_names,
            n_features,
            means,
            stds,
        })
    }

    /// Normalize a feature vector using training statistics
    fn normalize(self, features: &Vec[F64]) -> Vec[F64] @ L0 {
        let mut normalized = Vec::with_capacity(features.len());
        for i in 0..features.len() {
            let std = if self.stds[i] < 1e-12 { 1.0 } else { self.stds[i] };
            normalized.push((features[i] - self.means[i]) / std);
        }
        normalized
    }

    /// Predict tornado probability for a single storm
    fn predict_proba(self, features: &Vec[F64]) -> F64 @ L0 {
        // Normalize features
        let norm_features = self.normalize(features);

        // Accumulate tree predictions
        let mut f = self.init_pred;
        for tree in &self.trees {
            f += self.learning_rate * tree.predict(&norm_features);
        }

        // Apply sigmoid
        sigmoid(f)
    }

    /// Predict probabilities for a batch of storms
    fn predict_batch(self, batch: &Vec[Vec[F64]]) -> Vec[F64] @ L0 {
        batch.iter()
            .map(|features| self.predict_proba(features))
            .collect()
    }

    /// Get the number of trees in the ensemble
    fn n_trees(self) -> USize @ L0 {
        self.trees.len()
    }

    /// Get feature name by index
    fn feature_name(self, index: USize) -> Option[&str] @ L0 {
        self.feature_names.get(index).map(|s| s.as_str())
    }
}

// ============================================================
// Tests
// ============================================================

#[test]
fn test_sigmoid() {
    assert_approx_eq!(sigmoid(0.0), 0.5, 1e-6);
    assert_approx_eq!(sigmoid(10.0), 0.9999, 1e-3);
    assert_approx_eq!(sigmoid(-10.0), 0.0001, 1e-3);
}

#[test]
fn test_leaf_prediction() {
    let leaf = TreeNode::Leaf { value: 0.42 };
    let features = vec![1.0, 2.0, 3.0];
    assert_approx_eq!(leaf.predict(&features), 0.42, 1e-6);
}

#[test]
fn test_split_prediction() {
    let tree = TreeNode::Split {
        feature_index: 0,
        threshold: 1.5,
        left: Box::new(TreeNode::Leaf { value: -0.3 }),
        right: Box::new(TreeNode::Leaf { value: 0.7 }),
    };

    // feature[0] = 1.0 <= 1.5 -> left -> -0.3
    assert_approx_eq!(tree.predict(&vec![1.0, 0.0]), -0.3, 1e-6);

    // feature[0] = 2.0 > 1.5 -> right -> 0.7
    assert_approx_eq!(tree.predict(&vec![2.0, 0.0]), 0.7, 1e-6);
}
