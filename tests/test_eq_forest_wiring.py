"""The live earthquake scorer adopts the exported forest champion -- but only safely.

load_eq_forest() is the activation gate for the deployable accuracy champion: it
loads the forest ONLY when an export exists AND its feature indices fit the served
73-feature enhanced space, refusing any stale/mismatched export (which would
silently produce garbage). The forest is otherwise dormant.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def _load_scorer_module():
    # importing the scorer needs the earthquake ML stack; skip where unavailable
    pytest.importorskip("numpy")
    try:
        spec = importlib.util.spec_from_file_location(
            "fse_test", REPO / "scripts" / "fetch_and_score_earthquake.py")
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
    except Exception as exc:  # pragma: no cover - heavy deps absent
        pytest.skip(f"earthquake scorer not importable: {exc}")
    return m


def _forest_constants(feat0: int):
    scale = 1 << 20
    return {
        "spec": "omega-one/verifiable-forest/v1", "scale": scale, "op": "le",
        "classes": [0, 1],
        "feat": [feat0, -1, -1], "thr": [0.5, 0.0, 0.0],
        "left": [1, -1, -1], "right": [2, -1, -1],
        "value_fp": [[0, 0], [0, scale], [0, -scale]],
        "tree_root": [0], "default_left": [0, 0, 0], "base_fp": [0, 0],
    }


def test_no_export_means_dormant(tmp_path):
    m = _load_scorer_module()
    assert m.load_eq_forest(tmp_path) is None


def test_valid_export_is_adopted(tmp_path):
    m = _load_scorer_module()
    (tmp_path / "earthquake_forest_fp.json").write_text(json.dumps(_forest_constants(5)))
    scorer = m.load_eq_forest(tmp_path)
    assert scorer is not None
    assert scorer.raw_proba_one([0.0] * len(m.DEFINITIVE_EQ_FEATURE_NAMES)) > 0.5  # feat5<=0.5 -> +margin


def test_out_of_range_feature_is_refused(tmp_path):
    m = _load_scorer_module()
    # references feature index 99 but only 73 enhanced features exist -> refuse
    (tmp_path / "earthquake_forest_fp.json").write_text(json.dumps(_forest_constants(99)))
    assert m.load_eq_forest(tmp_path) is None
