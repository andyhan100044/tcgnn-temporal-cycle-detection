"""Tests for baselines (experiments/baselines/)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

import numpy as np
import pytest
import torch

from experiments.baselines import (
    GCNBaseline, GATBaseline, TGNBaseline, DCRNNBaseline, GLASSBaseline,
    build_baseline, BASELINES,
)
from experiments.baselines.xgb import XGBoostCycleClassifier, cycle_to_features


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def toy_cycles():
    return [
        {"nodes": [0, 1, 2], "times": [1, 2, 3], "amounts": [10.0, 10.0, 10.0]},
        {"nodes": [3, 4, 5], "times": [5, 6, 7], "amounts": [20.0, 20.0, 20.0]},
        {"nodes": [0, 3, 6], "times": [10, 11, 12], "amounts": [30.0, 30.0, 30.0]},
    ]


@pytest.fixture
def toy_node_features():
    return torch.randn(7, 16)


@pytest.fixture
def toy_node_id_map():
    return {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}


# ---------------------------------------------------------------------------
# PyTorch baselines
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["GCN", "GAT", "TGN", "DCRNN", "GLASS"])
def test_baseline_forward_shape(name, toy_cycles, toy_node_features, toy_node_id_map):
    model = build_baseline(name, in_dim=16, hidden_dim=32)
    logits = model(toy_cycles, toy_node_features, toy_node_id_map)
    assert logits.shape == (3, 1), f"{name} returned shape {logits.shape}"
    assert torch.isfinite(logits).all(), f"{name} produced non-finite logits"


@pytest.mark.parametrize("name", ["GCN", "GAT", "TGN", "DCRNN", "GLASS"])
def test_baseline_backward_runs(name, toy_cycles, toy_node_features, toy_node_id_map):
    model = build_baseline(name, in_dim=16, hidden_dim=32)
    logits = model(toy_cycles, toy_node_features, toy_node_id_map)
    loss = logits.sum()
    loss.backward()
    grads_present = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters())
    assert grads_present, f"{name} did not produce gradients"


def test_unknown_baseline_raises():
    with pytest.raises(ValueError):
        build_baseline("UNKNOWN", in_dim=16)


# ---------------------------------------------------------------------------
# XGBoost baseline
# ---------------------------------------------------------------------------
def test_xgb_cycle_features_shape():
    c = {"nodes": [0, 1, 2], "times": [1, 2, 3], "amounts": [10, 10, 10]}
    f = cycle_to_features(c)
    assert f.shape == (9,)


def test_xgb_fit_predict_runs():
    cycles = [
        {"nodes": [0, 1, 2], "times": [1, 2, 3], "amounts": [10, 10, 10]},
        {"nodes": [3, 4, 5], "times": [5, 6, 7], "amounts": [20, 20, 20]},
        {"nodes": [0, 3, 6], "times": [10, 11, 12], "amounts": [30, 30, 30]},
    ]
    labels = np.array([1, 0, 1])
    clf = XGBoostCycleClassifier(n_estimators=10)
    clf.fit(cycles, labels)
    preds = clf.predict(cycles)
    assert preds.shape == (3,)
    proba = clf.predict_proba(cycles)
    assert proba.shape == (3, 2)
    assert (proba.sum(axis=1) - 1.0).max() < 1e-5


def test_xgb_perfect_separation():
    """Two cycles with very different amounts should be perfectly separable."""
    cycles = [
        {"nodes": [0, 1, 2], "times": [1, 2, 3], "amounts": [10, 10, 10]},
        {"nodes": [3, 4, 5], "times": [5, 6, 7], "amounts": [10, 10, 10]},
        {"nodes": [0, 3, 6], "times": [10, 11, 12], "amounts": [1000, 1000, 1000]},
        {"nodes": [1, 4, 0], "times": [15, 16, 17], "amounts": [2000, 2000, 2000]},
    ]
    labels = np.array([1, 1, 0, 0])  # low amount = suspicious
    clf = XGBoostCycleClassifier(n_estimators=20)
    clf.fit(cycles, labels)
    preds = clf.predict(cycles)
    assert (preds == labels).mean() >= 0.75  # at least 75% on training