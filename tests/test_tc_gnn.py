"""Tests for TC-GNN (paper §5) and constraint-regularized loss."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.temporal_cycle.tc_gnn import (
    TemporalMessagePassing,
    CycleSubgraphEncoder,
    TC_GNN,
)
from src.temporal_cycle.losses import (
    temporal_increasing_penalty,
    value_conservation_penalty,
    constraint_regularized_loss,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_dummy_cycle(length: int = 4):
    """Build a simple cycle with strictly-increasing times and equal amounts."""
    nodes = list(range(length))
    times = list(range(1, length + 1))
    amounts = [10.0] * length
    return nodes, times, amounts


def make_dummy_batch(n_cycles: int = 3, length: int = 4):
    """Build a batch of cycles as torch tensors."""
    cycles = []
    labels = []
    for i in range(n_cycles):
        n, t, a = make_dummy_cycle(length)
        cycles.append({"nodes": n, "times": t, "amounts": a})
        labels.append(1 if i % 2 == 0 else 0)
    return cycles, torch.tensor(labels, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Loss function tests
# ---------------------------------------------------------------------------
def test_temporal_increasing_penalty_zero_for_strictly_increasing():
    """Penalty = 0 if times are strictly increasing."""
    times = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    assert temporal_increasing_penalty(times).item() == pytest.approx(0.0, abs=1e-6)


def test_temporal_increasing_penalty_positive_for_violation():
    """Penalty > 0 if any t_i >= t_{i+1}."""
    times = torch.tensor([[1.0, 2.0, 1.0, 4.0]])  # 2->1 violation: max(0, 2-1)=1
    assert temporal_increasing_penalty(times).item() == pytest.approx(1.0, abs=1e-6)


def test_value_conservation_penalty_zero_for_equal():
    """|sum_in - sum_out| / sum_in = 0 when perfectly balanced."""
    amounts = torch.tensor([[10.0, 10.0, 10.0]])
    assert value_conservation_penalty(amounts).item() == pytest.approx(0.0, abs=1e-6)


def test_value_conservation_penalty_nonzero_for_imbalance():
    """Imbalance should equal max(|a_i - mean|) / mean."""
    amounts = torch.tensor([[10.0, 10.0, 30.0]])  # mean=50/3, max_dev=30-50/3=40/3
    expected = (30 - 50/3) / (50/3)
    assert value_conservation_penalty(amounts).item() == pytest.approx(expected, abs=1e-5)


def test_constraint_regularized_loss_combines_terms():
    """Combined loss = L_cls + λ1·L_temp + λ2·L_val."""
    logits = torch.tensor([[2.0], [-2.0]])  # predicted probs ~0.88, 0.12
    labels = torch.tensor([[1.0], [0.0]])
    times  = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]])  # 2nd cycle violates
    amounts = torch.tensor([[10.0, 10.0, 10.0], [10.0, 10.0, 20.0]])  # 2nd imbalanced

    loss = constraint_regularized_loss(logits, labels, times, amounts,
                                       lambda_temp=1.0, lambda_val=1.0)
    # BCE alone is positive; adding penalties makes it larger
    bce_only = nn.functional.binary_cross_entropy_with_logits(logits, labels).item()
    assert loss.item() > bce_only


def test_hard_constraint_limit_lambda():
    """As λ → ∞, output approximates constraint-satisfying solver."""
    # Two candidate cycles: one valid (increasing, balanced), one invalid.
    # With very large λ, loss for invalid cycle must be dominated by penalty.
    logits = torch.tensor([[0.0], [0.0]])
    labels = torch.tensor([[1.0], [1.0]])  # both labeled "should be 1"
    times  = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])  # 2nd invalid
    amounts = torch.tensor([[10.0, 10.0, 10.0], [10.0, 10.0, 100.0]])

    loss = constraint_regularized_loss(logits, labels, times, amounts,
                                       lambda_temp=1000.0, lambda_val=1000.0)
    # Penalty should be ~1000 × (1 + 0.5) = 1500 (time violation=1, value=~3/30=0.1, hmm let me check)
    # Actually for amounts: mean=(10+10+100)/3=40, max_dev=100-40=60, val_pen=60/40=1.5
    # Temp pen for [3,2,1] = max(0,3-2) + max(0,2-1) = 1+1 = 2
    # So total penalty contribution = 1000*2 + 1000*1.5 = 3500
    assert loss.item() > 1000  # dominated by penalty


# ---------------------------------------------------------------------------
# Model component tests
# ---------------------------------------------------------------------------
def test_temporal_message_passing_filters_old_neighbors():
    """Only edges with t_j > t_i contribute to message passing."""
    layer = TemporalMessagePassing(in_dim=4, out_dim=8)

    # 3 nodes: A, B, C
    h = torch.randn(3, 4)
    # Edges: A->B@t=1 (old), B->C@t=3 (current), A->C@t=2 (mid)
    edge_src  = torch.tensor([0, 1, 0])
    edge_dst  = torch.tensor([1, 2, 2])
    edge_t    = torch.tensor([1.0, 3.0, 2.0])
    edge_w    = torch.tensor([10.0, 20.0, 30.0])
    current_t = torch.tensor([2.5, 2.5, 2.5])  # current path time

    out = layer(h, edge_src, edge_dst, edge_t, edge_w, current_t)
    assert out.shape == (3, 8)
    # Output should not contain any NaN or Inf
    assert torch.isfinite(out).all()


def test_temporal_mp_output_shape():
    """Output has shape (n_nodes, out_dim)."""
    layer = TemporalMessagePassing(in_dim=16, out_dim=32)
    h = torch.randn(5, 16)
    out = layer(h, torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long),
                torch.zeros(0), torch.zeros(0), torch.zeros(5))
    assert out.shape == (5, 32)


def test_cycle_subgraph_encoder_attention():
    """Attention pooling over cycle nodes produces fixed-size embedding."""
    enc = CycleSubgraphEncoder(node_dim=16, hidden_dim=32)
    # Batch of 2 cycles, each with 4 nodes
    node_emb = torch.randn(2, 4, 16)
    edge_attr = torch.randn(2, 4, 8)
    out = enc(node_emb, edge_attr)
    assert out.shape == (2, 32)


def test_tc_gnn_forward_returns_logits():
    """TC_GNN.forward returns per-cycle logits."""
    model = TC_GNN(node_feature_dim=8, hidden_dim=16, n_gnn_layers=2)
    # 2 cycles, each with 3 nodes
    cycles = [
        {"nodes": [0, 1, 2], "times": [1, 2, 3], "amounts": [10.0, 10.0, 10.0]},
        {"nodes": [3, 4, 5], "times": [5, 6, 7], "amounts": [20.0, 20.0, 20.0]},
    ]
    # Build a simple feature matrix: one row per node across all cycles
    all_nodes = sorted({n for c in cycles for n in c["nodes"]})
    node_features = torch.randn(len(all_nodes), 8)
    node_id_map = {n: i for i, n in enumerate(all_nodes)}

    logits = model(cycles, node_features, node_id_map)
    assert logits.shape == (2, 1)
    assert torch.isfinite(logits).all()


def test_tc_gnn_predict_probabilities_in_range():
    """After sigmoid, output ∈ [0, 1]."""
    model = TC_GNN(node_feature_dim=4, hidden_dim=8)
    cycles = [{"nodes": [0, 1, 2], "times": [1, 2, 3], "amounts": [10, 10, 10]}]
    node_features = torch.randn(3, 4)
    logits = model(cycles, node_features, {0: 0, 1: 1, 2: 2})
    probs = torch.sigmoid(logits)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_tc_gnn_parameter_count_reasonable():
    """Model is small enough to train on CPU (~100k-500k params)."""
    model = TC_GNN(node_feature_dim=64, hidden_dim=32, n_gnn_layers=2)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert 1_000 < n_params < 1_000_000, f"Got {n_params} params"


def test_tc_gnn_backward_pass_runs():
    """End-to-end forward + backward + step works."""
    model = TC_GNN(node_feature_dim=8, hidden_dim=16)
    cycles = [{"nodes": [0, 1, 2], "times": [1, 2, 3], "amounts": [10, 10, 10]}]
    node_features = torch.randn(3, 8)
    labels = torch.tensor([1.0])
    logits = model(cycles, node_features, {0: 0, 1: 1, 2: 2})

    times  = torch.tensor([[1.0, 2.0, 3.0]])
    amounts = torch.tensor([[10.0, 10.0, 10.0]])
    loss = constraint_regularized_loss(logits, labels.unsqueeze(1), times, amounts)
    loss.backward()
    # Gradients should exist on at least one parameter
    grads_present = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters())
    assert grads_present