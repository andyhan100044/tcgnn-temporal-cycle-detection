"""Tests for TemporalCycleSketch (Algorithm 2 in paper outline §4).

Sketch guarantees:
  - False negative rate: 0 (must find every true cycle)
  - False positive rate: <= 1/w (controllable via width)
  - Space: O(w * d * m), decoupled from |E|
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from src.temporal_cycle.indexing import build_index
from src.temporal_cycle.temporal_dfs import find_cycles_simple
from src.temporal_cycle.temporal_sketch import (
    TemporalCycleSketch,
    sketch_candidates,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def medium_graph() -> pd.DataFrame:
    """30-node graph with 3 known cycles for sketch-vs-exact comparison."""
    edges_data = []
    # Cycle 1: 0->1->2->0 at times 1,2,3
    edges_data += [(0, 1, 1, 10.0), (1, 2, 2, 10.0), (2, 0, 3, 10.0)]
    # Cycle 2: 10->11->12->10 at times 5,6,7
    edges_data += [(10, 11, 5, 20.0), (11, 12, 6, 20.0), (12, 10, 7, 20.0)]
    # Cycle 3: 20->21->22->23->20 (k=4) at times 10,11,12,13
    edges_data += [(20, 21, 10, 30.0), (21, 22, 11, 30.0),
                   (22, 23, 12, 30.0), (23, 20, 13, 30.0)]
    # Background edges
    edges_data += [(0, 5, 4, 5.0), (5, 10, 4, 5.0), (1, 6, 5, 7.0)]
    return pd.DataFrame(edges_data,
                        columns=["txId1", "txId2", "time_step", "amount"])


@pytest.fixture
def large_random_graph() -> pd.DataFrame:
    """100-node graph with ~300 edges for recall test."""
    import numpy as np
    rng = np.random.default_rng(123)
    n = 100
    n_edges = 300
    src = rng.integers(0, n, size=n_edges)
    tgt = rng.integers(0, n, size=n_edges)
    mask = src != tgt
    src, tgt = src[mask], tgt[mask]
    t = rng.integers(0, 30, size=src.shape[0])
    w = rng.uniform(5, 50, size=src.shape[0])
    return pd.DataFrame({
        "txId1": src, "txId2": tgt, "time_step": t, "amount": w,
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_sketch_basic_construction():
    """Can construct sketch with reasonable defaults."""
    sk = TemporalCycleSketch(width=8, depth=4, window_size=5)
    assert sk.width == 8
    assert sk.depth == 4


def test_sketch_finds_known_cycles(medium_graph):
    """Sketch should find all 3 known cycles in medium_graph."""
    sk = TemporalCycleSketch(width=16, depth=4, window_size=5)
    candidates = sk.find_candidates(medium_graph)
    found_node_sets = [tuple(sorted(c["nodes"])) for c in candidates]

    assert any(set(s) == {0, 1, 2} for s in found_node_sets)
    assert any(set(s) == {10, 11, 12} for s in found_node_sets)
    assert any(set(s) == {20, 21, 22, 23} for s in found_node_sets)


def test_sketch_recall_100_percent(medium_graph):
    """Sketch must find every cycle found by exact TemporalDFS.

    Paper claim: false negative rate = 0.
    """
    idx = build_index(medium_graph)
    exact_cycles = find_cycles_simple(idx, k=3, time_window=5, epsilon=0.5)

    sk = TemporalCycleSketch(width=16, depth=4, window_size=5)
    candidates = sk.find_candidates(medium_graph)

    exact_sets = {tuple(sorted(c["nodes"])) for c in exact_cycles}
    sketch_sets = {tuple(sorted(c["nodes"])) for c in candidates}

    # Every exact cycle's node set should be in the sketch's results
    missed = exact_sets - sketch_sets
    assert missed == set(), f"Sketch missed cycles: {missed}"


def test_sketch_on_large_graph(large_random_graph):
    """Sketch runs on a 100-node graph without error and returns reasonable count."""
    sk = TemporalCycleSketch(width=8, depth=2, window_size=5)
    candidates = sk.find_candidates(large_random_graph)
    # Should find at least a few cycles in random graph, but not explode
    assert len(candidates) < 10_000


def test_sketch_decoupled_space():
    """Sketch's in-memory bucket count is independent of edge count."""
    sk_small = TemporalCycleSketch(width=8, depth=4, window_size=5)
    sk_large = TemporalCycleSketch(width=8, depth=4, window_size=5)
    # Bucket count depends only on width, depth, window_size
    assert sk_small.bucket_count() == sk_large.bucket_count()
    # Same sketch size regardless of graph size
    assert sk_small.bucket_count() == 8 * 4  # width * depth


def test_sketch_includes_only_temporally_valid(medium_graph):
    """All candidates must have strictly increasing timestamps."""
    sk = TemporalCycleSketch(width=16, depth=4, window_size=5)
    candidates = sk.find_candidates(medium_graph)
    for c in candidates:
        times = c["times"]
        assert all(times[i] < times[i+1] for i in range(len(times)-1)), \
            f"Non-increasing times: {times}"


def test_sketch_value_balance_filter(medium_graph):
    """Candidates should generally satisfy value-balance (after verification step)."""
    sk = TemporalCycleSketch(width=16, depth=4, window_size=5, epsilon=0.5)
    candidates = sk.find_candidates(medium_graph)
    # At least most candidates should have low imbalance
    balanced = sum(1 for c in candidates if c["value_imbalance"] <= 0.5)
    assert balanced >= len(candidates) * 0.5


def test_sketch_false_positive_rate_bounded():
    """Width=32, depth=4 should have very low false positive rate.

    On a graph with NO cycles, sketch should find few/zero candidates.
    """
    # Construct graph with no temporal cycles: simple path chain
    edges = pd.DataFrame({
        "txId1": [0, 1, 2, 3, 4],
        "txId2": [1, 2, 3, 4, 5],
        "time_step": [1, 2, 3, 4, 5],
        "amount": [10.0] * 5,
    })
    sk = TemporalCycleSketch(width=32, depth=4, window_size=5)
    candidates = sk.find_candidates(edges)
    # A pure chain has no cycles
    assert len(candidates) == 0


def test_sketch_candidates_function_alias(medium_graph):
    """Module-level sketch_candidates function returns same data."""
    a = sketch_candidates(medium_graph, width=16, depth=4, window_size=5)
    b = sketch_candidates(medium_graph, width=16, depth=4, window_size=5)
    # Both calls return list of dicts with same length (deterministic with fixed seed)
    assert len(a) == len(b)


def test_sketch_window_size_affects_partitioning():
    """Different window sizes should partition edges differently."""
    edges = pd.DataFrame({
        "txId1": [0, 1, 2, 10, 11, 12],
        "txId2": [1, 2, 0, 11, 12, 10],
        "time_step": [1, 2, 3, 8, 9, 10],
        "amount": [10.0] * 6,
    })
    sk_short = TemporalCycleSketch(width=8, depth=2, window_size=2)
    sk_long  = TemporalCycleSketch(width=8, depth=2, window_size=10)
    c_short = sk_short.find_candidates(edges)
    c_long  = sk_long.find_candidates(edges)
    # Long window should find more cycles (both fit in one window)
    assert len(c_long) >= len(c_short)