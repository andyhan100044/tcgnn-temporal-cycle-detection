"""Tests for temporal index (src/temporal_cycle/indexing.py).

Following TDD: these tests are written FIRST. Implementation must satisfy them.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from src.temporal_cycle.indexing import TemporalIndex, build_index


# ---------------------------------------------------------------------------
# Fixture: simple toy graph for indexing tests
# ---------------------------------------------------------------------------
@pytest.fixture
def toy_edges() -> pd.DataFrame:
    """3 nodes (A, B, C), 3 edges forming a 3-cycle at times 1, 2, 3."""
    return pd.DataFrame({
        "txId1":   [0, 1, 2],          # A->B, B->C, C->A
        "txId2":   [1, 2, 0],
        "time_step": [1, 2, 3],
        "amount":  [10.0, 20.0, 30.0],
    })


@pytest.fixture
def multi_branch_edges() -> pd.DataFrame:
    """Node 0 has 4 outgoing edges to {1, 2, 3, 4} at times {5, 3, 7, 1}.

    After sorting: (0->4@1), (0->2@3), (0->1@5), (0->3@7).
    """
    return pd.DataFrame({
        "txId1":   [0, 0, 0, 0, 5, 5],
        "txId2":   [1, 2, 3, 4, 0, 1],
        "time_step": [5, 3, 7, 1, 8, 6],
        "amount":  [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_build_index_returns_temporalindex(toy_edges):
    idx = build_index(toy_edges)
    assert isinstance(idx, TemporalIndex)


def test_out_neighbors_sorted_by_time(multi_branch_edges):
    """Out-edges of node 0 must be sorted by time ascending."""
    idx = build_index(multi_branch_edges)
    out = idx.out_neighbors(0)
    times = [t for (_v, t, _w) in out]
    assert times == sorted(times)
    assert times == [1, 3, 5, 7]


def test_in_neighbors_sorted_by_time(toy_edges):
    """In-edges of node 0 (target of edge from 2) must be sorted."""
    idx = build_index(toy_edges)
    in_n = idx.in_neighbors(0)
    # Edge 2->0 at time 3 is the only incoming edge to node 0
    assert len(in_n) == 1
    assert in_n[0] == (2, 3, 30.0)


def test_out_neighbors_in_window(toy_edges):
    """out_neighbors_in_window(v, t_lo, t_hi) returns only edges in (t_lo, t_hi)."""
    idx = build_index(toy_edges)
    # From node 0, time range (0, 2): only edge at time 1 (A->B)
    out = idx.out_neighbors_in_window(0, 0, 2)
    assert len(out) == 1
    assert out[0] == (1, 1, 10.0)


def test_in_neighbors_in_window(multi_branch_edges):
    """Incoming edges to node 1 in window (0, 6): only (0->1@5)."""
    idx = build_index(multi_branch_edges)
    in_n = idx.in_neighbors_in_window(1, 0, 6)
    assert len(in_n) == 1
    assert in_n[0] == (0, 5, 10.0)


def test_empty_window_returns_empty(toy_edges):
    """Window that contains no edges returns empty list."""
    idx = build_index(toy_edges)
    assert idx.out_neighbors_in_window(0, 100, 200) == []
    assert idx.in_neighbors_in_window(0, 100, 200) == []


def test_node_with_no_edges(toy_edges):
    """Node 99 doesn't exist; out_neighbors returns empty."""
    idx = build_index(toy_edges)
    assert idx.out_neighbors(99) == []
    assert idx.in_neighbors(99) == []


def test_n_edges_property(toy_edges):
    """n_edges property equals total rows in input."""
    idx = build_index(toy_edges)
    assert idx.n_edges == 3


def test_build_complexity_log_linear(multi_branch_edges):
    """Build index twice — second call should be idempotent and fast."""
    idx1 = build_index(multi_branch_edges)
    idx2 = build_index(multi_branch_edges)
    assert idx1.n_edges == idx2.n_edges


def test_does_not_mutate_input(toy_edges):
    """build_index must NOT modify the input DataFrame."""
    edges_before = toy_edges.copy()
    _ = build_index(toy_edges)
    pd.testing.assert_frame_equal(toy_edges, edges_before)