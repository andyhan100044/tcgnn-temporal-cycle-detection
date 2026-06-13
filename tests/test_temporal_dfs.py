"""Tests for TemporalDFS (Algorithm 1 in paper outline §3)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from src.temporal_cycle.indexing import build_index
from src.temporal_cycle.temporal_dfs import find_temporal_cycles, find_cycles_simple


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def three_cycle() -> pd.DataFrame:
    """A->B@1, B->C@2, C->A@3 — a valid 3-cycle (k=3, Δt=5, ε=0.1)."""
    return pd.DataFrame({
        "txId1": [0, 1, 2], "txId2": [1, 2, 0],
        "time_step": [1, 2, 3], "amount": [10.0, 20.0, 30.0],
    })


@pytest.fixture
def cycle_out_of_order() -> pd.DataFrame:
    """Cycle with non-monotonic timestamps: A->B@3, B->C@2, C->A@1.
    Should be REJECTED by time-increasing constraint."""
    return pd.DataFrame({
        "txId1": [0, 1, 2], "txId2": [1, 2, 0],
        "time_step": [3, 2, 1], "amount": [10.0, 20.0, 30.0],
    })


@pytest.fixture
def cycle_too_wide() -> pd.DataFrame:
    """A->B@1, B->C@10, C->A@20 — span=19. With Δt=5, should be REJECTED."""
    return pd.DataFrame({
        "txId1": [0, 1, 2], "txId2": [1, 2, 0],
        "time_step": [1, 10, 20], "amount": [10.0, 20.0, 30.0],
    })


@pytest.fixture
def cycle_value_imbalanced() -> pd.DataFrame:
    """In=10+30=40, Out=20. Imbalance = |40-20|/40 = 0.5. ε=0.1 -> REJECT."""
    return pd.DataFrame({
        "txId1": [0, 1, 2], "txId2": [1, 2, 0],
        "time_step": [1, 2, 3], "amount": [10.0, 20.0, 30.0],
    })


@pytest.fixture
def cycle_value_balanced() -> pd.DataFrame({
    "txId1": [0, 1, 2], "txId2": [1, 2, 0],
    "time_step": [1, 2, 3], "amount": [20.0, 20.0, 20.0],
}):
    return pd.DataFrame({
        "txId1": [0, 1, 2], "txId2": [1, 2, 0],
        "time_step": [1, 2, 3], "amount": [20.0, 20.0, 20.0],
    })


@pytest.fixture
def multi_cycle_graph() -> pd.DataFrame:
    """Two disjoint 3-cycles: {0,1,2} and {10,11,12}. Expect 2 cycles."""
    return pd.DataFrame({
        "txId1":   [0, 1, 2, 10, 11, 12],
        "txId2":   [1, 2, 0, 11, 12, 10],
        "time_step": [1, 2, 3, 5, 6, 7],
        "amount": [15.0, 15.0, 15.0, 25.0, 25.0, 25.0],
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_finds_simple_3cycle(three_cycle):
    idx = build_index(three_cycle)
    cycles = find_cycles_simple(idx, k=3, time_window=10, epsilon=0.5)
    assert len(cycles) >= 1
    nodes_sets = [tuple(c["nodes"]) for c in cycles]
    # Cycle could be (0,1,2) or any rotation
    assert any(set(s) == {0, 1, 2} for s in nodes_sets)


def test_rejects_violating_time_order(cycle_out_of_order):
    idx = build_index(cycle_out_of_order)
    cycles = find_cycles_simple(idx, k=3, time_window=10, epsilon=0.5)
    # No cycle satisfies time-increasing on this graph
    assert cycles == []


def test_rejects_violating_window(cycle_too_wide):
    idx = build_index(cycle_too_wide)
    # Δt = 5, span is 19 -> reject
    cycles = find_cycles_simple(idx, k=3, time_window=5, epsilon=0.5)
    assert cycles == []


def test_rejects_value_imbalance(cycle_value_imbalanced):
    idx = build_index(cycle_value_imbalanced)
    # amounts in/out = 30 vs 20, ε=0.1 -> |30-20|/30 = 0.333 > 0.1
    cycles = find_cycles_simple(idx, k=3, time_window=10, epsilon=0.1)
    assert cycles == []


def test_accepts_value_balanced(cycle_value_balanced):
    idx = build_index(cycle_value_balanced)
    # amounts balanced (20 each), ε=0.5 should accept
    cycles = find_cycles_simple(idx, k=3, time_window=10, epsilon=0.5)
    assert len(cycles) >= 1


def test_finds_two_disjoint_cycles(multi_cycle_graph):
    idx = build_index(multi_cycle_graph)
    cycles = find_cycles_simple(idx, k=3, time_window=10, epsilon=0.5)
    cycle_node_sets = [set(c["nodes"]) for c in cycles]
    assert any(s == {0, 1, 2} for s in cycle_node_sets)
    assert any(s == {10, 11, 12} for s in cycle_node_sets)


def test_cycle_dict_has_required_fields(three_cycle):
    idx = build_index(three_cycle)
    cycles = find_cycles_simple(idx, k=3, time_window=10, epsilon=0.5)
    assert cycles, "expected at least one cycle"
    c = cycles[0]
    assert "nodes" in c
    assert "times" in c
    assert "amounts" in c
    assert len(c["nodes"]) == 3
    assert len(c["times"]) == 3
    assert len(c["amounts"]) == 3
    # Times must be strictly increasing
    assert c["times"] == sorted(set(c["times"]))
    assert all(c["times"][i] < c["times"][i+1] for i in range(len(c["times"]) - 1))


def test_find_temporal_cycles_alias_works(three_cycle):
    """The paper-facing name `find_temporal_cycles` should also work."""
    idx = build_index(three_cycle)
    cycles = find_temporal_cycles(idx, three_cycle, k_range=(3, 3),
                                  time_window=10, epsilon=0.5)
    assert cycles, "expected at least one cycle via find_temporal_cycles"


def test_k_range_supports_multiple_lengths():
    """Graph with k=3 and k=4 cycles; k_range=(3, 4) should find both."""
    edges = pd.DataFrame({
        # 3-cycle: 0->1@1, 1->2@2, 2->0@3
        "txId1": [0, 1, 2,   10, 11, 12, 13],
        "txId2": [1, 2, 0,   11, 12, 13, 10],
        "time_step": [1, 2, 3,   1, 2, 3, 4],
        "amount": [10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0],
    })
    idx = build_index(edges)
    cycles = find_temporal_cycles(idx, edges, k_range=(3, 4),
                                  time_window=10, epsilon=0.5)
    lengths = sorted({c["length"] for c in cycles})
    assert 3 in lengths
    assert 4 in lengths


def test_max_cycles_per_start_limit(three_cycle):
    """Limit cycles returned per starting node."""
    idx = build_index(three_cycle)
    cycles = find_cycles_simple(idx, k=3, time_window=10, epsilon=0.5,
                                 max_cycles_per_start=1)
    assert len(cycles) <= 3  # ≤ 1 per start node × 3 start nodes