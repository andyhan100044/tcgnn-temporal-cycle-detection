"""TemporalDFS — Algorithm 1 (exact) from paper outline §3.

Enumerates all temporal k-cycles in a directed graph G=(V,E,T) with weights,
subject to:
  C1: topological closure (return to start)
  C2: time-increasing    t_1 < t_2 < ... < t_k
  C3: time-window        t_k - t_1 <= Δt
  C4: value-conservation |sum_in - sum_out| / sum_in <= ε
  C5: node-distinct (simple cycle)

Strategy: recursive DFS with three-way pruning (time span, time-increasing,
value-conservation).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from .indexing import TemporalIndex


def _value_imbalance(amounts: List[float]) -> float:
    """Return |sum_in - sum_out| / max(sum_in, eps).

    Convention: a cycle's edge amounts are taken as one-directional flows;
    conservation means total inflow ≈ total outflow (within ε).
    """
    if not amounts:
        return 0.0
    total = sum(amounts)
    if total <= 0:
        return 0.0
    # For a directed cycle, in-flow equals out-flow trivially when amounts are
    # uniform; the ε threshold accommodates per-hop variation (FX, fees, mixer
    # overhead). Here we measure maximum deviation from mean as the imbalance.
    mean = total / len(amounts)
    max_dev = max(abs(a - mean) for a in amounts)
    return max_dev / mean


def _dfs_recursive(
    idx: TemporalIndex,
    start: int,
    current: int,
    depth: int,
    k_target: int,
    path: List[int],
    t_path: List[int],
    a_path: List[float],
    t0: int,
    time_window: int,
    epsilon: float,
    visited: set,
    results: List[Dict],
    max_per_start: int,
) -> None:
    if len(results) >= max_per_start:
        return

    if depth == k_target:
        # Try to close the cycle back to start
        closing = idx.in_neighbors_in_window(start, t_path[-1] - 1 if False else t_path[-1], t0 + time_window + 1)
        for (u, t_close, w_close) in closing:
            if u == current and t_close > t_path[-1] and (t_close - t_path[0]) <= time_window:
                # Build full cycle: nodes are distinct (no repeat of start),
                # but times/amounts include the closing edge (k edges total).
                full_nodes = list(path)  # already distinct (visited check)
                full_times = t_path + [t_close]
                full_amounts = a_path + [w_close]
                imb = _value_imbalance(full_amounts)
                if imb <= epsilon:
                    results.append({
                        "nodes": full_nodes,
                        "times": full_times,
                        "amounts": full_amounts,
                        "value_imbalance": imb,
                        "length": len(full_nodes),
                        "time_span": full_times[-1] - full_times[0],
                    })
        return

    t_max = t_path[0] + time_window
    # Prune: if remaining depth cannot fit within time window, stop.
    remaining_depth = k_target - depth
    if remaining_depth > (t_max - t_path[-1]):
        return

    # Iterate outgoing edges in time order
    for (nxt, t_edge, w_edge) in idx.out_neighbors(current):
        if t_edge <= t_path[-1]:
            continue  # C2: time-increasing
        if t_edge > t_max:
            break     # C3: span exceeded
        if nxt == start and depth < k_target - 1:
            # Cannot return to start before reaching target length
            continue
        if nxt in visited:
            continue  # C5: simple cycle
        # Pruning: value-conservation partial bound (use running balance)
        partial_amounts = a_path + [w_edge]
        imb_now = _value_imbalance(partial_amounts)
        if imb_now > epsilon * 1.5:  # tolerate some headroom
            continue

        path.append(nxt)
        t_path.append(t_edge)
        a_path.append(w_edge)
        visited.add(nxt)

        _dfs_recursive(
            idx, start, nxt, depth + 1,
            k_target, path, t_path, a_path,
            t0, time_window, epsilon,
            visited, results, max_per_start,
        )

        path.pop()
        t_path.pop()
        a_path.pop()
        visited.remove(nxt)

        if len(results) >= max_per_start:
            return


def find_cycles_simple(
    idx: TemporalIndex,
    k: int,
    time_window: int,
    epsilon: float,
    max_cycles_per_start: int = 50,
) -> List[Dict]:
    """Enumerate all length-k temporal cycles using TemporalDFS.

    Args:
        idx: TemporalIndex built from edges.
        k: cycle length (exact).
        time_window: max span Δt between first and last edge.
        epsilon: value-conservation threshold.
        max_cycles_per_start: per-start-node safety cap.

    Returns:
        List of cycle dicts {nodes, times, amounts, value_imbalance, length, time_span}.
    """
    all_cycles: List[Dict] = []
    for start in sorted(set(idx._out.keys()) | set(idx._in.keys())):
        out_edges = idx.out_neighbors(start)
        for (nxt, t0_edge, w0_edge) in out_edges:
            path = [start, nxt]
            t_path = [t0_edge, t0_edge]  # will overwrite second entry — fix:
            # The cycle starts at `start`, with first outgoing edge at t0_edge.
            # Path: [start] has no time; t_path tracks times of edges traversed.
            # We model as: t_path[0] = first edge time = t0_edge.
            t_path = [t0_edge]
            a_path = [w0_edge]
            visited = {start, nxt}
            _dfs_recursive(
                idx, start, nxt, 2,
                k, path, t_path, a_path,
                t0_edge, time_window, epsilon,
                visited, all_cycles, max_cycles_per_start,
            )
    return all_cycles


def find_temporal_cycles(
    idx: TemporalIndex,
    edges: pd.DataFrame,
    k_range: Tuple[int, int] = (3, 5),
    time_window: int = 14,
    epsilon: float = 0.1,
    max_cycles_per_start: int = 50,
) -> List[Dict]:
    """Public API matching paper outline §3.2.

    Searches for cycles of length k ∈ k_range.
    """
    out: List[Dict] = []
    for k in range(k_range[0], k_range[1] + 1):
        out.extend(find_cycles_simple(
            idx, k=k, time_window=time_window,
            epsilon=epsilon, max_cycles_per_start=max_cycles_per_start,
        ))
    return out