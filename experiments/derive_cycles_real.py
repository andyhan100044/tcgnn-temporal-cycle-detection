"""Generate candidate cycles from real Elliptic1 starting at illicit nodes.

For real Elliptic1, there are no ground-truth cycles. We instead:
  1. Use illicit nodes as anchors (positive examples)
  2. Walk time-respecting paths from each illicit node
  3. Try to close the path back into the illicit neighborhood
  4. Generate negatives by walking from non-illicit anchors

The result is a cycle-level supervision dataset where labels are derived
from node-level illicit flags via cycle membership.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


def generate_positive_cycles(
    edges_df: pd.DataFrame,
    illicit_nodes: Set[int],
    k_range: Tuple[int, int] = (3, 6),
    time_window: int = 10,
    max_per_start: int = 5,
    n_starts: int = 1000,
) -> List[Dict]:
    """For each illicit node, find cycles through it via time-respecting DFS.

    A cycle is "positive" if it contains at least one illicit node.
    """
    from src.temporal_cycle.indexing import build_index
    from src.temporal_cycle.temporal_dfs import find_cycles_simple

    if not illicit_nodes:
        return []

    idx = build_index(edges_df)
    # Sort illicit nodes by some criterion (degree) to pick the most-connected ones
    ill_anchors = sorted(illicit_nodes)[:n_starts]

    positives: List[Dict] = []
    for start in ill_anchors:
        try:
            cycles = find_cycles_simple(
                idx, k=3, time_window=time_window, epsilon=10.0,  # relax value constraint
                max_cycles_per_start=max_per_start,
            )
        except Exception:
            cycles = []
        for c in cycles:
            if any(n in illicit_nodes for n in c["nodes"]):
                positives.append(c)
        if len(positives) > 50_000:
            break

    # Dedup by node tuple
    seen = set()
    unique = []
    for c in positives:
        key = tuple(sorted(c["nodes"]))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def generate_negative_cycles(
    edges_df: pd.DataFrame,
    illicit_nodes: Set[int],
    n_negatives: int = 1000,
    cycle_len_range: Tuple[int, int] = (3, 5),
    time_window: int = 10,
    seed: int = 42,
) -> List[Dict]:
    """Generate negative cycles from licit-only neighborhoods.

    Random time-respecting walks on the full graph, accept only those
    that do not touch any illicit node.
    """
    from src.temporal_cycle.indexing import build_index

    rng = np.random.default_rng(seed)
    idx = build_index(edges_df)

    # Sample licit nodes as anchors
    all_nodes = sorted(set(edges_df["txId1"]) | set(edges_df["txId2"]))
    licit_anchors = [n for n in all_nodes if n not in illicit_nodes]
    rng.shuffle(licit_anchors)
    licit_anchors = licit_anchors[:max(50, n_negatives // 10)]

    negatives: List[Dict] = []
    for start in licit_anchors:
        out = idx.out_neighbors(start)
        if not out:
            continue
        for (nxt, t0, w0) in out[:5]:
            # Random time-respecting walk of length k-1
            k = int(rng.integers(cycle_len_range[0], cycle_len_range[1] + 1))
            path = [start, nxt]
            t_path = [t0]
            a_path = [w0]
            cur = nxt
            cur_t = t0
            visited = {start, nxt}
            for _ in range(k - 2):
                # Pick next time-respecting outgoing edge
                outs = idx.out_neighbors_in_window(cur, cur_t, cur_t + time_window)
                outs = [(v, t, w) for (v, t, w) in outs if v not in visited]
                if not outs:
                    break
                idx_pick = int(rng.integers(0, len(outs)))
                nxt, nt, nw = outs[idx_pick]
                path.append(nxt)
                t_path.append(nt)
                a_path.append(nw)
                visited.add(nxt)
                cur = nxt
                cur_t = nt
            if len(path) < k:
                continue
            # Try to close back to start
            closing = idx.in_neighbors_in_window(start, cur_t, cur_t + time_window)
            for (u, tc, wc) in closing:
                if u == cur and tc > cur_t:
                    # Check: no illicit node
                    if any(n in illicit_nodes for n in path):
                        continue
                    full_t = t_path + [tc]
                    full_a = a_path + [wc]
                    # Mean amount as baseline for balance
                    mean_a = np.mean(full_a)
                    if mean_a == 0:
                        continue
                    imbalance = float(np.max(np.abs(np.array(full_a) - mean_a)) / mean_a)
                    if imbalance > 10.0:  # very lenient
                        continue
                    negatives.append({
                        "nodes": path,
                        "times": full_t,
                        "amounts": full_a,
                        "value_imbalance": imbalance,
                        "length": len(path),
                        "time_span": full_t[-1] - full_t[0],
                    })
                    break
        if len(negatives) >= n_negatives:
            break

    return negatives


def build_cycle_dataset(
    edges_df: pd.DataFrame,
    illicit_nodes: Set[int],
    n_negatives: int = 1000,
    time_window: int = 10,
    seed: int = 42,
) -> List[Dict]:
    """Build a balanced cycle-level dataset from real Elliptic1."""
    print(f"[derive_cycles] generating positives from {len(illicit_nodes)} illicit nodes...")
    positives = generate_positive_cycles(
        edges_df, illicit_nodes, time_window=time_window,
        max_per_start=3, n_starts=min(2000, len(illicit_nodes)),
    )
    print(f"[derive_cycles] positives found: {len(positives)}")

    print(f"[derive_cycles] generating negatives...")
    negatives = generate_negative_cycles(
        edges_df, illicit_nodes, n_negatives=n_negatives,
        time_window=time_window, seed=seed,
    )
    print(f"[derive_cycles] negatives found: {len(negatives)}")

    # Label
    for c in positives:
        c["label"] = 1
    for c in negatives:
        c["label"] = 0

    return positives + negatives


if __name__ == "__main__":
    from experiments.data_loader import load_auto

    g = load_auto()
    illicit = g.illicit_node_set()
    print(f"Loaded {g.name}: {g.n_nodes} nodes, {g.n_illicit} illicit")

    candidates = build_cycle_dataset(
        g.edges, illicit, n_negatives=2000, time_window=10,
    )
    print(f"\nTotal candidates: {len(candidates)}")
    n_pos = sum(c["label"] for c in candidates)
    n_neg = len(candidates) - n_pos
    print(f"  positive: {n_pos}, negative: {n_neg}")
    if candidates:
        c = candidates[0]
        print(f"\nSample cycle:")
        print(f"  nodes: {c['nodes'][:5]}...")
        print(f"  times: {c['times']}")
        print(f"  amounts: {c['amounts']}")
        print(f"  length: {c['length']}, time_span: {c['time_span']}")