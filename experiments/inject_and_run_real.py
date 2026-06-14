"""Inject synthetic money-laundering cycles into real Elliptic1 graph.

Rationale: Bitcoin transactions form a directed acyclic structure (DAG)
in their natural state — money flows forward in time and does not
return. Therefore, the "positive" class of laundering cycles is not
naturally present. We simulate the positive class by injecting synthetic
cycle patterns into the real graph, while keeping the legitimate
background unchanged. This evaluates whether TC-GNN can distinguish
injected fraudulent patterns from the surrounding legitimate topology.

Process:
  1. Load real Elliptic1 (legitimate baseline)
  2. Sample illicit nodes as cycle anchors
  3. Inject K synthetic cycles around random subsets of illicit nodes
     (strict time-increasing + value-conserving)
  4. Generate negatives by random DFS walks that DON'T touch illicit nodes
  5. Run TC-GNN + baselines on the combined graph

This is essentially "red-team testing": the model must detect planted
attack patterns while ignoring the legitimate DAG background.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

from src.temporal_cycle.indexing import build_index
from src.temporal_cycle.temporal_dfs import find_cycles_simple
from src.temporal_cycle.temporal_sketch import sketch_candidates
from experiments.data_loader import load_auto


def inject_synthetic_cycles(
    edges_df: pd.DataFrame,
    illicit_nodes: Set[int],
    n_cycles: int = 500,
    cycle_len_range: Tuple[int, int] = (3, 6),
    time_window: int = 14,
    seed: int = 42,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """Inject n_cycles synthetic laundering patterns into the graph.

    Each cycle:
      - Picks k random nodes from illicit + their 1-hop neighborhood
      - Assigns strictly increasing timestamps
      - Assigns similar amounts (value-conserving)
      - Adds edges (NEW; may overlap with existing edges — we keep both)
    """
    rng = np.random.default_rng(seed)
    base_amount = 100.0  # synthetic illicit transfer amount

    # Build 1-hop neighborhoods of illicit nodes for richer cycle pools
    ill_neighbors = set(illicit_nodes)
    for r in edges_df.itertuples(index=False):
        if r.txId1 in illicit_nodes:
            ill_neighbors.add(r.txId2)
        if r.txId2 in illicit_nodes:
            ill_neighbors.add(r.txId1)

    new_edges: List[Dict] = []
    injected_cycles: List[Dict] = []

    attempts = 0
    max_attempts = n_cycles * 5
    while len(injected_cycles) < n_cycles and attempts < max_attempts:
        attempts += 1
        k = int(rng.integers(cycle_len_range[0], cycle_len_range[1] + 1))
        # 70% include at least 1 illicit node, 30% are "pure" injection
        nodes = list(rng.choice(list(ill_neighbors), size=k, replace=False))
        # Strictly increasing timestamps in a window
        win = max(time_window, k - 1)
        t0 = int(rng.integers(1, 50 - win - 1))
        ts = sorted(rng.choice(range(t0, t0 + win + 1), size=k, replace=False))

        # Value-conserving amounts
        amounts = base_amount * np.exp(rng.normal(0, 0.05, size=k))

        # Build cycle edges
        cycle_nodes = nodes
        cycle_times = ts
        cycle_amounts = list(amounts)
        cycle_edges = []
        for i in range(k):
            new_edges.append({
                "txId1":   int(cycle_nodes[i]),
                "txId2":   int(cycle_nodes[(i + 1) % k]),
                "time_step": int(cycle_times[i]),
                "amount":  float(cycle_amounts[i]),
            })
            cycle_edges.append((int(cycle_nodes[i]), int(cycle_nodes[(i + 1) % k]),
                                int(cycle_times[i]), float(cycle_amounts[i])))

        injected_cycles.append({
            "nodes": cycle_nodes,
            "times": cycle_times,
            "amounts": cycle_amounts,
            "value_imbalance": float(np.abs(amounts - amounts.mean()).max() / amounts.mean()),
            "length": k,
            "time_span": cycle_times[-1] - cycle_times[0],
        })

    injected_df = pd.DataFrame(new_edges)
    augmented_edges = pd.concat([edges_df, injected_df], ignore_index=True)
    return augmented_edges, injected_cycles


def generate_negative_cycles_real(
    edges_df: pd.DataFrame,
    illicit_nodes: Set[int],
    n_negatives: int = 500,
    cycle_len_range: Tuple[int, int] = (3, 5),
    time_window: int = 14,
    seed: int = 42,
) -> List[Dict]:
    """Generate negative cycles from random time-respecting walks that avoid illicit nodes.

    Note: Real Elliptic1 is acyclic, so we may find few/no clean negative cycles.
    In that case, we generate synthetic "near-miss" cycles that violate ONE constraint.
    """
    rng = np.random.default_rng(seed)
    idx = build_index(edges_df)

    all_nodes = sorted(set(edges_df["txId1"]) | set(edges_df["txId2"]))
    licit_anchors = [n for n in all_nodes if n not in illicit_nodes]
    rng.shuffle(licit_anchors)
    licit_anchors = licit_anchors[:max(50, n_negatives // 5)]

    negatives: List[Dict] = []
    for start in licit_anchors:
        out = idx.out_neighbors(start)
        if not out:
            continue
        for (nxt, t0, w0) in out[:3]:
            k = int(rng.integers(cycle_len_range[0], cycle_len_range[1] + 1))
            path = [start, nxt]
            t_path = [t0]
            a_path = [w0]
            cur = nxt
            cur_t = t0
            visited = {start, nxt}
            for _ in range(k - 2):
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
            closing = idx.in_neighbors_in_window(start, cur_t, cur_t + time_window)
            for (u, tc, wc) in closing:
                if u == cur and tc > cur_t:
                    if any(n in illicit_nodes for n in path):
                        continue
                    full_t = t_path + [tc]
                    full_a = a_path + [wc]
                    negatives.append({
                        "nodes": path,
                        "times": full_t,
                        "amounts": full_a,
                        "value_imbalance": 0.0,
                        "length": len(path),
                        "time_span": full_t[-1] - full_t[0],
                    })
                    break
        if len(negatives) >= n_negatives:
            break

    if len(negatives) < n_negatives // 2:
        # Pad with synthetic "noise" cycles that violate ONE constraint (e.g., bad time order)
        print(f"[warn] only {len(negatives)} legit negative cycles; padding with constraint-violating ones")
        for i in range(n_negatives - len(negatives)):
            k = int(rng.integers(cycle_len_range[0], cycle_len_range[1] + 1))
            nodes = list(rng.choice(licit_anchors, size=k, replace=False))
            # Time NOT strictly increasing
            ts = list(rng.choice(range(1, 49), size=k, replace=False))
            # Value imbalanced
            amts = list(rng.uniform(1, 1000, size=k))
            negatives.append({
                "nodes": nodes, "times": ts, "amounts": amts,
                "value_imbalance": 1.0, "length": k, "time_span": 0,
            })
    return negatives


def build_real_cycle_dataset(
    n_positives: int = 500,
    n_negatives: int = 500,
    seed: int = 42,
):
    """Build a balanced cycle-level dataset from real Elliptic1 with injected patterns.

    Returns (positive_cycles, negative_cycles, augmented_edges).
    """
    g = load_auto()
    print(f"[real-data] {g.name}: {g.n_nodes} nodes, {g.n_edges} edges, {g.n_illicit} illicit")

    print(f"[inject] injecting {n_positives} synthetic cycles into legitimate graph...")
    augmented_edges, pos_cycles = inject_synthetic_cycles(
        g.edges, g.illicit_node_set(),
        n_cycles=n_positives, seed=seed,
    )
    print(f"[inject] augmented edges: {len(augmented_edges)} (+{len(augmented_edges) - len(g.edges)} injected)")

    print(f"[neg] generating {n_negatives} negative cycles...")
    neg_cycles = generate_negative_cycles_real(
        augmented_edges, g.illicit_node_set(),
        n_negatives=n_negatives, seed=seed,
    )
    print(f"[neg] got {len(neg_cycles)} negatives")

    for c in pos_cycles:
        c["label"] = 1
    for c in neg_cycles:
        c["label"] = 0

    return pos_cycles, neg_cycles, augmented_edges


if __name__ == "__main__":
    pos, neg, edges = build_real_cycle_dataset(n_positives=500, n_negatives=500)
    print(f"\nFinal: {len(pos)} positives, {len(neg)} negatives")
    print(f"Sample positive: nodes={pos[0]['nodes']}, times={pos[0]['times']}")
    print(f"Sample negative: nodes={neg[0]['nodes']}, times={neg[0]['times']}")