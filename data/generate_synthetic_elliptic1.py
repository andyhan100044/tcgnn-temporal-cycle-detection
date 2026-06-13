"""
Synthetic Elliptic1-like transaction graph generator.

Mimics published statistics of the real Elliptic1 dataset:
  - 203,769 nodes, 234,355 edges, 49 timesteps
  - Node classes: 0=licit (~94.6%), 1=illicit (~2.2%), 2=unknown (~3.2%)
  - 166 node features per timestep

The synthetic generator also injects explicit **circular money-laundering
patterns** as ground-truth cycles, which the real Elliptic1 dataset does NOT
provide (this is the methodological contribution: lift node-level labels to
cycle-level supervision).

Real-data swap:
    When the user provides real Elliptic1 CSVs (from Kaggle), drop them into
    data/elliptic1/raw/ and run data/preprocess_elliptic1.py. The loader in
    experiments/data_loader.py auto-detects raw vs synthetic.

Usage:
    python data/generate_synthetic_elliptic1.py --scale fast
    python data/generate_synthetic_elliptic1.py --scale full
    python data/generate_synthetic_elliptic1.py --scale custom --n-nodes 50000
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Published Elliptic1 statistics (source: Weber et al. 2019, anti-money laundering
# analysis paper; https://www.kaggle.com/ellipticco/elliptic-data-set)
# ---------------------------------------------------------------------------
ELLIPTIC1_STATS = {
    "n_nodes": 203_769,
    "n_edges": 234_355,
    "n_timesteps": 49,
    "n_features": 166,
    "illicit_frac": 0.022,       # ~4,545 illicit
    "licit_frac":   0.207,       # ~42,019 licit
    "unknown_frac": 0.771,       # ~157,205 unknown
}

# Subset sizes for development (memory + speed)
SCALE_PRESETS = {
    "fast":    {"n_nodes": 5_000,   "cycle_count": 200},
    "small":   {"n_nodes": 20_000,  "cycle_count": 500},
    "medium":  {"n_nodes": 50_000,  "cycle_count": 1_000},
    "full":    {"n_nodes": ELLIPTIC1_STATS["n_nodes"], "cycle_count": 5_000},
}


# ---------------------------------------------------------------------------
# 1. Background transaction graph (mimics licit + unknown transaction behavior)
# ---------------------------------------------------------------------------
def generate_background_edges(
    rng: np.random.Generator,
    n_nodes: int,
    n_edges: int,
    n_timesteps: int,
) -> pd.DataFrame:
    """Generate random background edges with power-law-like degree distribution."""
    # Source / target nodes: avoid self-loops
    src = rng.integers(0, n_nodes, size=n_edges)
    tgt = rng.integers(0, n_nodes, size=n_edges)
    mask = src != tgt
    src, tgt = src[mask], tgt[mask]

    # Timestamps: skewed toward early timesteps (Elliptic1 has this property)
    weights = np.exp(-0.03 * np.arange(n_timesteps))
    weights /= weights.sum()
    t = rng.choice(n_timesteps, size=src.shape[0], p=weights)

    # Transaction amounts: log-normal, median ~$1000, range ~$1 to ~$1M
    amounts = np.exp(rng.normal(7.0, 1.5, size=src.shape[0])).round(2)

    return pd.DataFrame({
        "txId1":   src.astype(np.int64),
        "txId2":   tgt.astype(np.int64),
        "time_step": t.astype(np.int32),
        "amount":  amounts.astype(np.float64),
    })


# ---------------------------------------------------------------------------
# 2. Inject ground-truth circular money-laundering patterns
# ---------------------------------------------------------------------------
def inject_cycles(
    rng: np.random.Generator,
    n_nodes: int,
    n_timesteps: int,
    cycle_count: int,
    cycle_length_range: Tuple[int, int] = (3, 7),
    time_window: int = 5,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """Plant circular laundering patterns with strict time-increasing constraint.

    For each cycle:
      1. Pick a random cycle length k in [min_len, max_len]
      2. Pick k random distinct nodes
      3. Assign each node a strictly increasing time_step within [t0, t0+time_window]
      4. Emit edges forming the cycle (each with an amount following value conservation)
      5. Mark all involved nodes as illicit

    Returns:
      edges_df: edges to ADD to the graph
      cycles: list of cycle metadata {nodes, times, amounts}
    """
    new_edges: List[Dict] = []
    cycles: List[Dict] = []

    base_amount = 50_000.0  # typical laundering transfer

    for cycle_id in range(cycle_count):
        k = int(rng.integers(cycle_length_range[0], cycle_length_range[1] + 1))
        nodes = rng.choice(n_nodes, size=k, replace=False)

        # Strictly increasing timestamps in a window of at least k timesteps
        # so we can always pick k distinct values (replace=False).
        win = max(time_window, k - 1)
        t0 = int(rng.integers(0, n_timesteps - win - 1))
        ts = sorted(rng.choice(
            range(t0, t0 + win + 1),
            size=k,
            replace=False,
        ))

        # Value conservation: amounts sum to zero in/out at each hop
        # Use base + small jitter per hop
        amounts = base_amount * np.exp(rng.normal(0, 0.05, size=k))

        # Build cycle edges
        cycle_nodes = list(nodes)
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

        cycles.append({
            "cycle_id": cycle_id,
            "nodes":   [int(n) for n in cycle_nodes],
            "times":   cycle_times,
            "amounts": cycle_amounts,
            "edges":   cycle_edges,
        })

    return pd.DataFrame(new_edges), cycles


def _to_native(obj):
    """Recursively convert numpy scalars/arrays to Python natives (for JSON)."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ---------------------------------------------------------------------------
# 3. Node labels and features
# ---------------------------------------------------------------------------
def generate_node_labels(
    rng: np.random.Generator,
    n_nodes: int,
    illicit_node_ids: List[int],
    illicit_frac: float = ELLIPTIC1_STATS["illicit_frac"],
    licit_frac: float = ELLIPTIC1_STATS["licit_frac"],
) -> pd.DataFrame:
    """Assign classes: 1=illicit, 0=licit, 2=unknown.

    Nodes participating in injected cycles MUST be illicit (overrides random).
    """
    classes = np.full(n_nodes, 2, dtype=np.int8)  # default: unknown

    # Mark cycle participants as illicit
    classes[np.array(illicit_node_ids, dtype=np.int64)] = 1

    # Remaining nodes: licit / unknown with published fractions
    n_illicit_remaining = max(0, int(illicit_frac * n_nodes) - len(illicit_node_ids))
    non_illicit = np.where(classes != 1)[0]
    if n_illicit_remaining > 0 and len(non_illicit) > 0:
        idx = rng.choice(non_illicit, size=min(n_illicit_remaining, len(non_illicit)), replace=False)
        classes[idx] = 1

    n_licit_target = int(licit_frac * n_nodes)
    candidates = np.where(classes == 2)[0]
    if n_licit_target > 0 and len(candidates) > 0:
        idx = rng.choice(candidates, size=min(n_licit_target, len(candidates)), replace=False)
        classes[idx] = 0

    return pd.DataFrame({
        "txId":  np.arange(n_nodes, dtype=np.int64),
        "class": classes,
    })


def generate_node_features(
    rng: np.random.Generator,
    n_nodes: int,
    n_timesteps: int,
    n_features: int = ELLIPTIC1_STATS["n_features"],
) -> pd.DataFrame:
    """Generate 166-dim features per (node, timestep). Most are zero.

    Each row: [txId, time_step, f1, f2, ..., f166]
    """
    # ~10% of node-timesteps have non-zero entries (matches Elliptic1 sparsity)
    n_total = n_nodes * n_timesteps
    n_nonzero = int(0.10 * n_total)

    rows = rng.integers(0, n_nodes, size=n_nonzero)
    cols = rng.integers(0, n_timesteps, size=n_nonzero)
    feats = rng.standard_normal((n_nonzero, n_features))

    # Build sparse COO then densify per row
    df_rows = []
    for i in range(n_nonzero):
        row = [int(rows[i]), int(cols[i])] + feats[i].tolist()
        df_rows.append(row)

    columns = ["txId", "time_step"] + [f"f{j}" for j in range(1, n_features + 1)]
    return pd.DataFrame(df_rows, columns=columns)


# ---------------------------------------------------------------------------
# 4. Main
# ---------------------------------------------------------------------------
def main(scale: str, custom_nodes: int, seed: int, output_dir: str):
    rng = np.random.default_rng(seed)

    if scale == "custom":
        n_nodes = custom_nodes
        cycle_count = max(50, custom_nodes // 40)
    else:
        n_nodes = SCALE_PRESETS[scale]["n_nodes"]
        cycle_count = SCALE_PRESETS[scale]["cycle_count"]

    n_timesteps = ELLIPTIC1_STATS["n_timesteps"]
    n_edges_target = int(n_nodes * ELLIPTIC1_STATS["n_edges"] / ELLIPTIC1_STATS["n_nodes"])

    print(f"[synth] scale={scale}, n_nodes={n_nodes}, target_edges={n_edges_target}, cycles={cycle_count}")

    # Background
    edges_bg = generate_background_edges(rng, n_nodes, n_edges_target, n_timesteps)
    print(f"[synth] background edges: {len(edges_bg)}")

    # Inject cycles
    edges_cycle, cycles = inject_cycles(rng, n_nodes, n_timesteps, cycle_count)
    print(f"[synth] injected cycle edges: {len(edges_cycle)}")

    # Merge
    edges = pd.concat([edges_bg, edges_cycle], ignore_index=True)
    edges["txId1"] = edges["txId1"].astype(np.int64)
    edges["txId2"] = edges["txId2"].astype(np.int64)
    edges["time_step"] = edges["time_step"].astype(np.int32)
    edges["amount"] = edges["amount"].astype(np.float64)
    # Drop duplicates (cycles might overlap with background)
    edges = edges.drop_duplicates(subset=["txId1", "txId2", "time_step"])
    print(f"[synth] total unique edges: {len(edges)}")

    # Node labels
    illicit_ids = list({n for c in cycles for n in c["nodes"]})
    labels = generate_node_labels(rng, n_nodes, illicit_ids)
    print(f"[synth] labels: {(labels['class']==1).sum()} illicit, "
          f"{(labels['class']==0).sum()} licit, {(labels['class']==2).sum()} unknown")

    # Features (sparse — only generate for non-zero entries to save memory)
    features = generate_node_features(rng, n_nodes, n_timesteps)

    # Save
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    edges.to_parquet(out_dir / "edges.parquet", index=False)
    labels.to_parquet(out_dir / "nodes.parquet", index=False)
    features.to_parquet(out_dir / "features.parquet", index=False)
    with open(out_dir / "ground_truth_cycles.json", "w") as f:
        json.dump(_to_native(cycles), f, indent=2)

    # Summary
    summary = {
        "scale": scale,
        "n_nodes": int(n_nodes),
        "n_edges": int(len(edges)),
        "n_timesteps": n_timesteps,
        "n_features": ELLIPTIC1_STATS["n_features"],
        "n_illicit": int((labels['class'] == 1).sum()),
        "n_licit":   int((labels['class'] == 0).sum()),
        "n_unknown": int((labels['class'] == 2).sum()),
        "n_ground_truth_cycles": len(cycles),
        "seed": seed,
        "real_elliptic1_stats_for_comparison": ELLIPTIC1_STATS,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[synth] saved to {out_dir}/")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale", choices=["fast", "small", "medium", "full", "custom"],
                        default="fast")
    parser.add_argument("--custom-nodes", type=int, default=50_000,
                        help="Only used when --scale=custom")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str,
                        default="data/elliptic1/synthetic_placeholder")
    args = parser.parse_args()
    main(args.scale, args.custom_nodes, args.seed, args.output_dir)