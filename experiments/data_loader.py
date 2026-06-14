"""
Data loader for TCD experiments.

Auto-detects:
  1. Synthetic placeholder (data/elliptic1/synthetic_placeholder/) — default
  2. Real Elliptic1 (data/elliptic1/raw/) when user provides Kaggle CSVs
  3. Real Elliptic2 (data/elliptic2/raw/) — deferred per paper §6.1

Outputs a unified TemporalGraph object: edges + node labels + ground-truth cycles.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Unified data structure
# ---------------------------------------------------------------------------
@dataclass
class TemporalGraph:
    """Unified representation for any TCD dataset."""
    name: str
    edges: pd.DataFrame             # txId1, txId2, time_step, amount
    nodes: pd.DataFrame             # txId, class (1=illicit, 0=licit, 2=unknown)
    features: Optional[pd.DataFrame] = None  # txId, time_step, f1..f166 (sparse)
    ground_truth_cycles: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        # Basic validation
        assert {"txId1", "txId2", "time_step", "amount"}.issubset(self.edges.columns)
        assert {"txId", "class"}.issubset(self.nodes.columns)
        assert len(self.ground_truth_cycles) >= 0

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    @property
    def n_illicit(self) -> int:
        return int((self.nodes["class"] == 1).sum())

    @property
    def n_ground_truth_cycles(self) -> int:
        return len(self.ground_truth_cycles)

    def illicit_node_set(self) -> set:
        return set(self.nodes.loc[self.nodes["class"] == 1, "txId"].tolist())


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_synthetic_placeholder(path: str = "data/elliptic1/synthetic_placeholder") -> TemporalGraph:
    """Load the synthetic Elliptic1-like dataset."""
    p = Path(path)
    edges = pd.read_parquet(p / "edges.parquet")
    nodes = pd.read_parquet(p / "nodes.parquet")
    features = pd.read_parquet(p / "features.parquet") if (p / "features.parquet").exists() else None
    with open(p / "ground_truth_cycles.json") as f:
        cycles = json.load(f)
    with open(p / "summary.json") as f:
        summary = json.load(f)
    return TemporalGraph(
        name=f"synthetic_elliptic1_scale-{summary['scale']}",
        edges=edges, nodes=nodes, features=features,
        ground_truth_cycles=cycles, metadata=summary,
    )


def load_real_elliptic1(raw_dir: str = "data/elliptic1/raw") -> TemporalGraph:
    """Load real Elliptic1 from Kaggle CSVs.

    Expected files:
      - elliptic_txs_features.csv: NO HEADER, col1=txId, col2=time_step, col3..col167=f1..f165
      - elliptic_txs_edgelist.csv: (txId1, txId2)
      - elliptic_txs_classes.csv: (txId, class as string "1"/"2"/"unknown")

    Class mapping: "1"->1 illicit, "2"->0 licit, "unknown"->2.
    """
    p = Path(raw_dir)
    if not p.exists():
        raise FileNotFoundError(
            f"Real Elliptic1 not found at {p}. "
            "Place the three CSVs from Kaggle there or use load_synthetic_placeholder()."
        )

    classes_raw = pd.read_csv(p / "elliptic_txs_classes.csv")
    class_map = {"1": 1, "2": 0, "unknown": 2}
    classes_raw["class_str"] = classes_raw["class"].astype(str).str.strip()
    classes_raw["class"] = classes_raw["class_str"].map(class_map)
    classes_raw = classes_raw.drop(columns=["class_str"])

    edges_raw = pd.read_csv(p / "elliptic_txs_edgelist.csv")
    # Real Elliptic1 features.csv has NO HEADER; col1=txId, col2=time_step
    features = pd.read_csv(p / "elliptic_txs_features.csv", header=None)
    features.columns = ["txId", "time_step"] + [f"f{i}" for i in range(1, features.shape[1] - 1)]

    # Real Elliptic1 lacks: explicit timestamps on edges, amounts, ground-truth cycles.
    # We infer:
    #   - time_step per edge: max(time_step(txId1), time_step(txId2)) (causal approximation)
    #   - amount: 1.0 placeholder (Elliptic1 has no amounts)

    tx_t = features.groupby("txId")["time_step"].first().to_dict()
    edges = edges_raw.copy()
    edges["time_step"] = edges.apply(
        lambda r: max(tx_t.get(r["txId1"], 0), tx_t.get(r["txId2"], 0)), axis=1
    )
    edges["amount"] = 1.0

    nodes = classes_raw[["txId", "class"]].copy()

    return TemporalGraph(
        name="elliptic1_real",
        edges=edges, nodes=nodes, features=features,
        ground_truth_cycles=[],
        metadata={"source": "kaggle/ellipticco/elliptic-data-set",
                  "note": "amounts placeholder, ground-truth cycles derived"},
    )


def load_auto() -> TemporalGraph:
    """Auto-detect: prefer real Elliptic1, fall back to synthetic placeholder."""
    if Path("data/elliptic1/raw").exists() and (Path("data/elliptic1/raw") / "elliptic_txs_edgelist.csv").exists():
        print("[data_loader] using REAL Elliptic1 from data/elliptic1/raw/")
        return load_real_elliptic1()
    if Path("data/elliptic1/synthetic_placeholder").exists():
        print("[data_loader] using SYNTHETIC Elliptic1 placeholder")
        return load_synthetic_placeholder()
    raise FileNotFoundError(
        "No dataset found. Either run `python data/generate_synthetic_elliptic1.py` "
        "or provide real Elliptic1 CSVs in data/elliptic1/raw/."
    )


# ---------------------------------------------------------------------------
# Candidate cycle generation (lift node-level labels to cycle-level supervision)
# ---------------------------------------------------------------------------
def build_candidate_cycles_from_illicit(
    graph: TemporalGraph,
    max_candidates: int = 5_000,
    max_cycle_len: int = 7,
    min_cycle_len: int = 3,
    time_window: int = 14,
) -> List[Dict]:
    """Generate positive cycle candidates around illicit nodes.

    For each illicit node, perform time-respecting BFS to find small cycles
    containing it. This produces the SUPERVISED POSITIVE set (cycle-level).

    Negative cycles are sampled from purely-licit neighborhoods.

    Returns list of {nodes, times, amounts, label}.
    """
    # Lazy import to avoid hard dependency at module load
    from src.temporal_cycle.temporal_dfs import find_temporal_cycles, build_index

    index = build_index(graph.edges)
    illicit = graph.illicit_node_set()

    positives: List[Dict] = []
    for start in list(illicit)[:max_candidates]:
        try:
            cycles = find_temporal_cycles(
                index, graph.edges,
                k_range=(min_cycle_len, max_cycle_len),
                time_window=time_window,
                epsilon=0.2,
                max_cycles_per_start=5,
            )
        except Exception:
            cycles = []
        for c in cycles:
            if any(n in illicit for n in c["nodes"]):
                c["label"] = 1
                positives.append(c)
        if len(positives) >= max_candidates:
            break

    # Negatives: short paths among only-licit/unknown nodes
    licit_only = [n for n in graph.nodes["txId"] if n not in illicit]
    rng = np.random.default_rng(0)
    negatives: List[Dict] = []
    for start in rng.choice(licit_only, size=min(max_candidates, len(licit_only)), replace=False):
        try:
            cycles = find_temporal_cycles(
                index, graph.edges,
                k_range=(min_cycle_len, max_cycle_len),
                time_window=time_window,
                epsilon=0.2,
                max_cycles_per_start=3,
            )
        except Exception:
            cycles = []
        for c in cycles:
            if all(n not in illicit for n in c["nodes"]):
                c["label"] = 0
                negatives.append(c)
        if len(negatives) >= max_candidates:
            break

    return positives[:max_candidates] + negatives[:max_candidates]


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    g = load_auto()
    print(f"\n=== {g.name} ===")
    print(f"  nodes={g.n_nodes}, edges={g.n_edges}, illicit={g.n_illicit}")
    print(f"  time range={g.edges.time_step.min()}..{g.edges.time_step.max()}")
    print(f"  ground-truth cycles={g.n_ground_truth_cycles}")
    print(f"  amount range=${g.edges.amount.min():.0f}..${g.edges.amount.max():.0f}")