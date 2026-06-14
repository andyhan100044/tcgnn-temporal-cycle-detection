"""Baseline cycle-level classifiers for TC-GNN comparison.

All baselines share the same interface:
    forward(cycles, node_features, node_id_map) -> logits (B, 1)

Where `cycles` is a list of dicts with keys: nodes, times, amounts.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Common: graph Laplacian helpers (degree + adjacency)
# ---------------------------------------------------------------------------
def build_adjacency(cycles: List[Dict], n_nodes: int, device) -> torch.Tensor:
    """Build a sparse (n_nodes, n_nodes) adjacency from cycle edges.

    Note: `cycles[i]['nodes']` are expected to be LOCAL indices (0..n_nodes-1).
    Symmetrize to make Laplacian symmetric.
    """
    rows, cols = [], []
    for c in cycles:
        nodes = c["nodes"]
        k = len(nodes)
        for i in range(k):
            rows.append(nodes[i])
            cols.append(nodes[(i + 1) % k])
            rows.append(nodes[(i + 1) % k])  # symmetrize
            cols.append(nodes[i])
    if not rows:
        return torch.zeros(n_nodes, n_nodes, device=device)
    indices = torch.tensor([rows, cols], dtype=torch.long, device=device)
    values = torch.ones(len(rows), device=device)
    A = torch.sparse_coo_tensor(indices, values, (n_nodes, n_nodes)).to_dense()
    return A


def symmetric_normalize(A: torch.Tensor) -> torch.Tensor:
    """D^{-1/2} A D^{-1/2}."""
    deg = A.sum(dim=1).clamp(min=1.0)
    d_inv_sqrt = deg.pow(-0.5)
    return (A * d_inv_sqrt.unsqueeze(0)) * d_inv_sqrt.unsqueeze(1)


# ---------------------------------------------------------------------------
# Baseline 1: GCN (static)
# ---------------------------------------------------------------------------
class GCNBaseline(nn.Module):
    """Graph Convolutional Network — node-level, mean-pool cycle."""

    def __init__(self, in_dim: int, hidden_dim: int = 32, n_layers: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, cycles, node_features, node_id_map):
        device = node_features.device
        n_nodes = node_features.shape[0]  # use compact size from caller
        A = build_adjacency(cycles, n_nodes, device)
        A_norm = symmetric_normalize(A + torch.eye(n_nodes, device=device))

        h = self.in_proj(node_features)
        for layer in self.layers:
            h = F.relu(layer(A_norm @ h))

        # Mean-pool cycle nodes (cycles[i]['nodes'] are LOCAL compact indices)
        cycle_emb = []
        for c in cycles:
            ids = c["nodes"]  # already compact via caller-side remap
            cycle_emb.append(h[ids].mean(dim=0))
        cycle_emb = torch.stack(cycle_emb, dim=0)
        return self.classifier(cycle_emb)


# ---------------------------------------------------------------------------
# Baseline 2: GAT (static, attention)
# ---------------------------------------------------------------------------
class GATBaseline(nn.Module):
    """Graph Attention Network — single-head attention layer for simplicity."""

    def __init__(self, in_dim: int, hidden_dim: int = 32, n_layers: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.attn = nn.Linear(2 * hidden_dim, 1)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, cycles, node_features, node_id_map):
        device = node_features.device
        n_nodes = node_features.shape[0]
        h = self.in_proj(node_features)

        # Sparse adjacency indices
        A = build_adjacency(cycles, n_nodes, device)
        adj_pairs = (A > 0).nonzero(as_tuple=False)

        for layer in self.layers:
            h_new = layer(h)
            if adj_pairs.shape[0] > 0:
                src, dst = adj_pairs[:, 0], adj_pairs[:, 1]
                e = self.attn(torch.cat([h_new[src], h_new[dst]], dim=1)).squeeze(-1)
                e = F.leaky_relu(e)
                # softmax over destination
                e_max = torch.full((n_nodes,), -1e9, device=device)
                e_max.scatter_reduce_(0, dst, e, reduce="amax", include_self=False)
                e = torch.exp(e - e_max[dst])
                e_sum = torch.zeros(n_nodes, device=device)
                e_sum.scatter_add_(0, dst, e)
                msg = torch.zeros_like(h_new)
                weighted = e.unsqueeze(-1) * h_new[src]
                msg.index_add_(0, dst, weighted)
                h = F.elu(msg / e_sum.clamp(min=1e-9).unsqueeze(-1))
            else:
                h = F.elu(h_new)

        cycle_emb = []
        for c in cycles:
            ids = [node_id_map[n] for n in c["nodes"]]
            cycle_emb.append(h[ids].mean(dim=0))
        cycle_emb = torch.stack(cycle_emb, dim=0)
        return self.classifier(cycle_emb)


# ---------------------------------------------------------------------------
# Baseline 3: TGN-lite (dynamic, simplified)
# ---------------------------------------------------------------------------
class TGNBaseline(nn.Module):
    """Simplified TGN: per-node time-encoded memory + attention aggregation.

    This is a pragmatic adaptation: we don't maintain a memory bank across
    batches, but each forward pass embeds each node's max-time-step into its
    feature, then runs attention pooling.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 32, time_dim: int = 8):
        super().__init__()
        self.time_emb = nn.Linear(1, time_dim)
        self.in_proj = nn.Linear(in_dim + time_dim, hidden_dim)
        self.attn_q = nn.Linear(hidden_dim, hidden_dim)
        self.attn_k = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, cycles, node_features, node_id_map):
        device = node_features.device
        # Compute per-node max time across its cycles
        node_max_t = {}
        for c in cycles:
            for n in c["nodes"]:
                node_max_t[n] = max(node_max_t.get(n, 0), max(c["times"]))
        # Build per-node time embedding for nodes in cycles
        node_to_t = torch.zeros(node_features.shape[0], 1, device=device)
        for n, t in node_max_t.items():
            if n < node_to_t.shape[0]:
                node_to_t[n, 0] = float(t)
        t_emb = self.time_emb(node_to_t)
        h = self.in_proj(torch.cat([node_features, t_emb], dim=1))

        cycle_emb = []
        for c in cycles:
            ids = [node_id_map[n] for n in c["nodes"]]
            h_cycle = h[ids]                                    # (k, hidden)
            q = self.attn_q(h_cycle.mean(dim=0, keepdim=True))  # (1, hidden)
            k = self.attn_k(h_cycle)                            # (k, hidden)
            scores = (q * k).sum(dim=-1) / math.sqrt(h.shape[-1])
            attn = F.softmax(scores, dim=0)
            context = (attn.unsqueeze(-1) * h_cycle).sum(dim=0)
            cycle_emb.append(context)
        cycle_emb = torch.stack(cycle_emb, dim=0)
        return self.classifier(cycle_emb)


# ---------------------------------------------------------------------------
# Baseline 4: DCRNN-lite (recurrent, simplified)
# ---------------------------------------------------------------------------
class DCRNNBaseline(nn.Module):
    """Simplified DCRNN: GRU over cycle edge sequence (time-ordered).

    `in_dim` here refers to the per-edge feature dim we'll project TO (not from).
    Edge features are (time, amount, mean_amount, value_imbalance) = 4 dim,
    projected to in_dim then fed to GRU.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 32, edge_feat_dim: int = 4):
        super().__init__()
        self.edge_proj = nn.Linear(edge_feat_dim, in_dim)
        self.gru = nn.GRU(input_size=in_dim, hidden_size=hidden_dim,
                          batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def _build_seq(self, cycles):
        seqs = []
        for c in cycles:
            k = len(c["nodes"])
            if k == 0:
                continue
            t = torch.tensor(c["times"], dtype=torch.float32)
            w = torch.tensor(c["amounts"], dtype=torch.float32)
            mean_w = w.mean()
            imb = (w - mean_w).abs() / mean_w.clamp(min=1e-9)
            seq = torch.stack([t, w, mean_w.expand_as(t), imb], dim=1)  # (k, 4)
            seqs.append(seq)
        return seqs

    def forward(self, cycles, node_features, node_id_map):
        seqs = self._build_seq(cycles)
        if not seqs:
            return torch.zeros(len(cycles), 1, device=node_features.device)
        max_len = max(s.shape[0] for s in seqs)
        feat_dim = seqs[0].shape[1]
        batch = torch.zeros(len(seqs), max_len, feat_dim, device=node_features.device)
        for i, s in enumerate(seqs):
            batch[i, :s.shape[0]] = s.to(node_features.device)
        batch = self.edge_proj(batch)
        out, h_n = self.gru(batch)
        return self.classifier(h_n.squeeze(0))


# ---------------------------------------------------------------------------
# Baseline 5: GLASS-lite (subgraph-level, simplified)
# ---------------------------------------------------------------------------
class GLASSBaseline(nn.Module):
    """Simplified GLASS: subgraph encoding with positional + structural features."""

    def __init__(self, in_dim: int, hidden_dim: int = 32):
        super().__init__()
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        # Subgraph structural features (cycle-specific): length, mean_amount,
        # time_span, value_imbalance
        self.struct_proj = nn.Linear(4, hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, cycles, node_features, node_id_map):
        h = self.in_proj(node_features)
        cycle_emb = []
        for c in cycles:
            ids = [node_id_map[n] for n in c["nodes"]]
            node_pool = h[ids].mean(dim=0)
            amounts = torch.tensor(c["amounts"], dtype=torch.float32,
                                    device=node_features.device)
            mean_a = amounts.mean()
            imbalance = (amounts - mean_a).abs().max() / mean_a.clamp(min=1e-9)
            struct = torch.tensor([
                len(c["nodes"]),
                float(mean_a.item()),
                float(max(c["times"]) - min(c["times"])),
                float(imbalance.item()),
            ], device=node_features.device)
            struct_emb = self.struct_proj(struct)
            cycle_emb.append(torch.cat([node_pool, struct_emb], dim=0))
        cycle_emb = torch.stack(cycle_emb, dim=0)
        return self.classifier(cycle_emb)


# ---------------------------------------------------------------------------
# Registry for training scripts
# ---------------------------------------------------------------------------
BASELINES = {
    "GCN": GCNBaseline,
    "GAT": GATBaseline,
    "TGN": TGNBaseline,
    "DCRNN": DCRNNBaseline,
    "GLASS": GLASSBaseline,
}


def build_baseline(name: str, in_dim: int, **kwargs) -> nn.Module:
    if name not in BASELINES:
        raise ValueError(f"Unknown baseline: {name}. Available: {list(BASELINES)}")
    return BASELINES[name](in_dim, **kwargs)