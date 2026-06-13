"""TC-GNN: Constraint-driven Temporal Cycle GNN (paper §5).

Three-layer architecture:
  Layer 1: Temporal Message Passing — only aggregates t_j > t_i neighbors
  Layer 2: Cycle-Level Subgraph Encoding — attention over cycle nodes/edges
  Layer 3: Constraint-Regularized Loss — hard time/value constraints

Implemented in pure PyTorch (no PyG dependency).
"""
from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Layer 1: Temporal Message Passing
# ---------------------------------------------------------------------------
class TemporalMessagePassing(nn.Module):
    """Aggregates messages only from edges with t_j > t_i (time-increasing).

    Inputs:
      h:        (N, in_dim) node features
      edge_src: (E,) source nodes
      edge_dst: (E,) target nodes
      edge_t:   (E,) edge timestamps
      edge_w:   (E,) edge amounts
      current_t:(N,) current path time per node (for filtering)

    Output:
      (N, out_dim) per-node temporal-aware embeddings
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        # MLP for messages: input is concat(h_src, h_dst, amount, time_diff)
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * in_dim + 2, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(
        self,
        h: torch.Tensor,
        edge_src: torch.Tensor,
        edge_dst: torch.Tensor,
        edge_t: torch.Tensor,
        edge_w: torch.Tensor,
        current_t: torch.Tensor,
    ) -> torch.Tensor:
        n = h.shape[0]
        device = h.device

        if edge_src.numel() == 0:
            # No edges: just a learnable transform of h
            return self.update_mlp(torch.cat([h, torch.zeros(n, self.out_dim, device=device)], dim=1))

        # Filter edges: only keep those where t_edge > current_t[dst]
        # (i.e., incoming time-increasing edges to dst)
        t_dst = current_t[edge_dst]
        time_increasing_mask = edge_t > t_dst  # (E,)
        if not time_increasing_mask.any():
            return self.update_mlp(torch.cat([h, torch.zeros(n, self.out_dim, device=device)], dim=1))

        src_f = edge_src[time_increasing_mask]
        dst_f = edge_dst[time_increasing_mask]
        t_f   = edge_t[time_increasing_mask]
        w_f   = edge_w[time_increasing_mask]

        # Build messages
        h_src = h[src_f]                              # (E', in_dim)
        h_dst = h[dst_f]                              # (E', in_dim)
        dt = (t_f - t_dst[time_increasing_mask]).unsqueeze(1)  # time delta per edge
        w = w_f.unsqueeze(1)
        msg_input = torch.cat([h_src, h_dst, dt, w], dim=1)    # (E', 2*in_dim + 2)
        messages = self.msg_mlp(msg_input)                     # (E', out_dim)

        # Aggregate via mean over incoming time-increasing edges per node
        agg = torch.zeros(n, self.out_dim, device=device)
        counts = torch.zeros(n, 1, device=device)
        agg.index_add_(0, dst_f, messages)
        ones = torch.ones_like(messages[:, :1])
        counts.index_add_(0, dst_f, ones)
        agg = agg / counts.clamp(min=1.0)

        # Update step
        return self.update_mlp(torch.cat([h, agg], dim=1))


# ---------------------------------------------------------------------------
# Layer 2: Cycle-Level Subgraph Encoder
# ---------------------------------------------------------------------------
class CycleSubgraphEncoder(nn.Module):
    """Attention-based pooling over cycle nodes + edges → fixed-size embedding."""

    def __init__(self, node_dim: int, hidden_dim: int, edge_dim: int = 8):
        super().__init__()
        self.node_dim = node_dim
        self.hidden_dim = hidden_dim
        self.edge_proj = nn.Linear(edge_dim, node_dim)
        self.q_proj = nn.Linear(node_dim, hidden_dim)
        self.k_proj = nn.Linear(node_dim, hidden_dim)
        self.v_proj = nn.Linear(node_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        node_emb: torch.Tensor,    # (B, L, node_dim)
        edge_attr: torch.Tensor,   # (B, L, edge_dim) where edge_dim=8: [t, dt, w, w_norm, sin_t, cos_t, imb, imb_dev]
    ) -> torch.Tensor:
        # Inject edge info into node embeddings
        edge_bias = self.edge_proj(edge_attr)         # (B, L, node_dim)
        h = node_emb + edge_bias                      # (B, L, node_dim)

        # Attention pooling
        q = self.q_proj(h.mean(dim=1, keepdim=True))  # (B, 1, hidden_dim) — query = cycle mean
        k = self.k_proj(h)                            # (B, L, hidden_dim)
        v = self.v_proj(h)                            # (B, L, hidden_dim)
        scores = (q * k).sum(dim=-1) / math.sqrt(self.hidden_dim)  # (B, L)
        attn = F.softmax(scores, dim=1)                            # (B, L)
        context = (attn.unsqueeze(-1) * v).sum(dim=1)              # (B, hidden_dim)
        return self.out_proj(context)


# ---------------------------------------------------------------------------
# Full TC-GNN
# ---------------------------------------------------------------------------
class TC_GNN(nn.Module):
    """End-to-end TC-GNN.

    Inputs:
      cycles: list of {nodes: [int], times: [int], amounts: [float]}
      node_features: (N, node_feature_dim)
      node_id_map: dict mapping raw node id -> tensor index
    """

    def __init__(
        self,
        node_feature_dim: int,
        hidden_dim: int = 64,
        n_gnn_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(node_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gnn_layers = nn.ModuleList([
            TemporalMessagePassing(in_dim=hidden_dim, out_dim=hidden_dim)
            for _ in range(n_gnn_layers)
        ])
        self.cycle_encoder = CycleSubgraphEncoder(
            node_dim=hidden_dim, hidden_dim=hidden_dim,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def _gather_cycle_edges(
        self,
        cycles: List[Dict],
        local_map: Dict[int, int],
        device: torch.device,
    ) -> tuple:
        """For all cycles, build a flat edge list using LOCAL indices (0..len(all_nodes)-1)."""
        edge_src = []
        edge_dst = []
        edge_t   = []
        edge_w   = []
        for c in cycles:
            nodes = c["nodes"]
            times = c["times"]
            amounts = c["amounts"]
            k = len(nodes)
            for i in range(k):
                src = local_map[nodes[i]]
                dst = local_map[nodes[(i + 1) % k]]
                edge_src.append(src)
                edge_dst.append(dst)
                edge_t.append(float(times[i]))
                edge_w.append(float(amounts[i]))
        return (
            torch.tensor(edge_src, dtype=torch.long, device=device),
            torch.tensor(edge_dst, dtype=torch.long, device=device),
            torch.tensor(edge_t,   dtype=torch.float32, device=device),
            torch.tensor(edge_w,   dtype=torch.float32, device=device),
        )

    def _build_cycle_features(
        self,
        cycles: List[Dict],
        node_emb: torch.Tensor,
        local_map: Dict[int, int],
        device: torch.device,
    ) -> tuple:
        """Per-cycle: gather node embeddings and edge attributes into tensors.

        Uses local_map (raw node ID -> local index in node_emb).
        """
        max_len = max(len(c["nodes"]) for c in cycles)
        n_cycles = len(cycles)

        node_emb_batch = torch.zeros(n_cycles, max_len, node_emb.shape[1], device=device)
        edge_attr_batch = torch.zeros(n_cycles, max_len, 8, device=device)

        for ci, c in enumerate(cycles):
            nodes = c["nodes"]
            times = c["times"]
            amounts = c["amounts"]
            k = len(nodes)
            for j in range(k):
                idx = local_map[nodes[j]]
                node_emb_batch[ci, j] = node_emb[idx]

                t = float(times[j])
                w = float(amounts[j])
                w_arr = c["amounts"]
                mean_w = sum(w_arr) / len(w_arr)
                imb = abs(w - mean_w) / max(mean_w, 1e-9)
                imb_dev = (max(w_arr) - min(w_arr)) / max(mean_w, 1e-9)
                dt = (times[(j + 1) % k] - times[j]) if k > 1 else 0.0

                edge_attr_batch[ci, j] = torch.tensor([
                    t, dt, w, w / max(mean_w, 1e-9),
                    math.sin(t * 0.1), math.cos(t * 0.1),
                    imb, imb_dev,
                ], device=device)
        return node_emb_batch, edge_attr_batch

    def forward(
        self,
        cycles: List[Dict],
        node_features: torch.Tensor,
        node_id_map: Dict[int, int],
    ) -> torch.Tensor:
        device = node_features.device

        # 1. Map raw node IDs to local positions (only nodes that appear in cycles)
        all_nodes = sorted({n for c in cycles for n in c["nodes"]})
        local_map = {n: i for i, n in enumerate(all_nodes)}

        # 2. Compute per-node max-time (for time-filtering in message passing)
        node_to_cur_t = {}
        for c in cycles:
            for n in c["nodes"]:
                node_to_cur_t[n] = max(node_to_cur_t.get(n, 0), max(c["times"]))
        current_t_used = torch.tensor(
            [float(node_to_cur_t.get(n, 0)) for n in all_nodes],
            dtype=torch.float32, device=device,
        )

        # 3. Project full node_features, then restrict to used nodes
        h = self.input_proj(node_features)
        used_idx = torch.tensor(
            [node_id_map[n] for n in all_nodes], dtype=torch.long, device=device,
        )
        h_used = h[used_idx]

        # 4. Gather cycle edges using LOCAL indices (consistent with h_used)
        edge_src, edge_dst, edge_t, edge_w = self._gather_cycle_edges(
            cycles, local_map, device,
        )

        # 5. Apply GNN layers
        for layer in self.gnn_layers:
            h_used = layer(h_used, edge_src, edge_dst,
                           edge_t, edge_w, current_t_used)

        # 6. Build per-cycle embeddings using LOCAL indices
        node_emb_batch, edge_attr_batch = self._build_cycle_features(
            cycles, h_used, local_map, device,
        )
        cycle_emb = self.cycle_encoder(node_emb_batch, edge_attr_batch)
        return self.classifier(cycle_emb)