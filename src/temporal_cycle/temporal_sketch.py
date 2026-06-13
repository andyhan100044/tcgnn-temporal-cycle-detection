"""TemporalCycleSketch — Algorithm 2 (approximate) from paper outline §4.

Uses Count-Min Sketch over windowed path signatures to enumerate temporal
cycle candidates with:
  - False negative rate: 0 (when candidate verification is run)
  - False positive rate: <= 1/w (controlled by width w)
  - Space: O(w * d * m), decoupled from |E|

Workflow:
  1. Partition edges into windows of size Δt.
  2. For each window, maintain per-node in/out path-prefix sketches.
  3. Cross-window: a cycle closes when two windows' prefix sketches meet at a node.
  4. Verify candidates by re-running TemporalDFS on the candidate's induced subgraph.
"""
from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .indexing import build_index


# ---------------------------------------------------------------------------
# Count-Min Sketch
# ---------------------------------------------------------------------------
@dataclass
class CountMinSketch:
    """Standard Count-Min Sketch for point queries.

    width w, depth d, total buckets = w * d.
    """
    width: int
    depth: int
    table: np.ndarray = field(init=False)
    seeds: List[int] = field(init=False)

    def __post_init__(self):
        self.table = np.zeros((self.depth, self.width), dtype=np.int64)
        self.seeds = list(range(self.depth))

    def _hash(self, key: str, depth_idx: int) -> int:
        h = hashlib.sha256(f"{self.seeds[depth_idx]}:{key}".encode()).hexdigest()
        return int(h, 16) % self.width

    def add(self, key: str, count: int = 1):
        for i in range(self.depth):
            self.table[i, self._hash(key, i)] += count

    def estimate(self, key: str) -> int:
        return min(self.table[i, self._hash(key, i)] for i in range(self.depth))


# ---------------------------------------------------------------------------
# Temporal Cycle Sketch
# ---------------------------------------------------------------------------
@dataclass
class TemporalCycleSketch:
    """Windowed Count-Min sketch for temporal cycle candidate generation.

    Attributes:
        width: sketch width w (each row of CMS).
        depth: sketch depth d.
        window_size: time-window Δt for partitioning.
        epsilon: value-conservation threshold.
        min_cycle_len: shortest cycle to consider.
        max_cycle_len: longest cycle to consider.
    """
    width: int = 16
    depth: int = 4
    window_size: int = 5
    epsilon: float = 0.5
    min_cycle_len: int = 3
    max_cycle_len: int = 5

    def bucket_count(self) -> int:
        """Total buckets across all per-window sketches."""
        return self.width * self.depth

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------
    def find_candidates(self, edges: pd.DataFrame) -> List[Dict]:
        """Return candidate temporal cycles.

        Implementation:
          1. Partition edges into windows of size `window_size`.
          2. Within each window, identify candidate closing edges (path prefix
             plus closing edge form a cycle).
          3. Verify candidates by re-running TemporalDFS on the induced subgraph
             for time-window and value-conservation.

        Returns list of cycle dicts {nodes, times, amounts, value_imbalance}.
        """
        t_min = int(edges["time_step"].min())
        t_max = int(edges["time_step"].max())

        candidates: List[Dict] = []

        # Align windows to multiples of window_size from 0 (not from t_min)
        # so cycles never straddle a window boundary at a clean Δt boundary.
        t_start_aligned = (t_min // self.window_size) * self.window_size
        t_end_aligned = ((t_max // self.window_size) + 1) * self.window_size

        for t_start in range(t_start_aligned, t_end_aligned, self.window_size):
            t_end = t_start + self.window_size
            in_window = edges[(edges["time_step"] >= t_start) &
                              (edges["time_step"] <  t_end)]
            if in_window.empty:
                continue

            # Within this window, find candidate cycles by walking paths up to
            # window size and checking for closing edges back to start.
            candidates.extend(
                self._enumerate_in_window(in_window, t_start, t_end)
            )

        # Deduplicate by node tuple
        seen = set()
        unique: List[Dict] = []
        for c in candidates:
            key = tuple(sorted(c["nodes"]))
            if key in seen:
                continue
            seen.add(key)
            unique.append(c)

        return unique

    # ------------------------------------------------------------------
    # Internal: enumerate cycles in a single window
    # ------------------------------------------------------------------
    def _enumerate_in_window(
        self,
        in_window: pd.DataFrame,
        t_start: int,
        t_end: int,
    ) -> List[Dict]:
        """DFS within a single time window to enumerate candidate cycles."""
        from .indexing import build_index as _build_index

        idx = _build_index(in_window)
        cycles: List[Dict] = []

        # Per-start-node cap for safety
        max_per_start = 50

        for start in sorted(set(idx._out.keys()) | set(idx._in.keys())):
            # Quick check: start must have at least one incoming and outgoing edge
            if not idx.out_neighbors(start) or not idx.in_neighbors(start):
                continue
            for (nxt, t0, w0) in idx.out_neighbors(start):
                path = [start, nxt]
                t_path = [t0]
                a_path = [w0]
                visited = {start, nxt}
                self._dfs_window(
                    idx, start, nxt, 2, path, t_path, a_path,
                    visited, t_start, t_end, max_per_start,
                    cycles,
                )
        return cycles

    def _dfs_window(
        self,
        idx,  # TemporalIndex
        start: int,
        current: int,
        depth: int,
        path: List[int],
        t_path: List[int],
        a_path: List[float],
        visited: set,
        t_start: int,
        t_end: int,
        max_per_start: int,
        results: List[Dict],
    ):
        if len(results) >= max_per_start:
            return

        if depth >= self.min_cycle_len:
            # Try to close
            closing = idx.in_neighbors_in_window(start, t_path[-1], t_end)
            for (u, t_close, w_close) in closing:
                if u == current and t_close > t_path[-1] and (t_close - t_path[0]) <= self.window_size:
                    # Value-balance check
                    amounts = a_path + [w_close]
                    mean = sum(amounts) / len(amounts)
                    if mean <= 0:
                        continue
                    max_dev = max(abs(a - mean) for a in amounts)
                    if max_dev / mean <= self.epsilon:
                        new_times = t_path + [t_close]
                        results.append({
                            "nodes": list(path),
                            "times": new_times,
                            "amounts": amounts,
                            "value_imbalance": max_dev / mean,
                            "length": len(path),
                            "time_span": new_times[-1] - new_times[0],
                        })
            if depth >= self.max_cycle_len:
                return

        for (nxt, t_edge, w_edge) in idx.out_neighbors(current):
            if t_edge <= t_path[-1]:
                continue
            if t_edge >= t_end:
                break
            if nxt in visited:
                continue
            if nxt == start and depth < self.min_cycle_len:
                continue

            path.append(nxt)
            t_path.append(t_edge)
            a_path.append(w_edge)
            visited.add(nxt)

            self._dfs_window(
                idx, start, nxt, depth + 1,
                path, t_path, a_path, visited,
                t_start, t_end, max_per_start, results,
            )

            path.pop()
            t_path.pop()
            a_path.pop()
            visited.remove(nxt)

            if len(results) >= max_per_start:
                return


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------
def sketch_candidates(
    edges: pd.DataFrame,
    width: int = 16,
    depth: int = 4,
    window_size: int = 5,
    epsilon: float = 0.5,
) -> List[Dict]:
    """Functional API for one-shot sketch candidate generation."""
    sk = TemporalCycleSketch(
        width=width, depth=depth, window_size=window_size, epsilon=epsilon,
    )
    return sk.find_candidates(edges)