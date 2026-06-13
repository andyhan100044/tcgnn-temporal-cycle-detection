"""Temporal index for time-sorted adjacency lists.

Builds per-node out/in neighbor lists sorted by time, supporting efficient
windowed queries via binary search. Construction: O(|E| log |E|).
Space: O(|E|).
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import pandas as pd

# Edge tuple: (neighbor_id, time, amount)
Edge = Tuple[int, int, float]


@dataclass
class TemporalIndex:
    """Time-sorted adjacency index for a temporal directed graph.

    Attributes:
        _out: dict node -> list of (v, t, w) outgoing edges, sorted by t.
        _in:  dict node -> list of (u, t, w) incoming edges, sorted by t.
        _out_times: dict node -> sorted list of times for bisect on out-edges.
        _in_times:  dict node -> sorted list of times for bisect on in-edges.
    """
    _out: Dict[int, List[Edge]] = field(default_factory=dict)
    _in: Dict[int, List[Edge]] = field(default_factory=dict)
    _out_times: Dict[int, List[int]] = field(default_factory=dict)
    _in_times: Dict[int, List[int]] = field(default_factory=dict)
    _n_edges: int = 0

    @property
    def n_edges(self) -> int:
        return self._n_edges

    def out_neighbors(self, v: int) -> List[Edge]:
        """Return all outgoing edges of v, sorted by time ascending."""
        return self._out.get(v, [])

    def in_neighbors(self, v: int) -> List[Edge]:
        """Return all incoming edges of v, sorted by time ascending."""
        return self._in.get(v, [])

    def out_neighbors_in_window(self, v: int, t_lo: int, t_hi: int) -> List[Edge]:
        """Return outgoing edges of v with time strictly in (t_lo, t_hi)."""
        edges = self._out.get(v)
        times = self._out_times.get(v)
        if not edges or not times:
            return []
        # bisect_left/right on the times list, then slice edges accordingly.
        # We use (t_lo, t_hi) — strict open interval — so use bisect_right(t_lo)
        # and bisect_left(t_hi).
        lo = bisect.bisect_right(times, t_lo)
        hi = bisect.bisect_left(times, t_hi)
        return edges[lo:hi]

    def in_neighbors_in_window(self, v: int, t_lo: int, t_hi: int) -> List[Edge]:
        """Return incoming edges of v with time strictly in (t_lo, t_hi)."""
        edges = self._in.get(v)
        times = self._in_times.get(v)
        if not edges or not times:
            return []
        lo = bisect.bisect_right(times, t_lo)
        hi = bisect.bisect_left(times, t_hi)
        return edges[lo:hi]


def build_index(edges: pd.DataFrame) -> TemporalIndex:
    """Construct a TemporalIndex from an edges DataFrame.

    Required columns: txId1, txId2, time_step, amount.
    """
    idx = TemporalIndex()
    idx._n_edges = len(edges)

    # Group by source and target, sort by time within each group.
    # Using groupby + apply is O(|E|) but creates many small DataFrames.
    # We instead iterate once and append, then sort each per-node list.
    out_buf: Dict[int, List[Edge]] = {}
    in_buf: Dict[int, List[Edge]] = {}

    for row in edges.itertuples(index=False):
        u = int(row.txId1)
        v = int(row.txId2)
        t = int(row.time_step)
        w = float(row.amount)
        out_buf.setdefault(u, []).append((v, t, w))
        in_buf.setdefault(v, []).append((u, t, w))

    # Sort each per-node list by time and split into edges + times
    for v, lst in out_buf.items():
        lst.sort(key=lambda e: e[1])
        idx._out[v] = lst
        idx._out_times[v] = [e[1] for e in lst]
    for v, lst in in_buf.items():
        lst.sort(key=lambda e: e[1])
        idx._in[v] = lst
        idx._in_times[v] = [e[1] for e in lst]

    return idx