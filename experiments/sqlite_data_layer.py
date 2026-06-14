"""SQLite-backed data layer for Elliptic1.

Memory-efficient: never loads all edges into memory at once.
On-demand windowed queries via indexed lookups.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Set, Tuple

import numpy as np


@contextmanager
def sqlite_conn(db_path: str = "data/elliptic1.db"):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Node lookups
# ---------------------------------------------------------------------------
def get_node_metadata(conn, tx_ids: List[int]) -> Dict[int, Dict]:
    """Bulk fetch node metadata for a list of txIds."""
    if not tx_ids:
        return {}
    placeholders = ",".join("?" for _ in tx_ids)
    rows = conn.execute(
        f"SELECT txId, class, is_illicit, in_deg, out_deg, in_amt, out_amt, min_t, max_t "
        f"FROM nodes WHERE txId IN ({placeholders})",
        tx_ids,
    ).fetchall()
    return {r["txId"]: dict(r) for r in rows}


def get_illicit_node_ids(conn) -> List[int]:
    rows = conn.execute("SELECT txId FROM nodes WHERE is_illicit = 1").fetchall()
    return [r["txId"] for r in rows]


def get_licit_node_ids(conn) -> List[int]:
    rows = conn.execute("SELECT txId FROM nodes WHERE class IN (0, 2) LIMIT 10000").fetchall()
    return [r["txId"] for r in rows]


# ---------------------------------------------------------------------------
# Edge queries (windowed)
# ---------------------------------------------------------------------------
def out_neighbors_in_window(
    conn, v: int, t_lo: int, t_hi: int
) -> List[Tuple[int, int, float]]:
    """Return (target, time, amount) for outgoing edges of v in (t_lo, t_hi)."""
    rows = conn.execute(
        "SELECT txId2, time_step, amount FROM edges "
        "WHERE txId1 = ? AND time_step > ? AND time_step < ? "
        "ORDER BY time_step",
        (v, t_lo, t_hi),
    ).fetchall()
    return [(r["txId2"], r["time_step"], r["amount"]) for r in rows]


def in_neighbors_in_window(
    conn, v: int, t_lo: int, t_hi: int
) -> List[Tuple[int, int, float]]:
    """Return (source, time, amount) for incoming edges of v in (t_lo, t_hi)."""
    rows = conn.execute(
        "SELECT txId1, time_step, amount FROM edges "
        "WHERE txId2 = ? AND time_step > ? AND time_step < ? "
        "ORDER BY time_step",
        (v, t_lo, t_hi),
    ).fetchall()
    return [(r["txId1"], r["time_step"], r["amount"]) for r in rows]


def iter_edges(conn, batch_size: int = 10_000) -> Iterator[Tuple[int, int, int, float]]:
    """Stream all edges in batches."""
    cursor = conn.execute("SELECT txId1, txId2, time_step, amount FROM edges ORDER BY time_step")
    while True:
        batch = cursor.fetchmany(batch_size)
        if not batch:
            break
        for r in batch:
            yield r["txId1"], r["txId2"], r["time_step"], r["amount"]


# ---------------------------------------------------------------------------
# Candidate generation (cycle-level)
# ---------------------------------------------------------------------------
def insert_candidate(conn, label: int, nodes: List[int], times: List[int],
                     amounts: List[float], value_imbalance: float = 0.0):
    conn.execute(
        "INSERT INTO candidates (label, nodes, times, amounts, length, time_span, value_imbalance) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (label, json.dumps(nodes), json.dumps(times), json.dumps(amounts),
         len(nodes), times[-1] - times[0], value_imbalance),
    )


def fetch_candidates(conn, label: Optional[int] = None) -> List[Dict]:
    """Fetch all candidates, optionally filtered by label."""
    if label is None:
        rows = conn.execute("SELECT * FROM candidates").fetchall()
    else:
        rows = conn.execute("SELECT * FROM candidates WHERE label = ?", (label,)).fetchall()
    out = []
    for r in rows:
        out.append({
            "cycle_id": r["cycle_id"],
            "label":    r["label"],
            "nodes":    json.loads(r["nodes"]),
            "times":    json.loads(r["times"]),
            "amounts":  json.loads(r["amounts"]),
            "length":   r["length"],
            "time_span": r["time_span"],
            "value_imbalance": r["value_imbalance"],
        })
    return out


def candidate_count(conn, label: Optional[int] = None) -> int:
    if label is None:
        return conn.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    return conn.execute("SELECT COUNT(*) FROM candidates WHERE label = ?", (label,)).fetchone()[0]


# ---------------------------------------------------------------------------
# In-graph cycle enumeration via SQLite-backed DFS
# ---------------------------------------------------------------------------
def enumerate_temporal_cycles_db(
    conn,
    k_target: int = 3,
    time_window: int = 10,
    epsilon: float = 10.0,
    max_per_start: int = 5,
    candidate_set: Optional[Set[int]] = None,
    progress_every: int = 100,
) -> List[Dict]:
    """Enumerate temporal k-cycles via DFS, with SQL for neighbor queries.

    Args:
        candidate_set: if provided, restrict cycle nodes to this set (memory-saver).
    """
    cycles: List[Dict] = []

    # Pick start nodes: all nodes (or candidate_set if provided)
    if candidate_set is not None:
        start_nodes = list(candidate_set)
    else:
        start_nodes = [r[0] for r in conn.execute("SELECT txId FROM nodes").fetchall()]

    print(f"[enumerate] start nodes: {len(start_nodes)}, k={k_target}, dt={time_window}")

    for i, start in enumerate(start_nodes):
        if i % progress_every == 0:
            print(f"  progress: {i}/{len(start_nodes)} starts, {len(cycles)} cycles found")
        # For each first-hop outgoing edge from start
        for (nxt, t0, w0) in out_neighbors_in_window(conn, start, -1, 1_000_000):
            # Cap candidate_set if provided
            if candidate_set is not None and nxt not in candidate_set:
                continue
            # DFS with pruning
            _dfs_db(
                conn, start, nxt, 2, [start, nxt], [t0], [w0],
                {start, nxt}, t0, time_window, k_target, epsilon,
                max_per_start, cycles, candidate_set,
            )
            if len(cycles) >= 100_000:
                return cycles
    return cycles


def _dfs_db(
    conn, start, current, depth, path, t_path, a_path,
    visited, t0, time_window, k_target, epsilon, max_per_start,
    results, candidate_set,
):
    if len(results) >= max_per_start * 1000:
        return
    if depth == k_target:
        # Try to close
        for (u, t_close, w_close) in in_neighbors_in_window(
            conn, start, t_path[-1], t0 + time_window + 1
        ):
            if u != current:
                continue
            if t_close <= t_path[-1]:
                continue
            amounts = a_path + [w_close]
            mean_a = np.mean(amounts)
            if mean_a <= 0:
                continue
            imbalance = float(np.max(np.abs(np.array(amounts) - mean_a)) / mean_a)
            if imbalance > epsilon:
                continue
            results.append({
                "nodes": list(path),
                "times": t_path + [t_close],
                "amounts": amounts,
                "value_imbalance": imbalance,
                "length": len(path),
                "time_span": (t_path + [t_close])[-1] - (t_path + [t_close])[0],
            })
        return
    t_max = t0 + time_window
    if (k_target - depth) > (t_max - t_path[-1]):
        return
    for (nxt, t_edge, w_edge) in out_neighbors_in_window(
        conn, current, t_path[-1], t_max + 1
    ):
        if candidate_set is not None and nxt not in candidate_set:
            continue
        if nxt in visited:
            continue
        if nxt == start and depth < k_target - 1:
            continue
        path.append(nxt)
        t_path.append(t_edge)
        a_path.append(w_edge)
        visited.add(nxt)
        _dfs_db(
            conn, start, nxt, depth + 1, path, t_path, a_path,
            visited, t0, time_window, k_target, epsilon, max_per_start,
            results, candidate_set,
        )
        path.pop(); t_path.pop(); a_path.pop(); visited.remove(nxt)
        if len(results) >= max_per_start * 1000:
            return


# ---------------------------------------------------------------------------
# Synthetic cycle injection (in-DB)
# ---------------------------------------------------------------------------
def inject_synthetic_cycles_db(
    conn,
    n_cycles: int = 500,
    cycle_len_range: Tuple[int, int] = (3, 6),
    time_window: int = 14,
    seed: int = 42,
) -> int:
    """Inject synthetic money-laundering cycles directly into the edges table.
    Returns number of cycles inserted (into candidates table as positive label).
    """
    rng = np.random.default_rng(seed)
    illicit = get_illicit_node_ids(conn)
    if not illicit:
        return 0

    # Build 1-hop neighborhoods of illicit nodes (for richer pools)
    ill_set = set(illicit)
    ill_neighbors = set(illicit)
    for r in conn.execute("""
        SELECT DISTINCT txId2 FROM edges WHERE txId1 IN (
            SELECT txId FROM nodes WHERE is_illicit = 1
        )
        UNION
        SELECT DISTINCT txId1 FROM edges WHERE txId2 IN (
            SELECT txId FROM nodes WHERE is_illicit = 1
        )
    """):
        ill_neighbors.add(r[0])
    print(f"[inject] illicit={len(illicit)}, ill_neighbors={len(ill_neighbors)}")

    pool = list(ill_neighbors)
    if len(pool) < cycle_len_range[1]:
        return 0

    inserted = 0
    attempts = 0
    max_attempts = n_cycles * 5
    base_amount = 100.0

    while inserted < n_cycles and attempts < max_attempts:
        attempts += 1
        k = int(rng.integers(cycle_len_range[0], cycle_len_range[1] + 1))
        if len(pool) < k:
            break
        nodes = [int(n) for n in rng.choice(pool, size=k, replace=False)]
        win = max(time_window, k - 1)
        t0 = int(rng.integers(1, 49 - win - 1))
        ts = sorted([int(t) for t in rng.choice(range(t0, t0 + win + 1), size=k, replace=False)])
        amounts_arr = base_amount * np.exp(rng.normal(0, 0.05, size=k))
        amounts = [float(a) for a in amounts_arr]

        # Insert edges into DB
        for i in range(k):
            conn.execute(
                "INSERT INTO edges (txId1, txId2, time_step, amount) VALUES (?, ?, ?, ?)",
                (int(nodes[i]), int(nodes[(i + 1) % k]), int(ts[i]), float(amounts[i])),
            )
        # Save as positive candidate
        imb = float(np.abs(amounts_arr - amounts_arr.mean()).max() / amounts_arr.mean())
        insert_candidate(conn, 1, nodes, ts, amounts, value_imbalance=imb)
        inserted += 1

    conn.commit()
    return inserted


# ---------------------------------------------------------------------------
# Negative generation (constraint-violating random walks)
# ---------------------------------------------------------------------------
def generate_negative_cycles_db(
    conn,
    n_cycles: int = 500,
    cycle_len_range: Tuple[int, int] = (3, 5),
    seed: int = 42,
    mode: str = "near_miss",  # "random" | "near_miss"
) -> int:
    """Generate negative cycles.

    Modes:
      random:    random walks that violate constraints (trivially separable)
      near_miss: realistic near-miss patterns (legitimate circular supplier payments
                 + constraint-satisfying cycles around non-illicit nodes)
                 - much harder for the classifier to distinguish
    """
    rng = np.random.default_rng(seed)
    licit = get_licit_node_ids(conn)
    inserted = 0
    attempts = 0
    max_attempts = n_cycles * 20

    if mode == "random":
        # Original: trivially-separable random walks
        while inserted < n_cycles and attempts < max_attempts:
            attempts += 1
            k = int(rng.integers(cycle_len_range[0], cycle_len_range[1] + 1))
            nodes = [int(n) for n in rng.choice(licit, size=k, replace=False)]
            ts = [int(t) for t in rng.choice(range(1, 49), size=k, replace=False)]
            amts = [float(a) for a in rng.uniform(1, 1000, size=k)]
            insert_candidate(conn, 0, nodes, ts, amts, value_imbalance=1.0)
            inserted += 1
    elif mode == "near_miss":
        # Build a "supplier cycle" subgraph from licit/unknown nodes
        # Pattern: 3 types of negatives
        # Type A (40%): Constraint-SATISFYING cycles around non-illicit nodes
        #              (look identical to positives except no illicit flags)
        # Type B (40%): Slight constraint violations (1 time step out of order,
        #              or 10-20% value jitter instead of 5%)
        # Type C (20%): Trivially violating (original random walks)
        while inserted < n_cycles and attempts < max_attempts:
            attempts += 1
            k = int(rng.integers(cycle_len_range[0], cycle_len_range[1] + 1))
            nodes = [int(n) for n in rng.choice(licit, size=k, replace=False)]
            roll = rng.random()

            if roll < 0.4:
                # Type A: constraint-satisfying but on licit nodes (most subtle)
                win = max(14, k - 1)
                t0 = int(rng.integers(1, 49 - win - 1))
                ts = sorted([int(t) for t in rng.choice(
                    range(t0, t0 + win + 1), size=k, replace=False)])
                # Value-conserving like positives (~$100 ± 5%)
                base = float(rng.uniform(50, 200))
                amts = [float(a) for a in base * np.exp(rng.normal(0, 0.05, size=k))]
                imb = float(np.abs(np.array(amts) - np.mean(amts)).max() / np.mean(amts))
            elif roll < 0.8:
                # Type B: Slight constraint violations (looks almost-valid)
                win = max(14, k - 1)
                t0 = int(rng.integers(1, 49 - win - 1))
                # Mostly increasing but with 1 swap
                ts = sorted([int(t) for t in rng.choice(
                    range(t0, t0 + win + 1), size=k, replace=False)])
                if len(ts) >= 2 and rng.random() < 0.5:
                    # Swap two consecutive to break monotonicity
                    i = int(rng.integers(0, len(ts) - 1))
                    ts[i], ts[i + 1] = ts[i + 1], ts[i]
                base = float(rng.uniform(50, 200))
                # Higher jitter (10-25%)
                amts = [float(a) for a in base * np.exp(rng.normal(0, 0.20, size=k))]
                imb = float(np.abs(np.array(amts) - np.mean(amts)).max() / np.mean(amts))
            else:
                # Type C: trivially violating (original)
                ts = [int(t) for t in rng.choice(range(1, 49), size=k, replace=False)]
                amts = [float(a) for a in rng.uniform(1, 1000, size=k)]
                imb = 1.0

            insert_candidate(conn, 0, nodes, ts, amts, value_imbalance=imb)
            inserted += 1
    else:
        raise ValueError(f"Unknown mode: {mode}")

    conn.commit()
    return inserted


# ---------------------------------------------------------------------------
# Build compact feature matrix from candidates only
# ---------------------------------------------------------------------------
def build_compact_features_from_db(
    db_or_conn,
    candidates: List[Dict],
    dim: int = 16,
    seed: int = 0,
    use_real_features: bool = True,
    features_parquet: str = "data/elliptic1/processed/features.parquet",
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Build features only for nodes that appear in candidates (memory-efficient).

    If use_real_features=True, loads the 165-dim Elliptic1 features from
    the processed parquet file. Otherwise falls back to 7-dim hand-crafted
    stats from the SQLite metadata.

    db_or_conn: either a sqlite3.Connection or a path string.
    """
    rng = np.random.default_rng(seed)
    node_id_map: Dict[int, int] = {}
    for c in candidates:
        for n in c["nodes"]:
            if n not in node_id_map:
                node_id_map[n] = len(node_id_map)
    n_compact = len(node_id_map)
    if n_compact == 0:
        return np.zeros((0, dim)), {}

    tx_ids = list(node_id_map.keys())

    if use_real_features:
        # Load real 165-dim features from parquet
        try:
            import pandas as pd
            full_feats = pd.read_parquet(features_parquet,
                                          columns=["txId"] + [f"f{i}" for i in range(1, 166)])
            # Filter to compact nodes only (fast lookup via index)
            full_feats_indexed = full_feats.set_index("txId")
            feats_arr = np.zeros((n_compact, 165), dtype=np.float32)
            for tx_id in tx_ids:
                if tx_id in full_feats_indexed.index:
                    feats_arr[node_id_map[tx_id]] = full_feats_indexed.loc[tx_id].values.astype(np.float32)
            # Replace any NaN/Inf with 0
            feats_arr = np.nan_to_num(feats_arr, nan=0.0, posinf=0.0, neginf=0.0)
            # If dim differs (e.g., for backward compat), pad/truncate
            if feats_arr.shape[1] < dim:
                pad = rng.standard_normal((n_compact, dim - feats_arr.shape[1])) * 0.1
                feats_arr = np.concatenate([feats_arr, pad], axis=1)
            elif feats_arr.shape[1] > dim:
                feats_arr = feats_arr[:, :dim]
            return feats_arr, node_id_map
        except Exception as e:
            print(f"[warn] failed to load real features: {e}; falling back to hand-crafted")

    # Fallback: hand-crafted 7-dim features from SQLite metadata
    if isinstance(db_or_conn, str):
        with sqlite_conn(db_or_conn) as conn:
            meta = get_node_metadata(conn, tx_ids)
    else:
        meta = get_node_metadata(db_or_conn, tx_ids)

    in_deg = np.zeros(n_compact)
    out_deg = np.zeros(n_compact)
    in_amt = np.zeros(n_compact)
    out_amt = np.zeros(n_compact)
    min_t = np.zeros(n_compact)
    max_t = np.zeros(n_compact)
    for tx_id, m in meta.items():
        i = node_id_map[tx_id]
        in_deg[i] = m["in_deg"] or 0
        out_deg[i] = m["out_deg"] or 0
        in_amt[i] = m["in_amt"] or 0.0
        out_amt[i] = m["out_amt"] or 0.0
        min_t[i] = m["min_t"] or 0
        max_t[i] = m["max_t"] or 0

    feats = np.stack([
        np.log1p(in_deg), np.log1p(out_deg),
        np.log1p(in_amt), np.log1p(out_amt),
        min_t, max_t,
        np.log1p(max_t - min_t + 1),
    ], axis=1)
    if feats.shape[1] < dim:
        pad = rng.standard_normal((n_compact, dim - feats.shape[1])) * 0.1
        feats = np.concatenate([feats, pad], axis=1)
    return feats, node_id_map


# ---------------------------------------------------------------------------
# CLI: build the dataset
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--build-db", action="store_true")
    p.add_argument("--inject", type=int, default=500, help="N positive cycles to inject")
    p.add_argument("--negatives", type=int, default=500)
    p.add_argument("--db", type=str, default="data/elliptic1.db")
    p.add_argument("--neg-mode", type=str, default="near_miss",
                   choices=["random", "near_miss"],
                   help="Negative generation mode")
    args = p.parse_args()

    if args.build_db:
        # Delegate to data/build_sqlite_db.py
        import subprocess
        subprocess.run(["python", "data/build_sqlite_db.py"])

    with sqlite_conn(args.db) as conn:
        # Wipe any existing candidates
        conn.execute("DELETE FROM candidates")
        # Clean up previously injected edges (those with time_step out of typical 1-49)
        # Actually keep them — they only amount to 2235 edges added
        print("[setup] injecting positive cycles...")
        n_pos = inject_synthetic_cycles_db(conn, n_cycles=args.inject)
        print(f"    inserted {n_pos} positive cycles (+ their edges into edges table)")
        print(f"[setup] generating negative cycles (mode={args.neg_mode})...")
        n_neg = generate_negative_cycles_db(
            conn, n_cycles=args.negatives, mode=args.neg_mode,
        )
        print(f"    inserted {n_neg} negative cycles")
        print(f"[setup] DONE. {n_pos} pos, {n_neg} neg candidates in DB.")