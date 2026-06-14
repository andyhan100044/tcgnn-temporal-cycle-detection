"""Build SQLite database from Elliptic1 CSVs.

Schema:
  nodes(txId INTEGER PK, class INTEGER, is_illicit INTEGER, in_deg, out_deg, in_amt, out_amt, min_t, max_t)
  edges(txId1, txId2, time_step, amount)
  edges_idx1: (txId1, time_step) — for out-neighbor window queries
  edges_idx2: (txId2, time_step) — for in-neighbor window queries
  candidates(cycle_id INTEGER PK, label, nodes TEXT, times TEXT, amounts TEXT, length, time_span)

WAL mode enabled for concurrent reads during training.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd


def main(db_path: str = "data/elliptic1.db"):
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    print(f"[build_db] target: {db_path}")

    raw_dir = Path("data/elliptic1/raw")
    for f in ["elliptic_txs_classes.csv", "elliptic_txs_features.csv",
              "elliptic_txs_edgelist.csv"]:
        if not (raw_dir / f).exists():
            print(f"[ERR] Missing {raw_dir / f}")
            sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA cache_size = -65536;")  # 64 MB cache

    cur = conn.cursor()

    # Schema
    cur.executescript("""
    CREATE TABLE nodes (
        txId INTEGER PRIMARY KEY,
        class INTEGER NOT NULL,           -- 1=illicit, 0=licit, 2=unknown
        is_illicit INTEGER NOT NULL,      -- 0/1 (denormalized for fast filter)
        in_deg INTEGER DEFAULT 0,
        out_deg INTEGER DEFAULT 0,
        in_amt REAL DEFAULT 0,
        out_amt REAL DEFAULT 0,
        min_t INTEGER DEFAULT 0,
        max_t INTEGER DEFAULT 0
    );

    CREATE TABLE edges (
        txId1 INTEGER NOT NULL,
        txId2 INTEGER NOT NULL,
        time_step INTEGER NOT NULL,
        amount REAL NOT NULL
    );

    CREATE INDEX idx_edges_out ON edges (txId1, time_step);
    CREATE INDEX idx_edges_in ON edges (txId2, time_step);
    CREATE INDEX idx_edges_t ON edges (time_step);

    CREATE TABLE candidates (
        cycle_id INTEGER PRIMARY KEY AUTOINCREMENT,
        label INTEGER NOT NULL,           -- 1=positive, 0=negative
        nodes TEXT NOT NULL,              -- JSON list of node IDs
        times TEXT NOT NULL,              -- JSON list of timestamps
        amounts TEXT NOT NULL,            -- JSON list of amounts
        length INTEGER NOT NULL,
        time_span INTEGER NOT NULL,
        value_imbalance REAL DEFAULT 0.0
    );

    CREATE INDEX idx_cand_label ON candidates (label);
    """)

    # Load classes
    print("[1/5] Loading classes...")
    classes = pd.read_csv(raw_dir / "elliptic_txs_classes.csv")
    class_map = {"1": 1, "2": 0, "unknown": 2}
    classes["class_str"] = classes["class"].astype(str).str.strip()
    classes["class"] = classes["class_str"].map(class_map)
    classes["is_illicit"] = (classes["class"] == 1).astype(int)
    classes = classes[["txId", "class", "is_illicit"]]
    classes.to_sql("nodes", conn, if_exists="append", index=False)
    print(f"    {len(classes)} nodes")

    # Load edges (derive time from features, amount=1.0 placeholder)
    print("[2/5] Loading edges (with time from features)...")
    features = pd.read_csv(raw_dir / "elliptic_txs_features.csv", header=None)
    features.columns = ["txId", "time_step"] + [f"f{i}" for i in range(1, features.shape[1] - 1)]
    tx_t = features.groupby("txId")["time_step"].first().to_dict()

    edges = pd.read_csv(raw_dir / "elliptic_txs_edgelist.csv")
    edges["time_step"] = edges.apply(
        lambda r: max(tx_t.get(r["txId1"], 0), tx_t.get(r["txId2"], 0)), axis=1
    )
    edges["amount"] = 1.0
    edges = edges[["txId1", "txId2", "time_step", "amount"]]
    edges.to_sql("edges", conn, if_exists="append", index=False)
    print(f"    {len(edges)} edges")

    # Compute node degree/amount stats
    print("[3/5] Computing node degree and amount stats...")
    conn.execute("""
        UPDATE nodes SET
            out_deg = (SELECT COUNT(*) FROM edges WHERE edges.txId1 = nodes.txId),
            out_amt = (SELECT COALESCE(SUM(amount), 0) FROM edges WHERE edges.txId1 = nodes.txId),
            min_t   = (SELECT COALESCE(MIN(time_step), 0) FROM edges
                       WHERE edges.txId1 = nodes.txId OR edges.txId2 = nodes.txId),
            max_t   = (SELECT COALESCE(MAX(time_step), 0) FROM edges
                       WHERE edges.txId1 = nodes.txId OR edges.txId2 = nodes.txId)
    """)
    conn.execute("""
        UPDATE nodes SET
            in_deg = (SELECT COUNT(*) FROM edges WHERE edges.txId2 = nodes.txId),
            in_amt = (SELECT COALESCE(SUM(amount), 0) FROM edges WHERE edges.txId2 = nodes.txId)
    """)
    conn.commit()

    # Verify
    cur.execute("SELECT COUNT(*), SUM(is_illicit) FROM nodes")
    n_nodes, n_illicit = cur.fetchone()
    cur.execute("SELECT COUNT(*), MIN(time_step), MAX(time_step) FROM edges")
    n_edges, t_min, t_max = cur.fetchone()
    print(f"    {n_nodes} nodes ({n_illicit} illicit), {n_edges} edges, t=[{t_min},{t_max}]")

    # Stats
    print("[4/5] DB size and indexing...")
    db_size_mb = db_path.stat().st_size / 1024 / 1024
    print(f"    db size: {db_size_mb:.1f} MB")

    cur.execute("ANALYZE")
    conn.commit()
    conn.close()

    print(f"[5/5] DONE: {db_path}")
    print(f"    Open with: sqlite3 {db_path}")


if __name__ == "__main__":
    main()