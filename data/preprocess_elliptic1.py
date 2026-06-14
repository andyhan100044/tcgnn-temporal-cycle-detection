"""
Preprocess raw Elliptic1 CSVs into the project's standard TemporalGraph format.

Reads:
  data/elliptic1/raw/elliptic_txs_features.csv   (txId, time_step, f1..f166)
  data/elliptic1/raw/elliptic_txs_edgelist.csv   (txId1, txId2)
  data/elliptic1/raw/elliptic_txs_classes.csv    (txId, class)

Class mapping:  raw 1=illicit, 2=licit, 3=unknown
                ours 1=illicit, 0=licit, 2=unknown

Writes to data/elliptic1/processed/:
  - nodes.parquet  (txId, class)
  - edges.parquet  (txId1, txId2, time_step, amount)
  - features.parquet (txId, time_step, f1..f166)
  - summary.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


CLASS_MAP = {"1": 1, "2": 0, "unknown": 2, 1: 1, 2: 0, 3: 2}


def main():
    raw_dir = Path("data/elliptic1/raw")
    out_dir = Path("data/elliptic1/processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in ["elliptic_txs_features.csv", "elliptic_txs_edgelist.csv",
              "elliptic_txs_classes.csv"]:
        if not (raw_dir / f).exists():
            print(f"[ERR] Missing {raw_dir / f}")
            sys.exit(1)

    print("[1/4] Loading features...")
    # Real Elliptic1 features.csv has NO header: col1=txId, col2=time_step, col3..col167=f1..f165
    features = pd.read_csv(raw_dir / "elliptic_txs_features.csv", header=None)
    features.columns = ["txId", "time_step"] + [f"f{i}" for i in range(1, features.shape[1] - 1)]
    print(f"      shape={features.shape}, time_step range={features['time_step'].min()}..{features['time_step'].max()}")

    print("[2/4] Loading classes...")
    classes = pd.read_csv(raw_dir / "elliptic_txs_classes.csv")
    # Real Elliptic1 classes use strings: "1" (illicit), "2" (licit), "unknown"
    classes["class_str"] = classes["class"].astype(str).str.strip()
    classes["class"] = classes["class_str"].map(CLASS_MAP)
    if classes["class"].isna().any():
        print(f"[WARN] {classes['class'].isna().sum()} unmapped classes; sample: {classes[classes['class'].isna()].head(3)['class_str'].tolist()}")
    classes = classes.drop(columns=["class_str"])

    print("[3/4] Loading edges...")
    edges = pd.read_csv(raw_dir / "elliptic_txs_edgelist.csv")
    # Real Elliptic1 lacks: time on edges, amounts. Use heuristics:
    #   - time_step per edge = max(time_step(txId1), time_step(txId2)) — causal approximation
    #   - amount: 1.0 placeholder
    tx_t = features.groupby("txId")["time_step"].first().to_dict()
    edges["time_step"] = edges.apply(
        lambda r: max(tx_t.get(r["txId1"], 0), tx_t.get(r["txId2"], 0)), axis=1
    )
    edges["amount"] = 1.0

    print("[4/4] Saving to parquet...")
    classes[["txId", "class"]].to_parquet(out_dir / "nodes.parquet", index=False)
    edges[["txId1", "txId2", "time_step", "amount"]].to_parquet(
        out_dir / "edges.parquet", index=False
    )
    features.to_parquet(out_dir / "features.parquet", index=False)

    summary = {
        "n_nodes": int(len(classes)),
        "n_edges": int(len(edges)),
        "n_timesteps": int(features["time_step"].nunique()),
        "n_features": int(features.shape[1] - 2),
        "n_illicit": int((classes["class"] == 1).sum()),
        "n_licit":   int((classes["class"] == 0).sum()),
        "n_unknown": int((classes["class"] == 2).sum()),
        "source": "kaggle/ellipticco/elliptic-data-set",
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[DONE] Saved to {out_dir}/")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()