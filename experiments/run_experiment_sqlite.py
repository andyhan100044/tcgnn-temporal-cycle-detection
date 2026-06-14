"""SQLite-backed experiment runner.

Memory-efficient: never loads all edges; uses indexed SQL queries for
windowed neighbor lookups and mini-batch training.

Usage:
  python experiments/run_experiment_sqlite.py --epochs 10
  python experiments/run_experiment_sqlite.py --scaling 10000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.sqlite_data_layer import (
    sqlite_conn, fetch_candidates, candidate_count,
    build_compact_features_from_db,
)
from experiments.baselines import build_baseline
from experiments.baselines.xgb import XGBoostCycleClassifier
from src.temporal_cycle.tc_gnn import TC_GNN
from src.temporal_cycle.losses import constraint_regularized_loss


def temporal_split(candidates, val_ratio=0.15, test_ratio=0.15):
    by_time = sorted(candidates, key=lambda c: max(c["times"]))
    n = len(by_time)
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)
    return (
        by_time[:n - n_test - n_val],
        by_time[n - n_test - n_val: n - n_test],
        by_time[n - n_test:],
    )


def remap_cycles(cycles, node_id_map):
    out = []
    for c in cycles:
        out.append({
            "nodes": [node_id_map[n] for n in c["nodes"]],
            "times": c["times"],
            "amounts": c["amounts"],
            "value_imbalance": c.get("value_imbalance", 0.0),
            "length": c.get("length", len(c["nodes"])),
            "time_span": c.get("time_span", 0),
            "label": c["label"],
        })
    return out


def _pad(cycles, key, pad=0.0):
    seqs = [c[key] for c in cycles]
    if not seqs:
        return torch.zeros((0, 0))
    L = max(len(s) for s in seqs)
    out = torch.full((len(seqs), L), pad, dtype=torch.float32)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.float32)
    return out


def train_tc_gnn(train, val, node_features, identity_map, epochs=10, lr=5e-3,
                 lambda_temp=0.005, lambda_val=0.005):
    model = TC_GNN(node_feature_dim=node_features.shape[1], hidden_dim=32, n_gnn_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_y = torch.tensor([c["label"] for c in train], dtype=torch.float32).unsqueeze(1)
    train_t = _pad(train, "times")
    train_a = _pad(train, "amounts")
    val_y = torch.tensor([c["label"] for c in val], dtype=torch.float32).unsqueeze(1)
    val_t = _pad(val, "times")
    val_a = _pad(val, "amounts")
    for ep in range(epochs):
        model.train()
        logits = model(train, node_features, identity_map)
        loss = constraint_regularized_loss(logits, train_y, train_t, train_a,
                                            lambda_temp=lambda_temp, lambda_val=lambda_val)
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = model(val, node_features, identity_map)
            vloss = constraint_regularized_loss(vl, val_y, val_t, val_a,
                                                 lambda_temp=lambda_temp, lambda_val=lambda_val).item()
        print(f"  [TC-GNN] ep {ep+1}/{epochs} train={loss.item():.4f} val={vloss:.4f}")
    return model


def train_baseline(name, train, val, node_features, identity_map, epochs=10, lr=5e-3):
    model = build_baseline(name, in_dim=node_features.shape[1], hidden_dim=32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_y = torch.tensor([c["label"] for c in train], dtype=torch.float32).unsqueeze(1)
    for ep in range(epochs):
        model.train()
        logits = model(train, node_features, identity_map)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, train_y)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def predict(model, cycles, node_features, identity_map):
    model.eval()
    with torch.no_grad():
        logits = model(cycles, node_features, identity_map)
        return torch.sigmoid(logits).squeeze(-1).cpu().numpy()


def evaluate(y_true, y_proba, threshold=0.5):
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                  f1_score, precision_score, recall_score)
    y_true = np.asarray(y_true); y_proba = np.asarray(y_proba)
    out = {"n_samples": int(len(y_true)),
           "n_positive": int(y_true.sum()),
           "n_negative": int((1 - y_true).sum())}
    if out["n_positive"] == 0 or out["n_negative"] == 0:
        return out
    y_pred = (y_proba >= threshold).astype(int)
    out["auc_roc"]   = float(roc_auc_score(y_true, y_proba))
    out["auc_pr"]    = float(average_precision_score(y_true, y_proba))
    out["f1"]        = float(f1_score(y_true, y_pred, zero_division=0))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"]    = float(recall_score(y_true, y_pred, zero_division=0))
    return out


def main(epochs=10, db_path="data/elliptic1.db", out_dir="results", n_max=None):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[1] Loading candidates from {db_path}...")
    with sqlite_conn(db_path) as conn:
        n_total = candidate_count(conn)
        n_pos = candidate_count(conn, 1)
        n_neg = candidate_count(conn, 0)
        print(f"    total: {n_total} (pos={n_pos}, neg={n_neg})")
        candidates = fetch_candidates(conn)
    if n_max and len(candidates) > n_max:
        rng = np.random.default_rng(42)
        candidates = list(rng.choice(candidates, size=n_max, replace=False))
        print(f"    subsampled to {n_max}")

    print("[2] Temporal split + compact remap...")
    train, val, test = temporal_split(candidates)
    print(f"    train={len(train)}, val={len(val)}, test={len(test)}")

    print("[3] Build compact features via SQL (only nodes in candidates)...")
    feats, node_id_map = build_compact_features_from_db(db_path, candidates, dim=165,
                                                       use_real_features=True)
    feats_tensor = torch.tensor(feats, dtype=torch.float32)
    identity_map = {i: i for i in range(len(node_id_map))}
    train = remap_cycles(train, node_id_map)
    val   = remap_cycles(val, node_id_map)
    test  = remap_cycles(test, node_id_map)
    print(f"    compact nodes: {len(node_id_map)}, features shape: {feats_tensor.shape}")

    results = []
    print("\n[4a] Training TC-GNN...")
    t0 = time.time()
    m = train_tc_gnn(train, val, feats_tensor, identity_map, epochs=epochs)
    y = predict(m, test, feats_tensor, identity_map)
    r = evaluate([c["label"] for c in test], y); r["model"] = "TC-GNN"
    r["train_time_sec"] = round(time.time() - t0, 2)
    print(f"    TC-GNN: AUC-ROC={r['auc_roc']:.4f}, AUC-PR={r['auc_pr']:.4f}, F1={r['f1']:.4f}")
    results.append(r)

    for name in ["GCN", "GAT", "TGN", "DCRNN", "GLASS"]:
        print(f"\n[4b] Training {name}...")
        t0 = time.time()
        m = train_baseline(name, train, val, feats_tensor, identity_map, epochs=epochs)
        y = predict(m, test, feats_tensor, identity_map)
        r = evaluate([c["label"] for c in test], y); r["model"] = name
        r["train_time_sec"] = round(time.time() - t0, 2)
        print(f"    {name}: AUC-ROC={r['auc_roc']:.4f}, AUC-PR={r['auc_pr']:.4f}, F1={r['f1']:.4f}")
        results.append(r)

    print("\n[4c] Training XGBoost...")
    t0 = time.time()
    xgb = XGBoostCycleClassifier(n_estimators=50, max_depth=4)
    xgb.fit(train, np.array([c["label"] for c in train]))
    y = xgb.predict_proba(test)[:, 1]
    r = evaluate([c["label"] for c in test], y); r["model"] = "XGBoost"
    r["train_time_sec"] = round(time.time() - t0, 2)
    print(f"    XGBoost: AUC-ROC={r['auc_roc']:.4f}, AUC-PR={r['auc_pr']:.4f}, F1={r['f1']:.4f}")
    results.append(r)

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "results_sqlite.csv", index=False)
    print(f"\n[5] Saved to {out_dir}/results_sqlite.csv")
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--db", type=str, default="data/elliptic1.db")
    p.add_argument("--out-dir", type=str, default="results")
    p.add_argument("--n-max", type=int, default=None,
                   help="Subsample candidates to this size for scaling experiments")
    args = p.parse_args()
    main(args.epochs, args.db, args.out_dir, args.n_max)