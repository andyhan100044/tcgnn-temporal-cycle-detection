"""End-to-end experiment runner using REAL Elliptic1 + injected synthetic cycles.

Approach:
  1. Load real Elliptic1 (legitimate DAG baseline)
  2. Inject synthetic money-laundering cycle patterns as positive class
  3. Generate negative class from random time-respecting walks
  4. Train/val/test split, train TC-GNN + baselines
  5. Report AUC-ROC, AUC-PR, F1
  6. Save results/results_real_elliptic1.csv

This evaluates the model's ability to detect injected fraudulent patterns
within the legitimate transaction topology — a "red-team" stress test.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.data_loader import load_auto
from experiments.inject_and_run_real import build_real_cycle_dataset
from experiments.baselines import build_baseline
from experiments.baselines.xgb import XGBoostCycleClassifier
from src.temporal_cycle.tc_gnn import TC_GNN
from src.temporal_cycle.losses import constraint_regularized_loss


def _pad_sequences(cycles, key: str, pad_value: float = 0.0):
    seqs = [c[key] for c in cycles]
    if not seqs:
        return torch.zeros((0, 0))
    L = max(len(s) for s in seqs)
    out = torch.full((len(seqs), L), pad_value, dtype=torch.float32)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.float32)
    return out


def build_node_features(edges, n_nodes, dim=16, seed=0):
    """Per-node features: degree, in/out amount, time range."""
    rng = np.random.default_rng(seed)
    in_deg = np.zeros(n_nodes)
    out_deg = np.zeros(n_nodes)
    in_amt  = np.zeros(n_nodes)
    out_amt = np.zeros(n_nodes)
    min_t   = np.full(n_nodes, np.inf)
    max_t   = np.zeros(n_nodes)
    for r in edges.itertuples(index=False):
        u, v, t, w = int(r.txId1), int(r.txId2), int(r.time_step), float(r.amount)
        out_deg[u] += 1
        in_deg[v]  += 1
        out_amt[u] += w
        in_amt[v]  += w
        min_t[u] = min(min_t[u], t); min_t[v] = min(min_t[v], t)
        max_t[u] = max(max_t[u], t); max_t[v] = max(max_t[v], t)
    min_t[min_t == np.inf] = 0.0
    feats = np.stack([
        np.log1p(in_deg), np.log1p(out_deg),
        np.log1p(in_amt), np.log1p(out_amt),
        min_t, max_t,
        np.log1p(max_t - min_t + 1),
    ], axis=1)
    if feats.shape[1] < dim:
        pad = rng.standard_normal((n_nodes, dim - feats.shape[1])) * 0.1
        feats = np.concatenate([feats, pad], axis=1)
    return torch.tensor(feats, dtype=torch.float32)


def make_node_id_map(candidates):
    nodes = sorted({n for c in candidates for n in c["nodes"]})
    return {n: i for i, n in enumerate(nodes)}


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


def train_tc_gnn(train, val, node_features, node_id_map, epochs=15, lr=5e-3,
                 lambda_temp=0.005, lambda_val=0.005):
    model = TC_GNN(node_feature_dim=node_features.shape[1],
                   hidden_dim=32, n_gnn_layers=2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_labels = torch.tensor([c["label"] for c in train], dtype=torch.float32).unsqueeze(1)
    train_times  = _pad_sequences(train, "times")
    train_amounts= _pad_sequences(train, "amounts")
    val_labels = torch.tensor([c["label"] for c in val], dtype=torch.float32).unsqueeze(1)
    val_times  = _pad_sequences(val, "times")
    val_amounts= _pad_sequences(val, "amounts")
    for ep in range(epochs):
        model.train()
        logits = model(train, node_features, node_id_map)
        loss = constraint_regularized_loss(
            logits, train_labels, train_times, train_amounts,
            lambda_temp=lambda_temp, lambda_val=lambda_val,
        )
        opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = model(val, node_features, node_id_map)
            vloss = constraint_regularized_loss(
                vl, val_labels, val_times, val_amounts,
                lambda_temp=lambda_temp, lambda_val=lambda_val,
            ).item()
        print(f"  [TC-GNN] ep {ep+1}/{epochs} train={loss.item():.4f} val={vloss:.4f}")
    return model


def train_baseline(name, train, val, node_features, node_id_map, epochs=15, lr=5e-3):
    model = build_baseline(name, in_dim=node_features.shape[1], hidden_dim=32)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_labels = torch.tensor([c["label"] for c in train], dtype=torch.float32).unsqueeze(1)
    for ep in range(epochs):
        model.train()
        logits = model(train, node_features, node_id_map)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, train_labels)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model


def predict_proba(model, cycles, node_features, node_id_map):
    model.eval()
    with torch.no_grad():
        logits = model(cycles, node_features, node_id_map)
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


def main(n_positives=500, n_negatives=500, epochs=15, out_dir="results"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("[1] Build cycle dataset from real Elliptic1...")
    pos, neg, aug_edges = build_real_cycle_dataset(
        n_positives=n_positives, n_negatives=n_negatives,
    )
    candidates = pos + neg
    n_pos = sum(c["label"] for c in candidates)
    n_neg = len(candidates) - n_pos
    print(f"    candidates: {len(candidates)} ({n_pos} pos, {n_neg} neg)")

    print("[2] Temporal split...")
    train, val, test = temporal_split(candidates)
    print(f"    train={len(train)}, val={len(val)}, test={len(test)}")

    print("[3] Build node features from augmented graph...")
    # Build raw->compact mapping BEFORE remapping cycles
    raw_to_compact = make_node_id_map(candidates)
    n_compact = len(raw_to_compact)
    print(f"    n_compact={n_compact:,} (nodes in candidates)")

    # Remap cycle node IDs to compact 0..n_compact-1 indices
    def _remap(cycles):
        return [{
            "nodes": [raw_to_compact[n] for n in c["nodes"]],
            "times": c["times"],
            "amounts": c["amounts"],
            "value_imbalance": c.get("value_imbalance", 0.0),
            "length": c.get("length", len(c["nodes"])),
            "time_span": c.get("time_span", 0),
            "label": c["label"],
        } for c in cycles]

    train = _remap(train)
    val   = _remap(val)
    test  = _remap(test)
    candidates = train + val + test

    # After remap, cycles use compact indices. Pass identity dict so models
    # treat them as compact (since they index into node_features directly).
    node_id_map = {i: i for i in range(n_compact)}
    # Map edge IDs to compact local indices for feature computation
    # (we only need features for candidate-nodes, so we just iterate the compact set)
    in_deg = np.zeros(n_compact)
    out_deg = np.zeros(n_compact)
    in_amt  = np.zeros(n_compact)
    out_amt = np.zeros(n_compact)
    min_t   = np.full(n_compact, np.inf)
    max_t   = np.zeros(n_compact)
    rng = np.random.default_rng(0)
    for r in aug_edges.itertuples(index=False):
        u, v = int(r.txId1), int(r.txId2)
        if u not in node_id_map or v not in node_id_map:
            continue
        ui, vi = node_id_map[u], node_id_map[v]
        t, w = int(r.time_step), float(r.amount)
        out_deg[ui] += 1; in_deg[vi] += 1
        out_amt[ui] += w; in_amt[vi] += w
        min_t[ui] = min(min_t[ui], t); min_t[vi] = min(min_t[vi], t)
        max_t[ui] = max(max_t[ui], t); max_t[vi] = max(max_t[vi], t)
    min_t[min_t == np.inf] = 0.0
    feats = np.stack([
        np.log1p(in_deg), np.log1p(out_deg),
        np.log1p(in_amt), np.log1p(out_amt),
        min_t, max_t,
        np.log1p(max_t - min_t + 1),
    ], axis=1)
    dim = 16
    if feats.shape[1] < dim:
        pad = rng.standard_normal((n_compact, dim - feats.shape[1])) * 0.1
        feats = np.concatenate([feats, pad], axis=1)
    node_features = torch.tensor(feats, dtype=torch.float32)
    print(f"    features={node_features.shape}")

    results = []
    print("\n[4a] Training TC-GNN...")
    t0 = time.time()
    tc_gnn = train_tc_gnn(train, val, node_features, node_id_map, epochs=epochs)
    y_proba = predict_proba(tc_gnn, test, node_features, node_id_map)
    m = evaluate([c["label"] for c in test], y_proba); m["model"] = "TC-GNN"
    m["train_time_sec"] = round(time.time() - t0, 2)
    print(f"    TC-GNN: AUC-ROC={m['auc_roc']:.4f}, AUC-PR={m['auc_pr']:.4f}, F1={m['f1']:.4f}")
    results.append(m)

    for name in ["GCN", "GAT", "TGN", "DCRNN", "GLASS"]:
        print(f"\n[4b] Training {name}...")
        t0 = time.time()
        model = train_baseline(name, train, val, node_features, node_id_map, epochs=epochs)
        y_proba = predict_proba(model, test, node_features, node_id_map)
        m = evaluate([c["label"] for c in test], y_proba); m["model"] = name
        m["train_time_sec"] = round(time.time() - t0, 2)
        print(f"    {name}: AUC-ROC={m['auc_roc']:.4f}, AUC-PR={m['auc_pr']:.4f}, F1={m['f1']:.4f}")
        results.append(m)

    print("\n[4c] Training XGBoost...")
    t0 = time.time()
    xgb = XGBoostCycleClassifier(n_estimators=50, max_depth=4)
    xgb.fit(train, np.array([c["label"] for c in train]))
    y_proba = xgb.predict_proba(test)[:, 1]
    m = evaluate([c["label"] for c in test], y_proba); m["model"] = "XGBoost"
    m["train_time_sec"] = round(time.time() - t0, 2)
    print(f"    XGBoost: AUC-ROC={m['auc_roc']:.4f}, AUC-PR={m['auc_pr']:.4f}, F1={m['f1']:.4f}")
    results.append(m)

    # Save
    df = pd.DataFrame(results)
    df.to_csv(out_dir / "results_real_elliptic1.csv", index=False)
    with open(out_dir / "experiment_meta_real.json", "w") as f:
        json.dump({
            "data_source": "elliptic1_real + injected synthetic cycles",
            "n_real_nodes": 203769, "n_real_edges": 234355, "n_real_illicit": 4545,
            "n_injected_cycles": n_positives, "n_negatives": n_negatives,
            "epochs": epochs,
        }, f, indent=2)
    print(f"\n[5] Saved to {out_dir}/results_real_elliptic1.csv")
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-pos", type=int, default=500)
    p.add_argument("--n-neg", type=int, default=500)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--out-dir", type=str, default="results")
    args = p.parse_args()
    main(args.n_pos, args.n_neg, args.epochs, args.out_dir)