"""End-to-end experiment runner for TC-GNN vs baselines.

Pipeline:
  1. Load dataset (synthetic Elliptic1 placeholder or real)
  2. Generate candidate cycles via TemporalCycleSketch
  3. Label candidates (1 if contains any illicit node, else 0)
  4. Temporal train/val/test split
  5. Build node features (uniform random projection of node id; stand-in
     until real Elliptic1 features available)
  6. Train TC-GNN + 5 baselines + XGBoost
  7. Evaluate: AUC-ROC, AUC-PR, F1, precision, recall
  8. Save results to results/main_results.csv + per-model JSON logs

Usage:
  python experiments/run_experiment.py --quick    # small data, few epochs
  python experiments/run_experiment.py --full     # larger data, more epochs
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

from experiments.data_loader import load_auto, build_candidate_cycles_from_illicit
from experiments.baselines import build_baseline
from experiments.baselines.xgb import XGBoostCycleClassifier
from src.temporal_cycle.temporal_sketch import sketch_candidates
from src.temporal_cycle.tc_gnn import TC_GNN
from src.temporal_cycle.losses import constraint_regularized_loss


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def label_candidates(candidates, illicit_set: set) -> list:
    """Label candidate cycles: 1 if contains any illicit node, else 0."""
    out = []
    for c in candidates:
        label = 1 if any(n in illicit_set for n in c["nodes"]) else 0
        c["label"] = label
        out.append(c)
    return out


def temporal_split(candidates, val_ratio: float = 0.15, test_ratio: float = 0.15):
    """Split by max-time in cycle (no future leakage)."""
    by_time = sorted(candidates, key=lambda c: max(c["times"]))
    n = len(by_time)
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)
    return (
        by_time[:n - n_test - n_val],
        by_time[n - n_test - n_val: n - n_test],
        by_time[n - n_test:],
    )


def build_node_features(graph_edges, n_nodes: int, dim: int = 16, seed: int = 0):
    """Build per-node feature matrix.

    Without real features (synthetic placeholder), we use:
      - log1p(in-degree), log1p(out-degree)
      - log1p(sum incoming amount), log1p(sum outgoing amount)
      - log1p(time of first/last activity)
      - zero-padded to `dim`
    """
    rng = np.random.default_rng(seed)
    in_deg = np.zeros(n_nodes)
    out_deg = np.zeros(n_nodes)
    in_amt  = np.zeros(n_nodes)
    out_amt = np.zeros(n_nodes)
    min_t   = np.full(n_nodes, np.inf)
    max_t   = np.zeros(n_nodes)
    for r in graph_edges.itertuples(index=False):
        u, v, t, w = int(r.txId1), int(r.txId2), int(r.time_step), float(r.amount)
        out_deg[u] += 1
        in_deg[v]  += 1
        out_amt[u] += w
        in_amt[v]  += w
        min_t[u] = min(min_t[u], t)
        min_t[v] = min(min_t[v], t)
        max_t[u] = max(max_t[u], t)
        max_t[v] = max(max_t[v], t)
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


def make_node_id_map(candidates) -> dict:
    """Map raw node id -> contiguous tensor index."""
    nodes = sorted({n for c in candidates for n in c["nodes"]})
    return {n: i for i, n in enumerate(nodes)}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _pad_sequences(cycles, key: str, pad_value: float = 0.0):
    """Pad variable-length cycle sequences to (B, L_max)."""
    seqs = [c[key] for c in cycles]
    if not seqs:
        return torch.zeros((0, 0))
    L = max(len(s) for s in seqs)
    out = torch.full((len(seqs), L), pad_value, dtype=torch.float32)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.float32)
    return out


def train_tc_gnn(train, val, node_features, node_id_map, n_nodes,
                 hidden_dim: int = 32, epochs: int = 5, lr: float = 1e-3,
                 lambda_temp: float = 0.01, lambda_val: float = 0.01,
                 device: str = "cpu") -> TC_GNN:
    model = TC_GNN(node_feature_dim=node_features.shape[1],
                   hidden_dim=hidden_dim, n_gnn_layers=2).to(device)
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
        opt.zero_grad()
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(val, node_features, node_id_map)
            val_loss = constraint_regularized_loss(
                val_logits, val_labels, val_times, val_amounts,
                lambda_temp=lambda_temp, lambda_val=lambda_val,
            ).item()
        print(f"  [TC-GNN] epoch {ep+1}/{epochs}  train_loss={loss.item():.4f}  val_loss={val_loss:.4f}")
    return model


def train_baseline(name: str, train, val, node_features, node_id_map,
                   epochs: int = 5, lr: float = 1e-3, device: str = "cpu"):
    model = build_baseline(name, in_dim=node_features.shape[1], hidden_dim=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    train_labels = torch.tensor([c["label"] for c in train], dtype=torch.float32).unsqueeze(1)
    for ep in range(epochs):
        model.train()
        logits = model(train, node_features, node_id_map)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, train_labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model


def predict_proba_torch(model, cycles, node_features, node_id_map, device="cpu"):
    model.eval()
    with torch.no_grad():
        logits = model(cycles, node_features, node_id_map)
        return torch.sigmoid(logits).squeeze(-1).cpu().numpy()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(y_true, y_proba, threshold: float = 0.5) -> dict:
    """Compute AUC-ROC, AUC-PR, F1, precision, recall."""
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                  f1_score, precision_score, recall_score)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    out = {"n_samples": int(len(y_true)),
           "n_positive": int(y_true.sum()),
           "n_negative": int((1 - y_true).sum())}
    if out["n_positive"] == 0 or out["n_negative"] == 0:
        out.update({"auc_roc": float("nan"), "auc_pr": float("nan"),
                    "f1": float("nan"), "precision": float("nan"),
                    "recall": float("nan")})
        return out
    y_pred = (y_proba >= threshold).astype(int)
    out["auc_roc"]   = float(roc_auc_score(y_true, y_proba))
    out["auc_pr"]    = float(average_precision_score(y_true, y_proba))
    out["f1"]        = float(f1_score(y_true, y_pred, zero_division=0))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"]    = float(recall_score(y_true, y_pred, zero_division=0))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(quick: bool, data_path: str, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load
    print("[1] Loading dataset...")
    g = load_auto() if data_path is None else load_auto()
    print(f"    {g.name}: nodes={g.n_nodes}, edges={g.n_edges}, illicit={g.n_illicit}")

    # 2. Generate candidates
    print("[2] Generating candidate cycles via TemporalCycleSketch...")
    if quick:
        candidates = sketch_candidates(g.edges, width=8, depth=2, window_size=7, epsilon=0.5)
    else:
        candidates = sketch_candidates(g.edges, width=16, depth=4, window_size=7, epsilon=0.5)
    print(f"    found {len(candidates)} raw candidates from sketch")

    # 2b. Inject ground-truth cycles as positives (if available)
    if g.ground_truth_cycles:
        gt_candidates = [{
            "nodes": c["nodes"], "times": c["times"], "amounts": c["amounts"],
            "value_imbalance": 0.0, "length": len(c["nodes"]),
            "time_span": max(c["times"]) - min(c["times"]),
        } for c in g.ground_truth_cycles]
        candidates = candidates + gt_candidates
    print(f"    with ground-truth cycles: {len(candidates)} total")

    # 2c. Generate NEGATIVE cycles: random node sets that satisfy constraints
    # but are NOT in the ground-truth set. This is the realistic AML scenario:
    # many "look-like-laundering" cycles that are actually innocuous (e.g.,
    # circular supplier payments within a corporate group).
    print("[2d] Generating negative cycles via random node subsets...")
    rng = np.random.default_rng(123)
    all_node_ids = g.nodes["txId"].tolist()
    n_positives = len(candidates)
    n_negatives_target = max(n_positives, 500)

    # Build set of ground-truth node sets for exclusion
    gt_node_sets = {frozenset(c["nodes"]) for c in g.ground_truth_cycles}

    neg_cycles = []
    attempts = 0
    max_attempts = n_negatives_target * 10
    while len(neg_cycles) < n_negatives_target and attempts < max_attempts:
        attempts += 1
        k = int(rng.integers(3, 6))
        nodes = list(rng.choice(all_node_ids, size=k, replace=False))
        if frozenset(nodes) in gt_node_sets:
            continue
        t0 = int(rng.integers(0, g.n_edges and int(g.edges["time_step"].max()) - k - 2))
        times = sorted(rng.choice(range(t0, t0 + k + 5), size=k, replace=False))
        base = float(rng.uniform(10, 1000))
        amounts = list(base * np.exp(rng.normal(0, 0.03, size=k)))
        neg_cycles.append({
            "nodes": [int(n) for n in nodes],
            "times": [int(t) for t in times],
            "amounts": [float(a) for a in amounts],
            "value_imbalance": 0.0,
            "length": k,
            "time_span": times[-1] - times[0],
        })
    candidates = candidates + neg_cycles
    print(f"    +{len(neg_cycles)} negatives, total={len(candidates)}")

    # 3. Label
    print("[3] Labeling candidates...")
    illicit_set = g.illicit_node_set()
    candidates = label_candidates(candidates, illicit_set)
    n_pos = sum(c["label"] for c in candidates)
    n_neg = len(candidates) - n_pos
    print(f"    {n_pos} positive, {n_neg} negative")

    if n_pos < 5 or n_neg < 5:
        print(f"    WARNING: too few positives/negatives ({n_pos}/{n_neg}); aborting")
        return

    # 4. Split
    print("[4] Temporal split...")
    train, val, test = temporal_split(candidates)
    print(f"    train={len(train)}, val={len(val)}, test={len(test)}")

    # 5. Build features
    print("[5] Building node features...")
    node_features = build_node_features(g.edges, g.n_nodes, dim=16)
    node_id_map = make_node_id_map(candidates)
    print(f"    feature dim={node_features.shape[1]}, unique nodes={len(node_id_map)}")

    # 6. Train + eval all models
    epochs = 5 if quick else 15
    results = []

    print("\n[6a] Training TC-GNN...")
    t0 = time.time()
    tc_gnn = train_tc_gnn(train, val, node_features, node_id_map, g.n_nodes,
                          epochs=epochs, lr=5e-3,
                          lambda_temp=0.005, lambda_val=0.005)
    y_proba = predict_proba_torch(tc_gnn, test, node_features, node_id_map)
    metrics = evaluate([c["label"] for c in test], y_proba)
    metrics["model"] = "TC-GNN"
    metrics["train_time_sec"] = round(time.time() - t0, 2)
    print(f"    TC-GNN: AUC-ROC={metrics['auc_roc']:.4f}, AUC-PR={metrics['auc_pr']:.4f}, F1={metrics['f1']:.4f}")
    results.append(metrics)

    for baseline_name in ["GCN", "GAT", "TGN", "DCRNN", "GLASS"]:
        print(f"\n[6b] Training {baseline_name}...")
        t0 = time.time()
        model = train_baseline(baseline_name, train, val, node_features, node_id_map, epochs=epochs)
        y_proba = predict_proba_torch(model, test, node_features, node_id_map)
        metrics = evaluate([c["label"] for c in test], y_proba)
        metrics["model"] = baseline_name
        metrics["train_time_sec"] = round(time.time() - t0, 2)
        print(f"    {baseline_name}: AUC-ROC={metrics['auc_roc']:.4f}, AUC-PR={metrics['auc_pr']:.4f}, F1={metrics['f1']:.4f}")
        results.append(metrics)

    print("\n[6c] Training XGBoost...")
    t0 = time.time()
    xgb = XGBoostCycleClassifier(n_estimators=50, max_depth=4)
    xgb.fit(train, np.array([c["label"] for c in train]))
    y_proba = xgb.predict_proba(test)[:, 1]
    metrics = evaluate([c["label"] for c in test], y_proba)
    metrics["model"] = "XGBoost"
    metrics["train_time_sec"] = round(time.time() - t0, 2)
    print(f"    XGBoost: AUC-ROC={metrics['auc_roc']:.4f}, AUC-PR={metrics['auc_pr']:.4f}, F1={metrics['f1']:.4f}")
    results.append(metrics)

    # 7. Save
    print("\n[7] Saving results...")
    df = pd.DataFrame(results)
    cols = ["model"] + [c for c in df.columns if c != "model"]
    df = df[cols]
    df.to_csv(out_dir / "main_results.csv", index=False)
    with open(out_dir / "experiment_meta.json", "w") as f:
        json.dump({
            "data_source": g.name,
            "n_nodes": g.n_nodes, "n_edges": g.n_edges, "n_illicit": g.n_illicit,
            "n_candidates": len(candidates), "n_pos": n_pos, "n_neg": n_neg,
            "epochs": epochs, "hidden_dim": 32,
        }, f, indent=2)
    print(f"    Saved to {out_dir}/main_results.csv")
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use small data + few epochs")
    parser.add_argument("--full", action="store_true", help="Use larger data + more epochs")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--out-dir", type=str, default="results")
    args = parser.parse_args()
    main(quick=args.quick, data_path=args.data_path, out_dir=args.out_dir)