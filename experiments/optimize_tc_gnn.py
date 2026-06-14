"""Optimized TC-GNN training with focal loss, class weights, and LR scheduling.

Improvements over baseline TC-GNN:
  - Focal loss (gamma=2) to handle class imbalance
  - Class-weighted BCE based on inverse-frequency
  - Cosine annealing LR schedule
  - Early stopping on val AUC-PR
  - Larger hidden dim (64 instead of 32)
  - Gradient clipping for stability
  - Optional class-balanced sampling
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
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.sqlite_data_layer import (
    sqlite_conn, fetch_candidates, candidate_count,
    build_compact_features_from_db,
)
from src.temporal_cycle.tc_gnn import TC_GNN


# ---------------------------------------------------------------------------
# Loss variants
# ---------------------------------------------------------------------------
def focal_loss_with_logits(logits, labels, gamma: float = 2.0, alpha: float = 0.25):
    """Binary focal loss: down-weights easy examples, focuses on hard."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    p_t = p * labels + (1 - p) * (1 - labels)
    alpha_t = alpha * labels + (1 - alpha) * (1 - labels)
    focal = alpha_t * (1 - p_t) ** gamma * ce
    return focal.mean()


# ---------------------------------------------------------------------------
# Optimized training loop
# ---------------------------------------------------------------------------
def train_optimized(
    train, val, test, node_features, identity_map,
    epochs: int = 30,
    lr: float = 1e-3,
    hidden_dim: int = 64,
    lambda_temp: float = 0.01,
    lambda_val: float = 0.01,
    use_focal: bool = True,
    use_class_weight: bool = True,
    gamma: float = 2.0,
    alpha: float = 0.5,
    patience: int = 5,
):
    """Train TC-GNN with the full optimization stack."""
    # Compute class weights
    train_labels = [c["label"] for c in train]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32) if use_class_weight else None

    model = TC_GNN(node_feature_dim=node_features.shape[1],
                   hidden_dim=hidden_dim, n_gnn_layers=2, dropout=0.3)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    train_y = torch.tensor(train_labels, dtype=torch.float32).unsqueeze(1)
    train_t = _pad(train, "times")
    train_a = _pad(train, "amounts")
    val_y = torch.tensor([c["label"] for c in val], dtype=torch.float32).unsqueeze(1)
    val_t = _pad(val, "times")
    val_a = _pad(val, "amounts")
    test_y = [c["label"] for c in test]

    best_val_aucpr = -1.0
    best_state = None
    bad_epochs = 0
    history = {"train_loss": [], "val_loss": [], "val_auc_pr": []}

    for ep in range(epochs):
        model.train()
        logits = model(train, node_features, identity_map)
        if use_focal:
            cls_loss = focal_loss_with_logits(logits, train_y, gamma=gamma, alpha=alpha)
        else:
            cls_loss = F.binary_cross_entropy_with_logits(logits, train_y,
                                                          pos_weight=pos_weight)
        # Constraint losses (per-cycle averaged)
        if train_t.numel() > 0:
            diffs = train_t[:, :-1] - train_t[:, 1:]
            l_temp = torch.clamp(diffs, min=0.0).sum(dim=1).mean()
        else:
            l_temp = torch.tensor(0.0)
        if train_a.numel() > 0:
            mean = train_a.mean(dim=1, keepdim=True).clamp(min=1e-9)
            dev = (train_a - mean).abs()
            max_dev = dev.max(dim=1).values
            l_val = (max_dev / mean.squeeze(1)).mean()
        else:
            l_val = torch.tensor(0.0)
        loss = cls_loss + lambda_temp * l_temp + lambda_val * l_val

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        # Validation
        model.eval()
        with torch.no_grad():
            v_logits = model(val, node_features, identity_map)
            v_loss = cls_loss_fn(v_logits, val_y, val_t, val_a,
                                  use_focal, gamma, alpha, lambda_temp, lambda_val)
            v_proba = torch.sigmoid(v_logits).squeeze(-1).cpu().numpy()
            from sklearn.metrics import average_precision_score
            v_aucpr = average_precision_score([c["label"] for c in val], v_proba)

        history["train_loss"].append(loss.item())
        history["val_loss"].append(v_loss)
        history["val_auc_pr"].append(v_aucpr)

        improved = v_aucpr > best_val_aucpr
        if improved:
            best_val_aucpr = v_aucpr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        print(f"  [TC-GNN-opt] ep {ep+1:>2}/{epochs}  train={loss.item():.4f}  "
              f"val={v_loss:.4f}  val_AUC-PR={v_aucpr:.4f}  "
              f"{'↑ best' if improved else f'({bad_epochs}/{patience})'}")

        if bad_epochs >= patience:
            print(f"  Early stop at epoch {ep+1}")
            break

    # Restore best
    if best_state is not None:
        model.load_state_dict({k: v.to(node_features.device) for k, v in best_state.items()})

    # Final test
    model.eval()
    with torch.no_grad():
        t_logits = model(test, node_features, identity_map)
        t_proba = torch.sigmoid(t_logits).squeeze(-1).cpu().numpy()

    return model, t_proba, history


def cls_loss_fn(logits, labels, times, amounts, use_focal, gamma, alpha,
                lambda_temp, lambda_val):
    """Combined loss for validation (matches training)."""
    if use_focal:
        cls_loss = focal_loss_with_logits(logits, labels, gamma=gamma, alpha=alpha)
    else:
        cls_loss = F.binary_cross_entropy_with_logits(logits, labels)
    if times.numel() > 0:
        diffs = times[:, :-1] - times[:, 1:]
        l_temp = torch.clamp(diffs, min=0.0).sum(dim=1).mean()
    else:
        l_temp = torch.tensor(0.0)
    if amounts.numel() > 0:
        mean = amounts.mean(dim=1, keepdim=True).clamp(min=1e-9)
        dev = (amounts - mean).abs()
        max_dev = dev.max(dim=1).values
        l_val = (max_dev / mean.squeeze(1)).mean()
    else:
        l_val = torch.tensor(0.0)
    return (cls_loss + lambda_temp * l_temp + lambda_val * l_val).item()


def _pad(cycles, key, pad=0.0):
    seqs = [c[key] for c in cycles]
    if not seqs:
        return torch.zeros((0, 0))
    L = max(len(s) for s in seqs)
    out = torch.full((len(seqs), L), pad, dtype=torch.float32)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = torch.tensor(s, dtype=torch.float32)
    return out


def temporal_split(candidates, val_ratio=0.15, test_ratio=0.15):
    by_time = sorted(candidates, key=lambda c: max(c["times"]))
    n = len(by_time)
    n_test = int(n * test_ratio)
    n_val  = int(n * val_ratio)
    return (by_time[:n - n_test - n_val],
            by_time[n - n_test - n_val: n - n_test],
            by_time[n - n_test:])


def remap_cycles(cycles, node_id_map):
    return [{
        "nodes": [node_id_map[n] for n in c["nodes"]],
        "times": c["times"],
        "amounts": c["amounts"],
        "value_imbalance": c.get("value_imbalance", 0.0),
        "length": c.get("length", len(c["nodes"])),
        "time_span": c.get("time_span", 0),
        "label": c["label"],
    } for c in cycles]


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


def main(epochs=30, db_path="data/elliptic1.db", hidden_dim=64,
         use_focal=True, lambda_temp=0.01, lambda_val=0.01, out_dir="results",
         threshold: float = 0.5):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[1] Loading candidates from {db_path}...")
    with sqlite_conn(db_path) as conn:
        candidates = fetch_candidates(conn)
    print(f"    total: {len(candidates)}")

    train, val, test = temporal_split(candidates)
    feats, node_id_map = build_compact_features_from_db(db_path, candidates, dim=165,
                                                       use_real_features=True)
    # Z-score normalize features (helps TC-GNN's linear input projection)
    feats_mean = feats.mean(axis=0, keepdims=True)
    feats_std  = np.clip(feats.std(axis=0, keepdims=True), 1e-6, None)
    feats = (feats - feats_mean) / feats_std
    feats = torch.tensor(feats, dtype=torch.float32)
    identity = {i: i for i in range(len(node_id_map))}
    train = remap_cycles(train, node_id_map)
    val   = remap_cycles(val, node_id_map)
    test  = remap_cycles(test, node_id_map)
    print(f"    features shape: {feats.shape} (z-score normalized)")

    print(f"\n[2] Optimized TC-GNN training (hidden={hidden_dim}, focal={use_focal}, "
          f"lambda_temp={lambda_temp}, lambda_val={lambda_val})...")
    t0 = time.time()
    model, test_proba, history = train_optimized(
        train, val, test, feats, identity,
        epochs=epochs, hidden_dim=hidden_dim,
        use_focal=use_focal, lambda_temp=lambda_temp, lambda_val=lambda_val,
    )
    train_time = time.time() - t0

    metrics = evaluate([c["label"] for c in test], test_proba, threshold=threshold)
    metrics["model"] = f"TC-GNN-opt(h={hidden_dim},focal={use_focal},thr={threshold})"
    metrics["train_time_sec"] = round(train_time, 2)

    print(f"\n[3] Test metrics:")
    print(f"    AUC-ROC={metrics['auc_roc']:.4f}, AUC-PR={metrics['auc_pr']:.4f}, "
          f"F1={metrics['f1']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}")

    # Save
    df = pd.DataFrame([metrics])
    out_csv = out_dir / "results_tc_gnn_optimized.csv"
    df.to_csv(out_csv, index=False)
    with open(out_dir / "tc_gnn_optimized_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[4] Saved to {out_csv}")
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--no-focal", action="store_true")
    p.add_argument("--lambda-temp", type=float, default=0.01)
    p.add_argument("--lambda-val", type=float, default=0.01)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--db", type=str, default="data/elliptic1.db")
    p.add_argument("--out-dir", type=str, default="results")
    args = p.parse_args()
    main(args.epochs, args.db, args.hidden_dim,
         use_focal=not args.no_focal,
         lambda_temp=args.lambda_temp, lambda_val=args.lambda_val,
         out_dir=args.out_dir,
         threshold=args.threshold)