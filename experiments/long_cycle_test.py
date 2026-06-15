"""Long-cycle stress test: evaluate TC-GNN when positives have k=10+ hops.

Usage:
  python experiments/long_cycle_test.py --k-min 8 --k-max 12 --n-pos 200
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

from experiments.sqlite_data_layer import (
    sqlite_conn, fetch_candidates, candidate_count,
    build_compact_features_from_db, inject_synthetic_cycles_db,
    generate_negative_cycles_db,
)
from experiments.run_experiment_sqlite import temporal_split, remap_cycles
from experiments.run_experiment_sqlite import _pad
from experiments.optimize_tc_gnn import train_optimized, evaluate
from experiments.statistical_tests import delong_test, bootstrap_auc_ci
from src.temporal_cycle.tc_gnn import TC_GNN


def run_test(k_min, k_max, n_pos, n_neg, epochs, hidden_dim, db_path, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1] Building test set with k={k_min}-{k_max} positives...")
    # First clear candidates
    with sqlite_conn(db_path) as conn:
        conn.execute("DELETE FROM candidates")
        # Inject long-cycle positives
        n_inj = inject_synthetic_cycles_db(
            conn, n_cycles=n_pos,
            cycle_len_range=(k_min, k_max),
            time_window=20,  # wider window for longer cycles
            seed=42,
        )
        print(f"    injected {n_inj} long-cycle positives")

        # Generate negatives (realistic_aml + a few trivially violating)
        n_neg_gen = generate_negative_cycles_db(
            conn, n_cycles=n_neg, mode="realistic_aml",
            cycle_len_range=(k_min, k_max),  # similar length range
        )
        print(f"    generated {n_neg_gen} negatives")

        # Stats by cycle length
        cands = fetch_candidates(conn)
        print(f"    total candidates: {len(cands)}")
        from collections import Counter
        pos_lens = [c["length"] for c in cands if c["label"] == 1]
        neg_lens = [c["length"] for c in cands if c["label"] == 0]
        print(f"    pos length dist: {sorted(Counter(pos_lens).items())}")
        print(f"    neg length dist: {sorted(Counter(neg_lens).items())}")

    # Load + features + split
    print(f"[2] Loading and splitting...")
    with sqlite_conn(db_path) as conn:
        candidates = fetch_candidates(conn)
    train, val, test = temporal_split(candidates)
    feats, node_id_map = build_compact_features_from_db(
        db_path, candidates, dim=165, use_real_features=True)
    feats = (feats - feats.mean(axis=0, keepdims=True)) / np.clip(feats.std(axis=0, keepdims=True), 1e-6, None)
    feats_t = torch.tensor(feats, dtype=torch.float32)
    identity = {i: i for i in range(len(node_id_map))}
    train = remap_cycles(train, node_id_map)
    val   = remap_cycles(val, node_id_map)
    test  = remap_cycles(test, node_id_map)
    print(f"    features: {feats_t.shape}")

    # Train TC-GNN-opt
    print(f"[3] Training TC-GNN-opt (hidden={hidden_dim}, epochs={epochs})...")
    t0 = time.time()
    model, proba_opt, history = train_optimized(
        train, val, test, feats_t, identity,
        epochs=epochs, hidden_dim=hidden_dim,
        use_focal=True, lambda_temp=0.01, lambda_val=0.01,
    )
    train_time = time.time() - t0

    test_y = np.array([c["label"] for c in test])
    metrics_opt = evaluate(test_y, proba_opt, threshold=0.5)
    metrics_opt["model"] = f"TC-GNN-opt(k={k_min}-{k_max})"
    metrics_opt["train_time_sec"] = round(train_time, 2)

    # Also run baseline (no focal, less hidden) for comparison
    print(f"[4] Training TC-GNN-base for comparison...")
    base_model = TC_GNN(node_feature_dim=feats_t.shape[1],
                        hidden_dim=32, n_gnn_layers=2, dropout=0.2)
    opt = torch.optim.AdamW(base_model.parameters(), lr=5e-3)
    train_y = torch.tensor([c["label"] for c in train], dtype=torch.float32).unsqueeze(1)
    train_t = _pad(train, "times")
    train_a = _pad(train, "amounts")
    val_y = torch.tensor([c["label"] for c in val], dtype=torch.float32).unsqueeze(1)
    val_t = _pad(val, "times")
    val_a = _pad(val, "amounts")
    from src.temporal_cycle.losses import constraint_regularized_loss
    for ep in range(epochs):
        base_model.train()
        logits = base_model(train, feats_t, identity)
        loss = constraint_regularized_loss(logits, train_y, train_t, train_a)
        opt.zero_grad(); loss.backward(); opt.step()
    base_model.eval()
    with torch.no_grad():
        proba_base = torch.sigmoid(base_model(test, feats_t, identity)).squeeze(-1).cpu().numpy()
    metrics_base = evaluate(test_y, proba_base, threshold=0.5)
    metrics_base["model"] = f"TC-GNN-base(k={k_min}-{k_max})"
    metrics_base["train_time_sec"] = round(time.time() - t0, 2)

    # Save
    df = pd.DataFrame([metrics_base, metrics_opt])
    csv_path = out_dir / f"long_cycle_k{k_min}-{k_max}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[5] Saved to {csv_path}")
    print("\n" + df.to_string(index=False))
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k-min", type=int, default=8)
    p.add_argument("--k-max", type=int, default=12)
    p.add_argument("--n-pos", type=int, default=200)
    p.add_argument("--n-neg", type=int, default=200)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--db", type=str, default="data/elliptic1.db")
    p.add_argument("--out-dir", type=str, default="results")
    args = p.parse_args()
    run_test(args.k_min, args.k_max, args.n_pos, args.n_neg,
             args.epochs, args.hidden_dim, args.db, args.out_dir)


if __name__ == "__main__":
    main()