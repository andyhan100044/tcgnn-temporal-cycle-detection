"""Statistical significance: Bootstrap 95% CI + DeLong test for AUC comparison.

Usage:
  python experiments/statistical_tests.py --run-all
  python experiments/statistical_tests.py --proba-csv results/probabilities.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


# ---------------------------------------------------------------------------
# Bootstrap 95% CI for AUC
# ---------------------------------------------------------------------------
def bootstrap_auc_ci(
    y_true: np.ndarray, y_proba: np.ndarray,
    metric: str = "roc_auc", n_boot: int = 1000, alpha: float = 0.05, seed: int = 42,
) -> Tuple[float, float, float]:
    """Bootstrap (n_boot resamples) for 95% CI of AUC."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        y_t = y_true[idx]
        y_p = y_proba[idx]
        if len(np.unique(y_t)) < 2:
            continue
        if metric == "roc_auc":
            scores.append(roc_auc_score(y_t, y_p))
        else:
            scores.append(average_precision_score(y_t, y_p))
    if not scores:
        return float("nan"), float("nan"), float("nan")
    scores = np.array(scores)
    point = (roc_auc_score if metric == "roc_auc" else average_precision_score)(y_true, y_proba)
    lo = float(np.percentile(scores, 100 * alpha / 2))
    hi = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return float(point), lo, hi


# ---------------------------------------------------------------------------
# DeLong test for comparing two AUCs on the same data
# ---------------------------------------------------------------------------
def _compute_midrank(x):
    """Midrank for tied values (DeLong helper)."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        for k in range(i, j):
            T[k] = 0.5 * (i + j - 1) + 0.5
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1  # ranks 1-indexed
    return T2


def delong_test(y_true: np.ndarray, y_proba1: np.ndarray, y_proba2: np.ndarray):
    """DeLong test for two correlated AUCs (same test set).

    Returns: (auc1, auc2, z_score, p_value)
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba1 = np.asarray(y_proba1, dtype=float)
    y_proba2 = np.asarray(y_proba2, dtype=float)

    pos = y_true == 1
    neg = y_true == 0
    m = pos.sum()  # positives
    n = neg.sum()  # negatives
    if m == 0 or n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    # Placement values: V_i = (1/n) * #{j : X_j < X_i} for positive i
    #                   W_j = (1/m) * #{i : X_i > X_j} for negative j
    X = np.concatenate([y_proba1[pos], y_proba1[neg]])
    Y = np.concatenate([y_proba2[pos], y_proba2[neg]])

    # Compute AUC via Mann-Whitney (placement values)
    def _placement_values(probs):
        # V for positives
        ps = probs[:m]
        ns = probs[m:]
        V = np.zeros(m)
        for i, p in enumerate(ps):
            V[i] = np.mean((ns < p).astype(float) + 0.5 * (ns == p).astype(float))
        # W for negatives
        W = np.zeros(n)
        for j, p in enumerate(ns):
            W[j] = np.mean((ps > p).astype(float) + 0.5 * (ps == p).astype(float))
        return V, W

    V1, W1 = _placement_values(X)
    V2, W2 = _placement_values(Y)

    auc1 = float(V1.mean())
    auc2 = float(V2.mean())

    # Covariance matrix of (V, W) components
    S10 = np.cov(V1, V2, ddof=1)[0, 1]
    S01 = np.cov(W1, W2, ddof=1)[0, 1]
    var = S10 / m + S01 / n
    if var <= 0:
        return auc1, auc2, float("nan"), float("nan")
    z = (auc1 - auc2) / np.sqrt(var)
    from scipy.stats import norm
    p_value = 2 * (1 - norm.cdf(abs(z)))
    return auc1, auc2, float(z), float(p_value)


# ---------------------------------------------------------------------------
# Run all models, get predictions, compute statistics
# ---------------------------------------------------------------------------
def collect_predictions_from_csv(csv_path: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load predictions from results/results_sqlite.csv (per-model probabilities).

    For real statistics, we need per-sample predictions. This function
    re-trains each model and saves predictions to a file.
    """
    raise NotImplementedError("Use --run-all instead")


def run_all_and_save(db_path: str, out_csv: str = "results/probabilities.parquet"):
    """Train all 7 models, save (y_true, y_proba_per_model) to parquet."""
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from experiments.run_experiment_sqlite import (
        sqlite_conn, fetch_candidates, candidate_count,
        build_compact_features_from_db, temporal_split, remap_cycles,
        train_tc_gnn, train_baseline, predict, _pad,
    )
    from experiments.optimize_tc_gnn import (
        train_optimized, _pad as opt_pad, evaluate,
    )
    from experiments.baselines.xgb import XGBoostCycleClassifier
    from src.temporal_cycle.tc_gnn import TC_GNN

    print("[1] Load data...")
    with sqlite_conn(db_path) as conn:
        candidates = fetch_candidates(conn)
    train, val, test = temporal_split(candidates)
    feats, node_id_map = build_compact_features_from_db(db_path, candidates, dim=165,
                                                       use_real_features=True)
    # Z-score normalize
    feats = (feats - feats.mean(axis=0, keepdims=True)) / np.clip(feats.std(axis=0, keepdims=True), 1e-6, None)
    feats_t = torch.tensor(feats, dtype=torch.float32)
    identity = {i: i for i in range(len(node_id_map))}
    train = remap_cycles(train, node_id_map)
    val   = remap_cycles(val, node_id_map)
    test  = remap_cycles(test, node_id_map)

    test_y = np.array([c["label"] for c in test])
    probas: Dict[str, np.ndarray] = {}

    print("[2] Train TC-GNN (base)...")
    tc = train_tc_gnn(train, val, feats_t, identity, epochs=10)
    probas["TC-GNN"] = predict(tc, test, feats_t, identity)

    print("[3] Train TC-GNN (optimized)...")
    tc_opt, proba_opt, _ = train_optimized(
        train, val, test, feats_t, identity,
        epochs=30, hidden_dim=64, use_focal=True,
        lambda_temp=0.01, lambda_val=0.01,
    )
    probas["TC-GNN-opt"] = proba_opt

    for name in ["GCN", "GAT", "TGN", "DCRNN", "GLASS"]:
        print(f"[4] Training {name}...")
        m = train_baseline(name, train, val, feats_t, identity, epochs=10)
        probas[name] = predict(m, test, feats_t, identity)

    print("[5] Training XGBoost...")
    xgb = XGBoostCycleClassifier(n_estimators=50, max_depth=4)
    xgb.fit(train, np.array([c["label"] for c in train]))
    probas["XGBoost"] = xgb.predict_proba(test)[:, 1]

    # Save
    df = pd.DataFrame({"y_true": test_y})
    for name, p in probas.items():
        df[f"proba_{name}"] = p
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[6] Saved to {out_path}")
    return df


def compute_all_stats(proba_csv: str, out_csv: str):
    """Compute bootstrap CI + DeLong for all model pairs."""
    df = pd.read_parquet(proba_csv)
    y_true = df["y_true"].values
    models = [c.replace("proba_", "") for c in df.columns if c.startswith("proba_")]
    print(f"Models: {models}")

    # Bootstrap CI for each model
    print("\n=== Bootstrap 95% CI ===")
    ci_rows = []
    for m in models:
        proba = df[f"proba_{m}"].values
        roc_pt, roc_lo, roc_hi = bootstrap_auc_ci(y_true, proba, "roc_auc")
        pr_pt, pr_lo, pr_hi = bootstrap_auc_ci(y_true, proba, "pr_auc")
        ci_rows.append({
            "model": m,
            "auc_roc": roc_pt, "auc_roc_lo": roc_lo, "auc_roc_hi": roc_hi,
            "auc_pr": pr_pt, "auc_pr_lo": pr_lo, "auc_pr_hi": pr_hi,
        })
        print(f"  {m:>12}: AUC-ROC={roc_pt:.3f} [{roc_lo:.3f}, {roc_hi:.3f}], "
              f"AUC-PR={pr_pt:.3f} [{pr_lo:.3f}, {pr_hi:.3f}]")
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(out_csv, index=False)
    print(f"\nSaved CI to {out_csv}")

    # DeLong test: TC-GNN-opt vs each baseline
    print("\n=== DeLong test: TC-GNN-opt vs baselines ===")
    if "TC-GNN-opt" not in models:
        print("  [skip] TC-GNN-opt not in models")
        return ci_df, pd.DataFrame()
    tc_proba = df["proba_TC-GNN-opt"].values
    delong_rows = []
    for m in models:
        if m == "TC-GNN-opt":
            continue
        other_proba = df[f"proba_{m}"].values
        auc1, auc2, z, p = delong_test(y_true, tc_proba, other_proba)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        delong_rows.append({
            "comparison": f"TC-GNN-opt vs {m}",
            "auc_tc_gnn": auc1, "auc_other": auc2,
            "delta_auc": auc1 - auc2, "z_score": z, "p_value": p, "significance": sig,
        })
        print(f"  TC-GNN-opt vs {m:>10}: Δ AUC={auc1-auc2:+.3f}, z={z:.2f}, p={p:.4f} {sig}")
    delong_df = pd.DataFrame(delong_rows)
    out_delong = out_csv.replace(".csv", "_delong.csv")
    delong_df.to_csv(out_delong, index=False)
    print(f"\nSaved DeLong to {out_delong}")
    return ci_df, delong_df


def main(run_all: bool, proba_csv: str, out_csv: str):
    if run_all:
        proba_csv = proba_csv or "results/probabilities.parquet"
        print("=== Training all models and saving predictions ===")
        run_all_and_save("data/elliptic1.db", proba_csv)
    compute_all_stats(proba_csv, out_csv or "results/statistical_tests.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-all", action="store_true")
    p.add_argument("--proba-csv", type=str, default="results/probabilities.parquet")
    p.add_argument("--out-csv", type=str, default="results/statistical_tests.csv")
    args = p.parse_args()
    main(args.run_all, args.proba_csv, args.out_csv)