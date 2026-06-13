"""Constraint-regularized losses for TC-GNN (paper §5.2 layer 3)."""
from __future__ import annotations

import torch
import torch.nn as nn


def temporal_increasing_penalty(times: torch.Tensor) -> torch.Tensor:
    """L_temp = sum over cycles and adjacent pairs of max(0, t_i - t_{i+1}).

    times: (B, L) tensor where B = batch, L = cycle length (edge times).
    """
    if times.numel() == 0:
        return torch.tensor(0.0, requires_grad=True)
    diffs = times[:, :-1] - times[:, 1:]  # (B, L-1)
    return torch.clamp(diffs, min=0.0).sum()


def value_conservation_penalty(amounts: torch.Tensor) -> torch.Tensor:
    """L_val = sum over cycles of max|a - mean(a)| / mean(a).

    amounts: (B, L) tensor of per-edge amounts.
    """
    if amounts.numel() == 0:
        return torch.tensor(0.0, requires_grad=True)
    mean = amounts.mean(dim=1, keepdim=True).clamp(min=1e-9)  # (B, 1)
    dev = (amounts - mean).abs()
    max_dev = dev.max(dim=1).values  # (B,)
    return (max_dev / mean.squeeze(1)).sum()


def constraint_regularized_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    times: torch.Tensor,
    amounts: torch.Tensor,
    lambda_temp: float = 1.0,
    lambda_val: float = 1.0,
) -> torch.Tensor:
    """L = L_cls + λ1·L_temp + λ2·L_val.

    logits:  (B, 1) raw model output (BCE-with-logits).
    labels:  (B, 1) ground-truth (0/1).
    times:   (B, L) cycle edge timestamps.
    amounts: (B, L) cycle edge amounts.
    """
    bce = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    l_temp = temporal_increasing_penalty(times)
    l_val  = value_conservation_penalty(amounts)
    return bce + lambda_temp * l_temp + lambda_val * l_val