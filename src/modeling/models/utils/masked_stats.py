"""Shared masked statistics utilities for the predict-based models."""

from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor


def masked_mean(
    values: Tensor,
    mask_f: Tensor,
    mask_count: Tensor,
    mask_any: bool,
) -> Tensor:
    """Compute mean of values at masked positions."""
    if not mask_any:
        return values.new_zeros(())
    return (values * mask_f).sum() / mask_count


def masked_var(
    values: Tensor,
    mask_f: Tensor,
    mask_count: Tensor,
    mask_any: bool,
) -> Tensor:
    """Compute variance of values at masked positions."""
    if not mask_any:
        return values.new_zeros(())
    mean = (values * mask_f).sum() / mask_count
    return (((values - mean) ** 2) * mask_f).sum() / mask_count


def add_distribution_stats(
    stats: Dict[str, Tensor],
    prefix: str,
    values: Tensor,
) -> None:
    """Add mean/var/min/max/percentile stats for a 1-D tensor into stats dict."""
    vf = values.detach().float()
    stats[f"{prefix}_mean"] = vf.mean()
    stats[f"{prefix}_var"] = vf.var(unbiased=False)
    stats[f"{prefix}_min"] = vf.min()
    stats[f"{prefix}_max"] = vf.max()
    _add_quantile_distribution_stats(stats, prefix, vf)


def add_empty_distribution_stats(
    stats: Dict[str, Tensor],
    prefix: str,
    device: torch.device | str = "cpu",
) -> None:
    """Add NaN-filled distribution stats when there are no values.

    Ensures the stats dict has the same keys as ``add_distribution_stats``
    regardless of whether data was present, which is required for DDP
    all_reduce to use identically-shaped packed tensors on every rank.
    """
    nan = torch.tensor(float("nan"), device=device)
    stats[f"{prefix}_mean"] = nan
    stats[f"{prefix}_var"] = nan
    stats[f"{prefix}_min"] = nan
    stats[f"{prefix}_max"] = nan
    for p in (25, 50, 75, 90, 99):
        stats[f"{prefix}_p{p}"] = nan


@torch._dynamo.disable
def _add_quantile_distribution_stats(
    stats: Dict[str, Tensor],
    prefix: str,
    values: Tensor,
) -> None:
    """Compute percentile stats eagerly to avoid Dynamo symbolic-shape failures."""
    for p in (25, 50, 75, 90, 99):
        stats[f"{prefix}_p{p}"] = values.quantile(p / 100.0)
