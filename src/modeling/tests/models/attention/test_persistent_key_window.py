from __future__ import annotations

import math

import pytest
import torch

from modeling.models.attention.triton_reverse_sps_flash_attention import (
    reverse_sps_sliding_attention,
)
from modeling.models.attention.triton_sps_flash_attention import sps_sliding_attention


def _reference_persistent_key_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    persistent_key_window: int,
    persistent_key_remainder: int,
    temporary_key_window: int,
) -> torch.Tensor:
    _, _, n_ctx, _ = q.shape
    q_idx = torch.arange(n_ctx, device=q.device)
    k_idx = torch.arange(n_ctx, device=q.device)
    q_tok = q_idx // 2
    k_tok = k_idx // 2

    rel = q_tok[:, None] - k_tok[None, :]
    normal_bias = torch.where(
        rel > temporary_key_window,
        torch.full((n_ctx, n_ctx), -1.0e6, device=q.device),
        torch.zeros((n_ctx, n_ctx), device=q.device),
    )

    k_is_persistent = (k_idx % 2) == persistent_key_remainder
    persistent_bias = torch.where(
        rel > persistent_key_window,
        torch.full((n_ctx, n_ctx), -1.0e6, device=q.device),
        torch.zeros((n_ctx, n_ctx), device=q.device),
    )
    bias = torch.where(k_is_persistent[None, :], persistent_bias, normal_bias)
    is_self_persistent = (q_idx[:, None] == k_idx[None, :]) & k_is_persistent[None, :]
    bias = torch.where(is_self_persistent, torch.zeros_like(bias), bias)

    causal = q_idx[:, None] >= k_idx[None, :]
    scores = torch.einsum("bhqd,bhkd->bhqk", q.float(), k.float())
    scores = scores * (1.0 / math.sqrt(q.shape[-1])) + bias
    scores = torch.where(causal, scores, torch.full_like(scores, -1.0e6))
    probs = torch.softmax(scores, dim=-1)
    return torch.einsum("bhqk,bhkd->bhqd", probs, v.float()).to(q.dtype)


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("attention_fn", "persistent_key_remainder"),
    [
        (reverse_sps_sliding_attention, 1),
        (sps_sliding_attention, 0),
    ],
)
def test_persistent_key_window_matches_reference_on_cuda(
    attention_fn,
    persistent_key_remainder: int,
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("persistent-key-window Triton comparison requires CUDA")

    torch.manual_seed(0)
    # head_dim must be >= BLOCK_N (the kernel asserts BLOCK_N <= HEAD_DIM); the pinned
    # single autotune config under PYTEST_VERSION uses BLOCK_N=64, so head_dim=16 tripped
    # a compile-time static assert. 64 is the smallest head_dim compatible with that config.
    batch, heads, n_ctx, head_dim = 1, 1, 8, 64
    q = torch.randn(batch, heads, n_ctx, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    temporary_key_window = n_ctx
    persistent_key_window = 1

    actual = attention_fn(
        q,
        k,
        v,
        1.0 / math.sqrt(head_dim),
        temporary_key_window,
        warp_specialize=False,
        persistent_key_window=persistent_key_window,
    )
    expected = _reference_persistent_key_window_attention(
        q,
        k,
        v,
        persistent_key_window=persistent_key_window,
        persistent_key_remainder=persistent_key_remainder,
        temporary_key_window=temporary_key_window,
    )

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
