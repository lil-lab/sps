"""Fused single-launch KV-update kernel for SPS-family decode.

Replaces the per-step Python KV-update path in SPS-family inference. The
un-fused path issues ~20 micro-kernels per layer per step:

  _update_normal_memory:
    - gather old_k from nw.k[b, :, insert_pos[b], :]
    - torch.where(active, new_k, old_k)
    - scatter back to nw.k
    - same for v
    - update nw.len, nw.pos with torch.where + remainder
  _update_predict_memory: similar for br.k / br.v / br.len[b, h]

`fused_sps_kv_update` does all of that in one launch per layer, grid (B, H).
"""

from __future__ import annotations

import torch
from torch import Tensor

from modeling.models.utils.decode_attention import _next_power_of_2

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


if triton is not None and tl is not None:

    @triton.jit
    def _fused_kv_update_kernel(
        nw_k_ptr,            # [B, H, window_storage, D]
        nw_v_ptr,
        nw_len_ptr,          # [B] (per-batch, shared across heads)
        nw_pos_ptr,          # [B]
        br_k_ptr,            # [B, H, retained_storage, D]
        br_v_ptr,
        br_len_ptr,          # [B, H]
        k_window_ptr,        # [B, H, D] new key for window slot
        v_window_ptr,        # [B, H, D]
        k_retained_ptr,      # [B, H, D] new key for retained buffer
        v_retained_ptr,      # [B, H, D]
        active_mask_ptr,     # [B] bool/int8
        n_head: tl.constexpr,
        head_dim: tl.constexpr,
        window_storage: tl.constexpr,
        retained_storage: tl.constexpr,
        window_tokens: tl.constexpr,   # state.normal_window_tokens (== window_storage typically)
        block_d: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        # No d-block dim — head_dim <= 256 so a single block_d covers all of D.

        offs_d = tl.arange(0, block_d)
        d_mask = offs_d < head_dim

        # Read state scalars.
        active = tl.load(active_mask_ptr + pid_b)
        active = active != 0

        # ---- normal window update ----
        # window_tokens > 0 already guaranteed by Python entry (we don't call
        # this kernel when normal_window_tokens == 0).
        nw_len = tl.load(nw_len_ptr + pid_b)
        nw_pos = tl.load(nw_pos_ptr + pid_b)
        full = nw_len >= window_tokens
        insert_pos = tl.where(full, nw_pos, nw_len)

        # Predicated store: write k_window/v_window into nw.k/v at insert_pos.
        nw_kv_off = (pid_b * n_head + pid_h) * window_storage * head_dim + insert_pos * head_dim + offs_d
        new_k_w = tl.load(k_window_ptr + (pid_b * n_head + pid_h) * head_dim + offs_d, mask=d_mask, other=0.0)
        new_v_w = tl.load(v_window_ptr + (pid_b * n_head + pid_h) * head_dim + offs_d, mask=d_mask, other=0.0)
        store_mask = d_mask & active
        tl.store(nw_k_ptr + nw_kv_off, new_k_w, mask=store_mask)
        tl.store(nw_v_ptr + nw_kv_off, new_v_w, mask=store_mask)

        # Update nw.len, nw.pos — only one program per batch (pid_h == 0).
        if pid_h == 0:
            grow = active & (~full)
            advance = active & full
            new_len = tl.where(grow, nw_len + 1, nw_len)
            next_pos = (nw_pos + 1) % window_tokens
            new_pos = tl.where(advance, next_pos, nw_pos)
            tl.store(nw_len_ptr + pid_b, new_len)
            tl.store(nw_pos_ptr + pid_b, new_pos)

        # ---- predict retained update ----
        br_len = tl.load(br_len_ptr + pid_b * n_head + pid_h)
        br_kv_off = (pid_b * n_head + pid_h) * retained_storage * head_dim + br_len * head_dim + offs_d
        new_k_r = tl.load(k_retained_ptr + (pid_b * n_head + pid_h) * head_dim + offs_d, mask=d_mask, other=0.0)
        new_v_r = tl.load(v_retained_ptr + (pid_b * n_head + pid_h) * head_dim + offs_d, mask=d_mask, other=0.0)
        # Bound check: only write if br_len < retained_storage (defensive).
        in_bounds = br_len < retained_storage
        retain_store_mask = d_mask & active & in_bounds
        tl.store(br_k_ptr + br_kv_off, new_k_r, mask=retain_store_mask)
        tl.store(br_v_ptr + br_kv_off, new_v_r, mask=retain_store_mask)

        # Update br.len[b, h] (one program per (b, h), no race).
        new_br_len = tl.where(active & in_bounds, br_len + 1, br_len)
        tl.store(br_len_ptr + pid_b * n_head + pid_h, new_br_len)


    @triton.jit
    def _fused_kv_update_no_window_kernel(
        br_k_ptr,
        br_v_ptr,
        br_len_ptr,
        k_retained_ptr,
        v_retained_ptr,
        active_mask_ptr,
        n_head: tl.constexpr,
        head_dim: tl.constexpr,
        retained_storage: tl.constexpr,
        block_d: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_d = tl.arange(0, block_d)
        d_mask = offs_d < head_dim
        active = tl.load(active_mask_ptr + pid_b)
        active = active != 0
        br_len = tl.load(br_len_ptr + pid_b * n_head + pid_h)
        br_kv_off = (pid_b * n_head + pid_h) * retained_storage * head_dim + br_len * head_dim + offs_d
        new_k_r = tl.load(k_retained_ptr + (pid_b * n_head + pid_h) * head_dim + offs_d, mask=d_mask, other=0.0)
        new_v_r = tl.load(v_retained_ptr + (pid_b * n_head + pid_h) * head_dim + offs_d, mask=d_mask, other=0.0)
        in_bounds = br_len < retained_storage
        retain_store_mask = d_mask & active & in_bounds
        tl.store(br_k_ptr + br_kv_off, new_k_r, mask=retain_store_mask)
        tl.store(br_v_ptr + br_kv_off, new_v_r, mask=retain_store_mask)
        new_br_len = tl.where(active & in_bounds, br_len + 1, br_len)
        tl.store(br_len_ptr + pid_b * n_head + pid_h, new_br_len)


def fused_sps_kv_update(
    *,
    nw_k: Tensor,
    nw_v: Tensor,
    nw_len: Tensor,
    nw_pos: Tensor,
    br_k: Tensor,
    br_v: Tensor,
    br_len: Tensor,
    k_window_BxHxD: Tensor,
    v_window_BxHxD: Tensor,
    k_retained_BxHxD: Tensor,
    v_retained_BxHxD: Tensor,
    active_mask_B: Tensor,
    window_tokens: int,
) -> None:
    """Fused single-launch KV update for SPS-family decode.

    Replaces _update_normal_memory + _update_predict_memory. All tensors must
    be CUDA, contiguous, and dtype-compatible (k/v new tensors must match the
    cache dtype).
    """
    if triton is None or tl is None:
        raise RuntimeError("Triton is required for fused_sps_kv_update")

    batch_size, n_head, head_dim = k_window_BxHxD.shape
    assert nw_k.shape == nw_v.shape, "nw.k/v shape mismatch"
    assert nw_k.size(0) == batch_size and nw_k.size(1) == n_head
    window_storage = nw_k.size(2)
    retained_storage = br_k.size(2)
    assert br_k.shape == br_v.shape
    assert nw_len.shape == (batch_size,)
    assert nw_pos.shape == (batch_size,)
    assert br_len.shape == (batch_size, n_head)

    block_d = _next_power_of_2(head_dim)
    grid = (batch_size, n_head)

    # Coerce active mask to int8 for predictable load semantics.
    if active_mask_B.dtype != torch.int8:
        active_mask_int8 = active_mask_B.to(torch.int8)
    else:
        active_mask_int8 = active_mask_B
    active_mask_int8 = active_mask_int8.contiguous()

    if window_tokens <= 0:
        # Window disabled — only update retained buffer.
        _fused_kv_update_no_window_kernel[grid](
            br_k,
            br_v,
            br_len,
            k_retained_BxHxD.contiguous(),
            v_retained_BxHxD.contiguous(),
            active_mask_int8,
            n_head,
            head_dim,
            retained_storage,
            block_d,
            num_warps=2,
        )
        return

    _fused_kv_update_kernel[grid](
        nw_k,
        nw_v,
        nw_len,
        nw_pos,
        br_k,
        br_v,
        br_len,
        k_window_BxHxD.contiguous(),
        v_window_BxHxD.contiguous(),
        k_retained_BxHxD.contiguous(),
        v_retained_BxHxD.contiguous(),
        active_mask_int8,
        n_head,
        head_dim,
        window_storage,
        retained_storage,
        int(window_tokens),
        block_d,
        num_warps=2,
    )
