"""Shared batched masked SDPA for autoregressive decode with variable-length KV segments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

try:
    import triton
    import triton.language as tl
except Exception:
    triton = None
    tl = None


@dataclass
class KVSegment:
    """A contiguous block of keys/values with per-element validity lengths.

    k, v: [B, H, max_T, D]
    lengths: [B, H] (per-head) or [B] (shared across heads, broadcast).
    max_len_cap: optional host-side cap for the compiled scan length.
    """
    k: Tensor
    v: Tensor
    lengths: Tensor
    max_len_cap: int | None = None


if triton is not None and tl is not None:

    @triton.jit
    def _segmented_q1_accumulate(
        q,
        k_base,
        v_base,
        len_ptr,
        acc,
        l_i,
        m_i,
        batch_idx,
        head_idx,
        scale,
        n_head: tl.constexpr,
        head_dim: tl.constexpr,
        storage_t: tl.constexpr,
        scan_t: tl.constexpr,
        block_n: tl.constexpr,
        block_d: tl.constexpr,
    ):
        bh = batch_idx * n_head + head_idx
        offs_d = tl.arange(0, block_d)
        d_mask = offs_d < head_dim
        seg_len = tl.load(len_ptr + bh)
        for start in tl.range(0, scan_t, block_n):
            offs_n = start + tl.arange(0, block_n)
            valid_n = offs_n < seg_len
            kv_offset = ((batch_idx * n_head + head_idx) * storage_t + offs_n[:, None]) * head_dim + offs_d[None, :]
            k = tl.load(k_base + kv_offset, mask=valid_n[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            scores = tl.sum(k * q[None, :], axis=1) * scale
            scores = tl.where(valid_n, scores, -3.4028234663852886e38)

            block_m = tl.max(scores, axis=0)
            new_m = tl.maximum(m_i, block_m)
            p = tl.exp(scores - new_m)
            alpha = tl.exp(m_i - new_m)
            v = tl.load(v_base + kv_offset, mask=valid_n[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            l_i = l_i * alpha + tl.sum(p, axis=0)
            m_i = new_m
        return acc, l_i, m_i


    @triton.jit
    def _segmented_q1_attention_kernel(
        q_ptr,
        k0_ptr,
        v0_ptr,
        len0_ptr,
        k1_ptr,
        v1_ptr,
        len1_ptr,
        k2_ptr,
        v2_ptr,
        len2_ptr,
        extra_k_ptr,
        extra_v_ptr,
        out_ptr,
        scale,
        n_head: tl.constexpr,
        head_dim: tl.constexpr,
        t0_storage: tl.constexpr,
        t0_scan: tl.constexpr,
        t1_storage: tl.constexpr,
        t1_scan: tl.constexpr,
        t2_storage: tl.constexpr,
        t2_scan: tl.constexpr,
        n_extra: tl.constexpr,
        block_n: tl.constexpr,
        block_d: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        bh = batch_idx * n_head + head_idx
        offs_d = tl.arange(0, block_d)
        d_mask = offs_d < head_dim
        q = tl.load(q_ptr + bh * head_dim + offs_d, mask=d_mask, other=0.0).to(tl.float32)

        m_i = tl.full((), -3.4028234663852886e38, tl.float32)
        l_i = tl.full((), 0.0, tl.float32)
        acc = tl.zeros((block_d,), dtype=tl.float32)

        acc, l_i, m_i = _segmented_q1_accumulate(
            q,
            k0_ptr,
            v0_ptr,
            len0_ptr,
            acc,
            l_i,
            m_i,
            batch_idx,
            head_idx,
            scale,
            n_head,
            head_dim,
            t0_storage,
            t0_scan,
            block_n,
            block_d,
        )
        acc, l_i, m_i = _segmented_q1_accumulate(
            q,
            k1_ptr,
            v1_ptr,
            len1_ptr,
            acc,
            l_i,
            m_i,
            batch_idx,
            head_idx,
            scale,
            n_head,
            head_dim,
            t1_storage,
            t1_scan,
            block_n,
            block_d,
        )
        acc, l_i, m_i = _segmented_q1_accumulate(
            q,
            k2_ptr,
            v2_ptr,
            len2_ptr,
            acc,
            l_i,
            m_i,
            batch_idx,
            head_idx,
            scale,
            n_head,
            head_dim,
            t2_storage,
            t2_scan,
            block_n,
            block_d,
        )

        for start in tl.static_range(0, 2):
            if start < n_extra:
                extra_offset = ((batch_idx * n_head + head_idx) * n_extra + start) * head_dim + offs_d
                k = tl.load(extra_k_ptr + extra_offset, mask=d_mask, other=0.0).to(tl.float32)
                score = tl.sum(k * q, axis=0) * scale
                new_m = tl.maximum(m_i, score)
                p = tl.exp(score - new_m)
                alpha = tl.exp(m_i - new_m)
                v = tl.load(extra_v_ptr + extra_offset, mask=d_mask, other=0.0).to(tl.float32)
                acc = acc * alpha + p * v
                l_i = l_i * alpha + p
                m_i = new_m

        out = acc / l_i
        out = tl.where(l_i > 0.0, out, 0.0)
        tl.store(out_ptr + bh * head_dim + offs_d, out, mask=d_mask)


    @triton.jit
    def _segmented_q2_accumulate(
        q0,
        q1,
        k_base,
        v_base,
        len_ptr,
        acc0,
        l0,
        m0,
        acc1,
        l1,
        m1,
        batch_idx,
        head_idx,
        scale,
        n_head: tl.constexpr,
        head_dim: tl.constexpr,
        storage_t: tl.constexpr,
        scan_t: tl.constexpr,
        block_n: tl.constexpr,
        block_d: tl.constexpr,
    ):
        bh = batch_idx * n_head + head_idx
        offs_d = tl.arange(0, block_d)
        d_mask = offs_d < head_dim
        seg_len = tl.load(len_ptr + bh)
        for start in tl.range(0, scan_t, block_n):
            offs_n = start + tl.arange(0, block_n)
            valid_n = offs_n < seg_len
            kv_offset = ((batch_idx * n_head + head_idx) * storage_t + offs_n[:, None]) * head_dim + offs_d[None, :]
            k = tl.load(k_base + kv_offset, mask=valid_n[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
            v = tl.load(v_base + kv_offset, mask=valid_n[:, None] & d_mask[None, :], other=0.0).to(tl.float32)

            scores0 = tl.sum(k * q0[None, :], axis=1) * scale
            scores0 = tl.where(valid_n, scores0, -3.4028234663852886e38)
            block_m0 = tl.max(scores0, axis=0)
            new_m0 = tl.maximum(m0, block_m0)
            p0 = tl.exp(scores0 - new_m0)
            alpha0 = tl.exp(m0 - new_m0)
            acc0 = acc0 * alpha0 + tl.sum(p0[:, None] * v, axis=0)
            l0 = l0 * alpha0 + tl.sum(p0, axis=0)
            m0 = new_m0

            scores1 = tl.sum(k * q1[None, :], axis=1) * scale
            scores1 = tl.where(valid_n, scores1, -3.4028234663852886e38)
            block_m1 = tl.max(scores1, axis=0)
            new_m1 = tl.maximum(m1, block_m1)
            p1 = tl.exp(scores1 - new_m1)
            alpha1 = tl.exp(m1 - new_m1)
            acc1 = acc1 * alpha1 + tl.sum(p1[:, None] * v, axis=0)
            l1 = l1 * alpha1 + tl.sum(p1, axis=0)
            m1 = new_m1
        return acc0, l0, m0, acc1, l1, m1


    @triton.jit
    def _segmented_q2_add_extra(
        q,
        k,
        v,
        acc,
        l_i,
        m_i,
        scale,
    ):
        score = tl.sum(k * q, axis=0) * scale
        new_m = tl.maximum(m_i, score)
        p = tl.exp(score - new_m)
        alpha = tl.exp(m_i - new_m)
        acc = acc * alpha + p * v
        l_i = l_i * alpha + p
        m_i = new_m
        return acc, l_i, m_i


    @triton.jit
    def _segmented_q2_attention_kernel(
        q_ptr,
        k0_ptr,
        v0_ptr,
        len0_ptr,
        k1_ptr,
        v1_ptr,
        len1_ptr,
        current_normal_k_ptr,
        current_normal_v_ptr,
        current_predict_k_ptr,
        current_predict_v_ptr,
        out_ptr,
        scale,
        n_head: tl.constexpr,
        head_dim: tl.constexpr,
        t0_storage: tl.constexpr,
        t0_scan: tl.constexpr,
        t1_storage: tl.constexpr,
        t1_scan: tl.constexpr,
        block_n: tl.constexpr,
        block_d: tl.constexpr,
    ):
        batch_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        offs_d = tl.arange(0, block_d)
        d_mask = offs_d < head_dim
        q0_offset = ((batch_idx * 2 + 0) * n_head + head_idx) * head_dim + offs_d
        q1_offset = ((batch_idx * 2 + 1) * n_head + head_idx) * head_dim + offs_d
        bh_offset = (batch_idx * n_head + head_idx) * head_dim + offs_d

        q0 = tl.load(q_ptr + q0_offset, mask=d_mask, other=0.0).to(tl.float32)
        q1 = tl.load(q_ptr + q1_offset, mask=d_mask, other=0.0).to(tl.float32)

        m0 = tl.full((), -3.4028234663852886e38, tl.float32)
        l0 = tl.full((), 0.0, tl.float32)
        acc0 = tl.zeros((block_d,), dtype=tl.float32)
        m1 = tl.full((), -3.4028234663852886e38, tl.float32)
        l1 = tl.full((), 0.0, tl.float32)
        acc1 = tl.zeros((block_d,), dtype=tl.float32)

        acc0, l0, m0, acc1, l1, m1 = _segmented_q2_accumulate(
            q0,
            q1,
            k0_ptr,
            v0_ptr,
            len0_ptr,
            acc0,
            l0,
            m0,
            acc1,
            l1,
            m1,
            batch_idx,
            head_idx,
            scale,
            n_head,
            head_dim,
            t0_storage,
            t0_scan,
            block_n,
            block_d,
        )
        acc0, l0, m0, acc1, l1, m1 = _segmented_q2_accumulate(
            q0,
            q1,
            k1_ptr,
            v1_ptr,
            len1_ptr,
            acc0,
            l0,
            m0,
            acc1,
            l1,
            m1,
            batch_idx,
            head_idx,
            scale,
            n_head,
            head_dim,
            t1_storage,
            t1_scan,
            block_n,
            block_d,
        )

        current_normal_k = tl.load(current_normal_k_ptr + bh_offset, mask=d_mask, other=0.0).to(tl.float32)
        current_normal_v = tl.load(current_normal_v_ptr + bh_offset, mask=d_mask, other=0.0).to(tl.float32)
        acc0, l0, m0 = _segmented_q2_add_extra(q0, current_normal_k, current_normal_v, acc0, l0, m0, scale)
        acc1, l1, m1 = _segmented_q2_add_extra(q1, current_normal_k, current_normal_v, acc1, l1, m1, scale)

        current_predict_k = tl.load(current_predict_k_ptr + bh_offset, mask=d_mask, other=0.0).to(tl.float32)
        current_predict_v = tl.load(current_predict_v_ptr + bh_offset, mask=d_mask, other=0.0).to(tl.float32)
        acc1, l1, m1 = _segmented_q2_add_extra(q1, current_predict_k, current_predict_v, acc1, l1, m1, scale)

        out0 = acc0 / l0
        out1 = acc1 / l1
        out0 = tl.where(l0 > 0.0, out0, 0.0)
        out1 = tl.where(l1 > 0.0, out1, 0.0)
        tl.store(out_ptr + q0_offset, out0, mask=d_mask)
        tl.store(out_ptr + q1_offset, out1, mask=d_mask)


def _next_power_of_2(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


def _expanded_lengths_BxH(lengths: Tensor, batch_size: int, n_head: int) -> Tensor:
    if lengths.dim() == 1:
        return lengths.view(batch_size, 1).expand(batch_size, n_head).contiguous()
    return lengths.contiguous()


def _segment_scan_len(seg: KVSegment) -> int:
    max_t = int(seg.k.size(2))
    if max_t == 0:
        return 0
    if seg.max_len_cap is None:
        return max_t
    cap = int(seg.max_len_cap)
    if cap < 0:
        raise ValueError(f"KVSegment.max_len_cap must be non-negative, got {cap}")
    return max(1, min(max_t, cap))


def triton_segmented_q1_attention(
    q_BxHxD: Tensor,
    segments: list[KVSegment],
    *,
    extra_kv: list[tuple[Tensor, Tensor]] | None = None,
    attn_dtype: torch.dtype,
) -> Optional[Tensor]:
    """Run q=1 segmented decode attention without building padded KV tensors.

    Returns None when the Triton fast path is unavailable or the inputs are not
    compatible, allowing callers to fall back to masked_kv_attention.
    """
    if triton is None or tl is None or not q_BxHxD.is_cuda:
        return None
    if len(segments) > 3:
        return None
    if extra_kv is not None and len(extra_kv) > 2:
        return None

    batch_size, n_head, head_dim = q_BxHxD.shape
    if head_dim <= 0 or head_dim > 256:
        return None
    if any((not seg.k.is_cuda) or (not seg.v.is_cuda) for seg in segments):
        return None
    if any((not seg.k.is_contiguous()) or (not seg.v.is_contiguous()) for seg in segments):
        return None

    padded_segments = list(segments)
    zero_lengths = torch.zeros((batch_size, n_head), device=q_BxHxD.device, dtype=torch.long)
    dummy_kv = q_BxHxD.new_empty((batch_size, n_head, 1, head_dim), dtype=attn_dtype)
    while len(padded_segments) < 3:
        padded_segments.append(KVSegment(dummy_kv, dummy_kv, zero_lengths))

    prepared_segments: list[tuple[Tensor, Tensor, Tensor, int, int]] = []
    for seg in padded_segments:
        if seg.k.dim() != 4 or seg.v.dim() != 4:
            return None
        if seg.k.shape != seg.v.shape:
            return None
        if seg.k.size(0) != batch_size or seg.k.size(1) != n_head or seg.k.size(3) != head_dim:
            return None
        lengths_BxH = _expanded_lengths_BxH(seg.lengths, batch_size, n_head)
        if lengths_BxH.device != q_BxHxD.device:
            return None
        prepared_segments.append((seg.k, seg.v, lengths_BxH, int(seg.k.size(2)), _segment_scan_len(seg)))

    n_extra = 0 if extra_kv is None else len(extra_kv)
    if n_extra == 0:
        extra_k_BxHxExD = q_BxHxD.new_empty((batch_size, n_head, 1, head_dim), dtype=attn_dtype)
        extra_v_BxHxExD = q_BxHxD.new_empty((batch_size, n_head, 1, head_dim), dtype=attn_dtype)
    else:
        extra_k_BxHxExD = torch.stack([kv[0] for kv in extra_kv], dim=2).to(attn_dtype).contiguous()
        extra_v_BxHxExD = torch.stack([kv[1] for kv in extra_kv], dim=2).to(attn_dtype).contiguous()

    q_BxHxD = q_BxHxD.to(attn_dtype).contiguous()
    out_BxHxD = torch.empty((batch_size, n_head, head_dim), device=q_BxHxD.device, dtype=torch.float32)
    block_d = _next_power_of_2(head_dim)
    block_n = 64
    grid = (batch_size, n_head)
    _segmented_q1_attention_kernel[grid](
        q_BxHxD,
        prepared_segments[0][0],
        prepared_segments[0][1],
        prepared_segments[0][2],
        prepared_segments[1][0],
        prepared_segments[1][1],
        prepared_segments[1][2],
        prepared_segments[2][0],
        prepared_segments[2][1],
        prepared_segments[2][2],
        extra_k_BxHxExD,
        extra_v_BxHxExD,
        out_BxHxD,
        1.0 / math.sqrt(head_dim),
        n_head,
        head_dim,
        prepared_segments[0][3],
        prepared_segments[0][4],
        prepared_segments[1][3],
        prepared_segments[1][4],
        prepared_segments[2][3],
        prepared_segments[2][4],
        n_extra,
        block_n,
        block_d,
        num_warps=4,
    )
    return out_BxHxD


def triton_segmented_q2_attention(
    q_Bx2xHxD: Tensor,
    segments: list[KVSegment],
    *,
    current_normal_k_BxHxD: Tensor,
    current_normal_v_BxHxD: Tensor,
    current_predict_k_BxHxD: Tensor,
    current_predict_v_BxHxD: Tensor,
    attn_dtype: torch.dtype,
) -> Tensor:
    """Required Triton q=2 segmented decode attention for SPS-family generation.

    Slot 0 attends over retained/window segments plus the current normal key.
    Slot 1 attends over the same segments plus current normal and current predict.
    This function is intentionally fail-fast: production generation should not
    silently fall back to padded SDPA when the fast path is unavailable.
    """
    if triton is None or tl is None:
        raise RuntimeError("Triton is required for SPS-family cached generation attention")
    if not q_Bx2xHxD.is_cuda:
        raise RuntimeError("CUDA tensors are required for SPS-family cached generation attention")
    if len(segments) > 2:
        raise RuntimeError(f"SPS-family q2 attention supports at most 2 segments, got {len(segments)}")

    batch_size, n_slots, n_head, head_dim = q_Bx2xHxD.shape
    if n_slots != 2:
        raise RuntimeError(f"Expected q_Bx2xHxD slot dimension to be 2, got {n_slots}")
    if head_dim <= 0 or head_dim > 256:
        raise RuntimeError(f"Unsupported SPS-family q2 attention head_dim={head_dim}")
    if not q_Bx2xHxD.is_contiguous():
        q_Bx2xHxD = q_Bx2xHxD.contiguous()

    current_tensors = [
        current_normal_k_BxHxD,
        current_normal_v_BxHxD,
        current_predict_k_BxHxD,
        current_predict_v_BxHxD,
    ]
    for tensor in current_tensors:
        if not tensor.is_cuda:
            raise RuntimeError("Current-token K/V tensors must be CUDA tensors")
        if tensor.shape != (batch_size, n_head, head_dim):
            raise RuntimeError(
                "Current-token K/V tensors must have shape "
                f"{(batch_size, n_head, head_dim)}, got {tuple(tensor.shape)}"
            )
    current_normal_k_BxHxD = current_normal_k_BxHxD.to(attn_dtype).contiguous()
    current_normal_v_BxHxD = current_normal_v_BxHxD.to(attn_dtype).contiguous()
    current_predict_k_BxHxD = current_predict_k_BxHxD.to(attn_dtype).contiguous()
    current_predict_v_BxHxD = current_predict_v_BxHxD.to(attn_dtype).contiguous()

    padded_segments = list(segments)
    zero_lengths = torch.zeros((batch_size, n_head), device=q_Bx2xHxD.device, dtype=torch.long)
    dummy_kv = q_Bx2xHxD.new_empty((batch_size, n_head, 1, head_dim), dtype=attn_dtype)
    while len(padded_segments) < 2:
        padded_segments.append(KVSegment(dummy_kv, dummy_kv, zero_lengths, max_len_cap=0))

    prepared_segments: list[tuple[Tensor, Tensor, Tensor, int, int]] = []
    for seg in padded_segments:
        if seg.k.dim() != 4 or seg.v.dim() != 4:
            raise RuntimeError("Segment K/V tensors must be rank-4 [B,H,T,D]")
        if seg.k.shape != seg.v.shape:
            raise RuntimeError(f"Segment K/V shapes differ: {tuple(seg.k.shape)} vs {tuple(seg.v.shape)}")
        if seg.k.size(0) != batch_size or seg.k.size(1) != n_head or seg.k.size(3) != head_dim:
            raise RuntimeError(
                "Segment K/V tensors must match query batch/head/head_dim, got "
                f"{tuple(seg.k.shape)} for query {(batch_size, n_head, head_dim)}"
            )
        if (not seg.k.is_cuda) or (not seg.v.is_cuda):
            raise RuntimeError("Segment K/V tensors must be CUDA tensors")
        if (not seg.k.is_contiguous()) or (not seg.v.is_contiguous()):
            raise RuntimeError("Segment K/V tensors must be contiguous")
        lengths_BxH = _expanded_lengths_BxH(seg.lengths, batch_size, n_head)
        if lengths_BxH.device != q_Bx2xHxD.device:
            raise RuntimeError("Segment lengths must live on the same device as query")
        scan_len = _segment_scan_len(seg)
        prepared_segments.append((seg.k, seg.v, lengths_BxH, int(seg.k.size(2)), scan_len))

    q_Bx2xHxD = q_Bx2xHxD.to(attn_dtype).contiguous()
    out_Bx2xHxD = torch.empty(
        (batch_size, 2, n_head, head_dim),
        device=q_Bx2xHxD.device,
        dtype=torch.float32,
    )
    block_d = _next_power_of_2(head_dim)
    block_n = 64
    grid = (batch_size, n_head)
    _segmented_q2_attention_kernel[grid](
        q_Bx2xHxD,
        prepared_segments[0][0],
        prepared_segments[0][1],
        prepared_segments[0][2],
        prepared_segments[1][0],
        prepared_segments[1][1],
        prepared_segments[1][2],
        current_normal_k_BxHxD,
        current_normal_v_BxHxD,
        current_predict_k_BxHxD,
        current_predict_v_BxHxD,
        out_Bx2xHxD,
        1.0 / math.sqrt(head_dim),
        n_head,
        head_dim,
        prepared_segments[0][3],
        prepared_segments[0][4],
        prepared_segments[1][3],
        prepared_segments[1][4],
        block_n,
        block_d,
        num_warps=4,
    )
    return out_Bx2xHxD


def masked_kv_attention(
    q_BxHxD: Tensor,
    segments: list[KVSegment],
    *,
    extra_kv: list[tuple[Tensor, Tensor]] | None = None,
    attn_dtype: torch.dtype,
    check_finite: bool = False,
    error_context: str = "",
    prefer_triton: bool = False,
    require_triton: bool = False,
) -> Tensor:
    """Run scaled dot-product attention over variable-length KV segments.

    Assembles retained/window/current KV segments into a single padded tensor
    with a validity mask, then runs F.scaled_dot_product_attention.

    Args:
        q_BxHxD: query tensor [B, H, D].
        segments: list of KVSegment, each contributing a block of keys/values.
        extra_kv: optional list of (k_BxHxD, v_BxHxD) single-token entries
                  that are always valid (e.g. current token being attended to).
        attn_dtype: dtype for the attention computation.
        check_finite: if True, raise on invalid masks or non-finite values.
        error_context: string appended to error messages for debugging.

    Returns:
        out_BxHxD: attention output [B, H, D] in float32.
    """
    if (prefer_triton or require_triton) and not check_finite:
        triton_out = triton_segmented_q1_attention(
            q_BxHxD,
            segments,
            extra_kv=extra_kv,
            attn_dtype=attn_dtype,
        )
        if triton_out is not None:
            return triton_out
        if require_triton:
            raise RuntimeError(f"Triton segmented q1 attention is required but unavailable{error_context}")

    b, n_head, head_dim = q_BxHxD.shape
    device = q_BxHxD.device
    n_extra = len(extra_kv) if extra_kv else 0

    seg_maxlens = []
    for seg in segments:
        if seg.lengths.numel() == 0:
            seg_maxlens.append(0)
        else:
            seg_maxlens.append(min(int(seg.lengths.max().item()), _segment_scan_len(seg)))
    total_len = sum(seg_maxlens) + n_extra

    if total_len == 0:
        return q_BxHxD.new_zeros((b, n_head, head_dim), dtype=torch.float32)

    keys = torch.zeros((b, n_head, total_len, head_dim), device=device, dtype=attn_dtype)
    values = torch.zeros_like(keys)
    mask = torch.zeros((b, n_head, total_len), device=device, dtype=torch.bool)

    col = 0
    for seg, max_len in zip(segments, seg_maxlens):
        if max_len == 0:
            continue
        keys[:, :, col:col + max_len, :] = seg.k[:, :, :max_len, :]
        values[:, :, col:col + max_len, :] = seg.v[:, :, :max_len, :]
        pos = torch.arange(max_len, device=device)
        if seg.lengths.dim() == 1:
            # [B] lengths — shared across heads, broadcast
            valid = pos.view(1, max_len) < seg.lengths.unsqueeze(-1)  # [B, max_len]
            mask[:, :, col:col + max_len] = valid.unsqueeze(1)
        else:
            # [B, H] lengths — per-head
            valid = pos.view(1, 1, max_len) < seg.lengths.unsqueeze(-1)  # [B, H, max_len]
            mask[:, :, col:col + max_len] = valid
        col += max_len

    if extra_kv:
        for extra_k, extra_v in extra_kv:
            keys[:, :, col, :] = extra_k
            values[:, :, col, :] = extra_v
            mask[:, :, col] = True
            col += 1

    bh = b * n_head
    q_flat = q_BxHxD.reshape(bh, 1, head_dim)
    k_flat = keys.reshape(bh, total_len, head_dim)
    v_flat = values.reshape(bh, total_len, head_dim)
    mask_flat = mask.reshape(bh, 1, total_len)

    if check_finite:
        if (mask_flat.sum(dim=-1) <= 0).any():
            raise RuntimeError(f"Invalid mask with no valid keys{error_context}")
        if not (torch.isfinite(q_flat).all() and torch.isfinite(k_flat).all() and torch.isfinite(v_flat).all()):
            raise RuntimeError(f"Non-finite QKV{error_context}")

    out = F.scaled_dot_product_attention(
        q_flat, k_flat, v_flat,
        attn_mask=mask_flat,
        dropout_p=0.0,
        is_causal=False,
        scale=1.0 / math.sqrt(head_dim),
    )
    out_BxHxD = out.reshape(b, n_head, head_dim).float()

    if check_finite and not torch.isfinite(out_BxHxD).all():
        raise RuntimeError(f"Non-finite attention output{error_context}")

    return out_BxHxD
