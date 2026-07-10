from __future__ import annotations

import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from modeling.models.attention.triton_reverse_sps_flash_attention import (
    ATTN_FWD_CONFIGS,
    TensorDescriptor,
    _REVERSE_SPS_BWD_DKDV_CONFIGS,
    _REVERSE_SPS_BWD_DQ_CONFIGS,
    _attn_bwd_preprocess,
    _host_descriptor_pre_hook,
    _maybe_make_tensor_desc,
    is_blackwell,
    is_hip,
    is_hopper,
    keep,
    prune_invalid_configs,
    supports_host_descriptor,
)


@triton.jit
def _attn_fwd_inner_sps(
    acc,
    l_i,
    m_i,
    q,
    desc_k,
    desc_v,
    offset_y,
    dtype: tl.constexpr,
    start_m,
    qk_scale,
    off_z,
    off_h,
    document_ptr,
    stride_doz,
    stride_dot,
    HAS_DOCUMENT_MASK: tl.constexpr,
    temporary_key_window,
    persistent_key_window,
    HAS_PREDICT_BIAS: tl.constexpr,
    HAS_PERSISTENT_KEY_WINDOW: tl.constexpr,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    N_CTX: tl.constexpr,
    warp_specialize: tl.constexpr,
    IS_HOPPER: tl.constexpr,
):
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX
    offsetk_y = offset_y + lo
    if dtype == tl.float8e5:
        offsetv_y = offset_y * HEAD_DIM + lo
    else:
        offsetv_y = offset_y + lo
    if HAS_PREDICT_BIAS:
        # Layout (both variants): even parity = input token, odd parity = <predict>.
        # SPS variant: the persistent (window-exempt) slot is the EVEN/input token;
        # the <predict> token (odd) is the windowed, read-out slot.
        q_tok = offs_m // 2
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = desc_k.load([offsetk_y, 0]).T
        qk = tl.dot(q, k)
        qk = qk * qk_scale
        offs_n_abs = start_n + offs_n
        if HAS_DOCUMENT_MASK:
            docs_q_ptrs = document_ptr + off_z * stride_doz + offs_m * stride_dot
            docs_k_ptrs = document_ptr + off_z * stride_doz + offs_n_abs * stride_dot
            docs_q = tl.load(docs_q_ptrs)
            docs_k = tl.load(docs_k_ptrs)
            same_doc = docs_q[:, None] == docs_k[None, :]
            qk = qk + tl.where(same_doc, 0.0, -1.0e6)
        if HAS_PREDICT_BIAS:
            k_tok = offs_n_abs // 2

            # Hard sliding window for the temporary (non-persistent) keys, the
            # <predict> slot (odd parity) in the SPS variant.
            rel = q_tok[:, None] - k_tok[None, :]
            out_of_window = rel > temporary_key_window
            normal_bias = tl.where(out_of_window, -1.0e6, 0.0)

            k_is_persistent = (offs_n_abs % 2) == 0
            # Persistent keys — the input slot (even parity) in the SPS variant —
            # carry no additive bias; visible subject only to the optional
            # persistent-key window.
            persistent_bias = 0.0
            if HAS_PERSISTENT_KEY_WINDOW:
                persistent_rel = q_tok[:, None] - k_tok[None, :]
                persistent_bias = tl.where(persistent_rel <= persistent_key_window, 0.0, -1.0e6)
            attn_bias = tl.where(k_is_persistent[None, :], persistent_bias, normal_bias)
            is_self_persistent = (offs_m[:, None] == offs_n_abs[None, :]) & k_is_persistent[None, :]
            attn_bias = tl.where(is_self_persistent, 0.0, attn_bias)
            qk += attn_bias * 1.44269504
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        if not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
        else:
            acc = acc * alpha[:, None]
        if dtype == tl.float8e5:
            v = desc_v.load([0, offsetv_y]).T
        else:
            v = desc_v.load([offsetv_y, 0])
        p = p.to(dtype)
        acc = tl.dot(p, v, acc)
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    return acc, l_i, m_i


@triton.autotune(
    configs=list(filter(keep, ATTN_FWD_CONFIGS)),
    key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
    prune_configs_by={"early_config_prune": prune_invalid_configs},
)
@triton.jit
def _attn_fwd_sps(
    sm_scale,
    M,
    Z,
    H,
    desc_q,
    desc_k,
    desc_v,
    desc_o,
    N_CTX,
    document_ptr,
    stride_doz,
    stride_dot,
    temporary_key_window,
    persistent_key_window,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    FP8_OUTPUT: tl.constexpr,
    HAS_DOCUMENT_MASK: tl.constexpr,
    HAS_PREDICT_BIAS: tl.constexpr,
    HAS_PERSISTENT_KEY_WINDOW: tl.constexpr,
    STAGE: tl.constexpr,
    warp_specialize: tl.constexpr,
    IS_HOPPER: tl.constexpr,
    DTYPE_IS_BF16: tl.constexpr,
):
    dtype = tl.float8e5 if FP8_OUTPUT else (tl.bfloat16 if DTYPE_IS_BF16 else tl.float16)
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM])
    if FP8_OUTPUT:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[HEAD_DIM, y_dim], strides=[N_CTX, 1], block_shape=[HEAD_DIM, BLOCK_N])
    else:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(desc_o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1], block_shape=[BLOCK_M, HEAD_DIM])

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e9
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale
    qk_scale *= 1.44269504
    q = desc_q.load([qo_offset_y, 0])
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner_sps(
            acc,
            l_i,
            m_i,
            q,
            desc_k,
            desc_v,
            offset_y,
            dtype,
            start_m,
            qk_scale,
            off_z,
            off_h,
            document_ptr,
            stride_doz,
            stride_dot,
            HAS_DOCUMENT_MASK,
            temporary_key_window,
            persistent_key_window,
            HAS_PREDICT_BIAS,
            HAS_PERSISTENT_KEY_WINDOW,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            4 - STAGE,
            offs_m,
            offs_n,
            N_CTX,
            warp_specialize,
            IS_HOPPER,
        )
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner_sps(
            acc,
            l_i,
            m_i,
            q,
            desc_k,
            desc_v,
            offset_y,
            dtype,
            start_m,
            qk_scale,
            off_z,
            off_h,
            document_ptr,
            stride_doz,
            stride_dot,
            HAS_DOCUMENT_MASK,
            temporary_key_window,
            persistent_key_window,
            HAS_PREDICT_BIAS,
            HAS_PERSISTENT_KEY_WINDOW,
            BLOCK_M,
            HEAD_DIM,
            BLOCK_N,
            2,
            offs_m,
            offs_n,
            N_CTX,
            warp_specialize,
            IS_HOPPER,
        )
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(dtype))


@triton.autotune(
    configs=_REVERSE_SPS_BWD_DKDV_CONFIGS,
    key=[
        "N_CTX",
        "HEAD_DIM",
        "BLOCK_K",
        "HAS_DOCUMENT_MASK",
        "HAS_PERSISTENT_KEY_WINDOW",
    ],
)
@triton.jit
def _sps_bwd_dkdv_kernel(
    Q,
    K,
    V,
    sm_scale,
    DO,
    DK,
    DV,
    M,
    D,
    DOCUMENT_IDX,
    stride_z,
    stride_h,
    stride_tok,
    stride_d,
    stride_doc_z,
    stride_doc_t,
    H,
    N_CTX,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TEMPORARY_KEY_WINDOW: tl.constexpr,
    PERSISTENT_KEY_WINDOW: tl.constexpr,
    HAS_DOCUMENT_MASK: tl.constexpr,
    HAS_PERSISTENT_KEY_WINDOW: tl.constexpr,
):
    pid_k = tl.program_id(0)
    bhid = tl.program_id(1)

    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    off_chz = (bhid * N_CTX).to(tl.int64)

    Q += adj
    K += adj
    V += adj
    DO += adj
    DK += adj
    DV += adj
    M += off_chz
    D += off_chz
    DOCUMENT_IDX += (stride_doc_z * (bhid // H)).to(tl.int64)

    offs_d = tl.arange(0, HEAD_DIM)
    offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)

    k = tl.load(K + offs_k[:, None] * stride_tok + offs_d[None, :] * stride_d)
    v = tl.load(V + offs_k[:, None] * stride_tok + offs_d[None, :] * stride_d)

    k_tok = offs_k // 2
    k_is_persistent = (offs_k % 2) == 0
    docs_k = tl.load(DOCUMENT_IDX + offs_k * stride_doc_t) if HAS_DOCUMENT_MASK else tl.zeros([BLOCK_K], dtype=tl.int32)

    dk = tl.zeros([BLOCK_K, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_K, HEAD_DIM], dtype=tl.float32)

    q_start = pid_k * BLOCK_K
    num_q_blocks = (N_CTX - q_start) // BLOCK_Q
    for q_blk_idx in range(num_q_blocks):
        offs_q = q_start + q_blk_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
        q = tl.load(Q + offs_q[:, None] * stride_tok + offs_d[None, :] * stride_d)
        do = tl.load(DO + offs_q[:, None] * stride_tok + offs_d[None, :] * stride_d)
        m_q = tl.load(M + offs_q)
        Di = tl.load(D + offs_q)

        qk = tl.dot(q, tl.trans(k))
        q_tok = offs_q // 2
        docs_q = tl.load(DOCUMENT_IDX + offs_q * stride_doc_t) if HAS_DOCUMENT_MASK else tl.zeros([BLOCK_Q], dtype=tl.int32)
        rel = q_tok[:, None] - k_tok[None, :]
        out_of_window = rel > TEMPORARY_KEY_WINDOW
        normal_bias = tl.where(out_of_window, -1.0e6, 0.0)
        persistent_bias = 0.0
        if HAS_PERSISTENT_KEY_WINDOW:
            persistent_rel = q_tok[:, None] - k_tok[None, :]
            persistent_bias = tl.where(persistent_rel <= PERSISTENT_KEY_WINDOW, 0.0, -1.0e6)
        bias = tl.where(k_is_persistent[None, :], persistent_bias, normal_bias)
        is_self_persistent = (offs_q[:, None] == offs_k[None, :]) & k_is_persistent[None, :]
        bias = tl.where(is_self_persistent, 0.0, bias)
        qk = qk * (sm_scale * 1.44269504) + bias * 1.44269504

        p = tl.math.exp2(qk - m_q[:, None])
        causal_mask = offs_q[:, None] >= offs_k[None, :]
        if HAS_DOCUMENT_MASK:
            doc_mask = docs_q[:, None] == docs_k[None, :]
            p = tl.where(causal_mask & doc_mask, p, 0.0)
        else:
            p = tl.where(causal_mask, p, 0.0)

        p16 = p.to(do.dtype)
        dv += tl.dot(tl.trans(p16), do)

        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds16 = ds.to(q.dtype)
        dk += tl.dot(tl.trans(ds16), q)

    dk_ptrs = DK + offs_k[:, None] * stride_tok + offs_d[None, :] * stride_d
    tl.store(dk_ptrs, dk * sm_scale)
    dv_ptrs = DV + offs_k[:, None] * stride_tok + offs_d[None, :] * stride_d
    tl.store(dv_ptrs, dv)


@triton.autotune(
    configs=_REVERSE_SPS_BWD_DQ_CONFIGS,
    key=[
        "N_CTX",
        "HEAD_DIM",
        "BLOCK_K",
        "HAS_DOCUMENT_MASK",
        "HAS_PERSISTENT_KEY_WINDOW",
    ],
)
@triton.jit
def _sps_bwd_dq_kernel(
    Q,
    K,
    V,
    sm_scale,
    DO,
    DQ,
    M,
    D,
    DOCUMENT_IDX,
    stride_z,
    stride_h,
    stride_tok,
    stride_d,
    stride_doc_z,
    stride_doc_t,
    H,
    N_CTX,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TEMPORARY_KEY_WINDOW: tl.constexpr,
    PERSISTENT_KEY_WINDOW: tl.constexpr,
    HAS_DOCUMENT_MASK: tl.constexpr,
    HAS_PERSISTENT_KEY_WINDOW: tl.constexpr,
):
    pid_q = tl.program_id(0)
    bhid = tl.program_id(1)

    adj = (stride_h * (bhid % H) + stride_z * (bhid // H)).to(tl.int64)
    off_chz = (bhid * N_CTX).to(tl.int64)

    Q += adj
    K += adj
    V += adj
    DO += adj
    DQ += adj
    M += off_chz
    D += off_chz
    DOCUMENT_IDX += (stride_doc_z * (bhid // H)).to(tl.int64)

    offs_d = tl.arange(0, HEAD_DIM)
    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)

    q = tl.load(Q + offs_q[:, None] * stride_tok + offs_d[None, :] * stride_d)
    do = tl.load(DO + offs_q[:, None] * stride_tok + offs_d[None, :] * stride_d)
    m_q = tl.load(M + offs_q)
    Di = tl.load(D + offs_q)

    q_tok = offs_q // 2
    docs_q = tl.load(DOCUMENT_IDX + offs_q * stride_doc_t) if HAS_DOCUMENT_MASK else tl.zeros([BLOCK_Q], dtype=tl.int32)

    dq = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

    num_k_blocks = (pid_q * BLOCK_Q + BLOCK_Q) // BLOCK_K
    for k_blk_idx in range(num_k_blocks):
        offs_k = k_blk_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        kT = tl.load(K + offs_k[None, :] * stride_tok + offs_d[:, None] * stride_d)
        vT = tl.load(V + offs_k[None, :] * stride_tok + offs_d[:, None] * stride_d)

        k_tok = offs_k // 2
        k_is_persistent = (offs_k % 2) == 0
        docs_k = tl.load(DOCUMENT_IDX + offs_k * stride_doc_t) if HAS_DOCUMENT_MASK else tl.zeros([BLOCK_K], dtype=tl.int32)

        qk = tl.dot(q, kT)
        rel = q_tok[:, None] - k_tok[None, :]
        out_of_window = rel > TEMPORARY_KEY_WINDOW
        normal_bias = tl.where(out_of_window, -1.0e6, 0.0)
        persistent_bias = 0.0
        if HAS_PERSISTENT_KEY_WINDOW:
            persistent_rel = q_tok[:, None] - k_tok[None, :]
            persistent_bias = tl.where(persistent_rel <= PERSISTENT_KEY_WINDOW, 0.0, -1.0e6)
        bias = tl.where(k_is_persistent[None, :], persistent_bias, normal_bias)
        is_self_persistent = (offs_q[:, None] == offs_k[None, :]) & k_is_persistent[None, :]
        bias = tl.where(is_self_persistent, 0.0, bias)
        qk = qk * (sm_scale * 1.44269504) + bias * 1.44269504

        p = tl.math.exp2(qk - m_q[:, None])
        causal_mask = offs_q[:, None] >= offs_k[None, :]
        if HAS_DOCUMENT_MASK:
            doc_mask = docs_q[:, None] == docs_k[None, :]
            p = tl.where(causal_mask & doc_mask, p, 0.0)
        else:
            p = tl.where(causal_mask, p, 0.0)

        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds16 = ds.to(kT.dtype)
        dq += tl.dot(ds16, tl.trans(kT))

    dq_ptrs = DQ + offs_q[:, None] * stride_tok + offs_d[None, :] * stride_d
    dq *= sm_scale
    tl.store(dq_ptrs, dq)


class _sps_sliding_attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        sm_scale,
        temporary_key_window,
        warp_specialize=True,
        documents_idx_BxT=None,
        persistent_key_window=None,
    ):
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        head_dim_q, head_dim_k = q.shape[-1], k.shape[-1]
        head_dim_v = v.shape[-1]
        assert head_dim_q == head_dim_k and head_dim_k == head_dim_v
        assert head_dim_k in {16, 32, 64, 128, 256}
        assert q.shape[2] % 2 == 0, "Expected doubled sequence length (2T)."
        has_document_mask = documents_idx_BxT is not None
        has_persistent_key_window = persistent_key_window is not None
        persistent_key_window_value = -1 if persistent_key_window is None else int(persistent_key_window)
        if has_document_mask:
            assert documents_idx_BxT.shape[0] == q.shape[0]
            assert documents_idx_BxT.shape[1] == q.shape[2]
            assert documents_idx_BxT.is_cuda and documents_idx_BxT.is_contiguous()

        # Pad the sequence up to a multiple of BLOCK_M_MAX. This is a correctness
        # requirement, not a perf tweak: through the flat [B*H*N_CTX, D] descriptor
        # view, an unaligned final block spills into the next head's storage. See
        # "Sequence padding" in this directory's README.
        BLOCK_M_MAX = 128
        N_CTX_ORIG = q.shape[2]
        pad_amount = (-N_CTX_ORIG) % BLOCK_M_MAX  # 0 if already aligned
        if pad_amount > 0:
            q = F.pad(q, (0, 0, 0, pad_amount))  # pad seq dim (dim=2)
            k = F.pad(k, (0, 0, 0, pad_amount))
            v = F.pad(v, (0, 0, 0, pad_amount))
            if has_document_mask:
                documents_idx_BxT = F.pad(documents_idx_BxT, (0, pad_amount), value=-1)

        o = torch.empty_like(q)
        stage = 3
        extra_kern_args = {}
        if is_hip():
            waves_per_eu = 3 if head_dim_k <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        if supports_host_descriptor() and not (is_hopper() and warp_specialize):
            y_dim = q.shape[0] * q.shape[1] * q.shape[2]
            dummy_block = [1, 1]
            desc_q = TensorDescriptor(q, shape=[y_dim, head_dim_k], strides=[head_dim_k, 1], block_shape=dummy_block)
            if q.dtype == torch.float8_e5m2:
                desc_v = TensorDescriptor(v, shape=[head_dim_k, y_dim], strides=[q.shape[2], 1], block_shape=dummy_block)
            else:
                desc_v = TensorDescriptor(v, shape=[y_dim, head_dim_k], strides=[head_dim_k, 1], block_shape=dummy_block)
            desc_k = TensorDescriptor(k, shape=[y_dim, head_dim_k], strides=[head_dim_k, 1], block_shape=dummy_block)
            desc_o = TensorDescriptor(o, shape=[y_dim, head_dim_k], strides=[head_dim_k, 1], block_shape=dummy_block)
        else:
            desc_q = q
            desc_v = v
            desc_k = k
            desc_o = o

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        def grid(meta):
            return (triton.cdiv(q.shape[2], meta["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

        if is_blackwell() and warp_specialize:
            if head_dim_k == 128 and q.dtype == torch.float16:
                extra_kern_args["maxnreg"] = 168
            else:
                extra_kern_args["maxnreg"] = 80

        if has_document_mask:
            document_ptr = documents_idx_BxT
            stride_doz, stride_dot = documents_idx_BxT.stride()
        else:
            document_ptr = q
            stride_doz = stride_dot = 0

        _attn_fwd_sps[grid](
            sm_scale,
            M,
            q.shape[0],
            q.shape[1],
            desc_q,
            desc_k,
            desc_v,
            desc_o,
            N_CTX=q.shape[2],
            document_ptr=document_ptr,
            stride_doz=stride_doz,
            stride_dot=stride_dot,
            temporary_key_window=int(temporary_key_window),
            persistent_key_window=persistent_key_window_value,
            HEAD_DIM=head_dim_k,
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,
            HAS_DOCUMENT_MASK=has_document_mask,
            HAS_PREDICT_BIAS=True,
            HAS_PERSISTENT_KEY_WINDOW=has_persistent_key_window,
            STAGE=stage,
            warp_specialize=warp_specialize,
            IS_HOPPER=is_hopper(),
            DTYPE_IS_BF16=q.dtype == torch.bfloat16,
            **extra_kern_args,
        )

        o_padded = o
        o_out = o_padded[:, :, :N_CTX_ORIG, :] if pad_amount > 0 else o_padded

        if has_document_mask:
            ctx.save_for_backward(q, k, v, o_padded, M, documents_idx_BxT)
        else:
            ctx.save_for_backward(q, k, v, o_padded, M)
        ctx.sm_scale = sm_scale
        ctx.temporary_key_window = int(temporary_key_window)
        ctx.persistent_key_window = persistent_key_window_value
        ctx.has_persistent_key_window = has_persistent_key_window
        ctx.has_document_mask = has_document_mask
        ctx.n_ctx_orig = int(N_CTX_ORIG)
        ctx.pad_amount = int(pad_amount)
        return o_out

    @staticmethod
    def backward(ctx, do):
        if ctx.has_document_mask:
            q, k, v, o, M, documents_idx_BxT = ctx.saved_tensors
        else:
            q, k, v, o, M = ctx.saved_tensors
            documents_idx_BxT = q
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        o = o.contiguous()
        do = do.contiguous()
        if ctx.pad_amount > 0:
            do = F.pad(do, (0, 0, 0, ctx.pad_amount))
        if ctx.has_document_mask:
            documents_idx_BxT = documents_idx_BxT.contiguous()

        batch, n_head, n_ctx, head_dim = q.shape

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        pre_block = 128
        block_k = 64

        assert n_ctx % pre_block == 0
        pre_grid = (n_ctx // pre_block, batch * n_head)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](o, do, delta, batch, n_head, n_ctx, BLOCK_M=pre_block, HEAD_DIM=head_dim)

        num_k_blocks = n_ctx // block_k
        grid_dkdv = (num_k_blocks, batch * n_head)
        _sps_bwd_dkdv_kernel[grid_dkdv](
            q,
            k,
            v,
            ctx.sm_scale,
            do,
            dk,
            dv,
            M,
            delta,
            documents_idx_BxT,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            documents_idx_BxT.stride(0),
            documents_idx_BxT.stride(1) if ctx.has_document_mask else 0,
            n_head,
            n_ctx,
            BLOCK_K=block_k,
            HEAD_DIM=head_dim,
            TEMPORARY_KEY_WINDOW=ctx.temporary_key_window,
            PERSISTENT_KEY_WINDOW=ctx.persistent_key_window,
            HAS_DOCUMENT_MASK=ctx.has_document_mask,
            HAS_PERSISTENT_KEY_WINDOW=ctx.has_persistent_key_window,
        )

        grid_dq = lambda meta: (triton.cdiv(n_ctx, meta["BLOCK_Q"]), batch * n_head)
        _sps_bwd_dq_kernel[grid_dq](
            q,
            k,
            v,
            ctx.sm_scale,
            do,
            dq,
            M,
            delta,
            documents_idx_BxT,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            documents_idx_BxT.stride(0),
            documents_idx_BxT.stride(1) if ctx.has_document_mask else 0,
            n_head,
            n_ctx,
            BLOCK_K=block_k,
            HEAD_DIM=head_dim,
            TEMPORARY_KEY_WINDOW=ctx.temporary_key_window,
            PERSISTENT_KEY_WINDOW=ctx.persistent_key_window,
            HAS_DOCUMENT_MASK=ctx.has_document_mask,
            HAS_PERSISTENT_KEY_WINDOW=ctx.has_persistent_key_window,
        )

        dq_out = dq[:, :, : ctx.n_ctx_orig, :].to(q.dtype)
        dk_out = dk[:, :, : ctx.n_ctx_orig, :].to(k.dtype)
        dv_out = dv[:, :, : ctx.n_ctx_orig, :].to(v.dtype)

        return (
            dq_out,
            dk_out,
            dv_out,
            None,  # sm_scale
            None,  # temporary_key_window
            None,  # warp_specialize
            None,  # documents_idx_BxT
            None,  # persistent_key_window
        )


def sps_sliding_attention(
    q,
    k,
    v,
    sm_scale,
    temporary_key_window,
    warp_specialize=True,
    documents_idx_BxT=None,
    persistent_key_window=None,
):
    """
    Triton attention over the doubled input/predict (2T) sequence with a hard
    sliding window of temporary_key_window tokens over the temporary keys. SPS
    variant: the persistent (window-exempt) key sits at EVEN parity, which in the
    SPS model is the input token, while the <predict> token sits at odd parity and
    is the windowed, read-out slot. The persistent key always self-attends and is
    visible subject only to the optional persistent-key window. Supports optional
    per-document masking.
    """
    return _sps_sliding_attention.apply(
        q,
        k,
        v,
        sm_scale,
        temporary_key_window,
        warp_specialize,
        documents_idx_BxT,
        persistent_key_window,
    )
