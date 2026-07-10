"""
Triton Flash Attention (document-masked causal).

Plain Flash Attention v2 (Tri Dao) with optional per-document masking: a query
attends causally only to keys sharing its document id. This is the kernel used by
the Standard (full-attention) model.

Based on the Flash Attention v2 algorithm from Tri Dao.
"""

try:
    import pytest
except Exception:
    class _PytestStub:
        @staticmethod
        def parametrize(*args, **kwargs):
            def _decorator(fn):
                return fn
            return _decorator
    pytest = _PytestStub()
import torch
import torch.nn.functional as F
import os

import triton
import triton.language as tl
try:
    from triton.tools.tensor_descriptor import TensorDescriptor
except Exception:
    TensorDescriptor = None

try:
    DEVICE = triton.runtime.driver.active.get_active_torch_device()
except Exception:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def is_hip():
    try:
        return triton.runtime.driver.active.get_current_target().backend == "hip"
    except Exception:
        return False


def is_cuda():
    try:
        return triton.runtime.driver.active.get_current_target().backend == "cuda"
    except Exception:
        return False


def supports_host_descriptor():
    return TensorDescriptor is not None and is_cuda() and torch.cuda.get_device_capability()[0] >= 9


def is_blackwell():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 10


def is_hopper():
    return is_cuda() and torch.cuda.get_device_capability()[0] == 9


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    desc_k, desc_v,  #
                    offset_y, dtype: tl.constexpr, start_m, qk_scale,  #
                    off_z, off_h,  #
                    document_ptr, stride_doz, stride_dot, HAS_DOCUMENT_MASK: tl.constexpr,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, warp_specialize: tl.constexpr, IS_HOPPER: tl.constexpr):
    # range of values handled by this stage
    if STAGE == 1:
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    # causal = False
    else:
        lo, hi = 0, N_CTX
    offsetk_y = offset_y + lo
    if dtype == tl.float8e5:
        offsetv_y = offset_y * HEAD_DIM + lo
    else:
        offsetv_y = offset_y + lo
    # loop over k, v and update accumulator
    for start_n in tl.range(lo, hi, BLOCK_N, warp_specialize=warp_specialize):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # -- compute qk ----
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
        if STAGE == 2:
            mask = offs_m[:, None] >= (start_n + offs_n[None, :])
            qk = qk + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk -= m_ij[:, None]
        else:
            m_ij = tl.maximum(m_i, tl.max(qk, 1))
            qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        # -- compute correction factor
        alpha = tl.math.exp2(m_i - m_ij)
        l_ij = tl.sum(p, 1)
        # -- update output accumulator --
        if not IS_HOPPER and warp_specialize and BLOCK_M == 128 and HEAD_DIM == 128:
            BM: tl.constexpr = acc.shape[0]
            BN: tl.constexpr = acc.shape[1]
            acc0, acc1 = acc.reshape([BM, 2, BN // 2]).permute(0, 2, 1).split()
            acc0 = acc0 * alpha[:, None]
            acc1 = acc1 * alpha[:, None]
            acc = tl.join(acc0, acc1).permute(0, 2, 1).reshape([BM, BN])
        else:
            acc = acc * alpha[:, None]
        # prepare p and v for the dot
        if dtype == tl.float8e5:
            v = desc_v.load([0, offsetv_y]).T
        else:
            v = desc_v.load([offsetv_y, 0])
        p = p.to(dtype)
        acc = tl.dot(p, v, acc)
        # update m_i and l_i
        l_i = l_i * alpha + l_ij
        m_i = m_ij
        offsetk_y += BLOCK_N
        offsetv_y += BLOCK_N
    return acc, l_i, m_i


def _host_descriptor_pre_hook(nargs):
    BLOCK_M = nargs["BLOCK_M"]
    BLOCK_N = nargs["BLOCK_N"]
    HEAD_DIM = nargs["HEAD_DIM"]
    if TensorDescriptor is None or not isinstance(nargs["desc_q"], TensorDescriptor):
        return
    nargs["desc_q"].block_shape = [BLOCK_M, HEAD_DIM]
    if nargs["FP8_OUTPUT"]:
        nargs["desc_v"].block_shape = [HEAD_DIM, BLOCK_N]
    else:
        nargs["desc_v"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_k"].block_shape = [BLOCK_N, HEAD_DIM]
    nargs["desc_o"].block_shape = [BLOCK_M, HEAD_DIM]


if is_hip():
    NUM_STAGES_OPTIONS = [1]
elif supports_host_descriptor():
    NUM_STAGES_OPTIONS = [2, 3, 4]
else:
    NUM_STAGES_OPTIONS = [2, 3, 4]

configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w, pre_hook=_host_descriptor_pre_hook) \
    for BM in [64, 128]\
    for BN in [32, 64, 128]\
    for s in NUM_STAGES_OPTIONS \
    for w in [4, 8]\
]
if "PYTEST_VERSION" in os.environ:
    configs = [
        triton.Config(dict(BLOCK_M=128, BLOCK_N=64), num_stages=2, num_warps=4, pre_hook=_host_descriptor_pre_hook),
    ]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    return not (is_cuda() and torch.cuda.get_device_capability()[0] == 9 and BLOCK_M * BLOCK_N < 128 * 128
                and conf.num_warps == 8)


def prune_invalid_configs(configs, named_args, **kwargs):
    N_CTX = kwargs["N_CTX"]
    STAGE = kwargs["STAGE"]
    HEAD_DIM = kwargs["HEAD_DIM"]
    # Filter out configs that would trip the kernel's static asserts at JIT
    # compile time (notably `tl.static_assert(BLOCK_N <= HEAD_DIM)`); also
    # keep the existing geometric constraints (BLOCK_M fits N_CTX, BLOCK_M >=
    # BLOCK_N outside STAGE=1).
    return [
        conf for conf in configs if conf.kwargs.get("BLOCK_M", 0) <= N_CTX and (
            conf.kwargs.get("BLOCK_M", 0) >= conf.kwargs.get("BLOCK_N", 0) or STAGE == 1)
        and conf.kwargs.get("BLOCK_N", 0) <= HEAD_DIM
    ]


@triton.jit
def _maybe_make_tensor_desc(desc_or_ptr, shape, strides, block_shape):
    if isinstance(desc_or_ptr, tl.tensor_descriptor):
        return desc_or_ptr
    else:
        return tl.make_tensor_descriptor(desc_or_ptr, shape, strides, block_shape)


@triton.autotune(configs=list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM", "FP8_OUTPUT", "warp_specialize"],
                 prune_configs_by={'early_config_prune': prune_invalid_configs})
@triton.jit
def _attn_fwd(sm_scale, M,  #
              Z, H, desc_q, desc_k, desc_v, desc_o,  #
              N_CTX,  #
              document_ptr, stride_doz, stride_dot,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              FP8_OUTPUT: tl.constexpr,  #
              HAS_DOCUMENT_MASK: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              warp_specialize: tl.constexpr,  #
              IS_HOPPER: tl.constexpr,  #
              DTYPE_IS_BF16: tl.constexpr,  #
              ):
    dtype = tl.float8e5 if FP8_OUTPUT else (tl.bfloat16 if DTYPE_IS_BF16 else tl.float16)
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    y_dim = Z * H * N_CTX
    desc_q = _maybe_make_tensor_desc(desc_q, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])
    if FP8_OUTPUT:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[HEAD_DIM, y_dim], strides=[N_CTX, 1],
                                         block_shape=[HEAD_DIM, BLOCK_N])
    else:
        desc_v = _maybe_make_tensor_desc(desc_v, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                         block_shape=[BLOCK_N, HEAD_DIM])
    desc_k = _maybe_make_tensor_desc(desc_k, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_N, HEAD_DIM])
    desc_o = _maybe_make_tensor_desc(desc_o, shape=[y_dim, HEAD_DIM], strides=[HEAD_DIM, 1],
                                     block_shape=[BLOCK_M, HEAD_DIM])

    offset_y = off_z * (N_CTX * H) + off_h * N_CTX
    qo_offset_y = offset_y + start_m * BLOCK_M
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # Initialize to large finite negative rather than -inf.
    # When a tile is entirely masked (e.g. a row with no same-document causal keys,
    # score += -1e6), m_ij = max(m_i, -1e6) stays finite, so alpha = exp2(m_i - m_ij)
    # never evaluates -inf - -inf and produces no NaN.
    # l_i is initialized to 1.0, so fully-masked rows produce acc/l_i = 0/1 = 0 correctly.
    # 1e9 is safely below any real logit (QK/sqrt(d) is typically O(1..10)).
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e9
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    q = desc_q.load([qo_offset_y, 0])
    # stage 1: off-band
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        offset_y, dtype, start_m, qk_scale,  #
                                        off_z, off_h,  #
                                        document_ptr, stride_doz, stride_dot, HAS_DOCUMENT_MASK,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX,  #
                                        warp_specialize, IS_HOPPER)
    # stage 2: on-band
    if STAGE & 2:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q,  #
                                        desc_k, desc_v,  #
                                        offset_y, dtype, start_m, qk_scale,  #
                                        off_z, off_h,  #
                                        document_ptr, stride_doz, stride_dot, HAS_DOCUMENT_MASK,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX,  #
                                        warp_specialize, IS_HOPPER)
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    desc_o.store([qo_offset_y, 0], acc.to(dtype))


@triton.jit
def _attn_bwd_preprocess(O, DO,  #
                         Delta,  #
                         Z, H, N_CTX,  #
                         BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr  #
                         ):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_hz = tl.program_id(1)
    off_n = tl.arange(0, HEAD_DIM)
    o = tl.load(O + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :])
    do = tl.load(DO + off_hz * HEAD_DIM * N_CTX + off_m[:, None] * HEAD_DIM + off_n[None, :]).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    tl.store(Delta + off_hz * N_CTX + off_m, delta)


@triton.jit
def _full_bwd_dkdv_kernel(
    Q, K, V, sm_scale,
    DO,
    DK, DV,
    M, D,
    DOCUMENT_IDX,
    stride_z, stride_h, stride_tok, stride_d,
    stride_doc_z, stride_doc_t,
    H, N_CTX,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HAS_DOCUMENT_MASK: tl.constexpr,
):
    """Compute dK and dV."""
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

    docs_k = tl.load(DOCUMENT_IDX + offs_k * stride_doc_t) if HAS_DOCUMENT_MASK else tl.zeros([BLOCK_K], dtype=tl.int32)

    dk = tl.zeros([BLOCK_K, HEAD_DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_K, HEAD_DIM], dtype=tl.float32)

    # Sweep q-blocks (causal: only q >= k).
    q_start = pid_k * BLOCK_K
    num_q_blocks = (N_CTX - q_start) // BLOCK_Q
    for q_blk_idx in range(num_q_blocks):
        offs_q = q_start + q_blk_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
        q = tl.load(Q + offs_q[:, None] * stride_tok + offs_d[None, :] * stride_d)
        do = tl.load(DO + offs_q[:, None] * stride_tok + offs_d[None, :] * stride_d)
        m_q = tl.load(M + offs_q)
        Di = tl.load(D + offs_q)

        # Recompute scores.
        qk = tl.dot(q, tl.trans(k))
        qk = qk * (sm_scale * 1.44269504)

        docs_q = tl.load(DOCUMENT_IDX + offs_q * stride_doc_t) if HAS_DOCUMENT_MASK else tl.zeros([BLOCK_Q], dtype=tl.int32)

        # Reconstruct probabilities.
        p = tl.math.exp2(qk - m_q[:, None])
        causal_mask = (offs_q[:, None] >= offs_k[None, :])
        if HAS_DOCUMENT_MASK:
            doc_mask = docs_q[:, None] == docs_k[None, :]
            p = tl.where(causal_mask & doc_mask, p, 0.0)
        else:
            p = tl.where(causal_mask, p, 0.0)

        p16 = p.to(do.dtype)
        dv += tl.dot(tl.trans(p16), do)

        # Softmax backward.
        dp = tl.dot(do, tl.trans(v)).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds16 = ds.to(q.dtype)
        dk += tl.dot(tl.trans(ds16), q)

    # Write dK and dV.
    dk_ptrs = DK + offs_k[:, None] * stride_tok + offs_d[None, :] * stride_d
    tl.store(dk_ptrs, dk * sm_scale)
    dv_ptrs = DV + offs_k[:, None] * stride_tok + offs_d[None, :] * stride_d
    tl.store(dv_ptrs, dv)


@triton.jit
def _full_bwd_dq_kernel(
    Q, K, V, sm_scale,
    DO,
    DQ,
    M, D,
    DOCUMENT_IDX,
    stride_z, stride_h, stride_tok, stride_d,
    stride_doc_z, stride_doc_t,
    H, N_CTX,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HAS_DOCUMENT_MASK: tl.constexpr,
):
    """Compute dQ."""
    LN2: tl.constexpr = 0.6931471824645996

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

    docs_q = tl.load(DOCUMENT_IDX + offs_q * stride_doc_t) if HAS_DOCUMENT_MASK else tl.zeros([BLOCK_Q], dtype=tl.int32)

    dq = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

    # Sweep k-blocks (causal: only k <= q).
    num_k_blocks = (pid_q * BLOCK_Q + BLOCK_Q) // BLOCK_K
    for k_blk_idx in range(num_k_blocks):
        offs_k = k_blk_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        kT = tl.load(K + offs_k[None, :] * stride_tok + offs_d[:, None] * stride_d)
        vT = tl.load(V + offs_k[None, :] * stride_tok + offs_d[:, None] * stride_d)

        docs_k = tl.load(DOCUMENT_IDX + offs_k * stride_doc_t) if HAS_DOCUMENT_MASK else tl.zeros([BLOCK_K], dtype=tl.int32)

        # Recompute scores.
        qk = tl.dot(q, kT)
        qk = qk * (sm_scale * 1.44269504)

        # Reconstruct p.
        p = tl.math.exp2(qk - m_q[:, None])
        causal_mask = (offs_q[:, None] >= offs_k[None, :])
        if HAS_DOCUMENT_MASK:
            doc_mask = docs_q[:, None] == docs_k[None, :]
            p = tl.where(causal_mask & doc_mask, p, 0.0)
        else:
            p = tl.where(causal_mask, p, 0.0)

        # Softmax backward.
        dp = tl.dot(do, vT).to(tl.float32)
        ds = p * (dp - Di[:, None])
        ds16 = ds.to(kT.dtype)
        dq += tl.dot(ds16, tl.trans(kT))

    dq_ptrs = DQ + offs_q[:, None] * stride_tok + offs_d[None, :] * stride_d
    dq *= sm_scale
    tl.store(dq_ptrs, dq)


class _full_attention(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        sm_scale,
        warp_specialize=True,
        documents_idx_BxT=None,
    ):
        """
        Forward pass for document-masked causal attention.

        Args:
            q: [B, H, T, D]
            k: [B, H, T, D]
            v: [B, H, T, D]
            sm_scale: float
            warp_specialize: bool
            documents_idx_BxT: optional [B, T] document indices; a query only
                attends to keys sharing its document id.
        """
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        has_document_mask = documents_idx_BxT is not None
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
        stage = 3  # causal
        extra_kern_args = {}
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
        if supports_host_descriptor() and not (is_hopper() and warp_specialize):
            y_dim = q.shape[0] * q.shape[1] * q.shape[2]
            dummy_block = [1, 1]
            desc_q = TensorDescriptor(q, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            if q.dtype == torch.float8_e5m2:
                desc_v = TensorDescriptor(v, shape=[HEAD_DIM_K, y_dim], strides=[q.shape[2], 1], block_shape=dummy_block)
            else:
                desc_v = TensorDescriptor(v, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            desc_k = TensorDescriptor(k, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
            desc_o = TensorDescriptor(o, shape=[y_dim, HEAD_DIM_K], strides=[HEAD_DIM_K, 1], block_shape=dummy_block)
        else:
            desc_q = q
            desc_v = v
            desc_k = k
            desc_o = o

        def alloc_fn(size: int, align: int, _):
            return torch.empty(size, dtype=torch.int8, device="cuda")

        triton.set_allocator(alloc_fn)

        def grid(META):
            return (triton.cdiv(q.shape[2], META["BLOCK_M"]), q.shape[0] * q.shape[1], 1)

        if is_blackwell() and warp_specialize:
            if HEAD_DIM_K == 128 and q.dtype == torch.float16:
                extra_kern_args["maxnreg"] = 168
            else:
                extra_kern_args["maxnreg"] = 80

        if has_document_mask:
            document_ptr = documents_idx_BxT
            stride_doz, stride_dot = documents_idx_BxT.stride()
        else:
            document_ptr = q
            stride_doz = stride_dot = 0

        _attn_fwd[grid](
            sm_scale, M,
            q.shape[0], q.shape[1],
            desc_q, desc_k, desc_v, desc_o,
            N_CTX=q.shape[2],
            document_ptr=document_ptr, stride_doz=stride_doz, stride_dot=stride_dot,
            HEAD_DIM=HEAD_DIM_K,
            FP8_OUTPUT=q.dtype == torch.float8_e5m2,
            HAS_DOCUMENT_MASK=has_document_mask,
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

        BATCH, N_HEAD, N_CTX, HEAD_DIM = q.shape

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        PRE_BLOCK = 128
        NUM_WARPS = 4
        NUM_STAGES = 2
        BLOCK_Q_DKDV = 64
        BLOCK_Q_DQ = 128
        BLOCK_K = 64

        assert N_CTX % PRE_BLOCK == 0
        pre_grid = (N_CTX // PRE_BLOCK, BATCH * N_HEAD)
        delta = torch.empty_like(M)
        _attn_bwd_preprocess[pre_grid](
            o, do,
            delta,
            BATCH, N_HEAD, N_CTX,
            BLOCK_M=PRE_BLOCK, HEAD_DIM=HEAD_DIM,
        )

        # dK / dV.
        num_k_blocks = N_CTX // BLOCK_K
        grid_dkdv = (num_k_blocks, BATCH * N_HEAD)
        _full_bwd_dkdv_kernel[grid_dkdv](
            q, k, v, ctx.sm_scale,
            do,
            dk, dv,
            M, delta,
            documents_idx_BxT,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            documents_idx_BxT.stride(0), documents_idx_BxT.stride(1) if ctx.has_document_mask else 0,
            N_HEAD, N_CTX,
            BLOCK_Q=BLOCK_Q_DKDV, BLOCK_K=BLOCK_K,
            HEAD_DIM=HEAD_DIM,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
            HAS_DOCUMENT_MASK=ctx.has_document_mask,
        )

        # dQ.
        grid_dq = (N_CTX // BLOCK_Q_DQ, BATCH * N_HEAD)
        _full_bwd_dq_kernel[grid_dq](
            q, k, v, ctx.sm_scale,
            do,
            dq,
            M, delta,
            documents_idx_BxT,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            documents_idx_BxT.stride(0), documents_idx_BxT.stride(1) if ctx.has_document_mask else 0,
            N_HEAD, N_CTX,
            BLOCK_Q=BLOCK_Q_DQ, BLOCK_K=BLOCK_K,
            HEAD_DIM=HEAD_DIM,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
            HAS_DOCUMENT_MASK=ctx.has_document_mask,
        )

        dq_out = dq[:, :, : ctx.n_ctx_orig, :].to(q.dtype)
        dk_out = dk[:, :, : ctx.n_ctx_orig, :].to(k.dtype)
        dv_out = dv[:, :, : ctx.n_ctx_orig, :].to(v.dtype)

        return (
            dq_out,
            dk_out,
            dv_out,
            None,  # sm_scale
            None,  # warp_specialize
            None,  # documents_idx_BxT
        )


def full_attention(
    q,
    k,
    v,
    sm_scale,
    warp_specialize=True,
    documents_idx_BxT=None,
):
    """
    Triton document-masked causal attention (plain Flash Attention v2).

    A query attends causally to all keys sharing its document id (or to all
    causal keys when documents_idx_BxT is None). This is the kernel used by the
    Standard (full-attention) model.

    Args:
        q: [B, H, T, D]
        k: [B, H, T, D]
        v: [B, H, T, D]
        sm_scale: float
        warp_specialize: bool
        documents_idx_BxT: optional [B, T] document indices
    """
    return _full_attention.apply(
        q,
        k,
        v,
        sm_scale,
        warp_specialize,
        documents_idx_BxT,
    )
