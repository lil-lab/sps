"""
Tests for triton_full_flash_attention: document-masked causal attention.

Compares the Triton kernel against a plain PyTorch reference for:
- Forward pass correctness (plain causal and with document masking)
- Backward pass gradients (dq, dk, dv)
- Various shapes (batch, heads, seq len, head dim)
- warp_specialize on/off and bf16/fp16 dtypes

These parity tests pin ``warp_specialize=False`` — the path the Standard model runs
in production. The kernel's ``warp_specialize=True`` default fails to *compile* on
Hopper (sm_90): Triton's ``add_warp_specialize`` MLIR pass errors out. The dedicated
``test_warp_specialize_fwd_bwd`` still exercises both paths, skipping ``True`` on Hopper.
"""

import math
import pytest
import torch


def _import_full_attention():
    try:
        from modeling.models.attention.triton_full_flash_attention import full_attention
    except Exception as exc:
        pytest.skip(f"Triton full attention import failed: {exc}")
    return full_attention


def _pytorch_reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sm_scale: float,
    documents_idx_BxT: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Pure PyTorch reference: causal attention with optional per-document masking.
    """
    B, H, T, D = q.shape
    qf = q.float()
    kf = k.float()
    vf = v.float()

    scores = torch.matmul(qf, kf.transpose(-2, -1)) * sm_scale  # [B, H, T, T]

    # Causal mask
    causal = torch.ones(T, T, device=q.device, dtype=torch.bool).tril()
    scores = scores.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float("-inf"))

    # Document mask
    if documents_idx_BxT is not None:
        doc_q = documents_idx_BxT.unsqueeze(1).unsqueeze(3)  # [B, 1, T, 1]
        doc_k = documents_idx_BxT.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
        same_doc = (doc_q == doc_k)
        scores = scores.masked_fill(~same_doc, float("-inf"))

    p = torch.softmax(scores, dim=-1)
    out = torch.matmul(p, vf)
    return out


# ---------------------------------------------------------------------------
# Test configurations
# ---------------------------------------------------------------------------

SHAPES_SMALL = [
    # (B, H, T, D)
    (1, 1, 128, 64),
    (2, 4, 128, 64),
    (1, 2, 256, 128),
    (2, 8, 256, 64),
]

SHAPES_MEDIUM = [
    (1, 8, 512, 64),
    (2, 4, 512, 128),
    (1, 2, 1024, 64),
]


@pytest.mark.parametrize("B,H,T,D", SHAPES_SMALL)
def test_forward(B, H, T, D):
    """Forward pass: Triton vs PyTorch reference (plain causal attention)."""
    full_attention = _import_full_attention()
    torch.manual_seed(42)
    device = "cuda"

    q = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)
    sm_scale = 1.0 / math.sqrt(D)

    tri_out = full_attention(q, k, v, sm_scale, warp_specialize=False)
    ref_out = _pytorch_reference_attention(q, k, v, sm_scale)

    torch.testing.assert_close(
        tri_out.float(), ref_out.float(),
        atol=2e-2, rtol=1e-2,
        msg=f"Forward mismatch for shape ({B},{H},{T},{D})",
    )


@pytest.mark.parametrize("B,H,T,D", SHAPES_MEDIUM)
def test_forward_medium(B, H, T, D):
    """Forward pass at medium scale."""
    full_attention = _import_full_attention()
    torch.manual_seed(42)
    device = "cuda"

    q = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)
    sm_scale = 1.0 / math.sqrt(D)

    tri_out = full_attention(q, k, v, sm_scale, warp_specialize=False)
    ref_out = _pytorch_reference_attention(q, k, v, sm_scale)

    torch.testing.assert_close(
        tri_out.float(), ref_out.float(),
        atol=2e-2, rtol=1e-2,
        msg=f"Forward (medium) mismatch for shape ({B},{H},{T},{D})",
    )


@pytest.mark.parametrize("B,H,T,D", SHAPES_SMALL)
def test_backward(B, H, T, D):
    """Backward pass: compare dq, dk, dv against PyTorch autograd."""
    full_attention = _import_full_attention()
    torch.manual_seed(42)
    device = "cuda"

    q = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    sm_scale = 1.0 / math.sqrt(D)
    do = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)

    tri_out = full_attention(q, k, v, sm_scale, warp_specialize=False)
    tri_out.backward(do)
    tri_dq = q.grad.clone(); q.grad = None
    tri_dk = k.grad.clone(); k.grad = None
    tri_dv = v.grad.clone(); v.grad = None

    q_ref = q.detach().float().requires_grad_(True)
    k_ref = k.detach().float().requires_grad_(True)
    v_ref = v.detach().float().requires_grad_(True)

    ref_out = _pytorch_reference_attention(q_ref, k_ref, v_ref, sm_scale)
    ref_out.backward(do.float())

    atol = 5e-2
    torch.testing.assert_close(tri_dq.float(), q_ref.grad.float(), atol=atol, rtol=1e-1,
                               msg=f"dq mismatch for shape ({B},{H},{T},{D})")
    torch.testing.assert_close(tri_dk.float(), k_ref.grad.float(), atol=atol, rtol=1e-1,
                               msg=f"dk mismatch for shape ({B},{H},{T},{D})")
    torch.testing.assert_close(tri_dv.float(), v_ref.grad.float(), atol=atol, rtol=1e-1,
                               msg=f"dv mismatch for shape ({B},{H},{T},{D})")


@pytest.mark.parametrize("B,H,T,D", [(2, 4, 256, 64)])
def test_forward_with_document_mask(B, H, T, D):
    """Forward pass with document masking."""
    full_attention = _import_full_attention()
    torch.manual_seed(42)
    device = "cuda"

    q = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)
    sm_scale = 1.0 / math.sqrt(D)

    # Create document indices: 2 documents per batch
    doc_idx = torch.zeros(B, T, device=device, dtype=torch.int32)
    doc_idx[:, T // 2:] = 1

    tri_out = full_attention(q, k, v, sm_scale, warp_specialize=False, documents_idx_BxT=doc_idx)
    ref_out = _pytorch_reference_attention(q, k, v, sm_scale, documents_idx_BxT=doc_idx)

    torch.testing.assert_close(
        tri_out.float(), ref_out.float(),
        atol=2e-2, rtol=1e-2,
        msg=f"Forward (doc mask) mismatch for shape ({B},{H},{T},{D})",
    )


@pytest.mark.parametrize("B,H,T,D", [(2, 4, 256, 64)])
def test_backward_with_document_mask(B, H, T, D):
    """Backward pass with document masking."""
    full_attention = _import_full_attention()
    torch.manual_seed(42)
    device = "cuda"

    q = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    sm_scale = 1.0 / math.sqrt(D)
    do = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)

    doc_idx = torch.zeros(B, T, device=device, dtype=torch.int32)
    doc_idx[:, T // 2:] = 1

    tri_out = full_attention(q, k, v, sm_scale, warp_specialize=False, documents_idx_BxT=doc_idx)
    tri_out.backward(do)
    tri_dq = q.grad.clone(); q.grad = None
    tri_dk = k.grad.clone(); k.grad = None
    tri_dv = v.grad.clone(); v.grad = None

    q_ref = q.detach().float().requires_grad_(True)
    k_ref = k.detach().float().requires_grad_(True)
    v_ref = v.detach().float().requires_grad_(True)

    ref_out = _pytorch_reference_attention(q_ref, k_ref, v_ref, sm_scale, documents_idx_BxT=doc_idx)
    ref_out.backward(do.float())

    atol = 5e-2
    torch.testing.assert_close(tri_dq.float(), q_ref.grad.float(), atol=atol, rtol=1e-1, msg="dq mismatch (doc mask)")
    torch.testing.assert_close(tri_dk.float(), k_ref.grad.float(), atol=atol, rtol=1e-1, msg="dk mismatch (doc mask)")
    torch.testing.assert_close(tri_dv.float(), v_ref.grad.float(), atol=atol, rtol=1e-1, msg="dv mismatch (doc mask)")


@pytest.mark.parametrize("with_doc_mask", [False, True])
@pytest.mark.parametrize("B,H,T,D", [(1, 2, 200, 64)])
def test_backward_through_padding(B, H, T, D, with_doc_mask):
    """Forward and backward at a non-multiple-of-128 length (T=200, padded to 256).

    Exercises the grad-through-pad path: the forward rounds the sequence up to 128
    before launching, and the backward pads ``do`` to match before slicing the
    grads back to T. Run both plain and with a document mask.
    """
    full_attention = _import_full_attention()
    torch.manual_seed(1234)
    device = "cuda"

    q = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    sm_scale = 1.0 / math.sqrt(D)
    do = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)

    doc_idx = None
    if with_doc_mask:
        doc_idx = torch.zeros(B, T, device=device, dtype=torch.int32)
        doc_idx[:, T // 2:] = 1

    tri_out = full_attention(q, k, v, sm_scale, warp_specialize=False, documents_idx_BxT=doc_idx)
    assert tri_out.shape == (B, H, T, D)  # padding is sliced back off before return
    tri_out.backward(do)
    tri_dq = q.grad.clone(); q.grad = None
    tri_dk = k.grad.clone(); k.grad = None
    tri_dv = v.grad.clone(); v.grad = None

    q_ref = q.detach().float().requires_grad_(True)
    k_ref = k.detach().float().requires_grad_(True)
    v_ref = v.detach().float().requires_grad_(True)
    ref_out = _pytorch_reference_attention(q_ref, k_ref, v_ref, sm_scale, documents_idx_BxT=doc_idx)
    ref_out.backward(do.float())

    tag = "doc mask" if with_doc_mask else "plain"
    torch.testing.assert_close(tri_out.float(), ref_out.float(), atol=2e-2, rtol=1e-2,
                               msg=f"forward mismatch through padding ({tag})")
    atol = 5e-2
    torch.testing.assert_close(tri_dq.float(), q_ref.grad.float(), atol=atol, rtol=1e-1,
                               msg=f"dq mismatch through padding ({tag})")
    torch.testing.assert_close(tri_dk.float(), k_ref.grad.float(), atol=atol, rtol=1e-1,
                               msg=f"dk mismatch through padding ({tag})")
    torch.testing.assert_close(tri_dv.float(), v_ref.grad.float(), atol=atol, rtol=1e-1,
                               msg=f"dv mismatch through padding ({tag})")


@pytest.mark.parametrize("warp_specialize", [False, True])
@pytest.mark.parametrize("B,H,T,D", [(1, 2, 256, 64), (2, 4, 256, 64)])
def test_warp_specialize_fwd_bwd(B, H, T, D, warp_specialize):
    """Exercise both warp_specialize=False and True paths (fwd + bwd)."""
    full_attention = _import_full_attention()
    if warp_specialize and torch.cuda.get_device_capability()[0] == 9:
        pytest.skip(
            "warp_specialize=True fails to compile on Hopper (sm_90): Triton's "
            "add_warp_specialize MLIR pass errors. Production runs warp_specialize=False."
        )
    torch.manual_seed(909)
    device = "cuda"

    q = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16, requires_grad=True)
    sm_scale = 1.0 / math.sqrt(D)
    do = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)

    try:
        tri_out = full_attention(q, k, v, sm_scale, warp_specialize=warp_specialize)
    except TypeError as exc:
        pytest.skip(f"warp_specialize kwarg not exposed: {exc}")
    tri_out.backward(do)
    tri_dq = q.grad.clone(); q.grad = None
    tri_dk = k.grad.clone(); k.grad = None
    tri_dv = v.grad.clone(); v.grad = None

    q_ref = q.detach().float().requires_grad_(True)
    k_ref = k.detach().float().requires_grad_(True)
    v_ref = v.detach().float().requires_grad_(True)
    ref_out = _pytorch_reference_attention(q_ref, k_ref, v_ref, sm_scale)
    ref_out.backward(do.float())

    torch.testing.assert_close(tri_out.float(), ref_out.float(), atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(tri_dq.float(), q_ref.grad.float(), atol=5e-2, rtol=1e-1)
    torch.testing.assert_close(tri_dk.float(), k_ref.grad.float(), atol=5e-2, rtol=1e-1)
    torch.testing.assert_close(tri_dv.float(), v_ref.grad.float(), atol=5e-2, rtol=1e-1)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_dtype_support(dtype):
    """Test both bf16 and fp16 dtypes."""
    full_attention = _import_full_attention()
    torch.manual_seed(42)
    device = "cuda"
    B, H, T, D = 1, 2, 128, 64

    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype)
    sm_scale = 1.0 / math.sqrt(D)

    tri_out = full_attention(q, k, v, sm_scale, warp_specialize=False)
    ref_out = _pytorch_reference_attention(q, k, v, sm_scale)

    torch.testing.assert_close(
        tri_out.float(), ref_out.float(),
        atol=2e-2, rtol=1e-2,
        msg=f"Forward mismatch for dtype={dtype}",
    )
