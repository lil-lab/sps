"""
Long-context fwd+bwd parity for full_attention.

test_full_flash_attention.py covers shapes up to ~1024. This file extends
coverage to 2048/4096 with document masking, closer to production training
shapes.

The kernel only supports bf16/fp16 (no fp32), so tolerances stay in the
existing bf16 regime; the value of these tests is shape coverage, not
tighter precision.

Calls pin ``warp_specialize=False`` (the production path); the kernel's
``warp_specialize=True`` default fails to compile on Hopper. See the module
docstring of ``test_full_flash_attention.py``.
"""

import math
import pytest
import torch

from modeling.tests.attention.triton.test_full_flash_attention import (
    _import_full_attention,
    _pytorch_reference_attention,
)


LONG_SHAPES = [
    # (B, H, T, D)
    (1, 4, 2048, 64),
    (1, 2, 4096, 64),
    (1, 2, 2048, 128),
    # XS prefill shape: H=8 with T=4032 (NOT a multiple of BLOCK_M=128) hits the
    # tensor-descriptor cross-head overflow bug if the wrapper does not pad N_CTX.
    # See triton_full_flash_attention._full_attention.forward for the fix.
    (8, 8, 4032, 64),
    # Adjacent boundary cases — single-head and multi-head with a non-multiple T.
    (1, 1, 4032, 64),
    (1, 12, 4032, 64),
]


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.parametrize("B,H,T,D", LONG_SHAPES)
def test_forward_long_context(B, H, T, D):
    full_attention = _import_full_attention()
    torch.manual_seed(101)
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
        msg=f"Forward long-context mismatch ({B},{H},{T},{D})",
    )


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.parametrize("B,H,T,D", [(1, 4, 2048, 64)])
def test_backward_long_context(B, H, T, D):
    full_attention = _import_full_attention()
    torch.manual_seed(102)
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
    torch.testing.assert_close(tri_dq.float(), q_ref.grad.float(), atol=atol, rtol=1e-1, msg="dq long")
    torch.testing.assert_close(tri_dk.float(), k_ref.grad.float(), atol=atol, rtol=1e-1, msg="dk long")
    torch.testing.assert_close(tri_dv.float(), v_ref.grad.float(), atol=atol, rtol=1e-1, msg="dv long")


@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.parametrize("B,H,T,D", [(2, 4, 2048, 64)])
def test_forward_long_context_with_doc_mask(B, H, T, D):
    full_attention = _import_full_attention()
    torch.manual_seed(103)
    device = "cuda"

    q = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)
    k = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)
    v = torch.randn(B, H, T, D, device=device, dtype=torch.bfloat16)
    sm_scale = 1.0 / math.sqrt(D)

    # Three documents per row at irregular boundaries.
    doc_idx = torch.zeros(B, T, device=device, dtype=torch.int32)
    doc_idx[:, T // 3:] = 1
    doc_idx[:, (2 * T) // 3:] = 2

    tri_out = full_attention(q, k, v, sm_scale, warp_specialize=False, documents_idx_BxT=doc_idx)
    ref_out = _pytorch_reference_attention(q, k, v, sm_scale, documents_idx_BxT=doc_idx)

    torch.testing.assert_close(
        tri_out.float(), ref_out.float(),
        atol=2e-2, rtol=1e-2,
        msg="Forward long-context (doc mask) mismatch",
    )
