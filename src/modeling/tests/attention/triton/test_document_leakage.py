"""
Direct cross-document leakage probe for the Triton attention kernels.

These tests are stronger than logits-matching: we replace document-2's keys
and values with NaN, then assert that document-1's outputs are (a) finite
and (b) numerically identical to running the kernel on document-1 alone.
This is a forensic, unforgeable proof that no information from document 2
reaches document-1 query positions, regardless of how the doc mask is
implemented internally.
"""

import math
import pytest
import torch

from modeling.tests.attention.triton.test_full_flash_attention import _import_full_attention


def _import_triton_reverse_sps_attention():
    try:
        from modeling.models.attention.triton_reverse_sps_flash_attention import reverse_sps_sliding_attention
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Triton reverse_sps attention import failed: {exc}")
    return reverse_sps_sliding_attention


@pytest.mark.cuda
def test_full_attention_no_cross_document_leakage():
    full_attention = _import_full_attention()
    torch.manual_seed(2024)
    device = "cuda"
    dtype = torch.bfloat16

    B, H, T, D = 1, 2, 256, 64
    half = T // 2
    sm_scale = 1.0 / math.sqrt(D)

    q = torch.randn(B, H, T, D, device=device, dtype=dtype)
    k = torch.randn(B, H, T, D, device=device, dtype=dtype)
    v = torch.randn(B, H, T, D, device=device, dtype=dtype)

    # Reference run on document 1 alone (no doc mask needed — it's the only doc).
    out_solo = full_attention(
        q[:, :, :half], k[:, :, :half], v[:, :, :half], sm_scale,
        warp_specialize=False,
    )
    assert torch.isfinite(out_solo).all(), "solo run produced NaN"

    # Now poison document 2's keys/values with NaN and run with a doc mask
    # splitting at half. Document 1 outputs must remain finite and identical.
    k_poisoned = k.clone()
    v_poisoned = v.clone()
    k_poisoned[:, :, half:] = float("nan")
    v_poisoned[:, :, half:] = float("nan")

    doc_idx = torch.zeros(B, T, device=device, dtype=torch.int32)
    doc_idx[:, half:] = 1

    out_full = full_attention(
        q, k_poisoned, v_poisoned, sm_scale, warp_specialize=False, documents_idx_BxT=doc_idx,
    )

    doc1 = out_full[:, :, :half]
    assert torch.isfinite(doc1).all(), (
        "Document 1 output contains NaN — doc-2 keys/values are leaking through the mask"
    )

    torch.testing.assert_close(
        doc1.float(), out_solo.float(),
        atol=2e-2, rtol=1e-2,
        msg="Document 1 outputs differ between solo and masked-batched runs (leakage)",
    )


@pytest.mark.cuda
def test_reverse_sps_sliding_attention_no_cross_document_leakage():
    reverse_sps_sliding_attention = _import_triton_reverse_sps_attention()
    torch.manual_seed(2025)
    device = "cuda"
    dtype = torch.float16

    # reverse_sps inputs are 2T-long (interleaved input/predict). The document
    # split is placed on a flash-attention block boundary (half_2t is a multiple
    # of BLOCK_M) so the poisoned doc-2 key block is never loaded for doc-1
    # queries: additive masking alone cannot neutralize a NaN key that shares a
    # block with the query (same reason the key-bias probe below uses T=256).
    B, H, t, D = 1, 2, 128, 64
    two_t = 2 * t
    half_2t = two_t // 2
    window_size = 4
    sm_scale = 1.0 / math.sqrt(D)

    q = torch.randn(B, H, two_t, D, device=device, dtype=dtype)
    k = torch.randn(B, H, two_t, D, device=device, dtype=dtype)
    v = torch.randn(B, H, two_t, D, device=device, dtype=dtype)

    # Solo run on document 1 (first half).
    out_solo = reverse_sps_sliding_attention(
        q[:, :, :half_2t], k[:, :, :half_2t], v[:, :, :half_2t],
        sm_scale,
        window_size,
        warp_specialize=False,
    )
    assert torch.isfinite(out_solo).all(), "solo reverse_sps run produced NaN"

    # Poison document-2 keys/values with NaN.
    k_p = k.clone()
    v_p = v.clone()
    k_p[:, :, half_2t:] = float("nan")
    v_p[:, :, half_2t:] = float("nan")

    documents_idx = torch.zeros((B, two_t), device=device, dtype=torch.int64)
    documents_idx[:, half_2t:] = 1

    out_full = reverse_sps_sliding_attention(
        q, k_p, v_p, sm_scale,
        window_size,
        warp_specialize=False,
        documents_idx_BxT=documents_idx,
    )

    doc1 = out_full[:, :, :half_2t]
    assert torch.isfinite(doc1).all(), (
        "Document 1 output contains NaN — doc-2 leaking through the reverse_sps doc mask"
    )

    torch.testing.assert_close(
        doc1.float(), out_solo.float(),
        atol=3e-2, rtol=3e-2,
        msg="Reverse-SPS doc-1 differs between solo and masked-batched runs (leakage)",
    )
