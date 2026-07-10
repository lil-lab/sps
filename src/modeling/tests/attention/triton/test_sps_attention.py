import math

import pytest
import torch
import triton


def _import_triton_sps_attention():
    try:
        from modeling.models.attention.triton_sps_flash_attention import sps_sliding_attention
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Triton SPS attention import failed: {exc}")
    return sps_sliding_attention


def test_triton_sps_import_uses_triton_configs() -> None:
    from modeling.models.attention.triton_reverse_sps_flash_attention import ATTN_FWD_CONFIGS
    from modeling.models.attention.triton_sps_flash_attention import sps_sliding_attention

    assert sps_sliding_attention is not None
    assert ATTN_FWD_CONFIGS
    assert all(isinstance(conf, triton.Config) for conf in ATTN_FWD_CONFIGS)


def _sdpa_reference_attention_sps(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    window_size: int,
    documents_idx_BxT: torch.Tensor | None = None,
) -> torch.Tensor:
    two_t = q.shape[2]
    q_pos = torch.arange(two_t, device=q.device).view(two_t, 1)
    k_pos = torch.arange(two_t, device=q.device).view(1, two_t)
    q_tok = q_pos // 2
    k_tok = k_pos // 2
    is_permanent_key = (k_pos % 2 == 0)

    rel = q_tok - k_tok
    local_additive = torch.where(
        rel > window_size,
        torch.full((), -1.0e6, device=q.device, dtype=q.dtype),
        torch.zeros((), device=q.device, dtype=q.dtype),
    ).view(1, 1, two_t, two_t)

    # Permanent (predict) keys are always visible (no additive bias).
    attn_bias = torch.where(
        is_permanent_key.view(1, 1, 1, two_t),
        torch.zeros((), device=q.device, dtype=q.dtype),
        local_additive,
    )

    causal = torch.tril(torch.ones((two_t, two_t), device=q.device, dtype=torch.bool))
    causal_additive = torch.where(
        causal,
        torch.zeros((), device=q.device, dtype=q.dtype),
        torch.full((), float("-inf"), device=q.device, dtype=q.dtype),
    ).view(1, 1, two_t, two_t)
    attn_mask = attn_bias + causal_additive
    if documents_idx_BxT is not None:
        same_doc = (documents_idx_BxT.unsqueeze(-1) == documents_idx_BxT.unsqueeze(-2)).unsqueeze(1)
        doc_additive = torch.where(
            same_doc,
            torch.zeros((), device=q.device, dtype=q.dtype),
            torch.full((), float("-inf"), device=q.device, dtype=q.dtype),
        )
        attn_mask = attn_mask + doc_additive
    return torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=attn_mask,
        is_causal=False,
    )


@pytest.mark.cuda
@pytest.mark.parametrize("has_document_mask", [False, True])
def test_triton_sps_sliding_matches_sdpa_forward_backward(has_document_mask: bool):
    sps_sliding_attention = _import_triton_sps_attention()

    torch.manual_seed(17)
    device = "cuda"
    dtype = torch.bfloat16
    b, h, t, d = 1, 2, 16, 64
    two_t = 2 * t
    window_size = 3
    sm_scale = 1.0 / math.sqrt(d)

    q_tri = torch.randn(b, h, two_t, d, device=device, dtype=dtype, requires_grad=True)
    k_tri = torch.randn(b, h, two_t, d, device=device, dtype=dtype, requires_grad=True)
    v_tri = torch.randn(b, h, two_t, d, device=device, dtype=dtype, requires_grad=True)

    q_ref = q_tri.detach().clone().requires_grad_(True)
    k_ref = k_tri.detach().clone().requires_grad_(True)
    v_ref = v_tri.detach().clone().requires_grad_(True)

    documents_idx = None
    if has_document_mask:
        documents_idx = torch.tensor(
            [[0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2]],
            device=device,
            dtype=torch.long,
        ).repeat_interleave(2, dim=1)

    out_tri = sps_sliding_attention(
        q_tri,
        k_tri,
        v_tri,
        sm_scale,
        window_size,
        warp_specialize=False,
        documents_idx_BxT=documents_idx,
    )
    out_ref = _sdpa_reference_attention_sps(
        q_ref,
        k_ref,
        v_ref,
        window_size,
        documents_idx_BxT=documents_idx,
    )

    torch.testing.assert_close(out_tri, out_ref, atol=3e-2, rtol=3e-2)

    grad_out = torch.randn_like(out_tri)
    out_tri.backward(grad_out)
    out_ref.backward(grad_out)

    torch.testing.assert_close(q_tri.grad, q_ref.grad, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(k_tri.grad, k_ref.grad, atol=4e-2, rtol=4e-2)
    torch.testing.assert_close(v_tri.grad, v_ref.grad, atol=4e-2, rtol=4e-2)
