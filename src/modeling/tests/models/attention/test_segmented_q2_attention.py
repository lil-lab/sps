from __future__ import annotations

import math

import pytest
import torch

from modeling.models.utils.decode_attention import (
    KVSegment,
    triton,
    triton_segmented_q2_attention,
)


def _segment_len(segment: KVSegment, batch_idx: int, head_idx: int) -> int:
    if segment.lengths.dim() == 1:
        length = int(segment.lengths[batch_idx].item())
    else:
        length = int(segment.lengths[batch_idx, head_idx].item())
    if segment.max_len_cap is not None:
        length = min(length, int(segment.max_len_cap))
    return max(length, 0)


def _reference_q2_attention(
    q_Bx2xHxD: torch.Tensor,
    segments: list[KVSegment],
    *,
    current_normal_k_BxHxD: torch.Tensor,
    current_normal_v_BxHxD: torch.Tensor,
    current_predict_k_BxHxD: torch.Tensor,
    current_predict_v_BxHxD: torch.Tensor,
) -> torch.Tensor:
    bsz, _, n_head, head_dim = q_Bx2xHxD.shape
    out = torch.empty_like(q_Bx2xHxD, dtype=torch.float32)
    scale = 1.0 / math.sqrt(head_dim)
    for batch_idx in range(bsz):
        for head_idx in range(n_head):
            keys = []
            values = []
            for segment in segments:
                length = _segment_len(segment, batch_idx, head_idx)
                if length == 0:
                    continue
                keys.append(segment.k[batch_idx, head_idx, :length].float())
                values.append(segment.v[batch_idx, head_idx, :length].float())

            normal_k = current_normal_k_BxHxD[batch_idx, head_idx].float().view(1, head_dim)
            normal_v = current_normal_v_BxHxD[batch_idx, head_idx].float().view(1, head_dim)
            predict_k = current_predict_k_BxHxD[batch_idx, head_idx].float().view(1, head_dim)
            predict_v = current_predict_v_BxHxD[batch_idx, head_idx].float().view(1, head_dim)

            slot0_k = torch.cat([*keys, normal_k], dim=0) if keys else normal_k
            slot0_v = torch.cat([*values, normal_v], dim=0) if values else normal_v
            slot1_k = torch.cat([*keys, normal_k, predict_k], dim=0) if keys else torch.cat([normal_k, predict_k], dim=0)
            slot1_v = torch.cat([*values, normal_v, predict_v], dim=0) if values else torch.cat([normal_v, predict_v], dim=0)

            q0 = q_Bx2xHxD[batch_idx, 0, head_idx].float()
            q1 = q_Bx2xHxD[batch_idx, 1, head_idx].float()
            out[batch_idx, 0, head_idx] = torch.softmax(q0 @ slot0_k.T * scale, dim=-1) @ slot0_v
            out[batch_idx, 1, head_idx] = torch.softmax(q1 @ slot1_k.T * scale, dim=-1) @ slot1_v
    return out


@pytest.mark.cuda
@pytest.mark.parametrize("first_lengths", ["shared", "empty"])
def test_segmented_q2_attention_matches_reference(first_lengths: str) -> None:
    if (not torch.cuda.is_available()) or triton is None:
        pytest.skip("q2 segmented attention requires CUDA + Triton")

    torch.manual_seed(123)
    bsz, n_head, head_dim = 2, 3, 16
    q = torch.randn(bsz, 2, n_head, head_dim, device="cuda", dtype=torch.bfloat16)
    k0 = torch.randn(bsz, n_head, 7, head_dim, device="cuda", dtype=torch.bfloat16)
    v0 = torch.randn_like(k0)
    k1 = torch.randn(bsz, n_head, 11, head_dim, device="cuda", dtype=torch.bfloat16)
    v1 = torch.randn_like(k1)

    if first_lengths == "empty":
        len0 = torch.zeros((bsz,), device="cuda", dtype=torch.long)
        cap0 = 0
    else:
        len0 = torch.tensor([2, 5], device="cuda", dtype=torch.long)
        cap0 = 5
    len1 = torch.tensor(
        [[1, 3, 6], [2, 4, 7]],
        device="cuda",
        dtype=torch.long,
    )
    segments = [
        KVSegment(k0, v0, len0, max_len_cap=cap0),
        KVSegment(k1, v1, len1, max_len_cap=8),
    ]
    current_normal_k = torch.randn(bsz, n_head, head_dim, device="cuda", dtype=torch.bfloat16)
    current_normal_v = torch.randn_like(current_normal_k)
    current_predict_k = torch.randn_like(current_normal_k)
    current_predict_v = torch.randn_like(current_normal_k)

    actual = triton_segmented_q2_attention(
        q,
        segments,
        current_normal_k_BxHxD=current_normal_k,
        current_normal_v_BxHxD=current_normal_v,
        current_predict_k_BxHxD=current_predict_k,
        current_predict_v_BxHxD=current_predict_v,
        attn_dtype=torch.bfloat16,
    )
    expected = _reference_q2_attention(
        q,
        segments,
        current_normal_k_BxHxD=current_normal_k,
        current_normal_v_BxHxD=current_normal_v,
        current_predict_k_BxHxD=current_predict_k,
        current_predict_v_BxHxD=current_predict_v,
    )

    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
