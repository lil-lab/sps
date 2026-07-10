from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor


def sample_next_token(
    logits_BxV: Tensor,
    active_mask_B: Tensor,
    *,
    pad_token_id: int,
    suppressed_token_ids: Sequence[int] = (),
    do_sample: bool,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float] = None,
    forbidden_token_ids: Optional[Tensor] = None,
) -> Tensor:
    """Sample the next token from logits, respecting an active-batch mask.

    Args:
        logits_BxV: raw logits [B, V].
        active_mask_B: bool mask [B] — True for rows that should be sampled.
        pad_token_id: token id used for inactive rows.
        suppressed_token_ids: token ids that are always suppressed (e.g. predict, pad).
        do_sample: if True, sample from the distribution; otherwise argmax.
        temperature: sampling temperature (must be > 0 when do_sample=True).
        top_k: if set, restrict sampling to top-k logits.
        top_p: if set, nucleus-sample from the smallest prefix with cumulative
            probability >= top_p.
        forbidden_token_ids: optional tensor of additional token ids to suppress.

    Returns:
        next_token_B [B] — sampled tokens for active rows, pad_token_id elsewhere.
    """
    active_mask_B = active_mask_B.to(dtype=torch.bool, device=logits_BxV.device)
    next_token_B = torch.full(
        (logits_BxV.size(0),),
        pad_token_id,
        device=logits_BxV.device,
        dtype=torch.long,
    )
    if not bool(active_mask_B.any()):
        return next_token_B

    if top_k is not None and top_k <= 0:
        raise ValueError(f"top_k must be positive when provided, got {top_k}")
    if top_p is not None and not 0.0 < float(top_p) <= 1.0:
        raise ValueError(f"top_p must be in (0, 1], got {top_p}")
    if do_sample and temperature <= 0.0:
        raise ValueError(f"temperature must be > 0 when do_sample=True, got {temperature}")

    active_idx = active_mask_B.nonzero(as_tuple=False).flatten()
    logits_active = logits_BxV[active_idx].float().clone()
    for tid in suppressed_token_ids:
        logits_active[:, tid] = float("-inf")
    if forbidden_token_ids is not None:
        forbidden_token_ids = forbidden_token_ids.to(device=logits_active.device, dtype=torch.long)
        if forbidden_token_ids.numel() > 0:
            logits_active[:, forbidden_token_ids] = float("-inf")

    if do_sample:
        logits_active = logits_active / float(temperature)
        if top_k is not None:
            top_k = min(int(top_k), logits_active.size(-1))
            topk_values = torch.topk(logits_active, top_k, dim=-1).values[..., -1]
            logits_active = logits_active.masked_fill(logits_active < topk_values.unsqueeze(-1), float("-inf"))
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits_active, dim=-1, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_remove = cumulative_probs > float(top_p)
            sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
            sorted_remove[..., 0] = False
            remove_mask = torch.zeros_like(logits_active, dtype=torch.bool)
            remove_mask.scatter_(1, sorted_indices, sorted_remove)
            logits_active = logits_active.masked_fill(remove_mask, float("-inf"))
        probs = F.softmax(logits_active, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(1)
    else:
        sampled = logits_active.argmax(dim=-1)

    next_token_B[active_idx] = sampled.to(next_token_B.dtype)
    return next_token_B
