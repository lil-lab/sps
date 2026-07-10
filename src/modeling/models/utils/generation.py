from __future__ import annotations

"""Shared autoregressive generation loop for step-decoder models."""

from typing import Any, Callable, Optional

import torch
from torch import Tensor

from modeling.models.model import infer_is_real_tokens, validate_left_padded_tokens


BuildStateFn = Callable[[int, int, torch.device], Any]
DecodeStepFn = Callable[[Any, Tensor, Tensor], Tensor]
AdvanceStateFn = Callable[[Any, Tensor, Tensor], None]
SampleFn = Callable[[Tensor, Tensor], Tensor]
PrefillFn = Callable[[Tensor, int], tuple[Any, Tensor, str]]


def generate_with_step_decoder(
    model,
    idx_BxT: Tensor,
    max_new_tokens: int,
    *,
    do_sample: bool,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    stop_on_eos: bool,
    forbidden_token_ids: Optional[Tensor],
    build_state: BuildStateFn,
    decode_one_token: DecodeStepFn,
    advance_state: AdvanceStateFn,
    sample_next_token: SampleFn,
    context: str,
) -> Tensor:
    """Generate from any model that exposes a single-token decode step.

    The model-specific decoder owns attention/cache semantics. This helper owns
    prompt validation, left-padded batching, sampling, EOS stopping, and output
    shaping so generation behavior is consistent across model families.
    """
    del do_sample, temperature, top_k, top_p, forbidden_token_ids

    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
    if max_new_tokens == 0:
        return idx_BxT.clone()

    device = idx_BxT.device
    b, t = idx_BxT.size()
    is_real_BxT = infer_is_real_tokens(idx_BxT, model.config.pad_token_id)
    validate_left_padded_tokens(
        is_real_BxT,
        allow_all_pad=False,
        context=f"{context} generation prompts",
    )

    max_real_prompt_tokens = int(is_real_BxT.sum(dim=1).max().item())
    total_real_tokens = max_real_prompt_tokens + int(max_new_tokens)
    if total_real_tokens > model.freqs_cis.shape[0]:
        raise ValueError(
            f"Cannot generate {max_new_tokens} new tokens from a prompt with "
            f"{max_real_prompt_tokens} real tokens when block size is {model.freqs_cis.shape[0]}"
        )

    state = build_state(b, total_real_tokens, device)
    last_logits_BxV = model.lm_head.weight.new_zeros((b, model.config.vocab_size))
    for col in range(t):
        active_col_B = is_real_BxT[:, col]
        logits_step_BxV = decode_one_token(state, idx_BxT[:, col], active_col_B)
        last_logits_BxV = torch.where(
            active_col_B.unsqueeze(1),
            logits_step_BxV,
            last_logits_BxV,
        )
        advance_state(state, idx_BxT[:, col], active_col_B)

    generated_BxT = torch.full(
        (b, max_new_tokens),
        model.config.pad_token_id,
        device=device,
        dtype=idx_BxT.dtype,
    )
    finished_B = torch.zeros((b,), device=device, dtype=torch.bool)
    next_logits_BxV = last_logits_BxV

    for step in range(max_new_tokens):
        sample_mask_B = ~finished_B
        if not bool(sample_mask_B.any()):
            break

        next_token_B = sample_next_token(next_logits_BxV, sample_mask_B).to(idx_BxT.dtype)
        generated_BxT[:, step] = torch.where(
            sample_mask_B,
            next_token_B,
            generated_BxT[:, step],
        )

        newly_finished_B = sample_mask_B & stop_on_eos & (
            next_token_B == model.config.eos_token_id
        )
        decode_active_B = sample_mask_B & ~newly_finished_B
        if bool(decode_active_B.any()):
            next_logits_BxV_step = decode_one_token(state, next_token_B, decode_active_B)
            next_logits_BxV = torch.where(
                decode_active_B.unsqueeze(1),
                next_logits_BxV_step,
                next_logits_BxV,
            )
            advance_state(state, next_token_B, decode_active_B)

        finished_B = finished_B | newly_finished_B

    return torch.cat([idx_BxT, generated_BxT], dim=1)


def generate_with_batched_prefill(
    model,
    idx_BxT: Tensor,
    max_new_tokens: int,
    *,
    do_sample: bool,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    stop_on_eos: bool,
    forbidden_token_ids: Optional[Tensor],
    prefill_prompt: PrefillFn,
    decode_one_token: DecodeStepFn,
    advance_state: AdvanceStateFn,
    sample_next_token: SampleFn,
    context: str,
) -> Tensor:
    """Generate with a batched prompt prefill followed by cached decode steps."""
    del do_sample, temperature, top_k, top_p, forbidden_token_ids

    if max_new_tokens < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {max_new_tokens}")
    if max_new_tokens == 0:
        return idx_BxT.clone()

    device = idx_BxT.device
    b = idx_BxT.size(0)
    is_real_BxT = infer_is_real_tokens(idx_BxT, model.config.pad_token_id)
    validate_left_padded_tokens(
        is_real_BxT,
        allow_all_pad=False,
        context=f"{context} generation prompts",
    )

    max_real_prompt_tokens = int(is_real_BxT.sum(dim=1).max().item())
    total_real_tokens = max_real_prompt_tokens + int(max_new_tokens)
    if total_real_tokens > model.freqs_cis.shape[0]:
        raise ValueError(
            f"Cannot generate {max_new_tokens} new tokens from a prompt with "
            f"{max_real_prompt_tokens} real tokens when block size is {model.freqs_cis.shape[0]}"
        )

    state, next_logits_BxV, _prefill_mode = prefill_prompt(idx_BxT, max_new_tokens)

    generated_BxT = torch.full(
        (b, max_new_tokens),
        model.config.pad_token_id,
        device=device,
        dtype=idx_BxT.dtype,
    )
    finished_B = torch.zeros((b,), device=device, dtype=torch.bool)

    for step in range(max_new_tokens):
        sample_mask_B = ~finished_B
        if not bool(sample_mask_B.any()):
            break

        next_token_B = sample_next_token(next_logits_BxV, sample_mask_B).to(idx_BxT.dtype)
        generated_BxT[:, step] = torch.where(
            sample_mask_B,
            next_token_B,
            generated_BxT[:, step],
        )

        newly_finished_B = sample_mask_B & stop_on_eos & (
            next_token_B == model.config.eos_token_id
        )
        decode_active_B = sample_mask_B & ~newly_finished_B
        if step < max_new_tokens - 1 and bool(decode_active_B.any()):
            next_logits_BxV_step = decode_one_token(state, next_token_B, decode_active_B)
            next_logits_BxV = torch.where(
                decode_active_B.unsqueeze(1),
                next_logits_BxV_step,
                next_logits_BxV,
            )
            advance_state(state, next_token_B, decode_active_B)

        finished_B = finished_B | newly_finished_B

    return torch.cat([idx_BxT, generated_BxT], dim=1)
