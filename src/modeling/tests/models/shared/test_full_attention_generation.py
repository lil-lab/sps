from __future__ import annotations

import torch
import pytest

from modeling.tests._helpers import make_full_model


def _make_generation_model():
    torch.manual_seed(123)
    return make_full_model(
        block_size=12,
        vocab_size=32,
        eos_token_id=30,
        pad_token_id=31,
        use_triton_full_attention=False,
    )


def test_full_attention_cached_decode_matches_dense_after_prompt_eos() -> None:
    model = _make_generation_model()
    eos = model.config.eos_token_id
    prompt = torch.tensor([[1, eos, 2]], dtype=torch.long)
    next_token = torch.tensor([[4]], dtype=torch.long)

    with torch.no_grad():
        _, past_key_values = model._forward_generation_hidden_states(prompt)
        cached_documents = model.generate_document_idx(prompt)
        current_documents = cached_documents[:, -1:]
        cached_hidden, _ = model._forward_generation_hidden_states(
            next_token,
            past_key_values_Lx2=past_key_values,
            cache_lengths_B=torch.tensor([prompt.size(1)], dtype=torch.long),
            cached_documents_idx_BxK=cached_documents,
            current_documents_idx_BxT=current_documents,
        )
        cached_logits = model.lm_head(cached_hidden)[:, -1, :]

        dense_sequence = torch.cat([prompt, next_token], dim=1)
        dense_hidden, _ = model._forward_generation_hidden_states(dense_sequence)
        dense_logits = model.lm_head(dense_hidden)[:, -1, :]

    torch.testing.assert_close(cached_logits, dense_logits, atol=1e-5, rtol=1e-5)


def test_full_attention_generation_advances_document_after_generated_eos(monkeypatch) -> None:
    model = _make_generation_model()
    eos = model.config.eos_token_id
    forced_tokens = [eos, 4, 5]
    recorded_current_documents: list[torch.Tensor] = []
    original_decode_one_token = model._decode_one_token_step

    def _recording_decode_one_token(state, token_B, active_mask_B):
        if bool(active_mask_B.any()):
            recorded_current_documents.append(state.current_documents_idx_B.detach().cpu().clone())
        return original_decode_one_token(state, token_B, active_mask_B)

    def _forced_sample(logits_BxV, active_mask_B, **kwargs):
        del active_mask_B, kwargs
        return torch.tensor([forced_tokens.pop(0)], device=logits_BxV.device, dtype=torch.long)

    monkeypatch.setattr(model, "_decode_one_token_step", _recording_decode_one_token)
    monkeypatch.setattr(model, "_sample_next_token", _forced_sample)

    prompt = torch.tensor([[1, 2]], dtype=torch.long)
    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=3, do_sample=False, stop_on_eos=False)

    assert generated.tolist() == [[1, 2, eos, 4, 5]]
    assert [int(doc.item()) for doc in recorded_current_documents[-2:]] == [0, 1]


def test_full_attention_generate_uses_batched_prefill_before_decode(monkeypatch) -> None:
    model = _make_generation_model()
    decode_calls = 0
    original_decode_one_token = model._decode_one_token_step

    def _counting_decode_one_token(state, token_B, active_mask_B):
        nonlocal decode_calls
        decode_calls += 1
        return original_decode_one_token(state, token_B, active_mask_B)

    monkeypatch.setattr(model, "_decode_one_token_step", _counting_decode_one_token)

    prompt = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.no_grad():
        _ = model.generate(prompt, max_new_tokens=3, do_sample=False, stop_on_eos=False)

    assert decode_calls == 2


def test_full_attention_generate_matches_dense_greedy_with_padding_and_eos() -> None:
    torch.manual_seed(789)
    model = _make_generation_model()
    pad = model.config.pad_token_id
    eos = model.config.eos_token_id
    prompts = torch.tensor(
        [
            [pad, pad, 4, eos, 5, 6],
            [7, 8, 9, 10, eos, 12],
        ],
        dtype=torch.long,
    )

    def dense_greedy(prompt_BxT: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        generated = prompt_BxT.clone()
        for _ in range(max_new_tokens):
            hidden_BxTxC, _ = model._forward_generation_hidden_states(generated)
            next_token_B = model.lm_head(hidden_BxTxC)[:, -1, :].argmax(dim=-1)
            generated = torch.cat([generated, next_token_B[:, None]], dim=1)
        return generated

    with torch.no_grad():
        actual = model.generate(prompts, max_new_tokens=3, do_sample=False, stop_on_eos=False)
        expected = dense_greedy(prompts, max_new_tokens=3)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_full_attention_generate_left_padded_batch_matches_unpadded_rows() -> None:
    torch.manual_seed(456)
    model = _make_generation_model()
    pad = model.config.pad_token_id
    eos = model.config.eos_token_id
    prompts = torch.tensor(
        [
            [pad, pad, 4, eos, 5, 6],
            [7, 8, 9, 10, 11, 12],
        ],
        dtype=torch.long,
    )

    with torch.no_grad():
        batched = model.generate(prompts, max_new_tokens=3, do_sample=False)

    for row_idx, prompt in enumerate(prompts):
        first_real = int((prompt != pad).nonzero(as_tuple=False)[0].item())
        unpadded_prompt = prompt[first_real:].unsqueeze(0)
        with torch.no_grad():
            single = model.generate(unpadded_prompt, max_new_tokens=3, do_sample=False)
        assert torch.equal(
            batched[row_idx, :first_real],
            torch.full((first_real,), pad, dtype=prompt.dtype),
        )
        torch.testing.assert_close(batched[row_idx, first_real:], single[0], atol=0, rtol=0)


@pytest.mark.cuda
def test_full_attention_batched_generation_decode_matches_dense_on_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA generation parity test requires a GPU")

    torch.manual_seed(321)
    model = make_full_model(
        block_size=16,
        vocab_size=32,
        eos_token_id=30,
        pad_token_id=31,
        use_triton_full_attention=False,
    ).cuda()
    eos = model.config.eos_token_id
    idx = torch.tensor([[4, 5, 6, 7], [8, eos, 9, 10]], dtype=torch.long, device="cuda")

    with torch.no_grad():
        state, next_logits, mode = model._prefill_generation_state(
            idx,
            max_new_tokens=2,
            require_batched=True,
        )
        dense_hidden, _ = model._forward_generation_hidden_states(idx)
        expected_next_logits = model.lm_head(dense_hidden)[:, -1, : model.config.vocab_size]
        next_token = next_logits.argmax(dim=-1)
        decoded_logits = model._decode_generation_state(
            state,
            next_token,
            torch.ones(idx.size(0), device="cuda", dtype=torch.bool),
        )
        appended = torch.cat([idx, next_token[:, None]], dim=1)
        expected_hidden, _ = model._forward_generation_hidden_states(appended)
        expected_decoded_logits = model.lm_head(expected_hidden)[:, -1, : model.config.vocab_size]

    assert mode == "batched_prefill"
    torch.testing.assert_close(next_logits, expected_next_logits, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(decoded_logits, expected_decoded_logits, atol=2e-2, rtol=2e-2)
