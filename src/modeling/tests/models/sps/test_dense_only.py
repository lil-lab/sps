from __future__ import annotations

import types

import pytest
import torch

from modeling.models.reverse_sps import ReverseSPSConfig, ReverseSPSModel
from modeling.models.sps import SPSConfig, SPSModel
from modeling.models.delayed_state import DelayedStateConfig, DelayedStateModel


_CUDA_DENSE_MODEL_KWARGS = dict(hidden_size=128, intermediate_size=256)


def _make_sps_model(
    *,
    enable_triton_attention: bool = False,
    hidden_size: int = 32,
    intermediate_size: int = 96,
) -> SPSModel:
    config = SPSConfig(
        block_size=32,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dropout=0.0,
        bias=False,
        eos_token_id=63,
        pad_token_id=62,
        predict_token_id=61,
        window_size=2,
        enable_triton_attention=enable_triton_attention,
        warp_specialize=False,
    )
    model = SPSModel(config)
    model.eval()
    return model


def _make_reverse_sps_model(
    *,
    hidden_size: int = 32,
    intermediate_size: int = 96,
) -> ReverseSPSModel:
    config = ReverseSPSConfig(
        block_size=32,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dropout=0.0,
        bias=False,
        eos_token_id=63,
        pad_token_id=62,
        predict_token_id=61,
        window_size=2,
        enable_triton_attention=False,
        warp_specialize=False,
    )
    model = ReverseSPSModel(config)
    model.eval()
    return model


def _make_delayed_state_model(
    *,
    hidden_size: int = 32,
    intermediate_size: int = 96,
) -> DelayedStateModel:
    config = DelayedStateConfig(
        block_size=32,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        dropout=0.0,
        bias=False,
        eos_token_id=63,
        pad_token_id=62,
        predict_token_id=61,
        window_size=2,
        enable_triton_attention=False,
        warp_specialize=False,
    )
    model = DelayedStateModel(config)
    model.eval()
    return model


def _zero_transformer_blocks(model: SPSModel) -> None:
    with torch.no_grad():
        for block in model.transformer.h:
            for param in block.parameters():
                param.zero_()


def _manual_slot_logits(
    model: SPSModel,
    slot_token_ids: torch.Tensor,
) -> torch.Tensor:
    slot_emb = model.transformer.wte.weight[slot_token_ids]
    slot_hidden = model.transformer.output_norm(slot_emb)
    return model.lm_head(slot_hidden)


def _assert_removed_stats_absent(stats: dict[str, torch.Tensor]) -> None:
    for key in (
        "decision_rate",
        "decision_rate_soft",
        "decision_rate_hard",
        "predict_rate",
        "predict_rate_hard",
        "predict_rate_mode",
        "predict_loss_count",
        "predict_count",
        "alpha_variance",
        "alpha_variance_soft",
        "alpha_variance_hard",
        "alpha_soft_distance",
        "alpha_uncertainty",
    ):
        assert key not in stats
    assert all(not key.startswith("decision_rate_layer_") for key in stats)
    assert all(not key.startswith("predict_interval_") for key in stats)


def test_sps_forward_reads_odd_slot_logits() -> None:
    model = _make_sps_model()
    idx = torch.tensor([[4, 5]], dtype=torch.long)
    x_Bx2T = torch.randn(1, 4, model.config.hidden_size)

    def _fake_forward_hidden_states(self, idx_Bx2T, *, documents_idx_Bx2T):
        del self, idx_Bx2T, documents_idx_Bx2T
        return x_Bx2T

    model.forward_hidden_states = types.MethodType(_fake_forward_hidden_states, model)

    with torch.no_grad():
        logits = model(idx)

    expected = model.lm_head(x_Bx2T[:, 1::2])
    torch.testing.assert_close(logits, expected)
    assert not torch.allclose(logits, model.lm_head(x_Bx2T[:, ::2]))


def test_sps_triton_symbol_imports_cleanly() -> None:
    from modeling.models.attention.triton_reverse_sps_flash_attention import ATTN_FWD_CONFIGS
    from modeling.models.sps.core import triton_sps_sliding_attention

    assert triton_sps_sliding_attention is not None
    assert ATTN_FWD_CONFIGS


def test_sps_stats_omit_stochastic_fields() -> None:
    model = _make_sps_model()
    idx = torch.tensor([[4, 5]], dtype=torch.long)
    x_Bx2T = torch.randn(1, 4, model.config.hidden_size)

    def _fake_forward_hidden_states(self, idx_Bx2T, *, documents_idx_Bx2T):
        del self, idx_Bx2T, documents_idx_Bx2T
        return x_Bx2T

    model.forward_hidden_states = types.MethodType(_fake_forward_hidden_states, model)

    with torch.no_grad():
        _, _, stats = model(idx, idx.clone())

    assert set(stats) >= {"token_nll_sum", "token_nll_count", "token_count"}
    _assert_removed_stats_absent(stats)


def test_sps_advertises_forward_efficient() -> None:
    model = _make_sps_model()

    assert hasattr(model, "forward_efficient")


def test_sps_advertises_generate() -> None:
    model = _make_sps_model()

    assert hasattr(model, "generate")


@pytest.mark.cuda
def test_sps_forward_efficient_reads_odd_slot_logits() -> None:
    model = _make_sps_model().cuda()
    _zero_transformer_blocks(model)
    with torch.no_grad():
        model.transformer.wte.weight.zero_()
        model.transformer.wte.weight[4, :4] = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
        model.transformer.wte.weight[5, :4] = torch.tensor([2.0, 1.0, 0.5, 3.0], device="cuda")
        model.transformer.wte.weight[model.config.predict_token_id, :4] = torch.tensor([4.0, 3.0, 2.0, 1.0], device="cuda")

    idx = torch.tensor([[4, 5]], dtype=torch.long, device="cuda")
    expected = _manual_slot_logits(
        model,
        torch.full_like(idx, model.config.predict_token_id),
    )

    with torch.no_grad():
        logits = model.forward_efficient(idx)

    torch.testing.assert_close(logits, expected, atol=1e-5, rtol=1e-5)


def _greedy_next_token(model: SPSModel, idx_BxT: torch.Tensor) -> torch.Tensor:
    logits = model.forward_efficient(idx_BxT)
    next_logits = logits[:, -1, :].clone()
    next_logits[:, model.config.predict_token_id] = float("-inf")
    next_logits[:, model.config.pad_token_id] = float("-inf")
    return next_logits.argmax(dim=-1, keepdim=True).to(idx_BxT.dtype)


def _greedy_from_logits(model: SPSModel, next_logits: torch.Tensor) -> torch.Tensor:
    next_logits = next_logits.clone()
    next_logits[:, model.config.predict_token_id] = float("-inf")
    next_logits[:, model.config.pad_token_id] = float("-inf")
    return next_logits.argmax(dim=-1).to(dtype=torch.long)


def test_sps_generate_greedy_matches_stepwise_forward_efficient() -> None:
    from modeling.models.sps.core import triton_sps_sliding_attention

    if (not torch.cuda.is_available()) or triton_sps_sliding_attention is None:
        pytest.skip("SPS public generation requires CUDA + Triton")

    torch.manual_seed(0)
    model = _make_sps_model(
        enable_triton_attention=True,
        **_CUDA_DENSE_MODEL_KWARGS,
    ).cuda()
    eos = model.config.eos_token_id
    prompt = torch.tensor([[5, eos, 7, 8]], dtype=torch.long, device="cuda")

    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=3, do_sample=False)

    expected = prompt.clone()
    with torch.no_grad():
        for _ in range(3):
            expected = torch.cat([expected, _greedy_next_token(model, expected)], dim=1)

    torch.testing.assert_close(generated, expected, atol=0, rtol=0)


def test_sps_generate_left_padded_batch_matches_unpadded_rows() -> None:
    from modeling.models.sps.core import triton_sps_sliding_attention

    if (not torch.cuda.is_available()) or triton_sps_sliding_attention is None:
        pytest.skip("SPS public generation requires CUDA + Triton")

    torch.manual_seed(1)
    model = _make_sps_model(
        enable_triton_attention=True,
        **_CUDA_DENSE_MODEL_KWARGS,
    ).cuda()
    pad = model.config.pad_token_id
    eos = model.config.eos_token_id
    prompts = torch.tensor(
        [
            [pad, pad, 11, eos, 12, 13],
            [21, 22, 23, 24, 25, 26],
        ],
        dtype=torch.long,
        device="cuda",
    )

    with torch.no_grad():
        batched = model.generate(prompts, max_new_tokens=4, do_sample=False)

    for row_idx, prompt in enumerate(prompts):
        first_real = int((prompt != pad).nonzero(as_tuple=False)[0].item())
        unpadded_prompt = prompt[first_real:].unsqueeze(0)
        with torch.no_grad():
            single = model.generate(unpadded_prompt, max_new_tokens=4, do_sample=False)
        assert torch.equal(
            batched[row_idx, :first_real],
            torch.full((first_real,), pad, dtype=prompt.dtype, device=prompt.device),
        )
        torch.testing.assert_close(batched[row_idx, first_real:], single[0], atol=0, rtol=0)


def test_sps_and_reverse_sps_delayed_state_checkpoints_are_strictly_compatible() -> None:
    sps = _make_sps_model()
    reverse_sps = _make_reverse_sps_model()
    delayed_state = _make_delayed_state_model()

    sps_from_reverse_sps = sps.load_state_dict(reverse_sps.state_dict(), strict=True)
    reverse_sps_from_sps = reverse_sps.load_state_dict(sps.state_dict(), strict=True)
    sps_from_delayed_state = sps.load_state_dict(delayed_state.state_dict(), strict=True)
    delayed_state_from_sps = delayed_state.load_state_dict(sps.state_dict(), strict=True)

    assert sps_from_reverse_sps.missing_keys == []
    assert sps_from_reverse_sps.unexpected_keys == []
    assert reverse_sps_from_sps.missing_keys == []
    assert reverse_sps_from_sps.unexpected_keys == []
    assert sps_from_delayed_state.missing_keys == []
    assert sps_from_delayed_state.unexpected_keys == []
    assert delayed_state_from_sps.missing_keys == []
    assert delayed_state_from_sps.unexpected_keys == []


@pytest.mark.cuda
def test_sps_dense_forward_reads_odd_slot_logits_on_cuda() -> None:
    from modeling.models.sps.core import triton_sps_sliding_attention

    if (not torch.cuda.is_available()) or triton_sps_sliding_attention is None:
        pytest.skip("dense SPS forward requires CUDA + Triton")

    model = _make_sps_model(
        enable_triton_attention=True,
        **_CUDA_DENSE_MODEL_KWARGS,
    ).cuda()
    _zero_transformer_blocks(model)
    with torch.no_grad():
        model.transformer.wte.weight.zero_()
        model.transformer.wte.weight[4, :4] = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
        model.transformer.wte.weight[5, :4] = torch.tensor([2.0, 1.0, 0.5, 3.0], device="cuda")
        model.transformer.wte.weight[model.config.predict_token_id, :4] = torch.tensor([4.0, 3.0, 2.0, 1.0], device="cuda")

    idx = torch.tensor([[4, 5]], dtype=torch.long, device="cuda")
    expected = _manual_slot_logits(
        model,
        torch.full_like(idx, model.config.predict_token_id),
    )

    with torch.no_grad():
        logits = model(idx)

    torch.testing.assert_close(logits, expected, atol=2e-2, rtol=2e-2)


@pytest.mark.cuda
def test_sps_dense_forward_matches_forward_efficient_on_cuda() -> None:
    from modeling.models.sps.core import triton_sps_sliding_attention

    if (not torch.cuda.is_available()) or triton_sps_sliding_attention is None:
        pytest.skip("dense SPS forward requires CUDA + Triton")

    torch.manual_seed(2)
    model = _make_sps_model(
        enable_triton_attention=True,
        **_CUDA_DENSE_MODEL_KWARGS,
    ).cuda()
    idx = torch.tensor([[4, 5, 6, 7], [8, model.config.eos_token_id, 9, 10]], dtype=torch.long, device="cuda")

    with torch.no_grad():
        dense_logits = model(idx)
        efficient_logits = model.forward_efficient(idx)

    torch.testing.assert_close(dense_logits, efficient_logits, atol=2e-2, rtol=2e-2)


@pytest.mark.cuda
def test_sps_batched_generation_prefill_matches_forward_efficient_on_cuda() -> None:
    from modeling.models.sps.core import triton_sps_sliding_attention

    if (not torch.cuda.is_available()) or triton_sps_sliding_attention is None:
        pytest.skip("batched SPS generation prefill requires CUDA + Triton")

    torch.manual_seed(4)
    model = _make_sps_model(
        enable_triton_attention=True,
        **_CUDA_DENSE_MODEL_KWARGS,
    ).cuda()
    idx = torch.tensor([[4, 5, 6, 7], [8, model.config.eos_token_id, 9, 10]], dtype=torch.long, device="cuda")

    with torch.no_grad():
        state, next_logits, mode = model._prefill_generation_state(
            idx,
            max_new_tokens=2,
            require_batched=True,
        )
        expected_next_logits = model.forward_efficient(idx)[:, -1, :]
        next_token = _greedy_from_logits(model, next_logits)
        decoded_logits = model._decode_generation_state(state, next_token, torch.ones(2, device="cuda", dtype=torch.bool))
        model._advance_decode_state(state, next_token, torch.ones(2, device="cuda", dtype=torch.bool))
        expected_decoded_logits = model.forward_efficient(torch.cat([idx, next_token[:, None]], dim=1))[:, -1, :]

    assert mode == "batched_prefill"
    torch.testing.assert_close(next_logits, expected_next_logits, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(decoded_logits, expected_decoded_logits, atol=2e-2, rtol=2e-2)
