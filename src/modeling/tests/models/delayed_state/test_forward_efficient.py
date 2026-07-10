from __future__ import annotations

import pytest
import torch

from modeling.models.reverse_sps import ReverseSPSConfig, ReverseSPSModel
from modeling.models.reverse_sps.core import triton_reverse_sps_sliding_attention
from modeling.models.delayed_state import DelayedStateConfig, DelayedStateModel


_CUDA_DENSE_MODEL_KWARGS = dict(hidden_size=128, intermediate_size=256)
_REMOVED_STATS_KEYS = (
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
)


def _make_delayed_state_model(
    *,
    enable_triton_attention: bool = False,
    hidden_size: int = 32,
    intermediate_size: int = 96,
    window_size: int = 2,
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
        window_size=window_size,
        enable_triton_attention=enable_triton_attention,
        warp_specialize=False,
    )
    model = DelayedStateModel(config)
    model.eval()
    return model


def _make_reverse_sps_model(
    *,
    enable_triton_attention: bool = False,
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
        enable_triton_attention=enable_triton_attention,
        warp_specialize=False,
    )
    model = ReverseSPSModel(config)
    model.eval()
    return model


def _make_matching_models(
    *,
    enable_triton_attention: bool = False,
    hidden_size: int = 32,
    intermediate_size: int = 96,
) -> tuple[ReverseSPSModel, DelayedStateModel]:
    torch.manual_seed(0)
    reverse_sps = _make_reverse_sps_model(
        enable_triton_attention=enable_triton_attention,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    torch.manual_seed(1)
    delayed_state = _make_delayed_state_model(
        enable_triton_attention=enable_triton_attention,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    load_result = delayed_state.load_state_dict(reverse_sps.state_dict(), strict=True)
    assert load_result.missing_keys == []
    assert load_result.unexpected_keys == []
    return reverse_sps, delayed_state


def _zero_transformer_blocks(model: ReverseSPSModel | DelayedStateModel) -> None:
    with torch.no_grad():
        for block in model.transformer.h:
            for param in block.parameters():
                param.zero_()


def _manual_slot_logits(
    model: ReverseSPSModel | DelayedStateModel,
    token_ids: torch.Tensor,
) -> torch.Tensor:
    slot_emb = model.transformer.wte.weight[token_ids]
    slot_hidden = model.transformer.output_norm(slot_emb)
    return model.lm_head(slot_hidden)


def _assert_removed_stats_absent(stats: dict[str, torch.Tensor]) -> None:
    for key in _REMOVED_STATS_KEYS:
        assert key not in stats
    assert all(not key.startswith("decision_rate_layer_") for key in stats)
    assert all(not key.startswith("predict_interval_") for key in stats)


@pytest.mark.cuda
def test_delayed_state_forward_efficient_reads_predict_slot_logits() -> None:
    reverse_sps, delayed_state = _make_matching_models()
    reverse_sps = reverse_sps.cuda()
    delayed_state = delayed_state.cuda()
    _zero_transformer_blocks(reverse_sps)
    _zero_transformer_blocks(delayed_state)

    with torch.no_grad():
        delayed_state.transformer.wte.weight.zero_()
        delayed_state.transformer.wte.weight[4, :4] = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
        delayed_state.transformer.wte.weight[5, :4] = torch.tensor([2.0, 1.0, 0.5, 3.0], device="cuda")
        delayed_state.transformer.wte.weight[delayed_state.config.predict_token_id, :4] = torch.tensor([4.0, 3.0, 2.0, 1.0], device="cuda")
        reverse_sps.transformer.wte.weight.copy_(delayed_state.transformer.wte.weight)

    idx = torch.tensor([[4, 5]], dtype=torch.long, device="cuda")

    with torch.no_grad():
        delayed_state_logits = delayed_state.forward_efficient(idx)
        reverse_sps_logits = reverse_sps.forward_efficient(idx)

    expected_delayed_state = _manual_slot_logits(
        delayed_state,
        torch.full_like(idx, delayed_state.config.predict_token_id),
    )
    expected_reverse_sps = _manual_slot_logits(reverse_sps, idx)

    torch.testing.assert_close(delayed_state_logits, expected_delayed_state, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(reverse_sps_logits, expected_reverse_sps, atol=1e-5, rtol=1e-5)
    assert not torch.allclose(delayed_state_logits, reverse_sps_logits)


@pytest.mark.cuda
def test_delayed_state_forward_efficient_left_padding_is_invariant() -> None:
    torch.manual_seed(2)
    model = _make_delayed_state_model().cuda()
    pad = model.config.pad_token_id
    eos = model.config.eos_token_id
    unpadded = torch.tensor([[4, eos, 6, 7]], dtype=torch.long, device="cuda")
    padded = torch.tensor([[pad, pad, 4, eos, 6, 7]], dtype=torch.long, device="cuda")

    with torch.no_grad():
        logits_unpadded = model.forward_efficient(unpadded)
        logits_padded = model.forward_efficient(padded)

    torch.testing.assert_close(logits_unpadded, logits_padded[:, -unpadded.size(1) :, :], atol=2e-2, rtol=2e-2)


@pytest.mark.cuda
def test_delayed_state_stats_omit_stochastic_fields() -> None:
    model = _make_delayed_state_model().cuda()
    idx = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device="cuda")

    with torch.no_grad():
        _, _, stats = model.forward_efficient(idx, idx.clone())

    assert set(stats) >= {"token_nll_sum", "token_nll_count", "token_count"}
    _assert_removed_stats_absent(stats)


def test_delayed_state_generate_runs() -> None:
    if (not torch.cuda.is_available()) or triton_reverse_sps_sliding_attention is None:
        pytest.skip("Delayed State public generation requires CUDA + Triton")

    torch.manual_seed(3)
    model = _make_delayed_state_model(
        enable_triton_attention=True,
        **_CUDA_DENSE_MODEL_KWARGS,
    ).cuda()
    prompt = torch.tensor([[5, 6, 7, 8]], dtype=torch.long, device="cuda")

    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=3, do_sample=False)

    assert generated.shape == (1, 7)
    torch.testing.assert_close(generated[:, : prompt.size(1)], prompt)


def _greedy_from_logits(model: DelayedStateModel, next_logits: torch.Tensor) -> torch.Tensor:
    next_logits = next_logits.clone()
    next_logits[:, model.config.predict_token_id] = float("-inf")
    next_logits[:, model.config.pad_token_id] = float("-inf")
    return next_logits.argmax(dim=-1).to(dtype=torch.long)


@pytest.mark.cuda
@pytest.mark.parametrize("window_size", [0, 2])
def test_delayed_state_decode_predict_attention_sees_current_normal_before_window_update(
    window_size: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _make_delayed_state_model(window_size=window_size).cuda()
    state = model._build_decode_state(
        batch_size=1,
        max_tokens=4,
        device=torch.device("cuda"),
        attn_dtype=model.transformer.wte.weight.dtype,
    )
    calls: list[dict[str, object]] = []

    def _fake_attend(
        state,
        layer_idx,
        q_BxHxD,
        *,
        curr_normal_k_BxHxD=None,
        curr_normal_v_BxHxD=None,
        curr_predict_k_BxHxD=None,
        curr_predict_v_BxHxD=None,
        check_finite=False,
    ):
        del check_finite
        calls.append(
            {
                "layer_idx": layer_idx,
                "has_current_normal": curr_normal_k_BxHxD is not None
                and curr_normal_v_BxHxD is not None,
                "has_current_predict": curr_predict_k_BxHxD is not None
                and curr_predict_v_BxHxD is not None,
                "normal_window_len": int(state.normal_window.len[layer_idx][0].item()),
            }
        )
        return torch.zeros_like(q_BxHxD, dtype=torch.float32)

    monkeypatch.setattr(model, "_attend_from_decode_state", _fake_attend)

    with torch.no_grad():
        model._decode_one_token_step(
            state,
            torch.tensor([5], dtype=torch.long, device="cuda"),
            torch.tensor([True], device="cuda"),
            model.freqs_cis,
        )
        state.processed_tokens_B += 1
        state.next_doc_start_B.fill_(False)
        model._decode_one_token_step(
            state,
            torch.tensor([6], dtype=torch.long, device="cuda"),
            torch.tensor([True], device="cuda"),
            model.freqs_cis,
        )

    predict_calls = [call for call in calls if call["has_current_predict"]]
    assert len(predict_calls) == 2 * model.config.n_layer
    assert all(call["has_current_normal"] for call in predict_calls)
    if window_size > 0:
        assert [call["normal_window_len"] for call in predict_calls] == [0, 0, 1, 1]


def test_delayed_state_generation_window_cache_wraps_without_shifting() -> None:
    model = _make_delayed_state_model(window_size=2)
    state = model._build_decode_state(
        batch_size=1,
        max_tokens=5,
        device=torch.device("cpu"),
        attn_dtype=model.transformer.wte.weight.dtype,
    )
    n_head = model.config.n_head
    head_dim = model.config.hidden_size // model.config.n_head
    active = torch.tensor([True])

    def _token_kv(value: float) -> tuple[torch.Tensor, torch.Tensor]:
        k = torch.full((1, n_head, head_dim), value)
        v = torch.full((1, n_head, head_dim), value + 10.0)
        return k, v

    # Exercise the pure-PyTorch ring-buffer reference directly. The public decode
    # path fuses this window update into a single Triton (CUDA-only) kernel via
    # `_update_predict_memory`; `_update_normal_memory_unfused` is the equivalent
    # fallback and lets us unit-test the wrap semantics on CPU.
    for value in (1.0, 2.0, 3.0):
        k, v = _token_kv(value)
        model._update_normal_memory_unfused(state, 0, k, v, active)

    assert int(state.normal_window.len[0][0].item()) == 2
    assert int(state.normal_window.pos[0][0].item()) == 1
    observed = state.normal_window.k[0][0, 0, :, 0]
    torch.testing.assert_close(observed, torch.tensor([3.0, 2.0]))

    model._reset_layer_memory(state, 0, torch.tensor([True]))
    assert int(state.normal_window.len[0][0].item()) == 0
    assert int(state.normal_window.pos[0][0].item()) == 0


def test_delayed_state_and_reverse_sps_checkpoints_are_strictly_compatible() -> None:
    delayed_state = _make_delayed_state_model()
    reverse_sps = _make_reverse_sps_model()

    delayed_state_load = delayed_state.load_state_dict(reverse_sps.state_dict(), strict=True)
    reverse_sps_load = reverse_sps.load_state_dict(delayed_state.state_dict(), strict=True)

    assert delayed_state_load.missing_keys == []
    assert delayed_state_load.unexpected_keys == []
    assert reverse_sps_load.missing_keys == []
    assert reverse_sps_load.unexpected_keys == []


@pytest.mark.cuda
def test_delayed_state_dense_forward_reads_predict_slot_logits_on_cuda() -> None:
    if (not torch.cuda.is_available()) or triton_reverse_sps_sliding_attention is None:
        pytest.skip("dense Delayed State forward requires CUDA + Triton")

    model = _make_delayed_state_model(
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
def test_delayed_state_batched_generation_prefill_matches_forward_efficient_on_cuda() -> None:
    if (not torch.cuda.is_available()) or triton_reverse_sps_sliding_attention is None:
        pytest.skip("batched Delayed State generation prefill requires CUDA + Triton")

    torch.manual_seed(4)
    model = _make_delayed_state_model(
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
        active = torch.ones(2, device="cuda", dtype=torch.bool)
        decoded_logits = model._decode_generation_state(state, next_token, active)
        model._advance_decode_state(state, next_token, active)
        expected_decoded_logits = model.forward_efficient(torch.cat([idx, next_token[:, None]], dim=1))[:, -1, :]

    assert mode == "batched_prefill"
    torch.testing.assert_close(next_logits, expected_next_logits, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(decoded_logits, expected_decoded_logits, atol=2e-2, rtol=2e-2)
