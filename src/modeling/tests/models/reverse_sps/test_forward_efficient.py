from __future__ import annotations

import pytest
import torch

from modeling.models.reverse_sps import ReverseSPSConfig, ReverseSPSModel
from modeling.models.reverse_sps.core import triton_reverse_sps_sliding_attention
from modeling.tests._sliding_window_reference import window_score_mod_factory


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


def _assert_removed_stats_absent(stats: dict[str, torch.Tensor]) -> None:
    for key in _REMOVED_STATS_KEYS:
        assert key not in stats
    assert all(not key.startswith("decision_rate_layer_") for key in stats)
    assert all(not key.startswith("predict_interval_") for key in stats)


@pytest.mark.cuda
def test_forward_efficient_left_padding_is_invariant():
    torch.manual_seed(2)
    model = _make_reverse_sps_model().cuda()
    pad = model.config.pad_token_id
    eos = model.config.eos_token_id
    unpadded = torch.tensor([[4, eos, 6, 7]], dtype=torch.long, device="cuda")
    padded = torch.tensor([[pad, pad, 4, eos, 6, 7]], dtype=torch.long, device="cuda")

    with torch.no_grad():
        logits_unpadded = model.forward_efficient(unpadded)
        logits_padded = model.forward_efficient(padded)

    torch.testing.assert_close(logits_unpadded, logits_padded[:, -unpadded.size(1) :, :], atol=2e-2, rtol=2e-2)


@pytest.mark.cuda
def test_reverse_sps_stats_omit_stochastic_and_efficiency_fields():
    model = _make_reverse_sps_model().cuda()
    idx = torch.tensor([[1, 2, 3, 4]], dtype=torch.long, device="cuda")

    with torch.no_grad():
        _, _, stats = model.forward_efficient(idx, idx.clone())

    assert set(stats) >= {"token_nll_sum", "token_nll_count", "token_count"}
    assert not hasattr(model.config, "target_memory_access_rate")
    assert "efficiency_loss_term" not in stats
    assert "efficiency_excess" not in stats
    assert "bimodal_penalty" not in stats
    assert "bimodal_loss_term" not in stats
    assert "decision_head_bias" not in stats
    _assert_removed_stats_absent(stats)


def test_forward_efficient_does_not_return_intermediates() -> None:
    model = _make_reverse_sps_model()
    idx = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    with pytest.raises(TypeError):
        model.forward_efficient(idx, return_intermediates=True)


def test_reverse_sps_config_does_not_expose_fixed_compat_properties() -> None:
    config = _make_reverse_sps_model().config

    assert not hasattr(config, "can_see_since_last_predict")
    assert not hasattr(config, "apply_minimum_window_normals")


def test_generate_runs() -> None:
    if (not torch.cuda.is_available()) or triton_reverse_sps_sliding_attention is None:
        pytest.skip("Reverse SPS public generation requires CUDA + Triton")

    torch.manual_seed(3)
    model = _make_reverse_sps_model(
        enable_triton_attention=True,
        **_CUDA_DENSE_MODEL_KWARGS,
    ).cuda()
    prompt = torch.tensor([[5, 6, 7, 8]], dtype=torch.long, device="cuda")

    with torch.no_grad():
        generated = model.generate(prompt, max_new_tokens=3, do_sample=False)

    assert generated.shape == (1, 7)
    torch.testing.assert_close(generated[:, : prompt.size(1)], prompt)


@pytest.mark.parametrize("window_size", [0, 1, 2])
def test_reverse_sps_dense_visibility_matches_window_definition(window_size: int):
    t = 5
    score_mod = window_score_mod_factory(window_size=window_size)

    for token_idx in range(t):
        expected_normal_tokens = list(range(max(0, token_idx - window_size), token_idx + 1))
        for query_kind, q_idx in [("normal", 2 * token_idx), ("predict", 2 * token_idx + 1)]:
            visible_normal_tokens = []
            for k_idx in range(0, 2 * t, 2):
                if k_idx > q_idx:
                    continue
                score = score_mod(
                    torch.tensor(0.0),
                    torch.tensor(0),
                    torch.tensor(0),
                    torch.tensor(q_idx),
                    torch.tensor(k_idx),
                )
                if torch.isfinite(score):
                    visible_normal_tokens.append(k_idx // 2)

            assert visible_normal_tokens == expected_normal_tokens, (
                f"{query_kind} query at token {token_idx} should see normal tokens "
                f"{expected_normal_tokens} for window_size={window_size}, "
                f"got {visible_normal_tokens}"
            )


@pytest.mark.cuda
def test_dense_forward_matches_forward_efficient_on_cuda():
    if (not torch.cuda.is_available()) or triton_reverse_sps_sliding_attention is None:
        pytest.skip("dense Reverse SPS forward requires CUDA + Triton")

    model = _make_reverse_sps_model(
        enable_triton_attention=True,
        **_CUDA_DENSE_MODEL_KWARGS,
    ).cuda()
    idx = torch.tensor([[4, 5, 6, 7]], dtype=torch.long, device="cuda")
    targets = idx.clone()

    with torch.no_grad():
        dense_logits, dense_loss, _ = model(idx, targets)
        efficient_logits, efficient_loss, _ = model.forward_efficient(idx, targets)

    torch.testing.assert_close(dense_logits, efficient_logits, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(dense_loss, efficient_loss, atol=2e-2, rtol=2e-2)
