import pytest
import torch

from modeling.models.full_attention_model import Model, ModelConfig


def _make_config(*, use_triton_full_attention: bool) -> ModelConfig:
    # Constraints imposed by triton_full_flash_attention's pytest single-
    # config path (kernel uses `if "PYTEST_VERSION" in os.environ:` to
    # collapse autotune to ONE config: `BLOCK_M=128, BLOCK_N=64`):
    #   1. head_dim must be in {16, 32, 64, 128, 256} (kernel asserts).
    #   2. head_dim must be >= 64 (BLOCK_N=64 in the pytest config; the kernel
    #      then `tl.static_assert(BLOCK_N <= HEAD_DIM)`).
    #   3. N_CTX (the kernel's sequence length) must be >= 128 (BLOCK_M=128 in
    #      the pytest config; with N_CTX < BLOCK_M the kernel attempts to
    #      compile that single config anyway and trips the same static assert).
    # hidden_size=128 / n_head=2 -> head_dim=64 (valid).
    # block_size=256 leaves room for a length-128 prompt in the tests.
    return ModelConfig(
        block_size=256,
        vocab_size=64,
        n_layer=2,
        n_head=2,
        hidden_size=128,
        intermediate_size=384,
        dropout=0.0,
        bias=False,
        eos_token_id=63,
        pad_token_id=0,
        use_triton_full_attention=use_triton_full_attention,
    )


def test_triton_full_attention_flag_falls_back_cleanly_on_cpu():
    torch.manual_seed(1234)
    ref_model = Model(_make_config(use_triton_full_attention=False)).eval()
    flash_model = Model(_make_config(use_triton_full_attention=True)).eval()
    flash_model.load_state_dict(ref_model.state_dict())

    # Length 128 -- matches BLOCK_M in the kernel's PYTEST_VERSION single-config
    # path. Shorter would make BLOCK_M > N_CTX and trip the static assert.
    idx = torch.arange(1, 129, dtype=torch.long).unsqueeze(0) % 60 + 1
    idx[0, 16] = 63  # one EOS in the middle, exercising doc-boundary code
    ref_targets = idx.clone()
    ref_targets[:, :-1] = idx[:, 1:]
    flash_targets = ref_targets.clone()

    with torch.no_grad():
        ref_logits, ref_loss, _ = ref_model(idx, ref_targets)
        flash_logits, flash_loss, _ = flash_model(idx, flash_targets)

    torch.testing.assert_close(flash_logits, ref_logits)
    torch.testing.assert_close(flash_loss, ref_loss)


@pytest.mark.cuda
def test_triton_full_attention_matches_flex_attention_on_dense_cuda():
    try:
        from modeling.models.attention.triton_full_flash_attention import full_attention  # noqa: F401
    except Exception as exc:
        pytest.skip(f"Triton full attention unavailable: {exc}")

    torch.manual_seed(1234)
    device = "cuda"
    ref_model = Model(_make_config(use_triton_full_attention=False)).to(device).eval()
    flash_model = Model(_make_config(use_triton_full_attention=True)).to(device).eval()
    flash_model.load_state_dict(ref_model.state_dict())

    # Length 128 -- matches BLOCK_M in the kernel's PYTEST_VERSION single-config
    # path. Shorter would make BLOCK_M > N_CTX and trip the static assert.
    idx = (torch.arange(1, 129, device=device, dtype=torch.long).unsqueeze(0) % 60) + 1
    idx[0, 16] = 63  # one EOS in the middle, exercising doc-boundary code
    ref_targets = idx.clone()
    ref_targets[:, :-1] = idx[:, 1:]
    flash_targets = ref_targets.clone()

    with torch.no_grad():
        ref_logits, ref_loss, _ = ref_model(idx, ref_targets)
        flash_logits, flash_loss, _ = flash_model(idx, flash_targets)

    torch.testing.assert_close(flash_logits.float(), ref_logits.float(), atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(flash_loss.float(), ref_loss.float(), atol=2e-2, rtol=1e-2)


@pytest.mark.cuda
def test_triton_full_attention_matches_flex_attention_with_left_padding_cuda():
    """Triton prefill must match the flex/SDPA path on a LEFT-PADDED batch.

    Left-padding used to disable the Triton path (the old is_real_BxT.all()
    gate). Pad tokens land in isolated "fake" documents (generate_left_padded_
    document_idx), so the kernel's document masking already excludes pad keys;
    this asserts the Triton and flex paths agree on the real (non-pad) tokens.
    """
    try:
        from modeling.models.attention.triton_full_flash_attention import full_attention  # noqa: F401
    except Exception as exc:
        pytest.skip(f"Triton full attention unavailable: {exc}")

    torch.manual_seed(1234)
    device = "cuda"
    ref_model = Model(_make_config(use_triton_full_attention=False)).to(device).eval()
    flash_model = Model(_make_config(use_triton_full_attention=True)).to(device).eval()
    flash_model.load_state_dict(ref_model.state_dict())

    # Total length 128 (kernel PYTEST single-config needs N_CTX >= 128), with the
    # first 16 positions left-padded (pad_token_id=0) and 112 real tokens.
    pad_len = 16
    t = 128
    idx = (torch.arange(1, t + 1, device=device, dtype=torch.long).unsqueeze(0) % 60) + 1  # 1..60, never pad id 0
    idx[:, :pad_len] = 0  # left padding
    idx[0, 64] = 63  # one EOS in the real region -> exercises doc-boundary code

    # All-valid targets (token ids, never IGNORE_INDEX) so forward returns full
    # (B, T, V) logits; pad/eos positions are masked internally for the loss.
    targets = idx.clone()
    targets[:, :-1] = idx[:, 1:]

    with torch.no_grad():
        ref_logits, ref_loss, _ = ref_model(idx, targets.clone())
        flash_logits, flash_loss, _ = flash_model(idx, targets.clone())

    # Compare only the real (non-pad) positions; pad-position logits are discarded.
    torch.testing.assert_close(
        flash_logits[:, pad_len:, :].float(),
        ref_logits[:, pad_len:, :].float(),
        atol=2e-2,
        rtol=1e-2,
    )
    torch.testing.assert_close(flash_loss.float(), ref_loss.float(), atol=2e-2, rtol=1e-2)
