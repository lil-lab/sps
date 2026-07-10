# Test Suite

## Running tests

```bash
# All fast (CPU-safe) tests
uv run pytest

# Only CUDA tests
uv run pytest -m cuda

# Only slow diagnostics
uv run pytest --run-slow -m slow

# Everything including slow
uv run pytest --run-slow

# Directory-scoped runs
uv run pytest src/modeling/tests/core -v
uv run pytest src/modeling/tests/attention/triton -v

# Single file
uv run pytest src/modeling/tests/core/test_masking.py -v
```

Markers defined in `pyproject.toml`:

| Marker   | Meaning                                          |
|----------|--------------------------------------------------|
| `cuda`   | Requires CUDA GPU; auto-skipped when unavailable |
| `triton` | Requires Triton (implies CUDA); auto-skipped     |
| `slow`   | Long-running diagnostics; requires `--run-slow`  |

## Directory map

### `core/`

| File | What it tests |
|------|---------------|
| `test_masking.py` | `causal_mask` from `modeling/masking.py` |
| `test_document_idx.py` | `generate_document_idx`: no EOS, single/multiple EOS, boundaries, consecutive EOS, batching, left-padding |
| `test_masked_stats.py` | Masked distribution-stat helpers vs a torch reference, including survival under `torch.compile` with dynamic masked lengths |
| `test_checkpointing.py` | `CheckpointManager`: rolling vs named checkpoints, `ckpt` alias handling, save cadence / decay gating, and resolution order |
| `test_wandb_utils.py` | W&B dir resolution: `prepare_wandb_dir`, env fallback, and `default_wandb_dir_from_repo_config` expanding `system.data_root` |

### `attention/triton/`

| File | What it tests |
|------|---------------|
| `test_full_attention.py` | Triton full-attention: CPU fallback parity and CUDA flex-attention parity |
| `test_full_flash_attention.py` | Triton document-masked causal kernel: forward/backward parity vs PyTorch, document masking, warp_specialize, dtypes |
| `test_full_long_context.py` | Long-context (2048/4096, incl. T=4032 non-multiple-of-128 padding regression) fwd/bwd parity for the full-attention kernel |
| `test_sps_attention.py` | Triton SPS sliding attention vs SDPA reference: forward+backward parity, with and without document masking |
| `test_document_leakage.py` | Forensic cross-document leakage probe (NaN-poison) for the full-attention and SPS-family sliding kernels |

### `models/`

| File | What it tests |
|------|---------------|
| `sps/test_dense_only.py` | SPS dense + `forward_efficient` read the `<predict>` (odd) slot logits; greedy generation matches stepwise; left-padding invariance; stats schema |
| `reverse_sps/test_forward_efficient.py` | Reverse-SPS `forward_efficient` vs dense parity (CUDA), left-padding invariance, dense window visibility, and stats schema |
| `delayed_state/test_forward_efficient.py` | Delayed-State `forward_efficient` reads predict-slot logits, decode window-cache wrap, checkpoint compatibility with Reverse-SPS, and stats schema |
| `attention/test_persistent_key_window.py` | Persistent key-window (retained buffer) decode matches the reference (CUDA) |
| `attention/test_segmented_q2_attention.py` | Segmented q=2 decode attention matches the dense reference |
| `shared/test_full_attention_generation.py` | Full-attention cached decode vs dense greedy: batched prefill, EOS/document advance, and left-padded batches (CUDA) |
| `shared/test_left_padding_invariance.py` | Logit invariance to left-padding for the full-attention model |

### `evaluation/`

| File | What it tests |
|------|---------------|
| `test_metric_selection.py` | `choose_primary_metric_name`: `acc_norm`, LAMBADA perplexity, and bits-per-byte / word-perplexity fallbacks for rolling-perplexity tasks |
| `adapters/test_lm_adapter.py` | `CustomHFLMAdapter` with a dummy model: `_model_call`, shifted targets, loglikelihood, batching, padding, eval stats, auto batch size, `reset_eval_run_state` |
| `adapters/test_lm_adapter_real_models.py` | LM-eval adapter with real full-attention, reverse-SPS, and delayed-state models: loglikelihood, batching, generation, and error handling |
| `pipeline/test_benchmark_wiring.py` | `evaluate.main`: adapter flags, model stats, task validation, checkpoint resolution, wandb init/retry, and error handling |

## Shared utilities

- `conftest.py`: custom marker hooks (`cuda`, `triton`, `slow`) with auto-skip logic and the `device` fixture.
- `_helpers.py`: shared model factory functions and `forward_logits`.
- `_sliding_window_reference.py`: reference sliding-window score modifier used to validate the deterministic window visibility of the SPS-family kernels.

## Coverage gaps

| Module | Status |
|--------|--------|
| `modeling/models/utils/sampling.py` | Only the `forbidden_token_ids` path is tested |
| `modeling/models/utils/decode_attention.py` | No direct tests |
| `modeling/masking.py` | Partial direct coverage |
| `modeling/models/model.py` | `RMSNorm`, `precompute_freqs_cis`, and some position/padding helpers are untested directly |
| `training/dataset.py` | No tests |
| `training/sampler.py` | No tests |
