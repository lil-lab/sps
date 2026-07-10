#!/usr/bin/env python3
"""Benchmark generation timing for local checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch
from torch import Tensor

# Repo-root bootstrap. `analysis_common` lives in scripts/analysis and
# `export_main_results_table` in scripts/tables; neither is an installed package, so
# put their directories on sys.path before importing. `src/` is already installed via
# the editable package, so `modeling.*` imports work without help.
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _extra in ("scripts/analysis", "scripts/tables"):
    _extra_path = str(_REPO_ROOT / _extra)
    if _extra_path not in sys.path:
        sys.path.insert(0, _extra_path)

from analysis_common import DEFAULT_VAL_BIN, LoadedModel, load_checkpoint_model, parse_model_spec
from export_main_results_table import (
    render_plain_summary,
    write_efficiency_summary,
)


DEFAULT_OUTPUT_JSON = Path("outputs/generation_timing_correctness/THROUGHPUT_b16_all5_h100/results.json")
# Cluster-independent default: the fineweb-edu val.bin Kempner path baked into
# analysis_common is only a fallback. On another cluster, set BENCH_VAL_BIN (or pass
# --val-bin) to point at that cluster's copy of the validation shard.
DEFAULT_VAL_BIN_PATH = Path(os.environ.get("BENCH_VAL_BIN", str(DEFAULT_VAL_BIN)))


@dataclass
class TimedGeneration:
    tokens_BxT: Tensor
    logits_BxSxV: Tensor | None
    prefill_ms: float
    decode_ms: float
    peak_prefill_mib: float = math.nan
    peak_decode_mib: float = math.nan

    @property
    def total_ms(self) -> float:
        return self.prefill_ms + self.decode_ms


def _parse_int_list(values: Sequence[str]) -> list[int]:
    parsed: list[int] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part:
                parsed.append(int(part))
    if not parsed:
        raise ValueError("Expected at least one integer")
    return parsed


def _time_ms(device: torch.device, fn: Callable[[], Any]) -> tuple[float, Any]:
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize()
        return float(start.elapsed_time(end)), result

    start_time = time.perf_counter()
    result = fn()
    return (time.perf_counter() - start_time) * 1000.0, result


def _summarize(values: Sequence[float]) -> dict[str, float]:
    values = [float(v) for v in values]
    if not values:
        return {"mean": math.nan, "std": math.nan, "min": math.nan, "max": math.nan}
    return {
        "mean": float(statistics.fmean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _gpu_info() -> list[dict[str, str]]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        return []

    if completed.returncode != 0:
        return [{"error": completed.stderr.strip()}]

    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4:
            rows.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "memory_total_mib": parts[2],
                    "driver_version": parts[3],
                }
            )
    return rows


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _set_warp_specialize(loaded: "LoadedModel", value: bool) -> int:
    """Force warp_specialize on every attention module of an already-built model.

    Windowed models (sps / reverse_sps / delayed_state) default warp_specialize=True and
    cache it at construction (``self.warp_specialize = config.warp_specialize``), so the
    override must be applied to the live modules, not just the config. XL's SPS kernels
    fail to compile with warp_specialize on the current Triton
    (``NVGPUWarpSpecialization`` MLIR pass -> ``PassManager::run failed``); turning it off
    is numerically identical and only changes kernel scheduling/speed.
    """
    count = 0
    for module in loaded.model.modules():
        if hasattr(module, "warp_specialize"):
            module.warp_specialize = value
            count += 1
    if hasattr(loaded.config, "warp_specialize"):
        loaded.config.warp_specialize = value
    return count


def _provenance() -> dict[str, Any]:
    """Record everything needed to reproduce a run on another cluster.

    The benchmark hyperparameters live in ``settings``; this captures the *environment*
    those numbers depend on (code version, library versions, CUDA, and the env flags
    that influence kernels) so a ``results.json`` is self-documenting.
    """
    return {
        "git_commit": _git_commit(),
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "env": {
            # The alloc config affects the allocator (and thus peak-memory numbers); the
            # submit sets expandable_segments:True. torch>=2.9 reads PYTORCH_ALLOC_CONF;
            # older versions read PYTORCH_CUDA_ALLOC_CONF. Record both.
            "PYTORCH_ALLOC_CONF": os.environ.get("PYTORCH_ALLOC_CONF"),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        },
    }


def _forward_logits(model: torch.nn.Module, idx_BxT: Tensor) -> Tensor:
    try:
        result = model(idx_BxT)
        return result[0] if isinstance(result, tuple) else result
    except TypeError:
        pass

    if hasattr(model, "forward_hidden_states") and hasattr(model, "lm_head"):
        x_hidden = model.forward_hidden_states(idx_BxT, idx_BxT.clone())[0]
        return model.lm_head(x_hidden)

    raise TypeError(f"Do not know how to obtain logits from {type(model).__name__}")


def _build_forbidden_token_ids(config: Any, device: torch.device) -> Tensor | None:
    vocab_size = int(config.vocab_size)
    ids: set[int] = set()

    eos_token_id = getattr(config, "eos_token_id", None)
    if eos_token_id is not None:
        eos_token_id = int(eos_token_id)
        if 0 <= eos_token_id < vocab_size:
            ids.update(range(eos_token_id + 1, vocab_size))

    for attr in ("pad_token_id", "predict_token_id"):
        token_id = getattr(config, attr, None)
        if token_id is not None and 0 <= int(token_id) < vocab_size:
            ids.add(int(token_id))

    if not ids:
        return None
    return torch.tensor(sorted(ids), device=device, dtype=torch.long)


def _filtered_argmax(logits_BxV: Tensor, forbidden_token_ids: Tensor | None) -> Tensor:
    logits = logits_BxV.float().clone()
    if forbidden_token_ids is not None and forbidden_token_ids.numel() > 0:
        logits[:, forbidden_token_ids] = float("-inf")
    return logits.argmax(dim=-1)


def _sample_prompts(
    *,
    val_bin: Path,
    prompt_len: int,
    num_prompts: int,
    seed: int,
    vocab_size: int,
    device: torch.device,
) -> Tensor:
    if prompt_len <= 0:
        raise ValueError(f"prompt_len must be positive, got {prompt_len}")
    if num_prompts <= 0:
        raise ValueError(f"num_prompts must be positive, got {num_prompts}")
    if not val_bin.exists():
        raise FileNotFoundError(f"Validation token file not found: {val_bin}")

    data = np.memmap(val_bin, dtype=np.uint16, mode="r")
    max_start = len(data) - prompt_len
    if max_start < 0:
        raise ValueError(f"{val_bin} is too short for prompt_len={prompt_len}")

    rng = np.random.default_rng(seed + 1009 * prompt_len)
    prompts: list[np.ndarray] = []
    attempts = 0
    while len(prompts) < num_prompts:
        if attempts > num_prompts * 100:
            raise RuntimeError(
                f"Could not sample {num_prompts} valid prompts of length {prompt_len} "
                f"from {val_bin}"
            )
        attempts += 1
        start = int(rng.integers(0, max_start + 1))
        window = np.asarray(data[start : start + prompt_len], dtype=np.int64)
        if int(window.max(initial=0)) >= vocab_size:
            continue
        prompts.append(window)

    stacked = np.stack(prompts, axis=0)
    return torch.tensor(stacked, device=device, dtype=torch.long)


@torch.inference_mode()
def _public_generate(
    model: torch.nn.Module,
    prompts_BxT: Tensor,
    *,
    max_new_tokens: int,
    forbidden_token_ids: Tensor | None,
    collect_logits: bool,
) -> TimedGeneration:
    del collect_logits
    device = prompts_BxT.device

    def run_generate() -> Tensor:
        return model.generate(
            prompts_BxT,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            stop_on_eos=False,
            forbidden_token_ids=forbidden_token_ids,
        )

    elapsed_ms, generated_BxT = _time_ms(device, run_generate)
    return TimedGeneration(generated_BxT, None, 0.0, elapsed_ms)


@torch.inference_mode()
def _cached_prefill_generate(
    model: torch.nn.Module,
    prompts_BxT: Tensor,
    *,
    max_new_tokens: int,
    forbidden_token_ids: Tensor | None,
    collect_logits: bool,
    require_batched_prefill: bool,
) -> tuple[str, TimedGeneration]:
    prefill_fn = getattr(model, "_prefill_generation_state", None)
    decode_fn = getattr(model, "_decode_generation_state", None)
    advance_fn = getattr(model, "_advance_decode_state", None)
    if prefill_fn is None or decode_fn is None or advance_fn is None:
        if require_batched_prefill:
            raise RuntimeError(f"{type(model).__name__} does not expose cached generation prefill")
        return "public_generate", _public_generate(
            model,
            prompts_BxT,
            max_new_tokens=max_new_tokens,
            forbidden_token_ids=forbidden_token_ids,
            collect_logits=collect_logits,
        )

    device = prompts_BxT.device
    bsz = prompts_BxT.size(0)
    vocab_size = int(model.config.vocab_size)
    generated_BxS = torch.empty(
        (bsz, max_new_tokens),
        device=device,
        dtype=prompts_BxT.dtype,
    )
    logits_steps: list[Tensor] = []

    def prefill() -> tuple[Any, Tensor, str]:
        return prefill_fn(
            prompts_BxT,
            max_new_tokens,
            require_batched=require_batched_prefill,
        )

    if device.type == "cuda":
        torch.cuda.synchronize()
        # Free PyTorch's internal allocator cache so the reading below reflects
        # only tensors that are actually held by Python references (the KV
        # cache, model weights, the few small handles), not the high-water
        # mark of transient prefill activations that already went out of scope.
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    prefill_ms, prefill_result = _time_ms(device, prefill)
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak_prefill_mib = float(torch.cuda.max_memory_allocated() / (1024 ** 2))
    else:
        peak_prefill_mib = math.nan
    state, next_logits_BxV, prefill_mode = prefill_result
    if device.type == "cuda":
        # Drop transient prefill activations from the allocator cache before
        # we start measuring decode, so the decode peak reflects steady-state
        # KV cache + activations actually held during decode.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        decode_baseline_mib = float(torch.cuda.memory_allocated() / (1024 ** 2))
    else:
        decode_baseline_mib = math.nan
    if require_batched_prefill and prefill_mode != "batched_prefill":
        raise RuntimeError(f"Expected batched_prefill, got {prefill_mode!r}")

    decode_ms = 0.0
    active_B = torch.ones((bsz,), device=device, dtype=torch.bool)
    for step in range(max_new_tokens):
        if collect_logits:
            logits_steps.append(next_logits_BxV[:, :vocab_size].detach().float())

        next_token_B = _filtered_argmax(next_logits_BxV[:, :vocab_size], forbidden_token_ids).to(
            prompts_BxT.dtype
        )
        generated_BxS[:, step] = next_token_B
        if step == max_new_tokens - 1:
            break

        def decode_step() -> Tensor:
            logits_BxV = decode_fn(state, next_token_B, active_B)
            advance_fn(state, next_token_B, active_B)
            return logits_BxV

        elapsed_ms, next_logits_BxV = _time_ms(device, decode_step)
        decode_ms += elapsed_ms

    if device.type == "cuda":
        torch.cuda.synchronize()
        # Steady-state held memory at the end of decode: KV cache + model
        # weights + small live handles. This is the metric we actually want
        # to compare across methods because it reflects what each method
        # has to keep around for autoregressive generation.
        peak_decode_mib = float(torch.cuda.memory_allocated() / (1024 ** 2))
    else:
        peak_decode_mib = math.nan
    logits_BxSxV = torch.stack(logits_steps, dim=1) if collect_logits else None
    return prefill_mode, TimedGeneration(
        torch.cat([prompts_BxT, generated_BxS], dim=1),
        logits_BxSxV,
        prefill_ms,
        decode_ms,
        peak_prefill_mib=peak_prefill_mib,
        peak_decode_mib=peak_decode_mib,
    )


@torch.inference_mode()
def _dense_greedy_generate(
    model: torch.nn.Module,
    prompts_BxT: Tensor,
    *,
    max_new_tokens: int,
    forbidden_token_ids: Tensor | None,
    collect_logits: bool,
) -> TimedGeneration:
    device = prompts_BxT.device
    block_size = int(model.config.block_size)
    vocab_size = int(model.config.vocab_size)
    generated_BxT = prompts_BxT.clone()
    logits_steps: list[Tensor] = []
    prefill_ms = 0.0
    decode_ms = 0.0

    for step in range(max_new_tokens):
        context_BxT = generated_BxT[:, -block_size:]

        def forward_step() -> Tensor:
            logits_BxTxV = _forward_logits(model, context_BxT)
            return logits_BxTxV[:, -1, :vocab_size]

        elapsed_ms, next_logits_BxV = _time_ms(device, forward_step)
        if step == 0:
            prefill_ms += elapsed_ms
        else:
            decode_ms += elapsed_ms

        if collect_logits:
            logits_steps.append(next_logits_BxV.detach().float())

        next_token_B = _filtered_argmax(next_logits_BxV, forbidden_token_ids).to(prompts_BxT.dtype)
        generated_BxT = torch.cat([generated_BxT, next_token_B[:, None]], dim=1)

    logits_BxSxV = torch.stack(logits_steps, dim=1) if collect_logits else None
    return TimedGeneration(generated_BxT, logits_BxSxV, prefill_ms, decode_ms)


@torch.inference_mode()
def _full_attention_cached_generate(
    model: torch.nn.Module,
    prompts_BxT: Tensor,
    *,
    max_new_tokens: int,
    forbidden_token_ids: Tensor | None,
    collect_logits: bool,
) -> TimedGeneration:
    device = prompts_BxT.device
    bsz, prompt_len = prompts_BxT.shape
    vocab_size = int(model.config.vocab_size)
    generated_BxS = torch.empty(
        (bsz, max_new_tokens),
        device=device,
        dtype=prompts_BxT.dtype,
    )
    logits_steps: list[Tensor] = []

    with model._temporary_disable_generation_triton():
        def prefill() -> tuple[Tensor, Tensor, Any, Tensor]:
            cached_documents_idx_BxK = model.generate_document_idx(prompts_BxT)
            next_document_idx_B = cached_documents_idx_BxK[:, -1] + (
                prompts_BxT[:, -1] == model.config.eos_token_id
            ).to(cached_documents_idx_BxK.dtype)
            hidden_BxTxC, past_key_values = model._forward_generation_hidden_states(prompts_BxT)
            next_logits_BxV = model.lm_head(hidden_BxTxC)[:, -1, :vocab_size]
            return cached_documents_idx_BxK, next_document_idx_B, past_key_values, next_logits_BxV

        prefill_ms, prefill_result = _time_ms(device, prefill)
        cached_documents_idx_BxK, next_document_idx_B, past_key_values, next_logits_BxV = prefill_result
        decode_ms = 0.0
        current_len = prompt_len

        for step in range(max_new_tokens):
            if collect_logits:
                logits_steps.append(next_logits_BxV.detach().float())

            next_token_B = _filtered_argmax(next_logits_BxV, forbidden_token_ids).to(prompts_BxT.dtype)
            generated_BxS[:, step] = next_token_B
            current_documents_idx_BxT = next_document_idx_B.view(bsz, 1)

            def decode_step() -> tuple[Any, Tensor]:
                hidden_Bx1xC, next_past_key_values = model._forward_generation_hidden_states(
                    next_token_B.view(bsz, 1),
                    past_key_values_Lx2=past_key_values,
                    cache_lengths_B=torch.full(
                        (bsz,),
                        current_len,
                        device=device,
                        dtype=torch.long,
                    ),
                    cached_documents_idx_BxK=cached_documents_idx_BxK,
                    current_documents_idx_BxT=current_documents_idx_BxT,
                )
                logits_BxV = model.lm_head(hidden_Bx1xC)[:, -1, :vocab_size]
                return next_past_key_values, logits_BxV

            elapsed_ms, decode_result = _time_ms(device, decode_step)
            decode_ms += elapsed_ms
            past_key_values, next_logits_BxV = decode_result
            cached_documents_idx_BxK = torch.cat(
                [cached_documents_idx_BxK, current_documents_idx_BxT],
                dim=1,
            )
            next_document_idx_B = next_document_idx_B + (
                next_token_B == model.config.eos_token_id
            ).to(next_document_idx_B.dtype)
            current_len += 1

    logits_BxSxV = torch.stack(logits_steps, dim=1) if collect_logits else None
    return TimedGeneration(
        torch.cat([prompts_BxT, generated_BxS], dim=1),
        logits_BxSxV,
        prefill_ms,
        decode_ms,
    )


def _run_generation(
    loaded: LoadedModel,
    prompts_BxT: Tensor,
    *,
    max_new_tokens: int,
    forbidden_token_ids: Tensor | None,
    collect_logits: bool,
    require_batched_prefill: bool,
) -> tuple[str, TimedGeneration]:
    if hasattr(loaded.model, "generate"):
        return _cached_prefill_generate(
            loaded.model,
            prompts_BxT,
            max_new_tokens=max_new_tokens,
            forbidden_token_ids=forbidden_token_ids,
            collect_logits=collect_logits,
            require_batched_prefill=require_batched_prefill,
        )
    if require_batched_prefill:
        raise RuntimeError(f"{type(loaded.model).__name__} does not support batched prefill")
    return "dense_fallback", _dense_greedy_generate(
            loaded.model,
            prompts_BxT,
            max_new_tokens=max_new_tokens,
            forbidden_token_ids=forbidden_token_ids,
            collect_logits=collect_logits,
        )




def _benchmark_prompt_length(
    loaded: LoadedModel,
    *,
    val_bin: Path,
    prompt_len: int,
    num_prompts: int,
    max_new_tokens: int,
    warmup_iters: int,
    timed_iters: int,
    seed: int,
    device: torch.device,
    require_batched_prefill: bool,
) -> dict[str, Any]:
    block_size = int(loaded.config.block_size)
    if prompt_len + max_new_tokens > block_size:
        raise ValueError(
            f"prompt_len + max_new_tokens must be <= block_size for native generation: "
            f"{prompt_len} + {max_new_tokens} > {block_size}"
        )

    prompts_BxT = _sample_prompts(
        val_bin=val_bin,
        prompt_len=prompt_len,
        num_prompts=num_prompts,
        seed=seed,
        vocab_size=int(loaded.config.vocab_size),
        device=device,
    )
    forbidden_token_ids = _build_forbidden_token_ids(loaded.config, device)

    print(
        f"[{loaded.kind}] prompt_len={prompt_len}: timing "
        f"({num_prompts} prompts, {max_new_tokens} tokens, {timed_iters} timed)",
        flush=True,
    )
    for _ in range(warmup_iters):
        _run_generation(
            loaded,
            prompts_BxT,
            max_new_tokens=max_new_tokens,
            forbidden_token_ids=forbidden_token_ids,
            collect_logits=False,
            require_batched_prefill=require_batched_prefill,
        )

    mode = ""
    total_ms_values: list[float] = []
    prefill_ms_values: list[float] = []
    decode_ms_values: list[float] = []
    tokens_per_sec_values: list[float] = []
    decode_tokens_per_sec_values: list[float] = []
    peak_memory_values: list[float] = []        # legacy: prefill+decode combined
    peak_prefill_values: list[float] = []
    peak_decode_values: list[float] = []
    tokens_generated = int(num_prompts * max_new_tokens)
    decode_tokens_generated = int(num_prompts * max(max_new_tokens - 1, 0))

    for _ in range(timed_iters):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        mode, timed = _run_generation(
            loaded,
            prompts_BxT,
            max_new_tokens=max_new_tokens,
            forbidden_token_ids=forbidden_token_ids,
            collect_logits=False,
            require_batched_prefill=require_batched_prefill,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_memory_values.append(float(torch.cuda.max_memory_allocated() / (1024**2)))
        if not math.isnan(timed.peak_prefill_mib):
            peak_prefill_values.append(timed.peak_prefill_mib)
        if not math.isnan(timed.peak_decode_mib):
            peak_decode_values.append(timed.peak_decode_mib)

        total_ms_values.append(timed.total_ms)
        prefill_ms_values.append(timed.prefill_ms)
        decode_ms_values.append(timed.decode_ms)
        tokens_per_sec_values.append(tokens_generated / max(timed.total_ms / 1000.0, 1e-12))
        if decode_tokens_generated > 0 and timed.decode_ms > 0.0:
            decode_tokens_per_sec_values.append(
                decode_tokens_generated / max(timed.decode_ms / 1000.0, 1e-12)
            )
        else:
            decode_tokens_per_sec_values.append(math.nan)

    return {
        "prompt_len": int(prompt_len),
        "prompt_count": int(num_prompts),
        "batch_size": int(num_prompts),
        "new_tokens": int(max_new_tokens),
        "mode": mode,
        "timing": {
            "total_ms": _summarize(total_ms_values),
            "prefill_ms": _summarize(prefill_ms_values),
            "decode_ms": _summarize(decode_ms_values),
            "tokens_per_sec": _summarize(tokens_per_sec_values),
            "decode_tokens_per_sec": _summarize(decode_tokens_per_sec_values),
            "peak_cuda_memory_mib": _summarize(peak_memory_values),
            "peak_prefill_mib": _summarize(peak_prefill_values),
            "peak_decode_mib": _summarize(peak_decode_values),
            "warmup_iters": int(warmup_iters),
            "timed_iters": int(timed_iters),
        },
    }


def _model_record(
    loaded: LoadedModel,
    *,
    label: str,
    checkpoint_path: str,
    val_bin: Path,
    prompt_lens: Sequence[int],
    num_prompts_list: Sequence[int],
    max_new_tokens_list: Sequence[int],
    warmup_iters: int,
    timed_iters: int,
    seed: int,
    device: torch.device,
    require_batched_prefill: bool,
    sink: list | None = None,
    flush_cb: Callable[[], None] | None = None,
) -> dict[str, Any]:
    parameter_count = sum(p.numel() for p in loaded.model.parameters())
    record = {
        "kind": loaded.kind,
        "label": label,
        "run_name": Path(checkpoint_path).parent.name,
        "checkpoint_path": checkpoint_path,
        "parameter_count": int(parameter_count),
        "config": {
            "block_size": int(loaded.config.block_size),
            "vocab_size": int(loaded.config.vocab_size),
            "n_layer": int(loaded.config.n_layer),
            "n_head": int(loaded.config.n_head),
            "hidden_size": int(loaded.config.hidden_size),
            "window_size": getattr(loaded.config, "window_size", None),
        },
        "prompt_length_results": [],
    }
    # Register the (still-empty) record in the output now so per-batch flushes capture
    # progress even if a later batch hard-crashes the process.
    if sink is not None:
        sink.append(record)
    results = record["prompt_length_results"]
    for prompt_len in prompt_lens:
        for num_prompts in num_prompts_list:
            for max_new_tokens in max_new_tokens_list:
                try:
                    results.append(
                        _benchmark_prompt_length(
                            loaded,
                            val_bin=val_bin,
                            prompt_len=int(prompt_len),
                            num_prompts=int(num_prompts),
                            max_new_tokens=int(max_new_tokens),
                            warmup_iters=warmup_iters,
                            timed_iters=timed_iters,
                            seed=seed,
                            device=device,
                            require_batched_prefill=require_batched_prefill,
                        )
                    )
                except torch.cuda.OutOfMemoryError as exc:
                    # Record this batch as OOM and keep going to the *smaller* batches
                    # in the list (and the next model). This lets each config push up
                    # to its own memory ceiling without discarding the points below it.
                    print(
                        f"!! CUDA OOM on {label} at batch={num_prompts} "
                        f"(prompt_len={prompt_len}); recording and continuing",
                        flush=True,
                    )
                    results.append({
                        "prompt_len": int(prompt_len),
                        "prompt_count": int(num_prompts),
                        "batch_size": int(num_prompts),
                        "new_tokens": int(max_new_tokens),
                        "mode": "oom",
                        "error": f"OutOfMemoryError: {exc}",
                    })
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                # Flush after every batch so a hard crash (e.g. CUDA illegal memory
                # access at extreme batch, which is NOT a catchable OutOfMemoryError)
                # still leaves the completed batches on disk.
                if flush_cb is not None:
                    flush_cb()

    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure generation timing for local checkpoints."
    )
    parser.add_argument(
        "--spec",
        action="append",
        required=True,
        help="Model spec as kind:label:/path/to/checkpoint.pt. May be repeated.",
    )
    parser.add_argument("--prompt-lens", nargs="+", default=["128", "1024"])
    parser.add_argument("--num-prompts", nargs="+", default=["2"])
    parser.add_argument("--max-new-tokens", nargs="+", default=["64"])
    parser.add_argument("--warmup-iters", type=int, default=2)
    parser.add_argument("--timed-iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--val-bin", type=Path, default=DEFAULT_VAL_BIN_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument(
        "--allow-step-prefill",
        action="store_true",
        help="Allow generation implementations that prefill prompts token-by-token.",
    )
    parser.add_argument(
        "--config-name",
        default=os.environ.get("BENCHMARK", None),
        help="Name of the benchmark config this run reproduces (recorded in results.json).",
    )
    parser.add_argument(
        "--warp-specialize",
        choices=["keep", "on", "off"],
        default="keep",
        help="Override warp_specialize on all attention modules after load. 'keep' leaves "
        "the checkpoint/config default (use for xs/s/m/l to match historical runs). XL SPS "
        "kernels fail to compile with warp_specialize on the current Triton, so use 'off' "
        "for XL (numerically identical, perf-only). Recorded in settings.",
    )
    args = parser.parse_args()

    max_new_tokens_list = _parse_int_list(args.max_new_tokens)
    num_prompts_list = _parse_int_list(args.num_prompts)
    if any(value <= 0 for value in max_new_tokens_list):
        raise ValueError("--max-new-tokens values must be positive")
    if any(value <= 0 for value in num_prompts_list):
        raise ValueError("--num-prompts values must be positive")
    if args.warmup_iters < 0 or args.timed_iters <= 0:
        raise ValueError("--warmup-iters must be >= 0 and --timed-iters must be > 0")

    prompt_lens = _parse_int_list(args.prompt_lens)
    require_batched_prefill = not bool(args.allow_step_prefill)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("high")

    specs = [parse_model_spec(spec_text) for spec_text in args.spec]
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "gpu": _gpu_info(),
        "provenance": _provenance(),
        "settings": {
            "config_name": args.config_name,
            "warp_specialize": args.warp_specialize,
            "prompt_lens": prompt_lens,
            "num_prompts": num_prompts_list[0] if len(num_prompts_list) == 1 else num_prompts_list,
            "num_prompts_list": num_prompts_list,
            "max_new_tokens": max_new_tokens_list[0] if len(max_new_tokens_list) == 1 else max_new_tokens_list,
            "max_new_tokens_list": max_new_tokens_list,
            "warmup_iters": int(args.warmup_iters),
            "timed_iters": int(args.timed_iters),
            "seed": int(args.seed),
            "val_bin": str(args.val_bin),
            "require_batched_prefill": bool(require_batched_prefill),
        },
        "models": [],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    def _flush_output() -> None:
        # Write incrementally after every model so a later OOM / SLURM timeout still
        # leaves a results.json with all the models that did complete. (XL w4096 at
        # large batch can OOM a single 80GB GPU; large multi-model runs can outlast a
        # time limit.)
        args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True))

    n_specs = len(specs)
    for idx, spec in enumerate(specs, 1):
        print(f"=== Loading {spec.label} ({spec.kind}) [{idx}/{n_specs}] ===", flush=True)
        loaded = None
        try:
            loaded = load_checkpoint_model(spec, device)
            if args.warp_specialize != "keep":
                n = _set_warp_specialize(loaded, args.warp_specialize == "on")
                print(f"    warp_specialize={args.warp_specialize} applied to {n} module(s)", flush=True)
            record = _model_record(
                loaded,
                label=spec.label,
                checkpoint_path=spec.checkpoint_path,
                val_bin=args.val_bin,
                prompt_lens=prompt_lens,
                num_prompts_list=num_prompts_list,
                max_new_tokens_list=max_new_tokens_list,
                warmup_iters=int(args.warmup_iters),
                timed_iters=int(args.timed_iters),
                seed=int(args.seed),
                device=device,
                require_batched_prefill=require_batched_prefill,
                sink=output["models"],
                flush_cb=_flush_output,
            )
        except torch.cuda.OutOfMemoryError as exc:
            # Record the failure and keep going so the other models still get measured.
            print(f"!! CUDA OOM on {spec.label} ({spec.kind}); recording error and continuing", flush=True)
            record = {
                "kind": spec.kind,
                "label": spec.label,
                "checkpoint_path": spec.checkpoint_path,
                "error": f"OutOfMemoryError: {exc}",
            }
        finally:
            del loaded
            if device.type == "cuda":
                torch.cuda.empty_cache()
        # _model_record already appended its live record to output["models"] (via sink);
        # only the load/warp OOM-error record above still needs appending.
        if not any(m is record for m in output["models"]):
            output["models"].append(record)
        _flush_output()
        print(f"Wrote {args.output_json} ({idx}/{n_specs} models)", flush=True)

    summary_tsv = args.output_json.with_name("results_summary.tsv")
    # Only models that actually produced timings can feed the efficiency summary.
    summary_input = {**output, "models": [m for m in output["models"] if "prompt_length_results" in m]}
    try:
        rows = write_efficiency_summary(summary_input, summary_tsv=summary_tsv)
    except ValueError as exc:
        print(f"Skipped generation efficiency summary: {exc}", flush=True)
    else:
        print(render_plain_summary(rows), end="", flush=True)
        print(f"Wrote {summary_tsv}", flush=True)

if __name__ == "__main__":
    main()
