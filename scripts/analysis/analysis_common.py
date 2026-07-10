#!/usr/bin/env python3
"""Shared utilities for the gradient / persistent-window analyses and the speed
benchmark: model specs, checkpoint loading, warp-specialize control, validation-batch
sampling, and loss helpers."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


IGNORE_INDEX = -100
DEFAULT_FUTURE_HORIZON = 256
# Base dir for datasets/checkpoints. Defaults to the repo root; override with the
# DATA_ROOT env var (or pass explicit paths via CLI).
DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", Path(__file__).resolve().parents[2]))
DEFAULT_VAL_BIN = DEFAULT_DATA_ROOT / "data" / "fineweb-edu" / "val.bin"
MODEL_ORDER = ("full_attention", "sps", "delayed_state")
KIND_ALIASES = {
    "full": "full_attention",
    "full_attention": "full_attention",
    "standard": "full_attention",
    "sps": "sps",
    "reverse_sps": "reverse_sps",
    "delayed_state": "delayed_state",
}


@dataclass(frozen=True)
class ModelSpec:
    kind: str
    label: str
    checkpoint_path: str


@dataclass(frozen=True)
class SlotSpec:
    slot_id: str
    label: str
    position_kind: str

    def hidden_index(self, source_position: int) -> int:
        if self.position_kind == "single":
            return source_position
        if self.position_kind == "even":
            return 2 * source_position
        if self.position_kind == "odd":
            return 2 * source_position + 1
        raise ValueError(f"Unknown slot position kind: {self.position_kind}")


@dataclass(frozen=True)
class LoadedModel:
    model: nn.Module
    config: Any
    kind: str


@dataclass(frozen=True)
class SampledBatchInfo:
    start_offsets: list[int]
    document_end_offsets: list[int]
    document_lengths: list[int]
    source_positions_by_sample: list[list[int]]


@dataclass(frozen=True)
class ValBatch:
    x_BxT: torch.Tensor
    y_BxT: torch.Tensor
    info: SampledBatchInfo


SLOT_SPECS_BY_KIND = {
    "full_attention": (SlotSpec("input_token", "input token", "single"),),
    "sps": (
        SlotSpec("input_token", "input token", "even"),
        SlotSpec("predict_token", "predict token", "odd"),
    ),
    "delayed_state": (
        SlotSpec("input_token", "input token", "even"),
        SlotSpec("predict_token", "predict token", "odd"),
    ),
}


def normalize_kind(kind: str) -> str:
    key = kind.strip().lower().replace("-", "_")
    if key not in KIND_ALIASES:
        known = ", ".join(sorted(KIND_ALIASES))
        raise ValueError(f"Unknown model kind {kind!r}. Expected one of: {known}")
    return KIND_ALIASES[key]


def parse_model_spec(text: str) -> ModelSpec:
    parts = text.split(":", 2)
    if len(parts) != 3:
        raise ValueError(
            "--spec must have format kind:label:/path/to/checkpoint.pt, "
            f"got {text!r}"
        )
    kind = normalize_kind(parts[0])
    label = parts[1].strip() or kind.replace("_", " ").title()
    checkpoint_path = parts[2]
    return ModelSpec(kind=kind, label=label, checkpoint_path=checkpoint_path)


def slot_specs_for_kind(kind: str) -> tuple[SlotSpec, ...]:
    return SLOT_SPECS_BY_KIND[normalize_kind(kind)]


def detect_checkpoint_kind(checkpoint: dict[str, Any]) -> str:
    config = checkpoint.get("config", {})
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    target = str(model_cfg.get("_target_", ""))

    # Check reverse_sps BEFORE sps: "ReverseSPSModel".endswith("SPSModel") is True.
    if ".reverse_sps." in target or target.endswith("ReverseSPSModel"):
        return "reverse_sps"
    if ".sps." in target or target.endswith("SPSModel"):
        return "sps"
    if ".delayed_state." in target or target.endswith("DelayedStateModel"):
        return "delayed_state"
    if "full_attention_model" in target or target.endswith(".Model"):
        return "full_attention"

    model_args = checkpoint.get("model_args", {})
    if isinstance(model_args, dict) and "predict_token_id" not in model_args:
        return "full_attention"

    raise ValueError("Unable to infer model kind from checkpoint metadata")


def _construct_config(config_cls: type, model_args: dict[str, Any], *, kind: str) -> Any:
    valid_keys = {field.name for field in fields(config_cls)}
    filtered_args = {key: value for key, value in model_args.items() if key in valid_keys}
    skipped = sorted(set(model_args) - valid_keys)
    if skipped:
        print(f"WARNING: skipping obsolete {kind} model args: {', '.join(skipped)}")
    return config_cls(**filtered_args)


def add_warp_specialize_arg(parser: "argparse.ArgumentParser") -> None:
    """Add the shared ``--warp-specialize {keep,on,off}`` flag.

    Default ``keep`` leaves the checkpoint's trained value untouched (no regression for
    existing s/m/l runs). XL SPS / Delayed-State kernels fail to compile with
    warp_specialize on the current Triton, so pass ``off`` for XL.
    """
    parser.add_argument(
        "--warp-specialize",
        choices=("keep", "on", "off"),
        default="keep",
        help=(
            "Override warp_specialize on all attention modules after load. 'keep' leaves "
            "the trained value; XL windowed kernels fail to compile with warp_specialize "
            "on the current Triton, so use 'off' for XL."
        ),
    )


def warp_specialize_from_arg(value: str) -> bool | None:
    """Map the ``--warp-specialize`` choice to the loader's ``warp_specialize`` arg."""
    return {"keep": None, "on": True, "off": False}[value]


def set_warp_specialize(model: Any, config: Any, value: bool) -> int:
    """Force warp_specialize on every attention module of an already-built model.

    Windowed models (sps / reverse_sps / delayed_state) default warp_specialize=True and
    cache it at construction (``self.warp_specialize = config.warp_specialize``), so the
    override must be applied to the live modules, not just the config. XL's SPS kernels
    fail to compile with warp_specialize on the current Triton
    (``NVGPUWarpSpecialization`` MLIR pass -> ``PassManager::run failed``); turning it off
    is numerically identical and only changes kernel scheduling/speed. Mirrors
    ``scripts/benchmark/benchmark_generation_speed.py:_set_warp_specialize``.
    """
    count = 0
    for module in model.modules():
        if hasattr(module, "warp_specialize"):
            module.warp_specialize = value
            count += 1
    if hasattr(config, "warp_specialize"):
        config.warp_specialize = value
    return count


def load_checkpoint_model(
    spec: ModelSpec,
    device: torch.device,
    warp_specialize: bool | None = None,
) -> LoadedModel:
    checkpoint_path = Path(spec.checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    try:
        detected_kind = detect_checkpoint_kind(checkpoint)
    except ValueError:
        detected_kind = None
    if detected_kind is not None and detected_kind != spec.kind:
        print(
            f"WARNING: --spec kind {spec.kind!r} does not match checkpoint metadata "
            f"{detected_kind!r} for {checkpoint_path}"
        )

    model_args = dict(checkpoint.get("model_args", {}))
    if spec.kind == "sps":
        from modeling.models.sps import SPSConfig, SPSModel

        config = _construct_config(SPSConfig, model_args, kind=spec.kind)
        model = SPSModel(config)
    elif spec.kind == "delayed_state":
        from modeling.models.delayed_state import DelayedStateConfig, DelayedStateModel

        config = _construct_config(DelayedStateConfig, model_args, kind=spec.kind)
        model = DelayedStateModel(config)
    elif spec.kind == "full_attention":
        from modeling.models.full_attention_model import Model, ModelConfig

        config = _construct_config(ModelConfig, model_args, kind=spec.kind)
        model = Model(config)
    else:
        raise ValueError(f"Unsupported model kind: {spec.kind}")

    state_dict = dict(checkpoint["model"])
    for key in list(state_dict.keys()):
        if key.startswith("_orig_mod."):
            state_dict[key[len("_orig_mod."):]] = state_dict.pop(key)

    model_state_dict = model.state_dict()

    strict = True
    if "freqs_cis" in state_dict and "freqs_cis" in model_state_dict:
        if tuple(state_dict["freqs_cis"].shape) != tuple(model_state_dict["freqs_cis"].shape):
            state_dict.pop("freqs_cis", None)
            strict = False

    model.load_state_dict(state_dict, strict=strict)
    model.to(device)
    model.eval()

    if warp_specialize is not None:
        n = set_warp_specialize(model, config, warp_specialize)
        print(f"warp_specialize={'on' if warp_specialize else 'off'} applied to {n} module(s)")

    return LoadedModel(model=model, config=config, kind=spec.kind)


def _hidden_from_block_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported transformer block output type: {type(output)!r}")


class ModelAdapter:
    def __init__(self, loaded: LoadedModel):
        self.model = loaded.model
        self.config = loaded.config
        self.kind = loaded.kind

    @property
    def slot_specs(self) -> tuple[SlotSpec, ...]:
        return slot_specs_for_kind(self.kind)

    def forward_with_hooks(self, x_BxT: torch.Tensor) -> tuple[list[torch.Tensor], torch.Tensor]:
        captured: list[torch.Tensor | None] = [None for _ in range(self.config.n_layer)]
        hooks = []

        def make_hook(layer_idx: int):
            def hook_fn(_module, _inputs, output):
                hidden = _hidden_from_block_output(output)
                hidden.retain_grad()
                captured[layer_idx] = hidden

            return hook_fn

        for layer_idx, block in enumerate(self.model.transformer.h):
            hooks.append(block.register_forward_hook(make_hook(layer_idx)))

        try:
            if self.kind in {"sps", "delayed_state"}:
                _is_real, _documents_idx_BxT, documents_idx_Bx2T = (
                    self.model._expand_real_and_document_idx(x_BxT)
                )
                idx_Bx2T = self.model.add_predict_tokens(x_BxT)
                x_hidden = self.model.forward_hidden_states(
                    idx_Bx2T,
                    documents_idx_Bx2T=documents_idx_Bx2T,
                )
                logits = self.model.lm_head(x_hidden[:, 1::2])
            else:
                dummy_targets = x_BxT.clone()
                x_hidden, _targets, _is_real = self.model.forward_hidden_states(
                    x_BxT,
                    dummy_targets,
                )
                logits = self.model.lm_head(x_hidden)
        finally:
            for hook in hooks:
                hook.remove()

        if any(hidden is None for hidden in captured):
            missing = [str(i) for i, hidden in enumerate(captured) if hidden is None]
            raise RuntimeError(f"Failed to capture hidden states for layers: {', '.join(missing)}")

        return [hidden for hidden in captured if hidden is not None], logits


def default_val_bin() -> Path:
    env_path = os.environ.get("GRADIENT_ANALYSIS_VAL_BIN")
    if env_path:
        return Path(env_path)
    if DEFAULT_VAL_BIN.exists():
        return DEFAULT_VAL_BIN
    return Path("data") / "fineweb-edu" / "val.bin"


def make_val_batches(
    *,
    val_bin: Path,
    seqlen: int,
    num_batches: int,
    batch_size: int,
    device: torch.device,
    data_seed: int,
    position_seed: int,
    positions_per_batch: int,
    skip_first: int,
    future_horizon: int,
    boundary_token_ids: tuple[int, ...],
    batched_gradient_mode: str = "serial",
) -> list[ValBatch]:
    if not val_bin.exists():
        raise FileNotFoundError(f"Validation token file not found: {val_bin}")

    data = np.memmap(val_bin, dtype=np.uint16, mode="r")
    max_start = len(data) - seqlen - 1
    if max_start < 0:
        raise ValueError(f"{val_bin} is too short for seqlen={seqlen}")

    if positions_per_batch < 1:
        raise ValueError("positions_per_batch must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if skip_first < 0:
        raise ValueError("skip_first must be >= 0")
    if batched_gradient_mode not in {"serial", "shared_position"}:
        raise ValueError(f"Unsupported batched_gradient_mode={batched_gradient_mode!r}")
    if batched_gradient_mode == "shared_position" and positions_per_batch != 1:
        raise ValueError("shared_position mode requires positions_per_batch=1")

    boundary_ids = tuple(int(token_id) for token_id in boundary_token_ids)
    if not boundary_ids:
        raise ValueError("At least one boundary token id is required")

    boundary_mask = np.zeros(len(data), dtype=bool)
    for token_id in boundary_ids:
        boundary_mask |= data == token_id

    boundary_offsets = np.flatnonzero(boundary_mask)
    doc_starts = np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            boundary_offsets.astype(np.int64, copy=False) + 1,
        )
    )
    doc_starts = doc_starts[doc_starts <= max_start]
    if doc_starts.size == 0:
        raise ValueError(f"No document starts found in {val_bin}")

    doc_starts = doc_starts[~boundary_mask[doc_starts]]
    next_boundary_indices = np.searchsorted(boundary_offsets, doc_starts, side="left")
    doc_ends = np.full(doc_starts.shape, len(data), dtype=np.int64)
    has_next_boundary = next_boundary_indices < boundary_offsets.size
    doc_ends[has_next_boundary] = boundary_offsets[next_boundary_indices[has_next_boundary]]
    doc_lengths = doc_ends - doc_starts

    high_by_doc = np.minimum(seqlen - future_horizon, doc_lengths - future_horizon - 1)
    valid_position_counts = np.maximum(0, high_by_doc - skip_first)
    min_position_count = positions_per_batch if batched_gradient_mode == "serial" else 1
    candidate_mask = valid_position_counts >= min_position_count
    candidate_indices = np.flatnonzero(candidate_mask)
    if candidate_indices.size == 0:
        raise ValueError(
            "No document-start spans have enough boundary-free source positions "
            f"for seqlen={seqlen}, future_horizon={future_horizon}, "
            f"skip_first={skip_first}, positions_per_batch={positions_per_batch}"
        )

    weights = valid_position_counts[candidate_indices].astype(np.float64)
    weights /= weights.sum()
    valid_positions = np.arange(skip_first, seqlen - future_horizon)
    if valid_positions.size == 0:
        raise ValueError("No valid source positions remain after applying future_horizon")
    sorted_high_by_doc = np.sort(high_by_doc[candidate_indices])
    support_counts = (
        len(sorted_high_by_doc)
        - np.searchsorted(sorted_high_by_doc, valid_positions, side="right")
    )
    supported_position_mask = support_counts > 0
    shared_positions = valid_positions[supported_position_mask]
    shared_position_weights = support_counts[supported_position_mask].astype(np.float64)
    shared_position_weights /= shared_position_weights.sum()

    data_rng = np.random.default_rng(data_seed)
    batches: list[ValBatch] = []
    for batch_index in range(num_batches):
        if batched_gradient_mode == "shared_position":
            position_rng = np.random.default_rng(position_seed + batch_index)
            shared_position = int(
                position_rng.choice(shared_positions, p=shared_position_weights)
            )
            supporting_indices = candidate_indices[high_by_doc[candidate_indices] > shared_position]
            chosen_doc_indices = data_rng.choice(
                supporting_indices,
                size=batch_size,
                replace=True,
            )
            positions_by_sample = [[shared_position] for _ in range(batch_size)]
        else:
            chosen_doc_indices = data_rng.choice(
                candidate_indices,
                size=batch_size,
                replace=True,
                p=weights,
            )

        starts = doc_starts[chosen_doc_indices].astype(np.int64, copy=False)
        ends = doc_ends[chosen_doc_indices].astype(np.int64, copy=False)
        lengths = doc_lengths[chosen_doc_indices].astype(np.int64, copy=False)

        if batched_gradient_mode == "serial":
            position_rng = np.random.default_rng(position_seed + batch_index)
            positions_by_sample = []
            for length in lengths:
                high = int(min(seqlen - future_horizon, int(length) - future_horizon - 1))
                choices = np.arange(skip_first, high)
                positions = sorted(
                    position_rng.choice(
                        choices,
                        size=positions_per_batch,
                        replace=False,
                    )
                    .astype(int)
                    .tolist()
                )
                positions_by_sample.append(positions)

        rows = [
            np.asarray(data[int(start) : int(start) + seqlen + 1], dtype=np.int64)
            for start in starts
        ]
        tokens = torch.from_numpy(np.stack(rows, axis=0)).to(device=device, dtype=torch.long)
        batches.append(
            ValBatch(
                x_BxT=tokens[:, :-1],
                y_BxT=tokens[:, 1:],
                info=SampledBatchInfo(
                    start_offsets=starts.astype(int).tolist(),
                    document_end_offsets=ends.astype(int).tolist(),
                    document_lengths=lengths.astype(int).tolist(),
                    source_positions_by_sample=positions_by_sample,
                ),
            )
        )
    return batches


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _valid_loss(
    per_pos_loss_BxT: torch.Tensor,
    valid_mask_BxT: torch.Tensor,
    start: int,
    end: int,
) -> torch.Tensor | None:
    mask = valid_mask_BxT[:, start:end].float()
    count = mask.sum()
    if count.item() <= 0:
        return None
    return (per_pos_loss_BxT[:, start:end] * mask).sum() / count


def _valid_row_mean_loss_sum(
    per_pos_loss_BxT: torch.Tensor,
    valid_mask_BxT: torch.Tensor,
    start: int,
    end: int,
) -> torch.Tensor | None:
    mask = valid_mask_BxT[:, start:end].float()
    counts = mask.sum(dim=1)
    valid_rows = counts > 0
    if valid_rows.sum().item() <= 0:
        return None
    row_sums = (per_pos_loss_BxT[:, start:end] * mask).sum(dim=1)
    return (row_sums[valid_rows] / counts[valid_rows]).sum()
