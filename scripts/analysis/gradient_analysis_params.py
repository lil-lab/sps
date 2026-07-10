#!/usr/bin/env python3
"""Per-position PARAMETER-gradient analysis (current vs future loss).

Parameter-space analog of the hidden-state gradient analysis. Instead of reading the
gradient of a loss w.r.t. the *hidden state* at a given sequence position, this
script attributes the gradient w.r.t. the model *parameters* back to each
position -- i.e. it "pretends each position has its own copy of the weights" and
reads the gradient flowing into the position-``p`` copy.

For a linear ``y = x Wᵀ`` the contribution of position ``p`` to the weight
gradient is ``M_p = Σ_batch grad_out[:, p, :]ᵀ ⊗ input[:, p, :]`` (the outer
product of the gradient at the module's output at ``p`` and the module's input
activation at ``p``). This is the exact parameter-space analog of reading
``hidden.grad[:, p, :]``. For an RMSNorm weight (``y = xhat ⊙ w``) the
contribution is ``grad_out[:, p, :] ⊙ xhat[:, p, :]`` (elementwise).

We never materialise the (out, in) contribution matrices: a group's per-position
contribution NORM reduces to dot products of the small per-position
output-gradient / input-activation vectors (for a linear weight,
``||g ⊗ a||^2 = (g·g)(a·a)``). The norm is read immediately after each backward
and only scalars are kept.

Summary metrics, per (source position, slot, layer, module-type ∈ {attention,
mlp, norm}): present_norm, future_mean_norm, and per-offset future norms with the
future/present gradient ratio (future_over_present).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import warnings
import os
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.analysis.analysis_common import (
    add_warp_specialize_arg,
    set_warp_specialize,
    warp_specialize_from_arg,
)
from scripts.analysis import grad_params_io


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

MODULE_TYPES = ("attention", "mlp", "norm", "embedding")


# ---------------------------------------------------------------------------
# Specs / model loading (verbatim from analysis_common.py)
# ---------------------------------------------------------------------------
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
    valid_keys = {field_.name for field_ in fields(config_cls)}
    filtered_args = {key: value for key, value in model_args.items() if key in valid_keys}
    skipped = sorted(set(model_args) - valid_keys)
    if skipped:
        print(f"WARNING: skipping obsolete {kind} model args: {', '.join(skipped)}")
    return config_cls(**filtered_args)


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
    # NOTE: train() vs eval() does not matter for grad flow here; we keep eval()
    # for deterministic dropout (the base script does the same).
    model.eval()

    if warp_specialize is not None:
        n = set_warp_specialize(model, config, warp_specialize)
        print(f"warp_specialize={'on' if warp_specialize else 'off'} applied to {n} module(s)")

    return LoadedModel(model=model, config=config, kind=spec.kind)


# ---------------------------------------------------------------------------
# Module-type classification + per-submodule capture
# ---------------------------------------------------------------------------
try:  # for isinstance-based norm detection; fall back to type-name check.
    from modeling.models.model import RMSNorm as _RMSNorm
except Exception:  # pragma: no cover - import path differences
    _RMSNorm = None


def _is_rmsnorm(module: nn.Module) -> bool:
    if _RMSNorm is not None and isinstance(module, _RMSNorm):
        return True
    return type(module).__name__ == "RMSNorm"


def classify_module(qualified_name: str, module: nn.Module) -> str | None:
    """Map a submodule (within a transformer block) to a module-type group.

    Returns one of {"attention", "mlp", "norm"} for the leaf modules whose
    weights we attribute, or None for modules we ignore.
    """
    if _is_rmsnorm(module):
        return "norm"
    if isinstance(module, nn.Linear):
        low = qualified_name.lower()
        if "attn" in low or "attention" in low:
            return "attention"
        if "mlp" in low or "ffn" in low or "feed" in low:
            return "mlp"
        return "other"
    return None


@dataclass
class CapturedSub:
    layer: int
    module_type: str
    name: str
    kind: str  # "linear" | "norm"
    has_bias: bool
    eps: float | None
    # Filled by the forward hook (per batch forward):
    input: torch.Tensor | None = None
    output: torch.Tensor | None = None


def _hidden_from_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported submodule output type: {type(output)!r}")


class ModelAdapter:
    def __init__(self, loaded: LoadedModel):
        self.model = loaded.model
        self.config = loaded.config
        self.kind = loaded.kind

    @property
    def slot_specs(self) -> tuple[SlotSpec, ...]:
        return slot_specs_for_kind(self.kind)

    def _forward_logits(self, x_BxT: torch.Tensor) -> torch.Tensor:
        """Same forward branches as analysis_common.py's ModelAdapter, no block-output hooks."""
        if self.kind in {"sps", "delayed_state"}:
            _is_real, _documents_idx_BxT, documents_idx_Bx2T = (
                self.model._expand_real_and_document_idx(x_BxT)
            )
            idx_Bx2T = self.model.add_predict_tokens(x_BxT)
            x_hidden = self.model.forward_hidden_states(
                idx_Bx2T,
                documents_idx_Bx2T=documents_idx_Bx2T,
            )
            return self.model.lm_head(x_hidden[:, 1::2])
        dummy_targets = x_BxT.clone()
        x_hidden, _targets, _is_real = self.model.forward_hidden_states(
            x_BxT,
            dummy_targets,
        )
        return self.model.lm_head(x_hidden)

    def forward_with_capture(
        self, x_BxT: torch.Tensor
    ) -> tuple[list[CapturedSub], torch.Tensor]:
        subs: list[CapturedSub] = []
        hooks = []
        other_names: list[str] = []

        def make_hook(rec: CapturedSub):
            def hook_fn(_module, inputs, output):
                rec.input = inputs[0]
                out = _hidden_from_output(output)
                out.retain_grad()
                rec.output = out

            return hook_fn

        for layer_idx, block in enumerate(self.model.transformer.h):
            for name, module in block.named_modules():
                mtype = classify_module(name, module)
                if mtype is None:
                    continue
                if mtype == "other":
                    other_names.append(f"{layer_idx}.{name}")
                    continue
                rec = CapturedSub(
                    layer=layer_idx,
                    module_type=mtype,
                    name=f"{layer_idx}.{name}",
                    kind="norm" if mtype == "norm" else "linear",
                    has_bias=getattr(module, "bias", None) is not None,
                    eps=float(getattr(module, "eps", 0.0)) if _is_rmsnorm(module) else None,
                )
                hooks.append(module.register_forward_hook(make_hook(rec)))
                subs.append(rec)

        if other_names:
            print(
                "WARNING: unclassified Linear submodules (excluded from groups): "
                + ", ".join(other_names)
            )

        # Token embedding (input side). It feeds layer 0 and flows forward, so it
        # receives both present and future gradient like the in-block params. The
        # "input" is a one-hot token id, so the per-position contribution to the
        # embedding-weight gradient is exactly the gradient of the embedding
        # output at that position (forward factor a2 = 1, handled in _build_factors).
        # NOTE: wte is weight-tied to lm_head; we attribute only the input side.
        wte = getattr(self.model.transformer, "wte", None)
        if isinstance(wte, nn.Embedding):
            rec = CapturedSub(
                layer=0,
                module_type="embedding",
                name="wte",
                kind="embedding",
                has_bias=False,
                eps=None,
            )
            hooks.append(wte.register_forward_hook(make_hook(rec)))
            subs.append(rec)
        else:
            print("WARNING: transformer.wte is not an nn.Embedding; embedding skipped")

        try:
            logits = self._forward_logits(x_BxT)
        finally:
            for hook in hooks:
                hook.remove()

        missing = [rec.name for rec in subs if rec.output is None]
        if missing:
            raise RuntimeError(
                "Forward did not run these submodules (no output captured): "
                + ", ".join(missing)
            )
        return subs, logits


# ---------------------------------------------------------------------------
# Data sampling (verbatim from analysis_common.py)
# ---------------------------------------------------------------------------
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


def _document_table(
    data: np.ndarray,
    *,
    seqlen: int,
    future_horizon: int,
    skip_first: int,
    boundary_ids: tuple[int, ...],
) -> dict[str, np.ndarray]:
    """Per-document starts/ends/lengths and the boundary-clean safe-position range.

    Mirrors the boundary/``high_by_doc`` logic in ``make_val_batches`` (a document
    start is the token after a boundary, excluding boundary tokens; the safe range
    is ``[skip_first, high_by_doc)`` where ``high_by_doc`` keeps ``position +
    future_horizon`` inside both the document and the ``seqlen`` window)."""
    n = len(data)
    max_start = n - seqlen - 1
    boundary_mask = np.zeros(n, dtype=bool)
    for token_id in boundary_ids:
        boundary_mask |= data == token_id
    boundary_offsets = np.flatnonzero(boundary_mask)
    doc_starts = np.concatenate(
        (np.asarray([0], dtype=np.int64), boundary_offsets.astype(np.int64) + 1)
    )
    doc_starts = doc_starts[doc_starts <= max_start]
    doc_starts = doc_starts[~boundary_mask[doc_starts]]
    next_boundary_indices = np.searchsorted(boundary_offsets, doc_starts, side="left")
    doc_ends = np.full(doc_starts.shape, n, dtype=np.int64)
    has_next = next_boundary_indices < boundary_offsets.size
    doc_ends[has_next] = boundary_offsets[next_boundary_indices[has_next]]
    doc_lengths = doc_ends - doc_starts
    high_by_doc = np.minimum(seqlen - future_horizon, doc_lengths - future_horizon - 1)
    valid_position_counts = np.maximum(0, high_by_doc - skip_first)
    return {
        "doc_starts": doc_starts,
        "doc_ends": doc_ends,
        "doc_lengths": doc_lengths,
        "high_by_doc": high_by_doc,
        "valid_position_counts": valid_position_counts,
    }


def make_multiposition_batches(
    *,
    val_bin: Path,
    seqlen: int,
    num_documents: int,
    positions_per_document: int,
    batch_size: int,
    device: torch.device,
    data_seed: int,
    position_seed: int,
    skip_first: int,
    future_horizon: int,
    boundary_token_ids: tuple[int, ...],
    doc_start: int = 0,
    shard_index: int = 0,
    num_shards: int = 1,
) -> list[ValBatch]:
    """Sample documents ``[doc_start, num_documents)`` x ``positions_per_document``
    source positions and pack one shard's targets into batches of per-row positions.

    Each batch row is a single (document-window, source-position) **target** at a
    document start, with the source position drawn from the boundary-clean safe
    range so the full ``future_horizon`` stays inside the document.

    **Prefix-stable / incremental**: the eligible documents are ordered by one
    fixed seeded permutation and a run uses the slice ``[doc_start, num_documents)``
    of it; each document's positions are a prefix of a per-document seeded
    permutation keyed by the document's *global* ordinal. So the targets for docs
    ``[0, 500)`` are exactly the first part of the targets for ``[0, 2000)`` -- a
    later run with ``doc_start=500`` adds new, disjoint targets without changing
    (or re-running) the ones already computed. The slice's flat target list is
    sharded strided (``targets[shard_index::num_shards]``).
    """
    if not val_bin.exists():
        raise FileNotFoundError(f"Validation token file not found: {val_bin}")
    if num_documents < 1 or positions_per_document < 1:
        raise ValueError("num_documents and positions_per_document must be >= 1")
    if num_shards < 1 or not (0 <= shard_index < num_shards):
        raise ValueError(f"invalid shard {shard_index}/{num_shards}")

    data = np.memmap(val_bin, dtype=np.uint16, mode="r")
    boundary_ids = tuple(int(t) for t in boundary_token_ids)
    if not boundary_ids:
        raise ValueError("At least one boundary token id is required")
    tbl = _document_table(
        data,
        seqlen=seqlen,
        future_horizon=future_horizon,
        skip_first=skip_first,
        boundary_ids=boundary_ids,
    )
    doc_starts = tbl["doc_starts"]
    doc_ends = tbl["doc_ends"]
    doc_lengths = tbl["doc_lengths"]
    high_by_doc = tbl["high_by_doc"]
    valid_position_counts = tbl["valid_position_counts"]

    if doc_start < 0:
        raise ValueError("doc_start must be >= 0")
    eligible = np.flatnonzero(valid_position_counts >= positions_per_document)
    if eligible.size == 0:
        raise ValueError(
            "No documents have enough boundary-free source positions for "
            f"positions_per_document={positions_per_document} "
            f"(seqlen={seqlen}, future_horizon={future_horizon}, skip_first={skip_first})"
        )
    # One fixed seeded ordering of all eligible docs; a run uses a prefix slice.
    doc_order = np.random.default_rng(data_seed).permutation(eligible)
    doc_end = min(num_documents, doc_order.size)
    if num_documents > doc_order.size:
        print(
            f"WARNING: only {doc_order.size} documents have >= {positions_per_document} "
            f"safe positions; capping num_documents to {doc_order.size}."
        )
    selected = doc_order[doc_start:doc_end]

    # Flat document-major target list: (start, end, length, position). Positions
    # are a prefix of a per-document permutation keyed by the GLOBAL ordinal, so
    # they are identical regardless of which round computes the document.
    starts: list[int] = []
    ends: list[int] = []
    lengths: list[int] = []
    positions: list[int] = []
    for i, doc_idx in enumerate(selected):
        global_ordinal = doc_start + i
        high = int(high_by_doc[doc_idx])
        choices = np.arange(skip_first, high)
        pos_rng = np.random.default_rng(position_seed + global_ordinal)
        chosen_pos = np.sort(pos_rng.permutation(choices)[:positions_per_document])
        for p in chosen_pos:
            starts.append(int(doc_starts[doc_idx]))
            ends.append(int(doc_ends[doc_idx]))
            lengths.append(int(doc_lengths[doc_idx]))
            positions.append(int(p))

    # Shard (strided) then pack into batches of `batch_size` rows.
    sel = range(shard_index, len(starts), num_shards)
    s_starts = [starts[i] for i in sel]
    s_ends = [ends[i] for i in sel]
    s_lengths = [lengths[i] for i in sel]
    s_positions = [positions[i] for i in sel]

    batches: list[ValBatch] = []
    for b in range(0, len(s_starts), batch_size):
        bs = s_starts[b : b + batch_size]
        be = s_ends[b : b + batch_size]
        bl = s_lengths[b : b + batch_size]
        bp = s_positions[b : b + batch_size]
        rows = [
            np.asarray(data[int(st) : int(st) + seqlen + 1], dtype=np.int64)
            for st in bs
        ]
        tokens = torch.from_numpy(np.stack(rows, axis=0)).to(device=device, dtype=torch.long)
        batches.append(
            ValBatch(
                x_BxT=tokens[:, :-1],
                y_BxT=tokens[:, 1:],
                info=SampledBatchInfo(
                    start_offsets=bs,
                    document_end_offsets=be,
                    document_lengths=bl,
                    source_positions_by_sample=[[p] for p in bp],
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


def _valid_per_row_window_loss_sum(
    per_pos_loss_BxT: torch.Tensor,
    valid_mask_BxT: torch.Tensor,
    starts: torch.Tensor,
    width: int,
) -> torch.Tensor | None:
    """Per-row generalization of ``_valid_row_mean_loss_sum``.

    Each row ``r`` uses its own window ``[starts[r], starts[r] + width)`` (vs a
    single shared window). Returns the sum over rows of each row's valid-masked
    mean loss, or ``None`` if no row has a valid position in its window. Used by
    the ``per_row_position`` mode where batch rows carry different source
    positions; for ``width == 1`` this is just the per-row loss at ``starts[r]``.
    """
    B, T = per_pos_loss_BxT.shape
    offs = torch.arange(width, device=per_pos_loss_BxT.device)
    idx = starts[:, None] + offs[None, :]  # (B, width)
    in_range = (idx >= 0) & (idx < T)
    idx_c = idx.clamp(0, T - 1)
    gathered_loss = per_pos_loss_BxT.gather(1, idx_c)
    gathered_mask = valid_mask_BxT.gather(1, idx_c) & in_range
    mask = gathered_mask.float()
    counts = mask.sum(dim=1)
    valid_rows = counts > 0
    if valid_rows.sum().item() <= 0:
        return None
    row_sums = (gathered_loss * mask).sum(dim=1)
    return (row_sums[valid_rows] / counts[valid_rows]).sum()


def _ratio(future_norm: float | None, present_norm: float | None) -> float | None:
    if future_norm is None or present_norm is None:
        return None
    denom = future_norm + present_norm
    if denom <= 0.0:
        return None
    return future_norm / denom


def _fop(future_norm: float | None, present_norm: float | None) -> float | None:
    """Un-normalized future/present ratio (counterpart to ``_ratio``).

    Unlike ``_ratio`` (which yields ``future/(future+present)`` in ``[0,1]``)
    this is the raw ``future/present`` ratio, unbounded in ``[0, inf)``.
    """
    if future_norm is None or present_norm is None or present_norm <= 0.0:
        return None
    return future_norm / present_norm


# ---------------------------------------------------------------------------
# Per-position parameter-gradient contribution metrics
# ---------------------------------------------------------------------------
def _clear_grads(model: nn.Module, subs: list[CapturedSub]) -> None:
    model.zero_grad(set_to_none=True)
    for rec in subs:
        if rec.output is not None:
            rec.output.grad = None


def _rmsnorm_xhat(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Recompute the normalised activation (before the learnable scale)."""
    xf = x.float()
    return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)


@dataclass
class _SubFactor:
    """Position/item-localised forward factor for one submodule.

    For a Linear: ``a2 = ||input[b, hi, :]||^2`` (the contribution scales with
    this). For an RMSNorm: ``xhat`` is the normalised activation vector at the
    position so the elementwise contribution is ``grad_out ⊙ xhat``.
    """

    rec: CapturedSub
    a2: float | None = None
    xhat: torch.Tensor | None = None


def _build_factors(
    subs: list[CapturedSub], item: int, hidden_index: int
) -> dict[tuple[int, str], list[_SubFactor]]:
    """Group submodules by (layer, module_type) with their forward factors."""
    groups: dict[tuple[int, str], list[_SubFactor]] = {}
    for rec in subs:
        if rec.input is None:
            continue
        if rec.kind == "linear":
            a = rec.input[item, hidden_index, :].detach().float()
            factor = _SubFactor(rec=rec, a2=float(a.dot(a).item()))
        elif rec.kind == "embedding":
            # One-hot input (token id) -> forward factor is exactly 1; the
            # contribution norm reduces to ||grad_out|| (see _group_norm_sq).
            factor = _SubFactor(rec=rec, a2=1.0)
        else:  # norm
            xhat = _rmsnorm_xhat(rec.input[item, hidden_index, :], rec.eps or 0.0)
            factor = _SubFactor(rec=rec, xhat=xhat.detach())
        groups.setdefault((rec.layer, rec.module_type), []).append(factor)
    return groups


def _grad_at(rec: CapturedSub, item: int, hidden_index: int) -> torch.Tensor | None:
    g = rec.output.grad if rec.output is not None else None
    if g is None:
        return None
    return g[item, hidden_index, :].detach().float()


def _group_norm_sq(
    factors: list[_SubFactor], item: int, hidden_index: int
) -> float:
    """Squared norm of the per-position contribution vector for one group."""
    total = 0.0
    for f in factors:
        g = _grad_at(f.rec, item, hidden_index)
        if g is None:
            continue
        if f.rec.kind in ("linear", "embedding"):
            gg = float(g.dot(g).item())
            total += gg * (f.a2 or 0.0)
            if f.rec.has_bias:
                total += gg  # bias contribution is grad_out itself
        else:  # norm
            c = g * f.xhat
            total += float(c.dot(c).item())
    return total


def _compute_group_norms(
    factors_by_slot: dict[str, dict[tuple[int, str], list[_SubFactor]]],
    slot_specs: tuple[SlotSpec, ...],
    item: int,
    source_position: int,
) -> dict[str, dict[tuple[int, str], float]]:
    """Read each (layer, module_type) group's per-position contribution norm.

    Called immediately after a backward, while the submodule output ``.grad``s are
    still live, so only scalar norms are kept (no grad snapshots).
    """
    result: dict[str, dict[tuple[int, str], float]] = {}
    for slot in slot_specs:
        hidden_index = slot.hidden_index(source_position)
        slot_map: dict[tuple[int, str], float] = {}
        for key, group in factors_by_slot[slot.slot_id].items():
            slot_map[key] = math.sqrt(max(0.0, _group_norm_sq(group, item, hidden_index)))
        result[slot.slot_id] = slot_map
    return result


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GradientRecord:
    model_id: str
    model_label: str
    checkpoint_path: str
    batch_index: int
    sample_index: int
    source_position: int
    slot_id: str
    slot_label: str
    layer: int
    module_type: str
    present_norm: float
    future_mean_norm: float | None
    future_mean_ratio: float | None
    offset_norms: list[float | None]
    offset_ratios: list[float | None]
    batch_item_index: int = 0
    start_offset: int | None = None
    document_end_offset: int | None = None
    document_length: int | None = None


def _records_for_item(
    *,
    spec: ModelSpec,
    slot_specs: tuple[SlotSpec, ...],
    n_layers: int,
    source_position: int,
    batch_index: int,
    sample_index: int,
    batch_item_index: int,
    start_offset: int,
    document_end_offset: int,
    document_length: int,
    present_norms: dict[str, dict[tuple[int, str], float]],
    future_norms: dict[str, dict[tuple[int, str], float]] | None,
    offset_norms_per_offset: list[dict[str, dict[tuple[int, str], float]]],
    factors_by_slot: dict[str, dict[tuple[int, str], list[_SubFactor]]],
) -> list[GradientRecord]:
    records: list[GradientRecord] = []
    for slot in slot_specs:
        factors = factors_by_slot[slot.slot_id]
        for layer in range(n_layers):
            for module_type in MODULE_TYPES:
                key = (layer, module_type)
                if key not in factors:
                    continue

                present_norm = present_norms[slot.slot_id][key]
                future_norm = (
                    future_norms[slot.slot_id].get(key)
                    if future_norms is not None
                    else None
                )

                offset_norms: list[float | None] = []
                for off in offset_norms_per_offset:
                    snap = off.get(slot.slot_id) if off else None
                    offset_norms.append(snap.get(key) if snap is not None else None)

                records.append(
                    GradientRecord(
                        model_id=spec.kind,
                        model_label=spec.label,
                        checkpoint_path=spec.checkpoint_path,
                        batch_index=batch_index,
                        sample_index=sample_index,
                        source_position=source_position,
                        slot_id=slot.slot_id,
                        slot_label=slot.label,
                        layer=layer,
                        module_type=module_type,
                        present_norm=present_norm,
                        future_mean_norm=future_norm,
                        future_mean_ratio=_ratio(future_norm, present_norm),
                        offset_norms=offset_norms,
                        offset_ratios=[
                            _ratio(off_norm, present_norm) for off_norm in offset_norms
                        ],
                        batch_item_index=batch_item_index,
                        start_offset=start_offset,
                        document_end_offset=document_end_offset,
                        document_length=document_length,
                    )
                )
    return records


def _per_pos_loss(
    adapter: ModelAdapter, x_BxT: torch.Tensor, y_BxT: torch.Tensor, logits: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    eos_token_id = adapter.config.eos_token_id
    pad_token_id = adapter.config.pad_token_id
    valid_mask = (x_BxT != eos_token_id) & (x_BxT != pad_token_id)
    masked_targets = y_BxT.clone()
    masked_targets[~valid_mask] = IGNORE_INDEX
    per_pos_loss = F.cross_entropy(
        logits.float().reshape(-1, logits.size(-1)),
        masked_targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).reshape(y_BxT.shape)
    return per_pos_loss, valid_mask


# ---------------------------------------------------------------------------
# Measurement: serial mode
# ---------------------------------------------------------------------------
def measure_batch(
    *,
    adapter: ModelAdapter,
    spec: ModelSpec,
    x_BxT: torch.Tensor,
    y_BxT: torch.Tensor,
    batch_index: int,
    batch_info: SampledBatchInfo,
    future_horizon: int,
) -> list[GradientRecord]:
    model = adapter.model
    slot_specs = adapter.slot_specs

    model.zero_grad(set_to_none=True)
    with _autocast_context(x_BxT.device):
        subs, logits = adapter.forward_with_capture(x_BxT)

    per_pos_loss, valid_mask = _per_pos_loss(adapter, x_BxT, y_BxT, logits)
    n_layers = int(adapter.config.n_layer)
    records: list[GradientRecord] = []

    sample_index = 0
    for batch_item_index, positions in enumerate(batch_info.source_positions_by_sample):
        item_loss = per_pos_loss[batch_item_index : batch_item_index + 1]
        item_valid_mask = valid_mask[batch_item_index : batch_item_index + 1]
        start_offset = batch_info.start_offsets[batch_item_index]
        document_end_offset = batch_info.document_end_offsets[batch_item_index]
        document_length = batch_info.document_lengths[batch_item_index]

        for source_position in positions:
            present_loss = _valid_loss(
                item_loss, item_valid_mask, source_position, source_position + 1
            )
            if present_loss is None:
                continue

            # Forward factors per slot for this item/position.
            factors_by_slot = {
                slot.slot_id: _build_factors(
                    subs, batch_item_index, slot.hidden_index(source_position)
                )
                for slot in slot_specs
            }

            # --- present backward (read norms immediately) ---
            _clear_grads(model, subs)
            present_loss.backward(retain_graph=True)
            present_norms = _compute_group_norms(
                factors_by_slot, slot_specs, batch_item_index, source_position
            )

            # --- future-mean backward ---
            future_loss = _valid_loss(
                item_loss,
                item_valid_mask,
                source_position + 1,
                source_position + future_horizon + 1,
            )
            future_norms = None
            if future_loss is not None:
                _clear_grads(model, subs)
                future_loss.backward(retain_graph=True)
                future_norms = _compute_group_norms(
                    factors_by_slot, slot_specs, batch_item_index, source_position
                )

            # --- per-offset backward (norms only) ---
            offset_norms_per_offset: list[dict[str, dict[tuple[int, str], float]]] = []
            for offset in range(1, future_horizon + 1):
                offset_loss = _valid_loss(
                    item_loss,
                    item_valid_mask,
                    source_position + offset,
                    source_position + offset + 1,
                )
                if offset_loss is None:
                    offset_norms_per_offset.append({})
                    continue
                _clear_grads(model, subs)
                offset_loss.backward(retain_graph=True)
                offset_norms_per_offset.append(
                    _compute_group_norms(
                        factors_by_slot, slot_specs, batch_item_index, source_position
                    )
                )

            _clear_grads(model, subs)

            records.extend(
                _records_for_item(
                    spec=spec,
                    slot_specs=slot_specs,
                    n_layers=n_layers,
                    source_position=source_position,
                    batch_index=batch_index,
                    sample_index=sample_index,
                    batch_item_index=batch_item_index,
                    start_offset=start_offset,
                    document_end_offset=document_end_offset,
                    document_length=document_length,
                    present_norms=present_norms,
                    future_norms=future_norms,
                    offset_norms_per_offset=offset_norms_per_offset,
                    factors_by_slot=factors_by_slot,
                )
            )
            sample_index += 1

    return records


# ---------------------------------------------------------------------------
# Measurement: shared_position mode (one position shared across the batch)
# ---------------------------------------------------------------------------
def measure_batch_shared_position(
    *,
    adapter: ModelAdapter,
    spec: ModelSpec,
    x_BxT: torch.Tensor,
    y_BxT: torch.Tensor,
    batch_index: int,
    batch_info: SampledBatchInfo,
    future_horizon: int,
) -> list[GradientRecord]:
    model = adapter.model
    slot_specs = adapter.slot_specs

    if not batch_info.source_positions_by_sample:
        return []
    if any(len(positions) != 1 for positions in batch_info.source_positions_by_sample):
        raise ValueError("shared_position mode requires exactly one source position per sample")
    source_positions = [positions[0] for positions in batch_info.source_positions_by_sample]
    source_position = source_positions[0]
    if any(position != source_position for position in source_positions):
        raise ValueError("shared_position mode requires every sample in a batch to share p")

    model.zero_grad(set_to_none=True)
    with _autocast_context(x_BxT.device):
        subs, logits = adapter.forward_with_capture(x_BxT)

    per_pos_loss, valid_mask = _per_pos_loss(adapter, x_BxT, y_BxT, logits)
    batch_size = int(x_BxT.shape[0])
    n_layers = int(adapter.config.n_layer)
    records: list[GradientRecord] = []

    present_loss = _valid_row_mean_loss_sum(
        per_pos_loss, valid_mask, source_position, source_position + 1
    )
    if present_loss is None:
        return records

    # Forward factors per item/slot (depend only on the shared forward pass). The
    # backward is batch-level, so we read every item's per-group norm immediately
    # after each backward (no grad snapshots) and keep only scalars.
    factors_by_item = [
        {
            slot.slot_id: _build_factors(
                subs, batch_item_index, slot.hidden_index(source_position)
            )
            for slot in slot_specs
        }
        for batch_item_index in range(batch_size)
    ]

    def _norms_all_items() -> list[dict[str, dict[tuple[int, str], float]]]:
        return [
            _compute_group_norms(
                factors_by_item[batch_item_index],
                slot_specs,
                batch_item_index,
                source_position,
            )
            for batch_item_index in range(batch_size)
        ]

    _clear_grads(model, subs)
    present_loss.backward(retain_graph=True)
    present_norms_by_item = _norms_all_items()

    future_loss = _valid_row_mean_loss_sum(
        per_pos_loss,
        valid_mask,
        source_position + 1,
        source_position + future_horizon + 1,
    )
    future_norms_by_item: list[dict[str, dict[tuple[int, str], float]]] | None = None
    if future_loss is not None:
        _clear_grads(model, subs)
        future_loss.backward(retain_graph=True)
        future_norms_by_item = _norms_all_items()

    offset_norms_by_offset: list[
        list[dict[str, dict[tuple[int, str], float]]] | None
    ] = []
    for offset in range(1, future_horizon + 1):
        offset_loss = _valid_row_mean_loss_sum(
            per_pos_loss,
            valid_mask,
            source_position + offset,
            source_position + offset + 1,
        )
        if offset_loss is None:
            offset_norms_by_offset.append(None)
            continue
        _clear_grads(model, subs)
        offset_loss.backward(retain_graph=True)
        offset_norms_by_offset.append(_norms_all_items())

    _clear_grads(model, subs)

    # Per-item attribution from the precomputed scalar norms.
    for batch_item_index in range(batch_size):
        offset_norms_per_offset = [
            off[batch_item_index] if off is not None else {}
            for off in offset_norms_by_offset
        ]
        records.extend(
            _records_for_item(
                spec=spec,
                slot_specs=slot_specs,
                n_layers=n_layers,
                source_position=source_position,
                batch_index=batch_index,
                sample_index=batch_item_index,
                batch_item_index=batch_item_index,
                start_offset=batch_info.start_offsets[batch_item_index],
                document_end_offset=batch_info.document_end_offsets[batch_item_index],
                document_length=batch_info.document_lengths[batch_item_index],
                present_norms=present_norms_by_item[batch_item_index],
                future_norms=(
                    future_norms_by_item[batch_item_index]
                    if future_norms_by_item is not None
                    else None
                ),
                offset_norms_per_offset=offset_norms_per_offset,
                factors_by_slot=factors_by_item[batch_item_index],
            )
        )

    return records


# ---------------------------------------------------------------------------
# Measurement: per-row position mode (each row its own source position)
# ---------------------------------------------------------------------------
def measure_batch_per_row_position(
    *,
    adapter: ModelAdapter,
    spec: ModelSpec,
    x_BxT: torch.Tensor,
    y_BxT: torch.Tensor,
    batch_index: int,
    batch_info: SampledBatchInfo,
    future_horizon: int,
) -> list[GradientRecord]:
    """Amortized backward over a batch whose rows carry **different** source
    positions (the multi-position scheme). Identical cost/structure to
    ``measure_batch_shared_position`` -- one backward per (present, future-mean,
    each offset) over the per-row summed loss, with each row's contribution norms
    read from its own forward factors -- but the loss windows are per-row
    (``_valid_per_row_window_loss_sum``) and each row uses its own
    ``hidden_index(p_r)``. Reduces to ``shared_position`` when all rows share p."""
    model = adapter.model
    slot_specs = adapter.slot_specs

    if not batch_info.source_positions_by_sample:
        return []
    if any(len(positions) != 1 for positions in batch_info.source_positions_by_sample):
        raise ValueError("per_row_position mode requires exactly one source position per row")
    row_positions = [positions[0] for positions in batch_info.source_positions_by_sample]

    model.zero_grad(set_to_none=True)
    with _autocast_context(x_BxT.device):
        subs, logits = adapter.forward_with_capture(x_BxT)

    per_pos_loss, valid_mask = _per_pos_loss(adapter, x_BxT, y_BxT, logits)
    batch_size = int(x_BxT.shape[0])
    n_layers = int(adapter.config.n_layer)
    records: list[GradientRecord] = []

    base = torch.as_tensor(row_positions, device=x_BxT.device, dtype=torch.long)

    present_loss = _valid_per_row_window_loss_sum(per_pos_loss, valid_mask, base, 1)
    if present_loss is None:
        return records

    factors_by_item = [
        {
            slot.slot_id: _build_factors(
                subs, batch_item_index, slot.hidden_index(row_positions[batch_item_index])
            )
            for slot in slot_specs
        }
        for batch_item_index in range(batch_size)
    ]

    def _norms_all_items() -> list[dict[str, dict[tuple[int, str], float]]]:
        return [
            _compute_group_norms(
                factors_by_item[batch_item_index],
                slot_specs,
                batch_item_index,
                row_positions[batch_item_index],
            )
            for batch_item_index in range(batch_size)
        ]

    _clear_grads(model, subs)
    present_loss.backward(retain_graph=True)
    present_norms_by_item = _norms_all_items()

    future_loss = _valid_per_row_window_loss_sum(
        per_pos_loss, valid_mask, base + 1, future_horizon
    )
    future_norms_by_item: list[dict[str, dict[tuple[int, str], float]]] | None = None
    if future_loss is not None:
        _clear_grads(model, subs)
        future_loss.backward(retain_graph=True)
        future_norms_by_item = _norms_all_items()

    offset_norms_by_offset: list[
        list[dict[str, dict[tuple[int, str], float]]] | None
    ] = []
    for offset in range(1, future_horizon + 1):
        offset_loss = _valid_per_row_window_loss_sum(
            per_pos_loss, valid_mask, base + offset, 1
        )
        if offset_loss is None:
            offset_norms_by_offset.append(None)
            continue
        _clear_grads(model, subs)
        offset_loss.backward(retain_graph=True)
        offset_norms_by_offset.append(_norms_all_items())

    _clear_grads(model, subs)

    for batch_item_index in range(batch_size):
        offset_norms_per_offset = [
            off[batch_item_index] if off is not None else {}
            for off in offset_norms_by_offset
        ]
        records.extend(
            _records_for_item(
                spec=spec,
                slot_specs=slot_specs,
                n_layers=n_layers,
                source_position=row_positions[batch_item_index],
                batch_index=batch_index,
                sample_index=batch_item_index,
                batch_item_index=batch_item_index,
                start_offset=batch_info.start_offsets[batch_item_index],
                document_end_offset=batch_info.document_end_offsets[batch_item_index],
                document_length=batch_info.document_lengths[batch_item_index],
                present_norms=present_norms_by_item[batch_item_index],
                future_norms=(
                    future_norms_by_item[batch_item_index]
                    if future_norms_by_item is not None
                    else None
                ),
                offset_norms_per_offset=offset_norms_per_offset,
                factors_by_slot=factors_by_item[batch_item_index],
            )
        )

    return records


# ---------------------------------------------------------------------------
# Forward micro-batching (memory control)
# ---------------------------------------------------------------------------
def measure_batch_chunked(
    *,
    adapter: ModelAdapter,
    spec: ModelSpec,
    batch: "ValBatch",
    batch_index: int,
    future_horizon: int,
    mode: str,
    chunk_size: int,
) -> list[GradientRecord]:
    """Run a sampled batch through the measurement in row-chunks of ``chunk_size``.

    Peak memory is dominated by the captured per-submodule activations and the
    retained autograd graph for the rows whose forward is live at once, so
    processing fewer rows per forward bounds the peak. This is *exact*: the batch
    loss (``_valid_row_mean_loss_sum``) is a sum of independent per-row terms and
    transformer rows do not attend across the batch, so each row's contribution
    norms do not depend on which other rows share its forward. We only re-base the
    per-row index back to its global position within the batch so record identity
    (and the aggregate pooling keys) stay unique.
    """
    measure = {
        "shared_position": measure_batch_shared_position,
        "per_row_position": measure_batch_per_row_position,
    }.get(mode, measure_batch)
    n_rows = int(batch.x_BxT.shape[0])
    chunk = n_rows if chunk_size <= 0 else min(chunk_size, n_rows)
    cuda = batch.x_BxT.device.type == "cuda"

    records: list[GradientRecord] = []
    for start in range(0, n_rows, chunk):
        end = min(start + chunk, n_rows)
        info = batch.info
        chunk_info = SampledBatchInfo(
            start_offsets=info.start_offsets[start:end],
            document_end_offsets=info.document_end_offsets[start:end],
            document_lengths=info.document_lengths[start:end],
            source_positions_by_sample=info.source_positions_by_sample[start:end],
        )
        chunk_records = measure(
            adapter=adapter,
            spec=spec,
            x_BxT=batch.x_BxT[start:end],
            y_BxT=batch.y_BxT[start:end],
            batch_index=batch_index,
            batch_info=chunk_info,
            future_horizon=future_horizon,
        )
        if start > 0:
            chunk_records = [
                replace(
                    r,
                    batch_item_index=r.batch_item_index + start,
                    sample_index=r.sample_index + start,
                )
                for r in chunk_records
            ]
        records.extend(chunk_records)
        if cuda:
            torch.cuda.empty_cache()
    return records


# ---------------------------------------------------------------------------
# Aggregation + artifacts
# ---------------------------------------------------------------------------
def _finite_values(values: Iterable[float | None]) -> list[float]:
    finite = []
    for value in values:
        if value is None:
            continue
        if not math.isfinite(value):
            continue
        finite.append(float(value))
    return finite


def _stats(values: Iterable[float | None]) -> dict[str, float | int | None]:
    arr = np.asarray(_finite_values(values), dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": None, "std": None, "p10": None, "p50": None, "p90": None}
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def _mean_or_none(values: Iterable[float | None]) -> float | None:
    finite = _finite_values(values)
    if not finite:
        return None
    return float(np.mean(finite))


def _stats_from_col(col: np.ndarray) -> dict[str, float | int | None]:
    """``_stats`` for a 1-D array that already encodes missing/non-finite as NaN.

    Numerically identical to ``_stats`` (same mean/std/percentile semantics) but
    skips the per-value Python filtering so it can be called on pre-built columns
    of a vectorized aggregation.
    """
    finite = col[np.isfinite(col)]
    if finite.size == 0:
        return {"n": 0, "mean": None, "std": None, "p10": None, "p50": None, "p90": None}
    return {
        "n": int(finite.size),
        "mean": float(finite.mean()),
        "std": float(finite.std()),
        "p10": float(np.percentile(finite, 10)),
        "p50": float(np.percentile(finite, 50)),
        "p90": float(np.percentile(finite, 90)),
    }


def _col_stats_list(mat: np.ndarray) -> list[dict[str, float | int | None]]:
    """Vectorized per-column ``_stats`` for an ``(n_rows, horizon)`` array whose
    missing/non-finite entries are NaN. Returns one dict per column, identical to
    calling ``_stats_from_col`` on each column but with a single set of NaN-aware
    reductions over all columns at once (one ``nanpercentile`` instead of ~3 per
    column)."""
    if mat.ndim != 2:
        raise ValueError("expected a 2-D matrix")
    horizon = mat.shape[1]
    cnt = np.sum(np.isfinite(mat), axis=0)
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")  # all-NaN columns -> NaN, masked below
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        pcts = np.nanpercentile(mat, [10, 50, 90], axis=0)
    out: list[dict[str, float | int | None]] = []
    for k in range(horizon):
        if cnt[k] == 0:
            out.append({"n": 0, "mean": None, "std": None, "p10": None, "p50": None, "p90": None})
        else:
            out.append(
                {
                    "n": int(cnt[k]),
                    "mean": float(mean[k]),
                    "std": float(std[k]),
                    "p10": float(pcts[0, k]),
                    "p50": float(pcts[1, k]),
                    "p90": float(pcts[2, k]),
                }
            )
    return out


def _offset_matrix(
    records: list["GradientRecord"], attr: str, horizon: int
) -> np.ndarray:
    """Stack a per-record list attribute into an ``(n_records, horizon)`` float
    array, padding short rows and mapping ``None``/non-finite to NaN."""
    rows = [getattr(record, attr) for record in records]
    # Fastest path: uniform-length rows with no None convert straight to a float
    # array in one C call (no object round-trip). This is the common case for
    # raw per-module records.
    if rows and all(len(row) == horizon for row in rows):
        try:
            mat = np.array(rows, dtype=np.float64)
            mat[~np.isfinite(mat)] = np.nan
            return mat
        except (TypeError, ValueError):
            pass  # a None slipped in; fall through to the object-array path
    # Fall back: build an object array (handles None / ragged rows), then
    # convert to float with None/non-finite -> NaN.
    if rows and all(len(row) == horizon for row in rows):
        obj = np.array(rows, dtype=object)
    else:
        obj = np.full((len(records), horizon), None, dtype=object)
        for i, row in enumerate(rows):
            k = min(len(row), horizon)
            if k:
                obj[i, :k] = row[:k]
    mat = np.full((len(records), horizon), np.nan, dtype=np.float64)
    present = obj != None  # noqa: E711  (elementwise on object array)
    mat[present] = np.asarray(obj[present], dtype=np.float64)
    mat[~np.isfinite(mat)] = np.nan
    return mat


def _combine_norm(values: Iterable[float | None]) -> float | None:
    """L2-combine per-piece norms: ``sqrt(Σ vᵢ²)`` (None if no finite value).

    The norm of a concatenated vector equals the root-sum-of-squares of its
    pieces' norms, so this folds the per-``(layer, module_type)`` contribution
    norms into the norm of the whole parameter-gradient vector.
    """
    finite = _finite_values(values)
    if not finite:
        return None
    return float(math.sqrt(sum(v * v for v in finite)))


def _pool_records(
    records: list[GradientRecord],
    *,
    keep_layer: bool,
    module_type: str,
) -> list[GradientRecord]:
    """Pool the per-``(layer, module_type)`` contribution norms into a single
    parameter-gradient vector; downstream metrics are derived from the pooled
    norms just as for a single module.

    Two aggregates are built from this helper:

    - ``keep_layer=False`` (``module_type="all"``): pool across *every* layer and
      module at each position -> the whole-parameter-vector ratio. Layers fold
      into the vector, so the synthetic records carry a sentinel ``layer=0`` and
      there is no layer averaging downstream.
    - ``keep_layer=True`` (``module_type="all_layered"``): pool across modules
      *within each layer* -> one pooled vector per ``(layer, position)``. The layer
      axis is kept so the summary's ``layer_mean_future_over_present`` averages the
      per-layer future/present ratio over layers.
    """
    if not records:
        return []

    # Assign each record to a pooling group and remember one base record per
    # group (for the non-norm metadata carried into the synthetic record).
    bases: list[GradientRecord] = []
    gid_by_key: dict[tuple[Any, ...], int] = {}
    gids = np.empty(len(records), dtype=np.intp)
    for i, r in enumerate(records):
        key = (
            r.model_id,
            r.slot_id,
            r.layer if keep_layer else None,
            r.batch_index,
            r.sample_index,
            r.source_position,
            r.batch_item_index,
        )
        gid = gid_by_key.get(key)
        if gid is None:
            gid = len(bases)
            gid_by_key[key] = gid
            bases.append(r)
        gids[i] = gid
    n_groups = len(bases)
    horizon = max(len(r.offset_norms) for r in records)

    # Vectorized L2 combine within each group: sqrt(Σ vᵢ²) over finite vᵢ,
    # NaN where a group has no finite value (== _combine_norm returning None).
    def _grouped_l2(mat: np.ndarray) -> np.ndarray:
        finite = np.isfinite(mat)
        sq = np.where(finite, mat * mat, 0.0)
        sum_sq = np.zeros((n_groups,) + mat.shape[1:], dtype=np.float64)
        cnt = np.zeros((n_groups,) + mat.shape[1:], dtype=np.float64)
        np.add.at(sum_sq, gids, sq)
        np.add.at(cnt, gids, finite.astype(np.float64))
        return np.where(cnt > 0, np.sqrt(sum_sq), np.nan)

    present_in = np.array(
        [r.present_norm if r.present_norm is not None else np.nan for r in records],
        dtype=np.float64,
    )
    fmean_in = np.array(
        [r.future_mean_norm if r.future_mean_norm is not None else np.nan for r in records],
        dtype=np.float64,
    )
    onorms_in = _offset_matrix(records, "offset_norms", horizon)

    g_present = _grouped_l2(present_in)  # (n_groups,)
    g_fmean = _grouped_l2(fmean_in)  # (n_groups,)
    g_onorms = _grouped_l2(onorms_in)  # (n_groups, horizon)

    # Vectorized _ratio: future / (future + present), NaN unless both finite and
    # denom > 0 (matches _ratio's None cases).
    def _grouped_ratio(future: np.ndarray, present: np.ndarray) -> np.ndarray:
        denom = future + present
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = future / denom
        ratio[~(denom > 0.0)] = np.nan
        return ratio

    g_fmean_ratio = _grouped_ratio(g_fmean, g_present)
    g_oratios = _grouped_ratio(g_onorms, g_present[:, None])

    def _f(x: float) -> float | None:
        return float(x) if math.isfinite(x) else None

    def _opt_lists(mat: np.ndarray) -> list[list[float | None]]:
        # (n_groups, horizon) float array -> list of per-group lists with
        # non-finite mapped to None, built in one C-level conversion.
        obj = mat.astype(object)
        obj[~np.isfinite(mat)] = None
        return obj.tolist()

    onorms_lists = _opt_lists(g_onorms)
    oratios_lists = _opt_lists(g_oratios)

    synthesized: list[GradientRecord] = []
    for gi, base in enumerate(bases):
        present_val = g_present[gi]
        synthesized.append(
            GradientRecord(
                model_id=base.model_id,
                model_label=base.model_label,
                checkpoint_path=base.checkpoint_path,
                batch_index=base.batch_index,
                sample_index=base.sample_index,
                source_position=base.source_position,
                slot_id=base.slot_id,
                slot_label=base.slot_label,
                layer=base.layer if keep_layer else 0,
                module_type=module_type,
                present_norm=float(present_val) if math.isfinite(present_val) else 0.0,
                future_mean_norm=_f(g_fmean[gi]),
                future_mean_ratio=_f(g_fmean_ratio[gi]),
                offset_norms=onorms_lists[gi],
                offset_ratios=oratios_lists[gi],
                batch_item_index=base.batch_item_index,
                start_offset=base.start_offset,
                document_end_offset=base.document_end_offset,
                document_length=base.document_length,
            )
        )
    return synthesized


def _col_mean(sub: np.ndarray) -> np.ndarray:
    """Column-wise mean over finite entries (NaN where none), without
    ``np.nanmean``'s empty-slice warnings."""
    cnt = np.sum(~np.isnan(sub), axis=0)
    total = np.nansum(sub, axis=0)
    return np.where(cnt > 0, total / np.where(cnt > 0, cnt, 1), np.nan)


def _module_summary(
    *,
    present: np.ndarray,
    onorms: np.ndarray,
    fmean: np.ndarray,
    layer_ids: np.ndarray,
    horizon: int,
) -> dict[str, Any]:
    """Build one module's ``{"layers", "offsets"}`` summary from column arrays.

    Shared by the record-based (``aggregate_records``) and array-based
    (``aggregate_arrays``) paths so both produce byte-identical summaries.

    - ``present`` (N,), ``fmean`` (N,), ``layer_ids`` (N,): per-record present
      norm, future-mean norm, and layer id (missing -> NaN; layer is an int).
    - ``onorms`` (N, horizon): per-record per-offset future norms (missing -> NaN).
    """
    layers = sorted({int(v) for v in np.unique(layer_ids)})
    layer_rows = []
    for layer in layers:
        m = layer_ids == layer
        layer_rows.append(
            {
                "layer": layer,
                "present_norm": _stats_from_col(present[m]),
                "future_mean_norm": _stats_from_col(fmean[m]),
            }
        )

    # _fop: future/present, NaN unless present > 0.
    present_pos = np.where(present > 0.0, present, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ofop = onorms / present_pos[:, None]
    ofop[~np.isfinite(ofop)] = np.nan

    lm_fop = np.full((len(layers), horizon), np.nan, dtype=np.float64)
    for li, layer in enumerate(layers):
        mask = layer_ids == layer
        if mask.any():
            lm_fop[li] = _col_mean(ofop[mask])

    future_norm_stats = _col_stats_list(onorms)
    fop_stats = _col_stats_list(ofop)
    layer_fop_stats = _col_stats_list(lm_fop)
    offsets = [
        {
            "offset": offset,
            "future_norm": future_norm_stats[offset - 1],
            "future_over_present": fop_stats[offset - 1],
            "layer_mean_future_over_present": layer_fop_stats[offset - 1],
        }
        for offset in range(1, horizon + 1)
    ]
    return {"layers": layer_rows, "offsets": offsets}


def aggregate_records(
    records: list[GradientRecord],
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    horizon = int(metadata["future_horizon"])
    # Add the two whole-parameter aggregates alongside the per-module records;
    # they then flow through the existing per-offset summary unchanged.
    #   "all"         -> pool over all layers & modules (whole-vector ratio)
    #   "all_layered" -> pool over modules per layer, then layer-averaged ratio
    records = (
        list(records)
        + _pool_records(records, keep_layer=False, module_type="all")
        + _pool_records(records, keep_layer=True, module_type="all_layered")
    )
    model_map: dict[str, dict[str, Any]] = {}
    for record in records:
        model = model_map.setdefault(
            record.model_id,
            {
                "model_id": record.model_id,
                "model_label": record.model_label,
                "checkpoint_path": record.checkpoint_path,
                "slots": {},
            },
        )
        slot = model["slots"].setdefault(
            record.slot_id,
            {"slot_id": record.slot_id, "slot_label": record.slot_label, "modules": {}},
        )
        slot["modules"].setdefault(
            record.module_type, {"module_type": record.module_type, "layers": [], "offsets": []}
        )

    for model_id, model in model_map.items():
        model_records = [r for r in records if r.model_id == model_id]
        for slot_id, slot in model["slots"].items():
            slot_records = [r for r in model_records if r.slot_id == slot_id]
            for module_type, module in slot["modules"].items():
                mod_records = [r for r in slot_records if r.module_type == module_type]
                onorms = _offset_matrix(mod_records, "offset_norms", horizon)
                present = np.array(
                    [
                        r.present_norm if r.present_norm is not None else np.nan
                        for r in mod_records
                    ],
                    dtype=np.float64,
                )
                fmean = np.array(
                    [
                        r.future_mean_norm if r.future_mean_norm is not None else np.nan
                        for r in mod_records
                    ],
                    dtype=np.float64,
                )
                layer_ids = np.array([r.layer for r in mod_records])
                summ = _module_summary(
                    present=present,
                    onorms=onorms,
                    fmean=fmean,
                    layer_ids=layer_ids,
                    horizon=horizon,
                )
                module["layers"] = summ["layers"]
                module["offsets"] = summ["offsets"]

    ordered_models = sorted(
        model_map.values(),
        key=lambda model: MODEL_ORDER.index(model["model_id"])
        if model["model_id"] in MODEL_ORDER
        else len(MODEL_ORDER),
    )
    for model in ordered_models:
        slots = sorted(
            model["slots"].values(),
            key=lambda slot: 0 if slot["slot_id"] == "input_token" else 1,
        )
        module_order = (*MODULE_TYPES, "all", "all_layered")
        for slot in slots:
            slot["modules"] = [
                slot["modules"][m] for m in module_order if m in slot["modules"]
            ]
        model["slots"] = slots

    return {
        "metadata": metadata,
        "record_count": len(records),
        "models": ordered_models,
    }


def _grouped_l2_arr(values: np.ndarray, gids: np.ndarray, n_groups: int) -> np.ndarray:
    """L2-combine ``values`` within each group id: ``sqrt(Σ vᵢ²)`` over finite vᵢ,
    NaN where a group has no finite value. Mirrors ``_pool_records._grouped_l2``."""
    finite = np.isfinite(values)
    sq = np.where(finite, values * values, 0.0)
    shape = (n_groups,) + values.shape[1:]
    sum_sq = np.zeros(shape, dtype=np.float64)
    cnt = np.zeros(shape, dtype=np.float64)
    np.add.at(sum_sq, gids, sq)
    np.add.at(cnt, gids, finite.astype(np.float64))
    return np.where(cnt > 0, np.sqrt(sum_sq), np.nan)


def _pool_arrays(
    *,
    present: np.ndarray,
    onorms: np.ndarray,
    fmean: np.ndarray,
    slot_code: np.ndarray,
    layer: np.ndarray,
    target_index: np.ndarray,
    keep_layer: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pool per-(layer, module) contribution norms into whole-vector norms.

    Array analogue of ``_pool_records``. Groups by ``(slot, target[, layer])``
    and L2-combines ``present``/``onorms``/``fmean`` within each group. Returns
    ``(g_present, g_onorms, g_fmean, g_layer)`` where ``g_present`` has NaN groups
    filled with 0.0 (matching ``_pool_records``' synthetic present_norm) and
    ``g_layer`` is the per-group layer (sentinel 0 when ``keep_layer`` is False)."""
    cols = [slot_code, target_index]
    if keep_layer:
        cols.insert(1, layer)
    keys = np.stack(cols, axis=1)
    uniq, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.astype(np.intp, copy=False)
    n_groups = len(uniq)
    g_present = _grouped_l2_arr(present, inv, n_groups)
    g_present = np.where(np.isfinite(g_present), g_present, 0.0)
    g_onorms = _grouped_l2_arr(onorms, inv, n_groups)
    g_fmean = _grouped_l2_arr(fmean, inv, n_groups)
    if keep_layer:
        g_layer = uniq[:, 1].astype(np.int64)
    else:
        g_layer = np.zeros(n_groups, dtype=np.int64)
    return g_present, g_onorms, g_fmean, g_layer


def aggregate_arrays(data: dict[str, Any], *, metadata: dict[str, Any]) -> dict[str, Any]:
    """Build the summary dict from columnar arrays (binary format).

    Produces the exact same structure as ``aggregate_records`` (so the plotters
    and CSV writer are unchanged), but operates on the per-record column arrays
    returned by ``grad_params_io.load_run`` instead of ``GradientRecord`` objects
    -- required at the 16k-target scale where materializing records would OOM.
    """
    horizon = int(metadata["future_horizon"])
    present = np.asarray(data["present_norm"], dtype=np.float64)
    fmean = np.asarray(data["future_mean_norm"], dtype=np.float64)
    onorms = np.asarray(data["offset_norms"], dtype=np.float64)
    slot_code = np.asarray(data["slot_code"])
    module_code = np.asarray(data["module_code"])
    layer = np.asarray(data["layer"])
    target_index = np.asarray(data["target_index"])

    slot_table = data["slot_table"]  # [{"slot_id","slot_label"}, ...] by code
    module_table = data["module_table"]  # [module_type, ...] by code
    model_id = data["model_id"]
    model_label = data["model_label"]
    checkpoint_path = data["checkpoint_path"]

    slots_out: dict[str, dict[str, Any]] = {}
    present_slots = sorted({int(c) for c in np.unique(slot_code)})
    for sc in present_slots:
        smask = slot_code == sc
        slot_id = slot_table[sc]["slot_id"]
        slot_label = slot_table[sc]["slot_label"]
        modules_out: dict[str, dict[str, Any]] = {}

        # Raw per-module-type aggregates.
        for mc in sorted({int(c) for c in np.unique(module_code[smask])}):
            mtype = module_table[mc]
            m = smask & (module_code == mc)
            summ = _module_summary(
                present=present[m],
                onorms=onorms[m],
                fmean=fmean[m],
                layer_ids=layer[m].astype(np.int64),
                horizon=horizon,
            )
            modules_out[mtype] = {"module_type": mtype, **summ}

        # Synthetic whole-vector aggregates.
        for mtype, keep_layer in (("all", False), ("all_layered", True)):
            g_present, g_onorms, g_fmean, g_layer = _pool_arrays(
                present=present[smask],
                onorms=onorms[smask],
                fmean=fmean[smask],
                slot_code=slot_code[smask],
                layer=layer[smask].astype(np.int64),
                target_index=target_index[smask],
                keep_layer=keep_layer,
            )
            summ = _module_summary(
                present=g_present,
                onorms=g_onorms,
                fmean=g_fmean,
                layer_ids=g_layer,
                horizon=horizon,
            )
            modules_out[mtype] = {"module_type": mtype, **summ}

        module_order = (*MODULE_TYPES, "all", "all_layered")
        ordered_modules = [modules_out[m] for m in module_order if m in modules_out]
        slots_out[slot_id] = {
            "slot_id": slot_id,
            "slot_label": slot_label,
            "modules": ordered_modules,
        }

    ordered_slots = sorted(
        slots_out.values(), key=lambda s: 0 if s["slot_id"] == "input_token" else 1
    )
    model = {
        "model_id": model_id,
        "model_label": model_label,
        "checkpoint_path": checkpoint_path,
        "slots": ordered_slots,
    }
    return {
        "metadata": metadata,
        "record_count": int(present.shape[0]),
        "models": [model],
    }


def _json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats with None for JSON compliance."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def write_artifacts(
    *,
    records: list[GradientRecord],
    summary: dict[str, Any],
    output_dir: Path,
    skip_records: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if not skip_records:
        records_path = output_dir / "gradient_analysis_params_records.jsonl"
        with records_path.open("w") as f:
            for record in records:
                json.dump(_json_safe(asdict(record)), f, allow_nan=False)
                f.write("\n")

    summary_path = output_dir / "gradient_analysis_params_summary.json"
    with summary_path.open("w") as f:
        json.dump(_json_safe(summary), f, indent=2, allow_nan=False)

    csv_path = output_dir / "gradient_analysis_params_summary.csv"
    fieldnames = [
        "row_type",
        "model_id",
        "model_label",
        "slot_id",
        "slot_label",
        "module_type",
        "layer",
        "offset",
        "n",
        "present_norm_mean",
        "future_norm_mean",
        "future_over_present_mean",
        "layer_mean_future_over_present_mean",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model in summary["models"]:
            for slot in model["slots"]:
                for module in slot["modules"]:
                    for layer in module["layers"]:
                        writer.writerow(
                            {
                                "row_type": "layer",
                                "model_id": model["model_id"],
                                "model_label": model["model_label"],
                                "slot_id": slot["slot_id"],
                                "slot_label": slot["slot_label"],
                                "module_type": module["module_type"],
                                "layer": layer["layer"],
                                "offset": "",
                                "n": layer["present_norm"]["n"],
                                "present_norm_mean": layer["present_norm"]["mean"],
                                "future_norm_mean": layer["future_mean_norm"]["mean"],
                                "future_over_present_mean": "",
                                "layer_mean_future_over_present_mean": "",
                            }
                        )
                    for offset in module["offsets"]:
                        fop = offset["future_over_present"]
                        layer_fop = offset["layer_mean_future_over_present"]
                        writer.writerow(
                            {
                                "row_type": "offset",
                                "model_id": model["model_id"],
                                "model_label": model["model_label"],
                                "slot_id": slot["slot_id"],
                                "slot_label": slot["slot_label"],
                                "module_type": module["module_type"],
                                "layer": "",
                                "offset": offset["offset"],
                                "n": fop["n"],
                                "present_norm_mean": "",
                                "future_norm_mean": offset["future_norm"]["mean"],
                                "future_over_present_mean": fop["mean"],
                                "layer_mean_future_over_present_mean": layer_fop["mean"],
                            }
                        )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    specs = [parse_model_spec(spec_text) for spec_text in args.spec]
    device = torch.device(args.device)
    val_bin = Path(args.val_bin)
    output_dir = Path(args.output_dir)

    mode = args.batched_gradient_mode
    multipos = mode == "per_row_position"
    output_format = getattr(args, "output_format", "binary")
    num_shards = int(getattr(args, "num_shards", 1))
    shard_index = int(getattr(args, "shard_index", 0))
    num_documents = int(getattr(args, "num_documents", 2000))
    positions_per_document = int(getattr(args, "positions_per_document", 8))
    doc_start = int(getattr(args, "doc_start", 0))

    if args.future_horizon < 1:
        raise ValueError("--future-horizon must be >= 1")
    if args.seqlen <= args.future_horizon:
        raise ValueError("--seqlen must be larger than --future-horizon")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.skip_first < 0:
        raise ValueError("--skip-first must be >= 0")
    if not multipos:
        if args.num_batches < 1:
            raise ValueError("--num-batches must be >= 1")
        if args.positions_per_batch < 1:
            raise ValueError("--positions-per-batch must be >= 1")
        if mode == "shared_position" and args.positions_per_batch != 1:
            raise ValueError(
                "--batched-gradient-mode shared_position requires --positions-per-batch 1"
            )
    binary = output_format == "binary"
    if binary and len(specs) != 1:
        raise ValueError("binary output requires exactly one --spec (one model per dir)")
    sharded = multipos and num_shards > 1

    all_records: list[GradientRecord] = []
    sampling_metadata_by_model: list[dict[str, Any]] = []
    for spec in specs:
        if spec.kind == "sps" and device.type != "cuda":
            raise RuntimeError("SPS measurement requires CUDA/Triton")

        print(f"\n=== {spec.label} ({spec.kind}) ===")
        print(f"Loading checkpoint: {spec.checkpoint_path}")
        loaded = load_checkpoint_model(
            spec, device, warp_specialize=warp_specialize_from_arg(args.warp_specialize)
        )
        if args.seqlen > int(loaded.config.block_size):
            raise ValueError(
                f"seqlen={args.seqlen} exceeds {spec.label} block_size={loaded.config.block_size}"
            )
        print(
            "Config: "
            f"layers={loaded.config.n_layer}, heads={loaded.config.n_head}, "
            f"hidden={loaded.config.hidden_size}, block_size={loaded.config.block_size}"
        )
        if hasattr(loaded.config, "window_size"):
            print(f"Window size: {loaded.config.window_size}")

        adapter = ModelAdapter(loaded)
        boundary_token_ids = (
            int(loaded.config.eos_token_id),
            int(loaded.config.pad_token_id),
        )
        if multipos:
            batches = make_multiposition_batches(
                val_bin=val_bin,
                seqlen=args.seqlen,
                num_documents=num_documents,
                positions_per_document=positions_per_document,
                batch_size=args.batch_size,
                device=device,
                data_seed=args.data_seed,
                position_seed=args.position_seed,
                skip_first=args.skip_first,
                future_horizon=args.future_horizon,
                boundary_token_ids=boundary_token_ids,
                doc_start=doc_start,
                shard_index=shard_index,
                num_shards=num_shards,
            )
        else:
            batches = make_val_batches(
                val_bin=val_bin,
                seqlen=args.seqlen,
                num_batches=args.num_batches,
                batch_size=args.batch_size,
                device=device,
                data_seed=args.data_seed,
                position_seed=args.position_seed,
                positions_per_batch=args.positions_per_batch,
                skip_first=args.skip_first,
                future_horizon=args.future_horizon,
                boundary_token_ids=boundary_token_ids,
                batched_gradient_mode=mode,
            )
        model_records: list[GradientRecord] = []
        # The full per-batch sampling provenance is huge at the 16k-target scale
        # (and the binary per-target arrays already record it), so only keep it
        # for the legacy jsonl path.
        if not binary:
            sampling_metadata_by_model.append(
                {
                    "model_id": spec.kind,
                    "model_label": spec.label,
                    "checkpoint_path": spec.checkpoint_path,
                    "boundary_token_ids": list(boundary_token_ids),
                    "sampled_batches": [asdict(batch.info) for batch in batches],
                }
            )
        for batch_index, batch in enumerate(batches):
            n_positions = sum(len(p) for p in batch.info.source_positions_by_sample)
            print(f"Batch {batch_index + 1}/{len(batches)}: {n_positions} source positions")
            batch_records = measure_batch_chunked(
                adapter=adapter,
                spec=spec,
                batch=batch,
                batch_index=batch_index,
                future_horizon=args.future_horizon,
                mode=mode,
                chunk_size=args.forward_chunk_size,
            )
            model_records.extend(batch_records)
            if device.type == "cuda":
                torch.cuda.empty_cache()

        all_records.extend(model_records)
        print(f"Recorded {len(model_records)} rows for {spec.label}")
        del adapter
        del batches
        del loaded
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metadata = {
        "seqlen": args.seqlen,
        "future_horizon": args.future_horizon,
        "num_batches": args.num_batches,
        "positions_per_batch": args.positions_per_batch,
        "batch_size": args.batch_size,
        "num_documents": num_documents,
        "positions_per_document": positions_per_document,
        "doc_start": doc_start,
        "num_shards": num_shards,
        "device": args.device,
        "val_bin": str(val_bin),
        "data_seed": args.data_seed,
        "position_seed": args.position_seed,
        "skip_first": args.skip_first,
        "output_dir": str(output_dir),
        "model_specs": [asdict(spec) for spec in specs],
        "batched_gradient_mode": mode,
        "forward_chunk_size": int(args.forward_chunk_size),
        "sequence_start_mode": "document_start",
        "source_sampling": (
            "multiposition_uniform"
            if multipos
            else (
                "source_position_weighted_shared_position"
                if mode == "shared_position"
                else "source_position_weighted"
            )
        ),
        "boundary_clean_horizon_windows": True,
        "gradient_space": "parameters",
        "module_types": list(MODULE_TYPES),
        "sampling_by_model": sampling_metadata_by_model,
    }

    if binary:
        # Each run is an append-only document-range "round"; its shards live under
        # <out>/r_<doc_start>_<num_documents>/shard_<idx>. A later round (new doc
        # range) writes a sibling r_* dir and is folded in by re-merging -- the
        # rounds already on disk are never recomputed.
        round_dir = output_dir / (
            getattr(args, "round_name", None) or f"r_{doc_start}_{num_documents}"
        )
        shard_dir = round_dir / f"shard_{shard_index}"
        grad_params_io.write_run(
            shard_dir,
            all_records,
            metadata=metadata,
            horizon=int(args.future_horizon),
            module_types=MODULE_TYPES,
            shard_index=shard_index,
            num_shards=num_shards,
        )
        print(
            f"\nSaved round {round_dir.name} shard {shard_index}/{num_shards} "
            f"({len(all_records)} rows) to {shard_dir}"
        )
        if num_shards == 1:
            # Self-contained single-process run: merge (all rounds) + summarize now.
            return merge_run(output_dir)
        print("Run --merge on the output dir once all shards finish.")
        return {"shard_index": shard_index, "record_count": len(all_records)}

    summary = aggregate_records(all_records, metadata=metadata)
    write_artifacts(records=all_records, summary=summary, output_dir=output_dir)
    print(f"\nSaved analysis artifacts to {output_dir}")
    return summary


def merge_run(output_dir: Path) -> dict[str, Any]:
    """Merge all rounds' shards of a binary run and write its summary (no GPU)."""
    grad_params_io.merge_shards(output_dir)
    data = grad_params_io.load_run(output_dir)
    summary = aggregate_arrays(data, metadata=data["metadata"])
    write_artifacts(records=[], summary=summary, output_dir=output_dir, skip_records=True)
    print(f"Merged + summarized {output_dir} ({summary['record_count']} records)")
    return summary


def resummarize_from_records(output_dir: Path) -> dict[str, Any]:
    """Rebuild the summary JSON/CSV from a run's saved per-position records.

    Used to regenerate existing runs with the new ``module_type="all"`` aggregate
    without recomputing any gradients. Reuses the run's stored metadata and only
    rewrites the summary artifacts (the large records jsonl is left untouched).
    """
    # Binary runs carry their config in metadata.json; aggregate from the arrays.
    if grad_params_io.is_binary_run(output_dir):
        data = grad_params_io.load_run(output_dir)
        summary = aggregate_arrays(data, metadata=data["metadata"])
        write_artifacts(records=[], summary=summary, output_dir=output_dir, skip_records=True)
        print(f"Re-summarized {output_dir} ({summary['record_count']} records, binary)")
        return summary

    summary_path = output_dir / "gradient_analysis_params_summary.json"
    records_path = output_dir / "gradient_analysis_params_records.jsonl"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary (for metadata): {summary_path}")
    if not records_path.exists():
        raise FileNotFoundError(f"Missing records jsonl: {records_path}")

    metadata = json.loads(summary_path.read_text())["metadata"]
    field_names = {f.name for f in fields(GradientRecord)}
    records: list[GradientRecord] = []
    with records_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(GradientRecord(**{k: data[k] for k in field_names if k in data}))

    summary = aggregate_records(records, metadata=metadata)
    write_artifacts(records=records, summary=summary, output_dir=output_dir, skip_records=True)
    print(f"Re-summarized {output_dir} ({len(records)} records)")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure current-vs-future PARAMETER-gradient contributions per position."
    )
    parser.add_argument(
        "--resummarize",
        type=Path,
        default=None,
        help=(
            "Rebuild the summary JSON/CSV from an existing run's saved records "
            "(no GPU / no gradient recompute). Pass the run output directory."
        ),
    )
    parser.add_argument(
        "--spec",
        action="append",
        help="Model spec as kind:label:/path/to/checkpoint.pt. Repeat for each model.",
    )
    parser.add_argument(
        "--merge",
        type=Path,
        default=None,
        help=(
            "Merge a sharded binary run's shard_* subdirs and write its summary "
            "(no GPU). Pass the run output directory."
        ),
    )
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument("--future-horizon", type=int, default=DEFAULT_FUTURE_HORIZON)
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--positions-per-batch", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--num-documents",
        type=int,
        default=2000,
        help="per_row_position mode: number of documents to sample.",
    )
    parser.add_argument(
        "--positions-per-document",
        type=int,
        default=8,
        help="per_row_position mode: distinct safe source positions per document.",
    )
    parser.add_argument(
        "--doc-start",
        type=int,
        default=0,
        help=(
            "per_row_position mode: first document (in the stable seeded order) of "
            "this round. Use >0 to ADD a new disjoint document range later without "
            "recomputing earlier rounds (writes <out>/r_<doc_start>_<num_documents>/)."
        ),
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="Split the multiposition target list into this many shards (array job).",
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Which shard (0-based) this invocation computes.",
    )
    parser.add_argument(
        "--round-name",
        default=None,
        help=(
            "Override the round subdir name (default r_<doc_start>_<num_documents>). "
            "Used to keep runs with different --num-shards in separate dirs "
            "(e.g. r_0_500_s128) so they don't collide on shard index; the merge "
            "globs r_*/shard_* and dedups by target, so they still combine."
        ),
    )
    parser.add_argument(
        "--output-format",
        choices=("binary", "jsonl"),
        default="binary",
        help="binary = columnar .npy run dir (default); jsonl = legacy records.jsonl.",
    )
    parser.add_argument(
        "--forward-chunk-size",
        type=int,
        default=0,
        help=(
            "Process each sampled batch's rows in forward groups of this many "
            "rows to bound peak GPU memory (0 = whole batch at once). Results are "
            "identical; rows are independent and the loss sums per-row terms."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--val-bin", default=str(default_val_bin()))
    parser.add_argument("--output-dir", default="outputs/gradient_analysis_params/main")
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--position-seed", type=int, default=123)
    parser.add_argument("--skip-first", type=int, default=64)
    parser.add_argument(
        "--batched-gradient-mode",
        choices=("serial", "shared_position", "per_row_position"),
        default="serial",
        help=(
            "serial measures one source at a time; shared_position batches rows that "
            "share one source position; per_row_position batches rows with different "
            "per-row positions (the multiposition N-docs x K-positions scheme)."
        ),
    )
    add_warp_specialize_arg(parser)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.merge is not None:
        merge_run(args.merge)
        return
    if args.resummarize is not None:
        resummarize_from_records(args.resummarize)
        return
    if not args.spec:
        parser.error("--spec is required unless --resummarize or --merge is given")
    run_analysis(args)


if __name__ == "__main__":
    main()
