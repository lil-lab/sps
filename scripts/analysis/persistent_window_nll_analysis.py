#!/usr/bin/env python3
"""Measure NLL impact from forcing reduced persistent-token key visibility."""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.analysis import analysis_common as ga
from scripts.analysis import persistent_window_nll_io as pwio


IGNORE_INDEX = ga.IGNORE_INDEX
DEFAULT_SEQLEN = 2048
DEFAULT_POSITION_BIN_SIZE = 32
MODEL_ORDER = ga.MODEL_ORDER
PERSISTENT_KEY_WINDOW_MODEL_KINDS = set(MODEL_ORDER)
TRITON_PERSISTENT_KEY_WINDOW_MODEL_KINDS = {"sps", "delayed_state"}


@dataclass(frozen=True)
class GeneralNLLRecord:
    model_id: str
    model_label: str
    checkpoint_path: str
    batch_index: int
    batch_item_index: int
    applicable: bool
    persistent_key_window: int
    original_normal_window: int | None
    original_persistent_key_window: int | None
    baseline_nll_sum: float
    baseline_token_count: int
    baseline_nll: float | None
    forced_nll_sum: float | None
    forced_token_count: int | None
    forced_nll: float | None
    delta_nll: float | None
    start_offset: int | None = None
    document_end_offset: int | None = None
    document_length: int | None = None


@dataclass(frozen=True)
class PositionNLLRecord:
    model_id: str
    model_label: str
    checkpoint_path: str
    batch_index: int
    batch_item_index: int
    position: int
    applicable: bool
    persistent_key_window: int
    original_normal_window: int | None
    original_persistent_key_window: int | None
    baseline_nll: float
    forced_nll: float | None
    delta_nll: float | None
    start_offset: int | None = None
    document_end_offset: int | None = None
    document_length: int | None = None


def _persistent_window_default() -> int | None:
    raw = os.environ.get("PERSISTENT_WINDOW")
    if raw is None or raw == "":
        return None
    return int(raw)


def _autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _finite_values(values: Iterable[float | None]) -> list[float]:
    finite: list[float] = []
    for value in values:
        if value is None:
            continue
        if not math.isfinite(float(value)):
            continue
        finite.append(float(value))
    return finite


def _stats(values: Iterable[float | None]) -> dict[str, float | int | None]:
    arr = np.asarray(_finite_values(values), dtype=np.float64)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "sem": None,
            "ci95_low": None,
            "ci95_high": None,
            "p10": None,
            "p90": None,
        }

    mean = float(arr.mean())
    std = float(arr.std())
    if arr.size > 1:
        sem = float(arr.std(ddof=1) / math.sqrt(arr.size))
    else:
        sem = 0.0
    half_width = 1.96 * sem
    return {
        "n": int(arr.size),
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def _attention_modules(model: nn.Module) -> list[nn.Module]:
    modules: list[nn.Module] = []
    transformer = getattr(model, "transformer", None)
    blocks = getattr(transformer, "h", None)
    if blocks is None:
        return modules
    for block in blocks:
        attn = getattr(block, "attn", None)
        if attn is not None:
            modules.append(attn)
    return modules


def resolve_attention_windows(model: nn.Module) -> tuple[int | None, int | None]:
    modules = _attention_modules(model)
    if not modules:
        persistent_key = getattr(model, "forced_persistent_key_window", None)
        return None, None if persistent_key is None else int(persistent_key)
    attn = modules[0]
    normal = getattr(attn, "window_size_normal", getattr(attn, "window_size", None))
    persistent_key = getattr(attn, "persistent_key_window", None)
    return (
        None if normal is None else int(normal),
        None if persistent_key is None else int(persistent_key),
    )


@contextlib.contextmanager
def forced_persistent_key_window(
    model: nn.Module,
    *,
    kind: str,
    persistent_key_window: int,
):
    if kind not in PERSISTENT_KEY_WINDOW_MODEL_KINDS:
        yield False
        return

    if kind == "full_attention":
        attr_name = "forced_persistent_key_window"
        had_attr = hasattr(model, attr_name)
        previous = getattr(model, attr_name, None)
        setattr(model, attr_name, int(persistent_key_window))
        try:
            yield True
        finally:
            if had_attr:
                setattr(model, attr_name, previous)
            else:
                delattr(model, attr_name)
        return

    modules = _attention_modules(model)
    if not modules:
        raise ValueError(f"Could not find attention modules for {kind}")

    previous: list[tuple[nn.Module, int | None]] = []
    for attn in modules:
        if not hasattr(attn, "persistent_key_window"):
            raise ValueError(
                f"{attn.__class__.__name__} does not expose persistent_key_window"
            )
        previous.append((attn, getattr(attn, "persistent_key_window")))
        setattr(attn, "persistent_key_window", int(persistent_key_window))

    try:
        yield True
    finally:
        for attn, previous_window in previous:
            setattr(attn, "persistent_key_window", previous_window)


def _forward_logits(adapter: ga.ModelAdapter, x_BxT: torch.Tensor) -> torch.Tensor:
    model = adapter.model
    if adapter.kind in {"sps", "delayed_state"}:
        _is_real, _documents_idx_BxT, documents_idx_Bx2T = (
            model._expand_real_and_document_idx(x_BxT)
        )
        idx_Bx2T = model.add_predict_tokens(x_BxT)
        x_hidden = model.forward_hidden_states(
            idx_Bx2T,
            documents_idx_Bx2T=documents_idx_Bx2T,
        )
        return model.lm_head(x_hidden[:, 1::2])

    dummy_targets = x_BxT.clone()
    x_hidden, _targets, _is_real = model.forward_hidden_states(
        x_BxT,
        dummy_targets,
    )
    return model.lm_head(x_hidden)


def _per_position_nll(
    *,
    adapter: ga.ModelAdapter,
    x_BxT: torch.Tensor,
    y_BxT: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    eos_token_id = adapter.config.eos_token_id
    pad_token_id = adapter.config.pad_token_id
    valid_mask = (x_BxT != eos_token_id) & (x_BxT != pad_token_id)

    with torch.inference_mode(), _autocast_context(x_BxT.device):
        logits = _forward_logits(adapter, x_BxT)
        masked_targets = y_BxT.clone()
        masked_targets[~valid_mask] = IGNORE_INDEX
        per_pos_loss = F.cross_entropy(
            logits.float().reshape(-1, logits.size(-1)),
            masked_targets.reshape(-1),
            ignore_index=IGNORE_INDEX,
            reduction="none",
        ).reshape(y_BxT.shape)

    return per_pos_loss.float(), valid_mask


def _document_position_mask(
    *,
    length: int | None,
    device: torch.device,
    width: int,
) -> torch.Tensor:
    if length is None:
        return torch.ones(width, dtype=torch.bool, device=device)
    limit = max(0, min(width, int(length)))
    mask = torch.zeros(width, dtype=torch.bool, device=device)
    if limit > 0:
        mask[:limit] = True
    return mask


def _mean_loss_for_row(
    per_pos_loss_T: torch.Tensor,
    valid_mask_T: torch.Tensor,
) -> tuple[float, int, float | None]:
    count = int(valid_mask_T.sum().item())
    nll_sum = float(per_pos_loss_T[valid_mask_T].sum().item()) if count > 0 else 0.0
    mean = nll_sum / count if count > 0 else None
    return nll_sum, count, mean


def measure_batch(
    *,
    adapter: ga.ModelAdapter,
    spec: ga.ModelSpec,
    x_BxT: torch.Tensor,
    y_BxT: torch.Tensor,
    batch_index: int,
    batch_info: ga.SampledBatchInfo,
    persistent_key_window: int,
) -> tuple[list[GeneralNLLRecord], list[PositionNLLRecord]]:
    baseline_loss, valid_mask = _per_position_nll(
        adapter=adapter,
        x_BxT=x_BxT,
        y_BxT=y_BxT,
    )
    original_normal_window, original_persistent_key_window = resolve_attention_windows(adapter.model)

    with forced_persistent_key_window(
        adapter.model,
        kind=adapter.kind,
        persistent_key_window=persistent_key_window,
    ) as applicable:
        forced_loss = None
        if applicable:
            forced_loss, _forced_valid_mask = _per_position_nll(
                adapter=adapter,
                x_BxT=x_BxT,
                y_BxT=y_BxT,
            )

    general_records: list[GeneralNLLRecord] = []
    position_records: list[PositionNLLRecord] = []

    for batch_item_index in range(x_BxT.size(0)):
        document_mask = _document_position_mask(
            length=batch_info.document_lengths[batch_item_index],
            device=valid_mask.device,
            width=valid_mask.size(1),
        )
        row_valid = valid_mask[batch_item_index] & document_mask
        row_baseline = baseline_loss[batch_item_index]
        baseline_sum, baseline_count, baseline_nll = _mean_loss_for_row(row_baseline, row_valid)

        forced_sum = None
        forced_count = None
        forced_nll = None
        delta_nll = None
        if forced_loss is not None:
            forced_sum, forced_count, forced_nll = _mean_loss_for_row(
                forced_loss[batch_item_index],
                row_valid,
            )
            if forced_nll is not None and baseline_nll is not None:
                delta_nll = forced_nll - baseline_nll

        start_offset = batch_info.start_offsets[batch_item_index]
        document_end_offset = batch_info.document_end_offsets[batch_item_index]
        document_length = batch_info.document_lengths[batch_item_index]
        general_records.append(
            GeneralNLLRecord(
                model_id=spec.kind,
                model_label=spec.label,
                checkpoint_path=spec.checkpoint_path,
                batch_index=batch_index,
                batch_item_index=batch_item_index,
                applicable=applicable,
                persistent_key_window=persistent_key_window,
                original_normal_window=original_normal_window,
                original_persistent_key_window=original_persistent_key_window,
                baseline_nll_sum=baseline_sum,
                baseline_token_count=baseline_count,
                baseline_nll=baseline_nll,
                forced_nll_sum=forced_sum,
                forced_token_count=forced_count,
                forced_nll=forced_nll,
                delta_nll=delta_nll,
                start_offset=start_offset,
                document_end_offset=document_end_offset,
                document_length=document_length,
            )
        )

        valid_positions = torch.nonzero(row_valid, as_tuple=False).flatten().tolist()
        for position in valid_positions:
            baseline_nll = float(baseline_loss[batch_item_index, position].item())
            forced_nll = (
                None
                if forced_loss is None
                else float(forced_loss[batch_item_index, position].item())
            )
            position_records.append(
                PositionNLLRecord(
                    model_id=spec.kind,
                    model_label=spec.label,
                    checkpoint_path=spec.checkpoint_path,
                    batch_index=batch_index,
                    batch_item_index=batch_item_index,
                    position=int(position),
                    applicable=applicable,
                    persistent_key_window=persistent_key_window,
                    original_normal_window=original_normal_window,
                    original_persistent_key_window=original_persistent_key_window,
                    baseline_nll=baseline_nll,
                    forced_nll=forced_nll,
                    delta_nll=None if forced_nll is None else forced_nll - baseline_nll,
                    start_offset=start_offset,
                    document_end_offset=document_end_offset,
                    document_length=document_length,
                )
            )

    return general_records, position_records


def _make_document_start_batches(
    *,
    val_bin: Path,
    seqlen: int,
    num_batches: int,
    batch_size: int,
    device: torch.device,
    data_seed: int,
    boundary_token_ids: tuple[int, ...],
) -> list[ga.ValBatch]:
    if not val_bin.exists():
        raise FileNotFoundError(f"Validation token file not found: {val_bin}")

    data = np.memmap(val_bin, dtype=np.uint16, mode="r")
    max_start = len(data) - seqlen - 1
    if max_start < 0:
        raise ValueError(f"{val_bin} is too short for seqlen={seqlen}")

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

    candidate_indices = np.flatnonzero(doc_lengths >= seqlen + 1)
    if candidate_indices.size == 0:
        raise ValueError(
            "No document-start spans contain enough boundary-free tokens "
            f"for seqlen={seqlen}; need document_length >= {seqlen + 1}"
        )

    data_rng = np.random.default_rng(data_seed)
    batches: list[ga.ValBatch] = []
    for _batch_index in range(num_batches):
        chosen_doc_indices = data_rng.choice(
            candidate_indices,
            size=batch_size,
            replace=True,
        )
        starts = doc_starts[chosen_doc_indices].astype(np.int64, copy=False)
        ends = doc_ends[chosen_doc_indices].astype(np.int64, copy=False)
        lengths = doc_lengths[chosen_doc_indices].astype(np.int64, copy=False)
        rows = [
            np.asarray(data[int(start) : int(start) + seqlen + 1], dtype=np.int64)
            for start in starts
        ]
        tokens = torch.from_numpy(np.stack(rows, axis=0)).to(device=device, dtype=torch.long)
        batches.append(
            ga.ValBatch(
                x_BxT=tokens[:, :-1],
                y_BxT=tokens[:, 1:],
                info=ga.SampledBatchInfo(
                    start_offsets=starts.astype(int).tolist(),
                    document_end_offsets=ends.astype(int).tolist(),
                    document_lengths=lengths.astype(int).tolist(),
                    source_positions_by_sample=[[] for _ in range(batch_size)],
                ),
            )
        )
    return batches


# ---------------------------------------------------------------------------
# Sharded / incremental binary path (mirrors gradient_analysis_params).
# ---------------------------------------------------------------------------


def _document_table_for_sampling(
    data: np.ndarray, *, seqlen: int, boundary_ids: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-document starts/ends/lengths (a document start is the token after a
    boundary, excluding boundary tokens), matching ``_make_document_start_batches``."""
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
    return doc_starts, doc_ends, doc_lengths


def make_document_shard_batches(
    *,
    val_bin: Path,
    seqlen: int,
    num_documents: int,
    batch_size: int,
    device: torch.device,
    data_seed: int,
    boundary_token_ids: tuple[int, ...],
    doc_start: int = 0,
    shard_index: int = 0,
    num_shards: int = 1,
) -> list[ga.ValBatch]:
    """Sample documents ``[doc_start, num_documents)`` of one fixed seeded
    permutation (WITHOUT replacement) and pack this shard's documents into batches.

    Prefix-stable / incremental, aligned with ``gradient_analysis_params``: the
    eligible documents (those spanning the full ``seqlen + 1`` window) are ordered
    by a single ``data_seed`` permutation; a run uses the slice ``[doc_start,
    num_documents)`` of it, so a later run with ``doc_start=<old num_documents>``
    adds new, disjoint documents without changing (or recomputing) the ones already
    measured. The slice is sharded strided (``selected[shard_index::num_shards]``).
    """
    if not val_bin.exists():
        raise FileNotFoundError(f"Validation token file not found: {val_bin}")
    if num_documents < 1:
        raise ValueError("num_documents must be >= 1")
    if num_shards < 1 or not (0 <= shard_index < num_shards):
        raise ValueError(f"invalid shard {shard_index}/{num_shards}")
    if doc_start < 0:
        raise ValueError("doc_start must be >= 0")

    data = np.memmap(val_bin, dtype=np.uint16, mode="r")
    boundary_ids = tuple(int(t) for t in boundary_token_ids)
    if not boundary_ids:
        raise ValueError("At least one boundary token id is required")
    doc_starts, doc_ends, doc_lengths = _document_table_for_sampling(
        data, seqlen=seqlen, boundary_ids=boundary_ids
    )
    eligible = np.flatnonzero(doc_lengths >= seqlen + 1)
    if eligible.size == 0:
        raise ValueError(
            "No document-start spans contain enough boundary-free tokens "
            f"for seqlen={seqlen}; need document_length >= {seqlen + 1}"
        )
    # One fixed seeded ordering of all eligible docs; a run uses a prefix slice.
    doc_order = np.random.default_rng(data_seed).permutation(eligible)
    doc_end = min(num_documents, doc_order.size)
    if num_documents > doc_order.size:
        print(
            f"WARNING: only {doc_order.size} documents have >= {seqlen + 1} "
            f"boundary-free tokens; capping num_documents to {doc_order.size}."
        )
    selected = doc_order[doc_start:doc_end]
    shard_docs = selected[shard_index::num_shards]

    starts_all = [int(doc_starts[i]) for i in shard_docs]
    ends_all = [int(doc_ends[i]) for i in shard_docs]
    lengths_all = [int(doc_lengths[i]) for i in shard_docs]

    batches: list[ga.ValBatch] = []
    for b in range(0, len(starts_all), batch_size):
        bs = starts_all[b : b + batch_size]
        be = ends_all[b : b + batch_size]
        bl = lengths_all[b : b + batch_size]
        rows = [
            np.asarray(data[int(st) : int(st) + seqlen + 1], dtype=np.int64) for st in bs
        ]
        tokens = torch.from_numpy(np.stack(rows, axis=0)).to(device=device, dtype=torch.long)
        batches.append(
            ga.ValBatch(
                x_BxT=tokens[:, :-1],
                y_BxT=tokens[:, 1:],
                info=ga.SampledBatchInfo(
                    start_offsets=bs,
                    document_end_offsets=be,
                    document_lengths=bl,
                    source_positions_by_sample=[[] for _ in range(len(bs))],
                ),
            )
        )
    return batches


def _measure_losses(
    *,
    adapter: ga.ModelAdapter,
    x_BxT: torch.Tensor,
    y_BxT: torch.Tensor,
    persistent_key_window: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, bool]:
    """Baseline + forced per-position loss (B, T) and the shared validity mask."""
    baseline_loss, valid_mask = _per_position_nll(adapter=adapter, x_BxT=x_BxT, y_BxT=y_BxT)
    with forced_persistent_key_window(
        adapter.model,
        kind=adapter.kind,
        persistent_key_window=persistent_key_window,
    ) as applicable:
        forced_loss = None
        if applicable:
            forced_loss, _ = _per_position_nll(adapter=adapter, x_BxT=x_BxT, y_BxT=y_BxT)
    return baseline_loss, forced_loss, valid_mask, applicable


def measure_batch_columns(
    *,
    adapter: ga.ModelAdapter,
    x_BxT: torch.Tensor,
    y_BxT: torch.Tensor,
    batch_info: ga.SampledBatchInfo,
    persistent_key_window: int,
) -> tuple[dict[str, np.ndarray], bool]:
    """Per-(document, position) baseline/forced NLL as float32 arrays (NaN = invalid).

    The columnar analog of ``measure_batch``: it skips building millions of
    per-position record objects (prohibitive at large document counts) and returns
    the dense ``(B, T)`` matrices the binary format stores directly.
    """
    baseline_loss, forced_loss, valid_mask, applicable = _measure_losses(
        adapter=adapter,
        x_BxT=x_BxT,
        y_BxT=y_BxT,
        persistent_key_window=persistent_key_window,
    )
    width = baseline_loss.size(1)
    baseline = baseline_loss.detach().to(torch.float32).cpu().numpy().copy()
    if forced_loss is not None:
        forced = forced_loss.detach().to(torch.float32).cpu().numpy().copy()
    else:
        forced = np.full(baseline.shape, np.nan, dtype=np.float32)

    for i in range(baseline.shape[0]):
        document_mask = _document_position_mask(
            length=batch_info.document_lengths[i],
            device=valid_mask.device,
            width=width,
        )
        row_valid = (valid_mask[i] & document_mask).detach().cpu().numpy()
        baseline[i, ~row_valid] = np.nan
        forced[i, ~row_valid] = np.nan

    cols = {
        "baseline_nll": baseline.astype(np.float32, copy=False),
        "forced_nll": forced.astype(np.float32, copy=False),
        "doc_start_offset": np.asarray(batch_info.start_offsets, dtype=np.int64),
        "doc_end_offset": np.asarray(batch_info.document_end_offsets, dtype=np.int64),
        "doc_length": np.asarray(batch_info.document_lengths, dtype=np.int32),
    }
    return cols, applicable


def _empty_stats() -> dict[str, Any]:
    return {
        "n": 0,
        "mean": None,
        "std": None,
        "sem": None,
        "ci95_low": None,
        "ci95_high": None,
        "p10": None,
        "p90": None,
    }


def _stats_array(values: np.ndarray) -> dict[str, Any]:
    """``_stats`` over an ndarray of any shape (NaN-aware), used for bins/windows."""
    arr = np.asarray(values, dtype=np.float64).ravel()
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return _empty_stats()
    mean = float(arr.mean())
    std = float(arr.std())
    sem = float(arr.std(ddof=1) / math.sqrt(arr.size)) if arr.size > 1 else 0.0
    half_width = 1.96 * sem
    return {
        "n": int(arr.size),
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
    }


def _col_stats(mat: np.ndarray) -> list[dict[str, Any]]:
    """Per-column (per-position) ``_stats`` over the document axis (NaN-aware)."""
    rows, cols = mat.shape
    if rows == 0:
        return [_empty_stats() for _ in range(cols)]
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        finite = np.isfinite(mat)
        n = finite.sum(axis=0).astype(np.int64)
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        std_ddof1 = np.nanstd(mat, axis=0, ddof=1)
        pcts = np.nanpercentile(mat, [10.0, 90.0], axis=0)
    out: list[dict[str, Any]] = []
    for j in range(cols):
        nj = int(n[j])
        if nj == 0:
            out.append(_empty_stats())
            continue
        m = float(mean[j])
        sd = float(std[j])
        sem = float(std_ddof1[j] / math.sqrt(nj)) if nj > 1 and math.isfinite(std_ddof1[j]) else 0.0
        half_width = 1.96 * sem
        out.append(
            {
                "n": nj,
                "mean": m,
                "std": sd,
                "sem": sem,
                "ci95_low": m - half_width,
                "ci95_high": m + half_width,
                "p10": float(pcts[0, j]),
                "p90": float(pcts[1, j]),
            }
        )
    return out


def aggregate_arrays(data: dict[str, Any], *, metadata: dict[str, Any]) -> dict[str, Any]:
    """Build a single-model summary dict from binary columns.

    Produces the same per-model structure as ``aggregate_records`` (positions,
    position_bins, pre/post_window, general) so the figure/table consumers and the
    CSV writer are format-agnostic.
    """
    meta = data["metadata"]
    seqlen = int(metadata.get("seqlen") or meta.get("seqlen") or data["baseline_nll"].shape[1])
    position_bin_size = int(
        metadata.get("position_bin_size", meta.get("position_bin_size", DEFAULT_POSITION_BIN_SIZE))
    )
    if position_bin_size < 1:
        raise ValueError("position_bin_size must be >= 1")
    persistent_window = int(
        meta.get("persistent_key_window", metadata.get("persistent_key_window", 0))
    )

    baseline = np.asarray(data["baseline_nll"], dtype=np.float64)
    forced = np.asarray(data["forced_nll"], dtype=np.float64)
    delta = forced - baseline
    applicable = bool(meta.get("applicable", np.isfinite(forced).any()))

    base_cols = _col_stats(baseline)
    forced_cols = _col_stats(forced)
    delta_cols = _col_stats(delta)
    positions = [
        {
            "position": p,
            "baseline_nll": base_cols[p],
            "forced_nll": forced_cols[p],
            "delta_nll": delta_cols[p],
        }
        for p in range(seqlen)
    ]

    position_bins: list[dict[str, Any]] = []
    for start in range(0, seqlen, position_bin_size):
        end = min(seqlen - 1, start + position_bin_size - 1)
        sl = slice(start, end + 1)
        position_bins.append(
            {
                "bin_start": start,
                "bin_end": end,
                "bin_midpoint": (start + end) / 2.0,
                "baseline_nll": _stats_array(baseline[:, sl]),
                "forced_nll": _stats_array(forced[:, sl]),
                "delta_nll": _stats_array(delta[:, sl]),
            }
        )

    pre = slice(0, persistent_window + 1)
    post = slice(persistent_window + 1, seqlen)
    pre_window = {
        "baseline_nll": _stats_array(baseline[:, pre]),
        "forced_nll": _stats_array(forced[:, pre]),
        "delta_nll": _stats_array(delta[:, pre]),
    }
    post_window = {
        "baseline_nll": _stats_array(baseline[:, post]),
        "forced_nll": _stats_array(forced[:, post]),
        "delta_nll": _stats_array(delta[:, post]),
    }

    with np.errstate(all="ignore"):
        doc_delta = np.nanmean(delta, axis=1) if delta.shape[0] else np.zeros(0)
    baseline_token_count = int(np.isfinite(baseline).sum())
    forced_token_count = int(np.isfinite(forced).sum())
    overall_baseline = float(np.nanmean(baseline)) if baseline_token_count else None
    overall_forced = float(np.nanmean(forced)) if forced_token_count else None
    general = {
        "n": int(baseline.shape[0]),
        "baseline_token_count": baseline_token_count,
        "baseline_nll_sum": float(np.nansum(baseline)),
        "baseline_nll": overall_baseline,
        "forced_token_count": forced_token_count if applicable else None,
        "forced_nll_sum": float(np.nansum(forced)) if applicable else None,
        "forced_nll": overall_forced if applicable else None,
        "delta_nll": (
            overall_forced - overall_baseline
            if applicable and overall_baseline is not None and overall_forced is not None
            else None
        ),
        "row_delta_nll": _stats_array(doc_delta),
    }

    return {
        "model_key": "::".join(
            [
                str(meta.get("model_id")),
                str(meta.get("model_label")),
                str(meta.get("checkpoint_path")),
            ]
        ),
        "model_id": meta.get("model_id"),
        "model_label": meta.get("model_label"),
        "checkpoint_path": meta.get("checkpoint_path"),
        "applicable": applicable,
        "persistent_key_window": persistent_window,
        "original_normal_window": meta.get("original_normal_window"),
        "original_persistent_key_window": meta.get("original_persistent_key_window"),
        "general": general,
        "positions": positions,
        "position_bins": position_bins,
        "pre_window": pre_window,
        "post_window": post_window,
    }


def _summary_metadata(metas: list[dict[str, Any]]) -> dict[str, Any]:
    """Combined-summary metadata from the per-model run metadatas (shared config)."""
    base = dict(metas[0]) if metas else {}
    for k in (
        "model_id",
        "model_label",
        "checkpoint_path",
        "shard_index",
        "num_shards",
        "num_shards_merged",
        "document_count",
        "format",
        "schema_version",
        "applicable",
        "original_normal_window",
        "original_persistent_key_window",
    ):
        base.pop(k, None)
    base["model_specs"] = [
        {
            "kind": m.get("model_id"),
            "label": m.get("model_label"),
            "checkpoint_path": m.get("checkpoint_path"),
        }
        for m in metas
    ]
    base["document_counts"] = {
        str(m.get("model_id")): int(m.get("document_count", 0)) for m in metas
    }
    return base


def _write_summary(summary: dict[str, Any], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "persistent_window_nll_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    _write_summary_csv(summary, out_dir / "persistent_window_nll_summary.csv")


def merge_run(model_dir: Path) -> dict[str, Any]:
    """Merge a single model's shards (all rounds) and write its single-model summary."""
    model_dir = Path(model_dir)
    pwio.merge_shards(model_dir)
    data = pwio.load_run(model_dir)
    model = aggregate_arrays(data, metadata=data["metadata"])
    summary = {
        "metadata": _summary_metadata([data["metadata"]]),
        "general_record_count": 0,
        "position_record_count": 0,
        "models": [model],
    }
    _write_summary(summary, model_dir)
    print(f"Merged + summarized {model_dir} ({model['general']['n']} documents)")
    return summary


def combine_scale(
    scale_dir: Path,
    *,
    model_subdirs: list[str],
    persistent_window: int,
    position_bin_size: int | None = None,
) -> dict[str, Any]:
    """Merge each model's shards and write the combined per-scale summary the plot reads.

    Writes ``<scale_dir>/pw<persistent_window>/persistent_window_nll_summary.json``
    with one entry per model, mirroring the legacy multi-spec summary layout.
    """
    scale_dir = Path(scale_dir)
    models: list[dict[str, Any]] = []
    metas: list[dict[str, Any]] = []
    for sub in model_subdirs:
        md = scale_dir / sub
        if not pwio.is_binary_run(md) and not pwio._shard_dirs(md):
            print(f"combine: skipping {md} (no shards / not a binary run)")
            continue
        pwio.merge_shards(md)
        data = pwio.load_run(md)
        run_meta = dict(data["metadata"])
        if position_bin_size is not None:
            run_meta["position_bin_size"] = int(position_bin_size)
        metas.append(run_meta)
        models.append(aggregate_arrays(data, metadata=run_meta))
    if not models:
        raise FileNotFoundError(f"combine: no model data under {scale_dir}")
    models.sort(
        key=lambda m: _model_sort_key(
            (m["model_id"], m["model_label"], m["checkpoint_path"])
        )
    )
    out_dir = scale_dir / f"pw{int(persistent_window)}"
    summary = {
        "metadata": _summary_metadata(metas),
        "general_record_count": 0,
        "position_record_count": 0,
        "models": models,
    }
    _write_summary(summary, out_dir)
    docs = ", ".join(f"{m['model_id']}={m['general']['n']}" for m in models)
    print(f"Combined {len(models)} models -> {out_dir} (documents: {docs})")
    return summary


def _binary_run_metadata(
    *,
    args: argparse.Namespace,
    spec: ga.ModelSpec,
    normal_window: int | None,
    persistent_key_window_original: int | None,
    applicable: bool,
    doc_start: int,
    num_documents: int,
    num_shards: int,
) -> dict[str, Any]:
    return {
        "model_id": spec.kind,
        "model_label": spec.label,
        "checkpoint_path": spec.checkpoint_path,
        "seqlen": int(args.seqlen),
        "persistent_key_window": int(args.persistent_window),
        "position_bin_size": int(args.position_bin_size),
        "original_normal_window": normal_window,
        "original_persistent_key_window": persistent_key_window_original,
        "applicable": bool(applicable),
        "num_documents": int(num_documents),
        "doc_start": int(doc_start),
        "num_shards": int(num_shards),
        "device": args.device,
        "val_bin": str(args.val_bin),
        "data_seed": int(args.data_seed),
        "sequence_start_mode": "document_start",
        "document_sampling": "uniform_document_permutation_without_replacement_prefix_slice",
        "source_position_sampling": "disabled",
        "position_axis": "document_relative_query_position",
        "delta_definition": "forced_nll_minus_baseline_nll",
        "forced_window_target": "persistent_token_keys",
        "confidence_interval": "normal_approx_95_percent",
    }


def run_analysis_binary(args: argparse.Namespace) -> dict[str, Any]:
    """Compute one ``(model, shard)`` of the sharded binary run and publish it."""
    specs = [ga.parse_model_spec(spec_text) for spec_text in args.spec]
    if len(specs) != 1:
        raise ValueError("binary output requires exactly one --spec (one model per dir)")
    spec = specs[0]
    if args.persistent_window is None or args.persistent_window < 0:
        raise ValueError("--persistent-window (>=0) is required")
    if args.num_documents is None or args.num_documents < 1:
        raise ValueError("--num-documents (>=1) is required for binary output")
    if args.seqlen < 2:
        raise ValueError("--seqlen must be >= 2")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.position_bin_size < 1:
        raise ValueError("--position-bin-size must be >= 1")

    num_shards = int(args.num_shards)
    shard_index = int(args.shard_index)
    doc_start = int(args.doc_start)
    num_documents = int(args.num_documents)
    device = torch.device(args.device)
    val_bin = Path(args.val_bin)
    output_dir = Path(args.output_dir)

    if spec.kind in TRITON_PERSISTENT_KEY_WINDOW_MODEL_KINDS and device.type != "cuda":
        raise RuntimeError(f"{spec.kind} measurement requires CUDA/Triton")

    print(f"\n=== {spec.label} ({spec.kind}) shard {shard_index}/{num_shards} ===")
    loaded = ga.load_checkpoint_model(
        spec, device, warp_specialize=ga.warp_specialize_from_arg(args.warp_specialize)
    )
    if args.seqlen > int(loaded.config.block_size):
        raise ValueError(
            f"seqlen={args.seqlen} exceeds {spec.label} block_size={loaded.config.block_size}"
        )
    normal_window, persistent_key_window_original = resolve_attention_windows(loaded.model)
    adapter = ga.ModelAdapter(loaded)
    boundary_token_ids = (
        int(loaded.config.eos_token_id),
        int(loaded.config.pad_token_id),
    )
    batches = make_document_shard_batches(
        val_bin=val_bin,
        seqlen=args.seqlen,
        num_documents=num_documents,
        batch_size=args.batch_size,
        device=device,
        data_seed=args.data_seed,
        boundary_token_ids=boundary_token_ids,
        doc_start=doc_start,
        shard_index=shard_index,
        num_shards=num_shards,
    )

    col_parts: dict[str, list[np.ndarray]] = {
        "baseline_nll": [],
        "forced_nll": [],
        "doc_start_offset": [],
        "doc_end_offset": [],
        "doc_length": [],
    }
    any_applicable = False
    for batch_index, batch in enumerate(batches):
        cols, applicable = measure_batch_columns(
            adapter=adapter,
            x_BxT=batch.x_BxT,
            y_BxT=batch.y_BxT,
            batch_info=batch.info,
            persistent_key_window=args.persistent_window,
        )
        any_applicable = any_applicable or applicable
        for k in col_parts:
            col_parts[k].append(cols[k])
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"shard {shard_index}/{num_shards} batch {batch_index + 1}/{len(batches)}: "
            f"{batch.x_BxT.size(0)} documents"
        )

    if batches:
        merged_cols = {
            "baseline_nll": np.concatenate(col_parts["baseline_nll"], axis=0),
            "forced_nll": np.concatenate(col_parts["forced_nll"], axis=0),
            "doc_start_offset": np.concatenate(col_parts["doc_start_offset"]),
            "doc_end_offset": np.concatenate(col_parts["doc_end_offset"]),
            "doc_length": np.concatenate(col_parts["doc_length"]),
        }
    else:
        # More shards than documents in the slice -> publish an empty shard so the
        # work-pool still marks it done.
        merged_cols = {
            "baseline_nll": np.zeros((0, args.seqlen), dtype=np.float32),
            "forced_nll": np.zeros((0, args.seqlen), dtype=np.float32),
            "doc_start_offset": np.zeros((0,), dtype=np.int64),
            "doc_end_offset": np.zeros((0,), dtype=np.int64),
            "doc_length": np.zeros((0,), dtype=np.int32),
        }

    metadata = _binary_run_metadata(
        args=args,
        spec=spec,
        normal_window=normal_window,
        persistent_key_window_original=persistent_key_window_original,
        applicable=any_applicable,
        doc_start=doc_start,
        num_documents=num_documents,
        num_shards=num_shards,
    )
    round_name = getattr(args, "round_name", None) or f"r_{doc_start}_{num_documents}_s{num_shards}"
    shard_dir = output_dir / round_name / f"shard_{shard_index}"
    pwio.write_shard(
        shard_dir,
        merged_cols,
        metadata=metadata,
        shard_index=shard_index,
        num_shards=num_shards,
    )
    print(
        f"\nSaved round {round_name} shard {shard_index}/{num_shards} "
        f"({merged_cols['baseline_nll'].shape[0]} documents) to {shard_dir}"
    )

    del adapter, loaded, batches
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if num_shards == 1:
        return merge_run(output_dir)
    print("Run --combine on the scale dir (or --merge on this model dir) once all shards finish.")
    return {"shard_index": shard_index, "document_count": int(merged_cols["baseline_nll"].shape[0])}


def _aggregate_general(records: list[GeneralNLLRecord]) -> dict[str, Any]:
    baseline_count = sum(record.baseline_token_count for record in records)
    baseline_sum = sum(record.baseline_nll_sum for record in records)
    forced_records = [record for record in records if record.forced_nll_sum is not None]
    forced_count = sum(int(record.forced_token_count or 0) for record in forced_records)
    forced_sum = sum(float(record.forced_nll_sum or 0.0) for record in forced_records)

    baseline_nll = baseline_sum / baseline_count if baseline_count > 0 else None
    forced_nll = forced_sum / forced_count if forced_count > 0 else None
    delta_nll = (
        None
        if baseline_nll is None or forced_nll is None
        else forced_nll - baseline_nll
    )
    return {
        "n": len(records),
        "baseline_token_count": int(baseline_count),
        "baseline_nll_sum": float(baseline_sum),
        "baseline_nll": baseline_nll,
        "forced_token_count": int(forced_count) if forced_records else None,
        "forced_nll_sum": float(forced_sum) if forced_records else None,
        "forced_nll": forced_nll,
        "delta_nll": delta_nll,
        "row_delta_nll": _stats(record.delta_nll for record in records),
    }


def _model_key(record: GeneralNLLRecord | PositionNLLRecord) -> tuple[str, str, str]:
    return (record.model_id, record.model_label, record.checkpoint_path)


def _model_sort_key(key: tuple[str, str, str]) -> tuple[int, str, str]:
    model_id, model_label, checkpoint_path = key
    order = MODEL_ORDER.index(model_id) if model_id in MODEL_ORDER else len(MODEL_ORDER)
    return (order, model_label, checkpoint_path)


def _stat_block(records: list[PositionNLLRecord]) -> dict[str, Any]:
    return {
        "baseline_nll": _stats(record.baseline_nll for record in records),
        "forced_nll": _stats(record.forced_nll for record in records),
        "delta_nll": _stats(record.delta_nll for record in records),
    }


def aggregate_records(
    general_records: list[GeneralNLLRecord],
    position_records: list[PositionNLLRecord],
    *,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    seqlen = int(metadata.get("seqlen") or 0)
    if seqlen <= 0 and position_records:
        seqlen = max(record.position for record in position_records) + 1
    position_bin_size = int(metadata.get("position_bin_size", DEFAULT_POSITION_BIN_SIZE))
    if position_bin_size < 1:
        raise ValueError("position_bin_size must be >= 1")

    model_keys = sorted(
        {_model_key(record) for record in general_records}
        | {_model_key(record) for record in position_records},
        key=_model_sort_key,
    )

    models: list[dict[str, Any]] = []
    for key in model_keys:
        model_general = [record for record in general_records if _model_key(record) == key]
        model_positions = [record for record in position_records if _model_key(record) == key]
        exemplar = model_general[0] if model_general else model_positions[0]
        persistent_window = int(exemplar.persistent_key_window)

        positions = []
        for position in range(seqlen):
            records_at_position = [
                record for record in model_positions if record.position == position
            ]
            positions.append({"position": position, **_stat_block(records_at_position)})

        position_bins = []
        for start in range(0, seqlen, position_bin_size):
            end = min(seqlen - 1, start + position_bin_size - 1)
            bin_records = [
                record
                for record in model_positions
                if start <= record.position <= end
            ]
            position_bins.append(
                {
                    "bin_start": start,
                    "bin_end": end,
                    "bin_midpoint": (start + end) / 2.0,
                    **_stat_block(bin_records),
                }
            )

        pre_window_records = [
            record for record in model_positions if record.position <= persistent_window
        ]
        post_window_records = [
            record for record in model_positions if record.position > persistent_window
        ]
        models.append(
            {
                "model_key": "::".join(key),
                "model_id": exemplar.model_id,
                "model_label": exemplar.model_label,
                "checkpoint_path": exemplar.checkpoint_path,
                "applicable": any(record.applicable for record in model_general + model_positions),
                "persistent_key_window": exemplar.persistent_key_window,
                "original_normal_window": exemplar.original_normal_window,
                "original_persistent_key_window": exemplar.original_persistent_key_window,
                "general": _aggregate_general(model_general),
                "positions": positions,
                "position_bins": position_bins,
                "pre_window": _stat_block(pre_window_records),
                "post_window": _stat_block(post_window_records),
            }
        )

    return {
        "metadata": metadata,
        "general_record_count": len(general_records),
        "position_record_count": len(position_records),
        "models": models,
    }


def _write_stats_row(
    writer: csv.DictWriter,
    *,
    row_type: str,
    model: dict[str, Any],
    stats: dict[str, Any],
    token_count: int | str = "",
    position: int | str = "",
    bin_start: int | str = "",
    bin_end: int | str = "",
) -> None:
    delta = stats["delta_nll"]
    writer.writerow(
        {
            "row_type": row_type,
            "model_id": model["model_id"],
            "model_label": model["model_label"],
            "applicable": model["applicable"],
            "persistent_key_window": model["persistent_key_window"],
            "original_normal_window": model["original_normal_window"],
            "original_persistent_key_window": model["original_persistent_key_window"],
            "position": position,
            "bin_start": bin_start,
            "bin_end": bin_end,
            "n": delta["n"],
            "token_count": token_count,
            "baseline_nll_mean": stats["baseline_nll"]["mean"],
            "forced_nll_mean": stats["forced_nll"]["mean"],
            "delta_nll_mean": delta["mean"],
            "delta_nll_std": delta["std"],
            "delta_nll_sem": delta["sem"],
            "delta_nll_ci95_low": delta["ci95_low"],
            "delta_nll_ci95_high": delta["ci95_high"],
            "delta_nll_p10": delta["p10"],
            "delta_nll_p90": delta["p90"],
        }
    )


def _write_summary_csv(summary: dict[str, Any], csv_path: Path) -> None:
    """Write the long-format per-model/position/bin CSV from a summary dict.

    Shared by the legacy records path (``write_artifacts``) and the binary
    merge/combine path so both produce identical CSVs.
    """
    fieldnames = [
        "row_type",
        "model_id",
        "model_label",
        "applicable",
        "persistent_key_window",
        "original_normal_window",
        "original_persistent_key_window",
        "position",
        "bin_start",
        "bin_end",
        "n",
        "token_count",
        "baseline_nll_mean",
        "forced_nll_mean",
        "delta_nll_mean",
        "delta_nll_std",
        "delta_nll_sem",
        "delta_nll_ci95_low",
        "delta_nll_ci95_high",
        "delta_nll_p10",
        "delta_nll_p90",
    ]
    with Path(csv_path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model in summary["models"]:
            general = model["general"]
            writer.writerow(
                {
                    "row_type": "general",
                    "model_id": model["model_id"],
                    "model_label": model["model_label"],
                    "applicable": model["applicable"],
                    "persistent_key_window": model["persistent_key_window"],
                    "original_normal_window": model["original_normal_window"],
                    "original_persistent_key_window": model["original_persistent_key_window"],
                    "position": "",
                    "bin_start": "",
                    "bin_end": "",
                    "n": general["n"],
                    "token_count": general["baseline_token_count"],
                    "baseline_nll_mean": general["baseline_nll"],
                    "forced_nll_mean": general["forced_nll"],
                    "delta_nll_mean": general["delta_nll"],
                    "delta_nll_std": general["row_delta_nll"]["std"],
                    "delta_nll_sem": general["row_delta_nll"]["sem"],
                    "delta_nll_ci95_low": general["row_delta_nll"]["ci95_low"],
                    "delta_nll_ci95_high": general["row_delta_nll"]["ci95_high"],
                    "delta_nll_p10": general["row_delta_nll"]["p10"],
                    "delta_nll_p90": general["row_delta_nll"]["p90"],
                }
            )
            _write_stats_row(
                writer,
                row_type="pre_window",
                model=model,
                stats=model["pre_window"],
            )
            _write_stats_row(
                writer,
                row_type="post_window",
                model=model,
                stats=model["post_window"],
            )
            for row in model["positions"]:
                _write_stats_row(
                    writer,
                    row_type="position",
                    model=model,
                    stats=row,
                    position=row["position"],
                )
            for row in model["position_bins"]:
                _write_stats_row(
                    writer,
                    row_type="position_bin",
                    model=model,
                    stats=row,
                    bin_start=row["bin_start"],
                    bin_end=row["bin_end"],
                )


def write_artifacts(
    *,
    general_records: list[GeneralNLLRecord],
    position_records: list[PositionNLLRecord],
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    general_path = output_dir / "persistent_window_nll_general_records.jsonl"
    with general_path.open("w") as f:
        for record in general_records:
            json.dump(asdict(record), f, allow_nan=False)
            f.write("\n")

    position_path = output_dir / "persistent_window_nll_position_records.jsonl"
    with position_path.open("w") as f:
        for record in position_records:
            json.dump(asdict(record), f, allow_nan=False)
            f.write("\n")

    summary_path = output_dir / "persistent_window_nll_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)

    _write_summary_csv(summary, output_dir / "persistent_window_nll_summary.csv")


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    specs = [ga.parse_model_spec(spec_text) for spec_text in args.spec]
    device = torch.device(args.device)
    val_bin = Path(args.val_bin)
    output_dir = Path(args.output_dir)

    if args.persistent_window is None:
        raise ValueError("--persistent-window is required unless PERSISTENT_WINDOW is set")
    if args.persistent_window < 0:
        raise ValueError("--persistent-window must be >= 0")
    if args.seqlen < 2:
        raise ValueError("--seqlen must be >= 2")
    if args.num_batches < 1:
        raise ValueError("--num-batches must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.position_bin_size < 1:
        raise ValueError("--position-bin-size must be >= 1")

    all_general_records: list[GeneralNLLRecord] = []
    all_position_records: list[PositionNLLRecord] = []
    sampling_metadata_by_model: list[dict[str, Any]] = []
    for spec in specs:
        if spec.kind in TRITON_PERSISTENT_KEY_WINDOW_MODEL_KINDS and device.type != "cuda":
            raise RuntimeError(f"{spec.kind} measurement requires CUDA/Triton")

        print(f"\n=== {spec.label} ({spec.kind}) ===")
        print(f"Loading checkpoint: {spec.checkpoint_path}")
        loaded = ga.load_checkpoint_model(
            spec, device, warp_specialize=ga.warp_specialize_from_arg(args.warp_specialize)
        )
        if args.seqlen > int(loaded.config.block_size):
            raise ValueError(
                f"seqlen={args.seqlen} exceeds {spec.label} block_size={loaded.config.block_size}"
            )
        normal_window, persistent_key_window_original = resolve_attention_windows(loaded.model)
        print(
            "Config: "
            f"layers={loaded.config.n_layer}, heads={loaded.config.n_head}, "
            f"hidden={loaded.config.hidden_size}, block_size={loaded.config.block_size}"
        )
        print(
            "Window override: "
            f"normal={normal_window}, persistent_key={persistent_key_window_original} -> "
            f"{args.persistent_window if spec.kind in PERSISTENT_KEY_WINDOW_MODEL_KINDS else 'n/a'}"
        )

        adapter = ga.ModelAdapter(loaded)
        boundary_token_ids = (
            int(loaded.config.eos_token_id),
            int(loaded.config.pad_token_id),
        )
        batches = _make_document_start_batches(
            val_bin=val_bin,
            seqlen=args.seqlen,
            num_batches=args.num_batches,
            batch_size=args.batch_size,
            device=device,
            data_seed=args.data_seed,
            boundary_token_ids=boundary_token_ids,
        )
        sampling_metadata_by_model.append(
            {
                "model_id": spec.kind,
                "model_label": spec.label,
                "checkpoint_path": spec.checkpoint_path,
                "boundary_token_ids": list(boundary_token_ids),
                "sampled_batches": [asdict(batch.info) for batch in batches],
            }
        )

        model_general: list[GeneralNLLRecord] = []
        model_positions: list[PositionNLLRecord] = []
        for batch_index, batch in enumerate(batches):
            print(
                f"Batch {batch_index + 1}/{len(batches)}: "
                f"{batch.x_BxT.numel()} dense token positions"
            )
            general_records, position_records = measure_batch(
                adapter=adapter,
                spec=spec,
                x_BxT=batch.x_BxT,
                y_BxT=batch.y_BxT,
                batch_index=batch_index,
                batch_info=batch.info,
                persistent_key_window=args.persistent_window,
            )
            model_general.extend(general_records)
            model_positions.extend(position_records)
            if device.type == "cuda":
                torch.cuda.empty_cache()

        all_general_records.extend(model_general)
        all_position_records.extend(model_positions)
        print(
            f"Recorded {len(model_general)} general rows and "
            f"{len(model_positions)} position rows for {spec.label}"
        )
        del adapter
        del batches
        del loaded
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metadata = {
        "seqlen": args.seqlen,
        "persistent_key_window": args.persistent_window,
        "position_bin_size": args.position_bin_size,
        "num_batches": args.num_batches,
        "batch_size": args.batch_size,
        "device": args.device,
        "val_bin": str(val_bin),
        "data_seed": args.data_seed,
        "output_dir": str(output_dir),
        "model_specs": [asdict(spec) for spec in specs],
        "sequence_start_mode": "document_start",
        "document_sampling": "uniform_document_start_with_seqlen_plus_one_boundary_free_tokens",
        "source_position_sampling": "disabled",
        "position_axis": "document_relative_query_position",
        "delta_definition": "forced_nll_minus_baseline_nll",
        "forced_window_target": "persistent_token_keys",
        "confidence_interval": "normal_approx_95_percent",
        "sampling_by_model": sampling_metadata_by_model,
    }
    summary = aggregate_records(
        all_general_records,
        all_position_records,
        metadata=metadata,
    )
    write_artifacts(
        general_records=all_general_records,
        position_records=all_position_records,
        summary=summary,
        output_dir=output_dir,
    )
    print(f"\nSaved analysis artifacts to {output_dir}")
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    persistent_default = _persistent_window_default()
    parser = argparse.ArgumentParser(
        description=(
            "Measure NLL impact from forcing reduced persistent-token key visibility "
            "over dense document-relative positions."
        )
    )
    parser.add_argument(
        "--spec",
        action="append",
        required=False,
        help="Model spec as kind:label:/path/to/checkpoint.pt. Repeat for each model. "
        "Not required with --merge/--combine.",
    )
    parser.add_argument(
        "--persistent-window",
        type=int,
        default=persistent_default,
        required=persistent_default is None,
        help="Forced persistent-token key visibility window. Can also be set via PERSISTENT_WINDOW.",
    )
    parser.add_argument("--seqlen", type=int, default=DEFAULT_SEQLEN)
    parser.add_argument("--num-batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--position-bin-size", type=int, default=DEFAULT_POSITION_BIN_SIZE)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--val-bin", default=str(ga.default_val_bin()))
    parser.add_argument("--output-dir", default="outputs/persistent_window_nll_analysis/main")
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument(
        "--future-horizon",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--positions-per-batch",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--position-seed",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--skip-first",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    # --- Sharded / incremental binary path (mirrors gradient_analysis_params).
    parser.add_argument(
        "--output-format",
        choices=("jsonl", "binary"),
        default="jsonl",
        help="jsonl = legacy per-record artifacts (default); binary = sharded columnar "
        ".npy run dir for the parallel/incremental path.",
    )
    parser.add_argument(
        "--num-documents",
        type=int,
        default=None,
        help="Total documents [doc_start, num_documents) of the seeded permutation to "
        "measure (binary path). Without replacement; capped at the eligible count.",
    )
    parser.add_argument(
        "--doc-start",
        type=int,
        default=0,
        help="Start index into the seeded document permutation (binary path). To ADD "
        "samples later, re-run with --doc-start=<old num-documents>.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument(
        "--round-name",
        default=None,
        help="Round subdir name; defaults to r_<doc_start>_<num_documents>_s<num_shards>.",
    )
    parser.add_argument(
        "--merge",
        default=None,
        help="Merge one model dir's shards (all rounds) and write its single-model summary.",
    )
    parser.add_argument(
        "--combine",
        default=None,
        help="Scale dir whose model subdirs are merged + combined into one pw<window> summary.",
    )
    parser.add_argument(
        "--combine-models",
        default="sps,delayed_state",
        help="Comma-separated model subdir names for --combine (default sps,delayed_state).",
    )
    ga.add_warp_specialize_arg(parser)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.merge is not None:
        merge_run(Path(args.merge))
        return
    if args.combine is not None:
        if args.persistent_window is None:
            parser.error("--persistent-window is required with --combine")
        model_subdirs = [s for s in args.combine_models.split(",") if s]
        combine_scale(
            Path(args.combine),
            model_subdirs=model_subdirs,
            persistent_window=args.persistent_window,
            position_bin_size=args.position_bin_size,
        )
        return

    if not args.spec:
        parser.error("--spec is required unless --merge/--combine is given")
    if args.output_format == "binary":
        run_analysis_binary(args)
    else:
        run_analysis(args)


if __name__ == "__main__":
    main()
