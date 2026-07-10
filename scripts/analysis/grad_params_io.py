#!/usr/bin/env python3
"""Columnar binary IO for the parameter-gradient analysis.

The legacy artifact was one JSON line per ``(slot, layer, module, target)``
record, each carrying a 512-long ``offset_norms`` list plus repeated string
metadata. At the 16k-target scale that balloons to hundreds of GB and is slow to
load. This module stores the same per-record information as a directory of
``.npy`` columns (one array per field, memmap-friendly) plus a ``metadata.json``
sidecar that holds the run config and the slot/module code tables once.

Layout (``write_run`` / ``merge_shards`` produce the top level; each shard of a
sharded run is a ``shard_<idx>/`` subdir with the same per-record/per-target
arrays but no ``target_doc_index.npy``):

    <dir>/
      metadata.json
      offset_norms.npy          float32 (R, H)
      present_norm.npy          float32 (R,)
      future_mean_norm.npy      float32 (R,)
      future_mean_ratio.npy     float32 (R,)
      target_index.npy          int32   (R,)   -> row into the per-target arrays
      slot_code.npy             int8    (R,)
      module_code.npy           int8    (R,)
      layer.npy                 int16   (R,)
      target_start_offset.npy   int64   (T,)
      target_document_end.npy   int64   (T,)
      target_document_length.npy int32  (T,)
      target_source_position.npy int16  (T,)
      target_doc_index.npy      int32   (T,)   (top level only)

A "target" is one (document-window, source-position) pair; records sharing a
target pool together in the aggregation. ``target_index`` is derived from each
record's ``(batch_index, batch_item_index, source_position)`` so no extra fields
are needed on the measurement records.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
FORMAT_MARKER = "gradient_analysis_params_binary"

_PER_RECORD = {
    "offset_norms": "offset_norms.npy",
    "present_norm": "present_norm.npy",
    "future_mean_norm": "future_mean_norm.npy",
    "future_mean_ratio": "future_mean_ratio.npy",
    "target_index": "target_index.npy",
    "slot_code": "slot_code.npy",
    "module_code": "module_code.npy",
    "layer": "layer.npy",
}
_PER_TARGET = {
    "target_start_offset": "target_start_offset.npy",
    "target_document_end": "target_document_end.npy",
    "target_document_length": "target_document_length.npy",
    "target_source_position": "target_source_position.npy",
}
METADATA_FILE = "metadata.json"
DOC_INDEX_FILE = "target_doc_index.npy"


def is_binary_run(path: Path) -> bool:
    """True if ``path`` is a binary run dir (top level or a shard)."""
    return (path / METADATA_FILE).exists() and (path / "offset_norms.npy").exists()


def _offset_rows(records: list[Any], horizon: int) -> np.ndarray:
    """Stack each record's ``offset_norms`` into a float32 ``(R, horizon)`` array,
    mapping ``None``/non-finite to NaN. Fast path for uniform-length rows."""
    rows = [r.offset_norms for r in records]
    if rows and all(len(row) == horizon and None not in row for row in rows):
        mat = np.asarray(rows, dtype=np.float32)
    else:
        mat = np.full((len(rows), horizon), np.nan, dtype=np.float32)
        for i, row in enumerate(rows):
            k = min(len(row), horizon)
            for j in range(k):
                v = row[j]
                if v is not None:
                    mat[i, j] = v
    mat[~np.isfinite(mat)] = np.nan
    return mat


def records_to_columns(
    records: list[Any],
    *,
    horizon: int,
    module_types: tuple[str, ...],
) -> dict[str, Any]:
    """Convert measurement ``GradientRecord``s into the column dict.

    Returns per-record arrays, per-target arrays, the slot/module code tables,
    and the model identity (taken from the first record). Records must all belong
    to one model.
    """
    if not records:
        raise ValueError("records_to_columns: empty records")

    module_code_map = {m: i for i, m in enumerate(module_types)}

    # Slot table in first-appearance order.
    slot_order: list[str] = []
    slot_label: dict[str, str] = {}
    for r in records:
        if r.slot_id not in slot_label:
            slot_label[r.slot_id] = r.slot_label
            slot_order.append(r.slot_id)
    slot_code_map = {s: i for i, s in enumerate(slot_order)}

    # Target table from (batch_index, batch_item_index, source_position).
    tkey_to_idx: dict[tuple[int, int, int], int] = {}
    t_start: list[int] = []
    t_end: list[int] = []
    t_len: list[int] = []
    t_pos: list[int] = []
    target_index = np.empty(len(records), dtype=np.int32)
    for i, r in enumerate(records):
        key = (int(r.batch_index), int(r.batch_item_index), int(r.source_position))
        ti = tkey_to_idx.get(key)
        if ti is None:
            ti = len(tkey_to_idx)
            tkey_to_idx[key] = ti
            t_start.append(int(r.start_offset) if r.start_offset is not None else -1)
            t_end.append(
                int(r.document_end_offset) if r.document_end_offset is not None else -1
            )
            t_len.append(
                int(r.document_length) if r.document_length is not None else -1
            )
            t_pos.append(int(r.source_position))
        target_index[i] = ti

    present = np.array(
        [r.present_norm if r.present_norm is not None else np.nan for r in records],
        dtype=np.float32,
    )
    fmean = np.array(
        [
            r.future_mean_norm if r.future_mean_norm is not None else np.nan
            for r in records
        ],
        dtype=np.float32,
    )
    fratio = np.array(
        [
            r.future_mean_ratio if r.future_mean_ratio is not None else np.nan
            for r in records
        ],
        dtype=np.float32,
    )

    cols = {
        "offset_norms": _offset_rows(records, horizon),
        "present_norm": present,
        "future_mean_norm": fmean,
        "future_mean_ratio": fratio,
        "target_index": target_index,
        "slot_code": np.array(
            [slot_code_map[r.slot_id] for r in records], dtype=np.int8
        ),
        "module_code": np.array(
            [module_code_map[r.module_type] for r in records], dtype=np.int8
        ),
        "layer": np.array([int(r.layer) for r in records], dtype=np.int16),
        "target_start_offset": np.array(t_start, dtype=np.int64),
        "target_document_end": np.array(t_end, dtype=np.int64),
        "target_document_length": np.array(t_len, dtype=np.int32),
        "target_source_position": np.array(t_pos, dtype=np.int16),
    }
    tables = {
        "slot_table": [{"slot_id": s, "slot_label": slot_label[s]} for s in slot_order],
        "module_table": list(module_types),
        "model_id": records[0].model_id,
        "model_label": records[0].model_label,
        "checkpoint_path": records[0].checkpoint_path,
    }
    return {"cols": cols, "tables": tables}


def _compute_doc_index(start_offset: np.ndarray) -> np.ndarray:
    """Dense rank of ``start_offset`` -> per-target document id (global)."""
    _uniq, inv = np.unique(start_offset, return_inverse=True)
    return inv.astype(np.int32)


def _write_columns(
    out_dir: Path,
    cols: dict[str, np.ndarray],
    metadata: dict[str, Any],
    *,
    with_doc_index: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in {**_PER_RECORD, **_PER_TARGET}:
        np.save(out_dir / f"{name}.npy", cols[name])
    if with_doc_index:
        np.save(out_dir / DOC_INDEX_FILE, _compute_doc_index(cols["target_start_offset"]))
    (out_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2))


def write_run(
    out_dir: Path,
    records: list[Any],
    *,
    metadata: dict[str, Any],
    horizon: int,
    module_types: tuple[str, ...],
    shard_index: int | None = None,
    num_shards: int | None = None,
) -> dict[str, Any]:
    """Write one model's records as a binary run (or a shard if ``shard_index``).

    ``metadata`` is the run-level config; the slot/module tables and per-array
    counts are merged in. Shards omit ``target_doc_index`` (filled at merge).
    """
    packed = records_to_columns(records, horizon=horizon, module_types=module_types)
    cols, tables = packed["cols"], packed["tables"]
    meta = dict(metadata)
    meta.update(tables)
    meta.update(
        {
            "format": FORMAT_MARKER,
            "schema_version": SCHEMA_VERSION,
            "future_horizon": int(horizon),
            "record_count": int(cols["present_norm"].shape[0]),
            "target_count": int(cols["target_start_offset"].shape[0]),
            "dtype": "float32",
        }
    )
    is_shard = shard_index is not None
    if not is_shard:
        _write_columns(out_dir, cols, meta, with_doc_index=True)
        return meta

    # Shard write: publish atomically so concurrent workers (multiple partitions
    # scavenging the same task pool) and preempted/requeued tasks can never leave
    # a partial shard. Build in a private temp dir, then rename into place; the
    # final dir is only ever created by an atomic rename of a complete temp dir.
    meta["shard_index"] = int(shard_index)
    meta["num_shards"] = int(num_shards)
    final = Path(out_dir)
    if (final / METADATA_FILE).exists():
        return meta  # already published by another worker
    tmp = final.parent / f".{final.name}.tmp.{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    _write_columns(tmp, cols, meta, with_doc_index=False)
    try:
        os.rename(tmp, final)  # atomic when `final` does not yet exist
    except OSError:
        # Another worker published this shard first; discard our redundant copy.
        shutil.rmtree(tmp, ignore_errors=True)
    return meta


def _load_cols(run_dir: Path, *, mmap: bool) -> dict[str, np.ndarray]:
    mmap_mode = "r" if mmap else None
    out: dict[str, np.ndarray] = {}
    for name, fname in {**_PER_RECORD, **_PER_TARGET}.items():
        out[name] = np.load(run_dir / fname, mmap_mode=mmap_mode)
    doc_path = run_dir / DOC_INDEX_FILE
    if doc_path.exists():
        out["target_doc_index"] = np.load(doc_path, mmap_mode=mmap_mode)
    return out


def load_run(run_dir: Path, *, mmap: bool = True) -> dict[str, Any]:
    """Load a binary run into a flat dict of arrays + tables for ``aggregate_arrays``.

    Returns the per-record arrays (length R), the per-target arrays (length T,
    indexed by ``target_index``), the slot/module tables and the model identity.
    """
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / METADATA_FILE).read_text())
    cols = _load_cols(run_dir, mmap=mmap)
    data: dict[str, Any] = dict(cols)
    data["metadata"] = meta
    data["slot_table"] = meta["slot_table"]
    data["module_table"] = meta["module_table"]
    data["model_id"] = meta["model_id"]
    data["model_label"] = meta["model_label"]
    data["checkpoint_path"] = meta["checkpoint_path"]
    return data


def _shard_dirs(run_dir: Path) -> list[Path]:
    """All published shard dirs across rounds: ``<run>/r_*/shard_*`` (and legacy
    flat ``<run>/shard_*``). Only complete shards (with metadata.json) count."""
    out: list[Path] = []
    for pat in ("r_*/shard_*", "shard_*"):
        out.extend(
            p for p in run_dir.glob(pat)
            if p.is_dir() and (p / METADATA_FILE).exists()
        )
    return sorted(out)


def merge_shards(run_dir: Path, *, remove_shards: bool = False) -> dict[str, Any]:
    """Merge all rounds' shard dirs into top-level columnar arrays.

    Globs ``<run>/r_<doc_start>_<doc_end>/shard_*`` across every round, so a later
    incremental round (new document range) is folded in by simply re-running this
    -- the existing rounds are re-read, not recomputed. Per-record ``target_index``
    is rebased per shard; targets are **deduped by (start_offset, source_position)**
    so an accidentally re-run / overlapping round never double-counts. Shards are
    kept by default (``remove_shards=False``) so future rounds can be merged in.
    """
    run_dir = Path(run_dir)
    shard_dirs = _shard_dirs(run_dir)
    if not shard_dirs:
        raise FileNotFoundError(f"No shard dirs (r_*/shard_*) in {run_dir}")

    per_record: dict[str, list[np.ndarray]] = {k: [] for k in _PER_RECORD}
    per_target: dict[str, list[np.ndarray]] = {k: [] for k in _PER_TARGET}
    target_offset = 0
    base_meta: dict[str, Any] | None = None
    for sd in shard_dirs:
        meta = json.loads((sd / METADATA_FILE).read_text())
        cols = _load_cols(sd, mmap=True)
        for k in _PER_RECORD:
            arr = cols[k]
            if k == "target_index":
                arr = arr.astype(np.int64) + target_offset
            per_record[k].append(np.asarray(arr))
        for k in _PER_TARGET:
            per_target[k].append(np.asarray(cols[k]))
        target_offset += int(cols["target_start_offset"].shape[0])
        if base_meta is None:
            base_meta = meta

    merged: dict[str, np.ndarray] = {}
    for k in _PER_RECORD:
        merged[k] = np.concatenate(per_record[k])
    merged["target_index"] = merged["target_index"].astype(np.int64)
    for k in _PER_TARGET:
        merged[k] = np.concatenate(per_target[k])

    merged = _dedupe_targets(merged)
    merged["target_index"] = merged["target_index"].astype(np.int32)

    assert base_meta is not None
    out_meta = dict(base_meta)
    out_meta.pop("shard_index", None)
    out_meta.update(
        {
            "record_count": int(merged["present_norm"].shape[0]),
            "target_count": int(merged["target_start_offset"].shape[0]),
            "num_shards_merged": len(shard_dirs),
        }
    )
    _write_columns(run_dir, merged, out_meta, with_doc_index=True)
    if remove_shards:
        for sd in shard_dirs:
            shutil.rmtree(sd, ignore_errors=True)
        for tmp in run_dir.glob("**/.shard_*.tmp.*"):  # orphans from preempted workers
            shutil.rmtree(tmp, ignore_errors=True)
    return out_meta


def _dedupe_targets(merged: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Drop duplicate targets (same start_offset + source_position) and the records
    that point at the dropped copies, then compact ``target_index``.

    Disjoint incremental rounds never produce duplicates; this is a safety net for
    re-run / overlapping rounds so the aggregation can't double-count a target.
    """
    start = merged["target_start_offset"].astype(np.int64)
    pos = merged["target_source_position"].astype(np.int64)
    key = start * np.int64(1 << 20) + pos  # source_position < 2^20
    uniq_key, first_idx = np.unique(key, return_index=True)
    if uniq_key.size == merged["target_start_offset"].shape[0]:
        return merged  # no duplicates
    keeper_for_target = first_idx[np.searchsorted(uniq_key, key)]  # per old target
    rec_ti = merged["target_index"]
    keep_rec = keeper_for_target[rec_ti] == rec_ti
    kept_targets = np.sort(first_idx)
    new_index = np.full(keeper_for_target.shape[0], -1, dtype=np.int64)
    new_index[kept_targets] = np.arange(kept_targets.shape[0])
    out: dict[str, np.ndarray] = {}
    for k in _PER_RECORD:
        if k == "target_index":
            out[k] = new_index[rec_ti[keep_rec]].astype(np.int64)
        else:
            out[k] = merged[k][keep_rec]
    for k in _PER_TARGET:
        out[k] = merged[k][kept_targets]
    return out
