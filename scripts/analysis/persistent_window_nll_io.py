#!/usr/bin/env python3
"""Columnar binary IO for the persistent-window NLL analysis.

Mirrors ``grad_params_io.py`` but for the forced-persistent-window NLL measurement.
Each shard stores per-(document, position) baseline/forced NLL as memmap-friendly
``.npy`` columns plus a ``metadata.json`` sidecar. This lets the analysis run as
many independent ``(model, shard)`` SLURM tasks that each publish one shard
atomically, and lets later document-range "rounds" be folded in at merge time
without recomputing earlier rounds (the noise of a per-position mean is just
``~1/sqrt(num_documents)``, so adding documents straightforwardly de-noises it).

Layout (one model per dir; each shard of a sharded run is a ``shard_<idx>/``
subdir under a round dir ``r_<doc_start>_<num_documents>_s<num_shards>/``; the
merged top level lives directly in ``<dir>/``):

    <dir>/
      metadata.json
      baseline_nll.npy      float32 (T, S)   per (document, position); NaN = invalid
      forced_nll.npy        float32 (T, S)   per (document, position); NaN = invalid
      doc_start_offset.npy  int64   (T,)      document identity (dedupe key)
      doc_end_offset.npy    int64   (T,)
      doc_length.npy        int32   (T,)

A "document" is one ``seqlen``-length window anchored at a boundary-clean document
start; the position axis is the document-relative query position. ``delta_nll`` is
derived at aggregation as ``forced_nll - baseline_nll`` (both share the same
validity mask, so per-position ``delta`` means equal ``mean(forced) - mean(baseline)``).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
FORMAT_MARKER = "persistent_window_nll_binary"
METADATA_FILE = "metadata.json"

# 2D per-(document, position) columns and 1D per-document columns, with dtypes.
_COLS_2D = ("baseline_nll", "forced_nll")
_COLS_DOC = ("doc_start_offset", "doc_end_offset", "doc_length")
_DTYPES = {
    "baseline_nll": np.float32,
    "forced_nll": np.float32,
    "doc_start_offset": np.int64,
    "doc_end_offset": np.int64,
    "doc_length": np.int32,
}
_ALL_COLS = (*_COLS_2D, *_COLS_DOC)


def is_binary_run(path: Path) -> bool:
    """True if ``path`` holds binary columns (a merged top level or a shard)."""
    path = Path(path)
    return (path / METADATA_FILE).exists() and (path / "baseline_nll.npy").exists()


def _write_columns(out_dir: Path, cols: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in _ALL_COLS:
        np.save(out_dir / f"{name}.npy", np.asarray(cols[name], dtype=_DTYPES[name]))
    (out_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2))


def write_shard(
    out_dir: Path,
    cols: dict[str, np.ndarray],
    *,
    metadata: dict[str, Any],
    shard_index: int,
    num_shards: int,
) -> dict[str, Any]:
    """Publish one ``(model, shard)`` slice atomically.

    Build the columns in a private temp dir, then ``rename`` into place; the final
    dir is only ever created by an atomic rename of a complete temp dir, so
    concurrent workers (multiple partitions scavenging the same task pool) and
    preempted/requeued tasks can never leave a partial shard.
    """
    meta = dict(metadata)
    meta.update(
        {
            "format": FORMAT_MARKER,
            "schema_version": SCHEMA_VERSION,
            "seqlen": int(cols["baseline_nll"].shape[1]),
            "document_count": int(cols["baseline_nll"].shape[0]),
            "shard_index": int(shard_index),
            "num_shards": int(num_shards),
        }
    )
    final = Path(out_dir)
    if (final / METADATA_FILE).exists():
        return meta  # already published by another worker
    tmp = final.parent / f".{final.name}.tmp.{os.getpid()}"
    shutil.rmtree(tmp, ignore_errors=True)
    _write_columns(tmp, cols, meta)
    try:
        os.rename(tmp, final)  # atomic when `final` does not yet exist
    except OSError:
        # Another worker published this shard first; discard our redundant copy.
        shutil.rmtree(tmp, ignore_errors=True)
    return meta


def _load_cols(run_dir: Path, *, mmap: bool) -> dict[str, np.ndarray]:
    mmap_mode = "r" if mmap else None
    return {
        name: np.load(run_dir / f"{name}.npy", mmap_mode=mmap_mode) for name in _ALL_COLS
    }


def load_run(run_dir: Path, *, mmap: bool = True) -> dict[str, Any]:
    """Load a binary run (merged top level or a single shard) into arrays + metadata."""
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / METADATA_FILE).read_text())
    data: dict[str, Any] = dict(_load_cols(run_dir, mmap=mmap))
    data["metadata"] = meta
    return data


def _shard_dirs(run_dir: Path) -> list[Path]:
    """All published shard dirs across rounds: ``<run>/r_*/shard_*`` (and legacy
    flat ``<run>/shard_*``). Only complete shards (with metadata.json) count."""
    out: list[Path] = []
    for pat in ("r_*/shard_*", "shard_*"):
        out.extend(
            p for p in run_dir.glob(pat) if p.is_dir() and (p / METADATA_FILE).exists()
        )
    return sorted(out)


def _dedupe_docs(merged: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Drop duplicate documents (same ``doc_start_offset``), keeping the first.

    Disjoint incremental rounds (distinct ``[doc_start, num_documents)`` slices of
    one fixed permutation) never produce duplicates; this is a safety net for an
    accidentally re-run / overlapping round so the aggregation can't double-count.
    """
    key = np.asarray(merged["doc_start_offset"], dtype=np.int64)
    uniq, first_idx = np.unique(key, return_index=True)
    if uniq.size == key.shape[0]:
        return merged
    keep = np.sort(first_idx)
    return {k: np.asarray(merged[k])[keep] for k in _ALL_COLS}


def merge_shards(run_dir: Path, *, remove_shards: bool = False) -> dict[str, Any]:
    """Merge all rounds' shard dirs into top-level columnar arrays.

    Globs ``<run>/r_<...>/shard_*`` across every round, concatenates along the
    document axis and dedupes by ``doc_start_offset``, so a later incremental round
    (new document range) is folded in by simply re-running this -- the existing
    rounds are re-read, not recomputed. Shards are kept by default so future rounds
    can be merged in.
    """
    run_dir = Path(run_dir)
    shard_dirs = _shard_dirs(run_dir)
    if not shard_dirs:
        raise FileNotFoundError(f"No shard dirs (r_*/shard_*) in {run_dir}")

    parts: dict[str, list[np.ndarray]] = {k: [] for k in _ALL_COLS}
    base_meta: dict[str, Any] | None = None
    seqlen: int | None = None
    for sd in shard_dirs:
        meta = json.loads((sd / METADATA_FILE).read_text())
        cols = _load_cols(sd, mmap=True)
        s = int(cols["baseline_nll"].shape[1])
        if seqlen is None:
            seqlen = s
        elif s != seqlen:
            raise ValueError(f"shard {sd} seqlen {s} != {seqlen}")
        for k in _ALL_COLS:
            parts[k].append(np.asarray(cols[k]))
        if base_meta is None:
            base_meta = meta

    merged = {k: np.concatenate(parts[k], axis=0) for k in _ALL_COLS}
    merged = _dedupe_docs(merged)

    assert base_meta is not None
    out_meta = dict(base_meta)
    out_meta.pop("shard_index", None)
    out_meta.pop("num_shards", None)
    out_meta.update(
        {
            "document_count": int(merged["doc_start_offset"].shape[0]),
            "seqlen": int(seqlen),
            "num_shards_merged": len(shard_dirs),
        }
    )
    _write_columns(run_dir, merged, out_meta)
    if remove_shards:
        for sd in shard_dirs:
            shutil.rmtree(sd, ignore_errors=True)
        for tmp in run_dir.glob("**/.shard_*.tmp.*"):  # orphans from preempted workers
            shutil.rmtree(tmp, ignore_errors=True)
    return out_meta
