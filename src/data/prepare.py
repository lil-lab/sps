"""Tokenize FineWeb-Edu into train/val memmap files.

Step 1: Ensure dataset is cached at data_root (download if needed).
Step 2: Count val+train tokens in a single pass.
Step 3: Write val+train tokens in a single pass.

Uses content-hash-based splitting for consistent train/val assignment.
"""

import os
import sys
import time
import hashlib
import threading
from pathlib import Path
from queue import Empty
import numpy as np
import tiktoken
from tqdm import tqdm
import multiprocessing as mp
import hydra
from omegaconf import DictConfig, OmegaConf

import math
import random

DATASET = "HuggingFaceFW/fineweb-edu"
CONFIG = "sample-350BT"
BATCH_SIZE = 2000
NUM_PROC = max(1, math.floor(os.cpu_count() * 0.9))
TRAIN_SHUFFLE_BUFFER_SIZE = 200_000
SHUFFLE_SEED = 2357
VAL_FRACTION = 0.0005


def _env_flag(name: str) -> bool:
    return os.getenv(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def _hf_cache_dir(data_root: str) -> str:
    """Return HF cache base dir under data_root."""
    cache_base = Path(data_root).expanduser()
    if cache_base.name == "data":
        cache_base = cache_base.parent
    return str(cache_base / ".hf_cache")


def _set_hf_env(data_root: str, offline: bool = False) -> None:
    """Set HF env vars. Must be called BEFORE importing datasets/huggingface_hub."""
    hf_home = _hf_cache_dir(data_root)
    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_CACHE"] = os.path.join(hf_home, "hub")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_home, "datasets")
    if offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_DATASETS_OFFLINE", None)
        os.environ.pop("HF_HUB_OFFLINE", None)


def _is_val(doc_id: str, val_seed: int, val_fraction: float) -> bool:
    h = hashlib.md5(f"{val_seed}:{doc_id}".encode()).digest()
    return int.from_bytes(h[:8], "little") < int(val_fraction * (2 ** 64))


def _resolve_local_stream_files(data_root: str) -> list[str] | None:
    """Try to find cached parquet shards. Returns None if not cached."""
    hf_hub_cache = Path(_hf_cache_dir(data_root)) / "hub"
    repo_cache_dir = hf_hub_cache / "datasets--HuggingFaceFW--fineweb-edu"
    snapshots_dir = repo_cache_dir / "snapshots"
    refs_main = repo_cache_dir / "refs" / "main"

    if not snapshots_dir.exists():
        return None

    revision = None
    if refs_main.exists():
        revision = refs_main.read_text().strip()

    if revision:
        snapshot_dir = snapshots_dir / revision
    else:
        snapshot_candidates = [p for p in snapshots_dir.iterdir() if p.is_dir()]
        if not snapshot_candidates:
            return None
        snapshot_dir = max(snapshot_candidates, key=lambda p: p.stat().st_mtime)

    shard_dir = snapshot_dir / "sample" / "350BT"
    files = sorted(str(p) for p in shard_dir.glob("*.parquet"))
    if not files:
        return None
    return files


def _download_dataset(data_root: str) -> list[str]:
    """Download FineWeb-Edu parquet shards via huggingface_hub."""
    from huggingface_hub import snapshot_download
    print(f"      Downloading {DATASET} ({CONFIG})...", flush=True)
    snapshot_download(
        repo_id=DATASET,
        repo_type="dataset",
        allow_patterns="sample/350BT/*.parquet",
        cache_dir=os.path.join(_hf_cache_dir(data_root), "hub"),
    )
    files = _resolve_local_stream_files(data_root)
    if not files:
        raise RuntimeError("Download completed but no parquet shards found in cache.")
    return files


def _rows_per_file(files: list[str]) -> list[int]:
    import pyarrow.parquet as pq
    return [int(pq.ParquetFile(path).metadata.num_rows) for path in files]


def _apply_debug_file_limit(files: list[str], file_rows: list[int]) -> tuple[list[str], list[int]]:
    max_files = int(os.getenv("PREPARE_MAX_FILES", "0") or 0)
    if max_files <= 0 or max_files >= len(files):
        return files, file_rows

    mode = os.getenv("PREPARE_FILE_SELECT", "spread").strip().lower()
    offset = int(os.getenv("PREPARE_FILE_OFFSET", "0") or 0)
    n = len(files)

    if mode == "head":
        idxs = [(offset + i) % n for i in range(max_files)]
    else:
        step = max(1, n // max_files)
        idxs = [((offset + i * step) % n) for i in range(max_files)]

    sel_files = [files[i] for i in idxs]
    sel_rows = [file_rows[i] for i in idxs]
    return sel_files, sel_rows


def _progress_monitor(shared_counter, total, desc, unit, done_event):
    """Display a single tqdm bar in the main process, polling a shared counter."""
    bar = tqdm(total=total, desc=desc, unit=unit, unit_scale=True, dynamic_ncols=True)
    while not done_event.is_set():
        bar.n = shared_counter.value
        bar.refresh()
        time.sleep(0.25)
    bar.n = shared_counter.value
    bar.refresh()
    bar.close()


def _load_dataset_from_files(files, shuffle, shuffle_buffer_size, shuffle_seed):
    import datasets
    from datasets import load_dataset
    datasets.disable_progress_bars()
    ds = load_dataset("parquet", data_files=files, split="train", streaming=True)
    if shuffle:
        ds = ds.shuffle(buffer_size=shuffle_buffer_size, seed=shuffle_seed)
    return ds


# ---------------------------------------------------------------------------
# Pass 1: Count tokens (val + train jointly)
# ---------------------------------------------------------------------------

def _worker_count_both(
    rank, world_size, val_seed, val_fraction, source_files, source_file_rows,
    result_queue, shared_progress,
):
    enc = tiktoken.get_encoding("gpt2")
    rank_files = source_files[rank::world_size]
    if not rank_files:
        result_queue.put((rank, 0, 0))
        return

    ds = _load_dataset_from_files(rank_files, shuffle=False, shuffle_buffer_size=0, shuffle_seed=0)

    val_tokens = 0
    train_tokens = 0
    val_batch = []
    train_batch = []
    shared_pending = 0

    def _flush(batch, is_val_batch):
        nonlocal val_tokens, train_tokens
        if not batch:
            return
        batch_tokens = 0
        for tokens in enc.encode_ordinary_batch(batch):
            batch_tokens += len(tokens) + 1
        if is_val_batch:
            val_tokens += batch_tokens
        else:
            train_tokens += batch_tokens
        batch.clear()

    for ex in ds:
        shared_pending += 1
        if _is_val(ex["id"], val_seed, val_fraction):
            val_batch.append(ex["text"])
            if len(val_batch) >= BATCH_SIZE:
                _flush(val_batch, is_val_batch=True)
        else:
            train_batch.append(ex["text"])
            if len(train_batch) >= BATCH_SIZE:
                _flush(train_batch, is_val_batch=False)
        if shared_pending >= BATCH_SIZE:
            with shared_progress.get_lock():
                shared_progress.value += shared_pending
            shared_pending = 0

    _flush(val_batch, is_val_batch=True)
    _flush(train_batch, is_val_batch=False)
    if shared_pending > 0:
        with shared_progress.get_lock():
            shared_progress.value += shared_pending
    result_queue.put((rank, val_tokens, train_tokens))


def _count_tokens(val_seed, val_fraction, source_files, source_file_rows):
    print(f"\n[2/3] COUNT TOKENS (VAL+TRAIN)")
    print(f"      Workers: {NUM_PROC}")
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()

    shared_progress = ctx.Value("l", 0)
    done_event = threading.Event()
    total_docs = int(sum(source_file_rows))
    monitor_thread = threading.Thread(
        target=_progress_monitor,
        args=(shared_progress, total_docs, "count", "docs", done_event),
        daemon=True,
    )
    monitor_thread.start()

    procs = []
    for r in range(NUM_PROC):
        p = ctx.Process(
            target=_worker_count_both,
            args=(r, NUM_PROC, val_seed, val_fraction, source_files, source_file_rows, result_queue),
            kwargs=dict(shared_progress=shared_progress),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    done_event.set()
    monitor_thread.join()

    failed = [p for p in procs if p.exitcode not in (0, None)]
    if failed:
        failed_str = ", ".join(f"pid={p.pid} exit={p.exitcode}" for p in failed[:8])
        if len(failed) > 8:
            failed_str += f", ... (+{len(failed) - 8} more)"
        raise RuntimeError(f"{len(failed)}/{len(procs)} worker(s) failed during counting: {failed_str}")

    results = {}
    while len(results) < NUM_PROC:
        try:
            rank, val_tok, train_tok = result_queue.get(timeout=5.0)
        except Empty as e:
            raise RuntimeError("Timed out waiting for count results.") from e
        results[rank] = (val_tok, train_tok)

    val_worker_counts = [results[r][0] for r in range(NUM_PROC)]
    train_worker_counts = [results[r][1] for r in range(NUM_PROC)]
    val_total = sum(val_worker_counts)
    train_total = sum(train_worker_counts)
    print(f"      Val total:   {val_total:,} tokens ({val_total * 2 / 1e9:.2f} GB)")
    print(f"      Train total: {train_total:,} tokens ({train_total * 2 / 1e9:.2f} GB)")
    return val_worker_counts, train_worker_counts


# ---------------------------------------------------------------------------
# Pass 2: Write tokens (val + train jointly)
# ---------------------------------------------------------------------------

def _worker_write_both(
    rank, world_size, val_seed, val_fraction, source_files, source_file_rows,
    shared_progress, val_memmap_path, val_offset, val_expected,
    train_memmap_path, train_offset, train_expected,
):
    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    rank_files = source_files[rank::world_size]
    if not rank_files:
        return

    ds = _load_dataset_from_files(rank_files, shuffle=True,
                                   shuffle_buffer_size=TRAIN_SHUFFLE_BUFFER_SIZE,
                                   shuffle_seed=SHUFFLE_SEED)

    val_mmap = np.memmap(val_memmap_path, dtype=np.uint16, mode="r+")
    train_mmap = np.memmap(train_memmap_path, dtype=np.uint16, mode="r+")

    val_pos = val_offset
    train_pos = train_offset
    val_tokens = 0
    train_tokens = 0
    val_batch = []
    train_batch = []
    shared_pending = 0

    def _flush(batch, mmap, pos):
        written = 0
        for tokens in enc.encode_ordinary_batch(batch):
            n = len(tokens) + 1
            mmap[pos:pos + n - 1] = tokens
            mmap[pos + n - 1] = eot
            pos += n
            written += n
        batch.clear()
        return pos, written

    for ex in ds:
        shared_pending += 1
        if _is_val(ex["id"], val_seed, val_fraction):
            val_batch.append(ex["text"])
            if len(val_batch) >= BATCH_SIZE:
                val_pos, n = _flush(val_batch, val_mmap, val_pos)
                val_tokens += n
        else:
            train_batch.append(ex["text"])
            if len(train_batch) >= BATCH_SIZE:
                train_pos, n = _flush(train_batch, train_mmap, train_pos)
                train_tokens += n
        if shared_pending >= BATCH_SIZE:
            with shared_progress.get_lock():
                shared_progress.value += shared_pending
            shared_pending = 0

    if val_batch:
        val_pos, n = _flush(val_batch, val_mmap, val_pos)
        val_tokens += n
    if train_batch:
        train_pos, n = _flush(train_batch, train_mmap, train_pos)
        train_tokens += n
    if shared_pending > 0:
        with shared_progress.get_lock():
            shared_progress.value += shared_pending

    if val_tokens != val_expected:
        print(f"WARNING: rank {rank} val expected {val_expected} tokens but wrote {val_tokens}")
    if train_tokens != train_expected:
        print(f"WARNING: rank {rank} train expected {train_expected} tokens but wrote {train_tokens}")

    val_mmap.flush()
    train_mmap.flush()
    del val_mmap, train_mmap


def _write_tokens(val_seed, val_fraction, source_files, source_file_rows,
                  val_worker_counts, train_worker_counts, val_path, train_path):
    val_total = sum(val_worker_counts)
    train_total = sum(train_worker_counts)

    print(f"\n[3/3] WRITE TOKENS (VAL+TRAIN)")
    print(f"      Workers: {NUM_PROC}")
    print(f"      Val:   {val_path} ({val_total:,} tokens, {val_total * 2 / 1e9:.2f} GB)")
    print(f"      Train: {train_path} ({train_total:,} tokens, {train_total * 2 / 1e9:.2f} GB)")

    # Allocate memmaps
    for path, total in [(val_path, val_total), (train_path, train_total)]:
        mmap = np.memmap(path, dtype=np.uint16, mode="w+", shape=(total,))
        mmap.flush()
        del mmap

    # Compute per-worker offsets
    val_offsets = []
    s = 0
    for c in val_worker_counts:
        val_offsets.append(s)
        s += c

    train_offsets = []
    s = 0
    for c in train_worker_counts:
        train_offsets.append(s)
        s += c

    ctx = mp.get_context("spawn")
    shared_progress = ctx.Value("l", 0)
    done_event = threading.Event()
    total_docs = int(sum(source_file_rows))
    monitor_thread = threading.Thread(
        target=_progress_monitor,
        args=(shared_progress, total_docs, "write", "docs", done_event),
        daemon=True,
    )
    monitor_thread.start()

    procs = []
    for r in range(NUM_PROC):
        p = ctx.Process(
            target=_worker_write_both,
            args=(r, NUM_PROC, val_seed, val_fraction, source_files, source_file_rows,
                  shared_progress, str(val_path), val_offsets[r], val_worker_counts[r],
                  str(train_path), train_offsets[r], train_worker_counts[r]),
        )
        p.start()
        procs.append(p)
    for p in procs:
        p.join()

    done_event.set()
    monitor_thread.join()

    failed = [p for p in procs if p.exitcode not in (0, None)]
    if failed:
        failed_str = ", ".join(f"pid={p.pid} exit={p.exitcode}" for p in failed[:8])
        if len(failed) > 8:
            failed_str += f", ... (+{len(failed) - 8} more)"
        raise RuntimeError(f"{len(failed)}/{len(procs)} worker(s) failed during writing: {failed_str}")

    return val_total, train_total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="../../conf", config_name="config")
def main(cfg: DictConfig):
    mp.set_start_method("spawn", force=True)
    global NUM_PROC

    if not cfg.system.data_root:
        raise ValueError("cfg.system.data_root is required")

    data_root = cfg.system.data_root
    hf_cache = _hf_cache_dir(data_root)

    print(f"\n{'='*60}", flush=True)
    print(f"  FineWeb-Edu preparation ({DATASET}/{CONFIG})", flush=True)
    print(f"  data_root: {data_root}", flush=True)
    print(f"  hf_cache:  {hf_cache}", flush=True)
    print(f"{'='*60}", flush=True)

    # Step 1: Resolve dataset files, downloading if not cached.
    print(f"\n[1/3] DATASET", flush=True)
    _set_hf_env(data_root, offline=False)
    print(f"      Resolving local cache for {DATASET} ({CONFIG})...", flush=True)
    source_files = _resolve_local_stream_files(data_root)
    if source_files is None:
        print(f"      Cache not found, downloading...", flush=True)
        source_files = _download_dataset(data_root)
    source_file_rows = _rows_per_file(source_files)
    source_files, source_file_rows = _apply_debug_file_limit(source_files, source_file_rows)
    max_files = int(os.getenv("PREPARE_MAX_FILES", "0") or 0)
    if max_files > 0:
        mode = os.getenv("PREPARE_FILE_SELECT", "spread").strip().lower()
        offset = int(os.getenv("PREPARE_FILE_OFFSET", "0") or 0)
        print(
            f"      DEBUG: limiting to {len(source_files):,} shard files "
            f"(PREPARE_MAX_FILES={max_files}, PREPARE_FILE_SELECT={mode}, PREPARE_FILE_OFFSET={offset})."
        , flush=True)
    # Shuffle shard order so adjacent workers get unrelated shards in the output file
    rng = random.Random(SHUFFLE_SEED)
    shard_indices = list(range(len(source_files)))
    rng.shuffle(shard_indices)
    source_files = [source_files[i] for i in shard_indices]
    source_file_rows = [source_file_rows[i] for i in shard_indices]

    print(f"      Resolved {len(source_files):,} local shards.", flush=True)
    print(f"      Done.", flush=True)

    # Keep workers offline. They read directly from local parquet shards.
    _set_hf_env(data_root, offline=True)

    # Config
    dataset_dir = os.path.join(data_root, "data", cfg.data.dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    val_seed = int(getattr(cfg.data, "val_split_seed", 42))
    NUM_PROC = int(getattr(cfg.data, "num_proc", NUM_PROC))
    if NUM_PROC < 1:
        raise ValueError("cfg.data.num_proc must be >= 1")

    val_path = os.path.join(dataset_dir, "val.bin")
    train_path = os.path.join(dataset_dir, "train.bin")

    # Step 2: Count
    val_worker_counts, train_worker_counts = _count_tokens(
        val_seed=val_seed,
        val_fraction=VAL_FRACTION,
        source_files=source_files,
        source_file_rows=source_file_rows,
    )

    if _env_flag("PREPARE_COUNT_ONLY"):
        print("\nDEBUG: PREPARE_COUNT_ONLY=1, skipping write phase.", flush=True)
        val_tokens = sum(val_worker_counts)
        train_tokens = sum(train_worker_counts)
    else:
        # Step 3: Write both splits in a single pass
        val_tokens, train_tokens = _write_tokens(
            val_seed=val_seed,
            val_fraction=VAL_FRACTION,
            source_files=source_files,
            source_file_rows=source_file_rows,
            val_worker_counts=val_worker_counts,
            train_worker_counts=train_worker_counts,
            val_path=val_path,
            train_path=train_path,
        )

    print(f"\n{'='*60}", flush=True)
    print(f"  COMPLETE", flush=True)
    print(f"  Validation: {val_path} ({val_tokens:,} tokens)", flush=True)
    print(f"  Training:   {train_path} ({train_tokens:,} tokens)", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
