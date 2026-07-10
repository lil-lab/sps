# Reproducing the generation-speed results (main table)

Both speed columns of the main results table — **Throughput (×)** and **Peak Memory (×)** —
come from a **single** benchmark run: `scripts/benchmark/benchmark_generation_speed.py`,
driven by one cluster-independent config, `conf/benchmark/speed.yaml`. Hyperparameters (the
*what*) live in version-controlled YAML; checkpoint paths, `val.bin`, and the SLURM partition
(the *where*) are passed as environment variables — so you can reproduce the numbers on
another cluster with the same architectures but different checkpoint weights.

## The run

`seed=1337`, all four methods (Standard / 2× Memory / Delayed State / SPS), all five scales
(xs–xl), prompt/batch/new-tokens **1024 / 16 / 3072**, warmup/timed **1 / 2**, `val.bin` from
fineweb-edu. It emits one `results.json`, and the table script derives **both** the throughput
and peak-memory vs-Standard ratios from that single file — there is no separate prefill or
memory bench.

- **Batch 16 across all scales** — a larger batch OOMs XL "2× Memory" (w4096) on a single
  80 GB H100, so the whole table uses batch 16 for an apples-to-apples set.
- **XL needs `WARP_SPECIALIZE=off`** — its SPS kernels can't compile warp-specialized on
  current Triton (numerically identical, perf-only); xs–l already bake warp-off into their
  checkpoints, so one `WARP_SPECIALIZE=off` job reproduces all five.

## Run it

Dry-run first (`DRY_RUN=1`): it checks every checkpoint exists and prints the exact driver
args without burning an allocation.

```bash
WARP_SPECIALIZE=off \                     # required once XL is included (perf-only)
OUT_ROOT=/path/to/checkpoints \           # holds <run_name>/<ckpt>, e.g. xs_full_attention_20b/ckpt_...pt
VAL_BIN=/path/to/fineweb-edu/val.bin \
PARTITION=<gpu-partition> \
ACCOUNT=<slurm-account> GRES=<gres-type> \  # omit ACCOUNT if unused; GRES e.g. nvidia_h100_80gb_hbm3
bash scripts/benchmark/run.sh
```

Output lands in `outputs/generation_timing_correctness/<benchmark>_<scales>_<gres>/`
(`results.json`, `results_summary.tsv`, `driver_args.txt`).

| Var | | Meaning |
|---|---|---|
| `OUT_ROOT` | ✅ | Checkpoint root; path = `$OUT_ROOT/<run_name>/$CKPT_NAME` |
| `PARTITION` | ✅ | SLURM GPU partition |
| `WARP_SPECIALIZE` | | `keep`\|`on`\|`off`; use `off` for any run including XL |
| `VAL_BIN` · `CKPT_NAME` · `SCALES` | | val shard · ckpt filename · scale subset (default: all 5) |
| `DRY_RUN` | | `1` to validate + print args without submitting |

Plus the usual SLURM knobs (`ACCOUNT`, `GRES`, `CONSTRAINT`, `NODELIST`, `TIME_LIMIT`, `MEM`,
`CPUS`, `DEVICE`), documented in the `run.sh` header. For a different checkpoint-naming
scheme, edit the `run_name` templates in `conf/benchmark/speed.yaml`.

## Regenerate the table

```bash
uv run python scripts/tables/export_main_results_table.py \
  --results-json outputs/generation_timing_correctness/<your run dir>/results.json
```

Omit `--results-json` to use the committed default.

## Notes

- **Wall-clock on H100 80 GB.** Different silicon reproduces the *methodology*, not identical
  tok/s — compare the vs-Standard **ratios**, which port far better.
- Windowed models update their KV caches with a fused Triton kernel (numerically identical to
  the eager path); it is always on, with nothing to configure.
- Every `results.json` carries a `provenance` block (git commit, torch/CUDA versions) plus
  `settings.config_name`; `provenance.git_commit` should match the commit you ran.
