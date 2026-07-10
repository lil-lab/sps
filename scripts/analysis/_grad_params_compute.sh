#!/bin/bash
#SBATCH --job-name=grad_params_compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
# A tight walltime lets tasks backfill into free GPU slots instead of waiting for
# a full-length gap -- the shorter, the more idle windows it fits (key on the
# contended/preemptible requeue partitions). Sized for the slowest task: XL at
# 128 shards (chunk-2) measured ~23 min/shard, so 45 min carries margin. If you
# go back to 64 shards (XL ~46 min/shard), raise this to ~01:30:00.
#SBATCH --time=00:45:00
#SBATCH --gpu-bind=closest
#SBATCH --output=logs/slurm/grad_params_compute_%A_%a.out
#SBATCH --error=logs/slurm/grad_params_compute_%A_%a.err
#
# One array task = one (model, shard). Effective index = model_ordinal*NUM_SHARDS
# + shard. This script is a WORK-POOL worker: it skips shards already published
# by any worker, so the same array can be launched on several partitions at once
# (e.g. a reliable dedicated partition + a preemptible/requeue one) to stack GPUs
# past a per-user concurrency cap. ORDER=desc sweeps the pool from the high end so
# two partitions meet in the middle with minimal overlap.
# --partition / --account / --gres are supplied by the submitter (PARTITION /
# ACCOUNT / GRES); the --gres above is a neutral fallback for a bare sbatch.
# Submitted by run_gradient_analysis_params.sh (reads the exported MANIFEST).
set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT_DIR"

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NMODELS="$(grep -c . "$MANIFEST")"
NTASKS=$(( NMODELS * NUM_SHARDS ))
task="${SLURM_ARRAY_TASK_ID:?must run as an array task}"
# Reverse the sweep direction for scavenger partitions (ORDER=desc).
eff="${task}"
if [[ "${ORDER:-asc}" == "desc" ]]; then
  eff=$(( NTASKS - 1 - task ))
fi
model_ord=$(( eff / NUM_SHARDS ))
shard=$(( eff % NUM_SHARDS ))

# Read model_ord-th manifest line (1-based for sed).
line=$(sed -n "$(( model_ord + 1 ))p" "$MANIFEST")
if [[ -z "${line}" ]]; then
  echo "No manifest line for model_ord=${model_ord}; exiting."
  exit 0
fi
IFS=$'\t' read -r out_dir kind label ckpt warp <<<"${line}"

DOC_START="${DOC_START:-0}"
# Shard count is part of the round name so runs with different --num-shards live
# in separate dirs (shard index is relative to num_shards, so 64- and 128-shard
# shard_5 cover different targets). The merge globs r_*/shard_* and dedups by
# target, so they still combine into one summary.
round="r_${DOC_START}_${NUM_DOCUMENTS}_s${NUM_SHARDS}"

# Work-pool skip: another worker (or a prior run) already finished this shard.
if [[ -f "${out_dir}/${round}/shard_${shard}/metadata.json" ]]; then
  echo "shard ${shard} of ${out_dir}/${round} already done; skipping."
  exit 0
fi

echo "=== task ${task} (ORDER=${ORDER:-asc} eff=${eff}): model_ord=${model_ord} (${kind} ${label}) ${round} shard=${shard}/${NUM_SHARDS} -> ${out_dir} ==="
uv run python -m scripts.analysis.gradient_analysis_params \
  --spec "${kind}:${label}:${ckpt}" \
  --warp-specialize "${warp}" \
  --batched-gradient-mode per_row_position \
  --output-format binary \
  --seqlen "${SEQLEN}" \
  --future-horizon "${FUTURE_HORIZON}" \
  --num-documents "${NUM_DOCUMENTS}" \
  --positions-per-document "${POSITIONS_PER_DOCUMENT}" \
  --doc-start "${DOC_START}" \
  --num-shards "${NUM_SHARDS}" \
  --shard-index "${shard}" \
  --round-name "${round}" \
  --batch-size "${BATCH_SIZE}" \
  --forward-chunk-size "${FORWARD_CHUNK_SIZE}" \
  --skip-first "${SKIP_FIRST}" \
  --data-seed "${DATA_SEED}" \
  --position-seed "${POSITION_SEED}" \
  --output-dir "${out_dir}"
