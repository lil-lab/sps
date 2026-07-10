#!/bin/bash
#SBATCH --job-name=pwnll_compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
# Forward-only (2 passes/doc); a tight walltime backfills into free GPU slots.
# Sized for the slowest task (XL, 19GB ckpt load + ~1k docs/shard). Raise if you
# push NUM_DOCUMENTS very high with few shards.
#SBATCH --time=00:45:00
#SBATCH --gpu-bind=closest
#SBATCH --output=logs/slurm/pwnll_compute_%A_%a.out
#SBATCH --error=logs/slurm/pwnll_compute_%A_%a.err
#
# One array task = one (model, shard). Effective index = model_ord*NUM_SHARDS +
# shard. WORK-POOL worker: it skips shards already published by any worker, so the
# same array can run on several partitions at once (a reliable dedicated partition +
# preemptible scavengers) to stack GPUs past a per-user cap. ORDER=desc sweeps
# from the high end so two partitions meet in the middle with minimal overlap.
# --partition / --account / --gres are supplied by the submitter (PARTITION /
# ACCOUNT / GRES); the --gres above is a neutral fallback for a bare sbatch.
# Submitted by run_persistent_window_nll.sh (reads the exported MANIFEST).
set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT_DIR"

export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NMODELS="$(grep -c . "$MANIFEST")"
NTASKS=$(( NMODELS * NUM_SHARDS ))
task="${SLURM_ARRAY_TASK_ID:?must run as an array task}"
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
IFS=$'\t' read -r out_dir scale kind label ckpt warp batch_size <<<"${line}"

# Shard count is part of the round name so runs with different --num-shards live in
# separate dirs (shard index is relative to num_shards). The merge globs r_*/shard_*
# and dedups by document, so they still combine into one summary.
round="r_${DOC_START}_${NUM_DOCUMENTS}_s${NUM_SHARDS}"

# Work-pool skip: another worker (or a prior run) already finished this shard.
if [[ -f "${out_dir}/${round}/shard_${shard}/metadata.json" ]]; then
  echo "shard ${shard} of ${out_dir}/${round} already done; skipping."
  exit 0
fi

echo "=== task ${task} (ORDER=${ORDER:-asc} eff=${eff}): model_ord=${model_ord} (${scale} ${kind}) ${round} shard=${shard}/${NUM_SHARDS} bs=${batch_size} -> ${out_dir} ==="
uv run python -m scripts.analysis.persistent_window_nll_analysis \
  --spec "${kind}:${label}:${ckpt}" \
  --warp-specialize "${warp}" \
  --output-format binary \
  --persistent-window "${PERSISTENT_WINDOW}" \
  --seqlen "${SEQLEN}" \
  --num-documents "${NUM_DOCUMENTS}" \
  --doc-start "${DOC_START}" \
  --num-shards "${NUM_SHARDS}" \
  --shard-index "${shard}" \
  --round-name "${round}" \
  --batch-size "${batch_size}" \
  --position-bin-size "${POSITION_BIN_SIZE}" \
  --data-seed "${DATA_SEED}" \
  --output-dir "${out_dir}"
