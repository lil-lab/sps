#!/bin/bash
#SBATCH --job-name=grad_params_merge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/grad_params_merge_%j.out
#SBATCH --error=logs/slurm/grad_params_merge_%j.err
#
# Merge each model's shard_* subdirs (CPU only) and write summary JSON/CSV.
# --partition / --account come from the submitter (MERGE_PARTITION / MERGE_ACCOUNT).
# Submitted with afterok dependency on the compute array by
# run_gradient_analysis_params.sh (reads the exported MANIFEST).
set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT_DIR"

while IFS=$'\t' read -r out_dir kind label ckpt warp; do
  [[ -z "${out_dir}" ]] && continue
  echo "=== merge ${out_dir} (${kind} ${label}) ==="
  uv run python -m scripts.analysis.gradient_analysis_params --merge "${out_dir}"
done < "$MANIFEST"

echo "All merges complete -> ${OUTPUT_ROOT}"
