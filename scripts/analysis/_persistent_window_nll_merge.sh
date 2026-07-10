#!/bin/bash
#SBATCH --job-name=pwnll_merge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/pwnll_merge_%j.out
#SBATCH --error=logs/slurm/pwnll_merge_%j.err
#
# Per scale: merge each model's shard_* subdirs (all rounds, CPU only), then write
# one combined summary <root>/<scale>/pw<w>/persistent_window_nll_summary.json that
# the figure scripts read. --partition / --account come from the submitter
# (MERGE_PARTITION / MERGE_ACCOUNT). Submitted with afterok on the compute array by
# run_persistent_window_nll.sh (reads the exported MANIFEST).
set -euo pipefail

ROOT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$ROOT_DIR"

COMBINE_MODELS="${COMBINE_MODELS:-sps,delayed_state}"

# Unique scales from the manifest (field 2).
scales="$(cut -f2 "$MANIFEST" | sort -u)"
for scale in ${scales}; do
  echo "=== combine ${OUTPUT_ROOT}/${scale} (models: ${COMBINE_MODELS}) ==="
  uv run python -m scripts.analysis.persistent_window_nll_analysis \
    --combine "${OUTPUT_ROOT}/${scale}" \
    --combine-models "${COMBINE_MODELS}" \
    --persistent-window "${PERSISTENT_WINDOW}" \
    --position-bin-size "${POSITION_BIN_SIZE}"
done

echo "All combines complete -> ${OUTPUT_ROOT}"
