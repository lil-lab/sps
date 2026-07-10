#!/usr/bin/env bash
# Config-driven SLURM launcher for the generation-speed benchmark.
#
# Supersedes submit_generation_timing_correctness_test_gpu.sh and
# submit_generation_timing_fast_test_gpu.sh. The benchmark hyperparameters come from a
# cluster-independent config (conf/benchmark/<BENCHMARK>.yaml); everything cluster- or
# hardware-specific is passed as an environment variable here -- nothing is hardcoded.
#
# Reproduce the main-table speed columns on any cluster:
#
#   OUT_ROOT=/path/to/your/checkpoints \
#   VAL_BIN=/path/to/fineweb-edu/val.bin \
#   WARP_SPECIALIZE=off \
#   PARTITION=<gpu-partition> ACCOUNT=<slurm-account> GRES=<gres-type> \
#   bash scripts/benchmark/run.sh
#
# BENCHMARK selects conf/benchmark/<BENCHMARK>.yaml and defaults to 'speed' (the only
# shipped config). Add DRY_RUN=1 to print the resolved plan + validate checkpoints first.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# ---- Required (cluster-specific) ----
OUT_ROOT="${OUT_ROOT:?Set OUT_ROOT to your checkpoint root (contains <run_name>/<ckpt>)}"

# ---- Optional (cluster-specific) ----
BENCHMARK="${BENCHMARK:-speed}"        # selects conf/benchmark/<BENCHMARK>.yaml; only 'speed' ships
CKPT_NAME="${CKPT_NAME:-ckpt_tokens_20000145408_final.pt}"
VAL_BIN="${VAL_BIN:-}"                 # exported to the driver as BENCH_VAL_BIN
SCALES="${SCALES:-}"                   # blank => use the scales from the config
NUM_PROMPTS="${NUM_PROMPTS:-}"         # blank => batch size from the config (override, e.g. 16 for XL w4096)
METHODS="${METHODS:-}"                 # blank => all methods; else comma-separated labels (e.g. '2x Memory')
WARP_SPECIALIZE="${WARP_SPECIALIZE:-}" # keep|on|off; XL SPS kernels need 'off' (current Triton can't compile warp_specialize)
DEVICE="${DEVICE:-cuda}"

# ---- SLURM resources (cluster-specific) ----
PARTITION="${PARTITION:?Set PARTITION to your GPU partition}"
ACCOUNT="${ACCOUNT:-}"
GRES="${GRES:-}"                       # gres type, e.g. nvidia_h100_80gb_hbm3 (blank => gpu:1)
CONSTRAINT="${CONSTRAINT:-}"
NODELIST="${NODELIST:-}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
MEM="${MEM:-160G}"
CPUS="${CPUS:-8}"
MAIL_USER="${MAIL_USER:-}"
MAIL_TYPE="${MAIL_TYPE:-NONE}"

LOG_DIR="${LOG_DIR:-$ROOT_DIR/logs/slurm}"
SKIP_CKPT_CHECK="${SKIP_CKPT_CHECK:-0}"
DRY_RUN="${DRY_RUN:-0}"

scale_tag() {
  if [[ -n "$SCALES" ]]; then echo "$SCALES" | tr ' ' '_'; else echo "default"; fi
}
RESULT_ROOT="${RESULT_ROOT:-$ROOT_DIR/outputs/generation_timing_correctness/${BENCHMARK}_$(scale_tag)_$(echo "${GRES:-$PARTITION}" | tr -cd '[:alnum:]_')}"
OUTPUT_JSON="$RESULT_ROOT/results.json"
DRIVER_ARGS_FILE="$RESULT_ROOT/driver_args.txt"

mkdir -p "$LOG_DIR" "$RESULT_ROOT"

# ---- Resolve config -> driver args (cluster-independent hyperparams + per-spec) ----
loader_args=("$BENCHMARK" --out-root "$OUT_ROOT" --ckpt-name "$CKPT_NAME")
[[ -n "$SCALES" ]]      && loader_args+=(--scales "$SCALES")
[[ -n "$NUM_PROMPTS" ]]     && loader_args+=(--num-prompts "$NUM_PROMPTS")
[[ -n "$METHODS" ]]         && loader_args+=(--methods "$METHODS")
[[ -n "$WARP_SPECIALIZE" ]] && loader_args+=(--warp-specialize "$WARP_SPECIALIZE")

uv run python "$ROOT_DIR/scripts/benchmark/_load_config.py" "${loader_args[@]}" --emit argv > "$DRIVER_ARGS_FILE"

# ---- Validate every checkpoint exists before burning a GPU allocation ----
missing=0
while IFS= read -r ckpt; do
  if [[ ! -f "$ckpt" ]]; then
    echo "Missing checkpoint: $ckpt" >&2
    missing=$((missing + 1))
  fi
done < <(uv run python "$ROOT_DIR/scripts/benchmark/_load_config.py" "${loader_args[@]}" --emit checkpoints)
if [[ "$missing" -gt 0 && "$SKIP_CKPT_CHECK" != "1" ]]; then
  echo "$missing checkpoint(s) missing under OUT_ROOT=$OUT_ROOT. Set SKIP_CKPT_CHECK=1 to override." >&2
  exit 1
fi

# ---- Assemble sbatch resource args ----
SBATCH_GPU_ARGS=()
if [[ -n "$GRES" ]]; then SBATCH_GPU_ARGS+=(--gres=gpu:"$GRES":1); else SBATCH_GPU_ARGS+=(--gres=gpu:1); fi
[[ -n "$CONSTRAINT" ]] && SBATCH_GPU_ARGS+=(--constraint="$CONSTRAINT")
[[ -n "$NODELIST" ]]   && SBATCH_GPU_ARGS+=(--nodelist="$NODELIST")

SBATCH_EXTRA=()
[[ -n "$ACCOUNT" ]]   && SBATCH_EXTRA+=(--account="$ACCOUNT")
[[ -n "$MAIL_USER" ]] && SBATCH_EXTRA+=(--mail-user="$MAIL_USER" --mail-type="$MAIL_TYPE")

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] benchmark=$BENCHMARK"
  echo "  out_root=$OUT_ROOT  ckpt_name=$CKPT_NAME"
  echo "  val_bin=${VAL_BIN:-<driver default / BENCH_VAL_BIN>}"
  echo "  scales=${SCALES:-<from config>}  device=$DEVICE"
  echo "  partition=$PARTITION account=${ACCOUNT:-<none>} gres=${GRES:-gpu:1} constraint=${CONSTRAINT:-<none>}"
  echo "  result_root=$RESULT_ROOT"
  echo "  checkpoints missing=$missing"
  echo "  driver args:"
  sed 's/^/    /' "$DRIVER_ARGS_FILE"
  exit 0
fi

job_id=$(
  sbatch --parsable \
    --partition="$PARTITION" \
    "${SBATCH_EXTRA[@]}" \
    "${SBATCH_GPU_ARGS[@]}" \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME_LIMIT" \
    --job-name="bench_${BENCHMARK}" \
    --output="$LOG_DIR/%x_%j.out" \
    --error="$LOG_DIR/%x_%j.err" \
    --export=ALL,ROOT_DIR="$ROOT_DIR",DRIVER_ARGS_FILE="$DRIVER_ARGS_FILE",OUTPUT_JSON="$OUTPUT_JSON",DEVICE="$DEVICE",BENCH_VAL_BIN="$VAL_BIN",BENCHMARK="$BENCHMARK" \
    <<'SBATCH_EOF'
#!/usr/bin/env bash
set -euo pipefail
cd "$ROOT_DIR"

echo "=== generation-speed benchmark: $BENCHMARK ==="
echo "Host: $(hostname)  Date: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# BENCH_VAL_BIN is read by the driver as the --val-bin default; unset it if blank so the
# driver falls back to its built-in default rather than an empty path.
[[ -z "${BENCH_VAL_BIN:-}" ]] && unset BENCH_VAL_BIN
# torch >= 2.9 renamed PYTORCH_CUDA_ALLOC_CONF -> PYTORCH_ALLOC_CONF (old name warns and
# may be ignored). Set both so expandable_segments actually takes effect across versions.
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

mapfile -t DRIVER_ARGS < "$DRIVER_ARGS_FILE"
/usr/bin/time -v uv run python scripts/benchmark/benchmark_generation_speed.py \
  "${DRIVER_ARGS[@]}" \
  --device "$DEVICE" \
  --output-json "$OUTPUT_JSON"

echo "=== Done -> $OUTPUT_JSON ==="
SBATCH_EOF
)

echo "Submitted benchmark '$BENCHMARK': job $job_id"
echo "Results: $OUTPUT_JSON"
echo "Logs:    $LOG_DIR/bench_${BENCHMARK}_${job_id}.out"
