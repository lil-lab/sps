#!/bin/bash
# Submitter for the SHARDED multiposition parameter-gradient analysis.
#
# Samples NUM_DOCUMENTS documents x POSITIONS_PER_DOCUMENT source positions per
# model (default 2000 x 8 = 16000 targets) using batched_gradient_mode=
# per_row_position and the columnar binary output format. Work is split into
# NUM_SHARDS shards; one SLURM array task computes one (model, shard) and writes
# <out_dir>/shard_<idx>/. A dependent CPU job then merges each model's shards and
# writes its summary JSON/CSV.
#
# Usage:
#   sbatch? NO -- run this directly (it submits the jobs):
#     bash scripts/analysis/run_gradient_analysis_params.sh
#   Override via env, e.g.:
#     SCALES="l xl" FAMILIES="sps" NUM_SHARDS=64 \
#       bash scripts/analysis/run_gradient_analysis_params.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
mkdir -p logs/slurm

# Where outputs and checkpoints live. Defaults are repo-relative (DATA_ROOT=repo
# root); point them at a scratch filesystem via env if home is quota-limited.
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/gradient_analysis_params/multiposition}"
OUT_ROOT="${OUT_ROOT:-${DATA_ROOT}/out}"
CKPT_FILE="${CKPT_FILE:-ckpt_tokens_20000145408_final.pt}"

# ---- SLURM (cluster-specific; override via env, like scripts/benchmark/run.sh) ----
PARTITION="${PARTITION:?Set PARTITION to your GPU partition (e.g. PARTITION=gpu)}"
ACCOUNT="${ACCOUNT:-}"                 # SLURM account; blank => omit --account
GRES="${GRES:-}"                       # gres type, e.g. nvidia_h100_80gb_hbm3; blank => gpu:1
MERGE_PARTITION="${MERGE_PARTITION:-$PARTITION}"   # CPU merge job (a CPU partition is ideal)
MERGE_ACCOUNT="${MERGE_ACCOUNT:-$ACCOUNT}"

# Reusable sbatch resource args (mirrors scripts/benchmark/run.sh).
GPU_ARGS=(); if [[ -n "$GRES" ]]; then GPU_ARGS=(--gres=gpu:"$GRES":1); else GPU_ARGS=(--gres=gpu:1); fi
ACCT_ARGS=(); [[ -n "$ACCOUNT" ]] && ACCT_ARGS=(--account="$ACCOUNT")
MERGE_ACCT_ARGS=(); [[ -n "$MERGE_ACCOUNT" ]] && MERGE_ACCT_ARGS=(--account="$MERGE_ACCOUNT")

SCALES="${SCALES:-s m l xl}"
FAMILIES="${FAMILIES:-full_attention sps delayed_state}"
WARP_SPECIALIZE="${WARP_SPECIALIZE:-}"

# Multiposition sampling hypers.
SEQLEN="${SEQLEN:-1024}"
FUTURE_HORIZON="${FUTURE_HORIZON:-512}"
# Documents [DOC_START, NUM_DOCUMENTS) of the stable seeded order. To ADD samples
# later, re-run with DOC_START=<old NUM_DOCUMENTS> NUM_DOCUMENTS=<new total>; the
# new round is written alongside and folded in at merge without recomputing.
DOC_START="${DOC_START:-0}"
NUM_DOCUMENTS="${NUM_DOCUMENTS:-2000}"
POSITIONS_PER_DOCUMENT="${POSITIONS_PER_DOCUMENT:-8}"
NUM_SHARDS="${NUM_SHARDS:-32}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SKIP_FIRST="${SKIP_FIRST:-64}"
DATA_SEED="${DATA_SEED:-42}"
POSITION_SEED="${POSITION_SEED:-123}"
FORWARD_CHUNK_SIZE="${FORWARD_CHUNK_SIZE:-0}"
# Cap concurrently-running array tasks (GPU availability).
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-16}"

# (run_name suffix, kind, label) per family.
FAMILY_SPECS=(
  "full_attention_20b|full_attention|Full Attention"
  "sps_w64_20b|sps|SPS w64"
  "delayed_state_w64_20b|delayed_state|Delayed State w64"
)

# Build the model manifest (one line per model with an existing checkpoint):
#   out_dir<TAB>kind<TAB>label<TAB>ckpt<TAB>warp
MANIFEST="logs/slurm/grad_params_manifest_$(date +%s).tsv"
: > "$MANIFEST"
for scale in $SCALES; do
  for entry in "${FAMILY_SPECS[@]}"; do
    IFS='|' read -r family kind label <<<"$entry"
    if [[ " ${FAMILIES} " != *" ${kind} "* ]]; then
      continue
    fi
    run_name="${scale}_${family}"
    out_dir="${OUTPUT_ROOT}/${run_name}"
    ckpt_dir="${run_name}"
    ckpt="${OUT_ROOT}/${ckpt_dir}/${CKPT_FILE}"
    warp="${WARP_SPECIALIZE}"
    if [[ -z "${warp}" ]]; then
      if [[ "${scale}" == "xl" ]]; then warp="off"; else warp="keep"; fi
    fi
    if [[ ! -f "${ckpt}" ]]; then
      echo "SKIP ${run_name}: checkpoint not found at ${ckpt}"
      continue
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "${out_dir}" "${kind}" "${label}" "${ckpt}" "${warp}" >> "$MANIFEST"
  done
done

NMODELS="$(wc -l < "$MANIFEST")"
if [[ "${NMODELS}" -eq 0 ]]; then
  echo "No models with checkpoints found; nothing to submit."
  exit 1
fi
NTASKS=$(( NMODELS * NUM_SHARDS ))
echo "Manifest: ${MANIFEST} (${NMODELS} models)"
echo "Submitting compute array of ${NTASKS} tasks (${NMODELS} models x ${NUM_SHARDS} shards), %${ARRAY_CONCURRENCY} concurrent."

export OUTPUT_ROOT OUT_ROOT MANIFEST SEQLEN FUTURE_HORIZON DOC_START NUM_DOCUMENTS \
  POSITIONS_PER_DOCUMENT NUM_SHARDS BATCH_SIZE SKIP_FIRST DATA_SEED POSITION_SEED \
  FORWARD_CHUNK_SIZE

# --- Reliable dedicated array on $PARTITION (non-preemptible). It covers EVERY
#     shard (ascending) and is the completion backstop + merge gate.
#     ENABLE_RELIABLE=0 = scavenger-only mode (add more pools to an existing run
#     without spawning a second reliable array or merge).
compute_jid=""
if [[ "${ENABLE_RELIABLE:-1}" == "1" ]]; then
  compute_jid=$(sbatch --parsable \
    --partition="$PARTITION" "${ACCT_ARGS[@]}" "${GPU_ARGS[@]}" \
    --array=0-$((NTASKS - 1))%"${ARRAY_CONCURRENCY}" \
    --export=ALL,ORDER=asc \
    scripts/analysis/_grad_params_compute.sh)
  echo "Reliable compute array ($PARTITION, asc, %${ARRAY_CONCURRENCY}): ${compute_jid}"
else
  echo "ENABLE_RELIABLE=0: scavenger-only mode (no reliable array, no merge)."
fi

# --- Optional preemptible scavengers: same task pool swept from the other end
#     (ORDER=desc) on additional partitions to stack GPUs past a per-user cap.
#     They only accelerate; the reliable array guarantees completion. Empty by
#     default; set space-separated "partition:account:order:concurrency[:gres]"
#     specs, e.g. SCAVENGE_SPECS="gpu_requeue:myacct:desc:64".
SCAVENGE_SPECS="${SCAVENGE_SPECS:-}"
if [[ "${ENABLE_SCAVENGE:-1}" == "1" ]]; then
  for spec in ${SCAVENGE_SPECS}; do
    # spec = partition:account:order:concurrency[:gres]. The optional 5th field
    # sets this pool's gres (e.g. gpu:nvidia_h200:1) so a differently-provisioned
    # pool can join; it may itself contain ':' -- read captures the remainder
    # verbatim into s_gres. Omitted => fall back to the global GRES.
    IFS=':' read -r s_part s_acct s_order s_conc s_gres <<<"${spec}"
    if [[ -n "${s_gres:-}" ]]; then sc_gpu=(--gres="${s_gres}"); else sc_gpu=("${GPU_ARGS[@]}"); fi
    sc_acct=(); [[ -n "${s_acct}" ]] && sc_acct=(--account="${s_acct}")
    if jid=$(sbatch --parsable \
        --partition="${s_part}" "${sc_acct[@]}" "${sc_gpu[@]}" \
        --array=0-$((NTASKS - 1))%"${s_conc}" \
        --export=ALL,ORDER="${s_order}" \
        scripts/analysis/_grad_params_compute.sh 2>/dev/null); then
      echo "Scavenger array (${s_part}, ${s_order}, %${s_conc}${s_gres:+, ${s_gres}}): ${jid}"
    else
      echo "Scavenger ${s_part} (acct ${s_acct}) not available; skipping."
    fi
  done
fi

# --- Merge depends only on the reliable array: when it finishes, every shard is
#     present (computed by it or skipped because a scavenger published it first).
if [[ "${ENABLE_MERGE:-1}" == "1" && -n "${compute_jid}" ]]; then
  merge_jid=$(sbatch --parsable \
    --partition="$MERGE_PARTITION" "${MERGE_ACCT_ARGS[@]}" \
    --dependency=afterok:"${compute_jid}" \
    --export=ALL \
    scripts/analysis/_grad_params_merge.sh)
  echo "Merge job (afterok ${compute_jid}): ${merge_jid}"
else
  echo "Skipping merge submission (ENABLE_MERGE=0 or scavenger-only mode)."
fi
echo "Outputs -> ${OUTPUT_ROOT}"
