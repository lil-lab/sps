#!/bin/bash
# Submitter for the SHARDED / incremental persistent-window NLL analysis.
#
# Measures, per (scale, model), the per-document-relative-position NLL degradation
# from forcing a reduced persistent-key window (w'=PERSISTENT_WINDOW), over
# NUM_DOCUMENTS documents sampled WITHOUT replacement from a fixed seeded
# permutation (aligned with gradient_analysis_params). Each per-position mean has
# noise ~1/sqrt(num_documents), so adding documents straightforwardly de-noises
# the delta-NLL curve. Work is split into NUM_SHARDS shards; one SLURM array task
# computes one (model, shard) and writes <out_dir>/r_<...>/shard_<idx>/ columns.
# A dependent CPU job then merges each model's shards and writes one combined
# per-scale summary (<root>/<scale>/pw<w>/persistent_window_nll_summary.json) that
# the figure scripts read.
#
# Usage (run directly; it submits the jobs):
#   NUM_DOCUMENTS=8000 bash scripts/analysis/run_persistent_window_nll.sh
# Add more samples later WITHOUT recomputing (folded in at merge):
#   DOC_START=8000 NUM_DOCUMENTS=12000 bash scripts/analysis/run_persistent_window_nll.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
mkdir -p logs/slurm

# Where outputs and checkpoints live. Defaults are repo-relative (DATA_ROOT=repo
# root); point them at a scratch filesystem via env if home is quota-limited.
DATA_ROOT="${DATA_ROOT:-$ROOT_DIR}"
RUN_NAME="${RUN_NAME:-xs_s_m_l_xl_w64_sharded_pw64}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${DATA_ROOT}/persistent_window_nll_analysis/${RUN_NAME}}"
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

SCALES="${SCALES:-xs s m l xl}"
FAMILIES="${FAMILIES:-sps delayed_state}"
WARP_SPECIALIZE="${WARP_SPECIALIZE:-}"

# Measurement hypers.
PERSISTENT_WINDOW="${PERSISTENT_WINDOW:-64}"
SEQLEN="${SEQLEN:-2048}"
POSITION_BIN_SIZE="${POSITION_BIN_SIZE:-64}"
DATA_SEED="${DATA_SEED:-42}"
# Documents [DOC_START, NUM_DOCUMENTS) of the stable seeded order. To ADD samples
# later, re-run with DOC_START=<old NUM_DOCUMENTS> NUM_DOCUMENTS=<new total>; the
# new round is written alongside and folded in at merge without recomputing.
# Hard ceiling = eligible documents with >= SEQLEN+1 boundary-free tokens (~15194).
DOC_START="${DOC_START:-0}"
NUM_DOCUMENTS="${NUM_DOCUMENTS:-8000}"
NUM_SHARDS="${NUM_SHARDS:-16}"
# Cap concurrently-running array tasks (GPU availability).
ARRAY_CONCURRENCY="${ARRAY_CONCURRENCY:-16}"

# Per-scale dense-forward batch sizes (seqlen=2048); known-safe from prior runs.
# A non-empty global BATCH_SIZE overrides all; per-scale BATCH_<SC> overrides one.
batch_for_scale() {
  if [[ -n "${BATCH_SIZE:-}" ]]; then echo "${BATCH_SIZE}"; return; fi
  case "$1" in
    xs) echo "${BATCH_XS:-16}" ;;
    s)  echo "${BATCH_S:-16}" ;;
    m)  echo "${BATCH_M:-8}" ;;
    l)  echo "${BATCH_L:-4}" ;;
    xl) echo "${BATCH_XL:-2}" ;;
    *)  echo "8" ;;
  esac
}

# (canonical kind, label) per family.
FAMILY_SPECS=(
  "sps|SPS w64"
  "delayed_state|Delayed State w64"
)

# Build the model manifest (one line per (scale, model) with an existing checkpoint):
#   out_dir<TAB>scale<TAB>kind<TAB>label<TAB>ckpt<TAB>warp<TAB>batch_size
MANIFEST="logs/slurm/pwnll_manifest_$(date +%s).tsv"
: > "$MANIFEST"
for scale in $SCALES; do
  bs="$(batch_for_scale "${scale}")"
  for entry in "${FAMILY_SPECS[@]}"; do
    IFS='|' read -r kind label <<<"$entry"
    if [[ " ${FAMILIES} " != *" ${kind} "* ]]; then
      continue
    fi
    ckpt_dir="${scale}_${kind}_w64_20b"
    ckpt="${OUT_ROOT}/${ckpt_dir}/${CKPT_FILE}"
    out_dir="${OUTPUT_ROOT}/${scale}/${kind}"
    warp="${WARP_SPECIALIZE}"
    if [[ -z "${warp}" ]]; then
      if [[ "${scale}" == "xl" ]]; then warp="off"; else warp="keep"; fi
    fi
    if [[ ! -f "${ckpt}" ]]; then
      echo "SKIP ${scale}/${kind}: checkpoint not found at ${ckpt}"
      continue
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${out_dir}" "${scale}" "${kind}" "${label}" "${ckpt}" "${warp}" "${bs}" >> "$MANIFEST"
  done
done

NMODELS="$(grep -c . "$MANIFEST" || true)"
if [[ "${NMODELS}" -eq 0 ]]; then
  echo "No (scale, model) pairs with checkpoints found; nothing to submit."
  exit 1
fi
NTASKS=$(( NMODELS * NUM_SHARDS ))
echo "Manifest: ${MANIFEST} (${NMODELS} models)"
echo "Round r_${DOC_START}_${NUM_DOCUMENTS}_s${NUM_SHARDS}: ${NUM_DOCUMENTS} docs, ${NUM_SHARDS} shards/model."
echo "Submitting compute array of ${NTASKS} tasks (${NMODELS} models x ${NUM_SHARDS} shards), %${ARRAY_CONCURRENCY} concurrent."

export OUTPUT_ROOT OUT_ROOT MANIFEST PERSISTENT_WINDOW SEQLEN POSITION_BIN_SIZE \
  DATA_SEED DOC_START NUM_DOCUMENTS NUM_SHARDS

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
    scripts/analysis/_persistent_window_nll_compute.sh)
  echo "Reliable compute array ($PARTITION, asc, %${ARRAY_CONCURRENCY}): ${compute_jid}"
else
  echo "ENABLE_RELIABLE=0: scavenger-only mode (no reliable array, no merge)."
fi

# --- Optional preemptible scavengers: same task pool swept from the other end
#     (ORDER=desc) on additional partitions to stack GPUs past a per-user cap. They
#     only accelerate; the reliable array guarantees completion. Empty by default;
#     set space-separated "partition:account:order:concurrency[:gres]" specs,
#     e.g. SCAVENGE_SPECS="gpu_requeue:myacct:desc:64".
SCAVENGE_SPECS="${SCAVENGE_SPECS:-}"
if [[ "${ENABLE_SCAVENGE:-1}" == "1" ]]; then
  for spec in ${SCAVENGE_SPECS}; do
    # spec = partition:account:order:concurrency[:gres]; omitted gres => global GRES.
    IFS=':' read -r s_part s_acct s_order s_conc s_gres <<<"${spec}"
    if [[ -n "${s_gres:-}" ]]; then sc_gpu=(--gres="${s_gres}"); else sc_gpu=("${GPU_ARGS[@]}"); fi
    sc_acct=(); [[ -n "${s_acct}" ]] && sc_acct=(--account="${s_acct}")
    if jid=$(sbatch --parsable \
        --partition="${s_part}" "${sc_acct[@]}" "${sc_gpu[@]}" \
        --array=0-$((NTASKS - 1))%"${s_conc}" \
        --export=ALL,ORDER="${s_order}" \
        scripts/analysis/_persistent_window_nll_compute.sh 2>/dev/null); then
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
    scripts/analysis/_persistent_window_nll_merge.sh)
  echo "Merge job (afterok ${compute_jid}): ${merge_jid}"
else
  echo "Skipping merge submission (ENABLE_MERGE=0 or scavenger-only mode)."
fi
echo "Outputs -> ${OUTPUT_ROOT}"
