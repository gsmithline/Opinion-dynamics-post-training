#!/usr/bin/env bash
# FJ EGTA cross-evaluation runner for HTCondor.
# Loads each intervention's per-round adapter (or base model for static) and
# computes M[t,i,j] = mean NLL(model_j; prompts_i) over the labeled subset.
#
# Inputs come from env vars; defaults assume the bundled K=10 sweep:
#   5 SFT/SFT-KL betas + 4 RL-KL betas + 1 untuned-static row.
#
# Args: (none) -- everything driven by env to keep the .sub simple.
set -eo pipefail

REPO="${REPO:-/home/gsmithline/Opinion-dynamics-post-training}"
EGTA_REPO="${EGTA_REPO:-/home/gsmithline/evolutionary-prediction-games}"
CONDA_SH="${CONDA_SH:-/home/gsmithline/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-opdyn}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-/home/gsmithline/.wandb_key}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
SNAPSHOTS="${SNAPSHOTS:-0 5 10 15 20 25 30}"
PROMPTS_SCOPE="${PROMPTS_SCOPE:-labeled}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_LENGTH="${MAX_LENGTH:-200}"
OUT="${OUT:-${REPO}/pokec_dataset/results/egta_fj_bundle.npz}"
WANDB_NAME="${WANDB_NAME:-egta_fj_bundle}"

echo "[run_egta_fj] host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "[run_egta_fj] base_model=$BASE_MODEL snapshots=[$SNAPSHOTS] scope=$PROMPTS_SCOPE"
echo "[run_egta_fj] out=$OUT"

if [ ! -d "$EGTA_REPO/LLM_experiments" ]; then
    echo "[run_egta_fj] FATAL: $EGTA_REPO/LLM_experiments not found. Set EGTA_REPO." >&2
    exit 2
fi

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"

if [ -f "$WANDB_KEY_FILE" ]; then
    export WANDB_API_KEY="$(tr -d '[:space:]' < "$WANDB_KEY_FILE")"
fi
export WANDB_DIR="${WANDB_DIR:-$REPO/wandb}"
export HF_HOME="${HF_HOME:-/home/gsmithline/.cache/huggingface}"
export PYTHONPATH="${EGTA_REPO}:${PYTHONPATH:-}"

RES="${REPO}/pokec_dataset/results"
ADP="${REPO}/adapters"

# Trained interventions: (record, adapter_dir, label) triples.
RUNS=(
    "${RES}/llm_fj_sft_b0_egta_trajectory.pk      ${ADP}/fj_sft_b0_egta      sft_b0"
    "${RES}/llm_fj_sftkl_b0p01_egta_trajectory.pk ${ADP}/fj_sftkl_b0p01_egta sftkl_b0.01"
    "${RES}/llm_fj_sftkl_b0p1_egta_trajectory.pk  ${ADP}/fj_sftkl_b0p1_egta  sftkl_b0.1"
    "${RES}/llm_fj_sftkl_b1_egta_trajectory.pk    ${ADP}/fj_sftkl_b1_egta    sftkl_b1"
    "${RES}/llm_fj_sftkl_b10_egta_trajectory.pk   ${ADP}/fj_sftkl_b10_egta   sftkl_b10"
    "${RES}/llm_fj_rlkl_b0p01_egta_trajectory.pk  ${ADP}/fj_rlkl_b0p01_egta  rlkl_b0.01"
    "${RES}/llm_fj_rlkl_b0p1_egta_trajectory.pk   ${ADP}/fj_rlkl_b0p1_egta   rlkl_b0.1"
    "${RES}/llm_fj_rlkl_b1_egta_trajectory.pk     ${ADP}/fj_rlkl_b1_egta     rlkl_b1"
    "${RES}/llm_fj_rlkl_b10_egta_trajectory.pk    ${ADP}/fj_rlkl_b10_egta    rlkl_b10"
)

# Static (no-training) row. Inherits prompts from the first --run.
STATIC_RECORD="${RES}/llm_llm_untuned_fj_untuned_egta_trajectory.pk"
STATIC_LABEL="untuned"

# Sanity-check every artifact before launching the (long) cross-eval.
missing=0
for triple in "${RUNS[@]}"; do
    # shellcheck disable=SC2086
    set -- $triple
    rec="$1"; adir="$2"; lbl="$3"
    if [ ! -f "$rec" ]; then
        echo "[run_egta_fj] MISSING record: $rec ($lbl)" >&2; missing=1
    fi
    if [ ! -d "$adir" ]; then
        echo "[run_egta_fj] MISSING adapter dir: $adir ($lbl)" >&2; missing=1
    fi
    if [ ! -f "$adir/prompts_labeled.json" ]; then
        echo "[run_egta_fj] MISSING prompts_labeled.json in: $adir ($lbl)" >&2; missing=1
    fi
    for t in $SNAPSHOTS; do
        if [ ! -d "$adir/round_${t}" ]; then
            echo "[run_egta_fj] MISSING snapshot: $adir/round_${t} ($lbl)" >&2; missing=1
        fi
    done
done
if [ ! -f "$STATIC_RECORD" ]; then
    echo "[run_egta_fj] MISSING static record: $STATIC_RECORD" >&2; missing=1
fi
if [ "$missing" -ne 0 ]; then
    echo "[run_egta_fj] one or more artifacts missing -- aborting before cross-eval." >&2
    exit 3
fi

cd "$EGTA_REPO"

# Build argv. Each --run takes three positional args.
ARGS=(-m LLM_experiments.egta_fj
      --base-model "$BASE_MODEL"
      --prompts-scope "$PROMPTS_SCOPE"
      --batch-size "$BATCH_SIZE"
      --max-length "$MAX_LENGTH"
      --snapshot-times $SNAPSHOTS
      --out "$OUT"
      --wandb --wandb-project egta-analysis --wandb-name "$WANDB_NAME")
for triple in "${RUNS[@]}"; do
    # shellcheck disable=SC2086
    set -- $triple
    ARGS+=(--run "$1" "$2" "$3")
done
ARGS+=(--static-run "$STATIC_RECORD" "$BASE_MODEL" "$STATIC_LABEL")

echo "[run_egta_fj] launching: python ${ARGS[*]}"
python "${ARGS[@]}"

echo "[run_egta_fj] done -> $OUT"
