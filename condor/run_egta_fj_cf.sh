#!/usr/bin/env bash
# FJ EGTA cross-evaluation runner for the CF bundle.
# K=10: 5 SFT-KL trained rows + 4 CF static rows + 1 untuned static row.
#
# Differs from run_egta_fj.sh by replacing the 4 RL-KL trained rows with
# 4 CF rows. CF runs are passed as --static-run because the LM never trains;
# the policy is q*(v_k|x) = pi_ref * h_p^(1/beta) computed analytically, and
# the "model column" at every snapshot is just the base LM (which CF used to
# score pi_ref each round).
set -eo pipefail

REPO="${REPO:-/home/gsmithline/Opinion-dynamics-post-training}"
EGTA_REPO="${EGTA_REPO:-/home/gsmithline/evolutionary-prediction-games}"
CONDA_SH="${CONDA_SH:-/home/gsmithline/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-opdyn}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-/home/gsmithline/.wandb_key}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
SNAPSHOTS="${SNAPSHOTS:-5 10 15 20 25 30}"
PROMPTS_SCOPE="${PROMPTS_SCOPE:-labeled}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_LENGTH="${MAX_LENGTH:-200}"
OUT="${OUT:-${REPO}/pokec_dataset/results/egta_fj_bundle_cf.npz}"
WANDB_NAME="${WANDB_NAME:-egta_fj_bundle_cf}"

echo "[run_egta_fj_cf] host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "[run_egta_fj_cf] base_model=$BASE_MODEL snapshots=[$SNAPSHOTS] scope=$PROMPTS_SCOPE"
echo "[run_egta_fj_cf] out=$OUT"

if [ ! -d "$EGTA_REPO/LLM_experiments" ]; then
    echo "[run_egta_fj_cf] FATAL: $EGTA_REPO/LLM_experiments not found." >&2
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

# SFT-KL trained rows (kept from original bundle).
TRAINED_RUNS=(
    "${RES}/llm_fj_sft_b0_egta_trajectory.pk      ${ADP}/fj_sft_b0_egta      sft_b0"
    "${RES}/llm_fj_sftkl_b0p01_egta_trajectory.pk ${ADP}/fj_sftkl_b0p01_egta sftkl_b0.01"
    "${RES}/llm_fj_sftkl_b0p1_egta_trajectory.pk  ${ADP}/fj_sftkl_b0p1_egta  sftkl_b0.1"
    "${RES}/llm_fj_sftkl_b1_egta_trajectory.pk    ${ADP}/fj_sftkl_b1_egta    sftkl_b1"
    "${RES}/llm_fj_sftkl_b10_egta_trajectory.pk   ${ADP}/fj_sftkl_b10_egta   sftkl_b10"
)

# CF static rows. Each pairs a CF trajectory with the base LM model column.
# Labels use the cf_b<beta> shape so the default bifurcation_plot regex matches.
STATIC_RUNS=(
    "${RES}/llm_llm_untuned_fj_untuned_egta_trajectory.pk Qwen/Qwen2.5-0.5B-Instruct untuned"
    "${RES}/llm_fj_cf_b0p01_egta_trajectory.pk            Qwen/Qwen2.5-0.5B-Instruct cf_b0.01"
    "${RES}/llm_fj_cf_b0p1_egta_trajectory.pk             Qwen/Qwen2.5-0.5B-Instruct cf_b0.1"
    "${RES}/llm_fj_cf_b1_egta_trajectory.pk               Qwen/Qwen2.5-0.5B-Instruct cf_b1"
    "${RES}/llm_fj_cf_b10_egta_trajectory.pk              Qwen/Qwen2.5-0.5B-Instruct cf_b10"
)

# Pre-flight checks.
missing=0
for triple in "${TRAINED_RUNS[@]}"; do
    # shellcheck disable=SC2086
    set -- $triple
    rec="$1"; adir="$2"; lbl="$3"
    if [ ! -f "$rec" ]; then
        echo "[run_egta_fj_cf] MISSING record: $rec ($lbl)" >&2; missing=1
    fi
    if [ ! -d "$adir" ]; then
        echo "[run_egta_fj_cf] MISSING adapter dir: $adir ($lbl)" >&2; missing=1
    fi
    if [ ! -f "$adir/prompts_labeled.json" ]; then
        echo "[run_egta_fj_cf] MISSING prompts_labeled.json in: $adir ($lbl)" >&2; missing=1
    fi
    for t in $SNAPSHOTS; do
        if [ ! -d "$adir/round_${t}" ]; then
            echo "[run_egta_fj_cf] MISSING snapshot: $adir/round_${t} ($lbl)" >&2; missing=1
        fi
    done
done
for triple in "${STATIC_RUNS[@]}"; do
    # shellcheck disable=SC2086
    set -- $triple
    rec="$1"; lbl="$3"
    if [ ! -f "$rec" ]; then
        echo "[run_egta_fj_cf] MISSING static record: $rec ($lbl)" >&2; missing=1
    fi
done
if [ "$missing" -ne 0 ]; then
    echo "[run_egta_fj_cf] one or more artifacts missing -- aborting." >&2
    exit 3
fi

cd "$EGTA_REPO"

ARGS=(-m LLM_experiments.egta_fj
      --base-model "$BASE_MODEL"
      --prompts-scope "$PROMPTS_SCOPE"
      --batch-size "$BATCH_SIZE"
      --max-length "$MAX_LENGTH"
      --snapshot-times $SNAPSHOTS
      --out "$OUT"
      --wandb --wandb-project egta-analysis --wandb-name "$WANDB_NAME")
for triple in "${TRAINED_RUNS[@]}"; do
    # shellcheck disable=SC2086
    set -- $triple
    ARGS+=(--run "$1" "$2" "$3")
done
for triple in "${STATIC_RUNS[@]}"; do
    # shellcheck disable=SC2086
    set -- $triple
    ARGS+=(--static-run "$1" "$2" "$3")
done

echo "[run_egta_fj_cf] launching: python ${ARGS[*]}"
python "${ARGS[@]}"

echo "[run_egta_fj_cf] done -> $OUT"
