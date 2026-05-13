#!/usr/bin/env bash
# Closed-form FJ runner for HTCondor.
# Args: RUN_TAG KL_BETA
set -eo pipefail

RUN_TAG="$1"
KL_BETA="$2"

REPO="${REPO:-/home/gsmithline/Opinion-dynamics-post-training}"
CONDA_SH="${CONDA_SH:-/home/gsmithline/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-opdyn}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-/home/gsmithline/.wandb_key}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
RETRAIN_T="${RETRAIN_T:-30}"
CF_N_BINS="${CF_N_BINS:-11}"
OUT_DIR="${OUT_DIR:-pokec_dataset/results}"

echo "[run_cf_fj] host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "[run_cf_fj] tag=$RUN_TAG beta=$KL_BETA K=$CF_N_BINS T=$RETRAIN_T model=$BASE_MODEL"

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"

if [ -f "$WANDB_KEY_FILE" ]; then
    export WANDB_API_KEY="$(tr -d '[:space:]' < "$WANDB_KEY_FILE")"
fi
export WANDB_DIR="${WANDB_DIR:-$REPO/wandb}"
export HF_HOME="${HF_HOME:-/home/gsmithline/.cache/huggingface}"

cd "$REPO"

env \
    RUN_TAG="$RUN_TAG" \
    KL_BETA="$KL_BETA" \
    BASE_MODEL="$BASE_MODEL" \
    RETRAIN_T="$RETRAIN_T" \
    CF_N_BINS="$CF_N_BINS" \
    OUT_DIR="$OUT_DIR" \
    python run_cf_fj.py
