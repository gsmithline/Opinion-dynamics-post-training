#!/usr/bin/env bash
# Single-config runner for HTCondor.
# Args: RUN_TAG TRAINING_STYLE KL_BETA
set -eo pipefail

RUN_TAG="$1"
TRAINING_STYLE="$2"
KL_BETA="$3"

REPO="${REPO:-/home/gsmithline/Opinion-dynamics-post-training}"
CONDA_SH="${CONDA_SH:-/home/gsmithline/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-opdyn}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-/home/gsmithline/.wandb_key}"

RETRAIN_T="${RETRAIN_T:-30}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
SFT_EPOCHS="${SFT_EPOCHS:-1}"

echo "[run_one] host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "[run_one] tag=$RUN_TAG style=$TRAINING_STYLE beta=$KL_BETA T=$RETRAIN_T model=$BASE_MODEL"

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
    MODEL_NAME=llm \
    TRAINING_STYLE="$TRAINING_STYLE" \
    KL_BETA="$KL_BETA" \
    RETRAIN_T="$RETRAIN_T" \
    BASE_MODEL="$BASE_MODEL" \
    SFT_EPOCHS="$SFT_EPOCHS" \
    python pokec_simulations.py
