#!/usr/bin/env bash
# Single untuned-base-model readout runner for HTCondor.
# No training: loads BASE_MODEL, does one readout pass on unlabeled agents,
# runs the 30-round FJ simulation with fixed predictions.
# Args: RUN_TAG
set -eo pipefail

RUN_TAG="$1"

REPO="${REPO:-/home/gsmithline/Opinion-dynamics-post-training}"
CONDA_SH="${CONDA_SH:-/home/gsmithline/miniconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-opdyn}"
WANDB_KEY_FILE="${WANDB_KEY_FILE:-/home/gsmithline/.wandb_key}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
OUT_DIR="${OUT_DIR:-pokec_dataset/results}"

echo "[run_untuned] host=$(hostname) gpu=$(nvidia-smi -L 2>/dev/null | head -1 || echo none)"
echo "[run_untuned] tag=$RUN_TAG model=$BASE_MODEL out_dir=$OUT_DIR"

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
    BASE_MODEL="$BASE_MODEL" \
    OUT_DIR="$OUT_DIR" \
    python run_untuned_llm.py
