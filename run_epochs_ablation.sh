#!/usr/bin/env bash
# Quick ablation: same 4-round LLM sweep but with SFT_EPOCHS=2 per round.
# Saves under different RUN_TAG so it doesn't clobber the 1-epoch results.
set -euo pipefail

RETRAIN_T="${RETRAIN_T:-4}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PY="uv run python pokec_simulations.py"

run() {
    local tag="$1"; shift
    echo ""
    echo "========================================================================"
    echo "[run_epochs_ablation] $tag  (SFT_EPOCHS=2)"
    echo "========================================================================"
    env RUN_TAG="$tag" RETRAIN_T="$RETRAIN_T" BASE_MODEL="$BASE_MODEL" SFT_EPOCHS=2 "$@" $PY
}

run "llm_sft_e2"        MODEL_NAME=llm TRAINING_STYLE=sft    KL_BETA=0.0
run "llm_sftkl_b0p3_e2" MODEL_NAME=llm TRAINING_STYLE=sft_kl KL_BETA=0.3
run "llm_sftkl_b1_e2"   MODEL_NAME=llm TRAINING_STYLE=sft_kl KL_BETA=1.0
run "llm_sftkl_b3_e2"   MODEL_NAME=llm TRAINING_STYLE=sft_kl KL_BETA=3.0
run "llm_sftkl_b10_e2"  MODEL_NAME=llm TRAINING_STYLE=sft_kl KL_BETA=10.0

echo ""
echo "[run_epochs_ablation] done. Pickles: pokec_dataset/results/llm_*_e2_equilibrium.pk"
echo "Compare to 1-epoch runs in wandb via the _e2 suffix in run names."
