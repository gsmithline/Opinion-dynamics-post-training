#!/usr/bin/env bash
# Full sweep: baselines + LLM variants + KL-beta sweep.
# Each run is an independent wandb run, cached pickle tagged with RUN_TAG.
#
# Usage:
#   bash run_experiments.sh              # full sweep
#   bash run_experiments.sh baselines    # only fast baselines
#   bash run_experiments.sh llm          # only LLM variants
#
# Override common params via env:
#   RETRAIN_T=4 bash run_experiments.sh   # shorter smoke sweep

set -euo pipefail

RETRAIN_T="${RETRAIN_T:-4}"        # default: 4 rounds per run (smoke sweep ~5hrs total)
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
PY="uv run python pokec_simulations.py"

run() {
    local tag="$1"; shift
    echo ""
    echo "========================================================================"
    echo "[run_experiments] $tag"
    echo "  env: $*"
    echo "========================================================================"
    env RUN_TAG="$tag" RETRAIN_T="$RETRAIN_T" BASE_MODEL="$BASE_MODEL" "$@" $PY
}

run_baselines() {
    # Reference curves: all cheap (seconds to a minute each)
    run "perfect" MODEL_NAME=perfect
    run "mean"    MODEL_NAME=mean
    run "ridge"   MODEL_NAME=ridge
    run "mlp"     MODEL_NAME=neural_net
}

run_llm() {
    # Plain SFT (no anchor)
    run "llm_sft"        MODEL_NAME=llm TRAINING_STYLE=sft    KL_BETA=0.0

    # SFT + KL anchor sweep
    run "llm_sftkl_b0p3" MODEL_NAME=llm TRAINING_STYLE=sft_kl KL_BETA=0.3
    run "llm_sftkl_b1"   MODEL_NAME=llm TRAINING_STYLE=sft_kl KL_BETA=1.0
    run "llm_sftkl_b3"   MODEL_NAME=llm TRAINING_STYLE=sft_kl KL_BETA=3.0
    run "llm_sftkl_b10"  MODEL_NAME=llm TRAINING_STYLE=sft_kl KL_BETA=10.0
}

case "${1:-all}" in
    baselines) run_baselines ;;
    llm)       run_llm ;;
    all)       run_baselines; run_llm ;;
    *)         echo "usage: $0 [baselines|llm|all]"; exit 1 ;;
esac

echo ""
echo "[run_experiments] all done. Check wandb project 'opinion-dynamics-llm' for results."
