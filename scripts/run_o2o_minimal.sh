#!/usr/bin/env bash
set -euo pipefail

# Minimal offline-to-online run for C-line experiments.
#
# Examples:
#   RUN=td3_bc_o2o bash scripts/run_o2o_minimal.sh
#   RUN=atlas_o2o LABEL_PATH=/path/to/atlas_selector.npz bash scripts/run_o2o_minimal.sh

cd "$(dirname "$0")/.."

[ -d ".venv" ] && source .venv/bin/activate

ENV_NAME="${ENV:-hopper-medium-replay-v2}"
SEED="${SEED:-0}"
OFFLINE_STEPS="${OFFLINE_STEPS:-50000}"
ONLINE_STEPS="${ONLINE_STEPS:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
EVAL_FREQ_OFFLINE="${EVAL_FREQ_OFFLINE:-10000}"
EVAL_FREQ_ONLINE="${EVAL_FREQ_ONLINE:-1000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
RESULT_DIR="${RESULT_DIR:-results/o2o_minimal}"
RUN="${RUN:-td3_bc_o2o}"
USE_AIM="${USE_AIM:-0}"
USE_WANDB="${USE_WANDB:-0}"

TRACKING_ARGS=()
[ "$USE_AIM" = "1" ] && TRACKING_ARGS+=(--aim)
[ "$USE_WANDB" = "1" ] && TRACKING_ARGS+=(--wandb)

COMMON_ARGS=(
  --env "$ENV_NAME"
  --seed "$SEED"
  --offline_steps "$OFFLINE_STEPS"
  --online_steps "$ONLINE_STEPS"
  --eval_freq_offline "$EVAL_FREQ_OFFLINE"
  --eval_freq_online "$EVAL_FREQ_ONLINE"
  --eval_episodes "$EVAL_EPISODES"
  --batch_size "$BATCH_SIZE"
  --result_dir "$RESULT_DIR"
  --online_batch_fraction "${ONLINE_BATCH_FRACTION:-0.5}"
  --exploration_noise "${EXPLORATION_NOISE:-0.1}"
  --offline_bc_coef "${OFFLINE_BC_COEF:-1.0}"
  --online_bc_coef_start "${ONLINE_BC_COEF_START:-1.0}"
  --online_bc_coef_end "${ONLINE_BC_COEF_END:-0.0}"
  --online_trust_gate "${ONLINE_TRUST_GATE:-none}"
  --online_gate_temperature "${ONLINE_GATE_TEMPERATURE:-10.0}"
  --online_gate_min_weight "${ONLINE_GATE_MIN_WEIGHT:-0.0}"
  --online_gate_margin "${ONLINE_GATE_MARGIN:-0.0}"
  --online_gate_start_step "${ONLINE_GATE_START_STEP:-1}"
)

case "$RUN" in
  td3_bc_o2o)
    python -m algorithms.td3_bc_o2o \
      "${COMMON_ARGS[@]}" \
      --algo_name "${ALGO_NAME:-td3_bc_o2o_decay}" \
      "${TRACKING_ARGS[@]}"
    ;;
  td3_bc_o2o_fixed)
    python -m algorithms.td3_bc_o2o \
      "${COMMON_ARGS[@]}" \
      --online_bc_coef_start "${ONLINE_BC_COEF_START:-1.0}" \
      --online_bc_coef_end "${ONLINE_BC_COEF_END:-1.0}" \
      --algo_name "${ALGO_NAME:-td3_bc_o2o_fixed}" \
      "${TRACKING_ARGS[@]}"
    ;;
  atlas_o2o)
    if [ -z "${LABEL_PATH:-}" ]; then
      echo "LABEL_PATH is required for RUN=atlas_o2o" >&2
      exit 2
    fi
    python -m algorithms.td3_bc_o2o \
      "${COMMON_ARGS[@]}" \
      --label_path "$LABEL_PATH" \
      --label_score_key "${LABEL_SCORE_KEY:-trust_score}" \
      --label_min_weight "${LABEL_MIN_WEIGHT:-0.05}" \
      --algo_name "${ALGO_NAME:-atlas_o2o_decay}" \
      "${TRACKING_ARGS[@]}"
    ;;
  atlas_o2o_fixed)
    if [ -z "${LABEL_PATH:-}" ]; then
      echo "LABEL_PATH is required for RUN=atlas_o2o_fixed" >&2
      exit 2
    fi
    python -m algorithms.td3_bc_o2o \
      "${COMMON_ARGS[@]}" \
      --online_bc_coef_start "${ONLINE_BC_COEF_START:-1.0}" \
      --online_bc_coef_end "${ONLINE_BC_COEF_END:-1.0}" \
      --label_path "$LABEL_PATH" \
      --label_score_key "${LABEL_SCORE_KEY:-trust_score}" \
      --label_min_weight "${LABEL_MIN_WEIGHT:-0.05}" \
      --algo_name "${ALGO_NAME:-atlas_o2o_fixed}" \
      "${TRACKING_ARGS[@]}"
    ;;
  *)
    echo "Unknown RUN=$RUN" >&2
    echo "Supported: td3_bc_o2o, td3_bc_o2o_fixed, atlas_o2o, atlas_o2o_fixed" >&2
    exit 2
    ;;
esac
