#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

[ -d ".venv" ] && source .venv/bin/activate

export PYTHONUNBUFFERED=1
export D4RL_SUPPRESS_IMPORT_ERROR=1

ENV_NAME="${ENV:-hopper-medium-replay-v2}"
SEED="${SEED:-0}"
RESULT_DIR="${RESULT_DIR:-results/o2o_p1_qgate_eval20_20260514}"
LOG_DIR="${LOG_DIR:-logs/o2o_p1_qgate_eval20_20260514}"
ATLAS_LABEL="${ATLAS_LABEL:-results/atlas_labels/atlas_selector_hopper-medium-replay-v2_seed0.npz}"
IQL_LABEL="${IQL_LABEL:-results/atlas_labels/iql_qv_hopper-medium-replay-v2_seed0.npz}"

mkdir -p "$RESULT_DIR" "$LOG_DIR"

COMMON_ENV=(
  "ENV=$ENV_NAME"
  "SEED=$SEED"
  "OFFLINE_STEPS=${OFFLINE_STEPS:-50000}"
  "ONLINE_STEPS=${ONLINE_STEPS:-10000}"
  "EVAL_EPISODES=${EVAL_EPISODES:-20}"
  "EVAL_FREQ_OFFLINE=${EVAL_FREQ_OFFLINE:-10000}"
  "EVAL_FREQ_ONLINE=${EVAL_FREQ_ONLINE:-1000}"
  "BATCH_SIZE=${BATCH_SIZE:-256}"
  "RESULT_DIR=$RESULT_DIR"
  "USE_AIM=0"
  "USE_WANDB=0"
  "ONLINE_TRUST_GATE=qgap"
  "ONLINE_GATE_TEMPERATURE=${ONLINE_GATE_TEMPERATURE:-10.0}"
  "ONLINE_GATE_MIN_WEIGHT=${ONLINE_GATE_MIN_WEIGHT:-0.05}"
  "ONLINE_GATE_MARGIN=${ONLINE_GATE_MARGIN:-0.0}"
  "ONLINE_GATE_START_STEP=${ONLINE_GATE_START_STEP:-1}"
)

run_case() {
  local tag="$1"
  shift
  echo "===== START $tag $(date '+%F %T') ====="
  env "${COMMON_ENV[@]}" "$@" bash scripts/run_o2o_minimal.sh 2>&1 | tee "$LOG_DIR/${tag}.log"
  echo "===== DONE  $tag $(date '+%F %T') ====="
}

run_case "01_atlas_qgate_fixed" \
  "RUN=atlas_o2o_fixed" \
  "ALGO_NAME=atlas_o2o_eval20_qgate_fixed" \
  "LABEL_PATH=$ATLAS_LABEL" \
  "LABEL_SCORE_KEY=atlas_score" \
  "LABEL_MIN_WEIGHT=0.05"

run_case "02_ssar_iqlqv_qgate_fixed" \
  "RUN=atlas_o2o_fixed" \
  "ALGO_NAME=ssar_iqlqv_o2o_eval20_qgate_fixed" \
  "LABEL_PATH=$IQL_LABEL" \
  "LABEL_SCORE_KEY=hard_trust" \
  "LABEL_MIN_WEIGHT=0.05"

echo "===== ALL DONE $(date '+%F %T') ====="
