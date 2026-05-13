#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

[ -d ".venv" ] && source .venv/bin/activate

export PYTHONUNBUFFERED=1
export D4RL_SUPPRESS_IMPORT_ERROR=1

ENV_NAME="${ENV:-hopper-medium-replay-v2}"
SEED="${SEED:-0}"
OFFLINE_STEPS="${OFFLINE_STEPS:-50000}"
ONLINE_STEPS="${ONLINE_STEPS:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
EVAL_FREQ_OFFLINE="${EVAL_FREQ_OFFLINE:-10000}"
EVAL_FREQ_ONLINE="${EVAL_FREQ_ONLINE:-1000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
RESULT_DIR="${RESULT_DIR:-results/o2o_p0_eval20_20260514}"
LOG_DIR="${LOG_DIR:-logs/o2o_p0_eval20_20260514}"

ATLAS_LABEL="${ATLAS_LABEL:-results/atlas_labels/atlas_selector_hopper-medium-replay-v2_seed0.npz}"
IQL_LABEL="${IQL_LABEL:-results/atlas_labels/iql_qv_hopper-medium-replay-v2_seed0.npz}"
RANDOM_LABEL="${RANDOM_LABEL:-results/label_controls/random_subset_iqlqv_hopper-medium-replay-v2_seed0.npz}"

mkdir -p "$RESULT_DIR" "$LOG_DIR" "$(dirname "$RANDOM_LABEL")"

for required in "$ATLAS_LABEL" "$IQL_LABEL"; do
  if [ ! -f "$required" ]; then
    echo "missing required label file: $required" >&2
    exit 2
  fi
done

python3 scripts/make_label_control.py \
  --input "$IQL_LABEL" \
  --output "$RANDOM_LABEL" \
  --score_key hard_trust \
  --mode random_subset \
  --seed "$SEED" \
  --hard_threshold 0.5 | tee "$LOG_DIR/00_random_subset_label.log"

COMMON_ENV=(
  "ENV=$ENV_NAME"
  "SEED=$SEED"
  "OFFLINE_STEPS=$OFFLINE_STEPS"
  "ONLINE_STEPS=$ONLINE_STEPS"
  "EVAL_EPISODES=$EVAL_EPISODES"
  "EVAL_FREQ_OFFLINE=$EVAL_FREQ_OFFLINE"
  "EVAL_FREQ_ONLINE=$EVAL_FREQ_ONLINE"
  "BATCH_SIZE=$BATCH_SIZE"
  "RESULT_DIR=$RESULT_DIR"
  "USE_AIM=0"
  "USE_WANDB=0"
)

run_case() {
  local tag="$1"
  shift
  echo "===== START $tag $(date '+%F %T') ====="
  env "${COMMON_ENV[@]}" "$@" bash scripts/run_o2o_minimal.sh 2>&1 | tee "$LOG_DIR/${tag}.log"
  echo "===== DONE  $tag $(date '+%F %T') ====="
}

run_case "01_td3_bc_decay" \
  "RUN=td3_bc_o2o" \
  "ALGO_NAME=td3_bc_o2o_eval20_decay"

run_case "02_td3_bc_fixed" \
  "RUN=td3_bc_o2o_fixed" \
  "ALGO_NAME=td3_bc_o2o_eval20_fixed"

run_case "03_atlas_decay" \
  "RUN=atlas_o2o" \
  "ALGO_NAME=atlas_o2o_eval20_decay" \
  "LABEL_PATH=$ATLAS_LABEL" \
  "LABEL_SCORE_KEY=atlas_score" \
  "LABEL_MIN_WEIGHT=0.05"

run_case "04_atlas_fixed" \
  "RUN=atlas_o2o_fixed" \
  "ALGO_NAME=atlas_o2o_eval20_fixed" \
  "LABEL_PATH=$ATLAS_LABEL" \
  "LABEL_SCORE_KEY=atlas_score" \
  "LABEL_MIN_WEIGHT=0.05"

run_case "05_random_subset_decay" \
  "RUN=atlas_o2o" \
  "ALGO_NAME=random_subset_iqlqv_o2o_eval20_decay" \
  "LABEL_PATH=$RANDOM_LABEL" \
  "LABEL_SCORE_KEY=hard_trust" \
  "LABEL_MIN_WEIGHT=0.05"

run_case "06_ssar_iqlqv_decay" \
  "RUN=atlas_o2o" \
  "ALGO_NAME=ssar_iqlqv_o2o_eval20_decay" \
  "LABEL_PATH=$IQL_LABEL" \
  "LABEL_SCORE_KEY=hard_trust" \
  "LABEL_MIN_WEIGHT=0.05"

run_case "07_ssar_iqlqv_fixed" \
  "RUN=atlas_o2o_fixed" \
  "ALGO_NAME=ssar_iqlqv_o2o_eval20_fixed" \
  "LABEL_PATH=$IQL_LABEL" \
  "LABEL_SCORE_KEY=hard_trust" \
  "LABEL_MIN_WEIGHT=0.05"

python3 tools/summarize_o2o_results.py "$RESULT_DIR" 2>/dev/null || true
echo "===== ALL DONE $(date '+%F %T') ====="
