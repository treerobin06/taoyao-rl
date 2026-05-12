#!/bin/bash
# Exploration-stage C-track screen:
# one env, one seed, four lightweight models/variants.

set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

[ -d ".venv" ] && source .venv/bin/activate
if [ -f ".env.local" ]; then
  while IFS='=' read -r key value; do
    case "$key" in ""|\#*) continue ;; esac
    case "$key" in *[!A-Za-z0-9_]*|[0-9]*) continue ;; esac
    [ -z "${!key+x}" ] && export "$key=$value"
  done < .env.local
fi

export MUJOCO_GL=egl
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
if [ -f /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ]; then
  export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export D4RL_SUPPRESS_IMPORT_ERROR=1

ENV="${ENV:-hopper-medium-replay-v2}"
SEED="${SEED:-0}"
STEPS="${STEPS:-50000}"
EVAL_FREQ="${EVAL_FREQ:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
RESULT_DIR="${RESULT_DIR:-results/c_track_smoke}"
USE_AIM="${USE_AIM:-${AIM:-1}}"
USE_WANDB="${USE_WANDB:-${WANDB:-0}}"
RUNS="${RUNS:-bc td3_bc td3_bc_alpha5 rebrac_lite}"

COMMON_ARGS="--env $ENV --seed $SEED --steps $STEPS --eval_freq $EVAL_FREQ --eval_episodes $EVAL_EPISODES --result_dir $RESULT_DIR"
TRACKING_ARGS=""
[ "$USE_AIM" = "1" ] && TRACKING_ARGS="$TRACKING_ARGS --aim"
[ "$USE_WANDB" = "1" ] && TRACKING_ARGS="$TRACKING_ARGS --wandb"

echo "C-track smoke"
echo "  env=$ENV seed=$SEED steps=$STEPS eval_freq=$EVAL_FREQ eval_episodes=$EVAL_EPISODES"
echo "  runs=$RUNS"
echo "  result_dir=$RESULT_DIR"

for RUN in $RUNS; do
  echo ""
  echo "=== $RUN | $ENV | seed=$SEED | steps=$STEPS ==="
  case "$RUN" in
    bc)
      python -m algorithms.bc $COMMON_ARGS --batch_size 256 --algo_name bc $TRACKING_ARGS
      ;;
    td3_bc)
      python -m algorithms.td3_bc $COMMON_ARGS --batch_size 256 --algo_name td3_bc $TRACKING_ARGS
      ;;
    td3_bc_alpha5)
      python -m algorithms.td3_bc $COMMON_ARGS --batch_size 256 --alpha 5.0 --algo_name td3_bc_alpha5 $TRACKING_ARGS
      ;;
    rebrac_lite)
      python -m algorithms.rebrac $COMMON_ARGS --batch_size 1024 --num_critics 2 --algo_name rebrac_lite $TRACKING_ARGS
      ;;
    *)
      echo "Unknown RUN=$RUN" >&2
      exit 2
      ;;
  esac
done

echo ""
echo "Done. Results in $RESULT_DIR"
