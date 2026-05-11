#!/bin/bash
# TD3+BC pilot runs for C-track bring-up.
#
# Defaults are intentionally small: 1 seed, 2 envs, 100k gradient steps.
# Use this before launching full 1M-step / 3-seed experiments.

set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

[ -d ".venv" ] && source .venv/bin/activate
if [ -f ".env.local" ]; then
  set -a
  # shellcheck disable=SC1091
  source .env.local
  set +a
fi

export MUJOCO_GL=egl
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
if [ -f /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ]; then
  export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export D4RL_SUPPRESS_IMPORT_ERROR=1

STEPS="${STEPS:-100000}"
SEEDS="${SEEDS:-0}"
ENVS="${ENVS:-hopper-medium-v2 hopper-medium-replay-v2}"
EVAL_FREQ="${EVAL_FREQ:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
USE_AIM="${USE_AIM:-${AIM:-1}}"
USE_WANDB="${USE_WANDB:-${WANDB:-0}}"

for ENV in $ENVS; do
  for SEED in $SEEDS; do
    echo ""
    echo "=== TD3+BC | $ENV | seed=$SEED | steps=$STEPS ==="
    ARGS="--env $ENV --seed $SEED --steps $STEPS --eval_freq $EVAL_FREQ --eval_episodes $EVAL_EPISODES"
    [ "$USE_AIM" = "1" ] && ARGS="$ARGS --aim"
    [ "$USE_WANDB" = "1" ] && ARGS="$ARGS --wandb"
    python -m algorithms.td3_bc $ARGS
  done
done

echo ""
echo "Done. Results in results/td3_bc_*_seed*.jsonl"
