#!/bin/bash
# BC baseline 跑 3 seeds × hopper-medium-v2
# 全组任何人 setup 完都可以直接跑此脚本验证

set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

[ -d ".venv" ] && source .venv/bin/activate

export MUJOCO_GL=egl
export LD_LIBRARY_PATH="$HOME/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
if [ -f /usr/lib/x86_64-linux-gnu/libstdc++.so.6 ]; then
  export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
fi
export D4RL_SUPPRESS_IMPORT_ERROR=1

ENV="${ENV:-hopper-medium-v2}"
STEPS="${STEPS:-100000}"
USE_WANDB="${WANDB:-0}"  # WANDB=1 启用

for SEED in 0 1 2; do
  echo ""
  echo "=== BC | $ENV | seed=$SEED | steps=$STEPS ==="
  ARGS="--env $ENV --seed $SEED --steps $STEPS"
  [ "$USE_WANDB" = "1" ] && ARGS="$ARGS --wandb"
  python -m algorithms.bc $ARGS
done

echo ""
echo "Done. Results in results/bc_${ENV}_seed*.jsonl"
