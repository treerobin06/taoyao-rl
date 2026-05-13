#!/usr/bin/env bash
# Run one SSAR official-source seed check on the retained AutoDL instance.
#
# Intended use:
#   SEED=1 EVAL_EPISODES=20 OFFLINE_STEPS=100000 bash scripts/run_ssar_seed_check_autodl.sh
#
# This script copies the external SSAR checkout into a timestamped run dir,
# runs td3_bc.py with SSAR's own code, parses the eval curve, and preserves
# the generated IQL-qv cache for future reuse.

set -u

export MUJOCO_GL=egl
export LD_LIBRARY_PATH="/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export PYTHONUNBUFFERED=1
export SSAR_IQL_STEPS="${SSAR_IQL_STEPS:-1000000}"

PY="${PY:-/root/autodl-tmp/taoyao-rl/project/.venv/bin/python}"
BASE="${BASE:-/root/autodl-tmp/external_repos/SSAR}"
ENV_NAME="${ENV_NAME:-hopper-medium-replay-v2}"
SEED="${SEED:-1}"
OFFLINE_STEPS="${OFFLINE_STEPS:-100000}"
EVAL_FREQ="${EVAL_FREQ:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-20}"
RUN_ROOT="${RUN_ROOT:-/root/autodl-tmp/external_quick_logs/ssar_seed_check_$(date +%Y%m%d_%H%M%S)}"
NAME="SSAR_${ENV_NAME}_seed${SEED}_${OFFLINE_STEPS}_eval${EVAL_EPISODES}"
WORK="$RUN_ROOT/$NAME/src"
CACHE_BACKUP="/root/autodl-tmp/taoyao-rl-cache/SSAR/iql_qv/$ENV_NAME/seed$SEED"

mkdir -p "$RUN_ROOT" "$(dirname "$WORK")" "$CACHE_BACKUP"

{
  echo "RUN_ROOT=$RUN_ROOT"
  echo "name=$NAME"
  echo "setting=$ENV_NAME seed=$SEED offline_steps=$OFFLINE_STEPS eval_freq=$EVAL_FREQ eval_episodes=$EVAL_EPISODES"
  echo "base=$BASE"
  echo "cache_backup=$CACHE_BACKUP"
  echo "started=$(date -Is)"
} | tee "$RUN_ROOT/summary.log"

cp -a "$BASE" "$WORK"

echo "=== CACHE BEFORE ===" | tee -a "$RUN_ROOT/summary.log"
find "$WORK/model/iql_qv/$ENV_NAME/$SEED" -maxdepth 1 -type f -name '*_model.pth' -print 2>/dev/null | sort | tee -a "$RUN_ROOT/summary.log" || true

start=$(date +%s)
echo "=== START $NAME $(date -Is) ===" | tee -a "$RUN_ROOT/summary.log"
(
  cd "$WORK" &&
  "$PY" td3_bc.py \
    --env "$ENV_NAME" \
    --seed "$SEED" \
    --offline_timesteps "$OFFLINE_STEPS" \
    --online_timesteps 0 \
    --eval_freq "$EVAL_FREQ" \
    --n_episodes "$EVAL_EPISODES" \
    --device cuda
) > "$RUN_ROOT/$NAME.log" 2>&1
rc=$?
end=$(date +%s)
echo "=== END $NAME rc=$rc elapsed=$((end - start))s $(date -Is) ===" | tee -a "$RUN_ROOT/summary.log"

find "$WORK/model/iql_qv/$ENV_NAME/$SEED" -maxdepth 1 -type f -name '*_model.pth' -print0 2>/dev/null \
  | xargs -0 -I{} cp -n "{}" "$CACHE_BACKUP/" 2>/dev/null || true

echo "=== CACHE AFTER ===" | tee -a "$RUN_ROOT/summary.log"
find "$WORK/model/iql_qv/$ENV_NAME/$SEED" -maxdepth 1 -type f -name '*_model.pth' -print 2>/dev/null | sort | tee -a "$RUN_ROOT/summary.log" || true
find "$CACHE_BACKUP" -maxdepth 1 -type f -name '*_model.pth' -print 2>/dev/null | sort | tee -a "$RUN_ROOT/summary.log" || true

"$PY" - "$RUN_ROOT" "$NAME" "$RUN_ROOT/$NAME.log" "$ENV_NAME" "$SEED" "$OFFLINE_STEPS" "$EVAL_FREQ" "$EVAL_EPISODES" "$CACHE_BACKUP" <<'PY' | tee -a "$RUN_ROOT/summary.log"
from pathlib import Path
import json
import re
import sys

root = Path(sys.argv[1])
name = sys.argv[2]
log = Path(sys.argv[3])
env_name = sys.argv[4]
seed = int(sys.argv[5])
offline_steps = int(sys.argv[6])
eval_freq = int(sys.argv[7])
eval_episodes = int(sys.argv[8])
cache_backup = Path(sys.argv[9])

text = log.read_text(errors="ignore") if log.exists() else ""
matches = re.findall(
    r"Time steps:\s*(\d+).*?Evaluation over\s*(\d+)\s*episodes:\s*"
    r"([+-]?[0-9]+(?:\.[0-9]+)?)\s*,\s*D4RL score:\s*"
    r"([+-]?[0-9]+(?:\.[0-9]+)?)",
    text,
    re.S,
)
scores = [
    {
        "step": int(step),
        "episodes": int(episodes),
        "raw_return": float(raw),
        "normalized_score": float(score),
    }
    for step, episodes, raw, score in matches
]
best = max(scores, key=lambda x: x["normalized_score"], default=None)
out = {
    "name": name,
    "env": env_name,
    "seed": seed,
    "offline_steps": offline_steps,
    "eval_freq": eval_freq,
    "eval_episodes": eval_episodes,
    "log": str(log),
    "cache_backup": str(cache_backup),
    "scores": scores,
    "final": scores[-1]["normalized_score"] if scores else None,
    "best": best["normalized_score"] if best else None,
    "best_step": best["step"] if best else None,
}
(root / f"{name}_parsed.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
PY

echo "=== ALL DONE $(date -Is) ===" | tee -a "$RUN_ROOT/summary.log"
exit "$rc"
