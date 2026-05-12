#!/usr/bin/env bash
# Mechanism ablation runner for the retained AutoDL instance.
#
# Runs one env / one seed:
# - SSAR with cached IQL-qv, 100k offline steps
# - cheap SSAR without IQL action selection, 100k offline steps
# - ReBRAC-lite, 100k steps

set -u

export MUJOCO_GL=egl
export LD_LIBRARY_PATH="/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libstdc++.so.6${LD_PRELOAD:+:$LD_PRELOAD}"
export D4RL_SUPPRESS_IMPORT_ERROR=1
export PYTHONUNBUFFERED=1
export SSAR_IQL_STEPS=1000000

PY=/root/autodl-tmp/taoyao-rl/project/.venv/bin/python
BASE=/root/autodl-tmp/external_repos/SSAR
PROJECT=/root/autodl-tmp/taoyao-rl/project
RUN_ROOT="${1:-/root/autodl-tmp/external_quick_logs/mech_ablation_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$RUN_ROOT"

echo "RUN_ROOT=$RUN_ROOT" | tee "$RUN_ROOT/summary.log"
echo "setting=hopper-medium-replay-v2 seed=0 eval_freq=10000 eval_episodes=5" | tee -a "$RUN_ROOT/summary.log"
echo "runs=SSAR_cached_100k cheap_SSAR_no_iql_select_100k ReBRAC_lite_100k" | tee -a "$RUN_ROOT/summary.log"

parse_ssar() {
  local name="$1"
  local log="$2"
  "$PY" - "$RUN_ROOT" "$name" "$log" <<'PY'
from pathlib import Path
import json
import re
import sys

root = Path(sys.argv[1])
name = sys.argv[2]
log = Path(sys.argv[3])
text = log.read_text(errors="ignore") if log.exists() else ""
pairs = [
    (int(step), float(score))
    for step, score in re.findall(
        r"Time steps:\s*(\d+).*?D4RL score:\s*([+-]?[0-9]+(?:\.[0-9]+)?)",
        text,
        re.S,
    )
]
out = {
    "name": name,
    "log": str(log),
    "scores": pairs,
    "final": pairs[-1][1] if pairs else None,
    "best": max((v for _, v in pairs), default=None),
}
(root / f"{name}_parsed.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
PY
}

parse_rebrac() {
  local name="$1"
  local log="$2"
  local jsonl="$3"
  "$PY" - "$RUN_ROOT" "$name" "$log" "$jsonl" <<'PY'
from pathlib import Path
import json
import re
import sys

root = Path(sys.argv[1])
name = sys.argv[2]
log = Path(sys.argv[3])
jsonl = Path(sys.argv[4])
pairs = []
if jsonl.exists():
    for line in jsonl.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if "step" in rec and "normalized_score" in rec:
            pairs.append((int(rec["step"]), float(rec["normalized_score"])))
if not pairs:
    text = log.read_text(errors="ignore") if log.exists() else ""
    pairs = [
        (int(step.replace(",", "")), float(score))
        for step, score in re.findall(r"step=\s*([0-9,]+)\s*\|\s*norm=\s*([+-]?[0-9]+(?:\.[0-9]+)?)", text)
    ]
out = {
    "name": name,
    "log": str(log),
    "jsonl": str(jsonl),
    "scores": pairs,
    "final": pairs[-1][1] if pairs else None,
    "best": max((v for _, v in pairs), default=None),
}
(root / f"{name}_parsed.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
print(json.dumps(out, indent=2))
PY
}

run_ssar_cached() {
  local name=SSAR_cached_100k
  local work="$RUN_ROOT/$name/src"
  mkdir -p "$(dirname "$work")"
  cp -a "$BASE" "$work"

  echo "=== START $name $(date -Is) ===" | tee -a "$RUN_ROOT/summary.log"
  local start
  start=$(date +%s)
  (
    cd "$work" &&
    "$PY" td3_bc.py \
      --env hopper-medium-replay-v2 \
      --seed 0 \
      --offline_timesteps 100000 \
      --online_timesteps 0 \
      --eval_freq 10000 \
      --n_episodes 5 \
      --device cuda
  ) > "$RUN_ROOT/$name.log" 2>&1
  local rc=$?
  local end
  end=$(date +%s)
  echo "=== END $name rc=$rc elapsed=$((end - start))s $(date -Is) ===" | tee -a "$RUN_ROOT/summary.log"
  parse_ssar "$name" "$RUN_ROOT/$name.log" | tee -a "$RUN_ROOT/summary.log"
}

run_cheap_ssar() {
  local name=cheap_SSAR_no_iql_select_100k
  local work="$RUN_ROOT/$name/src"
  mkdir -p "$(dirname "$work")"
  cp -a "$BASE" "$work"

  "$PY" - "$work/config/td3_bc/hopper/medium_replay_v2.yaml" <<'PY'
from pathlib import Path
import sys

p = Path(sys.argv[1])
text = p.read_text()
text = text.replace("select_actions: true", "select_actions: false")
text = text.replace("select_method: iql", "select_method: none")
p.write_text(text)
PY

  echo "=== START $name $(date -Is) ===" | tee -a "$RUN_ROOT/summary.log"
  local start
  start=$(date +%s)
  (
    cd "$work" &&
    "$PY" td3_bc.py \
      --env hopper-medium-replay-v2 \
      --seed 0 \
      --offline_timesteps 100000 \
      --online_timesteps 0 \
      --eval_freq 10000 \
      --n_episodes 5 \
      --device cuda
  ) > "$RUN_ROOT/$name.log" 2>&1
  local rc=$?
  local end
  end=$(date +%s)
  echo "=== END $name rc=$rc elapsed=$((end - start))s $(date -Is) ===" | tee -a "$RUN_ROOT/summary.log"
  parse_ssar "$name" "$RUN_ROOT/$name.log" | tee -a "$RUN_ROOT/summary.log"
}

run_rebrac() {
  local name=ReBRAC_lite_100k
  local result_dir=results/c_track_mech_ablation
  local jsonl="$PROJECT/$result_dir/rebrac_lite_hopper-medium-replay-v2_seed0.jsonl"
  rm -f "$jsonl"

  echo "=== START $name $(date -Is) ===" | tee -a "$RUN_ROOT/summary.log"
  local start
  start=$(date +%s)
  (
    cd "$PROJECT" &&
    ENV=hopper-medium-replay-v2 \
    SEED=0 \
    STEPS=100000 \
    EVAL_FREQ=10000 \
    EVAL_EPISODES=5 \
    RESULT_DIR="$result_dir" \
    RUNS=rebrac_lite \
    USE_AIM=1 \
    USE_WANDB=0 \
    bash scripts/run_c_track_smoke.sh
  ) > "$RUN_ROOT/$name.log" 2>&1
  local rc=$?
  local end
  end=$(date +%s)
  echo "=== END $name rc=$rc elapsed=$((end - start))s $(date -Is) ===" | tee -a "$RUN_ROOT/summary.log"
  parse_rebrac "$name" "$RUN_ROOT/$name.log" "$jsonl" | tee -a "$RUN_ROOT/summary.log"
}

run_ssar_cached
run_cheap_ssar
run_rebrac

echo "=== ALL DONE $(date -Is) ===" | tee -a "$RUN_ROOT/summary.log"
