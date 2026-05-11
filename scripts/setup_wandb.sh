#!/bin/bash
# Local W&B setup helper.
#
# This script never writes secrets to git-tracked files. W&B login is user-level:
# the CLI stores credentials in the user's home directory, so all projects can
# reuse the same login.

set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

[ -d ".venv" ] && source .venv/bin/activate

GLOBAL_ENV="$HOME/.config/taoyao-rl/wandb.env"
if [ -f "$GLOBAL_ENV" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$GLOBAL_ENV"
  set +a
fi

if [ -f ".env.local" ]; then
  set -a
  # shellcheck disable=SC1091
  source .env.local
  set +a
fi

if ! command -v wandb >/dev/null 2>&1; then
  echo "wandb CLI not found."
  echo "Run setup first: bash setup_env.sh"
  echo "Or install into the current env: uv pip install wandb==0.16.0"
  exit 1
fi

export WANDB_PROJECT="${WANDB_PROJECT:-taoyao-rl}"

if [ -n "${WANDB_API_KEY:-}" ]; then
  echo "Logging in to W&B with WANDB_API_KEY from the current user environment"
  wandb login --relogin "$WANDB_API_KEY"
else
  echo "Starting user-level W&B login."
  echo "Create/log into W&B, then paste the key from: https://wandb.ai/authorize"
  wandb login --relogin
fi

echo ""
echo "W&B configured."
echo "Credential scope: user-level W&B login, reusable by all projects"
echo "Project: ${WANDB_PROJECT}"
if [ -n "${WANDB_ENTITY:-}" ]; then
  echo "Entity:  ${WANDB_ENTITY}"
else
  echo "Entity:  personal default workspace"
fi
echo ""
echo "Run with:"
echo "  USE_WANDB=1 bash scripts/run_td3_bc_pilot.sh"
