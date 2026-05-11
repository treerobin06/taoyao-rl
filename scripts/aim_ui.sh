#!/bin/bash
# Launch the local Aim dashboard for this project.

set -e
cd "$(dirname "${BASH_SOURCE[0]}")/.."

[ -d ".venv" ] && source .venv/bin/activate

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-43800}"
AIM_REPO="${AIM_REPO:-.}"

echo "Aim UI: http://${HOST}:${PORT}"
echo "Repo:   ${AIM_REPO}"
aim up --repo "$AIM_REPO" --host "$HOST" --port "$PORT"
