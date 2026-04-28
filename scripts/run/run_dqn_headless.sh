#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

MODE="${DQN_MODE:-train}"
WEIGHTS_PATH="${DQN_WEIGHTS_PATH:-$ROOT_DIR/bots/python/dqn/checkpoints/dqn_weights_curriculum.pt}"
LOG_PATH="${DQN_LOG_PATH:-$ROOT_DIR/bots/python/dqn/logs/dqn_headless.jsonl}"
EVAL_EPS="${DQN_EVAL_EPSILON:-0.0}"

cd "$ROOT_DIR"

if [ "$MODE" = "eval" ]; then
  python3 -m bots.python.dqn.runtime.run_bot --eval --eval-epsilon "$EVAL_EPS" --weights-path "$WEIGHTS_PATH" --log-path "$LOG_PATH"
else
  python3 -m bots.python.dqn.runtime.run_bot --weights-path "$WEIGHTS_PATH" --log-path "$LOG_PATH"
fi
