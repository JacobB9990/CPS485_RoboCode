#!/usr/bin/env bash
# Example headless commands for Tank Royale server + bots.
# Requires robocode-tankroyale-server installed and available on PATH.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"

cat <<'EOF'
HEADLESS SERVER (TERMINAL 1):
robocode-tankroyale-server --port=7654 --games=classic

DQN BOT (TERMINAL 2):
cd /Users/jacobbecker/CPS485_RoboCode
python3 -m bots.python.dqn.runtime.run_bot --eval --eval-epsilon 0.0 --weights-path bots/python/dqn/checkpoints/dqn_weights_curriculum.pt --log-path bots/python/dqn/logs/eval/eval_walls_headless.jsonl

OPPONENT BOT (TERMINAL 3):
cd /Users/jacobbecker/CPS485_RoboCode/SampleBots/Walls
python3 Walls.py

---
Swap only the opponent command for other tests:

Note: current server CLI (v0.38.2) does not support --config.
Config JSON files now live under configs/robocode for setup reference only.

SpinBot opponent:
cd /Users/jacobbecker/CPS485_RoboCode/SampleBots/SpinBot && python3 SpinBot.py
Target opponent:
cd /Users/jacobbecker/CPS485_RoboCode/SampleBots/Target && python3 Target.py
EOF
