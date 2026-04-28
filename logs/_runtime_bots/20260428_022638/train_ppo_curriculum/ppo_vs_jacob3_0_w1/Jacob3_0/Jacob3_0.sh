#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="/Users/adithya/CSProjects/CPS485_RoboCode${PYTHONPATH:+:$PYTHONPATH}"
export BOT_NAME="Jacob3_0"
export BOT_VERSION="1.0.0"
exec /Users/adithya/CSProjects/CPS485_RoboCode/.venv/bin/python3 -m Jacob3_0.runtime.dqn_bot --weights-path /Users/adithya/CSProjects/CPS485_RoboCode/Jacob3_0/checkpoints/dqn_weights.pt --log-path /Users/adithya/CSProjects/CPS485_RoboCode/logs/_runtime_bots/20260428_022638/train_ppo_curriculum/ppo_vs_jacob3_0_w1/Jacob3_0_raw.jsonl --eval --eval-epsilon 0.0
