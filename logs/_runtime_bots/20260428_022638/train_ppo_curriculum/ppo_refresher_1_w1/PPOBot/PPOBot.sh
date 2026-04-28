#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="/Users/adithya/CSProjects/CPS485_RoboCode${PYTHONPATH:+:$PYTHONPATH}"
exec /Users/adithya/CSProjects/CPS485_RoboCode/.venv/bin/python3 -m PPOBot.runtime.PPO_Bot --weights-path /Users/adithya/CSProjects/CPS485_RoboCode/PPOBot/checkpoints/ppo_weights.pt --log-path /Users/adithya/CSProjects/CPS485_RoboCode/logs/_runtime_bots/20260428_022638/train_ppo_curriculum/ppo_refresher_1_w1/PPOBot_raw.jsonl
