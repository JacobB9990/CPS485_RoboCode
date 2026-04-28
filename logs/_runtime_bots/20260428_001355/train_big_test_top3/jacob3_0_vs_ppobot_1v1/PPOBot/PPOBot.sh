#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="/Users/jacobbecker/CPS485_RoboCode${PYTHONPATH:+:$PYTHONPATH}"
exec /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m PPOBot.runtime.PPO_Bot --weights-path /Users/jacobbecker/CPS485_RoboCode/PPOBot/checkpoints/ppo_weights.pt --log-path /Users/jacobbecker/CPS485_RoboCode/logs/_runtime_bots/20260428_001355/train_big_test_top3/jacob3_0_vs_ppobot_1v1/PPOBot_raw.jsonl --eval
