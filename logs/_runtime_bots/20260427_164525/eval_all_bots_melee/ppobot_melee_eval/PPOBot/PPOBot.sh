#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="/Users/jacobbecker/CPS485_RoboCode${PYTHONPATH:+:$PYTHONPATH}"
exec /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m PPOBot.runtime.PPO_Bot --weights-path /Users/jacobbecker/CPS485_RoboCode/PPOBot/checkpoints/ppo_weights.pt --log-path /Users/jacobbecker/CPS485_RoboCode/logs/_runtime_bots/20260427_164525/eval_all_bots_melee/ppobot_melee_eval/PPOBot_raw.jsonl --eval
