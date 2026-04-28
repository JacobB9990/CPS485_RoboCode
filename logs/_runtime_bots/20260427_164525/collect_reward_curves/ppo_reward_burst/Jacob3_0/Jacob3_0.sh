#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="/Users/jacobbecker/CPS485_RoboCode${PYTHONPATH:+:$PYTHONPATH}"
export BOT_NAME="Jacob3_0"
export BOT_VERSION="1.0.0"
exec /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m Jacob3_0.runtime.dqn_bot --weights-path /Users/jacobbecker/CPS485_RoboCode/Jacob3_0/checkpoints/dqn_weights.pt --log-path /Users/jacobbecker/CPS485_RoboCode/logs/_runtime_bots/20260427_164525/collect_reward_curves/ppo_reward_burst/Jacob3_0_raw.jsonl --eval --eval-epsilon 0.0
