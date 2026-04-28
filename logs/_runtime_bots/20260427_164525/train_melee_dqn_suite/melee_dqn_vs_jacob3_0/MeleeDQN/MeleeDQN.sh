#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="/Users/jacobbecker/CPS485_RoboCode${PYTHONPATH:+:$PYTHONPATH}"
exec /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m MeleeDQN.runtime.melee_dqn_bot --weights-path /Users/jacobbecker/CPS485_RoboCode/MeleeDQN/checkpoints/melee_dqn_weights.pt --log-path /Users/jacobbecker/CPS485_RoboCode/logs/_runtime_bots/20260427_164525/train_melee_dqn_suite/melee_dqn_vs_jacob3_0/MeleeDQN_raw.jsonl --socket-port 5999
