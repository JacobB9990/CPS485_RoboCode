#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="/Users/jacobbecker/CPS485_RoboCode${PYTHONPATH:+:$PYTHONPATH}"
exec /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m SarsaBot.runtime.sarsa_bot --q-table-path /Users/jacobbecker/CPS485_RoboCode/SarsaBot/data/q_table_sarsa.json --log-path /Users/jacobbecker/CPS485_RoboCode/logs/_runtime_bots/20260427_164525/train_melee_dqn_suite/melee_dqn_4bot/SarsaBot_raw.jsonl --eval --eval-epsilon 0.0
