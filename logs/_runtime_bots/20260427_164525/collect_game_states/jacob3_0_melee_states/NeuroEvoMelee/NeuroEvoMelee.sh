#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="/Users/jacobbecker/CPS485_RoboCode${PYTHONPATH:+:$PYTHONPATH}"
exec /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m NeuroEvoMelee.runtime.neuroevo_melee_bot --genome /Users/jacobbecker/CPS485_RoboCode/NeuroEvoMelee/data/current_genome.json --telemetry /Users/jacobbecker/CPS485_RoboCode/logs/_runtime_bots/20260427_164525/collect_game_states/jacob3_0_melee_states/NeuroEvoMelee_raw.jsonl
