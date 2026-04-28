#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="/Users/jacobbecker/CPS485_RoboCode${PYTHONPATH:+:$PYTHONPATH}"
exec /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 -m NeuroEvoMelee.runtime.neuroevo_melee_bot --genome /Users/jacobbecker/CPS485_RoboCode/NeuroEvoMelee/data/current_genome.json --telemetry /Users/jacobbecker/CPS485_RoboCode/logs/_runtime_bots/20260427_164525/eval_all_bots_1v1/meleedqn_vs_neuroevomelee_eval/NeuroEvoMelee_raw.jsonl
