#!/usr/bin/env bash
# Curriculum training + fair evaluation commands for DQNBot
# Usage:
#   1) In terminal A, run one of the TRAIN/EVAL commands below from this file.
#   2) In terminal B, run exactly one opponent command (from OPPONENT COMMANDS).
#   3) Set rounds in the Tank Royale GUI and run the battle.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WEIGHTS="$ROOT/data/checkpoints/dqn/dqn_weights_curriculum.pt"
LOGS_DIR="$ROOT/logs/eval"

mkdir -p "$LOGS_DIR"

cd "$ROOT"

cat <<'EOF'
============================================================
DQN BOT COMMANDS (TERMINAL A)
============================================================

TRAINING (online learning enabled)
python3 -m bots.python.dqn.runtime.run_bot --weights-path data/checkpoints/dqn/dqn_weights_curriculum.pt --log-path logs/dqn/train_walls.jsonl
python3 -m bots.python.dqn.runtime.run_bot --weights-path data/checkpoints/dqn/dqn_weights_curriculum.pt --log-path logs/dqn/train_spinbot.jsonl
python3 -m bots.python.dqn.runtime.run_bot --weights-path data/checkpoints/dqn/dqn_weights_curriculum.pt --log-path logs/dqn/train_ramfire.jsonl
python3 -m bots.python.dqn.runtime.run_bot --weights-path data/checkpoints/dqn/dqn_weights_curriculum.pt --log-path logs/dqn/train_trackfire.jsonl
python3 -m bots.python.dqn.runtime.run_bot --weights-path data/checkpoints/dqn/dqn_weights_curriculum.pt --log-path logs/dqn/train_velocitybot.jsonl
python3 -m bots.python.dqn.runtime.run_bot --weights-path data/checkpoints/dqn/dqn_weights_curriculum.pt --log-path logs/dqn/train_crazy.jsonl
python3 -m bots.python.dqn.runtime.run_bot --weights-path data/checkpoints/dqn/dqn_weights_curriculum.pt --log-path logs/dqn/train_corners.jsonl

EVALUATION (no online learning, fixed epsilon)
python3 -m bots.python.dqn.runtime.run_bot --eval --eval-epsilon 0.0 --weights-path data/checkpoints/dqn/dqn_weights_curriculum.pt --log-path logs/eval/eval_walls.jsonl
python3 -m bots.python.dqn.runtime.run_bot --eval --eval-epsilon 0.0 --weights-path data/checkpoints/dqn/dqn_weights_curriculum.pt --log-path logs/eval/eval_spinbot.jsonl
python3 -m bots.python.dqn.runtime.run_bot --eval --eval-epsilon 0.0 --weights-path data/checkpoints/dqn/dqn_weights_curriculum.pt --log-path logs/eval/eval_ramfire.jsonl
python3 -m bots.python.dqn.runtime.run_bot --eval --eval-epsilon 0.0 --weights-path data/checkpoints/dqn/dqn_weights_curriculum.pt --log-path logs/eval/eval_trackfire.jsonl
python3 -m bots.python.dqn.runtime.run_bot --eval --eval-epsilon 0.0 --weights-path data/checkpoints/dqn/dqn_weights_curriculum.pt --log-path logs/eval/eval_velocitybot.jsonl
python3 -m bots.python.dqn.runtime.run_bot --eval --eval-epsilon 0.0 --weights-path data/checkpoints/dqn/dqn_weights_curriculum.pt --log-path logs/eval/eval_crazy.jsonl
python3 -m bots.python.dqn.runtime.run_bot --eval --eval-epsilon 0.0 --weights-path data/checkpoints/dqn/dqn_weights_curriculum.pt --log-path logs/eval/eval_corners.jsonl
python3 -m bots.python.dqn.runtime.run_bot --eval --eval-epsilon 0.0 --weights-path data/checkpoints/dqn/dqn_weights_curriculum.pt --log-path logs/eval/eval_target.jsonl
python3 -m bots.python.dqn.runtime.run_bot --eval --eval-epsilon 0.0 --weights-path data/checkpoints/dqn/dqn_weights_curriculum.pt --log-path logs/eval/eval_fire.jsonl
python3 -m bots.python.dqn.runtime.run_bot --eval --eval-epsilon 0.0 --weights-path data/checkpoints/dqn/dqn_weights_curriculum.pt --log-path logs/eval/eval_myfirstbot.jsonl

============================================================
OPPONENT COMMANDS (TERMINAL B, run one at a time)
============================================================

cd /Users/jacobbecker/CPS485_RoboCode/SampleBots/Walls && python3 Walls.py
cd /Users/jacobbecker/CPS485_RoboCode/SampleBots/SpinBot && python3 SpinBot.py
cd /Users/jacobbecker/CPS485_RoboCode/SampleBots/RamFire && python3 RamFire.py
cd /Users/jacobbecker/CPS485_RoboCode/SampleBots/TrackFire && python3 TrackFire.py
cd /Users/jacobbecker/CPS485_RoboCode/SampleBots/VelocityBot && python3 VelocityBot.py
cd /Users/jacobbecker/CPS485_RoboCode/SampleBots/Crazy && python3 Crazy.py
cd /Users/jacobbecker/CPS485_RoboCode/SampleBots/Corners && python3 Corners.py
cd /Users/jacobbecker/CPS485_RoboCode/SampleBots/Target && python3 Target.py
cd /Users/jacobbecker/CPS485_RoboCode/SampleBots/Fire && python3 Fire.py
cd /Users/jacobbecker/CPS485_RoboCode/SampleBots/MyFirstBot && python3 MyFirstBot.py

============================================================
RECOMMENDED SCHEDULE
============================================================

- Train 150 to 300 rounds per training opponent.
- Eval 100 rounds per eval opponent.
- Do 2 to 3 curriculum passes.
- Compare progress using eval logs only.

EOF
