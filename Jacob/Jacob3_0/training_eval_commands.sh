#!/usr/bin/env bash
# Curriculum training + fair evaluation commands for DQNBot
# Usage:
#   1) In terminal A, run one of the TRAIN/EVAL commands below from this file.
#   2) In terminal B, run exactly one opponent command (from OPPONENT COMMANDS).
#   3) Set rounds in the Tank Royale GUI and run the battle.

set -euo pipefail

ROOT="/Users/jacobbecker/CPS485_RoboCode"
DQN_DIR="$ROOT/Jacob/Jacob3_0"
BOTS_DIR="$ROOT/SampleBots"
WEIGHTS="$DQN_DIR/dqn_weights_curriculum.pt"
LOGS_DIR="$DQN_DIR/logs"

mkdir -p "$LOGS_DIR"

cd "$DQN_DIR"

cat <<'EOF'
============================================================
DQN BOT COMMANDS (TERMINAL A)
============================================================

TRAINING (online learning enabled)
python3 run_bot.py --weights-path dqn_weights_curriculum.pt --log-path logs/train_walls.jsonl
python3 run_bot.py --weights-path dqn_weights_curriculum.pt --log-path logs/train_spinbot.jsonl
python3 run_bot.py --weights-path dqn_weights_curriculum.pt --log-path logs/train_ramfire.jsonl
python3 run_bot.py --weights-path dqn_weights_curriculum.pt --log-path logs/train_trackfire.jsonl
python3 run_bot.py --weights-path dqn_weights_curriculum.pt --log-path logs/train_velocitybot.jsonl
python3 run_bot.py --weights-path dqn_weights_curriculum.pt --log-path logs/train_crazy.jsonl
python3 run_bot.py --weights-path dqn_weights_curriculum.pt --log-path logs/train_corners.jsonl

EVALUATION (no online learning, fixed epsilon)
python3 run_bot.py --eval --eval-epsilon 0.0 --weights-path dqn_weights_curriculum.pt --log-path logs/eval_walls.jsonl
python3 run_bot.py --eval --eval-epsilon 0.0 --weights-path dqn_weights_curriculum.pt --log-path logs/eval_spinbot.jsonl
python3 run_bot.py --eval --eval-epsilon 0.0 --weights-path dqn_weights_curriculum.pt --log-path logs/eval_ramfire.jsonl
python3 run_bot.py --eval --eval-epsilon 0.0 --weights-path dqn_weights_curriculum.pt --log-path logs/eval_trackfire.jsonl
python3 run_bot.py --eval --eval-epsilon 0.0 --weights-path dqn_weights_curriculum.pt --log-path logs/eval_velocitybot.jsonl
python3 run_bot.py --eval --eval-epsilon 0.0 --weights-path dqn_weights_curriculum.pt --log-path logs/eval_crazy.jsonl
python3 run_bot.py --eval --eval-epsilon 0.0 --weights-path dqn_weights_curriculum.pt --log-path logs/eval_corners.jsonl
python3 run_bot.py --eval --eval-epsilon 0.0 --weights-path dqn_weights_curriculum.pt --log-path logs/eval_target.jsonl
python3 run_bot.py --eval --eval-epsilon 0.0 --weights-path dqn_weights_curriculum.pt --log-path logs/eval_fire.jsonl
python3 run_bot.py --eval --eval-epsilon 0.0 --weights-path dqn_weights_curriculum.pt --log-path logs/eval_myfirstbot.jsonl

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
