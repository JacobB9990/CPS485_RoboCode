#!/usr/bin/env bash
# One-command overnight trainer for the packaged DQN bot.
#
# What it does:
# - Runs round-based curriculum battles against multiple opponents.
# - Uses Battle Runner to start actual games (not just bot connections).
# - Writes per-slice logs for debugging and progress tracking.
#
# Usage:
#   ./overnight_train.sh
#
# Optional env vars:
#   PASSES=3                 Number of curriculum passes (default: 2)
#   TRAIN_ROUNDS=60          Rounds per training matchup (default: 60)
#   EVAL_ROUNDS=100          Rounds per eval matchup (default: 100)
#   TRAIN_LIMIT=0            Limit number of training opponents (0 = all)
#   EVAL_LIMIT=0             Limit number of eval opponents (0 = all)
#   PORT=0                   Server port (0 = dynamic auto-port, default: 0)
#   WEIGHTS_NAME=my.pt       Weights file name in bots/python/dqn/checkpoints (default: dqn_weights_curriculum.pt)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUN_BATTLE_CMD="$ROOT_DIR/scripts/run/run_battle.sh"
DQN_BOT_DIR="$ROOT_DIR/bots/python/dqn/runtime"
LOG_ROOT="$ROOT_DIR/bots/python/dqn/logs/overnight_$(date +%Y%m%d_%H%M%S)"
PASSES="${PASSES:-2}"
TRAIN_ROUNDS="${TRAIN_ROUNDS:-60}"
EVAL_ROUNDS="${EVAL_ROUNDS:-100}"
TRAIN_LIMIT="${TRAIN_LIMIT:-0}"
EVAL_LIMIT="${EVAL_LIMIT:-0}"
PORT="${PORT:-0}"
WEIGHTS_NAME="${WEIGHTS_NAME:-dqn_weights_curriculum.pt}"
WEIGHTS_PATH="$ROOT_DIR/bots/python/dqn/checkpoints/$WEIGHTS_NAME"

mkdir -p "$LOG_ROOT"

if [ ! -x "$RUN_BATTLE_CMD" ]; then
  echo "Battle runner script not found or not executable: $RUN_BATTLE_CMD"
  exit 1
fi

if [ ! -d "$DQN_BOT_DIR" ]; then
  echo "Missing DQN booter directory: $DQN_BOT_DIR"
  exit 1
fi

if [ ! -f "$ROOT_DIR/scripts/run/tools/robocode-tankroyale-runner-0.38.2.jar" ]; then
  echo "Missing runner jar. Download with:"
  echo "  curl -L --fail -o \"$ROOT_DIR/scripts/run/tools/robocode-tankroyale-runner-0.38.2.jar\" https://github.com/robocode-dev/tank-royale/releases/download/v0.38.2/robocode-tankroyale-runner-0.38.2.jar"
  exit 1
fi

TRAIN_MATCHUPS=(
  "walls|/Users/jacobbecker/CPS485_RoboCode/SampleBots/Walls"
  "spinbot|/Users/jacobbecker/CPS485_RoboCode/SampleBots/SpinBot"
  "ramfire|/Users/jacobbecker/CPS485_RoboCode/SampleBots/RamFire"
  "trackfire|/Users/jacobbecker/CPS485_RoboCode/SampleBots/TrackFire"
  "velocitybot|/Users/jacobbecker/CPS485_RoboCode/SampleBots/VelocityBot"
  "crazy|/Users/jacobbecker/CPS485_RoboCode/SampleBots/Crazy"
  "corners|/Users/jacobbecker/CPS485_RoboCode/SampleBots/Corners"
)

EVAL_MATCHUPS=(
  "walls|/Users/jacobbecker/CPS485_RoboCode/SampleBots/Walls"
  "spinbot|/Users/jacobbecker/CPS485_RoboCode/SampleBots/SpinBot"
  "target|/Users/jacobbecker/CPS485_RoboCode/SampleBots/Target"
)
run_match() {
  local mode="$1"
  local matchup_name="$2"
  local bot_dir="$3"
  local rounds="$4"

  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"

  local runner_log="$LOG_ROOT/${stamp}_${mode}_${matchup_name}_runner.log"
  local dqn_jsonl="$LOG_ROOT/dqn_${mode}_${matchup_name}.jsonl"

  echo "[$(date '+%F %T')] START $mode vs $matchup_name for ${rounds} rounds"

  DQN_MODE="$mode" \
  DQN_WEIGHTS_PATH="$WEIGHTS_PATH" \
  DQN_LOG_PATH="$dqn_jsonl" \
  DQN_EVAL_EPSILON="0.0" \
  "$RUN_BATTLE_CMD" "$DQN_BOT_DIR" "$bot_dir" "$rounds" "$PORT" >"$runner_log" 2>&1

  echo "[$(date '+%F %T')] END   $mode vs $matchup_name"
}

echo "Overnight run started"
echo "  logs: $LOG_ROOT"
echo "  weights: $WEIGHTS_PATH"
echo "  embedded server port: $PORT"
echo "  passes: $PASSES"
echo "  train rounds per matchup: $TRAIN_ROUNDS"
echo "  eval rounds per matchup: $EVAL_ROUNDS"

for (( pass=1; pass<=PASSES; pass++ )); do
  echo "===== TRAIN PASS $pass / $PASSES ====="
  train_count=0
  for row in "${TRAIN_MATCHUPS[@]}"; do
    train_count=$(( train_count + 1 ))
    if [ "$TRAIN_LIMIT" -gt 0 ] && [ "$train_count" -gt "$TRAIN_LIMIT" ]; then
      break
    fi
    IFS='|' read -r name bot_dir <<< "$row"
    run_match "train" "$name" "$bot_dir" "$TRAIN_ROUNDS"
  done
done

echo "===== FINAL EVAL ====="
eval_count=0
for row in "${EVAL_MATCHUPS[@]}"; do
  eval_count=$(( eval_count + 1 ))
  if [ "$EVAL_LIMIT" -gt 0 ] && [ "$eval_count" -gt "$EVAL_LIMIT" ]; then
    break
  fi
  IFS='|' read -r name bot_dir <<< "$row"
  run_match "eval" "$name" "$bot_dir" "$EVAL_ROUNDS"
done

echo "Overnight run finished"
echo "Check logs and JSONL outputs in: $LOG_ROOT"
