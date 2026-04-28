#!/usr/bin/env bash
# Test trained DQN bot against specific opponents
# Usage:
#   ./test_trained_bot.sh walls 20              # Test 20 rounds vs Walls
#   ./test_trained_bot.sh target 50             # Test 50 rounds vs Target
#   ./test_trained_bot.sh spinbot 30            # etc

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <opponent_name> <rounds> [weights_file]"
  echo ""
  echo "Available opponents:"
  echo "  walls, spinbot, ramfire, trackfire, velocitybot, crazy, corners, target"
  echo ""
  echo "Examples:"
  echo "  $0 walls 20                                    # 20 rounds vs Walls"
  echo "  $0 target 50 dqn_weights_curriculum.pt        # 50 rounds vs Target with specific weights"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
OPPONENT_NAME="$1"
ROUNDS="$2"
WEIGHTS_FILE="${3:-dqn_weights_curriculum.pt}"

# Bot paths
declare -A OPPONENTS=(
  [walls]="/Users/jacobbecker/CPS485_RoboCode/SampleBots/Walls"
  [spinbot]="/Users/jacobbecker/CPS485_RoboCode/SampleBots/SpinBot"
  [ramfire]="/Users/jacobbecker/CPS485_RoboCode/SampleBots/RamFire"
  [trackfire]="/Users/jacobbecker/CPS485_RoboCode/SampleBots/TrackFire"
  [velocitybot]="/Users/jacobbecker/CPS485_RoboCode/SampleBots/VelocityBot"
  [crazy]="/Users/jacobbecker/CPS485_RoboCode/SampleBots/Crazy"
  [corners]="/Users/jacobbecker/CPS485_RoboCode/SampleBots/Corners"
  [target]="/Users/jacobbecker/CPS485_RoboCode/SampleBots/Target"
)

if [ -z "${OPPONENTS[$OPPONENT_NAME]:-}" ]; then
  echo "Unknown opponent: $OPPONENT_NAME"
  echo "Available: ${!OPPONENTS[@]}"
  exit 1
fi

OPPONENT_DIR="${OPPONENTS[$OPPONENT_NAME]}"
DQN_BOT_DIR="$ROOT_DIR/bots/python/dqn/runtime"
WEIGHTS_PATH="$ROOT_DIR/bots/python/dqn/checkpoints/$WEIGHTS_FILE"
LOG_DIR="$ROOT_DIR/bots/python/dqn/logs/eval/test_$(date +%Y%m%d_%H%M%S)"
TEST_JSONL="$LOG_DIR/dqn_eval_${OPPONENT_NAME}.jsonl"

mkdir -p "$LOG_DIR"

if [ ! -f "$WEIGHTS_PATH" ]; then
  echo "ERROR: Weights file not found: $WEIGHTS_PATH"
  echo "Available weights:"
  ls -lh "$ROOT_DIR/bots/python/dqn/checkpoints"/*.pt 2>/dev/null | awk '{print "  " $NF}' || echo "  (none)"
  exit 1
fi

if [ ! -d "$OPPONENT_DIR" ]; then
  echo "ERROR: Opponent not found: $OPPONENT_DIR"
  exit 1
fi

echo "Testing DQN bot vs $OPPONENT_NAME"
echo "  Weights: $WEIGHTS_PATH"
echo "  Rounds: $ROUNDS"
echo "  Log: $LOG_DIR"
echo ""

# Run battle in eval mode
DQN_MODE="eval" \
  DQN_WEIGHTS_PATH="$WEIGHTS_PATH" \
  DQN_LOG_PATH="$TEST_JSONL" \
  DQN_EVAL_EPSILON="0.0" \
  "$ROOT_DIR/scripts/run/run_battle.sh" "$DQN_BOT_DIR" "$OPPONENT_DIR" "$ROUNDS" 0 \
  > "$LOG_DIR/battle.log" 2>&1

# Quick stats
if [ -f "$TEST_JSONL" ]; then
  episodes=$(wc -l < "$TEST_JSONL")
  wins=$(grep -c '"winner": true' "$TEST_JSONL" || echo 0)
  avg_reward=$(python3 -c "
import json
rewards = [json.loads(l)['reward'] for l in open('$TEST_JSONL')]
print(f'{sum(rewards)/len(rewards):.2f}' if rewards else '0')
" || echo "N/A")
  
  echo "Results:"
  echo "  Episodes: $episodes"
  echo "  Wins: $wins / $episodes ($(( wins * 100 / episodes ))%)"
  echo "  Avg Reward: $avg_reward"
  echo ""
  echo "Full log: $LOG_DIR"
else
  echo "ERROR: No JSONL output generated"
  cat "$LOG_DIR/battle.log"
  exit 1
fi
