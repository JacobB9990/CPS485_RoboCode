#!/usr/bin/env bash
set -euo pipefail
SCRIPT_NAME="train_big_test_top3"
source "$(cd "$(dirname "$0")/.." && pwd)/lib/common.sh"
parse_rounds_override "$@"

# 15k rounds for extensive training on top 3 performers
ROUNDS_1V1="${ROUNDS_OVERRIDE:-15000}"
ROUNDS_MELEE="${ROUNDS_OVERRIDE:-5000}"

prepare_log_paths "$SCRIPT_NAME" "$SCRIPT_NAME"
start_script_logging
write_jsonl_header "$SCRIPT_NAME" "BigTestTop3"

echo "[BigTestTop3] Starting comprehensive test of top 3 bots: Jacob3_0, PPOBot, SarsaBot"
echo "[BigTestTop3] 1v1 rounds: $ROUNDS_1V1"
echo "[BigTestTop3] Melee rounds: $ROUNDS_MELEE"

# ============================================
# 1v1 Training Matrix: Jacob3_0 vs PPOBot
# ============================================
echo "[BigTestTop3] Jacob3_0 vs PPOBot 1v1 training..."
python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario \
  --script-name "$SCRIPT_NAME" \
  --scenario-name "jacob3_0_vs_ppobot_1v1" \
  --timestamp "$TIMESTAMP" \
  --output-jsonl "$JSONL_FILE" \
  --bot Jacob3_0 \
  --mode train \
  --rounds "$ROUNDS_1V1" \
  --opponents PPOBot

echo "[BigTestTop3] PPOBot vs Jacob3_0 1v1 training..."
python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario \
  --script-name "$SCRIPT_NAME" \
  --scenario-name "ppobot_vs_jacob3_0_1v1" \
  --timestamp "$TIMESTAMP" \
  --output-jsonl "$JSONL_FILE" \
  --bot PPOBot \
  --mode train \
  --rounds "$ROUNDS_1V1" \
  --opponents Jacob3_0

# ============================================
# 1v1 Training Matrix: Jacob3_0 vs SarsaBot
# ============================================
echo "[BigTestTop3] Jacob3_0 vs SarsaBot 1v1 training..."
python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario \
  --script-name "$SCRIPT_NAME" \
  --scenario-name "jacob3_0_vs_sarsabot_1v1" \
  --timestamp "$TIMESTAMP" \
  --output-jsonl "$JSONL_FILE" \
  --bot Jacob3_0 \
  --mode train \
  --rounds "$ROUNDS_1V1" \
  --opponents SarsaBot

echo "[BigTestTop3] SarsaBot vs Jacob3_0 1v1 training..."
python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario \
  --script-name "$SCRIPT_NAME" \
  --scenario-name "sarsabot_vs_jacob3_0_1v1" \
  --timestamp "$TIMESTAMP" \
  --output-jsonl "$JSONL_FILE" \
  --bot SarsaBot \
  --mode train \
  --rounds "$ROUNDS_1V1" \
  --opponents Jacob3_0

# ============================================
# 1v1 Training Matrix: PPOBot vs SarsaBot
# ============================================
echo "[BigTestTop3] PPOBot vs SarsaBot 1v1 training..."
python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario \
  --script-name "$SCRIPT_NAME" \
  --scenario-name "ppobot_vs_sarsabot_1v1" \
  --timestamp "$TIMESTAMP" \
  --output-jsonl "$JSONL_FILE" \
  --bot PPOBot \
  --mode train \
  --rounds "$ROUNDS_1V1" \
  --opponents SarsaBot

echo "[BigTestTop3] SarsaBot vs PPOBot 1v1 training..."
python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario \
  --script-name "$SCRIPT_NAME" \
  --scenario-name "sarsabot_vs_ppobot_1v1" \
  --timestamp "$TIMESTAMP" \
  --output-jsonl "$JSONL_FILE" \
  --bot SarsaBot \
  --mode train \
  --rounds "$ROUNDS_1V1" \
  --opponents PPOBot

# ============================================
# Melee Training: All three together
# ============================================
echo "[BigTestTop3] Jacob3_0 vs PPOBot + SarsaBot melee training..."
python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario \
  --script-name "$SCRIPT_NAME" \
  --scenario-name "jacob3_0_melee_top3" \
  --timestamp "$TIMESTAMP" \
  --output-jsonl "$JSONL_FILE" \
  --bot Jacob3_0 \
  --mode train \
  --rounds "$ROUNDS_MELEE" \
  --opponents PPOBot SarsaBot

echo "[BigTestTop3] PPOBot vs Jacob3_0 + SarsaBot melee training..."
python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario \
  --script-name "$SCRIPT_NAME" \
  --scenario-name "ppobot_melee_top3" \
  --timestamp "$TIMESTAMP" \
  --output-jsonl "$JSONL_FILE" \
  --bot PPOBot \
  --mode train \
  --rounds "$ROUNDS_MELEE" \
  --opponents Jacob3_0 SarsaBot

echo "[BigTestTop3] SarsaBot vs Jacob3_0 + PPOBot melee training..."
python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario \
  --script-name "$SCRIPT_NAME" \
  --scenario-name "sarsabot_melee_top3" \
  --timestamp "$TIMESTAMP" \
  --output-jsonl "$JSONL_FILE" \
  --bot SarsaBot \
  --mode train \
  --rounds "$ROUNDS_MELEE" \
  --opponents Jacob3_0 PPOBot

echo "[BigTestTop3] === ALL TOP 3 SCENARIOS COMPLETE ==="
echo "JSONL_PATH=$JSONL_FILE"
