#!/usr/bin/env bash
set -euo pipefail
SCRIPT_NAME="train_ppo_suite"
source "$(cd "$(dirname "$0")/.." && pwd)/lib/common.sh"
parse_rounds_override "$@"
ROUNDS_1V1="${ROUNDS_OVERRIDE:-200}"
ROUNDS_MELEE="${ROUNDS_OVERRIDE:-100}"
prepare_log_paths "$SCRIPT_NAME" "$SCRIPT_NAME"
start_script_logging
write_jsonl_header "$SCRIPT_NAME" "PPOBot"
python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario --script-name "$SCRIPT_NAME" --scenario-name "ppo_vs_jacob3_0" --timestamp "$TIMESTAMP" --output-jsonl "$JSONL_FILE" --bot PPOBot --mode train --rounds "$ROUNDS_1V1" --opponents Jacob3_0
python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario --script-name "$SCRIPT_NAME" --scenario-name "ppo_vs_melee_dqn" --timestamp "$TIMESTAMP" --output-jsonl "$JSONL_FILE" --bot PPOBot --mode train --rounds "$ROUNDS_1V1" --opponents MeleeDQN
python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario --script-name "$SCRIPT_NAME" --scenario-name "ppo_4bot" --timestamp "$TIMESTAMP" --output-jsonl "$JSONL_FILE" --bot PPOBot --mode train --rounds "$ROUNDS_MELEE" --opponents Jacob3_0 MeleeDQN SarsaBot NeuroEvoMelee
echo "JSONL_PATH=$JSONL_FILE"
