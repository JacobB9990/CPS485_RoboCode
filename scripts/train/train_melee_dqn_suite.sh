#!/usr/bin/env bash
set -euo pipefail
SCRIPT_NAME="train_melee_dqn_suite"
source "$(cd "$(dirname "$0")/.." && pwd)/lib/common.sh"
parse_rounds_override "$@"
ROUNDS="${ROUNDS_OVERRIDE:-100}"
prepare_log_paths "$SCRIPT_NAME" "$SCRIPT_NAME"
start_script_logging
write_jsonl_header "$SCRIPT_NAME" "MeleeDQN"
python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario --script-name "$SCRIPT_NAME" --scenario-name "melee_dqn_vs_jacob3_0" --timestamp "$TIMESTAMP" --output-jsonl "$JSONL_FILE" --bot MeleeDQN --mode train --rounds "$ROUNDS" --opponents Jacob3_0
python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario --script-name "$SCRIPT_NAME" --scenario-name "melee_dqn_vs_sarsa" --timestamp "$TIMESTAMP" --output-jsonl "$JSONL_FILE" --bot MeleeDQN --mode train --rounds "$ROUNDS" --opponents SarsaBot
python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario --script-name "$SCRIPT_NAME" --scenario-name "melee_dqn_4bot" --timestamp "$TIMESTAMP" --output-jsonl "$JSONL_FILE" --bot MeleeDQN --mode train --rounds "$ROUNDS" --opponents Jacob3_0 PPOBot SarsaBot NeuroEvoMelee
echo "JSONL_PATH=$JSONL_FILE"
