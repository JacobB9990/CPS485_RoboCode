#!/usr/bin/env bash
set -euo pipefail
SCRIPT_NAME="collect_game_states"
source "$(cd "$(dirname "$0")/.." && pwd)/lib/common.sh"
parse_rounds_override "$@"
ROUNDS="${ROUNDS_OVERRIDE:-200}"
prepare_log_paths "$SCRIPT_NAME" "game_states" "data"
start_script_logging
write_jsonl_header "$SCRIPT_NAME" "Jacob3_0,MeleeDQN"
python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario --script-name "$SCRIPT_NAME" --scenario-name "jacob3_0_1v1_states" --timestamp "$TIMESTAMP" --output-jsonl "$JSONL_FILE" --bot Jacob3_0 --mode eval --rounds "$ROUNDS" --collect-states --opponents MeleeDQN
python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario --script-name "$SCRIPT_NAME" --scenario-name "jacob3_0_melee_states" --timestamp "$TIMESTAMP" --output-jsonl "$JSONL_FILE" --bot Jacob3_0 --mode eval --rounds "$ROUNDS" --collect-states --opponents MeleeDQN PPOBot SarsaBot NeuroEvoMelee
python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario --script-name "$SCRIPT_NAME" --scenario-name "melee_dqn_1v1_states" --timestamp "$TIMESTAMP" --output-jsonl "$JSONL_FILE" --bot MeleeDQN --mode eval --rounds "$ROUNDS" --collect-states --opponents Jacob3_0
python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario --script-name "$SCRIPT_NAME" --scenario-name "melee_dqn_melee_states" --timestamp "$TIMESTAMP" --output-jsonl "$JSONL_FILE" --bot MeleeDQN --mode eval --rounds "$ROUNDS" --collect-states --opponents Jacob3_0 PPOBot SarsaBot NeuroEvoMelee
echo "JSONL_PATH=$JSONL_FILE"
