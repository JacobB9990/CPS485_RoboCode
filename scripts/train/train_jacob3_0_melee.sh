#!/usr/bin/env bash
set -euo pipefail
SCRIPT_NAME="train_jacob3_0_melee"
source "$(cd "$(dirname "$0")/.." && pwd)/lib/common.sh"
parse_rounds_override "$@"
ROUNDS="${ROUNDS_OVERRIDE:-100}"
prepare_log_paths "$SCRIPT_NAME" "$SCRIPT_NAME"
start_script_logging
write_jsonl_header "$SCRIPT_NAME" "Jacob3_0"
python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario --script-name "$SCRIPT_NAME" --scenario-name "jacob3_0_melee_3opps" --timestamp "$TIMESTAMP" --output-jsonl "$JSONL_FILE" --bot Jacob3_0 --mode train --rounds "$ROUNDS" --opponents MeleeDQN PPOBot SarsaBot
python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario --script-name "$SCRIPT_NAME" --scenario-name "jacob3_0_melee_5opps" --timestamp "$TIMESTAMP" --output-jsonl "$JSONL_FILE" --bot Jacob3_0 --mode train --rounds "$ROUNDS" --opponents MeleeDQN PPOBot SarsaBot NeuroEvoMelee MeleeDQN:LeagueMeleeA
python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario --script-name "$SCRIPT_NAME" --scenario-name "jacob3_0_melee_8opps" --timestamp "$TIMESTAMP" --output-jsonl "$JSONL_FILE" --bot Jacob3_0 --mode train --rounds "$ROUNDS" --opponents MeleeDQN PPOBot SarsaBot NeuroEvoMelee MeleeDQN:LeagueMeleeA PPOBot:LeagueMeleeA SarsaBot:LeagueMeleeA NeuroEvoMelee:LeagueMeleeA
echo "JSONL_PATH=$JSONL_FILE"
