#!/usr/bin/env bash
set -euo pipefail
SCRIPT_NAME="train_jacob3_0_mapsize"
source "$(cd "$(dirname "$0")/.." && pwd)/lib/common.sh"
parse_rounds_override "$@"
ROUNDS="${ROUNDS_OVERRIDE:-100}"
prepare_log_paths "$SCRIPT_NAME" "$SCRIPT_NAME"
start_script_logging
write_jsonl_header "$SCRIPT_NAME" "Jacob3_0"
python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario --script-name "$SCRIPT_NAME" --scenario-name "jacob3_0_sarsa_small" --timestamp "$TIMESTAMP" --output-jsonl "$JSONL_FILE" --bot Jacob3_0 --mode train --rounds "$ROUNDS" --arena-width 400 --arena-height 400 --opponents SarsaBot
python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario --script-name "$SCRIPT_NAME" --scenario-name "jacob3_0_sarsa_medium" --timestamp "$TIMESTAMP" --output-jsonl "$JSONL_FILE" --bot Jacob3_0 --mode train --rounds "$ROUNDS" --arena-width 800 --arena-height 800 --opponents SarsaBot
python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario --script-name "$SCRIPT_NAME" --scenario-name "jacob3_0_sarsa_large" --timestamp "$TIMESTAMP" --output-jsonl "$JSONL_FILE" --bot Jacob3_0 --mode train --rounds "$ROUNDS" --arena-width 1200 --arena-height 1200 --opponents SarsaBot
echo "JSONL_PATH=$JSONL_FILE"
