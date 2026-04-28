#!/usr/bin/env bash
set -euo pipefail
SCRIPT_NAME="train_sarsa_suite"
source "$(cd "$(dirname "$0")/.." && pwd)/lib/common.sh"
parse_rounds_override "$@"
ROUNDS="${ROUNDS_OVERRIDE:-100}"
prepare_log_paths "$SCRIPT_NAME" "$SCRIPT_NAME"
start_script_logging
write_jsonl_header "$SCRIPT_NAME" "SarsaBot"
for opponent in Jacob3_0 MeleeDQN PPOBot; do
  python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario \
    --script-name "$SCRIPT_NAME" \
    --scenario-name "sarsa_vs_${opponent,,}" \
    --timestamp "$TIMESTAMP" \
    --output-jsonl "$JSONL_FILE" \
    --bot SarsaBot \
    --mode train \
    --rounds "$ROUNDS" \
    --opponents "$opponent"
done
echo "JSONL_PATH=$JSONL_FILE"
