#!/usr/bin/env bash
set -euo pipefail
SCRIPT_NAME="eval_all_bots_1v1"
source "$(cd "$(dirname "$0")/.." && pwd)/lib/common.sh"
parse_rounds_override "$@"
ROUNDS="${ROUNDS_OVERRIDE:-50}"
prepare_log_paths "$SCRIPT_NAME" "eval_1v1_matrix" "eval"
start_script_logging
write_jsonl_header "$SCRIPT_NAME" "all"
bots=(Jacob3_0 MeleeDQN PPOBot SarsaBot NeuroEvoMelee)
for bot in "${bots[@]}"; do
  for opponent in "${bots[@]}"; do
    if [[ "$bot" == "$opponent" ]]; then
      continue
    fi
    python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario \
      --script-name "$SCRIPT_NAME" \
      --scenario-name "${bot,,}_vs_${opponent,,}_eval" \
      --timestamp "$TIMESTAMP" \
      --output-jsonl "$JSONL_FILE" \
      --bot "$bot" \
      --mode eval \
      --rounds "$ROUNDS" \
      --opponents "$opponent"
  done
done
echo "JSONL_PATH=$JSONL_FILE"
