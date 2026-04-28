#!/usr/bin/env bash
set -euo pipefail
SCRIPT_NAME="collect_reward_curves"
source "$(cd "$(dirname "$0")/.." && pwd)/lib/common.sh"
parse_rounds_override "$@"
ROUNDS="${ROUNDS_OVERRIDE:-50}"
prepare_log_paths "$SCRIPT_NAME" "reward_curves" "data"
start_script_logging
write_jsonl_header "$SCRIPT_NAME" "Jacob3_0,PPOBot,MeleeDQN,SarsaBot"
python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario --script-name "$SCRIPT_NAME" --scenario-name "jacob3_0_reward_burst" --timestamp "$TIMESTAMP" --output-jsonl "$JSONL_FILE" --bot Jacob3_0 --mode train --rounds "$ROUNDS" --opponents MeleeDQN
python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario --script-name "$SCRIPT_NAME" --scenario-name "ppo_reward_burst" --timestamp "$TIMESTAMP" --output-jsonl "$JSONL_FILE" --bot PPOBot --mode train --rounds "$ROUNDS" --opponents Jacob3_0
python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario --script-name "$SCRIPT_NAME" --scenario-name "melee_dqn_reward_burst" --timestamp "$TIMESTAMP" --output-jsonl "$JSONL_FILE" --bot MeleeDQN --mode train --rounds "$ROUNDS" --opponents PPOBot
python3 "$ROOT_DIR/scripts/lib/royale_harness.py" run-scenario --script-name "$SCRIPT_NAME" --scenario-name "sarsa_reward_burst" --timestamp "$TIMESTAMP" --output-jsonl "$JSONL_FILE" --bot SarsaBot --mode train --rounds "$ROUNDS" --opponents Jacob3_0
echo "JSONL_PATH=$JSONL_FILE"
