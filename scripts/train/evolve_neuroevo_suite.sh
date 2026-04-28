#!/usr/bin/env bash
set -euo pipefail
SCRIPT_NAME="evolve_neuroevo_suite"
source "$(cd "$(dirname "$0")/.." && pwd)/lib/common.sh"
parse_rounds_override "$@"
GENERATIONS="${ROUNDS_OVERRIDE:-20}"
prepare_log_paths "$SCRIPT_NAME" "$SCRIPT_NAME"
start_script_logging
write_jsonl_header "$SCRIPT_NAME" "NeuroEvoMelee"
python3 -m NeuroEvoMelee.training.train_neuroevo_melee --generations "$GENERATIONS" --log-path "$JSONL_FILE"
echo "JSONL_PATH=$JSONL_FILE"
