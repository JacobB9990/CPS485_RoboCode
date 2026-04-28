#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT_DIR/scripts/lib/common.sh"

PARALLEL=0
ROUNDS_ARG=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --parallel)
      PARALLEL=1
      shift
      ;;
    --rounds)
      ROUNDS_ARG=(--rounds "$2")
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

TIMESTAMP="$(timestamp_now)"
export PIPELINE_TIMESTAMP="$TIMESTAMP"
MASTER_LOG="$ROOT_DIR/logs/run_all_training_${TIMESTAMP}.log"
mkdir -p "$ROOT_DIR/logs"
exec > >(tee -a "$MASTER_LOG") 2>&1

scripts=(
  "scripts/train/train_jacob3_0_1v1.sh"
  "scripts/train/train_jacob3_0_melee.sh"
  "scripts/train/train_jacob3_0_mapsize.sh"
  "scripts/train/train_melee_dqn_suite.sh"
  "scripts/train/train_ppo_suite.sh"
  "scripts/train/train_sarsa_suite.sh"
  "scripts/train/evolve_neuroevo_suite.sh"
  "scripts/eval/eval_all_bots_1v1.sh"
  "scripts/eval/eval_all_bots_melee.sh"
  "scripts/data/collect_game_states.sh"
  "scripts/data/collect_reward_curves.sh"
)

start_epoch="$(date +%s)"
declare -a succeeded=()
declare -a failed=()

run_one() {
  local script_path="$1"
  if bash "$ROOT_DIR/$script_path" "${ROUNDS_ARG[@]}"; then
    succeeded+=("$script_path")
  else
    failed+=("$script_path")
  fi
}

if [[ "$PARALLEL" -eq 1 ]]; then
  declare -a pids=()
  for script_path in "${scripts[@]}"; do
    bash "$ROOT_DIR/$script_path" "${ROUNDS_ARG[@]}" &
    pids+=("$!")
  done
  for idx in "${!pids[@]}"; do
    if wait "${pids[$idx]}"; then
      succeeded+=("${scripts[$idx]}")
    else
      failed+=("${scripts[$idx]}")
    fi
  done
else
  for script_path in "${scripts[@]}"; do
    run_one "$script_path"
  done
fi

end_epoch="$(date +%s)"
runtime="$(( end_epoch - start_epoch ))"

echo
echo "Summary"
echo "Succeeded: ${#succeeded[@]}"
for item in "${succeeded[@]}"; do
  echo "  OK  $item"
done
echo "Failed: ${#failed[@]}"
for item in "${failed[@]}"; do
  echo "  ERR $item"
done
echo "Total runtime seconds: $runtime"
echo "JSONL manifest:"
find "$ROOT_DIR/logs" -type f -name "*${TIMESTAMP}.jsonl" | sort
