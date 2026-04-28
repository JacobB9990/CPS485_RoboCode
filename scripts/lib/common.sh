#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LOG_ROOT="$ROOT_DIR/logs"

timestamp_now() {
  if [[ -n "${PIPELINE_TIMESTAMP:-}" ]]; then
    printf '%s\n' "$PIPELINE_TIMESTAMP"
  else
    date +"%Y%m%d_%H%M%S"
  fi
}

parse_rounds_override() {
  ROUNDS_OVERRIDE=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --rounds)
        if [[ $# -lt 2 ]]; then
          echo "Missing value for --rounds" >&2
          exit 2
        fi
        ROUNDS_OVERRIDE="$2"
        shift 2
        ;;
      *)
        echo "Unknown argument: $1" >&2
        exit 2
        ;;
    esac
  done
}

prepare_log_paths() {
  local script_name="$1"
  local jsonl_basename="$2"
  local subdir="${3:-}"
  TIMESTAMP="$(timestamp_now)"
  local target_dir="$LOG_ROOT"
  if [[ -n "$subdir" ]]; then
    target_dir="$LOG_ROOT/$subdir"
  fi
  mkdir -p "$target_dir"
  LOG_FILE="$target_dir/${script_name}_${TIMESTAMP}.log"
  JSONL_FILE="$target_dir/${jsonl_basename}_${TIMESTAMP}.jsonl"
}

start_script_logging() {
  mkdir -p "$(dirname "$LOG_FILE")" "$(dirname "$JSONL_FILE")"
  exec > >(tee -a "$LOG_FILE") 2>&1
}

write_jsonl_header() {
  local script_name="$1"
  local bot_name="$2"
  printf '{"meta":{"script":"%s","bot":"%s","timestamp":"%s"}}\n' "$script_name" "$bot_name" "$TIMESTAMP" > "$JSONL_FILE"
}
