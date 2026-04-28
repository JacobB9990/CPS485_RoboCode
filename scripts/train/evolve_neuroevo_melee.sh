#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

POPULATION="${POPULATION:-24}"
GENERATIONS="${GENERATIONS:-30}"
SEED="${SEED:-1337}"
CURRENT_GENOME="${CURRENT_GENOME:-$ROOT_DIR/NeuroEvoMelee/data/current_genome.json}"
BEST_GENOME="${BEST_GENOME:-$ROOT_DIR/NeuroEvoMelee/data/best_genome.json}"
LOG_PATH="${LOG_PATH:-$ROOT_DIR/NeuroEvoMelee/logs/evolution_log.jsonl}"
EVALUATE_COMMAND="${EVALUATE_COMMAND:-}"

python3 -m NeuroEvoMelee.training.train_neuroevo_melee \
  --population "$POPULATION" \
  --generations "$GENERATIONS" \
  --seed "$SEED" \
  --current-genome "$CURRENT_GENOME" \
  --best-genome "$BEST_GENOME" \
  --log-path "$LOG_PATH" \
  --evaluate-command "$EVALUATE_COMMAND"
