#!/usr/bin/env bash
# Quick smoke test before overnight run.
# Runs a very short curriculum slice to verify server/bot orchestration works.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

if [ ! -x "./tools/robocode-tankroyale-server" ]; then
  echo "Missing local server launcher. Run: ./install_headless_server.sh"
  exit 1
fi

echo "[test] Starting quick smoke test (short round-based run)"
PASSES=1 TRAIN_ROUNDS=5 EVAL_ROUNDS=5 TRAIN_LIMIT=1 EVAL_LIMIT=1 ./overnight_train.sh

echo "[test] Smoke test complete"

echo "[test] Morning analysis for latest run:"
python3 ./analyze_overnight.py
