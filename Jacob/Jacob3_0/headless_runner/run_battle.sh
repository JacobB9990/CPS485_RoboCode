#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
RUNNER_JAR="$ROOT_DIR/tools/robocode-tankroyale-runner-0.38.2.jar"
SRC_DIR="$ROOT_DIR/headless_runner/src"
BUILD_DIR="$ROOT_DIR/headless_runner/build"

BOT_A="${1:-}"
BOT_B="${2:-}"
ROUNDS="${3:-50}"
PORT="${4:-0}"

if [ -z "$BOT_A" ] || [ -z "$BOT_B" ]; then
  echo "Usage: ./headless_runner/run_battle.sh <botA_dir> <botB_dir> [rounds] [port]"
  exit 2
fi

if [ ! -f "$RUNNER_JAR" ]; then
  echo "Missing runner jar: $RUNNER_JAR"
  echo "Download it with:"
  echo "  curl -L --fail -o \"$RUNNER_JAR\" https://github.com/robocode-dev/tank-royale/releases/download/v0.38.2/robocode-tankroyale-runner-0.38.2.jar"
  exit 1
fi

mkdir -p "$BUILD_DIR"

javac -cp "$RUNNER_JAR" -d "$BUILD_DIR" "$SRC_DIR/RunBattle.java"
java -cp "$BUILD_DIR:$RUNNER_JAR" RunBattle "$BOT_A" "$BOT_B" "$ROUNDS" "$PORT"
