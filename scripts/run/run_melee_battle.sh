#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNNER_JAR="$ROOT_DIR/tools/robocode-tankroyale-runner-0.38.2.jar"
SRC_DIR="$ROOT_DIR"
BUILD_DIR="$ROOT_DIR/.build"

ROUNDS="${1:-}"
PORT="${2:-0}"
shift 2 || true

if [ -z "$ROUNDS" ] || [ "$#" -lt 2 ]; then
  echo "Usage: ./scripts/run/run_melee_battle.sh <rounds> <port> <bot_dir> <bot_dir> [<bot_dir>...]"
  exit 2
fi

if [ ! -f "$RUNNER_JAR" ]; then
  echo "Missing runner jar: $RUNNER_JAR"
  echo "Download it with:"
  echo "  curl -L --fail -o \"$RUNNER_JAR\" https://github.com/robocode-dev/tank-royale/releases/download/v0.38.2/robocode-tankroyale-runner-0.38.2.jar"
  exit 1
fi

mkdir -p "$BUILD_DIR"

javac -cp "$RUNNER_JAR" -d "$BUILD_DIR" "$SRC_DIR/RunMeleeBattle.java"
java -cp "$BUILD_DIR:$RUNNER_JAR" RunMeleeBattle "$ROUNDS" "$PORT" "$@"
