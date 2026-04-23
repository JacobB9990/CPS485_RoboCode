#!/bin/sh
set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
ROOT_DIR="$(CDPATH= cd -- "$SCRIPT_DIR/../.." && pwd)"
DEPS_DIR="$ROOT_DIR/SampleBots/deps"

export BOT_NAME="SarsaBot"

if [ -x "$DEPS_DIR/install-dependencies.sh" ]; then
    "$DEPS_DIR/install-dependencies.sh"
fi

if [ -x "$DEPS_DIR/venv/bin/python" ]; then
    exec "$DEPS_DIR/venv/bin/python" -m bots.python.sarsa.runtime.sarsa_bot "$@"
elif command -v python3 >/dev/null 2>&1; then
    exec python3 -m bots.python.sarsa.runtime.sarsa_bot "$@"
elif command -v python >/dev/null 2>&1; then
    exec python -m bots.python.sarsa.runtime.sarsa_bot "$@"
else
    echo "Error: Python not found. Please install python3 or python." >&2
    exit 1
fi
