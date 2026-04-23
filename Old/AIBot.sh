#!/bin/sh
set -e

# Change to script directory
cd -- "$(dirname -- "$0")"

# Install dependencies (shared with SampleBots)
../SampleBots/deps/install-dependencies.sh

# Try venv python first, then system python
if [ -x "../SampleBots/deps/venv/bin/python" ]; then
    exec "../SampleBots/deps/venv/bin/python" "AIBot.py" "$@"
elif command -v python3 >/dev/null 2>&1; then
    exec python3 "AIBot.py" "$@"
elif command -v python >/dev/null 2>&1; then
    exec python "AIBot.py" "$@"
else
    echo "Error: Python not found. Please install python3 or python." >&2
    exit 1
fi
