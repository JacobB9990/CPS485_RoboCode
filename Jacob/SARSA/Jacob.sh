#!/bin/sh
set -e

# Change to this bot's directory
cd -- "$(dirname -- "$0")"

# Robocode Python API looks up <BOT_NAME>.json. Our bot class is SarsaBot.
export BOT_NAME="SarsaBot"

# Reuse the sample bots dependency installer
if [ -x "../SampleBots/deps/install-dependencies.sh" ]; then
    ../SampleBots/deps/install-dependencies.sh
fi

# Prefer sample-bots venv if it exists
if [ -x "../SampleBots/deps/venv/bin/python" ]; then
    exec "../SampleBots/deps/venv/bin/python" "main.py"
elif command -v python3 >/dev/null 2>&1; then
    exec python3 "main.py"
elif command -v python >/dev/null 2>&1; then
    exec python "main.py"
else
    echo "Error: Python not found. Please install python3 or python." >&2
    exit 1
fi
