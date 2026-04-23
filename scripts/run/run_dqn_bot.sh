#!/bin/bash
set -euo pipefail

export BOT_NAME="DQNBot"
export BOT_VERSION="1.0.0"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$ROOT_DIR"
python3 -m bots.python.dqn.runtime.run_bot "$@"
