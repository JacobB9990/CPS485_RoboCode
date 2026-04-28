#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python3 -m NeuroEvoMelee.runtime.neuroevo_melee_bot "$@"
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$ROOT_DIR/src"
BUILD_DIR="$ROOT_DIR/.build"
API_JAR="${BOT_API_JAR:-$ROOT_DIR/../../../scripts/run/tools/robocode-tankroyale-bot-api-0.38.2.jar}"

if [ ! -f "$API_JAR" ]; then
  echo "Missing Tank Royale Bot API jar: $API_JAR"
  echo "Download it with:"
  echo "  curl -L --fail -o \"$ROOT_DIR/../../../scripts/run/tools/robocode-tankroyale-bot-api-0.38.2.jar\" https://github.com/robocode-dev/tank-royale/releases/download/v0.38.2/robocode-tankroyale-bot-api-0.38.2.jar"
  exit 1
fi

mkdir -p "$BUILD_DIR"

# Compile every launch so training runs stay deterministic after source edits.
find "$SRC_DIR" -name '*.java' -print0 | xargs -0 javac -cp "$API_JAR" -d "$BUILD_DIR"

exec java -cp "$BUILD_DIR:$API_JAR" neuroevo.NeuroEvoMeleeBot
