#!/usr/bin/env bash
# Installs a local headless Tank Royale server launcher for this project.
# No sudo required.

set -euo pipefail

VERSION="0.38.2"
BASE_URL="https://github.com/robocode-dev/tank-royale/releases/download/v${VERSION}"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
TOOLS_DIR="$ROOT_DIR/tools"
SERVER_JAR="$TOOLS_DIR/robocode-tankroyale-server-${VERSION}.jar"
LAUNCHER="$TOOLS_DIR/robocode-tankroyale-server"

mkdir -p "$TOOLS_DIR"

if [ ! -f "$SERVER_JAR" ]; then
  echo "Downloading server jar v${VERSION}..."
  curl -L --fail -o "$SERVER_JAR" "$BASE_URL/robocode-tankroyale-server-${VERSION}.jar"
else
  echo "Server jar already exists: $SERVER_JAR"
fi

cat > "$LAUNCHER" <<EOF
#!/usr/bin/env bash
set -euo pipefail
java --enable-native-access=ALL-UNNAMED -jar "$SERVER_JAR" "\$@"
EOF

chmod +x "$LAUNCHER"

cat <<EOF

Installed local launcher:
  $LAUNCHER

Run headless server:
  $LAUNCHER --port=7654 --games=classic

Note: server v0.38.2 does not support --config. Battle setup JSON files are still
useful for documentation/reproducibility, but the server CLI itself only accepts
options like --games, --port, --tps, and secrets.

Optional: add to PATH for this shell session:
  export PATH="$TOOLS_DIR:\$PATH"

Then you can run:
  robocode-tankroyale-server --port=7654 --games=classic
EOF
