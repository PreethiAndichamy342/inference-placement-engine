#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.demo_pids"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No demo PID file found ($PID_FILE). Nothing to stop."
  exit 0
fi

echo "Stopping Ollama demo instances..."

while IFS=: read -r provider pid; do
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid"
    echo "  [$provider] PID $pid stopped"
  else
    echo "  [$provider] PID $pid was already stopped"
  fi
done < "$PID_FILE"

rm -f "$PID_FILE"
echo "Done."
