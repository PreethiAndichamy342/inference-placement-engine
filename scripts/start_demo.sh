#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_FILE="$SCRIPT_DIR/.demo_pids"

if [[ -f "$PID_FILE" ]]; then
  echo "Demo already running (found $PID_FILE). Run stop_demo.sh first."
  exit 1
fi

declare -A INSTANCES=(
  ["aws"]=11434
  ["gcp"]=11435
  ["onprem"]=11436
)

echo "Starting Ollama demo instances..."

> "$PID_FILE"

for provider in aws gcp onprem; do
  port="${INSTANCES[$provider]}"
  log="/tmp/ollama_${provider}.log"

  OLLAMA_HOST="0.0.0.0:${port}" ollama serve > "$log" 2>&1 &
  pid=$!
  echo "${provider}:${pid}" >> "$PID_FILE"
  echo "  [$provider] PID $pid on port $port (log: $log)"
done

echo "Waiting for instances to be ready..."
for provider in aws gcp onprem; do
  port="${INSTANCES[$provider]}"
  for i in $(seq 1 20); do
    if curl -sf "http://localhost:${port}/api/tags" > /dev/null 2>&1; then
      echo "  [$provider] port $port is up"
      break
    fi
    if [[ $i -eq 20 ]]; then
      echo "  [$provider] WARNING: port $port did not respond after 10s"
    fi
    sleep 0.5
  done
done

echo ""
echo "Demo instances running:"
echo "  AWS    -> http://localhost:11434"
echo "  GCP    -> http://localhost:11435"
echo "  On-Prem -> http://localhost:11436"
echo ""
echo "Model: tinyllama:latest available on all instances."
echo "Stop with: ./scripts/stop_demo.sh"
