#!/usr/bin/env bash
# monitor_runpod.sh — tail the RunPod progress log and print cost estimates
#
# Usage:
#   Local (SSH tunnel):  bash scripts/monitor_runpod.sh <log_file_path>
#   On pod directly:     bash scripts/monitor_runpod.sh /root/results/extended_eps/run_progress.log
#
# The script prints:
#   - Live log lines as they arrive  (via tail -f)
#   - Elapsed wall-clock time every 60 s
#   - Estimated RunPod cost (A100 @ $2.19/hr)
#   - Exits automatically when log contains "DONE" or "ERROR"

set -euo pipefail

LOG_FILE="${1:-/root/results/extended_eps/run_progress.log}"
COST_PER_HOUR=2.19
POLL_INTERVAL=60   # seconds between cost updates

if [[ ! -f "$LOG_FILE" ]]; then
  echo "[monitor] Waiting for log file: $LOG_FILE"
  while [[ ! -f "$LOG_FILE" ]]; do sleep 2; done
  echo "[monitor] Log file appeared."
fi

START_TS=$(date +%s)

cost_update() {
  local now
  now=$(date +%s)
  local elapsed=$(( now - START_TS ))
  local hours
  hours=$(echo "scale=4; $elapsed / 3600" | bc)
  local cost
  cost=$(echo "scale=4; $hours * $COST_PER_HOUR" | bc)
  printf "\n[monitor] elapsed=%ds  cost_so_far=\$%.4f / \$3.00 budget\n\n" \
         "$elapsed" "$cost"
}

# Background cost printer
(
  while true; do
    sleep "$POLL_INTERVAL"
    cost_update
  done
) &
COST_PID=$!

# Trap to clean up background process on exit
cleanup() {
  kill "$COST_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[monitor] Streaming $LOG_FILE  (Ctrl-C to stop)"
echo "[monitor] RunPod A100 rate: \$$COST_PER_HOUR/hr  |  Budget: \$3.00"
echo "---"

# Stream log; exit when DONE or ERROR line appears
tail -f "$LOG_FILE" | while IFS= read -r line; do
  echo "$line"
  if echo "$line" | grep -qE "(DONE|FATAL|ERROR)$"; then
    echo ""
    cost_update
    echo "[monitor] Run finished. Killing cost tracker."
    kill "$COST_PID" 2>/dev/null || true
    exit 0
  fi
done
