#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/colab_run.sh -f /abs/path/to/script.py [--accelerator T4|L4] [--timeout 1200]
  scripts/colab_run.sh -c "print('hello')" [--accelerator T4|L4] [--timeout 1200]

Defaults:
  accelerator: T4
  timeout: 1200

Notes:
  - Uses Colab CLI fallback path (`colab-exec`)
  - Intended for reproducible Codex runs when MCP tools are not available in-session
EOF
}

if ! command -v colab-exec >/dev/null 2>&1; then
  echo "ERROR: colab-exec not found in PATH" >&2
  exit 1
fi

ACCEL="T4"
TIMEOUT="1200"
MODE=""
PAYLOAD=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -f)
      MODE="file"
      PAYLOAD="${2:-}"
      shift 2
      ;;
    -c)
      MODE="code"
      PAYLOAD="${2:-}"
      shift 2
      ;;
    --accelerator)
      ACCEL="${2:-T4}"
      shift 2
      ;;
    --timeout)
      TIMEOUT="${2:-1200}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown arg: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$MODE" || -z "$PAYLOAD" ]]; then
  usage
  exit 2
fi

# Preflight: fail fast before heavy execution.
colab-exec --accelerator "$ACCEL" --timeout 120 "import torch; print('cuda_available', torch.cuda.is_available())" >/dev/null

if [[ "$MODE" == "file" ]]; then
  if [[ ! -f "$PAYLOAD" ]]; then
    echo "ERROR: script not found: $PAYLOAD" >&2
    exit 1
  fi
  colab-exec --accelerator "$ACCEL" --timeout "$TIMEOUT" -f "$PAYLOAD"
else
  colab-exec --accelerator "$ACCEL" --timeout "$TIMEOUT" "$PAYLOAD"
fi
