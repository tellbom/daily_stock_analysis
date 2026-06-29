#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source ".env"
  set +a
else
  echo "Missing .env. Copy .env.example to .env and fill required settings first." >&2
  exit 1
fi

PYTHON=".venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    python3 -m venv .venv
  elif command -v python >/dev/null 2>&1; then
    python -m venv .venv
  else
    echo "Python is not installed or not available in PATH." >&2
    exit 1
  fi
fi

"$PYTHON" -m pip install -r requirements.txt

exec "$PYTHON" main.py --serve-only "$@"
