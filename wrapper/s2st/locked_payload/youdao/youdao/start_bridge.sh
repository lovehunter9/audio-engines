#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source ./.env
  set +a
fi

exec uvicorn source_bridge:app --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
