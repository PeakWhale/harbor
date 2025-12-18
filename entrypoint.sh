#!/usr/bin/env sh
set -eu

PORT="${PORT:-8000}"

exec gunicorn \
  --workers 4 \
  --bind "0.0.0.0:${PORT}" \
  app:app

