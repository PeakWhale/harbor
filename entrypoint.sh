#!/usr/bin/env sh
set -eu

PORT="${PORT:-8000}"

# HARBOR_BUNDLE picks the bundle folder (default: the California housing demo).
exec gunicorn \
  --workers "${WORKERS:-2}" \
  --threads 4 \
  --bind "0.0.0.0:${PORT}" \
  "peakwhale_harbor.server:create_app()"
