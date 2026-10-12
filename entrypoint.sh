#!/usr/bin/env sh
set -eu

PORT="${PORT:-8000}"
# One OpenMP thread per request. The workers and threads below already use the CPUs.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

# HARBOR_BUNDLE picks the bundle folder (default: the California housing demo).
exec gunicorn \
  --workers "${WORKERS:-2}" \
  --threads 4 \
  --bind "0.0.0.0:${PORT}" \
  "peakwhale_harbor.server:create_app()"
