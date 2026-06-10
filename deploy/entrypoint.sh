#!/bin/bash
set -e

# Symlink session data to persistent volume
mkdir -p /data/sessions /data/exports
ln -sfn /data/sessions /app/backend/sessions
ln -sfn /data/exports /app/backend/exports

# Start nginx in background
nginx

# Start gunicorn (foreground) — 1 worker because SAM3 state (GPU model,
# sync manager, session cache) is in-process. gunicorn_config.py enforces
# the single-worker invariant at startup. If someone edits --workers
# below, the config hook will refuse to start the container.
exec gunicorn \
    --config /app/backend/gunicorn_config.py \
    --worker-class gthread \
    --workers 1 \
    --threads 4 \
    --timeout 3600 \
    --bind 0.0.0.0:5555 \
    --chdir /app/backend \
    --capture-output \
    --enable-stdio-inheritance \
    --access-logfile - \
    --error-logfile - \
    "app:create_app()"
