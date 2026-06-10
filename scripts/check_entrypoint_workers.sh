#!/usr/bin/env bash
# Static check: deploy/entrypoint.sh must keep gunicorn at --workers 1.
#
# This service holds SAM3 GPU state, the active GCS sync manager, and the
# session cache as in-process singletons. Running multiple gunicorn workers
# silently diverges those singletons while both workers race on the same
# sessions/ directory on disk.
#
# Runtime enforcement lives in backend/gunicorn_config.py; this script is
# the static/CI-side check that catches the change in review.
#
# Usage: scripts/check_entrypoint_workers.sh
# Exit: 0 if ok, 1 otherwise.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENTRYPOINT="$REPO_ROOT/deploy/entrypoint.sh"

if [[ ! -f "$ENTRYPOINT" ]]; then
  echo "check_entrypoint_workers: $ENTRYPOINT not found" >&2
  exit 1
fi

if ! grep -qE '^\s*--workers\s+1\s*\\?\s*$' "$ENTRYPOINT"; then
  echo "check_entrypoint_workers: FAIL" >&2
  echo "  $ENTRYPOINT must contain a line '--workers 1'." >&2
  echo "  The service cannot safely run >1 gunicorn worker." >&2
  exit 1
fi

if ! grep -qE '^\s*--config\s+/app/backend/gunicorn_config\.py\s*\\?\s*$' "$ENTRYPOINT"; then
  echo "check_entrypoint_workers: FAIL" >&2
  echo "  $ENTRYPOINT must pass --config /app/backend/gunicorn_config.py." >&2
  echo "  That config enforces the single-worker invariant at runtime." >&2
  exit 1
fi

echo "check_entrypoint_workers: ok"
