"""Gunicorn config that enforces the single-worker invariant.

This service holds all authoritative state in a single Python process:
SAM3 GPU model, _active_sync_manager, _active_session_cache, per-session
session_io_locks, ServiceState. Running more than one gunicorn worker
forks the process and silently diverges those in-memory singletons while
both workers race on the same sessions/ directory on disk.

The hooks below fail-fast at container startup so that flipping
`--workers N` in deploy/entrypoint.sh (or setting GUNICORN_CMD_ARGS /
WEB_CONCURRENCY) produces a loud crash instead of silent corruption.
"""
from __future__ import annotations

import logging
import sys
from typing import Any


_SINGLE_WORKER_REASON = (
    "This service holds SAM3 GPU state, sync managers, and session caches "
    "as in-process singletons. Running >1 worker silently diverges them."
)


def _assert_single_worker(workers: int, source: str) -> None:
    logger = logging.getLogger("gunicorn.error")
    if workers == 1:
        logger.info("startup | single-worker invariant ok (source=%s)", source)
        return
    logger.error(
        "startup | refusing to start: gunicorn workers=%d but must be 1 "
        "(source=%s). %s",
        workers, source, _SINGLE_WORKER_REASON,
    )
    sys.exit(1)


def when_ready(server: Any) -> None:
    """Arbiter-level hook — runs once before any worker is forked."""
    _assert_single_worker(server.cfg.workers, source="arbiter.when_ready")


def post_worker_init(worker: Any) -> None:
    """Worker-level hook — fallback if someone strips the --config flag.

    Each worker sees cfg.workers from the arbiter it was forked from,
    so this would catch a mis-configured entrypoint even if when_ready
    were bypassed.
    """
    _assert_single_worker(worker.cfg.workers, source="worker.post_worker_init")
