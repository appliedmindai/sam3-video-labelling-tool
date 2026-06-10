"""Tests for the gunicorn single-worker startup guard.

Regression: nothing in the Python process used to assert that gunicorn
was actually running with --workers 1. If deploy/entrypoint.sh drifted
to --workers 2, two processes would race on sessions/ with their own
in-process singletons and silently corrupt data.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import gunicorn_config


def _fake_server(workers: int):
    return SimpleNamespace(cfg=SimpleNamespace(workers=workers))


def _fake_worker(workers: int):
    return SimpleNamespace(cfg=SimpleNamespace(workers=workers))


def test_when_ready_passes_with_one_worker():
    # Should not raise, should not call sys.exit.
    gunicorn_config.when_ready(_fake_server(1))


def test_when_ready_exits_with_two_workers():
    with pytest.raises(SystemExit) as excinfo:
        gunicorn_config.when_ready(_fake_server(2))
    assert excinfo.value.code == 1


def test_when_ready_exits_with_zero_workers():
    # Defensive: zero is also not valid.
    with pytest.raises(SystemExit) as excinfo:
        gunicorn_config.when_ready(_fake_server(0))
    assert excinfo.value.code == 1


def test_post_worker_init_passes_with_one_worker():
    gunicorn_config.post_worker_init(_fake_worker(1))


def test_post_worker_init_exits_with_multiple_workers():
    with pytest.raises(SystemExit) as excinfo:
        gunicorn_config.post_worker_init(_fake_worker(4))
    assert excinfo.value.code == 1
