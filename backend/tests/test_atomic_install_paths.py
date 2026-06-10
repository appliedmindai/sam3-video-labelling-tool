"""Tests for C2 review fix: open_session + ExtractFramesStep must install
(sync_manager, session_cache) atomically instead of calling set_sync_manager
and set_session_cache separately.

Regression: the pre-fix two-call pattern left a race window where a
concurrent reader could observe `(sm_new, cache_old)`. mark_dirty_safe
would route writes to sm_new while reads still served cache_old — the
exact (sm, cache) tearing bug #79 / R31 was supposed to close.
"""

import json
import os
import threading
from unittest.mock import patch

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Flask test client with auth bypassed and SESSIONS_DIR pointed at
    tmp_path. SEGMENT_MODE is set to "cloud" so the cloud branches of
    the routes execute."""
    monkeypatch.setenv("SEGMENT_MODE", "")  # decorator bypass
    from app import create_app
    from app.config import (
        close_active_session,
        set_session_cache,
        set_sync_manager,
    )

    set_sync_manager(None)
    set_session_cache(None)

    import app.routes.session as session_mod
    import app.routes.segment as segment_mod
    monkeypatch.setattr(session_mod, "SESSIONS_DIR", str(tmp_path))
    monkeypatch.setattr(segment_mod, "SESSIONS_DIR", str(tmp_path))
    monkeypatch.setattr(session_mod, "SEGMENT_MODE", "cloud")

    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c, str(tmp_path)

    close_active_session()


def test_open_session_installs_sm_and_cache_atomically(tmp_path, monkeypatch):
    """open_session must NOT call set_sync_manager then set_session_cache
    in sequence — a concurrent reader could catch the (sm_new, cache_old)
    intermediate state. This test asserts install_active_session is the
    single entry point that transitions both globals.

    We call the view function inside an app/request context with
    g.bucket stubbed.
    """
    from app import create_app
    import app.routes.session as session_mod
    from app.services.session_cache import SessionCache

    sessions_dir = str(tmp_path)
    sid = "open-atomic"
    sdir = os.path.join(sessions_dir, sid)
    os.makedirs(sdir, exist_ok=True)
    with open(os.path.join(sdir, "meta.json"), "w") as f:
        json.dump({"original_name": "x.mov"}, f)
    with open(os.path.join(sdir, "state.json"), "w") as f:
        f.write("{}")

    monkeypatch.setattr(session_mod, "SESSIONS_DIR", sessions_dir)
    monkeypatch.setattr(session_mod, "SEGMENT_MODE", "cloud")

    # Fake GCSSyncManager
    class FakeSM:
        def __init__(self, *a, **kw):
            self.started = False
        def start(self, *a, **kw):
            self.started = True
        def stop(self):
            from app.services.gcs_sync import FlushResult
            return FlushResult(uploaded=0, failed=frozenset())
        def promote_deferred(self):
            pass
        def dirty_snapshot(self):
            return frozenset()

    import app.services.gcs_sync as gcs_sync_mod
    monkeypatch.setattr(gcs_sync_mod, "GCSSyncManager", FakeSM)

    install_calls: list[tuple] = []
    set_sm_calls: list[object] = []
    set_cache_calls: list[object] = []

    def fake_install(sm, cache):
        install_calls.append((sm, cache))
    def fake_set_sm(sm):
        set_sm_calls.append(sm)
    def fake_set_cache(cache):
        set_cache_calls.append(cache)

    # The route imports these three names lazily from app.config inside
    # the function body. Patch them on both config and on session_mod
    # (in case a top-level import already bound them).
    from app import config as cfg
    monkeypatch.setattr(cfg, "install_active_session", fake_install)
    monkeypatch.setattr(cfg, "set_sync_manager", fake_set_sm)
    monkeypatch.setattr(cfg, "set_session_cache", fake_set_cache)

    monkeypatch.setenv("SEGMENT_MODE", "")
    app = create_app()
    app.config["TESTING"] = True

    with app.test_request_context(f"/api/session/open/{sid}", method="POST"):
        from flask import g
        g.bucket = "fake-bucket"
        res = session_mod.open_session(sid)

    assert len(install_calls) == 1, (
        f"install_active_session must be the single entry point; "
        f"got install={len(install_calls)}, set_sm={len(set_sm_calls)}, "
        f"set_cache={len(set_cache_calls)}"
    )
    sm_arg, cache_arg = install_calls[0]
    assert sm_arg is not None
    assert isinstance(cache_arg, SessionCache)
    assert set_sm_calls == [], (
        "set_sync_manager was called — atomic-install refactor regressed"
    )
    assert set_cache_calls == [], (
        "set_session_cache was called — atomic-install refactor regressed"
    )


def test_extract_frames_step_installs_sm_and_cache_atomically(tmp_path, monkeypatch):
    """ExtractFramesStep.run in cloud mode must install (sm, cache)
    atomically via install_active_session — not set_sync_manager alone
    (which was the pre-fix behavior and a pre-existing #79-shaped hole).
    """
    from app.services.pipeline import ExtractFramesStep

    # Build a real test video so extract_frames_async returns > 0.
    # Piggy-back on the pattern from test_pipeline.py.
    import subprocess
    session_id = "extract-atomic"
    sessions_root = tmp_path / "sessions"
    sessions_root.mkdir()
    sdir = sessions_root / session_id
    sdir.mkdir()
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i",
        "testsrc=duration=1:size=160x120:rate=10",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(sdir / "video.mp4"),
    ], capture_output=True, check=True)

    monkeypatch.setattr("app.config.SESSIONS_DIR", str(sessions_root))

    # Stub GCS uploads to no-ops
    import app.services.gcs_storage as gcs_storage
    monkeypatch.setattr(gcs_storage, "upload_session_files",
                        lambda *a, **kw: None)
    monkeypatch.setattr(gcs_storage, "upload_directory",
                        lambda *a, **kw: 0)

    # Stub GCSSyncManager so no timer starts
    class FakeSM:
        def __init__(self, *a, **kw): pass
        def start(self, *a, **kw): pass
        def stop(self):
            from app.services.gcs_sync import FlushResult
            return FlushResult(uploaded=0, failed=frozenset())
        def promote_deferred(self): pass
        def dirty_snapshot(self): return frozenset()

    import app.services.gcs_sync as gcs_sync_mod
    monkeypatch.setattr(gcs_sync_mod, "GCSSyncManager", FakeSM)

    install_calls: list[tuple] = []
    set_sm_calls: list[object] = []

    def fake_install(sm, cache):
        install_calls.append((sm, cache))

    def fake_set_sm(sm):
        set_sm_calls.append(sm)

    from app import config as cfg
    monkeypatch.setattr(cfg, "install_active_session", fake_install)
    monkeypatch.setattr(cfg, "set_sync_manager", fake_set_sm)

    step = ExtractFramesStep(session_id, fps=2, bucket="bkt")
    result = step.run(on_progress=lambda p: None,
                      cancel_event=threading.Event())
    assert result is not None
    assert result["frame_count"] > 0

    # install_active_session called once with a non-None cache
    assert len(install_calls) == 1, (
        f"ExtractFramesStep must use install_active_session; "
        f"install_calls={len(install_calls)} set_sm_calls={len(set_sm_calls)}"
    )
    sm_arg, cache_arg = install_calls[0]
    assert sm_arg is not None
    assert cache_arg is not None, (
        "pre-fix bug: upload flow never installed a SessionCache"
    )
    assert set_sm_calls == [], (
        "set_sync_manager was called — upload flow regressed to non-atomic install"
    )
