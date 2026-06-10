"""Tests for app.config's global sync manager / session cache lifecycle.

Focus: set_sync_manager's marker-on-failure behavior — both the
"stop() returns failure" path (already covered in test_routes_session)
and the "stop() raises" path which silently lost data before B3.
"""

import os


def test_set_sync_manager_writes_marker_when_stop_raises(tmp_path, monkeypatch):
    """Regression (#77): if old.stop() itself raises (not just returns a
    failed FlushResult), the marker must still be written. Pre-B3 the
    outer try/except swallowed the exception and skipped the marker
    step entirely, losing the dirty file list on container recycle."""
    from app.config import set_sync_manager
    from app.services.gcs_sync import GCSSyncManager
    from app.services.unsynced_marker import read_marker

    old_dir = tmp_path / "old_raises"
    new_dir = tmp_path / "new_ok"
    old_dir.mkdir()
    new_dir.mkdir()
    (old_dir / "state.json").write_text("{}")
    (old_dir / "masks.json").write_text("{}")

    # Install a working sync manager with some dirty files
    monkeypatch.setattr(
        "app.services.gcs_sync._get_upload_file",
        lambda: (lambda *a, **kw: None),
    )
    set_sync_manager(None)
    old_sm = GCSSyncManager("bkt", "old_raises", str(old_dir))
    old_sm.mark_dirty("state.json")
    old_sm.mark_dirty("masks.json")
    set_sync_manager(old_sm)

    # Make stop() raise (e.g. GCS client blew up mid-flush).
    # dirty_snapshot() is still callable before we call set_sync_manager
    # so pre_stop_dirty inside the config function captures the files.
    def raising_stop():
        raise RuntimeError("google-cloud-storage client exploded")

    monkeypatch.setattr(old_sm, "stop", raising_stop)

    new_sm = GCSSyncManager("bkt", "new_ok", str(new_dir))
    try:
        set_sync_manager(new_sm)
        marker = read_marker(str(old_dir))
        assert marker is not None, (
            "set_sync_manager must persist a marker when stop() raises"
        )
        assert "state.json" in marker
        assert "masks.json" in marker
    finally:
        # Cleanup: avoid leaking into other tests. stop() of new_sm also
        # succeeds because upload_file is a no-op.
        set_sync_manager(None)


def test_set_sync_manager_no_marker_when_stop_clean(tmp_path, monkeypatch):
    """Happy path: stop() returns ok FlushResult — no marker written."""
    from app.config import set_sync_manager
    from app.services.gcs_sync import GCSSyncManager
    from app.services.unsynced_marker import read_marker, MARKER_FILENAME

    old_dir = tmp_path / "old_clean"
    new_dir = tmp_path / "new_clean"
    old_dir.mkdir()
    new_dir.mkdir()
    (old_dir / "state.json").write_text("{}")

    monkeypatch.setattr(
        "app.services.gcs_sync._get_upload_file",
        lambda: (lambda *a, **kw: None),
    )
    set_sync_manager(None)
    old_sm = GCSSyncManager("bkt", "old_clean", str(old_dir))
    old_sm.mark_dirty("state.json")
    set_sync_manager(old_sm)

    new_sm = GCSSyncManager("bkt", "new_clean", str(new_dir))
    try:
        set_sync_manager(new_sm)
        assert read_marker(str(old_dir)) is None
        assert not (old_dir / MARKER_FILENAME).exists()
    finally:
        set_sync_manager(None)


def test_close_active_session_atomic_swap(tmp_path, monkeypatch):
    """close_active_session clears both sync manager and session cache
    atomically under _globals_lock. Unlike set_sync_manager(None), it
    does NOT call stop() on the old manager — the caller owns the
    stop/marker pipeline."""
    from app.config import (
        set_sync_manager,
        set_session_cache,
        get_sync_manager,
        get_session_cache,
        close_active_session,
    )
    from app.services.gcs_sync import GCSSyncManager
    from app.services.session_cache import SessionCache

    monkeypatch.setattr(
        "app.services.gcs_sync._get_upload_file",
        lambda: (lambda *a, **kw: None),
    )

    sdir = tmp_path / "s"
    sdir.mkdir()
    (sdir / "state.json").write_text("{}")

    set_sync_manager(None)
    sm = GCSSyncManager("bkt", "s", str(sdir))
    set_sync_manager(sm)
    cache = SessionCache(str(sdir))
    set_session_cache(cache)

    # Pre-stop the manager so close_active_session doesn't even attempt it.
    sm.stop()

    returned = close_active_session()
    assert returned is sm, "close_active_session must return the old manager"
    assert get_sync_manager() is None
    assert get_session_cache() is None


def test_close_active_session_does_not_stop_old_manager(tmp_path, monkeypatch):
    """close_active_session must NOT call stop() on the old manager —
    the caller (close_session route) has already handled stop+marker."""
    from app.config import set_sync_manager, set_session_cache, close_active_session
    from app.services.gcs_sync import GCSSyncManager

    monkeypatch.setattr(
        "app.services.gcs_sync._get_upload_file",
        lambda: (lambda *a, **kw: None),
    )

    sdir = tmp_path / "noop"
    sdir.mkdir()

    set_sync_manager(None)
    sm = GCSSyncManager("bkt", "noop", str(sdir))
    stop_calls = {"n": 0}
    original_stop = sm.stop

    def counting_stop():
        stop_calls["n"] += 1
        return original_stop()

    monkeypatch.setattr(sm, "stop", counting_stop)
    set_sync_manager(sm)

    close_active_session()
    assert stop_calls["n"] == 0, \
        "close_active_session must not call stop() on the old manager"


def test_install_active_session_atomic_swap(tmp_path, monkeypatch):
    """B5 (#79): install_active_session swaps both globals under one
    lock. Concurrent get_sync_manager / get_session_cache never see a
    (new_sm, old_cache) or (old_sm, new_cache) intermediate."""
    from app.config import (
        install_active_session,
        get_sync_manager,
        get_session_cache,
        set_sync_manager,
    )
    from app.services.gcs_sync import GCSSyncManager
    from app.services.session_cache import SessionCache

    monkeypatch.setattr(
        "app.services.gcs_sync._get_upload_file",
        lambda: (lambda *a, **kw: None),
    )

    sdir = tmp_path / "new"
    sdir.mkdir()

    set_sync_manager(None)
    new_sm = GCSSyncManager("bkt", "new", str(sdir))
    new_cache = SessionCache(str(sdir))
    install_active_session(new_sm, new_cache)

    assert get_sync_manager() is new_sm
    assert get_session_cache() is new_cache

    set_sync_manager(None)


def test_install_active_session_stops_and_records_old_manager(tmp_path, monkeypatch):
    """The outgoing sync manager is stopped and, if its flush fails,
    an .unsynced marker is written — same guarantees as set_sync_manager."""
    from app.config import install_active_session, set_sync_manager
    from app.services.gcs_sync import GCSSyncManager
    from app.services.unsynced_marker import read_marker

    old_dir = tmp_path / "old_install"
    new_dir = tmp_path / "new_install"
    old_dir.mkdir()
    new_dir.mkdir()
    (old_dir / "state.json").write_text("{}")

    def failing_upload(*a, **kw):
        raise RuntimeError("gcs down")

    monkeypatch.setattr(
        "app.services.gcs_sync._get_upload_file",
        lambda: failing_upload,
    )

    set_sync_manager(None)
    old_sm = GCSSyncManager("bkt", "old_install", str(old_dir))
    old_sm.mark_dirty("state.json")
    set_sync_manager(old_sm)

    new_sm = GCSSyncManager("bkt", "new_install", str(new_dir))
    try:
        install_active_session(new_sm, None)
        marker = read_marker(str(old_dir))
        assert marker is not None
        assert "state.json" in marker
    finally:
        set_sync_manager(None)


def test_clear_active_session_for_resume_stops_old_manager(tmp_path, monkeypatch):
    """clear_active_session_for_resume is a named alias that installs
    (None, None) while triggering the same stop+record path as
    install_active_session."""
    from app.config import (
        clear_active_session_for_resume,
        set_sync_manager,
        set_session_cache,
        get_sync_manager,
        get_session_cache,
    )
    from app.services.gcs_sync import GCSSyncManager
    from app.services.session_cache import SessionCache

    monkeypatch.setattr(
        "app.services.gcs_sync._get_upload_file",
        lambda: (lambda *a, **kw: None),
    )

    sdir = tmp_path / "clear"
    sdir.mkdir()

    set_sync_manager(None)
    sm = GCSSyncManager("bkt", "clear", str(sdir))
    set_sync_manager(sm)
    set_session_cache(SessionCache(str(sdir)))

    clear_active_session_for_resume()

    assert get_sync_manager() is None
    assert get_session_cache() is None
    assert sm._stopped is True


def test_close_active_session_no_op_when_both_already_none(monkeypatch):
    """Calling close_active_session when nothing is installed returns
    None without error."""
    from app.config import set_sync_manager, set_session_cache, close_active_session

    set_sync_manager(None)
    set_session_cache(None)

    assert close_active_session() is None


def test_set_sync_manager_promote_deferred_raise_does_not_skip_marker(
    tmp_path, monkeypatch
):
    """Even if promote_deferred raises (unlikely, but defensive), the
    stop()/marker path must still run. The pre-stop snapshot will just
    miss deferred files — that's the honest limit."""
    from app.config import set_sync_manager
    from app.services.gcs_sync import GCSSyncManager
    from app.services.unsynced_marker import read_marker

    old_dir = tmp_path / "old_promote_raises"
    new_dir = tmp_path / "new_pr"
    old_dir.mkdir()
    new_dir.mkdir()
    (old_dir / "state.json").write_text("{}")

    def failing_upload(*a, **kw):
        raise RuntimeError("gcs down")

    monkeypatch.setattr(
        "app.services.gcs_sync._get_upload_file",
        lambda: failing_upload,
    )
    set_sync_manager(None)
    old_sm = GCSSyncManager("bkt", "old_promote_raises", str(old_dir))
    old_sm.mark_dirty("state.json")
    set_sync_manager(old_sm)

    def raising_promote():
        raise RuntimeError("unexpected bug")

    monkeypatch.setattr(old_sm, "promote_deferred", raising_promote)

    new_sm = GCSSyncManager("bkt", "new_pr", str(new_dir))
    try:
        # Must not raise — set_sync_manager's own try/except absorbs it.
        set_sync_manager(new_sm)
        marker = read_marker(str(old_dir))
        assert marker is not None
        assert "state.json" in marker
    finally:
        set_sync_manager(None)
