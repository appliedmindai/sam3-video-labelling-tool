"""Tests for the SIGTERM shutdown path (C1 review fix).

Regression: after sm.stop() returns, the SIGTERM handler used to leave
_active_sync_manager pointing at the stopped manager. Any request that
ran during Cloud Run's pre-SIGKILL grace window (up to ~10s) would see
the stopped manager, try mark_dirty, hit RuntimeError("stopped") and
drop the write after the single retry (which re-read the same stopped
manager).
"""

from app import _sigterm_flush_and_clear_globals
from app.config import (
    close_active_session,
    get_session_cache,
    get_sync_manager,
    set_session_cache,
    set_sync_manager,
)
from app.services.gcs_sync import GCSSyncManager
from app.services.session_cache import SessionCache


def _noop_upload_file(monkeypatch):
    monkeypatch.setattr(
        "app.services.gcs_sync._get_upload_file",
        lambda: (lambda *a, **kw: None),
    )


def test_sigterm_clears_globals_on_clean_flush(tmp_path, monkeypatch):
    """After _sigterm_flush_and_clear_globals, both globals are None and
    the previously-active manager is stopped."""
    _noop_upload_file(monkeypatch)
    close_active_session()  # ensure clean slate

    sdir = tmp_path / "s"
    sdir.mkdir()
    (sdir / "state.json").write_text("{}")

    sm = GCSSyncManager("bkt", "s", str(sdir))
    sm.mark_dirty("state.json")
    set_sync_manager(sm)
    set_session_cache(SessionCache(str(sdir)))

    assert get_sync_manager() is sm
    assert get_session_cache() is not None

    _sigterm_flush_and_clear_globals()

    assert get_sync_manager() is None, (
        "SIGTERM handler must clear _active_sync_manager after stop()"
    )
    assert get_session_cache() is None, (
        "SIGTERM handler must also clear _active_session_cache"
    )
    assert sm._stopped is True


def test_sigterm_clears_globals_even_when_flush_fails(tmp_path, monkeypatch):
    """Data-loss window: flush fails → marker is written → globals still
    cleared. A leftover stopped manager in the global slot would cause
    mark_dirty_safe to silently drop writes from in-flight requests."""
    from app.services.unsynced_marker import read_marker

    def failing_upload(*a, **kw):
        raise RuntimeError("gcs down")

    monkeypatch.setattr(
        "app.services.gcs_sync._get_upload_file", lambda: failing_upload,
    )
    close_active_session()

    sdir = tmp_path / "s_fail"
    sdir.mkdir()
    (sdir / "state.json").write_text("{}")

    sm = GCSSyncManager("bkt", "s_fail", str(sdir))
    sm.mark_dirty("state.json")
    set_sync_manager(sm)
    set_session_cache(SessionCache(str(sdir)))

    _sigterm_flush_and_clear_globals()

    assert get_sync_manager() is None
    assert get_session_cache() is None
    # Marker should exist (files failed to flush)
    marker = read_marker(str(sdir))
    assert marker is not None
    assert "state.json" in marker


def test_sigterm_noop_when_no_active_manager(monkeypatch):
    """With no active sync manager, the handler is a no-op."""
    close_active_session()
    # Should not raise and should leave globals at None
    _sigterm_flush_and_clear_globals()
    assert get_sync_manager() is None
    assert get_session_cache() is None


def test_sigterm_stop_failure_writes_marker(tmp_path, monkeypatch):
    """Regression (#97): mark_dirty arriving between flush_with_retry and
    stop() must not disappear. If stop()'s internal flush fails during
    SIGTERM, the handler must write an .unsynced marker for the delta —
    mirror of the close_session path at routes/segment.py:321-332.
    """
    from app.services.unsynced_marker import read_marker
    from app.services.gcs_sync import FlushResult

    close_active_session()

    sdir = tmp_path / "s_stop_fail"
    sdir.mkdir()
    (sdir / "state.json").write_text("{}")

    sm = GCSSyncManager("bkt", "s_stop_fail", str(sdir))
    set_sync_manager(sm)
    set_session_cache(SessionCache(str(sdir)))

    # Stub: flush_with_retry returns ok (clean) but also marks a file dirty,
    # simulating a concurrent worker thread that marks state.json between
    # flush_with_retry returning and stop() acquiring _lock. stop()'s
    # internal flush will then try to upload that file.
    def clean_retry_then_mark(*a, **kw):
        sm.mark_dirty("state.json")
        return FlushResult(uploaded=0, failed=frozenset())
    monkeypatch.setattr(sm, "flush_with_retry", clean_retry_then_mark)

    # stop()'s internal flush fails on the freshly-marked file.
    def failing_upload(*a, **kw):
        raise RuntimeError("gcs down during sigterm")
    monkeypatch.setattr(
        "app.services.gcs_sync._get_upload_file", lambda: failing_upload,
    )

    _sigterm_flush_and_clear_globals()

    # Globals cleared even though stop() failed.
    assert get_sync_manager() is None
    assert get_session_cache() is None
    # Marker written with the late file — the whole point of #97.
    marker = read_marker(str(sdir))
    assert marker is not None
    assert "state.json" in marker
