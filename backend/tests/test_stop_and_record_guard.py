"""Tests for H4 review fix: _stop_and_record must be a no-op when the
outgoing manager is already stopped.

Regression: if the close-path (close_session route or SIGTERM handler)
already called `old.stop()` and wrote the .unsynced marker, a later
call to `set_sync_manager(new)` would call `old.stop()` AGAIN —
re-triggering a dead GCS flush, surfacing spurious warnings, and
risking a stale dirty-set snapshot overwriting the authoritative
marker written by the close-path.
"""

import os

from app.config import _stop_and_record, set_sync_manager
from app.services.gcs_sync import GCSSyncManager


def _noop_upload(monkeypatch):
    monkeypatch.setattr(
        "app.services.gcs_sync._get_upload_file",
        lambda: (lambda *a, **kw: None),
    )


def test_stop_and_record_early_returns_on_stopped_manager(tmp_path, monkeypatch):
    """If _stopped is True, _stop_and_record must NOT call stop() again
    and must NOT write a marker (the close-path already handled that)."""
    _noop_upload(monkeypatch)
    set_sync_manager(None)

    sdir = tmp_path / "stopped"
    sdir.mkdir()
    (sdir / "state.json").write_text("{}")

    sm = GCSSyncManager("bkt", "stopped", str(sdir))
    sm.mark_dirty("state.json")
    # Simulate a close-path that stopped the manager successfully
    sm.stop()
    assert sm._stopped is True

    # Spy on stop() — it must NOT be called again
    stop_call_count = {"n": 0}
    original_stop = sm.stop

    def counting_stop():
        stop_call_count["n"] += 1
        return original_stop()

    monkeypatch.setattr(sm, "stop", counting_stop)

    _stop_and_record(sm)
    assert stop_call_count["n"] == 0, (
        "_stop_and_record must not call stop() on an already-stopped manager"
    )


def test_stop_and_record_early_returns_skips_dirty_snapshot(
    tmp_path, monkeypatch,
):
    """promote_deferred and dirty_snapshot are also skipped — the
    authoritative dirty list is whatever the close-path already
    persisted in the marker. Calling snapshot on a stopped manager
    would stale-copy the dirty set and risk overwriting the
    close-path's marker via the failed-stop fallback."""
    _noop_upload(monkeypatch)
    set_sync_manager(None)

    sdir = tmp_path / "snapshot_skipped"
    sdir.mkdir()
    (sdir / "state.json").write_text("{}")

    sm = GCSSyncManager("bkt", "snapshot_skipped", str(sdir))
    sm.stop()

    snapshot_calls = {"n": 0}
    promote_calls = {"n": 0}
    original_snapshot = sm.dirty_snapshot
    original_promote = sm.promote_deferred

    def counting_snapshot():
        snapshot_calls["n"] += 1
        return original_snapshot()

    def counting_promote():
        promote_calls["n"] += 1
        return original_promote()

    monkeypatch.setattr(sm, "dirty_snapshot", counting_snapshot)
    monkeypatch.setattr(sm, "promote_deferred", counting_promote)

    _stop_and_record(sm)

    assert snapshot_calls["n"] == 0
    assert promote_calls["n"] == 0


def test_set_sync_manager_after_external_stop_is_clean(tmp_path, monkeypatch):
    """End-to-end: close-path already ran stop() + persisted marker.
    A subsequent set_sync_manager(new) must NOT run flush against the
    dead manager, must NOT overwrite the close-path's marker, and
    must install the new manager cleanly."""
    from app.services.unsynced_marker import persist_unsynced, read_marker

    # Make upload fail so the close-path writes a marker with the REAL
    # dirty set.
    def failing_upload(*a, **kw):
        raise RuntimeError("gcs transient")

    monkeypatch.setattr(
        "app.services.gcs_sync._get_upload_file", lambda: failing_upload,
    )
    set_sync_manager(None)

    sdir = tmp_path / "seq"
    sdir.mkdir()
    (sdir / "state.json").write_text("{}")
    (sdir / "masks.json").write_text("{}")

    sm = GCSSyncManager("bkt", "seq", str(sdir))
    sm.mark_dirty("state.json")
    sm.mark_dirty("masks.json")
    set_sync_manager(sm)

    # Close-path: stop() raises because upload is failing. Persist marker.
    try:
        sm.stop()
    except Exception:
        pass  # we're just simulating close-path
    persist_unsynced(
        None, "seq", str(sdir),
        ["state.json", "masks.json"],
        reason="close_flush_failed",
    )
    close_marker = read_marker(str(sdir))
    assert close_marker is not None
    assert set(close_marker) == {"state.json", "masks.json"}

    # Now a concurrent request that saw get_sync_manager() == sm before
    # the close-path took effect calls set_sync_manager(None) or (new).
    # _stop_and_record MUST early-return without re-running stop/flush.
    stop_calls = {"n": 0}
    original_stop = sm.stop

    def spy_stop():
        stop_calls["n"] += 1
        return original_stop()

    monkeypatch.setattr(sm, "stop", spy_stop)

    new_sm = GCSSyncManager("bkt", "next", str(tmp_path / "next_dir"))
    (tmp_path / "next_dir").mkdir()
    try:
        set_sync_manager(new_sm)
    finally:
        set_sync_manager(None)

    assert stop_calls["n"] == 0, (
        "set_sync_manager(new) must not re-stop the already-stopped old manager"
    )
    # Marker still intact — close-path's authoritative list survived
    assert set(read_marker(str(sdir)) or []) == {"state.json", "masks.json"}
