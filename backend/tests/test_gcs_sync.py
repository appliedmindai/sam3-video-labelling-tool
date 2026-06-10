import os
from unittest.mock import patch, MagicMock

import pytest

from app.services.gcs_sync import GCSSyncManager


@pytest.fixture
def manager(tmp_path):
    return GCSSyncManager("am-seg-org1-test", "session-abc", str(tmp_path))


@pytest.fixture
def mock_upload():
    """Patch _get_upload_file so flush() never imports gcs_storage."""
    upload = MagicMock()
    with patch("app.services.gcs_sync._get_upload_file", return_value=upload):
        yield upload


# ------------------------------------------------------------------
# mark_dirty
# ------------------------------------------------------------------

def test_mark_dirty_adds_to_dirty_set(manager):
    manager.mark_dirty("state.json")
    assert "state.json" in manager._dirty


def test_mark_dirty_defers_masks_during_propagation(manager):
    manager.set_propagating(True)
    manager.mark_dirty("masks.json")
    assert "masks.json" not in manager._dirty
    assert "masks.json" in manager._deferred


def test_mark_dirty_does_not_defer_non_masks_during_propagation(manager):
    manager.set_propagating(True)
    manager.mark_dirty("state.json")
    assert "state.json" in manager._dirty
    assert "state.json" not in manager._deferred


# ------------------------------------------------------------------
# set_propagating
# ------------------------------------------------------------------

def test_set_propagating_false_promotes_deferred(manager):
    manager.set_propagating(True)
    manager.mark_dirty("masks.json")
    assert "masks.json" in manager._deferred

    manager.set_propagating(False)
    assert "masks.json" in manager._dirty
    assert len(manager._deferred) == 0


# ------------------------------------------------------------------
# flush
# ------------------------------------------------------------------

def test_flush_uploads_dirty_files(mock_upload, manager, tmp_path):
    (tmp_path / "state.json").write_text("{}")
    manager.mark_dirty("state.json")

    result = manager.flush()

    assert result.uploaded == 1
    assert result.ok
    mock_upload.assert_called_once_with(
        "am-seg-org1-test", "session-abc", "state.json",
        os.path.join(str(tmp_path), "state.json"),
    )
    assert len(manager._dirty) == 0


def test_flush_retries_on_failure(mock_upload, manager, tmp_path):
    (tmp_path / "state.json").write_text("{}")
    mock_upload.side_effect = RuntimeError("network error")

    manager.mark_dirty("state.json")
    result = manager.flush()

    assert result.uploaded == 0
    assert not result.ok
    assert "state.json" in result.failed
    assert "state.json" in manager._dirty


def test_flush_skips_nonexistent_files(mock_upload, manager):
    manager.mark_dirty("gone.json")
    result = manager.flush()

    assert result.uploaded == 0
    assert result.ok
    mock_upload.assert_not_called()
    assert len(manager._dirty) == 0


def test_flush_noop_when_nothing_dirty(mock_upload, manager):
    result = manager.flush()

    assert result.uploaded == 0
    assert result.ok
    mock_upload.assert_not_called()


def test_flush_uploads_multiple_files(mock_upload, manager, tmp_path):
    (tmp_path / "state.json").write_text("{}")
    (tmp_path / "masks.json").write_text("{}")
    manager.mark_dirty("state.json")
    manager.mark_dirty("masks.json")

    result = manager.flush()

    assert result.uploaded == 2
    assert result.ok
    assert mock_upload.call_count == 2
    assert len(manager._dirty) == 0


def test_flush_partial_failure(mock_upload, manager, tmp_path):
    """When one file fails and another succeeds, only the failed file is retried."""
    (tmp_path / "state.json").write_text("{}")
    (tmp_path / "masks.json").write_text("{}")
    manager.mark_dirty("state.json")
    manager.mark_dirty("masks.json")

    def side_effect(bucket_name, session_id, rel_path, local_path):
        if rel_path == "masks.json":
            raise RuntimeError("network error")

    mock_upload.side_effect = side_effect
    result = manager.flush()

    assert result.uploaded == 1
    assert not result.ok
    assert result.failed == frozenset({"masks.json"})
    assert "masks.json" in manager._dirty
    assert "state.json" not in manager._dirty


# ------------------------------------------------------------------
# FlushResult + flush_with_retry
# ------------------------------------------------------------------

def test_flush_returns_failed_files(tmp_path, monkeypatch):
    """flush() returns FlushResult with per-file failure info."""
    from app.services.gcs_sync import GCSSyncManager, FlushResult

    (tmp_path / "state.json").write_text("{}")
    (tmp_path / "masks.json").write_text("{}")

    manager = GCSSyncManager("test-bucket", "sess", str(tmp_path))
    manager.mark_dirty("state.json")
    manager.mark_dirty("masks.json")

    def failing_upload(bucket, sid, rel_path, local_path):
        if rel_path == "masks.json":
            raise Exception("GCS unreachable")

    monkeypatch.setattr("app.services.gcs_sync._get_upload_file",
                        lambda: failing_upload)
    result = manager.flush()
    assert isinstance(result, FlushResult)
    assert result.uploaded == 1  # state.json succeeded
    assert result.failed == frozenset({"masks.json"})  # masks.json stuck
    assert not result.ok
    # state.json should be removed from dirty; masks.json retained
    assert "state.json" not in manager._dirty
    assert "masks.json" in manager._dirty


def test_flush_with_retry_succeeds_after_transient_failure(tmp_path, monkeypatch):
    """flush_with_retry retries failures; succeeds when GCS recovers."""
    from app.services.gcs_sync import GCSSyncManager

    (tmp_path / "state.json").write_text("{}")
    manager = GCSSyncManager("test-bucket", "sess", str(tmp_path))
    manager.mark_dirty("state.json")

    attempt = {"n": 0}
    def flaky_upload(bucket, sid, rel_path, local_path):
        attempt["n"] += 1
        if attempt["n"] < 2:
            raise Exception("transient failure")
        # second attempt succeeds

    monkeypatch.setattr("app.services.gcs_sync._get_upload_file",
                        lambda: flaky_upload)
    # Use tiny backoff so the test is fast
    result = manager.flush_with_retry(max_attempts=3, initial_backoff_s=0.01, backoff_factor=1.1)
    assert result.ok
    assert result.uploaded == 1
    assert not manager.has_dirty()


def test_flush_with_retry_gives_up_on_persistent_failure(tmp_path, monkeypatch):
    """flush_with_retry returns failed files after max_attempts."""
    from app.services.gcs_sync import GCSSyncManager

    (tmp_path / "state.json").write_text("{}")
    manager = GCSSyncManager("test-bucket", "sess", str(tmp_path))
    manager.mark_dirty("state.json")

    call_count = {"n": 0}
    def always_fails(bucket, sid, rel_path, local_path):
        call_count["n"] += 1
        raise Exception("persistent failure")

    monkeypatch.setattr("app.services.gcs_sync._get_upload_file",
                        lambda: always_fails)
    result = manager.flush_with_retry(max_attempts=3, initial_backoff_s=0.01, backoff_factor=1.1)
    assert not result.ok
    assert result.failed == frozenset({"state.json"})
    assert result.uploaded == 0
    assert call_count["n"] == 3  # exactly max_attempts tries
    # File stays dirty for future retries
    assert manager.has_dirty()


# ------------------------------------------------------------------
# stop() prevents re-arming the timer (R3)
# ------------------------------------------------------------------

def test_stop_prevents_reschedule(tmp_path, monkeypatch):
    """stop() must prevent _tick from re-arming the timer, even if _tick
    was already in flight when stop() ran."""
    from app.services.gcs_sync import GCSSyncManager
    import time

    (tmp_path / "state.json").write_text("{}")
    manager = GCSSyncManager("test-bucket", "sess", str(tmp_path))
    monkeypatch.setattr("app.services.gcs_sync._get_upload_file",
                        lambda: (lambda *a, **kw: None))
    manager.start(interval=0.01)

    # Let a few ticks run
    time.sleep(0.05)

    manager.stop()

    # Even if a tick fires after stop(), _schedule must refuse to re-arm
    time.sleep(0.1)

    # Confirm: no live timer on the manager
    with manager._lock:
        assert manager._timer is None
        assert manager._stopped is True


def test_start_after_stop_raises(tmp_path):
    """A stopped manager cannot be restarted (misuse guard)."""
    from app.services.gcs_sync import GCSSyncManager

    (tmp_path / "state.json").write_text("{}")
    manager = GCSSyncManager("test-bucket", "sess", str(tmp_path))
    manager.stop()
    with pytest.raises(RuntimeError):
        manager.start()


# ------------------------------------------------------------------
# G1: stop() promotes deferred before flushing (issue #57)
# ------------------------------------------------------------------

def test_stop_promotes_deferred_before_flushing(mock_upload, manager, tmp_path):
    """Regression: SIGTERM during propagation must not lose deferred masks.

    Before the fix, GCSSyncManager.stop() called flush() which only iterated
    _dirty. If the propagation thread was killed before running its finally
    block (set_propagating(False)), masks.json stayed in _deferred forever
    and the container died with data loss.
    """
    (tmp_path / "masks.json").write_text("{}")
    manager.set_propagating(True)
    manager.mark_dirty("masks.json")
    # Propagation is still flagged active; masks.json is deferred
    assert "masks.json" in manager._deferred
    assert "masks.json" not in manager._dirty

    # Forced teardown (SIGTERM / container recycle) — stop must promote
    # deferred so the final flush actually uploads masks.json.
    result = manager.stop()
    assert result.ok, "stop() must upload deferred files, not drop them"
    mock_upload.assert_called_once()
    args = mock_upload.call_args[0]
    assert args[2] == "masks.json"


def test_stop_is_idempotent_and_clears_deferred(mock_upload, manager, tmp_path):
    """Calling stop twice must not re-upload or re-promote."""
    (tmp_path / "masks.json").write_text("{}")
    manager.set_propagating(True)
    manager.mark_dirty("masks.json")

    first = manager.stop()
    second = manager.stop()
    assert first.uploaded == 1
    assert second.uploaded == 0
    assert len(manager._deferred) == 0
    assert len(manager._dirty) == 0


def test_promote_deferred_without_stopping(manager, tmp_path):
    """promote_deferred is callable before stop so SIGTERM handler can
    flush_with_retry and then surface a marker for persistent failures."""
    (tmp_path / "masks.json").write_text("{}")
    manager.set_propagating(True)
    manager.mark_dirty("masks.json")
    assert "masks.json" in manager._deferred

    manager.promote_deferred()
    assert "masks.json" in manager._dirty
    assert len(manager._deferred) == 0
    # Propagating flag is also cleared so subsequent mark_dirty doesn't
    # route new writes back into deferred.
    manager.mark_dirty("masks.json")  # would defer if still propagating
    assert "masks.json" in manager._dirty


def test_promote_deferred_is_safe_when_nothing_deferred(manager):
    """No-op when deferred is empty."""
    manager.promote_deferred()
    assert len(manager._dirty) == 0
    assert len(manager._deferred) == 0


# ------------------------------------------------------------------
# G2: stop()'s final flush result must not be silently dropped (issue #58)
# ------------------------------------------------------------------

def test_stop_returns_failure_when_uploads_fail(mock_upload, manager, tmp_path):
    """stop() returns a FlushResult — callers must not ignore failures."""
    (tmp_path / "state.json").write_text("{}")
    manager.mark_dirty("state.json")
    mock_upload.side_effect = RuntimeError("GCS unavailable")

    result = manager.stop()
    assert not result.ok
    assert "state.json" in result.failed
    # Manager is stopped, but the failed file is preserved in _dirty so
    # a caller inspecting it can decide to write a marker.
    assert "state.json" in manager._dirty


def test_stop_picks_up_mark_dirty_between_operations(mock_upload, manager, tmp_path):
    """Simulates: flush_with_retry succeeds, then a late request writes
    state.json (e.g. bboxPadding debounce), then stop() is called. The
    late file must still be uploaded (or at minimum returned in the
    FlushResult so the caller can persist an unsynced marker)."""
    (tmp_path / "state.json").write_text("{}")

    # First pass — no-op flush (nothing dirty)
    result1 = manager.flush_with_retry(max_attempts=1)
    assert result1.ok
    assert result1.uploaded == 0

    # Late write sneaks in between flush_with_retry and stop()
    manager.mark_dirty("state.json")

    # stop's internal flush must pick it up
    stop_result = manager.stop()
    assert stop_result.ok
    assert stop_result.uploaded == 1


# ------------------------------------------------------------------
# R31 / B4: mark_dirty on stopped manager raises, mark_dirty_safe retries
# ------------------------------------------------------------------

def test_mark_dirty_raises_on_stopped_manager(tmp_path, monkeypatch):
    """R31 regression: a stopped manager silently accepted mark_dirty,
    adding to a _dirty set that would never flush. Now it raises so the
    caller can route the write to a freshly-installed manager."""
    monkeypatch.setattr("app.services.gcs_sync._get_upload_file",
                        lambda: (lambda *a, **kw: None))
    manager = GCSSyncManager("bkt", "sess", str(tmp_path))
    manager.stop()

    with pytest.raises(RuntimeError, match="stopped GCSSyncManager"):
        manager.mark_dirty("state.json")


def test_mark_dirty_safe_drops_when_no_manager(tmp_path, monkeypatch):
    """mark_dirty_safe with no active manager is a silent no-op.
    The session is being torn down and the file is still on local disk;
    the next resume will reconcile from disk."""
    from app.services.gcs_sync import mark_dirty_safe
    from app.config import set_sync_manager

    set_sync_manager(None)
    # Must not raise
    mark_dirty_safe("state.json")


def test_mark_dirty_safe_retries_after_stopped(tmp_path, monkeypatch):
    """If the first get_sync_manager() returns a stopped manager (race
    with close_session), mark_dirty_safe catches RuntimeError and retries.
    If the second lookup returns None, it returns silently."""
    from app.services.gcs_sync import mark_dirty_safe
    from app.config import set_sync_manager

    monkeypatch.setattr("app.services.gcs_sync._get_upload_file",
                        lambda: (lambda *a, **kw: None))

    manager = GCSSyncManager("bkt", "sess", str(tmp_path))
    set_sync_manager(None)  # start from clean slate

    # Install the manager then stop it directly (simulates close_session's
    # sm.stop() happening between a request's get_sync_manager() and its
    # mark_dirty call — except mark_dirty_safe re-reads get_sync_manager
    # per attempt so we install a stopped manager for the test).
    set_sync_manager(manager)
    manager._stopped = True  # directly — stop() would also promote/flush

    # After first attempt raises, mark_dirty_safe should re-call
    # get_sync_manager. We swap to None mid-test via a side-effect.
    call_count = {"n": 0}
    original_get = None

    def fake_get():
        call_count["n"] += 1
        if call_count["n"] == 1:
            return manager  # stopped — will raise
        return None  # retry sees no manager — no-op

    import app.services.gcs_sync as gcs_sync_mod
    monkeypatch.setattr(gcs_sync_mod, "get_sync_manager", fake_get, raising=False)
    # mark_dirty_safe imports get_sync_manager inline via `from app.config`.
    # Patch that too.
    import app.config as config_mod
    monkeypatch.setattr(config_mod, "get_sync_manager", fake_get)

    # Must not raise even though first lookup returned a stopped manager
    mark_dirty_safe("state.json")
    assert call_count["n"] == 2  # tried twice, gave up

    set_sync_manager(None)


def test_mark_dirty_safe_routes_to_fresh_manager_after_stop(tmp_path, monkeypatch):
    """When a request thread captures a stale manager before the global
    is swapped, mark_dirty_safe's retry picks up the new manager so the
    write is not dropped."""
    from app.services.gcs_sync import mark_dirty_safe
    from app.config import set_sync_manager

    monkeypatch.setattr("app.services.gcs_sync._get_upload_file",
                        lambda: (lambda *a, **kw: None))

    set_sync_manager(None)
    old_mgr = GCSSyncManager("bkt", "old", str(tmp_path))
    new_mgr = GCSSyncManager("bkt", "new", str(tmp_path))
    old_mgr._stopped = True  # simulate already-stopped

    call_count = {"n": 0}

    def fake_get():
        call_count["n"] += 1
        return old_mgr if call_count["n"] == 1 else new_mgr

    import app.config as config_mod
    monkeypatch.setattr(config_mod, "get_sync_manager", fake_get)

    mark_dirty_safe("state.json")
    # new_mgr received the write instead of being dropped
    assert "state.json" in new_mgr._dirty
    assert call_count["n"] == 2

    set_sync_manager(None)
