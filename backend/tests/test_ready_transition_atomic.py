"""Tests for H5 review fix: the on_complete → ready transition must be
atomic under _state_lock so a concurrent close_session can never
observe `phase=ready` while globals are `(None, None)`.

Regression: pre-fix, the pipeline thread released _state_lock after
the final generation check, ran on_complete (which installed globals
via install_active_session under _globals_lock), then re-acquired
_state_lock to set phase=ready. Between on_complete returning and the
ready transition, a concurrent close_session could call
close_active_session() — clearing globals to (None, None) — while
phase was still "initializing". Pipeline then set phase=ready on a
torn-down service, and the frontend observed a "ready" session with no
sync manager (mark_dirty_safe silently drops all subsequent writes).
"""

import threading
import time

import pytest

from app.services.pipeline import ServiceState


class _ControllableStep:
    """A pipeline step that signals when its progress callback fires and
    blocks there until the test releases it."""

    def __init__(self, phase: str):
        self.phase = phase

    def run(self, on_progress, cancel_event):
        # Zero-cost step — the interesting timing is in on_complete.
        on_progress(1.0)
        return None


def _reset_sam(sam):
    with sam._state_lock:
        sam._service_state = ServiceState()
    sam._pipeline_cancel.clear()
    if sam._pipeline_thread is not None and sam._pipeline_thread.is_alive():
        sam._pipeline_cancel.set()
        sam._pipeline_thread.join(timeout=5)
    sam._pipeline_thread = None


@pytest.fixture
def sam():
    from app.services.sam3_service import SAM3Service
    s = SAM3Service()
    _reset_sam(s)
    yield s
    _reset_sam(s)


def test_ready_and_globals_install_observed_atomically(sam, monkeypatch):
    """From an external thread's perspective, it's never possible to see
    `phase=ready` while on_complete has not yet installed its globals.
    The fix holds _state_lock across on_complete + ready transition."""
    observed_phases_when_in_on_complete: list[str] = []
    on_complete_entered = threading.Event()
    on_complete_release = threading.Event()

    def slow_on_complete():
        on_complete_entered.set()
        # Block inside on_complete so a probe thread can observe state.
        on_complete_release.wait(timeout=5)

    # Kick off a probe thread that samples phase while on_complete is
    # running. With _state_lock held across on_complete, get_service_state
    # must BLOCK — the probe will only return AFTER we release.
    def probe():
        on_complete_entered.wait(timeout=5)
        # This get_service_state call takes _state_lock; if the pipeline
        # holds it across on_complete, probe blocks until we release.
        state_at_probe = sam.get_service_state()
        observed_phases_when_in_on_complete.append(state_at_probe.phase)

    probe_t = threading.Thread(target=probe, name="probe")
    probe_t.start()

    ok, _ = sam.start_pipeline(
        "sid-atomic", "v.mp4", [_ControllableStep("initializing")],
        on_complete=slow_on_complete,
    )
    assert ok is True

    # Wait for on_complete to start
    assert on_complete_entered.wait(timeout=5)

    # Let probe thread try to read state — it should block on _state_lock.
    # Give it a brief window to attempt the read.
    time.sleep(0.15)
    # Probe is blocked on _state_lock — it hasn't recorded anything yet
    assert observed_phases_when_in_on_complete == [], (
        "probe saw state while pipeline held _state_lock in on_complete — "
        "the _state_lock guard is missing"
    )

    # Release on_complete; pipeline now sets phase=ready, releases lock.
    on_complete_release.set()
    sam._pipeline_thread.join(timeout=5)
    probe_t.join(timeout=5)

    # Probe observed phase AFTER lock release — must be "ready" (not
    # "initializing"). on_complete install + ready transition happened
    # atomically.
    assert observed_phases_when_in_on_complete == ["ready"], (
        f"probe observed {observed_phases_when_in_on_complete} — expected "
        f"['ready']. The transition is not atomic."
    )


def test_on_complete_failure_uses_same_state_lock(sam):
    """If on_complete raises, the error-phase transition happens while
    still holding _state_lock (no re-acquisition). A concurrent
    get_service_state must never see 'initializing' after the pipeline
    thread has returned from on_complete."""
    errors_in_on_complete = []

    def failing_on_complete():
        errors_in_on_complete.append("called")
        raise RuntimeError("install failed")

    ok, _ = sam.start_pipeline(
        "sid-fail", "v.mp4", [_ControllableStep("initializing")],
        on_complete=failing_on_complete,
    )
    assert ok is True
    sam._pipeline_thread.join(timeout=5)

    state = sam.get_service_state()
    assert state.phase == "error"
    assert "install failed" in (state.error or "")
    assert errors_in_on_complete == ["called"]


def test_no_deadlock_when_on_complete_takes_globals_lock(sam):
    """on_complete typically acquires _globals_lock (via
    install_active_session). Holding _state_lock then acquiring
    _globals_lock is a new nesting order — verify it does not deadlock.

    We don't expect any other thread to hold _globals_lock → _state_lock
    (config.py helpers don't call into SAM3Service), but assert the
    end-to-end flow completes cleanly."""
    from app.config import close_active_session, get_sync_manager, install_active_session
    from app.services.gcs_sync import GCSSyncManager
    from app.services.session_cache import SessionCache
    import tempfile

    close_active_session()

    def real_on_complete():
        # Mimic the resume route's on_complete: install then start timer.
        # Use a no-op SM backed by a temp dir.
        with tempfile.TemporaryDirectory() as tmpd:
            sm = GCSSyncManager("bkt", "rtt", tmpd)
            install_active_session(sm, SessionCache(tmpd))
            sm.start(interval=60)
            # Don't stop sm here — the test's cleanup will.

    try:
        ok, _ = sam.start_pipeline(
            "sid-nolock", "v.mp4", [_ControllableStep("initializing")],
            on_complete=real_on_complete,
        )
        assert ok is True
        sam._pipeline_thread.join(timeout=5)
        assert sam.get_service_state().phase == "ready"
        # globals were installed by on_complete under _state_lock
        assert get_sync_manager() is not None
    finally:
        sm_left = get_sync_manager()
        if sm_left is not None:
            try:
                sm_left.stop()
            except Exception:
                pass
        close_active_session()
