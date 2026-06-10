"""Tests for H6 fix: the close route's globals-clear + phase-to-idle
transition must happen atomically under `_state_lock` so a concurrent
pipeline cannot set `phase=ready` between our globals-clear and any
state observation (#93 — mirror of the H5 race).

Regression: pre-fix, the close route's `finally` block called
`close_active_session()` without holding `_state_lock`. A pipeline
thread that had just installed globals via on_complete (under
`_state_lock`) and was about to set `phase=ready` could have its ready
transition interleave AFTER the close path cleared globals — leaving
`phase=ready, globals=(None, None)` and silently dropping every
subsequent mark_dirty write.

The fix adds `SAM3Service.finalize_close(session_id, teardown)` which
wraps teardown + phase transition in `_state_lock`. This mirrors
`_finalize_ready`'s pattern on the pipeline side.
"""

import threading
import time

import pytest

from app.services.pipeline import ServiceState


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


# --- Structural test: finalize_close must take _state_lock around teardown ---

def test_finalize_close_holds_state_lock_during_teardown(sam):
    """Structural invariant: finalize_close must take `_state_lock`
    BEFORE invoking the teardown closure. A concurrent
    `get_service_state()` from another thread must block for the duration
    of teardown.

    This is the contract the close route relies on: the window between
    globals-clear and state-observation must be closed under one lock.
    """
    observed_during_teardown: list[str] = []
    teardown_entered = threading.Event()
    teardown_release = threading.Event()

    def slow_teardown():
        teardown_entered.set()
        # Hold the critical section long enough for the probe to attempt
        # a state read. If finalize_close holds _state_lock, the probe
        # will block on get_service_state and observe nothing until we
        # release.
        teardown_release.wait(timeout=5)

    def probe():
        teardown_entered.wait(timeout=5)
        # This call takes _state_lock. If finalize_close holds it across
        # teardown, probe blocks until teardown_release.
        state = sam.get_service_state()
        observed_during_teardown.append(state.phase)

    # Seed service state as "ready" for session under test
    with sam._state_lock:
        sam._service_state = ServiceState(
            phase="ready", session_id="sid-close", video_name="v.mp4",
        )

    probe_t = threading.Thread(target=probe, name="probe-close-lock")
    probe_t.start()

    closer_t = threading.Thread(
        target=sam.finalize_close,
        args=("sid-close", slow_teardown),
        name="closer",
    )
    closer_t.start()

    # Wait for teardown to begin
    assert teardown_entered.wait(timeout=5)

    # Give probe time to attempt its read. With the lock held, it must
    # block and record nothing.
    time.sleep(0.15)
    assert observed_during_teardown == [], (
        "probe observed service state during teardown — "
        "finalize_close is not holding _state_lock across teardown"
    )

    # Release teardown; finalize_close transitions to idle, releases lock.
    teardown_release.set()
    closer_t.join(timeout=5)
    probe_t.join(timeout=5)

    # Probe read AFTER the lock release — must see idle.
    assert observed_during_teardown == ["idle"], (
        f"probe observed {observed_during_teardown} — expected ['idle']. "
        f"The teardown + phase transition are not atomic."
    )


def test_finalize_close_transitions_to_idle_on_matching_session(sam):
    """When the active service state matches the closing session_id, the
    phase must transition to idle after teardown runs."""
    with sam._state_lock:
        sam._service_state = ServiceState(
            phase="ready", session_id="sid-match", video_name="v.mp4",
        )

    teardown_calls = []

    def teardown():
        teardown_calls.append("ran")

    sam.finalize_close("sid-match", teardown)
    assert teardown_calls == ["ran"]
    assert sam.get_service_state() == ServiceState()


def test_finalize_close_transitions_to_idle_when_service_is_idle(sam):
    """When the service has no active session, finalize_close still
    clears whatever residual state by transitioning to idle."""
    with sam._state_lock:
        sam._service_state = ServiceState()  # already idle

    sam.finalize_close("sid-anything", lambda: None)
    assert sam.get_service_state().phase == "idle"


def test_finalize_close_does_not_clobber_different_session_state(sam):
    """If a different session is currently the active one (e.g. a resume
    to session B races with a close of session A), finalize_close must
    NOT transition B's state to idle. Teardown still runs."""
    teardown_calls = []

    def teardown():
        teardown_calls.append("ran")

    with sam._state_lock:
        sam._service_state = ServiceState(
            phase="initializing", session_id="sid-B", video_name="v.mp4",
        )

    sam.finalize_close("sid-A", teardown)
    # Teardown MUST run even if session doesn't match — the close route
    # is responsible for its own globals-clear.
    assert teardown_calls == ["ran"]
    # But B's state must remain intact.
    state = sam.get_service_state()
    assert state.session_id == "sid-B"
    assert state.phase == "initializing"


def test_finalize_close_transitions_to_idle_even_if_teardown_raises(sam):
    """If teardown raises, finalize_close must still transition phase to
    idle before propagating the exception. Otherwise the service gets
    stuck in a transient phase forever."""
    with sam._state_lock:
        sam._service_state = ServiceState(
            phase="ready", session_id="sid-raise", video_name="v.mp4",
        )

    def bad_teardown():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        sam.finalize_close("sid-raise", bad_teardown)

    # Service state was reset to idle even though teardown raised.
    assert sam.get_service_state() == ServiceState()


# --- End-to-end race: close during on_complete ---


class _NoopStep:
    """Zero-cost pipeline step. The interesting timing is in on_complete."""

    def __init__(self, phase: str):
        self.phase = phase

    def run(self, on_progress, cancel_event):
        on_progress(1.0)
        return None


def test_close_during_on_complete_never_ends_with_ready_and_empty_globals(sam):
    """H6 regression: a concurrent close during the pipeline's
    on_complete → ready transition must NEVER leave the service in the
    inconsistent state `phase=ready, globals=(None, None)`.

    Setup:
      - Pipeline enters on_complete (under `_state_lock`), installs
        globals via install_active_session.
      - A concurrent close route clears globals via close_active_session,
        wrapped in finalize_close (which takes `_state_lock`).
      - Because finalize_close contends on the same _state_lock held by
        the pipeline across on_complete + ready transition, it must
        serialize after the pipeline finishes.

    Terminal state must be EITHER:
      - (phase=ready, globals=fresh) — close ran first, pipeline
        re-installed on top (not realistic for this sequencing; but
        permitted by the invariant).
      - (phase=idle, globals=(None, None)) — pipeline completed, close
        tore down cleanly.

    Never: (phase=ready, globals=(None, None)).
    """
    from app.config import (
        close_active_session,
        install_active_session,
        get_sync_manager,
    )
    from app.services.gcs_sync import GCSSyncManager
    from app.services.session_cache import SessionCache
    import tempfile

    # Make sure globals start clean so the test is deterministic.
    close_active_session()

    on_complete_entered = threading.Event()
    on_complete_release = threading.Event()

    with tempfile.TemporaryDirectory() as tmpd:
        def pipeline_on_complete():
            # Mimic resume route: install globals then block to widen the
            # race window.
            sm = GCSSyncManager("bkt", "sid-race", tmpd)
            install_active_session(sm, SessionCache(tmpd))
            on_complete_entered.set()
            # Wait for the close to attempt its teardown before we return
            # and let the pipeline set phase=ready.
            on_complete_release.wait(timeout=5)

        ok, _ = sam.start_pipeline(
            "sid-race", "v.mp4", [_NoopStep("initializing")],
            on_complete=pipeline_on_complete,
        )
        assert ok is True

        # Wait for on_complete to enter (globals now installed by pipeline).
        assert on_complete_entered.wait(timeout=5)

        # Launch the close teardown on another thread. It will block on
        # `_state_lock` because the pipeline thread holds it across
        # on_complete + ready transition.
        close_done = threading.Event()

        def closer():
            sam.finalize_close("sid-race", close_active_session)
            close_done.set()

        closer_t = threading.Thread(target=closer, name="closer")
        closer_t.start()

        # Give the closer a moment to attempt the lock. If the fix is
        # working, it will block here. If the fix is NOT in place (i.e.
        # close teardown runs without _state_lock), it would proceed and
        # clear globals now, while pipeline still has phase="initializing"
        # and is about to set phase=ready.
        time.sleep(0.15)

        # Assert: at this instant, either the closer blocked (expected)
        # or we've already raced into an inconsistent state (would fail).
        assert not close_done.is_set(), (
            "closer completed while pipeline still held _state_lock — "
            "finalize_close did not serialize against pipeline (#93 race)"
        )

        # Pre-release state: globals must still be the fresh ones the
        # pipeline installed.
        assert get_sync_manager() is not None, (
            "pipeline's installed sm was cleared before pipeline reached "
            "the ready transition — the H6 race happened"
        )

        # Now let pipeline finish: sets phase=ready, releases _state_lock.
        on_complete_release.set()
        sam._pipeline_thread.join(timeout=5)

        # Closer can now proceed: takes _state_lock, runs teardown (clears
        # globals), transitions phase to idle.
        assert close_done.wait(timeout=5)
        closer_t.join(timeout=5)

        # Terminal state: idle + no globals. Never ready + no globals.
        final_state = sam.get_service_state()
        final_sm = get_sync_manager()
        assert final_state.phase == "idle", (
            f"terminal phase={final_state.phase} — expected 'idle'"
        )
        assert final_sm is None, (
            f"terminal sm={final_sm} — expected None after close finalized"
        )
