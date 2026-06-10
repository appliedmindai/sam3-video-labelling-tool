import queue

import numpy as np
import pytest

def test_mask_to_polygons():
    from app.services.sam3_service import mask_to_polygons
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 30:70] = 1
    polygons = mask_to_polygons(mask)
    assert len(polygons) >= 1
    assert len(polygons[0]) >= 6  # at least 3 points

def test_mask_to_bbox():
    from app.services.sam3_service import mask_to_bbox
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 30:70] = 1
    bbox = mask_to_bbox(mask)
    assert bbox[0] == 30   # x
    assert bbox[1] == 20   # y
    assert bbox[2] == 40   # width
    assert bbox[3] == 60   # height

def test_mask_to_rle():
    from app.services.sam3_service import mask_to_rle
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:80, 30:70] = 1
    rle = mask_to_rle(mask)
    assert "counts" in rle
    assert "size" in rle
    assert rle["size"] == [100, 100]


# --- R32: SSE subscriber backpressure sentinel delivery ---

def test_put_critical_empty_queue_delivers():
    """Empty queue: put_critical just places the item."""
    from app.services.sam3_service import _put_critical
    q = queue.Queue(maxsize=2)
    _put_critical(q, None, "sid", "sentinel")
    assert q.get_nowait() is None


def test_put_critical_full_queue_drops_oldest_and_delivers_sentinel():
    """Full queue: put_critical drops the oldest item and delivers the sentinel."""
    from app.services.sam3_service import _put_critical
    q = queue.Queue(maxsize=2)
    q.put_nowait({"frame_idx": 0})
    q.put_nowait({"frame_idx": 1})
    assert q.full()

    _put_critical(q, None, "sid", "sentinel")

    # Oldest frame dropped, frame_idx=1 and sentinel remain.
    first = q.get_nowait()
    assert first == {"frame_idx": 1}
    second = q.get_nowait()
    assert second is None


def test_put_critical_full_queue_delivers_error_event():
    """Full queue: error event still reaches the subscriber."""
    from app.services.sam3_service import _put_critical
    q = queue.Queue(maxsize=1)
    q.put_nowait({"frame_idx": 0})
    err = {"error": "boom", "traceback": "..."}

    _put_critical(q, err, "sid", "error")

    remaining = q.get_nowait()
    assert remaining == err


# --- #81: SIGTERM joins the propagation daemon thread ---


def test_join_propagation_returns_true_when_no_thread_registered():
    """No registered thread: join_propagation must be a fast no-op returning True."""
    from app.services.sam3_service import SAM3Service

    sam = SAM3Service()
    sam._propagation_threads.pop("absent-sid", None)

    assert sam.join_propagation("absent-sid", timeout=0) is True


def test_join_propagation_returns_true_after_thread_finishes():
    """Registered thread that exits: join with a small timeout returns True."""
    import threading
    from app.services.sam3_service import SAM3Service

    sam = SAM3Service()
    sid = "test-join-finish"
    done = threading.Event()

    def quick_target():
        done.wait(timeout=0.1)

    t = threading.Thread(target=quick_target, daemon=True)
    with sam._propagation_lock:
        sam._propagation_threads[sid] = t
    t.start()
    done.set()

    try:
        assert sam.join_propagation(sid, timeout=2.0) is True
    finally:
        with sam._propagation_lock:
            sam._propagation_threads.pop(sid, None)


def test_join_propagation_returns_false_on_timeout():
    """Registered thread still running past timeout: returns False (SIGTERM
    must not deadlock waiting for an in-flight propagation step)."""
    import threading
    from app.services.sam3_service import SAM3Service

    sam = SAM3Service()
    sid = "test-join-timeout"
    release = threading.Event()

    def slow_target():
        release.wait(timeout=5.0)

    t = threading.Thread(target=slow_target, daemon=True)
    with sam._propagation_lock:
        sam._propagation_threads[sid] = t
    t.start()

    try:
        assert sam.join_propagation(sid, timeout=0.05) is False
    finally:
        release.set()
        t.join(timeout=2.0)
        with sam._propagation_lock:
            sam._propagation_threads.pop(sid, None)


def test_put_critical_saturated_subscriber_terminates_on_sentinel():
    """End-to-end: simulate a subscriber-style loop with a saturated queue.

    The subscriber side of SSE iterates `q.get()` and returns on None. When the
    queue is full at sentinel time, _put_critical must ensure the subscriber
    eventually receives the None terminator.
    """
    from app.services.sam3_service import _put_critical

    q = queue.Queue(maxsize=3)
    q.put_nowait({"frame_idx": 0})
    q.put_nowait({"frame_idx": 1})
    q.put_nowait({"frame_idx": 2})
    assert q.full()

    _put_critical(q, None, "sid", "sentinel")

    # Drain the queue like the SSE subscriber loop does.
    received = []
    while True:
        item = q.get(timeout=0.1)
        if item is None:
            break
        received.append(item)

    # We lost the oldest frame to make room for the sentinel, but the
    # subscriber DID terminate. That's the correctness bar for R32.
    assert len(received) == 2
    assert received[-1] == {"frame_idx": 2}


# --- #68 / R15: propagation finally must not resurrect closed session state ---


def _stub_propagation_entry(sam, sid):
    """Register enough state for _run_propagation to exercise finally/except
    without touching an actual SAM3 predictor."""
    import threading

    sam._backend = "native"
    sam._sessions[sid] = object()
    sam._session_meta[sid] = {"num_frames": 0}
    sam._propagation_state[sid] = {"status": "running"}
    sam._propagation_subscribers[sid] = []
    sam._cancel_events[sid] = threading.Event()


def test_run_propagation_success_pops_propagation_state():
    """Clean run: finally pops the entry (equivalent to idle under
    get_propagation_status's default) instead of writing a zombie."""
    from app.services.sam3_service import SAM3Service

    sam = SAM3Service()
    sid = "test-finally-success"
    _stub_propagation_entry(sam, sid)

    # Empty frame iterator — no frames to process, no error.
    sam._propagate_native = lambda *args, **kwargs: iter([])

    try:
        sam._run_propagation(sid, start_frame=0, reverse=False, persist_fn=lambda r: None)
        # Pop, don't resurrect.
        assert sid not in sam._propagation_state
        # get_propagation_status still returns idle for absent keys.
        assert sam.get_propagation_status(sid) == {"status": "idle"}
    finally:
        sam._sessions.pop(sid, None)
        sam._session_meta.pop(sid, None)
        sam._propagation_state.pop(sid, None)
        sam._propagation_subscribers.pop(sid, None)
        sam._cancel_events.pop(sid, None)


def test_run_propagation_finally_does_not_resurrect_closed_session_state():
    """R15: if close_session popped _propagation_state mid-run, the finally
    block must NOT resurrect the entry."""
    from app.services.sam3_service import SAM3Service

    sam = SAM3Service()
    sid = "test-finally-closed"
    _stub_propagation_entry(sam, sid)

    # Simulate close_session popping the state before the generator yields
    # anything (i.e., between try-block entry and finally).
    def simulate_close_iter(*args, **kwargs):
        sam._propagation_state.pop(sid, None)
        sam._propagation_subscribers.pop(sid, None)
        sam._cancel_events.pop(sid, None)
        return iter([])

    sam._propagate_native = simulate_close_iter

    try:
        sam._run_propagation(sid, start_frame=0, reverse=False, persist_fn=lambda r: None)
        # No zombie entry for a closed session.
        assert sid not in sam._propagation_state
    finally:
        sam._sessions.pop(sid, None)
        sam._session_meta.pop(sid, None)
        sam._propagation_state.pop(sid, None)
        sam._propagation_subscribers.pop(sid, None)
        sam._cancel_events.pop(sid, None)


def test_run_propagation_except_does_not_resurrect_closed_session_state():
    """R15 except branch: if propagation raises AFTER close_session cleared
    _propagation_state, the except block must NOT write a zombie 'failed'."""
    from app.services.sam3_service import SAM3Service

    sam = SAM3Service()
    sid = "test-except-closed"
    _stub_propagation_entry(sam, sid)

    def explode_after_close(*args, **kwargs):
        sam._propagation_state.pop(sid, None)

        def gen():
            raise RuntimeError("boom")
            yield  # pragma: no cover
        return gen()

    sam._propagate_native = explode_after_close

    try:
        sam._run_propagation(sid, start_frame=0, reverse=False, persist_fn=lambda r: None)
        assert sid not in sam._propagation_state
    finally:
        sam._sessions.pop(sid, None)
        sam._session_meta.pop(sid, None)
        sam._propagation_state.pop(sid, None)
        sam._propagation_subscribers.pop(sid, None)
        sam._cancel_events.pop(sid, None)


def test_run_propagation_except_writes_failed_when_session_still_open():
    """Regression guard: if the session is still registered when propagation
    raises, the except block still records 'failed' so callers can observe it."""
    from app.services.sam3_service import SAM3Service

    sam = SAM3Service()
    sid = "test-except-open"
    _stub_propagation_entry(sam, sid)

    def explode(*args, **kwargs):
        def gen():
            raise RuntimeError("boom")
            yield  # pragma: no cover
        return gen()

    sam._propagate_native = explode

    try:
        sam._run_propagation(sid, start_frame=0, reverse=False, persist_fn=lambda r: None)
        status = sam.get_propagation_status(sid)
        assert status["status"] == "failed"
        assert "boom" in status["error"]
    finally:
        sam._sessions.pop(sid, None)
        sam._session_meta.pop(sid, None)
        sam._propagation_state.pop(sid, None)
        sam._propagation_subscribers.pop(sid, None)
        sam._cancel_events.pop(sid, None)


# --- R7 / R30: _propagation_state lifecycle under _propagation_lock ---


def test_get_propagation_status_returns_snapshot_not_live_ref():
    """R7: get_propagation_status must return a snapshot copy so the caller
    cannot observe later mutations of the underlying dict."""
    from app.services.sam3_service import SAM3Service

    sam = SAM3Service()
    sid = "test-snapshot"
    with sam._propagation_lock:
        sam._propagation_state[sid] = {
            "status": "running",
            "start_frame": 0,
            "reverse": False,
            "frames_processed": 3,
        }

    try:
        snapshot = sam.get_propagation_status(sid)
        assert snapshot["frames_processed"] == 3

        # Mutate the live entry; the snapshot must not reflect it.
        with sam._propagation_lock:
            sam._propagation_state[sid]["frames_processed"] = 42

        assert snapshot["frames_processed"] == 3
        fresh = sam.get_propagation_status(sid)
        assert fresh["frames_processed"] == 42
    finally:
        with sam._propagation_lock:
            sam._propagation_state.pop(sid, None)


def test_get_propagation_status_idle_for_missing_session():
    """Missing session still returns the idle sentinel under the new lock path."""
    from app.services.sam3_service import SAM3Service

    sam = SAM3Service()
    assert sam.get_propagation_status("never-registered") == {"status": "idle"}


def test_concurrent_status_reads_and_counter_updates_are_coherent():
    """R7/R30: a reader hammering get_propagation_status concurrently with a
    writer bumping frames_processed must never see a non-integer or missing
    counter field — i.e. the snapshot is never torn."""
    import threading
    from app.services.sam3_service import SAM3Service

    sam = SAM3Service()
    sid = "test-concurrent"
    with sam._propagation_lock:
        sam._propagation_state[sid] = {
            "status": "running",
            "start_frame": 0,
            "reverse": False,
            "frames_processed": 0,
        }

    stop = threading.Event()
    observed_bad = []

    def reader():
        while not stop.is_set():
            snap = sam.get_propagation_status(sid)
            fp = snap.get("frames_processed")
            if fp is None or not isinstance(fp, int):
                observed_bad.append(snap)
                return

    def writer():
        for _ in range(500):
            with sam._propagation_lock:
                ps = sam._propagation_state.get(sid)
                if ps:
                    ps["frames_processed"] = ps.get("frames_processed", 0) + 1

    try:
        t_reader = threading.Thread(target=reader, daemon=True)
        t_reader.start()
        writer()
        stop.set()
        t_reader.join(timeout=2.0)
        assert observed_bad == []
        assert sam.get_propagation_status(sid)["frames_processed"] == 500
    finally:
        with sam._propagation_lock:
            sam._propagation_state.pop(sid, None)


# --- R8: _propagation_subscribers list mutation under _propagation_lock ---


def test_subscribe_propagation_returns_immediately_when_session_not_running():
    """No registered subscribers list -> generator returns without yielding."""
    from app.services.sam3_service import SAM3Service

    sam = SAM3Service()
    sid = "test-sub-no-session"
    # Ensure no entry exists.
    with sam._propagation_lock:
        sam._propagation_subscribers.pop(sid, None)

    gen = sam.subscribe_propagation(sid)
    with pytest.raises(StopIteration):
        next(gen)


def test_subscribe_propagation_installs_and_removes_queue_under_lock():
    """Single-subscriber happy path: queue is installed in the subscribers
    list under _propagation_lock, sentinel is delivered, finally block
    removes the queue under the same lock. No behaviour change vs the old
    unlocked path for single-subscriber flows."""
    import threading
    from app.services.sam3_service import SAM3Service

    sam = SAM3Service()
    sid = "test-sub-happy"
    with sam._propagation_lock:
        sam._propagation_state[sid] = {"status": "running", "frames_processed": 0}
        sam._propagation_subscribers[sid] = []

    received = []
    started = threading.Event()
    finished = threading.Event()

    def consumer():
        started.set()
        for item in sam.subscribe_propagation(sid):
            received.append(item)
        finished.set()

    t = threading.Thread(target=consumer, daemon=True)
    t.start()
    started.wait(timeout=2.0)

    # Wait for the consumer to install its queue under the lock.
    deadline = threading.Event()
    for _ in range(200):
        with sam._propagation_lock:
            installed = len(sam._propagation_subscribers.get(sid, ()))
        if installed == 1:
            break
        deadline.wait(0.01)
    assert installed == 1, "subscriber queue was never installed under the lock"

    # Push a frame, then a sentinel, the way _run_propagation does.
    with sam._propagation_lock:
        snapshot = list(sam._propagation_subscribers.get(sid, ()))
    for q in snapshot:
        q.put_nowait({"frame_idx": 7})
        q.put_nowait(None)

    finished.wait(timeout=2.0)
    assert finished.is_set()
    assert received == [{"frame_idx": 7}]

    # Finally block removed the queue under the lock.
    with sam._propagation_lock:
        assert sam._propagation_subscribers.get(sid) == []
        sam._propagation_subscribers.pop(sid, None)
        sam._propagation_state.pop(sid, None)


def test_subscribe_propagation_after_close_returns_cleanly():
    """R8 race: if close_session pops _propagation_subscribers between a
    stale read and the queue append, the new code (atomic check+append
    under the lock) must return immediately rather than orphan the queue."""
    from app.services.sam3_service import SAM3Service

    sam = SAM3Service()
    sid = "test-sub-after-close"
    # No entry installed -> close_session-equivalent already happened.
    with sam._propagation_lock:
        sam._propagation_subscribers.pop(sid, None)

    gen = sam.subscribe_propagation(sid)
    with pytest.raises(StopIteration):
        next(gen)


def test_run_propagation_fanout_tolerates_concurrent_subscribe_unsubscribe():
    """R8 core invariant: _run_propagation's per-frame fan-out (and the
    error/finally fan-outs) must not raise `RuntimeError: list changed
    size during iteration` even under heavy churn from
    subscribe_propagation / its finally block. The snapshot-under-lock
    pattern enforces this."""
    import threading
    import time
    from app.services.sam3_service import SAM3Service

    sam = SAM3Service()
    sid = "test-fanout-churn"
    _stub_propagation_entry(sam, sid)

    # 100 frames with a tiny per-frame yield to interleave with the churners.
    def lots_of_frames(*args, **kwargs):
        for i in range(100):
            yield {
                "frame_idx": i,
                "masks": {},
            }
            # Microsleep gives the churn threads a chance to mutate.
            time.sleep(0)

    sam._propagate_native = lots_of_frames

    stop_churn = threading.Event()
    churn_errors: list[BaseException] = []

    def churner():
        # Aggressively append/remove queues directly through the public API
        # path's mutation primitives (i.e. under _propagation_lock).
        try:
            while not stop_churn.is_set():
                with sam._propagation_lock:
                    subs = sam._propagation_subscribers.get(sid)
                    if subs is None:
                        return
                    import queue as _q
                    qx = _q.Queue(maxsize=1)
                    subs.append(qx)
                with sam._propagation_lock:
                    subs = sam._propagation_subscribers.get(sid)
                    if subs is not None:
                        try:
                            subs.remove(qx)
                        except ValueError:
                            pass
        except BaseException as e:  # pragma: no cover
            churn_errors.append(e)

    churners = [threading.Thread(target=churner, daemon=True) for _ in range(4)]
    for t in churners:
        t.start()

    try:
        sam._run_propagation(sid, start_frame=0, reverse=False, persist_fn=lambda r: None)
    finally:
        stop_churn.set()
        for t in churners:
            t.join(timeout=2.0)
        sam._sessions.pop(sid, None)
        sam._session_meta.pop(sid, None)
        with sam._propagation_lock:
            sam._propagation_state.pop(sid, None)
            sam._propagation_subscribers.pop(sid, None)
        sam._cancel_events.pop(sid, None)

    assert churn_errors == []


def test_run_propagation_finally_pops_subscribers_under_propagation_lock():
    """R8: the finally block must drop _propagation_subscribers[sid]
    inside _propagation_lock so a late-arriving subscribe_propagation
    sees a consistent absent / present view."""
    from app.services.sam3_service import SAM3Service

    sam = SAM3Service()
    sid = "test-finally-pops-subs"
    _stub_propagation_entry(sam, sid)

    sam._propagate_native = lambda *a, **kw: iter([])

    try:
        sam._run_propagation(sid, start_frame=0, reverse=False, persist_fn=lambda r: None)
        with sam._propagation_lock:
            assert sid not in sam._propagation_subscribers
    finally:
        sam._sessions.pop(sid, None)
        sam._session_meta.pop(sid, None)
        with sam._propagation_lock:
            sam._propagation_state.pop(sid, None)
            sam._propagation_subscribers.pop(sid, None)
        sam._cancel_events.pop(sid, None)


# --- R11 / #66: propagation persist_fn -> session_io_lock ordering invariant ---


def test_propagation_persist_fn_runs_inside_sam3_lock():
    """R11: propagation must hold SAM3._lock across persist_fn. Precondition
    for the lock ordering rule: if _lock were dropped, no inversion could
    deadlock and the invariant at session_lock.py:11-14 would be moot."""
    import threading
    from app.services.sam3_service import SAM3Service

    sam = SAM3Service()
    sid = "test-persist-inside-lock"
    _stub_propagation_entry(sam, sid)

    persist_entered = threading.Event()
    outer_can_acquire_lock_while_persist_running = []

    # Probe _lock from a different thread while persist_fn is running. _lock
    # is an RLock, so we MUST probe from a distinct thread — the propagation
    # thread would re-enter successfully.
    def probe():
        persist_entered.wait(timeout=2.0)
        # blocking=False is the non-deadlocking way to observe contention
        acquired = sam._lock.acquire(blocking=False)
        outer_can_acquire_lock_while_persist_running.append(acquired)
        if acquired:
            sam._lock.release()

    probe_thread = threading.Thread(target=probe, daemon=True)

    def persist_fn(result):
        persist_entered.set()
        # Give the probe thread a chance to try the lock
        probe_thread.join(timeout=1.0)

    def one_frame_iter(*args, **kwargs):
        return iter([{"frame_idx": 0, "masks": {}}])

    sam._propagate_native = one_frame_iter

    try:
        probe_thread.start()
        sam._run_propagation(sid, start_frame=0, reverse=False, persist_fn=persist_fn)
        probe_thread.join(timeout=2.0)
        assert persist_entered.is_set(), "persist_fn was never invoked"
        assert outer_can_acquire_lock_while_persist_running == [False], (
            "Another thread acquired SAM3._lock while persist_fn was running — "
            "invariant broken: persist_fn is no longer protected by _lock, so "
            "R11's lock-ordering rule no longer applies."
        )
    finally:
        sam._sessions.pop(sid, None)
        sam._session_meta.pop(sid, None)
        with sam._propagation_lock:
            sam._propagation_state.pop(sid, None)
        sam._propagation_subscribers.pop(sid, None)
        sam._cancel_events.pop(sid, None)


def test_propagation_persist_fn_acquires_session_io_lock_without_deadlock():
    """R11: persist_fn taking session_io_lock inside SAM3._lock is the
    canonical order. It must run to completion without self-deadlocking."""
    import threading
    from app.services.sam3_service import SAM3Service
    from app.services.session_lock import session_io_lock

    sam = SAM3Service()
    sid = "test-persist-takes-session-io-lock"
    _stub_propagation_entry(sam, sid)

    persist_completions = []

    def persist_fn(result):
        # Mimics segment.py:168-169 — the real persist_fn body.
        with session_io_lock(sid):
            persist_completions.append(result["frame_idx"])

    sam._propagate_native = lambda *args, **kwargs: iter([
        {"frame_idx": i, "masks": {}} for i in range(3)
    ])

    # Run propagation on a worker thread with a hard deadline. If the
    # canonical _lock -> session_io_lock order ever self-deadlocks (e.g. a
    # refactor routes persist through another code path that re-enters
    # session_io_lock differently), the thread never finishes.
    done = threading.Event()

    def runner():
        try:
            sam._run_propagation(sid, start_frame=0, reverse=False, persist_fn=persist_fn)
        finally:
            done.set()

    runner_thread = threading.Thread(target=runner, daemon=True)

    try:
        runner_thread.start()
        assert done.wait(timeout=5.0), "propagation thread did not finish — self-deadlock?"
        assert persist_completions == [0, 1, 2]
    finally:
        sam._sessions.pop(sid, None)
        sam._session_meta.pop(sid, None)
        with sam._propagation_lock:
            sam._propagation_state.pop(sid, None)
        sam._propagation_subscribers.pop(sid, None)
        sam._cancel_events.pop(sid, None)


def test_reverse_lock_order_deadlocks_as_documented():
    """R11: a thread holding session_io_lock then asking for SAM3._lock while
    propagation is mid-persist deadlocks. Timeout-based detection
    (_lock.acquire(timeout=0.5) returns False) records the signature so a
    future route that inverts the order is recognisable — not fixable at the
    lock layer, which is the whole reason for the comment at
    session_lock.py:11-14."""
    import threading
    from app.services.sam3_service import SAM3Service
    from app.services.session_lock import session_io_lock

    sam = SAM3Service()
    sid = "test-reverse-order-deadlocks"
    _stub_propagation_entry(sam, sid)

    persist_waiting = threading.Event()
    release_persist = threading.Event()
    inverted_acquire_result = []

    # persist_fn holds inside SAM3._lock; block here until the test signals
    # that it has observed the (would-be) deadlock signature.
    def persist_fn(result):
        persist_waiting.set()
        release_persist.wait(timeout=2.0)

    sam._propagate_native = lambda *args, **kwargs: iter([
        {"frame_idx": 0, "masks": {}},
    ])

    # Propagation owns SAM3._lock across persist_fn.
    prop_thread = threading.Thread(
        target=sam._run_propagation,
        args=(sid, 0, False, persist_fn),
        daemon=True,
    )

    # Inverter: acquires session_io_lock first, then tries SAM3._lock — the
    # reverse of propagation's order. Must NOT use RLock re-entry from the
    # propagation thread, so runs on its own thread. SAM3._lock is an RLock,
    # but acquire() from a different thread behaves as a normal lock.
    def inverter():
        persist_waiting.wait(timeout=2.0)
        with session_io_lock(sid):
            # If this acquires, the invariant is meaningless — propagation
            # would not actually be holding _lock when persist_fn runs.
            inverted_acquire_result.append(sam._lock.acquire(blocking=True, timeout=0.5))
            if inverted_acquire_result[-1]:
                sam._lock.release()

    inverter_thread = threading.Thread(target=inverter, daemon=True)

    try:
        prop_thread.start()
        inverter_thread.start()
        # Give the inverter its 0.5s window to fail to acquire _lock.
        inverter_thread.join(timeout=3.0)
        assert inverted_acquire_result == [False], (
            "Reverse-order acquire succeeded — either propagation isn't "
            "holding _lock across persist_fn, or the timeout was too short."
        )
    finally:
        # Let persist_fn return so propagation can release _lock and clean up.
        release_persist.set()
        prop_thread.join(timeout=2.0)
        inverter_thread.join(timeout=2.0)
        sam._sessions.pop(sid, None)
        sam._session_meta.pop(sid, None)
        with sam._propagation_lock:
            sam._propagation_state.pop(sid, None)
        sam._propagation_subscribers.pop(sid, None)
        sam._cancel_events.pop(sid, None)


# --- R9: benchmark route reads SAM3 internals without _lock ---

def test_debug_snapshot_returns_safe_keys_when_model_not_loaded():
    """Singleton with no model loaded: snapshot returns sentinel strings, not exceptions."""
    from app.services.sam3_service import SAM3Service

    sam = SAM3Service()
    snap = sam.debug_snapshot()

    expected_keys = {
        "device", "backend", "model", "model_params",
        "native_predictor_loaded", "sessions_loaded", "loaded_session_ids",
    }
    assert set(snap.keys()) == expected_keys
    assert isinstance(snap["sessions_loaded"], int)
    assert isinstance(snap["loaded_session_ids"], list)
    assert isinstance(snap["native_predictor_loaded"], bool)
    # Plain JSON-serialisable strings, never None — benchmark route consumes
    # these directly into a JSON response.
    assert isinstance(snap["device"], str)
    assert isinstance(snap["backend"], str)
    assert isinstance(snap["model"], str)
    assert isinstance(snap["model_params"], str)


def test_debug_snapshot_acquires_lock():
    """Snapshot must serialise against `_lock` holders. If a writer holds
    `_lock`, `debug_snapshot()` blocks until the writer releases it — proving
    the read of `_model` / `_sessions` is no longer torn against
    `_ensure_model` / `init_session` / `close_session`."""
    import threading
    from app.services.sam3_service import SAM3Service

    sam = SAM3Service()
    holder_acquired = threading.Event()
    holder_release = threading.Event()
    snapshot_started = threading.Event()
    snapshot_finished = threading.Event()
    snap_box: list = []

    def hold_lock():
        with sam._lock:
            holder_acquired.set()
            holder_release.wait(timeout=2.0)

    def take_snapshot():
        snapshot_started.set()
        snap_box.append(sam.debug_snapshot())
        snapshot_finished.set()

    holder = threading.Thread(target=hold_lock, daemon=True)
    snapper = threading.Thread(target=take_snapshot, daemon=True)
    try:
        holder.start()
        assert holder_acquired.wait(timeout=2.0), "holder failed to acquire _lock"
        snapper.start()
        assert snapshot_started.wait(timeout=2.0)
        # Snapshot must NOT complete while writer holds the lock.
        assert not snapshot_finished.wait(timeout=0.2), (
            "debug_snapshot returned while _lock was held by another thread — "
            "it is not actually acquiring _lock"
        )
        holder_release.set()
        assert snapshot_finished.wait(timeout=2.0), (
            "debug_snapshot did not return after _lock was released"
        )
        assert len(snap_box) == 1
    finally:
        holder_release.set()
        holder.join(timeout=2.0)
        snapper.join(timeout=2.0)


# --- R29 / #73: _cleanup_partial check-then-rmtree TOCTOU ---


def test_cleanup_partial_blocks_on_session_io_lock(tmp_path, monkeypatch):
    """R29: `_cleanup_partial` must acquire `session_io_lock(session_id)` so
    concurrent state/mask/prompt writers on the same RMW path are serialised
    out of the check-then-rmtree window. A writer that drops a valid state.json
    between the existence check and the rmtree would otherwise be blown away.

    We prove the lock is acquired by holding it from another thread and
    asserting that `_cleanup_partial` does not complete until we release it.
    """
    import threading

    from app.services import sam3_service
    from app.services.session_lock import session_io_lock

    monkeypatch.setattr(sam3_service, "SESSIONS_DIR", str(tmp_path))
    sam = sam3_service.SAM3Service()

    sid = "race-session"
    session_dir = tmp_path / sid
    session_dir.mkdir()
    # No state.json, no frames — _cleanup_partial WOULD rmtree this dir
    # if it could reach the rmtree line.

    cleanup_done = threading.Event()

    def run_cleanup():
        sam._cleanup_partial(sid)
        cleanup_done.set()

    lock = session_io_lock(sid)
    lock.acquire()
    try:
        cleaner = threading.Thread(target=run_cleanup, daemon=True)
        cleaner.start()
        # Cleanup thread is blocked on the lock we hold.
        assert not cleanup_done.wait(timeout=0.3), (
            "_cleanup_partial returned while session_io_lock was held by "
            "another thread — it is not acquiring the lock"
        )
        # While cleanup is blocked, simulate the concurrent writer that the
        # TOCTOU race would lose to: drop a valid state.json.
        (session_dir / "state.json").write_text('{"version": 1}')
    finally:
        lock.release()

    assert cleanup_done.wait(timeout=2.0)
    cleaner.join(timeout=2.0)

    # State.json was written while cleanup was blocked on the lock. Cleanup
    # now runs under the lock, re-checks, sees state.json, and bails out.
    # Without the fix, cleanup would have run first and rmtreed the dir.
    assert session_dir.exists()
    assert (session_dir / "state.json").exists()
    assert (session_dir / "state.json").read_text() == '{"version": 1}'


def test_cleanup_partial_with_no_concurrent_writer_still_removes_empty_session(
    tmp_path, monkeypatch
):
    """Sanity: the lock addition does not change the happy path. An empty
    session dir (no state.json, no frames) is still rmtreed by
    `_cleanup_partial`.
    """
    from app.services import sam3_service

    monkeypatch.setattr(sam3_service, "SESSIONS_DIR", str(tmp_path))
    sam = sam3_service.SAM3Service()

    sid = "empty-session"
    session_dir = tmp_path / sid
    session_dir.mkdir()

    sam._cleanup_partial(sid)
    assert not session_dir.exists()


def test_cleanup_partial_preserves_session_with_state_json(tmp_path, monkeypatch):
    """Sanity: a session with state.json is never removed, even when no
    concurrent writer is active.
    """
    from app.services import sam3_service

    monkeypatch.setattr(sam3_service, "SESSIONS_DIR", str(tmp_path))
    sam = sam3_service.SAM3Service()

    sid = "annotated-session"
    session_dir = tmp_path / sid
    session_dir.mkdir()
    (session_dir / "state.json").write_text('{"version": 1}')

    sam._cleanup_partial(sid)
    assert session_dir.exists()
    assert (session_dir / "state.json").exists()


def test_cleanup_partial_preserves_session_with_frames(tmp_path, monkeypatch):
    """Sanity: a session with extracted frames is never removed."""
    from app.services import sam3_service

    monkeypatch.setattr(sam3_service, "SESSIONS_DIR", str(tmp_path))
    sam = sam3_service.SAM3Service()

    sid = "extracted-session"
    session_dir = tmp_path / sid
    session_dir.mkdir()
    frames_dir = session_dir / "frames"
    frames_dir.mkdir()
    (frames_dir / "00001.jpg").write_bytes(b"fake-jpeg")

    sam._cleanup_partial(sid)
    assert session_dir.exists()
    assert (frames_dir / "00001.jpg").exists()
