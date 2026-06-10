"""Tests for backend.app.services.session_lock — R19 bounded-LRU."""
import threading

import pytest

from app.services import session_lock


@pytest.fixture(autouse=True)
def _reset_lock_cache():
    """Each test runs against a clean lock map."""
    with session_lock._locks_guard:
        session_lock._locks.clear()
    yield
    with session_lock._locks_guard:
        session_lock._locks.clear()


def test_returns_same_lock_for_same_session_id():
    lock_a = session_lock.session_io_lock("s1")
    lock_b = session_lock.session_io_lock("s1")
    assert lock_a is lock_b


def test_returns_distinct_locks_for_different_sessions():
    lock_a = session_lock.session_io_lock("s1")
    lock_b = session_lock.session_io_lock("s2")
    assert lock_a is not lock_b


def test_cap_is_enforced_and_oldest_evicted():
    # Fill cache beyond cap
    for i in range(session_lock._LOCK_CACHE_CAP + 100):
        session_lock.session_io_lock(f"s{i}")

    with session_lock._locks_guard:
        assert len(session_lock._locks) == session_lock._LOCK_CACHE_CAP
        # Oldest ones gone, newest ones retained
        assert "s0" not in session_lock._locks
        assert f"s{session_lock._LOCK_CACHE_CAP + 99}" in session_lock._locks


def test_hot_key_is_retained_after_eviction_pressure():
    hot = session_lock.session_io_lock("hot")
    # Fill cache beyond cap; between fills, keep re-touching the hot key
    # so LRU policy retains it.
    for i in range(session_lock._LOCK_CACHE_CAP + 100):
        session_lock.session_io_lock(f"cold{i}")
        if i % 10 == 0:
            session_lock.session_io_lock("hot")

    still_hot = session_lock.session_io_lock("hot")
    assert still_hot is hot, "LRU should retain recently-touched keys"


def test_cap_evicts_exactly_one_per_overflow():
    for i in range(session_lock._LOCK_CACHE_CAP):
        session_lock.session_io_lock(f"s{i}")
    with session_lock._locks_guard:
        assert len(session_lock._locks) == session_lock._LOCK_CACHE_CAP

    session_lock.session_io_lock("new")
    with session_lock._locks_guard:
        assert len(session_lock._locks) == session_lock._LOCK_CACHE_CAP
        assert "s0" not in session_lock._locks
        assert "new" in session_lock._locks


def test_concurrent_factory_is_safe():
    """Concurrent callers for the same key must receive the same lock object."""
    results: list[threading.Lock] = []
    results_lock = threading.Lock()
    barrier = threading.Barrier(16)

    def worker():
        barrier.wait()
        lock = session_lock.session_io_lock("shared")
        with results_lock:
            results.append(lock)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 16
    first = results[0]
    assert all(l is first for l in results)
