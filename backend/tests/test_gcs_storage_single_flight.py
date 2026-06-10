"""R5 — `_sessions_cache` single-flight gate (issue #62).

When N concurrent threads call `list_sessions` on a bucket whose cache
entry is missing or expired, only one thread should hit GCS; the rest
must wait on the per-bucket lock and read the populated cache.

The fake GCS client below counts `list_blobs` calls and sleeps inside
the call to widen the race window — without single-flight the assertion
`list_blobs_calls == 1` flips to 8 immediately.
"""
import threading
import time

import pytest

import app.services.gcs_storage as gcs_storage


class _FakeBlob:
    def __init__(self, name: str, size: int = 0):
        self.name = name
        self.size = size

    def exists(self) -> bool:  # no meta/state in the fake bucket
        return False


class _FakeBlobIterator:
    """Mimics google-cloud-storage's iterator: list() yields blobs;
    `.prefixes` is populated as a side effect of consuming the iterator."""

    def __init__(self, prefixes: list[str], blobs: list[_FakeBlob]):
        self._blobs = blobs
        self.prefixes = set(prefixes)

    def __iter__(self):
        return iter(self._blobs)


class _FakeBucket:
    def __init__(self, name: str):
        self.name = name

    def blob(self, path: str) -> _FakeBlob:
        return _FakeBlob(path)


class _CountingClient:
    """Records every `list_blobs` call. Sleeps `delay_s` inside the call
    to make the race window wide enough that any non-single-flight
    implementation will deterministically fan out."""

    def __init__(self, delay_s: float = 0.05):
        self.delay_s = delay_s
        self.list_blobs_calls = 0
        self._counter_lock = threading.Lock()

    def bucket(self, name: str) -> _FakeBucket:
        return _FakeBucket(name)

    def list_blobs(self, bucket, delimiter=None, prefix=None):
        with self._counter_lock:
            self.list_blobs_calls += 1
        time.sleep(self.delay_s)
        # Top-level call (delimiter='/') returns no prefixes => no per-session work
        return _FakeBlobIterator(prefixes=[], blobs=[])


@pytest.fixture(autouse=True)
def _clear_caches_and_locks():
    """Each test gets a clean module-level state."""
    gcs_storage._sessions_cache.clear()
    gcs_storage._sessions_cache_locks.clear()
    # Drop any cached client from prior tests
    if hasattr(gcs_storage._get_client, "_client"):
        del gcs_storage._get_client._client
    yield
    gcs_storage._sessions_cache.clear()
    gcs_storage._sessions_cache_locks.clear()


def test_concurrent_callers_share_one_gcs_list(monkeypatch):
    """8 threads racing into a cold cache must produce exactly 1 GCS call."""
    client = _CountingClient(delay_s=0.05)
    monkeypatch.setattr(gcs_storage, "_get_client", lambda: client)

    bucket = "bucket-a"
    results: list[list[dict]] = []
    results_lock = threading.Lock()
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()  # release all threads simultaneously
        out = gcs_storage.list_sessions(bucket)
        with results_lock:
            results.append(out)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
        assert not t.is_alive(), "worker hung"

    assert client.list_blobs_calls == 1, (
        f"single-flight broken: {client.list_blobs_calls} GCS calls for 8 callers"
    )
    assert len(results) == 8
    # All threads must observe the same populated list
    for r in results:
        assert r == results[0]


def test_different_buckets_do_not_serialise(monkeypatch):
    """Per-bucket locking must not serialise unrelated buckets."""
    client = _CountingClient(delay_s=0.05)
    monkeypatch.setattr(gcs_storage, "_get_client", lambda: client)

    barrier = threading.Barrier(2)

    def worker(bucket: str):
        barrier.wait()
        gcs_storage.list_sessions(bucket)

    t1 = threading.Thread(target=worker, args=("bucket-a",))
    t2 = threading.Thread(target=worker, args=("bucket-b",))
    start = time.monotonic()
    t1.start()
    t2.start()
    t1.join(timeout=5)
    t2.join(timeout=5)
    elapsed = time.monotonic() - start

    # Two buckets, each with 50ms sleep; if serialised we'd see ~100ms.
    # Allow generous slack for thread scheduling.
    assert elapsed < 0.09, f"different buckets serialised: {elapsed:.3f}s"
    assert client.list_blobs_calls == 2  # one per bucket


def test_warm_cache_skips_lock(monkeypatch):
    """A populated, fresh cache entry returns without entering the per-bucket
    lock — fast path is unchanged for the common case."""
    client = _CountingClient(delay_s=0.0)
    monkeypatch.setattr(gcs_storage, "_get_client", lambda: client)

    bucket = "bucket-a"
    first = gcs_storage.list_sessions(bucket)
    assert client.list_blobs_calls == 1

    # Subsequent callers within TTL must not increment the counter.
    for _ in range(5):
        again = gcs_storage.list_sessions(bucket)
        assert again == first
    assert client.list_blobs_calls == 1


def test_expired_cache_recomputes_under_single_flight(monkeypatch):
    """After the TTL expires, the next batch of concurrent callers must
    single-flight one repopulate, not fan out N×."""
    client = _CountingClient(delay_s=0.02)
    monkeypatch.setattr(gcs_storage, "_get_client", lambda: client)
    # TTL must be larger than `delay_s` so the re-populated entry is
    # treated as fresh by waiters released after the populate completes.
    monkeypatch.setattr(gcs_storage, "_SESSIONS_CACHE_TTL", 1.0)

    bucket = "bucket-a"
    # Prime the cache, then force expiry by rewinding the stored timestamp
    # past the TTL window. Avoids a real `time.sleep(TTL)` slowing the suite.
    gcs_storage.list_sessions(bucket)
    assert client.list_blobs_calls == 1
    ts, sessions = gcs_storage._sessions_cache[bucket]
    gcs_storage._sessions_cache[bucket] = (ts - 10.0, sessions)

    barrier = threading.Barrier(4)

    def worker():
        barrier.wait()
        gcs_storage.list_sessions(bucket)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    # First call + exactly one re-populate from the racing batch.
    assert client.list_blobs_calls == 2, (
        f"single-flight broken on expired cache: {client.list_blobs_calls} GCS calls"
    )
