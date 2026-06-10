import os
import threading
import time
import uuid

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")
UPLOAD_MAX_SIZE = 500 * 1024 * 1024  # 500MB

# SAM 3.1 — HuggingFace Transformers (Sam3TrackerVideoModel)
# Requires: huggingface-cli login (model is gated, needs HF auth)
SAM3_MODEL_ID = os.environ.get("SAM3_MODEL_ID", "facebook/sam3")
SAM3_DEVICE = os.environ.get("SAM3_DEVICE", "auto")  # auto | mps | cuda | cpu

# Backend selection: "auto" picks native on CUDA, HF elsewhere.
# "native" forces facebookresearch/sam3, "hf" forces HuggingFace Transformers.
SAM3_BACKEND = os.environ.get("SAM3_BACKEND", "auto")  # auto | native | hf

# Cloud mode (Cloud Run + GCS). When SEGMENT_MODE=cloud, sessions sync to
# the GCS bucket named by GCS_BUCKET; the before_request hook in
# app/__init__.py exposes it to routes as g.bucket.
SEGMENT_MODE = os.environ.get("SEGMENT_MODE", "")  # "cloud" or "" (standalone)
GCS_BUCKET = os.environ.get("GCS_BUCKET", "")

# Boot tracking — used by /api/health for container restart detection
BOOT_ID = str(uuid.uuid4())
BOOT_TIME = time.time()

FRAME_EXTRACTION_FPS = 2  # default frames per second to extract

os.makedirs(SESSIONS_DIR, exist_ok=True)
os.makedirs(EXPORTS_DIR, exist_ok=True)

# Serializes transitions of the active sync manager AND the active session
# cache. These two globals move together (both belong to a single "active
# session" concept); protecting them under one lock prevents a concurrent
# resume(B) from installing cache_B while close(A) is mid-stop() on sm_A.
#
# Lock hierarchy: `_globals_lock` is level #5 — acquired AFTER
# `SAM3._state_lock` (via `finalize_close`), `SAM3._lock`,
# `session_io_lock`, and `SessionCache._lock`, BEFORE
# `GCSSyncManager._lock` (via `mark_dirty_safe` → `get_sync_manager` →
# `mark_dirty`). See docs/TECHNICAL_REPORT.md § Concurrency model.
_globals_lock = threading.Lock()
_active_sync_manager = None
_active_session_cache = None


def get_sync_manager():
    with _globals_lock:
        return _active_sync_manager


def _stop_and_record(old, reason: str = "set_sync_manager_stop_failed") -> None:
    """Stop an outgoing sync manager and persist an .unsynced marker on
    any failure. Factored so both `set_sync_manager(new)` and
    `install_active_session(sm_new, cache_new)` can share the logic.

    Runs OUTSIDE the globals lock — stop() performs GCS uploads that
    can take seconds; blocking readers for that long would starve
    other requests. Safe because by the time we get here, `old` is no
    longer referenced by the active slots.
    """
    if old is None:
        return

    import logging
    logger = logging.getLogger(__name__)

    # Short-circuit when the manager is already stopped. The close-path
    # (close_session route) calls stop() itself and then may call
    # set_sync_manager(new) later. A second stop() would re-run flush()
    # against a GCS client that we've already torn down, surface
    # spurious errors, and — worse — snapshot a now-stale `_dirty` that
    # may include files that WERE successfully flushed but not removed
    # from the set because stop() raised mid-way. The close-path has
    # already persisted a marker if needed; we must not race with it.
    if getattr(old, "_stopped", False):
        return

    # Snapshot dirty state BEFORE stop() so we know what to record even
    # if stop() raises mid-flush. promote_deferred folds _deferred into
    # _dirty so the snapshot includes in-flight propagation writes.
    try:
        old.promote_deferred()
    except Exception:
        logger.warning(
            "_stop_and_record: promote_deferred raised", exc_info=True
        )
    try:
        pre_stop_dirty = sorted(old.dirty_snapshot())
    except Exception:
        logger.warning(
            "_stop_and_record: dirty_snapshot raised", exc_info=True
        )
        pre_stop_dirty = []

    result = None
    try:
        result = old.stop()
    except Exception:
        logger.warning(
            "_stop_and_record: old manager stop() failed", exc_info=True
        )

    # Decide which files still need a marker.
    if result is not None and result.ok:
        return  # clean flush — nothing to record
    if result is not None:
        failed = sorted(result.failed)
    else:
        # stop() raised; we cannot know which files made it up. Worst
        # case: treat everything that was dirty pre-stop as unsynced.
        # Over-records rather than under-records — upload_file is
        # idempotent so the next resume re-uploading a clean file is
        # harmless.
        failed = pre_stop_dirty

    if not failed:
        return

    try:
        from app.services.unsynced_marker import persist_unsynced
        # GCSSyncManager carries bucket_name/session_id; callers don't
        # need to thread them through. `bucket_name` may be falsy for
        # test doubles — persist_unsynced handles a missing bucket by
        # writing a local-only marker.
        persist_unsynced(
            getattr(old, "bucket_name", None),
            getattr(old, "session_id", "unknown"),
            getattr(old, "session_dir", ""),
            failed,
            reason=reason,
        )
    except Exception:
        logger.error(
            "_stop_and_record: failed to write unsynced marker",
            exc_info=True,
        )


def set_sync_manager(manager):
    """Install a new sync manager, stopping the old one first (if any).

    The pointer swap happens under _globals_lock so a concurrent
    get_sync_manager sees a consistent state. The .stop() call runs
    OUTSIDE the lock because it performs GCS uploads which can take
    seconds — blocking readers for that long would starve other
    requests.

    If the old manager's final flush fails OR stop() raises, an
    .unsynced marker is written to its session_dir so the next
    DownloadSessionStep can recover the dirty files. Without this, a
    session swap (e.g. resume while another session still has dirty
    files) would silently drop data on transient GCS failures.
    """
    global _active_sync_manager
    with _globals_lock:
        old = _active_sync_manager
        _active_sync_manager = manager
    _stop_and_record(old)


def get_session_cache():
    with _globals_lock:
        return _active_session_cache


def set_session_cache(cache):
    global _active_session_cache
    with _globals_lock:
        _active_session_cache = cache


def install_active_session(sync_manager, session_cache):
    """Atomically install a new sync manager AND session cache.

    Closes the two-tab resume race: pre-B5, `DownloadSessionStep`
    installed the sync manager mid-pipeline while `session_cache` was
    still pointing at the previous session. Tab A's `mark_dirty` would
    route to sm_B while cache_A still served reads — A's writes would
    land in B's session_dir on disk.

    This helper holds `_globals_lock` across BOTH swaps so no concurrent
    request can observe a (sm_new, cache_old) or (sm_old, cache_new)
    state. The outgoing sync manager (if any) is stopped+recorded
    OUTSIDE the lock, same as `set_sync_manager`.

    Callers (resume route, upload route) typically pair this with
    `clear_active_session_for_resume()` at the start of the request so
    A's manager is torn down before B's pipeline even begins.
    """
    global _active_sync_manager, _active_session_cache
    with _globals_lock:
        old_sm = _active_sync_manager
        _active_sync_manager = sync_manager
        _active_session_cache = session_cache
    _stop_and_record(old_sm)


def clear_active_session_for_resume():
    """Install `(None, None)` atomically and tear down the old manager.

    Called at the start of a resume request so mid-resume writes from
    concurrent requests fail-closed (mark_dirty_safe sees None and
    drops the write rather than routing it to the wrong session_dir).
    Equivalent to `install_active_session(None, None)` — exists as a
    named entry point for readability.
    """
    install_active_session(None, None)


def close_active_session():
    """Atomically clear both the sync manager and session cache.

    Holds `_globals_lock` across both swaps so no concurrent request
    can observe a (None, stale_cache) or (stale_sm, None) intermediate
    state. Returns the previous sync manager for the caller's records.

    Unlike `set_sync_manager(None)`, this helper does NOT stop the old
    manager — the caller (e.g. `close_session` route) is expected to
    have already called `stop()` and handled the marker-on-failure
    logic. This lets the close path own its own lifecycle and avoids
    double-stop when set_sync_manager(None) would otherwise re-run
    the stop+marker pipeline on an already-stopped manager.
    """
    global _active_sync_manager, _active_session_cache
    with _globals_lock:
        old = _active_sync_manager
        _active_sync_manager = None
        _active_session_cache = None
    return old
