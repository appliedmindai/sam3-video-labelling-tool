"""Background GCS sync manager with dirty-tracking and propagation deferral.

Tracks which session files have changed locally and periodically flushes
them to GCS.  During mask propagation, masks.json writes are deferred
(it's written every frame, hundreds of times per minute) and promoted
to dirty once propagation ends.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Lazy-loaded reference — set on first flush().
# Module-level so tests can patch ``app.services.gcs_sync._upload_file``.
_upload_file = None


def _get_upload_file():
    """Lazy-import upload_file from gcs_storage to avoid circular imports."""
    global _upload_file
    if _upload_file is None:
        from app.services.gcs_storage import upload_file
        _upload_file = upload_file
    return _upload_file


def mark_dirty_safe(rel_path: str) -> None:
    """Best-effort mark_dirty that tolerates a stop/swap race.

    The active sync manager may be stopped between `get_sync_manager()`
    returning it and the caller reaching `mark_dirty`. Once stopped,
    `mark_dirty` raises so the write isn't silently added to a dead
    `_dirty` set (that would never flush). We retry once to pick up the
    next-installed manager; if it's still None, we log and drop the
    write — the session is being torn down and the file is still on
    local disk for the next resume to reconcile.
    """
    from app.config import get_sync_manager
    for _ in range(2):
        sm = get_sync_manager()
        if sm is None:
            return
        try:
            sm.mark_dirty(rel_path)
            return
        except RuntimeError:
            # Manager was stopped between get and mark — loop once to
            # pick up a freshly-installed manager, then give up.
            continue
    logger.info(
        "mark_dirty_safe | no active sync manager for %s; local disk is authoritative",
        rel_path,
    )


@dataclass(frozen=True)
class FlushResult:
    """Outcome of one flush cycle.

    `uploaded` is the count of files successfully pushed to GCS in this
    cycle. `failed` is the set of files that remained dirty (either a
    mid-flight exception or the upload was retried).
    """
    uploaded: int
    failed: frozenset[str]

    @property
    def ok(self) -> bool:
        return not self.failed


class GCSSyncManager:
    """Tracks dirty files and periodically uploads them to GCS."""

    def __init__(self, bucket_name: str, session_id: str, session_dir: str):
        self.bucket_name = bucket_name
        self.session_id = session_id
        self.session_dir = session_dir

        # Lock hierarchy: `GCSSyncManager._lock` is level #6 — innermost
        # on the sync-manager path. No method on this class acquires any
        # other lock in the hierarchy. Upload I/O in `flush()` runs
        # OUTSIDE this lock by design. See
        # docs/TECHNICAL_REPORT.md § Concurrency model.
        self._lock = threading.Lock()
        self._dirty: set[str] = set()
        self._deferred: set[str] = set()
        self._propagating = False
        self._timer: threading.Timer | None = None
        self._stopped = False
        self._interval: float = 60.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mark_dirty(self, rel_path: str) -> None:
        """Mark a file as needing sync to GCS.

        During propagation, masks.json writes are deferred to avoid
        hundreds of redundant uploads.  All other files go straight
        to the dirty set.

        Raises RuntimeError if the manager has been stopped — callers
        that may race with a stop/close should use `mark_dirty_safe()`
        so the write can be routed to the next-installed manager.
        """
        with self._lock:
            if self._stopped:
                raise RuntimeError(
                    f"mark_dirty on stopped GCSSyncManager "
                    f"(session {self.session_id})"
                )
            if self._propagating and rel_path == "masks.json":
                self._deferred.add(rel_path)
            else:
                self._dirty.add(rel_path)

    def set_propagating(self, active: bool) -> None:
        """Toggle propagation mode.

        When propagation ends (active=False), any deferred files are
        promoted to the dirty set so they get picked up by the next
        flush.
        """
        with self._lock:
            self._propagating = active
            if not active:
                self._dirty.update(self._deferred)
                self._deferred.clear()

    def promote_deferred(self) -> None:
        """Promote any deferred files into the dirty set.

        Use this during forced teardowns (SIGTERM, close_session) when the
        propagation thread may not get a chance to run its finally block.
        After this call, flush() and flush_with_retry() will include the
        previously-deferred files.
        """
        with self._lock:
            if self._deferred:
                self._dirty.update(self._deferred)
                self._deferred.clear()
            self._propagating = False

    def has_dirty(self) -> bool:
        """Return True if there are files pending upload."""
        with self._lock:
            return bool(self._dirty)

    def dirty_snapshot(self) -> frozenset[str]:
        """Return a snapshot of currently-dirty files."""
        with self._lock:
            return frozenset(self._dirty)

    def flush(self) -> FlushResult:
        """Upload all dirty files to GCS.

        Returns a FlushResult. Successful uploads are removed from the dirty
        set. Failures remain dirty so a subsequent flush (periodic tick, retry,
        or an explicit flush_with_retry) can try again. Files that no longer
        exist on disk are silently removed from the dirty set.

        On successful canonical upload, any existing staged `.unsynced/<rel>`
        blob for the same file is deleted best-effort. This prevents a stale
        staged blob (left by a prior teardown that never cleaned up) from
        being promoted over current canonical on a later cold resume.
        """
        with self._lock:
            to_upload = self._dirty.copy()

        if not to_upload:
            return FlushResult(uploaded=0, failed=frozenset())

        upload_file = _get_upload_file()

        uploaded = 0
        succeeded: set[str] = set()
        failed: set[str] = set()
        uploaded_rel_paths: list[str] = []

        for rel_path in to_upload:
            local_path = os.path.join(self.session_dir, rel_path)
            if not os.path.isfile(local_path):
                logger.debug("Skipping %s — file does not exist", rel_path)
                succeeded.add(rel_path)  # Remove from dirty — nothing to upload
                continue
            try:
                upload_file(self.bucket_name, self.session_id, rel_path, local_path)
                uploaded += 1
                succeeded.add(rel_path)
                uploaded_rel_paths.append(rel_path)
            except Exception:
                logger.warning("Failed to upload %s, will retry", rel_path, exc_info=True)
                failed.add(rel_path)

        # Only remove files that were successfully uploaded (or no longer exist)
        if succeeded:
            with self._lock:
                self._dirty -= succeeded

        # Best-effort: drop any matching `.unsynced/<rel>` staged blob for
        # files we just uploaded canonically. Without this, stale staged
        # blobs from a previous teardown persist and can be promoted over
        # current canonical on the next cold resume (H3).
        if uploaded_rel_paths:
            try:
                from app.services import gcs_storage
                for rel_path in uploaded_rel_paths:
                    try:
                        gcs_storage.delete_staged_blob(
                            self.bucket_name, self.session_id, rel_path,
                        )
                    except Exception:
                        logger.debug(
                            "flush | stale staged blob cleanup failed for %s",
                            rel_path, exc_info=True,
                        )
            except Exception:
                # Import failure (test stubs, etc.) — don't let it kill flush
                logger.debug(
                    "flush | could not import gcs_storage for staged cleanup",
                    exc_info=True,
                )

        logger.info(
            "GCS sync: uploaded %d/%d files, %d failed (session %s)",
            uploaded, len(to_upload), len(failed), self.session_id,
        )
        return FlushResult(uploaded=uploaded, failed=frozenset(failed))

    def flush_with_retry(
        self,
        max_attempts: int = 3,
        initial_backoff_s: float = 1.0,
        backoff_factor: float = 2.0,
    ) -> FlushResult:
        """Flush, retrying failures with exponential backoff.

        Returns the final FlushResult — `failed` is the set of files that
        remained dirty after all attempts. The sync manager is NOT stopped
        or discarded; callers decide what to do with persistent failures.
        """
        total_uploaded = 0
        last_result = FlushResult(uploaded=0, failed=frozenset())
        backoff = initial_backoff_s
        for attempt in range(max_attempts):
            if attempt > 0:
                time.sleep(backoff)
                backoff *= backoff_factor
            result = self.flush()
            total_uploaded += result.uploaded
            last_result = result
            if result.ok:
                break
            logger.info(
                "flush_with_retry: attempt %d/%d left %d files dirty for session %s",
                attempt + 1, max_attempts, len(result.failed), self.session_id,
            )
        return FlushResult(uploaded=total_uploaded, failed=last_result.failed)

    # ------------------------------------------------------------------
    # Periodic timer
    # ------------------------------------------------------------------

    def start(self, interval: float = 60.0) -> None:
        """Start the periodic sync timer."""
        with self._lock:
            if self._stopped:
                raise RuntimeError("Cannot start a stopped GCSSyncManager")
            self._interval = interval
        self._schedule()

    def stop(self) -> FlushResult:
        """Stop the periodic timer and flush any remaining dirty files.

        Idempotent — calling stop twice is safe. Once stopped, subsequent
        _tick invocations (if one was already in flight) will NOT re-arm
        a new timer.

        Promotes any deferred files (e.g. masks.json that was being written
        during propagation) into the dirty set before flushing. Without
        this, a shutdown while propagation is in-flight would silently
        drop every mask produced during that propagation run — the
        propagation thread is daemon=True and can be killed mid-frame
        before it ever reaches the finally block that normally promotes
        deferred entries.

        Returns the final FlushResult. If files are still dirty, the caller
        is responsible for deciding how to handle them (retry, surface to
        user, etc.) — this method does NOT retry.
        """
        with self._lock:
            self._stopped = True
            if self._timer is not None:
                self._timer.cancel()
                self._timer = None
            # Promote deferred to dirty — nothing more will come in, so
            # there is no reason to hold back masks.json uploads.
            if self._deferred:
                self._dirty.update(self._deferred)
                self._deferred.clear()
            self._propagating = False
        return self.flush()

    def _schedule(self) -> None:
        with self._lock:
            if self._stopped:
                return
            self._timer = threading.Timer(self._interval, self._tick)
            self._timer.daemon = True
            self._timer.start()

    def _tick(self) -> None:
        try:
            self.flush()
        except Exception:
            logger.error("Periodic GCS sync failed", exc_info=True)
        finally:
            self._schedule()
