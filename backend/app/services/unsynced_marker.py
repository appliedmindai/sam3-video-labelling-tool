"""Crash-recovery marker for GCS uploads that failed at close/SIGTERM.

When close_session or SIGTERM cannot flush all dirty files to GCS after
retries, it writes `<session_dir>/.unsynced.json` listing the files that
still have un-uploaded local changes. The next DownloadSessionStep checks
for this marker and:
1. Skips downloading those files (local is authoritative)
2. Attempts to upload them to GCS (recovery)
3. Deletes the marker on success

This turns transient GCS outages at session close from silent data loss
into deferred recovery on the next session open."""

import json
import logging
import os
import time

logger = logging.getLogger(__name__)

MARKER_FILENAME = ".unsynced.json"


def write_marker(session_dir: str, unsynced_files: list[str], reason: str = "") -> None:
    """Write or update the unsynced marker with the list of dirty files."""
    if not unsynced_files:
        return
    path = os.path.join(session_dir, MARKER_FILENAME)
    try:
        os.makedirs(session_dir, exist_ok=True)
        payload = {
            "files": sorted(set(unsynced_files)),
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "reason": reason,
        }
        # Best-effort; if marker write fails we log and move on (the next
        # resume will just refresh from GCS as usual).
        tmp_path = f"{path}.tmp.{os.getpid()}"
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)
        os.rename(tmp_path, path)
        logger.info("unsynced | wrote marker for %d files at %s", len(payload["files"]), path)
    except Exception:
        logger.error("unsynced | failed to write marker at %s", path, exc_info=True)


def read_marker(session_dir: str) -> list[str] | None:
    """Return the list of unsynced files from a marker, or None if absent/invalid."""
    path = os.path.join(session_dir, MARKER_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
        files = payload.get("files")
        if isinstance(files, list) and all(isinstance(x, str) for x in files):
            return files
    except Exception:
        logger.warning("unsynced | could not read marker at %s", path, exc_info=True)
    return None


def clear_marker(session_dir: str) -> None:
    """Delete the marker file (recovery succeeded)."""
    path = os.path.join(session_dir, MARKER_FILENAME)
    try:
        os.unlink(path)
        logger.info("unsynced | cleared marker at %s", path)
    except FileNotFoundError:
        pass
    except Exception:
        logger.warning("unsynced | failed to clear marker at %s", path, exc_info=True)


def upload_marker_to_gcs(bucket: str, session_id: str, session_dir: str) -> None:
    """Best-effort: copy the local .unsynced.json marker into the session's
    GCS prefix so it survives container scale-to-zero.

    No-op if no local marker exists. Errors are logged and swallowed —
    the local marker is still authoritative if the container is warm
    on the next resume.
    """
    path = os.path.join(session_dir, MARKER_FILENAME)
    if not os.path.isfile(path):
        return
    try:
        from app.services import gcs_storage
        gcs_storage.upload_file(bucket, session_id, MARKER_FILENAME, path)
        logger.info(
            "unsynced | uploaded marker to %s/%s", session_id, MARKER_FILENAME,
        )
    except Exception:
        logger.error(
            "unsynced | failed to upload marker to GCS for session %s",
            session_id, exc_info=True,
        )


def persist_unsynced(
    bucket: str | None,
    session_id: str,
    session_dir: str,
    failed: list[str],
    reason: str,
) -> list[str]:
    """Persist the list of un-uploaded files into durable storage.

    Effects (in order):
    1. For each failed file that exists locally, attempt a staging
       upload to `<session_id>/.unsynced/<rel_path>` in GCS (if `bucket`
       is provided). The staged blob is a disjoint namespace — it does
       not overwrite canonical. Successful staging lands the bytes on
       GCS so they survive scale-to-zero.
    2. Write a local marker listing what we tried to save. Prefer the
       list of successfully-staged files when available (they are the
       ones the next resume can actually recover from GCS); if bucket
       is None or nothing staged, fall back to the original `failed`
       list so warm-container recovery (local disk) still applies.
    3. Upload the local marker to `<session_id>/.unsynced.json` on GCS
       so a cold container can still find the recovery index.

    Returns the sublist of `failed` that was successfully staged. Callers
    may log this for visibility.

    This is the single entry point for all teardown-time marker writes:
    SIGTERM handler, close_session route, beacon /flush, and
    set_sync_manager swap.
    """
    if not failed:
        return []

    staged: list[str] = []
    failed_to_stage: list[str] = []
    if bucket:
        from app.services import gcs_storage
        for rel in failed:
            local = os.path.join(session_dir, rel)
            if not os.path.isfile(local):
                logger.debug(
                    "unsynced | skip staging %s (not on disk)", rel,
                )
                continue
            try:
                gcs_storage.stage_unsynced_file(bucket, session_id, rel, local)
                staged.append(rel)
            except Exception:
                logger.error(
                    "unsynced | stage upload failed for %s (session %s)",
                    rel, session_id, exc_info=True,
                )
                failed_to_stage.append(rel)

    # The marker lists every dirty file that is still recoverable from
    # somewhere. Staged files recover from `.unsynced/<rel>` blobs on a
    # cold-container resume. Files that raised during staging are still
    # on local disk (they passed the isfile check), so a warm-container
    # resume recovers them via the mark_dirty_safe path. Files that were
    # skipped as missing-from-disk are NOT listed — they are not
    # recoverable anywhere. When bucket is None (local-only mode) we
    # list the full failed set so the next resume's recovery loop can
    # upload from disk.
    if bucket:
        marker_files = sorted(set(staged) | set(failed_to_stage))
    else:
        marker_files = sorted(failed)
    write_marker(session_dir, marker_files, reason=reason)
    if bucket:
        upload_marker_to_gcs(bucket, session_id, session_dir)
    return staged
