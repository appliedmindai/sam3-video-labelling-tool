"""GCS storage operations for the segment service.

All functions take bucket_name as the first parameter — this comes from
g.bucket, set per-request from the GCS_BUCKET env var (cloud mode only).
"""
import json
import logging
import os
import threading
import time

from google.cloud import storage as gcs

logger = logging.getLogger(__name__)

# Simple TTL cache for list_sessions
_sessions_cache: dict[str, tuple[float, list[dict]]] = {}
_SESSIONS_CACHE_TTL = 30.0  # seconds

# Single-flight gate for cache misses: when N concurrent callers see an
# expired entry, only the first hits GCS; the rest wait on the per-bucket
# lock and then read the populated cache. Lock hierarchy: leaf — never
# held together with any other lock. See docs/CONCURRENCY_AUDIT.md § Lock hierarchy.
_sessions_cache_locks: dict[str, threading.Lock] = {}
_sessions_cache_locks_guard = threading.Lock()


def _get_client() -> gcs.Client:
    """Lazy GCS client singleton."""
    if not hasattr(_get_client, "_client"):
        _get_client._client = gcs.Client()
    return _get_client._client


def list_sessions(bucket_name: str) -> list[dict]:
    """List all segmentation sessions from GCS.

    Scans top-level prefixes in the bucket and reads each meta.json.
    Results are cached for 30 seconds to avoid redundant GCS API calls.

    Cache misses are single-flighted per bucket: concurrent callers that
    arrive on a TTL boundary only issue one GCS list — the rest wait on
    the per-bucket lock and read the populated cache. Without this, eight
    concurrent `GET /api/video/sessions` calls would each fan out to
    `list_blobs` + per-session `meta.json`/`state.json` downloads.
    """
    now = time.monotonic()
    cached = _sessions_cache.get(bucket_name)
    if cached and (now - cached[0]) < _SESSIONS_CACHE_TTL:
        return cached[1]

    # Acquire (or create) the per-bucket lock under the guard, then drop
    # the guard before doing any I/O. Different buckets never serialise.
    with _sessions_cache_locks_guard:
        lock = _sessions_cache_locks.setdefault(bucket_name, threading.Lock())

    with lock:
        # Re-check: a sibling thread may have populated the cache while we
        # waited on the lock. Use a fresh `now` so a populate that took
        # longer than the TTL is still treated as the freshest answer.
        now = time.monotonic()
        cached = _sessions_cache.get(bucket_name)
        if cached and (now - cached[0]) < _SESSIONS_CACHE_TTL:
            return cached[1]

        client = _get_client()
        bucket = client.bucket(bucket_name)

        blobs = client.list_blobs(bucket, delimiter="/")
        _ = list(blobs)  # consume iterator to populate prefixes
        prefixes = list(blobs.prefixes)

        sessions = []
        for prefix in prefixes:
            session_id = prefix.rstrip("/")
            meta_blob = bucket.blob(f"{session_id}/meta.json")
            if not meta_blob.exists():
                continue
            try:
                meta = json.loads(meta_blob.download_as_text())
            except Exception:
                logger.warning("Failed to read meta.json for session %s", session_id)
                continue

            # Compute disk_size from blob sizes for this session
            session_blobs = list(client.list_blobs(bucket, prefix=f"{session_id}/"))
            disk_size = sum(b.size or 0 for b in session_blobs)

            # Read state.json for class/object counts
            state_blob = bucket.blob(f"{session_id}/state.json")
            state = {}
            try:
                if state_blob.exists():
                    state = json.loads(state_blob.download_as_text())
            except Exception:
                pass

            sessions.append({
                "session_id": session_id,
                "frame_count": meta.get("frame_count", 0),
                "original_name": meta.get("original_name", "Unknown"),
                "fps": meta.get("fps", 2),
                "video_info": meta.get("video_info", {}),
                "md5": meta.get("md5"),
                "created_at": meta.get("created_at"),
                "updated_at": meta.get("updated_at"),
                "disk_size": disk_size,
                "class_count": len(state.get("classes", [])),
                "object_count": len(state.get("objects", [])),
            })

        sessions.sort(key=lambda s: s.get("updated_at") or "", reverse=True)
        _sessions_cache[bucket_name] = (now, sessions)
        return sessions


def invalidate_sessions_cache(bucket_name: str | None = None) -> None:
    """Clear the session list cache. Call after upload or delete."""
    if bucket_name:
        _sessions_cache.pop(bucket_name, None)
    else:
        _sessions_cache.clear()


def find_session_by_md5_gcs(bucket_name: str, md5: str) -> str | None:
    """Check if a video with this MD5 already exists in any GCS session."""
    for session in list_sessions(bucket_name):
        if session.get("md5") == md5:
            return session["session_id"]
    return None


def upload_session_files(bucket_name: str, session_id: str, local_dir: str,
                         rel_paths: list[str]) -> None:
    """Upload specific files from a local session directory to GCS."""
    client = _get_client()
    bucket = client.bucket(bucket_name)

    for rel_path in rel_paths:
        local_file = os.path.join(local_dir, rel_path)
        if not os.path.isfile(local_file):
            continue
        blob = bucket.blob(f"{session_id}/{rel_path}")
        blob.upload_from_filename(local_file)
        logger.debug("Uploaded %s/%s to GCS", session_id, rel_path)


def upload_directory(bucket_name: str, session_id: str, local_dir: str,
                     subdir: str) -> int:
    """Upload all files in a subdirectory to GCS. Returns count uploaded."""
    client = _get_client()
    bucket = client.bucket(bucket_name)

    full_dir = os.path.join(local_dir, subdir)
    if not os.path.isdir(full_dir):
        return 0

    count = 0
    for fname in os.listdir(full_dir):
        fpath = os.path.join(full_dir, fname)
        if not os.path.isfile(fpath):
            continue
        blob = bucket.blob(f"{session_id}/{subdir}/{fname}")
        blob.upload_from_filename(fpath)
        count += 1

    logger.info("Uploaded %d files from %s/%s to GCS", count, session_id, subdir)
    return count


def download_session(bucket_name: str, session_id: str, local_dir: str,
                     on_progress=None) -> None:
    """Download a session from GCS to a local directory.

    Each file is downloaded to a tmp path and published via os.rename so a
    concurrent reader (e.g. GET /frame from a second tab mid-resume) never
    observes a half-written file. The tmp naming matches
    sweep_orphan_tmp_files so crashed downloads are cleaned at startup."""
    import uuid
    client = _get_client()
    bucket = client.bucket(bucket_name)

    os.makedirs(local_dir, exist_ok=True)

    prefix = f"{session_id}/"
    blobs = list(client.list_blobs(bucket, prefix=prefix))
    total = len(blobs)

    for i, blob in enumerate(blobs):
        rel_path = blob.name[len(prefix):]
        local_path = os.path.join(local_dir, rel_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        tmp_path = f"{local_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        try:
            blob.download_to_filename(tmp_path)
            os.rename(tmp_path, local_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise
        if on_progress:
            on_progress(i + 1, total)

    logger.info("Downloaded %d files for session %s from GCS", total, session_id)


def upload_file(bucket_name: str, session_id: str, rel_path: str,
                local_path: str) -> None:
    """Upload a single file to GCS."""
    client = _get_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"{session_id}/{rel_path}")
    blob.upload_from_filename(local_path)


def download_file_if_exists(bucket_name: str, session_id: str, rel_path: str,
                             local_path: str) -> bool:
    """Download a single file from GCS atomically (tmp + os.rename).

    Returns True if the file was downloaded, False if it doesn't exist in GCS.
    Used by DownloadSessionStep to refresh volatile annotation files without
    re-downloading the entire session. Atomic overwrite prevents readers
    from observing a half-written file (race with GET /state during resume)."""
    import uuid
    client = _get_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"{session_id}/{rel_path}")
    if not blob.exists():
        return False
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    tmp_path = f"{local_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        blob.download_to_filename(tmp_path)
        os.rename(tmp_path, local_path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass
        raise
    return True


def stage_unsynced_file(bucket_name: str, session_id: str, rel_path: str,
                        local_path: str) -> None:
    """Upload a dirty local file to the recovery-staging prefix.

    The blob is placed at `<session_id>/.unsynced/<rel_path>`, disjoint
    from the canonical `<session_id>/<rel_path>`. This lets the bytes
    survive scale-to-zero even when the canonical upload failed —
    DownloadSessionStep will promote the staged blob to canonical on
    the next resume.

    Raises on upload failure. Callers log + continue to the next file.
    """
    client = _get_client()
    blob = client.bucket(bucket_name).blob(
        f"{session_id}/.unsynced/{rel_path}"
    )
    blob.upload_from_filename(local_path)
    logger.info(
        "gcs_storage | staged %s to %s/.unsynced/%s",
        local_path, session_id, rel_path,
    )


def copy_staged_to_canonical(bucket_name: str, session_id: str,
                             rel_path: str) -> bool:
    """Promote `<session_id>/.unsynced/<rel_path>` to `<session_id>/<rel_path>`.

    Returns True iff a staged blob existed (and was therefore copied
    and deleted). Returns False if no staged blob was present — the
    caller should fall back to uploading from local disk.

    The copy is unconditional: staging only exists when the normal
    upload failed, so canonical is known-stale. Raises on GCS errors.
    """
    client = _get_client()
    bucket = client.bucket(bucket_name)
    src = bucket.blob(f"{session_id}/.unsynced/{rel_path}")
    if not src.exists():
        return False
    dst_name = f"{session_id}/{rel_path}"
    bucket.copy_blob(src, bucket, dst_name)
    src.delete()
    logger.info(
        "gcs_storage | promoted staged blob to canonical: %s", dst_name,
    )
    return True


def list_staged_rel_paths(bucket_name: str, session_id: str) -> list[str]:
    """Enumerate staged rel_paths under `<session_id>/.unsynced/`.

    Used for the orphan sweep on cold resume: if a staged blob exists
    but no marker references it (container died between staging and
    marker upload), recovery still finds it.
    """
    client = _get_client()
    bucket = client.bucket(bucket_name)
    prefix = f"{session_id}/.unsynced/"
    rel_paths: list[str] = []
    for blob in client.list_blobs(bucket, prefix=prefix):
        name = blob.name
        if name.endswith("/"):
            continue
        rel_paths.append(name[len(prefix):])
    return rel_paths


def delete_staged_blob(bucket_name: str, session_id: str, rel_path: str) -> None:
    """Remove `<session_id>/.unsynced/<rel_path>`. Best-effort."""
    client = _get_client()
    blob = client.bucket(bucket_name).blob(
        f"{session_id}/.unsynced/{rel_path}"
    )
    try:
        blob.delete()
    except Exception:
        # Missing blob is fine; any other error is logged but not fatal.
        logger.warning(
            "gcs_storage | could not delete staged blob %s", blob.name,
            exc_info=True,
        )


def delete_marker_blob(bucket_name: str, session_id: str, marker_name: str) -> None:
    """Remove `<session_id>/<marker_name>` from GCS. Best-effort."""
    client = _get_client()
    blob = client.bucket(bucket_name).blob(f"{session_id}/{marker_name}")
    try:
        blob.delete()
    except Exception:
        logger.warning(
            "gcs_storage | could not delete marker blob %s", blob.name,
            exc_info=True,
        )


def delete_session_gcs(bucket_name: str, session_id: str) -> int:
    """Delete all files for a session from GCS. Returns count deleted."""
    client = _get_client()
    bucket = client.bucket(bucket_name)

    blobs = list(client.list_blobs(bucket, prefix=f"{session_id}/"))
    for blob in blobs:
        blob.delete()

    logger.info("Deleted %d blobs for session %s from GCS", len(blobs), session_id)
    return len(blobs)
