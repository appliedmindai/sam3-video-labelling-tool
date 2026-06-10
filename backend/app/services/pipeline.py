"""Background pipeline for video processing stages.

ServiceState is a frozen (immutable) dataclass representing what the service
is currently doing. All transitions go through SAM3Service._state_lock and
replace the entire object via dataclasses.replace().
"""

from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, replace
from typing import Callable, Literal, Protocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServiceState:
    """Immutable snapshot of the service's current phase.

    One state for the whole service — not per-session. There is one GPU,
    one user, one thing happening at a time.
    """

    phase: Literal["idle", "extracting", "initializing", "ready", "error"] = "idle"
    session_id: str | None = None
    video_name: str | None = None
    progress: float = 0.0
    error: str | None = None
    cancel_requested: bool = False
    frame_count: int | None = None


class PipelineStep(Protocol):
    """Protocol for pipeline steps. Each step has a phase name and a run method."""

    phase: str

    def run(
        self,
        on_progress: Callable[[float], None],
        cancel_event: threading.Event,
    ) -> dict | None:
        """Execute the step.

        Args:
            on_progress: Call with 0.0-1.0 to report progress.
            cancel_event: Check .is_set() periodically; if True, clean up and return.

        Returns:
            Optional dict of fields to merge into ServiceState (e.g. {"frame_count": 600}).
        """
        ...


class ExtractFramesStep:
    """Pipeline step: extract video frames using ffmpeg with progress."""

    phase = "extracting"

    def __init__(self, session_id: str, fps: int = 2, max_dim: int = 2048, bucket: str | None = None):
        self._session_id = session_id
        self._fps = fps
        self._max_dim = max_dim
        self._bucket = bucket

    def run(
        self,
        on_progress: Callable[[float], None],
        cancel_event: threading.Event,
    ) -> dict | None:
        from app.config import SESSIONS_DIR
        from app.services.video_processor import extract_frames_async, get_video_info

        session_dir = os.path.join(SESSIONS_DIR, self._session_id)
        video_path = os.path.join(session_dir, "video.mp4")
        frames_dir = os.path.join(session_dir, "frames")

        frame_count = extract_frames_async(
            video_path,
            frames_dir,
            fps=self._fps,
            max_dim=self._max_dim,
            on_progress=on_progress,
            cancel_event=cancel_event,
        )

        if frame_count == 0:
            return None  # cancelled

        # Write meta.json
        video_info = get_video_info(video_path)
        meta = {
            "fps": self._fps,
            "frame_count": frame_count,
            "video_info": video_info,
        }
        meta_path = os.path.join(session_dir, "meta.json")
        # Preserve existing fields (md5, original_name) if meta already exists
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                existing = json.load(f)
            existing.update(meta)
            meta = existing
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Cloud mode: upload session files to GCS after extraction
        if self._bucket:
            from app.config import install_active_session
            from app.services import gcs_storage
            from app.services.gcs_sync import GCSSyncManager
            from app.services.session_cache import SessionCache

            gcs_storage.upload_session_files(
                self._bucket, self._session_id, session_dir,
                ["video.mp4", "meta.json"],
            )
            gcs_storage.upload_directory(
                self._bucket, self._session_id, session_dir, "frames",
            )

            sm = GCSSyncManager(self._bucket, self._session_id, session_dir)
            # Install (sm, cache) atomically. Previously we called
            # set_sync_manager alone and never installed a SessionCache
            # for the upload flow — route reads fell back to the disk
            # path with no cache invalidation, and a concurrent request
            # could observe (sm_new, cache_old_session). Install both now
            # so the invariant "sm + cache move together" holds on the
            # upload path too.
            install_active_session(sm, SessionCache(session_dir))
            sm.start()

        return {"frame_count": frame_count}


class InitSessionStep:
    """Pipeline step: initialize SAM3 session (load model + warmup)."""

    phase = "initializing"

    def __init__(self, session_id: str):
        self._session_id = session_id

    def run(
        self,
        on_progress: Callable[[float], None],
        cancel_event: threading.Event,
    ) -> dict | None:
        from app.config import SESSIONS_DIR
        from app.services.sam3_service import SAM3Service

        sam = SAM3Service()
        frames_dir = os.path.join(SESSIONS_DIR, self._session_id, "frames")

        if not os.path.isdir(frames_dir):
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

        on_progress(0.0)
        result = sam.init_session(self._session_id, frames_dir, on_progress=on_progress)

        # Return frame_count so resume pipelines (which skip extraction)
        # populate ServiceState.frame_count for the frontend.
        if result and "num_frames" in result:
            return {"frame_count": result["num_frames"]}
        return None


class DownloadSessionStep:
    """Pipeline step: download session from GCS (cloud mode only)."""

    phase = "initializing"

    def __init__(self, session_id: str, bucket: str):
        self._session_id = session_id
        self._bucket = bucket

    def _read_remote_marker_files(self) -> set[str] | None:
        """Fetch `<session_id>/.unsynced.json` from GCS and return the set
        of rel_paths it lists. Returns None when no marker exists (or
        cannot be read) — the caller must treat this as "no recovery
        work authorized", NOT as "promote everything staged".

        The marker is the single authoritative index of which staged
        blobs are newer than canonical. A staged blob without a marker
        reference is stale-or-orphaned and MUST NOT be promoted over
        canonical (H3): a concurrent tab on a fresh container may have
        just uploaded a newer canonical, and promoting a leftover
        staged blob on top would corrupt it.
        """
        import tempfile
        from app.services import gcs_storage
        from app.services.unsynced_marker import (
            MARKER_FILENAME, read_marker,
        )
        with tempfile.TemporaryDirectory() as tmpd:
            try:
                found = gcs_storage.download_file_if_exists(
                    self._bucket, self._session_id, MARKER_FILENAME,
                    os.path.join(tmpd, MARKER_FILENAME),
                )
            except Exception:
                logger.warning(
                    "download_step | failed to fetch remote marker for promotion gating",
                    exc_info=True,
                )
                return None
            if not found:
                return None
            files = read_marker(tmpd)
            if files is None:
                return None
            return set(files)

    def _promote_staged_blobs(
        self, cancel_event: threading.Event,
    ) -> set[str] | None:
        """Promote staged `.unsynced/<rel>` blobs to canonical in GCS.

        Scoped to files referenced by the GCS-stored `.unsynced.json`
        marker. No marker → no promotion. This closes H3: a staged blob
        without a marker is untrusted (could be a leftover orphan from
        a previous container that died after staging but before it
        persisted a marker — meanwhile a fresh container may have
        uploaded a newer canonical that the orphan sweep would clobber).

        Returns the set of rel_paths that were promoted to canonical.
        Returns None if cancelled.
        """
        from app.services import gcs_storage
        promoted: set[str] = set()

        # List staged blobs first — if there's nothing staged, we skip
        # the marker read entirely (saves a GCS round-trip on the hot
        # path where no recovery work exists).
        try:
            staged = gcs_storage.list_staged_rel_paths(
                self._bucket, self._session_id,
            )
        except Exception:
            logger.warning(
                "download_step | list_staged_rel_paths failed",
                exc_info=True,
            )
            return promoted

        if not staged:
            return promoted

        # Staged blobs exist — only promote those authorized by the
        # remote marker. No marker → no promotion (orphan sweep was the
        # data-loss vector this fix closes).
        marker_files = self._read_remote_marker_files()
        if not marker_files:
            logger.warning(
                "download_step | %d staged blob(s) present but no remote marker — "
                "leaving in place to avoid clobbering canonical: %s",
                len(staged), sorted(staged),
            )
            return promoted

        authorized = [rel for rel in staged if rel in marker_files]
        skipped = [rel for rel in staged if rel not in marker_files]
        if skipped:
            logger.warning(
                "download_step | %d staged blob(s) not in marker — leaving in place: %s",
                len(skipped), sorted(skipped),
            )

        for rel_path in authorized:
            if cancel_event.is_set():
                logger.info("download_step | cancelled during staging promotion")
                return None
            try:
                if gcs_storage.copy_staged_to_canonical(
                    self._bucket, self._session_id, rel_path,
                ):
                    promoted.add(rel_path)
            except Exception:
                logger.warning(
                    "download_step | failed to promote staged %s",
                    rel_path, exc_info=True,
                )
        if promoted:
            logger.info(
                "download_step | promoted %d staged blob(s) to canonical: %s",
                len(promoted), sorted(promoted),
            )
        return promoted

    def _pull_remote_marker_if_missing(self, session_dir: str) -> None:
        """Fetch `<session_id>/.unsynced.json` from GCS when local is absent.

        Cold-container edge case: the session_dir may already be
        hydrated (warm branch), but the .unsynced.json marker was
        written by a previous process whose local disk is gone. The
        marker is in GCS — pull it so the local recovery loop has a
        work list.
        """
        from app.services import gcs_storage
        from app.services.unsynced_marker import MARKER_FILENAME

        local = os.path.join(session_dir, MARKER_FILENAME)
        if os.path.isfile(local):
            return
        try:
            gcs_storage.download_file_if_exists(
                self._bucket, self._session_id, MARKER_FILENAME, local,
            )
        except Exception:
            logger.warning(
                "download_step | could not fetch remote marker",
                exc_info=True,
            )

    def _cleanup_local_staging_dir(self, session_dir: str) -> None:
        """Remove the local `.unsynced/` subdir after a full download.

        download_session() pulls every blob in the session prefix,
        including any `.unsynced/<rel>` staging blobs we failed to
        delete after promotion. The local files are unused — canonical
        copies are the source of truth — and would confuse any future
        code that walks session_dir.
        """
        staging_dir = os.path.join(session_dir, ".unsynced")
        if not os.path.isdir(staging_dir):
            return
        try:
            import shutil
            shutil.rmtree(staging_dir)
        except Exception:
            logger.warning(
                "download_step | could not remove local staging dir %s",
                staging_dir, exc_info=True,
            )

    def run(
        self,
        on_progress: Callable[[float], None],
        cancel_event: threading.Event,
    ) -> dict | None:
        from app.config import SESSIONS_DIR, SEGMENT_MODE, get_session_cache
        from app.services import gcs_storage
        from app.services.atomic_write import sweep_orphan_tmp_files
        from app.services.unsynced_marker import (
            read_marker, clear_marker, write_marker, upload_marker_to_gcs,
            MARKER_FILENAME,
        )

        session_dir = os.path.join(SESSIONS_DIR, self._session_id)
        frames_dir = os.path.join(session_dir, "frames")
        on_progress(0.0)

        if cancel_event.is_set():
            logger.info("download_step | cancelled before start")
            return None

        # R41: sweep stale *.tmp.* orphans from the session dir before we
        # touch it. The sync manager isn't installed for this session yet
        # and no propagation is running, so there is no live writer to race.
        removed = sweep_orphan_tmp_files(session_dir)
        if removed:
            logger.info(
                "download_step | swept %d orphan tmp file(s) in %s",
                removed, session_dir,
            )

        # Step 1: GCS-level staging recovery — runs before any local
        # download so that canonical blobs already reflect the
        # authoritative bytes by the time download_session runs.
        # Promotes `<session_id>/.unsynced/<rel_path>` → `<session_id>/<rel_path>`
        # for every staged blob (marker-indexed OR orphan).
        promoted = self._promote_staged_blobs(cancel_event)
        if promoted is None:  # cancelled
            return None

        # Decide branch by frames presence, not session_dir presence.
        # resume_session creates session_dir ahead of time (for SessionCache),
        # so session_dir alone is not a reliable signal of local hydration —
        # frames are the immutable artifact whose absence actually indicates
        # a cold container that needs a full GCS pull.
        #
        # ALSO: if meta.json claims N frames but the local dir has <N frames,
        # assume a prior download was killed mid-way (SIGTERM during first
        # resume) and force a full re-download. Using partial frames leaves
        # SAM3 in a broken state on the next init.
        needs_full_download = (
            not os.path.isdir(frames_dir) or not os.listdir(frames_dir)
        )
        if not needs_full_download:
            # Verify local frame count matches meta.json (if meta exists).
            # Missing meta is its own signal that we didn't finish downloading.
            meta_path = os.path.join(session_dir, "meta.json")
            if not os.path.exists(meta_path):
                logger.info(
                    "download_step | meta.json missing — forcing full re-download"
                )
                needs_full_download = True
            else:
                try:
                    with open(meta_path) as f:
                        expected = json.load(f).get("frame_count")
                    if expected is not None:
                        actual = len([
                            f for f in os.listdir(frames_dir)
                            if f.endswith(".jpg")
                        ])
                        if actual < expected:
                            logger.warning(
                                "download_step | partial frames detected "
                                "(%d/%d) — forcing full re-download",
                                actual, expected,
                            )
                            needs_full_download = True
                except Exception:
                    logger.warning(
                        "download_step | could not verify frame count",
                        exc_info=True,
                    )

        if needs_full_download:
            gcs_storage.download_session(self._bucket, self._session_id, session_dir)
            if cancel_event.is_set():
                logger.info("download_step | cancelled after full download")
                return None
            # download_session pulled the whole prefix including any
            # `.unsynced/` staging blobs we failed to delete after
            # promotion. Remove the local copy so it doesn't confuse
            # later code — the canonical copies (already promoted in
            # step 1) are all we need.
            self._cleanup_local_staging_dir(session_dir)
            # No local-marker recovery needed: staging was already
            # promoted canonically on GCS, so the fresh download
            # contains the authoritative bytes.
        else:
            # Warm branch: session_dir is already hydrated locally.
            # Fetch the remote marker (if any) so a container that was
            # recycled mid-edit can still see what the previous owner
            # failed to upload.
            self._pull_remote_marker_if_missing(session_dir)

            # Recovery step: re-upload any marker-listed local file
            # (staging promotion above is a superset, but some files
            # may have been written locally AFTER staging — warm
            # containers upload-from-disk covers that path).
            unsynced = read_marker(session_dir) or []
            recovered: set[str] = set(promoted)  # already canonical
            for rel_path in unsynced:
                if cancel_event.is_set():
                    logger.info("download_step | cancelled mid-recovery at %s", rel_path)
                    return None
                if rel_path in recovered:
                    continue
                local = os.path.join(session_dir, rel_path)
                if not os.path.isfile(local):
                    continue
                try:
                    gcs_storage.upload_file(self._bucket, self._session_id, rel_path, local)
                    recovered.add(rel_path)
                    logger.info("download_step | recovered %s from marker", rel_path)
                except Exception:
                    logger.warning("download_step | failed to recover %s", rel_path, exc_info=True)
            if unsynced:
                remaining = [r for r in unsynced if r not in recovered]
                if remaining:
                    # Partial recovery: rewrite the marker (local + remote)
                    # with only the still-unrecovered files. Leaving the
                    # original marker in place would make the next resume
                    # re-upload already-recovered files from stale local
                    # disk, silently overwriting any newer remote version
                    # another client may have written in the interim.
                    write_marker(
                        session_dir, remaining, reason="partial_recovery",
                    )
                    upload_marker_to_gcs(
                        self._bucket, self._session_id, session_dir,
                    )
                    logger.info(
                        "download_step | partial recovery: %d/%d recovered, "
                        "marker rewritten with remaining: %s",
                        len(unsynced) - len(remaining), len(unsynced),
                        sorted(remaining),
                    )
                else:
                    clear_marker(session_dir)
                    gcs_storage.delete_marker_blob(
                        self._bucket, self._session_id, MARKER_FILENAME,
                    )

            # Frames are present locally — refresh the small annotation files
            # so we don't serve stale state after container restart or
            # cross-device edits. Frames and video.mp4 are immutable so we
            # only refresh the three volatile files. Skip any file we just
            # uploaded (local is already consistent with GCS).
            for rel_path in ("state.json", "masks.json", "prompts.json"):
                if rel_path in recovered:
                    continue
                if cancel_event.is_set():
                    logger.info("download_step | cancelled mid-refresh at %s", rel_path)
                    return None
                try:
                    local = os.path.join(session_dir, rel_path)
                    refreshed = gcs_storage.download_file_if_exists(
                        self._bucket, self._session_id, rel_path, local,
                    )
                    if refreshed:
                        # Invalidate any cached copy of this file so the next
                        # load() re-reads from disk.
                        cache = get_session_cache()
                        if cache is not None and cache.session_dir == session_dir:
                            cache.invalidate(rel_path)
                except Exception:
                    logger.warning("download_step | failed to refresh %s", rel_path, exc_info=True)
        on_progress(0.2)

        # Sync manager installation moved to the route's on_complete
        # callback (B5) — pipeline no longer mutates globals. This
        # closes the two-tab race where tab A's mark_dirty could be
        # routed to sm_B while cache_A was still serving reads.
        on_progress(0.3)

        # Return video_name from meta.json so ServiceState shows the
        # filename instead of the session UUID (the resume route may not
        # have had access to meta.json before the download).
        meta_path = os.path.join(session_dir, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            name = meta.get("original_name")
            if name:
                logger.info("download_step | resolved video_name=%s from meta.json", name)
                return {"video_name": name}
            else:
                logger.warning("download_step | meta.json missing original_name: %s", list(meta.keys()))
        else:
            logger.warning("download_step | meta.json not found at %s", meta_path)
        return None
