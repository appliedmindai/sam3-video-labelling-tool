import logging
import os
import time

from flask import Flask, g, request
from flask_cors import CORS

from app.logging_config import configure_logging


def _cleanup_partial_sessions():
    """Remove session directories left in a broken state by container kills."""
    import shutil
    from app.config import SESSIONS_DIR
    from app.services.atomic_write import sweep_orphan_tmp_files

    if not os.path.isdir(SESSIONS_DIR):
        return

    logger = logging.getLogger(__name__)
    for name in os.listdir(SESSIONS_DIR):
        session_dir = os.path.join(SESSIONS_DIR, name)
        if not os.path.isdir(session_dir):
            continue
        video_path = os.path.join(session_dir, "video.mp4")
        frames_dir = os.path.join(session_dir, "frames")
        state_path = os.path.join(session_dir, "state.json")

        # Skip sessions that have been used for annotation (have state.json)
        if os.path.exists(state_path):
            continue

        # Remove sessions with video but no frames (extraction was interrupted)
        if os.path.exists(video_path) and not os.path.isdir(frames_dir):
            logger.info("startup | removing partial session %s (no frames)", name)
            shutil.rmtree(session_dir, ignore_errors=True)
            continue

        # Remove sessions with frames dir but zero frames
        if os.path.isdir(frames_dir):
            frame_count = len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
            if frame_count == 0:
                logger.info("startup | removing partial session %s (zero frames)", name)
                shutil.rmtree(session_dir, ignore_errors=True)

    # R41: sweep stale *.tmp.* files left by SIGKILL/OOM between json.dump
    # and os.rename in atomic_json_dump (and equivalent tmp paths in
    # gcs_storage.download_file_if_exists / unsynced_marker.write_marker).
    # Safe here: startup runs before any request thread exists.
    removed = sweep_orphan_tmp_files(SESSIONS_DIR)
    if removed:
        logger.info("startup | swept %d orphan tmp file(s) under %s", removed, SESSIONS_DIR)


def _sigterm_flush_and_clear_globals() -> None:
    """Flush the active sync manager, write an .unsynced marker on failure,
    stop the timer, and then atomically clear the global (sync_manager,
    session_cache) slots.

    Clearing the globals is the critical step that distinguishes this from
    an ordinary `stop()` — without it, any in-flight request that reads
    `_active_sync_manager` during Cloud Run's SIGTERM grace window (up to
    ~10s) would see a stopped manager and drop writes via mark_dirty_safe.
    """
    logger = logging.getLogger(__name__)
    from app.config import get_sync_manager, close_active_session
    sm = get_sync_manager()
    if sm is None:
        return
    sm.promote_deferred()
    result = sm.flush_with_retry(
        max_attempts=3, initial_backoff_s=0.5, backoff_factor=2.0,
    )
    if not result.ok:
        from app.services.unsynced_marker import persist_unsynced
        persist_unsynced(
            sm.bucket_name, sm.session_id, sm.session_dir,
            sorted(result.failed), reason="sigterm_flush_failed",
        )
        logger.error(
            "sigterm | %d files failed to flush, wrote marker",
            len(result.failed),
        )
    # stop() runs an internal final flush that may pick up a last-moment
    # mark_dirty from an in-flight worker thread (narrow window between
    # flush_with_retry returning and stop() acquiring the lock). Honour
    # that result: if stop's internal flush fails during SIGTERM, write
    # a marker so the delta is recoverable on next resume. Mirrors the
    # close_session path in routes/segment.py.
    stop_result = sm.stop()
    if not stop_result.ok:
        from app.services.unsynced_marker import persist_unsynced
        persist_unsynced(
            sm.bucket_name, sm.session_id, sm.session_dir,
            sorted(stop_result.failed), reason="sigterm_stop_failed",
        )
        logger.error(
            "sigterm | stop() left %d file(s) dirty, wrote marker",
            len(stop_result.failed),
        )
    close_active_session()


def create_app():
    configure_logging()

    logger = logging.getLogger(__name__)

    app = Flask(__name__)
    CORS(app)
    app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

    # Cloud mode stores sessions in a single GCS bucket (GCS_BUCKET env var).
    # Routes read it from g.bucket on every request; None in local mode.
    from app import config as app_config

    @app.before_request
    def _set_bucket_context():
        if os.environ.get("SEGMENT_MODE") == "cloud":
            g.bucket = app_config.GCS_BUCKET or None
        else:
            g.bucket = None

    # Health endpoint — no auth, no SAM3 lock, always responds instantly.
    # Used by frontend heartbeat for keepalive + boot detection.
    from flask import jsonify
    from app.config import BOOT_ID, BOOT_TIME

    @app.route("/api/health")
    def health():
        return jsonify({
            "ok": True,
            "boot_id": BOOT_ID,
            "uptime_s": int(time.time() - BOOT_TIME),
        })

    from app.routes import video_bp, session_bp, segment_bp, export_bp, benchmark_bp
    from app.routes.status import status_bp
    app.register_blueprint(video_bp, url_prefix="/api/video")
    app.register_blueprint(session_bp, url_prefix="/api/session")
    app.register_blueprint(segment_bp, url_prefix="/api/segment")
    app.register_blueprint(export_bp, url_prefix="/api/export")
    app.register_blueprint(benchmark_bp, url_prefix="/api/benchmark")
    app.register_blueprint(status_bp, url_prefix="/api")

    # --- Request lifecycle logging (cloud mode only) ---
    is_cloud = os.environ.get("SEGMENT_MODE") == "cloud"

    if is_cloud:
        @app.before_request
        def _start_timer():
            g.start_time = time.monotonic()
            # Extract session_id from URL path or request JSON for log context
            if request.view_args and "session_id" in request.view_args:
                g.session_id = request.view_args["session_id"]

        @app.after_request
        def _log_request(response):
            # Skip health checks and static files
            if request.path in ("/api/health", "/api/benchmark/health"):
                return response

            duration_ms = int(
                (time.monotonic() - getattr(g, "start_time", time.monotonic())) * 1000
            )
            logger.info(
                "%s %s %d %dms",
                request.method,
                request.path,
                response.status_code,
                duration_ms,
            )
            return response

    # Startup cleanup: remove partial sessions left by killed containers
    _cleanup_partial_sessions()

    # SIGTERM handler: gracefully stop pipeline on container shutdown
    import signal

    def _handle_sigterm(signum, frame):
        """Flush dirty GCS files and cancel background work before Cloud Run kills us."""
        logger = logging.getLogger(__name__)
        logger.info("SIGTERM received — cancelling background work and flushing GCS sync")

        # 1. Cancel active propagation (if any) so persist_fn stops marking files dirty.
        #    Then join the propagation thread briefly so its `finally` block has a chance
        #    to send SSE sentinels (graceful client close) and toggle set_propagating(False)
        #    itself, rather than racing sys.exit(0).
        try:
            from app.services.sam3_service import SAM3Service
            sam = SAM3Service()
            state = sam.get_service_state()
            if state.session_id:
                sam.cancel_propagation(state.session_id)
                if not sam.join_propagation(state.session_id, timeout=2):
                    logger.warning(
                        "sigterm | propagation thread did not exit within 2s; "
                        "promote_deferred will still run",
                    )
        except Exception:
            logger.warning("sigterm | cancel_propagation failed", exc_info=True)

        # 2. Cancel the pipeline (extraction / model init).
        try:
            from app.services.sam3_service import SAM3Service
            sam = SAM3Service()
            sam.cancel_pipeline()
            if sam._pipeline_thread is not None:
                sam._pipeline_thread.join(timeout=3)
        except Exception:
            logger.warning("sigterm | cancel_pipeline failed", exc_info=True)

        # 2b. R33: kill any ffmpeg subprocess directly, bypassing the
        # 0.5s cancel-event poll inside extract_frames_async. If the pipeline
        # thread's join timed out above, ffmpeg may still be writing frames —
        # terminating it here guarantees no orphan survives into sys.exit(0).
        try:
            from app.services.video_processor import kill_all_ffmpeg_procs
            kill_all_ffmpeg_procs(timeout=2.0)
        except Exception:
            logger.warning("sigterm | kill_all_ffmpeg_procs failed", exc_info=True)

        # 3. Flush dirty files, write .unsynced marker on failure, stop
        # the timer, and clear global slots. Clearing globals is load-bearing:
        # Cloud Run gives up to 10s between SIGTERM and SIGKILL, so any
        # in-flight request that reaches get_sync_manager after this handler
        # finishes must see None rather than a stopped manager.
        try:
            _sigterm_flush_and_clear_globals()
        except Exception:
            logger.error("sigterm | flush failed", exc_info=True)

        import sys
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)

    return app
