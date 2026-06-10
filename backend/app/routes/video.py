import hashlib
import json
import os
import shutil
import time
import uuid
from flask import Blueprint, request, jsonify, send_file, g
from app.config import SESSIONS_DIR, FRAME_EXTRACTION_FPS, SEGMENT_MODE
from app.services.session_manager import load_state, find_session_by_md5
from app.services.session_lock import session_io_lock
from app.routes.segment import sam

video_bp = Blueprint("video", __name__)


def _compute_md5(filepath: str) -> str:
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


@video_bp.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400

    video = request.files["video"]
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)

    video_path = os.path.join(session_dir, "video.mp4")
    video.save(video_path)

    md5 = _compute_md5(video_path)
    original_name = video.filename or "video.mp4"
    fps = request.form.get("fps", FRAME_EXTRACTION_FPS, type=int)
    max_resolution = request.form.get("max_resolution", 2048, type=int)

    # Duplicate check
    duplicate = False
    if SEGMENT_MODE == "cloud" and g.bucket:
        from app.services.gcs_storage import find_session_by_md5_gcs
        existing = find_session_by_md5_gcs(g.bucket, md5)
    else:
        existing = find_session_by_md5(md5)

    if existing:
        # Clean up the new session dir, use the existing one
        shutil.rmtree(session_dir, ignore_errors=True)
        session_id = existing
        session_dir = os.path.join(SESSIONS_DIR, session_id)
        # Read original_name from existing meta if available
        meta_path = os.path.join(session_dir, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            original_name = meta.get("original_name", original_name)
        duplicate = True

    # Write initial meta.json (before extraction)
    if not duplicate:
        meta = {
            "md5": md5,
            "original_name": original_name,
            "fps": fps,
        }
        if SEGMENT_MODE == "cloud":
            meta["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            meta["updated_at"] = meta["created_at"]
        meta_path = os.path.join(session_dir, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    # Start background pipeline
    from app.services.pipeline import ExtractFramesStep, InitSessionStep

    # Capture bucket reference from Flask request context for background thread use.
    # g.bucket is set by the before_request hook in cloud mode; not available in background threads.
    bucket = getattr(g, "bucket", None) if SEGMENT_MODE == "cloud" else None

    steps = []
    if not duplicate:
        steps.append(ExtractFramesStep(session_id, fps=fps, max_dim=max_resolution, bucket=bucket))
    steps.append(InitSessionStep(session_id))

    # Invalidate the list_sessions cache AFTER the pipeline uploads the new
    # session to GCS — not before. Calling invalidate before start_pipeline
    # would race a concurrent GET /api/video/sessions that could repopulate
    # the cache from a bucket that doesn't yet contain this session, leaving
    # the cache stale for up to the 30s TTL (R22). on_complete runs on the
    # pipeline thread after all steps succeed and before the ready transition.
    def _on_pipeline_complete() -> None:
        if bucket:
            from app.services.gcs_storage import invalidate_sessions_cache
            invalidate_sessions_cache(bucket)

    ok, reason = sam.start_pipeline(
        session_id, original_name, steps,
        on_complete=_on_pipeline_complete,
    )
    if not ok:
        return jsonify({"error": reason}), 409

    return jsonify({
        "session_id": session_id,
        "video_name": original_name,
        "duplicate": duplicate,
    })


@video_bp.route("/sessions", methods=["GET"])
def list_sessions():
    """List all existing sessions with their metadata."""
    if SEGMENT_MODE == "cloud" and g.bucket:
        from app.services.gcs_storage import list_sessions as gcs_list
        return jsonify(gcs_list(g.bucket))

    sessions = []
    if not os.path.isdir(SESSIONS_DIR):
        return jsonify(sessions)
    loaded_ids = sam.get_loaded_session_ids()
    for name in sorted(os.listdir(SESSIONS_DIR)):
        session_dir = os.path.join(SESSIONS_DIR, name)
        if not os.path.isdir(session_dir):
            continue
        frames_dir = os.path.join(session_dir, "frames")
        if not os.path.isdir(frames_dir):
            continue
        frame_count = len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
        if frame_count == 0:
            continue

        meta_path = os.path.join(session_dir, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        else:
            meta = {}

        # R14: read per-session state under session_io_lock so listing does
        # not observe a mid-write state.json from a concurrent writer.
        with session_io_lock(name):
            state = load_state(session_dir)

        state_path = os.path.join(session_dir, "state.json")
        updated_at = os.path.getmtime(state_path) if os.path.isfile(state_path) else os.path.getmtime(session_dir)

        total_size = 0
        for dirpath, _dirnames, filenames in os.walk(session_dir):
            for fname in filenames:
                total_size += os.path.getsize(os.path.join(dirpath, fname))

        sessions.append({
            "session_id": name,
            "frame_count": frame_count,
            "original_name": meta.get("original_name", "Unknown"),
            "fps": meta.get("fps", FRAME_EXTRACTION_FPS),
            "video_info": meta.get("video_info", {}),
            "class_count": len(state.get("classes", [])),
            "object_count": len(state.get("objects", [])),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(updated_at)),
            "disk_size": total_size,
            "live": name in loaded_ids,
        })
    sessions.sort(key=lambda s: s["updated_at"], reverse=True)
    return jsonify(sessions)


@video_bp.route("/sessions/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    """Delete a session and all its data."""
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    if not os.path.isdir(session_dir):
        if not (SEGMENT_MODE == "cloud" and g.bucket):
            return jsonify({"error": "Session not found"}), 404

    if SEGMENT_MODE == "cloud" and g.bucket:
        from app.services.gcs_storage import delete_session_gcs, invalidate_sessions_cache
        delete_session_gcs(g.bucket, session_id)
        invalidate_sessions_cache(g.bucket)

    if os.path.isdir(session_dir):
        shutil.rmtree(session_dir)
    return jsonify({"ok": True})


@video_bp.route("/frames/<session_id>", methods=["GET"])
def list_frames(session_id):
    frames_dir = os.path.join(SESSIONS_DIR, session_id, "frames")
    if not os.path.isdir(frames_dir):
        return jsonify({"error": "Session not found"}), 404
    frames = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    return jsonify({"session_id": session_id, "frame_count": len(frames), "frames": [{"index": i, "filename": f} for i, f in enumerate(frames)]})


@video_bp.route("/frame/<session_id>/<int:idx>", methods=["GET"])
def get_frame(session_id, idx):
    frames_dir = os.path.join(SESSIONS_DIR, session_id, "frames")
    filename = f"{idx:05d}.jpg"
    filepath = os.path.join(frames_dir, filename)
    if not os.path.isfile(filepath):
        return jsonify({"error": "Frame not found"}), 404
    return send_file(filepath, mimetype="image/jpeg")
