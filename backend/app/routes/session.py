import json
import os
from flask import Blueprint, request, jsonify, g
from app.config import SESSIONS_DIR, SEGMENT_MODE, get_session_cache
from app.services.session_manager import load_state, save_state, add_class, update_class, delete_class, reassign_object
from app.services.mask_storage import load_all_masks_rle, load_frame_masks_rle, delete_frame_object_mask, delete_object_masks_by_source, delete_frame_object_masks_batch, get_versions
from app.services.prompt_storage import load_all_prompts, delete_frame_object_prompt
from app.services.sam3_service import SAM3Service
from app.services.session_lock import session_io_lock

session_bp = Blueprint("session", __name__)
sam = SAM3Service()

@session_bp.route("/state/<session_id>", methods=["GET"])
def get_state(session_id):
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    if not os.path.isdir(session_dir):
        return jsonify({"error": "Session not found"}), 404
    return jsonify(load_state(session_dir, cache=get_session_cache()))

@session_bp.route("/state/<session_id>", methods=["PUT"])
def put_state(session_id):
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    if not os.path.isdir(session_dir):
        return jsonify({"error": "Session not found"}), 404
    payload = request.json or {}
    # Defense in depth: refuse to overwrite real state with the wipe fingerprint.
    # Fingerprint = classes is empty AND objects contains class_id<0 entries.
    # A legitimate PUT either has classes (any length) with matching objects,
    # or truly empty {classes:[], objects:[]} (user deleted everything).
    payload_classes = payload.get("classes") or []
    payload_objects = payload.get("objects") or []
    looks_like_wipe = (
        len(payload_classes) == 0
        and any(
            isinstance(o, dict) and o.get("class_id", 0) < 0
            for o in payload_objects
        )
    )
    # Optimistic concurrency: if the payload declares a `version`, it
    # must match what's currently on disk. Otherwise two tabs editing the
    # same session silently overwrite each other (last-writer-wins). PUTs
    # without a `version` field are treated as force-overwrite for backward
    # compatibility with external tools and bootstrap writes.
    payload_version = payload.get("version")
    cache = get_session_cache()
    with session_io_lock(session_id):
        current = load_state(session_dir, cache=cache)
        if payload_version is not None and payload_version != current.get("version", 0):
            return jsonify({
                "error": "state_version_conflict",
                "current_version": current.get("version", 0),
                "state": current,
            }), 409
        if looks_like_wipe:
            # Only reject if disk currently has non-empty classes (i.e., we'd lose data)
            if current.get("classes"):
                import logging
                logging.getLogger(__name__).warning(
                    "put_state | rejecting wipe fingerprint for session %s "
                    "(current has %d classes, payload has 0 classes + %d orphan objects)",
                    session_id, len(current["classes"]),
                    sum(1 for o in payload_objects if o.get("class_id", 0) < 0),
                )
                return jsonify({
                    "error": "Refusing to overwrite non-empty state with empty classes "
                             "+ orphan-only objects. This looks like the known "
                             "wipe-on-stale-frontend bug."
                }), 409
        # Anchor version to disk so a PUT without a `version` field (legacy
        # or external caller) still produces N+1, not 1.
        payload["version"] = current.get("version", 0)
        save_state(session_dir, payload, cache=cache)
        new_version = payload["version"] + 1
    return jsonify({"ok": True, "version": new_version})

@session_bp.route("/masks/<session_id>", methods=["GET"])
def get_masks(session_id):
    """Return all saved masks (RLE) with bbox and area for each frame/object."""
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    if not os.path.isdir(session_dir):
        return jsonify({"error": "Session not found"}), 404
    cache = get_session_cache()
    masks = load_all_masks_rle(session_dir, cache=cache)
    versions = get_versions(session_dir, cache=cache)
    return jsonify({"masks": masks, "versions": versions})


@session_bp.route("/masks/<session_id>/versions", methods=["GET"])
def get_mask_versions(session_id):
    """Return per-frame version counters for cache invalidation."""
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    if not os.path.isdir(session_dir):
        return jsonify({"error": "Session not found"}), 404
    versions = get_versions(session_dir, cache=get_session_cache())
    return jsonify({"versions": versions, "frame_count": len(versions)})


@session_bp.route("/masks/<session_id>/<int:frame_idx>", methods=["GET"])
def get_frame_masks(session_id, frame_idx):
    """Return masks for a single frame."""
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    if not os.path.isdir(session_dir):
        return jsonify({"error": "Session not found"}), 404
    result = load_frame_masks_rle(session_dir, frame_idx, cache=get_session_cache())
    return jsonify(result)


@session_bp.route("/prompts/<session_id>", methods=["GET"])
def get_prompts(session_id):
    """Return all saved prompts for this session."""
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    if not os.path.isdir(session_dir):
        return jsonify({"error": "Session not found"}), 404
    return jsonify(load_all_prompts(session_dir, cache=get_session_cache()))


@session_bp.route("/masks/<session_id>/<int:frame_idx>/<int:obj_id>", methods=["DELETE"])
def delete_mask(session_id, frame_idx, obj_id):
    """Delete a single object's mask from a single frame."""
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    if not os.path.isdir(session_dir):
        return jsonify({"error": "Session not found"}), 404
    cache = get_session_cache()
    with session_io_lock(session_id):
        deleted = delete_frame_object_mask(session_dir, frame_idx, obj_id, cache=cache)
        delete_frame_object_prompt(session_dir, frame_idx, obj_id, cache=cache)
    # Release the lock before SAM3 to avoid deadlock with propagation's persist_fn
    # (which holds SAM3._lock and wants session_io_lock).
    sam.clear_frame_object(session_id, frame_idx, obj_id)
    return jsonify({"ok": True, "deleted": deleted})


@session_bp.route("/masks/<session_id>/<int:obj_id>/by-source", methods=["DELETE"])
def delete_masks_by_source(session_id, obj_id):
    """Delete masks for an object filtered by source_keyframe and direction."""
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    if not os.path.isdir(session_dir):
        return jsonify({"error": "Session not found"}), 404
    data = request.json or {}
    source_keyframe = data.get("source_keyframe")
    direction = data.get("direction", "this")
    if direction not in ("left", "right", "this"):
        return jsonify({"error": f"Invalid direction: {direction}"}), 400
    current_frame = data.get("current_frame", 0)
    cache = get_session_cache()
    with session_io_lock(session_id):
        deleted = delete_object_masks_by_source(session_dir, obj_id, source_keyframe, direction, current_frame, cache=cache)
        # If deleting a keyframe (source_keyframe is None), also delete its prompt
        if source_keyframe is None:
            for frame_idx in deleted:
                delete_frame_object_prompt(session_dir, frame_idx, obj_id, cache=cache)
    # Release lock before SAM3 to avoid deadlock with propagation's persist_fn
    # (which holds SAM3._lock and wants session_io_lock).
    if deleted:
        sam.clear_frame_object_batch(session_id, deleted, obj_id)
    return jsonify({"deleted_frames": deleted})


@session_bp.route("/masks/<session_id>/<int:obj_id>/batch", methods=["DELETE"])
def delete_masks_batch(session_id, obj_id):
    """Delete masks for an object across a list of specific frames."""
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    if not os.path.isdir(session_dir):
        return jsonify({"error": "Session not found"}), 404
    data = request.get_json() or {}
    frame_indices = data.get("frame_indices", [])

    cache = get_session_cache()
    with session_io_lock(session_id):
        deleted = delete_frame_object_masks_batch(session_dir, obj_id, frame_indices, cache=cache)
        # Clear prompts for deleted frames
        for fi in deleted:
            delete_frame_object_prompt(session_dir, fi, obj_id, cache=cache)

    # Release lock before SAM3 to avoid deadlock with propagation's persist_fn
    # (which holds SAM3._lock and wants session_io_lock).
    if deleted:
        sam.clear_frame_object_batch(session_id, deleted, obj_id)

    return jsonify({"deleted_frames": deleted, "deleted_count": len(deleted)})


@session_bp.route("/classes/<session_id>", methods=["GET"])
def get_classes(session_id):
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    state = load_state(session_dir, cache=get_session_cache())
    return jsonify(state["classes"])

@session_bp.route("/classes/<session_id>", methods=["POST"])
def create_class(session_id):
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    cache = get_session_cache()
    data = request.json
    with session_io_lock(session_id):
        state = load_state(session_dir, cache=cache)
        state = add_class(state, data["name"], data["color"])
        save_state(session_dir, state, cache=cache)
        created = state["classes"][-1]
    return jsonify(created), 201

@session_bp.route("/classes/<session_id>/<int:class_id>", methods=["PUT"])
def edit_class(session_id, class_id):
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    cache = get_session_cache()
    data = request.json
    with session_io_lock(session_id):
        state = load_state(session_dir, cache=cache)
        state = update_class(state, class_id, data.get("name"), data.get("color"))
        save_state(session_dir, state, cache=cache)
    return jsonify({"ok": True})

@session_bp.route("/classes/<session_id>/<int:class_id>", methods=["DELETE"])
def remove_class(session_id, class_id):
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    cache = get_session_cache()
    with session_io_lock(session_id):
        state = load_state(session_dir, cache=cache)
        state = delete_class(state, class_id)
        save_state(session_dir, state, cache=cache)
    return jsonify({"ok": True})

@session_bp.route("/objects/<session_id>/<int:obj_id>/reassign", methods=["PUT"])
def reassign_obj(session_id, obj_id):
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    if not os.path.isdir(session_dir):
        return jsonify({"error": "Session not found"}), 404
    data = request.json
    new_class_id = data.get("class_id")
    if new_class_id is None:
        return jsonify({"error": "class_id required"}), 400
    cache = get_session_cache()
    with session_io_lock(session_id):
        state = load_state(session_dir, cache=cache)
        try:
            state = reassign_object(state, obj_id, new_class_id)
        except ValueError as e:
            return jsonify({"error": str(e)}), 404
        save_state(session_dir, state, cache=cache)
    return jsonify({"ok": True})


@session_bp.route("/open/<session_id>", methods=["POST"])
def open_session(session_id):
    """Download a session from GCS to local disk for editing."""
    if SEGMENT_MODE != "cloud" or not g.bucket:
        return jsonify({"ok": True})

    from app.services.gcs_storage import download_session
    from app.services.gcs_sync import GCSSyncManager
    from app.services.session_cache import SessionCache
    from app.config import install_active_session

    session_dir = os.path.join(SESSIONS_DIR, session_id)
    if not os.path.isdir(session_dir):
        download_session(g.bucket, session_id, session_dir)

    sm = GCSSyncManager(g.bucket, session_id, session_dir)
    sm.start(interval=60)
    # Install (sm, cache) atomically so concurrent requests never observe
    # (sm_new, cache_old). The previous two-call pattern
    # (set_sync_manager then set_session_cache) left a race window where
    # mark_dirty_safe would route to sm_new while reads still served
    # cache_old — the exact (sm, cache) tearing bug R31 targeted.
    install_active_session(sm, SessionCache(session_dir))

    return jsonify({"ok": True, "session_id": session_id})


@session_bp.route("/resume/<session_id>", methods=["POST"])
def resume_session(session_id):
    """Start background pipeline to make a session ready."""
    session_dir = os.path.join(SESSIONS_DIR, session_id)

    # Read video name from meta
    meta_path = os.path.join(session_dir, "meta.json")
    video_name = session_id  # fallback

    # Check local first, then GCS
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        video_name = meta.get("original_name", video_name)
    elif SEGMENT_MODE == "cloud" and g.bucket:
        # Session only exists in GCS — read meta.json directly so the
        # loading screen shows the filename instead of the session UUID.
        try:
            from google.cloud import storage as gcs
            client = gcs.Client()
            blob = client.bucket(g.bucket).blob(f"{session_id}/meta.json")
            if blob.exists():
                meta = json.loads(blob.download_as_text())
                video_name = meta.get("original_name", video_name)
        except Exception:
            pass  # fallback to UUID — DownloadSessionStep will resolve it later
    else:
        # Standalone mode with no local meta.json: nothing to resume from.
        return jsonify({"error": "Session not found"}), 404

    # Build pipeline
    from app.services.pipeline import InitSessionStep, DownloadSessionStep

    steps = []
    bucket = g.bucket if SEGMENT_MODE == "cloud" else None
    if bucket:
        steps.append(DownloadSessionStep(session_id, bucket))
    steps.append(InitSessionStep(session_id))

    # Always ensure session_dir exists — DownloadSessionStep writes files
    # into it and SessionCache's load() auto-refreshes from disk.
    os.makedirs(session_dir, exist_ok=True)

    # Clear the previous session's sync manager + cache atomically
    # BEFORE the pipeline starts. During the pipeline any concurrent
    # mark_dirty_safe from an orphan request will see (None, None) and
    # fail closed — better to drop a mid-resume write than corrupt
    # B's session_dir with A's data.
    from app.config import (
        clear_active_session_for_resume,
        install_active_session,
    )
    from app.services.session_cache import SessionCache
    from app.services.gcs_sync import GCSSyncManager

    clear_active_session_for_resume()

    def _on_pipeline_complete() -> None:
        # Install both globals atomically on the pipeline thread after
        # all steps succeeded, BEFORE the ready-phase transition. The
        # frontend's first post-ready request will see a fully hydrated
        # active session.
        new_sm = None
        if bucket:
            new_sm = GCSSyncManager(bucket, session_id, session_dir)
        install_active_session(new_sm, SessionCache(session_dir))
        if new_sm is not None:
            new_sm.start()

    ok, reason = sam.start_pipeline(
        session_id, video_name, steps,
        on_complete=_on_pipeline_complete,
    )
    if not ok:
        # Pipeline refused — globals are (None, None); nothing more to do.
        return jsonify({"error": reason}), 409

    return jsonify({
        "session_id": session_id,
        "video_name": video_name,
    }), 202
