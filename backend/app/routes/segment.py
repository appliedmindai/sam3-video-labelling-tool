import os
import logging
from flask import Blueprint, request, jsonify, Response
import json

import numpy as np
from pycocotools import mask as pmask_utils

logger = logging.getLogger(__name__)

from app.config import SESSIONS_DIR, SEGMENT_MODE, get_session_cache
from app.services.sam3_service import SAM3Service
from app.services.mask_storage import update_frame_masks, remove_object_masks, load_frame_masks_rle
from app.services.prompt_storage import save_prompt, load_prompt, delete_object_prompts
from app.services.session_lock import session_io_lock

segment_bp = Blueprint("segment", __name__)
sam = SAM3Service()

@segment_bp.route("/init/<session_id>", methods=["POST"])
def init_sam(session_id):
    frames_dir = os.path.join(SESSIONS_DIR, session_id, "frames")
    if not os.path.isdir(frames_dir):
        return jsonify({"error": "Session frames not found"}), 404
    try:
        result = sam.init_session(session_id, frames_dir)
    except RuntimeError as e:
        # Pipeline is running for a different session — don't initialize
        # against potentially-partial frames.
        return jsonify({"error": str(e)}), 409
    return jsonify(result)

@segment_bp.route("/click", methods=["POST"])
def click_segment():
    data = request.json
    session_dir = os.path.join(SESSIONS_DIR, data["session_id"])
    sam.ensure_active_object(data["session_id"], data["obj_id"], session_dir)
    result = sam.add_click(
        session_id=data["session_id"], frame_idx=data["frame_idx"],
        obj_id=data["obj_id"], points=data["points"], labels=data["labels"],
    )
    result["source_keyframe"] = None
    with session_io_lock(data["session_id"]):
        _persist_masks_from_result(session_dir, result)
        save_prompt(session_dir, data["frame_idx"], data["obj_id"], {
            "type": "click",
            "points": data["points"],
            "labels": data["labels"],
        }, cache=get_session_cache())
    return jsonify(result)

@segment_bp.route("/box", methods=["POST"])
def box_segment():
    data = request.json
    session_dir = os.path.join(SESSIONS_DIR, data["session_id"])
    sam.ensure_active_object(data["session_id"], data["obj_id"], session_dir)
    result = sam.add_box(
        session_id=data["session_id"], frame_idx=data["frame_idx"],
        obj_id=data["obj_id"], box=data["box"],
    )
    result["source_keyframe"] = None
    with session_io_lock(data["session_id"]):
        _persist_masks_from_result(session_dir, result)
        save_prompt(session_dir, data["frame_idx"], data["obj_id"], {
            "type": "box",
            "box": data["box"],
        }, cache=get_session_cache())
    return jsonify(result)

@segment_bp.route("/text", methods=["POST"])
def text_segment():
    data = request.json
    session_id = data["session_id"]
    frame_idx = data["frame_idx"]
    text = data["text"]
    # obj_id_start: the frontend passes the next available obj_id so the
    # backend can remap text model IDs to match the frontend's numbering.
    obj_id_start = data.get("obj_id_start")
    try:
        result = sam.add_text_prompt(session_id, frame_idx, text)
    except Exception as e:
        import traceback
        logger.error("text_segment failed: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": f"Text detection failed: {str(e)}"}), 500
    try:
        # Remap internal model obj_ids to frontend-assigned sequential IDs
        if obj_id_start is not None and result["instances"]:
            for i, inst in enumerate(result["instances"]):
                inst["obj_id"] = obj_id_start + i
        # Persist remapped masks to disk and register in tracker for propagation
        session_dir = os.path.join(SESSIONS_DIR, session_id)
        if result["instances"]:
            obj_masks = {}
            # Phase 1: SAM3 calls (must NOT be inside session_io_lock to avoid
            # deadlock with propagation's persist_fn which holds SAM3._lock).
            for inst in result["instances"]:
                decoded = pmask_utils.decode(inst["rle"])
                obj_masks[inst["obj_id"]] = decoded
                sam.add_mask(session_id, frame_idx, inst["obj_id"], decoded)
            # Phase 2: file writes (protected by session_io_lock).
            if obj_masks:
                cache = get_session_cache()
                with session_io_lock(session_id):
                    for inst in result["instances"]:
                        save_prompt(session_dir, frame_idx, inst["obj_id"], {
                            "type": "mask",
                            "rle": inst["rle"],
                        }, cache=cache)
                    update_frame_masks(session_dir, frame_idx, obj_masks, source_keyframe=None, cache=cache)
    except Exception as e:
        import traceback
        logger.error("text_segment persist failed: %s\n%s", e, traceback.format_exc())
        return jsonify({"error": f"Text detection succeeded but saving masks failed: {str(e)}"}), 500
    return jsonify(result)

@segment_bp.route("/propagate", methods=["POST"])
def propagate():
    data = request.json
    session_id = data["session_id"]
    start_frame = data.get("start_frame_idx")
    reverse = data.get("reverse", False)
    session_dir = os.path.join(SESSIONS_DIR, session_id)

    object_ids = data.get("object_ids")

    # Active object model: ensure only the target object is loaded
    if object_ids and len(object_ids) == 1:
        sam.ensure_active_object(session_id, object_ids[0], session_dir)
    else:
        # Multi-object: reset inference state and replay ALL requested objects
        # from scratch. Can't just add missing objects — SAM3 native predictor
        # rejects new objects after tracking has started on existing ones.
        sam.reset_and_replay_objects(session_id, session_dir, object_ids)
        replay = sam.replay_prompts_if_needed(session_id, session_dir, object_ids=object_ids)
        if replay["failed"] > 0 and replay["replayed"] == 0:
            return jsonify({
                "error": "All prompts failed to replay — cannot propagate",
                "replay": replay,
            }), 422

    # Resolve the actual source keyframe. Priority:
    # 1. If start_frame has its own prompt for the object, it IS a user keyframe —
    #    ensure_active_object has already replayed that prompt into SAM3 state,
    #    so SAM3 has direct context at start_frame. No rewind needed.
    # 2. Otherwise (start_frame is a propagated frame with no prompt), inherit
    #    source_keyframe from the stored mask and rewind propagation to the
    #    keyframe so SAM3 has full tracking context through the intermediate
    #    frames — without this, SAM3 jumps directly from the keyframe to
    #    start_frame and produces poor masks.
    #
    # Using prompts.json instead of mask.source_keyframe is authoritative: a
    # prior propagation pass can clobber a user-keyframe's mask with its own
    # source_keyframe (update_frame_masks is an upsert), but prompts.json
    # preserves the user's original keyframe identity.
    source_keyframe = start_frame
    effective_start = start_frame
    if start_frame is not None and object_ids:
        cache = get_session_cache()
        if len(object_ids) == 1:
            obj_id = object_ids[0]
            own_prompt = load_prompt(session_dir, start_frame, obj_id, cache=cache)
            if own_prompt is None:
                # No prompt at start_frame — this is a propagated frame.
                # Rewind to the mask's source_keyframe for tracking context.
                mask_data = load_frame_masks_rle(session_dir, start_frame, cache=cache)
                obj_mask = mask_data.get(str(obj_id))
                if obj_mask and obj_mask.get("source_keyframe") is not None:
                    source_keyframe = obj_mask["source_keyframe"]
                    effective_start = source_keyframe
        # Multi-object: use start_frame as source_keyframe for all.
        # Each object may have been keyframed on different frames, so the
        # propagation start frame is the most meaningful common reference.

    def persist_fn(result):
        # WARNING: persist_fn is invoked from inside SAM3._lock by the
        # propagation thread. Acquire session_io_lock carefully — never
        # call SAM3 from within this block or we deadlock against routes
        # that acquire session_io_lock first then SAM3._lock.
        with session_io_lock(session_id):
            _persist_masks_from_result(session_dir, result)
    logger.info(
        "Propagate request: session=%s start_frame=%s effective_start=%s reverse=%s object_ids=%s source_keyframe=%s",
        session_id, start_frame, effective_start, reverse, object_ids, source_keyframe,
    )

    try:
        sam.start_propagation(session_id, effective_start, reverse, persist_fn,
                               object_ids=object_ids, source_keyframe=source_keyframe)
    except ValueError:
        return jsonify({"error": "Propagation already running for this session"}), 409

    def generate():
        for result in sam.subscribe_propagation(session_id):
            yield f"data: {json.dumps(result)}\n\n"
        yield "data: {\"done\": true}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@segment_bp.route("/propagate/subscribe/<session_id>", methods=["GET"])
def subscribe_propagation(session_id):
    """SSE stream to reconnect to an in-progress propagation."""
    status = sam.get_propagation_status(session_id)
    if status["status"] != "running":
        return Response("data: {\"done\": true}\n\n", mimetype="text/event-stream")

    def generate():
        for result in sam.subscribe_propagation(session_id):
            yield f"data: {json.dumps(result)}\n\n"
        yield "data: {\"done\": true}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@segment_bp.route("/propagate/cancel/<session_id>", methods=["POST"])
def cancel_propagation(session_id):
    """Signal the background propagation thread to stop."""
    sam.cancel_propagation(session_id)
    return jsonify({"ok": True})


@segment_bp.route("/propagation-status/<session_id>", methods=["GET"])
def propagation_status(session_id):
    """Return current propagation status for a session."""
    return jsonify(sam.get_propagation_status(session_id))

@segment_bp.route("/remove_object", methods=["POST"])
def remove_object():
    data = request.json
    session_id = data["session_id"]
    obj_id = data["obj_id"]
    # SAM3 call first, outside the io lock.
    sam.remove_object(session_id, obj_id)
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    cache = get_session_cache()
    with session_io_lock(session_id):
        remove_object_masks(session_dir, obj_id, cache=cache)
        delete_object_prompts(session_dir, obj_id, cache=cache)
    return jsonify({"ok": True})

@segment_bp.route("/recalculate", methods=["POST"])
def recalculate():
    data = request.json
    session_dir = os.path.join(SESSIONS_DIR, data["session_id"])
    sam.ensure_active_object(data["session_id"], data["obj_id"], session_dir)
    prompt = load_prompt(session_dir, data["frame_idx"], data["obj_id"], cache=get_session_cache())
    if not prompt:
        return jsonify({"error": "No prompt stored for this frame/object"}), 404
    if prompt["type"] == "click":
        result = sam.add_click(
            session_id=data["session_id"], frame_idx=data["frame_idx"],
            obj_id=data["obj_id"], points=prompt["points"], labels=prompt["labels"],
        )
    else:
        result = sam.add_box(
            session_id=data["session_id"], frame_idx=data["frame_idx"],
            obj_id=data["obj_id"], box=prompt["box"],
        )
    result["source_keyframe"] = None
    with session_io_lock(data["session_id"]):
        _persist_masks_from_result(session_dir, result)
    return jsonify(result)

@segment_bp.route("/reset/<session_id>", methods=["POST"])
def reset(session_id):
    sam.reset_session(session_id)
    return jsonify({"ok": True})

@segment_bp.route("/close/<session_id>", methods=["POST"])
def close(session_id):
    """Release SAM3 inference state and flush GCS sync for this session.

    Returns 200 if all pending uploads succeeded. Returns 503 with the
    failing file list if GCS uploads failed after retries — in that case,
    the sync manager is KEPT ALIVE so the next periodic tick (or an
    explicit retry) can push the remaining files, and the local files
    are preserved on disk. The frontend is expected to show the user a
    retry-or-force-close dialog.
    """
    try:
        sam.close_session(session_id)
    finally:
        from app.config import get_sync_manager, close_active_session

        sm = get_sync_manager()
        if sm and SEGMENT_MODE == "cloud":
            # Promote any deferred files (masks.json if propagation was
            # cancelled by close_session) into dirty before the final
            # flush — the propagation thread may have observed cancel
            # and already promoted them, but if it was killed mid-C
            # we need to promote manually here.
            sm.promote_deferred()
            # Retry the final flush with backoff so transient GCS glitches
            # don't lose data. If files remain dirty, keep the sync manager
            # alive and return 503 so the user knows.
            result = sm.flush_with_retry(max_attempts=3, initial_backoff_s=1.0, backoff_factor=2.0)
            if not result.ok:
                logger.warning(
                    "close | %d file(s) failed to upload to GCS for session %s: %s",
                    len(result.failed), session_id, sorted(result.failed),
                )
                # Persist .unsynced marker + stage dirty bytes to GCS so
                # the NEXT DownloadSessionStep can recover even if the
                # container scales to zero before the user retries.
                from app.services.unsynced_marker import persist_unsynced
                session_dir = os.path.join(SESSIONS_DIR, session_id)
                persist_unsynced(
                    sm.bucket_name, session_id, session_dir,
                    sorted(result.failed), reason="close_flush_failed",
                )
                # Do NOT discard sync manager or session cache — keep them
                # active so subsequent retries (or the periodic timer) can
                # continue attempting upload.
                return jsonify({
                    "error": "Some changes could not be saved to cloud storage",
                    "unsynced_files": sorted(result.failed),
                    "retry_possible": True,
                }), 503

            # All flushed. stop() may pick up a last-moment mark_dirty
            # from an in-flight request (e.g. bboxPadding debounce save
            # arriving between flush_with_retry and this line). Honour
            # that result: if stop's internal flush fails, write a
            # marker and surface 503 so the client can retry.
            stop_result = sm.stop()
            if not stop_result.ok:
                logger.warning(
                    "close | stop()'s final flush left %d file(s) dirty for session %s: %s",
                    len(stop_result.failed), session_id, sorted(stop_result.failed),
                )
                from app.services.unsynced_marker import persist_unsynced
                session_dir = os.path.join(SESSIONS_DIR, session_id)
                persist_unsynced(
                    sm.bucket_name, session_id, session_dir,
                    sorted(stop_result.failed), reason="close_stop_failed",
                )
                # sm is stopped now. Clear both globals atomically under
                # _globals_lock so no concurrent request sees a stopped
                # manager still in the active slot (R31). Do not re-stop:
                # close_active_session does not call stop().
                #
                # Wrap in `finalize_close` so the globals-clear happens
                # under `_state_lock` — prevents a concurrent pipeline's
                # on_complete → ready transition from racing past our
                # globals-clear (H6, mirror of H5 race).
                sam.finalize_close(session_id, close_active_session)
                return jsonify({
                    "error": "Some changes could not be saved to cloud storage",
                    "unsynced_files": sorted(stop_result.failed),
                    "retry_possible": False,
                }), 503

        # Happy path (or non-cloud): atomically clear both globals so
        # any concurrent mark_dirty_safe() observes a consistent
        # (None, None) state and short-circuits. Under `_state_lock` so
        # a racing pipeline cannot set phase=ready between our globals
        # clear and the idle transition.
        sam.finalize_close(session_id, close_active_session)

    return jsonify({"ok": True})


@segment_bp.route("/flush", methods=["POST"])
def flush_sync():
    """Trigger an immediate GCS sync of all dirty files.

    Called by the frontend on beforeunload (via sendBeacon) so that
    annotations are persisted even if the user closes the tab without
    explicitly closing the session. Beacons cannot read responses, so
    this handler promotes deferred masks before flushing and writes an
    .unsynced marker on failure — the marker is the durable trace that
    the next resume's DownloadSessionStep recovers from.
    """
    if SEGMENT_MODE != "cloud":
        return jsonify({"ok": True, "uploaded": 0, "failed": []})
    from app.config import get_sync_manager
    sm = get_sync_manager()
    if not sm:
        return jsonify({"ok": True, "uploaded": 0, "failed": []})

    try:
        # Promote any in-flight deferred masks (propagation may still be
        # mid-flight when the tab closes) so the single flush attempt
        # covers them.
        sm.promote_deferred()
        # Beacons run on `beforeunload` — Cloud Run may give us only seconds
        # before the tab vanishes. One shot, no backoff; the periodic tick
        # and the next resume's marker recovery are the follow-ups.
        result = sm.flush_with_retry(max_attempts=1)
    except Exception:
        logger.error("beacon_flush | flush raised", exc_info=True)
        return jsonify({"ok": False, "uploaded": 0, "failed": []}), 200

    if not result.ok:
        logger.warning(
            "beacon_flush | %d file(s) failed for session %s: %s",
            len(result.failed), sm.session_id, sorted(result.failed),
        )
        try:
            from app.services.unsynced_marker import persist_unsynced
            persist_unsynced(
                sm.bucket_name, sm.session_id, sm.session_dir,
                sorted(result.failed),
                reason="beacon_flush_failed",
            )
        except Exception:
            logger.error("beacon_flush | marker write failed", exc_info=True)

    return jsonify({
        "ok": result.ok,
        "uploaded": result.uploaded,
        "failed": sorted(result.failed),
    }), 200


def _persist_masks_from_result(session_dir, result):
    """Extract binary masks from result RLEs and persist them."""
    from pycocotools import mask as pmask_utils
    import numpy as np
    frame_idx = result["frame_idx"]
    source_keyframe = result.get("source_keyframe")
    obj_masks = {}
    for obj_id_str, mask_data in result["masks"].items():
        decoded = pmask_utils.decode(mask_data["rle"])
        obj_masks[int(obj_id_str)] = decoded
    if obj_masks:
        update_frame_masks(session_dir, frame_idx, obj_masks, source_keyframe=source_keyframe, cache=get_session_cache())
