"""Service status and job control endpoints."""

import logging

from flask import Blueprint, jsonify

from app.config import BOOT_ID
from app.services.sam3_service import SAM3Service

logger = logging.getLogger(__name__)

status_bp = Blueprint("status", __name__)
sam = SAM3Service()


@status_bp.route("/status", methods=["GET"])
def get_status():
    """Return the current service state. No auth required."""
    state = sam.get_service_state()

    response = {
        "phase": state.phase,
        "session_id": state.session_id,
        "video_name": state.video_name,
        "progress": state.progress,
        "error": state.error,
        "frame_count": state.frame_count,
        "propagation": None,
        "boot_id": BOOT_ID,
    }

    # Include propagation status when session is ready
    if state.phase == "ready" and state.session_id:
        response["propagation"] = sam.get_propagation_status(state.session_id)

    return jsonify(response)


@status_bp.route("/job/cancel", methods=["POST"])
def cancel_job():
    """Cancel the active pipeline."""
    result = sam.cancel_pipeline()
    return jsonify({"status": result})


@status_bp.route("/status/dismiss-error", methods=["POST"])
def dismiss_error():
    """Clear the error phase so the UI can return to the session list."""
    result = sam.dismiss_error()
    return jsonify({"status": result})
