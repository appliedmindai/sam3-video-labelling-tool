import json
import os
import shutil
import tempfile
import zipfile
from flask import Blueprint, request, jsonify, send_file

from flask import g
from app.config import SESSIONS_DIR, EXPORTS_DIR, SEGMENT_MODE
from app.services.session_manager import load_state
from app.services.exporter import export_coco
from app.services.mask_storage import load_masks
from app.services.session_bundle import export_session_bundle, import_session_bundle
from app.services.session_lock import session_io_lock

export_bp = Blueprint("export", __name__)

@export_bp.route("/coco/<session_id>", methods=["POST"])
def export_coco_route(session_id):
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    # R14: snapshot state + masks under session_io_lock so concurrent writers
    # (propagation persist_fn, PUT state) cannot interleave a mid-write view
    # into the export.
    with session_io_lock(session_id):
        state = load_state(session_dir)
        masks = load_masks(session_dir)
    data = request.json or {}
    bbox_padding = data.get("bbox_padding", data.get("bbox_padding_pct", 0))
    if not masks:
        return jsonify({"error": "No masks found. Run propagation first."}), 400
    output_dir = os.path.join(EXPORTS_DIR, f"{session_id}_coco")
    export_coco(session_dir, state, masks, output_dir, bbox_padding_pct=bbox_padding)
    zip_path = shutil.make_archive(output_dir, "zip", output_dir)
    # Build download name from the original video filename
    meta_path = os.path.join(session_dir, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        stem = os.path.splitext(meta.get("original_name", "export"))[0]
    else:
        stem = "export"
    return send_file(zip_path, as_attachment=True, download_name=f"{stem}_coco.zip")


@export_bp.route("/session/<session_id>", methods=["POST"])
def export_session_route(session_id):
    session_dir = os.path.join(SESSIONS_DIR, session_id)
    if not os.path.isdir(session_dir):
        return jsonify({"error": "Session not found"}), 404

    data = request.json or {}
    include_video = data.get("include_video", False)

    meta_path = os.path.join(session_dir, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        stem = os.path.splitext(meta.get("original_name", "session"))[0]
    else:
        stem = "session"

    output_path = os.path.join(EXPORTS_DIR, f"{session_id}_session.zip")
    export_session_bundle(session_dir, output_path, include_video=include_video)
    return send_file(
        output_path,
        as_attachment=True,
        download_name=f"{stem}_session.zip",
    )


@export_bp.route("/import-session", methods=["POST"])
def import_session_route():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    uploaded = request.files["file"]
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        uploaded.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = import_session_bundle(tmp_path)

        # Cloud mode: upload imported session files to GCS so the session
        # survives container restarts and appears in the session list.
        if SEGMENT_MODE == "cloud" and g.bucket and not result.get("duplicate"):
            from app.services import gcs_storage
            session_id = result["session_id"]
            session_dir = os.path.join(SESSIONS_DIR, session_id)
            gcs_storage.upload_session_files(
                g.bucket, session_id, session_dir,
                ["meta.json", "state.json", "masks.json", "prompts.json", "video.mp4"],
            )
            gcs_storage.upload_directory(g.bucket, session_id, session_dir, "frames")

        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except (zipfile.BadZipFile, KeyError):
        return jsonify({"error": "Invalid session bundle file"}), 400
    except Exception:
        return jsonify({"error": "Failed to import session bundle"}), 500
    finally:
        os.unlink(tmp_path)

