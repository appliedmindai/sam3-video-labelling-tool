"""Export and import session bundles as portable zip files."""

import json
import os
import shutil
import uuid
import zipfile
from datetime import datetime, timezone

from app.config import SESSIONS_DIR
from app.services.session_manager import find_session_by_md5

# ---------------------------------------------------------------------------
# Versioned importer registry
# ---------------------------------------------------------------------------
_IMPORTERS = {}


def _register_importer(version):
    def decorator(fn):
        _IMPORTERS[version] = fn
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json(path):
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {}


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_session_bundle(
    session_dir: str,
    output_path: str,
    include_video: bool = False,
) -> str:
    """Package a session into a portable zip bundle.

    Returns the path to the created zip file.
    """
    meta = _read_json(os.path.join(session_dir, "meta.json"))
    state = _read_json(os.path.join(session_dir, "state.json"))
    masks = _read_json(os.path.join(session_dir, "masks.json"))
    prompts = _read_json(os.path.join(session_dir, "prompts.json"))

    frames_dir = os.path.join(session_dir, "frames")
    frame_files = sorted(
        f for f in os.listdir(frames_dir) if f.endswith(".jpg")
    ) if os.path.isdir(frames_dir) else []

    video_path = os.path.join(session_dir, "video.mp4")
    has_video = include_video and os.path.isfile(video_path)

    manifest = {
        "format": "sam3-annotator-session",
        "version": 1,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "includes_video": has_video,
        "session": {
            "original_name": meta.get("original_name", "unknown"),
            "frame_count": len(frame_files),
            "fps": meta.get("fps", 5),
            "video_md5": meta.get("md5", ""),
        },
    }

    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("meta.json", json.dumps(meta, indent=2))
        zf.writestr("state.json", json.dumps(state, indent=2))
        zf.writestr("masks.json", json.dumps(masks, indent=2))
        zf.writestr("prompts.json", json.dumps(prompts, indent=2))

        for fname in frame_files:
            zf.write(os.path.join(frames_dir, fname), f"frames/{fname}")

        if has_video:
            zf.write(video_path, "video.mp4")

    return output_path


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------

def import_session_bundle(
    zip_path: str,
    sessions_dir: str | None = None,
) -> dict:
    """Import a session bundle zip into the sessions directory.

    Returns dict with session_id, frame_count, original_name, fps, duplicate.
    """
    if sessions_dir is None:
        sessions_dir = SESSIONS_DIR

    with zipfile.ZipFile(zip_path, "r") as zf:
        raw = zf.read("manifest.json")
        manifest = json.loads(raw)

        fmt = manifest.get("format")
        if fmt != "sam3-annotator-session":
            raise ValueError(f"Unknown bundle format: {fmt}")

        version = manifest.get("version")
        importer = _IMPORTERS.get(version)
        if importer is None:
            raise ValueError(
                f"Unsupported bundle version: {version}. "
                f"Supported: {sorted(_IMPORTERS.keys())}"
            )

        return importer(zf, manifest, sessions_dir)


@_register_importer(1)
def _import_v1(
    zf: zipfile.ZipFile,
    manifest: dict,
    sessions_dir: str,
) -> dict:
    session_meta = manifest.get("session", {})
    video_md5 = session_meta.get("video_md5", "")

    # Duplicate check
    duplicate = False
    existing_id = None
    if video_md5:
        existing_id = find_session_by_md5(video_md5)

    if existing_id:
        existing_dir = os.path.join(sessions_dir, existing_id)
        frames_dir = os.path.join(existing_dir, "frames")
        frame_count = len([
            f for f in os.listdir(frames_dir) if f.endswith(".jpg")
        ]) if os.path.isdir(frames_dir) else 0
        existing_meta = _read_json(os.path.join(existing_dir, "meta.json"))
        return {
            "session_id": existing_id,
            "frame_count": frame_count,
            "original_name": existing_meta.get("original_name", session_meta.get("original_name", "unknown")),
            "fps": existing_meta.get("fps", session_meta.get("fps", 5)),
            "duplicate": True,
        }

    # Create new session
    session_id = str(uuid.uuid4())
    session_dir = os.path.join(sessions_dir, session_id)
    os.makedirs(session_dir, exist_ok=True)

    try:
        real_session = os.path.realpath(session_dir)
        total_extracted = 0
        max_extract_bytes = 10 * 1024 * 1024 * 1024  # 10 GB
        max_file_count = 100_000

        # Extract files (skip manifest.json — it stays only in the zip)
        file_count = 0
        for info in zf.infolist():
            if info.filename == "manifest.json":
                continue
            file_count += 1
            if file_count > max_file_count:
                raise ValueError("Too many files in zip archive")
            target = os.path.join(session_dir, info.filename)
            # Zip Slip protection
            if not os.path.realpath(target).startswith(real_session + os.sep):
                raise ValueError(f"Illegal path in zip: {info.filename}")
            if info.is_dir():
                os.makedirs(target, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as dst:
                    while True:
                        chunk = src.read(65536)
                        if not chunk:
                            break
                        total_extracted += len(chunk)
                        if total_extracted > max_extract_bytes:
                            raise ValueError("Zip archive too large when decompressed")
                        dst.write(chunk)
    except Exception:
        shutil.rmtree(session_dir, ignore_errors=True)
        raise

    frames_dir = os.path.join(session_dir, "frames")
    frame_count = len([
        f for f in os.listdir(frames_dir) if f.endswith(".jpg")
    ]) if os.path.isdir(frames_dir) else 0

    return {
        "session_id": session_id,
        "frame_count": frame_count,
        "original_name": session_meta.get("original_name", "unknown"),
        "fps": session_meta.get("fps", 5),
        "duplicate": False,
    }
