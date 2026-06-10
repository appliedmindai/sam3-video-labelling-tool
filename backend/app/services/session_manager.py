import json
import os

from app.config import SESSIONS_DIR
from app.services.atomic_write import atomic_json_dump

STATE_FILENAME = "state.json"
DEFAULT_STATE = {"classes": [], "objects": [], "version": 0}

def save_state(session_dir: str, state: dict, cache=None) -> None:
    """Persist state.json, bumping the monotonic `version` counter.

    `version` is the lost-update guard for PUT /api/session/state/<sid>
   . Every write increments it by one. Readers echo the value
    back; writers must match it or get 409.
    """
    state = {**state, "version": int(state.get("version", 0)) + 1}
    if cache is not None:
        cache.save(STATE_FILENAME, state, indent=2)
    else:
        path = os.path.join(session_dir, STATE_FILENAME)
        atomic_json_dump(state, path, indent=2)

    from app.services.gcs_sync import mark_dirty_safe
    mark_dirty_safe("state.json")

def load_state(session_dir: str, cache=None) -> dict:
    """Load the session state dict.

    Contract: returns whatever is on disk verbatim. Only synthesizes
    DEFAULT_STATE when there is genuinely nothing (file absent or cache empty
    AFTER a fresh disk check). Never masks real-but-empty state -- downstream
    code must be able to distinguish 'no file yet' from 'file has {}'.

    The `version` field is synthesized as 0 for legacy state.json files
    written before the lost-update guard was added.
    """
    def _normalize(data: dict) -> dict:
        return {"classes": data.get("classes", []),
                "objects": data.get("objects", []),
                "version": int(data.get("version", 0)),
                **{k: v for k, v in data.items()
                   if k not in ("classes", "objects", "version")}}

    if cache is not None:
        data = cache.load(STATE_FILENAME)
        # cache.load now auto-refreshes on empty; if we still got {}, disk
        # either doesn't have the file or it literally contains {}. Either
        # way, return DEFAULT_STATE with both keys so callers can safely
        # append to classes/objects without KeyError.
        if not data:
            return {"classes": [], "objects": [], "version": 0}
        # Ensure the essential keys exist even if the on-disk payload is
        # partial (defensive: older sessions or hand-edited files).
        return _normalize(data)
    path = os.path.join(session_dir, STATE_FILENAME)
    if not os.path.isfile(path):
        return {"classes": [], "objects": [], "version": 0}
    with open(path) as f:
        data = json.load(f)
    return _normalize(data)

def add_class(state: dict, name: str, color: str) -> dict:
    existing_ids = [c["id"] for c in state["classes"]]
    new_id = max(existing_ids, default=0) + 1
    state["classes"].append({"id": new_id, "name": name, "color": color})
    return state

def update_class(state: dict, class_id: int, name: str = None, color: str = None) -> dict:
    for c in state["classes"]:
        if c["id"] == class_id:
            if name is not None:
                c["name"] = name
            if color is not None:
                c["color"] = color
            break
    return state

def delete_class(state: dict, class_id: int) -> dict:
    state["classes"] = [c for c in state["classes"] if c["id"] != class_id]
    state["objects"] = [o for o in state["objects"] if o["class_id"] != class_id]
    return state


def reassign_object(state: dict, obj_id: int, new_class_id: int) -> dict:
    if not any(c["id"] == new_class_id for c in state["classes"]):
        raise ValueError(f"Class {new_class_id} not found")
    for obj in state["objects"]:
        if obj["obj_id"] == obj_id:
            obj["class_id"] = new_class_id
            return state
    raise ValueError(f"Object {obj_id} not found")


def find_session_by_md5(md5: str) -> str | None:
    """Check if any existing session has the same video MD5."""
    if not os.path.isdir(SESSIONS_DIR):
        return None
    for name in os.listdir(SESSIONS_DIR):
        meta_path = os.path.join(SESSIONS_DIR, name, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("md5") == md5:
                return name
    return None
