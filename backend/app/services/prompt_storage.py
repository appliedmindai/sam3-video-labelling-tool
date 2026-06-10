import json
import os

from app.services.atomic_write import atomic_json_dump

PROMPTS_FILENAME = "prompts.json"


def _load_prompts_file(session_dir, cache=None):
    if cache is not None:
        return cache.load(PROMPTS_FILENAME)
    path = os.path.join(session_dir, PROMPTS_FILENAME)
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def _save_prompts_file(session_dir, data, cache=None):
    if cache is not None:
        cache.save(PROMPTS_FILENAME, data)
    else:
        path = os.path.join(session_dir, PROMPTS_FILENAME)
        atomic_json_dump(data, path)


def save_prompt(session_dir, frame_idx, obj_id, prompt_data, cache=None):
    """Save or overwrite a prompt for a specific frame + object."""
    data = _load_prompts_file(session_dir, cache=cache)
    frame_key = str(frame_idx)
    obj_key = str(obj_id)
    # Clone nested dict before mutating.
    frame_entry = dict(data.get(frame_key, {}))
    frame_entry[obj_key] = prompt_data
    data[frame_key] = frame_entry
    _save_prompts_file(session_dir, data, cache=cache)

    from app.services.gcs_sync import mark_dirty_safe
    mark_dirty_safe("prompts.json")


def load_all_prompts(session_dir, cache=None):
    """Return the full prompts dict (frame_str -> obj_str -> prompt_data)."""
    return _load_prompts_file(session_dir, cache=cache)


def load_prompt(session_dir, frame_idx, obj_id, cache=None):
    """Return a single prompt or None."""
    data = _load_prompts_file(session_dir, cache=cache)
    frame_key = str(frame_idx)
    obj_key = str(obj_id)
    return data.get(frame_key, {}).get(obj_key)


def delete_object_prompts(session_dir, obj_id, cache=None):
    """Remove all prompts for a given object across all frames."""
    data = _load_prompts_file(session_dir, cache=cache)
    obj_key = str(obj_id)
    changed = False
    for frame_key in list(data.keys()):
        if obj_key in data[frame_key]:
            # Clone nested dict before mutating.
            frame_entry = dict(data[frame_key])
            del frame_entry[obj_key]
            if frame_entry:
                data[frame_key] = frame_entry
            else:
                del data[frame_key]
            changed = True
    if changed:
        _save_prompts_file(session_dir, data, cache=cache)

    from app.services.gcs_sync import mark_dirty_safe
    mark_dirty_safe("prompts.json")


def delete_frame_object_prompt(session_dir, frame_idx, obj_id, cache=None):
    """Remove a single prompt for a specific frame + object."""
    data = _load_prompts_file(session_dir, cache=cache)
    frame_key = str(frame_idx)
    obj_key = str(obj_id)
    if obj_key not in data.get(frame_key, {}):
        return
    # Clone nested dict before mutating.
    frame_entry = dict(data[frame_key])
    del frame_entry[obj_key]
    if frame_entry:
        data[frame_key] = frame_entry
    else:
        del data[frame_key]
    _save_prompts_file(session_dir, data, cache=cache)

    from app.services.gcs_sync import mark_dirty_safe
    mark_dirty_safe("prompts.json")
