import json
import logging
import os

import numpy as np
from app.services.atomic_write import atomic_json_dump
from pycocotools import mask as mask_utils

logger = logging.getLogger(__name__)

MASKS_FILENAME = "masks.json"
_VERSIONS_KEY = "_versions"

def _encode_mask(binary_mask):
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    rle["size"] = [int(s) for s in rle["size"]]
    return rle

def _decode_mask(rle):
    return mask_utils.decode(rle)

def unpack_entry(entry):
    """Unpack a mask entry, handling both old and new format.

    Old format: entry is a bare RLE dict {counts, size}
    New format: entry is {rle: {counts, size}, source_keyframe: int|null}

    Returns: (rle_dict, source_keyframe)
    """
    if "rle" in entry:
        return entry["rle"], entry.get("source_keyframe")
    return entry, None

def get_versions(session_dir, cache=None):
    """Return the per-frame version dict from masks.json."""
    if cache is not None:
        encoded = cache.load(MASKS_FILENAME)
    else:
        path = os.path.join(session_dir, MASKS_FILENAME)
        if not os.path.isfile(path):
            return {}
        with open(path) as f:
            encoded = json.load(f)
    return dict(encoded.get(_VERSIONS_KEY, {}))


def save_masks(session_dir, masks):
    """Save masks in legacy flat-RLE format (retained for backward-compat testing)."""
    encoded = {}
    for frame_idx, obj_masks in masks.items():
        encoded[str(frame_idx)] = {str(obj_id): _encode_mask(m) for obj_id, m in obj_masks.items()}
    path = os.path.join(session_dir, MASKS_FILENAME)
    atomic_json_dump(encoded, path)

def load_masks(session_dir, cache=None):
    """Load all masks from disk.

    Returns: {frame_idx: {obj_id: (mask_array, source_keyframe)}}
    """
    if cache is not None:
        encoded = cache.load(MASKS_FILENAME)
    else:
        path = os.path.join(session_dir, MASKS_FILENAME)
        if not os.path.isfile(path):
            return {}
        with open(path) as f:
            encoded = json.load(f)
    masks = {}
    for frame_str, obj_masks in encoded.items():
        if frame_str == _VERSIONS_KEY:
            continue
        masks[int(frame_str)] = {}
        for obj_id, entry in obj_masks.items():
            rle, source_kf = unpack_entry(entry)
            masks[int(frame_str)][int(obj_id)] = (_decode_mask(rle), source_kf)
    return masks

def remove_object_masks(session_dir, obj_id, cache=None):
    """Remove all masks for a given object from disk."""
    if cache is not None:
        encoded = cache.load(MASKS_FILENAME)
    else:
        path = os.path.join(session_dir, MASKS_FILENAME)
        if not os.path.isfile(path):
            return
        with open(path) as f:
            encoded = json.load(f)
    obj_str = str(obj_id)
    changed = False
    versions = dict(encoded.get(_VERSIONS_KEY, {}))
    for frame_key in list(encoded.keys()):
        if frame_key == _VERSIONS_KEY:
            continue
        if obj_str in encoded[frame_key]:
            # Clone nested dict before mutating.
            frame_entry = dict(encoded[frame_key])
            del frame_entry[obj_str]
            if frame_entry:
                encoded[frame_key] = frame_entry
            else:
                del encoded[frame_key]
            changed = True
            versions[frame_key] = versions.get(frame_key, 0) + 1
    if changed:
        encoded[_VERSIONS_KEY] = versions
        if cache is not None:
            cache.save(MASKS_FILENAME, encoded)
        else:
            atomic_json_dump(encoded, path)
        from app.services.gcs_sync import mark_dirty_safe
        mark_dirty_safe("masks.json")


def load_frame_masks_rle(session_dir, frame_idx, cache=None):
    """Load masks for a single frame, returning frontend-ready {obj_str: {rle, bbox, area, source_keyframe}}."""
    if cache is not None:
        encoded = cache.load(MASKS_FILENAME)
    else:
        path = os.path.join(session_dir, MASKS_FILENAME)
        if not os.path.isfile(path):
            return {}
        with open(path) as f:
            encoded = json.load(f)
    frame_key = str(frame_idx)
    obj_masks = encoded.get(frame_key, {})
    result = {}
    for obj_str, entry in obj_masks.items():
        rle, source_kf = unpack_entry(entry)
        binary = mask_utils.decode(rle)
        area = int(binary.sum())
        bbox = mask_utils.toBbox(rle).tolist()
        bbox = [int(v) for v in bbox]
        result[obj_str] = {"rle": rle, "bbox": bbox, "area": area, "source_keyframe": source_kf}
    return result


def load_all_masks_rle(session_dir, cache=None):
    """Load all masks, returning frontend-ready {frame_str: {obj_str: {rle, bbox, area, source_keyframe}}}."""
    if cache is not None:
        encoded = cache.load(MASKS_FILENAME)
    else:
        path = os.path.join(session_dir, MASKS_FILENAME)
        if not os.path.isfile(path):
            return {}
        with open(path) as f:
            encoded = json.load(f)
    result = {}
    for frame_str, obj_masks in encoded.items():
        if frame_str == _VERSIONS_KEY:
            continue
        result[frame_str] = {}
        for obj_str, entry in obj_masks.items():
            rle, source_kf = unpack_entry(entry)
            binary = mask_utils.decode(rle)
            area = int(binary.sum())
            bbox = mask_utils.toBbox(rle).tolist()
            bbox = [int(v) for v in bbox]
            result[frame_str][obj_str] = {"rle": rle, "bbox": bbox, "area": area, "source_keyframe": source_kf}
    return result


def delete_frame_object_mask(session_dir, frame_idx, obj_id, cache=None):
    """Delete a single object's mask from a single frame on disk."""
    if cache is not None:
        encoded = cache.load(MASKS_FILENAME)
    else:
        path = os.path.join(session_dir, MASKS_FILENAME)
        if not os.path.isfile(path):
            return False
        with open(path) as f:
            encoded = json.load(f)
    frame_key = str(frame_idx)
    obj_str = str(obj_id)
    if frame_key in encoded and obj_str in encoded[frame_key]:
        # Clone nested dict before mutating.
        frame_entry = dict(encoded[frame_key])
        del frame_entry[obj_str]
        if frame_entry:
            encoded[frame_key] = frame_entry
        else:
            del encoded[frame_key]

        # Clone _versions before bumping.
        versions = dict(encoded.get(_VERSIONS_KEY, {}))
        versions[frame_key] = versions.get(frame_key, 0) + 1
        encoded[_VERSIONS_KEY] = versions

        if cache is not None:
            cache.save(MASKS_FILENAME, encoded)
        else:
            atomic_json_dump(encoded, path)
        from app.services.gcs_sync import mark_dirty_safe
        mark_dirty_safe("masks.json")
        return True
    return False


def delete_object_masks_by_source(session_dir, obj_id, source_keyframe, direction, current_frame, cache=None):
    """Delete masks for an object filtered by source_keyframe and direction.

    direction: "left" (<= current_frame), "right" (>= current_frame), "this" (== current_frame)
    Returns list of deleted frame indices (sorted).
    """
    if cache is not None:
        encoded = cache.load(MASKS_FILENAME)
    else:
        path = os.path.join(session_dir, MASKS_FILENAME)
        if not os.path.isfile(path):
            return []
        with open(path) as f:
            encoded = json.load(f)
    obj_str = str(obj_id)
    deleted = []
    versions = dict(encoded.get(_VERSIONS_KEY, {}))
    for frame_key in list(encoded.keys()):
        if frame_key == _VERSIONS_KEY:
            continue
        frame_idx = int(frame_key)
        if obj_str not in encoded[frame_key]:
            continue
        _, entry_source = unpack_entry(encoded[frame_key][obj_str])
        if entry_source != source_keyframe:
            continue
        if direction == "left" and frame_idx > current_frame:
            continue
        if direction == "right" and frame_idx < current_frame:
            continue
        if direction == "this" and frame_idx != current_frame:
            continue
        # Clone nested dict before mutating.
        frame_entry = dict(encoded[frame_key])
        del frame_entry[obj_str]
        if frame_entry:
            encoded[frame_key] = frame_entry
        else:
            del encoded[frame_key]
        deleted.append(frame_idx)
        versions[frame_key] = versions.get(frame_key, 0) + 1
    if deleted:
        encoded[_VERSIONS_KEY] = versions
        if cache is not None:
            cache.save(MASKS_FILENAME, encoded)
        else:
            atomic_json_dump(encoded, path)
        from app.services.gcs_sync import mark_dirty_safe
        mark_dirty_safe("masks.json")
    return sorted(deleted)


def delete_frame_object_masks_batch(session_dir, obj_id, frame_indices, cache=None):
    """Delete masks for an object across a list of specific frames.

    Returns sorted list of frame indices where a mask was actually deleted.
    """
    if not frame_indices:
        return []
    if cache is not None:
        encoded = cache.load(MASKS_FILENAME)
    else:
        path = os.path.join(session_dir, MASKS_FILENAME)
        if not os.path.isfile(path):
            return []
        with open(path) as f:
            encoded = json.load(f)
    deleted = []
    obj_key = str(obj_id)
    versions = dict(encoded.get(_VERSIONS_KEY, {}))
    for fi in frame_indices:
        frame_key = str(fi)
        if frame_key in encoded and obj_key in encoded[frame_key]:
            # Clone nested dict before mutating.
            frame_entry = dict(encoded[frame_key])
            del frame_entry[obj_key]
            deleted.append(fi)
            if frame_entry:
                encoded[frame_key] = frame_entry
            else:
                del encoded[frame_key]
            versions[frame_key] = versions.get(frame_key, 0) + 1
    if deleted:
        encoded[_VERSIONS_KEY] = versions
        if cache is not None:
            cache.save(MASKS_FILENAME, encoded)
        else:
            atomic_json_dump(encoded, path)
        from app.services.gcs_sync import mark_dirty_safe
        mark_dirty_safe("masks.json")
    return sorted(deleted)


def update_frame_masks(session_dir, frame_idx, obj_masks, source_keyframe=None, cache=None):
    if cache is not None:
        encoded = cache.load(MASKS_FILENAME)
    else:
        path = os.path.join(session_dir, MASKS_FILENAME)
        if os.path.isfile(path):
            with open(path) as f:
                encoded = json.load(f)
        else:
            encoded = {}
    frame_key = str(frame_idx)
    # Clone nested dict before mutating -- old readers iterating the previous
    # top-level ref must see their original nested dict unchanged.
    frame_entry = dict(encoded.get(frame_key, {}))
    before_objs = set(frame_entry.keys())
    for obj_id, mask in obj_masks.items():
        frame_entry[str(obj_id)] = {
            "rle": _encode_mask(mask),
            "source_keyframe": source_keyframe,
        }
    encoded[frame_key] = frame_entry
    after_objs = set(frame_entry.keys())
    lost = before_objs - after_objs
    if lost:
        # Defense-in-depth: update_frame_masks is an upsert so objects should
        # never drop out. If this log ever fires it means the per-object
        # upsert invariant has been broken somewhere upstream.
        logger.error(
            "mask_storage | UNEXPECTED OBJECT LOSS | frame=%d | lost=%s | before=%s | after=%s",
            frame_idx, sorted(lost), sorted(before_objs), sorted(after_objs),
        )

    # Clone _versions dict before mutating.
    versions = dict(encoded.get(_VERSIONS_KEY, {}))
    versions[frame_key] = versions.get(frame_key, 0) + 1
    encoded[_VERSIONS_KEY] = versions

    if cache is not None:
        cache.save(MASKS_FILENAME, encoded)
    else:
        atomic_json_dump(encoded, path)

    from app.services.gcs_sync import mark_dirty_safe
    mark_dirty_safe("masks.json")
