import json
import os
import shutil
from datetime import datetime

import numpy as np
from PIL import Image

from app.services.sam3_service import mask_to_polygons, mask_to_bbox


def _pad_bbox(bbox, padding, img_width, img_height):
    """Expand a COCO [x, y, w, h] bbox by per-side padding percentages, clamped to image bounds.

    padding: a dict with keys top/bottom/left/right (percentages)
    """
    x, y, w, h = bbox
    pad_top = h * (padding.get("top", 0) / 100)
    pad_bottom = h * (padding.get("bottom", 0) / 100)
    pad_left = w * (padding.get("left", 0) / 100)
    pad_right = w * (padding.get("right", 0) / 100)
    if pad_top == 0 and pad_bottom == 0 and pad_left == 0 and pad_right == 0:
        return bbox
    new_x = max(0, x - pad_left)
    new_y = max(0, y - pad_top)
    new_w = min(img_width - new_x, w + pad_left + pad_right)
    new_h = min(img_height - new_y, h + pad_top + pad_bottom)
    return [int(new_x), int(new_y), int(new_w), int(new_h)]


def _normalise_padding(bbox_padding_pct):
    """Normalise bbox_padding_pct into per-object keyframe maps.

    Accepts:
      - 0 or a number (legacy: uniform)
      - {obj_id: {top,bottom,left,right}} (flat per-object)
      - {obj_id: {frame_idx: {top,bottom,left,right}}} (per-keyframe)

    Returns: dict of {int(obj_id): {int(frame_idx): {top,bottom,left,right}}}
    """
    if not isinstance(bbox_padding_pct, dict):
        return {}
    result = {}
    for obj_str, val in bbox_padding_pct.items():
        obj_id = int(obj_str)
        if not isinstance(val, dict):
            continue
        # Detect format: if value has "top" key, it's flat per-object
        if "top" in val:
            # Flat per-object — treat as a single keyframe at frame 0 that applies everywhere
            result[obj_id] = {0: val}
        else:
            # Per-keyframe: {frame_idx_str: {top,bottom,left,right}}
            result[obj_id] = {int(f): p for f, p in val.items()}
    return result


def _build_coco_annotations(state, masks, image_filenames, image_width, image_height, bbox_padding_pct=0):
    """Build COCO annotations.

    bbox_padding_pct can be:
      - a number (legacy: no padding)
      - a dict keyed by obj_id mapping to per-side padding dicts (flat)
      - a dict keyed by obj_id mapping to per-frame keyframe dicts
    """
    obj_to_class = {o["obj_id"]: o["class_id"] for o in state["objects"]}
    class_id_to_name = {c["id"]: c["name"] for c in state["classes"]}
    per_obj_keyframes = _normalise_padding(bbox_padding_pct)
    images = []
    annotations = []
    ann_id = 1
    for frame_idx, filename in enumerate(image_filenames):
        image_id = frame_idx + 1
        images.append({"id": image_id, "file_name": filename, "width": image_width, "height": image_height})
        if frame_idx not in masks:
            continue
        for obj_id, mask_entry in masks[frame_idx].items():
            # Handle both old format (bare ndarray) and new format (mask, source_keyframe)
            if isinstance(mask_entry, tuple):
                mask, source_kf = mask_entry
            else:
                mask, source_kf = mask_entry, None
            class_id = obj_to_class.get(obj_id)
            if class_id is None:
                continue
            polygons = mask_to_polygons(mask)
            if not polygons:
                continue
            bbox = mask_to_bbox(mask)
            # Look up padding by source keyframe (uniform), falling back to frame_idx
            keyframe_for_padding = source_kf if source_kf is not None else frame_idx
            obj_kf = per_obj_keyframes.get(obj_id, {})
            padding = obj_kf.get(keyframe_for_padding, {"top": 0, "bottom": 0, "left": 0, "right": 0})
            bbox = _pad_bbox(bbox, padding, image_width, image_height)
            area = int(mask.sum())
            ann = {
                "id": ann_id, "image_id": image_id, "category_id": class_id,
                "bbox": bbox, "area": area, "segmentation": polygons, "iscrowd": 0,
            }
            # Add noun_phrase for SAM 3 fine-tuning format compatibility
            class_name = class_id_to_name.get(class_id)
            if class_name:
                ann["noun_phrase"] = class_name
            annotations.append(ann)
            ann_id += 1
    categories = [{"id": c["id"], "name": c["name"], "supercategory": "object"} for c in state["classes"]]
    return {
        "info": {"description": "SAM 3 Annotation Export", "version": "1.0", "year": datetime.now().year, "date_created": datetime.now().strftime("%Y-%m-%d")},
        "licenses": [], "images": images, "categories": categories, "annotations": annotations,
    }


def export_coco(session_dir, state, masks, output_dir, bbox_padding_pct=0):
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(session_dir, "frames")
    filenames = sorted(f for f in os.listdir(frames_dir) if f.endswith(".jpg"))
    for f in filenames:
        shutil.copy2(os.path.join(frames_dir, f), os.path.join(output_dir, f))
    first_frame = Image.open(os.path.join(frames_dir, filenames[0]))
    w, h = first_frame.size
    coco = _build_coco_annotations(state, masks, filenames, w, h, bbox_padding_pct=bbox_padding_pct)
    with open(os.path.join(output_dir, "_annotations.coco.json"), "w") as f:
        json.dump(coco, f, indent=2)
    return output_dir


