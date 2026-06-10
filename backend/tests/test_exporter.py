import json
import os
import numpy as np
import pytest
from PIL import Image

from app.services.exporter import (
    _pad_bbox,
    _normalise_padding,
    _build_coco_annotations,
    export_coco,
)


@pytest.fixture
def mock_session(tmp_path):
    session_dir = tmp_path / "session"
    frames_dir = session_dir / "frames"
    frames_dir.mkdir(parents=True)
    for i in range(5):
        img = Image.new("RGB", (320, 240), color=(i * 50, 0, 0))
        img.save(frames_dir / f"{i:05d}.jpg")
    state = {
        "classes": [{"id": 1, "name": "person", "color": "#FF0000"}, {"id": 2, "name": "car", "color": "#00FF00"}],
        "objects": [{"obj_id": 1, "class_id": 1}, {"obj_id": 2, "class_id": 2}],
    }
    with open(session_dir / "state.json", "w") as f:
        json.dump(state, f)
    masks = {}
    for frame_idx in range(5):
        masks[frame_idx] = {}
        mask1 = np.zeros((240, 320), dtype=np.uint8)
        mask1[20:80, 30:90] = 1
        masks[frame_idx][1] = mask1
        mask2 = np.zeros((240, 320), dtype=np.uint8)
        mask2[150:220, 200:300] = 1
        masks[frame_idx][2] = mask2
    return str(session_dir), state, masks


# --- _pad_bbox ---

def test_pad_bbox_no_padding():
    bbox = [50, 50, 100, 100]
    padding = {"top": 0, "bottom": 0, "left": 0, "right": 0}
    assert _pad_bbox(bbox, padding, 640, 480) == bbox


def test_pad_bbox_uniform():
    bbox = [100, 100, 100, 100]
    padding = {"top": 10, "bottom": 10, "left": 10, "right": 10}
    result = _pad_bbox(bbox, padding, 640, 480)
    # left expands by 10% of w=100 -> 10px, so new x = 90
    assert result[0] == 90
    # top expands by 10% of h=100 -> 10px, so new y = 90
    assert result[1] == 90
    # width grows by 20px (10 left + 10 right)
    assert result[2] == 120
    # height grows by 20px (10 top + 10 bottom)
    assert result[3] == 120


def test_pad_bbox_clamped():
    """Padding should not exceed image bounds."""
    bbox = [0, 0, 100, 100]
    padding = {"top": 50, "bottom": 50, "left": 50, "right": 50}
    result = _pad_bbox(bbox, padding, 120, 120)
    assert result[0] >= 0
    assert result[1] >= 0
    assert result[0] + result[2] <= 120
    assert result[1] + result[3] <= 120


# --- _normalise_padding ---

def test_normalise_padding_number():
    assert _normalise_padding(0) == {}
    assert _normalise_padding(5) == {}


def test_normalise_padding_flat_per_object():
    raw = {"1": {"top": 5, "bottom": 5, "left": 10, "right": 10}}
    result = _normalise_padding(raw)
    assert 1 in result
    # Should be wrapped as a single keyframe at frame 0
    assert 0 in result[1]
    assert result[1][0]["top"] == 5


def test_normalise_padding_per_keyframe():
    raw = {
        "1": {
            "0": {"top": 5, "bottom": 5, "left": 0, "right": 0},
            "10": {"top": 15, "bottom": 15, "left": 10, "right": 10},
        }
    }
    result = _normalise_padding(raw)
    assert 1 in result
    assert 0 in result[1]
    assert 10 in result[1]
    assert result[1][0]["top"] == 5
    assert result[1][10]["top"] == 15


# --- Export with padding ---

def test_export_coco(mock_session, tmp_path):
    session_dir, state, masks = mock_session
    output_dir = str(tmp_path / "coco_export")
    export_coco(session_dir, state, masks, output_dir)
    ann_path = os.path.join(output_dir, "_annotations.coco.json")
    assert os.path.isfile(ann_path)
    with open(ann_path) as f:
        coco = json.load(f)
    assert len(coco["images"]) == 5
    assert len(coco["categories"]) == 2
    assert len(coco["annotations"]) == 10
    ann = coco["annotations"][0]
    assert "bbox" in ann
    assert "segmentation" in ann
    assert "category_id" in ann
    assert "area" in ann
    assert len(ann["bbox"]) == 4
    assert os.path.isfile(os.path.join(output_dir, "00000.jpg"))


def test_export_coco_with_per_keyframe_padding(mock_session, tmp_path):
    """Per-keyframe padding with bare ndarrays (old format) falls back to frame_idx lookup."""
    session_dir, state, masks = mock_session
    output_dir = str(tmp_path / "coco_padded")
    # Object 1 has padding at frame 0 = 0%, frame 4 = 50%
    # With bare ndarrays (no source_keyframe), each frame looks up its own frame_idx
    padding = {
        "1": {
            "0": {"top": 0, "bottom": 0, "left": 0, "right": 0},
            "4": {"top": 50, "bottom": 50, "left": 50, "right": 50},
        }
    }
    export_coco(session_dir, state, masks, output_dir, bbox_padding_pct=padding)
    with open(os.path.join(output_dir, "_annotations.coco.json")) as f:
        coco = json.load(f)
    # Find obj 1 annotations for frame 0 and frame 4
    obj1_anns = [a for a in coco["annotations"] if a["category_id"] == 1]
    frame0_ann = [a for a in obj1_anns if a["image_id"] == 1][0]  # image_id 1 = frame 0
    frame4_ann = [a for a in obj1_anns if a["image_id"] == 5][0]  # image_id 5 = frame 4
    # Frame 0: 0% padding -> tight bbox
    # Frame 4: 50% padding -> larger bbox
    assert frame4_ann["bbox"][2] > frame0_ann["bbox"][2]  # wider
    assert frame4_ann["bbox"][3] > frame0_ann["bbox"][3]  # taller
    # Frames 1-3 have no padding defined and no source_keyframe -> fall back to frame_idx
    # Since frames 1,2,3 have no entry in the padding dict, they get zero padding
    frame2_ann = [a for a in obj1_anns if a["image_id"] == 3][0]
    assert frame2_ann["bbox"][2] == frame0_ann["bbox"][2]  # same as tight (no padding)


def test_build_coco_annotations_no_padding(mock_session):
    """Annotations without padding should have tight bboxes."""
    session_dir, state, masks = mock_session
    filenames = [f"{i:05d}.jpg" for i in range(5)]
    coco = _build_coco_annotations(state, masks, filenames, 320, 240)
    assert len(coco["annotations"]) == 10
    # All bboxes should be tight (no padding applied)
    for ann in coco["annotations"]:
        bx, by, bw, bh = ann["bbox"]
        assert bw > 0 and bh > 0


def test_build_coco_annotations_bare_ndarray_fallback(mock_session):
    """Bare ndarrays (old format) fall back to frame_idx for padding lookup."""
    session_dir, state, masks = mock_session
    filenames = [f"{i:05d}.jpg" for i in range(5)]
    # Padding defined only at frame 0 and frame 4
    padding = {
        "1": {
            "0": {"top": 0, "bottom": 0, "left": 0, "right": 0},
            "4": {"top": 40, "bottom": 40, "left": 40, "right": 40},
        }
    }
    coco = _build_coco_annotations(state, masks, filenames, 320, 240, bbox_padding_pct=padding)
    obj1_anns = sorted(
        [a for a in coco["annotations"] if a["category_id"] == 1],
        key=lambda a: a["image_id"],
    )
    # Frame 0: 0% padding (exact match in dict)
    # Frames 1,2,3: no entry in padding dict -> zero padding (same as frame 0)
    # Frame 4: 40% padding
    bbox0 = obj1_anns[0]["bbox"]
    bbox2 = obj1_anns[2]["bbox"]
    bbox4 = obj1_anns[4]["bbox"]
    # Frames 0-3 should all have the same (tight) bbox width
    assert bbox0[2] == bbox2[2]
    # Frame 4 should be wider due to 40% padding
    assert bbox4[2] > bbox0[2]


def test_build_coco_annotations_keyframe_sourced_padding(mock_session):
    """Padding applied via source_keyframe: all frames from same keyframe get same padding."""
    session_dir, state, masks = mock_session
    filenames = [f"{i:05d}.jpg" for i in range(5)]
    # Convert masks to new format: (mask_array, source_keyframe)
    sourced_masks = {}
    for fi, obj_masks in masks.items():
        sourced_masks[fi] = {oid: (m, 0) for oid, m in obj_masks.items()}
    padding = {"1": {"0": {"top": 20, "bottom": 20, "left": 20, "right": 20}}}
    coco = _build_coco_annotations(state, sourced_masks, filenames, 320, 240, bbox_padding_pct=padding)
    obj1_anns = sorted(
        [a for a in coco["annotations"] if a["category_id"] == 1],
        key=lambda a: a["image_id"],
    )
    widths = [a["bbox"][2] for a in obj1_anns]
    heights = [a["bbox"][3] for a in obj1_anns]
    assert len(set(widths)) == 1, f"Expected uniform widths, got {widths}"
    assert len(set(heights)) == 1, f"Expected uniform heights, got {heights}"
    tight_w = 60  # mask region is 30:90 = 60px wide
    assert widths[0] > tight_w


def test_build_coco_annotations_noun_phrase(mock_session):
    """Annotations should include noun_phrase matching the class name."""
    session_dir, state, masks = mock_session
    filenames = [f"{i:05d}.jpg" for i in range(5)]
    coco = _build_coco_annotations(state, masks, filenames, 320, 240)
    # All annotations for class 1 ("person") should have noun_phrase = "person"
    for ann in coco["annotations"]:
        if ann["category_id"] == 1:
            assert ann["noun_phrase"] == "person"
        elif ann["category_id"] == 2:
            assert ann["noun_phrase"] == "car"


def test_build_coco_annotations_two_keyframes_different_padding(mock_session):
    """Frames from different keyframes get different padding."""
    session_dir, state, masks = mock_session
    filenames = [f"{i:05d}.jpg" for i in range(5)]
    sourced_masks = {}
    for fi, obj_masks in masks.items():
        source_kf = 0 if fi < 3 else 3
        sourced_masks[fi] = {oid: (m, source_kf) for oid, m in obj_masks.items()}
    padding = {
        "1": {
            "0": {"top": 0, "bottom": 0, "left": 0, "right": 0},
            "3": {"top": 40, "bottom": 40, "left": 40, "right": 40},
        }
    }
    coco = _build_coco_annotations(state, sourced_masks, filenames, 320, 240, bbox_padding_pct=padding)
    obj1_anns = sorted(
        [a for a in coco["annotations"] if a["category_id"] == 1],
        key=lambda a: a["image_id"],
    )
    assert obj1_anns[0]["bbox"][2] < obj1_anns[3]["bbox"][2]
    assert obj1_anns[0]["bbox"][2] == obj1_anns[1]["bbox"][2] == obj1_anns[2]["bbox"][2]
    assert obj1_anns[3]["bbox"][2] == obj1_anns[4]["bbox"][2]
