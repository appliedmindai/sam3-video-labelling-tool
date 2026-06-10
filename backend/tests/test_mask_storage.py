import numpy as np
import pytest

from app.services.mask_storage import (
    save_masks,
    load_masks,
    update_frame_masks,
    remove_object_masks,
    load_frame_masks_rle,
    load_all_masks_rle,
    delete_frame_object_mask,
    delete_object_masks_by_source,
    delete_frame_object_masks_batch,
    get_versions,
)


def _make_mask(h=100, w=100, region=None):
    """Create a test mask, optionally with a filled region (y1, y2, x1, x2)."""
    m = np.zeros((h, w), dtype=np.uint8)
    if region:
        y1, y2, x1, x2 = region
        m[y1:y2, x1:x2] = 1
    return m


def test_save_and_load_masks(tmp_path):
    masks = {
        0: {1: _make_mask()},
        5: {1: np.ones((100, 100), dtype=np.uint8), 2: _make_mask(region=(20, 50, 30, 70))},
    }
    session_dir = str(tmp_path)
    save_masks(session_dir, masks)
    loaded = load_masks(session_dir)
    assert set(loaded.keys()) == {0, 5}
    assert set(loaded[5].keys()) == {1, 2}
    mask_arr, source_kf = loaded[5][2]
    np.testing.assert_array_equal(mask_arr, masks[5][2])
    assert source_kf is None  # old format defaults to None


def test_update_masks_merges(tmp_path):
    session_dir = str(tmp_path)
    initial = {0: {1: np.ones((100, 100), dtype=np.uint8)}}
    save_masks(session_dir, initial)
    update_frame_masks(session_dir, 0, {2: _make_mask()})
    update_frame_masks(session_dir, 5, {1: np.ones((100, 100), dtype=np.uint8)})
    loaded = load_masks(session_dir)
    assert 1 in loaded[0]  # old format obj still present
    assert 2 in loaded[0]  # new format obj added
    assert 1 in loaded[5]


def test_remove_object_masks(tmp_path):
    session_dir = str(tmp_path)
    masks = {
        0: {1: _make_mask(region=(0, 10, 0, 10)), 2: _make_mask(region=(20, 30, 20, 30))},
        3: {1: _make_mask(region=(5, 15, 5, 15)), 2: _make_mask(region=(50, 60, 50, 60))},
        7: {2: _make_mask(region=(10, 20, 10, 20))},
    }
    save_masks(session_dir, masks)
    remove_object_masks(session_dir, 1)
    loaded = load_masks(session_dir)
    # Object 1 removed from all frames
    assert 1 not in loaded[0]
    assert 1 not in loaded[3]
    # Object 2 untouched
    assert 2 in loaded[0]
    assert 2 in loaded[3]
    assert 2 in loaded[7]


def test_remove_object_masks_nonexistent(tmp_path):
    """Removing a non-existent object should be a no-op."""
    session_dir = str(tmp_path)
    masks = {0: {1: _make_mask(region=(0, 10, 0, 10))}}
    save_masks(session_dir, masks)
    remove_object_masks(session_dir, 99)
    loaded = load_masks(session_dir)
    assert 1 in loaded[0]


def test_remove_object_masks_no_file(tmp_path):
    """Removing masks when no masks file exists should not error."""
    session_dir = str(tmp_path)
    remove_object_masks(session_dir, 1)  # no-op, no error


def test_delete_frame_object_mask(tmp_path):
    session_dir = str(tmp_path)
    masks = {
        0: {1: _make_mask(region=(0, 10, 0, 10)), 2: _make_mask(region=(20, 30, 20, 30))},
        5: {1: _make_mask(region=(5, 15, 5, 15))},
    }
    save_masks(session_dir, masks)
    # Delete obj 1 from frame 0
    result = delete_frame_object_mask(session_dir, 0, 1)
    assert result is True
    loaded = load_masks(session_dir)
    assert 1 not in loaded[0]
    assert 2 in loaded[0]
    # Frame 5 obj 1 untouched
    assert 1 in loaded[5]


def test_delete_frame_object_mask_removes_empty_frame(tmp_path):
    """Deleting the last object from a frame should remove the frame entry."""
    session_dir = str(tmp_path)
    masks = {0: {1: _make_mask(region=(0, 10, 0, 10))}}
    save_masks(session_dir, masks)
    delete_frame_object_mask(session_dir, 0, 1)
    loaded = load_masks(session_dir)
    assert 0 not in loaded


def test_delete_frame_object_mask_nonexistent(tmp_path):
    """Deleting a mask that doesn't exist returns False."""
    session_dir = str(tmp_path)
    masks = {0: {1: _make_mask(region=(0, 10, 0, 10))}}
    save_masks(session_dir, masks)
    result = delete_frame_object_mask(session_dir, 0, 99)
    assert result is False
    result = delete_frame_object_mask(session_dir, 99, 1)
    assert result is False


def test_delete_frame_object_mask_no_file(tmp_path):
    """Deleting when no masks file exists returns False."""
    session_dir = str(tmp_path)
    result = delete_frame_object_mask(session_dir, 0, 1)
    assert result is False


def test_load_frame_masks_rle(tmp_path):
    session_dir = str(tmp_path)
    mask = _make_mask(region=(10, 50, 20, 80))
    masks = {3: {1: mask, 2: _make_mask(region=(60, 90, 60, 90))}}
    save_masks(session_dir, masks)
    result = load_frame_masks_rle(session_dir, 3)
    assert "1" in result
    assert "2" in result
    # Each entry has rle, bbox, area
    entry = result["1"]
    assert "rle" in entry
    assert "bbox" in entry
    assert "area" in entry
    assert entry["area"] == int(mask.sum())
    # bbox should be [x, y, w, h]
    bx, by, bw, bh = entry["bbox"]
    assert bx == 20 and by == 10 and bw == 60 and bh == 40


def test_load_frame_masks_rle_empty_frame(tmp_path):
    session_dir = str(tmp_path)
    masks = {0: {1: _make_mask(region=(0, 10, 0, 10))}}
    save_masks(session_dir, masks)
    result = load_frame_masks_rle(session_dir, 99)
    assert result == {}


def test_load_frame_masks_rle_no_file(tmp_path):
    session_dir = str(tmp_path)
    result = load_frame_masks_rle(session_dir, 0)
    assert result == {}


def test_update_then_delete_then_load(tmp_path):
    """Integration: update masks, delete one, verify the rest."""
    session_dir = str(tmp_path)
    save_masks(session_dir, {})
    update_frame_masks(session_dir, 0, {1: _make_mask(region=(0, 10, 0, 10))})
    update_frame_masks(session_dir, 0, {2: _make_mask(region=(20, 30, 20, 30))})
    update_frame_masks(session_dir, 1, {1: _make_mask(region=(5, 15, 5, 15))})
    delete_frame_object_mask(session_dir, 0, 1)
    loaded = load_masks(session_dir)
    assert 1 not in loaded[0]
    assert 2 in loaded[0]
    assert 1 in loaded[1]
    # Verify via RLE endpoint too
    rle = load_frame_masks_rle(session_dir, 0)
    assert "1" not in rle
    assert "2" in rle


# ---------------------------------------------------------------------------
# source_keyframe tests
# ---------------------------------------------------------------------------


def test_update_frame_masks_with_source_keyframe(tmp_path):
    """update_frame_masks stores source_keyframe, retrievable via load_frame_masks_rle."""
    session_dir = str(tmp_path)
    mask = _make_mask(region=(10, 50, 20, 80))
    update_frame_masks(session_dir, 5, {1: mask}, source_keyframe=0)

    result = load_frame_masks_rle(session_dir, 5)
    assert "1" in result
    assert result["1"]["source_keyframe"] == 0
    # Also verify the core rle/bbox/area fields are still present
    assert "rle" in result["1"]
    assert "bbox" in result["1"]
    assert "area" in result["1"]
    assert result["1"]["area"] == int(mask.sum())


def test_update_frame_masks_keyframe_null_source(tmp_path):
    """source_keyframe=None is stored and returned as None."""
    session_dir = str(tmp_path)
    mask = _make_mask(region=(0, 10, 0, 10))
    update_frame_masks(session_dir, 0, {1: mask}, source_keyframe=None)

    result = load_frame_masks_rle(session_dir, 0)
    assert result["1"]["source_keyframe"] is None


def test_load_masks_returns_source_keyframe(tmp_path):
    """load_masks returns (mask_array, source_keyframe) tuples."""
    session_dir = str(tmp_path)
    mask = _make_mask(region=(10, 50, 20, 80))
    update_frame_masks(session_dir, 3, {1: mask}, source_keyframe=0)
    update_frame_masks(session_dir, 7, {2: mask}, source_keyframe=3)

    loaded = load_masks(session_dir)
    # Each value should be a (mask, source_keyframe) tuple
    mask_arr, source_kf = loaded[3][1]
    np.testing.assert_array_equal(mask_arr, mask)
    assert source_kf == 0

    mask_arr2, source_kf2 = loaded[7][2]
    np.testing.assert_array_equal(mask_arr2, mask)
    assert source_kf2 == 3


def test_backward_compat_missing_source_keyframe(tmp_path):
    """Old-format data (via save_masks) loads with source_keyframe=None."""
    session_dir = str(tmp_path)
    mask = _make_mask(region=(10, 50, 20, 80))
    # save_masks writes old flat RLE format
    save_masks(session_dir, {0: {1: mask}})

    # load_masks should return (mask, None) tuples
    loaded = load_masks(session_dir)
    mask_arr, source_kf = loaded[0][1]
    np.testing.assert_array_equal(mask_arr, mask)
    assert source_kf is None

    # load_frame_masks_rle should return source_keyframe=None
    result = load_frame_masks_rle(session_dir, 0)
    assert result["1"]["source_keyframe"] is None


# ---------------------------------------------------------------------------
# delete_object_masks_by_source tests
# ---------------------------------------------------------------------------


def test_delete_masks_by_source_right(tmp_path):
    """Delete right: removes frames >= current_frame with matching source_keyframe."""
    session_dir = str(tmp_path)
    save_masks(session_dir, {})
    for fi in range(5):
        update_frame_masks(session_dir, fi, {1: _make_mask(region=(0, 10, 0, 10))}, source_keyframe=0)
    deleted = delete_object_masks_by_source(session_dir, 1, source_keyframe=0, direction="right", current_frame=2)
    assert sorted(deleted) == [2, 3, 4]
    loaded = load_masks(session_dir)
    assert 1 in loaded[0]
    assert 1 in loaded[1]
    assert 2 not in loaded
    assert 3 not in loaded
    assert 4 not in loaded


def test_delete_masks_by_source_left(tmp_path):
    """Delete left: removes frames <= current_frame with matching source_keyframe."""
    session_dir = str(tmp_path)
    save_masks(session_dir, {})
    for fi in range(5):
        update_frame_masks(session_dir, fi, {1: _make_mask(region=(0, 10, 0, 10))}, source_keyframe=0)
    deleted = delete_object_masks_by_source(session_dir, 1, source_keyframe=0, direction="left", current_frame=2)
    assert sorted(deleted) == [0, 1, 2]
    loaded = load_masks(session_dir)
    assert 0 not in loaded
    assert 1 not in loaded
    assert 2 not in loaded
    assert 1 in loaded[3]
    assert 1 in loaded[4]


def test_delete_masks_by_source_this(tmp_path):
    """Delete this: removes only the current frame."""
    session_dir = str(tmp_path)
    save_masks(session_dir, {})
    for fi in range(3):
        update_frame_masks(session_dir, fi, {1: _make_mask(region=(0, 10, 0, 10))}, source_keyframe=0)
    deleted = delete_object_masks_by_source(session_dir, 1, source_keyframe=0, direction="this", current_frame=1)
    assert deleted == [1]
    loaded = load_masks(session_dir)
    assert 1 in loaded[0]
    assert 1 not in loaded.get(1, {})
    assert 1 in loaded[2]


def test_delete_masks_by_source_skips_other_keyframe(tmp_path):
    """Only deletes frames matching the given source_keyframe."""
    session_dir = str(tmp_path)
    save_masks(session_dir, {})
    for fi in range(3):
        update_frame_masks(session_dir, fi, {1: _make_mask(region=(0, 10, 0, 10))}, source_keyframe=0)
    for fi in range(3, 5):
        update_frame_masks(session_dir, fi, {1: _make_mask(region=(0, 10, 0, 10))}, source_keyframe=3)
    deleted = delete_object_masks_by_source(session_dir, 1, source_keyframe=0, direction="right", current_frame=0)
    assert sorted(deleted) == [0, 1, 2]
    loaded = load_masks(session_dir)
    assert 1 in loaded[3]
    assert 1 in loaded[4]


def test_delete_masks_by_source_preserves_other_objects(tmp_path):
    """Deleting one object's masks should not affect other objects on the same frames."""
    session_dir = str(tmp_path)
    save_masks(session_dir, {})
    for fi in range(3):
        update_frame_masks(session_dir, fi, {1: _make_mask(region=(0, 10, 0, 10))}, source_keyframe=0)
        update_frame_masks(session_dir, fi, {2: _make_mask(region=(20, 30, 20, 30))}, source_keyframe=0)
    deleted = delete_object_masks_by_source(session_dir, 1, source_keyframe=0, direction="right", current_frame=0)
    assert sorted(deleted) == [0, 1, 2]
    loaded = load_masks(session_dir)
    for fi in range(3):
        assert 2 in loaded[fi]


def test_delete_masks_by_source_null_keyframe(tmp_path):
    """Deleting masks with source_keyframe=None (keyframe itself)."""
    session_dir = str(tmp_path)
    save_masks(session_dir, {})
    update_frame_masks(session_dir, 5, {1: _make_mask(region=(0, 10, 0, 10))}, source_keyframe=None)
    deleted = delete_object_masks_by_source(session_dir, 1, source_keyframe=None, direction="this", current_frame=5)
    assert deleted == [5]


# ---------------------------------------------------------------------------
# delete_frame_object_masks_batch tests
# ---------------------------------------------------------------------------


def test_delete_frame_object_masks_batch(tmp_path):
    """Batch delete removes masks for specified frames only."""
    masks = {}
    for f in [0, 5, 10, 15, 20]:
        masks[f] = {1: _make_mask()}
    save_masks(tmp_path, masks)

    deleted = delete_frame_object_masks_batch(tmp_path, 1, [5, 10, 15])
    assert sorted(deleted) == [5, 10, 15]

    loaded = load_masks(tmp_path)
    assert 0 in loaded
    assert 20 in loaded
    assert 5 not in loaded
    assert 10 not in loaded
    assert 15 not in loaded


def test_delete_frame_object_masks_batch_ignores_missing(tmp_path):
    """Batch delete skips frames that don't have masks and returns only actually deleted."""
    masks = {0: {1: _make_mask()}, 5: {1: _make_mask()}}
    save_masks(tmp_path, masks)

    deleted = delete_frame_object_masks_batch(tmp_path, 1, [0, 3, 99])
    assert sorted(deleted) == [0]

    loaded = load_masks(tmp_path)
    assert 5 in loaded
    assert 0 not in loaded


def test_delete_frame_object_masks_batch_preserves_other_objects(tmp_path):
    """Batch delete only removes the target object, leaves others intact."""
    masks = {5: {1: _make_mask(), 2: _make_mask()}, 10: {1: _make_mask()}}
    save_masks(tmp_path, masks)

    deleted = delete_frame_object_masks_batch(tmp_path, 1, [5, 10])
    assert sorted(deleted) == [5, 10]

    loaded = load_masks(tmp_path)
    assert 5 in loaded
    assert 2 in loaded[5]
    assert 1 not in loaded[5]
    assert 10 not in loaded


def test_delete_frame_object_masks_batch_empty_list(tmp_path):
    """Batch delete with empty list is a no-op."""
    masks = {0: {1: _make_mask()}}
    save_masks(tmp_path, masks)

    deleted = delete_frame_object_masks_batch(tmp_path, 1, [])
    assert deleted == []

    loaded = load_masks(tmp_path)
    assert 0 in loaded


# ---------------------------------------------------------------------------
# SessionCache integration tests
# ---------------------------------------------------------------------------


from app.services.session_cache import SessionCache


def test_update_frame_masks_with_cache(tmp_path):
    session_dir = str(tmp_path)
    cache = SessionCache(session_dir)
    mask = _make_mask(region=(0, 10, 0, 10))
    update_frame_masks(session_dir, 0, {1: mask}, cache=cache)
    update_frame_masks(session_dir, 5, {2: mask}, cache=cache)
    loaded = load_masks(session_dir)
    assert 1 in loaded[0]
    assert 2 in loaded[5]


def test_load_frame_masks_rle_with_cache(tmp_path):
    session_dir = str(tmp_path)
    cache = SessionCache(session_dir)
    mask = _make_mask(region=(10, 50, 20, 80))
    update_frame_masks(session_dir, 3, {1: mask}, cache=cache)
    result = load_frame_masks_rle(session_dir, 3, cache=cache)
    assert "1" in result
    assert result["1"]["area"] == int(mask.sum())


def test_delete_frame_object_mask_with_cache(tmp_path):
    session_dir = str(tmp_path)
    cache = SessionCache(session_dir)
    mask = _make_mask(region=(0, 10, 0, 10))
    update_frame_masks(session_dir, 0, {1: mask, 2: mask}, cache=cache)
    deleted = delete_frame_object_mask(session_dir, 0, 1, cache=cache)
    assert deleted is True
    result = load_frame_masks_rle(session_dir, 0, cache=cache)
    assert "1" not in result
    assert "2" in result


# ---------------------------------------------------------------------------
# Per-frame version tracking tests
# ---------------------------------------------------------------------------


def test_update_frame_masks_increments_version(tmp_path):
    session_dir = str(tmp_path)
    cache = SessionCache(session_dir)
    mask = _make_mask(region=(0, 10, 0, 10))
    update_frame_masks(session_dir, 0, {1: mask}, cache=cache)
    versions = get_versions(session_dir, cache=cache)
    assert versions["0"] == 1
    update_frame_masks(session_dir, 0, {2: mask}, cache=cache)
    versions = get_versions(session_dir, cache=cache)
    assert versions["0"] == 2


def test_delete_increments_version(tmp_path):
    session_dir = str(tmp_path)
    cache = SessionCache(session_dir)
    mask = _make_mask(region=(0, 10, 0, 10))
    update_frame_masks(session_dir, 0, {1: mask, 2: mask}, cache=cache)
    v1 = get_versions(session_dir, cache=cache)["0"]
    delete_frame_object_mask(session_dir, 0, 1, cache=cache)
    v2 = get_versions(session_dir, cache=cache)["0"]
    assert v2 == v1 + 1


def test_versions_backward_compat_no_key(tmp_path):
    session_dir = str(tmp_path)
    save_masks(session_dir, {0: {1: _make_mask()}})
    versions = get_versions(session_dir)
    assert versions == {}


def test_versions_not_in_mask_output(tmp_path):
    session_dir = str(tmp_path)
    cache = SessionCache(session_dir)
    mask = _make_mask(region=(0, 10, 0, 10))
    update_frame_masks(session_dir, 0, {1: mask}, cache=cache)
    result = load_all_masks_rle(session_dir, cache=cache)
    assert "_versions" not in result
