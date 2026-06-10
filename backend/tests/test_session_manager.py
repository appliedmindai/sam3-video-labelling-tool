import os
import json
import pytest

@pytest.fixture
def session_dir(tmp_path):
    d = tmp_path / "test-session"
    d.mkdir()
    return str(d)

def test_save_and_load_state(session_dir):
    from app.services.session_manager import save_state, load_state
    state = {"classes": [{"id": 1, "name": "person", "color": "#FF0000"}], "objects": [{"obj_id": 1, "class_id": 1}]}
    save_state(session_dir, state)
    loaded = load_state(session_dir)
    # save_state bumps the version counter from 0 -> 1 (#80 lost-update guard).
    assert loaded == {**state, "version": 1}

def test_load_state_returns_default_when_empty(session_dir):
    from app.services.session_manager import load_state
    state = load_state(session_dir)
    assert state == {"classes": [], "objects": [], "version": 0}

def test_add_class(session_dir):
    from app.services.session_manager import load_state, add_class
    state = load_state(session_dir)
    state = add_class(state, "person", "#FF0000")
    assert len(state["classes"]) == 1
    assert state["classes"][0]["name"] == "person"
    assert state["classes"][0]["id"] == 1
    state = add_class(state, "car", "#00FF00")
    assert len(state["classes"]) == 2
    assert state["classes"][1]["id"] == 2

def test_save_and_load_state_with_bbox_padding(session_dir):
    from app.services.session_manager import save_state, load_state
    state = {
        "classes": [{"id": 1, "name": "person", "color": "#FF0000"}],
        "objects": [{"obj_id": 1, "class_id": 1}],
        "bbox_padding": {
            "1": {
                "0": {"top": 5, "bottom": 5, "left": 10, "right": 10},
                "12": {"top": 3, "bottom": 3, "left": 0, "right": 0},
            }
        },
    }
    save_state(session_dir, state)
    loaded = load_state(session_dir)
    assert loaded == {**state, "version": 1}
    assert loaded["bbox_padding"]["1"]["0"]["top"] == 5
    assert loaded["bbox_padding"]["1"]["12"]["left"] == 0

def test_class_operations_preserve_bbox_padding(session_dir):
    from app.services.session_manager import save_state, load_state, add_class
    state = {
        "classes": [],
        "objects": [],
        "bbox_padding": {"1": {"0": {"top": 5, "bottom": 5, "left": 10, "right": 10}}},
    }
    save_state(session_dir, state)
    state = load_state(session_dir)
    state = add_class(state, "car", "#00FF00")
    save_state(session_dir, state)
    loaded = load_state(session_dir)
    assert "bbox_padding" in loaded
    assert loaded["bbox_padding"]["1"]["0"]["top"] == 5
    assert len(loaded["classes"]) == 1

def test_delete_class(session_dir):
    from app.services.session_manager import load_state, add_class, delete_class
    state = load_state(session_dir)
    state = add_class(state, "person", "#FF0000")
    state = add_class(state, "car", "#00FF00")
    state = delete_class(state, 1)
    assert len(state["classes"]) == 1
    assert state["classes"][0]["name"] == "car"

def test_reassign_object(session_dir):
    from app.services.session_manager import load_state, add_class, reassign_object
    state = load_state(session_dir)
    state = add_class(state, "person", "#FF0000")
    state = add_class(state, "car", "#00FF00")
    state["objects"].append({"obj_id": 1, "class_id": 1})
    state = reassign_object(state, 1, 2)
    assert state["objects"][0]["class_id"] == 2

def test_reassign_object_invalid_class(session_dir):
    from app.services.session_manager import load_state, add_class, reassign_object
    state = load_state(session_dir)
    state = add_class(state, "person", "#FF0000")
    state["objects"].append({"obj_id": 1, "class_id": 1})
    with pytest.raises(ValueError):
        reassign_object(state, 1, 999)

def test_reassign_object_invalid_object(session_dir):
    from app.services.session_manager import load_state, add_class, reassign_object
    state = load_state(session_dir)
    state = add_class(state, "person", "#FF0000")
    with pytest.raises(ValueError):
        reassign_object(state, 999, 1)


def test_load_state_with_cache_then_disk_populated(tmp_path):
    """If cache was populated with {} when disk was empty, a subsequent load
    must pick up real data once the file appears on disk.

    This guards against the 'cache holds {} forever' bug that caused
    state.json wipes during the DownloadSessionStep race.
    """
    import json
    from app.services.session_manager import load_state
    from app.services.session_cache import SessionCache

    session_dir = str(tmp_path)
    cache = SessionCache(session_dir)

    # First load: file doesn't exist, cache stores {}
    first = load_state(session_dir, cache=cache)
    assert first == {"classes": [], "objects": [], "version": 0}

    # Disk now has real data (simulates DownloadSessionStep finishing)
    real = {"classes": [{"id": 1, "name": "x", "color": "#000"}],
            "objects": [{"obj_id": 1, "class_id": 1}]}
    import os
    with open(os.path.join(session_dir, "state.json"), "w") as f:
        json.dump(real, f)

    second = load_state(session_dir, cache=cache)
    # Legacy state.json without a `version` field is normalized to version: 0.
    assert second == {**real, "version": 0}, f"Expected real data, got {second}"


def test_load_state_returns_real_empty_state_as_is(tmp_path):
    """If state.json genuinely exists with empty classes and objects,
    load_state must preserve all on-disk fields (including bbox_padding).
    """
    import json
    import os
    from app.services.session_manager import load_state
    from app.services.session_cache import SessionCache

    session_dir = str(tmp_path)
    real = {"classes": [], "objects": [], "bbox_padding": {"1": {"0": {"top": 1}}}}
    with open(os.path.join(session_dir, "state.json"), "w") as f:
        json.dump(real, f)

    cache = SessionCache(session_dir)
    result = load_state(session_dir, cache=cache)
    # Legacy state.json is normalized with synthesized version: 0.
    expected = {**real, "version": 0}
    assert result == expected, f"Expected {expected}, got {result}"
    # Disk path (no cache) must agree
    assert load_state(session_dir) == expected
