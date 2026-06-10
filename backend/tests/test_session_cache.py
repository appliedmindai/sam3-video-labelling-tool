import json
import os

from app.services.session_cache import SessionCache


def test_load_returns_empty_dict_when_file_missing(tmp_path):
    """cache.load returns {} when the file doesn't exist on disk."""
    cache = SessionCache(str(tmp_path))
    result = cache.load("nonexistent.json")
    assert result == {}


def test_load_reads_from_disk_on_first_call(tmp_path):
    """First call to load reads JSON from disk and returns its content."""
    data = {"objects": {"1": {"name": "cat"}}, "nextId": 2}
    with open(os.path.join(str(tmp_path), "state.json"), "w") as f:
        json.dump(data, f)

    cache = SessionCache(str(tmp_path))
    result = cache.load("state.json")
    assert result == data


def test_load_returns_cached_dict_on_second_call(tmp_path):
    """Second call returns the cached data, even if disk has changed.

    Note: load() returns a shallow top-level copy on each call (reader/writer
    safety contract), so successive calls return equal-but-distinct dicts.
    What matters is that the content reflects the cached value, not a disk
    re-read."""
    data = {"key": "value"}
    filepath = os.path.join(str(tmp_path), "state.json")
    with open(filepath, "w") as f:
        json.dump(data, f)

    cache = SessionCache(str(tmp_path))
    first = cache.load("state.json")
    assert first == data

    # Overwrite the file on disk with different content
    with open(filepath, "w") as f:
        json.dump({"key": "changed"}, f)

    second = cache.load("state.json")
    # Content is served from cache, not re-read from disk
    assert second == {"key": "value"}
    # Top-level copies are independent -- mutating one must not affect the other
    assert second is not first
    first["mutated"] = True
    assert "mutated" not in cache.load("state.json")


def test_save_updates_cache_and_writes_to_disk(tmp_path):
    """save() updates the in-memory cache and writes to disk."""
    cache = SessionCache(str(tmp_path))
    data = {"objects": {"1": {"name": "dog"}}}

    cache.save("state.json", data)

    # In-memory cache should return the data
    assert cache.load("state.json") == data

    # Disk should have the data
    filepath = os.path.join(str(tmp_path), "state.json")
    with open(filepath) as f:
        on_disk = json.load(f)
    assert on_disk == data


def test_invalidate_clears_cached_entry(tmp_path):
    """invalidate() forces a re-read from disk on next load."""
    data = {"key": "original"}
    filepath = os.path.join(str(tmp_path), "state.json")
    with open(filepath, "w") as f:
        json.dump(data, f)

    cache = SessionCache(str(tmp_path))
    first = cache.load("state.json")
    assert first == {"key": "original"}

    # Change the file on disk
    with open(filepath, "w") as f:
        json.dump({"key": "updated"}, f)

    # Invalidate forces re-read
    cache.invalidate("state.json")
    second = cache.load("state.json")
    assert second == {"key": "updated"}
    assert second is not first


def test_clear_removes_all_cached_entries(tmp_path):
    """clear() drops all cached entries, forcing re-reads on next load."""
    filepath_a = os.path.join(str(tmp_path), "a.json")
    filepath_b = os.path.join(str(tmp_path), "b.json")
    with open(filepath_a, "w") as f:
        json.dump({"file": "a"}, f)
    with open(filepath_b, "w") as f:
        json.dump({"file": "b"}, f)

    cache = SessionCache(str(tmp_path))
    first_a = cache.load("a.json")
    first_b = cache.load("b.json")

    # Change both files on disk
    with open(filepath_a, "w") as f:
        json.dump({"file": "a_changed"}, f)
    with open(filepath_b, "w") as f:
        json.dump({"file": "b_changed"}, f)

    # Before clear, cached content is served (disk changes ignored)
    assert cache.load("a.json") == {"file": "a"}

    cache.clear()

    # After clear, both re-read from disk
    second_a = cache.load("a.json")
    second_b = cache.load("b.json")
    assert second_a == {"file": "a_changed"}
    assert second_b == {"file": "b_changed"}


def test_load_refreshes_when_cached_empty_but_disk_has_data(tmp_path):
    """Cache must not stick on an empty {} when disk later has content."""
    import json
    import os
    from app.services.session_cache import SessionCache

    cache = SessionCache(str(tmp_path))

    # 1. File absent → cache stores {}
    assert cache.load("state.json") == {}

    # 2. File appears with real data
    path = os.path.join(str(tmp_path), "state.json")
    with open(path, "w") as f:
        json.dump({"classes": [{"id": 1}], "objects": []}, f)

    # 3. Next load must return the real data, not the cached {}
    result = cache.load("state.json")
    assert result == {"classes": [{"id": 1}], "objects": []}


def test_load_does_not_refresh_when_cached_has_data_and_disk_changes(tmp_path):
    """Cache must keep serving cached data once it has real content, even if
    disk changes underneath — that's the whole point of caching."""
    import json
    import os
    from app.services.session_cache import SessionCache

    cache = SessionCache(str(tmp_path))
    path = os.path.join(str(tmp_path), "state.json")

    with open(path, "w") as f:
        json.dump({"classes": [{"id": 1, "name": "a"}]}, f)

    first = cache.load("state.json")
    assert first == {"classes": [{"id": 1, "name": "a"}]}

    # Disk mutated by someone else (not through this cache)
    with open(path, "w") as f:
        json.dump({"classes": [{"id": 1, "name": "b"}]}, f)

    # Second load must still return the cached 'a' (caching contract)
    second = cache.load("state.json")
    assert second == {"classes": [{"id": 1, "name": "a"}]}
