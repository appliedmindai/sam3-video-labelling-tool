"""Regression tests for atomic_json_dump — specifically the concurrent-writer
path exercised by bulk class import where the PUT /session/state autosave
races the POST /session/classes handler on the same state.json.

Before the fix, concurrent writers shared a single `<path>.tmp` and would
either collide on rename (FileNotFoundError) or interleave bytes into the
final file (JSONDecodeError: Extra data)."""

import json
import os
import threading
import time

import pytest

from app.services.atomic_write import atomic_json_dump, sweep_orphan_tmp_files


def test_concurrent_writers_produce_valid_json(tmp_path):
    target = str(tmp_path / "state.json")
    writer_count = 32
    iterations = 20
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def writer(writer_id: int) -> None:
        try:
            for i in range(iterations):
                atomic_json_dump(
                    {"writer": writer_id, "iter": i, "classes": [{"id": writer_id, "name": f"w{writer_id}", "color": "#fff"}]},
                    target,
                    indent=2,
                )
        except BaseException as exc:  # noqa: BLE001 — we want to capture every failure
            with errors_lock:
                errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(writer_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"writers raised: {errors[:3]}"

    # Final file must be valid JSON (no partial/interleaved writes on disk).
    with open(target) as f:
        data = json.load(f)
    assert "writer" in data
    assert 0 <= data["writer"] < writer_count

    # No tmp files should remain — unique tmp paths get renamed or cleaned up.
    leftovers = [p for p in os.listdir(tmp_path) if p.startswith("state.json.tmp")]
    assert leftovers == [], f"leftover tmp files: {leftovers}"


def test_unique_tmp_paths_across_threads(tmp_path, monkeypatch):
    # Sanity: two threads racing on the same target should not share a tmp path.
    target = str(tmp_path / "out.json")
    seen_tmp_paths: list[str] = []
    seen_lock = threading.Lock()
    original_open = open

    def recording_open(path, *args, **kwargs):
        if isinstance(path, str) and path.startswith(target) and ".tmp." in path:
            with seen_lock:
                seen_tmp_paths.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", recording_open)

    def writer(i: int) -> None:
        atomic_json_dump({"i": i}, target)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(seen_tmp_paths) == 8
    assert len(set(seen_tmp_paths)) == 8, "tmp paths collided between threads"


def test_write_failure_cleans_up_tmp(tmp_path, monkeypatch):
    target = str(tmp_path / "state.json")

    class BoomEncoder(json.JSONEncoder):
        def encode(self, o):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        # Swap json.dump to raise mid-write.
        def boom_dump(*_args, **_kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr("app.services.atomic_write.json.dump", boom_dump)
        atomic_json_dump({"x": 1}, target)

    # Tmp file must not leak on the failure path.
    leftovers = [p for p in os.listdir(tmp_path) if p.startswith("state.json.tmp")]
    assert leftovers == [], f"leftover tmp files after failed write: {leftovers}"


# --- sweep_orphan_tmp_files (R41) ---

def test_sweep_unlinks_stale_orphans_and_preserves_fresh(tmp_path):
    """Simulate SIGKILL-style orphans: tmp files that never got renamed.

    The sweep must remove stale orphans (older than max_age_s) and leave
    both fresh orphans and canonical files alone.
    """
    session = tmp_path / "sess1"
    session.mkdir()

    # Stale orphans — the three patterns produced by atomic_json_dump,
    # gcs_storage.download_file_if_exists, and unsynced_marker.write_marker.
    stale = [
        session / "state.json.tmp.12345.6789.abcdef0123456789",  # atomic_json_dump
        session / "masks.json.tmp.42.deadbeefcafef00d",          # download_file_if_exists
        session / ".unsynced.json.tmp.99",                       # write_marker
    ]
    # Nested under a subdir — sweep should recurse.
    subdir = session / "frames"
    subdir.mkdir()
    stale.append(subdir / "thumb.json.tmp.1.2")

    for p in stale:
        p.write_text("{}")
        old = time.time() - 7200  # 2h ago (> 1h threshold)
        os.utime(p, (old, old))

    # Fresh orphan — must NOT be swept.
    fresh = session / "state.json.tmp.55555.1.fresh"
    fresh.write_text("{}")

    # Canonical files — must be preserved.
    canonical = session / "state.json"
    canonical.write_text('{"foo": 1}')
    video = session / "video.mp4"
    video.write_text("binary")

    removed = sweep_orphan_tmp_files(str(tmp_path))

    assert removed == len(stale)
    for p in stale:
        assert not p.exists(), f"stale orphan not swept: {p}"
    assert fresh.exists(), "fresh orphan was incorrectly swept"
    assert canonical.exists(), "canonical state.json was swept"
    assert video.exists(), "video.mp4 was swept"


def test_sweep_tolerates_missing_root(tmp_path):
    missing = tmp_path / "does-not-exist"
    assert sweep_orphan_tmp_files(str(missing)) == 0


def test_sweep_returns_zero_when_no_orphans(tmp_path):
    (tmp_path / "state.json").write_text("{}")
    assert sweep_orphan_tmp_files(str(tmp_path)) == 0


def test_sweep_never_raises_on_unlink_error(tmp_path, monkeypatch):
    """If unlink fails (e.g. read-only FS), the sweep logs and moves on
    rather than blocking startup/resume."""
    orphan = tmp_path / "state.json.tmp.1.2.3"
    orphan.write_text("{}")
    old = time.time() - 7200
    os.utime(orphan, (old, old))

    def boom_unlink(_path):
        raise OSError("permission denied")

    monkeypatch.setattr("app.services.atomic_write.os.unlink", boom_unlink)

    # Must not raise — returns 0 because nothing was successfully removed.
    assert sweep_orphan_tmp_files(str(tmp_path)) == 0
    # Orphan is still there — we tolerated the error.
    assert orphan.exists()
