"""Reader-vs-writer stress test.

Before the SessionCache shallow-copy fix, concurrent GET endpoints (readers
iterating the cached dict) would race with POST endpoints (writers mutating
the same dict reference) producing 'dictionary changed size during iteration'.

This test exercises the hot path: many readers calling load_all_masks_rle
concurrently with many writers calling update_frame_masks through the
session_io_lock + SessionCache pipeline. No exceptions may occur; on-disk
JSON must be valid; reader snapshots must be internally consistent."""

import json
import os
import threading

import numpy as np


def _make_mask(h=32, w=32):
    m = np.zeros((h, w), dtype=np.uint8)
    m[5:25, 5:25] = 1
    return m


def test_concurrent_readers_and_writers_masks(tmp_path):
    from app.services.mask_storage import update_frame_masks, load_all_masks_rle
    from app.services.session_cache import SessionCache
    from app.services.session_lock import session_io_lock

    session_id = "rw-masks"
    session_dir = str(tmp_path / session_id)
    os.makedirs(session_dir, exist_ok=True)
    cache = SessionCache(session_dir)

    # Seed with some initial data so readers have something to iterate
    with session_io_lock(session_id):
        for fi in range(10):
            update_frame_masks(session_dir, fi, {1: _make_mask()}, source_keyframe=None, cache=cache)

    stop = threading.Event()
    reader_errors: list[Exception] = []
    writer_errors: list[Exception] = []

    def reader(tid: int):
        try:
            while not stop.is_set():
                # No lock on readers -- they rely on cache.load giving a safe snapshot.
                result = load_all_masks_rle(session_dir, cache=cache)
                # Internal consistency: every frame key must map to a dict.
                for k, v in result.items():
                    assert isinstance(v, dict), f"reader {tid}: frame {k} has {type(v).__name__}"
        except Exception as e:
            reader_errors.append(e)

    def writer(tid: int):
        try:
            for i in range(50):
                if stop.is_set():
                    return
                with session_io_lock(session_id):
                    update_frame_masks(session_dir, 100 + tid * 50 + i, {tid: _make_mask()},
                                       source_keyframe=None, cache=cache)
        except Exception as e:
            writer_errors.append(e)

    readers = [threading.Thread(target=reader, args=(t,)) for t in range(16)]
    writers = [threading.Thread(target=writer, args=(t,)) for t in range(8)]
    for t in readers: t.start()
    for t in writers: t.start()
    for t in writers: t.join()  # wait for writers to finish
    stop.set()
    for t in readers: t.join()

    assert writer_errors == [], f"writers raised: {writer_errors!r}"
    assert reader_errors == [], f"readers raised: {reader_errors!r}"

    # Final disk state valid
    with open(os.path.join(session_dir, "masks.json")) as f:
        disk = json.load(f)
    # Every write landed (no lost updates from writers)
    for tid in range(8):
        for i in range(50):
            frame_key = str(100 + tid * 50 + i)
            assert frame_key in disk, f"missing frame {frame_key}"


def test_concurrent_readers_and_writers_prompts(tmp_path):
    from app.services.prompt_storage import save_prompt, load_all_prompts
    from app.services.session_cache import SessionCache
    from app.services.session_lock import session_io_lock

    session_id = "rw-prompts"
    session_dir = str(tmp_path / session_id)
    os.makedirs(session_dir, exist_ok=True)
    cache = SessionCache(session_dir)

    with session_io_lock(session_id):
        for fi in range(10):
            save_prompt(session_dir, fi, 1, {"type": "click", "points": [[1.0, 1.0]], "labels": [1]},
                        cache=cache)

    stop = threading.Event()
    reader_errors: list[Exception] = []
    writer_errors: list[Exception] = []

    def reader(tid: int):
        try:
            while not stop.is_set():
                result = load_all_prompts(session_dir, cache=cache)
                for k, v in result.items():
                    assert isinstance(v, dict)
        except Exception as e:
            reader_errors.append(e)

    def writer(tid: int):
        try:
            for i in range(50):
                if stop.is_set():
                    return
                with session_io_lock(session_id):
                    save_prompt(session_dir, 100 + tid * 50 + i, tid,
                                {"type": "click", "points": [[float(tid), float(i)]], "labels": [1]},
                                cache=cache)
        except Exception as e:
            writer_errors.append(e)

    readers = [threading.Thread(target=reader, args=(t,)) for t in range(16)]
    writers = [threading.Thread(target=writer, args=(t,)) for t in range(8)]
    for t in readers: t.start()
    for t in writers: t.start()
    for t in writers: t.join()
    stop.set()
    for t in readers: t.join()

    assert writer_errors == [], f"writers raised: {writer_errors!r}"
    assert reader_errors == [], f"readers raised: {reader_errors!r}"

    with open(os.path.join(session_dir, "prompts.json")) as f:
        disk = json.load(f)
    for tid in range(8):
        for i in range(50):
            frame_key = str(100 + tid * 50 + i)
            assert frame_key in disk, f"missing frame {frame_key}"
