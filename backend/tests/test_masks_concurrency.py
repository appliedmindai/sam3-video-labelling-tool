"""Concurrent-writer stress test for masks.json + prompts.json RMW paths.

Before fix: two threads racing mask writes on the same SessionCache could
produce `RuntimeError: dictionary changed size during iteration` or lost
updates. This test reproduces the race and asserts neither happens after
the session_io_lock is in place.
"""
import json
import os
import threading
from pathlib import Path

import numpy as np
import pytest


def _make_mask(h=50, w=50):
    m = np.zeros((h, w), dtype=np.uint8)
    m[10:40, 10:40] = 1
    return m


def test_concurrent_mask_writers_do_not_corrupt(tmp_path, monkeypatch):
    """32 threads x 20 iterations of update_frame_masks against the same
    session/cache must not crash or lose writes."""
    from app.services.mask_storage import update_frame_masks, load_masks
    from app.services.session_cache import SessionCache
    from app.services.session_lock import session_io_lock

    session_id = "concurrent-session"
    session_dir = str(tmp_path / session_id)
    os.makedirs(session_dir, exist_ok=True)
    cache = SessionCache(session_dir)

    num_threads = 32
    iterations = 20
    errors: list[Exception] = []

    def worker(tid: int):
        try:
            for i in range(iterations):
                frame_idx = tid * 100 + i  # unique frame per (tid, i)
                with session_io_lock(session_id):
                    update_frame_masks(
                        session_dir, frame_idx, {tid: _make_mask()},
                        source_keyframe=None, cache=cache,
                    )
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(num_threads)]
    for t in threads: t.start()
    for t in threads: t.join()

    # No exceptions
    assert errors == [], f"Writers raised: {errors!r}"

    # Every (tid, i) write must be present — no lost updates
    loaded = load_masks(session_dir, cache=cache)
    expected_frames = {tid * 100 + i for tid in range(num_threads) for i in range(iterations)}
    assert set(loaded.keys()) == expected_frames

    # On-disk JSON must be valid
    with open(os.path.join(session_dir, "masks.json")) as f:
        json.load(f)  # raises if invalid


def test_concurrent_prompt_writers_do_not_corrupt(tmp_path, monkeypatch):
    """Same guarantee for prompts.json via save_prompt."""
    from app.services.prompt_storage import save_prompt, load_all_prompts
    from app.services.session_cache import SessionCache
    from app.services.session_lock import session_io_lock

    session_id = "concurrent-session-prompts"
    session_dir = str(tmp_path / session_id)
    os.makedirs(session_dir, exist_ok=True)
    cache = SessionCache(session_dir)

    num_threads = 32
    iterations = 20
    errors: list[Exception] = []

    def worker(tid: int):
        try:
            for i in range(iterations):
                frame_idx = tid * 100 + i
                with session_io_lock(session_id):
                    save_prompt(session_dir, frame_idx, tid, {
                        "type": "click",
                        "points": [[float(tid), float(i)]],
                        "labels": [1],
                    }, cache=cache)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(tid,)) for tid in range(num_threads)]
    for t in threads: t.start()
    for t in threads: t.join()

    assert errors == [], f"Writers raised: {errors!r}"

    loaded = load_all_prompts(session_dir, cache=cache)
    expected_count = num_threads * iterations
    actual_count = sum(len(v) for v in loaded.values())
    assert actual_count == expected_count, f"Lost updates: got {actual_count}, expected {expected_count}"

    with open(os.path.join(session_dir, "prompts.json")) as f:
        json.load(f)
