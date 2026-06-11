"""Integration tests for segment routes — beacon flush + close behaviors.

The beacon path (`POST /api/segment/flush`) runs on `beforeunload` via
sendBeacon, which means the client cannot react to the response body or
status. The handler must therefore write a durable .unsynced marker on
failure so the next resume's DownloadSessionStep can recover.
"""

import os

import pytest


@pytest.fixture
def cloud_app(tmp_path, monkeypatch):
    """Flask test client in cloud-mode with a GCSSyncManager installed.

    SEGMENT_MODE env stays "" but the segment module's SEGMENT_MODE
    constant is patched to "cloud" so the route's cloud-branch logic runs.
    """
    monkeypatch.setenv("SEGMENT_MODE", "")
    from app.config import set_sync_manager, set_session_cache
    from app.services.gcs_sync import GCSSyncManager

    sid = "beacon-sess"
    sdir = os.path.join(str(tmp_path), sid)
    os.makedirs(sdir)
    (tmp_path / sid / "masks.json").write_text("{}")

    import app.routes.segment as segment_mod
    monkeypatch.setattr(segment_mod, "SESSIONS_DIR", str(tmp_path))
    monkeypatch.setattr(segment_mod, "SEGMENT_MODE", "cloud")

    # Reset any leftover sync manager from a prior test
    set_sync_manager(None)
    sm = GCSSyncManager("test-bucket", sid, sdir)
    set_sync_manager(sm)

    from app import create_app
    app = create_app()
    app.config["TESTING"] = True

    with app.test_client() as tc:
        yield tc, sm, sdir

    set_sync_manager(None)
    set_session_cache(None)


def test_beacon_flush_promotes_deferred(cloud_app, monkeypatch):
    """A beacon fired mid-propagation must upload deferred masks.json."""
    tc, sm, sdir = cloud_app

    # Simulate: propagation is active, masks.json is currently deferred.
    sm.set_propagating(True)
    sm.mark_dirty("masks.json")
    assert "masks.json" in sm._deferred
    assert "masks.json" not in sm._dirty

    uploaded: list[str] = []
    monkeypatch.setattr(
        "app.services.gcs_sync._get_upload_file",
        lambda: (lambda bkt, s, rel, p: uploaded.append(rel)),
    )

    res = tc.post("/api/segment/flush")
    assert res.status_code == 200
    body = res.get_json()
    assert body["ok"] is True
    assert "masks.json" in uploaded


def test_beacon_flush_writes_marker_on_failure(cloud_app, monkeypatch):
    """Beacon must write .unsynced marker when upload fails so the next
    resume's DownloadSessionStep can recover."""
    tc, sm, sdir = cloud_app

    sm.mark_dirty("masks.json")

    def always_fail(bkt, s, rel, p):
        raise Exception("GCS down")

    monkeypatch.setattr("app.services.gcs_sync._get_upload_file",
                        lambda: always_fail)

    res = tc.post("/api/segment/flush")
    # Beacons cannot react — must always get 200
    assert res.status_code == 200
    body = res.get_json()
    assert body["ok"] is False
    assert "masks.json" in body["failed"]

    from app.services.unsynced_marker import read_marker
    marker = read_marker(sdir)
    assert marker is not None and "masks.json" in marker


def test_beacon_flush_returns_200_on_failure(cloud_app, monkeypatch):
    """Regardless of failure mode, the beacon endpoint returns HTTP 200.
    Non-200 status from a beacon is silently discarded by the browser."""
    tc, sm, _ = cloud_app
    sm.mark_dirty("masks.json")

    monkeypatch.setattr(
        "app.services.gcs_sync._get_upload_file",
        lambda: (_ for _ in ()).throw(RuntimeError("hard fail")),
    )

    res = tc.post("/api/segment/flush")
    assert res.status_code == 200


def test_beacon_flush_noop_when_no_sync_manager(tmp_path, monkeypatch):
    """If nothing is active (idle), the beacon is a silent no-op 200."""
    monkeypatch.setenv("SEGMENT_MODE", "")
    from app.config import set_sync_manager
    set_sync_manager(None)

    import app.routes.segment as segment_mod
    monkeypatch.setattr(segment_mod, "SESSIONS_DIR", str(tmp_path))
    monkeypatch.setattr(segment_mod, "SEGMENT_MODE", "cloud")

    from app import create_app
    app = create_app()
    app.config["TESTING"] = True

    with app.test_client() as tc:
        res = tc.post("/api/segment/flush")

    assert res.status_code == 200
    body = res.get_json()
    assert body["ok"] is True
    assert body["uploaded"] == 0
    assert body["failed"] == []


def test_text_detect_resets_tracker_before_bridging_masks(tmp_path, monkeypatch):
    """Text detection after a propagation must reset the tracker state before
    registering detected instances — the native SAM3 predictor rejects new
    object ids once tracking has started (regression: "Cannot add new object
    id 14 after tracking starts. All existing object ids: [13]").
    """
    import numpy as np
    from pycocotools import mask as pmask_utils

    monkeypatch.setenv("SEGMENT_MODE", "")
    from app.config import set_session_cache
    set_session_cache(None)

    sid = "text-sess"
    sdir = os.path.join(str(tmp_path), sid)
    os.makedirs(sdir)
    (tmp_path / sid / "masks.json").write_text("{}")
    (tmp_path / sid / "prompts.json").write_text("{}")

    import app.routes.segment as segment_mod
    monkeypatch.setattr(segment_mod, "SESSIONS_DIR", str(tmp_path))

    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:5, 2:5] = 1
    rle = pmask_utils.encode(np.asfortranarray(mask))
    rle = {"size": rle["size"], "counts": rle["counts"].decode("utf-8")}

    calls = []

    class FakeSam:
        # Simulates the native predictor after a propagation run.
        tracking_started = True

        def add_text_prompt(self, session_id, frame_idx, text):
            return {
                "frame_idx": frame_idx,
                "text": text,
                "instances": [
                    {"obj_id": 0, "rle": rle, "confidence": 0.9, "area": 9},
                ],
            }

        def reset_and_replay_objects(self, session_id, session_dir, object_ids):
            calls.append("reset")
            self.tracking_started = False

        def add_mask(self, session_id, frame_idx, obj_id, binary_mask):
            calls.append(f"add_mask:{obj_id}")
            if self.tracking_started:
                raise RuntimeError(
                    f"Cannot add new object id {obj_id} after tracking starts. "
                    "All existing object ids: [13]."
                )
            return {"frame_idx": frame_idx, "masks": {}}

    monkeypatch.setattr(segment_mod, "sam", FakeSam())

    from app import create_app
    app = create_app()
    app.config["TESTING"] = True

    with app.test_client() as tc:
        res = tc.post("/api/segment/text", json={
            "session_id": sid,
            "frame_idx": 4,
            "text": "buttons",
            "obj_id_start": 14,
        })

    assert res.status_code == 200, res.get_json()
    assert calls and calls[0] == "reset", f"tracker not reset before add_mask: {calls}"
    assert "add_mask:14" in calls
