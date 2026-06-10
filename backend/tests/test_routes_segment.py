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
