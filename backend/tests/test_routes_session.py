"""Integration tests for session routes — focus on cache lifecycle and
wipe-prevention guarantees."""

import json
import os
from unittest.mock import patch

import pytest


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Flask test client with sessions dir pointed at tmp_path and
    cloud-mode auth stubbed to always succeed."""
    monkeypatch.setenv("SEGMENT_MODE", "")  # standalone mode for simpler tests
    from app import create_app
    import app.routes.session as session_mod
    import app.routes.segment as segment_mod
    from app.config import set_session_cache
    # Reset the module-global session cache so prior tests can't leak a cache
    # pointing at a stale tmp_path into this test's load_state calls.
    set_session_cache(None)
    monkeypatch.setattr(session_mod, "SESSIONS_DIR", str(tmp_path))
    monkeypatch.setattr(segment_mod, "SESSIONS_DIR", str(tmp_path))
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c, str(tmp_path)
    set_session_cache(None)


def test_resume_session_clears_stale_cache_before_pipeline_runs(client):
    """B5 (#79): switching from session A to B must clear A's cache
    BEFORE the pipeline starts so mid-resume writes fail-closed instead
    of landing in B's session_dir. The new cache for B is installed by
    on_complete on the pipeline thread — if start_pipeline is mocked
    and on_complete never fires, get_session_cache() remains None
    during the pipeline window."""
    from app.config import set_session_cache, get_session_cache
    from app.services.session_cache import SessionCache

    tc, sessions_dir = client
    session_a = "aaaa"
    session_b = "bbbb"
    os.makedirs(os.path.join(sessions_dir, session_a), exist_ok=True)
    os.makedirs(os.path.join(sessions_dir, session_b), exist_ok=True)
    with open(os.path.join(sessions_dir, session_a, "state.json"), "w") as f:
        json.dump({"classes": [{"id": 1, "name": "from_a"}], "objects": []}, f)
    with open(os.path.join(sessions_dir, session_b, "state.json"), "w") as f:
        json.dump({"classes": [{"id": 2, "name": "from_b"}], "objects": []}, f)
    with open(os.path.join(sessions_dir, session_a, "meta.json"), "w") as f:
        json.dump({"original_name": "A.mov"}, f)
    with open(os.path.join(sessions_dir, session_b, "meta.json"), "w") as f:
        json.dump({"original_name": "B.mov"}, f)

    # Simulate user opened A first — cache points at A
    cache_a = SessionCache(os.path.join(sessions_dir, session_a))
    cache_a.load("state.json")
    set_session_cache(cache_a)

    # Mock start_pipeline so on_complete never fires. Cache should be
    # None during the pipeline window (fail-closed), NOT still cache_a.
    with patch("app.routes.session.sam.start_pipeline",
               return_value=(True, None)):
        res = tc.post(f"/api/session/resume/{session_b}")
    assert res.status_code == 202

    # cache_a is gone, cache_b hasn't been installed yet — None is the
    # correct state for a resume in progress.
    assert get_session_cache() is None, (
        "stale cache must be cleared before pipeline runs"
    )


def test_resume_session_installs_cache_via_on_complete(client):
    """When on_complete fires (pipeline completes), the cache for B is
    installed. Covered here by capturing the on_complete callback from
    the start_pipeline mock and invoking it."""
    from app.config import set_session_cache, get_session_cache

    tc, sessions_dir = client
    sid = "completing"
    os.makedirs(os.path.join(sessions_dir, sid), exist_ok=True)
    with open(os.path.join(sessions_dir, sid, "meta.json"), "w") as f:
        json.dump({"original_name": "c.mov"}, f)
    set_session_cache(None)

    captured: dict[str, object] = {}

    def capturing_start(session_id, video_name, steps, on_complete=None):
        captured["on_complete"] = on_complete
        return True, "started"

    with patch("app.routes.session.sam.start_pipeline",
               side_effect=capturing_start):
        res = tc.post(f"/api/session/resume/{sid}")

    assert res.status_code == 202
    # Before on_complete fires, cache is None
    assert get_session_cache() is None

    # Simulate pipeline completion
    assert captured["on_complete"] is not None
    captured["on_complete"]()

    active = get_session_cache()
    assert active is not None
    assert active.session_dir.endswith(sid)


def test_put_state_refuses_to_wipe_non_empty_state_with_orphan_only_payload(client):
    """If state.json on disk has classes with assignments, reject a PUT that
    would replace it with classes=[] and objects containing class_id<0 entries
    (the fingerprint of the wipe bug)."""
    tc, sessions_dir = client
    sid = "wipe-guard"
    os.makedirs(os.path.join(sessions_dir, sid), exist_ok=True)
    # Disk has real data
    with open(os.path.join(sessions_dir, sid, "state.json"), "w") as f:
        json.dump({
            "classes": [{"id": 1, "name": "a", "color": "#000"}],
            "objects": [{"obj_id": 1, "class_id": 1}, {"obj_id": 2, "class_id": 1}],
        }, f)

    # Attempt to wipe: empty classes + orphan-only objects
    bad_payload = {
        "classes": [],
        "objects": [
            {"obj_id": 1, "class_id": -1},
            {"obj_id": 2, "class_id": -1},
        ],
    }
    res = tc.put(f"/api/session/state/{sid}", json=bad_payload)
    assert res.status_code == 409, f"expected 409 rejection, got {res.status_code}"

    # Disk is unchanged
    with open(os.path.join(sessions_dir, sid, "state.json")) as f:
        on_disk = json.load(f)
    assert on_disk["classes"] == [{"id": 1, "name": "a", "color": "#000"}]
    assert all(o["class_id"] == 1 for o in on_disk["objects"])


def test_put_state_allows_empty_state_when_disk_is_also_empty(client):
    """A new session whose state.json has {classes:[], objects:[]} must still
    accept an empty PUT — the guard only fires when we'd be LOSING data."""
    tc, sessions_dir = client
    sid = "fresh"
    os.makedirs(os.path.join(sessions_dir, sid), exist_ok=True)
    with open(os.path.join(sessions_dir, sid, "state.json"), "w") as f:
        json.dump({"classes": [], "objects": []}, f)

    res = tc.put(f"/api/session/state/{sid}",
                 json={"classes": [], "objects": [], "bbox_padding": {}})
    assert res.status_code == 200


def test_put_state_allows_user_deleting_all_classes(client):
    """User intentionally removing all classes must work — guard only blocks
    the specific wipe fingerprint (empty classes + orphan-only objects)."""
    tc, sessions_dir = client
    sid = "intentional-clear"
    os.makedirs(os.path.join(sessions_dir, sid), exist_ok=True)
    with open(os.path.join(sessions_dir, sid, "state.json"), "w") as f:
        json.dump({
            "classes": [{"id": 1, "name": "a", "color": "#000"}],
            "objects": [{"obj_id": 1, "class_id": 1}],
        }, f)

    # User deletes both class and object — empty but consistent
    res = tc.put(f"/api/session/state/{sid}",
                 json={"classes": [], "objects": []})
    assert res.status_code == 200


def test_put_state_stale_version_returns_409(client):
    """Two tabs editing the same session: the second PUT with a stale version
    must be rejected with 409 and a payload containing the current state so
    the client can reconcile."""
    tc, sessions_dir = client
    sid = "lost-update"
    os.makedirs(os.path.join(sessions_dir, sid), exist_ok=True)
    with open(os.path.join(sessions_dir, sid, "state.json"), "w") as f:
        json.dump({
            "classes": [{"id": 1, "name": "a", "color": "#000"}],
            "objects": [],
            "version": 5,
        }, f)

    # Tab A writes with the fresh version, bumps to 6.
    res_a = tc.put(
        f"/api/session/state/{sid}",
        json={"classes": [{"id": 1, "name": "a", "color": "#000"},
                          {"id": 2, "name": "new", "color": "#fff"}],
              "objects": [],
              "version": 5},
    )
    assert res_a.status_code == 200
    assert res_a.get_json()["version"] == 6

    # Tab B submits with the stale version -> 409 with current state.
    res_b = tc.put(
        f"/api/session/state/{sid}",
        json={"classes": [{"id": 1, "name": "renamed", "color": "#000"}],
              "objects": [],
              "version": 5},
    )
    assert res_b.status_code == 409
    body = res_b.get_json()
    assert body["error"] == "state_version_conflict"
    assert body["current_version"] == 6
    # Tab A's class addition is preserved on disk.
    assert len(body["state"]["classes"]) == 2


def test_put_state_matching_version_succeeds_and_bumps(client):
    """A PUT whose version matches disk succeeds and the response returns
    the new version."""
    tc, sessions_dir = client
    sid = "happy-version"
    os.makedirs(os.path.join(sessions_dir, sid), exist_ok=True)
    with open(os.path.join(sessions_dir, sid, "state.json"), "w") as f:
        json.dump({"classes": [], "objects": [], "version": 3}, f)

    res = tc.put(
        f"/api/session/state/{sid}",
        json={"classes": [{"id": 1, "name": "a", "color": "#000"}],
              "objects": [],
              "version": 3},
    )
    assert res.status_code == 200
    assert res.get_json()["version"] == 4

    with open(os.path.join(sessions_dir, sid, "state.json")) as f:
        on_disk = json.load(f)
    assert on_disk["version"] == 4


def test_put_state_without_version_is_backward_compatible(client):
    """Legacy callers that omit `version` bypass the version check (force
    overwrite) — required for external tools, the wipe-guard path, and any
    call chain that didn't load version first. The write still bumps the
    disk version from whatever it was."""
    tc, sessions_dir = client
    sid = "legacy-put"
    os.makedirs(os.path.join(sessions_dir, sid), exist_ok=True)
    with open(os.path.join(sessions_dir, sid, "state.json"), "w") as f:
        json.dump({"classes": [], "objects": [], "version": 7}, f)

    res = tc.put(
        f"/api/session/state/{sid}",
        json={"classes": [{"id": 1, "name": "a", "color": "#000"}],
              "objects": []},
    )
    assert res.status_code == 200
    # Version bumps relative to disk (7 -> 8), NOT relative to 0.
    assert res.get_json()["version"] == 8

    with open(os.path.join(sessions_dir, sid, "state.json")) as f:
        on_disk = json.load(f)
    assert on_disk["version"] == 8


def test_get_state_synthesizes_version_for_legacy_files(client):
    """state.json files written before the lost-update guard have no
    `version` key — GET must synthesize version: 0 so clients always have
    a starting anchor."""
    tc, sessions_dir = client
    sid = "legacy-get"
    os.makedirs(os.path.join(sessions_dir, sid), exist_ok=True)
    with open(os.path.join(sessions_dir, sid, "state.json"), "w") as f:
        json.dump({
            "classes": [{"id": 1, "name": "a", "color": "#000"}],
            "objects": [],
        }, f)

    res = tc.get(f"/api/session/state/{sid}")
    assert res.status_code == 200
    body = res.get_json()
    assert body["version"] == 0
    assert len(body["classes"]) == 1


def test_resume_session_leaves_globals_none_when_pipeline_rejects(client):
    """B5 (#79): when start_pipeline refuses, the globals stay (None,
    None) — the previous cache was already cleared before the pipeline
    call, and no new cache is installed because on_complete never fires.

    The prior invariant ("keep cache_a alive on 409") was weaker: a
    stale cache_a could still serve reads even though the user had
    moved on. The new contract is stricter — resume is a commitment
    to tear down the previous session, whether or not the pipeline
    accepts. If the user retries, they get a fresh cache; if they
    hit another session, that session's resume installs its own."""
    from app.config import set_session_cache, get_session_cache
    from app.services.session_cache import SessionCache

    tc, sessions_dir = client

    session_a = "active-a"
    os.makedirs(os.path.join(sessions_dir, session_a), exist_ok=True)
    cache_a = SessionCache(os.path.join(sessions_dir, session_a))
    set_session_cache(cache_a)

    session_b = "rejected-b"
    os.makedirs(os.path.join(sessions_dir, session_b), exist_ok=True)
    with open(os.path.join(sessions_dir, session_b, "meta.json"), "w") as f:
        json.dump({"original_name": "B.mov"}, f)

    with patch("app.routes.session.sam.start_pipeline",
               return_value=(False, "another session is active")):
        res = tc.post(f"/api/session/resume/{session_b}")
    assert res.status_code == 409

    # New contract: cache cleared (no longer cache_a), not installed as B.
    active = get_session_cache()
    assert active is None, (
        f"expected None after pipeline rejection, got {active!r}"
    )


# ------------------------------------------------------------------
# close_session: GCS flush retry + 503 behavior (Issue #55)
# ------------------------------------------------------------------

def test_close_session_returns_503_when_gcs_upload_fails(tmp_path, monkeypatch):
    """If GCS is unreachable during close, route returns 503 with the
    file list, KEEPS the sync manager alive, and does NOT clear the cache."""
    # Auth decorator only enforces when env var is 'cloud' — keep it "" so
    # tests bypass auth, but set the module-level SEGMENT_MODE to 'cloud'
    # so the route's cloud-branch logic runs.
    monkeypatch.setenv("SEGMENT_MODE", "")
    from app.config import set_sync_manager, get_sync_manager, set_session_cache, get_session_cache
    from app.services.session_cache import SessionCache
    from app.services.gcs_sync import GCSSyncManager

    sessions_dir = str(tmp_path)
    sid = "close-fail"
    sdir = os.path.join(sessions_dir, sid)
    os.makedirs(sdir)
    (tmp_path / sid / "state.json").write_text('{"classes": []}')

    import app.routes.segment as segment_mod
    monkeypatch.setattr(segment_mod, "SESSIONS_DIR", sessions_dir)
    monkeypatch.setattr(segment_mod, "SEGMENT_MODE", "cloud")

    # Reset module-global sync manager before installing our own (avoid
    # set_sync_manager(new) triggering stop() on a leftover manager).
    set_sync_manager(None)
    sm = GCSSyncManager("test-bucket", sid, sdir)
    sm.mark_dirty("state.json")
    set_sync_manager(sm)
    set_session_cache(SessionCache(sdir))

    # Stub upload to always fail
    def always_fails(*a, **kw):
        raise Exception("GCS down")
    monkeypatch.setattr("app.services.gcs_sync._get_upload_file",
                        lambda: always_fails)

    # Stub sam.close_session so it doesn't try to touch real SAM3 state
    monkeypatch.setattr("app.routes.segment.sam.close_session", lambda sid: None)

    from app import create_app
    app = create_app()
    app.config["TESTING"] = True
    try:
        with app.test_client() as tc:
            res = tc.post(f"/api/segment/close/{sid}")

        assert res.status_code == 503
        body = res.get_json()
        assert body["unsynced_files"] == ["state.json"]
        assert body["retry_possible"] is True

        # Sync manager MUST still be active for retry
        assert get_sync_manager() is sm
        # Session cache also preserved (frontend can retry)
        assert get_session_cache() is not None
    finally:
        # cleanup: avoid leaking state into other tests (set_sync_manager(None)
        # would call stop()->flush() which hits our failing stub again, but it's
        # safe — stop() just logs and returns a FlushResult).
        set_sync_manager(None)
        set_session_cache(None)


def test_close_session_returns_200_on_successful_flush(tmp_path, monkeypatch):
    """Happy path: flush succeeds, sync manager is stopped and discarded,
    route returns 200."""
    monkeypatch.setenv("SEGMENT_MODE", "")
    from app.config import set_sync_manager, get_sync_manager, set_session_cache, get_session_cache
    from app.services.session_cache import SessionCache
    from app.services.gcs_sync import GCSSyncManager

    sessions_dir = str(tmp_path)
    sid = "close-ok"
    sdir = os.path.join(sessions_dir, sid)
    os.makedirs(sdir)
    (tmp_path / sid / "state.json").write_text('{"classes": []}')

    import app.routes.segment as segment_mod
    monkeypatch.setattr(segment_mod, "SESSIONS_DIR", sessions_dir)
    monkeypatch.setattr(segment_mod, "SEGMENT_MODE", "cloud")

    set_sync_manager(None)
    sm = GCSSyncManager("test-bucket", sid, sdir)
    sm.mark_dirty("state.json")
    set_sync_manager(sm)
    set_session_cache(SessionCache(sdir))

    monkeypatch.setattr("app.services.gcs_sync._get_upload_file",
                        lambda: (lambda *a, **kw: None))
    monkeypatch.setattr("app.routes.segment.sam.close_session", lambda sid: None)

    from app import create_app
    app = create_app()
    app.config["TESTING"] = True
    try:
        with app.test_client() as tc:
            res = tc.post(f"/api/segment/close/{sid}")

        assert res.status_code == 200
        assert get_sync_manager() is None
        assert get_session_cache() is None
    finally:
        set_sync_manager(None)
        set_session_cache(None)


def test_close_session_surfaces_stop_failure_as_503(tmp_path, monkeypatch):
    """Regression (#58): a mark_dirty arriving between flush_with_retry and
    stop() must not disappear. If stop()'s internal flush fails, the route
    must return 503 + write an .unsynced marker, not silently drop the file.
    """
    monkeypatch.setenv("SEGMENT_MODE", "")
    from app.config import set_sync_manager, set_session_cache
    from app.services.session_cache import SessionCache
    from app.services.gcs_sync import GCSSyncManager

    sessions_dir = str(tmp_path)
    sid = "close-stop-fail"
    sdir = os.path.join(sessions_dir, sid)
    os.makedirs(sdir)
    (tmp_path / sid / "state.json").write_text('{}')

    import app.routes.segment as segment_mod
    monkeypatch.setattr(segment_mod, "SESSIONS_DIR", sessions_dir)
    monkeypatch.setattr(segment_mod, "SEGMENT_MODE", "cloud")

    set_sync_manager(None)
    sm = GCSSyncManager("test-bucket", sid, sdir)
    # Clean start — _dirty empty
    set_sync_manager(sm)
    set_session_cache(SessionCache(sdir))

    # Stub upload: succeed on everything EXCEPT a late file added after
    # flush_with_retry. Simulates the race: initial flush is clean, a request
    # thread marks state.json dirty after flush_with_retry returned, then
    # stop's internal flush tries it and GCS is temporarily down.
    call_count = {"n": 0}
    def upload(bucket, session, rel_path, local_path):
        # First call = flush_with_retry's initial attempt (no dirty files)
        # Subsequent calls (from stop's internal flush) fail
        call_count["n"] += 1
        if call_count["n"] > 0:
            raise Exception("GCS transient down")
    monkeypatch.setattr("app.services.gcs_sync._get_upload_file",
                        lambda: upload)

    # sam.close_session is stubbed; it's the place where a late write
    # could land. Simulate by calling mark_dirty right before the route's
    # stop() call. We patch close_session to mark dirty — this mirrors the
    # "concurrent request marks dirty after flush_with_retry" window.
    def close_and_mark(s):
        sm.mark_dirty("state.json")
    monkeypatch.setattr("app.routes.segment.sam.close_session", close_and_mark)

    from app import create_app
    app = create_app()
    app.config["TESTING"] = True
    try:
        with app.test_client() as tc:
            res = tc.post(f"/api/segment/close/{sid}")

        assert res.status_code == 503
        body = res.get_json()
        assert "state.json" in body["unsynced_files"]

        # Must have written an unsynced marker so next resume recovers
        from app.services.unsynced_marker import read_marker
        marker = read_marker(sdir)
        assert marker is not None and "state.json" in marker
    finally:
        set_sync_manager(None)
        set_session_cache(None)


def test_close_session_promotes_deferred_masks_on_close(tmp_path, monkeypatch):
    """Regression (#57): close_session must promote deferred masks.json
    before flushing. Before the fix, if propagation was cancelled by
    close_session but the propagation thread hadn't run its finally yet,
    masks.json would stay in _deferred and be dropped."""
    monkeypatch.setenv("SEGMENT_MODE", "")
    from app.config import set_sync_manager, set_session_cache
    from app.services.session_cache import SessionCache
    from app.services.gcs_sync import GCSSyncManager

    sessions_dir = str(tmp_path)
    sid = "close-deferred"
    sdir = os.path.join(sessions_dir, sid)
    os.makedirs(sdir)
    (tmp_path / sid / "masks.json").write_text('{}')

    import app.routes.segment as segment_mod
    monkeypatch.setattr(segment_mod, "SESSIONS_DIR", sessions_dir)
    monkeypatch.setattr(segment_mod, "SEGMENT_MODE", "cloud")

    uploaded: list[str] = []
    monkeypatch.setattr("app.services.gcs_sync._get_upload_file",
                        lambda: (lambda bkt, s, rel, p: uploaded.append(rel)))
    monkeypatch.setattr("app.routes.segment.sam.close_session", lambda s: None)

    set_sync_manager(None)
    sm = GCSSyncManager("test-bucket", sid, sdir)
    # Simulate an in-flight propagation scenario: propagating flag is set
    # and masks.json is stuck in _deferred.
    sm.set_propagating(True)
    sm.mark_dirty("masks.json")
    assert "masks.json" in sm._deferred
    assert "masks.json" not in sm._dirty

    set_sync_manager(sm)
    set_session_cache(SessionCache(sdir))

    from app import create_app
    app = create_app()
    app.config["TESTING"] = True
    try:
        with app.test_client() as tc:
            res = tc.post(f"/api/segment/close/{sid}")
        assert res.status_code == 200
        assert "masks.json" in uploaded
    finally:
        set_sync_manager(None)
        set_session_cache(None)


def test_set_sync_manager_writes_marker_on_old_stop_failure(tmp_path, monkeypatch):
    """Regression (#58): if the old manager's stop() has persistent failures,
    an .unsynced marker must be written to its session_dir so the next resume
    can recover."""
    from app.config import set_sync_manager
    from app.services.gcs_sync import GCSSyncManager
    from app.services.unsynced_marker import read_marker

    old_dir = tmp_path / "old_sess"
    new_dir = tmp_path / "new_sess"
    old_dir.mkdir()
    new_dir.mkdir()
    (old_dir / "state.json").write_text("{}")
    (new_dir / "state.json").write_text("{}")

    # Upload fails only for the old manager's files
    def failing_upload(bkt, sid, rel, path):
        if "old_sess" in path:
            raise Exception("transient GCS glitch")

    monkeypatch.setattr("app.services.gcs_sync._get_upload_file",
                        lambda: failing_upload)

    old_sm = GCSSyncManager("bkt", "old_sess", str(old_dir))
    old_sm.mark_dirty("state.json")
    set_sync_manager(old_sm)

    new_sm = GCSSyncManager("bkt", "new_sess", str(new_dir))
    try:
        # Swap to new manager — old one's stop() will fail. We must write
        # a marker so the files aren't lost on container recycle.
        set_sync_manager(new_sm)

        marker = read_marker(str(old_dir))
        assert marker is not None, "Expected .unsynced marker on old session dir"
        assert "state.json" in marker
    finally:
        set_sync_manager(None)


def test_resume_cloud_installs_sync_manager_via_on_complete(tmp_path, monkeypatch):
    """B5 (#79): in cloud mode, on_complete creates and installs a
    GCSSyncManager paired with the SessionCache. Until on_complete
    fires, get_sync_manager() returns None — no orphan request can
    route writes to the new session_dir before the pipeline is done."""
    monkeypatch.setenv("SEGMENT_MODE", "cloud")
    from app.config import (
        set_sync_manager, set_session_cache,
        get_sync_manager, get_session_cache,
    )

    sid = "cloud-resume"
    sdir = os.path.join(str(tmp_path), sid)
    os.makedirs(sdir)
    with open(os.path.join(sdir, "meta.json"), "w") as f:
        json.dump({"original_name": "r.mov"}, f)

    import app.routes.session as session_mod
    monkeypatch.setattr(session_mod, "SESSIONS_DIR", str(tmp_path))
    monkeypatch.setattr(session_mod, "SEGMENT_MODE", "cloud")

    set_sync_manager(None)
    set_session_cache(None)

    captured: dict[str, object] = {}

    def capturing_start(session_id, video_name, steps, on_complete=None):
        captured["on_complete"] = on_complete
        return True, "started"

    # No real GCS needed — GCSSyncManager's start() schedules a Timer
    # that we need to cancel. Patch its upload function so any fire is
    # a no-op.
    monkeypatch.setattr(
        "app.services.gcs_sync._get_upload_file",
        lambda: (lambda *a, **kw: None),
    )

    # The before_request hook reads app.config.GCS_BUCKET in cloud mode —
    # patch it so the route sees g.bucket="test-bucket".
    monkeypatch.setattr("app.config.GCS_BUCKET", "test-bucket")

    from app import create_app
    app = create_app()
    app.config["TESTING"] = True

    try:
        with patch("app.routes.session.sam.start_pipeline",
                   side_effect=capturing_start):
            with app.test_client() as tc:
                res = tc.post(f"/api/session/resume/{sid}")

        assert res.status_code == 202
        # Pre-completion: both globals None
        assert get_sync_manager() is None
        assert get_session_cache() is None

        # Simulate pipeline completion on the pipeline thread
        captured["on_complete"]()

        sm = get_sync_manager()
        cache = get_session_cache()
        assert sm is not None, "sync manager must be installed after on_complete"
        assert cache is not None, "session cache must be installed after on_complete"
        assert sm.session_id == sid
        assert sm.session_dir == sdir
        assert cache.session_dir == sdir
    finally:
        # Cleanup: stop the sm's periodic timer
        sm = get_sync_manager()
        if sm is not None:
            try:
                sm.stop()
            except Exception:
                pass
        set_sync_manager(None)
        set_session_cache(None)


def test_set_sync_manager_is_thread_safe(tmp_path, monkeypatch):
    """Concurrent set_sync_manager + get_sync_manager must never return
    a half-swapped state."""
    from app.config import set_sync_manager, get_sync_manager
    from app.services.gcs_sync import GCSSyncManager
    import threading

    # Stub upload so sync manager stop()s don't attempt real GCS calls.
    monkeypatch.setattr("app.services.gcs_sync._get_upload_file",
                        lambda: (lambda *a, **kw: None))

    (tmp_path / "state.json").write_text("{}")
    managers = [
        GCSSyncManager("bkt", f"s{i}", str(tmp_path)) for i in range(4)
    ]
    errors: list[Exception] = []

    def swapper(mgr):
        try:
            for _ in range(20):
                set_sync_manager(mgr)
        except Exception as e:
            errors.append(e)

    def reader():
        try:
            for _ in range(200):
                sm = get_sync_manager()
                if sm is not None:
                    # Access an attribute to verify we got a coherent object
                    _ = sm.session_id
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=swapper, args=(m,)) for m in managers]
    threads += [threading.Thread(target=reader) for _ in range(4)]
    for t in threads: t.start()
    for t in threads: t.join()
    assert errors == [], f"globals race: {errors!r}"
    set_sync_manager(None)  # cleanup
