"""Tests for ServiceState transitions and pipeline runner."""

import json
import os
import threading
import time

import pytest

from app.services.pipeline import ServiceState, PipelineStep


class FakeStep:
    """A fake pipeline step for testing."""

    def __init__(self, phase: str, duration: float = 0.0, result: dict | None = None,
                 error: Exception | None = None):
        self.phase = phase
        self._duration = duration
        self._result = result
        self._error = error
        self.ran = False

    def run(self, on_progress, cancel_event):
        self.ran = True
        if self._error:
            raise self._error
        steps = max(1, int(self._duration / 0.05))
        for i in range(steps):
            if cancel_event.is_set():
                return None
            on_progress((i + 1) / steps)
            time.sleep(0.05)
        return self._result


class TestServiceState:
    def test_default_is_idle(self):
        state = ServiceState()
        assert state.phase == "idle"
        assert state.session_id is None
        assert state.progress == 0.0

    def test_frozen_prevents_mutation(self):
        state = ServiceState()
        with pytest.raises(AttributeError):
            state.phase = "extracting"

    def test_replace_creates_new_instance(self):
        from dataclasses import replace
        state = ServiceState(phase="idle")
        new_state = replace(state, phase="extracting", session_id="abc")
        assert state.phase == "idle"
        assert new_state.phase == "extracting"
        assert new_state.session_id == "abc"


class TestStartPipeline:
    """Tests using the SAM3Service pipeline methods.

    These tests instantiate SAM3Service directly. The singleton is fine
    for testing since we control the process.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        """Reset SAM3Service state before each test."""
        from app.services.sam3_service import SAM3Service
        self.sam = SAM3Service()
        # Reset pipeline state to idle
        with self.sam._state_lock:
            self.sam._service_state = ServiceState()
        # Ensure cancel event is cleared
        self.sam._pipeline_cancel.clear()
        # Wait for any leftover pipeline thread from a prior test
        if self.sam._pipeline_thread is not None and self.sam._pipeline_thread.is_alive():
            self.sam._pipeline_cancel.set()
            self.sam._pipeline_thread.join(timeout=5)
        self.sam._pipeline_thread = None
        yield

    def test_start_pipeline_sets_phase(self):
        step = FakeStep(phase="extracting", duration=0.1)
        ok, reason = self.sam.start_pipeline("sid-1", "video.mp4", [step])
        assert ok is True
        assert reason == "started"
        # Give thread a moment to start
        time.sleep(0.05)
        state = self.sam.get_service_state()
        assert state.phase == "extracting"
        assert state.session_id == "sid-1"
        # Wait for completion
        self.sam._pipeline_thread.join(timeout=5)
        state = self.sam.get_service_state()
        assert state.phase == "ready"
        assert state.progress == 1.0

    def test_concurrent_pipeline_rejected(self):
        step = FakeStep(phase="extracting", duration=0.5)
        ok1, _ = self.sam.start_pipeline("sid-1", "v1.mp4", [step])
        assert ok1 is True
        ok2, reason = self.sam.start_pipeline("sid-2", "v2.mp4", [step])
        assert ok2 is False
        assert reason == "pipeline_already_running"
        self.sam._pipeline_cancel.set()
        self.sam._pipeline_thread.join(timeout=5)

    def test_cancel_stops_pipeline(self):
        step = FakeStep(phase="extracting", duration=2.0)
        self.sam.start_pipeline("sid-1", "video.mp4", [step])
        time.sleep(0.1)
        result = self.sam.cancel_pipeline()
        assert result == "cancel_requested"
        self.sam._pipeline_thread.join(timeout=5)
        state = self.sam.get_service_state()
        assert state.phase == "idle"

    def test_error_sets_error_phase(self):
        step = FakeStep(
            phase="extracting",
            error=RuntimeError("disk full"),
        )
        self.sam.start_pipeline("sid-1", "video.mp4", [step])
        self.sam._pipeline_thread.join(timeout=5)
        state = self.sam.get_service_state()
        assert state.phase == "error"
        assert "disk full" in state.error

    def test_multi_step_pipeline(self):
        step1 = FakeStep(phase="extracting", duration=0.1, result={"frame_count": 100})
        step2 = FakeStep(phase="initializing", duration=0.1)
        self.sam.start_pipeline("sid-1", "video.mp4", [step1, step2])
        self.sam._pipeline_thread.join(timeout=5)
        state = self.sam.get_service_state()
        assert state.phase == "ready"
        assert state.frame_count == 100

    def test_cancel_nothing_running(self):
        result = self.sam.cancel_pipeline()
        assert result == "nothing_to_cancel"

    def test_progress_updates(self):
        step = FakeStep(phase="extracting", duration=0.3)
        self.sam.start_pipeline("sid-1", "video.mp4", [step])
        time.sleep(0.15)
        state = self.sam.get_service_state()
        assert state.progress > 0.0
        assert state.progress < 1.0
        self.sam._pipeline_thread.join(timeout=5)


class TestExtractFramesAsync:
    """Test the async frame extraction with progress."""

    @pytest.fixture
    def tmp_session(self, tmp_path):
        """Create a minimal session directory with a short test video."""
        session_dir = tmp_path / "test-session"
        session_dir.mkdir()
        video_path = str(session_dir / "video.mp4")
        import subprocess
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i",
            "testsrc=duration=2:size=320x240:rate=10",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            video_path,
        ], capture_output=True, check=True)
        return session_dir

    def test_extracts_frames_with_progress(self, tmp_session):
        from app.services.video_processor import extract_frames_async

        frames_dir = str(tmp_session / "frames")
        progress_values = []

        count = extract_frames_async(
            str(tmp_session / "video.mp4"),
            frames_dir,
            fps=2,
            on_progress=lambda p: progress_values.append(p),
            poll_interval=0.1,
        )

        assert count > 0
        assert len(os.listdir(frames_dir)) == count
        assert len(progress_values) > 0
        assert progress_values[-1] == 1.0

    def test_cancel_stops_extraction(self, tmp_session):
        from app.services.video_processor import extract_frames_async

        frames_dir = str(tmp_session / "frames")
        cancel = threading.Event()
        cancel.set()  # Cancel immediately

        count = extract_frames_async(
            str(tmp_session / "video.mp4"),
            frames_dir,
            fps=2,
            cancel_event=cancel,
            poll_interval=0.1,
        )

        assert count == 0


class TestExtractFramesStep:
    """Test the ExtractFramesStep pipeline integration."""

    @pytest.fixture
    def session_dir(self, tmp_path, monkeypatch):
        """Create a session directory with a test video and patch SESSIONS_DIR."""
        sessions_root = tmp_path / "sessions"
        sessions_root.mkdir()
        session_id = "step-test-session"
        session_dir = sessions_root / session_id
        session_dir.mkdir()

        video_path = str(session_dir / "video.mp4")
        import subprocess
        subprocess.run([
            "ffmpeg", "-y", "-f", "lavfi", "-i",
            "testsrc=duration=2:size=320x240:rate=10",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            video_path,
        ], capture_output=True, check=True)

        monkeypatch.setattr("app.config.SESSIONS_DIR", str(sessions_root))
        return session_id, session_dir

    def test_step_extracts_and_writes_meta(self, session_dir):
        from app.services.pipeline import ExtractFramesStep

        session_id, session_path = session_dir
        step = ExtractFramesStep(session_id, fps=2)

        progress_values = []
        cancel = threading.Event()

        result = step.run(
            on_progress=lambda p: progress_values.append(p),
            cancel_event=cancel,
        )

        assert result is not None
        assert result["frame_count"] > 0

        # Frames were actually created
        frames_dir = session_path / "frames"
        assert frames_dir.exists()
        assert len(list(frames_dir.glob("*.jpg"))) == result["frame_count"]

        # meta.json was written
        meta_path = session_path / "meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert meta["fps"] == 2
        assert meta["frame_count"] == result["frame_count"]
        assert "video_info" in meta
        assert meta["video_info"]["width"] == 320
        assert meta["video_info"]["height"] == 240

    def test_step_preserves_existing_meta(self, session_dir):
        from app.services.pipeline import ExtractFramesStep

        session_id, session_path = session_dir

        # Write pre-existing meta with extra fields
        meta_path = session_path / "meta.json"
        meta_path.write_text(json.dumps({"md5": "abc123", "original_name": "my_video.mp4"}))

        step = ExtractFramesStep(session_id, fps=2)
        cancel = threading.Event()
        result = step.run(on_progress=lambda p: None, cancel_event=cancel)

        assert result is not None
        meta = json.loads(meta_path.read_text())
        assert meta["md5"] == "abc123"
        assert meta["original_name"] == "my_video.mp4"
        assert meta["frame_count"] == result["frame_count"]

    def test_step_cancel_returns_none(self, session_dir):
        from app.services.pipeline import ExtractFramesStep

        session_id, _ = session_dir
        step = ExtractFramesStep(session_id, fps=2)

        cancel = threading.Event()
        cancel.set()  # Cancel immediately

        result = step.run(on_progress=lambda p: None, cancel_event=cancel)
        assert result is None


class TestDownloadSessionStepStagingRecovery:
    """B2: staged `.unsynced/<rel>` blobs must be promoted to canonical
    on cold resume (scale-to-zero path). Works WITH or WITHOUT a marker
    (orphan sweep)."""

    def _fake_sm(self, monkeypatch):
        import app.services.gcs_sync as gcs_sync_mod

        class FakeSM:
            def __init__(self, *a, **kw): pass
            def start(self, *a, **kw): pass
            def stop(self): pass
            def mark_dirty(self, path): pass
            def promote_deferred(self): pass
            def dirty_snapshot(self): return frozenset()

        monkeypatch.setattr(gcs_sync_mod, "GCSSyncManager", FakeSM)

    def test_promotes_staged_blobs_to_canonical_before_download(
        self, tmp_path, monkeypatch,
    ):
        """On cold start, _promote_staged_blobs runs BEFORE download_session.
        Staged blobs referenced by the remote marker are copied to
        canonical so the fresh download picks up the authoritative bytes."""
        from app.services.pipeline import DownloadSessionStep
        from app.services.unsynced_marker import MARKER_FILENAME

        session_id = "stage-recover"
        session_dir = tmp_path / session_id
        session_dir.mkdir()  # empty — cold start

        import app.config as config_mod
        monkeypatch.setattr(config_mod, "SESSIONS_DIR", str(tmp_path))
        monkeypatch.setattr(config_mod, "SEGMENT_MODE", "cloud")

        events: list[str] = []
        promoted: list[str] = []
        # Remote marker authorizes promotion of both files (H3: marker
        # presence is now required).
        marker_body = (
            '{"files": ["state.json", "masks.json"], '
            '"written_at": "", "reason": ""}'
        )

        def fake_list_staged(bucket, sid):
            events.append("list_staged")
            return ["state.json", "masks.json"]

        def fake_copy(bucket, sid, rel):
            events.append(f"copy:{rel}")
            promoted.append(rel)
            return True

        def fake_download_session(bucket, sid, local_dir, on_progress=None):
            events.append("download_session")
            os.makedirs(os.path.join(local_dir, "frames"), exist_ok=True)
            with open(os.path.join(local_dir, "frames", "0.jpg"), "w") as f:
                f.write("frame")
            with open(os.path.join(local_dir, "meta.json"), "w") as f:
                f.write('{"original_name": "x.mov"}')

        def fake_download_file(bkt, sid, rel_path, local_path):
            if rel_path == MARKER_FILENAME:
                with open(local_path, "w") as f:
                    f.write(marker_body)
                return True
            return False

        import app.services.gcs_storage as gcs_storage
        monkeypatch.setattr(gcs_storage, "list_staged_rel_paths", fake_list_staged, raising=False)
        monkeypatch.setattr(gcs_storage, "copy_staged_to_canonical", fake_copy, raising=False)
        monkeypatch.setattr(gcs_storage, "download_session", fake_download_session)
        monkeypatch.setattr(gcs_storage, "download_file_if_exists",
                            fake_download_file, raising=False)
        self._fake_sm(monkeypatch)

        DownloadSessionStep(session_id, "bkt").run(lambda p: None, threading.Event())

        # Promotion must have run before download_session
        list_idx = events.index("list_staged")
        download_idx = events.index("download_session")
        assert list_idx < download_idx, (
            f"staging promotion must precede download_session; got {events}"
        )
        assert set(promoted) == {"state.json", "masks.json"}

    def test_orphan_staged_without_marker_is_NOT_promoted(
        self, tmp_path, monkeypatch,
    ):
        """H3: a staged blob without a GCS-stored marker MUST NOT be
        promoted to canonical. Unscoped promotion (the previous orphan
        sweep) was unsafe — a fresh container could have uploaded a
        newer canonical while an old staged blob lingered, and promotion
        would clobber the newer bytes."""
        from app.services.pipeline import DownloadSessionStep

        session_id = "orphan-sess"
        session_dir = tmp_path / session_id
        session_dir.mkdir()
        (session_dir / "frames").mkdir()
        (session_dir / "frames" / "0.jpg").write_text("frame")
        (session_dir / "meta.json").write_text('{"original_name": "x.mov"}')
        # No .unsynced.json marker locally OR on GCS — the staged blob
        # exists but is unauthorized.

        import app.config as config_mod
        monkeypatch.setattr(config_mod, "SESSIONS_DIR", str(tmp_path))
        monkeypatch.setattr(config_mod, "SEGMENT_MODE", "cloud")

        copied: list[str] = []

        import app.services.gcs_storage as gcs_storage
        monkeypatch.setattr(
            gcs_storage, "list_staged_rel_paths",
            lambda b, s: ["state.json"], raising=False,
        )

        def copy_staged(bkt, sid, rel):
            copied.append(rel)
            return True

        # download_file_if_exists returns False for marker (no marker on GCS)
        monkeypatch.setattr(gcs_storage, "copy_staged_to_canonical",
                            copy_staged, raising=False)
        monkeypatch.setattr(gcs_storage, "download_file_if_exists",
                            lambda *a, **kw: False, raising=False)
        monkeypatch.setattr(gcs_storage, "download_session",
                            lambda *a, **kw: None)
        monkeypatch.setattr(gcs_storage, "upload_file",
                            lambda *a, **kw: None, raising=False)
        monkeypatch.setattr(gcs_storage, "delete_marker_blob",
                            lambda *a, **kw: None, raising=False)
        self._fake_sm(monkeypatch)

        DownloadSessionStep(session_id, "bkt").run(lambda p: None, threading.Event())
        assert copied == [], (
            "staged blob without a marker must NOT be promoted — the "
            "orphan sweep was the data-loss vector H3 targets"
        )

    def test_staged_promoted_only_for_marker_listed_rel_paths(
        self, tmp_path, monkeypatch,
    ):
        """With a remote marker listing ['state.json'], only state.json
        is promoted even if `.unsynced/` contains other staged blobs."""
        from app.services.pipeline import DownloadSessionStep
        from app.services.unsynced_marker import MARKER_FILENAME

        session_id = "scoped-promotion"
        session_dir = tmp_path / session_id
        session_dir.mkdir()
        (session_dir / "frames").mkdir()
        (session_dir / "frames" / "0.jpg").write_text("frame")
        (session_dir / "meta.json").write_text('{"original_name": "x.mov"}')

        import app.config as config_mod
        monkeypatch.setattr(config_mod, "SESSIONS_DIR", str(tmp_path))
        monkeypatch.setattr(config_mod, "SEGMENT_MODE", "cloud")

        # Marker authorizes state.json only — masks.json is orphaned
        marker_body = '{"files": ["state.json"], "written_at": "", "reason": ""}'

        copied: list[str] = []

        def fake_download_file(bkt, sid, rel_path, local_path):
            if rel_path == MARKER_FILENAME:
                with open(local_path, "w") as f:
                    f.write(marker_body)
                return True
            return False

        def copy_staged(bkt, sid, rel):
            copied.append(rel)
            return True

        import app.services.gcs_storage as gcs_storage
        monkeypatch.setattr(
            gcs_storage, "list_staged_rel_paths",
            lambda b, s: ["state.json", "masks.json"], raising=False,
        )
        monkeypatch.setattr(gcs_storage, "copy_staged_to_canonical",
                            copy_staged, raising=False)
        monkeypatch.setattr(gcs_storage, "download_file_if_exists",
                            fake_download_file, raising=False)
        monkeypatch.setattr(gcs_storage, "download_session",
                            lambda *a, **kw: None)
        monkeypatch.setattr(gcs_storage, "upload_file",
                            lambda *a, **kw: None, raising=False)
        monkeypatch.setattr(gcs_storage, "delete_marker_blob",
                            lambda *a, **kw: None, raising=False)
        self._fake_sm(monkeypatch)

        DownloadSessionStep(session_id, "bkt").run(lambda p: None, threading.Event())
        assert copied == ["state.json"], (
            f"only marker-listed blobs must be promoted; got {copied}"
        )

    def test_pulls_remote_marker_when_local_missing(
        self, tmp_path, monkeypatch,
    ):
        """Cold warm-branch case: session_dir already has frames (e.g. a
        previous container's local disk persisted), but the marker was
        written by a different process. Fetch from GCS."""
        from app.services.pipeline import DownloadSessionStep
        from app.services.unsynced_marker import MARKER_FILENAME

        session_id = "remote-marker"
        session_dir = tmp_path / session_id
        session_dir.mkdir()
        (session_dir / "frames").mkdir()
        (session_dir / "frames" / "0.jpg").write_text("frame")
        (session_dir / "meta.json").write_text('{"original_name": "x.mov"}')
        (session_dir / "state.json").write_text("{}")
        # No local marker. Stage a simulated remote marker.
        marker_text = '{"files": ["state.json"], "written_at": "", "reason": "t"}'

        import app.config as config_mod
        monkeypatch.setattr(config_mod, "SESSIONS_DIR", str(tmp_path))
        monkeypatch.setattr(config_mod, "SEGMENT_MODE", "cloud")

        def fake_download_file(bkt, sid, rel_path, local_path):
            if rel_path == MARKER_FILENAME:
                with open(local_path, "w") as f:
                    f.write(marker_text)
                return True
            return False

        uploaded_rel: list[str] = []

        def fake_upload(bkt, sid, rel_path, local_path):
            uploaded_rel.append(rel_path)

        import app.services.gcs_storage as gcs_storage
        monkeypatch.setattr(gcs_storage, "list_staged_rel_paths",
                            lambda b, s: [], raising=False)
        monkeypatch.setattr(gcs_storage, "copy_staged_to_canonical",
                            lambda *a, **kw: False, raising=False)
        monkeypatch.setattr(gcs_storage, "download_file_if_exists",
                            fake_download_file, raising=False)
        monkeypatch.setattr(gcs_storage, "download_session",
                            lambda *a, **kw: None)
        monkeypatch.setattr(gcs_storage, "upload_file",
                            fake_upload, raising=False)
        monkeypatch.setattr(gcs_storage, "delete_marker_blob",
                            lambda *a, **kw: None, raising=False)
        self._fake_sm(monkeypatch)

        DownloadSessionStep(session_id, "bkt").run(lambda p: None, threading.Event())
        # state.json was uploaded during the warm recovery loop
        assert "state.json" in uploaded_rel

    def test_cleans_up_local_staging_dir_after_full_download(
        self, tmp_path, monkeypatch,
    ):
        """download_session pulls the whole prefix including `.unsynced/`
        subdirs. We must clean that up so it doesn't pollute local state."""
        from app.services.pipeline import DownloadSessionStep

        session_id = "cleanup-sess"
        session_dir = tmp_path / session_id
        session_dir.mkdir()

        import app.config as config_mod
        monkeypatch.setattr(config_mod, "SESSIONS_DIR", str(tmp_path))
        monkeypatch.setattr(config_mod, "SEGMENT_MODE", "cloud")

        def fake_download_session(bucket, sid, local_dir, on_progress=None):
            # Simulate: download pulled canonical + staging subdir
            os.makedirs(os.path.join(local_dir, "frames"), exist_ok=True)
            with open(os.path.join(local_dir, "frames", "0.jpg"), "w") as f:
                f.write("frame")
            with open(os.path.join(local_dir, "meta.json"), "w") as f:
                f.write('{"original_name": "x.mov"}')
            # Staging subdir — must be removed
            staging = os.path.join(local_dir, ".unsynced")
            os.makedirs(staging, exist_ok=True)
            with open(os.path.join(staging, "masks.json"), "w") as f:
                f.write("stale")

        import app.services.gcs_storage as gcs_storage
        monkeypatch.setattr(gcs_storage, "list_staged_rel_paths",
                            lambda b, s: [], raising=False)
        monkeypatch.setattr(gcs_storage, "copy_staged_to_canonical",
                            lambda *a, **kw: False, raising=False)
        monkeypatch.setattr(gcs_storage, "download_session",
                            fake_download_session)
        monkeypatch.setattr(gcs_storage, "download_file_if_exists",
                            lambda *a, **kw: False, raising=False)
        self._fake_sm(monkeypatch)

        DownloadSessionStep(session_id, "bkt").run(lambda p: None, threading.Event())
        assert not (session_dir / ".unsynced").exists(), (
            "local .unsynced/ must be cleaned up after full download"
        )


class TestDownloadSessionStepRefresh:
    """DownloadSessionStep must refresh the small annotation files from GCS
    even when session_dir exists locally — stale local state.json /
    masks.json / prompts.json is the root of cross-device wipe/staleness bugs."""

    def test_refreshes_state_masks_prompts_when_dir_exists(self, tmp_path, monkeypatch):
        from app.services.pipeline import DownloadSessionStep

        session_id = "sess"
        session_dir = tmp_path / session_id
        session_dir.mkdir()
        # Pre-populate local with stale data + a frames dir so the step
        # takes the partial-refresh branch (frames present => warm container)
        (session_dir / "frames").mkdir()
        (session_dir / "frames" / "0.jpg").write_text("fake")
        (session_dir / "state.json").write_text('{"classes": [], "objects": []}')
        (session_dir / "masks.json").write_text("{}")
        (session_dir / "prompts.json").write_text("{}")
        (session_dir / "meta.json").write_text('{"original_name": "x.mov"}')

        import app.config as config_mod
        monkeypatch.setattr(config_mod, "SESSIONS_DIR", str(tmp_path))
        monkeypatch.setattr(config_mod, "SEGMENT_MODE", "cloud")

        # Mock gcs_storage with a "fresh" remote state
        fresh_state = '{"classes": [{"id": 1, "name": "real"}], "objects": []}'
        fresh_masks = '{"_versions": {"0": 1}, "0": {"1": {"rle": {"counts": "a", "size": [1, 1]}}}}'
        fresh_prompts = '{"0": {"1": {"type": "click"}}}'

        downloaded = {}
        def fake_download_file(bucket, sid, rel_path, local_path):
            content = {
                "state.json": fresh_state,
                "masks.json": fresh_masks,
                "prompts.json": fresh_prompts,
            }.get(rel_path)
            if content is not None:
                with open(local_path, "w") as f:
                    f.write(content)
                downloaded[rel_path] = True
                return True
            return False

        import app.services.gcs_storage as gcs_storage
        monkeypatch.setattr(gcs_storage, "download_file_if_exists", fake_download_file, raising=False)
        monkeypatch.setattr(gcs_storage, "download_session", lambda *a, **kw: None)

        # Mock GCSSyncManager so we don't start real threads
        import app.services.gcs_sync as gcs_sync_mod

        class FakeSM:
            def __init__(self, *a, **kw): pass
            def start(self, *a, **kw): pass
            def stop(self): pass
            def mark_dirty(self, path): pass

        monkeypatch.setattr(gcs_sync_mod, "GCSSyncManager", FakeSM)

        import threading
        step = DownloadSessionStep(session_id, "test-bucket")
        step.run(lambda p: None, threading.Event())

        # Verify local files were overwritten with GCS content
        assert (session_dir / "state.json").read_text() == fresh_state
        assert (session_dir / "masks.json").read_text() == fresh_masks
        assert (session_dir / "prompts.json").read_text() == fresh_prompts
        assert downloaded == {"state.json": True, "masks.json": True, "prompts.json": True}

    def test_invalidates_session_cache_entries_for_refreshed_files(self, tmp_path, monkeypatch):
        """After refreshing the three volatile files, any in-memory
        SessionCache entries for them must be invalidated so the next load
        picks up the fresh disk content."""
        from app.services.pipeline import DownloadSessionStep
        from app.services.session_cache import SessionCache
        from app.config import set_session_cache

        session_id = "sess2"
        session_dir = tmp_path / session_id
        session_dir.mkdir()
        # Frames must be present so we stay on the partial-refresh branch
        (session_dir / "frames").mkdir()
        (session_dir / "frames" / "0.jpg").write_text("fake")
        (session_dir / "state.json").write_text('{"classes": [], "objects": []}')
        (session_dir / "meta.json").write_text('{}')

        import app.config as config_mod
        monkeypatch.setattr(config_mod, "SESSIONS_DIR", str(tmp_path))
        monkeypatch.setattr(config_mod, "SEGMENT_MODE", "cloud")

        cache = SessionCache(str(session_dir))
        cache.load("state.json")  # populates cache with stale data
        set_session_cache(cache)

        # GCS has fresh state
        fresh = '{"classes": [{"id": 99, "name": "new"}], "objects": []}'
        def fake_download(bucket, sid, rel_path, local_path):
            if rel_path == "state.json":
                with open(local_path, "w") as f:
                    f.write(fresh)
                return True
            return False
        import app.services.gcs_storage as gcs_storage
        monkeypatch.setattr(gcs_storage, "download_file_if_exists", fake_download, raising=False)
        monkeypatch.setattr(gcs_storage, "download_session", lambda *a, **kw: None)

        import app.services.gcs_sync as gcs_sync_mod

        class FakeSM:
            def __init__(self, *a, **kw): pass
            def start(self, *a, **kw): pass
            def stop(self): pass
            def mark_dirty(self, path): pass

        monkeypatch.setattr(gcs_sync_mod, "GCSSyncManager", FakeSM)

        import threading
        DownloadSessionStep(session_id, "test-bucket").run(lambda p: None, threading.Event())

        # Next cache.load must return the fresh data, not the stale in-memory dict
        result = cache.load("state.json")
        assert result == {"classes": [{"id": 99, "name": "new"}], "objects": []}
        # Cleanup global cache for test isolation
        set_session_cache(None)

    def test_recovery_uploads_local_files_listed_in_marker(self, tmp_path, monkeypatch):
        """If a previous close/SIGTERM wrote .unsynced marker, next
        DownloadSessionStep uploads the listed local files to GCS instead of
        overwriting them with stale GCS content."""
        from app.services.pipeline import DownloadSessionStep
        from app.services.unsynced_marker import write_marker

        session_id = "recovery-sess"
        session_dir = tmp_path / session_id
        session_dir.mkdir()
        # Frames must be present so the step takes the partial-refresh branch
        (session_dir / "frames").mkdir()
        (session_dir / "frames" / "0.jpg").write_text("fake")
        # Local has "good" state with real classes — the authoritative version
        good_state = '{"classes": [{"id": 1, "name": "good"}], "objects": []}'
        (session_dir / "state.json").write_text(good_state)
        (session_dir / "meta.json").write_text('{"original_name": "r.mov"}')

        # Marker says state.json is unsynced from last close
        write_marker(str(session_dir), ["state.json"], reason="test")

        import app.config as config_mod
        monkeypatch.setattr(config_mod, "SESSIONS_DIR", str(tmp_path))
        monkeypatch.setattr(config_mod, "SEGMENT_MODE", "cloud")

        uploaded = {}
        def fake_upload(bucket, sid, rel_path, local_path):
            with open(local_path) as f:
                uploaded[rel_path] = f.read()
        stale_gcs = '{"classes": [], "objects": []}'
        def fake_download(bucket, sid, rel_path, local_path):
            with open(local_path, "w") as f:
                f.write(stale_gcs)
            return True

        import app.services.gcs_storage as gcs_storage
        monkeypatch.setattr(gcs_storage, "upload_file", fake_upload, raising=False)
        monkeypatch.setattr(gcs_storage, "download_file_if_exists", fake_download, raising=False)
        monkeypatch.setattr(gcs_storage, "download_session", lambda *a, **kw: None)

        import app.services.gcs_sync as gcs_sync_mod

        class FakeSM:
            def __init__(self, *a, **kw): pass
            def start(self, *a, **kw): pass
            def stop(self): pass
            def mark_dirty(self, path): pass

        monkeypatch.setattr(gcs_sync_mod, "GCSSyncManager", FakeSM)

        import threading
        DownloadSessionStep(session_id, "bkt").run(lambda p: None, threading.Event())

        # state.json uploaded (recovered), NOT overwritten
        assert uploaded.get("state.json") == good_state
        assert (session_dir / "state.json").read_text() == good_state
        # marker cleared after successful recovery
        from app.services.unsynced_marker import MARKER_FILENAME
        assert not (session_dir / MARKER_FILENAME).exists()

    def test_partial_recovery_rewrites_marker_with_remaining_files(
        self, tmp_path, monkeypatch,
    ):
        """#82: When only some marker-listed files upload successfully,
        DownloadSessionStep must rewrite the marker (local + remote) with
        only the still-unrecovered files — never leave the original full
        list in place, which would cause already-recovered files to be
        re-uploaded from stale local disk on the next resume (silently
        overwriting newer remote versions written by another client)."""
        from app.services.pipeline import DownloadSessionStep
        from app.services.unsynced_marker import (
            write_marker, read_marker, MARKER_FILENAME,
        )

        session_id = "partial-recovery-sess"
        session_dir = tmp_path / session_id
        session_dir.mkdir()
        (session_dir / "frames").mkdir()
        (session_dir / "frames" / "0.jpg").write_text("fake")
        (session_dir / "state.json").write_text('{"ok": true}')
        (session_dir / "masks.json").write_text('{"frames": {}}')
        (session_dir / "meta.json").write_text('{"original_name": "p.mov"}')

        # Marker lists two files; simulate state.json upload succeeding
        # and masks.json upload raising.
        write_marker(
            str(session_dir), ["state.json", "masks.json"], reason="test",
        )

        import app.config as config_mod
        monkeypatch.setattr(config_mod, "SESSIONS_DIR", str(tmp_path))
        monkeypatch.setattr(config_mod, "SEGMENT_MODE", "cloud")

        uploaded: list[tuple[str, str]] = []

        def fake_upload(bucket, sid, rel_path, local_path):
            # Succeed for state.json and for the marker rewrite;
            # fail for masks.json to simulate a partial recovery.
            if rel_path == "masks.json":
                raise RuntimeError("simulated gcs failure")
            uploaded.append((rel_path, open(local_path).read()))

        delete_calls: list[str] = []

        def fake_delete_marker(bucket, sid, marker_name):
            delete_calls.append(marker_name)

        import app.services.gcs_storage as gcs_storage
        monkeypatch.setattr(gcs_storage, "upload_file", fake_upload, raising=False)
        monkeypatch.setattr(
            gcs_storage, "download_file_if_exists",
            lambda *a, **kw: False, raising=False,
        )
        monkeypatch.setattr(gcs_storage, "download_session",
                            lambda *a, **kw: None)
        monkeypatch.setattr(gcs_storage, "delete_marker_blob",
                            fake_delete_marker, raising=False)

        import app.services.gcs_sync as gcs_sync_mod

        class FakeSM:
            def __init__(self, *a, **kw): pass
            def start(self, *a, **kw): pass
            def stop(self): pass
            def mark_dirty(self, path): pass

        monkeypatch.setattr(gcs_sync_mod, "GCSSyncManager", FakeSM)

        import threading
        DownloadSessionStep(session_id, "bkt").run(
            lambda p: None, threading.Event(),
        )

        # Marker still present and contains ONLY the unrecovered file.
        remaining = read_marker(str(session_dir))
        assert remaining == ["masks.json"], (
            f"partial recovery must rewrite marker with remaining files "
            f"only; got {remaining}"
        )
        # Remote marker was NOT deleted (still have work to do next resume).
        assert delete_calls == [], (
            f"delete_marker_blob must not be called on partial recovery; "
            f"got {delete_calls}"
        )
        # state.json was uploaded successfully (not re-attempted on the
        # next resume because masks.json will be the only remaining entry).
        state_uploads = [u for u in uploaded if u[0] == "state.json"]
        assert len(state_uploads) == 1
        assert state_uploads[0][1] == '{"ok": true}'
        # The new marker was re-uploaded to GCS (via upload_file with the
        # marker filename) so cold containers also see the reduced list.
        marker_uploads = [u for u in uploaded if u[0] == MARKER_FILENAME]
        assert len(marker_uploads) == 1, (
            "partial-recovery marker must be uploaded to GCS so cold "
            "containers get the reduced list"
        )
        # And the remote marker payload lists only the remaining file.
        import json as _json
        marker_payload = _json.loads(marker_uploads[0][1])
        assert marker_payload["files"] == ["masks.json"]
        assert marker_payload["reason"] == "partial_recovery"

    def test_full_recovery_still_clears_marker(self, tmp_path, monkeypatch):
        """Regression: the all-succeeded happy path must still clear both
        the local marker and delete the remote marker blob."""
        from app.services.pipeline import DownloadSessionStep
        from app.services.unsynced_marker import (
            write_marker, read_marker, MARKER_FILENAME,
        )

        session_id = "full-recovery-sess"
        session_dir = tmp_path / session_id
        session_dir.mkdir()
        (session_dir / "frames").mkdir()
        (session_dir / "frames" / "0.jpg").write_text("fake")
        (session_dir / "state.json").write_text('{"ok": true}')
        (session_dir / "masks.json").write_text('{}')
        (session_dir / "meta.json").write_text('{"original_name": "f.mov"}')

        write_marker(
            str(session_dir), ["state.json", "masks.json"], reason="test",
        )

        import app.config as config_mod
        monkeypatch.setattr(config_mod, "SESSIONS_DIR", str(tmp_path))
        monkeypatch.setattr(config_mod, "SEGMENT_MODE", "cloud")

        uploaded: list[str] = []

        def fake_upload(bucket, sid, rel_path, local_path):
            uploaded.append(rel_path)

        delete_calls: list[str] = []

        def fake_delete_marker(bucket, sid, marker_name):
            delete_calls.append(marker_name)

        import app.services.gcs_storage as gcs_storage
        monkeypatch.setattr(gcs_storage, "upload_file", fake_upload, raising=False)
        monkeypatch.setattr(
            gcs_storage, "download_file_if_exists",
            lambda *a, **kw: False, raising=False,
        )
        monkeypatch.setattr(gcs_storage, "download_session",
                            lambda *a, **kw: None)
        monkeypatch.setattr(gcs_storage, "delete_marker_blob",
                            fake_delete_marker, raising=False)

        import app.services.gcs_sync as gcs_sync_mod

        class FakeSM:
            def __init__(self, *a, **kw): pass
            def start(self, *a, **kw): pass
            def stop(self): pass
            def mark_dirty(self, path): pass

        monkeypatch.setattr(gcs_sync_mod, "GCSSyncManager", FakeSM)

        import threading
        DownloadSessionStep(session_id, "bkt").run(
            lambda p: None, threading.Event(),
        )

        # Local marker cleared.
        assert read_marker(str(session_dir)) is None
        # Remote marker deleted.
        assert delete_calls == [MARKER_FILENAME]
        # Both files were uploaded from local, no partial-recovery marker
        # rewrite happened (no marker upload in uploaded[]).
        assert "state.json" in uploaded
        assert "masks.json" in uploaded
        assert MARKER_FILENAME not in uploaded, (
            "full recovery must not re-upload the marker; it was deleted"
        )

    def test_cancel_event_stops_refresh_loop(self, tmp_path, monkeypatch):
        """DownloadSessionStep must check cancel_event and bail early."""
        from app.services.pipeline import DownloadSessionStep

        session_id = "cancel-sess"
        session_dir = tmp_path / session_id
        session_dir.mkdir()
        # Frames present → partial-refresh branch
        (session_dir / "frames").mkdir()
        (session_dir / "frames" / "0.jpg").write_text("fake")
        (session_dir / "state.json").write_text("{}")
        (session_dir / "meta.json").write_text("{}")

        import app.config as config_mod
        monkeypatch.setattr(config_mod, "SESSIONS_DIR", str(tmp_path))
        monkeypatch.setattr(config_mod, "SEGMENT_MODE", "cloud")

        import app.services.gcs_storage as gcs_storage
        call_count = {"n": 0}
        def fake_download(*a, **kw):
            call_count["n"] += 1
            return False
        monkeypatch.setattr(gcs_storage, "download_file_if_exists", fake_download, raising=False)
        monkeypatch.setattr(gcs_storage, "download_session", lambda *a, **kw: None)

        import app.services.gcs_sync as gcs_sync_mod

        class FakeSM:
            def __init__(self, *a, **kw): pass
            def start(self, *a, **kw): pass
            def stop(self): pass
            def mark_dirty(self, path): pass

        monkeypatch.setattr(gcs_sync_mod, "GCSSyncManager", FakeSM)

        import threading
        cancel = threading.Event()
        cancel.set()  # immediately cancelled
        result = DownloadSessionStep(session_id, "bkt").run(lambda p: None, cancel)
        assert result is None
        assert call_count["n"] == 0, "cancel_event ignored — refresh still ran"


class TestDownloadSessionStepColdStart:
    """On a fresh container, the session_dir may already have been created
    by resume_session (for SessionCache) but the frames dir is empty/missing.
    DownloadSessionStep must still perform a full GCS download so frames,
    video.mp4, and meta.json are hydrated before InitSessionStep runs."""

    def test_full_download_when_frames_dir_missing(self, tmp_path, monkeypatch):
        from app.services.pipeline import DownloadSessionStep

        session_id = "cold-sess"
        session_dir = tmp_path / session_id
        # Simulate resume_session having pre-created the empty dir
        session_dir.mkdir()

        import app.config as config_mod
        monkeypatch.setattr(config_mod, "SESSIONS_DIR", str(tmp_path))
        monkeypatch.setattr(config_mod, "SEGMENT_MODE", "cloud")

        called = {"download_session": False, "download_file": False}

        def fake_download_session(bucket, sid, local_dir, on_progress=None):
            called["download_session"] = True
            os.makedirs(os.path.join(local_dir, "frames"), exist_ok=True)
            with open(os.path.join(local_dir, "frames", "0.jpg"), "w") as f:
                f.write("fake-frame")
            with open(os.path.join(local_dir, "meta.json"), "w") as f:
                f.write('{"original_name": "x.mov"}')

        def fake_download_file(*a, **kw):
            called["download_file"] = True
            return False

        import app.services.gcs_storage as gcs_storage
        monkeypatch.setattr(gcs_storage, "download_session", fake_download_session)
        monkeypatch.setattr(gcs_storage, "download_file_if_exists", fake_download_file, raising=False)

        import app.services.gcs_sync as gcs_sync_mod

        class FakeSM:
            def __init__(self, *a, **kw): pass
            def start(self, *a, **kw): pass
            def stop(self): pass
            def mark_dirty(self, path): pass

        monkeypatch.setattr(gcs_sync_mod, "GCSSyncManager", FakeSM)

        DownloadSessionStep(session_id, "test-bucket").run(lambda p: None, threading.Event())

        assert called["download_session"] is True, "full download must run when frames are missing"
        assert called["download_file"] is False, "partial refresh must NOT run on cold start"
        assert (session_dir / "frames" / "0.jpg").exists()

    def test_full_download_when_frames_dir_empty(self, tmp_path, monkeypatch):
        """An empty frames dir is treated the same as a missing one."""
        from app.services.pipeline import DownloadSessionStep

        session_id = "empty-sess"
        session_dir = tmp_path / session_id
        session_dir.mkdir()
        (session_dir / "frames").mkdir()  # present but empty

        import app.config as config_mod
        monkeypatch.setattr(config_mod, "SESSIONS_DIR", str(tmp_path))
        monkeypatch.setattr(config_mod, "SEGMENT_MODE", "cloud")

        called = {"download_session": False}

        def fake_download_session(bucket, sid, local_dir, on_progress=None):
            called["download_session"] = True
            with open(os.path.join(local_dir, "frames", "0.jpg"), "w") as f:
                f.write("fake-frame")

        import app.services.gcs_storage as gcs_storage
        monkeypatch.setattr(gcs_storage, "download_session", fake_download_session)
        monkeypatch.setattr(gcs_storage, "download_file_if_exists", lambda *a, **kw: False, raising=False)

        import app.services.gcs_sync as gcs_sync_mod

        class FakeSM:
            def __init__(self, *a, **kw): pass
            def start(self, *a, **kw): pass
            def stop(self): pass
            def mark_dirty(self, path): pass

        monkeypatch.setattr(gcs_sync_mod, "GCSSyncManager", FakeSM)

        DownloadSessionStep(session_id, "test-bucket").run(lambda p: None, threading.Event())

        assert called["download_session"] is True


class TestDownloadSessionStepPartialDetection:
    """Issue #60: a previous download killed mid-way by SIGTERM leaves the
    frames dir with a partial set of .jpg files. Without detection, the next
    resume takes the warm-container branch and reuses the broken local state,
    then SAM3 init fails on the first missing frame. Detection compares local
    frame count against meta.json's frame_count claim."""

    def _fake_sm(self, monkeypatch):
        import app.services.gcs_sync as gcs_sync_mod

        class FakeSM:
            def __init__(self, *a, **kw): pass
            def start(self, *a, **kw): pass
            def stop(self): pass
            def mark_dirty(self, path): pass

        monkeypatch.setattr(gcs_sync_mod, "GCSSyncManager", FakeSM)

    def test_partial_frames_trigger_full_redownload(self, tmp_path, monkeypatch):
        from app.services.pipeline import DownloadSessionStep

        session_id = "partial-sess"
        session_dir = tmp_path / session_id
        session_dir.mkdir()
        # Simulate interrupted download: meta says 5 frames but only 2 on disk
        (session_dir / "frames").mkdir()
        (session_dir / "frames" / "00001.jpg").write_text("fake")
        (session_dir / "frames" / "00002.jpg").write_text("fake")
        (session_dir / "meta.json").write_text('{"frame_count": 5, "original_name": "x.mov"}')
        (session_dir / "state.json").write_text("{}")

        import app.config as config_mod
        monkeypatch.setattr(config_mod, "SESSIONS_DIR", str(tmp_path))
        monkeypatch.setattr(config_mod, "SEGMENT_MODE", "cloud")

        called = {"download_session": 0, "download_file": 0}

        def fake_download_session(bucket, sid, local_dir, on_progress=None):
            called["download_session"] += 1
            # Complete the download
            for i in range(5):
                path = os.path.join(local_dir, "frames", f"{i+1:05d}.jpg")
                with open(path, "w") as f:
                    f.write("fake")

        def fake_download_file(*a, **kw):
            called["download_file"] += 1
            return False

        import app.services.gcs_storage as gcs_storage
        monkeypatch.setattr(gcs_storage, "download_session", fake_download_session)
        monkeypatch.setattr(gcs_storage, "download_file_if_exists", fake_download_file, raising=False)
        self._fake_sm(monkeypatch)

        DownloadSessionStep(session_id, "bkt").run(lambda p: None, threading.Event())

        assert called["download_session"] == 1, "partial frames must trigger full re-download"
        assert called["download_file"] == 0, "partial refresh must not run when we re-downloaded"

    def test_missing_meta_triggers_full_redownload(self, tmp_path, monkeypatch):
        """Missing meta.json on a session_dir-with-frames is its own signal
        that the previous download didn't finish."""
        from app.services.pipeline import DownloadSessionStep

        session_id = "no-meta-sess"
        session_dir = tmp_path / session_id
        session_dir.mkdir()
        (session_dir / "frames").mkdir()
        (session_dir / "frames" / "00001.jpg").write_text("fake")
        # NO meta.json

        import app.config as config_mod
        monkeypatch.setattr(config_mod, "SESSIONS_DIR", str(tmp_path))
        monkeypatch.setattr(config_mod, "SEGMENT_MODE", "cloud")

        called = {"download_session": 0}

        def fake_download_session(bucket, sid, local_dir, on_progress=None):
            called["download_session"] += 1
            with open(os.path.join(local_dir, "meta.json"), "w") as f:
                f.write('{"frame_count": 1, "original_name": "x.mov"}')

        import app.services.gcs_storage as gcs_storage
        monkeypatch.setattr(gcs_storage, "download_session", fake_download_session)
        monkeypatch.setattr(gcs_storage, "download_file_if_exists", lambda *a, **kw: False, raising=False)
        self._fake_sm(monkeypatch)

        DownloadSessionStep(session_id, "bkt").run(lambda p: None, threading.Event())

        assert called["download_session"] == 1

    def test_complete_frames_do_not_trigger_redownload(self, tmp_path, monkeypatch):
        """Happy path: meta says 3 frames, disk has 3 frames → partial refresh
        runs, not full re-download."""
        from app.services.pipeline import DownloadSessionStep

        session_id = "complete-sess"
        session_dir = tmp_path / session_id
        session_dir.mkdir()
        (session_dir / "frames").mkdir()
        for i in range(3):
            (session_dir / "frames" / f"{i+1:05d}.jpg").write_text("fake")
        (session_dir / "meta.json").write_text('{"frame_count": 3, "original_name": "x.mov"}')
        (session_dir / "state.json").write_text("{}")

        import app.config as config_mod
        monkeypatch.setattr(config_mod, "SESSIONS_DIR", str(tmp_path))
        monkeypatch.setattr(config_mod, "SEGMENT_MODE", "cloud")

        called = {"download_session": 0, "download_file": 0}

        def fake_download_session(*a, **kw):
            called["download_session"] += 1

        def fake_download_file(*a, **kw):
            called["download_file"] += 1
            return False

        import app.services.gcs_storage as gcs_storage
        monkeypatch.setattr(gcs_storage, "download_session", fake_download_session)
        monkeypatch.setattr(gcs_storage, "download_file_if_exists", fake_download_file, raising=False)
        self._fake_sm(monkeypatch)

        DownloadSessionStep(session_id, "bkt").run(lambda p: None, threading.Event())

        assert called["download_session"] == 0, "complete frames must NOT trigger re-download"
        # download_file called for refresh of the 3 volatile files
        assert called["download_file"] >= 1
