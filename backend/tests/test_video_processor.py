import os
import subprocess
import threading
import time

import pytest

@pytest.fixture
def sample_video(tmp_path):
    video_path = tmp_path / "test.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=red:s=320x240:d=2",
         "-pix_fmt", "yuv420p", str(video_path)],
        capture_output=True, check=True,
    )
    return str(video_path)

def test_extract_frames_creates_jpegs(sample_video, tmp_path):
    from app.services.video_processor import extract_frames
    output_dir = str(tmp_path / "frames")
    frame_count = extract_frames(sample_video, output_dir, fps=2)
    assert frame_count >= 3
    files = sorted(os.listdir(output_dir))
    assert all(f.endswith(".jpg") for f in files)
    assert files[0] == "00000.jpg"

def test_extract_frames_respects_fps(sample_video, tmp_path):
    from app.services.video_processor import extract_frames
    output_dir = str(tmp_path / "frames")
    frame_count = extract_frames(sample_video, output_dir, fps=1)
    assert frame_count >= 1
    assert frame_count <= 3


def test_extract_frames_async_atomic_publish(sample_video, tmp_path):
    """Readers must never observe the final frames_dir until extraction is
    complete. ffmpeg writes into `<frames_dir>.partial/`; we poll for the
    final dir's presence during extraction and assert it only appears after
    run finishes (#87)."""
    from app.services.video_processor import extract_frames_async

    output_dir = str(tmp_path / "frames")
    staging_dir = output_dir + ".partial"
    observed_final_during_extraction: list[bool] = []
    observed_staging_during_extraction: list[bool] = []
    stop_poll = threading.Event()

    def poll() -> None:
        while not stop_poll.is_set():
            # Snapshot both paths; if we ever see final_dir while staging_dir
            # is still present, atomic publish is broken.
            final_exists = os.path.isdir(output_dir)
            staging_exists = os.path.isdir(staging_dir)
            if staging_exists:
                observed_staging_during_extraction.append(True)
                if final_exists:
                    observed_final_during_extraction.append(True)
            time.sleep(0.02)

    poll_thread = threading.Thread(target=poll)
    poll_thread.start()
    try:
        frame_count = extract_frames_async(sample_video, output_dir, fps=2)
    finally:
        stop_poll.set()
        poll_thread.join(timeout=2)

    assert frame_count >= 3
    assert os.path.isdir(output_dir)
    # Staging dir must be gone after atomic rename.
    assert not os.path.isdir(staging_dir)
    # We must have observed the staging dir at some point (proves ffmpeg
    # actually wrote there), and we must NEVER have seen both the staging
    # dir and the final dir coexisting (proves the publish is atomic).
    assert observed_staging_during_extraction, (
        "Poll thread never saw the staging dir — test is not exercising the race"
    )
    assert not observed_final_during_extraction, (
        "Final frames_dir became visible while staging was still active — "
        "readers could see a partial set"
    )


def test_extract_frames_async_cancel_cleans_staging(sample_video, tmp_path):
    """Cancellation mid-extract must leave neither the final nor the staging
    dir behind."""
    from app.services.video_processor import extract_frames_async

    output_dir = str(tmp_path / "frames")
    staging_dir = output_dir + ".partial"

    cancel = threading.Event()
    cancel.set()  # cancel immediately on first poll

    frame_count = extract_frames_async(
        sample_video, output_dir, fps=2, cancel_event=cancel,
    )

    assert frame_count == 0
    assert not os.path.isdir(output_dir)
    assert not os.path.isdir(staging_dir)


def test_kill_all_ffmpeg_procs_is_noop_when_empty():
    """Safe to call with nothing registered (the common case outside pipelines)."""
    from app.services.video_processor import kill_all_ffmpeg_procs
    assert kill_all_ffmpeg_procs(timeout=0.5) == 0


def test_kill_all_ffmpeg_procs_terminates_running_extraction(sample_video, tmp_path):
    """R33 / #86: running extractions registered with the process-global
    ffmpeg registry must be killable without relying on the cancel_event
    poll interval. Simulates the SIGTERM path: extraction in flight, the
    handler calls kill_all_ffmpeg_procs before sys.exit(0)."""
    from app.services import video_processor
    from app.services.video_processor import (
        extract_frames_async,
        kill_all_ffmpeg_procs,
    )

    output_dir = str(tmp_path / "frames")
    started = threading.Event()
    result: dict = {}

    # Use a much slower poll so the cancel_event would take seconds to fire —
    # proving kill_all_ffmpeg_procs doesn't depend on cooperative polling.
    def run() -> None:
        try:
            result["count"] = extract_frames_async(
                sample_video, output_dir, fps=2, poll_interval=5.0,
            )
        except Exception as e:
            result["error"] = e
        finally:
            started.set()

    t = threading.Thread(target=run)
    t.start()

    # Wait until ffmpeg is registered (it happens right after Popen).
    deadline = time.time() + 5.0
    while time.time() < deadline:
        with video_processor._ffmpeg_lock:
            if video_processor._active_ffmpeg_procs:
                break
        time.sleep(0.01)
    else:
        pytest.fail("ffmpeg subprocess was never registered")

    killed = kill_all_ffmpeg_procs(timeout=2.0)
    assert killed >= 1

    # extract_frames_async's except-path does process.wait(timeout=5) after
    # its own process.kill(), so give the thread comfortably more than that.
    t.join(timeout=15.0)
    assert not t.is_alive(), "extraction thread did not exit after kill"

    # Registry must be drained by the finally block in extract_frames_async.
    with video_processor._ffmpeg_lock:
        assert not video_processor._active_ffmpeg_procs, (
            "Popen was not unregistered after extraction exit"
        )


def test_extract_frames_async_leftover_staging_is_cleaned(sample_video, tmp_path):
    """A leftover `<frames_dir>.partial/` from a previous crashed run must
    not prevent a fresh extraction from succeeding."""
    from app.services.video_processor import extract_frames_async

    output_dir = str(tmp_path / "frames")
    staging_dir = output_dir + ".partial"
    os.makedirs(staging_dir)
    # Drop a bogus file into the leftover staging dir.
    with open(os.path.join(staging_dir, "junk.jpg"), "wb") as f:
        f.write(b"not-a-real-jpeg")

    frame_count = extract_frames_async(sample_video, output_dir, fps=2)

    assert frame_count >= 3
    assert os.path.isdir(output_dir)
    assert not os.path.isdir(staging_dir)
    # Junk file must not have survived — staging was wiped before the run.
    assert "junk.jpg" not in os.listdir(output_dir)


def test_extract_frames_async_does_not_deadlock_on_large_ffmpeg_stderr(tmp_path, monkeypatch):
    """ffmpeg emits a continuous progress/stat line to stderr at the default
    log level. An undrained subprocess.PIPE deadlocks once that output exceeds
    the ~64KB OS pipe buffer: ffmpeg blocks on write(), stops producing frames,
    and never exits — extraction hangs mid-way. Short clips finish before the
    buffer fills, which is why it stayed hidden until a full-length, high-fps
    upload (IMG_4130.MOV at fps=15 hung at ~48%). stderr must therefore be
    redirected to a file, never handed an undrained PIPE.
    """
    import app.services.video_processor as vp

    monkeypatch.setattr(
        vp, "get_video_info",
        lambda p: {"duration": 10.0, "fps": 30.0, "width": 640, "height": 480},
    )

    captured = {}

    class FakePopen:
        def __init__(self, cmd, stdout=None, stderr=None):
            captured["stderr"] = stderr
            self.returncode = 0
            # Mirror real Popen: .stderr is a stream only when stderr=PIPE.
            self.stderr = None
            staging = os.path.dirname(cmd[-1])
            # Flood stderr well past the 64KB pipe buffer. A real file absorbs
            # this; an undrained PIPE would block ffmpeg in production.
            if hasattr(stderr, "write"):
                stderr.write(b"x" * (256 * 1024))
            for i in range(3):
                open(os.path.join(staging, f"{i:05d}.jpg"), "wb").close()

        def poll(self):
            return 0

        def wait(self, timeout=None):
            return 0

        def kill(self):
            pass

    monkeypatch.setattr(vp.subprocess, "Popen", FakePopen)

    output_dir = str(tmp_path / "frames")
    count = vp.extract_frames_async("/fake/video.mp4", output_dir, fps=2)

    assert captured["stderr"] is not subprocess.PIPE, (
        "ffmpeg stderr must not be an undrained PIPE — it deadlocks on >64KB output"
    )
    assert count == 3
    assert os.path.isdir(output_dir)
    # The stderr log is a scratch sibling of output_dir and must be cleaned up.
    assert not os.path.exists(output_dir + ".ffmpeg-stderr.log")
