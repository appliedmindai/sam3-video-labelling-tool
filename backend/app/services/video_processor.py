import logging
import os
import shutil
import subprocess
import threading
import json
import time


# R33: process-global registry of live ffmpeg subprocesses.
#
# Background: `_handle_sigterm` (app/__init__.py) cancels the pipeline thread
# via cooperative cancel_event and joins it for at most 3s. `extract_frames_async`
# polls the cancel_event every 0.5s, so in the worst case ffmpeg doesn't see
# `terminate()` until ~0.5s after SIGTERM, and then `process.wait(timeout=5)`
# can push the pipeline-thread exit past the 3s join budget. `sys.exit(0)` then
# runs with ffmpeg still alive, orphaning it to the kernel.
#
# Fix: every Popen used for frame extraction registers itself here. The SIGTERM
# handler calls `kill_all_ffmpeg_procs()` directly — bypassing the poll
# interval entirely — so no ffmpeg survives into `sys.exit(0)`.
_active_ffmpeg_procs: "set[subprocess.Popen]" = set()
_ffmpeg_lock = threading.Lock()


def _register_proc(p: subprocess.Popen) -> None:
    with _ffmpeg_lock:
        _active_ffmpeg_procs.add(p)


def _unregister_proc(p: subprocess.Popen) -> None:
    with _ffmpeg_lock:
        _active_ffmpeg_procs.discard(p)


def kill_all_ffmpeg_procs(timeout: float = 2.0) -> int:
    """Terminate every registered ffmpeg subprocess, then kill any still alive.

    Called from the SIGTERM handler before `sys.exit(0)`. Returns the number
    of processes we asked to stop. Safe to call with an empty registry.
    Idempotent: finished processes are naturally no-ops for terminate()/kill().
    """
    logger = logging.getLogger(__name__)
    with _ffmpeg_lock:
        snapshot = list(_active_ffmpeg_procs)
    if not snapshot:
        return 0

    for p in snapshot:
        try:
            if p.poll() is None:
                p.terminate()
        except Exception:
            logger.warning("kill_all_ffmpeg_procs | terminate failed", exc_info=True)

    deadline = time.monotonic() + timeout
    for p in snapshot:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            p.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            try:
                p.kill()
                p.wait(timeout=1.0)
            except Exception:
                logger.warning(
                    "kill_all_ffmpeg_procs | kill failed for pid=%s",
                    getattr(p, "pid", "?"), exc_info=True,
                )
        except Exception:
            logger.warning("kill_all_ffmpeg_procs | wait failed", exc_info=True)

    logger.info("kill_all_ffmpeg_procs | signalled %d ffmpeg process(es)", len(snapshot))
    return len(snapshot)


def _parse_frame_rate(rate_str: str) -> float:
    """Safely parse FFprobe frame rate string like '30/1' or '29.97'."""
    if "/" in rate_str:
        num, den = rate_str.split("/", 1)
        try:
            return float(num) / float(den)
        except (ValueError, ZeroDivisionError):
            return 30.0
    try:
        return float(rate_str)
    except ValueError:
        return 30.0


def extract_frames(video_path: str, output_dir: str, fps: int = 2, max_dim: int = 2048) -> int:
    os.makedirs(output_dir, exist_ok=True)
    # Scale so longest side <= max_dim, preserve aspect ratio, ensure even dims
    scale_filter = (
        f"scale='if(gte(iw,ih),min({max_dim},iw),-2)':'if(lt(iw,ih),min({max_dim},ih),-2)'"
    )
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"fps={fps},{scale_filter}",
            "-q:v", "2",
            "-start_number", "0",
            os.path.join(output_dir, "%05d.jpg"),
        ],
        capture_output=True,
        check=True,
    )
    frames = [f for f in os.listdir(output_dir) if f.endswith(".jpg")]
    return len(frames)


def extract_frames_async(
    video_path: str,
    output_dir: str,
    fps: int = 2,
    max_dim: int = 2048,
    on_progress: 'Callable[[float], None] | None' = None,
    cancel_event: 'threading.Event | None' = None,
    poll_interval: float = 0.5,
) -> int:
    """Extract frames using ffmpeg with progress reporting via file counting.

    Unlike extract_frames(), this uses Popen instead of run() so we can
    report progress and respond to cancellation during extraction.

    To avoid readers observing a partially-written JPEG, ffmpeg writes
    into a staging sibling directory `<output_dir>.partial/` and we rename
    it to `output_dir` atomically on success. Routes that read frames check
    for `output_dir` presence — they either see nothing or a fully-written
    set of JPEGs, never a truncated file.

    Returns the frame count, or 0 if cancelled.
    """
    parent_dir = os.path.dirname(output_dir) or "."
    os.makedirs(parent_dir, exist_ok=True)
    staging_dir = output_dir + ".partial"
    # Clean any leftover staging from a previous crashed attempt.
    if os.path.isdir(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)

    # Get expected frame count from video duration
    info = get_video_info(video_path)
    duration = info.get("duration", 0)
    expected_frames = max(1, int(duration * fps))

    scale_filter = (
        f"scale='if(gte(iw,ih),min({max_dim},iw),-2)':'if(lt(iw,ih),min({max_dim},ih),-2)'"
    )
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps},{scale_filter}",
        "-q:v", "2", "-start_number", "0",
        os.path.join(staging_dir, "%05d.jpg"),
    ]

    # ffmpeg emits a continuous progress/stat line to stderr at the default
    # log level. An undrained subprocess.PIPE deadlocks once that output
    # exceeds the ~64KB OS pipe buffer: ffmpeg blocks on write(), stops
    # producing frames, and never exits — so extraction hangs mid-way on
    # longer or higher-fps videos (short clips finish before the buffer fills,
    # which is why this stayed hidden). Redirect stderr to a scratch file; it
    # never blocks, and we read it back only if ffmpeg fails.
    stderr_log_path = output_dir + ".ffmpeg-stderr.log"
    stderr_file = open(stderr_log_path, "w+b")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=stderr_file,
    )
    # R33: register so the SIGTERM handler can kill us directly
    # without waiting on the 0.5s cancel-event poll.
    _register_proc(process)

    try:
        while process.poll() is None:
            if cancel_event and cancel_event.is_set():
                process.terminate()
                process.wait(timeout=5)
                shutil.rmtree(staging_dir, ignore_errors=True)
                return 0

            found = len([f for f in os.listdir(staging_dir) if f.endswith(".jpg")])
            if on_progress and expected_frames > 0:
                on_progress(min(found / expected_frames, 0.99))

            time.sleep(poll_interval)

        # ffmpeg finished — check exit code
        if process.returncode != 0:
            stderr_file.flush()
            stderr_file.seek(0)
            stderr_output = stderr_file.read().decode(errors="replace")
            raise RuntimeError(
                f"ffmpeg exited with code {process.returncode}: {stderr_output[-500:]}"
            )

        frame_count = len([f for f in os.listdir(staging_dir) if f.endswith(".jpg")])
        if frame_count == 0:
            raise RuntimeError("ffmpeg produced zero frames")

        # Atomic publish: rename staging to final. POSIX rename() is atomic
        # for same-filesystem moves, so readers see either no output_dir or
        # the complete set of frames — never a partial JPEG.
        # If output_dir already exists (e.g. retry after a partial write
        # outside this function), remove it first so rename succeeds.
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.rename(staging_dir, output_dir)

        if on_progress:
            on_progress(1.0)

        return frame_count

    except Exception:
        process.kill()
        process.wait(timeout=5)
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise
    finally:
        _unregister_proc(process)
        try:
            stderr_file.close()
        except Exception:
            pass
        if os.path.exists(stderr_log_path):
            os.remove(stderr_log_path)


def get_video_info(video_path: str) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            video_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    info = json.loads(result.stdout)
    video_stream = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if video_stream is None:
        raise ValueError("Uploaded file contains no video stream")
    return {
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "duration": float(video_stream.get("duration", 0)),
        "fps": _parse_frame_rate(video_stream.get("r_frame_rate", "30/1")),
    }
