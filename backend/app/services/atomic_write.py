"""Atomic JSON file writes — write to a unique .tmp, then os.rename().

Prevents corruption when:
- Auto-save flush reads a file being written by propagation
- SIGTERM arrives during a file write
- Any crash during json.dump()
- Two writers target the same file concurrently (e.g. `PUT /session/state`
  racing the class-import `POST /session/classes` handler — without a unique
  tmp per call they would truncate each other's tmp and produce a
  FileNotFoundError on rename, or interleave bytes into `state.json`).

os.rename() is atomic on POSIX (same filesystem), so the file is
always in a complete state on disk. With a unique tmp per call, last
rename wins and no writer ever sees a tmp file disappear mid-flight.
"""
import contextlib
import json
import logging
import os
import threading
import time
import uuid

logger = logging.getLogger(__name__)

# Orphans older than this are safe to sweep. atomic_json_dump renames in ms;
# this is three orders of magnitude beyond any legitimate writer.
ORPHAN_TMP_AGE_S = 3600


def atomic_json_dump(data: dict, path: str, indent: int | None = None) -> None:
    """Write JSON atomically: write to a unique .tmp file, then rename over the target."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}"
    try:
        with open(tmp_path, "w") as f:
            json.dump(data, f, indent=indent)
        os.rename(tmp_path, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_path)
        raise


def sweep_orphan_tmp_files(root_dir: str, max_age_s: float = ORPHAN_TMP_AGE_S) -> int:
    """Unlink orphaned `*.tmp.*` files under root_dir older than max_age_s.

    Closes R41: if the process dies from SIGKILL / OOM / segfault between
    `json.dump` and `os.rename` in atomic_json_dump (or the equivalent
    download/marker tmp paths), the tmp file orphans. These accumulate
    over time, waste tmpfs space, and can clutter debug listings. The
    sweep is safe because:
      - `atomic_json_dump` completes in milliseconds; max_age_s is orders
        of magnitude beyond any legitimate writer's window
      - Callers invoke this at moments when no live writer targets the
        session (startup; resume before activation)

    Returns the number of files unlinked. Never raises — sweep errors
    are logged and swallowed so startup/resume are never blocked by a
    disk-cleanup problem.
    """
    if not os.path.isdir(root_dir):
        return 0
    cutoff = time.time() - max_age_s
    removed = 0
    try:
        # Match the three tmp patterns produced by atomic_json_dump,
        # gcs_storage.download_file_if_exists, and unsynced_marker.write_marker.
        # All share the `<canonical>.tmp.<pid>` prefix. os.walk (not glob)
        # because glob skips dotfiles by default (`.unsynced.json.tmp.*`)
        # and we need to catch those too.
        for dirpath, _dirnames, filenames in os.walk(root_dir):
            for name in filenames:
                if ".tmp." not in name:
                    continue
                tmp = os.path.join(dirpath, name)
                try:
                    if not os.path.isfile(tmp):
                        continue
                    if os.path.getmtime(tmp) >= cutoff:
                        continue
                    os.unlink(tmp)
                    removed += 1
                    logger.info("sweep_tmp | unlinked orphan %s", tmp)
                except FileNotFoundError:
                    # Raced with another sweep or a legitimate rename — ignore.
                    pass
                except OSError as exc:
                    logger.warning(
                        "sweep_tmp | could not unlink %s: %s", tmp, exc,
                    )
    except Exception:
        logger.warning(
            "sweep_tmp | sweep failed under %s", root_dir, exc_info=True,
        )
    return removed
