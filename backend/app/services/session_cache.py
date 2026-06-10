"""In-memory cache for session JSON files (masks.json, prompts.json, state.json).

Lazy-loads from disk on first read, updates in-place on writes, flushes to disk
synchronously via atomic_json_dump.

Reader/writer safety contract:
- load(filename) returns a SHALLOW TOP-LEVEL COPY. Callers can freely add or
  remove top-level keys without affecting the cache or other in-flight callers.
  However, nested values (dicts/lists one level down) are shared references.
- Writers that mutate nested values MUST clone them first, then call save()
  with the fully-updated top-level dict. This keeps readers holding old
  top-level refs safe because nobody mutates the nested dicts they point to.
- save(filename, data) stores `data` as the new cache entry and writes to disk
  atomically. Callers must NOT mutate `data` after calling save(), as the cache
  now owns it and other readers may be iterating it.

Thread safety: SessionCache holds an internal RLock around its map operations.
External per-session write serialization is provided separately by
session_io_lock() in app.services.session_lock.
"""
import json
import os
import threading

from app.services.atomic_write import atomic_json_dump


class SessionCache:
    def __init__(self, session_dir: str) -> None:
        self._session_dir = session_dir
        self._cache: dict[str, dict] = {}
        # Lock hierarchy: `SessionCache._lock` is level #4 — acquired
        # under `session_io_lock` in RMW paths, but also taken
        # standalone by GET readers. Nothing inside SessionCache acquires
        # any other lock in the hierarchy. See
        # docs/CONCURRENCY_AUDIT.md § Lock hierarchy.
        self._lock = threading.RLock()

    @property
    def session_dir(self) -> str:
        return self._session_dir

    def load(self, filename: str) -> dict:
        """Return a shallow top-level copy of the cached dict.

        Caching rule: once a non-empty dict has been cached, it is the source
        of truth (disk mutations outside of this cache are intentionally ignored
        for the lifetime of the cache -- callers must go through save()/invalidate()).

        Refresh exception: if the cached entry is empty ({}), we treat it as
        "not yet populated" and re-check disk on every load. This handles the
        race where the file didn't exist at first-read time but was created
        afterwards (e.g., by DownloadSessionStep finishing in the background).

        Returns a NEW top-level dict -- callers can safely add/remove top-level
        keys without affecting other callers. Nested values are shared; writers
        must clone nested dicts they intend to mutate.
        """
        with self._lock:
            cached = self._cache.get(filename)
            if cached:
                return dict(cached)

            filepath = os.path.join(self._session_dir, filename)
            if os.path.exists(filepath):
                with open(filepath) as f:
                    data = json.load(f)
            else:
                data = {}

            self._cache[filename] = data
            return dict(data)

    def save(self, filename: str, data: dict, indent: int | None = None) -> None:
        """Replace the cache entry with `data` and write to disk atomically.

        Callers must NOT mutate `data` after calling save() -- the cache now
        owns it and other readers may be iterating it concurrently.
        """
        with self._lock:
            self._cache[filename] = data
            filepath = os.path.join(self._session_dir, filename)
            atomic_json_dump(data, filepath, indent=indent)

    def invalidate(self, filename: str) -> None:
        """Remove entry from cache, forcing a re-read on next load."""
        with self._lock:
            self._cache.pop(filename, None)

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
