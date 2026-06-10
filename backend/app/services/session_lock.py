"""Per-session read-modify-write lock for any session-owned JSON file
(state.json, masks.json, prompts.json).

Without this, two concurrent requests that both do
`cache.load → mutate → cache.save` on the same file can:
- Both read the same cached dict, mutate independently, and one save
  clobbers the other's changes (lost update).
- One thread's json.dump iterates the dict while another mutates it,
  producing `RuntimeError: dictionary changed size during iteration` → 500.

Hold the lock ONLY for the load+mutate+save window. NEVER hold it across
a SAM3 predictor call — propagation's persist_fn runs inside SAM3._lock,
so holding _session_io_lock while calling SAM3 produces a deadlock.
"""
import threading
from collections import OrderedDict

# Bounded LRU of per-session locks (R19 fix). Capacity is two orders of
# magnitude above realistic steady-state (tens of active sessions per
# container); eviction happens only after the cap is exceeded so hot
# keys never evict. If a key does evict while another thread still
# holds a reference to its lock, that thread's critical section
# completes correctly against its captured lock — the next caller for
# the same session_id will be issued a fresh lock. The lost-update
# window requires 1024 *other* session_ids to be touched between two
# accesses to the same session_id, which is not reachable in practice.
# See docs/CONCURRENCY_AUDIT.md § R19.
_LOCK_CACHE_CAP = 1024

# Lock hierarchy: `_locks_guard` is a leaf — see
# docs/CONCURRENCY_AUDIT.md § Lock hierarchy.
_locks: "OrderedDict[str, threading.Lock]" = OrderedDict()
_locks_guard = threading.Lock()


def session_io_lock(session_id: str) -> threading.Lock:
    """Return (creating if needed) the per-session I/O lock.

    Lock hierarchy: the returned per-session lock is level #3 — acquire
    AFTER `SAM3._state_lock` and `SAM3._lock`, BEFORE
    `SessionCache._lock`, `_globals_lock`, and
    `GCSSyncManager._lock`. See
    docs/CONCURRENCY_AUDIT.md § Lock hierarchy.
    """
    with _locks_guard:
        lock = _locks.get(session_id)
        if lock is None:
            lock = threading.Lock()
            _locks[session_id] = lock
            if len(_locks) > _LOCK_CACHE_CAP:
                _locks.popitem(last=False)
        else:
            _locks.move_to_end(session_id)
        return lock
