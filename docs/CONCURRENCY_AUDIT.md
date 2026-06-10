# Backend concurrency audit — SAM3 Video Labelling Tool

**Last reviewed: 2026-04-14 — status annotations added** (many findings closed since initial filing; see "Fixed since initial filing" below).

> **Note (open-source release):** the token-auth middleware (`app/middleware/auth.py`, `_token_locks`, `_auth_cache`) discussed in some findings (e.g. R19) was removed in the open-source release — cloud mode now reads a fixed `GCS_BUCKET` env var. Those findings are kept for historical context; the lock-hierarchy and sync-manager analysis is still current.

Scope: `backend/app/` on worktree `concurrency-hardening` at HEAD (post PR #49 / #50 / #54).
Runtime model: gunicorn `--threads 4`, Cloud Run concurrency=8, `max-instances=1`. One container, many request threads, several daemon threads, one SAM3 GPU, one `_active_session_cache`, one `_active_sync_manager`, one `ServiceState`.

Thesis (original): **locking was piecemeal and the invariants were unenforced.** `session_io_lock` protects writer-vs-writer on state/masks/prompts, but *nothing* synchronised readers against writers on the same in-memory dict. The "active cache / active sync / active service state" globals were plain module attributes with zero locking. As of 2026-04-14 most of the CRITICAL / HIGH findings have been fixed — see TL;DR below. A second tier of MEDIUM/LOW concerns is still open.

This document is now a living architecture + status reference: each finding has a **Status** line, and a "Since this audit was filed" block documents the primitives that were added to close them.

---

## TL;DR

### Fixed since initial filing

- **R1** — SessionCache reader/writer race (`221f3b0`). `load()` now returns a shallow top-level copy; writers clone nested dicts before mutating; `SessionCache` has an internal `_lock`.
- **R2 / #55 / #58** — `close_session` silently dropped failed uploads — two-commit fix. `c64552d` (#55) added `flush_with_retry` + `.unsynced` marker + 503; `d8ef2be` (#58) added `promote_deferred()` into the teardown sequence so deferred masks from active propagation aren't lost.
- **R3** — `GCSSyncManager._timer` use-after-stop (`0024960`). `_stopped` flag, timer lifecycle and schedule/tick all guarded by `_lock`; `start()` refuses stopped managers.
- **R4** — `_active_session_cache` / `_active_sync_manager` lifecycle race (`0024960` / `8c40dda`). Swaps now serialised by `_globals_lock`; failed handoff writes an `.unsynced` marker.
- **R6** — `init_session` now gated against active pipeline for a different session (`0024960`). 409 returned at `segment.py:29-32`.
- **R10 / #57** — SIGTERM handler — two-commit fix. `8c40dda` restructured the handler (cancel propagation first, retry flush, marker on failure, then `sm.stop()`); `d8ef2be` (#57) added `sm.promote_deferred()` before the retry so in-flight propagation masks are not silently dropped. (The original claim that `DownloadSessionStep` lacked cancel checks was incorrect — four checks exist.)
- **R12** — Atomic annotation-file refresh in `DownloadSessionStep` (`8c40dda` + `23e8482`). Volatile files now written via tmp + rename. Full `download_session` remains non-atomic but only runs when `session_dir` is absent, so no concurrent reader exists.
- **R13** — `_active_object.pop` now inside `_lock` in every callsite (`0024960`).
- **R27** — `close_session` pops all bookkeeping dicts inside `_lock` (`0024960`).
- **#80** — Same-session lost-update on `PUT /api/session/state/<sid>`. `state.json` now carries a monotonic `version` counter; `put_state` 409s on stale submissions and returns the current state so the frontend can reconcile. Legacy PUTs without a `version` field remain a force-overwrite escape hatch.
- **R14 / #67** — `export_coco` (`export.py:22,25`) and local `list_sessions` (`video.py:136`) now snapshot state/masks under `session_io_lock(session_id)`. Defensive hardening — no concurrent writer exists on these paths today, but any future writer would have silently produced partial exports. Minimal perf cost; export is not a hot path.
- **R38 / #89** — Single-worker invariant now enforced at runtime via `backend/gunicorn_config.py` (`when_ready` + `post_worker_init` hooks exit non-zero when `workers != 1`). `deploy/entrypoint.sh` loads the config via `--config /app/backend/gunicorn_config.py`. `scripts/check_entrypoint_workers.sh` provides a static check that the entrypoint keeps `--workers 1` and the `--config` flag. Side finding (`--threads 4` < Cloud Run `--concurrency 8`) is still untriaged; tracked separately if it ever bites.
- **R32 / #85** — SSE subscriber backpressure sentinel delivery. Error events and the end-of-stream sentinel now go through `_put_critical` (`sam3_service.py:_put_critical`) which drops the oldest queued frame and retries `put_nowait` on `queue.Full`, guaranteeing that a backed-up SSE reader still receives the terminal `None` and any error event. Per-frame results remain lossy by design (lossy under load is acceptable; silent hang on terminate is not). See PR #99.
- **#81** — SIGTERM handler now joins the propagation daemon thread (`0991d88`, PR #100). `SAM3Service` tracks the propagation thread per session in `_propagation_threads` and exposes `join_propagation(session_id, timeout)`. The SIGTERM handler calls it with a 2s budget after `cancel_propagation` and BEFORE `_sigterm_flush_and_clear_globals`, giving the propagation `finally` block a chance to deliver the SSE `None` sentinel and toggle `set_propagating(False)` itself rather than racing `sys.exit(0)`. `promote_deferred()` in the flush path remains the load-bearing safety net for masks. Tests in `test_sam3_service.py` cover absent-thread, fast-exit, and timeout paths.
- **#82** — `DownloadSessionStep` partial-recovery marker rewrite. The previous happy-path check (`if unsynced and all(r in recovered for r in unsynced)`) cleared the marker only when EVERY file uploaded successfully; on partial recovery both the local `.unsynced.json` and its GCS copy were left with the original full list. Next resume then re-uploaded the already-recovered files from stale local disk, silently overwriting any newer remote version another client may have written in the interim. Fix: on partial recovery rewrite the marker (local via `write_marker`, remote via `upload_marker_to_gcs`) with only the still-unrecovered files; on full recovery keep the old `clear_marker` + `delete_marker_blob` path. Tests in `test_pipeline.py::TestDownloadSessionStepRefresh` cover both the partial and full-recovery branches.
- **R34 / #87** — Atomic frame-dir publish. ffmpeg now writes JPEGs into `<frames_dir>.partial/` and `ExtractFramesStep` promotes the staging dir via `os.rename` (POSIX-atomic on same filesystem). `GET /api/video/frame/<sid>/<i>` and `GET /api/video/frames/<sid>` branch on `os.path.isdir(frames_dir)` — readers now see either no `frames_dir` (returns 404) or a complete set of fully-written JPEGs. A request thread can no longer `send_file` a partially-written frame while ffmpeg is mid-write. Cancellation and leftover-staging paths are cleaned; tests in `test_video_processor.py` (`test_extract_frames_async_atomic_publish`, `test_extract_frames_async_cancel_cleans_staging`, `test_extract_frames_async_leftover_staging_is_cleaned`) cover all three. Fix commit: see git log for `fix/issue-87-atomic-frame-extraction`.
- **R15 / #68** — Propagation thread's `finally`/`except` no longer resurrects closed-session state (`56ea768`, PR #103). Both branches now acquire `_propagation_lock` and only mutate `_propagation_state[session_id]` when the entry is still present (i.e., `close_session` didn't pop it mid-run); on clean success the entry is `pop()`ed rather than overwritten with `{"status": "idle"}` (equivalent under `get_propagation_status`'s missing-key default, but avoids zombie accumulation across many runs). Tests in `test_sam3_service.py` (`test_run_propagation_success_pops_propagation_state`, `test_run_propagation_finally_does_not_resurrect_closed_session_state`, `test_run_propagation_except_does_not_resurrect_closed_session_state`, `test_run_propagation_except_writes_failed_when_session_still_open`) cover the success, close-during-run (finally branch), close-during-error (except branch), and still-open error paths. R7/R30 (broader read/write synchronization) remain open.
- **R33 / #86** — ffmpeg subprocess orphan risk on SIGTERM closed. `extract_frames_async` registered its `Popen` with a new process-global registry (`_active_ffmpeg_procs` + `_ffmpeg_lock` in `backend/app/services/video_processor.py`), and the SIGTERM handler now calls `kill_all_ffmpeg_procs(timeout=2.0)` directly after cancelling the pipeline thread but before the globals-flush step. This bypasses the 0.5s cancel-event poll inside `extract_frames_async` entirely: even if the pipeline thread's 3s join timeout expires with ffmpeg mid-`process.wait(timeout=5)`, the SIGTERM path `terminate()`s (and, after the 2s budget, `kill()`s) every registered ffmpeg before `sys.exit(0)` runs. Registration/unregistration is wrapped around the single `Popen` call site; the `finally` block drains the registry so the common (non-SIGTERM) cancel + exception paths remain the authoritative cleanup. Tests in `test_video_processor.py` (`test_kill_all_ffmpeg_procs_is_noop_when_empty`, `test_kill_all_ffmpeg_procs_terminates_running_extraction`) cover the empty-registry fast path and a live extraction spawned with `poll_interval=5.0` to prove the kill does not depend on cooperative polling. Fix commit: see git log for `fix/issue-86-ffmpeg-sigterm-registry`.
- **R41 / #92** — `atomic_json_dump` tmp-file orphan sweep. SIGKILL / OOM / C-extension segfault between `json.dump` and `os.rename` orphans the `<canonical>.tmp.<pid>.<tid>.<uuid>` tmp file; the equivalent `.tmp.<pid>` patterns in `gcs_storage.download_file_if_exists` and `unsynced_marker.write_marker` share the risk. New `sweep_orphan_tmp_files(root_dir, max_age_s=3600)` in `backend/app/services/atomic_write.py` walks `root_dir` and unlinks `*.tmp.*` entries with mtime older than 1h (three orders of magnitude beyond any legitimate writer — safe against live races). Wired at two points: startup (`backend/app/__init__.py::_cleanup_partial_sessions`, runs before any request thread exists) and resume (`DownloadSessionStep.run` before annotation-file refresh, where no sync manager or propagation is yet active for the session). Cloud Run tmpfs wipes on SIGKILL so orphans do not survive container death in prod — this closes the local-dev leak and guards warm-container scale-to-one lifetimes. Tests in `test_atomic_write.py` (`test_sweep_unlinks_stale_orphans_and_preserves_fresh`, `test_sweep_tolerates_missing_root`, `test_sweep_returns_zero_when_no_orphans`, `test_sweep_never_raises_on_unlink_error`) cover stale/fresh discrimination across all three tmp patterns (including dotfile `.unsynced.json.tmp.*` — the sweep uses `os.walk`, not `glob`, so dotfiles are matched). Fix commit: see git log for `fix/issue-92-sweep-orphan-tmp-files`.
- **#97** — SIGTERM handler now captures `sm.stop()`'s `FlushResult`. Previously the return value was discarded, so any `mark_dirty` that arrived in the narrow window between `flush_with_retry` returning and `stop()` acquiring `_lock` would be attempted by `stop()`'s internal final flush — and if that upload failed (GCS brownout during shutdown) the delta was silently dropped. Fix mirrors the close_session pattern (`routes/segment.py:321-332`): on non-ok `stop_result`, write a `persist_unsynced(..., reason="sigterm_stop_failed")` marker so the delta is recoverable on next resume. ~20 LOC in `app/__init__.py`; new regression test `test_sigterm_handler.py::test_sigterm_stop_failure_writes_marker` simulates the concurrent-mark-during-stop path. R10 residual window (noted in original R10 entry) is now closed. Fix commit: see git log for `fix/issue-97-sigterm-stop-result`.
- **#118** — `persist_unsynced` under-recorded on partial-staging failure. Discovered during validation of this audit's concurrency hardening pass (not in the original filing). `marker_files = staged if staged else sorted(failed)` (`unsynced_marker.py:155`) dropped files that raised during `stage_unsynced_file` whenever any other file staged successfully — mixed-success failure modes left the failed-to-stage files with NEITHER a `.unsynced/<rel>` blob on GCS NOR a marker entry, so neither cold- nor warm-container resume could recover them. Fix tracks `failed_to_stage` explicitly (distinct from skipped-missing-from-disk, which stays excluded since those files are not recoverable anywhere) and sets `marker_files = sorted(set(staged) | set(failed_to_stage))` in cloud mode. Local-only mode (bucket=None) still uses `sorted(failed)`. New regression test `test_persist_unsynced_partial_staging_records_all_dirty_files` in `test_unsynced_marker.py` mocks half the dirty files to raise and asserts the marker contains both sets. Existing `test_persist_unsynced_skips_missing_local_files` preserved (missing-from-disk still excluded). Fix commit: see git log for `fix/persist-unsynced-partial-staging`.
- **R11 / #66** — Propagation `persist_fn` -> SAM3 deadlock invariant now has test coverage (`bf2335e`, PR #109). Three tests in `test_sam3_service.py` lock in the canonical `SAM3._lock` (#2) -> `session_io_lock` (#3) order. `test_propagation_persist_fn_runs_inside_sam3_lock` proves propagation holds `_lock` across `persist_fn` (the precondition that makes ordering matter). `test_propagation_persist_fn_acquires_session_io_lock_without_deadlock` runs a realistic `persist_fn` body under a 5s hard deadline on a worker thread, so the canonical order cannot silently self-deadlock. `test_reverse_lock_order_deadlocks_as_documented` reproduces the failure mode: while propagation holds `_lock` across a blocked `persist_fn`, a second thread that acquires `session_io_lock` first and then asks for `_lock` times out — recording the deadlock signature a future maintainer of a misordered route will see. No production code change; the WARNING comment at `segment.py:163-167` and `session_lock.py:11-14` remains the canonical rule. A runtime re-entry assertion on `SAM3Service` public methods (the proposal in #66) was deliberately scoped out: it would enforce a stronger rule ("`persist_fn` must not touch SAM3 at all") than R11 documents ("`persist_fn` must not acquire `_lock`"), and the three new tests already catch both regressions. Fix commit: see git log for `fix/issue-66-persist-fn-deadlock-test`.
- **R7 / R30 / #63 / #74** — `_propagation_state` lifecycle lock (`5290b04`, PR #108). `_propagation_lock` now guards every read and write, not just the start-of-propagation gate: `get_propagation_status` snapshots under the lock (was a bare `dict(state)` copy), the inner-loop `frames_processed` read-modify-write is now locked, and `close_session`'s `pop` nests `_propagation_lock` inside `_lock` (safe because `_propagation_lock` is a leaf — no call site acquires `_lock` while holding it). Lock-declaration comment updated from "gates status-check-then-set" to "guards all reads/writes". Concurrent readers can no longer observe a torn `frames_processed` or a flip-flopped status. New tests in `test_sam3_service.py` cover snapshot isolation, idle-for-missing, and a 500-iteration concurrent reader/writer that asserts the counter field is never absent or non-integer. R15 (#68) closed the `except` / `finally` resurrection path; this closes the broader read/write synchronization.
- **R8 / #64** — `_propagation_subscribers` list mutation now serialised by `_propagation_lock` (`a54619d`, PR #110). Append (`subscribe_propagation`), remove (its finally block), and iterate (per-frame fan-out + error/finally fan-outs in `_run_propagation`) all share the same lock; iteration uses a snapshot-then-iterate pattern so the lock is never held across `queue.put_nowait` / `_put_critical`. The TOCTOU on `subscribe_propagation` (propagation finally pop between `get` and `append`) is closed — atomic check + append under the lock. `close_session` and `_run_propagation` finally consolidate the pops into the same `_propagation_lock` critical section as `_propagation_state` / `_propagation_threads`. Lock-declaration comment updated to record the broader scope. New tests in `test_sam3_service.py` (under `R8: _propagation_subscribers list mutation under _propagation_lock`) cover the no-session early-return, single-subscriber happy path, post-close subscribe, the finally-pops-subscribers invariant, and a 4-thread concurrent subscribe/unsubscribe churn against a 100-frame fan-out (would raise `RuntimeError: list changed size during iteration` on the unlocked code).
- **R9 / #65** — Benchmark route no longer reads SAM3 internals without `_lock` (`5ce4178`, PR #111). New `SAM3Service.debug_snapshot()` accessor (`sam3_service.py`) acquires `self._lock` briefly and returns a plain dict (`device`, `backend`, `model`, `model_params`, `native_predictor_loaded`, `sessions_loaded`, `loaded_session_ids`); the `parameters()` iteration runs inside the lock so a concurrent `_ensure_model` cannot cause a torn read or a partial-init `param_count`. `GET /api/benchmark/health` and the `POST /api/benchmark/<sid>` response builder both call `sam.debug_snapshot()` instead of poking `sam._device`, `sam._model`, `sam._sessions`, `sam._native_predictor`, `sam._backend`. Tests in `test_sam3_service.py` (`test_debug_snapshot_returns_safe_keys_when_model_not_loaded`, `test_debug_snapshot_acquires_lock`) cover the no-model-loaded fast path and prove the call serialises against an external `_lock` holder. Diagnostic-only fix; no behaviour change.
- **R5 / #62** — `_sessions_cache` cache-miss single-flight. Concurrent `GET /api/video/sessions` callers that arrived on a TTL boundary each fanned out to a fresh `list_blobs` + per-session `meta.json`/`state.json` walk — N-way duplicate work under the documented `concurrency=8`. Fix mirrors the `_token_locks` pattern in `app/middleware/auth.py`: a leaf `_sessions_cache_locks_guard` protects a `dict[str, Lock]` map; the per-bucket lock wraps the populate path with a re-check of the cache after acquisition so siblings released after the populate read the populated entry instead of re-fetching. Fast path (TTL-valid cache) is unchanged — no lock acquired. Different buckets never serialise (lock is per-bucket). The lock map is intentionally not pruned — bounded by the bucket count (one entry per GCS bucket the service ever touches), so unlike `_token_locks` / `session_lock._locks` (now bounded-LRU under #69) it has a natural low ceiling and does not need eviction. Tests in `test_gcs_storage_single_flight.py` cover the 8-thread cold-cache race (asserts exactly 1 GCS call), bucket independence (no cross-bucket serialisation), the warm-cache fast path (no lock-induced GCS calls), and a forced-expiry race (one re-populate, not N). Fix commit: see git log for `fix/issue-62-sessions-cache-single-flight`.
- **R22 / #71** — **FIXED** — `invalidate_sessions_cache` is no longer called before the pipeline uploads the new session to GCS. Previously `upload_video` invoked `invalidate_sessions_cache(g.bucket)` immediately after `start_pipeline` returned (`video.py:96-98`), so any concurrent `GET /api/video/sessions` between the invalidate and `ExtractFramesStep`'s first GCS upload could repopulate the cache from a bucket that didn't yet contain the new session — leaving the cache stale for up to the 30s TTL. Fix moves the invalidation into an `on_complete` callback passed to `start_pipeline` (same pattern as the resume route), so the cache clears on the pipeline thread after every step has succeeded and all GCS uploads are durable, but before the `ready`-phase transition the frontend observes. On pipeline failure `on_complete` is not called — the cache stays as-is, which is correct because the failed session isn't in GCS either. No new tests (upload route has no existing tests; the cache single-flight invariants covered by `test_gcs_storage_single_flight.py` are unchanged). Fix commit: see git log for `fix/issue-71-invalidate-cache-after-upload`.
- **R39 / #90** — Process-wide lock hierarchy documented. Seven locks (`_state_lock`, `SAM3._lock`, `session_io_lock`, `SessionCache._lock`, `_globals_lock`, `GCSSyncManager._lock`, `_locks_guard` / `_token_locks_guard`) had no canonical acquire order — only two pair-wise inversions were written down (`session_lock.py:11-14`, `sam3_service.py:1451-1457`). The rest was implicit and greppable only by reading every `with self._lock`. Fix is documentation-only (runtime checker deferred as follow-up if a deadlock ever manifests): a new top-level "Lock hierarchy" section in this doc lists the seven locks top→bottom with the code paths that established each nesting, and every lock declaration carries a one-line `# Lock hierarchy: #N in docs/CONCURRENCY_AUDIT.md` comment so `git grep "Lock hierarchy"` surfaces all the sites when auditing a new multi-lock acquisition. No code paths change. See git log for `fix/issue-90-document-lock-hierarchy`.
- **R37 / #88** — Cold `_ensure_model` heartbeat logging. On a scale-from-zero Cloud Run container, the first request that reaches any SAM3 method blocks `_lock` for ~30s while weights are transferred to GPU and the backbone warms. The existing logs emitted only a single "Loading native SAM 3 model" line followed by silence until load completed — indistinguishable from a stall when an operator read Cloud Run logs during a support incident. Fix adds a `SAM3Service._log_heartbeat_during(operation, interval_s=5.0)` context manager that spawns a short-lived daemon thread logging `"sam3 | <operation> still in progress | elapsed=Ns"` every 5s until the block exits, and wraps the model-load section of `_ensure_model` with it. A new bookend line (`"sam3 | cold model load begins | backend=... | device=..."`) paired with the existing post-load summary gives operators a complete cold-load timeline. Severity was LOW — no 500, no data loss, only visibility — so options (a) eager init at boot and (b) 503+Retry-After with a new polling contract were both rejected as more invasive than the problem warrants ((a) would add ~30s to every cold-start even for cheap requests like `/api/status`; (b) would require frontend changes to distinguish cold-init 503s from other 503s). No test added — the heartbeat is a log-only side effect on a code path that needs an actual GPU model load to exercise, and mocking would test the mock. Fix commit: see git log for `fix/issue-88-ensure-model-heartbeat-logs`.

### Still open

- **R19 / #69** — **FIXED** — `_token_locks` and `session_lock._locks` now bounded-LRU. Both lock maps are `OrderedDict`s capped at 1024 entries (`_TOKEN_LOCK_CACHE_CAP` in `app/middleware/auth.py`, `_LOCK_CACHE_CAP` in `app/services/session_lock.py`). Every access `move_to_end`s on hit; a miss appends and `popitem(last=False)`s when the cap is exceeded. Cap sits two orders of magnitude above realistic steady-state (tens of active sessions / rotating tokens per container lifetime) so hot keys never evict. Eviction of a cold `_token_locks` entry is benign (worst case: two siblings hit the Dashboard API, `_auth_cache` writes are GIL-safe); eviction of a cold `session_lock._locks` entry would only cause a lost-update window if 1024 *other* session_ids were touched more recently than this one, which is not reachable in this service. Weak-ref eviction was rejected for `session_lock._locks` because `with session_io_lock(sid):` only keeps a strong ref during the critical section — a GC between two callers would hand out different locks for the same key. Tests: `test_session_lock.py` (cap enforcement, hot-key retention, concurrent-factory safety) and `test_auth_middleware.py::test_token_lock_cache_is_bounded_and_lru`. Fix commit: see git log for `fix/issue-69-bounded-lru-lock-maps`.
- **R20** — **FIXED in `20d393e`** — both `replay_prompts_if_needed` and `ensure_active_object` now pass `cache=get_session_cache()` to `load_all_prompts`, matching the writer path. Reader and writer now share the same in-memory snapshot; disk is no longer touched at read time.
- **R28** — Informational: `_lock` is `RLock`, re-entry in `replay_prompts_if_needed` is fine.
- **R29 / #73** — **FIXED** — `_cleanup_partial` now holds `session_io_lock(session_id)` across the whole check+rmtree block (`sam3_service.py:_cleanup_partial`). Any concurrent RMW writer (`state.json`, masks, prompts) that takes the same lock is serialised out of the window, so a late-arriving `POST /api/session/state/<sid>` can no longer drop a valid state.json between the existence check and `shutil.rmtree`. Lock ordering is safe: callers (`_run_pipeline` cancel paths) hold `_state_lock` (#1), `_cleanup_partial` never takes SAM3 `_lock` (#2), and `session_io_lock` (#3) is strictly below both — no new inversion. Tests in `test_sam3_service.py` under "R29 / #73" cover the lock-held blocking invariant and three sanity paths (empty-session removal, state.json preservation, frames preservation). Fix commit: see git log for `fix/issue-73-cleanup-partial-toctou`.
- **R31** — **FIXED in `244ffb2`** — `mark_dirty` now raises `RuntimeError` when the manager is stopped; all 9 call sites migrated to `mark_dirty_safe(rel_path)` which retries once via `get_sync_manager()`. Combined with B4's atomic `close_active_session`, writes are routed to the next-installed manager or fail-closed when none is active.
- **R40** — frontend beforeunload audit → **FIXED** — see `docs/FRONTEND_DATA_LOSS_AUDIT.md` (master issue #121). All 13 findings (F1–F13) resolved: F1/F3/F6/F12 via PRs #129/#130/#128, F13 (`MASKS_WINDOW` eviction causing reload-time client data regression) via commit `658fad5` (revision `segment-service-00052-hkc`).

### Scope notes

- **Frontend mask cache / IndexedDB** is OUT OF SCOPE of this backend audit, but is a relevant boundary for any data-loss argument. `frontend/src/maskCache.ts` is strictly read-through (client never pushes cached masks back to the server; staleness resolved via `/api/session/masks/<sid>/versions`). The frontend audit (`docs/FRONTEND_DATA_LOSS_AUDIT.md`, issue #121) covers the `beforeunload` / `pagehide` → `flushSyncBeacon` → `POST /api/segment/flush` handshake from the client side, plus the broader post-propagation render/reload paths. All 13 findings (F1–F13) are now FIXED — see the audit doc for details.

### Deploy reference

- Latest deployed commit: track in `docs/deploy-segment-cloud-run.md` or via `gcloud run revisions list --service=segment-service`. Verify before treating a "FIXED" status as live in production.

---

## Since this audit was filed — primitives added to close the findings

This block documents the new pieces of machinery introduced by `0024960`, `8c40dda`, `221f3b0`, `23e8482`, and `c64552d`. Reviewers coming to this doc for architecture should read this section before the historical finding list — the in-source comments on these primitives are the current source of truth.

### `unsynced_marker.py` — deferred-recovery marker

A JSON sidecar (`<session_dir>/.unsynced.json`) that records which GCS uploads were still dirty when a session had to be torn down without a clean flush.

Written by:
- **SIGTERM handler** (`app/__init__.py:151-156`) — after `flush_with_retry` still leaves files dirty, before `sys.exit(0)`.
- **`close_session` route** (`segment.py:302-304`) — when `flush_with_retry` fails persistently; 503 returned to client with `retry_possible: true`, sync manager kept alive.
- **`close_session` route fallback** (`segment.py:325-327`) — when the final `sm.stop()` flush fails; 503 returned with `retry_possible: false`.
- **`set_sync_manager`** (`config.py:80-91`) — when replacing one manager with another and the old manager's `stop()` leaves files dirty.

Read by:
- **`DownloadSessionStep`** (`pipeline.py:244-260`) on resume — re-uploads listed files from local disk (local is authoritative) and clears the marker on full success.

Happens-before contract: a file ending up in the marker means the local copy is newer than GCS. The next `DownloadSessionStep.run` MUST upload-from-local before any GCS refresh pass for the same `rel_path`, otherwise we would overwrite the local authoritative copy with stale GCS content. This is enforced by the `recovered: set[str]` check at `pipeline.py:267-269` which skips refresh for any path just uploaded from the marker.

### `SessionCache._lock` (`221f3b0`)

Internal `threading.RLock` on `SessionCache` itself (`session_cache.py:32`). Guards the `_cache` map for `load` / `save` / `invalidate` / `clear`. The previous contract ("caller holds `session_io_lock`") was half-enforced — readers (GET routes) didn't. Now:

1. `load(filename)` returns a NEW shallow top-level dict. Callers may freely add or remove top-level keys.
2. Nested values are still shared references, so **writers must clone nested dicts before mutating** (see `mask_storage.update_frame_masks` at `mask_storage.py:310, 319` for the pattern).
3. `save(filename, data)` rebinds the cache entry atomically and writes to disk via `atomic_json_dump`.

This alone does not give point-in-time snapshot consistency — `GET /masks` can still see a mid-propagation mix of persisted and not-yet-persisted frames — but it eliminates the `RuntimeError: dictionary changed size during iteration` class and the "iterate a half-mutated RLE" corruption class.

### `GCSSyncManager._stopped` timer guards (`0024960`)

`_stopped: bool` flag (`gcs_sync.py:60`) and lifecycle hardening added in `0024960`:
- `start()` raises if already stopped (`gcs_sync.py:198-202`).
- `stop()` takes `_lock`, sets `_stopped = True`, cancels the timer (`gcs_sync.py:204-234`).
- `_schedule()` early-returns if `_stopped` (`gcs_sync.py:236-242`).
- `_tick()` still re-calls `_schedule()` in its `finally`, but `_schedule` now no-ops, so the post-stop re-arm is closed.

Note: `0024960`'s `stop()` did NOT promote deferred files into dirty. That behavior was added later by `d8ef2be` (see below). Do not `git blame` the promote line to this commit.

### `promote_deferred()` and promote-on-stop (`d8ef2be`, PR #59, closes #57/#58)

`promote_deferred()` (`gcs_sync.py:93-105`) and the reshape of `stop()` to promote-then-flush were added in `d8ef2be` (PR #59). This is the **critical invariant** for not losing masks on teardown: it MUST be called before the final flush on every forced-teardown path (SIGTERM, close_session, set_sync_manager).

During propagation, masks.json writes are deferred (not uploaded every frame) because `_propagating=True`. If the propagation thread is daemon=True (it is) and the process dies before its `finally` block runs `set_propagating(False)` to promote, all masks produced during that propagation run would be lost. `promote_deferred()` is the explicit path for the teardown handlers to force that promotion themselves.

Call sites (all added/updated by `d8ef2be`):
- `app/__init__.py:151` — SIGTERM handler calls `promote_deferred()` before `flush_with_retry`.
- `segment.py:288` — `close_session` route calls `promote_deferred()` before `flush_with_retry`.
- `config.py:73` — `set_sync_manager` calls `old.promote_deferred()` before stopping the old manager.

> See the agent-memory note `project_gcs_sync_teardown_invariant.md` for the rationale.

### Atomic annotation-file refresh + partial-download detection (`23e8482`)

1. **`gcs_storage.download_file_if_exists`** (`gcs_storage.py:176-199`) — writes to `<path>.tmp.<pid>.<uuid>` and `os.rename`s. Used by `DownloadSessionStep` for `state.json` / `masks.json` / `prompts.json` refresh (`pipeline.py:273-285`). Closes the read-during-mutate race where a concurrent `GET /state` could observe a truncated file.

2. **Partial-download detection** (`pipeline.py:205-232`) — `DownloadSessionStep` verifies local frame count against `meta.json.frame_count`. If fewer frames are on disk than meta claims (e.g. SIGTERM mid-download on a prior resume), it forces a full re-download instead of proceeding with a broken session. Closes the "session_dir exists but frames missing → SAM3 init silently succeeds on a truncated video" crash mode.

Note: the full `gcs_storage.download_session` (`gcs_storage.py:144-164`) is still non-atomic (`blob.download_to_filename(local_path)` direct), but it only runs when `session_dir` is absent or frames are missing — no concurrent reader can be inside the session at that point.

### Critical data-loss fixes B1-B5 (2026-04-14)

Plan: `docs/superpowers/plans/2026-04-14-critical-data-loss-fixes.md`. Five independent fixes, landed separately:

**B1 (`0805974`) — Beacon `/api/segment/flush` hardening (#75).** The `beforeunload` beacon now runs `promote_deferred() → flush_with_retry(max_attempts=1)` and writes an `.unsynced` marker on failure so the next resume's `DownloadSessionStep` can recover. Always returns HTTP 200 (beacons drop non-200 silently). See `segment.py::flush_sync`.

**B3 (`84fd07e`) — `set_sync_manager` marker-on-stop-raised (#77).** Previously, if `old.stop()` itself raised, the outer try/except swallowed the exception and skipped the marker-write path entirely. Now `dirty_snapshot()` is captured before `stop()`, stop is in its own try/except, and the marker is written on both "returned failed" and "raised" paths. Over-records rather than under-records — upload is idempotent. See `config.py::set_sync_manager`.

**B4 (`244ffb2`) — `mark_dirty` stopped-check + `close_active_session` (#78, closes R31 #84).** `GCSSyncManager.mark_dirty` now raises `RuntimeError` if stopped. Module-level helper `mark_dirty_safe(rel_path)` wraps the 9 callers in mask/prompt/session storage so a stop+swap race routes the write to the next-installed manager. New `config.close_active_session()` clears both sync-manager and session-cache globals atomically under `_globals_lock` without double-stopping the caller-owned sm.

**B2 (`298d032`) — GCS staging for scale-to-zero recovery (#76).** The `.unsynced.json` marker is now ALSO uploaded to GCS at `<session_id>/.unsynced.json`, and the dirty bytes themselves are staged to `<session_id>/.unsynced/<rel_path>`. New `gcs_storage` helpers: `stage_unsynced_file`, `copy_staged_to_canonical`, `list_staged_rel_paths`, `delete_staged_blob`, `delete_marker_blob`. New `unsynced_marker.persist_unsynced` is the single teardown-time entry point — used by SIGTERM handler, close_session (flush-fail, stop-fail), set_sync_manager, and beacon. `DownloadSessionStep._promote_staged_blobs` runs BEFORE `download_session` on cold resume and includes an orphan sweep (staged blobs without a marker are still recovered).

**B5 (`1779c29`) — Atomic sync-manager + cache swap on resume (#79).** Closes the two-tab race where `DownloadSessionStep` installed `sm_B` mid-pipeline while `cache_A` was still serving reads. New `config.install_active_session(sm, cache)` and `clear_active_session_for_resume()` swap both globals atomically under `_globals_lock`. `SAM3Service.start_pipeline` accepts `on_complete: Callable[[], None]` which fires on the pipeline thread after all steps succeed and BEFORE the ready transition — the frontend's first post-ready request sees a fully hydrated session. Pipeline never mutates globals; it's pure "hydrate the session_dir, then signal complete." `_stop_and_record(old)` is factored so `set_sync_manager` and `install_active_session` share the stop+marker pipeline.

Invariant established by B4+B5: globals move only through `(None, None) → (sm_B, cache_B) → (None, None)`. There is no observable `(sm_A, cache_B)` or `(None, cache_B)` intermediate state.

---

## Lock hierarchy (#90 / R39)

The process has seven non-trivial locks. Any code path that holds more than one at a time MUST acquire them in the order below (top → bottom). Releasing may happen in any order. This ordering has been inferred from the existing code; the "Established-by" column cites the code path that already nests the two locks in question, which is what pins each level to the one above it.

New code that needs to hold two of these locks at once must:
1. Either add itself to the hierarchy at a level consistent with the table,
2. Or justify in a comment why the ordering is irrelevant for this call (e.g. "lock B is never held by any thread that could be waiting on lock A").

When in doubt, **do not nest.** Drop the outer lock before acquiring the inner; re-check the condition after re-entering.

| # | Lock | File / line | Typical hold duration | Established-by |
|---|---|---|---|---|
| 1 | `SAM3Service._state_lock` | `sam3_service.py:283` | microseconds (atomic `ServiceState` swap) | `finalize_close` holds `_state_lock` across a teardown callable that takes `_globals_lock` via `close_active_session` (`sam3_service.py:1420-1425`). Pins `_state_lock` above `_globals_lock`. |
| 2 | `SAM3Service._lock` (RLock) | `sam3_service.py:272` | ms (single inference) to **minutes** (propagation) | `_run_propagation` holds `_lock` for the entire loop and calls `persist_fn` per frame, which acquires `session_io_lock` (`sam3_service.py:1180-1218`, `segment.py:168`). Pins `_lock` above `session_io_lock`. The progress callback in init also re-enters `_state_lock` under `_lock`, which is why `start_pipeline` probes `_lock` non-blocking BEFORE taking `_state_lock` (`sam3_service.py:1451-1457`). |
| 3 | `session_io_lock(sid)` | `session_lock.py:21` | ms to tens of ms | Every RMW route acquires `session_io_lock`, then the cache ops inside it take `SessionCache._lock`, and `mark_dirty_safe` briefly takes `_globals_lock` → `GCSSyncManager._lock`. Pins `session_io_lock` above `SessionCache._lock`, `_globals_lock`, and `GCSSyncManager._lock`. |
| 4 | `SessionCache._lock` (RLock) | `session_cache.py:32` | microseconds | Held only for map mutations / shallow copies. No nesting observed in the outward direction — nothing inside `SessionCache` acquires any other lock. |
| 5 | `_globals_lock` | `config.py:38` | microseconds (pointer swap) | `mark_dirty_safe` acquires `_globals_lock` via `get_sync_manager()` and then calls `mark_dirty` which takes `GCSSyncManager._lock` (the `_globals_lock` is released between the two by `get_sync_manager`, but semantically they are ordered). `finalize_close` also takes `_state_lock` → `_globals_lock`. Pins `_globals_lock` above `GCSSyncManager._lock`. |
| 6 | `GCSSyncManager._lock` | `gcs_sync.py:84` | microseconds (set mutations) | Innermost lock on the sync-manager path. No sync-manager method acquires any other lock. Upload I/O inside `flush()` happens OUTSIDE the lock by design (`gcs_sync.py:168-199`). |
| 7 | `_locks_guard` (`session_lock.py:18`), `_token_locks_guard` (`auth.py`), `_token_locks[key]` (`auth.py`) | — | microseconds | Leaf bookkeeping. Never acquired while any of 1–6 is held (and by inspection, nothing under these locks re-enters any of 1–6). |

### Established ordering rules (carry forward)

Two inversions were already written down in comments; they are reproduced here as the authoritative list:

1. **Never hold `session_io_lock` across a SAM3 predictor call.** `sam3_service._run_propagation` holds `SAM3._lock` for the whole run and calls `persist_fn` per frame; `persist_fn` acquires `session_io_lock` (`segment.py:163-169`). A route that took `session_io_lock` first and then called SAM3 would invert this and deadlock against propagation. The rule is enforced at every `session_io_lock` call site in the route handlers — SAM3 calls always happen OUTSIDE the `with session_io_lock(...):` block (see `session.py:133-138`, `session.py:156-165`, `segment.py:98-113`, `segment.py:229-232`). First stated in `session_lock.py:11-14`.
2. **`start_pipeline` must probe `_lock` non-blocking before taking `_state_lock`.** The progress callback (`_make_progress_fn`) runs inside `_lock` (during init) and takes `_state_lock`, which would invert `_state_lock` → `_lock`. `start_pipeline` sidesteps this by a non-blocking `_lock.acquire(); _lock.release()` probe BEFORE `_state_lock` (`sam3_service.py:1451-1457`). First stated in the comment block at those lines.

### Deadlock-by-construction paths already in the codebase

These are not violations; they are the paths that **pinned** the ordering above. They are listed so the next reviewer doesn't mistake them for bugs:

- `_run_propagation`: holds `SAM3._lock` → per-frame `persist_fn` → `session_io_lock(sid)` → `SessionCache._lock` → (via `mark_dirty_safe`) `_globals_lock` → `GCSSyncManager._lock`. Every lock below `SAM3._lock` in the table is reachable from inside it.
- `finalize_close`: holds `SAM3._state_lock` → `teardown()` → `close_active_session` → `_globals_lock`. Levels 1 and 5.
- `route handlers` (PUT /state, click, box, delete, etc.): acquire `_globals_lock` (briefly via `get_session_cache()`), release, then `session_io_lock` → `SessionCache._lock` → `GCSSyncManager._lock`. Because `_globals_lock` is always released before `session_io_lock` is acquired, there is no route-path nesting of 5-before-3; the only `_globals_lock` → `GCSSyncManager._lock` chain is the `mark_dirty_safe` path, which is itself underneath `session_io_lock` in the RMW routes.

### Why this is documentation-only

A runtime lock-order checker was considered and rejected for this change. It would require wrapping every one of the seven locks in an instrumented context manager that records per-thread acquisition order, plus a per-lock ID that survives `threading.Lock` / `RLock` construction — invasive for an issue that has not yet produced an observed deadlock. If a deadlock does surface, the follow-up is to add that wrapper + a tests-only debug mode that asserts order, not to revisit this doc.

### Rule for reviewers

Every PR that **adds a new `with self._lock:` or equivalent** where the enclosing context already holds another lock from the table MUST:
- Add itself to the table (if it introduces a new lock), or
- Add a one-line comment at the call site: `# Lock hierarchy: N before M — see docs/CONCURRENCY_AUDIT.md § Lock hierarchy`.

`git grep "Lock hierarchy"` surfaces every site currently annotated — use it as the starting point when auditing a new multi-lock acquisition.

---

## Section 1 — Thread inventory

This process can have, simultaneously:

### 1.1 Gunicorn main thread
- Started by gunicorn; runs Flask's signal handling and the process-level SIGTERM handler registered in `app/__init__.py:131`.
- Touches: `_active_sync_manager`, `SAM3Service._pipeline_thread`, `SAM3Service._sessions` (indirectly via `sam.cancel_pipeline` + `sam._pipeline_thread.join`).

### 1.2 Gunicorn worker request threads (up to 4; Cloud Run concurrency=8)
- 4 threads share one Python process, each serving a Flask request.
- Touch: everything the request handlers touch — `SessionCache`, `session_io_lock`, SAM3 `_lock`/`_state_lock`, `GCSSyncManager._lock`, module globals in `config.py`, `_sessions_cache` in `gcs_storage.py`, `_auth_cache` + `_token_locks` in `auth.py`.
- Note: Cloud Run concurrency is 8 but gunicorn is `--threads 4`. Cloud Run will queue the other 4 — but if gunicorn workers > 1 ever gets flipped on, the in-memory singleton assumptions break entirely. Today: single worker, 4 threads.

### 1.3 `SAM3Service._pipeline_thread`
- Spawned by `SAM3Service.start_pipeline` (`sam3_service.py:1351`) as a daemon=False `threading.Thread` named `pipeline-<prefix>`.
- Runs `_run_pipeline` which sequentially runs each `PipelineStep.run(on_progress, cancel_event)`. Each step may:
  - `ExtractFramesStep`: spawn ffmpeg subprocess (see 1.4), poll directory, write `meta.json`, optionally upload to GCS, optionally create+install a `GCSSyncManager` via `set_sync_manager`.
  - `DownloadSessionStep`: blocking GCS downloads, `set_sync_manager`, `cache.invalidate` on the active `SessionCache`.
  - `InitSessionStep`: calls `sam.init_session` which itself acquires `self._lock` (re-entrant) from the pipeline thread.
- Touches: `_service_state` (under `_state_lock`), `_sessions`, `_session_meta`, `_active_object`, `_lock` (from `init_session`), `SESSIONS_DIR`, `_active_sync_manager`, `_active_session_cache`, and the SAM3 backbone/model.
- Lifecycle: non-daemon, joined by SIGTERM handler with timeout=5.

### 1.4 ffmpeg subprocess (within `ExtractFramesStep`)
- Spawned via `subprocess.Popen` in `video_processor.py:76`. Not a Python thread but shares the frames directory.
- The pipeline thread polls the directory every 0.5s to compute progress. While ffmpeg writes frames, Flask handler threads can call `GET /api/video/frames/<sid>` and `GET /api/video/sessions` which also `os.listdir(frames_dir)` — generally benign (we only care about `.jpg` files, and Python's `os.listdir` is atomic per call).

### 1.5 Propagation thread (`SAM3Service._run_propagation`)
- Spawned inside `start_propagation` at `sam3_service.py:1114`, always `daemon=True`.
- Runs the inference loop, acquires `self._lock` at `sam3_service.py:1134` and holds it for the *entire* propagation (can be minutes). Inside the lock, calls `persist_fn` per frame, which acquires `session_io_lock(session_id)` from inside `SAM3._lock`.
- Touches: `_sessions`, `_session_meta`, `_propagation_state`, `_propagation_subscribers`, `_cancel_events`, `_active_sync_manager` (via `get_sync_manager().set_propagating(...)`), and writes masks through `_persist_masks_from_result` → `mask_storage.update_frame_masks` → `SessionCache.save` → `atomic_json_dump`.
- Notifies subscriber queues (SSE handlers below).

### 1.6 SSE subscriber threads
- Flask request threads serving `POST /api/segment/propagate` and `GET /api/segment/propagate/subscribe/<sid>` (`segment.py:184`).
- Each one calls `sam.subscribe_propagation(session_id)` which appends a new `queue.Queue(maxsize=50)` into `self._propagation_subscribers[session_id]` (mutation of a list with no lock), pulls results out, and removes the queue on finally. Multiple subscribers per session are allowed.

### 1.7 `GCSSyncManager._timer`
- A `threading.Timer` spawned by `_schedule()`; each tick spawns a new timer at `_tick()` finally. Runs `flush()` which iterates `self._dirty`, does network uploads (blocking), then re-arms.
- Touches: `_dirty`, `_deferred`, `_propagating` (under `_lock`), session files on disk (read-only, via `os.path.isfile` and `upload_file`).

### 1.8 SIGTERM handler (`_handle_sigterm` in `app/__init__.py:115`)
- Installed by `signal.signal(SIGTERM, ...)`. Python signal handlers in CPython run on the **main thread** — if the main thread is in a C extension (like SAM3 inference or ffmpeg's `wait()`), delivery is deferred until the extension returns. Practically: runs after whatever blocking call the main thread was doing.
- Calls `sm.stop()` (flush + cancel timer), `sam.cancel_pipeline()`, and `sam._pipeline_thread.join(timeout=5)`, then `raise SystemExit(0)`.
- Touches: `_active_sync_manager`, `SAM3Service._state_lock`, `_cancel_events` (via `cancel_propagation` — actually it only calls `cancel_pipeline`, not `cancel_propagation`), and the pipeline thread.
- There's a SECOND SIGTERM handler in `backend/run.py:11` that also flushes `sm.flush()` then `sys.exit(0)`. But `create_app` re-registers the handler at line 131 — so `run.py`'s handler is overwritten. However on Cloud Run with gunicorn, `run.py` is NOT the entry point; gunicorn imports `app` directly, so only `create_app`'s handler is installed. Worth noting anyway.

### 1.9 Auth token refresh critical section (`auth.py`)
- No dedicated background thread, but `_token_locks[cache_key]` is a per-token lock that serializes Dashboard API calls across the 4 request threads for the same token. Works correctly.
- `_auth_cache` is a plain dict mutated under the per-token lock but READ without it (line 82, 97). In CPython this is OK for dict get because of the GIL, but visibility of the tuple write is fine because tuple assignment to a dict key is atomic at the bytecode level. Still, the `_token_locks` dict itself is guarded by `_token_locks_guard` — correct.

### 1.10 `queue.Queue` internal thread-safety
- `queue.Queue` is thread-safe; subscriber queues and `put_nowait`/`get(timeout=30)` are fine.

**Threads you'd expect but don't exist:** no worker pool for uploads (flushes happen on the timer thread or on the close route thread); no dedicated frame-loader thread (lazy); no HTTP/2 bidi streaming thread.

---

## Section 2 — Shared state inventory

Notation: R = readers, W = writers. Locks in **bold**.

### 2.1 Module globals in `app/config.py`
| Name | Type | R | W | Lock |
|---|---|---|---|---|
| `_active_sync_manager` | `GCSSyncManager \| None` | every request thread (mask_storage, prompt_storage, session_manager call `get_sync_manager()` to mark dirty), propagation thread (`set_propagating`), SIGTERM handler, pipeline thread (ExtractFramesStep / DownloadSessionStep), close route | `set_sync_manager`: pipeline thread (both steps), open_session route, resume_session (via DownloadSessionStep), close route (sets to None) | **`_globals_lock`** (`0024960`). Pointer swap under lock; old manager's `stop()` runs outside lock to avoid starving readers during GCS upload; `.unsynced` marker written on persistent stop failure. |
| `_active_session_cache` | `SessionCache \| None` | every request thread (all routes that pass `cache=get_session_cache()`), DownloadSessionStep (`cache.invalidate`) | `set_session_cache`: open_session, resume_session, close route | **`_globals_lock`** (`0024960`). |
| `BOOT_ID`, `BOOT_TIME`, `SESSIONS_DIR`, `SAM3_*` | immutable config | everyone | set once at import | N/A |

### 2.2 `app/services/gcs_storage.py`
| Name | Type | R | W | Lock |
|---|---|---|---|---|
| `_sessions_cache` | `dict[str, tuple[float, list[dict]]]` | request threads hitting `GET /api/video/sessions`, `POST /api/video/upload` (duplicate check via `find_session_by_md5_gcs`) | same, plus `invalidate_sessions_cache` | **None** |
| `_get_client._client` | GCS client singleton | all GCS ops | first caller | **None** (GCS client is thread-safe per google-cloud-storage docs) |

### 2.3 `app/services/session_lock.py`
| Name | Type | R | W | Lock |
|---|---|---|---|---|
| `_locks` | `dict[str, threading.Lock]` | every call to `session_io_lock` | same | **`_locks_guard`** |

### 2.4 `app/services/session_cache.py`
| Name | Type | R | W | Lock |
|---|---|---|---|---|
| `SessionCache._cache` (the outer dict) | `dict[str, dict]` | every route that passes `cache=` (i.e. state/masks/prompts loads), DownloadSessionStep (`invalidate`, `session_dir`) | `save`, `invalidate`, `clear` — called from routes under `session_io_lock` AND (GET handlers) without it | **`SessionCache._lock`** (RLock, `221f3b0`). Guards the map for `load` / `save` / `invalidate` / `clear`. |
| The inner dict per filename (the `data` dict returned by `load`) | `dict` | same as above | writers clone nested dicts before mutating (`update_frame_masks` at `mask_storage.py:310, 319`), then rebind via `cache.save()` | **Shallow-copy-on-read** (`221f3b0`). `load()` returns a new top-level dict; nested dicts are shared-ref, so writers must clone before mutating. |

### 2.5 `SAM3Service` class attributes (singleton)
| Name | Type | R | W | Lock |
|---|---|---|---|---|
| `_model`, `_processor`, `_native_predictor`, `_text_model`, `_text_processor`, `_device`, `_backend`, `_cuda_ampere_plus`, `_native_autocast_dtype` | loaded model state | `_lock` holders; diagnostic reads via `debug_snapshot()` (also under `_lock`) | `_ensure_model`, `_ensure_text_model` under `_lock` | **`_lock`** for both readers and writers (R9 fixed via #65) |
| `_sessions` | `dict[session_id → inference_state]` | `_lock` holders; `debug_snapshot()` snapshots under `_lock` | `_lock` holders | **`_lock`** mostly; `get_loaded_session_ids` reads without lock |
| `_session_meta` | `dict[session_id → dict]` | `_lock` holders | `_lock` holders | **`_lock`** |
| `_active_object` | `dict[session_id → obj_id]` | `ensure_active_object`, `remove_object`, `reset_session`, `close_session` — all under `_lock` except some `pop` at line 1288–1297 which are **outside** the lock | same | **`_lock`** (mostly; pops leak) |
| `_text_sessions` | `dict` | `_lock` | `_lock` | **`_lock`** |
| `_propagation_state` | `dict[session_id → status dict]` | `start_propagation` (under `_propagation_lock`), `get_propagation_status`, `_run_propagation`, `close_session`, `_handle_sigterm` (indirectly) | same, plus final "idle" write at line 1191 | **`_propagation_lock`** for entry, then no lock for the rest — broken |
| `_propagation_subscribers` | `dict[session_id → list[Queue]]` | `subscribe_propagation`, `_run_propagation`, `close_session` | same | **None for list mutation** |
| `_cancel_events` | `dict[session_id → Event]` | `cancel_propagation`, `_run_propagation`, `close_session` | same | **None** |
| `_lock` | RLock | — | — | — |
| `_state_lock` | Lock | — | — | — |
| `_propagation_lock` | Lock | — | — | — |
| `_service_state` | `ServiceState` (frozen) | `get_service_state`, `_run_pipeline`, `dismiss_error`, `cancel_pipeline`, `start_pipeline`, `close_session`, status route | same | **`_state_lock`** — clean |
| `_pipeline_thread` | `Thread \| None` | SIGTERM handler | `start_pipeline` | **None** — SIGTERM reads while main thread is assigning |
| `_pipeline_cancel` | Event | pipeline thread, routes (via `cancel_pipeline`) | `start_pipeline.clear()`, `cancel_pipeline.set()` | **Event is thread-safe** |

### 2.6 `SessionCache` cached inner dicts — THE core shared state
- `cache._cache['state.json']` — read by GET /state, put/create/edit/delete class, reassign_object, validate wipe. Writer: save_state. Reader-outside-lock: `GET /state`.
- `cache._cache['masks.json']` — read by GET /masks, GET /masks/versions, GET /masks/<frame>, propagate preflight (load_frame_masks_rle), recalculate. Writer: update_frame_masks (propagation persist_fn + click/box/text routes + recalculate). Reader-outside-lock: all four.
- `cache._cache['prompts.json']` — read by GET /prompts. Writer: save_prompt, delete_*_prompt. Reader-outside-lock: GET /prompts.

### 2.7 `GCSSyncManager` state
| Name | Type | Lock |
|---|---|---|
| `_dirty`, `_deferred`, `_propagating` | set, set, bool | **`_lock`** (except writes to bucket_name/session_id/session_dir — set once in `__init__`) |
| `_timer` | `threading.Timer \| None` | **None** — written by `_schedule` + `stop`, read by `stop` |
| `_interval` | float | **None** — written in `start`, read in `_schedule` |

### 2.8 Auth middleware `auth.py`
| Name | Lock |
|---|---|
| `_auth_cache` | per-token `_token_locks[key]` for writes; reads without lock (CPython GIL saves us) |
| `_token_locks` | **`_token_locks_guard`** |

### 2.9 Disk files under `sessions/<sid>/`
| File | Writers | Readers |
|---|---|---|
| `state.json` | save_state (under session_io_lock, via cache.save → atomic_json_dump), direct atomic_json_dump when cache=None, DownloadSessionStep (blob.download_to_filename — **non-atomic overwrite**) | `GET /state`, `GET /classes`, routes that RMW, `list_sessions` (local), `session_bundle.export_session_bundle` (reads outside lock) |
| `masks.json` | update_frame_masks, delete_*_mask, remove_object_masks (under session_io_lock), DownloadSessionStep (non-atomic) | `GET /masks*`, propagation preflight, export_coco (uses `load_masks` with NO cache), session_bundle export |
| `prompts.json` | save_prompt, delete_* (under session_io_lock), DownloadSessionStep | load_all_prompts from `replay_prompts_if_needed` (called from propagate without session_io_lock) and `ensure_active_object` (under `_lock` — reads disk outside session_io_lock) |
| `meta.json` | upload_video (direct open(w)), ExtractFramesStep (direct open(w)), DownloadSessionStep (overwrite non-atomically), import_session_bundle | resume_session, list_sessions (local), export routes, video.sessions, upload_video duplicate check |
| `frames/%05d.jpg` | ffmpeg subprocess, DownloadSessionStep, import_session_bundle | `GET /api/video/frame/<sid>/<idx>`, LazyFrameLoader / LazyProcessedFrames, export_coco, list_sessions |
| `video.mp4` | upload_video, DownloadSessionStep, import_session_bundle | ExtractFramesStep (ffmpeg read), list_sessions (size only) |

---

## Section 3 — Lock inventory

| Lock | Defined in | Protects | Hold duration | Callers |
|---|---|---|---|---|
| `_locks_guard` | `session_lock.py:18` | `_locks` dict lookup/insert | microseconds | `session_io_lock()` |
| per-session `session_io_lock(sid)` | `session_lock.py:17` | session JSON RMW cycles (state/masks/prompts cache ops + atomic_json_dump) | ms to tens of ms | every WRITE route in segment.py/session.py; also read-then-write routes. Per comment: never held across SAM3 calls. |
| `SAM3Service._lock` (RLock) | `sam3_service.py:242` | all SAM3 predictor access, `_sessions`, `_session_meta`, `_active_object`, `_text_sessions` | long: single `add_click` is ms; **propagation holds for minutes**. RLock → same thread can re-enter (pipeline thread → init_session does this intentionally) | every SAM3 API + `start_pipeline` acquires-then-releases to probe |
| `SAM3Service._state_lock` | `sam3_service.py:252` | `_service_state` frozen dataclass | microseconds | `get_service_state`, `start_pipeline`, `_run_pipeline`, `cancel_pipeline`, `dismiss_error`, `close_session` |
| `SAM3Service._propagation_lock` | `sam3_service.py:246` | gate `start_propagation` status-check-then-set | microseconds | only `start_propagation` — **readers don't hold it**, so the invariant is half-enforced |
| `GCSSyncManager._lock` | `gcs_sync.py:37` | `_dirty`, `_deferred`, `_propagating` | microseconds; **released during upload_file (which is slow + blocking)** | `mark_dirty`, `set_propagating`, `flush` (twice — snapshot + prune), `stop` does NOT take the lock |
| `_token_locks_guard` | `auth.py:36` | `_token_locks` dict | microseconds | `require_cloud_auth` |
| per-token `_token_locks[key]` | `auth.py:35` | serialize Dashboard API call per token | up to 5s (timeout) | `require_cloud_auth` |

Locks **NOT** in this codebase but probably should be:
- `_active_sync_manager` / `_active_session_cache` globals (plain attribute writes)
- `SessionCache._cache` internals — the whole point is "caller holds the lock", but half the callers don't
- `GCSSyncManager._timer` / `_interval` — the timer lifecycle is unsynchronized
- `_sessions_cache` in `gcs_storage.py`
- `_propagation_state`, `_propagation_subscribers`, `_cancel_events` dicts (after the single `start_propagation` gate)

---

## Section 4 — Race hunt

For each race below: category (`TOCTOU`, `read-during-mutate`, `lost-update`, `cross-thread-vis`, `deadlock`, `orphaned-resource`), severity, reproduction, file:line.

### R1 — Reader iterates cached JSON while writer mutates in place
**Status: FIXED in `221f3b0`** — `SessionCache.load()` now returns a shallow top-level copy (`session_cache.py:54-67`) and `update_frame_masks` clones nested dicts before mutating (`mask_storage.py:297-331`). Readers iterating the returned dict are no longer racing with writers. Nested values are still shared references, so future writers must maintain the clone-before-mutate discipline.
**Category:** read-during-mutate
**Severity:** CRITICAL — intermittent 500s (`dictionary changed size during iteration`) + silent data corruption (partial RLE decoded). The `flask.jsonify` serializer calls `json.dumps` which iterates the dict. So does `mask_storage.load_all_masks_rle` (the `for frame_str, obj_masks in encoded.items(): ... for obj_str, entry in obj_masks.items():`).

**Mutation source.** `mask_storage.update_frame_masks` (`mask_storage.py:279`) mutates the inner dict in place:
```py
encoded = cache.load(MASKS_FILENAME)      # returns live reference
...
if frame_key not in encoded: encoded[frame_key] = {}   # outer mutation
encoded[frame_key][str(obj_id)] = {...}                # inner mutation
encoded[_VERSIONS_KEY][frame_key] = ...                # outer mutation
cache.save(MASKS_FILENAME, encoded)                    # rebinds outer pointer
```
This mutation happens under `session_io_lock` (good). The problem: the **readers below don't take that lock**.

**Readers missing `session_io_lock`:**
- `session.py:21` `get_state` → `load_state(..., cache=get_session_cache())` → returns dict → `jsonify`. Writer: any PUT/POST in session.py (under lock) + `reassign_obj` + `create_class`.
- `session.py:72–74` `get_masks` → `load_all_masks_rle(..., cache=cache)` + `get_versions` → `jsonify`. Writer: every propagation persist_fn and every click/box/text route.
- `session.py:84` `get_mask_versions` → `get_versions(..., cache=cache)` → `jsonify`. Same writer.
- `session.py:95` `get_frame_masks` → `load_frame_masks_rle` → `jsonify`. Same writer.
- `session.py:106` `get_prompts` → `load_all_prompts` → `jsonify`. Writer: `save_prompt` (under lock) — but `delete_object_prompts` called from `segment.py:226` is under lock, also fine; the reader STILL races.
- `session.py:182` `get_classes` → `load_state` → returns `state["classes"]` reference → `jsonify`. Writer: any `create_class` / `delete_class`.
- `segment.py:146` (propagate preflight) — `load_frame_masks_rle(session_dir, start_frame, cache=get_session_cache())` runs outside `session_io_lock`. Reader races with persist_fn writer.
- `segment.py:235` `recalculate` → `load_prompt(..., cache=get_session_cache())` outside the session_io_lock.

**Reproduction.** Thread A is propagating: every frame it writes `masks.json` via `update_frame_masks` holding `session_io_lock(sid)`. Thread B fires `GET /api/session/masks/<sid>`. Load hits cache, returns the live inner dict; `jsonify` starts iterating. Thread A completes a frame (releasing the lock), then starts the next frame (acquiring the lock again) and mutates the inner dict `encoded[frame_key] = {}`. Thread B is mid-iteration of `encoded.items()` → `RuntimeError: dictionary changed size during iteration`. 500. This is observable under real load.

Even between persist calls, `cache.save(MASKS_FILENAME, encoded)` at `session_cache.py:50` assigns `self._cache[filename] = data` — this rebinds the outer pointer, but since Thread A built `encoded` by mutating the **same dict** it got from `load()`, this assignment is a no-op for the dict identity. The mutation happens live.

**Sub-race R1a — Cache inner-dict identity swap race.** If a second writer calls `cache.save` with a newly constructed dict (e.g., if `update_frame_masks` ever wrote `cache.save(MASKS_FILENAME, {})`), readers would iterate the old dict — benign. Today, `update_frame_masks` mutates-then-saves-the-same-ref, so readers iterate the live one. Bug either way.

### R2 — `set_sync_manager` / `set_session_cache` have no lock, and call `.stop()` racing writers
**Status: FIXED in `c64552d` (#55) + `d8ef2be` (#58)** — two-commit fix:
- `c64552d` (closes #55): `close_session` route adds retry + `.unsynced` marker write + 503 response with `unsynced_files` so transient GCS failures are no longer silently dropped (`segment.py:264-344`).
- `d8ef2be` (closes #58): adds `sm.promote_deferred()` calls to the teardown paths (close_session, SIGTERM, set_sync_manager) so deferred masks from an active propagation are moved into `_dirty` before the final `flush_with_retry`. Without this, `c64552d`'s retry/marker logic would have had nothing to flush when the tab closed during propagation.
Globals swap is separately serialised under `_globals_lock` — see R4.
Note: `c64552d` by itself predates `promote_deferred()`. Do not `git blame` the promote line to it.
**Category:** lost-update, orphaned-resource
**Severity:** HIGH — data loss possible (stop-during-write on propagation)
**File:** `config.py:39-43`, `segment.py:269-274`

`set_sync_manager(None)` calls the old `sm.stop()` unconditionally. `stop()` cancels the timer and synchronously `flush()`es. The flush iterates `_dirty` (under `_lock`) and does slow uploads with the lock released for each upload. Meanwhile Thread B could call `mark_dirty("prompts.json")` → it goes into `_dirty` — but `set_sync_manager(None)` has already detached the manager from the global. Further writes after `set_sync_manager(None)` return None from `get_sync_manager()` and don't mark dirty at all. The flush just completed, but the writes-in-flight between "flush snapshot" and "set_sync_manager(None)" are NOT in the flush, and the NEW dirty marks are dropped on the floor.

Also: `close_session` in `segment.py:259` does `sam.close_session(sid)` then `sm.stop()` → if the upload fails, the failed file stays in `_dirty`, but then we `set_sync_manager(None)` which forgets it. #55 confirmed.

**Reproduction:**
1. User on session A: propagation finishes, `set_propagating(False)` promotes deferred `masks.json` to dirty.
2. User on session A: tab close → `POST /api/segment/flush` (queued behind something) AND `POST /api/segment/close/A` arrive nearly simultaneously.
3. Thread X enters `close(A)`, calls `sam.close_session(A)`, then `sm.stop()` (flushing). Upload fails (transient 503). File stays in `_dirty` but we can't see it — `set_sync_manager(None)`.
4. Masks are lost.

### R3 — `GCSSyncManager._timer` double-schedule / use-after-stop
**Status: FIXED in `0024960`** — `_stopped: bool` added (`gcs_sync.py:60`); `stop()` sets `_stopped = True` + cancels timer under `_lock` (`gcs_sync.py:223-234`); `_schedule()` early-returns if `_stopped` (`gcs_sync.py:236-242`); `start()` refuses stopped managers (`gcs_sync.py:198-202`). Post-stop re-arm closed.
**Category:** lost-update, orphaned-resource
**Severity:** MEDIUM — zombie timer thread; can upload to wrong session if re-used
**File:** `gcs_sync.py:126-144`

```py
def stop(self) -> None:
    if self._timer is not None:
        self._timer.cancel()
        self._timer = None
    self.flush()

def _tick(self) -> None:
    try:
        self.flush()
    except Exception:
        ...
    finally:
        self._schedule()    # always re-schedules
```

Sequence:
1. Timer thread enters `_tick`. Main thread calls `stop()`, `self._timer.cancel()` is a no-op since the timer already fired. `_timer = None`.
2. Main thread's `stop()` blocks on `flush()`.
3. Timer thread's `_tick` is also in `flush()` — they both run in parallel. Both iterate `_dirty` (under `_lock`, so serialised), but one sees the dirty set, the other sees it empty.
4. Timer thread finishes `flush()`, enters `_schedule()` finally, creates a new `Timer`, assigns `self._timer = new_timer`. **The main thread already set `_timer = None`; it's now overwritten.**
5. `stop()` completes. Caller thinks the sync manager is dead. The new timer is alive forever.

On Cloud Run, the container lives across many sessions. Over time this leaks threads and, worse, sends flushes on a `session_dir` that has been rmtree'd, raising exceptions in an unmonitored daemon.

### R4 — `_active_session_cache` / `_active_sync_manager` lifecycle race
**Status: FIXED in `0024960` / `8c40dda`** — `_globals_lock` serialises swaps of both globals (`config.py:38, 44, 64-66, 99, 105`). `set_sync_manager` now calls `old.promote_deferred()` + `old.stop()` outside the lock, then writes an `.unsynced` marker if files remain dirty (`config.py:48-95`). `set_session_cache` guarded identically.
**Category:** cross-thread-vis, lost-update
**Severity:** HIGH — writes routed to wrong session
**File:** `config.py:39-53`, `routes/session.py:261-325`, `routes/segment.py:269-274`

`resume_session(B)` and `close(A)` (or two resumes for different sessions from two tabs) interleave:
1. Session A is active: `_active_session_cache` points at `/sessions/A`.
2. Thread 1: `POST /api/session/resume/B` — after pipeline accepted, calls `set_session_cache(SessionCache('/sessions/B'))`. Between `start_pipeline` returning ok and `set_session_cache`, there's a window.
3. Thread 2: `POST /api/segment/click` for session A arrives (frontend was still using session A). Handler reads `get_session_cache()` — depending on timing, gets either the A cache or the B cache. If B: `save_prompt('/sessions/A', ..., cache=B_cache)` writes the prompt into session A's dir (session_dir arg is A), but updates `B_cache._cache['prompts.json']` — so now session B's cache thinks it has A's prompts in memory.
4. On next `GET /prompts` for B, B's cache returns A's prompts.

Also: `set_sync_manager(new)` calls `.stop()` on the previous one. During that `.stop()`, the global is still the old manager. Concurrent writes in-flight that do `get_sync_manager().mark_dirty(...)` succeed against the dying manager. Once `.stop()` returns, those marks are flushed — but if stop's flush already snapshotted before mark_dirty happened, they're dropped.

### R5 — `_sessions_cache` thundering herd + race
**Status: FIXED (#62 / PR fix/issue-62-sessions-cache-single-flight).** Per-bucket single-flight lock now wraps the populate path; concurrent callers on a TTL boundary issue exactly one GCS list. See TL;DR entry for R5 / #62.
**Category:** perf + lost-update
**Severity:** LOW–MEDIUM — perf; potential cache-tuple write overlap
**File:** `gcs_storage.py:17, 28-87`

N concurrent `GET /api/video/sessions` hit `list_sessions`. All see expired cache, all do the expensive `list_blobs` twice plus per-session `state.json`/`meta.json` download. Meanwhile `invalidate_sessions_cache` can run between the two, so winner chosen at random. Benign data-wise, but N*several-seconds of GCS roundtrips.

### R6 — `init_session` not gated against pipeline
**Status: FIXED in `0024960`** — `init_session` now reads `_service_state` and raises if another session is `extracting` / `initializing` (`sam3_service.py:453-460`). The route translates this into a 409 response (`segment.py:29-32`).
**Category:** TOCTOU, cross-thread-vis
**Severity:** MEDIUM — 500 or inconsistent state
**File:** `segment.py:21-33`, `sam3_service.py:448-460`

The status endpoint exposes the pipeline phase; the frontend is supposed to wait for "ready" before calling `/segment/init`. But nothing server-side enforces it. If the frontend races — e.g., user navigates back then forward — and a pipeline is `extracting`, `POST /api/segment/init/<sid>` calls `sam.init_session` which acquires `_lock` and tries to walk the frames dir. If ffmpeg hasn't produced enough frames, `LazyFrameLoader` fails with "no images found" and throws 500. Worse: if it partially succeeds, `_sessions[sid]` gets a session with `num_frames` = count-at-that-moment, which won't grow. Then the pipeline's `InitSessionStep` hits the fast path at line 452 and returns the broken state.

### R7 — `_propagation_state` read/write outside `_propagation_lock`
**Status: FIXED (#63 / PR fix/issue-74-propagation-state-lifecycle-lock, bundled with R30).** `_propagation_lock` now guards every read and write of `_propagation_state`. `get_propagation_status` snapshots under the lock; the inner-loop `frames_processed` increment runs under the lock; `close_session`'s pop nests `_propagation_lock` inside `_lock`. See TL;DR entry for R7 / R30 / #63 / #74.
**Category:** cross-thread-vis, lost-update
**Severity:** MEDIUM — inconsistent status reported to UI; `close_session` during propagation still leaks events
**File:** `sam3_service.py:1104-1206`, `1312-1326`

- `start_propagation` takes `_propagation_lock` only to set the "running" state.
- `_run_propagation` later writes `self._propagation_state[session_id]["frames_processed"]` and `{"status": "failed"}` and `{"status": "idle"}` with NO lock.
- `close_session` pops `_propagation_state`, `_propagation_subscribers`, `_cancel_events` dicts without any lock — while the propagation thread is potentially mid-loop reading them. `close_session` does acquire `_lock` first, but the propagation thread holds `_lock` for the whole run; `cancel_propagation` is called BEFORE `_lock` — so the cancel is set, propagation's next iteration sees it, breaks, and runs its finally. Race: if `close_session` reaches the `pop` lines before the propagation finally, the propagation thread's finally does `self._propagation_state[session_id] = {"status": "idle"}` — but close_session already popped it. The bare assignment re-creates the key after close_session zeroed it. A subsequent `get_service_state` sees a stale session id. Benign (status only), but the leaked `_cancel_events` event object survives.

### R8 — `_propagation_subscribers` list mutation
**Status: FIXED in `a54619d` (PR #110, closes #64).** All reads/writes of `_propagation_subscribers` now serialised by `_propagation_lock` (the leaf lock that already guarded `_propagation_state` / `_propagation_threads`). Per-frame fan-out and the error/finally fan-outs in `_run_propagation` use a snapshot-then-iterate pattern: `list(...)` under the lock, then iterate the snapshot without the lock so `queue.put_nowait` / `_put_critical` are never called under any subscriber-list lock. `subscribe_propagation` does check + append atomically under the lock so the propagation finally cannot pop the list between them and orphan the queue (which would have blocked the SSE reader for 30s on every `q.get`); its finally removes under the same lock. `close_session` and the `_run_propagation` finally block consolidate `_propagation_subscribers.pop` into the same `_propagation_lock` critical section as `_propagation_state.pop` / `_propagation_threads.pop` (single fence, no extra lock acquires). Lock-declaration comment updated to record the broader scope. Tests in `test_sam3_service.py` (under `R8: _propagation_subscribers list mutation under _propagation_lock`) cover the no-session early-return, single-subscriber happy path, post-close subscribe, the finally-pops-subscribers invariant, and a 4-thread concurrent subscribe/unsubscribe churn against a 100-frame fan-out (would raise `RuntimeError: list changed size during iteration` on the unlocked code).
**Category:** read-during-mutate
**Severity:** LOW (was) — `RuntimeError: list changed size during iteration`
**File (historical):** `sam3_service.py:1176, 1189, 1198, 1257-1277`

Original finding (preserved for historical context): propagation thread iterated `for q in self._propagation_subscribers.get(session_id, []):` while subscriber routes appended / removed from the same list with no synchronisation. Python list append/remove are bytecode-atomic but concurrent list iteration + append/remove can raise `RuntimeError` in CPython. Practically rare since subscribers are bounded and each modifies at boundaries, but it was not safe by contract.

### R9 — Benchmark route reads `sam._model`/`sam._sessions` without lock
**Status: FIXED via #65** — `SAM3Service.debug_snapshot()` is the single accessor, taken under `_lock`. Benchmark route consumes its plain dict; no direct internal reads remain. See TL;DR entry above.
**Category:** cross-thread-vis
**Severity:** LOW — health endpoint returns partially-initialized model info
**File:** `benchmark.py:72-74, 82, 234, 236, 239-241`

`sam._model` is set by `_ensure_model` under `_lock`. The `GET /api/benchmark/health` reads it without lock. If `_ensure_model` is mid-way through (e.g., `from_pretrained` returned but `.to(device)` not yet), `health` may see a CPU model and `param_dtype` may be wrong. Not dangerous but confusing.

### R10 — SIGTERM handler flushes during ongoing writes
**Status: LARGELY FIXED in `8c40dda` + `d8ef2be` (#57)** — two-commit fix:
- `8c40dda`: handler now (1) cancels active propagation BEFORE touching the sync manager (`app/__init__.py:121-128`), (2) cancels the pipeline and joins with 3s timeout (`app/__init__.py:131-138`), (3) uses `flush_with_retry(max_attempts=3)` (`app/__init__.py:152`), (4) writes a `.unsynced` marker + logs on persistent failure (`app/__init__.py:153-156`), and (5) calls `sm.stop()` at the end (`app/__init__.py:157`) to cancel the timer.
- `d8ef2be` (closes #57): added `sm.promote_deferred()` before `flush_with_retry` at `app/__init__.py:151`. Without this, in-flight propagation masks sitting in `_deferred` would never be flushed on shutdown — the core bug #57 tracked.

The original audit's claim that `DownloadSessionStep` lacked cancel checks was wrong — four checks exist at `pipeline.py:186, 236, 247, 270`. The earlier framing that "SIGTERM doesn't call stop()" was also wrong; it does (line 157).

**Residual exposure (R31 below):** the window between `flush_with_retry` returning and `sm.stop()` finishing — and even briefly after `sm.stop()` returns — allows a concurrent worker thread to call `mark_dirty` on a manager whose timer is cancelled and whose flushes are over. `mark_dirty` does NOT check `_stopped`, so these writes enter `_dirty` and are silently lost when `sys.exit(0)` fires. Narrow but real. See R31.
**Category:** orphaned-resource, lost-update
**Severity:** HIGH (on shutdown only)
**File:** `app/__init__.py:115-162`, `gcs_sync.py:204-234`

On SIGTERM (Cloud Run preemption, deploy), the signal handler runs on the main thread. Worker threads are still processing requests. `sm.stop()` flushes with `_lock` held only during snapshot; during slow uploads, a worker thread can `mark_dirty("masks.json")` — this is covered by `_lock`, but the uploading timer iteration already snapshotted before the mark. The mark stays in `_dirty`. Then `set_sync_manager(None)`? No — SIGTERM handler doesn't call `set_sync_manager(None)`, it only calls `sm.stop()`. After `stop()` returns, the global still points at the manager with residual dirty files, but `raise SystemExit(0)` tears down the process. **Residual dirty files are lost.**

Also: `sam.cancel_pipeline()` then `sam._pipeline_thread.join(timeout=5)`. Pipeline checks `cancel_event.is_set()` in loops (ExtractFramesStep) but `DownloadSessionStep` does NOT — it has no `cancel_event.is_set()` check during `download_session`. A multi-GB session download ignores cancellation and the 5s join times out silently; Python then kills the container mid-download, leaving a partial local session dir. On restart, `_cleanup_partial_sessions` deletes sessions without `state.json` but keeps ones with `state.json` even if frames are partial — so a session with `state.json` (downloaded first) and half the frames sticks around. Next resume fails.

### R11 — Propagation persist_fn deadlock invariant is implicit
**Status: FIXED via #66** (test-only; no production code change). See the TL;DR entry above for the three tests in `test_sam3_service.py` that lock in the `SAM3._lock` -> `session_io_lock` order. The invariant ("routes must never call SAM3 while holding `session_io_lock`, and persist_fn must never call SAM3 at all") still lives as a WARNING comment at `segment.py:163-167`, but a regression that inverts the order will now trip `test_reverse_lock_order_deadlocks_as_documented` (0.5s timeout on `_lock.acquire` while `session_io_lock` is held on the opposite side). `text_segment` at `segment.py:76-114` correctly splits into "phase 1 SAM3 calls outside the lock" / "phase 2 file writes inside the lock", but a future maintainer who merges the two for atomicity will deadlock against propagation — and now has a failing test to tell them why.
**Category:** deadlock
**Severity:** CRITICAL-but-latent — cannot happen today, will happen the first time someone edits persist_fn
**File:** `segment.py:163-169`, comments on `session.py:120-122`, `session_lock.py:13`

Lock ordering: routes take `session_io_lock` first (mostly without SAM3 calls under it — `delete_mask` explicitly releases it before `sam.clear_frame_object`). Propagation holds `_lock` and calls persist_fn which takes `session_io_lock`. If ANY route ever holds `session_io_lock` and then calls a SAM3 method that takes `_lock`, deadlock. Today the codebase is clean; the comments are load-bearing. **There is no test that would catch a regression.**

There's a near-miss in `segment.py:69-109` (`text_segment`):
```py
for inst in result["instances"]:
    decoded = pmask_utils.decode(inst["rle"])
    obj_masks[inst["obj_id"]] = decoded
    sam.add_mask(session_id, frame_idx, inst["obj_id"], decoded)   # takes _lock
# ...
with session_io_lock(session_id):
    save_prompt(...)
    update_frame_masks(...)
```
Here SAM3 is called outside `session_io_lock` — safe. But it's invited: a change that moves `sam.add_mask` inside the `with session_io_lock` block (because logically it's part of the same atomic op) deadlocks immediately against propagation.

### R12 — `DownloadSessionStep` non-atomic overwrite
**Status: PARTIALLY FIXED in `8c40dda` + `23e8482`** — the resume-refresh path that races with live GET traffic now uses tmp + rename: `gcs_storage.download_file_if_exists` at `gcs_storage.py:176-199`, called from the refresh loop at `pipeline.py:267-285`. Cache is invalidated AFTER the atomic rename, so a concurrent `GET /state` either sees the old complete file or the new complete file — never a truncated one. Full `download_session` at `gcs_storage.py:144-164` is still non-atomic (direct `blob.download_to_filename`) but only runs when `session_dir` is absent or frames are missing — no concurrent reader exists in that branch.
**Category:** read-during-mutate, file-torn
**Severity:** HIGH — half-written JSON visible to readers
**File:** `pipeline.py:234-285`, `gcs_storage.py:144-164, 176-199`

`blob.download_to_filename(local_path)` writes directly to `local_path`, NOT to a tmp + rename. So while `DownloadSessionStep` refreshes `state.json` / `masks.json` / `prompts.json`, a concurrent `GET /state` (no session_io_lock!) can read a half-written file → `json.JSONDecodeError`. Even the refresh path that DOES invalidate the cache afterwards (`cache.invalidate`) can't help the reader that opened the file mid-download.

Reproduction: resume triggers pipeline; pipeline downloads state.json. In parallel, a heartbeat / tab visibility re-enter fires `GET /state` — open() returns a file handle to a truncated file, json.load raises → 500.

Also: `download_session` in cloud cold-start can overwrite `frames/NNNNN.jpg` files while Flask is mid `send_file` on them. send_file uses sendfile(2); file descriptor is held but truncate-rewrite races could send a mix of old and new bytes. (Cloud Run's filesystem is in-memory tmpfs, so no guarantees about atomic overwrites.)

### R13 — `_active_object.pop` outside `_lock`
**Status: FIXED in `0024960`** — every `_active_object.pop` is now inside the `_lock` block: `remove_object` at `sam3_service.py:1290/1302`, `reset_session` at `1305/1310`, `clear_frame_object` at `1507/1512`, `clear_frame_object_batch` at `1516/1521`.
**Category:** cross-thread-vis
**Severity:** LOW — Propagation thread holds `_lock`, meanwhile `remove_object` could modify `_active_object`. Next `ensure_active_object` may see stale or missing entry and redo replay.

### R14 — `SessionCache.save` call inside `session_io_lock` races disk-level readers outside the lock
**Status: FIXED via #67** — `export_coco_route` now wraps `load_state` + `load_masks` in `session_io_lock(session_id)` so the export snapshot is consistent with whichever writer was last to persist. `list_sessions` likewise holds `session_io_lock(name)` around each per-session `load_state` read. No concurrent writer exists on these paths today; this is defensive hardening so any future writer (e.g. a background GC, a new cleanup endpoint) cannot silently produce partial exports.
**Category:** file-torn, for callers bypassing cache
**Severity:** MEDIUM
**File:** `session_cache.py:69-78`, `routes/video.py:136`, `routes/export.py:22, 25`

`export_coco` calls `load_masks(session_dir)` with NO cache — goes straight to disk. During cloud mode, if propagation persist_fn is running (writes via cache.save → atomic_json_dump), the disk file is atomically replaced. `load_masks` open(path) vs rename race: POSIX rename is atomic relative to open, so the reader either sees the old or the new file. **But** if `session_io_lock` is NOT taken by export_coco, and persist_fn wrote an RLE that's internally consistent but a different frame_idx set than a racing reader expects, the exporter may miss frames that were just added — silent correctness, not a crash. Acceptable only if export is explicitly snapshot-at-start.

Also `video.list_sessions` (local mode, `video.py:136`): `load_state(session_dir)` without cache, during a pipeline's state write — same race, returns empty or old state for that session. Stamped onto the UI list.

### R15 — `_propagation_state[session_id]` assignment after cancel
**Status: FIXED (`56ea768`, PR #103, closes #68).** The propagation thread's `except` and `finally` blocks now take `_propagation_lock` and guard `_propagation_state[session_id]` mutation on the entry still being present. Clean runs `pop()` the entry instead of writing `{"status": "idle"}` (equivalent under `get_propagation_status`'s missing-key default). No more zombie entries for closed sessions.
**Category:** lost-update
**Severity:** LOW

At `sam3_service.py:1203-1204`, after an exception path, `if self._propagation_state.get(session_id, {}).get("status") != "failed": self._propagation_state[session_id] = {"status": "idle"}`. If `close_session` already popped the dict at line 1324, this bare assignment re-inserts `{"status": "idle"}` into a freshly-cleared structure. No-op in practice but conceptually wrong — and if `close_session` then tried to pop again, KeyError would be silent.

### R16 — `_pipeline_thread` attribute read in SIGTERM handler
**Status: STILL OPEN as of 2026-04-14 (non-issue under CPython GIL, no action).** Read now at `app/__init__.py:135`. Still outside any lock; still safe under CPython GIL.
**Category:** cross-thread-vis
**Severity:** LOW

`app/__init__.py:135` — `if sam._pipeline_thread is not None: sam._pipeline_thread.join(...)`. The read is outside any lock. CPython protects pointer swaps by the GIL, so you either see None or a valid Thread reference, but not partial. Benign.

### R17 — `_auth_cache` shared-write with different threads unlocking
**Status: STILL OPEN as of 2026-04-14 (non-issue, no action).** Logic unchanged; still correct.
**Category:** none (benign)
**File:** `auth.py:29, 82, 137`

The double-check with per-token lock is correct. Tuple assignment + dict set under GIL is atomic. No bug here. Noting so it doesn't get flagged falsely.

### R18 — `upload_video` writes `meta.json` with plain `open(w)` not atomic
**Status: STILL OPEN as of 2026-04-14 (benign, no action).** Upload is still single-threaded per session (fresh UUID); no concurrent reader.
**Category:** file-torn
**Severity:** LOW — upload is single-threaded per session (session_id is fresh UUID); duplicate-check uses a different one. Benign today.

### R19 — `_token_locks` unbounded growth
**Status: FIXED (#69) — see TL;DR entry for the bounded-LRU design and the `fix/issue-69-bounded-lru-lock-maps` commit. Both `auth._token_locks` and `session_lock._locks` are now `OrderedDict`s with a 1024-entry cap and per-access `move_to_end`/`popitem(last=False)`.**
**Category:** memory leak
**Severity:** LOW

Every new token creates a new `threading.Lock()` in `_token_locks` and `_auth_cache`. These are never pruned. Over a long-lived container, with rotating Firebase tokens, the dicts grow unbounded. Same issue in `session_lock._locks`: a lock per session id, forever.

### R20 — `replay_prompts_if_needed` + `ensure_active_object` read `prompts.json` disk without cache AND without session_io_lock
**Status: FIXED in `20d393e` (#70).** Both readers now pass `cache=get_session_cache()` to `load_all_prompts`, matching the writer path (`save_prompt` / `delete_*_prompt` all write through the same cache). Cache reads take `SessionCache._lock` (#4) for the shallow top-level copy — safely below `SAM3._lock` (#2) in the hierarchy — so we avoid `session_io_lock` (#3) entirely. This keeps rule #1 intact: no `session_io_lock` held across SAM3 predictor calls.
**Category:** file-torn
**Severity:** MEDIUM
**File:** `sam3_service.py:1041, 1122`

Historical: these called `load_all_prompts(session_dir)` with `cache=None` → direct `open(path)` + `json.load`. Inside `_lock` but not `session_io_lock`. Benign with current writers (atomic rename covers it), but fragile if a future writer forgot the lock. The cache route eliminates both the torn-read risk and the wasted disk I/O.

### R21 — `cancel_event` side-read race in propagation loop
**Status: STILL OPEN as of 2026-04-14 (non-issue, no action).** Pattern unchanged at `sam3_service.py:1141`; `close_session` pops `_cancel_events` at `1326`. Propagation thread still holds its local reference — no bug.
**Category:** cross-thread-vis
**Severity:** very low

`sam3_service.py:1141` `cancel_event = self._cancel_events.get(session_id)` at thread start. Line 1156 checks `cancel_event and cancel_event.is_set()`. `close_session` pops the dict at line 1326 — the propagation thread still holds the local reference. Good. No bug.

### R22 — `invalidate_sessions_cache` called BEFORE upload completes
**Status: FIXED (#71).** See the "Fixed" list at the top of this doc for the full write-up. Invalidation now runs from an `on_complete` callback on the pipeline, after every step (including GCS uploads) has succeeded.
**Category:** perf/consistency
**Severity:** LOW
**File:** `video.py` (`upload_video`)

### R23 — `GCSSyncManager.flush` release-reacquire window drops failed entries after set clear
**Status: STILL OPEN as of 2026-04-14 (analysis confirmed benign, no action).** Pattern preserved at `gcs_sync.py:117-160`. The only real risk — "user deletes a session mid-flush" — remains intentional behaviour.
**Category:** lost-update
**Severity:** MEDIUM
**File:** `gcs_sync.py:117-160`

```py
with self._lock:
    to_upload = self._dirty.copy()
# ... do uploads (slow, no lock)
if succeeded:
    with self._lock:
        self._dirty -= succeeded
```
Between "snapshot" and "prune", `mark_dirty("masks.json")` can add a NEW entry to `_dirty` for masks.json. That's fine — the new entry wasn't in `succeeded`, so it survives. But if `_propagating=True` is set between snapshot and prune, AND the new mark is for masks.json, it gets routed to `_deferred` instead of `_dirty`. Still fine (deferred → promoted on end). So actually benign. Keep.

BUT: the "succeeded" set includes files that "no longer exist on disk" (line 97 comment). If the user deletes a session mid-flush (rmtree), the flush silently succeeds and `_dirty -= succeeded` drops those. If the session was supposed to be deleted anyway, this is intended. OK.

### R24 — Cloud Run tmpfs + `os.rename` atomicity
**Status: STILL OPEN as of 2026-04-14 (non-issue, no action).** Guarantee unchanged.
**Category:** file-torn
**Severity:** informational

`atomic_json_dump` relies on POSIX rename atomicity. Cloud Run's tmpfs is in-memory but still POSIX-compliant; rename is atomic. No race here. Good.

### R25 — `ExtractFramesStep.run` reads `os.listdir(output_dir)` while ffmpeg writes
**Status: STILL OPEN as of 2026-04-14 (non-issue, no action).** Pattern unchanged.
**Category:** benign, file-visibility
**Severity:** very low

`os.listdir` is a snapshot; count changes per poll. Fine.

### R26 — `_auth_cache` cross-tenant if SHA256 collides
**Status: STILL OPEN as of 2026-04-14 (non-issue, no action).** Hashing strategy unchanged.
**Category:** non-issue
**Severity:** negligible

32-hex-char prefix of SHA256 gives 128 bits of space. Negligible collision risk. Note: comment says "JWT headers share the same base64 prefix across tokens from the same Firebase project — using token[:32] would cause cross-tenant cache collisions." Correct reasoning.

### R27 — `close_session` pops meta after releasing SAM3 lock
**Status: FIXED in `0024960`** — all pops now live inside the `_lock` block at `sam3_service.py:1317-1326`. No window remains where another thread can initialise `_sessions[sid]` between the pops.
**Category:** cross-thread-vis
**Severity:** LOW
**File:** `sam3_service.py:1317-1326`

Historically, the pops of `_session_meta`, `_active_object`, `_propagation_state`, `_propagation_subscribers`, `_cancel_events` ran AFTER releasing `_lock`. Another thread could call `sam.init_session(sid)` between the `_sessions.pop` (inside lock) and the `_session_meta.pop` (outside) and cause `_sessions[sid]` to exist while `_session_meta[sid]` was missing. Now all pops are under the same `_lock`.

### R28 — Multiple SAM3 calls in `replay_prompts_if_needed` — reentrant `_lock`
**Status: STILL OPEN as of 2026-04-14** — informational only, no action needed. `replay_prompts_if_needed` (`sam3_service.py:970-1035`) still holds `_lock` and invokes `self.add_click` / `self.add_box` / `self.add_mask` (at `1006, 1011, 1017`), relying on `_lock` being an `RLock`. Correct behaviour.
**Category:** correctness (not a race)
**Severity:** informational

`replay_prompts_if_needed` holds `_lock` and calls `self.add_click`, which also acquires `_lock`. Since `_lock` is `RLock`, this is fine. The inner `add_click` updates `_active_object` correctly. Good.

### R29 — `_cleanup_partial` called after `_service_state` reset
**Status: FIXED (#73).** `_cleanup_partial` now wraps the whole existence-check + rmtree block in `with session_io_lock(session_id):`. Concurrent writers on the same session's JSON files (state.json, masks.json, prompts.json) take the same lock on their RMW path and are serialised out of the cleanup window, so a late-arriving state.json write can no longer be blown away. See TL;DR entry for R29 / #73 above.
**Category:** TOCTOU
**Severity:** LOW
**File:** `sam3_service.py:1421, 1448, 1478-1499`

The pipeline's cancel path calls `_cleanup_partial(session_id)`, which rmtrees the session dir if no state.json and frames is empty/missing. But the check uses `os.path.exists` and `os.listdir` — between those and `shutil.rmtree`, another thread (a late-arriving `POST /api/session/state/<sid>` with content) could write a valid state.json. Then rmtree blows it away. Window is small; cancel path is rare. Still a real race.

### R30 — `_propagation_lock` guards only status check-then-set
**Status: FIXED (#74 / PR fix/issue-74-propagation-state-lifecycle-lock, bundled with R7).** `_propagation_lock` now covers the full `_propagation_state` lifecycle, not just the start-gate. See TL;DR entry for R7 / R30 / #63 / #74.
**Category:** race/inconsistency
**Severity:** LOW

Two concurrent `POST /api/segment/propagate` for the same session both reach `start_propagation`. One wins under `_propagation_lock`. The other ValueError-s. Fine. But if the FIRST is finishing up (inside the thread finally), and the SECOND calls `start_propagation`: the thread could set `_propagation_state[sid] = {"status": "idle"}` AFTER the second call has set `{"status": "running"}`. That's because the finally at line 1204 doesn't take `_propagation_lock` — it just does a bare assignment. Winner: the assignment that runs last. Possible symptom: user sees propagation flip from running to idle briefly and back.

### R31 — `mark_dirty` does not check `_stopped`; late writes enter a dead set
**Status: FIXED in `244ffb2` (B4, 2026-04-14)** — `mark_dirty` now raises `RuntimeError` if the manager is stopped. All 9 callers (mask/prompt/session storage) migrated to `mark_dirty_safe(rel_path)` which catches the exception and retries once via `get_sync_manager()` so a stop+swap race routes the write to the next-installed manager. `config.close_active_session()` pairs this with an atomic sync-manager + session-cache teardown so no observable `(stopped_sm, live_cache)` state exists. Combined with B2 (durable GCS marker + staging), all three loss windows below are closed. Kept here for historical context.
**Category:** post-stop write loss, narrow-window
**Severity:** LOW-MED — narrow timing but real loss path; mostly mitigated by `promote_deferred()` + `.unsynced` marker in the happy path, but those run before `stop()`, not after.
**File:** `gcs_sync.py:67-78`; race windows at `app/__init__.py:151-157` (SIGTERM) and `segment.py:319-341` + `config.py:48-95` (close/resume swap).

Concretely, three loss windows exist:

1. **SIGTERM race.** Between `flush_with_retry` returning (line 152) and `sm.stop()` completing (line 157), worker threads are still running. A request thread calling `mark_dirty("masks.json")` from inside `_persist_masks_from_result` enters `_dirty`. `sm.stop()` cancels the timer but does not re-flush. `sys.exit(0)` then fires. The late mark_dirty is lost with no marker.

2. **Close-session race.** `close_session` route path calls `sm.stop()` (segment.py) then eventually `set_sync_manager(None)` (config.py). Any concurrent request that read the global `_active_sync_manager` before the None swap holds a reference to the stopped manager. `mark_dirty` on it silently adds to a dead `_dirty`.

3. **Resume swap race.** In `set_sync_manager(new)`, the old manager is stopped *outside* the `_globals_lock` (`config.py:71-91` comment: "old manager's stop() runs outside lock to avoid starving readers"). During that window, concurrent requests that captured the old reference before the pointer swap can `mark_dirty` on a stopped manager. Even if the new manager was installed first, any thread that read the pointer earlier and held the reference hits the same dead path.

**Proposed fix (low-risk):** in `mark_dirty`, when `self._stopped` is True, one of:
- (a) raise `RuntimeError("manager is stopped")` and let callers catch & retry via `get_sync_manager()`;
- (b) write directly to `.unsynced` marker (append the filename to the marker JSON under the manager's `session_dir`), so the recovery path on next resume picks it up.

(a) is cleaner but requires audit of all `mark_dirty` call sites (mask_storage, prompt_storage, session_manager, segment routes) to add a single retry. (b) is self-contained and silently correct but depends on the marker being on GCS (see #76 / B2) for scale-to-zero recovery to work.

**Scope dependency:** (b) is only effective if the marker survives container termination (current marker is local-only — see #76 / B2). Fix ordering: address #76 first, then R31 via option (b) OR combine them into a single "durable marker + post-stop fallback write" change.

### R32 — SSE subscriber backpressure drops end-of-stream sentinel
**Status: FIXED — see "Fixed since initial filing" above.** Error events and the terminal sentinel now go through `_put_critical`, which drops-oldest-on-full to guarantee delivery. Original finding preserved below for context.
**Category:** lost-signal, backpressure
**Severity:** MEDIUM — subscriber never terminates cleanly when queue fills
**File:** `sam3_service.py:1176-1180, 1189-1193, 1198-1202`

Propagation subscriber queues are `queue.Queue(maxsize=50)` (`sam3_service.py:1258-1260` via `subscribe_propagation`). The propagation thread publishes via `put_nowait` at three sites — per-frame result (`sam3_service.py:1178`), terminal error (`sam3_service.py:1191`), and the end-of-stream sentinel `None` (`sam3_service.py:1200`). All three catch `queue.Full` and silently `pass`. If a slow subscriber (e.g., a backgrounded tab or a TCP-stalled SSE client) falls behind by >50 frames, the queue fills, and any of the three puts can be dropped.

The critical case is the sentinel: `get(timeout=30)` on the subscriber side is what drives SSE `event: done` framing. If the sentinel is dropped, the subscriber doesn't terminate cleanly — it hangs until the 30s read timeout, then the SSE stream closes abruptly without a `done` event, and the frontend can't distinguish "propagation finished" from "network glitch." Proposed fix: on `queue.Full` for the sentinel, drain one item with `get_nowait()` and retry, OR bump `maxsize` specifically for the sentinel path. See #85.

### R33 — ExtractFramesStep ffmpeg subprocess lifecycle on SIGTERM
**Status: FIXED** — PR closing #86. See TL;DR "Fixed since initial filing".
**Category:** orphaned-resource
**Severity:** LOW — cancel path correctly calls `terminate()`/`kill()`; residual risk is only the SIGTERM join timeout racing ffmpeg's 0.5s poll interval.
**File:** `backend/app/services/video_processor.py:82-117`, `backend/app/__init__.py:136`

The cancel-event path in `extract_frames_async` is correct: line 84-87 checks `cancel_event.is_set()` every `poll_interval` (default 0.5s) and `process.terminate()` → `process.wait(timeout=5)` → return 0. The except-path at 111-114 runs `process.kill()` + `process.wait(timeout=5)` on any other failure. No orphan in the steady-state cancel path.

Narrow residual: SIGTERM handler at `app/__init__.py:134-136` calls `sam.cancel_pipeline()` and then joins the pipeline thread with `timeout=3`. If ffmpeg is between polls (up to 0.5s between `cancel_event.is_set()` checks), and the `process.wait(timeout=5)` inside `extract_frames_async` has started, the pipeline thread's own 3s join budget can expire before ffmpeg responds. `sys.exit(0)` then fires and Cloud Run kills the container, leaving the ffmpeg child as an orphan that the kernel reaps on container death — no data corruption, just abrupt termination. Kept on the audit for completeness. See #86.

### R34 — LazyFrameLoader / frame reads race with ffmpeg/download writer
**Status: STILL OPEN as of 2026-04-14** — tracked in #87.
**Category:** read-during-mutate, file-torn
**Severity:** MEDIUM — truncated JPEG served to client
**File:** `backend/app/routes/video.py:191-199` (GET frame route, `send_file`); `backend/app/services/video_processor.py:73` (ffmpeg writes `%05d.jpg` direct); `backend/app/services/pipeline.py` / `gcs_storage.download_session` (DownloadSessionStep writes frames non-atomically).

`GET /api/video/frame/<sid>/<idx>` (`video.py:197-199`) checks `os.path.isfile(filepath)` and calls `send_file(filepath)`. ffmpeg writes frames directly to `os.path.join(output_dir, "%05d.jpg")` (`video_processor.py:73`) — no tmp+rename. DownloadSessionStep's `gcs_storage.download_session` at `gcs_storage.py:144-164` also uses `blob.download_to_filename(local_path)` directly (non-atomic).

Scenario: ExtractFramesStep or DownloadSessionStep is mid-write on frame N; the frontend (or preload queue) fires `GET /api/video/frame/<sid>/N`. `os.path.isfile` returns True as soon as the inode exists; `send_file` uses `sendfile(2)` over the open FD, so on Cloud Run's tmpfs a concurrent writer can extend or truncate the file while the kernel is streaming bytes. Result: client decodes a partial JPEG or an empty image. No crash — silent visual glitch that looks like a model bug. Fix: write to `<path>.tmp.<uuid>` then `os.rename` (same pattern as `gcs_storage.download_file_if_exists` from `23e8482`). See #87.

### R35 — SKIPPED (false positive; `_text_sessions` already popped in `close_session` at `sam3_service.py:1321`)

### R36 — SKIPPED (false positive; `start_pipeline`'s phase-check under `_state_lock` prevents concurrent A↔A resume double-start; the resume route's defer-cache-swap fix (`8331203`) closed the last gap)

### R37 — Cold-start request race with `_ensure_model`
**Status: FIXED — see "Fixed since initial filing" above.** Heartbeat logs added to `_ensure_model` via `SAM3Service._log_heartbeat_during`; operator-visible bookend + 5s interval tick cover the ~30s cold-load window. Original finding preserved below for context.
**Category:** user-visible blocking
**Severity:** LOW — no cryptic 500, just poor UX
**File:** `backend/app/services/sam3_service.py:257` (`_ensure_model`); `backend/app/routes/segment.py:21-33` (init_session call)

`_ensure_model` lazily loads the SAM3 model the first time it's called, under `_lock`. On a cold Cloud Run container (scale-from-zero or after a recycle), the first request that hits any SAM3 method blocks for ~30s while weights are transferred from the baked-in image cache to GPU, the backbone warms up, and the native predictor initialises.

The status endpoint doesn't surface "model loading" as a phase — the frontend just sees a slow request. During that ~30s window, other concurrent requests pile up on `_lock` and all appear hung. No crash, no 500, no progress feedback. Fix ideas: (a) kick off `_ensure_model` in a warmup thread at process boot (eager-init at import time), (b) add a `loading_model` phase to `ServiceState` and surface it to the frontend so the spinner has a reason, or (c) fire the model load from the pipeline itself before `InitSessionStep`. See #88.

### R38 — Single-worker invariant not asserted at startup
**Status: FIXED — see "Fixed since initial filing" above.** Runtime enforcement lives in `backend/gunicorn_config.py`; static check in `scripts/check_entrypoint_workers.sh`. Original finding preserved below for context.

**Category:** lifecycle guardrail
**Severity:** MEDIUM — silent corruption if someone changes the entrypoint
**File:** `deploy/entrypoint.sh:15` (`--workers 1`); no startup assertion anywhere in `app/__init__.py`.

Multiple in-memory singletons assume workers=1: `SAM3Service` (singleton via `__new__`), `_active_sync_manager`, `_active_session_cache`, `_sessions_cache`, `_auth_cache`, `_token_locks`, `_propagation_state`, `_propagation_subscribers`. Flipping gunicorn to `--workers 2` would create two disjoint in-memory state maps plus two copies of the GPU singleton — silent data divergence and GPU OOM on warmup.

The entrypoint is the only place enforcing `--workers 1`; nothing at runtime detects or rejects multi-worker configs. Proposed fix: in `create_app` (or at module import time), check `os.environ.get("GUNICORN_NUM_WORKERS")` or detect via `multiprocessing` / parent-PID tricks, and raise on startup if >1 worker. Low risk, high value — a single assertion blocks an entire class of silent breakage.

Side finding worth tracking separately (not part of R38): `--threads 4` is less than Cloud Run `--concurrency 8`. Up to 4 extra requests queue inside gunicorn on a busy container. If request latency is higher than expected under load, this is likely the reason. See #89.

### R39 — Lock hierarchy undocumented / unenforced across 7 process-wide locks
**Status: STILL OPEN as of 2026-04-14** — tracked in #90.
**Category:** latent deadlock
**Severity:** MEDIUM — any future maintainer can introduce a deadlock by acquiring in reversed order
**File:** `session_lock.py:17-18`, `sam3_service.py:242, 246, 252`, `gcs_sync.py:37`, `session_cache.py:32`, `config.py` (`_globals_lock`), `auth.py:35-36` (per-token + guard).

Inventory of process-wide locks currently live in this codebase:
1. `SAM3Service._lock` (RLock) — GPU + predictor + session dicts.
2. `SAM3Service._state_lock` — `_service_state` transitions.
3. `SAM3Service._propagation_lock` — `start_propagation` gate.
4. `session_io_lock(sid)` (per-session) + `_locks_guard` — JSON RMW.
5. `GCSSyncManager._lock` — dirty/deferred sets, timer lifecycle.
6. `SessionCache._lock` (RLock) — shallow cache snapshots.
7. `config._globals_lock` — `_active_sync_manager` / `_active_session_cache` swaps.
8. `auth._token_locks_guard` + per-token locks — auth serialisation.

Only two orderings are explicitly documented:
- `session_lock.py:11-14` — "never hold `session_io_lock` across a SAM3 predictor call" (propagation's persist_fn would deadlock against SAM3 `_lock`).
- `sam3_service.py:1358-1363` — `start_pipeline` acquires/releases `_lock` BEFORE entering `_state_lock` to avoid inverted ordering vs the progress-callback path.

There is no canonical lock-hierarchy doc, no runtime enforcement, and no lint. A maintainer who writes, say, `with session_io_lock(sid): sam.reset_session(...)` would deadlock immediately against an active propagation (that's the exact scenario R11 warns about — see also the near-miss in `segment.py` text_segment). Proposed fix: add a module-level `LOCK_HIERARCHY` comment block at the top of `sam3_service.py` enumerating the allowed ordering, AND a lightweight runtime assertion wrapper (thread-local list of held locks, assert ordering on acquire). See #90.

### R40 — Frontend beforeunload / flushSyncBeacon / /api/segment/flush handshake (frontend scope)
**Status: FIXED** — Frontend audit completed in `docs/FRONTEND_DATA_LOSS_AUDIT.md`. Master issue #121.
**Category:** out-of-scope placeholder → **audited**
**Severity:** HIGH (originally N/A — now assessed)
**File:** `frontend/src/api.ts:396-411` (`flushSyncBeacon`); `frontend/src/App.tsx:489-510` (unload handlers).

This backend audit was intentionally scoped server-side. The frontend audit (master issue #121) covers:
- **F1 (HIGH)** — `pagehide` handler for Safari/iOS tab close → **FIXED** PR #130
- **F3 (MED)** — SPA navigation flush → **FIXED** by F1 (PR #130)
- **F6 (MED)** — SSE reconnect missed frames → **FIXED** PR #129
- **F12 (HIGH)** — Cancel propagation before close → PR #128 (pending)
- See `docs/FRONTEND_DATA_LOSS_AUDIT.md` for full findings (F1–F12).

### R41 — `atomic_json_dump` tmp file orphan on SIGKILL/OOM
**Status: FIXED** — PR closing #92. See TL;DR "Fixed since initial filing".
**Category:** orphaned-resource
**Severity:** LOW — disk leak; no correctness impact
**File:** `backend/app/services/atomic_write.py:23-34`

`atomic_json_dump` writes to `<path>.tmp.<pid>.<tid>.<uuid>` and `os.rename`s (`atomic_write.py:26, 30`). The rename is atomic on POSIX; the exception path at 31-33 unlinks the tmp on any Python-level exception. However, if the process dies between `json.dump` (line 29) and `os.rename` (line 30) from SIGKILL (Cloud Run forced termination after grace period expires, OOM killer, segfault in a C extension), Python's `except` clause never runs and the tmp file is orphaned in the session directory.

Over a long-lived container with repeated scale-to-zero / crash cycles, `sessions/<sid>/` accumulates `.tmp.<pid>.<tid>.<uuid>` files that nothing cleans up. They don't corrupt the canonical files (rename never ran, so `state.json` / `masks.json` / `prompts.json` are intact), but they waste tmpfs space and clutter debug listings.

Fix: `sweep_orphan_tmp_files(root_dir, max_age_s=3600)` in `atomic_write.py` walks `root_dir` and unlinks `*.tmp.*` entries older than 1h. Invoked at startup (`_cleanup_partial_sessions`) and at resume (`DownloadSessionStep.run`, before annotation refresh). Covers the three tmp patterns produced by `atomic_json_dump`, `gcs_storage.download_file_if_exists`, and `unsynced_marker.write_marker`. See TL;DR entry for details.

---

## Section 5 — Recommendations (ranked)

Ordered by severity × blast radius.

### P0 — Fix readers (R1)

Every GET that returns cached JSON to the frontend must hold `session_io_lock(sid)` for the full duration of "load + serialize", OR we must give readers a snapshot they can serialize safely.

**Preferred:** make `SessionCache.load()` return a deep copy on cache hit (at least for the outer + first level of nesting), and keep writers mutating the live cached dict. Deep-copy cost is negligible compared to the JSON response itself. Code change is tiny:
```py
import copy
def load(self, filename):
    cached = self._cache.get(filename)
    if cached:
        return copy.deepcopy(cached)   # only callers under session_io_lock get the live one
    # ... existing disk load path
```
But then *writers* need the live reference. Add a `load_for_write(filename)` that returns the live dict and is only called under `session_io_lock`. Writer call sites: `update_frame_masks`, `save_prompt`, `delete_*`, `save_state`. Readers everywhere else use `load()`.

**Alternative:** take `session_io_lock` in every reader. Downsides: propagation persist_fn holds the lock for each frame; readers will block the propagation UI. Deep-copy is faster.

**Even simpler:** make readers call `json.dumps` inside the lock (using `jsonify` equivalent), then release. Not friendly with Flask's `jsonify`.

Recommendation: deep-copy on `load`, reserve `load_for_write` for writers. Cost ~microseconds for typical masks.json sizes.

### P0 — Atomic `DownloadSessionStep` overwrite (R12)

`gcs_storage.download_file_if_exists` should download to `path + '.tmp.<uuid>'` and `os.rename` at the end. Same for `download_session` when refreshing an existing dir. This is a 6-line change.

### P0 — `GCSSyncManager._timer` use-after-stop (R3)

Rework to a `stop_event` pattern:
```py
def stop(self):
    with self._lock:
        self._stopped = True
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
    self.flush()

def _tick(self):
    try:
        self.flush()
    finally:
        with self._lock:
            if not self._stopped:
                self._schedule()
```
And guard `_schedule` behind the same flag. Prevents the post-stop re-arm.

### P1 — Serialize `_active_session_cache` / `_active_sync_manager` (R2, R4)

Put a module-level `threading.Lock` in `config.py` (`_globals_lock`) and protect `get_sync_manager`/`set_sync_manager`/`get_session_cache`/`set_session_cache`. Then make setters transactional: grab lock, swap, release, THEN stop old manager outside the lock.

Better: make close/resume serialise via a session-lifecycle lock in `SAM3Service`. A single `_lifecycle_lock` protects the `(_sessions, _active_sync_manager, _active_session_cache, _service_state)` transition from A→B.

### P1 — Fix `close_session` → `sm.stop` dropped uploads (#55)

Change to:
```py
sm.cancel_timer()        # stop scheduling
retries = sm.flush_with_retry(max_attempts=3)
if sm.pending_count() > 0:
    logger.error("close: %d files still dirty for session %s", sm.pending_count(), sid)
```
Then decide: either fail the close (return 5xx and let the frontend retry), or intentionally abandon with an explicit log line and structured metric. Not a silent drop.

### P1 — Gate `init_session` against pipeline (R6)

In `init_session`, check `_service_state.phase`:
```py
with self._state_lock:
    state = self._service_state
if state.phase not in ("idle", "ready") and state.session_id == session_id:
    raise RuntimeError("Pipeline in progress for this session; wait for ready")
```
Or have `segment.init_sam` route redirect to the existing pipeline status.

### P2 — Lock `_propagation_state` dict (R7, R15, R30)

Either make `_propagation_lock` a proper RLock that guards ALL reads and writes of `_propagation_state`, or replace the dict with a per-session mini-object that has its own lock. The current pattern ("gate the start, free-for-all everywhere else") is error-prone.

### P2 — Single-flight `_sessions_cache` (R5)

Wrap the cache miss in a per-bucket lock so only one thread does the GCS list. Pattern matches the `_token_locks` approach in auth.

### P2 — Guard `_active_object` pops (R13)

Move every `pop` into the `_lock` block. Three-line change each.

### P2 — Fix `close_session` meta pops (R27)

Move all `_session_meta.pop`, `_active_object.pop`, `_propagation_*.pop`, `_cancel_events.pop` into the `_lock` block alongside `_sessions.pop`.

### P2 — `_handle_sigterm` cancels propagation too (R10)

Today it calls `sam.cancel_pipeline()` but not `sam.cancel_propagation(sid)` for each active session. Cloud Run's pre-stop hook is a 10s drain; if propagation is running, the container will be killed mid-frame, the ffmpeg subprocess + SAM3 GPU state are torn down abruptly. Cancel propagation first, wait briefly, then flush, then stop timer, then flush once more, then exit.

Also consider `DownloadSessionStep.run` should check `cancel_event.is_set()` periodically — wrap `download_session` with an `on_progress` callback that also probes cancel.

### P3 — Unbounded `_token_locks` and `session_lock._locks` growth (R19) — FIXED (#69)

Add TTL cleanup or LRU. Low priority unless long-running containers show memory creep. Closed by `fix/issue-69-bounded-lru-lock-maps` — bounded LRU (1024 cap) on both maps.

### P3 — Add a concurrency test that proves persist_fn never deadlocks (R11)

A pytest with a fake persist_fn that attempts `sam.remove_object(...)` inside — should fail-fast with a lock-ordering assertion (e.g., raise on re-entry). Or at minimum, a comment-level `# noqa: lock-order` linter that grep-s for `sam.` calls inside `with session_io_lock(` blocks.

### P3 — Don't call `invalidate_sessions_cache` before upload succeeded (R22) — FIXED (#71)

Moved the invalidation into an `on_complete` callback on `sam.start_pipeline` so it runs after every pipeline step has succeeded (including all GCS uploads) but before the `ready`-phase transition. Closed by `fix/issue-71-invalidate-cache-after-upload`.

### P3 — `export_coco` / `video.list_sessions` snapshot consistency (R14)

If propagation is running during export, export should either take `session_io_lock` once to snapshot the disk-level masks + state, or explicitly document that export is best-effort during live editing. Same for local `list_sessions` reading each session's state.json with no lock.

---

## Appendix — invariants as they stand today

Things that are NOT races:
- `SAM3Service._lock` protects the GPU + model initialization — correct use of RLock.
- `_service_state` is always mutated under `_state_lock` with `dataclasses.replace` — correct.
- `atomic_json_dump` with unique tmp path — correct after PR #50.
- `session_io_lock` covers writer-vs-writer on state.json/masks.json/prompts.json — correct after PR #54.
- Wipe-fingerprint rejection in `put_state` under `session_io_lock` — correct after PR #49.
- Per-token auth lock — correct.
- GCS client singleton is thread-safe.

Things that MAY be races but actually aren't, documented so they don't get "fixed":
- `cancel_event` captured in propagation thread (R21) — fine.
- `_auth_cache` dict writes (R17) — fine under GIL.
- POSIX rename atomicity on tmpfs (R24) — fine.
- `_pipeline_thread` pointer read (R16) — fine under GIL.

The rest: fix list above.
