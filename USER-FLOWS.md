# USER-FLOWS.md — Verification Harness

> Behavioral contract + regression checklist for the SAM3 Video Labelling Tool.
> When code and this document disagree, this document wins — or is explicitly
> amended in the same commit that changes the behavior.

---

## How to use this harness

The loop, after any behavioral change:

1. List the files you touched.
2. Look each up in the **Impact map** → collect the flow IDs and global invariants.
3. Execute each flow's **Verify** steps in order. Tags say who runs them:
   - `[API]` — Claude: curl against Flask (`http://localhost:5555`)
   - `[DISK]` — Claude: inspect files under `backend/sessions/<session_id>/`
     (`masks.json`, `state.json`, `prompts.json`) or exported zips
   - `[BROWSER]` — Claude drives a real browser against `http://localhost:5173`
     using the fixture's canonical click coordinates
   - `[HUMAN]` — only a person can judge (mask visual quality, drag feel)
4. Check each flow's Must NOT lines, including every global invariant they reference.
5. Report per flow: PASS / FAIL (with evidence) / NOT RUN (with reason).
   Hand the user the remaining `[HUMAN]` checklist.

**Degradation rule:** if no browser-automation tool is available in the
session, each `[BROWSER]` step is reported NOT RUN and appended to the
`[HUMAN]` checklist — never silently skipped.

**Poll budget:** Polling steps (e.g. waiting for `phase: "ready"`) poll every 1–2 s and FAIL after 5 minutes for extraction/init on the fixture video, or immediately if phase becomes `"error"` (unless the flow expects `error`).

**Environment:** local dev (`make dev`: Flask :5555 + Vite :5173, conda env
`sam3-annotator`, MPS). Cloud-mode-only paths (GCS sync, Cloud Run lifecycle)
cannot be fully verified locally — see "Cloud mode blind spots" in CLAUDE.md;
mark those steps NOT RUN locally and use the Deploy smoke set after deploys.

**Flow IDs** use the format `UF-<area>.<n>`:

| Range | Area |
|---|---|
| UF-1.x | Session lifecycle |
| UF-2.x | Class & object management |
| UF-3.x | Segmentation |
| UF-4.x | Propagation |
| UF-5.x | Mask deletion |
| UF-6.x | Frame navigation/playback |
| UF-7.x | Persistence & state sync |
| UF-8.x | Export/import |
| UF-9.x | Settings |
| UF-10.x | Canvas interactions |
| UF-11.x | Caching |
| UF-12 | Recovery (single entry, no sub-numbering) |

Flow IDs are stable and never renumbered (see Maintenance rules).

---

## Global negative invariants (N1–N10)

This document defines two kinds of negative checks: **global invariants** (N1–N10, this section) and **per-flow Must NOT lines** (which may reference global invariants by ID). Global invariants are stated once here, referenced by ID from individual flows.

- **N1** Propagation only touches targeted objects — propagating object(s) must not create, modify, or delete masks for any object not in the propagation set.
  _Check:_ byte-compare non-target objects' RLE strings in `masks.json` before/after propagation.

- **N2** Deleted masks stay deleted — disk deletion must also clear SAM3 in-memory state; propagation must never regenerate them.
  _Check:_ delete a mask, propagate across that frame, assert the mask is absent afterward.

- **N3** A correction click behaves identically to a first-time click — no residual inference state biases the result.
  _Check:_ delete the object's mask on frame F (which also clears its inference state), re-click the same coords, and compare against a fresh object clicked at the same coords → IoU ≥ 0.99 (via iou.py).

- **N4** UI state = backend state — SAM3 inference state contains only the active object(s); no stale objects accumulate.
  _Check (code review, not runtime):_ the only code paths that add objects to SAM3 inference state are the click/box/text prompt handlers and `reset_and_replay_objects` — all gated on the user's active selection. Verify no other call site registers objects, and that `remove_object`/`close_session` delete every bookkeeping entry for the removed IDs in `backend/app/services/sam3_service.py`.

- **N5** Auto-save never writes the "wipe fingerprint" (empty classes + objects with `class_id < 0`).
  _Check:_ inspect persisted `state.json` after any save — it must never contain `{"classes": [], "objects": [...with class_id < 0...]}` (the "wipe fingerprint").

- **N6** Close never silently loses data — debounced saves flushed before close; GCS sync failure surfaces the retry dialog, never a silent exit.
  _Check:_ close with a pending bboxPadding edit → reopen → padding survived; in cloud mode, a GCS failure during close must surface the retry dialog (cloud mode only — NOT RUN locally).

- **N7** Export contains exactly what's annotated — bbox padding matches `state.json`; no unannotated frames in COCO output.
  _Check:_ COCO `instances.json` bboxes equal mask bbox + padding from `state.json`; exported image count == annotated-frame count.

- **N8** `/api/status` phase always reflects reality — it drives all frontend top-level routing.
  _Check:_ at each checkpoint a flow designates, `GET /api/status` returns exactly the `phase` value that flow's Verify step specifies. (Flows are the source of expected values.)

- **N9** Cancel actually cancels — no orphaned propagation thread holding `SAM3Service._lock`.
  _Check:_ after cancel, `GET /api/segment/propagation-status/{session_id}` reports not-running, and a subsequent click responds in < 10 s on MPS (vs. minutes if the lock were still held).

- **N10** Multi-object reset+replay loses no prompts — every selected object's prompts are replayed from disk after `_reset_inference_state`.
  _Check:_ after multi-object propagation, every selected object has masks on the new frames; `prompts.json` byte-identical.

---

## Test fixture

```
tests/fixtures/harness/
├── sample.mp4            # 5–10 s, ≤640 px, 2–3 visually distinct objects
├── golden_session.zip    # known-good exported session (masks/prompts/state)
├── fixture.json          # canonical click coords per object on the keyframe,
│                         #   expected object count, IoU thresholds
├── iou.py                # canonical IoU: decodes two RLE masks with the backend's decoder, prints IoU
└── README.md             # how goldens were made, how to regenerate
```

Assets are committed in a follow-up commit; until then, fixture-dependent steps report NOT RUN.

**Comparison rules:**

- SAM3 output vs. golden masks: **IoU ≥ 0.80** (SAM3 is not bit-exact across MPS/CUDA backends).
- "Untouched data didn't change" checks (e.g. N1): **byte equality** of RLE strings.
- `golden_session.zip` doubles as the import-flow test asset (UF-8.2).
- All IoU values in this document are computed with `tests/fixtures/harness/iou.py` — the canonical method; until the fixture lands, IoU-dependent steps report NOT RUN.

---

## Tier 1 flows

### UF-1.1 Upload Video

**Contract:** Selecting a video, optional FPS (default 5), and optional max resolution (default 2048), then clicking Upload, starts a background pipeline that extracts frames and initializes SAM3, landing the user in the annotation UI without further interaction. If the tab is closed during extraction or initialization and reopened, the frontend re-attaches to the in-progress pipeline by polling `GET /api/status` and resumes showing the progress view at the correct phase.

**Must NOT:**
- Leave `GET /api/status` returning `phase: "extracting"` or `phase: "initializing"` after a pipeline failure — the phase must transition to `"error"` (N8) (induce: upload a non-video file — see UF-12 Verify).
- Create a session directory without `meta.json` — `meta.json` is written synchronously in `upload_video()` (`routes/video.py`) before `start_pipeline` is called.
- Accept a second concurrent upload while a pipeline is running — `start_pipeline` returns `(False, reason)` and the route responds 409 with `{"error": reason}`.
- Use a form field name other than `video` for the file and `fps` / `max_resolution` for the parameters (field names defined in `upload_video()` in `routes/video.py` and `uploadVideo` in `api.ts`).

**Verify:**
1. `[API]` `curl -s -X POST http://localhost:5555/api/video/upload -F "video=@tests/fixtures/harness/sample.mp4" -F "fps=5" -F "max_resolution=2048"` → Expect: HTTP 200, JSON body contains `session_id` (UUID string) and `duplicate: false`.
2. `[API]` Poll `GET http://localhost:5555/api/status` every 500 ms → Expect: `phase` walks `"extracting"` → `"initializing"` → `"ready"` in order; `progress` is non-decreasing within each phase; `session_id` matches the value from step 1.
3. `[DISK]` Inspect `backend/sessions/<session_id>/` → Expect: `meta.json` exists with `fps: 5`; `frames/` directory exists; `frames/*.jpg` count ≈ video duration (seconds) × 5 (±1 frame).
4. `[BROWSER]` Upload `tests/fixtures/harness/sample.mp4` via the UI file picker with default settings → Expect: progress bar advances through extraction and initialization steps; annotation UI (canvas + class panel) renders without a page reload.
5. `[HUMAN]` Inspect the canvas after upload → Expect: first frame renders full-width, undistorted, correct aspect ratio.

---

### UF-1.2 Resume Session

**Contract:** Clicking Resume on a session card calls `POST /api/session/resume/<session_id>`, which starts a background pipeline (local: `InitSessionStep` only; cloud: `DownloadSessionStep` then `InitSessionStep`), returns 202 immediately, and lands the user in the annotation UI with all prior classes, objects, masks, and prompts intact once the pipeline reaches `"ready"`. Resume also reconnects to any in-progress propagation (full coverage in UF-4.3).

**Must NOT:**
- Lose or mutate any persisted mask, prompt, or class data during resume — `masks.json`, `prompts.json`, and `state.json` must be byte-identical to their pre-resume snapshots (N5 applies: state.json must never contain the wipe fingerprint post-resume).
- Permit auto-save to fire before both state and mask loads have succeeded — `App.tsx`'s `sessionLoadedRef` guard gates all auto-saves; a resume that returns `"ready"` before the frontend has fetched state/masks must not trigger a save.
- Silently drop orphaned masks — objects whose class was deleted surface as `class_id: -1` synthetics in `App.tsx loadSession`; they must appear in the object list, not be silently omitted.
- Accept a second concurrent resume while a pipeline is already running — `start_pipeline` returns `(False, reason)` and the route responds 409.

**Verify:**
1. `[API]` Snapshot `masks.json`, `prompts.json`, and `state.json` for the golden session, then POST the resume endpoint: `curl -s -X POST http://localhost:5555/api/session/resume/<golden_session_id>` → Expect: HTTP 202, JSON body contains `session_id` and `video_name`.
2. `[API]` Poll `GET http://localhost:5555/api/status` → Expect: `phase` walks `"initializing"` → `"ready"`; `session_id` matches. Then `GET http://localhost:5555/api/session/state/<session_id>`, `GET http://localhost:5555/api/session/masks/<session_id>`, `GET http://localhost:5555/api/session/prompts/<session_id>` → Expect: values match the pre-resume snapshots.
3. `[BROWSER]` Click Resume on the golden session card → Expect: initialization progress view appears, then annotation UI renders; object list shows the fixture's expected object count; masks render as colored overlays on the keyframe.
4. `[DISK]` After resume completes, `diff` `masks.json` against the pre-resume snapshot → Expect: byte-identical.

---

### UF-1.5 Close Session (incl. GCS retry path)

**Contract:** Closing a session flushes any pending debounced saves (including bboxPadding edits), cancels any in-flight propagation, releases SAM3 inference state, and returns the user to the session list. In cloud mode, if GCS upload fails after 3 retries, the backend returns HTTP 503 with `{"error": ..., "unsynced_files": [...], "retry_possible": true/false}` and the frontend surfaces a retry dialog listing the unsynced files; "Retry upload" re-attempts the close; "Close anyway" (when `retry_possible: false`) closes without re-upload. Dismissing the dialog keeps the session open.

**Precondition:** golden session open (run UF-1.2 steps 1–2 first); use its `<session_id>`.

**Must NOT:**
- Lose a bboxPadding edit made less than 500 ms before close — `handleCloseSession` explicitly flushes state before clearing `sessionId`, bypassing the debounce timer (N6).
- Exit silently on GCS upload failure in cloud mode — must return HTTP 503 with `unsynced_files` array and surface the retry dialog (N6; cloud mode only — NOT RUN locally).
- Leave `GET /api/status` returning a phase other than `"idle"` after a successful close (N8).
- Leave a propagation thread holding `SAM3Service._lock` after close — `close_session` cancels propagation before acquiring the lock; after close, `GET /api/segment/propagation-status/<session_id>` must report `{"status": "idle"}` (N9).
- Crash or return a 5xx when closing a session whose SAM3 state was never initialized — `close_session` treats an absent session as a no-op (`self._sessions.pop(session_id, None)`) and the route returns 200.

**Verify:**
1. `[BROWSER]` Annotate a bbox on the keyframe, adjust the padding slider, then immediately click Close (within 500 ms of the padding change) → Resume the session → Expect: the padding value from before close is present and matches what was set.
2. `[API]` `curl -s -X POST http://localhost:5555/api/segment/close/<session_id>` → Expect: HTTP 200, `{"ok": true}`. Then `curl -s http://localhost:5555/api/status` → Expect: `{"phase": "idle", "session_id": null, ...}`. Also `GET /api/segment/propagation-status/<session_id>` → `{"status": "idle"}`.
3. `[API]` `POST /api/segment/close/<never-initialized-id>` → Expect: HTTP 200 (graceful no-op).
4. `[API]` After close, `curl -s -X POST http://localhost:5555/api/segment/click -H "Content-Type: application/json" -d '{"session_id":"<session_id>","frame_idx":0,"obj_id":1,"points":[[100,100]],"labels":[1]}'` → Expect: HTTP 500 (Flask unhandled `KeyError` from `self._sessions[session_id]` in `add_click`) — not a hang, responds within 10 s. Note: this is the current behavior; a future hardening task should return 404 instead.
5. `[HUMAN]` (cloud mode only) Close the session during a simulated or forced GCS outage → Expect: retry dialog appears listing the unsynced file names; "Retry upload" re-attempts and succeeds when GCS is restored; "Close anyway" closes and returns to the session list.

---

### UF-3.1 Click Segmentation + Undo

**Contract:** With a class or object selected, a left-click adds a positive point (label 1) and a right-click a negative point (label 0); each click accumulates with prior clicks on the same object/frame and the full accumulated list is re-submitted to `POST /api/segment/click`, which re-runs SAM3 and persists both the mask and the click prompt. Ctrl/Cmd+Z removes the last accumulated point and re-runs SAM3 with the remaining points; if no points remain, the mask and prompt are cleared in the frontend's React state only (the backend-persisted mask is not deleted — a "Missing Mask" panel with a "Calculate Mask" button appears in the sidebar if the saved prompt no longer has a corresponding rendered mask). If the selected class has multiple existing objects and no object is directly selected, an ObjectChoiceDialog asks which object to annotate (or to create a new one); if the class has no objects, one is auto-created.

**Precondition:** golden session open (run UF-1.2 steps 1–2 first); use its `<session_id>`. Fixture coordinates and IoU threshold come from `tests/fixtures/harness/fixture.json`; IoU computed via `tests/fixtures/harness/iou.py`.

**Must NOT:**
- A correction click after clearing behave differently from a first-time click — no residual inference state must bias the result (N3).
- A click on one object mutate another object's RLE in `masks.json` — byte-compare non-target entries before and after (N1 at click scope).
- A failed inference silently persist a prompt without a mask — the "Missing Mask" sidebar panel (visible when a prompt exists but no mask is rendered for the selected object on the current frame) must surface the discrepancy; it does not appear in the normal happy-path but is the designed affordance for this failure mode (code review, not runtime, for the normal path). `[BROWSER]`-verifiable: force an inference error to occur and confirm the panel appears (see Verify step 6).

**Verify:**
1. `[DISK]` Snapshot `masks.json` and `prompts.json` for the open session before any clicks.
2. `[API]` `curl -s -X POST http://localhost:5555/api/segment/click -H "Content-Type: application/json" -d '{"session_id":"<session_id>","frame_idx":0,"obj_id":<obj_1_id>,"points":[<fixture_click_xy_obj1>],"labels":[1]}'` → Expect: HTTP 200; response JSON has `frame_idx: 0`, `masks` key containing `"<obj_1_id>"`, and the RLE decodes to a non-zero binary mask (decode using pycocotools and confirm `mask.sum() > 0`); IoU vs golden keyframe mask ≥ 0.80 (NOT RUN until fixture lands).
3. `[DISK]` Inspect `prompts.json` → Expect: `prompts["0"]["<obj_1_id>"]` has `type: "click"`, `points` matching the submitted coordinates, `labels: [1]`. Inspect `masks.json` → Expect: `masks["0"]["<obj_1_id>"]` entry is present. All other objects' RLE strings are byte-identical to the pre-click snapshot.
4. `[API]` On a fresh object (`obj_id: <new_obj_id>` not previously annotated), POST the same click coordinates (first clear the previous object's inference state by posting a click for a different obj_id first, then clicking back) → Expect: IoU between the two results ≥ 0.99 (N3) (NOT RUN until fixture lands).
5. `[BROWSER]` With the Click tool active and the target class selected, click the fixture's canonical click coordinates for object 1 → Expect: a colored mask overlay appears on the canvas in the class color; a colored dot marks the click position; status bar reads "Ready".
6. `[BROWSER]` Press Ctrl+Z (or Cmd+Z) once → Expect: the click dot disappears; if other click points remain for this object on this frame, the mask updates to reflect the remaining points; if no points remain, the mask overlay disappears and — after selecting the object in the sidebar — the "Missing Mask" panel becomes visible in the annotation panel (because the prompt was cleared from local React state but the object still shows as a keyframe object).
7. `[HUMAN]` After a click, inspect the mask visually → Expect: mask tightly follows the object boundary, no spurious bleed into adjacent objects.

---

### UF-3.2 Box Segmentation

**Contract:** With the Box tool active and a class or object selected, dragging a rectangle larger than 5×5 pixels releases a box prompt: the frontend calls `POST /api/segment/box` with the tightest axis-aligned `[x1, y1, x2, y2]` in absolute pixel coordinates, SAM3 segments the dominant object inside the box, and the result is persisted to `masks.json` and `prompts.json` (type `"box"`). The same object-choice logic as click applies: if the selected class has multiple existing objects, an ObjectChoiceDialog fires; if none, one is auto-created. Boxes smaller than 5×5 px are silently discarded by the frontend before any API call; there is no backend guard for this.

**Precondition:** golden session open (run UF-1.2 steps 1–2 first); use its `<session_id>`. Fixture box coordinates come from `tests/fixtures/harness/fixture.json`.

**Must NOT:**
- A zero-area or near-zero box (≤ 5 px on either axis) reach the backend — the guard is in `VideoCanvas.tsx` `handleMouseUp` (`x2 - x1 > 5 && y2 - y1 > 5`); there is no additional backend guard. Verify that sending a zero-area box directly to the API does NOT trigger the guard (code review of frontend, not runtime). `[API]` Sending `{"box":[100,100,100,100],...}` directly bypasses the frontend guard and reaches SAM3 — this is expected behavior for the direct API; the harness documents it as a known gap.
- A box prompt overwrite another object's RLE in `masks.json` — byte-compare non-target entries before and after (N1 at click scope).
- The persisted prompt record a type other than `"box"` — in `prompts.json`, the entry under `[frame_idx][obj_id]` must have `type: "box"` and a `box` field; `save_prompt` in `prompt_storage.py` writes exactly what `segment.py`'s `box_segment` route passes: `{"type": "box", "box": data["box"]}`.

**Verify:**
1. `[DISK]` Snapshot `masks.json` and `prompts.json` before drawing any box.
2. `[API]` `curl -s -X POST http://localhost:5555/api/segment/box -H "Content-Type: application/json" -d '{"session_id":"<session_id>","frame_idx":0,"obj_id":<obj_2_id>,"box":[<fixture_box_obj2>]}'` → Expect: HTTP 200; response contains `frame_idx: 0` and `masks["<obj_2_id>"]` with a non-zero RLE; IoU vs golden ≥ 0.80 (NOT RUN until fixture lands).
3. `[DISK]` Inspect `prompts.json` → Expect: `prompts["0"]["<obj_2_id>"]` has `type: "box"` and `box` matching the submitted `[x1, y1, x2, y2]`. Inspect `masks.json` → Expect: `masks["0"]["<obj_2_id>"]` present. All other objects' RLE strings byte-identical to the pre-box snapshot.
4. `[BROWSER]` With the Box tool active, drag the fixture's canonical box coordinates for object 2 → Expect: a semi-transparent rectangle preview appears while dragging; on mouse-up, the preview disappears and a colored mask overlay appears; status reads "Ready".
5. `[BROWSER]` Drag a box smaller than 5×5 px (a near-stationary click) → Expect: no API call is made, no mask appears, no error in the status bar.

---

### UF-3.3 Text Detection

**Contract:** With the Detect tool active, entering a text query and submitting calls `POST /api/segment/text` with `{session_id, frame_idx, text, obj_id_start}`, where `obj_id_start` is computed by the frontend as `max(existing obj_ids) + 1`. The backend remaps its internal model IDs to sequential frontend IDs starting at `obj_id_start`. Each detected instance is assigned to the selected or auto-resolved/created class, registered in `state.json`, and its mask written to `masks.json`; the prompt type persisted in `prompts.json` is `"mask"` (RLE of the detected mask), not `"text"`. All detected instances are multi-selected after detection, with the PropagationBar showing one chip per instance. Pressing the toolbar's Cancel button (which triggers AbortController.abort()) cancels the in-flight fetch client-side; the backend request continues executing to completion on the server and its result is discarded when the aborted response arrives.

**Precondition:** golden session open (run UF-1.2 steps 1–2 first); use its `<session_id>`. `text_query` and `expected_text_detections` come from `tests/fixtures/harness/fixture.json`.

**Must NOT:**
- Detected instances overwrite or reuse existing `obj_id` values — `obj_id_start` is `max(all existing obj_ids) + 1`, computed in `handleDetect` in `App.tsx`; instances are assigned IDs `obj_id_start`, `obj_id_start + 1`, … sequentially. Verify: after detection the minimum new `obj_id` in `state.json` equals the pre-detection `max(obj_ids) + 1`.
- Detection mutate existing objects' RLE strings in `masks.json` — byte-compare all pre-existing mask entries before and after (N1).
- An aborted detection leave partially-registered objects or partial mask entries — because abort is purely client-side (the backend completes and writes to disk), a cancelled request WILL leave masks and state entries on disk if the backend finished before the abort signal propagated; the frontend discards the result and shows "Detection cancelled" in the status bar. This is the actual behavior: a client-side abort does not prevent the backend from completing. Annotators should be aware that a cancelled detection may still persist data; a future task should add server-side cancellation (code review, not runtime).

**Verify:**
1. `[DISK]` Snapshot `masks.json`, `prompts.json`, and `state.json` for the open session; record `max_obj_id_before = max(obj_ids in state.json)`.
2. `[API]` `curl -s -X POST http://localhost:5555/api/segment/text -H "Content-Type: application/json" -d '{"session_id":"<session_id>","frame_idx":0,"text":"<fixture_text_query>","obj_id_start":<max_obj_id_before+1>}'` → Expect: HTTP 200; `result.instances` length equals `fixture.json`'s `expected_text_detections`; each instance has `obj_id` in the range `[obj_id_start, obj_id_start + N - 1]`, a non-empty RLE, and a non-zero `area` (NOT RUN until fixture lands).
3. `[DISK]` Inspect `state.json` → Expect: `objects` array contains `N` new entries with `obj_id` values starting at `max_obj_id_before + 1` and the correct `class_id`. Inspect `prompts.json` → Expect: each new `obj_id` has a prompt entry with `type: "mask"` and an `rle` field. Inspect `masks.json` → Expect: all pre-existing RLE strings byte-identical to the pre-detection snapshot.
4. `[BROWSER]` With the Detect tool active, enter the fixture's `text_query` and press Detect → Expect: the status bar shows `Found N "<text_query>" instance(s) — all selected, ready to propagate` where N equals `fixture.json`'s `expected_text_detections`; N colored chips appear in the PropagationBar; the canvas shows N mask overlays in the class color.
5. `[BROWSER]` Click Cancel (or press Escape) during a detect request → Expect: status bar shows "Detection cancelled" within 10 s of cancellation; no new objects appear in the object list in the frontend (the frontend discards the result); note that backend may have completed and written to disk — verify by checking `state.json` for any new entries after cancel (acceptable behavior, documented limitation above).

---

## Tier 2 flows

<!-- populated in Tasks 2–6 of docs/superpowers/plans/2026-06-10-user-flows-harness.md -->

---

## Impact map

<!-- populated in Tasks 2–6 of docs/superpowers/plans/2026-06-10-user-flows-harness.md -->

---

## Deploy smoke set

<!-- populated in Tasks 2–6 of docs/superpowers/plans/2026-06-10-user-flows-harness.md -->

---

## Maintenance rules

1. A PR that changes user-facing behavior must update the affected flow entry in the same commit.
2. Every postmortem'd bug adds a Must NOT line to the relevant flow (or a global invariant) — the regression → invariant pipeline.
3. Flow IDs are stable; retired flows are marked deprecated, never deleted or renumbered.
4. The impact map gains a row whenever a new code area is created.
5. New flows get the next unused number in their range; new ranges append.
