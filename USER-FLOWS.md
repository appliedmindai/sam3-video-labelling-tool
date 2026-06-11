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
│                         #   expected object count, IoU thresholds,
│                         #   and the golden session's object-ID mapping (obj_id per fixture object)
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

**Contract:** With a class or object selected, a left-click adds a positive point (label 1) and a right-click a negative point (label 0); each click accumulates with prior clicks on the same object/frame and the full accumulated list is re-submitted to `POST /api/segment/click`, which re-runs SAM3 and persists both the mask and the click prompt. Ctrl/Cmd+Z removes the last accumulated point and re-runs SAM3 with the remaining points; if no points remain, both the mask and the prompt are cleared from the frontend's React state only — `masks.json` and `prompts.json` on disk are untouched, and the "Missing Mask" panel does NOT appear (its condition requires a prompt present in React state with the mask absent; undo-to-zero clears both). Residual note: after undo-to-zero, resuming the session reloads the undone mask from disk (UI state diverges from disk state until a new prompt overwrites it — see N4). If the selected class already has one or more objects and no object is directly selected, an ObjectChoiceDialog asks which object to refine (or to create a new one); a class with no objects auto-creates one.

**Precondition:** golden session open (run UF-1.2 steps 1–2 first); use its `<session_id>`. Fixture coordinates and IoU threshold come from `tests/fixtures/harness/fixture.json`; IoU computed via `tests/fixtures/harness/iou.py`; object IDs from `fixture.json`.

**Must NOT:**
- Treat a correction click differently from a first-time click — no residual inference state must bias the result (N3).
- Mutate another object's RLE in `masks.json` when segmenting a given object — byte-compare non-target entries before and after (N1 at click scope).
- Silently persist a prompt without a mask on a failed inference — the "Missing Mask" sidebar panel (visible when a prompt exists in React state but no mask is rendered for the selected object on the current frame, e.g. after a failed inference or a page reload mid-edit that wrote the prompt to disk before the mask was saved) must surface the discrepancy; it does not appear in the normal happy-path or after undo-to-zero (which clears both). `[BROWSER]`-verifiable: force an inference error to occur and confirm the panel appears (induce: POST a click with an out-of-range `frame_idx` — or stop Flask mid-click for the UI path).

**Verify:**
1. `[DISK]` Snapshot `masks.json` and `prompts.json` for the open session before any clicks. → Expect: snapshots saved.
2. `[API]` `curl -s -X POST http://localhost:5555/api/segment/click -H "Content-Type: application/json" -d '{"session_id":"<session_id>","frame_idx":0,"obj_id":<obj_1_id>,"points":[<fixture_click_xy_obj1>],"labels":[1]}'` → Expect: HTTP 200; response JSON has `frame_idx: 0`, `masks` key containing `"<obj_1_id>"`, and the RLE decodes to a non-empty binary mask (decode with `tests/fixtures/harness/iou.py`'s decoder — the backend's RLE format: LEB128 delta, column-major — and confirm the mask is non-empty); IoU vs golden keyframe mask ≥ 0.80 (NOT RUN until fixture lands).
3. `[DISK]` Inspect `prompts.json` → Expect: `prompts["0"]["<obj_1_id>"]` has `type: "click"`, `points` matching the submitted coordinates, `labels: [1]`. Inspect `masks.json` → Expect: `masks["0"]["<obj_1_id>"]` entry is present. All other objects' RLE strings are byte-identical to the pre-click snapshot.
4. `[API]` DELETE the mask for obj 1 on the keyframe (`DELETE /api/session/masks/<session_id>/0/<obj_1_id>`), re-POST the step-2 click for obj 1, and compare the returned RLE against the step-2 response → Expect: IoU ≥ 0.99 via `iou.py` (N3) (NOT RUN until fixture lands).
5. `[BROWSER]` With the Click tool active and the target class selected, click the fixture's canonical click coordinates for object 1 → Expect: a colored mask overlay appears on the canvas in the class color; a colored dot marks the click position; status bar reads "Ready".
6. `[BROWSER]` Press Ctrl+Z (or Cmd+Z) once → Expect: the click dot disappears; the point dot and mask overlay both disappear from the canvas (step 5 made exactly one click, so no points remain), the "Missing Mask" panel does NOT appear (both prompt and mask were cleared from React state), and `masks.json` on disk still contains the undone mask.
7. `[HUMAN]` After a click, inspect the mask visually → Expect: mask tightly follows the object boundary, no spurious bleed into adjacent objects.

---

### UF-3.2 Box Segmentation

**Contract:** With the Box tool active and a class or object selected, dragging a rectangle larger than 5×5 pixels releases a box prompt: the frontend calls `POST /api/segment/box` with the tightest axis-aligned `[x1, y1, x2, y2]` in absolute pixel coordinates, SAM3 segments the dominant object inside the box, and the result is persisted to `masks.json` and `prompts.json` (type `"box"`). The same object-choice logic as click applies: if the selected class already has one or more objects and no object is directly selected, an ObjectChoiceDialog asks which object to refine (or to create a new one); a class with no objects auto-creates one. Boxes smaller than 5×5 px are silently discarded by the frontend before any API call; there is no backend guard for this.

**Precondition:** golden session open (run UF-1.2 steps 1–2 first); use its `<session_id>`. Fixture box coordinates come from `tests/fixtures/harness/fixture.json`; object IDs from `fixture.json`.

**Must NOT:**
- Allow a zero-area box drawn in the UI to reach the backend — guarded by the >5px check in `VideoCanvas.tsx` `handleMouseUp` (`x2 - x1 > 5 && y2 - y1 > 5`); direct API calls bypass this guard (known gap, no backend validation). `[API]` Sending `{"box":[100,100,100,100],...}` directly bypasses the frontend guard and reaches SAM3 — this is expected behavior for the direct API; the harness documents it as a known gap.
- Mutate another object's RLE in `masks.json` when segmenting a given object via box — byte-compare non-target entries before and after (N1 at box scope).
- Persist a type other than `"box"` in the prompt record — in `prompts.json`, the entry under `[frame_idx][obj_id]` must have `type: "box"` and a `box` field; `save_prompt` in `prompt_storage.py` writes exactly what `segment.py`'s `box_segment` route passes: `{"type": "box", "box": data["box"]}`.

**Verify:**
1. `[DISK]` Snapshot `masks.json` and `prompts.json` before drawing any box. → Expect: snapshots saved.
2. `[API]` `curl -s -X POST http://localhost:5555/api/segment/box -H "Content-Type: application/json" -d '{"session_id":"<session_id>","frame_idx":0,"obj_id":<obj_2_id>,"box":[<fixture_box_obj2>]}'` → Expect: HTTP 200; response contains `frame_idx: 0` and `masks["<obj_2_id>"]` with a non-zero RLE; IoU vs golden ≥ 0.80 (NOT RUN until fixture lands).
3. `[DISK]` Inspect `prompts.json` → Expect: `prompts["0"]["<obj_2_id>"]` has `type: "box"` and `box` matching the submitted `[x1, y1, x2, y2]`. Inspect `masks.json` → Expect: `masks["0"]["<obj_2_id>"]` present. All other objects' RLE strings byte-identical to the pre-box snapshot.
4. `[BROWSER]` With the Box tool active, drag the fixture's canonical box coordinates for object 2 → Expect: a semi-transparent rectangle preview appears while dragging; on mouse-up, the preview disappears and a colored mask overlay appears; status reads "Ready".
5. `[BROWSER]` Drag a box smaller than 5×5 px (a near-stationary click) → Expect: no API call is made, no mask appears, no error in the status bar → Expect: `prompts.json` unchanged (no new box prompt) — observe via re-diff against the step-1 snapshot.

---

### UF-3.3 Text Detection

**Contract:** With the Detect tool active, entering a text query and submitting calls `POST /api/segment/text` with `{session_id, frame_idx, text, obj_id_start}`, where `obj_id_start` is computed by the frontend as `max(existing obj_ids) + 1`. The backend remaps its internal model IDs to sequential frontend IDs starting at `obj_id_start`, writing only `prompts.json` and `masks.json`; it does NOT write `state.json`. Object registration in `state.json` happens exclusively via the frontend's `persistState` PUT call inside `handleDetect` in `App.tsx` after the API response is received. The prompt type persisted in `prompts.json` is `"mask"` (RLE of the detected mask), not `"text"`. All detected instances are multi-selected after detection, with the PropagationBar showing one chip per instance. Pressing the toolbar's Cancel button triggers `AbortController.abort()`, which cancels the in-flight fetch client-side only — the backend request continues executing to completion on the server and its result is discarded when the aborted response arrives; a cancelled detection may therefore still persist masks and state entries on disk if the backend finished before the abort propagated (known limitation; a future task should add server-side cancellation).

**Precondition:** golden session open (run UF-1.2 steps 1–2 first); use its `<session_id>`. `text_query` and `expected_text_detections` come from `tests/fixtures/harness/fixture.json`; object IDs from `fixture.json`.

**Must NOT:**
- Detected instances overwrite or reuse existing `obj_id` values — `obj_id_start` is `max(all existing obj_ids) + 1`, computed in `handleDetect` in `App.tsx`; instances are assigned IDs `obj_id_start`, `obj_id_start + 1`, … sequentially. Verify: after detection the minimum new `obj_id` in `state.json` (written by the frontend's `persistState`, not the backend) equals the pre-detection `max(obj_ids) + 1`.
- Detection mutate existing objects' RLE strings in `masks.json` — byte-compare all pre-existing mask entries before and after (N1).
- Guarantee that an aborted detection leaves no disk state — because abort is client-side only (see Contract), a cancelled request will leave masks and state entries on disk if the backend finished before the abort propagated; the frontend discards the result and shows "Detection cancelled" in the status bar (code review, not runtime).

**Verify:**
1. `[DISK]` Snapshot `masks.json`, `prompts.json`, and `state.json` for the open session; record `max_obj_id_before = max(obj_ids in state.json)`. → Expect: snapshots saved.
2. `[API]` `curl -s -X POST http://localhost:5555/api/segment/text -H "Content-Type: application/json" -d '{"session_id":"<session_id>","frame_idx":0,"text":"<fixture_text_query>","obj_id_start":<max_obj_id_before+1>}'` → Expect: HTTP 200; `result.instances` length equals `fixture.json`'s `expected_text_detections`; each instance has `obj_id` in the range `[obj_id_start, obj_id_start + N - 1]`, a non-empty RLE, and a non-zero `area` (NOT RUN until fixture lands).
3. `[DISK]` Inspect `state.json` → Expect: UNCHANGED from the pre-detection snapshot (the backend pure-API call does not write state.json). Inspect `prompts.json` → Expect: each new `obj_id` (starting at `max_obj_id_before + 1`) has a prompt entry with `type: "mask"` and an `rle` field. Inspect `masks.json` → Expect: N new entries for the detected obj_ids; all pre-existing RLE strings byte-identical to the pre-detection snapshot.
4. `[BROWSER]` With the Detect tool active, enter the fixture's `text_query` and press Detect → Expect: the status bar shows `Found N "<text_query>" instance(s) — all selected, ready to propagate` where N equals `fixture.json`'s `expected_text_detections`; N colored chips appear in the PropagationBar; the canvas shows N mask overlays in the class color; `state.json` on disk now contains N new object entries with `obj_id` values starting at `max_obj_id_before + 1` and the correct `class_id` (written by the frontend's `persistState` call after the API response).
5. `[BROWSER]` Click the toolbar's Cancel button immediately after starting Detect (detection takes several seconds on the fixture; if it completes first, re-run) → Expect: status bar shows "Detection cancelled" within 10 s of cancellation; no new objects appear in the object list in the frontend (the frontend discards the result); note that the backend may have completed and written to disk — verify by checking `state.json` for any new entries after cancel (acceptable behavior — see Contract). Note: Escape is wired only to multi-select clearing, not to detection cancellation.

---

### UF-4.1 Single-Object Propagation

**Contract:** With one object selected and a mask on the current frame, clicking Back, Both, or Forward posts `POST /api/segment/propagate`, which starts a background thread holding `SAM3Service._lock` and streams per-frame masks over SSE. Each SSE event is a `data: <JSON>\n\n` line whose JSON is either a `FrameResult` (`{frame_idx, masks, source_keyframe?}`) or the sentinel `{"done": true}`. The frontend's `consumeSseFrames` in `api.ts` invokes `onFrame` for each `FrameResult` and returns when the sentinel arrives. Frames where any mask's `confidence` field is less than 0.75 (the `CONFIDENCE_THRESHOLD` constant in `App.tsx`) are flagged with an amber timeline tick and accumulate in the `confidenceWarnings` state; a banner with frame count and a "Go to frame" shortcut appears after propagation completes. The `source_keyframe` field on each persisted mask in `masks.json` records the user-authored keyframe that initiated the propagation run.

**Precondition:** Golden session open (run UF-1.2 steps 1–2 first); use its `<session_id>`. Object 1 must have a click or box prompt on frame 0 (run UF-3.1 or UF-3.2 step 2 first). Object 2 must also have a mask on frame 0 (run UF-3.2 step 2 for `<obj_2_id>` to establish the non-target baseline). Fixture values from `tests/fixtures/harness/fixture.json`; IoU via `tests/fixtures/harness/iou.py`.

**Must NOT:**
- Create, modify, or delete masks for any non-target object (N1 — headline invariant): byte-compare all non-target objects' RLE strings in `masks.json` before and after propagation. Induce: propagate object 1 forward while object 2 has a mask on frame 0; diff the before/after snapshots for object 2's entries — any change is a failure.
- Regenerate a previously deleted mask (N2): delete the keyframe mask of object 2 (`DELETE /api/session/masks/<session_id>/0/<obj_2_id>`), then propagate object 1; inspect `masks.json` — object 2's frame 0 entry must remain absent after propagation.
- Leave `GET /api/segment/propagation-status/<session_id>` reporting `status: "running"` after the SSE stream delivers `{"done": true}` (N8): the propagation finally block pops the entry under `_propagation_lock` so `get_propagation_status` returns `{"status": "idle"}` (missing key → idle default).
- Lose the `source_keyframe` metadata on propagated masks: each propagated frame's mask entry in `masks.json` must have a `source_keyframe` key equal to the integer frame index of the user's keyframe (not `null`).

**Verify:**
1. `[DISK]` Snapshot `masks.json` (save as `masks_before.json`). Record the RLE strings for all non-target object entries. → Expect: snapshot saved.
2. `[API]` `curl -s -X POST http://localhost:5555/api/segment/propagate -H "Content-Type: application/json" -d '{"session_id":"<session_id>","start_frame_idx":0,"reverse":false,"object_ids":[<obj_1_id>]}'` and consume the SSE response: for each `data:` line parse JSON; collect all `FrameResult` events until `{"done": true}`. → Expect: HTTP 200 with `Content-Type: text/event-stream`; every `FrameResult` has `frame_idx` in `[1, num_frames-1]` and `masks` containing only the key `"<obj_1_id>"`; `{"done": true}` arrives as the final event.
3. `[API]` `curl -s http://localhost:5555/api/segment/propagation-status/<session_id>` → Expect: `{"status": "idle"}` (N8 checkpoint).
4. `[DISK]` Inspect `masks.json` → Expect: every propagated frame entry for `<obj_1_id>` has `source_keyframe: 0` (or the fixture's keyframe index, whichever was used); all non-target object RLE strings are byte-identical to the `masks_before.json` snapshot (N1). If object 2's frame 0 entry was deleted in the Precondition, confirm it is absent (N2) (NOT RUN until fixture lands — depends on prior delete step).
5. `[API]` For each frame index in `fixture.json`'s `propagation_sample_frames`, fetch `GET http://localhost:5555/api/session/masks/<session_id>/<frame_idx>` and compare `masks["<obj_1_id>"].rle` against the corresponding golden mask using `tests/fixtures/harness/iou.py` → Expect: IoU ≥ 0.80 for every sampled frame (NOT RUN until fixture lands).
6. `[BROWSER]` In the annotation UI, select object 1 and click Forward in the PropagationBar → Expect: green timeline ticks appear frame-by-frame as propagation streams; the canvas advances to each completed frame live; amber ticks appear on frames where confidence < 0.75; after completion the status bar reports the count of low-confidence frames (or "Propagation complete" if none); a "Go to frame N" shortcut is visible in the warning banner for each amber-ticked frame.
7. `[HUMAN]` Spot-check frames at approximately 25%, 50%, and 75% of the video → Expect: the mask follows the object boundary without bleed into adjacent objects.

---

### UF-4.2 Multi-Object Propagation

**Contract:** Shift+clicking objects in the sidebar or canvas builds a multi-select set displayed as colored chips in the PropagationBar. When propagation is triggered with multiple objects selected, the frontend passes `object_ids: [<id1>, <id2>, ...]` in the request body. The backend calls `reset_and_replay_objects` (which calls `_reset_inference_state` clearing all object registrations, then clears `_active_object`), then calls `replay_prompts_if_needed` to re-register all requested objects' prompts from `prompts.json`. This full reset+replay is mandatory because the native SAM3 predictor raises `RuntimeError("Cannot add new object id N after tracking starts")` if new objects are registered after any propagation has run — incremental addition is not possible. SAM3 then tracks all objects in a single pass. Selection (`selectedObjIds`) persists in React state after propagation completes, enabling immediate reverse-direction propagation without re-selecting.

**Precondition:** Golden session open (run UF-1.2 steps 1–2 first). Objects 1 and 2 each have a prompt and mask on frame 0 (run UF-3.1 step 2 for obj 1, UF-3.2 step 2 for obj 2). If a third object exists, record its frame 0 RLE as the non-target baseline. Fixture values from `tests/fixtures/harness/fixture.json`.

**Must NOT:**
- Drop any selected object during reset+replay (N10): after multi-object propagation with `object_ids:[<obj_1_id>, <obj_2_id>]`, both objects must have masks on the new frames. Verify by inspecting `masks.json` for each propagated frame.
- Touch unselected objects' masks (N1): byte-compare any third object's RLE strings in `masks.json` before and after — must be byte-identical.
- Mutate `prompts.json` during replay (code review, not runtime): `replay_prompts_if_needed` calls `load_all_prompts` for reading, then calls `add_click`/`add_box`/`add_mask` on the SAM3 predictor — none of these write to `prompts.json`; the file must be byte-identical before and after multi-object propagation. Verify by diff.
- Attempt to incrementally add objects after tracking started (code review, not runtime): the multi-object path in `segment.py propagate()` calls `sam.reset_and_replay_objects` unconditionally when `len(object_ids) != 1`, which clears tracking state before replay. The single-object path uses `ensure_active_object` (also resets if switching objects). Neither path calls `add_new_points_or_box` on an already-tracking predictor with a new `obj_id`. Reference: `backend/app/routes/segment.py` `propagate()` and CLAUDE.md § "SAM3 & Device Constraints".

**Verify:**
1. `[DISK]` Snapshot `masks.json` (save as `masks_before.json`) and `prompts.json` (save as `prompts_before.json`). → Expect: snapshots saved.
2. `[API]` `curl -s -X POST http://localhost:5555/api/segment/propagate -H "Content-Type: application/json" -d '{"session_id":"<session_id>","start_frame_idx":0,"reverse":false,"object_ids":[<obj_1_id>,<obj_2_id>]}'` and consume SSE until `{"done": true}`. → Expect: HTTP 200; every `FrameResult` event's `masks` contains keys for both `"<obj_1_id>"` and `"<obj_2_id>"`; no other obj_id appears in any event.
3. `[DISK]` Inspect `masks.json` → Expect: every propagated frame has entries for both `<obj_1_id>` and `<obj_2_id>`; any third object's RLE strings are byte-identical to `masks_before.json` (N1, N10). Diff `prompts.json` against `prompts_before.json` → Expect: byte-identical (N10).
4. `[BROWSER]` Shift+click object 1 and object 2 in the sidebar → Expect: two colored chips appear in the PropagationBar, one per object. Click Forward → Expect: propagation starts; both objects' masks advance on each frame; canvas shows two overlapping colored overlays. After completion, the PropagationBar still shows both chips (selection persists) and the Back button is enabled for immediate reverse propagation.
5. `[API]` After multi-object propagation completes, POST a single-object click for object 1 on frame 0 → Expect: HTTP 200 within 10 s (the reset+replay inside `ensure_active_object` for the subsequent click confirms no lock is held and inference state is accessible).

---

### UF-4.3 Propagation Reconnect

**Contract:** If the browser tab is closed or the SSE connection is dropped while propagation is running, the backend continues: the propagation thread holds `SAM3Service._lock` and persists masks frame-by-frame via `persist_fn`. Reopening the tab triggers `loadSession` which calls `reconnectPropagation` (fire-and-forget). `reconnectPropagation` calls `GET /api/segment/propagation-status/<session_id>`; if `status == "running"`, it calls `subscribePropagation` (`GET /api/segment/propagate/subscribe/<session_id>`) which returns an SSE stream of the remaining frames. Frames processed before reconnect are backfilled by reading each missed frame's masks from the API in the background. The guard against a second concurrent propagation is the backend: `start_propagation` raises `ValueError` (→ HTTP 409) if `status == "running"`, so a second `POST /api/segment/propagate` is rejected; `reconnectPropagation` does not POST propagate — it only subscribes.

**Precondition:** Golden session open (run UF-1.2 steps 1–2 first); use its `<session_id>`. Object 1 has a prompt on frame 0. `make dev` running. Fixture video must have at least 10 frames so propagation takes > 5 s.

**Must NOT:**
- Duplicate or skip frames on reconnect: after a full propagation run (either undisturbed or after reconnect), `masks.json` must contain exactly one entry per frame in the expected range — no frame appears twice and no frame in the range is missing. Verify by counting keys and confirming each `frame_idx` in the range has exactly one entry per object.
- Spawn a second propagation on reconnect (code review, not runtime): `reconnectPropagation` in `App.tsx` calls `subscribePropagation` (a GET SSE subscribe), never `propagate` (a POST that starts a new run). The backend's `subscribe_propagation` route guards with `if status["status"] != "running": return done-sentinel immediately`. A second `POST /api/segment/propagate` during a running propagation returns HTTP 409 from `start_propagation`.

**Verify:**
1. `[API]` Start propagation: `curl -s -X POST http://localhost:5555/api/segment/propagate -H "Content-Type: application/json" -d '{"session_id":"<session_id>","start_frame_idx":0,"reverse":false,"object_ids":[<obj_1_id>]}'` — do NOT consume the full stream. After receiving at least 3 `FrameResult` events (≥ 3 frames processed), close the curl connection with Ctrl+C. → Expect: curl exits; `GET http://localhost:5555/api/segment/propagation-status/<session_id>` returns `{"status":"running","frames_processed":<N>}` with N ≥ 3.
2. `[API]` Reconnect to the running propagation: `curl -s http://localhost:5555/api/segment/propagate/subscribe/<session_id>` (GET, SSE) and consume until `{"done": true}`. → Expect: HTTP 200 with `Content-Type: text/event-stream`; `FrameResult` events arrive for the remaining frames; final sentinel `{"done": true}` delivered; no duplicate frame_idx values in the combined step-1 + step-2 event logs.
3. `[API]` After the subscribe stream closes: `GET http://localhost:5555/api/segment/propagation-status/<session_id>` → Expect: `{"status": "idle"}` (propagation completed and cleaned up).
4. `[DISK]` Inspect `masks.json` → Expect: entries exist for every frame in `[1, num_frames-1]` for `<obj_1_id>`; each frame appears exactly once; no gaps and no duplicates.
5. `[API]` Count entries in `masks.json` from a completed undisturbed propagation (re-run UF-4.1 on a fresh session) and compare to the step-4 reconnect run → Expect: identical frame count for the target object.
6. `[BROWSER]` While propagation is running (watch the green ticks advancing on the timeline), close the browser tab. Reopen `http://localhost:5173` and resume the session → Expect: the progress bar resumes mid-propagation (not from 0%); green ticks appear for already-completed frames; remaining frames complete; final mask count equals an undisturbed run.

---

### UF-4.4 Cancel Propagation

**Contract:** Clicking the Stop button in the PropagationBar calls `propagateAbortRef.current.abort()` (which drops the SSE connection client-side) and then `POST /api/segment/propagate/cancel/<session_id>`. The cancel route calls `sam.cancel_propagation(session_id)`, which sets the `threading.Event` in `_cancel_events[session_id]`. The propagation loop in `_run_propagation` checks `cancel_event.is_set()` once per frame, after `persist_fn` has written the current frame's masks but before queuing the result to subscribers. Frames persisted before the cancel check retains their masks on disk; the cancel is best-effort — one additional frame may complete between the event being set and the loop checking it. After the loop exits, the finally block sets `sm.set_propagating(False)`, delivers the `None` sentinel to all subscribers (ending their SSE streams), and pops `_propagation_state` so `get_propagation_status` returns idle. The UI returns to interactive state: `propagating` is reset to `false`, `propagationProgress` to 0.

**Precondition:** Golden session open (run UF-1.2 steps 1–2 first); use its `<session_id>`. Object 1 has a prompt on frame 0. Fixture video must have at least 10 frames.

**Must NOT:**
- Leave the propagation thread holding `SAM3Service._lock` after cancel (N9): after cancel, `GET /api/segment/propagation-status/<session_id>` must report `{"status": "idle"}` within one frame's inference time (~5 s on MPS per CLAUDE.md's ~2.9 s/frame benchmark, rounded up with margin), and a subsequent `POST /api/segment/click` must respond within 10 s (the lock being held would block indefinitely, not for exactly 10 s — the 10 s bound detects pathological hangs).
- Roll back already-persisted frames: masks written to `masks.json` before the cancel event was checked must remain in the file after cancel. Verify by snapshotting `masks.json` immediately after the cancel response and confirming the pre-cancel frames' entries are intact.
- Leave `GET /api/segment/propagation-status/<session_id>` reporting `status: "running"` after cancel completes (N8): the finally block in `_run_propagation` pops `_propagation_state` under `_propagation_lock`.

**Verify:**
1. `[DISK]` Snapshot `masks.json` before propagation (save as `masks_pre_prop.json`). → Expect: snapshot saved.
2. `[API]` Start propagation in a background process: `curl -s -X POST http://localhost:5555/api/segment/propagate -H "Content-Type: application/json" -d '{"session_id":"<session_id>","start_frame_idx":0,"reverse":false,"object_ids":[<obj_1_id>]}' &`. Wait for `frames_processed` to reach at least 2: `curl -s http://localhost:5555/api/segment/propagation-status/<session_id>` in a poll loop (1 s interval, max 30 s) until `frames_processed >= 2`. Record `N = frames_processed` at cancel time.
3. `[API]` Cancel: `curl -s -X POST http://localhost:5555/api/segment/propagate/cancel/<session_id>` → Expect: HTTP 200, `{"ok": true}`.
4. `[API]` Poll `GET http://localhost:5555/api/segment/propagation-status/<session_id>` every 1 s for up to 10 s → Expect: `{"status": "idle"}` within 10 s (N8, N9). If it remains `"running"` after 10 s, report FAIL — the propagation thread is not respecting the cancel event.
5. `[DISK]` Inspect `masks.json` → Expect: at least N entries for `<obj_1_id>` (frames 1 through N, noting the off-by-one from the per-frame check — exactly N or N+1 frames may have been written); all pre-cancel entries that were present match `masks_pre_prop.json` plus the new propagated frames (no rollback). No masks were deleted.
6. `[API]` `curl -s -X POST http://localhost:5555/api/segment/click -H "Content-Type: application/json" -d '{"session_id":"<session_id>","frame_idx":0,"obj_id":<obj_1_id>,"points":[<fixture_click_xy_obj1>],"labels":[1]}'` → Expect: HTTP 200 within 10 s (N9 — confirms `_lock` is not held by a zombie propagation thread).
7. `[API]` `curl -s http://localhost:5555/api/segment/propagation-status/<session_id>` → Expect: `{"status": "idle"}` (final N8 checkpoint after successful click).
8. `[BROWSER]` Start propagation via the PropagationBar Forward button; after 2–3 green ticks appear on the timeline, click Stop → Expect: the PropagationBar returns to its pre-propagation state within 5 s; the progress bar disappears; the timeline shows green ticks only for completed frames; annotation tools (click, box) are responsive immediately.

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
