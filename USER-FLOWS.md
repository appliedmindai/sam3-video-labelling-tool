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

**Scenario tests:** for broad changes (SAM3 service, propagation, persistence)
or before a release, also run the end-to-end **Scenario tests (S1–S5)** — they
compose the flows into full user journeys; S4 is the model-regression canary.

**Poll budget:** Polling steps (e.g. waiting for `phase: "ready"`) poll every 1–2 s and FAIL after 5 minutes for extraction/init on the fixture video, or immediately if phase becomes `"error"` (unless the flow expects `error`).

**SSE steps:** always use `curl -sN` (`--no-buffer`) when consuming `text/event-stream` responses; without `-N`, curl buffers piped output and events appear late or not at all. Terminate early with `timeout <secs> curl -sN ...` or by killing the process.

**Environment:** local dev (`make dev`: Flask :5555 + Vite :5173, conda env
`sam2-annotator` — note: the env name predates the SAM3 rename, MPS). Cloud-mode-only paths (GCS sync, Cloud Run lifecycle)
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

## Global negative invariants (N1–N11)

This document defines two kinds of negative checks: **global invariants** (N1–N11, this section) and **per-flow Must NOT lines** (which may reference global invariants by ID). Global invariants are stated once here, referenced by ID from individual flows.

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
  _Check:_ COCO `_annotations.coco.json` bboxes equal mask bbox + padding from `state.json`; exported image count == annotated-frame count.
  (Violation recorded 2026-06-10 — exporter shipped ALL frames — resolved 2026-06-11: `export_coco` now emits only frames that produce at least one annotation, in both `images[]` and the copied files. Regression tests: `test_export_coco_only_annotated_frames`, `test_export_coco_frame_without_valid_annotations_excluded` in `backend/tests/test_exporter.py`.)

- **N8** `/api/status` phase always reflects reality — it drives all frontend top-level routing.
  _Check:_ at each checkpoint a flow designates, `GET /api/status` returns exactly the `phase` value that flow's Verify step specifies. (Flows are the source of expected values.)

- **N9** Cancel actually cancels — no orphaned propagation thread holding `SAM3Service._lock`.
  _Check:_ after cancel, `GET /api/segment/propagation-status/{session_id}` reports not-running, and a subsequent click responds in < 10 s on MPS (vs. minutes if the lock were still held).

- **N10** Multi-object reset+replay loses no prompts — every selected object's prompts are replayed from disk after `_reset_inference_state`.
  _Check:_ after multi-object propagation, every selected object has masks on the new frames; `prompts.json` byte-identical.

- **N11** Auth is all-or-nothing — when `AUTH_PASSWORD` is set, no API
  endpoint (including `/api/health`) responds without a valid
  `X-Auth-Token`; when unset, no endpoint demands one. The password never
  appears in URLs (query params leak into request logs).
  _Check:_ `backend/tests/test_auth.py` covers the 401/200 matrix; grep
  `frontend/src` for `X-Auth-Token` — it must only travel as a header.

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

All fixture assets are committed. The sample video is the first 8 s of `IMG_4130.MOV` (Breville Barista Express — the RF-DETR dataset v2 source); goldens were generated on MPS on 2026-06-10 (see the fixture README for the full recipe). Fixture objects: obj 1 "pressure gauge" (click), obj 2 "portafilter" (box), keyframe 0.

**Comparison rules:**

- SAM3 output vs. golden masks: **IoU ≥ 0.80** (SAM3 is not bit-exact across MPS/CUDA backends).
- "Untouched data didn't change" checks (e.g. N1): **byte equality** of RLE strings.
- `golden_session.zip` doubles as the import-flow test asset (UF-8.2).
- All IoU values in this document are computed with `tests/fixtures/harness/iou.py` — the canonical method.

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
2. `[API]` Poll `GET http://localhost:5555/api/status` every 500 ms → Expect: `phase` reaches `"ready"`; any intermediate phases observed appear in the order `"extracting"` → `"initializing"` → `"ready"` (on a warm model the fixture pipeline completes in ~1 s, so intermediate phases may be missed entirely even at 500 ms polling — observed 2026-06-10); `progress` is non-decreasing within each phase; `session_id` matches the value from step 1.
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
2. `[API]` `curl -s -X POST http://localhost:5555/api/segment/click -H "Content-Type: application/json" -d '{"session_id":"<session_id>","frame_idx":0,"obj_id":<obj_1_id>,"points":[<fixture_click_xy_obj1>],"labels":[1]}'` → Expect: HTTP 200; response JSON has `frame_idx: 0`, `masks` key containing `"<obj_1_id>"`, and the RLE decodes to a non-empty binary mask (decode with `tests/fixtures/harness/iou.py` — the backend encodes masks as pycocotools COCO compressed RLE via `_encode_mask` in `mask_storage.py`; iou.py uses the same library — and confirm the mask is non-empty); IoU vs golden keyframe mask ≥ 0.80.
3. `[DISK]` Inspect `prompts.json` → Expect: `prompts["0"]["<obj_1_id>"]` has `type: "click"`, `points` matching the submitted coordinates, `labels: [1]`. Inspect `masks.json` → Expect: `masks["0"]["<obj_1_id>"]` entry is present. All other objects' RLE strings are byte-identical to the pre-click snapshot.
4. `[API]` DELETE the mask for obj 1 on the keyframe (`DELETE /api/session/masks/<session_id>/0/<obj_1_id>`), re-POST the step-2 click for obj 1, and compare the returned RLE against the step-2 response → Expect: IoU ≥ 0.99 via `iou.py` (N3).
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
2. `[API]` `curl -s -X POST http://localhost:5555/api/segment/box -H "Content-Type: application/json" -d '{"session_id":"<session_id>","frame_idx":0,"obj_id":<obj_2_id>,"box":[<fixture_box_obj2>]}'` → Expect: HTTP 200; response contains `frame_idx: 0` and `masks["<obj_2_id>"]` with a non-zero RLE; IoU vs golden ≥ 0.80.
3. `[DISK]` Inspect `prompts.json` → Expect: `prompts["0"]["<obj_2_id>"]` has `type: "box"` and `box` matching the submitted `[x1, y1, x2, y2]`. Inspect `masks.json` → Expect: `masks["0"]["<obj_2_id>"]` present. All other objects' RLE strings byte-identical to the pre-box snapshot.
4. `[BROWSER]` With the Box tool active, drag the fixture's canonical box coordinates for object 2 → Expect: a semi-transparent rectangle preview appears while dragging; on mouse-up, the preview disappears and a colored mask overlay appears; status reads "Ready".
5. `[BROWSER]` Drag a box smaller than 5×5 px (a near-stationary click) → Expect: no API call is made, no mask appears, no error in the status bar → Expect: `prompts.json` unchanged (no new box prompt) — observe via re-diff against the step-1 snapshot.

---

### UF-3.3 Text Detection

**Contract:** With the Detect tool active, entering a text query and submitting calls `POST /api/segment/text` with `{session_id, frame_idx, text, obj_id_start}`, where `obj_id_start` is computed by the frontend as `max(existing obj_ids) + 1`. The backend remaps its internal model IDs to sequential frontend IDs starting at `obj_id_start`, writing only `prompts.json` and `masks.json`; it does NOT write `state.json`. Object registration in `state.json` happens exclusively via the frontend's `persistState` PUT call inside `handleDetect` in `App.tsx` after the API response is received. The prompt type persisted in `prompts.json` is `"mask"` (RLE of the detected mask), not `"text"`. All detected instances are multi-selected after detection, with the PropagationBar showing one chip per instance. Pressing the toolbar's Cancel button triggers `AbortController.abort()`, which cancels the in-flight fetch client-side only — the backend request continues executing to completion on the server and its result is discarded when the aborted response arrives; a cancelled detection may therefore still persist masks and prompts entries on disk if the backend finished before the abort propagated (known limitation; a future task should add server-side cancellation).

**Precondition:** golden session open (run UF-1.2 steps 1–2 first); use its `<session_id>`. `text_query` and `expected_text_detections` come from `tests/fixtures/harness/fixture.json`; object IDs from `fixture.json`.

**Must NOT:**
- Detected instances overwrite or reuse existing `obj_id` values — `obj_id_start` is `max(all existing obj_ids) + 1`, computed in `handleDetect` in `App.tsx`; instances are assigned IDs `obj_id_start`, `obj_id_start + 1`, … sequentially. Verify: after detection the minimum new `obj_id` in `state.json` (written by the frontend's `persistState`, not the backend) equals the pre-detection `max(obj_ids) + 1`.
- Detection mutate existing objects' RLE strings in `masks.json` — byte-compare all pre-existing mask entries before and after (N1).
- Guarantee that an aborted detection leaves no disk state — because abort is client-side only (see Contract), a cancelled request will leave masks and prompts entries on disk if the backend finished before the abort propagated; the frontend discards the result and shows "Detection cancelled" in the status bar (code review, not runtime).

**Verify:**
1. `[DISK]` Snapshot `masks.json`, `prompts.json`, and `state.json` for the open session; record `max_obj_id_before = max(obj_ids in state.json)`. → Expect: snapshots saved.
2. `[API]` `curl -s -X POST http://localhost:5555/api/segment/text -H "Content-Type: application/json" -d '{"session_id":"<session_id>","frame_idx":0,"text":"<fixture_text_query>","obj_id_start":<max_obj_id_before+1>}'` → Expect: HTTP 200; `result.instances` length equals `fixture.json`'s `expected_text_detections`; each instance has `obj_id` in the range `[obj_id_start, obj_id_start + N - 1]`, a non-empty RLE, and a non-zero `area`.
3. `[DISK]` Inspect `state.json` → Expect: UNCHANGED from the pre-detection snapshot (the backend pure-API call does not write state.json). Inspect `prompts.json` → Expect: each new `obj_id` (starting at `max_obj_id_before + 1`) has a prompt entry with `type: "mask"` and an `rle` field. Inspect `masks.json` → Expect: N new entries for the detected obj_ids; all pre-existing RLE strings byte-identical to the pre-detection snapshot.
4. `[BROWSER]` With the Detect tool active, enter the fixture's `text_query` and press Detect → Expect: the status bar shows `Found N "<text_query>" instance(s) — all selected, ready to propagate` where N equals `fixture.json`'s `expected_text_detections`; N colored chips appear in the PropagationBar; the canvas shows N mask overlays in the class color; `state.json` on disk now contains N new object entries with `obj_id` values starting at `max_obj_id_before + 1` and the correct `class_id` (written by the frontend's `persistState` call after the API response).
5. `[BROWSER]` Click the toolbar's Cancel button immediately after starting Detect (detection takes several seconds on the fixture; if it completes first, re-run) → Expect: status bar shows "Detection cancelled" within 10 s of cancellation; no new objects appear in the object list in the frontend (the frontend discards the result); note that the backend may have completed and written to disk — verify by checking `state.json` for any new entries after cancel (acceptable behavior — see Contract). Note: Escape is wired only to multi-select clearing, not to detection cancellation.

---

### UF-4.1 Single-Object Propagation

**Contract:** With one object selected and a mask on the current frame, clicking Back, Both, or Forward posts `POST /api/segment/propagate`, which starts a background thread holding `SAM3Service._lock` and streams per-frame masks over SSE. Each SSE event is a `data: <JSON>\n\n` line whose JSON is either a `FrameResult` (`{frame_idx, masks, source_keyframe?}`) or the sentinel `{"done": true}`. The frontend's `consumeSseFrames` in `api.ts` invokes `onFrame` for each `FrameResult` and returns when the sentinel arrives. Frames where any mask's `confidence` field is less than 0.75 (the `CONFIDENCE_THRESHOLD` constant in `App.tsx`) are flagged with an amber timeline tick and accumulate in the `confidenceWarnings` state; a banner with frame count and a "Go to frame" shortcut appears after propagation completes. The `source_keyframe` field on each persisted mask in `masks.json` records the user-authored keyframe that initiated the propagation run.

**Precondition:** Golden session open (run UF-1.2 steps 1–2 first); use its `<session_id>`. Object 1 must have a click or box prompt on frame 0 (run UF-3.1 or UF-3.2 step 2 first). Object 2 must also have a prompt and mask on frame 0 (run UF-3.2 step 2 for `<obj_2_id>`) — this mask is kept throughout as the N1 non-target baseline; do not delete it before or during this flow. Fixture values from `tests/fixtures/harness/fixture.json`; IoU via `tests/fixtures/harness/iou.py`.

**Must NOT:**
- Create, modify, or delete masks for any non-target object (N1 — headline invariant): byte-compare all non-target objects' RLE strings in `masks.json` before and after propagation. Induce: propagate object 1 forward while object 2 has a mask on frame 0; diff the before/after snapshots for object 2's entries — any change is a failure.
- Regenerate a previously deleted mask (N2 — genuine check): after step-2 propagation of object 1 completes, delete object 1's keyframe mask and its prompt via `DELETE /api/session/masks/<session_id>/0/<obj_1_id>` (this route calls both `delete_frame_object_mask` and `delete_frame_object_prompt`, so the prompt is also removed from `prompts.json` — acknowledge that both are gone). Then POST propagate `object_ids:[<obj_1_id>]` again. With no remaining prompts for obj 1, `ensure_active_object` reads `load_all_prompts` → finds nothing for obj 1 → replays 0 prompts → inference state has no registered objects → the propagation iterator raises or yields nothing → the `_run_propagation` exception handler delivers an error event to subscribers and the SSE stream ends without `{"done": true}`. Inspect `masks.json` after this error propagation: object 1's masks must NOT be regenerated or extended beyond what was already on disk before the second propagation attempt. Full deletion-flow coverage in UF-5.1.
- Leave `GET /api/segment/propagation-status/<session_id>` reporting `status: "running"` after the SSE stream delivers `{"done": true}` (N9): the propagation finally block pops the entry under `_propagation_lock` so `get_propagation_status` returns `{"status": "idle"}` (missing key → idle default).
- Lose the `source_keyframe` metadata on propagated masks: each propagated frame's mask entry in `masks.json` must have a `source_keyframe` key equal to the integer frame index of the user's keyframe (not `null`).

**Verify:**
1. `[DISK]` Snapshot `masks.json` (save as `masks_before.json`). Record the RLE strings for all non-target object entries. Object 2 must have a mask on frame 0 from the Precondition; do NOT delete it here (it is the N1 baseline). → Expect: snapshot saved.
2. `[API]` `curl -sN -X POST http://localhost:5555/api/segment/propagate -H "Content-Type: application/json" -d '{"session_id":"<session_id>","start_frame_idx":0,"reverse":false,"object_ids":[<obj_1_id>]}'` and consume the SSE response (`-N`/`--no-buffer` required — without it curl buffers the stream and events appear late or not at all): for each `data:` line parse JSON; collect all `FrameResult` events until `{"done": true}`. `num_frames` = the `frame_count` field in `backend/sessions/<session_id>/meta.json` (written by `ExtractFramesStep`; alternatively `ls backend/sessions/<session_id>/frames/ | wc -l`). → Expect: HTTP 200 with `Content-Type: text/event-stream`; every `FrameResult` has `frame_idx` in `[0, num_frames-1]` (forward propagation from frame 0 includes frame 0 itself — `propagate_in_video_iterator` processes `range(start_frame_idx, end_frame_idx+1)`; the first event is typically frame 0 and is an upsert of the keyframe mask) and `masks` containing only the key `"<obj_1_id>"`; `{"done": true}` arrives as the final event.
3. `[API]` `curl -s http://localhost:5555/api/segment/propagation-status/<session_id>` → Expect: `{"status": "idle"}` (N9 checkpoint).
4. `[DISK]` Inspect `masks.json` → Expect: every propagated frame entry for `<obj_1_id>` has `source_keyframe: 0` (the Precondition pins keyframe 0); all non-target object RLE strings are byte-identical to the `masks_before.json` snapshot (N1).
5. `[API]` N2 check: `DELETE http://localhost:5555/api/session/masks/<session_id>/0/<obj_1_id>` → Expect: HTTP 200. Verify `masks.json` and `prompts.json` no longer contain frame 0 entries for `<obj_1_id>`. Then POST propagate `object_ids:[<obj_1_id>]` again and consume the SSE stream → Expect: the stream delivers an error event (JSON with `"error"` key — on HF/MPS: `IndexError: list index out of range` from the model with zero registered objects; confirmed 2026-06-10), followed by a `{"done": true}` terminator event; `masks.json` must NOT contain new or restored masks for `<obj_1_id>` beyond those already on disk before this second propagation attempt (N2). `GET /api/segment/propagation-status/<session_id>` → Expect: `{"status": "failed", "error": ...}` (confirmed on HF/MPS).
6. `[API]` For each frame index in `fixture.json`'s `propagation_sample_frames`, fetch `GET http://localhost:5555/api/session/masks/<session_id>/<frame_idx>` and compare `masks["<obj_1_id>"].rle` against the corresponding golden mask using `tests/fixtures/harness/iou.py` → Expect: IoU ≥ 0.80 for every sampled frame.
7. `[BROWSER]` In the annotation UI, select object 1 and click Forward in the PropagationBar → Expect: green timeline ticks appear frame-by-frame as propagation streams; the canvas advances to each completed frame live; amber ticks appear on frames where confidence < 0.75; after completion: if any frames have confidence < 0.75, the status bar shows the low-confidence frame count and a warning banner appears with a single "Go to frame N" shortcut (N = the first/lowest-index amber-ticked frame); if no frames have confidence < 0.75, a completion status is shown and no banner appears.
8. `[HUMAN]` Spot-check frames at approximately 25%, 50%, and 75% of the video → Expect: the mask follows the object boundary without bleed into adjacent objects.

---

### UF-4.2 Multi-Object Propagation

**Contract:** Shift+clicking objects in the sidebar or canvas builds a multi-select set displayed as colored chips in the PropagationBar. When propagation is triggered with multiple objects selected, the frontend passes `object_ids: [<id1>, <id2>, ...]` in the request body. The backend calls `reset_and_replay_objects` (which calls `_reset_inference_state` clearing all object registrations, then clears `_active_object`), then calls `replay_prompts_if_needed` to re-register all requested objects' prompts from `prompts.json`. This full reset+replay is mandatory because the native SAM3 predictor raises `RuntimeError("Cannot add new object id N after tracking starts")` if new objects are registered after any propagation has run — incremental addition is not possible. SAM3 then tracks all objects in a single pass. Selection (`selectedObjIds`) persists in React state after propagation completes, enabling immediate reverse-direction propagation without re-selecting.

**Precondition:** Golden session open (run UF-1.2 steps 1–2 first). Objects 1 and 2 each have a prompt and mask on frame 0 (run UF-3.1 step 2 for obj 1, UF-3.2 step 2 for obj 2). If a third object exists, record its frame 0 RLE as the non-target baseline. Fixture values from `tests/fixtures/harness/fixture.json`.

**Must NOT:**
- Drop any selected object during reset+replay (N10): after multi-object propagation with `object_ids:[<obj_1_id>, <obj_2_id>]`, both objects must have masks on the new frames. Verify by inspecting `masks.json` for each propagated frame.
- Touch unselected objects' masks (N1): byte-compare any third object's RLE strings in `masks.json` before and after — must be byte-identical. If the fixture has no third object, report this N1 sub-check NOT RUN.
- Mutate `prompts.json` during replay (code review, not runtime): `replay_prompts_if_needed` calls `load_all_prompts` for reading, then calls `add_click`/`add_box`/`add_mask` on the SAM3 predictor — none of these write to `prompts.json`; the file must be byte-identical before and after multi-object propagation. Verify by diff.
- Attempt to incrementally add objects after tracking started (code review, not runtime): the multi-object path in `segment.py propagate()` calls `sam.reset_and_replay_objects` unconditionally when `len(object_ids) != 1`, which clears tracking state before replay. The single-object path uses `ensure_active_object` (also resets if switching objects). Neither path calls `add_new_points_or_box` on an already-tracking predictor with a new `obj_id`. Reference: `backend/app/routes/segment.py` `propagate()` and CLAUDE.md § "SAM3 & Device Constraints".

**Verify:**
1. `[DISK]` Snapshot `masks.json` (save as `masks_before.json`) and `prompts.json` (save as `prompts_before.json`). → Expect: snapshots saved.
2. `[API]` `curl -sN -X POST http://localhost:5555/api/segment/propagate -H "Content-Type: application/json" -d '{"session_id":"<session_id>","start_frame_idx":0,"reverse":false,"object_ids":[<obj_1_id>,<obj_2_id>]}'` and consume SSE until `{"done": true}` (`-N`/`--no-buffer` required for SSE). → Expect: HTTP 200; every `FrameResult` event's `masks` contains keys for both `"<obj_1_id>"` and `"<obj_2_id>"`; no other obj_id appears in any event.
3. `[DISK]` Inspect `masks.json` → Expect: every propagated frame has entries for both `<obj_1_id>` and `<obj_2_id>`; any third object's RLE strings are byte-identical to `masks_before.json` (N1, N10). Diff `prompts.json` against `prompts_before.json` → Expect: byte-identical (N10).
4. `[BROWSER]` Shift+click object 1 and object 2 in the sidebar → Expect: two colored chips appear in the PropagationBar, one per object. Click Forward → Expect: propagation starts; both objects' masks advance on each frame; canvas shows two overlapping colored overlays. After completion, the PropagationBar still shows both chips (selection persists) and the Back button is enabled for immediate reverse propagation.
5. `[API]` After multi-object propagation completes, POST a single-object click for object 1 on frame 0 → Expect: HTTP 200 within 10 s (the reset+replay inside `ensure_active_object` for the subsequent click confirms no lock is held and inference state is accessible).

---

### UF-4.3 Propagation Reconnect

**Contract:** If the browser tab is closed or the SSE connection is dropped while propagation is running, the backend continues: the propagation thread holds `SAM3Service._lock` and persists masks frame-by-frame via `persist_fn`. Reopening the tab triggers `loadSession` which calls `reconnectPropagation` (fire-and-forget). `reconnectPropagation` calls `GET /api/segment/propagation-status/<session_id>`; if `status == "running"`, it calls `subscribePropagation` (`GET /api/segment/propagate/subscribe/<session_id>`) which returns an SSE stream of the remaining frames. Frames processed before reconnect are backfilled by reading each missed frame's masks from the API in the background. The guard against a second concurrent propagation is the backend: `start_propagation` raises `ValueError` (→ HTTP 409) if `status == "running"`, so a second `POST /api/segment/propagate` is rejected; `reconnectPropagation` does not POST propagate — it only subscribes.

**Precondition:** Golden session open (run UF-1.2 steps 1–2 first); use its `<session_id>`. Object 1 has a prompt on frame 0. `make dev` running. Fixture video must have at least 10 frames so propagation takes > 5 s. Step 5 requires a fresh undisturbed UF-4.1 run on a separate session for the frame-count comparison baseline.

**Must NOT:**
- Skip frames on reconnect: after a full propagation run (either undisturbed or after reconnect), `masks.json` must contain one entry per frame in the expected range with no gaps. Verify at disk level by counting `frame_idx` keys and confirming each is present exactly once. Note: `masks.json` is keyed by frame index — duplicate keys are structurally impossible in JSON (last-writer-wins), so "no duplicates" is guaranteed by construction and need not be verified at disk level; the SSE event log for the reconnected stream alone cannot prove completeness (frames processed before reconnect do not appear in the reconnected stream's events — `subscribe_propagation` delivers only frames that arrive after the new queue was installed).
- Spawn a second propagation on reconnect (code review, not runtime): `reconnectPropagation` in `App.tsx` calls `subscribePropagation` (a GET SSE subscribe), never `propagate` (a POST that starts a new run). The backend's `subscribe_propagation` route guards with `if status["status"] != "running": return done-sentinel immediately`. A second `POST /api/segment/propagate` during a running propagation returns HTTP 409 from `start_propagation`.

**Verify:**
1. `[API]` Start propagation: `curl -sN -X POST http://localhost:5555/api/segment/propagate -H "Content-Type: application/json" -d '{"session_id":"<session_id>","start_frame_idx":0,"reverse":false,"object_ids":[<obj_1_id>]}'` (`-N` required — without it, buffering delays events and the Ctrl+C interrupt may fire before any events are visible) — do NOT consume the full stream. After receiving at least 3 `FrameResult` events (≥ 3 frames processed), close the curl connection with Ctrl+C. → Expect: curl exits; `GET http://localhost:5555/api/segment/propagation-status/<session_id>` returns `{"status":"running","frames_processed":<N>}` with N ≥ 3.
2. `[API]` Reconnect to the running propagation: `curl -sN http://localhost:5555/api/segment/propagate/subscribe/<session_id>` (GET, SSE — `-N` required) and consume until `{"done": true}`. → Expect: HTTP 200 with `Content-Type: text/event-stream`; `FrameResult` events arrive for frames that were processed after this subscriber subscribed (frames processed while disconnected do NOT appear in this stream — a fresh queue receives only new results); no duplicate `frame_idx` values within this step-2 event log alone; final sentinel `{"done": true}` delivered. Completeness is proven only by the step-4 disk check, not by the union of step-1 and step-2 event logs.
3. `[API]` After the subscribe stream closes: `GET http://localhost:5555/api/segment/propagation-status/<session_id>` → Expect: `{"status": "idle"}` (propagation completed and cleaned up).
4. `[DISK]` Inspect `masks.json` → Expect: entries exist for every frame in `[0, num_frames-1]` for `<obj_1_id>` (forward propagation from frame 0 includes frame 0); no gaps in the covered range. This is the authoritative completeness check.
5. `[DISK]` Count `<obj_1_id>` entries in `masks.json` from a completed undisturbed UF-4.1 run (on the fresh baseline session established in the Precondition) and compare to the step-4 reconnect run → Expect: identical frame count for the target object.
6. `[BROWSER]` While propagation is running (watch the green ticks advancing on the timeline), close the browser tab. Reopen `http://localhost:5173` and resume the session → Expect: the progress bar resumes mid-propagation (not from 0%); green ticks appear for already-completed frames; remaining frames complete; final mask count equals an undisturbed run.

---

### UF-4.4 Cancel Propagation

**Contract:** Clicking the Stop button in the PropagationBar calls `propagateAbortRef.current.abort()` (which drops the SSE connection client-side) and then `POST /api/segment/propagate/cancel/<session_id>`. The cancel route calls `sam.cancel_propagation(session_id)`, which sets the `threading.Event` in `_cancel_events[session_id]`. The propagation loop in `_run_propagation` checks `cancel_event.is_set()` as the FIRST statement of the frame loop body, BEFORE `persist_fn` is called — so the frame that was being inferred when cancel arrives is discarded and never persisted. Best-effort means already-persisted frames stay on disk; the in-flight frame (yielded by the iterator but not yet passed to `persist_fn`) is dropped. The N or N+1 tolerance in step 5 accounts for frames that complete inference (and therefore enter the next iteration's cancel check) between the status poll recording N and the cancel POST being processed. After the loop exits, the finally block sets `sm.set_propagating(False)`, delivers the `None` sentinel to all subscribers (ending their SSE streams), and pops `_propagation_state` so `get_propagation_status` returns idle. The UI returns to interactive state: `propagating` is reset to `false`, `propagationProgress` to 0.

**Precondition:** Golden session open (run UF-1.2 steps 1–2 first); use its `<session_id>`. Object 1 has a prompt on frame 0. Fixture video must have at least 10 frames.

**Must NOT:**
- Leave the thread holding `SAM3Service._lock` or leave `GET /api/segment/propagation-status/<session_id>` reporting `status: "running"` after cancel completes (N9): after cancel, `propagation-status` must report `{"status": "idle"}` within one frame's inference time (~5 s on MPS), and a subsequent `POST /api/segment/click` must respond within 10 s; the finally block in `_run_propagation` releases `_lock` and pops `_propagation_state` under `_propagation_lock`.
- Roll back already-persisted frames: masks written to `masks.json` before the cancel event was checked must remain in the file after cancel. Verify by snapshotting `masks.json` immediately after the cancel response and confirming the pre-cancel frames' entries are intact.

**Verify:**
1. `[DISK]` Snapshot `masks.json` before propagation (save as `masks_pre_prop.json`). → Expect: snapshot saved.
2. `[API]` Start propagation in a background process: `curl -sN -X POST http://localhost:5555/api/segment/propagate -H "Content-Type: application/json" -d '{"session_id":"<session_id>","start_frame_idx":0,"reverse":false,"object_ids":[<obj_1_id>]}' &` (`-N` required for SSE — without it the backgrounded curl buffers and you will not see events until the buffer fills). Wait for `frames_processed` to reach at least 2: `curl -s http://localhost:5555/api/segment/propagation-status/<session_id>` in a poll loop (1 s interval, max 30 s) until `frames_processed >= 2`. Record `N = frames_processed` at cancel time. → Expect: poll returns `{"status":"running","frames_processed":<N>}` with N ≥ 2 before the 30 s timeout.
3. `[API]` Cancel: `curl -s -X POST http://localhost:5555/api/segment/propagate/cancel/<session_id>` → Expect: HTTP 200, `{"ok": true}`.
4. `[API]` Poll `GET http://localhost:5555/api/segment/propagation-status/<session_id>` every 1 s for up to 10 s → Expect: `{"status": "idle"}` within 10 s (N9). If it remains `"running"` after 10 s, report FAIL — the propagation thread is not respecting the cancel event.
5. `[DISK]` Inspect `masks.json` → Expect: exactly N or N+1 entries for `<obj_1_id>` (frames 0 through N-1, or frames 0 through N if one more frame completed between the status poll and the cancel POST — frame 0 is counted because `frames_processed` is incremented after `persist_fn` for every yielded frame including the keyframe upsert; "at least N" is intentionally dropped); all pre-cancel entries match `masks_pre_prop.json` plus the new propagated frames (no rollback). No masks were deleted.
6. `[API]` `curl -s -X POST http://localhost:5555/api/segment/click -H "Content-Type: application/json" -d '{"session_id":"<session_id>","frame_idx":0,"obj_id":<obj_1_id>,"points":[<fixture_click_xy_obj1>],"labels":[1]}'` → Expect: HTTP 200 within 10 s (N9 — confirms `_lock` is not held by a zombie propagation thread).
7. `[API]` `curl -s http://localhost:5555/api/segment/propagation-status/<session_id>` → Expect: `{"status": "idle"}` (final N9 checkpoint after successful click).
8. `[BROWSER]` Start propagation via the PropagationBar Forward button; after 2–3 green ticks appear on the timeline, click Stop → Expect: the PropagationBar returns to its pre-propagation state within 5 s; the progress bar disappears; the timeline shows green ticks only for completed frames; annotation tools (click, box) are responsive immediately.

---

### UF-5.1 Batch Mask Deletion

**Contract:** With an object selected, pressing Delete or Backspace (outside any text input) opens the DeleteMasksPanel. The panel shows the object's masked frames, two frame-range selectors (Start/End) with thumbnails and keyframe indicators, and a Delete button that calls `DELETE /api/session/masks/<session_id>/<obj_id>/batch` with body `{"frame_indices": [...]}`. The backend deletes the specified frames from `masks.json` and `prompts.json`, calls `sam.clear_frame_object_batch` to reset SAM3 in-memory inference state for the session, and returns `{"deleted_frames": [...], "deleted_count": N}`. The toolbar Delete button on the annotation canvas also opens this panel. Full deletion-flow coverage for the single-frame case is in UF-4.1's N2 check.

**Precondition:** Golden session open (run UF-1.2 steps 1–2 first); use its `<session_id>`. Object 1 must have masks on at least 7 frames (run UF-4.1 step 2 first to propagate). Fixture values from `tests/fixtures/harness/fixture.json`.

**Must NOT:**
- Allow a deleted mask to be regenerated from stale SAM3 in-memory state on a subsequent propagation (N2): the batch-delete route calls `sam.clear_frame_object_batch`, which calls `_reset_inference_state` — clearing ALL per-object tracking state for this session. After batch-delete, the next propagation must replay prompts from disk before any inference; the deleted frames' masks must not reappear (N2). Note: resetting the full inference state is intentional — the native SAM3 predictor does not support selective per-frame eviction. Subsequent propagation from a surviving keyframe will legitimately re-track the deleted range (correct behavior); the violation is resurrecting the deleted keyframe's mask itself (covered by UF-4.1's N2 recipe — reference that check rather than duplicating it).
- Touch any other object's masks or frames outside the specified `frame_indices` (N1): byte-compare all non-target objects' RLE strings in `masks.json` before and after — must be byte-identical; byte-compare `masks.json` entries for the target object outside the range — must be byte-identical.
- Leave `prompts.json` with dangling prompt entries for deleted keyframes without the Missing Mask panel surfacing them: the batch-delete route calls `delete_frame_object_prompt` for each deleted frame, so prompts are removed atomically with masks. If a prompt survives deletion (code defect), the Missing Mask panel must surface it for the user to clear — the panel's condition is: a prompt exists in React state for the selected object on the current frame but no mask is rendered.

**Verify:**
1. `[DISK]` Snapshot `masks.json` (save as `masks_before.json`) and `prompts.json` (save as `prompts_before.json`). Object 1 must have masks on frames 0–6. Record the RLE strings and frame keys for object 2 (non-target baseline). → Expect: snapshots saved.
2. `[API]` Delete frames 1–5 for object 1: `curl -s -X DELETE http://localhost:5555/api/session/masks/<session_id>/<obj_1_id>/batch -H "Content-Type: application/json" -d '{"frame_indices":[1,2,3,4,5]}'` → Expect: HTTP 200; response JSON has `deleted_frames: [1,2,3,4,5]` (sorted) and `deleted_count: 5`.
3. `[DISK]` Inspect `masks.json` → Expect: frame keys `"1"` through `"5"` have no entry for `str(<obj_1_id>)` (or the frame key is absent entirely if obj 1 was the only object on that frame); frames `"0"` and `"6"` still contain obj 1's RLE; all non-target object RLE strings are byte-identical to `masks_before.json` (N1). Inspect `prompts.json` → Expect: frame keys `"1"` through `"5"` have no entry for obj 1 (prompts deleted atomically with masks for those frames).
4. `[API]` N2 check: follow the same recipe as UF-4.1 step 5 using the surviving keyframe (frame 0) — propagate object 1 forward, then verify that frames 1–5 do not contain resurrected obj 1 masks beyond what propagation legitimately re-tracks from frame 0. Reference UF-4.1 step 5 for the exact recipe; frame 0's mask and prompt survived, so propagation replays from it and regenerates masks on frames 1–5 — that is correct behavior. The violation would be a mask on a frame whose prompt was deleted appearing without a prior propagation step, which this step confirms does not occur by checking that the full-propagation result matches an undisturbed propagation baseline.
5. `[BROWSER]` Select object 1. Press the Delete key (not inside any text input) → Expect: the DeleteMasksPanel appears in the right sidebar, showing frame-range selectors with thumbnails and keyframe indicators; the panel displays the count of masked frames and whether any are keyframes; the status bar is unchanged. Set start=1, end=3 and click Delete → Expect: the panel closes; the timeline ticks for frames 1–3 (for object 1) disappear; the status bar shows "Deleted 3 masks".

---

### UF-8.1 COCO Export

**Contract:** With a session open, clicking Export Annotations calls `POST /api/export/coco/<session_id>` with body `{"bbox_padding": <bbox_padding_state>}`, where `bbox_padding_state` is the per-object per-keyframe padding dict from `state.json` (shape `{obj_id_str: {frame_idx_str: {top, bottom, left, right}}}`). The backend loads state and masks under `session_io_lock`, runs `export_coco`, and serves a zip file named `{videoStem}_coco.zip` as an attachment. The zip contains: `_annotations.coco.json` (the COCO JSON, **not** `instances.json`) and the JPEG files of annotated frames only — a frame appears in `images[]` and as a file iff it produced at least one annotation (N7). Bbox padding is applied per-object per-keyframe: for each mask, the exporter looks up padding by `source_keyframe` (falling back to `frame_idx`), then calls `_pad_bbox` which expands a COCO `[x, y, w, h]` bbox by per-side height/width percentages and clamps to image bounds.

**Must NOT:**
- Include unannotated frames anywhere in the export (N7) — not in `annotations`, not in `images[]`, not as image files. Every `annotations[].image_id` must reference an entry present in `images[]`.
- Ignore saved bbox padding — exported bboxes must equal the mask tight-bbox expanded by the per-side padding from `state.json` (N7).
- Leak absolute server paths inside `_annotations.coco.json` — all `images[].file_name` values must be bare filenames (e.g. `"frame_0000.jpg"`), not absolute paths.

**Verify:**
1. `[API]` `curl -s -X POST http://localhost:5555/api/export/coco/<session_id> -H "Content-Type: application/json" -d '{"bbox_padding":{}}' -o /tmp/test_coco.zip -w "%{http_code}"` → Expect: HTTP 200; `/tmp/test_coco.zip` is a valid zip file (`python3 -c "import zipfile; z=zipfile.ZipFile('/tmp/test_coco.zip'); print(z.namelist())"`).
2. `[DISK]` Unzip to `/tmp/test_coco_out/`: `unzip -q /tmp/test_coco.zip -d /tmp/test_coco_out`. Inspect the zip namelist → Expect: `_annotations.coco.json` is present; all other entries are `*.jpg` files (frame images). Count jpg files in zip → Expect: count equals the number of annotated frames (== `images[]` length; the untouched golden session has all 40 annotated, so batch-delete one frame's masks for both objects first to observe the exclusion). Parse `_annotations.coco.json` → Expect: valid JSON with keys `info`, `licenses`, `images`, `categories`, `annotations`; every `annotations[].category_id` maps to a real entry in `categories`; no `images[].file_name` begins with `/` or contains a path separator other than what's expected for the flat filename.
3. `[DISK]` Annotated-frame count check: count `len(annotations)` unique `image_id` values → Expect: equals the number of frames that have at least one mask in `masks.json` for any object.
4. `[DISK]` Padding arithmetic check (requires a fixture frame with known padding). Given a mask on frame F for object O with `source_keyframe = K`, and padding `{top: T%, bottom: B%, left: L%, right: R%}` from `state.json` at key `[O][K]`, and tight bbox `[x, y, w, h]` from the mask: expected exported bbox = `[max(0, x - w*L/100), max(0, y - h*T/100), min(img_width - new_x, w + w*L/100 + w*R/100), min(img_height - new_y, h + h*T/100 + h*B/100)]` (all terms integer-truncated). Locate that annotation in `_annotations.coco.json` and compare → Expect: bbox matches the formula above (the fixture's golden session has padding on obj 1's keyframe — see fixture.json `bbox_padding`).
5. `[DISK]` Segmentation area check: for one annotation, decode its `segmentation` polygons (COCO RLE polygon format) and confirm the enclosed area is within 1% of the annotation's `area` field → Expect: areas match within 1%.

---

### UF-8.2 Session Export/Import Round-Trip

**Contract:** Clicking Export Session calls `POST /api/export/session/<session_id>` with body `{"include_video": true|false}`. The backend calls `export_session_bundle`, which packages `manifest.json`, `meta.json`, `state.json`, `masks.json`, `prompts.json`, and optionally `frames/*.jpg` and `video.mp4` into a zip, and serves it as `{videoStem}_session.zip`. Importing calls `POST /api/export/import-session` with the zip as a `file` multipart upload. The importer checks the manifest format and version, then deduplicates by `video_md5` from `manifest.session.video_md5` (which is read from `meta.json`'s `md5` field at export time) by scanning all local `meta.json` files via `find_session_by_md5`. If a session with the same MD5 already exists, it returns `{"session_id": <existing>, "duplicate": true}` without extracting any files. If not, it creates a new session directory and extracts files one by one with zip-slip protection (each path is checked against `os.path.realpath(target).startswith(real_session + os.sep)`) and bomb protection (10 GB decompressed limit, 100,000 file count limit). The import response on success (non-duplicate) is `{"session_id": <new_uuid>, "frame_count": N, "original_name": ..., "fps": ..., "duplicate": false}`. Masks, prompts, and state are written byte-for-byte from the zip (no transformation); `meta.json` is also written verbatim. After import, the session must be resumed (via `POST /api/session/resume/<session_id>`) to initialize SAM3 before annotation.

**Precondition:** Golden session zip available at `tests/fixtures/harness/golden_session.zip`. Its expected object count, object IDs, and mask frame count come from `tests/fixtures/harness/fixture.json`.

**Must NOT:**
- Lose or mutate any mask, prompt, or class during round-trip: `masks.json`, `prompts.json`, and `state.json` extracted from the import zip must be byte-identical to the corresponding files in the source session. The exporter writes them verbatim via `json.dumps(load_json(...))` — the only possible difference is JSON whitespace/key-order from a re-serialization round-trip; verify by parsing both and comparing as Python dicts (structural equality, not byte equality, because `json.dumps` indentation may differ from the original file's formatting).
- Execute or extract files outside the expected session directory: the importer checks `os.path.realpath(target).startswith(real_session + os.sep)` for every zip entry before extracting; any entry that traverses outside raises `ValueError` and rolls back via `shutil.rmtree`. There is no further OS-level sandbox. Verify by attempting to import a crafted zip with a `../evil.txt` entry → Expect: HTTP 400 with `{"error": "Illegal path in zip: ..."}` and no file created outside the session dir.
- Import a duplicate session's files on top of the existing session: when `video_md5` matches an existing session, the route returns `{"duplicate": true, "session_id": <existing>}` and extracts nothing; the existing session's files are untouched.

**Verify:**
1. `[API]` Export with frames: `curl -s -X POST http://localhost:5555/api/export/session/<session_id> -H "Content-Type: application/json" -d '{"include_video":false}' -o /tmp/test_session.zip -w "%{http_code}"` → Expect: HTTP 200; `/tmp/test_session.zip` is a valid zip. Inspect namelist → Expect: `manifest.json`, `meta.json`, `state.json`, `masks.json`, `prompts.json` all present; `frames/` entries present if frames exist locally; `video.mp4` absent (include_video=false).
2. `[API]` First delete the source session (`DELETE /api/video/sessions/<source_id>`) — importing while a session with the same `video_md5` exists short-circuits via `find_session_by_md5` and returns `duplicate: true` with the EXISTING session's id (nothing extracted). Then import: `curl -s -X POST http://localhost:5555/api/export/import-session -F "file=@/tmp/test_session.zip"` → Expect: HTTP 200; JSON body has `session_id` (a new UUID), `duplicate: false`, `frame_count` and `original_name` matching source. Note: the GCS upload of the imported session files after extraction is cloud mode only — NOT RUN locally.
3. `[DISK]` Compare files between the source session and the newly imported session. For each of `masks.json`, `prompts.json`, `state.json`: parse both files as JSON and compare as Python dicts → Expect: structurally identical (same keys, same values, same nesting). `meta.json` may differ only in fields added by the `export_session_bundle` step (none — it is written verbatim from the source's `meta.json`).
4. `[API]` Duplicate-detection check: re-import the same zip → Expect: HTTP 200; `duplicate: true`; `session_id` matches the session from step 2; no new session directory created under `backend/sessions/`.
5. `[API]` Zip-slip guard: craft a zip that contains a valid `manifest.json` (so the importer passes the format/version checks and reaches the extraction loop) PLUS a traversal entry. A zip with only `../evil.txt` and no `manifest.json` dies at the manifest read with a `KeyError` → HTTP 400 "Invalid session bundle file" before the zip-slip guard is ever reached. Craft with:
   ```
   python3 -c "
   import zipfile, json
   manifest = {'format': 'sam3-annotator-session', 'version': 1, 'session': {'video_md5': '', 'original_name': 'evil', 'frame_count': 0, 'fps': 5}}
   z = zipfile.ZipFile('/tmp/evil.zip', 'w')
   z.writestr('manifest.json', json.dumps(manifest))
   z.writestr('../evil.txt', 'pwned')
   z.close()
   "
   ```
   Then: `curl -s -X POST http://localhost:5555/api/export/import-session -F "file=@/tmp/evil.zip"` → Expect: HTTP 400 with `{"error": "Illegal path in zip: ../evil.txt"}` (from the `ValueError("Illegal path in zip: ...")` guard in `session_bundle.py`); `backend/evil.txt` does not exist and no files were extracted outside the session dir (the `except` block calls `shutil.rmtree` on the partial session dir).
6. `[BROWSER]` From the session list, import `tests/fixtures/harness/golden_session.zip` via the import file-picker → Expect: the imported session appears in the session list; clicking Resume initializes SAM3 and opens the annotation UI; the object list shows the fixture's expected object count from `fixture.json`; masks render on the keyframe with the fixture's expected object IDs.

---

### UF-12 Recovery (container restart / error phase / state version conflict)

**Contract:**
- **(a) Backend restart:** The frontend polls `GET /api/status` (via `useServiceStatus`) every few seconds. Each response includes a `boot_id` (a UUID generated at Flask startup in `app/config.py`). When the phase drops from `"ready"` to `"idle"` (detected by `prevPhaseRef` in `App.tsx`), the frontend resets `sessionId` to `null` and sets the status bar to `"Session lost — server restarted. Please select a session."` — the session list re-appears. Persisted session files (`masks.json`, `state.json`, `prompts.json`) are untouched by the restart; resuming the session via the list restores all prior annotations.
- **(b) Pipeline error phase:** A pipeline failure (e.g. invalid video, model load failure) transitions `ServiceState.phase` to `"error"` with `ServiceState.error` set to a string message. `GET /api/status` returns `{"phase": "error", "error": "<message>", ...}`. The UI shows the error; `POST /api/status/dismiss-error` calls `sam.dismiss_error()`, which (if and only if phase is currently `"error"`) replaces `ServiceState` with a fresh `ServiceState(phase="idle")` and returns `{"status": "dismissed"}`. If phase is not `"error"`, it returns `{"status": "not_in_error"}` and does nothing.
- **(c) State version conflict (two-tab editing):** Every `PUT /api/session/state/<session_id>` payload carries a `version` integer. The backend reads the current state from disk (under `session_io_lock`), compares payload `version` to `current.get("version", 0)`, and 409s if they differ: `{"error": "state_version_conflict", "current_version": <N>, "state": <full_current_state>}`. The frontend's `putSessionState` throws `StateVersionConflictError` carrying `err.currentVersion` and `err.currentState`; the `persistState` caller catches it and calls `reconcileFromServer`, which sets `stateVersionRef.current = serverVersion` and updates React state (classes, objects, bbox_padding) from the server snapshot, then sets the status bar to `"Another tab updated this session — reloaded from server."`. The wipe-fingerprint guard (N5) is enforced in **both** the frontend (`stateIsSafeToPersist` in `App.tsx`, which gates `persistState` and all `putSessionState` callers) **and** the backend (`put_state` route, which 409s if `classes` is empty and any object has `class_id < 0` and current disk state has non-empty classes). A 409 from the version guard does NOT bypass the wipe-fingerprint guard — the loser's payload is rejected; `reconcileFromServer` then overwrites local state with the server's authoritative snapshot.

**Must NOT:**
- Corrupt persisted session files during a backend restart — `masks.json`, `state.json`, and `prompts.json` must be byte-identical before and after a kill+restart. Flask uses atomic writes (`atomic_json_dump`) so a mid-write kill leaves the previous file intact.
- Leave `GET /api/status` returning `phase: "error"` after a successful `POST /api/status/dismiss-error` — the phase must transition to `"idle"` (N8). Dismiss is a no-op when not in error phase (returns `"not_in_error"`; phase unchanged).
- Allow a 409 loser to silently overwrite the winner's state: the backend rejects the stale PUT (409); the frontend's `reconcileFromServer` replaces local state with the server snapshot and updates `stateVersionRef.current` so the next auto-save sends the correct version. The wipe-fingerprint guard (N5) is enforced on both sides — see Contract above for where each guard lives.

**Verify:**
1. `[DISK]` With a session open and at least one annotated frame, snapshot `masks.json`, `state.json`, and `prompts.json` (save as `*_pre_kill.json`). → Expect: snapshots saved.
2. `[API]` Kill the Flask process. The dev server runs either as `python3 -m flask ... --port 5555` (`make dev`) or as `python3 run.py` (direct) — `pgrep -f "flask.*5555"` matches only the former; use `pgrep -f "run.py"` for the latter (it may return several PIDs — shell wrapper plus the werkzeug reloader child; kill them all). Restart the backend: `cd backend && PYTORCH_ENABLE_MPS_FALLBACK=1 python3 -m flask --app app:create_app run --port 5555 &`. Poll `GET http://localhost:5555/api/status` every 1 s for up to 30 s → Expect: returns `{"phase": "idle", "session_id": null, "boot_id": "<new_uuid>"}` once Flask is up; `boot_id` differs from the pre-kill value (confirms a fresh process).
3. `[DISK]` `diff` each file against its pre-kill snapshot → Expect: `masks.json`, `state.json`, and `prompts.json` are byte-identical (restart does not mutate session files).
4. `[API]` Upload a non-video file renamed to `.mp4` to force a pipeline error: `echo "not a video" > /tmp/bad.mp4 && curl -s -X POST http://localhost:5555/api/video/upload -F "video=@/tmp/bad.mp4" -F "fps=5"` → Expect: HTTP 200 with a `session_id`. Poll `GET http://localhost:5555/api/status` every 1 s for up to 60 s → Expect: `phase` reaches `"error"` (not `"extracting"` indefinitely); `error` field contains a non-empty string message (N8). (This is the induce target UF-1.1 references.)
5. `[API]` Dismiss the error: `curl -s -X POST http://localhost:5555/api/status/dismiss-error` → Expect: HTTP 200, `{"status": "dismissed"}`. Then `curl -s http://localhost:5555/api/status` → Expect: `{"phase": "idle", "session_id": null, ...}` (N8, phase is idle not error).
6. `[API]` State version conflict: open the golden session (resume it) to get `<session_id>` and `version_N` (from `GET /api/session/state/<session_id>`, field `version`). **Snapshot:** save the full response body as `state_pre_conflict.json` (this is the golden state to restore after the test). PUT state with the current version to advance it: `curl -s -X PUT http://localhost:5555/api/session/state/<session_id> -H "Content-Type: application/json" -d '{"classes":[{"id":1,"name":"Test","color":"#ff0000"}],"objects":[],"version":<version_N>}'` → Expect: HTTP 200, `{"ok": true, "version": <version_N+1>}`. Now PUT again with the **stale** version `version_N` (simulating the second tab): `curl -s -X PUT http://localhost:5555/api/session/state/<session_id> -H "Content-Type: application/json" -d '{"classes":[{"id":1,"name":"Stale","color":"#0000ff"}],"objects":[],"version":<version_N>}'` → Expect: HTTP 409; response body is `{"error": "state_version_conflict", "current_version": <version_N+1>, "state": {<current full state>}}`. **Restore:** PUT the snapshotted state back using the current server version (`version_N+1`): reconstruct the original body from `state_pre_conflict.json` with `"version"` replaced by `<version_N+1>` and POST it: `curl -s -X PUT http://localhost:5555/api/session/state/<session_id> -H "Content-Type: application/json" -d '<original_state_body_with_version_N+1>'` → Expect: HTTP 200, `{"ok": true, "version": <version_N+2>}`. Restore method: final PUT with the server's current version — chosen because it uses the same idempotent route already exercised by the test and does not require filesystem access or a process restart; the version increments by one extra but the data is authoritative.
7. `[BROWSER]` With the session open in the browser, kill and restart the Flask backend (step 2). Within the next status-poll cycle (up to 10 s) → Expect: the status bar shows "Session lost — server restarted. Please select a session."; the annotation UI is replaced by the session list; the session list is functional (sessions appear, Resume works). (NOT RUN if browser automation is unavailable — append to HUMAN checklist.)

---

## Tier 2 flows

Shallow coverage for flows not fully elaborated in Tier 1. One row each; deep verification is referenced to the Tier-1 entry that owns it.

| ID | Flow | Contract (one line) | Invariant to check (one line) |
|---|---|---|---|
| UF-1.3 | Import session entry point | `VideoUpload.tsx` shows an Import section with a zip file picker and an Import button; clicking it calls `importSession` in `api.ts` (XHR upload to `POST /api/export/import-session`), then auto-resumes the new session via `POST /api/session/resume/<new_id>`; deep coverage in UF-8.2. | Button is enabled only after a `.zip` file is chosen; a second import while one is in-flight shows progress %, not a second trigger. |
| UF-1.4 | Delete session | Clicking the trash icon on a session card calls `DELETE /api/video/sessions/<session_id>`; the backend removes the session directory from disk (and the GCS prefix in cloud mode); the frontend evicts any cached frames from IndexedDB via `evictSession` in `maskCache.ts` and removes the card from the list. | Deleting session A must not touch the session directories or IndexedDB entries of any other session. |
| UF-1.6 | Cancel extraction/init job | The Cancel button in `PipelineProgress.tsx` calls `POST /api/job/cancel`; the backend signals the running pipeline to stop and `ServiceState` returns to `idle`; the frontend shows the session list. | After cancel, `GET /api/status` returns `phase: "idle"` within the poll budget (N8); no half-extracted session blocks a subsequent upload. |
| UF-1.7 | Password gate (cloud deploys) | When the backend has `AUTH_PASSWORD` set, any 401 response triggers `onUnauthorized` in `api.ts`; `App.tsx` renders `PasswordGate.tsx`, which verifies the entered password against `GET /api/health` (distinguishing wrong password from unreachable server), persists it to `localStorage["sam3-auth-token"]` via `setAuthToken`, then calls `refetchStatus` to resume normal routing. All requests (fetch via `apiFetch`, both XHR uploads, the keepalive flush beacon) carry `X-Auth-Token`. | The gate never appears in local dev (no `AUTH_PASSWORD` ⇒ no 401s); a wrong password shows an inline error without storing anything; the password is sent only as a header, never in a URL (N11). |
| UF-2.1 | Create class | `POST /api/session/classes/<session_id>` with `{name, color}`; `add_class` in `session_manager.py` appends a new entry with the next sequential integer `id` and returns it; the frontend adds it to React state via `persistState`. | The new class's `id` equals `max(existing ids) + 1` (or 1 if no classes exist); `state.json` on disk reflects the addition. |
| UF-2.2 | Rename class | `PUT /api/session/classes/<session_id>/<class_id>` with `{name}`; `update_class` in `session_manager.py` updates only the `name` field. | All other class fields (id, color) and all object `class_id` references are unchanged in `state.json`. |
| UF-2.3 | Change class color | `PUT /api/session/classes/<session_id>/<class_id>` with `{color}`; the canvas and sidebar immediately re-render masks and labels in the new color. | Color update does not alter mask data or prompt data; only `state.json` changes. |
| UF-2.4 | Toggle class visibility | `handleToggleClassVisibility` in `App.tsx` toggles the class `id` in `hiddenClassIds` state; hidden ⇒ masks are not rendered and objects of that class are deselected from `selectedObjIds`. | Toggling visibility does not write to disk and does not alter `masks.json`, `state.json`, or `prompts.json`. |
| UF-2.5 | Delete class | `DELETE /api/session/classes/<session_id>/<class_id>`; `delete_class` in `session_manager.py` removes the class and all its objects from `state.json`; the frontend then calls `POST /api/segment/remove_object` for each deleted object (N2, N4). | Objects belonging to the deleted class must have their masks and prompts removed from disk and their inference state cleared in SAM3. |
| UF-2.6 | Select class | `selectClass` in `App.tsx` sets `selectedClassId`, clears `selectedObjIds`, resets `toolMode` to `"pointer"`, and unhides the class if it was hidden. | Selecting a class does not trigger any backend call and does not modify any persisted state. |
| UF-2.7 | Auto-create object on first click for an empty class | When the selected class has no objects and the user clicks the canvas with the Click or Box tool, a new object is created automatically without an ObjectChoiceDialog; covered in UF-3.1 Contract. | Cross-ref UF-3.1 — no additional verification needed here. |
| UF-2.8 | Select object (single) | `selectObj` in `App.tsx` sets `selectedObjIds` to `{id}` and `selectedClassId` to the object's class; if the object's class is deleted (`class_id === -1`), the ClassChoiceDialog opens (UF-2.13); `toolMode` resets to `"pointer"` (N4). | Selecting object A deselects any previously selected object; backend inference state is not changed by selection alone — it is loaded on the next segmentation/propagation action. |
| UF-2.9 | Multi-select via Shift+click | `toggleObjSelection` in `App.tsx` adds/removes the clicked object from `selectedObjIds`; the PropagationBar shows one colored chip per selected object; covered in UF-4.2. | Cross-ref UF-4.2 — chips appear for each selected object; selection persists after propagation. |
| UF-2.10 | Clear selection (Esc with 2+ selected / click empty canvas) | Pressing Esc when `selectedObjIds.size >= 2` clears multi-select back to a single-object (or no) selection; clicking empty canvas area with the pointer tool deselects everything. | After clear, `selectedObjIds` is empty (or size ≤ 1 on single-Esc); no backend call is made. |
| UF-2.11 | Delete object across all frames | `handleObjectDeleted` in `App.tsx` calls `POST /api/segment/remove_object` with `{session_id, obj_id}`; the backend calls `sam.remove_object` (clears inference state, N4), `remove_object_masks` (deletes all frames in `masks.json`, N2), and `delete_object_prompts` (deletes all prompts in `prompts.json`). | No other object's masks, prompts, or class references are altered (N1); `GET /api/segment/propagation-status` must not report the deleted object in any subsequent status. |
| UF-2.12 | Reassign object to another class | `PUT /api/session/objects/<session_id>/<obj_id>/reassign` with `{class_id}`; `reassign_object` in `session_manager.py` updates the object's `class_id` in `state.json`; the frontend updates React state and re-renders mask overlays in the new class color. | The object's masks and prompts on disk are unaffected; only `state.json` changes. |
| UF-2.13 | Orphan-object reassignment dialog | When an object's class is deleted, selecting that object (by clicking its mask on canvas or its entry in the sidebar) opens the `ClassChoiceDialog`; the user picks a class, which calls `PUT …/reassign` (UF-2.12) to persist; or creates a new class first. | If the user dismisses the dialog without choosing, the object remains with `class_id === -1` in React state and `selectObj` returns without setting a valid `selectedClassId`. |
| UF-3.4 | Recalculate keyframe mask | `POST /api/segment/recalculate` re-runs SAM3 inference on the current frame using the object's existing prompt from `prompts.json`, then overwrites the frame's mask in `masks.json`; the frontend updates the mask overlay. | The recalculation does not alter other objects' masks (N1) and does not modify `prompts.json`. |
| UF-3.5 | Erase prompt via Missing Mask panel | The Missing Mask panel (visible when a prompt exists in React state but no mask is rendered for the selected object on the current frame) shows an "Erase prompt" button; clicking it calls `DELETE /api/session/masks/<session_id>/<frame_idx>/<obj_id>` which removes both the mask and prompt entry for that frame; the frontend clears local mask and prompt state. | After erase, the Missing Mask panel disappears for that frame; `masks.json` and `prompts.json` have no entry for that object/frame; no other object's data is touched (N1). |
| UF-4.5 | Confidence-warning review | After propagation, frames with `confidence < 0.75` get amber timeline ticks and accumulate in `confidenceWarnings`; a banner shows the count and a single "Go to frame N" shortcut (N = first/lowest-index amber-ticked frame); cross-ref UF-4.1 Verify step 7. | Cross-ref UF-4.1 — banner appears only if at least one frame is below threshold; "Go to frame" navigates to that frame without triggering any API call. |
| UF-5.2 | Zoom-to-mask + bbox padding sliders | Clicking the inspect toggle in `VideoCanvas.tsx` calls `zoomToMask`, which saves current zoom/pan and centers the view on the selected object's bbox + padding; the four padding sliders in `App.tsx` (top/bottom/left/right, −5–150%) update `bboxPadding` state keyed by `[obj_id][source_keyframe]`; the 500 ms debounced auto-save persists to `state.json`; padding affects export only (cross-ref UF-8.1). | Padding is stored per source-keyframe (not per frame); adjusting padding on one object must not alter another object's padding entries in `state.json`. |
| UF-6.1 | Play/pause + playback FPS selector | The play/pause button in `FrameNavigator.tsx` toggles `isPlaying`; while playing, a `setInterval` at `1000 / playbackFps` ms advances `currentFrame`; clicking the FPS label cycles through `FPS_OPTIONS`. | Play/pause does not trigger any API call; playback stops automatically at the last frame. |
| UF-6.2 | Timeline navigation (slider / arrow keys; tick colors) | Dragging the slider or pressing left/right arrow keys changes `currentFrame`; ticks are amber (`#f59e0b`, 80% opacity) for low-confidence frames and the selected-object color (or `#94a3b8` fallback) for normal inferred frames; keyframe positions are marked with a distinct indicator in `FrameNavigator.tsx`. | Slider and arrow key navigation do not trigger API calls; tick color reflects `lowConfidenceFrames` state, not disk data. |
| UF-6.3 | Prev/next mask jump | The `ChevronsLeft`/`ChevronsRight` buttons (shown when an object is selected) call `onPrevMask`/`onNextMask` in `App.tsx`, which jump to the nearest frame in `selectedObjFrames` before/after `currentFrame`. | Jump buttons are absent when no object is selected; they do not trigger API calls. |
| UF-7.1 | Auto-save state | `bboxPadding` changes are debounced 500 ms then flushed via `PUT /api/session/state/<session_id>` with a version token; `persistState` is also called synchronously on class/object mutations; the `stateIsSafeToPersist` guard (N5) blocks saves when the wipe fingerprint is detected. | Auto-save is gated by `sessionLoadedRef` — it must not fire before both state and mask loads have succeeded (cross-ref UF-1.2 Must NOT); `state.json` on disk must never contain the wipe fingerprint (N5). |
| UF-7.2 | beforeunload/pagehide flush beacon | `window.addEventListener("beforeunload", ...)` and `"pagehide"` both call `flushSyncBeacon` from `api.ts`, which posts `POST /api/segment/flush` with `keepalive: true`; the backend promotes deferred masks and attempts one GCS flush. | The beacon fires for both events (Safari/iOS fires `pagehide` instead of `beforeunload` with bfcache); a failed flush writes an unsynced marker but does not throw (N6). |
| UF-9.1 | Theme toggle | `ThemeToggle.tsx` reads `localStorage.getItem("theme")` on mount (falling back to `window.matchMedia("(prefers-color-scheme: dark)")`), and toggles between `"light"` and `"dark"` by setting `document.documentElement.classList` and writing to `localStorage`. | Theme persists across page reloads; system preference is honored when no stored value exists; toggle does not trigger any API call. |
| UF-10.1 | Zoom & pan | Wheel/trackpad scroll zooms (ctrl+wheel or mouse-wheel with no `deltaX`) or pans (two-finger scroll); middle-click or Alt+left-click activates drag-pan mode in `VideoCanvas.tsx`. | Zooming and panning do not trigger API calls; manual zoom/pan exits inspect mode if it was active. |
| UF-10.2 | Click mask to select / Shift+click toggle | Left-clicking a mask pixel on the canvas with the pointer tool selects that object (calls `selectObj`); Shift+left-click toggles it in/out of `selectedObjIds`; cross-ref UF-2.8/2.9. | Cross-ref UF-2.8 and UF-2.9 — canvas click-to-select uses `hitTestMask` (column-major RLE, LEB128 delta encoding); it must not misidentify which object was hit when masks overlap. |
| UF-11.1 | Frame caching to IndexedDB | After session load, `cacheSession` in `frameCache.ts` downloads all frame JPEGs in the background and stores them in IndexedDB; a progress indicator is shown while `cachingFrames` is true; the canvas reads frames from IDB to avoid repeat network fetches. | Caching runs in the background and must not block annotation; the progress indicator disappears when all frames are cached or on error. |
| UF-11.2 | Mask caching (versioned smart load) | `loadSessionMasksSmart` in `api.ts` fetches per-frame version numbers, compares them against IDB-cached versions via `findStaleFrames`, and fetches only stale/missing frames (full fetch if >30% stale or <20 frames); results are stored in IDB via `putMasks`. | Cached masks must never serve a version older than the server's current version; the version key in IDB matches the server's `versions` response. |
| UF-11.3 | Refresh masks button | Clicking the refresh button calls `handleRefreshMasks` in `App.tsx`, which calls `evictSession` to drop the IDB cache for the session, then calls `loadSessionMasksSmart` to re-fetch all masks; the status bar shows `"Refreshed: N frames with masks"`. | Refresh must evict only the current session's IDB entries, not other sessions'; the mask count in the status message equals the number of frames that have at least one mask on disk. |

---

## Impact map

After any change, look up every touched file below, collect the listed flow IDs and invariants, and re-verify them.

| Code touched | Re-verify flows | Invariants |
|---|---|---|
| `backend/app/services/sam3_service.py`, `backend/app/routes/segment.py` | UF-3.1–3.5, UF-4.1–4.4 | N1–N4, N9, N10 |
| `backend/app/services/mask_storage.py`, `backend/app/services/prompt_storage.py` | UF-3.1–3.5, UF-4.1–4.4, UF-5.1, UF-5.2, UF-8.1, UF-8.2 | N2, N7 |
| `backend/app/routes/session.py`, `backend/app/services/session_manager.py` | UF-1.2, UF-2.1–2.13, UF-7.1, UF-7.2, UF-12 | N5, N8 |
| `backend/app/services/gcs_sync.py`, `backend/app/services/gcs_storage.py` (cloud-mode paths) | UF-1.2, UF-1.4, UF-1.5, UF-7.2, UF-11.1, UF-11.2, UF-11.3 | N6 — plus CLAUDE.md "Cloud mode blind spots" |
| `backend/app/services/pipeline.py`, `backend/app/routes/video.py` | UF-1.1, UF-1.4, UF-12 | N8 |
| `backend/app/routes/export.py`, `backend/app/services/exporter.py`, `backend/app/services/session_bundle.py` | UF-8.1, UF-8.2, UF-1.3 | N7 (note the recorded violation in N7) |
| `backend/app/routes/status.py`, `backend/app/services/pipeline.py` (ServiceState) | UF-12, UF-1.1, UF-4.4 | N8, N9 — `POST /api/job/cancel` and `POST /api/status/dismiss-error` both live here |
| `backend/app/services/session_lock.py` | All flows that use `session_io_lock` (UF-3.x, UF-4.x, UF-7.x, UF-8.x) | N8 — a deadlock here hangs the entire service |
| `backend/app/services/atomic_write.py` | UF-12, UF-7.1 | N5, N6 — `atomic_json_dump` is the sole write path for all JSON session files; a restart during a write must not corrupt files |
| `backend/app/services/unsynced_marker.py` | UF-1.5, UF-7.2 | N6 — retry dialog and GCS sync failure path |
| `backend/app/services/video_processor.py` | UF-1.1 (frame extraction step) | N8 — extraction failure must transition to `"error"`, not hang |
| `backend/app/services/session_cache.py` | UF-1.2, UF-2.x, UF-3.x (any route passing `cache=get_session_cache()`) | N5 — stale cache entries must not cause a stale-state read that bypasses the version guard |
| `frontend/src/App.tsx` | All `[BROWSER]` steps; minimally UF-1.1, UF-1.2, UF-1.5, UF-2.x, UF-4.x, UF-7.1, UF-7.2, UF-12 | N5, N6 — `stateIsSafeToPersist`, `sessionLoadedRef`, and `reconcileFromServer` all live here |
| `frontend/src/api.ts` | All flows matching the changed endpoint(s) | — verify the Tier-1/Tier-2 rows for every modified function |
| `frontend/src/components/VideoCanvas.tsx` | UF-3.1, UF-3.2, UF-5.2, UF-10.1, UF-10.2 | — `hitTestMask` (column-major RLE), zoom/pan bindings, inspect mode |
| `frontend/src/components/AnnotationPanel.tsx` | UF-2.1–2.13, UF-3.x | — class and object list rendering, visibility toggles |
| `frontend/src/components/PropagationBar.tsx`, `frontend/src/components/ToolBar.tsx` | UF-3.x, UF-4.1–4.4 | N9 — cancel button wires to both client-side abort and `POST .../cancel` |
| `frontend/src/components/FrameNavigator.tsx`, `frontend/src/components/DeleteMasksPanel.tsx` | UF-5.1, UF-5.2, UF-6.1–6.3 | — tick colors, prev/next mask jump, play/pause, batch delete UI |
| `frontend/src/components/VideoUpload.tsx`, `frontend/src/components/ExportDialog.tsx`, `frontend/src/components/ExportSessionDialog.tsx` | UF-1.1, UF-1.3, UF-1.4, UF-8.1, UF-8.2 | — import/export entry points, delete session button |
| `frontend/src/maskCache.ts`, `frontend/src/frameCache.ts` | UF-11.1, UF-11.2, UF-11.3, UF-1.2 | — IDB version keys, evict-session scope, `loadSessionMasksSmart` stale-ratio logic |
| `frontend/src/hooks/useServiceStatus.ts` | UF-12, UF-1.1, UF-1.2 | N8 — polling interval changes affect session-loss detection latency |
| `backend/app/__init__.py` (auth hook), `frontend/src/components/PasswordGate.tsx` | UF-1.7 — plus a spot-check that one authenticated flow still works end-to-end (e.g. UF-1.1) | N11 |
| `deploy/deploy.sh`, `cloudbuild-native.yaml`, `DEPLOY.md`, `deploy/README.md` | Deploy smoke set (after the next real deploy) | N11 — `smoke` asserts the 401-without-token contract |
| `frontend/src/components/ThemeToggle.tsx` | UF-9.1 | — localStorage key, system-preference fallback |
| `frontend/src/components/ClassChoiceDialog.tsx` | UF-2.13 | — orphan-object reassignment dialog |

| `backend/app/routes/benchmark.py` | — (internal perf testing; no user flow) | — |
| `frontend/src/components/PipelineProgress.tsx` | UF-1.1, UF-1.6 | N8 — progress display + job cancel button |
| `frontend/src/components/ObjectChoiceDialog.tsx`, `ConfirmDialog.tsx`, `ErrorDialog.tsx`, `StatusToast.tsx`, `TrackBarPreview.tsx` | UF-3.1, UF-3.2 (object choice); UF-1.5, UF-2.x (confirm); UF-12 (error display); UF-5.1 (preview thumbnails) | — |

A file not listed here ⇒ add a row in the same PR (Maintenance rule 4).

---

## Deploy smoke set

Run after every deploy, against the deployed URL — step 0 is automated: `./deploy/deploy.sh smoke`. For manual curls, every request needs `-H "X-Auth-Token: $(./deploy/deploy.sh password)"`.

| # | Flow(s) | What it verifies |
|---|---|---|
| 0 | UF-1.7 | `./deploy/deploy.sh smoke` — unauthenticated `GET /api/health` returns 401; with `X-Auth-Token`, health returns `ok` and `GET /api/status` returns `{"phase": "idle", ...}` |
| 1 | UF-1.1 | Upload a new video via the UI; progress bar advances through extraction and initialization; annotation UI appears |
| 2 | UF-1.1 (tab-close during extraction) | Close the tab during extraction, reopen — progress bar resumes at the correct phase |
| 3 | UF-1.2 (tab-close in annotation UI) | Close the tab while in the annotation UI, reopen — auto-resumes into the annotation UI |
| 4 | UF-1.2 | Click Resume on an existing session from the list; cloud: GCS download + init; annotation UI renders with prior masks |
| 5 | UF-3.1 + UF-4.1 | Click an object on the keyframe; propagate forward; masks appear frame-by-frame |
| 6 | UF-4.2 | Shift+click two objects; propagate — both objects tracked simultaneously |
| 7 | UF-1.6 | Click Cancel during model init (`POST /api/job/cancel`); returns to session list; `GET /api/status` shows idle |
| 8 | UF-1.5 | Close session; `GET /api/status` returns `{"phase": "idle", "session_id": null}` |
| 9 | UF-1.7 | Open the URL in a fresh browser profile — the password gate appears; a wrong password shows an inline error; the printed password unlocks and survives a reload |

---

## Scenario tests (S1–S5)

End-to-end scenarios that compose Tier-1 flows into the user journeys that matter most.
Run them against the committed fixture (import `tests/fixtures/harness/golden_session.zip`
per UF-8.2 to get a known-good session). Each scenario references the flows it composes —
the per-step mechanics (exact curls, payloads, expected bodies) live in those flow entries;
the scenario adds the journey-level expectation. Scenarios with `[BROWSER]` legs follow the
degradation rule when no browser tool is available.

### S1 — Propagation survives a UI disconnect

*Composes: UF-4.1, UF-4.3.*

1. `[BROWSER]` Import + resume the golden session; select object 1; start Forward propagation from frame 0. → Expect: masks advance live on the timeline.
2. `[BROWSER]` After ~3 frames have ticked, close the tab (no session close — just kill the tab).
3. `[API]` Confirm the backend kept going: poll `GET /api/segment/propagation-status/<session_id>` → Expect: `running` with `frames_processed` still increasing.
4. `[BROWSER]` Reopen `http://localhost:5173`. → Expect: auto-resumes into the annotation UI, reconnects to the stream (UF-4.3), progress continues from the live frame — not from 0.
5. `[DISK]` After completion: obj 1 has masks on frames 0–39, contiguous, exactly once (the disk gap-check from UF-4.3 step 4 is the completeness authority — frames processed while disconnected appear in no event log).

Verified 2026-06-10 at API level (drop SSE after 3 events → re-subscribe → contiguous 0–39 on disk). `[BROWSER]` legs pending browser tooling.

### S2 — Selection scopes propagation: only the selected mask propagates

*Composes: UF-4.1, global invariant N1.*

The golden session has two annotated objects. Propagating one must not touch the other —
this is the harness's headline invariant (the SAM2-era bug that silently corrupted
non-target masks).

1. `[DISK]` Snapshot `masks.json`.
2. `[BROWSER]` Select ONLY object 1 (single click on its mask — no Shift). PropagationBar shows one chip. Propagate Forward.
3. `[API]` Every SSE FrameResult contains obj 1 and nothing else.
4. `[DISK]` Object 2's RLE strings byte-identical to the snapshot on every frame (N1).
5. Repeat with object 2 selected → object 1 untouched.

Verified 2026-06-10 at API level (40/40 events obj-1-only; obj 2 byte-identical). The
`[BROWSER]` leg additionally proves the UI's selection state is what reaches the backend (N4).

### S3 — Closing the session never loses data, in any situation

*Composes: UF-1.5, UF-7.1, UF-7.2, UF-12; global invariant N6.*

Run each variant from a resumed golden session with at least one fresh edit (e.g. nudge a
bbox padding slider). After each, resume and assert: `masks.json`, `prompts.json`
byte-identical to their pre-variant snapshots, and the fresh edit survived in `state.json`.

| Variant | Procedure | Expected survival mechanism |
|---|---|---|
| a. Clean close | Close button → confirm | UF-1.5: explicit flush before `close` |
| b. Close < 500 ms after a padding edit | Move slider, immediately close | `handleCloseSession` flushes the debounced save (N6) |
| c. Close during propagation | Start propagation, then close | Close cancels propagation first; frames persisted before cancel survive (UF-4.4) |
| d. Tab close, no session close | Kill the tab mid-session | `beforeunload`/`pagehide` beacon → `POST /api/segment/flush` (UF-7.2) |
| e. Backend killed mid-session | `pgrep -f "run.py"` (or `"flask.*5555"`) → kill all PIDs, restart | Atomic writes — no torn files (UF-12 steps 1–3) |
| f. Backend killed mid-propagation | Kill during an active propagation run | Frames persisted before the kill survive; the in-flight frame is lost (acceptable — never partially written); resume re-inits SAM3 |

Variants a, e verified 2026-06-10 (byte-identical after restart; clean close → idle).
b, d are `[BROWSER]`; c, f are `[API]`-runnable but not yet exercised — run them on the
next harness pass.

### S4 — Golden keyframe replay: same prompts ⇒ same masks ⇒ same tracks

*Composes: UF-3.1, UF-3.2, UF-4.1. This is the model-regression canary: if SAM3, the
dtype patches, or the prompt-replay path drift, this scenario degrades first.*

For each object in the golden session, individually:

1. `[DISK]` Read the object's prompt from the golden `prompts.json` (obj 1: click `[160, 296]`, label 1; obj 2: box `[168, 345, 225, 460]` — canonical copies in `fixture.json`).
2. `[API]` Batch-delete ALL of that object's masks (UF-5.1 route, all 40 frames). → Expect: object has zero masks; the other object untouched.
3. `[API]`/`[BROWSER]` Re-apply the prompt exactly as recorded (click at the same coords / draw the same box). → Expect: keyframe mask IoU ≥ 0.80 vs the golden keyframe mask (via `iou.py`; observed 1.00 on the build machine — same model + device is near-deterministic).
4. `[API]` Propagate Forward from frame 0. → Expect: completes for all 40 frames; on `fixture.json.propagation_sample_frames` (10, 25, 39), IoU ≥ 0.80 vs golden (observed 1.00).
5. Restore for the next object: the re-application + propagation IS the restore.

Verified 2026-06-10 for the click+delete+re-click path (IoU 1.00) and full propagation
(IoU 1.00 on all sample frames); the explicit per-object delete-all → replay → propagate
sweep is the canonical regression run going forward.

### S5 — Multi-select journey: detect, propagate together, reverse

*Composes: UF-3.3, UF-4.2; global invariants N1, N10.*

1. `[BROWSER]` On the golden keyframe, Shift+click both objects → two chips in the PropagationBar.
2. `[BROWSER]` Propagate Forward → both masks advance together in one pass; selection persists after completion.
3. `[BROWSER]` Immediately propagate Back (no re-selection) → reverse pass runs with the same set.
4. `[DISK]` `prompts.json` byte-identical throughout (N10); every selected object has masks on every propagated frame.

Multi-object forward propagation verified 2026-06-10 at API level (both objects in all
40 events, 0.96–0.99 confidence). Selection persistence and the reverse leg are `[BROWSER]`.

---

## Maintenance rules

1. A PR that changes user-facing behavior must update the affected flow entry in the same commit.
2. Every postmortem'd bug adds a Must NOT line to the relevant flow (or a global invariant) — the regression → invariant pipeline.
3. Flow IDs are stable; retired flows are marked deprecated, never deleted or renumbered.
4. The impact map gains a row whenever a new code area is created.
5. New flows get the next unused number in their range; new ranges append.
