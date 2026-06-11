# USER-FLOWS.md Verification Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create `USER-FLOWS.md` — a layered behavioral-contract + regression-checklist harness with negative invariants — plus a committed test fixture, wired into CLAUDE.md.

**Architecture:** Single markdown document at repo root with stable flow IDs (`UF-x.y`), global negative invariants (`N1–N10`), tagged verification steps (`[API]/[DISK]/[BROWSER]/[HUMAN]`), an impact map (code area → flows), and a fixture under `tests/fixtures/harness/`. Spec: `docs/superpowers/specs/2026-06-10-user-flows-harness-design.md`. Flow inventory source: the codebase exploration of 2026-06-10 (App.tsx, api.ts, backend routes).

**Tech Stack:** Markdown; curl against Flask `:5555`; ffmpeg; the running app (`make dev`, conda env `sam3-annotator`) for fixture generation.

**Prerequisite (user-provided):** A sample video, 5–10 s, any ffmpeg-readable format, 2–3 visually distinct persistent objects. Tasks 1–7 do NOT depend on it; Tasks 8–9 are blocked until it arrives.

**Flow ID scheme (fixed now, never renumbered):**

| Range | Area | Tier 1 entries |
|---|---|---|
| UF-1.x | Session lifecycle | 1.1 Upload, 1.2 Resume, 1.5 Close (incl. GCS retry) |
| UF-2.x | Class & object management | — (all Tier 2: 2.1 create class … 2.13 orphan reassignment) |
| UF-3.x | Segmentation | 3.1 Click+undo, 3.2 Box, 3.3 Text detect |
| UF-4.x | Propagation | 4.1 Single, 4.2 Multi, 4.3 Reconnect, 4.4 Cancel |
| UF-5.x | Mask deletion | 5.1 Batch delete by range |
| UF-6.x | Frame navigation/playback | — (Tier 2) |
| UF-7.x | Persistence & state sync | — (Tier 2; version conflict lives in UF-12) |
| UF-8.x | Export/import | 8.1 COCO export, 8.2 Session export/import round-trip |
| UF-9.x | Settings (theme) | — (Tier 2) |
| UF-10.x | Canvas interactions | — (Tier 2) |
| UF-11.x | Caching | — (Tier 2) |
| UF-12 | Recovery | 12 Recovery (container restart, error phase, state version conflict) — one Tier-1 entry |

Tier-1 total: 14 entries.

---

### Task 1: USER-FLOWS.md skeleton — usage loop, invariants, fixture contract, maintenance rules

> **COMPLETED** (commits dc20610 + d69b314). The step text below is the original
> draft; the committed USER-FLOWS.md supersedes it where they differ (quality-review
> amendments: literal [DISK] path, falsifiable N4/N8, iou.py, clearing procedure in N3,
> thresholds in N9). Do NOT re-execute this task from this text.

**Files:**
- Create: `USER-FLOWS.md` (repo root)

- [ ] **Step 1: Write the document skeleton with these sections, in this order**

```markdown
# USER-FLOWS.md — Verification Harness

> Behavioral contract + regression checklist for the SAM3 Video Labelling Tool.
> When code and this document disagree, this document wins — or is explicitly
> amended in the same commit that changes the behavior.

## How to use this harness
## Global negative invariants (N1–N10)
## Test fixture
## Tier 1 flows
## Tier 2 flows
## Impact map
## Deploy smoke set
## Maintenance rules
```

- [ ] **Step 2: Write "How to use this harness"**

Content (verbatim intent, wording may be polished):

```markdown
## How to use this harness

The loop, after any behavioral change:

1. List the files you touched.
2. Look each up in the **Impact map** → collect the flow IDs and global invariants.
3. Execute each flow's **Verify** steps in order. Tags say who runs them:
   - `[API]` — Claude: curl against Flask (`http://localhost:5555`)
   - `[DISK]` — Claude: inspect session files under the backend sessions dir
     (`masks.json`, `state.json`, `prompts.json`) or exported zips
   - `[BROWSER]` — Claude drives a real browser against `http://localhost:5173`
     using the fixture's canonical click coordinates
   - `[HUMAN]` — only a person can judge (mask visual quality, drag feel)
4. Check each flow's **Must NOT** lines, including every global invariant they reference.
5. Report per flow: PASS / FAIL (with evidence) / NOT RUN (with reason).
   Hand the user the remaining `[HUMAN]` checklist.

**Degradation rule:** if no browser-automation tool is available in the
session, each `[BROWSER]` step is reported NOT RUN and appended to the
`[HUMAN]` checklist — never silently skipped.

**Environment:** local dev (`make dev`: Flask :5555 + Vite :5173, conda env
`sam3-annotator`, MPS). Cloud-mode-only paths (GCS sync, Cloud Run lifecycle)
cannot be fully verified locally — see "Cloud mode blind spots" in CLAUDE.md;
mark those steps NOT RUN locally and use the Deploy smoke set after deploys.
```

- [ ] **Step 3: Write "Global negative invariants" — N1–N10 exactly as specified in the spec**

Copy the ten invariants from the spec (`docs/superpowers/specs/2026-06-10-user-flows-harness-design.md` § "Global negative invariants"), each with one added sentence stating its concrete check method:

- N1 check: byte-compare non-target objects' RLE strings in `masks.json` before/after propagation.
- N2 check: delete a mask, propagate across that frame, assert the mask is absent afterward.
- N3 check: same click coords on a frame with prior cleared state vs. a fresh object → IoU ≥ 0.99 between the two masks.
- N4 check: backend inference state holds only active object IDs (assert via `remove_object`/replay behavior; documented as a code-reading check on `sam3_service.py`).
- N5 check: grep persisted `state.json` after any save — never `{"classes": [], "objects": [...class_id < 0...]}`.
- N6 check: close with pending bboxPadding edit → reopen → padding survived; simulate GCS failure (cloud) → retry dialog appears.
- N7 check: COCO `instances.json` bboxes equal mask bbox + padding from `state.json`; image count == annotated-frame count.
- N8 check: `GET /api/status` phase matches actual service activity at every flow boundary.
- N9 check: after cancel, `GET /api/segment/propagation-status/{id}` reports not-running and a subsequent click responds within normal latency (lock released).
- N10 check: after multi-object propagation, every selected object has masks on new frames; `prompts.json` unchanged.

- [ ] **Step 4: Write "Test fixture" section**

```markdown
## Test fixture

tests/fixtures/harness/
├── sample.mp4            # 5–10 s, ≤640 px, 2–3 visually distinct objects
├── golden_session.zip    # known-good exported session (masks/prompts/state)
├── fixture.json          # canonical click coords per object on the keyframe,
│                         #   expected object count, IoU thresholds
└── README.md             # how goldens were made, how to regenerate

**Comparison rules:**
- SAM3 output vs. golden masks: **IoU ≥ 0.80** (SAM3 is not bit-exact across
  MPS/CUDA backends).
- "Untouched data didn't change" checks (e.g. N1): **byte equality** of RLE
  strings.
- `golden_session.zip` doubles as the import-flow test asset (UF-8.2).
```

- [ ] **Step 5: Write "Maintenance rules" section** — the four rules from the spec verbatim (PR updates flow entry in same commit; bug → Must NOT line; IDs stable/deprecate-never-delete; impact map row per new code area). Add rule 5: "New flows get the next unused number in their range; new ranges append."

- [ ] **Step 6: Leave `## Tier 1 flows`, `## Tier 2 flows`, `## Impact map`, `## Deploy smoke set` as headers with a one-line `<!-- populated in Tasks 2–6 -->` comment** (removed by Task 6).

- [ ] **Step 7: Commit**

```bash
git add USER-FLOWS.md
git commit -m "docs: USER-FLOWS.md skeleton — usage loop, invariants N1-N10, fixture contract"
```

---

### Task 2: Tier-1 group A — session lifecycle (UF-1.1, UF-1.2, UF-1.5)

> **COMPLETED** (see git history dc20610..e5ca25a). The committed USER-FLOWS.md/CLAUDE.md supersede this draft text where they differ (review amendments). Do NOT re-execute this task.


**Files:**
- Modify: `USER-FLOWS.md` (under `## Tier 1 flows`)

Every entry uses the anatomy: `### UF-x.y Name` → **Contract** (2–3 sentences) → **Must NOT** (bullets, referencing global invariants by ID) → **Verify** (numbered, tagged, each with `→ Expect:`).

- [ ] **Step 1: Write UF-1.1 Upload Video**

Content requirements (write as full prose entry):
- Contract: selecting a video + FPS (default 5) + max resolution (default 2048) and clicking Upload extracts frames, initializes SAM3, and lands in the annotation UI without user intervention. Closing the tab during extraction and reopening resumes the progress view at the correct phase.
- Must NOT: leave `/api/status` stuck in `extracting`/`initializing` after failure (N8); create a session directory without `meta.json`; allow a second concurrent upload (ServiceState is service-wide).
- Verify:
  1. `[API]` `curl -F video=@tests/fixtures/harness/sample.mp4 -F fps=5 http://localhost:5555/api/video/upload` → Expect: 200 with `session_id`.
  2. `[API]` poll `GET /api/status` → Expect: `extracting` → `initializing` → `ready`, monotonic progress.
  3. `[DISK]` session dir contains `meta.json` and `frames/` with count ≈ duration×fps.
  4. `[BROWSER]` upload via UI → progress bar advances through Upload/Extract/Load Model → annotation UI appears.
  5. `[HUMAN]` first frame renders correctly on the canvas.

- [ ] **Step 2: Write UF-1.2 Resume Session**

- Contract: clicking Resume on a session card re-initializes SAM3 for that session, reloads classes/objects/masks/prompts, reconnects to any in-progress propagation, and lands in the annotation UI with prior state intact.
- Must NOT: lose or mutate any persisted mask/prompt/class on resume (byte-compare `masks.json` before/after); permit auto-save before both state and mask loads succeed (N5 — the `sessionLoadedRef` guard); silently drop orphaned masks (objects with deleted classes appear as `class_id=-1` synthetics, UF-2.13).
- Verify:
  1. `[API]` `POST /api/session/resume/{sessionId}` → Expect: 200; status reaches `ready`.
  2. `[API]` `GET /api/session/state/{id}`, `GET /api/session/masks/{id}`, `GET /api/session/prompts/{id}` → Expect: classes/objects/masks/prompts match pre-close values.
  3. `[BROWSER]` resume golden session from list → object list shows fixture's expected object count; masks render on keyframe.
  4. `[DISK]` `masks.json` byte-identical to its pre-resume content.

- [ ] **Step 3: Write UF-1.5 Close Session (incl. GCS retry path)**

- Contract: closing flushes pending debounced saves, cancels in-flight propagation, releases SAM3 inference state, and returns to the session list. In cloud mode, GCS sync failure surfaces a retry dialog (Retry / Close anyway); dismissing keeps the session open.
- Must NOT: lose a bboxPadding edit made <500 ms before close (N6 — explicit flush in `handleCloseSession`); exit silently on GCS failure (N6); leave `/api/status` ≠ `idle` after close (N8); leave propagation thread holding `SAM3Service._lock` (N9).
- Verify:
  1. `[BROWSER]` adjust a bbox padding slider, immediately close session → resume → Expect: padding value survived.
  2. `[API]` `POST /api/segment/close/{id}` → Expect: 200; `GET /api/status` → `idle`.
  3. `[API]` after close, `POST /api/segment/click` for that session → Expect: clean error (no SAM3 state), not a hang.
  4. `[HUMAN]` (cloud only) close during a forced GCS outage → retry dialog with file list appears; "Retry upload" loops; "Close anyway" closes.

- [ ] **Step 4: Commit**

```bash
git add USER-FLOWS.md
git commit -m "docs: harness Tier-1 group A — session lifecycle flows UF-1.x"
```

---

### Task 3: Tier-1 group B — segmentation (UF-3.1, UF-3.2, UF-3.3)

> **COMPLETED** (see git history dc20610..e5ca25a). The committed USER-FLOWS.md/CLAUDE.md supersede this draft text where they differ (review amendments). Do NOT re-execute this task.


**Files:**
- Modify: `USER-FLOWS.md`

- [ ] **Step 1: Write UF-3.1 Click Segmentation + Undo**

- Contract: with a class selected, left-click adds a positive point, right-click a negative point; each click refines the mask. Ctrl/Cmd+Z removes the last point and re-runs SAM3; undoing the only point removes the mask. If the class has multiple objects, an ObjectChoiceDialog asks which (or new).
- Must NOT: a correction click after clear behaving differently from a first-time click (N3); a click on object A mutating object B's mask in `masks.json` (N1 spirit at click scope); a failed inference persisting a prompt without a mask silently (must surface the "Missing Mask" panel, UF-2.x/Tier 2).
- Verify:
  1. `[API]` `POST /api/segment/click` with fixture's canonical coords for object 1 → Expect: 200, `masks.{obj_id}` present, RLE decodes, IoU ≥ 0.80 vs. golden keyframe mask.
  2. `[DISK]` `prompts.json` gained one click prompt; `masks.json` gained the frame/object entry; other objects' entries byte-identical.
  3. `[API]` repeat the same click on a fresh object → IoU ≥ 0.99 between the two results (N3).
  4. `[BROWSER]` click fixture coords on canvas → mask overlay appears in class color; Ctrl+Z → point dot disappears, mask updates.
  5. `[HUMAN]` mask hugs the object boundary.

- [ ] **Step 2: Write UF-3.2 Box Segmentation**

- Contract: with Box tool active, dragging a rectangle segments the dominant object inside it; same object-choice logic as clicks.
- Must NOT: zero-area boxes reaching the backend; box prompt overwriting other objects' masks (byte-compare); prompt type recorded as anything but `box` in `prompts.json`.
- Verify:
  1. `[API]` `POST /api/segment/box` with fixture's canonical box for object 2 → Expect: 200, mask IoU ≥ 0.80 vs. golden.
  2. `[DISK]` `prompts.json` entry has `type: "box"` with the submitted coords.
  3. `[BROWSER]` drag fixture box on canvas → preview rectangle while dragging, mask on release.

- [ ] **Step 3: Write UF-3.3 Text Detection**

- Contract: Detect tool + text query runs whole-frame multi-instance detection; each instance gets a sequential obj_id, all auto-assigned to the matching/selected/new class, and all auto-selected (multi-select) ready to propagate. Cancel/Escape aborts.
- Must NOT: detected instances clobbering existing obj_ids (must start at `obj_id_start` = max+1); detection mutating existing masks (byte-compare); leaving partial objects registered after abort.
- Verify:
  1. `[API]` `POST /api/segment/text` with fixture's `text_query` → Expect: instance count == `fixture.json.expected_text_detections`.
  2. `[DISK]` new objects appended in `state.json` with correct class; pre-existing masks byte-identical.
  3. `[BROWSER]` run Detect from toolbar → status shows "Found N … all selected"; PropagationBar shows N chips.

- [ ] **Step 4: Commit**

```bash
git add USER-FLOWS.md
git commit -m "docs: harness Tier-1 group B — segmentation flows UF-3.x"
```

---

### Task 4: Tier-1 group C — propagation (UF-4.1–4.4)

> **COMPLETED** (see git history dc20610..e5ca25a). The committed USER-FLOWS.md/CLAUDE.md supersede this draft text where they differ (review amendments). Do NOT re-execute this task.


**Files:**
- Modify: `USER-FLOWS.md`

- [ ] **Step 1: Write UF-4.1 Single-Object Propagation**

- Contract: with one object selected and a mask on the current frame, Back/Both/Forward tracks it across frames, streaming masks live over SSE; frames with confidence < 0.75 are flagged (amber banner + timeline ticks) with a "Go to frame" shortcut.
- Must NOT: create/modify/delete masks for any non-target object (N1 — byte-compare); regenerate previously deleted masks (N2); report `propagation-status` running after completion (N8/N9); lose the keyframe's `source_keyframe` metadata.
- Verify:
  1. `[API]` `POST /api/segment/propagate {start_frame_idx, reverse:false, object_ids:[1]}` and consume SSE → Expect: FrameResults for all frames ahead; only obj 1 present in every event.
  2. `[DISK]` obj 2's pre-existing RLEs byte-identical before/after (N1); deleted-mask frame from the N2 setup stays empty.
  3. `[API]` propagated masks vs. golden propagation: IoU ≥ 0.80 on sampled frames listed in `fixture.json`.
  4. `[BROWSER]` timeline gains green ticks; amber ticks iff low-confidence frames exist.
  5. `[HUMAN]` spot-check 3 frames for tracking quality.

- [ ] **Step 2: Write UF-4.2 Multi-Object Propagation**

- Contract: Shift+click builds a multi-select set (chips in PropagationBar); propagation resets SAM3 inference state, replays ALL selected objects' prompts from disk, and tracks them in one pass. Selection persists afterward for immediate reverse propagation.
- Must NOT: drop any selected object during reset+replay (N10 — every selected object has masks on new frames); touch unselected objects' masks (N1); mutate `prompts.json` during replay; incrementally add objects post-tracking (native SAM3 raises — must reset first; see CLAUDE.md SAM3 constraints).
- Verify:
  1. `[API]` propagate `object_ids:[1,2]` → Expect: every SSE FrameResult contains both obj 1 and obj 2.
  2. `[DISK]` `prompts.json` byte-identical before/after; obj 3 (unselected, if present) masks byte-identical.
  3. `[BROWSER]` Shift+click two objects → two chips; propagate → both masks advance together; selection persists after completion.

- [ ] **Step 3: Write UF-4.3 Propagation Reconnect**

- Contract: closing the tab mid-propagation does not stop it; the backend continues and persists masks. Reopening reconnects to the SSE stream, backfills missed frames, and shows live progress.
- Must NOT: duplicate or skip frames on reconnect (frame indices in `masks.json` contiguous over the propagated range); spawn a second propagation on reconnect.
- Verify:
  1. `[API]` start propagation, drop the SSE connection after ~3 events, reconnect via `GET /api/segment/propagate/subscribe/{id}` → Expect: stream resumes; final `masks.json` covers the full range exactly once.
  2. `[BROWSER]` close tab mid-propagation, reopen `:5173` → progress resumes; mask count matches a never-disconnected run.

- [ ] **Step 4: Write UF-4.4 Cancel Propagation**

- Contract: Stop aborts the client stream and best-effort cancels the backend; completed frames keep their masks; the UI returns to an interactive state promptly.
- Must NOT: leave the propagation thread holding `_lock` (N9 — next click must respond at normal latency); roll back already-persisted frames; leave status `running`.
- Verify:
  1. `[API]` start propagation, `POST /api/segment/propagate/cancel/{id}` after ~2 events → Expect: `propagation-status` not-running within one frame's inference time.
  2. `[API]` immediately `POST /api/segment/click` → Expect: normal-latency 200 (lock released, N9).
  3. `[DISK]` frames processed before cancel retain masks.

- [ ] **Step 5: Commit**

```bash
git add USER-FLOWS.md
git commit -m "docs: harness Tier-1 group C — propagation flows UF-4.x"
```

---

### Task 5: Tier-1 group D — deletion, export, recovery (UF-5.1, UF-8.1, UF-8.2, UF-12)

> **COMPLETED** (see git history dc20610..e5ca25a). The committed USER-FLOWS.md/CLAUDE.md supersede this draft text where they differ (review amendments). Do NOT re-execute this task.


**Files:**
- Modify: `USER-FLOWS.md`

- [ ] **Step 1: Write UF-5.1 Batch Mask Deletion**

- Contract: Delete (button or Delete/Backspace key) opens the DeleteMasksPanel for the selected object; choosing a start/end range deletes those frames' masks for that object only.
- Must NOT: deleted masks regenerating on the next propagation (N2 — the in-memory `clear_frame_object` rule); deletion touching other objects' masks or other frames (byte-compare outside the range); prompts surviving for fully-deleted keyframes without surfacing the "Missing Mask"/erase affordance.
- Verify:
  1. `[API]` `DELETE /api/session/masks/{id}/{objId}/batch` with a 5-frame range → Expect: 200; those frames absent for that object in `masks.json`; all other entries byte-identical.
  2. `[API]` propagate across the deleted range → Expect: deleted frames repopulate ONLY because propagation legitimately re-tracks (document: post-delete propagation regenerating masks from a *remaining* keyframe is correct; regenerating from stale in-memory state of the *deleted* mask is the N2 violation — distinguish by deleting the keyframe itself and confirming no mask reappears at it).
  3. `[BROWSER]` Delete key with object selected → panel opens with thumbnails; range delete updates timeline ticks.

- [ ] **Step 2: Write UF-8.1 COCO Export**

- Contract: Export Annotations downloads `{videoName}_coco.zip` containing `instances.json` (images/annotations/categories) and `images/` with only annotated frames; bbox padding from `state.json` is applied to exported bboxes.
- Must NOT: include unannotated frames (N7); export bboxes ignoring saved padding (N7); leak absolute server paths in `instances.json`.
- Verify:
  1. `[API]` `POST /api/export/coco/{id}` → Expect: 200 zip; unzip: image count == annotated-frame count; every annotation's `category_id` maps to a real class.
  2. `[DISK]` for one fixture frame with known padding: exported bbox == mask tight bbox expanded by the padding in `state.json`.
  3. `[API]` `instances.json` parses; RLE/polygon decodes to the mask's area ±1 %.

- [ ] **Step 3: Write UF-8.2 Session Export/Import Round-Trip**

- Contract: Export Session downloads a portable zip (meta/state/masks/prompts, optional frames+video); importing that zip recreates an equivalent session (deduped by video MD5) that resumes into the annotation UI.
- Must NOT: round-trip lose or mutate any mask/prompt/class (export → import → compare `masks.json`, `prompts.json`, classes/objects byte/structurally identical); import execute files from the zip beyond the expected manifest.
- Verify:
  1. `[API]` `POST /api/export/session/{id} {include_video:true}` → Expect: zip with `meta.json`, `state.json`, `masks.json`, `prompts.json`.
  2. `[API]` `POST /api/export/import-session` with that zip → Expect: 200 with `session_id`; new session's files structurally equal to source.
  3. `[BROWSER]` Import golden_session.zip from the session list → session appears and resumes with fixture's expected objects/masks.

- [ ] **Step 4: Write UF-12 Recovery (container restart / error phase / version conflict)**

- Contract: (a) if the backend restarts, the frontend detects `ready→idle`, shows "Session lost — server restarted", and the session list resumes cleanly; (b) a failed pipeline lands in `error` phase with a message, dismissible via `POST /api/status/dismiss-error` back to `idle`; (c) concurrent edits from two tabs hit the `state.json` version guard — loser gets 409, reloads server state, and shows "Another tab updated this session".
- Must NOT: a restart corrupting persisted session files (resume after restart shows identical masks/state); dismiss-error leaving phase ≠ `idle` (N8); a 409 loser silently overwriting the winner's state (last-writer-wins must be the SERVER's resolution, with the wipe-fingerprint guard N5 still enforced).
- Verify:
  1. `[API]` kill and restart Flask mid-session → `GET /api/status` → `idle`; resume → masks byte-identical to pre-kill.
  2. `[API]` `PUT /api/session/state/{id}` twice with the same stale `version` → Expect: second gets 409 carrying server state.
  3. `[API]` upload an invalid video file → status `error` with message; `POST /api/status/dismiss-error` → `idle`.
  4. `[BROWSER]` with app open, restart backend → "Session lost" status appears; session list functional.

- [ ] **Step 5: Commit**

```bash
git add USER-FLOWS.md
git commit -m "docs: harness Tier-1 group D — deletion, export, recovery flows"
```

---

### Task 6: Tier-2 table, impact map, deploy smoke set

> **COMPLETED** (see git history dc20610..e5ca25a). The committed USER-FLOWS.md/CLAUDE.md supersede this draft text where they differ (review amendments). Do NOT re-execute this task.


**Files:**
- Modify: `USER-FLOWS.md`

- [ ] **Step 1: Write the Tier-2 table** — columns `ID | Flow | Contract | Invariant to check`. One row each (use the 2026-06-10 exploration inventory for endpoint/component specifics):

UF-1.3 Import session (entry point; deep coverage lives in UF-8.2) · UF-1.4 Delete session (removes dir + GCS + IndexedDB; must not touch other sessions) · UF-2.1 Create class (POST `/api/session/classes`, next-ID assignment) · UF-2.2 Rename class · UF-2.3 Change class color · UF-2.4 Toggle class visibility (hidden ⇒ masks not rendered, objects deselected) · UF-2.5 Delete class (cascades object removal + prompts + masks; N2) · UF-2.6 Select class (resets tool to pointer, unhides) · UF-2.7 Auto-create object on first click · UF-2.8 Select object (single) · UF-2.9 Multi-select via Shift+click · UF-2.10 Clear selection (Esc/blank click) · UF-2.11 Delete object (all frames; N2/N4) · UF-2.12 Reassign object class · UF-2.13 Orphan-object reassignment dialog · UF-3.4 Recalculate keyframe mask (same prompt, fresh inference) · UF-3.5 Erase prompt ("Missing Mask" panel) · UF-4.5 Confidence-warning review (banner, go-to-frame, amber ticks) · UF-6.1 Play/pause + FPS · UF-6.2 Timeline navigation (slider/keys; tick colors blue=keyframe, green=mask, amber=low-conf) · UF-6.3 Prev/next mask jump · UF-7.1 Auto-save state (debounce 500 ms; version guard; N5) · UF-7.2 beforeunload flush beacon · UF-9.1 Theme toggle (persists in localStorage) · UF-10.1 Zoom & pan · UF-10.2 Click mask to select / Shift toggle · UF-5.2 Zoom-to-mask + bbox padding sliders (per-object per-keyframe; auto-save; export-only effect) · UF-11.1 Frame caching (IndexedDB, background) · UF-11.2 Mask caching (versioned smart load) · UF-11.3 Refresh masks (evict + refetch).

- [ ] **Step 2: Write the Impact map table** — columns `Code touched | Re-verify flows | Invariants`. Rows (minimum):

| Code | Flows | Invariants |
|---|---|---|
| `backend/app/services/sam3_service.py`, `routes/segment.py` | UF-3.1–3.3, UF-4.1–4.4 | N1–N4, N9, N10 |
| `services/mask_storage.py`, `services/prompt_storage.py` | UF-3.x, UF-5.1, UF-8.x | N2, N7 |
| `routes/session.py`, `services/session_manager.py` | UF-1.2, UF-2.x, UF-7.x, UF-12 | N5, N8 |
| `services/gcs_sync.py`, cloud-mode paths | UF-1.2, UF-1.5, UF-11.x | N6 (+ CLAUDE.md "Cloud mode blind spots") |
| `services/pipeline.py`, `routes/video.py` | UF-1.1, UF-1.4, UF-12 | N8 |
| `routes/export.py`, `services/exporter.py` | UF-8.1, UF-8.2 | N7 |
| `frontend/src/App.tsx` | all Tier-1 `[BROWSER]` steps; minimally UF-1.x, UF-7.1 | N5, N6 |
| `frontend/src/api.ts` | flows matching the endpoints changed | — |
| `components/VideoCanvas.tsx` | UF-3.1, UF-3.2, UF-10.x | — |
| `components/AnnotationPanel.tsx` | UF-2.x | — |
| `components/PropagationBar.tsx`, `ToolBar.tsx` | UF-3.x, UF-4.x | — |
| `components/FrameNavigator.tsx`, `DeleteMasksPanel.tsx` | UF-5.1, UF-6.x | — |
| `components/VideoUpload.tsx`, export dialogs | UF-1.x, UF-8.x | — |
| `maskCache.ts`, `frameCache.ts` | UF-11.x, UF-1.2 | — |

- [ ] **Step 3: Write the Deploy smoke set** — port CLAUDE.md's 9-row deploy table as a named flow list: `UF-1.1, UF-1.2 (resume from list), UF-3.1, UF-4.1, UF-4.2, UF-1.5, UF-12 (cancel-during-init + status idle)`, each row keeping its one-line "what it verifies" and noting `curl $URL/api/status` as step 0. Remove the `<!-- populated -->` comments left by Task 1.

- [ ] **Step 4: Commit**

```bash
git add USER-FLOWS.md
git commit -m "docs: harness Tier-2 table, impact map, deploy smoke set"
```

---

### Task 7: CLAUDE.md integration

> **COMPLETED** (see git history dc20610..e5ca25a). The committed USER-FLOWS.md/CLAUDE.md supersede this draft text where they differ (review amendments). Do NOT re-execute this task.


**Files:**
- Modify: `sam3-video-labelling-tool/CLAUDE.md` (the "User Flows" section and "Deploy smoke tests" subsection)

- [ ] **Step 1: Replace the "## User Flows" section body** (keep the header) with:

```markdown
Full flow catalog, negative invariants (N1–N10), test fixture, and the
verification harness live in **[USER-FLOWS.md](USER-FLOWS.md)**.

**After any behavioral change:** look up the touched files in the harness's
Impact map and verify the listed flows + invariants before claiming success.
When code and USER-FLOWS.md disagree, the document wins or must be amended in
the same commit.
```

- [ ] **Step 2: Replace the "### Deploy smoke tests" table** with a pointer: "Run the **Deploy smoke set** in [USER-FLOWS.md](USER-FLOWS.md) after every deploy." Keep the surrounding "Testing" prose intact.

- [ ] **Step 3: Verify no other CLAUDE.md sections reference the removed table** (`grep -n "smoke" CLAUDE.md`) → fix any dangling references.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: point CLAUDE.md at USER-FLOWS.md harness; move smoke tests"
```

---

### Task 8: Build the fixture (BLOCKED until sample video provided)

**Files:**
- Create: `tests/fixtures/harness/sample.mp4` (user-provided, re-encoded if > 640 px or > 10 s: `ffmpeg -i in.mp4 -vf scale=640:-2 -t 10 -an sample.mp4`)
- Create: `tests/fixtures/harness/golden_session.zip`
- Create: `tests/fixtures/harness/fixture.json`
- Create: `tests/fixtures/harness/iou.py`
- Create: `tests/fixtures/harness/README.md`

- [ ] **Step 1: Start the app** — `make dev` (conda env `sam3-annotator`); wait for `GET http://localhost:5555/api/status` → `{"phase":"idle"}`.

- [ ] **Step 2: Upload sample.mp4 via `[API]`** (UF-1.1 step 1 verbatim); record `session_id`; wait for `ready`.

- [ ] **Step 3: Choose canonical coordinates** — `Read` 2–3 extracted frame images from the session dir; pick, per object: one positive click point near its center, and a bounding box; pick a `text_query` matching an object category. **The keyframe is pinned to frame 0** (the Tier-1 flows hardcode `frame_idx: 0`); if objects aren't all visible on frame 0, trim/re-encode the video so they are.

- [ ] **Step 4: Create golden annotations** — via `[API]`: create 2–3 classes; click-segment object 1, box-segment object 2 (text-detect object 3 if present); propagate all objects forward (UF-4.2 path). Visually confirm by `Read`ing a few composite/overlay outputs or checking mask areas are sane (> 1 % and < 90 % of frame).

- [ ] **Step 4b: Set bbox padding on object 1's keyframe** — `PUT /api/session/state/<session_id>` with a known padding (e.g. `{"1": {"0": {"top": 10, "bottom": 10, "left": 5, "right": 5}}}`) so UF-8.1 step 4 (padding arithmetic) has a padded golden to verify against. Record the values in `fixture.json.bbox_padding`.

- [ ] **Step 5: Export golden** — `POST /api/export/session/{id}` with `include_video:false` → save as `tests/fixtures/harness/golden_session.zip`.

- [ ] **Step 6: Write `fixture.json`** with this exact shape:

```json
{
  "video": "sample.mp4",
  "fps": 5,
  "keyframe": 0,
  "objects": [
    {"obj_id": 1, "class": "<name>", "click": [x, y], "box": null},
    {"obj_id": 2, "class": "<name>", "click": null, "box": [x1, y1, x2, y2]}
  ],
  "text_query": "<category>",
  "expected_text_detections": 0,
  "expected_object_count": 2,
  "mask_frame_count": 0,
  "bbox_padding": {"1": {"0": {"top": 10, "bottom": 10, "left": 5, "right": 5}}},
  "iou_thresholds": {"vs_golden": 0.80, "repeat_click": 0.99},
  "propagation_sample_frames": [2, 5, 9]
}
```

(`expected_text_detections`, `expected_object_count` (count in the golden session's `state.json`, including text-detected objects), `mask_frame_count` (frames with ≥1 mask in the golden `masks.json`), and `propagation_sample_frames` filled with measured values; `bbox_padding` echoes Step 4b.)

- [ ] **Step 6b: Write `iou.py`** — decodes two RLE mask strings using the backend's RLE decoder (see `backend/app/services/mask_storage.py` or equivalent — verify the actual module) and prints IoU; used by all IoU checks in USER-FLOWS.md.

- [ ] **Step 7: Write `README.md`** — what the video shows, how each golden was produced (exact API calls), the regeneration procedure ("delete goldens, re-run Task 8 steps 1–6"), and the note that goldens were generated on MPS so CUDA comparisons rely on the IoU ≥ 0.80 tolerance.

- [ ] **Step 8: Update USER-FLOWS.md fixture section** if any measured value differs from the documented contract (e.g., object count).

- [ ] **Step 9: Commit**

```bash
git add tests/fixtures/harness/ USER-FLOWS.md
git commit -m "test: add harness fixture — sample video, golden session, canonical prompts"
```

---

### Task 9: Exercise the harness end-to-end (acceptance criterion 7)

**Files:**
- Modify: `USER-FLOWS.md` (amendments where steps don't run as written)

- [ ] **Step 1: Fresh run** — delete the Task-8 working session (keep fixture files), restart `make dev`.

- [ ] **Step 2: Execute every Tier-1 flow's `[API]` and `[DISK]` steps in ID order** against the fixture, recording PASS/FAIL/NOT RUN per step. `[BROWSER]` steps: run if a browser tool is available this session, else mark NOT RUN per the degradation rule.

   Empirical confirmations this run must settle (claims that were source-traced for native CUDA but hedged for the local HF/MPS backend): (a) UF-4.1's N2 recipe expects propagating a prompt-less object to yield an SSE `error` event rather than `{"done": true}` with zero frames — confirm on HF/MPS and amend the Expect if reality differs; (b) UF-4.4's "N or N+1 frames persisted" tolerance under cancel timing.

- [ ] **Step 3: Fix the document, not just note failures** — every step that is wrong-as-written (bad endpoint, wrong expected value, impossible ordering) gets corrected in USER-FLOWS.md. If a step reveals an actual app bug, do NOT fix the app in this task — file it and add the corresponding Must NOT line per maintenance rule 2.

- [ ] **Step 4: Report** — produce the per-flow PASS/FAIL table + the `[HUMAN]` checklist for Danilo (this is the harness's standard output format; its first real production).

- [ ] **Step 5: Commit**

```bash
git add USER-FLOWS.md
git commit -m "docs: harness amendments from first end-to-end exercise"
```
