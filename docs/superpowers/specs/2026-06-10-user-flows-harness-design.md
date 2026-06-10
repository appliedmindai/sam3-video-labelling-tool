# USER-FLOWS.md Harness — Design Spec

**Date:** 2026-06-10
**Status:** Approved by Danilo (brainstorming session)
**Deliverable:** A `USER-FLOWS.md` document at the repo root, linked from `CLAUDE.md`, that serves as the verification harness for all development on the SAM3 Video Labelling Tool — plus a committed test fixture under `tests/fixtures/harness/`.

## Purpose & usage model

The document is **both** a behavioral contract and a regression checklist ("layered"):

- Each flow states a short **Contract** — what the user believes is true. When code and contract disagree, the contract wins or is explicitly amended in the same commit.
- Each Tier-1 flow carries concrete **verification steps** with observable expected outcomes, executable after any change.
- Each Tier-1 flow carries **Must NOT** negative invariants — falsifiable statements of what must never happen, each with a concrete comparison method. This operationalizes the lesson from `experiment-video-annotation-tool/blog/lessons-from-building-with-llms-and-open-source.md`: positive requirements confirm, negative requirements find bugs.

**The harness loop:** change code → look up touched files in the impact map → verify the listed flows and invariants → report results before claiming success.

## Executor model: hybrid, machine-first

Verification steps are tagged by executor:

| Tag | Meaning |
|---|---|
| `[API]` | Claude runs it: curl against Flask (`:5555`), SSE consumption, status polls |
| `[DISK]` | Claude inspects session files: `masks.json`, `state.json`, `prompts.json`, exports |
| `[BROWSER]` | Claude drives a real browser against the Vite UI (`:5173`) — clicking canonical fixture coordinates, reading DOM/timeline state |
| `[HUMAN]` | Only a person can judge: mask visual quality, drag feel, overlay rendering |

Claude runs `[API]`/`[DISK]`/`[BROWSER]` itself and hands the user a short `[HUMAN]` checklist at the end.

**Degradation rule:** `[BROWSER]` steps require a browser-automation tool in the session (e.g. Claude Code browser control or a Playwright MCP). When none is available, each `[BROWSER]` step is reported as NOT RUN and added to the `[HUMAN]` checklist — never silently skipped.

## Coverage: tiered

- **Tier 1 (~14 flows, full depth):** Upload→annotate, Resume session, Click segmentation + undo, Box segmentation, Text detection, Single-object propagation, Multi-object propagation, Cancel propagation, Propagation reconnect (tab close/reopen), Mask deletion (single + batch), Close session (incl. GCS-failure retry path), Export COCO, Export/Import session zip, Recovery (container restart / error phase / state version conflict).
- **Tier 2 (everything else):** one row each in a compact table — ID, one-line contract, one observable invariant. Covers class/object management, frame navigation, playback, zoom/pan, bbox padding, theme, caching, refresh masks, orphan-object reassignment, etc.

Flow IDs (`UF-x.y`) are stable forever and never reused. Commits, issues, and verification reports reference them.

## Tier-1 flow anatomy

```markdown
### UF-4.1 Single-Object Propagation

**Contract:** 2–3 sentences of user-facing intent.

**Must NOT:**
- Falsifiable negative invariants, each with a concrete check method
  (e.g., "compare other objects' RLEs byte-for-byte before/after").
- May reference global invariants by ID (N1, N2, …).

**Verify:**
1. [API] step → expected outcome
2. [DISK] step → expected outcome
3. [BROWSER] step → expected outcome
4. [HUMAN] step → what the person should see
```

## Global negative invariants (N1–N10)

Stated once in their own section, referenced by ID from flows:

- **N1** Propagation only touches targeted objects — propagating object(s) must not create, modify, or delete masks for any object not in the propagation set.
- **N2** Deleted masks stay deleted — disk deletion must also clear SAM3 in-memory state; propagation must never regenerate them.
- **N3** A correction click behaves identically to a first-time click — no residual inference state biases the result.
- **N4** UI state = backend state — SAM3 inference state contains only the active object(s); no stale objects accumulate.
- **N5** Auto-save never writes the "wipe fingerprint" (empty classes + objects with `class_id < 0`).
- **N6** Close never silently loses data — debounced saves flushed before close; GCS sync failure surfaces the retry dialog, never a silent exit.
- **N7** Export contains exactly what's annotated — bbox padding matches `state.json`; no unannotated frames in COCO output.
- **N8** `/api/status` phase always reflects reality — it drives all frontend top-level routing.
- **N9** Cancel actually cancels — no orphaned propagation thread holding `SAM3Service._lock`.
- **N10** Multi-object reset+replay loses no prompts — every selected object's prompts are replayed from disk after `_reset_inference_state`.

When a new bug is found and fixed, its invariant is added as a Must-NOT line (or a new N-rule) in the same PR — the "regression → invariant" pipeline.

## Test fixture

```
tests/fixtures/harness/
├── sample.mp4            # ~5–8 s, ≤640 px, 2–3 visually distinct objects
├── golden_session.zip    # exported session: known-good masks/prompts/state
├── fixture.json          # canonical click coords per object on the keyframe,
│                         #   expected object count, IoU thresholds
└── README.md             # how goldens were made, how to regenerate them
```

- `fixture.json` makes `[API]`/`[BROWSER]` steps deterministic: known click coordinates on a known frame.
- **Mask comparison rule:** new segmentation output vs. golden masks uses **IoU ≥ 0.80** (SAM3 is not bit-exact across MPS/CUDA). **Byte-equality** is reserved for "untouched data didn't change" checks (e.g., N1).
- `golden_session.zip` doubles as the import-flow test asset.
- **Division of labor:** Danilo provides only the raw sample video (5–10 s, any ffmpeg-readable format, 2–3 visually distinct objects that persist across frames). Claude does everything else during implementation: uploads it through the app (MPS backend), reads the extracted frame images to choose canonical click coordinates, segments and propagates the objects, exports the session as `golden_session.zip`, and writes `fixture.json` and the README. No fixture work falls on the user.

## Impact map

A table in USER-FLOWS.md mapping code areas → flow IDs + invariants to re-verify. Rows for at least: `sam3_service.py` + `segment.py` routes; `mask_storage.py` + `prompt_storage.py`; `session.py` routes + `session_manager.py`; `gcs_sync.py` + cloud-mode paths; `pipeline.py`; `export.py` + `exporter.py`; `App.tsx`; `VideoCanvas.tsx`; `AnnotationPanel.tsx`; `PropagationBar.tsx` + `ToolBar.tsx`; `FrameNavigator.tsx` + `DeleteMasksPanel.tsx`; `VideoUpload.tsx` + dialogs; `api.ts`; cache modules (`maskCache.ts`, `frameCache.ts`). Cloud-mode rows carry a note pointing at CLAUDE.md's "Cloud mode blind spots".

## CLAUDE.md integration

- The current "User Flows" section in `sam3-video-labelling-tool/CLAUDE.md` shrinks to a pointer: full catalog, negative invariants, and harness live in `USER-FLOWS.md`; after any behavioral change, consult the impact map and verify listed flows before claiming success.
- The existing "Deploy smoke tests" table in CLAUDE.md moves into USER-FLOWS.md as a named subset ("Deploy smoke set": the flow IDs to run after every deploy); CLAUDE.md's table is replaced by a pointer. One place for verification steps.

## Maintenance rules (stated inside USER-FLOWS.md)

1. A PR that changes user-facing behavior must update the affected flow entry in the same commit.
2. Every postmortem'd bug adds a Must-NOT invariant to the relevant flow (or a global N-rule).
3. Flow IDs are stable; retired flows are marked deprecated, never deleted or renumbered.
4. The impact map gains a row whenever a new code area is created.

## Out of scope (deferred, by explicit decision)

- Executable harness scripts (`tests/harness/*.sh`) — Approach B, a natural upgrade once the prose harness proves itself.
- A `.claude/skills/verify-flows` project skill — Approach C, same.
- Automated Playwright/pytest E2E suites derived from the flows.

## Acceptance criteria

1. `USER-FLOWS.md` exists at the repo root with all sections above; ~14 Tier-1 flows at full depth; Tier-2 table covering the remaining inventory (~35 flows from the exploration of 2026-06-10).
2. Every Tier-1 flow has Contract, ≥1 Must-NOT line, and tagged Verify steps with expected outcomes.
3. Global invariants N1–N10 present; flows reference them by ID.
4. Impact map covers all backend services/routes and frontend components.
5. Fixture directory committed with sample video, golden session, `fixture.json`, README; IoU rule documented.
6. CLAUDE.md updated to point at the harness.
7. The harness is exercised once end-to-end against the fixture on local MPS to prove the steps actually run as written.
