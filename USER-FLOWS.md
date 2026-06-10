# USER-FLOWS.md — Verification Harness

> Behavioral contract + regression checklist for the SAM3 Video Labelling Tool.
> When code and this document disagree, this document wins — or is explicitly
> amended in the same commit that changes the behavior.

---

## How to use this harness

The loop, after any behavioral change:

1. List the files you touched.
2. Look each up in the **Impact map** → collect the flow IDs and N-invariants.
3. Execute each flow's **Verify** steps in order. Tags say who runs them:
   - `[API]` — Claude: curl against Flask (`http://localhost:5555`)
   - `[DISK]` — Claude: inspect session files under the backend sessions dir
     (`masks.json`, `state.json`, `prompts.json`) or exported zips
   - `[BROWSER]` — Claude drives a real browser against `http://localhost:5173`
     using the fixture's canonical click coordinates
   - `[HUMAN]` — only a person can judge (mask visual quality, drag feel)
4. Check every **Must NOT** invariant listed for those flows.
5. Report per flow: PASS / FAIL (with evidence) / NOT RUN (with reason).
   Hand the user the remaining `[HUMAN]` checklist.

**Degradation rule:** if no browser-automation tool is available in the
session, each `[BROWSER]` step is reported NOT RUN and appended to the
`[HUMAN]` checklist — never silently skipped.

**Environment:** local dev (`make dev`: Flask :5555 + Vite :5173, conda env
`sam3-annotator`, MPS). Cloud-mode-only paths (GCS sync, Cloud Run lifecycle)
cannot be fully verified locally — see "Cloud mode blind spots" in CLAUDE.md;
mark those steps NOT RUN locally and use the Deploy smoke set after deploys.

---

## Global negative invariants (N1–N10)

These invariants are stated once here and referenced by ID from individual flows. When a new bug is found and fixed, its invariant is added as a Must-NOT line on the relevant flow (or a new N-rule) in the same PR — the "regression → invariant" pipeline.

- **N1** Propagation only touches targeted objects — propagating object(s) must not create, modify, or delete masks for any object not in the propagation set.
  _Check:_ byte-compare non-target objects' RLE strings in `masks.json` before/after propagation.

- **N2** Deleted masks stay deleted — disk deletion must also clear SAM3 in-memory state; propagation must never regenerate them.
  _Check:_ delete a mask, propagate across that frame, assert the mask is absent afterward.

- **N3** A correction click behaves identically to a first-time click — no residual inference state biases the result.
  _Check:_ same click coords on a frame with prior cleared state vs. a fresh object → IoU ≥ 0.99 between the two masks.

- **N4** UI state = backend state — SAM3 inference state contains only the active object(s); no stale objects accumulate.
  _Check:_ backend inference state holds only active object IDs (a code-reading check on `backend/app/services/sam3_service.py` — assert via `remove_object`/replay behavior).

- **N5** Auto-save never writes the "wipe fingerprint" (empty classes + objects with `class_id < 0`).
  _Check:_ inspect persisted `state.json` after any save — it must never contain `{"classes": [], "objects": [...with class_id < 0...]}` (the "wipe fingerprint").

- **N6** Close never silently loses data — debounced saves flushed before close; GCS sync failure surfaces the retry dialog, never a silent exit.
  _Check:_ close with a pending bboxPadding edit → reopen → padding survived; in cloud mode, a GCS failure during close must surface the retry dialog.

- **N7** Export contains exactly what's annotated — bbox padding matches `state.json`; no unannotated frames in COCO output.
  _Check:_ COCO `instances.json` bboxes equal mask bbox + padding from `state.json`; exported image count == annotated-frame count.

- **N8** `/api/status` phase always reflects reality — it drives all frontend top-level routing.
  _Check:_ `GET /api/status` phase matches actual service activity at every flow boundary.

- **N9** Cancel actually cancels — no orphaned propagation thread holding `SAM3Service._lock`.
  _Check:_ after cancel, `GET /api/segment/propagation-status/{id}` reports not-running, and a subsequent click responds within normal latency (lock released).

- **N10** Multi-object reset+replay loses no prompts — every selected object's prompts are replayed from disk after `_reset_inference_state`.
  _Check:_ after multi-object propagation, every selected object has masks on the new frames; `prompts.json` unchanged.

---

## Test fixture

```
tests/fixtures/harness/
├── sample.mp4            # 5–10 s, ≤640 px, 2–3 visually distinct objects
├── golden_session.zip    # known-good exported session (masks/prompts/state)
├── fixture.json          # canonical click coords per object on the keyframe,
│                         #   expected object count, IoU thresholds
└── README.md             # how goldens were made, how to regenerate
```

(assets committed separately)

**Comparison rules:**

- SAM3 output vs. golden masks: **IoU ≥ 0.80** (SAM3 is not bit-exact across MPS/CUDA backends).
- "Untouched data didn't change" checks (e.g. N1): **byte equality** of RLE strings.
- `golden_session.zip` doubles as the import-flow test asset (UF-8.2).

---

## Tier 1 flows

<!-- populated in Tasks 2–6 of docs/superpowers/plans/2026-06-10-user-flows-harness.md -->

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
2. Every postmortem'd bug adds a Must-NOT invariant to the relevant flow (or a global N-rule) — the regression → invariant pipeline.
3. Flow IDs are stable; retired flows are marked deprecated, never deleted or renumbered.
4. The impact map gains a row whenever a new code area is created.
5. New flows get the next unused number in their range; new ranges append.
