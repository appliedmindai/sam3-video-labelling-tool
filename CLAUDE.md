# CLAUDE.md

## Project

SAM3 Video Labelling Tool — video annotation powered by SAM 3.1. React 18 + TypeScript + Vite frontend, Flask + PyTorch backend. Dual-backend: HuggingFace Transformers on MPS (Mac), native `facebookresearch/sam3` on CUDA (production). Runs locally or on Cloud Run + GCS (`SEGMENT_MODE=cloud`, bucket from `GCS_BUCKET` env var — no auth layer).

## Commands

```bash
# Development (runs both backend + frontend)
make dev                      # Flask :5555 + Vite :5173

# Backend only
cd backend && python3 run.py  # Flask dev server on :5555

# Frontend only
cd frontend && npm run dev    # Vite dev server on :5173 (proxies /api → :5555)
cd frontend && npm run build  # Type check + production build
cd frontend && npx tsc --noEmit  # Type check only

# Tests
cd backend && python3 -m pytest tests/ -v

# Conda environment
eval "$(/opt/homebrew/bin/conda shell.zsh hook)" && conda activate sam2-annotator  # env name predates the SAM3 rename
```

## Architecture

```
backend/app/
  routes/         # Flask blueprints: video, session, segment, export, status
  services/       # Business logic: sam3_service, mask_storage, session_manager, exporter, gcs_sync
frontend/src/
  App.tsx         # Root component — all state lives here, passed via props
  api.ts          # REST client (XHR for uploads, fetch for everything else)
  types.ts        # Shared TypeScript interfaces
  components/     # UI components (VideoCanvas, ClassPanel, ObjectList, ToolBar, etc.)
  components/ui/  # shadcn/ui primitives
```

API: 5 blueprints under `/api/` — video (upload/frames), session (state/masks/classes), segment (SAM3 init/click/box/propagate), export (COCO/RF-DETR), status.

Data flow: click/box → SAM3 inference → RLE mask → persist to `masks.json` → return to frontend. Propagation streams via SSE.

Cloud mode: `g.bucket` is set per-request by a `before_request` hook from the `GCS_BUCKET` env var; sessions sync to GCS via `GCSSyncManager`.

## User Flows

Full flow catalog, global negative invariants (N1–N11), test fixture, and the
verification harness live in **[USER-FLOWS.md](USER-FLOWS.md)**.

**After any behavioral change:** look up the touched files in the harness's
Impact map and verify the listed flows + invariants before claiming success.
When code and USER-FLOWS.md disagree, the document wins — or must be amended in
the same commit.

## Principles

**KISS.** Write the minimum code that solves the current problem. Three similar lines are better than a premature abstraction. No feature flags, no backwards-compat shims, no hypothetical future requirements.

**Ask before doing.** Before making architectural decisions, changing behavior, or touching code outside the immediate task scope — ask. Don't refactor adjacent code, add error handling for impossible scenarios, or "improve" things that weren't requested.

**Confirm destructive actions.** Never force-push, delete branches, amend published commits, or run destructive git/shell commands without explicit confirmation. Measure twice, cut once.

**Clean names over comments.** Functions, variables, and types should be self-documenting. Only add comments when the *why* isn't obvious from the code. Don't add docstrings, type annotations, or comments to code you didn't change.

**Don't over-engineer.** No extra configurability, no helper utilities for one-time operations, no abstractions that serve a single caller. If you're adding complexity, you should be able to explain why it's necessary *right now*.

**Verify library behavior with negative requirements.** When integrating with third-party libraries (especially SAM3), don't just confirm that a method does what we want — ask what else it does that we *don't* want. Before building on any library method: (1) state the assumption explicitly ("propagation only touches the target object"), (2) read the implementation looking for violations of that assumption, not confirmations, and (3) flag any side effects, implicit scope, or state mutations that affect things outside the immediate call. Positive requirements ("does it propagate?") will always get a yes. Negative requirements ("does it touch objects we didn't ask about?") find the bugs.

**UI state = backend state.** The user's mental model and the app's internal state must be one. SAM3's inference state should only contain the object the user is currently interacting with ("active object" model). When the user switches objects, the backend resets and replays only that object's prompts from disk. The frontend selection drives backend state — never let stale objects accumulate in memory.

**Do the thing. Not the thing around the thing.** When asked to fix bug X, fix X. Don't also refactor Y, add "helpful" Z, or change a working subsystem because it "should" be different. If propagation speed is good, don't touch the tracker's storage config while fixing something else.

**Ask before changing what works.** Before modifying any subsystem that's currently working well, state: (1) what you want to change, (2) what it affects, (3) what it does NOT affect, and (4) get confirmation.

**One change, one commit, one deploy.** Don't batch 5 fixes hoping they all work. Each change should be independently verifiable. If a build fails, you know exactly which change broke it.

**Verify with the actual build command.** Run `tsc -b && vite build` (the Docker build's exact check), not just `tsc --noEmit`. The production build is stricter — unused variables, forward references, and import issues that `--noEmit` misses will fail the Docker build.

**Cloud Run is stateful-in-practice.** This service holds GPU model state, local disk sessions, and sync managers in memory. Treat it like a VM, not a function. Key rules:
- `max-instances=1, concurrency=8` — single container handles up to 8 concurrent requests. GPU serialization is enforced in-process by `SAM3Service._lock`. Do NOT raise max-instances above 1 — multiple containers lose state.
- Deploys create new containers — the old one dies. Warn the user before deploying during active sessions.
- The heartbeat keeps the container alive. Without active requests for 30 min, it scales to zero and all in-memory state is lost. GCS sync preserves masks/prompts/state; SAM3 session must be re-initialized on next use.

## Code Style

### TypeScript (frontend)
- Strict mode, no unused locals/parameters
- shadcn/ui + Tailwind CSS v4 (indigo primary, zinc base, Noto Sans)
- Icons: lucide-react
- Path alias: `@/` → `./src/`
- State management: props from App.tsx, no context/redux
- `useCallback` with correct dependency arrays — watch for stale closures

### Python (backend)
- Flask blueprints + services layer (no database — JSON files on disk)
- SAM3Service is a singleton (shared across blueprints via `__new__`)
- Always use `python3` (macOS has no `python` on PATH)
- Threading: `self._lock` (RLock) guards all SAM3 predictor access; propagation runs in a daemon thread

## GPU Selection — Read This Before Deploying

**SAM3 was designed for Ampere+ GPUs (A100, H100, L4).** Deploying on older hardware wastes engineering time on dtype workarounds that a $0.70/hr L4 avoids entirely.

We spent significant effort optimizing SAM3 for T4 (Turing, compute cap 7.5) — patching bfloat16→float16 autocast, working around dtype mismatches in SAM3's internals, disabling Flash Attention. **An L4 at similar cost would have worked natively with zero patches.**

| GPU | $/hr (GCP) | Compute Cap | bfloat16 | Flash Attn | SAM3 experience |
|---|---|---|---|---|---|
| **T4** | ~$0.35 | 7.5 (Turing) | Emulated (2x penalty) | No | Needs float16 autocast hacks |
| **L4** | ~$0.70 | 8.9 (Ada) | Native | Yes | Works out of the box |
| **A10G** | ~$1.00 | 8.6 (Ampere) | Native | Yes | Works out of the box |
| **A100** | ~$3.00 | 8.0 (Ampere) | Native | Yes | Fastest, designed for this |

**Rule: for any transformer model that uses bfloat16 (SAM3, LLMs, diffusion models), deploy on compute cap >= 8.0.** The time saved on debugging dtype issues far exceeds the marginal GPU cost difference. See `blog/sam3-native-cuda-the-dtype-maze.md` for the full story.

### Benchmark results (3 objects, 20 frames)

| Backend | GPU | ms/frame | FPS |
|---|---|---|---|
| SAM 3.1 HF Transformers | T4 | 3,041 | 0.33 |
| SAM 3.1 native float32 | T4 | 3,185 | 0.31 |
| **SAM 3.1 native fp16 autocast** | **T4** | **735** | **1.35** |
| **SAM 3.1 native bfloat16** | **L4** | **267** | **3.74** |

## SAM3 & Device Constraints

These are critical — getting them wrong causes silent data corruption or crashes:

**MPS (Mac local dev):**
- **Autocast and pin_memory fail on MPS.** We monkey-patch autocast to return nullcontext and wrap post_process_masks to fall back to CPU. See `_apply_mps_patches()`.
- **`PYTORCH_ENABLE_MPS_FALLBACK=1`** must be set before importing torch (done in `run.py`).
- **Frames and inference state must live on CPU.** Keeping them on MPS causes memory pressure that degrades propagation from 5.5s→70s/frame. CPU stays steady at ~2.9s/frame. The CPU→MPS transfer cost (~1ms) is negligible.

**CUDA (production):**
- **T4 cannot do bfloat16 natively.** Use float32 weights + `torch.amp.autocast(dtype=float16)`. Patch `addmm_act` to use float16 instead of bfloat16. See `blog/sam3-native-cuda-the-dtype-maze.md`.
- **Ampere+ GPUs (L4, A10G, A100, H100):** bfloat16 works natively. No patches needed. Flash Attention available.

**Both:**
- **In-memory state must match disk state.** When deleting masks, you must also call `clear_frame_object()` on the SAM3 inference state — otherwise propagation regenerates deleted masks from memory.
- **RLE is column-major.** Linear index = `col * height + row`. The `decodeRleCounts` function uses LEB128 delta encoding. `hitTestMask` and `drawMaskFromRle` both depend on this ordering.
- **One SAM3 session = one inference state.** The predictor holds per-object output dicts, prompt inputs, and tracking markers. `close_session()` must cancel propagation before acquiring the lock, then clean up all bookkeeping dicts.
- **Cannot add new objects after tracking starts.** The native SAM3 predictor raises `RuntimeError("Cannot add new object id N after tracking starts")` if you try to register a new object after any propagation has run. For multi-object propagation, you must **reset the inference state first** (`_reset_inference_state`) then replay ALL objects from scratch via `reset_and_replay_objects()` — you cannot incrementally add missing objects. This is enforced in `sam3_tracking_predictor.py:_obj_id_to_idx()`. The single-object path (`ensure_active_object`) already handles this correctly; the multi-object path must explicitly reset before replay.

## Testing

- Backend: pytest in `backend/tests/` — tests cover mask utilities, storage, session manager, exporter, routes, concurrency
- Frontend: no test suite — rely on `tsc --noEmit` and manual verification
- Always run `npx tsc --noEmit` after frontend changes before claiming success

### Cloud mode blind spots

The test suite runs locally with no GCS, no real SAM3 model, and no GPU. **All tests passing does not mean cloud mode works.** These paths are only exercised in production:

- `GCSSyncManager` lifecycle (start, flush, stop, propagation deferral)
- `DownloadSessionStep` / `gcs_storage.download_session()` — GCS → local disk
- `ExtractFramesStep` GCS upload (video, meta, frames to bucket)
- `meta.json` availability — in cloud mode, meta.json may not exist locally until `DownloadSessionStep` runs. Code that reads meta in route handlers (before the pipeline) must handle this.
- SAM3 model loading on CUDA (L4) — `_ensure_model()`, `init_state()`, backbone warmup timings

**Rule: when writing code that touches GCS, sync managers, or cloud-mode-only paths, always verify constructor signatures and method parameters against the actual source.** Plans and specs may contain wrong assumptions. Read the real `__init__` before calling it.

### Deploy smoke tests

Run the **Deploy smoke set** in [USER-FLOWS.md](USER-FLOWS.md) after every deploy. It covers status, upload, tab-close recovery, resume, single- and multi-object propagation, cancel-during-init, and close.

## Gotchas

- The `bboxPadding` auto-save uses a 500ms debounce. If the session closes within that window, the timer is cancelled. `handleCloseSession` explicitly flushes state before clearing `sessionId`.
- `confirmDialog.onConfirm` is typed as `() => void` but `handleCloseSession` passes an async function. This works because React ignores the returned Promise, but the pattern is fragile.
- `Object.keys(masks)` returns integer keys in ascending numeric order (V8 behavior). The mask draw loop and hit-test loop both rely on this insertion order being consistent.
- Propagation holds `self._lock` for the entire run. Any operation that needs the lock (like `close_session`) will block until the current frame finishes inference.
- **Flask `g` is not available in background threads.** Any value from the request context (e.g., `g.bucket`) that a pipeline step needs must be captured at construction time in the route handler and passed into the step's `__init__`. Accessing `g` inside `step.run()` raises `RuntimeError`.
- **`ServiceState` is service-wide, not per-session.** Only one pipeline or active session at a time. The `phase` field drives the frontend's top-level routing via `GET /api/status`.
