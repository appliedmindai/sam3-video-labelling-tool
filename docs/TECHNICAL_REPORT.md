# SAM3 Video Labelling Tool — Technical Report

This document explains the engineering decisions behind the tool: how it's put together, why it runs on the hardware it runs on, and what it took to make SAM 3.1 fast and cheap to operate. It assumes you've read the [README](../README.md).

## Contents

1. [Architecture](#architecture)
2. [Dual SAM3 backend](#dual-sam3-backend)
3. [GPU selection](#gpu-selection)
4. [Cloud Run as a stateful GPU service](#cloud-run-as-a-stateful-gpu-service)
5. [Docker layer caching](#docker-layer-caching)
6. [Apple Silicon (MPS) constraints](#apple-silicon-mps-constraints)
7. [Running it: local and cloud](#running-it-local-and-cloud)
8. [Further reading](#further-reading)

---

## Architecture

```
React 18 + TS + Vite  ──/api──▶  Flask (gunicorn, 1 worker × 4 threads)
  canvas rendering                  routes/    video | session | segment | export | status
  IndexedDB mask cache              services/  SAM3Service (singleton, GPU)
                                               mask/prompt/session storage (JSON on disk)
                                               GCSSyncManager (cloud mode)
                                               pipeline (extract → init, background thread)
```

- **No database.** Sessions are directories: `video.mp4`, extracted JPEG frames, `meta.json`, `state.json`, `masks.json`, `prompts.json`. This makes sessions trivially portable (zip export/import) and the cloud sync story simple (upload dirty files).
- **Masks are RLE-encoded** (column-major, LEB128 delta counts) end to end — SAM3 output is encoded once on the backend and decoded on the canvas. A full HD mask is a few KB instead of a 2 MB bitmap.
- **One service state.** The backend holds exactly one active session and exposes its phase (`idle | extracting | initializing | ready | error`) via `GET /api/status`. The frontend polls this and routes its top-level UI off it, so a reloaded tab lands exactly where the backend is — including resuming a propagation already in flight.
- **"Active object" model.** SAM3's inference state only ever contains the object(s) the user is currently working with. Switching objects resets the predictor state and replays that object's prompts from disk. This came out of a hard-won lesson: video predictors in the SAM family propagate *every* object registered in their inference state, not just the one you asked about — letting stale objects accumulate silently corrupts masks you already approved. UI state and backend state must be the same thing.
- **Propagation streams over SSE.** A propagation run holds the GPU lock and streams per-frame results; the frontend renders masks as they arrive and shows per-frame confidence so you can spot drift and drop a correction keyframe.

The client-side data path — IndexedDB frame/mask caches, per-frame version counters for delta sync, and the SSE streaming protocol — is documented in [MASK_FRAME_STREAMING.md](MASK_FRAME_STREAMING.md).

## Dual SAM3 backend

The tool ships two interchangeable SAM 3.1 integration paths, selected automatically (`SAM3_BACKEND=auto`):

| Backend | Used on | Why it exists |
|---|---|---|
| HuggingFace Transformers (`Sam3TrackerVideoModel`) | Apple Silicon / CPU | Pure-PyTorch path that runs on MPS. Also provides text-prompted segmentation everywhere. |
| Native `facebookresearch/sam3` | CUDA | Triton kernels, Flash Attention, Object Multiplex — ~4x faster multi-object tracking than the HF path on the same GPU. |

The native predictor cannot register a new object after tracking has started (`Cannot add new object id N after tracking starts`). The multi-object path therefore resets the inference state and replays **all** selected objects' prompts from disk before propagating — incremental registration is not possible, and the reset-and-replay is the correctness anchor for everything multi-object.

## GPU selection

**Rule of thumb: for any transformer model that uses bfloat16 (SAM3 included), deploy on compute capability ≥ 8.0.** SAM3 was designed for Ampere+ GPUs.

We learned this the expensive way, on a T4 (Turing, compute cap 7.5):

- SAM3's native code creates bfloat16 tensors in three places: `torch.autocast` decorators, `torch.amp.autocast` context managers, and a fused kernel (`perflib/fused.py`) that *hardcodes* `.to(torch.bfloat16)`.
- T4 has no native bfloat16 — it's emulated at roughly a 2x penalty, and mixing emulated-bfloat16 activations with float32 weights crashes with `mat1 and mat2 must have the same dtype`.
- Making it work required float32 weights + a float16 autocast, a monkey-patched fused op, and disabling Flash Attention. Days of engineering to reach 735 ms/frame.
- An L4 (Ada, compute cap 8.9) runs the same code **with zero patches** at 267 ms/frame, for roughly twice the hourly price of a T4.

Benchmarks (3 objects, 20 frames, 640×360):

| Backend | GPU | ms/frame | FPS |
|---|---|---|---|
| SAM 3.1 HF Transformers | T4 | 3,041 | 0.33 |
| SAM 3.1 native, float32 | T4 | 3,185 | 0.31 |
| SAM 3.1 native, fp16 autocast (patched) | T4 | 735 | 1.35 |
| **SAM 3.1 native, bfloat16** | **L4** | **267** | **3.74** |

The dtype workarounds for sub-Ampere GPUs are preserved in the codebase (they activate automatically on compute cap < 8.0), but the deployment default is L4. The full debugging story is in [blog/sam3-native-cuda-the-dtype-maze.md](../blog/sam3-native-cuda-the-dtype-maze.md).

| GPU | ~$/hr (GCP) | Compute cap | bfloat16 | Flash Attn | Verdict |
|---|---|---|---|---|---|
| T4 | 0.35 | 7.5 (Turing) | emulated | no | Needs dtype hacks; don't |
| **L4** | **0.70** | **8.9 (Ada)** | **native** | **yes** | **Deployment default** |
| A10G | 1.00 | 8.6 (Ampere) | native | yes | Works, costlier |
| A100 | 3.00 | 8.0 (Ampere) | native | yes | Overkill for annotation |

## Cloud Run as a stateful GPU service

Cloud Run is built for stateless request handlers; this service is deliberately not one. It holds the loaded SAM3 model, the active session's frames and inference state, and the GCS sync manager in container memory. Treating Cloud Run like a VM that happens to scale to zero required four decisions:

**`max-instances=1`.** All state is in-process. A second instance would receive requests for sessions it has never seen. Single-instance is enforced at deploy time, and gunicorn is pinned to one worker (the config refuses to start with more) for the same reason.

**`concurrency=8` with in-process GPU serialization.** GPU access is serialized by an `RLock` inside `SAM3Service`, not by Cloud Run's admission control. Concurrency must exceed gunicorn's 4 threads: a propagation run is one long SSE request, and with `concurrency=1` Cloud Run would 429 every frame fetch and status poll for its entire duration.

**Scale-to-zero + GCS sync.** An idle GPU at $0.70/hr is the dominant cost, so `min-instances=0`. Everything that must survive instance death syncs to GCS: a dirty-file tracker batches uploads of `masks.json`/`prompts.json`/`state.json`, defers uploads during propagation (so sync never competes with the GPU), and flushes on session close. A SIGTERM handler gets 30s of grace to cancel propagation, kill ffmpeg, and flush; anything that still fails gets an `.unsynced` marker so the next resume re-uploads it rather than silently dropping work. The frontend sends a 5-minute heartbeat while the tab is active so the container doesn't scale away mid-session, and stops after 30 min of user inactivity so it can.

**Boot detection.** `/api/status` includes a `boot_id`; when the frontend sees it change, it knows the container was recycled and re-runs session resume instead of trusting stale client state.

### Concurrency model

One process owns everything: gunicorn is pinned to a single worker (`gunicorn_config.py` refuses to start with more — forking would silently diverge the in-process singletons) running 4 threads, with Cloud Run admitting up to 8 concurrent requests. Within that process, locks follow a strict acquire order — a thread may take a higher-numbered lock while holding a lower-numbered one, never the reverse:

| Level | Lock | Guards |
|---|---|---|
| 1 | `SAM3Service._state_lock` | `ServiceState` (phase) transitions |
| 2 | `SAM3Service._lock` (RLock) | all SAM3 predictor / GPU access |
| 3 | `session_io_lock(session_id)` | per-session file I/O (state/masks/prompts) |
| 4 | `SessionCache._lock` | the active session's in-memory file cache |
| 5 | `config._globals_lock` | the active (sync manager, session cache) slots |
| 6 | `GCSSyncManager._lock` | dirty/deferred sets and the flush timer |

Leaf locks (`SAM3Service._propagation_lock`, the lock-map guards, the session-list cache locks) are never held together with anything else. The one known deadlock signature — taking `session_io_lock` and then asking for `SAM3._lock` while a propagation holds `_lock` and calls a persist function that needs `session_io_lock` — is regression-tested in `test_sam3_service.py`. Sync-manager lifecycle is the other load-bearing invariant: the active (sync manager, session cache) pair is swapped atomically under `_globals_lock`, the outgoing manager is stopped *outside* the lock (its final flush does GCS I/O), and any flush failure persists an `.unsynced` marker so the next resume re-uploads the delta.

Comments and test names cite finding IDs (`R7`, `R8`, `H5`, …) from the internal concurrency audit that hardened this backend. The audit itself isn't shipped, but the IDs remain as stable cross-reference labels tying each invariant's implementation comment to its regression tests.

## Docker layer caching

Cold builds of the production image take ~15-20 minutes; code-only rebuilds take ~3-6. Two mechanisms make that happen:

**1. Weights live in GCS, not HuggingFace.** `facebook/sam3` is a gated HF model. Instead of passing an HF token into the build (cache-busting, secret-handling headaches) or downloading at runtime (slow cold starts, runtime token dependency), Cloud Build copies the ~7 GB of weights from a GCS bucket into the build context, and `populate_hf_cache.py` lays them out as a pre-populated HF Hub cache inside the image. The image then sets `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` so `from_pretrained()` can never try (and fail) to reach the gated repo from a tokenless container.

**2. Layer ordering + `--cache-from`.** Cloud Build's workers are ephemeral, so `cloudbuild-native.yaml` pulls the previous image first and builds with `--cache-from`. `Dockerfile.native` is ordered most-stable-first so a code change invalidates as little as possible:

```
pytorch/pytorch base image          (~5 GB, never changes)
apt deps (ffmpeg, nginx)            (rarely changes)
native sam3 clone + pip install     (rarely changes)
flash-attn                          (rarely changes)
backend/requirements.txt + pip      (changes when deps change)
model weights + HF cache layout     (changes on new SAM release)
backend code                        (changes every deploy)   ← cache cut-off lands here
frontend build (multi-stage)        (changes every deploy)
```

A typical deploy rebuilds only the last two layers; PyTorch, SAM3, Flash Attention, and the 7 GB of weights are reused from the cached image.

## Apple Silicon (MPS) constraints

Local development on a Mac runs the HF backend on MPS, with three non-obvious constraints baked into the code:

- **Autocast and `pin_memory` fail on MPS.** Autocast is monkey-patched to a nullcontext and mask post-processing falls back to CPU (`_apply_mps_patches()`). `PYTORCH_ENABLE_MPS_FALLBACK=1` is set before torch is imported.
- **Frames and inference state must live on CPU.** Keeping them on MPS creates memory pressure that degrades propagation from ~5.5s to ~70s/frame as the video grows; on CPU it holds steady (~2.9s/frame), and the per-frame CPU→MPS transfer costs ~1ms.
- **In-memory state must match disk state.** Deleting a mask also clears it from SAM3's inference state — otherwise the next propagation resurrects it from memory.

## Running it: local and cloud

**Local (Apple Silicon or NVIDIA):** follow the [README Quick Start](../README.md#quick-start-local) — conda env, `pip install -r backend/requirements.txt`, `huggingface-cli login`, `npm install`, `make dev`. Flask serves on :5555, Vite on :5173 with an `/api` proxy.

**Local Docker (NVIDIA):** `docker build -t sam3-annotator . && docker run --gpus all -p 8080:8080 -v $(pwd)/data:/data sam3-annotator`. nginx serves the built frontend and proxies `/api` to gunicorn; sessions persist in the `/data` volume.

**Cloud Run + GCS:** the full walkthrough — staging weights, Cloud Build, service account, deploy flags, and securing the service (it has **no built-in auth**) — is in [deploy-cloud-run.md](deploy-cloud-run.md).

## Further reading

- [MASK_FRAME_STREAMING.md](MASK_FRAME_STREAMING.md) — how the frontend and backend minimize bandwidth while editing: RLE-everywhere masks, version-vector delta sync, IndexedDB caches, and server-push propagation streaming.
- [blog/sam3-native-cuda-the-dtype-maze.md](../blog/sam3-native-cuda-the-dtype-maze.md) — the full T4 → L4 story: every dtype failure mode, every patch, and the benchmark data behind the GPU selection rule.

---

*Written by Danilo Gasques ([@danilogr](https://github.com/danilogr)) and Claude (Opus & Fable).*
