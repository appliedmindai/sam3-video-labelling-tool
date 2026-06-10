# Mask & Frame Streaming: Bandwidth Optimizations

A frame-by-frame video annotation UI has a brutal data profile: the browser needs **every extracted frame** for scrubbing, and **every mask of every object on every frame** for rendering, hit-testing, and review. Done naively — refetch frames on every scrub, ship masks as bitmaps, re-download the whole mask set after every edit — a single editing session moves gigabytes and feels like molasses on anything but localhost. This matters doubly in cloud mode, where the backend is a Cloud Run container that may be a continent away.

This document describes the optimizations that keep the wire quiet while editing. The guiding principles:

1. **Pixels cross the network once.** Frames are immutable after extraction; cache them on the client forever (bounded by LRU).
2. **Masks are never bitmaps.** Compressed RLE end-to-end, decoded straight onto the canvas.
3. **Send only what changed.** Per-frame version counters turn "reload the session" into a handful of small requests.
4. **Push, don't poll.** Propagation streams results over one HTTP response; nothing re-asks for what the server already pushed.

Relevant code: `frontend/src/frameCache.ts`, `frontend/src/maskCache.ts`, `frontend/src/api.ts` (`loadSessionMasksSmart`, `propagate`, `subscribePropagation`), `frontend/src/components/VideoCanvas.tsx` (`decodeRleCounts`, `drawMaskFromRle`, `hitTestMask`), `backend/app/services/mask_storage.py`, `backend/app/routes/segment.py`, `backend/app/services/gcs_sync.py`.

---

## 1. Frames: download once, then never again

### Shrink the source first

The browser never sees the video file. At upload time the backend extracts JPEG stills at a configurable rate (FPS picker in the upload dialog) and caps the longest edge (`max_resolution`, default 2048 px). Annotation doesn't need 60 fps or 4K — a 5-minute clip at 2 fps is ~600 frames of modest JPEGs instead of a multi-hundred-MB video the client would have to seek through.

### IndexedDB blob cache (`frameCache.ts`)

Every frame fetched from `/api/video/frame/<session>/<idx>` is stored as a Blob in IndexedDB and served to the canvas as a `blob:` URL from then on. Three tiers:

1. **In-memory map** `frameKey → blob URL` — repeat views of a frame within a page lifetime cost nothing, not even an IDB read.
2. **IndexedDB** — survives reloads. Reopening a session yesterday's tab already cached does **zero** frame requests.
3. **Network** — only on a true miss, and the fetch immediately back-fills both tiers.

### Background prefetch with resume

When a session opens, `cacheSession()` eagerly downloads all frames in the background (visible as a "caching frames" progress indicator) so scrubbing never stalls on the network:

- Fetches in **batches of 4** with `Promise.allSettled` — enough parallelism to saturate the link without starving the API requests the user is actively generating, and one failed frame doesn't abort the run.
- **Skips frames already in IDB** by scanning the session's key index first — an interrupted prefetch resumes where it left off instead of restarting from frame 0.
- `isSessionCached()` short-circuits the whole thing when the session is already complete.

### Bounded storage

Both frame and mask caches keep an LRU of **5 sessions** keyed by `lastAccessedAt`; evicting a session deletes its blobs, metadata, and revokes its in-memory blob URLs. A handful of cached sessions is a few hundred MB of disk — cheap — but unbounded growth would eventually hit browser storage quotas and start failing writes. Every cache path also degrades gracefully: if IndexedDB is unavailable (private browsing, quota exceeded), the app falls back to direct fetches and memory-only caching with a single console warning.

**Net effect:** each frame crosses the network exactly once per device, amortized over every scrub, zoom, propagation review, and page reload that follows.

---

## 2. Masks: compressed RLE end-to-end

A mask is a binary bitmap the size of a frame. Shipped raw at 1080p that's ~2 MB per object per frame; even PNG-encoded it's tens to hundreds of KB. Multiply by objects × frames and "show me my session" becomes a gigabyte problem.

Instead, masks live and travel as **COCO compressed RLE** (pycocotools format: column-major run lengths, LEB128-style delta-encoded into a compact ASCII `counts` string). A typical object mask is **a few KB** — two to three orders of magnitude smaller than the bitmap it describes — and the encoding is done once, on the backend, at inference time.

The frontend never inflates RLE into a full bitmap either:

- **`drawMaskFromRle`** decodes runs directly into the canvas `ImageData`, skipping background runs in O(1) per run instead of touching every pixel.
- **`hitTestMask`** answers "did the user click inside this mask?" by walking the run-length counts to the clicked pixel's linear index (`col * height + row` — the column-major ordering is load-bearing) without materializing anything.
- The backend attaches **`bbox` and `area`** to every mask payload, so zoom-to-object, sorting, and layout never require a decode at all.

---

## 3. Editing: version-vector delta sync

The masks for a session are persisted in one `masks.json`, but the frontend must not re-download all of it after every click — a correction click on frame 412 changes one frame, not six hundred.

### Per-frame version counters

`masks.json` carries a `_versions` dict: a **monotonic counter per frame**, bumped by every mutation that touches that frame (new mask, propagation write, deletion, batch delete). The counters ride along at zero extra cost on writes the backend was doing anyway.

### The sync protocol (`loadSessionMasksSmart`)

On session load / refresh / external-change detection:

1. `GET /session/masks/<id>/versions` — a **tiny** payload (one integer per annotated frame), no mask data.
2. Diff against the version metadata stored in IndexedDB (`findStaleFrames`): a frame is stale if it's missing locally or its server counter is higher.
3. Serve every up-to-date frame **from IndexedDB**. Fetch only the stale ones via the per-frame endpoint `GET /session/masks/<id>/<frame>`, **8 requests in flight at a time**, writing each result back to the cache with its new version.
4. `findOrphanedFrames` reconciles deletions: frames present locally but absent from the server's version dict (deleted from another device, or by a destructive edit) get evicted rather than resurrected.

### The 30% heuristic

Per-frame round-trips beat a full fetch only when few frames changed. If a session has **> 20 annotated frames and > 30% of them are stale**, `loadSessionMasksSmart` abandons the delta path and issues one bulk `GET /session/masks/<id>` — request overhead would otherwise eat the savings. Any error on the smart path (IDB unavailable, malformed metadata) falls back to the same full fetch, so the optimization can only ever help, never break loading.

### Two-tier mask cache (`maskCache.ts`)

Same shape as the frame cache: an in-memory map (LRU, 128 frame-entries) for instant repeat access, IndexedDB (LRU, 5 sessions) for persistence, write-through on every edit so the cache is always as fresh as the UI. Each cached frame stores its version, which is what makes step 2 above a pure-local computation.

**Net effect:** the steady-state cost of "reopen the session and continue editing" is one versions request plus one small RLE payload per frame you actually touched since last time.

---

## 4. Propagation: server-push streaming over SSE

Propagation is the highest-traffic moment in the app: SAM3 produces masks for potentially hundreds of frames in one run. Polling for results would mean re-fetching a growing mask set on a timer; instead, results are **pushed**.

### One request, many results

`POST /segment/propagate` holds the response open as a **Server-Sent Events stream**. As each frame finishes inference, the backend yields one event:

```
data: {"frame_idx": 137, "masks": {"2": {"rle": …, "bbox": …, "area": …, "confidence": 0.91}}, …}
```

Each event carries only that frame's RLE masks — a few KB — and the client paints it immediately, so the user watches the propagation sweep across the timeline in real time. A `{"done": true}` sentinel terminates the stream; errors travel in-band with tracebacks. The client reads the stream with `fetch` + `ReadableStream` (buffer-split on the SSE delimiter), and cancellation is an `AbortSignal` plus a cancel endpoint — no result is ever transferred twice and no polling loop ever runs.

### Don't let persistence fight the stream

While events arrive, the client writes masks to the **in-memory cache only** (`putMasksMemoryOnly`) — hundreds of per-frame IndexedDB transactions would jank the very canvas the user is watching. When the run completes (or is cancelled), `flushToIDB` persists everything for the session in a **single IDB transaction**, then refreshes the stored version vector so the next delta sync knows these frames are current.

### Reconnect without recompute

If the tab reloads mid-propagation (or the laptop briefly drops Wi-Fi), the run keeps going server-side — and `GET /segment/propagate/subscribe/<id>` attaches a new SSE stream to it. The backend fans results out to per-subscriber **bounded queues**; on overflow it drops the oldest *data* event for that subscriber (the UI heals via delta sync) but guarantees delivery of error/done sentinels so streams always terminate cleanly. The reconnecting client receives only the frames computed from that point on, and the version-vector sync (§3) backfills the ones it missed — the GPU never re-runs a frame for the benefit of a flaky connection.

This is also why the tool works through Cloud Run: SSE is plain HTTP, survives the proxy chain (`proxy_buffering off` in nginx), and one propagation run costs exactly one admission slot.

---

## 5. Cloud mode: the same idea, pointed at GCS

In cloud mode the container's disk is ephemeral, so sessions sync to GCS — and the same "don't re-send what didn't change" discipline applies on that leg:

- **Dirty-file tracking.** The sync manager uploads only files marked dirty since the last flush, on a periodic timer, not the whole session directory.
- **Propagation deferral.** During a propagation run, `masks.json` is rewritten after *every frame* — uploading it each time would race the GPU for CPU and egress hundreds of copies of a growing file. Writes to it are instead **deferred**: the sync manager parks them in a separate set and promotes them to the dirty set when propagation ends, so the file uploads **once**, after the run.
- **Flush at the boundaries.** Session close, SIGTERM (container shutdown), and a `keepalive`-fetch beacon on tab unload all force a final flush; failures write an `.unsynced` marker so the next resume re-uploads exactly the delta.

---

## 6. What this adds up to

| Scenario | Naive cost | With these optimizations |
|---|---|---|
| Scrub through a 600-frame session, twice | 1,200 frame fetches | 600 (first pass, prefetched in background) — then 0, forever |
| Reload the tab on a cached session | 600 frames + full mask set | 1 versions request (~KB) |
| Fix one bad mask, reload elsewhere | full mask set | 1 versions request + 1 frame's RLE (~KB) |
| Render/hit-test masks on a frame | decode + ship bitmaps (MB each) | RLE (~KB each), decoded run-wise on canvas |
| Watch a 300-frame propagation | poll + refetch growing mask set | 300 SSE events, each frame's RLE exactly once |
| Tab reload mid-propagation | restart the run, recompute on GPU | re-subscribe; missed frames backfilled via delta sync |
| GCS sync during propagation (cloud) | hundreds of masks.json uploads | one upload, after the run |

None of this changes correctness semantics: every cache is reconcilable from the server's version vector, every fallback path is a full fetch, and the server's `masks.json` remains the single source of truth.
