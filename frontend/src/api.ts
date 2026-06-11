import type { UploadResult, ObjectClass, FrameResult, SessionState, SessionSummary, BboxPadding, KeyframePrompt, TextSegmentResult, PropagationStatus, ServiceStatus, MaskResult } from "./types.ts";
import {
  findStaleFrames,
  findOrphanedFrames,
  getCachedMasks,
  putMasks,
  evictFrame,
  storeSessionMeta,
} from "./maskCache.ts";

const BASE = (import.meta.env.VITE_API_BASE as string) || "/api";

// ---- Shared-password auth (cloud deploys) -------------------------------
// Every request carries X-Auth-Token when a token is known. On any 401 the
// registered callback fires and App.tsx shows the PasswordGate. In local
// dev the backend never 401s, so none of this activates.

const AUTH_STORAGE_KEY = "sam3-auth-token";

let authToken: string | null = localStorage.getItem(AUTH_STORAGE_KEY);
let unauthorizedCallback: (() => void) | null = null;

export function setAuthToken(token: string): void {
  authToken = token;
  localStorage.setItem(AUTH_STORAGE_KEY, token);
}

export function onUnauthorized(callback: () => void): void {
  unauthorizedCallback = callback;
}

/** fetch() with the auth header injected and 401 detection. */
async function apiFetch(url: string, init?: RequestInit): Promise<Response> {
  // Headers normalizes all HeadersInit shapes (plain object, Headers, array)
  const headers = new Headers(init?.headers);
  if (authToken) headers.set("X-Auth-Token", authToken);
  const res = await fetch(url, { ...init, headers });
  if (res.status === 401) unauthorizedCallback?.();
  return res;
}

/** Check a candidate password against the backend without storing it. */
export async function verifyPassword(candidate: string): Promise<boolean> {
  try {
    const res = await fetch(`${BASE}/health`, {
      headers: { "X-Auth-Token": candidate },
    });
    return res.ok;
  } catch {
    return false;
  }
}

export async function uploadVideo(
  file: File,
  fps: number = 5,
  onProgress?: (fraction: number) => void,
  maxResolution: number = 2048,
): Promise<UploadResult> {
  const form = new FormData();
  form.append("video", file);
  form.append("fps", String(fps));
  form.append("max_resolution", String(maxResolution));

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BASE}/video/upload`);
    if (authToken) xhr.setRequestHeader("X-Auth-Token", authToken);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(e.loaded / e.total);
      }
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        if (xhr.status === 401) unauthorizedCallback?.();
        reject(new Error(xhr.responseText));
      }
    };

    xhr.onerror = () => reject(new Error("Upload failed"));
    xhr.send(form);
  });
}

// PropagationStatus is imported from types.ts and re-exported for backward compat
export type { PropagationStatus } from "./types.ts";

export async function initSegmentationModel(
  sessionId: string,
): Promise<{
  num_frames: number;
  video_height: number;
  video_width: number;
  propagation?: PropagationStatus | null;
}> {
  const res = await apiFetch(`${BASE}/segment/init/${sessionId}`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/** Construct the frame URL. */
export function getFrameUrl(sessionId: string, idx: number): string {
  return `${BASE}/video/frame/${sessionId}/${idx}`;
}

/** Fetch a frame as a Blob. Retries once on failure. */
export async function fetchFrameBlob(sessionId: string, idx: number): Promise<Blob> {
  const url = getFrameUrl(sessionId, idx);
  let res = await apiFetch(url);
  if (!res.ok) {
    // Retry once
    res = await apiFetch(url);
    if (!res.ok) throw new Error(`Frame fetch failed: ${res.status}`);
  }
  return res.blob();
}

export async function getClasses(sessionId: string): Promise<ObjectClass[]> {
  const res = await apiFetch(`${BASE}/session/classes/${sessionId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function createClass(
  sessionId: string,
  name: string,
  color: string,
): Promise<ObjectClass> {
  const res = await apiFetch(`${BASE}/session/classes/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, color }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteClass(
  sessionId: string,
  classId: number,
): Promise<void> {
  const res = await apiFetch(`${BASE}/session/classes/${sessionId}/${classId}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(await res.text());
}

export async function renameClass(
  sessionId: string,
  classId: number,
  name: string,
): Promise<void> {
  const res = await apiFetch(`${BASE}/session/classes/${sessionId}/${classId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name }),
  });
  if (!res.ok) throw new Error(await res.text());
}

export async function updateClassColor(
  sessionId: string,
  classId: number,
  color: string,
): Promise<void> {
  const res = await apiFetch(`${BASE}/session/classes/${sessionId}/${classId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ color }),
  });
  if (!res.ok) throw new Error(await res.text());
}

export async function textSegment(
  sessionId: string,
  frameIdx: number,
  text: string,
  objIdStart: number,
  signal?: AbortSignal,
): Promise<TextSegmentResult> {
  const res = await apiFetch(`${BASE}/segment/text`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      frame_idx: frameIdx,
      text,
      obj_id_start: objIdStart,
    }),
    signal,
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function clickSegment(
  sessionId: string,
  frameIdx: number,
  objId: number,
  points: number[][],
  labels: number[],
): Promise<FrameResult> {
  const res = await apiFetch(`${BASE}/segment/click`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      frame_idx: frameIdx,
      obj_id: objId,
      points,
      labels,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function boxSegment(
  sessionId: string,
  frameIdx: number,
  objId: number,
  box: [number, number, number, number],
): Promise<FrameResult> {
  const res = await apiFetch(`${BASE}/segment/box`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      frame_idx: frameIdx,
      obj_id: objId,
      box,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/**
 * Consume an SSE response of FrameResult events: parse `data:` lines,
 * surface in-band errors, stop on the `{"done": true}` sentinel, and
 * invoke `onFrame` for each result.
 */
async function consumeSseFrames(
  res: Response,
  onFrame?: (result: FrameResult) => void,
): Promise<void> {
  const reader = res.body?.getReader();
  if (!reader) return;

  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split("\n\n");
      buffer = parts.pop() ?? "";

      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data: ")) continue;
        const json = line.slice(6);
        try {
          const parsed = JSON.parse(json) as FrameResult & { done?: boolean; error?: string; traceback?: string };
          if (parsed.error) {
            const err = new Error(parsed.error) as Error & { traceback?: string };
            err.traceback = parsed.traceback;
            throw err;
          }
          if (parsed.done) return;
          if (onFrame) onFrame(parsed);
        } catch (e) {
          if (e instanceof Error && (e as Error & { traceback?: string }).traceback) throw e;
          // skip malformed events
        }
      }
    }
  } finally {
    reader.cancel().catch(() => {});
  }
}

export async function propagate(
  sessionId: string,
  startFrame?: number,
  reverse?: boolean,
  onFrame?: (result: FrameResult) => void,
  signal?: AbortSignal,
  objectIds?: number[],
): Promise<void> {
  const body: Record<string, unknown> = { session_id: sessionId };
  if (startFrame !== undefined) body.start_frame_idx = startFrame;
  if (reverse !== undefined) body.reverse = reverse;
  if (objectIds !== undefined) body.object_ids = objectIds;

  const res = await apiFetch(`${BASE}/segment/propagate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok) throw new Error(await res.text());
  return consumeSseFrames(res, onFrame);
}

export async function subscribePropagation(
  sessionId: string,
  onFrame?: (result: FrameResult) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await apiFetch(`${BASE}/segment/propagate/subscribe/${sessionId}`, {
    signal,
  });
  if (!res.ok) throw new Error(await res.text());
  return consumeSseFrames(res, onFrame);
}

export async function getPropagationStatus(
  sessionId: string,
): Promise<PropagationStatus> {
  const res = await apiFetch(`${BASE}/segment/propagation-status/${sessionId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function cancelPropagation(
  sessionId: string,
): Promise<void> {
  const res = await apiFetch(`${BASE}/segment/propagate/cancel/${sessionId}`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(await res.text());
}

export class CloseSessionUnsyncedError extends Error {
  readonly unsyncedFiles: string[];
  readonly retryPossible: boolean;

  constructor(unsyncedFiles: string[], retryPossible: boolean, message: string) {
    super(message);
    this.name = "CloseSessionUnsyncedError";
    this.unsyncedFiles = unsyncedFiles;
    this.retryPossible = retryPossible;
  }
}

export async function closeSession(
  sessionId: string,
): Promise<void> {
  const res = await apiFetch(`${BASE}/segment/close/${sessionId}`, {
    method: "POST",
  });
  if (res.status === 503) {
    let payload: { unsynced_files?: string[]; retry_possible?: boolean; error?: string } = {};
    try {
      payload = await res.json();
    } catch {
      // body not JSON — fall through with defaults
    }
    throw new CloseSessionUnsyncedError(
      payload.unsynced_files ?? [],
      payload.retry_possible ?? true,
      payload.error ?? "Some changes could not be saved to cloud storage",
    );
  }
  if (!res.ok) throw new Error(await res.text());
}

/** Fire-and-forget GCS flush via sendBeacon (works during beforeunload). */
export function flushSyncBeacon(): void {
  const url = `${BASE}/segment/flush`;
  // sendBeacon only supports Blob/FormData/URLSearchParams, not custom headers.
  // Use a keepalive fetch instead, which survives page unload.
  apiFetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
    keepalive: true,
  }).catch(() => {});
}

export async function removeObject(
  sessionId: string,
  objId: number,
): Promise<void> {
  const res = await apiFetch(`${BASE}/segment/remove_object`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, obj_id: objId }),
  });
  if (!res.ok) throw new Error(await res.text());
}

export async function reassignObject(
  sessionId: string,
  objId: number,
  newClassId: number,
): Promise<void> {
  const res = await apiFetch(
    `${BASE}/session/objects/${sessionId}/${objId}/reassign`,
    {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ class_id: newClassId }),
    },
  );
  if (!res.ok) throw new Error(`Reassign failed: ${res.status}`);
}

export async function exportCoco(
  sessionId: string,
  bboxPadding: Record<number, Record<number, BboxPadding>> = {},
): Promise<Blob> {
  const res = await apiFetch(`${BASE}/export/coco/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ bbox_padding: bboxPadding }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.blob();
}

export async function listSessions(): Promise<SessionSummary[]> {
  const res = await apiFetch(`${BASE}/video/sessions`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteSession(sessionId: string): Promise<void> {
  const res = await apiFetch(`${BASE}/video/sessions/${sessionId}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(await res.text());
}

export interface SessionMasksResponse {
  masks: Record<number, Record<number, MaskResult>>;
  versions: Record<string, number>;
}

export async function loadSessionMasks(
  sessionId: string,
): Promise<SessionMasksResponse> {
  const res = await apiFetch(`${BASE}/session/masks/${sessionId}`);
  if (!res.ok) throw new Error(await res.text());
  const raw = await res.json();
  const rawMasks: Record<string, Record<string, MaskResult>> = raw.masks ?? {};
  const versions: Record<string, number> = raw.versions ?? {};

  const masks: Record<number, Record<number, MaskResult>> = {};
  for (const [frameStr, objMasks] of Object.entries(rawMasks)) {
    const frameMasks: Record<number, MaskResult> = {};
    for (const [objStr, mask] of Object.entries(objMasks)) {
      frameMasks[Number(objStr)] = mask;
    }
    masks[Number(frameStr)] = frameMasks;
  }
  return { masks, versions };
}

export async function loadFrameMasks(
  sessionId: string,
  frameIdx: number,
): Promise<Record<number, MaskResult>> {
  const res = await apiFetch(`${BASE}/session/masks/${sessionId}/${frameIdx}`);
  if (!res.ok) throw new Error(await res.text());
  const raw: Record<string, MaskResult> = await res.json();
  const result: Record<number, MaskResult> = {};
  for (const [objStr, mask] of Object.entries(raw)) {
    result[Number(objStr)] = mask;
  }
  return result;
}

export async function fetchMaskVersions(
  sessionId: string,
): Promise<Record<string, number>> {
  const res = await apiFetch(`${BASE}/session/masks/${sessionId}/versions`);
  if (!res.ok) throw new Error(await res.text());
  const data: { versions: Record<string, number>; frame_count: number } = await res.json();
  return data.versions;
}

/**
 * Load session masks with an IndexedDB cache fast path.
 *
 * 1. Fetch lightweight `/versions` endpoint
 * 2. Compare to locally cached versions
 * 3. Load unchanged frames from IDB, fetch only stale/missing frames
 * 4. Evict orphaned frames (deleted on server) from local cache
 *
 * Falls back to full fetch if cache unavailable, too many stale frames,
 * or any error occurs.
 */
export async function loadSessionMasksSmart(
  sessionId: string,
): Promise<SessionMasksResponse> {
  try {
    const versions = await fetchMaskVersions(sessionId);
    const allFrames = Object.keys(versions).map(Number);
    const stale = await findStaleFrames(sessionId, versions);

    // If >30% of frames are stale, full fetch is more efficient than many round-trips
    const STALE_RATIO_THRESHOLD = 0.3;
    const FRAME_COUNT_THRESHOLD = 20;
    if (
      allFrames.length === 0 ||
      (allFrames.length > FRAME_COUNT_THRESHOLD &&
        stale.length / allFrames.length > STALE_RATIO_THRESHOLD)
    ) {
      return loadSessionMasks(sessionId);
    }

    const staleSet = new Set(stale);
    const masks: Record<number, Record<number, MaskResult>> = {};
    const toFetch: number[] = [];

    for (const frameIdx of allFrames) {
      if (staleSet.has(frameIdx)) {
        toFetch.push(frameIdx);
        continue;
      }
      const cached = await getCachedMasks(sessionId, frameIdx);
      if (cached && Object.keys(cached).length > 0) {
        masks[frameIdx] = cached;
      } else {
        toFetch.push(frameIdx);
      }
    }

    // Fetch stale + missing frames in parallel (capped)
    const CONCURRENCY = 8;
    for (let i = 0; i < toFetch.length; i += CONCURRENCY) {
      const batch = toFetch.slice(i, i + CONCURRENCY);
      await Promise.all(
        batch.map(async (frameIdx) => {
          const frameMasks = await loadFrameMasks(sessionId, frameIdx);
          if (Object.keys(frameMasks).length > 0) {
            masks[frameIdx] = frameMasks;
          }
          putMasks(sessionId, frameIdx, frameMasks, versions[frameIdx] ?? 0).catch(() => {});
        }),
      );
    }

    // Reconcile: evict frames that exist locally but not on server (deleted elsewhere)
    const orphans = await findOrphanedFrames(sessionId, versions);
    await Promise.all(orphans.map((f) => evictFrame(sessionId, f)));

    // Update cached version metadata
    storeSessionMeta(sessionId, versions).catch(() => {});

    return { masks, versions };
  } catch {
    // Any error — fall back to full fetch
    return loadSessionMasks(sessionId);
  }
}

export async function deleteFrameMask(
  sessionId: string,
  frameIdx: number,
  objId: number,
): Promise<void> {
  const res = await apiFetch(`${BASE}/session/masks/${sessionId}/${frameIdx}/${objId}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(await res.text());
}

export async function deleteFrameMasksBySource(
  sessionId: string,
  objId: number,
  sourceKeyframe: number | null,
  direction: "left" | "right" | "this",
  currentFrame: number,
): Promise<{ deleted_frames: number[] }> {
  const res = await apiFetch(`${BASE}/session/masks/${sessionId}/${objId}/by-source`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      source_keyframe: sourceKeyframe,
      direction,
      current_frame: currentFrame,
    }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function deleteFrameMasksBatch(
  sessionId: string,
  objId: number,
  frameIndices: number[],
): Promise<{ deleted_frames: number[]; deleted_count: number }> {
  const res = await apiFetch(`${BASE}/session/masks/${sessionId}/${objId}/batch`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ frame_indices: frameIndices }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function getSessionState(
  sessionId: string,
): Promise<SessionState> {
  const res = await apiFetch(`${BASE}/session/state/${sessionId}`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export class StateVersionConflictError extends Error {
  readonly currentVersion: number;
  readonly currentState: SessionState;

  constructor(currentVersion: number, currentState: SessionState) {
    super("state_version_conflict");
    this.name = "StateVersionConflictError";
    this.currentVersion = currentVersion;
    this.currentState = currentState;
  }
}

export async function putSessionState(
  sessionId: string,
  state: SessionState,
): Promise<number> {
  const res = await apiFetch(`${BASE}/session/state/${sessionId}`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(state),
  });
  if (res.status === 409) {
    let payload: { error?: string; current_version?: number; state?: SessionState } = {};
    try {
      payload = await res.json();
    } catch {
      // body not JSON — fall through
    }
    if (payload.error === "state_version_conflict" && payload.state !== undefined) {
      throw new StateVersionConflictError(
        payload.current_version ?? 0,
        payload.state,
      );
    }
    // wipe-fingerprint 409: preserve existing text-throw contract
    throw new Error(payload.error ?? "Conflict");
  }
  if (!res.ok) throw new Error(await res.text());
  const body: { ok: boolean; version?: number } = await res.json();
  return body.version ?? 0;
}

export async function loadPrompts(
  sessionId: string,
): Promise<Record<number, Record<number, KeyframePrompt>>> {
  const res = await apiFetch(`${BASE}/session/prompts/${sessionId}`);
  if (!res.ok) throw new Error(await res.text());
  const raw: Record<string, Record<string, KeyframePrompt>> = await res.json();
  const result: Record<number, Record<number, KeyframePrompt>> = {};
  for (const [frameStr, objPrompts] of Object.entries(raw)) {
    const inner: Record<number, KeyframePrompt> = {};
    for (const [objStr, prompt] of Object.entries(objPrompts)) {
      inner[Number(objStr)] = prompt;
    }
    result[Number(frameStr)] = inner;
  }
  return result;
}

export async function recalculatePrompt(
  sessionId: string,
  frameIdx: number,
  objId: number,
): Promise<FrameResult> {
  const res = await apiFetch(`${BASE}/segment/recalculate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, frame_idx: frameIdx, obj_id: objId }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function exportSession(
  sessionId: string,
  includeVideo: boolean = false,
): Promise<Blob> {
  const res = await apiFetch(`${BASE}/export/session/${sessionId}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ include_video: includeVideo }),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.blob();
}

export interface ImportSessionResult {
  session_id: string;
  frame_count: number;
  original_name: string;
  fps: number;
  duplicate?: boolean;
}

export async function importSession(
  file: File,
  onProgress?: (fraction: number) => void,
): Promise<ImportSessionResult> {
  const form = new FormData();
  form.append("file", file);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BASE}/export/import-session`);
    if (authToken) xhr.setRequestHeader("X-Auth-Token", authToken);
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(e.loaded / e.total);
      }
    };

    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(JSON.parse(xhr.responseText));
      } else {
        if (xhr.status === 401) unauthorizedCallback?.();
        reject(new Error(xhr.responseText));
      }
    };

    xhr.onerror = () => reject(new Error("Import failed"));
    xhr.send(form);
  });
}

export async function openSession(sessionId: string): Promise<void> {
  const res = await apiFetch(`${BASE}/session/open/${sessionId}`, {
    method: "POST",
  });
  if (!res.ok) throw new Error(await res.text());
}

export interface HealthResponse {
  ok: boolean;
  boot_id: string;
  uptime_s: number;
}

/** Lightweight health check — no SAM3 lock; auth handled by apiFetch. */
export async function checkHealth(): Promise<HealthResponse> {
  const res = await apiFetch(`${BASE}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

/** Full service status. Returns phase, progress, session info. */
export async function getServiceStatus(): Promise<ServiceStatus> {
  const resp = await apiFetch(`${BASE}/status`);
  if (!resp.ok) throw new Error(`Status check failed: ${resp.status}`);
  return resp.json();
}

/** Resume an existing session — starts the background pipeline for model init. */
export async function resumeSession(
  sessionId: string,
): Promise<{ session_id: string; video_name: string }> {
  const resp = await apiFetch(`${BASE}/session/resume/${sessionId}`, {
    method: "POST",
  });
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`Resume failed (${resp.status}): ${text}`);
  }
  return resp.json();
}

/** Cancel the current background pipeline job (extraction or init). */
export async function cancelJob(): Promise<{ status: string }> {
  const resp = await apiFetch(`${BASE}/job/cancel`, {
    method: "POST",
  });
  if (!resp.ok) throw new Error(`Cancel failed: ${resp.status}`);
  return resp.json();
}

/** Clear the error phase on the backend so the UI can return to the session list. */
export async function dismissPipelineError(): Promise<{ status: string }> {
  const resp = await apiFetch(`${BASE}/status/dismiss-error`, {
    method: "POST",
  });
  if (!resp.ok) throw new Error(`Dismiss error failed: ${resp.status}`);
  return resp.json();
}
