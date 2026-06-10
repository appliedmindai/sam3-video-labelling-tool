import { useState, useCallback, useEffect, useMemo, useRef } from "react";
import type { ObjectClass, TrackedObject, MaskResult, FrameResult, ClickPoint, BboxPadding, KeyframePrompt, ConfidenceWarning } from "./types.ts";
import {
  clickSegment,
  boxSegment,
  textSegment,
  propagate,
  subscribePropagation,
  getPropagationStatus,
  cancelPropagation,
  closeSession,
  CloseSessionUnsyncedError,
  flushSyncBeacon,
  removeObject,
  putSessionState,
  StateVersionConflictError,
  getSessionState,
  loadSessionMasksSmart,
  loadFrameMasks,
  fetchMaskVersions,
  deleteFrameMask,
  deleteFrameMasksBatch,
  loadPrompts,
  recalculatePrompt,
  createClass,
  checkHealth,
  dismissPipelineError,
} from "./api.ts";
import {
  getCachedMasks,
  putMasks,
  putMasksMemoryOnly,
  flushToIDB,
  clearSessionMemory,
  evictSession,
  storeSessionMeta,
} from "./maskCache.ts";
import { cacheSession, isSessionCached } from "./frameCache.ts";
import { useServiceStatus } from "./hooks/useServiceStatus";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Slider } from "@/components/ui/slider";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Loader2, RefreshCw, Video, X } from "lucide-react";
import { timeAgo } from "./utils.ts";
import StatusToast from "./components/StatusToast.tsx";
import VideoUpload from "./components/VideoUpload.tsx";
import PipelineProgress from "./components/PipelineProgress.tsx";
import VideoCanvas from "./components/VideoCanvas.tsx";
import FrameNavigator from "./components/FrameNavigator.tsx";
import AnnotationPanel, { PALETTE } from "./components/AnnotationPanel.tsx";
import PropagationBar from "./components/PropagationBar.tsx";
import ExportDialog from "./components/ExportDialog.tsx";
import ExportSessionDialog from "./components/ExportSessionDialog.tsx";
import ToolBar from "./components/ToolBar.tsx";
import ConfirmDialog from "./components/ConfirmDialog.tsx";
import ErrorDialog from "./components/ErrorDialog.tsx";
import ObjectChoiceDialog from "./components/ObjectChoiceDialog.tsx";
import ClassChoiceDialog from "./components/ClassChoiceDialog.tsx";
import DeleteMasksPanel from "./components/DeleteMasksPanel.tsx";
import ThemeToggle from "./components/ThemeToggle.tsx";

type PendingAnnotation =
  | { type: "click"; x: number; y: number; label: number; frameIdx: number }
  | { type: "box"; box: [number, number, number, number]; frameIdx: number };

// Confidence below this threshold triggers a degradation warning
const CONFIDENCE_THRESHOLD = 0.75;

function getPaddingForKeyframe(
  bboxPadding: Record<number, Record<number, BboxPadding>>,
  objId: number,
  sourceKeyframe: number | null,
  frameIdx: number,
): BboxPadding {
  const zero: BboxPadding = { top: 0, bottom: 0, left: 0, right: 0 };
  const kf = sourceKeyframe ?? frameIdx;
  return bboxPadding[objId]?.[kf] ?? zero;
}

function App() {
  // Server-driven UI state — polls /api/status for phase-based routing
  const { status: serviceStatus, refetch: refetchStatus } = useServiceStatus();

  const lastActivityRef = useRef(Date.now());

  const [sessionId, setSessionId] = useState<string | null>(null);
  const sessionIdRef = useRef<string | null>(null);
  useEffect(() => { sessionIdRef.current = sessionId; }, [sessionId]);
  const [frameCount, setFrameCount] = useState(0);
  const [currentFrame, setCurrentFrame] = useState(0);

  const [classes, setClasses] = useState<ObjectClass[]>([]);
  const [objects, setObjects] = useState<TrackedObject[]>([]);
  const [selectedClassId, setSelectedClassId] = useState<number | null>(null);
  const [selectedObjIds, setSelectedObjIds] = useState<Set<number>>(new Set());
  const selectedObjId = selectedObjIds.size === 1 ? [...selectedObjIds][0] : null;

  const [toolMode, setToolMode] = useState<"pointer" | "click" | "box">("pointer");

  const [masks, setMasks] = useState<
    Record<number, Record<number, MaskResult>>
  >({});
  // Ref mirrors masks state — used in callbacks to avoid capturing the full object
  const masksRef = useRef<Record<number, Record<number, MaskResult>>>({});
  masksRef.current = masks;
  // Per-frame set of obj_ids that have masks — survives mask LRU eviction
  const [frameObjIndex, setFrameObjIndex] = useState<Map<number, Set<number>>>(new Map());

  const [propagating, setPropagating] = useState(false);
  const propagatingRef = useRef(false);
  propagatingRef.current = propagating;
  const [propagationProgress, setPropagationProgress] = useState(0);
  const propagateAbortRef = useRef<AbortController | null>(null);

  const [confidenceWarnings, setConfidenceWarnings] = useState<ConfidenceWarning[]>([]);
  const [lowConfidenceFrames, setLowConfidenceFrames] = useState<Set<number>>(new Set());

  const [showExport, setShowExport] = useState(false);
  const [showSessionExport, setShowSessionExport] = useState(false);
  const [showDeletePanel, setShowDeletePanel] = useState(false);

  const [confirmDialog, setConfirmDialog] = useState<{
    title: string;
    description: string;
    confirmLabel: string;
    onConfirm: () => void;
    cancelLabel?: string;
    onCancel?: () => void;
  } | null>(null);

  const [errorDialog, setErrorDialog] = useState<{
    title: string;
    message: string;
    traceback?: string;
  } | null>(null);

  const [clickPoints, setClickPoints] = useState<ClickPoint[]>([]);
  const [prompts, setPrompts] = useState<Record<number, Record<number, KeyframePrompt>>>({});
  // Per-object, per-frame padding keyframes: objId → frameIdx → padding
  const [bboxPadding, setBboxPadding] = useState<Record<number, Record<number, BboxPadding>>>({});
  const [segmenting, setSegmenting] = useState(false);
  const [textPromptLoading, setTextPromptLoading] = useState(false);
  const [detectAbortController, setDetectAbortController] = useState<AbortController | null>(null);

  const [objectChoice, setObjectChoice] = useState<{
    existingObjs: TrackedObject[];
    pendingAction: PendingAnnotation;
  } | null>(null);

  const [orphanObjId, setOrphanObjId] = useState<number | null>(null);

  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackFps, setPlaybackFps] = useState(5);


  const [hiddenClassIds, setHiddenClassIds] = useState<Set<number>>(new Set());

  const [videoName, setVideoName] = useState<string | null>(null);
  const [lastSaved, setLastSaved] = useState<Date | null>(null);
  // Tick counter to force re-render so timeAgo display stays fresh
  const [, setTimeAgoTick] = useState(0);

  const [status, setStatus] = useState("Ready");
  const [closingSession, setClosingSession] = useState(false);

  const [cachingFrames, setCachingFrames] = useState(false);
  const [cachingProgress, setCachingProgress] = useState({ cached: 0, total: 0 });
  const [refreshingMasks, setRefreshingMasks] = useState(false);

  // Refresh timeAgo display every 30 seconds
  useEffect(() => {
    if (!lastSaved) return;
    const interval = setInterval(() => setTimeAgoTick((t) => t + 1), 30_000);
    return () => clearInterval(interval);
  }, [lastSaved]);

  // Track user activity — heartbeat stops after 30 min idle to let container scale down
  useEffect(() => {
    const onActivity = () => { lastActivityRef.current = Date.now(); };
    const events = ["mousemove", "keydown", "mousedown", "touchstart", "scroll"] as const;
    events.forEach((e) => window.addEventListener(e, onActivity, { passive: true }));
    return () => { events.forEach((e) => window.removeEventListener(e, onActivity)); };
  }, []);

  // Heartbeat — keepalive only (session loss detected via /api/status polling).
  // In cloud mode this keeps the Cloud Run container from scaling to zero
  // while the tab is active; locally it's a cheap no-op ping.
  useEffect(() => {
    const HEARTBEAT_MS = 5 * 60 * 1000;
    const IDLE_TIMEOUT_MS = 30 * 60 * 1000;
    const tick = async () => {
      if (Date.now() - lastActivityRef.current > IDLE_TIMEOUT_MS) return;
      try { await checkHealth(); } catch { /* server unreachable — will retry next tick */ }
    };
    void tick();
    const id = setInterval(tick, HEARTBEAT_MS);
    return () => clearInterval(id);
  }, []);

  // Detect session loss (container recycle): if phase drops from "ready" to "idle"
  const prevPhaseRef = useRef<string | null>(null);
  useEffect(() => {
    if (!serviceStatus) return;
    const prev = prevPhaseRef.current;
    prevPhaseRef.current = serviceStatus.phase;

    // If we were in "ready" and now idle, the session was lost
    if (prev === "ready" && serviceStatus.phase === "idle") {
      sessionLoadedRef.current = null;
      setSessionId(null);
      setStatus("Session lost — server restarted. Please select a session.");
    }
  }, [serviceStatus?.phase]); // eslint-disable-line react-hooks/exhaustive-deps

  // Track in-flight per-frame mask fetches to avoid duplicate requests
  const loadingFramesRef = useRef<Set<number>>(new Set());

  const runFrameCaching = useCallback(async (sid: string, fc: number) => {
    const alreadyCached = await isSessionCached(sid);
    if (alreadyCached) return;
    setCachingFrames(true);
    setCachingProgress({ cached: 0, total: fc });
    try {
      await cacheSession(sid, fc, (cached, total) => {
        setCachingProgress({ cached, total });
      });
    } finally {
      setCachingFrames(false);
    }
  }, []);

  const reconnectPropagation = useCallback(
    async (sid: string, totalFrames: number) => {
      const collectedWarnings: ConfidenceWarning[] = [];
      try {
        const status = await getPropagationStatus(sid);
        if (status.status !== "running") return;

        const abortController = new AbortController();
        propagateAbortRef.current = abortController;
        setPropagating(true);
        setConfidenceWarnings([]);
        setLowConfidenceFrames(new Set());
        setPropagationProgress(
          totalFrames > 0 && status.frames_processed
            ? status.frames_processed / totalFrames
            : 0,
        );

        const missedFrameCount = status.frames_processed ?? 0;
        if (missedFrameCount > 0) {
          setStatus(`Reconnected — loading ${missedFrameCount} missed frame(s)...`);
        } else {
          setStatus("Reconnected to propagation...");
        }

        let framesProcessed = status.frames_processed ?? 0;

        const onFrame = (result: FrameResult) => {
          framesProcessed++;
          setPropagationProgress(totalFrames > 0 ? framesProcessed / totalFrames : 0);
          updateMasksFromResult(result);
          setCurrentFrame(result.frame_idx);

          for (const [objIdStr, maskData] of Object.entries(result.masks)) {
            if (maskData.confidence != null && maskData.confidence < CONFIDENCE_THRESHOLD) {
              collectedWarnings.push({
                frame_idx: result.frame_idx,
                obj_id: Number(objIdStr),
                confidence: maskData.confidence,
              });
            }
          }
          if (collectedWarnings.length > 0) {
            setConfidenceWarnings([...collectedWarnings]);
          }
        };

        // Load missed frames in the background (frames processed before reconnect).
        // This runs concurrently with subscribePropagation — does not block live updates.
        if (missedFrameCount > 0 && status.start_frame != null) {
          const startFrame = status.start_frame;
          const reverse = status.reverse ?? false;
          const missedFrames: number[] = [];
          for (let i = 0; i < missedFrameCount; i++) {
            const frameIdx = reverse ? startFrame - i : startFrame + i;
            if (frameIdx >= 0 && frameIdx < totalFrames) {
              missedFrames.push(frameIdx);
            }
          }
          // Load missed frames in background — don't await
          (async () => {
            for (const frameIdx of missedFrames) {
              if (abortController.signal.aborted) break;
              try {
                const frameMasks = await loadFrameMasks(sid, frameIdx);
                if (Object.keys(frameMasks).length > 0) {
                  putMasks(sid, frameIdx, frameMasks).catch(() => {});
                  setMasks((prev) => ({ ...prev, [frameIdx]: frameMasks }));
                }
              } catch {
                // Ignore errors loading individual missed frames
              }
            }
          })();
        }

        await subscribePropagation(sid, onFrame, abortController.signal);

        if (collectedWarnings.length > 0) {
          setStatus(`Propagation complete — ${new Set(collectedWarnings.map((w) => w.frame_idx)).size} frame(s) with low confidence`);
        } else {
          setStatus("Propagation complete");
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") {
          setStatus("Propagation cancelled");
        } else {
          const msg = err instanceof Error ? err.message : String(err);
          const tb = (err as Error & { traceback?: string })?.traceback;
          setStatus(`Propagation error: ${msg}`);
          setErrorDialog({
            title: "Propagation failed",
            message: msg,
            traceback: tb,
          });
        }
      } finally {
        propagateAbortRef.current = null;
        setPropagating(false);
        setPropagationProgress(0);
        setLowConfidenceFrames(new Set(collectedWarnings.map((w) => w.frame_idx)));
        flushToIDB(sid).catch(() => {});
        fetchMaskVersions(sid)
          .then((v) => storeSessionMeta(sid, v))
          .catch(() => {});
      }
    },
    [],
  );

  // Load session data from backend — called when pipeline reaches "ready"
  const loadSession = useCallback(async (sid: string, fc: number, videoName: string) => {
    // Clear the loaded-guard: we're starting a fresh load, so subsequent
    // auto-saves must wait until this load completes successfully.
    sessionLoadedRef.current = null;
    setSessionId(sid);
    setFrameCount(fc);
    setCurrentFrame(0);
    setMasks({});
    setFrameObjIndex(new Map());
    setSelectedClassId(null);
    setSelectedObjIds(new Set());
    setClickPoints([]);
    setPrompts({});
    setVideoName(videoName);
    setLastSaved(new Date());

    await runFrameCaching(sid, fc);

    let loadedObjects: TrackedObject[] = [];
    let stateLoadedOk = false;
    try {
      const state = await getSessionState(sid);
      setClasses(state.classes || []);
      loadedObjects = state.objects || [];
      setObjects(loadedObjects);
      setBboxPadding(state.bbox_padding || {});
      // Anchor lost-update counter so subsequent PUTs send If-Match-like version.
      stateVersionRef.current = state.version ?? 0;
      stateLoadedOk = true;
    } catch {
      setClasses([]);
      setObjects([]);
      setBboxPadding({});
      stateVersionRef.current = 0;
    }
    let masksLoadedOk = false;
    try {
      const { masks: savedMasks, versions } = await loadSessionMasksSmart(sid);
      // Build per-frame obj_id index from full dataset before evicting
      const index = new Map<number, Set<number>>();
      for (const [frameStr, objMasks] of Object.entries(savedMasks)) {
        const fi = Number(frameStr);
        const objIds = Object.keys(objMasks).map(Number).filter((id) => objMasks[id] != null);
        if (objIds.length > 0) index.set(fi, new Set(objIds));
      }
      setFrameObjIndex(index);
      // Detect orphaned masks (obj_ids in masks but not in objects)
      const knownObjIds = new Set(loadedObjects.map((o) => o.obj_id));
      const allMaskObjIds = new Set<number>();
      for (const objIds of index.values()) {
        for (const id of objIds) allMaskObjIds.add(id);
      }
      const orphanIds = [...allMaskObjIds].filter((id) => !knownObjIds.has(id));
      if (orphanIds.length > 0) {
        const orphans: TrackedObject[] = orphanIds.map((id) => ({ obj_id: id, class_id: -1 }));
        loadedObjects = [...loadedObjects, ...orphans];
        setObjects(loadedObjects);
      }
      // Seed mask cache with loaded masks (memory-only to avoid IDB write storm)
      for (const [frameKey, objMasks] of Object.entries(savedMasks)) {
        const fi = Number(frameKey);
        const frameVersion = versions[String(fi)] ?? 0;
        putMasksMemoryOnly(sid, fi, objMasks, frameVersion);
      }
      // Store version metadata for future session resume
      storeSessionMeta(sid, versions).catch(() => {});
      const allAnnotated = index;
      setMasks(savedMasks);
      const maskFrameCount = allAnnotated.size;
      let totalObjs = 0;
      for (const s of index.values()) totalObjs += s.size;
      console.info(
        `[load] session=${sid} frames=${maskFrameCount} objects-total=${totalObjs}`,
      );
      setStatus(
        maskFrameCount > 0
          ? `Session loaded with ${maskFrameCount} annotated frames.`
          : "Session loaded. Continue annotating.",
      );
      masksLoadedOk = true;
    } catch {
      setStatus("Session loaded. Continue annotating.");
    }

    // Only permit auto-save to write state.json if BOTH loads succeeded.
    // If either failed, we might have classes=[] or objects=[] not because
    // that's the truth but because the load failed — never overwrite disk
    // with that.
    if (stateLoadedOk && masksLoadedOk) {
      sessionLoadedRef.current = sid;
    }

    try {
      const savedPrompts = await loadPrompts(sid);
      setPrompts(savedPrompts);
      // Reconstruct clickPoints from persisted click prompts so dots
      // appear on canvas and undo works after resume
      const restoredPoints: ClickPoint[] = [];
      for (const [frameStr, objPrompts] of Object.entries(savedPrompts)) {
        const frameIdx = Number(frameStr);
        for (const [objStr, prompt] of Object.entries(objPrompts)) {
          if (prompt.type === "click" && prompt.points && prompt.labels) {
            const objId = Number(objStr);
            for (let i = 0; i < prompt.points.length; i++) {
              restoredPoints.push({
                x: prompt.points[i][0],
                y: prompt.points[i][1],
                label: prompt.labels[i],
                frameIdx,
                objId,
              });
            }
          }
        }
      }
      if (restoredPoints.length > 0) {
        setClickPoints(restoredPoints);
      }
    } catch {
      // silent — prompts may not exist for older sessions
    }
    // Fire-and-forget: reconnect to any in-progress propagation
    reconnectPropagation(sid, fc);
  }, [reconnectPropagation, runFrameCaching]);

  // Auto-load session data when pipeline completes (phase transitions to "ready")
  const hasLoadedSessionRef = useRef<string | null>(null);
  useEffect(() => {
    if (!serviceStatus) return;
    if (serviceStatus.phase !== "ready" || !serviceStatus.session_id) return;
    if (hasLoadedSessionRef.current === serviceStatus.session_id) return;

    hasLoadedSessionRef.current = serviceStatus.session_id;
    loadSession(serviceStatus.session_id, serviceStatus.frame_count ?? 0, serviceStatus.video_name ?? "");
  }, [serviceStatus?.phase, serviceStatus?.session_id, loadSession]); // eslint-disable-line react-hooks/exhaustive-deps

  // Flush dirty GCS files when the user closes the tab or navigates away.
  // Safari/iOS and modern Chrome with bfcache may skip beforeunload but fire pagehide.
  useEffect(() => {
    const handleUnload = (): void => {
      if (sessionId) {
        flushSyncBeacon();
      }
    };
    window.addEventListener("beforeunload", handleUnload);
    window.addEventListener("pagehide", handleUnload);
    return () => {
      window.removeEventListener("beforeunload", handleUnload);
      window.removeEventListener("pagehide", handleUnload);
    };
  }, [sessionId]);

  // Tracks the sessionId for which BOTH getSessionState and loadSessionMasksSmart
  // have successfully resolved. The bboxPadding auto-save and all other
  // putSessionState callers MUST NOT write state.json unless this matches the
  // current sessionId — writing while partially-loaded can wipe classes/objects.
  const sessionLoadedRef = useRef<string | null>(null);

  // Lost-update guard counter (#80). The session's monotonic state version as
  // last seen by this tab. Sent on every PUT /state; the backend 409s if it's
  // stale. Updated on successful GET and PUT.
  const stateVersionRef = useRef<number>(0);

  // Safety guard: refuse to persist a state payload that looks corrupt.
  // Corrupt = classes is empty AND we have one or more objects with class_id < 0
  // (i.e., synthetic orphans). This is the fingerprint of the wipe bug.
  const stateIsSafeToPersist = useCallback(
    (cls: ObjectClass[], objs: TrackedObject[]): boolean => {
      if (cls.length === 0 && objs.some((o) => o.class_id < 0)) return false;
      return true;
    },
    [],
  );

  const handleCloseSession = useCallback(() => {
    // Prevent duplicate close attempts
    if (closingSession) return;
    setConfirmDialog({
      title: "Close Session",
      description:
        "Close this session and return to the session list? Your annotations are saved automatically.",
      confirmLabel: "Close Session",
      onConfirm: async () => {
        setConfirmDialog(null);
        setClosingSession(true);
        try {
          // Cancel propagation first if running — backend close_session waits for
          // propagation_lock, so we must cancel to avoid blocking the UI.
          if (propagatingRef.current && sessionId) {
            propagateAbortRef.current?.abort();
            try {
              await cancelPropagation(sessionId);
            } catch {
              // best-effort cancel
            }
            setPropagating(false);
          }
          // Flush any pending state (bboxPadding, classes, objects) before clearing sessionId,
          // but ONLY if loadSession finished for this session and the payload is safe.
          // Otherwise we might overwrite disk with a partially-loaded / corrupt snapshot.
          if (
            sessionId &&
            sessionLoadedRef.current === sessionId &&
            stateIsSafeToPersist(classes, objects)
          ) {
            try {
              const newVersion = await putSessionState(sessionId, {
                classes,
                objects,
                bbox_padding: bboxPaddingRef.current,
                version: stateVersionRef.current,
              });
              stateVersionRef.current = newVersion;
            } catch {
              // best-effort flush (including StateVersionConflictError — user
              // is closing the session so we intentionally drop the local
              // in-flight write rather than overwrite the other tab's changes).
            }
          }
          if (sessionId) {
            // Release SAM3 inference state on the backend (resets ServiceState to idle).
            // If the backend reports unsynced files (503), let the user decide:
            // retry, force-close (local preserved via .unsynced marker), or cancel
            // and keep the session open.
            let closed = false;
            let attempt = 0;
            while (!closed) {
              try {
                await closeSession(sessionId);
                closed = true;
              } catch (err) {
                if (err instanceof CloseSessionUnsyncedError) {
                  attempt += 1;
                  const choice = await new Promise<"retry" | "force" | "cancel">(
                    (resolve) => {
                      setConfirmDialog({
                        title: "Couldn't save to cloud",
                        description:
                          `${err.unsyncedFiles.length} file(s) couldn't be uploaded to cloud storage ` +
                          `(attempt ${attempt}): ${err.unsyncedFiles.join(", ")}. ` +
                          `Your changes are preserved locally and will be re-uploaded the next ` +
                          `time you open this session.`,
                        confirmLabel: "Retry upload",
                        onConfirm: () => {
                          setConfirmDialog(null);
                          resolve("retry");
                        },
                        cancelLabel: "Close anyway",
                        onCancel: () => {
                          setConfirmDialog(null);
                          resolve("force");
                        },
                      });
                    },
                  );
                  if (choice === "retry") {
                    continue;
                  }
                  if (choice === "force") {
                    closed = true;
                    break;
                  }
                  // "cancel" — keep session open, do NOT continue with the
                  // rest of handleCloseSession (which clears sessionId etc.)
                  setClosingSession(false);
                  return;
                } else {
                  // Non-503 error — log and force-close to avoid trapping the user
                  console.error("closeSession failed:", err);
                  closed = true;
                  break;
                }
              }
            }
            clearSessionMemory(sessionId);
          }
          // Reset auto-load guard so a new session can be loaded
          hasLoadedSessionRef.current = null;
          sessionLoadedRef.current = null;
          setSessionId(null);
          setFrameCount(0);
          setCurrentFrame(0);
          setClasses([]);
          setObjects([]);
          setMasks({});
          setFrameObjIndex(new Map());
          setSelectedClassId(null);
          setSelectedObjIds(new Set());
          setClickPoints([]);
          setPrompts({});
          setBboxPadding({});
          setToolMode("pointer");
          setVideoName(null);
          setLastSaved(null);
          setConfidenceWarnings([]);
          setLowConfidenceFrames(new Set());
          setStatus("Ready");
          // Immediately reflect idle state from server
          refetchStatus();
        } finally {
          setClosingSession(false);
        }
      },
    });
  }, [sessionId, classes, objects, refetchStatus, stateIsSafeToPersist, closingSession]);

  // Force-refresh masks from the server, bypassing the IndexedDB cache.
  // Use this when the canvas state looks out of sync with what the backend
  // has on disk — or to validate that "missing" masks are just a client-side
  // staleness issue and not real data loss.
  const handleRefreshMasks = useCallback(async () => {
    if (!sessionId || refreshingMasks) return;
    setRefreshingMasks(true);
    setStatus("Refreshing masks from server...");
    try {
      // Drop the IDB cache entirely for this session so loadSessionMasksSmart
      // can't serve stale/partial records — force a full re-fetch.
      await evictSession(sessionId);
      const { masks: serverMasks, versions } = await loadSessionMasksSmart(sessionId);
      const index = new Map<number, Set<number>>();
      let totalObjects = 0;
      for (const [frameStr, objMasks] of Object.entries(serverMasks)) {
        const fi = Number(frameStr);
        const objIds = Object.keys(objMasks).map(Number).filter((id) => objMasks[id] != null);
        if (objIds.length > 0) {
          index.set(fi, new Set(objIds));
          totalObjects += objIds.length;
        }
      }
      setFrameObjIndex(index);
      setMasks(serverMasks);
      storeSessionMeta(sessionId, versions).catch(() => {});
      console.info(
        `[refresh] ${Object.keys(serverMasks).length} frames, ${totalObjects} mask entries from server`,
      );
      setStatus(`Refreshed: ${Object.keys(serverMasks).length} frames with masks`);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setStatus(`Refresh failed: ${msg}`);
      console.error("[refresh] failed", err);
    } finally {
      setRefreshingMasks(false);
    }
  }, [sessionId, refreshingMasks]);

  // Lazy-load masks for current frame when navigating beyond LRU window,
  // or when propagation eviction left incomplete data (missing objects).
  useEffect(() => {
    if (!sessionId) return;
    const hasIndex = frameObjIndex.has(currentFrame);
    if (!hasIndex) return;
    const frameData = masksRef.current[currentFrame];
    if (frameData != null) {
      // During propagation, SSE events are updating masks live — don't re-fetch
      if (propagating) return;
      // After propagation, check that all tracked objects are present.
      // Propagation eviction may have dropped masks for non-propagated objects
      // when a frame was evicted and later re-added with only the propagated object.
      const expectedObjs = frameObjIndex.get(currentFrame)!;
      if (Array.from(expectedObjs).every((id) => id in frameData)) return;
    }
    if (loadingFramesRef.current.has(currentFrame)) return;

    loadingFramesRef.current.add(currentFrame);
    const frame = currentFrame;
    const sid = sessionId;

    (async () => {
      try {
        let frameMasks = await getCachedMasks(sid, frame);
        if (!frameMasks) {
          frameMasks = await loadFrameMasks(sid, frame);
          putMasks(sid, frame, frameMasks).catch(() => {});
        }
        const gotCount = Object.keys(frameMasks).length;
        const expectedCount = frameObjIndex.get(frame)?.size ?? 0;
        if (gotCount < expectedCount) {
          console.warn(
            `[lazy-load] frame=${frame} expected ${expectedCount} objects, got ${gotCount} — server may be stale or data actually missing`,
          );
        }
        if (gotCount > 0) {
          setMasks((prev) => ({ ...prev, [frame]: frameMasks }));
        }
        // Empty or error: leave frameObjIndex intact. Transient empty reads
        // (writer race, flaky GCS) must not erase timeline ticks — only
        // explicit user delete actions remove them.
      } catch {
        // same as above — preserve the tick
      } finally {
        loadingFramesRef.current.delete(frame);
      }
    })();
  }, [sessionId, currentFrame, frameObjIndex, propagating]);

  // Ref mirrors bboxPadding — lets persistState always use the latest value
  // without needing bboxPadding in its dependency array (which would re-create
  // every callback that depends on persistState).
  const bboxPaddingRef = useRef(bboxPadding);
  bboxPaddingRef.current = bboxPadding;

  // Apply a server-authoritative state snapshot on 409 reconciliation.
  // Replaces the local classes/objects/bbox_padding and advances the version
  // anchor so the next PUT is consistent with the server.
  const reconcileFromServer = useCallback(
    (serverVersion: number, serverState: {
      classes: ObjectClass[];
      objects: TrackedObject[];
      bbox_padding?: Record<number, Record<number, BboxPadding>>;
    }) => {
      stateVersionRef.current = serverVersion;
      setClasses(serverState.classes || []);
      setObjects(serverState.objects || []);
      setBboxPadding(serverState.bbox_padding || {});
      setStatus("Another tab updated this session — reloaded from server.");
    },
    [],
  );

  const persistState = useCallback(
    async (cls: ObjectClass[], objs: TrackedObject[]) => {
      if (!sessionId) return;
      if (sessionLoadedRef.current !== sessionId) return;
      if (!stateIsSafeToPersist(cls, objs)) return;
      try {
        const newVersion = await putSessionState(sessionId, {
          classes: cls,
          objects: objs,
          bbox_padding: bboxPaddingRef.current,
          version: stateVersionRef.current,
        });
        stateVersionRef.current = newVersion;
        setLastSaved(new Date());
      } catch (err) {
        if (err instanceof StateVersionConflictError) {
          reconcileFromServer(err.currentVersion, err.currentState);
        }
        // otherwise silent
      }
    },
    [sessionId, stateIsSafeToPersist, reconcileFromServer],
  );

  // Auto-save when bboxPadding changes (slider edits don't call persistState directly).
  // Track the previous sessionId so we skip the initial fire when a new session loads —
  // at that point classes/objects are still [] and writing them would wipe state.json.
  const prevAutoSaveSessionRef = useRef<string | null>(null);

  useEffect(() => {
    if (!sessionId) return;
    if (prevAutoSaveSessionRef.current !== sessionId) {
      // sessionId just changed — skip this fire, let loadSession handle the initial state
      prevAutoSaveSessionRef.current = sessionId;
      return;
    }
    // Safety gate: never auto-save until loadSession has fully completed for
    // this sessionId. Firing earlier can capture classes=[] / objects=[] in
    // the closure and wipe state.json.
    if (sessionLoadedRef.current !== sessionId) return;
    // Safety gate: never persist a payload that looks corrupt (the wipe fingerprint).
    if (!stateIsSafeToPersist(classes, objects)) return;
    // Use a debounce so rapid slider drags don't flood the backend
    const timer = setTimeout(() => {
      putSessionState(sessionId, {
        classes,
        objects,
        bbox_padding: bboxPadding,
        version: stateVersionRef.current,
      })
        .then((newVersion) => {
          stateVersionRef.current = newVersion;
          setLastSaved(new Date());
        })
        .catch((err) => {
          if (err instanceof StateVersionConflictError) {
            reconcileFromServer(err.currentVersion, err.currentState);
          }
        });
    }, 500);
    return () => clearTimeout(timer);
  }, [sessionId, bboxPadding, stateIsSafeToPersist, reconcileFromServer]); // eslint-disable-line react-hooks/exhaustive-deps

  const selectClass = useCallback(
    (id: number | null) => {
      setSelectedClassId(id);
      setSelectedObjIds(new Set());
      setToolMode("pointer");
      // Unhide the class if it was hidden
      if (id != null) {
        setHiddenClassIds((prev) => {
          if (!prev.has(id)) return prev;
          const next = new Set(prev);
          next.delete(id);
          return next;
        });
      }
    },
    [],
  );

  const selectObj = useCallback(
    (id: number | null) => {
      setSelectedObjIds(id != null ? new Set([id]) : new Set());
      if (id != null) {
        const obj = objects.find((o) => o.obj_id === id);
        if (obj) {
          const cls = classes.find((c) => c.id === obj.class_id);
          if (cls) {
            setSelectedClassId(obj.class_id);
          } else {
            // Object exists but class was deleted — prompt for class assignment
            setOrphanObjId(id);
          }
        } else {
          // Object not in state (mask survived class deletion) — add as orphan
          const orphan: TrackedObject = { obj_id: id, class_id: -1 };
          setObjects((prev) => [...prev, orphan]);
          setOrphanObjId(id);
        }
      }
      setToolMode("pointer");
    },
    [objects, classes],
  );

  const toggleObjSelection = useCallback(
    (id: number) => {
      if (propagating) return;
      setSelectedObjIds((prev) => {
        const next = new Set(prev);
        if (next.has(id)) {
          next.delete(id);
        } else {
          next.add(id);
        }
        return next;
      });
      const obj = objects.find((o) => o.obj_id === id);
      if (obj) {
        setSelectedClassId(obj.class_id);
      }
    },
    [objects, propagating],
  );

  const getOrCreateObjId = useCallback(
    (objs: TrackedObject[], frameMasks: Record<number, MaskResult>): [number, TrackedObject[]] => {
      if (selectedObjId != null) {
        return [selectedObjId, objs];
      }

      if (selectedClassId == null) {
        return [-1, objs];
      }

      // If an object of the selected class already has a mask on this frame, reuse it
      const classObjs = objs.filter((o) => o.class_id === selectedClassId);
      const existing = classObjs.find((o) => frameMasks[o.obj_id] != null);
      if (existing) {
        return [existing.obj_id, objs];
      }

      const maxId = objs.reduce(
        (max, o) => Math.max(max, o.obj_id),
        0,
      );
      const newObjId = maxId + 1;
      const newObj: TrackedObject = {
        obj_id: newObjId,
        class_id: selectedClassId,
      };
      const updatedObjs = [...objs, newObj];
      return [newObjId, updatedObjs];
    },
    [selectedClassId, selectedObjId],
  );

  const executeClickPoint = useCallback(
    async (x: number, y: number, label: number, frameIdx: number, objId: number, updatedObjs: TrackedObject[]) => {
      if (!sessionId) return;

      if (updatedObjs !== objects) {
        setObjects(updatedObjs);
        setSelectedObjIds(new Set([objId]));
        persistState(classes, updatedObjs);
      } else {
        setSelectedObjIds(new Set([objId]));
      }

      const newPoint: ClickPoint = { x, y, label, frameIdx, objId };
      const existingForObj = clickPoints.filter(
        (p) => p.objId === objId && p.frameIdx === frameIdx,
      );
      const allPoints = [...existingForObj, newPoint];
      setClickPoints((prev) => [...prev, newPoint]);

      setSegmenting(true);
      setStatus("Segmenting...");
      try {
        const result = await clickSegment(
          sessionId,
          frameIdx,
          objId,
          allPoints.map((p) => [p.x, p.y]),
          allPoints.map((p) => p.label),
        );
        updateMasksFromResult(result);
        setPrompts((prev) => ({
          ...prev,
          [frameIdx]: {
            ...prev[frameIdx],
            [objId]: {
              type: "click",
              points: allPoints.map((p) => [p.x, p.y]),
              labels: allPoints.map((p) => p.label),
            },
          },
        }));
        setStatus("Ready");
      } catch (err) {
        // Roll back the optimistically-added click point
        setClickPoints((prev) => prev.filter((p) => p !== newPoint));
        setStatus(
          `Segment error: ${err instanceof Error ? err.message : String(err)}`,
        );
      } finally {
        setSegmenting(false);
      }
    },
    [sessionId, classes, objects, clickPoints, persistState],
  );

  const handleClickPoint = useCallback(
    async (x: number, y: number, label: number, frameIdx: number) => {
      if (!sessionId || segmenting) return;
      if (selectedClassId == null && selectedObjId == null) {
        setStatus("Select a class or object first");
        return;
      }

      // If an object is already selected, use it directly
      if (selectedObjId != null) {
        const frameMasks = masksRef.current[frameIdx] ?? {};
        const [objId, updatedObjs] = getOrCreateObjId(objects, frameMasks);
        if (objId < 0) return;
        executeClickPoint(x, y, label, frameIdx, objId, updatedObjs);
        return;
      }

      // No object selected — check if class has existing objects
      if (selectedClassId != null) {
        const classObjs = objects.filter((o) => o.class_id === selectedClassId);
        if (classObjs.length === 0) {
          // No objects — auto-create
          const frameMasks = masksRef.current[frameIdx] ?? {};
          const [objId, updatedObjs] = getOrCreateObjId(objects, frameMasks);
          if (objId < 0) return;
          executeClickPoint(x, y, label, frameIdx, objId, updatedObjs);
        } else {
          // Objects exist — ask user
          setObjectChoice({
            existingObjs: classObjs,
            pendingAction: { type: "click", x, y, label, frameIdx },
          });
        }
      }
    },
    [sessionId, segmenting, selectedClassId, selectedObjId, objects, getOrCreateObjId, executeClickPoint],
  );

  const handleUndoClick = useCallback(
    async () => {
      if (!sessionId || segmenting) return;
      const framePoints = clickPoints.filter(
        (p) => p.frameIdx === currentFrame && selectedObjId != null && p.objId === selectedObjId,
      );
      if (framePoints.length === 0) return;

      const lastPoint = framePoints[framePoints.length - 1];
      const remaining = clickPoints.filter((p) => p !== lastPoint);
      setClickPoints(remaining);

      const remainingForObj = remaining.filter(
        (p) => p.objId === lastPoint.objId && p.frameIdx === lastPoint.frameIdx,
      );

      if (remainingForObj.length > 0) {
        setSegmenting(true);
        setStatus("Segmenting...");
        try {
          const result = await clickSegment(
            sessionId,
            lastPoint.frameIdx,
            lastPoint.objId,
            remainingForObj.map((p) => [p.x, p.y]),
            remainingForObj.map((p) => p.label),
          );
          updateMasksFromResult(result);
          // Backend already updated the prompt via clickSegment; sync local state
          setPrompts((prev) => ({
            ...prev,
            [lastPoint.frameIdx]: {
              ...prev[lastPoint.frameIdx],
              [lastPoint.objId]: {
                type: "click",
                points: remainingForObj.map((p) => [p.x, p.y]),
                labels: remainingForObj.map((p) => p.label),
              },
            },
          }));
          setStatus("Ready");
        } catch (err) {
          setStatus(
            `Segment error: ${err instanceof Error ? err.message : String(err)}`,
          );
        } finally {
          setSegmenting(false);
        }
      } else {
        // No points left for this object on this frame — clear its mask and prompt
        setMasks((prev) => {
          const frameMasks = { ...(prev[lastPoint.frameIdx] ?? {}) };
          delete frameMasks[lastPoint.objId];
          return { ...prev, [lastPoint.frameIdx]: frameMasks };
        });
        setPrompts((prev) => {
          const framePrompts = { ...(prev[lastPoint.frameIdx] ?? {}) };
          delete framePrompts[lastPoint.objId];
          if (Object.keys(framePrompts).length === 0) {
            const next = { ...prev };
            delete next[lastPoint.frameIdx];
            return next;
          }
          return { ...prev, [lastPoint.frameIdx]: framePrompts };
        });
        setStatus("Ready");
      }
    },
    [sessionId, segmenting, clickPoints, currentFrame, selectedObjId],
  );

  // Ctrl+Z / Cmd+Z to undo last click point
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (
        (e.metaKey || e.ctrlKey) && e.key === "z" && !e.shiftKey &&
        !(e.target instanceof HTMLInputElement) &&
        !(e.target instanceof HTMLTextAreaElement)
      ) {
        e.preventDefault();
        handleUndoClick();
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [handleUndoClick]);

  const executeBoxDraw = useCallback(
    async (
      box: [number, number, number, number],
      frameIdx: number,
      objId: number,
      updatedObjs: TrackedObject[],
    ) => {
      if (!sessionId) return;

      if (updatedObjs !== objects) {
        setObjects(updatedObjs);
        setSelectedObjIds(new Set([objId]));
        persistState(classes, updatedObjs);
      } else {
        setSelectedObjIds(new Set([objId]));
      }

      setSegmenting(true);
      setStatus("Segmenting...");
      try {
        const result = await boxSegment(sessionId, frameIdx, objId, box);
        updateMasksFromResult(result);
        setPrompts((prev) => ({
          ...prev,
          [frameIdx]: { ...prev[frameIdx], [objId]: { type: "box", box } },
        }));
        setStatus("Ready");
      } catch (err) {
        setStatus(
          `Segment error: ${err instanceof Error ? err.message : String(err)}`,
        );
      } finally {
        setSegmenting(false);
      }
    },
    [sessionId, classes, objects, persistState],
  );

  const handleBoxDraw = useCallback(
    async (
      box: [number, number, number, number],
      frameIdx: number,
    ) => {
      if (!sessionId || segmenting) return;
      if (selectedClassId == null && selectedObjId == null) {
        setStatus("Select a class or object first");
        return;
      }

      // If an object is already selected, use it directly
      if (selectedObjId != null) {
        const frameMasks = masksRef.current[frameIdx] ?? {};
        const [objId, updatedObjs] = getOrCreateObjId(objects, frameMasks);
        if (objId < 0) return;
        executeBoxDraw(box, frameIdx, objId, updatedObjs);
        return;
      }

      // No object selected — check if class has existing objects
      if (selectedClassId != null) {
        const classObjs = objects.filter((o) => o.class_id === selectedClassId);
        if (classObjs.length === 0) {
          // No objects — auto-create
          const frameMasks = masksRef.current[frameIdx] ?? {};
          const [objId, updatedObjs] = getOrCreateObjId(objects, frameMasks);
          if (objId < 0) return;
          executeBoxDraw(box, frameIdx, objId, updatedObjs);
        } else {
          // Objects exist — ask user
          setObjectChoice({
            existingObjs: classObjs,
            pendingAction: { type: "box", box, frameIdx },
          });
        }
      }
    },
    [sessionId, segmenting, selectedClassId, selectedObjId, objects, getOrCreateObjId, executeBoxDraw],
  );

  const handleDetect = useCallback(
    async (text: string) => {
      if (!sessionId || textPromptLoading) return;

      const abortCtrl = new AbortController();
      setDetectAbortController(abortCtrl);
      setTextPromptLoading(true);
      setStatus(`Detecting "${text}"...`);
      try {
        // Auto-resolve class: use selected, match by name, or create new
        let classId = selectedClassId;
        if (classId == null) {
          const existing = classes.find(
            (c) => c.name.toLowerCase() === text.trim().toLowerCase(),
          );
          if (existing) {
            classId = existing.id;
            setSelectedClassId(existing.id);
          } else {
            const hue = Math.floor(Math.random() * 360);
            const color = `hsl(${hue}, 70%, 50%)`;
            const cls = await createClass(sessionId, text, color);
            setClasses((prev) => [...prev, cls]);
            setSelectedClassId(cls.id);
            classId = cls.id;
          }
        }

        const maxId = objects.reduce((max, o) => Math.max(max, o.obj_id), 0);
        const objIdStart = maxId + 1;
        const result = await textSegment(sessionId, currentFrame, text, objIdStart, abortCtrl.signal);
        if (result.instances.length === 0) {
          setStatus(`No instances of "${text}" found on this frame`);
          return;
        }

        let updatedObjs = [...objects];
        const frameMasks: Record<number, MaskResult> = {};
        const detectedIds: number[] = [];

        for (const inst of result.instances) {
          updatedObjs = [
            ...updatedObjs,
            { obj_id: inst.obj_id, class_id: classId },
          ];
          frameMasks[inst.obj_id] = {
            rle: inst.rle,
            bbox: inst.bbox,
            area: inst.area,
            source_keyframe: null,
            confidence: inst.confidence,
          };
          detectedIds.push(inst.obj_id);
        }

        setObjects(updatedObjs);
        setMasks((prev) => {
          const existing = prev[currentFrame] ?? {};
          return { ...prev, [currentFrame]: { ...existing, ...frameMasks } };
        });
        setFrameObjIndex((prev) => {
          const existing = prev.get(currentFrame) ?? new Set();
          const merged = new Set(existing);
          for (const id of detectedIds) merged.add(id);
          const next = new Map(prev);
          next.set(currentFrame, merged);
          return next;
        });
        // Multi-select all detected instances so user can propagate them all at once
        setSelectedObjIds(new Set(detectedIds));

        persistState(classes, updatedObjs);
        setStatus(
          `Found ${result.instances.length} "${text}" instance(s) — all selected, ready to propagate`,
        );
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") {
          setStatus("Detection cancelled");
        } else {
          setStatus(
            `Detect error: ${err instanceof Error ? err.message : String(err)}`,
          );
        }
      } finally {
        setTextPromptLoading(false);
        setDetectAbortController(null);
      }
    },
    [sessionId, textPromptLoading, selectedClassId, currentFrame, objects, classes, persistState],
  );

  function updateMasksFromResult(result: FrameResult) {
    setMasks((prev) => {
      const frameMasks = { ...(prev[result.frame_idx] ?? {}) };
      // Merge top-level source_keyframe into each mask entry — the backend
      // SSE stream and click/box endpoints put it on the result, not per-mask.
      const sourceKf = result.source_keyframe !== undefined
        ? result.source_keyframe
        : null;
      for (const [objIdStr, maskData] of Object.entries(result.masks)) {
        frameMasks[Number(objIdStr)] = {
          ...maskData,
          source_keyframe: maskData.source_keyframe !== undefined
            ? maskData.source_keyframe
            : sourceKf,
        };
      }
      return { ...prev, [result.frame_idx]: frameMasks };
    });
    const resultObjIds = Object.keys(result.masks).map(Number);
    setFrameObjIndex((prev) => {
      const existing = prev.get(result.frame_idx);
      const merged = new Set(existing);
      for (const id of resultObjIds) merged.add(id);
      if (existing && existing.size === merged.size) return prev;
      const next = new Map(prev);
      next.set(result.frame_idx, merged);
      return next;
    });
    // Write-through to mask cache
    if (sessionId) {
      const frameMasksForCache: Record<number, MaskResult> = {};
      for (const [objIdStr, maskData] of Object.entries(result.masks)) {
        const sourceKf = maskData.source_keyframe !== undefined
          ? maskData.source_keyframe
          : (result.source_keyframe !== undefined ? result.source_keyframe : null);
        frameMasksForCache[Number(objIdStr)] = { ...maskData, source_keyframe: sourceKf };
      }
      if (propagating) {
        putMasksMemoryOnly(sessionId, result.frame_idx, frameMasksForCache);
      } else {
        putMasks(sessionId, result.frame_idx, frameMasksForCache).catch(() => {});
      }
    }
  }

  const handlePropagate = useCallback(
    async (direction: "forward" | "backward" | "both") => {
      if (!sessionId) return;
      const abortController = new AbortController();
      propagateAbortRef.current = abortController;
      setPropagating(true);
      setPropagationProgress(0);
      setToolMode("pointer");
      setStatus("Propagating...");

      // Clear previous warnings
      setConfidenceWarnings([]);
      setLowConfidenceFrames(new Set());

      let framesProcessed = 0;
      const totalFrames =
        direction === "both" ? frameCount * 2 : frameCount;

      const collectedWarnings: ConfidenceWarning[] = [];

      const onFrame = (result: FrameResult) => {
        framesProcessed++;
        setPropagationProgress(framesProcessed / totalFrames);
        updateMasksFromResult(result);
        setCurrentFrame(result.frame_idx);

        // Check each mask's confidence against threshold
        for (const [objIdStr, maskData] of Object.entries(result.masks)) {
          if (maskData.confidence != null && maskData.confidence < CONFIDENCE_THRESHOLD) {
            collectedWarnings.push({
              frame_idx: result.frame_idx,
              obj_id: Number(objIdStr),
              confidence: maskData.confidence,
            });
          }
        }
        // Update warnings state live during propagation
        if (collectedWarnings.length > 0) {
          setConfidenceWarnings([...collectedWarnings]);
        }
      };

      const objectIds = selectedObjIds.size > 0 ? [...selectedObjIds] : undefined;
      try {
        if (direction === "forward" || direction === "both") {
          await propagate(sessionId, currentFrame, false, onFrame, abortController.signal, objectIds);
        }
        if (direction === "backward" || direction === "both") {
          await propagate(sessionId, currentFrame, true, onFrame, abortController.signal, objectIds);
        }
        if (collectedWarnings.length > 0) {
          setStatus(`Propagation complete — ${new Set(collectedWarnings.map((w) => w.frame_idx)).size} frame(s) with low confidence`);
        } else {
          setStatus("Propagation complete");
        }
      } catch (err) {
        if (abortController.signal.aborted) {
          setStatus("Propagation cancelled");
        } else {
          const msg = err instanceof Error ? err.message : String(err);
          const tb = (err as Error & { traceback?: string })?.traceback;
          setStatus(`Propagation error: ${msg}`);
          setErrorDialog({
            title: "Propagation failed",
            message: msg,
            traceback: tb,
          });
        }
      } finally {
        propagateAbortRef.current = null;
        setPropagating(false);
        setPropagationProgress(0);
        // Always sync timeline markers to whatever warnings were collected
        // (handles cancel/abort case where the try block didn't complete)
        setLowConfidenceFrames(new Set(collectedWarnings.map((w) => w.frame_idx)));
        if (sessionId) {
          flushToIDB(sessionId).catch(() => {});
          fetchMaskVersions(sessionId)
            .then((v) => storeSessionMeta(sessionId, v))
            .catch(() => {});
        }
      }
    },
    [sessionId, currentFrame, frameCount, selectedObjIds],
  );

  const handleCancelPropagate = useCallback(() => {
    propagateAbortRef.current?.abort();
    if (sessionId) {
      cancelPropagation(sessionId).catch(() => {});
    }
  }, [sessionId]);

  const handleGoToFrame = useCallback((frame: number) => {
    setCurrentFrame(frame);
  }, []);

  const handleObjectChoice = useCallback(
    (choice: number | "new") => {
      if (!objectChoice) return;
      const { pendingAction, existingObjs } = objectChoice;
      setObjectChoice(null);

      if (choice === "new") {
        // Create a new object for this class
        const maxId = objects.reduce((max, o) => Math.max(max, o.obj_id), 0);
        const newObjId = maxId + 1;
        const classId = existingObjs[0].class_id;
        const newObj: TrackedObject = { obj_id: newObjId, class_id: classId };
        const updatedObjs = [...objects, newObj];

        if (pendingAction.type === "click") {
          executeClickPoint(pendingAction.x, pendingAction.y, pendingAction.label, pendingAction.frameIdx, newObjId, updatedObjs);
        } else {
          executeBoxDraw(pendingAction.box, pendingAction.frameIdx, newObjId, updatedObjs);
        }
      } else {
        // Use existing object
        const objId = choice;
        if (pendingAction.type === "click") {
          executeClickPoint(pendingAction.x, pendingAction.y, pendingAction.label, pendingAction.frameIdx, objId, objects);
        } else {
          executeBoxDraw(pendingAction.box, pendingAction.frameIdx, objId, objects);
        }
      }
    },
    [objectChoice, objects, executeClickPoint, executeBoxDraw],
  );

  const handleClassCreated = useCallback(
    (cls: ObjectClass) => {
      const updated = [...classes, cls];
      setClasses(updated);
      setSelectedClassId(cls.id);
      setSelectedObjIds(new Set());
      setToolMode("pointer");
      persistState(updated, objects);
    },
    [classes, objects, persistState],
  );

  const handleClassDeleted = useCallback(
    async (classId: number) => {
      const affectedObjects = objects.filter((o) => o.class_id === classId);

      // Remove each object via API (cleans up masks, prompts, SAM3 state on backend)
      if (sessionId) {
        for (const obj of affectedObjects) {
          try {
            await removeObject(sessionId, obj.obj_id);
          } catch {
            // continue cleanup
          }
        }
      }

      // Clean up frontend state for all affected objects
      const affectedIds = new Set(affectedObjects.map((o) => o.obj_id));

      setMasks((prev) => {
        const updated = { ...prev };
        for (const frameIdx of Object.keys(updated)) {
          const fi = Number(frameIdx);
          const frameMasks = { ...updated[fi] };
          for (const objId of affectedIds) {
            delete frameMasks[objId];
          }
          if (Object.keys(frameMasks).length === 0) {
            delete updated[fi];
          } else {
            updated[fi] = frameMasks;
          }
        }
        return updated;
      });

      setBboxPadding((prev) => {
        const updated = { ...prev };
        for (const objId of affectedIds) {
          delete updated[objId];
        }
        return updated;
      });

      setPrompts((prev) => {
        const next: Record<number, Record<number, KeyframePrompt>> = {};
        for (const [frameStr, framePrompts] of Object.entries(prev)) {
          const updated = { ...framePrompts };
          for (const objId of affectedIds) {
            delete updated[objId];
          }
          if (Object.keys(updated).length > 0) {
            next[Number(frameStr)] = updated;
          }
        }
        return next;
      });

      setFrameObjIndex((prev) => {
        const next = new Map<number, Set<number>>();
        for (const [fi, objIds] of prev) {
          const remaining = new Set(objIds);
          for (const objId of affectedIds) {
            remaining.delete(objId);
          }
          if (remaining.size > 0) next.set(fi, remaining);
        }
        return next;
      });

      const updatedClasses = classes.filter((c) => c.id !== classId);
      const updatedObjects = objects.filter((o) => o.class_id !== classId);
      setClasses(updatedClasses);
      setObjects(updatedObjects);
      if (selectedClassId === classId) setSelectedClassId(null);
      setSelectedObjIds((prev) => {
        const next = new Set(prev);
        for (const objId of affectedIds) {
          next.delete(objId);
        }
        return next;
      });
      persistState(updatedClasses, updatedObjects);
    },
    [sessionId, classes, objects, selectedClassId, persistState],
  );

  const handleClassRenamed = useCallback(
    (classId: number, newName: string) => {
      const updated = classes.map((c) =>
        c.id === classId ? { ...c, name: newName } : c,
      );
      setClasses(updated);
      persistState(updated, objects);
    },
    [classes, objects, persistState],
  );

  const handleClassColorChanged = useCallback(
    (classId: number, color: string) => {
      const updated = classes.map((c) =>
        c.id === classId ? { ...c, color } : c,
      );
      setClasses(updated);
      persistState(updated, objects);
    },
    [classes, objects, persistState],
  );

  const handleToggleClassVisibility = useCallback((classId: number) => {
    setHiddenClassIds((prev) => {
      const next = new Set(prev);
      if (next.has(classId)) next.delete(classId);
      else {
        next.add(classId);
        // Deselect class and object if they belong to the class being hidden
        if (selectedClassId === classId) setSelectedClassId(null);
        setSelectedObjIds((prev) => {
          const next = new Set(prev);
          for (const id of prev) {
            const obj = objects.find((o) => o.obj_id === id);
            if (obj?.class_id === classId) next.delete(id);
          }
          return next;
        });
      }
      return next;
    });
  }, [objects, selectedClassId]);

  const handleObjectDeleted = useCallback(
    async (objId: number) => {
      if (!sessionId) return;
      try {
        await removeObject(sessionId, objId);
      } catch {
        // continue
      }
      const updatedObjects = objects.filter((o) => o.obj_id !== objId);
      setObjects(updatedObjects);
      setSelectedObjIds((prev) => {
        const next = new Set(prev);
        next.delete(objId);
        return next;
      });

      setMasks((prev) => {
        const updated = { ...prev };
        for (const frameIdx of Object.keys(updated)) {
          const fi = Number(frameIdx);
          const frameMasks = { ...updated[fi] };
          delete frameMasks[objId];
          updated[fi] = frameMasks;
        }
        return updated;
      });

      setBboxPadding((prev) => {
        const updated = { ...prev };
        delete updated[objId];
        return updated;
      });

      // Remove all prompts for this object
      setPrompts((prev) => {
        const next: Record<number, Record<number, KeyframePrompt>> = {};
        for (const [frameStr, framePrompts] of Object.entries(prev)) {
          const updated = { ...framePrompts };
          delete updated[objId];
          if (Object.keys(updated).length > 0) {
            next[Number(frameStr)] = updated;
          }
        }
        return next;
      });

      setFrameObjIndex((prev) => {
        const next = new Map<number, Set<number>>();
        for (const [fi, objIds] of prev) {
          const remaining = new Set(objIds);
          remaining.delete(objId);
          if (remaining.size > 0) next.set(fi, remaining);
        }
        return next;
      });

      persistState(classes, updatedObjects);
    },
    [sessionId, objects, classes, persistState],
  );

  const handleObjectReassigned = useCallback(
    (objId: number, newClassId: number) => {
      const updatedObjects = objects.map((o) =>
        o.obj_id === objId ? { ...o, class_id: newClassId } : o,
      );
      setObjects(updatedObjects);
      persistState(classes, updatedObjects);
    },
    [objects, classes, persistState],
  );

  const handleOrphanChooseClass = useCallback(
    (classId: number) => {
      if (orphanObjId == null) return;
      const updatedObjects = objects.map((o) =>
        o.obj_id === orphanObjId ? { ...o, class_id: classId } : o,
      );
      setObjects(updatedObjects);
      setSelectedClassId(classId);
      setOrphanObjId(null);
      persistState(classes, updatedObjects);
    },
    [orphanObjId, objects, classes, persistState],
  );

  const handleOrphanCreateClass = useCallback(
    async (name: string) => {
      if (orphanObjId == null || !sessionId) return;
      try {
        const color = PALETTE[classes.length % PALETTE.length];
        const cls = await createClass(sessionId, name, color);
        const updatedClasses = [...classes, cls];
        const updatedObjects = objects.map((o) =>
          o.obj_id === orphanObjId ? { ...o, class_id: cls.id } : o,
        );
        setClasses(updatedClasses);
        setObjects(updatedObjects);
        setSelectedClassId(cls.id);
        setOrphanObjId(null);
        persistState(updatedClasses, updatedObjects);
      } catch (err) {
        console.error("Failed to create class:", err);
      }
    },
    [orphanObjId, sessionId, classes, objects, persistState],
  );

  // Frames where the selected object has masks (for timeline filtering)
  const selectedObjFrames = useMemo(() => {
    if (selectedObjId == null) return new Set<number>();
    const frames = new Set<number>();
    for (const [fi, objIds] of frameObjIndex) {
      if (objIds.has(selectedObjId)) frames.add(fi);
    }
    return frames;
  }, [selectedObjId, frameObjIndex]);

  const handleBatchDelete = useCallback(
    async (frameIndices: number[]) => {
      if (!sessionId || selectedObjId == null || frameIndices.length === 0) return;
      try {
        const { deleted_frames } = await deleteFrameMasksBatch(
          sessionId,
          selectedObjId,
          frameIndices,
        );
        // Update local masks state
        setMasks((prev) => {
          const next = { ...prev };
          for (const fi of deleted_frames) {
            if (next[fi]) {
              const frame = { ...next[fi] };
              delete frame[selectedObjId];
              if (Object.keys(frame).length === 0) {
                delete next[fi];
              } else {
                next[fi] = frame;
              }
            }
          }
          return next;
        });
        // Update frameObjIndex
        setFrameObjIndex((prev) => {
          const next = new Map(prev);
          for (const fi of deleted_frames) {
            const objs = next.get(fi);
            if (objs) {
              const updated = new Set(objs);
              updated.delete(selectedObjId);
              if (updated.size === 0) {
                next.delete(fi);
              } else {
                next.set(fi, updated);
              }
            }
          }
          return next;
        });
        // Update prompts
        setPrompts((prev) => {
          const next = { ...prev };
          for (const fi of deleted_frames) {
            if (next[fi] && selectedObjId in next[fi]) {
              const frame = { ...next[fi] };
              delete frame[selectedObjId];
              if (Object.keys(frame).length === 0) {
                delete next[fi];
              } else {
                next[fi] = frame;
              }
            }
          }
          return next;
        });
        // Update click points
        setClickPoints((prev) =>
          prev.filter(
            (p) => !(p.objId === selectedObjId && deleted_frames.includes(p.frameIdx)),
          ),
        );
        setStatus(`Deleted ${deleted_frames.length} mask${deleted_frames.length !== 1 ? "s" : ""}`);
        setShowDeletePanel(false);
      } catch (err) {
        setStatus(`Delete failed: ${err instanceof Error ? err.message : String(err)}`);
      }
    },
    [sessionId, selectedObjId],
  );

  // Delete key to open delete masks panel
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (
        (e.key === "Delete" || e.key === "Backspace") &&
        !(e.target instanceof HTMLInputElement) &&
        !(e.target instanceof HTMLTextAreaElement) &&
        !e.metaKey && !e.ctrlKey &&
        selectedObjId != null
      ) {
        e.preventDefault();
        setShowDeletePanel(true);
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedObjId]);

  // Escape key clears multi-select
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key === "Escape" && selectedObjIds.size >= 2 && !propagating) {
        setSelectedObjIds(new Set());
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [selectedObjIds, propagating]);

  // Auto-close delete panel when context changes
  useEffect(() => {
    setShowDeletePanel(false);
  }, [selectedObjId, sessionId]);

  // "Display frame" — the last frame that has complete mask data.
  // The UI renders this frame to avoid flicker when navigating to frames
  // whose masks haven't loaded yet. currentFrame still advances immediately
  // so fetching starts, but the visible UI only updates once data arrives.
  const masksLoading = frameObjIndex.has(currentFrame) && masks[currentFrame] == null;
  const displayFrameRef = useRef(currentFrame);
  if (!masksLoading) displayFrameRef.current = currentFrame;
  const displayFrame = displayFrameRef.current;

  const currentMasks = masks[displayFrame] ?? {};

  const currentFramePadding = useMemo(() => {
    const result: Record<number, BboxPadding> = {};
    const frameMasks = masks[displayFrame] ?? {};
    for (const objIdStr of Object.keys(frameMasks)) {
      const objId = Number(objIdStr);
      const sourceKf = frameMasks[objId]?.source_keyframe;
      result[objId] = getPaddingForKeyframe(bboxPadding, objId, sourceKf, displayFrame);
    }
    return result;
  }, [bboxPadding, displayFrame, masks]);

  const handlePlayToggle = useCallback(() => {
    setIsPlaying((prev) => !prev);
  }, []);

  // Sorted array of frames with masks for the selected object (cached for skip buttons)
  const sortedObjFrames = useMemo(() => {
    return [...selectedObjFrames].sort((a, b) => a - b);
  }, [selectedObjFrames]);

  const handlePrevMask = useCallback(() => {
    const prev = sortedObjFrames.filter((f) => f < currentFrame).pop();
    if (prev != null) setCurrentFrame(prev);
  }, [sortedObjFrames, currentFrame]);

  const handleNextMask = useCallback(() => {
    const next = sortedObjFrames.find((f) => f > currentFrame);
    if (next != null) setCurrentFrame(next);
  }, [sortedObjFrames, currentFrame]);

  const handleRecalculate = useCallback(async () => {
    if (!sessionId || selectedObjId == null || segmenting) return;
    const prompt = prompts[currentFrame]?.[selectedObjId];
    if (!prompt) return;
    setSegmenting(true);
    setStatus("Recalculating...");
    try {
      const result = await recalculatePrompt(sessionId, currentFrame, selectedObjId);
      updateMasksFromResult(result);
      setStatus("Recalculated");
    } catch (err) {
      setStatus(`Recalculate error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setSegmenting(false);
    }
  }, [sessionId, selectedObjId, currentFrame, prompts, segmenting]);

  const handleErasePrompt = useCallback(async () => {
    if (!sessionId || selectedObjId == null) return;
    try {
      await deleteFrameMask(sessionId, currentFrame, selectedObjId);
      // Clear local mask
      setMasks((prev) => {
        const frameMasks = { ...(prev[currentFrame] ?? {}) };
        delete frameMasks[selectedObjId];
        return { ...prev, [currentFrame]: frameMasks };
      });
      // Clear local prompt
      setPrompts((prev) => {
        const framePrompts = { ...(prev[currentFrame] ?? {}) };
        delete framePrompts[selectedObjId];
        if (Object.keys(framePrompts).length === 0) {
          const next = { ...prev };
          delete next[currentFrame];
          return next;
        }
        return { ...prev, [currentFrame]: framePrompts };
      });
      // Clear local click points for this frame/object
      setClickPoints((prev) =>
        prev.filter((p) => !(p.frameIdx === currentFrame && p.objId === selectedObjId))
      );
      // Update frame-object index
      setFrameObjIndex((prev) => {
        const next = new Map(prev);
        const objIds = next.get(currentFrame);
        if (objIds) {
          const updated = new Set(objIds);
          updated.delete(selectedObjId);
          if (updated.size === 0) next.delete(currentFrame);
          else next.set(currentFrame, updated);
        }
        return next;
      });
      setStatus("Prompt erased");
    } catch (err) {
      setStatus(`Erase error: ${err instanceof Error ? err.message : String(err)}`);
    }
  }, [sessionId, selectedObjId, currentFrame]);

  // Frames where user placed prompts (clicks/boxes) — true keyframes
  // Only show keyframes for the selected object; empty when none selected
  const keyframes = useMemo(() => {
    if (selectedObjId == null) return new Set<number>();
    const frames = new Set<number>();
    // From persisted prompts for the selected object
    for (const [frameStr, objPrompts] of Object.entries(prompts)) {
      if (selectedObjId in objPrompts) {
        frames.add(Number(frameStr));
      }
    }
    // From in-flight clickPoints for the selected object
    for (const p of clickPoints) {
      if (p.objId === selectedObjId) {
        frames.add(p.frameIdx);
      }
    }
    return frames;
  }, [prompts, clickPoints, selectedObjId]);

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-background">
      {/* Status toast */}
      <StatusToast message={status} />

      {/* Body */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Sidebar — only shown when a session is active */}
        {sessionId && (
          <aside className="w-64 shrink-0 animate-slide-in-left border-r border-border bg-card">
            <ScrollArea className="h-full">
              <div className="flex flex-col gap-4 py-4 pl-3 pr-3">
                {/* Session info card */}
                <div className="min-w-0 rounded-lg border border-border bg-accent/30 p-3">
                  <div className="flex items-start gap-2">
                    <Video className="mt-0.5 h-4 w-4 shrink-0 text-muted-foreground" />
                    <span
                      className="min-w-0 flex-1 text-sm font-medium text-foreground break-words [overflow-wrap:anywhere]"
                      title={videoName ?? "Untitled"}
                    >
                      {videoName ?? "Untitled"}
                    </span>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="h-6 w-6 shrink-0"
                      onClick={handleRefreshMasks}
                      disabled={refreshingMasks || closingSession}
                      title="Re-download all masks from the server"
                    >
                      {refreshingMasks ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <RefreshCw className="h-3.5 w-3.5" />}
                    </Button>
                    <Button variant="ghost" size="icon" className="h-6 w-6 shrink-0" onClick={handleCloseSession} disabled={closingSession}>
                      {closingSession ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <X className="h-3.5 w-3.5" />}
                    </Button>
                  </div>
                  <div className="mt-1.5 space-y-0.5 pl-6">
                    <p className="text-[11px] text-muted-foreground">
                      {frameCount} frames &middot; {classes.length} class{classes.length !== 1 ? "es" : ""}
                    </p>
                    <p className="text-[11px] text-muted-foreground">
                      {objects.length} object{objects.length !== 1 ? "s" : ""} &middot; {lastSaved ? `Saved ${timeAgo(lastSaved)}` : "Not saved yet"}
                    </p>
                  </div>
                  <div className="mt-3 space-y-2">
                    <ExportDialog
                      sessionId={sessionId!}
                      videoName={videoName ?? "Untitled"}
                      open={showExport}
                      onOpen={() => setShowExport(true)}
                      onClose={() => setShowExport(false)}
                      onStatus={setStatus}
                      bboxPadding={bboxPadding}
                    />
                    <ExportSessionDialog
                      sessionId={sessionId!}
                      videoName={videoName ?? "Untitled"}
                      open={showSessionExport}
                      onOpen={() => setShowSessionExport(true)}
                      onClose={() => setShowSessionExport(false)}
                      onStatus={setStatus}
                    />
                  </div>
                </div>
                <Separator />
                <AnnotationPanel
                  classes={classes}
                  objects={objects}
                  selectedClassId={selectedClassId}
                  selectedObjId={selectedObjId}
                  sessionId={sessionId}
                  hiddenClassIds={hiddenClassIds}
                  frameObjIndex={frameObjIndex}
                  frameCount={frameCount}
                  confidenceWarnings={confidenceWarnings}
                  onClassCreated={handleClassCreated}
                  onClassDeleted={handleClassDeleted}
                  onClassSelected={selectClass}
                  onClassRenamed={handleClassRenamed}
                  onClassColorChanged={handleClassColorChanged}
                  onToggleVisibility={handleToggleClassVisibility}
                  onObjectSelected={selectObj}
                  onObjectDeleted={handleObjectDeleted}
                  onObjectReassigned={handleObjectReassigned}
                  selectedObjIds={selectedObjIds}
                  onToggleObjSelection={toggleObjSelection}
                />
              </div>
            </ScrollArea>
          </aside>
        )}

        {/* Main Content */}
        <main className="flex flex-1 flex-col overflow-hidden bg-background">
          {sessionId ? (
            <>
              {/* Frame caching overlay — blocks canvas until all frames are cached */}
              {cachingFrames && (
                <div className="flex flex-1 items-center justify-center">
                  <div className="flex flex-col items-center gap-5">
                    <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
                    <p className="text-lg font-semibold text-foreground">
                      Caching frames...
                    </p>
                    <Progress value={cachingProgress.total > 0 ? (cachingProgress.cached / cachingProgress.total) * 100 : 0} className="h-1.5 w-full max-w-sm" />
                    <p className="text-sm text-muted-foreground">
                      {cachingProgress.cached} / {cachingProgress.total}
                    </p>
                  </div>
                </div>
              )}
              {/* Canvas area — fills available space, no scroll */}
              <div className={`relative flex flex-1 flex-col items-center overflow-hidden${cachingFrames ? " hidden" : ""}`}>
                {/* Selected object label — glass overlay */}
                {selectedObjIds.size === 1 && (() => {
                  const obj = objects.find((o) => o.obj_id === selectedObjId);
                  const cls = obj ? classes.find((c) => c.id === obj.class_id) : null;
                  if (!cls) return null;
                  return (
                    <div
                      className="absolute bottom-2 left-2 z-10 flex items-center overflow-hidden rounded-lg border border-white/20 px-2 py-0.5 shadow-sm backdrop-blur-xl"
                      style={{ backgroundColor: cls.color + "40" }}
                    >
                      <span className="text-[11px] font-medium text-black dark:text-white">
                        {cls.name} #{selectedObjId}
                      </span>
                    </div>
                  );
                })()}
                {selectedObjIds.size >= 2 && (
                  <div className="absolute bottom-2 left-2 z-10 flex items-center overflow-hidden rounded-lg border border-white/20 bg-black/30 px-2 py-0.5 shadow-sm backdrop-blur-xl">
                    <span className="text-[11px] font-medium text-black dark:text-white">
                      {selectedObjIds.size} objects selected
                    </span>
                  </div>
                )}

                <div className="w-full min-h-0 flex-1">
                  <VideoCanvas
                    sessionId={sessionId}
                    frameIdx={displayFrame}
                    masks={currentMasks}
                    classes={classes}
                    objects={objects}
                    toolMode={toolMode}
                    selectedObjId={selectedObjId}
                    hiddenClassIds={hiddenClassIds}
                    clickPoints={(() => {
                      if (selectedObjId == null) return [];
                      const live = clickPoints.filter(
                        (p) => p.frameIdx === displayFrame && p.objId === selectedObjId
                      );
                      if (live.length > 0) return live;
                      // Derive from saved prompt so dots appear even when mask is missing
                      const prompt = prompts[displayFrame]?.[selectedObjId];
                      if (prompt?.type === "click" && prompt.points && prompt.labels) {
                        return prompt.points.map((pt, i) => ({
                          x: pt[0], y: pt[1], label: prompt.labels![i],
                          frameIdx: displayFrame, objId: selectedObjId,
                        }));
                      }
                      return [];
                    })()}
                    bboxPadding={currentFramePadding}
                    segmenting={segmenting}
                    onClickPoint={handleClickPoint}
                    onBoxDraw={handleBoxDraw}
                    onMaskClick={selectObj}
                    onDeleteMasks={selectedObjId != null ? () => setShowDeletePanel(true) : undefined}
                    selectedObjIds={selectedObjIds}
                    onShiftMaskClick={toggleObjSelection}
                    onCancelMultiSelect={() => setSelectedObjIds(new Set())}
                  />
                </div>
              </div>
              {/* Pinned frame navigator — hidden during frame caching */}
              <div className={`shrink-0 border-t border-border bg-card px-4 py-2${cachingFrames ? " hidden" : ""}`}>
                <FrameNavigator
                  currentFrame={currentFrame}
                  frameCount={frameCount}
                  keyframes={keyframes}
                  selectedObjFrames={selectedObjFrames}
                  selectedObjColor={
                    selectedObjId != null
                      ? classes.find((c) => c.id === objects.find((o) => o.obj_id === selectedObjId)?.class_id)?.color
                      : undefined
                  }
                  lowConfidenceFrames={lowConfidenceFrames}
                  isPlaying={isPlaying}
                  playbackFps={playbackFps}
                  onChange={setCurrentFrame}
                  onPlayToggle={handlePlayToggle}
                  onPlaybackFpsChange={setPlaybackFps}
                  onPrevMask={selectedObjId != null && selectedObjFrames.size > 0 ? handlePrevMask : undefined}
                  onNextMask={selectedObjId != null && selectedObjFrames.size > 0 ? handleNextMask : undefined}
                  sessionId={sessionId ?? undefined}
                  masks={masks}
                  selectedObjId={selectedObjId}
                  bboxPadding={bboxPadding}
                />
              </div>
            </>
          ) : (
            <div className="relative flex h-full items-center justify-center">
              <div className="absolute right-4 top-4 z-10">
                <ThemeToggle />
              </div>
              {serviceStatus === null ? (
                /* Loading — waiting for first /api/status response */
                <div className="flex items-center justify-center">
                  <Loader2 className="h-6 w-6 animate-spin text-zinc-500" />
                </div>
              ) : serviceStatus.phase === "error" ? (
                <PipelineProgress
                  status={serviceStatus}
                  onDismissError={async () => {
                    try {
                      await dismissPipelineError();
                    } catch {
                      // Even if the call fails, refetching will reveal the current truth
                    }
                    refetchStatus();
                  }}
                />
              ) : serviceStatus.phase === "extracting" || serviceStatus.phase === "initializing" ? (
                /* Pipeline in progress */
                <PipelineProgress status={serviceStatus} />
              ) : cachingFrames ? (
                <div className="flex flex-col items-center gap-5">
                  <p className="text-lg font-semibold text-foreground">
                    Caching frames...
                  </p>
                  <Progress value={cachingProgress.total > 0 ? (cachingProgress.cached / cachingProgress.total) * 100 : 0} className="h-1.5 w-full max-w-sm" />
                  <p className="text-sm text-muted-foreground">
                    {cachingProgress.cached} / {cachingProgress.total}
                  </p>
                </div>
              ) : (
                /* Idle — show session list / upload */
                <VideoUpload
                  onStatus={setStatus}
                  onPipelineStarted={refetchStatus}
                />
              )}
            </div>
          )}
        </main>

        {/* Right Sidebar — only shown when a session is active */}
        {sessionId && (
          <aside className="w-64 shrink-0 animate-slide-in-right border-l border-border bg-card">
          {showDeletePanel && selectedObjId != null ? (
            <DeleteMasksPanel
              sessionId={sessionId}
              selectedObjId={selectedObjId}
              objects={objects}
              classes={classes}
              masks={masks}
              frameObjIndex={frameObjIndex}
              prompts={prompts}
              bboxPadding={bboxPadding}
              initialFrame={currentFrame}
              currentFrame={currentFrame}
              onDelete={handleBatchDelete}
              onClose={() => setShowDeletePanel(false)}
            />
          ) : (
          <ScrollArea className="h-full">
            <div className="flex flex-col gap-4 p-4">
                  <ToolBar
                    toolMode={toolMode}
                    onToolModeChange={setToolMode}
                    onUndo={handleUndoClick}
                    canUndo={clickPoints.some((p) => p.frameIdx === currentFrame && selectedObjId != null && p.objId === selectedObjId)}
                    onDeleteMasks={() => setShowDeletePanel(true)}
                    canDeleteMasks={selectedObjId != null && !propagating}
                    onDetect={handleDetect}
                    onCancelDetect={detectAbortController ? () => detectAbortController.abort() : undefined}
                    detectEnabled={!!sessionId && !propagating}
                    detectLoading={textPromptLoading}
                    selectedClassName={classes.find((c) => c.id === selectedClassId)?.name}
                    propagating={propagating}
                    multiSelect={selectedObjIds.size >= 2}
                  />
                  {selectedObjIds.size > 0 && (
                    <>
                      <Separator />
                      <PropagationBar
                        sessionId={sessionId}
                        currentFrame={currentFrame}
                        frameCount={frameCount}
                        propagating={propagating}
                        progress={propagationProgress}
                        onPropagate={handlePropagate}
                        onCancel={handleCancelPropagate}
                        confidenceWarnings={confidenceWarnings}
                        onGoToFrame={handleGoToFrame}
                        selectedObjIds={selectedObjIds}
                        objects={objects}
                        classes={classes}
                        onRemoveFromSelection={(objId: number) => {
                          setSelectedObjIds((prev) => {
                            const next = new Set(prev);
                            next.delete(objId);
                            return next;
                          });
                        }}
                      />
                    </>
                  )}
                  {selectedObjId != null && currentMasks[selectedObjId] && (
                    <>
                      {prompts[currentFrame]?.[selectedObjId] && (
                        <>
                          <Separator />
                          <div className="flex flex-col gap-2">
                            <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                              Keyframe
                            </h3>
                            <p className="text-[10px] text-muted-foreground">
                              {prompts[currentFrame][selectedObjId].type === "click"
                                ? `${prompts[currentFrame][selectedObjId].points?.length ?? 0} click(s)`
                                : "Bounding box"}
                            </p>
                            <Button
                              variant="outline"
                              size="sm"
                              className="h-7 text-xs"
                              disabled={segmenting || propagating}
                              onClick={handleRecalculate}
                            >
                              Recalculate Keyframe Mask
                            </Button>
                          </div>
                        </>
                      )}
                      <Separator />
                      <div className="flex flex-col gap-3">
                        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                          Bbox Padding
                        </h3>
                        <p className="text-[10px] text-muted-foreground">
                          Object {selectedObjId} — {classes.find((c) => c.id === objects.find((o) => o.obj_id === selectedObjId)?.class_id)?.name ?? "unknown"}
                        </p>
                        {(() => {
                          const sourceKf = currentMasks[selectedObjId]?.source_keyframe;
                          if (sourceKf == null) {
                            return (
                              <p className="text-[10px] text-muted-foreground">
                                Frame {currentFrame} (keyframe)
                              </p>
                            );
                          }
                          return (
                            <p className="text-[10px] text-muted-foreground">
                              Frame {currentFrame}, from keyframe {sourceKf}{" "}
                              <button
                                className="text-primary underline hover:text-primary/80"
                                onClick={() => setCurrentFrame(sourceKf)}
                              >
                                Jump to keyframe
                              </button>
                            </p>
                          );
                        })()}
                        {/* sliders — spatial layout */}
                        {(() => {
                          const padSlider = (side: "top" | "bottom" | "left" | "right") => {
                            const val = Math.round(currentFramePadding[selectedObjId]?.[side] ?? 0);
                            return (
                              <div className="flex flex-col gap-1">
                                <Label className="text-[10px] text-muted-foreground capitalize">
                                  {side}: {val}%
                                </Label>
                                <Slider
                                  value={[val]}
                                  min={-5}
                                  max={150}
                                  step={1}
                                  onValueChange={([v]) => {
                                    const sourceKf = currentMasks[selectedObjId]?.source_keyframe ?? currentFrame;
                                    setBboxPadding((prev) => ({
                                      ...prev,
                                      [selectedObjId]: {
                                        ...prev[selectedObjId],
                                        [sourceKf]: {
                                          ...{ top: 0, bottom: 0, left: 0, right: 0 },
                                          ...(prev[selectedObjId]?.[sourceKf] ??
                                            currentFramePadding[selectedObjId]),
                                          [side]: v,
                                        },
                                      },
                                    }));
                                  }}
                                />
                              </div>
                            );
                          };
                          return (
                            <div className="flex flex-col gap-2">
                              {padSlider("top")}
                              <div className="grid grid-cols-2 gap-3">
                                {padSlider("left")}
                                {padSlider("right")}
                              </div>
                              {padSlider("bottom")}
                            </div>
                          );
                        })()}
                        {(() => {
                          const sourceKf = currentMasks[selectedObjId]?.source_keyframe ?? currentFrame;
                          return bboxPadding[selectedObjId]?.[sourceKf] ? (
                            <Button
                              variant="ghost"
                              size="sm"
                              className="h-7 text-[10px] text-muted-foreground"
                              onClick={() =>
                                setBboxPadding((prev) => {
                                  const objKf = { ...prev[selectedObjId] };
                                  delete objKf[sourceKf];
                                  if (Object.keys(objKf).length === 0) {
                                    const next = { ...prev };
                                    delete next[selectedObjId];
                                    return next;
                                  }
                                  return { ...prev, [selectedObjId]: objKf };
                                })
                              }
                            >
                              Remove padding for keyframe {sourceKf}
                            </Button>
                          ) : null;
                        })()}
                      </div>
                    </>
                  )}
                  {/* Show recalculate when prompt exists but mask is missing */}
                  {selectedObjId != null && !currentMasks[selectedObjId] && prompts[currentFrame]?.[selectedObjId] && (
                    <>
                      <Separator />
                      <div className="flex flex-col gap-2">
                        <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                          Missing Mask
                        </h3>
                        <p className="text-[10px] text-muted-foreground">
                          Object {selectedObjId} has a saved {prompts[currentFrame][selectedObjId].type === "click"
                            ? `prompt (${prompts[currentFrame][selectedObjId].points?.length ?? 0} click${(prompts[currentFrame][selectedObjId].points?.length ?? 0) !== 1 ? "s" : ""})`
                            : "prompt (bounding box)"} on this frame but no computed mask.
                        </p>
                        <div className="flex gap-2">
                          <Button
                            variant="outline"
                            size="sm"
                            className="h-7 flex-1 text-xs"
                            disabled={segmenting || propagating}
                            onClick={handleRecalculate}
                          >
                            Calculate Mask
                          </Button>
                          <Button
                            variant="outline"
                            size="sm"
                            className="h-7 flex-1 text-xs text-destructive hover:text-destructive"
                            disabled={segmenting || propagating}
                            onClick={handleErasePrompt}
                          >
                            Erase Prompt
                          </Button>
                        </div>
                      </div>
                    </>
                  )}
            </div>
          </ScrollArea>
          )}
        </aside>
        )}
      </div>

      <ConfirmDialog
        open={confirmDialog != null}
        title={confirmDialog?.title ?? ""}
        description={confirmDialog?.description ?? ""}
        confirmLabel={confirmDialog?.confirmLabel}
        cancelLabel={confirmDialog?.cancelLabel}
        onConfirm={() => confirmDialog?.onConfirm()}
        onCancel={() => {
          const custom = confirmDialog?.onCancel;
          if (custom) {
            custom();
          } else {
            setConfirmDialog(null);
          }
        }}
      />

      <ErrorDialog
        open={errorDialog != null}
        title={errorDialog?.title ?? ""}
        message={errorDialog?.message ?? ""}
        traceback={errorDialog?.traceback}
        onDismiss={() => setErrorDialog(null)}
      />

      <ObjectChoiceDialog
        open={objectChoice != null}
        className={
          objectChoice
            ? classes.find((c) => c.id === objectChoice.existingObjs[0]?.class_id)?.name ?? "Unknown"
            : ""
        }
        existingObjects={objectChoice?.existingObjs ?? []}
        onChoose={handleObjectChoice}
        onCancel={() => setObjectChoice(null)}
      />

      <ClassChoiceDialog
        open={orphanObjId != null}
        objId={orphanObjId ?? 0}
        classes={classes}
        onChooseClass={handleOrphanChooseClass}
        onCreateClass={handleOrphanCreateClass}
        onCancel={() => {
          setOrphanObjId(null);
          setSelectedObjIds(new Set());
        }}
      />
    </div>
  );
}

export default App;
