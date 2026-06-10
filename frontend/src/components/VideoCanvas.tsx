import { useRef, useEffect, useCallback, useState } from "react";
import { getFrame } from "../frameCache.ts";
import type { MaskResult, ObjectClass, TrackedObject, ClickPoint, BboxPadding } from "../types.ts";
import { Crop, Loader2, Trash2, X } from "lucide-react";

interface Props {
  sessionId: string | null;
  frameIdx: number;
  masks: Record<number, MaskResult>;
  classes: ObjectClass[];
  objects: TrackedObject[];
  toolMode: "pointer" | "click" | "box";
  selectedObjId: number | null;
  selectedObjIds: Set<number>;
  clickPoints: ClickPoint[];
  bboxPadding: Record<number, BboxPadding>;
  segmenting: boolean;
  onClickPoint: (
    x: number,
    y: number,
    label: number,
    frameIdx: number,
  ) => void;
  onBoxDraw: (
    box: [number, number, number, number],
    frameIdx: number,
  ) => void;
  onMaskClick?: (objId: number) => void;
  onShiftMaskClick?: (objId: number) => void;
  onCancelMultiSelect?: () => void;
  onDeleteMasks?: () => void;
  hiddenClassIds?: Set<number>;
}

// LRU cache for decoded RLE counts — avoids re-parsing on every overlay render
const RLE_CACHE_MAX = 64;
const rleCache = new Map<string, Int32Array>();

function decodeRleCounts(encoded: string): Int32Array {
  const cached = rleCache.get(encoded);
  if (cached) return cached;

  const counts: number[] = [];
  let p = 0;
  while (p < encoded.length) {
    let x = 0;
    let k = 0;
    let more = true;
    while (more) {
      const c = encoded.charCodeAt(p) - 48;
      p++;
      x |= (c & 0x1f) << (5 * k);
      more = !!(c & 0x20);
      k++;
      if (!more && c & 0x10) x |= -1 << (5 * k);
    }
    if (counts.length > 2) x += counts[counts.length - 2];
    counts.push(x);
  }

  const result = Int32Array.from(counts);

  // Evict oldest entries if cache is full
  if (rleCache.size >= RLE_CACHE_MAX) {
    const firstKey = rleCache.keys().next().value!;
    rleCache.delete(firstKey);
  }
  rleCache.set(encoded, result);

  return result;
}

/** Check if image-space (x, y) falls inside a foreground RLE run.
 *  RLE is column-major: linearIdx = x * rleH + y. */
function hitTestMask(
  x: number,
  y: number,
  rle: { counts: string; size: number[] },
): boolean {
  const ix = Math.round(x);
  const iy = Math.round(y);
  const [rleH] = rle.size;
  const linearIdx = ix * rleH + iy;
  const counts = decodeRleCounts(rle.counts);

  let cumulative = 0;
  for (let i = 0; i < counts.length; i++) {
    cumulative += counts[i];
    if (cumulative > linearIdx) {
      // Odd-indexed runs are foreground
      return i % 2 === 1;
    }
  }
  return false;
}

function drawMaskFromRle(
  imageData: ImageData,
  rle: { counts: string; size: number[] },
  color: string,
  imgWidth: number,
  imgHeight: number,
  alpha: number = 100,
) {
  const counts = decodeRleCounts(rle.counts);
  const [rleH] = rle.size;
  const r = parseInt(color.slice(1, 3), 16);
  const g = parseInt(color.slice(3, 5), 16);
  const b = parseInt(color.slice(5, 7), 16);

  let pixelIdx = 0;
  for (let i = 0; i < counts.length; i++) {
    const runLen = counts[i];
    if (i % 2 === 0) {
      // Background run — skip without per-pixel loop
      pixelIdx += runLen;
      continue;
    }
    for (let j = 0; j < runLen; j++) {
      const col = Math.floor(pixelIdx / rleH);
      const row = pixelIdx % rleH;
      if (row < imgHeight && col < imgWidth) {
        const offset = (row * imgWidth + col) * 4;
        imageData.data[offset] = r;
        imageData.data[offset + 1] = g;
        imageData.data[offset + 2] = b;
        imageData.data[offset + 3] = alpha;
      }
      pixelIdx++;
    }
  }
}

export default function VideoCanvas({
  sessionId,
  frameIdx,
  masks,
  classes,
  objects,
  toolMode,
  selectedObjId,
  selectedObjIds,
  clickPoints,
  bboxPadding,
  segmenting,
  onClickPoint,
  onBoxDraw,
  onMaskClick,
  onShiftMaskClick,
  onCancelMultiSelect,
  onDeleteMasks,
  hiddenClassIds,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);         // single visible canvas
  const offscreenRef = useRef<HTMLCanvasElement | null>(null); // offscreen buffer
  const maskCanvasRef = useRef<HTMLCanvasElement | null>(null); // temp canvas for mask compositing
  const frameImageRef = useRef<HTMLImageElement | null>(null); // current frame image
  const frameImageIdxRef = useRef<number>(-1);               // which frameIdx the loaded image is for
  const [frameImageVersion, setFrameImageVersion] = useState(0); // triggers repaint when frame loads
  const containerRef = useRef<HTMLDivElement>(null);
  const imageDataRef = useRef<ImageData | null>(null);
  const [naturalSize, setNaturalSize] = useState({ width: 800, height: 600 });
  const [containerSize, setContainerSize] = useState({ width: 800, height: 600 });
  const [boxStart, setBoxStart] = useState<{ x: number; y: number } | null>(null);
  const [boxCurrent, setBoxCurrent] = useState<{ x: number; y: number } | null>(null);
  const [contextMenu, setContextMenu] = useState<{ x: number; y: number } | null>(null);

  // Helpers to lazily create/resize the offscreen and mask canvases
  function getOffscreen(width: number, height: number): HTMLCanvasElement {
    if (!offscreenRef.current) {
      offscreenRef.current = document.createElement("canvas");
    }
    const c = offscreenRef.current;
    if (c.width !== width || c.height !== height) {
      c.width = width;
      c.height = height;
    }
    return c;
  }

  function getMaskCanvas(width: number, height: number): HTMLCanvasElement {
    if (!maskCanvasRef.current) {
      maskCanvasRef.current = document.createElement("canvas");
    }
    const c = maskCanvasRef.current;
    if (c.width !== width || c.height !== height) {
      c.width = width;
      c.height = height;
    }
    return c;
  }

  // Zoom & Pan state — refs hold the source of truth for synchronous
  // reads in the wheel handler; state drives rendering.
  const [zoom, setZoom] = useState(1);
  const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
  const zoomRef = useRef(1);
  const panOffsetRef = useRef({ x: 0, y: 0 });
  const isPanningRef = useRef(false);
  const panStartRef = useRef({ x: 0, y: 0 });
  const panOffsetStartRef = useRef({ x: 0, y: 0 });

  // Inspect mode — zoom-to-mask toggle
  const [inspecting, setInspecting] = useState(false);
  const inspectingRef = useRef(false);
  const savedZoomRef = useRef<{ zoom: number; pan: { x: number; y: number } } | null>(null);

  // fitScale: how much to scale the natural-size canvas to fit the container
  const fitScale = Math.min(
    containerSize.width / naturalSize.width,
    containerSize.height / naturalSize.height,
    1, // never upscale beyond natural size
  );

  // Track container size with ResizeObserver
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        setContainerSize({
          width: entry.contentRect.width,
          height: entry.contentRect.height,
        });
      }
    });
    ro.observe(container);
    return () => ro.disconnect();
  }, []);

  // Keep inspectingRef in sync so the wheel handler (which has [] deps) can read it
  useEffect(() => { inspectingRef.current = inspecting; }, [inspecting]);

  // Load frame image from IndexedDB cache — stores result in refs and triggers repaint
  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;

    getFrame(sessionId, frameIdx).then((blobUrl) => {
      if (cancelled) return;
      const img = new Image();
      img.onload = () => {
        if (cancelled) return;
        setNaturalSize({ width: img.naturalWidth, height: img.naturalHeight });
        frameImageRef.current = img;
        frameImageIdxRef.current = frameIdx;
        setFrameImageVersion((v) => v + 1);
      };
      img.src = blobUrl;
      // Note: do NOT revoke blobUrl here — it's managed by blobUrlCache
      // in frameCache.ts and will be revoked when the session is evicted.
    });

    return () => { cancelled = true; };
  }, [sessionId, frameIdx]);

  // Unified paint: composite frame + masks + overlays offscreen, then atomic blit
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    // Don't repaint until we have a frame image for the current frameIdx
    if (!frameImageRef.current || frameImageIdxRef.current !== frameIdx) return;

    const { width, height } = naturalSize;
    const offscreen = getOffscreen(width, height);
    const ctx = offscreen.getContext("2d")!;

    // 1. Draw frame image
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(frameImageRef.current, 0, 0);

    // 2. Draw masks
    const objIds = Object.keys(masks).map(Number);
    if (objIds.length > 0) {
      // Reuse ImageData to avoid allocating ~8MB per render
      if (
        !imageDataRef.current ||
        imageDataRef.current.width !== width ||
        imageDataRef.current.height !== height
      ) {
        imageDataRef.current = ctx.createImageData(width, height);
      }
      const imageData = imageDataRef.current;
      imageData.data.fill(0);

      for (const objId of objIds) {
        const maskResult = masks[objId];
        if (!maskResult?.rle) continue;
        const obj = objects.find((o) => o.obj_id === objId);
        const objClassId = obj?.class_id;
        if (objClassId != null && hiddenClassIds?.has(objClassId)) continue;
        const cls = obj ? classes.find((c) => c.id === obj.class_id) : null;
        const color = cls?.color ?? "#ff0000";
        const hasSelection = selectedObjIds.size > 0;
        const dimmed = hasSelection && !selectedObjIds.has(objId);
        drawMaskFromRle(imageData, maskResult.rle, color, width, height, dimmed ? 40 : 100);
      }

      // Put mask ImageData onto a temp canvas, then drawImage to alpha-blend
      // onto the offscreen buffer (putImageData would replace pixels, not blend)
      const mc = getMaskCanvas(width, height);
      const maskCtx = mc.getContext("2d", { willReadFrequently: true })!;
      maskCtx.clearRect(0, 0, width, height);
      maskCtx.putImageData(imageData, 0, 0);
      ctx.drawImage(mc, 0, 0);

      // 3. Draw bboxes and labels
      for (const objId of objIds) {
        const maskResult = masks[objId];
        if (!maskResult?.bbox) continue;
        const obj = objects.find((o) => o.obj_id === objId);
        const objClassId = obj?.class_id;
        if (objClassId != null && hiddenClassIds?.has(objClassId)) continue;
        const cls = obj ? classes.find((c) => c.id === obj.class_id) : null;
        const color = cls?.color ?? "#ff0000";
        const [bx, by, bw, bh] = maskResult.bbox;

        // Apply per-object, per-side bbox padding
        const objPad = bboxPadding[objId] ?? { top: 0, bottom: 0, left: 0, right: 0 };
        const padTop = bh * (objPad.top / 100);
        const padBottom = bh * (objPad.bottom / 100);
        const padLeft = bw * (objPad.left / 100);
        const padRight = bw * (objPad.right / 100);
        const drawX = Math.max(0, bx - padLeft);
        const drawY = Math.max(0, by - padTop);
        const drawW = Math.min(width - drawX, bw + padLeft + padRight);
        const drawH = Math.min(height - drawY, bh + padTop + padBottom);

        const isSelected = selectedObjIds.has(objId);
        const dimmed = selectedObjIds.size > 0 && !isSelected;
        if (dimmed) ctx.globalAlpha = 0.35;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        if (dimmed) {
          ctx.setLineDash([6, 4]);
        }
        ctx.strokeRect(drawX, drawY, drawW, drawH);
        ctx.setLineDash([]);

        const label = cls?.name ?? `obj ${objId}`;
        ctx.fillStyle = color;
        ctx.font = "12px 'Noto Sans', sans-serif";
        const textWidth = ctx.measureText(label).width;
        ctx.fillRect(drawX, drawY - 16, textWidth + 6, 16);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(label, drawX + 3, drawY - 4);
        if (dimmed) ctx.globalAlpha = 1.0;
      }
    }

    // 4. Draw click point markers — scale inversely with zoom so they
    // stay a constant screen size regardless of zoom level.
    const effectiveScale = fitScale * zoom;
    const markerRadius = Math.max(3, 8 / effectiveScale);
    const markerFontSize = Math.max(6, Math.round(12 / effectiveScale));
    const markerLineWidth = Math.max(0.5, 1.5 / effectiveScale);

    for (const pt of clickPoints) {
      const isPositive = pt.label === 1;
      const cx = pt.x;
      const cy = pt.y;

      ctx.beginPath();
      ctx.arc(cx, cy, markerRadius, 0, Math.PI * 2);
      ctx.fillStyle = isPositive ? "rgba(34, 197, 94, 0.9)" : "rgba(239, 68, 68, 0.9)";
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = markerLineWidth;
      ctx.stroke();

      ctx.fillStyle = "#ffffff";
      ctx.font = `bold ${markerFontSize}px 'Noto Sans', sans-serif`;
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(isPositive ? "+" : "\u2212", cx, cy);
      ctx.textAlign = "start";
      ctx.textBaseline = "alphabetic";
    }

    // 5. Box tool preview
    if (toolMode === "box" && boxStart && boxCurrent) {
      ctx.strokeStyle = "#6366f1";
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 5]);
      ctx.strokeRect(
        boxStart.x,
        boxStart.y,
        boxCurrent.x - boxStart.x,
        boxCurrent.y - boxStart.y,
      );
      ctx.setLineDash([]);
    }

    // 6. Atomic paint: copy offscreen → visible canvas
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const visibleCtx = canvas.getContext("2d")!;
    visibleCtx.drawImage(offscreen, 0, 0);

  }, [frameIdx, frameImageVersion, masks, naturalSize, classes, objects, toolMode, selectedObjIds, boxStart, boxCurrent, clickPoints, bboxPadding, zoom, fitScale, hiddenClassIds]);

  // Wheel handler — pinch-to-zoom (ctrlKey) vs two-finger scroll (pan)
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    function handleWheel(e: WheelEvent) {
      e.preventDefault();

      // Manual zoom/pan exits inspect mode
      if (inspectingRef.current) {
        savedZoomRef.current = null;
        setInspecting(false);
      }

      // Pinch-to-zoom on trackpad (ctrlKey) or mouse scroll wheel (no deltaX)
      const isZoom = e.ctrlKey || (e.deltaX === 0 && e.deltaMode === 0 && !e.shiftKey);

      if (isZoom) {
        const rect = container!.getBoundingClientRect();
        const cx = e.clientX - rect.left;
        const cy = e.clientY - rect.top;

        const prevZoom = zoomRef.current;
        const prevPan = panOffsetRef.current;
        // ctrlKey pinch uses larger deltas, so scale down more
        const sensitivity = e.ctrlKey ? 0.005 : 0.001;
        const delta = -e.deltaY * sensitivity;
        const newZoom = Math.min(10, Math.max(0.5, prevZoom + delta * prevZoom));
        const scale = newZoom / prevZoom;

        // Keep the image point under the cursor stationary
        const newPan = {
          x: cx - scale * (cx - prevPan.x),
          y: cy - scale * (cy - prevPan.y),
        };

        zoomRef.current = newZoom;
        panOffsetRef.current = newPan;
        setZoom(newZoom);
        setPanOffset(newPan);
      } else {
        // Two-finger trackpad scroll → pan
        const prevPan = panOffsetRef.current;
        const newPan = {
          x: prevPan.x - e.deltaX,
          y: prevPan.y - e.deltaY,
        };
        panOffsetRef.current = newPan;
        setPanOffset(newPan);
      }
    }

    container.addEventListener("wheel", handleWheel, { passive: false });
    return () => container.removeEventListener("wheel", handleWheel);
  }, []);

  // Compute centered pan offset for the current container/image size
  const centeredOffset = useCallback(() => {
    return {
      x: (containerSize.width - naturalSize.width * fitScale) / 2,
      y: (containerSize.height - naturalSize.height * fitScale) / 2,
    };
  }, [containerSize, naturalSize, fitScale]);

  // Center the image when session changes
  useEffect(() => {
    setInspecting(false);
    savedZoomRef.current = null;
    const offset = centeredOffset();
    zoomRef.current = 1;
    panOffsetRef.current = offset;
    setZoom(1);
    setPanOffset(offset);
  }, [sessionId]);

  // Re-center when the image natural size or container size changes (at zoom=1)
  useEffect(() => {
    if (zoomRef.current !== 1) return;
    const offset = centeredOffset();
    panOffsetRef.current = offset;
    setPanOffset(offset);
  }, [centeredOffset]);

  // Convert screen coordinates to image coordinates
  const canvasToImage = useCallback(
    (clientX: number, clientY: number) => {
      const container = containerRef.current;
      if (!container) return null;
      const rect = container.getBoundingClientRect();
      // Undo pan, then undo (fitScale * zoom) to get image coordinates
      const totalScale = fitScale * zoom;
      const x = (clientX - rect.left - panOffset.x) / totalScale;
      const y = (clientY - rect.top - panOffset.y) / totalScale;
      return { x, y };
    },
    [fitScale, zoom, panOffset],
  );

  function handleMouseDown(e: React.MouseEvent<HTMLDivElement>) {
    // Middle-click or Alt+left-click -> pan
    if (e.button === 1 || (e.button === 0 && e.altKey)) {
      if (inspecting) {
        savedZoomRef.current = null;
        setInspecting(false);
      }
      isPanningRef.current = true;
      panStartRef.current = { x: e.clientX, y: e.clientY };
      panOffsetStartRef.current = { ...panOffset };
      e.preventDefault();
      return;
    }

    if (toolMode === "box" && e.button === 0) {
      const pt = canvasToImage(e.clientX, e.clientY);
      if (pt) {
        setBoxStart(pt);
        setBoxCurrent(pt);
      }
      e.preventDefault();
    }
  }

  function handleMouseMove(e: React.MouseEvent<HTMLDivElement>) {
    if (isPanningRef.current) {
      const dx = e.clientX - panStartRef.current.x;
      const dy = e.clientY - panStartRef.current.y;
      const newPan = {
        x: panOffsetStartRef.current.x + dx,
        y: panOffsetStartRef.current.y + dy,
      };
      panOffsetRef.current = newPan;
      setPanOffset(newPan);
      return;
    }

    if (toolMode === "box" && boxStart) {
      const pt = canvasToImage(e.clientX, e.clientY);
      if (pt) setBoxCurrent(pt);
    }
  }

  function handleMouseUp(e: React.MouseEvent<HTMLDivElement>) {
    if (isPanningRef.current) {
      isPanningRef.current = false;
      return;
    }

    if (toolMode === "box" && boxStart) {
      const pt = canvasToImage(e.clientX, e.clientY);
      if (pt) {
        const x1 = Math.min(boxStart.x, pt.x);
        const y1 = Math.min(boxStart.y, pt.y);
        const x2 = Math.max(boxStart.x, pt.x);
        const y2 = Math.max(boxStart.y, pt.y);
        if (x2 - x1 > 5 && y2 - y1 > 5) {
          onBoxDraw([x1, y1, x2, y2], frameIdx);
        }
      }
      setBoxStart(null);
      setBoxCurrent(null);
    }
  }

  function handleClick(e: React.MouseEvent<HTMLDivElement>) {
    if (isPanningRef.current) return;
    if (e.altKey) return;

    if (toolMode === "pointer") {
      const pt = canvasToImage(e.clientX, e.clientY);
      if (!pt) return;
      // Hit-test masks in reverse order (topmost first)
      const objIds = Object.keys(masks).map(Number);
      for (let i = objIds.length - 1; i >= 0; i--) {
        const objId = objIds[i];
        const maskResult = masks[objId];
        if (!maskResult?.rle) continue;
        const hitObj = objects.find((o) => o.obj_id === objId);
        const hitClassId = hitObj?.class_id;
        if (hitClassId != null && hiddenClassIds?.has(hitClassId)) continue;
        if (hitTestMask(pt.x, pt.y, maskResult.rle)) {
          if (e.shiftKey && onShiftMaskClick) {
            onShiftMaskClick(objId);
          } else if (onMaskClick) {
            onMaskClick(objId);
          }
          return;
        }
      }
      return;
    }

    if (toolMode !== "click") return;
    const pt = canvasToImage(e.clientX, e.clientY);
    if (!pt) return;
    onClickPoint(pt.x, pt.y, 1, frameIdx);
  }

  function handleContextMenu(e: React.MouseEvent<HTMLDivElement>) {
    e.preventDefault();
    if (toolMode === "pointer" && onDeleteMasks && onMaskClick) {
      const pt = canvasToImage(e.clientX, e.clientY);
      if (!pt) return;
      // Hit-test masks to find object under cursor
      const objIds = Object.keys(masks).map(Number);
      for (let i = objIds.length - 1; i >= 0; i--) {
        const objId = objIds[i];
        const maskResult = masks[objId];
        if (!maskResult?.rle) continue;
        const hitObj = objects.find((o) => o.obj_id === objId);
        const hitClassId = hitObj?.class_id;
        if (hitClassId != null && hiddenClassIds?.has(hitClassId)) continue;
        if (hitTestMask(pt.x, pt.y, maskResult.rle)) {
          onMaskClick(objId);
          const rect = containerRef.current?.getBoundingClientRect();
          if (rect) {
            setContextMenu({ x: e.clientX - rect.left, y: e.clientY - rect.top });
          }
          return;
        }
      }
      return;
    }
    if (toolMode !== "click") return;
    const pt = canvasToImage(e.clientX, e.clientY);
    if (!pt) return;
    onClickPoint(pt.x, pt.y, 0, frameIdx);
  }

  function resetZoom() {
    if (inspecting) {
      savedZoomRef.current = null;
      setInspecting(false);
    }
    const offset = centeredOffset();
    zoomRef.current = 1;
    panOffsetRef.current = offset;
    setZoom(1);
    setPanOffset(offset);
  }

  function zoomToNative() {
    if (inspecting) {
      savedZoomRef.current = null;
      setInspecting(false);
    }
    const nativeZoom = 1 / fitScale;
    const offset = {
      x: (containerSize.width - naturalSize.width) / 2,
      y: (containerSize.height - naturalSize.height) / 2,
    };
    zoomRef.current = nativeZoom;
    panOffsetRef.current = offset;
    setZoom(nativeZoom);
    setPanOffset(offset);
  }

  function zoomToMask() {
    if (selectedObjId == null) return;
    const mask = masks[selectedObjId];
    if (!mask?.bbox) return;

    const [bx, by, bw, bh] = mask.bbox;
    const objPad = bboxPadding[selectedObjId] ?? { top: 0, bottom: 0, left: 0, right: 0 };
    const padTop = bh * (objPad.top / 100);
    const padBottom = bh * (objPad.bottom / 100);
    const padLeft = bw * (objPad.left / 100);
    const padRight = bw * (objPad.right / 100);

    // Padded bbox in image coords, clamped to image bounds
    const px = Math.max(0, bx - padLeft);
    const py = Math.max(0, by - padTop);
    const px2 = Math.min(naturalSize.width, bx + bw + padRight);
    const py2 = Math.min(naturalSize.height, by + bh + padBottom);
    const pw = px2 - px;
    const ph = py2 - py;
    if (pw <= 0 || ph <= 0) return;

    // Compute zoom: how much to multiply fitScale so the padded bbox fills the container
    // Cap at 300% total scale (3.0 / fitScale) so small objects don't over-zoom
    const newZoom = Math.min(
      containerSize.width / (pw * fitScale),
      containerSize.height / (ph * fitScale),
      3 / fitScale,
    );

    // Compute pan: center the padded bbox in the container
    const centerX = (px + pw / 2) * fitScale * newZoom;
    const centerY = (py + ph / 2) * fitScale * newZoom;
    const newPan = {
      x: containerSize.width / 2 - centerX,
      y: containerSize.height / 2 - centerY,
    };

    zoomRef.current = newZoom;
    panOffsetRef.current = newPan;
    setZoom(newZoom);
    setPanOffset(newPan);
  }

  function toggleInspect() {
    if (inspecting) {
      // Restore saved zoom/pan
      const saved = savedZoomRef.current;
      if (saved) {
        zoomRef.current = saved.zoom;
        panOffsetRef.current = saved.pan;
        setZoom(saved.zoom);
        setPanOffset(saved.pan);
      }
      savedZoomRef.current = null;
      setInspecting(false);
    } else {
      if (selectedObjId == null || !masks[selectedObjId]?.bbox) return;
      // Save current zoom/pan before inspecting
      savedZoomRef.current = { zoom: zoomRef.current, pan: { ...panOffsetRef.current } };
      setInspecting(true);
      zoomToMask();
    }
  }

  // Auto-follow: re-center on mask when frame, padding, or selection changes while inspecting
  useEffect(() => {
    if (!inspecting) return;
    if (selectedObjId == null || !masks[selectedObjId]?.bbox) return;
    zoomToMask();
  }, [inspecting, frameIdx, selectedObjId, masks, bboxPadding, containerSize]);

  // Exit inspect when object is deselected or mask disappears
  useEffect(() => {
    if (!inspecting) return;
    if (selectedObjId == null || !masks[selectedObjId]?.bbox) {
      savedZoomRef.current = null;
      setInspecting(false);
    }
  }, [inspecting, selectedObjId, masks]);

  const selectedColor =
    selectedObjId != null
      ? (() => {
          const obj = objects.find((o) => o.obj_id === selectedObjId);
          const cls = obj ? classes.find((c) => c.id === obj.class_id) : null;
          return cls?.color ?? null;
        })()
      : null;

  // The total CSS scale applied to the canvases
  const totalScale = fitScale * zoom;

  return (
    <div
      className="relative h-full w-full overflow-hidden"
      ref={containerRef}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onClick={handleClick}
      onContextMenu={handleContextMenu}
      style={{
        cursor: segmenting
          ? "wait"
          : toolMode === "click" || toolMode === "box"
            ? "crosshair"
            : "default",
      }}
    >
      <div
        style={{
          transform: `translate(${panOffset.x}px, ${panOffset.y}px) scale(${totalScale})`,
          transformOrigin: "0 0",
          position: "relative",
          width: naturalSize.width,
          height: naturalSize.height,
        }}
      >
        <canvas
          ref={canvasRef}
          className="block rounded-lg"
          style={{
            border: selectedColor
              ? `2px solid ${selectedColor}`
              : undefined,
            boxShadow: "0 1px 3px 0 rgba(0,0,0,0.04)",
          }}
        />
      </div>

      {/* Zoom indicator */}
      <div className="absolute bottom-2 right-2 flex items-center overflow-hidden rounded-lg border border-white/20 bg-white/60 shadow-sm backdrop-blur-xl dark:border-white/10 dark:bg-zinc-900/50">
        <span className="px-2 py-0.5 text-[11px] tabular-nums text-foreground">
          {Math.round(totalScale * 100)}%
        </span>
        <button
          className={`border-l border-white/20 px-2 py-0.5 text-[11px] font-medium dark:border-white/10 ${zoom === 1 ? "text-muted-foreground" : "text-foreground hover:bg-black/10 dark:hover:bg-white/10"}`}
          disabled={zoom === 1}
          onClick={(e) => {
            e.stopPropagation();
            resetZoom();
          }}
        >
          Fit
        </button>
        <button
          className={`border-l border-white/20 px-2 py-0.5 text-[11px] font-medium dark:border-white/10 ${Math.round(totalScale * 100) === 100 ? "text-muted-foreground" : "text-foreground hover:bg-black/10 dark:hover:bg-white/10"}`}
          disabled={Math.round(totalScale * 100) === 100}
          onClick={(e) => {
            e.stopPropagation();
            zoomToNative();
          }}
        >
          1:1
        </button>
        <button
          className={`border-l border-white/20 px-1.5 py-0.5 dark:border-white/10 ${
            inspecting
              ? "bg-primary text-primary-foreground"
              : selectedObjId != null && masks[selectedObjId]?.bbox
                ? "text-foreground hover:bg-black/10 dark:hover:bg-white/10"
                : "text-muted-foreground"
          }`}
          disabled={!inspecting && (selectedObjId == null || !masks[selectedObjId]?.bbox)}
          onClick={(e) => {
            e.stopPropagation();
            toggleInspect();
          }}
          title="Zoom to selected mask"
        >
          <Crop className="h-3 w-3" />
        </button>
      </div>

      {/* Cancel multi-selection */}
      {selectedObjIds.size >= 2 && onCancelMultiSelect && (
        <button
          className="absolute bottom-2 left-1/2 z-10 -translate-x-1/2 flex items-center gap-1.5 rounded-full border border-white/20 bg-black/50 px-3 py-1 text-[11px] font-medium text-white shadow-sm backdrop-blur-xl hover:bg-black/70 transition-colors"
          onClick={(e) => { e.stopPropagation(); onCancelMultiSelect(); }}
        >
          <X className="h-3 w-3" />
          Cancel multi-selection
        </button>
      )}

      {/* Segmenting overlay */}
      {segmenting && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/20">
          <div className="flex items-center gap-2 rounded-lg bg-white/90 px-4 py-2 shadow-sm">
            <Loader2 className="h-4 w-4 animate-spin text-primary" />
            <span className="text-xs font-medium text-foreground">
              Segmenting...
            </span>
          </div>
        </div>
      )}

      {/* Context menu */}
      {contextMenu && onDeleteMasks && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => setContextMenu(null)}
            onContextMenu={(e) => { e.preventDefault(); setContextMenu(null); }}
          />
          <div
            className="absolute z-50 min-w-[160px] rounded-md border border-border bg-popover py-1 shadow-md"
            style={{ left: contextMenu.x, top: contextMenu.y }}
          >
            <button
              type="button"
              className="flex w-full items-center gap-2 px-3 py-1.5 text-xs text-foreground hover:bg-accent"
              onClick={() => {
                setContextMenu(null);
                onDeleteMasks();
              }}
            >
              <Trash2 className="h-3.5 w-3.5" />
              Delete masks...
            </button>
          </div>
        </>
      )}
    </div>
  );
}
