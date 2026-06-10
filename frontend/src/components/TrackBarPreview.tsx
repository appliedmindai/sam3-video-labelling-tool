import { useRef, useEffect, useState, useMemo } from "react";
import { loadFrameMasks } from "../api.ts";
import { getCachedMasks, putMasks } from "../maskCache.ts";
import { getFrame } from "../frameCache.ts";
import type { MaskResult, BboxPadding } from "../types.ts";

interface Props {
  sessionId: string;
  frameIdx: number;
  /** Mask from the LRU window (may be null even if mask exists on disk) */
  mask: MaskResult | null;
  /** Whether this frame is known to have a mask for the selected object */
  hasMask: boolean;
  selectedObjId: number;
  color: string;
  /** Bbox padding config — per-object per-source-keyframe */
  bboxPadding: Record<number, Record<number, BboxPadding>>;
  /** Whether this frame is a keyframe (user-placed prompt) */
  isKeyframe?: boolean;
  /** Override preview width (default 160) */
  previewWidth?: number;
  /** Override preview max height (default 120) */
  previewMaxHeight?: number;
}

// Decode RLE counts (LEB128 delta encoding, column-major)
function decodeRleCounts(encoded: string): Int32Array {
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
  return Int32Array.from(counts);
}

function drawMaskOverlay(
  ctx: CanvasRenderingContext2D,
  rle: { counts: string; size: number[] },
  color: string,
  cropX: number,
  cropY: number,
  cropW: number,
  cropH: number,
  drawW: number,
  drawH: number,
) {
  const counts = decodeRleCounts(rle.counts);
  const [rleH] = rle.size;
  const r = parseInt(color.slice(1, 3), 16);
  const g = parseInt(color.slice(3, 5), 16);
  const b = parseInt(color.slice(5, 7), 16);

  // Read existing frame pixels so mask blends on top (not replaces)
  const imageData = ctx.getImageData(0, 0, drawW, drawH);
  const scaleX = drawW / cropW;
  const scaleY = drawH / cropH;
  const alpha = 100 / 255;

  let pixelIdx = 0;
  for (let i = 0; i < counts.length; i++) {
    const runLen = counts[i];
    if (i % 2 === 0) {
      pixelIdx += runLen;
      continue;
    }
    for (let j = 0; j < runLen; j++) {
      const col = Math.floor(pixelIdx / rleH);
      const row = pixelIdx % rleH;
      const cx = col - cropX;
      const cy = row - cropY;
      if (cx >= 0 && cx < cropW && cy >= 0 && cy < cropH) {
        const dx = Math.floor(cx * scaleX);
        const dy = Math.floor(cy * scaleY);
        if (dx < drawW && dy < drawH) {
          const offset = (dy * drawW + dx) * 4;
          imageData.data[offset] = Math.round(imageData.data[offset] * (1 - alpha) + r * alpha);
          imageData.data[offset + 1] = Math.round(imageData.data[offset + 1] * (1 - alpha) + g * alpha);
          imageData.data[offset + 2] = Math.round(imageData.data[offset + 2] * (1 - alpha) + b * alpha);
          imageData.data[offset + 3] = 255;
        }
      }
      pixelIdx++;
    }
  }
  ctx.putImageData(imageData, 0, 0);
}

// In-memory image cache — hot layer on top of IndexedDB to avoid repeated IDB reads during rapid hover
const imageCache = new Map<string, HTMLImageElement>();
const IMAGE_CACHE_MAX = 8;

function imageCacheKey(sessionId: string, frameIdx: number): string {
  return `${sessionId}/${frameIdx}`;
}

export function prefetchFrameImage(sessionId: string, frameIdx: number) {
  const key = imageCacheKey(sessionId, frameIdx);
  if (imageCache.has(key)) return;
  // Fire-and-forget: load from IndexedDB into in-memory image cache
  getFrame(sessionId, frameIdx).then((blobUrl) => {
    if (imageCache.has(key)) return;
    const img = new Image();
    img.onload = () => {
      if (imageCache.size >= IMAGE_CACHE_MAX) {
        const firstKey = imageCache.keys().next().value!;
        imageCache.delete(firstKey);
      }
      imageCache.set(key, img);
    };
    img.src = blobUrl;
  });
}

function getCachedImage(sessionId: string, frameIdx: number): HTMLImageElement | null {
  return imageCache.get(imageCacheKey(sessionId, frameIdx)) ?? null;
}

const PREVIEW_WIDTH = 160;
const PREVIEW_MAX_HEIGHT = 120;

export default function TrackBarPreview({
  sessionId,
  frameIdx,
  mask: maskProp,
  hasMask,
  selectedObjId,
  color,
  bboxPadding,
  isKeyframe,
  previewWidth: previewWidthProp,
  previewMaxHeight: previewMaxHeightProp,
}: Props) {
  const previewWidth = previewWidthProp ?? PREVIEW_WIDTH;
  const previewMaxHeight = previewMaxHeightProp ?? PREVIEW_MAX_HEIGHT;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  const [fetchedMask, setFetchedMask] = useState<MaskResult | null>(null);

  // Resolve mask: prefer prop (from LRU window), fall back to fetched
  const mask = maskProp ?? fetchedMask;

  // Load or retrieve cached image
  useEffect(() => {
    const cached = getCachedImage(sessionId, frameIdx);
    if (cached) {
      setImg(cached);
      return;
    }
    let cancelled = false;
    const key = imageCacheKey(sessionId, frameIdx);

    getFrame(sessionId, frameIdx).then((blobUrl) => {
      if (cancelled) return;
      const image = new Image();
      image.onload = () => {
        if (cancelled) return;
        if (imageCache.size >= IMAGE_CACHE_MAX) {
          const firstKey = imageCache.keys().next().value!;
          imageCache.delete(firstKey);
        }
        imageCache.set(key, image);
        setImg(image);
      };
      image.src = blobUrl;
    });

    return () => { cancelled = true; };
  }, [sessionId, frameIdx]);

  // Fetch mask data when not available from props but known to exist
  useEffect(() => {
    if (maskProp) {
      setFetchedMask(null);
      return;
    }
    if (!hasMask) {
      setFetchedMask(null);
      return;
    }

    let cancelled = false;

    (async () => {
      try {
        const cached = await getCachedMasks(sessionId, frameIdx);
        if (cached && cached[selectedObjId]) {
          if (!cancelled) setFetchedMask(cached[selectedObjId]);
          return;
        }
        const frameMasks = await loadFrameMasks(sessionId, frameIdx);
        if (cancelled) return;
        putMasks(sessionId, frameIdx, frameMasks).catch(() => {});
        setFetchedMask(frameMasks[selectedObjId] ?? null);
      } catch {
        if (!cancelled) setFetchedMask(null);
      }
    })();

    return () => { cancelled = true; };
  }, [sessionId, frameIdx, maskProp, hasMask, selectedObjId]);

  // Resolve padding for this mask's source keyframe (same logic as App.tsx)
  const padding = useMemo(() => {
    const zero: BboxPadding = { top: 0, bottom: 0, left: 0, right: 0 };
    if (!mask) return zero;
    const kf = mask.source_keyframe ?? frameIdx;
    return bboxPadding[selectedObjId]?.[kf] ?? zero;
  }, [mask, frameIdx, selectedObjId, bboxPadding]);

  // Compute padded crop geometry from bbox + padding (matches InspectView)
  const cropInfo = useMemo(() => {
    if (!img || !mask) return null;
    const [bx, by, bw, bh] = mask.bbox;
    const padTop = bh * (padding.top / 100);
    const padBottom = bh * (padding.bottom / 100);
    const padLeft = bw * (padding.left / 100);
    const padRight = bw * (padding.right / 100);
    const sx = Math.max(0, bx - padLeft);
    const sy = Math.max(0, by - padTop);
    const sx2 = Math.min(img.naturalWidth, bx + bw + padRight);
    const sy2 = Math.min(img.naturalHeight, by + bh + padBottom);
    const sw = sx2 - sx;
    const sh = sy2 - sy;
    const scale = Math.min(previewWidth / sw, previewMaxHeight / sh);
    const drawW = Math.round(sw * scale);
    const drawH = Math.round(sh * scale);
    return { sx, sy, sw, sh, drawW, drawH, bx, by, bw, bh };
  }, [img, mask, padding, previewWidth, previewMaxHeight]);

  // Draw cropped preview with mask overlay and bbox
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !img) return;
    const ctx = canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) return;

    if (!mask || !cropInfo) {
      canvas.width = 0;
      canvas.height = 0;
      return;
    }

    const { sx, sy, sw, sh, drawW, drawH, bx, by, bw, bh } = cropInfo;
    if (drawW <= 0 || drawH <= 0 || !isFinite(drawW) || !isFinite(drawH)) {
      canvas.width = 0;
      canvas.height = 0;
      return;
    }
    canvas.width = drawW;
    canvas.height = drawH;
    ctx.clearRect(0, 0, drawW, drawH);
    ctx.drawImage(img, sx, sy, sw, sh, 0, 0, drawW, drawH);

    // Draw mask overlay
    drawMaskOverlay(ctx, mask.rle, color, sx, sy, sw, sh, drawW, drawH);

    // Draw bbox as dashed rect (like InspectView)
    const scaleX = drawW / sw;
    const scaleY = drawH / sh;
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.strokeRect(
      (bx - sx) * scaleX,
      (by - sy) * scaleY,
      bw * scaleX,
      bh * scaleY,
    );
    ctx.setLineDash([]);
  }, [img, mask, color, cropInfo]);

  const borderClass = isKeyframe
    ? "rounded-lg border-2 border-black bg-popover shadow-md dark:border-white"
    : "rounded-lg border border-border bg-popover shadow-md";

  if (!hasMask) {
    return (
      <div className={`flex flex-col items-center gap-1 p-1.5 ${borderClass}`}>
        <div className="flex items-center justify-center rounded bg-muted/50" style={{ width: previewWidth, height: previewMaxHeight }}>
          <span className="text-xs text-muted-foreground">No Mask</span>
        </div>
        <span className="text-[10px] tabular-nums text-muted-foreground">
          Frame {frameIdx}{isKeyframe && " · Keyframe"}
        </span>
      </div>
    );
  }

  return (
    <div className={`flex flex-col items-center gap-1 p-1.5 ${borderClass}`}>
      {mask ? (
        <canvas ref={canvasRef} className="block rounded" style={{ maxWidth: previewWidth }} />
      ) : (
        <div className="flex items-center justify-center rounded bg-muted/50" style={{ width: previewWidth, height: previewMaxHeight }}>
          <span className="text-[10px] text-muted-foreground">Loading...</span>
        </div>
      )}
      <span className="text-[10px] tabular-nums text-muted-foreground">
        Frame {frameIdx}{isKeyframe && " · Keyframe"}
      </span>
    </div>
  );
}
