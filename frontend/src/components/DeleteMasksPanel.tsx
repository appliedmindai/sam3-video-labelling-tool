import { useCallback, useEffect, useMemo, useState } from "react";
import { AlertTriangle, ChevronLeft, ChevronRight, Crosshair, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import type { MaskResult, ObjectClass, TrackedObject, BboxPadding } from "@/types";
import TrackBarPreview from "./TrackBarPreview";

/* ------------------------------------------------------------------ */
/*  FrameSelector                                                      */
/* ------------------------------------------------------------------ */

const PREVIEW_W = 220;
const PREVIEW_H = 160;

interface FrameSelectorProps {
  label: string;
  sessionId: string;
  frameIdx: number;
  mask: MaskResult | null;
  hasMask: boolean;
  selectedObjId: number;
  color: string;
  bboxPadding: Record<number, Record<number, BboxPadding>>;
  isKeyframe: boolean;
  maskedFrames: number[];
  currentFrame: number;
  onChange: (frame: number) => void;
}

function FrameSelector({
  label,
  sessionId,
  frameIdx,
  mask,
  hasMask,
  selectedObjId,
  color,
  bboxPadding,
  isKeyframe,
  maskedFrames,
  currentFrame,
  onChange,
}: FrameSelectorProps) {
  const minFrame = maskedFrames[0] ?? 0;
  const maxFrame = maskedFrames[maskedFrames.length - 1] ?? 0;

  const handlePrev = useCallback(() => {
    for (let i = maskedFrames.length - 1; i >= 0; i--) {
      if (maskedFrames[i] < frameIdx) {
        onChange(maskedFrames[i]);
        return;
      }
    }
  }, [maskedFrames, frameIdx, onChange]);

  const handleNext = useCallback(() => {
    for (let i = 0; i < maskedFrames.length; i++) {
      if (maskedFrames[i] > frameIdx) {
        onChange(maskedFrames[i]);
        return;
      }
    }
  }, [maskedFrames, frameIdx, onChange]);

  return (
    <div className="space-y-2">
      {/* Label + frame number input */}
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-muted-foreground">{label}</span>
        <input
          type="number"
          value={frameIdx}
          min={minFrame}
          max={maxFrame}
          onChange={(e) => {
            const v = parseInt(e.target.value, 10);
            if (!isNaN(v)) onChange(v);
          }}
          className="h-6 w-16 rounded border border-border bg-background px-1.5 text-right text-xs tabular-nums text-foreground outline-none focus:ring-1 focus:ring-primary"
        />
      </div>

      {/* Slider — always visible */}
      <div className="flex items-center gap-1.5">
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6 shrink-0"
          onClick={handlePrev}
          disabled={frameIdx <= minFrame}
        >
          <ChevronLeft className="h-3.5 w-3.5" />
        </Button>
        <Slider
          value={[frameIdx]}
          min={minFrame}
          max={maxFrame}
          step={1}
          onValueChange={([v]) => onChange(v)}
          className="flex-1"
        />
        <Button
          variant="ghost"
          size="icon"
          className="h-6 w-6 shrink-0"
          onClick={handleNext}
          disabled={frameIdx >= maxFrame}
        >
          <ChevronRight className="h-3.5 w-3.5" />
        </Button>
      </div>

      {/* Select current frame button */}
      <Button
        variant="outline"
        size="sm"
        className="w-full gap-1.5 text-xs"
        onClick={() => onChange(currentFrame)}
      >
        <Crosshair className="h-3 w-3" />
        Select this frame
      </Button>

      {/* Fixed-size preview using TrackBarPreview */}
      <TrackBarPreview
        sessionId={sessionId}
        frameIdx={frameIdx}
        mask={mask}
        hasMask={hasMask}
        selectedObjId={selectedObjId}
        color={color}
        bboxPadding={bboxPadding}
        isKeyframe={isKeyframe}
        previewWidth={PREVIEW_W}
        previewMaxHeight={PREVIEW_H}
      />
    </div>
  );
}

/* ------------------------------------------------------------------ */
/*  DeleteMasksPanel                                                   */
/* ------------------------------------------------------------------ */

interface Props {
  sessionId: string;
  selectedObjId: number;
  objects: TrackedObject[];
  classes: ObjectClass[];
  masks: Record<number, Record<number, MaskResult>>;
  frameObjIndex: Map<number, Set<number>>;
  prompts: Record<number, Record<number, unknown>>;
  bboxPadding: Record<number, Record<number, BboxPadding>>;
  initialFrame: number;
  currentFrame: number;
  onDelete: (frameIndices: number[]) => Promise<void>;
  onClose: () => void;
}

export default function DeleteMasksPanel({
  sessionId,
  selectedObjId,
  objects,
  classes,
  masks,
  frameObjIndex,
  prompts,
  bboxPadding,
  initialFrame,
  currentFrame,
  onDelete,
  onClose,
}: Props) {
  // Frames that have masks for this object (sorted)
  const maskedFrames = useMemo(() => {
    const frames: number[] = [];
    frameObjIndex.forEach((objIds, frameIdx) => {
      if (objIds.has(selectedObjId)) frames.push(frameIdx);
    });
    return frames.sort((a, b) => a - b);
  }, [frameObjIndex, selectedObjId]);

  const [startFrame, setStartFrame] = useState(initialFrame);
  const [endFrame, setEndFrame] = useState(initialFrame);
  const [deleting, setDeleting] = useState(false);

  // Object + class info
  const obj = objects.find((o) => o.obj_id === selectedObjId);
  const cls = obj ? classes.find((c) => c.id === obj.class_id) : null;
  const color = cls?.color ?? "#6366f1";

  // Keyframes for the selected object
  const keyframeSet = useMemo(() => {
    const kfs = new Set<number>();
    for (const [frameStr, objPrompts] of Object.entries(prompts)) {
      if (selectedObjId in (objPrompts as Record<number, unknown>)) {
        kfs.add(Number(frameStr));
      }
    }
    return kfs;
  }, [prompts, selectedObjId]);

  // Normalized range
  const lo = Math.min(startFrame, endFrame);
  const hi = Math.max(startFrame, endFrame);

  // Frames in range (only those that have masks)
  const framesInRange = useMemo(() => {
    return maskedFrames.filter((f) => f >= lo && f <= hi);
  }, [maskedFrames, lo, hi]);

  const keyframesInRange = useMemo(
    () => framesInRange.filter((f) => keyframeSet.has(f)),
    [framesInRange, keyframeSet],
  );

  const handleDelete = useCallback(async () => {
    setDeleting(true);
    try {
      await onDelete(framesInRange);
    } finally {
      setDeleting(false);
    }
  }, [framesInRange, onDelete]);

  // Close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose]);

  return (
    <div className="flex h-full flex-col">
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-3">
        <div>
          <h2 className="text-sm font-semibold">Delete Masks</h2>
          <p className="text-xs text-muted-foreground">
            {cls?.name ?? "Object"} #{selectedObjId}
          </p>
        </div>
        <Button variant="ghost" size="icon" className="h-6 w-6" onClick={onClose}>
          <X className="h-3.5 w-3.5" />
        </Button>
      </div>

      <Separator />

      {maskedFrames.length === 0 ? (
        <div className="flex flex-1 items-center justify-center px-4 py-8">
          <p className="text-xs text-muted-foreground">No masks for this object</p>
        </div>
      ) : (
        <div className="flex-1 space-y-4 overflow-y-auto px-4 py-3">
          <FrameSelector
            label="Start Frame"
            sessionId={sessionId}
            frameIdx={startFrame}
            mask={masks[startFrame]?.[selectedObjId] ?? null}
            hasMask={frameObjIndex.get(startFrame)?.has(selectedObjId) ?? false}
            selectedObjId={selectedObjId}
            color={color}
            bboxPadding={bboxPadding}
            isKeyframe={keyframeSet.has(startFrame)}
            maskedFrames={maskedFrames}
            currentFrame={currentFrame}
            onChange={setStartFrame}
          />

          <Separator />

          <FrameSelector
            label="End Frame"
            sessionId={sessionId}
            frameIdx={endFrame}
            mask={masks[endFrame]?.[selectedObjId] ?? null}
            hasMask={frameObjIndex.get(endFrame)?.has(selectedObjId) ?? false}
            selectedObjId={selectedObjId}
            color={color}
            bboxPadding={bboxPadding}
            isKeyframe={keyframeSet.has(endFrame)}
            maskedFrames={maskedFrames}
            currentFrame={currentFrame}
            onChange={setEndFrame}
          />
        </div>
      )}

      <Separator />

      {/* Summary + warnings + actions */}
      <div className="space-y-3 px-4 py-3">
        <p className="text-xs text-muted-foreground">
          {framesInRange.length} mask{framesInRange.length !== 1 ? "s" : ""}
          {keyframesInRange.length > 0 &&
            ` · ${keyframesInRange.length} keyframe${keyframesInRange.length !== 1 ? "s" : ""}`}
        </p>

        {keyframesInRange.length > 0 && (
          <div className="flex items-start gap-2 rounded-md border border-amber-200 bg-amber-50 p-2 dark:border-amber-900/50 dark:bg-amber-950/30">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-600" />
            <p className="text-[11px] text-amber-800 dark:text-amber-200">
              {keyframesInRange.length} keyframe{keyframesInRange.length !== 1 ? "s" : ""} will be
              deleted. Their prompts cannot be recovered.
            </p>
          </div>
        )}

        <p className="text-[11px] text-muted-foreground">This cannot be undone.</p>

        <div className="flex gap-2">
          <Button variant="outline" size="sm" className="flex-1" onClick={onClose}>
            Cancel
          </Button>
          <Button
            variant="destructive"
            size="sm"
            className="flex-1"
            disabled={framesInRange.length === 0 || deleting}
            onClick={handleDelete}
          >
            {deleting
              ? "Deleting..."
              : `Delete ${framesInRange.length}`}
          </Button>
        </div>
      </div>
    </div>
  );
}
