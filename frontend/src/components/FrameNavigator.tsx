import { useEffect, useCallback, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import { ChevronsLeft, ChevronsRight, ChevronLeft, ChevronRight, Play, Pause } from "lucide-react";
import TrackBarPreview, { prefetchFrameImage } from "./TrackBarPreview.tsx";

interface Props {
  currentFrame: number;
  frameCount: number;
  keyframes: Set<number>;
  /** Frames where the currently selected object has masks (empty if none selected) */
  selectedObjFrames: Set<number>;
  /** Color of the selected object's class */
  selectedObjColor?: string;
  /** Frames with low confidence scores after propagation */
  lowConfidenceFrames?: Set<number>;
  isPlaying: boolean;
  playbackFps: number;
  onChange: (frame: number) => void;
  onPlayToggle: () => void;
  onPlaybackFpsChange: (fps: number) => void;
  onPrevMask?: () => void;
  onNextMask?: () => void;
  /** Session ID for loading frame images in hover preview */
  sessionId?: string;
  /** All masks data (for hover preview rendering) */
  masks?: Record<number, Record<number, import("../types.ts").MaskResult>>;
  /** Currently selected object ID (for hover preview) */
  selectedObjId?: number | null;
  /** Bbox padding config per object per source keyframe */
  bboxPadding?: Record<number, Record<number, import("../types.ts").BboxPadding>>;
}

const FPS_OPTIONS = [2, 5, 10, 15, 30];

export default function FrameNavigator({
  currentFrame,
  frameCount,
  keyframes,
  selectedObjFrames,
  selectedObjColor,
  lowConfidenceFrames,
  isPlaying,
  playbackFps,
  onChange,
  onPlayToggle,
  onPlaybackFpsChange,
  onPrevMask,
  onNextMask,
  sessionId,
  masks,
  selectedObjId,
  bboxPadding,
}: Props) {
  const goPrev = useCallback(() => {
    if (currentFrame > 0) onChange(currentFrame - 1);
  }, [currentFrame, onChange]);

  const goNext = useCallback(() => {
    if (currentFrame < frameCount - 1) onChange(currentFrame + 1);
  }, [currentFrame, frameCount, onChange]);

  useEffect(() => {
    function handleKey(e: KeyboardEvent) {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      )
        return;
      if (e.key === "ArrowLeft") {
        e.preventDefault();
        goPrev();
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        goNext();
      } else if (e.key === " ") {
        e.preventDefault();
        onPlayToggle();
      }
    }
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [goPrev, goNext, onPlayToggle]);

  // Playback interval
  const frameRef = useRef(currentFrame);
  frameRef.current = currentFrame;
  const frameCountRef = useRef(frameCount);
  frameCountRef.current = frameCount;

  useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(() => {
      const next = frameRef.current + 1;
      if (next >= frameCountRef.current) {
        onPlayToggle(); // auto-pause at end
      } else {
        onChange(next);
      }
    }, 1000 / playbackFps);
    return () => clearInterval(interval);
  }, [isPlaying, playbackFps, onChange, onPlayToggle]);

  // Hover preview state
  const trackBarRef = useRef<HTMLDivElement>(null);
  const [hoverFrame, setHoverFrame] = useState<number | null>(null);
  const [hoverX, setHoverX] = useState(0);

  const handleTrackMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const fraction = Math.max(0, Math.min(1, x / rect.width));
    const frame = Math.round(fraction * (frameCount - 1));
    setHoverFrame(frame);
    setHoverX(x);

    // Prefetch adjacent frames
    if (sessionId) {
      for (let offset = -2; offset <= 2; offset++) {
        const f = frame + offset;
        if (f >= 0 && f < frameCount) {
          prefetchFrameImage(sessionId, f);
        }
      }
    }
  }, [frameCount, sessionId]);

  const handleTrackMouseLeave = useCallback(() => {
    setHoverFrame(null);
  }, []);

  // Build track segments: contiguous runs of frames with masks, plus keyframe positions
  const trackSegments = useMemo(() => {
    const segments: Array<{
      start: number;
      end: number;
      type: "inferred" | "low-confidence";
    }> = [];
    const keyframePositions: number[] = [];

    if (frameCount <= 1) return { segments, keyframePositions };

    // Collect keyframe positions
    for (const frame of keyframes) {
      keyframePositions.push(frame);
    }
    keyframePositions.sort((a, b) => a - b);

    // Build sorted list of frames with masks
    const sortedFrames = Array.from(selectedObjFrames).sort((a, b) => a - b);
    if (sortedFrames.length === 0) return { segments, keyframePositions };

    // Group consecutive frames into segments
    let segStart = sortedFrames[0];
    let segEnd = sortedFrames[0];
    let hasLowConf = lowConfidenceFrames?.has(sortedFrames[0]) ?? false;

    for (let i = 1; i < sortedFrames.length; i++) {
      const frame = sortedFrames[i];
      if (frame === segEnd + 1) {
        // Extend current segment
        segEnd = frame;
        if (lowConfidenceFrames?.has(frame)) hasLowConf = true;
      } else {
        // Flush current segment
        segments.push({
          start: segStart,
          end: segEnd,
          type: hasLowConf ? "low-confidence" : "inferred",
        });
        // Start new segment
        segStart = frame;
        segEnd = frame;
        hasLowConf = lowConfidenceFrames?.has(frame) ?? false;
      }
    }
    // Flush last segment
    segments.push({
      start: segStart,
      end: segEnd,
      type: hasLowConf ? "low-confidence" : "inferred",
    });

    return { segments, keyframePositions };
  }, [frameCount, keyframes, selectedObjFrames, lowConfidenceFrames]);

  if (frameCount === 0) return null;

  const maxFrame = frameCount - 1;
  const hasLowConfidence = lowConfidenceFrames != null && lowConfidenceFrames.size > 0;

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center gap-3 rounded-lg border border-border bg-card px-3 py-2">
        {onPrevMask && (
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 shrink-0"
            onClick={onPrevMask}
            disabled={isPlaying}
            title="Previous mask"
          >
            <ChevronsLeft className="h-4 w-4" />
          </Button>
        )}

        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7 shrink-0"
          onClick={goPrev}
          disabled={currentFrame <= 0 || isPlaying}
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>

        <Button
          variant={isPlaying ? "default" : "ghost"}
          size="icon"
          className="h-7 w-7 shrink-0"
          onClick={onPlayToggle}
          disabled={frameCount <= 1}
        >
          {isPlaying ? (
            <Pause className="h-4 w-4" />
          ) : (
            <Play className="h-4 w-4" />
          )}
        </Button>

        <div className="relative flex-1">
          <Slider
            value={[currentFrame]}
            min={0}
            max={maxFrame}
            step={1}
            onValueChange={([v]) => onChange(v)}
            className="flex-1"
          />

          {/* Track bar — mx-2 matches the slider thumb's in-bounds inset (size-4 thumb
              center travels [8px, width-8px]), so frame percentages align with the thumb */}
          <div
            ref={trackBarRef}
            className="relative mx-2 mt-1 h-2.5 cursor-pointer rounded-full bg-muted/50"
            onMouseMove={handleTrackMouseMove}
            onMouseLeave={handleTrackMouseLeave}
            onClick={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              const fraction = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
              onChange(Math.round(fraction * maxFrame));
            }}
          >
            {/* Mask presence segments */}
            {maxFrame > 0 && trackSegments.segments.map((seg, i) => {
              const left = (seg.start / maxFrame) * 100;
              const width = ((seg.end - seg.start) / maxFrame) * 100;
              return (
                <div
                  key={i}
                  className="absolute top-0 h-full rounded-full"
                  style={{
                    left: `${left}%`,
                    width: `${Math.max(width, 0.3)}%`,
                    backgroundColor: seg.type === "low-confidence" ? "#f59e0b" : (selectedObjColor ?? "#94a3b8"),
                    opacity: seg.type === "low-confidence" ? 0.8 : 0.7,
                  }}
                />
              );
            })}
            {/* Keyframe markers -- thick black vertical lines */}
            {maxFrame > 0 && trackSegments.keyframePositions.map((frame) => (
              <div
                key={`kf-${frame}`}
                className="absolute top-0 h-full w-[3px] rounded-full bg-black dark:bg-white"
                style={{
                  left: `${(frame / maxFrame) * 100}%`,
                  transform: "translateX(-50%)",
                }}
              />
            ))}

            {/* Hover preview tooltip */}
            {hoverFrame != null && sessionId && selectedObjId != null && (
              <div
                className="pointer-events-none absolute bottom-full mb-2"
                style={{
                  left: Math.max(80, Math.min((trackBarRef.current?.clientWidth ?? 0) - 80, hoverX)),
                  transform: "translateX(-50%)",
                }}
              >
                <TrackBarPreview
                  sessionId={sessionId}
                  frameIdx={hoverFrame}
                  mask={masks?.[hoverFrame]?.[selectedObjId] ?? null}
                  hasMask={selectedObjFrames.has(hoverFrame)}
                  selectedObjId={selectedObjId}
                  color={selectedObjColor ?? "#94a3b8"}
                  bboxPadding={bboxPadding ?? {}}
                  isKeyframe={keyframes.has(hoverFrame)}
                />
              </div>
            )}
          </div>
        </div>

        <Button
          variant="ghost"
          size="icon"
          className="h-7 w-7 shrink-0"
          onClick={goNext}
          disabled={currentFrame >= maxFrame || isPlaying}
        >
          <ChevronRight className="h-4 w-4" />
        </Button>

        {onNextMask && (
          <Button
            variant="ghost"
            size="icon"
            className="h-7 w-7 shrink-0"
            onClick={onNextMask}
            disabled={isPlaying}
            title="Next mask"
          >
            <ChevronsRight className="h-4 w-4" />
          </Button>
        )}

        <button
          className="shrink-0 rounded px-1.5 py-0.5 text-[10px] tabular-nums text-muted-foreground hover:bg-accent"
          onClick={() => {
            const idx = FPS_OPTIONS.indexOf(playbackFps);
            const next = FPS_OPTIONS[(idx + 1) % FPS_OPTIONS.length];
            onPlaybackFpsChange(next);
          }}
          title="Click to change playback speed"
        >
          {playbackFps} fps
        </button>

        <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
          {currentFrame} / {maxFrame}
        </span>
      </div>

      {/* Legend */}
      {(trackSegments.segments.length > 0 || trackSegments.keyframePositions.length > 0) && (
        <div className="flex items-center justify-center gap-4 text-[10px] text-muted-foreground">
          {trackSegments.keyframePositions.length > 0 && (
            <span className="flex items-center gap-1">
              <span className="inline-block h-2.5 w-[3px] rounded-full bg-black dark:bg-white" />
              Keyframe
            </span>
          )}
          {trackSegments.segments.length > 0 && (
            <span className="flex items-center gap-1">
              <span
                className="inline-block h-2.5 w-3 rounded-full"
                style={{
                  backgroundColor: selectedObjColor ?? "#94a3b8",
                  opacity: 0.7,
                }}
              />
              Mask
            </span>
          )}
          {hasLowConfidence && (
            <span className="flex items-center gap-1">
              <span className="inline-block h-2.5 w-3 rounded-full bg-amber-500 opacity-80" />
              Low confidence
            </span>
          )}
        </div>
      )}
    </div>
  );
}
