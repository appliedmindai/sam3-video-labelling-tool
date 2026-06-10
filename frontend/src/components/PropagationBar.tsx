import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { ArrowLeft, ArrowRight, ArrowLeftRight, Loader2, Square, AlertTriangle, X } from "lucide-react";
import type { ConfidenceWarning, TrackedObject, ObjectClass } from "../types";

interface Props {
  sessionId: string | null;
  currentFrame: number;
  frameCount: number;
  propagating: boolean;
  progress: number;
  onPropagate: (direction: "forward" | "backward" | "both") => void;
  onCancel: () => void;
  confidenceWarnings?: ConfidenceWarning[];
  onGoToFrame?: (frame: number) => void;
  selectedObjIds: Set<number>;
  objects: TrackedObject[];
  classes: ObjectClass[];
  onRemoveFromSelection?: (objId: number) => void;
}

export default function PropagationBar({
  sessionId,
  currentFrame,
  frameCount: _frameCount,
  propagating,
  progress,
  onPropagate,
  onCancel,
  confidenceWarnings = [],
  onGoToFrame,
  selectedObjIds,
  objects,
  classes,
  onRemoveFromSelection,
}: Props) {
  const disabled = !sessionId || propagating;
  const [warningDismissed, setWarningDismissed] = useState(false);

  // Reset dismissed state when a new propagation starts
  useEffect(() => {
    if (propagating) setWarningDismissed(false);
  }, [propagating]);

  const affectedFrameCount = new Set(confidenceWarnings.map((w) => w.frame_idx)).size;
  const firstWarningFrame = confidenceWarnings.length > 0
    ? confidenceWarnings.reduce((min, w) => w.frame_idx < min ? w.frame_idx : min, confidenceWarnings[0].frame_idx)
    : null;

  return (
    <div className="flex flex-col gap-3">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        {selectedObjIds.size >= 2
          ? `Propagate ${selectedObjIds.size} Objects`
          : "Mask Propagation"}
      </h3>

      {selectedObjIds.size >= 2 && (
        <div className="flex flex-wrap gap-1">
          {[...selectedObjIds].map((objId) => {
            const obj = objects.find((o) => o.obj_id === objId);
            const cls = obj ? classes.find((c) => c.id === obj.class_id) : null;
            return (
              <span
                key={objId}
                className="inline-flex items-center gap-1 rounded-full border border-border bg-card px-2 py-0.5 text-[10px] font-medium"
              >
                {cls && (
                  <span
                    className="inline-block h-2 w-2 rounded-full"
                    style={{ backgroundColor: cls.color }}
                  />
                )}
                Object #{objId}
                {onRemoveFromSelection && (
                  <button
                    className="ml-0.5 rounded-full p-0.5 hover:bg-muted"
                    onClick={(e) => {
                      e.stopPropagation();
                      onRemoveFromSelection(objId);
                    }}
                  >
                    <X className="h-2.5 w-2.5" />
                  </button>
                )}
              </span>
            );
          })}
        </div>
      )}

      <div className="grid grid-cols-3 gap-1 rounded-lg border border-border bg-card p-1">
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="sm"
              className="h-auto flex-col gap-0.5 px-1 py-1.5"
              disabled={disabled}
              onClick={() => onPropagate("backward")}
            >
              <ArrowLeft className="h-4 w-4" />
              <span className="text-[10px] leading-none">Back</span>
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Track objects backward from current frame</p>
          </TooltipContent>
        </Tooltip>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="sm"
              className="h-auto flex-col gap-0.5 px-1 py-1.5"
              disabled={disabled}
              onClick={() => onPropagate("both")}
            >
              <ArrowLeftRight className="h-4 w-4" />
              <span className="text-[10px] leading-none">Both</span>
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Track objects in both directions</p>
          </TooltipContent>
        </Tooltip>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="sm"
              className="h-auto flex-col gap-0.5 px-1 py-1.5"
              disabled={disabled}
              onClick={() => onPropagate("forward")}
            >
              <ArrowRight className="h-4 w-4" />
              <span className="text-[10px] leading-none">Forward</span>
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Track objects forward from current frame</p>
          </TooltipContent>
        </Tooltip>
      </div>

      {propagating && (
        <div className="flex flex-col gap-1.5">
          <div className="flex items-center gap-2">
            <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
            <span className="text-xs text-muted-foreground">
              Frame {currentFrame} &middot; {Math.round(progress * 100)}%
            </span>
          </div>
          {/* Live warning during propagation */}
          {affectedFrameCount > 0 && (
            <div className="flex items-center gap-1.5 text-xs text-amber-500">
              <AlertTriangle className="h-3 w-3 shrink-0" />
              <span>{affectedFrameCount} frame(s) with low confidence</span>
            </div>
          )}
          <Button
            variant="destructive"
            size="sm"
            className="h-7 gap-1.5 text-xs"
            onClick={onCancel}
          >
            <Square className="h-3 w-3" />
            Stop
          </Button>
        </div>
      )}

      {/* Post-propagation warning */}
      {!propagating && affectedFrameCount > 0 && !warningDismissed && (
        <div className="rounded-md border border-amber-500/30 bg-amber-500/10 p-2.5">
          <div className="flex items-start gap-2">
            <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-amber-500" />
            <div className="flex-1 text-xs text-amber-500">
              <p className="font-medium">
                {affectedFrameCount} frame(s) may have degraded tracking
              </p>
              <p className="mt-1 text-amber-500/80">
                Consider adding a new keyframe to improve accuracy.
              </p>
              {firstWarningFrame != null && onGoToFrame && (
                <button
                  className="mt-1.5 font-medium underline hover:text-amber-400"
                  onClick={() => onGoToFrame(firstWarningFrame)}
                >
                  Go to frame {firstWarningFrame}
                </button>
              )}
            </div>
            <button
              className="shrink-0 rounded p-0.5 text-amber-500 hover:bg-amber-500/20"
              onClick={() => setWarningDismissed(true)}
            >
              <X className="h-3 w-3" />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
