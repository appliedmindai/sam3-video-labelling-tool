import { Loader2, Check, X } from "lucide-react";
import { Progress } from "@/components/ui/progress";
import type { ServiceStatus } from "../types";
import { cancelJob } from "../api";
import { useState } from "react";

interface Props {
  status: ServiceStatus;
  /** Called after user dismisses an error (navigates back to session list). */
  onDismissError?: () => void;
}

/** Phase step labels for the progress indicator. */
const PHASES = [
  { key: "uploading", label: "Upload" },
  { key: "extracting", label: "Extract Frames" },
  { key: "initializing", label: "Load Model" },
] as const;

export default function PipelineProgress({ status, onDismissError }: Props) {
  const [cancelling, setCancelling] = useState(false);

  const phaseLabel =
    status.phase === "extracting"
      ? `Extracting frames — ${status.video_name}`
      : status.phase === "initializing"
        ? `Loading model — ${status.video_name}`
        : status.phase === "error"
          ? `Failed — ${status.error}`
          : "";

  const handleCancel = async () => {
    setCancelling(true);
    try {
      await cancelJob();
    } catch {
      // ignore — poll will pick up the state change
    }
  };

  if (status.phase === "error") {
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 p-8">
        <div className="flex items-center gap-2 text-red-500">
          <X className="h-5 w-5" />
          <span className="text-sm font-medium">Pipeline failed</span>
        </div>
        <p className="text-sm text-zinc-500 max-w-md text-center">{status.error}</p>
        {onDismissError && (
          <button
            onClick={onDismissError}
            className="text-sm text-indigo-500 hover:text-indigo-400 underline"
          >
            Back to sessions
          </button>
        )}
      </div>
    );
  }

  const progressPercent = Math.round(status.progress * 100);

  return (
    <div className="flex flex-col items-center justify-center h-full gap-6 p-8">
      <p className="text-sm text-zinc-400">{phaseLabel}</p>

      <div className="w-full max-w-sm">
        <Progress value={progressPercent} className="h-2" />
      </div>

      <p className="text-xs text-zinc-500">{progressPercent}%</p>

      {/* Phase step indicators */}
      <div className="flex items-center gap-4 text-xs text-zinc-500">
        {PHASES.map(({ key, label }) => {
          const phaseOrder = ["uploading", "extracting", "initializing"];
          const currentIdx = phaseOrder.indexOf(status.phase);
          const stepIdx = phaseOrder.indexOf(key);

          const isDone = stepIdx < currentIdx;
          const isActive = key === status.phase;

          return (
            <div key={key} className="flex items-center gap-1.5">
              {isDone ? (
                <Check className="h-3.5 w-3.5 text-emerald-500" />
              ) : isActive ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin text-indigo-500" />
              ) : (
                <div className="h-3.5 w-3.5 rounded-full border border-zinc-600" />
              )}
              <span className={isActive ? "text-zinc-200" : ""}>{label}</span>
            </div>
          );
        })}
      </div>

      {/* Cancel button during initializing */}
      {status.phase === "initializing" && (
        <button
          onClick={handleCancel}
          disabled={cancelling}
          className="text-xs text-zinc-500 hover:text-zinc-300 underline disabled:opacity-50"
        >
          {cancelling ? "Cancelling..." : "Cancel"}
        </button>
      )}
    </div>
  );
}
