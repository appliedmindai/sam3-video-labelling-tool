import { useState, useCallback, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { MousePointer, MousePointerClick, Square, Trash2, Undo2, ScanSearch, Loader2 } from "lucide-react";

interface Props {
  toolMode: "pointer" | "click" | "box";
  onToolModeChange: (mode: "pointer" | "click" | "box") => void;
  onUndo: () => void;
  canUndo: boolean;
  onDeleteMasks?: () => void;
  canDeleteMasks?: boolean;
  onDetect?: (text: string) => void;
  onCancelDetect?: () => void;
  detectEnabled?: boolean;
  detectLoading?: boolean;
  selectedClassName?: string;
  propagating?: boolean;
  multiSelect?: boolean;
}

export default function ToolBar({
  toolMode, onToolModeChange, onUndo, canUndo,
  onDeleteMasks, canDeleteMasks,
  onDetect, onCancelDetect, detectEnabled, detectLoading, selectedClassName,
  propagating, multiSelect,
}: Props) {
  const [textInput, setTextInput] = useState("");
  const [showTextInput, setShowTextInput] = useState(false);

  // Pre-fill with class name when it changes or panel opens
  useEffect(() => {
    if (showTextInput && selectedClassName) {
      setTextInput(selectedClassName);
    }
  }, [showTextInput, selectedClassName]);

  const handleSubmit = useCallback(() => {
    const trimmed = textInput.trim();
    if (!trimmed || !onDetect) return;
    onDetect(trimmed);
  }, [textInput, onDetect]);

  const creationDisabled = propagating || multiSelect;

  return (
    <div className="flex flex-col gap-2">
      <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        Tools
      </h3>
      <div className="grid grid-cols-3 gap-1 rounded-lg border border-border bg-card p-1">
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant={toolMode === "pointer" ? "default" : "ghost"}
              size="sm"
              onClick={() => onToolModeChange("pointer")}
              className="h-auto flex-col gap-0.5 px-1 py-1.5"
            >
              <MousePointer className="h-4 w-4" />
              <span className="text-[10px] leading-none">Select</span>
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Navigate and select objects on canvas</p>
          </TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="sm"
              onClick={onUndo}
              disabled={!canUndo}
              className="h-auto flex-col gap-0.5 px-1 py-1.5"
            >
              <Undo2 className="h-4 w-4" />
              <span className="text-[10px] leading-none">Undo</span>
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Undo last click point ({navigator.platform.includes("Mac") ? "\u2318" : "Ctrl"}+Z)</p>
          </TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant="ghost"
              size="sm"
              onClick={onDeleteMasks}
              disabled={!canDeleteMasks}
              className="h-auto flex-col gap-0.5 px-1 py-1.5"
            >
              <Trash2 className="h-4 w-4" />
              <span className="text-[10px] leading-none">Delete</span>
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Delete masks for selected object (Del)</p>
          </TooltipContent>
        </Tooltip>
      </div>

      <h3 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
        Mask Creation
      </h3>
      <div className="grid grid-cols-3 gap-1 rounded-lg border border-border bg-card p-1">
        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant={toolMode === "click" && !creationDisabled ? "default" : "ghost"}
              size="sm"
              onClick={() => onToolModeChange("click")}
              disabled={creationDisabled}
              className="h-auto flex-col gap-0.5 px-1 py-1.5"
            >
              <MousePointerClick className="h-4 w-4" />
              <span className="text-[10px] leading-none">Click</span>
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Left-click: positive point / Right-click: negative point</p>
          </TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant={toolMode === "box" && !creationDisabled ? "default" : "ghost"}
              size="sm"
              onClick={() => onToolModeChange("box")}
              disabled={creationDisabled}
              className="h-auto flex-col gap-0.5 px-1 py-1.5"
            >
              <Square className="h-4 w-4" />
              <span className="text-[10px] leading-none">Box</span>
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Click and drag to draw a bounding box</p>
          </TooltipContent>
        </Tooltip>

        <Tooltip>
          <TooltipTrigger asChild>
            <Button
              variant={showTextInput && !creationDisabled ? "default" : "ghost"}
              size="sm"
              onClick={() => setShowTextInput((v) => !v)}
              disabled={!detectEnabled || creationDisabled}
              className="h-auto flex-col gap-0.5 px-1 py-1.5"
            >
              {detectLoading ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <ScanSearch className="h-4 w-4" />
              )}
              <span className="text-[10px] leading-none">Detect</span>
            </Button>
          </TooltipTrigger>
          <TooltipContent>
            <p>Detect objects by description (pre-filled with class name)</p>
          </TooltipContent>
        </Tooltip>
      </div>

      {showTextInput && !creationDisabled && (
        <div className="flex items-center gap-1.5">
          <Input
            type="text"
            placeholder="Describe object (e.g. cup, person)..."
            value={textInput}
            onChange={(e) => setTextInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") handleSubmit();
            }}
            disabled={detectLoading}
            className="h-8 text-xs"
            autoFocus
          />
          {detectLoading ? (
            <Button
              size="sm"
              variant="destructive"
              onClick={onCancelDetect}
              className="h-8 px-3 shrink-0"
            >
              Cancel
            </Button>
          ) : (
            <Button
              size="sm"
              onClick={handleSubmit}
              disabled={!textInput.trim()}
              className="h-8 px-3 shrink-0"
            >
              Find
            </Button>
          )}
        </div>
      )}

      {toolMode === "click" && !creationDisabled && (
        <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
          <span className="flex items-center gap-1">
            <span className="inline-block h-2.5 w-2.5 rounded-full bg-green-500" />
            Left = include
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block h-2.5 w-2.5 rounded-full bg-red-500" />
            Right = exclude
          </span>
        </div>
      )}
    </div>
  );
}
