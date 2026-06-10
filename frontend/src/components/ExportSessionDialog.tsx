import { useState } from "react";
import { exportSession } from "../api.ts";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { PackageOpen, Loader2 } from "lucide-react";

interface Props {
  sessionId: string | null;
  videoName: string;
  open: boolean;
  onOpen: () => void;
  onClose: () => void;
  onStatus: (msg: string) => void;
}

export default function ExportSessionDialog({
  sessionId,
  videoName,
  open,
  onOpen,
  onClose,
  onStatus,
}: Props) {
  const [exporting, setExporting] = useState(false);
  const [includeVideo, setIncludeVideo] = useState(false);

  async function handleExport() {
    if (!sessionId) return;
    setExporting(true);
    onStatus("Exporting session...");

    try {
      const blob = await exportSession(sessionId, includeVideo);

      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      const stem = videoName.replace(/\.[^.]+$/, "") || "session";
      a.download = `${stem}_session.zip`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      onStatus("Session export complete");
      onClose();
    } catch (err) {
      onStatus(`Export error: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setExporting(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={(v) => (v ? onOpen() : onClose())}>
      <DialogTrigger asChild>
        <Button variant="outline" className="h-9 w-full gap-2 text-xs font-medium">
          <PackageOpen className="h-3.5 w-3.5" />
          Export Session
        </Button>
      </DialogTrigger>

      <DialogContent
        className="sm:max-w-sm"
        showCloseButton={false}
        onInteractOutside={(e) => e.preventDefault()}
        onEscapeKeyDown={(e) => e.preventDefault()}
      >
        <DialogHeader>
          <DialogTitle className="text-sm">Export Session</DialogTitle>
          <DialogDescription className="text-xs">
            Export this session as a portable bundle that can be imported on another machine.
          </DialogDescription>
        </DialogHeader>

        <div className="flex items-start gap-2 py-2">
          <input
            type="checkbox"
            id="include-video"
            checked={includeVideo}
            onChange={(e) => setIncludeVideo(e.target.checked)}
            disabled={exporting}
            className="mt-0.5 h-4 w-4 rounded border-border accent-primary"
          />
          <div className="flex flex-col gap-0.5">
            <Label htmlFor="include-video" className="text-xs font-medium cursor-pointer">
              Include original video file
            </Label>
            <p className="text-[11px] text-muted-foreground">
              Frames are always included. The video increases file size but is not required.
            </p>
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" size="sm" onClick={onClose} disabled={exporting}>
            Cancel
          </Button>
          <Button size="sm" onClick={handleExport} disabled={exporting} className="gap-1.5">
            {exporting && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
            {exporting ? "Exporting..." : "Export"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
