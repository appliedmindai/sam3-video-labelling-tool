import { useState } from "react";
import type { BboxPadding } from "../types.ts";
import { exportCoco } from "../api.ts";
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
import { Download, Loader2 } from "lucide-react";

interface Props {
  sessionId: string | null;
  videoName: string;
  open: boolean;
  onOpen: () => void;
  onClose: () => void;
  onStatus: (msg: string) => void;
  /** Per-object per-frame padding keyframes: objId → frameIdx → padding */
  bboxPadding?: Record<number, Record<number, BboxPadding>>;
}

export default function ExportDialog({
  sessionId,
  videoName,
  open,
  onOpen,
  onClose,
  onStatus,
  bboxPadding = {},
}: Props) {
  const [exporting, setExporting] = useState(false);

  async function handleExport() {
    if (!sessionId) return;
    setExporting(true);
    onStatus("Exporting...");

    try {
      const blob = await exportCoco(sessionId, bboxPadding);

      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      const stem = videoName.replace(/\.[^.]+$/, "") || "export";
      a.download = `${stem}_coco.zip`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);

      onStatus("Export complete");
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
        <Button className="h-9 w-full gap-2 text-xs font-medium">
          <Download className="h-3.5 w-3.5" />
          Export Annotations
        </Button>
      </DialogTrigger>

      <DialogContent
        className="sm:max-w-sm"
        showCloseButton={false}
        onInteractOutside={(e) => e.preventDefault()}
        onEscapeKeyDown={(e) => e.preventDefault()}
      >
        <DialogHeader>
          <DialogTitle className="text-sm">Export Annotations</DialogTitle>
          <DialogDescription className="text-xs">
            Export annotated frames as COCO JSON with bboxes and segmentation polygons.
          </DialogDescription>
        </DialogHeader>

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
