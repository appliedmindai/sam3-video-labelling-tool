import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { ChevronDown, ChevronRight } from "lucide-react";

interface Props {
  open: boolean;
  title: string;
  message: string;
  traceback?: string;
  onDismiss: () => void;
}

export default function ErrorDialog({
  open,
  title,
  message,
  traceback,
  onDismiss,
}: Props) {
  const [showDetails, setShowDetails] = useState(false);

  return (
    <Dialog open={open} onOpenChange={(v) => { if (!v) onDismiss(); }}>
      <DialogContent className="max-w-lg">
        <DialogHeader>
          <DialogTitle className="text-destructive">{title}</DialogTitle>
          <DialogDescription>{message}</DialogDescription>
        </DialogHeader>

        {traceback && (
          <div>
            <button
              type="button"
              onClick={() => setShowDetails(!showDetails)}
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              {showDetails ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
              {showDetails ? "Hide details" : "Show details"}
            </button>
            {showDetails && (
              <pre className="mt-2 max-h-64 overflow-auto rounded bg-muted p-3 text-xs leading-relaxed">
                {traceback}
              </pre>
            )}
          </div>
        )}

        <DialogFooter>
          <Button variant="outline" onClick={onDismiss}>
            Dismiss
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
