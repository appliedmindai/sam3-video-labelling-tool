import { useEffect, useRef, useState } from "react";
import { X } from "lucide-react";

interface Props {
  message: string;
  autoHideMs?: number;
}

export default function StatusToast({ message, autoHideMs = 3000 }: Props) {
  const [visible, setVisible] = useState(false);
  const [displayMessage, setDisplayMessage] = useState(message);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    // Don't show "Ready" as a toast
    if (!message || message === "Ready") {
      setVisible(false);
      return;
    }

    setDisplayMessage(message);
    setVisible(true);

    // Clear any existing timer
    if (timerRef.current) clearTimeout(timerRef.current);

    // Auto-hide after delay (but not for ongoing states like "Segmenting..." or errors)
    const isOngoing = message.includes("...") || message.toLowerCase().includes("error");
    if (!isOngoing) {
      timerRef.current = setTimeout(() => setVisible(false), autoHideMs);
    }

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [message, autoHideMs]);

  if (!visible) return null;

  const isError = displayMessage.toLowerCase().includes("error");

  return (
    <div className="pointer-events-none fixed inset-x-0 top-4 z-50 flex justify-center">
      <div
        className={`pointer-events-auto flex items-center gap-2 rounded-full border px-4 py-1.5 backdrop-blur-xl transition-opacity ${
          isError
            ? "border-destructive/30 bg-destructive/15 text-destructive"
            : "border-border/50 bg-background/70 text-foreground"
        }`}
      >
        <span className="text-xs font-medium">{displayMessage}</span>
        <button
          className="rounded-full p-0.5 transition-colors hover:bg-foreground/10"
          onClick={() => setVisible(false)}
        >
          <X className="h-3 w-3" />
        </button>
      </div>
    </div>
  );
}
