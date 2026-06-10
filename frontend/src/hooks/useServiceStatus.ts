import { useCallback, useEffect, useRef, useState } from "react";
import { getServiceStatus } from "../api";
import type { ServiceStatus } from "../types";

/**
 * Polls GET /api/status with variable intervals:
 * - 1s when phase is "extracting" or "initializing" (active pipeline)
 * - 30s when phase is "ready" (session-loss detection)
 * - No polling when "idle" or "error" (user action required)
 *
 * Returns null until the first fetch completes.
 */
export function useServiceStatus(): {
  status: ServiceStatus | null;
  refetch: () => void;
} {
  const [status, setStatus] = useState<ServiceStatus | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const fetchStatus = useCallback(async () => {
    try {
      const s = await getServiceStatus();
      setStatus(s);
    } catch {
      // Silently ignore — service may be starting up
    }
  }, []);

  // Initial fetch on mount
  useEffect(() => {
    fetchStatus();
  }, [fetchStatus]);

  // Set up polling with variable interval based on phase
  useEffect(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    if (!status) return;

    let intervalMs: number;
    if (status.phase === "extracting" || status.phase === "initializing") {
      intervalMs = 1000;
    } else if (status.phase === "ready") {
      intervalMs = 30_000;
    } else {
      // idle or error — no polling
      return;
    }

    intervalRef.current = setInterval(fetchStatus, intervalMs);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [status?.phase, fetchStatus]);

  return { status, refetch: fetchStatus };
}
