import { useState } from "react";
import { Loader2, Lock } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { setAuthToken, verifyPassword } from "../api";

interface PasswordGateProps {
  onUnlocked: () => void;
}

/**
 * Full-screen password prompt shown when the backend returns 401
 * (cloud deploys set AUTH_PASSWORD). Verifies the candidate against
 * /api/health, persists it via setAuthToken, then unlocks the app.
 */
export function PasswordGate({ onUnlocked }: PasswordGateProps) {
  const [value, setValue] = useState("");
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState(false);

  const submit = async () => {
    const candidate = value.trim();
    if (!candidate || checking) return;
    setChecking(true);
    setError(false);
    const ok = await verifyPassword(candidate);
    setChecking(false);
    if (ok) {
      setAuthToken(candidate);
      onUnlocked();
    } else {
      setError(true);
    }
  };

  return (
    <div className="flex h-screen items-center justify-center bg-background">
      <div className="w-80 rounded-lg border border-border bg-card p-6">
        <div className="mb-4 flex items-center gap-2">
          <Lock className="h-4 w-4 text-muted-foreground" />
          <h1 className="text-sm font-medium text-foreground">
            This deployment is password-protected
          </h1>
        </div>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void submit();
          }}
          className="space-y-3"
        >
          <Input
            type="password"
            placeholder="three-word-password"
            value={value}
            onChange={(e) => setValue(e.target.value)}
            autoFocus
          />
          {error && (
            <p className="text-xs text-destructive">
              Wrong password. It was printed by the deploy script
              (`./deploy/deploy.sh password` recovers it).
            </p>
          )}
          <Button type="submit" className="w-full" disabled={checking || !value.trim()}>
            {checking ? <Loader2 className="h-4 w-4 animate-spin" /> : "Unlock"}
          </Button>
        </form>
      </div>
    </div>
  );
}
