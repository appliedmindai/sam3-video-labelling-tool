# Shared-Password Auth + Deploy Tooling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a shared three-word password gate to the cloud deployment (Flask header check + frontend PasswordGate) and a full lifecycle deploy script (`deploy/deploy.sh up/down/status/logs/password/smoke`) with state tracking, hash-based build caching, and agent-facing docs.

**Architecture:** Auth is a `before_request` hook in `backend/app/__init__.py` activated only when the `AUTH_PASSWORD` env var is set; the frontend funnels every request through a new `apiFetch()` wrapper in `frontend/src/api.ts` that injects `X-Auth-Token` and triggers a `PasswordGate` screen on 401. The deploy script is a single commented bash file with python3 helpers for JSON state, content hashing, and password generation; it records created resources in a gitignored state file so `down` deletes only what it created.

**Tech Stack:** Flask (Python 3.11), React 18 + TypeScript + Vite, bash + python3 + gcloud/gsutil, Cloud Run (L4 GPU), GCS, Cloud Build.

**Spec:** `docs/superpowers/specs/2026-06-11-shared-password-auth-design.md`

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `backend/app/__init__.py` | Modify | Auth `before_request` hook; jsonify import; health comment |
| `backend/tests/test_auth.py` | Create | Auth matrix tests |
| `frontend/src/api.ts` | Modify | Token state, `setAuthToken`, `onUnauthorized`, `apiFetch`, `verifyPassword`; all fetches via `apiFetch`; XHR headers |
| `frontend/src/components/PasswordGate.tsx` | Create | Password entry screen |
| `frontend/src/App.tsx` | Modify | `authRequired` state, gate render, callback registration |
| `cloudbuild-native.yaml` | Modify | `_CACHE_IMAGE` substitution so hash-tagged builds keep layer caching |
| `deploy/deploy.sh` | Create | Full lifecycle CLI |
| `.gitignore` | Modify | Ignore `deploy/.deploy-state.json` |
| `DEPLOY.md` | Create | Full deploy reference (humans + agents) |
| `deploy/README.md` | Create | ~1–2k-token agent quick-reference |
| `docs/deploy-cloud-run.md` | Replace | Thin pointer to DEPLOY.md |
| `USER-FLOWS.md` | Modify | UF-1.7 row, N11 invariant, impact-map rows, smoke set update |

Conda env for backend tests: `eval "$(/opt/homebrew/bin/conda shell.zsh hook)" && conda activate sam2-annotator`. Always `python3`, never `python`.

---

### Task 1: Backend auth hook (TDD)

**Files:**
- Test: `backend/tests/test_auth.py`
- Modify: `backend/app/__init__.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_auth.py`:

```python
"""Shared-password auth: when AUTH_PASSWORD is set, every request must
carry a matching X-Auth-Token header — including /api/health, because the
frontend heartbeat keeps the billed GPU container alive. When unset
(local dev), everything stays open."""

import pytest

from app import create_app
from app.services.pipeline import ServiceState

PASSWORD = "crimson-otter-lantern"


def make_client(monkeypatch, password):
    if password is None:
        monkeypatch.delenv("AUTH_PASSWORD", raising=False)
    else:
        monkeypatch.setenv("AUTH_PASSWORD", password)
    app = create_app()
    app.config["TESTING"] = True

    # Reset SAM3Service singleton state (may be dirty from other tests)
    from app.services.sam3_service import SAM3Service
    sam = SAM3Service()
    with sam._state_lock:
        sam._service_state = ServiceState()

    return app.test_client()


class TestAuthEnabled:
    def test_missing_header_is_401(self, monkeypatch):
        client = make_client(monkeypatch, PASSWORD)
        resp = client.get("/api/health")
        assert resp.status_code == 401
        assert resp.get_json() == {"error": "unauthorized"}

    def test_wrong_password_is_401(self, monkeypatch):
        client = make_client(monkeypatch, PASSWORD)
        resp = client.get("/api/health", headers={"X-Auth-Token": "wrong-guess-here"})
        assert resp.status_code == 401

    def test_correct_password_is_200(self, monkeypatch):
        client = make_client(monkeypatch, PASSWORD)
        resp = client.get("/api/health", headers={"X-Auth-Token": PASSWORD})
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True

    def test_status_requires_auth_too(self, monkeypatch):
        client = make_client(monkeypatch, PASSWORD)
        assert client.get("/api/status").status_code == 401
        resp = client.get("/api/status", headers={"X-Auth-Token": PASSWORD})
        assert resp.status_code == 200

    def test_options_preflight_passes_without_token(self, monkeypatch):
        # CORS preflights never carry custom headers; they must not 401.
        client = make_client(monkeypatch, PASSWORD)
        resp = client.options("/api/status")
        assert resp.status_code != 401


class TestAuthDisabled:
    def test_no_password_means_open(self, monkeypatch):
        client = make_client(monkeypatch, None)
        assert client.get("/api/health").status_code == 200
        assert client.get("/api/status").status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && python3 -m pytest tests/test_auth.py -v`
Expected: `TestAuthEnabled` tests FAIL (200 where 401 expected); `TestAuthDisabled` passes.

- [ ] **Step 3: Implement the hook**

In `backend/app/__init__.py`:

(a) Change the top-level Flask import (line 5):

```python
from flask import Flask, g, jsonify, request
```

(b) Insert the auth block in `create_app()` immediately after `app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024` and before the `# Cloud mode stores sessions...` comment:

```python
    # Shared-password auth for cloud deploys. When AUTH_PASSWORD is set
    # (deploy/deploy.sh sets it on the Cloud Run service), every request —
    # including /api/health, whose heartbeat keeps the billed GPU container
    # alive — must carry a matching X-Auth-Token header. Unset → no-op
    # (local dev, tests). OPTIONS is exempt: CORS preflights never carry
    # custom headers.
    auth_password = os.environ.get("AUTH_PASSWORD", "")
    if auth_password:
        import hmac

        @app.before_request
        def _check_auth():
            if request.method == "OPTIONS":
                return None
            token = request.headers.get("X-Auth-Token", "")
            if not hmac.compare_digest(token, auth_password):
                return jsonify({"error": "unauthorized"}), 401
```

(c) The health endpoint block currently reads:

```python
    # Health endpoint — no auth, no SAM3 lock, always responds instantly.
    # Used by frontend heartbeat for keepalive + boot detection.
    from flask import jsonify
    from app.config import BOOT_ID, BOOT_TIME
```

Replace with (comment updated, redundant inner import dropped):

```python
    # Health endpoint — no SAM3 lock, always responds instantly. Used by
    # the frontend heartbeat for keepalive + boot detection, and by the
    # PasswordGate to verify a candidate password. Requires X-Auth-Token
    # like everything else when AUTH_PASSWORD is set.
    from app.config import BOOT_ID, BOOT_TIME
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && python3 -m pytest tests/test_auth.py -v`
Expected: all 6 PASS.

- [ ] **Step 5: Run the full backend suite (auth must not break existing tests — they run without AUTH_PASSWORD, so the hook is a no-op)**

Run: `cd backend && python3 -m pytest tests/ -q`
Expected: all pass, same count as before plus 6.

If any existing test fails because `AUTH_PASSWORD` leaks from the environment, the fix is in the test run, not the code: ensure `make_client` is the only place setting it (monkeypatch auto-undoes per test).

- [ ] **Step 6: Commit**

```bash
git add backend/tests/test_auth.py backend/app/__init__.py
git commit -m "feat(backend): shared-password auth via X-Auth-Token when AUTH_PASSWORD is set"
```

---

### Task 2: Frontend token plumbing in api.ts

**Files:**
- Modify: `frontend/src/api.ts`

No frontend test suite — verification is `npx tsc --noEmit` plus grep checks.

- [ ] **Step 1: Add the auth module block**

In `frontend/src/api.ts`, immediately after `const BASE = ...` (line 11), insert:

```typescript
// ---- Shared-password auth (cloud deploys) -------------------------------
// Every request carries X-Auth-Token when a token is known. On any 401 the
// registered callback fires and App.tsx shows the PasswordGate. In local
// dev the backend never 401s, so none of this activates.

const AUTH_STORAGE_KEY = "sam3-auth-token";

let authToken: string | null = localStorage.getItem(AUTH_STORAGE_KEY);
let unauthorizedCallback: (() => void) | null = null;

export function setAuthToken(token: string): void {
  authToken = token;
  localStorage.setItem(AUTH_STORAGE_KEY, token);
}

export function onUnauthorized(callback: () => void): void {
  unauthorizedCallback = callback;
}

/** fetch() with the auth header injected and 401 detection. */
async function apiFetch(url: string, init?: RequestInit): Promise<Response> {
  const headers: Record<string, string> = {
    ...(authToken ? { "X-Auth-Token": authToken } : {}),
    ...((init?.headers as Record<string, string> | undefined) ?? {}),
  };
  const res = await fetch(url, { ...init, headers });
  if (res.status === 401) unauthorizedCallback?.();
  return res;
}

/** Check a candidate password against the backend without storing it. */
export async function verifyPassword(candidate: string): Promise<boolean> {
  try {
    const res = await fetch(`${BASE}/health`, {
      headers: { "X-Auth-Token": candidate },
    });
    return res.ok;
  } catch {
    return false;
  }
}
```

- [ ] **Step 2: Route every fetch through apiFetch**

Replace every remaining `fetch(` call in `api.ts` with `apiFetch(` — **except** the two inside `apiFetch` and `verifyPassword` themselves. This includes:

- every `const res = await fetch(...)` / `const resp = await fetch(...)` call,
- both lines in `fetchFrameBlob` (`let res = await fetch(url);` and the retry `res = await fetch(url);`),
- the fire-and-forget keepalive call in `flushSyncBeacon`:

```typescript
  apiFetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
    keepalive: true,
  }).catch(() => {});
```

- [ ] **Step 3: Add the header to both XHR uploads**

In `uploadVideo`, after `xhr.open("POST", `${BASE}/video/upload`);` add:

```typescript
    if (authToken) xhr.setRequestHeader("X-Auth-Token", authToken);
```

and in its `xhr.onload` failure branch, before `reject`:

```typescript
        if (xhr.status === 401) unauthorizedCallback?.();
```

Apply the same two changes to `importSession` (after `xhr.open("POST", `${BASE}/export/import-session`);` and in its failure branch).

- [ ] **Step 4: Update stale comments**

- `checkHealth` doc comment: change `/** Lightweight health check — no auth, no SAM3 lock. */` to `/** Lightweight health check — no SAM3 lock; auth handled by apiFetch. */`
- `getServiceStatus` doc comment: change `/** Full service status — no auth needed. Returns phase, progress, session info. */` to `/** Full service status. Returns phase, progress, session info. */`

- [ ] **Step 5: Verify no bare fetch remains and types check**

```bash
cd frontend
grep -n "fetch(" src/api.ts | grep -v "apiFetch(" 
```
Expected: exactly 2 hits — the `fetch(` inside `apiFetch` and the one inside `verifyPassword`.

```bash
npx tsc --noEmit
```
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/api.ts
git commit -m "feat(frontend): inject X-Auth-Token on all API traffic via apiFetch"
```

---

### Task 3: PasswordGate component + App wiring

**Files:**
- Create: `frontend/src/components/PasswordGate.tsx`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Create the component**

Create `frontend/src/components/PasswordGate.tsx`:

```tsx
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
```

- [ ] **Step 2: Wire into App.tsx**

(a) Add `onUnauthorized` to the existing `import { ... } from "./api"` list (App.tsx imports many api functions near the top — add the name alphabetically into that list).

(b) Add the component import next to the other component imports:

```tsx
import { PasswordGate } from "./components/PasswordGate";
```

(c) Inside `function App()`, right after the `useServiceStatus()` line (`const { status: serviceStatus, refetch: refetchStatus } = useServiceStatus();`), add:

```tsx
  const [authRequired, setAuthRequired] = useState(false);

  // 401 from any API call (cloud deploys) → show the password gate.
  useEffect(() => {
    onUnauthorized(() => setAuthRequired(true));
  }, []);
```

(d) Immediately above App's final `return (` (the top-level render, currently line ~1936), add:

```tsx
  if (authRequired) {
    return (
      <PasswordGate
        onUnlocked={() => {
          setAuthRequired(false);
          refetchStatus();
        }}
      />
    );
  }
```

`refetchStatus()` matters: `useServiceStatus` does its initial fetch on mount; if that got a 401, `status` is null and polling never started — the explicit refetch restarts the status flow after unlock.

NOTE (React rules): this early return must sit BELOW every hook call in App. Placing it directly above the final `return (` guarantees that.

- [ ] **Step 3: Type check + production build check**

```bash
cd frontend && npx tsc --noEmit && npm run build
```
Expected: both succeed (`npm run build` runs `tsc -b && vite build`, the Docker build's exact check).

- [ ] **Step 4: Manual local verification**

```bash
cd <repo-root> && AUTH_PASSWORD=test-pass-word make dev
```
In the browser (http://localhost:5173): the PasswordGate appears (the initial /api/status poll 401s). Enter `wrong` → inline error. Enter `test-pass-word` → app proceeds to the session list. Reload the page → no gate (localStorage). Then stop, run plain `make dev` → no gate. Clear the stored token afterwards in devtools (`localStorage.removeItem("sam3-auth-token")`) to avoid confusion later.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/PasswordGate.tsx frontend/src/App.tsx
git commit -m "feat(frontend): PasswordGate screen on 401 with localStorage persistence"
```

---

### Task 4: cloudbuild-native.yaml — cache image substitution

**Files:**
- Modify: `cloudbuild-native.yaml`

The deploy script tags images by source hash (`sam3-native:<hash>`). The current cloudbuild file pulls `${_IMAGE}` for layer cache — a brand-new hash tag never exists, so every build would be a cold ~20 min build. Fix: pull/tag a stable `_CACHE_IMAGE` (`:latest`) alongside the hash tag.

- [ ] **Step 1: Update the build steps**

Replace the `docker pull`, `docker build`, and `docker push` steps and the `images:`/`substitutions:` sections with:

```yaml
  # Pull the rolling cache tag (:latest) for Docker layer cache. The build
  # itself is tagged by source hash, so the hash tag never pre-exists;
  # :latest always points at the previous successful build. For code-only
  # changes this reuses the heavy layers (PyTorch, SAM3, Flash Attention,
  # model weights) and cuts build time from ~20 min to ~2-3 min. Allowed
  # to fail on first build or after image prune.
  - name: 'gcr.io/cloud-builders/docker'
    entrypoint: 'bash'
    args:
      - '-c'
      - 'docker pull ${_CACHE_IMAGE} || true'

  - name: 'gcr.io/cloud-builders/docker'
    args:
      - 'build'
      - '--cache-from'
      - '${_CACHE_IMAGE}'
      - '-f'
      - 'Dockerfile.native'
      - '-t'
      - '${_IMAGE}'
      - '-t'
      - '${_CACHE_IMAGE}'
      - '.'

  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '${_IMAGE}']

  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', '${_CACHE_IMAGE}']

images:
  - '${_IMAGE}'
  - '${_CACHE_IMAGE}'
```

and at the bottom:

```yaml
# Override these per-project (deploy/deploy.sh does this automatically):
#   gcloud builds submit --config=cloudbuild-native.yaml \
#     --substitutions=_IMAGE=...:SRC_HASH,_CACHE_IMAGE=...:latest,_MODELS_BUCKET=YOUR-MODELS-BUCKET
substitutions:
  _IMAGE: 'us-central1-docker.pkg.dev/YOUR_PROJECT/cloud-run-images/sam3-native:latest'
  _CACHE_IMAGE: 'us-central1-docker.pkg.dev/YOUR_PROJECT/cloud-run-images/sam3-native:latest'
  _MODELS_BUCKET: 'YOUR-MODELS-BUCKET'
```

The first `gsutil` weights step is unchanged.

- [ ] **Step 2: Commit**

```bash
git add cloudbuild-native.yaml
git commit -m "build: tag images by source hash with :latest as rolling cache tag"
```

---

### Task 5: deploy/deploy.sh

**Files:**
- Create: `deploy/deploy.sh` (mode 755)
- Modify: `.gitignore`

- [ ] **Step 1: Write the script**

Create `deploy/deploy.sh` with the complete content below, then `chmod +x deploy/deploy.sh`:

```bash
#!/usr/bin/env bash
#
# deploy.sh — provision, deploy, inspect, and tear down the SAM3 Video
# Labelling Tool on Google Cloud Run (L4 GPU) + GCS.
#
# Usage:
#   ./deploy/deploy.sh [command] [flags]
#
# Commands:
#   up                 (default) idempotent provision + deploy. Safe to
#                      re-run: every step checks current state and skips
#                      work that is already done.
#   status             show configured + live deployment state
#   logs [-f]          read recent Cloud Run logs (-f tails; needs gcloud beta)
#   password           print the current shared password (from service env)
#   generate-password  print a fresh three-word password (no GCP calls)
#   smoke              authenticated smoke checks against the deployed URL
#   down [--purge]     delete resources this script created (state-tracked)
#
# Flags:
#   --yes / -y         skip confirmation prompts. REQUIRED for `down` in a
#                      non-interactive shell.
#   --rotate-password  generate a new password on this `up`
#   --purge            with `down`: also delete the GCS buckets (annotation
#                      sessions + model weights) after an extra confirmation
#
# Configuration (env vars; only PROJECT is needed on the first run — every
# value is remembered in deploy/.deploy-state.json afterwards):
#   PROJECT          GCP project id (falls back to `gcloud config` default)
#   REGION           Cloud Run region          (default: us-east4 — has L4s)
#   SERVICE          Cloud Run service name    (default: sam3-annotator)
#   REPO_NAME        Artifact Registry repo    (default: cloud-run-images)
#   MODELS_BUCKET    SAM3 weights bucket       (default: $PROJECT-sam3-models)
#   SESSIONS_BUCKET  annotation data bucket    (default: $PROJECT-sam3-sessions)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
STATE_FILE="$SCRIPT_DIR/.deploy-state.json"

# ---------------------------------------------------------------- logging --

say()  { echo "[deploy] $*"; }
warn() { echo "[deploy] WARNING: $*" >&2; }
die()  { echo "[deploy] ERROR: $*" >&2; exit 1; }

# ------------------------------------------------------------ state file --
# .deploy-state.json is a flat string->string map. Keys:
#   project / region / service / repo_name / models_bucket / sessions_bucket
#     — remembered config so later commands need no env vars
#   created_registry / created_sa / created_service /
#   created_models_bucket / created_sessions_bucket / uploaded_weights
#     — "true" when THIS script created the resource. `down` only deletes
#       resources marked created; pre-existing ones are never touched.
#   last_image / last_deploy_at — informational, shown by `status`.

state_get() { # state_get KEY -> value (empty if missing)
  [ -f "$STATE_FILE" ] || { echo ""; return; }
  python3 - "$1" "$STATE_FILE" <<'PYEOF'
import json, sys
key, path = sys.argv[1], sys.argv[2]
try:
    data = json.load(open(path))
except Exception:
    data = {}
print(data.get(key, ""))
PYEOF
}

state_set() { # state_set KEY VALUE
  python3 - "$1" "$2" "$STATE_FILE" <<'PYEOF'
import json, os, sys
key, value, path = sys.argv[1], sys.argv[2], sys.argv[3]
data = {}
if os.path.exists(path):
    try:
        data = json.load(open(path))
    except Exception:
        data = {}
data[key] = value
with open(path, "w") as f:
    json.dump(data, f, indent=2, sort_keys=True)
    f.write("\n")
PYEOF
}

# ------------------------------------------------------------ prompts -----
# Repo convention: prompts must be TTY-guarded. Non-interactive `up` runs
# auto-proceed with a log line (never a silent exit 0). Destructive
# commands (`down`) REFUSE to run non-interactively without --yes.

confirm() { # confirm "question"
  if [ "$YES" = "true" ]; then return 0; fi
  if [ -t 0 ]; then
    read -r -p "[deploy] $1 (y/N) " reply
    [[ "$reply" =~ ^[Yy]$ ]] || { say "Aborted."; exit 1; }
  else
    say "non-interactive shell — auto-proceeding (stdin is not a TTY)"
  fi
}

confirm_destructive() { # confirm_destructive "question"
  if [ "$YES" = "true" ]; then return 0; fi
  if [ -t 0 ]; then
    read -r -p "[deploy] $1 (y/N) " reply
    [[ "$reply" =~ ^[Yy]$ ]] || { say "Aborted."; exit 1; }
  else
    die "destructive command in a non-interactive shell requires --yes"
  fi
}

# ------------------------------------------------------------- config -----
# Resolution order: env var > state file > default. The first `up`
# persists everything, so day-2 commands are just `./deploy/deploy.sh logs`.

resolve_config() {
  PROJECT="${PROJECT:-$(state_get project)}"
  if [ -z "$PROJECT" ]; then
    PROJECT="$(gcloud config get-value project 2>/dev/null || true)"
    [ -n "$PROJECT" ] && warn "PROJECT not set — using gcloud default '$PROJECT'"
  fi
  [ -n "$PROJECT" ] || die "set PROJECT=<gcp-project-id> (first run only — it is then remembered in $STATE_FILE)"

  REGION="${REGION:-$(state_get region)}";                 REGION="${REGION:-us-east4}"
  SERVICE="${SERVICE:-$(state_get service)}";              SERVICE="${SERVICE:-sam3-annotator}"
  REPO_NAME="${REPO_NAME:-$(state_get repo_name)}";        REPO_NAME="${REPO_NAME:-cloud-run-images}"
  MODELS_BUCKET="${MODELS_BUCKET:-$(state_get models_bucket)}"
  MODELS_BUCKET="${MODELS_BUCKET:-$PROJECT-sam3-models}"
  SESSIONS_BUCKET="${SESSIONS_BUCKET:-$(state_get sessions_bucket)}"
  SESSIONS_BUCKET="${SESSIONS_BUCKET:-$PROJECT-sam3-sessions}"

  SA_NAME="sam3-annotator"
  SA_EMAIL="$SA_NAME@$PROJECT.iam.gserviceaccount.com"
  CACHE_IMAGE="$REGION-docker.pkg.dev/$PROJECT/$REPO_NAME/sam3-native:latest"
}

persist_config() {
  state_set project "$PROJECT"
  state_set region "$REGION"
  state_set service "$SERVICE"
  state_set repo_name "$REPO_NAME"
  state_set models_bucket "$MODELS_BUCKET"
  state_set sessions_bucket "$SESSIONS_BUCKET"
}

# ----------------------------------------------------------- password -----
# Three words from a ~250-word list ≈ 24 bits of entropy — right-sized for
# "keep strangers off my GPU", not bank-grade. secrets.choice is CSPRNG.

generate_password() {
  python3 - <<'PYEOF'
import secrets
WORDS = sorted(set("""
otter fox wolf bear hawk crane heron finch robin wren lynx hare moose bison seal whale
crab squid gecko viper cobra eagle falcon raven crow dove swan goose duck owl ibis stork
salmon trout perch bass pike carp shark manta orca walrus badger weasel marten stoat ferret mole
red blue green amber coral teal cyan navy plum rose gold silver copper bronze ivory pearl
crimson scarlet maroon violet indigo lilac mauve olive lime mint sage jade ruby topaz onyx opal
river lake pond creek brook delta marsh dune mesa butte cliff ridge peak summit valley glen
forest grove birch cedar maple oak pine aspen willow alder elm fir spruce laurel hazel rowan
storm cloud frost snow hail sleet mist fog dew rain wind gale breeze thunder ember spark
lantern anchor compass kettle hammer chisel ladder bucket barrel basket candle mirror saddle paddle anvil crate
button ribbon needle spool loom quill scroll ledger atlas globe prism lens gear pulley lever wheel
apple pear peach mango grape lemon melon berry fig date honey clove basil thyme ginger pepper
drum flute cello banjo chord tempo waltz polka opera lyric verse hymn organ viola reed bell
comet nova lunar solar orbit nebula quasar meteor zenith aurora eclipse cosmos saturn vega rigel astro
tundra steppe prairie canyon fjord atoll lagoon harbor jetty quay wharf isle reef shoal cove bay
baker mason smith potter weaver tailor scribe ranger pilot sailor scout knight squire bard monk friar
pebble boulder quartz flint slate marble granite basalt coal iron zinc cobalt nickel chrome cliffs dunes
""".split()))
print("-".join(secrets.choice(WORDS) for _ in range(3)))
PYEOF
}

# ---------------------------------------------------------- source hash ---
# The image tag is the hash of every tracked file that goes into the build.
# Same source → same tag → Cloud Build is skipped entirely. Working-tree
# content is hashed (uncommitted edits count); brand-new untracked files
# don't change the hash until `git add` (acceptable: real deploys flow
# through git).

compute_source_hash() {
  ( cd "$REPO_ROOT" &&
    git ls-files backend frontend Dockerfile.native .dockerignore \
        deploy/nginx.conf deploy/entrypoint.sh cloudbuild-native.yaml \
      | LC_ALL=C sort \
      | tr '\n' '\0' \
      | xargs -0 shasum -a 256 \
      | shasum -a 256 | cut -c1-12 )
}

# -------------------------------------------------------- live queries ----

service_exists() {
  gcloud run services describe "$SERVICE" --region "$REGION" \
    --project "$PROJECT" >/dev/null 2>&1
}

service_url() {
  gcloud run services describe "$SERVICE" --region "$REGION" \
    --project "$PROJECT" --format='value(status.url)' 2>/dev/null || true
}

deployed_image() {
  gcloud run services describe "$SERVICE" --region "$REGION" \
    --project "$PROJECT" \
    --format='value(spec.template.spec.containers[0].image)' 2>/dev/null || true
}

get_deployed_password() {
  gcloud run services describe "$SERVICE" --region "$REGION" \
    --project "$PROJECT" --format=json 2>/dev/null | python3 -c '
import json, sys
try:
    svc = json.load(sys.stdin)
    envs = svc["spec"]["template"]["spec"]["containers"][0].get("env", [])
    print(next((e.get("value", "") for e in envs if e.get("name") == "AUTH_PASSWORD"), ""))
except Exception:
    print("")
'
}

# ------------------------------------------------------------ up steps ----

step_preflight() {
  say "1/8 preflight"
  command -v gcloud >/dev/null || die "gcloud CLI not found — install the Google Cloud SDK"
  command -v gsutil >/dev/null || die "gsutil not found — install the Google Cloud SDK"
  command -v python3 >/dev/null || die "python3 not found"
  command -v git >/dev/null || die "git not found"

  local account
  account=$(gcloud auth list --filter=status:ACTIVE --format='value(account)' 2>/dev/null || true)
  [ -n "$account" ] || die "no active gcloud account — run: gcloud auth login"
  say "  account: $account"
  say "  project: $PROJECT  region: $REGION  service: $SERVICE"

  gcloud projects describe "$PROJECT" >/dev/null 2>&1 \
    || die "project '$PROJECT' not found or no access"

  # Billing must be on or API enablement / Cloud Build will fail with an
  # opaque error later. Check it now with a clear message instead.
  local billing
  billing=$(gcloud billing projects describe "$PROJECT" \
    --format='value(billingEnabled)' 2>/dev/null || echo "unknown")
  if [ "$billing" = "False" ]; then
    die "billing is not enabled on '$PROJECT' — link a billing account first"
  elif [ "$billing" = "unknown" ]; then
    warn "could not verify billing (missing permission?) — continuing"
  fi
}

step_enable_apis() {
  say "2/8 enable APIs"
  local needed="run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com storage.googleapis.com"
  local enabled
  enabled=$(gcloud services list --enabled --project "$PROJECT" \
    --format='value(config.name)')
  local api
  for api in $needed; do
    if grep -qx "$api" <<<"$enabled"; then
      say "  $api already enabled — skip"
    else
      say "  enabling $api"
      gcloud services enable "$api" --project "$PROJECT"
      # Note: `down` never disables APIs — other workloads may rely on them.
    fi
  done
}

step_registry() {
  say "3/8 Artifact Registry repo"
  if gcloud artifacts repositories describe "$REPO_NAME" --location "$REGION" \
       --project "$PROJECT" >/dev/null 2>&1; then
    say "  repo $REPO_NAME exists — skip"
  else
    say "  creating repo $REPO_NAME in $REGION"
    gcloud artifacts repositories create "$REPO_NAME" \
      --repository-format=docker --location "$REGION" --project "$PROJECT"
    state_set created_registry "true"
  fi
}

step_buckets() {
  say "4/8 GCS buckets"
  local name key
  for spec in "$MODELS_BUCKET:created_models_bucket" "$SESSIONS_BUCKET:created_sessions_bucket"; do
    name="${spec%%:*}"; key="${spec##*:}"
    if gsutil ls -b "gs://$name" >/dev/null 2>&1; then
      say "  gs://$name exists — skip"
    else
      say "  creating gs://$name in $REGION"
      gsutil mb -p "$PROJECT" -l "$REGION" "gs://$name"
      state_set "$key" "true"
    fi
  done
}

step_weights() {
  say "5/8 SAM3 model weights"
  if gsutil -q stat "gs://$MODELS_BUCKET/sam3/config.json" 2>/dev/null; then
    say "  weights present in gs://$MODELS_BUCKET/sam3/ — skip (never re-pushes ~7 GB)"
    return
  fi
  # The gated facebook/sam3 model needs a HuggingFace login — the one
  # genuinely manual prerequisite. Download once, upload once; after that
  # this step always skips.
  command -v hf >/dev/null || die "weights missing from gs://$MODELS_BUCKET/sam3/ and the 'hf' CLI is not installed.
  Fix:  pip install -U huggingface_hub
        hf auth login            # account needs access to gated facebook/sam3
        ./deploy/deploy.sh up    # re-run"
  local tmp
  tmp=$(mktemp -d)
  say "  downloading facebook/sam3 (~7 GB) to $tmp"
  hf download facebook/sam3 --local-dir "$tmp/sam3"
  say "  uploading to gs://$MODELS_BUCKET/sam3/"
  gsutil -m cp "$tmp/sam3/"* "gs://$MODELS_BUCKET/sam3/"
  rm -rf "$tmp"
  state_set uploaded_weights "true"
}

step_service_account() {
  say "6/8 service account"
  if gcloud iam service-accounts describe "$SA_EMAIL" --project "$PROJECT" >/dev/null 2>&1; then
    say "  $SA_EMAIL exists — skip"
  else
    say "  creating $SA_EMAIL"
    gcloud iam service-accounts create "$SA_NAME" \
      --display-name="SAM3 Video Labelling Tool" --project "$PROJECT"
    state_set created_sa "true"
  fi
  # Idempotent: re-granting an existing binding is a no-op.
  gsutil iam ch "serviceAccount:$SA_EMAIL:roles/storage.objectAdmin" \
    "gs://$SESSIONS_BUCKET" >/dev/null
  say "  $SA_EMAIL has storage.objectAdmin on gs://$SESSIONS_BUCKET"
}

step_build() {
  say "7/8 container image"
  SRC_HASH=$(compute_source_hash)
  IMAGE="$REGION-docker.pkg.dev/$PROJECT/$REPO_NAME/sam3-native:$SRC_HASH"
  if gcloud artifacts docker images describe "$IMAGE" --project "$PROJECT" >/dev/null 2>&1; then
    say "  image for source hash $SRC_HASH already in registry — skip Cloud Build"
    return
  fi
  say "  building $IMAGE (first build ~15-20 min; cached code-only builds ~3-6 min)"
  gcloud builds submit \
    --config="$REPO_ROOT/cloudbuild-native.yaml" \
    --substitutions=_IMAGE="$IMAGE",_CACHE_IMAGE="$CACHE_IMAGE",_MODELS_BUCKET="$MODELS_BUCKET" \
    --project "$PROJECT" \
    "$REPO_ROOT"
}

step_deploy() {
  say "8/8 deploy to Cloud Run"
  local current_image="" current_password=""
  if service_exists; then
    current_image=$(deployed_image)
    current_password=$(get_deployed_password)
  fi

  # Reuse the live password on redeploys so users/browsers stay logged in;
  # generate fresh only on first deploy or explicit --rotate-password.
  if [ -n "$current_password" ] && [ "$ROTATE_PASSWORD" != "true" ]; then
    PASSWORD="$current_password"
  else
    PASSWORD=$(generate_password)
    [ -n "$current_password" ] && say "  rotating password (old one stops working)"
  fi

  if [ "$current_image" = "$IMAGE" ] && [ "$PASSWORD" = "$current_password" ]; then
    say "  service already runs image :$SRC_HASH with the current password — skip deploy"
    return
  fi

  confirm "Deploy $SERVICE ($SRC_HASH) to $REGION in project $PROJECT?"

  # Flag rationale (full discussion in DEPLOY.md):
  #   --gpu nvidia-l4         SAM3 bfloat16 runs natively (do NOT use T4)
  #   --max-instances 1       stateful service: GPU model + sessions in memory
  #   --concurrency 8         lets frame fetches run during a propagation SSE
  #   --min-instances 0       scale to zero when idle (GCS keeps the data)
  #   --allow-unauthenticated the shared password is the gate, not IAM
  gcloud run deploy "$SERVICE" \
    --image "$IMAGE" \
    --region "$REGION" \
    --project "$PROJECT" \
    --service-account "$SA_EMAIL" \
    --gpu 1 \
    --gpu-type nvidia-l4 \
    --cpu 8 \
    --memory 24Gi \
    --timeout 3600 \
    --concurrency 8 \
    --min-instances 0 \
    --max-instances 1 \
    --no-cpu-throttling \
    --termination-grace-period 30 \
    --port 8080 \
    --allow-unauthenticated \
    --set-env-vars "SEGMENT_MODE=cloud,GCS_BUCKET=$SESSIONS_BUCKET,SAM3_BACKEND=native,AUTH_PASSWORD=$PASSWORD"

  [ -z "$current_image" ] && state_set created_service "true"
  state_set last_image "$IMAGE"
  state_set last_deploy_at "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

cmd_up() {
  resolve_config
  step_preflight
  persist_config
  step_enable_apis
  step_registry
  step_buckets
  step_weights
  step_service_account
  step_build
  step_deploy

  local url
  url=$(service_url)
  echo
  say "=============================================================="
  say " Deployed: $url"
  say " Password: $(get_deployed_password)"
  say "=============================================================="
  say " Open the URL and enter the password when prompted."
  say " Recover it any time:   ./deploy/deploy.sh password"
  say " Smoke-test the API:    ./deploy/deploy.sh smoke"
  say " Re-running 'up' keeps this password (use --rotate-password to change it)."
}

# ------------------------------------------------------- other commands ---

cmd_status() {
  resolve_config
  say "config   project=$PROJECT region=$REGION service=$SERVICE"
  say "buckets  sessions=gs://$SESSIONS_BUCKET models=gs://$MODELS_BUCKET"
  say "state    $STATE_FILE"
  if [ -f "$STATE_FILE" ]; then
    sed 's/^/[deploy]   /' "$STATE_FILE"
  else
    say "  (no state file yet — 'up' has not run)"
  fi
  if service_exists; then
    local img hash
    img=$(deployed_image)
    hash=$(compute_source_hash)
    say "live     url=$(service_url)"
    say "live     image=$img"
    if [[ "$img" == *":$hash" ]]; then
      say "live     source: in sync with working tree (hash $hash)"
    else
      say "live     source: working tree hash is $hash — run 'up' to deploy it"
    fi
  else
    say "live     service not deployed"
  fi
}

cmd_logs() {
  resolve_config
  if [ "$FOLLOW" = "true" ]; then
    say "tailing logs (Ctrl-C to stop)"
    gcloud beta run services logs tail "$SERVICE" --region "$REGION" --project "$PROJECT"
  else
    gcloud run services logs read "$SERVICE" --region "$REGION" --project "$PROJECT" --limit 100
  fi
}

cmd_password() {
  resolve_config
  service_exists || die "service '$SERVICE' is not deployed"
  local pw
  pw=$(get_deployed_password)
  [ -n "$pw" ] || die "service has no AUTH_PASSWORD set (deployed without this script?)"
  echo "$pw"
}

cmd_smoke() {
  resolve_config
  service_exists || die "service '$SERVICE' is not deployed"
  local url pw code
  url=$(service_url)
  pw=$(get_deployed_password)
  say "smoke-testing $url (first request may take ~90s on GPU cold start)"

  say "1/3 unauthenticated request is rejected"
  code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 180 "$url/api/health")
  [ "$code" = "401" ] || die "expected 401 without token, got $code"

  say "2/3 /api/health with token"
  curl -fsS --max-time 180 -H "X-Auth-Token: $pw" "$url/api/health" \
    | grep -q '"ok"' || die "/api/health failed with token"

  say "3/3 /api/status with token"
  curl -fsS --max-time 60 -H "X-Auth-Token: $pw" "$url/api/status" \
    | grep -q '"phase"' || die "/api/status failed with token"

  say "smoke OK"
}

cmd_down() {
  resolve_config
  [ -f "$STATE_FILE" ] || die "no state file at $STATE_FILE — nothing recorded as created by this script"

  say "down will delete resources marked created in $STATE_FILE:"
  [ "$(state_get created_service)" = "true" ]         && say "  - Cloud Run service $SERVICE"
  [ "$(state_get created_sa)" = "true" ]              && say "  - service account $SA_EMAIL"
  [ "$(state_get created_registry)" = "true" ]        && say "  - Artifact Registry repo $REPO_NAME (and its images)"
  if [ "$PURGE" = "true" ]; then
    [ "$(state_get created_sessions_bucket)" = "true" ] && say "  - bucket gs://$SESSIONS_BUCKET (ANNOTATION DATA)"
    [ "$(state_get created_models_bucket)" = "true" ]   && say "  - bucket gs://$MODELS_BUCKET (model weights)"
  fi
  confirm_destructive "Proceed with teardown in project $PROJECT?"

  if [ "$(state_get created_service)" = "true" ]; then
    say "deleting Cloud Run service $SERVICE"
    gcloud run services delete "$SERVICE" --region "$REGION" --project "$PROJECT" --quiet
    state_set created_service "false"
  elif service_exists; then
    warn "service $SERVICE exists but was not created by this script — leaving it"
  fi

  if [ "$(state_get created_sa)" = "true" ]; then
    say "deleting service account $SA_EMAIL"
    gcloud iam service-accounts delete "$SA_EMAIL" --project "$PROJECT" --quiet
    state_set created_sa "false"
  fi

  if [ "$(state_get created_registry)" = "true" ]; then
    say "deleting Artifact Registry repo $REPO_NAME"
    gcloud artifacts repositories delete "$REPO_NAME" --location "$REGION" \
      --project "$PROJECT" --quiet
    state_set created_registry "false"
  fi

  if [ "$PURGE" = "true" ]; then
    confirm_destructive "PERMANENTLY delete gs://$SESSIONS_BUCKET (all annotation data) and gs://$MODELS_BUCKET (weights)?"
    if [ "$(state_get created_sessions_bucket)" = "true" ]; then
      gsutil -m rm -r "gs://$SESSIONS_BUCKET"
      state_set created_sessions_bucket "false"
    fi
    if [ "$(state_get created_models_bucket)" = "true" ]; then
      gsutil -m rm -r "gs://$MODELS_BUCKET"
      state_set created_models_bucket "false"
    fi
  else
    say "kept: GCS buckets (annotation data + weights). Use 'down --purge' to delete them."
  fi
  say "down complete. Enabled APIs are never disabled (other workloads may use them)."
}

# --------------------------------------------------------------- main -----

COMMAND="up"
YES="false"
ROTATE_PASSWORD="false"
PURGE="false"
FOLLOW="false"

while [ $# -gt 0 ]; do
  case "$1" in
    up|down|status|logs|password|generate-password|smoke) COMMAND="$1" ;;
    --yes|-y)           YES="true" ;;
    --rotate-password)  ROTATE_PASSWORD="true" ;;
    --purge)            PURGE="true" ;;
    -f)                 FOLLOW="true" ;;
    -h|--help)          sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)                  die "unknown argument: $1 (see --help)" ;;
  esac
  shift
done

case "$COMMAND" in
  up)                cmd_up ;;
  status)            cmd_status ;;
  logs)              cmd_logs ;;
  password)          cmd_password ;;
  generate-password) generate_password ;;
  smoke)             cmd_smoke ;;
  down)              cmd_down ;;
esac
```

- [ ] **Step 2: Make executable and gitignore the state file**

```bash
chmod +x deploy/deploy.sh
```

Append to `.gitignore` (after the `# Environment (local overrides)` section):

```
# Deploy state (per-machine, contains resource bookkeeping)
deploy/.deploy-state.json
```

- [ ] **Step 3: Verify without touching GCP**

```bash
bash -n deploy/deploy.sh                      # syntax check → silent
./deploy/deploy.sh --help                     # prints the usage header
./deploy/deploy.sh generate-password          # → e.g. "ember-rowan-quay"
./deploy/deploy.sh generate-password          # → different password
```
Expected: help text renders; two different three-word lowercase passwords joined by `-`.

Also verify the state helpers in isolation:

```bash
STATE=/tmp/test-state.json
python3 - <<'EOF'
# mirrors state_set/state_get logic
import json
path = "/tmp/test-state.json"
json.dump({"project": "p1", "created_sa": "true"}, open(path, "w"))
print(json.load(open(path))["created_sa"])
EOF
rm -f /tmp/test-state.json
```
Expected: `true`.

If `shellcheck` is installed, run `shellcheck deploy/deploy.sh` and fix anything at error severity (style warnings can stay).

- [ ] **Step 4: Commit**

```bash
git add deploy/deploy.sh .gitignore
git commit -m "feat(deploy): lifecycle script — idempotent up, state-tracked down, logs/password/smoke"
```

---

### Task 6: Documentation — DEPLOY.md, deploy/README.md, doc pointer

**Files:**
- Create: `DEPLOY.md`
- Create: `deploy/README.md`
- Replace content: `docs/deploy-cloud-run.md`

- [ ] **Step 1: Create `deploy/README.md`** (the short-context agent entry point — keep it under ~120 lines, no narrative):

```markdown
# Deploy quick-reference

Operate the GCP deployment of this tool entirely through `deploy/deploy.sh`.
This file is self-contained for agents; humans see [DEPLOY.md](../DEPLOY.md)
for background and first-time setup.

## Commands

| Task | Command |
|---|---|
| Deploy (first time) | `PROJECT=<gcp-project-id> ./deploy/deploy.sh up` |
| Re-deploy after a code change | `./deploy/deploy.sh up` (config is remembered; unchanged steps skip) |
| What is deployed right now | `./deploy/deploy.sh status` |
| Get the shared password | `./deploy/deploy.sh password` |
| Read logs (last 100 lines) | `./deploy/deploy.sh logs` |
| Tail logs | `./deploy/deploy.sh logs -f` |
| Smoke-test the live service | `./deploy/deploy.sh smoke` |
| Rotate the password | `./deploy/deploy.sh up --rotate-password` |
| Tear down (keeps data) | `./deploy/deploy.sh down` |
| Tear down including buckets | `./deploy/deploy.sh down --purge` |

Non-interactive runs (agents): append `--yes`. `down` refuses to run
non-interactively without it; `up` auto-proceeds with a log line.

## Auth

Every `/api/*` endpoint — including `/api/health` — requires the header
`X-Auth-Token: <password>`. Without it: `401 {"error": "unauthorized"}`.

```bash
URL=$(./deploy/deploy.sh status | grep 'url=' | sed 's/.*url=//')
PW=$(./deploy/deploy.sh password)
curl -H "X-Auth-Token: $PW" "$URL/api/status"   # → {"phase": "idle", ...}
curl "$URL/api/status"                          # → 401
```

The first request after an idle period takes 60–90 s (GPU cold start —
the container scales to zero). Use `curl --max-time 180` and retry once.

## State

- `deploy/.deploy-state.json` (gitignored) remembers project/region/names
  and which resources the script created. `down` deletes only resources
  marked created there — it never touches pre-existing infrastructure.
- The password lives in the Cloud Run service env (`AUTH_PASSWORD`), not
  in git and not in the state file.
- The image tag is a hash of the build inputs; `up` skips Cloud Build when
  the registry already has that hash.

## Prerequisites (first deploy only)

- `gcloud` CLI authenticated (`gcloud auth login`) with Owner/Editor on the
  project; billing enabled.
- If the models bucket is empty: HuggingFace CLI with access to the gated
  `facebook/sam3` model (`pip install -U huggingface_hub && hf auth login`).
  Weights upload once (~7 GB), then never again.

## Common errors

| Symptom | Fix |
|---|---|
| `PERMISSION_DENIED` enabling APIs | account needs Owner/Editor on the project |
| weights missing + `hf` not installed | `pip install -U huggingface_hub && hf auth login`, re-run `up` |
| 401 from every endpoint | missing/wrong `X-Auth-Token` — `./deploy/deploy.sh password` |
| curl timeout on first request | GPU cold start — retry after ~90 s |
| `Missing: <pkg> from lock file` in Cloud Build | frontend lockfile generated on macOS; regenerate with `npm install` inside a Linux container |
| 429 `Rate exceeded.` during propagation | do not change `--concurrency 8` / `--max-instances 1` (see DEPLOY.md) |
```

- [ ] **Step 2: Create `DEPLOY.md`** at the repo root. Carry over the durable content of `docs/deploy-cloud-run.md` (cost table, flag rationale, troubleshooting) reorganized around the script:

```markdown
# Deploying to Google Cloud Run (L4 GPU)

One script provisions everything: APIs, Artifact Registry, GCS buckets,
model weights, the container image, and the Cloud Run service — and tears
it down again. It is idempotent: re-running skips everything already done.

**Coding agents:** the compact runbook is [deploy/README.md](deploy/README.md).
Point your agent there and say "deploy this".

## Quickstart

```bash
# One-time prerequisites
gcloud auth login                      # account with Owner/Editor + billing
pip install -U huggingface_hub         # only if the weights bucket is empty
hf auth login                          # needs access to gated facebook/sam3

# Deploy
PROJECT=your-project-id ./deploy/deploy.sh up
```

The script prints the service URL and a generated three-word password at
the end. Open the URL, enter the password once per browser, annotate.
Region defaults to `us-east4` (Cloud Run L4 availability); override with
`REGION=...` on the first run. All settings are remembered in
`deploy/.deploy-state.json` afterwards.

## What `up` does (and skips)

| # | Step | Skipped when |
|---|---|---|
| 1 | Preflight: gcloud auth, project access, billing | never (cheap) |
| 2 | Enable APIs (run, cloudbuild, artifactregistry, storage) | already enabled |
| 3 | Create Artifact Registry repo | repo exists |
| 4 | Create models + sessions buckets | buckets exist |
| 5 | Download facebook/sam3 weights, upload to models bucket (~7 GB) | `gs://<models>/sam3/` populated |
| 6 | Create service account + grant bucket access | SA exists (grant is idempotent) |
| 7 | Cloud Build the image, tagged by source hash | registry already has this hash |
| 8 | `gcloud run deploy` | service already runs this hash + password |

Step 7 is the expensive one: the image tag is a hash of `backend/`,
`frontend/`, `Dockerfile.native`, and the deploy configs. Unchanged source
→ same tag → no build. Changed source → new tag, built with `--cache-from`
the previous image, so code-only builds take ~3–6 min (first build
~15–20 min).

## Authentication

The deploy is public (`--allow-unauthenticated`) but every `/api/*`
endpoint — including `/api/health` — requires `X-Auth-Token: <password>`,
enforced by the Flask backend (constant-time compare). The password is
three random words (~24 bits), generated at first deploy, stored only in
the Cloud Run service env (`AUTH_PASSWORD`).

- The web UI prompts once per browser and stores the password in
  localStorage.
- `./deploy/deploy.sh password` prints it; `up --rotate-password` replaces
  it (existing browsers must re-enter).
- Threat model: keeps strangers off a ~$0.70/hr GPU and out of your
  annotation data. It is a shared password over TLS, not per-user auth.
  For stricter needs put Cloud Run IAM (`--no-allow-unauthenticated` + 
  `gcloud run services proxy`) or IAP in front instead.

## Day-2 operations

```bash
./deploy/deploy.sh status      # config + live state + source-hash sync
./deploy/deploy.sh logs        # last 100 log lines
./deploy/deploy.sh logs -f     # tail (gcloud beta)
./deploy/deploy.sh smoke       # 401-without-token, health, status checks
./deploy/deploy.sh password    # recover the password
./deploy/deploy.sh down        # delete service/SA/repo created by the script
./deploy/deploy.sh down --purge  # ...and the buckets (annotation data!)
```

`down` reads `deploy/.deploy-state.json` and deletes **only** resources the
script itself created — pre-existing infrastructure is never touched.
Enabled APIs are never disabled. Buckets survive unless `--purge`.

After every deploy, run the **Deploy smoke set** in
[USER-FLOWS.md](USER-FLOWS.md#deploy-smoke-set).

## Why these Cloud Run flags

- `--gpu-type nvidia-l4` — L4 (compute cap 8.9) runs SAM3's bfloat16 path
  natively with Flash Attention. Do not substitute a T4 — see
  [docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md#gpu-selection) for the
  benchmark data and the dtype workarounds a T4 forces.
- `--max-instances 1` — the service is **stateful**: SAM3 inference state,
  local session dirs, and the GCS sync manager live in container memory.
  A second instance would not share that state.
- `--concurrency 8` — matches gunicorn's threads; lets frame fetches run in
  parallel with a long propagation SSE stream. Lower values cause 429s.
- `--min-instances 0` — scales to zero after ~30 min idle. GCS sync
  persists masks/prompts/state; the next visit re-downloads the session.
- `--timeout 3600` — propagation streams over a single SSE request.
- `--termination-grace-period 30` — the SIGTERM handler flushes dirty
  files to GCS before the container dies.

## Cost

| Component | Cost | When |
|---|---|---|
| Cloud Run (L4 GPU + 8 vCPU + 24 GiB) | ~$0.70–0.90/hr | only while an instance is up |
| Cloud Run (idle) | $0 | scales to zero after ~30 min |
| GCS storage | ~$0.02/GB/month | video + frames + masks + weights |
| Artifact Registry | ~$0.10/GB/month | ~10 GB per image hash kept |

Typical usage: 2–3 hours/day of annotation ≈ $2/day of GPU time. Old image
hashes accumulate in Artifact Registry — delete stale tags occasionally or
add a cleanup policy.

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| Cold start takes 60–90 s | model load + warmup on first request | expected; weights are baked into the image |
| 401 from every endpoint | missing/wrong `X-Auth-Token` | `./deploy/deploy.sh password`; UI: enter it in the gate |
| `Missing: ... from lock file` in Cloud Build | macOS-generated frontend lockfile | regenerate `package-lock.json` with `npm install` in a Linux container |
| `Rate exceeded.` + 429s during propagation | concurrency below gunicorn threads | keep `--concurrency 8`; never raise `--max-instances` |
| Session gone after a long break | scaled to zero, memory state dropped | expected — reopen the session; it re-downloads from GCS |
| Permission denied on GCS | SA missing bucket role | re-run `up` (step 6 re-grants) |
| weights download fails | no access to gated facebook/sam3 | request access on HuggingFace, `hf auth login`, re-run |
```

- [ ] **Step 3: Replace `docs/deploy-cloud-run.md`** with a pointer (keep the file so old links keep working):

```markdown
# Deploying to Google Cloud Run

Deployment is now driven by a single script — see **[DEPLOY.md](../DEPLOY.md)**
at the repo root (humans) and **[deploy/README.md](../deploy/README.md)**
(coding-agent quick-reference).

```bash
PROJECT=your-project-id ./deploy/deploy.sh up
```

The manual step-by-step gcloud walkthrough that used to live here was
folded into DEPLOY.md when the script was introduced (2026-06-11); the
git history of this file preserves the original.
```

- [ ] **Step 4: Sanity check + commit**

Verify `deploy/README.md` stays compact: `wc -l deploy/README.md` ≤ ~120 lines.

```bash
git add DEPLOY.md deploy/README.md docs/deploy-cloud-run.md
git commit -m "docs: DEPLOY.md + agent quick-reference; deploy-cloud-run.md now points there"
```

---

### Task 7: USER-FLOWS.md amendments

**Files:**
- Modify: `USER-FLOWS.md`

- [ ] **Step 1: Add invariant N11** at the end of the "Global negative invariants" section (after N10), and update the section heading from `(N1–N10)` to `(N1–N11)`:

```markdown
- **N11** Auth is all-or-nothing — when `AUTH_PASSWORD` is set, no API
  endpoint (including `/api/health`) responds without a valid
  `X-Auth-Token`; when unset, no endpoint demands one. The password never
  appears in URLs (query params leak into request logs).
  _Check:_ `backend/tests/test_auth.py` covers the 401/200 matrix; grep
  `frontend/src` for `X-Auth-Token` — it must only travel as a header.
```

Also update every other "N1–N10" mention: in USER-FLOWS.md's "How to use this harness" section (if present) and in `CLAUDE.md`'s "User Flows" section ("global negative invariants (N1–N10)" → "(N1–N11)"). Stage `CLAUDE.md` in this task's commit too.

- [ ] **Step 2: Add UF-1.7 to the Tier 2 flows table** (after the UF-1.6 row, matching the 4-column format):

```markdown
| UF-1.7 | Password gate (cloud deploys) | When the backend has `AUTH_PASSWORD` set, any 401 response triggers `onUnauthorized` in `api.ts`; `App.tsx` renders `PasswordGate.tsx`, which verifies the entered password against `GET /api/health`, persists it to `localStorage["sam3-auth-token"]` via `setAuthToken`, then calls `refetchStatus` to resume normal routing. All requests (fetch via `apiFetch`, both XHR uploads, the keepalive flush beacon) carry `X-Auth-Token`. | The gate never appears in local dev (no `AUTH_PASSWORD` ⇒ no 401s); a wrong password shows an inline error without storing anything; the password is sent only as a header, never in a URL (N11). |
```

- [ ] **Step 3: Add impact-map rows** (in the impact map table, after the `frontend/src/hooks/useServiceStatus.ts` row):

```markdown
| `backend/app/__init__.py` (auth hook), `frontend/src/components/PasswordGate.tsx` | UF-1.7 — plus a spot-check that one authenticated flow still works end-to-end (e.g. UF-1.1) | N11 |
| `deploy/deploy.sh`, `cloudbuild-native.yaml`, `DEPLOY.md`, `deploy/README.md` | Deploy smoke set (after the next real deploy) | N11 — `smoke` asserts the 401-without-token contract |
```

- [ ] **Step 4: Update the Deploy smoke set.** Replace the intro line and step 0 row, and append a step 9:

Intro line becomes:

```markdown
Run after every deploy, against the deployed URL — step 0 is automated: `./deploy/deploy.sh smoke`. For manual curls, every request needs `-H "X-Auth-Token: $(./deploy/deploy.sh password)"`.
```

Step 0 row becomes:

```markdown
| 0 | UF-1.7 | `./deploy/deploy.sh smoke` — unauthenticated `GET /api/health` returns 401; with `X-Auth-Token`, health returns `ok` and `GET /api/status` returns `{"phase": "idle", ...}` |
```

Append after step 8:

```markdown
| 9 | UF-1.7 | Open the URL in a fresh browser profile — the password gate appears; a wrong password shows an inline error; the printed password unlocks and survives a reload |
```

- [ ] **Step 5: Commit**

```bash
git add USER-FLOWS.md
git commit -m "docs(flows): UF-1.7 password gate, N11 auth invariant, smoke set auth steps"
```

---

### Task 8: Final verification

- [ ] **Step 1: Full backend suite**

```bash
cd backend && python3 -m pytest tests/ -q
```
Expected: all pass.

- [ ] **Step 2: Frontend production check** (the Docker build's exact check)

```bash
cd frontend && npx tsc --noEmit && npm run build
```
Expected: both succeed.

- [ ] **Step 3: Script checks**

```bash
bash -n deploy/deploy.sh && ./deploy/deploy.sh generate-password
```
Expected: silence, then a three-word password.

- [ ] **Step 4: Local end-to-end gate check** (manual, if not done in Task 3)

`AUTH_PASSWORD=test-pass-word make dev` → gate appears → wrong password rejected inline → correct password unlocks → reload skips the gate → plain `make dev` never shows it.

- [ ] **Step 5: Real deploy cycle** (requires GCP access — run only with the user's go-ahead, since a deploy replaces the live container)

```bash
PROJECT=<project> ./deploy/deploy.sh up        # full provision + deploy
./deploy/deploy.sh up                          # second run: every step skips
./deploy/deploy.sh smoke                       # 401 + authenticated checks
./deploy/deploy.sh password                    # prints the password
```
Then run the Deploy smoke set steps 1–9 from USER-FLOWS.md in the browser.
Do NOT run `down` against a project with live data unless explicitly asked.

- [ ] **Step 6: Wrap up**

Per CLAUDE.md, behavioral changes and USER-FLOWS.md were amended in the same series of commits. Confirm `git status` is clean and report results (including anything NOT RUN, e.g. the real deploy cycle).
```
