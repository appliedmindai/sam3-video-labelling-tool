# Shared-Password Auth for Cloud Deployment — Design

**Date:** 2026-06-11
**Status:** Approved

## Problem

The cloud (Cloud Run + GCS) deployment has no built-in authentication. The
current options are the gcloud IAM proxy (requires `gcloud` on every client
machine) or a fully public URL (exposes a ~$0.70/hr GPU and every session in
the bucket to anyone who finds it).

## Goal

A single shared password — three auto-generated words — that:

1. is generated and shown **once** by a deploy script,
2. is sent on **all** web requests from the frontend,
3. is validated by the Flask backend on every `/api/*` request,
4. is entered by the user once per browser via a small password screen.

Threat model: keep strangers off a paid GPU and out of session data. Not
multi-user, not bank-grade. TLS is provided by Cloud Run.

## Decisions (made during brainstorming)

- **Deploy script** (new `deploy/deploy-cloud-run.sh`) generates the password
  and passes it to Cloud Run as an env var — not a doc snippet, not
  backend-generated.
- **Frontend** prompts once and persists the password in `localStorage`.
- **`/api/health` is also protected.** The heartbeat keeps the GPU container
  alive (billed time), so an open health endpoint would let anyone keep the
  service warm. The frontend heartbeat already goes through `api.ts` and can
  send the header.
- **Mechanism: Flask header check + app-level prompt** (chosen over nginx
  basic auth and query-param tokens). The check lives in Flask so it protects
  the API regardless of what sits in front; query params were rejected because
  tokens leak into request logs.

## Design

### 1. Deploy script — `deploy/deploy-cloud-run.sh`

Wraps the existing `gcloud run deploy` command from `docs/deploy-cloud-run.md`:

- Requires the env vars the doc already establishes (`PROJECT`, `REGION`,
  `IMAGE`, `SESSIONS_BUCKET`); fails fast with a clear message if any is
  missing.
- Generates the password with `python3 -c` using `secrets.choice` over an
  embedded list of ~256 short common words → `crimson-otter-lantern` style
  (~24 bits of entropy).
- Deploys with `--allow-unauthenticated` (the password replaces IAM as the
  gate) and adds `AUTH_PASSWORD=<password>` to `--set-env-vars`, alongside the
  existing `SEGMENT_MODE=cloud,GCS_BUCKET=...,SAM3_BACKEND=native`.
- Confirmation prompt is TTY-guarded per the repo deploy-script convention
  (non-interactive stdin auto-proceeds with a log line; never a silent
  `exit 0`).
- On success, prints the service URL and the password once, with a note that
  re-running the script generates a **new** password and invalidates the old
  one (and that the value remains visible to project admins in the Cloud Run
  service config).

### 2. Backend — `app/config.py` + `app/__init__.py`

- `app/config.py` reads `AUTH_PASSWORD` from env (`None` when unset).
- A `before_request` hook in `create_app()`: if `AUTH_PASSWORD` is set, every
  request — including `/api/health` — must carry a matching `X-Auth-Token`
  header, compared with `hmac.compare_digest`. Missing or wrong →
  `401 {"error": "unauthorized"}`.
- When `AUTH_PASSWORD` is unset (local dev, tests), the hook is a no-op —
  zero behavior change locally.
- The `/api/health` route comment ("no auth") is updated, since it is no
  longer true when `AUTH_PASSWORD` is set.

### 3. Frontend — `api.ts` interception + `PasswordGate`

- `api.ts` gains:
  - a module-level token initialized from `localStorage["sam3-auth-token"]`,
  - a `setAuthToken(token)` export that updates the module state and
    localStorage,
  - an `onUnauthorized(callback)` registration,
  - an `apiFetch(url, init)` wrapper that injects `X-Auth-Token` and invokes
    the callback on any 401 response.
- All `fetch(` calls inside `api.ts` switch to `apiFetch(`. The XHR upload
  adds one `xhr.setRequestHeader("X-Auth-Token", ...)` after `open()`.
- `App.tsx` registers the callback → sets an `authRequired` state → renders a
  `PasswordGate` component (centered card, password input + submit, shadcn
  styling). Submit validates the candidate against `/api/health`; success
  saves the token via `setAuthToken` and clears `authRequired`; failure shows
  an inline error and stays on the gate.
- Local dev: the backend never returns 401, so the gate never appears.

### 4. Docs & flows

- `docs/deploy-cloud-run.md`: the deploy step points at the script; the
  security section gains the shared-password option as the recommended
  default for solo use. The IAM-proxy option stays documented for stricter
  setups.
- `USER-FLOWS.md`: amended in the same commit — new auth-gate flow, plus a
  note that all API requests carry `X-Auth-Token` in cloud mode (per the
  harness rule in CLAUDE.md).

### 5. Testing

- Backend pytest, with `AUTH_PASSWORD` set: missing header → 401, wrong
  header → 401, correct header → 200; with it unset: everything open.
  `/api/health` is included in the matrix.
- Frontend: `npx tsc --noEmit` and the production check `tsc -b && vite
  build`; manual gate verification via `make dev` with `AUTH_PASSWORD` set
  locally.

## Out of scope

Rate limiting, multiple users / per-user passwords, password rotation
endpoints, session expiry, and protecting the static HTML/JS (the app is open
source; only the API needs the gate).
