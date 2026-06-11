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
| Generate a new password (no GCP calls) | `./deploy/deploy.sh generate-password` |
| Read logs (last 100 lines) | `./deploy/deploy.sh logs` |
| Tail logs | `./deploy/deploy.sh logs -f` (requires `gcloud components install beta`) |
| Smoke-test the live service | `./deploy/deploy.sh smoke` |
| Rotate the password | `./deploy/deploy.sh up --rotate-password` |
| Tear down (keeps data) | `./deploy/deploy.sh down` |
| Tear down including buckets | `./deploy/deploy.sh down --purge` |

Non-interactive runs (agents): append `--yes`. `down` refuses to run
non-interactively without it; `up` auto-proceeds with a log line.
`down --purge` always prompts for bucket deletion interactively, even with
`--yes` (bucket deletion is irrecoverable; the extra prompt is intentional).

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
| `log tailing requires the gcloud beta component` | `gcloud components install beta` |
