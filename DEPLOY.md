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
./deploy/deploy.sh logs -f     # tail (requires: gcloud components install beta)
./deploy/deploy.sh smoke       # 401-without-token, health, status checks
./deploy/deploy.sh password    # recover the password
./deploy/deploy.sh generate-password  # print a fresh password (no GCP calls)
./deploy/deploy.sh down        # delete service/SA/repo created by the script
./deploy/deploy.sh down --purge  # ...and the buckets (annotation data!)
```

`down` reads `deploy/.deploy-state.json` and deletes **only** resources the
script itself created — pre-existing infrastructure is never touched.
Enabled APIs are never disabled. Buckets survive unless `--purge`.

`down --purge` always prompts interactively for bucket deletion even when
`--yes` is passed — bucket deletion is irrecoverable and the extra
confirmation is intentional.

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
| `log tailing requires the gcloud beta component` | gcloud beta not installed | `gcloud components install beta` |
