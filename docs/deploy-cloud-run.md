# Deploying to Google Cloud Run (L4 GPU)

This guide deploys the SAM3 Video Labelling Tool to Cloud Run with an NVIDIA L4 GPU and a GCS bucket for session storage. The service scales to zero when idle, so you only pay for GPU time while annotating.

## Prerequisites

- A GCP project with billing enabled and the Cloud Run, Cloud Build, and Artifact Registry APIs turned on
- A region with Cloud Run GPU (L4) availability (e.g. `us-east4`, `us-central1` — check [Cloud Run GPU locations](https://cloud.google.com/run/docs/configuring/services/gpu))
- A HuggingFace account with access to the gated [facebook/sam3](https://huggingface.co/facebook/sam3) model (one-time, for downloading weights)
- `gcloud` and `gsutil` CLIs authenticated

Set these once for the commands below:

```bash
export PROJECT=your-project-id
export REGION=us-east4
export MODELS_BUCKET=your-models-bucket      # holds SAM3 weights (build time)
export SESSIONS_BUCKET=your-sessions-bucket  # holds annotation sessions (runtime)
export IMAGE=$REGION-docker.pkg.dev/$PROJECT/cloud-run-images/sam3-native:latest
```

## 1. Stage the Model Weights in GCS (one time)

The production image bakes the SAM3 weights in at build time instead of downloading them from HuggingFace at runtime. This removes the HF token dependency from both the image and the running container, and makes builds reproducible. Download the weights once and upload them to your own bucket:

```bash
# Download the gated model locally (requires huggingface-cli login)
hf download facebook/sam3 --local-dir /tmp/sam3

# Create the models bucket and upload
gsutil mb -p $PROJECT -l $REGION gs://$MODELS_BUCKET
gsutil -m cp /tmp/sam3/* gs://$MODELS_BUCKET/sam3/
```

The weights are ~7 GB and only change when Meta publishes a new SAM release.

## 2. Build the Image with Cloud Build

```bash
# Create the Artifact Registry repo (one time)
gcloud artifacts repositories create cloud-run-images \
  --repository-format=docker --location=$REGION --project=$PROJECT

gcloud builds submit \
  --config=cloudbuild-native.yaml \
  --substitutions=_IMAGE=$IMAGE,_MODELS_BUCKET=$MODELS_BUCKET \
  --project=$PROJECT
```

The first build takes ~15-20 min (PyTorch base image, native SAM3 install, Flash Attention, weight baking). Subsequent code-only builds take ~3-6 min because `cloudbuild-native.yaml` pulls the previous image and builds with `--cache-from`, reusing the heavy layers. See the [technical report](TECHNICAL_REPORT.md#docker-layer-caching) for how the Dockerfile is ordered to make this work.

## 3. Create a Service Account and Sessions Bucket

```bash
gcloud iam service-accounts create sam3-annotator \
  --display-name="SAM3 Video Labelling Tool" --project=$PROJECT

gsutil mb -p $PROJECT -l $REGION gs://$SESSIONS_BUCKET

gsutil iam ch \
  serviceAccount:sam3-annotator@$PROJECT.iam.gserviceaccount.com:roles/storage.objectAdmin \
  gs://$SESSIONS_BUCKET
```

## 4. Deploy

```bash
gcloud run deploy sam3-annotator \
  --image $IMAGE \
  --region $REGION \
  --project $PROJECT \
  --service-account sam3-annotator@$PROJECT.iam.gserviceaccount.com \
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
  --no-allow-unauthenticated \
  --set-env-vars "SEGMENT_MODE=cloud,GCS_BUCKET=$SESSIONS_BUCKET,SAM3_BACKEND=native"
```

Why each flag matters:

- `--gpu 1 --gpu-type nvidia-l4` — L4 (compute capability 8.9) runs SAM3's bfloat16 path natively with Flash Attention. Do not substitute a T4 — see the [technical report](TECHNICAL_REPORT.md#gpu-selection) for the benchmark data and the dtype workarounds a T4 forces.
- `--max-instances 1` — this service is **stateful**: SAM3 inference state, local session directories, and the GCS sync manager live in container memory. A second instance would not share that state and requests routed to it would fail with "model not loaded". Do not raise this without adding session affinity and cross-instance coordination.
- `--concurrency 8` — matches gunicorn's `--threads 4` with headroom. GPU work is serialized in-process by `SAM3Service._lock`; the extra concurrency lets non-GPU requests (frame fetches, mask JSON) run in parallel with a long propagation SSE stream. With concurrency 1, Cloud Run's admission layer 429s every request during propagation.
- `--min-instances 0` — scales to zero after ~30 min idle. GCS sync persists masks/prompts/state, so nothing is lost; the next visit re-downloads the session and re-initializes SAM3.
- `--timeout 3600` — long propagation runs stream over a single SSE request.
- `--termination-grace-period 30` — the SIGTERM handler cancels propagation and flushes dirty files to GCS before the container dies.
- `--no-cpu-throttling` — keeps the model warm between requests while an instance is up.

## 5. Access and Security

**This tool has no built-in authentication.** Anyone who can reach the URL can upload videos, run the GPU, and read every session in the bucket. Pick one of:

### Option A (recommended): Cloud Run IAM + proxy

Deploy with `--no-allow-unauthenticated` (as above) and tunnel through the gcloud proxy when you want to annotate:

```bash
gcloud run services proxy sam3-annotator --region $REGION --project $PROJECT --port 8080
# open http://localhost:8080
```

The proxy injects your gcloud identity token; only principals with `roles/run.invoker` can reach the service. Grant access per user:

```bash
gcloud run services add-iam-policy-binding sam3-annotator \
  --region $REGION --project $PROJECT \
  --member="user:someone@example.com" --role="roles/run.invoker"
```

### Option B: your own auth layer

Put an authenticating reverse proxy (Identity-Aware Proxy via a load balancer, Cloudflare Access, oauth2-proxy, etc.) in front and keep the service private to it.

### Option C: public (`--allow-unauthenticated`)

Only sensible for short-lived demos on throwaway buckets. You are exposing a GPU that costs ~$0.70/hr to the open internet.

## 6. Verify

```bash
URL=$(gcloud run services describe sam3-annotator --region $REGION --project $PROJECT --format='value(status.url)')

# With Option A, run the proxy and use http://localhost:8080 instead of $URL
curl $URL/api/status        # → {"phase": "idle", ...}
curl $URL/api/health        # → {"ok": true, ...}
```

Then run through the smoke tests in [CLAUDE.md](../CLAUDE.md#deploy-smoke-tests): upload a video, propagate an object, close the session, resume it.

## Cost Estimation

| Component | Cost | When |
|---|---|---|
| Cloud Run (L4 GPU + 8 vCPU + 24 GiB) | ~$0.70-0.90/hr | Only while an instance is up |
| Cloud Run (idle) | $0 | Scales to zero after ~30 min inactivity |
| GCS storage | ~$0.02/GB/month | Video + frames + masks |
| GCS operations | ~$0.005/1000 writes | Negligible (dirty-file sync) |
| Artifact Registry | ~$0.10/GB/month | ~10 GB image |

Typical usage: 2-3 hours/day of annotation ≈ $2/day of GPU time.

## Troubleshooting

| Issue | Cause | Fix |
|---|---|---|
| Cold start takes 60-90s | Model load + backbone warmup on first request | Expected. Weights are baked into the image; nothing to download. |
| `Missing: ... from lock file` in Cloud Build | Frontend lockfile generated on macOS missing Linux optional deps | Regenerate `package-lock.json` with `npm install` inside a Linux container. |
| OOM during propagation | Too many objects or very long video | Should not happen on L4 (24 GB VRAM). Reduce extraction FPS or video resolution. |
| Permission denied on GCS | Service account missing bucket role | Grant `roles/storage.objectAdmin` on the sessions bucket (step 3). |
| `Rate exceeded.` toast + 429s during propagation | `--concurrency` below gunicorn's thread count: the propagation SSE stream consumes the only admission slot | Keep `--concurrency 8` (≥ gunicorn `--threads 4`). Do not raise `--max-instances` instead. |
| Session gone after a long break | Container scaled to zero and in-memory state was dropped | Expected. Reopen the session from the list — it re-downloads from GCS. |
