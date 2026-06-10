# SAM3 Video Labelling Tool

A video annotation tool powered by [SAM 3.1](https://github.com/facebookresearch/sam3) for labeling objects in video and exporting datasets for [RF-DETR](https://github.com/roboflow/rf-detr) and other object detection models. Runs locally on Apple Silicon (MPS) or NVIDIA GPUs (CUDA), and deploys to Google Cloud Run with an L4 GPU and Google Cloud Storage for session persistence.

![Screenshot](docs/screenshot.gif)

## Features

- **Click, box & text segmentation** — SAM 3.1 generates precise masks from clicks, bounding boxes, or natural language text prompts
- **Multi-object propagation** — Shift+click to select multiple objects, propagate them together in a single SAM3 pass for better boundary handling
- **Mask propagation** — Label key frames, propagate forward/backward/both with per-frame confidence tracking
- **Multi-class, multi-instance** — Unlimited annotation classes with color coding and independent object tracking
- **Zoom-to-mask inspect** — One-click zoom into any object's bounding box with adjustable padding, tools stay active
- **Visual mask deletion** — Select a frame range with previews, delete masks by source keyframe and direction
- **Session management** — Auto-saves to disk (local) or GCS (cloud); export/import sessions as portable zip bundles
- **COCO export** — Standard COCO JSON with bounding boxes and segmentation polygons
- **Dark mode** — System preference detection with manual toggle

## Quick Start (local)

**Prerequisites:** Python 3.11+ with conda/miniforge, Node.js 22+, Apple Silicon Mac or NVIDIA GPU

```bash
# 1. Backend
conda create -n sam3-annotator python=3.11 -y
conda activate sam3-annotator
cd backend
pip install -r requirements.txt
huggingface-cli login  # SAM 3.1 weights auto-download from HF Hub (gated model)

# 2. Frontend
cd ../frontend
npm install

# 3. Run both
cd ..
make dev   # Backend on :5555, frontend on :5173
```

Device selection is automatic: MPS on Apple Silicon, CUDA on NVIDIA, CPU otherwise. The SAM3 backend is also automatic — the native `facebookresearch/sam3` predictor on CUDA, HuggingFace Transformers elsewhere. Override with `SAM3_DEVICE` and `SAM3_BACKEND` env vars.

## Docker (local, NVIDIA GPU)

```bash
docker build -t sam3-annotator .
docker run --gpus all -p 8080:8080 -v $(pwd)/data:/data sam3-annotator
```

Session data persists in the `/data` volume. Uses nginx + gunicorn with SAM 3.1.

## Cloud Run Deployment

Deploys to Cloud Run with an NVIDIA L4 GPU and a GCS bucket for session storage — scales to zero when idle (~$0.70/hr only while annotating). See **[docs/deploy-cloud-run.md](docs/deploy-cloud-run.md)** for the full guide, including the model-weights bucket setup and security options.

```bash
# 1. Build (~3-6 min with layer cache, ~15-20 min on first build)
gcloud builds submit --config=cloudbuild-native.yaml \
  --substitutions=_IMAGE=$REGION-docker.pkg.dev/$PROJECT/cloud-run-images/sam3-native:latest,_MODELS_BUCKET=$MODELS_BUCKET

# 2. Deploy
gcloud run deploy sam3-annotator \
  --image $REGION-docker.pkg.dev/$PROJECT/cloud-run-images/sam3-native:latest \
  --region us-east4 \
  --gpu 1 --gpu-type nvidia-l4 --cpu 8 --memory 24Gi \
  --timeout 3600 --concurrency 8 \
  --min-instances 0 --max-instances 1 \
  --no-cpu-throttling \
  --set-env-vars "SEGMENT_MODE=cloud,GCS_BUCKET=$SESSIONS_BUCKET,SAM3_BACKEND=native"
```

Key constraints: `max-instances=1` (stateful — SAM3 inference state lives in memory), `concurrency=8` (GPU access serialized by an in-process lock), scales to zero when idle.

> **⚠️ No built-in authentication.** This tool has no user accounts or token auth. If you deploy it publicly reachable, anyone with the URL can use your GPU and read your sessions. Use Cloud Run IAM (`--no-allow-unauthenticated` + `gcloud run services proxy`) or put your own auth layer in front. See the security section of the deploy guide.

## Technical Report

[docs/TECHNICAL_REPORT.md](docs/TECHNICAL_REPORT.md) explains the engineering decisions: the dual SAM3 backend, GPU selection (why L4 and not T4), Cloud Run statefulness, the Docker layer-caching strategy that cuts rebuilds from ~20 min to ~3 min, and the MPS-specific constraints for Apple Silicon. [docs/MASK_FRAME_STREAMING.md](docs/MASK_FRAME_STREAMING.md) covers the bandwidth optimizations — RLE masks, version-vector delta sync, IndexedDB frame/mask caches, and SSE propagation streaming. The `blog/` directory has longer-form write-ups.

## Project Structure

```
backend/
  app/
    routes/          Flask blueprints: video, session, segment, export, status
    services/        SAM3Service, mask/prompt/session storage, GCS sync, pipeline
  tests/             pytest suite
frontend/            React 18 + TypeScript + Vite (shadcn/ui, Tailwind v4)
deploy/              nginx.conf, entrypoint.sh (Docker)
scripts/             Benchmark utilities, model cache helper
docs/                Deploy guide, technical report, concurrency audit
blog/                Engineering write-ups
Dockerfile           HuggingFace backend image
Dockerfile.native    Native SAM3 backend image (CUDA, production)
cloudbuild-native.yaml  Cloud Build config for the production image
```

## Authors

- **Danilo Gasques** ([@danilogr](https://github.com/danilogr))
- **Claude** (Anthropic — Claude Opus & Claude Fable wrote most of the implementation, working with Claude Code)

## License

[MIT](LICENSE) — provided **as is**, without warranty of any kind. Use at your own risk: this tool was built for internal annotation workflows and is shared in the hope it's useful, not as a supported product. SAM 3.1 model weights are distributed by Meta under their own license terms — review them at [facebook/sam3](https://huggingface.co/facebook/sam3).
