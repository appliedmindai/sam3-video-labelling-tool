# --- Stage 1: Build frontend ---
FROM node:22-slim AS frontend-build
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- Stage 2: Runtime ---
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg nginx git libgl1-mesa-glx libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install SAM 3.1 via HuggingFace Transformers
# For native CUDA backend with Triton kernels, use Dockerfile.native instead.
RUN pip install --no-cache-dir "transformers>=5.0.0" huggingface_hub

# Install backend deps
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt

# Download and cache the gated SAM 3.1 model at build time
ARG HF_TOKEN=""
RUN if [ -n "$HF_TOKEN" ]; then \
        python -c "from huggingface_hub import login; login(token='$HF_TOKEN')" && \
        python -c "from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor; \
            Sam3TrackerVideoModel.from_pretrained('facebook/sam3'); \
            Sam3TrackerVideoProcessor.from_pretrained('facebook/sam3'); \
            print('SAM 3.1 model cached successfully')"; \
    else \
        echo 'WARNING: No HF_TOKEN provided. Model will be downloaded at first run.'; \
    fi

# Copy backend code
COPY backend/ /app/backend/

# Copy frontend build
COPY --from=frontend-build /build/dist /usr/share/nginx/html

# Copy deploy configs
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
COPY deploy/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Remove default nginx config that conflicts
RUN rm -f /etc/nginx/sites-enabled/default

EXPOSE 8080
CMD ["/app/entrypoint.sh"]
