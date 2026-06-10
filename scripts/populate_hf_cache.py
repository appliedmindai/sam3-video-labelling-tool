"""Populate the HuggingFace Hub cache from local model files.

Used during Docker build to pre-cache SAM3 weights downloaded from GCS,
so from_pretrained('facebook/sam3') and build_sam3_video_model() find
them locally without network access.
"""
import hashlib
import os

src_dir = "/app/models/sam3"
cache_base = os.path.expanduser("~/.cache/huggingface/hub/models--facebook--sam3")
snap_dir = os.path.join(cache_base, "snapshots", "local")
blob_dir = os.path.join(cache_base, "blobs")
refs_dir = os.path.join(cache_base, "refs")

os.makedirs(snap_dir, exist_ok=True)
os.makedirs(blob_dir, exist_ok=True)
os.makedirs(refs_dir, exist_ok=True)

for fname in os.listdir(src_dir):
    src_path = os.path.join(src_dir, fname)
    if not os.path.isfile(src_path):
        continue
    h = hashlib.sha256()
    with open(src_path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            h.update(chunk)
    blob_path = os.path.join(blob_dir, h.hexdigest())
    if not os.path.exists(blob_path):
        os.link(src_path, blob_path)
    snap_path = os.path.join(snap_dir, fname)
    os.symlink(blob_path, snap_path)
    print(f"Cached: {fname} ({os.path.getsize(src_path) / 1e6:.1f} MB)")

with open(os.path.join(refs_dir, "main"), "w") as f:
    f.write("local")

print(f"HF cache populated at {cache_base}")
