#!/usr/bin/env python3
"""Stress test: propagate 3 objects across 1000+ frames with memory monitoring.

Monitors RSS memory, GPU memory (MPS), CPU usage, and per-frame inference time.
Detects memory leaks by comparing start vs end memory after propagation.

Usage:
    python3 backend/scripts/stress_test.py \
        --video path/to/video.mp4 \
        --fps 30 --max-frames 1500 \
        --output-dir /tmp/sam3-stress
"""
import argparse
import json
import logging
import os
import resource
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import torch
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-12s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("stress")


def get_rss_mb():
    """Resident set size in MB (macOS/Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def get_mps_memory_mb():
    """MPS allocated memory in MB, or None if not available."""
    try:
        if torch.backends.mps.is_available():
            return torch.mps.current_allocated_memory() / (1024 * 1024)
    except Exception:
        pass
    return None


class MemoryTracker:
    def __init__(self):
        self.samples = []

    def sample(self, label=""):
        rss = get_rss_mb()
        mps = get_mps_memory_mb()
        entry = {"time": time.time(), "rss_mb": rss, "mps_mb": mps, "label": label}
        self.samples.append(entry)
        return entry

    def summary(self):
        if not self.samples:
            return {}
        rss_vals = [s["rss_mb"] for s in self.samples]
        mps_vals = [s["mps_mb"] for s in self.samples if s["mps_mb"] is not None]
        return {
            "rss_start_mb": rss_vals[0],
            "rss_end_mb": rss_vals[-1],
            "rss_peak_mb": max(rss_vals),
            "rss_delta_mb": rss_vals[-1] - rss_vals[0],
            "mps_start_mb": mps_vals[0] if mps_vals else None,
            "mps_end_mb": mps_vals[-1] if mps_vals else None,
            "mps_peak_mb": max(mps_vals) if mps_vals else None,
            "mps_delta_mb": (mps_vals[-1] - mps_vals[0]) if len(mps_vals) >= 2 else None,
            "samples": len(self.samples),
        }


def run_stress_test(frames_dir, output_dir, num_objects=3):
    from app.services.sam3_service import SAM3Service

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    num_frames = len(frame_files)
    img = Image.open(os.path.join(frames_dir, frame_files[0]))
    width, height = img.size

    logger.info("=" * 70)
    logger.info("SAM 3.1 STRESS TEST")
    logger.info("Frames: %d @ %dx%d", num_frames, width, height)
    logger.info("Objects: %d", num_objects)
    logger.info("=" * 70)

    mem = MemoryTracker()
    mem.sample("before_model_load")

    service = SAM3Service()
    sid = "stress-test"

    # ── Model Load ──
    t0 = time.time()
    service._ensure_model()
    load_time = time.time() - t0
    mem.sample("after_model_load")
    logger.info("Model loaded in %.1fs | RSS=%.0fMB | MPS=%.0fMB",
                load_time, mem.samples[-1]["rss_mb"], mem.samples[-1]["mps_mb"] or 0)

    # ── Init Session ──
    t0 = time.time()
    result = service.init_session(sid, frames_dir)
    init_time = time.time() - t0
    mem.sample("after_init_session")
    logger.info("Session init in %.1fs | frames=%d | RSS=%.0fMB | MPS=%.0fMB",
                init_time, result["num_frames"], mem.samples[-1]["rss_mb"],
                mem.samples[-1]["mps_mb"] or 0)

    # ── Add clicks for N objects at different positions ──
    positions = [
        (width // 4, height // 4),
        (3 * width // 4, height // 4),
        (width // 2, 3 * height // 4),
    ]
    for obj_id in range(1, num_objects + 1):
        x, y = positions[(obj_id - 1) % len(positions)]
        t0 = time.time()
        r = service.add_click(sid, 0, obj_id, [[x, y]], [1])
        click_ms = int((time.time() - t0) * 1000)
        area = r["masks"].get(obj_id, {}).get("area", 0)
        conf = r["masks"].get(obj_id, {}).get("confidence", 0)
        logger.info("Object %d click: area=%d conf=%.3f time=%dms", obj_id, area, conf or 0, click_ms)

    mem.sample("after_clicks")

    # ── Propagation with all objects ──
    object_ids = list(range(1, num_objects + 1))
    frame_times = []
    frame_areas = {oid: [] for oid in object_ids}
    propagated = []

    def persist_fn(result):
        propagated.append(result)

    logger.info("Starting forward propagation of %d objects across %d frames...", num_objects, num_frames)
    t_prop_start = time.time()

    service.start_propagation(sid, 0, False, persist_fn, object_ids=object_ids)

    frame_count = 0
    for result in service.subscribe_propagation(sid):
        if "error" in result:
            logger.error("PROPAGATION ERROR: %s", result["error"])
            break

        frame_count += 1
        fi = result["frame_idx"]

        for oid in object_ids:
            area = result["masks"].get(oid, {}).get("area", 0)
            frame_areas[oid].append(area)

        # Sample memory every 100 frames
        if frame_count % 100 == 0:
            elapsed = time.time() - t_prop_start
            fps = frame_count / elapsed if elapsed > 0 else 0
            mem_entry = mem.sample(f"frame_{fi}")
            logger.info(
                "Frame %d/%d (%.0f%%) | %.1f fps | RSS=%.0fMB | MPS=%.0fMB",
                fi, num_frames, 100 * frame_count / num_frames, fps,
                mem_entry["rss_mb"], mem_entry["mps_mb"] or 0,
            )

    t_prop_end = time.time()
    prop_duration = t_prop_end - t_prop_start
    mem.sample("after_propagation")

    # ── Results ──
    logger.info("=" * 70)
    logger.info("PROPAGATION COMPLETE")
    logger.info("  Frames propagated: %d", frame_count)
    logger.info("  Total time: %.1fs", prop_duration)
    logger.info("  Average FPS: %.2f", frame_count / prop_duration if prop_duration > 0 else 0)
    logger.info("  Average ms/frame: %.0f", (prop_duration / frame_count * 1000) if frame_count > 0 else 0)

    for oid in object_ids:
        areas = frame_areas[oid]
        if areas:
            nonzero = [a for a in areas if a > 0]
            lost = len(areas) - len(nonzero)
            logger.info("  Object %d: avg_area=%d lost=%d/%d frames",
                        oid, int(np.mean(nonzero)) if nonzero else 0, lost, len(areas))

    mem_summary = mem.summary()
    logger.info("MEMORY:")
    logger.info("  RSS: %.0fMB start → %.0fMB end (peak %.0fMB, delta %+.0fMB)",
                mem_summary["rss_start_mb"], mem_summary["rss_end_mb"],
                mem_summary["rss_peak_mb"], mem_summary["rss_delta_mb"])
    if mem_summary["mps_start_mb"] is not None:
        logger.info("  MPS: %.0fMB start → %.0fMB end (peak %.0fMB, delta %+.0fMB)",
                    mem_summary["mps_start_mb"], mem_summary["mps_end_mb"],
                    mem_summary["mps_peak_mb"], mem_summary["mps_delta_mb"])

    # Check for memory leak (>500MB growth is suspicious)
    leak_threshold = 500
    if mem_summary["rss_delta_mb"] > leak_threshold:
        logger.warning("POTENTIAL MEMORY LEAK: RSS grew by %.0fMB (threshold=%dMB)",
                       mem_summary["rss_delta_mb"], leak_threshold)

    # ── Cleanup ──
    service.close_session(sid)
    mem.sample("after_close")
    logger.info("After close: RSS=%.0fMB | MPS=%.0fMB",
                mem.samples[-1]["rss_mb"], mem.samples[-1]["mps_mb"] or 0)

    # ── Save results ──
    results = {
        "num_frames": num_frames,
        "num_objects": num_objects,
        "resolution": f"{width}x{height}",
        "propagation_time_s": prop_duration,
        "frames_propagated": frame_count,
        "avg_fps": frame_count / prop_duration if prop_duration > 0 else 0,
        "avg_ms_per_frame": (prop_duration / frame_count * 1000) if frame_count > 0 else 0,
        "memory": mem_summary,
        "memory_samples": mem.samples,
        "per_object_areas": {str(k): v for k, v in frame_areas.items()},
    }
    results_path = os.path.join(output_dir, "stress_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)
    logger.info("=" * 70)

    return mem_summary["rss_delta_mb"] < leak_threshold


def main():
    parser = argparse.ArgumentParser(description="SAM 3.1 stress test")
    parser.add_argument("--video", required=True, help="Path to video")
    parser.add_argument("--fps", type=int, default=30, help="Frame extraction FPS")
    parser.add_argument("--max-frames", type=int, default=1500, help="Max frames")
    parser.add_argument("--objects", type=int, default=3, help="Number of objects")
    parser.add_argument("--output-dir", default="/tmp/sam3-stress")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    frames_dir = os.path.join(args.output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Extract frames
    logger.info("Extracting frames at %d FPS (max %d)...", args.fps, args.max_frames)
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    from app.services.video_processor import extract_frames
    extract_frames(args.video, frames_dir, fps=args.fps)
    all_frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    for f in all_frames[args.max_frames:]:
        os.remove(os.path.join(frames_dir, f))
    actual = min(len(all_frames), args.max_frames)
    logger.info("Extracted %d frames", actual)

    success = run_stress_test(frames_dir, args.output_dir, num_objects=args.objects)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
