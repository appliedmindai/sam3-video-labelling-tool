"""Benchmark endpoint for automated performance testing.

Runs a standardized segmentation + propagation test on a loaded session
and returns structured JSON with timing, memory, and quality metrics.

Usage:
    POST /api/benchmark/<session_id>
    Body (optional): {"objects": 3, "propagate_frames": null}

Response:
    {
        "session_id": "...",
        "resolution": "432x768",
        "num_frames": 500,
        "device": "cuda",
        "model": "Sam3TrackerVideoModel",
        "model_params": "466M",
        "results": {
            "init": {"time_ms": 1234},
            "clicks": [
                {"obj_id": 1, "time_ms": 45, "area": 12345, "confidence": 0.95},
                ...
            ],
            "propagation": {
                "direction": "forward",
                "total_frames": 500,
                "total_time_ms": 150000,
                "avg_ms_per_frame": 300,
                "fps": 3.33,
                "per_frame": [
                    {"frame_idx": 0, "time_ms": 280, "masks": {1: {"area": 123}, ...}},
                    ...
                ],
                "object_summary": {
                    "1": {"avg_area": 18000, "lost_frames": 5, "total_frames": 500},
                    ...
                }
            },
            "memory": {
                "before_init": {"rss_mb": 400, "gpu_mb": 0},
                "after_init": {"rss_mb": 5000, "gpu_mb": 1800},
                "after_clicks": {"rss_mb": 5000, "gpu_mb": 1900},
                "after_propagation": {"rss_mb": 5000, "gpu_mb": 3200},
                "after_close": {"rss_mb": 5000, "gpu_mb": 1900}
            }
        }
    }
"""
import logging
import os
import resource
import time

import torch
from flask import Blueprint, jsonify, request

from app.config import SESSIONS_DIR
from app.services.sam3_service import SAM3Service

logger = logging.getLogger(__name__)
benchmark_bp = Blueprint("benchmark", __name__)
sam = SAM3Service()

_start_time = time.time()


@benchmark_bp.route("/health", methods=["GET"])
def health():
    """Health check with model info — use this to verify deployment is alive."""
    snap = sam.debug_snapshot()
    mem = _mem_snapshot("health")
    return jsonify({
        "status": "ok",
        "uptime_s": int(time.time() - _start_time),
        "device": snap["device"],
        "model": snap["model"],
        "model_params": snap["model_params"],
        "sessions_loaded": snap["sessions_loaded"],
        "memory": mem,
    })


def _mem_snapshot(label=""):
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
    gpu_mb = None
    try:
        if torch.cuda.is_available():
            gpu_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        elif torch.backends.mps.is_available():
            gpu_mb = torch.mps.current_allocated_memory() / (1024 * 1024)
    except Exception:
        pass
    return {"rss_mb": round(rss_mb), "gpu_mb": round(gpu_mb) if gpu_mb is not None else None, "label": label}


@benchmark_bp.route("/<session_id>", methods=["POST"])
def run_benchmark(session_id):
    """Run a standardized benchmark on a loaded session."""
    frames_dir = os.path.join(SESSIONS_DIR, session_id, "frames")
    if not os.path.isdir(frames_dir):
        return jsonify({"error": "Session frames not found"}), 404

    data = request.json or {}
    num_objects = data.get("objects", 3)
    max_propagate = data.get("propagate_frames", None)  # None = all frames

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    num_frames = len(frame_files)
    if num_frames == 0:
        return jsonify({"error": "No frames found"}), 404

    from PIL import Image
    img = Image.open(os.path.join(frames_dir, frame_files[0]))
    width, height = img.size

    # Use a dedicated benchmark session ID to avoid conflicts
    bench_sid = f"benchmark-{session_id}"
    memory = {}

    logger.info("benchmark | start | session=%s | frames=%d | resolution=%dx%d | objects=%d",
                session_id, num_frames, width, height, num_objects)

    try:
        memory["before_init"] = _mem_snapshot("before_init")

        # ── Init ──
        t0 = time.time()
        init_result = sam.init_session(bench_sid, frames_dir)
        init_ms = int((time.time() - t0) * 1000)
        memory["after_init"] = _mem_snapshot("after_init")
        logger.info("benchmark | init | time_ms=%d", init_ms)

        # ── Clicks ──
        positions = [
            (width // 4, height // 4),
            (3 * width // 4, height // 4),
            (width // 2, 3 * height // 4),
            (width // 4, 3 * height // 4),
            (3 * width // 4, 3 * height // 4),
        ]
        click_results = []
        for obj_id in range(1, num_objects + 1):
            x, y = positions[(obj_id - 1) % len(positions)]
            t0 = time.time()
            r = sam.add_click(bench_sid, 0, obj_id, [[x, y]], [1])
            click_ms = int((time.time() - t0) * 1000)
            mask_data = r["masks"].get(obj_id, {})
            click_results.append({
                "obj_id": obj_id,
                "point": [x, y],
                "time_ms": click_ms,
                "area": mask_data.get("area", 0),
                "confidence": mask_data.get("confidence"),
            })
            logger.info("benchmark | click | obj=%d | time_ms=%d | area=%d",
                        obj_id, click_ms, mask_data.get("area", 0))

        memory["after_clicks"] = _mem_snapshot("after_clicks")

        # ── Propagation ──
        object_ids = list(range(1, num_objects + 1))
        per_frame = []
        per_object_areas = {oid: [] for oid in object_ids}
        prop_results = []

        memory_timeline = []  # Memory samples during propagation

        def persist_fn(result):
            prop_results.append(result)
            # Sample memory every 50 frames for timeline
            if len(prop_results) % 50 == 0 or len(prop_results) == 1:
                memory_timeline.append({
                    "frame": result["frame_idx"],
                    "count": len(prop_results),
                    **_mem_snapshot(f"frame_{result['frame_idx']}"),
                })

        t_prop_start = time.time()
        sam.start_propagation(bench_sid, 0, False, persist_fn, object_ids=object_ids)

        # Wait for propagation to finish via subscriber (acts as a barrier)
        for result in sam.subscribe_propagation(bench_sid):
            if "error" in result:
                logger.error("benchmark | propagation_error | %s", result["error"])
                break
            if max_propagate and len(prop_results) >= max_propagate:
                sam.cancel_propagation(bench_sid)
                break

        t_prop_end = time.time()

        # Process results collected by persist_fn (guaranteed complete)
        for result in prop_results:
            fi = result["frame_idx"]
            frame_entry = {"frame_idx": fi, "masks": {}}
            for oid in object_ids:
                area = result["masks"].get(oid, {}).get("area", 0)
                per_object_areas[oid].append(area)
                frame_entry["masks"][oid] = {"area": area}
            per_frame.append(frame_entry)

        frame_count = len(prop_results)
        prop_total_ms = int((t_prop_end - t_prop_start) * 1000)
        avg_ms = prop_total_ms // frame_count if frame_count > 0 else 0
        fps = frame_count / (t_prop_end - t_prop_start) if t_prop_end > t_prop_start else 0

        memory["after_propagation"] = _mem_snapshot("after_propagation")

        # Object summary
        import numpy as np
        object_summary = {}
        for oid in object_ids:
            areas = per_object_areas[oid]
            nonzero = [a for a in areas if a > 0]
            object_summary[str(oid)] = {
                "avg_area": int(np.mean(nonzero)) if nonzero else 0,
                "lost_frames": len(areas) - len(nonzero),
                "total_frames": len(areas),
            }

        logger.info("benchmark | propagation_complete | frames=%d | total_ms=%d | avg_ms=%d | fps=%.2f",
                    frame_count, prop_total_ms, avg_ms, fps)

        # ── Cleanup ──
        sam.close_session(bench_sid)
        memory["after_close"] = _mem_snapshot("after_close")

        # ── Build response ──
        snap = sam.debug_snapshot()

        response = {
            "session_id": session_id,
            "resolution": f"{width}x{height}",
            "num_frames": num_frames,
            "device": snap["device"],
            "backend": snap["backend"],
            "model": snap["model"],
            "model_params": snap["model_params"],
            "results": {
                "init": {"time_ms": init_ms},
                "clicks": click_results,
                "propagation": {
                    "direction": "forward",
                    "total_frames": frame_count,
                    "total_time_ms": prop_total_ms,
                    "avg_ms_per_frame": avg_ms,
                    "fps": round(fps, 3),
                    "per_frame": per_frame,
                    "object_summary": object_summary,
                },
                "memory": memory,
                "memory_timeline": memory_timeline,
            },
        }

        logger.info("benchmark | done | session=%s | fps=%.2f | avg_ms=%d", session_id, fps, avg_ms)
        return jsonify(response)

    except Exception as e:
        sam.close_session(bench_sid)
        logger.error("benchmark | error | %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500
