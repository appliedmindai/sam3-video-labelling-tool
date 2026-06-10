#!/usr/bin/env python3
"""End-to-end CLI test for SAM 3.1 video annotation pipeline.

Exercises the full pipeline without the web UI:
  1. Uploads a video (or uses existing session frames)
  2. Initializes SAM 3.1 session
  3. Runs click/box segmentation
  4. Runs propagation
  5. Validates mask outputs
  6. Optionally saves overlay PNGs for visual verification

Usage:
    # Test with a video file (extracts frames first)
    python3 backend/scripts/test_sam3.py --video path/to/video.mp4

    # Test with existing session frames directory
    python3 backend/scripts/test_sam3.py --frames-dir backend/sessions/<id>/frames

    # Save overlay PNGs for visual inspection
    python3 backend/scripts/test_sam3.py --video path/to/video.mp4 --save-overlays --output-dir /tmp/sam3-test

    # Quick smoke test (model load + single click only)
    python3 backend/scripts/test_sam3.py --video path/to/video.mp4 --quick
"""
import argparse
import json
import logging
import os
import sys
import tempfile
import time

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_sam3")


def extract_test_frames(video_path: str, output_dir: str, fps: int = 2, max_frames: int = 30):
    """Extract frames from video for testing."""
    from app.services.video_processor import extract_frames
    info = extract_frames(video_path, output_dir, fps=fps)
    # Limit frames for faster testing
    frames = sorted(os.listdir(output_dir))
    if len(frames) > max_frames:
        for f in frames[max_frames:]:
            os.remove(os.path.join(output_dir, f))
    return len(min(frames, key=lambda x: x) if frames else []), info


def save_overlay(frame_path: str, rle_mask: dict, output_path: str, color=(0, 255, 0), alpha=0.4):
    """Save a frame with mask overlay for visual verification."""
    frame = cv2.imread(frame_path)
    if frame is None:
        return
    mask = mask_utils.decode(rle_mask).astype(bool)
    overlay = frame.copy()
    overlay[mask] = (np.array(color) * alpha + overlay[mask] * (1 - alpha)).astype(np.uint8)
    # Draw contour
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, 2)
    cv2.imwrite(output_path, overlay)


class SAM3Tester:
    def __init__(self, frames_dir: str, output_dir: str = None, save_overlays: bool = False):
        self.frames_dir = frames_dir
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="sam3-test-")
        self.save_overlays = save_overlays
        self.session_id = "test-session"
        self.service = None
        self.results = {"passed": 0, "failed": 0, "errors": []}

        # Get frame list
        self.frame_files = sorted([
            f for f in os.listdir(frames_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        assert len(self.frame_files) > 0, f"No frames found in {frames_dir}"

        # Get frame dimensions
        img = Image.open(os.path.join(frames_dir, self.frame_files[0]))
        self.width, self.height = img.size
        logger.info("Test setup: %d frames, %dx%d, output=%s",
                     len(self.frame_files), self.width, self.height, self.output_dir)

    def _check(self, name: str, condition: bool, msg: str = ""):
        if condition:
            self.results["passed"] += 1
            logger.info("  PASS: %s %s", name, msg)
        else:
            self.results["failed"] += 1
            self.results["errors"].append(f"{name}: {msg}")
            logger.error("  FAIL: %s %s", name, msg)

    def test_model_load(self):
        """Test that the SAM 3.1 model loads successfully."""
        logger.info("=== Test: Model Load ===")
        from app.services.sam3_service import SAM3Service
        t0 = time.time()
        self.service = SAM3Service()
        # Force singleton reset for clean testing
        SAM3Service._instance = None
        SAM3Service._instance = self.service
        self.service._initialized = True
        self.service._ensure_model()
        load_time = time.time() - t0
        self._check("model_loaded", self.service._model is not None, f"in {load_time:.1f}s")
        self._check("processor_loaded", self.service._processor is not None)
        self._check("device_set", self.service._device is not None, f"device={self.service._device}")

    def test_init_session(self):
        """Test session initialization with video frames."""
        logger.info("=== Test: Init Session ===")
        t0 = time.time()
        result = self.service.init_session(self.session_id, self.frames_dir)
        init_time = time.time() - t0
        self._check("session_init", result is not None, f"in {init_time:.1f}s")
        self._check("num_frames", result["num_frames"] == len(self.frame_files),
                     f"expected={len(self.frame_files)} got={result['num_frames']}")
        self._check("video_height", result["video_height"] == self.height)
        self._check("video_width", result["video_width"] == self.width)

    def test_click_segmentation(self):
        """Test click-based segmentation on the first frame."""
        logger.info("=== Test: Click Segmentation ===")
        # Click in the center of the frame
        cx, cy = self.width // 2, self.height // 2
        t0 = time.time()
        result = self.service.add_click(
            session_id=self.session_id,
            frame_idx=0,
            obj_id=1,
            points=[[cx, cy]],
            labels=[1],
        )
        inf_time = time.time() - t0
        self._check("click_returns_result", result is not None, f"in {inf_time:.3f}s")
        self._check("click_has_frame_idx", result["frame_idx"] == 0)
        self._check("click_has_masks", len(result["masks"]) > 0)

        if 1 in result["masks"]:
            mask_data = result["masks"][1]
            self._check("click_has_rle", "rle" in mask_data)
            self._check("click_has_bbox", "bbox" in mask_data)
            self._check("click_has_area", "area" in mask_data)
            self._check("click_mask_area_positive", mask_data["area"] > 0,
                         f"area={mask_data['area']}")
            self._check("click_has_confidence", mask_data["confidence"] is not None,
                         f"confidence={mask_data.get('confidence')}")

            # Verify RLE decodes to valid binary mask
            decoded = mask_utils.decode(mask_data["rle"])
            self._check("click_rle_shape", decoded.shape == (self.height, self.width),
                         f"shape={decoded.shape}")
            self._check("click_rle_binary", set(np.unique(decoded)).issubset({0, 1}))

            if self.save_overlays:
                frame_path = os.path.join(self.frames_dir, self.frame_files[0])
                overlay_path = os.path.join(self.output_dir, "click_overlay_frame0.png")
                save_overlay(frame_path, mask_data["rle"], overlay_path)
                logger.info("  Saved overlay: %s", overlay_path)

    def test_box_segmentation(self):
        """Test box-based segmentation."""
        logger.info("=== Test: Box Segmentation ===")
        # Box covering center quarter of frame
        x1, y1 = self.width // 4, self.height // 4
        x2, y2 = 3 * self.width // 4, 3 * self.height // 4

        # Use a different object ID
        t0 = time.time()
        result = self.service.add_box(
            session_id=self.session_id,
            frame_idx=0,
            obj_id=2,
            box=[x1, y1, x2, y2],
        )
        inf_time = time.time() - t0
        self._check("box_returns_result", result is not None, f"in {inf_time:.3f}s")
        self._check("box_has_masks", len(result["masks"]) > 0)

        if 2 in result["masks"]:
            mask_data = result["masks"][2]
            self._check("box_mask_area_positive", mask_data["area"] > 0,
                         f"area={mask_data['area']}")

            if self.save_overlays:
                frame_path = os.path.join(self.frames_dir, self.frame_files[0])
                overlay_path = os.path.join(self.output_dir, "box_overlay_frame0.png")
                save_overlay(frame_path, mask_data["rle"], overlay_path, color=(255, 0, 0))
                logger.info("  Saved overlay: %s", overlay_path)

    def test_propagation(self):
        """Test forward propagation from frame 0."""
        logger.info("=== Test: Forward Propagation ===")
        if len(self.frame_files) < 3:
            logger.warning("  SKIP: Need at least 3 frames for propagation test")
            return

        # Reset and add a fresh click for propagation
        self.service.reset_session(self.session_id)
        self.service.init_session(self.session_id, self.frames_dir)

        cx, cy = self.width // 2, self.height // 2
        self.service.add_click(self.session_id, 0, 1, [[cx, cy]], [1])

        propagated_frames = []
        errors = []

        def persist_fn(result):
            propagated_frames.append(result)

        t0 = time.time()
        self.service.start_propagation(
            self.session_id, start_frame=0, reverse=False,
            persist_fn=persist_fn, object_ids=[1],
        )

        # Collect results from subscriber
        results_from_sse = []
        for result in self.service.subscribe_propagation(self.session_id):
            if "error" in result:
                errors.append(result["error"])
                break
            results_from_sse.append(result)

        prop_time = time.time() - t0
        self._check("propagation_no_errors", len(errors) == 0,
                     f"errors={errors}" if errors else "")
        self._check("propagation_frames_received", len(propagated_frames) > 0,
                     f"got {len(propagated_frames)} frames in {prop_time:.1f}s")
        self._check("propagation_sse_matches", len(results_from_sse) == len(propagated_frames),
                     f"sse={len(results_from_sse)} persist={len(propagated_frames)}")

        if propagated_frames:
            # Check first propagated frame
            first = propagated_frames[0]
            self._check("prop_has_source_keyframe", "source_keyframe" in first)
            if 1 in first.get("masks", {}):
                self._check("prop_first_has_mask", first["masks"][1]["area"] > 0)

            # Save propagation overlays
            if self.save_overlays:
                os.makedirs(os.path.join(self.output_dir, "propagation"), exist_ok=True)
                for result in propagated_frames[:10]:  # First 10 frames
                    fi = result["frame_idx"]
                    if fi < len(self.frame_files) and 1 in result.get("masks", {}):
                        frame_path = os.path.join(self.frames_dir, self.frame_files[fi])
                        overlay_path = os.path.join(
                            self.output_dir, "propagation", f"frame_{fi:04d}.png")
                        save_overlay(frame_path, result["masks"][1]["rle"], overlay_path)
                logger.info("  Saved %d propagation overlays", min(10, len(propagated_frames)))

    def test_text_segmentation(self, text_query: str = "object"):
        """Test text-prompted segmentation using Sam3VideoModel."""
        logger.info("=== Test: Text Segmentation (query=%r) ===", text_query)
        t0 = time.time()
        result = self.service.add_text_prompt(
            session_id=self.session_id,
            frame_idx=0,
            text=text_query,
        )
        inf_time = time.time() - t0
        self._check("text_returns_result", result is not None, f"in {inf_time:.3f}s")
        self._check("text_has_frame_idx", result["frame_idx"] == 0)
        self._check("text_has_text", result["text"] == text_query)
        self._check("text_has_instances", isinstance(result["instances"], list))

        instances = result["instances"]
        if len(instances) > 0:
            logger.info("  Found %d instance(s) for %r", len(instances), text_query)
            for i, inst in enumerate(instances):
                self._check(f"text_inst{i}_has_obj_id", "obj_id" in inst,
                             f"obj_id={inst.get('obj_id')}")
                self._check(f"text_inst{i}_has_rle", "rle" in inst)
                self._check(f"text_inst{i}_has_bbox", "bbox" in inst,
                             f"bbox={inst.get('bbox')}")
                self._check(f"text_inst{i}_area_positive", inst.get("area", 0) > 0,
                             f"area={inst.get('area')}")
                self._check(f"text_inst{i}_has_confidence", inst.get("confidence") is not None,
                             f"confidence={inst.get('confidence')}")

                # Verify RLE decodes correctly
                decoded = mask_utils.decode(inst["rle"])
                self._check(f"text_inst{i}_rle_shape",
                             decoded.shape == (self.height, self.width),
                             f"shape={decoded.shape}")

                if self.save_overlays and i < 3:
                    frame_path = os.path.join(self.frames_dir, self.frame_files[0])
                    overlay_path = os.path.join(
                        self.output_dir,
                        f"text_{text_query}_inst{i}_frame0.png",
                    )
                    save_overlay(frame_path, inst["rle"], overlay_path, color=(0, 128, 255))
                    logger.info("  Saved overlay: %s", overlay_path)
        else:
            logger.warning("  No instances found for %r — this may be expected for generic videos", text_query)

    def test_close_session(self):
        """Test session cleanup."""
        logger.info("=== Test: Close Session ===")
        self.service.close_session(self.session_id)
        self._check("session_closed", self.session_id not in self.service._sessions)
        self._check("text_session_closed", self.session_id not in self.service._text_sessions)

    def run_all(self, quick: bool = False, text_query: str = "object"):
        """Run all tests and report results."""
        logger.info("Starting SAM 3.1 end-to-end test")
        logger.info("=" * 60)

        try:
            self.test_model_load()
            self.test_init_session()
            self.test_click_segmentation()
            if not quick:
                self.test_box_segmentation()
                self.test_text_segmentation(text_query)
                self.test_propagation()
            self.test_close_session()
        except Exception as e:
            logger.error("Test suite crashed: %s", e, exc_info=True)
            self.results["errors"].append(f"CRASH: {e}")
            self.results["failed"] += 1

        logger.info("=" * 60)
        logger.info("Results: %d passed, %d failed", self.results["passed"], self.results["failed"])
        if self.results["errors"]:
            logger.error("Failures:")
            for err in self.results["errors"]:
                logger.error("  - %s", err)

        # Write results JSON
        results_path = os.path.join(self.output_dir, "test_results.json")
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info("Results written to %s", results_path)

        return self.results["failed"] == 0


def main():
    parser = argparse.ArgumentParser(description="SAM 3.1 end-to-end test")
    parser.add_argument("--video", help="Path to test video file")
    parser.add_argument("--frames-dir", help="Path to existing frames directory")
    parser.add_argument("--output-dir", default="/tmp/sam3-test", help="Output directory")
    parser.add_argument("--save-overlays", action="store_true", help="Save overlay PNGs")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test only")
    parser.add_argument("--fps", type=int, default=2, help="Frame extraction FPS")
    parser.add_argument("--max-frames", type=int, default=20, help="Max frames to test with")
    parser.add_argument("--text-query", default="object", help="Text prompt to test (default: 'object')")
    args = parser.parse_args()

    if not args.video and not args.frames_dir:
        parser.error("Either --video or --frames-dir is required")

    os.makedirs(args.output_dir, exist_ok=True)

    frames_dir = args.frames_dir
    if args.video:
        frames_dir = os.path.join(args.output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        logger.info("Extracting frames from %s at %d FPS (max %d)", args.video, args.fps, args.max_frames)
        extract_test_frames(args.video, frames_dir, fps=args.fps, max_frames=args.max_frames)
        logger.info("Extracted %d frames to %s", len(os.listdir(frames_dir)), frames_dir)

    tester = SAM3Tester(
        frames_dir=frames_dir,
        output_dir=args.output_dir,
        save_overlays=args.save_overlays,
    )
    success = tester.run_all(quick=args.quick, text_query=args.text_query)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
