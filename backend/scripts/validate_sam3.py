#!/usr/bin/env python3
"""Comprehensive SAM 3.1 validation suite.

Goes beyond the basic smoke test to validate:
  1. Multiple objects on the same frame
  2. Object switching (active object model)
  3. Forward AND reverse propagation
  4. Mask deletion → re-propagation (no stale state)
  5. Session lifecycle (init, close, re-init)
  6. Different video resolutions
  7. Edge cases (click near border, empty mask, very small object)
  8. Mask persistence round-trip (save → load → compare)

Usage:
    python3 backend/scripts/validate_sam3.py --video path/to/video.mp4
"""
import argparse
import json
import logging
import os
import sys
import tempfile
import time

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-20s %(levelname)-5s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validate_sam3")


class ValidationSuite:
    def __init__(self, frames_dir: str, output_dir: str):
        self.frames_dir = frames_dir
        self.output_dir = output_dir
        self.frame_files = sorted([
            f for f in os.listdir(frames_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        img = Image.open(os.path.join(frames_dir, self.frame_files[0]))
        self.width, self.height = img.size
        self.passed = 0
        self.failed = 0
        self.errors = []

        from app.services.sam3_service import SAM3Service
        self.service = SAM3Service()

    def check(self, name, condition, msg=""):
        if condition:
            self.passed += 1
            logger.info("  PASS %s %s", name, msg)
        else:
            self.failed += 1
            self.errors.append(f"{name}: {msg}")
            logger.error("  FAIL %s %s", name, msg)

    # ── Test 1: Multi-object on same frame ──────────────────────────
    def test_multi_object(self):
        logger.info("=== Test: Multiple Objects on Same Frame ===")
        sid = "val-multi-obj"
        self.service.init_session(sid, self.frames_dir)

        # Click in top-left quadrant (object 1)
        r1 = self.service.add_click(sid, 0, 1, [[self.width // 4, self.height // 4]], [1])
        self.check("multi_obj1_has_mask", 1 in r1["masks"], f"area={r1['masks'].get(1, {}).get('area')}")

        # Click in bottom-right quadrant (object 2)
        r2 = self.service.add_click(sid, 0, 2, [[3 * self.width // 4, 3 * self.height // 4]], [1])
        self.check("multi_obj2_has_mask", 2 in r2["masks"], f"area={r2['masks'].get(2, {}).get('area')}")

        # Masks should not overlap (or at least be different)
        if 1 in r1["masks"] and 2 in r2["masks"]:
            m1 = mask_utils.decode(r1["masks"][1]["rle"])
            m2 = mask_utils.decode(r2["masks"][2]["rle"])
            overlap = np.logical_and(m1, m2).sum()
            union = np.logical_or(m1, m2).sum()
            self.check("multi_obj_different", overlap < union * 0.5,
                        f"overlap={overlap} union={union}")

        self.service.close_session(sid)

    # ── Test 2: Object switching (active object model) ──────────────
    def test_object_switching(self):
        logger.info("=== Test: Object Switching ===")
        sid = "val-obj-switch"
        session_dir = tempfile.mkdtemp(prefix="val-prompts-")
        self.service.init_session(sid, self.frames_dir)

        from app.services.prompt_storage import save_prompt

        # Add click for object 1
        r1 = self.service.add_click(sid, 0, 1, [[self.width // 4, self.height // 4]], [1])
        save_prompt(session_dir, 0, 1, {"type": "click", "points": [[self.width // 4, self.height // 4]], "labels": [1]})
        area1 = r1["masks"].get(1, {}).get("area", 0)
        self.check("switch_obj1_initial", area1 > 0, f"area={area1}")

        # Switch to object 2 via ensure_active_object
        self.service.ensure_active_object(sid, 2, session_dir)
        r2 = self.service.add_click(sid, 0, 2, [[3 * self.width // 4, 3 * self.height // 4]], [1])
        save_prompt(session_dir, 0, 2, {"type": "click", "points": [[3 * self.width // 4, 3 * self.height // 4]], "labels": [1]})
        area2 = r2["masks"].get(2, {}).get("area", 0)
        self.check("switch_obj2_after_switch", area2 > 0, f"area={area2}")

        # Switch back to object 1 — should replay from stored prompts
        self.service.ensure_active_object(sid, 1, session_dir)
        # After replay, object 1 should be in inference state
        # Verify by running a propagation with object 1
        if len(self.frame_files) >= 3:
            results = []
            self.service.start_propagation(sid, 0, False, lambda r: results.append(r), object_ids=[1])
            for r in self.service.subscribe_propagation(sid):
                if "error" in r:
                    break
            self.check("switch_obj1_replayed", len(results) > 0,
                        f"propagated {len(results)} frames after switch-back")

        self.service.close_session(sid)

    # ── Test 3: Forward + Reverse propagation ───────────────────────
    def test_bidirectional_propagation(self):
        logger.info("=== Test: Bidirectional Propagation ===")
        if len(self.frame_files) < 5:
            logger.warning("  SKIP: need 5+ frames")
            return

        sid = "val-bidir"
        self.service.init_session(sid, self.frames_dir)

        # Click on middle frame
        mid = len(self.frame_files) // 2
        self.service.add_click(sid, mid, 1, [[self.width // 2, self.height // 2]], [1])

        # Forward propagation
        fwd_results = []
        self.service.start_propagation(sid, mid, False, lambda r: fwd_results.append(r), object_ids=[1])
        for r in self.service.subscribe_propagation(sid):
            if "error" in r:
                break
        self.check("bidir_fwd_results", len(fwd_results) > 0, f"fwd={len(fwd_results)} frames")

        # Need to re-init for reverse since propagation modifies state
        self.service.reset_session(sid)
        self.service.init_session(sid, self.frames_dir)
        self.service.add_click(sid, mid, 1, [[self.width // 2, self.height // 2]], [1])

        # Reverse propagation
        rev_results = []
        self.service.start_propagation(sid, mid, True, lambda r: rev_results.append(r), object_ids=[1])
        for r in self.service.subscribe_propagation(sid):
            if "error" in r:
                break
        self.check("bidir_rev_results", len(rev_results) > 0, f"rev={len(rev_results)} frames")

        # Forward should go to end, reverse should go to start
        if fwd_results:
            fwd_frames = [r["frame_idx"] for r in fwd_results]
            self.check("bidir_fwd_direction", all(f >= mid for f in fwd_frames),
                        f"frames={fwd_frames}")
        if rev_results:
            rev_frames = [r["frame_idx"] for r in rev_results]
            self.check("bidir_rev_direction", all(f <= mid for f in rev_frames),
                        f"frames={rev_frames}")

        self.service.close_session(sid)

    # ── Test 4: Mask deletion and clean re-propagation ──────────────
    def test_mask_deletion_no_stale_state(self):
        logger.info("=== Test: Mask Deletion (No Stale State) ===")
        if len(self.frame_files) < 3:
            logger.warning("  SKIP: need 3+ frames")
            return

        sid = "val-delete"
        self.service.init_session(sid, self.frames_dir)

        # Click and propagate
        self.service.add_click(sid, 0, 1, [[self.width // 2, self.height // 2]], [1])
        results1 = []
        self.service.start_propagation(sid, 0, False, lambda r: results1.append(r), object_ids=[1])
        for r in self.service.subscribe_propagation(sid):
            if "error" in r:
                break

        # Now clear frame object (simulates mask deletion)
        self.service.clear_frame_object(sid, 0, 1)

        # After clearing, inference state should be reset.
        # A new click on a DIFFERENT position should give a DIFFERENT mask.
        r_new = self.service.add_click(sid, 0, 1, [[self.width // 8, self.height // 8]], [1])
        new_area = r_new["masks"].get(1, {}).get("area", 0)
        old_area = results1[0]["masks"].get(1, {}).get("area", 0) if results1 else 0
        self.check("delete_clears_state", new_area != old_area,
                    f"old_area={old_area} new_area={new_area} (should differ)")

        self.service.close_session(sid)

    # ── Test 5: Session lifecycle ───────────────────────────────────
    def test_session_lifecycle(self):
        logger.info("=== Test: Session Lifecycle ===")
        sid = "val-lifecycle"

        # Init
        r = self.service.init_session(sid, self.frames_dir)
        self.check("lifecycle_init", r["num_frames"] == len(self.frame_files))

        # Re-init (should be fast path)
        t0 = time.time()
        r2 = self.service.init_session(sid, self.frames_dir)
        fast_ms = (time.time() - t0) * 1000
        self.check("lifecycle_reinit_fast", fast_ms < 100, f"{fast_ms:.0f}ms")

        # Close
        self.service.close_session(sid)
        self.check("lifecycle_closed", sid not in self.service._sessions)

        # Re-init after close (should work fresh)
        r3 = self.service.init_session(sid, self.frames_dir)
        self.check("lifecycle_reinit_after_close", r3["num_frames"] == len(self.frame_files))
        self.service.close_session(sid)

    # ── Test 6: Edge cases ──────────────────────────────────────────
    def test_edge_cases(self):
        logger.info("=== Test: Edge Cases ===")
        sid = "val-edge"
        self.service.init_session(sid, self.frames_dir)

        # Click near border (top-left corner)
        r1 = self.service.add_click(sid, 0, 1, [[5, 5]], [1])
        self.check("edge_corner_click", "masks" in r1)

        # Click with negative label (background point)
        r2 = self.service.add_click(sid, 0, 2,
                                     [[self.width // 2, self.height // 2],
                                      [self.width // 4, self.height // 4]],
                                     [1, 0])
        self.check("edge_neg_label", "masks" in r2)

        # Box segmentation
        r3 = self.service.add_box(sid, 0, 3, [10, 10, self.width - 10, self.height - 10])
        area3 = r3["masks"].get(3, {}).get("area", 0)
        self.check("edge_large_box", area3 > 0, f"area={area3}")

        # Very small box
        r4 = self.service.add_box(sid, 0, 4,
                                   [self.width // 2 - 5, self.height // 2 - 5,
                                    self.width // 2 + 5, self.height // 2 + 5])
        self.check("edge_tiny_box", "masks" in r4)

        self.service.close_session(sid)

    # ── Test 7: Mask RLE round-trip ─────────────────────────────────
    def test_mask_persistence_roundtrip(self):
        logger.info("=== Test: Mask Persistence Round-Trip ===")
        sid = "val-persist"
        self.service.init_session(sid, self.frames_dir)

        r = self.service.add_click(sid, 0, 1, [[self.width // 2, self.height // 2]], [1])
        if 1 not in r["masks"]:
            self.check("persist_has_mask", False, "no mask returned")
            self.service.close_session(sid)
            return

        original_rle = r["masks"][1]["rle"]
        original_area = r["masks"][1]["area"]

        # Decode and re-encode
        decoded = mask_utils.decode(original_rle)
        re_encoded = mask_utils.encode(np.asfortranarray(decoded))
        re_encoded["counts"] = re_encoded["counts"].decode("utf-8")
        re_encoded["size"] = [int(s) for s in re_encoded["size"]]

        # Re-decode
        re_decoded = mask_utils.decode(re_encoded)

        self.check("persist_roundtrip_lossless",
                    np.array_equal(decoded, re_decoded),
                    f"shape={decoded.shape}")
        self.check("persist_rle_matches",
                    original_rle["counts"] == re_encoded["counts"],
                    f"len_orig={len(original_rle['counts'])} len_re={len(re_encoded['counts'])}")

        # Save and load via mask_storage
        # update_frame_masks expects {obj_id: binary_np_array}, not the dict
        from app.services.mask_storage import update_frame_masks, load_frame_masks_rle
        tmp_session = tempfile.mkdtemp(prefix="val-storage-")
        update_frame_masks(tmp_session, 0, {1: decoded})
        loaded = load_frame_masks_rle(tmp_session, 0)
        self.check("persist_storage_roundtrip",
                    loaded.get("1", {}).get("rle", {}).get("counts") == original_rle["counts"])

        self.service.close_session(sid)

    # ── Test 8: Propagation mask continuity ─────────────────────────
    def test_propagation_continuity(self):
        logger.info("=== Test: Propagation Mask Continuity ===")
        if len(self.frame_files) < 5:
            logger.warning("  SKIP: need 5+ frames")
            return

        sid = "val-continuity"
        self.service.init_session(sid, self.frames_dir)
        self.service.add_click(sid, 0, 1, [[self.width // 2, self.height // 2]], [1])

        results = []
        self.service.start_propagation(sid, 0, False, lambda r: results.append(r), object_ids=[1])
        for r in self.service.subscribe_propagation(sid):
            if "error" in r:
                break

        if len(results) < 3:
            self.check("continuity_enough_frames", False, f"only {len(results)} frames")
            self.service.close_session(sid)
            return

        # Check mask areas don't jump wildly between consecutive frames
        areas = [r["masks"].get(1, {}).get("area", 0) for r in results]
        max_jump = 0
        for i in range(1, len(areas)):
            if areas[i - 1] > 0:
                jump = abs(areas[i] - areas[i - 1]) / areas[i - 1]
                max_jump = max(max_jump, jump)

        self.check("continuity_no_wild_jumps", max_jump < 5.0,
                    f"max_relative_jump={max_jump:.2f} areas={areas}")

        # Check confidences exist
        confs = [r["masks"].get(1, {}).get("confidence") for r in results]
        valid_confs = [c for c in confs if c is not None]
        self.check("continuity_has_confidences", len(valid_confs) > 0,
                    f"valid={len(valid_confs)}/{len(confs)}")

        self.service.close_session(sid)

    def run_all(self):
        logger.info("=" * 60)
        logger.info("SAM 3.1 Comprehensive Validation Suite")
        logger.info("Frames: %d @ %dx%d from %s", len(self.frame_files), self.width, self.height, self.frames_dir)
        logger.info("=" * 60)

        tests = [
            self.test_session_lifecycle,
            self.test_multi_object,
            self.test_object_switching,
            self.test_edge_cases,
            self.test_mask_persistence_roundtrip,
            self.test_bidirectional_propagation,
            self.test_mask_deletion_no_stale_state,
            self.test_propagation_continuity,
        ]

        for test in tests:
            try:
                test()
            except Exception as e:
                logger.error("CRASH in %s: %s", test.__name__, e, exc_info=True)
                self.errors.append(f"CRASH in {test.__name__}: {e}")
                self.failed += 1

        logger.info("=" * 60)
        logger.info("RESULTS: %d passed, %d failed", self.passed, self.failed)
        if self.errors:
            logger.error("FAILURES:")
            for e in self.errors:
                logger.error("  - %s", e)
        logger.info("=" * 60)

        results_path = os.path.join(self.output_dir, "validation_results.json")
        with open(results_path, "w") as f:
            json.dump({"passed": self.passed, "failed": self.failed, "errors": self.errors}, f, indent=2)
        logger.info("Results: %s", results_path)
        return self.failed == 0


def main():
    parser = argparse.ArgumentParser(description="SAM 3.1 comprehensive validation")
    parser.add_argument("--video", help="Path to video file")
    parser.add_argument("--frames-dir", help="Path to existing frames directory")
    parser.add_argument("--output-dir", default="/tmp/sam3-validation")
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=15)
    args = parser.parse_args()

    if not args.video and not args.frames_dir:
        parser.error("Either --video or --frames-dir required")

    os.makedirs(args.output_dir, exist_ok=True)

    frames_dir = args.frames_dir
    if args.video:
        frames_dir = os.path.join(args.output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        from app.services.video_processor import extract_frames
        extract_frames(args.video, frames_dir, fps=args.fps)
        all_frames = sorted(os.listdir(frames_dir))
        for f in all_frames[args.max_frames:]:
            os.remove(os.path.join(frames_dir, f))
        logger.info("Extracted %d frames", min(len(all_frames), args.max_frames))

    suite = ValidationSuite(frames_dir, args.output_dir)
    success = suite.run_all()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
