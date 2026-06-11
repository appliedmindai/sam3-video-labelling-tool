#!/usr/bin/env python3
"""Canonical IoU for the USER-FLOWS.md harness.

Decodes two RLE masks with the backend's own decoder (pycocotools COCO
compressed RLE — the exact format `_encode_mask` in
backend/app/services/mask_storage.py produces) and prints their IoU.

Usage:
    python3 iou.py '<rle_json_a>' '<rle_json_b>'
    python3 iou.py --file masks_a.json --file masks_b.json --frame 0 --obj 1

An RLE JSON is {"counts": "<str>", "size": [h, w]} — either bare (old format)
or wrapped as {"rle": {...}, "source_keyframe": ...} (new format); both are
accepted, matching `unpack_entry` in mask_storage.py.
"""
import argparse
import json
import sys

import numpy as np
from pycocotools import mask as mask_utils


def unwrap(entry):
    if "rle" in entry:
        return entry["rle"]
    return entry


def decode(entry):
    rle = dict(unwrap(entry))
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("utf-8")
    return mask_utils.decode(rle)


def iou(a, b):
    ma, mb = decode(a).astype(bool), decode(b).astype(bool)
    if ma.shape != mb.shape:
        raise SystemExit(f"shape mismatch: {ma.shape} vs {mb.shape}")
    union = np.logical_or(ma, mb).sum()
    if union == 0:
        return 1.0  # both empty
    return float(np.logical_and(ma, mb).sum()) / float(union)


def from_masks_file(path, frame, obj):
    data = json.load(open(path))
    return data[str(frame)][str(obj)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rle", nargs="*", help="two RLE JSON strings")
    p.add_argument("--file", action="append", default=[], help="masks.json path (give twice)")
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--obj", type=int, default=1)
    args = p.parse_args()

    if len(args.file) == 2:
        a = from_masks_file(args.file[0], args.frame, args.obj)
        b = from_masks_file(args.file[1], args.frame, args.obj)
    elif len(args.rle) == 2:
        a, b = json.loads(args.rle[0]), json.loads(args.rle[1])
    else:
        p.error("give two RLE JSON strings, or --file twice with --frame/--obj")

    print(f"{iou(a, b):.4f}")


if __name__ == "__main__":
    main()
