# Harness Test Fixture

Assets for the verification harness in [USER-FLOWS.md](../../../USER-FLOWS.md).

## Contents

| File | What it is |
|---|---|
| `sample.mp4` | 8 s, 360×640, 30 fps — Breville Barista Express espresso machine (first 8 s of `IMG_4130.MOV`, the RF-DETR dataset v2 source video; see `rf-detr-vision/finetuning/TRAINING_LOG.md`). Camera slowly zooms from full-machine view toward the pressure gauge and portafilter; both stay fully visible in every frame. |
| `golden_session.zip` | Known-good exported session (`sam3-annotator-session` v1, no video): 2 classes, 2 objects, masks on all 40 frames, 2 prompts, bbox padding on object 1's keyframe. Doubles as the import asset for UF-8.2. |
| `fixture.json` | Canonical values for harness steps: click/box coordinates on keyframe 0, object-ID mapping, text query + expected detection count, IoU thresholds, propagation sample frames, bbox padding. |
| `iou.py` | Canonical IoU computation — decodes RLE with pycocotools (the backend's own encoder format, `_encode_mask` in `backend/app/services/mask_storage.py`). All IoU values in USER-FLOWS.md are computed with this script. |

## How the goldens were produced (2026-06-10, MPS / HF Transformers backend)

1. `ffmpeg -i ~/Downloads/IMG_4130.MOV -t 8 -vf "scale=-2:640" -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p -an -movflags +faststart sample.mp4`
2. Upload: `POST /api/video/upload` with `fps=5` → 40 frames extracted.
3. Classes: `POST /api/session/classes` × 2 → `pressure gauge` (id 1), `portafilter` (id 2).
4. Click segment obj 1: `POST /api/segment/click` `{"frame_idx":0,"obj_id":1,"points":[[160,296]],"labels":[1]}` → mask area 1758, confidence 0.96.
5. Box segment obj 2: `POST /api/segment/box` `{"frame_idx":0,"obj_id":2,"box":[168,345,225,460]}` → mask area 3193, confidence 0.98.
6. Register objects + bbox padding: `PUT /api/session/state` (objects 1→class 1, 2→class 2; padding `{"1":{"0":{top:10,bottom:10,left:5,right:5}}}`).
7. Multi-object propagation: `POST /api/segment/propagate` `{"start_frame_idx":0,"reverse":false,"object_ids":[1,2]}` → 40 FrameResults, both objects in every event, confidence 0.96–0.99 throughout (no low-confidence frames).
8. Export: `POST /api/export/session` `{"include_video":false}` → `golden_session.zip`.
9. `expected_text_detections` measured: `POST /api/segment/text` with `"pressure gauge"` on frame 0 → exactly 1 instance (bbox within 1 px of the click-segmented gauge). Measured after the export, so the golden zip contains only the 2 canonical objects.

## Regenerating

Delete `golden_session.zip` and `fixture.json`, then re-run steps 2–9 (step 1 only if `sample.mp4` itself changes — that also changes `video_md5` and invalidates import-dedup assumptions). Masks were generated on MPS (HF Transformers backend); CUDA-generated masks will differ slightly — that is why all golden comparisons use IoU ≥ 0.80, never byte equality (see USER-FLOWS.md § Test fixture).

## Notes

- The conda env on this machine is `sam2-annotator` (despite the repo name).
- The propagation run produced zero sub-0.75-confidence frames, so UF-4.1 step 7 exercises its "no banner" branch with this fixture.
