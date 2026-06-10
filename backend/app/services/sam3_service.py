import contextlib
import logging
import os
import queue
import threading
import time
import traceback
from dataclasses import replace
from typing import Callable

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils

from app.config import SAM3_MODEL_ID, SAM3_DEVICE, SAM3_BACKEND, SESSIONS_DIR, get_session_cache
from app.services.pipeline import ServiceState, PipelineStep
from app.services.prompt_storage import load_all_prompts
from app.services.session_lock import session_io_lock

logger = logging.getLogger(__name__)


class LazyFrameLoader:
    """Load and normalize JPEG frames on demand for the native SAM3 predictor.

    Implements __getitem__/__len__ so it can replace the pre-loaded tensor
    in inference_state["images"]. Frames stay on CPU (offload_video_to_cpu=True)
    and are cached after first access so repeated reads are free.
    """

    def __init__(self, frames_dir: str, image_size: int):
        frame_names = [
            p for p in sorted(os.listdir(frames_dir))
            if os.path.splitext(p)[-1].lower() in (".jpg", ".jpeg")
        ]
        if not frame_names:
            raise RuntimeError(f"no images found in {frames_dir}")
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
        self._img_paths = [os.path.join(frames_dir, f) for f in frame_names]
        self._image_size = image_size
        self._img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
        self._img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
        self._cache: dict[int, torch.Tensor] = {}
        self.video_height: int | None = None
        self.video_width: int | None = None

    def __getitem__(self, index: int) -> torch.Tensor:
        if index in self._cache:
            return self._cache[index]
        img_pil = Image.open(self._img_paths[index]).convert("RGB")
        video_width, video_height = img_pil.size
        img_np = np.array(img_pil.resize((self._image_size, self._image_size)))
        if img_np.dtype == np.uint8:
            img_np = img_np / 255.0
        img = torch.from_numpy(img_np).permute(2, 0, 1).float()
        if self.video_height is None:
            self.video_height = video_height
            self.video_width = video_width
        normalized = (img - self._img_mean) / self._img_std
        self._cache[index] = normalized
        return normalized

    def __len__(self) -> int:
        return len(self._img_paths)


class LazyProcessedFrames(dict):
    """Load and process JPEG frames on demand for the HF SAM3 backend.

    Behaves like the dict[int, Tensor] that Sam3TrackerVideoInferenceSession
    expects for processed_frames. Frames are processed through the video
    processor on first access, then cached.
    """

    def __init__(self, frames_dir: str, num_frames: int, processor, dtype: torch.dtype,
                 storage_device: str):
        super().__init__()
        frame_files = sorted(
            [f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
            key=lambda p: int(os.path.splitext(p)[0]),
        )
        self._frame_paths = [os.path.join(frames_dir, f) for f in frame_files]
        self._num_frames = num_frames
        self._processor = processor
        self._dtype = dtype
        self._storage_device = storage_device

    def __missing__(self, frame_idx: int) -> torch.Tensor:
        img = Image.open(self._frame_paths[frame_idx]).convert("RGB")
        processed = self._processor.video_processor(
            videos=[img], device=self._storage_device, return_tensors="pt",
        )
        tensor = processed.pixel_values_videos[0][0].to(
            torch.device(self._storage_device), dtype=self._dtype,
        )
        self[frame_idx] = tensor
        return tensor

    def __len__(self) -> int:
        return self._num_frames

    def __contains__(self, key: object) -> bool:
        # Report presence only for actually-cached frames; __missing__ handles
        # lazy loading when __getitem__ is called for an uncached index.
        return super().__contains__(key)


# Save original autocast references before any patching.
# The HF/MPS backend needs autocast disabled (bfloat16 not supported on MPS),
# but the native CUDA backend needs it ENABLED (bfloat16 model + bfloat16
# activations = no dtype mismatch, and Triton kernels benefit from it).
_OrigAutocast = torch.amp.autocast
_OrigTorchAutocast = torch.autocast


def get_device(override: str = "auto") -> torch.device:
    if override != "auto":
        return torch.device(override)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _apply_mps_patches():
    """Patch HuggingFace Transformers for MPS compatibility.

    Known issues:
    1. pin_memory() fails on MPS — tensors can't be pinned to MPS memory
    2. torch.amp.autocast(device_type='mps') raises unsupported scalarType
    """
    import transformers.models.sam3_tracker_video.processing_sam3_tracker_video as proc_module

    # Patch 1: Remove pin_memory calls that crash on MPS.
    _orig_post_process = proc_module.Sam3TrackerVideoProcessor.post_process_masks

    def _patched_post_process(self, masks, original_sizes, **kwargs):
        try:
            return _orig_post_process(self, masks, original_sizes, **kwargs)
        except RuntimeError as e:
            if "pin_memory" in str(e) or "MPS" in str(e):
                logger.warning("MPS pin_memory workaround triggered in post_process_masks")
                device = masks.device if hasattr(masks, 'device') else None
                if device and device.type == 'mps':
                    masks_cpu = masks.cpu()
                    result = _orig_post_process(self, masks_cpu, original_sizes, **kwargs)
                    if hasattr(result, 'to'):
                        return result.to(device)
                    return result
            raise

    proc_module.Sam3TrackerVideoProcessor.post_process_masks = _patched_post_process

    # Patch 2: Disable autocast on MPS — bfloat16 is not supported.
    # Replace torch.autocast with a no-op so SAM3's @autocast decorators
    # and context managers don't try to cast to bfloat16 on MPS.
    class _NoOpAutocast:
        """Replacement for torch.autocast that does nothing."""
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def __call__(self, func):
            return func

    torch.amp.autocast = _NoOpAutocast
    torch.autocast = _NoOpAutocast
    logger.info("Applied MPS patches (pin_memory, autocast disabled) for SAM3")


def _disable_autocast():
    """Disable torch.autocast globally (no-op replacement).

    Used by non-native backends where the model is float32 but SAM3's
    internal code applies autocast decorators that create bfloat16 activations.
    """
    class _NoOpAutocast:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def __call__(self, func):
            return func

    torch.amp.autocast = _NoOpAutocast
    torch.autocast = _NoOpAutocast


def _resolve_backend(device: torch.device) -> str:
    """Determine which SAM3 backend to use based on config and device."""
    if SAM3_BACKEND == "native":
        return "native"
    if SAM3_BACKEND == "hf":
        return "hf"
    # auto: native on CUDA, HF everywhere else
    if device.type == "cuda":
        return "native"
    return "hf"


def _put_critical(q: "queue.Queue", item, session_id: str, kind: str) -> None:
    """Put a critical item (error/sentinel) on a subscriber queue, guaranteeing delivery.

    Per-frame propagation results are lossy by design under backpressure, but the
    terminal sentinel (None) and error events must reach the subscriber so the SSE
    stream terminates cleanly. On queue.Full, drop the oldest item and retry. The
    final put_nowait is still wrapped in try/except because the subscriber may have
    exited between the get and the put, but in practice the retry always succeeds
    since this is the only thread writing to the queue.
    """
    try:
        q.put_nowait(item)
    except queue.Full:
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
            logger.warning(
                "sam3 | subscriber queue full, dropped oldest to deliver %s | session=%s",
                kind, session_id,
            )
        except queue.Full:
            logger.error(
                "sam3 | failed to deliver %s event after drain | session=%s",
                kind, session_id,
            )


class SAM3Service:
    """Manages SAM 3.1 video tracker lifecycle per session.

    Uses the singleton pattern — all SAM3Service() calls return the same
    instance so that session state is shared across Flask blueprints.

    Two backends:
    - "hf": HuggingFace Transformers (Sam3TrackerVideoModel) — MPS/CPU
    - "native": facebookresearch/sam3 native predictor — CUDA with Triton kernels

    The backend is auto-selected based on device (CUDA → native, else → hf)
    or forced via SAM3_BACKEND env var.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._backend = None                # "native" or "hf"
        self._model = None
        self._processor = None
        self._device = None
        # Native backend: video tracking predictor from facebookresearch/sam3
        self._native_predictor = None
        # Text-prompted model (Sam3VideoModel) — lazy-loaded on first text prompt
        self._text_model = None
        self._text_processor = None
        self._text_sessions = {}            # session_id -> Sam3VideoInferenceSession
        # Lock hierarchy: `SAM3._lock` is level #2 — acquire AFTER
        # `_state_lock`, BEFORE `session_io_lock` and anything
        # below it. RLock so the pipeline thread can re-enter via
        # init_session. Held for the ENTIRE propagation run (minutes),
        # so sites that take both `_state_lock` and `_lock` must be
        # audited against the start_pipeline probe pattern (see
        # `start_pipeline` for the canonical "probe _lock non-blocking
        # before taking _state_lock" inversion fix). See
        # docs/TECHNICAL_REPORT.md § Concurrency model.
        self._lock = threading.RLock()
        self._sessions = {}                 # session_id -> inference session (HF or native)
        self._session_meta = {}             # session_id -> {height, width, num_frames, frames_dir}
        self._active_object = {}            # session_id -> obj_id currently loaded
        # Lock hierarchy: `_propagation_lock` guards every read and write of
        # `_propagation_state`, `_propagation_subscribers`, and
        # `_propagation_threads` — not just the start-of-propagation gate
        # (R7/R30/R8 fix). Never held across any other lock in the
        # hierarchy — treated as a leaf for ordering purposes. It is safe
        # to take `_propagation_lock` briefly while holding `_lock` (e.g.
        # `close_session`, the per-frame fan-out in `_run_propagation`)
        # because no call site acquires `_lock` while holding
        # `_propagation_lock`. Iteration of the subscribers list is done
        # via a snapshot taken under the lock so the lock is released
        # before any `queue.put_nowait`. See docs/TECHNICAL_REPORT.md §
        # Concurrency model.
        self._propagation_lock = threading.Lock()
        self._propagation_state = {}        # session_id -> {status, start_frame, reverse, frames_processed}
        self._propagation_subscribers = {}  # session_id -> [queue.Queue, ...]
        self._cancel_events = {}            # session_id -> threading.Event
        self._propagation_threads: dict[str, threading.Thread] = {}  # session_id -> daemon thread handle

        # --- Service-level state (pipeline) ---
        # Lock hierarchy: `_state_lock` is level #1 (topmost). Held
        # briefly for `ServiceState` transitions. `finalize_close`
        # deliberately nests `_globals_lock` underneath it. See
        # docs/TECHNICAL_REPORT.md § Concurrency model.
        self._state_lock = threading.Lock()
        self._service_state = ServiceState()
        self._pipeline_thread: threading.Thread | None = None
        self._pipeline_cancel = threading.Event()

    @contextlib.contextmanager
    def _log_heartbeat_during(self, operation: str, interval_s: float = 5.0):
        """Emit a heartbeat log line every `interval_s` seconds until the
        block exits. R37: cold `_ensure_model` blocks `_lock` for ~30s on a
        fresh Cloud Run container; without these heartbeats the operator
        log shows a single "Loading native SAM 3 model" line and then
        silence until load completes, which is indistinguishable from a
        stall.
        """
        stop = threading.Event()
        t0 = time.time()

        def _tick() -> None:
            while not stop.wait(interval_s):
                elapsed = int(time.time() - t0)
                logger.info("sam3 | %s still in progress | elapsed=%ds", operation, elapsed)

        thread = threading.Thread(target=_tick, name=f"sam3-heartbeat-{operation}", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=interval_s)

    def _ensure_model(self, on_progress: Callable[[float], None] | None = None):
        if self._model is not None or self._native_predictor is not None:
            return

        self._device = get_device(SAM3_DEVICE)
        self._backend = _resolve_backend(self._device)

        # R37: on a cold container the load below takes ~30s on L4 and
        # holds `_lock` the whole time. Log a clear bookend + heartbeat so
        # operators reading Cloud Run logs can distinguish "loading model"
        # from "process stuck".
        logger.info(
            "sam3 | cold model load begins | backend=%s | device=%s",
            self._backend, self._device,
        )
        if on_progress:
            on_progress(0.15)  # device resolved, starting model load
        with self._log_heartbeat_during("model load"):
            if self._backend == "native":
                self._init_native_backend()
            else:
                self._init_hf_backend()
        if on_progress:
            on_progress(0.45)  # model loaded

    def _init_native_backend(self):
        """Use facebookresearch/sam3 native predictor for CUDA.

        Dtype strategy based on GPU compute capability:
        - Ampere+ (8.0+): native bfloat16 — no patches needed, SAM3 default path
        - Turing (7.5): float16 autocast — patch addmm_act, use FP16 Tensor Cores
        """
        # Detect GPU capability to choose dtype strategy
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(self._device)
            self._cuda_ampere_plus = cap[0] >= 8
        else:
            self._cuda_ampere_plus = False

        if self._cuda_ampere_plus:
            # Ampere+: native bfloat16, no patches needed
            self._native_autocast_dtype = torch.bfloat16
            logger.info("GPU compute capability %d.%d — using native bfloat16", cap[0], cap[1])
        else:
            # Turing/older: patch addmm_act to use float16 instead of bfloat16
            self._native_autocast_dtype = torch.float16
            logger.info("GPU compute capability %d.%d — patching to float16 (no native bfloat16)", cap[0], cap[1])

            import sam3.perflib.fused as fused_mod
            _orig_addmm_act_op = fused_mod.addmm_act_op

            def _fp16_addmm_act(activation, linear, mat1):
                bias = linear.bias.detach().to(torch.float16)
                weight = linear.weight.detach().to(torch.float16)
                mat1 = mat1.to(torch.float16)
                mat1_flat = mat1.view(-1, mat1.shape[-1])
                use_gelu = activation in [torch.nn.functional.gelu, torch.nn.GELU]
                y = _orig_addmm_act_op(bias, mat1_flat, weight.t(), beta=1, alpha=1, use_gelu=use_gelu)
                return y.view(mat1.shape[:-1] + (y.shape[-1],))

            fused_mod.addmm_act = _fp16_addmm_act
            try:
                import sam3.model.vitdet as vitdet_mod
                vitdet_mod.addmm_act = _fp16_addmm_act
            except ImportError:
                pass

        from sam3.model_builder import build_sam3_video_model

        logger.info("Loading native SAM 3 model on device %s (autocast %s)", self._device, self._native_autocast_dtype)
        t0 = time.time()

        sam3_model = build_sam3_video_model(device=str(self._device))
        if not self._cuda_ampere_plus:
            sam3_model = sam3_model.float()  # float32 weights for non-Ampere

        self._native_predictor = sam3_model.tracker
        self._native_predictor.backbone = sam3_model.detector.backbone

        param_count = sum(p.numel() for p in sam3_model.parameters()) / 1e6
        param_dtype = next(sam3_model.parameters()).dtype
        load_ms = int((time.time() - t0) * 1000)
        logger.info("Native SAM 3 loaded in %dms | device=%s | dtype=%s | params=%.0fM",
                     load_ms, self._device, param_dtype, param_count)

        # Store a reference for health/benchmark endpoints
        self._model = sam3_model

    def _init_hf_backend(self):
        """Use HuggingFace Transformers for MPS/CPU.

        The HF backend loads the model in float32. Autocast must be disabled
        to prevent bfloat16 activations from mismatching float32 weights.
        On MPS we also need pin_memory patches.
        """
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

        logger.info("Loading HF SAM 3.1 model %s on device %s", SAM3_MODEL_ID, self._device)

        if self._device.type == "mps":
            _apply_mps_patches()  # Disables autocast + fixes pin_memory
        else:
            # Non-MPS HF path (e.g., forced HF on CUDA) — still need to
            # disable autocast to prevent bfloat16/float32 mismatch.
            _disable_autocast()
            logger.info("Disabled autocast for HF backend on %s", self._device)

        t0 = time.time()
        self._model = Sam3TrackerVideoModel.from_pretrained(SAM3_MODEL_ID)
        self._model = self._model.to(self._device)
        self._model.eval()
        self._processor = Sam3TrackerVideoProcessor.from_pretrained(SAM3_MODEL_ID)
        load_ms = int((time.time() - t0) * 1000)
        logger.info("HF SAM 3.1 model loaded in %dms | device=%s | params=%s",
                     load_ms, self._device,
                     f"{sum(p.numel() for p in self._model.parameters()) / 1e6:.0f}M")

    def _ensure_text_model(self):
        """Lazy-load Sam3VideoModel for text-prompted segmentation.

        Load model and processor into locals and only commit them to `self`
        after both succeed. Otherwise a processor-load failure (e.g. HF Hub
        401 on gated facebook/sam3 while probing optional files such as
        chat_template.json) would leave `_text_model` set but
        `_text_processor=None`, and every subsequent call would short-circuit
        on the existence guard and raise a confusing
        'NoneType.video_processor' error instead of retrying.
        """
        if self._text_model is not None and self._text_processor is not None:
            return

        from transformers import Sam3VideoModel, Sam3VideoProcessor

        if self._device is None:
            self._device = get_device(SAM3_DEVICE)

        logger.info("Loading SAM 3.1 text model %s on device %s", SAM3_MODEL_ID, self._device)

        t0 = time.time()
        text_model = Sam3VideoModel.from_pretrained(SAM3_MODEL_ID)
        text_model = text_model.to(self._device)
        text_model.eval()
        text_processor = Sam3VideoProcessor.from_pretrained(SAM3_MODEL_ID)
        self._text_model = text_model
        self._text_processor = text_processor
        load_ms = int((time.time() - t0) * 1000)
        logger.info("SAM 3.1 text model loaded in %dms | device=%s | params=%s",
                     load_ms, self._device,
                     f"{sum(p.numel() for p in self._text_model.parameters()) / 1e6:.0f}M")

    def _process_text_frames_batched(self, frames: list[Image.Image], dtype: torch.dtype,
                                      storage_device: str, batch_size: int = 64) -> dict[int, torch.Tensor]:
        """Process frames through the text video processor in batches (prevents OOM)."""
        processed_frames = {}
        device = torch.device(storage_device)
        for start in range(0, len(frames), batch_size):
            batch = frames[start:start + batch_size]
            processed = self._text_processor.video_processor(
                videos=batch, device=storage_device, return_tensors="pt",
            )
            batch_tensor = processed.pixel_values_videos[0]
            for j in range(batch_tensor.shape[0]):
                processed_frames[start + j] = batch_tensor[j].to(device, dtype=dtype)
        return processed_frames

    def _load_frames_as_pil(self, frames_dir: str) -> list[Image.Image]:
        """Load JPEG frames from directory as PIL Images, sorted by filename."""
        frame_files = sorted([
            f for f in os.listdir(frames_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        if not frame_files:
            raise ValueError(f"No image frames found in {frames_dir}")
        frames = []
        for fname in frame_files:
            img = Image.open(os.path.join(frames_dir, fname)).convert("RGB")
            frames.append(img)
        return frames

    def _process_frames_batched(self, frames: list[Image.Image], dtype: torch.dtype,
                                storage_device: str, batch_size: int = 64) -> dict[int, torch.Tensor]:
        """Process PIL frames through the video processor in batches.

        The default HF path stacks all frames into one tensor before resizing,
        which OOMs on long videos (e.g., 1394 frames × 1920×1080 = ~32 GiB).
        This processes in chunks and builds the per-frame dict directly.
        """
        processed_frames = {}
        device = torch.device(storage_device)
        for start in range(0, len(frames), batch_size):
            batch = frames[start:start + batch_size]
            processed = self._processor.video_processor(
                videos=batch, device=storage_device, return_tensors="pt",
            )
            # pixel_values_videos shape: [1, batch_len, C, H, W]
            batch_tensor = processed.pixel_values_videos[0]
            for j in range(batch_tensor.shape[0]):
                processed_frames[start + j] = batch_tensor[j].to(device, dtype=dtype)
        return processed_frames

    def init_session(self, session_id: str, frames_dir: str, on_progress: Callable[[float], None] | None = None):
        # Guard: if a pipeline is currently doing extraction/init for a
        # DIFFERENT session, we cannot initialize yet — frames for this
        # session may not be complete. The route layer should check
        # /api/status and wait for 'ready', but we double-check here.
        with self._state_lock:
            phase = self._service_state.phase
            active_sid = self._service_state.session_id
        if phase in ("extracting", "initializing") and active_sid != session_id:
            raise RuntimeError(
                f"Cannot initialize session {session_id}: pipeline is "
                f"{phase} for session {active_sid}. Wait for ready phase."
            )

        with self._lock:
            # Fast path: session already loaded (check inside lock to avoid
            # TOCTOU race with close_session popping from self._sessions)
            if session_id in self._sessions:
                meta = self._session_meta[session_id]
                prop = self.get_propagation_status(session_id)
                return {
                    "num_frames": meta["num_frames"],
                    "video_height": meta["height"],
                    "video_width": meta["width"],
                    "propagation": prop if prop["status"] == "running" else None,
                }

            if on_progress:
                on_progress(0.1)  # signal that init has started
            self._ensure_model(on_progress)
            if on_progress:
                on_progress(0.5)
            t0 = time.time()

            if self._backend == "native":
                return self._init_session_native(session_id, frames_dir, t0, on_progress)
            else:
                return self._init_session_hf(session_id, frames_dir, t0, on_progress)

    def _init_session_native(self, session_id: str, frames_dir: str, t0: float, on_progress: Callable[[float], None] | None = None):
        """Initialize a session using the native facebookresearch/sam3 predictor."""
        # Lazy frame loading: build a LazyFrameLoader that reads JPEG frames
        # on demand instead of pre-loading all of them into a single tensor.
        # This drops init memory from O(num_frames) to O(1).
        lazy_images = LazyFrameLoader(frames_dir, self._native_predictor.image_size)
        # Load frame 0 to get video dimensions (matches init_state's warmup)
        lazy_images[0]
        height = lazy_images.video_height
        width = lazy_images.video_width
        num_frames = len(lazy_images)

        # Use the predictor's own init_state to build the inference_state dict.
        # This guarantees ALL required keys are present (tracking_has_started,
        # output_dict, consolidated_frame_inds, etc.) without us hand-copying
        # them — which previously caused KeyError crashes when SAM3 added keys
        # that our manual dict didn't have.
        # After init_state returns, we swap in our lazy loader for "images".
        with torch.inference_mode(), _OrigAutocast(device_type="cuda", dtype=self._native_autocast_dtype):
            inference_state = self._native_predictor.init_state(
                video_height=height,
                video_width=width,
                num_frames=num_frames,
                offload_video_to_cpu=True,
                offload_state_to_cpu=False,
            )
            if on_progress:
                on_progress(0.7)
            # Replace the (None) images with our lazy loader
            inference_state["images"] = lazy_images
            # Warm up backbone on frame 0 (same as the video_path init path does)
            self._native_predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)
            if on_progress:
                on_progress(1.0)

        self._sessions[session_id] = inference_state
        self._session_meta[session_id] = {
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "frames_dir": frames_dir,
        }
        init_ms = int((time.time() - t0) * 1000)
        logger.info(
            "sam3 | init_session [native] | session=%s | device=%s | frames=%d | "
            "resolution=%dx%d | init_time_ms=%d",
            session_id, self._device, num_frames, width, height, init_ms,
        )
        return {
            "num_frames": num_frames,
            "video_height": height,
            "video_width": width,
        }

    def _init_session_hf(self, session_id: str, frames_dir: str, t0: float, on_progress: Callable[[float], None] | None = None):
        """Initialize a session using HuggingFace Transformers."""
        # Read one frame to get dimensions without loading all frames.
        frame_files = sorted(
            [f for f in os.listdir(frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
            key=lambda p: int(os.path.splitext(p)[0]),
        )
        if not frame_files:
            raise ValueError(f"No image frames found in {frames_dir}")
        first_img = Image.open(os.path.join(frames_dir, frame_files[0])).convert("RGB")
        width, height = first_img.size
        num_frames = len(frame_files)

        # Use float32 everywhere — bfloat16 causes dtype mismatches during
        # propagation on CUDA ("mat1 and mat2 must have the same dtype,
        # but got BFloat16 and Float") and MPS doesn't support it at all.
        dtype = torch.float32

        from transformers.models.sam3_tracker_video.processing_sam3_tracker_video import (
            Sam3TrackerVideoInferenceSession,
        )

        # Memory strategy per device:
        # - video_storage: always on CPU (too large for GPU on long videos)
        # - inference_state: CPU for MPS (stress test showed 5.5s→70s/frame
        #   degradation on MPS vs steady 2.9s/frame on CPU), GPU for CUDA.
        storage_device = "cpu"
        if self._device.type == "cuda":
            state_device = str(self._device)
        else:
            state_device = "cpu"

        # Lazy frame loading: frames are processed through the video processor
        # on first access instead of all at once during init.
        lazy_frames = LazyProcessedFrames(
            frames_dir, num_frames, self._processor, dtype, storage_device,
        )
        inference_session = Sam3TrackerVideoInferenceSession(
            video=None,
            video_height=height,
            video_width=width,
            inference_device=str(self._device),
            video_storage_device=storage_device,
            inference_state_device=state_device,
            dtype=dtype,
        )
        if on_progress:
            on_progress(0.7)
        inference_session.processed_frames = lazy_frames
        self._sessions[session_id] = inference_session
        if on_progress:
            on_progress(1.0)
        self._session_meta[session_id] = {
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "frames_dir": frames_dir,
        }
        init_ms = int((time.time() - t0) * 1000)
        logger.info(
            "sam3 | init_session [hf] | session=%s | device=%s | frames=%d | "
            "resolution=%dx%d | init_time_ms=%d",
            session_id, self._device, num_frames, width, height, init_ms,
        )
        return {
            "num_frames": num_frames,
            "video_height": height,
            "video_width": width,
        }

    def add_click(self, session_id, frame_idx, obj_id, points, labels):
        with self._lock:
            session = self._sessions[session_id]
            meta = self._session_meta[session_id]
            t0 = time.time()

            if self._backend == "native":
                result = self._add_click_native(session, meta, frame_idx, obj_id, points, labels)
            else:
                result = self._add_click_hf(session, meta, frame_idx, obj_id, points, labels)

            inf_ms = int((time.time() - t0) * 1000)
            masks_summary = {k: {"area": v["area"], "conf": v["confidence"]}
                            for k, v in result["masks"].items()}
            logger.info(
                "sam3 | add_click [%s] | session=%s | frame=%d | obj=%d | "
                "points=%s | labels=%s | masks=%s | inference_ms=%d",
                self._backend, session_id, frame_idx, obj_id, points, labels,
                masks_summary, inf_ms,
            )
            return result

    def _add_click_native(self, state, meta, frame_idx, obj_id, points, labels):
        """Add click via native facebookresearch/sam3 predictor.

        Native API expects relative coordinates (0-1 range), so we convert
        absolute pixel coords to relative before passing.
        """
        h, w = meta["height"], meta["width"]
        rel_points = [[x / w, y / h] for x, y in points]
        points_tensor = torch.tensor(rel_points, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.int32)

        with torch.inference_mode(), _OrigAutocast(device_type="cuda", dtype=self._native_autocast_dtype):
            _, out_obj_ids, low_res_masks, video_res_masks = self._native_predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=int(obj_id),
                points=points_tensor,
                labels=labels_tensor,
                clear_old_points=True,
            )
        return self._format_native_mask_result(frame_idx, out_obj_ids, video_res_masks)

    def _add_click_hf(self, session, meta, frame_idx, obj_id, points, labels):
        """Add click via HuggingFace Transformers."""
        # Format: [image_level [object_level [point_level [x, y]]]]
        pt_list = [[[list(p) for p in points]]]
        lb_list = [[[int(l) for l in labels]]]

        self._processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=frame_idx,
            obj_ids=[int(obj_id)],
            input_points=pt_list,
            input_labels=lb_list,
            original_size=(meta["height"], meta["width"]),
            clear_old_inputs=True,
        )

        with torch.inference_mode():
            output = self._model(session, frame_idx=frame_idx)

        return self._format_mask_result(
            frame_idx, output.object_ids, output.pred_masks,
            meta["height"], meta["width"],
        )

    def add_box(self, session_id, frame_idx, obj_id, box):
        with self._lock:
            session = self._sessions[session_id]
            meta = self._session_meta[session_id]
            t0 = time.time()

            if self._backend == "native":
                result = self._add_box_native(session, meta, frame_idx, obj_id, box)
            else:
                result = self._add_box_hf(session, meta, frame_idx, obj_id, box)

            inf_ms = int((time.time() - t0) * 1000)
            masks_summary = {k: {"area": v["area"], "conf": v["confidence"]}
                            for k, v in result["masks"].items()}
            logger.info(
                "sam3 | add_box [%s] | session=%s | frame=%d | obj=%d | "
                "box=%s | masks=%s | inference_ms=%d",
                self._backend, session_id, frame_idx, obj_id, box, masks_summary, inf_ms,
            )
            return result

    def _add_box_native(self, state, meta, frame_idx, obj_id, box):
        """Add box via native predictor. Convert absolute [x1,y1,x2,y2] to relative."""
        h, w = meta["height"], meta["width"]
        rel_box = np.array([[box[0] / w, box[1] / h, box[2] / w, box[3] / h]], dtype=np.float32)

        with torch.inference_mode(), _OrigAutocast(device_type="cuda", dtype=self._native_autocast_dtype):
            _, out_obj_ids, low_res_masks, video_res_masks = self._native_predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=int(obj_id),
                box=rel_box,
            )
        return self._format_native_mask_result(frame_idx, out_obj_ids, video_res_masks)

    def _add_box_hf(self, session, meta, frame_idx, obj_id, box):
        """Add box via HuggingFace Transformers."""
        box_list = [[[float(v) for v in box]]]

        self._processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=frame_idx,
            obj_ids=[int(obj_id)],
            input_boxes=box_list,
            original_size=(meta["height"], meta["width"]),
            clear_old_inputs=True,
        )

        with torch.inference_mode():
            output = self._model(session, frame_idx=frame_idx)

        return self._format_mask_result(
            frame_idx, output.object_ids, output.pred_masks,
            meta["height"], meta["width"],
        )

    def add_mask(self, session_id, frame_idx, obj_id, binary_mask):
        """Register a binary mask in the tracker session for propagation.

        Used to bridge text-prompted masks (from Sam3VideoModel) into the
        tracker (Sam3TrackerVideoModel) so they can be propagated.
        """
        with self._lock:
            session = self._sessions[session_id]
            meta = self._session_meta[session_id]
            t0 = time.time()

            if self._backend == "native":
                result = self._add_mask_native(session, meta, frame_idx, obj_id, binary_mask)
            else:
                result = self._add_mask_hf(session, meta, frame_idx, obj_id, binary_mask)

            inf_ms = int((time.time() - t0) * 1000)
            logger.info(
                "sam3 | add_mask [%s] | session=%s | frame=%d | obj=%d | inference_ms=%d",
                self._backend, session_id, frame_idx, obj_id, inf_ms,
            )
            return result

    def _add_mask_native(self, state, meta, frame_idx, obj_id, binary_mask):
        """Add mask via native predictor."""
        mask_array = np.asarray(binary_mask, dtype=np.uint8)
        if mask_array.ndim == 3:
            mask_array = mask_array[:, :, 0]
        mask_tensor = torch.tensor(mask_array, dtype=torch.float32)
        with torch.inference_mode(), _OrigAutocast(device_type="cuda", dtype=self._native_autocast_dtype):
            _, out_obj_ids, low_res_masks, video_res_masks = self._native_predictor.add_new_mask(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=int(obj_id),
                mask=mask_tensor,
            )
        return self._format_native_mask_result(frame_idx, out_obj_ids, video_res_masks)

    def _add_mask_hf(self, session, meta, frame_idx, obj_id, binary_mask):
        """Add mask via HuggingFace Transformers."""
        mask_array = np.array(binary_mask, dtype=np.uint8)

        self._processor.add_inputs_to_inference_session(
            inference_session=session,
            frame_idx=frame_idx,
            obj_ids=[int(obj_id)],
            input_masks=[mask_array],
            original_size=(meta["height"], meta["width"]),
            clear_old_inputs=True,
        )

        with torch.inference_mode():
            output = self._model(session, frame_idx=frame_idx)

        return self._format_mask_result(
            frame_idx, output.object_ids, output.pred_masks,
            meta["height"], meta["width"],
        )

    def add_text_prompt(self, session_id, frame_idx, text):
        """Run text-prompted segmentation using Sam3VideoModel.

        Uses a separate model (Sam3VideoModel) from click/box (Sam3TrackerVideoModel).
        Detects all instances matching the text description on the given frame.

        Returns: {
            "frame_idx": int,
            "text": str,
            "instances": [
                {"obj_id": int, "rle": {...}, "bbox": [x,y,w,h], "area": int, "confidence": float},
                ...
            ]
        }
        """
        with self._lock:
            self._ensure_text_model()
            meta = self._session_meta.get(session_id)
            if meta is None:
                raise ValueError(f"Session {session_id} not initialized. Call init_session first.")

            t0 = time.time()

            # Get or create a text inference session for this session
            text_session = self._text_sessions.get(session_id)
            if text_session is None:
                frames = self._load_frames_as_pil(meta["frames_dir"])
                dtype = torch.float32
                if self._device.type == "cuda":
                    dtype = torch.bfloat16

                # Use batched frame processing to prevent OOM on long videos,
                # same pattern as the tracker session in init_session.
                from transformers.models.sam3_video.processing_sam3_video import (
                    Sam3VideoInferenceSession,
                )

                # Same hybrid strategy as tracker session
                if self._device.type == "cuda":
                    storage_device = str(self._device)
                    state_device = str(self._device)
                else:
                    storage_device = "cpu"
                    state_device = "cpu"

                processed_frames = self._process_text_frames_batched(
                    frames, dtype, storage_device,
                )
                text_session = Sam3VideoInferenceSession(
                    video=None,
                    video_height=meta["height"],
                    video_width=meta["width"],
                    inference_device=str(self._device),
                    video_storage_device=storage_device,
                    inference_state_device=state_device,
                    dtype=dtype,
                )
                text_session.processed_frames = processed_frames
                self._text_sessions[session_id] = text_session

            # Reset text session inference state before each new prompt.
            # The text model accumulates state from previous prompts, causing
            # errors on the second call if not reset.
            text_session.reset_inference_session()

            # Add text prompt
            self._text_processor.add_text_prompt(text_session, text)

            # Run inference on the target frame
            # Use autocast on CUDA to match dtype expectations (bfloat16 model + float32 internals)
            with torch.inference_mode():
                if self._device.type == "cuda":
                    with _OrigAutocast(device_type="cuda", dtype=self._native_autocast_dtype if hasattr(self, '_native_autocast_dtype') else torch.float16):
                        output = self._text_model(text_session, frame_idx=frame_idx)
                else:
                    output = self._text_model(text_session, frame_idx=frame_idx)

            # Post-process to get upscaled masks, boxes, scores, prompt_to_obj_ids
            post = self._text_processor.postprocess_outputs(
                text_session, output,
                original_sizes=[[meta["height"], meta["width"]]],
            )

            # Build result — one entry per detected instance for this text prompt
            instances = []
            prompt_obj_ids = post["prompt_to_obj_ids"].get(text, [])
            all_obj_ids = post["object_ids"].tolist()
            masks_tensor = post["masks"]       # (N, H, W) bool
            scores_tensor = post["scores"]     # (N,)
            boxes_tensor = post["boxes"]       # (N, 4) xyxy

            for obj_id in prompt_obj_ids:
                if obj_id not in all_obj_ids:
                    continue
                idx = all_obj_ids.index(obj_id)
                binary = masks_tensor[idx].cpu().numpy().astype(np.uint8)
                area = int(binary.sum())
                if area == 0:
                    continue
                score = round(float(scores_tensor[idx].cpu().item()), 4)
                # Convert xyxy box to COCO [x, y, w, h]
                box_xyxy = boxes_tensor[idx].cpu().tolist()
                bbox = [
                    int(box_xyxy[0]),
                    int(box_xyxy[1]),
                    int(box_xyxy[2] - box_xyxy[0]),
                    int(box_xyxy[3] - box_xyxy[1]),
                ]
                instances.append({
                    "obj_id": obj_id,
                    "rle": mask_to_rle(binary),
                    "bbox": bbox,
                    "area": area,
                    "confidence": score,
                })

            inf_ms = int((time.time() - t0) * 1000)
            logger.info(
                "sam3 | add_text_prompt | session=%s | frame=%d | text=%r | "
                "instances=%d | inference_ms=%d",
                session_id, frame_idx, text, len(instances), inf_ms,
            )
            return {
                "frame_idx": frame_idx,
                "text": text,
                "instances": instances,
            }

    def _get_registered_objects(self, session):
        """Return set of obj_ids currently registered in the inference state."""
        if self._backend == "native":
            return set(session.get("obj_id_to_idx", {}).keys())
        else:
            registered = set()
            for idx in range(session.get_obj_num()):
                registered.add(session.obj_idx_to_id(idx))
            return registered

    def _reset_inference_state(self, session):
        """Reset the inference state for both backends.

        Must clear BOTH tracking results AND object registrations.
        Without clearing registrations, ensure_active_object sees stale
        objects and propagation crashes at preflight.
        """
        if self._backend == "native":
            self._native_predictor._reset_tracking_results(session)
            self._native_predictor.clear_all_points_in_video(session)
            # Clear object registrations — without this, stale objects
            # remain registered and cause preflight errors during propagation.
            session["obj_id_to_idx"].clear()
            session["obj_idx_to_id"].clear()
            session["obj_ids"].clear()
        else:
            session.reset_inference_session()

    def reset_and_replay_objects(self, session_id: str, session_dir: str, object_ids: list[int] | None) -> None:
        """Reset inference state so all requested objects can be replayed fresh.

        SAM3's native predictor rejects new objects after tracking has started.
        For multi-object propagation, we must clear tracking results and object
        registrations first, then replay_prompts_if_needed adds them all cleanly.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            registered = self._get_registered_objects(session)
            if registered:
                logger.info(
                    "sam3 | reset_for_multi_object | session=%s | clearing %d objects for replay of %s",
                    session_id, len(registered), object_ids,
                )
                self._reset_inference_state(session)
            self._active_object.pop(session_id, None)

    def replay_prompts_if_needed(self, session_id, session_dir, object_ids=None):
        """Replay stored prompts into SAM3 for objects missing from inference state."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return {"replayed": 0, "failed": 0, "skipped": 0}

            registered = self._get_registered_objects(session)

            if object_ids is not None:
                missing = [oid for oid in object_ids if oid not in registered]
                if not missing:
                    return {"replayed": 0, "failed": 0, "skipped": 0}
                needed_obj_ids = set(missing)
                logger.info("Objects %s missing from inference state, replaying prompts", missing)
            else:
                if registered:
                    return {"replayed": 0, "failed": 0, "skipped": 0}
                needed_obj_ids = None

            # Route through SessionCache so we see the latest committed write
            # without touching disk. Writers all save via the same cache, so
            # cache is authoritative. Avoids needing session_io_lock here —
            # we can't hold session_io_lock across the SAM3 predictor calls
            # below (lock hierarchy rule #1).
            all_prompts = load_all_prompts(session_dir, cache=get_session_cache())
            if not all_prompts:
                return {"replayed": 0, "failed": 0, "skipped": 0}

            replayed = 0
            failed = 0
            skipped = 0

            for frame_str, obj_prompts in all_prompts.items():
                frame_idx = int(frame_str)
                for obj_str, prompt in obj_prompts.items():
                    obj_id = int(obj_str)
                    if needed_obj_ids is not None and obj_id not in needed_obj_ids:
                        continue
                    try:
                        if prompt["type"] == "click":
                            self.add_click(
                                session_id, frame_idx, obj_id,
                                prompt["points"], prompt["labels"],
                            )
                        elif prompt["type"] == "box":
                            self.add_box(
                                session_id, frame_idx, obj_id,
                                prompt["box"],
                            )
                        elif prompt["type"] == "mask":
                            decoded = mask_utils.decode(prompt["rle"])
                            self.add_mask(
                                session_id, frame_idx, obj_id, decoded,
                            )
                        else:
                            skipped += 1
                            continue
                        replayed += 1
                    except Exception:
                        logger.warning(
                            "Failed to replay prompt frame=%d obj=%d: %s",
                            frame_idx, obj_id, traceback.format_exc(),
                        )
                        failed += 1

            logger.info(
                "sam3 | replay_prompts | session=%s | replayed=%d | failed=%d | skipped=%d",
                session_id, replayed, failed, skipped,
            )
            return {"replayed": replayed, "failed": failed, "skipped": skipped}

    def ensure_active_object(self, session_id, obj_id, session_dir):
        """Ensure inference state contains only the target object.

        No-op if obj_id is already the active object.
        Otherwise resets tracking and replays the target's prompts from disk.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return

            # Fast path: same object, already loaded
            registered = self._get_registered_objects(session)

            if (self._active_object.get(session_id) == obj_id
                    and obj_id in registered):
                return

            # Reset inference state to clear all objects
            if registered:
                logger.info(
                    "sam3 | active_object_switch | session=%s | %s -> %s | clearing %d objects",
                    session_id, self._active_object.get(session_id), obj_id,
                    len(registered),
                )
                self._reset_inference_state(session)

            self._active_object[session_id] = obj_id

            # Replay this object's prompts via SessionCache (R20). Cache
            # read matches the writer path
            # and avoids a session_io_lock acquisition that would invert the
            # "no session_io_lock across SAM3 predictor calls" rule.
            all_prompts = load_all_prompts(session_dir, cache=get_session_cache())
            if not all_prompts:
                return

            obj_key = str(obj_id)
            replayed = 0
            for frame_str, obj_prompts in all_prompts.items():
                prompt = obj_prompts.get(obj_key)
                if prompt is None:
                    continue
                frame_idx = int(frame_str)
                try:
                    if prompt["type"] == "click":
                        self.add_click(
                            session_id, frame_idx, obj_id,
                            prompt["points"], prompt["labels"],
                        )
                    elif prompt["type"] == "box":
                        self.add_box(
                            session_id, frame_idx, obj_id,
                            prompt["box"],
                        )
                    elif prompt["type"] == "mask":
                        decoded = mask_utils.decode(prompt["rle"])
                        self.add_mask(
                            session_id, frame_idx, obj_id, decoded,
                        )
                    replayed += 1
                except Exception:
                    logger.warning(
                        "Failed to replay prompt frame=%d obj=%d: %s",
                        frame_idx, obj_id, traceback.format_exc(),
                    )
            if replayed > 0:
                logger.info("sam3 | replayed %d prompts for object %d", replayed, obj_id)

    def get_propagation_status(self, session_id):
        # Snapshot under _propagation_lock so readers never observe a
        # torn dict mid-update (frames_processed increment, status
        # transitions). See R7/R30.
        with self._propagation_lock:
            state = self._propagation_state.get(session_id)
            if state:
                return dict(state)
        return {"status": "idle"}

    def start_propagation(self, session_id, start_frame, reverse, persist_fn,
                          object_ids=None, source_keyframe=None):
        if source_keyframe is None:
            source_keyframe = start_frame
        with self._propagation_lock:
            current = self._propagation_state.get(session_id)
            if current and current["status"] == "running":
                raise ValueError("Propagation already running for this session")

            self._propagation_state[session_id] = {
                "status": "running",
                "start_frame": start_frame,
                "reverse": reverse,
                "frames_processed": 0,
            }
            self._propagation_subscribers[session_id] = []
            self._cancel_events[session_id] = threading.Event()

        t = threading.Thread(
            target=self._run_propagation,
            args=(session_id, start_frame, reverse, persist_fn, object_ids, source_keyframe),
            daemon=True,
        )
        with self._propagation_lock:
            self._propagation_threads[session_id] = t
        t.start()

    def _run_propagation(self, session_id, start_frame, reverse, persist_fn,
                         object_ids=None, source_keyframe=None):
        logger.info(
            "sam3 | propagation_start [%s] | session=%s | start_frame=%s | "
            "reverse=%s | object_ids=%s | source_keyframe=%s",
            self._backend, session_id, start_frame, reverse, object_ids, source_keyframe,
        )
        cancel_event = self._cancel_events.get(session_id)
        from app.config import get_sync_manager
        sm = get_sync_manager()
        if sm:
            sm.set_propagating(True)
        try:
            with self._lock:
                session = self._sessions[session_id]
                meta = self._session_meta[session_id]

                if self._backend == "native":
                    frame_iter = self._propagate_native(session, start_frame, reverse, object_ids)
                else:
                    frame_iter = self._propagate_hf(session, meta, start_frame, reverse, object_ids)

                frame_t0 = time.time()
                for result in frame_iter:
                    if cancel_event and cancel_event.is_set():
                        break

                    frame_ms = int((time.time() - frame_t0) * 1000)
                    frame_idx = result["frame_idx"]

                    masks_summary = {k: v["area"] for k, v in result["masks"].items()}
                    logger.info(
                        "sam3 | propagate [%s] | session=%s | frame=%d | objects=%d | "
                        "frame_ms=%d | masks=%s",
                        self._backend, session_id, frame_idx, len(masks_summary), frame_ms,
                        masks_summary,
                    )
                    frame_t0 = time.time()

                    result["source_keyframe"] = source_keyframe
                    persist_fn(result)

                    # R8: snapshot under _propagation_lock (leaf), iterate
                    # without it. Releasing before put_nowait avoids holding
                    # the lock across queue ops; mutations elsewhere
                    # (subscribe/unsubscribe) take the same lock so the
                    # snapshot cannot mutate mid-iteration.
                    with self._propagation_lock:
                        subs_snapshot = list(self._propagation_subscribers.get(session_id, ()))
                    for q in subs_snapshot:
                        try:
                            q.put_nowait(result)
                        except queue.Full:
                            pass

                    # R7: per-frame counter update under _propagation_lock so
                    # a concurrent get_propagation_status reader observes a
                    # coherent snapshot rather than a torn read-modify-write.
                    with self._propagation_lock:
                        ps = self._propagation_state.get(session_id)
                        if ps:
                            ps["frames_processed"] = ps.get("frames_processed", 0) + 1

        except Exception as e:
            logger.error("sam3 | propagation_error | session=%s | error=%s", session_id, e)
            error_event = {"error": str(e), "traceback": traceback.format_exc()}
            # R8: snapshot subscribers under _propagation_lock before delivery.
            with self._propagation_lock:
                err_subs_snapshot = list(self._propagation_subscribers.get(session_id, ()))
            for q in err_subs_snapshot:
                _put_critical(q, error_event, session_id, "error")
            # Only overwrite if the session wasn't closed mid-run; otherwise
            # close_session already popped the entry and we must not resurrect it.
            with self._propagation_lock:
                if session_id in self._propagation_state:
                    self._propagation_state[session_id] = {"status": "failed", "error": str(e)}
        finally:
            if sm:
                sm.set_propagating(False)
            # R8: snapshot subscribers under _propagation_lock before delivery.
            with self._propagation_lock:
                final_subs_snapshot = list(self._propagation_subscribers.get(session_id, ()))
            for q in final_subs_snapshot:
                _put_critical(q, None, session_id, "sentinel")
            # Under _propagation_lock: if the session is still registered and
            # did not fail, drop the entry (get_propagation_status returns
            # {"status": "idle"} for missing keys, so pop is equivalent to
            # writing idle and avoids zombie entries for closed sessions).
            # R8: _propagation_subscribers.pop merged into the same lock block
            # so subscribe_propagation cannot install a queue into a list that
            # is about to be discarded.
            with self._propagation_lock:
                current = self._propagation_state.get(session_id)
                if current is not None and current.get("status") != "failed":
                    self._propagation_state.pop(session_id, None)
                self._propagation_subscribers.pop(session_id, None)
                self._propagation_threads.pop(session_id, None)
            self._cancel_events.pop(session_id, None)

    def _propagate_native(self, state, start_frame, reverse, object_ids):
        """Generator: propagate via native facebookresearch/sam3 predictor.

        Note: native SAM3 doesn't support per-object filtering in propagation
        (raises "Per-object tracking yet for batched inference not implemented").
        We propagate ALL objects and filter results after.
        """
        num_frames = state.get("num_frames", 10000)
        with torch.inference_mode(), _OrigAutocast(device_type="cuda", dtype=self._native_autocast_dtype):
            for (frame_idx, out_obj_ids, low_res_masks,
                 video_res_masks, obj_scores) in self._native_predictor.propagate_in_video(
                state,
                start_frame_idx=start_frame,
                max_frame_num_to_track=num_frames,
                reverse=reverse,
                propagate_preflight=True,
            ):
                result = self._format_native_mask_result(frame_idx, out_obj_ids, video_res_masks)

                # Filter to requested objects if specified
                if object_ids is not None:
                    result["masks"] = {
                        k: v for k, v in result["masks"].items()
                        if int(k) in object_ids
                    }
                yield result

    def _propagate_hf(self, session, meta, start_frame, reverse, object_ids):
        """Generator: propagate via HuggingFace Transformers."""
        with torch.inference_mode():
            for output in self._model.propagate_in_video_iterator(
                session,
                start_frame_idx=start_frame,
                reverse=reverse,
            ):
                result = self._format_mask_result(
                    output.frame_idx, output.object_ids, output.pred_masks,
                    meta["height"], meta["width"],
                )

                # Filter to requested objects if specified
                if object_ids is not None:
                    result["masks"] = {
                        k: v for k, v in result["masks"].items()
                        if int(k) in object_ids
                    }
                yield result

    def subscribe_propagation(self, session_id):
        # R8: install the queue under _propagation_lock so the
        # _run_propagation finally block cannot pop the list between the
        # check and the append (otherwise the queue would be orphaned and
        # the subscriber would block 30s on every q.get).
        q = queue.Queue(maxsize=50)
        with self._propagation_lock:
            subscribers = self._propagation_subscribers.get(session_id)
            if subscribers is None:
                return
            subscribers.append(q)
        try:
            while True:
                try:
                    result = q.get(timeout=30)
                except queue.Empty:
                    if self.get_propagation_status(session_id)["status"] != "running":
                        return
                    continue
                if result is None:
                    return
                yield result
        finally:
            # R8: remove under the same lock. ValueError still possible if
            # the propagation finally already popped the whole list (the
            # `subs is not None` branch will be skipped); harmless.
            with self._propagation_lock:
                subs = self._propagation_subscribers.get(session_id)
                if subs is not None:
                    try:
                        subs.remove(q)
                    except ValueError:
                        pass

    def cancel_propagation(self, session_id):
        event = self._cancel_events.get(session_id)
        if event:
            event.set()

    def join_propagation(self, session_id, timeout: float | None = None) -> bool:
        """Wait for the propagation thread for `session_id` to finish.

        Returns True if the thread is no longer alive (finished or never
        existed), False if the join timed out. Caller is responsible for
        having already called `cancel_propagation` if the goal is to cut
        the run short.
        """
        with self._propagation_lock:
            t = self._propagation_threads.get(session_id)
        if t is None:
            return True
        t.join(timeout=timeout)
        return not t.is_alive()

    def get_loaded_session_ids(self):
        return set(self._sessions.keys())

    def debug_snapshot(self) -> dict:
        """Return a thread-safe snapshot of model + session state for diagnostics.

        Acquires `_lock` briefly so the read of `_model`, `_native_predictor`,
        `_device`, `_backend`, and `_sessions` cannot tear against
        `_ensure_model` / `init_session` / `close_session` writers (R9). Does
        no I/O inside the lock — `parameters()` iteration touches Python-level
        metadata only.

        Returned dict is plain JSON-serialisable; callers (benchmark route)
        consume it without further locking.
        """
        with self._lock:
            device = str(self._device) if self._device else "not_loaded"
            backend = self._backend if self._backend else "unknown"
            if self._native_predictor is not None and self._model is not None:
                model_name = "Sam3TrackerPredictor (native)"
                model_params = f"{sum(p.numel() for p in self._model.parameters()) / 1e6:.0f}M"
            elif self._model is not None:
                model_name = type(self._model).__name__
                model_params = f"{sum(p.numel() for p in self._model.parameters()) / 1e6:.0f}M"
            else:
                model_name = "not_loaded"
                model_params = "0"
            return {
                "device": device,
                "backend": backend,
                "model": model_name,
                "model_params": model_params,
                "native_predictor_loaded": self._native_predictor is not None,
                "sessions_loaded": len(self._sessions),
                "loaded_session_ids": list(self._sessions.keys()),
            }

    def remove_object(self, session_id, obj_id):
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                if self._backend == "native":
                    self._native_predictor.remove_object(session, obj_id=obj_id, strict=False)
                else:
                    # Reset inference state — the active object model means we
                    # only ever have one object loaded, so a full reset is fine.
                    session.reset_inference_session()
                logger.info("sam3 | remove_object [%s] | session=%s | obj=%d",
                            self._backend, session_id, obj_id)
            if self._active_object.get(session_id) == obj_id:
                self._active_object.pop(session_id, None)

    def reset_session(self, session_id):
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                self._reset_inference_state(session)
            self._text_sessions.pop(session_id, None)
            self._active_object.pop(session_id, None)

    def close_session(self, session_id):
        # cancel_propagation must run BEFORE acquiring _lock: propagation
        # holds _lock for the whole run, so close_session can only enter
        # the lock once propagation observes the cancel event and exits.
        self.cancel_propagation(session_id)
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session is not None:
                self._reset_inference_state(session)
            self._text_sessions.pop(session_id, None)
            self._session_meta.pop(session_id, None)
            self._active_object.pop(session_id, None)
            # R7/R30/R8: _propagation_state and _propagation_subscribers
            # pop through _propagation_lock so concurrent
            # get_propagation_status readers, subscribe_propagation
            # appenders, and the propagation thread's finally block all see
            # a consistent view. Safe to nest under _lock because
            # _propagation_lock is a leaf.
            with self._propagation_lock:
                self._propagation_state.pop(session_id, None)
                self._propagation_subscribers.pop(session_id, None)
            self._cancel_events.pop(session_id, None)

        # Reset service state if this was the active session
        with self._state_lock:
            if (self._service_state.session_id == session_id
                    and self._service_state.phase == "ready"):
                self._service_state = ServiceState(phase="idle")

    def finalize_close(
        self,
        session_id: str,
        teardown: Callable[[], None],
    ) -> None:
        """Run `teardown` and transition service state to idle atomically
        under `_state_lock`.

        Mirror of the H5 _finalize_ready pattern on the close side. The
        close route's teardown (flush + stop + close_active_session) runs
        OUTSIDE `_state_lock` because it performs GCS I/O that can take
        seconds. But the final globals-clear (`close_active_session`) and
        the phase transition to idle must happen atomically, or a
        concurrent pipeline that has just installed globals + is about
        to set phase=ready can sneak its ready transition in AFTER we
        clear globals — leaving `phase=ready, globals=(None, None)` and
        silently dropping every subsequent `mark_dirty` write.

        The caller (close route) is responsible for:
          - Running the slow flush/stop logic BEFORE calling this helper.
          - Passing a `teardown` closure that performs the final cheap,
            non-blocking work: typically `close_active_session()`.

        Lock ordering: `_state_lock` -> `_globals_lock` (via teardown's
        `close_active_session`). This matches the H5 ready-path ordering
        so the two helpers cannot deadlock each other.

        State transition rule: transition to idle if the active service
        state matches `session_id` OR there is no active session. Never
        clobber a different session's state — a concurrent resume to a
        different session owns its own lifecycle.
        """
        with self._state_lock:
            try:
                teardown()
            finally:
                if self._service_state.session_id in (None, session_id):
                    self._service_state = ServiceState(phase="idle")

    # ---- Pipeline management ----

    def get_service_state(self) -> ServiceState:
        """Return the current service state (thread-safe read)."""
        with self._state_lock:
            return self._service_state

    def start_pipeline(
        self,
        session_id: str,
        video_name: str,
        steps: list[PipelineStep],
        on_complete: Callable[[], None] | None = None,
    ) -> tuple[bool, str]:
        """Start a background pipeline. Returns (success, reason).

        If `on_complete` is provided, it is invoked on the pipeline
        thread after all steps succeed and BEFORE the state transitions
        to "ready". Callers use this to install global state
        (sync manager, session cache) atomically after the session is
        fully hydrated but before the frontend observes the ready phase.
        The callback must not raise — exceptions are logged and then
        trigger the error-phase transition.
        """
        # Check propagation lock BEFORE entering state lock to avoid
        # inverted lock ordering (_state_lock -> _lock vs _lock -> _state_lock
        # in the progress callback path). TOCTOU gap is accepted — worst case
        # is a pipeline that fails during init_session with a clear error.
        if not self._lock.acquire(blocking=False):
            return False, "propagation_in_progress"
        self._lock.release()

        with self._state_lock:
            if self._service_state.phase not in ("idle", "error"):
                return False, "pipeline_already_running"

            self._pipeline_cancel.clear()
            self._service_state = ServiceState(
                phase=steps[0].phase,
                session_id=session_id,
                video_name=video_name,
            )

        self._pipeline_thread = threading.Thread(
            target=self._run_pipeline,
            args=(session_id, video_name, steps, on_complete),
            name=f"pipeline-{session_id[:8]}",
        )
        self._pipeline_thread.start()
        return True, "started"

    def cancel_pipeline(self) -> str:
        """Request cancellation of the active pipeline."""
        with self._state_lock:
            if self._service_state.phase in ("idle", "ready", "error"):
                return "nothing_to_cancel"
            self._service_state = replace(
                self._service_state, cancel_requested=True
            )
            self._pipeline_cancel.set()
        return "cancel_requested"

    def dismiss_error(self) -> str:
        """Transition from error → idle so the UI can return to the session list.

        No-op when not in error phase. Never touches an active pipeline.
        """
        with self._state_lock:
            if self._service_state.phase != "error":
                return "not_in_error"
            self._service_state = ServiceState(phase="idle")
        return "dismissed"

    def _make_progress_fn(self, session_id: str) -> Callable[[float], None]:
        """Create a thread-safe progress callback bound to a session."""
        def update(progress: float) -> None:
            with self._state_lock:
                if self._service_state.session_id == session_id:
                    self._service_state = replace(
                        self._service_state, progress=progress
                    )
        return update

    def _run_pipeline(
        self,
        session_id: str,
        video_name: str,
        steps: list[PipelineStep],
        on_complete: Callable[[], None] | None = None,
    ) -> None:
        """Execute pipeline steps sequentially in a background thread."""
        try:
            for step in steps:
                with self._state_lock:
                    # Generation check: exit if superseded or cancelled
                    if self._service_state.session_id != session_id:
                        return
                    if self._service_state.cancel_requested:
                        self._cleanup_partial(session_id)
                        self._service_state = ServiceState(phase="idle")
                        return
                    self._service_state = replace(
                        self._service_state,
                        phase=step.phase,
                        progress=0.0,
                    )

                result = step.run(
                    self._make_progress_fn(session_id),
                    self._pipeline_cancel,
                )

                # Merge step results (e.g. frame_count, video_name) into state
                if result:
                    logger.info("pipeline | merge_result | session=%s | %s", session_id, result)
                    with self._state_lock:
                        if self._service_state.session_id == session_id:
                            self._service_state = replace(
                                self._service_state, **result
                            )

            # Final cancel + generation checks, on_complete (globals
            # install), and the ready-phase transition must all happen
            # under a single _state_lock critical section. Without this,
            # a concurrent close_session between on_complete returning
            # and the ready transition would observe phase="initializing"
            # (not "ready", so the close-route phase-reset is skipped),
            # clear the globals to (None, None), and leave the pipeline
            # thread to then set phase="ready" — frontend sees a ready
            # session with no sync manager, and mark_dirty_safe silently
            # drops every subsequent write.
            #
            # Safety of holding _state_lock across on_complete:
            #  - on_complete takes `_globals_lock` (via install_active_session)
            #    which is disjoint; no existing code path takes _globals_lock
            #    then waits on _state_lock, so no new deadlock ordering.
            #  - on_complete's GCSSyncManager.start() schedules a timer
            #    but does not block.
            #  - The critical section is short (no GCS I/O — the GCS work
            #    in _stop_and_record is for the OUTGOING manager which
            #    was cleared by clear_active_session_for_resume at the
            #    start of the resume route).
            with self._state_lock:
                if self._service_state.cancel_requested:
                    self._cleanup_partial(session_id)
                    self._service_state = ServiceState(phase="idle")
                    return
                # Generation check: another session has superseded us.
                # Do not install that session's globals from this thread.
                if self._service_state.session_id != session_id:
                    return

                if on_complete is not None:
                    try:
                        on_complete()
                    except Exception as e:
                        logger.error(
                            "pipeline | on_complete raised | session=%s",
                            session_id, exc_info=e,
                        )
                        # Still holding _state_lock — safe to mutate
                        # state directly (avoid lock re-acquisition).
                        if self._service_state.session_id == session_id:
                            self._service_state = ServiceState(
                                phase="error",
                                session_id=session_id,
                                video_name=video_name,
                                error=f"post-pipeline install failed: {e}",
                            )
                        return

                if self._service_state.session_id == session_id:
                    self._service_state = replace(
                        self._service_state,
                        phase="ready",
                        progress=1.0,
                    )
            logger.info(
                "pipeline | complete | session=%s | video=%s",
                session_id, video_name,
            )
        except Exception as e:
            logger.error(
                "pipeline | error | session=%s | %s", session_id, e,
                exc_info=e,
            )
            with self._state_lock:
                if self._service_state.session_id == session_id:
                    self._service_state = ServiceState(
                        phase="error",
                        session_id=session_id,
                        video_name=video_name,
                        error=str(e),
                    )

    def _cleanup_partial(self, session_id: str) -> None:
        """Remove a partially-created session directory.

        Only removes if the session has no annotation state (state.json) AND
        has no complete frame extraction (frames dir missing or empty). This
        prevents deleting valid duplicate sessions that were reused.

        R29: the check-then-rmtree window is serialised against concurrent
        writers (state.json RMW, frame extraction, mask/prompt persistence)
        by holding `session_io_lock(session_id)` across the whole block. Lock
        hierarchy: callers hold `_state_lock` and this method never
        takes SAM3 `_lock`, so acquiring `session_io_lock` here
        respects the documented order (see docs/TECHNICAL_REPORT.md §
        Concurrency model).
        """
        import shutil
        session_dir = os.path.join(SESSIONS_DIR, session_id)
        with session_io_lock(session_id):
            if not os.path.isdir(session_dir):
                return
            # Never delete sessions that have been annotated
            if os.path.exists(os.path.join(session_dir, "state.json")):
                return
            # Never delete sessions with completed frame extraction
            frames_dir = os.path.join(session_dir, "frames")
            if os.path.isdir(frames_dir):
                frame_count = len([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
                if frame_count > 0:
                    return
            shutil.rmtree(session_dir, ignore_errors=True)
            logger.info("pipeline | cleanup_partial | session=%s", session_id)

    def clear_frame_object(self, session_id, frame_idx, obj_id):
        """Clear in-memory state for a single object on a single frame.

        Both backends reset the entire inference state (no per-frame clearing).
        This prevents propagation from regenerating deleted masks from stale memory.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            self._reset_inference_state(session)
            self._active_object.pop(session_id, None)

    def clear_frame_object_batch(self, session_id, frame_indices, obj_id):
        """Clear in-memory state for an object across multiple frames."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return
            self._reset_inference_state(session)
            self._active_object.pop(session_id, None)

    def _format_native_mask_result(self, frame_idx, obj_ids, video_res_masks):
        """Convert native predictor output to the standard FrameResult format.

        video_res_masks: tensor at original video resolution, logit values
        (positive = foreground, negative = background). Shape: [N, 1, H, W]

        When the model runs in bfloat16, logits arrive as bfloat16.
        Convert to float32 before sigmoid to ensure accurate thresholding.
        """
        results = {}
        if video_res_masks is not None and video_res_masks.numel() > 0:
            for i, obj_id in enumerate(obj_ids):
                if i >= video_res_masks.shape[0]:
                    continue
                # video_res_masks shape: [N, 1, H, W] — take first channel
                # Convert to float32 for accurate sigmoid + thresholding
                logits = video_res_masks[i, 0].float()
                probs = torch.sigmoid(logits)
                binary = (probs > 0.5).cpu().numpy().astype(np.uint8)
                area = int(binary.sum())
                if area > 0:
                    mask_bool = probs > 0.5
                    confidence = round(float(probs[mask_bool].mean().cpu().item()), 4)
                else:
                    confidence = None
                results[int(obj_id)] = {
                    "rle": mask_to_rle(binary),
                    "bbox": mask_to_bbox(binary),
                    "area": area,
                    "confidence": confidence,
                }
        return {"frame_idx": frame_idx, "masks": results}

    def _format_mask_result(self, frame_idx, obj_ids, pred_masks, height, width):
        """Convert HF model output to the standard FrameResult format.

        pred_masks: tensor of logit masks from the model
        Returns: {"frame_idx": int, "masks": {obj_id: {rle, bbox, area, confidence}}}
        """
        results = {}

        # Upscale low-res masks to original resolution if needed
        # pred_masks shape: [num_objects, num_masks, low_H, low_W]
        if pred_masks is not None and pred_masks.numel() > 0:
            if pred_masks.shape[-2:] != (height, width):
                masks_tensor = torch.nn.functional.interpolate(
                    pred_masks, size=(height, width),
                    mode="bilinear", align_corners=False,
                )
            else:
                masks_tensor = pred_masks

            for i, obj_id in enumerate(obj_ids):
                if i >= masks_tensor.shape[0]:
                    continue
                # Get logits for this object — shape [num_masks, H, W]
                logits = masks_tensor[i]
                if logits.dim() == 3:
                    logits = logits[0]  # Take best mask (first channel)

                probs = torch.sigmoid(logits)
                binary = (probs > 0.5).cpu().numpy().astype(np.uint8)
                area = int(binary.sum())

                if area > 0:
                    mask_bool = probs > 0.5
                    confidence = round(float(probs[mask_bool].mean().cpu().item()), 4)
                else:
                    confidence = None

                results[int(obj_id)] = {
                    "rle": mask_to_rle(binary),
                    "bbox": mask_to_bbox(binary),
                    "area": area,
                    "confidence": confidence,
                }

        return {"frame_idx": frame_idx, "masks": results}


# --- Utility functions (shared with exporter) ---

def mask_to_polygons(binary_mask: np.ndarray) -> list:
    mask_uint8 = (binary_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        if len(contour) < 3:
            continue
        epsilon = 0.001 * cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(contour, epsilon, closed=True)
        simplified = approx.squeeze(axis=1)
        if simplified.ndim == 2 and len(simplified) >= 3:
            polygons.append(simplified.flatten().tolist())
    return polygons


def mask_to_bbox(binary_mask: np.ndarray) -> list:
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if not rows.any():
        return [0, 0, 0, 0]
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [int(x_min), int(y_min), int(x_max - x_min + 1), int(y_max - y_min + 1)]


def mask_to_rle(binary_mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    rle["size"] = [int(s) for s in rle["size"]]
    return rle
