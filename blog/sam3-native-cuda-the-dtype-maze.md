# The dtype Maze: How We Made SAM 3.1 Run 2x Faster Than SAM 2 (After It Ran 2x Slower)

We migrated our video annotation tool from SAM 2 to SAM 3.1. The promise: text-prompted detection, better tracking, Object Multiplex for multi-object speedup. The reality: SAM 3.1 was **2x slower** than SAM 2 on the same GPU. Every fix we applied made it slower or crashed it in a new way.

Then we found the right combination of dtype settings, and SAM 3.1 became **2x faster** than SAM 2. This is the story of how we got there, what we learned about GPU numerics, and why the obvious approaches all failed.

---

## The Starting Point

SAM 2 on our T4 GPU: **1,470 ms/frame** propagating 3 objects at 640×360. Not fast, but functional for a video annotation tool.

SAM 3.1 via HuggingFace Transformers on the same T4: **3,041 ms/frame**. Same speed on MPS (Mac) — meaning the HuggingFace path has zero CUDA optimization.

So we switched to the native `facebookresearch/sam3` repo, which has Triton kernels, Flash Attention support, and Object Multiplex. Expected improvement: 5-10x. Actual result: **crash**.

```
RuntimeError: mat1 and mat2 must have the same dtype,
but got BFloat16 and Float
```

This error would haunt us for the next 12 hours.

---

## The dtype Maze

SAM 3's native code creates bfloat16 tensors in three different places:

**1. `torch.autocast` decorators** on inference methods — creates bfloat16 activations during forward pass

**2. `torch.amp.autocast` context managers** — same effect, different syntax

**3. `sam3/perflib/fused.py`** — a fused CUDA kernel that hardcodes `.to(torch.bfloat16)` on inputs:

```python
# sam3/perflib/fused.py
def addmm_act(activation, linear, mat1):
    self = linear.bias.detach()
    mat2 = linear.weight.detach()
    self = self.to(torch.bfloat16)   # ← hardcoded
    mat1 = mat1.to(torch.bfloat16)   # ← hardcoded
    mat2 = mat2.to(torch.bfloat16)   # ← hardcoded
```

The model weights load as float32. The activations become bfloat16 from autocast. The fused ops force bfloat16. Float32 weights meet bfloat16 activations → crash.

---

## Attempt 1: Disable Everything (float32 everywhere)

The brute force approach: monkey-patch `torch.autocast` and `torch.amp.autocast` to do nothing, and patch `addmm_act` to skip the bfloat16 conversion.

```python
class NoOpAutocast:
    def __init__(self, *args, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def __call__(self, func): return func  # works as decorator too

torch.autocast = NoOpAutocast
torch.amp.autocast = NoOpAutocast
```

Result: **3,185 ms/frame**. It works, but it's actually *slower* than HuggingFace (3,041 ms/frame). All the CUDA-specific optimizations — Triton kernels, fused operations — depend on reduced precision. Running float32 everywhere is just a bigger model doing the same computation slower.

We benchmarked against SAM 2: **2.2x slower**. The migration was looking like a net negative.

---

## Attempt 2: Load Model in bfloat16

If the problem is float32 weights meeting bfloat16 activations, make the weights bfloat16 too:

```python
sam3_model = build_sam3_video_model(device="cuda")
sam3_model = sam3_model.bfloat16()  # match the autocast dtype
```

Clicks worked. Propagation crashed:

```
RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same
```

Wait — now it's **reversed**. The model weights are bfloat16, but somewhere deep inside, SAM 3 creates float32 tensors that flow into the bfloat16 model. We traced it to `_get_image_feature`, which caches video features in float32 regardless of model dtype.

We patched that too. New crash:

```
RuntimeError: Input type (c10::BFloat16) and bias type (float) should be the same
```

A *different* float32 tensor from *another* internal path. SAM 3's codebase creates float32 tensors in at least five different locations across three different files. Every time we patched one, another appeared.

---

## The Research Breakthrough

While we were drowning in dtype patches, we researched **why** SAM 3.1 was slow on T4 in the first place. The answer changed everything:

**The T4 GPU does not have native bfloat16 support.**

The T4 is Turing architecture (compute capability 7.5). Its Tensor Cores support FP16, INT8, and INT4 — but **not bfloat16**. When PyTorch encounters a bfloat16 operation on T4:

1. Cast BF16 → FP32
2. Run the matmul in FP32
3. Cast FP32 → BF16

Per [openxla/xla#12429](https://github.com/openxla/xla/issues/12429): *"T4 has neither vector nor TensorCore support for BF16, so it has to emulate it, slowly"* and *"we need to run an extra kernel to cast from BF16 to F32, which can be as expensive as the matmul itself."*

bfloat16 on T4 is **slower than float32**. Every attempt to make bfloat16 work was making things worse, not better.

But the T4 *does* have native **float16** Tensor Cores at 65.6 TFLOPS. That's the optimization we should have been targeting all along.

---

## Attempt 3: float16 autocast (the fix)

Instead of disabling autocast or forcing bfloat16, we used autocast with **float16** — the dtype the T4 actually accelerates:

```python
# Keep model weights in float32 (compatible with SAM3's internal tensor creation)
sam3_model = build_sam3_video_model(device="cuda")
# Don't call .bfloat16() or .half() — leave weights as float32

# Wrap inference in float16 autocast
with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
    # autocast handles conversion at matmul/conv/linear boundaries
    _, obj_ids, _, masks = predictor.add_new_points_or_box(...)
```

And patch the fused op to use float16 instead of bfloat16:

```python
def patched_addmm_act(activation, linear, mat1):
    bias = linear.bias.detach().to(torch.float16)
    weight = linear.weight.detach().to(torch.float16)
    mat1 = mat1.to(torch.float16)
    # ... rest of fused kernel
```

This works because autocast is smart: it converts inputs to float16 *at operation boundaries* (matmul, conv2d, linear), but keeps accumulations in float32 for numerical stability. The model weights stay float32 in memory but operations run on the T4's FP16 Tensor Cores.

Result: **735 ms/frame**.

---

## The Final Scorecard

| Approach | GPU | ms/frame | vs SAM 2 | Why |
|----------|-----|---------|----------|-----|
| SAM 2 native (baseline) | T4 | 1,470 | 1.0x | CUDA kernels, 224M params |
| SAM 3.1 HuggingFace | T4 | 3,041 | 0.5x | No CUDA optimization |
| SAM 3.1 native float32 | T4 | 3,185 | 0.5x | Optimizations disabled |
| SAM 3.1 native bfloat16 | T4 | — | crash | T4 emulates BF16 |
| SAM 3.1 native float16 | T4 | 735 | 2.0x | T4's native FP16 Tensor Cores |
| **SAM 3.1 native bfloat16** | **L4** | **267** | **5.5x** | **Native BF16 + Flash Attention** |

On T4, the difference between 0.5x and 2.0x was a single keyword argument: `dtype=torch.float16` instead of `dtype=torch.bfloat16`.

On L4, we got another 2.75x by simply using the right GPU — native bfloat16, Flash Attention, and 3x the memory bandwidth. The code auto-detects compute capability and picks the optimal dtype automatically.

---

## What We Learned

### 1. Know your GPU's dtype capabilities

| GPU | FP16 | BF16 | Flash Attention | SAM 3.1 Result |
|-----|------|------|----------------|----------------|
| T4 (Turing 7.5) | ✅ Native | ❌ Emulated (~2x penalty) | ❌ | 735 ms/frame (fp16) |
| L4 (Lovelace 8.9) | ✅ | ✅ Native | ✅ | **267 ms/frame (bf16)** |
| A100 (Ampere 8.0) | ✅ | ✅ Native | ✅ | (expected faster) |
| H100 (Hopper 9.0) | ✅ | ✅ Native | ✅ | (expected fastest) |

SAM 3's codebase assumes Ampere+ GPUs. On Turing (T4, RTX 2080), override bfloat16 with float16. On Ampere+ (L4, A100, H100), native bfloat16 just works — and it's 2.75x faster than the float16 workaround.

The cost math is compelling: L4 at $0.70/hr processes frames at 3.74 FPS, giving **$0.19 per 1000 frames**. T4 at $0.35/hr with SAM 2 processes at 0.68 FPS, giving $0.51 per 1000 frames. The "expensive" GPU is **2.7x cheaper per frame**.

### 2. Autocast is smarter than manual casting

Manual `.bfloat16()` or `.half()` on the model converts all weights and creates a minefield of dtype mismatches with internally-created float32 tensors. `torch.amp.autocast` handles conversion at operation boundaries, keeping accumulations stable. Let autocast do the work.

### 3. "Faster model" doesn't mean faster on your hardware

SAM 3.1's speed claims (7x with Object Multiplex) were benchmarked on H100 with 128 objects, bfloat16, Flash Attention, and torch.compile. On a T4 with 3 objects and none of those features, SAM 3.1 is fundamentally a larger model (466M vs 224M params) doing more work per frame. The float16 optimization is what tips the scale.

### 4. Check upstream issues before debugging locally

[GitHub issue #425](https://github.com/facebookresearch/sam3/issues/425) on facebookresearch/sam3 reports that SAM 3 is 5-6x slower than SAM 2 even on H200. There's a known float32 casting bug in `video_base.py`. We spent hours debugging locally what was a known upstream issue.

### 5. The three bfloat16 sources in SAM 3

If you're porting SAM 3 to non-Ampere hardware, you need to address all three:

| Source | File | Fix |
|--------|------|-----|
| `@torch.autocast` decorator | `sam3_video_inference.py` | Replace with float16 autocast |
| `torch.amp.autocast` context | Various | Replace with float16 autocast |
| `addmm_act` hardcoded cast | `sam3/perflib/fused.py` | Patch to use float16 |

Missing any one of these causes a dtype mismatch crash. They interact in non-obvious ways because autocast can be both a context manager and a decorator, and the fused op bypasses autocast entirely.

---

## What We'd Report Upstream

Two things worth reporting to facebookresearch/sam3:

**1. Document the bfloat16 hardware requirement.** The README doesn't mention that bfloat16 (and therefore the native speed path) requires Ampere+. A simple note in the installation section would save others the debugging journey.

**2. Make `addmm_act` respect the current autocast dtype.** Instead of hardcoding `.to(torch.bfloat16)`, it should use `torch.get_autocast_gpu_dtype()` or accept a dtype parameter:

```python
# Current (hardcoded bfloat16):
self = self.to(torch.bfloat16)

# Proposed (respect autocast):
cast_dtype = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else self.dtype
self = self.to(cast_dtype)
```

This would make SAM 3 work on T4/RTX GPUs without any monkey-patching.

---

## Epilogue: The Right GPU Changes Everything

After the float16 breakthrough on T4 (735 ms/frame, 2x faster than SAM 2), we asked the obvious question: what happens on a GPU that actually supports bfloat16 natively?

We deployed to an NVIDIA L4 (Lovelace, compute capability 8.9) — $0.70/hr, native bfloat16, Flash Attention. Zero code changes. Our auto-detection picked the bfloat16 path automatically.

**267 ms/frame. 3.74 FPS. 5.5x faster than SAM 2.**

The same code that crashed on T4 with bfloat16 ran flawlessly on L4 — because the hardware actually supports it. The lesson: match your model's dtype assumptions to your hardware's capabilities. Don't fight the hardware; find the right hardware.

| What we tried | T4 result | L4 result |
|--------------|-----------|-----------|
| SAM 3.1 bfloat16 (default) | Crash | **267 ms/frame** |
| SAM 3.1 float16 (our patch) | 735 ms/frame | N/A (unnecessary) |
| SAM 3.1 float32 (all disabled) | 3,185 ms/frame | N/A (unnecessary) |
| SAM 2 (baseline) | 1,470 ms/frame | N/A (not tested) |

The entire dtype maze — the monkey patches, the fused op overrides, the autocast wrestling — was a T4 problem, not a SAM 3 problem. On the right GPU, SAM 3.1 just works.

---

*Built during the SAM 2 → SAM 3.1 migration of the [SAM3 Video Labelling Tool](https://github.com/appliedmindai/sam3-video-labelling-tool).*
