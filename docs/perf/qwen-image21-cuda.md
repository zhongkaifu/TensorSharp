# Qwen-Image-2.1 CUDA execution

TensorSharp keeps a complete Qwen-Image-2.1 velocity-prediction graph alive across
denoising steps. This removes repeated graph construction, allocation planning,
weight binding, and mask generation, and supplies stable tensor and device
addresses to unchanged upstream ggml's CUDA graph implementation.

The same graph-lifetime optimization is enabled by default on Metal. Metal
retains ggml metadata and scratch allocations; CUDA capture and executable
replay discussed below are specific to CUDA.

The implementation is in
[`ggml_ops_qwen_image21.cpp`](../../TensorSharp.GGML.Native/ggml_ops_qwen_image21.cpp).
It changes execution and storage, without skipping denoising steps, transformer
layers, or attention tokens. Floating-point differences between attention kernels
still require numerical and image validation.

## Graph and input lifetimes

Each CUDA or Metal device has at most two retained graphs, allowing positive and negative
CFG layouts to coexist. Each graph owns its ggml context and a dedicated gallocr
allocation. The key includes every weight pointer, type, shape and byte count;
normalization epsilon; token and head dimensions; all segment boundaries and
source offsets; backend identity; and the attention controls below. Host addresses
of dynamic inputs are not keys. Images, text conditioning, timestep embeddings,
and RoPE tables are uploaded from the current caller on every invocation.

The key also records the prefix-cache mode, the cache a graph reads, the
tensor-parallel rank count, and each applied LoRA's factor buffers (its scales
are folded into them), so a graph built for another adapter, or for none, is
never reused. With the default
[prefix KV cache](../models/qwenimage21.md#prefix-kv-cache)
(`TS_QWEN21_PREFIX_CACHE=0` disables it), a request's first step runs the whole
sequence and stores the text and reference-image prefix in a graph that is built
for that step and freed afterwards; it is never retained. Later steps run only
the target tokens against that stored prefix in the retained cached-step graph.
When the cache is declined or disabled, the full-sequence graph is the retained
one. A CFG run keeps one cache per branch. Under `--tp`, each rank retains its
own graphs; see
[CUDA graphs and tensor parallelism](../models/qwenimage21.md#cuda-graphs-and-tensor-parallelism).

Resident weight bindings are checked against the live weight-cache maps before
reuse. Weights denied a resident-cache allocation remain in that graph's own
allocation. These constant leaves and masks are marked as outputs as well as
inputs: ggml's allocator can recycle an ordinary input after its last consumer,
which would otherwise corrupt constants on the next execution.

Host-weight invalidation retires graphs using that pointer even when no shared
resident copy exists. Global host-cache clearing, reusable-scratch release and
backend shutdown retire all Qwen-Image-2.1 graphs. Native callers must invalidate
immutable weight pointers before changing or freeing their storage.

Before allocating another graph, TensorSharp measures its scratch requirement.
If current free device memory cannot cover it with 512 MiB of headroom, the other
slot is released. Consequently, CFG shapes may rebuild repeatedly on a device
that cannot hold both. Two slots do not guarantee that a model and its scratch
fit the GPU; Windows shared-memory spill and allocation failure remain possible.
Cached streamed weights also consume device memory outside the shared weight-cache
budget. This path does not implement layer streaming or CPU offload.

Upstream ggml performs CUDA capture after compatible graph executions stabilize.
Its CUDA graph map sweeps every five seconds and removes graphs unused for ten
seconds, so recently retired shapes can retain CUDA executable metadata briefly.
TensorSharp's trace confirms its own ggml graph reuse, not that CUDA capture or
replay actually occurred. Establish capture with upstream debug logging or a CUDA
profiler when making capture-specific claims.

## Attention and managed preparation

Image segments are bidirectional within their existing causal-prefix boundary.
CUDA flash attention can consume their exact KV lengths without padding, so these
segments no longer allocate or upload a quadratic all-valid/padding mask. Text
segments retain their causal masks.

CUDA flash attention already converts F32 K/V to F16 internally. TensorSharp
performs those conversions once per layer and shares the resulting K/V across
segment views, avoiding repeated copies and conversions of overlapping prefixes.
Attention retains F32 accumulation. Unsupported flash shapes still use the explicit
attention path with the original F32 Q/K/V. No quantization or approximate
attention method is added.

[`QwenImage21DiT.cs`](../../TensorSharp.Models/Models/QwenImage/QwenImage21DiT.cs)
retains at most two segment/RoPE layouts keyed by text length, image slots and
reference/target geometry. Keys copy caller arrays, so later array mutation cannot
return stale layouts. Text-to-image prediction pins the existing latent-token
array directly; editing still assembles reference and target tokens together.

| Control | Default | Purpose |
| --- | --- | --- |
| `TS_QWEN21_GRAPH_REUSE` | `1` on CUDA and Metal | `0` rebuilds the native graph for each prediction. CPU and Vulkan retain transient execution. |
| `TS_QWEN21_PAD_MASK` | `0` | `1` restores the prior CUDA padded-mask/F32-KV preparation for attention comparisons. |
| `TS_QWEN21_FLASH` | `1` | `0` uses explicit attention as a numerical reference. |
| `TS_QWEN21_GRAPH_TRACE` | `0` | `1` logs graph builds, scratch size and execution counts; it is not CUDA capture instrumentation. |
| `TS_QWEN21_PREFIX_CACHE` | on | `0` recomputes the text/reference prefix on every step instead of storing its keys and values. |
| `TS_GGML_LOG_DEBUG` | `0` | `1` exposes upstream debug messages, including CUDA graph warmup and reset decisions. |
| `GGML_CUDA_DISABLE_GRAPHS` | unset | Upstream presence-based switch to disable CUDA graphs independently of TensorSharp graph reuse. |

Use fresh processes for upstream environment controls, which ggml may read only
once. Keep prompts, model quantization, resolution, steps, CFG, seed, sampler and
sigma schedule fixed when comparing modes.

## VAE execution

The CUDA Qwen-Image-2.1 VAE can emit its complete encoder or decoder into the
existing TensorSharp-owned VAE graph interpreter. Intermediate features stay on
the device between convolution, normalization, activation, attention and shortcut
operations. The new average-down shortcut includes the causal front-padding
zeros in its divisor; duplicate-up retains the final temporal sample, matching
the single-frame managed path. Neither shortcut changes the learned topology.

Real Qwen-Image-2.1 decoder activations can exceed the F16 maximum before the
final normalization. Both the fused graph and its per-convolution fallback
therefore request TensorSharp-owned F32 im2col and F32 matrix accumulation.
Changing GEMM accumulation alone is insufficient: conversion of a finite
activation above 65504 into F16 already creates infinity. The existing generic
convolution API keeps its previous behavior; the 2.1 call chain uses a scoped
precision selection and accounts for four-byte im2col scratch when tiling.

The op lists and stable unmanaged weight buffers are retained with the VAE
weights. Their device-cache entries are invalidated before the buffers are freed.
Each invocation still constructs a ggml execution graph; this VAE path does not
claim persistent CUDA graph replay. Allocation or unsupported-op failures fall
back to the existing per-convolution path. Non-finite fused outputs also trigger
that fallback, and non-finite final RGBA is rejected before pixel conversion.

Set `TS_QWEN21_VAE_FUSED=0` for the per-convolution baseline. The existing global
`TS_QWEN_VAE_FUSED=0` and `TS_QWEN_VAE_GPU=0` controls continue to disable this
path. `TS_QWEN21_VAE_TRACE=1` reports fused execution or per-convolution tensor
statistics and stops at the first non-finite intermediate. Qwen-Image-2.1 uses
cuDNN automatically when its runtime is available. `TS_VAE_CUDNN_CONV=0`
disables it; `1` explicitly enables it. Other VAEs retain their existing opt-in
behavior.
Qwen-Image-2.1 requests cuDNN FMA math to prevent implicit F16/TF32 conversion.
Without cuDNN, the interpreter chooses ggml im2col/GEMM or direct convolution based
on the existing `TS_QWEN_VAE_FUSED_IM2COL_BUDGET` scratch limit.

Whole-VAE graphs may need substantial activation scratch at large resolutions.
Paired real-weight encode/decode validation is required in addition to the native
shortcut tests; passing the shortcut tests alone does not validate image quality.

## Reference-image vision encoder

On GGML CUDA, Qwen-Image-2.1 runs a fused transformer range through each
deepstack tap, projects that tap, then starts the next range. This preserves all
three intermediate image embeddings while avoiding a host synchronization for
every block. The MLPs use upstream `ggml_gelu_erf`, as required by Qwen3-VL;
the main and deepstack projectors also use the native erf activation. Older
Qwen3.5 callers retain their existing ABI and tanh-GELU behavior.

`TS_QWEN21_VISION_FUSED=0` restores the previous per-block vision path and its
managed erf approximation. The new kernel evaluates erf directly, so the
comparison is numerical rather than bitwise. Use the companion probe's `vision`
command to compare the main embedding and all three deepstack embeddings with
identical image preprocessing. `TS_QWEN35_VENC_FUSED=0` disables transformer
range fusion independently; CPU and Metal retain their previous execution.

## Conditioning encoder precision

Qwen3-VL layers in the fused conditioning encoder now use ordinary quantized
matrix multiplication, matching the per-operation implementation. The older
trunk divided every quantized projection input by 1024 and restored the scale
after multiplication. That transformation can underflow CUDA's F16 activation
quantization scales for small normalized inputs. Layers with Q/K head norms
identify the Qwen3-VL path.
`TS_QWEN21_TE_PRESCALE=1` restores the former scaling for numerical comparisons.
Use `TS_QWEN_TE_FUSED=0` as the per-operation reference, and compare on the same
backend before interpreting cross-backend quantization differences.
For a selected-layer comparison, set `TS_QWEN_TE_TRACE_DIR` to an ignored
validation directory and `TS_QWEN_TE_TRACE_LAYER` to the zero-based layer index.
Both paths write corresponding planar F32 intermediate tensors. Native tracing
retains those outputs, which can inhibit fusion; compare traced and untraced
final outputs before treating the trace as representative. The per-operation
head normalization uses a host double-precision sum while the CUDA graph uses
device reductions, so subsequent quantization can amplify initially small
rounding differences. Raw conditioning agreement alone does not validate image
quality.

Measured conditioning errors and their limitations are recorded in the ignored
validation report `docs/validation/qwen-image21-cuda/REPORT.md` (local validation
evidence, not committed).

## Reference implementations inspected

| Repository revision | Relevant source | Lesson and benchmark scope |
| --- | --- | --- |
| stable-diffusion.cpp `c678dfe704a2230342376b46add9c8ca736a653d` | `src/model/diffusion/qwen_image_2_1.hpp`, `src/core/compute_workspace.cpp`, `src/core/ggml_runner.cpp` | Complete segmented single-stream DiT graphs; mask-free image segments; reused allocation workspace and graph cuts for memory management. Supports Qwen-Image-2.1 text-to-image and editing with the dedicated VAE and Qwen3-VL encoder. |
| llama.cpp `ce8caa6e60a03093351d6016a818720e0d46f0fb` | `src/llama-context.cpp:process_ubatch`, `src/llama-context.h`, `ggml/src/ggml-cuda/ggml-cuda.cu` | Reuse a graph only when all topology parameters match; retain separate arenas for distinct graph variants; preserve addresses for CUDA capture. Provides encoder/kernel comparisons, not an end-to-end Qwen-Image-2.1 diffusion pipeline. |
| ComfyUI-GGUF `6ea2651e7df66d7585f6ffee804b20e92fb38b8a` | `ops.py:GGMLOps.Linear`, `GGMLLayer.cast_bias_weight`, `dequant.py`, `nodes.py:UnetLoaderGGUF` | Quantized weights are dequantized for PyTorch linear operations, with device casting and optional compilation delegated to PyTorch/ComfyUI. The plugin alone is not a standalone Qwen-Image-2.1 benchmark; record the host ComfyUI revision and workflow before comparing. |

TensorSharp's inspected ggml revision is
`456172ec733a135778adcd32d00e576a58232e45`, with an unchanged working tree.
The inspected stable-diffusion.cpp checkout had its ggml submodule at
`8e800cef2948046cc47f9db6090491c6128ca42c`, differing from the top-level index.
The llama.cpp checkout had a local edit to `examples/eval-callback/eval-callback.cpp`.
Record working-tree state alongside commit IDs for reproducible comparisons.

## Validation

Configure the native build with
`-DTENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON`
and `-DTENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON`, then build
`GgmlOpsQwenImage21Test` and `GgmlOpsQwenImage21VaeShortcutTest`.
Given the configured build directory in `$nativeBuild`:

```powershell
ctest --test-dir $nativeBuild -C Release -R qwen-image21-whole-graph --output-on-failure
ctest --test-dir $nativeBuild -C Release -R qwen-image21-vae-shortcuts --output-on-failure
dotnet test InferenceWeb.Tests --filter QwenImage21
```

The CTest fixture generates an explicit-attention CPU reference in a separate
process before checking CUDA or Metal. On macOS, keep Metal enabled and configure
`-DTENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON`; do not enable CUDA.
The test exercises whole-graph dynamic inputs, changed
shapes/segments/weight descriptors, fused and separate MLP layouts, zero-prefix
input reachability, cache invalidation and recovery. These synthetic weights test
execution correctness; they cannot establish full-model image quality.
Metal runs 159 forwards and excludes the forced device-copy-budget scenario:
its host weight mappings do not use that budget. The independent NumPy operator
oracle is `python3 eng/tests/qwen-image21-dit.py --backend metal`.

The end-to-end runner accepts exact model paths and writes generated evidence to
ignored `docs/validation/` by default:

```powershell
python eng/validation/qwen-image21-bench.py --backend ggml_cuda --engine tensorsharp --models-dir C:\Works\models\qwen-image-2.1 --prompt "A red ceramic teapot on a wooden table." --width 1024 --height 1024 --steps 40 --cfg 1 --repeat 3
```

Override `--vae`, `--text-encoder`, `--mmproj` and `--cli` when their paths differ.
For a stable-diffusion.cpp comparison use `--engine both --sd-cli <binary>` and
`--match-sigmas`; matching CLI step counts alone does not ensure matching flow
schedules. Run `--mode edit --image <file>` and multi-reference cases separately.
Record warm-step and total time, memory, output inspection, failures and skipped
scenarios. Pixel statistics or a smoke-test image alone do not establish semantic
quality or performance parity.

The runner records Windows process peak working set and samples whole-device
NVIDIA memory, utilization and temperature when available. Device memory
includes the desktop and other processes, and sampling can miss brief peaks.
Cold process times include loading and uncontrolled OS file-cache effects.
Use cooldowns between engines on laptops and inspect the recorded temperatures.
For the local Windows reference build, the matched comparison command is:

```powershell
python eng/validation/qwen-image21-bench.py --backend ggml_cuda --engine both --sd-cli artifacts/qwen-image21-sd-cuda/bin/Release/sd-cli.exe --sd-ggml-repo ExternalProjects/ggml --match-sigmas --sd-extra=--offload-to-cpu --cooldown-seconds 90 --width 1024 --height 1024 --steps 40 --cfg 1
```

The reference's CPU offload option releases staged model weights between phases
while keeping CUDA compute enabled. Check its logs for the actual placement and
whether decoding required automatic tiling. Runs that recovered through a tiled
VAE fallback are recorded separately from runs that completed the requested path.

Local measurements, model hashes, images, coverage and known limitations belong
in ignored `docs/validation/` or `artifacts/`. This session's evidence and results
are in `docs/validation/qwen-image21-cuda/REPORT.md` (local validation evidence, not
committed).
No unavailable model/device scenario counts as a pass.
