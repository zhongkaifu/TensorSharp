Qwen-Image-2.1 native regression tests
====================================

Configure with `TENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON`, build
`GgmlOpsQwenImage21Test` and `GgmlOpsQwenImage21VaeShortcutTest`, then run
`ctest --test-dir <build-directory> -R qwen-image21 --output-on-failure`.

The CPU whole-graph test writes a synthetic explicit-attention reference fixture
under the build directory; the CUDA, Vulkan and Metal whole-graph tests compare
against it. On Vulkan, the test skips process teardown after printing its verdict:
upstream ggml-vulkan keeps its VkInstance until exit, and NVIDIA's driver threads
can fault while libraries unload (reproduced with ggml alone). The CUDA test compares forwards with that reference,
including changed inputs and shapes, graph reuse, weight invalidation, and forced
graph-owned weights. The Metal test compares against the same fixture;
it excludes the device-copy-budget scenario because Metal maps host weights
directly, so a device-copy cap cannot force graph-owned constants there.
`TS_QWEN21_GRAPH_REUSE=0` disables graph reuse on CUDA and Metal for comparisons.

The public shared-budget bridge is also exercised with cache-only and graph-inclusive
scopes. Zero credit must refuse graph allocation without crashing or leaking credit;
raising the quota must reproduce the control output. On CUDA/Metal, a tight quota
forces retirement of the other cached graph shape before retrying. Prefix storage
must remain charged after graph reset and return its credit on release. Repeated
attach/run/dispose cycles must leave no live reservations. These use small synthetic
weights; they do not establish a whole-model or process memory limit.

Every shape with a prefix also runs the prefix KV cache. Each forward with a cache
key is compared with the uncached graph on the same inputs: the first stores the
prefix (extract), and later steps with a new target and timestep read it (cached).
This covers every storage type (the default must match bit for bit; F16, F32,
Q8_0 and Q8_0_V are held to rounding tolerances), retained and transient graphs,
flash and explicit attention, a graph reset and a scratch release mid-request,
release and re-extraction, two interleaved CFG keys, a key reused for another
layout, and a cache declined by `TS_QWEN21_PREFIX_CACHE_MAX_MIB=0`, which must
fall back to the uncached graph. The test then shards the model over two ranks:
a real two-GPU group on CUDA (device collective) and Vulkan (host reduction), or,
on CPU and Metal, a loopback group (`TSGgml_TensorParallelInitLoopback`: backend
instances on the one device, host reduction); a single-GPU CUDA or Vulkan machine
prints a SKIP for this part. It compares sharded forwards, with
and without the cache, against the unsharded graph, and checks that ranks
disagreeing on a descriptor are refused. The loopback group exercises the
per-rank graphs, segment schedule and reduction but not NCCL or P2P, and says
nothing about multi-GPU speed.
The independent NumPy operator oracle is `eng/tests/qwen-image21-dit.py` and accepts
`--backend cpu`, `--backend metal`, or `--backend cuda`.
VAE tests check independent shortcut/convolution oracles
and finite activations above the FP16 range. These are numerical regressions;
they do not establish real-model image quality or performance parity.

Metal F32 convolution tests run in separate processes with MPS enabled and with
`TS_VAE_MPS_CONV=0`. Full-precision convolutions use MPS F32 when its shape is
supported and ggml's direct F32 convolution otherwise. F32 im2col plus F32
accumulation is insufficient on Metal because its matrix-matrix kernel stages
F32 operands in F16. The scalar oracle includes 32-output-channel projections
that exercise that matrix path, values above 65504 in both signs, changed inputs,
right/bottom asymmetric padding with stride two, and unequal X/Y stride or
padding that require the direct fallback. Both standalone and fused VAE calls
are checked. `TS_GGML_NODE_PROFILE=1` reports vendor calls and direct fallbacks;
unavailable vendor offload must not be counted as MPS coverage.
The wide-range cases run again after explicit scratch release and after backend
shutdown/recreation. Those operations retire the MPS graph/staging cache; the
TensorSharp MPS bridge uses ARC so replacing buffers or clearing cache entries
releases their Objective-C ownership.
The Metal tests also run the temporal shortcut fixtures (literal, synthetic and
invalid cases). Unchanged upstream ggml-metal pads only at the end of a dimension,
so the average-down shortcut builds its leading zero time slices as `ggml_fill`
plus `ggml_concat` on a backend that refuses the leading pad; CPU and CUDA keep
`ggml_pad_ext`. Until 2026-09-30 that one PAD node refused the whole fused encoder
graph on Metal, so every edit encoded its reference image on the per-convolution
path (the decoder's duplicate-up shortcut needs no pad and already ran fused). The
narrow-normalization fixture still runs on CPU/CUDA only and is not Metal coverage.

On a cuDNN-enabled Windows build, `qwen-image21-vae-missing-cudnn-cuda` runs in a
fresh process with an isolated stub `cudnn64_9.dll` that exports no cuDNN
functions. Its sibling DLL names are fixture copies generated in the build
directory; no installed runtime is modified. This reproduces an incomplete
runtime export table, which must fall back to full-F32 ggml convolution rather
than call a null function pointer. The test verifies the stub was actually
loaded and the complete VAE numerical checks passed. A wholly absent runtime
returns the same unavailable API state, but is not separately simulated.

Run VAE checks with `TS_VAE_CUDNN_CONV=0` for the ggml path and `=1` for vendor
offload when available. Unset selects vendor offload automatically for Qwen21.
`TS_GGML_NODE_PROFILE=1` shows actual vendor calls and fallbacks. CUDA absence
returns skip code 77 and must not be counted as passing device coverage.
