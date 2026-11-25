Qwen-Image-2.1 native regression tests
====================================

Configure with `TENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON`, build
`GgmlOpsQwenImage21VaeShortcutTest`, then run
`ctest --test-dir <build-directory> -R qwen-image21 --output-on-failure`.

The transformer has no native test of its own: it runs as the diffusion runtime's
graph programs, which `InferenceWeb.Tests/QwenImage21DenoiserTests.cs` checks on the
backend `TS_TEST_GGML_BACKEND` pins against the retired native graph's frozen
outputs (`InferenceWeb.Tests/Fixtures/QwenImage21/LegacyGoldens`, through the prefix
KV cache and with LoRA adapters).

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

On a cuDNN-enabled Windows build, two tests each run in a fresh process with an
isolated stub `cudnn64_9.dll`. Its sibling DLL names are fixture copies generated
in the build directory; no installed runtime is modified.

- `qwen-image21-vae-missing-cudnn-cuda`: the stub reports the CUDA major this
  build targets (`cudnnGetCudartVersion`), so the loader's CUDA-major check
  passes, and then exports none of the cuDNN functions the loader resolves. This
  reproduces an incomplete runtime export table, which must fall back to full-F32
  ggml convolution rather than call a null function pointer.
- `qwen-image21-vae-cuda12-cudnn-cuda`: the stub exports every function the
  loader resolves but reports a CUDA 12.8 build. cuda12 and cuda13 builds of
  cuDNN 9 share every file name, so the loader must decline it on the CUDA major
  alone. Each stub function counts its calls, and the test requires none.

Both tests verify that the stub was actually loaded and queried, and that the
complete VAE numerical checks passed on the ggml fallback. A wholly absent
runtime returns the same unavailable API state, but is not separately simulated.

Run VAE checks with `TS_VAE_CUDNN_CONV=0` for the ggml path and `=1` for vendor
offload when available. Unset selects vendor offload automatically for Qwen21.
`TS_GGML_NODE_PROFILE=1` shows actual vendor calls and fallbacks. CUDA absence
returns skip code 77 and must not be counted as passing device coverage.
