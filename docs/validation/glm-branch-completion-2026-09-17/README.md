# GLM branch completion — 2026-09-17

The `biglane-glm-fixes` work at `b7f224ac` fixes GLM end-of-message stopping,
the GLM-4/5 pre-tokenizer, GLM-5.3-Flash reasoning parsing and fallback prompts,
delayed structured-output grammar activation, and RoPE positions on every device
during native batched decode. Review found one additional parser defect: a
`</think>` inside a JSON string was interpreted as a protocol boundary. The
parser now keeps quoted markers as JSON data, including escaped quotes and
streaming splits inside the marker. A real closing marker outside a JSON string
still ends reasoning.

The native regression now checks one, two, and three GPUs independently and
reports unavailable device counts as skipped. Its gate queries CUDA directly:
the test initializer normally pins the process-global GGML backend to CPU, while
the native GLM executor owns separate CUDA backends. The tokenizer integration
test also accepts an explicit GGUF file, resolves shard one before reading
metadata, and checks that `<|observation|>` is an end-of-generation token.

## Environment and upstream integrity

- Local: macOS arm64, .NET SDK 10.0.302; managed tests only.
- Remote: Linux x86-64, .NET SDK 10.0.401, CUDA 12.8, NVIDIA A40 GPUs.
- Isolated remote checkout: `/workspace/ts-codex-glm-completion-20260917/repo`.
- ggml: `456172ec733a135778adcd32d00e576a58232e45`, a clean, unchanged upstream
  checkout. No ggml patches or source rewrites were applied.
- Native build: Release, `TENSORSHARP_GGML_NATIVE_ENABLE_CUDA=ON`,
  `TENSORSHARP_GGML_NATIVE_ENABLE_VULKAN=OFF`,
  `TENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON`, `CMAKE_CUDA_ARCHITECTURES=86-real`.
- Native library SHA-256:
  `3a993d77bac35418698396d3cd2ae4d50a3a31c6bf58bb71dfe47dc3063c8fa8`.
- GPU execution acquired `/workspace/locks/gpuN.lock` for every visible device.

## Actual coverage

| Run | Passed | Skipped | Evidence |
| --- | ---: | ---: | --- |
| Local focused GLM, tokenizer, and regression tests | 63 | 4 | `local-managed.trx` |
| Local related output-parser, structured-output, chat-template/protocol and BPE tests | 85 | 2 | `local-protocol-tokenizer.trx` |
| Remote focused managed suite with GLM-5.3-Flash tokenizer metadata | 64 | 0 | `glm-managed-remote.trx` |
| GLM-5.2 UD-IQ2_XXS tokenizer | 3 | 0 | `glm52-UD-IQ2_XXS.trx` |
| GLM-5.3 UD-Q2_K_XL tokenizer | 3 | 0 | `glm53-UD-Q2_K_XL.trx` |
| GLM-5.3-Flash tokenizer, explicit last-shard file input | 3 | 0 | `glm-explicit-last-shard.trx` |
| Native batched decode: 1, 2 and 3 A40 GPUs | 3 | 0 | `glm-cuda-batched.trx`, `cuda-batched-tests.log` |
| Native gating with only one visible A40 | 1 | 2 | `glm-one-visible-gpu.trx`, `one-visible-gpu-tests.log` |

Every real-tokenizer run checks the same 422 reference token-ID cases, both
pre-tokenizer aliases, and `<|observation|>` stopping. Model metadata came from
`/workspace/models/glm52/UD-IQ2_XXS`, `/workspace/models/glm53/UD-Q2_K_XL`, and
`/workspace/models/glm53-flash/UD-Q2_K_XL`.

The native fixture compares full logits and greedy tokens for two independent
sequences, with 24- and 37-token prompts, over six teacher-forced decode steps.
Both prompts exceed the fixture's sparse indexer top-k. The worst absolute
batched-versus-serial logit difference was **zero** in all three device counts.
The logs confirm four trunk layers were placed as 4, 2+2, and 2+1+1 across the
requested GPUs; these were actual layer splits.

As a negative control, only TensorSharp's `ggml_ops_glm_dsa.cpp` was temporarily
replaced in the isolated validation directory by its version immediately before
commit `3a704b00`. The same two-GPU test then **failed at step zero**, with a
maximum logit difference of `0.02817966416478157`, against its `0.001` tolerance.
See `glm-negative-control.trx` and `negative-control-tests.log`. The fixed source
and library were restored afterward; the ggml dependency remained unchanged
throughout. This expected failure demonstrates that the regression catches the
original cross-device position bug.

After restoration, restricting visibility to one GPU passed the one-device
case and explicitly skipped the two- and three-device cases. The restored
TensorSharp native source SHA-256 was
`a845f714e8fada90c769b4a64683140779f5303c3672de641b5543c656f77534`;
both the build library and its copy in the test output retained the
fixed-library hash above. The explicit last-shard tokenizer run confirms only
shard one's metadata is opened even when the requested path is another shard.

The four local focused skips are three unavailable CUDA cases and the missing
local real-model metadata. The related-suite skips require unavailable MiniMax
and Gemma model fixtures. None is counted as a pass. The run counts overlap and
must not be summed as distinct tests.

## Reproduction

Build the native library with the options above, then build
`InferenceWeb.Tests/InferenceWeb.Tests.csproj -c Release` with
`TENSORSHARP_GGML_NATIVE_SKIP=true TENSORSHARP_MLX_NATIVE_SKIP=true`. The test
project copies the newly built native library from its native build directory.

```sh
CUDA_VISIBLE_DEVICES=0,1,2 TS_TEST_GLM_CUDA=1 \
  dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj -c Release --no-build \
  --filter FullyQualifiedName~GlmDsaNativeBatchedDecodeLayerSplitTests

TS_TEST_MODEL_DIR=/workspace/models/glm53-flash/UD-Q2_K_XL \
  dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj -c Release --no-build \
  --filter FullyQualifiedName~Glm4TokenizerParityTests
```

## Limits

This validates parser/tokenizer behavior and synthetic native correctness. It
does not qualify real-model generation quality, concurrent agent turns, vision,
long-context behavior, or throughput. No performance benchmark or speedup claim
is made. The full real GLM checkpoints were read for tokenizer metadata only;
their trained-weight forward passes and the historical llama.cpp real-model
parity tests were not run. The checked-in tokenizer oracle was consumed rather
than regenerated, and Windows, Metal, Vulkan, and device counts above three were
not exercised.
