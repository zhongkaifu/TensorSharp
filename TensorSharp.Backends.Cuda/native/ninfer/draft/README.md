# Original NInfer DFlash2 components

`../tensorsharp_q35_ninfer_draft.cu` is a separate, experimental CUDA-driver module. Nothing loads it during ordinary model serving. `CudaNinferDraftModule` exposes bounded components; it does not implement target verification/commit, prefix-cache integration or an attached drafter.

The module consumes **original** adjacent-K NVFP4 code bytes and m128x4-swizzled E4M3 scales. Its activations, norms, dynamic-convolution bases and 2048-position KV ring use actual BF16 storage. It does not accept the target adapter's permuted code/scale planes. Each original draft tensor can be streamed once through `ICudaNinferOriginalProjectionSource.ReadOriginal`; validate all encoded scales and the retained weight-divisor trailer before publishing device weights. Do not create both original and permuted device copies.

The 12 headers under `upstream/` are unchanged bytes from Ignis `e600a7d9e0d5921d767a6cb464373583b3cdcaf4`; `upstream/manifest.json` records their Git blobs and SHA256 values. The reference NInfer fork revision is `a00648cb828457986cf5b4b4f712b4cbcd7af0d1`. Apache-2.0 attribution and notices are preserved. No ggml files are used or modified. TensorSharp adapters and their source attribution are separate from those headers.

Example explicit component build on an SM120a CUDA host:

```sh
nvcc -std=c++17 -arch=sm_120a --ptx --ftz=false --fmad=true \
  -I TensorSharp.Backends.Cuda/native/ninfer/draft/upstream/kernel/vendor/src \
  TensorSharp.Backends.Cuda/native/ninfer/tensorsharp_q35_ninfer_draft.cu \
  -o artifacts/ninfer-draft.ptx
```

Do not use `--use_fast_math`. CPU admission tests are not native qualification. Native component comparisons, guards, repeated module lifetimes, multi-lane ring wrap/clone, target acceptance/rollback, and model-level quality remain separate required checks before serving integration or performance claims.

Full target W8 output-head policy is the initial integration target. The original Q4G64 mapped shortlist has a loading descriptor, but this module intentionally exposes no shortlist head kernel.
