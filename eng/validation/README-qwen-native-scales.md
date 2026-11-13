# Qwen native scale and graph-replay checks

The fixtures in `TensorSharp.GGML.Native/tests` use small deterministic F32
weights to test graph behavior without downloading a model. They run the real
native operators. They are correctness checks, not model-quality or performance
benchmarks.

Enable `TENSORSHARP_GGML_NATIVE_BUILD_TESTS` in the native CMake build, along with
the intended CUDA or Metal backend, then build `GgmlOpsDFlashScaleTest`. Run:

```sh
ctest --test-dir TensorSharp.GGML.Native/build --output-on-failure -R 'dflash-sidecar-scales'
```

The Qwen 3.5 slot-stable arena and its fixture (`GgmlOpsQwen35ArenaScaleTest`) were
removed with `Qwen35Model`. The shared decoder runtime's sidecar scales and batch-width
transitions are covered by the managed `Qwen35RuntimeTests` and
`SharedRuntimeConformanceTests` lanes on each GPU backend.

The DFlash fixture covers both DFlash versions, injection and eight-row draft
blocks, borrowed embedding/head scales, layer projections, dynamic convolution,
selector codebooks, ring-cache bytes, confidence, sampled IDs and replay. It also
warms an unscaled graph before the scaled call using the same weight pointers.
This verifies that scale values participate in graph-cache identity. Managed
`DFlashSidecarRowsTests` covers the per-op embedding and selector gather fallback
with actual Q8 blocks and non-unit sidecar scales.

The fixtures were developed against unchanged ggml revision
`da8e2a54b919557b816b1c2beb432ccb766f7ce0`. The CUDA matrix must pass with graph
capture enabled; disabling capture is a diagnostic control, not a passing
production validation.

On visible SM12 CUDA devices, TensorSharp defaults `GGML_CUDA_PDL` to `0` before
CUDA backend enumeration. This keeps CUDA graph capture and fusion enabled while
disabling programmatic dependent launch. On the tested RTX 5090, the default
upstream PDL path produced incorrect logits and recurrent state starting at the
first capture of the 4-lane F32 case after the 2-lane cases of the (since removed)
Qwen 3.5 arena fixture. `GGML_CUDA_PDL=0`
restored exact results; clearing the backend's cached graph entry and replacing
the GDN output copies with explicit dependencies did not fix it. Those failed
workarounds are not part of the implementation.

Explicit `GGML_CUDA_PDL=0` and `GGML_CUDA_PDL=1` settings are preserved. Setting
`1` opts back into the unsafe, unqualified path on the affected device; do not
use it for production correctness qualification. The compatibility guard logs
its default once. `cuda-launch-policy-*` tests check unset, explicit 0/1 and
unaffected-device policies without hardware, and `cuda-launch-policy-device-*`
check that the real CUDA backend keeps an explicit setting. Device qualification
must also run the runtime's batched ggml_cuda lanes (`SlotBatchesMatchEachSequenceAlone`,
`ConcurrentRequestsAreBatched`) with this environment variable initially unset.
Other CUDA architectures retain their existing defaults; they are not covered by
the RTX 5090 validation.

Keep logs in ignored `artifacts/` or `docs/validation/`. A skipped backend (exit
77) is unavailable coverage, not a passing device test. Record the actual ggml
revision and device with each validation run.

The NVFP4 microbenchmarks alongside this file are independent experiments. Their
source headers contain build commands. They compare narrow MMVQ against padded
Q8 MMQ and a BF16 Tensor Core prototype, optionally linking an unmodified Velox
skinny-kernel object. Their synthetic matrices omit some production fused
projections, and BF16 changes activation rounding. They do not enable any
production kernel or establish end-to-end parity. In particular, the measured
Qwen3.8 GGUF artifacts had no sidecar scale tensors; the scale compatibility
changes cannot explain an improvement on those files.
