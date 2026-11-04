# Owned FI placement bridge component gate

This new standalone probe tests the two TensorSharp-owned placement kernels. It
does not load a model, project weights, invoke SwiGLU, score task quality or
measure performance. Default invocation is PLAN. Only `--execute` accesses CUDA.
The actual bridge image must match an explicit SHA and its wrapper validates
the actual two-symbol device ABI. No upstream or frozen probe files are changed.

The CPU BF16 oracle compares FP64 distances to the neighboring representable
values and resolves ties by significand parity. It does not reuse the production
integer rounding bias. Fixtures include all65,280 finite BF16 values widened to
F32;195,840 midpoint/one-F32-ULP-neighbor values of both signs; signed zero, F32
and BF16 subnormals, normal boundaries, and finite-to-BF16-infinity overflow.
Expected overflow infinities are checked as exact bits rather than incorrectly
reported as nonfinite failures. Input counts cross255/256/257 and the actual
maximum8×17408; no unsupported input is silently shortened.

The independent scatter oracle assembles semantic eight-row gate/up chunks and
256-row Q/gate head pairs, followed by K/V slices. It checks full unique row tags
and arbitrary32-bit payloads, including infinities, NaN payloads, signed zero and
subnormals. Scatter is a bit-preserving permutation; no float arithmetic or
finiteness requirement is applied to these payloads. Identity shapes are N128,
5120,6144,17408,34816; gate/up N34816; GQA N14336. Every shape runs T1/7/8.

All57 fixtures must complete two separate module lifetimes and three output
poisons:114 records/342 launches. A nondefault CUDA stream is synchronized before
readback. Complete guarded input/output bytes, exact CPU outputs, allocation
immutability, same-case AA/cross-lifetime bits and rejected-call nonmutation are
mandatory. Deliberately non16-byte-aligned but legal F32/BF16 offsets exercise
compact placement. All456 retained fixture/expected/guarded-output files are
rehashed before finalization. The fixed1GiB raw-output cap is a refusal bound;
planned output payload is below100MB. Only a write-once `completed-result.json`
and SHA sidecar, after module/stream/context disposal, mark successful completion.

```sh
dotnet build eng/validation/NinferFlashInferBridgeProbe/NinferFlashInferBridgeProbe.csproj -c Release
dotnet eng/validation/NinferFlashInferBridgeProbe/bin/Release/net10.0/NinferFlashInferBridgeProbe.dll --self-test
dotnet eng/validation/NinferFlashInferBridgeProbe/bin/Release/net10.0/NinferFlashInferBridgeProbe.dll --execute --module /absolute/bridge.fatbin --expect-sha256 ACTUAL_SHA256 --out /fresh/artifacts/bridge-components
```

CPU tests or unavailable devices never qualify GPU execution. Passing this
placement gate does not establish FI projection math, row-map composition with
real weights, model quality, speculative equivalence or end-to-end speed. Those
remain separate actual component and whole-model gates.
