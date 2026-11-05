# Fixed GateUp + SwiGLU exact-fusion probe

This standalone validation project compile-links the original A4 probe's
`Reference.cs` and `Device.cs` unchanged. It qualifies only T8, N34816, K5120,
split1, on the existing CC12.0/170SM device. It has no engine dispatch hook.

Three numerical arms use the same source weights/input:

1. Actual production EA `ts_q35_gemm_w4a4_8` plus original helper `ts_ni_swiglu`.
2. Newly compiled same-source `ts_q35_gemm_w4a4_8` plus the same helper.
3. New `ts_q35_gemm_w4a4_8_swiglu_bf16`, directly writing compact final F32
   values carrying the original final BF16 RNE result, without a second helper.

The primary EA image SHA is fixed. Helper, EA and candidate paths and actual
SHAs are all explicit. All three GEMM entries have the original 12-slot ABI,
grid272x1x1, block256, dynamic shared43520. Split1 scratch/counters are guarded
but not given invented numerical, zero/reset, or repeat semantics.

```sh
dotnet build eng/validation/NinferGateUpSwiGluProbe/NinferGateUpSwiGluProbe.csproj -c Release -p:TensorSharpSkipCudaPtx=true
dotnet eng/validation/NinferGateUpSwiGluProbe/bin/Release/net10.0/NinferGateUpSwiGluProbe.dll --self-test
```

Default with no arguments is PLAN and does not initialize CUDA. Root alone
executes the actual command:

```sh
dotnet NinferGateUpSwiGluProbe.dll \
  --helper-module /absolute/helper.ptx --expect-helper-sha256 ACTUAL_LOADED_IMAGE_SHA \
  --legacy-module /absolute/ea/tensorsharp_q35_kernels.fatbin \
  --expect-legacy-sha256 ea0e322a36c48211750e0f569bd12c44f9d62caa6d3249c37f1547679bdccbea \
  --candidate-module /absolute/new.fatbin --expect-candidate-sha256 ACTUAL_SHA \
  --out /absolute/ignored/artifacts/FRESH --timing
```

The original three fixtures (`exact-binary`, `broad-finite`,
`ties-zero-and-last-column`) run in two module lifetimes, with three output
poisons per lifetime. All quantized codes/scales/row scales must match the
unchanged CPU reference. Controls0/1 capture all278528 raw F32 values and check
full bits plus the unchanged decoded-FP64 bound. Fused raw is explicitly
**NOT_CAPTURED**; no unavailable intermediate or independent expf oracle is
invented. All139264 final F32 values must match old EA+helper bits, remain
finite/BF16-valued, repeat across poisons and module lifetimes, and preserve
all input bytes and allocation guards. There are six fixture/lifetime records
and144 actual component kernel executions (3+3+2 per pass).

Optional timing begins only after these numerical checks. A frozen dense
pseudorandom weight fixture is direct-qualified by all three arms. Its raw
controls require full exact GPU bits and the original fixed active-row FP64
witnesses, explicitly a smaller numerical scope than full fixture FP64.
Only actual EA+helper and fused arms are timed. Driver graph inspection checks
all actual arguments, geometry and producer/consumer dependencies: old graph
quant/GEMM/SwiGLU has3 nodes, fused graph quant/fused has2, with no resets.

One fixed5ABBA campaign uses128 graph replays per interval, plus one separate
warmup. Every interval checks the complete final output, old raw when available,
quantized payloads, input immutability and guards. The predeclared exploratory
gate is speedup>1.01, savings>=0.001ms/full chain, and at least4/5 paired-round
speedups>1.005. Numeric/component completion is separate from this performance
gate. Timing totals are2560 graph replays,6400 timed kernels,50 warmup kernels,
and8 direct-admission kernels. Capturing does not execute graph kernels.

These are warm resident synthetic component chains, not model quality,
physical cold-cache behavior, complete-model performance or default adoption.
Raw CAS is bounded to128MiB and fully rehashed after CUDA teardown. A write-once
`completed-result.json` and SHA sidecar are written after the numerical and
lifecycle gates; `timing.predeclaredWin` separately retains the performance
decision, including failures.
