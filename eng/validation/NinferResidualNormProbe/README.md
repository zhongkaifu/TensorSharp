# Verify T8 residual and norm components

This standalone tool tests two fixed shapes: `output` is N5120/K6144/split8;
`down` is N5120/K17408/split8. It does not change production dispatch, native
sources or defaults. It compile-links the existing A4 projection reference and
device/guard helpers without changing those files.

```sh
dotnet build eng/validation/NinferResidualNormProbe -c Release -p:TensorSharpSkipCudaPtx=true
dotnet eng/validation/NinferResidualNormProbe/bin/Release/net10.0/NinferResidualNormProbe.dll --self-test --shape output
dotnet eng/validation/NinferResidualNormProbe/bin/Release/net10.0/NinferResidualNormProbe.dll --self-test --shape down
```

CPU self-tests do not register CUDA. They check the original FP4 fixtures, new
projection payloads, signed zero, BF16 ties and adjacent inputs, effective F32
weights and the fixed independent norm envelope. The contraction witness uses
dot3, F32 alpha1/3 and residual-1: separate FMUL/FADD yields zero while FMA does
not. No underflow fixture is used to stretch the original projection bound.

No arguments prints PLAN. Actual execution requires pinned native files and a
fresh absolute ignored **local POSIX** output. Preserve failed campaigns. The
candidate PTX's scalar argument types and the driver's actual parameter offsets
and sizes are checked separately; the operator must bind PTX/fatbin/source/build
and device provenance before execution. Use the actual R34 old BN8 image and
unchanged NInfer helper, not an earlier BN32 baseline.

```sh
dotnet /ABS/IMMUTABLE_RUNTIME/NinferResidualNormProbe.dll \
  --execute --shape output \
  --helper-module /ABS/old-helper.fatbin --expect-helper-sha256 ACTUAL_SHA \
  --legacy-module /ABS/actual-R34-main.fatbin --expect-legacy-sha256 ACTUAL_SHA \
  --candidate-module /ABS/new-main.fatbin --expect-candidate-sha256 ACTUAL_SHA \
  --candidate-ptx /ABS/new-main.ptx --expect-candidate-ptx-sha256 ACTUAL_SHA \
  --out /ABS/artifacts/FRESH
```

Run both shapes in fresh outputs. `--timing` first requires that invocation's
complete numeric campaign to pass, then measures its complete chain. Stop on any
failure, and accept no combined candidate until both shapes have passed all
required numerical and timing gates. There is no cross-process admission
certificate or automatic retry.

Three arms use actual-old `_8`+old AddNorm, same-source `_8`+old AddNorm, and
new residual `_8`+readonly PlainNorm. Five fixtures run across two module
lifetimes with three A/B/A poisons. All used split partials, unrounded F32 block,
BF16-canonical F32 residual and BF16-canonical norm output must match completely.
Every quantized code/scale/rowScale must match the independent CPU quantizer;
sparse and special projection fixtures have the original decoded FP64 check
(dense witness coverage is bounded below). Norms additionally
use an independent FP64 positive square sum and a fixed, conservative envelope
covering F32 reduction/rsqrt/products and final BF16 rounding. Norm error is
evaluated from the exact represented residual; projection error is checked
separately. No independent approximation replaces full old/new bit equality.

The new norm must leave its complete input unchanged. Separate semantic cases
check old `(-0)+(-0)` versus new readonly `-0` and old initial-null `(-0)+(+0)`.
The initial-null case must remain the old path in engine integration; this tool
does not implement or certify that routing. Use `NinferResidualNormBodyProbe`
for the separate fixed whole-model state/output comparison. Guarded allocations, immutable
weights/inputs, zero returned counters and AA across both module lifetimes are
mandatory. There are 270 projection/norm kernel calls plus18 signed-zero norm
calls per shape: **288 actual direct submissions** on a complete run.

Optional timing uses only actual-old versus new, excluding same-source from the
timing arms. Each arm captures reset_x→quant→GEMM→norm in one graph, including
the identical D2D residual reset in its event interval. Actual graph copy/kernel
arguments, functions, shapes and the full dependency chain are inspected. Every
replay starts with the same seed, never a repeatedly accumulated residual.
One and six independent dense-weight allocations are evaluated separately.
Working-set bytes are recorded; neither route is called physically cold or
proof of DRAM traffic. Dense projection FP64 covers only192 active-row witnesses;
every full GPU output/partial/residual/norm bit is still required to match.

Each address mode has five ABBA rounds, 64 timed graph replays per interval and
one symmetric outside warmup. Full outputs, quantization, counters, guards and
input immutability are rechecked after every interval. `passed` means valid
numerics and measurement; **`predeclaredWin` is separate**, requiring median
speedup strictly greater than1.05 and at least4/5 paired wins. Both address modes
must win before `timingAllAddressModesPredeclaredWin` becomes true. Per shape the
two modes submit27,300 timed/warmup kernels and9,100 reset copies; 42 direct
admission kernels are separate. Captured host submissions are not device runs.

Complete raw arrays are CAS files, checked again before finalization. The hard
raw limit is512MiB **per shape process**; a budget refusal is failure, not skipped
coverage. Fixtures/weights are released one at a time. `result.json` is progress
or failure only. After all module/stream/context disposal, file-pin and CAS
checks, `completed-result.json` and its SHA sidecar are written once. Source,
device and whole-runtime provenance require the operator's immutable campaign.
No engine state, feature taps, Verify/Commit integration, model quality or
end-to-end performance is qualified by this component tool.
