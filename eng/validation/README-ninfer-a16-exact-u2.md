# Optional exact-A16 two-phase-unroll candidate

This is one focused scheduling hypothesis in a separate native file/image. It
changes only T1's compile-time phase-unroll hint from fully unrolled10/12/34
512-value phases to2. All phase counts are even. Two-phase unroll is intended to
reduce replicated instruction-body pressure while retaining limited adjacent
load scheduling. Actual instruction-cache/occupancy effects and speed must be
measured; no performance benefit is inferred from source size.

All non-comment tokens outside that single constexpr expression must match the
qualified vector source with SHA256
`159f348e3cb6427aa8f21ee9c78f75ffe20c68b5928bfcceade205499337ecf4`.
The checker pins that source and verifies the entire token-level delta, all32
lanes' phase/low-high FMA column order, and divisible-by-two phase geometry.
T2/T3 remain unroll1. Hardware FP4x2 decoding, code/scale layout, F32 inputs and
outputs, coefficient placement, one-accumulator FMA order, XOR reduction,
signed-zero handling, all nine symbols/ABI and256-thread/16-row grid stay the
same. This is a source/order check, not proof of compiler output or device bits.

```sh
python3 eng/validation/check-ninfer-a16-exact-u2-contract.py
python3 eng/validation/build-ninfer-a16-legacy-exact-u2-module.py --output /absolute/artifacts/exact-u2-fresh.fatbin
```

The fresh builder refuses an existing image, runs the static contract first,
and records source/module SHA256, exact compiler/version/flags, the contract
result and actual compilation output. It uses the same stdc++17/O3/fmad=true/
ftz=false/sm120a policy as the qualified vector image; no fast-math flag or
arithmetic-chain tuning is introduced. Derived scheduling attribution and all
original license/notice references are retained in the new provenance file.

Use the unchanged `CudaNinferA16LegacyExactModule`, strict90-fixture component
probe and twelve-profile timing tool with the new image and fresh evidence:

```sh
dotnet eng/validation/NinferA16LegacyExactProbe/bin/Release/net10.0/NinferA16LegacyExactProbe.dll --execute --module /absolute/artifacts/exact-u2-fresh.fatbin --legacy /absolute/artifacts/frozen-r5-helper.fatbin --out /absolute/artifacts/exact-u2-components-fresh
dotnet eng/validation/NinferA16LegacyExactTimingProbe/bin/Release/net10.0/NinferA16LegacyExactTimingProbe.dll --execute --legacy-module /absolute/artifacts/frozen-r5-helper.fatbin --candidate-module /absolute/artifacts/exact-u2-fresh.fatbin --legacy-component /absolute/artifacts/legacy-components.json --candidate-component /absolute/artifacts/exact-u2-components-fresh/result.json --out /absolute/artifacts/exact-u2-timing-fresh
```

All existing exact CPU/GPU comparisons, fixed FP64 bounds, poison/guard/source
immutability/repeat checks and matching-image certificate gates remain required.
Do not adjust or weaken them. Preserve every failed attempt and actual array.
Managed or CPU-only checks do not qualify CUDA compilation, GPU execution,
model logits/text quality, end-to-end latency or serving adoption. A successful
kernel timing comparison still requires the separate full-model/quality gates.
Existing qualified sources, wrappers, engine options/defaults and ggml sources
are unchanged. This candidate has no automatic serving activation.
