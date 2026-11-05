# Optional exact-legacy A16 batch candidate

This separate module extends the per-token frozen legacy A16 arithmetic through
T=8. It is OFF and has no engine hook. Existing qualified T1..3 scalar/vector
files, images, wrappers and probes are unchanged. Target A4 verification and
serving defaults are unchanged.

The candidate uses the existing TS low8/high8 code planes, row-major scales,
hardware FP4x2 decode, one ordered FMA accumulator per row/token, and the legacy
XOR reduction. Inputs and outputs remain F32 storage; callers supply values
already representable as BF16, and projection output is unrounded F32. T2..8
reuse weights and stage inputs at `column ^ ((column >> 4) & 31)` within each
512-value token phase. This changes addresses only. The CPU test proves a
bijection and distinct 32-bit shared banks for both contiguous warp stores and
`lane*16+j` reads. Actual device validation remains necessary.

The optional wrapper is `CudaNinferA16BatchExactModule`, with explicit environment
name `TS_Q35_NINFER_A16_BATCH_EXACT_PTX` and numerical policy
`ninfer-target-legacy-per-token-a16-batch-t1-8-f32-v1`. Creating the ordinary
NInfer engine does not load this module.

```sh
python3 eng/validation/build-ninfer-a16-batch-exact-module.py --output /absolute/artifacts/a16-batch-exact.fatbin
dotnet build eng/validation/NinferA16BatchExactProbe/NinferA16BatchExactProbe.csproj -c Release
dotnet eng/validation/NinferA16BatchExactProbe/bin/Release/net10.0/NinferA16BatchExactProbe.dll --self-test
dotnet eng/validation/NinferA16BatchExactProbe/bin/Release/net10.0/NinferA16BatchExactProbe.dll --execute --module /absolute/artifacts/a16-batch-exact.fatbin --legacy /absolute/artifacts/frozen-r5-helper.fatbin --out /absolute/artifacts/a16-batch-exact-components
```

The 240 fixed fixtures cover every T1..8, all three input widths, output counts
1/17/129, three scale/input distributions and analytical cancellation. Strided
inputs, poisoned output padding, guards, source immutability, two module lifetimes
and A/B/A destination poisoning remain mandatory. Every written F32 bit must
match all three independent comparisons:

1. The frozen one-token legacy CPU oracle reading original artifact byte layouts.
2. The original scalar CUDA helper called with the complete T-row batch.
3. Independent original scalar CUDA helper calls with T=1, one per input row.

The unchanged predeclared FP64 bound is
`gamma_(K/32+16)*sum(abs(original_fp4*fp8*inverseD*BF16input))`, u=2^-24. Every
output must also satisfy that bound. All three complete GPU outputs are retained
and SHA checked. The analytical low/high cancellation result is64 for every T;
no adjacent-K or multi-chain arithmetic is substituted.

Both component and timing tools keep replaceable `result.json` progress, then
write authoritative `completed-result.json` once with CreateNew and its exact-byte
`completed-result.json.sha256` sidecar only after all gates pass. Preserve failed
attempts and use fresh ignored output directories.

The separate timing probe uses the frozen xorshift64 generator/seed, original
weight bytes, and existing first-three-token input prefix; the input stream is
continued deterministically to eight rows. A local comparison against the frozen
T1..3 timing assembly checks unchanged weight planes and the input prefix. Timing
uses full gate-up/down/output/attention-input shapes at T4/7/8, five ABBA blocks,
CUDA events, and the existing warm-repeat / over-two-L2-address rotation labels.
All warmup, CPU preparation, copying and validation is outside timed intervals.
Rotation is not proof of a physically cold cache.

```sh
dotnet build eng/validation/NinferA16BatchExactTimingProbe/NinferA16BatchExactTimingProbe.csproj -c Release
dotnet eng/validation/NinferA16BatchExactTimingProbe/bin/Release/net10.0/NinferA16BatchExactTimingProbe.dll --self-test
dotnet eng/validation/NinferA16BatchExactTimingProbe/bin/Release/net10.0/NinferA16BatchExactTimingProbe.dll --execute --legacy-module /absolute/artifacts/frozen-r5-helper.fatbin --candidate-module /absolute/artifacts/a16-batch-exact.fatbin --legacy-component /absolute/artifacts/legacy-components.json --candidate-component /absolute/artifacts/a16-batch-exact-components/completed-result.json --out /absolute/artifacts/a16-batch-exact-timing
```

Timing requires matching completed component evidence, the exact legacy image
used for those comparisons, all original capture hashes, and a valid completed
certificate sidecar. Full outputs at every resident address must match the CPU
oracle before timing. These are synthetic matrix measurements, not end-to-end
performance. Managed builds and CPU tests cannot qualify native compilation, GPU
execution, dense real-model logits, speculative verification/rollback/cache state,
text quality, concurrency or serving adoption. Those checks remain separate.
No model or device scenario may be counted as passed before it actually runs.
