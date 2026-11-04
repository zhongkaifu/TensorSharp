# NInfer cached RMSNorm component probe

This is an experimental validation tool; no serving path uses this candidate.
Build its isolated module with `eng/validation/build-ninfer-norm-cache.py`.

This independent tool compares the actual old `ts_ni_add_norm` image with the
separate `ts_ni_add_norm_t8_cached` candidate. It changes no serving policy.
The only admitted shape is T8/K5120, grid8/block256, epsilon1e-6. Both images
are explicit and checked against CLI SHA256 pins, including the image actually
loaded by the driver. Use direct fatbin paths to avoid sibling-image ambiguity.

```sh
dotnet run --project eng/validation/NinferNormCacheProbe -- --self-test
dotnet eng/validation/NinferNormCacheProbe/bin/Release/net10.0/NinferNormCacheProbe.dll \
  --legacy-module /absolute/old-helper.fatbin --expect-legacy-sha256 OLD_SHA256 \
  --candidate-module /absolute/cached-norm.fatbin --expect-candidate-sha256 CANDIDATE_SHA256 \
  --out /absolute/artifacts/norm-component
```

Seven fixtures cross null delta, disjoint delta, delta==x and delta==output.
They include nonuniform F32 unit-offset effective weights, cancellation, BF16
ties and adjacent values, signed zero, the final column and lane/row boundaries.
Two module lifetimes and A/B/A poisons require 56 unique records/336 norm
launches. Every residual and norm value must match the old GPU image bit for
bit, including zeros, across lifetimes. Residuals also match independent CPU
F32-add/BF16-RNE bits. Both norm outputs must be finite BF16 values within a
fixed FP64 envelope documented in `Reference.cs`; this scoped envelope never
replaces the exact GPU gate. All calls use a nondefault stream. All allocation
guards and immutable inputs are checked. Delta==output necessarily seeds the
output with fixed delta values after poisoning, because those bytes are inputs.

Add `--timing` to the same command to run timing only after all component gates
pass. Timing uses the nonuniform/disjoint fixture and five paired ABBA rounds,
64 graph replays per interval, with one symmetric outside warmup. Each arm
captures **two device-copy resets plus its norm**; reset costs are included
identically. Thus every replay starts from the original residual/output, rather
than repeatedly adding delta. Every interval checks full qualified output bits,
guards and immutable seeds. This reports reset+norm graph latency, not isolated
norm latency or model throughput. Numerical completion and the predeclared
>5% median improvement/at least4 of5 paired wins are separate results.

Full outputs use deduplicated SHA256 CAS files, compared by bytes before reuse.
The authoritative final is write-once `completed-result.json` plus its SHA256
sidecar, written after graphs, modules and context are disposed and raw files
are rehashed. `result.json` is progress only. A CPU self-test is not GPU/model
qualification. Native compilation, GPU execution, model quality and end-to-end
performance require separate actual evidence and are not inferred from skips.
