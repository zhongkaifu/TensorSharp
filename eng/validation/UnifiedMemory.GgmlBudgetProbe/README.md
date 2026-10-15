# GGML native cache shared-budget probe

This tool runs exact F32 matrix-vector operations on one or two CUDA ranks and
checks the `GgmlCacheBudgetScope` bridge against native allocation telemetry.
It requires an unchanged upstream ggml checkout and a TensorSharp native build
with `TENSORSHARP_GGML_NATIVE_BUILD_TESTS=ON`; the six rollback scenarios use
TensorSharp-owned allocation-failure hooks.

```sh
dotnet build eng/validation/UnifiedMemory.GgmlBudgetProbe -c Release \
  -p:TensorSharpSkipGgmlNative=true -p:NuGetAudit=false
# Copy the matching test-enabled libGgmlOps.so into the output directory first.
dotnet eng/validation/UnifiedMemory.GgmlBudgetProbe/bin/Release/net10.0/UnifiedMemory.GgmlBudgetProbe.dll \
  2 artifacts/unified-memory/ggml-budget-two-ranks.json
```

The report records exact numerical results, per-rank native cache bytes and
managed pool accounting. Cases cover external reservations, both common and
per-rank capacity limits, lazy-cache streaming fallback, refused explicit
preloads, rollback before allocation/after allocation/after commit, callback
rooting through GC, failed disposal while allocations live, cleanup and scope
reattachment. Missing CUDA devices return exit 77 and `Passed=false`.

Each native allocation charges its rank's GPU constraint and `shared-cache`, a
deliberate common quota. On discrete CUDA GPUs this common quota does not
represent another physical host copy. These overlapping constraints must not
be summed as physical usage. The tool does not measure graph scratch, backend
pools, driver overhead, live KV outside these caches, model inference or
cross-device collectives. Two-rank coverage uses independent operations on
each GPU. Generated JSON and logs belong under ignored `artifacts/` or
`docs/validation/`.
