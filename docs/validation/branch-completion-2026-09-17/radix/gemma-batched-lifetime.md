# Inherited Gemma batched input lifetime failure

The failed original Gemma ABBA process was **control arm `b-1`**, not candidate
arm `a`. Its runner identifies `b` and `c` as baseline `99d4f949` and `a` as the
original M2 worktree. It ran Gemma 4 E4B Q8_0 on Metal, with chunk size 1024,
one warmup pass, three measured passes, and concurrency 1/4/8. At
2026-09-17 13:43:31 UTC, PID 82726 exited 139 during measured pass 3's
concurrency-8 scenario. Only two completed passes were written.

The matching macOS crash report records `EXC_BAD_ACCESS`, a read fault in
`_platform_memmove` called by `TSGgml_Gemma4ModelDecodeBatchedEx + 10324`.
Its loaded native Mach-O UUID is `0B1D5651-5163-3EFD-A0D4-A3372E4B663F`, matching
the surviving baseline library SHA-256
`f5a5810ab09a4e2165e4a9228f0b300ff9d988569780c76cc50bc19fbbec4caf`.
The original M2 native copy has the identical UUID and SHA-256.

Disassembly places the return PC `0xa4090` immediately after the
`ggml_backend_tensor_set` call at `0xa408c` that uploads `hidden_data` into
`current` (`ggml_ops_gemma4_batched.cpp`, line 876 in the baseline). The faulting
source address is `0x717f524000`; the copy length is 71,680 bytes, exactly
`2560 * 7 * sizeof(float)`. Seven active rows are consistent with one of the
eight requests finishing before the failing decode step. This is an input
upload failure before graph execution, not a flash-attention kernel abort.

The baseline benchmark program and native batched source match `99d4f949`
byte-for-byte. The managed `TryForwardBatchedFusedDecode` method is identical
in that baseline, M2 `9ae80b49`, and the combined integration before this fix.
The native batched source is also identical to original target `e10dd42b`;
the combined integration only changes flash-attention construction elsewhere
in the function. Thus the evidence identifies an inherited defect and does
not establish an M2 regression. The archived managed assembly records version
2.8.6 without a source SHA; the source attribution comes from the runner and
the matching surviving source, not an embedded assembly revision.

## Fix and regression

The managed method created an embedding tensor, extracted its raw pointer, and
then no longer used the tensor on the in-kernel PLE path. An optimized JIT may
stop rooting that tensor before allocating the logits array or entering the
long native graph construction. Its storage inherits the `RefCounted`
finalizer, which releases the GGML allocation. A raw pointer alone does not
keep that owner alive.

Commit `258e6368` changes that local to `using Tensor hidden`, keeping ownership
through the native call and disposing it on every exit. The native function
copies the hidden input into its own tensor and synchronizes before returning,
so the scope covers the required lifetime. Native sources and binaries were
unchanged for this fix.

An internal, default-null hook observes only a weak storage reference just
before native invocation. The serialized model regression forces full GC and
finalization there, then checks that the storage remains reachable, owns its
allocation, and has a nonzero pointer before native code can dereference it.
The hook is reset in `finally`. The test then requires actual fused execution
and finite logits for widths 8, 8, 7, 7, 4, 4, 2, 2, covering graph creation,
reuse, and a shrinking active set.

With `DOTNET_TieredCompilation=0`, the test **failed before the fix** with
`Batched hidden storage was collected before native upload.` After the fix,
both this test and the existing batched parity test passed: **2 passed, 0
failed, 0 skipped**. Parity used widths 2/3/4, eleven fused steps at each width,
zero fallback steps, and identical greedy continuations, including a
581-token prompt beyond the SWA window. Before-fix cold Tier 0 execution
passed, so the optimized-JIT setting is necessary to reproduce the ownership
failure immediately rather than depending on warmup and JIT promotion.

These runs used the Apple M5 Pro and Gemma file SHA-256
`34be82b17b4942d389b9b527170c4b058027abdd32531fda063d3d97dd8ce80a`.
The native library SHA-256 remained
`64e85b3b33c5fe9042fa1af66fb0fb6f7f1fb7d79784612ae0dfa2dde3640d3f`,
built against unchanged upstream ggml
`456172ec733a135778adcd32d00e576a58232e45`. Exact outcomes and identities are
recorded in [gemma-batched-lifetime.json](gemma-batched-lifetime.json).

```sh
DOTNET_TieredCompilation=0 TS_TEST_GGML_BACKEND=metal \
TS_TEST_MODEL_DIR=/path/to/gemma-4-E4B MAX_CONTEXT=8192 \
dotnet test InferenceWeb.Tests/InferenceWeb.Tests.csproj -c Release --no-build \
  --filter 'FullyQualifiedName~Gemma4BatchedDecodeLifetimeTests|FullyQualifiedName~Gemma4BatchedFusedDecodeParityTests'
```

No new complete ABBA series was run, so its performance gate remains
unqualified. The old series has seventeen successful processes and one failed
control process, fifty-three of fifty-four requested measured passes. Fixing
the ownership defect does not retroactively validate that incomplete series.
CUDA validation of the final combined build is recorded separately by the
root task.
