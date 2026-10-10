# Bounded Q8 prefill experiment

`GgmlOpsQ8PrefillBench` is a standalone CUDA research target. It widens bounded
Q8_0 row tiles to exact F32 and uses pedantic F32 cuBLAS GEMM. Production model
dispatch is unchanged. Build it against unchanged upstream ggml and record its
revision, compiler, device and executable hash with every result.

```powershell
cmake --build TensorSharp.GGML.Native/build-windows --target GgmlOpsQ8PrefillBench
TensorSharp.GGML.Native/build-windows/GgmlOpsQ8PrefillBench.exe --check
TensorSharp.GGML.Native/build-windows/GgmlOpsQ8PrefillBench.exe --benchmark 1024 7168 643 32
TensorSharp.GGML.Native/build-windows/GgmlOpsQ8PrefillBench.exe --benchmark 1024 7168 643 32 128
```

Unset `TS_GGML_Q8_PARALLEL_VECTOR` and `TS_GGML_Q8_PARALLEL_SMALL_BATCH` for the
serial control, and unset `TS_GGML_Q8_PREFILL_TILE` or set it to `32`.
The executable explicitly pins that control to 32 before the first launch;
production automatic tiling must not silently change its historical timings.
The benchmark arguments are K, M, N, scratch ceiling in MiB and an optional
inner-product chunk (zero/default disables it; otherwise a multiple of 32 up to K).
Integer arguments reject signs, suffixes and overflow instead of silently
selecting a different experiment.
No arguments after `--benchmark` runs several Qwen projection geometries at two
ceilings. Run one GPU process at a time, exclude concurrent builds/downloads,
and retain stderr together with stdout. Performance measurements are synthetic
projection timings, including widening and optional input packing. They do not
measure model prefill or establish an optimal tile choice.

One owned arena reserves its actual payload through `SharedCacheCharge` kind 2
before allocation. Its charge includes 4 MiB explicit cuBLAS workspace, aligned
F32 weight rows and, for strided input, the packed activation tile. It returns
credit after synchronization, handle destruction and device release. Baseline
weights/input/output, handle metadata, driver and cuBLAS internal allocations
are excluded and reported separately; this is not a process VRAM cap. The test
uses process-fatal checks and is not a production failure-recovery owner.

The optional chunked experiment computes successive short pedantic F32 GEMMs,
adds their partial results in FP64 and converts once to F32. Weights and inputs
keep their original precision. This changes reduction order and uses two more
budgeted output tiles: aligned `rows*columns*sizeof(float)` partials and aligned
`rows*columns*sizeof(double)` accumulators. There is no unaccounted K-dependent
stack of partial matrices. Tail columns/rows scatter to the full output stride;
the last K chunk may be shorter. This research path does not use TF32 or narrow
activations and is not a production dispatcher option.

`--check` covers padded/interleaved inputs, tail rows, N=2/8/9/16/17/32/33/65,
full-output finite checks and canaries, exact admission and one-byte-short
rejection without physical allocation. Small fixtures check every result
against an independent scalar FP64 dot product. Benchmark fixtures check 31
evenly spaced rows by 17 columns, including boundaries; remaining output is
checked only for finiteness. Weight decoding uses independently constructed
logical scales/integers. Both candidate and existing serial control receive the
same absolute gate `1e-4 + 6e-6 * abs(reference)` and relative L2 gate `4e-6`.
The small suite repeats all shapes/admission checks with chunks 0, 32 and 64,
including the K=96/chunk=64 tail. Existing serial control failures remain failures.

Reports retain the number of failed samples, maximum error and relative L2 for
every tile. A failed tile is excluded from the reported best qualified tile;
any failed admitted tile or serial control returns 1. Rejected admission is
reported as such, not as an executed projection. A benchmark with no admitted
candidate also returns 1; only the explicit `--check` refusal cases expect that
condition. Reports separate admitted and qualified candidate counts.
Exit 77 means CUDA unavailable.
The default process may deliberately return 1 for a numerical investigation.

The initial K=1024/M=7168/N=643 run found several faster SGEMM tiles outside the
original absolute gate, despite relative L2 near 7e-7. The failing cancellation
values and analytical F32 error-bound diagnostics are retained. These are not
accepted performance candidates, and the gate was not relaxed to qualify them.
Those early runs overlapped CPU compilation; their timing is diagnostic only.
Version 5 also applies the absolute gate to the serial control, so an existing
kernel is never presumed to be ground truth. Generated evidence stays ignored
under `artifacts/unified-memory-adaptive/q8-prefill-*`.

Version 6's K=1024/M=7168/N=643, 32 MiB experiments retain the same gates. All
admitted chunk-128 and chunk-64 tiles pass the 527-sample oracle, but they take
longer than the measured serial control. Some chunk-256 tiles still fail the
absolute gate (one cancellation error is only slightly above the unchanged bound).
The old serial control still fails two samples, so all four benchmark processes
exit 1. None is promoted or claimed to solve prefill performance. Small checks
also completed under CUDA Compute Sanitizer memcheck with zero errors/leaked
bytes. The sanitizer binary identity precedes later CLI and empty-admission reporting changes;
the final executable receives its own small numerical checks and identity record.
These results are one synthetic geometry without locked clocks or a balanced
model benchmark, and do not establish whole-model precision or throughput.
