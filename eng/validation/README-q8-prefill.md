# Bounded Q8 prefill experiment

`GgmlOpsQ8PrefillBench` is a standalone CUDA research target. It widens bounded
Q8_0 row tiles to exact F32 and uses pedantic F32 cuBLAS GEMM. Production model
dispatch is unchanged. Build it against unchanged upstream ggml and record its
revision, compiler, device and executable hash with every result.

```powershell
cmake --build TensorSharp.GGML.Native/build-windows --target GgmlOpsQ8PrefillBench
TensorSharp.GGML.Native/build-windows/GgmlOpsQ8PrefillBench.exe --check
TensorSharp.GGML.Native/build-windows/GgmlOpsQ8PrefillBench.exe --benchmark 1024 7168 643 32
```

Unset `TS_GGML_Q8_PARALLEL_VECTOR` and `TS_GGML_Q8_PARALLEL_SMALL_BATCH` for the
serial control. The benchmark arguments are K, M, N and scratch ceiling in MiB.
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

`--check` covers padded/interleaved inputs, tail rows, N=2/8/9/16/17/32/33/65,
full-output finite checks and canaries, exact admission and one-byte-short
rejection without physical allocation. Small fixtures check every result
against an independent scalar FP64 dot product. Benchmark fixtures check 31
evenly spaced rows by 17 columns, including boundaries; remaining output is
checked only for finiteness. Weight decoding uses independently constructed
logical scales/integers. Both candidate and existing serial control receive the
same absolute gate `1e-4 + 6e-6 * abs(reference)` and relative L2 gate `4e-6`.

Reports retain the number of failed samples, maximum error and relative L2 for
every tile. A failed tile is excluded from the reported best qualified tile;
any failed admitted tile or serial control returns 1. Rejected admission is
reported as such, not as an executed projection. Exit 77 means CUDA unavailable.
The default process may deliberately return 1 for a numerical investigation.

The initial K=1024/M=7168/N=643 run found several faster SGEMM tiles outside the
original absolute gate, despite relative L2 near 7e-7. The failing cancellation
values and analytical F32 error-bound diagnostics are retained. These are not
accepted performance candidates, and the gate was not relaxed to qualify them.
Those early runs overlapped CPU compilation; their timing is diagnostic only.
Version 5 also applies the absolute gate to the serial control, so an existing
kernel is never presumed to be ground truth. Generated evidence stays ignored
under `artifacts/unified-memory-adaptive/q8-prefill-*`.
