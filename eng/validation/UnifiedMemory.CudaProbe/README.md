# Unified-memory CUDA validation

This probe uses real CUDA allocations, an integer CUDA kernel, completion events,
the residency scheduler and real spill files. It needs .NET 10 and a working
NVIDIA driver. No model download, native GGML build, cuBLAS installation or nvcc is
needed: the tiny validation kernel is supplied as PTX and compiled by the driver.

Run from the repository root on the GPU machine:

```sh
dotnet run --project eng/tests/unified-memory/UnifiedMemory.Tests.csproj -c Release \
  -- --json artifacts/unified-memory/runtime.json

dotnet run --project eng/validation/UnifiedMemory.CudaProbe/UnifiedMemory.CudaProbe.csproj \
  -c Release -- --devices 0 --peer false --json artifacts/unified-memory/gpu0.json

dotnet run --project eng/validation/UnifiedMemory.CudaProbe/UnifiedMemory.CudaProbe.csproj \
  -c Release -- --devices 0,1 --peer false --json artifacts/unified-memory/multi-gpu-staged.json

# Separately validate every directed peer pair before opting into direct P2P.
dotnet run --project eng/validation/UnifiedMemory.CudaProbe/UnifiedMemory.CudaProbe.csproj \
  -c Release -- --devices 0,1 --peer true --json artifacts/unified-memory/multi-gpu-peer.json
```

Use CUDA-visible device ordinals. More than two devices can be listed; every
selected device executes the kernel, and every distinct directed device pair is
checked. The RAM budget covers one resident 1 MiB payload plus two 64 KiB transfer
slots. Each GPU has a 2 MiB declared allocation budget, forcing residency pressure
without filling the physical GPU. Payload quotas exclude test reference arrays,
driver/context/module overhead, runtime metadata and the OS page cache.

| Check | Evidence |
| --- | --- |
| Device execution and fence | Every selected GPU mutates all 262,144 uint32 elements; event retirement precedes byte-for-byte verification |
| Same-device copy | A direct device copy preserves a 1 MiB + 17-byte buffer, including its partial staging tail; zero initialization and cancelled writes are checked |
| VRAM/RAM/SSD pressure | Mutable state is evicted under one-page residency limits and restored exactly; a positive spill count is required |
| Concurrent readers | Sixteen callers acquire and verify the same completed device replica |
| Multi-GPU transfer | Full-buffer parity for every directed pair; reports peer or host-staged route |
| Collective lifetime | All selected device replicas remain leased until all rank events complete |
| Partial working set | Occupied GPU rejects another resource and releases the other rank's partial pin |
| Cleanup | Every declared budget returns to zero after physical resource disposal |
| Request envelopes | All selected GPU peaks reserve together; closing a request retains live device charges, blocks the next request, and admits it only after every rank is freed; CUDA free-memory counters confirm release |

Exit code **0** means all requested checks passed. **1** means a check failed,
including an explicitly requested peer route that is unsupported. **2** means
CUDA initialization or a selected device was unavailable. A one-device run does
not count as a multi-GPU test. JSON always records hardware availability and
individual completed checks; unavailable scenarios are never passing checks. All
directed routes are checked independently before the pressure cases. A failed
route records the first mismatching byte and both SHA-256 hashes, and prevents
the broader peer-enabled run from being reported as passing.

This is a scheduler/backend test, not production-model or tensor-parallel
inference validation. It neither proves asynchronous DMA/compute overlap nor
provides throughput claims. Run the existing model probes and the design's
quality/concurrency matrix separately with the actual model files, quantization,
device topology and concurrency settings. Some cloud PCIe/IOMMU configurations
advertise peer access but return corrupt data; staged copies remain the default.

Validation on 2026-10-07 used two NVIDIA A40 GPUs (46,068 MiB each), Ubuntu 24.04,
driver 570.211.01 and .NET SDK 10.0.401. GPU 0 and GPU 1 each passed all five
single-device checks. The two-device run with `--peer false` passed all eleven
checks, including both directed staged routes and physical request cleanup.
Both direct peer routes failed full-buffer parity despite advertised support and
a PCIe PIX topology. These failures are not counted as passing multi-GPU peer
validation. Keep `enablePeerCopies` disabled on this deployment.

`peer-copy-check.cu` independently checks the same hardware through the CUDA
Runtime without TensorSharp or .NET. It compares every word after both synchronous
and asynchronous peer copies in both directions:

```sh
nvcc -arch=sm_86 eng/validation/UnifiedMemory.CudaProbe/peer-copy-check.cu \
  -o artifacts/unified-memory/peer-copy-check
artifacts/unified-memory/peer-copy-check 0 1
```

With CUDA 12.8.93 on this VM, all four native peer-copy checks also failed parity.
The GPU 0 -> 1 copies mismatched all 262,144 words; GPU 1 -> 0 mismatched 245,760.
This independent control establishes that the direct route is unsafe on this
deployment; it does not identify the underlying driver/platform cause. Logs and
JSON results belong in ignored `artifacts/unified-memory/`. These residency probes
do not link ggml or validate model inference; separate native/model validation must
record the unchanged upstream revision and its actual coverage.

For GGML model execution on this topology, configure the TensorSharp native build
with `-DGGML_CUDA_NO_PEER_COPY=ON`; TensorSharp's CMake configuration preserves this
explicit upstream option. This selects ggml's existing host-copy fallback without
changing upstream sources. The residency adapter's `enablePeerCopies: false` flag
does not configure ggml or NCCL. The independent model comparison uses
`GGML_CUDA_ALLREDUCE=none` and `NCCL_P2P_DISABLE=1` together with the staged native
build, and checks full teacher-forced logit rows through `eng/ForcedLogitProbe`.
These transport choices provide a correctness control, not a throughput claim.
