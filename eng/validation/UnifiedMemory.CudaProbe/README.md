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
| VRAM/RAM/SSD pressure | Mutable state is evicted under one-page residency limits and restored exactly; a positive spill count is required |
| Concurrent readers | Sixteen callers acquire and verify the same completed device replica |
| Multi-GPU transfer | Full-buffer parity for every directed pair; reports peer or host-staged route |
| Collective lifetime | All selected device replicas remain leased until all rank events complete |
| Partial working set | Occupied GPU rejects another resource and releases the other rank's partial pin |
| Cleanup | Every declared budget returns to zero after physical resource disposal |

Exit code **0** means all requested checks passed. **1** means a check failed,
including an explicitly requested peer route that is unsupported. **2** means
CUDA initialization or a selected device was unavailable. A one-device run does
not count as a multi-GPU test. JSON always records hardware availability and
individual completed checks; unavailable scenarios are never passing checks.

This is a scheduler/backend test, not production-model or tensor-parallel
inference validation. It neither proves asynchronous DMA/compute overlap nor
provides throughput claims. Run the existing model probes and the design's
quality/concurrency matrix separately with the actual model files, quantization,
device topology and concurrency settings. Some cloud PCIe/IOMMU configurations
advertise peer access but return corrupt data; staged copies remain the default.

At this revision the probe builds successfully but real GPU execution is pending.
The offered SSH environment was unreachable from the development session and the
specified SSH key was unavailable. No remote files or installed software were changed.
