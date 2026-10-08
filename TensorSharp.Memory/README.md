# TensorSharp unified memory foundation

This package implements a model-independent residency and budgeting core for host
RAM, accelerator allocations and file-backed resources. It is an **opt-in foundation**,
not a switch that makes all existing TensorSharp models out-of-core.

The current integration extracts the existing Qwen4Exp CUDA/UMA placement policy into
`TieredPlacementPlanner` and calls it from the original model path. Existing model
constructors, captured GGML graphs, KV implementations, server admission and media
pipelines still need the execution adapters described in
[the design and coverage plan](../docs/design/unified-memory.zh-CN.md).

## Implemented

| Component | Behavior |
| --- | --- |
| `MemoryBudget` | Atomic reservations across named pools; reserve before allocate; request envelopes transfer credit to allocations without counting twice |
| `TieredMemoryScheduler` | Shared read leases, exclusive writes, resource epochs/versions, single-flight loads, LRU of resident allocations, host demotion, safe eviction |
| `BoundedTransfers` | Pre-reserved, fixed-size native staging buffers; bounded concurrent transfers; exact reads; cancellation rollback |
| `FileDataSource` / `ResourceSlice` | Shared shard handles and exact byte ranges; no full tensor temporary, mmap or prefault |
| `SsdSpillStore` | Private process-lifetime files, quota, atomic publication, SHA-256 validation of restored mutable state |
| `MemoryRequestQueue` | Bounded FIFO queue; full declared request-peak reservation; cancellation; explicit pressure instead of overcommit |
| `HostMemoryBackend` | Actual native host allocations and copies |
| `CudaResidencyBackend` | Explicit CUDA allocation/copy/free adapter, compiled here; **not exercised on hardware** |
| `GgufMemoryCatalog` | GGUF and split-GGUF tensor regions, original quantized bytes, operator-defined slices |

`ResourceKind` is descriptive metadata, not a model-specific policy switch. The core
does not reinterpret layouts, quantize weights, omit experts, shorten context or
change model precision. A model adapter must supply the complete legal working set
and preserve its own operator, state, shape and positional semantics.

## Minimal API use

```csharp
using TensorSharp.Memory;
using TensorSharp.Runtime;

// catalog must outlive scheduler registrations; one handle per GGUF shard.
using var gguf = new GgufFile(modelPath);
using var catalog = gguf.CreateMemoryCatalog(exactModelRevision, epoch: 1);
var budget = new MemoryBudget(new[] {
    new MemoryCharge("local/ram", 512L << 20),
    new MemoryCharge("local/ssd", 8L << 30),
});
var host = new HostMemoryBackend("local/ram");
using var transfers = new BoundedTransfers(budget, "local/ram", 1 << 20, 2);
using var spill = new SsdSpillStore(budget, "local/ssd", spillDirectory, transfers);
await using var memory = new TieredMemoryScheduler(
    budget, new[] { host }, transfers, spill, host.Location);

var (resource, source) = catalog.Get(tensorName);
memory.Register(resource, source);
using (var lease = await memory.AcquireAsync(resource.Key, host.Location)) {
    // Bind lease.Pointer only for the duration of this lease. If submitting
    // asynchronous work, await lease.ReleaseAfterAsync(aRealCompletionFence).
    byte[] firstBytes = new byte[(int)Math.Min(64, lease.ByteLength)];
    await lease.ReadAsync(0, firstBytes);
}
memory.Unregister(resource.Key);
```

This example reads tensor bytes; it does not invoke an inference kernel. The runnable
test harness includes a batched F32 matrix-vector workload that streams an on-disk
matrix larger than its managed payload budget and checks every output against a
mathematical reference.

For request-owned allocations, pass `AdmittedMemoryRequest.Envelope` as
`allocationEnvelope` to `AcquireAsync`. When such an allocation is freed, its credit
returns to the live request. Closing a request frees only unused credit; surviving
allocations remain charged until actually freed. Shared weights should use a model
lifetime rather than a request lifetime. Request peak estimates must include maximum
KV growth, recurrent state, scratch and temporaries, with allocator alignment.

## Ownership and failure rules

- A resource key includes owner/revision, epoch and name. Tenant or adapter-specific
  state requires a different owner. Registering an existing key fails.
- Backing sources are borrowed and immutable. Keep their handles open and bytes
  unchanged until all registrations using them are removed.
- Allocate/copy methods must quiesce before returning or throwing. Buffers must report
  every budget pool they consume, including host mirrors. New buffers must be zeroed.
- A lease is a single-consumer handle. Await its I/O before releasing it. Native GPU
  use needs a real completion fence, not the task that merely submitted the kernel.
- `ReleaseAfterAsync` does not release on a failed fence. Recover/synchronize the
  device before manually releasing it; otherwise retain/quarantine the allocation.
- `AcquireReadSetAsync` releases partial pins on pressure or conflicts. Retry the
  whole operator working set, microbatch, or use a supported tensor partition.
- Best-effort prefetch does not evict existing demand data. In-flight reads still
  consume reserved transfer bandwidth and memory.
- One resource larger than a pool fails explicitly. The executor must provide a
  correct tiled kernel; arbitrary byte slicing alone does not make a kernel tiled.
- Spill restore checks its hash before publishing the destination. These files are
  temporary swap state, not durable restart checkpoints. No persistent KV sharing,
  crash recovery, encryption layer, multi-node transport or disk LRU is implemented.
- Close order: quiesce execution, unregister/dispose scheduler, dispose spill store,
  dispose transfer buffers, close source catalogs/backend contexts.

Budgets account for declared live allocation payloads and alignment. They are **not
process RSS limits**: .NET metadata, native allocator/driver overhead, OS page cache,
existing native caches and other processes need separate measurement and headroom.
The file backend uses buffered I/O. The CUDA adapter deliberately uses completed
copies and explicit default-stream synchronization; it does not yet overlap DMA
with compute. CUDA driver granularity is a conservative configurable estimate and
must be calibrated against measured free memory on the target device.

## Validation

```sh
dotnet run --project eng/tests/unified-memory/UnifiedMemory.Tests.csproj -c Release \
  -- --json artifacts/unified-memory/results.json
```

The harness runs real host allocations and filesystem I/O, including concurrent
mutation/eviction, cancellation, failure rollback, checksum failure, request budgeting,
GGUF shards and byte-exact restoration. Accelerator state-machine tests are explicitly
labelled host-backed simulations. The physical filesystem medium is not assumed to
be NVMe. The corruption-injection case uses Unix semantics; the CI lane is Linux x64
and ARM64. No hardware/model case is counted as a passing test without running it.

The design document records current test results, remaining integrations and the
quality/latency matrix required before enabling this core for production models.
