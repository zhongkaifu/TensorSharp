# TensorSharp unified memory foundation

This package implements a model-independent residency and budgeting core for host
RAM, accelerator allocations and file-backed resources. It is an **opt-in foundation**,
not a switch that makes all existing TensorSharp models out-of-core.

The integration extracts the existing Qwen4Exp CUDA/UMA placement policy into
`TieredPlacementPlanner`, connects the continuous executor's host KV snapshots to
bounded RAM/SSD storage, and supports executor-supplied multi-pool request admission.
Existing model constructors, captured GGML graphs, native KV/holder arenas and media
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
| `CudaResidencyBackend` / `CudaExecutionFence` | CUDA allocations, completion events, direct device copy and opt-in P2P; compiled, **not exercised on hardware** |
| Runtime `PagedKvStorage` | Actual capture/inject path uses scoped leases, bounded native scratch, SSD restore and best-effort next-page prefetch |
| Runtime `RequestMemoryAdmission` | Full peak reservation before prefix materialization, retained through physical release; budget/command wakeups instead of polling |
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

## Runtime integration

`SchedulerConfig.KvSnapshots` enables bounded host snapshot storage in the existing
`InferenceEngine` and `BatchExecutor` capture/inject path. CLI inference sessions,
Server/Chat and TensorAgent engine hosts that use `SchedulerConfig.FromEnvironment`
can configure it without model-name switches:

```sh
export TS_SCHED_KV_RAM_BYTES=1073741824
export TS_SCHED_KV_SSD_BYTES=8589934592
export TS_SCHED_KV_SPILL_DIRECTORY=/path/on/ssd/tensorsharp-kv
```

These are **per-engine host-snapshot budgets**, not whole-model VRAM/RAM limits.
RAM includes one full capture scratch, fixed staging and resident snapshot pages.
It must fit at least one resident page plus scratch and staging. SSD bytes default
to zero; an exhausted store reports pressure and preserves the authoritative page.
Do not sum independently configured engine budgets beyond the machine's available
capacity. Native model KV, holders, weights, activations and process overhead still
need their own accounting/adapters. Models that do not export host snapshots have
no data on this path; the flags do not virtualize their native device state.

`PagedKvStorage.Acquire` returns a scoped `KvSnapshotLease`. Raw `GetSpan` calls are
rejected for tiered storage. Prefix references retain logical pages while their
payload may be spilled; the final release removes the resource, and block-id reuse
gets a new epoch. The model's synchronous extract/inject contracts, partial-tail
layout and recurrent-prefix boundary rules remain unchanged. Capture uses a
pre-reserved scratch so a declined extraction cannot corrupt a prior snapshot.
Next-page prefetch only uses spare residency and is joined before recycling state.

An executor can set `SchedulerConfig.MemoryAdmission` to a `RequestMemoryAdmission`
with a shared `MemoryBudget` and a conservative request-cost function. This reserves
all declared pool peaks before prefix adoption or forwarding, bounds the waiting
queue, rejects impossible requests, and retains reservations through finish,
preemption and cancellation until the engine's model-release hook succeeds.
`SequenceState.MemoryEnvelope` lends allocation credit to request-owned resources.
Retained allocations must be charged separately or drawn from that envelope so
closing the request does not forget live memory. Estimates are not inferred from
model labels, and this policy is not automatically enabled for unadapted models.

The working-set API also accepts `ResourcePlacement` entries spanning multiple
devices. A partial failure releases every acquired pin. `ResourceLeaseSet` can
retire against a fence covering every rank. CUDA peer copies default **off** because
some PCIe/IOMMU topologies advertise support but corrupt data; unsupported routes
use the existing bounded host transfer slots. Before passing `enablePeerCopies:
true`, run the directed-pair validation on that machine. This is not a distributed
transaction coordinator or automatic tensor-parallel model integration.

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

The follow-up implementation passes 30 standalone cases and 66 selected runtime /
placement regression cases, including state-dependent token parity between resident
and spilling concurrent inference and recurrent checkpoint preservation. These use
synthetic models, not production model files. The hardware probe is documented in
[`eng/validation/UnifiedMemory.CudaProbe`](../eng/validation/UnifiedMemory.CudaProbe/README.md).
It compiles; the development session has neither a CUDA driver nor access to the
offered SSH VM, so real single-/multi-GPU validation remains pending.
