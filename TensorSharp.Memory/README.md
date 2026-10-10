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
| `HostMemoryInfo` | Model/backend-independent host observations: Windows physical RAM, Linux visible cgroup hierarchy, macOS free/purgeable RAM, iOS/Mac Catalyst app allowance |
| `CudaResidencyBackend` / `CudaExecutionFence` | CUDA allocations, completion events and bounded host-staged multi-GPU copy; exercised on two A40s, direct P2P failed independent machine checks |
| Runtime `PagedKvStorage` | Actual capture/inject path uses scoped leases, bounded native scratch, SSD restore and best-effort next-page prefetch |
| Runtime `RequestMemoryAdmission` | Full peak reservation before prefix materialization, retained through physical release; budget/command wakeups instead of polling |
| GGML `GgmlCacheBudgetScope` | Optional native lazy-copy/preload allocation charges in an existing managed budget; install before caches, retain credit through physical release |
| `HostAllocationBudgetScope` | GGML host pool and `HostBuffers` allocations reserve aligned bytes before allocation; retained pool blocks and failed OS frees keep their credit |
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

`SchedulerConfig.KvSnapshots` selects bounded host snapshot storage in the existing
`InferenceEngine` and `BatchExecutor` per-sequence capture/inject path. Model-owned
paged arrays, per-request fused holders and retained end states are disabled for
this engine so those routes cannot silently bypass the configured storage. Linear
model forwards still use the backend's available GPU kernels. CLI inference sessions,
Server/Chat and TensorAgent engine hosts that use `SchedulerConfig.FromEnvironment`
can configure it without model-name switches:

```sh
export TS_SCHED_KV_RAM_BYTES=1073741824
export TS_SCHED_KV_SSD_BYTES=8589934592
export TS_SCHED_KV_SPILL_DIRECTORY=/path/on/ssd/tensorsharp-kv
```

These are **per-engine host-snapshot budgets**, not whole-model VRAM/RAM limits.
Programmatic configuration can instead share the same physical pools across
engines, native cache ownership and request admission:

```csharp
var budget = new MemoryBudget(new[] {
    new MemoryCharge("node0/ram", 2L << 30),
    new MemoryCharge("node0/ssd", 16L << 30),
    new MemoryCharge("node0/gpu0", 8L << 30),
});
var snapshots = KvSnapshotOptions.FromSharedBudget(
    budget, "node0/ram", "node0/ssd", spillDirectory);
// TensorSharp.GGML; install before loading/preloading native caches.
using var nativeCaches = new GgmlCacheBudgetScope(
    budget, new[] { new[] { "node0/gpu0" } });
// Set SchedulerConfig.KvSnapshots = snapshots when constructing engines.
// Stop work, dispose engines/models and clear native caches before scope disposal.
```

Import `TensorSharp.Runtime.Paged` and `TensorSharp.GGML` for the example. UMA
rank mappings may include both RAM and GPU pools to constrain one physical copy.
Shared capacities replace the independent per-engine limits; adapters release
only their own charges and can evict only their own pages. `MemoryUsage` then
reports the entire shared budget, including other owners. It may remain nonzero
after one engine is disposed. Do not also reserve these same cache/snapshot bytes
in a request envelope: this integration owns their allocation charges directly.

The default native scope covers lazy device-copy, explicit preload and selected
expert caches. Pass `includeGraphBuffers: true` to additionally cover the wired
context buffers and reuse gallocr arenas in common operators and the Gemma4/Qwen35
graphs, plus Qwen4Exp graph arenas, recurrent state and device state snapshots.
Qwen4Exp batched arena admission failures leave sequence holders available for a
retry after credit becomes available. Other executors, some live KV/holders, backend pools and driver overhead
remain outside it; this is not a whole-model allocation limit. Attaching after
covered allocation or detaching while allocations remain is rejected. Stop model
work, dispose models and release covered native caches/compute buffers before
scope disposal; failed disposal keeps callbacks and charges alive for a retry.

The four-argument overload can also charge the compact MoE expert file-read
arena and DeepSeek demand-read staging to process-wide host pools in the same budget:

```csharp
using var nativeCaches = new GgmlCacheBudgetScope(
    budget, new[] { new[] { "node0/gpu0" } },
    includeGraphBuffers: true, hostPools: new[] { "node0/ram" });
```

This replaces the scope in the earlier example; only one scope can be active.
Host staging competes atomically with other owners of those RAM pools, retains
its charge until physical release, and does not consume a discrete GPU pool.
Admission failure stops the expert operation before reading or allocating the
arena. Empty-cache trim releases the compact arena; model destruction releases
DeepSeek demand-read staging. A non-null host mapping must be nonempty;
omitting it preserves the previous cache/graph-only contract. Mapped source
pages, CPU long-prefill scratch, OS file cache and driver overhead remain outside
this host coverage, so it is not a process RSS limit.
Host coverage uses a distinct native export: mixing these managed assemblies
with an older native library fails at attachment instead of silently omitting
RAM accounting. Cache-only and graph-only constructors retain their old ABI.

RAM includes one full capture scratch, fixed staging and resident snapshot pages.
It must fit at least one resident page plus scratch and staging, each rounded to
64-byte allocation alignment. SSD bytes default to zero; each spilled page is
charged at 4096-byte file allocation alignment. An exhausted store reports pressure
and preserves the authoritative page.
Do not sum independently configured engine budgets beyond the machine's available
capacity. Native model KV, holders, weights, activations and process overhead still
need their own accounting/adapters. Construction rejects a model without complete,
cross-sequence-restorable host snapshots, a zero-size snapshot, or a block larger
than its restorable window. Qwen 3.5 enables this mode for the validated dense,
no-MTP, single-rank GGML CUDA path, including GDN recurrent state and attention KV;
its MoE, MTP, tensor-parallel and other backend paths still reject this mode.
The existing complete-holder route remains available when this mode is unset.
These flags do not virtualize native device state.
With prefix caching disabled, a lone request never swaps and produces no snapshots;
that case cannot establish spill/restore coverage.

`PagedKvStorage.Acquire` returns a scoped `KvSnapshotLease`. Raw `GetSpan` calls are
rejected for tiered storage. Prefix references retain logical pages while their
payload may be spilled; the final release removes the resource, and block-id reuse
gets a new epoch. Failed page cleanup retains the unreleased sequence references;
failed retained-payload release retains its queued keys and accounting for an
idempotent retry. The model's synchronous extract/inject contracts, partial-tail
layout and recurrent-prefix boundary rules remain unchanged. Capture uses a
pre-reserved scratch so a declined extraction cannot corrupt a prior snapshot.
Ownership swaps preserve published full pages and refresh only the mutable partial
tail; immutable pages do not incur a new SSD write on each decode turn.
Next-page prefetch only uses spare residency and is joined before recycling state.

An executor can set `SchedulerConfig.MemoryAdmission` to a `RequestMemoryAdmission`
with a shared `MemoryBudget` and a conservative request-cost function. This reserves
all declared pool peaks before prefix adoption or forwarding, bounds the waiting
queue, rejects impossible requests (including queued peaks that become impossible
after a capacity reduction), and retains reservations through finish,
preemption and cancellation until the engine's model-release hook succeeds.
Failed model release hooks retain their request ownership and charges, even after
generation has finished. `InferenceEngine.Dispose` retries these releases, reports
failures without declaring cleanup complete, and may be retried after recovery.
If its worker does not quiesce before the shutdown timeout, disposal also fails
instead of authorizing the caller to free buffers still used by a native step.
`SequenceState.MemoryEnvelope` lends allocation credit to request-owned resources.
Retained allocations must be charged separately or drawn from that envelope so
closing the request does not forget live memory. Estimates are not inferred from
model labels, and this policy is not automatically enabled for unadapted models.

When `KvSnapshots.SharedBudget` and `MemoryAdmission.Budget` are the same instance,
the executor automatically draws page/restore/prefetch allocations from the request
envelope. SSD spill and accelerator demotion follow the original allocation's live
envelope too. An insufficient envelope fails even if unrelated global credit is
available. After the envelope closes, retained prefix allocations remain charged;
their later relocation competes for global free capacity. Capture scratch and
transfer buffers are charged once for the engine, outside request peaks.

`RequestMemoryAdmission.ForKvSnapshots(options, blockBytes, blockTokens,
maxRunningRequests, executionHeadroomBytes, additionalPeak)` derives the snapshot
portion from actual model page geometry and shared headroom. It divides optional
RAM residency across planned concurrency and reserves spill space only when the
request cannot stay fully resident. Construct it before the engine allocates its
scratch/staging, after loading persistent owners. Recreate the policy at a quiescent
boundary when workload/capacity policy changes; submitted peaks stay fixed. Supply
other per-request peaks with `additionalPeak`, and separately charge persistent
weights/arenas. This helper does not make an incomplete model estimator complete.
Cold prefixes and other engines' caches still require explicit reclaim coordination.

### Qualified adaptive request accounting

`AdaptiveModelSession` now attaches both the native graph/cache scope and
`HostAllocationBudgetScope` before loading the model. The host scope covers wired
GGML pool allocations and owned host weight buffers. Pool reuse keeps the actual
larger allocation charged; returning a tensor to a pool is not a physical release.
Failed unmaps remain charged and can be retried. Model teardown trims the returned
host blocks before detaching the scopes. Failed adaptive construction exposes
`UnreleasedHostBudgetScope` as well as its native scope when cleanup cannot finish.

For the qualified dense single-rank text path, use
`session.CreateRequestMemoryAdmission(snapshots, blockTokens, maximumRunningRequests,
prefillChunkTokens)` with snapshots on that session's shared RAM pool. Create it
after loading persistent owners and before constructing the engine. Configure
`TS_SCHED_DISABLE_BATCHED=1`, disable prefix caching and speculation, and keep the
engine's prefill chunk within `session.Plan.SelectedChunkTokens`. Unsupported
execution paths are rejected instead of silently bypassing the adapter.

`EstimateRequestPeak` derives live host/device KV and recurrent state from model
geometry, rounds KV capacity to allocation blocks, allows old/new state to coexist
during growth, and includes graph workspace and streaming projection headroom.
Snapshot residency and possible spill are added separately. While each sequence
executes synchronously under the compute lock, its host/native allocations consume
its reserved envelope, including native worker callbacks. They cannot borrow a
different request's unused credit. Closing a request releases only unused credit;
retained buffers remain charged until physical release. This process-wide adapter
supports one serialized model lane, not concurrent engines or asynchronous scope use.

`MemoryBudget.HighWatermarks()` gives exact per-pool maxima since construction:
`Owned` includes unused reservations and committed bytes, while `Committed` is
instrumented allocation payload. These values are neither process RSS nor an OS
hard cap. The estimator remains conservative and is not a complete forecast for
GC/tokenizer memory, mapped-page residency, all native host scratch, vendor heaps,
media, batched holders or tensor-parallel execution. Shared capacities and request
residency plans still need refresh at a quiescent policy boundary.

The native paged-attention sessions also use graph-budget allocation/free hooks.
Their process-owned registry keeps CUDA buffers out of Windows thread destructors;
quiescent trim releases live and retired workers' cache payload before backend
teardown. Cache keys include backend identity. This does not permit trimming while
another model operation is executing.

The working-set API also accepts `ResourcePlacement` entries spanning multiple
devices. A partial failure releases every acquired pin. `ResourceLeaseSet` can
retire against a fence covering every rank. CUDA peer copies default **off** because
some PCIe/IOMMU topologies advertise support but corrupt data; unsupported routes
use the existing bounded host transfer slots. Before passing `enablePeerCopies:
true`, run the directed-pair validation on that machine. This is not a distributed
transaction coordinator or automatic tensor-parallel model integration.

## Explicit Qwen3.5 weight streaming

`TensorSharp.Models.WeightStreamingOptions` opts the dense, text-only Qwen3.5
GGML CUDA adapter into file-backed Q8_0 projections and embeddings:

```csharp
var budget = new MemoryBudget(new[] {
    new MemoryCharge("ram", 2L << 20),
    new MemoryCharge("gpu0", 2L << 20),
});
var streaming = new WeightStreamingOptions(budget, "ram", new[] { "gpu0" },
    tileBytes: 1 << 20, tokenTileRows: 32);
using var model = ModelBase.Create(modelPath, BackendType.GgmlCuda,
    weightStreaming: streaming);
```

This mode reads original GGUF ranges without full-file prefault, mmap, copied
fusion packs or weight preloads. A fixed host tile contains complete Q8_0 output
rows; each CUDA session owns only its aligned input, weight tile and output tile.
Rows and token batches shrink to fit the remaining shared constraints. A complete
input/reduction axis always stays together. Embeddings read only requested rows.
No tile pointer becomes a native cache key or captured graph input. Synchronous
completion makes it safe to overwrite the tile. Failed physical cleanup retains
the native handle and its reservation for a later disposal retry.
If a forward fails, earlier layers may already have changed KV/recurrent state.
Another `Forward`/`ForwardRefill` is refused until `ResetKVCache` succeeds; replay
the entire request after reset. An operator pressure error is not an atomic
rollback of the whole model step.

The example's 2 MiB capacities cover **streamed weight staging and operation
workspaces**, not the whole process or all device memory. Existing activations,
live KV/recurrent state, bounded resident F32 parameters, runtime/driver overhead
and the OS file cache require separate headroom. `StreamingWeightUsage` reports
file bytes read, executed tiles and workspace peaks;
`Qwen35Model.StreamingResidentParameterBytes` reports the explicitly whitelisted
F32 constants separately. Other quantization formats, tensor parallelism,
multimodal execution, speculation (including weight-free N-gram), MTP/draft and
unadapted model families are rejected rather
than silently loading all weights. Optional read-ahead reserves a second host
tile and starts the next file read before consuming the current tile. Small or
shared host/device quotas retain single-buffer execution, and disposal drains
pending reads before refunding their memory. CUDA copies remain synchronous;
read-ahead counters alone do not establish a throughput improvement.

## Adaptive dense loading

`TensorSharp.Models.AdaptiveModelSession` is an explicit single-CUDA execution
lane for dense Gemma4/Qwen35. It forecasts weights in their actual representations,
fusion peaks, KV/state and per-phase workspaces from model geometry and the
admitted context/prefill. It prefers the existing resident graph, then smaller
prefill chunks, and only then a supported file-streaming path. Hardware observation
uses available physical memory; Linux also observes the process's visible cgroup
hierarchy. Operator ceilings and headroom are independent controls.

```csharp
using var session = AdaptiveModelSession.Create(modelPath,
    new AdaptiveModelMemoryOptions(ContextTokens: 4096, PrefillTokens: 1024)
    { MaximumDeviceBytes = 12L << 30 });
// At an idle request boundary; false means stop admission and release/replan.
if (!session.RefreshCapacity()) throw new MemoryPressureException("Memory pressure");
float[] logits = session.Model.ForwardRefill(promptTokenIds);
int nextToken = session.Model.SampleGreedy(logits);
```

Do not share process-global GGML caches with another model during the session.
The session installs graph-budget hooks before loading and disposes its model
before detaching them. Host estimates are not an enforced RSS cap; mappings,
unhooked native pools and other processes still require headroom. Container
namespaces can hide further parent limits. Unsupported architectures, quantized
streaming layouts, speculation and multi-device configurations are not silently
substituted. A new request geometry needs a new plan at an idle boundary.
See [AdaptiveMemoryProbe](../eng/validation/AdaptiveMemoryProbe/README.md) for
actual resident comparisons and their model/device limits.

The overload `AdaptiveModelSession.Create(path, options, sharedBudget)` borrows
existing `AdaptiveModelSession.HostPool` and `DevicePool` constraints. It neither
raises nor rewrites caller-owned capacities, accounts for existing owners when
planning optional caches, and returns only its own credits at disposal. Host
staging hooks are enabled as well as routed device/graph hooks. This does not
permit a second simultaneous process-global GGML model/scope. Private session
capacity refresh uses `MemoryBudget.TrySetCapacities` to update pools atomically;
a failed shrink changes none. No-op refreshes and credit returned to a live request
envelope do not wake engines waiting for globally available capacity.

Linux observations include a conservative portion of the current cgroup's clean
inactive file cache, so an idle container filled by model reads is not mistaken
for anonymous-memory exhaustion. Active, dirty, mapped, shared and unevictable
pages are excluded; v2 requires a verified leaf with protection counters. The
same leaf credit is bounded by every visible ancestor and `/proc/meminfo`.
These observations remain forecasts with safety headroom, not atomic OS quotas.

## Platform boundaries

`HostMemoryInfo.Capture()` lives in the model-independent memory assembly and
does not initialize CUDA or MLX. Windows and Linux use their existing physical
memory/cgroup providers. macOS uses `hw.memsize` and Mach VM counters; iOS and
Mac Catalyst also cap availability by `os_proc_available_memory()`. Apple free
pages already include speculative pages. Inactive/dirty/compressed pages are
not assumed free, and zero app allowance grants zero new capacity. These are
conservative observations, not Jetsam protection or a process hard limit.

The ledger, residency scheduler, host KV paging, request admission and planner
use byte counts and capability/pool mappings, not a particular OS quota API.
UMA adapters must charge device allocations to physical RAM and the device
working-set constraint without inventing a second physical allocation. Linux
cgroups and CUDA MPS are optional external validation tools, never runtime
requirements. Other platforms need their own hard-limit acceptance evidence.

Host observation support does not qualify every executor: `AdaptiveModelSession`
still requires the single-CUDA dense adapter. Metal/MLX and iOS execution need
their own allocation hooks and measured workspace/UMA adapters before adopting
this entry point. Shared budgets remain explicit. Apple P/Invoke and real model
execution must be exercised on Apple hosts; Windows arithmetic/ABI tests alone
do not constitute Apple runtime acceptance.

For Qwen checkpoints with trailing NextN layers, adaptive loading fixes a
trunk-only `ModelMemoryPolicy` before construction. The geometry excludes the
same draft tensors and layers as the loader; active speculation is refused.
Resident execution and host snapshots may use this omitted-draft trunk.
NextN file streaming remains unqualified and is rejected. Direct callers without
this policy retain the existing learned-draft loading behavior.

Qwen geometry accepts an explicit attention head dimension independently of
embedding width. Its request forecast includes the retained capture-safe decode
graph alongside prefill scratch and live state: the decoder keeps per-layer
activation slots, so a one-layer scratch estimate can force slow fallback even
with free global budget. `NativeAllocationRefusals` exposes bounded diagnostic
evidence for this distinction without treating normal pressure as a callback error.

## Ownership and failure rules

- A resource key includes owner/revision, epoch and name. Tenant or adapter-specific
  state requires a different owner. Registering an existing key fails.
- Backing sources are borrowed and immutable. Keep their handles open and bytes
  unchanged until all registrations using them are removed.
- Allocate/copy methods must quiesce before returning or throwing. Buffers must report
  every budget pool they consume, including host mirrors. New buffers must be zeroed.
- An allocation whose initialization and physical cleanup both fail must be returned
  through `ResourceAllocationException`. Failed initialization or transfer rollback
  keeps the allocation quarantined and charged; `Unregister` retries its cleanup.
  Failed partial copies are never published as usable replicas.
- A lease is a single-consumer handle. Await its I/O before releasing it. Native GPU
  use needs a real completion fence, not the task that merely submitted the kernel.
- `ReleaseAfterAsync` does not release on a failed fence. Recover/synchronize the
  device before manually releasing it; otherwise retain/quarantine the allocation.
- `AcquireReadSetAsync` releases partial pins on pressure or conflicts. Retry the
  whole operator working set, microbatch, or use a supported tensor partition.
- Best-effort prefetch does not evict existing demand data. In-flight reads still
  consume reserved transfer bandwidth and memory. A physical allocator refusal
  is a prefetch miss. Demand allocation retries after evicting idle replicas from
  the affected pools, bounded by the initial resident set; it never evicts leases
  or borrows another request's reserved credit. A refused RAM demotion can spill
  directly to disk. Full disk or failed cleanup preserves the recovery source
  (or quarantined allocation), rather than returning fictitious free credit.
  `MemorySchedulerStats.PhysicalAllocationRefusals` distinguishes these physical
  refusals from logical quota exhaustion.
- One resource larger than a pool fails explicitly. The executor must provide a
  correct tiled kernel; arbitrary byte slicing alone does not make a kernel tiled.
- Spill restore checks its hash before publishing the destination. These files are
  temporary swap state, not durable restart checkpoints. No persistent KV sharing,
  crash recovery, encryption layer, multi-node transport or disk LRU is implemented.
  Writes complete and the file closes before atomic publication, without forcing
  per-page durable media synchronization. A complete snapshot overwrite avoids
  reading the superseded SSD version; allocation failure retains the old source.
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

When `SchedulerConfig.MemoryAdmission` blocks an otherwise idle engine, it first
reclaims its own parked model buffers and unlocked prefix payloads. Prefixes are
evicted in priority/LRU order, checking actual multi-pool headroom after every
release and stopping once the request can fit. Newest/public prefixes remain
evictable for admission; pins and pending donations still protect live state.
If credit is still insufficient, the worker sleeps outside the model lock and
wakes on a budget change or command. Its own partial reclaim does not busy-spin.
Blocked engines now publish pressure to other engines using the same ledger.
Each peer reclaims its own idle state on its own worker, under its own model lock,
only when no sequence is running and the compute gate is open. Notifications
coalesce once per budget generation; cancelled demand cannot cause later eviction.
This is cooperative reclamation, not global LRU or strict inter-engine fairness.
It does not permit concurrent AdaptiveModelSessions over process-global GGML caches,
revoke live owners, or supply missing model-specific peak estimates.
`MemoryBudget.CanReserve` only observes headroom; atomic reservation still decides
admission. A closed compute gate permits cancellation of unadmitted requests but
defers commands releasing or retaining model state until the gate opens.

`ContinuousBatchSchedulerTests.Engine_RepeatedPressureAndCancellationPreservesFifoProgressAndCredit`
accepts `TS_TEST_SHARED_MEMORY_SOAK_CYCLES=100000` (64 by default, at most 1000000).
Every cycle submits eight requests behind an external budget owner, cancels four,
and verifies survivors' FIFO progress, tokens and complete credit return. It uses
a controlled model, so a large cycle count does not establish long-running GPU
service stability or global fairness between engines.

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

The follow-up validation on the supplied two-A40 VM includes the Linux standalone
harness, runtime/placement/prefix regressions, five real CUDA checks on each GPU
and eleven two-device host-staged checks. Direct peer copies fail on this machine,
including an independent CUDA Runtime control, and are not enabled by default.
See [`UnifiedMemory.CudaProbe`](../eng/validation/UnifiedMemory.CudaProbe/README.md).

[`UnifiedMemory.ModelProbe`](../eng/validation/UnifiedMemory.ModelProbe/README.md)
compares actual Gemma 4 E2B Q4_K_M and dense Qwen 3.5 0.8B Q8_0 generated tokens
against independent requests and complete teacher-forced logit rows after repeated
file-backed snapshot restoration. Qwen support is limited to a single CUDA rank
without MTP layers; unsupported model/backend capabilities still fail explicitly.
These do not establish whole-model weight streaming, media support or
tensor-parallel snapshot coverage. The design records exact executed coverage,
known failures and benchmark limitations; generated evidence stays in ignored
`artifacts/` and is not committed.

[`UnifiedMemory.LongContextProbe`](../eng/validation/UnifiedMemory.LongContextProbe/README.md)
runs separate reference/candidate processes, records real prompt lengths, admitted
concurrency, request estimates, exact ledger maxima and observed RSS/device usage.
Its Linux cgroup wrapper refuses unavailable enforcement. The Windows
[`run-memory-job.py`](../eng/validation/run-memory-job.py) wrapper verifies a Job
private-commit ceiling with an allocation-refusal canary before running the model;
this does not cap RSS, file-cache residency or VRAM. The design records 8K,
32K, 64K and 128K configured-context results, without counting queued requests as
simultaneously running or logical quotas as physical hard limits.

Adaptive request admission binds its snapshot configuration, block size,
running-request limit and prefill workspace shape. `PrefillChunkTokenLimit`
also caps the scheduler's solo and contended prefill policies; the bound
admission can impose a smaller cap. A prefix-free, non-speculative engine
limited to one running request reserves execution memory without request
snapshot pages, even with queued followers. Engine capture/transfer scratch
remains charged. Gemma's estimate includes retained sliding-window prefill
staging; this is still a qualified adapter forecast, not a whole-process cap.
