// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Memory.Hardware;
using TensorSharp.Memory.Planning;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models;

/// <summary>One sequential text execution lane. Context includes generated tokens.
/// Limits are operator ceilings, not a fraction of a model's file size.</summary>
public sealed record AdaptiveModelMemoryOptions(int ContextTokens, int PrefillTokens)
{
    public long MaximumHostBytes { get; init; } = long.MaxValue;
    public long MaximumDeviceBytes { get; init; } = long.MaxValue;
    public long? HostHeadroomBytes { get; init; }
    public long? DeviceHeadroomBytes { get; init; }
    /// <summary>Forecast for additional uninstrumented host growth at a request
    /// boundary, beyond observed resident use. This is not GC/tokenizer ownership.</summary>
    public long HostUninstrumentedBurstBytes { get; init; } = 128L << 20;
    /// <summary>Additional backend/driver growth beyond currently reported use.</summary>
    public long DeviceUninstrumentedBurstBytes { get; init; } = 64L << 20;
    public int MaximumPrefillChunkTokens { get; init; } = 2048;
    public bool AllowWeightStreaming { get; init; } = true;
    public int StreamingTileBytes { get; init; } = 16 << 20;
    /// <summary>Ceiling for optional RAM weight reuse. Actual capacity comes from
    /// physical availability minus the admitted request peak; zero disables reuse.</summary>
    public long MaximumStreamingHostCacheBytes { get; init; } = long.MaxValue;
    /// <summary>Ceiling for optional complete device weight arenas. Shared physical
    /// availability and execution-phase workspace forecasts further bound retention.</summary>
    public long MaximumStreamingDeviceCacheBytes { get; init; } = long.MaxValue;
    /// <summary>Automatic idle CUDA workspace reuse, bounded by hardware/request
    /// slack and a share of optional device retention. Zero explicitly disables it.</summary>
    public long MaximumStreamingWorkspaceCacheBytes { get; init; } = long.MaxValue;
}

public readonly record struct InferenceHardwareMemory(long HostTotal, long HostAvailable,
    long DeviceTotal, long DeviceAvailable)
{
    /// <summary>Optional independent allowance for new host commitments. This is
    /// not extra RAM or proof of a hard cap. CUDA/driver commitment is not owned
    /// by the host ledger and can consume this allowance between observations.</summary>
    public long? HostCommitLimit { get; init; }
    public long? HostCommitAvailable { get; init; }
    /// <summary>Read physical availability, not GC heap usage or an earlier budget.
    /// Unknown measurements fail closed. Initializing the CUDA backend is explicit.</summary>
    public static InferenceHardwareMemory CaptureCuda()
    {
        GgmlBasicOps.EnsureBackendAvailable(GgmlBackendType.Cuda);
        if (!GgmlBasicOps.TryGetDeviceMemoryInfo(out long free, out long total) || total <= 0)
            throw new NotSupportedException("The backend did not report device memory availability.");
        var host = HostMemoryInfo.CaptureAllocationConstraints();
        return new(host.Total, host.Available, total, free)
        { HostCommitLimit = host.CommitLimit, HostCommitAvailable = host.CommitAvailable };
    }

    internal static (long Total, long Available) CaptureHost() => HostMemoryInfo.Capture();

    internal void ValidateCommit()
    {
        if (HostCommitLimit.HasValue != HostCommitAvailable.HasValue
            || HostCommitLimit is <= 0 || HostCommitAvailable is < 0
            || HostCommitAvailable > HostCommitLimit)
            throw new ArgumentException("Host commit observation must be a complete, finite limit/available pair.");
    }

    internal (long Total, long Available) HostAllocationCapacity()
    {
        // Validate each observation before intersecting: a tighter commit
        // ceiling must not conceal an impossible physical sample.
        if (HostTotal <= 0 || HostAvailable < 0 || HostAvailable > HostTotal)
            throw new ArgumentException("Invalid physical host capacity observation.");
        ValidateCommit();
        return (Math.Min(HostTotal, HostCommitLimit ?? long.MaxValue),
            Math.Min(HostAvailable, HostCommitAvailable ?? long.MaxValue));
    }
}

/// <summary>Hardware/request-aware loading for dense Gemma4/Qwen35 on one CUDA
/// device. A legal resident graph is preferred to row streaming, so adding a
/// budget does not force every token through SSD/PCIe. Model geometry estimates
/// are admission forecasts; only allocations instrumented by the GGML scope and
/// streaming allocator are enforced. OS page cache, driver pools and managed
/// objects are not a hard RSS quota. Do not share process-global GGML caches
/// with another model while this session is alive.</summary>
public sealed class AdaptiveModelSession : IDisposable
{
    public const string HostPool = "adaptive/ram";
    public const string DevicePool = "adaptive/cuda0";
    private readonly AdaptiveModelMemoryOptions _options;
    private readonly GgmlCacheBudgetScope _nativeBudget;
    private readonly HostAllocationBudgetScope _hostBudget;
    private readonly DenseMemoryProfile _profile;
    private readonly bool _ownsBudget;
    private readonly long _hostCacheReserveBytes;
    private readonly long _deviceCacheReserveBytes;
    private readonly PhysicalMemoryCapacityPlanner _physicalPlanner = new();
    private bool _disposed;

    private AdaptiveModelSession(ModelBase model, MemoryBudget budget, GgmlCacheBudgetScope nativeBudget,
        InferenceMemoryPlan plan, AdaptiveModelMemoryOptions options, long hostCacheReserveBytes, long deviceCacheReserveBytes,
        bool ownsBudget, HostAllocationBudgetScope hostBudget, DenseMemoryProfile profile)
    { Model = model; Budget = budget; _nativeBudget = nativeBudget; Plan = plan; _options = options;
        _hostCacheReserveBytes = hostCacheReserveBytes; _deviceCacheReserveBytes = deviceCacheReserveBytes; _ownsBudget = ownsBudget;
        _hostBudget = hostBudget; _profile = profile; }

    public ModelBase Model { get; }
    public MemoryBudget Budget { get; }
    public InferenceMemoryPlan Plan { get; }
    /// <summary>Latest admission-boundary measurements. Unknown physical owners
    /// remain explicit; neither this plan nor its margins establish an OS hard cap.</summary>
    public PhysicalMemoryCapacityPlan? LastPhysicalCapacityPlan { get; private set; }
    public Exception? AccountingError => _nativeBudget.CallbackError;
    public (long Bytes, long PeakBytes, int Allocations) HostAllocationUsage => _hostBudget.Usage;
    public IReadOnlyList<GgmlAllocationUsage> NativeAllocationUsage => _nativeBudget.AllocationUsage;
    public (long Count, GgmlAllocationRefusal? First, GgmlAllocationRefusal? Last) NativeAllocationRefusals => _nativeBudget.AllocationRefusals;

    public static AdaptiveModelSession Create(string path, AdaptiveModelMemoryOptions options)
        => CreateCore(path, options, null);

    /// <summary>Borrow a shared ledger containing HostPool and DevicePool. The
    /// caller owns capacities; loading/refresh never raises or rewrites them.
    /// Other RAM owners (including KV snapshots) may use the same ledger. Only
    /// one process-global GGML scope/model is supported, as in the private form.</summary>
    public static AdaptiveModelSession Create(string path, AdaptiveModelMemoryOptions options, MemoryBudget budget)
    {
        ArgumentNullException.ThrowIfNull(budget);
        return CreateCore(path, options, budget);
    }

    private static AdaptiveModelSession CreateCore(string path, AdaptiveModelMemoryOptions options, MemoryBudget? sharedBudget)
    {
        ArgumentNullException.ThrowIfNull(options);
        Validate(options);
        if (TensorSharp.Runtime.Speculative.SpeculationOptions.FromEnvironment().Enabled)
            throw new NotSupportedException("AdaptiveModelSession admits trunk-only execution; speculative execution requires a qualified memory adapter.");
        DenseMemoryProfile profile;
        using (var gguf = GgufFile.OpenForMemoryPlanning(path))
            profile = DenseMemoryProfile.Read(gguf, options, ModelBase.RetainsAllHostQuantizedWeights, omitEmbeddedDraftWeights: true);
        var hardware = InferenceHardwareMemory.CaptureCuda();
        var budget = sharedBudget ?? new MemoryBudget([new(HostPool, options.MaximumHostBytes), new(DevicePool, options.MaximumDeviceBytes)]);
        var plan = PlanLoad(profile, options, hardware, budget, sharedBudget != null);
        if (!plan.Accepted)
            throw new MemoryPressureException(string.Join(Environment.NewLine, plan.Rejections.Select(r => r.Reason).Distinct()));
        if (sharedBudget == null)
            budget.RefreshCapacities(plan.Capacities.Select(c => new MemoryCharge(c.Pool, c.DesiredCapacity)));

        // Install before model preload. Callback reservations arbitrate actual
        // buffer sizes/rounding, rather than committing the whole forecast and
        // charging those same bytes a second time in allocation callbacks.
        var nativeBudget = new GgmlCacheBudgetScope(budget, new[] { new[] { DevicePool } }, includeGraphBuffers: true, hostPools: [HostPool]);
        HostAllocationBudgetScope? hostBudget = null;
        try
        {
            hostBudget = new(budget, [HostPool]);
            WeightStreamingOptions? streaming = plan.SelectedCandidate!.Placement == InferenceWeightPlacement.SsdStreaming
                ? new(budget, HostPool, [DevicePool], options.StreamingTileBytes, Math.Min(32, plan.SelectedChunkTokens))
                {
                    HostCacheBytes = HostCacheLimit(plan, profile.SourceWeightBytes, options.MaximumStreamingHostCacheBytes),
                    // Retention starts after loading. The required read tile is
                    // already charged; leave the rest of the execution forecast
                    // available. An optional second tile still competes for slack.
                    HostCacheReserveBytes = Math.Max(0, HostExecutionPeak(plan) - options.StreamingTileBytes),
                    DeviceCacheBytes = DeviceCacheLimit(plan, profile.SourceWeightBytes, options.MaximumStreamingDeviceCacheBytes),
                    DeviceCacheReserveBytes = ExecutionPeak(plan, DevicePool),
                    DeviceWorkspaceCacheBytes = DeviceCacheLimit(plan, profile.SourceWeightBytes, options.MaximumStreamingWorkspaceCacheBytes),
                    DeviceWorkspaceCacheReserveBytes = ExecutionPeak(plan, DevicePool)
                } : null;
            var policy = new ModelMemoryPolicy(options.ContextTokens, plan.SelectedChunkTokens) { OmitEmbeddedDraftWeights = true };
            var model = ModelBase.Create(path, BackendType.GgmlCuda, 1, null!, null!, 1, streaming!, policy);
            return new(model, budget, nativeBudget, plan, options, streaming?.HostCacheReserveBytes ?? 0,
                streaming?.DeviceCacheReserveBytes ?? 0, sharedBudget == null, hostBudget, profile);
        }
        catch (Exception creation)
        {
            // Constructors with a memory policy unwind their owned resources.
            // Never detach a callback with a live native allocation.
            try { hostBudget?.Dispose(); nativeBudget.Dispose(); }
            catch (Exception cleanup)
            {
                throw new AdaptiveModelAllocationException(nativeBudget, budget, new AggregateException(creation, cleanup), hostBudget);
            }
            throw;
        }
    }

    /// <summary>Refresh between forwards while the caller owns the model compute
    /// lock and the compute gate is open. Other admitted requests may retain live
    /// state; idle trimming never releases that state. Shrinking does not revoke
    /// live owners or discard KV. False means stop new admission; release work or
    /// reload a smaller supported placement. Do not vary graph shapes per token.</summary>
    public bool RefreshCapacity()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var hw = InferenceHardwareMemory.CaptureCuda();
        long cachedBytes = Model.StreamingWeightUsage?.HostCacheBytes ?? 0;
        var streaming = Model.StreamingWeightUsage;
        long deviceCachedBytes = checked((streaming?.DeviceCacheBytes ?? 0) + (streaming?.DeviceWorkspaceCacheBytes ?? 0));
        var physical = CapturePhysicalPlan(hw);
        if (!physical.Accepted) return false;
        if (physical.Pools.Any(pool => RequiresIdleTrim(pool, _hostCacheReserveBytes, cachedBytes,
                _deviceCacheReserveBytes, deviceCachedBytes)))
        {
            Model.TrimIdleMemory();
            hw = InferenceHardwareMemory.CaptureCuda();
            physical = CapturePhysicalPlan(hw);
            if (!physical.Accepted) return false;
        }
        bool admitted = physical.Pools.All(p => InferenceMemoryPlanner.GetCapacity(p).OverTargetBytes == 0);
        // Persist the target: after an owner releases, its former protective floor
        // must not reopen credit while the machine remains under pressure.
        if (_ownsBudget) admitted &= Budget.RefreshCapacities(physical.DesiredCapacities).All(p => p.OverTarget == 0);
        return admitted && AccountingError == null;
    }

    private static long DesiredCapacity(InferenceMemoryPool pool)
        => InferenceMemoryPlanner.GetCapacity(pool).DesiredCapacity;

    private PhysicalMemoryCapacityPlan CapturePhysicalPlan(InferenceHardwareMemory hw)
    {
        var now = DateTimeOffset.UtcNow;
        var ledger = Budget.Snapshot();
        var ceilings = Budget.CapacityTargets().ToDictionary(p => p.Pool);
        long hostCeiling = Math.Min(_options.MaximumHostBytes, ceilings[HostPool].Ceiling);
        long deviceCeiling = Math.Min(_options.MaximumDeviceBytes, ceilings[DevicePool].Ceiling);
        // The scope is installed before any cache owner exists. These native
        // counters report actual cache payload on this CUDA rank, a lower bound
        // on its allocator occupancy. Callback charges can include reservations
        // above actual buffer sizes; graph kind alone also does not prove device
        // residency, so neither is sufficient evidence for adding bytes back.
        long deviceIncluded = GgmlBasicOps.TryGetCacheMemoryUsage(0, out var cache)
            ? checked(cache.DeviceCopyCommittedBytes + cache.PreloadCommittedBytes) : 0;
        var observations = new List<PhysicalMemoryObservation>
        {
            new("host-available", HostPool, "host-ram", PhysicalMemoryMetric.HostAvailable,
                hw.HostTotal, hw.HostAvailable, now)
            {
                MaximumBudgetBytes = hostCeiling,
                Headroom = new() { MinimumBytes = HostHeadroom(hw, _options, hostCeiling), FutureBurstBytes = _options.HostUninstrumentedBurstBytes }
            },
            new("cuda-allocator-available", DevicePool, "cuda0", PhysicalMemoryMetric.AllocatorAvailable,
                hw.DeviceTotal, hw.DeviceAvailable, now)
            {
                IncludedCommittedBytes = deviceIncluded, MaximumBudgetBytes = deviceCeiling,
                Headroom = new() { MinimumBytes = DeviceHeadroom(hw, _options, deviceCeiling), FutureBurstBytes = _options.DeviceUninstrumentedBurstBytes }
            }
        };
        AddHostCommitObservation(observations, hw, now, hostCeiling,
            HostHeadroom(hw, _options, hostCeiling), _options.HostUninstrumentedBurstBytes);
        return LastPhysicalCapacityPlan = _physicalPlanner.Plan(observations, ledger, now);
    }

    internal static void AddHostCommitObservation(List<PhysicalMemoryObservation> observations,
        InferenceHardwareMemory hw, DateTimeOffset now, long ceiling, long headroom, long burst)
    {
        hw.ValidateCommit();
        if (hw.HostCommitLimit is not long total || hw.HostCommitAvailable is not long available) return;
        observations.Add(new("host-commit-available", HostPool, "host-commit", PhysicalMemoryMetric.PrivateCommitAvailable,
            total, available, now)
        {
            MaximumBudgetBytes = ceiling,
            Headroom = new() { MinimumBytes = headroom, FutureBurstBytes = burst }
            // No allocator proves which host-ledger bytes are in process commit;
            // do not add existing owners back or manufacture enforcement evidence.
        });
    }

    internal static bool RequiresIdleTrim(InferenceMemoryPool pool, long hostReserve, long cachedBytes,
        long deviceReserve = 0, long deviceCachedBytes = 0)
    {
        decimal owners = (decimal)pool.Accounting.Reserved + pool.Accounting.Committed;
        // A cache can still fit its payload quota while preventing the next
        // request from obtaining its workspace or untracked forecast headroom.
        decimal holdout = pool.Pool == HostPool && cachedBytes > 0 ? hostReserve
            : pool.Pool == DevicePool && deviceCachedBytes > 0 ? deviceReserve : 0;
        return DesiredCapacity(pool) < owners + holdout;
    }

    private static long HostExecutionPeak(InferenceMemoryPlan plan)
        => ExecutionPeak(plan, HostPool);

    private static long ExecutionPeak(InferenceMemoryPlan plan, string pool)
    {
        var peak = plan.PoolPeaks.Single(p => p.Pool == pool);
        return Math.Max(peak.Prefill, peak.Decode);
    }

    internal static long HostCacheLimit(InferenceMemoryPlan plan, long sourceBytes, long ceiling)
        => CacheLimit(plan, HostPool, sourceBytes, ceiling);

    internal static long DeviceCacheLimit(InferenceMemoryPlan plan, long sourceBytes, long ceiling)
        => CacheLimit(plan, DevicePool, sourceBytes, ceiling);

    private static long CacheLimit(InferenceMemoryPlan plan, string pool, long sourceBytes, long ceiling)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(sourceBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(ceiling);
        if (plan.SelectedCandidate?.Placement != InferenceWeightPlacement.SsdStreaming) return 0;
        long capacity = plan.Capacities.Single(p => p.Pool == pool).AdditionalAvailable;
        long peak = ExecutionPeak(plan, pool);
        return Math.Min(Math.Min(sourceBytes, ceiling), Math.Max(0, capacity - peak));
    }

    public void Dispose()
    {
        if (_disposed) return;
        Model.Dispose();
        _hostBudget.Dispose();
        _nativeBudget.Dispose();
        _disposed = true;
    }

    /// <summary>Conservative request-owned execution peak, separate from shared
    /// weights and snapshot pages. Includes simultaneous old/new KV during
    /// growth and native graph scratch. It does not certify file-cache RSS or
    /// driver/GC overhead. Only the qualified serial text lane is supported.</summary>
    public InferenceMemoryBytes EstimateRequestPeak(int promptTokens, int maximumNewTokens, int prefillChunkTokens)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int context = checked(promptTokens + maximumNewTokens);
        if (promptTokens <= 0 || maximumNewTokens <= 0 || context > _options.ContextTokens
            || prefillChunkTokens <= 0 || prefillChunkTokens > Plan.SelectedChunkTokens)
            throw new ArgumentOutOfRangeException(nameof(promptTokens), "Request exceeds the admitted context or prefill shape.");
        return _profile.RequestPeak(context, Math.Min(prefillChunkTokens, promptTokens),
            Plan.SelectedCandidate!.Placement == InferenceWeightPlacement.SsdStreaming);
    }

    /// <summary>Combine execution and host-KV peaks on this ledger. Configure the
    /// engine for explicit non-speculative PerSequence execution, with no prefix
    /// caching or media. Each allocation consumes the admitted envelope; a
    /// underestimated shape fails closed instead of borrowing another request's
    /// credit. Retained buffers remain charged after the request completes.</summary>
    public RequestMemoryAdmission CreateRequestMemoryAdmission(KvSnapshotOptions snapshots, int blockTokens,
        int maximumRunningRequests, int prefillChunkTokens, int maximumQueuedRequests = 1024)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(snapshots);
        if (prefillChunkTokens <= 0 || prefillChunkTokens > Plan.SelectedChunkTokens)
            throw new ArgumentOutOfRangeException(nameof(prefillChunkTokens));
        if (!ReferenceEquals(snapshots.SharedBudget, Budget) || snapshots.RamPool != HostPool)
            throw new ArgumentException("Snapshots must share this session's host ledger.", nameof(snapshots));
        IReadOnlyList<MemoryCharge> ExecutionPeak(SequenceState seq)
        {
            if (seq.MediaSpans.Count != 0) throw new NotSupportedException("Adaptive request forecasting is qualified for text only.");
            var peak = EstimateRequestPeak(seq.PromptTokens.Count, seq.MaxNewTokens, prefillChunkTokens);
            return new MemoryCharge[] { new(HostPool, peak.Host), new(DevicePool, peak.Device) };
        }
        var admission = RequestMemoryAdmission.ForKvSnapshots(snapshots, Model.ComputeKVBlockByteSize(blockTokens),
            blockTokens, maximumRunningRequests, additionalPeak: ExecutionPeak, maxQueuedRequests: maximumQueuedRequests);
        // The bound shape rejects prefix caching. With exactly one running
        // request, queued requests cannot cause a state swap, so omit snapshot
        // page/spill reservations in that case. Engine-owned capture scratch
        // and staging remain charged. Multiple running requests include snapshots.
        Func<SequenceState, IReadOnlyList<MemoryCharge>> estimate = maximumRunningRequests == 1
            ? ExecutionPeak : admission.EstimatePeak;
        return new(Budget, estimate, maximumQueuedRequests)
        {
            ExecutionShape = new(snapshots, blockTokens, maximumRunningRequests, prefillChunkTokens),
            BeforeAdmission = _ =>
            {
                // The engine invokes this at a new-envelope boundary while its
                // compute gate is open and the model lock is held. Refresh once
                // here, never during submission or each token. Borrowed budgets
                // keep their caller's ceilings; the atomic reservation checks
                // these additional constraints against the latest live ledger.
                if (!RefreshCapacity())
                    throw new MemoryPressureException("Fresh physical observations refuse request admission.");
                return LastPhysicalCapacityPlan!.DesiredCapacities;
            },
            EnterSerialExecution = seq =>
            {
                ObjectDisposedException.ThrowIf(_disposed, this);
                var envelope = seq.MemoryEnvelope ?? throw new InvalidOperationException("Execution requires an admitted request envelope.");
                var native = _nativeBudget.EnterExecution(envelope);
                try { return new ExecutionScope(native, _hostBudget.EnterExecution(envelope)); }
                catch { native.Dispose(); throw; }
            }
        };
    }

    private sealed class ExecutionScope(IDisposable native, IDisposable host) : IDisposable
    {
        public void Dispose() { try { host.Dispose(); } finally { native.Dispose(); } }
    }

    internal static InferenceMemoryPlan PlanLoad(DenseMemoryProfile profile, AdaptiveModelMemoryOptions options,
        InferenceHardwareMemory hardware, MemoryBudget budget, bool sharedBudget = false)
    {
        Validate(options);
        var candidates = new List<InferenceExecutionCandidate>();
        int maximum = Math.Min(options.PrefillTokens, options.MaximumPrefillChunkTokens);
        for (int chunk = maximum; chunk > 0; chunk = chunk == 1 ? 0 : Math.Max(1, chunk / 2))
        {
            var workspace = profile.Workspace(chunk, options.ContextTokens);
            candidates.Add(new()
            {
                Name = $"resident-{chunk}", Placement = InferenceWeightPlacement.Resident,
                PrefillChunkTokens = chunk, PreservesExecutionGraph = true,
                // Residency in discrete VRAM does not require a second full
                // anonymous RAM copy. The loader retains demand-paged file views
                // for row lookups; these are OS cache, not owned host payload.
                // Retained F32/explicitly retained quantized copies remain host
                // owners; fusion is only the extra construction peak.
                KeepResidentHostWeights = false,
                Persistent = new(Host: profile.ResidentHostWeightBytes),
                LoadingWorkspace = new(Host: checked(profile.FusionBytes + (128L << 20))),
                PrefillWorkspace = workspace, DecodeWorkspace = profile.Workspace(1, options.ContextTokens)
            });
            if (options.AllowWeightStreaming)
                candidates.Add(new()
                {
                    Name = $"ssd-{chunk}", Placement = InferenceWeightPlacement.SsdStreaming,
                    PrefillChunkTokens = chunk, CapabilityRefusal = profile.StreamingRefusal,
                    // One tile is required. The executor may reserve a second
                    // tile from otherwise available host quota for read-ahead;
                    // tight quotas retain the single-buffer execution path.
                    TransferBuffer = new(Host: options.StreamingTileBytes + (4L << 20)), TransferBufferCount = 1,
                    LoadingWorkspace = new(Host: 128L << 20),
                    PrefillWorkspace = workspace with { Device = checked(workspace.Device + profile.LargestProjectionBytes) },
                    DecodeWorkspace = profile.Workspace(1, options.ContextTokens) with
                    { Device = checked(profile.Workspace(1, options.ContextTokens).Device + options.StreamingTileBytes * 2L) }
                });
        }
        return InferenceMemoryPlanner.Plan(new()
        {
            Pools = Pools(hardware, budget, options, sharedBudget), HostPools = [HostPool], DevicePools = [DevicePool],
            Model = profile.Model, Workload = new(options.ContextTokens, options.PrefillTokens, 1, 1), Candidates = candidates
        });
    }

    private static InferenceMemoryPool[] Pools(InferenceHardwareMemory hw, MemoryBudget budget, AdaptiveModelMemoryOptions o, bool sharedBudget = false)
    {
        var hostCapacity = hw.HostAllocationCapacity();
        var snapshots = budget.Snapshot().ToDictionary(p => p.Pool);
        long hostCeiling = sharedBudget ? Math.Min(o.MaximumHostBytes, snapshots[HostPool].Capacity) : o.MaximumHostBytes;
        long deviceCeiling = sharedBudget ? Math.Min(o.MaximumDeviceBytes, snapshots[DevicePool].Capacity) : o.MaximumDeviceBytes;
        // A small partition/quota on a large machine must not inherit headroom
        // proportional to the whole machine. Availability still independently
        // constrains the plan; explicit operator headroom always takes priority.
        return
        [
            new(HostPool, hostCapacity.Total, hostCapacity.Available, HostHeadroom(hw, o, hostCeiling),
                snapshots[HostPool], hostCeiling),
            new(DevicePool, hw.DeviceTotal, hw.DeviceAvailable, DeviceHeadroom(hw, o, deviceCeiling),
                snapshots[DevicePool], deviceCeiling)
        ];
    }

    private static long HostHeadroom(InferenceHardwareMemory hw, AdaptiveModelMemoryOptions o, long ceiling)
        => o.HostHeadroomBytes ?? Math.Max(512L << 20, Math.Min(hw.HostTotal, ceiling) / 16);
    private static long DeviceHeadroom(InferenceHardwareMemory hw, AdaptiveModelMemoryOptions o, long ceiling)
        => o.DeviceHeadroomBytes ?? GpuMemoryBudget.ResolveHeadroomBytes(Math.Min(hw.DeviceTotal, ceiling));

    private static void Validate(AdaptiveModelMemoryOptions o)
    {
        if (o.ContextTokens <= 0 || o.PrefillTokens <= 0 || o.PrefillTokens > o.ContextTokens
            || o.MaximumPrefillChunkTokens <= 0 || o.StreamingTileBytes <= 0
            || o.MaximumHostBytes < 0 || o.MaximumDeviceBytes < 0
            || o.MaximumStreamingHostCacheBytes < 0 || o.MaximumStreamingDeviceCacheBytes < 0 || o.MaximumStreamingWorkspaceCacheBytes < 0
            || o.HostUninstrumentedBurstBytes < 0 || o.DeviceUninstrumentedBurstBytes < 0
            || o.HostHeadroomBytes < 0 || o.DeviceHeadroomBytes < 0)
            throw new ArgumentOutOfRangeException(nameof(o));
    }
}

/// <summary>Construction could not release every native owner. Callbacks stay
/// rooted and credit remains charged. After resolving/releasing those owners,
/// retry UnreleasedBudgetScope.Dispose; never create a replacement ledger that
/// forgets the still-live allocation.</summary>
public sealed class AdaptiveModelAllocationException : InvalidOperationException
{
    internal AdaptiveModelAllocationException(GgmlCacheBudgetScope scope, MemoryBudget budget, Exception inner, HostAllocationBudgetScope? hostScope = null)
        : base("Adaptive model construction failed with live native budget owners; accounting remains attached.", inner)
    { UnreleasedBudgetScope = scope; Budget = budget; UnreleasedHostBudgetScope = hostScope; }
    public GgmlCacheBudgetScope UnreleasedBudgetScope { get; }
    public HostAllocationBudgetScope? UnreleasedHostBudgetScope { get; }
    public MemoryBudget Budget { get; }
}
