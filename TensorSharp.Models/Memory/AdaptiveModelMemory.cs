// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using TensorSharp.Memory;
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
    /// <summary>Read physical availability, not GC heap usage or an earlier budget.
    /// Unknown measurements fail closed. Initializing the CUDA backend is explicit.</summary>
    public static InferenceHardwareMemory CaptureCuda()
    {
        GgmlBasicOps.EnsureBackendAvailable(GgmlBackendType.Cuda);
        if (!GgmlBasicOps.TryGetDeviceMemoryInfo(out long free, out long total) || total <= 0)
            throw new NotSupportedException("The backend did not report device memory availability.");
        var host = CaptureHost();
        return new(host.Total, host.Available, total, free);
    }

    internal static (long Total, long Available) CaptureHost()
    {
        if (OperatingSystem.IsWindows())
        {
            var status = new MemoryStatus { Length = (uint)Marshal.SizeOf<MemoryStatus>() };
            if (!GlobalMemoryStatusEx(ref status))
                throw new IOException("GlobalMemoryStatusEx could not read physical memory.");
            return (checked((long)status.TotalPhysical), checked((long)status.AvailablePhysical));
        }
        if (OperatingSystem.IsLinux())
            return HostMemoryAvailability.CaptureLinux();
        throw new PlatformNotSupportedException("An available-physical-memory provider is required for this host.");
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct MemoryStatus
    {
        public uint Length, Load;
        public ulong TotalPhysical, AvailablePhysical, TotalPageFile, AvailablePageFile,
            TotalVirtual, AvailableVirtual, AvailableExtendedVirtual;
    }
    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static extern bool GlobalMemoryStatusEx(ref MemoryStatus status);
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
    public Exception? AccountingError => _nativeBudget.CallbackError;
    public (long Bytes, long PeakBytes, int Allocations) HostAllocationUsage => _hostBudget.Usage;

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
        using var gguf = new GgufFile(path);
        var profile = DenseMemoryProfile.Read(gguf, options, ModelBase.RetainsAllHostQuantizedWeights);
        var hardware = InferenceHardwareMemory.CaptureCuda();
        var budget = sharedBudget ?? new MemoryBudget([new(HostPool, 0), new(DevicePool, 0)]);
        var plan = PlanLoad(profile, options, hardware, budget, sharedBudget != null);
        if (!plan.Accepted)
            throw new MemoryPressureException(string.Join(Environment.NewLine, plan.Rejections.Select(r => r.Reason).Distinct()));
        if (sharedBudget == null && !budget.TrySetCapacities(plan.Capacities.Select(c => new MemoryCharge(c.Pool, c.ProtectedCapacity))))
            throw new MemoryPressureException("Memory owners changed during load admission.");

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
            var policy = new ModelMemoryPolicy(options.ContextTokens, plan.SelectedChunkTokens);
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

    /// <summary>Refresh at a quiescent request boundary. Shrinking does not revoke
    /// live owners or discard KV. False means stop new admission; release work or
    /// reload a smaller supported placement. Do not vary graph shapes per token.</summary>
    public bool RefreshCapacity()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var hw = InferenceHardwareMemory.CaptureCuda();
        long cachedBytes = Model.StreamingWeightUsage?.HostCacheBytes ?? 0;
        var streaming = Model.StreamingWeightUsage;
        long deviceCachedBytes = checked((streaming?.DeviceCacheBytes ?? 0) + (streaming?.DeviceWorkspaceCacheBytes ?? 0));
        if (Pools(hw, Budget, _options, !_ownsBudget).Any(pool => RequiresIdleTrim(pool, _hostCacheReserveBytes, cachedBytes,
                _deviceCacheReserveBytes, deviceCachedBytes)))
        {
            Model.TrimIdleMemory();
            hw = InferenceHardwareMemory.CaptureCuda();
        }
        bool admitted = true;
        var updates = new List<MemoryCharge>();
        foreach (var pool in Pools(hw, Budget, _options, !_ownsBudget))
        {
            var owners = checked(pool.Accounting.Reserved + pool.Accounting.Committed);
            long desired = DesiredCapacity(pool);
            admitted &= desired >= owners;
            updates.Add(new(pool.Pool, Math.Max(owners, desired)));
        }
        if (_ownsBudget) admitted &= Budget.TrySetCapacities(updates);
        return admitted && AccountingError == null;
    }

    private static long DesiredCapacity(InferenceMemoryPool pool)
    {
        var physical = Math.Min((decimal)pool.TotalBytes, (decimal)pool.Accounting.Committed + pool.AvailableBytes);
        return (long)Math.Max(0, Math.Min(pool.MaximumBudgetBytes, physical - pool.HeadroomBytes));
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
        // The bound shape rejects prefix caching and more than one running
        // request. Queued requests cannot cause a state swap. Engine-owned
        // capture scratch/staging remain charged, but page/spill reservations
        // would only strand credit for a path this engine cannot execute.
        Func<SequenceState, IReadOnlyList<MemoryCharge>> estimate = maximumRunningRequests == 1
            ? ExecutionPeak : admission.EstimatePeak;
        return new(Budget, estimate, maximumQueuedRequests)
        {
            ExecutionShape = new(snapshots, blockTokens, maximumRunningRequests, prefillChunkTokens),
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
        var snapshots = budget.Snapshot().ToDictionary(p => p.Pool);
        return
        [
            new(HostPool, hw.HostTotal, hw.HostAvailable, o.HostHeadroomBytes ?? Math.Max(512L << 20, hw.HostTotal / 16),
                snapshots[HostPool], sharedBudget ? Math.Min(o.MaximumHostBytes, snapshots[HostPool].Capacity) : o.MaximumHostBytes),
            new(DevicePool, hw.DeviceTotal, hw.DeviceAvailable, o.DeviceHeadroomBytes ?? GpuMemoryBudget.ResolveHeadroomBytes(hw.DeviceTotal),
                snapshots[DevicePool], sharedBudget ? Math.Min(o.MaximumDeviceBytes, snapshots[DevicePool].Capacity) : o.MaximumDeviceBytes)
        ];
    }

    private static void Validate(AdaptiveModelMemoryOptions o)
    {
        if (o.ContextTokens <= 0 || o.PrefillTokens <= 0 || o.PrefillTokens > o.ContextTokens
            || o.MaximumPrefillChunkTokens <= 0 || o.StreamingTileBytes <= 0
            || o.MaximumHostBytes < 0 || o.MaximumDeviceBytes < 0
            || o.MaximumStreamingHostCacheBytes < 0 || o.MaximumStreamingDeviceCacheBytes < 0 || o.MaximumStreamingWorkspaceCacheBytes < 0
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
