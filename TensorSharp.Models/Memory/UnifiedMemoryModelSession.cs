// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Memory.Planning;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Models;

/// <summary>Operator ceilings. Automatic plans additionally preserve physical
/// RAM/commit/VRAM headroom; these values are not OS or driver hard limits.</summary>
public sealed record UnifiedMemoryModelOptions
{
    public int ContextTokens { get; init; }
    public int PrefillChunkTokens { get; init; } = 512;
    // Small prefill uses selected-expert reuse with admitted microgroups;
    // longer tails use compact CUDA batches. Joint decode fuses all live lanes.
    public int ExpertCachePrefillRows { get; init; } = 32;
    public int MaximumRunningRequests { get; init; } = 3;
    // Each startup thinking mode has two distinct reusable boundaries.
    public int PrefixCheckpointCount { get; init; } = 4;
    public int RetainedConversationCount { get; init; } = 2;
    public long MaximumHostBytes { get; init; } = long.MaxValue;
    public long MaximumDeviceBytes { get; init; } = long.MaxValue;
    public long MaximumExpertCacheBytes { get; init; } = long.MaxValue;
    public bool EnableAsyncExpertPrefill { get; init; } = false;
    public bool EnableExpertCacheCapacityGrowth { get; init; } = false;
    /// <summary>Experimental GLM5Next selected-expert CUDA tier. Requires its
    /// matching native planner and remains off until real-model qualification.</summary>
    public bool EnableGlmSelectedExperts { get; init; } = false;
    /// <summary>Admit GLM prefill widths from each actual prompt instead of
    /// reserving the largest width at the logical context ceiling. Requires
    /// selected-expert planning and remains an explicit Host/session policy.</summary>
    public bool EnableAdaptiveGlmPrefill { get; init; } = false;
    /// <summary>Optional upper bound for adaptive GLM prefill only. Null inherits
    /// the common prefill bound; ordinary callers and Qwen retain their existing
    /// options. Actual GLM requests still admit a width from physical capacity.</summary>
    public int? AdaptiveGlmPrefillChunkTokens { get; init; }
    internal int ResolveGlmPrefillChunkTokens()
    {
        if (PrefillChunkTokens is < 1 or > 512)
            throw new ArgumentOutOfRangeException(nameof(PrefillChunkTokens));
        int rows = EnableAdaptiveGlmPrefill ? AdaptiveGlmPrefillChunkTokens ?? PrefillChunkTokens : PrefillChunkTokens;
        if (rows is < 1 or > 1024) throw new ArgumentOutOfRangeException(nameof(AdaptiveGlmPrefillChunkTokens));
        return rows;
    }
}

/// <summary>One Qwen4Exp GGML CUDA model, including its optional image tower.
/// Owns the actual allocation scopes and expert-cache policy for the complete
/// load/serve/unload lifetime. Instrumented allocations are quota enforced;
/// mapped pages, GC and vendor/driver memory remain observed with headroom.
/// Admitted requests retain independent complete hybrid holders. Eligible
/// decode cohorts share one separately owned fused graph; excess callers queue.</summary>
public sealed partial class UnifiedMemoryModelSession : IUnifiedMemoryModelSession
{
    public const string HostPool = "unified/ram";
    public const string DevicePool = "unified/cuda0";
    private readonly UnifiedMemoryModelOptions _options;
    private readonly Qwen4ExpUnifiedGeometry _geometry;
    private readonly Qwen35VisionMemoryGeometry? _vision;
    private readonly long _hostHeadroom, _deviceHeadroom;
    private readonly long _hostCeiling, _deviceCeiling;
    private readonly int _contextTokens, _prefillTokens;
    private GgmlQwen4ExpArenaScope? _jointArena;
    private BudgetReservation? _jointEnvelope;
    private InferenceMemoryBytes _jointTarget;
    private int _jointContextHighWater;
    private int _checkpointCopies;
    private GgmlCacheBudgetScope? _native;
    private HostAllocationBudgetScope? _host;
    private GgmlHostMoeExpertCacheScope? _experts;
    private GgmlHostMoeFilePipelineScope? _filePipeline;
    private BudgetReservation? _expertEnvelope;
    private Qwen4ExpModel? _model;
    private readonly Dictionary<string, SequenceState> _requestOwners = new(StringComparer.Ordinal);
    private readonly Dictionary<string, (int Capacity, int Prefill, bool IsPrefill)> _requestShapes = new(StringComparer.Ordinal);
    private readonly Dictionary<SequenceState, int> _requestPrefillCaps = new();
    private bool _disposed;

    private UnifiedMemoryModelSession(UnifiedMemoryModelOptions options, Qwen4ExpUnifiedGeometry geometry,
        Qwen35VisionMemoryGeometry? vision, InferenceHardwareMemory hardware, int contextTokens)
    {
        _options = options; _geometry = geometry; _vision = vision; _contextTokens = contextTokens;
        _prefillTokens = Math.Min(options.PrefillChunkTokens, contextTokens);
        var hostCapacity = hardware.HostAllocationCapacity();
        _hostHeadroom = Math.Max(512L << 20, Math.Min(hostCapacity.Total, options.MaximumHostBytes) / 16);
        _deviceHeadroom = Math.Max(768L << 20, Math.Min(hardware.DeviceTotal, options.MaximumDeviceBytes) / 16);
        _hostCeiling = Math.Min(options.MaximumHostBytes, Math.Max(0, hostCapacity.Available - _hostHeadroom));
        _deviceCeiling = Math.Min(options.MaximumDeviceBytes, Math.Max(0, hardware.DeviceAvailable - _deviceHeadroom));
        if (_hostCeiling <= 0 || _deviceCeiling <= 0)
            throw new MemoryPressureException("No usable RAM/commit or CUDA capacity remains after unified-memory headroom.");
        Budget = new MemoryBudget([new(HostPool, _hostCeiling), new(DevicePool, _deviceCeiling)]);
    }

    /// <summary>Null only on a failed construction retained for cleanup retry.</summary>
    public ModelBase? Model => _model;
    public MemoryBudget Budget { get; }
    public int ContextTokens => _contextTokens;
    public int PrefillChunkTokens => _prefillTokens;
    public string FeatureReport =>
        $"[unified-memory] enabled: qwen4exp, one GGML CUDA device; context={_contextTokens}, prefill={_prefillTokens}, active-request-limit={_options.MaximumRunningRequests}\n" +
        $"[unified-memory] allocation ceilings: RAM={_hostCeiling / 1048576.0:F0} MiB, CUDA={_deviceCeiling / 1048576.0:F0} MiB; " +
        $"expert-cache={(_experts?.BudgetBytes ?? 0) / 1048576.0:F0} MiB, expert-prefill-rows={_experts?.MaxPrefillRows ?? 0}\n" +
        $"[unified-memory] on: shared native/host payload accounting, hardware/commit headroom, request admission, " +
        $"bounded expert reuse and pressure trimming, file staging for CUDA/CPU expert execution, fused model kernels{(_vision != null ? ", image preparation admission" : "")}\n" +
        "[unified-memory] concurrency: continuous batching with admitted hybrid holders and a separately charged joint fused-decode arena; bounded in-memory exact-state prefix checkpoints when enabled\n" +
        "[unified-memory] pressure: inactive retained complete-holder GPU state can demote to its charged host mirrors; request prefill width is admitted after actual prefix materialization, with one bounded device-cache tradeoff for long prefills\n" +
        (_filePipeline?.Enabled == true
            ? "[unified-memory] async expert file prefill: eligible one-device CUDA graphs, at least 64 rows and 32 MiB payload; two charged pinned slots bounded to 16 MiB\n"
            : "[unified-memory] async expert file prefill: off (parallel reads passed model correctness and improved isolated prefill; server/shared-budget and cross-platform qualification are pending)\n") +
        "[unified-memory] off: live block-KV/SSD paging (no admitted whole-state preemption/recovery adapter); " +
        "MTP/TP (not qualified with this request envelope); video/audio (no admitted preparation adapter)\n" +
        "[unified-memory] limits: mapped/page-cache residency, GC, vendor/driver pools are observed with margins, " +
        "not fully attributed or OS-enforced RAM/VRAM hard caps.";

    public static IUnifiedMemoryModelSession Create(string path, string? mmprojPath, BackendType backend)
        => Create(path, mmprojPath, backend, new UnifiedMemoryModelOptions());

    public static IUnifiedMemoryModelSession Create(string path, string? mmprojPath, BackendType backend,
        UnifiedMemoryModelOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        ValidateOptions(options);
        if (backend != BackendType.GgmlCuda)
            throw new NotSupportedException("--unified-memory requires a qualified architecture on ggml_cuda.");
        if (ParallelDegree("TENSORSHARP_TP_DEGREE") != 1 || ParallelDegree("TENSORSHARP_LAYER_SPLIT_DEGREE") != 1
            || !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TENSORSHARP_TP_NODE_ID"))
            || !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TENSORSHARP_TP_PEERS")))
            throw new NotSupportedException("--unified-memory has no qualified tensor/layer-parallel adapter.");
        if (SpeculationOptions.FromEnvironment().Enabled
            || !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("TS_SPEC_DRAFT_MODEL")))
            throw new NotSupportedException("--unified-memory currently requires speculative execution and draft loading to be disabled.");
        if (MoeCpuOffloadConfig.IsExplicitlySet)
            throw new NotSupportedException("--unified-memory selects expert placement from its shared budget; remove explicit CPU MoE placement overrides.");
        string? fileReadMode = Environment.GetEnvironmentVariable("TS_HOST_MOE_FILE_READ");
        if (fileReadMode != null && fileReadMode != "1")
            throw new NotSupportedException("--unified-memory requires bounded expert file staging, including its CPU fallback; remove TS_HOST_MOE_FILE_READ or set it to 1.");
        GgmlBasicOps.RequireSharedExpertCacheBudgetKind();
        Qwen4ExpUnifiedGeometry geometry;
        using (var file = GgufFile.OpenForMemoryPlanning(path))
        {
            string? architecture = file.GetString("general.architecture");
            if (architecture is "glm5next" or "glm5-next")
                return GlmUnifiedMemoryModelSession.Create(path, mmprojPath, backend, options);
            if (architecture != "qwen4exp")
                throw new NotSupportedException($"--unified-memory has no execution adapter for architecture '{architecture}'.");
            geometry = Qwen4ExpUnifiedGeometry.Read(file);
        }
        Qwen35VisionMemoryGeometry? vision = string.IsNullOrWhiteSpace(mmprojPath) || mmprojPath == "none"
            ? null : Qwen35VisionMemoryGeometry.Read(mmprojPath);
        int context = options.ContextTokens;
        if (context == 0)
        {
            string? configured = Environment.GetEnvironmentVariable("MAX_CONTEXT");
            if (!string.IsNullOrWhiteSpace(configured))
            {
                if (!int.TryParse(configured, NumberStyles.None, CultureInfo.InvariantCulture, out context) || context <= 0)
                    throw new ArgumentException("MAX_CONTEXT must be a positive token count.");
            }
            else context = Math.Min(32768, geometry.DeclaredContextTokens);
        }
        if (context <= 0 || context > geometry.DeclaredContextTokens)
            throw new ArgumentOutOfRangeException(nameof(options), "Unified context exceeds the model's declared context.");
        var session = new UnifiedMemoryModelSession(options, geometry, vision, InferenceHardwareMemory.CaptureCuda(), context);
        try
        {
            var warmShape = geometry.RequestPeak(Math.Min(context, 8192), session._prefillTokens);
            var decodeShape = geometry.RequestPeak(Math.Min(context, 8192), 1);
            long denseDevice = checked(geometry.LoadingDeviceBoundBytes + (vision?.DeviceWeightBytes ?? 0));
            long loadHost = checked(geometry.ResidentHostPayloadBytes + geometry.LoadingHostTemporaryBytes
                + (vision?.ResidentHostBytes ?? 0) + (vision?.LoadingTemporaryBytes ?? 0) + (128L << 20));
            if (loadHost > session._hostCeiling || denseDevice + warmShape.Device > session._deviceCeiling)
                throw new MemoryPressureException("Unified load forecast cannot fit dense/vision weights and execution workspace within available RAM/commit/VRAM.");
            // This is an optional maximum, not a resident allocation. Reserve
            // wide-prefill workspace when that request is admitted; otherwise
            // the largest supported prefill would permanently suppress reuse
            // even during a smaller request. PrepareCapacity trims before work.
            long cache = SelectExpertCacheBytes(session._deviceCeiling, denseDevice, decodeShape.Device,
                geometry.ExpertSourceBytes, geometry.AllLayerExpertCacheMinimumBytes, options.MaximumExpertCacheBytes);
            Console.WriteLine($"[unified-memory] load forecast: dense+vision CUDA={denseDevice / 1048576.0:F0} MiB, " +
                $"prefill forecast={warmShape.Device / 1048576.0:F0} MiB, decode reserve={decodeShape.Device / 1048576.0:F0} MiB, " +
                $"host load={loadHost / 1048576.0:F0} MiB; " +
                $"automatic expert-cache ceiling={cache / 1048576.0:F0} MiB");
            session._native = new(session.Budget, [[DevicePool]], includeGraphBuffers: true, hostPools: [HostPool]);
            session._host = new(session.Budget, [HostPool]);
            session._filePipeline = new(options.EnableAsyncExpertPrefill);
            session._experts = new(cache, Math.Min(options.ExpertCachePrefillRows, session._prefillTokens), geometry.Layers,
                options.MaximumRunningRequests, options.EnableExpertCacheCapacityGrowth);
            session._expertEnvelope = session.Budget.Reserve([new(DevicePool, cache)]);
            session._native.SetSharedExpertEnvelope(session._expertEnvelope);
            session._jointArena = new(1);
            session._jointEnvelope = session.Budget.Reserve([new(HostPool, 0), new(DevicePool, 0)]);
            session._native.SetSharedJointEnvelope(session._jointEnvelope);
            var policy = new ModelMemoryPolicy(context, session._prefillTokens)
            {
                OmitEmbeddedDraftWeights = true, ExpertCacheBytes = cache,
                ExpertCachePrefillRows = session._experts.MaxPrefillRows,
                DeviceExecutionReserveBytes = checked(warmShape.Device + (vision?.DeviceWeightBytes ?? 0))
            };
            session._model = (Qwen4ExpModel)ModelBase.Create(path, backend, 1, null!, null!, 1, null!, policy);
            if (vision != null) session._model.LoadVisionEncoder(mmprojPath!);
            session._model.ConfigureUnifiedVisionAdmission(session.EnterVisionPreparation);
            Console.WriteLine(session.FeatureReport);
            return session;
        }
        catch (Exception creation)
        {
            if (creation is Qwen4ExpConstructionCleanupException partial) session._model = partial.UnreleasedModel;
            try { session.Dispose(); }
            catch (Exception cleanup) { throw new UnifiedMemoryModelLoadException(session, new AggregateException(creation, cleanup)); }
            throw;
        }
    }

    internal static long SelectExpertCacheBytes(long available, long dense, long execution, long source,
        long minimum, long ceiling)
    {
        if (new[] { available, dense, execution, source, minimum, ceiling }.Any(x => x < 0))
            throw new ArgumentOutOfRangeException(nameof(available));
        long spare = (long)Math.Max(0, (decimal)available - dense - execution);
        long result = Math.Min(ceiling, Math.Min(source, spare)) / (1L << 20) * (1L << 20);
        return minimum > 0 && result >= minimum ? result : 0;
    }

    public SchedulerConfig ConfigureScheduler(SchedulerConfig requested)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        ArgumentNullException.ThrowIfNull(requested);
        if (requested.Speculation.Enabled || requested.KvSnapshots != null || requested.MemoryAdmission != null)
            throw new NotSupportedException("Unified Qwen execution cannot combine with independent KV snapshots, speculation or another admission adapter.");
        int block = requested.BlockSize > 0 ? requested.BlockSize : SchedulerConfig.DefaultBlockSize;
        _checkpointCopies = requested.EnablePrefixCaching
            ? _options.PrefixCheckpointCount + _options.RetainedConversationCount + 1 : 0;
        return new SchedulerConfig
        {
            BlockSize = block, NumBlocks = Math.Max(1, checked(((_contextTokens + block - 1) / block)
                * (_options.MaximumRunningRequests + (requested.EnablePrefixCaching
                    ? _options.PrefixCheckpointCount + _options.RetainedConversationCount : 0)))),
            MaxNumRunningSequences = _options.MaximumRunningRequests, MaxNumBatchedTokens = _prefillTokens,
            MaxPrefillChunkSize = _prefillTokens, SoloPrefillChunkSize = _prefillTokens,
            PrefillChunkTokenLimit = _prefillTokens, EnablePrefixCaching = requested.EnablePrefixCaching,
            StopRepetition = requested.StopRepetition, DecodeQuantumTokens = requested.DecodeQuantumTokens,
            Speculation = SpeculationOptions.Disabled,
            ExecutionOptionsOverride = new ExecutionOptions
            {
                BatchedPathDisabled = true, PerSeqFusedEnabled = true, BatchedFusedDecodeEnabled = true,
                RetainedFusedCacheBudget = requested.EnablePrefixCaching ? _options.RetainedConversationCount : 0,
                PrefixCheckpointBudget = requested.EnablePrefixCaching ? _options.PrefixCheckpointCount : 0, KvHolderPoolMax = 0
            },
            MemoryAdmission = new RequestMemoryAdmission(Budget, EstimateRequest)
            {
                EstimateMaterializationPeak = EstimateMaterialization,
                ReestimateAfterPhysicalCheck = true,
                UsePerSequenceHolders = true,
                BeforeAdmissionWithRunning = PrepareRequestCapacity,
                AfterPrefixMaterialization = CompletePrefixMaterialization,
                PrefillTokenLimit = AdmittedPrefillLimit,
                RevalidateExecution = RevalidateSequenceAllocations,
                AfterPrefill = CompletePrefill,
                AfterSequenceMemoryRelease = RecoverReleasedSequenceCapacity,
                EnterSerialExecution = EnterSequenceAllocations,
                EnterBatchedExecution = sequences => EnterAllocations(_jointEnvelope
                    ?? throw new InvalidOperationException("Unified batch execution requires its shared arena envelope."))
            }
        };
    }

    private int AdmittedPrefillLimit(SequenceState sequence)
        => _requestPrefillCaps != null && _requestPrefillCaps.TryGetValue(sequence, out int rows) ? rows : _prefillTokens;

    private InferenceMemoryBytes RequestPeak(SequenceState seq) => RequestPeakForRows(seq, AdmittedPrefillLimit(seq));

    private int ForecastPrefillRows(SequenceState sequence, int maximum)
    {
        if (sequence.NumComputedTokens >= sequence.PromptTokens.Count) return 1;
        // The last, smaller Forward can reuse an outstanding/pooled wide input
        // and allocate a different scratch bucket at the same time. Keep the
        // width already granted to this request until CompletePrefill releases
        // its transient state. Before selection, each candidate is still priced
        // independently; a lower candidate must not inherit the global ceiling.
        if (_requestPrefillCaps != null && _requestPrefillCaps.TryGetValue(sequence, out int admitted))
            return admitted;
        return Math.Min(maximum, Math.Max(1, sequence.PromptTokens.Count - sequence.NumComputedTokens));
    }

    private InferenceMemoryBytes RequestPeakForRows(SequenceState seq, int prefillRows)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        int rows = ForecastPrefillRows(seq, prefillRows);
        var peak = _model!.GetUnifiedMemoryRequestPeak(RequiredContext(seq), rows,
            // Copies can be published at exact prefill boundaries, including
            // the prompt end. After that publication completed, their fixed
            // tensors remain charged children; decode growth must not pretend
            // every immutable checkpoint grows with the active KV holder.
            checkpointCopies: seq.NumComputedTokens < seq.PromptTokens.Count ? _checkpointCopies : 0,
            independentHolder: true, requestId: seq.RequestId);
        if (seq.NumComputedTokens >= seq.PromptTokens.Count && seq.MemoryEnvelope != null)
        {
            // The root can own previously published copies, pooled blocks or
            // host graph payloads in addition to this holder. A child floor
            // alone protects those bytes but leaves no credit for new KV or
            // scratch when it exceeds the live execution peak. Preserve every
            // actual host child, then reserve the complete live growth peak.
            // This conservatively counts an existing live seed twice until
            // exact per-payload attribution is available; it never projects
            // immutable checkpoints at the new decode capacity.
            long children = seq.MemoryEnvelope.LiveChildCharges.Single(c => c.Pool == HostPool).Bytes;
            peak = new(checked(peak.Host + children), peak.Device);
        }
        // Ring slots are physical pinned host payload, independent of the CUDA
        // graph and CPU wide-prefill arenas. Forecast the complete admitted
        // ring even when source/threshold checks later decline the async path.
        return _filePipeline?.Enabled == true && rows >= _filePipeline.MinimumRows
            ? new(checked(peak.Host + _filePipeline.MaximumRingBytes), peak.Device) : peak;
    }

    // Output allowance is a termination limit, not the current physical shape.
    // Reserve each real holder capacity and grow its quota before allocating.
    private static int RequiredContext(SequenceState sequence)
        => Math.Max(sequence.PromptTokens.Count, checked(sequence.NumComputedTokens + 1));

    private (int Capacity, int Prefill, bool IsPrefill) RequestShape(SequenceState sequence)
        => (_model!.GetUnifiedMemoryCapacity(RequiredContext(sequence), requestId: sequence.RequestId),
            ForecastPrefillRows(sequence, AdmittedPrefillLimit(sequence)),
            sequence.NumComputedTokens < sequence.PromptTokens.Count);

    private IReadOnlyList<MemoryCharge> EstimateRequest(SequenceState seq) => IncrementalPeak(RequestPeak(seq));

    private InferenceMemoryBytes MaterializationPeak(SequenceState sequence)
        => _model!.GetUnifiedMemoryRequestPeak(RequiredContext(sequence), 1,
            checkpointCopies: _checkpointCopies, independentHolder: true, requestId: sequence.RequestId);

    private IReadOnlyList<MemoryCharge> EstimateMaterialization(SequenceState sequence)
        => IncrementalPeak(MaterializationPeak(sequence));

    private IReadOnlyList<MemoryCharge> PrepareRequestCapacity(SequenceState sequence, IReadOnlyList<SequenceState> running)
    {
        PruneFinishedRequestOwners();
        var request = MaterializationPeak(sequence);
        if (running.Count == 0)
        {
            ReleaseIdleJointCredit();
            return PrepareCapacity(request);
        }
        int tokens = _model!.GetUnifiedMemoryCapacity(RequiredContext(sequence), requestId: sequence.RequestId);
        foreach (var live in running)
            tokens = Math.Max(tokens, _model.GetUnifiedMemoryCapacity(RequiredContext(live), requestId: live.RequestId));
        // Native keeps its context high water mark even when a replacement
        // entry has more lanes. Maxing byte totals alone loses this shape axis.
        tokens = Math.Max(tokens, _jointContextHighWater);
        var joint = _geometry.JointDecodePeak(tokens, running.Count + 1);
        // Arena capacity has a high water mark until an explicit idle trim.
        // Reserve that shared growth before the second request is admitted.
        joint = new(Math.Max(joint.Host, _jointTarget.Host), Math.Max(joint.Device, _jointTarget.Device));
        var total = new InferenceMemoryBytes(checked(request.Host + joint.Host - _jointTarget.Host),
            checked(request.Device + joint.Device - _jointTarget.Device));
        var limits = PrepareCapacity(total, recoverExpertCache: false);
        if (!_jointEnvelope!.TryResizeTotal([new(HostPool, joint.Host), new(DevicePool, joint.Device)], limits))
            throw new MemoryPressureException("The shared fused-decode arena cannot reserve its state replicas and workspace before admission.");
        _jointTarget = joint;
        _jointContextHighWater = tokens;
        // The shared reservation is now part of the ledger. The caller still
        // atomically acquires this request's separate envelope with these limits.
        return limits;
    }

    internal void ReleaseIdleJointCredit()
    {
        // A cancelled second admission may never have executed a joint graph.
        // Release its unused forecast at the next quiescent admission boundary,
        // while existing arena payloads keep their own child reservations.
        var envelope = _jointEnvelope
            ?? throw new InvalidOperationException("Unified joint ownership is not available.");
        if (!envelope.TryResizeTotal([new(HostPool, 0), new(DevicePool, 0)]))
            throw new InvalidOperationException("Unused joint reservation could not be released.");
        var owned = envelope.OwnedCharges;
        _jointTarget = new(owned.Single(c => c.Pool == HostPool).Bytes,
            owned.Single(c => c.Pool == DevicePool).Bytes);
        // Native retains its context capacity even when no request is running.
        // Keep _jointContextHighWater so the next pair reserves that full shape.
    }

    private void CompletePrefixMaterialization(SequenceState sequence)
    {
        var envelope = sequence.MemoryEnvelope
            ?? throw new InvalidOperationException("Prefix materialization ended without its admitted owner.");
        // The initial phase owned complete holder/copy/export state, without
        // borrowing credit from a presumed prefix hit. Only the actual published
        // computed-token count now chooses the remaining prefill shape. Admit
        // its net growth before Forward, preserving every live child charge.
        PruneFinishedRequestOwners();
        var (joint, tokens) = ExecutionJointTarget(sequence);
        if (_native!.CallbackError is { } error)
            throw new InvalidOperationException("Unified allocation accounting failed.", error);
        // Allocate bookkeeping before the successful ledger transaction; publishing
        // its selected width must not fail afterwards with an allocation error.
        _requestOwners.EnsureCapacity(_requestOwners.Count + 1);
        _requestShapes.EnsureCapacity(_requestShapes.Count + 1);
        _requestPrefillCaps.EnsureCapacity(_requestPrefillCaps.Count + 1);
        bool alreadyAdmitted = _requestPrefillCaps.ContainsKey(sequence);
        bool preferredReclaimAttempted = false;
        int maximum = Math.Min(AdmittedPrefillLimit(sequence), Math.Max(1, sequence.PromptTokens.Count - sequence.NumComputedTokens));
        bool TrySelect(out UnifiedMemoryPrefillPlan plan)
        {
            // One quiescent native observation per bounded pass, not per candidate.
            long boundTrunk = BoundTrunkWeights();
            IReadOnlyList<MemoryCharge> Estimate(int rows) => IncrementalPeak(RequestPeakForRows(sequence, rows), boundTrunk);
            if (alreadyAdmitted)
                return UnifiedMemoryPrefillAdmission.TrySelect(Budget, envelope, _jointEnvelope!, joint, maximum,
                    Estimate, PhysicalConstraints(), out plan, allowDowngrade: false);
            return UnifiedMemoryPrefillAdmission.TrySelectPerformance(Budget, envelope, _jointEnvelope!, joint, maximum,
                Estimate, PhysicalConstraints, _experts!.BudgetBytes, _geometry.AllLayerExpertCacheMinimumBytes,
                (reclaim, limits) =>
                {
                    long previous = _experts.BudgetBytes;
                    ResizeExpertCache(reclaim.TargetCacheBytes, limits);
                    Console.WriteLine($"[unified-memory] request {sequence.RequestId}: preferred prefill {reclaim.PreferredRows} rows, " +
                        $"CUDA shortfall={reclaim.DeviceShortfallBytes / 1048576.0:F0} MiB; " +
                        $"expert cache {previous / 1048576.0:F0} -> {_experts.BudgetBytes / 1048576.0:F0} MiB (one admission-boundary reclaim)");
                }, ref preferredReclaimAttempted, out plan);
        }
        if (!TrySelect(out var selected))
        {
            if (preferredReclaimAttempted)
                throw new MemoryPressureException("No admitted prefill width fits after one bounded device-cache reclaim.");
            // Pure candidates never trim a cache. Reclamation is a separate,
            // bounded phase for the smallest actual execution shape, followed by
            // a fresh complete pass; it cannot manufacture live-owner refunds.
            long previousCache = _experts!.BudgetBytes;
            PrepareCapacity(() =>
            {
                var additional = ExecutionEnvelopeGrowth(envelope,
                    IncrementalPeak(RequestPeakForRows(sequence, alreadyAdmitted ? maximum : 1)), _jointEnvelope!, joint);
                return [new(HostPool, additional.Host), new(DevicePool, additional.Device)];
            }, recoverExpertCache: false);
            // The minimum-shape maintenance may itself retire optional cache.
            // Do not add a second preference-driven trim on its retry.
            preferredReclaimAttempted = _experts.BudgetBytes != previousCache;
            if (!TrySelect(out selected))
                throw new MemoryPressureException("No admitted prefill width fits the current shared physical constraints.");
        }
        _jointTarget = joint;
        _jointContextHighWater = tokens;
        if (!alreadyAdmitted) _requestPrefillCaps[sequence] = selected.Rows;
        _requestOwners[sequence.RequestId] = sequence;
        _requestShapes[sequence.RequestId] = RequestShape(sequence);
        if (!alreadyAdmitted && selected.Rows < maximum)
            Console.WriteLine($"[unified-memory] request {sequence.RequestId}: prefill width {maximum} -> {selected.Rows} rows within shared physical constraints.");
        RecoverExpertCache(PhysicalConstraints(), additionalDeviceBytes: 0, "prefix-materialized");
    }

    private void CompletePrefill(SequenceState sequence)
    {
        var envelope = sequence.MemoryEnvelope ?? throw new InvalidOperationException("Prefill ended without its admitted owner.");
        _model!.ReleaseUnifiedPrefillGraphs(sequence.RequestId);
        // Return transient native file staging before shrinking its unused
        // reservation. Cache weights and all live sequence state remain owned.
        _experts!.Trim(_experts.CachedBytes);
        if (!envelope.TryResizeTotal(IncrementalPeak(RequestPeak(sequence))))
            throw new MemoryPressureException("The completed prefill could not transition to its bounded decode peak.");
        _requestShapes[sequence.RequestId] = RequestShape(sequence);
        // All live requests and the joint graph already own their decode peaks
        // in the ledger. Reuse newly released prefill credit now, otherwise a
        // second arrival can disable CUDA experts for its entire generation.
        RecoverExpertCache(PhysicalConstraints(), additionalDeviceBytes: 0, "prefill-complete");
    }

    private void RecoverReleasedSequenceCapacity(IReadOnlyList<SequenceState> running)
    {
        if (_native!.CallbackError is { } error)
            throw new InvalidOperationException("Optional expert recovery is disabled after a native accounting callback failure.", error);
        // A sibling can keep generating for minutes after another request
        // finishes or is cancelled. Its returned unused peak must become
        // available to expert reuse now, not only when a new request arrives.
        // Native holders and retained checkpoints still own their children;
        // joint payloads also retain their own charge and capacity high water.
        PruneFinishedRequestOwners();
        if (running.Count == 0) ReleaseIdleJointCredit();
        RecoverExpertCache(PhysicalConstraints(), additionalDeviceBytes: 0, "sequence-released");
    }

    private MemoryCharge[] IncrementalPeak(InferenceMemoryBytes peak)
    {
        // This exclusive, trunk-only adapter preloads non-expert quantized
        // weights only. The explicit preload registry excludes both lazy KV
        // copies and the selected-expert cache (separate charge kind 4).
        return IncrementalPeak(peak, BoundTrunkWeights());
    }

    private static long BoundTrunkWeights()
        => GgmlBasicOps.TryGetCacheMemoryUsage(0, out var cache) ? cache.PreloadCommittedBytes : 0;

    internal MemoryCharge[] IncrementalPeak(InferenceMemoryBytes peak, long boundTrunkWeights)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(boundTrunkWeights);
        // Text execution does not run the image tower. Its live bindings already
        // own ledger credit; cold vision bindings are forecast by media admission
        // only, rather than reserving the whole tower again for every text request.
        long pendingTrunk = Math.Max(0, _geometry.LoadingDeviceBoundBytes - boundTrunkWeights);
        // Opaque host payloads and graph arenas are not reusable credit:
        // text/image shapes can differ and old/new arenas can coexist.
        long host = checked(peak.Host + (64L << 20));
        // Selected experts have one session envelope. Charging the unused
        // allowance again per request would reserve the same cache N times.
        long device = checked(pendingTrunk + peak.Device);
        return [new(HostPool, host), new(DevicePool, device)];
    }

    /// <summary>Called only under the compute lock before a new request/encoder,
    /// never once per token. Re-sample after trimming; accounted cache allowance
    /// is deliberately not assumed to equal physically reclaimable VRAM.</summary>
    private IReadOnlyList<MemoryCharge> PrepareCapacity(InferenceMemoryBytes peak, bool recoverExpertCache = true)
        => PrepareCapacity(() => IncrementalPeak(peak), recoverExpertCache);

    private IReadOnlyList<MemoryCharge> PrepareCapacity(Func<IReadOnlyList<MemoryCharge>> estimate,
        bool recoverExpertCache = true)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        var model = _model ?? throw new InvalidOperationException("Unified model is not available for request admission.");
        if (_native!.CallbackError != null) throw new InvalidOperationException("Unified allocation accounting failed.", _native.CallbackError);
        // Wide CPU prefill reuses one charged file-read arena across layers.
        // Release that idle arena before forecasting a new request instead of
        // reserving a second copy. Keep all accounted CUDA expert entries warm.
        _experts!.Trim(_experts.CachedBytes);
        IReadOnlyList<MemoryCharge> limits = PhysicalConstraints();
        var needed = estimate();
        long previousCacheBudget = _experts!.BudgetBytes;
        if (!Fits(needed, limits))
        {
            // Preserve useful exact state and warm experts before destroying
            // either cache. A retained holder already has charged host mirrors;
            // only the bounded recurrent export scratch needs a new owner.
            var demotion = UnifiedMemoryReclamationPolicy.DemoteRetainedDeviceState(
                Budget, DevicePool, new(needed, limits), PhysicalConstraints, estimate, TryDemoteRetainedHostKv);
            needed = demotion.Needed; limits = demotion.Limits;
            // Keep the largest optional cache that leaves the measured deficit.
            var ledger = Budget.Snapshot().ToDictionary(p => p.Pool);
            long free = Math.Max(0, limits.Single(c => c.Pool == DevicePool).Bytes
                - ledger[DevicePool].Committed - ledger[DevicePool].Reserved);
            long deficit = Math.Max(0, needed.Single(c => c.Pool == DevicePool).Bytes - free);
            if (deficit > 0)
            {
                // Leave a little resize hysteresis for lazily materialized
                // driver pools. Repeated tiny per-layer reductions would throw
                // away every warm entry even for the same request geometry.
                long reduced = Math.Max(0, _experts.BudgetBytes - deficit - (128L << 20));
                ResizeExpertCache(reduced >= _geometry.AllLayerExpertCacheMinimumBytes ? reduced : 0, limits);
            }
            limits = PhysicalConstraints(); needed = estimate();
            if (!Fits(needed, limits))
            {
                model.TrimIdleMemory();
                limits = PhysicalConstraints(); needed = estimate();
            }
            if (!Fits(needed, limits))
            {
                model.TrimUnifiedPrefixCheckpoints();
                limits = PhysicalConstraints(); needed = estimate();
            }
            var afterTrim = Budget.Snapshot().Single(p => p.Pool == DevicePool);
            bool deviceStillFull = (decimal)needed.Single(c => c.Pool == DevicePool).Bytes
                + afterTrim.Committed + afterTrim.Reserved > limits.Single(c => c.Pool == DevicePool).Bytes;
            if (deviceStillFull && _experts!.BudgetBytes > 0)
            {
                ResizeExpertCache(0, limits);
                limits = PhysicalConstraints(); needed = estimate();
            }
            if (!Fits(needed, limits))
                throw new MemoryPressureException("Unified request peak exceeds currently available RAM/commit/VRAM after optional cache reclamation. " +
                    string.Join("; ", needed.Select(c => $"{c.Pool}: additional={c.Bytes / 1048576.0:F0} MiB, " +
                        $"constraint={limits.Single(p => p.Pool == c.Pool).Bytes / 1048576.0:F0} MiB"))
                    + ". " + DescribeExecutionOwnership());
        }
        // Recover optional reuse at request boundaries when a smaller shape or
        // reduced external pressure leaves room. Never use accounted bytes as a
        // substitute for the physical re-sample, and avoid tiny resize churn.
        if (recoverExpertCache)
            RecoverExpertCache(limits, needed.Single(c => c.Pool == DevicePool).Bytes, "request-admission");
        if (previousCacheBudget != _experts.BudgetBytes)
            Console.WriteLine($"[unified-memory] request-boundary expert cache: {previousCacheBudget / 1048576.0:F0} -> " +
                $"{_experts.BudgetBytes / 1048576.0:F0} MiB (physical capacity and request peak)");
        return limits;
    }

    private bool TryDemoteRetainedHostKv(IReadOnlyList<MemoryCharge> limits)
    {
        var model = _model!;
        if (!model.SupportsUnifiedRetainedHostKvDemotion || model.UnifiedRetainedDeviceHolderCount == 0) return false;
        long scratch = checked(model.GetUnifiedRetainedHostKvDemotionScratchBytes() + (32L << 20));
        BudgetReservation? maintenance;
        try { maintenance = Budget.TryReserve([new(HostPool, scratch), new(DevicePool, 0)], limits); }
        catch (MemoryPressureException) { return false; } // No device state was touched.
        if (maintenance == null) return false;
        using (maintenance)
        using (EnterAllocations(maintenance))
        {
            // Export/release errors propagate. In particular, a quarantined
            // partial release must never be reported as successful reclamation.
            int count = model.DemoteUnifiedRetainedHostKv(maximumHolders: 1);
            if (count == 0) return false;
            Console.WriteLine($"[unified-memory] pressure-demoted {count} retained complete holder(s) to charged host mirrors; physical capacity will be re-sampled.");
            return true;
        }
        // A retained scratch pool block can outlive this closed root. Its child
        // charge remains physical ownership and is never returned prematurely.
    }

    private string DescribeExecutionOwnership()
    {
        string Bytes(IReadOnlyList<MemoryCharge> charges) => string.Join(",", charges.Select(c => $"{c.Pool}={c.Bytes}"));
        return "ledger=" + string.Join(";", Budget.Snapshot().Select(p =>
            $"{p.Pool}(committed={p.Committed},reserved={p.Reserved},capacity={p.Capacity})"))
            + "; joint(targetHost=" + _jointTarget.Host + ",targetDevice=" + _jointTarget.Device
            + ",context=" + _jointContextHighWater + ",owned=" + Bytes(_jointEnvelope!.OwnedCharges)
            + ",children=" + Bytes(_jointEnvelope.LiveChildCharges) + ")"
            + "; requests=" + string.Join(";", _requestOwners.Values.Where(seq => seq.MemoryEnvelope != null).Select(seq =>
            {
                var shape = RequestShape(seq);
                var target = IncrementalPeak(RequestPeak(seq));
                return $"{seq.RequestId}(status={seq.Status},computed={seq.NumComputedTokens},prompt={seq.PromptTokens.Count},"
                    + $"capacity={shape.Capacity},prefill={shape.Prefill},target={Bytes(target)},"
                    + $"owned={Bytes(seq.MemoryEnvelope!.OwnedCharges)},children={Bytes(seq.MemoryEnvelope.LiveChildCharges)})";
            }));
    }

    private void RecoverExpertCache(IReadOnlyList<MemoryCharge> limits, long additionalDeviceBytes, string phase)
    {
        var current = Budget.Snapshot().Single(p => p.Pool == DevicePool);
        long previous = _experts!.BudgetBytes;
        long slack = Math.Max(0, limits.Single(c => c.Pool == DevicePool).Bytes
            - current.Committed - current.Reserved - additionalDeviceBytes);
        long recovery = Math.Min(Math.Max(0, slack - (128L << 20)),
            _experts.MaximumBudgetBytes - previous) / (1L << 20) * (1L << 20);
        long recovered = checked(previous + recovery);
        if (recovery < 512L << 20 || recovered < _geometry.AllLayerExpertCacheMinimumBytes) return;
        ResizeExpertCache(recovered, limits);
        if (_experts.BudgetBytes != previous)
            Console.WriteLine($"[unified-memory] {phase} expert cache: {previous / 1048576.0:F0} -> " +
                $"{_experts.BudgetBytes / 1048576.0:F0} MiB (live reservations and fresh physical capacity)");
    }

    private void ResizeExpertCache(long bytes, IReadOnlyList<MemoryCharge> limits)
        => UnifiedMemoryExpertCapacity.Resize(_experts!, _expertEnvelope!, DevicePool, bytes,
            immutableMinimum: 0, limits);

    private bool Fits(IReadOnlyList<MemoryCharge> peak, IReadOnlyList<MemoryCharge> limits)
    {
        var owned = Budget.Snapshot().ToDictionary(p => p.Pool);
        return peak.All(c => (decimal)c.Bytes + owned[c.Pool].Committed + owned[c.Pool].Reserved
            <= limits.Single(p => p.Pool == c.Pool).Bytes);
    }

    private MemoryCharge[] PhysicalConstraints()
    {
        var hardware = InferenceHardwareMemory.CaptureCuda();
        var host = hardware.HostAllocationCapacity();
        var ledger = Budget.Snapshot().ToDictionary(p => p.Pool);
        // Express additional physical allowance as an absolute ledger constraint.
        // Existing reserved credit still consumes it; no mapped/driver owner is
        // invented, and the configured allocation ceiling never increases.
        long hostLimit = (long)Math.Min(_hostCeiling, (decimal)ledger[HostPool].Committed + Math.Max(0, host.Available - _hostHeadroom));
        long deviceLimit = (long)Math.Min(_deviceCeiling, (decimal)ledger[DevicePool].Committed + Math.Max(0, hardware.DeviceAvailable - _deviceHeadroom));
        return [new(HostPool, hostLimit), new(DevicePool, deviceLimit)];
    }

    private IDisposable EnterVisionPreparation(QwenVisionPreparationPlan plan)
    {
        if (plan.ExpandedPromptTokens > _contextTokens) throw new MemoryPressureException("Image-expanded prompt exceeds the unified context.");
        var peak = VisionPreparationPeak(plan);
        // Image encoding is followed immediately by text execution. Do not
        // regrow the expert cache for this temporarily smaller workspace only
        // to evict its warm entries again at text admission.
        var constraints = PrepareCapacity(peak, recoverExpertCache: false);
        var owned = new OwnedExecution();
        var envelope = Budget.TryReserve(IncrementalPeak(peak), constraints)
            ?? throw new MemoryPressureException("Image preparation could not reserve its unified peak.");
        owned.Envelope = envelope;
        try { owned.Execution = EnterAllocations(envelope); return owned; }
        catch { owned.Dispose(); throw; }
    }

    internal static InferenceMemoryBytes VisionPreparationPeak(QwenVisionPreparationPlan plan)
        // Keep the full cold-tower + embedding + workspace forecast. The trunk
        // preload registry cannot be used as credit for any vision allocation.
        => new(plan.HostPeakBytes, plan.DevicePeakBytes);

    private IDisposable EnterAllocations(BudgetReservation envelope)
    {
        var owned = new AllocationExecution();
        try
        {
            owned.Native = _native!.EnterExecution(envelope);
            owned.Host = _host!.EnterExecution(envelope);
            return owned;
        }
        catch { owned.Dispose(); throw; }
    }

    private IDisposable EnterSequenceAllocations(SequenceState sequence)
    {
        EnsureSequenceCapacity(sequence, allowReclamation: true);
        return EnterAllocations(sequence.MemoryEnvelope!);
    }

    private void RevalidateSequenceAllocations(SequenceState sequence)
        => EnsureSequenceCapacity(sequence, allowReclamation: false);

    private void EnsureSequenceCapacity(SequenceState sequence, bool allowReclamation)
    {
        var envelope = sequence.MemoryEnvelope
            ?? throw new InvalidOperationException("Unified execution requires an admitted envelope.");
        var shape = RequestShape(sequence);
        // Prefix adoption may call this before publishing its final phase. The
        // initial envelope already owns that shape; do not re-admit a clone.
        if (_requestShapes.TryGetValue(sequence.RequestId, out var previous) && previous != shape)
        {
            PruneFinishedRequestOwners();
            var peak = IncrementalPeak(RequestPeak(sequence));
            var (joint, tokens) = ExecutionJointTarget(sequence);
            var additional = ExecutionEnvelopeGrowth(envelope, peak, _jointEnvelope!, joint);
            if (!allowReclamation)
            {
                // A revoked prefix can change phase even at row1. Its full peak
                // must fit before replay/Forward; current scopes cannot be nested
                // by optional cache-demotion maintenance. Refusal stays local.
                ResizeExecutionEnvelopes(envelope, peak, joint, tokens, PhysicalConstraints(), requirePhysicalFit: true);
            }
            else if (additional.Host > 0 || additional.Device > 0)
            {
                // These are already incremental ledger charges. Adding pending
                // trunk weights or the request's 64-MiB host margin again would
                // evict expert reuse for space the owners already reserved.
                var limits = PrepareCapacity(() =>
                {
                    var growth = ExecutionEnvelopeGrowth(envelope, peak, _jointEnvelope!, joint);
                    return [new(HostPool, growth.Host), new(DevicePool, growth.Device)];
                }, recoverExpertCache: false);
                ResizeExecutionEnvelopes(envelope, peak, joint, tokens, limits);
            }
            else ResizeExecutionEnvelopes(envelope, peak, joint, tokens);
            _requestShapes[sequence.RequestId] = shape;
        }
    }

    private (InferenceMemoryBytes Target, int ContextTokens) ExecutionJointTarget(SequenceState sequence)
    {
        int lanes = _requestOwners.Count + (_requestOwners.ContainsKey(sequence.RequestId) ? 0 : 1);
        int tokens = Math.Max(RequestShape(sequence).Capacity, _jointContextHighWater);
        foreach (var live in _requestOwners.Values) tokens = Math.Max(tokens, RequestShape(live).Capacity);
        var joint = lanes >= 2 ? _geometry.JointDecodePeak(tokens, lanes) : _jointTarget;
        return (new(Math.Max(joint.Host, _jointTarget.Host), Math.Max(joint.Device, _jointTarget.Device)), tokens);
    }

    private void PruneFinishedRequestOwners()
    {
        foreach (var stale in _requestOwners.Where(p => p.Value.Status.IsFinished()).Select(p => p.Key).ToArray())
        {
            _requestPrefillCaps.Remove(_requestOwners[stale]);
            _requestOwners.Remove(stale);
            _requestShapes.Remove(stale);
        }
    }

    internal static InferenceMemoryBytes ExecutionEnvelopeGrowth(BudgetReservation request,
        IReadOnlyList<MemoryCharge> requestPeak, BudgetReservation joint, InferenceMemoryBytes jointTarget)
    {
        var holderOwned = request.OwnedCharges;
        var arenaOwned = joint.OwnedCharges;
        long Net(string pool, long arenaTarget) => checked(
            Math.Max(requestPeak.Single(c => c.Pool == pool).Bytes, request.LiveChildCharges.Single(c => c.Pool == pool).Bytes)
            - holderOwned.Single(c => c.Pool == pool).Bytes
            + Math.Max(arenaTarget, joint.LiveChildCharges.Single(c => c.Pool == pool).Bytes)
            - arenaOwned.Single(c => c.Pool == pool).Bytes);
        return new(Math.Max(0, Net(HostPool, jointTarget.Host)), Math.Max(0, Net(DevicePool, jointTarget.Device)));
    }

    internal void ResizeExecutionEnvelopes(BudgetReservation request, IReadOnlyList<MemoryCharge> requestPeak,
        InferenceMemoryBytes joint, int contextTokens, IReadOnlyList<MemoryCharge>? limits = null, bool requirePhysicalFit = false)
    {
        if (!Budget.TryResizeTotals([(_jointEnvelope!, [new(HostPool, joint.Host), new(DevicePool, joint.Device)]),
            (request, requestPeak)], limits, requirePhysicalFit))
            throw new MemoryPressureException("The next holder growth cannot fit the shared budget; generation was stopped before allocation.");
        _jointTarget = joint;
        _jointContextHighWater = contextTokens;
    }

    private sealed class AllocationExecution : IDisposable
    {
        internal IDisposable? Native, Host;
        public void Dispose() { try { Host?.Dispose(); } finally { Native?.Dispose(); } }
    }
    private sealed class OwnedExecution : IDisposable
    {
        internal IDisposable? Execution, Envelope;
        public void Dispose() { try { Execution?.Dispose(); } finally { Envelope?.Dispose(); } }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _model?.Dispose(); _model = null;
        _jointArena?.Dispose(); _jointArena = null;
        _experts?.Dispose(); _experts = null;
        _filePipeline?.Dispose(); _filePipeline = null;
        _native?.SetSharedJointEnvelope(null);
        _jointEnvelope?.Dispose(); _jointEnvelope = null;
        _native?.SetSharedExpertEnvelope(null);
        _expertEnvelope?.Dispose(); _expertEnvelope = null;
        _host?.Dispose(); _host = null;
        _native?.Dispose(); _native = null;
        _disposed = true;
    }

    private static int ParallelDegree(string name)
    {
        string? value = Environment.GetEnvironmentVariable(name);
        if (string.IsNullOrWhiteSpace(value)) return 1;
        return int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out int degree) && degree > 0
            ? degree : throw new ArgumentException(name + " must be a positive degree.");
    }

    private static void ValidateOptions(UnifiedMemoryModelOptions options)
    {
        if (options.ContextTokens < 0 || options.PrefillChunkTokens is < 1 or > 512
            || options.ExpertCachePrefillRows is < 1 or > 128
            || options.MaximumRunningRequests is < 1 or > 8 || options.PrefixCheckpointCount is < 0 or > 8
            || options.RetainedConversationCount is < 0 or > 8
            || options.MaximumHostBytes <= 0 || options.MaximumDeviceBytes <= 0 || options.MaximumExpertCacheBytes < 0)
            throw new ArgumentOutOfRangeException(nameof(options));
    }
}

internal readonly record struct UnifiedMemoryPrefillPlan(int Rows, IReadOnlyList<MemoryCharge> Peak);
internal readonly record struct UnifiedMemoryPrefillCacheReclaim(int PreferredRows, long TargetCacheBytes, long DeviceShortfallBytes);

/// <summary>Actual prefill-shape admission. The pure pass never reclaims; the
/// performance pass can trade optional cache for a wider shape once, then must
/// re-sample and pass the same strict atomic ownership transaction.</summary>
internal static class UnifiedMemoryPrefillAdmission
{
    internal static IEnumerable<int> Candidates(int maximum)
    {
        if (maximum is < 1 or > 512) throw new ArgumentOutOfRangeException(nameof(maximum));
        yield return maximum;
        // Evaluate each actual shape; do not infer a fit by binary search or
        // assume every backend's graph/cache geometry is monotonic in rows.
        for (int rows = 256; rows >= 1; rows /= 2)
            if (rows < maximum) yield return rows;
    }

    internal static bool TrySelect(MemoryBudget budget, BudgetReservation request, BudgetReservation joint,
        InferenceMemoryBytes jointTarget, int maximum, Func<int, IReadOnlyList<MemoryCharge>> estimate,
        IReadOnlyList<MemoryCharge> limits, out UnifiedMemoryPrefillPlan selected, bool allowDowngrade = true, int minimumRows = 1)
    {
        if (minimumRows < 1 || minimumRows > maximum) throw new ArgumentOutOfRangeException(nameof(minimumRows));
        foreach (int rows in Candidates(maximum))
        {
            if (rows < minimumRows) break;
            var peak = estimate(rows);
            if (!TryAdmit(budget, request, joint, jointTarget, peak, limits))
            {
                if (!allowDowngrade) break;
                continue;
            }
            selected = new(rows, peak);
            return true;
        }
        selected = default;
        return false;
    }

    private static bool TryAdmit(MemoryBudget budget, BudgetReservation request, BudgetReservation joint,
        InferenceMemoryBytes jointTarget, IReadOnlyList<MemoryCharge> peak, IReadOnlyList<MemoryCharge> limits)
        => budget.TryResizeTotals([(joint, [new(UnifiedMemoryModelSession.HostPool, jointTarget.Host),
            new(UnifiedMemoryModelSession.DevicePool, jointTarget.Device)]), (request, peak)],
            limits, requirePhysicalFit: true);

    internal static bool TrySelectPerformance(MemoryBudget budget, BudgetReservation request, BudgetReservation joint,
        InferenceMemoryBytes jointTarget, int maximum, Func<int, IReadOnlyList<MemoryCharge>> estimate,
        Func<IReadOnlyList<MemoryCharge>> constraints, long cacheBytes, long minimumCacheBytes,
        Action<UnifiedMemoryPrefillCacheReclaim, IReadOnlyList<MemoryCharge>> reclaim, ref bool reclaimAttempted,
        out UnifiedMemoryPrefillPlan selected)
    {
        if (reclaimAttempted)
            return TrySelect(budget, request, joint, jointTarget, maximum, estimate, constraints(), out selected);
        var limits = constraints();
        // Preserve the widest affordable actual shape before accepting a
        // smaller one. A pure 128-row fit must not hide a cheap cache tradeoff
        // that would protect the configured 512-row prefill. Conversely a host
        // shortfall cannot be cured by retiring CUDA weights. Price every shape
        // independently; short prefixes keep the original no-maintenance path.
        foreach (int rows in Candidates(maximum))
        {
            var peak = estimate(rows);
            if (TryAdmit(budget, request, joint, jointTarget, peak, limits))
            {
                selected = new(rows, peak);
                return true;
            }
            if (rows < 32 || !TryPlanDeviceCacheReclaim(budget, request, joint, jointTarget, rows, peak, limits,
                    cacheBytes, minimumCacheBytes, out var reduction)) continue;
            // Native releases/fences physical entries before their persistent
            // owner refunds credit. At most one coarse trim per admission;
            // every retry observes fresh hardware and an atomic post-ledger fit.
            reclaimAttempted = true;
            reclaim(reduction, limits);
            limits = constraints();
            return TrySelect(budget, request, joint, jointTarget, maximum, estimate, limits, out selected);
        }
        selected = default;
        return false;
    }

    private static bool TryPlanDeviceCacheReclaim(MemoryBudget budget, BudgetReservation request, BudgetReservation joint,
        InferenceMemoryBytes jointTarget, int rows, IReadOnlyList<MemoryCharge> requestPeak, IReadOnlyList<MemoryCharge> limits,
        long cacheBytes, long minimumCacheBytes, out UnifiedMemoryPrefillCacheReclaim reduction)
    {
        reduction = default;
        if (minimumCacheBytes <= 0 || cacheBytes <= minimumCacheBytes) return false;
        var ledger = budget.Snapshot().ToDictionary(p => p.Pool);
        var requestOwned = request.OwnedCharges; var requestChildren = request.LiveChildCharges;
        var jointOwned = joint.OwnedCharges; var jointChildren = joint.LiveChildCharges;
        long Bytes(IReadOnlyList<MemoryCharge> charges, string pool) => charges.Single(c => c.Pool == pool).Bytes;
        long Shortfall(string pool, long jointBytes)
        {
            var current = ledger[pool];
            long post = checked(current.Committed + current.Reserved
                + Math.Max(Bytes(requestPeak, pool), Bytes(requestChildren, pool)) - Bytes(requestOwned, pool)
                + Math.Max(jointBytes, Bytes(jointChildren, pool)) - Bytes(jointOwned, pool));
            return Math.Max(0, checked(post - Math.Min(current.Capacity, Bytes(limits, pool))));
        }
        if (Shortfall(UnifiedMemoryModelSession.HostPool, jointTarget.Host) != 0) return false;
        long deficit = Shortfall(UnifiedMemoryModelSession.DevicePool, jointTarget.Device);
        if (deficit <= 0 || deficit > cacheBytes - minimumCacheBytes) return false;
        long target = Math.Max(minimumCacheBytes, cacheBytes - deficit - Math.Min(64L << 20, cacheBytes - deficit - minimumCacheBytes));
        reduction = new(rows, target, deficit);
        return true;
    }
}

/// <summary>Cleanup failed during construction. Keep this owner and retry its
/// Dispose before loading any replacement; allocation callbacks stay attached.</summary>
public sealed class UnifiedMemoryModelLoadException : InvalidOperationException
{
    public UnifiedMemoryModelLoadException(IUnifiedMemoryModelSession session, Exception inner)
        : base("Unified model construction failed with unreleased owners; cleanup must be retried.", inner)
        => UnreleasedSession = session;
    public IUnifiedMemoryModelSession UnreleasedSession { get; }
}
