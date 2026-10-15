// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Runtime.CompilerServices;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Memory.Planning;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Models;

/// <summary>Single-device native GLM5Next shared-budget owner. Native weights,
/// complete MLA/indexer/KDA slots and cached scheduler payloads use the same
/// ledger as managed tensors and upload/media staging. Source mappings, GC and
/// vendor pools are observed with headroom rather than claimed as hard-capped.</summary>
public sealed class GlmUnifiedMemoryModelSession : IUnifiedMemoryModelSession
{
    private const string HostPool = UnifiedMemoryModelSession.HostPool;
    private const string DevicePool = UnifiedMemoryModelSession.DevicePool;
    private const int GraphEntriesWithTransient = GlmDsaModel.UnifiedGraphCacheCapacity + 1;
    private readonly UnifiedMemoryModelOptions _options;
    private readonly GlmUnifiedMemoryGeometry _geometry;
    private readonly GlmUnifiedVisionMemoryGeometry? _vision;
    private readonly long _hostCeiling, _deviceCeiling, _hostHeadroom, _deviceHeadroom;
    private readonly int _context, _prefill;
    private readonly int _prefixCopies;
    private int _effectivePrefixCopies;
    private GgmlCacheBudgetScope? _native;
    private HostAllocationBudgetScope? _host;
    private BudgetReservation? _joint;
    private GgmlHostMoeExpertCacheScope? _experts;
    private BudgetReservation? _expertEnvelope;
    private long _expertCacheBytes, _expertMinimumBytes, _expertMinimumForecastBytes;
    private readonly long _executionReserveBytes;
    private int _expertGroups;
    private GlmDsaModel? _model;
    private bool _disposed;
    private int _cpuLayers;
    private GlmUnifiedGraphShape? _lastGraphShape;
    private BudgetReservation? _executingRequest;
    private SequenceState? _executingSequence;
    private int _prefixCaptureDiagnostics;
    private sealed class PrefillState(int cap)
    {
        internal int Cap = cap;
        internal InferenceMemoryBytes Peak;
        internal WeakReference<SequenceState>[] Siblings = [];
    }
    private readonly ConditionalWeakTable<SequenceState, PrefillState> _requestPrefill = new();
    private readonly GlmUnifiedPrimarySlotClaim _primarySlotClaim = new();

    private GlmUnifiedMemoryModelSession(UnifiedMemoryModelOptions options, GlmUnifiedMemoryGeometry geometry,
        GlmUnifiedVisionMemoryGeometry? vision, InferenceHardwareMemory hardware, int context)
    {
        _options = options; _geometry = geometry; _vision = vision; _context = context;
        var host = hardware.HostAllocationCapacity();
        _hostHeadroom = Math.Max(512L << 20, Math.Min(host.Total, options.MaximumHostBytes) / 16);
        _deviceHeadroom = Math.Max(768L << 20, Math.Min(hardware.DeviceTotal, options.MaximumDeviceBytes) / 16);
        _hostCeiling = Math.Min(options.MaximumHostBytes, Math.Max(0, host.Available - _hostHeadroom));
        _deviceCeiling = Math.Min(options.MaximumDeviceBytes, Math.Max(0, hardware.DeviceAvailable - _deviceHeadroom));
        if (_hostCeiling <= 0 || _deviceCeiling <= 0) throw new MemoryPressureException("No GLM RAM/commit/CUDA capacity remains after headroom.");
        var prefix = UnifiedMemoryPrefixCapacity.Select(GgmlGlmNative.SupportsSlotCopy()
            ? options.PrefixCheckpointCount : 0, geometry.SlotBytes(context), _deviceCeiling);
        GlmSelectedExpertMemoryPlan plan;
        while (true)
        {
            try
            {
                plan = GlmSelectedExpertMemoryPlan.SelectForSession(geometry, options, context, _deviceCeiling,
                    checked((vision?.DeviceWeightBytes ?? 0) + prefix.PlacementReserveBytes
                        // Fresh clone destinations coexist with the idle primary.
                        // Count it rather than assuming a donation consumes it.
                        + (prefix.RetainedCopies > 0 ? geometry.SlotBytes(context) : 0)));
                break;
            }
            catch (MemoryPressureException) when (prefix.RetainedCopies > 0)
            {
                int copies = prefix.RetainedCopies - 1;
                prefix = new(copies, copies > 0 ? checked((long)copies * geometry.SlotBytes(context)) : 0);
            }
        }
        _prefixCopies = prefix.RetainedCopies;
        _prefill = plan.PrefillRows;
        _cpuLayers = plan.CpuExpertLayers;
        _expertCacheBytes = plan.CacheCapacityBytes;
        _expertMinimumForecastBytes = plan.CacheMinimumForecastBytes;
        _executionReserveBytes = plan.ExecutionReserveBytes;
        Budget = new MemoryBudget([new(HostPool, _hostCeiling), new(DevicePool, _deviceCeiling)]);
    }
    public ModelBase? Model => _model;
    public MemoryBudget Budget { get; }
    public int ContextTokens => _context;
    public int PrefillChunkTokens => _prefill;
    public string FeatureReport =>
        $"[unified-memory] enabled: {_geometry.Architecture}, native GGML CUDA; context={_context}, prefill={_prefill}, active-request-limit={_options.MaximumRunningRequests}\n" +
        $"[unified-memory] allocation ceilings: RAM={_hostCeiling / 1048576.0:F0} MiB, CUDA={_deviceCeiling / 1048576.0:F0} MiB; CPU-expert-leading-layers={_cpuLayers}/{_geometry.Layers}\n" +
        "[unified-memory] on: native weight/constant/slot/graph accounting, host tensor/upload accounting, physical RAM/commit/VRAM headroom, request admission, native joint decode, bounded donation of exact terminal-state conversation prefixes, admitted still-image preparation\n" +
        $"[unified-memory] public prefix capacity: planned={_prefixCopies}, effective={_effectivePrefixCopies}; retained complete-state copies are placed, optional capture-before-eviction requires live spare capacity, no recurrent rewind or disk persistence\n" +
        (_experts == null
            ? (_options.EnableGlmSelectedExperts && _cpuLayers == 0
                ? "[unified-memory] GLM experts: all routed weights are resident on CUDA; a compact selected-expert tier is unnecessary for this capacity plan\n"
                : "[unified-memory] GLM experts: capacity-placed resident CUDA layers and pageable mapped CPU layers; selected-expert CUDA reuse is off pending full-model qualification\n")
            : $"[unified-memory] GLM experts: experimental pooled CUDA reuse; groups={_expertGroups}, budget={_expertCacheBytes / 1048576.0:F0} MiB, metadata forecast={_expertMinimumForecastBytes / 1048576.0:F0} MiB, exact minimum={_expertMinimumBytes / 1048576.0:F0} MiB, planned decode widths=1..{_options.MaximumRunningRequests}, prefill rows up to {_prefill} (requested {_options.ResolveGlmPrefillChunkTokens()}); wide prefill protects its entire routed union; bounded source-file staging; full-model performance remains unqualified\n") +
        (_options.EnableAdaptiveGlmPrefill
            ? "[unified-memory] adaptive GLM prefill: actual prompt/joint shapes, full physical KV slots, immutable expert phase floor, fenced request-boundary cache reclaim; widths may decrease before scheduling under pressure\n" : "") +
        "[unified-memory] asynchronous file/transfer/compute overlap is off pending GLM qualification\n" +
        "[unified-memory] off: host KV paging, MTP/draft and TP (complete shared-budget adapters remain unqualified), video/audio\n" +
        "[unified-memory] limits: mapping/page-cache residency, GC, native metadata and vendor/driver pools retain observed headroom; no complete OS RAM/VRAM hard-cap claim.";

    public static GlmUnifiedMemoryModelSession Create(string path, string? mmprojPath, BackendType backend,
        UnifiedMemoryModelOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (backend != BackendType.GgmlCuda || options.MaximumRunningRequests is < 1 or > 8
            || options.PrefillChunkTokens is < 1 or > 512 || options.ContextTokens < 0
            || options.MaximumHostBytes <= 0 || options.MaximumDeviceBytes <= 0 || options.MaximumExpertCacheBytes < 0
            || options.RetainedConversationCount is < 0 or > 8 || options.PrefixCheckpointCount is < 0 or > 8)
            throw new NotSupportedException("Unified GLM requires GGML CUDA and valid single-device request geometry.");
        if (Environment.GetEnvironmentVariable("TS_GLM_NATIVE") == "0" || SpeculationOptions.FromEnvironment().Enabled)
            throw new NotSupportedException("Unified GLM requires its native, non-speculative executor.");
        if (options.EnableAsyncExpertPrefill || (options.EnableGlmSelectedExperts && options.MaximumRunningRequests > 3))
            throw new NotSupportedException("Experimental unified GLM supports at most three selected-expert decode rows and no asynchronous expert pipeline.");
        if (options.EnableAdaptiveGlmPrefill && !options.EnableGlmSelectedExperts)
            throw new NotSupportedException("Adaptive GLM prefill requires its complete selected-expert planning policy.");
        _ = options.ResolveGlmPrefillChunkTokens(); // Validate the separate GLM bound before native/model allocation.
        GgmlGlmNative.RequireUnifiedBudgetSupport();
        if (options.EnableGlmSelectedExperts) GgmlGlmNative.RequireSelectedExpertPlanningSupport();
        GlmUnifiedMemoryGeometry geometry;
        using (var file = GgufFile.OpenForMemoryPlanning(path)) geometry = GlmUnifiedMemoryGeometry.Read(file);
        var vision = string.IsNullOrWhiteSpace(mmprojPath) || mmprojPath == "none" ? null : GlmUnifiedVisionMemoryGeometry.Read(mmprojPath);
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
        if (context <= 0 || context > geometry.DeclaredContextTokens) throw new ArgumentOutOfRangeException(nameof(options));
        var owner = new GlmUnifiedMemoryModelSession(options, geometry, vision, InferenceHardwareMemory.CaptureCuda(), context);
        try
        {
            owner._native = new(owner.Budget, [[DevicePool], [HostPool]], includeGraphBuffers: true, hostPools: [HostPool]);
            owner._host = new(owner.Budget, [HostPool]);
            owner._joint = owner.Budget.Reserve([new(HostPool, 0), new(DevicePool, 0)]);
            owner._native.SetSharedJointEnvelope(owner._joint);
            if (owner._expertCacheBytes > 0)
            {
                owner._expertEnvelope = owner.Budget.Reserve([new(DevicePool, owner._expertCacheBytes)]);
                owner._native.SetSharedExpertEnvelope(owner._expertEnvelope);
            }
            var policy = new ModelMemoryPolicy(context, owner._prefill)
            {
                OmitEmbeddedDraftWeights = true, NativeMoeCpuLayers = owner._cpuLayers,
                DeviceExecutionReserveBytes = owner._executionReserveBytes
            };
            owner._model = (GlmDsaModel)ModelBase.Create(path, backend, 1, null!, null!, 1, null!, policy);
            if (owner._model.MaxContextLength != context) throw new InvalidOperationException("Native GLM changed an explicit unified context.");
            owner._model.ConfigureUnifiedMemory(checked(options.RetainedConversationCount + owner._prefixCopies));
            if (owner._prefixCopies > 0)
                owner._model.ConfigureUnifiedPrefixCopies(geometry.SlotBytes(context), owner.EnterPrefixCapture,
                    owner.PrefixCloneSpareBytes);
            if (owner._expertCacheBytes > 0)
            {
                owner._expertGroups = owner._model.UnifiedExpertGroups;
                if (owner._expertGroups <= 0) throw new NotSupportedException("No complete mapped GLM expert layout groups were found.");
                owner._experts = new(owner._expertCacheBytes, owner._prefill, owner._expertGroups, options.MaximumRunningRequests);
                owner._expertMinimumBytes = owner._model.PlanUnifiedExpertMinimum(options.MaximumRunningRequests);
                if (owner._expertCacheBytes < owner._expertMinimumBytes)
                    throw new MemoryPressureException($"GLM pooled expert budget {owner._expertCacheBytes} bytes cannot protect its complete admitted union/graph minimum {owner._expertMinimumBytes} bytes.");
                owner._model.ConfigureUnifiedSelectedExperts();
                owner._model.TrimUnifiedMappedResidency(); // Advisory only; no imputed released-byte credit.
            }
            owner._model.ConfigureUnifiedGraphAdmission(owner.PrepareGraph);
            owner._model.ConfigureUnifiedPrimarySlotReuse(requestId => owner._primarySlotClaim.AllowsBinding(requestId, owner.HasNativeSlot));
            if (vision != null) owner._model.LoadVisionEncoder(mmprojPath!);
            owner._model.ConfigureUnifiedVisionAdmission(owner.EnterVisionPreparation);
            Console.WriteLine(owner.FeatureReport);
            return owner;
        }
        catch (Exception creation)
        {
            if (creation is GlmConstructionCleanupException partial) owner._model = partial.UnreleasedModel;
            try { owner.Dispose(); }
            catch (Exception cleanup) { throw new UnifiedMemoryModelLoadException(owner, new AggregateException(creation, cleanup)); }
            throw;
        }
    }

    public SchedulerConfig ConfigureScheduler(SchedulerConfig requested)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (requested.Speculation.Enabled || requested.KvSnapshots != null || requested.MemoryAdmission != null)
            throw new NotSupportedException("Unified GLM cannot combine with independent speculation or snapshot/admission adapters.");
        int block = requested.BlockSize > 0 ? requested.BlockSize : SchedulerConfig.DefaultBlockSize;
        _effectivePrefixCopies = requested.EnablePrefixCaching
            ? Math.Min(_prefixCopies, (requested.ExecutionOptionsOverride ?? ExecutionOptions.FromEnvironment()).PrefixCheckpointBudget) : 0;
        return new SchedulerConfig
        {
            BlockSize = block, NumBlocks = checked((_context + block - 1) / block *
                (_options.MaximumRunningRequests + _options.RetainedConversationCount + _effectivePrefixCopies + 1)),
            MaxNumRunningSequences = _options.MaximumRunningRequests, MaxNumBatchedTokens = _prefill,
            MaxPrefillChunkSize = _prefill, SoloPrefillChunkSize = _prefill, PrefillChunkTokenLimit = _prefill,
            EnablePrefixCaching = requested.EnablePrefixCaching, StopRepetition = requested.StopRepetition,
            DecodeQuantumTokens = requested.DecodeQuantumTokens, Speculation = SpeculationOptions.Disabled,
            ExecutionOptionsOverride = new ExecutionOptions
            {
                BatchedPathDisabled = true, PerSeqFusedEnabled = true, BatchedFusedDecodeEnabled = true,
                RetainedFusedCacheBudget = requested.EnablePrefixCaching ? _options.RetainedConversationCount : 0,
                PrefixCheckpointBudget = _effectivePrefixCopies, KvHolderPoolMax = 0
            },
            MemoryAdmission = new RequestMemoryAdmission(Budget, EstimateRequest)
            {
                UsePerSequenceHolders = true, ReestimateAfterPhysicalCheck = true,
                EstimateMaterializationPeak = EstimateMaterialization,
                BeforeAdmissionWithRunning = PrepareAdmission,
                AfterPrefixMaterialization = CompletePrefix, AfterPrefill = CompletePrefill,
                PrefillTokenLimit = AdmittedPrefillLimit,
                AfterSequenceMemoryRelease = RecoverReleasedCapacity,
                EnterSerialExecution = EnterSerial,
                EnterBatchedExecution = _ => EnterAllocations(_joint
                    ?? throw new InvalidOperationException("GLM joint execution requires its shared graph envelope."))
            }
        };
    }
    private InferenceMemoryBytes RequestPeak(SequenceState seq, bool materializing = false)
        => RequestPeakForRows(seq, AdmittedPrefillLimit(seq), materializing);
    private int AdmittedPrefillLimit(SequenceState seq)
        => _requestPrefill.TryGetValue(seq, out var state) ? state.Cap : _prefill;
    private InferenceMemoryBytes RequestPeakForRows(SequenceState seq, int maximumRows, bool materializing)
    {
        int tokens = checked(seq.PromptTokens.Count + seq.MaxNewTokens);
        if (tokens > _context) throw new MemoryPressureException("GLM request exceeds its unified context.");
        int rows = Math.Min(maximumRows, Math.Max(1, seq.PromptTokens.Count - seq.NumComputedTokens));
        var staging = _geometry.Workspace(Math.Max(1, seq.PromptTokens.Count), rows, outputRows: 1);
        // A Submit estimate always includes a new full slot. At the fenced
        // materialization boundary the model can reuse an actual idle primary
        // or donated holder. Existing child ownership is never revoked.
        // Copy materialization initially uses a new independent slot. Unlike a
        // donation it must not consume/reset a primary still advertised by the
        // radix tree. Its complete allocation is admitted before adoption.
        bool reusable = _effectivePrefixCopies > 0 ? HasNativeSlot(seq) : CanReuseSlotFor(seq);
        long slot = materializing && reusable ? 0 : _geometry.SlotBytes(_context);
        return new(staging.Host, slot);
    }
    private bool HasNativeSlot(SequenceState seq) => _model!.CanReuseUnifiedSlot(seq.RequestId, includeIdlePrimary: false);
    private bool CanReuseSlotFor(SequenceState seq) => _primarySlotClaim.CanReuse(seq, HasNativeSlot(seq),
        _model!.CanReuseUnifiedSlot(seq.RequestId), HasNativeSlot);
    private void PublishPrimarySlotClaim(WeakReference<SequenceState> prepared)
    {
        if (!prepared.TryGetTarget(out var seq)) throw new InvalidOperationException("GLM materialization lost its live request.");
        if (!CanReuseSlotFor(seq)) return; // This request owns its complete new-slot allowance instead.
        _primarySlotClaim.PublishAfterMaterialization(prepared, HasNativeSlot(seq),
            _model!.CanReuseUnifiedSlot(seq.RequestId), HasNativeSlot);
    }
    private IReadOnlyList<MemoryCharge> EstimateRequest(SequenceState seq) => Charges(RequestPeak(seq));
    private IReadOnlyList<MemoryCharge> EstimateMaterialization(SequenceState seq) => Charges(RequestPeak(seq, true));
    private static MemoryCharge[] Charges(InferenceMemoryBytes bytes) => [new(HostPool, bytes.Host), new(DevicePool, bytes.Device)];
    private IReadOnlyList<MemoryCharge> PrepareAdmission(SequenceState seq, IReadOnlyList<SequenceState> running)
    {
        if (_options.EnableAdaptiveGlmPrefill) return PrepareAdaptiveAdmission(seq, running);
        if (_native!.CallbackError != null) throw new InvalidOperationException("GLM allocation accounting failed.", _native.CallbackError);
        var physical = ObservePhysicalConstraints();
        var limits = physical.Limits;
        var graphs = GlmUnifiedAdmissionGraphPlan.Create(_geometry, _context, _prefill,
            seq.PromptTokens.Count, seq.NumComputedTokens, running.Select(live => live.NumComputedTokens).ToArray(),
            GraphEntriesWithTransient);
        var target = PreserveReplacement(graphs.MaximumPeak);
        var needed = EstimateMaterialization(seq);
        MemoryCharge[] growth = [];
        bool? mappedTrimHint = null;
        bool FitsCombined()
        {
            var owned = _joint!.OwnedCharges.ToDictionary(c => c.Pool, c => c.Bytes);
            growth = Charges(target).Select(c => new MemoryCharge(c.Pool,
                checked(c.Bytes - owned[c.Pool] + needed.Single(n => n.Pool == c.Pool).Bytes))).ToArray();
            physical = physical with { Ledger = Budget.Snapshot() };
            return physical.Fits(growth);
        }
        MemoryPressureException Refusal(string message)
        {
            string Describe(IEnumerable<MemoryCharge> charges) => string.Join(", ", charges.Select(c =>
                FormattableString.Invariant($"{c.Pool}={c.Bytes}")));
            return new MemoryPressureException(message + " " + FormattableString.Invariant(
                $"request={seq.RequestId}; prompt={seq.PromptTokens.Count}; computed={seq.NumComputedTokens}; max-new={seq.MaxNewTokens}; logical-context={_context}; serial-graph-context={graphs.SerialContext}; joint-graph-context={graphs.JointContext}; prefill-rows={graphs.PrefillRows}; joint-rows={graphs.JointRows}; active-requests={running.Count}; joint-target=[{Describe(Charges(target))}]; joint-owned=[{Describe(_joint!.OwnedCharges)}]; joint-children=[{Describe(_joint.LiveChildCharges)}]; materialization=[{Describe(needed)}]; mapped-trim-hint={mappedTrimHint?.ToString() ?? "not-attempted"}; ")
                + physical.Describe(growth));
        }
        if (!FitsCombined())
        {
            _model!.TrimUnifiedGraphs(); _model.TrimUnifiedRetainedSlots();
            physical = ObservePhysicalConstraints(); limits = physical.Limits;
            _lastGraphShape = null;
            target = PreserveReplacement(graphs.MaximumPeak);
            needed = EstimateMaterialization(seq);
            if (!FitsCombined())
            {
                // Pageable expert scans can consume observed RAM without an
                // anonymous Host ticket. At this quiescent boundary hint only
                // those mappings, and accept no imputed released-byte credit.
                if (physical.TryRefreshMappedHostPressure(growth, _model.TrimUnifiedMappedResidency,
                    ObservePhysicalConstraints, out var refreshed, out mappedTrimHint))
                {
                    physical = refreshed; limits = physical.Limits;
                }
                if (!FitsCombined()) throw Refusal("GLM slot and shared graph peak exceed current RAM/commit/VRAM after idle graph/prefix reclamation.");
            }
        }
        // Combined capacity is checked before consuming any shared credit. The
        // scheduler then acquires the separate materialization envelope under
        // its compute lock; live shared children preserve their physical floor.
        if (!_joint!.TryResizeTotal(Charges(target), limits))
        {
            FitsCombined(); // Refresh the observed ledger if another owner changed it before the atomic reserve.
            throw Refusal("GLM shared graph admission could not reserve its replacement peak.");
        }
        return limits;
    }
    private void CompletePrefix(SequenceState seq)
    {
        // Allocate claim bookkeeping before native/ledger mutations. Admission
        // estimates do not claim anything: several waiting requests may sample
        // an idle primary, but only a successfully materialized first request
        // may promise its reuse before BindSequenceCache actually checks it out.
        var primaryClaim = GlmUnifiedPrimarySlotClaim.Prepare(seq);
        if (_options.EnableAdaptiveGlmPrefill)
        {
            var state = _requestPrefill.GetValue(seq, _ => new(_prefill));
            var running = state.Siblings.Select(reference => reference.TryGetTarget(out var live) ? live : null)
                .Where(live => live != null && !live.Status.IsFinished()).Cast<SequenceState>().ToArray();
            PrepareAdaptiveAdmission(seq, running, materialized: true);
            PublishPrimarySlotClaim(primaryClaim);
            RecoverExpertCapacity();
            return;
        }
        if (!seq.MemoryEnvelope!.TryResizeTotal(EstimateMaterialization(seq))) throw new MemoryPressureException("GLM prefix transition cannot preserve its request state/staging peak.");
        PublishPrimarySlotClaim(primaryClaim);
    }
    private void CompletePrefill(SequenceState seq)
    {
        _model!.TrimUnifiedGraphs(onlyPrefill: true);
        _experts?.Trim(_experts.CachedBytes); // Return charged idle file staging; retain CUDA weights.
        _lastGraphShape = null;
        int context = PaddedContext(Math.Max(1, checked(seq.NumComputedTokens + 1)));
        var graph = PreserveReplacement(WithLivePrefillPromises(
            _geometry.GraphPeak(context, 1, false, GraphEntriesWithTransient), except: seq));
        if (!Budget.TryResizeTotals([(_joint!, Charges(graph)), (seq.MemoryEnvelope!, EstimateMaterialization(seq))]))
            throw new MemoryPressureException("GLM prefill transition cannot preserve its shared graph and request state peak.");
        RecoverExpertCapacity();
    }

    private static InferenceMemoryBytes Bytes(IReadOnlyList<MemoryCharge> charges)
        => new(charges.Single(c => c.Pool == HostPool).Bytes, charges.Single(c => c.Pool == DevicePool).Bytes);
    private InferenceMemoryBytes WithLivePrefillPromises(InferenceMemoryBytes graph, SequenceState? except = null)
    {
        if (!_options.EnableAdaptiveGlmPrefill) return graph;
        foreach (var entry in _requestPrefill)
        {
            var sequence = entry.Key;
            if (ReferenceEquals(sequence, except) || sequence.MemoryEnvelope == null
                || sequence.Status.IsFinished() || sequence.NumComputedTokens >= sequence.PromptTokens.Count) continue;
            graph = new(Math.Max(graph.Host, entry.Value.Peak.Host), Math.Max(graph.Device, entry.Value.Peak.Device));
        }
        return graph;
    }

    private IReadOnlyList<MemoryCharge> PrepareAdaptiveAdmission(SequenceState seq,
        IReadOnlyList<SequenceState> running, bool materialized = false)
    {
        if (_native!.CallbackError is { } callback) throw new InvalidOperationException("GLM allocation accounting failed.", callback);
        // Allocate all publication bookkeeping before native or ledger mutation.
        var state = _requestPrefill.GetValue(seq, _ => new(_prefill));
        var siblings = running.Select(live => new WeakReference<SequenceState>(live)).ToArray();
        var heads = running.Select(live => live.NumComputedTokens).ToArray();
        int maximum = Math.Min(_prefill, Math.Max(1, seq.PromptTokens.Count - seq.NumComputedTokens));
        var physical = ObservePhysicalConstraints();
        bool mayReclaimExperts = true;
        GlmUnifiedPrefillCandidate selected = default;
        (InferenceMemoryBytes Joint, InferenceMemoryBytes Request) Estimate(int rows)
        {
            var graphs = GlmUnifiedAdmissionGraphPlan.Create(_geometry, _context, rows,
                seq.PromptTokens.Count, seq.NumComputedTokens, heads, GraphEntriesWithTransient);
            var target = PreserveReplacement(WithLivePrefillPromises(graphs.MaximumPeak, except: seq));
            var request = RequestPeakForRows(seq, rows, materializing: true);
            if (materialized) request = GlmUnifiedPrefillAdmission.Floor(request, Bytes(seq.MemoryEnvelope!.LiveChildCharges));
            return (target, request);
        }
        bool Select()
        {
            physical = physical with { Ledger = Budget.Snapshot() };
            return GlmUnifiedPrefillAdmission.TrySelect(physical.Ledger, physical.Limits,
                Bytes(_joint!.OwnedCharges), materialized ? Bytes(seq.MemoryEnvelope!.OwnedCharges) : default,
                _experts?.BudgetBytes ?? 0,
                mayReclaimExperts ? _expertMinimumBytes : _experts?.BudgetBytes ?? 0, maximum, Estimate, out selected);
        }
        if (!Select())
        {
            _model!.TrimUnifiedGraphs(); _model.TrimUnifiedRetainedSlots();
            _experts?.Trim(_experts.CachedBytes);
            _lastGraphShape = null; physical = ObservePhysicalConstraints();
            if (!Select())
            {
                // This advisory hint has no imputed refund. Re-sample tracked
                // physical capacity before retrying the complete candidate set.
                var smallest = Estimate(1);
                var ownedJoint = Bytes(_joint!.OwnedCharges);
                var ownedRequest = materialized ? Bytes(seq.MemoryEnvelope!.OwnedCharges) : default;
                var minimumGrowth = new InferenceMemoryBytes(
                    checked(smallest.Joint.Host - ownedJoint.Host + smallest.Request.Host - ownedRequest.Host),
                    checked(smallest.Joint.Device - ownedJoint.Device + smallest.Request.Device - ownedRequest.Device));
                var charges = Charges(minimumGrowth);
                long reclaimableExperts = (_experts?.BudgetBytes ?? 0) - _expertMinimumBytes;
                if (physical.Shortfall(HostPool, charges) > 0 && physical.Shortfall(DevicePool, charges) <= reclaimableExperts)
                {
                    _model.TrimUnifiedMappedResidency(); physical = ObservePhysicalConstraints();
                }
                if (!Select()) throw new MemoryPressureException(
                    $"GLM adaptive prefill cannot preserve its complete request, graph and expert phase floors: request={seq.RequestId}; prompt={seq.PromptTokens.Count}; logical-context={_context}; maximum-rows={maximum}; "
                    + physical.Describe(charges));
            }
        }
        if (GlmUnifiedPrefillAdmission.ShouldReclaimRetained(materialized, seq.NumComputedTokens,
            seq.PrefixCacheReusedTokens, selected.Rows, maximum))
        {
            // The completed lookup confirmed no usable state for this request.
            // Only idle terminal holders are eligible; active slots and every
            // pending prefill promise remain charged throughout the retry.
            _model!.TrimUnifiedRetainedSlots(terminalOnly: true); _lastGraphShape = null;
            physical = ObservePhysicalConstraints();
            if (!Select()) throw new MemoryPressureException("GLM adaptive prefill failed its fresh physical check after idle terminal-state reclamation.");
        }
        if (_experts != null && selected.ExpertTargetBytes < _experts.BudgetBytes)
        {
            long previous = _experts.BudgetBytes;
            UnifiedMemoryExpertCapacity.Resize(_experts, _expertEnvelope!, DevicePool, selected.ExpertTargetBytes,
                _expertMinimumBytes, physical.Limits);
            // Native release does not prove equivalent physical availability.
            physical = ObservePhysicalConstraints();
            mayReclaimExperts = false;
            if (!Select()) throw new MemoryPressureException("GLM adaptive prefill failed its fresh physical check after fenced expert reclamation.");
            Console.WriteLine($"[unified-memory] request {seq.RequestId}: fenced expert cache {previous / 1048576.0:F0} -> {_experts.BudgetBytes / 1048576.0:F0} MiB for actual prefill shape");
        }
        var admittedGraph = GlmUnifiedAdmissionGraphPlan.Create(_geometry, _context, selected.Rows,
            seq.PromptTokens.Count, seq.NumComputedTokens, heads, GraphEntriesWithTransient);
        var targets = new List<(BudgetReservation Envelope, IReadOnlyList<MemoryCharge> TotalTarget)>
            { (_joint!, Charges(selected.JointTarget)) };
        if (materialized) targets.Add((seq.MemoryEnvelope!, Charges(selected.RequestTarget)));
        // A new request's separate materialization envelope is acquired next by
        // the scheduler. Include its whole growth in this pre-transaction check.
        if (!materialized)
        {
            var owned = Bytes(_joint!.OwnedCharges);
            var growth = new InferenceMemoryBytes(checked(selected.JointTarget.Host - owned.Host + selected.RequestTarget.Host),
                checked(selected.JointTarget.Device - owned.Device + selected.RequestTarget.Device));
            physical = physical with { Ledger = Budget.Snapshot() };
            if (!physical.Fits(Charges(growth))) throw new MemoryPressureException("GLM adaptive prefill combined materialization peak changed before reservation.");
        }
        if (!Budget.TryResizeTotals(targets, physical.Limits, requirePhysicalFit: true))
            throw new MemoryPressureException("GLM adaptive prefill could not atomically admit its actual graph/request targets.");
        state.Cap = selected.Rows; state.Siblings = siblings;
        state.Peak = admittedGraph.PrefillPeak;
        if (selected.Rows < maximum)
            Console.WriteLine($"[unified-memory] request {seq.RequestId}: prefill {maximum} -> {selected.Rows} rows for actual prompt/joint shapes and immutable expert floor");
        return physical.Limits;
    }

    private void RecoverExpertCapacity()
    {
        if (!_options.EnableAdaptiveGlmPrefill || _experts == null) return;
        if (_native!.CallbackError is { } callback) throw new InvalidOperationException("GLM optional expert recovery is disabled after an accounting failure.", callback);
        var physical = ObservePhysicalConstraints();
        var device = physical.Ledger.Single(p => p.Pool == DevicePool);
        long limit = Math.Min(device.Capacity, physical.Limits.Single(c => c.Pool == DevicePool).Bytes);
        long slack = Math.Max(0, limit - device.Committed - device.Reserved);
        long increase = Math.Max(0, slack - (128L << 20));
        long previous = _experts.BudgetBytes;
        long target = Math.Min(_experts.MaximumBudgetBytes, checked(previous + increase));
        target = Math.Max(previous, target / (1L << 20) * (1L << 20));
        if (target - previous < (512L << 20)) return;
        if (UnifiedMemoryExpertCapacity.Resize(_experts, _expertEnvelope!, DevicePool, target, _expertMinimumBytes, physical.Limits))
            Console.WriteLine($"[unified-memory] released request workspace: expert-cache ceiling {previous / 1048576.0:F0} -> {target / 1048576.0:F0} MiB within live reservations; retained tables grow only for an admitted route/shape requirement");
    }

    private void RecoverReleasedCapacity(IReadOnlyList<SequenceState> running)
    {
        if (!_options.EnableAdaptiveGlmPrefill) return;
        if (_native!.CallbackError is { } callback)
            throw new InvalidOperationException("GLM released-capacity recovery is disabled after an accounting failure.", callback);
        // The engine has fenced model release and disposed this request's root.
        // Invalidate only managed shape bookkeeping: native decode graphs may
        // remain reusable, but their unused large-prefill credit must not stay
        // hidden until a 256-token graph boundary happens to change the shape.
        _lastGraphShape = null;
        InferenceMemoryBytes PrefillPeak(SequenceState sequence)
            => _requestPrefill.TryGetValue(sequence, out var state) ? state.Peak : default;
        var plan = GlmUnifiedReleaseGraphPlan.Create(_geometry, _context, running, PrefillPeak,
            default, GraphEntriesWithTransient);
        if (plan.RunningRows == 0)
            _model!.TrimUnifiedGraphs(); // All graphs are idle; terminal KV slots stay live.
        else if (!plan.HasRunningPrefill)
            _model!.TrimUnifiedGraphs(onlyPrefill: true); // No surviving request needs a PP graph.
        plan = GlmUnifiedReleaseGraphPlan.Create(_geometry, _context, running, PrefillPeak,
            default, GraphEntriesWithTransient);
        var target = plan.RunningRows == 0
            ? GlmUnifiedPrefillAdmission.Floor(default, Bytes(_joint!.LiveChildCharges))
            : PreserveReplacement(WithLivePrefillPromises(plan.Target));
        if (!_joint.TryResizeTotal(Charges(target)))
            throw new MemoryPressureException("GLM sequence release could not preserve the remaining graph and physical-child floors.");
        // Cache recovery sees the corrected shared ownership and a fresh
        // physical sample only after graph retirement/credit shrink complete.
        RecoverExpertCapacity();
    }

    private int PaddedContext(int tokens)
        => Math.Min(_context, checked((Math.Max(1, tokens) + 255) / 256 * 256));
    private InferenceMemoryBytes PreserveReplacement(InferenceMemoryBytes twoGraphs)
    {
        // The native cache retains one graph while allocating its replacement.
        // An earlier wider PP graph can be larger than this next decode graph.
        return GlmUnifiedPrefillAdmission.PreserveReplacement(twoGraphs,
            Bytes(_joint!.LiveChildCharges), GraphEntriesWithTransient);
    }
    private void PrepareGraph(GlmUnifiedGraphShape shape)
    {
        int rowLimit = shape.Batched ? _options.MaximumRunningRequests
            : _executingSequence is { } sequence ? AdmittedPrefillLimit(sequence) : _prefill;
        if (shape.Rows <= 0 || shape.Rows > rowLimit || shape.ContextTokens <= 0 || shape.ContextTokens > _context)
            throw new MemoryPressureException("GLM graph shape exceeds the admitted execution geometry.");
        shape = shape with { ContextTokens = PaddedContext(shape.ContextTokens) };
        if (_lastGraphShape == shape) return;
        var target = PreserveReplacement(WithLivePrefillPromises(
            _geometry.GraphPeak(shape.ContextTokens, shape.Rows, shape.Batched, GraphEntriesWithTransient)));
        bool Resize(IReadOnlyList<MemoryCharge>? limits = null) => _executingRequest is { } request
            ? Budget.TryResizeTotals([(_joint!, Charges(target)), (request, request.OwnedCharges)], limits)
            : _joint!.TryResizeTotal(Charges(target), limits);
        var owned = _joint!.OwnedCharges.ToDictionary(c => c.Pool, c => c.Bytes);
        bool grows = target.Host > owned[HostPool] || target.Device > owned[DevicePool];
        if (!Resize(grows ? PhysicalConstraints() : null))
        {
            _model!.TrimUnifiedGraphs(); _model.TrimUnifiedRetainedSlots(); _lastGraphShape = null;
            target = PreserveReplacement(WithLivePrefillPromises(
                _geometry.GraphPeak(shape.ContextTokens, shape.Rows, shape.Batched, GraphEntriesWithTransient)));
            var limits = PhysicalConstraints();
            if (!Resize(limits))
            {
                // File mappings have no anonymous payload ticket. Their pages
                // remain reclaimable, but a very large CPU expert scan can
                // consume currently observed RAM before its next graph grows.
                // Ask this quiescent model to release only those read-only
                // mapping pages, then authorize growth from a fresh observation.
                bool hinted = _model.TrimUnifiedMappedResidency();
                limits = PhysicalConstraints();
                if (!Resize(limits))
                {
                    string Describe(MemoryCharge charge) => $"{charge.Pool}={charge.Bytes}";
                    string ledger = string.Join(", ", Budget.Snapshot().Select(p =>
                        $"{p.Pool}: reserved={p.Reserved}, committed={p.Committed}, capacity={p.Capacity}"));
                    throw new MemoryPressureException($"GLM next graph cannot reserve its replacement peak before dispatch: " +
                        $"shape={shape}; target=[{string.Join(", ", Charges(target).Select(Describe))}]; " +
                        $"limits=[{string.Join(", ", limits.Select(Describe))}]; {ledger}; mapped-trim-hint={hinted}.");
                }
            }
        }
        _lastGraphShape = shape;
    }
    private IDisposable EnterSerial(SequenceState seq)
    {
        var request = seq.MemoryEnvelope ?? throw new InvalidOperationException("GLM execution requires an admitted envelope.");
        if (_executingRequest != null) throw new InvalidOperationException("GLM request allocation scopes cannot nest.");
        var scope = EnterAllocations(request);
        try
        {
            var execution = new SerialExecution(this, scope);
            _executingRequest = request;
            _executingSequence = seq;
            return execution;
        }
        catch { scope.Dispose(); throw; }
    }
    private sealed class SerialExecution(GlmUnifiedMemoryModelSession owner, IDisposable scope) : IDisposable
    {
        public void Dispose() { try { scope.Dispose(); } finally { owner._executingRequest = null; owner._executingSequence = null; } }
    }
    private bool Fits(IReadOnlyList<MemoryCharge> bytes, IReadOnlyList<MemoryCharge> limits)
    {
        var owned = Budget.Snapshot().ToDictionary(p => p.Pool);
        return bytes.All(c => (decimal)c.Bytes + owned[c.Pool].Reserved + owned[c.Pool].Committed <= limits.Single(p => p.Pool == c.Pool).Bytes);
    }
    private GlmUnifiedPhysicalObservation ObservePhysicalConstraints()
        => GlmUnifiedPhysicalObservation.Create(InferenceHardwareMemory.CaptureCuda(), Budget.Snapshot(),
            _hostCeiling, _deviceCeiling, _hostHeadroom, _deviceHeadroom);
    private MemoryCharge[] PhysicalConstraints() => ObservePhysicalConstraints().Limits;
    private long PrefixCloneSpareBytes()
        => _executingRequest?.Charges.Single(c => c.Pool == DevicePool).Bytes
            ?? Budget.Snapshot().Single(p => p.Pool == DevicePool).Available;
    private IDisposable? EnterPrefixCapture(int tokens)
    {
        var sequence = _executingSequence;
        if (_effectivePrefixCopies <= 0 || sequence == null || tokens <= 0
            || !sequence.PublicCheckpointBoundaries.Contains(tokens)
            || _model!.RetainedCopiedSlotCount >= _effectivePrefixCopies + 1)
            return null;
        var request = _executingRequest ?? throw new InvalidOperationException("GLM prefix capture requires its executing request envelope.");
        // Prefix materialization runs before binding the actual native slot.
        // An idle primary may therefore have satisfied the slot forecast without
        // consuming this request's allowance. Settle that unused Device credit
        // now, preserving physical children and all Host/staging and joint-graph
        // promises. This is a ledger adjustment, not a physical memory refund.
        if (!HasNativeSlot(sequence)) return null;
        long previousDevice = request.OwnedCharges.Single(c => c.Pool == DevicePool).Bytes;
        bool settled = OwnedAllocationScope.ReleaseUnusedPoolReservation(request, DevicePool);
        long releasedDevice = previousDevice - request.OwnedCharges.Single(c => c.Pool == DevicePool).Bytes;
        if (settled && _options.EnableAdaptiveGlmPrefill && tokens < sequence.PromptTokens.Count)
        {
            // The completed forward has advanced the sequence before this
            // checkpoint. Its old wide-prefill promise no longer describes the
            // remaining tail. Reuse the ordinary atomic admission transaction,
            // retaining physical children, replacement graphs and every other
            // pending prefill promise. Snapshot the current admitted cohort:
            // siblings recorded at this request's start can omit newer decoders.
            if (sequence.NumComputedTokens != tokens)
                throw new InvalidOperationException("GLM checkpoint state does not match its admitted request head.");
            var siblings = UnifiedMemoryAdmissionCohort.SnapshotOtherAdmitted(
                _requestPrefill.Select(entry => entry.Key), sequence);
            try { _ = PrepareAdaptiveAdmission(sequence, siblings, materialized: true); }
            catch (MemoryPressureException) { return null; } // A discretionary cache must not reject this request.
        }
        var charges = Charges(new(0, _geometry.SlotBytes(_context)));
        var physical = ObservePhysicalConstraints();
        var envelope = settled ? OwnedAllocationScope.TryReserveOptional(Budget, charges, physical.Limits) : null;
        try
        {
            if (System.Threading.Interlocked.Increment(ref _prefixCaptureDiagnostics) <= 8)
            {
                Console.WriteLine($"[unified-memory] public prefix capture: boundary={tokens}, prompt={sequence.PromptTokens.Count}, remaining={sequence.PromptTokens.Count - tokens}, " +
                    $"reservation={(envelope != null ? "admitted" : settled ? "unavailable" : "unsettled")}, unused-device-credit-released={releasedDevice}, " +
                    $"request-owned={Bytes(request.OwnedCharges)}, request-children={Bytes(request.LiveChildCharges)}, " +
                    $"joint-owned={Bytes(_joint!.OwnedCharges)}, joint-children={Bytes(_joint.LiveChildCharges)}, " +
                    $"expert-ceiling={_experts?.BudgetBytes ?? 0}; before-optional-admission: {physical.Describe(charges)}");
            }
            if (envelope == null) return null; // Optional cache misses must not reject the user request.
            // Closing the root refunds slack, while each actual native child
            // keeps its checkpoint charged until its physical buffer is freed.
            return OwnedAllocationScope.Enter(envelope, () => _native!.EnterAllocationOwner(envelope));
        }
        catch { envelope?.Dispose(); throw; }
    }
    private IDisposable EnterVisionPreparation(GlmVisionPreparationPlan plan)
    {
        var limits = PhysicalConstraints();
        var needed = Charges(new(plan.HostPeakBytes, plan.DevicePeakBytes));
        if (!Fits(needed, limits)) { _model!.TrimUnifiedGraphs(); _model.TrimUnifiedRetainedSlots(); limits = PhysicalConstraints(); }
        var envelope = Budget.TryReserve(needed, limits) ?? throw new MemoryPressureException("GLM image preparation exceeds the shared budget.");
        try { return OwnedAllocationScope.Enter(envelope, () => EnterAllocations(envelope)); }
        catch { envelope.Dispose(); throw; }
    }
    private IDisposable EnterAllocations(BudgetReservation envelope)
    {
        IDisposable native = _native!.EnterExecution(envelope);
        IDisposable? host = null;
        try { host = _host!.EnterExecution(envelope); return new OwnedExecution(host, native); }
        catch { try { host?.Dispose(); } finally { native.Dispose(); } throw; }
    }
    private sealed class OwnedExecution(IDisposable first, IDisposable second) : IDisposable
    {
        public void Dispose() { try { first.Dispose(); } finally { second.Dispose(); } }
    }
    public void Dispose()
    {
        if (_disposed) return;
        _model?.Dispose(); _model = null;
        _experts?.Dispose(); _experts = null;
        _native?.SetSharedExpertEnvelope(null);
        _expertEnvelope?.Dispose(); _expertEnvelope = null;
        _native?.SetSharedJointEnvelope(null);
        _joint?.Dispose(); _joint = null;
        _host?.Dispose(); _host = null;
        _native?.Dispose(); _native = null;
        _disposed = true;
    }
}
