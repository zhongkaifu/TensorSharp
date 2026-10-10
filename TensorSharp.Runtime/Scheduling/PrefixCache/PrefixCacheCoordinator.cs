// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using Microsoft.Extensions.Logging;
using TensorSharp.Runtime.Paged;

namespace TensorSharp.Runtime.Scheduling.PrefixCache;

/// <summary>The engine worker's single owner of radix keys, reusable state and model payloads.</summary>
internal sealed partial class PrefixCacheCoordinator : IPrefixPayloadSink, IPayloadValidator
{
    private static long s_engineSerial;
    internal readonly PrefixTree _tree;
    private readonly BlockPool _pool;
    private readonly IModelArchitecture _model;
    private readonly ContinuousBatchScheduler _scheduler;
    private readonly IPrefixCacheModel _cacheModel;
    private readonly ILogger _logger;
    private readonly Dictionary<SequenceState, RequestKey> _requests = new();
    private readonly Queue<string> _invalidated = new();
    private readonly HashSet<long> _storeMisses = new();
    private readonly MatchPlan _plan = new();
    private string? _primaryKey;
    private IPrefixCheckpointStore? _checkpointStore;

    private sealed record RequestKey(KeyRope Key, int Scope, MediaSpanRecord[] Spans);

    internal PrefixCacheCoordinator(IModelArchitecture model, BlockPool pool,
        ContinuousBatchScheduler scheduler, PrefixCacheCapabilities capabilities, ILogger logger)
    {
        _model = model;
        _pool = pool;
        _scheduler = scheduler;
        _cacheModel = (IPrefixCacheModel)model;
        _logger = logger;
        var options = ExecutionOptions.FromEnvironment();
        bool batchedEnabled = !options.BatchedPathDisabled && scheduler.Config.KvSnapshots == null;
        // A materialized holder is only readable through the fused per-request route, a paged end state only
        // through the batched route. Without that route (--no-continuous-batching, TS_PER_SEQ_FUSED=0, or a model
        // that lacks it) no placeholder blocks may be adopted for it.
        bool endStateRoute = capabilities.PagedEndStates
            ? batchedEnabled && model is IBatchedPagedModel { BatchedForwardAvailable: true }
            : batchedEnabled && options.PerSeqFusedEnabled
              && model is IBatchedPagedModel { SupportsPerSequenceFusedForward: true };
        if (!endStateRoute)
            capabilities = capabilities with
            {
                EndState = EndStateSupport.None, CanCaptureCopy = false,
                AdoptPrimaryOnDisplacement = false, DeferPrimaryConversion = false, PagedEndStates = false,
            };
        _tree = new PrefixTree(new PrefixTreeOptions
        {
            Capabilities = capabilities,
            EngineSerial = Interlocked.Increment(ref s_engineSerial),
            BlockSize = pool.BlockSize,
            ContextLength = model.MaxContextLength > 0 ? model.MaxContextLength : int.MaxValue,
            PublicMax = options.PrefixCheckpointBudget,
            ScopedEndStateLeavesMax = options.RetainedFusedCacheBudgetFor(scheduler.Config.MaxNumRunningSequences),
            PageHost = new PageHost(pool),
            PageHostBytes = InferenceEngine.ComputeBlockByteSize(model, pool.BlockSize),
            PoolPagesCap = pool.NumBlocks,
            BatchedPagedEnabled = batchedEnabled,
            PayloadValidator = this,
            QuerySpareBytes = _cacheModel.QuerySpareBytes,
            HostRamBytes = GC.GetGCMemoryInfo().TotalAvailableMemoryBytes,
        });
        _cacheModel.AttachPrefixCache(this);
    }

    internal PrefixTree Tree => _tree;
    internal int RequestCount => _requests.Count;
    internal bool PrimaryAvailable { get; set; }
    internal bool RequiresSoleAdmission { get; private set; }
    private static bool RetentionEnabled
        => ExecutionOptions.FromEnvironment() is { RetainedFusedCacheBudget: null or > 0 };
    private static bool PublicCheckpointsEnabled
        => ExecutionOptions.FromEnvironment() is { PrefixCheckpointBudget: > 0 };
    internal bool CheckpointsSupported => _tree.Caps.CanCaptureCopy && (PublicCheckpointsEnabled || RetentionEnabled);
    internal bool PublicCheckpointsSupported => PublicCheckpointsEnabled && CheckpointsSupported;
    internal string? LastSource { get; private set; }

    /// <summary>The last admission plan, for the TS_CB_DEBUG trace.</summary>
    internal string LastPlanDescription => _plan.ToString();
    internal int LastBlockedByScope { get; private set; }
    /// <summary>Why the last <see cref="ComputeReusablePrefix"/> reused nothing, in words, or null when it
    /// reused something or nothing cached matched at all.</summary>
    internal string? LastDeclineReason { get; private set; }
    internal IPrefixCheckpointStore? CheckpointStore
    {
        get => Volatile.Read(ref _checkpointStore);
        set => Volatile.Write(ref _checkpointStore, value);
    }

    internal void EnsureRequest(SequenceState sequence)
    {
        if (_requests.ContainsKey(sequence)) return;
        var spans = new MediaSpanRecord[sequence.MediaSpans.Count];
        for (int i = 0; i < spans.Length; i++)
        {
            var span = sequence.MediaSpans[i];
            spans[i] = MediaSpanRecord.FromContentId(span.Start, span.End, span.ContentId);
        }
        ScopeId scope;
        if (sequence.CacheScope is null)
            scope = ScopeId.NewFresh();
        else
        {
            byte[] digest = SHA256.HashData(Encoding.UTF8.GetBytes(sequence.CacheScope));
            scope = ScopeId.FromHex(Convert.ToHexString(digest.AsSpan(0, 16)));
        }
        int scopeIx = _tree.InternScope(scope, sequence.CacheScope is null ? ScopeKind.Unscoped : ScopeKind.Session);
        var key = RadixKeyBuilder.BuildPrompt(sequence.PromptTokens, spans, _tree.KeyPool);
        _requests.Add(sequence, new RequestKey(key, scopeIx, spans));
        _tree.NoteRequests(scopeIx, 1, 0);
    }

    internal KeyRope GetKey(SequenceState sequence)
    {
        EnsureRequest(sequence);
        KeyRope key = _requests[sequence].Key;
        for (int i = key.Length; i < sequence.NumTotalTokens; i++)
            key.Append(KeyElem.Text(sequence.TokenAt(i)), _tree.KeyPool);
        return key;
    }

    internal int GetScope(SequenceState sequence) { EnsureRequest(sequence); return _requests[sequence].Scope; }
    internal MediaSpanRecord[] GetSpans(SequenceState sequence) { EnsureRequest(sequence); return _requests[sequence].Spans; }

    internal MatchRequest BuildRequest(SequenceState sequence, ExpectedRoute route)
    {
        KeyRope key = GetKey(sequence);
        // At least one prompt token is always forwarded, for the logits (K6). A family may need more: DeepSeek
        // V4.1 records the rewind checkpoint its next thinking turn depends on only at the end of a
        // MULTI-token forward, so a regenerate that forwarded a single token left none behind.
        int limit = sequence.PromptTokens.Count - Math.Max(1, _tree.Caps.MinTailPrefillTokens);
        if (sequence.CacheBreakpoints is not null) limit = Math.Min(limit, sequence.CacheBreakpointLimit);
        // Some vision encoders require the original prefill boundary to preserve positions.
        if (!_model.CanPrefillMediaAfterReusedPrefix(sequence.PromptTokens.Count) && sequence.MediaSpans.Count > 0)
            limit = Math.Min(limit, CheckpointsSupported ? sequence.SharedPrefixTokens : 0);
        return new MatchRequest(key, sequence.PromptTokens.Count, GetScope(sequence), sequence.SharedPrefixTokens,
            Math.Max(0, limit), GetSpans(sequence), route, PrimaryAvailable);
    }

    private ExpectedRoute PredictRoute()
    {
        var options = ExecutionOptions.FromEnvironment();
        if (_scheduler.Config.KvSnapshots == null && !options.BatchedPathDisabled && options.PerSeqFusedEnabled
            && _model is IBatchedPagedModel { SupportsPerSequenceFusedForward: true })
            return ExpectedRoute.PerSequenceFused;
        return ExpectedRoute.Primary;
    }

    internal int ComputeReusablePrefix(SequenceState sequence)
    {
        Drain();
        RequiresSoleAdmission = false;
        LastSource = null;
        TryRestoreCheckpoint(sequence);
        var request = BuildRequest(sequence, PredictRoute());
        _tree.Plan(request, _plan);
        if (request.Route == ExpectedRoute.Primary
            && _plan.Mode is MaterializeMode.CloneEndState or MaterializeMode.DonateEndState
                or MaterializeMode.ConvertPrimaryThenClone)
            _plan.Reset();
        if (!_plan.HasReuse && PredictRoute() == ExpectedRoute.Primary
            && !ExecutionOptions.FromEnvironment().BatchedPathDisabled
            && _model is IBatchedPagedModel { BatchedForwardAvailable: true } paged
            && (sequence.MediaSpans.Count == 0 || paged.SupportsBatchedMultimodal))
        {
            request = BuildRequest(sequence, ExpectedRoute.BatchedPaged);
            _tree.Plan(request, _plan);
        }
        // A holder is read by the per-request route only; a paged end state by the batched route only.
        if (PredictRoute() != ExpectedRoute.PerSequenceFused
            && _plan.Mode is MaterializeMode.CloneEndState or MaterializeMode.DonateEndState
                or MaterializeMode.ConvertPrimaryThenClone
            && !(request.Route == ExpectedRoute.BatchedPaged && IsPagedEndState(_plan)))
            _plan.Reset();
        LastBlockedByScope = _plan.BlockedByScope;
        LastDeclineReason = _plan.HasReuse ? null : DescribeDecline(_plan);
        // The executor never continues the live cache for a request with explicit cache breakpoints
        // (BatchExecutor.EnsureOwnership), so do not promise it one: admission announced the reuse and
        // execution took it back.
        if (_plan.Mode == MaterializeMode.KeepPrimary && sequence.CacheBreakpoints is not null)
        {
            _plan.Reset();
            LastDeclineReason = "the live cache is not continued for a request with explicit cache breakpoints";
        }
        return _plan.Length;
    }

    /// <summary>The first specific reason a plan reused nothing: a rewind of the cached conversation, then
    /// the live cache, a retained state, pages. Sources that were simply absent or lost to a longer one say
    /// nothing. Before it existed the log said only "0/N tokens", which is all a user saw when every
    /// DeepSeek V4.1 thinking turn re-prefilled its whole prompt.</summary>
    private static string? DescribeDecline(MatchPlan plan)
        => Describe("rewinding the cached conversation", plan.TruncationDecline)
           ?? Describe("the live cache", plan.PrimaryDecline)
           ?? Describe("the retained state", plan.EndStateDecline)
           ?? Describe("the cached pages", plan.PageDecline);

    private static string? Describe(string source, SourceDecline decline)
    {
        string? why = decline switch
        {
            SourceDecline.NotPermitted => "belongs to another conversation",
            SourceDecline.Clamped => "is cut short by a media span, a breakpoint or the rewind cap",
            SourceDecline.PrimaryBusy => "is busy with other requests",
            SourceDecline.PrimaryClaimed => "is already claimed this step",
            SourceDecline.ModelRefused => "is declined by the model",
            SourceDecline.DonateOnlyShared => "would give up a state kept for later requests",
            SourceDecline.RouteUnreadable => "is unreadable on this request's route",
            SourceDecline.CloneCost => "is too short to clone",
            SourceDecline.MmThreshold => "is below the media reuse threshold",
            _ => null,
        };
        return why == null ? null : $"{source} {why}";
    }

    /// <summary>A new request can wait for a longer exact public checkpoint that this step's
    /// producer is still computing. This is only a scheduling hint: admission must
    /// subsequently match and materialize the published payload through the tree.</summary>
    internal bool CanSharePendingPublicCheckpoint(SequenceState sequence, SequenceState producer, int reusableLength = 0)
    {
        if (!PublicCheckpointsEnabled || !CheckpointsSupported
            || PredictRoute() != ExpectedRoute.PerSequenceFused
            || sequence.SharedPrefixTokens <= 0 || sequence.FirstScheduledAt is not null || sequence.NumComputedTokens != 0
            || sequence.PrefixCacheReusedTokens != 0 || ReferenceEquals(sequence, producer)
            || producer.Status != SequenceStatus.Running)
            return false;

        foreach (int length in producer.PublicCheckpointBoundaries)
        {
            if (length <= reusableLength || length <= producer.NumComputedTokens
                || length > sequence.SharedPrefixTokens || !producer.IsPublicCheckpointBoundary(length)) continue;
            if (MatchesPublicPrefix(sequence, producer, length)) return true;
        }
        return false;
    }

    private bool MatchesPublicPrefix(SequenceState sequence, SequenceState producer, int length)
    {
        var request = BuildRequest(sequence, ExpectedRoute.PerSequenceFused);
        if (_tree.Rules.ClampLength(length, request) != length) return false;
        KeyRope source = GetKey(producer);
        for (int offset = 0; offset < length;)
        {
            ReadOnlySpan<long> part = request.Key.Segment(offset, length - offset);
            if (!part.SequenceEqual(source.Segment(offset, part.Length))) return false;
            offset += part.Length;
        }

        // Media key elements are abbreviated hashes. Compare full identities and
        // span boundaries as well; equal placeholder tokens alone are insufficient.
        MediaSpanRecord[] producerSpans = GetSpans(producer);
        int index = 0;
        foreach (MediaSpanRecord span in request.Spans)
        {
            if (span.Start >= length) break;
            if (span.End > length || index >= producerSpans.Length || producerSpans[index] != span)
                return false;
            index++;
        }
        return index >= producerSpans.Length || producerSpans[index].Start >= length;
    }

    internal bool TryAdopt(SequenceState sequence, int length, Action<SequenceState, int> pendingTruncation)
    {
        if (length <= 0 || sequence.BlockTable.NumBlocks != 0 || ComputeReusablePrefix(sequence) != length)
            return false;
        LockReceipt receipt = _tree.Acquire(_plan);
        try
        {
            if (_plan.Kind == CandidateKind.Pages)
            {
                if (!TryAdoptPages(sequence, _plan, receipt)) return false;
                LastSource = "radix pages";
                return true;
            }
            RadixNode node = _plan.PayloadNode!;
            EndStatePayload payload = node.EndState!;
            if (_pagedEndStates.ContainsKey(payload.Key)) return TryAdoptPagedEndState(sequence, node, payload, length);
            int count = (int)(((long)length + _pool.BlockSize - 1) / _pool.BlockSize);
            sequence.BlockTable.EnsureBlockCapacity(count);
            EnsureFreePages(count);
            KvBlock[]? blocks = _pool.AllocateNew(count);
            if (blocks is null) return false;
            bool transferred = false;
            try
            {
                bool primary = _plan.Mode == MaterializeMode.KeepPrimary;
                if (primary)
                {
                    RequiresSoleAdmission = true;
                    sequence.UsesLiveCacheContinuation = true;
                    _tree.DetachEndState(node, ReleaseReason.Rollback, enqueue: false);
                    _primaryKey = null;
                }
                else
                {
                    bool donate = _plan.Mode == MaterializeMode.DonateEndState;
                    if (!donate && _plan.Mode != MaterializeMode.CloneEndState) return false;
                    if (!donate && !ReserveClone(payload.Key, length)) return false;
                    var materialize = new MaterializeRequest(donate ? MaterializeOp.Donate : MaterializeOp.Clone,
                        payload.Key, sequence.RequestId, payload.Footprint.Tokens, length);
                    if (!_cacheModel.TryMaterialize(materialize)) return false;
                    if (donate) _tree.DetachEndState(node, ReleaseReason.Rollback, enqueue: false);
                    if (length < payload.Footprint.Tokens) pendingTruncation(sequence, length);
                }
                foreach (KvBlock block in blocks) sequence.BlockTable.AppendBlock(block);
                sequence.SetComputedTokensForPrefixAdoption(length);
                sequence.PrefixCacheReusedTokens = length;
                sequence.PrefixCheckpointTaken = length >= sequence.SharedPrefixTokens && sequence.SharedPrefixTokens > 0;
                transferred = true;
                LastSource = primary ? "radix primary cache" : "radix end state";
                return true;
            }
            finally { if (!transferred) _pool.Free(blocks); }
        }
        finally { _tree.Release(ref receipt); Drain(); }
    }

    private bool ReserveClone(string key, int tokens)
    {
        ResourceVector bytes = _cacheModel.EstimateCloneBytes(key, tokens);
        for (int i = 1; i < ResourceVector.ClassCount; i++)
        {
            var cls = (ResourceClass)i;
            long spare = _cacheModel.QuerySpareBytes(cls);
            if (spare >= 0 && bytes[cls] > spare)
            {
                if (!_tree.Evict(cls, bytes[cls] - spare, ReleaseReason.Pressure, EvictionTier.ScopeNewest)) return false;
            }
        }
        Drain();
        return true;
    }

    internal void CaptureCheckpoint(SequenceState sequence)
    {
        if (!CheckpointsSupported) return;
        int length = sequence.NumComputedTokens;
        bool publicBoundary = sequence.IsPublicCheckpointBoundary(length);
        // Recurrent caches cannot rewind the generated tail. Keep the exact prompt
        // boundary in its private scope so retries and branches can resume there.
        bool promptBoundary = length == sequence.PromptTokens.Count && sequence.CacheScope is not null;
        bool explicitBoundary = false;
        if (sequence.CacheBreakpoints is not null)
        {
            foreach (int boundary in sequence.CacheBreakpoints)
                if (boundary == length) { explicitBoundary = true; break; }
            if (!explicitBoundary) return;
        }
        if (length <= 0 || (!publicBoundary && !explicitBoundary && !promptBoundary)) return;
        if (length <= sequence.SharedPrefixTokens ? !PublicCheckpointsEnabled : !RetentionEnabled) return;
        if (publicBoundary && length == sequence.SharedPrefixTokens) sequence.PrefixCheckpointTaken = true;
        Drain();
        RadixNode node = _tree.Insert(GetKey(sequence), length, GetScope(sequence),
            publicBoundary ? length : sequence.SharedPrefixTokens,
            explicitBoundary ? NodeFlags.EndsAtBreakpoint : promptBoundary ? NodeFlags.PromptEnd : NodeFlags.None,
            GetSpans(sequence));
        // A media-key collision can stop insertion before this exact state boundary.
        // Never associate the full state with a shallower prefix (including Root).
        if (node.Depth != length) return;
        if (node.EndState is not null) return;
        // Releasing a deeper retained state may otherwise collect this still-empty
        // ancestor and recycle its object before the capture retry publishes it.
        LockReceipt publication = _tree.AcquirePath(node);
        try
        {
            string key = _tree.MintKey();
            bool captured = _cacheModel.TryCaptureCopy(sequence.RequestId, key, out var footprint);
            if (!captured && ReleaseOldestScopedPayload())
                captured = _cacheModel.TryCaptureCopy(sequence.RequestId, key, out footprint);
            if (!captured) return;
            if (Publish(node, key, footprint, length, PayloadOrigin.CaptureCopy) && publicBoundary)
                SaveCheckpoint(sequence, key, length);
        }
        finally { _tree.Release(ref publication); }
        Trim();
    }

    internal bool RetainFinished(SequenceState sequence, bool primary)
    {
        Drain();
        int length = Math.Min(sequence.NumComputedTokens, sequence.NumTotalTokens);
        if (length < _tree.Caps.MinRetainTokens || sequence.CacheBreakpoints is not null) return false;
        KeyRope key = GetKey(sequence);
        RadixNode node = _tree.Insert(key, length, GetScope(sequence), sequence.SharedPrefixTokens,
            NodeFlags.None, GetSpans(sequence));
        if (node.Depth != length) return false;
        if (node.EndState is not null) return false;
        LockReceipt publication = _tree.AcquirePath(node);
        try
        {
            string payloadKey = _tree.MintKey();
            if (primary && _tree.Caps.DeferPrimaryConversion)
            {
                RegisterPrimary(node, payloadKey, length);
                return false; // The live primary still belongs to the executor, not a retained holder.
            }
            bool captured = RetentionEnabled && _tree.Caps.EndState != EndStateSupport.None
                && _cacheModel.TryCaptureDonate(sequence.RequestId, payloadKey, length, out _);
            if (!captured && RetentionEnabled && _tree.Caps.EndState != EndStateSupport.None
                && _model is IBatchedPagedModel holder && holder.HasFusedSequenceCache(sequence.RequestId)
                && ReleaseOldestScopedPayload())
                captured = _cacheModel.TryCaptureDonate(sequence.RequestId, payloadKey, length, out _);
            PayloadOrigin origin = PayloadOrigin.Donation;
            bool conversionFailed = false;
            if (!captured && RetentionEnabled && primary && _tree.Caps.AdoptPrimaryOnDisplacement)
            {
                captured = _cacheModel.TryConvertPrimary(payloadKey, length, out _);
                if (!captured && ReleaseOldestScopedPayload())
                    captured = _cacheModel.TryConvertPrimary(payloadKey, length, out _);
                origin = PayloadOrigin.PrimaryConversion;
                conversionFailed = !captured;
            }
            if (captured)
            {
                bool published = Publish(node, payloadKey, _cacheModel.MeasureEndState(payloadKey), length, origin);
                Trim();
                return published;
            }
            // A conversion can fail AFTER adopting the primary: the holder adapter then releases what it
            // adopted, and DeepSeek V4.1 resets a released slot to position 0. Advertising that primary let an
            // exact continuation keep it (no rewind, so nothing asks the model) and decode from an empty cache.
            // Keep it only while the model still reports this sequence's length.
            if (conversionFailed && _cacheModel is IPrefixCacheModelDiagnostics diagnostics
                && diagnostics.PrimaryCacheLength != length)
            {
                InvalidatePrimary();
                _tree.CollectIfEmpty(node);
                return false;
            }
            if (primary && _tree.Caps.PrimaryResident)
                RegisterPrimary(node, payloadKey, length);
            else _tree.CollectIfEmpty(node);
            return false;
        }
        finally { _tree.Release(ref publication); }
    }

    private void RegisterPrimary(RadixNode node, string payloadKey, int length)
    {
        InvalidatePrimary();
        var attached = _tree.AttachEndState(node, new EndStatePayload
        {
            Key = payloadKey, Kind = EndStateKind.PrimaryResident,
            Footprint = new PayloadFootprint(length, length, default, 0),
        });
        if (attached is AttachResult.Attached or AttachResult.Revived) _primaryKey = payloadKey;
        else _tree.CollectIfEmpty(node);
    }

    private bool Publish(RadixNode node, string key, PayloadFootprint footprint, int length, PayloadOrigin origin)
    {
        if (footprint.Tokens != length)
        {
            _tree.Reclaim.Enqueue(key, ReleaseReason.Invalidated, footprint.Bytes);
            _tree.CollectIfEmpty(node);
            return false;
        }
        var result = _tree.AttachEndState(node, new EndStatePayload
        {
            Key = key, Kind = _tree.Caps.Class == FamilyClass.N ? EndStateKind.NativeSlot : EndStateKind.Holder,
            Origin = origin, Footprint = footprint, DeviceDirty = origin != PayloadOrigin.CaptureCopy,
        });
        if (result == AttachResult.Refused)
        {
            _tree.Reclaim.Enqueue(key, ReleaseReason.Invalidated, footprint.Bytes);
            _tree.CollectIfEmpty(node);
        }
        return result is AttachResult.Attached or AttachResult.Revived;
    }

    private bool ReleaseOldestScopedPayload()
    {
        RadixNode? victim = null;
        foreach (string key in _tree.PayloadKeys)
        {
            if (!_tree.TryGetNodeByKey(key, out var node) || node.ScopeIx == 0
                || node.EndState!.Kind == EndStateKind.PrimaryResident
                || node.StateLockRef != 0 || node.PinRef != 0 || node.IsDonationPending) continue;
            if (victim is null || node.LastAccess < victim.LastAccess) victim = node;
        }
        if (victim is null) return false;
        _tree.DetachEndState(victim, ReleaseReason.Evicted);
        _tree.CollectIfEmpty(victim);
        Drain();
        return true;
    }

    internal void InvalidatePrimary()
    {
        if (_primaryKey is null) return;
        _tree.InvalidatePayload(_primaryKey);
        _primaryKey = null;
        RetireEmptyScopes();
    }

    /// <summary>Before execution overwrites an idle live primary, preserve it as a retained holder if
    /// the family opted into deferred conversion. An exact continuation already claimed the primary
    /// during admission and removed its marker, so this operation leaves that live state untouched.</summary>
    internal bool DisplacePrimary()
    {
        Drain();
        if (_primaryKey is null) return false;
        string key = _primaryKey;
        if (!_tree.Caps.DeferPrimaryConversion || !RetentionEnabled
            || !_tree.Caps.AdoptPrimaryOnDisplacement || _tree.Caps.EndState == EndStateSupport.None
            || !_tree.TryGetNodeByKey(key, out var node) || node.EndState?.Kind != EndStateKind.PrimaryResident)
        {
            InvalidatePrimary();
            return false;
        }
        int length = node.EndState.Footprint.Tokens;
        bool converted;
        PayloadFootprint convertedFootprint;
        try
        {
            if (_cacheModel.TryMeasurePrimaryEndState(length, out var footprint))
                for (int i = 0; i < ResourceVector.ClassCount; i++)
                {
                    var cls = (ResourceClass)i;
                    if (footprint.Bytes[cls] <= _tree.AbsoluteCap(cls)) continue;
                    // Moving this holder would allocate a replacement, only for Trim to immediately
                    // discard it. The live primary remains intact for the ordinary owner reset.
                    InvalidatePrimary();
                    return false;
                }
            converted = _cacheModel.TryConvertPrimary(key, length, out convertedFootprint);
        }
        catch (Exception ex)
        {
            // Retention is optional. Allocating a replacement primary must not turn a new request
            // into a failed batch, and a conversion that threw may already have moved or freed state.
            InvalidatePrimary();
            _cacheModel.ReleasePayloads(new[] { key }, ReleaseReason.Invalidated);
            _logger.LogWarning(ex, "Could not preserve the displaced primary cache; this request prefills normally.");
            return true;
        }
        if (!converted)
        {
            // A refusal may have released the adopted primary. Its marker must disappear regardless
            // of whether the family left the old state intact; this step is about to overwrite it.
            InvalidatePrimary();
            return _cacheModel is IPrefixCacheModelDiagnostics diagnostics && diagnostics.PrimaryCacheLength != length;
        }
        _tree.DetachEndState(node, ReleaseReason.Rollback, enqueue: false);
        _primaryKey = null;
        Publish(node, key, convertedFootprint, length, PayloadOrigin.PrimaryConversion);
        Trim();
        return true;
    }

    internal void ReleaseRequest(SequenceState sequence)
    {
        if (!_requests.Remove(sequence, out var request)) return;
        _tree.NoteRequests(request.Scope, -1, 0);
        _tree.ReleaseRopeOwner(request.Key);
        if (sequence.CacheScope is null) _tree.RetireScope(_tree.Scopes[request.Scope].Id);
        RetireEmptyScopes();
        Drain();
    }

    internal void ReleaseRequest(string requestId)
    {
        SequenceState? found = null;
        foreach (var sequence in _requests.Keys)
            if (sequence.RequestId == requestId) { found = sequence; break; }
        if (found is not null) ReleaseRequest(found);
    }

    public void OnPayloadInvalidated(string payloadKey, InvalidationReason reason) => _invalidated.Enqueue(payloadKey);
    public bool CanMaterialize(string payloadKey, int payloadTokens, int targetTokens)
        => _cacheModel.CanMaterialize(payloadKey, payloadTokens, targetTokens);
    public bool CanRewindPrimary(int payloadTokens, int targetTokens)
        => _cacheModel.CanRewindPrimary(payloadTokens, targetTokens);

    internal void Drain()
    {
        bool hadInvalidations = _invalidated.Count > 0;
        while (_invalidated.TryDequeue(out var key)) _tree.InvalidatePayload(key);
        int refused = _tree.FlushQueuedInvalidations();
        int released = _tree.Reclaim.Drain(ReleasePayloads);
        if (hadInvalidations || refused > 0 || released > 0) RetireEmptyScopes();
    }

    private void RetireEmptyScopes()
    {
        // A name may be interned again by a later request. Keeping empty, idle
        // scope records would otherwise retain every conversation ever served.
        // Scan only at request completion or a payload release, never per token.
        foreach (ScopeRecord scope in _tree.Scopes.LiveRecords())
            if (scope.Index != 0 && scope.NodeCount == 0
                && scope.WaitingRequests == 0 && scope.RunningRequests == 0)
                _tree.RetireScope(scope.Id);
    }

    private void Trim(RadixNode? requestedPublicState = null)
    {
        Drain();
        _tree.EnforceCountSubCaps(requestedPublicState);
        // Public prefixes have eviction priority, not an exemption from byte caps.
        _tree.EnforceCaps(EvictionTier.PublicTop);
        Drain();
    }

    internal string TrimIdleMemory()
    {
        Drain();
        int before = _tree.PayloadKeys.Count;
        _tree.RelieveMemoryPressure(PressureLevel.Critical);
        Drain();
        _model.TrimIdleMemory();
        Drain();
        RetireEmptyScopes();
        return $"evicted {before - _tree.PayloadKeys.Count} radix payload(s); kept {_tree.PayloadKeys.Count} payload(s)";
    }

    internal void Reset()
    {
        foreach (var request in _requests.Values) _tree.ReleaseRopeOwner(request.Key);
        _requests.Clear();
        _tree.Reset();
        _primaryKey = null;
        _storeMisses.Clear();
        Drain();
    }

    /// <summary>Admission needs actual released credit, not the tree's predicted
    /// byte counts. Drain each victim before rechecking the shared multi-pool budget.
    /// Stop as soon as work fits, preserving the remaining reusable prefixes.</summary>
    internal void ReclaimForAdmission(Func<bool> canAdmit)
    {
        Drain();
        while (!canAdmit() && _tree.EvictOneForAdmission()) Drain();
    }

    internal void Detach() => _cacheModel.DetachPrefixCache();

    private static int[] PrefixTokens(SequenceState sequence, int length)
    {
        var tokens = new int[length];
        for (int i = 0; i < length; i++) tokens[i] = sequence.TokenAt(i);
        return tokens;
    }

    private static long PrefixStoreHash(ReadOnlySpan<int> tokens)
    {
        long hash = 1469598103934665603L;
        foreach (int token in tokens) hash = unchecked((hash ^ token) * 1099511628211L);
        return hash;
    }

    private void TryRestoreCheckpoint(SequenceState sequence)
    {
        var store = CheckpointStore;
        if (store is null || !PublicCheckpointsEnabled || !CheckpointsSupported || !_tree.Caps.Persistable
            || sequence.CacheBreakpoints is not null || sequence.MediaSpans.Count > 0) return;
        // Longest first: loading a shorter ancestor must not evict an already
        // usable descendant when the checkpoint budget is small.
        _tree.Plan(BuildRequest(sequence, PredictRoute()), _plan);
        int residentLength = _plan.Length;
        for (int i = sequence.PublicCheckpointBoundaries.Count - 1; i >= 0; i--)
            if (sequence.PublicCheckpointBoundaries[i] > residentLength
                && TryRestoreCheckpoint(sequence, store, sequence.PublicCheckpointBoundaries[i])) return;
    }

    private bool TryRestoreCheckpoint(SequenceState sequence, IPrefixCheckpointStore store, int length)
    {
        int[] tokens = PrefixTokens(sequence, length);
        long hash = PrefixStoreHash(tokens);
        if (_storeMisses.Contains(hash)) return false;
        RadixNode node = _tree.Insert(GetKey(sequence), length, GetScope(sequence), length,
            NodeFlags.None, GetSpans(sequence));
        // A resident state that could be materialized was accounted for by the
        // plan above. A non-materializable payload must not hide a shorter hit.
        if (node.EndState is not null) return false;
        string key = _tree.MintKey();
        try
        {
            if (store.TryOpen(_tree.Caps.NamespaceFingerprint, tokens, out var stream))
            {
                using (stream)
                    if (_cacheModel.TryImport(key, length, stream, out var footprint)
                        && Publish(node, key, footprint, length, PayloadOrigin.DiskImport))
                    {
                        // This ancestor may serve a new branch beyond the two
                        // already resident endpoints. Prefer normal LRU for this
                        // import, without exempting it from any count or byte cap.
                        Trim(requestedPublicState: node);
                        _tree.Plan(BuildRequest(sequence, PredictRoute()), _plan);
                        return _plan.Length >= length;
                    }
            }
            _storeMisses.Add(hash);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "Restoring radix prefix checkpoint failed; recomputing the prefix.");
            _storeMisses.Add(hash);
        }
        _tree.CollectIfEmpty(node);
        Drain();
        return false;
    }

    private void SaveCheckpoint(SequenceState sequence, string key, int length)
    {
        var store = CheckpointStore;
        if (store is null || !_tree.Caps.Persistable || sequence.MediaSpans.Count > 0) return;
        try
        {
            int[] tokens = PrefixTokens(sequence, length);
            if (store.Save(_tree.Caps.NamespaceFingerprint, tokens, stream =>
            {
                if (!_cacheModel.TryExport(key, stream)) throw new InvalidOperationException("The model declined prefix export.");
            })) _storeMisses.Remove(PrefixStoreHash(tokens));
        }
        catch (Exception ex) { _logger.LogWarning(ex, "Saving radix prefix checkpoint failed."); }
    }
}
