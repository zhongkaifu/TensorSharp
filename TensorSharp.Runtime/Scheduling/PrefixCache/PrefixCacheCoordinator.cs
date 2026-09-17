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
        // A materialized holder is only readable through the fused per-request
        // route. Primary-only overrides must not adopt placeholder blocks for it.
        if (options.BatchedPathDisabled || !options.PerSeqFusedEnabled
            || model is not IBatchedPagedModel { SupportsPerSequenceFusedForward: true })
            capabilities = capabilities with
            {
                EndState = EndStateSupport.None, CanCaptureCopy = false,
                AdoptPrimaryOnDisplacement = false,
            };
        _tree = new PrefixTree(new PrefixTreeOptions
        {
            Capabilities = capabilities,
            EngineSerial = Interlocked.Increment(ref s_engineSerial),
            BlockSize = pool.BlockSize,
            ContextLength = model.MaxContextLength > 0 ? model.MaxContextLength : int.MaxValue,
            PublicMax = options.PrefixCheckpointBudget,
            ScopedEndStateLeavesMax = options.RetainedFusedCacheBudget,
            PageHost = new PageHost(pool),
            PageHostBytes = InferenceEngine.ComputeBlockByteSize(model, pool.BlockSize),
            PoolPagesCap = pool.NumBlocks,
            BatchedPagedEnabled = !options.BatchedPathDisabled,
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
        => ExecutionOptions.FromEnvironment() is { RetainedFusedCacheEnabled: true, RetainedFusedCacheBudget: > 0 };
    private static bool PublicCheckpointsEnabled
        => ExecutionOptions.FromEnvironment() is { PrefixCheckpointsEnabled: true, PrefixCheckpointBudget: > 0 };
    internal bool CheckpointsSupported => _tree.Caps.CanCaptureCopy && (PublicCheckpointsEnabled || RetentionEnabled);
    internal string? LastSource { get; private set; }
    internal int LastBlockedByScope { get; private set; }
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
        int limit = sequence.PromptTokens.Count - 1;
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
        if (!options.BatchedPathDisabled && options.PerSeqFusedEnabled
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
        if (PredictRoute() != ExpectedRoute.PerSequenceFused
            && _plan.Mode is MaterializeMode.CloneEndState or MaterializeMode.DonateEndState
                or MaterializeMode.ConvertPrimaryThenClone)
            _plan.Reset();
        LastBlockedByScope = _plan.BlockedByScope;
        return _plan.Length;
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
        bool publicBoundary = length == sequence.SharedPrefixTokens && !sequence.PrefixCheckpointTaken;
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
        if (publicBoundary) sequence.PrefixCheckpointTaken = true;
        Drain();
        RadixNode node = _tree.Insert(GetKey(sequence), length, GetScope(sequence), sequence.SharedPrefixTokens,
            explicitBoundary ? NodeFlags.EndsAtBreakpoint : promptBoundary ? NodeFlags.PromptEnd : NodeFlags.None,
            GetSpans(sequence));
        if (node.EndState is not null) return;
        string key = _tree.MintKey();
        bool captured = _cacheModel.TryCaptureCopy(sequence.RequestId, key, out var footprint);
        if (!captured && ReleaseOldestScopedPayload())
            captured = _cacheModel.TryCaptureCopy(sequence.RequestId, key, out footprint);
        if (!captured)
        {
            _tree.CollectIfEmpty(node);
            return;
        }
        if (Publish(node, key, footprint, length, PayloadOrigin.CaptureCopy) && publicBoundary)
            SaveCheckpoint(sequence, key, length);
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
        if (node.EndState is not null) return false;
        string payloadKey = _tree.MintKey();
        bool captured = RetentionEnabled && _tree.Caps.EndState != EndStateSupport.None
            && _cacheModel.TryCaptureDonate(sequence.RequestId, payloadKey, length, out _);
        if (!captured && RetentionEnabled && _tree.Caps.EndState != EndStateSupport.None
            && _model is IBatchedPagedModel holder && holder.HasFusedSequenceCache(sequence.RequestId)
            && ReleaseOldestScopedPayload())
            captured = _cacheModel.TryCaptureDonate(sequence.RequestId, payloadKey, length, out _);
        PayloadOrigin origin = PayloadOrigin.Donation;
        if (!captured && RetentionEnabled && primary && _tree.Caps.AdoptPrimaryOnDisplacement)
        {
            captured = _cacheModel.TryConvertPrimary(payloadKey, length, out _);
            if (!captured && ReleaseOldestScopedPayload())
                captured = _cacheModel.TryConvertPrimary(payloadKey, length, out _);
            origin = PayloadOrigin.PrimaryConversion;
        }
        if (captured)
        {
            bool published = Publish(node, payloadKey, _cacheModel.MeasureEndState(payloadKey), length, origin);
            Trim();
            return published;
        }
        if (primary && _tree.Caps.PrimaryResident)
        {
            InvalidatePrimary();
            _tree.AttachEndState(node, new EndStatePayload
            {
                Key = payloadKey, Kind = EndStateKind.PrimaryResident,
                Footprint = new PayloadFootprint(length, length, default, 0),
            });
            _primaryKey = payloadKey;
        }
        else _tree.CollectIfEmpty(node);
        return false;
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

    internal void Drain()
    {
        bool hadInvalidations = _invalidated.Count > 0;
        while (_invalidated.TryDequeue(out var key)) _tree.InvalidatePayload(key);
        int refused = _tree.FlushQueuedInvalidations();
        int released = _tree.Reclaim.Drain(_cacheModel.ReleasePayloads);
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

    private void Trim()
    {
        Drain();
        _tree.EnforceCountSubCaps();
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

    internal void Detach() => _cacheModel.DetachPrefixCache();

    private static int[] PrefixTokens(SequenceState sequence, int length)
    {
        var tokens = new int[length];
        for (int i = 0; i < length; i++) tokens[i] = sequence.TokenAt(i);
        return tokens;
    }

    private void TryRestoreCheckpoint(SequenceState sequence)
    {
        var store = CheckpointStore;
        int length = sequence.SharedPrefixTokens;
        if (store is null || !PublicCheckpointsEnabled || !CheckpointsSupported || !_tree.Caps.Persistable || length <= 0
            || sequence.CacheBreakpoints is not null || sequence.MediaSpans.Count > 0) return;
        int[] tokens = PrefixTokens(sequence, length);
        long hash = 1469598103934665603L;
        foreach (int token in tokens) hash = unchecked((hash ^ token) * 1099511628211L);
        if (_storeMisses.Contains(hash)) return;
        RadixNode node = _tree.Insert(GetKey(sequence), length, GetScope(sequence), length,
            NodeFlags.None, GetSpans(sequence));
        if (node.EndState is not null) return;
        string key = _tree.MintKey();
        try
        {
            if (store.TryOpen(_tree.Caps.NamespaceFingerprint, tokens, out var stream))
            {
                using (stream)
                    if (_cacheModel.TryImport(key, length, stream, out var footprint)
                        && Publish(node, key, footprint, length, PayloadOrigin.DiskImport))
                    {
                        Trim();
                        return;
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
    }

    private void SaveCheckpoint(SequenceState sequence, string key, int length)
    {
        var store = CheckpointStore;
        if (store is null || !_tree.Caps.Persistable || sequence.MediaSpans.Count > 0) return;
        try
        {
            store.Save(_tree.Caps.NamespaceFingerprint, PrefixTokens(sequence, length), stream =>
            {
                if (!_cacheModel.TryExport(key, stream)) throw new InvalidOperationException("The model declined prefix export.");
            });
        }
        catch (Exception ex) { _logger.LogWarning(ex, "Saving radix prefix checkpoint failed."); }
    }
}
