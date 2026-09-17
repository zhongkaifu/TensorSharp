// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests.PrefixCache.Fakes;

/// <summary>What one oracle fake emulates (DESIGN §12.3 table).</summary>
internal sealed record OracleTraits
{
    public required string Name { get; init; }
    public required FamilyClass Class { get; init; }
    public PrefixCacheMode Readiness { get; init; } = PrefixCacheMode.Tree;

    // End states
    public EndStateSupport EndState { get; init; }
    public bool CanCaptureCopy { get; init; }
    /// <summary>Per-request caches are native slots with a hard count (class N).</summary>
    public int NativeSlotLimit { get; init; }
    public bool PrimaryResident { get; init; } = true;
    public bool AdoptPrimaryOnDisplacement { get; init; }
    public int MinRetainTokens { get; init; } = 8;
    public bool Persistable { get; init; }
    /// <summary>A forward leaves the state device-authoritative (a holder must be settled before a copy).</summary>
    public bool DeviceDirtyOnForward { get; init; }

    // Truncation
    public TruncationKind Truncation { get; init; }
    /// <summary>W, ring slack, or the ModelDecides span.</summary>
    public int TruncationParameter { get; init; }
    public int TruncationGranularity { get; init; } = 1;
    public int RewindCapTokens { get; init; } = 16;
    /// <summary>A rewind the rules forbid still "succeeds" and yields a wrong state (Gemma's wrapped ring).</summary>
    public bool ForbiddenRewindCorrupts { get; init; }

    // Pages
    public PageSupport Pages { get; init; }
    public bool PagesNeedStateAtEnd { get; init; }
    public int PageWindowTokens { get; init; }
    public bool SupportsCopyPagedToHolder { get; init; }

    public bool ReuseAcrossMediaSpan { get; init; } = true;
}

/// <summary>
/// The shared body of the seven oracle fakes (DESIGN §12.3). Each implements the model interfaces
/// the engine drives (<see cref="IModelArchitecture"/>, <see cref="IBatchedPagedModel"/>) and the
/// prefix-cache contract (<see cref="IPrefixCacheModel"/>, <see cref="IPrefixCacheModelDiagnostics"/>)
/// directly, as an independent reference for the contract's semantics: the Runtime's holder adapter
/// is checked against the same conformance script, never used here.
///
/// <para>Single-threaded like a real model (the engine calls it from its worker only); every
/// member asserts nothing about threads, but <see cref="StateMemberCalls"/> counts contract calls
/// so an engine test can prove Legacy and Shadow never reach them (I26).</para>
/// </summary>
internal class OracleModel : IModelArchitecture, IBatchedPagedModel, IPrefixCacheModel, IPrefixCacheModelDiagnostics
{
    internal const int DefaultVocab = 251;
    private const uint ExportMagic = 0x4F52434B; // "ORCK"

    private readonly Dictionary<string, OracleCache> _holders = new(StringComparer.Ordinal);
    private readonly Dictionary<string, OracleCache> _retained = new(StringComparer.Ordinal);
    private readonly Dictionary<int, ulong> _paged = new();
    private readonly HashSet<string> _faults = new(StringComparer.Ordinal);
    private OracleCache _primary = new();
    private string? _activeKey;
    private long _serial;

    internal OracleModel(OracleTraits traits, int blockSize = 16, int vocab = DefaultVocab)
    {
        Traits = traits;
        BlockSize = blockSize;
        Config = new ModelConfig { VocabSize = vocab, Architecture = "oracle-" + traits.Name };
        Tokenizer = new OracleTokenizer(vocab);
    }

    internal OracleTraits Traits { get; }
    internal int BlockSize { get; }
    internal IPrefixPayloadSink? Sink { get; private set; }
    internal long StateMemberCalls { get; private set; }
    internal long SpareBytes { get; set; } = -1;
    internal List<(string Key, InvalidationReason Reason)> ReportedInvalidations { get; } = new();

    /// <summary>Makes the next call of <paramref name="operation"/> fail: clone, capture, donate,
    /// return, convert, export, import, truncate, copy-paged.</summary>
    internal void FailNext(string operation) => _faults.Add(operation);

    private bool Fault(string operation) => _faults.Remove(operation);

    private bool HasHolders => Traits.EndState != EndStateSupport.None;
    private bool IsNative => Traits.NativeSlotLimit > 0;
    private OracleCache Active => _activeKey == null ? _primary : _holders[_activeKey];

    // ------------------------------------------------------------------ IModelArchitecture

    public ModelConfig Config { get; }
    public ITokenizer Tokenizer { get; }
    public IMultimodalInjector MultimodalInjector => null!;
    public IBackendExecutionPlan ExecutionPlan => null!;

    public float[] Forward(int[] tokens)
    {
        if (tokens == null || tokens.Length == 0) throw new ArgumentException("tokens required", nameof(tokens));
        OracleCache cache = Active;
        foreach (int t in tokens)
        {
            if ((uint)t >= (uint)Config.VocabSize) throw new ArgumentOutOfRangeException(nameof(tokens), t, "token outside the oracle vocabulary");
            cache.Append(t);
        }
        cache.ForwardBoundaries.Add(cache.Length);
        if (!Traits.DeviceDirtyOnForward) cache.Flush();
        cache.WasBound |= _activeKey != null;
        return OracleHash.Logits(cache.State, Config.VocabSize);
    }

    public void ResetKVCache()
    {
        OracleCache cache = Active;
        cache.TruncateTo(0, corrupt: false);
    }

    public bool SupportsKVCacheTruncation => Traits.Truncation != TruncationKind.None;
    public int KVCacheTruncationGranularity => Traits.TruncationGranularity;

    public bool CanTruncateKVCache(int cachedTokenCount, int targetTokenCount)
        => targetTokenCount >= 0 && targetTokenCount <= cachedTokenCount
           && (targetTokenCount == cachedTokenCount || RewindAllowed(cachedTokenCount, targetTokenCount));

    /// <summary>The family's truncation rule for a rewind from <paramref name="cached"/> to <paramref name="target"/> &lt; cached.</summary>
    internal bool RewindAllowed(int cached, int target) => Traits.Truncation switch
    {
        TruncationKind.None => false,
        TruncationKind.Any => target % Traits.TruncationGranularity == 0,
        TruncationKind.WithinUnwrappedWindow => cached <= Traits.TruncationParameter || target == 0,
        TruncationKind.WithinRingSlack => cached - target <= Traits.TruncationParameter,
        TruncationKind.ModelDecides => cached - target <= Traits.TruncationParameter && target % Traits.TruncationGranularity == 0,
        _ => false,
    };

    public void TruncateKVCache(int tokenCount)
    {
        if (!TryTruncateKVCache(tokenCount))
            throw new InvalidOperationException($"{Traits.Name}: truncation to {tokenCount} refused");
    }

    public bool TryTruncateKVCache(int tokenCount)
    {
        OracleCache cache = Active;
        if (tokenCount < 0 || tokenCount > cache.Length) return false;
        if (tokenCount == cache.Length) return true;
        if (Fault("truncate")) return false;
        if (Traits.Truncation == TruncationKind.None) return false;
        bool allowed = RewindAllowed(cache.Length, tokenCount);
        if (!allowed && !Traits.ForbiddenRewindCorrupts) return false;
        cache.TruncateTo(tokenCount, corrupt: !allowed);
        return true;
    }

    public bool SupportsKVStateSnapshot => (Traits.Pages & PageSupport.A1HostSlab) != 0;
    public bool SupportsCrossSequenceKvReuse => SupportsKVStateSnapshot;
    public int MaxReusablePrefixTokens => Traits.PageWindowTokens > 0 ? Traits.PageWindowTokens : int.MaxValue;
    public bool SupportsReuseAcrossMediaSpan => Traits.ReuseAcrossMediaSpan;
    public int MaxContextLength => 4096;
    public string KVStateFingerprint => $"oracle-{Traits.Name}|v1|vocab={Config.VocabSize}";
    public bool RequiresPerBlockCapture => Traits.PagesNeedStateAtEnd;
    public long ComputeKVBlockByteSize(int tokenCount) => 8L * tokenCount + (Traits.PagesNeedStateAtEnd ? 8 : 0);

    public bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
    {
        if (!SupportsKVStateSnapshot) return false;
        OracleCache cache = Active;
        if (startToken < 0 || tokenCount <= 0 || startToken + tokenCount > cache.Length) return false;
        if (destination.Length != ComputeKVBlockByteSize(tokenCount)) return false;
        cache.Flush();
        for (int i = 0; i < tokenCount; i++)
            MemoryMarshal.Write(destination.Slice(8 * i, 8), cache.Chain[startToken + i + 1]);
        if (Traits.PagesNeedStateAtEnd)
        {
            ulong restorable = cache.ForwardBoundaries.Contains(startToken + tokenCount) ? 1UL : 0UL;
            MemoryMarshal.Write(destination.Slice(8 * tokenCount, 8), restorable);
        }
        return true;
    }

    public bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source)
    {
        if (!SupportsKVStateSnapshot) return false;
        OracleCache cache = Active;
        if (destToken != cache.Length || tokenCount <= 0 || source.Length != ComputeKVBlockByteSize(tokenCount)) return false;
        for (int i = 0; i < tokenCount; i++)
            cache.Chain.Add(MemoryMarshal.Read<ulong>(source.Slice(8 * i, 8)));
        if (Traits.PagesNeedStateAtEnd && MemoryMarshal.Read<ulong>(source.Slice(8 * tokenCount, 8)) == 0)
            cache.Chain[^1] ^= OracleHash.Poison;   // recurrent state at a non-boundary is not restorable
        cache.ForwardBoundaries.Add(cache.Length);
        cache.Flush();
        return true;
    }

    public void Dispose() { }

    // ------------------------------------------------------------------ IBatchedPagedModel

    public bool BatchedForwardAvailable => (Traits.Pages & PageSupport.A2ModelPaged) != 0;

    /// <summary>Model-paged (A2) forward: a token's predecessor state is READ through the block
    /// table, so a page shared with the wrong content changes the output.</summary>
    public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx)
    {
        if (!BatchedForwardAvailable) throw new NotSupportedException($"{Traits.Name} has no model-paged storage");
        var results = new List<float[]>(ctx.Sequences.Count);
        for (int s = 0; s < ctx.Sequences.Count; s++)
        {
            SequenceState seq = ctx.Sequences[s];
            int start = ctx.QueryStartLoc[s];
            int count = ctx.NumScheduledTokens[s];
            ulong state = OracleHash.Seed;
            for (int q = start; q < start + count; q++)
            {
                int position = ctx.Positions[q];
                int token = ctx.OverrideFlatTokens != null ? ctx.OverrideFlatTokens[q] : seq.TokenAt(position);
                ulong previous = position == 0
                    ? OracleHash.Seed
                    : _paged.TryGetValue(SlotOf(ctx.BlockTables[s], position - 1), out ulong p) ? p : 0UL;
                state = OracleHash.Mix(previous, token, position);
                _paged[ctx.SlotMapping[q]] = state;
            }
            results.Add(OracleHash.Logits(state, Config.VocabSize));
        }
        return results;
    }

    private int SlotOf(int[] blockTable, int position) => blockTable[position / BlockSize] * BlockSize + position % BlockSize;

    public bool SupportsLinearKVMigration => BatchedForwardAvailable;

    public bool TryMigrateLinearKVToPaged(SequenceState owner, int blockSize)
    {
        if (!BatchedForwardAvailable || blockSize != BlockSize) return false;
        OracleCache cache = Active;
        if (owner.BlockTable.CapacityTokens < cache.Length) return false;
        int[] table = owner.BlockTable.Blocks.Select(b => b.Id).ToArray();
        for (int pos = 0; pos < cache.Length; pos++)
            _paged[SlotOf(table, pos)] = cache.Chain[pos + 1];
        return true;
    }

    public bool SupportsPerSequenceFusedForward => HasHolders;

    public bool BindSequenceCache(string requestId)
    {
        if (string.IsNullOrEmpty(requestId)) throw new ArgumentException("RequestId required", nameof(requestId));
        if (!HasHolders) return false;
        if (string.Equals(_activeKey, requestId, StringComparison.Ordinal)) return false;
        bool fresh = false;
        if (!_holders.TryGetValue(requestId, out OracleCache? holder))
        {
            EnsureNativeSlotFor(requestId);
            holder = new OracleCache { Serial = ++_serial };
            _holders[requestId] = holder;
            fresh = true;
        }
        _activeKey = requestId;
        holder.WasBound = true;
        return fresh;
    }

    /// <summary>Class N: a new slot beyond the limit reclaims the oldest retained slot and reports it
    /// through the sink (DEC-23), the way DeepSeek V4.1's <c>ReclaimRetainedPrimary</c> does.</summary>
    private void EnsureNativeSlotFor(string requestId)
    {
        if (!IsNative) return;
        int used = 1 + _holders.Count + _retained.Count;   // the primary always owns a slot
        if (used < Traits.NativeSlotLimit) return;
        var victim = _retained.OrderBy(kv => kv.Value.Serial).Select(kv => kv.Key).FirstOrDefault();
        if (victim == null)
            throw new InvalidOperationException($"{Traits.Name}: no native slot for {requestId} ({used} of {Traits.NativeSlotLimit} in use)");
        _retained.Remove(victim);
        ReportedInvalidations.Add((victim, InvalidationReason.NativeSlotReclaimed));
        Sink?.OnPayloadInvalidated(victim, InvalidationReason.NativeSlotReclaimed);
    }

    public void AdoptPrimaryCacheToFused(string requestId)
    {
        if (string.IsNullOrEmpty(requestId) || !HasHolders) return;
        if (_activeKey != null || _holders.ContainsKey(requestId)) return;
        _primary.WasBound = true;
        _holders[requestId] = _primary;
        _activeKey = requestId;
        _primary = new OracleCache { Serial = ++_serial };
    }

    public void RestorePrimaryCache() => _activeKey = null;

    public bool HasFusedSequenceCache(string requestId) => requestId != null && _holders.ContainsKey(requestId);

    public void OnSequenceReleased(string requestId)
    {
        if (string.IsNullOrEmpty(requestId)) return;
        if (string.Equals(_activeKey, requestId, StringComparison.Ordinal)) _activeKey = null;
        if (_holders.Remove(requestId, out OracleCache? holder)) holder.Retired = true;
    }

    public bool SupportsRetainedFusedCache => HasHolders;

    public bool RetainSequenceCache(string requestId) => RetainAs(requestId, requestId);

    /// <summary>The key-parameterised retain (used directly by the adapter-backed oracle).</summary>
    internal bool RetainAs(string requestId, string key)
    {
        if (string.IsNullOrEmpty(requestId) || string.IsNullOrEmpty(key)) return false;
        if (!_holders.TryGetValue(requestId, out OracleCache? holder) || _retained.ContainsKey(key)) return false;
        if (string.Equals(_activeKey, requestId, StringComparison.Ordinal)) _activeKey = null;
        _holders.Remove(requestId);
        holder.Serial = ++_serial;
        _retained.Add(key, holder);
        return true;
    }

    public bool TryRebindRetainedCache(string retainedRequestId, string newRequestId)
    {
        if (string.IsNullOrEmpty(retainedRequestId) || string.IsNullOrEmpty(newRequestId)) return false;
        if (!_retained.TryGetValue(retainedRequestId, out OracleCache? holder) || holder.Retired) return false;
        if (_holders.ContainsKey(newRequestId)) return false;
        _retained.Remove(retainedRequestId);
        _holders.Add(newRequestId, holder);
        return true;
    }

    public void DiscardRetainedCache(string requestId)
    {
        if (requestId != null && _retained.Remove(requestId, out OracleCache? holder)) holder.Retired = true;
    }

    public bool SupportsPrefixCheckpoints => Traits.CanCaptureCopy;

    public bool TryCheckpointActiveCache(string key)
    {
        if (!Traits.CanCaptureCopy || string.IsNullOrEmpty(key)) return false;
        if (_retained.ContainsKey(key) || _holders.ContainsKey(key)) return false;
        OracleCache active = Active;
        active.Flush();                          // P20: the copy reads host bytes
        OracleCache copy = active.CopyHost();
        copy.Serial = ++_serial;
        _retained.Add(key, copy);
        return true;
    }

    /// <summary>Refuses a device-dirty source, like Gemma 4 and Qwen 3.5: a caller that did not
    /// settle the holder first gets a refusal, never a silently stale copy.</summary>
    public bool TryCloneRetainedCache(string retainedKey, string newRequestId)
    {
        if (Traits.EndState != EndStateSupport.CopyAndDonate) return false;
        if (string.IsNullOrEmpty(retainedKey) || string.IsNullOrEmpty(newRequestId)) return false;
        if (!_retained.TryGetValue(retainedKey, out OracleCache? source) || source.Retired || source.DeviceDirty) return false;
        if (_holders.ContainsKey(newRequestId) || string.Equals(_activeKey, newRequestId, StringComparison.Ordinal)) return false;
        OracleCache copy = source.CopyHost();
        copy.Serial = ++_serial;
        _holders.Add(newRequestId, copy);
        return true;
    }

    public bool SupportsRetainedCacheSerialization => Traits.Persistable;

    public bool TryExportRetainedCache(string key, Stream destination)
    {
        if (!Traits.Persistable || destination == null || key == null) return false;
        if (!_retained.TryGetValue(key, out OracleCache? cache) || cache.Retired || cache.DeviceDirty) return false;
        var w = new BinaryWriter(destination, System.Text.Encoding.UTF8, leaveOpen: true);
        w.Write(ExportMagic);
        w.Write(KVStateFingerprint);
        w.Write(cache.Length);
        w.Write(cache.RopeDelta);
        foreach (ulong v in cache.Chain) w.Write(v);
        w.Flush();
        return true;
    }

    public bool TryImportRetainedCache(string key, Stream source)
    {
        if (!Traits.Persistable || source == null || string.IsNullOrEmpty(key)) return false;
        if (_retained.ContainsKey(key) || _holders.ContainsKey(key)) return false;
        try
        {
            var r = new BinaryReader(source, System.Text.Encoding.UTF8, leaveOpen: true);
            if (r.ReadUInt32() != ExportMagic || r.ReadString() != KVStateFingerprint) return false;
            int length = r.ReadInt32();
            if (length < 0 || length > MaxContextLength) return false;
            var cache = new OracleCache { RopeDelta = r.ReadInt32(), Serial = ++_serial };
            cache.Chain.Clear();
            for (int i = 0; i <= length; i++) cache.Chain.Add(r.ReadUInt64());
            cache.ForwardBoundaries.Add(length);
            cache.Flush();
            _retained.Add(key, cache);
            return true;
        }
        catch (EndOfStreamException)
        {
            return false;
        }
    }

    // ------------------------------------------------------------------ IPrefixCacheModel

    public PrefixCacheCapabilities GetPrefixCacheCapabilities() => new()
    {
        Class = Traits.Class,
        Readiness = Traits.Readiness,
        NamespaceFingerprint = KVStateFingerprint,
        EndState = Traits.EndState,
        CanCaptureCopy = Traits.CanCaptureCopy,
        AdoptPrimaryOnDisplacement = Traits.AdoptPrimaryOnDisplacement,
        PrimaryResident = Traits.PrimaryResident,
        MinRetainTokens = Traits.MinRetainTokens,
        Truncation = Traits.Truncation,
        TruncationParameter = Traits.TruncationParameter,
        TruncationGranularity = Traits.TruncationGranularity,
        RewindCapTokens = Traits.RewindCapTokens,
        Pages = Traits.Pages,
        PagesNeedStateAtEnd = Traits.PagesNeedStateAtEnd,
        PageWindowTokens = Traits.PageWindowTokens,
        SupportsCopyPagedToHolder = Traits.SupportsCopyPagedToHolder,
        ReuseAcrossMediaSpan = Traits.ReuseAcrossMediaSpan,
        Persistable = Traits.Persistable,
        MaxRetainedNativeSlots = 0,
    };

    public void AttachPrefixCache(IPrefixPayloadSink sink)
    {
        StateMemberCalls++;
        Sink = sink ?? throw new ArgumentNullException(nameof(sink));
    }

    public bool TryCaptureCopy(string requestId, string payloadKey, out PayloadFootprint footprint)
    {
        StateMemberCalls++;
        footprint = default;
        if (Fault("capture")) return false;
        // The request's cache must be the active one; the primary serves the request that owns it.
        if (_activeKey != null && !string.Equals(_activeKey, requestId, StringComparison.Ordinal)) return false;
        if (!TryCheckpointActiveCache(payloadKey)) return false;
        footprint = Measure(_retained[payloadKey]);
        return true;
    }

    public bool TryCaptureDonate(string requestId, string payloadKey, int length, out PayloadFootprint footprint)
    {
        StateMemberCalls++;
        footprint = default;
        if (!HasHolders || Fault("donate")) return false;
        if (string.IsNullOrEmpty(requestId) || string.IsNullOrEmpty(payloadKey) || _retained.ContainsKey(payloadKey)) return false;
        if (!_holders.TryGetValue(requestId, out OracleCache? holder) || length <= 0 || length > holder.Length) return false;
        if (length < holder.Length)
        {
            if (!RewindAllowed(holder.Length, length)) return false;
            holder.TruncateTo(length, corrupt: false);
        }
        if (!RetainAs(requestId, payloadKey)) return false;
        footprint = Measure(holder);
        return true;
    }

    public bool TryConvertPrimary(string payloadKey, int length, out PayloadFootprint footprint)
    {
        StateMemberCalls++;
        footprint = default;
        if (!Traits.AdoptPrimaryOnDisplacement || Fault("convert")) return false;
        if (string.IsNullOrEmpty(payloadKey) || _retained.ContainsKey(payloadKey) || _activeKey != null) return false;
        if (_primary.Length != length || length <= 0) return false;
        OracleCache adopted = _primary;
        adopted.WasBound = true;
        adopted.Serial = ++_serial;
        _primary = new OracleCache { Serial = ++_serial };
        _retained.Add(payloadKey, adopted);
        footprint = Measure(adopted);
        return true;
    }

    public bool TryMaterialize(in MaterializeRequest request)
    {
        StateMemberCalls++;
        if (!CanMaterializeCore(request.PayloadKey, request.PayloadTokens, request.TargetTokens)) return false;
        if (string.IsNullOrEmpty(request.TargetRequestId) || _holders.ContainsKey(request.TargetRequestId)) return false;
        OracleCache source = _retained[request.PayloadKey];
        switch (request.Op)
        {
            case MaterializeOp.Clone:
                if (Traits.EndState != EndStateSupport.CopyAndDonate || Fault("clone")) return false;
                source.Flush();                  // settle a donated, device-authoritative holder (G-12)
                return TryCloneRetainedCache(request.PayloadKey, request.TargetRequestId);
            case MaterializeOp.Donate:
                if (Fault("donate")) return false;
                return TryRebindRetainedCache(request.PayloadKey, request.TargetRequestId);
            default:
                return false;
        }
    }

    public bool TryReturnDonation(string requestId, string payloadKey)
    {
        StateMemberCalls++;
        if (Fault("return")) return false;
        if (string.Equals(_activeKey, requestId, StringComparison.Ordinal)) return false;   // a bound holder is no longer the payload
        return RetainAs(requestId, payloadKey);
    }

    public bool CanMaterialize(string payloadKey, int payloadTokens, int targetTokens)
    {
        StateMemberCalls++;
        return CanMaterializeCore(payloadKey, payloadTokens, targetTokens);
    }

    private bool CanMaterializeCore(string payloadKey, int payloadTokens, int targetTokens)
    {
        if (payloadKey == null || !_retained.TryGetValue(payloadKey, out OracleCache? cache) || cache.Retired) return false;
        if (cache.Length != payloadTokens || targetTokens < 0 || targetTokens > payloadTokens) return false;
        return targetTokens == payloadTokens || RewindAllowed(payloadTokens, targetTokens);
    }

    public void ReleasePayloads(ReadOnlySpan<string> payloadKeys, ReleaseReason reason)
    {
        StateMemberCalls++;
        bool released = false;
        foreach (string key in payloadKeys)
        {
            if (key != null && _retained.Remove(key, out OracleCache? cache))
            {
                cache.Retired = true;
                released = true;
            }
        }
        if (released) DecodeGraphResets++;   // one reset per batch, never one per payload (DEC-24)
        ReleaseCalls++;
    }

    internal int ReleaseCalls { get; private set; }

    public PayloadFootprint MeasureEndState(string payloadKey)
    {
        StateMemberCalls++;
        return payloadKey != null && _retained.TryGetValue(payloadKey, out OracleCache? cache) ? Measure(cache) : default;
    }

    private PayloadFootprint Measure(OracleCache cache)
    {
        long rows = 16L * cache.CapacityTokens;
        ResourceVector bytes = default;
        if (IsNative)
        {
            bytes.NativeSlot = 16L * MaxContextLength;
        }
        else
        {
            bytes.HostKv = rows;
            if (cache.WasBound && Traits.DeviceDirtyOnForward) bytes.DeviceKv = rows;
            if (Traits.Class == FamilyClass.R) bytes.StateSnapshot = 1024;
        }
        return new PayloadFootprint(cache.Length, cache.CapacityTokens, bytes, cache.RopeDelta);
    }

    public ResourceVector EstimateCloneBytes(string payloadKey, int targetTokens)
    {
        StateMemberCalls++;
        if (Traits.EndState != EndStateSupport.CopyAndDonate || payloadKey == null || !_retained.TryGetValue(payloadKey, out OracleCache? cache))
            return default;
        ResourceVector bytes = default;
        bytes.HostKv = 16L * cache.CapacityTokens;
        if (Traits.Class == FamilyClass.R) bytes.StateSnapshot = 1024;
        return bytes;
    }

    public bool TryCopyPagedToHolder(ReadOnlySpan<int> blockIds, int tokens, string requestId)
    {
        StateMemberCalls++;
        if (!Traits.SupportsCopyPagedToHolder || Fault("copy-paged")) return false;
        if (string.IsNullOrEmpty(requestId) || _holders.ContainsKey(requestId) || tokens <= 0 || tokens > blockIds.Length * BlockSize) return false;
        var holder = new OracleCache { Serial = ++_serial };
        int[] table = blockIds.ToArray();
        for (int pos = 0; pos < tokens; pos++)
            holder.Chain.Add(_paged.TryGetValue(SlotOf(table, pos), out ulong v) ? v : 0UL);
        holder.ForwardBoundaries.Add(tokens);
        holder.Flush();
        _holders.Add(requestId, holder);
        return true;
    }

    public bool TryExport(string payloadKey, Stream destination)
    {
        StateMemberCalls++;
        return !Fault("export") && TryExportRetainedCache(payloadKey, destination);
    }

    public bool TryImport(string payloadKey, int tokens, Stream source, out PayloadFootprint footprint)
    {
        StateMemberCalls++;
        footprint = default;
        if (Fault("import") || !TryImportRetainedCache(payloadKey, source)) return false;
        if (_retained[payloadKey].Length != tokens)
        {
            _retained.Remove(payloadKey);
            return false;
        }
        footprint = Measure(_retained[payloadKey]);
        return true;
    }

    public bool TryBeginImport(int tokens, out object? importTicket)
    {
        StateMemberCalls++;
        importTicket = null;
        return false;   // lanes arrive with M7c
    }

    public bool RunImportRead(object importTicket, Stream source) => false;

    public bool TryCommitImport(object importTicket, string payloadKey, out PayloadFootprint footprint)
    {
        StateMemberCalls++;
        footprint = default;
        return false;
    }

    public void AbortImport(object importTicket) => StateMemberCalls++;

    public long QuerySpareBytes(ResourceClass cls)
    {
        StateMemberCalls++;
        return cls == ResourceClass.PoolPages ? -1 : SpareBytes;
    }

    // ------------------------------------------------------------------ IPrefixCacheModelDiagnostics

    public IReadOnlyCollection<string> RetainedPayloadKeys => _retained.Keys.ToArray();
    public int PrivateHolderCount => _holders.Count;
    public int PrimaryCacheLength => _primary.Length;
    public long DecodeGraphResets { get; private set; }

    // ------------------------------------------------------------------ test visibility

    /// <summary>The state after the active cache's last token (tests compare it with a cold run).</summary>
    internal ulong ActiveState => Active.State;
    internal int ActiveLength => Active.Length;
    internal bool IsDeviceDirty(string payloadKey) => _retained.TryGetValue(payloadKey, out OracleCache? c) && c.DeviceDirty;

    /// <summary>Flush a retained holder's device-authoritative state to the host (the family side of
    /// <c>SettleForCopy</c>, used by the adapter-backed oracle).</summary>
    internal bool Settle(string payloadKey)
    {
        if (payloadKey == null || !_retained.TryGetValue(payloadKey, out OracleCache? cache) || cache.Retired) return false;
        cache.Flush();
        return true;
    }
}
