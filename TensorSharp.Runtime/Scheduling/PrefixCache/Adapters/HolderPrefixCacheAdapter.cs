// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.IO;

namespace TensorSharp.Runtime.Scheduling.PrefixCache;

/// <summary>
/// A family whose end states are per-request holders (or native slots) that already live through
/// <see cref="IBatchedPagedModel"/>'s retained-holder lifecycle: Gemma 4, Qwen 3.5, Qwen4Exp and
/// DeepSeek V4.1. <see cref="HolderPrefixCacheAdapter"/> supplies the <see cref="IPrefixCacheModel"/>
/// members that map onto calls the family already has (DESIGN §4.8 table); the family adds only
/// the members below plus <see cref="IPrefixCacheModel.GetPrefixCacheCapabilities"/>,
/// <see cref="IPrefixCacheModel.AttachPrefixCache"/>, <see cref="IPrefixCacheModel.TryConvertPrimary"/>,
/// <see cref="IPrefixCacheModel.CanMaterialize"/>, <see cref="IPrefixCacheModel.MeasureEndState"/>,
/// <see cref="IPrefixCacheModel.EstimateCloneBytes"/> and <see cref="IPrefixCacheModel.QuerySpareBytes"/>.
///
/// <para>Implementing this interface changes nothing for a model's existing callers: every member
/// is new, or reachable only through <see cref="IPrefixCacheModel"/>, which the engine calls only
/// in <see cref="PrefixCacheMode.Tree"/>.</para>
/// </summary>
public interface IHolderPrefixCacheModel : IPrefixCacheModel, IBatchedPagedModel, IModelArchitecture
{
    /// <summary>The key-parameterised form of <see cref="IBatchedPagedModel.RetainSequenceCache"/>:
    /// move <paramref name="requestId"/>'s private holder into the retained set under
    /// <paramref name="payloadKey"/>. Zero copy.</summary>
    bool RetainSequenceCacheAs(string requestId, string payloadKey);

    /// <summary>The batched form of <see cref="IBatchedPagedModel.DiscardRetainedCache"/>: release
    /// every listed retained holder (unknown keys ignored), recycling into a holder pool where the
    /// family has one, with at most one decode-graph reset for the whole batch (DEC-24).</summary>
    void DiscardRetainedCaches(ReadOnlySpan<string> payloadKeys, ReleaseReason reason);

    /// <summary>Make a retained holder host-authoritative so a copy reads current bytes (G-12, §6.1):
    /// between steps, bind it, run the family's synchronous host flush, rebind the previous cache.
    /// True when the holder is (now) safe to copy; false when it is unknown or cannot be settled.</summary>
    bool SettleForCopy(string payloadKey);

    bool IPrefixCacheModel.TryCaptureCopy(string requestId, string payloadKey, out PayloadFootprint footprint)
        => HolderPrefixCacheAdapter.TryCaptureCopy(this, requestId, payloadKey, out footprint);

    bool IPrefixCacheModel.TryCaptureDonate(string requestId, string payloadKey, int length, out PayloadFootprint footprint)
        => HolderPrefixCacheAdapter.TryCaptureDonate(this, requestId, payloadKey, length, out footprint);

    bool IPrefixCacheModel.TryMaterialize(in MaterializeRequest request)
        => HolderPrefixCacheAdapter.TryMaterialize(this, request);

    bool IPrefixCacheModel.TryReturnDonation(string requestId, string payloadKey)
        => !string.IsNullOrEmpty(requestId) && !string.IsNullOrEmpty(payloadKey)
           && RetainSequenceCacheAs(requestId, payloadKey);

    void IPrefixCacheModel.ReleasePayloads(ReadOnlySpan<string> payloadKeys, ReleaseReason reason)
        => DiscardRetainedCaches(payloadKeys, reason);

    bool IPrefixCacheModel.TryCopyPagedToHolder(ReadOnlySpan<int> blockIds, int tokens, string requestId) => false;

    bool IPrefixCacheModel.TryExport(string payloadKey, Stream destination)
        => !string.IsNullOrEmpty(payloadKey) && destination != null && TryExportRetainedCache(payloadKey, destination);

    bool IPrefixCacheModel.TryImport(string payloadKey, int tokens, Stream source, out PayloadFootprint footprint)
        => HolderPrefixCacheAdapter.TryImport(this, payloadKey, tokens, source, out footprint);

    // Lanes arrive with M7c, per model, after the §12.6 byte-identity proofs.
    bool IPrefixCacheModel.TryBeginImport(int tokens, out object? importTicket)
    {
        importTicket = null;
        return false;
    }

    bool IPrefixCacheModel.RunImportRead(object importTicket, Stream source) => false;

    bool IPrefixCacheModel.TryCommitImport(object importTicket, string payloadKey, out PayloadFootprint footprint)
    {
        footprint = default;
        return false;
    }

    void IPrefixCacheModel.AbortImport(object importTicket) { }
}

/// <summary>
/// The shared end-state members of <see cref="IHolderPrefixCacheModel"/> (DESIGN §4.8, §6.1): each maps a
/// contract operation onto the retained-holder calls a family already implements.
/// </summary>
public static class HolderPrefixCacheAdapter
{
    /// <summary><c>TryCaptureCopy</c> = <c>TryCheckpointActiveCache(key)</c>, which flushes the active cache
    /// first. The request's cache must be the active one, as it is right after its forward.</summary>
    public static bool TryCaptureCopy(IHolderPrefixCacheModel model, string requestId, string payloadKey, out PayloadFootprint footprint)
    {
        footprint = default;
        if (string.IsNullOrEmpty(requestId) || string.IsNullOrEmpty(payloadKey)) return false;
        if (!model.TryCheckpointActiveCache(payloadKey)) return false;
        footprint = model.MeasureEndState(payloadKey);
        return true;
    }

    /// <summary>
    /// <c>TryCaptureDonate</c> = <c>RetainSequenceCacheAs(requestId, key)</c>. A holder longer than
    /// <paramref name="length"/> is rewound first where the family's truncation reaches (the holder is
    /// re-keyed back, bound, truncated and retained again); otherwise the donation is refused and the
    /// holder stays the request's, so its release disposes it exactly as a refused retention does today.
    /// </summary>
    public static bool TryCaptureDonate(IHolderPrefixCacheModel model, string requestId, string payloadKey, int length, out PayloadFootprint footprint)
    {
        footprint = default;
        if (string.IsNullOrEmpty(requestId) || string.IsNullOrEmpty(payloadKey) || length <= 0) return false;
        if (!model.HasFusedSequenceCache(requestId)) return false;
        if (!model.RetainSequenceCacheAs(requestId, payloadKey)) return false;
        int held = model.MeasureEndState(payloadKey).Tokens;
        if (held != length && !TryRewindDonation(model, requestId, payloadKey, held, length))
        {
            if (!model.HasFusedSequenceCache(requestId) && !model.TryRebindRetainedCache(payloadKey, requestId))
                model.DiscardRetainedCaches(new[] { payloadKey }, ReleaseReason.Rollback);
            return false;
        }
        footprint = model.MeasureEndState(payloadKey);
        return true;
    }

    private static bool TryRewindDonation(IHolderPrefixCacheModel model, string requestId, string payloadKey, int held, int length)
    {
        if (held < length || !model.CanTruncateKVCache(held, length)) return false;
        if (!model.TryRebindRetainedCache(payloadKey, requestId)) return false;
        // Bound as an existing holder (never fresh: it was just re-keyed to this request).
        if (model.BindSequenceCache(requestId)) return false;
        if (!model.TryTruncateKVCache(length)) return false;
        return model.RetainSequenceCacheAs(requestId, payloadKey)
               && model.MeasureEndState(payloadKey).Tokens == length;
    }

    /// <summary>
    /// <c>TryMaterialize</c>: revalidated with <c>CanMaterialize</c> (P21); Clone =
    /// <c>SettleForCopy(key)</c> then <c>TryCloneRetainedCache(key, id)</c>; Donate =
    /// <c>TryRebindRetainedCache(key, id)</c>. A target below the payload's length is applied by the
    /// caller at the first bind (the payload itself is never truncated, P7).
    /// </summary>
    public static bool TryMaterialize(IHolderPrefixCacheModel model, in MaterializeRequest request)
    {
        if (string.IsNullOrEmpty(request.PayloadKey) || string.IsNullOrEmpty(request.TargetRequestId)) return false;
        if (!model.CanMaterialize(request.PayloadKey, request.PayloadTokens, request.TargetTokens)) return false;
        return request.Op switch
        {
            MaterializeOp.Clone => model.SettleForCopy(request.PayloadKey)
                                   && model.TryCloneRetainedCache(request.PayloadKey, request.TargetRequestId),
            MaterializeOp.Donate => model.TryRebindRetainedCache(request.PayloadKey, request.TargetRequestId),
            _ => false,
        };
    }

    /// <summary><c>TryImport</c> = <c>TryImportRetainedCache(key, stream)</c>; an import that does not hold
    /// exactly <paramref name="tokens"/> tokens is discarded and refused.</summary>
    public static bool TryImport(IHolderPrefixCacheModel model, string payloadKey, int tokens, Stream source, out PayloadFootprint footprint)
    {
        footprint = default;
        if (string.IsNullOrEmpty(payloadKey) || source == null || tokens <= 0) return false;
        if (!model.TryImportRetainedCache(payloadKey, source)) return false;
        PayloadFootprint imported = model.MeasureEndState(payloadKey);
        if (imported.Tokens != tokens)
        {
            model.DiscardRetainedCaches(new[] { payloadKey }, ReleaseReason.Invalidated);
            return false;
        }
        footprint = imported;
        return true;
    }

    /// <summary>
    /// <c>TryConvertPrimary</c> for a holder family: <c>AdoptPrimaryCacheToFused(key)</c> (zero copy; the
    /// model gives the primary a fresh allocation) then <c>RetainSequenceCacheAs(key, key)</c>. Refused
    /// before anything moves when a holder is checked out or the primary does not hold
    /// <paramref name="length"/> tokens.
    /// </summary>
    public static bool TryConvertPrimary(IHolderPrefixCacheModel model, string payloadKey, int length, out PayloadFootprint footprint)
    {
        footprint = default;
        if (string.IsNullOrEmpty(payloadKey) || length <= 0 || model.HasFusedSequenceCache(payloadKey)) return false;
        if (model is IPrefixCacheModelDiagnostics diagnostics && diagnostics.PrimaryCacheLength != length) return false;
        model.AdoptPrimaryCacheToFused(payloadKey);
        if (!model.HasFusedSequenceCache(payloadKey)) return false;   // a holder was checked out: nothing adopted
        if (!model.RetainSequenceCacheAs(payloadKey, payloadKey))
        {
            model.OnSequenceReleased(payloadKey);
            return false;
        }
        PayloadFootprint converted = model.MeasureEndState(payloadKey);
        if (converted.Tokens != length)
        {
            model.DiscardRetainedCaches(new[] { payloadKey }, ReleaseReason.Invalidated);
            return false;
        }
        footprint = converted;
        return true;
    }
}

/// <summary>
/// A family with no end states (class P parity, MuseGlimmer, Nemotron, HunyuanDense): every end-state and
/// lane member refuses, and releases are no-ops. The family supplies its capability record and
/// <see cref="IPrefixCacheModel.QuerySpareBytes"/>; its pages keep using the existing
/// <c>TryExtractKVBlock</c>/<c>TryInjectKVBlock</c> (A1) and <c>ForwardBatch</c> (A2) members.
/// </summary>
public interface IPageOnlyPrefixCacheModel : IPrefixCacheModel
{
    void IPrefixCacheModel.AttachPrefixCache(IPrefixPayloadSink sink) { }

    bool IPrefixCacheModel.TryCaptureCopy(string requestId, string payloadKey, out PayloadFootprint footprint)
    {
        footprint = default;
        return false;
    }

    bool IPrefixCacheModel.TryCaptureDonate(string requestId, string payloadKey, int length, out PayloadFootprint footprint)
    {
        footprint = default;
        return false;
    }

    bool IPrefixCacheModel.TryConvertPrimary(string payloadKey, int length, out PayloadFootprint footprint)
    {
        footprint = default;
        return false;
    }

    bool IPrefixCacheModel.TryMaterialize(in MaterializeRequest request) => false;
    bool IPrefixCacheModel.TryReturnDonation(string requestId, string payloadKey) => false;
    bool IPrefixCacheModel.CanMaterialize(string payloadKey, int payloadTokens, int targetTokens) => false;
    void IPrefixCacheModel.ReleasePayloads(ReadOnlySpan<string> payloadKeys, ReleaseReason reason) { }
    PayloadFootprint IPrefixCacheModel.MeasureEndState(string payloadKey) => default;
    ResourceVector IPrefixCacheModel.EstimateCloneBytes(string payloadKey, int targetTokens) => default;
    bool IPrefixCacheModel.TryCopyPagedToHolder(ReadOnlySpan<int> blockIds, int tokens, string requestId) => false;
    bool IPrefixCacheModel.TryExport(string payloadKey, Stream destination) => false;

    bool IPrefixCacheModel.TryImport(string payloadKey, int tokens, Stream source, out PayloadFootprint footprint)
    {
        footprint = default;
        return false;
    }

    bool IPrefixCacheModel.TryBeginImport(int tokens, out object? importTicket)
    {
        importTicket = null;
        return false;
    }

    bool IPrefixCacheModel.RunImportRead(object importTicket, Stream source) => false;

    bool IPrefixCacheModel.TryCommitImport(object importTicket, string payloadKey, out PayloadFootprint footprint)
    {
        footprint = default;
        return false;
    }

    void IPrefixCacheModel.AbortImport(object importTicket) { }
}
