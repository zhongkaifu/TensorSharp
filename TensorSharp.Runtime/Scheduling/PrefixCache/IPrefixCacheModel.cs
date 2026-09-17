// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;

namespace TensorSharp.Runtime.Scheduling.PrefixCache;

/// <summary>How a cached end state reaches a request (§4.8).</summary>
public enum MaterializeOp : byte
{
    /// <summary>Copy the payload into the request's private holder; the payload stays cached.</summary>
    Clone,
    /// <summary>Re-key the payload to the request, zero copy; the payload leaves the cache.</summary>
    Donate,
}

/// <summary>Why a model dropped a payload behind the tree's back (DEC-23).</summary>
public enum InvalidationReason : byte { ModelBudgetEviction, NativeSlotReclaimed, BackendRecreated, TrimmedByModel }

/// <summary>
/// One materialization. <see cref="TargetTokens"/> below <see cref="PayloadTokens"/> means the
/// caller truncates at the first bind (the executor's pending truncation); the model never
/// truncates a payload in <see cref="IPrefixCacheModel.TryMaterialize"/> (P7).
/// </summary>
public readonly record struct MaterializeRequest(
    MaterializeOp Op, string PayloadKey, string TargetRequestId,
    int PayloadTokens, int TargetTokens);

/// <summary>Where a model reports the payloads it still evicts itself (DEC-23).</summary>
public interface IPrefixPayloadSink
{
    /// <summary>Worker thread only. MUST NOT mutate the tree; the coordinator appends to its
    /// deferred queue and applies the invalidation before its next tree read.</summary>
    void OnPayloadInvalidated(string payloadKey, InvalidationReason reason);
}

/// <summary>
/// The per-family prefix-cache contract (DESIGN §4.8). A family declares its rules in
/// <see cref="PrefixCacheCapabilities"/> and implements only the state operations below; the
/// Runtime evaluates the rules. Payload keys are tree-minted (<c>pc:{engine}:{serial}</c>) and
/// never equal a request id (I10).
///
/// <para>Threading: every member runs on the engine worker thread, between steps, except
/// <see cref="RunImportRead"/> (a lane thread, M7c).</para>
///
/// <para>Inert until attached: in <see cref="PrefixCacheMode.Legacy"/> and
/// <see cref="PrefixCacheMode.Shadow"/> the engine calls nothing here but
/// <see cref="GetPrefixCacheCapabilities"/> (I26), and a model behaves exactly as it did
/// before it implemented this interface until <see cref="AttachPrefixCache"/> is called.</para>
/// </summary>
public interface IPrefixCacheModel
{
    /// <summary>The capability record; read once at engine construction.</summary>
    PrefixCacheCapabilities GetPrefixCacheCapabilities();

    /// <summary>Tree mode only. The model stops self-evicting to fit its own budgets (it
    /// refuses instead) and reports any eviction it still performs through the sink.</summary>
    void AttachPrefixCache(IPrefixPayloadSink sink);

    // ---- end states ----

    /// <summary>Deep copy of the request's ACTIVE cache (bound holder or primary) at its
    /// current length. Flushes device-authoritative state first (P20). The result is
    /// host-authoritative. The request's cache must be the active one.</summary>
    bool TryCaptureCopy(string requestId, string payloadKey, out PayloadFootprint footprint);

    /// <summary>Move a finished or preempted request's private holder (or native slot) under
    /// <paramref name="payloadKey"/>. A <paramref name="length"/> below the holder's length
    /// truncates first, only where the family's truncation allows it. Zero copy.</summary>
    bool TryCaptureDonate(string requestId, string payloadKey, int length, out PayloadFootprint footprint);

    /// <summary>PrimaryResident conversion: adopt the primary cache and retain it under
    /// <paramref name="payloadKey"/>, zero copy of K/V (the model allocates a fresh primary).
    /// Only when <see cref="PrefixCacheCapabilities.AdoptPrimaryOnDisplacement"/>.</summary>
    bool TryConvertPrimary(string payloadKey, int length, out PayloadFootprint footprint);

    /// <summary>Clone: copy the payload into the request's private holder, settling a
    /// device-dirty payload first. Donate: re-key payload → request. Neither binds.</summary>
    bool TryMaterialize(in MaterializeRequest request);

    /// <summary>Admission rollback of a Donate: re-key request → payload (the holder was never bound).</summary>
    bool TryReturnDonation(string requestId, string payloadKey);

    /// <summary>Side-effect free. False means the payload is invalid for this target (P21).</summary>
    bool CanMaterialize(string payloadKey, int payloadTokens, int targetTokens);

    /// <summary>Batched and idempotent (unknown keys are ignored). Recycles into holder pools
    /// where possible and issues at most one decode-graph reset per call (DEC-24).</summary>
    void ReleasePayloads(ReadOnlySpan<string> payloadKeys, ReleaseReason reason);

    /// <summary>What a cached payload holds and charges. <c>default</c> for an unknown key.</summary>
    PayloadFootprint MeasureEndState(string payloadKey);

    /// <summary>Bytes a clone of <paramref name="payloadKey"/> truncated to
    /// <paramref name="targetTokens"/> would allocate (the pre-clone budget check).</summary>
    ResourceVector EstimateCloneBytes(string payloadKey, int targetTokens);

    // ---- pages (class P, M7a) ----

    /// <summary>Copy model-paged (A2) rows of <paramref name="blockIds"/> into a fresh private
    /// holder of <paramref name="requestId"/> covering <paramref name="tokens"/> tokens.</summary>
    bool TryCopyPagedToHolder(ReadOnlySpan<int> blockIds, int tokens, string requestId);

    // ---- persistence (Persistable only) ----

    bool TryExport(string payloadKey, System.IO.Stream destination);

    /// <summary>Create <paramref name="payloadKey"/> from an export of the same model; false and
    /// nothing created when the bytes do not describe <paramref name="tokens"/> tokens of this model.</summary>
    bool TryImport(string payloadKey, int tokens, System.IO.Stream source, out PayloadFootprint footprint);

    // ---- lanes (M7c; OffThreadImportRead only) ----

    /// <summary>Worker: allocate an unbound import destination.</summary>
    bool TryBeginImport(int tokens, out object? importTicket);
    /// <summary>Lane thread: host memcpy only.</summary>
    bool RunImportRead(object importTicket, System.IO.Stream source);
    /// <summary>Worker: publish the imported destination under <paramref name="payloadKey"/>.</summary>
    bool TryCommitImport(object importTicket, string payloadKey, out PayloadFootprint footprint);
    /// <summary>Worker: free an import that will not be committed.</summary>
    void AbortImport(object importTicket);

    // ---- memory ----

    /// <summary>Spare bytes of <paramref name="cls"/>; −1 when unknown.</summary>
    long QuerySpareBytes(ResourceClass cls);
}

/// <summary>Debug and test visibility into a model's payload ownership (I10, I14, I22).</summary>
public interface IPrefixCacheModelDiagnostics
{
    /// <summary>Keys of every cached payload the model holds (retained holders, checkpoints, native slots).</summary>
    IReadOnlyCollection<string> RetainedPayloadKeys { get; }
    /// <summary>Per-request private holders (or native slots) that are not cached payloads.</summary>
    int PrivateHolderCount { get; }
    /// <summary>Tokens in the primary (single-stream) cache.</summary>
    int PrimaryCacheLength { get; }
    /// <summary>Decode-graph resets issued since load (<c>ModelBase.DecodeGraphResets</c>).</summary>
    long DecodeGraphResets { get; }
}
