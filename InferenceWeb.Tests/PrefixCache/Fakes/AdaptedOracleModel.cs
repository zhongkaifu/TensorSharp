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
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests.PrefixCache.Fakes;

/// <summary>
/// A holder oracle that gets its <see cref="IPrefixCacheModel"/> end-state members from the Runtime's
/// <see cref="HolderPrefixCacheAdapter"/> instead of implementing them, exactly as Gemma 4, Qwen 3.5,
/// Qwen4Exp and DeepSeek V4.1 do. It supplies only what a real family supplies: the retained-holder
/// lifecycle of <see cref="IBatchedPagedModel"/> (forwarded to the oracle), the key-parameterised
/// retain, the batched discard, the settle and the capability/measurement members. Running the
/// conformance script on it checks the adapter against the oracle's independent implementation.
/// </summary>
internal sealed class AdaptedOracleModel : IHolderPrefixCacheModel, IPrefixCacheModelDiagnostics
{
    private readonly bool _settles;

    internal AdaptedOracleModel(OracleModel inner, bool settles = true)
    {
        Inner = inner;
        _settles = settles;
    }

    internal OracleModel Inner { get; }

    // ---- the family additions (DESIGN §4.8) ----

    public bool RetainSequenceCacheAs(string requestId, string payloadKey) => Inner.RetainAs(requestId, payloadKey);

    public void DiscardRetainedCaches(ReadOnlySpan<string> payloadKeys, ReleaseReason reason) => Inner.ReleasePayloads(payloadKeys, reason);

    /// <summary>With <c>settles: false</c> this is the bug the settle exists to prevent: the holder stays
    /// device-authoritative and the oracle refuses the copy.</summary>
    public bool SettleForCopy(string payloadKey) => _settles ? Inner.Settle(payloadKey) : Inner.RetainedPayloadKeys.Contains(payloadKey);

    public PrefixCacheCapabilities GetPrefixCacheCapabilities() => Inner.GetPrefixCacheCapabilities();
    public void AttachPrefixCache(IPrefixPayloadSink sink) => Inner.AttachPrefixCache(sink);
    public bool TryConvertPrimary(string payloadKey, int length, out PayloadFootprint footprint)
        => HolderPrefixCacheAdapter.TryConvertPrimary(this, payloadKey, length, out footprint);
    public bool CanMaterialize(string payloadKey, int payloadTokens, int targetTokens) => Inner.CanMaterialize(payloadKey, payloadTokens, targetTokens);
    public PayloadFootprint MeasureEndState(string payloadKey) => Inner.MeasureEndState(payloadKey);
    public ResourceVector EstimateCloneBytes(string payloadKey, int targetTokens) => Inner.EstimateCloneBytes(payloadKey, targetTokens);
    public long QuerySpareBytes(ResourceClass cls) => Inner.QuerySpareBytes(cls);

    // ---- diagnostics ----

    public IReadOnlyCollection<string> RetainedPayloadKeys => Inner.RetainedPayloadKeys;
    public int PrivateHolderCount => Inner.PrivateHolderCount;
    public int PrimaryCacheLength => Inner.PrimaryCacheLength;
    public long DecodeGraphResets => Inner.DecodeGraphResets;

    // ---- IModelArchitecture ----

    public ModelConfig Config => Inner.Config;
    public ITokenizer Tokenizer => Inner.Tokenizer;
    public IMultimodalInjector MultimodalInjector => Inner.MultimodalInjector;
    public IBackendExecutionPlan ExecutionPlan => Inner.ExecutionPlan;
    public float[] Forward(int[] tokens) => Inner.Forward(tokens);
    public void ResetKVCache() => Inner.ResetKVCache();
    public bool SupportsKVCacheTruncation => Inner.SupportsKVCacheTruncation;
    public void TruncateKVCache(int tokenCount) => Inner.TruncateKVCache(tokenCount);
    public bool TryTruncateKVCache(int tokenCount) => Inner.TryTruncateKVCache(tokenCount);
    public int KVCacheTruncationGranularity => Inner.KVCacheTruncationGranularity;
    public bool CanTruncateKVCache(int cachedTokenCount, int targetTokenCount) => Inner.CanTruncateKVCache(cachedTokenCount, targetTokenCount);
    public bool SupportsKVStateSnapshot => Inner.SupportsKVStateSnapshot;
    public bool SupportsCrossSequenceKvReuse => Inner.SupportsCrossSequenceKvReuse;
    public int MaxReusablePrefixTokens => Inner.MaxReusablePrefixTokens;
    public bool SupportsReuseAcrossMediaSpan => Inner.SupportsReuseAcrossMediaSpan;
    public int MaxContextLength => Inner.MaxContextLength;
    public string KVStateFingerprint => Inner.KVStateFingerprint;
    public bool RequiresPerBlockCapture => Inner.RequiresPerBlockCapture;
    public long ComputeKVBlockByteSize(int tokenCount) => Inner.ComputeKVBlockByteSize(tokenCount);
    public bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination) => Inner.TryExtractKVBlock(startToken, tokenCount, destination);
    public bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source) => Inner.TryInjectKVBlock(destToken, tokenCount, source);
    public void Dispose() => Inner.Dispose();

    // ---- IBatchedPagedModel ----

    public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx) => Inner.ForwardBatch(ctx);
    public bool BatchedForwardAvailable => Inner.BatchedForwardAvailable;
    public bool SupportsLinearKVMigration => Inner.SupportsLinearKVMigration;
    public bool TryMigrateLinearKVToPaged(SequenceState owner, int blockSize) => Inner.TryMigrateLinearKVToPaged(owner, blockSize);
    public void OnSequenceReleased(string requestId) => Inner.OnSequenceReleased(requestId);
    public bool SupportsPerSequenceFusedForward => Inner.SupportsPerSequenceFusedForward;
    public bool BindSequenceCache(string requestId) => Inner.BindSequenceCache(requestId);
    public void AdoptPrimaryCacheToFused(string requestId) => Inner.AdoptPrimaryCacheToFused(requestId);
    public void RestorePrimaryCache() => Inner.RestorePrimaryCache();
    public bool HasFusedSequenceCache(string requestId) => Inner.HasFusedSequenceCache(requestId);
    public bool SupportsRetainedFusedCache => Inner.SupportsRetainedFusedCache;
    public bool RetainSequenceCache(string requestId) => Inner.RetainSequenceCache(requestId);
    public bool TryRebindRetainedCache(string retainedRequestId, string newRequestId) => Inner.TryRebindRetainedCache(retainedRequestId, newRequestId);
    public void DiscardRetainedCache(string requestId) => Inner.DiscardRetainedCache(requestId);
    public bool SupportsPrefixCheckpoints => Inner.SupportsPrefixCheckpoints;
    public bool TryCheckpointActiveCache(string key) => Inner.TryCheckpointActiveCache(key);
    public bool TryCloneRetainedCache(string retainedKey, string newRequestId) => Inner.TryCloneRetainedCache(retainedKey, newRequestId);
    public bool SupportsRetainedCacheSerialization => Inner.SupportsRetainedCacheSerialization;
    public bool TryExportRetainedCache(string key, Stream destination) => Inner.TryExportRetainedCache(key, destination);
    public bool TryImportRetainedCache(string key, Stream source) => Inner.TryImportRetainedCache(key, source);
}
