// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// Regression tests for cross-request KV-cache prefix reuse on the per-sequence
/// FUSED concurrent-decode path (the high-throughput path a sliding-window model
/// like Gemma 4 takes for N&gt;=2 concurrent requests).
///
/// Bug: that path keeps each request's full K/V in its own per-request holder and
/// never writes the shared paged blocks, so a finished concurrent request left
/// NOTHING in the prefix-cache pool — and a sliding-window model's pool can't
/// restore a long prefix anyway. A multi-turn follow-up ("请继续") submitted after
/// a concurrent round therefore re-prefilled the whole conversation from scratch
/// (KV-cache reuse ratio 0). The fix retains a small LRU of finished fused holders
/// and re-adopts one for a follow-up whose prompt exactly extends it.
///
/// Uses a deterministic <see cref="FusedStubModel"/> that mimics a sliding-window
/// model on the fused path (per-request holders, capped pooled reuse) without a
/// real LLM, so the scheduler/executor/engine glue is exercised end-to-end.
/// </summary>
public class RetainedFusedCacheTests
{
    private const int BlockSize = 8;
    private const int VocabSize = 16;
    private const int Cap = 16;         // sliding-window cap (pooled reuse ceiling)
    private const int PeakToken = 3;    // greedy argmax always lands here

    [Fact]
    public async Task ConcurrentRound_ThenParallelFollowUps_ReuseFullPrefix()
    {
        // ---- Reproduce the bug: retention OFF -> follow-ups get 0 reuse. ----
        var (offA, offB) = await RunTwoRoundsAsync(retentionEnabled: false);
        Assert.Equal(0, offA.PrefixCacheReusedTokens);
        Assert.Equal(0, offB.PrefixCacheReusedTokens);

        // ---- Verify the fix: retention ON -> follow-ups reuse the whole prefix. ----
        var (onA, onB) = await RunTwoRoundsAsync(retentionEnabled: true);

        // Each follow-up's prompt = round-1 (prompt+output) + a short suffix, so the
        // reused prefix must equal the entire retained conversation (well past Cap).
        Assert.True(onA.PrefixCacheReusedTokens > Cap,
            $"follow-up A reused {onA.PrefixCacheReusedTokens} tokens (expected > {Cap})");
        Assert.True(onB.PrefixCacheReusedTokens > Cap,
            $"follow-up B reused {onB.PrefixCacheReusedTokens} tokens (expected > {Cap})");
        Assert.Equal(onA.PromptTokenCount - SuffixLen, onA.PrefixCacheReusedTokens);
        Assert.Equal(onB.PromptTokenCount - SuffixLen, onB.PrefixCacheReusedTokens);

        // High-performance check: reuse ratio is near-total (only the short new
        // suffix is re-prefilled), i.e. the multi-turn follow-up no longer pays to
        // recompute the whole conversation.
        double pctA = 100.0 * onA.PrefixCacheReusedTokens / onA.PromptTokenCount;
        double pctB = 100.0 * onB.PrefixCacheReusedTokens / onB.PromptTokenCount;
        Assert.True(pctA >= 80.0, $"follow-up A reuse {pctA:F1}% too low");
        Assert.True(pctB >= 80.0, $"follow-up B reuse {pctB:F1}% too low");
    }

    /// <summary>
    /// Single-stream ("请继续") analogue of the bug above, on a model whose KV cache
    /// is BLOCK-QUANTIZED (q4_0 / q8_0). Such a model declines the batched paged path
    /// — its <see cref="IBatchedPagedModel.ForwardBatch"/> throws NotSupported and it
    /// reports <c>SupportsLinearKVMigration=false</c> — but it still keeps a single
    /// live linear cache exactly like the f16 path (<see cref="FusedStubModel"/>
    /// matches this shape: ForwardBatch throws, SupportsPerSequenceFusedForward=true,
    /// SupportsLinearKVMigration defaults to false).
    ///
    /// Repro of the user report ("--kv-cache-dtype q4_0 makes KV cache reuse 0 on a
    /// follow-up turn; f16 reuses fully"): pre-fix, a block-quant N=1 step skipped the
    /// N=1 fast path (gated on SupportsLinearKVMigration) and fell into the
    /// ExecuteStepBatched attempt, which cleared <c>_liveCacheValid</c> BEFORE
    /// ForwardBatch threw. The per-seq fallback's EnsureOwnership then saw the
    /// stale-false flag and aborted the live-cache continuation, re-prefilling the
    /// whole conversation (PrefixCacheReusedTokens reset to 0). f16 took the fast path
    /// and never tripped that flag, hence the dtype-specific symptom.
    /// </summary>
    [Fact]
    public async Task SingleStream_BlockQuantLikeModel_LiveCacheContinuation_ReusesFullPrefix()
    {
        var model = new FusedStubModel();
        using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);

        // Turn 1: ONE sequence, prompt longer than the pooled-reuse Cap so only
        // live-cache continuation (not the capped pool) can reuse it on turn 2.
        var prompt1 = Enumerable.Repeat(1, PromptLen).ToList();
        var seq1 = new SequenceState("t1", prompt1, Round1NewTokens, BlockSize, SamplingConfig.Greedy);
        var (_, out1) = await DrainAsync(engine.SubmitRequest(seq1));

        // Turn 2: "请继续" — prompt = turn-1 (prompt + output) + a short new suffix.
        // Submitted only AFTER turn 1 fully drained, so the whole conversation runs
        // single-stream (N=1) and exercises live-cache continuation, not the
        // concurrent retained-fused path.
        var prompt2 = new List<int>(prompt1);
        prompt2.AddRange(out1);
        prompt2.AddRange(Enumerable.Repeat(PeakToken, SuffixLen));
        var seq2 = new SequenceState("t2", prompt2, 8, BlockSize, SamplingConfig.Greedy);
        var (c2, _) = await DrainAsync(engine.SubmitRequest(seq2));

        // The reused prefix must equal the entire turn-1 conversation (well past Cap):
        // only the short new suffix is re-prefilled. Pre-fix this was 0.
        Assert.True(c2.PrefixCacheReusedTokens > Cap,
            $"single-stream follow-up reused {c2.PrefixCacheReusedTokens} tokens " +
            $"(expected > {Cap}); reuse 0 is the reported q4_0 bug.");
        Assert.Equal(c2.PromptTokenCount - SuffixLen, c2.PrefixCacheReusedTokens);

        double pct = 100.0 * c2.PrefixCacheReusedTokens / c2.PromptTokenCount;
        Assert.True(pct >= 80.0, $"single-stream follow-up reuse {pct:F1}% too low");
    }

    private const int PromptLen = 24;   // > Cap so only the live holder can reuse it
    private const int Round1NewTokens = 24;
    private const int SuffixLen = 4;

    private async Task<(InferenceCompletion a, InferenceCompletion b)> RunTwoRoundsAsync(bool retentionEnabled)
    {
        string prev = Environment.GetEnvironmentVariable("TS_RETAINED_FUSED_CACHE");
        Environment.SetEnvironmentVariable("TS_RETAINED_FUSED_CACHE", retentionEnabled ? "1" : "0");
        try
        {
            var model = new FusedStubModel();
            using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);

            // ---- Round 1: two distinct conversations, submitted in parallel. ----
            var promptA = Enumerable.Repeat(1, PromptLen).ToList();
            var promptB = Enumerable.Repeat(2, PromptLen).ToList();
            var seqA1 = new SequenceState("A1", promptA, Round1NewTokens, BlockSize, SamplingConfig.Greedy);
            var seqB1 = new SequenceState("B1", promptB, Round1NewTokens, BlockSize, SamplingConfig.Greedy);

            // Submit BOTH before draining so the engine admits them together (N=2)
            // and serves them through the per-sequence fused path.
            var hA1 = engine.SubmitRequest(seqA1);
            var hB1 = engine.SubmitRequest(seqB1);
            var rA1 = DrainAsync(hA1);
            var rB1 = DrainAsync(hB1);
            await Task.WhenAll(rA1, rB1);
            var (_, outA1) = rA1.Result;
            var (_, outB1) = rB1.Result;

            // ---- Round 2: "请继续" — each follow-up extends its own conversation. ----
            var followA = new List<int>(promptA);
            followA.AddRange(outA1);
            followA.AddRange(Enumerable.Repeat(PeakToken, SuffixLen));
            var followB = new List<int>(promptB);
            followB.AddRange(outB1);
            followB.AddRange(Enumerable.Repeat(PeakToken, SuffixLen));

            var seqA2 = new SequenceState("A2", followA, 8, BlockSize, SamplingConfig.Greedy);
            var seqB2 = new SequenceState("B2", followB, 8, BlockSize, SamplingConfig.Greedy);
            var hA2 = engine.SubmitRequest(seqA2);
            var hB2 = engine.SubmitRequest(seqB2);
            var rA2 = DrainAsync(hA2);
            var rB2 = DrainAsync(hB2);
            await Task.WhenAll(rA2, rB2);
            return (rA2.Result.completion, rB2.Result.completion);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_RETAINED_FUSED_CACHE", prev);
        }
    }

    private static async Task<(InferenceCompletion completion, List<int> output)> DrainAsync(InferenceRequestHandle handle)
    {
        var output = new List<int>();
        await foreach (var t in handle.Tokens.ReadAllAsync())
            output.Add(t);
        var completion = await handle.Completion;
        return (completion, output);
    }

    private static SchedulerConfig Config() => new()
    {
        MaxNumBatchedTokens = 1024,
        MaxNumRunningSequences = 8,
        MaxPrefillChunkSize = 256,
        SoloPrefillChunkSize = 256,
        NumBlocks = 256,
        BlockSize = BlockSize,
        EnablePrefixCaching = true,
        DecodeQuantumTokens = 1, // rotate eagerly so both round-1 seqs interleave
    };

    /// <summary>
    /// Deterministic stub that mimics a sliding-window model on the per-sequence
    /// fused path: each RequestId gets its own (in-memory) K/V holder, pooled reuse
    /// is capped at <see cref="Cap"/>, and finished holders can be retained and
    /// re-keyed. Forward only tracks a per-holder token count; logits always peak at
    /// <see cref="PeakToken"/> so greedy decode is deterministic.
    /// </summary>
    private sealed class FusedStubModel : IModelArchitecture, IBatchedPagedModel
    {
        private sealed class Holder { public int SeqLen; }

        private readonly Dictionary<string, Holder> _holders = new(StringComparer.Ordinal);
        private readonly Dictionary<string, Holder> _retained = new(StringComparer.Ordinal);
        private string _activeKey;            // null => primary active
        private Holder _primary = new();

        private Holder Active => _activeKey == null ? _primary : _holders[_activeKey];

        public FusedStubModel() => Tokenizer = new StubTokenizer();

        public ModelConfig Config { get; } = new ModelConfig { VocabSize = VocabSize };
        public ITokenizer Tokenizer { get; }
        public IMultimodalInjector MultimodalInjector => null;
        public IBackendExecutionPlan ExecutionPlan => null;
        public bool SupportsKVCacheTruncation => true;

        // The fused path never reads paged storage, but the engine still sizes the
        // block pool from this, so it must be > 0.
        public long ComputeKVBlockByteSize(int tokenCount) => 32L * tokenCount;

        public float[] Forward(int[] tokens)
        {
            Active.SeqLen += tokens.Length;
            var logits = new float[VocabSize];
            logits[PeakToken] = 10.0f;
            return logits;
        }

        public void ResetKVCache() => Active.SeqLen = 0;
        public void TruncateKVCache(int tokenCount) => Active.SeqLen = Math.Min(Active.SeqLen, tokenCount);
        public void Dispose() { }

        // Sliding-window model: snapshot fine for own decode, capped cross-seq reuse.
        public bool SupportsKVStateSnapshot => true;
        public bool SupportsCrossSequenceKvReuse => true;
        public int MaxReusablePrefixTokens => Cap;
        public string KVStateFingerprint => "fused-stub";

        // ---- IBatchedPagedModel: per-sequence fused forward + retention ----
        public bool SupportsPerSequenceFusedForward => true;

        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx)
            => throw new NotSupportedException("fused stub only serves the per-sequence fused path");

        public bool BindSequenceCache(string requestId)
        {
            if (string.Equals(_activeKey, requestId, StringComparison.Ordinal)) return false;
            bool fresh;
            if (!_holders.TryGetValue(requestId, out var h)) { h = new Holder(); _holders[requestId] = h; fresh = true; }
            else fresh = false;
            _activeKey = requestId;
            return fresh;
        }

        public void AdoptPrimaryCacheToFused(string requestId)
        {
            if (_activeKey != null || _holders.ContainsKey(requestId)) return;
            _holders[requestId] = _primary; // hand the live primary state to the holder
            _activeKey = requestId;
            _primary = new Holder();
        }

        public void RestorePrimaryCache()
        {
            // Holders are referenced objects in _holders, so just repoint to primary.
            if (_activeKey != null) _activeKey = null;
        }

        public bool HasFusedSequenceCache(string requestId) => _holders.ContainsKey(requestId);

        public void OnSequenceReleased(string requestId)
        {
            if (string.Equals(_activeKey, requestId, StringComparison.Ordinal)) _activeKey = null;
            _holders.Remove(requestId);
        }

        public bool RetainSequenceCache(string requestId)
        {
            if (!_holders.TryGetValue(requestId, out var h)) return false;
            if (string.Equals(_activeKey, requestId, StringComparison.Ordinal)) _activeKey = null;
            _holders.Remove(requestId);
            _retained[requestId] = h;
            return true;
        }

        public bool TryRebindRetainedCache(string retainedRequestId, string newRequestId)
        {
            if (!_retained.TryGetValue(retainedRequestId, out var h)) return false;
            _retained.Remove(retainedRequestId);
            _holders[newRequestId] = h;
            return true;
        }

        public void DiscardRetainedCache(string requestId) => _retained.Remove(requestId);

        private sealed class StubTokenizer : ITokenizer
        {
            public StubTokenizer()
            {
                Vocab = new string[RetainedFusedCacheTests.VocabSize];
                for (int i = 0; i < RetainedFusedCacheTests.VocabSize; i++) Vocab[i] = i.ToString();
            }
            public string[] Vocab { get; }
            public int BosTokenId => -1;
            public int[] EosTokenIds => Array.Empty<int>();
            public int VocabSize => Vocab.Length;
            public List<int> Encode(string text, bool addSpecial = true) => new();
            public string Decode(List<int> ids) => string.Join(",", ids);
            public void AppendTokenBytes(int tokenId, List<byte> buffer)
            {
                foreach (var b in System.Text.Encoding.UTF8.GetBytes(tokenId.ToString())) buffer.Add(b);
            }
            public bool IsEos(int tokenId) => false;
            public int LookupToken(string tokenStr) => -1;
        }
    }
}
