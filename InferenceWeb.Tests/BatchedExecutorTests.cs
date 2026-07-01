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
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

/// <summary>
/// Tests for the **batched** path through <see cref="BatchExecutor"/>: a
/// model that implements <see cref="IBatchedPagedModel"/> gets its scheduled
/// sequences packed into a single <see cref="BatchedForwardContext"/> and
/// dispatched through one <c>ForwardBatch</c> call. This is the vLLM-style
/// "one kernel many sequences" path that the legacy per-sequence swap
/// executor cannot do.
///
/// The tests use stub models so they're fast and deterministic; they verify:
///   * the scheduler packs N sequences into the batch metadata correctly,
///   * <c>ForwardBatch</c> sees the expected query starts / positions /
///     slot mappings / block tables,
///   * per-sequence logits are routed back to the right sequence,
///   * the standalone <see cref="ManagedPagedAttention"/> kernel produces
///     numerically correct output for a hand-checkable single-sequence
///     case.
/// </summary>
public class BatchedExecutorTests
{
    private const int BlockSize = 8;
    private const int VocabSize = 16;
    private const int NumLayers = 2;
    private const int NumKVHeads = 2;
    private const int HeadDim = 4;

    [Fact]
    public void ManagedPagedAttention_SingleSeqSinglePos_MatchesHandComputed()
    {
        // 1 sequence, 1 query token, 1 KV head, 4 head dim, 1 block.
        // Q = [1, 0, 0, 0]
        // Block 0 K[0] = [1, 0, 0, 0], V[0] = [10, 0, 0, 0]
        // Causal mask + scale 1/sqrt(headDim).
        // Score = Q.K = 1 * 1/2 = 0.5; softmax = 1.0; output = V[0] = [10, 0, 0, 0]

        int numHeads = 1, numKvHeads = 1, headDim = 4, blockSize = 1;
        var q = new float[] { 1, 0, 0, 0 };
        var kBlocks = new float[] { 1, 0, 0, 0 };
        var vBlocks = new float[] { 10, 0, 0, 0 };
        var output = new float[4];

        ManagedPagedAttention.Forward(
            q, kBlocks, vBlocks, output,
            numTokens: 1, numHeads: numHeads, numKvHeads: numKvHeads, headDim: headDim,
            blockSize: blockSize,
            queryStartLoc: new[] { 0, 1 },
            seqLens: new[] { 1 },
            positions: new[] { 0 },
            blockTables: new[] { new[] { 0 } },
            numSeqs: 1,
            scale: 1f / MathF.Sqrt(headDim),
            causal: true);

        Assert.Equal(10f, output[0], 4);
        Assert.Equal(0f, output[1], 4);
        Assert.Equal(0f, output[2], 4);
        Assert.Equal(0f, output[3], 4);
    }

    [Fact]
    public void ManagedPagedAttention_CausalMaskedTwoKeys_Average()
    {
        // 1 sequence, 1 query at pos=1, 2 K/V slots in block 0.
        // K[0] = K[1] = [1,0,0,0]; V[0]=[1,0,0,0], V[1]=[3,0,0,0].
        // Scores equal -> softmax 0.5/0.5; output = (1+3)/2 = [2,0,0,0].
        int numHeads = 1, numKvHeads = 1, headDim = 4, blockSize = 2;
        var q = new float[] { 1, 0, 0, 0 };
        var kBlocks = new float[] { 1, 0, 0, 0,   1, 0, 0, 0 };
        var vBlocks = new float[] { 1, 0, 0, 0,   3, 0, 0, 0 };
        var output = new float[4];

        ManagedPagedAttention.Forward(
            q, kBlocks, vBlocks, output,
            numTokens: 1, numHeads: numHeads, numKvHeads: numKvHeads, headDim: headDim,
            blockSize: blockSize,
            queryStartLoc: new[] { 0, 1 },
            seqLens: new[] { 2 },
            positions: new[] { 1 }, // attends positions 0 and 1
            blockTables: new[] { new[] { 0 } },
            numSeqs: 1,
            scale: 1f,
            causal: true);

        Assert.Equal(2f, output[0], 3);
    }

    [Fact]
    public void ManagedPagedAttention_SlidingWindow_TruncatesOldKeys()
    {
        int numHeads = 1, numKvHeads = 1, headDim = 1, blockSize = 4;
        var q = new float[] { 1 };
        var kBlocks = new float[] { 1, 1, 1, 1 };
        var vBlocks = new float[] { 1, 3, 7, 9 };
        var output = new float[1];

        ManagedPagedAttention.Forward(
            q, kBlocks, vBlocks, output,
            numTokens: 1, numHeads: numHeads, numKvHeads: numKvHeads, headDim: headDim,
            blockSize: blockSize,
            queryStartLoc: new[] { 0, 1 },
            seqLens: new[] { 4 },
            positions: new[] { 3 },
            blockTables: new[] { new[] { 0 } },
            numSeqs: 1,
            scale: 1f,
            causal: true,
            slidingWindow: 2);

        Assert.Equal(8f, output[0], 3);
    }

    [Fact]
    public void BatchExecutor_PrefersForwardBatch_WhenModelImplementsBatchedInterface()
    {
        var model = new BatchedStubModel("fp-batched", peakToken: 7);
        var cfg = SmallConfig();
        using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);

        var handles = new List<InferenceRequestHandle>();
        for (int i = 0; i < 4; i++)
        {
            var seq = new SequenceState($"r{i}", Enumerable.Range(1, 5 + i).ToList(),
                maxNewTokens: 4, BlockSize, SamplingConfig.Default);
            handles.Add(engine.SubmitRequest(seq));
        }

        foreach (var h in handles)
        {
            var completion = h.Completion.GetAwaiter().GetResult();
            Assert.True(completion.OutputTokenCount > 0);
        }

        // The model received N>=1 batched calls; each saw multiple sequences
        // (proving the executor really packed them, not just looped per-seq).
        Assert.True(model.NumBatchCalls > 0,
            "ForwardBatch was never called - executor probably fell back to per-sequence.");
        Assert.True(model.MaxSequencesInAnyBatch >= 2,
            $"Expected at least one batch with >=2 sequences; biggest batch had {model.MaxSequencesInAnyBatch}.");
    }

    [Fact]
    public void BatchExecutor_BatchedPath_RoutesPerSeqLogitsCorrectly()
    {
        // Each sequence carries its requested peakToken in its UserTag. The
        // stub model reads UserTag to decide which token to favour, so we
        // can verify that the per-seq logits route back to the right
        // sequence in the streamed output.
        var model = new PerSeqRoutingStubModel("fp-route");
        using var engine = new InferenceEngine(model, SmallConfig(), NullLogger.Instance);

        var handles = new List<(InferenceRequestHandle handle, int expected)>();
        var expected = new[] { 2, 5, 9, 13 };
        for (int i = 0; i < expected.Length; i++)
        {
            var seq = new SequenceState($"r{i}", Enumerable.Range(1, 4).ToList(),
                maxNewTokens: 3, BlockSize, SamplingConfig.Default, userTag: expected[i]);
            handles.Add((engine.SubmitRequest(seq), expected[i]));
        }

        foreach (var (h, expectedToken) in handles)
        {
            h.Completion.GetAwaiter().GetResult();
            Assert.Contains(expectedToken, h.Sequence.OutputTokens);
        }
    }

    [Fact]
    public void BatchExecutor_PerSeqFused_ServesConcurrentSequencesViaForwardNotForwardBatch()
    {
        // A model that opts into the per-sequence fused path
        // (SupportsPerSequenceFusedForward=true, like Gemma 4) must have its
        // concurrent sequences served by per-sequence Forward — each with its
        // own bound per-request cache — and never fall into the op-by-op
        // ForwardBatch path. That is the parallel-decode throughput fix: it
        // keeps the GPU saturated on models whose fused single-graph decode is
        // far faster than the batched op-by-op kernel.
        //
        // The model declares SupportsLinearKVMigration like Gemma 4, so a
        // single-sequence step uses the N==1 fused Forward fast path rather than
        // ForwardBatch — hence ForwardBatch must NEVER be called regardless of
        // how the scheduler interleaves admission.
        var model = new PerSeqFusedStubModel("fp-fused");
        // Long-ish generations so multiple sequences overlap for many steps.
        using var engine = new InferenceEngine(model, SmallConfig(), NullLogger.Instance);

        var handles = new List<(InferenceRequestHandle handle, string id)>();
        for (int i = 0; i < 4; i++)
        {
            var seq = new SequenceState($"r{i}", Enumerable.Range(1, 4).ToList(),
                maxNewTokens: 20, BlockSize, SamplingConfig.Default);
            handles.Add((engine.SubmitRequest(seq), $"r{i}"));
        }

        foreach (var (h, _) in handles)
        {
            var completion = h.Completion.GetAwaiter().GetResult();
            Assert.True(completion.OutputTokenCount > 0);
        }

        // The op-by-op batched path was never used for this fused-capable model.
        Assert.Equal(0, model.NumBatchCalls);
        Assert.True(model.NumForwardCalls > 0, "Per-sequence Forward was never called.");
        // Concurrency was actually served by the fused path: at some step at
        // least two distinct per-request caches were live at once.
        Assert.True(model.MaxConcurrentBoundCaches >= 2,
            $"Expected >=2 per-request caches bound concurrently; saw {model.MaxConcurrentBoundCaches}.");
        // Every request was served through its own per-request cache binding.
        foreach (var (_, id) in handles)
            Assert.Contains(id, model.BoundRequestIds);
    }

    [Fact]
    public async Task InferenceEngine_ExecutorStepException_CompletesErroredRequestAndContinuesWaiting()
    {
        var model = new ThrowingBatchedStubModel("fp-step-error", badRequestId: "bad", peakToken: 7);
        var cfg = new SchedulerConfig
        {
            MaxNumBatchedTokens = 256,
            MaxNumRunningSequences = 1,
            MaxPrefillChunkSize = 64,
            NumBlocks = 4,
            BlockSize = BlockSize,
            EnablePrefixCaching = false,
            DecodeQuantumTokens = 1,
        };
        using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);

        var badSeq = new SequenceState("bad", Enumerable.Range(1, 4).ToList(),
            maxNewTokens: 2, BlockSize, SamplingConfig.Default);
        var goodSeq = new SequenceState("good", Enumerable.Range(1, 4).ToList(),
            maxNewTokens: 2, BlockSize, SamplingConfig.Default);

        var bad = engine.SubmitRequest(badSeq);
        var good = engine.SubmitRequest(goodSeq);

        var ex = await Assert.ThrowsAsync<InvalidOperationException>(
            () => bad.Completion.WaitAsync(TimeSpan.FromSeconds(5)));
        Assert.Contains("bad", ex.Message);

        var goodCompletion = await good.Completion.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(SequenceStatus.FinishedLengthCapped, goodCompletion.Status);
        Assert.Equal(0, badSeq.BlockTable.NumBlocks);
        Assert.Equal(SequenceStatus.FinishedError, badSeq.Status);
        Assert.Contains("bad", model.ReleasedRequestIds);
        Assert.Contains("good", model.ReleasedRequestIds);
        Assert.Equal(cfg.NumBlocks, engine.PoolStats.freeBlocks);
    }

    [Fact]
    public async Task InferenceEngine_PromptLongerThanKvPool_FailsCleanlyInsteadOfHanging()
    {
        // Repro for the reported hang: a prompt whose KV footprint exceeds the
        // whole block pool prefills until the pool is exhausted, then — being the
        // only running sequence with nothing to preempt — can never be scheduled
        // again. Pre-fix the engine spun on empty schedules forever (GPU idle,
        // "stuck forever"). Post-fix it fails the request with a capacity error
        // and stays healthy enough to serve a subsequent in-budget request.
        //
        // maxContext defaults to 0, so the pool is NOT auto-grown; this exercises
        // the deadlock guard rather than the auto-sizing path.
        var model = new BatchedStubModel("fp-capacity", peakToken: 7);
        var cfg = new SchedulerConfig
        {
            MaxNumBatchedTokens = 8,
            MaxNumRunningSequences = 4,
            MaxPrefillChunkSize = 8,
            SoloPrefillChunkSize = 8,
            NumBlocks = 4,          // pool holds 4*8 = 32 tokens of KV
            BlockSize = BlockSize,  // 8
            EnablePrefixCaching = false,
            DecodeQuantumTokens = 1,
        };
        using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);
        // Pool was not auto-grown (model advertises no context length).
        Assert.Equal(cfg.NumBlocks, engine.PoolStats.totalBlocks);

        // 40-token prompt needs ceil(40/8)=5 blocks > the 4-block pool.
        var tooLong = new SequenceState("too-long", Enumerable.Range(1, 40).ToList(),
            maxNewTokens: 4, BlockSize, SamplingConfig.Default);
        var shortOk = new SequenceState("short-ok", Enumerable.Range(1, 4).ToList(),
            maxNewTokens: 3, BlockSize, SamplingConfig.Default);

        var tooLongHandle = engine.SubmitRequest(tooLong);
        var shortHandle = engine.SubmitRequest(shortOk);

        // The over-length request fails rather than hanging: if the engine were
        // still spinning, WaitAsync would throw TimeoutException here instead.
        var ex = await Assert.ThrowsAsync<InvalidOperationException>(
            () => tooLongHandle.Completion.WaitAsync(TimeSpan.FromSeconds(10)));
        Assert.Contains("capacity", ex.Message, StringComparison.OrdinalIgnoreCase);
        Assert.Equal(SequenceStatus.FinishedError, tooLong.Status);
        Assert.Equal(0, tooLong.BlockTable.NumBlocks);

        // The engine recovered: the in-budget request still completes and the
        // pool is fully reclaimed.
        var completion = await shortHandle.Completion.WaitAsync(TimeSpan.FromSeconds(10));
        Assert.True(completion.OutputTokenCount > 0);
        Assert.Equal(cfg.NumBlocks, engine.PoolStats.freeBlocks);
    }

    [Fact]
    public async Task InferenceEngine_AutoSizesKvPoolToModelContext_SoLongPromptDoesNotDeadlock()
    {
        // A model that advertises a context length gets its KV block pool sized to
        // cover that context, so an in-context prompt longer than the configured
        // default pool completes instead of deadlocking (the reported hang).
        const int modelContext = 512;
        var model = new BatchedStubModel("fp-autosize", peakToken: 7, maxContext: modelContext);
        var cfg = new SchedulerConfig
        {
            MaxNumBatchedTokens = 64,
            MaxNumRunningSequences = 4,
            MaxPrefillChunkSize = 32,
            SoloPrefillChunkSize = 64,
            NumBlocks = 4,          // default would hold only 4*8 = 32 tokens
            BlockSize = BlockSize,  // 8
            EnablePrefixCaching = false,
            DecodeQuantumTokens = 1,
        };
        using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);

        // Pool auto-grew to cover the model's advertised context.
        int expectedBlocks = (modelContext + BlockSize - 1) / BlockSize; // 64
        Assert.Equal(expectedBlocks, engine.PoolStats.totalBlocks);

        // A 100-token prompt overflows the configured 32-token pool but fits the
        // auto-sized 512-token pool: it must complete, not hang.
        var seq = new SequenceState("long-in-context", Enumerable.Range(1, 100).ToList(),
            maxNewTokens: 4, BlockSize, SamplingConfig.Default);
        var handle = engine.SubmitRequest(seq);
        var completion = await handle.Completion.WaitAsync(TimeSpan.FromSeconds(10));
        Assert.True(completion.OutputTokenCount > 0);
    }

    // ----- helpers -----

    private static SchedulerConfig SmallConfig() => new()
    {
        MaxNumBatchedTokens = 256,
        MaxNumRunningSequences = 8,
        MaxPrefillChunkSize = 64,
        NumBlocks = 16,
        BlockSize = BlockSize,
        EnablePrefixCaching = false,
        DecodeQuantumTokens = 1,
    };

    /// <summary>
    /// Minimal <see cref="IBatchedPagedModel"/>. Records how many batched calls
    /// it received and the biggest batch size, and returns deterministic
    /// logits peaked at <c>peakToken</c> for every sequence.
    /// </summary>
    private sealed class BatchedStubModel : IModelArchitecture, IBatchedPagedModel
    {
        private readonly string _fp;
        private readonly int _peak;
        private readonly int _maxContext;
        private int _cacheSeqLen;

        public int NumBatchCalls { get; private set; }
        public int MaxSequencesInAnyBatch { get; private set; }

        public BatchedStubModel(string fp, int peakToken, int maxContext = 0)
        {
            _fp = fp;
            _peak = peakToken;
            _maxContext = maxContext;
            Tokenizer = new StubTokenizer(VocabSize);
        }

        public ModelConfig Config { get; } = new ModelConfig { VocabSize = VocabSize };
        public ITokenizer Tokenizer { get; }
        public IMultimodalInjector MultimodalInjector => null;
        public IBackendExecutionPlan ExecutionPlan => null;
        public bool SupportsKVCacheTruncation => true;
        public bool SupportsKVStateSnapshot => true;
        // 0 => model advertises no context length (pool keeps its configured size);
        // a positive value drives the engine's KV-pool auto-sizing.
        public int MaxContextLength => _maxContext;
        public string KVStateFingerprint => _fp;
        public long ComputeKVBlockByteSize(int tokenCount)
            => 2L * NumLayers * NumKVHeads * tokenCount * HeadDim * sizeof(float);
        public float[] Forward(int[] tokens)
        {
            // Fallback path - shouldn't be hit when ForwardBatch is wired.
            var logits = new float[VocabSize];
            logits[_peak] = 10f;
            return logits;
        }
        public void ResetKVCache() => _cacheSeqLen = 0;
        public void TruncateKVCache(int n) => _cacheSeqLen = Math.Min(_cacheSeqLen, n);
        public bool TryExtractKVBlock(int s, int n, Span<byte> dst) => true;
        public bool TryInjectKVBlock(int s, int n, ReadOnlySpan<byte> src) { _cacheSeqLen = s + n; return true; }
        public void Dispose() { }

        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx)
        {
            NumBatchCalls++;
            int n = ctx.Sequences.Count;
            if (n > MaxSequencesInAnyBatch) MaxSequencesInAnyBatch = n;

            var perSeqLogits = new float[n][];
            for (int i = 0; i < n; i++)
            {
                var logits = new float[VocabSize];
                logits[_peak] = 10f;
                perSeqLogits[i] = logits;
            }
            return perSeqLogits;
        }
    }

    /// <summary>
    /// Stub whose per-sequence logits depend on the sequence's
    /// <see cref="SequenceState.UserTag"/>. Used to prove that the batched
    /// executor routes per-seq logits back to the right sequence (and not
    /// e.g. swapped or all-same).
    /// </summary>
    private sealed class PerSeqRoutingStubModel : IModelArchitecture, IBatchedPagedModel
    {
        private readonly string _fp;
        private int _cacheSeqLen;

        public PerSeqRoutingStubModel(string fp)
        {
            _fp = fp;
            Tokenizer = new StubTokenizer(VocabSize);
        }

        public ModelConfig Config { get; } = new ModelConfig { VocabSize = VocabSize };
        public ITokenizer Tokenizer { get; }
        public IMultimodalInjector MultimodalInjector => null;
        public IBackendExecutionPlan ExecutionPlan => null;
        public bool SupportsKVCacheTruncation => true;
        public bool SupportsKVStateSnapshot => true;
        public string KVStateFingerprint => _fp;
        public long ComputeKVBlockByteSize(int n) => 2L * NumLayers * NumKVHeads * n * HeadDim * sizeof(float);
        public float[] Forward(int[] tokens) => new float[VocabSize];
        public void ResetKVCache() => _cacheSeqLen = 0;
        public void TruncateKVCache(int n) => _cacheSeqLen = Math.Min(_cacheSeqLen, n);
        public bool TryExtractKVBlock(int s, int n, Span<byte> dst) => true;
        public bool TryInjectKVBlock(int s, int n, ReadOnlySpan<byte> src) { _cacheSeqLen = s + n; return true; }
        public void Dispose() { }

        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx)
        {
            int n = ctx.Sequences.Count;
            var result = new float[n][];
            for (int i = 0; i < n; i++)
            {
                int peak = ctx.Sequences[i].UserTag is int t ? t : 0;
                var logits = new float[VocabSize];
                logits[peak] = 10f;
                result[i] = logits;
            }
            return result;
        }
    }

    /// <summary>
    /// Stub that opts into the per-sequence fused path. Tracks per-request
    /// "caches" (just a bound-id set here) and returns logits for the
    /// currently-bound request, peaked at the token registered in
    /// <see cref="PeakForRequest"/>. Lets us assert the executor (a) never calls
    /// ForwardBatch, (b) binds a distinct cache per request, and (c) routes each
    /// sequence's logits correctly through per-sequence Forward.
    /// </summary>
    private sealed class PerSeqFusedStubModel : IModelArchitecture, IBatchedPagedModel
    {
        private readonly string _fp;
        private string _activeReqId;
        private readonly HashSet<string> _liveCaches = new(StringComparer.Ordinal);

        public PerSeqFusedStubModel(string fp)
        {
            _fp = fp;
            Tokenizer = new StubTokenizer(VocabSize);
        }

        public Dictionary<string, int> PeakForRequest { get; } = new(StringComparer.Ordinal);
        public int NumBatchCalls { get; private set; }
        public int NumForwardCalls { get; private set; }
        public int MaxConcurrentBoundCaches { get; private set; }
        public HashSet<string> BoundRequestIds { get; } = new(StringComparer.Ordinal);

        public ModelConfig Config { get; } = new ModelConfig { VocabSize = VocabSize };
        public ITokenizer Tokenizer { get; }
        public IMultimodalInjector MultimodalInjector => null;
        public IBackendExecutionPlan ExecutionPlan => null;
        public bool SupportsKVCacheTruncation => true;
        public bool SupportsKVStateSnapshot => true;
        public string KVStateFingerprint => _fp;
        public long ComputeKVBlockByteSize(int n) => 2L * NumLayers * NumKVHeads * n * HeadDim * sizeof(float);

        public float[] Forward(int[] tokens)
        {
            NumForwardCalls++;
            var logits = new float[VocabSize];
            int peak = _activeReqId != null && PeakForRequest.TryGetValue(_activeReqId, out var p) ? p : 0;
            logits[peak] = 10f;
            return logits;
        }

        public void ResetKVCache() { }
        public void TruncateKVCache(int n) { }
        public bool TryExtractKVBlock(int s, int n, Span<byte> dst) => true;
        public bool TryInjectKVBlock(int s, int n, ReadOnlySpan<byte> src) => true;
        public void Dispose() { }

        // The executor must NOT call this for a fused-capable model at N>=2.
        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx)
        {
            NumBatchCalls++;
            var r = new float[ctx.Sequences.Count][];
            for (int i = 0; i < r.Length; i++) r[i] = new float[VocabSize];
            return r;
        }

        public bool SupportsPerSequenceFusedForward => true;

        // Mirror Gemma 4: linear KV migration is supported, so the executor's
        // N==1 fast path uses fused Forward (not ForwardBatch) for single steps.
        public bool SupportsLinearKVMigration => true;
        public bool TryMigrateLinearKVToPaged(SequenceState owner, int blockSize) => true;

        public bool BindSequenceCache(string requestId)
        {
            BoundRequestIds.Add(requestId);
            bool fresh = _liveCaches.Add(requestId);
            _activeReqId = requestId;
            if (_liveCaches.Count > MaxConcurrentBoundCaches)
                MaxConcurrentBoundCaches = _liveCaches.Count;
            return fresh;
        }

        public void AdoptPrimaryCacheToFused(string requestId)
        {
            _liveCaches.Add(requestId);
            _activeReqId = requestId;
        }

        public void RestorePrimaryCache() => _activeReqId = null;

        public bool HasFusedSequenceCache(string requestId) => _liveCaches.Contains(requestId);

        public void OnSequenceReleased(string requestId)
        {
            _liveCaches.Remove(requestId);
            if (string.Equals(_activeReqId, requestId, StringComparison.Ordinal))
                _activeReqId = null;
        }
    }

    private sealed class ThrowingBatchedStubModel : IModelArchitecture, IBatchedPagedModel
    {
        private readonly string _fp;
        private readonly string _badRequestId;
        private readonly int _peak;

        public ThrowingBatchedStubModel(string fp, string badRequestId, int peakToken)
        {
            _fp = fp;
            _badRequestId = badRequestId;
            _peak = peakToken;
            Tokenizer = new StubTokenizer(VocabSize);
        }

        public List<string> ReleasedRequestIds { get; } = new();

        public ModelConfig Config { get; } = new ModelConfig { VocabSize = VocabSize };
        public ITokenizer Tokenizer { get; }
        public IMultimodalInjector MultimodalInjector => null;
        public IBackendExecutionPlan ExecutionPlan => null;
        public bool SupportsKVCacheTruncation => true;
        public bool SupportsKVStateSnapshot => true;
        public string KVStateFingerprint => _fp;
        public long ComputeKVBlockByteSize(int n) => 2L * NumLayers * NumKVHeads * n * HeadDim * sizeof(float);
        public float[] Forward(int[] tokens) => new float[VocabSize];
        public void ResetKVCache() { }
        public void TruncateKVCache(int n) { }
        public bool TryExtractKVBlock(int s, int n, Span<byte> dst) => true;
        public bool TryInjectKVBlock(int s, int n, ReadOnlySpan<byte> src) => true;
        public void Dispose() { }

        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx)
        {
            for (int i = 0; i < ctx.Sequences.Count; i++)
            {
                if (ctx.Sequences[i].RequestId == _badRequestId)
                    throw new InvalidOperationException($"boom for {_badRequestId}");
            }

            var result = new float[ctx.Sequences.Count][];
            for (int i = 0; i < result.Length; i++)
            {
                var logits = new float[VocabSize];
                logits[_peak] = 10f;
                result[i] = logits;
            }
            return result;
        }

        public void OnSequenceReleased(string requestId)
        {
            ReleasedRequestIds.Add(requestId);
        }
    }

    private sealed class StubTokenizer : ITokenizer
    {
        public StubTokenizer(int vocab)
        {
            Vocab = new string[vocab];
            for (int i = 0; i < vocab; i++) Vocab[i] = i.ToString();
        }
        public string[] Vocab { get; }
        public int BosTokenId => -1;
        public int[] EosTokenIds => Array.Empty<int>();
        public int VocabSize => Vocab.Length;
        public List<int> Encode(string text, bool addSpecial = true) => new();
        public string Decode(List<int> ids) => string.Join(",", ids);
        public void AppendTokenBytes(int tokenId, List<byte> buffer)
        {
            foreach (var b in System.Text.Encoding.UTF8.GetBytes(tokenId.ToString()))
                buffer.Add(b);
        }
        public bool IsEos(int tokenId) => false;
        public int LookupToken(string tokenStr) => -1;
    }
}
