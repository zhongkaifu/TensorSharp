// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

/// <summary>
/// Unit tests for the new paged KV pool, the continuous-batching scheduler,
/// the executor and the engine glue. They use a deterministic
/// <see cref="StubModel"/> that produces predictable logits and supports the
/// KV-snapshot contract, so we can drive the engine without loading a real
/// LLM.
/// </summary>
public class ContinuousBatchSchedulerTests
{
    private const int BlockSize = 8;
    private const int VocabSize = 16;
    private const int NumLayers = 2;
    private const int NumKVHeads = 2;
    private const int HeadDim = 4;

    [Fact]
    public void BlockPool_AllocateAndFree_RoundTrips()
    {
        var pool = NewPool(numBlocks: 4);
        var blocks = pool.AllocateNew(2);
        Assert.NotNull(blocks);
        Assert.Equal(2, blocks.Length);
        Assert.Equal(2, pool.NumFreeBlocks);

        foreach (var b in blocks)
            Assert.Equal(1, b.RefCount);

        pool.Free(blocks);
        Assert.Equal(4, pool.NumFreeBlocks);
        foreach (var b in blocks)
            Assert.Equal(0, b.RefCount);
    }

    [Fact]
    public void BlockPool_RefCountedShare_IsThreadSafeInSerial()
    {
        var pool = NewPool(numBlocks: 2);
        var b = pool.AllocateNew(1)[0];
        Assert.Equal(1, b.RefCount);
        pool.Touch(b);
        Assert.Equal(2, b.RefCount);
        pool.Free(b);
        Assert.Equal(1, b.RefCount);
        Assert.Equal(1, pool.NumFreeBlocks);
        pool.Free(b);
        Assert.Equal(0, b.RefCount);
        Assert.Equal(2, pool.NumFreeBlocks);
    }

    [Fact]
    public void BlockPool_RegistersHash_EnablesPrefixHit()
    {
        var pool = NewPool(numBlocks: 4);
        var b = pool.AllocateNew(1)[0];
        var hash = KvBlockHasher.ComputeBlockHashes(
            Enumerable.Range(0, BlockSize).ToList(), BlockSize, "fp")[0];
        pool.RegisterFullBlock(b, hash, BlockSize);

        Assert.True(pool.TryFindByHash(hash, out var found));
        Assert.Same(b, found);
    }

    [Fact]
    public void BlockTable_AdvanceTokens_TracksUsedBlocks()
    {
        var pool = NewPool(numBlocks: 4);
        var seq = new SequenceState("r0", new[] { 1, 2, 3, 4, 5 }, maxNewTokens: 5, BlockSize, SamplingConfig.Default);
        var blocks = pool.AllocateNew(1);
        seq.BlockTable.AppendBlock(blocks[0]);
        seq.AdvanceComputedTokens(5);
        Assert.Equal(5, seq.NumComputedTokens);
        Assert.Equal(BlockSize, seq.BlockTable.CapacityTokens);
        Assert.Equal(BlockSize - 5, seq.BlockTable.FreeSlotsInCurrentBlocks);
    }

    [Fact]
    public void Scheduler_AdmitsWaitingSequence_AllocatesBlocks()
    {
        var pool = NewPool(numBlocks: 4);
        var sched = NewScheduler(pool, "fp");
        var seq = NewSequence("r0", promptLen: 12, maxNew: 4);
        sched.Submit(seq);

        var step = sched.Schedule();
        Assert.Single(step.ScheduledWork);
        Assert.Equal(seq, step.ScheduledWork[0].Sequence);
        Assert.True(step.ScheduledWork[0].IsPrefill);
        Assert.True(step.ScheduledWork[0].NumScheduledTokens > 0);
        Assert.Equal(SequenceStatus.Running, seq.Status);
        Assert.True(seq.BlockTable.NumBlocks >= 2);
    }

    [Fact]
    public void Scheduler_MultipleSequences_RunConcurrently()
    {
        var pool = NewPool(numBlocks: 16);
        var sched = NewScheduler(pool, "fp");

        var seqs = new List<SequenceState>();
        for (int i = 0; i < 3; i++)
        {
            var s = NewSequence($"r{i}", promptLen: 5 + i, maxNew: 4);
            sched.Submit(s);
            seqs.Add(s);
        }

        var step = sched.Schedule();
        Assert.Equal(3, step.ScheduledWork.Count);
        Assert.All(seqs, s => Assert.Equal(SequenceStatus.Running, s.Status));
    }

    [Fact]
    public void Engine_DriveOneSequence_ProducesTokens()
    {
        var model = new StubModel("fp-x", peakToken: 7);
        using var engine = new InferenceEngine(model, SmallConfig(), NullLogger.Instance);

        var seq = NewSequence("r0", promptLen: 10, maxNew: 5);
        var handle = engine.SubmitRequest(seq);

        var tokens = ReadAll(handle, timeoutMs: 5000);
        Assert.True(tokens.Count > 0);
        // The stub model always picks token 7, so the output should contain 7s
        // (until EOS or max).
        Assert.Contains(7, tokens);

        var completion = handle.Completion.GetAwaiter().GetResult();
        Assert.True(completion.OutputTokenCount > 0);
    }

    [Fact]
    public void Engine_ParallelSequences_AllComplete()
    {
        var model = new StubModel("fp-par", peakToken: 7);
        using var engine = new InferenceEngine(model, SmallConfig(), NullLogger.Instance);

        var handles = new List<InferenceRequestHandle>();
        for (int i = 0; i < 4; i++)
        {
            var seq = NewSequence($"r{i}", promptLen: 6 + i, maxNew: 4);
            handles.Add(engine.SubmitRequest(seq));
        }

        foreach (var h in handles)
        {
            var completion = h.Completion.GetAwaiter().GetResult();
            Assert.True(completion.OutputTokenCount > 0,
                $"Sequence {h.RequestId} produced no tokens.");
        }
    }

    [Fact]
    public void Engine_RespectsMaxTokens()
    {
        var model = new StubModel("fp-cap", peakToken: 5);
        using var engine = new InferenceEngine(model, SmallConfig(), NullLogger.Instance);
        var seq = NewSequence("r0", promptLen: 4, maxNew: 3);
        var handle = engine.SubmitRequest(seq);

        var completion = handle.Completion.GetAwaiter().GetResult();
        Assert.Equal(3, completion.OutputTokenCount);
        Assert.Equal(SequenceStatus.FinishedLengthCapped, completion.Status);
    }

    [Fact]
    public void Engine_StopsOnEos()
    {
        var model = new StubModel("fp-eos", peakToken: 0); // 0 is EOS
        model.SetEos(0);
        using var engine = new InferenceEngine(model, SmallConfig(), NullLogger.Instance);
        var seq = NewSequence("r0", promptLen: 4, maxNew: 10);
        var handle = engine.SubmitRequest(seq);

        var completion = handle.Completion.GetAwaiter().GetResult();
        Assert.Equal(SequenceStatus.FinishedStopped, completion.Status);
        Assert.Equal("eos", completion.FinishReason);
    }

    [Fact]
    public void Engine_PrefixCacheHit_ReducesUncomputedTokens()
    {
        var model = new StubModel("fp-prefix", peakToken: 3);
        var cfg = SmallConfig();
        using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);

        // Prime the cache: run a request that produces full blocks.
        var promptA = Enumerable.Range(10, BlockSize * 2).ToArray();
        var seqA = NewSequenceFromTokens("rA", promptA, maxNew: 2);
        engine.SubmitRequest(seqA).Completion.GetAwaiter().GetResult();

        // Same prompt comes in again on a NEW sequence.
        var seqB = NewSequenceFromTokens("rB", promptA, maxNew: 2);
        var handleB = engine.SubmitRequest(seqB);
        handleB.Completion.GetAwaiter().GetResult();
        // We expect at least one block worth of prefix-cache reuse.
        Assert.True(seqB.PrefixCacheReusedTokens >= BlockSize,
            $"Expected >= {BlockSize} prefix-cache tokens, got {seqB.PrefixCacheReusedTokens}");
    }

    [Fact]
    public void TrimComputedToTotalTokens_RestoresInvariantAfterTailTruncation()
    {
        // Mirrors what the engine does after a mid-batch speculative stop:
        // the step advanced the committed count over the whole forwarded
        // batch, then the output tail was truncated, leaving the committed
        // count ahead of the token list.
        var seq = NewSequence("r0", promptLen: BlockSize, maxNew: 64);
        var pool = NewPool(numBlocks: 4);
        var blocks = pool.AllocateNew(3);
        foreach (var b in blocks) seq.BlockTable.AppendBlock(b);

        seq.AdvanceComputedTokens(BlockSize);          // prompt committed (block 0 full)
        for (int i = 0; i < 4; i++) seq.AppendOutputToken(200 + i);
        seq.AdvanceComputedTokens(4);                  // speculative batch: computed=12, total=12

        // Engine truncates the output tail to a single surviving token.
        seq.OutputTokens.RemoveRange(seq.OutputTokens.Count - 3, 3); // total=9, computed=12
        Assert.True(seq.NumComputedTokens > seq.NumTotalTokens);

        seq.TrimComputedToTotalTokens();
        Assert.Equal(seq.NumTotalTokens, seq.NumComputedTokens);
    }

    [Fact]
    public void NotifyStop_AfterMidBatchSpeculativeStop_DoesNotThrow()
    {
        // Regression for the --mtp-spec crash: a speculative step advances
        // NumComputedTokens past a generated block boundary, then a mid-batch
        // EOS truncates the output tail so NumComputedTokens > NumTotalTokens.
        // FinishSequence -> CacheFullBlocksForSequence used to index TokenAt()
        // past the end of the (shortened) token list and throw
        // ArgumentOutOfRangeException(pos). This drives the scheduler's
        // defense-in-depth clamp WITHOUT the engine's trim to prove the
        // finish path is crash-safe on its own.
        var pool = NewPool(numBlocks: 8);
        var sched = NewScheduler(pool, "fp");

        var seq = NewSequence("r0", promptLen: BlockSize, maxNew: 64); // prompt fills block 0
        var blocks = pool.AllocateNew(3);                              // 24-token capacity
        foreach (var b in blocks) seq.BlockTable.AppendBlock(b);

        seq.AdvanceComputedTokens(BlockSize);          // prompt committed: computed=8, total=8
        for (int i = 0; i < 6; i++) seq.AppendOutputToken(100 + i);
        seq.AdvanceComputedTokens(6);                  // decode to position 14

        // Speculative step forwards [sampled, d1, d2, d3] (all 3 drafts
        // accepted): 4 output tokens appended, committed count advanced 4 ->
        // crosses the block-1 boundary at 16.
        for (int i = 0; i < 4; i++) seq.AppendOutputToken(200 + i);
        seq.AdvanceComputedTokens(4);                  // computed=18, total=18

        // The sampled (first) token was EOS: keep only it, drop the 3 drafts.
        seq.OutputTokens.RemoveRange(seq.OutputTokens.Count - 3, 3); // total=15, computed=18
        Assert.True(seq.NumComputedTokens > seq.NumTotalTokens);

        var output = new SchedulerOutput();
        var ex = Record.Exception(
            () => sched.NotifyStop(seq, SequenceStatus.FinishedStopped, "eos", output));
        Assert.Null(ex);
        Assert.Contains("r0", output.FinishedRequestIds);
        Assert.True(seq.Status.IsFinished());
    }

    // ----------------------- helpers -----------------------

    private static BlockPool NewPool(int numBlocks)
    {
        long blockBytes = 2L * NumLayers * NumKVHeads * BlockSize * HeadDim * sizeof(float);
        return new BlockPool(numBlocks, BlockSize, blockBytes);
    }

    private static ContinuousBatchScheduler NewScheduler(BlockPool pool, string fp)
    {
        var cfg = new SchedulerConfig
        {
            MaxNumBatchedTokens = 1024,
            MaxNumRunningSequences = 16,
            MaxPrefillChunkSize = 64,
            NumBlocks = pool.NumBlocks,
            BlockSize = BlockSize,
            EnablePrefixCaching = true,
            DecodeQuantumTokens = BlockSize,
        };
        return new ContinuousBatchScheduler(cfg, pool, fp, NullLogger.Instance);
    }

    private static SchedulerConfig SmallConfig() => new()
    {
        MaxNumBatchedTokens = 256,
        MaxNumRunningSequences = 8,
        MaxPrefillChunkSize = 64,
        NumBlocks = 16,
        BlockSize = BlockSize,
        EnablePrefixCaching = true,
        DecodeQuantumTokens = BlockSize,
    };

    private static SequenceState NewSequence(string id, int promptLen, int maxNew)
    {
        var tokens = Enumerable.Range(1, promptLen).ToList();
        return new SequenceState(id, tokens, maxNew, BlockSize, SamplingConfig.Default);
    }

    private static SequenceState NewSequenceFromTokens(string id, int[] tokens, int maxNew)
        => new(id, tokens.ToList(), maxNew, BlockSize, SamplingConfig.Default);

    private static List<int> ReadAll(InferenceRequestHandle handle, int timeoutMs)
    {
        var list = new List<int>();
        var deadline = DateTime.UtcNow.AddMilliseconds(timeoutMs);
        while (DateTime.UtcNow < deadline)
        {
            if (handle.Tokens.TryRead(out var t))
            {
                list.Add(t);
                continue;
            }
            if (handle.Completion.IsCompleted) break;
            handle.Tokens.WaitToReadAsync().AsTask().Wait(50);
        }
        return list;
    }

    /// <summary>
    /// Deterministic stub model used in the engine tests above. Returns logits
    /// that always peak at a configurable token id, supports the full
    /// KV-snapshot contract (so the executor can swap state between
    /// sequences), and exposes a no-op tokenizer that recognises a single EOS.
    /// </summary>
    private sealed class StubModel : IModelArchitecture
    {
        private readonly string _fp;
        private readonly int _peak;
        private byte[] _state = Array.Empty<byte>();
        private int _cacheSeqLen;
        private int _eos = -1;

        public StubModel(string fingerprint, int peakToken)
        {
            _fp = fingerprint;
            _peak = peakToken;
            Tokenizer = new StubTokenizer(VocabSize, this);
        }

        public ModelConfig Config { get; } = new ModelConfig { VocabSize = VocabSize };
        public ITokenizer Tokenizer { get; }
        public IMultimodalInjector MultimodalInjector => null;
        public IBackendExecutionPlan ExecutionPlan => null;
        public bool SupportsKVCacheTruncation => true;

        public void SetEos(int eosId) => _eos = eosId;
        public int CurrentSeqLen => _cacheSeqLen;

        public float[] Forward(int[] tokens)
        {
            // Pretend to write KV state for each input token.
            int newCount = _cacheSeqLen + tokens.Length;
            long needed = ComputeKVBlockByteSize(newCount);
            if (_state.Length < needed) Array.Resize(ref _state, (int)needed);
            for (int t = 0; t < tokens.Length; t++)
            {
                long start = ComputeKVBlockByteSize(_cacheSeqLen + t);
                long end = ComputeKVBlockByteSize(_cacheSeqLen + t + 1);
                byte v = (byte)(tokens[t] & 0xFF);
                for (long i = start; i < end; i++)
                    _state[i] = v;
            }
            _cacheSeqLen += tokens.Length;

            var logits = new float[VocabSize];
            logits[_peak] = 10.0f;
            return logits;
        }

        public void ResetKVCache() => _cacheSeqLen = 0;
        public void TruncateKVCache(int tokenCount) => _cacheSeqLen = Math.Min(_cacheSeqLen, tokenCount);
        public void Dispose() { }

        public bool SupportsKVStateSnapshot => true;
        public string KVStateFingerprint => _fp;

        public long ComputeKVBlockByteSize(int tokenCount)
            => 2L * NumLayers * NumKVHeads * tokenCount * HeadDim * sizeof(float);

        public bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        {
            long expected = ComputeKVBlockByteSize(tokenCount);
            if (destination.Length != expected) return false;
            if (startToken < 0 || startToken + tokenCount > _cacheSeqLen) return false;
            long startByte = ComputeKVBlockByteSize(startToken);
            new ReadOnlySpan<byte>(_state, (int)startByte, (int)expected).CopyTo(destination);
            return true;
        }

        public bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source)
        {
            if (destToken != _cacheSeqLen) return false;
            long expected = ComputeKVBlockByteSize(tokenCount);
            if (source.Length != expected) return false;
            long needed = ComputeKVBlockByteSize(destToken + tokenCount);
            if (_state.Length < needed) Array.Resize(ref _state, (int)needed);
            long startByte = ComputeKVBlockByteSize(destToken);
            source.CopyTo(new Span<byte>(_state, (int)startByte, (int)expected));
            _cacheSeqLen = destToken + tokenCount;
            return true;
        }

        private sealed class StubTokenizer : ITokenizer
        {
            private readonly StubModel _owner;
            public StubTokenizer(int vocab, StubModel owner)
            {
                _owner = owner;
                Vocab = new string[vocab];
                for (int i = 0; i < vocab; i++) Vocab[i] = i.ToString();
            }
            public string[] Vocab { get; }
            public int BosTokenId => -1;
            public int[] EosTokenIds => _owner._eos >= 0 ? new[] { _owner._eos } : Array.Empty<int>();
            public int VocabSize => Vocab.Length;
            public List<int> Encode(string text, bool addSpecial = true) => new();
            public string Decode(List<int> ids) => string.Join(",", ids);
            public void AppendTokenBytes(int tokenId, List<byte> buffer)
            {
                foreach (var b in System.Text.Encoding.UTF8.GetBytes(tokenId.ToString()))
                    buffer.Add(b);
            }
            public bool IsEos(int tokenId) => _owner._eos == tokenId;
            public int LookupToken(string tokenStr) => -1;
        }
    }
}
