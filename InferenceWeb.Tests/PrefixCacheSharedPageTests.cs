// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// A family whose prefix cache keeps a shared prompt prefix only as pages that resume where a forward ended
// (PrefixCacheCapabilities.PagesNeedStateAtEnd, the recurrent families' rule) shares it only from a page a
// per-sequence forward ended at. Two ways the prefix went unshared: a first request that reached the engine alone
// with a prompt running past the shared prefix split its forward only before its own last block, so every page of
// the shared prefix was cached without state (its forward now also ends at the shared page,
// PrefixCacheCoordinator.SharedPrefixPageBoundary); and a restore point that arrived after such a stateless page was
// dropped as a duplicate of it (it now replaces the stateless page). The scheduler's lone-page-producer hold, which
// served such a family with a batched route and a linear-to-paged migration (the legacy Nemotron-H and GPT-OSS
// models), went with the migration: no family has both any more.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests;

public sealed class PrefixCacheSharedPageTests
{
    private const int BlockSize = 8;
    private const int VocabSize = 16;
    private const int PeakToken = 3;
    // The system prompt every request starts with, and the shared prefix the chat layer marks.
    private const int SharedTokens = 42;
    // The page boundary the requests can share: the shared prefix, block-aligned.
    private const int SharedPage = SharedTokens - SharedTokens % BlockSize;
    private const int TailTokens = 6;

    [Fact]
    public async Task Engine_ALoneFirstPromptPastTheSharedPrefix_LeavesARestorePointThere()
    {
        // The first request reached the idle engine alone, so nothing waited for it, and its prompt
        // ran past the shared prefix. Its forward was split only before its own last block (56), so
        // it cached every page of the shared prefix without state: the burst behind it found no page
        // to resume (an A40 run whose first request was the 2067-token one, behind a 1890-token
        // system prompt: kvReused=0 on every request of every repeat). Its forward now also ends at
        // the shared page, and the whole burst resumes there.
        var model = new RecurrentPageStubModel();
        using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);
        var gate = new ComputeGate();
        engine.ComputeGate = gate;

        var first = Long("long", SharedTokens);
        await RunAlone(engine, first);
        int tailSplit = LongPromptTokens - LongPromptTokens % BlockSize;
        Assert.Equal(new[] { SharedPage, tailSplit - SharedPage, LongPromptTokens - tailSplit }, model.LoneForwards);

        var burst = await RunBurst(engine, gate, 3);
        Assert.All(burst, s => Assert.Equal(SharedPage, s.PrefixCacheReusedTokens));
        int promptTokens = SharedTokens + TailTokens;
        Assert.Equal(LongPromptTokens + 3 * (promptTokens - SharedPage), model.PromptTokensForwarded);
    }

    [Fact]
    public async Task Engine_ARestorePointAtTheSharedPage_ReplacesTheStatelessPageCachedFirst()
    {
        // A request of a longer system prompt variant (a 50-token shared prefix, page 48) cached the
        // page of the 42-token ancestor without state. The next lone request of the shorter variant
        // ended its forward exactly at that page, but the tree kept the stateless page that arrived
        // first and dropped the restore point as a duplicate - and no request reused the prefix
        // again until the page was evicted. The restore point now replaces it.
        var model = new RecurrentPageStubModel();
        using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);
        int variantShared = SharedPage + BlockSize + 2;
        await RunAlone(engine, Long("variant", variantShared));
        Assert.True(model.LoneForwards[0] > SharedPage, "the variant's restore point lies past the shared page");

        var producer = Shared("r1", 0);
        await RunAlone(engine, producer);
        Assert.Equal(0, producer.PrefixCacheReusedTokens);
        var later = Shared("r2", 1);
        await RunAlone(engine, later);
        Assert.Equal(SharedPage, later.PrefixCacheReusedTokens);
    }

    private static SchedulerConfig Config(int maxRunning = 8) => new()
    {
        MaxNumBatchedTokens = 256,
        MaxNumRunningSequences = maxRunning,
        MaxPrefillChunkSize = 64,
        NumBlocks = 128,
        BlockSize = BlockSize,
        EnablePrefixCaching = true,
        DecodeQuantumTokens = BlockSize,
    };

    private static (ContinuousBatchScheduler, BatchExecutor, BlockPool) NewScheduler(
        RecurrentPageStubModel model, SchedulerConfig? config = null)
    {
        var cfg = config ?? Config();
        var pool = new BlockPool(cfg.NumBlocks, cfg.BlockSize, model.ComputeKVBlockByteSize(cfg.BlockSize));
        var sched = new ContinuousBatchScheduler(cfg, pool, NullLogger.Instance,
            requiresPerBlockCapture: true);
        var executor = new BatchExecutor(model, pool, sched, NullLogger.Instance);
        executor.InitializeRadixCache(cfg);
        return (sched, executor, pool);
    }

    // A prompt whose own message runs past the shared page: alone, its last full forward ends at 56.
    private const int LongPromptTokens = SharedTokens + 20;

    /// <summary>A chat turn with a long message after the system prompt, which this request
    /// declares <paramref name="sharedPrefixTokens"/> tokens long.</summary>
    private static SequenceState Long(string id, int sharedPrefixTokens)
    {
        var tokens = Enumerable.Range(1, SharedTokens)
            .Concat(Enumerable.Range(300, LongPromptTokens - SharedTokens))
            .ToList();
        return new SequenceState(id, tokens, 2, BlockSize, SamplingConfig.Greedy,
            sharedPrefixTokens: sharedPrefixTokens);
    }

    /// <summary>Run <paramref name="seq"/> on the idle engine, alone, to completion.</summary>
    private static async Task RunAlone(InferenceEngine engine, SequenceState seq)
    {
        var handle = engine.SubmitRequest(seq);
        var completion = await handle.Completion.WaitAsync(TimeSpan.FromSeconds(30));
        Assert.Null(seq.Error);
        Assert.Equal(SequenceStatus.FinishedLengthCapped, completion.Status);
    }

    /// <summary>A burst of <paramref name="count"/> chat turns that reaches the engine before its
    /// next step, run to completion.</summary>
    private static async Task<List<SequenceState>> RunBurst(InferenceEngine engine, ComputeGate gate, int count)
    {
        gate.Close();
        var burst = Enumerable.Range(0, count).Select(i => Shared($"burst{i}", i)).ToList();
        var handles = burst.Select(s => engine.SubmitRequest(s)).ToList();
        gate.Open();
        foreach (var handle in handles)
        {
            var completion = await handle.Completion.WaitAsync(TimeSpan.FromSeconds(30));
            Assert.Null(handle.Sequence.Error);
            Assert.Equal(SequenceStatus.FinishedLengthCapped, completion.Status);
        }
        return burst;
    }

    /// <summary>A chat turn: the shared system prompt, then this request's own message.</summary>
    private static SequenceState Shared(string id, int index, IRequestAdapter? adapter = null)
    {
        var tokens = Enumerable.Range(1, SharedTokens)
            .Concat(Enumerable.Range(100 + index * TailTokens, TailTokens))
            .ToList();
        return new SequenceState(id, tokens, 2, BlockSize, SamplingConfig.Greedy,
            sharedPrefixTokens: SharedTokens, adapter: adapter);
    }

    private sealed class StubAdapter : IRequestAdapter
    {
        public string Id => "style#1";
        public bool IsRetired => false;
    }

    /// <summary>
    /// A recurrent page family, as the legacy Nemotron-H: the linear cache holds token rows and one
    /// running state, so an extracted block carries the state at the end of the forward that
    /// preceded it, and a forward refuses to run on a state that does not match the rows (a resume
    /// from a block that is no restore point). It has no batched route: every request runs per sequence.
    /// </summary>
    private sealed class RecurrentPageStubModel : IModelArchitecture, IBatchedPagedModel, IPageOnlyPrefixCacheModel
    {
        private readonly List<int> _rows = new();
        private int _statePosition;

        public List<int> LoneForwards { get; } = new();
        public int PromptTokensForwarded { get; private set; }
        /// <summary>Called on the engine thread after each per-sequence forward.</summary>
        public Action? OnForward { get; set; }

        public ModelConfig Config { get; } = new() { VocabSize = VocabSize };
        public ITokenizer Tokenizer { get; } = new NumberTokenizer();
        public IMultimodalInjector MultimodalInjector => null!;
        public IBackendExecutionPlan ExecutionPlan => null!;
        public bool SupportsKVCacheTruncation => false;
        public bool SupportsKVStateSnapshot => true;
        public bool RequiresPerBlockCapture => true;
        public string KVStateFingerprint => "lone-page-stub";

        public PrefixCacheCapabilities GetPrefixCacheCapabilities() => new()
        {
            Class = FamilyClass.R,
            NamespaceFingerprint = KVStateFingerprint,
            EndState = EndStateSupport.None,
            PrimaryResident = true,
            Truncation = TruncationKind.None,
            Pages = PageSupport.A1HostSlab,
            PagesNeedStateAtEnd = true,
        };

        public long QuerySpareBytes(ResourceClass cls) => -1;

        public float[] ForwardPrompt(int[] tokens)
        {
            PromptTokensForwarded += tokens.Length;
            return Forward(tokens);
        }

        public float[] Forward(int[] tokens)
        {
            RequireStateAtRowsEnd();
            if (tokens.Length > 1)
                LoneForwards.Add(tokens.Length);
            _rows.AddRange(tokens);
            _statePosition = _rows.Count;
            OnForward?.Invoke();
            return Logits();
        }

        private void RequireStateAtRowsEnd()
        {
            if (_statePosition != _rows.Count)
                throw new InvalidOperationException(
                    $"The running state is at {_statePosition}, the rows end at {_rows.Count}: resumed from no restore point.");
        }

        public void ResetKVCache()
        {
            _rows.Clear();
            _statePosition = 0;
        }

        public void TruncateKVCache(int tokenCount) => throw new NotSupportedException();
        public void Dispose() { }

        public long ComputeKVBlockByteSize(int tokenCount) => (tokenCount + 1L) * sizeof(int);

        public bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        {
            if (startToken < 0 || startToken + tokenCount > _rows.Count
                || destination.Length != ComputeKVBlockByteSize(tokenCount))
                return false;
            for (int i = 0; i < tokenCount; i++)
                BitConverter.TryWriteBytes(destination.Slice(i * sizeof(int), sizeof(int)), _rows[startToken + i]);
            BitConverter.TryWriteBytes(destination.Slice(tokenCount * sizeof(int), sizeof(int)), _statePosition);
            return true;
        }

        public bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source)
        {
            if (destToken != _rows.Count || source.Length != ComputeKVBlockByteSize(tokenCount))
                return false;
            for (int i = 0; i < tokenCount; i++)
                _rows.Add(BitConverter.ToInt32(source.Slice(i * sizeof(int), sizeof(int))));
            _statePosition = BitConverter.ToInt32(source.Slice(tokenCount * sizeof(int), sizeof(int)));
            return true;
        }

        // ---- IBatchedPagedModel: no batched route.

        public bool BatchedForwardAvailable => false;

        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx)
            => throw new NotSupportedException("the stub has no batched route");

        private static float[] Logits()
        {
            var logits = new float[VocabSize];
            logits[PeakToken] = 10f;
            return logits;
        }

        private sealed class NumberTokenizer : ITokenizer
        {
            public string[] Vocab { get; } = Enumerable.Range(0, PrefixCacheSharedPageTests.VocabSize)
                .Select(i => i.ToString()).ToArray();
            public int BosTokenId => -1;
            public int[] EosTokenIds => Array.Empty<int>();
            public int VocabSize => Vocab.Length;
            public List<int> Encode(string text, bool addSpecial = true) => new();
            public string Decode(List<int> ids) => string.Join(",", ids);
            public void AppendTokenBytes(int tokenId, List<byte> buffer) { }
            public bool IsEos(int tokenId) => false;
            public int LookupToken(string tokenStr) => -1;
        }
    }
}
