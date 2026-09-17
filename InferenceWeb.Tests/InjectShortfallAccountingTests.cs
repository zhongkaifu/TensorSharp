// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// M0e (radix design D10/P16): a pooled-prefix inject that stops early is a MISS,
// never a partial state.
//
// BatchExecutor.InjectAllBlocks used to stop at the first block the model refused
// and return nothing. The sequence kept the computed-token count admission had
// given it, so the next forward appended at the model's real head while numbering
// positions as if the whole prefix were there, and PrefixCacheReusedTokens still
// reported the promised reuse. The fix corrects the reuse to what was materialized,
// resets the request to that length and forwards the lost tail again before the
// step's own work.
//
// The fake model folds every (position, token) it holds into its logits, so any
// state that differs from a cold run changes the greedy stream.
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

public sealed class InjectShortfallAccountingTests
{
    private const int BlockSize = 8;
    private const int VocabSize = 97;

    [Fact]
    public async Task RefusedBlockAtAdmission_CorrectsReuse_AndOutputEqualsCold()
    {
        int[] shared = Enumerable.Range(0, 3 * BlockSize).Select(i => 3 + (i * 11) % 80).ToArray();
        int[] promptA = shared.Concat(new[] { 5, 6, 7 }).ToArray();
        int[] promptB = shared.Concat(new[] { 9, 10, 11, 12, 13 }).ToArray();

        int[] coldB = await RunColdAsync(promptB, maxNew: 6);

        var model = new HistoryHashModel();
        using var engine = new InferenceEngine(model, Config(prefixCaching: true), NullLogger.Instance);
        await RunAsync(engine, "a", promptA, maxNew: 4);

        // Block 2 (tokens 16..23) is refused when B's cached prefix is injected.
        model.RefuseInjectAt = 2 * BlockSize;
        var b = new SequenceState("b", promptB.ToList(), 6, BlockSize, SamplingConfig.Greedy);
        var completion = await engine.SubmitRequest(b).Completion.WaitAsync(TimeSpan.FromSeconds(20));

        Assert.Equal(1, model.RefusedInjects);
        // Admission matched the three full blocks; only two were materialized.
        Assert.Equal(2 * BlockSize, completion.PrefixCacheReusedTokens);
        Assert.Equal(2 * BlockSize, b.PrefixCacheReusedTokens);
        // The lost block was forwarded again, starting exactly where the inject stopped.
        Assert.Contains(model.Forwards, f => f.Start == 2 * BlockSize && f.Count == BlockSize);
        Assert.Equal(coldB, b.OutputTokens.ToArray());
        Assert.Equal(promptB.Length + b.OutputTokens.Count, b.NumComputedTokens);
    }

    [Fact]
    public async Task RefusedBlockAtAdmission_OnTheFirstBlock_ReusesNothing_AndOutputEqualsCold()
    {
        int[] shared = Enumerable.Range(0, 2 * BlockSize).Select(i => 4 + (i * 13) % 80).ToArray();
        int[] promptA = shared.Concat(new[] { 1, 2 }).ToArray();
        int[] promptB = shared.Concat(new[] { 30, 31, 32 }).ToArray();
        int[] coldB = await RunColdAsync(promptB, maxNew: 5);

        var model = new HistoryHashModel();
        using var engine = new InferenceEngine(model, Config(prefixCaching: true), NullLogger.Instance);
        await RunAsync(engine, "a", promptA, maxNew: 3);

        model.RefuseInjectAt = 0;
        var b = new SequenceState("b", promptB.ToList(), 5, BlockSize, SamplingConfig.Greedy);
        var completion = await engine.SubmitRequest(b).Completion.WaitAsync(TimeSpan.FromSeconds(20));

        Assert.Equal(1, model.RefusedInjects);
        Assert.Equal(0, completion.PrefixCacheReusedTokens);
        Assert.Equal(coldB, b.OutputTokens.ToArray());
    }

    [Fact]
    public async Task RefusedBlockOnADecodersSwapIn_RecomputesItsGeneratedTail_AndOutputEqualsCold()
    {
        // Two concurrent requests on the linear (single cache) path with a one-token
        // decode quantum: ownership rotates every step, so each swap-in re-injects the
        // sequence's prompt AND generated tokens from its blocks. One of those injects
        // is refused mid-generation; the generated tokens behind it must be forwarded
        // again (the scheduler would otherwise just decode on top of the gap).
        int[] promptA = Enumerable.Range(0, 11).Select(i => 20 + i).ToArray();
        int[] promptB = Enumerable.Range(0, 13).Select(i => 50 + (i * 3) % 40).ToArray();
        const int maxNew = 14;
        int[] coldA = await RunColdAsync(promptA, maxNew);
        int[] coldB = await RunColdAsync(promptB, maxNew);

        // Control: the same concurrent run with every inject accepted.
        var (controlModel, _, _) = await RunConcurrentAsync(promptA, promptB, maxNew, refuseInjectAt: null);
        Assert.Equal(0, controlModel.RefusedInjects);
        Assert.True(controlModel.InjectCalls > 0, "the control run never swapped ownership, so nothing was injected");

        // The first inject of block 2 (tokens 16..23) that happens is refused: neither
        // prompt reaches it, so what it held is always generated output.
        var (model, a, b) = await RunConcurrentAsync(promptA, promptB, maxNew, refuseInjectAt: 2 * BlockSize);

        Assert.Equal(1, model.RefusedInjects);
        Assert.Equal(coldA, a.OutputTokens.ToArray());
        Assert.Equal(coldB, b.OutputTokens.ToArray());
        // The rows behind the refused block were forwarded again (whatever it held,
        // from token 16 to the decoder's head), on top of what the control forwarded.
        int extra = model.Forwards.Sum(f => f.Count) - controlModel.Forwards.Sum(f => f.Count);
        Assert.True(extra >= 1, $"expected the lost tail to be re-forwarded, forwarded {extra} extra tokens");
        Assert.Contains(model.Forwards, f => f.Start == 2 * BlockSize && f.Count == extra);
    }

    [Fact]
    public async Task RefusedBlockOnAFreshFusedHolder_CorrectsReuse_AndOutputEqualsCold()
    {
        // The second inject site: a request admitted with pooled reuse while another
        // request is running goes to the per-sequence fused path, whose FRESH holder is
        // filled from the pool before its first forward.
        int[] shared = Enumerable.Range(0, 3 * BlockSize).Select(i => 6 + (i * 17) % 80).ToArray();
        int[] promptA = shared.Concat(new[] { 2, 3, 4 }).ToArray();
        int[] promptB = shared.Concat(new[] { 40, 41, 42, 43, 44 }).ToArray();
        int[] promptC = Enumerable.Range(0, 10).Select(i => 60 + i).ToArray();
        const int maxNewB = 6, maxNewC = 120;
        int[] coldB = await RunColdFusedAsync(promptB, maxNewB);
        int[] coldC = await RunColdFusedAsync(promptC, maxNewC);

        var model = new FusedHistoryHashModel();
        using var engine = new InferenceEngine(model, FusedConfig(prefixCaching: true), NullLogger.Instance);
        await RunAsync(engine, "a", promptA, maxNew: 4);

        // c runs alone first (the single-stream path); b arrives while it decodes.
        int decodesBeforeC = model.DecodeForwards;
        var c = new SequenceState("c", promptC.ToList(), maxNewC, BlockSize, SamplingConfig.Greedy);
        var hc = engine.SubmitRequest(c);
        var until = DateTime.UtcNow.AddSeconds(20);
        while (model.DecodeForwards - decodesBeforeC < 2 && DateTime.UtcNow < until)
            await Task.Delay(1);
        Assert.True(model.DecodeForwards - decodesBeforeC >= 2, "c never started decoding");

        model.RefuseInjectAt = 2 * BlockSize;
        var b = new SequenceState("b", promptB.ToList(), maxNewB, BlockSize, SamplingConfig.Greedy);
        var completionB = await engine.SubmitRequest(b).Completion.WaitAsync(TimeSpan.FromSeconds(30));
        await hc.Completion.WaitAsync(TimeSpan.FromSeconds(30));

        Assert.Equal(1, model.RefusedInjects);
        Assert.Equal("b", model.RefusedInjectCache);
        Assert.Equal(2 * BlockSize, completionB.PrefixCacheReusedTokens);
        Assert.Equal(coldB, b.OutputTokens.ToArray());
        Assert.Equal(coldC, c.OutputTokens.ToArray());
    }

    [Fact]
    public async Task PerBlockCaptureModel_BacksOffToTheLastRestorableBlock_AndOutputEqualsCold()
    {
        // A recurrent model restores its running state from the LAST injected block.
        // With 12-token prefill chunks, blocks 0 and 1 are captured mid-chunk (not
        // restorable) and block 2 at a chunk end (restorable), so admission adopts all
        // three. Refusing block 2 leaves blocks 0-1 injected, whose saved state is the
        // chunk-end state at token 24, not 16: the only exact resume point is zero.
        int[] shared = Enumerable.Range(0, 3 * BlockSize).Select(i => 5 + (i * 19) % 80).ToArray();
        int[] promptA = shared.Concat(new[] { 7, 8, 9 }).ToArray();
        int[] promptB = shared.Concat(new[] { 20, 21, 22, 23 }).ToArray();

        int[] coldB;
        {
            var coldModel = new RecurrentHashModel();
            using var coldEngine = new InferenceEngine(coldModel, Config(prefixCaching: false, prefillChunk: 12), NullLogger.Instance);
            coldB = (await RunAsync(coldEngine, "cold", promptB, maxNew: 6)).OutputTokens.ToArray();
        }

        var model = new RecurrentHashModel();
        using var engine = new InferenceEngine(model, Config(prefixCaching: true, prefillChunk: 12), NullLogger.Instance);
        await RunAsync(engine, "a", promptA, maxNew: 3);

        model.RefuseInjectAt = 2 * BlockSize;
        var b = new SequenceState("b", promptB.ToList(), 6, BlockSize, SamplingConfig.Greedy);
        var completion = await engine.SubmitRequest(b).Completion.WaitAsync(TimeSpan.FromSeconds(20));

        Assert.Equal(1, model.RefusedInjects);
        Assert.Equal(0, completion.PrefixCacheReusedTokens);
        Assert.Equal(coldB, b.OutputTokens.ToArray());
    }

    [Fact]
    public async Task RecomputedTailThatCrossesThePromptEnd_ForwardsPromptAndGeneratedTokensSeparately()
    {
        // Block 1 (tokens 8..15) straddles both prompt ends. Letting the first two
        // injects of it through makes the refused one a decoder that already generated
        // past its prompt, so the lost tail holds prompt AND generated tokens. The
        // planned steps never mix the two in one forward (a prompt slice queues media
        // embeddings and an M-RoPE table sized to its prompt tokens), so the recompute
        // must not either.
        int[] promptA = Enumerable.Range(0, 11).Select(i => 20 + i).ToArray();
        int[] promptB = Enumerable.Range(0, 13).Select(i => 50 + (i * 3) % 40).ToArray();
        const int maxNew = 10;
        int[] coldA = await RunColdAsync(promptA, maxNew);
        int[] coldB = await RunColdAsync(promptB, maxNew);
        var (controlModel, _, _) = await RunConcurrentAsync(promptA, promptB, maxNew, refuseInjectAt: null);

        var (model, a, b) = await RunConcurrentAsync(promptA, promptB, maxNew, refuseInjectAt: BlockSize, refuseSkip: 2);

        Assert.Equal(1, model.RefusedInjects);
        Assert.Equal(coldA, a.OutputTokens.ToArray());
        Assert.Equal(coldB, b.OutputTokens.ToArray());
        // Only the recompute starts a forward at token 8.
        var first = Assert.Single(model.Forwards, f => f.Start == BlockSize);
        int extra = model.Forwards.Sum(f => f.Count) - controlModel.Forwards.Sum(f => f.Count);
        Assert.True(extra > first.Count, $"the refused inject lost no generated tokens (extra {extra}, first {first.Count})");
        Assert.Contains(first.Start + first.Count, new[] { promptA.Length, promptB.Length });
    }

    private static async Task<(HistoryHashModel Model, SequenceState A, SequenceState B)> RunConcurrentAsync(
        int[] promptA, int[] promptB, int maxNew, int? refuseInjectAt, int refuseSkip = 0)
    {
        var model = new HistoryHashModel { RefuseInjectAt = refuseInjectAt, RefuseInjectSkip = refuseSkip };
        using var engine = new InferenceEngine(model, Config(prefixCaching: false, decodeQuantum: 1), NullLogger.Instance);
        var a = new SequenceState("a", promptA.ToList(), maxNew, BlockSize, SamplingConfig.Greedy);
        var b = new SequenceState("b", promptB.ToList(), maxNew, BlockSize, SamplingConfig.Greedy);
        var ha = engine.SubmitRequest(a);
        var hb = engine.SubmitRequest(b);
        await Task.WhenAll(ha.Completion, hb.Completion).WaitAsync(TimeSpan.FromSeconds(30));
        return (model, a, b);
    }

    // ------------------------------------------------------------------ helpers

    private static SchedulerConfig Config(bool prefixCaching, int decodeQuantum = BlockSize, int prefillChunk = 64) => new()
    {
        MaxNumBatchedTokens = 64,
        MaxNumRunningSequences = 4,
        MaxPrefillChunkSize = prefillChunk,
        NumBlocks = 32,
        BlockSize = BlockSize,
        EnablePrefixCaching = prefixCaching,
        DecodeQuantumTokens = decodeQuantum,
    };

    private static async Task<int[]> RunColdAsync(int[] prompt, int maxNew)
    {
        var model = new HistoryHashModel();
        using var engine = new InferenceEngine(model, Config(prefixCaching: false), NullLogger.Instance);
        var seq = await RunAsync(engine, "cold", prompt, maxNew);
        Assert.Equal(0, model.RefusedInjects);
        return seq.OutputTokens.ToArray();
    }

    private static SchedulerConfig FusedConfig(bool prefixCaching) => new()
    {
        MaxNumBatchedTokens = 64,
        MaxNumRunningSequences = 4,
        MaxPrefillChunkSize = 64,
        NumBlocks = 64,
        BlockSize = BlockSize,
        EnablePrefixCaching = prefixCaching,
    };

    private static async Task<int[]> RunColdFusedAsync(int[] prompt, int maxNew)
    {
        var model = new FusedHistoryHashModel();
        using var engine = new InferenceEngine(model, FusedConfig(prefixCaching: false), NullLogger.Instance);
        var seq = await RunAsync(engine, "cold", prompt, maxNew);
        Assert.Equal(0, model.RefusedInjects);
        return seq.OutputTokens.ToArray();
    }

    private static async Task<SequenceState> RunAsync(InferenceEngine engine, string id, int[] prompt, int maxNew)
    {
        var seq = new SequenceState(id, prompt.ToList(), maxNew, BlockSize, SamplingConfig.Greedy);
        var completion = await engine.SubmitRequest(seq).Completion.WaitAsync(TimeSpan.FromSeconds(20));
        Assert.Equal(SequenceStatus.FinishedLengthCapped, completion.Status);
        return seq;
    }

    /// <summary>
    /// A single linear cache whose rows are the tokens themselves. Logits peak at a
    /// hash of every (position, token) row, so a missing row, a row at the wrong
    /// position or a stale tail all change the next token. Blocks serialize the rows;
    /// <see cref="RefuseInjectAt"/> refuses the first inject that starts at that token.
    /// </summary>
    private sealed class HistoryHashModel : IModelArchitecture
    {
        private readonly List<int> _rows = new();
        private readonly object _gate = new();

        public int? RefuseInjectAt { get; set; }
        /// <summary>How many matching injects to let through before refusing one.</summary>
        public int RefuseInjectSkip { get; set; }
        public int RefusedInjects { get; private set; }
        public int InjectCalls { get; private set; }
        public List<(int Start, int Count)> Forwards { get; } = new();

        public ModelConfig Config { get; } = new() { VocabSize = VocabSize };
        public ITokenizer Tokenizer { get; } = new NumberTokenizerPublic();
        public IMultimodalInjector MultimodalInjector => null;
        public IBackendExecutionPlan ExecutionPlan => null;
        public bool SupportsKVCacheTruncation => true;
        public bool SupportsKVStateSnapshot => true;
        public string KVStateFingerprint => "history-hash";

        public float[] Forward(int[] tokens)
        {
            lock (_gate)
            {
                Forwards.Add((_rows.Count, tokens.Length));
                _rows.AddRange(tokens);
                ulong h = 1469598103934665603UL;
                for (int p = 0; p < _rows.Count; p++)
                {
                    h = (h ^ (ulong)p) * 1099511628211UL;
                    h = (h ^ (ulong)_rows[p]) * 1099511628211UL;
                }
                var logits = new float[VocabSize];
                logits[(int)(h % (ulong)(VocabSize - 1)) + 1] = 10f;
                return logits;
            }
        }

        public void ResetKVCache() { lock (_gate) _rows.Clear(); }

        public void TruncateKVCache(int tokenCount)
        {
            lock (_gate)
            {
                if (tokenCount < _rows.Count)
                    _rows.RemoveRange(tokenCount, _rows.Count - tokenCount);
            }
        }

        public void Dispose() { }

        public long ComputeKVBlockByteSize(int tokenCount) => (long)tokenCount * sizeof(int);

        public bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        {
            lock (_gate)
            {
                if (startToken < 0 || startToken + tokenCount > _rows.Count) return false;
                if (destination.Length < tokenCount * sizeof(int)) return false;
                for (int i = 0; i < tokenCount; i++)
                    BitConverter.TryWriteBytes(destination.Slice(i * sizeof(int), sizeof(int)), _rows[startToken + i]);
                return true;
            }
        }

        public bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source)
        {
            lock (_gate)
            {
                InjectCalls++;
                if (RefuseInjectAt == destToken && RefuseInjectSkip > 0)
                {
                    RefuseInjectSkip--;
                }
                else if (RefuseInjectAt == destToken)
                {
                    RefuseInjectAt = null;
                    RefusedInjects++;
                    return false;
                }
                if (destToken != _rows.Count) return false;
                for (int i = 0; i < tokenCount; i++)
                    _rows.Add(BitConverter.ToInt32(source.Slice(i * sizeof(int), sizeof(int))));
                return true;
            }
        }

        internal sealed class NumberTokenizerPublic : ITokenizer
        {
            public string[] Vocab { get; } = Enumerable.Range(0, InjectShortfallAccountingTests.VocabSize).Select(i => i.ToString()).ToArray();
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

    private static float[] PeakLogits(ulong h)
    {
        var logits = new float[VocabSize];
        logits[(int)(h % (ulong)(VocabSize - 1)) + 1] = 10f;
        return logits;
    }

    private static ulong Fold(ulong h, int position, int token)
    {
        h = (h ^ (ulong)position) * 1099511628211UL;
        return (h ^ (ulong)token) * 1099511628211UL;
    }

    private const ulong Seed = 1469598103934665603UL;

    /// <summary>
    /// <see cref="HistoryHashModel"/> with one cache per request (the per-sequence
    /// fused contract): a primary cache for the single-stream path and a holder per
    /// bound RequestId. Logits hash every (position, token) row of the ACTIVE cache.
    /// </summary>
    private sealed class FusedHistoryHashModel : IModelArchitecture, IBatchedPagedModel
    {
        private readonly object _gate = new();
        private readonly Dictionary<string, List<int>> _holders = new(StringComparer.Ordinal);
        private List<int> _primary = new();
        private List<int> _active;
        private string _activeKey;

        public FusedHistoryHashModel() => _active = _primary;

        public int? RefuseInjectAt { get; set; }
        public int RefusedInjects { get; private set; }
        public string RefusedInjectCache { get; private set; }

        private int _decodeForwardCount;

        /// <summary>Single-token forwards so far, on any cache.</summary>
        public int DecodeForwards => Volatile.Read(ref _decodeForwardCount);

        public ModelConfig Config { get; } = new() { VocabSize = VocabSize };
        public ITokenizer Tokenizer { get; } = new HistoryHashModel.NumberTokenizerPublic();
        public IMultimodalInjector MultimodalInjector => null;
        public IBackendExecutionPlan ExecutionPlan => null;
        public bool SupportsKVCacheTruncation => true;
        public bool SupportsKVStateSnapshot => true;
        public string KVStateFingerprint => "fused-history-hash";

        public float[] Forward(int[] tokens)
        {
            lock (_gate)
            {
                if (tokens.Length == 1)
                {
                    Interlocked.Increment(ref _decodeForwardCount);
                    // Slow decode slightly so the second request reliably arrives while
                    // the first is still running.
                    Thread.Sleep(1);
                }
                _active.AddRange(tokens);
                ulong h = Seed;
                for (int p = 0; p < _active.Count; p++) h = Fold(h, p, _active[p]);
                return PeakLogits(h);
            }
        }

        public void ResetKVCache() { lock (_gate) _active.Clear(); }

        public void TruncateKVCache(int tokenCount)
        {
            lock (_gate)
            {
                if (tokenCount < _active.Count)
                    _active.RemoveRange(tokenCount, _active.Count - tokenCount);
            }
        }

        public void Dispose() { }

        public long ComputeKVBlockByteSize(int tokenCount) => (long)tokenCount * sizeof(int);

        public bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        {
            lock (_gate)
            {
                if (startToken < 0 || startToken + tokenCount > _active.Count) return false;
                for (int i = 0; i < tokenCount; i++)
                    BitConverter.TryWriteBytes(destination.Slice(i * sizeof(int), sizeof(int)), _active[startToken + i]);
                return true;
            }
        }

        public bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source)
        {
            lock (_gate)
            {
                if (RefuseInjectAt == destToken)
                {
                    RefuseInjectAt = null;
                    RefusedInjects++;
                    RefusedInjectCache = _activeKey ?? "<primary>";
                    return false;
                }
                if (destToken != _active.Count) return false;
                for (int i = 0; i < tokenCount; i++)
                    _active.Add(BitConverter.ToInt32(source.Slice(i * sizeof(int), sizeof(int))));
                return true;
            }
        }

        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx)
            => throw new InvalidOperationException("the paged batched path is not part of this fake");

        public bool BatchedForwardAvailable => false;
        public bool SupportsLinearKVMigration => true;
        public bool TryMigrateLinearKVToPaged(SequenceState owner, int blockSize) => false;
        public bool SupportsPerSequenceFusedForward => true;
        public bool CanBatchDecode(string requestId, int position) => false;

        public bool BindSequenceCache(string requestId)
        {
            lock (_gate)
            {
                bool fresh = !_holders.TryGetValue(requestId, out var holder);
                if (fresh) _holders[requestId] = holder = new List<int>();
                _active = holder;
                _activeKey = requestId;
                return fresh;
            }
        }

        public void AdoptPrimaryCacheToFused(string requestId)
        {
            lock (_gate)
            {
                _holders[requestId] = _primary;
                _primary = new List<int>();
                _active = _holders[requestId];
                _activeKey = requestId;
            }
        }

        public void RestorePrimaryCache()
        {
            lock (_gate) { _active = _primary; _activeKey = null; }
        }

        public bool HasFusedSequenceCache(string requestId)
        {
            lock (_gate) return _holders.ContainsKey(requestId);
        }

        public void OnSequenceReleased(string requestId)
        {
            lock (_gate)
            {
                if (_holders.Remove(requestId, out var holder) && ReferenceEquals(_active, holder))
                {
                    _active = _primary;
                    _activeKey = null;
                }
            }
        }
    }

    /// <summary>
    /// A per-block-capture (recurrent) fake: its logits come from a running state
    /// folded over every forwarded (position, token), and a block snapshot carries
    /// that state AS OF THE EXTRACTION, so an injected prefix resumes from the state
    /// its last block was captured at - exactly the hazard a mid-chunk capture has.
    /// </summary>
    private sealed class RecurrentHashModel : IModelArchitecture
    {
        private readonly object _gate = new();
        private readonly List<int> _rows = new();
        private ulong _state = Seed;

        public int? RefuseInjectAt { get; set; }
        public int RefusedInjects { get; private set; }

        public ModelConfig Config { get; } = new() { VocabSize = VocabSize };
        public ITokenizer Tokenizer { get; } = new HistoryHashModel.NumberTokenizerPublic();
        public IMultimodalInjector MultimodalInjector => null;
        public IBackendExecutionPlan ExecutionPlan => null;
        public bool SupportsKVCacheTruncation => false;
        public bool SupportsKVStateSnapshot => true;
        public bool RequiresPerBlockCapture => true;
        public string KVStateFingerprint => "recurrent-hash";

        public float[] Forward(int[] tokens)
        {
            lock (_gate)
            {
                foreach (int t in tokens)
                {
                    _state = Fold(_state, _rows.Count, t);
                    _rows.Add(t);
                }
                return PeakLogits(_state);
            }
        }

        public void ResetKVCache() { lock (_gate) { _rows.Clear(); _state = Seed; } }

        public void TruncateKVCache(int tokenCount) => throw new NotSupportedException();

        public void Dispose() { }

        public long ComputeKVBlockByteSize(int tokenCount) => (long)tokenCount * sizeof(int) + sizeof(ulong);

        public bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        {
            lock (_gate)
            {
                if (startToken < 0 || startToken + tokenCount > _rows.Count) return false;
                for (int i = 0; i < tokenCount; i++)
                    BitConverter.TryWriteBytes(destination.Slice(i * sizeof(int), sizeof(int)), _rows[startToken + i]);
                BitConverter.TryWriteBytes(destination.Slice(tokenCount * sizeof(int), sizeof(ulong)), _state);
                return true;
            }
        }

        public bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source)
        {
            lock (_gate)
            {
                if (RefuseInjectAt == destToken)
                {
                    RefuseInjectAt = null;
                    RefusedInjects++;
                    return false;
                }
                if (destToken != _rows.Count) return false;
                for (int i = 0; i < tokenCount; i++)
                    _rows.Add(BitConverter.ToInt32(source.Slice(i * sizeof(int), sizeof(int))));
                _state = BitConverter.ToUInt64(source.Slice(tokenCount * sizeof(int), sizeof(ulong)));
                return true;
            }
        }
    }
}
