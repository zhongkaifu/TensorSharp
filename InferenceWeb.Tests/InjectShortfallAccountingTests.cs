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

    private static async Task<(HistoryHashModel Model, SequenceState A, SequenceState B)> RunConcurrentAsync(
        int[] promptA, int[] promptB, int maxNew, int? refuseInjectAt)
    {
        var model = new HistoryHashModel { RefuseInjectAt = refuseInjectAt };
        using var engine = new InferenceEngine(model, Config(prefixCaching: false, decodeQuantum: 1), NullLogger.Instance);
        var a = new SequenceState("a", promptA.ToList(), maxNew, BlockSize, SamplingConfig.Greedy);
        var b = new SequenceState("b", promptB.ToList(), maxNew, BlockSize, SamplingConfig.Greedy);
        var ha = engine.SubmitRequest(a);
        var hb = engine.SubmitRequest(b);
        await Task.WhenAll(ha.Completion, hb.Completion).WaitAsync(TimeSpan.FromSeconds(30));
        return (model, a, b);
    }

    // ------------------------------------------------------------------ helpers

    private static SchedulerConfig Config(bool prefixCaching, int decodeQuantum = BlockSize) => new()
    {
        MaxNumBatchedTokens = 64,
        MaxNumRunningSequences = 4,
        MaxPrefillChunkSize = 64,
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
        public int RefusedInjects { get; private set; }
        public int InjectCalls { get; private set; }
        public List<(int Start, int Count)> Forwards { get; } = new();

        public ModelConfig Config { get; } = new() { VocabSize = VocabSize };
        public ITokenizer Tokenizer { get; } = new NumberTokenizer();
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
                if (RefuseInjectAt == destToken)
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

        private sealed class NumberTokenizer : ITokenizer
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
}
