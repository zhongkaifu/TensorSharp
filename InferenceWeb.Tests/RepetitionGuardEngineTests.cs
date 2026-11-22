// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

/// <summary>
/// The long-block rule through the engine, the one place every model, backend and host
/// generates through. A model that writes the same 300-token block over and over used to run
/// to its token limit: 115,498 tokens on TensorSharp.Server before the client gave up, and the
/// phone's default limit is 262,144. Now the engine ends it with finish reason
/// <c>repetition</c> once the copies fill 4,096 tokens. A block that changes a token in every
/// copy still runs to the limit, as does a request that turned the guard off.
/// </summary>
public sealed class RepetitionGuardEngineTests
{
    private const int LeadIn = 37, Period = 300, FirstBlockToken = 400, FirstLeadInToken = 100, FirstNumber = 800;

    [Fact]
    public async Task ABlockLoopPastTheShortRulesReach_IsStoppedWithFinishReasonRepetition()
    {
        using var model = new CycleModel(Looping);
        using var engine = Engine(model);
        var seq = Sequence("block-loop", SamplingConfig.Greedy, maxTokens: 20_000);
        InferenceRequestHandle handle = engine.SubmitRequest(seq);

        int[] output = await Collect(handle);
        InferenceCompletion completion = await handle.Completion;

        // The first check after the copies fill MinBlockSpan tokens.
        int expected = RoundUp(LeadIn + RepetitionGuard.MinBlockSpan, RepetitionGuard.BlockCheckInterval);
        Assert.Equal(SequenceStatus.FinishedStopped, completion.Status);
        Assert.Equal(RepetitionGuard.FinishReason, completion.FinishReason);
        Assert.Equal(expected, completion.OutputTokenCount);
        Assert.Equal(expected, output.Length);
        Assert.Equal(Enumerable.Range(0, expected).Select(Looping), output);
    }

    [Fact]
    public async Task ABlockThatChangesATokenInEveryCopy_RunsToTheLimit()
    {
        using var model = new CycleModel(Numbered);
        using var engine = Engine(model);
        InferenceRequestHandle handle = engine.SubmitRequest(Sequence("numbered", SamplingConfig.Greedy, maxTokens: 6_000));

        int[] output = await Collect(handle);
        InferenceCompletion completion = await handle.Completion;
        Assert.Equal("max_tokens", completion.FinishReason);
        Assert.Equal(6_000, output.Length);
    }

    [Fact]
    public async Task TheRequestsOwnOptOut_AlsoCoversTheBlockRule()
    {
        using var model = new CycleModel(Looping);
        using var engine = Engine(model);
        SamplingConfig config = SamplingConfig.Greedy;
        config.StopRepetition = false;
        InferenceRequestHandle handle = engine.SubmitRequest(Sequence("opt-out", config, maxTokens: 6_000));

        int[] output = await Collect(handle);
        Assert.Equal("max_tokens", (await handle.Completion).FinishReason);
        Assert.Equal(6_000, output.Length);
    }

    /// <summary>A lead-in of distinct tokens, then one 300-token block for ever.</summary>
    private static int Looping(int index) =>
        index < LeadIn ? FirstLeadInToken + index : FirstBlockToken + (index - LeadIn) % Period;

    /// <summary>The same block, its first token replaced by the copy's number, as a model numbers slides.</summary>
    private static int Numbered(int index) =>
        index % Period == 0 ? FirstNumber + index / Period : FirstBlockToken + index % Period;

    private static int RoundUp(int value, int multiple) => (value + multiple - 1) / multiple * multiple;

    private static SequenceState Sequence(string id, SamplingConfig config, int maxTokens) =>
        new(id, new[] { 1, 1 }, maxTokens, 8, config);

    private static InferenceEngine Engine(CycleModel model) => new(model, new SchedulerConfig
    {
        MaxNumBatchedTokens = 8, MaxNumRunningSequences = 4,
        MaxPrefillChunkSize = 8, NumBlocks = 4096, BlockSize = 8,
        EnablePrefixCaching = false, DecodeQuantumTokens = 1,
    }, NullLogger.Instance);

    private static async Task<int[]> Collect(InferenceRequestHandle handle)
    {
        var result = new List<int>();
        // A hang guard only; nothing here measures time.
        using var deadline = new CancellationTokenSource(TimeSpan.FromMinutes(2));
        await foreach (int token in handle.Tokens.ReadAllAsync(deadline.Token)) result.Add(token);
        await handle.Completion.WaitAsync(deadline.Token);
        return result.ToArray();
    }

    private sealed class IdTokenizer : ITokenizer
    {
        public string[] Vocab { get; } = Enumerable.Range(0, CycleModel.VocabSize).Select(i => $"<{i}>").ToArray();
        public int BosTokenId => -1;
        public int[] EosTokenIds => new[] { 0 };
        public int VocabSize => Vocab.Length;
        public List<int> Encode(string text, bool addSpecial = true) => new() { LookupToken(text) };
        public string Decode(List<int> ids) => string.Concat(ids.Select(i => Vocab[i]));
        public void AppendTokenBytes(int tokenId, List<byte> bytes) => bytes.AddRange(Encoding.UTF8.GetBytes(Vocab[tokenId]));
        public bool IsEos(int tokenId) => tokenId == 0;
        public int LookupToken(string token) => Array.IndexOf(Vocab, token);
    }

    /// <summary>Writes <c>output(i)</c> as its i-th output token, whatever the prompt; never a stop token.</summary>
    private sealed class CycleModel : IModelArchitecture
    {
        public const int VocabSize = 1024;
        private const int PromptLength = 2;
        private readonly Func<int, int> _output;
        private int _length;

        public CycleModel(Func<int, int> output) => _output = output;

        public ModelConfig Config { get; } = new() { VocabSize = VocabSize };
        public ITokenizer Tokenizer { get; } = new IdTokenizer();
        public IMultimodalInjector MultimodalInjector => null!;
        public IBackendExecutionPlan ExecutionPlan => null!;
        public bool SupportsKVCacheTruncation => true;
        public bool SupportsKVStateSnapshot => false;
        public string KVStateFingerprint => "repetition-guard-engine-test";

        public float[] Forward(int[] tokens)
        {
            _length += tokens.Length;
            var logits = new float[VocabSize];
            logits[_output(Math.Max(0, _length - PromptLength))] = 20;
            return logits;
        }

        public void ResetKVCache() => _length = 0;
        public void TruncateKVCache(int count) => _length = Math.Min(_length, count);
        public long ComputeKVBlockByteSize(int count) => count;
        public bool TryExtractKVBlock(int start, int count, Span<byte> destination) => false;
        public bool TryInjectKVBlock(int start, int count, ReadOnlySpan<byte> source) => false;
        public void Dispose() { }
    }
}
