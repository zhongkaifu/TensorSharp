using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using InferenceWeb.Tests.PrefixCache.Fakes;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using Xunit;

namespace InferenceWeb.Tests.PrefixCache;

/// <summary>
/// <see cref="ISoloPrefillBudgetModel.CheckpointGapTokens"/>: a public shared-prefix boundary just past the
/// prefix a request reused does not end a prefill step of its own when the model declares a gap, and does
/// when it declares none or when nothing was reused.
/// </summary>
public sealed class CheckpointGapTests
{
    /// <summary>A holder family with any rewind (MiniMax-M3.1's rules) that declares a checkpoint gap.</summary>
    private sealed class GapOracle : OracleModel, ISoloPrefillBudgetModel
    {
        private readonly int _gap;

        internal GapOracle(int gap) : base(OracleFakes.P(8).Traits, 8) => _gap = gap;

        public int PreferredSoloPrefillTokens(int start, int remaining) => 0;

        public int CheckpointGapTokens => _gap;
    }

    private const int Common = 30, Shared = 33;

    private static SchedulerConfig Configuration() => new()
    {
        BlockSize = 8, NumBlocks = 256, MaxNumBatchedTokens = 64,
        MaxPrefillChunkSize = 16, SoloPrefillChunkSize = 64,
        EnablePrefixCaching = true, StopRepetition = false,
    };

    /// <summary>30 common tokens, then 3 that differ per request (inside the 33-token public prefix, as a
    /// per-request nonce in a system line is), then 10 more.</summary>
    private static List<int> Prompt(int salt) =>
        Enumerable.Range(1, Common).Concat(new[] { salt, salt + 1, salt + 2 }).Concat(Enumerable.Range(salt + 20, 10)).ToList();

    private static SequenceState Request(string id, List<int> tokens) =>
        new(id, tokens, 3, 8, SamplingConfig.Greedy, cacheScope: "conversation", sharedPrefixTokens: Shared);

    // The finished state is kept after the completion is signalled; the tests read the model and the tree next.
    private static async Task<InferenceCompletion> Run(InferenceEngine engine, SequenceState sequence)
    {
        InferenceCompletion completion = await engine.SubmitRequest(sequence).Completion.WaitAsync(TimeSpan.FromSeconds(10));
        await engine.WhenStepSettledAsync().WaitAsync(TimeSpan.FromSeconds(10));
        return completion;
    }

    [Theory]
    [InlineData(0, new[] { 3, 10 })]
    [InlineData(16, new[] { 13 })]
    public async Task BoundaryJustPastAReusedPrefix_GetsAStepOnlyWithoutAGap(int gap, int[] steps)
    {
        var model = new GapOracle(gap);
        using var engine = new InferenceEngine(model, Configuration());

        await Run(engine, Request("first", Prompt(100)));
        // Nothing reused: the first request's prefill still ends a step at the boundary.
        Assert.Equal(Shared, model.ForwardLengths[0]);

        model.ForwardLengths.Clear();
        SequenceState second = Request("second", Prompt(140));
        InferenceCompletion done = await Run(engine, second);
        Assert.Equal(Common, done.PrefixCacheReusedTokens);
        Assert.Equal(steps, model.ForwardLengths.Take(steps.Length));
        Assert.Equal(1, model.ForwardLengths[steps.Length]);   // then decode

        // The output does not depend on where the steps ended.
        using var cold = new InferenceEngine(new GapOracle(0), Configuration());
        SequenceState reference = Request("reference", Prompt(140));
        await Run(cold, reference);
        Assert.Equal(reference.OutputTokens, second.OutputTokens);
    }
}
