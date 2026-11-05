// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using InferenceWeb.Tests.PrefixCache.Fakes;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests.PrefixCache;

public class RecurrentPromptCheckpointTests
{
    private static OracleTraits Traits(int tail = 1) => new()
    {
        Name = "prompt-checkpoint", Class = FamilyClass.R,
        EndState = EndStateSupport.CopyAndDonate, CanCaptureCopy = true,
        AdoptPrimaryOnDisplacement = true, DeferPrimaryConversion = true,
        PrimaryResident = true, DeviceDirtyOnForward = true,
        Truncation = TruncationKind.None, MinTailPrefillTokens = tail,
    };

    private static SchedulerConfig Config(bool enabled = true, int tailMin = 0) => new()
    {
        BlockSize = 8, NumBlocks = 256, MaxNumBatchedTokens = 128,
        MaxPrefillChunkSize = 32, SoloPrefillChunkSize = 128,
        EnablePrefixCaching = enabled, StopRepetition = false,
        // These cases cover retry semantics on a 64-token prompt; the default skips a
        // checkpoint that would protect fewer than 1024 tokens (see TailThreshold below).
        PromptTailCheckpointMinTokens = tailMin,
    };

    private static List<int> Tokens() => Enumerable.Range(1, 64).ToList();

    private static SequenceState Request(string id, string? scope = "chat",
        IReadOnlyList<PromptMediaSpan>? media = null, IReadOnlyList<int>? breaks = null, bool tailOptional = false) =>
        new(id, Tokens(), 3, 8, SamplingConfig.Greedy, cacheScope: scope,
            sharedPrefixTokens: 16, mediaSpans: media, cacheBreakpoints: breaks) { PromptTailCheckpointOptional = tailOptional };

    private static async Task<InferenceCompletion> Run(InferenceEngine engine, SequenceState sequence) =>
        await engine.SubmitRequest(sequence).Completion.WaitAsync(TimeSpan.FromSeconds(10));

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public async Task IdenticalPrompt_ReusesAResumableState_AndPreservesColdOutput(int tail)
    {
        var model = new OracleModel(Traits(tail), 8);
        using var engine = new InferenceEngine(model, Config());
        var first = Request("first");
        Assert.Equal(0, (await Run(engine, first)).PrefixCacheReusedTokens);
        Assert.Contains(model.RetainedPayloadKeys, key => model.MeasureEndState(key).Tokens == 64 - tail);
        // The old 64-token snapshot could not serve a retry: this recurrent
        // model cannot rewind it to the mandatory logits-producing tail.
        Assert.DoesNotContain(model.RetainedPayloadKeys, key => model.MeasureEndState(key).Tokens == 64);

        for (int i = 0; i < 3; i++)
        {
            var repeated = Request("repeat-" + i);
            Assert.Equal(64 - tail, (await Run(engine, repeated)).PrefixCacheReusedTokens);
            Assert.Equal(first.OutputTokens, repeated.OutputTokens);
        }

        using var cold = new InferenceEngine(new OracleModel(Traits(tail), 8), Config(false));
        var reference = Request("cold");
        await Run(cold, reference);
        Assert.Equal(reference.OutputTokens, first.OutputTokens);
    }

    [Fact]
    public async Task ReplacedLatestScreenshot_ReusesItsTextPrefix_AndNotTheOldImage()
    {
        var model = new OracleModel(Traits(), 8);
        using var engine = new InferenceEngine(model, Config());
        var oldMedia = new[] { new PromptMediaSpan(20, 28, "earlier"), new PromptMediaSpan(48, 56, "old-screen") };
        var first = Request("first", media: oldMedia);
        await Run(engine, first);
        Assert.Contains(model.RetainedPayloadKeys, key => model.MeasureEndState(key).Tokens == 48);

        var changedMedia = new[] { oldMedia[0], new PromptMediaSpan(48, 56, "new-screen") };
        var replaced = Request("replaced", media: changedMedia);
        Assert.Equal(48, (await Run(engine, replaced)).PrefixCacheReusedTokens);

        var changedEarlier = Request("changed-earlier", media:
            new[] { new PromptMediaSpan(20, 28, "different-earlier"), changedMedia[1] });
        Assert.Equal(16, (await Run(engine, changedEarlier)).PrefixCacheReusedTokens);

        // This oracle validates cache-state and token equality, not vision
        // inference quality. GPU image tests must cover the actual encoder.
        using var cold = new InferenceEngine(new OracleModel(Traits(), 8), Config(false));
        var reference = Request("cold", media: changedMedia);
        await Run(cold, reference);
        Assert.Equal(reference.OutputTokens, replaced.OutputTokens);
    }

    [Theory]
    [InlineData("other")]
    [InlineData(null)]
    public async Task OtherScopes_CannotUseThePrivateRetryCheckpoint(string? otherScope)
    {
        using var engine = new InferenceEngine(new OracleModel(Traits(), 8), Config());
        await Run(engine, Request("first"));
        Assert.Equal(16, (await Run(engine, Request("other", scope: otherScope))).PrefixCacheReusedTokens);
    }

    [Fact]
    public async Task ExplicitCachePolicy_DoesNotGainAutomaticCheckpoints()
    {
        var model = new OracleModel(Traits(), 8);
        using var engine = new InferenceEngine(model, Config());
        await Run(engine, Request("first", breaks: new[] { 16 }));
        Assert.All(model.RetainedPayloadKeys, key => Assert.Equal(16, model.MeasureEndState(key).Tokens));
        Assert.Equal(16, (await Run(engine, Request("repeat", breaks: new[] { 16 }))).PrefixCacheReusedTokens);
        Assert.Equal(0, (await Run(engine, Request("disabled", breaks: Array.Empty<int>()))).PrefixCacheReusedTokens);
    }

    [Theory]
    [InlineData(2)]
    [InlineData(4)]
    public async Task ParallelRetries_KeepIndependentStateAndColdOutput(int count)
    {
        using var engine = new InferenceEngine(new OracleModel(Traits(), 8), Config());
        var first = Request("first");
        await Run(engine, first);
        var gate = new ComputeGate();
        gate.Close();
        engine.ComputeGate = gate;
        var requests = Enumerable.Range(0, count).Select(i => Request("parallel-" + i)).ToArray();
        var handles = requests.Select(r => engine.SubmitRequest(r)).ToArray();
        gate.Open();
        var completions = await Task.WhenAll(handles.Select(h => h.Completion)).WaitAsync(TimeSpan.FromSeconds(10));
        Assert.All(completions, c => Assert.Equal(63, c.PrefixCacheReusedTokens));
        Assert.All(requests, r => Assert.Equal(first.OutputTokens, r.OutputTokens));
    }

    [Fact]
    public async Task TailThreshold_KeepsTheCheckpointOfAReplyThatIsNotRenderedBack()
    {
        var model = new OracleModel(Traits(), 8);
        using var engine = new InferenceEngine(model, Config(tailMin: 1024));
        await Run(engine, Request("first"));
        Assert.Equal(63, (await Run(engine, Request("repeat"))).PrefixCacheReusedTokens);
    }

    [Fact]
    public async Task TailThreshold_SkipsTheShortPromptCheckpointAndItsExtraForward()
    {
        var model = new OracleModel(Traits(), 8);
        using var engine = new InferenceEngine(model, Config(tailMin: 1024));
        var first = Request("first", tailOptional: true);
        await Run(engine, first);
        // No 63-token prompt-tail state: an exact retry resumes from the 16-token shared prefix.
        Assert.DoesNotContain(model.RetainedPayloadKeys, key => model.MeasureEndState(key).Tokens == 63);
        var repeated = Request("repeat", tailOptional: true);
        Assert.Equal(16, (await Run(engine, repeated)).PrefixCacheReusedTokens);
        Assert.Equal(first.OutputTokens, repeated.OutputTokens);
    }
}
