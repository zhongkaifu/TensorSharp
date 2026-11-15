// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using InferenceWeb.Tests.PrefixCache.Fakes;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests.PrefixCache;

public class RecurrentPromptCheckpointTests
{
    private static OracleTraits Traits(int tail = 1, bool tailWithMedia = false) => new()
    {
        Name = "prompt-checkpoint", Class = FamilyClass.R,
        EndState = EndStateSupport.CopyAndDonate, CanCaptureCopy = true,
        AdoptPrimaryOnDisplacement = true, DeferPrimaryConversion = true,
        PrimaryResident = true, DeviceDirtyOnForward = true,
        Truncation = TruncationKind.None, MinTailPrefillTokens = tail,
        CheckpointPromptTailWithMedia = tailWithMedia,
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
        IReadOnlyList<PromptMediaSpan>? media = null, IReadOnlyList<int>? breaks = null, bool tailOptional = false,
        SamplingConfig? sampling = null) =>
        new(id, Tokens(), 3, 8, sampling ?? SamplingConfig.Greedy, cacheScope: scope,
            sharedPrefixTokens: 16, mediaSpans: media, cacheBreakpoints: breaks) { PromptTailCheckpointOptional = tailOptional };

    // The finished state is kept after the completion is signalled; the tests read the model and the tree next.
    private static async Task<InferenceCompletion> Run(InferenceEngine engine, SequenceState sequence)
    {
        InferenceCompletion completion = await engine.SubmitRequest(sequence).Completion.WaitAsync(TimeSpan.FromSeconds(10));
        await engine.WhenStepSettledAsync().WaitAsync(TimeSpan.FromSeconds(10));
        return completion;
    }

    /// <summary>What the consumer was streamed, besides the completion.</summary>
    private static async Task<(List<int> Streamed, InferenceCompletion Completion)> Stream(InferenceEngine engine, SequenceState sequence)
    {
        var handle = engine.SubmitRequest(sequence);
        var streamed = new List<int>();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        await foreach (int token in handle.Tokens.ReadAllAsync(timeout.Token))
            streamed.Add(token);
        InferenceCompletion completion = await handle.Completion.WaitAsync(TimeSpan.FromSeconds(10));
        await engine.WhenStepSettledAsync().WaitAsync(TimeSpan.FromSeconds(10));
        return (streamed, completion);
    }

    /// <summary>The tree's payload at the prompt's end that kept the prompt's last logits, or null.</summary>
    private static RadixNode? PromptEndCheckpoint(InferenceEngine engine)
    {
        PrefixTree tree = engine.RadixCache!.Tree;
        foreach (string key in tree.PayloadKeys)
            if (tree.TryGetNodeByKey(key, out RadixNode node) && node.EndState?.PromptLogits is not null)
                return node;
        return null;
    }

    [Theory]
    [InlineData(1, 64)]
    [InlineData(2, 62)]
    public async Task IdenticalPrompt_ReusesAResumableState_AndPreservesColdOutput(int tail, int resumed)
    {
        var model = new OracleModel(Traits(tail), 8);
        using var engine = new InferenceEngine(model, Config());
        var first = Request("first");
        Assert.Equal(0, (await Run(engine, first)).PrefixCacheReusedTokens);
        // A one-token tail: the checkpoint sits at the prompt's end, with the last row's logits.
        // A family that forwards more than the last token on a reuse keeps it before that tail:
        // the 64-token state could not serve a retry, which must forward the tail for the logits.
        Assert.Contains(model.RetainedPayloadKeys, key => model.MeasureEndState(key).Tokens == resumed);
        if (tail > 1)
            Assert.DoesNotContain(model.RetainedPayloadKeys, key => model.MeasureEndState(key).Tokens == 64);

        for (int i = 0; i < 3; i++)
        {
            var repeated = Request("repeat-" + i);
            Assert.Equal(resumed, (await Run(engine, repeated)).PrefixCacheReusedTokens);
            Assert.Equal(first.OutputTokens, repeated.OutputTokens);
        }

        using var cold = new InferenceEngine(new OracleModel(Traits(tail), 8), Config(false));
        var reference = Request("cold");
        await Run(cold, reference);
        Assert.Equal(reference.OutputTokens, first.OutputTokens);
    }

    /// <summary>The prompt's last step is no longer split one token short of the end for the checkpoint
    /// (an extra single-row weight pass on every request): it runs whole, and the checkpoint taken after it
    /// keeps the row it computed. A family that must forward more than one tail token still splits.</summary>
    [Theory]
    [InlineData(1, new[] { 16, 48, 1, 1, 1 })]
    [InlineData(2, new[] { 16, 46, 2, 1, 1, 1 })]
    public async Task APrompt_RunsItsLastStepWhole_AndItsCheckpointKeepsThatRow(int tail, int[] forwards)
    {
        var model = new OracleModel(Traits(tail), 8);
        using var engine = new InferenceEngine(model, Config());
        await Run(engine, Request("first"));
        Assert.Equal(forwards, model.ForwardLengths);

        RadixNode? end = PromptEndCheckpoint(engine);
        if (tail > 1)
        {
            Assert.Null(end);
            return;
        }
        Assert.NotNull(end);
        Assert.Equal(64, end!.Depth);
        // The very row a cold prefill of the prompt computes.
        var reference = new OracleModel(Traits(), 8);
        Assert.Equal(reference.Forward(Tokens().ToArray()), end.EndState!.PromptLogits);
    }

    /// <summary>The same prompt sent again resumes at its end with nothing to prefill: its first token comes
    /// from the logits the checkpoint kept and is streamed once, and its steps are the decodes alone.</summary>
    [Fact]
    public async Task IdenticalPrompt_ResumesWithZeroPrefill_AndStreamsTheColdTokens()
    {
        var model = new OracleModel(Traits(), 8);
        using var engine = new InferenceEngine(model, Config());
        var first = Request("first");
        var (firstStreamed, _) = await Stream(engine, first);
        int before = model.ForwardLengths.Count;

        var repeated = Request("repeat");
        var (streamed, completion) = await Stream(engine, repeated);
        Assert.Equal(64, completion.PrefixCacheReusedTokens);
        Assert.Equal(new[] { 1, 1, 1 }, model.ForwardLengths.Skip(before));
        Assert.Equal(first.OutputTokens, repeated.OutputTokens);
        Assert.Equal(firstStreamed, streamed);
        Assert.Equal(repeated.OutputTokens, streamed);

        using var cold = new InferenceEngine(new OracleModel(Traits(), 8), Config(false));
        var reference = Request("cold");
        await Run(cold, reference);
        Assert.Equal(reference.OutputTokens, repeated.OutputTokens);
    }

    /// <summary>A sampled request draws its first token from the kept logits exactly as from the row a
    /// prefill computes: with the same seed a resumed retry, the request that computed the row and a cold
    /// engine all draw the same tokens (and the draws do depend on the seed).</summary>
    [Fact]
    public async Task SampledRetry_DrawsFromTheKeptLogits_AsFromAComputedRow()
    {
        static SamplingConfig Sampled(int seed) => new()
        {
            Temperature = 5f, TopK = 0, TopP = 1f, MinP = 0f, RepetitionPenalty = 1f, Seed = seed, StopRepetition = false,
        };
        var model = new OracleModel(Traits(), 8);
        using var engine = new InferenceEngine(model, Config());
        var first = Request("first", sampling: Sampled(7));
        await Run(engine, first);
        var repeated = Request("repeat", sampling: Sampled(7));
        Assert.Equal(64, (await Run(engine, repeated)).PrefixCacheReusedTokens);
        Assert.Equal(first.OutputTokens, repeated.OutputTokens);

        using var cold = new InferenceEngine(new OracleModel(Traits(), 8), Config(false));
        var reference = Request("cold", sampling: Sampled(7));
        await Run(cold, reference);
        Assert.Equal(reference.OutputTokens, repeated.OutputTokens);

        // Not a greedy draw in disguise: other seeds draw other first tokens from the same kept row.
        var firstTokens = new HashSet<int>();
        for (int seed = 100; seed < 106; seed++)
        {
            var other = Request("seed-" + seed, sampling: Sampled(seed));
            Assert.Equal(64, (await Run(engine, other)).PrefixCacheReusedTokens);
            firstTokens.Add(other.OutputTokens[0]);
        }
        Assert.True(firstTokens.Count > 1, "the sampled first tokens should depend on the seed");
    }

    /// <summary>The kept logits are charged with the checkpoint (a vocabulary of F32 host bytes) and leave
    /// the accounting with it.</summary>
    [Fact]
    public async Task KeptLogits_AreChargedToTheCheckpoint_AndReleasedWithIt()
    {
        var model = new OracleModel(Traits(), 8);
        using var engine = new InferenceEngine(model, Config());
        await Run(engine, Request("first"));
        RadixNode? end = PromptEndCheckpoint(engine);
        Assert.NotNull(end);
        EndStatePayload payload = end!.EndState!;
        long logitsBytes = sizeof(float) * (long)OracleModel.DefaultVocab;
        Assert.Equal(logitsBytes, EndStatePayload.PromptLogitsBytes(payload.PromptLogits!).HostKv);
        PayloadFootprint measured = model.MeasureEndState(payload.Key);
        Assert.Equal(measured.Bytes.HostKv + logitsBytes, payload.Bytes.HostKv);
        Assert.Equal(measured.Bytes.StateSnapshot, payload.Bytes.StateSnapshot);
        Assert.Equal(payload.Bytes, end.Bytes);

        PrefixTree tree = engine.RadixCache!.Tree;
        long charged = 0;
        foreach (string key in tree.PayloadKeys)
            if (tree.TryGetNodeByKey(key, out RadixNode node))
                charged += node.EndState!.Bytes.HostKv;
        Assert.Equal(charged, tree.Cached.HostKv);
        Assert.True(tree.Evict(ResourceClass.HostKv, long.MaxValue / 2, ReleaseReason.Evicted, EvictionTier.PublicTop)
            || tree.Cached.HostKv == 0);
        Assert.Equal(0, tree.Cached.HostKv);
        Assert.Null(PromptEndCheckpoint(engine));
    }

    /// <summary>Prompts with media keep their boundaries: before the newest span and, for a family that
    /// checkpoints the tail of a media prompt, one token short of the end, whose retry forwards that token.</summary>
    [Fact]
    public async Task MediaPrompt_KeepsItsBoundaries()
    {
        var model = new OracleModel(Traits(tailWithMedia: true), 8);
        using var engine = new InferenceEngine(model, Config());
        var media = new[] { new PromptMediaSpan(24, 32, "screen") };
        var first = Request("first", media: media);
        Assert.Equal(24, engine.RadixCache!.AutomaticCheckpointBoundary(first));
        Assert.Equal(63, engine.RadixCache.PromptTailCheckpointBoundary(first));
        var text = Request("text");
        Assert.Equal(64, engine.RadixCache.AutomaticCheckpointBoundary(text));
        Assert.Equal(0, engine.RadixCache.PromptTailCheckpointBoundary(text));

        await Run(engine, first);
        Assert.Equal(new[] { 16, 8, 39, 1, 1, 1, 1 }, model.ForwardLengths);
        Assert.Contains(model.RetainedPayloadKeys, key => model.MeasureEndState(key).Tokens == 24);
        Assert.Contains(model.RetainedPayloadKeys, key => model.MeasureEndState(key).Tokens == 63);
        Assert.Null(PromptEndCheckpoint(engine));

        int before = model.ForwardLengths.Count;
        var repeated = Request("repeat", media: media);
        Assert.Equal(63, (await Run(engine, repeated)).PrefixCacheReusedTokens);
        Assert.Equal(new[] { 1, 1, 1, 1 }, model.ForwardLengths.Skip(before));
        Assert.Equal(first.OutputTokens, repeated.OutputTokens);
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
        Assert.All(completions, c => Assert.Equal(64, c.PrefixCacheReusedTokens));
        Assert.All(requests, r => Assert.Equal(first.OutputTokens, r.OutputTokens));
    }

    [Fact]
    public async Task TailThreshold_KeepsTheCheckpointOfAReplyThatIsNotRenderedBack()
    {
        var model = new OracleModel(Traits(), 8);
        using var engine = new InferenceEngine(model, Config(tailMin: 1024));
        await Run(engine, Request("first"));
        Assert.Equal(64, (await Run(engine, Request("repeat"))).PrefixCacheReusedTokens);
    }

    [Fact]
    public async Task TailThreshold_SkipsTheShortPromptCheckpoint()
    {
        var model = new OracleModel(Traits(), 8);
        using var engine = new InferenceEngine(model, Config(tailMin: 1024));
        var first = Request("first", tailOptional: true);
        await Run(engine, first);
        // No prompt-end state: an exact retry resumes from the 16-token shared prefix.
        Assert.DoesNotContain(model.RetainedPayloadKeys, key => model.MeasureEndState(key).Tokens == 64);
        Assert.Null(PromptEndCheckpoint(engine));
        var repeated = Request("repeat", tailOptional: true);
        Assert.Equal(16, (await Run(engine, repeated)).PrefixCacheReusedTokens);
        Assert.Equal(first.OutputTokens, repeated.OutputTokens);
    }
}
