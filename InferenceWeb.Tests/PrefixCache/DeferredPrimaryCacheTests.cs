using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using InferenceWeb.Tests.PrefixCache.Fakes;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;
using Xunit;

namespace InferenceWeb.Tests.PrefixCache;

public class DeferredPrimaryCacheTests
{
    private static OracleTraits Traits => new()
    {
        Name = "deferred-recurrent", Class = FamilyClass.R, EndState = EndStateSupport.CopyAndDonate,
        CanCaptureCopy = true, AdoptPrimaryOnDisplacement = true, PrimaryResident = true,
        DeferPrimaryConversion = true, Truncation = TruncationKind.None,
        FailedConversionReleasesPrimary = true,
    };

    private static SchedulerConfig Configuration(bool enabled = true) => new()
    {
        BlockSize = 8, NumBlocks = 256, MaxNumBatchedTokens = 64,
        MaxPrefillChunkSize = 16, SoloPrefillChunkSize = 64,
        EnablePrefixCaching = enabled, StopRepetition = false,
    };

    private static SequenceState Request(string id, List<int> prompt, string scope) =>
        new(id, prompt, 3, 8, SamplingConfig.Greedy, cacheScope: scope);

    private static List<int> Extend(SequenceState sequence, params int[] tail) =>
        sequence.PromptTokens.Concat(sequence.OutputTokens).Take(sequence.NumComputedTokens).Concat(tail).ToList();

    // The finished state is kept after the completion is signalled; the tests read the model and the tree next.
    private static async Task<InferenceCompletion> Run(InferenceEngine engine, SequenceState sequence)
    {
        InferenceCompletion completion = await engine.SubmitRequest(sequence).Completion.WaitAsync(TimeSpan.FromSeconds(10));
        await engine.WhenStepSettledAsync().WaitAsync(TimeSpan.FromSeconds(10));
        return completion;
    }

    private static async Task AssertColdParity(SequenceState actual)
    {
        using var cold = new InferenceEngine(new OracleModel(Traits, 8), Configuration(false));
        var reference = Request("reference", actual.PromptTokens, "reference");
        await Run(cold, reference);
        Assert.Equal(reference.OutputTokens, actual.OutputTokens);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ExactContinuation_WithNoRetentionHeadroom_NeverConvertsTheLivePrimary(bool singleSequenceFused)
    {
        var traits = Traits with { Pages = singleSequenceFused ? PageSupport.A2ModelPaged : PageSupport.None };
        var model = new OracleModel(traits, 8) { SpareBytes = 1 };
        model.FailNext("convert"); // A conversion would destroy the valid primary in this fixture.
        using var engine = new InferenceEngine(model, Configuration());
        var first = Request("first", Enumerable.Range(1, 32).ToList(), "conversation");
        await Run(engine, first);
        Assert.Equal(0, model.PrimaryConversionCalls);
        Assert.Equal(first.NumComputedTokens, model.PrimaryCacheLength);
        Assert.Empty(model.RetainedPayloadKeys);

        var second = Request("second", Extend(first, 101, 102), "conversation");
        Assert.Equal(first.NumComputedTokens, (await Run(engine, second)).PrefixCacheReusedTokens);
        Assert.Equal(0, model.PrimaryConversionCalls);
        Assert.Equal(second.NumComputedTokens, model.PrimaryCacheLength);
        await AssertColdParity(second);

        var third = Request("third", Extend(second, 111, 112), "conversation");
        Assert.Equal(second.NumComputedTokens, (await Run(engine, third)).PrefixCacheReusedTokens);
        Assert.Equal(0, model.PrimaryConversionCalls);
        await AssertColdParity(third);
    }

    [Fact]
    public async Task ImpossibleDisplacement_DeclinesBeforeConversion_AndKeepsTheNewPrimaryReusable()
    {
        var model = new OracleModel(Traits with { MeasurePrimaryFootprint = true }, 8) { SpareBytes = 1 };
        model.FailNext("convert-throw"); // Prove conversion is never reached, even after headroom recovers.
        using var engine = new InferenceEngine(model, Configuration());
        var first = Request("first", Enumerable.Range(1, 32).ToList(), "original");
        await Run(engine, first);
        model.SpareBytes = 1024 * 1024;

        var other = Request("other", Extend(first, 101, 102), "other");
        Assert.Equal(0, (await Run(engine, other)).PrefixCacheReusedTokens);
        Assert.Equal(0, model.PrimaryConversionCalls);
        Assert.Empty(model.RetainedPayloadKeys);
        await AssertColdParity(other);

        var next = Request("other-next", Extend(other, 111, 112), "other");
        Assert.Equal(other.NumComputedTokens, (await Run(engine, next)).PrefixCacheReusedTokens);
        Assert.Equal(0, model.PrimaryConversionCalls);
        await AssertColdParity(next);

        var original = Request("original-next", Extend(next, 121, 122), "original");
        Assert.Equal(0, (await Run(engine, original)).PrefixCacheReusedTokens);
        Assert.Equal(0, model.PrimaryConversionCalls);
        await AssertColdParity(original);
    }

    [Fact]
    public async Task FeasibleDisplacement_DoesNotConfuseTransientSpareWithTheAbsoluteCap()
    {
        var model = new OracleModel(Traits with { MeasurePrimaryFootprint = true }, 8) { SpareBytes = 4096 };
        using var engine = new InferenceEngine(model, Configuration());
        var first = Request("first", Enumerable.Range(1, 32).ToList(), "original");
        await Run(engine, first);
        // The holder is already allocated. Its bytes become cached during publication,
        // so the prepublication cached-plus-spare term is not an absolute retention limit.
        model.SpareBytes = 1;
        var other = Request("other", Enumerable.Range(71, 32).ToList(), "other");
        Assert.Equal(0, (await Run(engine, other)).PrefixCacheReusedTokens);
        Assert.Equal(1, model.PrimaryConversionCalls);
        Assert.Contains(model.RetainedPayloadKeys, key => model.MeasureEndState(key).Tokens == first.NumComputedTokens);
        await AssertColdParity(other);

        var branch = Request("branch", Extend(first, 111, 112), "original");
        Assert.Equal(first.NumComputedTokens, (await Run(engine, branch)).PrefixCacheReusedTokens);
        await AssertColdParity(branch);
    }

    [Fact]
    public async Task FailedDisplacement_InvalidatesTheOldPrimary_AndKeepsScopesIsolated()
    {
        var model = new OracleModel(Traits, 8) { SpareBytes = 1 };
        model.FailNext("convert");
        using var engine = new InferenceEngine(model, Configuration());
        var first = Request("first", Enumerable.Range(1, 32).ToList(), "original");
        await Run(engine, first);
        var nextPrompt = Extend(first, 101, 102);
        var other = Request("other", nextPrompt, "other");
        Assert.Equal(0, (await Run(engine, other)).PrefixCacheReusedTokens);
        Assert.Equal(1, model.PrimaryConversionCalls);
        Assert.Empty(model.RetainedPayloadKeys);
        await AssertColdParity(other);

        var continuation = Request("other-next", Extend(other, 111, 112), "other");
        Assert.Equal(other.NumComputedTokens, (await Run(engine, continuation)).PrefixCacheReusedTokens);
        Assert.Equal(1, model.PrimaryConversionCalls);
        await AssertColdParity(continuation);

        // Returning to the original scope cannot claim the other conversation's
        // live cache, even when it sends exactly that cache's token sequence.
        var original = Request("original-next", Extend(continuation, 121, 122), "original");
        Assert.Equal(0, (await Run(engine, original)).PrefixCacheReusedTokens);
        await AssertColdParity(original);
    }

    [Fact]
    public async Task ContinuedPrimary_AdmittedWithANeighbor_PreservesBothOutputsAndLaterReuse()
    {
        var model = new OracleModel(Traits, 8);
        using var engine = new InferenceEngine(model, Configuration());
        var first = Request("first", Enumerable.Range(1, 32).ToList(), "original");
        await Run(engine, first);
        var continuation = Request("continuation", Extend(first, 101, 102), "original");
        var neighbor = Request("neighbor", Enumerable.Range(71, 32).ToList(), "neighbor");
        var gate = new ComputeGate();
        gate.Close();
        engine.ComputeGate = gate;
        var continued = engine.SubmitRequest(continuation);
        var other = engine.SubmitRequest(neighbor);
        gate.Open();
        var results = await Task.WhenAll(continued.Completion, other.Completion).WaitAsync(TimeSpan.FromSeconds(10));
        Assert.Equal(first.NumComputedTokens, results[0].PrefixCacheReusedTokens);
        Assert.Equal(0, results[1].PrefixCacheReusedTokens);
        await AssertColdParity(continuation);
        await AssertColdParity(neighbor);

        foreach (var previous in new[] { continuation, neighbor })
        {
            var next = Request(previous.RequestId + "-next", Extend(previous, 111, 112), previous.CacheScope!);
            Assert.Equal(previous.NumComputedTokens, (await Run(engine, next)).PrefixCacheReusedTokens);
            await AssertColdParity(next);
        }
    }

    [Fact]
    public async Task EosFinish_ContinuesTheEntireForwardedState_WithoutConversion()
    {
        var prompt = Enumerable.Range(1, 32).ToList();
        using var predictor = new OracleModel(Traits, 8);
        float[] logits = predictor.Forward(prompt.ToArray());
        int eos = Array.IndexOf(logits, logits.Max());
        var model = new OracleModel(Traits, 8, tokenizer: new OracleTokenizer(OracleModel.DefaultVocab, eos)) { SpareBytes = 1 };
        using var engine = new InferenceEngine(model, Configuration());
        var first = Request("first", prompt, "conversation");
        var completed = await Run(engine, first);
        Assert.Equal(SequenceStatus.FinishedStopped, completed.Status);
        Assert.Equal("eos", completed.FinishReason);
        Assert.Equal(new[] { eos }, first.OutputTokens);
        Assert.Equal(0, model.PrimaryConversionCalls);

        var next = Request("next", Extend(first, 101, 102), "conversation");
        Assert.Equal(first.NumComputedTokens, (await Run(engine, next)).PrefixCacheReusedTokens);
        using var cold = new InferenceEngine(new OracleModel(Traits, 8,
            tokenizer: new OracleTokenizer(OracleModel.DefaultVocab, eos)), Configuration(false));
        var reference = Request("reference", next.PromptTokens, "reference");
        await Run(cold, reference);
        Assert.Equal(reference.OutputTokens, next.OutputTokens);
        Assert.Equal(0, model.PrimaryConversionCalls);
    }

    [Fact]
    public async Task ForwardFailure_AfterClaimingPrimary_DoesNotPublishTheDamagedState()
    {
        var model = new OracleModel(Traits, 8) { SpareBytes = 1 };
        using var engine = new InferenceEngine(model, Configuration());
        var first = Request("first", Enumerable.Range(1, 32).ToList(), "conversation");
        await Run(engine, first);
        model.FailNext("forward");
        var failed = Request("failed", Extend(first, 101, 102), "conversation");
        await Assert.ThrowsAsync<InvalidOperationException>(() => Run(engine, failed));
        Assert.Equal(SequenceStatus.FinishedError, failed.Status);

        var next = Request("next", Extend(first, 111, 112), "conversation");
        Assert.Equal(0, (await Run(engine, next)).PrefixCacheReusedTokens);
        await AssertColdParity(next);
        var following = Request("following", Extend(next, 121, 122), "conversation");
        Assert.Equal(next.NumComputedTokens, (await Run(engine, following)).PrefixCacheReusedTokens);
        await AssertColdParity(following);
    }

    [Fact]
    public async Task AdapterRetentionException_AfterAdoption_DoesNotLeakOrFailTheIncomingRequest()
    {
        foreach (bool published in new[] { false, true })
        {
            var model = new AdaptedOracleModel(new OracleModel(Traits with { CanCaptureCopy = false }, 8));
            using var engine = new InferenceEngine(model, Configuration());
            var first = Request("first", Enumerable.Range(1, 32).ToList(), "original");
            await Run(engine, first);
            model.RetainFailure = new OutOfMemoryException("retained dictionary allocation failed");
            model.RetainFailureAfterPublication = published;

            var other = Request("other", Extend(first, 101, 102), "other");
            var completed = await Run(engine, other);
            Assert.Equal(SequenceStatus.FinishedLengthCapped, completed.Status);
            Assert.Equal(0, completed.PrefixCacheReusedTokens);
            Assert.Equal(0, model.PrivateHolderCount);
            Assert.Empty(model.RetainedPayloadKeys);
            await AssertColdParity(other);

            var continuation = Request("other-next", Extend(other, 111, 112), "other");
            Assert.Equal(other.NumComputedTokens, (await Run(engine, continuation)).PrefixCacheReusedTokens);
            Assert.Equal(0, model.PrivateHolderCount);
            await AssertColdParity(continuation);
        }
    }

    [Fact]
    public async Task ConversionFootprint_IsPublishedWithoutRepeatingItsMeasurement()
    {
        var model = new AdaptedOracleModel(new OracleModel(Traits with { CanCaptureCopy = false }, 8));
        using var engine = new InferenceEngine(model, Configuration());
        var first = Request("first", Enumerable.Range(1, 32).ToList(), "original");
        await Run(engine, first);
        model.FailMeasurementOnCall = 2;

        var other = Request("other", Enumerable.Range(71, 32).ToList(), "other");
        Assert.Equal(0, (await Run(engine, other)).PrefixCacheReusedTokens);
        Assert.Equal(1, model.MeasurementCalls);
        Assert.Single(model.RetainedPayloadKeys);
        Assert.Equal(0, model.PrivateHolderCount);
        await AssertColdParity(other);
    }

    [Fact]
    public async Task DestructiveConversionException_DoesNotFailTheIncomingRequest()
    {
        var model = new OracleModel(Traits, 8) { SpareBytes = 1 };
        model.FailNext("convert-throw");
        using var engine = new InferenceEngine(model, Configuration());
        var first = Request("first", Enumerable.Range(1, 32).ToList(), "original");
        await Run(engine, first);
        Assert.Equal(0, model.PrimaryConversionCalls);

        var other = Request("other", Extend(first, 101, 102), "other");
        var completed = await Run(engine, other);
        Assert.Equal(SequenceStatus.FinishedLengthCapped, completed.Status);
        Assert.Equal(0, completed.PrefixCacheReusedTokens);
        Assert.Equal(1, model.PrimaryConversionCalls);
        await AssertColdParity(other);

        var continuation = Request("other-next", Extend(other, 111, 112), "other");
        Assert.Equal(other.NumComputedTokens, (await Run(engine, continuation)).PrefixCacheReusedTokens);
        await AssertColdParity(continuation);
    }

    [Fact]
    public async Task SuccessfulDisplacement_PreservesTheOldBranch_ForLaterExactReuse()
    {
        var model = new OracleModel(Traits, 8);
        using var engine = new InferenceEngine(model, Configuration());
        var first = Request("first", Enumerable.Range(1, 32).ToList(), "original");
        await Run(engine, first);
        Assert.Equal(0, model.PrimaryConversionCalls);

        var other = Request("other", Enumerable.Range(71, 32).ToList(), "other");
        Assert.Equal(0, (await Run(engine, other)).PrefixCacheReusedTokens);
        Assert.Equal(1, model.PrimaryConversionCalls);
        Assert.Contains(model.RetainedPayloadKeys, key => model.MeasureEndState(key).Tokens == first.NumComputedTokens);
        await AssertColdParity(other);

        var branch = Request("branch", Extend(first, 111, 112), "original");
        Assert.Equal(first.NumComputedTokens, (await Run(engine, branch)).PrefixCacheReusedTokens);
        Assert.Equal(2, model.PrimaryConversionCalls);
        await AssertColdParity(branch);
    }
}
