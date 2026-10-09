// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using InferenceWeb.Tests.PrefixCache.Fakes;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests.PrefixCache;

public sealed class CheckpointPublicationLifetimeTests
{
    private static SchedulerConfig Configuration(bool cache = true) => new()
    {
        BlockSize = 8, NumBlocks = 128, MaxNumRunningSequences = 1,
        MaxNumBatchedTokens = 64, SoloPrefillChunkSize = 64,
        EnablePrefixCaching = cache, StopRepetition = false,
    };

    private static SequenceState Request(string id, List<int> prompt, IReadOnlyList<PromptMediaSpan>? media = null) =>
        new(id, prompt, 3, 8, SamplingConfig.Greedy, cacheScope: "compacted-conversation", mediaSpans: media);

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void ShorterHistoryPublicationSurvivesEvictingItsOnlyDescendant(bool donate)
    {
        using var model = new BoundedRetainedModel();
        var pool = new BlockPool(128, 8, 64);
        var scheduler = new ContinuousBatchScheduler(Configuration(), pool);
        var cache = new PrefixCacheCoordinator(model, pool, scheduler,
            model.GetPrefixCacheCapabilities(), NullLogger.Instance);
        var longer = Request("longer", Enumerable.Range(1, 48).ToList());
        var shorter = Request("shorter", longer.PromptTokens.Take(32).ToList());
        try
        {
            ForwardPrompt(model, pool, longer);
            cache.CaptureCheckpoint(longer);
            string previous = Assert.Single(model.RetainedPayloadKeys);
            model.OnSequenceReleased(longer.RequestId);
            cache.ReleaseRequest(longer);

            ForwardPrompt(model, pool, shorter);
            // Inserting the shorter prefix splits the old edge: the new state
            // belongs on an empty ancestor of the sole old retained state.
            // Capacity is genuinely full until that descendant is released.
            if (donate) Assert.True(cache.RetainFinished(shorter, primary: false));
            else cache.CaptureCheckpoint(shorter);

            Assert.Equal(1, model.CapacityRefusals);
            string current = Assert.Single(model.RetainedPayloadKeys);
            Assert.NotEqual(previous, current);
            Assert.True(cache.Tree.TryGetNodeByKey(current, out var node));
            Assert.True(node.InTree);
            Assert.False(node.IsRoot);
            Assert.Equal(32, node.Depth);
            Assert.Equal(32, node.EndState!.Footprint.Tokens);
            Assert.Equal(0, node.LockRef);
            Assert.False(cache.Tree.Root.HasPayload);
            model.OnSequenceReleased(shorter.RequestId);
            cache.ReleaseRequest(shorter);

            Assert.True(model.TryMaterialize(new MaterializeRequest(MaterializeOp.Clone,
                current, "restored", 32, 32)));
            model.BindSequenceCache("restored");
            float[] restored = model.Forward([149, 150]);
            using var cold = OracleFakes.R(8);
            float[] expected = cold.Forward(shorter.PromptTokens.Concat(new[] { 149, 150 }).ToArray());
            Assert.Equal(expected, restored);
            model.OnSequenceReleased("restored");
        }
        finally
        {
            cache.Reset();
            cache.Detach();
            model.OnSequenceReleased(longer.RequestId);
            model.OnSequenceReleased(shorter.RequestId);
            longer.BlockTable.ReleaseAll(pool);
            shorter.BlockTable.ReleaseAll(pool);
        }
        Assert.Empty(model.RetainedPayloadKeys);
        Assert.Equal(pool.NumBlocks, pool.NumFreeBlocks);
    }

    [Fact]
    public async Task CompletionWaitsUntilFinishedStatePublicationHasSettled()
    {
        using var model = new BoundedRetainedModel();
        using var resume = new ManualResetEventSlim();
        var publishing = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        model.BeforeConversion = () =>
        {
            publishing.TrySetResult();
            if (!resume.Wait(TimeSpan.FromSeconds(10))) throw new TimeoutException("Publication was not resumed.");
        };
        using var engine = new InferenceEngine(model, Configuration());
        var sequence = Request("publication-boundary", Enumerable.Range(1, 48).ToList());
        var handle = engine.SubmitRequest(sequence);
        try
        {
            await publishing.Task.WaitAsync(TimeSpan.FromSeconds(10));
            // The old retained checkpoint has been evicted, but the finished
            // state has not been published yet. This is a real worker boundary,
            // not a timing assumption or a concurrent read of the model's maps.
            Assert.False(handle.Completion.IsCompleted);
            while (handle.Tokens.TryRead(out _)) { }
            Assert.False(handle.Tokens.Completion.IsCompleted);
        }
        finally { resume.Set(); }
        await handle.Completion.WaitAsync(TimeSpan.FromSeconds(10));
        Assert.Single(model.RetainedPayloadKeys);
        Assert.Equal(1, engine.TotalCompleted);
    }

    [Fact]
    public async Task CompactedAgentHistoryUnderRetainedCapacityPressureStillGeneratesAndReusesExactState()
    {
        using var model = new BoundedRetainedModel();
        using var engine = new InferenceEngine(model, Configuration());
        var previous = Request("before-compaction", Enumerable.Range(1, 48).ToList());
        await Complete(engine, previous);
        Assert.Single(model.RetainedPayloadKeys);

        var compacted = Request("after-compaction", previous.PromptTokens.Take(32).ToList());
        InferenceCompletion compactedDone = await Complete(engine, compacted);
        Assert.Equal(0, compactedDone.PrefixCacheReusedTokens); // Recurrent state cannot rewind.
        Assert.True(model.CapacityRefusals > 0);
        await AssertCold(compacted);

        var nextPrompt = compacted.PromptTokens.Concat(compacted.OutputTokens)
            .Take(compacted.NumComputedTokens).Concat(new[] { 149, 150 }).ToList();
        var next = Request("next-tool-round", nextPrompt);
        Assert.Equal(compacted.NumComputedTokens, (await Complete(engine, next)).PrefixCacheReusedTokens);
        await AssertCold(next);
        engine.Dispose();
        Assert.Empty(model.RetainedPayloadKeys);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void MediaHashCollisionNeverPublishesTheFullStateAtRoot(bool donate)
    {
        using var model = new BoundedRetainedModel();
        var pool = new BlockPool(128, 8, 64);
        var scheduler = new ContinuousBatchScheduler(Configuration(), pool);
        var cache = new PrefixCacheCoordinator(model, pool, scheduler,
            model.GetPrefixCacheCapabilities(), NullLogger.Instance);
        // Equal compact keys, distinct complete 256-bit content identities.
        var first = Request("media-a", Enumerable.Range(1, 16).ToList(),
            [new PromptMediaSpan(0, 2, new MediaId256(1, 0, 0, 17).ToString())]);
        var second = Request("media-b", first.PromptTokens.ToList(),
            [new PromptMediaSpan(0, 2, new MediaId256(2, 0, 0, 17).ToString())]);
        try
        {
            ForwardPrompt(model, pool, first);
            cache.CaptureCheckpoint(first);
            string original = Assert.Single(model.RetainedPayloadKeys);
            model.OnSequenceReleased(first.RequestId);
            cache.ReleaseRequest(first);
            ForwardPrompt(model, pool, second);
            if (donate) Assert.False(cache.RetainFinished(second, primary: false));
            else cache.CaptureCheckpoint(second);
            Assert.Equal(original, Assert.Single(model.RetainedPayloadKeys));
            Assert.Equal(0, model.CapacityRefusals); // No model capture for a mismatched key.
            Assert.False(cache.Tree.Root.HasPayload);
            Assert.True(cache.Tree.Counters.MediaHashCollisions > 0);
            Assert.Equal(16, model.ActiveLength); // Execution state was not moved or discarded.
        }
        finally
        {
            cache.Reset();
            cache.Detach();
            model.OnSequenceReleased(first.RequestId);
            model.OnSequenceReleased(second.RequestId);
            first.BlockTable.ReleaseAll(pool);
            second.BlockTable.ReleaseAll(pool);
        }
        Assert.Equal(pool.NumBlocks, pool.NumFreeBlocks);
    }

    private static void ForwardPrompt(OracleModel model, BlockPool pool, SequenceState sequence)
    {
        int blocks = (sequence.PromptTokens.Count + pool.BlockSize - 1) / pool.BlockSize;
        foreach (var block in pool.AllocateNew(blocks)!) sequence.BlockTable.AppendBlock(block);
        model.BindSequenceCache(sequence.RequestId);
        model.Forward(sequence.PromptTokens.ToArray());
        sequence.SetComputedTokensForPrefixAdoption(sequence.PromptTokens.Count);
    }

    private static async Task<InferenceCompletion> Complete(InferenceEngine engine, SequenceState sequence)
    {
        var result = await engine.SubmitRequest(sequence).Completion.WaitAsync(TimeSpan.FromSeconds(10));
        Assert.Null(sequence.Error);
        return result;
    }

    private static async Task AssertCold(SequenceState sequence)
    {
        using var cold = new InferenceEngine(OracleFakes.R(8), Configuration(cache: false));
        var reference = Request("cold", sequence.PromptTokens.ToList());
        await Complete(cold, reference);
        Assert.Equal(reference.OutputTokens, sequence.OutputTokens);
    }

    private sealed class BoundedRetainedModel : OracleModel, IPrefixCacheModel
    {
        internal int CapacityRefusals { get; private set; }
        internal Action? BeforeConversion { get; set; }
        internal BoundedRetainedModel() : base(new OracleTraits
        {
            Name = "one-retained-recurrent", Class = FamilyClass.R,
            EndState = EndStateSupport.CopyAndDonate, CanCaptureCopy = true,
            AdoptPrimaryOnDisplacement = true, DeviceDirtyOnForward = true,
            Truncation = TruncationKind.None, Pages = PageSupport.None,
        }, 8) { }

        bool IPrefixCacheModel.TryCaptureCopy(string requestId, string payloadKey, out PayloadFootprint footprint)
        {
            if (RetainedPayloadKeys.Count == 0) return TryCaptureCopy(requestId, payloadKey, out footprint);
            CapacityRefusals++;
            footprint = default;
            return false;
        }

        bool IPrefixCacheModel.TryCaptureDonate(string requestId, string payloadKey, int length, out PayloadFootprint footprint)
        {
            if (RetainedPayloadKeys.Count == 0) return TryCaptureDonate(requestId, payloadKey, length, out footprint);
            CapacityRefusals++;
            footprint = default;
            return false;
        }

        bool IPrefixCacheModel.TryConvertPrimary(string payloadKey, int length, out PayloadFootprint footprint)
        {
            if (RetainedPayloadKeys.Count == 0)
            {
                BeforeConversion?.Invoke();
                return TryConvertPrimary(payloadKey, length, out footprint);
            }
            CapacityRefusals++;
            footprint = default;
            return false;
        }
    }
}
