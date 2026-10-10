using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using InferenceWeb.Tests.PrefixCache.Fakes;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;
using Xunit;

namespace InferenceWeb.Tests.PrefixCache;

public class RadixHolderEngineTests
{
    private static SchedulerConfig Configuration(bool enabled = true, int numBlocks = 256) => new()
    {
        BlockSize = 8, NumBlocks = numBlocks, MaxNumBatchedTokens = 64,
        MaxPrefillChunkSize = 16, SoloPrefillChunkSize = 64,
        EnablePrefixCaching = enabled, StopRepetition = false,
    };

    private static List<int> Tokens(int count, int start = 1) => Enumerable.Range(start, count).ToList();

    private static SequenceState Request(string id, List<int> tokens, string? scope = "conversation",
        int shared = 0, IReadOnlyList<int>? breaks = null) => new(id, tokens, 3, 8, SamplingConfig.Greedy,
            cacheScope: scope, sharedPrefixTokens: shared, cacheBreakpoints: breaks);

    private static async Task<InferenceCompletion> Run(InferenceEngine engine, SequenceState sequence)
        => await engine.SubmitRequest(sequence).Completion.WaitAsync(TimeSpan.FromSeconds(10));

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task FinishedHolder_ReusesExactState_WithSequentialRequestIdReuse(bool recurrent)
    {
        var model = recurrent ? OracleFakes.R(8) : OracleFakes.S(8);
        using var engine = new InferenceEngine(model, Configuration());
        Assert.True(engine.PrefixCacheActive);
        var first = Request("repeat", Tokens(32));
        await Run(engine, first);
        var prompt = first.PromptTokens.Concat(first.OutputTokens)
            .Take(first.NumComputedTokens).Concat(new[] { 101, 102, 103 }).ToList();
        var second = Request("repeat", prompt);
        var completion = await Run(engine, second);
        Assert.Equal(first.NumComputedTokens, completion.PrefixCacheReusedTokens);

        using var cold = new InferenceEngine(recurrent ? OracleFakes.R(8) : OracleFakes.S(8), Configuration(false));
        var baseline = Request("cold", prompt);
        await Run(cold, baseline);
        Assert.Equal(baseline.OutputTokens, second.OutputTokens);
        Assert.All(model.RetainedPayloadKeys, key => Assert.StartsWith("pc:", key));
    }

    [Fact]
    public async Task PublicCheckpoint_IsClonedAcrossScopes_AndSurvivesEngineRestart()
    {
        var store = new MemoryCheckpointStore();
        using (var engine = new InferenceEngine(OracleFakes.R(8), Configuration()))
        {
            engine.PrefixCheckpointStore = store;
            await Run(engine, Request("seed", Tokens(32), "a", shared: 16));
            var prompt = Tokens(16).Concat(Tokens(16, 81)).ToList();
            var second = Request("other", prompt, "b", shared: 16);
            Assert.Equal(16, (await Run(engine, second)).PrefixCacheReusedTokens);
        }
        Assert.NotNull(store.Bytes);
        using var restarted = new InferenceEngine(OracleFakes.R(8), Configuration());
        restarted.PrefixCheckpointStore = store;
        var after = Request("restart", Tokens(16).Concat(Tokens(16, 61)).ToList(), "c", shared: 16);
        Assert.Equal(16, (await Run(restarted, after)).PrefixCacheReusedTokens);
    }

    [Fact]
    public void PublicCheckpointBoundaries_AreImmutableHintsWithinThePublicPrefix()
    {
        var hints = new List<int> { 19, -1, 0, 19, 70, 11 };
        var sequence = new SequenceState("hints", Tokens(61), 3, 8, SamplingConfig.Greedy,
            sharedPrefixTokens: 37, publicCheckpointBoundaries: hints);
        hints.Clear();
        Assert.Equal(new[] { 11, 19, 37 }, sequence.PublicCheckpointBoundaries);
        Assert.Equal(37, sequence.SharedPrefixTokens);
    }

    [Fact]
    public async Task ParentAndChildren_ReuseExactAncestorThenFullBranches_WithinTwoPublicSlots()
    {
        const string setting = "TS_PREFIX_CHECKPOINTS_MAX";
        string? previous = Environment.GetEnvironmentVariable(setting);
        Environment.SetEnvironmentVariable(setting, "2");
        try
        {
            const int ancestor = 19, parentPublic = 47, childPublic = 43;
            var parentPrompt = Tokens(61);
            var childPrefix = Tokens(ancestor).Concat(Tokens(childPublic - ancestor, 81)).ToList();
            var prompts = new[] { childPrefix.Concat(Tokens(18, 121)).ToList(), childPrefix.Concat(Tokens(19, 151)).ToList() };
            var gate = new ComputeGate();
            var model = new CountingRecurrentOracle();
            using var engine = new InferenceEngine(model, Configuration()) { ComputeGate = gate };
            var parent = new SequenceState("parent", parentPrompt, 3, 8, SamplingConfig.Greedy,
                cacheScope: "root", sharedPrefixTokens: parentPublic, publicCheckpointBoundaries: new[] { ancestor });
            await Run(engine, parent);
            Assert.Contains(model.RetainedPayloadKeys, key => model.MeasureEndState(key).Tokens == ancestor);

            gate.Close();
            int before = model.ForwardedTokens;
            var children = prompts.Select((prompt, index) => new SequenceState("child-" + index, prompt, 3, 8,
                SamplingConfig.Greedy, cacheScope: "child-scope-" + index, sharedPrefixTokens: childPublic,
                publicCheckpointBoundaries: new[] { ancestor })).ToArray();
            var handles = children.Select(child => engine.SubmitRequest(child)).ToArray();
            gate.Open();
            var completions = await Task.WhenAll(handles.Select(handle => handle.Completion)).WaitAsync(TimeSpan.FromSeconds(10));
            Assert.Equal(ancestor, completions[0].PrefixCacheReusedTokens);
            Assert.Equal(childPublic, completions[1].PrefixCacheReusedTokens);
            int forwarded = model.ForwardedTokens - before;

            var coldModel = new CountingRecurrentOracle();
            using var cold = new InferenceEngine(coldModel, Configuration(false));
            foreach (var child in children)
            {
                var reference = Request("cold-" + child.RequestId, child.PromptTokens);
                await Run(cold, reference);
                Assert.Equal(reference.OutputTokens, child.OutputTokens);
            }
            Assert.Equal(ancestor + childPublic, coldModel.ForwardedTokens - forwarded);
            // Three public states cannot fit. Once both branches are available,
            // the shorter ancestor is evicted instead of either warm endpoint.
            Assert.DoesNotContain(model.RetainedPayloadKeys, key => model.MeasureEndState(key).Tokens == ancestor);
            var warmParent = Request("warm-parent", parentPrompt, "new-root", shared: parentPublic);
            Assert.Equal(parentPublic, (await Run(engine, warmParent)).PrefixCacheReusedTokens);
            Assert.Equal(parent.OutputTokens, warmParent.OutputTokens);
            var warmChild = Request("warm-child", prompts[0], "new-child", shared: childPublic);
            Assert.Equal(childPublic, (await Run(engine, warmChild)).PrefixCacheReusedTokens);
            Assert.Equal(children[0].OutputTokens, warmChild.OutputTokens);
        }
        finally { Environment.SetEnvironmentVariable(setting, previous); }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task AncestorCheckpoint_RespectsExplicitProducerPolicy(bool marked)
    {
        using var engine = new InferenceEngine(OracleFakes.R(8), Configuration());
        var parent = new SequenceState("parent", Tokens(61), 3, 8, SamplingConfig.Greedy,
            cacheScope: "root", sharedPrefixTokens: 47, publicCheckpointBoundaries: new[] { 19 },
            cacheBreakpoints: marked ? new[] { 19 } : Array.Empty<int>());
        await Run(engine, parent);
        var child = new SequenceState("child", Tokens(19).Concat(Tokens(42, 81)).ToList(), 3, 8,
            SamplingConfig.Greedy, cacheScope: "child", sharedPrefixTokens: 43,
            publicCheckpointBoundaries: new[] { 19 });
        Assert.Equal(marked ? 19 : 0, (await Run(engine, child)).PrefixCacheReusedTokens);
        using var cold = new InferenceEngine(OracleFakes.R(8), Configuration(false));
        var reference = Request("reference", child.PromptTokens);
        await Run(cold, reference);
        Assert.Equal(reference.OutputTokens, child.OutputTokens);
    }

    [Fact]
    public async Task AncestorCheckpoint_IsRestoredForADifferentFullPrefixAfterRestart()
    {
        var store = new MultipleCheckpointStore();
        using (var seed = new InferenceEngine(OracleFakes.R(8), Configuration()))
        {
            seed.PrefixCheckpointStore = store;
            await Run(seed, new SequenceState("parent", Tokens(61), 3, 8, SamplingConfig.Greedy,
                sharedPrefixTokens: 47, publicCheckpointBoundaries: new[] { 19 }));
        }
        Assert.Equal(new[] { 19, 47 }, store.SavedLengths.Order());
        using var engine = new InferenceEngine(OracleFakes.R(8), Configuration());
        engine.PrefixCheckpointStore = store;
        var child = new SequenceState("child", Tokens(19).Concat(Tokens(42, 81)).ToList(), 3, 8,
            SamplingConfig.Greedy, cacheScope: "child", sharedPrefixTokens: 43,
            publicCheckpointBoundaries: new[] { 19 });
        Assert.Equal(19, (await Run(engine, child)).PrefixCacheReusedTokens);
        Assert.Equal(new[] { 43, 19 }, store.OpenedLengths.TakeLast(2));
        using var cold = new InferenceEngine(OracleFakes.R(8), Configuration(false));
        var reference = Request("reference", child.PromptTokens);
        await Run(cold, reference);
        Assert.Equal(reference.OutputTokens, child.OutputTokens);
    }

    [Fact]
    public async Task ResidentFullCheckpoint_PreventsImportingAnEvictedShorterAncestor_AfterEarlierDiskMiss()
    {
        const string setting = "TS_PREFIX_CHECKPOINTS_MAX";
        string? previous = Environment.GetEnvironmentVariable(setting);
        Environment.SetEnvironmentVariable(setting, "2");
        try
        {
            var store = new MultipleCheckpointStore();
            using var engine = new InferenceEngine(OracleFakes.R(8), Configuration());
            engine.PrefixCheckpointStore = store;
            await Run(engine, new SequenceState("parent", Tokens(61), 3, 8, SamplingConfig.Greedy,
                cacheScope: "parent", sharedPrefixTokens: 47, publicCheckpointBoundaries: new[] { 19 }));
            await Run(engine, new SequenceState("child", Tokens(19).Concat(Tokens(42, 81)).ToList(), 3, 8,
                SamplingConfig.Greedy, cacheScope: "child", sharedPrefixTokens: 43,
                publicCheckpointBoundaries: new[] { 19 }));
            int opens = store.OpenedLengths.Count;
            var next = new SequenceState("next-parent", Tokens(61), 3, 8, SamplingConfig.Greedy,
                cacheScope: "next-parent", sharedPrefixTokens: 47, publicCheckpointBoundaries: new[] { 19 });
            Assert.Equal(47, (await Run(engine, next)).PrefixCacheReusedTokens);
            Assert.Equal(opens, store.OpenedLengths.Count);
        }
        finally { Environment.SetEnvironmentVariable(setting, previous); }
    }

    [Fact]
    public async Task SavedAncestor_ClearsEarlierDiskMiss_AndRestoresForAThirdBranchWithinTwoPublicSlots()
    {
        const string setting = "TS_PREFIX_CHECKPOINTS_MAX";
        string? previous = Environment.GetEnvironmentVariable(setting);
        Environment.SetEnvironmentVariable(setting, "2");
        try
        {
            var store = new MultipleCheckpointStore();
            var model = OracleFakes.R(8);
            using var engine = new InferenceEngine(model, Configuration());
            engine.PrefixCheckpointStore = store;
            await Run(engine, new SequenceState("parent", Tokens(61), 3, 8, SamplingConfig.Greedy,
                cacheScope: "parent", sharedPrefixTokens: 47, publicCheckpointBoundaries: new[] { 19 }));
            Assert.Contains(19, store.OpenedLengths); // Missed before the parent captured it.
            await Run(engine, new SequenceState("child", Tokens(19).Concat(Tokens(42, 81)).ToList(), 3, 8,
                SamplingConfig.Greedy, cacheScope: "child", sharedPrefixTokens: 43,
                publicCheckpointBoundaries: new[] { 19 }));
            Assert.DoesNotContain(model.RetainedPayloadKeys, key => model.MeasureEndState(key).Tokens == 19);
            int previousAncestorOpens = store.OpenedLengths.Count(length => length == 19);
            var third = new SequenceState("third-role", Tokens(19).Concat(Tokens(42, 141)).ToList(), 3, 8,
                SamplingConfig.Greedy, cacheScope: "third-role", sharedPrefixTokens: 41,
                publicCheckpointBoundaries: new[] { 19 });
            Assert.Equal(19, (await Run(engine, third)).PrefixCacheReusedTokens);
            Assert.Equal(previousAncestorOpens + 1, store.OpenedLengths.Count(length => length == 19));
            Assert.DoesNotContain(model.RetainedPayloadKeys, key => model.MeasureEndState(key).Tokens == 19);
            Assert.InRange(model.RetainedPayloadKeys.Count(key => model.MeasureEndState(key).Tokens is 19 or 41 or 43 or 47), 1, 2);
            using var cold = new InferenceEngine(OracleFakes.R(8), Configuration(false));
            var reference = Request("reference", third.PromptTokens);
            await Run(cold, reference);
            Assert.Equal(reference.OutputTokens, third.OutputTokens);
        }
        finally { Environment.SetEnvironmentVariable(setting, previous); }
    }

    [Fact]
    public async Task LegacyFullCheckpoint_DoesNotInventAnEarlierRecurrentState()
    {
        var store = new MultipleCheckpointStore();
        using (var legacy = new InferenceEngine(OracleFakes.R(8), Configuration()))
        {
            legacy.PrefixCheckpointStore = store;
            await Run(legacy, Request("legacy-parent", Tokens(61), "root", shared: 47));
        }
        Assert.Equal(new[] { 47 }, store.SavedLengths);
        using var engine = new InferenceEngine(OracleFakes.R(8), Configuration());
        engine.PrefixCheckpointStore = store;
        var parent = new SequenceState("parent", Tokens(61), 3, 8, SamplingConfig.Greedy,
            cacheScope: "root", sharedPrefixTokens: 47, publicCheckpointBoundaries: new[] { 19 });
        Assert.Equal(47, (await Run(engine, parent)).PrefixCacheReusedTokens);
        var prompt = Tokens(19).Concat(Tokens(42, 81)).ToList();
        var child = new SequenceState("child", prompt, 3, 8, SamplingConfig.Greedy,
            cacheScope: "child", sharedPrefixTokens: 43, publicCheckpointBoundaries: new[] { 19 });
        Assert.Equal(0, (await Run(engine, child)).PrefixCacheReusedTokens);
        var warm = Request("warm-child", prompt, "new-child", shared: 43);
        Assert.Equal(43, (await Run(engine, warm)).PrefixCacheReusedTokens);
        using var cold = new InferenceEngine(OracleFakes.R(8), Configuration(false));
        var reference = Request("reference", prompt);
        await Run(cold, reference);
        Assert.Equal(reference.OutputTokens, child.OutputTokens);
        Assert.Equal(reference.OutputTokens, warm.OutputTokens);
    }

    [Fact]
    public void PendingAncestorCheckpoint_CanServeADifferentFullPublicPrefix()
    {
        using var model = OracleFakes.R(8);
        var (_, cache) = PendingCheckpointScheduler(model);
        var parent = new SequenceState("parent", Tokens(61), 3, 8, SamplingConfig.Greedy,
            sharedPrefixTokens: 47, publicCheckpointBoundaries: new[] { 19 });
        parent.Status = SequenceStatus.Running;
        var child = new SequenceState("child", Tokens(19).Concat(Tokens(42, 81)).ToList(), 3, 8,
            SamplingConfig.Greedy, sharedPrefixTokens: 43, publicCheckpointBoundaries: new[] { 19 });
        Assert.True(cache!.CanSharePendingPublicCheckpoint(child, parent));
        Assert.False(cache.CanSharePendingPublicCheckpoint(child, parent, reusableLength: 19));
        var noReuse = new SequenceState("disabled", child.PromptTokens, 3, 8, SamplingConfig.Greedy,
            sharedPrefixTokens: 43, publicCheckpointBoundaries: new[] { 19 }, cacheBreakpoints: Array.Empty<int>());
        Assert.False(cache.CanSharePendingPublicCheckpoint(noReuse, parent));
        parent.Status = SequenceStatus.FinishedAborted;
        Assert.False(cache.CanSharePendingPublicCheckpoint(child, parent));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void DisabledPublicCheckpoints_DoNotSplitPrefill_ButPreservePrivateBreakpoints(bool privateBreakpoint)
    {
        const string publicSetting = "TS_PREFIX_CHECKPOINTS_MAX", retainedSetting = "TS_RETAINED_FUSED_CACHE_MAX";
        string? previousPublic = Environment.GetEnvironmentVariable(publicSetting);
        string? previousRetained = Environment.GetEnvironmentVariable(retainedSetting);
        Environment.SetEnvironmentVariable(publicSetting, "0");
        Environment.SetEnvironmentVariable(retainedSetting, null);
        try
        {
            using var model = OracleFakes.R(8);
            var (scheduler, cache) = PendingCheckpointScheduler(model);
            Assert.True(cache!.CheckpointsSupported);
            Assert.False(cache.PublicCheckpointsSupported);
            var sequence = new SequenceState("parent", Tokens(61), 3, 8, SamplingConfig.Greedy,
                sharedPrefixTokens: 37, publicCheckpointBoundaries: new[] { 11 },
                cacheBreakpoints: privateBreakpoint ? new[] { 53 } : null);
            scheduler.Submit(sequence);
            Assert.Equal(privateBreakpoint ? 53 : 61, Assert.Single(scheduler.Schedule().ScheduledWork).NumScheduledTokens);
        }
        finally
        {
            Environment.SetEnvironmentVariable(publicSetting, previousPublic);
            Environment.SetEnvironmentVariable(retainedSetting, previousRetained);
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ConcurrentSiblings_ComputeTheirPublicPrefixOnce_AndMatchColdOutputs(bool warm)
    {
        const int shared = 37; // Deliberately neither a page nor a prefill boundary.
        var prefix = Tokens(shared);
        var prompts = new[] { prefix.Concat(Tokens(24, 81)).ToList(), prefix.Concat(Tokens(25, 111)).ToList() };
        var cached = await RunConcurrentSiblings(prompts, shared, enabled: true, warm: warm);
        var cold = await RunConcurrentSiblings(prompts, shared, enabled: false);

        Assert.Equal(warm ? shared : 0, cached.Completions[0].PrefixCacheReusedTokens);
        Assert.Equal(shared, cached.Completions[1].PrefixCacheReusedTokens);
        Assert.All(cold.Completions, completion => Assert.Equal(0, completion.PrefixCacheReusedTokens));
        for (int i = 0; i < prompts.Length; i++)
            Assert.Equal(cold.Requests[i].OutputTokens, cached.Requests[i].OutputTokens);
        Assert.Equal((warm ? 2 : 1) * shared, cold.ForwardedTokens - cached.ForwardedTokens);
    }

    [Fact]
    public async Task RefusedPublicCapture_ReleasesTheSiblingToPrefillCorrectly()
    {
        const int shared = 37;
        var prompts = new[] { Tokens(shared).Concat(Tokens(24, 81)).ToList(), Tokens(shared).Concat(Tokens(25, 111)).ToList() };
        var refused = await RunConcurrentSiblings(prompts, shared, enabled: true, failCapture: true);
        var cold = await RunConcurrentSiblings(prompts, shared, enabled: false);
        Assert.All(refused.Completions, completion => Assert.Equal(0, completion.PrefixCacheReusedTokens));
        Assert.Equal(cold.ForwardedTokens, refused.ForwardedTokens);
        for (int i = 0; i < prompts.Length; i++)
            Assert.Equal(cold.Requests[i].OutputTokens, refused.Requests[i].OutputTokens);
    }

    [Fact]
    public async Task PublicPrefixSiblings_WithCapacityForOnlyOnePrompt_CompleteCorrectly()
    {
        const int shared = 37;
        var prompts = new[] { Tokens(shared).Concat(Tokens(24, 81)).ToList(), Tokens(shared).Concat(Tokens(25, 111)).ToList() };
        // Seventy-two token slots fit either request, but not both prompts. The
        // sibling must not hold resources that prevent its producer finishing.
        var pressured = await RunConcurrentSiblings(prompts, shared, enabled: true, numBlocks: 9);
        var cold = await RunConcurrentSiblings(prompts, shared, enabled: false, numBlocks: 9);
        Assert.Equal(0, pressured.Completions[0].PrefixCacheReusedTokens);
        Assert.Equal(shared, pressured.Completions[1].PrefixCacheReusedTokens);
        Assert.Equal(shared, cold.ForwardedTokens - pressured.ForwardedTokens);
        for (int i = 0; i < prompts.Length; i++)
            Assert.Equal(cold.Requests[i].OutputTokens, pressured.Requests[i].OutputTokens);
    }

    [Fact]
    public void DeferredSibling_DoesNotBlockAnUnrelatedWaitingRequest()
    {
        using var model = OracleFakes.R(8);
        var (scheduler, _) = PendingCheckpointScheduler(model);
        var first = Request("producer", Tokens(61), "a", shared: 37);
        var sibling = Request("sibling", Tokens(37).Concat(Tokens(24, 81)).ToList(), "b", shared: 37);
        var unrelated = Request("unrelated", Tokens(48, 101), "c", shared: 37);
        scheduler.Submit(first);
        scheduler.Submit(sibling);
        scheduler.Submit(unrelated);

        SchedulerOutput step = scheduler.Schedule();
        Assert.Equal(new[] { "producer", "unrelated" }, step.ScheduledWork.Select(work => work.Sequence.RequestId));
        Assert.Equal(SequenceStatus.Waiting, sibling.Status);
        Assert.Equal(1, scheduler.WaitingCount);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void FinishedProducer_DoesNotKeepItsSiblingWaiting(bool error)
    {
        using var model = OracleFakes.R(8);
        var (scheduler, _) = PendingCheckpointScheduler(model);
        var first = Request("producer", Tokens(61), "a", shared: 37);
        var sibling = Request("sibling", Tokens(37).Concat(Tokens(24, 81)).ToList(), "b", shared: 37);
        scheduler.Submit(first);
        scheduler.Submit(sibling);
        Assert.Single(scheduler.Schedule().ScheduledWork);
        Assert.Equal(SequenceStatus.Waiting, sibling.Status);
        if (error) scheduler.NotifyError(first, new InvalidOperationException("producer failed"));
        else Assert.True(scheduler.Abort(first.RequestId));

        Assert.Same(sibling, Assert.Single(scheduler.Schedule().ScheduledWork).Sequence);
        Assert.Equal(0, sibling.PrefixCacheReusedTokens);
    }

    [Theory]
    [InlineData(false, false)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void MissingEligiblePublicCheckpoint_DoesNotDeferSiblings(bool enabled, bool explicitNone)
    {
        using var model = OracleFakes.R(8);
        var (scheduler, _) = PendingCheckpointScheduler(model, enabled);
        var first = Request("producer", Tokens(61), "a", shared: explicitNone ? 37 : 0,
            breaks: explicitNone ? Array.Empty<int>() : null);
        var sibling = Request("sibling", Tokens(61), "b", shared: 37);
        scheduler.Submit(first);
        scheduler.Submit(sibling);
        Assert.Equal(2, scheduler.Schedule().ScheduledWork.Count);
        Assert.Equal(0, scheduler.WaitingCount);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PendingPublicCheckpoint_RequiresExactMediaIdentityAndAnEligibleConsumer(bool sameMedia)
    {
        using var model = new CountingRecurrentOracle();
        var (_, cache) = PendingCheckpointScheduler(model);
        string image = new string('0', 63) + "1";
        string differentImageWithSameAbbreviatedKey = "1" + new string('0', 62) + "1";
        var producer = new SequenceState("producer", Tokens(61), 3, 8, SamplingConfig.Greedy,
            mediaSpans: new[] { new PromptMediaSpan(8, 12, image) }, sharedPrefixTokens: 37);
        producer.Status = SequenceStatus.Running;
        var consumer = new SequenceState("consumer", Tokens(61), 3, 8, SamplingConfig.Greedy,
            mediaSpans: new[] { new PromptMediaSpan(8, 12, sameMedia ? image : differentImageWithSameAbbreviatedKey) },
            sharedPrefixTokens: 37);
        Assert.Equal(sameMedia, cache!.CanSharePendingPublicCheckpoint(consumer, producer));

        var disabled = Request("disabled", Tokens(61), "b", shared: 37, breaks: Array.Empty<int>());
        Assert.False(cache.CanSharePendingPublicCheckpoint(disabled, producer));
        var pool = new BlockPool(8, 8, 64);
        foreach (var block in pool.AllocateNew(5)!) producer.BlockTable.AppendBlock(block);
        producer.AdvanceComputedTokens(37);
        Assert.False(cache.CanSharePendingPublicCheckpoint(consumer, producer));
    }

    private static (ContinuousBatchScheduler Scheduler, PrefixCacheCoordinator? Cache) PendingCheckpointScheduler(
        OracleModel model, bool enabled = true)
    {
        SchedulerConfig cfg = Configuration(enabled);
        var pool = new BlockPool(cfg.NumBlocks, cfg.BlockSize, 64);
        var scheduler = new ContinuousBatchScheduler(cfg, pool);
        var executor = new BatchExecutor(model, pool, scheduler);
        executor.InitializeRadixCache(cfg);
        if (executor.PrefixCheckpointsSupported) scheduler.EnablePrefixCheckpoints();
        return (scheduler, executor.RadixCache);
    }

    private sealed record ConcurrentResult(SequenceState[] Requests, InferenceCompletion[] Completions, int ForwardedTokens);

    private static async Task<ConcurrentResult> RunConcurrentSiblings(List<int>[] prompts, int shared,
        bool enabled, bool warm = false, bool failCapture = false, int numBlocks = 256)
    {
        var gate = new ComputeGate();
        var model = new CountingRecurrentOracle();
        using var engine = new InferenceEngine(model, Configuration(enabled, numBlocks)) { ComputeGate = gate };
        if (warm)
            await Run(engine, new SequenceState("warmup", prompts[0].Take(shared).ToList(), 1, 8,
                SamplingConfig.Greedy, sharedPrefixTokens: shared));
        gate.Close();
        if (failCapture) model.FailNext("capture");
        int before = model.ForwardedTokens;
        var requests = prompts.Select((prompt, index) => Request("sibling-" + index, prompt,
            "separate-scope-" + index, shared: shared)).ToArray();
        var handles = requests.Select(request => engine.SubmitRequest(request)).ToArray();
        gate.Open();
        var completions = await Task.WhenAll(handles.Select(handle => handle.Completion)).WaitAsync(TimeSpan.FromSeconds(10));
        Assert.All(completions, completion => Assert.Equal(SequenceStatus.FinishedLengthCapped, completion.Status));
        return new ConcurrentResult(requests, completions, model.ForwardedTokens - before);
    }

    private sealed class CountingRecurrentOracle() : OracleModel(new OracleTraits
    {
        Name = "counted-recurrent", Class = FamilyClass.R, EndState = EndStateSupport.CopyAndDonate,
        CanCaptureCopy = true, AdoptPrimaryOnDisplacement = true, Truncation = TruncationKind.None,
        DeviceDirtyOnForward = true,
    }, 8), IModelArchitecture
    {
        private int _forwardedTokens;
        internal int ForwardedTokens => Volatile.Read(ref _forwardedTokens);
        float[] IModelArchitecture.Forward(int[] tokens)
        {
            float[] logits = base.Forward(tokens);
            Interlocked.Add(ref _forwardedTokens, tokens.Length);
            return logits;
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(13)]
    public async Task WholeSystemPromptWarmup_IsPublicAcrossNewSessions_AndPersists(int prefixLength)
    {
        var store = new MemoryCheckpointStore();
        using (var engine = new InferenceEngine(OracleFakes.R(8), Configuration()))
        {
            engine.PrefixCheckpointStore = store;
            var warmup = new SequenceState("startup-warmup", Tokens(prefixLength), 1, 8,
                SamplingConfig.Greedy, sharedPrefixTokens: prefixLength);
            Assert.Equal(prefixLength, warmup.SharedPrefixTokens);
            await Run(engine, warmup);

            foreach (string scope in new[] { "first-session", "new-session" })
            {
                var prompt = Tokens(prefixLength).Concat(new[] { 81, 82, 83 }).ToList();
                var request = Request(scope, prompt, scope, shared: prefixLength);
                Assert.Equal(prefixLength, (await Run(engine, request)).PrefixCacheReusedTokens);

                using var cold = new InferenceEngine(OracleFakes.R(8), Configuration(false));
                var reference = Request("cold", prompt);
                await Run(cold, reference);
                Assert.Equal(reference.OutputTokens, request.OutputTokens);
            }
        }
        Assert.NotNull(store.Bytes);
        using var restarted = new InferenceEngine(OracleFakes.R(8), Configuration());
        restarted.PrefixCheckpointStore = store;
        var afterRestart = Request("restarted-session", Tokens(prefixLength).Concat(new[] { 91, 92 }).ToList(),
            "restarted-session", shared: prefixLength);
        Assert.Equal(prefixLength, (await Run(restarted, afterRestart)).PrefixCacheReusedTokens);
    }

    /// <summary>Qwen 3.5's prefix-cache shape (exact-length holders, persistable) with its
    /// real media rule: a reused prefix may continue across an image span
    /// (Qwen35Model's <c>ReuseAcrossMediaSpan</c>).</summary>
    private static OracleModel MediaRecurrent() => new(new OracleTraits
    {
        Name = "R-media", Class = FamilyClass.R, EndState = EndStateSupport.CopyAndDonate,
        CanCaptureCopy = true, AdoptPrimaryOnDisplacement = true, Persistable = true,
        DeviceDirtyOnForward = true, Truncation = TruncationKind.None, Pages = PageSupport.None,
        ReuseAcrossMediaSpan = true,
    }, 8);

    private static SequenceState ImageTurn(string id, List<int> prompt, IReadOnlyList<PromptMediaSpan> spans,
        int shared, string scope) => new(id, prompt, 3, 8, SamplingConfig.Greedy,
            mediaSpans: spans, cacheScope: scope, sharedPrefixTokens: shared);

    private static List<int> Continue(SequenceState previous, params int[] tail) =>
        previous.PromptTokens.Concat(previous.OutputTokens).Take(previous.NumComputedTokens).Concat(tail).ToList();

    // Live, on Qwen3.8 27B: the warm-up published the 7,219-token shared prompt, and an
    // image chat right after it logged "reuse=0", because the attachment's name was in a
    // tool declaration inside that prefix. With the prefix identical, an image placed after
    // it is the ordinary case: the chat reuses the warm-up's checkpoint, a text follow-up
    // its own turn through the image, and a later turn that attaches another file the turn
    // before it -- each with the output a cold engine computes.
    [Fact]
    public async Task ImageChat_ReusesTheWarmupPrefix_AndEachFollowUpReusesThePreviousTurn()
    {
        const int shared = 16;
        List<int> system = Tokens(shared);
        var store = new MultipleCheckpointStore();
        using var engine = new InferenceEngine(MediaRecurrent(), Configuration());
        engine.PrefixCheckpointStore = store;
        await Run(engine, new SequenceState("warmup", system.Concat(new[] { 201, 202 }).ToList(), 1, 8,
            SamplingConfig.Greedy, cacheScope: "warmup", sharedPrefixTokens: shared));
        Assert.Contains(shared, store.SavedLengths);

        var cat = new PromptMediaSpan(shared + 3, shared + 11, new string('a', 64));
        List<int> imagePrompt = system.Concat(new[] { 203, 204, 205 })
            .Concat(Enumerable.Repeat(240, 8)).Concat(new[] { 206, 207 }).ToList();
        var image = ImageTurn("image", imagePrompt, new[] { cat }, shared, "chat");
        Assert.Equal(shared, (await Run(engine, image)).PrefixCacheReusedTokens);

        var text = ImageTurn("text", Continue(image, 208, 209), new[] { cat }, shared, "chat");
        int textReuse = (await Run(engine, text)).PrefixCacheReusedTokens;
        Assert.True(textReuse >= cat.End, $"the follow-up reused {textReuse}, not past the image ({cat.End})");
        Assert.Equal(image.NumComputedTokens, textReuse);

        List<int> laterPrompt = Continue(text, 210);
        var dog = new PromptMediaSpan(laterPrompt.Count, laterPrompt.Count + 8, new string('b', 64));
        laterPrompt.AddRange(Enumerable.Repeat(241, 8));
        laterPrompt.Add(211);
        var later = ImageTurn("later", laterPrompt, new[] { cat, dog }, shared, "chat");
        Assert.True((await Run(engine, later)).PrefixCacheReusedTokens >= text.NumComputedTokens);
        // The media prefix is never what a store holds: only the text-only shared prefix was saved.
        Assert.All(store.SavedLengths, length => Assert.True(length <= cat.Start));

        using var cold = new InferenceEngine(MediaRecurrent(), Configuration(false));
        foreach (SequenceState warm in new[] { image, text, later })
        {
            var reference = new SequenceState("cold-" + warm.RequestId, warm.PromptTokens.ToList(), 3, 8,
                SamplingConfig.Greedy, mediaSpans: warm.MediaSpans);
            await Run(cold, reference);
            Assert.Equal(reference.OutputTokens, warm.OutputTokens);
        }
    }

    // An image as the first message after a restart (or after the resident checkpoint was
    // evicted) restores the shared prefix from disk like a text chat: the store key is the
    // prefix's text tokens, and the image comes after them. Refusing every media request
    // here meant such a chat prefilled the whole shared prompt again.
    [Fact]
    public async Task ImageChatAfterRestart_RestoresTheSharedPrefixFromDisk()
    {
        const int shared = 16;
        List<int> system = Tokens(shared);
        var store = new MultipleCheckpointStore();
        using (var engine = new InferenceEngine(MediaRecurrent(), Configuration()))
        {
            engine.PrefixCheckpointStore = store;
            await Run(engine, new SequenceState("warmup", system.Concat(new[] { 201 }).ToList(), 1, 8,
                SamplingConfig.Greedy, cacheScope: "warmup", sharedPrefixTokens: shared));
        }

        using var restarted = new InferenceEngine(MediaRecurrent(), Configuration());
        restarted.PrefixCheckpointStore = store;
        var span = new PromptMediaSpan(shared + 2, shared + 10, new string('c', 64));
        List<int> prompt = system.Concat(new[] { 212, 213 }).Concat(Enumerable.Repeat(242, 8)).Concat(new[] { 214 }).ToList();
        var request = ImageTurn("after-restart", prompt, new[] { span }, shared, "fresh");
        Assert.Equal(shared, (await Run(restarted, request)).PrefixCacheReusedTokens);
        Assert.Contains(shared, store.OpenedLengths);
    }

    // With no warm-up -- the first message after a launch whose warm-up was skipped, or a
    // host that never warms -- the image chat itself is the first request to compute the
    // shared prefix. That prefix holds no media, so it is saved like a text chat's; a
    // boundary a span starts before is not. This used to refuse every request with any
    // media, so the next launch prefilled the whole shared prompt again.
    [Theory]
    [InlineData(19, 8, new[] { 16 })]   // the image is in the user turn, behind the shared prompt
    [InlineData(12, 3, new[] { 8 })]    // a span inside the shared prefix: only the ancestor before it
    public async Task ImageChatWithoutAWarmup_SavesTheSharedPrefixBeforeItsImage(int spanStart, int spanLength, int[] saved)
    {
        const int shared = 16;
        var store = new MultipleCheckpointStore();
        using var engine = new InferenceEngine(MediaRecurrent(), Configuration());
        engine.PrefixCheckpointStore = store;
        var span = new PromptMediaSpan(spanStart, spanStart + spanLength, new string('e', 64));
        List<int> prompt = Tokens(spanStart).Concat(Enumerable.Repeat(243, spanLength))
            .Concat(new[] { 215, 216, 217, 218 }).ToList();
        var request = new SequenceState("first", prompt, 3, 8, SamplingConfig.Greedy,
            mediaSpans: new[] { span }, cacheScope: "chat", sharedPrefixTokens: shared,
            publicCheckpointBoundaries: new[] { 8 });

        Assert.Equal(0, (await Run(engine, request)).PrefixCacheReusedTokens);
        foreach (int length in saved)
            Assert.Contains(length, store.SavedLengths);
        Assert.All(store.SavedLengths, length => Assert.True(length < span.Start,
            $"saved a {length}-token prefix at or past the image at {span.Start}"));

        // What was saved is the text prefix the store is keyed by: a text chat after a
        // restart restores it.
        using var restarted = new InferenceEngine(MediaRecurrent(), Configuration());
        restarted.PrefixCheckpointStore = store;
        int longest = saved.Max();
        var text = new SequenceState("text", prompt.Take(longest).Concat(new[] { 219, 220 }).ToList(), 3, 8,
            SamplingConfig.Greedy, cacheScope: "other", sharedPrefixTokens: longest);
        Assert.Equal(longest, (await Run(restarted, text)).PrefixCacheReusedTokens);
    }

    // The gate is per boundary: a checkpoint is looked up only when every media span of the
    // request starts at or after it, because the store key says nothing of an image's pixels.
    [Theory]
    [InlineData(-1, new[] { 16, 8 })]   // no media
    [InlineData(16, new[] { 16, 8 })]   // the span starts exactly at the shared boundary
    [InlineData(12, new[] { 8 })]       // a span inside the shared prefix: only the ancestor
    [InlineData(4, new int[0])]         // a span before every boundary: nothing
    public void CheckpointRestore_IsGatedPerBoundaryByTheFirstMediaSpan(int spanStart, int[] opened)
    {
        using var model = MediaRecurrent();
        var (_, cache) = PendingCheckpointScheduler(model);
        var store = new MultipleCheckpointStore();
        cache!.CheckpointStore = store;
        PromptMediaSpan[]? spans = spanStart < 0
            ? null
            : new[] { new PromptMediaSpan(spanStart, spanStart + 8, new string('d', 64)) };
        var sequence = new SequenceState("gate", Tokens(40), 3, 8, SamplingConfig.Greedy,
            mediaSpans: spans, cacheScope: "gate", sharedPrefixTokens: 16,
            publicCheckpointBoundaries: new[] { 8 });

        Assert.Equal(0, cache.ComputeReusablePrefix(sequence));
        Assert.Equal(opened, store.OpenedLengths);
    }

    [Fact]
    public void DeclineReason_NamesAPromptThatLeftItsOwnSharedPrefix_AndTheRealClamp()
    {
        // Every cached prefix ends inside this prompt's shared prefix: its shared part drifted.
        var drifted = new MatchPlan { Structural = 5_300 };
        Assert.Equal("the prompt leaves every cached prefix at token 5300, inside its 7219-token shared prefix",
            PrefixCacheCoordinator.DescribeDecline(drifted, publicBoundary: 7_219, rewindCapTokens: 16));

        // Gemma 4's rewind past the cap is said as that, and never blamed on a media span --
        // not even one that cut ANOTHER source: each source is described by its own clamps.
        var rewound = new MatchPlan
        {
            Structural = 7_400,
            TruncationDecline = SourceDecline.Clamped,
            TruncationClamps = ClampReasons.RewindCap,
            EndStateDecline = SourceDecline.Clamped,
            EndStateClamps = ClampReasons.Media,
            Clamps = ClampReasons.RewindCap | ClampReasons.Media,
        };
        string reason = PrefixCacheCoordinator.DescribeDecline(rewound, publicBoundary: 7_219, rewindCapTokens: 16)!;
        Assert.Equal("rewinding the cached conversation is cut short by the 16-token rewind cap", reason);
        Assert.DoesNotContain("media", reason, StringComparison.Ordinal);

        var media = new MatchPlan
        {
            Structural = 7_400,
            EndStateDecline = SourceDecline.Clamped,
            EndStateClamps = ClampReasons.Media,
            Clamps = ClampReasons.Media,
        };
        Assert.Equal("the retained state is cut short by a media span",
            PrefixCacheCoordinator.DescribeDecline(media, publicBoundary: 7_219, rewindCapTokens: 16));

        // Every flag has words; only a cut no flag records is "a clamp".
        foreach (ClampReasons flag in Enum.GetValues<ClampReasons>().Where(flag => flag != ClampReasons.None))
            Assert.NotEqual("a clamp", PrefixCacheCoordinator.ClampWords(flag, rewindCapTokens: 16));
        Assert.Equal("the media reuse threshold and the minimum clone length",
            PrefixCacheCoordinator.ClampWords(ClampReasons.MmThreshold | ClampReasons.CloneCost, rewindCapTokens: 16));
        Assert.Equal("a clamp", PrefixCacheCoordinator.ClampWords(ClampReasons.None, rewindCapTokens: 16));

        // Nothing matched at all, or a longer source simply won: nothing to say.
        Assert.Null(PrefixCacheCoordinator.DescribeDecline(new MatchPlan(), publicBoundary: 7_219, rewindCapTokens: 16));
    }

    [Fact]
    public async Task ExplicitNone_DoesNotReadOrPublishPayloads()
    {
        var model = OracleFakes.R(8);
        using var engine = new InferenceEngine(model, Configuration());
        var first = Request("first", Tokens(32), breaks: Array.Empty<int>());
        await Run(engine, first);
        Assert.Empty(model.RetainedPayloadKeys);
        var second = Request("second", Tokens(32), breaks: Array.Empty<int>());
        Assert.Equal(0, (await Run(engine, second)).PrefixCacheReusedTokens);
    }

    [Fact]
    public async Task OtherScope_CannotUsePrivateEndState()
    {
        using var engine = new InferenceEngine(OracleFakes.R(8), Configuration());
        var first = Request("first", Tokens(32), "a");
        await Run(engine, first);
        var next = first.PromptTokens.Concat(first.OutputTokens).Take(first.NumComputedTokens)
            .Concat(new[] { 91, 92 }).ToList();
        Assert.Equal(0, (await Run(engine, Request("other", next, "b"))).PrefixCacheReusedTokens);
    }

    [Fact]
    public async Task PrimaryOnlyFamily_ContinuesItsRadixPrefix()
    {
        var traits = new OracleTraits
        {
            Name = "primary", Class = FamilyClass.S, EndState = EndStateSupport.None,
            Pages = PageSupport.None, Truncation = TruncationKind.Any, PrimaryResident = true,
        };
        using var engine = new InferenceEngine(new OracleModel(traits, 8), Configuration());
        var first = Request("first", Tokens(32));
        await Run(engine, first);
        var prompt = first.PromptTokens.Concat(first.OutputTokens).Take(first.NumComputedTokens)
            .Concat(new[] { 71, 72 }).ToList();
        var second = Request("second", prompt);
        Assert.Equal(first.NumComputedTokens, (await Run(engine, second)).PrefixCacheReusedTokens);
        using var baseline = new InferenceEngine(new OracleModel(traits, 8), Configuration(false));
        var cold = Request("cold", prompt);
        await Run(baseline, cold);
        Assert.Equal(cold.OutputTokens, second.OutputTokens);
    }

    [Fact]
    public async Task Shutdown_ReleasesAllTreeOwnedPayloads()
    {
        var model = OracleFakes.R(8);
        var engine = new InferenceEngine(model, Configuration());
        await Run(engine, Request("seed", Tokens(32), shared: 16));
        engine.Dispose();
        Assert.Empty(model.RetainedPayloadKeys);
    }

    [Fact]
    public async Task CacheOverrides_DisableOwnedPayloadCopies()
    {
        const string privateSetting = "TS_RETAINED_FUSED_CACHE_MAX", publicSetting = "TS_PREFIX_CHECKPOINTS_MAX";
        string? previousPrivate = Environment.GetEnvironmentVariable(privateSetting);
        string? previousPublic = Environment.GetEnvironmentVariable(publicSetting);
        try
        {
            Environment.SetEnvironmentVariable(privateSetting, "0");
            Environment.SetEnvironmentVariable(publicSetting, "0");
            var model = OracleFakes.R(8);
            using var engine = new InferenceEngine(model, Configuration());
            await Run(engine, Request("seed", Tokens(32), shared: 16));
            var other = Request("other", Tokens(32), "different", shared: 16);
            Assert.Equal(0, (await Run(engine, other)).PrefixCacheReusedTokens);
            Assert.Empty(model.RetainedPayloadKeys);
        }
        finally
        {
            Environment.SetEnvironmentVariable(privateSetting, previousPrivate);
            Environment.SetEnvironmentVariable(publicSetting, previousPublic);
        }
    }

    [Fact]
    public async Task MemoryPressure_EvictsOldConversation_AndRecomputesCorrectly()
    {
        var model = OracleFakes.R(8);
        using var engine = new InferenceEngine(model, Configuration());
        var first = Request("first", Tokens(32), "a");
        await Run(engine, first);
        await Run(engine, Request("latest", Tokens(32, 61), "b"));
        engine.TrimIdleMemory();
        var prompt = first.PromptTokens.Concat(first.OutputTokens).Take(first.NumComputedTokens)
            .Concat(new[] { 101, 102 }).ToList();
        var after = Request("after-trim", prompt, "a");
        Assert.Equal(0, (await Run(engine, after)).PrefixCacheReusedTokens);
        using var cold = new InferenceEngine(OracleFakes.R(8), Configuration(false));
        var reference = Request("reference", prompt);
        await Run(cold, reference);
        Assert.Equal(reference.OutputTokens, after.OutputTokens);
    }

    [Fact]
    public void PublicCheckpoint_CannotRemainAboveHardResourceCap()
    {
        using var model = OracleFakes.R(8);
        model.SpareBytes = 1; // The recurrent state alone needs 1,024 bytes.
        var pool = new BlockPool(32, 8, 64);
        var scheduler = new ContinuousBatchScheduler(Configuration(), pool);
        var cache = new PrefixCacheCoordinator(model, pool, scheduler,
            model.GetPrefixCacheCapabilities(), NullLogger.Instance);
        var sequence = Request("prefix", Tokens(16), shared: 16);
        foreach (var block in pool.AllocateNew(2)!) sequence.BlockTable.AppendBlock(block);
        model.Forward(sequence.PromptTokens.ToArray());
        sequence.SetComputedTokensForPrefixAdoption(16);
        cache.CaptureCheckpoint(sequence);
        Assert.Empty(model.RetainedPayloadKeys);
        Assert.Equal(0, cache.Tree.PublicEndStateCount);
        Assert.True(cache.Tree.Cached.StateSnapshot <= cache.Tree.EffectiveCap(ResourceClass.StateSnapshot));
        cache.Reset();
        pool.Free(sequence.BlockTable.Clear());
    }

    [Fact]
    public void CompletedColdConversations_DoNotAccumulateScopeMetadata()
    {
        using var model = OracleFakes.R(8);
        var pool = new BlockPool(32, 8, 64);
        var scheduler = new ContinuousBatchScheduler(Configuration(), pool);
        var cache = new PrefixCacheCoordinator(model, pool, scheduler,
            model.GetPrefixCacheCapabilities(), NullLogger.Instance);
        for (int i = 0; i < 500; i++)
        {
            var request = Request($"request-{i}", Tokens(16), $"conversation-{i}", breaks: Array.Empty<int>());
            cache.EnsureRequest(request);
            cache.ReleaseRequest(request);
        }
        Assert.Equal(0, cache.RequestCount);
        Assert.Equal(1, cache.Tree.Scopes.LiveCount); // Only the public scope remains.
        Assert.Equal(2, cache.Tree.Scopes.Capacity); // The same private slot was reused.
        cache.Reset();
    }

    [Theory]
    [InlineData("TS_SCHED_DISABLE_BATCHED", "1")]
    [InlineData("TS_PER_SEQ_FUSED", "0")]
    public async Task DisablingHolderExecutionRoute_RecomputesWithoutAdoptingUnreadableHolders(string name, string value)
    {
        string? previous = Environment.GetEnvironmentVariable(name);
        try
        {
            Environment.SetEnvironmentVariable(name, value);
            using var engine = new InferenceEngine(OracleFakes.R(8), Configuration());
            var first = Request("first", Tokens(32));
            await Run(engine, first);
            var prompt = Tokens(32).Concat(new[] { 81, 82 }).ToList();
            var after = Request("after", prompt);
            await Run(engine, after);
            using var cold = new InferenceEngine(OracleFakes.R(8), Configuration(false));
            var reference = Request("reference", prompt);
            await Run(cold, reference);
            Assert.Equal(reference.OutputTokens, after.OutputTokens);
        }
        finally { Environment.SetEnvironmentVariable(name, previous); }
    }

    [Fact]
    public async Task ShutdownWithActiveHolders_ReleasesRequestsAndCompletesConsumers()
    {
        var gate = new ComputeGate();
        gate.Close();
        var model = new PausingOracle(gate);
        var engine = new InferenceEngine(model, Configuration()) { ComputeGate = gate };
        var first = engine.SubmitRequest(new SequenceState("a", Tokens(32), 100, 8,
            SamplingConfig.Greedy, cacheScope: "a"));
        var second = engine.SubmitRequest(new SequenceState("b", Tokens(32, 61), 100, 8,
            SamplingConfig.Greedy, cacheScope: "b"));
        gate.Open();
        try
        {
            Assert.True(SpinWait.SpinUntil(() => model.ForwardCalls >= 2 && engine.StepsHeldByGate > 0,
                TimeSpan.FromSeconds(5)), "Both requests should pause with live per-request holders.");
            Assert.True(model.PrivateHolderCount > 0);
        }
        finally { engine.Dispose(); }
        Assert.Equal(0, model.PrivateHolderCount);
        Assert.Empty(model.RetainedPayloadKeys);
        await Assert.ThrowsAsync<ObjectDisposedException>(() => first.Completion);
        await Assert.ThrowsAsync<ObjectDisposedException>(() => second.Completion);
    }

    private sealed class PausingOracle(ComputeGate gate) : OracleModel(new OracleTraits
    {
        Name = "paused", Class = FamilyClass.R, EndState = EndStateSupport.CopyAndDonate,
        CanCaptureCopy = true, AdoptPrimaryOnDisplacement = true, Truncation = TruncationKind.None,
    }, 8), IModelArchitecture
    {
        private int _forwardCalls;
        internal int ForwardCalls => Volatile.Read(ref _forwardCalls);
        float[] IModelArchitecture.Forward(int[] tokens)
        {
            var logits = base.Forward(tokens);
            Interlocked.Increment(ref _forwardCalls);
            gate.Close();
            return logits;
        }
    }

    [Fact]
    public async Task AbortingNonBatchedModel_ReleasesRadixRequestMetadata()
    {
        var inner = new OracleModel(new OracleTraits
        {
            Name = "non-batched", Class = FamilyClass.S, Truncation = TruncationKind.Any,
            EndState = EndStateSupport.None, Pages = PageSupport.None,
        }, 8);
        using var model = new NonBatchedOracle(inner);
        using var engine = new InferenceEngine(model, Configuration());
        model.AfterForward = () => engine.Abort("cancelled");
        var request = new SequenceState("cancelled", Tokens(32), 100, 8,
            SamplingConfig.Greedy, cacheScope: "scope");
        var completion = await Run(engine, request);
        Assert.Equal(SequenceStatus.FinishedAborted, completion.Status);
        Assert.Equal(0, Assert.IsType<PrefixCacheCoordinator>(inner.Sink).RequestCount);
    }

    private sealed class NonBatchedOracle(OracleModel inner) : IModelArchitecture, IPrefixCacheModel
    {
        internal Action? AfterForward;
        public ModelConfig Config => inner.Config;
        public ITokenizer Tokenizer => inner.Tokenizer;
        public IMultimodalInjector MultimodalInjector => inner.MultimodalInjector;
        public IBackendExecutionPlan ExecutionPlan => inner.ExecutionPlan;
        public bool SupportsKVCacheTruncation => true;
        public float[] Forward(int[] tokens)
        {
            float[] logits = inner.Forward(tokens);
            Action? callback = AfterForward;
            AfterForward = null;
            callback?.Invoke();
            return logits;
        }
        public void ResetKVCache() => inner.ResetKVCache();
        public void TruncateKVCache(int tokenCount) => inner.TruncateKVCache(tokenCount);
        public void Dispose() => inner.Dispose();
        public PrefixCacheCapabilities GetPrefixCacheCapabilities() => inner.GetPrefixCacheCapabilities();
        public void AttachPrefixCache(IPrefixPayloadSink sink) => inner.AttachPrefixCache(sink);
        public bool TryCaptureCopy(string id, string key, out PayloadFootprint footprint) => inner.TryCaptureCopy(id, key, out footprint);
        public bool TryCaptureDonate(string id, string key, int length, out PayloadFootprint footprint) => inner.TryCaptureDonate(id, key, length, out footprint);
        public bool TryConvertPrimary(string key, int length, out PayloadFootprint footprint) => inner.TryConvertPrimary(key, length, out footprint);
        public bool TryMaterialize(in MaterializeRequest request) => inner.TryMaterialize(request);
        public bool TryReturnDonation(string id, string key) => inner.TryReturnDonation(id, key);
        public bool CanMaterialize(string key, int count, int target) => inner.CanMaterialize(key, count, target);
        public void ReleasePayloads(ReadOnlySpan<string> keys, ReleaseReason reason) => inner.ReleasePayloads(keys, reason);
        public PayloadFootprint MeasureEndState(string key) => inner.MeasureEndState(key);
        public ResourceVector EstimateCloneBytes(string key, int count) => inner.EstimateCloneBytes(key, count);
        public bool TryCopyPagedToHolder(ReadOnlySpan<int> ids, int tokens, string id) => inner.TryCopyPagedToHolder(ids, tokens, id);
        public bool TryExport(string key, Stream stream) => inner.TryExport(key, stream);
        public bool TryImport(string key, int tokens, Stream stream, out PayloadFootprint footprint) => inner.TryImport(key, tokens, stream, out footprint);
        public bool TryBeginImport(int tokens, out object? ticket) => inner.TryBeginImport(tokens, out ticket);
        public bool RunImportRead(object ticket, Stream stream) => inner.RunImportRead(ticket, stream);
        public bool TryCommitImport(object ticket, string key, out PayloadFootprint footprint) => inner.TryCommitImport(ticket, key, out footprint);
        public void AbortImport(object ticket) => inner.AbortImport(ticket);
        public long QuerySpareBytes(ResourceClass cls) => inner.QuerySpareBytes(cls);
    }

    private sealed class MemoryCheckpointStore : IPrefixCheckpointStore
    {
        internal byte[]? Bytes;
        private int[]? _tokens;
        public bool TryOpen(string modelFingerprint, ReadOnlySpan<int> prefixTokens, out Stream payload)
        {
            if (Bytes is not null && prefixTokens.SequenceEqual(_tokens))
            {
                payload = new MemoryStream(Bytes, writable: false);
                return true;
            }
            payload = null!;
            return false;
        }
        public bool Save(string modelFingerprint, ReadOnlySpan<int> prefixTokens, Action<Stream> writePayload)
        {
            using var stream = new MemoryStream();
            writePayload(stream);
            _tokens = prefixTokens.ToArray();
            Bytes = stream.ToArray();
            return true;
        }
    }

    private sealed class MultipleCheckpointStore : IPrefixCheckpointStore
    {
        private readonly Dictionary<string, byte[]> _payloads = new();
        internal readonly List<int> SavedLengths = new();
        internal readonly List<int> OpenedLengths = new();
        public bool TryOpen(string modelFingerprint, ReadOnlySpan<int> prefixTokens, out Stream payload)
        {
            OpenedLengths.Add(prefixTokens.Length);
            if (_payloads.TryGetValue(modelFingerprint + ":" + string.Join(',', prefixTokens.ToArray()), out byte[]? bytes))
            {
                payload = new MemoryStream(bytes, writable: false);
                return true;
            }
            payload = null!;
            return false;
        }
        public bool Save(string modelFingerprint, ReadOnlySpan<int> prefixTokens, Action<Stream> writePayload)
        {
            using var stream = new MemoryStream();
            writePayload(stream);
            _payloads[modelFingerprint + ":" + string.Join(',', prefixTokens.ToArray())] = stream.ToArray();
            SavedLengths.Add(prefixTokens.Length);
            return true;
        }
    }
}
