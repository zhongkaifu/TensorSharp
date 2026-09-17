// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Linq;
using System.Threading.Tasks;
using InferenceWeb.Tests.PrefixCache.Fakes;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests.PrefixCache;

public class RadixPagedEngineTests : IDisposable
{
    private const int BlockSize = 8;
    private readonly string? _savedFastPath = Environment.GetEnvironmentVariable("TS_BATCHED_N1_FAST_PATH");

    public RadixPagedEngineTests()
    {
        // Exercise real A2 page production, rather than the oracle's optional
        // primary-cache fast path (primary reuse has separate coverage).
        Environment.SetEnvironmentVariable("TS_BATCHED_N1_FAST_PATH", "0");
    }

    public void Dispose() => Environment.SetEnvironmentVariable("TS_BATCHED_N1_FAST_PATH", _savedFastPath);

    private static OracleModel Model(PageSupport pages) => new(new OracleTraits
    {
        Name = "engine-pages-" + pages,
        Class = FamilyClass.P,
        Pages = pages,
        PrimaryResident = false,
        Truncation = TruncationKind.Any,
    }, BlockSize);

    private static SchedulerConfig Config(int blocks = 16, bool enabled = true) => new()
    {
        BlockSize = BlockSize,
        NumBlocks = blocks,
        MaxNumBatchedTokens = 32,
        SoloPrefillChunkSize = 32,
        MaxPrefillChunkSize = 16,
        EnablePrefixCaching = enabled,
        StopRepetition = false,
    };

    private static async Task<SequenceState> Run(InferenceEngine engine, string id, int[] prompt,
        string? scope = "chat", int publicTokens = 0, int[]? breakpoints = null,
        PromptMediaSpan[]? media = null)
    {
        var seq = new SequenceState(id, prompt, 3, BlockSize, SamplingConfig.Greedy,
            cacheScope: scope, sharedPrefixTokens: publicTokens, cacheBreakpoints: breakpoints, mediaSpans: media);
        await engine.SubmitRequest(seq).Completion.WaitAsync(TimeSpan.FromSeconds(10));
        Assert.Equal(SequenceStatus.FinishedLengthCapped, seq.Status);
        return seq;
    }

    [Theory]
    [InlineData(PageSupport.A1HostSlab)]
    [InlineData(PageSupport.A2ModelPaged)]
    public async Task WarmPagesReusePrefixAndMatchColdOutput(PageSupport pages)
    {
        using var model = Model(pages);
        using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);
        int[] prompt = Enumerable.Range(1, 25).ToArray();
        var cold = await Run(engine, "cold", prompt);
        var warm = await Run(engine, "warm", prompt);
        Assert.Equal(0, cold.PrefixCacheReusedTokens);
        Assert.Equal(24, warm.PrefixCacheReusedTokens);
        Assert.Equal(cold.OutputTokens, warm.OutputTokens);
        Assert.Equal(0, engine.PoolStats.hashedBlocks);

        int[] branch = prompt.ToArray();
        branch[17] = 100;
        var reusedBranch = await Run(engine, "branch", branch);
        using var freshModel = Model(pages);
        using var fresh = new InferenceEngine(freshModel, Config(enabled: false), NullLogger.Instance);
        var coldBranch = await Run(fresh, "branch-cold", branch);
        Assert.Equal(16, reusedBranch.PrefixCacheReusedTokens);
        Assert.Equal(coldBranch.OutputTokens, reusedBranch.OutputTokens);
    }

    [Theory]
    [InlineData(PageSupport.A1HostSlab)]
    [InlineData(PageSupport.A2ModelPaged)]
    public async Task AnotherScopeOnlyReusesThePublicPrefix(PageSupport pages)
    {
        using var model = Model(pages);
        using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);
        int[] prompt = Enumerable.Range(1, 25).ToArray();
        var first = await Run(engine, "first", prompt, "a", publicTokens: 8);
        var other = await Run(engine, "other", prompt, "b", publicTokens: 8);
        Assert.Equal(8, other.PrefixCacheReusedTokens);
        Assert.Equal(first.OutputTokens, other.OutputTokens);
        var own = await Run(engine, "own", prompt, "a", publicTokens: 8);
        Assert.Equal(24, own.PrefixCacheReusedTokens);
    }

    [Theory]
    [InlineData(PageSupport.A1HostSlab)]
    [InlineData(PageSupport.A2ModelPaged)]
    public async Task ExplicitPolicyCapsReuseAndEmptyPolicyDisablesIt(PageSupport pages)
    {
        using var model = Model(pages);
        using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);
        int[] prompt = Enumerable.Range(1, 25).ToArray();
        var first = await Run(engine, "first", prompt);
        var capped = await Run(engine, "capped", prompt, breakpoints: new[] { 8 });
        Assert.Equal(8, capped.PrefixCacheReusedTokens);
        var disabled = await Run(engine, "disabled", prompt, breakpoints: Array.Empty<int>());
        Assert.Equal(0, disabled.PrefixCacheReusedTokens);
        Assert.Equal(first.OutputTokens, disabled.OutputTokens);
    }

    [Theory]
    [InlineData(PageSupport.A1HostSlab)]
    [InlineData(PageSupport.A2ModelPaged)]
    public async Task CacheEvictionMakesRoomForUnrelatedPrompts(PageSupport pages)
    {
        using var model = Model(pages);
        using var engine = new InferenceEngine(model, Config(blocks: 4), NullLogger.Instance);
        using var referenceModel = Model(pages);
        using var reference = new InferenceEngine(referenceModel, Config(blocks: 4, enabled: false), NullLogger.Instance);
        for (int i = 0; i < 6; i++)
        {
            int[] prompt = Enumerable.Range(1 + i * 25, 25).ToArray();
            var cached = await Run(engine, "cached-" + i, prompt);
            var cold = await Run(reference, "cold-" + i, prompt);
            Assert.Equal(cold.OutputTokens, cached.OutputTokens);
        }
    }

    [Theory]
    [InlineData(PageSupport.A1HostSlab)]
    [InlineData(PageSupport.A2ModelPaged)]
    public async Task ConcurrentBranchesMatchIndependentColdRuns(PageSupport pages)
    {
        using var model = Model(pages);
        using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);
        int[] prefix = Enumerable.Range(1, 25).ToArray();
        await Run(engine, "seed", prefix);
        var gate = new ComputeGate();
        gate.Close();
        engine.ComputeGate = gate;
        var branches = Enumerable.Range(0, 3).Select(i => new SequenceState("branch-" + i,
            prefix.Take(24).Concat(new[] { 100 + i }).ToArray(), 3, BlockSize, SamplingConfig.Greedy,
            cacheScope: "chat")).ToArray();
        var handles = branches.Select(s => engine.SubmitRequest(s)).ToArray();
        gate.Open();
        await Task.WhenAll(handles.Select(h => h.Completion)).WaitAsync(TimeSpan.FromSeconds(10));
        using var referenceModel = Model(pages);
        using var reference = new InferenceEngine(referenceModel, Config(enabled: false), NullLogger.Instance);
        foreach (var branch in branches)
        {
            var cold = await Run(reference, "cold-" + branch.RequestId, branch.PromptTokens.ToArray());
            Assert.Equal(24, branch.PrefixCacheReusedTokens);
            Assert.Equal(cold.OutputTokens, branch.OutputTokens);
        }
    }

    [Fact]
    public async Task ChangedMediaStopsReuseBeforeTheMediaSpan()
    {
        using var model = Model(PageSupport.A1HostSlab);
        using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);
        int[] prompt = Enumerable.Range(1, 25).ToArray();
        var original = new[] { new PromptMediaSpan(8, 16, "original-image") };
        await Run(engine, "seed", prompt, media: original);
        Assert.Equal(24, (await Run(engine, "same", prompt, media: original)).PrefixCacheReusedTokens);
        var changed = new[] { new PromptMediaSpan(8, 16, "changed-image") };
        Assert.Equal(8, (await Run(engine, "changed", prompt, media: changed)).PrefixCacheReusedTokens);
    }

    [Fact]
    public async Task UnscopedRequestsShareOnlyTheirPublicPrefix()
    {
        using var model = Model(PageSupport.A1HostSlab);
        using var engine = new InferenceEngine(model, Config(), NullLogger.Instance);
        int[] prompt = Enumerable.Range(1, 25).ToArray();
        await Run(engine, "seed", prompt, scope: null, publicTokens: 8);
        Assert.Equal(8, (await Run(engine, "next", prompt, scope: null, publicTokens: 8)).PrefixCacheReusedTokens);
    }
}
