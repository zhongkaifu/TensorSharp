// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using InferenceWeb.Tests.PrefixCache.Fakes;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Cli;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

public sealed class CliInferenceSessionTests
{
    private static SchedulerConfig Config => new()
    {
        BlockSize = 16,
        NumBlocks = 64,
        MaxNumRunningSequences = 1,
        MaxNumBatchedTokens = 128,
        SoloPrefillChunkSize = 128,
        StopRepetition = false,
    };

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void WarmPromptAndFollowUpReuseRadixWithColdEquivalentOutput(bool recurrent)
    {
        using var model = recurrent ? OracleFakes.R() : OracleFakes.P();
        using var session = new CliInferenceSession(model, Config, NullLogger.Instance);
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        int[] warm = Enumerable.Range(1, 64).ToArray();
        session.Generate(warm, 1, SamplingConfig.Greedy, cancellationToken: timeout.Token);
        int[] prompt = warm.Concat(new[] { 81, 82, 83 }).ToArray();
        var reused = session.Generate(prompt, 4, SamplingConfig.Greedy, cancellationToken: timeout.Token);

        using var reference = recurrent ? OracleFakes.R() : OracleFakes.P();
        using var cold = new CliInferenceSession(reference, Config, NullLogger.Instance);
        var expected = cold.Generate(prompt, 4, SamplingConfig.Greedy, cancellationToken: timeout.Token,
            enablePrefixCache: false);
        Assert.Equal(64, reused.Completion.PrefixCacheReusedTokens);
        Assert.Equal(expected.Tokens, reused.Tokens);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PublicWarmupSurvivesNewChatsWithoutSharingPrivateTokens(bool recurrent)
    {
        using var model = recurrent ? OracleFakes.R() : OracleFakes.P();
        using var session = new CliInferenceSession(model, Config, NullLogger.Instance);
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        int[] system = Enumerable.Range(1, 13).ToArray();
        session.Generate(system, 1, SamplingConfig.Greedy, cancellationToken: timeout.Token,
            sharedPrefixTokens: system.Length);

        int[] prompt = system.Concat(Enumerable.Range(51, 32)).ToArray();
        session.StartNewConversation();
        var first = session.Generate(prompt, 3, SamplingConfig.Greedy, cancellationToken: timeout.Token,
            sharedPrefixTokens: system.Length);
        Assert.Equal(system.Length, first.Completion.PrefixCacheReusedTokens);

        session.StartNewConversation();
        Assert.Equal(0, session.CachedTokens);
        // Even identical user text in an independent chat cannot adopt the old
        // chat's private prompt or generated tail.
        var next = session.Generate(prompt, 3, SamplingConfig.Greedy, cancellationToken: timeout.Token,
            sharedPrefixTokens: system.Length);
        Assert.NotEqual(first.Sequence.CacheScope, next.Sequence.CacheScope);
        Assert.Equal(system.Length, next.Completion.PrefixCacheReusedTokens);

        using var reference = recurrent ? OracleFakes.R() : OracleFakes.P();
        using var cold = new CliInferenceSession(reference, Config, NullLogger.Instance);
        Assert.Equal(cold.Generate(prompt, 3, SamplingConfig.Greedy, cancellationToken: timeout.Token,
            enablePrefixCache: false).Tokens, next.Tokens);
        Assert.Equal(first.Tokens, next.Tokens);
    }

    [Fact]
    public void IndependentChatsWithoutPublicPrefixAndFreshEnginesStartCold()
    {
        using var model = OracleFakes.P();
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        int[] prompt = Enumerable.Range(1, 64).ToArray();
        using (var session = new CliInferenceSession(model, Config, NullLogger.Instance))
        {
            session.Generate(prompt, 3, SamplingConfig.Greedy, cancellationToken: timeout.Token);
            session.StartNewConversation();
            Assert.Equal(0, session.Generate(prompt, 3, SamplingConfig.Greedy,
                cancellationToken: timeout.Token).Completion.PrefixCacheReusedTokens);
            session.Generate(prompt, 3, SamplingConfig.Greedy, cancellationToken: timeout.Token,
                sharedPrefixTokens: 32);
            session.StartNewConversation();
            Assert.Equal(0, session.Generate(prompt, 3, SamplingConfig.Greedy,
                cancellationToken: timeout.Token, enablePrefixCache: false,
                sharedPrefixTokens: 32).Completion.PrefixCacheReusedTokens);
        }
        // CLI has no disk checkpoint store. A new engine, like a new process,
        // cannot reuse public state from the disposed engine.
        using var fresh = new CliInferenceSession(model, Config, NullLogger.Instance);
        Assert.Equal(0, fresh.Generate(prompt, 3, SamplingConfig.Greedy,
            cancellationToken: timeout.Token, sharedPrefixTokens: 32).Completion.PrefixCacheReusedTokens);
    }

    [Fact]
    public void SharedPrefixProbesExcludeUserContentAndValidateTheActualRender()
    {
        List<int> Render(IReadOnlyList<ChatMessage> messages, bool generationPrompt)
        {
            var tokens = Enumerable.Range(1, 14).ToList();
            if (messages.Count > 1)
                tokens.AddRange(messages[1].Content.Select(c => 1000 + (int)c));
            return tokens;
        }
        var prefix = CliSharedPrefix.Compute("system", false, Render);
        Assert.Equal(Enumerable.Range(1, 13), prefix);
        var actual = Render(new[] { new ChatMessage { Role = "system", Content = "system" },
            new ChatMessage { Role = "user", Content = "private" } }, true);
        Assert.Equal(13, CliSharedPrefix.MatchingLength(prefix, actual));
        actual[0] = 99;
        Assert.Equal(0, CliSharedPrefix.MatchingLength(prefix, actual));
        Assert.Empty(CliSharedPrefix.Compute(null, false,
            (_, _) => throw new InvalidOperationException("no public prompt to render")));
    }

    [Fact]
    public void ResetAndExplicitCacheOptOutStartCold()
    {
        using var model = OracleFakes.P();
        using var session = new CliInferenceSession(model, Config, NullLogger.Instance);
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        int[] prompt = Enumerable.Range(1, 64).ToArray();
        var first = session.Generate(prompt, 3, SamplingConfig.Greedy, cancellationToken: timeout.Token);
        var disabled = session.Generate(prompt, 3, SamplingConfig.Greedy, cancellationToken: timeout.Token,
            enablePrefixCache: false);
        Assert.Equal(0, disabled.Completion.PrefixCacheReusedTokens);
        Assert.Equal(first.Tokens, disabled.Tokens);

        session.Reset();
        Assert.Equal(0, session.CachedTokens);
        var reset = session.Generate(prompt, 3, SamplingConfig.Greedy, cancellationToken: timeout.Token);
        Assert.Equal(0, reset.Completion.PrefixCacheReusedTokens);
        Assert.Equal(first.Tokens, reset.Tokens);
    }

    [Fact]
    public void ZeroOutputBudgetPrefillsWithoutEmittingTokens()
    {
        using var model = OracleFakes.P();
        using var session = new CliInferenceSession(model, Config, NullLogger.Instance);
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        int[] prompt = Enumerable.Range(1, 64).ToArray();
        var result = session.Generate(prompt, 0, SamplingConfig.Greedy,
            _ => throw new InvalidOperationException("zero-budget request emitted a token"), timeout.Token);
        Assert.Empty(result.Tokens);
        Assert.Equal(0, result.Completion.OutputTokenCount);
        Assert.Equal(result.Sequence.NumComputedTokens, session.CachedTokens);
        Assert.True(session.CachedTokens >= prompt.Length);
    }

    [Fact]
    public void CancellationStopsDeliveredTokensAndAllowsTheNextTurn()
    {
        using var model = OracleFakes.P();
        using var session = new CliInferenceSession(model, Config, NullLogger.Instance);
        using var cancellation = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        int[] prompt = Enumerable.Range(1, 64).ToArray();
        var cancelled = session.Generate(prompt, 8, SamplingConfig.Greedy, _ =>
        {
            cancellation.Cancel();
            return true;
        }, cancellation.Token);
        Assert.Single(cancelled.Tokens);

        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        var next = session.Generate(prompt, 3, SamplingConfig.Greedy, cancellationToken: timeout.Token);
        using var reference = OracleFakes.P();
        using var cold = new CliInferenceSession(reference, Config, NullLogger.Instance);
        Assert.Equal(cold.Generate(prompt, 3, SamplingConfig.Greedy, cancellationToken: timeout.Token).Tokens, next.Tokens);
    }

    [Fact]
    public void StreamStopAndCallbackFailureLeaveSessionUsable()
    {
        using var model = OracleFakes.P();
        using var session = new CliInferenceSession(model, Config, NullLogger.Instance);
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        int[] prompt = Enumerable.Range(1, 64).ToArray();
        var stopped = session.Generate(prompt, 8, SamplingConfig.Greedy, _ => false, timeout.Token);
        Assert.Single(stopped.Tokens);

        var failure = new InvalidOperationException("console stream failed");
        var caught = Assert.Throws<InvalidOperationException>(() => session.Generate(
            prompt, 8, SamplingConfig.Greedy, _ => throw failure, timeout.Token));
        Assert.Same(failure, caught);

        var recovered = session.Generate(prompt, 3, SamplingConfig.Greedy, cancellationToken: timeout.Token);
        using var reference = OracleFakes.P();
        using var cold = new CliInferenceSession(reference, Config, NullLogger.Instance);
        Assert.Equal(cold.Generate(prompt, 3, SamplingConfig.Greedy, cancellationToken: timeout.Token).Tokens,
            recovered.Tokens);
    }

    // ---- scoped generation (sub-agents) ------------------------------------------

    private static SchedulerConfig ConcurrentConfig => new()
    {
        BlockSize = 16,
        NumBlocks = 128,
        MaxNumRunningSequences = 4,
        MaxNumBatchedTokens = 128,
        SoloPrefillChunkSize = 128,
        StopRepetition = false,
    };

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ScopedGenerationsRunTogetherAndShareOnlyThePublicPrefix(bool recurrent)
    {
        using var model = recurrent ? OracleFakes.R() : OracleFakes.P();
        using var session = new CliInferenceSession(model, ConcurrentConfig, NullLogger.Instance);
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        int[] system = Enumerable.Range(1, 29).ToArray();
        session.Generate(system, 1, SamplingConfig.Greedy, cancellationToken: timeout.Token,
            sharedPrefixTokens: system.Length);
        string consoleScope = session.CacheScope;
        int consoleCached = session.CachedTokens;

        // Three sub-agents at once, each in a scope of its own, starting with the same
        // public prefix and diverging after it.
        int[][] prompts = Enumerable.Range(0, 3)
            .Select(i => system.Concat(Enumerable.Range(100 + 10 * i, 9)).ToArray())
            .ToArray();
        CliInferenceSession.Result[] results = await Task.WhenAll(prompts.Select(prompt =>
            session.GenerateInScopeAsync(prompt, 4, SamplingConfig.Greedy, CliInferenceSession.NewAgentScope(),
                cancellationToken: timeout.Token, sharedPrefixTokens: system.Length)));

        using var reference = recurrent ? OracleFakes.R() : OracleFakes.P();
        using var cold = new CliInferenceSession(reference, ConcurrentConfig, NullLogger.Instance);
        for (int i = 0; i < prompts.Length; i++)
        {
            Assert.Equal(system.Length, results[i].Completion.PrefixCacheReusedTokens);
            Assert.Equal(cold.Generate(prompts[i], 4, SamplingConfig.Greedy, cancellationToken: timeout.Token,
                enablePrefixCache: false).Tokens, results[i].Tokens);
        }
        Assert.Equal(3, results.Select(r => r.Sequence.CacheScope).Distinct().Count());
        Assert.DoesNotContain(consoleScope, results.Select(r => r.Sequence.CacheScope));

        // The console's own bookkeeping is untouched by the agents' requests.
        Assert.Equal(consoleScope, session.CacheScope);
        Assert.Equal(consoleCached, session.CachedTokens);
    }

    [Fact]
    public async Task ACancelledScopedGenerationThrowsAndLeavesTheEngineUsable()
    {
        using var model = OracleFakes.P();
        using var session = new CliInferenceSession(model, ConcurrentConfig, NullLogger.Instance);
        using var cancellation = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        int[] prompt = Enumerable.Range(1, 64).ToArray();

        await Assert.ThrowsAnyAsync<OperationCanceledException>(() => session.GenerateInScopeAsync(
            prompt, 64, SamplingConfig.Greedy, CliInferenceSession.NewAgentScope(), _ =>
            {
                cancellation.Cancel();
                return true;
            }, cancellation.Token));

        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        var next = session.Generate(prompt, 3, SamplingConfig.Greedy, cancellationToken: timeout.Token);
        using var reference = OracleFakes.P();
        using var cold = new CliInferenceSession(reference, ConcurrentConfig, NullLogger.Instance);
        Assert.Equal(cold.Generate(prompt, 3, SamplingConfig.Greedy, cancellationToken: timeout.Token).Tokens, next.Tokens);
    }

    [Fact]
    public async Task TheFirstConcurrentCallersShareOneEngine()
    {
        // Sub-agents and the parent race to create the engine on their first rounds. Two
        // engines over one model would each drive its state as if it owned it; one engine
        // proves itself by serving the public prefix the first caller left to the others.
        using var model = OracleFakes.P();
        using var session = new CliInferenceSession(model, ConcurrentConfig, NullLogger.Instance);
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        int[] prompt = Enumerable.Range(1, 40).ToArray();

        CliInferenceSession.Result[] first = await Task.WhenAll(Enumerable.Range(0, 4).Select(_ => Task.Run(() =>
            session.GenerateInScopeAsync(prompt, 2, SamplingConfig.Greedy, CliInferenceSession.NewAgentScope(),
                cancellationToken: timeout.Token, sharedPrefixTokens: prompt.Length))));
        Assert.Equal(4, first.Length);

        var after = await session.GenerateInScopeAsync(prompt.Concat(new[] { 90, 91 }).ToArray(), 2,
            SamplingConfig.Greedy, CliInferenceSession.NewAgentScope(), cancellationToken: timeout.Token,
            sharedPrefixTokens: prompt.Length);
        Assert.Equal(prompt.Length, after.Completion.PrefixCacheReusedTokens);
    }
}
