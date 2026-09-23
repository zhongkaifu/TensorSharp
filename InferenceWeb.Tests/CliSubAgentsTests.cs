// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using InferenceWeb.Tests.PrefixCache.Fakes;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.AgentHost.Agents;
using TensorSharp.Cli;
using TensorSharp.Runtime.Scheduling;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// Sub-agents in the CLI: the flags, the tool declarations, and the child generator —
/// above all that a sub-agent's first prompt continues the public prefix its parent's
/// rounds put in the engine, which is the whole performance case for rendering it with
/// the parent's own renderer.
/// </summary>
public sealed class CliSubAgentsTests
{
    private static SchedulerConfig Config => new()
    {
        BlockSize = 16,
        NumBlocks = 512,
        MaxNumRunningSequences = 4,
        MaxNumBatchedTokens = 256,
        SoloPrefillChunkSize = 128,
        StopRepetition = false,
    };

    private const string SystemPrompt =
        "You are a careful assistant. Use your tools to do the work, and answer briefly.";

    // ---- flags ----------------------------------------------------------------

    private static EnvScope CleanEnvironment()
    {
        var env = new EnvScope();
        env.Set(SubAgentOptions.EnableEnvVar, null);
        env.Set(SubAgentOptions.MaxThreadsEnvVar, null);
        env.Set(SubAgentOptions.MaxDepthEnvVar, null);
        return env;
    }

    [Fact]
    public void Parse_TakesTheFlagsAndTheirValuesOutOfTheArgumentList()
    {
        using EnvScope env = CleanEnvironment();
        string[] args =
        {
            "--model", "m.gguf", "--sub-agents", "--sub-agents-max-threads", "3",
            "--sub-agents-max-depth=2", "--max-tokens", "7",
        };

        SubAgentOptions options = CliSubAgents.Parse(args, out string[] remaining);

        Assert.True(options.Enabled);
        Assert.Equal(3, options.MaxThreads);
        Assert.Equal(2, options.MaxDepth);
        // The "3" is gone with its flag: the CLI's switch never sees a bare value it could
        // mistake for something else.
        Assert.Equal(new[] { "--model", "m.gguf", "--max-tokens", "7" }, remaining);
    }

    [Fact]
    public void Parse_WithNoSubAgentFlags_LeavesTheArgumentsAndTheFeatureAlone()
    {
        using EnvScope env = CleanEnvironment();
        string[] args = { "--model", "m.gguf", "--code-exec", "--input", "p.txt" };

        SubAgentOptions options = CliSubAgents.Parse(args, out string[] remaining);

        Assert.False(options.Enabled);
        Assert.Equal(args, remaining);
        Assert.Null(options.ApplyEngineDefaults());
    }

    [Theory]
    [InlineData("--sub-agents-max-threads", "0")]
    [InlineData("--sub-agents-max-threads", "17")]
    [InlineData("--sub-agents-max-depth", "five")]
    public void Parse_AnOutOfRangeValue_IsAConfigurationError(string flag, string value)
    {
        using EnvScope env = CleanEnvironment();
        Assert.Throws<ArgumentException>(() => CliSubAgents.Parse(new[] { "--sub-agents", flag, value }, out _));
    }

    [Fact]
    public void Parse_TheEnvironmentTurnsItOn()
    {
        using EnvScope env = CleanEnvironment();
        env.Set(SubAgentOptions.EnableEnvVar, "1");
        env.Set(SubAgentOptions.MaxThreadsEnvVar, "2");

        SubAgentOptions options = CliSubAgents.Parse(new[] { "--model", "m.gguf" }, out string[] remaining);

        Assert.True(options.Enabled);
        Assert.Equal(2, options.MaxThreads);
        Assert.Equal(new[] { "--model", "m.gguf" }, remaining);
    }

    // ---- declarations -----------------------------------------------------------

    [Fact]
    public void AppendAgentTools_WhenOff_ReturnsTheListUntouched()
    {
        var tools = new List<ToolFunction> { Tool("shell") };
        Assert.Same(tools, CliSubAgents.AppendAgentTools(tools, new SubAgentOptions(), null));
        Assert.Null(CliSubAgents.AppendAgentTools(null, new SubAgentOptions(), null));
        Assert.Single(tools);
    }

    [Fact]
    public void AppendAgentTools_WhenOn_DeclaresThemAfterEveryOtherTool()
    {
        var tools = new List<ToolFunction> { Tool("skills_read"), Tool("shell"), Tool("apply_patch") };

        List<ToolFunction> merged = CliSubAgents.AppendAgentTools(tools, new SubAgentOptions { Enabled = true }, null);

        // The existing block is a prefix, untouched, so turning the flag on only ever
        // grows the tool block at its end.
        Assert.Equal(tools.Select(t => t.Name), merged.Take(tools.Count).Select(t => t.Name));
        Assert.Equal(SubAgentTools.Declare().Select(t => t.Name), merged.Skip(tools.Count).Select(t => t.Name));
        Assert.Equal(3, tools.Count);
    }

    [Fact]
    public void AppendAgentTools_AnOperatorToolOfTheSameName_KeepsItsDefinition()
    {
        ToolFunction theirs = Tool("Wait_Agent");
        var tools = new List<ToolFunction> { theirs };

        List<ToolFunction> merged = CliSubAgents.AppendAgentTools(tools, new SubAgentOptions { Enabled = true }, null);

        Assert.Same(theirs, merged[0]);
        Assert.Single(merged, t => string.Equals(t.Name, SkillToolNames.WaitAgent, StringComparison.OrdinalIgnoreCase));
        Assert.Equal(SubAgentTools.Declare().Count, merged.Count);
    }

    [Fact]
    public async Task Memoize_AgentsStartingTogether_ComputeThePrefixOnce()
    {
        int computed = 0;
        var tools = new List<ToolFunction> { Tool("shell"), Tool("spawn_agent") };
        Func<List<ChatMessage>, List<ToolFunction>, IReadOnlyList<int>> prefix = CliSubAgents.Memoize((_, _) =>
        {
            Interlocked.Increment(ref computed);
            Thread.Sleep(50);
            return new List<int> { 1, 2, 3 };
        });
        List<ChatMessage> Messages(string task) => new()
        {
            new() { Role = "system", Content = SystemPrompt },
            new() { Role = "user", Content = task },
        };

        // Each agent's loop hands the generator its own copy of the parent's tool list.
        IReadOnlyList<int>[] results = await Task.WhenAll(Enumerable.Range(0, 4).Select(i =>
            Task.Run(() => prefix(Messages("task " + i), new List<ToolFunction>(tools)))));

        Assert.Equal(1, computed);
        Assert.All(results, r => Assert.Equal(new[] { 1, 2, 3 }, r));

        // A different system prompt is a different prefix.
        prefix(new List<ChatMessage> { new() { Role = "system", Content = "other" } }, tools);
        Assert.Equal(2, computed);
    }

    // ---- the child generator ----------------------------------------------------

    [Fact]
    public async Task AFreshSubAgentsFirstPrompt_ContinuesTheParentsPublicPrefix()
    {
        using var model = OracleFakes.P();
        using var inference = new CliInferenceSession(model, Config, NullLogger.Instance);
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        var renderer = new CliRoundRenderer(new TagRenderer(), new SmallTokenizer(), "tmpl",
            model.Config.Architecture, enableThinking: false);
        List<ToolFunction> tools = CliSubAgents.AppendAgentTools(
            new List<ToolFunction> { Tool("shell") }, new SubAgentOptions { Enabled = true }, null);

        // The parent's round, exactly as the interactive session runs it: the public
        // prefix computed from its system prompt and tools, declared to the engine.
        List<int> shared = SharedPrefix(renderer, tools);
        Assert.True(shared.Count > 32, $"the fixture's public prefix is only {shared.Count} tokens");
        var parent = new List<ChatMessage>
        {
            new() { Role = "system", Content = SystemPrompt },
            new() { Role = "user", Content = "Use a sub-agent to add 2 and 3." },
        };
        List<int> parentPrompt = renderer.Render(parent, tools, out _);
        inference.Generate(parentPrompt, 2, SamplingConfig.Greedy, cancellationToken: timeout.Token,
            sharedPrefixTokens: CliSharedPrefix.MatchingLength(shared, parentPrompt));

        var rounds = new ConcurrentQueue<CliSubAgentRound>();
        var generation = new CliSubAgentGeneration(model, renderer, inference, maxTokens: 3,
            SamplingConfig.Greedy, enablePrefixCache: true, (_, _) => shared, NullLogger.Instance)
        {
            OnRound = rounds.Enqueue,
        };
        using var runtime = Runtime(generation, timeout.Token);
        runtime.Root.Bind(parent, tools);

        SkillToolResult spawned = await runtime.Root.ExecuteAsync(
            Call(SkillToolNames.SpawnAgent, ("message", "Add 2 and 3.")), null, timeout.Token);
        Assert.True(spawned.Ok, spawned.Content);
        SkillToolResult waited = await runtime.Root.ExecuteAsync(
            Call(SkillToolNames.WaitAgent), null, timeout.Token);
        Assert.True(waited.Ok, waited.Content);
        Assert.Contains("agent_1 completed", waited.Content);

        CliSubAgentRound round = Assert.Single(rounds);
        Assert.Equal("agent_1", round.AgentId);
        Assert.False(round.Fork);
        // Its own scope: a fresh agent shares only the public prefix with its parent.
        Assert.NotEqual(inference.CacheScope, round.CacheScope);
        // Prefix identity: the agent's prompt starts with the parent's system message and
        // tool block, token for token...
        Assert.Equal(shared.Count, CliSharedPrefix.MatchingLength(shared, round.Prompt));
        Assert.Equal(shared.Count, round.SharedPrefixTokens);
        Assert.True(CommonPrefix(parentPrompt, round.Prompt) >= shared.Count);
        // ...so the engine continues the checkpoint the parent's round left, rather than
        // prefilling the instructions again.
        Assert.Equal(shared.Count, round.ReusedTokens);
    }

    [Fact]
    public async Task AForkedSubAgent_RunsInTheParentsCacheScope()
    {
        using var model = OracleFakes.P();
        using var inference = new CliInferenceSession(model, Config, NullLogger.Instance);
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        var renderer = new CliRoundRenderer(new TagRenderer(), new SmallTokenizer(), "tmpl",
            model.Config.Architecture, enableThinking: false);
        List<ToolFunction> tools = CliSubAgents.AppendAgentTools(
            new List<ToolFunction> { Tool("shell") }, new SubAgentOptions { Enabled = true }, null);
        List<int> shared = SharedPrefix(renderer, tools);

        var conversation = new List<ChatMessage>
        {
            new() { Role = "system", Content = SystemPrompt },
            new() { Role = "user", Content = "Fork a sub-agent to check the result." },
        };
        List<int> parentPrompt = renderer.Render(conversation, tools, out string whitespace);
        CliInferenceSession.Result parentRound = inference.Generate(parentPrompt, 2, SamplingConfig.Greedy,
            cancellationToken: timeout.Token, sharedPrefixTokens: CliSharedPrefix.MatchingLength(shared, parentPrompt));
        ToolCall spawn = Call(SkillToolNames.SpawnAgent, ("message", "Check the result."), ("fork_context", true));
        conversation.Add(new ChatMessage
        {
            Role = "assistant",
            Content = string.Empty,
            ToolCalls = new List<ToolCall> { spawn },
            RawOutputTokens = parentRound.Tokens,
            RawPromptTrailingWhitespace = whitespace,
        });

        var rounds = new ConcurrentQueue<CliSubAgentRound>();
        var generation = new CliSubAgentGeneration(model, renderer, inference, maxTokens: 3,
            SamplingConfig.Greedy, enablePrefixCache: true, (_, _) => shared, NullLogger.Instance)
        {
            OnRound = rounds.Enqueue,
        };
        using var runtime = Runtime(generation, timeout.Token);
        runtime.Root.Bind(conversation, tools);

        Assert.True((await runtime.Root.ExecuteAsync(spawn, null, timeout.Token)).Ok);
        Assert.True((await runtime.Root.ExecuteAsync(Call(SkillToolNames.WaitAgent), null, timeout.Token)).Ok);

        CliSubAgentRound round = Assert.Single(rounds);
        Assert.True(round.Fork);
        Assert.Equal(inference.CacheScope, round.CacheScope);
        Assert.Equal(shared.Count, CliSharedPrefix.MatchingLength(shared, round.Prompt));
        // Past the public boundary: in the parent's scope the fork continues the parent's
        // private conversation too, which is the only reason to fork at all.
        Assert.True(round.ReusedTokens > shared.Count,
            $"a forked agent reused {round.ReusedTokens} tokens, no more than the public prefix ({shared.Count})");
    }

    [Fact]
    public async Task SeveralSubAgents_GenerateConcurrentlyThroughOneEngine()
    {
        using var model = OracleFakes.P();
        using var inference = new CliInferenceSession(model, Config, NullLogger.Instance);
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        var renderer = new CliRoundRenderer(new TagRenderer(), new SmallTokenizer(), "tmpl",
            model.Config.Architecture, enableThinking: false);
        List<ToolFunction> tools = CliSubAgents.AppendAgentTools(
            new List<ToolFunction> { Tool("shell") }, new SubAgentOptions { Enabled = true }, null);
        List<int> shared = SharedPrefix(renderer, tools);
        var parent = new List<ChatMessage>
        {
            new() { Role = "system", Content = SystemPrompt },
            new() { Role = "user", Content = "Three tasks, one sub-agent each." },
        };
        List<int> parentPrompt = renderer.Render(parent, tools, out _);
        inference.Generate(parentPrompt, 2, SamplingConfig.Greedy, cancellationToken: timeout.Token,
            sharedPrefixTokens: CliSharedPrefix.MatchingLength(shared, parentPrompt));

        var rounds = new ConcurrentQueue<CliSubAgentRound>();
        var generation = new CliSubAgentGeneration(model, renderer, inference, maxTokens: 8,
            SamplingConfig.Greedy, enablePrefixCache: true, (_, _) => shared, NullLogger.Instance)
        {
            OnRound = rounds.Enqueue,
        };
        using var runtime = Runtime(generation, timeout.Token);
        runtime.Root.Bind(parent, tools);

        foreach (string task in new[] { "Sum 1..1000.", "Count primes below 10000.", "Fibonacci 50." })
        {
            Assert.True((await runtime.Root.ExecuteAsync(
                Call(SkillToolNames.SpawnAgent, ("message", task)), null, timeout.Token)).Ok);
        }
        var answers = new StringBuilder();
        while (runtime.Root.HasOutstandingWork)
        {
            SkillToolResult waited = await runtime.Root.ExecuteAsync(Call(SkillToolNames.WaitAgent), null, timeout.Token);
            Assert.True(waited.Ok, waited.Content);
            answers.Append(waited.Content);
        }

        Assert.Equal(3, rounds.Count);
        Assert.Equal(new[] { "agent_1", "agent_2", "agent_3" }, rounds.Select(r => r.AgentId).OrderBy(id => id));
        Assert.Equal(3, rounds.Select(r => r.CacheScope).Distinct().Count());
        Assert.All(rounds, r => Assert.Equal(shared.Count, r.ReusedTokens));
        foreach (string id in new[] { "agent_1", "agent_2", "agent_3" })
            Assert.Contains(id + " completed", answers.ToString());
    }

    [Fact]
    public async Task ASubAgentRoundIsParsedLikeTheParents()
    {
        // The generator hands the loop what the parent's rounds hand theirs: the parse,
        // the raw tokens to splice back and the prompt's trailing whitespace.
        using var model = OracleFakes.P();
        using var inference = new CliInferenceSession(model, Config, NullLogger.Instance);
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(30));
        var renderer = new CliRoundRenderer(new TagRenderer(), new SmallTokenizer(), "tmpl",
            model.Config.Architecture, enableThinking: false);
        var generation = new CliSubAgentGeneration(model, renderer, inference, maxTokens: 4,
            SamplingConfig.Greedy, enablePrefixCache: true, null, NullLogger.Instance);
        SkillTurnGenerator generate = generation.CreateGenerator(new SubAgentLaunch { AgentId = "agent_9", Depth = 1 });

        var messages = new List<ChatMessage>
        {
            new() { Role = "system", Content = SystemPrompt },
            new() { Role = "user", Content = "Say something." },
        };
        SkillTurnOutput output = await generate(messages, new List<ToolFunction> { Tool("shell") }, timeout.Token);

        Assert.NotNull(output.RawTokens);
        Assert.Equal(4, output.RawTokens!.Count);
        Assert.Equal(renderer.Tokenizer.Decode(output.RawTokens.ToList()), output.Parsed.Content);
        Assert.NotNull(output.RawPromptTrailingWhitespace);
        // The session's own bookkeeping belongs to the console's requests, not the agents'.
        Assert.Equal(0, inference.CachedTokens);
    }

    // ---- fixtures ---------------------------------------------------------------

    private static SubAgentRuntime Runtime(CliSubAgentGeneration generation, CancellationToken turn) => new(
        new SubAgentOptions { Enabled = true },
        new SubAgentHostBinding { CreateGenerator = generation.CreateGenerator },
        new SkillToolContext(Array.Empty<Skill>()),
        null,
        turn);

    /// <summary>The public prefix the way the interactive session computes it.</summary>
    private static List<int> SharedPrefix(CliRoundRenderer renderer, List<ToolFunction> tools)
    {
        var kv = new KVCachePromptRenderer(new TagRenderer());
        return CliSharedPrefix.Compute(SystemPrompt, tools.Count > 0, (messages, generationPrompt) =>
            kv.RenderToTokens(renderer.Tokenizer, "tmpl", new List<ChatMessage>(messages),
                renderer.Architecture, generationPrompt, out _, out _, tools: tools, enableThinking: false));
    }

    private static int CommonPrefix(IReadOnlyList<int> a, IReadOnlyList<int> b)
    {
        int n = Math.Min(a.Count, b.Count);
        int i = 0;
        while (i < n && a[i] == b[i])
            i++;
        return i;
    }

    private static ToolFunction Tool(string name) => new()
    {
        Name = name,
        Description = "The " + name + " tool.",
        Parameters = new Dictionary<string, ToolParameter>(),
    };

    private static ToolCall Call(string name, params (string Key, object Value)[] arguments)
    {
        var call = new ToolCall { Id = "call_" + Guid.NewGuid().ToString("N").Substring(0, 8), Name = name };
        foreach ((string key, object value) in arguments)
            call.Arguments[key] = value;
        return call;
    }

    /// <summary>A template in miniature: tool names, then one tagged segment per message.</summary>
    private sealed class TagRenderer : IPromptRenderer
    {
        public string Render(string template, List<ChatMessage> messages, bool addGenerationPrompt = true,
            string architecture = null, List<ToolFunction> tools = null, bool enableThinking = false)
        {
            var text = new StringBuilder(template).Append('|');
            if (tools != null)
                text.Append("[tools:").Append(string.Join(",", tools.Select(t => t.Name))).Append(']');
            foreach (ChatMessage message in messages)
                text.Append('<').Append(message.Role).Append('>').Append(message.Content).Append("</>");
            if (addGenerationPrompt)
                text.Append("<assistant>");
            return text.ToString();
        }
    }

    /// <summary>One token per character, folded into the oracle model's small vocabulary.</summary>
    private sealed class SmallTokenizer : ITokenizer
    {
        public string[] Vocab => Array.Empty<string>();
        public int BosTokenId => -1;
        public int[] EosTokenIds => Array.Empty<int>();
        public int VocabSize => OracleModel.DefaultVocab;
        public List<int> Encode(string text, bool addSpecial = true) =>
            (text ?? string.Empty).Select(c => 20 + c % 200).ToList();
        public string Decode(List<int> ids) => new(ids.Select(id => (char)('a' + id % 26)).ToArray());
        public void AppendTokenBytes(int tokenId, List<byte> buffer) => buffer.Add((byte)('a' + tokenId % 26));
        public bool IsEos(int tokenId) => false;
        public int LookupToken(string tokenStr) => -1;
    }
}
