using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.AgentHost.Agents;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Server;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.RequestParsers;
using TensorSharp.Server.Skills;

namespace InferenceWeb.Tests;

public class MultiAgentHostTests
{
    private static ServerHostingOptions Options(params string[] flags) =>
        ServerOptionsBuilder.Build(new[] { "--model", "test.gguf" }.Concat(flags).ToArray(), Path.GetTempPath());

    private static SkillRequestPlan Plan(ServerHostingOptions options, bool allowTools = true,
        bool? enabled = null, string architecture = "nemotron_h_moe", List<ToolFunction> clientTools = null) =>
        SkillRequestPlan.Create(null, null, false, clientTools, architecture, 32768, options,
            out _, allowTools: allowTools, multiAgent: enabled);

    [Fact]
    public void Delegation_IsIndependentOfSkillsAndCode_AndCanBeSuppressedPerRequest()
    {
        ServerHostingOptions options = Options();
        SkillRequestPlan plan = Plan(options);
        Assert.NotNull(plan);
        Assert.False(plan.IsEmpty);
        Assert.All(plan.Tools, tool => Assert.True(MultiAgentTools.IsTool(tool.Name)));
        Assert.Null(plan.ToolContext.CodeRunner);
        Assert.Null(plan.ToolContext.ScriptRunner);
        Assert.Empty(plan.ToolContext.Reachable);
        Assert.Null(Plan(options, enabled: false));
        Assert.Null(Plan(options, allowTools: false));
        Assert.Null(Plan(options, architecture: "mistral3"));
        Assert.Null(Plan(Options("--no-multi-agent"), enabled: true));
    }

    [Fact]
    public void ServerFlags_ResolveBounds_AndClientToolNamesWin()
    {
        ServerHostingOptions options = Options("--agents-max-concurrent=2", "--agents-max-count", "5",
            "--agents-max-depth", "1", "--agents-max-rounds", "6", "--agents-max-generations", "20",
            "--agents-timeout", "90", "--agents-max-result-chars", "2000", "--agents-allow-worker-tools");
        Assert.Equal(2, options.MultiAgent.MaxConcurrentAgents);
        Assert.Equal(5, options.MultiAgent.MaxAgents);
        Assert.Equal(1, options.MultiAgent.MaxDepth);
        Assert.Equal(6, options.MultiAgent.MaxRoundsPerAgent);
        Assert.Equal(20, options.MultiAgent.MaxTotalChildGenerations);
        Assert.Equal(90, options.MultiAgent.AgentTimeoutSeconds);
        Assert.Equal(2000, options.MultiAgent.MaxResultCharacters);
        Assert.True(options.MultiAgent.AllowWorkerTools);
        var owned = new ToolFunction { Name = "spawn_agent", Description = "client implementation" };
        SkillRequestPlan plan = Plan(options, clientTools: new List<ToolFunction> { owned });
        Assert.Same(owned, Assert.Single(plan.Tools, t => t.Name == "spawn_agent"));
        SkillTools.Partition(new[] { new ToolCall { Name = "spawn_agent" } }, plan.ClientTools,
            out var host, out var client, out var unknown);
        Assert.Empty(host);
        Assert.Single(client);
        Assert.Empty(unknown);
    }

    /// <summary>
    /// TensorAgent's Sub-agents switch moves delegation on the running host, the way its
    /// skills switch does. The next request plan has to see the change, a request still
    /// cannot turn back on what the host turned off, and every limit has to survive the
    /// move: a repoint that quietly reset a bound to its default would be a second,
    /// invisible settings change. Every limit is varied by reflection, so a limit added
    /// to MultiAgentOptions later is covered without anyone remembering this test.
    /// </summary>
    [Fact]
    public void RepointMultiAgent_ReachesTheNextPlan_AndKeepsEveryLimit()
    {
        var configured = new MultiAgentOptions();
        foreach (PropertyInfo property in typeof(MultiAgentOptions).GetProperties())
        {
            if (property.Name == nameof(MultiAgentOptions.Enabled))
                continue;
            object value = property.GetValue(configured);
            if (value is int number)
                property.SetValue(configured, number + 1);
            else if (value is bool flag)
                property.SetValue(configured, !flag);
            else
                Assert.Fail($"Teach this test to vary MultiAgentOptions.{property.Name} ({property.PropertyType.Name}).");
        }
        configured.Validate();
        ServerHostingOptions options = HostingOptions(configured);
        Assert.NotNull(Plan(options));

        options.RepointMultiAgent(true);
        Assert.Same(configured, options.MultiAgent);

        options.RepointMultiAgent(false);
        Assert.False(options.MultiAgent.Enabled);
        Assert.Null(Plan(options));
        Assert.Null(Plan(options, enabled: true));

        options.RepointMultiAgent(true);
        SkillRequestPlan plan = Plan(options);
        Assert.NotNull(plan);
        Assert.Contains(plan.Tools, tool => tool.Name == MultiAgentTools.Spawn);
        Assert.Same(options.MultiAgent, plan.MultiAgent);
        Assert.NotSame(configured, options.MultiAgent);
        foreach (PropertyInfo property in typeof(MultiAgentOptions).GetProperties())
            Assert.Equal(property.GetValue(configured), property.GetValue(options.MultiAgent));
    }

    private static ServerHostingOptions HostingOptions(MultiAgentOptions multiAgent) => new(
        startupModelPath: null,
        startupMmProjPath: null,
        defaultBackend: "ggml_cpu",
        supportedBackends: Array.Empty<BackendOption>(),
        defaultMaxTokens: 512,
        maxTokensPinned: false,
        defaultVideoFrames: 0,
        defaultVideoFps: 0,
        defaultVideoWidth: 0,
        defaultVideoHeight: 0,
        defaultVideoSteps: 0,
        defaultVideoMode: null,
        uploadDirectory: string.Empty,
        logDirectory: string.Empty,
        fileLoggingEnabled: false,
        samplingDefaults: new SamplingDefaults(new SamplingConfig()),
        multiAgent: multiAgent);

    [Theory]
    [InlineData("--agents-max-concurrent", "0")]
    [InlineData("--agents-max-depth", "9")]
    [InlineData("--agents-timeout", "-1")]
    [InlineData("--agents-max-result-chars", "999999")]
    [InlineData("--agents-max-count", "invalid")]
    public void InvalidAgentBounds_FailAtStartup(string flag, string value) =>
        Assert.ThrowsAny<ArgumentException>(() => Options(flag, value));

    [Theory]
    [InlineData("{}", null)]
    [InlineData("{\"multi_agent\":false}", false)]
    [InlineData("{\"multi_agent\":true}", true)]
    [InlineData("{\"multi_agent\":\"false\"}", null)]
    public void RequestControl_ReadsOnlyBoolean(string json, bool? expected)
    {
        using JsonDocument document = JsonDocument.Parse(json);
        Assert.Equal(expected, SkillSelectionParser.ParseMultiAgent(document.RootElement));
    }

    [Theory]
    [InlineData(8)]
    [InlineData(1)]
    public async Task StreamingLoop_CollectsChildEvidenceBeforeFinalAnswer_EvenAtRoundLimit(int maxRounds)
    {
        SkillRequestPlan plan = Plan(Options("--skills-max-rounds", maxRounds.ToString()));
        var messages = new List<ChatMessage> { new() { Role = "user", Content = "review independently" } };
        await using var agents = new MultiAgentSession(messages, plan.Tools, plan.ToolContext,
            id => (_, _, _) => Task.FromResult(new SkillTurnOutput(new ParsedOutput
            {
                Content = "CHILD_PRIVATE: checked evidence E-17",
            })), plan.MultiAgent);
        plan.Agents = agents;
        int round = 0;
        SkillChatGeneration generate = (history, _, ct) =>
        {
            round++;
            if (round == 1)
                return Emit("<tool_call>{\"name\":\"spawn_agent\",\"arguments\":{\"task_name\":\"review\",\"task\":\"Check E-17 independently\"}}</tool_call>", ct);
            if (maxRounds > 1 && round == 2)
                return Emit("PREMATURE_FINAL", ct);
            Assert.Contains(history, message => message.Content?.Contains("checked evidence E-17", StringComparison.Ordinal) == true);
            if (maxRounds > 1)
            {
                ChatMessage provisional = Assert.Single(history, message => message.Content == "PREMATURE_FINAL");
                Assert.Equal(new[] { 11, 22 }, provisional.RawOutputTokens);
                Assert.Equal("\n", provisional.RawPromptTrailingWhitespace);
                Assert.Equal("generation-prefix", provisional.RawGenerationSuffix);
            }
            return Emit("Verified final synthesis.", ct);
        };
        var updates = new List<ChatStreamUpdate>();
        await foreach (ChatStreamUpdate update in SkillChatLoop.RunAsync("nemotron_h_moe", messages,
            plan, false, generate, null, CancellationToken.None)) updates.Add(update);
        string visible = string.Concat(updates.Where(u => !u.Done).Select(u => u.Piece));
        Assert.Equal("Verified final synthesis.", visible);
        Assert.DoesNotContain("CHILD_PRIVATE", visible);
        Assert.DoesNotContain("PREMATURE_FINAL", visible);
        Assert.False(agents.HasPendingResults("/root"));
        Assert.Single(updates, u => u.Done);
        Assert.Contains(plan.Invocations, i => i.Tool == "spawn_agent" && i.Ok);
    }

    [Fact]
    public async Task StreamingLoop_SimpleTasksStreamImmediately_WithoutCreatingChildren()
    {
        SkillRequestPlan plan = Plan(Options());
        var messages = new List<ChatMessage> { new() { Role = "user", Content = "What is 17+25?" } };
        await using var agents = new MultiAgentSession(messages, plan.Tools, plan.ToolContext,
            id => throw new InvalidOperationException("A simple answer must not create a child."), plan.MultiAgent);
        plan.Agents = agents;
        var updates = new List<ChatStreamUpdate>();
        await foreach (ChatStreamUpdate update in SkillChatLoop.RunAsync("nemotron_h_moe", messages,
            plan, false, (_, _, ct) => Emit("The answer is 42.", ct), null, CancellationToken.None))
            updates.Add(update);
        Assert.Equal("The answer is 42.", string.Concat(updates.Where(u => !u.Done).Select(u => u.Piece)));
        Assert.True(updates.Count(u => !string.IsNullOrEmpty(u.Piece)) > 1);
        Assert.Empty(plan.Invocations);
        Assert.Equal(0, agents.TotalChildGenerations);
    }

    private static async IAsyncEnumerable<ChatStreamUpdate> Emit(string text,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        foreach (char c in text)
        {
            cancellationToken.ThrowIfCancellationRequested();
            yield return ChatStreamUpdate.Text(c.ToString());
            await Task.Yield();
        }
        yield return new ChatStreamUpdate("", true, 10, 5, 0, 0, 0, 0, "stop")
        {
            RawOutputTokens = new List<int> { 11, 22 },
            RawPromptTrailingWhitespace = "\n",
            RawGenerationSuffix = "generation-prefix",
        };
    }
}
