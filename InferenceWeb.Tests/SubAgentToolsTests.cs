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
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
using TensorSharp.AgentHost.Agents;

namespace InferenceWeb.Tests;

/// <summary>
/// The five sub-agent tool declarations, the name classifier, and how the shared tool
/// dispatcher treats agent calls. Model-free.
///
/// <para>
/// The declarations are read on every turn of every agentic request that enables
/// sub-agents, so the same rules the other built-in declarations follow apply here:
/// flat parameters only, a non-empty <c>required</c> whenever there are parameters (the
/// Jinja path otherwise marks EVERY parameter required), and no vendor, product or
/// worked example in any description.
/// </para>
/// </summary>
public class SubAgentToolsTests : IDisposable
{
    private readonly string _base;

    public SubAgentToolsTests()
    {
        _base = Path.Combine(Path.GetTempPath(), "ts-subagent-tools-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_base);
    }

    public void Dispose()
    {
        try { Directory.Delete(_base, recursive: true); } catch { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    private static readonly string[] ExpectedNames =
        { "spawn_agent", "send_input", "wait_agent", "close_agent", "list_agents" };

    // ---- declarations ----------------------------------------------------------

    [Fact]
    public void Declare_IsExactlyTheFiveAgentTools_InDeclarationOrder()
    {
        List<ToolFunction> tools = SubAgentTools.Declare();

        Assert.Equal(ExpectedNames, tools.Select(t => t.Name).ToArray());
        Assert.Equal(ExpectedNames, SkillToolNames.AgentTools.ToArray());
    }

    [Fact]
    public void Declare_ReturnsAFreshListEachTime()
    {
        // A host appends these to a per-request tool list; a shared instance mutated by
        // one request would leak into the next.
        List<ToolFunction> first = SubAgentTools.Declare();
        List<ToolFunction> second = SubAgentTools.Declare();
        Assert.NotSame(first, second);
        Assert.NotSame(first[0], second[0]);
    }

    [Fact]
    public void Declare_ParametersAreFlatAndWellFormed()
    {
        var namePattern = new Regex("^[a-z][a-z0-9_]*$", RegexOptions.CultureInvariant);
        var flatTypes = new HashSet<string>(StringComparer.Ordinal) { "string", "integer", "boolean" };

        foreach (ToolFunction tool in SubAgentTools.Declare())
        {
            Assert.Matches(namePattern, tool.Name);
            Assert.False(string.IsNullOrWhiteSpace(tool.Description), tool.Name + " has no description");
            Assert.NotNull(tool.Parameters);
            Assert.NotNull(tool.Required);

            foreach ((string name, ToolParameter parameter) in tool.Parameters)
            {
                Assert.Matches(namePattern, name);
                Assert.Contains(parameter.Type, flatTypes);
                Assert.False(string.IsNullOrWhiteSpace(parameter.Description), $"{tool.Name}.{name} has no description");
            }

            foreach (string required in tool.Required)
                Assert.True(tool.Parameters.ContainsKey(required), $"{tool.Name} requires undeclared '{required}'");
            Assert.Equal(tool.Required.Count, tool.Required.Distinct(StringComparer.Ordinal).Count());

            // The Jinja rendering path marks every parameter required when Required is
            // empty, so an "all optional" declaration is not expressible: a tool with
            // parameters must name at least one required parameter.
            if (tool.Parameters.Count > 0)
                Assert.NotEmpty(tool.Required);
        }
    }

    [Fact]
    public void Declare_RequiredParametersAreTheOnesTheReadersNeed()
    {
        Dictionary<string, ToolFunction> tools = SubAgentTools.Declare().ToDictionary(t => t.Name);

        Assert.Equal(new[] { "message" }, tools["spawn_agent"].Required);
        Assert.Equal(new[] { "message", "fork_context" }, tools["spawn_agent"].Parameters.Keys.ToArray());
        Assert.Equal("boolean", tools["spawn_agent"].Parameters["fork_context"].Type);

        Assert.Equal(new[] { "target", "message" }, tools["send_input"].Required);
        Assert.Equal(new[] { "targets" }, tools["wait_agent"].Required);
        Assert.Equal("integer", tools["wait_agent"].Parameters["timeout_ms"].Type);
        Assert.Equal(new[] { "target" }, tools["close_agent"].Required);
        Assert.Empty(tools["list_agents"].Parameters);
        Assert.Empty(tools["list_agents"].Required);
    }

    [Fact]
    public void Declare_TheJinjaRenderingKeepsTimeoutOptional()
    {
        // The reason wait_agent declares Required = ["targets"] at all: with an empty
        // list the Jinja declaration builder would force timeout_ms on every call.
        Dictionary<string, ToolFunction> tools = SubAgentTools.Declare().ToDictionary(t => t.Name);

        Dictionary<string, object> wait = ChatTemplate.BuildToolDeclaration(tools["wait_agent"]);
        var waitParameters = (Dictionary<string, object>)((Dictionary<string, object>)wait["function"])["parameters"];
        Assert.Equal(new object[] { "targets" }, ((List<object>)waitParameters["required"]).ToArray());

        Dictionary<string, object> spawn = ChatTemplate.BuildToolDeclaration(tools["spawn_agent"]);
        var spawnParameters = (Dictionary<string, object>)((Dictionary<string, object>)spawn["function"])["parameters"];
        Assert.Equal(new object[] { "message" }, ((List<object>)spawnParameters["required"]).ToArray());

        Dictionary<string, object> list = ChatTemplate.BuildToolDeclaration(tools["list_agents"]);
        var listParameters = (Dictionary<string, object>)((Dictionary<string, object>)list["function"])["parameters"];
        Assert.False(listParameters.ContainsKey("required"));
    }

    [Fact]
    public void Declare_NoDescriptionNamesAVendorProductOrWorkedExample()
    {
        string[] mustNotAppear =
        {
            // The list the code-tool declarations are held to (AgentAppHostTests).
            "yahoo", "yfinance", "day_gainers", "scrIds", "quoteType", "regularMarket",
            "screener", "gainers", "ticker", "NASDAQ", "S&P",
            "openai", "github.com/", "api_key", "bitcoin", "weather.com",
            // The reference these tools were modelled on, and other vendors/products:
            // the source comments cite them, the declarations must not.
            "codex", "claude", "anthropic", "chatgpt", "gpt-", "gemini", "llama", "qwen",
            // Worked examples.
            "e.g.", "for example", "for instance", "example:",
        };

        foreach (ToolFunction tool in SubAgentTools.Declare())
        {
            IEnumerable<(string Where, string Text)> texts = new[] { (tool.Name, tool.Description) }
                .Concat(tool.Parameters.Select(p => (tool.Name + "." + p.Key, p.Value.Description)));
            foreach ((string where, string text) in texts)
            {
                foreach (string banned in mustNotAppear)
                {
                    Assert.False(
                        text.Contains(banned, StringComparison.OrdinalIgnoreCase),
                        $"the '{where}' description names '{banned}'. A declaration is read on every turn, so a "
                        + "task-specific example or a product name in it biases every unrelated request.");
                }
            }
        }
    }

    [Fact]
    public void Declare_TheNumbersInWaitAgentsDescriptionAreTheOnesEnforced()
    {
        string timeout = SubAgentTools.Declare().Single(t => t.Name == "wait_agent").Parameters["timeout_ms"].Description;
        Assert.Contains("Default 300000", timeout, StringComparison.Ordinal);
        Assert.Contains("minimum 10000", timeout, StringComparison.Ordinal);
        Assert.Contains("maximum 3600000", timeout, StringComparison.Ordinal);
        Assert.Equal(300_000, SubAgentTools.DefaultWaitMs);
        Assert.Equal(10_000, SubAgentTools.MinWaitMs);
        Assert.Equal(3_600_000, SubAgentTools.MaxWaitMs);
    }

    // ---- names -------------------------------------------------------------------

    [Fact]
    public void IsAgentTool_IsOrdinalAndExact()
    {
        foreach (string name in ExpectedNames)
        {
            Assert.True(SkillToolNames.IsAgentTool(name), name);
            Assert.True(SkillTools.IsBuiltInTool(name), name);
        }

        foreach (string? name in new[]
                 {
                     null, "", "Spawn_Agent", "SPAWN_AGENT", "spawn_agent ", " spawn_agent", "spawn-agent",
                     "spawnagent", "spawn_agents", "agent", "wait", "close", "list", "shell", "skills_list",
                 })
        {
            Assert.False(SkillToolNames.IsAgentTool(name), name ?? "(null)");
        }
    }

    [Fact]
    public void AgentToolNames_DoNotCollideWithCodeOrSkillTools()
    {
        foreach (string name in ExpectedNames)
        {
            Assert.False(SkillToolNames.IsCodeTool(name), name);
            Assert.False(SkillTools.IsSkillTool(name), name);
        }
    }

    [Fact]
    public void Partition_PutsAgentCallsInTheBuiltInBucket()
    {
        var calls = ExpectedNames.Select(n => new ToolCall { Name = n })
            .Append(new ToolCall { Name = "get_weather" })
            .Append(new ToolCall { Name = "no_such_tool" })
            .ToList();

        SkillTools.Partition(
            calls, new[] { new ToolFunction { Name = "get_weather" } },
            out List<ToolCall> builtIn, out List<ToolCall> client, out List<ToolCall> unknown);

        Assert.Equal(ExpectedNames, builtIn.Select(c => c.Name).ToArray());
        Assert.Equal(new[] { "get_weather" }, client.Select(c => c.Name).ToArray());
        Assert.Equal(new[] { "no_such_tool" }, unknown.Select(c => c.Name).ToArray());

        // With no client roster (the CLI's --tools contract) an agent call is still ours.
        SkillTools.Partition(calls.Take(5), null, out builtIn, out client, out unknown);
        Assert.Equal(5, builtIn.Count);
        Assert.Empty(client);
        Assert.Empty(unknown);
    }

    [Fact]
    public void Partition_AClientThatDeclaresTheSameNameWins()
    {
        var calls = new List<ToolCall> { new() { Name = "spawn_agent" }, new() { Name = "wait_agent" } };

        SkillTools.Partition(
            calls, new[] { new ToolFunction { Name = "spawn_agent" } },
            out List<ToolCall> builtIn, out List<ToolCall> client, out List<ToolCall> unknown);

        Assert.Equal(new[] { "spawn_agent" }, client.Select(c => c.Name).ToArray());
        Assert.Equal(new[] { "wait_agent" }, builtIn.Select(c => c.Name).ToArray());
        Assert.Empty(unknown);
    }

    // ---- dispatch ------------------------------------------------------------------

    [Theory]
    [InlineData("spawn_agent")]
    [InlineData("send_input")]
    [InlineData("wait_agent")]
    [InlineData("close_agent")]
    [InlineData("list_agents")]
    public void Execute_WithoutARuntime_SaysSubAgentsAreNotEnabled(string name)
    {
        var call = new ToolCall
        {
            Name = name,
            Arguments = new Dictionary<string, object?> { ["message"] = "do it", ["target"] = "agent_1", ["targets"] = "all" },
        };

        SkillToolResult result = SkillTools.Execute(call, new SkillToolContext(Array.Empty<Skill>()));

        Assert.False(result.Ok);
        Assert.Equal("Error: sub-agents are not enabled on this host. Do the task yourself.", result.Content);
    }

    [Fact]
    public void Execute_AnAgentCall_DoesNotStageAttachments_ButABuiltInDoes()
    {
        string source = Path.Combine(_base, "upload.csv");
        File.WriteAllText(source, "a,b\n1,2\n");
        var workspaces = new SessionWorkspaceManager(Path.Combine(_base, "workspaces"));

        // Without a runtime: the refusal path must not touch the workspace.
        SessionWorkspace plain = workspaces.GetOrCreate("no-runtime");
        var plainContext = new SkillToolContext(Array.Empty<Skill>())
        {
            Workspace = plain,
            CodeInputFiles = new[] { new CodeInputFile("form.csv", source) },
        };
        SkillToolResult refused = SkillTools.Execute(
            new ToolCall { Name = "spawn_agent", Arguments = new Dictionary<string, object?> { ["message"] = "x" } },
            plainContext);
        Assert.False(refused.Ok);
        Assert.False(File.Exists(Path.Combine(plain.WorkDirectory, "form.csv")),
            "an agent call staged the attachments");

        // With a runtime: the synchronous dispatch path answers and still stages nothing.
        SessionWorkspace withRuntime = workspaces.GetOrCreate("with-runtime");
        var baseContext = new SkillToolContext(Array.Empty<Skill>())
        {
            Workspace = withRuntime,
            CodeInputFiles = new[] { new CodeInputFile("form.csv", source) },
        };
        using (var harness = new SubAgentHarness(context: baseContext))
        {
            SkillToolContext rootContext = harness.Runtime.RootContext;
            Assert.Same(harness.Runtime.Root, rootContext.Agents);
            Assert.Same(withRuntime, rootContext.Workspace);
            Assert.Same(baseContext.CodeInputFiles, rootContext.CodeInputFiles);

            SkillToolResult listed = SkillTools.Execute(new ToolCall { Name = "list_agents" }, rootContext);
            Assert.True(listed.Ok, listed.Content);
            Assert.Equal("You have no sub-agents. Start one with spawn_agent.", listed.Content);
            Assert.False(File.Exists(Path.Combine(withRuntime.WorkDirectory, "form.csv")),
                "an agent call staged the attachments");

            // Control: the same context DOES stage for an ordinary built-in, so the
            // assertions above are not vacuous.
            SkillTools.Execute(new ToolCall { Name = SkillTools.ListToolName }, rootContext);
            Assert.True(File.Exists(Path.Combine(withRuntime.WorkDirectory, "form.csv")),
                "the control built-in did not stage; the no-staging checks above prove nothing");
        }
    }

    [Fact]
    public void WithAgents_CopiesEveryOtherField()
    {
        var skills = Array.Empty<Skill>();
        var context = new SkillToolContext(skills, maxReadBytes: 1234)
        {
            CodeInputFiles = new[] { new CodeInputFile("a.txt", "/nowhere/a.txt") },
        };
        using var harness = new SubAgentHarness(context: context);

        SkillToolContext attached = context.WithAgents(harness.Runtime.Root);
        Assert.Same(harness.Runtime.Root, attached.Agents);
        Assert.Same(context.Reachable, attached.Reachable);
        Assert.Equal(1234, attached.MaxReadBytes);
        Assert.Same(context.CodeInputFiles, attached.CodeInputFiles);
        Assert.Null(context.Agents);

        Assert.Null(attached.WithAgents(null).Agents);
    }

    // ---- argument readers ------------------------------------------------------------

    private static ToolCall With(string name, object? value) =>
        new() { Name = "x", Arguments = new Dictionary<string, object?> { [name] = value } };

    [Theory]
    [InlineData("true", true)]
    [InlineData("True", true)]
    [InlineData(" yes ", true)]
    [InlineData("1", true)]
    [InlineData("false", false)]
    [InlineData("no", false)]
    [InlineData("0", false)]
    [InlineData("", false)]
    public void ReadFlag_AcceptsTheSpellingsModelsSend(string text, bool expected)
    {
        Assert.Equal(expected, SubAgentTools.ReadFlag(With("fork_context", text), "fork_context"));
    }

    [Fact]
    public void ReadFlag_AcceptsJsonAndBoxedValues()
    {
        Assert.True(SubAgentTools.ReadFlag(With("f", true), "f"));
        Assert.False(SubAgentTools.ReadFlag(With("f", false), "f"));
        Assert.True(SubAgentTools.ReadFlag(With("f", JsonDocument.Parse("true").RootElement), "f"));
        Assert.False(SubAgentTools.ReadFlag(With("f", JsonDocument.Parse("false").RootElement), "f"));
        Assert.True(SubAgentTools.ReadFlag(With("f", JsonDocument.Parse("1").RootElement), "f"));
        Assert.True(SubAgentTools.ReadFlag(With("f", JsonDocument.Parse("\"true\"").RootElement), "f"));
        Assert.True(SubAgentTools.ReadFlag(With("f", 1L), "f"));
        Assert.False(SubAgentTools.ReadFlag(With("f", 0), "f"));
        Assert.False(SubAgentTools.ReadFlag(With("f", null), "f"));
        Assert.False(SubAgentTools.ReadFlag(new ToolCall { Name = "x" }, "f"));
    }

    [Fact]
    public void ReadIds_AcceptsEveryShape()
    {
        Assert.Null(SubAgentTools.ReadIds(new ToolCall { Name = "x" }, "targets"));
        Assert.Equal(new[] { "agent_1", "agent_2" }, SubAgentTools.ReadIds(With("targets", "agent_1, agent_2"), "targets"));
        Assert.Equal(new[] { "agent_1", "agent_2" }, SubAgentTools.ReadIds(With("targets", "agent_1 agent_2"), "targets"));
        Assert.Equal(new[] { "agent_1", "agent_2" }, SubAgentTools.ReadIds(With("targets", "[\"agent_1\", \"agent_2\"]"), "targets"));
        Assert.Equal(new[] { "agent_1", "agent_2" },
            SubAgentTools.ReadIds(With("targets", JsonDocument.Parse("[\"agent_1\",\"agent_2\"]").RootElement), "targets"));
        Assert.Equal(new[] { "1" }, SubAgentTools.ReadIds(With("targets", JsonDocument.Parse("1").RootElement), "targets"));
        Assert.Equal(new[] { "agent_1" }, SubAgentTools.ReadIds(With("targets", new List<string> { "agent_1", "agent_1" }), "targets"));
        Assert.Empty(SubAgentTools.ReadIds(With("targets", ""), "targets")!);

        // The first argument name that is present wins; later aliases are only fallbacks.
        Assert.Equal(new[] { "agent_3" }, SubAgentTools.ReadIds(With("agent_ids", "agent_3"), "targets", "agent_ids"));
    }

    [Fact]
    public void TryReadTimeout_DefaultsClampsAndRejects()
    {
        Assert.True(SubAgentTools.TryReadTimeout(new ToolCall { Name = "x" }, out int ms, out string? note, out string? error));
        Assert.Equal(SubAgentTools.DefaultWaitMs, ms);
        Assert.Null(note);
        Assert.Null(error);

        Assert.True(SubAgentTools.TryReadTimeout(With("timeout_ms", 60_000L), out ms, out note, out error));
        Assert.Equal(60_000, ms);
        Assert.Null(note);

        Assert.True(SubAgentTools.TryReadTimeout(With("timeout_ms", 5), out ms, out note, out _));
        Assert.Equal(SubAgentTools.MinWaitMs, ms);
        Assert.Equal("Requested timeout of 5 ms was clamped to 10000 ms.", note);

        Assert.True(SubAgentTools.TryReadTimeout(With("timeout_ms", "7200000"), out ms, out note, out _));
        Assert.Equal(SubAgentTools.MaxWaitMs, ms);
        Assert.Equal("Requested timeout of 7200000 ms was clamped to 3600000 ms.", note);

        Assert.True(SubAgentTools.TryReadTimeout(With("timeout_ms", long.MaxValue), out ms, out _, out _));
        Assert.Equal(SubAgentTools.MaxWaitMs, ms);

        foreach (object value in new object[] { 0, -5, -1L })
        {
            Assert.False(SubAgentTools.TryReadTimeout(With("timeout_ms", value), out _, out _, out error));
            Assert.Equal("timeout_ms must be greater than zero", error);
        }
    }
}
