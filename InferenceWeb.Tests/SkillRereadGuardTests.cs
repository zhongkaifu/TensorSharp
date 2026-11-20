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
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;
using TensorSharp.Server;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// A skill file asked for again in the same turn is answered from the first read, and a model that asks once more
/// after being told so stops calling tools. LFM3-Omni-2B, offered the server's skills, re-read the same SKILL.md round
/// after round and restated the same sentence each time until the round cap (eng/lfm3/loop): the user saw one sentence
/// over and over.
/// </summary>
public sealed class SkillRereadGuardTests : IDisposable
{
    private const string Architecture = "nemotron_h_moe";
    private readonly string _baseDir;

    public SkillRereadGuardTests()
    {
        _baseDir = Path.Combine(Path.GetTempPath(), "ts-skill-reread-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_baseDir);
        string dir = Path.Combine(_baseDir, "alpha");
        Directory.CreateDirectory(dir);
        File.WriteAllText(Path.Combine(dir, "SKILL.md"),
            "---\nname: alpha\ndescription: does alpha things\n---\n\nUse the reference in guide.md.\n");
        File.WriteAllText(Path.Combine(dir, "guide.md"), "The magic number is 1238.");
    }

    public void Dispose()
    {
        try { Directory.Delete(_baseDir, recursive: true); } catch { /* best effort */ }
    }

    private SkillRequestPlan Plan()
    {
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = new[] { _baseDir } });
        ServerHostingOptions options = ServerOptionsBuilder.Build(
            new[] { "--model", "x.gguf", "--skills-dir", _baseDir }, _baseDir);
        SkillRequestPlan plan = SkillRequestPlan.Create(
            registry, new[] { "alpha" }, false, null, Architecture,
            contextTokens: 32768, options, out IReadOnlyList<string> unknown);
        Assert.Empty(unknown);
        return plan;
    }

    private static string Read(string path) =>
        "<think>reading</think>Let me read the guide.<tool_call>\n{\"name\": \"skills_read\", \"arguments\": {\"skill\": \"alpha\", \"path\": \""
        + path + "\"}}\n</tool_call>";

    /// <summary>Replays the rounds in order (the last one again past the end) and records what each round was shown.</summary>
    private static SkillChatGeneration Replay(List<List<ChatMessage>> seen, params string[] rounds)
    {
        int round = 0;
        return (messages, tools, ct) =>
        {
            seen.Add(messages.Select(m => new ChatMessage { Role = m.Role, Content = m.Content }).ToList());
            return Emit(rounds[Math.Min(round++, rounds.Length - 1)], ct);
        };
    }

    private static async IAsyncEnumerable<ChatStreamUpdate> Emit(string text, [EnumeratorCancellation] CancellationToken ct)
    {
        foreach (char c in text)
        {
            ct.ThrowIfCancellationRequested();
            yield return ChatStreamUpdate.Text(c.ToString());
            await Task.Yield();
        }
        yield return new ChatStreamUpdate(string.Empty, true, 10, 20, 5, 0, 0, 0, "stop");
    }

    private static async Task<string> Run(SkillRequestPlan plan, SkillChatGeneration generate)
    {
        var messages = new List<ChatMessage> { new() { Role = "user", Content = "go" } };
        var content = new System.Text.StringBuilder();
        await foreach (ChatStreamUpdate u in SkillChatLoop.RunAsync(
            Architecture, messages, plan, true, generate, logger: null, CancellationToken.None))
        {
            if (!u.Done)
                content.Append(u.Piece);
        }
        return content.ToString();
    }

    [Fact]
    public async Task ASkillFileReadAgain_IsAnsweredFromTheFirstRead()
    {
        SkillRequestPlan plan = Plan();
        var seen = new List<List<ChatMessage>>();
        string content = await Run(plan, Replay(seen, Read("guide.md"), Read("guide.md"), "<think>ok</think>It is 1238."));

        Assert.Equal(3, seen.Count);
        Assert.Contains(seen[1], m => m.Content.Contains("The magic number is 1238.", StringComparison.Ordinal));
        // The repeat is told the file has not changed; its text is not sent a second time.
        ChatMessage repeat = seen[2].Last();
        Assert.Contains("alpha/guide.md has not changed since you read it", repeat.Content, StringComparison.Ordinal);
        Assert.DoesNotContain("The magic number", repeat.Content, StringComparison.Ordinal);
        Assert.Equal(1, seen[2].Count(m => m.Content.Contains("The magic number is 1238.", StringComparison.Ordinal)));
        Assert.Equal(2, plan.Invocations.Count);
        Assert.All(plan.Invocations, i => Assert.Equal("guide.md", i.ResourcePath));
        Assert.EndsWith("It is 1238.", content, StringComparison.Ordinal);
        Assert.False(plan.StuckRepeating);
    }

    [Fact]
    public async Task AskingAgainAfterBeingTold_EndsToolUseAndTheNextRoundAnswers()
    {
        SkillRequestPlan plan = Plan();
        var seen = new List<List<ChatMessage>>();
        // Every round would re-read the file, to the round cap, were the loop not to stop it.
        string content = await Run(plan, Replay(seen, Read("guide.md"), Read("guide.md"), Read("guide.md"),
            "<think>done</think>It is 1238."));

        Assert.True(plan.StuckRepeating);
        // Three tool rounds, then the final round, not the eight-round cap.
        Assert.Equal(4, seen.Count);
        Assert.Equal(SkillChatLoop.StuckRepeatingNotice, seen[3].Last().Content);
        Assert.EndsWith("It is 1238.", content, StringComparison.Ordinal);
    }

    [Fact]
    public async Task AnotherFileOfTheSameSkill_IsReadAsUsual()
    {
        SkillRequestPlan plan = Plan();
        var seen = new List<List<ChatMessage>>();
        await Run(plan, Replay(seen, Read("guide.md"), Read("SKILL.md"), "<think>ok</think>Done."));

        Assert.Equal(3, seen.Count);
        Assert.Contains("Use the reference in guide.md.", seen[2].Last().Content, StringComparison.Ordinal);
        Assert.DoesNotContain("has not changed", seen[2].Last().Content, StringComparison.Ordinal);
    }

    [Fact]
    public void ACoordinationCallIsKeyedByItsArguments_AndAWaitIsNot()
    {
        SkillRequestPlan plan = Plan();
        ToolCall Call(string name, string arguments) => new()
        {
            Name = name,
            Arguments = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>(arguments),
        };
        string a = SkillChatLoop.RepeatKey(Call("spawn_agent", "{\"task_name\": \"x\", \"agent_type\": \"worker\"}"), plan);
        string b = SkillChatLoop.RepeatKey(Call("spawn_agent", "{\"agent_type\": \"worker\", \"task_name\": \"x\"}"), plan);
        string c = SkillChatLoop.RepeatKey(Call("spawn_agent", "{\"task_name\": \"y\", \"agent_type\": \"worker\"}"), plan);
        Assert.NotNull(a);
        Assert.Equal(a, b);
        Assert.NotEqual(a, c);
        Assert.Null(SkillChatLoop.RepeatKey(Call("wait_agent", "{\"targets\": [\"x\"]}"), plan));
        Assert.Null(SkillChatLoop.RepeatKey(Call("shell", "{\"command\": \"ls\"}"), plan));
    }

    [Fact]
    public void ARepeatedFailedCall_IsNotRunAndQuotesTheEarlierError()
    {
        SkillToolResult repeated = SkillChatLoop.RepeatedCallResult(new RepeatedToolCall
        {
            Tool = "spawn_agent", Error = "{\"error\":\"agent_type must be explorer, reviewer or worker.\"}",
        });
        Assert.False(repeated.Ok);
        Assert.Contains("this exact spawn_agent call already failed", repeated.Content, StringComparison.Ordinal);
        Assert.Contains("agent_type must be explorer, reviewer or worker.", repeated.Content, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData("{\"skill\": \"alpha\"}", "alpha\nSKILL.md\n0")]
    [InlineData("{\"skill\": \"alpha\", \"path\": \"alpha/guide.md\"}", "alpha\nguide.md\n0")]
    [InlineData("{\"skill\": \"alpha\", \"path\": \"./guide.md\", \"offset\": 64}", "alpha\nguide.md\n64")]
    [InlineData("{\"skill\": \"nobody\"}", null)]
    public void TheReadKeyIsWhatTheReadResolves(string arguments, string expected)
    {
        SkillRequestPlan plan = Plan();
        var call = new ToolCall
        {
            Name = SkillTools.ReadToolName,
            Arguments = System.Text.Json.JsonSerializer.Deserialize<Dictionary<string, object>>(arguments),
        };
        Assert.Equal(expected, SkillTools.RepeatableReadKey(call, plan.ToolContext));
        Assert.Null(SkillTools.RepeatableReadKey(new ToolCall { Name = SkillTools.ListToolName }, plan.ToolContext));
    }

    [Fact]
    public async Task TheSkillListAskedForAgain_IsAnsweredFromTheFirstListing()
    {
        SkillRequestPlan plan = Plan();
        var seen = new List<List<ChatMessage>>();
        const string list = "<think>which</think>Let me check the skills.<tool_call>\n{\"name\": \"skills_list\", \"arguments\": {}}\n</tool_call>";
        await Run(plan, Replay(seen, list, list, "<think>ok</think>Done."));

        Assert.Equal(3, seen.Count);
        Assert.Contains("alpha", seen[1].Last().Content, StringComparison.Ordinal);
        Assert.StartsWith("The skill list has not changed since you listed it", seen[2].Last().Content, StringComparison.Ordinal);
        Assert.NotNull(SkillChatLoop.RepeatKey(new ToolCall { Name = SkillTools.ListToolName }, plan));
    }
}
