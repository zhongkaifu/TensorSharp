// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Runtime.CompilerServices;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Runtime;
using TensorSharp.Server;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Skills;

namespace InferenceWeb.Tests;

/// <summary>
/// Pins the server-side handoff of the exact prompt boundary that preceded a raw
/// assistant-token run. Gemma tool continuations can have no whitespace where an
/// ordinary assistant turn has a newline, so preserving only the raw tokens is not
/// sufficient to reconstruct the live-cache prefix.
/// </summary>
public sealed class RawPromptBoundaryPropagationTests : IDisposable
{
    private const string Architecture = "qwen35";
    private readonly string _skillsDir = Path.Combine(
        Path.GetTempPath(), "ts-prompt-boundary-" + Guid.NewGuid().ToString("N"));

    public RawPromptBoundaryPropagationTests()
    {
        string skillDir = Path.Combine(_skillsDir, "alpha");
        Directory.CreateDirectory(skillDir);
        File.WriteAllText(
            Path.Combine(skillDir, "SKILL.md"),
            "---\nname: alpha\ndescription: boundary test skill\n---\n\nRead the requested reference.\n");
        File.WriteAllText(Path.Combine(skillDir, "one.md"), "one");
        File.WriteAllText(Path.Combine(skillDir, "two.md"), "two");
    }

    public void Dispose()
    {
        try { Directory.Delete(_skillsDir, recursive: true); } catch { /* best effort */ }
        GC.SuppressFinalize(this);
    }

    [Fact]
    public void RecordedTurns_RetainEveryAssistantBoundaryAcrossAugmentation()
    {
        var firstRaw = new List<int> { 101, 102 };
        var secondRaw = new List<int> { 201, 202, 203 };
        var store = new ConversationTranscriptStore(maxChains: 16, maxTokens: 10_000);

        store.Record(
            new List<ChatMessage> { new() { Role = "user", Content = "Q1" } },
            new ChatMessage { Role = "assistant", Content = "RAW1", RawOutputTokens = firstRaw, RawPromptTrailingWhitespace = "\n" },
            new EmittedAssistantTurn("PARSED1", null, null, "RAW1", false),
            scope: "s");
        store.Record(
            new List<ChatMessage>
            {
                new() { Role = "user", Content = "Q1" },
                new() { Role = "assistant", Content = "PARSED1" },
                new() { Role = "user", Content = "Q2" },
            },
            new ChatMessage { Role = "assistant", Content = "RAW2", RawOutputTokens = secondRaw, RawPromptTrailingWhitespace = string.Empty },
            new EmittedAssistantTurn("PARSED2", null, null, "RAW2", false),
            scope: "s");

        // A later HTTP request contains client-visible text rather than raw model
        // output. Augmentation must restore BOTH token runs and their own boundaries.
        var nextRequest = new List<ChatMessage>
        {
            new() { Role = "user", Content = "Q1" },
            new() { Role = "assistant", Content = "PARSED1" },
            new() { Role = "user", Content = "Q2" },
            new() { Role = "assistant", Content = "PARSED2" },
            new() { Role = "user", Content = "Q3" },
        };

        List<ChatMessage> augmented = store.Augment(nextRequest).History;

        Assert.Same(firstRaw, augmented[1].RawOutputTokens);
        Assert.Equal("\n", augmented[1].RawPromptTrailingWhitespace);
        Assert.Same(secondRaw, augmented[3].RawOutputTokens);
        Assert.Equal(string.Empty, augmented[3].RawPromptTrailingWhitespace);
    }

    [Fact]
    public async Task SkillChatLoop_CarriesEachRoundsOwnBoundaryIntoTheNextGeneration()
    {
        SkillRequestPlan plan = CreatePlan();
        var snapshots = new List<List<ChatMessage>>();
        string[] rounds =
        {
            ToolRound("one.md"),
            ToolRound("two.md"),
            "<think>done</think>finished",
        };
        List<int>[] rawTokens =
        {
            new() { 101, 102 },
            new() { 201, 202, 203 },
            new() { 301 },
        };
        string[] boundaries = { "\n", string.Empty, " \n" };
        int round = 0;

        SkillChatGeneration generate = (messages, _, ct) =>
        {
            snapshots.Add(messages.Select(CloneForAssertion).ToList());
            int current = round++;
            return Emit(rounds[current], rawTokens[current], boundaries[current], ct);
        };

        var updates = new List<ChatStreamUpdate>();
        await foreach (ChatStreamUpdate update in SkillChatLoop.RunAsync(
            Architecture,
            new List<ChatMessage> { new() { Role = "user", Content = "go" } },
            plan,
            enableThinking: true,
            generate,
            logger: null,
            CancellationToken.None))
        {
            updates.Add(update);
        }

        Assert.Equal(3, snapshots.Count);
        Assert.DoesNotContain(snapshots[0], m => m.Role == "assistant");

        ChatMessage firstRound = Assert.Single(snapshots[1], m => m.Role == "assistant");
        Assert.Equal(new[] { 101, 102 }, firstRound.RawOutputTokens);
        Assert.Equal("\n", firstRound.RawPromptTrailingWhitespace);

        ChatMessage[] thirdPromptRounds = snapshots[2].Where(m => m.Role == "assistant").ToArray();
        Assert.Equal(2, thirdPromptRounds.Length);
        Assert.Equal("\n", thirdPromptRounds[0].RawPromptTrailingWhitespace);
        Assert.Equal(string.Empty, thirdPromptRounds[1].RawPromptTrailingWhitespace);
        Assert.Equal(new[] { 201, 202, 203 }, thirdPromptRounds[1].RawOutputTokens);

        ChatStreamUpdate terminal = Assert.Single(updates, u => u.Done);
        Assert.Equal(new[] { 301 }, terminal.RawOutputTokens);
        Assert.Equal(" \n", terminal.RawPromptTrailingWhitespace);
        Assert.Equal(2, plan.Invocations.Count);
    }

    private SkillRequestPlan CreatePlan()
    {
        var registry = new SkillRegistry(new SkillRegistryOptions { Roots = new[] { _skillsDir } });
        ServerHostingOptions options = ServerOptionsBuilder.Build(
            new[] { "--model", "x.gguf", "--skills-dir", _skillsDir }, _skillsDir);
        SkillRequestPlan plan = SkillRequestPlan.Create(
            registry,
            new[] { "alpha" },
            discovery: false,
            clientTools: null,
            Architecture,
            contextTokens: 32768,
            options,
            out IReadOnlyList<string> unknown);

        Assert.Empty(unknown);
        return Assert.IsType<SkillRequestPlan>(plan);
    }

    private static string ToolRound(string path) =>
        "<think>read it</think><tool_call>\n" +
        $"{{\"name\": \"skills_read\", \"arguments\": {{\"skill\": \"alpha\", \"path\": \"{path}\"}}}}\n" +
        "</tool_call>";

    private static ChatMessage CloneForAssertion(ChatMessage message) => new()
    {
        Role = message.Role,
        Content = message.Content,
        RawOutputTokens = message.RawOutputTokens == null
            ? null
            : new List<int>(message.RawOutputTokens),
        RawPromptTrailingWhitespace = message.RawPromptTrailingWhitespace,
    };

    private static async IAsyncEnumerable<ChatStreamUpdate> Emit(
        string text,
        IReadOnlyList<int> rawTokens,
        string boundary,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        yield return ChatStreamUpdate.Text(text);
        await Task.Yield();
        yield return new ChatStreamUpdate(string.Empty, true, 10, rawTokens.Count, 5, 0, 0, 0, "stop")
        {
            RawOutputTokens = rawTokens,
            RawPromptTrailingWhitespace = boundary,
        };
    }
}
