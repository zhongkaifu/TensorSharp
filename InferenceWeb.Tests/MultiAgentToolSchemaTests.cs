// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.

using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using TensorSharp.AgentHost.Agents;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class MultiAgentToolSchemaTests
{
    [Fact]
    public void MergeUsesAStableSharedToolPrefixWithoutChangingSchemasOrInputs()
    {
        var suppliedWait = new ToolFunction { Name = MultiAgentTools.Wait, Description = "Keep this existing schema." };
        var input = new List<ToolFunction>
        {
            new() { Name = "client_lookup" },
            new() { Name = SkillTools.ReadToolName },
            new() { Name = SkillTools.RunToolName },
            suppliedWait,
            new() { Name = SkillToolNames.ReadFile },
            new() { Name = "client_submit" },
        };
        ToolFunction[] originalObjects = input.ToArray();
        string originalSchemas = JsonSerializer.Serialize(input);
        List<ToolFunction> merged = MultiAgentTools.Merge(input);
        Assert.Equal(new[]
        {
            SkillTools.ReadToolName, MultiAgentTools.Wait, SkillToolNames.ReadFile,
            MultiAgentTools.Spawn, MultiAgentTools.Send, MultiAgentTools.Close, MultiAgentTools.List,
            "client_lookup", SkillTools.RunToolName, "client_submit",
        }, merged.Select(tool => tool.Name));
        Assert.Equal(originalObjects, input);
        Assert.Equal(originalSchemas, JsonSerializer.Serialize(input));
        foreach (ToolFunction original in originalObjects)
            Assert.Same(original, Assert.Single(merged, tool => tool.Name == original.Name));
        Assert.Same(suppliedWait, Assert.Single(merged, tool => tool.Name == MultiAgentTools.Wait));
        Assert.Equal(merged, MultiAgentTools.Merge(merged));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ProfilesPreserveReadOnlyToolPrefixAndWithholdUnscopedMutation(bool allowWorkerTools)
    {
        var tools = SkillTools.BuiltIn(allowScripts: true);
        tools.Add(new() { Name = SkillToolNames.Shell, Description = "Execute a workspace command." });
        tools.Add(new() { Name = "client_submit", Description = "An external client tool." });
        List<ToolFunction> parentTools = MultiAgentTools.Merge(tools);
        var messages = new List<ChatMessage> { new() { Role = "system", Content = "Preserve governing instructions." } };
        await using var session = new MultiAgentSession(messages, parentTools, new SkillToolContext([]),
            _ => (_, _, _) => Task.FromResult(new SkillTurnOutput(new ParsedOutput { Content = "unused" })),
            new MultiAgentOptions { Enabled = true, AllowWorkerTools = allowWorkerTools });
        string parent = ChatTemplate.RenderQwen35(messages, tools: parentTools);
        IReadOnlyList<MultiAgentPromptProfile> profiles = session.GetPromptProfiles();
        foreach (MultiAgentPromptProfile profile in profiles)
        {
            Assert.DoesNotContain(profile.Tools, tool => tool.Name == "client_submit");
            bool readOnly = profile.Messages[0].Content!.Contains("You are read-only.");
            if (!readOnly)
            {
                Assert.True(allowWorkerTools);
                // A declaration alone cannot authorize execution in a child workspace.
                Assert.DoesNotContain(profile.Tools, tool => tool.Name is SkillTools.RunToolName or SkillToolNames.Shell);
                continue;
            }
            Assert.DoesNotContain(profile.Tools, tool => tool.Name is SkillTools.RunToolName or SkillToolNames.Shell);
            Assert.Equal(JsonSerializer.Serialize(parentTools.Take(profile.Tools.Count)), JsonSerializer.Serialize(profile.Tools));
            string child = ChatTemplate.RenderQwen35(profile.Messages.ToList(), tools: profile.Tools.ToList());
            int childToolsEnd = child.IndexOf("\n</tools>", StringComparison.Ordinal);
            Assert.True(childToolsEnd > 0);
            Assert.StartsWith(child[..childToolsEnd], parent);
        }
    }

    [Theory]
    [InlineData("system")]
    [InlineData("developer")]
    public void CoordinationPrompt_PreservesStableCacheBoundaryAndHistoryMetadata(string role)
    {
        const string preamble = "Stable governing instructions.";
        var first = new ChatMessage
        {
            Role = role,
            Content = preamble + "  ",
            CacheControl = new CacheControlMarker(),
            ContentCacheBreakpoints = new List<int> { 6 },
        };
        var prior = new ChatMessage
        {
            Role = "assistant", Content = "Earlier answer.",
            RawOutputTokens = new List<int> { 11, 22, 33 },
            RawPromptTrailingWhitespace = "\n",
            RawGenerationSuffix = "<think>\n",
            AttachmentPaths = new List<string> { "/uploads/source.pdf" },
            AttachmentNames = new List<string> { "report.pdf" },
            TextFilePaths = new List<string> { "/uploads/notes.txt" },
            ToolCallId = "call-17",
        };

        List<ChatMessage> injected = MultiAgentPrompt.Apply(
            new List<ChatMessage> { first, prior }, new MultiAgentOptions());

        Assert.Equal(2, injected.Count);
        Assert.Equal(role, injected[0].Role);
        Assert.StartsWith(preamble + "\n\n", injected[0].Content);
        Assert.Contains("[TensorSharp multi-agent coordination]", injected[0].Content);
        Assert.Null(injected[0].CacheControl);
        Assert.Equal(new[] { 6, preamble.Length }, injected[0].ContentCacheBreakpoints);
        Assert.NotNull(first.CacheControl);
        Assert.Equal(preamble + "  ", first.Content);
        Assert.Equal(new[] { 6 }, first.ContentCacheBreakpoints);

        Assert.NotSame(prior, injected[1]);
        Assert.Equal(prior.RawOutputTokens, injected[1].RawOutputTokens);
        Assert.Equal(prior.RawPromptTrailingWhitespace, injected[1].RawPromptTrailingWhitespace);
        Assert.Equal(prior.RawGenerationSuffix, injected[1].RawGenerationSuffix);
        Assert.Equal(prior.ToolCallId, injected[1].ToolCallId);
        Assert.Equal(prior.AttachmentPaths, injected[1].AttachmentPaths);
        Assert.Equal(prior.AttachmentNames, injected[1].AttachmentNames);
        Assert.Equal(prior.TextFilePaths, injected[1].TextFilePaths);
        injected[1].AttachmentPaths!.Clear();
        injected[1].AttachmentNames!.Clear();
        injected[1].RawOutputTokens!.Clear();
        Assert.Single(prior.AttachmentPaths);
        Assert.Single(prior.AttachmentNames);
        Assert.Equal(3, prior.RawOutputTokens.Count);
    }

    [Fact]
    public void JinjaToolSchema_PreservesOptionalWaitAndRequiredSpawnParameters()
    {
        // Inspect the actual model-visible Jinja context, not only the C# schema.
        string rendered = ChatTemplate.RenderFromGgufTemplate(
            "{{ tools | tojson }}", new List<ChatMessage>(),
            addGenerationPrompt: false, tools: MultiAgentTools.Create());
        using JsonDocument document = JsonDocument.Parse(rendered);
        JsonElement Function(string name) => document.RootElement.EnumerateArray()
            .Select(tool => tool.GetProperty("function"))
            .Single(function => function.GetProperty("name").GetString() == name);

        JsonElement wait = Function(MultiAgentTools.Wait).GetProperty("parameters");
        Assert.Equal(0, wait.GetProperty("required").GetArrayLength());
        Assert.Equal("string", wait.GetProperty("properties").GetProperty("agent_id")
            .GetProperty("type").GetString());
        Assert.Equal("integer", wait.GetProperty("properties").GetProperty("timeout_ms")
            .GetProperty("type").GetString());

        JsonElement spawn = Function(MultiAgentTools.Spawn).GetProperty("parameters");
        Assert.Equal(new[] { "task_name", "task" }, spawn.GetProperty("required")
            .EnumerateArray().Select(parameter => parameter.GetString()).ToArray());
        Assert.True(spawn.GetProperty("properties").TryGetProperty("agent_type", out _));
        Assert.Equal("string", spawn.GetProperty("properties").GetProperty("depends_on").GetProperty("type").GetString());
        JsonElement permissions = spawn.GetProperty("properties").GetProperty("permissions");
        Assert.Equal("string", permissions.GetProperty("type").GetString());
        Assert.Contains("read-only", permissions.GetProperty("description").GetString());
        Assert.Contains("workspace-write", permissions.GetProperty("description").GetString());
        Assert.Equal("string", spawn.GetProperty("properties").GetProperty("input_files").GetProperty("type").GetString());
    }
}
