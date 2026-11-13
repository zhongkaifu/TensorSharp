using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.AgentHost.Agents;

namespace InferenceWeb.Tests;

public sealed class SharedPrefixRenderTests : IDisposable
{
    private readonly ModelLifecycleService _lifecycle = new(NullLogger.Instance);
    private readonly InferenceEngineHost _host;
    private readonly KVCachePromptRenderer _renderer = new(new TestRenderer());
    private readonly ChatGenerationPipeline _pipeline;

    public SharedPrefixRenderTests()
    {
        _host = new InferenceEngineHost(_lifecycle, NullLogger.Instance);
        _pipeline = new ChatGenerationPipeline(_lifecycle, _host, _renderer,
            new InferenceTelemetry(NullLogger.Instance), NullLogger.Instance);
    }

    public void Dispose() { _host.Dispose(); _lifecycle.Dispose(); }

    [Fact]
    public void WarmupAndFreshSessionsDeclareTheSameSystemPrefixWithoutSharingUserText()
    {
        ModelBase model = Model(new CharTokenizer());
        using var warmup = new ChatSession();
        using var fresh = new ChatSession();
        string warmScope = warmup.ResolveCacheScope(null);
        Assert.NotEqual(warmScope, fresh.ResolveCacheScope(null));

        int warmLength = AssertWholePrefix(model, "hi");
        Assert.Equal(warmLength, AssertWholePrefix(model, "A different question in a new session."));
        fresh.ResetConversation();
        Assert.NotEqual(warmScope, fresh.ResolveCacheScope(null));
        Assert.Equal(warmLength, AssertWholePrefix(model, "Another question after newChat."));
    }

    [Fact]
    public void TemplatePreambleIsSharedWithoutSystemMessagesOnlyForFamiliesThatOptIn()
    {
        ModelBase model = Model(new CharTokenizer());
        model.Config.ChatTemplate = "preamble: cutoff 2026-01-04, current date, reasoning level, channels";
        Assert.True(model.Config.ChatTemplate.Length + 1 >= ChatGenerationPipeline.MinSharedPrefixTokens);
        int Shared(string arch, string user)
        {
            var history = new List<ChatMessage> { new() { Role = "user", Content = user } };
            List<int> prompt = _renderer.RenderToTokens(model.Tokenizer, model.Config.ChatTemplate, history,
                arch, addGenerationPrompt: true);
            return _pipeline.ComputeSharedPrefixTokens(model, history, prompt, arch, [], false);
        }

        // Muse Spark: the preamble its template renders for an empty conversation, never user text.
        int expected = _renderer.RenderToTokens(model.Tokenizer, model.Config.ChatTemplate, [],
            "muse-spark", addGenerationPrompt: false).Count;
        Assert.Equal(expected, Shared("muse-spark", "Write about hash tables."));
        Assert.Equal(expected, Shared("muse-spark", "A different first question."));
        // Families without the opt-in keep requiring system messages or tools.
        Assert.Equal(0, Shared("qwen35", "Write about hash tables."));
    }

    [Theory]
    [InlineData("type")]
    [InlineData("description")]
    [InlineData("enum")]
    [InlineData("required")]
    [InlineData("raw-schema")]
    public void ChangedToolSchemaWithTheSameParameterCountIsRenderedAgain(string change)
    {
        ModelBase model = Model(new CharTokenizer());
        var tool = new ToolFunction
        {
            Name = "inspect", Description = "Inspect the supplied value.",
            Parameters = new() { ["value"] = new ToolParameter { Type = "string", Description = "Old value." } },
            Required = new() { "value" },
        };
        var tools = new List<ToolFunction> { tool };
        AssertWholePrefix(model, "hi", tools);
        switch (change)
        {
            case "type": tool.Parameters["value"].Type = "number"; break;
            case "description": tool.Parameters["value"].Description = "Changed value."; break;
            case "enum": tool.Parameters["value"].Enum.Add("new-option"); break;
            case "required": tool.Required.Clear(); break;
            case "raw-schema": tool.ParametersSchemaJson = "{\"type\":\"object\",\"additionalProperties\":false}"; break;
        }
        AssertWholePrefix(model, "A fresh chat with changed tools.", tools);
    }

    [Fact]
    public void ModelReloadWithDifferentTokenizerDoesNotReuseOldTokenIds()
    {
        AssertWholePrefix(Model(new CharTokenizer()), "hi");
        AssertWholePrefix(Model(new CharTokenizer(100_000)), "New model, same system text.");
    }

    [Fact]
    public void TemplateChangeWithTheSameTokenizerDoesNotReuseOldRender()
    {
        ModelBase model = Model(new CharTokenizer());
        AssertWholePrefix(model, "hi");
        model.Config.ChatTemplate = "changed-template";
        AssertWholePrefix(model, "A fresh chat after changing the template.");
    }

    [Fact]
    public void AgentProfilesCaptureTheExactCommonAncestorBeforeDifferentToolDeclarations()
    {
        ModelBase model = Model(new CharTokenizer());
        var common = new ToolFunction { Name = "skills_read", Description = new string('r', 120) };
        var rootTools = new List<ToolFunction> { common, new() { Name = "skills_run" } };
        var childTools = new List<ToolFunction> { common, new() { Name = "spawn_agent" } };
        var root = new List<ChatMessage> { new() { Role = "system", Content = new string('s', 120) } };
        var child = new List<ChatMessage> { new() { Role = "system", Content = new string('s', 120) + " Read-only reviewer." } };
        var profiles = new[] { new MultiAgentPromptProfile(root, rootTools), new MultiAgentPromptProfile(child, childTools) };
        List<int> rootPrompt = Render(model, root, rootTools, "Parent private task.");
        List<int> childPrompt = Render(model, child, childTools, "Child private task.");
        int rootShared = _pipeline.ComputeSharedPrefixTokens(model, root, rootPrompt, "qwen35", rootTools, false);
        int childShared = _pipeline.ComputeSharedPrefixTokens(model, child, childPrompt, "qwen35", childTools, false);
        int expected = CommonPrefix(rootPrompt, childPrompt);

        int rootBoundary = Assert.Single(_pipeline.ComputePublicCheckpointBoundaries(
            model, rootPrompt, rootShared, profiles, "qwen35", false));
        int childBoundary = Assert.Single(_pipeline.ComputePublicCheckpointBoundaries(
            model, childPrompt, childShared, profiles, "qwen35", false));
        Assert.Equal(expected, rootBoundary);
        Assert.Equal(rootBoundary, childBoundary);
        Assert.InRange(rootBoundary, ChatGenerationPipeline.MinSharedPrefixTokens, Math.Min(rootShared, childShared) - 1);
        Assert.Equal(rootPrompt.Take(rootBoundary), childPrompt.Take(childBoundary));
        Assert.NotEqual(rootPrompt[rootBoundary], childPrompt[childBoundary]);
    }

    [Fact]
    public void AgentProfilesKeepOneAncestorUnderTheDefaultTwoCheckpointBudget()
    {
        ModelBase model = Model(new CharTokenizer());
        var root = new List<ChatMessage> { new() { Role = "system", Content = new string('s', 200) } };
        var longer = new List<ChatMessage> { new() { Role = "system", Content = new string('s', 150) + "worker" } };
        var shorter = new List<ChatMessage> { new() { Role = "system", Content = new string('s', 90) + "reviewer" } };
        List<int> prompt = Render(model, root, [], "Private parent task.");
        int shared = _pipeline.ComputeSharedPrefixTokens(model, root, prompt, "qwen35", [], false);
        int expected = CommonPrefix(prompt, Render(model, shorter, [], "Private child task."));
        var profiles = new[] { new MultiAgentPromptProfile(longer, []), new MultiAgentPromptProfile(shorter, []) };
        Assert.Equal(expected, Assert.Single(_pipeline.ComputePublicCheckpointBoundaries(
            model, prompt, shared, profiles, "qwen35", false)));
    }

    [Fact]
    public void AgentProfileCheckpointsNeverExtendPastTheCurrentPublicPrefix()
    {
        ModelBase model = Model(new CharTokenizer());
        var messages = new List<ChatMessage>
        {
            new() { Role = "system", Content = new string('s', 100) },
            new() { Role = "user", Content = new string('p', 200) },
        };
        List<int> prompt = _renderer.RenderToTokens(model.Tokenizer, model.Config.ChatTemplate, messages,
            "qwen35", addGenerationPrompt: true);
        var profiles = new[] { new MultiAgentPromptProfile(messages, []) };
        int shared = _pipeline.ComputeSharedPrefixTokens(model, messages, prompt, "qwen35", [], false);
        Assert.True(shared < prompt.Count - 200);
        Assert.Empty(_pipeline.ComputePublicCheckpointBoundaries(model, prompt, shared, profiles, "qwen35", false));
        Assert.Empty(_pipeline.ComputePublicCheckpointBoundaries(model, prompt, 64, profiles, "qwen35", false));
        Assert.Empty(_pipeline.ComputePublicCheckpointBoundaries(model, prompt, 0, profiles, "qwen35", false));
    }

    [Fact]
    public void UnusableAgentProfileDoesNotInventACommonCheckpoint()
    {
        ModelBase model = Model(new CharTokenizer());
        var root = new List<ChatMessage> { new() { Role = "system", Content = new string('s', 100) } };
        List<int> prompt = Render(model, root, [], "Parent task.");
        int shared = _pipeline.ComputeSharedPrefixTokens(model, root, prompt, "qwen35", [], false);
        var mismatched = new[] { new MultiAgentPromptProfile(
            [new() { Role = "system", Content = "A different system prompt." }], []) };
        Assert.Empty(_pipeline.ComputePublicCheckpointBoundaries(model, prompt, shared, mismatched, "qwen35", false));
        var media = new[] { new MultiAgentPromptProfile(
            [new() { Role = "system", Content = new string('s', 100), ImagePaths = new() { "image.png" } }], []) };
        Assert.Empty(_pipeline.ComputePublicCheckpointBoundaries(model, prompt, shared, media, "qwen35", false));
    }

    private List<int> Render(ModelBase model, List<ChatMessage> governing, List<ToolFunction> tools, string user) =>
        _renderer.RenderToTokens(model.Tokenizer, model.Config.ChatTemplate,
            new List<ChatMessage>(governing) { new() { Role = "user", Content = user } },
            "qwen35", addGenerationPrompt: true, tools: tools);

    private static int CommonPrefix(IReadOnlyList<int> a, IReadOnlyList<int> b)
    {
        int count = 0;
        while (count < a.Count && count < b.Count && a[count] == b[count]) count++;
        return count;
    }

    private int AssertWholePrefix(ModelBase model, string user, List<ToolFunction> tools = null)
    {
        var shared = new List<ChatMessage>
        {
            new() { Role = "system", Content = new string('s', 100) },
            new() { Role = "developer", Content = "Keep each answer concise and use the supplied tools." },
        };
        var history = new List<ChatMessage>(shared) { new() { Role = "user", Content = user } };
        List<int> prompt = _renderer.RenderToTokens(model.Tokenizer, model.Config.ChatTemplate, history,
            "qwen35", addGenerationPrompt: true, tools: tools);
        List<int> expected = _renderer.RenderToTokens(model.Tokenizer, model.Config.ChatTemplate, shared,
            "qwen35", addGenerationPrompt: false, tools: tools);
        int measured = _pipeline.ComputeSharedPrefixTokens(model, history, prompt, "qwen35", tools, false);
        Assert.True(expected.Count >= ChatGenerationPipeline.MinSharedPrefixTokens);
        Assert.Equal(expected.Count, measured);
        Assert.True(measured < prompt.Count);
        return measured;
    }

    private static ModelBase Model(ITokenizer tokenizer)
    {
        var model = (Qwen35RuntimeModel)RuntimeHelpers.GetUninitializedObject(typeof(Qwen35RuntimeModel));
        typeof(ModelBase).GetProperty(nameof(ModelBase.Tokenizer))!.SetValue(model, tokenizer);
        typeof(ModelBase).GetField("<Config>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic)!
            .SetValue(model, new ModelConfig { Architecture = "qwen35", ChatTemplate = "first-template" });
        return model;
    }

    private sealed class TestRenderer : IPromptRenderer
    {
        public string Render(string template, List<ChatMessage> messages, bool addGenerationPrompt = true,
            string architecture = null, List<ToolFunction> tools = null, bool enableThinking = false)
        {
            var text = new StringBuilder(template).Append('|');
            if (tools != null)
                foreach (var tool in tools)
                    text.Append(JsonSerializer.Serialize(tool)).Append(tool.ParametersSchemaJson);
            foreach (var message in messages)
                text.Append('<').Append(message.Role).Append('>').Append(message.Content).Append("</>");
            if (addGenerationPrompt) text.Append("<assistant>");
            return text.ToString();
        }
    }

    private sealed class CharTokenizer(int offset = 0) : ITokenizer
    {
        public string[] Vocab => Array.Empty<string>();
        public int BosTokenId => 0;
        public int[] EosTokenIds => new[] { 1 };
        public int VocabSize => char.MaxValue + offset + 1;
        public List<int> Encode(string text, bool addSpecial = true) => text.Select(c => c + offset).ToList();
        public string Decode(List<int> ids) => new(ids.Select(id => (char)(id - offset)).ToArray());
        public void AppendTokenBytes(int tokenId, List<byte> buffer) => buffer.AddRange(Encoding.UTF8.GetBytes(new[] { (char)(tokenId - offset) }));
        public bool IsEos(int tokenId) => tokenId == 1;
        public int LookupToken(string tokenStr) => tokenStr.Length == 1 ? tokenStr[0] + offset : -1;
    }
}
