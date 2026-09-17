using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;

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

    private int AssertWholePrefix(ModelBase model, string user, List<ToolFunction> tools = null)
    {
        var shared = new List<ChatMessage>
        {
            new() { Role = "system", Content = new string('s', 100) },
            new() { Role = "developer", Content = "Keep each answer concise and use the supplied tools." },
        };
        var history = new List<ChatMessage>(shared) { new() { Role = "user", Content = user } };
        List<int> prompt = _renderer.RenderToTokens(model.Tokenizer, model.Config.ChatTemplate, history,
            "qwen3", addGenerationPrompt: true, tools: tools);
        List<int> expected = _renderer.RenderToTokens(model.Tokenizer, model.Config.ChatTemplate, shared,
            "qwen3", addGenerationPrompt: false, tools: tools);
        int measured = _pipeline.ComputeSharedPrefixTokens(model, history, prompt, "qwen3", tools, false);
        Assert.True(expected.Count >= ChatGenerationPipeline.MinSharedPrefixTokens);
        Assert.Equal(expected.Count, measured);
        Assert.True(measured < prompt.Count);
        return measured;
    }

    private static ModelBase Model(ITokenizer tokenizer)
    {
        var model = (Qwen3Model)RuntimeHelpers.GetUninitializedObject(typeof(Qwen3Model));
        typeof(ModelBase).GetProperty(nameof(ModelBase.Tokenizer))!.SetValue(model, tokenizer);
        typeof(ModelBase).GetField("<Config>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic)!
            .SetValue(model, new ModelConfig { Architecture = "qwen3", ChatTemplate = "first-template" });
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
