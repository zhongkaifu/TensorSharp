// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Nemotron-H Reasoning-128K switches thinking with a "{'reasoning': True|False}" marker
// in the system prompt. The renderer used to PREPEND it, so a thinking-on and a
// thinking-off request diverged at the fifth token: a host's ~7.2k-token system prompt
// warmed up in one mode was no use to the other. The marker now closes the system
// section, and the two modes share the whole system text and tool declarations.
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.Extensions.Logging.Abstractions;

namespace InferenceWeb.Tests;

public sealed class NemotronHSharedPrefixTests : IDisposable
{
    // tokenizer.chat_template of the Nemotron-H 8B/47B Reasoning-128K GGUFs (as in
    // NemotronHReasoningTemplateTests).
    private const string ShippedTemplate =
        "{{ '<SPECIAL_10>System\n' }}{%- if messages and messages[0]['role'] == 'system' -%}{{ messages[0]['content'].strip() }}{%- endif -%}" +
        "{% for message in (messages[1:] if messages[0]['role'] == 'system' else messages) %}{%- if message['role'] == 'user' -%}" +
        "{{ '\n<SPECIAL_11>User\n' + message['content'].strip() + '\n<SPECIAL_11>Assistant\n' }}{%- if loop.last -%}" +
        "{%- if messages[0]['role'] == 'system' -%}{%- if \"{'reasoning': True}\" in messages[0]['content'] -%}{{ '<think>\n' }}" +
        "{%- elif \"{'reasoning': False}\" in messages[0]['content'] -%}{{ '<think></think>' }}{%- endif -%}{%- endif -%}{%- endif -%}" +
        "{%- elif message['role'] == 'assistant' -%}{{ message['content'].strip() }}{%- endif -%}{%- endfor -%}";

    private const string Arch = "nemotron_h";
    private const string MarkerStem = "{'reasoning': ";

    private readonly ModelLifecycleService _lifecycle = new(NullLogger.Instance);
    private readonly InferenceEngineHost _host;
    private readonly KVCachePromptRenderer _renderer = new(new GgufPromptRenderer());
    private readonly ChatGenerationPipeline _pipeline;

    public NemotronHSharedPrefixTests()
    {
        _host = new InferenceEngineHost(_lifecycle, NullLogger.Instance);
        _pipeline = new ChatGenerationPipeline(_lifecycle, _host, _renderer,
            new InferenceTelemetry(NullLogger.Instance), NullLogger.Instance);
    }

    public void Dispose() { _host.Dispose(); _lifecycle.Dispose(); }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void ThinkingOnAndOff_ShareTheWholeSystemSectionUpToTheMarker(bool withTools)
    {
        ModelBase model = Model(new CharTokenizer());
        string systemText = "You are a careful assistant. " + new string('s', 400) + " Follow the skill instructions.";
        var system = new ChatMessage { Role = "system", Content = systemText };
        List<ToolFunction> tools = withTools ? Tools() : null;
        var history = new List<ChatMessage> { system, new() { Role = "user", Content = "What is 17 + 25?" } };

        List<int> on = RenderPrompt(model, history, tools, thinking: true);
        List<int> off = RenderPrompt(model, history, tools, thinking: false);
        string onText = model.Tokenizer.Decode(on);
        string offText = model.Tokenizer.Decode(off);

        // The two modes diverge exactly at the marker's value: everything before
        // "{'reasoning': " - the system text and, when declared, the tools - is shared.
        int markerAt = onText.IndexOf(MarkerStem, StringComparison.Ordinal);
        Assert.True(markerAt > 0, "the thinking-on render carries no reasoning marker");
        Assert.Equal(markerAt, offText.IndexOf(MarkerStem, StringComparison.Ordinal));
        int crossMode = CommonPrefix(on, off);
        Assert.Equal(markerAt + MarkerStem.Length, crossMode);
        Assert.True(onText.IndexOf(systemText, StringComparison.Ordinal) + systemText.Length <= markerAt,
            "the system text does not lie wholly before the marker");
        if (withTools)
            Assert.True(onText.IndexOf("</tools>", StringComparison.Ordinal) < markerAt,
                "the tool declarations do not lie wholly before the marker");

        // Each mode's own shared prefix (what the engine warms and checkpoints) is the whole
        // rendered system section, marker included, and starts with the cross-mode run.
        foreach (var (prompt, text, thinking) in new[] { (on, onText, true), (off, offText, false) })
        {
            int sectionEnd = text.IndexOf("\n<SPECIAL_11>User\n", StringComparison.Ordinal);
            int shared = _pipeline.ComputeSharedPrefixTokens(model, history, prompt, Arch, tools ?? [], thinking);
            Assert.Equal(sectionEnd, shared);
            Assert.True(shared > crossMode && shared - crossMode <= "False}".Length,
                $"mode {(thinking ? "on" : "off")} shares {shared}, the modes share {crossMode}");

            // A new chat of the same mode with another question shares all of it.
            var other = new List<ChatMessage> { system, new() { Role = "user", Content = "Name the capital of France." } };
            List<int> otherPrompt = RenderPrompt(model, other, tools, thinking);
            Assert.True(CommonPrefix(prompt, otherPrompt) >= shared);
        }
    }

    private List<int> RenderPrompt(ModelBase model, List<ChatMessage> history, List<ToolFunction> tools, bool thinking)
        => _renderer.RenderToTokens(model.Tokenizer, model.Config.ChatTemplate, history, Arch,
            addGenerationPrompt: true, tools: tools, enableThinking: thinking);

    private static List<ToolFunction> Tools() => new()
    {
        new ToolFunction
        {
            Name = "skills_read",
            Description = "Read one skill's instructions.",
            Parameters = new Dictionary<string, ToolParameter> { ["name"] = new() { Type = "string", Description = "Skill name." } },
            Required = new List<string> { "name" },
        },
        new ToolFunction { Name = "shell", Description = "Run a shell command." },
    };

    private static int CommonPrefix(IReadOnlyList<int> a, IReadOnlyList<int> b)
    {
        int count = 0;
        while (count < a.Count && count < b.Count && a[count] == b[count]) count++;
        return count;
    }

    private static ModelBase Model(ITokenizer tokenizer)
    {
        var model = (NemotronModel)RuntimeHelpers.GetUninitializedObject(typeof(NemotronModel));
        typeof(ModelBase).GetProperty(nameof(ModelBase.Tokenizer))!.SetValue(model, tokenizer);
        typeof(ModelBase).GetField("<Config>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic)!
            .SetValue(model, new ModelConfig { Architecture = Arch, ChatTemplate = ShippedTemplate });
        return model;
    }

    /// <summary>One token per character, so token positions are text positions.</summary>
    private sealed class CharTokenizer : ITokenizer
    {
        public string[] Vocab => Array.Empty<string>();
        public int BosTokenId => 0;
        public int[] EosTokenIds => new[] { 1 };
        public int VocabSize => char.MaxValue + 1;
        public List<int> Encode(string text, bool addSpecial = true) => text.Select(c => (int)c).ToList();
        public string Decode(List<int> ids) => new(ids.Select(id => (char)id).ToArray());
        public void AppendTokenBytes(int tokenId, List<byte> buffer) => buffer.AddRange(Encoding.UTF8.GetBytes(new[] { (char)tokenId }));
        public bool IsEos(int tokenId) => tokenId == 1;
        public int LookupToken(string tokenStr) => tokenStr.Length == 1 ? tokenStr[0] : -1;
    }
}
