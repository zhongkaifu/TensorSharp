// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Nemotron-H 8B Reasoning-128K opens the answer with "<think></think>" (or "<think>\n")
// but renders a past assistant turn as its bare content. Nothing declared that suffix, so
// in TensorAgent a follow-up's prompt left the cache six tokens before the previous prompt
// ended (structural=7252 of a 7258-token prompt) and, the recurrent state being unable to
// rewind, reused only the 7168-token system prompt pages: every follow-up re-prefilled the
// whole conversation. A tool round did the same, its call re-rendered as JSON where the
// model had written a Python call. And the model writes the template's "\n" before its end
// token itself, so a spliced turn doubled it.
namespace InferenceWeb.Tests;

public sealed class NemotronHFollowUpReuseTests
{
    private const string ReasoningTemplate =
        "{{ '<SPECIAL_10>System\n' }}{% for message in messages %}{{ '\n<SPECIAL_11>Assistant\n' }}{% endfor %}";
    private const string ChatMlTemplate = "{% for message in messages %}<|im_start|>{{ message['role'] }}{% endfor %}";

    private static readonly List<ToolFunction> Tools =
    [
        new()
        {
            Name = "shell",
            Parameters = new Dictionary<string, ToolParameter> { ["command"] = new() { Type = "string" } },
            Required = ["command"],
        },
    ];

    [Theory]
    [InlineData(false, "<think></think>")]
    [InlineData(true, "<think>\n")]
    public void TheGenerationSuffix_DependsOnTheTemplate(bool thinking, string suffix)
    {
        Assert.Equal(suffix, KVCachePromptRenderer.GetAssistantGenerationSuffix("nemotron_h", thinking, ReasoningTemplate));
        Assert.Equal(string.Empty, KVCachePromptRenderer.GetAssistantGenerationSuffix("nemotron_h", thinking, ChatMlTemplate));
        Assert.Equal(string.Empty, KVCachePromptRenderer.GetAssistantGenerationSuffix("nemotron_h", thinking));
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void AFollowUp_StartsWithEverythingTheCacheHolds(bool thinking)
    {
        var tokenizer = new ByteTokenizer();
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        var first = new List<ChatMessage>
        {
            new() { Role = "system", Content = "You are terse." },
            new() { Role = "user", Content = "What is the capital of France?" },
        };
        List<int> prompt1 = renderer.RenderToTokens(tokenizer, ReasoningTemplate, first, "nemotron_h", true,
            out _, out string trailing, tools: null, enableThinking: thinking);
        List<int> raw = tokenizer.Encode(thinking ? "France's capital.\n</think>\nParis.\n" : "Paris.\n", addSpecial: false);
        string suffix = TensorSharp.Server.ChatGenerationPipeline.RecordedGenerationSuffix(
            tokenizer, prompt1, "nemotron_h", thinking, ReasoningTemplate);
        Assert.Equal(thinking ? "<think>\n" : "<think></think>", suffix);

        var second = new List<ChatMessage>(first)
        {
            new() { Role = "assistant", Content = "Paris.", RawOutputTokens = raw, RawGenerationSuffix = suffix,
                RawPromptTrailingWhitespace = trailing },
            new() { Role = "user", Content = "And Italy?" },
        };
        List<int> prompt2 = renderer.RenderToTokens(tokenizer, ReasoningTemplate, second, "nemotron_h", true,
            tools: null, enableThinking: thinking);

        // The cache holds the reply AND its end token: the model wrote "Paris.\n" and then
        // <SPECIAL_11>, the template's own "\n<SPECIAL_11>" turn close.
        var cached = new List<int>(prompt1);
        cached.AddRange(raw);
        cached.AddRange(tokenizer.Encode("<SPECIAL_11>", addSpecial: false));
        AssertStartsWith(cached, prompt2, tokenizer);
    }

    /// <summary>
    /// A caller's system prompt can switch the template with <c>{'reasoning': True|False}</c>
    /// whatever the request's flag says. The recorded suffix is what the prompt ended with,
    /// so the re-render puts back the block the cache holds; it used to record nothing for
    /// the other mode, and the follow-up left the cache at the first assistant boundary.
    /// </summary>
    [Theory]
    [InlineData(false, "{'reasoning': True}", "<think>\n", "France's capital.\n</think>\nParis.\n")]
    [InlineData(true, "{'reasoning': False}", "<think></think>", "Paris.\n")]
    public void AMarkerThatOverridesTheRequest_StillStartsWithEverythingTheCacheHolds(
        bool requestThinking, string marker, string opening, string reply)
    {
        var tokenizer = new ByteTokenizer();
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        var first = new List<ChatMessage>
        {
            new() { Role = "system", Content = "You are terse. " + marker },
            new() { Role = "user", Content = "What is the capital of France?" },
        };
        List<int> prompt1 = renderer.RenderToTokens(tokenizer, ReasoningTemplate, first, "nemotron_h", true,
            out _, out string trailing, tools: null, enableThinking: requestThinking);
        Assert.EndsWith(opening, tokenizer.Decode(prompt1));
        List<int> raw = tokenizer.Encode(reply, addSpecial: false);
        string suffix = TensorSharp.Server.ChatGenerationPipeline.RecordedGenerationSuffix(
            tokenizer, prompt1, "nemotron_h", requestThinking, ReasoningTemplate);
        Assert.Equal(opening, suffix);

        var second = new List<ChatMessage>(first)
        {
            new() { Role = "assistant", Content = "Paris.", RawOutputTokens = raw, RawGenerationSuffix = suffix,
                RawPromptTrailingWhitespace = trailing },
            new() { Role = "user", Content = "And Italy?" },
        };
        List<int> prompt2 = renderer.RenderToTokens(tokenizer, ReasoningTemplate, second, "nemotron_h", true,
            tools: null, enableThinking: requestThinking);

        var cached = new List<int>(prompt1);
        cached.AddRange(raw);
        cached.AddRange(tokenizer.Encode("<SPECIAL_11>", addSpecial: false));
        AssertStartsWith(cached, prompt2, tokenizer);
    }

    [Fact]
    public void AToolRound_StartsWithEverythingTheCacheHolds()
    {
        var tokenizer = new ByteTokenizer();
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        var first = new List<ChatMessage> { new() { Role = "user", Content = "Lines in notes.txt?" } };
        List<int> prompt1 = renderer.RenderToTokens(tokenizer, ReasoningTemplate, first, "nemotron_h", true,
            out _, out string trailing, tools: Tools, enableThinking: false);
        // What the model wrote, a step past the call the turn ended at.
        List<int> raw = tokenizer.Encode("Counting.\n<TOOLCALL>[shell(\"wc -l < notes.txt\")]</TOOLCALL>\n", addSpecial: false);
        string suffix = TensorSharp.Server.ChatGenerationPipeline.RecordedGenerationSuffix(
            tokenizer, prompt1, "nemotron_h", false, ReasoningTemplate);

        var second = new List<ChatMessage>(first)
        {
            new()
            {
                Role = "assistant", Content = "Counting.", RawOutputTokens = raw, RawGenerationSuffix = suffix,
                RawPromptTrailingWhitespace = trailing,
                ToolCalls = [new ToolCall { Name = "shell", Arguments = new Dictionary<string, object?> { ["command"] = "wc -l < notes.txt" } }],
            },
            new() { Role = "tool", Content = "3" },
        };
        List<int> prompt2 = renderer.RenderToTokens(tokenizer, ReasoningTemplate, second, "nemotron_h", true,
            tools: Tools, enableThinking: false);

        var cached = new List<int>(prompt1);
        cached.AddRange(raw);
        AssertStartsWith(cached, prompt2, tokenizer);
        Assert.EndsWith("<TOOL_RESPONSE>[3]</TOOL_RESPONSE>\n<SPECIAL_11>Assistant\n<think></think>", tokenizer.Decode(prompt2));
    }

    private static void AssertStartsWith(List<int> cached, List<int> prompt, ITokenizer tokenizer)
    {
        int same = 0;
        while (same < cached.Count && same < prompt.Count && cached[same] == prompt[same]) same++;
        Assert.True(same == cached.Count,
            $"the prompt leaves the cache at token {same} of {cached.Count}: cache '…{Tail(tokenizer, cached, same)}' vs prompt '…{Tail(tokenizer, prompt, same)}'");
    }

    private static string Tail(ITokenizer tokenizer, List<int> tokens, int at)
        => tokenizer.Decode(tokens.GetRange(Math.Max(0, at - 24), Math.Min(tokens.Count, at + 24) - Math.Max(0, at - 24)));

    /// <summary>One token per UTF-8 byte; enough to compare prompts token for token.</summary>
    private sealed class ByteTokenizer : ITokenizer
    {
        public string[] Vocab => Enumerable.Range(0, 256).Select(i => ((char)i).ToString()).ToArray();
        public int VocabSize => 256;
        public int BosTokenId => -1;
        public int[] EosTokenIds => Array.Empty<int>();
        public bool IsEos(int id) => false;
        public int LookupToken(string token) => -1;
        public List<int> Encode(string text, bool addSpecial = true)
            => System.Text.Encoding.UTF8.GetBytes(text ?? string.Empty).Select(b => (int)b).ToList();
        public string Decode(List<int> tokens) => System.Text.Encoding.UTF8.GetString(tokens.Select(t => (byte)t).ToArray());
        public void AppendTokenBytes(int token, List<byte> bytes) => bytes.Add((byte)token);
    }
}
