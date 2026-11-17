// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using System.Text.Json;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.ProtocolAdapters;
using TensorSharp.Server.RequestParsers;

namespace InferenceWeb.Tests;

/// <summary>
/// Liquid LFM3 / LFM3-Omni chat protocol ("lfm3"): the GGUF templates against llama.cpp's
/// renders, the fallback renderer, the registry entry, reply parsing and multi-turn KV reuse.
///
/// <para>Fixtures/Lfm3/chat_reference.json holds llama-server <c>/apply-template</c> output for
/// OpenAI request bodies (llama.cpp revision recorded in the file); the two templates beside it
/// are the GGUFs' own <c>tokenizer.chat_template</c> strings. Regenerate all three with
/// eng/lfm3/chat-template-fixtures.py. Each case runs the SAME request body through
/// TensorSharp's OpenAI message and tool parsers, so a parser that loses something the
/// template prints fails here too.</para>
/// </summary>
public class Lfm3ChatProtocolTests : IDisposable
{
    private const string Arch = "lfm3";
    private const string Bos = "<|startoftext|>";

    internal sealed record ReferenceCase(string Template, string Name, JsonElement Request, string Expected);

    private static readonly Lazy<IReadOnlyList<ReferenceCase>> Cases = new(LoadCases);
    private static readonly Lazy<string> TextTemplate = new(() => ReadResource("text_chat_template.jinja"));
    private static readonly Lazy<string> OmniTemplate = new(() => ReadResource("omni_chat_template.jinja"));

    private readonly string _uploads = Path.Combine(Path.GetTempPath(), "ts-lfm3-protocol-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try { Directory.Delete(_uploads, recursive: true); } catch (DirectoryNotFoundException) { }
    }

    private static string ReadResource(string name)
    {
        using Stream stream = typeof(Lfm3ChatProtocolTests).Assembly.GetManifestResourceStream(
            "InferenceWeb.Tests.Fixtures.Lfm3." + name)!;
        using var reader = new StreamReader(stream, Encoding.UTF8);
        return reader.ReadToEnd();
    }

    private static IReadOnlyList<ReferenceCase> LoadCases()
    {
        using var doc = JsonDocument.Parse(ReadResource("chat_reference.json"));
        return doc.RootElement.GetProperty("cases").EnumerateArray().Select(c => new ReferenceCase(
            c.GetProperty("template").GetString()!,
            c.GetProperty("name").GetString()!,
            c.GetProperty("request").Clone(),
            c.GetProperty("expected").GetString()!)).ToList();
    }

    public static IEnumerable<object[]> CaseIds()
        => LoadCases().Select(c => new object[] { c.Template, c.Name });

    private static ReferenceCase Find(string template, string name)
        => Cases.Value.Single(c => c.Template == template && c.Name == name);

    /// <summary>The request as the OpenAI adapter hands it to the renderer.</summary>
    private (List<ChatMessage> Messages, List<ToolFunction>? Tools, bool Think, bool AddGenerationPrompt) Parse(JsonElement request)
    {
        Directory.CreateDirectory(_uploads);
        var messages = ChatMessageParser.ParseOpenAI(request.GetProperty("messages"), new UploadStoragePolicy(_uploads),
            architecture: Arch);
        var tools = ToolFunctionParser.ParseOpenAI(request);
        Assert.True(ThinkingSwitchParser.TryParse(request, out bool think, out _, out string? error), error);
        bool addGenerationPrompt = !request.TryGetProperty("add_generation_prompt", out JsonElement add) || add.GetBoolean();
        return (messages, tools, think, addGenerationPrompt);
    }

    private static string TemplateFor(ReferenceCase c) => c.Template == "omni" ? OmniTemplate.Value : TextTemplate.Value;

    [Theory]
    [MemberData(nameof(CaseIds))]
    public void GgufTemplate_RendersExactlyWhatLlamaCppRenders(string template, string name)
    {
        ReferenceCase c = Find(template, name);
        var (messages, tools, think, addGenerationPrompt) = Parse(c.Request);
        string rendered = ChatTemplate.RenderFromGgufTemplate(TemplateFor(c), messages, addGenerationPrompt, Arch, tools, think);
        // TensorSharp renders bos_token as "" and the tokenizer adds BOS (ModelBase.ResolveAddBosToken);
        // llama.cpp renders its text. One BOS either way.
        Assert.Equal(c.Expected, Bos + rendered);
    }

    [Theory]
    [MemberData(nameof(CaseIds))]
    public void FallbackRenderer_RendersExactlyWhatLlamaCppRenders(string template, string name)
    {
        ReferenceCase c = Find(template, name);
        var (messages, tools, think, addGenerationPrompt) = Parse(c.Request);
        string rendered = ChatTemplate.RenderLfm3(messages, addGenerationPrompt, tools, think);
        Assert.Equal(c.Expected, Bos + rendered);
    }

    [Fact]
    public void ATemplateLessGguf_FallsBackToTheLfm3Renderer()
    {
        ReferenceCase c = Find("text", "tool_round_on");
        var (messages, tools, think, addGenerationPrompt) = Parse(c.Request);
        string rendered = ChatTemplate.RenderFromGgufTemplate(template: "", messages, addGenerationPrompt, Arch, tools, think);
        Assert.Equal(c.Expected, Bos + rendered);
    }

    [Fact]
    public void Registry_DeclaresTheLfm3Protocol()
    {
        ChatProtocol protocol = ChatProtocolRegistry.For("lfm3")!;
        Assert.Equal("lfm3", protocol.Id);
        Assert.Null(protocol.PreferOwnRenderer);
        Assert.IsType<Qwen35OutputParser>(OutputParserFactory.Create(Arch));
        Assert.False(protocol.OutputParserAlwaysRequired);
        Assert.Equal("<think>", KVCachePromptRenderer.GetAssistantGenerationSuffix(Arch, enableThinking: true));
        Assert.Equal("<think></think>", KVCachePromptRenderer.GetAssistantGenerationSuffix(Arch, enableThinking: false));
        Assert.True(protocol.EmitsEmptyThinkBlockForPastTurns!(true));
        Assert.Equal(ToolCallRawSplicing.Always, protocol.ToolCallRawSplicing);
        Assert.True(protocol.RendersToolDeclarations);
        Assert.True(protocol.RendersToolResultMessages);
        Assert.True(protocol.RendersAssistantReasoning);
        Assert.True(protocol.CapsVideoFrames);
        Assert.Equal("</think>", protocol.ThinkingBudgetEndToken);
        Assert.Equal("</think>", protocol.ThinkingGrammarActivationTrigger);
    }

    [Fact]
    public void MediaPlaceholders_AreTheTemplatesOwnSpansBeforeTheText()
    {
        var image = new ChatMessage { Role = "user", Content = "Q", ImagePaths = ["a.png", "b.png"], AudioPaths = ["c.wav"] };
        var video = new ChatMessage
        {
            Role = "user", Content = "Q", IsVideo = true, ImagePaths = ["f0.png", "f1.png", "f2.png"],
            ImageTimestamps = [0.0, 0.5, 1.0],
        };
        List<ChatMessage> injected = ChatTemplate.InjectMultimodalTokens([image, video], Arch);
        Assert.Equal(
            "<|vision_start|><|image|><|vision_end|><|vision_start|><|image|><|vision_end|>" +
            "<|vision_start|><|audio|><|vision_end|>Q", injected[0].Content);
        // A video is ONE span; the expander lays its timestamped frame pairs out inside it.
        Assert.Equal("<|vision_start|><|video|><|vision_end|>Q", injected[1].Content);
        Assert.Equal(ChatTemplate.Lfm3ImagePlaceholder, "<|vision_start|><|image|><|vision_end|>");
        Assert.Equal(ChatTemplate.Lfm3VideoPlaceholder, "<|vision_start|><|video|><|vision_end|>");
        Assert.Equal(ChatTemplate.Lfm3AudioPlaceholder, "<|vision_start|><|audio|><|vision_end|>");
    }

    // ---------------------------------------------------------------- reply parsing
    // Raw replies are llama.cpp greedy continuations of the same prompts (LFM3-1.7B-Q4_K_M and
    // LFM3-Omni-2B-Q4_K_M, llama.cpp d3c3e242c), tags as the vocabulary detokenizes them.

    private const string TwoCityCalls =
        "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=unit>\ncelsius\n</parameter>\n" +
        "</function>\n</tool_call>\n\n<tool_call>\n<function=get_weather>\n<parameter=city>\nTokyo\n</parameter>\n" +
        "<parameter=unit>\ncelsius\n</parameter>\n</function>\n</tool_call>";

    private const string ToolReasoning =
        "We need to get the weather for two cities: Paris and Tokyo, both using Celsius. The tool `get_weather` requires " +
        "parameters `city` and optionally `unit`. Since the user specified \"celsius\", we should include that for each call." +
        "\n\nI'll make two separate calls to `get_weather` for each city with unit set to \"celsius\".";

    private static List<ToolFunction> WeatherTool() => ToolFunction.ParseList("""
        [{"type":"function","function":{"name":"get_weather","description":"Get the current weather for a city.",
          "parameters":{"type":"object","properties":{"city":{"type":"string","description":"City name"},
          "unit":{"type":"string","description":"Temperature unit","enum":["celsius","fahrenheit"]}},"required":["city"]}}}]
        """);

    private static (string Content, string Thinking, List<ToolCall> Calls) ParseStreamed(string raw, bool thinking, int chunk,
        List<ToolFunction>? tools = null)
    {
        var parser = OutputParserFactory.Create(Arch);
        parser.Init(thinking, tools ?? WeatherTool());
        var content = new StringBuilder();
        var reasoning = new StringBuilder();
        var calls = new List<ToolCall>();
        void Collect(ParsedOutput delta)
        {
            content.Append(delta.Content);
            reasoning.Append(delta.Thinking);
            if (delta.ToolCalls != null) calls.AddRange(delta.ToolCalls);
        }
        for (int i = 0; i < raw.Length; i += chunk)
            Collect(parser.Add(raw.Substring(i, Math.Min(chunk, raw.Length - i)), false));
        Collect(parser.Add("", true));
        return (content.ToString(), reasoning.ToString(), calls);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(17)]
    [InlineData(4096)]
    public void ThinkingOnReply_SplitsReasoningAndBothParallelCalls(int chunk)
    {
        var (content, thinking, calls) = ParseStreamed(ToolReasoning + "</think>" + TwoCityCalls, thinking: true, chunk);
        Assert.Equal(ToolReasoning, thinking);
        Assert.True(string.IsNullOrWhiteSpace(content), content);
        Assert.Equal(["Paris", "Tokyo"], calls.Select(c => (string)c.Arguments["city"]!));
        Assert.All(calls, c => Assert.Equal("get_weather", c.Name));
        Assert.All(calls, c => Assert.Equal("celsius", c.Arguments["unit"]));
        Assert.Equal([0, 1], calls.Select(c => c.Index));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(5)]
    [InlineData(4096)]
    public void ThinkingOffReply_IsTheCallsAlone(int chunk)
    {
        var (content, thinking, calls) = ParseStreamed(TwoCityCalls, thinking: false, chunk);
        Assert.Equal("", thinking);
        Assert.True(string.IsNullOrWhiteSpace(content), content);
        Assert.Equal(2, calls.Count);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(4096)]
    public void ACallWrittenWhileReasoning_EndsTheReasoning(int chunk)
    {
        // llama.cpp's Qwen3-Coder parser (which serves this template) ends reasoning at
        // "</think>" or "<tool_call>"; the reply never closed its block.
        var (content, thinking, calls) = ParseStreamed(
            "Call the tool.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>",
            thinking: true, chunk);
        // Streamed one character at a time the newline before the tag has already gone out
        // as reasoning by the time "<tool_call>" proves it was the end of the block.
        Assert.Equal("Call the tool.", thinking.TrimEnd());
        Assert.True(string.IsNullOrWhiteSpace(content), content);
        Assert.Equal("Paris", Assert.Single(calls).Arguments["city"]);
    }

    [Theory]
    [InlineData(true, "We need to compute 17 * 23. Let's do it quickly: 17 * 20 = 340, 17 * 3 = 51, sum = 391. So answer is 391.\n\nThus final answer: 391.</think>\\(17 \\times 23 = 391\\)",
        "We need to compute 17 * 23. Let's do it quickly: 17 * 20 = 340, 17 * 3 = 51, sum = 391. So answer is 391.\n\nThus final answer: 391.",
        "\\(17 \\times 23 = 391\\)")]
    [InlineData(false, "The product of 17 and 23 is **391**.", "", "The product of 17 and 23 is **391**.")]
    // LFM3-Omni does not reason: with thinking on it closes the block at once.
    [InlineData(true, "</think>391", "", "391")]
    public void PlainReplies_ParseAsTheModelMeantThem(bool thinking, string raw, string expectedThinking, string expectedContent)
    {
        foreach (int chunk in new[] { 1, 2, 4096 })
        {
            var (content, reasoning, calls) = ParseStreamed(raw, thinking, chunk, tools: []);
            Assert.Equal(expectedThinking, reasoning);
            Assert.Equal(expectedContent, content);
            Assert.Empty(calls);
        }
    }

    [Fact]
    public void DeclaredParameterTypes_ComeBackTyped()
    {
        var tools = ToolFunction.ParseList("""
            [{"type":"function","function":{"name":"search_docs","description":"Search.",
              "parameters":{"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer"},
              "exact":{"type":"boolean"}},"required":["query"]}}}]
            """);
        var (_, _, calls) = ParseStreamed(
            "<tool_call>\n<function=search_docs>\n<parameter=query>\nline one\nline two\n</parameter>\n" +
            "<parameter=limit>\n5\n</parameter>\n<parameter=exact>\ntrue\n</parameter>\n</function>\n</tool_call>",
            thinking: false, chunk: 7, tools);
        ToolCall call = Assert.Single(calls);
        Assert.Equal("line one\nline two", call.Arguments["query"]);
        Assert.Equal(5, Convert.ToInt32(call.Arguments["limit"]));
        Assert.Equal(true, call.Arguments["exact"]);
    }

    [Fact]
    public void NonStreamingCollector_ExtractsTheSameCalls()
    {
        var collector = new ChatStreamCollector();
        collector.Add(new ChatStreamUpdate { Piece = ToolReasoning + "</think>" + TwoCityCalls });
        var output = collector.Resolve(Arch, true, WeatherTool());
        Assert.Equal(2, output.ToolCalls!.Count);
        Assert.Equal(ToolReasoning, output.Thinking);
        Assert.True(string.IsNullOrWhiteSpace(output.Content));
    }

    // ---------------------------------------------------------------- multi-turn KV reuse

    /// <summary>One token per character, so a token index is a character index.</summary>
    private sealed class CharacterTokenizer : ITokenizer
    {
        public string[] Vocab => [];
        public int BosTokenId => -1;
        public int[] EosTokenIds => [];
        public int VocabSize => char.MaxValue + 1;
        public List<int> Encode(string text, bool addSpecial = true) => text.Select(c => (int)c).ToList();
        public string Decode(List<int> ids) => new(ids.Select(i => (char)i).ToArray());
        public void AppendTokenBytes(int tokenId, List<byte> buffer)
            => buffer.AddRange(Encoding.UTF8.GetBytes(((char)tokenId).ToString()));
        public bool IsEos(int tokenId) => false;
        public int LookupToken(string tokenStr) => tokenStr.Length == 1 ? tokenStr[0] : -1;
    }

    /// <summary>
    /// The template drops a past turn's reasoning and re-renders it as
    /// <c>&lt;think&gt;&lt;/think&gt;</c> + answer, so without raw-token splicing the second
    /// turn would diverge right after the first <c>assistant\n</c>. With it, the whole first
    /// prompt plus every token the model generated is the second prompt's prefix, in both
    /// thinking modes and on both templates.
    /// </summary>
    [Theory]
    [InlineData("text", true)]
    [InlineData("text", false)]
    [InlineData("omni", false)]
    public void ASecondTurn_ExtendsTheFirstTurnsCacheExactly(string templateName, bool thinking)
    {
        string template = templateName == "omni" ? OmniTemplate.Value : TextTemplate.Value;
        var tokenizer = new CharacterTokenizer();
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        List<ChatMessage> history =
        [
            new() { Role = "system", Content = "You are a concise assistant." },
            new() { Role = "user", Content = "What is 17*23?" },
        ];
        List<int> turn1 = renderer.RenderToTokens(tokenizer, template, history, Arch, true, enableThinking: thinking);
        Assert.EndsWith(thinking ? "assistant\n<think>" : "assistant\n<think></think>", tokenizer.Decode(turn1));

        string generatedText = thinking ? "17*20=340, 17*3=51.</think>391" : "391";
        List<int> generated = tokenizer.Encode(generatedText, addSpecial: false);
        history.Add(new ChatMessage
        {
            Role = "assistant",
            Content = "391",
            Thinking = thinking ? "17*20=340, 17*3=51." : null,
            RawOutputTokens = generated,
        });
        history.Add(new ChatMessage { Role = "user", Content = "And 3+3?" });
        List<int> turn2 = renderer.RenderToTokens(tokenizer, template, history, Arch, true, enableThinking: thinking);

        List<int> cache = [.. turn1, .. generated];
        Assert.Equal(cache, turn2.Take(cache.Count));
        Assert.StartsWith("<|im_end|>\n<|im_start|>user\nAnd 3+3?", tokenizer.Decode(turn2.Skip(cache.Count).ToList()));
    }

    /// <summary>
    /// A tool round inside one turn: the model reasoned, then called. The round's raw tokens are
    /// spliced back (ToolCallRawSplicing.Always) and the tool result follows them, so the round
    /// after the result continues the live cache instead of re-prefilling the conversation.
    /// </summary>
    [Fact]
    public void AToolRound_ExtendsTheCallingRoundsCacheExactly()
    {
        var tokenizer = new CharacterTokenizer();
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        List<ToolFunction> tools = WeatherTool();
        List<ChatMessage> history = [new() { Role = "user", Content = "Weather in Paris?" }];
        List<int> round1 = renderer.RenderToTokens(tokenizer, TextTemplate.Value, history, Arch, true, tools, enableThinking: true);

        const string call = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
        List<int> generated = tokenizer.Encode("Use the tool.</think>" + call, addSpecial: false);
        history.Add(new ChatMessage
        {
            Role = "assistant",
            Content = "",
            Thinking = "Use the tool.",
            ToolCalls = [new ToolCall { Id = "call_1", Name = "get_weather", Arguments = new() { ["city"] = "Paris" } }],
            RawOutputTokens = generated,
        });
        history.Add(new ChatMessage { Role = "tool", ToolCallId = "call_1", Content = "20C" });
        List<int> round2 = renderer.RenderToTokens(tokenizer, TextTemplate.Value, history, Arch, true, tools, enableThinking: true);

        List<int> cache = [.. round1, .. generated];
        Assert.Equal(cache, round2.Take(cache.Count));
        Assert.Equal("<|im_end|>\n<|im_start|>tool\n<tool_response>\n20C\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>",
            tokenizer.Decode(round2.Skip(cache.Count).ToList()));
    }
}
