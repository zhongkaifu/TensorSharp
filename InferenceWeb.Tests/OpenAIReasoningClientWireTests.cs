// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Issue #386: a .NET client on the OpenAI SDK and Microsoft.Extensions.AI (MEAI) asked for
// reasoning with ChatOptions.Reasoning.Effort and never received TextReasoningContent.
// MEAI sends that option as `reasoning_effort` on Chat Completions and as
// `reasoning: {"effort": ...}` on the Responses API, and reads the reasoning back from
// `delta.reasoning_content` / `message.reasoning_content` (Chat Completions) and from the
// Responses API's reasoning items: `response.reasoning_summary_text.delta` events and the
// item's `summary` parts. These tests pin both halves without a model: what turns thinking
// on, and the fields and events the reasoning comes back in. A scripted qwen35 model
// reasons only when the prompt opened its think block, so reasoning in a reply proves the
// request turned thinking on.
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.ProtocolAdapters;
using TensorSharp.Server.RequestParsers;
using TensorSharp.Server.Responses;

namespace InferenceWeb.Tests;

public sealed class OpenAIReasoningClientWireTests : IDisposable
{
    private const string Reasoning = "Seventeen times twenty is 340, plus 51.";
    private const string Answer = "17 * 23 = 391.";

    private readonly string _root = Path.Combine(Path.GetTempPath(), "ts-reasoning-wire-" + Guid.NewGuid().ToString("N"));
    private readonly string _model;

    public OpenAIReasoningClientWireTests()
    {
        Directory.CreateDirectory(_root);
        _model = Path.Combine(_root, "probe.gguf");
        BackendFailureWarmupTests.WriteProbeGguf(_model);
    }

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch { /* best effort */ }
    }

    // ---------------------------------------------------------------- what turns thinking on

    [Theory]
    // A named level other than none asks to think, as MEAI's ChatOptions.Reasoning sends it.
    [InlineData("{\"reasoning_effort\":\"high\"}", true, true)]
    [InlineData("{\"reasoning_effort\":\"medium\"}", true, true)]
    [InlineData("{\"reasoning_effort\":\"low\"}", true, true)]
    [InlineData("{\"reasoning_effort\":\"minimal\"}", true, true)]
    [InlineData("{\"reasoning_effort\":\"xhigh\"}", true, true)]
    [InlineData("{\"reasoning_effort\":\" High \"}", true, true)]
    [InlineData("{\"reasoning\":{\"effort\":\"high\"}}", true, true)]
    // "none" is OpenAI's "do not reason"; absent, null or empty leave the default (off).
    [InlineData("{\"reasoning_effort\":\"none\"}", false, false)]
    [InlineData("{\"reasoning_effort\":null}", false, false)]
    [InlineData("{\"reasoning_effort\":\"\"}", false, false)]
    [InlineData("{}", false, false)]
    // An explicit switch decides when both are sent.
    [InlineData("{\"think\":false,\"reasoning_effort\":\"high\"}", false, false)]
    [InlineData("{\"chat_template_kwargs\":{\"enable_thinking\":false},\"reasoning_effort\":\"high\"}", false, false)]
    [InlineData("{\"think\":true,\"reasoning_effort\":\"none\"}", true, false)]
    [InlineData("{\"think\":true}", true, false)]
    public void ThinkingSwitch_ReadsANamedReasoningLevelWhenNoSwitchIsSent(string json, bool think, bool fromEffort)
    {
        using JsonDocument doc = JsonDocument.Parse(json);
        Assert.True(ThinkingSwitchParser.TryParse(doc.RootElement, out bool parsed, out bool parsedFromEffort, out _,
            out string? error), error);
        Assert.Equal(think, parsed);
        Assert.Equal(fromEffort, parsedFromEffort);
    }

    /// <summary>The reasoning-level part of the Qwen3.8-27B and Qwen3.8 Flash Next GGUF
    /// templates: with thinking on, xhigh (the default, and high) and low each put a paragraph
    /// at the head of the system block; medium puts none. (The Qwen3.8-9B distill of issue
    /// #386 ships the Qwen 3.5 template, which has no such paragraph.)</summary>
    private const string Qwen38EffortTemplate =
        "{%- set reasoning_instructions = '' %}"
        + "{%- if enable_thinking is undefined or enable_thinking is true %}"
        + "{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}"
        + "{%- if resolved_reasoning_effort == 'high' %}{%- set resolved_reasoning_effort = 'xhigh' %}{%- endif %}"
        + "{%- if resolved_reasoning_effort == 'xhigh' %}{%- set reasoning_instructions = 'Reasoning effort is set to xhigh. Please think carefully through the task.' %}"
        + "{%- elif resolved_reasoning_effort == 'low' %}{%- set reasoning_instructions = 'Reasoning effort is set to low. Keep your thinking brief.' %}{%- endif %}"
        + "{%- endif %}"
        + "{%- for message in messages %}"
        + "{%- if message.role == 'system' %}{{- '<|im_start|>system\n' + (reasoning_instructions + '\n\n' if reasoning_instructions else '') + message.content + '<|im_end|>\n' }}"
        + "{%- else %}{{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>\n' }}{%- endif %}"
        + "{%- endfor %}"
        + "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\n' }}"
        + "{%- if enable_thinking is defined and enable_thinking is false %}{{- '<think>\n\n</think>\n\n' }}{%- else %}{{- '<think>\n' }}{%- endif %}"
        + "{%- endif %}";

    /// <summary>
    /// A conversation's shared prefix (its system block) is measured at the request's own
    /// reasoning level. Measured without it, a "medium" or "low" request with thinking on was
    /// compared with the default xhigh paragraph and shared nothing past "&lt;|im_start|&gt;system".
    /// </summary>
    [Theory]
    [InlineData(null)]
    [InlineData("low")]
    [InlineData("medium")]
    [InlineData("high")]
    public void Qwen38_SharedPrefixIsMeasuredAtTheRequestsReasoningLevel(string? effort)
    {
        using var lifecycle = new ModelLifecycleService(NullLogger.Instance);
        using var host = new InferenceEngineHost(lifecycle, NullLogger.Instance);
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        var pipeline = new ChatGenerationPipeline(lifecycle, host, renderer, new InferenceTelemetry(NullLogger.Instance),
            NullLogger.Instance);
        using var model = new ThinkingScriptModel(_model, Qwen38EffortTemplate);
        var history = new List<ChatMessage>
        {
            new() { Role = "system", Content = "You are a helpful assistant. Answer briefly, and show the arithmetic you used to get there." },
            new() { Role = "user", Content = "What is 17 * 23?" },
        };

        List<int> prompt = renderer.RenderToTokens(model.Tokenizer, Qwen38EffortTemplate, history, "qwen35",
            addGenerationPrompt: true, enableThinking: true, reasoningEffort: effort);
        int systemBlock = renderer.RenderToTokens(model.Tokenizer, Qwen38EffortTemplate, history.GetRange(0, 1), "qwen35",
            addGenerationPrompt: false, enableThinking: true, reasoningEffort: effort).Count;
        Assert.True(systemBlock >= ChatGenerationPipeline.MinSharedPrefixTokens);

        Assert.Equal(systemBlock, pipeline.ComputeSharedPrefixTokens(model, history, prompt, "qwen35", [], true, effort));
        // Qwen3.8 Flash Next's template has the same paragraphs.
        Assert.True(ChatProtocolRegistry.For("qwen4exp")!.RendersReasoningEffort);
    }

    // ---------------------------------------------------------------- Chat Completions

    /// <summary>The request MEAI's OpenAIChatClient sends for ChatOptions.Reasoning.Effort =
    /// High: the level as reasoning_effort, and no thinking switch.</summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task ChatCompletions_Streaming_ReasoningEffortStreamsReasoningContent(bool loop)
    {
        string sse = await Chat("\"reasoning_effort\":\"high\"", stream: true, loop);
        (string content, string reasoning) = ChatDeltas(sse);

        Assert.Equal(Reasoning, reasoning.Trim());
        Assert.Equal(Answer, content.Trim());
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task ChatCompletions_NonStreaming_ReasoningEffortReturnsReasoningContent(bool loop)
    {
        using JsonDocument doc = JsonDocument.Parse(await Chat("\"reasoning_effort\":\"high\"", stream: false, loop));
        JsonElement message = doc.RootElement.GetProperty("choices")[0].GetProperty("message");

        Assert.Equal(Answer, message.GetProperty("content").GetString()!.Trim());
        Assert.Equal(Reasoning, message.GetProperty("reasoning_content").GetString()!.Trim());
        // The field this route returned it as before reasoning_content, kept for its readers.
        Assert.Equal(Reasoning, message.GetProperty("thinking").GetString()!.Trim());
    }

    [Theory]
    [InlineData("\"reasoning_effort\":\"none\"")]
    [InlineData("\"think\":false,\"reasoning_effort\":\"high\"")]
    [InlineData("\"max_tokens\":200")]
    public async Task ChatCompletions_WithoutThinking_TheModelAnswersWithoutReasoning(string fields)
    {
        string sse = await Chat(fields, stream: true, loop: false);
        (string content, string reasoning) = ChatDeltas(sse);

        Assert.Equal(Answer, content.Trim());
        Assert.Empty(reasoning);
    }

    [Theory]
    // A family whose JSON grammar applies from the first token (no ThinkingGrammarActivationTrigger):
    // thinking that only reasoning_effort asked for gives way, and the request is admitted
    // (404 is the hosted-model guard), as it was before reasoning_effort turned thinking on.
    [InlineData("\"reasoning_effort\":\"high\",\"response_format\":{\"type\":\"json_object\"}", 404)]
    // An explicit switch is still refused with response_format there.
    [InlineData("\"think\":true,\"response_format\":{\"type\":\"json_object\"}", 400)]
    public async Task ChatCompletions_ResponseFormat_OutranksOnlyAnImpliedThinkingSwitch(string fields, int status)
    {
        var context = new DefaultHttpContext();
        string request = "{\"model\":\"not-hosted.gguf\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"stream\":false," + fields + "}";
        context.Request.Body = new MemoryStream(Encoding.UTF8.GetBytes(request));
        context.Request.ContentType = "application/json";
        context.Response.Body = new MemoryStream();
        var options = ServerOptionsBuilder.Build(new[] { "--model", Path.Combine(_root, "hosted.gguf"), "--no-skills" }, _root);
        using (var service = new UnloadedService("qwen35"))
        {
            await new OpenAIChatAdapter(service, options, Uploads(), Registry(), null, Workspaces(),
                NullLoggerFactory.Instance).ChatCompletionsAsync(context);
        }
        Assert.Equal(status, context.Response.StatusCode);
    }

    private sealed class UnloadedService(string architecture) : ModelService { public override string Architecture => architecture; }

    // ---------------------------------------------------------------- Responses API

    /// <summary>
    /// MEAI's Responses client reads a non-streamed reasoning item's <c>summary</c> parts
    /// (ReasoningResponseItem.GetSummaryText), so the reasoning is one <c>summary_text</c>
    /// part of a <c>reasoning</c> item that precedes the message.
    /// </summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task Responses_NonStreaming_ReturnsAReasoningItemBeforeTheMessage(bool loop)
    {
        using JsonDocument doc = JsonDocument.Parse(await Responses("\"reasoning\":{\"effort\":\"high\"}", stream: false, loop));
        JsonElement[] output = doc.RootElement.GetProperty("output").EnumerateArray().ToArray();

        Assert.Equal(new[] { "reasoning", "message" }, output.Select(i => i.GetProperty("type").GetString()));
        AssertReasoningItem(output[0]);
        Assert.Equal(Answer, output[1].GetProperty("content")[0].GetProperty("text").GetString()!.Trim());
    }

    /// <summary>
    /// MEAI's streaming Responses client turns <c>response.reasoning_summary_text.delta</c>
    /// into TextReasoningContent. The reasoning item streams as OpenAI streams one: added,
    /// its summary part added, the text deltas, text and part done, the item done, and then
    /// the message.
    /// </summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task Responses_Streaming_StreamsTheReasoningAsSummaryTextEvents(bool loop)
    {
        string sse = await Responses("\"reasoning\":{\"effort\":\"high\"}", stream: true, loop);
        List<JsonElement> events = SseData(sse);

        // The event sequence, with each run of deltas read as one.
        var types = new List<string>();
        foreach (JsonElement e in events)
        {
            string type = e.GetProperty("type").GetString()!;
            if (types.Count == 0 || types[^1] != type || !type.EndsWith(".delta", StringComparison.Ordinal))
                types.Add(type);
        }
        Assert.Equal(new[]
        {
            "response.created",
            "response.output_item.added", "response.reasoning_summary_part.added", "response.reasoning_summary_text.delta",
            "response.reasoning_summary_text.done", "response.reasoning_summary_part.done", "response.output_item.done",
            "response.output_item.added", "response.content_part.added", "response.output_text.delta",
            "response.output_text.done", "response.content_part.done", "response.output_item.done",
            "response.completed",
        }, types);

        JsonElement added = events.First(e => e.GetProperty("type").GetString() == "response.output_item.added");
        Assert.Equal(0, added.GetProperty("output_index").GetInt32());
        Assert.Equal("reasoning", added.GetProperty("item").GetProperty("type").GetString());
        string itemId = added.GetProperty("item").GetProperty("id").GetString()!;
        Assert.StartsWith("rs_", itemId, StringComparison.Ordinal);

        var deltas = new StringBuilder();
        foreach (JsonElement e in events.Where(e => e.GetProperty("type").GetString() == "response.reasoning_summary_text.delta"))
        {
            Assert.Equal(itemId, e.GetProperty("item_id").GetString());
            Assert.Equal(0, e.GetProperty("output_index").GetInt32());
            Assert.Equal(0, e.GetProperty("summary_index").GetInt32());
            deltas.Append(e.GetProperty("delta").GetString());
        }
        Assert.Equal(Reasoning, deltas.ToString().Trim());
        JsonElement textDone = events.Single(e => e.GetProperty("type").GetString() == "response.reasoning_summary_text.done");
        Assert.Equal(deltas.ToString(), textDone.GetProperty("text").GetString());

        string answer = string.Concat(events.Where(e => e.GetProperty("type").GetString() == "response.output_text.delta")
            .Select(e => e.GetProperty("delta").GetString()));
        Assert.Equal(Answer, answer.Trim());
        Assert.All(events.Where(e => e.GetProperty("type").GetString() == "response.output_text.delta"),
            e => Assert.Equal(1, e.GetProperty("output_index").GetInt32()));

        JsonElement[] output = events[^1].GetProperty("response").GetProperty("output").EnumerateArray().ToArray();
        Assert.Equal(new[] { "reasoning", "message" }, output.Select(i => i.GetProperty("type").GetString()));
        AssertReasoningItem(output[0]);
        Assert.Equal(itemId, output[0].GetProperty("id").GetString());
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task Responses_EffortNone_TheModelAnswersWithoutReasoning(bool stream)
    {
        string body = await Responses("\"reasoning\":{\"effort\":\"none\"}", stream, loop: false);
        JsonElement response = stream
            ? SseData(body)[^1].GetProperty("response")
            : JsonDocument.Parse(body).RootElement;

        JsonElement[] output = response.GetProperty("output").EnumerateArray().ToArray();
        Assert.Equal(new[] { "message" }, output.Select(i => i.GetProperty("type").GetString()));
        Assert.Equal(Answer, output[0].GetProperty("content")[0].GetProperty("text").GetString()!.Trim());
        if (stream)
            Assert.DoesNotContain("response.reasoning_summary", body, StringComparison.Ordinal);
    }

    private static void AssertReasoningItem(JsonElement item)
    {
        Assert.Equal("completed", item.GetProperty("status").GetString());
        JsonElement part = Assert.Single(item.GetProperty("summary").EnumerateArray());
        Assert.Equal("summary_text", part.GetProperty("type").GetString());
        Assert.Equal(Reasoning, part.GetProperty("text").GetString()!.Trim());
    }

    /// <summary>
    /// MEAI sends a reasoning item back before the assistant message it led to (its
    /// TextReasoningContent becomes a ReasoningResponseItem with a summary). That is the
    /// message's reasoning, as reasoning_content is on a Chat Completions history message.
    /// </summary>
    [Fact]
    public void ResponsesInput_AReasoningItemIsTheNextAssistantMessagesReasoning()
    {
        const string input = """
        [
          {"type":"message","role":"user","content":[{"type":"input_text","text":"What is 17 * 23?"}]},
          {"type":"reasoning","id":"rs_1","summary":[{"type":"summary_text","text":"Seventeen times "},{"type":"summary_text","text":"twenty-three."}]},
          {"type":"message","role":"assistant","content":[{"type":"output_text","text":"391."}]},
          {"type":"reasoning","id":"rs_2","summary":[],"content":[{"type":"reasoning_text","text":"Raw reasoning."}]},
          {"type":"message","role":"assistant","content":"Again 391."},
          {"type":"message","role":"user","content":"Thanks."}
        ]
        """;
        using JsonDocument doc = JsonDocument.Parse(input);
        List<ChatMessage> messages = ChatMessageParser.ParseResponsesInput(doc.RootElement, null, Uploads());

        Assert.Equal(new[] { "user", "assistant", "assistant", "user" }, messages.Select(m => m.Role));
        Assert.Equal("Seventeen times twenty-three.", messages[1].Thinking);
        Assert.Equal("391.", messages[1].Content);
        Assert.Equal("Raw reasoning.", messages[2].Thinking);
        Assert.Null(messages[0].Thinking);
        Assert.Null(messages[3].Thinking);
    }

    [Fact]
    public void ResponsesInput_AReasoningItemWithoutAnAssistantMessageAttachesToNothing()
    {
        const string input = """
        [
          {"type":"message","role":"user","content":"Read it."},
          {"type":"reasoning","summary":[{"type":"summary_text","text":"Call the tool."}]},
          {"type":"function_call","call_id":"call_1","name":"read","arguments":"{}"},
          {"type":"message","role":"assistant","content":"Done."},
          {"type":"reasoning","summary":[{"type":"summary_text","text":"Trailing."}]}
        ]
        """;
        using JsonDocument doc = JsonDocument.Parse(input);
        List<ChatMessage> messages = ChatMessageParser.ParseResponsesInput(doc.RootElement, null, Uploads());

        Assert.Equal(new[] { "user", "assistant" }, messages.Select(m => m.Role));
        Assert.All(messages, m => Assert.Null(m.Thinking));
    }

    // ---------------------------------------------------------------- surfaces

    private Task<string> Chat(string fields, bool stream, bool loop)
    {
        string request = "{\"model\":\"probe.gguf\",\"messages\":[{\"role\":\"system\",\"content\":\"You are a helpful assistant.\"},"
            + "{\"role\":\"user\",\"content\":\"What is 17 * 23?\"}],\"stream\":" + (stream ? "true" : "false")
            + (stream ? ",\"stream_options\":{\"include_usage\":true}" : "") + LoopField(loop) + "," + fields + "}";
        return Invoke(request, (svc, options, ctx) => new OpenAIChatAdapter(svc, options, Uploads(), Registry(), null,
            Workspaces(), NullLoggerFactory.Instance).ChatCompletionsAsync(ctx));
    }

    private Task<string> Responses(string fields, bool stream, bool loop)
    {
        string request = "{\"model\":\"probe.gguf\",\"instructions\":\"You are a helpful assistant.\","
            + "\"input\":[{\"type\":\"message\",\"role\":\"user\",\"content\":[{\"type\":\"input_text\",\"text\":\"What is 17 * 23?\"}]}],"
            + "\"stream\":" + (stream ? "true" : "false") + LoopField(loop) + "," + fields + "}";
        return Invoke(request, async (svc, options, ctx) =>
        {
            using var store = new InMemoryResponsesStore();
            await new OpenAIResponsesAdapter(svc, options, Uploads(), Registry(), null, Workspaces(),
                NullLoggerFactory.Instance, store).CreateResponseAsync(ctx);
        });
    }

    /// <summary>The request field that keeps an API request off the sub-agent loop.</summary>
    private static string LoopField(bool loop) => loop ? string.Empty : ",\"multi_agent\":false";

    private static (string Content, string Reasoning) ChatDeltas(string sse)
    {
        var content = new StringBuilder();
        var reasoning = new StringBuilder();
        foreach (JsonElement data in SseData(sse))
        {
            if (!data.TryGetProperty("choices", out JsonElement choices) || choices.GetArrayLength() == 0)
                continue;
            JsonElement delta = choices[0].GetProperty("delta");
            if (delta.TryGetProperty("content", out JsonElement c) && c.ValueKind == JsonValueKind.String)
                content.Append(c.GetString());
            if (delta.TryGetProperty("reasoning_content", out JsonElement r) && r.ValueKind == JsonValueKind.String)
                reasoning.Append(r.GetString());
        }
        return (content.ToString(), reasoning.ToString());
    }

    private static List<JsonElement> SseData(string sse)
    {
        var data = new List<JsonElement>();
        foreach (string line in sse.Split('\n'))
        {
            if (!line.StartsWith("data: ", StringComparison.Ordinal) || line == "data: [DONE]")
                continue;
            using JsonDocument doc = JsonDocument.Parse(line.Substring(6));
            data.Add(doc.RootElement.Clone());
        }
        return data;
    }

    private async Task<string> Invoke(string request, Func<ModelService, ServerHostingOptions, HttpContext, Task> call)
    {
        var context = new DefaultHttpContext();
        context.Request.Body = new MemoryStream(Encoding.UTF8.GetBytes(request));
        context.Request.ContentType = "application/json";
        context.Response.Body = new MemoryStream();
        ServerHostingOptions options = ServerOptionsBuilder.Build(
            new[] { "--model", _model, "--backend", "cpu", "--max-tokens", "400", "--no-skills" }, _root);
        using (ModelService service = Service())
        {
            service.LoadModel(_model, null, options.DefaultBackend);
            await call(service, options, context);
        }
        context.Response.Body.Position = 0;
        return await new StreamReader(context.Response.Body).ReadToEndAsync();
    }

    private static ModelService Service()
    {
        var service = new ModelService(NullLogger<ModelService>.Instance, (path, _, _, _) => new ThinkingScriptModel(path));
        service.EngineHost.SchedulerConfigOverride = new SchedulerConfig
        {
            BlockSize = 16, NumBlocks = 2_048, MaxNumRunningSequences = 2,
            MaxNumBatchedTokens = 4_096, MaxPrefillChunkSize = 4_096, SoloPrefillChunkSize = 4_096,
            EnablePrefixCaching = false, Speculation = SpeculationOptions.Disabled,
        };
        return service;
    }

    private static SkillRegistry Registry() => new(new SkillRegistryOptions { Roots = Array.Empty<string>() });

    private UploadStoragePolicy Uploads() => new(Path.Combine(_root, "uploads"));

    private SessionWorkspaceManager Workspaces() => new(Path.Combine(_root, "workspaces"));

    // ---------------------------------------------------------------- the model

    /// <summary>
    /// A qwen35 model without a GGUF template (the family's own renderer renders the prompt)
    /// that writes one character per token after the generation prompt. When the prompt
    /// opened the think block (<c>&lt;think&gt;\n</c>) it reasons, closes the block and
    /// answers; when the prompt closed it (<c>&lt;think&gt;\n\n&lt;/think&gt;\n\n</c>) it
    /// only answers. So the reply says which prompt the request rendered.
    /// </summary>
    private sealed class ThinkingScriptModel : ModelBase
    {
        private const string AssistantMarker = "<|im_start|>assistant\n";
        private const string ThinkingOn = "<think>\n";
        private const string ThinkingOff = "<think>\n\n</think>\n\n";
        private static readonly string Thought = Reasoning + "\n</think>\n\n" + Answer;
        private readonly StringBuilder _seen = new();

        public ThinkingScriptModel(string path, string? template = null) : base(path, BackendType.Cpu)
        {
            Config = new ModelConfig { Architecture = "qwen35", VocabSize = 128, NumLayers = 1, ChatTemplate = template };
            Tokenizer = new CharTokenizer();
            _maxContextLength = 32_768;
        }

        public override bool SupportsKVStateSnapshot => true;
        public override bool SupportsCrossSequenceKvReuse => false;
        public override string KVStateFingerprint => "reasoning-client-wire";
        public override long ComputeKVBlockByteSize(int tokenCount) => 4L * tokenCount;
        public override bool TryExtractKVBlock(int start, int count, Span<byte> destination) => true;
        public override bool TryInjectKVBlock(int start, int count, ReadOnlySpan<byte> source) => true;

        protected override float[] ForwardCore(int[] tokens)
        {
            foreach (int token in tokens)
                _seen.Append((char)token);
            var logits = new float[128];
            logits[Next()] = 10;
            return logits;
        }

        private int Next()
        {
            string seen = _seen.ToString();
            int start = seen.LastIndexOf(AssistantMarker, StringComparison.Ordinal);
            if (start < 0)
                return 1;
            start += AssistantMarker.Length;
            string script;
            if (seen.AsSpan(start).StartsWith(ThinkingOff))
            {
                start += ThinkingOff.Length;
                script = Answer;
            }
            else if (seen.AsSpan(start).StartsWith(ThinkingOn))
            {
                start += ThinkingOn.Length;
                script = Thought;
            }
            else
                return 1;
            int at = seen.Length - start;
            return at < script.Length ? script[at] : 1;
        }

        protected override void ResetKVCacheCore() => _seen.Clear();
    }

    /// <summary>One token per character; anything outside ASCII is read as '?', which no
    /// script and no marker uses.</summary>
    private sealed class CharTokenizer : ITokenizer
    {
        public string[] Vocab => Enumerable.Range(0, 128).Select(i => ((char)i).ToString()).ToArray();
        public int VocabSize => 128;
        public int BosTokenId => -1;
        public int[] EosTokenIds => new[] { 1 };
        public bool IsEos(int id) => id == 1;
        public int LookupToken(string token) => -1;
        public List<int> Encode(string text, bool addSpecial = true) => text.Select(c => c < 128 ? (int)c : '?').ToList();
        public string Decode(List<int> tokens) => new(tokens.Select(id => (char)id).ToArray());
        public void AppendTokenBytes(int token, List<byte> bytes) => bytes.Add((byte)token);
    }
}
