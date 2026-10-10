// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// The thinking-off stray reasoning of Nemotron-H Reasoning-128K through every surface that
// serves it, end to end: a scripted nemotron_h model whose GGUF carries the Reasoning-128K
// template writes a logged reply one character per token, the real pipeline renders the
// prompt and announces its tail, and each surface (the OpenAI chat and Responses APIs, the
// Ollama API, and the Web UI with and without skills) is read the way its clients read it.
// The parser tests prove the parser; these prove each surface asks for it, primes it and
// applies what it says, so none of that wiring can be dropped without a failure here.
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.ProtocolAdapters;
using TensorSharp.Server.Responses;

namespace InferenceWeb.Tests;

public sealed class NemotronHThinkOffServingTests : IDisposable
{
    // tokenizer.chat_template of nvidia_Nemotron-H-8B-Reasoning-128K, as far as detection reads it.
    private const string ReasoningTemplate =
        "{{ '<SPECIAL_10>System\n' }}{% for message in messages %}{{ '\n<SPECIAL_11>Assistant\n' }}{% endfor %}";

    private const string Answer =
        "The file 'notes.txt' has **7 lines**, as previously determined. No need to read it again since it hasn't changed.";

    /// <summary>A reply that asks no tool and has nothing to close, for the structured cases:
    /// a JSON value that quotes the block's tags and an empty call list.</summary>
    private const string TagsJson = "{\"tags\": [\"<think>\", \"</think>\", \"<TOOLCALL>[]</TOOLCALL>\"], \"count\": 3}";

    private readonly string _root = Path.Combine(Path.GetTempPath(), "ts-thinkoff-serving-" + Guid.NewGuid().ToString("N"));
    private readonly string _model;
    private readonly string _skills;

    public NemotronHThinkOffServingTests()
    {
        Directory.CreateDirectory(_root);
        _model = Path.Combine(_root, "probe.gguf");
        BackendFailureWarmupTests.WriteProbeGguf(_model);
        _skills = Path.Combine(_root, "skills");
        string notes = Path.Combine(_skills, "notes");
        Directory.CreateDirectory(notes);
        File.WriteAllText(Path.Combine(notes, "SKILL.md"), "---\nname: notes\ndescription: reads notes\n---\n\nRead them.\n");
    }

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch { /* best effort */ }
    }

    // ---------------------------------------------------------------- stray reasoning, every surface

    // Every API surface twice: as a request is served by default, through the sub-agent
    // root loop, which parses the reply and hands the adapter separated pieces; and with
    // "multi_agent": false, where the adapter's own parser reads the raw stream.

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task OpenAIChat_Streaming_SendsOnlyTheAnswerAsContent(bool loop)
    {
        string sse = await OpenAIChat(NemotronHLoggedReplies.ToolRoundWithAnswer, stream: true, loop);
        (string content, string reasoning) = OpenAIDeltas(sse);

        Assert.Equal(Answer, content.Trim());
        Assert.DoesNotContain("</think>", content);
        Assert.StartsWith("Okay, the user", reasoning);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task OpenAIChat_NonStreaming_ReturnsOnlyTheAnswer(bool loop)
    {
        using JsonDocument doc = JsonDocument.Parse(await OpenAIChat(NemotronHLoggedReplies.ToolRoundWithAnswer, stream: false, loop));
        JsonElement message = doc.RootElement.GetProperty("choices")[0].GetProperty("message");

        Assert.Equal(Answer, message.GetProperty("content").GetString()!.Trim());
    }

    [Theory]
    [InlineData(true, true)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(false, false)]
    public async Task Ollama_ReturnsOnlyTheAnswer(bool stream, bool loop)
    {
        string request = "{\"model\":\"probe.gguf\",\"messages\":[{\"role\":\"user\",\"content\":\"How many lines?\"}],"
            + "\"think\":false,\"stream\":" + (stream ? "true" : "false") + ",\"options\":{\"num_predict\":900}" + LoopField(loop) + "}";
        string body = await Invoke(Service(NemotronHLoggedReplies.ToolRoundWithAnswer), request, Options(skills: false),
            (svc, options, ctx) => new OllamaAdapter(svc, options, Uploads(), Registry(false), null, Workspaces(),
                NullLoggerFactory.Instance).ChatAsync(ctx));

        var content = new StringBuilder();
        foreach (string line in body.Split('\n', StringSplitOptions.RemoveEmptyEntries))
        {
            using JsonDocument doc = JsonDocument.Parse(line);
            if (doc.RootElement.TryGetProperty("message", out JsonElement message)
                && message.TryGetProperty("content", out JsonElement piece))
                content.Append(piece.GetString());
        }
        Assert.Equal(Answer, content.ToString().Trim());
    }

    [Theory]
    [InlineData(true, true)]
    [InlineData(true, false)]
    [InlineData(false, true)]
    [InlineData(false, false)]
    public async Task Responses_ReturnsOnlyTheAnswer(bool stream, bool loop)
    {
        string request = "{\"model\":\"probe.gguf\",\"input\":\"How many lines?\",\"max_output_tokens\":900,\"stream\":"
            + (stream ? "true" : "false") + LoopField(loop) + "}";
        string body = await Invoke(Service(NemotronHLoggedReplies.ToolRoundWithAnswer), request, Options(skills: false),
            async (svc, options, ctx) =>
            {
                using var store = new InMemoryResponsesStore();
                await new OpenAIResponsesAdapter(svc, options, Uploads(), Registry(false), null, Workspaces(),
                    NullLoggerFactory.Instance, store).CreateResponseAsync(ctx);
            });

        var text = new StringBuilder();
        if (stream)
        {
            foreach (JsonElement data in SseData(body))
                if (data.TryGetProperty("type", out JsonElement type) && type.GetString() == "response.output_text.delta")
                    text.Append(data.GetProperty("delta").GetString());
        }
        else
        {
            using JsonDocument doc = JsonDocument.Parse(body);
            foreach (JsonElement item in doc.RootElement.GetProperty("output").EnumerateArray())
                if (item.TryGetProperty("content", out JsonElement parts))
                    foreach (JsonElement part in parts.EnumerateArray())
                        text.Append(part.GetProperty("text").GetString());
        }
        Assert.Equal(Answer, text.ToString().Trim());
    }

    /// <summary>The request field that keeps an API request off the sub-agent loop.</summary>
    private static string LoopField(bool loop) => loop ? string.Empty : ",\"multi_agent\":false";

    /// <summary>
    /// The page can take text back, so it is shown the undecided reply at once and then,
    /// at the close: the reasoning, a <c>replace</c> with what is left of the answer (here
    /// nothing), and the answer. Without the parser it would read the reasoning and the tag
    /// as the answer; with a parser that cannot retract it would see nothing until the close.
    /// Through each path a page turn takes: the page's own parser (no skills, no
    /// sub-agents), the skills loop, and the sub-agent root, which is the default.
    /// </summary>
    [Theory]
    [InlineData(WebUiPath.Plain)]
    [InlineData(WebUiPath.Skills)]
    [InlineData(WebUiPath.Agents)]
    public async Task WebUi_ShowsTheReplyAtOnce_ThenTakesTheReasoningBack(WebUiPath path)
    {
        List<Frame> frames = await WebUi(NemotronHLoggedReplies.ToolRoundWithAnswer, path, think: false);

        AssertShownThenRetracted(frames);
    }

    public enum WebUiPath { Plain, Skills, Agents }

    /// <summary>
    /// The page's rescue for a turn that reasoned past its budget runs it again with
    /// thinking off, which is exactly when this template reasons past its closed block. The
    /// retry goes through the same loop as the turn, and takes text back like it. (On the
    /// page's own parser the rescue never runs today: that path counts every streamed
    /// piece, reasoning included, as output, so a turn that reasoned has "produced
    /// something" and is not retried.)
    /// </summary>
    [Fact]
    public async Task WebUi_TheThinkingOffRetry_AlsoTakesTheReasoningBack()
    {
        // Thinking on: the reasoning runs past three quarters of the 800-token allowance, so
        // the budget stops it with no answer and the page retries with thinking off, where
        // the whole reply fits.
        // Varied words: a repeated sentence would be ended by the repetition guard instead.
        string[] words = "the file has lines and each one counts so I check them again before I answer with a number".Split(' ');
        var rng = new Random(7);
        var reasoningText = new StringBuilder("Okay, the user wants the count.");
        while (reasoningText.Length < 640)
            reasoningText.Append(' ').Append(words[rng.Next(words.Length)]);
        string reasoning = reasoningText.Append('.').ToString();
        string script = reasoning.TrimEnd() + "\n</think>\n\n" + Answer + "\n";
        Assert.InRange(reasoning.Length, 601, 800 - (script.Length - reasoning.Length) - 1);

        List<Frame> frames = await WebUi(script, WebUiPath.Agents, think: true, maxTokens: 800);

        int retry = frames.FindIndex(f => f.Thinking?.Contains("answering directly instead", StringComparison.Ordinal) == true);
        Assert.True(retry >= 0, "the turn was not retried with thinking off: " + frames[^1].Json);
        AssertShownThenRetracted(frames.Skip(retry + 1).ToList());
    }

    private static void AssertShownThenRetracted(List<Frame> frames)
    {
        int replace = frames.FindIndex(f => f.Replace != null);
        int thinking = frames.FindIndex(f => f.Thinking?.StartsWith("Okay, the user", StringComparison.Ordinal) == true);
        Assert.True(replace > 0, "nothing was taken back");
        string shownBeforeTheClose = string.Concat(frames.Take(replace).Select(f => f.Token));
        Assert.StartsWith("Okay, the user", shownBeforeTheClose);
        Assert.True(thinking >= 0 && thinking < replace, "the reasoning did not arrive before the replace");
        Assert.Equal(string.Empty, frames[replace].Replace!.Trim());

        string shown = string.Empty;
        foreach (Frame frame in frames)
        {
            if (frame.Replace != null) shown = frame.Replace;
            if (frame.Token != null) shown += frame.Token;
        }
        Assert.Equal(Answer, shown.Trim());
    }

    // ---------------------------------------------------------------- structured output

    /// <summary>
    /// With thinking off, a <c>response_format</c> grammar shapes the reply from its first
    /// token, so it is the JSON and cannot be stray reasoning. Primed for it anyway, the
    /// parser took the <c>&lt;/think&gt;</c> inside a string value for a close and split
    /// the object in two: json_schema answered 422, json_object streamed its broken tail.
    /// No parser runs over it at all, as before the template needed one: unprimed, it
    /// would still drop the empty call list from the third string.
    /// </summary>
    [Theory]
    [InlineData("json_schema", true)]
    [InlineData("json_schema", false)]
    [InlineData("json_object", true)]
    [InlineData("json_object", false)]
    public async Task ResponseFormat_JsonThatQuotesTheTags_ArrivesWhole(string kind, bool stream)
    {
        string format = kind == "json_object"
            ? "{\"type\":\"json_object\"}"
            : "{\"type\":\"json_schema\",\"json_schema\":{\"name\":\"tags\",\"strict\":true,\"schema\":{\"type\":\"object\","
              + "\"properties\":{\"tags\":{\"type\":\"array\",\"items\":{\"type\":\"string\"}},\"count\":{\"type\":\"integer\"}},"
              + "\"required\":[\"tags\",\"count\"],\"additionalProperties\":false}}}";
        string body = await OpenAIChat(TagsJson, stream, loop: true, "\"response_format\":" + format);

        string json;
        if (stream)
        {
            Assert.DoesNotContain("invalid_response_error", body);
            json = OpenAIDeltas(body).Content;
        }
        else
        {
            using JsonDocument doc = JsonDocument.Parse(body);
            json = doc.RootElement.GetProperty("choices")[0].GetProperty("message").GetProperty("content").GetString()!;
        }
        using JsonDocument parsed = JsonDocument.Parse(json);
        Assert.Equal(new[] { "<think>", "</think>", "<TOOLCALL>[]</TOOLCALL>" },
            parsed.RootElement.GetProperty("tags").EnumerateArray().Select(t => t.GetString()));
        Assert.Equal(3, parsed.RootElement.GetProperty("count").GetInt32());
    }

    [Fact]
    public void AReplyConstrainedFromItsFirstToken_IsNeverPrimedForStrayReasoning()
    {
        var unconstrained = new SamplingConfig();
        var allowList = new SamplingConfig { FirstTokenAllowList = new[] { (int)'{' } };
        Assert.False(OutputParserFactory.ConstrainsReplyStart(unconstrained));
        Assert.True(OutputParserFactory.ConstrainsReplyStart(allowList));

        Assert.True(ChatGenerationPipeline.AnnouncesGenerationSuffix("nemotron_h", ReasoningTemplate, "<think></think>", unconstrained));
        Assert.False(ChatGenerationPipeline.AnnouncesGenerationSuffix("nemotron_h", ReasoningTemplate, "<think></think>", allowList));
        Assert.True(OutputParserFactory.IsAlwaysRequired("nemotron_h", ReasoningTemplate, unconstrained));
        Assert.False(OutputParserFactory.IsAlwaysRequired("nemotron_h", ReasoningTemplate, allowList));
        // A family whose parser is always required keeps it, constrained or not.
        Assert.True(OutputParserFactory.IsAlwaysRequired("gemma4", null, allowList));
        // Gemma 4's open thought channel is announced whatever the sampler does.
        Assert.True(ChatGenerationPipeline.AnnouncesGenerationSuffix("gemma4", null, "<|channel>thought\n", allowList));
    }

    /// <summary>
    /// The pipeline makes the same call for every consumer: it announces the closed block
    /// to the parsers of an unconstrained reply, and not to those of a reply the sampler
    /// shapes from its first token, while the turn's record keeps the prompt's tail either
    /// way.
    /// </summary>
    [Theory]
    [InlineData("none")]
    [InlineData("grammar")]
    [InlineData("allow-list")]
    public async Task ThePipeline_AnnouncesTheClosedBlock_OnlyForAnUnconstrainedReply(string constraint)
    {
        using ModelService service = Service(TagsJson);
        service.LoadModel(_model, null, "cpu");
        ITokenizer tokenizer = service.Model.Tokenizer;
        SamplingConfig sampling = SamplingConfig.Greedy.Clone();
        if (constraint == "grammar")
            sampling.Grammar = TensorSharp.Runtime.Grammar.GrammarLibrary.NewConstraint(
                TensorSharp.Runtime.Grammar.GrammarLibrary.ForJsonObject(tokenizer), tokenizer);
        else if (constraint == "allow-list")
            sampling.FirstTokenAllowList = new[] { (int)'{' };

        var announced = new List<string>();
        var text = new StringBuilder();
        string? recorded = null;
        await foreach (ChatStreamUpdate update in service.ChatStreamWithSkillsAsync(
                           new List<ChatMessage> { new() { Role = "user", Content = "Which tags?" } }, 200,
                           CancellationToken.None, sampling, tools: null, enableThinking: false, skills: null))
        {
            if (update.Done)
                recorded = update.RawGenerationSuffix;
            else if (update.RawGenerationSuffix != null)
                announced.Add(update.RawGenerationSuffix);
            else
                text.Append(update.Piece);
        }

        Assert.Equal(TagsJson, text.ToString());
        Assert.Equal("<think></think>", recorded);
        Assert.Equal(constraint == "none" ? new[] { "<think></think>" } : Array.Empty<string>(), announced);
    }

    // ---------------------------------------------------------------- keep-alives while text is held

    /// <summary>
    /// An append-only stream holds an undecided reply until it is decided, and sends
    /// nothing while it does. At a few tokens a second that passed a proxy's idle timeout
    /// (nginx: 60 s) and the client got a 504. While text is held the stream now sends what
    /// its clients skip: an SSE comment, or an Ollama chunk with an empty message. Here a
    /// reply decoded at about 3 ms a token is held for about 1.5 s against a 60 ms interval.
    /// Through the adapters' own parsers and through the sub-agent root loop, which holds
    /// the text itself and hands the adapter an empty update for each piece it holds.
    /// </summary>
    [Theory]
    [InlineData("openai", false)]
    [InlineData("openai", true)]
    [InlineData("responses", false)]
    [InlineData("responses", true)]
    [InlineData("ollama", false)]
    [InlineData("ollama", true)]
    public async Task AHeldStream_SendsKeepAlivesUntilTheAnswer(string surface, bool loop)
    {
        TimeSpan interval = TimeSpan.FromMilliseconds(60);
        ModelService service = Service(NemotronHLoggedReplies.ToolRoundWithAnswer, msPerToken: 3);
        string selection = LoopField(loop);
        var lines = new List<string>();
        var content = new StringBuilder();
        int keepAlivesBeforeTheAnswer = 0;
        if (surface == "ollama")
        {
            string request = "{\"model\":\"probe.gguf\",\"messages\":[{\"role\":\"user\",\"content\":\"How many lines?\"}],"
                + "\"think\":false,\"stream\":true,\"options\":{\"num_predict\":900}" + selection + "}";
            string body = await Invoke(service, request, Options(skills: false),
                (svc, options, ctx) => new OllamaAdapter(svc, options, Uploads(), Registry(false), null, Workspaces(),
                    NullLoggerFactory.Instance) { KeepAliveInterval = interval }.ChatAsync(ctx));
            foreach (string line in body.Split('\n', StringSplitOptions.RemoveEmptyEntries))
            {
                using JsonDocument doc = JsonDocument.Parse(line);
                JsonElement root = doc.RootElement;
                string piece = root.TryGetProperty("message", out JsonElement m) && m.TryGetProperty("content", out JsonElement c)
                    ? c.GetString() ?? "" : "";
                bool empty = piece.Length == 0 && !root.GetProperty("done").GetBoolean()
                    && !(m.ValueKind == JsonValueKind.Object && m.TryGetProperty("thinking", out _));
                if (empty && content.Length == 0)
                    keepAlivesBeforeTheAnswer++;
                content.Append(piece);
            }
        }
        else
        {
            string request = surface == "openai"
                ? "{\"model\":\"probe.gguf\",\"messages\":[{\"role\":\"user\",\"content\":\"How many lines?\"}],\"max_tokens\":900,\"stream\":true" + selection + "}"
                : "{\"model\":\"probe.gguf\",\"input\":\"How many lines?\",\"max_output_tokens\":900,\"stream\":true" + selection + "}";
            string body = await Invoke(service, request, Options(skills: false), surface == "openai"
                ? (svc, options, ctx) => new OpenAIChatAdapter(svc, options, Uploads(), Registry(false), null, Workspaces(),
                    NullLoggerFactory.Instance) { KeepAliveInterval = interval }.ChatCompletionsAsync(ctx)
                : async (svc, options, ctx) =>
                {
                    using var store = new InMemoryResponsesStore();
                    await new OpenAIResponsesAdapter(svc, options, Uploads(), Registry(false), null, Workspaces(),
                        NullLoggerFactory.Instance, store) { KeepAliveInterval = interval }.CreateResponseAsync(ctx);
                });
            foreach (string evt in body.Split("\n\n", StringSplitOptions.RemoveEmptyEntries))
            {
                if (evt == ": keep-alive")
                {
                    if (content.Length == 0)
                        keepAlivesBeforeTheAnswer++;
                    continue;
                }
                string? data = evt.Split('\n').FirstOrDefault(l => l.StartsWith("data: ", StringComparison.Ordinal))?.Substring(6);
                if (data == null || data == "[DONE]")
                    continue;
                using JsonDocument doc = JsonDocument.Parse(data);
                JsonElement root = doc.RootElement;
                if (surface == "openai" && root.TryGetProperty("choices", out JsonElement choices) && choices.GetArrayLength() > 0
                    && choices[0].GetProperty("delta").TryGetProperty("content", out JsonElement c) && c.ValueKind == JsonValueKind.String)
                    content.Append(c.GetString());
                if (surface == "responses" && root.TryGetProperty("type", out JsonElement type)
                    && type.GetString() == "response.output_text.delta")
                    content.Append(root.GetProperty("delta").GetString());
            }
        }

        Assert.Equal(Answer, content.ToString().Trim());
        Assert.True(keepAlivesBeforeTheAnswer >= 3, $"{keepAlivesBeforeTheAnswer} keep-alives before the answer");
    }

    // ---------------------------------------------------------------- surfaces

    private Task<string> OpenAIChat(string script, bool stream, bool loop, string extra = "")
    {
        string request = "{\"model\":\"probe.gguf\",\"messages\":[{\"role\":\"user\",\"content\":\"How many lines?\"}],"
            + "\"max_tokens\":900,\"stream\":" + (stream ? "true" : "false") + LoopField(loop)
            + (extra.Length == 0 ? "" : "," + extra) + "}";
        return Invoke(Service(script), request, Options(skills: false),
            (svc, options, ctx) => new OpenAIChatAdapter(svc, options, Uploads(), Registry(false), null, Workspaces(),
                NullLoggerFactory.Instance).ChatCompletionsAsync(ctx));
    }

    private async Task<List<Frame>> WebUi(string script, WebUiPath path, bool think, int maxTokens = 900)
    {
        using ModelService service = Service(script);
        service.LoadModel(_model, null, "cpu");
        var sessions = new SessionManager();
        ChatSession session = sessions.CreateSession();
        bool skills = path == WebUiPath.Skills;
        var chat = new WebUiChatService(service, sessions, Options(skills), Uploads(), Registry(skills),
            codeRunner: null, Workspaces(), codeArtifacts: null, NullLoggerFactory.Instance);
        var request = new Dictionary<string, object>
        {
            ["sessionId"] = session.Id,
            ["messages"] = new[] { new { role = "user", content = "How many lines does notes.txt have?" } },
            ["think"] = think,
            ["maxTokens"] = maxTokens,
            ["skills"] = skills ? new[] { "notes" } : Array.Empty<string>(),
        };
        if (path != WebUiPath.Agents)
            request["multi_agent"] = false;
        JsonElement body = JsonSerializer.Deserialize<JsonElement>(JsonSerializer.Serialize(request));

        var frames = new List<Frame>();
        await foreach (object frame in chat.ChatStreamAsync(body, CancellationToken.None))
            frames.Add(Frame.Of(frame));
        Assert.DoesNotContain(frames, f => f.Error != null);
        return frames;
    }

    private sealed record Frame(string? Token, string? Thinking, string? Replace, string? Error, string Json)
    {
        public static Frame Of(object frame)
        {
            string json = JsonSerializer.Serialize(frame);
            using JsonDocument doc = JsonDocument.Parse(json);
            JsonElement root = doc.RootElement;
            return new Frame(Text(root, "token"), Text(root, "thinking"), Text(root, "replace"), Text(root, "error"), json);
        }

        private static string? Text(JsonElement root, string name)
            => root.TryGetProperty(name, out JsonElement value) && value.ValueKind == JsonValueKind.String ? value.GetString() : null;
    }

    private static (string Content, string Reasoning) OpenAIDeltas(string sse)
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

    private async Task<string> Invoke(ModelService service, string request, ServerHostingOptions options,
        Func<ModelService, ServerHostingOptions, HttpContext, Task> call)
    {
        var context = new DefaultHttpContext();
        context.Request.Body = new MemoryStream(Encoding.UTF8.GetBytes(request));
        context.Request.ContentType = "application/json";
        context.Response.Body = new MemoryStream();
        using (service)
        {
            // Loaded first, as the host loads it at startup: a request's plan is made from
            // the loaded model's family before the guard would load it.
            service.LoadModel(_model, null, options.DefaultBackend);
            await call(service, options, context);
        }
        context.Response.Body.Position = 0;
        return await new StreamReader(context.Response.Body).ReadToEndAsync();
    }

    private ModelService Service(string script, int msPerToken = 0)
    {
        var service = new ModelService(NullLogger<ModelService>.Instance,
            (path, _, _, _) => new ScriptModel(path, ReasoningTemplate, script, msPerToken));
        service.EngineHost.SchedulerConfigOverride = new SchedulerConfig
        {
            BlockSize = 16, NumBlocks = 2_048, MaxNumRunningSequences = 2,
            MaxNumBatchedTokens = 4_096, MaxPrefillChunkSize = 4_096, SoloPrefillChunkSize = 4_096,
            EnablePrefixCaching = false, Speculation = SpeculationOptions.Disabled,
        };
        return service;
    }

    private ServerHostingOptions Options(bool skills)
    {
        var args = new List<string> { "--model", _model, "--backend", "cpu", "--max-tokens", "900" };
        if (skills) args.AddRange(new[] { "--skills-dir", _skills });
        else args.Add("--no-skills");
        return ServerOptionsBuilder.Build(args.ToArray(), _root);
    }

    private SkillRegistry Registry(bool skills)
        => new(new SkillRegistryOptions { Roots = skills ? new[] { _skills } : Array.Empty<string>() });

    private UploadStoragePolicy Uploads() => new(Path.Combine(_root, "uploads"));

    private SessionWorkspaceManager Workspaces() => new(Path.Combine(_root, "workspaces"));

    // ---------------------------------------------------------------- the model

    /// <summary>
    /// A nemotron_h model with the Reasoning-128K template that writes <c>script</c> one
    /// character per token after the generation prompt (the last assistant marker and the
    /// block the prompt opened or closed), then EOS. It finds where the reply starts in the
    /// tokens it has been fed, so it answers whatever prompt a surface renders.
    /// </summary>
    private sealed class ScriptModel : ModelBase
    {
        private const string AssistantMarker = "<SPECIAL_11>Assistant\n";
        private readonly string _script;
        private readonly StringBuilder _seen = new();

        private readonly int _msPerToken;

        public ScriptModel(string path, string template, string script, int msPerToken) : base(path, BackendType.Cpu)
        {
            _script = script;
            _msPerToken = msPerToken;
            Config = new ModelConfig { Architecture = "nemotron_h", VocabSize = 128, NumLayers = 1, ChatTemplate = template };
            Tokenizer = new CharTokenizer();
            _maxContextLength = 32_768;
        }

        public override bool SupportsKVStateSnapshot => true;
        public override bool SupportsCrossSequenceKvReuse => false;
        public override string KVStateFingerprint => "nemotron-thinkoff-serving";
        public override long ComputeKVBlockByteSize(int tokenCount) => 4L * tokenCount;
        public override bool TryExtractKVBlock(int start, int count, Span<byte> destination) => true;
        public override bool TryInjectKVBlock(int start, int count, ReadOnlySpan<byte> source) => true;

        protected override float[] ForwardCore(int[] tokens)
        {
            if (_msPerToken > 0 && tokens.Length == 1)
                Thread.Sleep(_msPerToken);
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
            if (seen.AsSpan(start).StartsWith("<think></think>"))
                start += "<think></think>".Length;
            else if (seen.AsSpan(start).StartsWith("<think>\n"))
                start += "<think>\n".Length;
            int at = seen.Length - start;
            return at < _script.Length ? _script[at] : 1;
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
