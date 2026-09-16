// Regression coverage for the "Gemma 4 output unrelated to the prompt" bug.
//
// Root-cause class: the prompt that reaches the model must (a) actually contain
// the user's question and (b) carry exactly one BOS token. Two failure modes
// could violate that for Gemma 4:
//
//   1. The GGUF chat template captures each message's content into a block-form
//      {% set captured %}...{% endset %}. A Jinja engine that can't evaluate a
//      feature the template uses can render that capture EMPTY, silently dropping
//      the user's question — the model then answers nothing in particular, i.e.
//      "output unrelated to the prompt". ChatTemplate now guards against this by
//      verifying the rendered prompt still contains the last user message and
//      falling back to the hardcoded template otherwise.
//
//   2. The hardcoded fallback used to emit a literal "<bos>" while the tokenizer
//      also prepends BOS (add_bos_token=true), producing a double BOS. Fixed by
//      letting the tokenizer own the BOS, matching every other renderer.
//
//   3. The MIRROR of (2): some GGUF builds (e.g. gemma-4-31B-it-UD-IQ2_M) set
//      add_bos_token=false and rely on the template's "{{ bos_token }}" to emit
//      the BOS. TensorSharp renders bos_token empty AND the tokenizer won't add
//      one, so the prompt ends up with ZERO BOS. A Gemma model with a missing BOS
//      produces a coherent opening that then collapses into repeating a single
//      token ("...是一个一个一个..."). ModelBase.ResolveAddBosToken now detects a
//      template-declared BOS and lets the tokenizer own it (exactly one BOS). The
//      Generate_* tests assert the output does not degenerate into repetition.
//
// The unit tests need no model and run everywhere. The end-to-end tests load the
// real 12B GGUF when TS_GEMMA4_12B (or ~/work/model/gemma-4-12b-it-Q8_0.gguf)
// is present, otherwise they skip.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class Gemma4PromptRenderReproTests
{
    private readonly ITestOutputHelper _output;
    public Gemma4PromptRenderReproTests(ITestOutputHelper output) { _output = output; }

    // ---------- unit tests (no model) ----------

    [Fact]
    public void HardcodedGemma4_DoesNotEmitLiteralBos_AvoidsDoubleBos()
    {
        var history = new List<ChatMessage> { new() { Role = "user", Content = "请介绍最终幻想7" } };

        // Forcing an empty template selects the hardcoded RenderGemma4 path.
        string rendered = ChatTemplate.RenderFromGgufTemplate(
            template: "", messages: history, addGenerationPrompt: true, architecture: "gemma4");

        Assert.DoesNotContain("<bos>", rendered);
        Assert.Contains("请介绍最终幻想7", rendered);
        // The turn framing must still be present.
        Assert.Contains("<|turn>user", rendered);
        Assert.Contains("<|turn>model", rendered);
    }

    [Fact]
    public void RenderFromGguf_FallsBackToHardcoded_WhenJinjaDropsUserContent()
    {
        // A valid-but-wrong template that renders non-empty text yet ignores the
        // messages entirely — exactly what a mis-evaluated block-set capture looks
        // like (user question vanishes). The guard must detect the missing user
        // text and fall back to the hardcoded template, which includes it.
        const string droppingTemplate = "{{ bos_token }}SYSTEM PREAMBLE WITH NO MESSAGES";
        var history = new List<ChatMessage> { new() { Role = "user", Content = "请介绍最终幻想7" } };

        string rendered = ChatTemplate.RenderFromGgufTemplate(
            droppingTemplate, history, addGenerationPrompt: true, architecture: "gemma4");

        Assert.Contains("请介绍最终幻想7", rendered);          // user content recovered
        Assert.DoesNotContain("SYSTEM PREAMBLE WITH NO MESSAGES", rendered); // jinja output discarded
    }

    [Fact]
    public void RenderFromGguf_KeepsJinjaOutput_WhenUserContentPresent()
    {
        // A correct minimal template that DOES include the message content must be
        // used as-is (the guard must not produce false positives).
        const string goodTemplate =
            "{{ bos_token }}{% for m in messages %}<|turn>{{ m['role'] }}\n{{ m['content'] }}<turn|>\n{% endfor %}";
        var history = new List<ChatMessage> { new() { Role = "user", Content = "请介绍最终幻想7" } };

        string rendered = ChatTemplate.RenderFromGgufTemplate(
            goodTemplate, history, addGenerationPrompt: true, architecture: "gemma4");

        Assert.Contains("请介绍最终幻想7", rendered);
        Assert.Contains("<|turn>user", rendered);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RenderFromGguf_RestoresGemma4PromptNewline(bool thinking)
    {
        const string template =
            "<|turn>user\n{{ messages[0]['content'] }}<turn|>\n" +
            "{% if add_generation_prompt %}<|turn>model\n{% endif %}";
        var history = new List<ChatMessage> { new() { Role = "user", Content = "hello" } };

        string rendered = ChatTemplate.RenderFromGgufTemplate(
            template,
            history,
            addGenerationPrompt: true,
            architecture: "gemma4",
            enableThinking: thinking);

        Assert.EndsWith("<|turn>model\n", rendered);

        var tokenizer = CreateCharacterGemma4Tokenizer(rendered);
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        List<int> tokens = renderer.RenderToTokens(
            tokenizer,
            template,
            history,
            architecture: "gemma4",
            addGenerationPrompt: true,
            enableThinking: thinking);
        int newlineId = tokenizer.LookupToken("\n");
        Assert.Equal(newlineId, tokens[^1]);
        Assert.NotEqual(newlineId, tokens[^2]);
    }

    [Fact]
    public void HardcodedGemma4_ThinkingDisabled_LeavesTheModelTurnOpen()
    {
        var history = new List<ChatMessage> { new() { Role = "user", Content = "hello" } };
        string rendered = ChatTemplate.RenderGemma4(history, enableThinking: false);
        Assert.EndsWith("<|turn>model\n", rendered);
        Assert.DoesNotContain("<|channel>", rendered);
    }

    [Fact]
    public void RenderFromGguf_PreservesPublisherSuppliedChannelBlock()
    {
        const string template = "<|turn>user\n{{ messages[0]['content'] }}<turn|>\n" +
            "<|turn>model\n<|channel>thought\n<channel|>";
        var history = new List<ChatMessage> { new() { Role = "user", Content = "hello" } };
        string rendered = ChatTemplate.RenderFromGgufTemplate(
            template, history, architecture: "gemma4", enableThinking: false);
        Assert.EndsWith("<|turn>model\n<|channel>thought\n<channel|>", rendered);
    }

    [Fact]
    public void RenderFromGguf_DoesNotAddGemma4ModelTurnWithoutGenerationPrompt()
    {
        const string template =
            "<|turn>user\n{{ messages[0]['content'] }}<turn|>\n" +
            "{% if add_generation_prompt %}<|turn>model\n{% endif %}";
        var history = new List<ChatMessage> { new() { Role = "user", Content = "hello" } };

        string rendered = ChatTemplate.RenderFromGgufTemplate(
            template,
            history,
            addGenerationPrompt: false,
            architecture: "gemma4",
            enableThinking: true);

        Assert.DoesNotContain("<|turn>model", rendered);
        Assert.EndsWith("<turn|>", rendered);
    }

    private const string Gemma4CachedToolReasoningTemplate =
        "{%- macro format_tool_response_block(tool_name, response) -%}" +
        "{{- '<|tool_response>' + response + '<tool_response|>' -}}" +
        "{%- endmacro -%}" +
        "{%- set ns = namespace(prev_message_type=None) -%}" +
        "{%- set loop_messages = messages -%}" +
        "{%- set ns_turn = namespace(last_user_idx=-1) -%}" +
        "{%- for message in loop_messages -%}" +
        "{%- if message['role'] == 'user' -%}" +
        "{%- set ns_turn.last_user_idx = loop.index0 -%}" +
        "{%- endif -%}" +
        "{%- endfor -%}" +
        "{%- for message in loop_messages -%}" +
        "{%- if message['role'] != 'tool' -%}" +
        "{%- set thinking_text = message.get('reasoning') or message.get('reasoning_content') -%}" +
        "{%- if thinking_text and loop.index0 > ns_turn.last_user_idx and message.get('tool_calls') -%}" +
        "{{- '<|channel>thought\\n' + thinking_text + '\\n<channel|>' -}}" +
        "{%- endif -%}" +
        "{%- if message['tool_calls'] -%}" +
        "{%- set ns.prev_message_type = 'tool_call' -%}" +
        "{{- '<CALL>' -}}" +
        "{%- elif message['role'] == 'user' -%}" +
        "{{- '<U>' + message['content'] -}}" +
        "{%- elif message['role'] == 'assistant' -%}" +
        "{{- '<A>' + message['content'] -}}" +
        "{%- endif -%}" +
        "{%- if message.get('tool_responses') -%}" +
        "{%- elif message.get('tool_calls') -%}" +
        "{%- for k in range(loop.index0 + 1, loop_messages | length) -%}" +
        "{%- endfor -%}" +
        "{%- endif -%}" +
        "{%- else -%}" +
        "{{- format_tool_response_block('', message['content']) -}}" +
        "{%- endif -%}" +
        "{%- endfor -%}";

    private static List<ChatMessage> CachedGemma4ToolHistory(bool cached)
    {
        return new List<ChatMessage>
        {
            new() { Role = "user", Content = "make the game" },
            new()
            {
                Role = "assistant",
                Content = "",
                Thinking = "inspect the workspace",
                ToolCalls = new List<ToolCall>
                {
                    new()
                    {
                        Name = "shell",
                        Arguments = new Dictionary<string, object> { ["command"] = "pwd" },
                    },
                },
                RawOutputTokens = cached ? new List<int> { 101, 102 } : null,
            },
            new() { Role = "tool", Content = "/workspace" },
            new() { Role = "user", Content = "now add a boss" },
        };
    }

    [Fact]
    public void CachedGemma4ToolRound_ReplaysReasoningAfterANewerUserTurn()
    {
        string rendered = ChatTemplate.RenderFromGgufTemplate(
            Gemma4CachedToolReasoningTemplate,
            CachedGemma4ToolHistory(cached: true),
            addGenerationPrompt: false,
            architecture: "gemma4",
            enableThinking: true);

        const string thought = "<|channel>thought\ninspect the workspace<channel|>";
        const string result = "<|tool_response>/workspace<tool_response|>";
        Assert.Contains(thought, rendered);
        Assert.Contains(result, rendered);
        Assert.True(rendered.IndexOf(thought, StringComparison.Ordinal)
            < rendered.IndexOf("<CALL>", StringComparison.Ordinal));
        Assert.True(rendered.IndexOf("<CALL>", StringComparison.Ordinal)
            < rendered.IndexOf(result, StringComparison.Ordinal));
        Assert.DoesNotContain(ChatTemplate.ReasoningEndSentinel, rendered);
    }

    [Fact]
    public void NonCachedGemma4ToolRound_KeepsCanonicalLastUserReasoningGate()
    {
        string rendered = ChatTemplate.RenderFromGgufTemplate(
            Gemma4CachedToolReasoningTemplate,
            CachedGemma4ToolHistory(cached: false),
            addGenerationPrompt: false,
            architecture: "gemma4",
            enableThinking: true);

        Assert.DoesNotContain("inspect the workspace", rendered);
        Assert.Contains("<CALL>", rendered);
        Assert.Contains("<|tool_response>/workspace<tool_response|>", rendered);
        Assert.Contains("now add a boss", rendered);
    }

    private static BpeTokenizer CreateCharacterGemma4Tokenizer(string text)
    {
        string[] vocab = text
            .Select(c => c.ToString())
            .Append("\u2581")
            .Distinct(StringComparer.Ordinal)
            .ToArray();
        return new BpeTokenizer(
            vocab,
            Enumerable.Repeat(1, vocab.Length).ToArray(),
            Array.Empty<string>(),
            bosTokenId: -1,
            eosTokenIds: Array.Empty<int>(),
            addBos: false,
            addEos: false,
            preTokenizerType: "gemma4");
    }

    // ---------- end-to-end tests (need the real 12B GGUF) ----------

    private static string? FindModel()
    {
        string[] candidates =
        {
            Environment.GetEnvironmentVariable("TS_GEMMA4_12B") ?? "",
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                "work", "model", "gemma-4-12b-it-Q8_0.gguf"),
        };
        return candidates.FirstOrDefault(p => !string.IsNullOrEmpty(p) && File.Exists(p));
    }

    private static string? FindBpeModel()
    {
        string[] candidates =
        {
            Environment.GetEnvironmentVariable("TS_GEMMA4_BPE_MODEL") ?? "",
            Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                "work", "model", "mtp_gemma4", "gemma-4-26B-A4B-it-Q4_K_M.gguf"),
        };
        return candidates.FirstOrDefault(p => !string.IsNullOrEmpty(p) && File.Exists(p));
    }

    [ModelFact("TS_GEMMA4_BPE_MODEL")]
    public void RealGemma4BpeTokenizer_MatchesLlamaCppOracle()
    {
        string? modelPath = FindBpeModel();
        if (modelPath == null) { _output.WriteLine("Gemma 4 BPE GGUF not found; skipping"); return; }

        using var gguf = new GgufFile(modelPath);
        Assert.Equal("gemma4", gguf.GetString("tokenizer.ggml.model"));

        ITokenizer tokenizer = ModelBase.CreateTokenizerFromGguf(gguf);
        Assert.IsType<BpeTokenizer>(tokenizer);

        Assert.Equal(new[] { 2, 23391 }, tokenizer.Encode("hello", addSpecial: true));
        Assert.Equal(
            new[] { 54593, 786, 1003, 614, 763, 76215, 236761 },
            tokenizer.Encode("Tell me about an abacus.", addSpecial: false));
        Assert.Equal(
            new[] { 1364, 237001, 74235 },
            tokenizer.Encode("coöperate", addSpecial: false));
        Assert.Equal(
            new[] { 12324, 236787, 3714, 596, 91577, 1083, 1847, 236783 },
            tokenizer.Encode("JSON: {\"abbreviation\": true}", addSpecial: false));

        const string noThinkingPrompt =
            "<|turn>user\nhello<turn|>\n<|turn>model\n<|channel>thought\n<channel|>";
        Assert.Equal(
            new[] { 2, 105, 2364, 107, 23391, 106, 107, 105, 4368, 107, 100, 45518, 107, 101 },
            tokenizer.Encode(noThinkingPrompt, addSpecial: true));

        // Gemma 4 GGUFs exist with two upstream system-think preamble revisions.
        // Keep both byte shapes and both token sequences as explicit oracles: the only
        // difference is whether <|think|> and <turn|> are adjacent or newline-separated.
        // Accepting either rendered shape keeps the test useful across those revisions
        // without weakening it to a whitespace-normalized comparison.
        const string compactSystemThinkPrompt =
            "<|turn>system\n<|think|><turn|>\n<|turn>user\nhello<turn|>\n<|turn>model\n";
        const string separatedSystemThinkPrompt =
            "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nhello<turn|>\n<|turn>model\n";
        int[] compactSystemThinkTokens =
            { 2, 105, 9731, 107, 98, 106, 107, 105, 2364, 107, 23391, 106, 107, 105, 4368, 107 };
        int[] separatedSystemThinkTokens =
            { 2, 105, 9731, 107, 98, 107, 106, 107, 105, 2364, 107, 23391, 106, 107, 105, 4368, 107 };

        Assert.Equal(
            compactSystemThinkTokens,
            tokenizer.Encode(compactSystemThinkPrompt, addSpecial: true));
        Assert.Equal(
            separatedSystemThinkTokens,
            tokenizer.Encode(separatedSystemThinkPrompt, addSpecial: true));

        var history = new List<ChatMessage> { new() { Role = "user", Content = "hello" } };
        string renderedThinkingPrompt = ChatTemplate.RenderFromGgufTemplate(
            gguf.GetString("tokenizer.chat_template"),
            history,
            addGenerationPrompt: true,
            architecture: "gemma4",
            enableThinking: true);
        Assert.True(
            renderedThinkingPrompt == compactSystemThinkPrompt
                || renderedThinkingPrompt == separatedSystemThinkPrompt,
            "The embedded Gemma 4 template must render one of the two known exact " +
            "system-think preamble revisions.");
        int[] renderedThinkingTokens = renderedThinkingPrompt == compactSystemThinkPrompt
            ? compactSystemThinkTokens
            : separatedSystemThinkTokens;
        Assert.Equal(
            renderedThinkingTokens,
            tokenizer.Encode(renderedThinkingPrompt, addSpecial: true));

        Assert.Equal(
            "Hello! How can I help you today?",
            tokenizer.Decode(new List<int> { 9259, 236888, 2088, 740, 564, 1601, 611, 3124, 236881 }));
    }

    [ModelFact("TS_GEMMA4_BPE_MODEL")]
    public void RealTemplate_CodeToolTranscript_RemainsAnExactPrefixAfterANewUserTurn()
    {
        // Metadata/tokenizer-only regression for the code-exec live-cache failure.
        // No weights or backend are loaded. Deriving every expected prefix from this
        // GGUF's own template keeps the test valid across Gemma template revisions.
        string? modelPath = FindBpeModel();
        if (modelPath == null) { _output.WriteLine("Gemma 4 BPE GGUF not found; skipping"); return; }

        using var gguf = new GgufFile(modelPath);
        ITokenizer tokenizer = ModelBase.CreateTokenizerFromGguf(gguf);
        string template = gguf.GetString("tokenizer.chat_template") ?? "";
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        var tools = new List<ToolFunction>
        {
            new()
            {
                Name = "write_file",
                Description = "Write a text file.",
                Parameters = new Dictionary<string, ToolParameter>
                {
                    ["path"] = new() { Type = "string", Description = "Path" },
                    ["content"] = new() { Type = "string", Description = "Content" },
                },
                Required = new List<string> { "path", "content" },
            },
        };

        var baseHistory = new List<ChatMessage>
        {
            new() { Role = "system", Content = "Use tools when requested." },
            new() { Role = "user", Content = "Create a tiny file." },
        };
        var firstPrompt = renderer.RenderToTokens(
            tokenizer, template, baseHistory, "gemma4", true,
            out _, out string firstBoundary, tools: tools, enableThinking: true);

        const string firstRawText =
            "<|channel>thought\n<channel|>" +
            "<|tool_call>call:write_file{content:<|\"|><h1>ok</h1><|\"|>,path:<|\"|>game.html<|\"|>}<tool_call|>";
        List<int> firstRaw = tokenizer.Encode(firstRawText, addSpecial: false);

        // The sampler is allowed to produce a valid but non-canonical BPE
        // decomposition. Replace one ordinary merged token by byte-fallback tokens:
        // decoded text (and therefore parser output) stays identical, while a later
        // structured reserialization cannot recover the sampled token IDs.
        List<int> canonicalFirstRaw = new(firstRaw);
        Assert.True(ReplaceOneTokenWithByteFallback((BpeTokenizer)tokenizer, firstRaw));
        Assert.NotEqual(canonicalFirstRaw, firstRaw);
        Assert.Equal(firstRawText, tokenizer.Decode(firstRaw));

        var firstParser = new Gemma4OutputParser();
        firstParser.Init(enableThinking: true, tools);
        var parsedThinking = new System.Text.StringBuilder();
        var parsedContent = new System.Text.StringBuilder();
        var parsedCalls = new List<ToolCall>();
        foreach (char ch in tokenizer.Decode(firstRaw))
        {
            ParsedOutput delta = firstParser.Add(ch.ToString(), done: false);
            parsedThinking.Append(delta.Thinking);
            parsedContent.Append(delta.Content);
            if (delta.ToolCalls != null) parsedCalls.AddRange(delta.ToolCalls);
        }
        ParsedOutput parsedLast = firstParser.Add(string.Empty, done: true);
        parsedThinking.Append(parsedLast.Thinking);
        parsedContent.Append(parsedLast.Content);
        if (parsedLast.ToolCalls != null) parsedCalls.AddRange(parsedLast.ToolCalls);

        var firstLive = new List<int>(firstPrompt);
        firstLive.AddRange(firstRaw);

        var afterTool = new List<ChatMessage>(baseHistory)
        {
            new()
            {
                Role = "assistant",
                Content = parsedContent.ToString(),
                Thinking = parsedThinking.ToString(),
                ToolCalls = parsedCalls,
                RawOutputTokens = firstRaw,
                RawPromptTrailingWhitespace = firstBoundary,
                // Part-scoped content offsets cannot be recovered after raw replay,
                // so this conservatively marks the start of the raw run. The
                // message-scoped marker must close that run before the canonical
                // template forwards the following tool result.
                ContentCacheBreakpoints = new List<int> { 0 },
                CacheControl = new CacheControlMarker(),
            },
            new() { Role = "tool", Content = "Created game.html" },
        };
        var secondPrompt = renderer.RenderToTokens(
            tokenizer, template, afterTool, "gemma4", true,
            out List<int>? firstRoundBreakpoints, out string secondBoundary,
            tools: tools, enableThinking: true);
        int firstLcp = LongestCommonPrefix(firstLive, secondPrompt);
        Assert.Equal(firstLive.Count, firstLcp);
        Assert.Equal(
            new[] { firstPrompt.Count, firstPrompt.Count + firstRaw.Count },
            firstRoundBreakpoints);
        Assert.Contains("Created game.html", tokenizer.Decode(secondPrompt));

        const string secondRawText =
            "<|channel>thought\nThe file is ready.<channel|>Done.";
        List<int> secondRaw = tokenizer.Encode(secondRawText, addSpecial: false);
        var secondLive = new List<int>(secondPrompt);
        secondLive.AddRange(secondRaw);

        var followUp = new List<ChatMessage>(afterTool)
        {
            new()
            {
                Role = "assistant",
                Content = "Done.",
                Thinking = "The file is ready.",
                RawOutputTokens = secondRaw,
                RawPromptTrailingWhitespace = secondBoundary,
            },
            new() { Role = "user", Content = "Send the link again." },
        };
        var thirdPrompt = renderer.RenderToTokens(
            tokenizer, template, followUp, "gemma4", true,
            tools: tools, enableThinking: true);
        int secondLcp = LongestCommonPrefix(secondLive, thirdPrompt);

        Assert.Equal(secondLive.Count, secondLcp);
    }

    [ModelFact("TS_GEMMA4_12B")]
    public void RealTemplate_RendersUserText_AndSingleBos()
    {
        string? modelPath = FindModel();
        if (modelPath == null) { _output.WriteLine("gemma-4-12b GGUF not found; skipping"); return; }

        using var gguf = new GgufFile(modelPath);
        string chatTemplate = gguf.GetString("tokenizer.chat_template") ?? "";
        ITokenizer tokenizer = ModelBase.CreateTokenizerFromGguf(gguf);
        int bosId = tokenizer.BosTokenId;

        var history = new List<ChatMessage> { new() { Role = "user", Content = "请介绍最终幻想7" } };
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        List<int> tokens = renderer.RenderToTokens(
            tokenizer, chatTemplate, history, "gemma4", addGenerationPrompt: true);

        _output.WriteLine($"tokens: {string.Join(" ", tokens)}");
        var userText = tokenizer.Encode("请介绍最终幻想7", addSpecial: false);
        Assert.True(ContainsSubsequence(tokens, userText),
            "rendered prompt does not contain the user's text tokens");
        Assert.Equal(1, CountLeading(tokens, bosId)); // exactly one BOS
    }

    [ModelFact("TS_GEMMA4_12B")]
    public async Task Generate_FinalFantasyPrompt_ProducesRelevantOutput()
    {
        string? modelPath = FindModel();
        if (modelPath == null) { _output.WriteLine("gemma-4-12b GGUF not found; skipping"); return; }

        BackendType backend = OperatingSystem.IsMacOS() ? BackendType.GgmlMetal : BackendType.GgmlCpu;
        var model = (Gemma4Model)ModelBase.Create(modelPath, backend);
        try
        {
            string mmproj = Path.Combine(Path.GetDirectoryName(modelPath)!, "gemma-4-12b-mmproj-BF16.gguf");
            if (File.Exists(mmproj)) model.MultimodalInjector.LoadProjectors(mmproj);

            var history = new List<ChatMessage> { new() { Role = "user", Content = "请介绍最终幻想7" } };
            var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
            List<int> promptTokens = renderer.RenderToTokens(
                model.Tokenizer, model.Config.ChatTemplate, history, "gemma4", addGenerationPrompt: true);

            // Drive the same continuous-batching engine the server uses.
            var cfg = SchedulerConfig.FromEnvironment();
            using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);
            var seq = new SequenceState("ff7", promptTokens, maxNewTokens: 64, blockSize: cfg.BlockSize,
                samplingConfig: SamplingConfig.Greedy);
            var handle = engine.SubmitRequest(seq);
            var outToks = new List<int>();
            await foreach (var t in handle.Tokens.ReadAllAsync())
                outToks.Add(t);
            string text = model.Tokenizer.Decode(outToks);
            _output.WriteLine($"output ({outToks.Count} tok): {text}");

            Assert.True(outToks.Count > 0, "engine produced no tokens");
            // The answer must actually be about Final Fantasy 7.
            bool relevant = text.Contains("最终幻想") || text.Contains("Final Fantasy") ||
                            text.Contains("FF7") || text.Contains("FF");
            Assert.True(relevant, $"output is unrelated to the prompt: {text}");

            // Degeneration guard: a missing BOS (or other compute bug) makes the model
            // emit a coherent opening and then collapse into repeating a single token
            // ("...是一个一个一个一个..."). The prefix still contains "最终幻想", so the
            // relevance check alone does NOT catch it. Require that the generated tokens
            // are reasonably diverse - genuine prose keeps most tokens distinct, whereas
            // a repetition collapse drives the distinct-token ratio toward zero.
            int distinctTokens = outToks.Distinct().Count();
            Assert.True(distinctTokens >= outToks.Count / 2,
                $"output degenerated into repetition ({distinctTokens} distinct of {outToks.Count} tokens): {text}");
        }
        finally
        {
            model.Dispose();
        }
    }

    [ModelFact("TS_GEMMA4_12B")]
    public async Task Generate_ThreeDistinctPromptsInParallel_EachStaysOnTopic()
    {
        string? modelPath = FindModel();
        if (modelPath == null) { _output.WriteLine("gemma-4-12b GGUF not found; skipping"); return; }

        BackendType backend = OperatingSystem.IsMacOS() ? BackendType.GgmlMetal : BackendType.GgmlCpu;
        var model = (Gemma4Model)ModelBase.Create(modelPath, backend);
        try
        {
            var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
            var cfg = SchedulerConfig.FromEnvironment();
            using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);

            // Distinct topics. If the batched (batch>1) path cross-contaminates
            // sequences (e.g. the 16:1 GQA global layers with 1 KV head), an
            // answer would drift onto another request's topic or degenerate.
            var prompts = new (string q, string[] keys)[]
            {
                ("请介绍最终幻想7",   new[] { "最终幻想", "Final", "FF" }),
                ("请介绍时间简史这本书", new[] { "时间简史", "霍金", "宇宙", "Hawking" }),
                ("请介绍量子力学",     new[] { "量子", "粒子", "physics", "微观" }),
            };

            async Task<string> Run(string q)
            {
                var hist = new List<ChatMessage> { new() { Role = "user", Content = q } };
                var toks = renderer.RenderToTokens(model.Tokenizer, model.Config.ChatTemplate, hist, "gemma4", true);
                var seq = new SequenceState(Guid.NewGuid().ToString("N"), toks, maxNewTokens: 64,
                    blockSize: cfg.BlockSize, samplingConfig: SamplingConfig.Greedy);
                var handle = engine.SubmitRequest(seq);
                var outToks = new List<int>();
                await foreach (var t in handle.Tokens.ReadAllAsync()) outToks.Add(t);
                return model.Tokenizer.Decode(outToks);
            }

            var tasks = prompts.Select(p => Run(p.q)).ToArray();
            string[] outs = await Task.WhenAll(tasks);

            for (int i = 0; i < prompts.Length; i++)
            {
                _output.WriteLine($"[{prompts[i].q}] -> {outs[i].Replace("\n", " ")}");
                Assert.True(outs[i].Length > 0, $"req {i} empty");
                bool onTopic = prompts[i].keys.Any(k => outs[i].Contains(k));
                Assert.True(onTopic, $"req {i} ('{prompts[i].q}') drifted off-topic: {outs[i]}");
            }
        }
        finally
        {
            model.Dispose();
        }
    }

    // Guards the GGML-Metal async-compute hazard: the seqLen>1 prefill mixes CPU
    // (Parallel.For) writes with GPU kernels, and under lazy-sync those writes can
    // be invisible to the next GPU op — corrupting the prompt's contribution and
    // producing coherent-but-off-topic output. The multi-token prefill must yield
    // the SAME next-token as feeding the prompt one token at a time (decode path),
    // and must be deterministic across repeats.
    [ModelFact("TS_GEMMA4_12B")]
    public void PrefillNextToken_MatchesIncrementalDecode_AndIsDeterministic()
    {
        string? modelPath = FindModel();
        if (modelPath == null) { _output.WriteLine("gemma-4-12b GGUF not found; skipping"); return; }

        BackendType backend = OperatingSystem.IsMacOS() ? BackendType.GgmlMetal : BackendType.GgmlCpu;
        var model = (Gemma4Model)ModelBase.Create(modelPath, backend);
        try
        {
            var history = new List<ChatMessage> { new() { Role = "user", Content = "请介绍最终幻想7" } };
            var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
            int[] prompt = renderer.RenderToTokens(model.Tokenizer, model.Config.ChatTemplate, history, "gemma4", true).ToArray();

            // Reference: incremental seqLen=1 forwards (the always-correct decode path).
            model.ResetKVCache();
            float[] incLogits = null;
            for (int i = 0; i < prompt.Length; i++) incLogits = model.Forward(new[] { prompt[i] });
            int incTop = ArgMax(incLogits);
            _output.WriteLine($"incremental next-token = {incTop} '{model.Tokenizer.Decode(new List<int> { incTop })}'");

            // Multi-token prefill, twice, must match the reference and each other.
            int prefillTop0, prefillTop1;
            model.ResetKVCache();
            prefillTop0 = ArgMax(model.Forward(prompt));
            model.ResetKVCache();
            prefillTop1 = ArgMax(model.Forward(prompt));
            _output.WriteLine($"prefill next-token (rep0/rep1) = {prefillTop0} / {prefillTop1}");

            Assert.Equal(incTop, prefillTop0);
            Assert.Equal(prefillTop0, prefillTop1);
        }
        finally { model.Dispose(); }
    }

    // ---------- helpers ----------

    private static bool ReplaceOneTokenWithByteFallback(BpeTokenizer tokenizer, List<int> ids)
    {
        var specialIds = new HashSet<int>(tokenizer.SpecialTokenIds);
        for (int i = 0; i < ids.Count; i++)
        {
            if (specialIds.Contains(ids[i]))
                continue;

            string piece = tokenizer.Decode(new List<int> { ids[i] });
            if (piece.Length == 0)
                continue;

            var replacement = new List<int>();
            bool complete = true;
            foreach (byte value in System.Text.Encoding.UTF8.GetBytes(piece))
            {
                int byteId = tokenizer.LookupToken($"<0x{value:X2}>");
                if (byteId < 0)
                {
                    complete = false;
                    break;
                }
                replacement.Add(byteId);
            }

            if (!complete
                || replacement.SequenceEqual(new[] { ids[i] })
                || tokenizer.Decode(replacement) != piece)
            {
                continue;
            }

            ids.RemoveAt(i);
            ids.InsertRange(i, replacement);
            return true;
        }
        return false;
    }

    private static bool ContainsSubsequence(List<int> hay, List<int> needle)
    {
        if (needle.Count == 0) return true;
        for (int i = 0; i + needle.Count <= hay.Count; i++)
        {
            bool ok = true;
            for (int j = 0; j < needle.Count; j++)
                if (hay[i + j] != needle[j]) { ok = false; break; }
            if (ok) return true;
        }
        return false;
    }

    private static int CountLeading(List<int> toks, int id)
    {
        int n = 0;
        foreach (int t in toks) { if (t == id) n++; else break; }
        return n;
    }

    private static int LongestCommonPrefix(IReadOnlyList<int> left, IReadOnlyList<int> right)
    {
        int n = Math.Min(left.Count, right.Count);
        int i = 0;
        while (i < n && left[i] == right[i]) i++;
        return i;
    }

    private static int ArgMax(float[] a)
    {
        int best = 0;
        for (int i = 1; i < a.Length; i++) if (a[i] > a[best]) best = i;
        return best;
    }
}
