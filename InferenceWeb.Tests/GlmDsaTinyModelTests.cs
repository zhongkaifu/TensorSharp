// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// glm-dsa architecture tests that need no model download.
//
// GlmDsaSyntheticModelBuilder writes a 1.9 MB glm-dsa GGUF with the same block
// shape as GLM-5.2 and a deterministic weight generator, so these run anywhere
// and still cover the parts of the architecture that are easy to get wrong:
// MLA absorption, the DSA lightning indexer's top-k selection (its top_k is 8,
// so a 24-token prompt is already sparse), the sigmoid-gated MoE with its
// routing bias and shared expert, chunked prefill, and multi-step decode.
//
// GOLDENS: the expected token ids come from llama.cpp b200-9731ad3 evaluating
// the SAME file through its own glm-dsa graph. Regenerate with
// `.parity/glm_parity.cpp` (see .parity/README.md) if the reference changes.
using System;
using System.IO;
using System.Linq;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class GlmDsaTinyModelTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly string _dir;

    public GlmDsaTinyModelTests(ITestOutputHelper output)
    {
        _output = output;
        _dir = Path.Combine(Path.GetTempPath(), "ts-glm-tiny-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private string BuildModel(int[] indexerTypes = null, bool quantize = false) =>
        GlmDsaSyntheticModelBuilder.Write(Path.Combine(_dir, "tiny-glm-dsa.gguf"), indexerTypes, quantize);

    /// <summary>24 tokens; with indexer top_k = 8 this is already the SPARSE path.</summary>
    private static int[] Prompt24() => Enumerable.Range(0, 24).Select(i => 65 + (i * 7) % 50).ToArray();

    /// <summary>60 tokens: several full ubatches worth of sparse attention.</summary>
    private static int[] Prompt60() => Enumerable.Range(0, 60).Select(i => 65 + (i * 13) % 60).ToArray();

    // llama.cpp b200-9731ad3, same file, greedy.
    private static readonly int[] Golden24 = { 84, 46, 77, 48, 3, 37, 9, 3 };
    private static readonly int[] Golden60 = { 33, 33, 3, 37, 9, 37, 9, 37 };

    private static int[] Greedy(ModelBase model, int[] prompt, int count)
    {
        model.ResetKVCache();
        float[] logits = model.ForwardRefill(prompt);
        var produced = new int[count];
        for (int i = 0; i < count; i++)
        {
            int best = 0;
            for (int v = 1; v < logits.Length; v++) if (logits[v] > logits[best]) best = v;
            produced[i] = best;
            if (i + 1 < count) logits = model.Forward(new[] { best });
        }
        return produced;
    }

    [Fact]
    public void GlmDsa_LoadsAndReportsArchitecture()
    {
        using var model = ModelBase.Create(BuildModel(), BackendType.Cpu);
        Assert.Equal("glm-dsa", model.Config.Architecture);
        Assert.Equal(GlmDsaSyntheticModelBuilder.Hidden, model.Config.HiddenSize);
        Assert.Equal(GlmDsaSyntheticModelBuilder.NumHeads, model.Config.NumHeads);
        Assert.Equal(GlmDsaSyntheticModelBuilder.NumExperts, model.Config.NumExperts);
        Assert.Equal(GlmDsaSyntheticModelBuilder.ExpertsUsed, model.Config.NumExpertsUsed);
        Assert.Equal(GlmDsaSyntheticModelBuilder.VocabSize, model.Config.VocabSize);
        // block_count includes the NextN block; the trunk the graph runs does not.
        Assert.Equal(GlmDsaSyntheticModelBuilder.NumBlocks - GlmDsaSyntheticModelBuilder.NumNextn,
                     model.Config.NumLayers);
    }

    [Fact]
    public void GlmDsa_SparsePrompt_MatchesLlamaCpp()
    {
        using var model = ModelBase.Create(BuildModel(), BackendType.Cpu);
        Assert.Equal(Golden24, Greedy(model, Prompt24(), Golden24.Length));
    }

    [Fact]
    public void GlmDsa_LongerSparsePrompt_MatchesLlamaCpp()
    {
        using var model = ModelBase.Create(BuildModel(), BackendType.Cpu);
        Assert.Equal(Golden60, Greedy(model, Prompt60(), Golden60.Length));
    }

    /// <summary>
    /// A chunked prefill has to leave the indexer cache in the same state a
    /// single-shot prefill would: a chunk that is still shorter than top_k skips
    /// the SELECTION but must still cache its keys, or a later chunk scores rows
    /// that were never written.
    /// </summary>
    [Theory]
    [InlineData(3)]
    [InlineData(7)]
    [InlineData(16)]
    [InlineData(64)]
    public void GlmDsa_ChunkedPrefill_MatchesSingleShot(int chunk)
    {
        string path = BuildModel();
        string previous = Environment.GetEnvironmentVariable("TS_PREFILL_CHUNK");
        Environment.SetEnvironmentVariable("TS_PREFILL_CHUNK", chunk.ToString());
        try
        {
            using var model = ModelBase.Create(path, BackendType.Cpu);
            Assert.Equal(Golden60, Greedy(model, Prompt60(), Golden60.Length));
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_PREFILL_CHUNK", previous);
        }
    }

    /// <summary>
    /// The "shared indexer" layout GLM-5.2 actually ships: only some layers score
    /// the cache, the rest reuse the previous full layer's selection.
    /// </summary>
    [Fact]
    public void GlmDsa_SharedIndexerLayers_Run()
    {
        // Layers 0 and 2 compute a fresh top-k; 1 and 3 inherit it.
        using var model = ModelBase.Create(BuildModel(new[] { 1, 0, 1, 0 }), BackendType.Cpu);
        int[] produced = Greedy(model, Prompt24(), 4);
        Assert.All(produced, t => Assert.InRange(t, 0, GlmDsaSyntheticModelBuilder.VocabSize - 1));
        _output.WriteLine("shared-indexer continuation: " + string.Join(' ', produced));
    }

    /// <summary>
    /// Same model, Q8_0 weights: covers the quantized matmul routers and
    /// <c>mul_mat_id</c> over quantized experts, which is what every real
    /// checkpoint uses. Golden from llama.cpp on the Q8_0 file.
    /// </summary>
    [Fact]
    public void GlmDsa_Quantized_MatchesLlamaCpp()
    {
        using var model = ModelBase.Create(BuildModel(quantize: true), BackendType.Cpu);
        int[] expected = { 84, 46, 77, 48, 3, 37, 9, 3 };
        Assert.Equal(expected, Greedy(model, Prompt24(), expected.Length));
    }

    [Fact]
    public void GlmDsa_ChatTemplate_MatchesModelFormat()
    {
        var messages = new System.Collections.Generic.List<ChatMessage>
        {
            new ChatMessage { Role = "user", Content = "hi" },
        };

        string thinking = ChatTemplate.RenderGlmDsa(messages, addGenerationPrompt: true, enableThinking: true);
        Assert.Equal("[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>hi<|assistant|><think>", thinking);

        string plain = ChatTemplate.RenderGlmDsa(messages, addGenerationPrompt: true, enableThinking: false);
        Assert.Equal("[gMASK]<sop><|user|>hi<|assistant|><think></think>", plain);
    }

    [Fact]
    public void GlmDsa_OutputParser_SplitsThinkingContentAndToolCalls()
    {
        var parser = OutputParserFactory.Create("glm-dsa");
        parser.Init(enableThinking: true, tools: null);

        var a = parser.Add("weighing options</think>Here you go.", done: false);
        Assert.Equal("weighing options", a.Thinking);
        Assert.Equal("Here you go.", a.Content);

        var b = parser.Add("<tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value>" +
                           "<arg_key>days</arg_key><arg_value>3</arg_value></tool_call>", done: true);
        Assert.NotNull(b.ToolCalls);
        var call = Assert.Single(b.ToolCalls);
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("Paris", call.Arguments["city"]);
        // tojson'd numbers come back as numbers, not the literal text.
        Assert.Equal(3L, call.Arguments["days"]);
    }

    [Fact]
    public void Glm5Next_OutputParser_ThinkFalse_KeepsTheAlwaysOpenReasoningOutOfContent()
    {
        // GLM-5.3-Flash's published template ends every generation prompt with an
        // open <think>, so a think:false reply still starts inside the reasoning block.
        var parser = OutputParserFactory.Create("glm5next");
        parser.Init(enableThinking: false, tools: null);

        var streamed = parser.Add("The user wants only the integer.\n\n17 + 25 = 42", done: false);
        Assert.Equal(string.Empty, streamed.Content);
        // Prose streams as reasoning while it is generated (clients see progress).
        Assert.StartsWith("The user wants", streamed.Thinking);
        var rest = parser.Add("</think>42", done: true);
        Assert.Equal("42", streamed.Content + rest.Content);
        Assert.DoesNotContain("</think>", streamed.Content + rest.Content);
        Assert.Contains("17 + 25 = 42", streamed.Thinking + rest.Thinking);
    }

    [Fact]
    public void Glm5Next_OutputParser_ThinkFalse_UnclosedReplyIsTheAnswer()
    {
        // response_format under think:false constrains from the first token, so the
        // reply never writes </think>: it is the answer, not reasoning.
        var parser = OutputParserFactory.Create("glm5next");
        parser.Init(enableThinking: false, tools: null);
        var first = parser.Add("{\"name\":", done: false);
        Assert.True(string.IsNullOrEmpty(first.Content + first.Thinking));
        var parsed = parser.Add(" \"Mars\"}", done: true);
        Assert.Equal("{\"name\": \"Mars\"}", first.Content + parsed.Content);
        Assert.True(string.IsNullOrEmpty(first.Thinking + parsed.Thinking));
    }

    [Fact]
    public void Glm5Next_OutputParser_ThinkTrue_Unchanged()
    {
        var parser = OutputParserFactory.Create("glm5next");
        parser.Init(enableThinking: true, tools: null);
        var a = parser.Add("reasoning", done: false);
        var b = parser.Add("</think>answer", done: true);
        Assert.Equal("reasoning", a.Thinking + b.Thinking);
        Assert.Equal("answer", a.Content + b.Content);
    }

    [Fact]
    public void GlmDsa_OutputParser_ThinkFalse_StillStartsInContent()
    {
        // glm-dsa (GLM-5.2 / 5.3) renders an immediately-closed <think></think> when
        // thinking is off, so its reply is content from the first token.
        var parser = OutputParserFactory.Create("glm-dsa");
        parser.Init(enableThinking: false, tools: null);
        var parsed = parser.Add("42", done: true);
        Assert.Equal("42", parsed.Content);
    }

    [Fact]
    public void Glm5Next_DeclaresAThinkingGrammarTrigger()
    {
        Assert.Equal("</think>", OutputParserFactory.GrammarActivationTrigger("glm5next", enableThinking: true));
        Assert.Null(OutputParserFactory.GrammarActivationTrigger("glm5next", enableThinking: false));
    }

    [Theory]
    [InlineData("glm5next", false)]
    [InlineData("glm5next", true)]
    [InlineData("glm-dsa", true)]
    public void Glm_OutputParser_StreamedCharByChar_ToolCallAfterReasoningIsParsed(string arch, bool enableThinking)
    {
        // The stream ends at the <|observation|> stop, straight after </tool_call>.
        const string reply = "\nThe user wants the weather.</think>\n<tool_call>get_weather"
            + "<arg_key>city</arg_key><arg_value>Paris</arg_value>"
            + "<arg_key>days</arg_key><arg_value>3</arg_value></tool_call>";
        var parser = OutputParserFactory.Create(arch);
        parser.Init(enableThinking, tools: null);
        string content = "", thinking = "";
        var calls = new List<ToolCall>();
        foreach (char c in reply)
        {
            var p = parser.Add(c.ToString(), done: false);
            content += p.Content; thinking += p.Thinking;
            if (p.ToolCalls != null) calls.AddRange(p.ToolCalls);
        }
        var last = parser.Add("", done: true);
        content += last.Content; thinking += last.Thinking;
        if (last.ToolCalls != null) calls.AddRange(last.ToolCalls);

        var call = Assert.Single(calls);
        Assert.Equal("get_weather", call.Name);
        Assert.Equal("Paris", call.Arguments["city"]);
        Assert.Equal(3L, call.Arguments["days"]);
        Assert.Equal("\nThe user wants the weather.", thinking);
        Assert.Equal("\n", content);
    }

    [Fact]
    public void Glm5Next_OutputParser_ThinkFalse_StreamedJsonAnswerIsContentAndPartialCloseTagIsHeld()
    {
        var json = OutputParserFactory.Create("glm5next");
        json.Init(enableThinking: false, tools: null);
        string content = "", thinking = "";
        foreach (string piece in new[] { "\n", " {", "\"a\"", ": 1", "}" })
        {
            var p = json.Add(piece, done: false);
            content += p.Content; thinking += p.Thinking;
        }
        var end = json.Add("", done: true);
        Assert.Equal("\n {\"a\": 1}", content + end.Content);
        Assert.Equal(string.Empty, thinking + end.Thinking);

        var prose = OutputParserFactory.Create("glm5next");
        prose.Init(enableThinking: false, tools: null);
        var a = prose.Add("ok</thi", done: false);
        Assert.Equal("ok", a.Thinking);
        var b = prose.Add("nk>42", done: true);
        Assert.Equal("42", a.Content + b.Content);
        Assert.Equal(string.Empty, b.Thinking);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void Glm5Next_FallbackRenderer_AlwaysOpensThinking_SoThinkFalseAnswersStayContent(bool enableThinking)
    {
        // An empty template takes the hardcoded fallback (as does a Jinja failure). The
        // published GLM-5.3-Flash template ends every generation prompt with an open
        // <think>, and the glm5next parser assumes it: a fallback that closed the block
        // under think:false made the model answer straight away, and that answer was
        // then parsed as hidden reasoning with an empty content.
        var messages = new List<ChatMessage> { new() { Role = "user", Content = "What is 17 + 25?" } };
        string prompt = ChatTemplate.RenderFromGgufTemplate(string.Empty, messages, addGenerationPrompt: true,
            architecture: "glm5next", tools: null, enableThinking: enableThinking);
        Assert.EndsWith("<|assistant|><think>", prompt);
        Assert.EndsWith(ChatProtocolRegistry.For("glm5next")!.AssistantGenerationSuffix!(enableThinking)!, prompt);

        var parser = OutputParserFactory.Create("glm5next");
        parser.Init(enableThinking, tools: null);
        var a = parser.Add("17 + 25 = 42.", done: false);
        var b = parser.Add("</think>42", done: true);
        Assert.Equal("42", a.Content + b.Content);
    }
}
