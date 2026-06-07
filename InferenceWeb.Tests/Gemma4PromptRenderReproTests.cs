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

    [Fact]
    public void RealTemplate_RendersUserText_AndSingleBos()
    {
        string? modelPath = FindModel();
        if (modelPath == null) { _output.WriteLine("gemma-4-12b GGUF not found; skipping"); return; }

        using var gguf = new GgufFile(modelPath);
        var vocab = gguf.GetStringArray("tokenizer.ggml.tokens")!;
        var tokenTypes = gguf.GetInt32Array("tokenizer.ggml.token_type")!;
        var scores = gguf.GetFloatArray("tokenizer.ggml.scores")!;
        int bosId = (int)gguf.GetUint32("tokenizer.ggml.bos_token_id");
        int eosId = (int)gguf.GetUint32("tokenizer.ggml.eos_token_id");
        bool addBosMeta = gguf.GetBool("tokenizer.ggml.add_bos_token", false);
        string chatTemplate = gguf.GetString("tokenizer.chat_template") ?? "";
        // Mirror production BOS resolution: a template with a leading {{ bos_token }}
        // requires a BOS even when add_bos_token=false (TensorSharp renders bos_token
        // empty, so the tokenizer must own it). See ModelBase.ResolveAddBosToken.
        bool addBos = ModelBase.ResolveAddBosToken(addBosMeta, bosId, chatTemplate);
        var tokenizer = new SentencePieceTokenizer(vocab, tokenTypes, scores, bosId,
            new[] { eosId }, addBos, false);

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

    [Fact]
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

    [Fact]
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
    [Fact]
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

    private static int ArgMax(float[] a)
    {
        int best = 0;
        for (int i = 1; i < a.Length; i++) if (a[i] > a[best]) best = i;
        return best;
    }
}
