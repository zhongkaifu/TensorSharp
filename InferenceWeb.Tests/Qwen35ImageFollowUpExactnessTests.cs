// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Reuse past an image on Qwen 3.5 against REAL weights: a conversation
// text -> image -> text -> text whose turns continue the cache must produce the
// tokens (and, within tolerance, the logits) a cold prefill of the same prompts
// produces. Before the M-RoPE delta, decode after an image ran at the absolute KV
// index while a re-prefill used the compressed positions, so the two diverged and
// every reuse path had to stop at the first image.
//
// Three parts:
//  A. The engine. Turns 3 and 4 must reuse past the image (the whole previous turn),
//     and every turn's greedy tokens must equal a cold engine's.
//  B. The model directly, where logits are observable: a turn built by prefilling the
//     previous prompt, DECODING its reply and prefilling the new suffix, against a cold
//     prefill of the whole new prompt - logits within the tolerance below, same argmax,
//     for the first token and each of the following greedy steps.
//  C. A checkpoint of a cache that went through an image (non-zero delta), exported to
//     bytes, imported and cloned into a holder, decodes exactly like the cache it was
//     taken from; a version-1 file is refused.
//
// Opt-in:
//   TS_TEST_MODEL_DIR=<dir with Qwen3.5-9B*.gguf and its mmproj>
//   TS_TEST_QWEN35_MMPROJ=<mmproj path>   (optional when the dir holds one Qwen3.5-9B mmproj)
//   TS_TEST_QWEN35_IMAGE=<image path>     (optional; a synthetic picture otherwise)
//   TS_TEST_GGML_BACKEND=metal|cuda|cpu
//   TS_TEST_QWEN35_LOGIT_TOLERANCE=<max |dlogit|>  (optional; see LogitTolerance)
using System.Diagnostics;
using ImageMagick;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class Qwen35ImageFollowUpExactnessTests
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";
    private const string ModelPattern = "qwen3.5-9b-q8_0|qwen3.5-9b";
    private const int EngineNewTokens = 24;
    private const int DirectSteps = EngineNewTokens;
    private readonly ITestOutputHelper _output;

    public Qwen35ImageFollowUpExactnessTests(ITestOutputHelper output) { _output = output; }

    /// <summary>
    /// The documented bound for reuse-vs-cold logits on one backend: the reused turn's
    /// reply rows were written by the decode graph and the cold turn's by the prefill
    /// graph, which are different kernels (flash-attention decode vs batched prefill
    /// attention, quantized-activation vs float matmuls on CUDA, NeoX vs interleaved
    /// M-RoPE on equal axes), so the logits are close, not bit-identical.
    /// docs/models/qwen35.md records the measured values behind these defaults.
    /// </summary>
    private static float LogitTolerance(BackendType backend) =>
        float.TryParse(Environment.GetEnvironmentVariable("TS_TEST_QWEN35_LOGIT_TOLERANCE"),
            System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float t)
            ? t : backend == BackendType.GgmlMetal ? 0.1f : 3.0f;

    /// <summary>
    /// How far the image conversation's worst logit difference may exceed the text-only
    /// control's on the same backend. Reuse past an image is exact when the image adds no
    /// error of its own; the longer image context does amplify the same kernel
    /// differences somewhat (measured 2.3x on Metal, 2.6x on CUDA), while decoding at the
    /// pre-fix positions measured 310x on Metal.
    /// </summary>
    private static float ControlRatio =>
        float.TryParse(Environment.GetEnvironmentVariable("TS_TEST_QWEN35_CONTROL_RATIO"),
            System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out float r)
            ? r : 4.0f;

    [ModelFact(EnvModelDir, ModelPattern)]
    public async Task ReuseAfterAnImage_MatchesColdPrefill()
    {
        using var ctx = Context.Open(_output);
        if (ctx == null) return;
        // Every part runs and reports before anything is asserted, so a failure still
        // shows the logit margins that tell a real defect from a near tie.
        var failures = new List<string>();

        // ---------- A. the engine ----------
        var turns = new List<Turn>();
        var reuseOutputs = new List<List<int>>();
        var reused = new List<int>();
        using (var engine = new InferenceEngine(ctx.Model, Config(), NullLogger.Instance))
        {
            List<int> previous = null;
            List<int> previousOut = null;
            for (int k = 0; k < 4; k++)
            {
                var turn = ctx.BuildTurn(k, previous, previousOut);
                turns.Add(turn);
                var (completion, output) = await ctx.GenerateAsync(engine, turn, $"reuse-{k}");
                reuseOutputs.Add(output);
                reused.Add(completion.PrefixCacheReusedTokens);
                _output.WriteLine($"[reuse t{k + 1}] prompt {turn.Expanded.Count} reused {completion.PrefixCacheReusedTokens} " +
                                  $"({100.0 * completion.PrefixCacheReusedTokens / turn.Expanded.Count:F1}%): {ctx.Decode(output)}");
                previous = turn.Unexpanded;
                previousOut = output;
            }
        }

        var engineDivergence = new int[4];
        for (int k = 0; k < 4; k++)
        {
            using var engine = new InferenceEngine(ctx.Model, Config(), NullLogger.Instance);
            var (completion, output) = await ctx.GenerateAsync(engine, turns[k], $"cold-{k}");
            Assert.Equal(0, completion.PrefixCacheReusedTokens);
            _output.WriteLine($"[cold  t{k + 1}] {ctx.Decode(output)}");
            engineDivergence[k] = FirstDifference(reuseOutputs[k], output);
            if (engineDivergence[k] >= 0)
                _output.WriteLine($"[engine t{k + 1}] reused and cold greedy tokens first differ at step {engineDivergence[k]}");
        }

        int imageEnd = turns[1].ImageSpanEnd;
        Assert.True(imageEnd > 0, "turn 2 carries the image");
        for (int k = 2; k < 4; k++)
        {
            int previousCached = turns[k - 1].Expanded.Count + reuseOutputs[k - 1].Count - 1;
            if (reused[k] <= imageEnd)
                failures.Add($"turn {k + 1} reused {reused[k]} tokens, which stops before the image ends at {imageEnd}");
            if (reused[k] < previousCached)
                failures.Add($"turn {k + 1} reused {reused[k]} tokens; the previous turn left {previousCached} in the cache");
        }

        // ---------- B. the model, logits ----------
        // A turn built by prefilling the previous prompt, DECODING its reply and
        // prefilling the new suffix, against a cold prefill of the whole prompt.
        var image = CompareDirect(ctx, "image", turns, reuseOutputs);

        // A text-only conversation through the same comparison: what the backend's
        // decode and prefill kernels differ by when no image is involved. Reuse past an
        // image is exact when it adds nothing to that.
        var textTurns = new List<Turn>();
        var textOutputs = new List<List<int>>();
        for (int k = 0; k < 4; k++)
        {
            var turn = ctx.BuildTurn(k, k == 0 ? null : textTurns[k - 1].Unexpanded, k == 0 ? null : textOutputs[k - 1],
                Context.TextOnlyUserTexts, imageTurn: -1);
            textTurns.Add(turn);
            textOutputs.Add(ctx.RunDirect(turn, prefix: null, prefixOutput: null, EngineNewTokens).Tokens);
        }
        var text = CompareDirect(ctx, "text control", textTurns, textOutputs);

        float bound = LogitTolerance(ctx.Backend);
        float ratio = text.Worst > 0 ? image.Worst / text.Worst : float.PositiveInfinity;
        _output.WriteLine($"[direct] worst max |dlogit|: image {image.Worst:F5}, text control {text.Worst:F5} " +
                          $"(ratio {ratio:F2}, allowed {ControlRatio}); tolerance {bound}");
        if (image.Worst > bound)
            failures.Add($"reuse after an image: max |dlogit| {image.Worst} exceeds the {ctx.Backend} tolerance {bound} (text control {text.Worst})");
        if (text.Worst > bound)
            failures.Add($"text control: max |dlogit| {text.Worst} exceeds the {ctx.Backend} tolerance {bound}");
        if (image.Worst > ControlRatio * text.Worst)
            failures.Add($"reuse after an image: max |dlogit| {image.Worst} is {ratio:F1}x the text-only control's {text.Worst} (allowed {ControlRatio}x)");

        // Greedy tokens: identical, except where the cold run's top-2 margin at the
        // first differing step is inside the logit difference measured there - a tie the
        // kernels may break either way, not a position error.
        for (int k = 1; k < 4; k++)
        {
            int s = engineDivergence[k];
            if (s < 0)
                continue;
            if (!image.StepStats.TryGetValue((k, s), out var st))
            {
                failures.Add($"turn {k + 1}: engine tokens differ at step {s}, beyond the {DirectSteps} compared logit steps");
                continue;
            }
            if (st.Margin >= st.Diff)
                failures.Add($"turn {k + 1}: engine tokens differ at step {s} although the cold top-2 margin {st.Margin} exceeds the logit difference {st.Diff}");
            else
                _output.WriteLine($"[tie] turn {k + 1} step {s}: cold top-2 margin {st.Margin:F4} < max |dlogit| {st.Diff:F4}");
        }
        if (engineDivergence[0] >= 0)
            failures.Add($"turn 1 has nothing to reuse, yet its tokens differ at step {engineDivergence[0]}");
        failures.AddRange(image.Failures);

        // ---------- C. a checkpoint through an image, to bytes and back ----------
        if (ctx.Model is IBatchedPagedModel paged && paged.SupportsRetainedCacheSerialization)
            ctx.CheckpointRoundTrip(paged, turns[2], reuseOutputs[2]);
        else
            _output.WriteLine("[checkpoint] backend has no serializable checkpoints; part C not run");

        foreach (var f in failures) _output.WriteLine("[failure] " + f);
        Assert.Empty(failures);
    }

    private sealed class Comparison
    {
        public float Worst;
        public readonly Dictionary<(int Turn, int Step), (float Diff, float Margin)> StepStats = new();
        public readonly List<string> Failures = new();
    }

    private Comparison CompareDirect(Context ctx, string label, List<Turn> turns, List<List<int>> outputs)
    {
        var result = new Comparison();
        for (int k = 1; k < turns.Count; k++)
        {
            var cold = ctx.RunDirect(turns[k], prefix: null, prefixOutput: null, DirectSteps);
            var warm = ctx.RunDirect(turns[k], turns[k - 1], outputs[k - 1], DirectSteps);
            var diffs = new List<string>();
            for (int s = 0; s < cold.Logits.Count; s++)
            {
                float d = MaxAbsDiff(cold.Logits[s], warm.Logits[s]);
                float margin = TopMargin(cold.Logits[s]);
                result.Worst = Math.Max(result.Worst, d);
                result.StepStats[(k, s)] = (d, margin);
                diffs.Add($"{d:F3}/{margin:F2}");
                if (cold.Tokens[s] != warm.Tokens[s])
                {
                    if (margin >= d)
                        result.Failures.Add($"{label} turn {k + 1} step {s}: argmax differs although the cold top-2 margin {margin} exceeds max |dlogit| {d}");
                    break;   // past a divergence the two runs decode different tokens
                }
            }
            _output.WriteLine($"[direct {label} t{k + 1}] suffix {turns[k].Expanded.Count - (turns[k - 1].Expanded.Count + outputs[k - 1].Count - 1)} tokens, " +
                              $"max |dlogit| / cold top-2 margin per step: {string.Join(" ", diffs)}");
        }
        return result;
    }

    private static int FirstDifference(List<int> a, List<int> b)
    {
        int n = Math.Min(a.Count, b.Count);
        for (int i = 0; i < n; i++)
            if (a[i] != b[i]) return i;
        return a.Count == b.Count ? -1 : n;
    }

    private static float TopMargin(float[] a)
    {
        float first = float.NegativeInfinity, second = float.NegativeInfinity;
        foreach (float v in a)
        {
            if (v > first) { second = first; first = v; }
            else if (v > second) second = v;
        }
        return first - second;
    }

    /// <summary>
    /// The concurrent paths: an image conversation and a text conversation run side by
    /// side, so each decodes through its own per-request holder (and, on CUDA and Metal,
    /// the arena batched decode, whose RoPE positions come from each holder's delta), and
    /// each follow-up continues the holder its previous turn retained. Every turn must
    /// reuse the previous one past the image and produce the tokens a cold solo prefill
    /// of the same prompt produces.
    /// </summary>
    [ModelFact(EnvModelDir, ModelPattern)]
    public async Task ConcurrentReuseAfterAnImage_MatchesColdPrefill()
    {
        using var ctx = Context.Open(_output);
        if (ctx == null) return;
        if (ctx.Backend is not (BackendType.GgmlMetal or BackendType.GgmlCuda))
        {
            // Holders, their retention and the arena exist on CUDA and Metal only. Elsewhere
            // concurrency runs the paged batched forward or the KV swap, which retain no
            // conversation state to reuse, and the paged batched decode of a TEXT request can
            // differ from a solo decode on a near tie (ggml_cpu, measured) - neither is what
            // this test is about.
            _output.WriteLine($"[concurrent] {ctx.Backend} has no per-request holders; test not applicable");
            return;
        }

        var imageTurns = new List<Turn>();
        var textTurns = new List<Turn>();
        var imageOut = new List<List<int>>();
        var textOut = new List<List<int>>();
        var imageReused = new List<int>();
        using (var engine = new InferenceEngine(ctx.Model, Config(), NullLogger.Instance))
        {
            for (int k = 0; k < 4; k++)
            {
                var a = ctx.BuildTurn(k, k == 0 ? null : imageTurns[k - 1].Unexpanded, k == 0 ? null : imageOut[k - 1],
                    Context.UserTexts, imageTurn: 1);
                var b = ctx.BuildTurn(k, k == 0 ? null : textTurns[k - 1].Unexpanded, k == 0 ? null : textOut[k - 1],
                    Context.TextOnlyUserTexts, imageTurn: -1);
                imageTurns.Add(a);
                textTurns.Add(b);
                var ta = ctx.GenerateAsync(engine, a, $"conc-img-{k}", scope: "conversation-image");
                var tb = ctx.GenerateAsync(engine, b, $"conc-txt-{k}", scope: "conversation-text");
                await Task.WhenAll(ta, tb);
                imageOut.Add(ta.Result.output);
                textOut.Add(tb.Result.output);
                imageReused.Add(ta.Result.completion.PrefixCacheReusedTokens);
                _output.WriteLine($"[concurrent t{k + 1}] image prompt {a.Expanded.Count} reused {ta.Result.completion.PrefixCacheReusedTokens}: {ctx.Decode(ta.Result.output)}");
                _output.WriteLine($"[concurrent t{k + 1}] text  prompt {b.Expanded.Count} reused {tb.Result.completion.PrefixCacheReusedTokens}: {ctx.Decode(tb.Result.output)}");
            }
        }

        // The arena batched decode is where each sequence's RoPE position comes from its
        // holder's delta; say whether it served these steps (CUDA and Metal only).
        long arenaSteps = ctx.Model is Qwen35Model q35 ? q35.ArenaBatchedDecodeSteps : 0;
        _output.WriteLine($"[concurrent] arena batched decode steps: {arenaSteps}");
        if (ctx.Backend is BackendType.GgmlMetal or BackendType.GgmlCuda)
            Assert.True(arenaSteps > 0, "the concurrent turns never reached the arena batched decode");

        var mismatches = new List<string>();
        for (int k = 0; k < 4; k++)
        {
            foreach (var (turn, output, label) in new[] { (imageTurns[k], imageOut[k], "image"), (textTurns[k], textOut[k], "text") })
            {
                using var engine = new InferenceEngine(ctx.Model, Config(), NullLogger.Instance);
                var (completion, cold) = await ctx.GenerateAsync(engine, turn, $"conc-cold-{label}-{k}");
                Assert.Equal(0, completion.PrefixCacheReusedTokens);
                if (!cold.SequenceEqual(output))
                    mismatches.Add($"{label} turn {k + 1}: concurrent {ctx.Decode(output)} | cold {ctx.Decode(cold)}");
            }
        }
        foreach (var m in mismatches) _output.WriteLine("[mismatch] " + m);
        Assert.Empty(mismatches);

        int imageEnd = imageTurns[1].ImageSpanEnd;
        for (int k = 2; k < 4; k++)
            Assert.True(imageReused[k] > imageEnd,
                $"concurrent image turn {k + 1} reused {imageReused[k]} tokens, which stops before the image ends at {imageEnd}");
    }

    private static SchedulerConfig Config() => new()
    {
        MaxNumBatchedTokens = 4096,
        MaxNumRunningSequences = 4,
        MaxPrefillChunkSize = 1024,
        SoloPrefillChunkSize = 1024,
        NumBlocks = 128,
        BlockSize = 256,
        EnablePrefixCaching = true,
        DecodeQuantumTokens = 256,
    };

    private static float MaxAbsDiff(float[] a, float[] b)
    {
        float m = 0;
        for (int i = 0; i < a.Length; i++) m = Math.Max(m, Math.Abs(a[i] - b[i]));
        return m;
    }

    private static int ArgMax(float[] a)
    {
        int best = 0;
        for (int i = 1; i < a.Length; i++) if (a[i] > a[best]) best = i;
        return best;
    }

    private sealed record Turn(int Index, List<int> Unexpanded, List<int> Expanded, List<ChatMessage> History, int ImageSpanEnd);

    private sealed class DirectRun
    {
        public List<int> Tokens { get; } = new();
        public List<float[]> Logits { get; } = new();
    }

    private sealed class Context : IDisposable
    {
        private readonly ITestOutputHelper _output;
        private readonly string _tempDir;
        private readonly string _image;
        private int _prepSerial;

        public ModelBase Model { get; }
        public BackendType Backend { get; }

        private Context(ITestOutputHelper output, ModelBase model, BackendType backend, string image, string tempDir)
        {
            _output = output;
            Model = model;
            Backend = backend;
            _image = image;
            _tempDir = tempDir;
        }

        public static Context Open(ITestOutputHelper output)
        {
            string dir = Environment.GetEnvironmentVariable(EnvModelDir);
            string modelPath = dir == null ? null : TestGates.FindGguf(dir, ModelPattern);
            if (modelPath == null) { output.WriteLine("no Qwen3.5-9B model; skipping"); return null; }
            string mmproj = Environment.GetEnvironmentVariable("TS_TEST_QWEN35_MMPROJ");
            if (string.IsNullOrEmpty(mmproj))
            {
                var candidates = Directory.GetFiles(Path.GetDirectoryName(modelPath)!, "*.gguf")
                    .Where(p => Path.GetFileName(p).Contains("mmproj", StringComparison.OrdinalIgnoreCase))
                    .ToList();
                mmproj = candidates.FirstOrDefault(p => Path.GetFileName(p).Contains("Qwen3.5-9B", StringComparison.OrdinalIgnoreCase))
                    ?? (candidates.Count == 1 ? candidates[0] : null);
            }
            Assert.True(mmproj != null && File.Exists(mmproj),
                "The Qwen3.5-9B vision projector was not found: set TS_TEST_QWEN35_MMPROJ.");

            BackendType backend = (Environment.GetEnvironmentVariable("TS_TEST_GGML_BACKEND") ?? "cpu")
                .Trim().ToLowerInvariant() switch
            {
                "metal" => BackendType.GgmlMetal,
                "cuda" => BackendType.GgmlCuda,
                _ => BackendType.GgmlCpu,
            };

            string temp = Path.Combine(Path.GetTempPath(), "q35-img-exact-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(temp);
            string image = Environment.GetEnvironmentVariable("TS_TEST_QWEN35_IMAGE");
            if (string.IsNullOrEmpty(image))
                image = WriteSyntheticImage(Path.Combine(temp, "scene.png"));

            var sw = Stopwatch.StartNew();
            var model = ModelBase.Create(modelPath, backend);
            model.MultimodalInjector.LoadProjectors(mmproj);
            output.WriteLine($"model {Path.GetFileName(modelPath)} + {Path.GetFileName(mmproj)} on {backend}, image {image}, loaded in {sw.Elapsed.TotalSeconds:F1}s");
            return new Context(output, model, backend, image, temp);
        }

        private static string WriteSyntheticImage(string path)
        {
            // Something to describe: a sky, a sun, a house and a lawn.
            using var img = new MagickImage(new MagickColor("#87CEEB"), 448, 336);
            new ImageMagick.Drawing.Drawables()
                .FillColor(new MagickColor("#2E8B57")).Rectangle(0, 240, 448, 336)
                .FillColor(new MagickColor("#FFD700")).Circle(370, 70, 370, 110)
                .FillColor(new MagickColor("#B22222")).Rectangle(120, 150, 260, 250)
                .FillColor(new MagickColor("#8B4513")).Polygon(new PointD(110, 150), new PointD(190, 90), new PointD(270, 150))
                .Draw(img);
            img.Write(path, MagickFormat.Png);
            return path;
        }

        private int Special(string token)
        {
            int id = Model.Tokenizer.LookupToken(token);
            Assert.True(id >= 0, $"the tokenizer has no {token} token");
            return id;
        }

        private List<int> Text(string text) => Model.Tokenizer.Encode(text, addSpecial: false);

        public static readonly string[] UserTexts =
        {
            "Name three primary colors, one per line.",
            "Describe the picture in two sentences.",
            "What is the most prominent color in the picture, and where is it?",
            "Now suggest a short title for the picture.",
        };

        public static readonly string[] TextOnlyUserTexts =
        {
            "List three rivers in Europe.",
            "Which of them is the longest?",
            "Name one city on that river.",
            "Give one fact about that city.",
        };

        /// <summary>Turn <paramref name="k"/>'s prompt at the token level: the previous
        /// prompt, the previous reply's raw tokens (closed with im_end when it did not
        /// end there) and the new user message with the generation prompt. Turn 2
        /// carries the image. Token-level concatenation makes every turn an exact
        /// extension of the one before, which is what reuse needs.</summary>
        public Turn BuildTurn(int k, List<int> previous, List<int> previousOutput)
            => BuildTurn(k, previous, previousOutput, UserTexts, imageTurn: 1);

        public Turn BuildTurn(int k, List<int> previous, List<int> previousOutput, string[] userTexts, int imageTurn)
        {
            int imStart = Special("<|im_start|>"), imEnd = Special("<|im_end|>");
            var tokens = new List<int>();
            if (previous == null)
            {
                tokens.Add(imStart);
                tokens.AddRange(Text("system\nYou are a helpful assistant. Answer briefly."));
                tokens.Add(imEnd);
                tokens.AddRange(Text("\n"));
            }
            else
            {
                tokens.AddRange(previous);
                tokens.AddRange(previousOutput);
                if (previousOutput.Count == 0 || previousOutput[^1] != imEnd)
                    tokens.Add(imEnd);
                tokens.AddRange(Text("\n"));
            }
            tokens.Add(imStart);
            tokens.AddRange(Text("user\n"));
            var history = new List<ChatMessage>();
            if (imageTurn >= 0 && k >= imageTurn)
                history.Add(new ChatMessage { Role = "user", Content = userTexts[imageTurn], ImagePaths = new List<string> { _image } });
            if (k == imageTurn)
            {
                tokens.Add(Special("<|vision_start|>"));
                tokens.Add(Special("<|image_pad|>"));
                tokens.Add(Special("<|vision_end|>"));
            }
            tokens.AddRange(Text(userTexts[k]));
            tokens.Add(imEnd);
            tokens.AddRange(Text("\n"));
            tokens.Add(imStart);
            tokens.AddRange(Text("assistant\n"));
            int thinkOpen = Model.Tokenizer.LookupToken("<think>");
            int thinkClose = Model.Tokenizer.LookupToken("</think>");
            if (thinkOpen >= 0 && thinkClose >= 0)
            {
                tokens.Add(thinkOpen);
                tokens.AddRange(Text("\n\n"));
                tokens.Add(thinkClose);
                tokens.AddRange(Text("\n\n"));
            }

            string probe = $"turn-probe-{k}-{_prepSerial++}";
            List<int> expanded;
            int imageEnd = 0;
            lock (Model.GpuComputeLock)
            {
                expanded = history.Count == 0
                    ? new List<int>(tokens)
                    : Model.MultimodalInjector.ProcessPromptTokens(history, new List<int>(tokens), probe);
                foreach (var span in Model.MultimodalInjector.GetPreparedMediaSpans(probe))
                    imageEnd = Math.Max(imageEnd, span.End);
                Model.MultimodalInjector.ClearPreparedPromptState(probe);
            }
            return new Turn(k, tokens, expanded, history, imageEnd);
        }

        /// <summary>Prepare <paramref name="turn"/> for one request: expanded tokens and
        /// media spans in the injector bucket <paramref name="requestId"/>.</summary>
        private (List<int> Tokens, IReadOnlyList<PromptMediaSpan> Spans) Prepare(Turn turn, string requestId)
        {
            lock (Model.GpuComputeLock)
            {
                if (turn.History.Count == 0)
                    return (new List<int>(turn.Unexpanded), Array.Empty<PromptMediaSpan>());
                var tokens = Model.MultimodalInjector.ProcessPromptTokens(turn.History, new List<int>(turn.Unexpanded), requestId);
                Assert.Equal(turn.Expanded, tokens);
                return (tokens, Model.MultimodalInjector.GetPreparedMediaSpans(requestId));
            }
        }

        public async Task<(InferenceCompletion completion, List<int> output)> GenerateAsync(
            InferenceEngine engine, Turn turn, string requestId, string scope = "q35-image-conversation")
        {
            var (tokens, spans) = Prepare(turn, requestId);
            try
            {
                var seq = new SequenceState(requestId, tokens, EngineNewTokens, 256, SamplingConfig.Greedy,
                    mediaSpans: spans, cacheScope: scope);
                var handle = engine.SubmitRequest(seq);
                var output = new List<int>();
                await foreach (int t in handle.Tokens.ReadAllAsync())
                    output.Add(t);
                var completion = await handle.Completion;
                return (completion, output);
            }
            finally
            {
                Model.MultimodalInjector.ClearPreparedPromptState(requestId);
            }
        }

        private float[] ForwardSlice(List<int> tokens, int start, int count, string requestId)
        {
            Model.MultimodalInjector.QueuePromptEmbeddingsForSlice(start, count, requestId);
            return (float[])Model.Forward(tokens.GetRange(start, count).ToArray()).Clone();
        }

        /// <summary>Greedy steps of <paramref name="turn"/> straight on the model. With a
        /// <paramref name="prefix"/>, the cache is first built the way a conversation
        /// builds it - the previous prompt prefilled, its reply DECODED token by token
        /// (all but the last sampled token) - and only the new suffix is prefilled.</summary>
        public DirectRun RunDirect(Turn turn, Turn prefix, List<int> prefixOutput, int steps)
        {
            var run = new DirectRun();
            string id = $"direct-{turn.Index}-{_prepSerial++}";
            string prefixId = $"direct-prefix-{turn.Index}-{_prepSerial++}";
            var (tokens, _) = Prepare(turn, id);
            try
            {
                lock (Model.GpuComputeLock)
                {
                    Model.ResetKVCache();
                    int cached = 0;
                    if (prefix != null)
                    {
                        var (prefixTokens, _) = Prepare(prefix, prefixId);
                        ForwardSlice(prefixTokens, 0, prefixTokens.Count, prefixId);
                        for (int i = 0; i < prefixOutput.Count - 1; i++)
                            Model.Forward(new[] { prefixOutput[i] });
                        cached = prefixTokens.Count + prefixOutput.Count - 1;
                        for (int i = 0; i < cached; i++)
                        {
                            int expected = i < prefixTokens.Count ? prefixTokens[i] : prefixOutput[i - prefixTokens.Count];
                            Assert.Equal(expected, tokens[i]);
                        }
                    }
                    float[] logits = ForwardSlice(tokens, cached, tokens.Count - cached, id);
                    for (int s = 0; s < steps; s++)
                    {
                        run.Logits.Add(logits);
                        int next = ArgMax(logits);
                        run.Tokens.Add(next);
                        if (s + 1 < steps)
                            logits = (float[])Model.Forward(new[] { next }).Clone();
                    }
                }
            }
            finally
            {
                Model.MultimodalInjector.ClearPreparedPromptState(id);
                Model.MultimodalInjector.ClearPreparedPromptState(prefixId);
            }
            return run;
        }

        /// <summary>Part C: checkpoint the primary cache after an image turn (non-zero
        /// M-RoPE delta), continue the primary as the reference, then export the
        /// checkpoint, import it under a new key, clone it into a holder and decode the
        /// same tokens there.</summary>
        public void CheckpointRoundTrip(IBatchedPagedModel paged, Turn turn, List<int> reply)
        {
            string id = $"ckpt-{_prepSerial++}";
            var (tokens, _) = Prepare(turn, id);
            try
            {
                lock (Model.GpuComputeLock)
                {
                    Model.ResetKVCache();
                    ForwardSlice(tokens, 0, tokens.Count, id);
                    for (int i = 0; i < 4 && i < reply.Count - 1; i++)
                        Model.Forward(new[] { reply[i] });
                    Assert.True(paged.TryCheckpointActiveCache("q35-img-ckpt"), "checkpoint of the primary cache");

                    var reference = new List<float[]>();
                    int next = reply[Math.Min(4, reply.Count - 1)];
                    var forced = new List<int>();
                    for (int s = 0; s < 6; s++)
                    {
                        forced.Add(next);
                        float[] l = (float[])Model.Forward(new[] { next }).Clone();
                        reference.Add(l);
                        next = ArgMax(l);
                    }

                    var bytes = new MemoryStream();
                    Assert.True(paged.TryExportRetainedCache("q35-img-ckpt", bytes), "export");
                    byte[] payload = bytes.ToArray();
                    paged.DiscardRetainedCache("q35-img-ckpt");

                    // A version-1 file (no delta) is refused.
                    byte[] v1 = (byte[])payload.Clone();
                    BitConverter.GetBytes(1).CopyTo(v1, 4);
                    Assert.False(paged.TryImportRetainedCache("q35-img-ckpt-v1", new MemoryStream(v1)), "a version-1 checkpoint must be refused");

                    Assert.True(paged.TryImportRetainedCache("q35-img-ckpt-restored", new MemoryStream(payload)), "import");
                    Assert.True(paged.TryCloneRetainedCache("q35-img-ckpt-restored", "q35-img-ckpt-clone"), "clone");
                    paged.BindSequenceCache("q35-img-ckpt-clone");
                    float worst = 0;
                    try
                    {
                        for (int s = 0; s < forced.Count; s++)
                        {
                            float[] l = Model.Forward(new[] { forced[s] });
                            float d = MaxAbsDiff(reference[s], l);
                            worst = Math.Max(worst, d);
                            Assert.Equal(ArgMax(reference[s]), ArgMax(l));
                            Assert.True(d <= LogitTolerance(Backend), $"restored checkpoint step {s}: max |dlogit| {d}");
                        }
                    }
                    finally
                    {
                        paged.RestorePrimaryCache();
                        paged.OnSequenceReleased("q35-img-ckpt-clone");
                        paged.DiscardRetainedCache("q35-img-ckpt-restored");
                    }
                    _output.WriteLine($"[checkpoint] {payload.Length / 1048576.0:F1} MB file through an image; restored decode max |dlogit| {worst:F5} over {forced.Count} steps; version-1 file refused");
                }
            }
            finally
            {
                Model.MultimodalInjector.ClearPreparedPromptState(id);
            }
        }

        public string Decode(List<int> tokens)
        {
            try { return Model.Tokenizer.Decode(tokens).Replace("\n", "\\n"); }
            catch { return string.Join(",", tokens); }
        }

        public void Dispose()
        {
            Model.Dispose();
            try { Directory.Delete(_tempDir, true); } catch { }
        }
    }
}
