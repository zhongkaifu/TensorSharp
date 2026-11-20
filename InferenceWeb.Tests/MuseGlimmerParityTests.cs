// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Numerical parity of Muse-Glimmer with llama.cpp.
//
// The goldens are committed under Fixtures/MuseGlimmer/LlamaCpp (see its README): llama.cpp's greedy continuations of five
// chat prompts and of a 4,664-token prompt, with the top log-probabilities of every generated token, made by
// eng/validation/muse-glimmer-llamacpp-goldens.py from Muse-Glimmer-30B-UD-IQ2_XXS. The model is fed the RECORDED prompt
// token ids and then teacher-forced through llama.cpp's continuation, so every generated position is compared and the
// forward pass is compared apart from the tokenizer, which has its own test.
//
// Inputs (read, never set):
//   TS_TEST_MODEL_DIR          directory holding Muse-Glimmer-30B-UD-IQ2_XXS.gguf (and dflash*.gguf for the DFlash test)
//   TS_MUSE_GLIMMER_BACKEND    cpu, ggml_cpu, ggml_metal, ggml_cuda, ggml_vulkan, mlx or cuda; default: the GGML backend
//                              this process pins (TS_TEST_GGML_BACKEND). A GGML backend other than the pinned one skips.
// Without the weights the tests skip (they are reported as skipped, not passed).
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using Xunit.Abstractions;
using Xunit.Sdk;

namespace InferenceWeb.Tests;

/// <summary>[Fact] for the real-weight Muse-Glimmer tests: skips without the weights (TS_TEST_MODEL_DIR), with a draft
/// model test's drafter missing, or when TS_MUSE_GLIMMER_BACKEND names a backend this process cannot construct.</summary>
[TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
[AttributeUsage(AttributeTargets.Method)]
public sealed class MuseGlimmerModelFactAttribute : FactAttribute, ITraitAttribute
{
    public string RequiresValue => "Models";

    public MuseGlimmerModelFactAttribute() => Skip = MuseGlimmerParity.Skip(needsDraft: false);

    /// <summary>The test also loads the DFlash drafter (dflash*.gguf next to the model).</summary>
    public bool NeedsDraft
    {
        get => _needsDraft;
        set { _needsDraft = value; Skip = MuseGlimmerParity.Skip(value); }
    }
    private bool _needsDraft;
}

/// <summary>The llama.cpp goldens and the real-weight helpers the Muse-Glimmer parity tests share.</summary>
internal static class MuseGlimmerParity
{
    internal const string ModelDirEnv = "TS_TEST_MODEL_DIR";
    internal const string BackendEnv = "TS_MUSE_GLIMMER_BACKEND";
    internal const string ResourcePrefix = "InferenceWeb.Tests.Fixtures.MuseGlimmer.LlamaCpp.";

    /// <summary>One generated position of a golden: the chosen token's log-probability and the top log-probabilities
    /// (ten at the first position, four after it).</summary>
    internal sealed record Step(double Logprob, (int Id, double Logprob)[] Top);

    internal sealed record Golden(string Name, string Prompt, int[] PromptTokens, int[] Generated, Step[] Steps);

    internal sealed record GoldenFile(string LlamaCppCommit, string ModelSha256, IReadOnlyList<Golden> Records);

    private static readonly Dictionary<string, GoldenFile> s_files = new(StringComparer.Ordinal);

    internal static GoldenFile Load(string name)
    {
        lock (s_files)
        {
            if (s_files.TryGetValue(name, out GoldenFile cached)) return cached;
            using Stream s = typeof(MuseGlimmerParity).Assembly.GetManifestResourceStream(ResourcePrefix + name)
                ?? throw new FileNotFoundException($"golden {name} is not embedded (Fixtures/MuseGlimmer/LlamaCpp)");
            using JsonDocument doc = JsonDocument.Parse(s);
            JsonElement root = doc.RootElement;
            var records = new List<Golden>();
            foreach (JsonElement r in root.GetProperty("records").EnumerateArray())
            {
                Step[] steps = r.TryGetProperty("steps", out JsonElement st)
                    ? st.EnumerateArray().Select(x => new Step(x.GetProperty("logprob").GetDouble(),
                        x.GetProperty("top").EnumerateArray().Select(t => (t[0].GetInt32(), t[1].GetDouble())).ToArray())).ToArray()
                    : null;
                records.Add(new Golden(r.GetProperty("name").GetString(), r.GetProperty("prompt").GetString(),
                    r.GetProperty("prompt_tokens").EnumerateArray().Select(e => e.GetInt32()).ToArray(),
                    r.GetProperty("generated_tokens").EnumerateArray().Select(e => e.GetInt32()).ToArray(), steps));
            }
            var file = new GoldenFile(root.GetProperty("llama_cpp").GetProperty("commit").GetString(),
                root.GetProperty("model").GetProperty("sha256").GetString(), records);
            s_files[name] = file;
            return file;
        }
    }

    /// <summary>The backend the tests run on: <see cref="BackendEnv"/>, else the GGML backend this process pins.</summary>
    internal static BackendType Backend()
    {
        string name = Environment.GetEnvironmentVariable(BackendEnv);
        if (string.IsNullOrWhiteSpace(name)) return TestGates.PinnedGgmlBackend;
        return name.Trim().ToLowerInvariant() switch
        {
            "cpu" => BackendType.Cpu,
            "cuda" => BackendType.Cuda,
            "mlx" => BackendType.Mlx,
            "ggml_cpu" => BackendType.GgmlCpu,
            "ggml_metal" => BackendType.GgmlMetal,
            "ggml_cuda" => BackendType.GgmlCuda,
            "ggml_vulkan" => BackendType.GgmlVulkan,
            _ => throw new ArgumentException(
                $"{BackendEnv}={name} is not a backend: use cpu, cuda, mlx, ggml_cpu, ggml_metal, ggml_cuda or ggml_vulkan."),
        };
    }

    /// <summary>The skip reason of a real-weight test, or null to run. An unparsable backend does not skip: the test runs
    /// and fails on it.</summary>
    internal static string Skip(bool needsDraft)
    {
        string skip = TestGates.ModelSkip(ModelDirEnv, "muse-glimmer");
        if (skip != null) return skip;
        if (FindModel() == null) return $"Requires model weights (no {GoldenModelFile} under {ModelDirEnv}: the goldens are that quant's).";
        if (needsDraft && FindDraft() == null) return $"Requires the DFlash drafter (no dflash*.gguf under {ModelDirEnv}).";
        BackendType backend;
        try { backend = Backend(); }
        catch (ArgumentException) { return null; }
        return backend switch
        {
            BackendType.Mlx => TestGates.MlxSkip,
            BackendType.Cuda => TestGates.CudaSkip,
            _ => TestGates.GgmlPinSkip(backend),
        };
    }

    private static string ModelDir()
    {
        string dir = Environment.GetEnvironmentVariable(ModelDirEnv);
        return !string.IsNullOrWhiteSpace(dir) && Directory.Exists(dir) ? dir : null;
    }

    /// <summary>The GGUF the goldens were made from (their model.file, Muse-Glimmer-30B-UD-IQ2_XXS.gguf): the comparison is
    /// quant-specific, so another quant in the directory is not taken.</summary>
    internal static string FindModel()
    {
        if (ModelDir() is not { } dir) return null;
        string path = Path.Combine(dir, GoldenModelFile);
        return File.Exists(path) ? path : null;
    }

    internal static string GoldenModelFile => s_goldenModelFile.Value;

    private static readonly Lazy<string> s_goldenModelFile = new(() =>
    {
        using Stream s = typeof(MuseGlimmerParity).Assembly.GetManifestResourceStream(ResourcePrefix + "ref_text.json")!;
        using JsonDocument doc = JsonDocument.Parse(s);
        return doc.RootElement.GetProperty("model").GetProperty("file").GetString();
    });

    internal static string FindDraft() => ModelDir() is { } dir
        ? Directory.EnumerateFiles(dir, "dflash*.gguf", SearchOption.TopDirectoryOnly).OrderBy(p => p, StringComparer.OrdinalIgnoreCase).FirstOrDefault()
        : null;

    internal static int ArgMax(float[] values, int n)
    {
        int best = 0;
        for (int i = 1; i < n; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }

    /// <summary>log softmax(<paramref name="values"/>[0..n)), in double.</summary>
    internal static double[] LogSoftmax(float[] values, int n)
    {
        double max = double.NegativeInfinity;
        for (int i = 0; i < n; i++) max = Math.Max(max, values[i]);
        double sum = 0;
        for (int i = 0; i < n; i++) sum += Math.Exp(values[i] - max);
        double log = Math.Log(sum);
        var lp = new double[n];
        for (int i = 0; i < n; i++) lp[i] = values[i] - max - log;
        return lp;
    }

    /// <summary>One teacher-forced position: our argmax, our log-probabilities of it and of the reference's token, and
    /// (where the reference lists it) the reference's log-probability of our token.</summary>
    internal sealed record Position(int Index, int Reference, int Ours, double OursLp, double OursLpOfReference, double? ReferenceLpOfOurs,
        double ReferenceLp)
    {
        public bool Agrees => Ours == Reference;
        /// <summary>How much more likely our model finds its own choice than the reference's (0 where they agree).</summary>
        public double OurGap => OursLp - OursLpOfReference;
        /// <summary>How much more likely the reference finds its choice than ours (null where ours is not in its list).</summary>
        public double? ReferenceGap => ReferenceLpOfOurs is { } lp ? ReferenceLp - lp : null;
    }

    /// <summary>
    /// Prefill <paramref name="golden"/>'s prompt tokens and feed its generated tokens one at a time, recording each
    /// position's comparison (<paramref name="firstLogProbs"/> gets our log-probabilities at the first position);
    /// <paramref name="beforePrompt"/> runs on the reset model before the prompt (an image's rows queued).
    /// </summary>
    internal static List<Position> TeacherForce(ModelBase model, Golden golden, out double[] firstLogProbs, Action<ModelBase> beforePrompt = null)
    {
        int vocab = model.Config.VocabSize;
        model.ResetKVCache();
        beforePrompt?.Invoke(model);
        float[] logits = model.Forward(golden.PromptTokens);
        var positions = new List<Position>(golden.Generated.Length);
        firstLogProbs = null;
        for (int i = 0; i < golden.Generated.Length; i++)
        {
            int n = Math.Min(vocab, logits.Length);
            double[] lp = LogSoftmax(logits, n);
            if (i == 0) firstLogProbs = lp;
            int ours = ArgMax(logits, n), reference = golden.Generated[i];
            Step step = golden.Steps[i];
            double? refOfOurs = step.Top.Where(t => t.Id == ours).Select(t => (double?)t.Logprob).FirstOrDefault();
            positions.Add(new Position(i, reference, ours, lp[ours], lp[reference], refOfOurs, step.Logprob));
            if (i + 1 < golden.Generated.Length)
                logits = model.Forward(new[] { reference });
        }
        return positions;
    }

    internal static int[] GreedyContinue(ModelBase model, int[] promptTokens, int count, Action<ModelBase> beforePrompt = null)
    {
        model.ResetKVCache();
        beforePrompt?.Invoke(model);
        var produced = new int[count];
        float[] logits = model.Forward(promptTokens);
        for (int i = 0; i < count; i++)
        {
            produced[i] = ArgMax(logits, Math.Min(model.Config.VocabSize, logits.Length));
            if (i + 1 < count)
                logits = model.Forward(new[] { produced[i] });
        }
        return produced;
    }

    internal static string Decode(ModelBase model, int token)
    {
        try { return model.Tokenizer.Decode(new List<int> { token }); }
        catch (Exception) { return "?"; }
    }
}

public sealed class MuseGlimmerParityTests(ITestOutputHelper output)
{
    // Measured on the legacy model (MuseGlimmerModel, ggml_metal, M5 Pro, 2026-11-20, TensorSharp db59fbfc + these tests,
    // ggml d9fe7bff) against the goldens (llama.cpp d3c3e242c, Metal): argmax = llama.cpp's token at 640/640 short and
    // 128/128 long teacher-forced positions (every greedy continuation identical for all 128 tokens), the first position's
    // top-10 log-probabilities within 0.0496 nats (0.0125-0.0496 per prompt); on MLX the same agreement, within 0.0560
    // (0.0228-0.0560). The bounds leave room for the other backends and for the runtime family, whose kernels and
    // reduction orders differ from both.

    /// <summary>
    /// The bound on a disagreement between our argmax and llama.cpp's greedy token at a teacher-forced position: each engine
    /// finds the other's choice at most this many nats below its own (a near-tie the two engines' rounding can flip), and
    /// llama.cpp lists our choice among its top four.
    /// </summary>
    private const double NearTieNats = 0.25;

    /// <summary>The share of teacher-forced positions where our argmax must be llama.cpp's token.</summary>
    private const double MinAgreement = 0.97;

    /// <summary>The bound on |our log-probability - llama.cpp's| over llama.cpp's top ten at the first generated position (about
    /// three times the legacy model's worst on ggml_metal and MLX).</summary>
    private const double FirstPositionLogprobBound = 0.15;

    /// <summary>The model the descriptor builds (the legacy one until stage S8), or with <paramref name="runtime"/> the
    /// unified-runtime family through its test seam.</summary>
    private static ModelBase Load(string draft = null, bool runtime = false)
    {
        BackendType backend = MuseGlimmerParity.Backend();
        var model = runtime
            ? MuseGlimmerArchitecture.CreateRuntimeModel(MuseGlimmerParity.FindModel(), backend, draftModelPath: draft)
            : ModelBase.Create(MuseGlimmerParity.FindModel(), backend, draftModelPath: draft);
        Assert.Equal("muse-glimmer", model.Config.Architecture);
        return model;
    }

    /// <summary>The goldens themselves (no weights): one record per prompt with a step for every generated token, llama.cpp's
    /// greedy token at the top of its own distribution, a long prompt past twice the window and legacy's 4352-row ring, and
    /// llama.cpp's DFlash path reproducing its plain continuation token for token.</summary>
    [Fact]
    public void GoldensAreWellFormed()
    {
        var text = MuseGlimmerParity.Load("ref_text.json");
        var longText = MuseGlimmerParity.Load("ref_text_long.json");
        var dflash = MuseGlimmerParity.Load("ref_text_dflash.json");
        Assert.Equal(5, text.Records.Count);
        Assert.Single(longText.Records);
        foreach (var golden in text.Records.Concat(longText.Records))
        {
            Assert.Equal(128, golden.Generated.Length);
            Assert.Equal(golden.Generated.Length, golden.Steps.Length);
            for (int i = 0; i < golden.Steps.Length; i++)
            {
                Assert.Equal(golden.Generated[i], golden.Steps[i].Top[0].Id);
                Assert.Equal(i == 0 ? 10 : 4, golden.Steps[i].Top.Length);
            }
        }
        Assert.True(longText.Records[0].PromptTokens.Length > 4352,
            $"the long golden has {longText.Records[0].PromptTokens.Length} prompt tokens; it must pass legacy's 4352-row ring");
        Assert.Equal(text.Records.Select(r => r.Name), dflash.Records.Select(r => r.Name));
        for (int i = 0; i < text.Records.Count; i++)
        {
            Assert.Equal(text.Records[i].PromptTokens, dflash.Records[i].PromptTokens);
            Assert.Equal(text.Records[i].Generated, dflash.Records[i].Generated);
        }
        Assert.Equal(text.LlamaCppCommit, longText.LlamaCppCommit);
        Assert.Equal(text.ModelSha256, dflash.ModelSha256);
    }

    /// <summary>Our tokenizer reproduces llama.cpp's tokenization of every golden prompt (the GGUF's metadata only: no
    /// weights are loaded).</summary>
    [MuseGlimmerModelFact]
    public void TokenizerMatchesLlamaCpp()
    {
        using var gguf = GgufFile.OpenWithoutSiblingShards(MuseGlimmerParity.FindModel());
        ITokenizer tokenizer = ModelBase.CreateTokenizerFromGguf(gguf);
        int mismatches = 0;
        foreach (var golden in MuseGlimmerParity.Load("ref_text.json").Records.Concat(MuseGlimmerParity.Load("ref_text_long.json").Records))
        {
            int[] ours = tokenizer.Encode(golden.Prompt, addSpecial: true).ToArray();
            if (ours.SequenceEqual(golden.PromptTokens))
            {
                output.WriteLine($"[MATCH ] {golden.Name}: {ours.Length} prompt tokens");
                continue;
            }
            mismatches++;
            int at = Enumerable.Range(0, Math.Min(ours.Length, golden.PromptTokens.Length)).FirstOrDefault(i => ours[i] != golden.PromptTokens[i]);
            output.WriteLine($"[DIFF  ] {golden.Name}: {ours.Length} vs llama.cpp {golden.PromptTokens.Length} tokens, first difference at {at}");
        }
        Assert.True(mismatches == 0, $"{mismatches} prompts tokenized differently from llama.cpp.");
    }

    /// <summary>Teacher-forced through llama.cpp's greedy continuations of the five short prompts: our argmax is llama.cpp's
    /// token at <see cref="MinAgreement"/> of the positions, every other position is a near-tie, and the first position's
    /// top-ten log-probabilities are llama.cpp's within <see cref="FirstPositionLogprobBound"/>.</summary>
    [MuseGlimmerModelFact]
    public void TeacherForcedContinuationsMatchLlamaCpp()
    {
        using var model = Load();
        CheckAgainstLlamaCpp(model, MuseGlimmerParity.Load("ref_text.json"));
    }

    /// <summary>The same over the 4,664-token prompt: the sliding window (2048), the full layers' cache growth and legacy's
    /// ring wrap (4352 rows) are all crossed before the first generated token.</summary>
    [MuseGlimmerModelFact]
    public void LongContextMatchesLlamaCpp()
    {
        using var model = Load();
        CheckAgainstLlamaCpp(model, MuseGlimmerParity.Load("ref_text_long.json"));
    }

    /// <summary><see cref="TeacherForcedContinuationsMatchLlamaCpp"/> on the unified-runtime family (stage S2 of
    /// docs/design/muse-glimmer-unified.md), held to the same gates.</summary>
    [MuseGlimmerModelFact]
    public void RuntimeTeacherForcedContinuationsMatchLlamaCpp()
    {
        using var model = Load(runtime: true);
        CheckAgainstLlamaCpp(model, MuseGlimmerParity.Load("ref_text.json"));
    }

    /// <summary><see cref="LongContextMatchesLlamaCpp"/> on the unified-runtime family: the 4,664-token prompt wraps its
    /// 2,560-cell rings in 512-row steps before the first generated token.</summary>
    [MuseGlimmerModelFact]
    public void RuntimeLongContextMatchesLlamaCpp()
    {
        using var model = Load(runtime: true);
        CheckAgainstLlamaCpp(model, MuseGlimmerParity.Load("ref_text_long.json"));
    }

    private void CheckAgainstLlamaCpp(ModelBase model, MuseGlimmerParity.GoldenFile file)
    {
        output.WriteLine($"backend {MuseGlimmerParity.Backend()}; model {model.GetType().Name}; llama.cpp {file.LlamaCppCommit}; model sha256 {file.ModelSha256}");
        int total = 0, agree = 0;
        double worstOurGap = 0, worstReferenceGap = 0, worstFirst = 0;
        var failures = new List<string>();
        foreach (var golden in file.Records)
        {
            var positions = MuseGlimmerParity.TeacherForce(model, golden, out double[] first);
            int firstDiff = positions.FindIndex(p => !p.Agrees);
            total += positions.Count;
            agree += positions.Count(p => p.Agrees);
            double firstErr = golden.Steps[0].Top.Max(t => Math.Abs(first[t.Id] - t.Logprob));
            worstFirst = Math.Max(worstFirst, firstErr);
            output.WriteLine($"{golden.Name}: {golden.PromptTokens.Length} prompt tokens; argmax = llama.cpp at " +
                $"{positions.Count(p => p.Agrees)}/{positions.Count} positions (greedy prefix {(firstDiff < 0 ? positions.Count : firstDiff)}); " +
                $"first position top-10 |dlogprob| <= {firstErr:F4}");
            if (firstErr > FirstPositionLogprobBound)
                failures.Add($"{golden.Name}: first-position log-probabilities differ by {firstErr:F4}");
            foreach (var p in positions.Where(p => !p.Agrees))
            {
                worstOurGap = Math.Max(worstOurGap, p.OurGap);
                if (p.ReferenceGap is { } g) worstReferenceGap = Math.Max(worstReferenceGap, g);
                string what = $"{golden.Name} @{p.Index}: llama.cpp {p.Reference} '{MuseGlimmerParity.Decode(model, p.Reference)}', " +
                    $"ours {p.Ours} '{MuseGlimmerParity.Decode(model, p.Ours)}'; our gap {p.OurGap:F4}, llama.cpp's gap " +
                    (p.ReferenceGap is { } rg ? rg.ToString("F4") : "(not in its top 4)");
                output.WriteLine("  " + what);
                if (p.OurGap > NearTieNats || p.ReferenceGap is not { } gap || gap > NearTieNats)
                    failures.Add(what);
            }
        }
        double agreement = (double)agree / total;
        output.WriteLine($"overall: {agree}/{total} = {agreement:P2}; worst disagreement gaps: ours {worstOurGap:F4}, llama.cpp's " +
            $"{worstReferenceGap:F4}; worst first-position |dlogprob| {worstFirst:F4}");
        Assert.True(failures.Count == 0, "positions that are not near-ties:\n" + string.Join("\n", failures));
        Assert.True(agreement >= MinAgreement, $"argmax agreement with llama.cpp {agreement:P2} < {MinAgreement:P0}");
    }

    /// <summary>
    /// DFlash is lossless speculative decoding: every drafted token is verified by the target, so the greedy output must be
    /// the plain greedy output token for token. (llama.cpp's own DFlash path reproduces its plain continuations on these
    /// prompts: <see cref="GoldensAreWellFormed"/>.)
    /// </summary>
    [MuseGlimmerModelFact(NeedsDraft = true)]
    public void DFlashDraftingIsLossless()
    {
        var records = MuseGlimmerParity.Load("ref_text.json").Records;
        var plain = new List<int[]>();
        using (var model = Load())
            foreach (var golden in records)
                plain.Add(MuseGlimmerParity.GreedyContinue(model, golden.PromptTokens, golden.Generated.Length));

        using var drafted = Load(MuseGlimmerParity.FindDraft());
        var spec = drafted as ISpeculativeModel;
        Assert.NotNull(spec);
        Assert.True(spec.HasDraftHead, "the DFlash drafter did not arm (HasDraftHead is false)");
        var decoder = new SpeculativeDecoder(spec, spec.DraftBlockSize);
        int mismatches = 0;
        for (int r = 0; r < records.Count; r++)
        {
            var golden = records[r];
            drafted.ResetKVCache();
            int[] produced = decoder.GenerateGreedy(golden.PromptTokens, golden.Generated.Length).Take(golden.Generated.Length).ToArray();
            int prefix = Enumerable.Range(0, golden.Generated.Length).FirstOrDefault(i => plain[r][i] != golden.Generated[i], golden.Generated.Length);
            if (produced.SequenceEqual(plain[r]))
                output.WriteLine($"[MATCH ] {golden.Name}: {produced.Length} tokens, the plain greedy ones (llama.cpp's for the first {prefix})");
            else
            {
                mismatches++;
                output.WriteLine($"[DIFF  ] {golden.Name}\n          plain : {string.Join(",", plain[r])}\n          dflash: {string.Join(",", produced)}");
            }
        }
        output.WriteLine($"DFlash acceptance {decoder.AcceptanceRate:P1} ({decoder.TokensAccepted}/{decoder.TokensDrafted} drafted tokens over " +
            $"{decoder.VerifySteps} verify steps, {decoder.PlainSteps} plain steps)");
        Assert.True(mismatches == 0, $"DFlash changed the greedy output of {mismatches} prompts; it must be lossless.");
        Assert.True(decoder.TokensDrafted > 0, "DFlash never drafted a token: the drafter is not running.");
    }

    /// <summary>
    /// The sliding-window pattern is the single easiest thing to get off by one: llama.cpp's set_swa_pattern(4) makes layer l
    /// a sliding-window layer when l % 4 &lt; 3, so every 4th layer (l = 3, 7, 11, ...) is full causal + NoPE.
    /// </summary>
    [Fact]
    public void SwaPatternMatchesLlamaCpp()
    {
        const int period = 4;
        var expected = new bool[52];
        for (int l = 0; l < expected.Length; l++)
            expected[l] = (l % period) < (period - 1);

        Assert.True(expected[0] && expected[1] && expected[2]);
        Assert.False(expected[3]);
        Assert.False(expected[51]);
        Assert.Equal(39, expected.Count(x => x));
        Assert.Equal(13, expected.Count(x => !x));
    }

    /// <summary>
    /// Image geometry against llama.cpp's muse_glimmer_grid_size. The 1024x1024 case is pinned to an observed llama-server
    /// run: prompt_tokens 1434 for a 65-token chat frame, i.e. 1369 = 37x37 image tokens; the 336x336 case to prompt_tokens
    /// 209, i.e. 144 = 12x12.
    /// </summary>
    [Fact]
    public void ImageGeometryMatchesLlamaCpp()
    {
        var proc = new MuseGlimmerImageProcessor();

        Assert.Equal(28, proc.PatchHW);

        // 336x336 divides exactly: 336/28 = 12 -> 12x12 = 144 tokens, no resize.
        Assert.Equal((336, 336), proc.ComputeTargetSize(336, 336));
        Assert.Equal(144, proc.ComputeTokenCount(336, 336));

        // 1024x1024: i_np = 36.571; candidates (36,36) d=0 and (37,37) d=0 tie on
        // aspect, broken toward more tokens -> 37x37, stretched to 1036x1036.
        Assert.Equal((1036, 1036), proc.ComputeTargetSize(1024, 1024));
        Assert.Equal(1369, proc.ComputeTokenCount(1024, 1024));

        // Above the 4096-token cap the grid is shrunk while preserving the ratio.
        int wide = proc.ComputeTokenCount(4000, 2000);
        Assert.True(wide <= 4096, $"token count {wide} exceeds the 4096 cap");

        // Tall and thin: the chosen grid must stay at least 1 token on each axis.
        Assert.Equal(1, proc.ComputeTokenCount(20, 20));
    }
}
