// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Nodes;
using TensorSharp;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// The legacy Muse-Glimmer model on the real weights, frozen per lane in stage S0 of the unified-runtime migration
/// (docs/design/muse-glimmer-unified.md section 4.2) into Fixtures/MuseGlimmer/LegacyReal/&lt;lane&gt;.json: its greedy
/// continuations of the llama.cpp goldens' prompts (MuseGlimmerParityTests: five chat prompts and the 4,664-token one, 128
/// tokens each) with the top four log-probabilities of every position, and the teacher-forced perplexity of
/// benchmarks/engine_comparison/assets/prefill_corpus.txt the way eng/validation/MuseGlimmerKvGrowProbe's --nll-text scores
/// it (a 64-token prefix in one forward, then one-row forwards). The runtime family is held to these (token-identical or a
/// near-tie; perplexity within 0.3%). The writer went with the legacy model in stage S9.
/// </summary>
internal static class MuseGlimmerLegacyReal
{
    internal const string ResourcePrefix = "InferenceWeb.Tests.Fixtures.MuseGlimmer.LegacyReal.";
    internal const int TopKept = 4;
    internal const int NllPrefix = 64;

    internal sealed record Continuation(string Name, string PromptSha256, int[] Tokens, (int Id, double Logprob)[][] Top);

    internal sealed record Perplexity(string TextSha256, int Prefix, int Scored, double MeanNll, double Value, double Top1Agreement);

    internal sealed record Lane(string Key, string ModelSha256, IReadOnlyList<Continuation> Continuations, Perplexity Nll);

    /// <summary>LegacyReal/f32_reference.json (eng/validation/MuseGlimmerF32Reference): each distinct legacy continuation
    /// teacher-forced from its prompt under exact arithmetic (the managed backend's exactly dequantized weights times F32
    /// rows, an F32 KV cache), with the top <see cref="ReferenceTopKept"/> log-probabilities of every step and the lanes
    /// whose frozen continuation it is.</summary>
    internal const string ReferenceResource = ResourcePrefix + "f32_reference.json";
    internal const int ReferenceTopKept = 10;

    internal sealed record ReferenceSequence(string Name, string PromptSha256, string TokensSha256, IReadOnlyList<string> Lanes,
        (int Id, double Logprob)[][] Top);

    internal static string CorpusPath()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir != null; dir = dir.Parent)
        {
            string candidate = Path.Combine(dir.FullName, "benchmarks", "engine_comparison", "assets", "prefill_corpus.txt");
            if (File.Exists(candidate)) return candidate;
        }
        throw new FileNotFoundException("benchmarks/engine_comparison/assets/prefill_corpus.txt was not found above the test output");
    }

    internal static string PromptSha(int[] tokens)
    {
        byte[] bytes = new byte[tokens.Length * sizeof(int)];
        Buffer.BlockCopy(tokens, 0, bytes, 0, bytes.Length);
        return Convert.ToHexStringLower(SHA256.HashData(bytes));
    }

    /// <summary>The goldens whose prompts are continued: the short set, then the long prompt.</summary>
    internal static IEnumerable<MuseGlimmerParity.Golden> Prompts()
        => MuseGlimmerParity.Load("ref_text.json").Records.Concat(MuseGlimmerParity.Load("ref_text_long.json").Records);

    /// <summary>Greedy from <paramref name="golden"/>'s prompt for its length, with each position's top log-probabilities.</summary>
    internal static Continuation Continue(ModelBase model, MuseGlimmerParity.Golden golden)
    {
        int count = golden.Generated.Length, vocab = model.Config.VocabSize;
        model.ResetKVCache();
        float[] logits = model.Forward(golden.PromptTokens);
        var tokens = new int[count];
        var top = new (int, double)[count][];
        for (int i = 0; i < count; i++)
        {
            int n = Math.Min(vocab, logits.Length);
            double[] lp = MuseGlimmerParity.LogSoftmax(logits, n);
            top[i] = TopOf(lp, n, TopKept);
            tokens[i] = top[i][0].Item1;
            if (i + 1 < count)
                logits = model.Forward(new[] { tokens[i] });
        }
        return new Continuation(golden.Name, PromptSha(golden.PromptTokens), tokens, top);
    }

    /// <summary>The <paramref name="k"/> largest of <paramref name="lp"/>[0..n), largest first (the lower id first on a tie).</summary>
    internal static (int Id, double Logprob)[] TopOf(double[] lp, int n, int k)
    {
        var best = new List<(int Id, double Logprob)>(k + 1);
        for (int j = 0; j < n; j++)
        {
            if (best.Count == k && lp[j] <= best[^1].Logprob) continue;
            int at = best.Count;
            while (at > 0 && best[at - 1].Logprob < lp[j]) at--;
            best.Insert(at, (j, lp[j]));
            if (best.Count > k) best.RemoveAt(best.Count - 1);
        }
        return best.ToArray();
    }

    /// <summary>The corpus's teacher-forced perplexity (MuseGlimmerKvGrowProbe --nll-text's procedure).</summary>
    internal static Perplexity ScoreCorpus(ModelBase model)
    {
        string path = CorpusPath();
        int[] all = model.Tokenizer.Encode(File.ReadAllText(path), addSpecial: true).ToArray();
        int vocab = model.Config.VocabSize;
        model.ResetKVCache();
        float[] logits = model.Forward(all[..NllPrefix]);
        double nll = 0;
        int agree = 0, scored = all.Length - NllPrefix;
        for (int i = 0; i < scored; i++)
        {
            int next = all[NllPrefix + i], n = Math.Min(vocab, logits.Length);
            double[] lp = MuseGlimmerParity.LogSoftmax(logits, n);
            nll -= lp[next];
            if (MuseGlimmerParity.ArgMax(logits, n) == next) agree++;
            if (i + 1 < scored)
                logits = model.Forward(new[] { next });
        }
        return new Perplexity(Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(path))), NllPrefix, scored, nll / scored,
            Math.Exp(nll / scored), (double)agree / scored);
    }

    private static readonly Lazy<Dictionary<string, Lane>> s_lanes = new(Load);

    private static Dictionary<string, Lane> Load()
    {
        var lanes = new Dictionary<string, Lane>(StringComparer.Ordinal);
        var asm = typeof(MuseGlimmerLegacyReal).Assembly;
        foreach (string name in asm.GetManifestResourceNames().Where(n => n.StartsWith(ResourcePrefix, StringComparison.Ordinal) && n != ReferenceResource))
        {
            using Stream s = asm.GetManifestResourceStream(name)!;
            using JsonDocument doc = JsonDocument.Parse(s);
            JsonElement root = doc.RootElement;
            var continuations = root.GetProperty("continuations").EnumerateArray().Select(c => new Continuation(
                c.GetProperty("name").GetString(), c.GetProperty("promptSha256").GetString(),
                c.GetProperty("tokens").EnumerateArray().Select(t => t.GetInt32()).ToArray(),
                c.GetProperty("top").EnumerateArray().Select(p => p.EnumerateArray().Select(t => (t[0].GetInt32(), t[1].GetDouble())).ToArray()).ToArray()))
                .ToList();
            JsonElement p = root.GetProperty("perplexity");
            var nll = new Perplexity(p.GetProperty("textSha256").GetString(), p.GetProperty("prefix").GetInt32(), p.GetProperty("scored").GetInt32(),
                p.GetProperty("meanNll").GetDouble(), p.GetProperty("perplexity").GetDouble(), p.GetProperty("top1Agreement").GetDouble());
            string lane = root.GetProperty("lane").GetString();
            lanes[lane] = new Lane(lane, root.GetProperty("model").GetProperty("sha256").GetString(), continuations, nll);
        }
        return lanes;
    }

    internal static IReadOnlyCollection<string> Lanes => s_lanes.Value.Keys;

    internal static Lane Find(string lane) => s_lanes.Value.GetValueOrDefault(lane);

    private static readonly Lazy<(string ModelSha256, IReadOnlyList<ReferenceSequence> Sequences)> s_reference = new(LoadReference);

    private static (string, IReadOnlyList<ReferenceSequence>) LoadReference()
    {
        using Stream s = typeof(MuseGlimmerLegacyReal).Assembly.GetManifestResourceStream(ReferenceResource)
            ?? throw new FileNotFoundException($"{ReferenceResource} is not embedded");
        using JsonDocument doc = JsonDocument.Parse(s);
        JsonElement root = doc.RootElement;
        var sequences = root.GetProperty("sequences").EnumerateArray().Select(q => new ReferenceSequence(
            q.GetProperty("name").GetString(), q.GetProperty("promptSha256").GetString(), q.GetProperty("tokensSha256").GetString(),
            q.GetProperty("lanes").EnumerateArray().Select(l => l.GetString()).ToArray(),
            q.GetProperty("top").EnumerateArray().Select(p => p.EnumerateArray().Select(t => (t[0].GetInt32(), t[1].GetDouble())).ToArray()).ToArray()))
            .ToList();
        return (root.GetProperty("model").GetProperty("sha256").GetString(), sequences);
    }

    /// <summary>The F32 reference of <paramref name="want"/>, <paramref name="lane"/>'s frozen continuation: the fixture's
    /// sequence that lists the lane, checked against the continuation's prompt, tokens and weights.</summary>
    internal static ReferenceSequence Reference(Lane lane, Continuation want)
    {
        var (weights, sequences) = s_reference.Value;
        var reference = sequences.FirstOrDefault(r => r.Name == want.Name && r.Lanes.Contains(lane.Key));
        Assert.True(reference != null, $"f32_reference.json has no sequence of {want.Name} for {lane.Key}");
        Assert.Equal(lane.ModelSha256, weights);
        Assert.Equal(want.PromptSha256, reference.PromptSha256);
        Assert.Equal(PromptSha(want.Tokens), reference.TokensSha256);
        Assert.Equal(want.Tokens.Length, reference.Top.Length);
        return reference;
    }
}

/// <summary>
/// The unified-runtime family (<see cref="MuseGlimmerRuntimeModel"/>, through its test seam) on the real weights against the
/// legacy outputs frozen on the same lane (<see cref="MuseGlimmerLegacyReal"/>), with docs/design/muse-glimmer-unified.md
/// section 4.2's gates: each greedy continuation is legacy's token for token, or it first leaves legacy's at a near-tie, by
/// legacy's numbers (the token it takes is in legacy's top four there, less than <see cref="NearTieNats"/> below legacy's
/// choice) or by the F32 reference's (the token it takes is in the reference's top ten there, and the reference puts
/// legacy's token less than <see cref="TieBound"/> above it); teacher-forced over the lane's continuations, the runtime is not
/// further from the F32 reference than legacy is (paired one-sided tests at 1% on the same steps: the |dlogprob| of the
/// reference's top-1 token within a margin of <see cref="NonInferiorityShare"/> of legacy's own, and argmax agreement); the
/// corpus's teacher-forced perplexity is legacy's within <see cref="PerplexityBound"/>
/// relative and its top-1 agreement within <see cref="Top1Bound"/>. A move past <see cref="InvestigateBound"/> is printed
/// for investigation (legacy's own two Mac backends are 0.006% apart).
/// </summary>
public sealed class MuseGlimmerRuntimeRealTests(ITestOutputHelper output)
{
    internal const double NearTieNats = 0.05;
    internal const double PerplexityBound = 3e-3;
    internal const double InvestigateBound = 5e-4;
    internal const double Top1Bound = 0.005;

    /// <summary>
    /// How far the F32 reference may put legacy's token above ours where a continuation leaves legacy's: how far a decode
    /// step's margin between the reference's top two tokens moves from the reference's margin, at the 99th percentile of the
    /// steps where that margin is below 1 nat (the steps a parting can come from). Measured on direct CUDA's decode path
    /// teacher-forced over all ten sequences of f32_reference.json in six runs whose accuracy against the reference is the
    /// same (c593f324, 81bbd0d2, and four split and product settings on 81bbd0d2): 1,362 steps pooled, the refrigerator
    /// parting the rule was written for left out; p99 0.14896 (docs/design/muse-glimmer-unified.md section 4.2).
    /// </summary>
    internal const double TieBound = 0.1490;

    /// <summary>The teacher-forced accuracy gate's one-sided level: 1%, so a mean difference whose lower bound (less 2.33
    /// standard errors) is over the margin, or a binomial tail under 0.01, is the runtime being further from the F32
    /// reference than legacy.</summary>
    internal const double OneSidedZ99 = 2.33;
    internal const double OneSidedAlpha = 0.01;

    /// <summary>The non-inferiority margin of the top-1 |dlogprob| test, a share of legacy's own mean top-1 |dlogprob|
    /// against the reference on the lane: the floor of legitimate kernel variation, as six equally accurate kernel settings
    /// on direct CUDA spread 0.000593 around legacy's 0.0058 there (docs/design/muse-glimmer-unified.md section 4.2).</summary>
    internal const double NonInferiorityShare = 0.10;

    private static MuseGlimmerLegacyReal.Lane Frozen(BackendType backend)
    {
        var frozen = MuseGlimmerLegacyReal.Find(MuseGlimmerGoldens.LaneKey(backend));
        Assert.True(frozen != null, $"no real-weight legacy outputs frozen on {MuseGlimmerGoldens.LaneKey(backend)} (lanes: {string.Join("; ", MuseGlimmerLegacyReal.Lanes)})");
        return frozen;
    }

    [MuseGlimmerModelFact]
    public void RuntimeGreedyContinuationsAreLegacysOrLeaveThemAtANearTie()
    {
        BackendType backend = MuseGlimmerParity.Backend();
        var frozen = Frozen(backend);
        using ModelBase model = MuseGlimmerArchitecture.CreateRuntimeModel(MuseGlimmerParity.FindModel(), backend);
        var prompts = MuseGlimmerLegacyReal.Prompts().ToList();
        Assert.Equal(prompts.Count, frozen.Continuations.Count);
        var failures = new List<string>();
        int identical = 0;
        double worstPrefixDelta = 0;
        for (int r = 0; r < prompts.Count; r++)
        {
            var want = frozen.Continuations[r];
            Assert.Equal(want.PromptSha256, MuseGlimmerLegacyReal.PromptSha(prompts[r].PromptTokens));
            var got = MuseGlimmerLegacyReal.Continue(model, prompts[r]);
            int first = Enumerable.Range(0, want.Tokens.Length).FirstOrDefault(i => want.Tokens[i] != got.Tokens[i], -1);
            int prefix = first < 0 ? want.Tokens.Length : first;
            for (int i = 0; i < prefix; i++)
                worstPrefixDelta = Math.Max(worstPrefixDelta, Math.Abs(want.Top[i][0].Logprob - got.Top[i][0].Logprob));
            if (first < 0)
            {
                identical++;
                output.WriteLine($"[SAME  ] {want.Name}: {want.Tokens.Length} tokens, legacy's");
                continue;
            }
            var legacyTop = want.Top[first];
            double legacyMargin = legacyTop.Length > 1 ? legacyTop[0].Logprob - legacyTop[1].Logprob : double.PositiveInfinity;
            int ours = got.Tokens[first];
            double? legacyGap = legacyTop.Where(t => t.Id == ours).Select(t => (double?)(legacyTop[0].Logprob - t.Logprob)).FirstOrDefault();
            // The reference was teacher-forced on legacy's continuation, whose prefix up to here is ours: its step is this one.
            var referenceTop = MuseGlimmerLegacyReal.Reference(frozen, want).Top[first];
            double? referenceOurs = referenceTop.Where(t => t.Id == ours).Select(t => (double?)t.Logprob).FirstOrDefault();
            double? referenceLegacy = referenceTop.Where(t => t.Id == want.Tokens[first]).Select(t => (double?)t.Logprob).FirstOrDefault();
            // Legacy's token outside the reference's top ten is at most the tenth's log-probability: a gap at most that.
            double? referenceGap = referenceOurs is { } ro ? (referenceLegacy ?? referenceTop[^1].Logprob) - ro : null;
            string text = model.Tokenizer.Decode(got.Tokens.ToList()).Replace('\n', ' ');
            string what = $"{want.Name} @{first}: legacy {want.Tokens[first]} '{MuseGlimmerParity.Decode(model, want.Tokens[first])}', " +
                $"ours {ours} '{MuseGlimmerParity.Decode(model, ours)}'; legacy's top-2 margin {legacyMargin:F4}, legacy's gap to ours " +
                (legacyGap is { } g ? g.ToString("F4") : "(not in its top 4)") + "; the F32 reference's gap from legacy's token to ours " +
                (referenceGap is { } rg ? (referenceLegacy == null ? "at most " : "") + rg.ToString("F4") : "(ours not in its top 10)") +
                $" (TieBound {TieBound:F4})";
            output.WriteLine($"[DIFF  ] {what}\n          ours: {(text.Length > 400 ? text[..400] + "..." : text)}");
            bool legacysTie = legacyGap is { } gap && gap < NearTieNats;
            bool referencesTie = referenceGap is { } refGap && refGap < TieBound;
            if (!legacysTie && !referencesTie)
                failures.Add(what);
        }
        output.WriteLine($"{identical}/{prompts.Count} continuations identical to legacy's; worst |dlogprob| of legacy's token over the shared " +
            $"prefixes {worstPrefixDelta:F4}");
        Assert.True(failures.Count == 0, "continuations that leave legacy's at a position that is not a near-tie:\n" + string.Join("\n", failures));
    }

    /// <summary>
    /// Teacher-forced over every frozen continuation of the lane (the prompt in one forward, then one-row forwards of
    /// legacy's tokens), the runtime is not further from the F32 reference than legacy is, by two paired tests on the same
    /// steps, each one-sided at 1%; legacy's numbers come from its frozen top four (greedy, so its argmax is its own token).
    /// <list type="bullet">
    /// <item>The |dlogprob| of the reference's top-1 token: d = the runtime's less legacy's at each step where that token is
    /// in legacy's top four; fails if mean(d) less <see cref="OneSidedZ99"/> standard errors (sd(d) / sqrt(n)) is over the
    /// margin, <see cref="NonInferiorityShare"/> of legacy's own mean |dlogprob| over those steps.</item>
    /// <item>Argmax agreement with the reference: an exact binomial (McNemar) on the steps where exactly one of the two takes
    /// the reference's top-1; fails if the steps where only legacy does are that many or more with probability under
    /// <see cref="OneSidedAlpha"/>, a fair coin deciding each.</item>
    /// </list>
    /// The tests compare the tree under test with legacy step by step; the margin is legacy's own error on the lane, scaled.
    /// </summary>
    [MuseGlimmerModelFact]
    public void RuntimeIsNoFurtherFromTheF32ReferenceThanLegacyTeacherForced()
    {
        BackendType backend = MuseGlimmerParity.Backend();
        var frozen = Frozen(backend);
        using ModelBase model = MuseGlimmerArchitecture.CreateRuntimeModel(MuseGlimmerParity.FindModel(), backend);
        var prompts = MuseGlimmerLegacyReal.Prompts().ToList();
        int steps = 0, onlyLegacy = 0, onlyOurs = 0;
        var d = new List<double>();
        double legacyError = 0;
        foreach (var want in frozen.Continuations)
        {
            var reference = MuseGlimmerLegacyReal.Reference(frozen, want);
            var golden = prompts.First(p => p.Name == want.Name);
            Assert.Equal(want.PromptSha256, MuseGlimmerLegacyReal.PromptSha(golden.PromptTokens));
            model.ResetKVCache();
            float[] logits = model.Forward(golden.PromptTokens);
            for (int i = 0; i < want.Tokens.Length; i++)
            {
                int n = Math.Min(model.Config.VocabSize, logits.Length);
                double[] lp = MuseGlimmerParity.LogSoftmax(logits, n);
                var (refId, refLp) = reference.Top[i][0];
                steps++;
                bool ours = MuseGlimmerParity.ArgMax(logits, n) == refId, legacyAgrees = want.Top[i][0].Id == refId;
                if (legacyAgrees && !ours) onlyLegacy++;
                if (ours && !legacyAgrees) onlyOurs++;
                if (want.Top[i].Any(t => t.Id == refId))
                {
                    double legacys = Math.Abs(want.Top[i].First(t => t.Id == refId).Logprob - refLp);
                    d.Add(Math.Abs(lp[refId] - refLp) - legacys);
                    legacyError += legacys;
                }
                if (i + 1 < want.Tokens.Length)
                    logits = model.Forward(new[] { want.Tokens[i] });
            }
        }
        double mean = d.Average();
        double sd = Math.Sqrt(d.Sum(x => (x - mean) * (x - mean)) / (d.Count - 1));
        double se = sd / Math.Sqrt(d.Count);
        double margin = NonInferiorityShare * legacyError / d.Count, lower = mean - OneSidedZ99 * se;
        double tail = BinomialUpperTail(onlyLegacy, onlyLegacy + onlyOurs);
        output.WriteLine($"{frozen.Key}: {steps} teacher-forced steps against the F32 reference. Top-1 |dlogprob|, ours less legacy's: " +
            $"n {d.Count}, mean(d) {mean:+0.000000;-0.000000}, sd {sd:F6}, SE {se:F6}, z {(se > 0 ? mean / se : 0):+0.000;-0.000}; " +
            $"lower bound mean(d) - {OneSidedZ99} SE {lower:+0.000000;-0.000000} against the margin {margin:F6} ({NonInferiorityShare:P0} of " +
            $"legacy's {legacyError / d.Count:F6}). Argmax: only legacy takes the reference's top-1 at {onlyLegacy}, only ours at " +
            $"{onlyOurs}, one-sided binomial p {tail:F4} (fails under {OneSidedAlpha})");
        Assert.True(lower <= margin,
            $"the runtime's top-1 |dlogprob| against the F32 reference is larger than legacy's by more than the margin: mean(d) {mean:F6} " +
            $"over {d.Count} steps, SE {se:F6}, lower bound {lower:F6}, margin {margin:F6}");
        Assert.True(tail >= OneSidedAlpha,
            $"the runtime's argmax leaves the F32 reference's at more steps than legacy's: only legacy's agrees at {onlyLegacy}, " +
            $"only ours at {onlyOurs}, p {tail:F4}");
    }

    /// <summary>P(X >= k) for X binomial over <paramref name="n"/> fair trials (1 when n is 0).</summary>
    internal static double BinomialUpperTail(int k, int n)
    {
        double term = Math.Pow(0.5, n), tail = 0;
        for (int j = 0; j <= n; j++)
        {
            if (j >= k) tail += term;
            term = term * (n - j) / (j + 1);
        }
        return Math.Min(1, tail);
    }

    [MuseGlimmerModelFact]
    public void RuntimePerplexityIsLegacys()
    {
        BackendType backend = MuseGlimmerParity.Backend();
        var frozen = Frozen(backend);
        using ModelBase model = MuseGlimmerArchitecture.CreateRuntimeModel(MuseGlimmerParity.FindModel(), backend);
        var got = MuseGlimmerLegacyReal.ScoreCorpus(model);
        double rel = got.Value / frozen.Nll.Value - 1, top1 = got.Top1Agreement - frozen.Nll.Top1Agreement;
        output.WriteLine($"perplexity {got.Value:F6} (legacy {frozen.Nll.Value:F6}, {rel:+0.0000%;-0.0000%}), mean NLL {got.MeanNll:F6} " +
            $"(legacy {frozen.Nll.MeanNll:F6}), top-1 {got.Top1Agreement:P2} (legacy {frozen.Nll.Top1Agreement:P2}) over {got.Scored} tokens" +
            (Math.Abs(rel) > InvestigateBound ? $"; moved more than {InvestigateBound:P2}: investigate" : ""));
        Assert.Equal(frozen.Nll.TextSha256, got.TextSha256);
        Assert.Equal(frozen.Nll.Scored, got.Scored);
        Assert.True(Math.Abs(rel) <= PerplexityBound, $"perplexity {got.Value:F5} vs legacy {frozen.Nll.Value:F5} ({rel:P3})");
        Assert.True(Math.Abs(top1) <= Top1Bound, $"top-1 agreement {got.Top1Agreement:P2} vs legacy {frozen.Nll.Top1Agreement:P2}");
    }
}
