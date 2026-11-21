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
        foreach (string name in asm.GetManifestResourceNames().Where(n => n.StartsWith(ResourcePrefix, StringComparison.Ordinal)))
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
}

/// <summary>
/// The unified-runtime family (<see cref="MuseGlimmerRuntimeModel"/>, through its test seam) on the real weights against the
/// legacy outputs frozen on the same lane (<see cref="MuseGlimmerLegacyReal"/>), with docs/design/muse-glimmer-unified.md
/// section 4.2's gates: each greedy continuation is legacy's token for token, or it first leaves legacy's at a near-tie (the
/// token it takes is in legacy's top four there, less than <see cref="NearTieNats"/> below legacy's choice); the corpus's
/// teacher-forced perplexity is legacy's within <see cref="PerplexityBound"/> relative and its top-1 agreement within
/// <see cref="Top1Bound"/>. A move past <see cref="InvestigateBound"/> is printed for investigation (legacy's own two Mac
/// backends are 0.006% apart).
/// </summary>
public sealed class MuseGlimmerRuntimeRealTests(ITestOutputHelper output)
{
    internal const double NearTieNats = 0.05;
    internal const double PerplexityBound = 3e-3;
    internal const double InvestigateBound = 5e-4;
    internal const double Top1Bound = 0.005;

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
            string text = model.Tokenizer.Decode(got.Tokens.ToList()).Replace('\n', ' ');
            string what = $"{want.Name} @{first}: legacy {want.Tokens[first]} '{MuseGlimmerParity.Decode(model, want.Tokens[first])}', " +
                $"ours {ours} '{MuseGlimmerParity.Decode(model, ours)}'; legacy's top-2 margin {legacyMargin:F4}, legacy's gap to ours " +
                (legacyGap is { } g ? g.ToString("F4") : "(not in its top 4)");
            output.WriteLine($"[DIFF  ] {what}\n          ours: {(text.Length > 400 ? text[..400] + "..." : text)}");
            if (legacyGap is not { } gap || gap >= NearTieNats)
                failures.Add(what);
        }
        output.WriteLine($"{identical}/{prompts.Count} continuations identical to legacy's; worst |dlogprob| of legacy's token over the shared " +
            $"prefixes {worstPrefixDelta:F4}");
        Assert.True(failures.Count == 0, "continuations that leave legacy's at a position that is not a near-tie:\n" + string.Join("\n", failures));
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
