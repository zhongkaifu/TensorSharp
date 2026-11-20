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
/// near-tie; perplexity within 0.3%); until stage S9 the legacy model is held to them on the lane that froze them.
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

    internal static JsonObject ContinuationJson(Continuation c) => new()
    {
        ["name"] = c.Name,
        ["promptSha256"] = c.PromptSha256,
        ["tokens"] = new JsonArray(c.Tokens.Select(t => (JsonNode)t).ToArray()),
        ["top"] = new JsonArray(c.Top.Select(p => (JsonNode)new JsonArray(p.Select(t => (JsonNode)new JsonArray(t.Id, Math.Round(t.Logprob, 6))).ToArray())).ToArray()),
    };
}

public sealed class MuseGlimmerLegacyRealTests(ITestOutputHelper output)
{
    /// <summary>The legacy model on the lane that froze its real-weight continuations reproduces them token for token, each
    /// position's top log-probabilities within 1e-3 (the same build's arithmetic; a change there moved the oracle).</summary>
    [MuseGlimmerModelFact]
    public void LegacyReproducesItsFrozenContinuations()
    {
        BackendType backend = MuseGlimmerParity.Backend();
        string lane = MuseGlimmerGoldens.LaneKey(backend);
        var frozen = MuseGlimmerLegacyReal.Find(lane);
        Assert.True(frozen != null, $"no real-weight legacy outputs frozen on {lane} (lanes: {string.Join("; ", MuseGlimmerLegacyReal.Lanes)})");
        using var model = new MuseGlimmerModel(MuseGlimmerParity.FindModel(), backend);
        var prompts = MuseGlimmerLegacyReal.Prompts().ToList();
        Assert.Equal(prompts.Count, frozen.Continuations.Count);
        double worst = 0;
        for (int r = 0; r < prompts.Count; r++)
        {
            var want = frozen.Continuations[r];
            Assert.Equal(want.PromptSha256, MuseGlimmerLegacyReal.PromptSha(prompts[r].PromptTokens));
            var got = MuseGlimmerLegacyReal.Continue(model, prompts[r]);
            Assert.Equal(want.Tokens, got.Tokens);
            for (int i = 0; i < want.Top.Length; i++)
                for (int k = 0; k < want.Top[i].Length; k++)
                {
                    Assert.Equal(want.Top[i][k].Id, got.Top[i][k].Id);
                    worst = Math.Max(worst, Math.Abs(want.Top[i][k].Logprob - got.Top[i][k].Logprob));
                }
            output.WriteLine($"{want.Name}: {want.Tokens.Length} tokens as frozen");
        }
        output.WriteLine($"worst top-{MuseGlimmerLegacyReal.TopKept} |dlogprob| {worst:E2}");
        Assert.True(worst <= 1e-3, $"the legacy log-probabilities moved by {worst:E2} since the freeze");
    }

    /// <summary>The legacy model's teacher-forced perplexity of the corpus is the frozen one (within 1e-4 relative) on the lane
    /// that froze it.</summary>
    [MuseGlimmerModelFact]
    public void LegacyReproducesItsFrozenPerplexity()
    {
        BackendType backend = MuseGlimmerParity.Backend();
        var frozen = MuseGlimmerLegacyReal.Find(MuseGlimmerGoldens.LaneKey(backend));
        Assert.True(frozen != null, $"no real-weight legacy outputs frozen on {MuseGlimmerGoldens.LaneKey(backend)}");
        using var model = new MuseGlimmerModel(MuseGlimmerParity.FindModel(), backend);
        var got = MuseGlimmerLegacyReal.ScoreCorpus(model);
        output.WriteLine($"perplexity {got.Value:F5} (frozen {frozen.Nll.Value:F5}), top-1 {got.Top1Agreement:P2} over {got.Scored} tokens");
        Assert.Equal(frozen.Nll.TextSha256, got.TextSha256);
        Assert.Equal(frozen.Nll.Scored, got.Scored);
        Assert.True(Math.Abs(got.Value / frozen.Nll.Value - 1) <= 1e-4, $"perplexity {got.Value:F5} vs frozen {frozen.Nll.Value:F5}");
    }
}

/// <summary>Writes Fixtures/MuseGlimmer/LegacyReal/&lt;lane&gt;.json (<see cref="MuseGlimmerLegacyReal"/>) on the backend the parity
/// tests run on, into the directory TS_TEST_FREEZE_MUSE_GLIMMER_REAL names. Needs the weights (TS_TEST_MODEL_DIR).</summary>
[Trait("Category", "Bench")]
public sealed class MuseGlimmerLegacyRealWriterTests(ITestOutputHelper output)
{
    private const string OutEnv = "TS_TEST_FREEZE_MUSE_GLIMMER_REAL";

    [FreezeFact(OutEnv)]
    public void FreezeTheLegacyRealWeightOutputs()
    {
        Assert.True(MuseGlimmerParity.Skip(needsDraft: false) == null, MuseGlimmerParity.Skip(needsDraft: false));
        BackendType backend = MuseGlimmerParity.Backend();
        string lane = MuseGlimmerGoldens.LaneKey(backend), modelPath = MuseGlimmerParity.FindModel();
        string modelSha = File.Exists(modelPath + ".done") ? File.ReadAllText(modelPath + ".done").Split(' ')[0] : MuseGlimmerGoldens.FileSha(modelPath);
        using var model = new MuseGlimmerModel(modelPath, backend);
        var continuations = new JsonArray();
        foreach (var golden in MuseGlimmerLegacyReal.Prompts())
        {
            var c = MuseGlimmerLegacyReal.Continue(model, golden);
            int prefix = Enumerable.Range(0, c.Tokens.Length).FirstOrDefault(i => c.Tokens[i] != golden.Generated[i], c.Tokens.Length);
            output.WriteLine($"{golden.Name}: {c.Tokens.Length} tokens, llama.cpp's for the first {prefix}");
            continuations.Add(MuseGlimmerLegacyReal.ContinuationJson(c));
        }
        var nll = MuseGlimmerLegacyReal.ScoreCorpus(model);
        output.WriteLine($"perplexity {nll.Value:F5} (mean NLL {nll.MeanNll:F5}), top-1 {nll.Top1Agreement:P2} over {nll.Scored} tokens");
        var file = new JsonObject
        {
            ["lane"] = lane,
            ["about"] = "The legacy Muse-Glimmer model's greedy continuations of the llama.cpp goldens' prompts and its teacher-forced " +
                "perplexity of prefill_corpus.txt on the real weights, frozen on this lane in stage S0 of the unified-runtime migration " +
                "(InferenceWeb.Tests/MuseGlimmerLegacyRealTests.cs reads them).",
            ["producer"] = "MuseGlimmerModel (legacy)",
            ["provenance"] = Environment.GetEnvironmentVariable("TS_TEST_FREEZE_PROVENANCE") ?? "",
            ["frozenUtc"] = DateTime.UtcNow.ToString("O"),
            ["model"] = new JsonObject { ["file"] = Path.GetFileName(modelPath), ["sha256"] = modelSha },
            ["kvCacheDtype"] = model.KvCacheDtype.ToString(),
            ["continuations"] = continuations,
            ["perplexity"] = new JsonObject
            {
                ["text"] = "benchmarks/engine_comparison/assets/prefill_corpus.txt",
                ["textSha256"] = nll.TextSha256,
                ["prefix"] = nll.Prefix,
                ["scored"] = nll.Scored,
                ["meanNll"] = nll.MeanNll,
                ["perplexity"] = nll.Value,
                ["top1Agreement"] = nll.Top1Agreement,
            },
        };
        string path = Path.Combine(Environment.GetEnvironmentVariable(OutEnv), MuseGlimmerGoldens.FileName(lane));
        File.WriteAllText(path, file.ToJsonString(new JsonSerializerOptions { WriteIndented = false }) + "\n");
        output.WriteLine($"wrote {path}");
    }
}
