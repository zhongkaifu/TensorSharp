// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Raon-1 nano against Transformers. Fixtures/Raon/hf_goldens.json is KRAFTON/Raon-1-nano's own modeling code
// (trust_remote_code) in float32 on the CPU, made by eng/validation/raon-hf-goldens.py: five chat prompts rendered by
// apply_chat_template (enable_thinking=False), their token ids, greedy continuations with top log-probabilities, and HF
// tokenizer ids for a text corpus. The runtime is fed the RECORDED prompt ids and teacher-forced through the reference
// continuation; the GGUF is Q8_0, so the bounds allow its rounding.
//
// Inputs (read, never set):
//   TS_TEST_MODEL_DIR   directory holding raon-1-nano-Q8_0.gguf (the goldens' model.file)
//   TS_RAON_BACKEND     cpu, ggml_cpu, ggml_metal, ggml_cuda or ggml_vulkan; default: the GGML backend this process pins
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;
using Xunit.Sdk;

namespace InferenceWeb.Tests;

/// <summary>[Fact] for the real-weight Raon tests: skips without raon-1-nano-Q8_0.gguf under TS_TEST_MODEL_DIR, or when
/// TS_RAON_BACKEND names a GGML backend this process does not pin.</summary>
[TraitDiscoverer("InferenceWeb.Tests.RequiresTraitDiscoverer", "InferenceWeb.Tests")]
[AttributeUsage(AttributeTargets.Method)]
public sealed class RaonModelFactAttribute : FactAttribute, ITraitAttribute
{
    public string RequiresValue => "Models";

    public RaonModelFactAttribute() => Skip = RaonParityTests.SkipReason();
}

public sealed class RaonParityTests
{
    private const string ModelDirEnv = "TS_TEST_MODEL_DIR";
    private const string BackendEnv = "TS_RAON_BACKEND";

    /// <summary>Teacher-forced positions where our argmax must be the reference's; the rest must be near-ties.</summary>
    private const double MinAgreement = 0.9;
    /// <summary>A disagreeing position's reference top-2 margin (nats) under which it counts as a near-tie.</summary>
    private const double NearTieNats = 0.75;
    /// <summary>The first generated position's top-ten log-probabilities against the reference. Measured (2026-11-22, RTX 5090
    /// box): ggml_cuda 0.09-0.28 nats, ggml_cpu 0.14-0.39, the managed backend 0.13-0.46; Transformers itself with Q8_0-rounded
    /// weights and activations is 0.09-0.24 from its float32 run, and the f16/f32 KV cache alone moves ggml_cuda's worst prompt
    /// by 0.26.</summary>
    private const double FirstPositionLogprobBound = 0.75;

    private readonly ITestOutputHelper _output;

    public RaonParityTests(ITestOutputHelper output) => _output = output;

    private sealed record Step(double Logprob, (int Id, double Logprob)[] Top);
    private sealed record Golden(string Name, string User, string Prompt, int[] PromptTokens, int[] Generated, Step[] Steps);

    private static readonly Lazy<JsonDocument> s_doc = new(() =>
    {
        using Stream s = typeof(RaonParityTests).Assembly.GetManifestResourceStream("InferenceWeb.Tests.Fixtures.Raon.hf_goldens.json")
            ?? throw new FileNotFoundException("Fixtures/Raon/hf_goldens.json is not embedded");
        return JsonDocument.Parse(s);
    });

    private static IReadOnlyList<Golden> Goldens() => s_doc.Value.RootElement.GetProperty("records").EnumerateArray()
        .Select(r => new Golden(r.GetProperty("name").GetString(), r.GetProperty("user").GetString(), r.GetProperty("prompt").GetString(),
            Ints(r.GetProperty("prompt_tokens")), Ints(r.GetProperty("generated_tokens")),
            r.GetProperty("steps").EnumerateArray().Select(x => new Step(x.GetProperty("logprob").GetDouble(),
                x.GetProperty("top").EnumerateArray().Select(t => (t[0].GetInt32(), t[1].GetDouble())).ToArray())).ToArray()))
        .ToList();

    private static int[] Ints(JsonElement e) => e.EnumerateArray().Select(x => x.GetInt32()).ToArray();

    private static string ModelFile => s_doc.Value.RootElement.GetProperty("model").GetProperty("file").GetString();

    /// <summary>The goldens' model.file under TS_TEST_MODEL_DIR; a TS_TEST_MODEL_DIR naming a GGUF file is taken as it is (another
    /// quant, measured against the same Q8_0 bounds).</summary>
    private static string FindModel()
    {
        string dir = Environment.GetEnvironmentVariable(ModelDirEnv);
        if (string.IsNullOrWhiteSpace(dir)) return null;
        if (File.Exists(dir)) return dir;
        string path = Path.Combine(dir, ModelFile);
        return File.Exists(path) ? path : null;
    }

    private static BackendType Backend()
    {
        string name = Environment.GetEnvironmentVariable(BackendEnv);
        if (string.IsNullOrWhiteSpace(name)) return TestGates.PinnedGgmlBackend;
        return name.Trim().ToLowerInvariant() switch
        {
            "cpu" => BackendType.Cpu,
            "ggml_cpu" => BackendType.GgmlCpu,
            "ggml_metal" => BackendType.GgmlMetal,
            "ggml_cuda" => BackendType.GgmlCuda,
            "ggml_vulkan" => BackendType.GgmlVulkan,
            _ => throw new ArgumentException($"{BackendEnv}={name}: use cpu, ggml_cpu, ggml_metal, ggml_cuda or ggml_vulkan."),
        };
    }

    internal static string SkipReason()
    {
        string skip = TestGates.ModelSkip(ModelDirEnv);
        if (skip != null) return skip;
        if (FindModel() == null) return $"Requires model weights (no {ModelFile} under {ModelDirEnv}: the goldens are that quant's).";
        try { return TestGates.GgmlPinSkip(Backend()); }
        catch (ArgumentException) { return null; }
    }

    /// <summary>The GGUF's tokenizer gives Transformers' ids for every golden prompt and the corpus (metadata only).</summary>
    [RaonModelFact]
    public void TokenizerMatchesTransformers()
    {
        using var gguf = new GgufFile(FindModel());
        ITokenizer tokenizer = ModelBase.CreateTokenizerFromGguf(gguf);
        var cases = Goldens().Select(g => (g.Name, g.Prompt, g.PromptTokens))
            .Concat(s_doc.Value.RootElement.GetProperty("tokenizer").EnumerateArray()
                .Select(c => ("corpus", c.GetProperty("text").GetString(), Ints(c.GetProperty("ids")))));
        var mismatches = new List<string>();
        foreach (var (name, text, expected) in cases)
        {
            int[] actual = tokenizer.Encode(text, addSpecial: true).ToArray();
            if (actual.SequenceEqual(expected)) continue;
            int at = 0;
            while (at < Math.Min(actual.Length, expected.Length) && actual[at] == expected[at]) at++;
            mismatches.Add($"{name} differs at {at}: HF [{string.Join(",", expected.Skip(at).Take(8))}] ours " +
                $"[{string.Join(",", actual.Skip(at).Take(8))}] in {JsonSerializer.Serialize(text)}");
        }
        foreach (string m in mismatches) _output.WriteLine(m);
        Assert.Empty(mismatches);
    }

    /// <summary>The GGUF's chat template renders every golden's user turn as apply_chat_template did, generation prompt
    /// (and its trailing newlines) included.</summary>
    [RaonModelFact]
    public void ChatTemplateMatchesTransformers()
    {
        using var gguf = new GgufFile(FindModel());
        string template = gguf.GetString("tokenizer.chat_template");
        foreach (Golden g in Goldens())
        {
            var messages = new List<ChatMessage> { new() { Role = "user", Content = g.User } };
            Assert.Equal(g.Prompt, ChatTemplate.RenderFromGgufTemplate(template, messages, addGenerationPrompt: true,
                "raon1nano", enableThinking: false));
        }
    }

    /// <summary>Teacher-forced through Transformers' greedy continuations: our argmax is the reference's at
    /// <see cref="MinAgreement"/> of the positions and every other position is a near-tie, and the first position's top-ten
    /// log-probabilities are within <see cref="FirstPositionLogprobBound"/>.</summary>
    [RaonModelFact]
    public void TeacherForcedContinuationsMatchTransformers()
    {
        using ModelBase model = ModelBase.Create(FindModel(), Backend());
        Assert.Equal("raon1nano", model.Config.Architecture);
        int vocab = model.Config.VocabSize, agree = 0, total = 0;
        var failures = new List<string>();
        foreach (Golden g in Goldens())
        {
            model.ResetKVCache();
            float[] logits = model.Forward(g.PromptTokens);
            int matched = 0;
            for (int i = 0; i < g.Generated.Length; i++)
            {
                int n = Math.Min(vocab, logits.Length);
                double[] lp = MuseGlimmerParity.LogSoftmax(logits, n);
                int ours = MuseGlimmerParity.ArgMax(logits, n), reference = g.Generated[i];
                Step step = g.Steps[i];
                if (i == 0)
                {
                    double delta = step.Top.Max(t => Math.Abs(lp[t.Id] - t.Logprob));
                    _output.WriteLine($"{g.Name}: first position top-10 max |delta| = {delta:F4} nats");
                    if (delta > FirstPositionLogprobBound) failures.Add($"{g.Name}: first position delta {delta:F4} > {FirstPositionLogprobBound}");
                }
                total++;
                if (ours == reference) { agree++; matched++; }
                else
                {
                    double margin = step.Top.Length > 1 ? step.Top[0].Logprob - step.Top[1].Logprob : double.PositiveInfinity;
                    _output.WriteLine($"{g.Name}[{i}]: ours {ours} vs HF {reference} (HF top-2 margin {margin:F3}, ours gap {lp[ours] - lp[reference]:F3})");
                    if (margin > NearTieNats) failures.Add($"{g.Name}[{i}]: disagreement at a {margin:F3}-nat margin");
                }
                if (i + 1 < g.Generated.Length)
                    logits = model.Forward(new[] { reference });
            }
            _output.WriteLine($"{g.Name}: {matched}/{g.Generated.Length} teacher-forced positions agree");
        }
        _output.WriteLine($"total: {agree}/{total}");
        Assert.True(agree >= MinAgreement * total, $"only {agree}/{total} positions agree");
        Assert.Empty(failures);
    }
}
