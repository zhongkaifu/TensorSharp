// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Qwen3.6 NextN/MTP speculative decoding: correctness + performance.
//
// Greedy speculative decoding is verification-gated: every emitted token is
// the trunk's argmax given the same prefix, so the output stream must match
// plain greedy decoding except where batched-vs-sequential kernel ordering
// flips a near-tie argmax (the same FP-drift tolerance the legacy-vs-batched
// Qwen3.5 correctness test uses).
//
// Opt-in via TS_MTP_E2E=1 (loads a ~10-12 GB GGUF). Model resolution order:
//   1. TS_MTP_MODEL       — explicit .gguf path
//   2. TS_MTP_MODEL_DIR   — directory scanned for Qwen3.6 GGUFs
//   3. C:\Works\models\mtp (default test location)
// Backend via TS_MTP_BACKEND: ggml_cpu (default) | ggml_cuda | cpu.
// TS_MTP_NEW_TOKENS / TS_MTP_DRAFT override generation length / draft window.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using TensorSharp;
using TensorSharp.Models;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class Qwen36MtpTests
{
    private readonly ITestOutputHelper _output;
    public Qwen36MtpTests(ITestOutputHelper output) { _output = output; }

    private const string DefaultModelDir = @"C:\Works\models\mtp";

    [Fact]
    public void Mtp_SpeculativeGreedy_MatchesBaselineGreedy()
    {
        string modelPath = ResolveModel();
        if (modelPath == null) { _output.WriteLine("[mtp] opt-in not set or model missing; skipping"); return; }

        int maxNew = EnvInt("TS_MTP_NEW_TOKENS", 32);
        int maxDraft = EnvInt("TS_MTP_DRAFT", 8);

        _output.WriteLine($"[mtp] loading {Path.GetFileName(modelPath)} backend={ResolveBackend()}");
        using var model = (Qwen35Model)ModelBase.Create(modelPath, ResolveBackend());
        Xunit.Assert.True(model.HasMtp, "model should expose a NextN/MTP draft head");

        string prompt = "Q: What is the capital of France?\nA: The capital of France is Paris.\n" +
                        "Q: What is the capital of Japan?\nA:";
        int[] tokens = model.Tokenizer.Encode(prompt, addSpecial: false).ToArray();
        _output.WriteLine($"[mtp] prompt tokens: {tokens.Length}");

        // Baseline: plain greedy decode through the standard Forward path.
        model.ResetKVCache();
        var baseline = new List<int>();
        var swBasePrefill = Stopwatch.StartNew();
        float[] logits = model.ForwardRefill(tokens);
        swBasePrefill.Stop();
        var swBaseDecode = Stopwatch.StartNew();
        int t = Argmax(logits);
        baseline.Add(t);
        for (int i = 1; i < maxNew; i++)
        {
            logits = model.Forward(new[] { t });
            t = Argmax(logits);
            baseline.Add(t);
        }
        swBaseDecode.Stop();

        // Speculative: MTP draft + batched trunk verification.
        var spec = new MtpSpeculativeDecoder(model, maxDraft);
        List<int> specTokens = spec.GenerateGreedy(tokens, maxNew);

        string baseText = model.Tokenizer.Decode(baseline);
        string specText = model.Tokenizer.Decode(specTokens);
        _output.WriteLine($"[mtp] baseline ({swBaseDecode.Elapsed.TotalSeconds:F1}s decode): \"{Trim(baseText)}\"");
        _output.WriteLine($"[mtp] spec     ({spec.LastDecodeSeconds:F1}s decode): \"{Trim(specText)}\"");
        _output.WriteLine($"[mtp] baseline tokens: [{string.Join(",", baseline)}]");
        _output.WriteLine($"[mtp] spec tokens:     [{string.Join(",", specTokens)}]");
        _output.WriteLine($"[mtp] drafted={spec.TokensDrafted} accepted={spec.TokensAccepted} " +
            $"({spec.AcceptanceRate:P0}), verifySteps={spec.VerifySteps}, plainSteps={spec.PlainSteps}, " +
            $"rollbacks={spec.RollbackSteps}");

        int compareLen = Math.Min(baseline.Count, specTokens.Count);
        int matchPrefix = 0;
        while (matchPrefix < compareLen && baseline[matchPrefix] == specTokens[matchPrefix])
            matchPrefix++;
        _output.WriteLine($"[mtp] prefix match {matchPrefix}/{compareLen}");

        Xunit.Assert.Equal(maxNew, specTokens.Count);
        // Verification gating means divergence can only come from FP drift on a
        // near-tie argmax; require at least half the stream to match (random
        // agreement on a 150k vocab is ~0).
        Xunit.Assert.True(matchPrefix >= compareLen / 2,
            $"spec/baseline prefix match {matchPrefix}/{compareLen} below 50% — structural divergence suspected");
        Xunit.Assert.True(spec.TokensDrafted > 0, "draft head never produced a candidate");
        Xunit.Assert.True(spec.TokensAccepted > 0, "no drafted token was ever accepted");
    }

    [Fact]
    public void Mtp_PerfBench_SpecVsBaseline()
    {
        string modelPath = ResolveModel();
        if (modelPath == null || Environment.GetEnvironmentVariable("TS_MTP_BENCH") != "1")
        { _output.WriteLine("[mtp-bench] opt-in not set; skipping"); return; }

        int maxNew = EnvInt("TS_MTP_NEW_TOKENS", 64);
        int maxDraft = EnvInt("TS_MTP_DRAFT", 8);

        _output.WriteLine($"[mtp-bench] loading {Path.GetFileName(modelPath)} backend={ResolveBackend()}");
        using var model = (Qwen35Model)ModelBase.Create(modelPath, ResolveBackend());
        Xunit.Assert.True(model.HasMtp);

        string prompt = "Write a short story about a robot learning to paint. " +
                        "Once upon a time, in a small workshop at the edge of the city,";
        int[] tokens = model.Tokenizer.Encode(prompt, addSpecial: false).ToArray();

        // Warmup (kernel compilation, memory pools) with a tiny generation.
        model.ResetKVCache();
        model.ForwardRefill(tokens);
        model.Forward(new[] { tokens[^1] });

        // Baseline greedy.
        model.ResetKVCache();
        var baseline = new List<int>();
        var swPrefill = Stopwatch.StartNew();
        float[] logits = model.ForwardRefill(tokens);
        swPrefill.Stop();
        var swDecode = Stopwatch.StartNew();
        int t = Argmax(logits);
        baseline.Add(t);
        for (int i = 1; i < maxNew; i++)
        {
            logits = model.Forward(new[] { t });
            t = Argmax(logits);
            baseline.Add(t);
        }
        swDecode.Stop();
        double baseTps = (maxNew - 1) / swDecode.Elapsed.TotalSeconds;

        // Speculative greedy.
        var spec = new MtpSpeculativeDecoder(model, maxDraft);
        string pminEnv = Environment.GetEnvironmentVariable("TS_MTP_PMIN");
        if (!string.IsNullOrEmpty(pminEnv) && float.TryParse(pminEnv, out float pmin))
            spec.MinDraftProb = pmin;
        List<int> specTokens = spec.GenerateGreedy(tokens, maxNew);
        double specTps = (specTokens.Count - 1) / spec.LastDecodeSeconds;

        _output.WriteLine($"[mtp-bench] prompt={tokens.Length} tok, gen={maxNew} tok, draft window={maxDraft}");
        _output.WriteLine($"[mtp-bench] baseline: prefill {swPrefill.Elapsed.TotalSeconds:F2}s, " +
            $"decode {swDecode.Elapsed.TotalSeconds:F2}s = {baseTps:F2} tok/s");
        _output.WriteLine($"[mtp-bench] spec:     prefill {spec.LastPrefillSeconds:F2}s, " +
            $"decode {spec.LastDecodeSeconds:F2}s = {specTps:F2} tok/s");
        _output.WriteLine($"[mtp-bench] speedup: {specTps / baseTps:F2}x | acceptance {spec.AcceptanceRate:P0} " +
            $"({spec.TokensAccepted}/{spec.TokensDrafted}), verifySteps={spec.VerifySteps}, " +
            $"plainSteps={spec.PlainSteps}, rollbacks={spec.RollbackSteps}");
        _output.WriteLine($"[mtp-bench] baseline: \"{Trim(model.Tokenizer.Decode(baseline))}\"");
        _output.WriteLine($"[mtp-bench] spec:     \"{Trim(model.Tokenizer.Decode(specTokens))}\"");
    }

    [Fact]
    public void Mtp_Profile_LayerTypeSplit()
    {
        // Opt-in profiling (TS_MTP_PROFILE=1): where does a speculative step's
        // trunk time go — attention layers, recurrent (GDN) layers, or the LM
        // head? Drives optimization of the speculative path's per-pass cost.
        string modelPath = ResolveModel();
        if (modelPath == null || Environment.GetEnvironmentVariable("TS_MTP_PROFILE") != "1")
        { _output.WriteLine("[mtp-profile] opt-in not set; skipping"); return; }

        int maxNew = EnvInt("TS_MTP_NEW_TOKENS", 48);
        int maxDraft = EnvInt("TS_MTP_DRAFT", 8);

        _output.WriteLine($"[mtp-profile] loading {Path.GetFileName(modelPath)} backend={ResolveBackend()}");
        using var model = (Qwen35Model)ModelBase.Create(modelPath, ResolveBackend());
        Xunit.Assert.True(model.HasMtp);

        string prompt = "Write a long, detailed essay about the history of video games. Begin:";
        int[] tokens = model.Tokenizer.Encode(prompt, addSpecial: false).ToArray();

        // Warmup.
        model.ResetKVCache();
        model.ForwardRefill(tokens);
        model.Forward(new[] { tokens[^1] });

        var spec = new MtpSpeculativeDecoder(model, maxDraft);
        model.ResetSpecLayerTimings();
        var specTokens = spec.GenerateGreedy(tokens, maxNew);

        double msPerTick = 1000.0 / Stopwatch.Frequency;
        double attnMs = model.SpecAttnLayerTicks * msPerTick;
        double recMs = model.SpecRecurrentLayerTicks * msPerTick;
        double headMs = model.SpecLmHeadTicks * msPerTick;
        double decodeMs = spec.LastDecodeSeconds * 1000;
        _output.WriteLine($"[mtp-profile] tokens={specTokens.Count} decode={decodeMs:F0}ms " +
            $"({decodeMs / specTokens.Count:F0} ms/token) acceptance={spec.AcceptanceRate:P0} " +
            $"verify={spec.VerifySteps} plain={spec.PlainSteps} rollbacks={spec.RollbackSteps}");
        _output.WriteLine($"[mtp-profile] trunk attention layers: {attnMs:F0} ms ({100 * attnMs / decodeMs:F1}% of decode)");
        _output.WriteLine($"[mtp-profile] trunk recurrent (GDN) layers: {recMs:F0} ms ({100 * recMs / decodeMs:F1}%)");
        _output.WriteLine($"[mtp-profile] trunk LM head: {headMs:F0} ms ({100 * headMs / decodeMs:F1}%)");
        _output.WriteLine($"[mtp-profile] unaccounted (draft head, catch-up, emb, copies): {decodeMs - attnMs - recMs - headMs:F0} ms");
    }

    private static int Argmax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++)
            if (v[i] > v[best]) best = i;
        return best;
    }

    private static int EnvInt(string name, int fallback)
    {
        string s = Environment.GetEnvironmentVariable(name);
        return !string.IsNullOrEmpty(s) && int.TryParse(s, out int v) && v > 0 ? v : fallback;
    }

    private static BackendType ResolveBackend() =>
        (Environment.GetEnvironmentVariable("TS_MTP_BACKEND") ?? "ggml_cpu").ToLowerInvariant() switch
        {
            "ggml_cuda" => BackendType.GgmlCuda,
            "ggml_metal" => BackendType.GgmlMetal,
            "cuda" => BackendType.Cuda,
            "cpu" => BackendType.Cpu,
            _ => BackendType.GgmlCpu,
        };

    private static string ResolveModel()
    {
        if (Environment.GetEnvironmentVariable("TS_MTP_E2E") != "1")
            return null;

        string explicitPath = Environment.GetEnvironmentVariable("TS_MTP_MODEL");
        if (!string.IsNullOrEmpty(explicitPath) && File.Exists(explicitPath))
            return explicitPath;

        string dir = Environment.GetEnvironmentVariable("TS_MTP_MODEL_DIR");
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir))
            dir = DefaultModelDir;
        if (!Directory.Exists(dir))
            return null;

        // Prefer the MoE model (3B active params decodes much faster on CPU).
        return Directory.GetFiles(dir, "*.gguf")
            .Where(p =>
            {
                string n = Path.GetFileName(p).ToLowerInvariant();
                return n.Contains("qwen3.6") && !n.Contains("mmproj");
            })
            .OrderByDescending(p => Path.GetFileName(p).Contains("A3B", StringComparison.OrdinalIgnoreCase))
            .FirstOrDefault();
    }

    private static string Trim(string s, int len = 160)
        => string.IsNullOrEmpty(s) ? string.Empty
           : (s.Length <= len ? s : s.Substring(0, len) + "...").Replace("\n", "\\n");
}
