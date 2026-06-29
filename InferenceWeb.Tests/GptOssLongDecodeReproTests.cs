// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Repro for the text_long hang: gpt-oss decode at long context (~1.2k tokens)
// goes through GptOssModel.Forward -> TransformerBlock -> the fused decode
// attention kernel (TS_GPTOSS_FUSED_DECODE). Times prefill + each decode step
// with a per-step watchdog so a stall fails fast instead of hanging the runner.
//
// Opt-in via TS_TEST_MODEL_DIR pointing at the directory containing a GptOss
// GGUF (`gpt-oss-*.gguf`). Slow.
using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class GptOssLongDecodeReproTests
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";

    private readonly ITestOutputHelper _output;
    public GptOssLongDecodeReproTests(ITestOutputHelper output) { _output = output; }

    [Fact]
    public void GptOss_LongContextDecode_DoesNotStall()
    {
        var modelPath = FindGptOss();
        if (modelPath == null) { _output.WriteLine("[gptoss-long] no model; skipping"); return; }
        _output.WriteLine($"[gptoss-long] loading {Path.GetFileName(modelPath)}");

        BackendType backend = OperatingSystem.IsMacOS() ? BackendType.GgmlMetal : BackendType.GgmlCpu;
        using var model = TensorSharp.Models.ModelBase.Create(modelPath, backend);

        int vocab = model.Config?.VocabSize ?? 0;
        _output.WriteLine($"[gptoss-long] vocab={vocab} layers={model.Config?.NumLayers} fusedDecodeEnv={Environment.GetEnvironmentVariable("TS_GPTOSS_FUSED_DECODE") ?? "(default-on)"}");

        // Build a realistic long prompt (~1.2k tokens) from the benchmark asset
        // if available, else a synthetic repeated passage.
        string text = LoadLongText();
        var promptTokens = model.Tokenizer.Encode(text, addSpecial: true).ToArray();
        // Trim/pad to land comfortably inside the 4096 fused-decode cap but well
        // past the short-context cases that already work. Override with
        // TS_REPRO_PROMPT_LEN to A/B short vs long context.
        int targetLen = 1400;
        string lenEnv = Environment.GetEnvironmentVariable("TS_REPRO_PROMPT_LEN");
        if (!string.IsNullOrEmpty(lenEnv) && int.TryParse(lenEnv, out int wantLen) && wantLen > 1)
            targetLen = wantLen;
        if (promptTokens.Length > targetLen) promptTokens = promptTokens.Take(targetLen).ToArray();
        _output.WriteLine($"[gptoss-long] prompt tokens={promptTokens.Length}");

        model.ResetKVCache();

        // Prefill with a watchdog.
        RunWithWatchdog("prefill", TimeSpan.FromSeconds(180), () =>
        {
            var sw = Stopwatch.StartNew();
            var logits = model.ForwardRefill(promptTokens);
            sw.Stop();
            _output.WriteLine($"[gptoss-long] prefill {promptTokens.Length} tok in {sw.ElapsedMilliseconds} ms");
            return logits;
        }, out float[] lastLogits);

        // Greedy decode a few tokens, each with a generous per-token watchdog so
        // we can read the timing breakdown for a single long-context decode step.
        int decodeSteps = 8;
        string stepsEnv = Environment.GetEnvironmentVariable("TS_REPRO_DECODE_STEPS");
        if (!string.IsNullOrEmpty(stepsEnv) && int.TryParse(stepsEnv, out int ds) && ds > 0)
            decodeSteps = ds;
        model.ResetForwardTiming();
        var totalSw = Stopwatch.StartNew();
        int ok = 0;
        for (int i = 0; i < decodeSteps; i++)
        {
            int next = ArgMax(lastLogits);
            var sw = Stopwatch.StartNew();
            try
            {
                lastLogits = model.Forward(new[] { next });
                sw.Stop();
                ok++;
                _output.WriteLine($"[gptoss-long] decode step {i} ({sw.ElapsedMilliseconds} ms) tok={next} -> next={ArgMax(lastLogits)}");
            }
            catch (Exception ex)
            {
                sw.Stop();
                _output.WriteLine($"[gptoss-long] decode step {i} FAILED after {sw.ElapsedMilliseconds} ms: {ex.GetType().Name}: {ex.Message}");
                throw;
            }
        }
        totalSw.Stop();
        double tps = decodeSteps / (totalSw.ElapsedMilliseconds / 1000.0);
        _output.WriteLine($"[gptoss-long] {ok}/{decodeSteps} decode tokens in {totalSw.ElapsedMilliseconds} ms = {tps:F2} tok/s");
        model.PrintTimingStats();
    }

    private static void RunWithWatchdog(string label, TimeSpan timeout, Func<float[]> work, out float[] result)
    {
        float[] captured = null;
        Exception err = null;
        var t = new Thread(() =>
        {
            try { captured = work(); }
            catch (Exception ex) { err = ex; }
        }) { IsBackground = true };
        t.Start();
        if (!t.Join(timeout))
            throw new Xunit.Sdk.XunitException($"STALL: '{label}' did not return within {timeout.TotalSeconds}s (GPU-idle hang reproduced).");
        if (err != null)
            throw new Xunit.Sdk.XunitException($"'{label}' threw: {err}");
        result = captured;
    }

    private static int ArgMax(float[] v)
    {
        int best = 0; float bv = float.NegativeInfinity;
        for (int i = 0; i < v.Length; i++) if (v[i] > bv) { bv = v[i]; best = i; }
        return best;
    }

    private static string LoadLongText()
    {
        // Walk up to find the benchmark asset; fall back to synthetic.
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        for (int i = 0; i < 8 && dir != null; i++, dir = dir.Parent)
        {
            var p = Path.Combine(dir.FullName, "benchmarks", "engine_comparison", "assets", "long_text.txt");
            if (File.Exists(p)) return File.ReadAllText(p) + "\n\nSummarize the passage above in two sentences.";
        }
        return string.Concat(Enumerable.Repeat(
            "The transformer architecture revolutionized natural language processing by relying on self-attention. ", 200))
            + "\n\nSummarize the passage above in two sentences.";
    }

    private static string FindGptOss()
    {
        string dir = Environment.GetEnvironmentVariable(EnvModelDir);
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir)) return null;
        return Directory.GetFiles(dir, "*.gguf").Where(p =>
        {
            var n = Path.GetFileName(p).ToLowerInvariant();
            return (n.Contains("gpt-oss") || n.Contains("gpt_oss") || n.Contains("gptoss"))
                && !n.Contains("mmproj");
        }).OrderBy(p => Path.GetFileName(p)).FirstOrDefault();
    }
}
