// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Reproduces the user-reported parallel-decode throughput regression:
// a single in-flight request keeps the GPU busy (one fused decode graph
// per token), but two concurrent requests drop GPU utilisation sharply
// because the engine switches to the op-by-op batched paged path.
//
// This bench loads ONE model and measures aggregate decode throughput at
// concurrency 1, 2, and 4 using the same long-generation Chinese prompts
// the user ran. Opt in by pointing TS_TEST_MODEL_DIR at ~/work/model (or
// set TS_BENCH_MODEL to an explicit .gguf path). The test never fails on
// throughput; it just prints a table so we can compare before/after a fix.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Runtime.Scheduling;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class ParallelThroughputBench
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";
    private const string EnvModelPath = "TS_BENCH_MODEL";

    private readonly ITestOutputHelper _output;
    public ParallelThroughputBench(ITestOutputHelper output) { _output = output; }

    // The user's exact scenario: gemma-4-12b, two long-generation prompts.
    [Fact] public Task Gemma4_12b_ParallelDecodeScaling() => RunScaling("gemma-4-12b-it-Q8_0.gguf");
    [Fact] public Task Gemma4_E4B_ParallelDecodeScaling() => RunScaling("gemma-4-E4B-it-Q8_0.gguf");
    // MoE Gemma 4 (26B total / 4B active). This is the model the user ran into
    // the round-robin regression with: ForwardBatch can't run MoE layers and,
    // before the per-sequence-fused fix, SupportsPerSequenceFusedForward was
    // gated off for MoE — so two concurrent requests fell back to the serial
    // KV-swap path and decoded round-robin instead of in parallel.
    [Fact] public Task Gemma4_26B_A4B_ParallelDecodeScaling() => RunScaling("mtp_gemma4/gemma-4-26B-A4B-it-Q4_K_M.gguf");
    [Fact] public Task Qwen36_27b_ParallelDecodeScaling() => RunScaling("Qwen3.6-27B-IQ4_XS.gguf");
    [Fact] public Task GptOss_20b_ParallelDecodeScaling() => RunScaling("gpt-oss-20b-Q8_0.gguf");

    private async Task RunScaling(string modelFileName)
    {
        string modelPath = ResolveModel(modelFileName);
        if (modelPath == null)
        {
            _output.WriteLine($"[bench] model {modelFileName} not found; set {EnvModelDir} or {EnvModelPath}. Skipping.");
            return;
        }

        _output.WriteLine($"[bench] loading {Path.GetFileName(modelPath)}");
        using var ctx = new BenchEngine(modelPath);
        bool batched = ctx.Model is IBatchedPagedModel;
        _output.WriteLine($"[bench] arch={ctx.Model.Config?.Architecture} batchedModel={batched}");

        // Long-generation prompts (decode-bound) mirroring the user's run.
        string[] prompts =
        {
            "请详细介绍最终幻想7。",
            "请详细介绍时间简史。",
            "请详细介绍量子力学的发展历史。",
            "请详细介绍唐诗宋词的艺术特点。",
        };
        const int maxNewTokens = 160;

        // Warm-up: load Metal pipelines, JIT hot paths.
        await RunConcurrency(ctx, prompts, 1, maxNewTokens: 8, warm: true);

        var results = new List<(int n, double wall, int outToks, double aggTps, double perReqTps)>();
        foreach (int n in new[] { 1, 2, 4 })
        {
            var (wall, outToks, texts) = await RunConcurrency(ctx, prompts, n, maxNewTokens, warm: false);
            double aggTps = wall > 0 ? outToks / wall : 0;
            double perReqTps = aggTps / n;
            results.Add((n, wall, outToks, aggTps, perReqTps));
            _output.WriteLine($"[bench] n={n}: wall={wall:F2}s out={outToks} aggTps={aggTps:F1} perReqTps={perReqTps:F1}");

            // Correctness guard: concurrent decode must not produce degenerate
            // (token-repeat-loop) output. A state-corruption bug in the
            // per-sequence cache binding would surface as long immediate repeats.
            for (int i = 0; i < texts.Count; i++)
            {
                int rep = LongestImmediateRepeat(texts[i]);
                _output.WriteLine($"[bench]   n={n} req{i}: repeat={rep} preview=\"{Preview(texts[i], 80)}\"");
                Assert.True(rep < 20, $"n={n} req{i} output degenerate (longest immediate repeat={rep}): {Preview(texts[i], 160)}");
            }
        }

        _output.WriteLine("");
        _output.WriteLine($"===== {Path.GetFileName(modelPath)} parallel decode scaling =====");
        _output.WriteLine($"{"conc",4} {"wall(s)",8} {"outTok",7} {"aggTps",8} {"perReqTps",10}");
        foreach (var r in results)
            _output.WriteLine($"{r.n,4} {r.wall,8:F2} {r.outToks,7} {r.aggTps,8:F1} {r.perReqTps,10:F1}");
        double baseTps = results[0].aggTps;
        foreach (var r in results)
            _output.WriteLine($"  n={r.n}: aggregate throughput {(baseTps > 0 ? r.aggTps / baseTps : 0):F2}x vs n=1");
    }

    private async Task<(double wall, int outToks, List<string> texts)> RunConcurrency(
        BenchEngine ctx, string[] prompts, int n, int maxNewTokens, bool warm)
    {
        var sw = Stopwatch.StartNew();
        var tasks = new List<Task<(int count, string text)>>(n);
        for (int i = 0; i < n; i++)
        {
            string prompt = prompts[i % prompts.Length];
            string reqId = $"{(warm ? "warm" : "run")}-n{n}-{i}";
            tasks.Add(SubmitAndCount(ctx, prompt, maxNewTokens, reqId));
        }
        var done = await Task.WhenAll(tasks);
        sw.Stop();
        return (sw.Elapsed.TotalSeconds, done.Sum(d => d.count), done.Select(d => d.text).ToList());
    }

    private static async Task<(int count, string text)> SubmitAndCount(
        BenchEngine ctx, string prompt, int maxNewTokens, string reqId)
    {
        var history = new List<ChatMessage> { new() { Role = "user", Content = prompt } };
        var tokens = ctx.Renderer.RenderToTokens(
            ctx.Model.Tokenizer,
            ctx.Model.Config?.ChatTemplate,
            history,
            ctx.Model.Config?.Architecture ?? string.Empty,
            addGenerationPrompt: true,
            tools: null,
            enableThinking: false);

        var seq = new SequenceState(reqId, tokens, maxNewTokens, ctx.BlockSize, SamplingConfig.Greedy);
        var handle = ctx.Engine.SubmitRequest(seq);
        int count = 0;
        var outTokens = new List<int>();
        try
        {
            await foreach (var tok in handle.Tokens.ReadAllAsync())
            {
                count++;
                outTokens.Add(tok);
            }
        }
        catch { }
        await handle.Completion;
        string text = ctx.Model.Tokenizer.Decode(outTokens);
        return (count, text);
    }

    // Longest run of repeated identical substrings of length 2-8 (token-loop
    // detector). "abcabcabc" -> 3, "abcdef" -> 1.
    private static int LongestImmediateRepeat(string s)
    {
        if (string.IsNullOrEmpty(s)) return 0;
        int best = 1;
        for (int unit = 2; unit <= 8 && unit * 2 <= s.Length; unit++)
        {
            int run = 1;
            for (int i = 0; i + unit * 2 <= s.Length; i++)
            {
                bool match = true;
                for (int k = 0; k < unit; k++)
                    if (s[i + k] != s[i + unit + k]) { match = false; break; }
                if (match) { run++; if (run > best) best = run; i += unit - 1; }
                else run = 1;
            }
        }
        return best;
    }

    private static string Preview(string s, int maxLen)
    {
        if (string.IsNullOrEmpty(s)) return string.Empty;
        s = s.Replace("\n", " ").Replace("\r", " ").Trim();
        return s.Length <= maxLen ? s : s.Substring(0, maxLen) + "...";
    }

    private static string ResolveModel(string fileName)
    {
        string explicitPath = Environment.GetEnvironmentVariable(EnvModelPath);
        if (!string.IsNullOrEmpty(explicitPath) && File.Exists(explicitPath))
            return explicitPath;
        string dir = Environment.GetEnvironmentVariable(EnvModelDir);
        if (!string.IsNullOrEmpty(dir) && Directory.Exists(dir))
        {
            string candidate = Path.Combine(dir, fileName);
            if (File.Exists(candidate)) return candidate;
        }
        // Fallback to the user's standard model directory.
        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string standard = Path.Combine(home, "work", "model", fileName);
        return File.Exists(standard) ? standard : null;
    }

    private sealed class BenchEngine : IDisposable
    {
        public TensorSharp.Models.ModelBase Model { get; }
        public KVCachePromptRenderer Renderer { get; }
        public InferenceEngine Engine { get; }
        public int BlockSize { get; }

        public BenchEngine(string modelPath)
        {
            BackendType backend = OperatingSystem.IsMacOS() ? BackendType.GgmlMetal : BackendType.GgmlCpu;
            Model = TensorSharp.Models.ModelBase.Create(modelPath, backend);
            Renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
            BlockSize = 256;
            var cfg = new SchedulerConfig
            {
                MaxNumBatchedTokens = 4096,
                MaxNumRunningSequences = 8,
                MaxPrefillChunkSize = 1024,
                NumBlocks = 256,
                BlockSize = BlockSize,
                EnablePrefixCaching = true,
                DecodeQuantumTokens = BlockSize,
            };
            Engine = new InferenceEngine(Model, cfg, NullLogger.Instance);
        }

        public void Dispose()
        {
            Engine.Dispose();
            Model.Dispose();
        }
    }
}
