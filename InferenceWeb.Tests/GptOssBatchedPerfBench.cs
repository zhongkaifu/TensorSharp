// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// GptOss batched vs legacy per-seq KV-swap performance bench. Mirrors the
// Nemotron / Qwen 3.5 perf benches: warm up once, run scenarios of n=1,3,5
// parallel sequences, report wall + tps + managed/working-set memory deltas
// for each path.
//
// Opt-in via TS_TEST_MODEL_DIR (containing a `gpt-oss-*.gguf`). Not in default
// CI; ~minutes per scenario.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Runtime.Scheduling;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class GptOssBatchedPerfBench
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";
    private const string OptInVar = "TS_GPTOSS_BATCHED";

    private readonly ITestOutputHelper _output;
    public GptOssBatchedPerfBench(ITestOutputHelper output) { _output = output; }

    [Fact]
    public Task GptOss_BatchedVsLegacy()
        => RunScenarios(new[]
        {
            ("single-seq",     1, 8),
            ("three-parallel", 3, 6),
            ("five-parallel",  5, 6),
        });

    private async Task RunScenarios((string label, int n, int maxNewTokens)[] scenarios)
    {
        var modelPath = FindGptOss();
        if (modelPath == null) { _output.WriteLine("[gptoss-perf] no model; skipping"); return; }
        _output.WriteLine($"[gptoss-perf] loading {Path.GetFileName(modelPath)} (this takes a while)");

        using var ctx = new BenchContext(modelPath);

        await RunPath(ctx, MakeShortPrompts(1), maxNewTokens: 2, optIn: false, warm: true);

        foreach (var (label, n, maxNewTokens) in scenarios)
        {
            var prompts = MakeShortPrompts(n);
            var legacy  = await RunPath(ctx, prompts, maxNewTokens, optIn: false, warm: false);
            var batched = await RunPath(ctx, prompts, maxNewTokens, optIn: true,  warm: false);
            Report(label, n, legacy, batched);
        }
    }

    private async Task<RunStats> RunPath(
        BenchContext ctx, List<string> prompts, int maxNewTokens,
        bool optIn, bool warm)
    {
        Environment.SetEnvironmentVariable(OptInVar, optIn ? "1" : "0");
        ctx.Model.ResetKVCache();
        ctx.SumLastPromptTokens = 0;

        GC.Collect(2, GCCollectionMode.Forced, blocking: true);
        GC.WaitForPendingFinalizers();
        GC.Collect(2, GCCollectionMode.Forced, blocking: true);
        long managedBefore = GC.GetTotalMemory(forceFullCollection: false);
        long workingSetBefore = Process.GetCurrentProcess().WorkingSet64;
        long peakManaged = managedBefore;
        long peakWorkingSet = workingSetBefore;

        int n = prompts.Count;
        var sw = Stopwatch.StartNew();
        var tasks = new List<Task<int>>(n);
        for (int i = 0; i < n; i++)
        {
            int reqIdx = i;
            tasks.Add(SubmitAndCount(ctx, prompts[reqIdx], maxNewTokens,
                $"r{(warm ? "w" : (optIn ? "b" : "l"))}-{reqIdx}"));
        }

        using var cts = new System.Threading.CancellationTokenSource();
        var samplingTask = Task.Run(async () =>
        {
            while (!cts.IsCancellationRequested)
            {
                long m = GC.GetTotalMemory(forceFullCollection: false);
                long ws = Process.GetCurrentProcess().WorkingSet64;
                if (m > peakManaged) peakManaged = m;
                if (ws > peakWorkingSet) peakWorkingSet = ws;
                try { await Task.Delay(50, cts.Token); } catch { }
            }
        });

        var outCounts = await Task.WhenAll(tasks);
        sw.Stop();
        cts.Cancel();
        try { await samplingTask; } catch { }

        long managedAfter = GC.GetTotalMemory(forceFullCollection: false);
        long workingSetAfter = Process.GetCurrentProcess().WorkingSet64;

        return new RunStats
        {
            Wall = sw.Elapsed,
            OutputTokens = outCounts.Sum(),
            PromptTokens = ctx.SumLastPromptTokens,
            ManagedBeforeBytes = managedBefore,
            ManagedAfterBytes  = managedAfter,
            ManagedPeakBytes   = peakManaged,
            WorkingSetBefore   = workingSetBefore,
            WorkingSetAfter    = workingSetAfter,
            WorkingSetPeak     = peakWorkingSet,
        };
    }

    private async Task<int> SubmitAndCount(
        BenchContext ctx, string prompt, int maxNewTokens, string reqId)
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
        System.Threading.Interlocked.Add(ref ctx.SumLastPromptTokens, tokens.Count);

        var seq = new SequenceState(reqId, tokens, maxNewTokens, ctx.BlockSize, SamplingConfig.Default);
        var handle = ctx.Engine.SubmitRequest(seq);
        int count = 0;
        try
        {
            await foreach (var _ in handle.Tokens.ReadAllAsync())
                count++;
        }
        catch { }
        await handle.Completion;
        return count;
    }

    private void Report(string label, int n, RunStats legacy, RunStats batched)
    {
        double legacySec  = legacy.Wall.TotalSeconds;
        double batchedSec = batched.Wall.TotalSeconds;
        double legacyTps  = legacySec  > 0 ? legacy.OutputTokens  / legacySec  : 0;
        double batchedTps = batchedSec > 0 ? batched.OutputTokens / batchedSec : 0;
        double speedup    = legacySec  > 0 ? legacySec / Math.Max(batchedSec, 1e-9) : 0;
        double tpsRatio   = legacyTps  > 0 ? batchedTps / legacyTps : 0;

        _output.WriteLine("");
        _output.WriteLine($"========== [gptoss-perf] {label} (n={n}) ==========");
        _output.WriteLine($"  legacy  : wall={legacySec,7:F2}s out={legacy.OutputTokens,4} prompt={legacy.PromptTokens,5} tps={legacyTps,6:F2}");
        _output.WriteLine($"  batched : wall={batchedSec,7:F2}s out={batched.OutputTokens,4} prompt={batched.PromptTokens,5} tps={batchedTps,6:F2}");
        _output.WriteLine($"  speedup : wall {speedup,5:F2}x   tps {tpsRatio,5:F2}x");
        _output.WriteLine($"  memory legacy : managed peak={MB(legacy.ManagedPeakBytes),7:F1} MiB  delta={MB(legacy.ManagedAfterBytes - legacy.ManagedBeforeBytes),+7:F1} MiB  ws peak={MB(legacy.WorkingSetPeak),7:F1} MiB  delta={MB(legacy.WorkingSetAfter - legacy.WorkingSetBefore),+7:F1} MiB");
        _output.WriteLine($"  memory batched: managed peak={MB(batched.ManagedPeakBytes),7:F1} MiB  delta={MB(batched.ManagedAfterBytes - batched.ManagedBeforeBytes),+7:F1} MiB  ws peak={MB(batched.WorkingSetPeak),7:F1} MiB  delta={MB(batched.WorkingSetAfter - batched.WorkingSetBefore),+7:F1} MiB");
        _output.WriteLine("");
    }

    private static double MB(long bytes) => bytes / (1024.0 * 1024.0);

    private static List<string> MakeShortPrompts(int n)
    {
        string[] templates =
        {
            "Q: What is two plus two? Answer with one sentence.\nA:",
            "Q: Write a single-sentence haiku about computers.\nA:",
            "Q: Name three colors. Reply with a comma-separated list.\nA:",
            "Q: What is the boiling point of water at sea level?\nA:",
            "Q: Translate 'good morning' to Spanish.\nA:",
            "Q: Who wrote Hamlet?\nA:",
        };
        var prompts = new List<string>(n);
        for (int i = 0; i < n; i++)
            prompts.Add(templates[i % templates.Length]);
        return prompts;
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

    private struct RunStats
    {
        public TimeSpan Wall;
        public int OutputTokens;
        public int PromptTokens;
        public long ManagedBeforeBytes;
        public long ManagedAfterBytes;
        public long ManagedPeakBytes;
        public long WorkingSetBefore;
        public long WorkingSetAfter;
        public long WorkingSetPeak;
    }

    private sealed class BenchContext : IDisposable
    {
        public TensorSharp.Models.ModelBase Model { get; }
        public KVCachePromptRenderer Renderer { get; }
        public InferenceEngine Engine { get; }
        public int BlockSize { get; }
        public int SumLastPromptTokens;

        public BenchContext(string modelPath)
        {
            BackendType backend = OperatingSystem.IsMacOS()
                ? BackendType.GgmlMetal : BackendType.GgmlCpu;
            Model = TensorSharp.Models.ModelBase.Create(modelPath, backend);
            Renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
            BlockSize = 256;
            var cfg = new SchedulerConfig
            {
                MaxNumBatchedTokens = 4096,
                MaxNumRunningSequences = 8,
                MaxPrefillChunkSize = 1024,
                NumBlocks = 128,
                BlockSize = BlockSize,
                EnablePrefixCaching = true,
                DecodeQuantumTokens = BlockSize,
            };
            Engine = new InferenceEngine(Model, cfg, NullLogger.Instance);
        }

        public void Dispose()
        {
            Engine?.Dispose();
            Model?.Dispose();
        }
    }
}
