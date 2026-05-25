// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Performance benchmark for Gemma 4's batched paged-attention path
// (`TS_GEMMA4_BATCHED=1`) compared against the per-sequence KV-swap
// fallback that BatchExecutor uses when ForwardBatch throws.
//
// The model is loaded ONCE per benchmark method and exercised through
// both paths back-to-back so the model-load cost (multi-second for E4B
// Q8_0) is amortised. Toggling between paths is done via the
// `_pagedOptIn` token below, which sets/unsets `TS_GEMMA4_BATCHED`
// before each scenario.
//
// Opt-in via TS_TEST_MODEL_DIR pointing at the directory containing
// gemma-4-E4B-it-Q8_0.gguf. The model is ~4 GB so this is a slow test
// (~minutes) and isn't run in CI by default.
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

public class Gemma4BatchedPerfBench
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";
    private const string OptInVar = "TS_GEMMA4_BATCHED";

    private readonly ITestOutputHelper _output;
    public Gemma4BatchedPerfBench(ITestOutputHelper output) { _output = output; }

    // Five short prompts in parallel - representative of a typical chat
    // workload where each request is a few hundred tokens of prompt and
    // we want low p50 latency.
    [Fact]
    public Task Gemma4_ShortPromptsParallel_BatchedVsLegacy() =>
        RunComparison(
            label: "short-parallel",
            numRequests: 5,
            maxNewTokens: 24,
            promptFactory: i => MakeShortPrompts(i));

    // Four long ~500-token prompts in parallel - the regime where the
    // batched paged kernel was expected to show its biggest win over the
    // per-seq KV-swap path (gather + GPU launch overhead amortised across
    // the longer attention compute).
    [Fact]
    public Task Gemma4_LongPromptsParallel_BatchedVsLegacy() =>
        RunComparison(
            label: "long-parallel",
            numRequests: 4,
            maxNewTokens: 6,
            promptFactory: i => MakeLongPrompts(i));

    // Single sequence - sanity check that batched isn't WORSE than legacy
    // on the degenerate batch=1 case. The fixed per-call overheads
    // (graph build, gather) shouldn't drown out the layer compute.
    [Fact]
    public Task Gemma4_SingleSequence_BatchedVsLegacy() =>
        RunComparison(
            label: "single-seq",
            numRequests: 1,
            maxNewTokens: 24,
            promptFactory: i => MakeShortPrompts(i));

    // Eight prompts in parallel - the larger-batch case where the
    // paged-attention amortisation should be most visible. Need
    // SchedulerConfig.MaxNumRunningSequences >= 8 (already set in
    // BenchContext).
    [Fact]
    public Task Gemma4_EightPromptsParallel_BatchedVsLegacy() =>
        RunComparison(
            label: "batch8-parallel",
            numRequests: 8,
            maxNewTokens: 16,
            promptFactory: i => MakeShortPrompts(i));

    // ----------------------------------------------------------------

    private async Task RunComparison(
        string label,
        int numRequests,
        int maxNewTokens,
        Func<int, List<string>> promptFactory)
    {
        // Stable env state for both runs: force the legacy unfused path
        // (no fused-layer-prefill, no fused-decode kernel) so the two
        // forward computations are apples-to-apples on the same op
        // graph. The only difference between the two timed runs below is
        // whether ForwardBatch executes (batched) or throws-and-falls-back
        // to per-seq KV swap (legacy).
        Environment.SetEnvironmentVariable("TS_FUSED_LAYER_PREFILL", "0");

        var modelPath = FindGemma4();
        if (modelPath == null) { _output.WriteLine("[gemma4-perf] no model; skipping"); return; }
        _output.WriteLine($"[gemma4-perf] loading {Path.GetFileName(modelPath)}");

        using var ctx = new BenchContext(modelPath);

        var prompts = promptFactory(numRequests);
        // Warm-up pass through the engine: first request loads CUDA/Metal
        // pipelines, JITs hot paths, populates the prefix cache. Without
        // this the FIRST measured pass eats all the cold-start cost and
        // looks much slower than reality.
        await RunPath(ctx, prompts.Take(1).ToList(), maxNewTokens: 4, optIn: false, warm: true);

        // Legacy first (the BatchExecutor fallback when ForwardBatch
        // throws). Then batched. Order matters less because the prefix
        // cache is per-request and the prompts are diverse.
        var legacy  = await RunPath(ctx, prompts, maxNewTokens, optIn: false, warm: false);
        var batched = await RunPath(ctx, prompts, maxNewTokens, optIn: true,  warm: false);

        Report(label, numRequests, legacy, batched);
    }

    private async Task<RunStats> RunPath(
        BenchContext ctx, List<string> prompts, int maxNewTokens,
        bool optIn, bool warm)
    {
        // The Gemma 4 batched path is now the DEFAULT, so "legacy" means
        // explicitly opting out via TS_GEMMA4_BATCHED=0 (which forces
        // ForwardBatch to throw NotSupportedException, triggering the
        // per-seq KV-swap fallback in BatchExecutor).
        if (optIn) Environment.SetEnvironmentVariable(OptInVar, "1");
        else       Environment.SetEnvironmentVariable(OptInVar, "0");

        // Reset every per-engine state we can to keep the two timed runs
        // independent. The InferenceEngine itself maintains a prefix
        // cache; in this benchmark we deliberately let it survive across
        // the two runs so each path sees the same cache-warm-state, but
        // we still drop the model-internal KV cache (the per-seq path
        // would re-populate it anyway).
        ctx.Model.ResetKVCache();
        ctx.SumLastPromptTokens = 0;

        int n = prompts.Count;
        var sw = Stopwatch.StartNew();
        var tasks = new List<Task<int>>(n);
        for (int i = 0; i < n; i++)
        {
            int reqIdx = i;
            tasks.Add(SubmitAndCount(ctx, prompts[reqIdx], maxNewTokens, $"r{(warm ? "w" : (optIn ? "b" : "l"))}-{reqIdx}"));
        }
        var outCounts = await Task.WhenAll(tasks);
        sw.Stop();

        int totalOut = outCounts.Sum();
        int totalPrompt = ctx.SumLastPromptTokens; // populated by SubmitAndCount
        return new RunStats
        {
            Wall = sw.Elapsed,
            OutputTokens = totalOut,
            PromptTokens = totalPrompt,
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
        _output.WriteLine($"========== [gemma4-perf] {label} (n={n}) ==========");
        _output.WriteLine($"  legacy  : wall={legacySec,7:F2}s out={legacy.OutputTokens,4} prompt={legacy.PromptTokens,5} tps={legacyTps,6:F1}");
        _output.WriteLine($"  batched : wall={batchedSec,7:F2}s out={batched.OutputTokens,4} prompt={batched.PromptTokens,5} tps={batchedTps,6:F1}");
        _output.WriteLine($"  speedup : wall {speedup,5:F2}x   tps {tpsRatio,5:F2}x");
        _output.WriteLine("");
    }

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
            "Q: What is the chemical symbol for gold?\nA:",
            "Q: In which year did the first moon landing happen?\nA:",
        };
        var prompts = new List<string>(n);
        for (int i = 0; i < n; i++)
            prompts.Add(templates[i % templates.Length]);
        return prompts;
    }

    private static List<string> MakeLongPrompts(int n)
    {
        // Each prompt is ~500 tokens of repeated filler text + a short
        // question. The filler exercises the per-layer attention compute
        // (which scales linearly in kv-len) much more than the fixed
        // graph-build / gather overhead, which is where the batched
        // paged kernel should outpace the per-seq KV-swap path.
        string filler = string.Concat(Enumerable.Repeat(
            "The quick brown fox jumps over the lazy dog. ", 80));
        string[] questions =
        {
            "Q: What is two plus two?\nA:",
            "Q: Name a primary color.\nA:",
            "Q: Capital of France?\nA:",
            "Q: Year of the moon landing?\nA:",
        };
        var prompts = new List<string>(n);
        for (int i = 0; i < n; i++)
            prompts.Add(filler + questions[i % questions.Length]);
        return prompts;
    }

    private static string FindGemma4()
    {
        string dir = Environment.GetEnvironmentVariable(EnvModelDir);
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir)) return null;
        return Directory.GetFiles(dir, "*.gguf").FirstOrDefault(p =>
        {
            var n = Path.GetFileName(p).ToLowerInvariant();
            return n.Contains("gemma-4-e4b") && !n.Contains("mmproj") && !n.Contains("assistant");
        });
    }

    private struct RunStats
    {
        public TimeSpan Wall;
        public int OutputTokens;
        public int PromptTokens;
    }

    private sealed class BenchContext : IDisposable
    {
        public TensorSharp.Models.ModelBase Model { get; }
        public KVCachePromptRenderer Renderer { get; }
        public InferenceEngine Engine { get; }
        public int BlockSize { get; }
        public int SumLastPromptTokens; // updated by request callbacks

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
            Engine.Dispose();
            Model.Dispose();
        }
    }
}
