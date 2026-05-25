// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Nemotron 3 batched paged-attention correctness vs the legacy per-seq KV-swap.
// Mirror of Qwen35BatchedCorrectnessTests: greedy sampling on both paths for the
// same prompt, assert at least 50% prefix match across the first several tokens
// (some FP drift is expected with batched-path numerical reordering; structural
// divergence would produce near-0% match).
//
// Opt-in via TS_TEST_MODEL_DIR pointing at the directory containing the
// Nemotron 3 GGUF. Slow (model load + two forward runs per prompt) and the
// 30B-A3B IQ2_XXS variant is CPU-bound on Apple Metal — single run can take
// several minutes per scenario, so this is not in default CI.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Runtime.Scheduling;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class NemotronBatchedCorrectnessTests
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";
    private const string OptInVar = "TS_NEMOTRON_BATCHED";

    private readonly ITestOutputHelper _output;
    public NemotronBatchedCorrectnessTests(ITestOutputHelper output) { _output = output; }

    [Fact]
    public async Task Nemotron_Greedy_LegacyAndBatchedAgree()
    {
        var modelPath = FindNemotron();
        if (modelPath == null) { _output.WriteLine("[nemo-corr] no model; skipping"); return; }
        _output.WriteLine($"[nemo-corr] loading {Path.GetFileName(modelPath)}");

        using var ctx = new CorrCtx(modelPath);

        string[] prompts =
        {
            "Q: What is two plus two?\nA:",
            "Q: Who wrote Hamlet?\nA:",
            "Q: Translate 'hello' to French.\nA:",
        };

        const int maxNewTokens = 8;
        int totalCompared = 0;
        int totalMatching = 0;

        foreach (var prompt in prompts)
        {
            // Reset model-internal Mamba2 + KV state between runs so each path
            // sees a freshly-zeroed cache; without this state from path A leaks
            // into path B.
            ctx.Model.ResetKVCache();
            var legacyTokens = await GenerateGreedy(ctx, prompt, maxNewTokens, optIn: false);

            ctx.Model.ResetKVCache();
            var batchedTokens = await GenerateGreedy(ctx, prompt, maxNewTokens, optIn: true);

            int matchPrefix = 0;
            int compareLen = Math.Min(legacyTokens.Count, batchedTokens.Count);
            for (int i = 0; i < compareLen; i++)
            {
                if (legacyTokens[i] == batchedTokens[i]) matchPrefix++;
                else break;
            }
            totalCompared += compareLen;
            totalMatching += matchPrefix;

            string legacyText = ctx.Model.Tokenizer.Decode(legacyTokens);
            string batchedText = ctx.Model.Tokenizer.Decode(batchedTokens);
            _output.WriteLine(
                $"[nemo-corr] prompt=\"{Trim(prompt, 50)}\" " +
                $"legacy=[{string.Join(",", legacyTokens)}] " +
                $"batched=[{string.Join(",", batchedTokens)}] " +
                $"matchPrefix={matchPrefix}/{compareLen}");
            _output.WriteLine($"  legacy:  \"{Trim(legacyText, 60)}\"");
            _output.WriteLine($"  batched: \"{Trim(batchedText, 60)}\"");
        }

        double matchRate = totalCompared > 0 ? (double)totalMatching / totalCompared : 0;
        _output.WriteLine($"[nemo-corr] overall prefix-match {totalMatching}/{totalCompared} = {matchRate:P0}");

        Xunit.Assert.True(matchRate >= 0.5,
            $"Legacy/batched prefix-match rate {matchRate:P0} below 50% — structural divergence suspected.");
    }

    private async Task<List<int>> GenerateGreedy(CorrCtx ctx, string prompt, int maxNewTokens, bool optIn)
    {
        Environment.SetEnvironmentVariable(OptInVar, optIn ? "1" : "0");

        var history = new List<ChatMessage> { new() { Role = "user", Content = prompt } };
        var tokens = ctx.Renderer.RenderToTokens(
            ctx.Model.Tokenizer, ctx.Model.Config?.ChatTemplate, history,
            ctx.Model.Config?.Architecture ?? string.Empty,
            addGenerationPrompt: true, tools: null, enableThinking: false);

        string reqId = $"corr-{(optIn ? "b" : "l")}-{Guid.NewGuid():N}";
        var seq = new SequenceState(reqId, tokens, maxNewTokens, ctx.BlockSize, SamplingConfig.Greedy);
        var handle = ctx.Engine.SubmitRequest(seq);
        var outs = new List<int>();
        try
        {
            await foreach (var tok in handle.Tokens.ReadAllAsync())
                outs.Add(tok);
        }
        catch { /* engine surfaces errors via Completion */ }
        await handle.Completion;
        return outs;
    }

    private static string Trim(string s, int len)
        => string.IsNullOrEmpty(s) ? string.Empty
           : (s.Length <= len ? s : s.Substring(0, len) + "...").Replace("\n", "\\n");

    private static string FindNemotron()
    {
        string dir = Environment.GetEnvironmentVariable(EnvModelDir);
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir)) return null;
        return Directory.GetFiles(dir, "*.gguf").Where(p =>
        {
            var n = Path.GetFileName(p).ToLowerInvariant();
            return n.Contains("nemotron") && !n.Contains("mmproj");
        }).OrderBy(p => Path.GetFileName(p)).FirstOrDefault();
    }

    private sealed class CorrCtx : IDisposable
    {
        public TensorSharp.Models.ModelBase Model { get; }
        public KVCachePromptRenderer Renderer { get; }
        public InferenceEngine Engine { get; }
        public int BlockSize { get; }

        public CorrCtx(string modelPath)
        {
            BackendType backend = OperatingSystem.IsMacOS()
                ? BackendType.GgmlMetal : BackendType.GgmlCpu;
            Model = TensorSharp.Models.ModelBase.Create(modelPath, backend);
            Renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
            BlockSize = 256;
            var cfg = new SchedulerConfig
            {
                MaxNumBatchedTokens = 4096,
                MaxNumRunningSequences = 4,
                MaxPrefillChunkSize = 1024,
                NumBlocks = 64,
                BlockSize = BlockSize,
                EnablePrefixCaching = false,
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
