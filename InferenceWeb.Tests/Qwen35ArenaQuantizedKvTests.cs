// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The arena batched decode (the ONE fused graph that decodes N>=2 concurrent
// requests together, ggml_ops_qwen35_batched_arena.cpp) used to refuse any
// block-quantized K/V cache. The solo whole-model graph had already been widened
// to Q8_0/Q4_0 on CUDA and Metal; the arena's own gate was simply never updated
// with it. Because every agent config here pins `--kv-cache-dtype q8_0` (Metal
// charges KV twice, so a long agent context runs out of room before it runs out
// of speed), that stale gate meant concurrency NEVER batched in practice: two
// chats open at once fell back to round-robin, one full 27B weight sweep per
// sequence per token, and aggregate throughput stayed at 1x.
//
// Widening a gate on a kernel path nothing had ever run at that dtype is only
// safe if something checks the numbers. That is what this file is for: it drives
// the real model twice at Q8_0 - once solo (which cannot use the arena) and once
// with a second sequence in flight (which must) - and requires the concurrent
// answer to match the solo one. A structural bug in the quantized arena (wrong
// row pitch, a misaligned block offset, an unseeded slot) diverges at the first
// token; the test would fail there rather than in a user's chat.
//
// Opt in with TS_TEST_MODEL_DIR pointing at a directory holding a Qwen 3.5/3.6/3.8
// 27B GGUF. Slow: one 17.9 GB model load.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public class Qwen35ArenaQuantizedKvTests : IDisposable
{
    private const string EnvModelDir = "TS_TEST_MODEL_DIR";

    /// <summary>The ONE pattern the gate and the loader share (see
    /// Qwen35BatchedCorrectnessTests for why that matters).</summary>
    private const string Qwen35Gguf = "qwen3.5-27b|qwen3.6-27b|qwen3.8-27b";

    private const int BlockSize = 256;

    private readonly ITestOutputHelper _output;
    public Qwen35ArenaQuantizedKvTests(ITestOutputHelper output) { _output = output; }

    // The test sets the process-wide dtype; put it back exactly, or every later model in
    // the process gets a q8_0 cache nobody asked for.
    private readonly KvCacheDtype _restoreDtype = KvCacheDtypeConfig.Current;
    private readonly bool _restoreExplicit = KvCacheDtypeConfig.IsExplicitlySet;

    public void Dispose() => KvCacheDtypeConfig.RestoreForTests(_restoreDtype, _restoreExplicit);

    [ModelFact(EnvModelDir, Qwen35Gguf)]
    public async Task ArenaBatchedDecode_WithQ8KvCache_MatchesSoloDecode()
    {
        string dir = Environment.GetEnvironmentVariable(EnvModelDir);
        string modelPath = string.IsNullOrEmpty(dir) ? null : TestGates.FindSmallestGguf(dir, Qwen35Gguf);
        Assert.True(modelPath != null,
            $"[ModelFact] admitted this test but no '{Qwen35Gguf}' GGUF was found under {EnvModelDir}.");

        // The dtype under test. Set before the model is created: ModelBase reads
        // KvCacheDtypeConfig.Current at construction.
        KvCacheDtypeConfig.Set(KvCacheDtype.Q8_0);

        // The process's pinned GGML backend: the native bridge allows one per process.
        BackendType backend = TestGates.PinnedGgmlBackend;
        _output.WriteLine($"[arena-q8] loading {Path.GetFileName(modelPath)} on {backend} with a q8_0 KV cache");

        using var model = ModelBase.Create(modelPath, backend);
        Assert.Equal(KvCacheDtype.Q8_0, model.KvCacheDtype);

        var cfg = new SchedulerConfig
        {
            MaxNumBatchedTokens = 4096,
            MaxNumRunningSequences = 4,
            MaxPrefillChunkSize = 1024,
            NumBlocks = 64,
            BlockSize = BlockSize,
            // Off, so the concurrent run cannot quietly adopt the solo run's state
            // and decode a prefix it never computed through the arena.
            EnablePrefixCaching = false,
            // Rotate every token so both sequences stay in decode together, which
            // is the only state in which the arena graph is reachable.
            DecodeQuantumTokens = 1,
        };
        using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);

        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        List<int> Render(string prompt) => renderer.RenderToTokens(
            model.Tokenizer, model.Config?.ChatTemplate,
            new List<ChatMessage> { new() { Role = "user", Content = prompt } },
            model.Config?.Architecture ?? string.Empty,
            addGenerationPrompt: true, tools: null, enableThinking: false);

        const string subject = "Q: Name the three primary additive colours.\nA:";
        const string companion = "Q: What is the boiling point of water at sea level?\nA:";
        const int maxNewTokens = 16;

        var subjectTokens = Render(subject);
        var companionTokens = Render(companion);

        // ---- Baseline: solo. N=1 never reaches the arena. ----
        model.ResetKVCache();
        var solo = await DrainAsync(engine.SubmitRequest(
            new SequenceState("arena-solo", subjectTokens, maxNewTokens, BlockSize, SamplingConfig.Greedy)));

        // ---- Under test: the same request with a second sequence in flight. ----
        // Capture stderr so a decline (which would silently fall back to the
        // round-robin path and make the comparison vacuous) is detectable.
        var stderr = new StringWriter();
        var previousErr = Console.Error;
        List<int> concurrent, other;
        try
        {
            Console.SetError(stderr);
            model.ResetKVCache();
            var subjectHandle = engine.SubmitRequest(
                new SequenceState("arena-batched", subjectTokens, maxNewTokens, BlockSize, SamplingConfig.Greedy));
            var companionHandle = engine.SubmitRequest(
                new SequenceState("arena-companion", companionTokens, maxNewTokens, BlockSize, SamplingConfig.Greedy));
            var subjectRun = DrainAsync(subjectHandle);
            var companionRun = DrainAsync(companionHandle);
            await Task.WhenAll(subjectRun, companionRun);
            concurrent = await subjectRun;
            other = await companionRun;
        }
        finally
        {
            Console.SetError(previousErr);
        }

        string declines = stderr.ToString();
        _output.WriteLine($"[arena-q8] solo=[{string.Join(",", solo)}]");
        _output.WriteLine($"[arena-q8] batched=[{string.Join(",", concurrent)}]");
        _output.WriteLine($"[arena-q8] companion produced {other.Count} token(s)");
        if (declines.Length > 0)
            _output.WriteLine($"[arena-q8] stderr:\n{declines}");

        // The point of the change: a q8_0 cache must no longer be a reason to refuse.
        Assert.DoesNotContain("KV cache dtype", declines, StringComparison.Ordinal);

        // And it must have actually batched. Matching tokens alone prove nothing
        // here - the round-robin fallback produces correct tokens too - so require
        // the positive signal that arena steps were served.
        // Qwen35Model's arena steps, or the shared runtime's batched slot steps (one dense ggml_cuda /
        // ggml_metal device routes to the runtime).
        long arenaSteps = model switch
        {
            Qwen35Model q35 => q35.ArenaBatchedDecodeSteps,
            StandardDecoderModel runtime => runtime.BatchedForwardCount,
            _ => 0,
        };
        _output.WriteLine($"[arena-q8] arena batched decode steps: {arenaSteps}");
        Assert.True(arenaSteps > 0,
            "the arena batched decode never ran, so this comparison says nothing about it"
            + (declines.Length > 0 ? $"; stderr said: {declines}" : " and it printed no decline"));

        Assert.NotEmpty(solo);
        Assert.NotEmpty(concurrent);
        Assert.NotEmpty(other);

        // Greedy on both runs, so the token streams must agree. A little drift is
        // allowed at the tail (the arena accumulates its attention in a different
        // order), but a structural fault diverges at token 0.
        int compareLen = Math.Min(solo.Count, concurrent.Count);
        int matching = 0;
        while (matching < compareLen && solo[matching] == concurrent[matching]) matching++;
        _output.WriteLine($"[arena-q8] prefix match {matching}/{compareLen}");
        _output.WriteLine($"  solo:    \"{model.Tokenizer.Decode(solo)}\"");
        _output.WriteLine($"  batched: \"{model.Tokenizer.Decode(concurrent)}\"");
        Assert.True(matching >= Math.Max(4, compareLen / 2),
            $"q8_0 arena batched decode diverged from solo decode after {matching}/{compareLen} tokens: "
            + $"solo=[{string.Join(",", solo)}] batched=[{string.Join(",", concurrent)}]");
    }

    private static async Task<List<int>> DrainAsync(InferenceRequestHandle handle)
    {
        var tokens = new List<int>();
        await foreach (var token in handle.Tokens.ReadAllAsync())
            tokens.Add(token);
        await handle.Completion;
        return tokens;
    }
}
