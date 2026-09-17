// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Hy-MT2-1.8B served every /v1/chat/completions as HTTP 500 ("Continuous-batching
// engine is unavailable for this model"): HunyuanDenseModel implemented neither a
// batched paged forward nor the K/V-state snapshot contract, so the server had no
// engine to schedule it on. These tests pin the snapshot contract on a tiny
// synthetic hunyuan-dense GGUF: extract/inject reproduces an uninterrupted run, and
// interleaved concurrent requests on the engine produce exactly the tokens each
// request produces alone.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using Xunit;

namespace InferenceWeb.Tests;

public class HunyuanDenseServingTests : IDisposable
{
    private readonly string _dir;

    public HunyuanDenseServingTests()
    {
        _dir = Path.Combine(Path.GetTempPath(), "ts-hunyuan-serving-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private string BuildModel() => new DenseDecoderSyntheticModelBuilder
    {
        Architecture = "hunyuan-dense",
        PreTokenizer = "hunyuan-dense",
        IncludeMistralControlTokens = false,
        IncludeQkNorms = true,
        RopeBase = 10000f,
    }.Write(Path.Combine(_dir, "tiny-hunyuan-dense.gguf"));

    /// <summary>Managed CPU by default, so the portable suite needs no native library.
    /// TS_TEST_GGML_BACKEND=cuda|metal|ggml_cpu runs the same contract on a GGML backend,
    /// whose device-side tensor caches are what the invalidation after inject protects.</summary>
    private static BackendType Backend =>
        (Environment.GetEnvironmentVariable("TS_TEST_GGML_BACKEND") ?? "").Trim().ToLowerInvariant() switch
        {
            "cuda" or "ggml_cuda" => BackendType.GgmlCuda,
            "metal" or "ggml_metal" => BackendType.GgmlMetal,
            "ggml_cpu" => BackendType.GgmlCpu,
            _ => BackendType.Cpu,
        };

    private static int ArgMax(float[] logits)
    {
        int best = 0;
        for (int v = 1; v < logits.Length; v++) if (logits[v] > logits[best]) best = v;
        return best;
    }

    [Fact]
    public void HunyuanDense_ExposesTheSnapshotContractTheEngineRequires()
    {
        using var model = ModelBase.Create(BuildModel(), Backend);
        Assert.IsType<HunyuanDenseModel>(model);
        Assert.True(model.SupportsKVStateSnapshot,
            "without a snapshot contract (or ForwardBatch) InferenceEngineHost.TryGetEngine returns null and every chat request is a 500");
        Assert.True(model.SupportsCrossSequenceKvReuse);
        Assert.StartsWith("hunyuan-dense|", model.KVStateFingerprint);
        Assert.True(model.ComputeKVBlockByteSize(16) > 0);
    }

    [Fact]
    public void HunyuanDense_ExtractInjectRoundTrip_ReproducesTheUninterruptedRun()
    {
        using var model = ModelBase.Create(BuildModel(), Backend);
        int[] prompt = Enumerable.Range(0, 24).Select(i => 65 + (i * 7) % 50).ToArray();
        const int next = 90;

        model.ResetKVCache();
        model.ForwardRefill(prompt);
        float[] expected = (float[])model.Forward(new[] { next }).Clone();

        model.ResetKVCache();
        model.ForwardRefill(prompt);
        var bytes = new byte[model.ComputeKVBlockByteSize(prompt.Length)];
        Assert.True(model.TryExtractKVBlock(0, prompt.Length, bytes));

        // Dirty the cache with a different sequence, then restore the snapshot.
        model.ResetKVCache();
        model.ForwardRefill(Enumerable.Range(0, 30).Select(i => 100 + i).ToArray());
        model.ResetKVCache();
        Assert.True(model.TryInjectKVBlock(0, prompt.Length, bytes));

        float[] restored = model.Forward(new[] { next });
        Assert.Equal(expected, restored);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task HunyuanDense_ConcurrentEngineRequests_MatchEachRequestServedAlone(bool batched)
    {
        string previous = Environment.GetEnvironmentVariable("TS_HUNYUAN_BATCHED");
        Environment.SetEnvironmentVariable("TS_HUNYUAN_BATCHED", batched ? null : "0");
        try
        {
            await RunConcurrentEngineRequests(batched);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_HUNYUAN_BATCHED", previous);
        }
    }

    private async Task RunConcurrentEngineRequests(bool batched)
    {
        string path = BuildModel();
        const int maxNewTokens = 10;
        var prompts = new List<int[]>
        {
            Enumerable.Range(0, 19).Select(i => 65 + (i * 7) % 50).ToArray(),
            Enumerable.Range(0, 27).Select(i => 97 + (i * 5) % 26).ToArray(),
            Enumerable.Range(0, 13).Select(i => 48 + (i * 3) % 10).ToArray(),
        };

        // Reference: plain greedy decode, one prompt at a time, straight on the model.
        var expected = new List<int[]>();
        using (var reference = ModelBase.Create(path, Backend))
        {
            foreach (int[] prompt in prompts)
            {
                reference.ResetKVCache();
                float[] logits = reference.ForwardRefill(prompt);
                var produced = new List<int>();
                for (int i = 0; i < maxNewTokens; i++)
                {
                    int token = ArgMax(logits);
                    produced.Add(token);
                    if (reference.Tokenizer.EosTokenIds.Contains(token)) break;
                    if (i + 1 < maxNewTokens) logits = reference.Forward(new[] { token });
                }
                expected.Add(produced.ToArray());
            }
        }

        using var model = ModelBase.Create(path, Backend);
        // Batched: one paged ForwardBatch per step over every running sequence.
        // Not batched: the K/V snapshot path swaps sequences through the linear cache.
        Assert.Equal(batched, ((IBatchedPagedModel)model).BatchedForwardAvailable);
        var cfg = new SchedulerConfig
        {
            MaxNumBatchedTokens = 256,
            MaxNumRunningSequences = 4,
            MaxPrefillChunkSize = 8,
            NumBlocks = 32,
            BlockSize = 16,
            EnablePrefixCaching = true,
            // A tiny quantum forces the executor to swap K/V ownership between the
            // sequences many times inside each request.
            DecodeQuantumTokens = 2,
        };
        using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);

        var handles = prompts.Select((p, i) => engine.SubmitRequest(
            new SequenceState($"hy{i}", p, maxNewTokens, cfg.BlockSize, SamplingConfig.Greedy))).ToList();
        var outputs = await Task.WhenAll(handles.Select(async h =>
        {
            var tokens = new List<int>();
            await foreach (int t in h.Tokens.ReadAllAsync())
                tokens.Add(t);
            await h.Completion;
            return tokens.ToArray();
        }));

        for (int i = 0; i < prompts.Count; i++)
        {
            // The engine may or may not surface a terminating EOS token; compare the
            // tokens both produced.
            int[] got = outputs[i];
            int[] want = expected[i];
            int n = Math.Min(got.Length, want.Length);
            Assert.True(n > 0, $"request {i} produced nothing");
            Assert.Equal(want.Take(n), got.Take(n));
        }
    }

    /// <summary>
    /// A prompt longer than one block, served, finished, then served again: the second
    /// request adopts the first one's full blocks from the prefix-cache index and must
    /// produce the same greedy tokens. Hy-MT2 produced fluent Chinese echo instead of a
    /// translation on every repeat of a 516-token prompt when this went wrong.
    /// </summary>
    [Theory]
    [InlineData("hunyuan-dense", true)]
    [InlineData("hunyuan-dense", false)]
    [InlineData("mistral3", true)]
    public async Task RepeatedLongPrompt_ReusesPrefixBlocks_AndMatchesTheFirstRun(string architecture, bool batched)
    {
        string previous = Environment.GetEnvironmentVariable("TS_HUNYUAN_BATCHED");
        Environment.SetEnvironmentVariable("TS_HUNYUAN_BATCHED", batched ? null : "0");
        try
        {
            var builder = architecture == "mistral3"
                ? new DenseDecoderSyntheticModelBuilder { Architecture = "mistral3" }
                : new DenseDecoderSyntheticModelBuilder
                {
                    Architecture = "hunyuan-dense", PreTokenizer = "hunyuan-dense",
                    IncludeMistralControlTokens = false, IncludeQkNorms = true, RopeBase = 10000f,
                };
            string path = builder.Write(Path.Combine(_dir, architecture + ".gguf"));
            using var model = ModelBase.Create(path, Backend);
            var cfg = new SchedulerConfig
            {
                MaxNumBatchedTokens = 256,
                MaxNumRunningSequences = 4,
                MaxPrefillChunkSize = 64,
                NumBlocks = 32,
                BlockSize = 16,
                EnablePrefixCaching = true,
            };
            using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);
            int[] prompt = Enumerable.Range(0, 43).Select(i => 65 + (i * 11) % 57).ToArray();

            async Task<(int[] Tokens, int Reused)> Serve(string id)
            {
                var handle = engine.SubmitRequest(new SequenceState(id, prompt, 8, cfg.BlockSize, SamplingConfig.Greedy,
                    cacheScope: "repeat-prompt-conversation"));
                var tokens = new List<int>();
                await foreach (int t in handle.Tokens.ReadAllAsync())
                    tokens.Add(t);
                var completion = await handle.Completion;
                return (tokens.ToArray(), completion.PrefixCacheReusedTokens);
            }

            var first = await Serve("first");
            var second = await Serve("second");
            var third = await Serve("third");
            Assert.Equal(0, first.Reused);
            Assert.True(second.Reused > 0, "the repeat should adopt the first run's full blocks");
            Assert.Equal(first.Tokens, second.Tokens);
            Assert.Equal(first.Tokens, third.Tokens);

            // And several repeats at once, all adopting the same shared blocks.
            var concurrent = await Task.WhenAll(Serve("c0"), Serve("c1"), Serve("c2"));
            foreach (var run in concurrent)
                Assert.Equal(first.Tokens, run.Tokens);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_HUNYUAN_BATCHED", previous);
        }
    }
    /// <summary>
    /// The same repeat through conversation scopes: a batched paged step leaves its
    /// blocks only in the model's paged arrays, and the scope salt past the public prefix
    /// decides which of them another conversation may adopt. Both conversations must
    /// still produce the first run's greedy tokens.
    /// </summary>
    [Fact]
    public async Task RepeatedLongPrompt_InAnotherConversation_AdoptsOnlyThePublicPrefix()
    {
        string previous = Environment.GetEnvironmentVariable("TS_HUNYUAN_BATCHED");
        Environment.SetEnvironmentVariable("TS_HUNYUAN_BATCHED", null);
        try
        {
            using var model = ModelBase.Create(BuildModel(), Backend);
            var cfg = new SchedulerConfig
            {
                MaxNumBatchedTokens = 256,
                MaxNumRunningSequences = 4,
                MaxPrefillChunkSize = 64,
                NumBlocks = 32,
                BlockSize = 16,
                EnablePrefixCaching = true,
            };
            using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);
            int[] prompt = Enumerable.Range(0, 43).Select(i => 65 + (i * 11) % 57).ToArray();
            const int publicPrefix = 16;

            async Task<(int[] Tokens, int Reused)> Serve(string id, string scope)
            {
                var handle = engine.SubmitRequest(new SequenceState(id, prompt, 8, cfg.BlockSize, SamplingConfig.Greedy,
                    sharedPrefixTokens: publicPrefix, cacheScope: scope));
                var tokens = new List<int>();
                await foreach (int t in handle.Tokens.ReadAllAsync())
                    tokens.Add(t);
                var completion = await handle.Completion;
                return (tokens.ToArray(), completion.PrefixCacheReusedTokens);
            }

            var a1 = await Serve("A1", "scope-a");
            var b1 = await Serve("B1", "scope-b");
            var a2 = await Serve("A2", "scope-a");
            Assert.Equal(0, a1.Reused);
            Assert.InRange(b1.Reused, 1, publicPrefix);
            Assert.True(a2.Reused > publicPrefix, $"the owner should adopt past the public prefix, reused {a2.Reused}");
            Assert.Equal(a1.Tokens, b1.Tokens);
            Assert.Equal(a1.Tokens, a2.Tokens);
        }
        finally
        {
            Environment.SetEnvironmentVariable("TS_HUNYUAN_BATCHED", previous);
        }
    }
}
