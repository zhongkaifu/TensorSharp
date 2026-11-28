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
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The shared-runtime mechanisms of <see cref="RaonRuntimeModel"/> on the synthetic F32 checkpoint of
/// <see cref="RaonSyntheticModelBuilder"/>: batched slots (each its own MLA rows, RDA state and conv window), continuous
/// batching, checkpoints and speculative verify. The forward itself is held to Transformers on the real weights
/// (<see cref="RaonParityTests"/>); here the pinned GGML backend is held to the managed one.
/// </summary>
public sealed class RaonRuntimeTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-raon-runtime-" + Guid.NewGuid().ToString("N"));

    public RaonRuntimeTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { }
    }

    private string Fixture(string name = "raon") => new RaonSyntheticModelBuilder().Write(Path.Combine(_dir, name + ".gguf"));

    private static RaonRuntimeModel Create(string path, BackendType backend = BackendType.Cpu) => new(path, backend);

    private static int[] Prompt(int length) => Enumerable.Range(0, length).Select(i => 3 + (i * 7) % 120).ToArray();

    [Fact]
    public void TheFactoryCreatesTheRuntime()
    {
        var builder = new RaonSyntheticModelBuilder();
        using ModelBase model = ModelBase.Create(Fixture(), BackendType.Cpu);
        Assert.IsType<RaonRuntimeModel>(model);
        Assert.Equal(builder.Layers, model.Config.NumLayers);
        Assert.Equal(builder.Vocab, model.Config.VocabSize);
    }

    [Fact]
    public void BackendsWithoutGraphsAreRefused()
        => Assert.Throws<NotSupportedException>(() => ModelBase.Create(Fixture(), BackendType.Cuda));

    /// <summary>A prompt in one chunk and in pieces gives the same logits: the RDA state and conv window and the MLA rows carry
    /// across steps. GPU lanes compare two runs whose F32 products of the random weights round differently (ggml-cuda's MMF
    /// and cuBLAS take TF32): six random layers amplify that to 0.13 relL2 on an RTX 5090, 0.032 with NVIDIA_TF32_OVERRIDE=0
    /// (which reaches cuBLAS only); two layers stay within 1e-2.</summary>
    [Theory]
    [InlineData("managed")]
    [InlineData("ggml")]
    public void ChunkedPrefillMatchesOneChunk(string executor)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        BackendType backend = executor == "ggml" ? TestGates.PinnedGgmlBackend : BackendType.Cpu;
        using var model = Create(Fixture(), backend);
        int[] prompt = Prompt(19);
        float[] whole = (float[])model.Forward(prompt).Clone();
        model.ResetKVCache();
        float[] pieces = null;
        foreach (int[] chunk in new[] { prompt[..5], prompt[5..6], prompt[6..] })
            pieces = model.Forward(chunk);
        AssertClose(whole, pieces, backend is BackendType.Cpu or BackendType.GgmlCpu ? 1e-5 : 0.25, $"{backend} chunked");
    }

    /// <summary>The pinned GGML backend's prompt and decode logits are the managed backend's (ggml_cpu ~1e-4 relL2 in F32;
    /// ggml_cuda ~2e-2, its cuBLAS F32 products taking TF32). The logits are copied: a GGML backend's returned array is reused by
    /// its next step.</summary>
    [Fact]
    public void PinnedGgmlBackendMatchesTheManagedBackend()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture();
        int[] prompt = Prompt(17), decode = { 9, 41, 77 };
        List<float[]> Run(BackendType backend)
        {
            using var model = Create(path, backend);
            var logits = new List<float[]> { (float[])model.Forward(prompt).Clone() };
            foreach (int t in decode) logits.Add((float[])model.Forward(new[] { t }).Clone());
            return logits;
        }
        List<float[]> managed = Run(BackendType.Cpu), ggml = Run(TestGates.PinnedGgmlBackend);
        double tolerance = TestGates.PinnedGgmlBackend == BackendType.GgmlCpu ? 5e-4 : 5e-2;
        for (int i = 0; i < managed.Count; i++) AssertClose(managed[i], ggml[i], tolerance, $"step {i}");
    }

    /// <summary>Whole-layer stages answer as one device: each stage's layers run the one device's ops, the hc streams and
    /// the per-layer embeddings (the stage carry) passing from stage to stage, the RDA states with their layers.</summary>
    [Theory]
    [InlineData(2)]
    [InlineData(3)]
    public void PlacementMatchesOneDevice(int stages)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture();
        int[] prompt = Prompt(17), decode = { 9, 41, 77, 5 };
        List<float[]> Run(RaonRuntimeModel model)
        {
            using (model)
            {
                var logits = new List<float[]> { (float[])model.Forward(prompt).Clone() };
                foreach (int t in decode) logits.Add((float[])model.Forward(new[] { t }).Clone());
                Assert.True(model.GraphRuns > 0, "no graph ran");
                return logits;
            }
        }
        List<float[]> one = Run(Create(path));
        var placed = new RaonRuntimeModel(path, BackendType.Cpu, layerSplitDegree: stages);
        Assert.Equal(stages, placed.Mesh.Stages);
        List<float[]> split = Run(placed);
        for (int i = 0; i < one.Count; i++) AssertClose(one[i], split[i], 1e-6, $"{stages} stages, step {i}");
    }

    [Fact]
    public void SpeculativeDecodingMatchesPlainGreedy()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture();
        SpeculativeRuntimeChecks.AssertMatchesPlainGreedy(() => Create(path), "raon1nano");
    }

    [Fact]
    public void SlotBatchesMatchEachSequenceAlone()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture();
        SlotParity.AssertBatchedSlotsMatchAlone(() => Create(path), 1e-5, "raon1nano");
    }

    [Fact]
    public async Task ConcurrentRequestsAreBatched()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture();
        const int maxNewTokens = 8;
        var prompts = new List<int[]>
        {
            Enumerable.Range(0, 19).Select(i => 65 + (i * 7) % 50).ToArray(),
            Enumerable.Range(0, 27).Select(i => 97 + (i * 5) % 26).ToArray(),
            Enumerable.Range(0, 13).Select(i => 48 + (i * 3) % 10).ToArray(),
        };

        var expected = new List<int[]>();
        using (var reference = Create(path))
        {
            foreach (int[] prompt in prompts)
            {
                reference.ResetKVCache();
                float[] logits = reference.ForwardRefill(prompt);
                var produced = new List<int>();
                for (int i = 0; i < maxNewTokens; i++)
                {
                    int token = Array.IndexOf(logits, logits.Max());
                    if (reference.Tokenizer.EosTokenIds.Contains(token)) break;
                    produced.Add(token);
                    if (i + 1 < maxNewTokens) logits = reference.Forward(new[] { token });
                }
                expected.Add(produced.ToArray());
            }
        }

        using var model = Create(path);
        Assert.True(((IBatchedPagedModel)model).SupportsPerSequenceFusedForward);
        var cfg = new SchedulerConfig
        {
            MaxNumBatchedTokens = 256,
            MaxNumRunningSequences = 4,
            MaxPrefillChunkSize = 8,
            NumBlocks = 32,
            BlockSize = 16,
            EnablePrefixCaching = true,
            DecodeQuantumTokens = 2,
        };
        using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);
        var handles = prompts.Select((p, i) => engine.SubmitRequest(
            new SequenceState($"req{i}", p, maxNewTokens, cfg.BlockSize, SamplingConfig.Greedy))).ToList();
        var outputs = await Task.WhenAll(handles.Select(async h =>
        {
            var tokens = new List<int>();
            await foreach (int t in h.Tokens.ReadAllAsync())
                tokens.Add(t);
            await h.Completion;
            return tokens.ToArray();
        })).WaitAsync(TimeSpan.FromMinutes(3));

        Assert.True(model.BatchedForwardCount > 0, "the engine must serve these requests in batched steps over their slots");
        for (int i = 0; i < prompts.Count; i++)
            Assert.Equal(expected[i], outputs[i]);
    }

    [Theory]
    [InlineData("ggml")]
    [InlineData("managed")]
    public void CheckpointsContinueExactly(string executor)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        BackendType backend = executor == "ggml" ? TestGates.PinnedGgmlBackend : BackendType.Cpu;
        string path = Fixture();
        CheckpointChecks.AssertCheckpointsContinueExactly(() => Create(path, backend), backend == BackendType.GgmlCuda ? 1e-2 : 1e-5,
            $"raon1nano {backend}", deviceCopies: executor == "ggml");
    }

    private static void AssertClose(float[] expected, float[] actual, double relL2, string what = null)
    {
        Assert.Equal(expected.Length, actual.Length);
        double num = 0, den = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = actual[i] - expected[i];
            num += d * d;
            den += (double)expected[i] * expected[i];
        }
        double err = Math.Sqrt(num / Math.Max(den, 1e-30));
        Assert.True(err <= relL2, $"{what}: relL2 {err:E3} > {relL2:E1}");
    }
}
