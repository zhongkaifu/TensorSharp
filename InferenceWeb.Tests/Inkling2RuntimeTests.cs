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
/// <see cref="Inkling2RuntimeModel"/> against <see cref="Inkling2Reference"/>, the double-precision forward written from
/// llama.cpp's inkling2 graphs, on the synthetic F32 checkpoints of <see cref="Inkling2SyntheticModelBuilder"/>: a prompt past
/// the sliding window and the relative extent, then decode steps, for the plain and the looped stack.
/// </summary>
public sealed class Inkling2RuntimeTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-inkling2-runtime-" + Guid.NewGuid().ToString("N"));
    private readonly Xunit.Abstractions.ITestOutputHelper _output;

    public Inkling2RuntimeTests(Xunit.Abstractions.ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { }
    }

    private static double RelL2(float[] actual, double[] expected)
    {
        double num = 0, den = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            double d = actual[i] - expected[i];
            num += d * d;
            den += expected[i] * expected[i];
        }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }

    [Fact]
    public void TheFactoryCreatesTheRuntime()
    {
        var builder = new Inkling2SyntheticModelBuilder();
        string path = builder.Write(Path.Combine(_dir, "factory.gguf"));
        using ModelBase model = ModelBase.Create(path, BackendType.Cpu);
        Assert.IsType<Inkling2RuntimeModel>(model);
        Assert.Equal(builder.Slots, model.Config.NumLayers);
        Assert.Equal(builder.Vocab, model.Config.VocabSize);
    }

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void MatchesTheReferenceForward(bool looped) => AssertMatchesReference(BackendType.Cpu, looped, looped ? 5e-5 : 2e-5);

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    /// <summary>ggml_cpu runs the same native lowering in F32 (relL2 ~1e-6 plain, ~2e-5 looped). ggml-cuda takes F16 operands in
    /// the F16 experts' products (~5e-3 plain), and the synthetic loop's three gated passes over random weights amplify that
    /// to ~1e-1: the managed and ggml_cpu runs prove the recipe, ggml-cuda only that its kernels run it.</summary>
    public void PinnedGgmlBackendMatchesTheReferenceForward(bool looped)
        => AssertMatchesReference(TestGates.PinnedGgmlBackend, looped,
            TestGates.PinnedGgmlBackend == BackendType.GgmlCpu ? 1e-4 : looped ? 0.3 : 1e-2);

    /// <summary>Greedy speculative decoding (scripted drafts, partly accepted) emits exactly the plain greedy tokens: a verify
    /// window keeps every layer's convolution inputs (K, V, the attention and MoE outputs) and the commit takes the accepted
    /// rows' windows; the rejected rows' ring cells are written again later.</summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void SpeculativeDecodingMatchesPlainGreedy(bool looped)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = new Inkling2SyntheticModelBuilder { Looped = looped }.Write(Path.Combine(_dir, $"inkling2-spec-{looped}.gguf"));
        SpeculativeRuntimeChecks.AssertMatchesPlainGreedy(() =>
        {
            var model = new Inkling2RuntimeModel(path, BackendType.Cpu);
            model.GraphLoweringEnabled = true;
            return model;
        }, $"inkling2 looped={looped}");
    }

    /// <summary>A fixture whose vocabulary holds the token ids of the shared slot, engine and checkpoint scripts, with conv
    /// windows of 4.5 KB a layer (192 channels x 6 columns), as the real model's are large (80 KB): the GGML graph builder
    /// keeps a state leaf under 4 KB in per-run compute memory (uploaded, then downloaded), not in the device copy a slot
    /// copy on the device reads and writes.</summary>
    private string ScriptFixture(bool looped)
        => new Inkling2SyntheticModelBuilder { Looped = looped, Vocab = 125, VocabPadded = 128, ConvTaps = 7 }
            .Write(Path.Combine(_dir, $"inkling2-script-{looped}.gguf"));

    private static Inkling2RuntimeModel Create(string path, BackendType backend = BackendType.Cpu)
    {
        var model = new Inkling2RuntimeModel(path, backend);
        model.GraphLoweringEnabled = true;
        return model;
    }

    /// <summary>Several requests' slots in batched steps (each slot its own attention rows, rings and conv windows) give
    /// each request's rows alone.</summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void SlotBatchesMatchEachSequenceAlone(bool looped)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = ScriptFixture(looped);
        SlotParity.AssertBatchedSlotsMatchAlone(() => Create(path), looped ? 5e-5 : 1e-5, $"inkling2 looped={looped}");
    }

    /// <summary>The same on the pinned GGML backend, plain stack (ggml-cuda's F16 expert operands round batched rows apart
    /// from lone ones, and the synthetic loop amplifies that; see <see cref="PinnedGgmlBackendMatchesTheReferenceForward"/>).</summary>
    [Fact]
    public void SlotBatchesMatchEachSequenceAloneOnThePinnedGgmlBackend()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = ScriptFixture(looped: false);
        SlotParity.AssertBatchedSlotsMatchAlone(() => Create(path, TestGates.PinnedGgmlBackend),
            TestGates.PinnedGgmlBackend == BackendType.GgmlCpu ? 1e-4 : 2.5e-2, $"inkling2 {TestGates.PinnedGgmlBackend}");
    }

    /// <summary>The engine's greedy streams of three requests served together (batched steps over their slots, prefill
    /// chunks, decode quanta) equal each request's alone.</summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public async Task ConcurrentRequestsAreBatched(bool looped)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = ScriptFixture(looped);
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
                    if (reference.Tokenizer.EosTokenIds.Contains(token)) break;   // the engine's stream ends without it
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

    /// <summary>A shared-prefix checkpoint (attention rows, sliding rings and every layer's conv windows), cloned or written
    /// to a file and read back, continues exactly as a fresh run; on a GGML backend also through the device copy program.</summary>
    [Theory]
    [InlineData("ggml")]
    [InlineData("managed")]
    public void CheckpointsContinueExactly(string executor)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        BackendType backend = executor == "ggml" ? TestGates.PinnedGgmlBackend : BackendType.Cpu;
        string path = ScriptFixture(looped: true);
        CheckpointChecks.AssertCheckpointsContinueExactly(() => Create(path, backend), backend == BackendType.GgmlCuda ? 1e-2 : 1e-5,
            $"inkling2 {backend}", deviceCopies: executor == "ggml");
    }

    private void AssertMatchesReference(BackendType backend, bool looped, double tolerance, int promptLength = 17)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        var builder = new Inkling2SyntheticModelBuilder { Looped = looped };
        string path = builder.Write(Path.Combine(_dir, $"inkling2-{looped}.gguf"));
        var reference = Inkling2Reference.Load(path);
        int[] prompt = Enumerable.Range(0, promptLength).Select(i => 3 + (i * 7) % 55).ToArray();
        int[] decode = { 11, 12, 13, 14, 15, 16, 17, 18 };
        double[][] expected = reference.Logits(prompt.Concat(decode).ToArray());

        using var model = new Inkling2RuntimeModel(path, backend);
        model.GraphLoweringEnabled = true;
        _output.WriteLine($"graph refusal: {model.GraphRefusalForTests ?? "none"}");
        var actual = new List<(int Position, float[] Logits)> { (prompt.Length - 1, (float[])model.ForwardRefill(prompt).Clone()) };
        for (int t = 0; t < decode.Length; t++)
            actual.Add((prompt.Length + t, (float[])model.Forward(new[] { decode[t] }).Clone()));
        Assert.True(model.GraphRuns > 0, "no graph runs");
        double[] rel = actual.Select(a => RelL2(a.Logits, expected[a.Position])).ToArray();
        _output.WriteLine($"{backend} looped={looped}: relL2 per row {string.Join(" ", rel.Select(r => r.ToString("E1")))}");
        for (int i = 0; i < rel.Length; i++)
            Assert.True(rel[i] <= tolerance, $"row {actual[i].Position}: relL2 {rel[i]:E2} > {tolerance:E1}");
    }
}
