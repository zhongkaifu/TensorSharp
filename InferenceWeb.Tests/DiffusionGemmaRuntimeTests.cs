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
using System.Threading;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// DiffusionGemma on the shared decoder runtime (<see cref="DiffusionGemmaModel"/>) over a synthetic diffusion-gemma file on
/// the 26B-A4B backbone (<see cref="Gemma4MoeSyntheticModelBuilder"/> with <c>Diffusion</c>: fused gate|up experts, V = K
/// global layers, a 32-token window, a 16-row canvas, the self-conditioning MLP and the encoder layer scales): the canvas
/// step's attention (every canvas row reads every other, the prompt from the first canvas row's window), steps that commit
/// nothing, a block's prompt that reuses the cached rows, the on-device sampler held to the host's, structured reads, the
/// pinned GGML backend against the managed one, and the mechanisms the runtime provides it (batched canvases, routed experts
/// on the host). The real checkpoint was held to the legacy model on an A40 by eng/validation/DiffusionGemmaAB.
/// </summary>
[Collection("MoeCpuOffloadConfig")]   // the host-experts test sets the process-wide expert-offload policy
public sealed class DiffusionGemmaRuntimeTests : IDisposable
{
    private const int C = 16;
    private const double NoiseTolerance = 0.025;

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-diffusiongemma-runtime-" + Guid.NewGuid().ToString("N"));
    private readonly ITestOutputHelper _output;

    public DiffusionGemmaRuntimeTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
    }

    public void Dispose()
    {
        MoeCpuOffloadConfig.Reset();
        try { Directory.Delete(_dir, recursive: true); } catch { }
    }

    private string Fixture(string name = "moe", bool[] pattern = null)
    {
        string path = Path.Combine(_dir, name + ".gguf");
        if (File.Exists(path)) return path;
        var builder = pattern == null
            ? new Gemma4MoeSyntheticModelBuilder { Diffusion = true, CanvasLength = C }
            : new Gemma4MoeSyntheticModelBuilder { Diffusion = true, CanvasLength = C, Pattern = pattern };
        return builder.Write(path);
    }

    private static DiffusionGemmaModel Create(string path, BackendType backend = BackendType.Cpu) => new(path, backend);

    private static BackendType Executor(string executor) => executor == "ggml" ? TestGates.PinnedGgmlBackend : BackendType.Cpu;

    private static int[] Tokens(int count, int salt) => Enumerable.Range(0, count).Select(i => 5 + (i * 7 + salt) % 200).ToArray();

    private static double RelL2(float[] a, float[] b)
    {
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++) { double d = a[i] - b[i]; num += d * d; den += (double)b[i] * b[i]; }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }

    // ---------------------------------------------------------------- the model

    [Fact]
    public void TheFactoryCreatesTheRuntime()
    {
        using ModelBase model = ModelBase.Create(Fixture(), BackendType.Cpu);
        var diffusion = Assert.IsType<DiffusionGemmaModel>(model);
        Assert.IsAssignableFrom<Gemma4DecoderModel>(model);
        Assert.Equal(C, diffusion.CanvasLength);
        Assert.Equal(Gemma4MoeSyntheticModelBuilder.MaskToken, diffusion.MaskTokenId);
        Assert.True(diffusion.SelfConditioningEnabled);
        Assert.True(diffusion.SupportsPromptKvCache);
    }

    [Fact]
    public void AutoregressiveForwardIsRefused()
    {
        using var model = Create(Fixture());
        Assert.Throws<NotSupportedException>(() => model.Forward(new[] { 1, 5, 6 }));
    }

    /// <summary>
    /// The canvas step's attention, on one sliding-window layer (so a prompt row's keys are its embedding's alone): a canvas
    /// row reads every other canvas row, and the prompt rows from the first canvas row's window on (positions &gt; P - W, the
    /// legacy model's P - swa + 1 for every canvas row), none before it.
    /// </summary>
    [Fact]
    public void CanvasRowsReadEachOtherAndThePromptFromTheFirstRowsWindow()
    {
        using var model = Create(Fixture("one-local", new[] { true }));
        model.SelfConditioningEnabled = false;
        const int window = 32, p = 60;
        int floor = p - window + 1;
        int[] prompt = Tokens(p, 1);
        int[] canvas = Tokens(C, 7);
        float[] Logits(int[] pr, int[] cv)
        {
            model.PrefillPrompt(pr);
            return (float[])model.DecodeCanvas(cv, null, 0f, 1f).Clone();
        }
        float[] baseLogits = Logits(prompt, canvas);
        int vocab = model.VocabSize;
        float[] Row(float[] l, int r) => l.AsSpan(r * vocab, vocab).ToArray();

        // A prompt token below the window's first position does not reach the canvas; the first one inside does.
        int[] below = (int[])prompt.Clone();
        below[floor - 1] = 250 - below[floor - 1] % 7;
        int[] inside = (int[])prompt.Clone();
        inside[floor] = 250 - inside[floor] % 7;
        Assert.Equal(baseLogits, Logits(below, canvas));
        Assert.NotEqual(baseLogits, Logits(inside, canvas));

        // The last canvas token reaches the first canvas row (both directions), and the first reaches the last.
        int[] lastChanged = (int[])canvas.Clone();
        lastChanged[C - 1] = 251;
        int[] firstChanged = (int[])canvas.Clone();
        firstChanged[0] = 251;
        Assert.NotEqual(Row(baseLogits, 0), Row(Logits(prompt, lastChanged), 0));
        Assert.NotEqual(Row(baseLogits, C - 1), Row(Logits(prompt, firstChanged), C - 1));
    }

    /// <summary>A canvas step commits nothing: the same canvas after another one reads the same prompt rows.</summary>
    [Theory]
    [InlineData("managed")]
    [InlineData("ggml")]
    public void CanvasStepsCommitNothing(string executor)
    {
        using var model = Create(Fixture(), Executor(executor));
        model.PrefillPrompt(Tokens(45, 3));
        float[] first = (float[])model.DecodeCanvas(Tokens(C, 1), null, 0f, 1f).Clone();
        model.DecodeCanvas(Tokens(C, 9), null, 0f, 1f);
        float[] again = (float[])model.DecodeCanvas(Tokens(C, 1), null, 0f, 1f).Clone();
        Assert.Equal(first, again);
    }

    /// <summary>A block's prompt extends the last one's: the cached rows are kept and only the new ones prefill, and the canvas
    /// reads what a fresh prefill of the whole prompt gives (here past the 32-token window, with the canvas on a ring's second
    /// lap on no layer).</summary>
    [Theory]
    [InlineData("managed")]
    [InlineData("ggml")]
    public void CheckpointsContinueExactly(string executor)
    {
        BackendType backend = Executor(executor);
        int[] first = Tokens(27, 5), whole = first.Concat(Tokens(C, 11)).ToArray();
        int[] canvas = Tokens(C, 2);
        float[] continued, fresh;
        using (var model = Create(Fixture(), backend))
        {
            model.PrefillPrompt(first);
            model.DecodeCanvas(canvas, null, 0f, 1f);
            model.PrefillPrompt(whole);
            continued = (float[])model.DecodeCanvas(canvas, null, 0f, 1f).Clone();
        }
        using (var model = Create(Fixture(), backend))
        {
            model.PrefillPrompt(whole);
            fresh = (float[])model.DecodeCanvas(canvas, null, 0f, 1f).Clone();
        }
        double rel = RelL2(continued, fresh);
        _output.WriteLine($"{executor}: continued vs fresh relL2 {rel:E3}");
        Assert.True(rel <= 1e-4, $"continued block prompt: relL2 {rel:E2}");
    }

    /// <summary>The unified call (prompt and canvas together) is the prompt-cached one.</summary>
    [Fact]
    public void ForwardCanvasIsThePromptCachedStep()
    {
        using var model = Create(Fixture());
        int[] prompt = Tokens(20, 4), canvas = Tokens(C, 6);
        float[] unified = (float[])model.ForwardCanvas(prompt.Concat(canvas).ToArray(), prompt.Length).Clone();
        model.ResetKVCache();
        model.PrefillPrompt(prompt);
        Assert.Equal(unified, model.DecodeCanvas(canvas, null, 0f, 1f));
    }

    /// <summary>The pinned GGML backend's canvas logits (with self-conditioning from the step before) are the managed
    /// backend's within summation-order noise.</summary>
    [Fact]
    public void GgmlMatchesManaged()
    {
        int[] prompt = Tokens(45, 8), canvas = Tokens(C, 3);
        List<float[]> Run(BackendType backend)
        {
            using var model = Create(Fixture(), backend);
            model.PrefillPrompt(prompt);
            float[] a = (float[])model.DecodeCanvas(canvas, null, 0f, 1f).Clone();
            float[] b = (float[])model.DecodeCanvas(Tokens(C, 4), a, 1f, 1.25f).Clone();
            return new List<float[]> { a, b };
        }
        List<float[]> managed = Run(BackendType.Cpu), ggml = Run(TestGates.PinnedGgmlBackend);
        for (int i = 0; i < managed.Count; i++)
        {
            double rel = RelL2(ggml[i], managed[i]);
            _output.WriteLine($"{TestGates.PinnedGgmlBackend} step {i}: relL2 {rel:E3}");
            Assert.True(rel <= NoiseTolerance, $"step {i}: relL2 {rel:E2}");
        }
    }

    // ---------------------------------------------------------------- sampling

    /// <summary>The graph's sampler (argmax, entropy, the multinomial draw, the top-K for self-conditioning) is the host
    /// sampler's arithmetic over the same logits.</summary>
    [Fact]
    public void DeviceSamplerMatchesTheHostSampler()
    {
        using var model = Create(Fixture());
        model.DeviceSamplingForTests = true;
        model.PrefillPrompt(Tokens(30, 2));
        int[] canvas = Tokens(C, 5);
        const float tempInv = 1.6f;
        int vocab = model.VocabSize, k = model.SelfCondTopK;
        var rng = new Random(3);
        float[] u = Enumerable.Range(0, C).Select(_ => (float)rng.NextDouble()).ToArray();
        int[] argmax = new int[C], sampled = new int[C], topIds = new int[C * k];
        float[] entropy = new float[C], topProbs = new float[C * k];
        Assert.True(model.DecodeCanvasSampled(canvas, null, null, 0f, tempInv, u, k, argmax, entropy, sampled, topIds, topProbs));
        float[] logits = model.DecodeCanvas(canvas, null, 0f, 1f);
        int sampledAgree = 0;
        for (int pos = 0; pos < C; pos++)
        {
            var row = logits.AsSpan(pos * vocab, vocab).ToArray().Select(v => v * tempInv).ToArray();
            int best = Array.IndexOf(row, row.Max());
            double max = row[best], z = row.Sum(v => Math.Exp(v - max));
            double h = Math.Log(z) + max - row.Sum(v => Math.Exp(v - max) * v) / z;
            double target = u[pos] * z, cum = 0;
            int draw = vocab - 1;
            for (int t = 0; t < vocab; t++) { cum += Math.Exp(row[t] - max); if (cum >= target) { draw = t; break; } }
            Assert.Equal(best, argmax[pos]);
            Assert.True(Math.Abs(h - entropy[pos]) < 1e-3, $"position {pos}: entropy {entropy[pos]} against {h}");
            if (draw == sampled[pos]) sampledAgree++;
            var top = Enumerable.Range(0, vocab).OrderByDescending(t => row[t]).Take(k).ToArray();
            Assert.Equal(top.Order(), topIds.Skip(pos * k).Take(k).Order());
            double tmax = top.Max(t => (double)row[t]), tz = top.Sum(t => Math.Exp(row[t] - tmax));
            for (int j = 0; j < k; j++)
            {
                int id = topIds[pos * k + j];
                Assert.True(Math.Abs(Math.Exp(row[id] - tmax) / tz - topProbs[pos * k + j]) < 1e-4, $"position {pos}: top-K weight of {id}");
            }
        }
        // A draw at a cumulative-sum boundary may round either way; the rest are the host's.
        Assert.True(sampledAgree >= C - 1, $"{sampledAgree}/{C} draws match the host's");
    }

    /// <summary>A whole generation runs on either sampler and stops within its block budget.</summary>
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void GenerateRunsOnEitherSampler(bool device)
    {
        using var model = Create(Fixture());
        model.DeviceSamplingForTests = device;
        var p = new DiffusionEbParams { MaxDenoisingSteps = 4, Seed = 3, MaxBlocks = 2, ConfidenceThreshold = -1f, StabilityThreshold = int.MaxValue };
        List<int> tokens = new DiffusionGemmaSampler(model).Generate(Tokens(12, 1), p);
        _output.WriteLine($"device={device}: {tokens.Count} tokens [{string.Join(",", tokens.Take(16))}]");
        Assert.InRange(tokens.Count, 0, 2 * C);
        Assert.All(tokens, t => Assert.InRange(t, 0, model.VocabSize - 1));
    }

    // ---------------------------------------------------------------- the runtime's mechanisms

    /// <summary>Several sequences' canvases in one step read each sequence's own cache: each sequence's logits are its own
    /// step's alone.</summary>
    [Theory]
    [InlineData("managed")]
    [InlineData("ggml")]
    public void SlotBatchesMatchEachSequenceAlone(string executor)
    {
        using var model = Create(Fixture(), Executor(executor));
        DiffusionSeqState a = model.CreateSeqState(), b = model.CreateSeqState();
        try
        {
            model.PrefillSeq(a, Tokens(45, 1));
            model.PrefillSeq(b, Tokens(23, 9));
            int[] ca = Tokens(C, 2), cb = Tokens(C, 8);
            float[] aloneA = (float[])model.DecodeCanvasSeq(a, ca, null, 0f, 1f).Clone();
            float[] aloneB = (float[])model.DecodeCanvasSeq(b, cb, null, 0f, 1f).Clone();
            float[][] together = model.DecodeCanvasBatched(new[] { a, b }, new[] { ca, cb }, new float[2][], new float[2], new[] { 1f, 1f });
            double ra = RelL2(together[0], aloneA), rb = RelL2(together[1], aloneB);
            _output.WriteLine($"{executor}: batched vs alone relL2 {ra:E3} / {rb:E3}");
            Assert.True(ra <= NoiseTolerance && rb <= NoiseTolerance, $"relL2 {ra:E2} / {rb:E2}");
        }
        finally
        {
            model.DisposeSeqState(a);
            model.DisposeSeqState(b);
        }
    }

    /// <summary>The scheduler's batched block runs every live request's canvas in one step per denoising step.</summary>
    [Fact]
    public void ConcurrentRequestsAreBatched()
    {
        using var model = Create(Fixture());
        var p = new DiffusionEbParams { MaxDenoisingSteps = 3, Seed = 5, MaxBlocks = 1, ConfidenceThreshold = -1f, StabilityThreshold = int.MaxValue };
        DiffusionSeqState a = model.CreateSeqState(), b = model.CreateSeqState();
        try
        {
            var runs = new[]
            {
                new DiffusionSeqRun(Tokens(20, 1), p, a, CancellationToken.None, null),
                new DiffusionSeqRun(Tokens(31, 4), p, b, CancellationToken.None, null),
            };
            var sampler = new DiffusionGemmaSampler(model) { BatchedForward = true };
            sampler.RunBlockBatched(runs);
            Assert.Equal(p.MaxDenoisingSteps, model.BatchedCanvasStepsForTests);
            Assert.All(runs, r => Assert.True(r.Response.Count > 0 || r.Done));
        }
        finally
        {
            model.DisposeSeqState(a);
            model.DisposeSeqState(b);
        }
    }

    /// <summary>Routed experts kept in host memory (--cpu-moe) give the device experts' canvas logits.</summary>
    [Theory]
    [InlineData("managed")]
    [InlineData("ggml")]
    public void HostExpertsMatchDeviceExperts(string executor)
    {
        BackendType backend = Executor(executor);
        int[] prompt = Tokens(40, 6), canvas = Tokens(C, 3);
        float[] Run()
        {
            using var model = Create(Fixture(), backend);
            model.PrefillPrompt(prompt);
            return (float[])model.DecodeCanvas(canvas, null, 0f, 1f).Clone();
        }
        MoeCpuOffloadConfig.Reset();
        float[] device = Run();
        MoeCpuOffloadConfig.SetAllLayers();
        try
        {
            double rel = RelL2(Run(), device);
            _output.WriteLine($"--cpu-moe {backend}: relL2 {rel:E3}");
            Assert.True(rel <= NoiseTolerance, $"--cpu-moe {backend}: relL2 {rel:E2}");
        }
        finally
        {
            MoeCpuOffloadConfig.Reset();
        }
    }

    /// <summary>A structured read's few rows and labels are its full-vocabulary reference's.</summary>
    [Fact]
    public void StructuredReadsMatchTheirReference()
    {
        using var model = Create(Fixture());
        int[] prompt = Tokens(25, 3), canvas = Enumerable.Repeat(model.MaskTokenId, 8).ToArray();
        int[] positions = { 0, 3, 7 };
        int[][] labels = { new[] { 40, 41, 42 }, new[] { 99, 7 }, new[] { 10, 11, 12, 13 } };
        float[][] fast = model.ReadStructured(prompt, canvas, positions, labels);
        float[][] reference = model.ReadStructuredReference(prompt, canvas, positions, labels);
        for (int r = 0; r < fast.Length; r++)
        {
            Assert.Equal(1f, fast[r].Sum(), 3);
            for (int j = 0; j < fast[r].Length; j++)
                Assert.True(Math.Abs(fast[r][j] - reference[r][j]) < 1e-4, $"row {r} label {j}: {fast[r][j]} against {reference[r][j]}");
        }
    }
}
