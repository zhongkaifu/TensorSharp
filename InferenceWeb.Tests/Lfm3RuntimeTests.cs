// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// LiquidAI LFM3 on the shared decoder runtime (Lfm3RuntimeModel): the recipe - short convolutions, full and sliding-window
// attention with the head-wise gate, the zero-tail rotary as a permuted partial one, sandwich norms - matches an independent
// reference forward written from llama.cpp's graph, and every runtime mechanism holds for it: tensor parallelism, whole-layer
// stages, their hybrid, batched slots, continuous batching, checkpoints, speculation, decode steps that advance in place.
// Devices are simulated (SimulatedDeviceGroup): functional evidence, not GPU evidence.
using Microsoft.Extensions.Logging.Abstractions;

using TensorSharp;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

public sealed partial class Lfm3RuntimeTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-lfm3-runtime-" + Guid.NewGuid().ToString("N"));
    private readonly KvCacheDtype _dtype = KvCacheDtypeConfig.Current;
    private readonly bool _dtypeExplicit = KvCacheDtypeConfig.IsExplicitlySet;
    private readonly Xunit.Abstractions.ITestOutputHelper _output;

    public Lfm3RuntimeTests(Xunit.Abstractions.ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        KvCacheDtypeConfig.RestoreForTests(_dtype, _dtypeExplicit);
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private static Lfm3SyntheticModelBuilder Builder(string variant) => variant switch
    {
        // Untied: the small files' greedy streams then depend on the history (a tied head repeats its last token).
        "dense" => new Lfm3SyntheticModelBuilder { UntiedOutput = true },
        // The real checkpoints' head width: direct CUDA's flash kernels run 128-wide heads only.
        "dense128" => new Lfm3SyntheticModelBuilder { UntiedOutput = true, HeadDimension = 128 },
        // The real checkpoints' layout: the head is the token embedding.
        "tied" => new Lfm3SyntheticModelBuilder(),
        // F32 everywhere: the reference forward's exact weights.
        "f32" => new Lfm3SyntheticModelBuilder { WeightType = SyntheticGguf.GgmlType.F32 },
        // Quantized projections: the full layers' q/k rows permute as Q8_0 blocks (fused into attn_qkv).
        "q8" => new Lfm3SyntheticModelBuilder { WeightType = SyntheticGguf.GgmlType.Q8_0, UntiedOutput = true },
        // LiquidAI's NVFP4 layout: NVFP4 feed-forwards with .scale sidecars, Q8_0 elsewhere.
        "nvfp4" => new Lfm3SyntheticModelBuilder { WeightType = SyntheticGguf.GgmlType.Q8_0, Nvfp4Ffn = true, UntiedOutput = true },
        _ => throw new ArgumentOutOfRangeException(nameof(variant)),
    };

    private string Fixture(string variant) => Builder(variant).Write(Path.Combine(_dir, $"lfm3-{variant}.gguf"));

    private static Lfm3RuntimeModel Create(string path, int tp = 1, int stages = 1, BackendType backend = BackendType.Cpu)
        => new(path, backend, tp, tp > 1 ? new SimulatedDeviceGroup(tp * stages) : null, stages);

    /// <summary>Prefill past the sliding window, decode, reset, prefill, decode: a recurrent stack cannot rewind.</summary>
    private static List<float[]> Script(ModelBase model)
    {
        var rows = new List<float[]>();
        int[] prompt = Enumerable.Range(0, 21).Select(i => 40 + (i * 11) % 150).ToArray();
        rows.Add((float[])model.ForwardRefill(prompt).Clone());
        foreach (int token in new[] { 71, 72, 73, 74, 75, 76 })
            rows.Add((float[])model.Forward(new[] { token }).Clone());
        model.ResetKVCache();
        rows.Add((float[])model.ForwardRefill(new[] { 33, 34, 35, 36, 37, 38, 39 }).Clone());
        rows.Add((float[])model.Forward(new[] { 40 }).Clone());
        return rows;
    }

    /// <summary>Backends whose matmuls round activations (ggml_cpu to BF16 / Q8; ggml_cuda and ggml_vulkan likewise; direct
    /// CUDA's BF16 GEMMs past a few rows and its per-op BF16 products take BF16 activations, its quantized products q8_1 ones:
    /// A40, 2026-11-17, a 43-row prompt read whole sits 1.04e-2 from the same prompt read a token at a time on BF16 weights,
    /// 4e-7 on F32 weights).</summary>
    private static bool RoundsActivations(BackendType backend)
        => backend is BackendType.GgmlCpu or BackendType.GgmlCuda or BackendType.GgmlVulkan or BackendType.Cuda;

    private static double RelL2(float[] a, float[] b)
    {
        Assert.Equal(a.Length, b.Length);
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = a[i] - b[i];
            num += d * d;
            den += (double)b[i] * b[i];
        }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }

    private static int ArgMax(float[] logits)
    {
        int best = 0;
        for (int i = 1; i < logits.Length; i++) if (logits[i] > logits[best]) best = i;
        return best;
    }

    private void AssertRows(List<float[]> actual, List<float[]> expected, double tolerance, string label)
    {
        Assert.Equal(expected.Count, actual.Count);
        double[] rel = actual.Select((row, i) => RelL2(row, expected[i])).ToArray();
        _output.WriteLine($"{label}: relL2 per row {string.Join(" ", rel.Select(r => r.ToString("E1")))}");
        for (int i = 0; i < expected.Count; i++)
            Assert.True(rel[i] <= tolerance, $"{label} row {i}: relL2 {rel[i]:E2} > {tolerance:E1}");
    }

    // ---------------------------------------------------------------- the recipe

    [Fact]
    public void TheFactoryCreatesTheRuntime()
    {
        using ModelBase model = ModelBase.Create(Fixture("dense"), BackendType.Cpu);
        var lfm3 = Assert.IsType<Lfm3RuntimeModel>(model);
        Assert.Equal(Lfm3SyntheticModelBuilder.KvHeads, model.Config.NumKVHeads);
        Assert.Contains("|swa=8:", lfm3.KVStateFingerprint, StringComparison.Ordinal);
        Assert.Contains("|kperm=16", lfm3.KVStateFingerprint, StringComparison.Ordinal);
    }

    /// <summary>
    /// The runtime against a reference forward written from llama.cpp's LFM3 graph in double precision (F32 weights):
    /// unpermuted heads rotated with the rope_freqs factors (llama.cpp's form), the sliding window as llama.cpp masks it, the
    /// short convolutions, the gate, the sandwich norms, the tied head. Equal logits prove the permuted partial rotary, the
    /// window, the conv state and every hook on the managed executor's graphs, the per-op path, and the device backends.
    /// </summary>
    [Theory]
    [InlineData("graphs")]
    [InlineData("per-op")]
    public void MatchesTheReferenceForward(string path)
        => AssertMatchesReference(BackendType.Cpu, path == "graphs", 2e-5);

    /// <summary>F32 weights: ggml_cpu rounds no activation and sits 1e-6 from double precision; ggml_metal's products and flash
    /// attention take F16 operands and sit ~2e-3 away, its per-op path as its graphs (M5 Pro, 2026-11-17).</summary>
    [Theory]
    [InlineData("graphs")]
    [InlineData("per-op")]
    public void PinnedGgmlBackendMatchesTheReferenceForward(string path)
        => AssertMatchesReference(TestGates.PinnedGgmlBackend, path == "graphs", TestGates.PinnedGgmlBackend == BackendType.GgmlCpu ? 1e-4 : 5e-3);

    /// <summary>MLX computes ~2e-3 from double precision on these F32 weights for every layer kind alike (an attention-only or a
    /// sliding-only stack too, with or without the zero tail: M5 Pro, 2026-11-17); Qwen 3.5's MLX rows sit 6e-4 .. 1.2e-3 from
    /// the managed executor.</summary>
    [MlxFact]
    public void MlxMatchesTheReferenceForward() => AssertMatchesReference(BackendType.Mlx, graphs: true, 5e-3);

    private void AssertMatchesReference(BackendType backend, bool graphs, double tolerance, KvCacheDtype kv = KvCacheDtype.F32)
    {
        KvCacheDtypeConfig.RestoreForTests(kv, true);
        Lfm3SyntheticModelBuilder builder = Builder("f32");
        string path = builder.Write(Path.Combine(_dir, "lfm3-f32.gguf"));
        var reference = new Lfm3Reference(builder);
        int[] prompt = Enumerable.Range(0, 21).Select(i => 40 + (i * 11) % 150).ToArray();
        int[] decode = { 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82 };
        var expected = new List<float[]> { reference.Prompt(prompt) };
        foreach (int token in decode) expected.Add(reference.Step(token));

        using var model = Create(path, backend: backend);
        model.GraphLoweringEnabled = graphs;
        var actual = new List<float[]> { (float[])model.ForwardRefill(prompt).Clone() };
        foreach (int token in decode) actual.Add((float[])model.Forward(new[] { token }).Clone());
        Assert.True(graphs ? model.GraphRuns > 0 : model.GraphRuns == 0, $"graphs={graphs} but {model.GraphRuns} graph runs");
        AssertRingsWrapped(model, prompt.Length + decode.Length);
        AssertRows(actual, expected, tolerance, $"{backend} graphs={graphs}");
    }

    /// <summary>
    /// The full layers' rotary: llama.cpp turns NeoX pairs (i, i + 32) of a 128-dim head at 1e6^(-2i/128) / factor[i], the
    /// factors 1 for 32 pairs and 1e30 for the rest. TensorSharp permutes each head to [0..31, 64..95, 32..63, 96..127] and turns
    /// NeoX pairs (i, i + 32) of the first 64 dims at 1000^(-2i/64): the same angles on the same pairs, so the scores (and the
    /// rotated vectors, permuted back) agree.
    /// </summary>
    [Fact]
    public void PartialRotaryOnPermutedHeadsEqualsTheZeroTailFactors()
    {
        const int d = 128, half = 64, pairs = 32;
        var rng = new Random(7);
        double[] q = Enumerable.Range(0, d).Select(_ => rng.NextDouble() * 2 - 1).ToArray();
        double[] k = Enumerable.Range(0, d).Select(_ => rng.NextDouble() * 2 - 1).ToArray();
        int[] perm = Enumerable.Range(0, pairs).Concat(Enumerable.Range(half, pairs)).Concat(Enumerable.Range(pairs, half - pairs))
            .Concat(Enumerable.Range(half + pairs, half - pairs)).ToArray();
        float partialBase = Lfm3RuntimeModel.PartialRotaryBase(1e6f, pairs, d);
        Assert.Equal(1000f, partialBase);
        foreach (int p in new[] { 0, 1, 7, 255, 4096, 262143 })
        {
            double[] qRef = RopeWithFactors(q, p, d, 1e6, pairs), kRef = RopeWithFactors(k, p - 3, d, 1e6, pairs);
            double[] qPerm = perm.Select(j => q[j]).ToArray(), kPerm = perm.Select(j => k[j]).ToArray();
            double[] qTs = RopePartial(qPerm, p, 2 * pairs, partialBase), kTs = RopePartial(kPerm, p - 3, 2 * pairs, partialBase);
            for (int j = 0; j < d; j++)
            {
                Assert.True(Math.Abs(qTs[j] - qRef[perm[j]]) < 2e-6 * (1 + Math.Abs(p) / 1000.0), $"p={p} dim {j}: {qTs[j]} vs {qRef[perm[j]]}");
                Assert.True(Math.Abs(kTs[j] - kRef[perm[j]]) < 2e-6 * (1 + Math.Abs(p) / 1000.0), $"p={p} key dim {j}");
            }
            double dotRef = qRef.Zip(kRef, (a, b) => a * b).Sum(), dotTs = qTs.Zip(kTs, (a, b) => a * b).Sum();
            Assert.True(Math.Abs(dotRef - dotTs) < 1e-5, $"p={p}: q.k {dotTs} vs {dotRef}");
        }

        // The model's own head layout for the fixture's 64-dim heads (16 of 32 pairs turn).
        using var model = Create(Fixture("dense"));
        Assert.Equal(Enumerable.Range(0, 16).Concat(Enumerable.Range(32, 16)).Concat(Enumerable.Range(16, 16)).Concat(Enumerable.Range(48, 16)),
            model.HeadPermutationForTests);
    }

    /// <summary>ggml's NeoX rope (ggml-cpu ops.cpp) with frequency factors: pair (i, i + n/2) turns by p base^(-2i/n) / ff[i].</summary>
    private static double[] RopeWithFactors(double[] x, int p, int n, double freqBase, int ones)
    {
        var y = (double[])x.Clone();
        for (int i = 0; i < n / 2; i++)
        {
            double theta = p * Math.Pow(freqBase, -2.0 * i / n) / (i < ones ? 1.0 : 1.0000000150474662e+30);
            double c = Math.Cos(theta), s = Math.Sin(theta);
            y[i] = x[i] * c - x[i + n / 2] * s;
            y[i + n / 2] = x[i] * s + x[i + n / 2] * c;
        }
        return y;
    }

    /// <summary>NeoX rope over the first <paramref name="dims"/> dims (pairs (i, i + dims/2)); the rest pass through.</summary>
    private static double[] RopePartial(double[] x, int p, int dims, double freqBase)
    {
        var y = (double[])x.Clone();
        for (int i = 0; i < dims / 2; i++)
        {
            double theta = p * Math.Pow(freqBase, -2.0 * i / dims);
            double c = Math.Cos(theta), s = Math.Sin(theta);
            y[i] = x[i] * c - x[i + dims / 2] * s;
            y[i + dims / 2] = x[i] * s + x[i + dims / 2] * c;
        }
        return y;
    }

    // ---------------------------------------------------------------- backends

    [Theory]
    [InlineData("dense")]
    [InlineData("q8")]
    public void PinnedGgmlBackendMatchesTheManagedReference(string variant) => CompareBackend(variant, TestGates.PinnedGgmlBackend);

    [MlxTheory]
    [InlineData("dense")]
    [InlineData("q8")]
    public void MlxMatchesTheManagedReference(string variant) => CompareBackend(variant, BackendType.Mlx);

    private void CompareBackend(string variant, BackendType backend)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(variant);
        List<float[]> reference;
        using (var managed = Create(path))
            reference = Script(managed);
        using var model = Create(path, backend: backend);
        AssertRows(Script(model), reference, Tolerance(variant, backend), $"{variant} on {backend}");
        Assert.True(model.GraphRuns > 0, $"{variant} on {backend}: no step ran as a graph");
    }

    /// <summary>The managed executor rounds the activations of a quantized product as ggml-cpu does (q8 on ggml_cpu sits 9e-8
    /// from it where no window slides), and that rounding flips with the last bits of its inputs: against a backend that does
    /// not round them the same way (ggml_metal, MLX; ggml_cuda, ggml_vulkan and direct CUDA, whose prompt products round
    /// otherwise) a quantized fixture sits 3-6e-2 away (A40: ggml_cuda 6.0e-2 q8, 6.4e-2 nvfp4; direct CUDA 3.6e-2 q8,
    /// 5.0e-2 nvfp4), as the real Q8_0 checkpoint's CPU and Metal runs of llama.cpp itself do (2-4e-2). Unquantized fixtures
    /// compare tightly.</summary>
    private static double Tolerance(string variant, BackendType backend)
        => variant is "q8" or "nvfp4" ? (backend == BackendType.GgmlCpu ? 5e-2 : 8e-2) : RoundsActivations(backend) ? 2.5e-2 : 5e-3;

    /// <summary>On every graph executor each step (attention, short convolutions on the slot's window) runs as one graph and
    /// computes what the per-op path computes.</summary>
    [Theory]
    [InlineData("dense", "ggml")]
    [InlineData("q8", "ggml")]
    [InlineData("dense", "managed")]
    public void WholeForwardGraphMatchesThePerOpPath(string variant, string executor)
        => AssertGraphMatchesPerOp(variant, executor == "ggml" ? TestGates.PinnedGgmlBackend : BackendType.Cpu);

    [MlxFact]
    public void WholeForwardGraphMatchesThePerOpPathOnMlx() => AssertGraphMatchesPerOp("dense", BackendType.Mlx);

    private void AssertGraphMatchesPerOp(string variant, BackendType backend)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(variant);
        List<float[]> perOp;
        using (var model = Create(path, backend: backend))
        {
            model.GraphLoweringEnabled = false;
            perOp = Script(model);
            Assert.Equal(0, model.GraphRuns);
        }
        using var graph = Create(path, backend: backend);
        List<float[]> rows = Script(graph);
        Assert.True(graph.GraphRuns >= rows.Count, "every step must run as a whole-forward graph");
        AssertRows(rows, perOp, Tolerance(variant, backend), $"{variant} on {backend}: graph vs per-op");
    }

    /// <summary>One request alternating between graph steps and per-op steps (as an adapter step forces) keeps its caches and
    /// short-conv windows coherent across both bindings.</summary>
    [Fact]
    public void GraphAndPerOpStepsShareTheState() => AssertGraphAndPerOpShareTheState(TestGates.PinnedGgmlBackend);

    private void AssertGraphAndPerOpShareTheState(BackendType backend)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense");
        int[] prompt = Enumerable.Range(0, 17).Select(i => 40 + (i * 11) % 150).ToArray();
        int[] tokens = { 71, 72, 73, 74, 75, 76 };
        var perOp = new List<float[]>();
        using (var model = Create(path, backend: backend))
        {
            model.GraphLoweringEnabled = false;
            perOp.Add((float[])model.ForwardRefill(prompt).Clone());
            foreach (int t in tokens) perOp.Add((float[])model.Forward(new[] { t }).Clone());
        }
        using var mixed = Create(path, backend: backend);
        var rows = new List<float[]> { (float[])mixed.ForwardRefill(prompt).Clone() };
        for (int i = 0; i < tokens.Length; i++)
        {
            mixed.GraphLoweringEnabled = i is not (2 or 3);   // two per-op steps in the middle
            rows.Add((float[])mixed.Forward(new[] { tokens[i] }).Clone());
        }
        Assert.True(mixed.GraphRuns > 0);
        AssertRows(rows, perOp, RoundsActivations(backend) ? 2.5e-2 : 5e-3, $"mixed vs per-op on {backend}");
    }

    // ---------------------------------------------------------------- the sliding window

    /// <summary>
    /// A prompt read at once, read one token at a time (decode steps), and read in chunks that each straddle the 8-key window
    /// give the same last logits: the window's band masks for prompt chunks, verify-width chunks and decode rows agree, and
    /// the short convolutions carry their windows across chunks.
    /// </summary>
    [Theory]
    [InlineData("managed")]
    [InlineData("ggml")]
    public void PromptMatchesTokenByTokenAndChunkedAcrossTheWindow(string executor)
        => AssertPromptPaths(executor == "ggml" ? TestGates.PinnedGgmlBackend : BackendType.Cpu);

    [MlxFact]
    public void PromptMatchesTokenByTokenAndChunkedAcrossTheWindowOnMlx() => AssertPromptPaths(BackendType.Mlx);

    private void AssertPromptPaths(BackendType backend, KvCacheDtype kv = KvCacheDtype.F32, string variant = "dense", double tolerance = 0)
    {
        KvCacheDtypeConfig.RestoreForTests(kv, true);
        string path = Fixture(variant);
        int[] prompt = Enumerable.Range(0, 43).Select(i => 40 + (i * 13) % 170).ToArray();
        using var model = Create(path, backend: backend);
        float[] whole = (float[])model.ForwardRefill(prompt).Clone();

        model.ResetKVCache();
        float[] stepped = null;
        foreach (int t in prompt) stepped = (float[])model.Forward(new[] { t }).Clone();

        model.ResetKVCache();
        float[] chunked = null;
        // 5 rows, a verify-width 11, 3, a prompt-width 19, 5: every chunk crosses a window edge.
        int at = 0;
        foreach (int n in new[] { 5, 11, 3, 19, 5 })
        {
            chunked = (float[])model.Forward(prompt.AsSpan(at, n).ToArray()).Clone();
            at += n;
        }
        Assert.Equal(prompt.Length, at);
        AssertRingsWrapped(model, prompt.Length);
        if (tolerance == 0) tolerance = RoundsActivations(backend) ? 2.5e-2 : 2e-3;
        _output.WriteLine($"{backend} kv={kv}: token by token vs whole relL2 {RelL2(stepped, whole):E2}, chunked vs whole {RelL2(chunked, whole):E2}");
        Assert.True(RelL2(stepped, whole) <= tolerance, $"{backend} kv={kv}: token by token vs whole relL2 {RelL2(stepped, whole):E2}");
        Assert.True(RelL2(chunked, whole) <= tolerance, $"{backend} kv={kv}: chunked vs whole relL2 {RelL2(chunked, whole):E2}");
    }

    /// <summary>
    /// A prompt step whose head reads only its sequences' last rows runs the last layer's feed-forward on those rows alone
    /// (LFM3 opts into PrunesLastLayerRows, as llama.cpp's graphs cut the stream to the output rows): the logits of a step over
    /// every row, for lone prompts (a view of the last row; one past the window) and for three prompts in one batched prefill
    /// (their last rows gathered).
    /// </summary>
    [Theory]
    [InlineData("managed")]
    [InlineData("ggml")]
    public void PrunedLastLayerMatchesEveryRow(string executor)
        => AssertPrunedLastLayer(executor == "ggml" ? TestGates.PinnedGgmlBackend : BackendType.Cpu);

    [MlxFact]
    public void PrunedLastLayerMatchesEveryRowOnMlx() => AssertPrunedLastLayer(BackendType.Mlx);

    private void AssertPrunedLastLayer(BackendType backend)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense");
        int[][] prompts =
        {
            Enumerable.Range(0, 23).Select(i => 65 + (i * 7) % 50).ToArray(),
            Enumerable.Range(0, 6).Select(i => 97 + (i * 5) % 26).ToArray(),
            Enumerable.Range(0, 13).Select(i => 48 + (i * 3) % 10).ToArray(),
        };
        List<float[]> Run(bool prune, out int prunedSteps)
        {
            using var model = Create(path, backend: backend);
            model.LastLayerPruningDisabledForTests = !prune;
            var rows = new List<float[]>();
            foreach (int[] prompt in prompts)
            {
                model.ResetKVCache();
                rows.Add((float[])model.ForwardRefill(prompt).Clone());
            }
            var slots = Assert.IsAssignableFrom<IBatchedPagedModel>(model);
            string[] ids = { "a", "b", "c" };
            foreach (string id in ids) Assert.True(slots.BindSequenceCache(id), "a new request gets a fresh slot");
            var logits = new float[ids.Length][];
            Assert.True(slots.TryForwardBatchedFusedPrefill(ids, prompts, new int[ids.Length], logits));
            rows.AddRange(logits.Select(l => (float[])l.Clone()));
            prunedSteps = model.LastLayerPrunedSteps;
            return rows;
        }
        List<float[]> every = Run(prune: false, out int none);
        List<float[]> pruned = Run(prune: true, out int steps);
        Assert.Equal(0, none);
        Assert.True(steps >= prompts.Length + 1, $"{backend}: {steps} steps pruned their last layer");
        // The managed executor's F32 products block by rows (1.5e-6 between a row alone and in a batch).
        double tolerance = backend == BackendType.Cpu ? 1e-5 : RoundsActivations(backend) ? 2.5e-2 : 5e-3;
        AssertRows(pruned, every, tolerance, $"{backend}: last layer over the last rows vs every row");
    }

    /// <summary>A decode step that repeats the last one a position on runs again without a trace, past the window and across
    /// its buckets (the band's first cell moves a bucket at a time): every logit is the traced step's, for one request and
    /// for two in one step.</summary>
    [Theory]
    [InlineData("ggml")]
    [InlineData("managed")]
    public void AdvancedDecodeStepsMatchTracedOnes(string executor) => AssertAdvancedDecodeMatchesTraced(BackendOf(executor), KvCacheDtype.F32);

    /// <summary><see cref="AdvancedDecodeStepsMatchTracedOnes"/> on <paramref name="backend"/>; <paramref name="advanced"/> sees
    /// the model whose steps advanced (a recorder's counters) before it is disposed.</summary>
    private void AssertAdvancedDecodeMatchesTraced(BackendType backend, KvCacheDtype kv, string path = null,
        Action<Lfm3RuntimeModel> advanced = null, int minAdvances = 250)
    {
        KvCacheDtypeConfig.RestoreForTests(kv, true);
        path ??= Fixture("dense");

        List<float[]> Run(bool advance, out int advances)
        {
            using var model = Create(path, backend: backend);
            model.DecodeAdvanceDisabled = !advance;
            var unified = (IUnifiedBatchModel)model;
            var trace = new List<float[]>();
            string[] ids = { "a", "b" };
            int[][] prompts = { Enumerable.Range(0, 20).Select(i => 40 + i % 7).ToArray(), Enumerable.Range(0, 9).Select(i => 50 + i).ToArray() };
            var next = new int[2];
            var lengths = new int[2];
            for (int s = 0; s < 2; s++)
            {
                unified.EnsureSequenceSlot(ids[s], 400);
                var logits = new float[1][];
                Assert.True(unified.TryForwardUnified(new[] { ids[s] }, new[] { prompts[s] }, new[] { 0 }, logits));
                next[s] = ArgMax(logits[0]);
                lengths[s] = prompts[s].Length;
            }
            void Decode(string[] which, int steps)
            {
                for (int t = 0; t < steps; t++)
                {
                    int[] seqs = which.Select(id => Array.IndexOf(ids, id)).ToArray();
                    var logits = new float[seqs.Length][];
                    Assert.True(unified.TryForwardUnified(which, seqs.Select(s => new[] { next[s] }).ToArray(),
                        seqs.Select(s => lengths[s]).ToArray(), logits));
                    for (int i = 0; i < seqs.Length; i++)
                    {
                        trace.Add((float[])logits[i].Clone());
                        next[seqs[i]] = ArgMax(logits[i]);
                        lengths[seqs[i]]++;
                    }
                }
            }
            Decode(new[] { "a" }, 60);
            Decode(new[] { "a", "b" }, 30);
            Decode(new[] { "b" }, 260);
            advances = model.DecodeAdvances;
            AssertRingsWrapped(model, 0);
            if (advance) advanced?.Invoke(model);
            return trace;
        }

        List<float[]> traced = Run(advance: false, out int none);
        List<float[]> stepped = Run(advance: true, out int advances);
        Assert.Equal(0, none);
        Assert.True(advances > minAdvances, $"{advances} decode steps advanced");
        Assert.Equal(traced.Count, stepped.Count);
        for (int i = 0; i < traced.Count; i++)
            Assert.True(traced[i].AsSpan().SequenceEqual(stepped[i]), $"{backend} kv={kv} row {i}: advanced logits differ (relL2 {RelL2(stepped[i], traced[i]):E2})");
    }

    /// <summary>
    /// A verify window that straddles the sliding window's edge (rows 6..10 around 8 keys) gives every row the logits plain
    /// decoding gives it, and a commit that keeps part of it (the short-conv windows taken from the window's history, the
    /// attention rows past it rewound) continues exactly as plain decoding of the kept tokens.
    /// </summary>
    [Theory]
    [InlineData("ggml")]
    [InlineData("managed")]
    [InlineData("per-op")]
    public void VerifyWindowsAcrossTheWindowCommitTheKeptRows(string executor)
        => AssertVerifyWindowCommits(executor, new[] { 41, 52, 63, 74, 85, 96 });

    private void AssertVerifyWindowCommits(string executor, int[] prompt, KvCacheDtype kv = KvCacheDtype.F32, double tolerance = 0)
    {
        KvCacheDtypeConfig.RestoreForTests(kv, true);
        string path = Fixture("dense");
        BackendType backend = BackendOf(executor);
        int[] window = { 101, 112, 123, 134, 145 };
        int[] after = { 77, 78, 79 };
        const int kept = 2;
        int vocab;

        var plain = new List<float[]>();
        var continued = new List<float[]>();
        using (var reference = Create(path, backend: backend))
        {
            vocab = reference.Config.VocabSize;
            reference.GraphLoweringEnabled = !PerOp(executor);
            reference.ForwardRefill(prompt);
            foreach (int t in window) plain.Add((float[])reference.Forward(new[] { t }).Clone());
            reference.ResetKVCache();
            reference.ForwardRefill(prompt.Concat(window.Take(kept)).ToArray());
            foreach (int t in after) continued.Add((float[])reference.Forward(new[] { t }).Clone());
        }

        using var model = Create(path, backend: backend);
        model.GraphLoweringEnabled = !PerOp(executor);
        var spec = (ISpeculativeModel)model;
        model.ForwardRefill(prompt);
        var logits = new float[window.Length * vocab];
        spec.SpecForward(window, null, logits, allLogitsRows: true);
        if (tolerance == 0) tolerance = RoundsActivations(backend) ? 2.5e-2 : 2e-3;
        for (int r = 0; r < window.Length; r++)
        {
            float[] row = logits.AsSpan(r * vocab, vocab).ToArray();
            Assert.True(RelL2(row, plain[r]) <= tolerance, $"{executor} verify row {r}: relL2 {RelL2(row, plain[r]):E2}");
        }
        spec.SpecOnVerifyAccepted(kept - 1, window.Length);
        spec.SpecRewindCache(prompt.Length + kept);
        for (int i = 0; i < after.Length; i++)
        {
            float[] row = model.Forward(new[] { after[i] });
            Assert.True(RelL2(row, continued[i]) <= tolerance, $"{executor} after the commit, row {i}: relL2 {RelL2(row, continued[i]):E2}");
        }
        AssertRingsWrapped(model, prompt.Length + window.Length);
    }

    // ---------------------------------------------------------------- the runtime's mechanisms

    [Fact]
    public void ARecurrentStackRewindsOnlyToItsLengthOrToZero()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        using var model = Create(Fixture("dense"));
        float[] first = (float[])model.ForwardRefill(Enumerable.Range(0, 12).Select(i => 50 + i).ToArray()).Clone();
        Assert.False(model.CanTruncateKVCache(12, 10));
        Assert.False(model.TryTruncateKVCache(10));
        Assert.False(model.SupportsKVStateSnapshot);
        var caps = model.GetPrefixCacheCapabilities();
        Assert.Equal(TensorSharp.Runtime.Scheduling.PrefixCache.FamilyClass.R, caps.Class);
        Assert.True(model.TryTruncateKVCache(0));
        float[] again = model.ForwardRefill(Enumerable.Range(0, 12).Select(i => 50 + i).ToArray());
        Assert.Equal(first, again);
    }

    public static TheoryData<int, int> Placements() => new() { { 2, 1 }, { 1, 2 }, { 1, 4 }, { 2, 2 } };

    /// <summary>Tensor parallelism (channel-sharded short convolutions, head-sharded gate), whole-layer stages and their hybrid
    /// give the single device's logits.</summary>
    [Theory]
    [MemberData(nameof(Placements))]
    public void PlacementMatchesOneDevice(int tp, int stages)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense");
        List<float[]> reference;
        using (var single = Create(path))
            reference = Script(single);
        var group = tp > 1 ? new SimulatedDeviceGroup(tp * stages) : null;
        using var placed = new Lfm3RuntimeModel(path, BackendType.Cpu, tp, group, stages);
        Assert.Equal(tp, placed.Mesh.TensorParallel);
        Assert.Equal(stages, placed.Mesh.Stages);
        List<float[]> actual = Script(placed);
        if (tp > 1 && stages == 1) Assert.True(group.AllReduceCount > 0 && placed.GraphRuns > 0, $"tp={tp}: no per-rank graph ran");
        AssertRows(actual, reference, tp == 1 ? 0 : 1e-4, $"tp={tp} stages={stages}");
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 1)]
    [InlineData(1, 2)]
    [InlineData(2, 2)]
    public void SlotBatchesMatchEachSequenceAlone(int tp, int stages) => CompareSlotBatches(tp, stages, BackendType.Cpu);

    [Fact]
    public void SlotBatchesMatchEachSequenceAloneOnThePinnedGgmlBackend() => CompareSlotBatches(1, 1, TestGates.PinnedGgmlBackend);

    [MlxFact]
    public void SlotBatchesMatchEachSequenceAloneOnMlx() => CompareSlotBatches(1, 1, BackendType.Mlx);

    private void CompareSlotBatches(int tp, int stages, BackendType backend, KvCacheDtype kv = KvCacheDtype.F32, double tolerance = 0)
    {
        KvCacheDtypeConfig.RestoreForTests(kv, true);
        string path = Fixture("dense");
        // The managed executor's batched rows sit 1.5e-6 from the lone ones (its F32 products' blocking follows the rows).
        if (tolerance == 0) tolerance = backend == BackendType.Cpu ? 1e-5 : RoundsActivations(backend) ? 2.5e-2 : 5e-3;
        SlotParity.AssertBatchedSlotsMatchAlone(() => Create(path, tp, stages, backend), tolerance, $"lfm3 tp={tp} stages={stages} {backend} kv={kv}");
    }

    public static TheoryData<int, int> BatchedPlacements() => new() { { 1, 1 }, { 2, 1 }, { 1, 2 }, { 2, 2 } };

    [Theory]
    [MemberData(nameof(BatchedPlacements))]
    public Task ConcurrentRequestsAreBatched(int tp, int stages) => AssertConcurrentRequestsAreBatched(tp, stages, BackendType.Cpu, KvCacheDtype.F32);

    /// <summary>The engine's greedy streams of three requests served together on <paramref name="backend"/> equal each request's
    /// alone on the same backend (<paramref name="nearTie"/>: a stream may part only where plain greedy's top two logits are
    /// that close, for backends whose batched rows round apart from lone ones).</summary>
    private async Task AssertConcurrentRequestsAreBatched(int tp, int stages, BackendType backend, KvCacheDtype kv, double nearTie = 0)
    {
        KvCacheDtypeConfig.RestoreForTests(kv, true);
        string path = Fixture("dense");
        const int maxNewTokens = 8;
        var prompts = new List<int[]>
        {
            Enumerable.Range(0, 19).Select(i => 65 + (i * 7) % 50).ToArray(),
            Enumerable.Range(0, 27).Select(i => 97 + (i * 5) % 26).ToArray(),
            Enumerable.Range(0, 13).Select(i => 48 + (i * 3) % 10).ToArray(),
        };

        var expected = new List<int[]>();
        var margins = new List<double[]>();
        using (var reference = Create(path, backend: backend))
        {
            foreach (int[] prompt in prompts)
            {
                reference.ResetKVCache();
                float[] logits = reference.ForwardRefill(prompt);
                var produced = new List<int>();
                var margin = new List<double>();
                for (int i = 0; i < maxNewTokens; i++)
                {
                    int token = ArgMax(logits);
                    produced.Add(token);
                    margin.Add(TopTwoMargin(logits));
                    if (reference.Tokenizer.EosTokenIds.Contains(token)) break;
                    if (i + 1 < maxNewTokens) logits = reference.Forward(new[] { token });
                }
                expected.Add(produced.ToArray());
                margins.Add(margin.ToArray());
            }
        }

        using var model = Create(path, tp, stages, backend);
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
        {
            int n = Math.Min(outputs[i].Length, expected[i].Length);
            Assert.True(n > 0, $"request {i} produced nothing");
            // Equal up to the first step whose plain greedy is a near-tie (none with nearTie 0).
            int at = Enumerable.Range(0, n).FirstOrDefault(t => outputs[i][t] != expected[i][t], -1);
            if (at >= 0)
                Assert.True(margins[i][at] <= nearTie,
                    $"{backend} request {i} parts from its lone greedy stream at token {at} ({outputs[i][at]} vs {expected[i][at]}, top-two margin {margins[i][at]:E2})");
        }
    }

    /// <summary>The relative gap between the two largest logits.</summary>
    private static double TopTwoMargin(float[] logits)
    {
        float first = float.NegativeInfinity, second = float.NegativeInfinity, scale = 0;
        foreach (float v in logits)
        {
            scale = Math.Max(scale, Math.Abs(v));
            if (v > first) { second = first; first = v; }
            else if (v > second) second = v;
        }
        return (first - second) / Math.Max(scale, 1e-30f);
    }

    /// <summary>A shared-prefix checkpoint (attention rows and short-conv windows), cloned or written to a file and read back,
    /// continues exactly as a fresh run; on a GGML backend also through the device copy program.</summary>
    [Theory]
    [InlineData("ggml")]
    [InlineData("managed")]
    public void CheckpointsContinueExactly(string executor)
        => AssertCheckpointsContinue(BackendOf(executor), KvCacheDtype.F32, deviceCopies: executor == "ggml");

    private void AssertCheckpointsContinue(BackendType backend, KvCacheDtype kv, bool deviceCopies)
    {
        KvCacheDtypeConfig.RestoreForTests(kv, true);
        string path = Fixture("dense");
        CheckpointChecks.AssertCheckpointsContinueExactly(() => Create(path, backend: backend), 1e-5, $"lfm3 {backend} kv={kv}", deviceCopies);
    }

    /// <summary>Greedy speculative decoding (n-gram drafts, partly accepted) emits exactly the plain greedy tokens: a verify
    /// window keeps the short-conv inputs and attends under the sliding window, and the accept count selects the prefix.</summary>
    [Theory]
    [InlineData("ggml")]
    [InlineData("managed")]
    public void SpeculativeDecodingMatchesPlainGreedy(string executor)
        => AssertSpeculationMatchesPlainGreedy(BackendOf(executor), KvCacheDtype.F32, RoundsActivations(BackendOf(executor)) ? 1e-2 : 0);

    private void AssertSpeculationMatchesPlainGreedy(BackendType backend, KvCacheDtype kv, double tieTolerance)
    {
        KvCacheDtypeConfig.RestoreForTests(kv, true);
        string path = Fixture("dense");
        SpeculativeRuntimeChecks.AssertMatchesPlainGreedy(() => Create(path, backend: backend), $"lfm3 {backend} kv={kv}", tieTolerance: tieTolerance);
    }

    /// <summary>Two ignore_eos requests speculating together (n-gram) run through their EOS draws exactly as plain decoding
    /// does, each to its length limit (the batched verify used to end a lane at an EOS anchor and then fail every request of
    /// its next step). Scheduler logic, every family alike: the managed executor is exact.</summary>
    [Fact]
    public async Task IgnoreEosRequestsSpeculateThroughEos()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        var options = new SpeculationOptions { Enabled = true, SpeculatorName = SpeculatorRegistry.NGram, MaxDraftTokens = 3, MinDraftProb = 0 };
        var paths = new Dictionary<int, string>();
        await SpeculativeRuntimeChecks.AssertIgnoreEosSpeculationMatchesPlain(eos =>
        {
            if (!paths.TryGetValue(eos, out string path))
                paths[eos] = path = new Lfm3SyntheticModelBuilder { UntiedOutput = true, Eos = (uint)eos }
                    .Write(Path.Combine(_dir, $"lfm3-eos{eos}.gguf"));
            return Create(path);
        }, 2, options, "lfm3 ignore_eos");
    }

    /// <summary>Speculation on per-rank graphs (tensor parallelism 2): each rank's short-conv states keep their channels'
    /// verify history and commit on their own device.</summary>
    [Fact]
    public void SpeculationOnRankGraphsMatchesPlainGreedy()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense");
        SpeculativeRuntimeChecks.AssertMatchesPlainGreedy(() => Create(path, tp: 2), "lfm3 tp2");
    }

    [MlxFact]
    public void SpeculativeDecodingMatchesPlainGreedyOnMlx()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense");
        SpeculativeRuntimeChecks.AssertMatchesPlainGreedy(() => Create(path, backend: BackendType.Mlx), "lfm3 mlx", tieTolerance: 1e-2);
    }

    // ---------------------------------------------------------------- NVFP4

    /// <summary>NVFP4 blocks widened to F16 hold exactly ggml's dequantized values (every NVFP4 value is an F16).</summary>
    [Fact]
    public unsafe void Nvfp4WidensToF16Exactly()
    {
        var rng = new Random(11);
        const int values = 64 * 300;
        var raw = new byte[values / 64 * 36];
        rng.NextBytes(raw);
        // Every UE4M3 byte, NaN (0x7F) and the ignored sign bit included.
        for (int b = 0; b < raw.Length / 36; b++)
            for (int s = 0; s < 4; s++) raw[b * 36 + s] = (byte)((b * 4 + s) % 256);
        var expected = new float[values];
        fixed (byte* p = raw)
        {
            NativeDequant.DequantizeToFloat32(40, (IntPtr)p, expected, 0, values);
            IntPtr f16 = ModelBase.WidenNvfp4ToF16((IntPtr)p, values);
            try
            {
                var halves = new ReadOnlySpan<System.Half>((void*)f16, values);
                for (int i = 0; i < values; i++)
                    Assert.True(BitConverter.SingleToInt32Bits((float)halves[i]) == BitConverter.SingleToInt32Bits(expected[i]) || (expected[i] == 0 && (float)halves[i] == 0),
                        $"value {i}: F16 {(float)halves[i]} vs NVFP4 {expected[i]}");
            }
            finally
            {
                QuantizedWeight.FreeBuffer(f16);
            }
        }
    }

    /// <summary>LiquidAI's NVFP4 layout on every Mac lane: the managed CPU and ggml_cpu run NVFP4 directly, ggml_metal and
    /// MLX (no NVFP4 matmul) on the exact F16 widening, all with the .scale sidecars applied as output multipliers.</summary>
    [Fact]
    public void Nvfp4FeedForwardsOnThePinnedGgmlBackendMatchTheManagedReference() => AssertNvfp4(TestGates.PinnedGgmlBackend);

    [MlxFact]
    public void Nvfp4FeedForwardsOnMlxMatchTheManagedReference() => AssertNvfp4(BackendType.Mlx);

    /// <summary>--nvfp4-prompt-activations q8 on the pinned GGML backend: the prompt rows' NVFP4 products keep 8-bit activations
    /// (ggml-cuda's q8 route where it would run W4A4, on CC 12.x; every other route has them already), so the class bound of
    /// 8-bit activations holds on every GPU, the W4A4 class bound's place included.</summary>
    [Fact]
    public void Nvfp4FeedForwardsWithEightBitPromptActivationsOnThePinnedGgmlBackendMatchTheManagedReference()
    {
        using (Nvfp4PromptActivationsConfig.ForCurrentFlowInTests(Nvfp4PromptActivations.Q8))
            AssertNvfp4(TestGates.PinnedGgmlBackend, Tolerance("nvfp4", TestGates.PinnedGgmlBackend));
    }

    private void AssertNvfp4(BackendType backend, double tolerance = 0, Action<ModelBase> configure = null)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("nvfp4");
        List<float[]> reference;
        using (var managed = Create(path))
        {
            Assert.Equal(40, managed.QuantizedWeightTypeForTests("blk.0.ffn_down.weight"));
            reference = Script(managed);
        }
        if (tolerance <= 0 && backend == BackendType.GgmlCuda && GgmlCudaRunsW4A4())
            tolerance = W4A4Tolerance(path, reference);
        using var model = Create(path, backend: backend);
        configure?.Invoke(model);
        int expectedType = ModelBase.WidensNvfp4ToF16(backend) ? 1 : 40;
        Assert.Equal(expectedType, model.QuantizedWeightTypeForTests("blk.0.ffn_down.weight"));
        AssertRows(Script(model), reference, tolerance > 0 ? tolerance : Tolerance("nvfp4", backend), $"nvfp4 on {backend}");
    }

    /// <summary>ggml-cuda multiplies NVFP4 weights with NVFP4-quantized activations (W4A4, ggml_prec Q4 for src1: mmq.cu's
    /// ggml_cuda_mmq_get_prec_src1) on GPUs with Blackwell's FP4 tensor cores (CC 12.x), for the products MMQ takes (prompt
    /// rows); decode rows stay on q8_1 matvecs. A test cannot switch that (GGML_CUDA_MMQ_PREC is process environment).
    /// W4A4 prompt rows are the default on purpose, as in llama.cpp: on an RTX PRO 4500 they run prompts 1.5x (LFM3-1.7B)
    /// to 2.3x (Qwen3.8-27B) faster than ggml-cuda's q8_1 route, at a perplexity cost (wikitext-2, 32 x 1024, every row a
    /// prompt row): LFM3-1.7B NVFP4 +6.6% (22.06 against 20.70; llama.cpp's own +5.9%), Qwen3.8-27B NVFP4 +1.9% (6.563
    /// against 6.441). Decode speed is unchanged.</summary>
    private static bool GgmlCudaRunsW4A4()
    {
        try { return TensorSharp.Cuda.CudaDevice.GetDevice(0).ComputeCapabilityMajor == 12; }
        catch (Exception) { return false; }
    }

    /// <summary>
    /// The bound for W4A4 prompt rows, derived on the fixture from ggml-cuda's own arithmetic: the managed executor with
    /// ggml-cuda's NVFP4 activation quantizer applied to the rows of its NVFP4 products of more than 16 rows
    /// (<see cref="Nvfp4ActivationQuantizer"/> through ManagedQuantizedOps.Nvfp4PromptRowsForTests) is one W4A4 realization
    /// of the script; its largest row distance from the exact reference is what W4A4 rounding costs here (0.380 on this
    /// fixture; ggml-cuda's own rows on an RTX PRO 4500 sit 0.214-0.378 from it). Another realization of the same quantizer (ggml-cuda's, whose
    /// inputs differ in their last bits) is as far: this small network turns the flipped FP4 codes of last-bit differences
    /// into logit changes as large as the rounding itself (ggml-cuda against the emulation: 0.15-0.30 on the same rows). The
    /// bound is 1.5 times the emulation's own distance. A dropped weight scale (2.5e-3 here) or a wrong layout lands far
    /// past it; the q8-activation lanes keep the 8e-2 bound.
    /// </summary>
    private double W4A4Tolerance(string path, List<float[]> reference)
    {
        List<float[]> emulated;
        ManagedQuantizedOps.Nvfp4PromptRowsForTests = Nvfp4ActivationQuantizer.RoundRows;
        try
        {
            using var managed = Create(path);
            emulated = Script(managed);
        }
        finally
        {
            ManagedQuantizedOps.Nvfp4PromptRowsForTests = null;
        }
        double own = emulated.Select((row, i) => RelL2(row, reference[i])).Max();
        _output.WriteLine($"nvfp4 W4A4 emulation: largest row relL2 {own:E2} from the exact reference; bound {1.5 * own:E2}");
        Assert.True(own > 1e-2, "the W4A4 emulation must round the prompt products' activations");
        return 1.5 * own;
    }

    /// <summary>The .scale sidecars matter: the same checkpoint with them ignored gives other logits (so the comparison above
    /// cannot pass with the scales dropped).</summary>
    [Fact]
    public void Nvfp4ScalesAreApplied()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        using var model = Create(Fixture("nvfp4"));
        Assert.True(model.QuantizedWeightScaleForTests("blk.0.ffn_down.weight") is > 2.4e-3f and < 2.6e-3f);
    }
}
