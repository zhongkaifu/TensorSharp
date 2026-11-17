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

public sealed class Lfm3RuntimeTests : IDisposable
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

    /// <summary>Backends whose matmuls round activations (ggml_cpu to BF16 / Q8; ggml_cuda and ggml_vulkan likewise).</summary>
    private static bool RoundsActivations(BackendType backend)
        => backend is BackendType.GgmlCpu or BackendType.GgmlCuda or BackendType.GgmlVulkan;

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

    private void AssertMatchesReference(BackendType backend, bool graphs, double tolerance)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
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
    /// not round them (ggml_metal, MLX) a quantized fixture sits 3-6e-2 away, as the real Q8_0 checkpoint's CPU and Metal runs
    /// of llama.cpp itself do (2-4e-2). Unquantized fixtures compare tightly.</summary>
    private static double Tolerance(string variant, BackendType backend)
        => variant is "q8" or "nvfp4" ? (RoundsActivations(backend) ? 5e-2 : 8e-2) : RoundsActivations(backend) ? 2.5e-2 : 5e-3;

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
    public void GraphAndPerOpStepsShareTheState()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense");
        BackendType backend = TestGates.PinnedGgmlBackend;
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
        AssertRows(rows, perOp, RoundsActivations(backend) ? 2.5e-2 : 5e-3, "mixed vs per-op");
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

    private void AssertPromptPaths(BackendType backend)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense");
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
        double tolerance = RoundsActivations(backend) ? 2.5e-2 : 2e-3;
        Assert.True(RelL2(stepped, whole) <= tolerance, $"{backend}: token by token vs whole relL2 {RelL2(stepped, whole):E2}");
        Assert.True(RelL2(chunked, whole) <= tolerance, $"{backend}: chunked vs whole relL2 {RelL2(chunked, whole):E2}");
    }

    /// <summary>A decode step that repeats the last one a position on runs again without a trace, past the window and across
    /// its buckets (the band's first cell moves a bucket at a time): every logit is the traced step's, for one request and
    /// for two in one step.</summary>
    [Theory]
    [InlineData("ggml")]
    [InlineData("managed")]
    public void AdvancedDecodeStepsMatchTracedOnes(string executor)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense");
        BackendType backend = executor == "ggml" ? TestGates.PinnedGgmlBackend : BackendType.Cpu;

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
            return trace;
        }

        List<float[]> traced = Run(advance: false, out int none);
        List<float[]> advanced = Run(advance: true, out int advances);
        Assert.Equal(0, none);
        Assert.True(advances > 250, $"{advances} decode steps advanced");
        Assert.Equal(traced.Count, advanced.Count);
        for (int i = 0; i < traced.Count; i++)
            Assert.True(traced[i].AsSpan().SequenceEqual(advanced[i]), $"row {i}: advanced logits differ (relL2 {RelL2(advanced[i], traced[i]):E2})");
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
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense");
        BackendType backend = executor == "ggml" ? TestGates.PinnedGgmlBackend : BackendType.Cpu;
        int[] prompt = { 41, 52, 63, 74, 85, 96 };
        int[] window = { 101, 112, 123, 134, 145 };
        int[] after = { 77, 78, 79 };
        const int kept = 2;
        int vocab;

        var plain = new List<float[]>();
        var continued = new List<float[]>();
        using (var reference = Create(path, backend: backend))
        {
            vocab = reference.Config.VocabSize;
            reference.GraphLoweringEnabled = executor != "per-op";
            reference.ForwardRefill(prompt);
            foreach (int t in window) plain.Add((float[])reference.Forward(new[] { t }).Clone());
            reference.ResetKVCache();
            reference.ForwardRefill(prompt.Concat(window.Take(kept)).ToArray());
            foreach (int t in after) continued.Add((float[])reference.Forward(new[] { t }).Clone());
        }

        using var model = Create(path, backend: backend);
        model.GraphLoweringEnabled = executor != "per-op";
        var spec = (ISpeculativeModel)model;
        model.ForwardRefill(prompt);
        var logits = new float[window.Length * vocab];
        spec.SpecForward(window, null, logits, allLogitsRows: true);
        double tolerance = RoundsActivations(backend) ? 2.5e-2 : 2e-3;
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

    private void CompareSlotBatches(int tp, int stages, BackendType backend)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense");
        // The managed executor's batched rows sit 1.5e-6 from the lone ones (its F32 products' blocking follows the rows).
        double tolerance = backend == BackendType.Cpu ? 1e-5 : RoundsActivations(backend) ? 2.5e-2 : 5e-3;
        SlotParity.AssertBatchedSlotsMatchAlone(() => Create(path, tp, stages, backend), tolerance, $"lfm3 tp={tp} stages={stages} {backend}");
    }

    public static TheoryData<int, int> BatchedPlacements() => new() { { 1, 1 }, { 2, 1 }, { 1, 2 }, { 2, 2 } };

    [Theory]
    [MemberData(nameof(BatchedPlacements))]
    public async Task ConcurrentRequestsAreBatched(int tp, int stages)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense");
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
                    int token = ArgMax(logits);
                    produced.Add(token);
                    if (reference.Tokenizer.EosTokenIds.Contains(token)) break;
                    if (i + 1 < maxNewTokens) logits = reference.Forward(new[] { token });
                }
                expected.Add(produced.ToArray());
            }
        }

        using var model = Create(path, tp, stages);
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
            Assert.Equal(expected[i].Take(n), outputs[i].Take(n));
        }
    }

    /// <summary>A shared-prefix checkpoint (attention rows and short-conv windows), cloned or written to a file and read back,
    /// continues exactly as a fresh run; on a GGML backend also through the device copy program.</summary>
    [Theory]
    [InlineData("ggml")]
    [InlineData("managed")]
    public void CheckpointsContinueExactly(string executor)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense");
        bool ggml = executor == "ggml";
        CheckpointChecks.AssertCheckpointsContinueExactly(() => Create(path, backend: ggml ? TestGates.PinnedGgmlBackend : BackendType.Cpu),
            1e-5, $"lfm3 {executor}", deviceCopies: ggml);
    }

    /// <summary>Greedy speculative decoding (n-gram drafts, partly accepted) emits exactly the plain greedy tokens: a verify
    /// window keeps the short-conv inputs and attends under the sliding window, and the accept count selects the prefix.</summary>
    [Theory]
    [InlineData("ggml")]
    [InlineData("managed")]
    public void SpeculativeDecodingMatchesPlainGreedy(string executor)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense");
        BackendType backend = executor == "ggml" ? TestGates.PinnedGgmlBackend : BackendType.Cpu;
        SpeculativeRuntimeChecks.AssertMatchesPlainGreedy(() => Create(path, backend: backend), $"lfm3 {executor}",
            tieTolerance: RoundsActivations(backend) ? 1e-2 : 0);
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

    private void AssertNvfp4(BackendType backend)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("nvfp4");
        List<float[]> reference;
        using (var managed = Create(path))
        {
            Assert.Equal(40, managed.QuantizedWeightTypeForTests("blk.0.ffn_down.weight"));
            reference = Script(managed);
        }
        using var model = Create(path, backend: backend);
        int expectedType = ModelBase.WidensNvfp4ToF16(backend) ? 1 : 40;
        Assert.Equal(expectedType, model.QuantizedWeightTypeForTests("blk.0.ffn_down.weight"));
        AssertRows(Script(model), reference, Tolerance("nvfp4", backend), $"nvfp4 on {backend}");
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
