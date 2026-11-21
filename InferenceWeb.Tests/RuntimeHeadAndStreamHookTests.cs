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
using System.Net;
using System.Net.Sockets;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.Distributed;
using TensorSharp.Graph;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// The shared decoder runtime's family hooks for the head's logit transform (<c>LogitScale</c>, <c>FinalLogitSoftcap</c>),
/// the post norms' epsilon (<c>PostNormEps</c>) and the stream input on the per-op path (<c>ApplyStreamInput</c>, the twin
/// of <c>EmitStreamInput</c>): stage S1 of the Muse-Glimmer migration (docs/design/muse-glimmer-unified.md gaps G1-G3),
/// which Gemma 4 uses too. A test family on the dense synthetic fixture declares them (<see cref="Hooked"/>); each hook is
/// checked against an exact identity on every executor (the managed backend's, the pinned GGML backend's, MLX's), as
/// whole-step programs and on the per-op path, and through the placements whose per-op code differs (tensor parallelism,
/// whole-layer stages, a multi-node pipeline):
/// <list type="bullet">
/// <item>G1: a transformed model's logits are the untransformed model's through the transform computed on the host.</item>
/// <item>G2: scaling a site's block output projection by 1/16 and its post norm's epsilon by 1/256 leaves the logits as
/// they were (RMSNorm(c x, c^2 eps) = RMSNorm(x, eps)); with the epsilon left alone they move.</item>
/// <item>G3: a stream input that scales the embedded rows by 4 is the token embedding stored 4 times larger (and the media
/// rows queued 4 times larger): applied once, after the media rows join the stream, on every path.</item>
/// </list>
/// </summary>
public sealed class RuntimeHeadAndStreamHookTests(ITestOutputHelper output) : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-runtime-hooks-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    // ---------------------------------------------------------------- the test family

    /// <summary>What <see cref="Hooked"/> declares: the logit transform, sandwich post norms with their epsilons (NaN: the
    /// default, Config.Eps), and a stream input that scales the embedded rows.</summary>
    internal sealed record Hooks(float Scale = 1f, float Cap = 0f, bool PostNorms = false, float MixerEps = float.NaN,
        float FfnEps = float.NaN, float StreamScale = 1f);

    /// <summary>Mistral 3 on the shared runtime with <see cref="Hooks"/> declared. The primary constructor's parameters are
    /// stored before the base constructor runs, so the hooks hold during loading too.</summary>
    private sealed class Hooked(string path, BackendType backend, Hooks hooks, int tp = 1, ITensorParallelGroup group = null, int stages = 1)
        : Mistral3Model(path, backend, tp, group, stages)
    {
        protected override float LogitScale => hooks.Scale;
        protected override float FinalLogitSoftcap => hooks.Cap;
        protected override string MixerPostNormWeight(int layer) => hooks.PostNorms ? LayerName(layer, "post_attention_norm.weight") : null;
        protected override string FfnPostNormWeight(int layer) => hooks.PostNorms ? LayerName(layer, "post_ffw_norm.weight") : null;

        protected override float PostNormEps(int layer, BlockSite site)
        {
            float eps = site == BlockSite.Mixer ? hooks.MixerEps : hooks.FfnEps;
            return float.IsNaN(eps) ? base.PostNormEps(layer, site) : eps;
        }

        protected override int EmitStreamInput(GraphStep step, int embedded)
            => hooks.StreamScale == 1f ? embedded : step.Op(GraphOp.Scale, stackalloc[] { embedded }, default, stackalloc[] { hooks.StreamScale });

        protected override void ApplyStreamInput(Tensor embedded)
        {
            if (hooks.StreamScale != 1f) Ops.Mul(embedded, embedded, hooks.StreamScale);
        }
    }

    private const int Layers = 4;

    /// <summary>The dense fixture (F32 weights), with post norms where asked and its tensors rewritten by
    /// <paramref name="transform"/> (a site's block output scaled, the embedding scaled).</summary>
    private string Fixture(string name, bool postNorms = false, Func<string, float[], float[]> transform = null)
    {
        Directory.CreateDirectory(_dir);
        return new DenseDecoderSyntheticModelBuilder
        {
            Architecture = "mistral3", LayerCount = Layers, IncludePostNorms = postNorms, Transform = transform,
        }.Write(Path.Combine(_dir, name + ".gguf"));
    }

    private static Func<string, float[], float[]> Scaled(string suffix, float factor)
        => (name, values) => name.EndsWith(suffix, StringComparison.Ordinal) ? values.Select(v => v * factor).ToArray() : null;

    /// <summary>A fixed script over one cache: a prefill past one prompt step, decode, a truncation and refill, a reset, a
    /// short prefill and decode.</summary>
    private static List<float[]> Script(ModelBase model)
    {
        var rows = new List<float[]>();
        int[] prompt = Enumerable.Range(0, 23).Select(i => 40 + (i * 11) % 150).ToArray();
        rows.Add((float[])model.ForwardRefill(prompt).Clone());
        foreach (int token in new[] { 71, 72, 73, 74 })
            rows.Add((float[])model.Forward(new[] { token }).Clone());
        Assert.True(model.TryTruncateKVCache(12));
        rows.Add((float[])model.ForwardRefill(new[] { 91, 92, 93, 94, 95 }).Clone());
        rows.Add((float[])model.Forward(new[] { 96 }).Clone());
        model.ResetKVCache();
        rows.Add((float[])model.ForwardRefill(new[] { 33, 34, 35, 36, 37, 38, 39 }).Clone());
        rows.Add((float[])model.Forward(new[] { 40 }).Clone());
        return rows;
    }

    /// <summary>The script's logits on a new model, its steps as programs (<paramref name="graphs"/>) or on the per-op path,
    /// held to the path asked for unless <paramref name="anyPath"/> (a placement may decline programs).</summary>
    private static List<float[]> Run(Func<ModelBase> create, bool graphs, Func<ModelBase, List<float[]>> script = null, bool anyPath = false)
    {
        using var _ = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        using ModelBase model = create();
        var decoder = (StandardDecoderModel)model;
        decoder.GraphLoweringEnabled = graphs;
        List<float[]> rows = (script ?? Script)(model);
        if (!graphs) Assert.Equal(0, decoder.GraphRuns);
        else if (!anyPath) Assert.True(decoder.GraphRuns > 0, "the steps did not run as programs");
        return rows;
    }

    private static double RelL2(float[] a, float[] b)
    {
        Assert.Equal(b.Length, a.Length);
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = (double)a[i] - b[i];
            num += d * d;
            den += (double)b[i] * b[i];
        }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }

    private static double Worst(List<float[]> a, List<float[]> b)
    {
        Assert.Equal(b.Count, a.Count);
        return a.Select((row, i) => RelL2(row, b[i])).Max();
    }

    /// <summary>Bit for bit where every product is an F32 product on the host (the managed backend, ggml_cpu); within
    /// rounding where an executor's kernels may take other products or fuse other operations.</summary>
    private static bool Exact(BackendType backend) => backend is BackendType.Cpu or BackendType.GgmlCpu;

    // ---------------------------------------------------------------- G1: the logit transform

    private const float Scale = 2.5f, Cap = 0.6f;

    /// <summary>cap * tanh(scale * x / cap), in double.</summary>
    private static float[] Transformed(float[] raw) => raw.Select(x => (float)(Cap * Math.Tanh(Scale * (double)x / Cap))).ToArray();

    /// <summary>The largest |got - expected| over the cap (a logit's error in units of the transform's range).</summary>
    private static double WorstAbsOverCap(List<float[]> got, List<float[]> expected)
        => got.Select((row, i) => row.Select((v, j) => Math.Abs((double)v - expected[i][j])).Max()).Max() / Cap;

    /// <summary>On every executor and path, the transformed model's logits are the plain model's logits through the
    /// transform (within float rounding of the transform's four operations), and the transform is not the identity on this
    /// fixture (the cap bends the logits: the plain ones reach well past it).</summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void TheLogitTransformIsAppliedOnTheManagedBackend(bool graphs) => VerifyLogitTransform(BackendType.Cpu, graphs);

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void TheLogitTransformIsAppliedOnThePinnedGgmlBackend(bool graphs) => VerifyLogitTransform(TestGates.PinnedGgmlBackend, graphs);

    [MlxTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void TheLogitTransformIsAppliedOnMlx(bool graphs) => VerifyLogitTransform(BackendType.Mlx, graphs);

    private void VerifyLogitTransform(BackendType backend, bool graphs)
    {
        string path = Fixture("dense");
        List<float[]> raw = Run(() => new Hooked(path, backend, new Hooks()), graphs);
        List<float[]> got = Run(() => new Hooked(path, backend, new Hooks(Scale, Cap)), graphs);
        List<float[]> expected = raw.Select(Transformed).ToList();
        double err = WorstAbsOverCap(got, expected);
        double rawMax = raw.Max(r => r.Max(Math.Abs));
        output.WriteLine($"{backend} {(graphs ? "graphs" : "per-op")}: largest raw |logit| {rawMax:F3} (x scale {rawMax * Scale:F3} vs cap {Cap}); " +
            $"worst |transformed - host transform| / cap {err:E2}");
        Assert.True(rawMax * Scale > 2 * Cap, "the fixture's logits do not reach the softcap's curve");
        Assert.True(err <= 1e-5, $"{backend}: the logits are not the transformed ones (worst {err:E2} of the cap)");
    }

    /// <summary>Every row's logits of a scored prompt chunk (the all-row head, which the speculators' verify windows and the
    /// quality probe read) are transformed on every executor.</summary>
    [Fact]
    public void AllRowLogitsAreTransformedOnTheManagedBackend() => VerifyAllRowLogits(BackendType.Cpu);

    [Fact]
    public void AllRowLogitsAreTransformedOnThePinnedGgmlBackend() => VerifyAllRowLogits(TestGates.PinnedGgmlBackend);

    [MlxFact]
    public void AllRowLogitsAreTransformedOnMlx() => VerifyAllRowLogits(BackendType.Mlx);

    private void VerifyAllRowLogits(BackendType backend)
    {
        string path = Fixture("dense");
        int[] prompt = Enumerable.Range(0, 21).Select(i => 30 + (i * 13) % 170).ToArray();
        List<float[]> Rows(Hooks hooks) => Run(() => new Hooked(path, backend, hooks), graphs: true,
            model => new List<float[]> { (float[])((StandardDecoderModel)model).ForwardPromptRowsForTests(prompt).Clone() });
        float[] raw = Rows(new Hooks())[0], got = Rows(new Hooks(Scale, Cap))[0];
        Assert.Equal(prompt.Length * DenseDecoderSyntheticModelBuilderVocab(), raw.Length);
        double err = WorstAbsOverCap(new List<float[]> { got }, new List<float[]> { Transformed(raw) });
        output.WriteLine($"{backend}: {prompt.Length} rows, worst |transformed - host transform| / cap {err:E2}");
        Assert.True(err <= 1e-5, $"{backend}: the all-row logits are not the transformed ones ({err:E2})");
    }

    private static int DenseDecoderSyntheticModelBuilderVocab() => new DenseDecoderSyntheticModelBuilder { Architecture = "mistral3" }.VocabSize;

    /// <summary>Several requests' slots in batched steps (the multi-sequence head) give what each request gives alone (the
    /// single-sequence head), with the transform on, on every executor.</summary>
    [Fact]
    public void BatchedSlotsAreTransformedAsEachSequenceAloneOnTheManagedBackend() => VerifyBatchedSlots(BackendType.Cpu);

    [Fact]
    public void BatchedSlotsAreTransformedAsEachSequenceAloneOnThePinnedGgmlBackend() => VerifyBatchedSlots(TestGates.PinnedGgmlBackend);

    [MlxFact]
    public void BatchedSlotsAreTransformedAsEachSequenceAloneOnMlx() => VerifyBatchedSlots(BackendType.Mlx);

    private void VerifyBatchedSlots(BackendType backend)
    {
        string path = Fixture("dense");
        using var _ = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        // SharedRuntimeConformanceTests' bounds (managed rows are computed row by row; device kernels may pick another matmul
        // for one row than for several: 5e-3 on MLX, ggml_cuda and ggml_vulkan, 1e-3 elsewhere), times the transform's largest
        // slope (the scale) for the device executors: the transform scales a rounding difference by up to the scale and
        // compresses the logits it is relative to. Measured, transformed / untransformed: ggml_metal 1.21e-3; ggml_cuda
        // 2.72e-3 / 2.15e-3 and ggml_vulkan 9.5e-4 / 8.4e-4 (RTX PRO 4500 Blackwell): ggml_cuda's batched rows are past 1e-3
        // before the transform, as the conformance tests measured them (1.4e-3 / 2.2e-3).
        double tolerance = backend == BackendType.Cpu ? 1e-6
            : (backend is BackendType.Mlx or BackendType.GgmlCuda or BackendType.GgmlVulkan ? 5e-3 : 1e-3) * Scale;
        SlotParity.AssertBatchedSlotsMatchAlone(() => new Hooked(path, backend, new Hooks(Scale, Cap)), tolerance, $"transformed {backend}");
    }

    /// <summary>Tensor parallelism (per-rank programs, with the vocabulary-split head where the ranks score their slices of the
    /// vocabulary), whole-layer stages and their hybrid (whose tensor-parallel stages run the per-op ranks with programs
    /// off; one tensor-parallel stage always runs per-rank programs) on simulated devices give the one-device transformed
    /// logits.</summary>
    [Theory]
    [InlineData(2, 1, true)]
    [InlineData(1, 2, true)]
    [InlineData(1, 2, false)]
    [InlineData(2, 2, true)]
    [InlineData(2, 2, false)]
    public void PlacementsTransformAsOneDevice(int tp, int stages, bool graphs)
    {
        string path = Fixture("dense");
        var hooks = new Hooks(Scale, Cap);
        // The plain one-device logits through the host's transform: a reference no placement's transform takes part in.
        List<float[]> one = Run(() => new Hooked(path, BackendType.Cpu, new Hooks()), graphs).Select(Transformed).ToList();
        bool split = false, programs = false;
        List<float[]> placed = Run(() => new Hooked(path, BackendType.Cpu, hooks, tp, tp > 1 ? new SimulatedDeviceGroup(tp * stages) : null, stages), graphs,
            model =>
            {
                List<float[]> rows = Script(model);
                split = ((StandardDecoderModel)model).HeadSplitByVocabulary;
                programs = ((StandardDecoderModel)model).GraphRuns > 0;
                return rows;
            }, anyPath: true);
        double worst = Worst(placed, one);
        output.WriteLine($"tp {tp} stages {stages} {(graphs ? "graphs" : "per-op")}: worst relL2 {worst:E2} " +
            $"(steps as programs: {programs}; head split by vocabulary: {split})");
        Assert.True(worst <= (tp == 1 ? 1e-6 : 1e-4), $"tp {tp} stages {stages}: relL2 {worst:E2}");
    }

    /// <summary>A family without a transform emits none: the logits of a model declaring scale 1 and no cap are the bytes of
    /// the same model declaring nothing (the default hooks), on both paths.</summary>
    [Fact]
    public void TheDefaultTransformIsNoTransformOnTheManagedBackend() => VerifyNoTransform(BackendType.Cpu);

    [Fact]
    public void TheDefaultTransformIsNoTransformOnThePinnedGgmlBackend() => VerifyNoTransform(TestGates.PinnedGgmlBackend);

    private void VerifyNoTransform(BackendType backend)
    {
        string path = Fixture("dense");
        foreach (bool graphs in new[] { true, false })
        {
            List<float[]> declared = Run(() => new Hooked(path, backend, new Hooks(1f, 0f)), graphs);
            List<float[]> plain = Run(() => new Mistral3Model(path, backend), graphs);
            for (int i = 0; i < plain.Count; i++)
                Assert.Equal(plain[i], declared[i]);
        }
    }

    // ---------------------------------------------------------------- G2: the post norms' epsilon

    private const float MixerEps = 1e-3f, FfnEps = 2e-3f, OutputScale = 1f / 16, EpsScale = OutputScale * OutputScale;

    public static TheoryData<bool, string> Sites() => new() { { true, "mixer" }, { true, "ffn" }, { false, "mixer" }, { false, "ffn" } };

    /// <summary>
    /// Each site's post norm takes its own epsilon on every executor and path: with the site's block output projection 1/16
    /// as large and its epsilon 1/256 as large, the logits are the unscaled model's (bit for bit on the managed backend,
    /// ggml_cpu and MLX, within rounding where an executor's norm is not exact under a power-of-two scale: ggml_metal), while
    /// the same scaled model with the epsilon left alone moves well past that. The other site keeps its own epsilon in
    /// both, so a site that read the other's epsilon fails too.
    /// </summary>
    [Theory]
    [MemberData(nameof(Sites))]
    public void EachPostNormTakesItsOwnEpsilonOnTheManagedBackend(bool graphs, string site) => VerifyPostNormEps(BackendType.Cpu, graphs, site);

    [Theory]
    [MemberData(nameof(Sites))]
    public void EachPostNormTakesItsOwnEpsilonOnThePinnedGgmlBackend(bool graphs, string site) => VerifyPostNormEps(TestGates.PinnedGgmlBackend, graphs, site);

    [MlxTheory]
    [MemberData(nameof(Sites))]
    public void EachPostNormTakesItsOwnEpsilonOnMlx(bool graphs, string site) => VerifyPostNormEps(BackendType.Mlx, graphs, site);

    private void VerifyPostNormEps(BackendType backend, bool graphs, string site)
    {
        string suffix = site == "mixer" ? "attn_output.weight" : "ffn_down.weight";
        string plainPath = Fixture("post", postNorms: true);
        string scaledPath = Fixture("post-" + site, postNorms: true, transform: Scaled(suffix, OutputScale));
        Hooks Eps(bool scaled) => new(PostNorms: true,
            MixerEps: site == "mixer" && scaled ? MixerEps * EpsScale : MixerEps,
            FfnEps: site == "ffn" && scaled ? FfnEps * EpsScale : FfnEps);

        List<float[]> reference = Run(() => new Hooked(plainPath, backend, Eps(false)), graphs);
        List<float[]> scaled = Run(() => new Hooked(scaledPath, backend, Eps(true)), graphs);
        List<float[]> unscaledEps = Run(() => new Hooked(scaledPath, backend, Eps(false)), graphs);
        double same = Worst(scaled, reference), moved = Worst(unscaledEps, reference);
        output.WriteLine($"{backend} {(graphs ? "graphs" : "per-op")} {site}: scaled output and epsilon relL2 {same:E2}; " +
            $"scaled output alone {moved:E2}");
        // ggml_metal's rms_norm is not exact under a power-of-two scale (measured up to 1.8e-4); the scaled output with its
        // epsilon left alone moves four orders of magnitude more.
        Assert.True(Exact(backend) ? same == 0 : same <= 1e-3, $"{backend} {site}: the epsilon did not follow the output's scale ({same:E2})");
        Assert.True(moved >= 1e-2 && moved >= 100 * same, $"{backend} {site}: the fixture does not show the epsilon ({moved:E2})");
    }

    /// <summary>The post norms' epsilon holds on the per-op tensor-parallel layers (each rank's reduced partial is normed),
    /// whole-layer stages and their hybrid.</summary>
    [Theory]
    [InlineData(2, 1, true)]
    [InlineData(1, 2, true)]
    [InlineData(1, 2, false)]
    [InlineData(2, 2, false)]
    public void PlacementsTakeThePostNormEpsilon(int tp, int stages, bool graphs)
    {
        string plainPath = Fixture("post", postNorms: true);
        string scaledPath = Fixture("post-both", postNorms: true,
            transform: (name, values) => Scaled("attn_output.weight", OutputScale)(name, values) ?? Scaled("ffn_down.weight", OutputScale)(name, values));
        ITensorParallelGroup Group() => tp > 1 ? new SimulatedDeviceGroup(tp * stages) : null;
        List<float[]> reference = Run(() => new Hooked(plainPath, BackendType.Cpu, new Hooks(PostNorms: true, MixerEps: MixerEps, FfnEps: FfnEps)), graphs);
        List<float[]> placed = Run(() => new Hooked(scaledPath, BackendType.Cpu,
            new Hooks(PostNorms: true, MixerEps: MixerEps * EpsScale, FfnEps: FfnEps * EpsScale), tp, Group(), stages), graphs, anyPath: true);
        double worst = Worst(placed, reference);
        output.WriteLine($"tp {tp} stages {stages} {(graphs ? "graphs" : "per-op")}: relL2 {worst:E2}");
        Assert.True(worst <= (tp == 1 ? 0 : 1e-4), $"tp {tp} stages {stages}: relL2 {worst:E2}");
    }

    // ---------------------------------------------------------------- G3: the stream input

    private const float StreamScale = 4f;

    /// <summary>The fixture whose token embedding is stored <see cref="StreamScale"/> times larger.</summary>
    private string ScaledEmbedding() => Fixture("embed-x4", transform: Scaled("token_embd.weight", StreamScale));

    /// <summary>
    /// The stream input runs once on every executor and path: a model scaling the embedded rows by 4 gives the logits of
    /// the model whose embedding is stored 4 times larger (bit for bit on the managed backend and ggml_cpu; within rounding
    /// where an executor's kernels see other values), and not the unscaled model's.
    /// </summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void TheStreamInputRunsOnceOnTheManagedBackend(bool graphs) => VerifyStreamInput(BackendType.Cpu, graphs);

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void TheStreamInputRunsOnceOnThePinnedGgmlBackend(bool graphs) => VerifyStreamInput(TestGates.PinnedGgmlBackend, graphs);

    [MlxTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void TheStreamInputRunsOnceOnMlx(bool graphs) => VerifyStreamInput(BackendType.Mlx, graphs);

    private void VerifyStreamInput(BackendType backend, bool graphs)
    {
        string path = Fixture("dense"), scaledPath = ScaledEmbedding();
        List<float[]> hooked = Run(() => new Hooked(path, backend, new Hooks(StreamScale: StreamScale)), graphs);
        List<float[]> stored = Run(() => new Hooked(scaledPath, backend, new Hooks()), graphs);
        List<float[]> plain = Run(() => new Hooked(path, backend, new Hooks()), graphs);
        double same = Worst(hooked, stored), moved = Worst(plain, stored);
        output.WriteLine($"{backend} {(graphs ? "graphs" : "per-op")}: hook vs stored relL2 {same:E2}; without the hook {moved:E2}");
        Assert.True(Exact(backend) ? same == 0 : same <= 1e-4, $"{backend}: the stream input did not run once ({same:E2})");
        Assert.True(moved >= 1e-2, $"{backend}: the fixture does not show the stream input ({moved:E2})");
    }

    /// <summary>
    /// Media rows join the stream before the stream input, as the graph path's injection does: queued media rows scaled by
    /// the hook give what the same rows stored 4 times larger give without it, on both paths of every executor.
    /// </summary>
    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void MediaRowsTakeTheStreamInputOnTheManagedBackend(bool graphs) => VerifyMediaRows(BackendType.Cpu, graphs);

    [Theory]
    [InlineData(true)]
    [InlineData(false)]
    public void MediaRowsTakeTheStreamInputOnThePinnedGgmlBackend(bool graphs) => VerifyMediaRows(TestGates.PinnedGgmlBackend, graphs);

    [MlxTheory]
    [InlineData(true)]
    [InlineData(false)]
    public void MediaRowsTakeTheStreamInputOnMlx(bool graphs) => VerifyMediaRows(BackendType.Mlx, graphs);

    private void VerifyMediaRows(BackendType backend, bool graphs)
    {
        string path = Fixture("dense"), scaledPath = ScaledEmbedding();
        int hidden = DenseDecoderSyntheticModelBuilder.Hidden;
        var random = new Random(3);
        float[] media = Enumerable.Range(0, 5 * hidden).Select(_ => (float)(random.NextDouble() - 0.5) * 0.2f).ToArray();
        int[] prompt = Enumerable.Range(0, 17).Select(i => 50 + (i * 7) % 120).ToArray();
        Func<ModelBase, List<float[]>> Script(float mediaScale) => model =>
        {
            var rows = new Tensor(model.DirectAllocatorForTelemetry, DType.Float32, 5, hidden);
            rows.SetElementsAsFloat(media.Select(v => v * mediaScale).ToArray());
            ((StandardDecoderModel)model).QueueMediaForTests(rows, 6);
            var logits = new List<float[]> { (float[])model.Forward(prompt).Clone() };
            logits.Add((float[])model.Forward(new[] { 77 }).Clone());
            return logits;
        };
        List<float[]> hooked = Run(() => new Hooked(path, backend, new Hooks(StreamScale: StreamScale)), graphs, Script(1f));
        List<float[]> stored = Run(() => new Hooked(scaledPath, backend, new Hooks()), graphs, Script(StreamScale));
        List<float[]> unscaledMedia = Run(() => new Hooked(scaledPath, backend, new Hooks()), graphs, Script(1f));
        double same = Worst(hooked, stored), moved = Worst(unscaledMedia, stored);
        output.WriteLine($"{backend} {(graphs ? "graphs" : "per-op")}: media through the hook vs stored relL2 {same:E2}; media unscaled {moved:E2}");
        Assert.True(Exact(backend) ? same == 0 : same <= 1e-4, $"{backend}: the media rows did not take the stream input ({same:E2})");
        Assert.True(moved >= 1e-3, $"{backend}: the fixture does not show the media rows ({moved:E2})");
    }

    /// <summary>The stream input runs once under tensor parallelism (before the stream is replicated to the ranks), whole-layer
    /// stages and their hybrid, on the per-op path and as programs.</summary>
    [Theory]
    [InlineData(2, 1, true)]
    [InlineData(1, 2, true)]
    [InlineData(1, 2, false)]
    [InlineData(2, 2, true)]
    [InlineData(2, 2, false)]
    public void PlacementsRunTheStreamInputOnce(int tp, int stages, bool graphs)
    {
        string path = Fixture("dense"), scaledPath = ScaledEmbedding();
        ITensorParallelGroup Group() => tp > 1 ? new SimulatedDeviceGroup(tp * stages) : null;
        List<float[]> stored = Run(() => new Hooked(scaledPath, BackendType.Cpu, new Hooks()), graphs);
        List<float[]> placed = Run(() => new Hooked(path, BackendType.Cpu, new Hooks(StreamScale: StreamScale), tp, Group(), stages), graphs, anyPath: true);
        double worst = Worst(placed, stored);
        output.WriteLine($"tp {tp} stages {stages} {(graphs ? "graphs" : "per-op")}: relL2 {worst:E2}");
        Assert.True(worst <= (tp == 1 ? 0 : 1e-4), $"tp {tp} stages {stages}: relL2 {worst:E2}");
    }

    /// <summary>
    /// A multi-node pipeline (two nodes on loopback; single-device stages, which run layer by layer, and a tensor-parallel
    /// stage on the driver) applies the stream input once, on the node that embeds, with the logit transform and the post
    /// norms' epsilon declared too: the driver's logits are the one-device model's as programs.
    /// </summary>
    [Theory]
    [InlineData("1x1,1x1")]
    [InlineData("1x2,1x1")]
    public async Task AMultiNodePipelineAppliesTheHooksOnce(string shapes)
    {
        string path = Fixture("post", postNorms: true);
        var hooks = new Hooks(Scale, Cap, PostNorms: true, MixerEps: MixerEps, FfnEps: FfnEps, StreamScale: StreamScale);
        // One device as programs: the graph's hooks, which no per-op code takes part in (a pipeline's stages run per op).
        List<float[]> reference = Run(() => new Hooked(path, BackendType.Cpu, hooks), graphs: true);

        using var _ = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        var nodeShapes = shapes.Split(',').Select(s => s.Split('x').Select(int.Parse).ToArray()).ToArray();
        IPEndPoint[] endpoints = Endpoints(nodeShapes.Length);
        var models = await Task.WhenAll(Enumerable.Range(0, nodeShapes.Length).Select(node => Task.Run(() =>
        {
            int stages = nodeShapes[node][0], tp = nodeShapes[node][1];
            var group = new DistributedLayerSplitGroup(new SimulatedDeviceGroup(stages * tp), node, endpoints, tp, 0);
            try
            {
                using var flow = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
                return (ModelBase)new Hooked(path, BackendType.Cpu, hooks, 1, group, 1);
            }
            catch { group.Dispose(); throw; }
        }))).WaitAsync(TimeSpan.FromSeconds(60));
        var workers = models.Skip(1).Select(model => Task.Run(model.RunDistributedWorkerLoop)).ToArray();
        try
        {
            ModelBase driver = models[0];
            driver.BeginDistributedDriver();
            List<float[]> actual = await Task.Run(() => Script(driver)).WaitAsync(TimeSpan.FromSeconds(120));
            double worst = Worst(actual, reference);
            output.WriteLine($"{shapes}: relL2 {worst:E2}");
            // The managed programs and the per-op path round alike up to the order of a few sums (the conformance tests
            // hold them to 1e-5); tensor-parallel partials reorder the sums too.
            Assert.True(worst <= 1e-4, $"{shapes}: relL2 {worst:E2}");
            ((DistributedLayerSplitGroup)typeof(ModelBase).GetProperty("LayerSplitGroup",
                System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.GetValue(driver)!).BroadcastControl(4, []);
            await Task.WhenAll(workers).WaitAsync(TimeSpan.FromSeconds(20));
        }
        finally
        {
            foreach (var model in models) model.Dispose();
        }
    }

    private static IPEndPoint[] Endpoints(int count)
    {
        var listeners = Enumerable.Range(0, count).Select(_ => new TcpListener(IPAddress.Loopback, 0)).ToArray();
        try
        {
            foreach (var listener in listeners) listener.Start();
            return listeners.Select(listener => (IPEndPoint)listener.LocalEndpoint).ToArray();
        }
        finally { foreach (var listener in listeners) listener.Stop(); }
    }
}
