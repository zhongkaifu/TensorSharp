// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Qwen 3.5 on the shared decoder runtime (Qwen35RuntimeModel): the hybrid Gated DeltaNet +
// gated-attention recipe matches the family's own implementation, and every runtime
// mechanism - tensor parallelism, whole-layer stages, their hybrid, continuous batching -
// holds for a recurrent stack as it does for attention-only ones. Devices are simulated
// (SimulatedDeviceGroup), so this is functional evidence, not GPU evidence.
using Microsoft.Extensions.Logging.Abstractions;

using TensorSharp;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

[Collection("MoeCpuOffloadConfig")]
public sealed class Qwen35RuntimeTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-qwen35-runtime-" + Guid.NewGuid().ToString("N"));
    private readonly KvCacheDtype _dtype = KvCacheDtypeConfig.Current;
    private readonly bool _dtypeExplicit = KvCacheDtypeConfig.IsExplicitlySet;

    public Qwen35RuntimeTests()
    {
        Directory.CreateDirectory(_dir);
        MoeCpuOffloadConfig.Reset();
    }

    public void Dispose()
    {
        MoeCpuOffloadConfig.Reset();
        KvCacheDtypeConfig.RestoreForTests(_dtype, _dtypeExplicit);
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private string Fixture(string variant) => variant switch
    {
        "dense" => new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true }.Write(Path.Combine(_dir, "q35-dense.gguf")),
        // In-projection sources of two types (as mixed-quant checkpoints have): the graph runs them
        // unpacked, the per-op block packs them on first use.
        // Three value heads per key head, as the 27B (value head h reads key head h mod group_count).
        "dense-gqa" => new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true, SsmVHeads = 6 }.Write(Path.Combine(_dir, "q35-dense-gqa.gguf")),
        "dense-nextn" => new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true, NextN = true }.Write(Path.Combine(_dir, "q35-dense-nextn.gguf")),
        "dense-mixed" => new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true, GateProjectionType = SyntheticGguf.GgmlType.F32 }
            .Write(Path.Combine(_dir, "q35-dense-mixed.gguf")),
        // BF16 weights: no activation quantization, so the comparison is tight.
        "moe" => new Qwen35MoeSyntheticModelBuilder { Layers = 8, UniformWeightType = SyntheticGguf.GgmlType.BF16 }.Write(Path.Combine(_dir, "q35-moe.gguf")),
        // The quantized mix (IQ3_S / IQ4_XS experts, Q8_0 projections) quantizes activations, so
        // a rounding flip upstream moves outputs by ~1e-2 in either implementation.
        "moe-quantized" => new Qwen35MoeSyntheticModelBuilder { Layers = 8 }.Write(Path.Combine(_dir, "q35-moe-q.gguf")),
        _ => throw new ArgumentOutOfRangeException(nameof(variant)),
    };

    private static Qwen35RuntimeModel Create(string path, int tp = 1, int stages = 1, BackendType backend = BackendType.Cpu)
        => new(path, backend, tp, tp > 1 ? new SimulatedDeviceGroup(tp * stages) : null, stages);

    /// <summary>The factory's route: one ggml_cuda / ggml_metal / MLX device runs a dense checkpoint on the
    /// shared runtime; routed experts, --tp, and every other backend stay with Qwen35Model.</summary>
    [Theory]
    [InlineData("dense", BackendType.GgmlCuda, 1, true)]
    [InlineData("dense", BackendType.GgmlMetal, 1, true)]
    [InlineData("dense-nextn", BackendType.GgmlCuda, 1, true)]
    [InlineData("dense", BackendType.GgmlCuda, 2, false)]
    [InlineData("dense", BackendType.GgmlCpu, 1, false)]
    [InlineData("dense", BackendType.Cuda, 1, false)]
    [InlineData("dense", BackendType.Mlx, 1, true)]
    [InlineData("moe", BackendType.Mlx, 1, false)]
    [InlineData("moe", BackendType.GgmlCuda, 1, false)]
    [InlineData("moe", BackendType.GgmlMetal, 1, false)]
    public void OneDeviceRouteServesDenseCheckpointsOnGgmlCudaAndMetal(string variant, BackendType backend, int tp, bool runtime)
    {
        using var probe = new GgufFile(Fixture(variant));
        var context = new TensorSharp.Models.Architecture.ModelCreateContext(probe.FilePaths[0], backend, probe, tp);
        Assert.Equal(runtime, Qwen35RuntimeModel.ServesOneDevice(context));
        Assert.Equal(runtime, Qwen35RuntimeModel.Serves(context));
    }

    /// <summary>Prefill, decode, reset, prefill, decode: a recurrent stack cannot rewind.</summary>
    private static List<float[]> Script(ModelBase model)
    {
        var rows = new List<float[]>();
        int[] prompt = Enumerable.Range(0, 17).Select(i => 40 + (i * 11) % 150).ToArray();
        rows.Add((float[])model.ForwardRefill(prompt).Clone());
        foreach (int token in new[] { 71, 72, 73, 74 })
            rows.Add((float[])model.Forward(new[] { token }).Clone());
        model.ResetKVCache();
        rows.Add((float[])model.ForwardRefill(new[] { 33, 34, 35, 36, 37, 38, 39 }).Clone());
        rows.Add((float[])model.Forward(new[] { 40 }).Clone());
        return rows;
    }

    /// <summary>Graph vs per-op bound: the per-op path runs a mixed-type in-projection through the
    /// same graph lowering, so every variant is bound alike.</summary>
    private static double PerOpTolerance(string variant)
        => RoundsActivations(TestGates.PinnedGgmlBackend) ? 2.5e-2 : 5e-3;

    /// <summary>Backends whose matmuls round activations (ggml_cpu to BF16; ggml_cuda to BF16 / Q8_1,
    /// with different kernels for one row than for several): measured on ggml_cuda (RTX PRO 6000,
    /// 2026-11-10) up to 7.4e-3 against the managed reference and 5.2e-3 graph vs per-op on these
    /// fixtures, ggml_cpu up to 1.5e-2.</summary>
    private static bool RoundsActivations(BackendType backend) => backend is BackendType.GgmlCpu or BackendType.GgmlCuda;

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

    [Theory]
    [InlineData("dense", 1e-4)]
    [InlineData("dense-gqa", 1e-4)]
    [InlineData("moe", 1e-4)]
    // Measured 2026-11-10 (M5 Pro): 7.9e-3; the legacy implementation's own prefill and decode
    // disagree by 2.5e-3 on this fixture, and ggml vs the managed kernels by 5e-2.
    [InlineData("moe-quantized", 3e-2)]
    public void MatchesTheFamilyImplementation(string variant, double tolerance)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(variant);
        List<float[]> legacy;
        using (var family = new Qwen35Model(path, BackendType.Cpu))
            legacy = Script(family);
        using var runtime = Create(path);
        List<float[]> actual = Script(runtime);
        for (int i = 0; i < legacy.Count; i++)
        {
            double rel = RelL2(actual[i], legacy[i]);
            Assert.True(rel <= tolerance, $"{variant} row {i}: relL2 {rel:E2} from Qwen35Model");
        }
    }

    /// <summary>The runtime recipe on a real compute backend (the pinned GGML backend, MLX)
    /// against the managed reference, as the shared-runtime conformance does for every family.</summary>
    [Theory]
    [InlineData("dense")]
    [InlineData("dense-mixed")]
    [InlineData("dense-gqa")]
    [InlineData("moe")]
    public void PinnedGgmlBackendMatchesTheManagedReference(string variant) => CompareBackend(variant, TestGates.PinnedGgmlBackend);

    [MlxTheory]
    [InlineData("dense")]
    [InlineData("dense-gqa")]
    [InlineData("moe")]
    public void MlxMatchesTheManagedReference(string variant) => CompareBackend(variant, BackendType.Mlx);

    /// <summary>Every MoE layer's experts in host memory (--n-cpu-moe all) on the pinned GGML backend: the step's
    /// graph pauses for them (GraphOp.MoeHostExperts). The host's expert kernels answer as the accelerator's do
    /// (except quantized experts on Metal and CUDA, which quantize the activations unlike the host: Qwen35Model's own
    /// host and accelerator runs sit 0.1-0.5 apart on this fixture there, from route flips), and
    /// they keep the runtime as close to Qwen35Model's offloaded run as the accelerator's keep it to Qwen35Model's.</summary>
    [Theory]
    [InlineData("moe")]
    [InlineData("moe-quantized")]
    public void HostExpertsInTheGraphMatchTheAcceleratorAndQwen35Model(string variant)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        BackendType backend = TestGates.PinnedGgmlBackend;
        string path = Fixture(variant);
        List<float[]> device, host, legacyDevice, legacyHost, tiered;
        MoeCpuOffloadConfig.Reset();
        try
        {
            using (var model = Create(path, backend: backend)) device = Script(model);
            using (var model = new Qwen35Model(path, backend)) legacyDevice = Script(model);
            MoeCpuOffloadConfig.SetAllLayers();
            using (var model = Create(path, backend: backend))
            {
                host = Script(model);
                Assert.True(model.GraphRuns > 0, "the offloaded steps did not run as graphs");
            }
            using (var model = new Qwen35Model(path, backend)) legacyHost = Script(model);
            // A tier holding every expert (fixed slots): every route runs on the accelerator at the pause.
            using (var model = Create(path, backend: backend))
            {
                Assert.True(model.RegisterExpertTiersForTests(int.MaxValue, 0) > 0, "no layer took a tier");
                tiered = Script(model);
            }
        }
        finally
        {
            MoeCpuOffloadConfig.Reset();
        }
        // ggml_cuda: the paused graphs' last row (a second request's decode) sits 1.4e-2 from the unpaused one's with
        // the tier and with the host alike (BF16, RTX PRO 6000); ggml_metal 5.6e-3.
        Assert.True(tiered.Select((r, i) => RelL2(r, device[i])).All(r => r <= (variant == "moe" ? 2e-2 : 3e-2)),
            $"{variant} on {backend}: a tier holding every expert, relL2 from the accelerator's experts: " +
            string.Join(", ", tiered.Select((r, i) => RelL2(r, device[i]).ToString("E2"))));
        string Rows(List<float[]> a, List<float[]> b) => string.Join(", ", a.Select((r, i) => RelL2(r, b[i]).ToString("E2")));
        // Quantized experts: the accelerators quantize the activations their own way (Metal not at all, CUDA as
        // q8_1), so only ggml_cpu, whose experts are the host's kernels, compares them.
        if (variant == "moe" || backend == BackendType.GgmlCpu)
        {
            // BF16 experts: ggml-cuda rounds the activations to BF16 (1.4e-2 measured on an RTX PRO 6000).
            double tolerance = variant == "moe" ? 2e-2 : 3e-2;
            Assert.True(host.Select((r, i) => RelL2(r, device[i])).All(r => r <= tolerance),
                $"{variant} on {backend}: relL2 from the accelerator's experts: {Rows(host, device)}");
        }
        // The runtime and Qwen35Model route this fixture's near-ties differently: with quantized experts they sit up
        // to 6.1e-2 apart even on the accelerator (ggml_cpu), with BF16 experts 1.2e-2 (ggml_cuda). Worst row, host
        // experts against Qwen35Model's (2026-11-11): ggml_metal 1.9e-3 (moe) / 2.6e-2 (moe-quantized), ggml_cuda
        // 1.1e-2 (moe).
        double fromLegacy = variant == "moe" ? 2e-2 : 7e-2;
        Assert.True(host.Select((r, i) => RelL2(r, legacyHost[i])).All(r => r <= fromLegacy),
            $"{variant} on {backend}: host experts relL2 from Qwen35Model's {Rows(host, legacyHost)}; accelerator experts {Rows(device, legacyDevice)}");
    }

    private void CompareBackend(string variant, BackendType backend)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(variant);
        List<float[]> reference;
        using (var managed = Create(path))
            reference = Script(managed);
        using var model = Create(path, backend: backend);
        List<float[]> actual = Script(model);
        // Measured on an M5 Pro (2026-11-10), worst row: ggml_metal 1.0e-3 (dense) / 1.4e-3
        // (moe), MLX 6.3e-4 / 1.2e-3; ggml_cpu 4.4e-3 / 1.5e-2, where ggml's BF16 matmul rounds
        // the activations to BF16 - Qwen35Model on ggml_cpu sits 4.9e-3 / 1.5e-2 from the same
        // reference. The Gated DeltaNet recurrence itself runs the same host code everywhere.
        // MLX's MoE row 0 (the prompt chunk) sits 5.9e-3 away since the runtime runs as programs on
        // MLX (M5 Pro, 2026-11-10): MLX and the managed executor pick different experts in 3 of the
        // fixture's 232 routings, all near-ties (top-k margins 7.6e-4 .. 2.4e-3 relative); the gated
        // delta kernel matches the host recurrence to 3e-7 on the same inputs.
        double tolerance = RoundsActivations(backend) ? 2.5e-2 : backend == BackendType.Mlx && variant == "moe" ? 1e-2 : 5e-3;
        for (int i = 0; i < reference.Count; i++)
        {
            double rel = RelL2(actual[i], reference[i]);
            Assert.True(rel <= tolerance, $"{variant} on {backend} row {i}: relL2 {rel:E2}");
        }
    }

    public static TheoryData<int, int> Placements() => new() { { 2, 1 }, { 1, 2 }, { 1, 4 }, { 2, 2 } };

    [Theory]
    [MemberData(nameof(Placements))]
    public void PlacementMatchesOneDevice(int tp, int stages)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense");
        List<float[]> reference;
        using (var single = Create(path))
            reference = Script(single);
        using var placed = Create(path, tp, stages);
        Assert.Equal(tp, placed.Mesh.TensorParallel);
        Assert.Equal(stages, placed.Mesh.Stages);
        List<float[]> actual = Script(placed);
        double tolerance = tp == 1 ? 0 : 1e-4;
        for (int i = 0; i < reference.Count; i++)
        {
            double rel = RelL2(actual[i], reference[i]);
            Assert.True(rel <= tolerance, $"tp={tp} stages={stages} row {i}: relL2 {rel:E2} > {tolerance:E0}");
        }
    }

    /// <summary>Routed experts on the host (--cpu-moe), on one device and on whole-layer
    /// stages, answer as the experts on the device: Qwen 3.5 MoE offload on every backend
    /// Qwen35Model keeps them in device buffers (cuda, mlx) goes through here.</summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    public void HostExpertsMatchDeviceExperts(int stages)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("moe");
        List<float[]> reference;
        using (var device = Create(path))
            reference = Script(device);
        MoeCpuOffloadConfig.SetAllLayers();
        using var offloaded = Create(path, stages: stages);
        List<float[]> actual = Script(offloaded);
        for (int i = 0; i < reference.Count; i++)
        {
            double rel = RelL2(actual[i], reference[i]);
            Assert.True(rel <= 1e-4, $"--cpu-moe, {stages} stage(s), row {i}: relL2 {rel:E2}");
        }
    }

    [Fact]
    public void ARecurrentStackRewindsOnlyToItsLengthOrToZero()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        using var model = Create(Fixture("dense"));
        float[] first = (float[])model.ForwardRefill(Enumerable.Range(0, 12).Select(i => 50 + i).ToArray()).Clone();
        Assert.False(model.CanTruncateKVCache(12, 10));
        Assert.False(model.TryTruncateKVCache(10));
        Assert.False(model.SupportsKVStateSnapshot);
        // The prefix cache sees a recurrent stack: no pages to resume from, no rewinds.
        var caps = model.GetPrefixCacheCapabilities();
        Assert.Equal(TensorSharp.Runtime.Scheduling.PrefixCache.FamilyClass.R, caps.Class);
        Assert.Equal(TensorSharp.Runtime.Scheduling.PrefixCache.TruncationKind.None, caps.Truncation);
        Assert.Equal(TensorSharp.Runtime.Scheduling.PrefixCache.PageSupport.None, caps.Pages);
        Assert.True(model.TryTruncateKVCache(12));
        Assert.True(model.TryTruncateKVCache(0));
        float[] again = model.ForwardRefill(Enumerable.Range(0, 12).Select(i => 50 + i).ToArray());
        Assert.Equal(first, again);
    }

    /// <summary>Slots carry the Gated DeltaNet state with the attention rows: batched steps over
    /// several requests reproduce each request alone, in every placement and on the device backends
    /// (device recurrence on GGML).</summary>
    [Theory]
    [InlineData("dense", 1, 1)]
    [InlineData("moe", 1, 1)]
    [InlineData("dense", 2, 1)]
    [InlineData("dense", 1, 2)]
    [InlineData("moe", 1, 2)]
    [InlineData("dense", 2, 2)]
    public void SlotBatchesMatchEachSequenceAlone(string variant, int tp, int stages)
        => CompareSlotBatches(variant, tp, stages, BackendType.Cpu);

    [Theory]
    [InlineData("dense")]
    [InlineData("moe")]
    public void SlotBatchesMatchEachSequenceAloneOnThePinnedGgmlBackend(string variant)
        => CompareSlotBatches(variant, 1, 1, TestGates.PinnedGgmlBackend);

    [MlxTheory]
    [InlineData("dense")]
    [InlineData("moe")]
    public void SlotBatchesMatchEachSequenceAloneOnMlx(string variant)
        => CompareSlotBatches(variant, 1, 1, BackendType.Mlx);

    private void CompareSlotBatches(string variant, int tp, int stages, BackendType backend)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(variant);
        double tolerance = backend == BackendType.Cpu ? 1e-6 : RoundsActivations(backend) ? 2.5e-2 : 5e-3;
        SlotParity.AssertBatchedSlotsMatchAlone(() => Create(path, tp, stages, backend), tolerance,
            $"qwen35 {variant} tp={tp} stages={stages} {backend}");
    }

    /// <summary>On a GGML backend every step (attention, Gated DeltaNet with its slot state, MoE)
    /// runs as one whole-forward graph and computes what the per-op path computes.</summary>
    [Theory]
    [InlineData("dense")]
    [InlineData("dense-mixed")]
    [InlineData("dense-gqa")]
    [InlineData("moe")]
    public void WholeForwardGraphMatchesThePerOpPath(string variant)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(variant);
        List<float[]> perOp;
        using (var model = Create(path, backend: TestGates.PinnedGgmlBackend))
        {
            model.GraphLoweringEnabled = false;
            perOp = Script(model);
        }
        using var graph = Create(path, backend: TestGates.PinnedGgmlBackend);
        List<float[]> rows = Script(graph);
        Assert.True(graph.GraphRuns >= rows.Count, "every step must run as a whole-forward graph");
        Assert.True(graph.GraphReplays > 0, "consecutive decode tokens must replay their graph");
        double tolerance = PerOpTolerance(variant);
        for (int i = 0; i < perOp.Count; i++)
        {
            double rel = RelL2(rows[i], perOp[i]);
            Assert.True(rel <= tolerance, $"{variant} row {i}: graph vs per-op relL2 {rel:E2}");
        }
    }

    /// <summary>One request alternating between graph steps and per-op steps (as an adapter step
    /// forces) keeps its attention cache and recurrent state coherent across both bindings.</summary>
    [Theory]
    [InlineData("dense")]
    [InlineData("dense-mixed")]
    [InlineData("dense-gqa")]
    [InlineData("moe")]
    public void GraphAndPerOpStepsShareTheState(string variant)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(variant);
        int[] prompt = Enumerable.Range(0, 17).Select(i => 40 + (i * 11) % 150).ToArray();
        int[] tokens = { 71, 72, 73, 74, 75, 76 };
        var perOp = new List<float[]>();
        using (var model = Create(path, backend: TestGates.PinnedGgmlBackend))
        {
            model.GraphLoweringEnabled = false;
            perOp.Add((float[])model.ForwardRefill(prompt).Clone());
            foreach (int t in tokens) perOp.Add((float[])model.Forward(new[] { t }).Clone());
        }
        using var mixed = Create(path, backend: TestGates.PinnedGgmlBackend);
        var rows = new List<float[]> { (float[])mixed.ForwardRefill(prompt).Clone() };
        for (int i = 0; i < tokens.Length; i++)
        {
            mixed.GraphLoweringEnabled = i is not (2 or 3);   // two per-op steps in the middle
            rows.Add((float[])mixed.Forward(new[] { tokens[i] }).Clone());
        }
        Assert.True(mixed.GraphRuns > 0);
        double tolerance = PerOpTolerance(variant);
        for (int i = 0; i < perOp.Count; i++)
        {
            double rel = RelL2(rows[i], perOp[i]);
            Assert.True(rel <= tolerance, $"{variant} row {i}: mixed vs per-op relL2 {rel:E2}");
        }
    }

    /// <summary>Greedy speculative decoding on the runtime (n-gram drafts, so verifies are partly
    /// accepted and partly rejected) emits exactly the tokens plain greedy decoding emits: the
    /// verify keeps each row's recurrent state and the accept count selects the accepted prefix's.</summary>
    [Theory]
    [InlineData("dense")]
    [InlineData("dense-mixed")]
    [InlineData("moe")]
    public void SpeculativeDecodingMatchesPlainGreedy(string variant)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(variant);
        // ggml_cuda's verify kernels round differently than its one-row kernels: the moe fixture's
        // token 7 is a 0.36% tie there (RTX PRO 6000, 2026-11-10) and the verify takes the runner-up.
        SpeculativeRuntimeChecks.AssertMatchesPlainGreedy(() => Create(path, backend: TestGates.PinnedGgmlBackend), $"qwen35 {variant}",
            tieTolerance: RoundsActivations(TestGates.PinnedGgmlBackend) ? 1e-2 : 0);
    }

    /// <summary>A decode step that repeats the last one a position on runs again without a trace: every logit is the
    /// traced step's, for one request and for two in one step, across a decode window's bucket edge. (An executor
    /// whose plans are not replayed, the managed CPU, traces every step anyway.)</summary>
    [Theory]
    [InlineData("dense")]
    [InlineData("moe")]
    public void AdvancedDecodeStepsMatchTracedOnes(string variant)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        BackendType backend = TestGates.PinnedGgmlBackend;
        string path = Fixture(variant);

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
                unified.EnsureSequenceSlot(ids[s], 320);
                var logits = new float[1][];
                Assert.True(unified.TryForwardUnified(new[] { ids[s] }, new[] { prompts[s] }, new[] { 0 }, logits));
                next[s] = ArgMax(logits[0]);
                lengths[s] = prompts[s].Length;
            }
            // "a" alone past the 64-row bucket edge, then both together, then "b" alone past the 256-row one.
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
            Decode(new[] { "a" }, 50);
            Decode(new[] { "a", "b" }, 30);
            Decode(new[] { "b" }, 220);
            advances = model.DecodeAdvances;
            return trace;
        }

        List<float[]> traced = Run(advance: false, out int none);
        List<float[]> advanced = Run(advance: true, out int advances);
        Assert.Equal(0, none);
        Assert.True(advances > 200, $"{advances} decode steps advanced");
        Assert.Equal(traced.Count, advanced.Count);
        for (int i = 0; i < traced.Count; i++)
            Assert.True(traced[i].AsSpan().SequenceEqual(advanced[i]), $"row {i}: advanced logits differ (relL2 {RelL2(advanced[i], traced[i]):E2})");
    }

    /// <summary>The engine's unified step speculates: n-gram and MTP drafts of one request or of several at once,
    /// each over its own drafter state and verified together on their slots, give the plain greedy streams.</summary>
    [Theory]
    [InlineData("dense", SpeculatorRegistry.NGram, BackendType.Cpu)]
    [InlineData("moe", SpeculatorRegistry.NGram, BackendType.Cpu)]
    [InlineData("dense-nextn", SpeculatorRegistry.DraftHead, BackendType.Cpu)]
    [InlineData("dense", SpeculatorRegistry.NGram, BackendType.GgmlCpu)]
    [InlineData("dense-nextn", SpeculatorRegistry.DraftHead, BackendType.GgmlCpu)]
    public async Task EngineSpeculationMatchesPlain(string variant, string speculator, BackendType backend)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        if (backend == BackendType.GgmlCpu) backend = TestGates.PinnedGgmlBackend;
        string path = Fixture(variant);
        var options = new SpeculationOptions { Enabled = true, SpeculatorName = speculator, MaxDraftTokens = 3, MinDraftProb = 0 };
        await SpeculativeRuntimeChecks.AssertEngineSpeculationMatchesPlain(() => Create(path, backend: backend), options,
            $"qwen35 {variant} {speculator} {backend}");
    }

    /// <summary>The NextN/MTP block drafts through the runtime's generic draft layer: speculation
    /// emits the plain greedy tokens.</summary>
    [Fact]
    public void MtpSpeculationMatchesPlainGreedy()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense-nextn");
        SpeculativeRuntimeChecks.AssertDraftHeadMatchesPlainGreedy(() => Create(path, backend: TestGates.PinnedGgmlBackend), "qwen35 nextn",
            tieTolerance: RoundsActivations(TestGates.PinnedGgmlBackend) ? 1e-2 : 0);
    }

    /// <summary>On MLX speculation runs off the graphs: the per-op verify keeps the Gated DeltaNet's
    /// window in MLX arrays and the draft layer runs per op; both emit the plain greedy tokens.</summary>
    [MlxFact]
    public void MlxSpeculationMatchesPlainGreedy()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense-nextn");
        SpeculativeRuntimeChecks.AssertMatchesPlainGreedy(() => Create(path, backend: BackendType.Mlx), "qwen35 mlx");
        SpeculativeRuntimeChecks.AssertDraftHeadMatchesPlainGreedy(() => Create(path, backend: BackendType.Mlx), "qwen35 mlx nextn");
    }

    /// <summary>The per-op draft layer (MLX) computes what the graph one does.</summary>
    [MlxFact]
    public void MlxDraftHeadMatchesTheFamilyImplementation()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense-nextn");
        List<float[]> legacy;
        using (var family = new Qwen35Model(path, BackendType.Cpu))
            legacy = DraftTrace(family);
        using var runtime = Create(path, backend: BackendType.Mlx);
        List<float[]> actual = DraftTrace(runtime);
        Assert.Equal(legacy.Count, actual.Count);
        for (int i = 0; i < legacy.Count; i++)
        {
            double rel = RelL2(actual[i], legacy[i]);
            Assert.True(rel <= 5e-3, $"mlx draft row {i}: relL2 {rel:E2} from Qwen35Model");
        }
    }

    /// <summary>The runtime's draft layer computes what Qwen35Model's MTP head does: prompt
    /// catch-up, two chained draft steps, a gap (no trunk rows) and the fused catch-up step.</summary>
    [Fact]
    public void MtpDraftHeadMatchesTheFamilyImplementation()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense-nextn");
        List<float[]> legacy;
        using (var family = new Qwen35Model(path, BackendType.Cpu))
            legacy = DraftTrace(family);
        using var runtime = Create(path, backend: TestGates.PinnedGgmlBackend);
        List<float[]> actual = DraftTrace(runtime);
        double tolerance = RoundsActivations(TestGates.PinnedGgmlBackend) ? 2.5e-2 : 5e-3;
        Assert.Equal(legacy.Count, actual.Count);
        for (int i = 0; i < legacy.Count; i++)
        {
            double rel = RelL2(actual[i], legacy[i]);
            Assert.True(rel <= tolerance, $"draft row {i}: relL2 {rel:E2} from Qwen35Model");
        }
    }

    private static List<float[]> DraftTrace(ModelBase model)
    {
        var spec = Assert.IsAssignableFrom<TensorSharp.Runtime.Speculative.ISpeculativeModel>(model);
        Assert.Equal(TensorSharp.Runtime.Speculative.DraftHeadKind.PerToken, spec.DraftHeadKind);
        int hidden = model.Config.HiddenSize, vocab = model.Config.VocabSize;
        int[] prompt = Enumerable.Range(0, 17).Select(i => 40 + (i * 11) % 150).ToArray();
        int n = prompt.Length;
        var rows = new List<float[]>();
        var hAll = new float[n * hidden];
        var logits = new float[vocab];
        spec.SpecForward(prompt, hAll, logits, false);
        rows.Add((float[])logits.Clone());
        rows.Add(hAll[((n - 1) * hidden)..]);
        // Row k of the catch-up is the hidden row of the token before tokens[k].
        spec.DraftCatchUp(prompt[1..], hAll[..((n - 1) * hidden)], 1);
        var draft = new float[vocab];
        var h1 = new float[hidden];
        int next = ArgMax(logits);
        spec.DraftStep(next, hAll[((n - 1) * hidden)..], n, draft, h1);
        rows.Add((float[])draft.Clone());
        rows.Add((float[])h1.Clone());
        var h2 = new float[hidden];
        spec.DraftStep(ArgMax(draft), h1, n + 1, draft, h2);
        rows.Add((float[])draft.Clone());
        rows.Add((float[])h2.Clone());

        // The trunk takes `next`; the head skips it (a gap: no trunk rows), then folds the replay
        // of `after` into the draft that follows `third`.
        var hNext = new float[hidden];
        spec.SpecForward(new[] { next }, hNext, logits, false);
        spec.DraftCatchUp(new[] { next }, null, n);
        int after = ArgMax(logits);
        var hAfter = new float[hidden];
        spec.SpecForward(new[] { after }, hAfter, logits, false);
        int third = ArgMax(logits);
        var h3 = new float[hidden];
        spec.DraftCatchUpAndStep(new[] { after, third }, hNext.Concat(hAfter).ToArray(), n + 1, draft, h3);
        rows.Add((float[])draft.Clone());
        rows.Add(h3);
        return rows;
    }

    /// <summary>An image span (media rows over its placeholders, a (T, H, W) table whose rotary positions
    /// differ from the KV indices) and the decode after it (KV index + delta): the graph's multi-axis
    /// rotary, the per-op host rotation and the managed reference agree.</summary>
    [Fact]
    public void ImageSpanPositionsAgreeAcrossPaths()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense");
        List<float[]> reference;
        using (var managed = Create(path))
            reference = MediaScript(managed, out _);
        using var model = Create(path, backend: TestGates.PinnedGgmlBackend);
        List<float[]> graph = MediaScript(model, out int delta);
        Assert.Equal(-4, delta);   // 8 rows take 4 positions
        model.ResetKVCache();
        model.GraphLoweringEnabled = false;
        List<float[]> perOp = MediaScript(model, out _);
        double tolerance = RoundsActivations(TestGates.PinnedGgmlBackend) ? 2.5e-2 : 5e-3;
        for (int i = 0; i < reference.Count; i++)
        {
            Assert.True(RelL2(graph[i], reference[i]) <= tolerance, $"graph row {i}: relL2 {RelL2(graph[i], reference[i]):E2}");
            Assert.True(RelL2(perOp[i], reference[i]) <= tolerance, $"per-op row {i}: relL2 {RelL2(perOp[i], reference[i]):E2}");
        }
        // The positions matter: the same rows at scalar positions answer differently.
        model.ResetKVCache();
        model.GraphLoweringEnabled = true;
        float[] scalar = model.ForwardRefill(MediaPrompt);
        Assert.True(RelL2(scalar, graph[0]) > 10 * tolerance, "the position table changed nothing");
    }

    /// <summary>The same image span through Qwen35Model (its fused graphs run ggml's M-RoPE).</summary>
    [Fact]
    public void ImageSpanMatchesTheFamilyImplementation()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense");
        List<float[]> legacy;
        using (var family = new Qwen35Model(path, TestGates.PinnedGgmlBackend))
            legacy = MediaScript(family, out _);
        using var runtime = Create(path, backend: TestGates.PinnedGgmlBackend);
        List<float[]> actual = MediaScript(runtime, out _);
        double tolerance = RoundsActivations(TestGates.PinnedGgmlBackend) ? 2.5e-2 : 5e-3;
        for (int i = 0; i < legacy.Count; i++)
            Assert.True(RelL2(actual[i], legacy[i]) <= tolerance, $"row {i}: relL2 {RelL2(actual[i], legacy[i]):E2} from Qwen35Model");
    }

    [MlxFact]
    public void MlxImageSpanMatchesTheManagedReference()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense");
        List<float[]> reference;
        using (var managed = Create(path))
            reference = MediaScript(managed, out _);
        using var model = Create(path, backend: BackendType.Mlx);
        List<float[]> actual = MediaScript(model, out _);
        for (int i = 0; i < reference.Count; i++)
            Assert.True(RelL2(actual[i], reference[i]) <= 5e-3, $"mlx row {i}: relL2 {RelL2(actual[i], reference[i]):E2}");
    }

    private static readonly int[] MediaPrompt = Enumerable.Range(0, 24).Select(i => 40 + (i * 7) % 60).ToArray();

    /// <summary>Rows 5..12 are a 2 x 4 image span at base 5 (T 5, H 5-6, W 5-8); the text after it
    /// resumes at 9; three decode steps follow.</summary>
    private static List<float[]> MediaScript(ModelBase model, out int delta)
    {
        int hidden = model.Config.HiddenSize, rows = MediaPrompt.Length;
        var table = new int[3 * rows];
        for (int i = 0; i < rows; i++)
        {
            if (i >= 5 && i < 13)
            {
                table[3 * i] = 5;
                table[3 * i + 1] = 5 + (i - 5) / 4;
                table[3 * i + 2] = 5 + (i - 5) % 4;
            }
            else
            {
                int p = i < 5 ? i : 9 + (i - 13);
                table[3 * i] = table[3 * i + 1] = table[3 * i + 2] = p;
            }
        }
        var media = new float[8 * hidden];
        var rng = new Random(7);
        for (int j = 0; j < media.Length; j++) media[j] = (float)(rng.NextDouble() - 0.5) * 0.4f;
        var embeddings = new Tensor(new TensorSharp.Cpu.CpuAllocator(BlasEnum.DotNet), DType.Float32, 8, hidden);
        embeddings.SetElementsAsFloat(media);

        ((TensorSharp.Models.Architecture.IMRoPEPositionSink)model).SetMRoPEPositions(table);
        ((TensorSharp.Models.Architecture.IVisionCapableModel)model).SetVisionEmbeddings(embeddings, 5);
        var result = new List<float[]> { (float[])model.ForwardRefill(MediaPrompt).Clone() };
        delta = (model as Qwen35RuntimeModel)?.ActiveRopeDelta ?? 0;
        foreach (int token in new[] { 71, 72, 73 })
            result.Add((float[])model.Forward(new[] { token }).Clone());
        return result;
    }

    /// <summary>Two whole-layer stages on two GGML devices run one graph per stage: the same logits as
    /// one device (graph and per-op), speculation and the MTP draft layer on the last stage, and
    /// checkpoints across the stages' devices.</summary>
    [GgmlTheory(BackendType.GgmlCuda, Devices = 2)]
    [InlineData("dense")]
    [InlineData("moe")]
    public void StageGraphsMatchOneDevice(string variant)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(variant);
        List<float[]> single;
        using (var one = Create(path, backend: BackendType.GgmlCuda))
            single = Script(one);
        using var staged = Create(path, 1, 2, BackendType.GgmlCuda);
        Assert.Equal(2, staged.Mesh.Stages);
        List<float[]> graph = Script(staged);
        Assert.True(staged.GraphRuns > 0 && staged.GraphReplays > 0, "the stages did not run as replayed graphs");
        staged.ResetKVCache();
        staged.GraphLoweringEnabled = false;
        List<float[]> perOp = Script(staged);
        for (int i = 0; i < single.Count; i++)
        {
            Assert.True(RelL2(graph[i], single[i]) <= 1e-3, $"stage graphs row {i}: relL2 {RelL2(graph[i], single[i]):E2}");
            Assert.True(RelL2(perOp[i], single[i]) <= 2.5e-2, $"per-op stages row {i}: relL2 {RelL2(perOp[i], single[i]):E2}");
        }
        SpeculativeRuntimeChecks.AssertMatchesPlainGreedy(() => Create(path, 1, 2, BackendType.GgmlCuda), $"qwen35 {variant} stages", tieTolerance: 1e-2);
        CheckpointChecks.AssertCheckpointsContinueExactly(() => Create(path, 1, 2, BackendType.GgmlCuda), 1e-5, $"qwen35 {variant} stages", deviceCopies: true);
    }

    [GgmlFact(BackendType.GgmlCuda)]
    public void MtpDraftsOnTheLastStage()
    {
        if (TestGates.GgmlDevicesSkip(BackendType.GgmlCuda, 2) != null) return;
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense-nextn");
        SpeculativeRuntimeChecks.AssertDraftHeadMatchesPlainGreedy(() => Create(path, 1, 2, BackendType.GgmlCuda), "qwen35 nextn stages", tieTolerance: 1e-2);
    }

    /// <summary>A shared-prefix checkpoint (attention rows and Gated DeltaNet state), cloned or written
    /// to a file and read back, continues exactly as a fresh run.</summary>
    [Theory]
    [InlineData("dense")]
    [InlineData("moe")]
    public void CheckpointsContinueExactly(string variant)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(variant);
        CheckpointChecks.AssertCheckpointsContinueExactly(() => Create(path, backend: TestGates.PinnedGgmlBackend), 1e-5, $"qwen35 {variant}", deviceCopies: true);
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

    private static int ArgMax(float[] logits)
    {
        int best = 0;
        for (int v = 1; v < logits.Length; v++) if (logits[v] > logits[best]) best = v;
        return best;
    }
}
