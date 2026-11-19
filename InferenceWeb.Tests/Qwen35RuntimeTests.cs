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
    private readonly Xunit.Abstractions.ITestOutputHelper _output;

    public Qwen35RuntimeTests(Xunit.Abstractions.ITestOutputHelper output)
    {
        _output = output;
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
        // The 0.8B's layout: the head reads the token embedding.
        "dense-tied" => new Qwen35DenseSyntheticModelBuilder { Layers = 8 }.Write(Path.Combine(_dir, "q35-dense-tied.gguf")),
        "dense-nextn" => new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true, NextN = true }.Write(Path.Combine(_dir, "q35-dense-nextn.gguf")),
        "dense-mixed" => new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true, GateProjectionType = SyntheticGguf.GgmlType.F32 }
            .Write(Path.Combine(_dir, "q35-dense-mixed.gguf")),
        // Bonsai2's PRISM layout: rotated PQ2_0 projections, head and embedding, grouped value heads (3 per key head).
        "dense-prism" => new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true, SsmVHeads = 6, Ff = 512, Prism = true }
            .Write(Path.Combine(_dir, "q35-dense-prism.gguf")),
        // BF16 weights: no activation quantization, so the comparison is tight.
        "moe" => new Qwen35MoeSyntheticModelBuilder { Layers = 8, UniformWeightType = SyntheticGguf.GgmlType.BF16 }.Write(Path.Combine(_dir, "q35-moe.gguf")),
        // The quantized mix (IQ3_S / IQ4_XS experts, Q8_0 projections) quantizes activations, so
        // a rounding flip upstream moves outputs by ~1e-2 in either implementation.
        "moe-quantized" => new Qwen35MoeSyntheticModelBuilder { Layers = 8 }.Write(Path.Combine(_dir, "q35-moe-q.gguf")),
        // Eight routes a token: the route-tail skip's rank 7 is one of them.
        "moe-k8" => new Qwen35MoeSyntheticModelBuilder { Layers = 4, TopK = 8, UniformWeightType = SyntheticGguf.GgmlType.BF16 }
            .Write(Path.Combine(_dir, "q35-moe-k8.gguf")),
        // Two KV heads, so tensor parallelism can split the attention layers.
        "moe-tp" => new Qwen35MoeSyntheticModelBuilder { Layers = 8, UniformWeightType = SyntheticGguf.GgmlType.BF16, KvHeads = 2 }
            .Write(Path.Combine(_dir, "q35-moe-tp.gguf")),
        _ => throw new ArgumentOutOfRangeException(nameof(variant)),
    };

    private static Qwen35RuntimeModel Create(string path, int tp = 1, int stages = 1, BackendType backend = BackendType.Cpu)
        => new(path, backend, tp, tp > 1 ? new SimulatedDeviceGroup(tp * stages) : null, stages);

    /// <summary>A lone CUDA prompt whose full step does not fit the device releases finished requests' retained slots, the
    /// oldest first and only until the step fits, each reported to the prefix cache as reclaimed (kept, they narrowed every
    /// later prompt's steps on a 16 GB card).</summary>
    [CudaFact]
    public void ALonePromptReleasesTheOldestRetainedSlotsItNeeds()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F16, true);
        using var model = Create(Fixture("dense"), backend: BackendType.Cuda);
        var sink = new InferenceWeb.Tests.PrefixCache.Fakes.RecordingPayloadSink();
        model.AttachPrefixCache(sink);
        var unified = (IUnifiedBatchModel)model;
        string[] keys = { "pc:test:1", "pc:test:2", "pc:test:3" };
        for (int i = 0; i < keys.Length; i++)
        {
            string id = $"done{i}";
            unified.EnsureSequenceSlot(id, 64);
            Assert.True(unified.TryForwardUnified(new[] { id }, new[] { Enumerable.Range(0, 12).Select(t => 30 + i + t).ToArray() }, new[] { 0 },
                new float[1][]));
            Assert.True(model.RetainSequenceCacheAs(id, keys[i]));
        }
        // The first lone prompt sizes its steps from the device; then the device has room only with one slot left retained.
        model.PreferredSoloPrefillTokens(0, 1000);
        model.PromptFreeBytesForTests = () => model.RetainedPayloadKeys.Count > 1 ? 0 : long.MaxValue;
        model.PreferredSoloPrefillTokens(0, 100_000);
        Assert.Equal(new[] { keys[2] }, model.RetainedPayloadKeys);
        Assert.Equal(new[] { (keys[0], TensorSharp.Runtime.Scheduling.PrefixCache.InvalidationReason.NativeSlotReclaimed), (keys[1], TensorSharp.Runtime.Scheduling.PrefixCache.InvalidationReason.NativeSlotReclaimed) },
            sink.Reports);
    }

    /// <summary>The factory's route: the shared runtime serves every checkpoint on every backend and placement (the
    /// direct-CUDA engine that once took dense NVFP4 checkpoints on one GPU was removed).</summary>
    [Theory]
    [InlineData("dense")]
    [InlineData("moe")]
    public void TheFactoryCreatesTheRuntime(string variant)
    {
        using ModelBase model = ModelBase.Create(Fixture(variant), BackendType.Cpu);
        Assert.IsType<Qwen35RuntimeModel>(model);
    }

    /// <summary>A golden of the deleted Qwen35Model (Fixtures/Qwen35Legacy).</summary>
    private static List<float[]> LegacyGolden(string name)
    {
        using Stream stream = typeof(Qwen35RuntimeTests).Assembly.GetManifestResourceStream(
            $"InferenceWeb.Tests.Fixtures.Qwen35Legacy.{name}.json") ?? throw new FileNotFoundException(name);
        return System.Text.Json.JsonSerializer.Deserialize<List<float[]>>(stream)!;
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

    /// <summary>Backends whose matmuls round activations (ggml_cpu to BF16; ggml_cuda and ggml_vulkan to
    /// BF16 / Q8_1, with different kernels for one row than for several): measured on ggml_cuda (RTX PRO
    /// 6000, 2026-11-10) up to 7.4e-3 against the managed reference and 5.2e-3 graph vs per-op on these
    /// fixtures, ggml_vulkan (A40, 2026-11-12) 6.3e-3 graph vs per-op, ggml_cpu up to 1.5e-2.</summary>
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

    [Theory]
    [InlineData("dense", 1e-4)]
    [InlineData("dense-gqa", 1e-4)]
    [InlineData("moe", 1e-4)]
    // Measured 2026-11-10 (M5 Pro): 7.9e-3; the legacy implementation's own prefill and decode
    // disagree by 2.5e-3 on this fixture, and ggml vs the managed kernels by 5e-2. The managed kernels' sums follow the
    // runner's vector width: 3.03e-2 on some linux-x64 CI runners (and an RTX 2000 Ada VM's host), under 3e-2 on others.
    [InlineData("moe-quantized", 4e-2)]
    public void MatchesTheFamilyImplementation(string variant, double tolerance)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(variant);
        List<float[]> legacy = LegacyGolden($"script_{variant.Replace('-', '_')}_cpu");
        using var runtime = Create(path);
        List<float[]> actual = Script(runtime);
        Assert.Equal(legacy.Count, actual.Count);
        for (int i = 0; i < legacy.Count; i++)
        {
            double rel = RelL2(actual[i], legacy[i]);
            Assert.True(rel <= tolerance, $"{variant} row {i}: relL2 {rel:E2} from Qwen35Model's");
        }
    }

    /// <summary>Bonsai2's PRISM rotations on the pinned GGML backend: the runtime's step graphs read every rotated
    /// product's input rotated (the GGML graph builder's weight_product, Q/K/V and gate/up sharing one rotation; the
    /// grouped value heads permuted before ssm_out's) and embed through the inverse transform, as Qwen35Model's fused
    /// graphs did (its rows on ggml_cpu are the golden).</summary>
    [Fact]
    public void PrismRotationsMatchTheFamilyGolden()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        BackendType backend = TestGates.PinnedGgmlBackend;
        string path = Fixture("dense-prism");
        List<float[]> legacy = LegacyGolden("script_dense_prism_ggmlcpu");
        using var runtime = Create(path, backend: backend);
        List<float[]> actual = Script(runtime);
        Assert.True(runtime.GraphRuns > 0, "the steps did not run as graphs");
        // Measured 2026-11-12 (M5 Pro) against Qwen35Model on the same backend: ggml_metal 1.1e-5 at most, ggml_cpu
        // 2.0e-2 (its BF16-rounded activations, as PerOpTolerance bounds); the golden is ggml_cpu's, so every backend
        // sits within that rounding of it, plus its own where it rounds activations too (ggml_cuda on an RTX A5000:
        // 2.52e-2). ggml_cpu on ARM64 no longer rounds them (BF16 products over F32 activations,
        // ggml_ops_cpu_products.cpp: LFM3's BF16 fixture moved from 1.1-2.0e-2 to 1.3-2.2e-6 off the managed reference),
        // so it sits the golden's own rounding away (1.8-3.0e-2 per row on an M5 Pro, 2026-11-17). With the rotations
        // left out the first row is 1.2 away.
        double tolerance = backend is BackendType.GgmlCuda or BackendType.GgmlVulkan or BackendType.GgmlCpu ? 3.5e-2 : 2.5e-2;
        double[] rel = actual.Take(legacy.Count).Select((row, i) => RelL2(row, legacy[i])).ToArray();
        _output.WriteLine($"{backend}: relL2 per row from Qwen35Model {string.Join(" ", rel.Select(r => r.ToString("E2")))}");
        for (int i = 0; i < legacy.Count; i++)
            Assert.True(rel[i] <= tolerance, $"row {i}: relL2 {rel[i]:E2} from Qwen35Model");
    }

    /// <summary>Off GGML no path applies the rotations: the load says so instead of answering unrotated.</summary>
    [Fact]
    public void PrismRotationsAreRefusedOffGgml()
    {
        string path = Fixture("dense-prism");
        var error = Assert.Throws<NotSupportedException>(() => Create(path));
        Assert.Contains("GGML", error.Message);
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
    public void HostExpertsInTheGraphMatchTheAccelerator(string variant)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        BackendType backend = TestGates.PinnedGgmlBackend;
        string path = Fixture(variant);
        List<float[]> device, host, tiered;
        MoeCpuOffloadConfig.Reset();
        try
        {
            using (var model = Create(path, backend: backend)) device = Script(model);
            MoeCpuOffloadConfig.SetAllLayers();
            using (var model = Create(path, backend: backend))
            {
                host = Script(model);
                Assert.True(model.GraphRuns > 0, "the offloaded steps did not run as graphs");
            }
            // A tier holding every expert (fixed slots): every route runs on the accelerator at the pause.
            using (var model = Create(path, backend: backend))
            {
                Assert.True(model.RegisterExpertTiersForTests(int.MaxValue, 0) > 0, "no layer took a tier");
                long held = TensorSharp.GGML.GgmlBasicOps.MoeOffloadCounters()[0];
                tiered = Script(model);
                Assert.True(TensorSharp.GGML.GgmlBasicOps.MoeOffloadCounters()[0] > held, "the tier held no route: every one ran on the host");
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
            // BF16 experts: ggml-cuda rounds the activations to BF16 (1.4e-2 measured on an RTX PRO 6000). Quantized ones on
            // ggml_cpu: the prompt rows are the same bits and the decode rows carry the activations' rounding through the
            // recurrent state (3.4e-2 measured on linux-arm64, under 3e-2 on x64).
            double tolerance = variant == "moe" ? 2e-2 : 4e-2;
            Assert.True(host.Select((r, i) => RelL2(r, device[i])).All(r => r <= tolerance),
                $"{variant} on {backend}: relL2 from the accelerator's experts: {Rows(host, device)}");
        }
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
    public void PlacementMatchesOneDevice(int tp, int stages) => AssertPlacementMatchesOneDevice("dense", tp, stages);

    /// <summary>On per-rank graphs an untied head is split by vocabulary (each rank scores its rows, the host joins
    /// them), and the logits are one device's; a head tied to the embedding stays whole on rank 0.</summary>
    [Fact]
    public void RankGraphsSplitAnUntiedHeadByVocabulary()
    {
        using (var untied = Create(Fixture("dense"), tp: 2))
            Assert.True(untied.HeadSplitByVocabulary);
        using (var tied = Create(Fixture("dense-tied"), tp: 2))
            Assert.False(tied.HeadSplitByVocabulary);
        AssertPlacementMatchesOneDevice("dense", 2, 1);
        AssertPlacementMatchesOneDevice("dense-tied", 2, 1);
    }

    /// <summary>Routed experts under tensor parallelism: each rank's graph runs its slice of every expert (sharded by
    /// width like a dense feed-forward) and the partials meet in the all-reduce.</summary>
    [Fact]
    public void RoutedExpertsUnderTensorParallelismMatchOneDevice() => AssertPlacementMatchesOneDevice("moe-tp", 2, 1);

    /// <summary>Per-rank graphs on direct CUDA, two ranks on one GPU (each its own allocator and stream): one KV head
    /// per rank (the dense fixture's two), and routed experts sharded by width.</summary>
    [CudaTheory]
    [InlineData("dense", KvCacheDtype.F32)]
    [InlineData("dense", KvCacheDtype.F16)]
    [InlineData("dense-tied", KvCacheDtype.F16)]
    [InlineData("moe-tp", KvCacheDtype.F32)]
    public void DirectCudaRankGraphsMatchOneDevice(string variant, KvCacheDtype kv)
    {
        KvCacheDtypeConfig.RestoreForTests(kv, true);
        string path = Fixture(variant);
        List<float[]> reference;
        using (var single = Create(path, backend: BackendType.Cuda))
            reference = Script(single);
        var group = new OneGpuCudaGroup(2);
        using var placed = new Qwen35RuntimeModel(path, BackendType.Cuda, 2, group, 1);
        List<float[]> actual = Script(placed);
        Assert.True(group.AllReduceCount > 0 && placed.GraphRuns > 0, $"{variant}: no per-rank graph ran");
        for (int i = 0; i < reference.Count; i++)
        {
            double rel = RelL2(actual[i], reference[i]);
            // BF16 experts run on BF16-rounded activations here (the tensor-core products), as on the
            // activation-rounding backends: 8.2e-3 measured on an A40, within 5e-3 on an RTX 2000 Ada.
            Assert.True(rel <= (variant == "moe-tp" ? 2.5e-2 : 5e-3), $"{variant} {kv} row {i}: relL2 {rel:E2}");
        }
    }

    /// <summary>Per-rank graphs on two GPUs (the P2P all-reduce of --tp 2) and whole-layer stages on two GPUs
    /// (--layer-split 2), on the CUDA executor: every row answers as one GPU's (the placements the direct engine never
    /// served; the 27B NVFP4 on two 16 GB cards needs one of them).</summary>
    [MultiCudaTheory(2)]
    [InlineData("dense", 2, 1)]
    [InlineData("moe-tp", 2, 1)]
    [InlineData("dense", 1, 2)]
    [InlineData("moe", 1, 2)]
    public void DirectCudaPlacementsOnTwoGpusMatchOneGpu(string variant, int tp, int stages)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(variant);
        List<float[]> reference;
        using (var single = Create(path, backend: BackendType.Cuda))
            reference = Script(single);
        using var group = tp > 1 ? new TensorSharp.Cuda.TensorParallelGroup(tp) : null;
        using var placed = new Qwen35RuntimeModel(path, BackendType.Cuda, tp, group, stages);
        Assert.Equal(tp, placed.Mesh.TensorParallel);
        Assert.Equal(stages, placed.Mesh.Stages);
        List<float[]> actual = Script(placed);
        Assert.True(placed.GraphRuns > 0, $"{variant}: no graph ran");
        for (int i = 0; i < reference.Count; i++)
        {
            double rel = RelL2(actual[i], reference[i]);
            // As DirectCudaRankGraphsMatchOneDevice: BF16 experts sharded by width round apart; stages are exact.
            Assert.True(rel <= (variant == "moe-tp" ? 2.5e-2 : 5e-3), $"{variant} tp={tp} stages={stages} row {i}: relL2 {rel:E2}");
        }
    }

    /// <summary>Speculation on per-rank graphs: each rank's verify keeps its recurrent states' window, and the commit runs
    /// per rank on its own share of the heads; the NextN draft layer runs whole (unsharded) on rank 0.</summary>
    [Fact]
    public void SpeculationOnRankGraphsMatchesPlainGreedy()
    {
        SpeculativeRuntimeChecks.AssertMatchesPlainGreedy(() => Create(Fixture("dense"), tp: 2), "qwen35 dense tp2");
        SpeculativeRuntimeChecks.AssertDraftHeadMatchesPlainGreedy(() => Create(Fixture("dense-nextn"), tp: 2), "qwen35 nextn tp2");
    }

    /// <summary>Per-rank graphs on ggml_cuda over two GPUs: each rank's program compiles to its device's ggml graph, and
    /// the ranks' graphs run together, pausing at every all-reduce while the devices sum the partial (exact F32).</summary>
    [GgmlTheory(BackendType.GgmlCuda, Devices = 2)]
    [InlineData("dense", KvCacheDtype.F32)]
    [InlineData("dense", KvCacheDtype.F16)]
    [InlineData("dense-tied", KvCacheDtype.F16)]
    [InlineData("moe-tp", KvCacheDtype.F32)]
    public void GgmlCudaRankGraphsMatchOneDevice(string variant, KvCacheDtype kv)
    {
        KvCacheDtypeConfig.RestoreForTests(kv, true);
        string path = Fixture(variant);
        List<float[]> reference;
        using (var single = Create(path, backend: BackendType.GgmlCuda))
            reference = Script(single);
        using var placed = new Qwen35RuntimeModel(path, BackendType.GgmlCuda, 2, null, 1);
        Assert.Equal(2, placed.Mesh.TensorParallel);
        List<float[]> actual = Script(placed);
        Assert.True(placed.GraphRuns > 0 && placed.GraphReplays > 0, $"{variant}: the per-rank graphs did not run and replay");
        for (int i = 0; i < reference.Count; i++)
        {
            double rel = RelL2(actual[i], reference[i]);
            Assert.True(rel <= (variant == "moe-tp" ? 2.5e-2 : 5e-3), $"{variant} {kv} row {i}: relL2 {rel:E2}");
        }
    }

    /// <summary>On per-rank graphs a shared-prefix checkpoint and its clone are copied rank by rank on each rank's device
    /// (each rank's share of the caches and states), and the clone continues exactly as the request did. A NextN draft
    /// layer runs whole on rank 0, so the other ranks hold no tensor of its cache (Qwen3.8-27B with its MTP layer failed
    /// every request under --tp 2 at its first checkpoint).</summary>
    [GgmlTheory(BackendType.GgmlCuda, Devices = 2)]
    [InlineData("dense")]
    [InlineData("dense-nextn")]
    [InlineData("moe-tp")]
    public void RankGraphCheckpointsCopyOnEachRanksDevice(string variant)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(variant);
        CheckpointChecks.AssertCheckpointsContinueExactly(() => new Qwen35RuntimeModel(path, BackendType.GgmlCuda, 2, null, 1),
            1e-5, $"qwen35 {variant} tp2 ranks", deviceCopies: true);
    }

    /// <summary>Two tensor-parallel ranks on CUDA device 0, each with its own allocator and stream; the all-reduce sums
    /// through the host.</summary>
    private sealed class OneGpuCudaGroup : ITensorParallelGroup
    {
        private readonly TensorSharp.Cuda.CudaAllocator[] _allocators;

        public OneGpuCudaGroup(int degree) =>
            _allocators = Enumerable.Range(0, degree).Select(_ => new TensorSharp.Cuda.CudaAllocator(0)).ToArray();

        public int Degree => _allocators.Length;
        public bool IsActive => Degree > 1;
        public int GlobalDegree => Degree;
        public int GlobalRankOffset => 0;
        public int NodeCount => 1;
        public int AllReduceCount { get; private set; }
        public IAllocator GetAllocator(int rank) => _allocators[rank];

        public void AllReduce(Tensor[] tensors)
        {
            AllReduceCount++;
            foreach (var allocator in _allocators) allocator.Synchronize();
            int n = (int)tensors[0].Storage.ElementCount;
            float[] sum = tensors[0].GetElementsAsFloat(n);
            for (int r = 1; r < tensors.Length; r++)
            {
                float[] part = tensors[r].GetElementsAsFloat(n);
                for (int i = 0; i < n; i++) sum[i] += part[i];
            }
            foreach (var tensor in tensors)
            {
                tensor.SetElementsAsFloat(sum);
                tensor.EnsureDeviceCurrent();
            }
            foreach (var allocator in _allocators) allocator.Synchronize();
        }

        public void Synchronize()
        {
            foreach (var allocator in _allocators) allocator.Synchronize();
        }

        public void Barrier() { }
        public void BroadcastControl(int op, int[] payload) => throw new NotSupportedException();
        public (int op, int[] payload) ReceiveControl() => throw new NotSupportedException();

        public void Dispose()
        {
            foreach (var allocator in _allocators) allocator.Dispose();
        }
    }

    private void AssertPlacementMatchesOneDevice(string variant, int tp, int stages)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(variant);
        List<float[]> reference;
        using (var single = Create(path))
            reference = Script(single);
        var group = tp > 1 ? new SimulatedDeviceGroup(tp * stages) : null;
        using var placed = new Qwen35RuntimeModel(path, BackendType.Cpu, tp, group, stages);
        Assert.Equal(tp, placed.Mesh.TensorParallel);
        Assert.Equal(stages, placed.Mesh.Stages);
        List<float[]> actual = Script(placed);
        // One stage of tensor parallelism runs per-rank graphs: their partials meet in the graphs' all-reduces.
        if (tp > 1 && stages == 1) Assert.True(group.AllReduceCount > 0 && placed.GraphRuns > 0, $"{variant} tp={tp}: no per-rank graph ran");
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

    /// <summary>The same on the tensor executors whose device is not the host: an offloaded layer's routes the tier does
    /// not hold run on the host (ggml's CPU kernels), the tier's on the device, and a prompt chunk wider than the host
    /// kernel streams every selected expert to the device for the step. One implementation for every backend.</summary>
    [MlxTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void HostExpertsOnMlxMatchDeviceExperts(bool tier) => AssertHostExpertsMatchDevice(BackendType.Mlx, tier);

    [CudaTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void HostExpertsOnDirectCudaMatchDeviceExperts(bool tier) => AssertHostExpertsMatchDevice(BackendType.Cuda, tier);

    /// <summary>The route-tail skip on the tensor executors' expert host (Strata's, on by default): in a step the host
    /// computes, a host expert every row routes at rank 7 or lower is left out; a streamed prompt chunk computes every route.
    /// The same answers as the exact sums for the prompt, slightly different ones after.</summary>
    [MlxFact]
    public void RouteTailSkipOnMlxLeavesOutLowRankHostRoutes() => AssertRouteTailSkip(BackendType.Mlx);

    [CudaFact]
    public void RouteTailSkipOnDirectCudaLeavesOutLowRankHostRoutes() => AssertRouteTailSkip(BackendType.Cuda);

    private void AssertRouteTailSkip(BackendType backend)
    {
        using var kv = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        using var cpuMoe = MoeCpuOffloadConfig.AllLayersForCurrentFlowInTests();
        string path = Fixture("moe-k8");
        (List<float[]> Rows, long HostRoutes) Run(int rank)
        {
            using var model = Create(path, backend: backend);
            model.RouteTailSkipForTests = (rank, 100);
            List<float[]> rows = Script(model);
            return (rows, model.ExpertHostCountsForTests.HostRoutes);
        }
        (List<float[]> exact, long exactRoutes) = Run(0);
        (List<float[]> skipped, long skippedRoutes) = Run(7);
        _output.WriteLine($"{backend}: host routes {exactRoutes} exact, {skippedRoutes} with the tail skip");
        Assert.True(skippedRoutes < exactRoutes && skippedRoutes * 8 >= exactRoutes * 7 - 8,
            $"host routes {skippedRoutes} with the skip, {exactRoutes} without");
        Assert.Equal(exact[0], skipped[0]);   // the streamed prompt chunk: every route
        double worst = Enumerable.Range(1, exact.Count - 1).Max(i => RelL2(skipped[i], exact[i]));
        Assert.True(worst > 0 && worst < 5e-2, $"rows after the prompt moved {worst:E2}");
    }

    /// <summary>A decode step's misses partly on the device (the Qwen3.8 engines' hybrid misses): up to four of a layer's
    /// host experts a call routes are staged over the link from their page-locked, device-mapped pages into device slots and
    /// computed there, only the rest rung to the host: far fewer host routes, the prompt (a streamed chunk) the same. The decode
    /// rows are not compared: the host's routes round the activations to Q8_K, the device's to q8_1, and on this random model
    /// that flips a top-k (as in <see cref="QuantizedHostExpertsOnDirectCudaMatchDeviceExperts"/>); CudaMoeStagingKernelTests
    /// checks the staging byte for byte.</summary>
    [CudaFact]
    public void HostMissesStagedOnDirectCudaAnswerAsTheHost()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        using var cpuMoe = MoeCpuOffloadConfig.AllLayersForCurrentFlowInTests();
        string path = Fixture("moe-quantized");
        (List<float[]> Rows, long HostRoutes) Run(int gpuMisses)
        {
            using var model = Create(path, backend: BackendType.Cuda);
            model.RouteTailSkipForTests = (0, 100);
            model.WaitHostWarmForTests();
            ((TensorSharp.Cuda.CudaGraphExecutor)model.GraphExecutorForTests).GpuMissesForTests = gpuMisses;
            List<float[]> rows = Script(model);
            return (rows, model.ExpertHostCountsForTests.HostRoutes);
        }
        (List<float[]> host, long hostRoutes) = Run(0);
        (List<float[]> staged, long stagedRoutes) = Run(TensorSharp.Cuda.CudaGraphExecutor.DefaultGpuMisses);
        _output.WriteLine($"host routes {hostRoutes} with the host computing every miss, {stagedRoutes} with misses staged");
        Assert.True(stagedRoutes * 2 < hostRoutes, $"host routes {stagedRoutes} with staging, {hostRoutes} without: nothing was staged");
        Assert.Equal(host[0], staged[0]);   // the streamed prompt chunk
        Assert.All(staged, row => Assert.All(row, v => Assert.True(float.IsFinite(v))));
    }

    /// <summary>
    /// Adaptive tier trades overlapped with the decode on direct CUDA (CudaGraphExecutor.TierTrades: a round's copies on a
    /// stream of their own, the tables moved a step boundary apart) give the logits blocking trades give, and no trades, bit
    /// for bit - in steps run as they come and in recorded steps, whose kernels read the tables when they run. Every miss of
    /// a decode row is computed on the device here (from the tier's slot, a staging copy or a miss slot: the same kernel over
    /// the same bytes), so only a step that read bytes still in flight could differ. Each step first routes one host expert
    /// of every layer often, so every round trades; the overlapped rounds' trades are committed after steps ran beside them.
    /// </summary>
    [CudaTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void OverlappedTierTradesGiveTheBlockingTradesLogitsOnDirectCuda(bool recorded)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F16, true);
        using var cpuMoe = MoeCpuOffloadConfig.AllLayersForCurrentFlowInTests();
        string path = Fixture("moe-quantized");
        (List<float[]> Rows, long Overlapped, long Blocking, long Beside, long RecordedBeside, int Recordings, long HostRoutes) Run(
            TensorSharp.Cuda.CudaGraphExecutor.TierTrades trades)
        {
            using var model = Create(path, backend: BackendType.Cuda);
            model.RouteTailSkipForTests = (0, 100);
            model.DecodeCaptureDisabled = !recorded;
            model.WaitHostWarmForTests();
            var executor = (TensorSharp.Cuda.CudaGraphExecutor)model.GraphExecutorForTests;
            // Every miss of a decode row staged into a device slot: no route is left to the host's kernels.
            executor.GpuMissesForTests = 16;
            executor.UploadsTierForTests = true;
            executor.TierTradesForTests = trades;
            int experts = model.ExpertCountForTests;
            int[] layers = Enumerable.Range(0, model.Config.NumLayers).Where(l => model.ExpertHostForTests(l) != null).ToArray();
            Assert.True(layers.Length > 0, "the fixture has no offloaded MoE layer");
            foreach (int l in layers) model.SetExpertTierForTests(l, Enumerable.Range(0, experts / 2).ToArray());
            TensorSharp.Graph.IGraphExpertHost host = model.ExpertHostForTests(layers[0]);
            void RouteAHostExpertOften()
            {
                foreach (int l in layers)
                {
                    int e = Enumerable.Range(0, experts).First(x => !host.IsResident(l, x));
                    host.Record(l, Enumerable.Repeat(e, 40).ToArray());
                }
            }
            var rows = new List<float[]>();
            RouteAHostExpertOften();
            rows.Add((float[])model.ForwardRefill(Enumerable.Range(0, 17).Select(i => 40 + (i * 11) % 150).ToArray()).Clone());
            for (int i = 0; i < 24; i++)
            {
                RouteAHostExpertOften();
                rows.Add((float[])model.Forward(new[] { 60 + (i * 7) % 90 }).Clone());
            }
            return (rows, executor.OverlappedTrades, executor.BlockingTrades, executor.StepsBesideTrades, executor.RecordedStepsBesideTrades,
                model.DecodeRecordings, model.ExpertHostCountsForTests.HostRoutes);
        }
        var off = Run(TensorSharp.Cuda.CudaGraphExecutor.TierTrades.Off);
        var blocking = Run(TensorSharp.Cuda.CudaGraphExecutor.TierTrades.Blocking);
        var overlapped = Run(TensorSharp.Cuda.CudaGraphExecutor.TierTrades.Overlapped);
        _output.WriteLine($"recorded={recorded}: blocking trades {blocking.Blocking}; overlapped trades {overlapped.Overlapped} "
            + $"(+{overlapped.Blocking} blocking), steps beside them {overlapped.Beside} ({overlapped.RecordedBeside} recorded launches), "
            + $"recordings {overlapped.Recordings}");
        Assert.Equal(0, off.Overlapped + off.Blocking);
        // The decode's cost: no route ever reached the host's kernels - no expert left the device's reach at any step (a
        // lock replaced while serving takes its pages' experts off the device; the rounds only add locks).
        Assert.Equal(0, off.HostRoutes);
        Assert.Equal(0, blocking.HostRoutes);
        Assert.Equal(0, overlapped.HostRoutes);
        Assert.True(blocking.Blocking > 0 && blocking.Overlapped == 0, $"blocking rounds: {blocking.Blocking} blocking, {blocking.Overlapped} overlapped trades");
        Assert.True(overlapped.Overlapped > 0 && overlapped.Blocking == 0,
            $"overlapped rounds: {overlapped.Overlapped} overlapped, {overlapped.Blocking} blocking trades (no staging to borrow?)");
        Assert.True(overlapped.Beside > 0, "no step ran while a round's copies were in flight");
        if (recorded)
            Assert.True(overlapped.Recordings > 0 && overlapped.RecordedBeside > 0,
                $"recordings {overlapped.Recordings}, recorded launches beside a round {overlapped.RecordedBeside}");
        for (int i = 0; i < off.Rows.Count; i++)
        {
            Assert.True(blocking.Rows[i].AsSpan().SequenceEqual(off.Rows[i]), $"row {i}: blocking trades moved it {RelL2(blocking.Rows[i], off.Rows[i]):E2}");
            Assert.True(overlapped.Rows[i].AsSpan().SequenceEqual(off.Rows[i]), $"row {i}: overlapped trades moved it {RelL2(overlapped.Rows[i], off.Rows[i]):E2}");
        }
    }

    /// <summary>
    /// Adaptive trades beside every way a step's misses go on direct CUDA - rung to the host (no miss slots), or staged into
    /// miss slots with a short chunk's surplus rung - over three conversations of a long prompt chunk (staged through the arenas
    /// a round borrows), a 7-row continuation the doorbell takes and recorded decode steps: every step finishes. Run on a worker
    /// thread with a deadline, so a step left waiting on the device (a ring nobody answers, a stream waiting on a copy that
    /// never comes) fails here instead of hanging the run.
    /// </summary>
    [CudaTheory]
    [InlineData(0)]
    [InlineData(TensorSharp.Cuda.CudaGraphExecutor.DefaultGpuMisses)]
    public void TierTradesBesideRungAndStagedMissesFinishOnDirectCuda(int gpuMisses)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F16, true);
        using var cpuMoe = MoeCpuOffloadConfig.AllLayersForCurrentFlowInTests();
        string path = Fixture("moe-quantized");
        (long Trades, long Beside, long HostRoutes, long Doorbells, int Rows) Run()
        {
            using var model = Create(path, backend: BackendType.Cuda);
            model.WaitHostWarmForTests();
            var executor = (TensorSharp.Cuda.CudaGraphExecutor)model.GraphExecutorForTests;
            executor.GpuMissesForTests = gpuMisses;
            executor.UploadsTierForTests = true;
            int experts = model.ExpertCountForTests;
            int[] layers = Enumerable.Range(0, model.Config.NumLayers).Where(l => model.ExpertHostForTests(l) != null).ToArray();
            foreach (int l in layers) model.SetExpertTierForTests(l, Enumerable.Range(0, experts / 2).ToArray());
            TensorSharp.Graph.IGraphExpertHost host = model.ExpertHostForTests(layers[0]);
            void RouteAHostExpertOften()
            {
                foreach (int l in layers)
                    host.Record(l, Enumerable.Repeat(Enumerable.Range(0, experts).First(x => !host.IsResident(l, x)), 40).ToArray());
            }
            var rows = new List<float[]>();
            for (int turn = 0; turn < 3; turn++)
            {
                if (turn > 0) model.ResetKVCache();
                RouteAHostExpertOften();
                rows.Add((float[])model.ForwardRefill(Enumerable.Range(0, 17 + turn).Select(i => 40 + (i * 13 + turn) % 150).ToArray()).Clone());
                RouteAHostExpertOften();
                rows.Add((float[])model.Forward(Enumerable.Range(0, 7).Select(i => 50 + (i * 5 + turn) % 90).ToArray()).Clone());
                for (int i = 0; i < 9; i++)
                {
                    RouteAHostExpertOften();
                    rows.Add((float[])model.Forward(new[] { 60 + (i * 7 + turn) % 90 }).Clone());
                }
            }
            Assert.All(rows, row => Assert.All(row, v => Assert.True(float.IsFinite(v))));
            return (executor.OverlappedTrades + executor.BlockingTrades, executor.StepsBesideTrades, model.ExpertHostCountsForTests.HostRoutes,
                executor.DoorbellCalls, rows.Count);
        }
        var run = Task.Run(Run);
        Assert.True(run.Wait(TimeSpan.FromMinutes(5)),
            $"gpuMisses={gpuMisses}: the conversations did not finish in 5 minutes (a step waits on the device for something nobody provides)");
        var r = run.Result;
        _output.WriteLine($"gpuMisses={gpuMisses}: {r.Rows} steps, {r.Trades} trades, {r.Beside} steps beside a round, "
            + $"{r.HostRoutes} host routes, {r.Doorbells} doorbell calls");
        Assert.True(r.Trades > 0, "no trade was made");
        Assert.True(r.Doorbells > 0, "no step rang the doorbell");
        if (gpuMisses == 0) Assert.True(r.HostRoutes > 0, "no route was rung to the host");
    }

    /// <summary>
    /// Where the host could not pin the host experts' packed copies (dropped here before the first step), their file ranges are
    /// page-locked instead, and a trade's outgoing expert is page-locked and mapped before it leaves the tier on direct CUDA
    /// (the load locks the host experts it planned, not the tier's: here the tier's locks are released before the first step):
    /// a worker locks the outgoing experts and the round's copies begin a counted step later. Nothing is unlocked while the
    /// model serves, so no expert is ever taken off the device: with every decode miss staged in a miss slot no route reaches
    /// the host's kernels, the locked total within the stacks grows only by what the worker locked, and a prompt chunk after
    /// the rounds stages every layer's host experts on the device (none reads its routes back).
    /// </summary>
    [CudaFact]
    public void TradedOutExpertsAreLockedBeforeTheyLeaveTheTierWithoutPackedCopiesOnDirectCuda()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F16, true);
        using var cpuMoe = MoeCpuOffloadConfig.AllLayersForCurrentFlowInTests();
        using var model = Create(Fixture("moe-quantized"), backend: BackendType.Cuda);
        model.RouteTailSkipForTests = (0, 100);
        model.WaitHostWarmForTests();
        var executor = (TensorSharp.Cuda.CudaGraphExecutor)model.GraphExecutorForTests;
        executor.GpuMissesForTests = 16;
        executor.UploadsTierForTests = true;
        executor.UnlocksTierAtLoadForTests = true;
        model.DropHostExpertCopiesForTests();
        Assert.Equal(0, model.HostExpertCopyBytesForTests);
        int experts = model.ExpertCountForTests;
        int[] layers = Enumerable.Range(0, model.Config.NumLayers).Where(l => model.ExpertHostForTests(l) != null).ToArray();
        Assert.True(layers.Length > 0, "the fixture has no offloaded MoE layer");
        foreach (int l in layers) model.SetExpertTierForTests(l, Enumerable.Range(0, experts / 2).ToArray());
        TensorSharp.Graph.IGraphExpertHost host = model.ExpertHostForTests(layers[0]);
        void RouteAHostExpertOften()
        {
            foreach (int l in layers)
                host.Record(l, Enumerable.Repeat(Enumerable.Range(0, experts).First(x => !host.IsResident(l, x)), 40).ToArray());
        }
        RouteAHostExpertOften();
        model.ForwardRefill(Enumerable.Range(0, 17).Select(i => 40 + (i * 11) % 150).ToArray());
        long lockedAtStart = executor.LockedExpertBytesForTests(), lockedByWorkerAtStart = executor.TradeLockedBytes;
        for (int i = 0; i < 20; i++)
        {
            RouteAHostExpertOften();
            model.Forward(new[] { 60 + (i * 7) % 90 });
        }
        executor.FinishTradesForTests();
        long locked = executor.LockedExpertBytesForTests(), lockedByWorker = executor.TradeLockedBytes - lockedByWorkerAtStart;
        long trades = executor.OverlappedTrades + executor.BlockingTrades, hostRoutes = model.ExpertHostCountsForTests.HostRoutes;
        _output.WriteLine($"{trades} trades ({executor.UnlockableTrades} dropped), {executor.LockingRounds} rounds waited for locks; locked "
            + $"{lockedAtStart} -> {locked} bytes in the stacks, {lockedByWorker} by the worker; host routes {hostRoutes}");
        Assert.True(trades > 0, "no trade was made");
        Assert.True(executor.LockingRounds > 0 && lockedByWorker > 0, "no round locked its outgoing experts first");
        Assert.Equal(0, executor.UnlockableTrades);
        Assert.Equal(0, hostRoutes);
        Assert.True(locked >= lockedAtStart && locked - lockedAtStart <= lockedByWorker,
            $"locked {lockedAtStart} -> {locked} bytes in the stacks, the worker {lockedByWorker}");

        // A prompt chunk continuing the conversation (wider than the doorbell takes): every layer's host experts on the device.
        long device = executor.DeviceStagedLayers, planned = executor.HostPlannedLayers;
        float[] logits = model.Forward(Enumerable.Range(0, 17).Select(i => 30 + (i * 13) % 150).ToArray());
        _output.WriteLine($"the prompt chunk: {executor.DeviceStagedLayers - device} layers staged on the device, "
            + $"{executor.HostPlannedLayers - planned} host-planned");
        Assert.Equal(0, executor.HostPlannedLayers - planned);
        Assert.Equal(layers.Length, executor.DeviceStagedLayers - device);
        Assert.All(logits, v => Assert.True(float.IsFinite(v)));
    }

    /// <summary>
    /// The host experts packed in pinned host memory on direct CUDA (the direct engines' layout, StandardDecoderModel's
    /// TryPackHostExperts; packed again here for the tier the test names, as a planned tier's load packs them): forced trades
    /// move each outgoing expert into the incoming one's place, so no host memory is registered while the model serves - no
    /// lock round, nothing page-locked by the executor or by the load - and every host expert stays device-readable: with
    /// every decode miss staged in a miss slot no route reaches the host's kernels, and a prompt chunk after the rounds stages
    /// every MoE layer on the device. Overlapped and blocking trades give the logits no trades give, bit for bit.
    /// </summary>
    [CudaFact]
    public void PackedHostExpertsTradeWithoutRegisteringHostMemoryOnDirectCuda()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F16, true);
        using var cpuMoe = MoeCpuOffloadConfig.AllLayersForCurrentFlowInTests();
        string path = Fixture("moe-quantized");
        (List<float[]> Rows, long Trades) Run(TensorSharp.Cuda.CudaGraphExecutor.TierTrades trades)
        {
            using var model = Create(path, backend: BackendType.Cuda);
            model.RouteTailSkipForTests = (0, 100);
            model.WaitHostWarmForTests();
            var executor = (TensorSharp.Cuda.CudaGraphExecutor)model.GraphExecutorForTests;
            executor.GpuMissesForTests = 16;
            executor.UploadsTierForTests = true;
            executor.TierTradesForTests = trades;
            int experts = model.ExpertCountForTests;
            int[] layers = Enumerable.Range(0, model.Config.NumLayers).Where(l => model.ExpertHostForTests(l) != null).ToArray();
            Assert.True(layers.Length > 0, "the fixture has no offloaded MoE layer");
            foreach (int l in layers) model.SetExpertTierForTests(l, Enumerable.Range(0, experts / 2).ToArray());
            Assert.True(model.RepackHostExpertsForTests(), "the host experts were not packed");
            Assert.False(model.HostRangesLockedForTests, "the load page-locked the experts' file ranges beside the packed copies");
            TensorSharp.Graph.IGraphExpertHost host = model.ExpertHostForTests(layers[0]);
            void RouteAHostExpertOften()
            {
                foreach (int l in layers)
                    host.Record(l, Enumerable.Repeat(Enumerable.Range(0, experts).First(x => !host.IsResident(l, x)), 40).ToArray());
            }
            var rows = new List<float[]>();
            RouteAHostExpertOften();
            rows.Add((float[])model.ForwardRefill(Enumerable.Range(0, 17).Select(i => 40 + (i * 11) % 150).ToArray()).Clone());
            for (int i = 0; i < 20; i++)
            {
                RouteAHostExpertOften();
                rows.Add((float[])model.Forward(new[] { 60 + (i * 7) % 90 }).Clone());
            }
            executor.FinishTradesForTests();
            long tradesMade = executor.OverlappedTrades + executor.BlockingTrades;
            _output.WriteLine($"{trades}: {tradesMade} trades; lock rounds {executor.LockingRounds}, executor locks {executor.TradePinnedBytesForTests} "
                + $"bytes; unmapped host experts {executor.UnmappedHostExpertsForTests()}; host routes {model.ExpertHostCountsForTests.HostRoutes}");
            Assert.Equal(0, executor.LockingRounds);
            Assert.Equal(0, executor.TradeLockedBytes);
            Assert.Equal(0, executor.TradePinnedBytesForTests);
            Assert.False(model.HostRangesLockedForTests);
            Assert.Equal(0, executor.UnmappedHostExpertsForTests());
            Assert.Equal(0, model.ExpertHostCountsForTests.HostRoutes);

            // A prompt chunk continuing the conversation (wider than the doorbell takes): every layer's host experts on the device.
            long device = executor.DeviceStagedLayers, planned = executor.HostPlannedLayers;
            rows.Add((float[])model.Forward(Enumerable.Range(0, 17).Select(i => 30 + (i * 13) % 150).ToArray()).Clone());
            Assert.Equal(0, executor.HostPlannedLayers - planned);
            Assert.Equal(layers.Length, executor.DeviceStagedLayers - device);
            return (rows, tradesMade);
        }
        var off = Run(TensorSharp.Cuda.CudaGraphExecutor.TierTrades.Off);
        var blocking = Run(TensorSharp.Cuda.CudaGraphExecutor.TierTrades.Blocking);
        var overlapped = Run(TensorSharp.Cuda.CudaGraphExecutor.TierTrades.Overlapped);
        Assert.Equal(0, off.Trades);
        Assert.True(blocking.Trades > 0 && overlapped.Trades > 0, $"trades: blocking {blocking.Trades}, overlapped {overlapped.Trades}");
        for (int i = 0; i < off.Rows.Count; i++)
        {
            Assert.All(off.Rows[i], v => Assert.True(float.IsFinite(v)));
            Assert.True(blocking.Rows[i].AsSpan().SequenceEqual(off.Rows[i]), $"row {i}: blocking trades moved it {RelL2(blocking.Rows[i], off.Rows[i]):E2}");
            Assert.True(overlapped.Rows[i].AsSpan().SequenceEqual(off.Rows[i]), $"row {i}: overlapped trades moved it {RelL2(overlapped.Rows[i], off.Rows[i]):E2}");
        }
    }

    /// <summary>A trade gives the outgoing expert the incoming one's place (HostExpertCopies.Move): the incoming expert has none
    /// from then on (a late request for it reads the mapping), a place the outgoing expert held before is freed, and the
    /// places - the pinned bytes - never change.</summary>
    [Fact]
    public void HostExpertCopiesGiveTheOutgoingExpertTheIncomingOnesPlace()
    {
        var stacks = new[] { (IntPtr)0x10000, (IntPtr)0x20000, (IntPtr)0x30000 };
        var devices = new[] { (IntPtr)0x110000, (IntPtr)0x120000, (IntPtr)0x130000 };
        // Experts 1 and 3 on the host (places 0 and 1), 0 and 2 in the tier.
        var copies = new TensorSharp.Graph.HostExpertCopies(stacks, devices, new long[] { 256, 512, 1024 }, new[] { -1, 0, -1, 1 });
        Assert.Equal(2, copies.Places);
        Assert.Equal(new[] { -1, 0, -1, 1 }, Enumerable.Range(0, 4).Select(copies.PlaceOf).ToArray());
        Assert.Equal((IntPtr)(0x20000 + 512), copies.Bytes(1, 3));
        Assert.Equal((IntPtr)(0x130000 + 1024), copies.DeviceBytes(2, 3));
        Assert.Equal(IntPtr.Zero, copies.Bytes(0, 0));
        // Expert 3 comes into the tier, expert 0 leaves it: 0 takes place 1.
        copies.Move(into: 3, outOf: 0);
        Assert.Equal(new[] { 1, 0, -1, -1 }, Enumerable.Range(0, 4).Select(copies.PlaceOf).ToArray());
        Assert.Equal((IntPtr)(0x10000 + 256), copies.Bytes(0, 0));
        Assert.Throws<InvalidOperationException>(() => copies.Move(into: 2, outOf: 3));   // 2 has no place to give
        Assert.Equal(2, copies.Places);
    }

    /// <summary>Process memory standing in for pinned host memory: a backend whose host alone reads the packed copies.</summary>
    private sealed class ProcessMemoryHostAllocator : TensorSharp.Graph.IMappedHostAllocator
    {
        public unsafe IntPtr AllocateMappedHost(long bytes, out IntPtr devicePointer)
        {
            IntPtr p = (IntPtr)System.Runtime.InteropServices.NativeMemory.AlignedAlloc((nuint)bytes, 4096);
            devicePointer = p;
            return p;
        }

        public unsafe void FreeMappedHost(IntPtr host) => System.Runtime.InteropServices.NativeMemory.AlignedFree((void*)host);
    }

    /// <summary>The host's kernels compute host experts from packed copies (here in process memory, on MLX, where the host
    /// computes every expert outside the tier) with the sums they compute from the model file's mapping, bit for bit: the ids
    /// index the copies' places, the same bytes in the same order.</summary>
    [MlxFact]
    public void HostExpertsFromPackedCopiesMatchTheMappingOnMlx()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        using var cpuMoe = MoeCpuOffloadConfig.AllLayersForCurrentFlowInTests();
        string path = Fixture("moe-quantized");
        (List<float[]> Rows, long HostRoutes) Run(bool packed)
        {
            using var model = Create(path, backend: BackendType.Mlx);
            model.RouteTailSkipForTests = (0, 100);
            int experts = model.ExpertCountForTests;
            foreach (int l in Enumerable.Range(0, model.Config.NumLayers).Where(l => model.ExpertHostForTests(l) != null))
                model.SetExpertTierForTests(l, Enumerable.Range(0, experts / 2).ToArray());
            if (packed)
            {
                Assert.True(model.PackHostExpertsForTests(new ProcessMemoryHostAllocator()), "the host experts were not packed");
                Assert.True(model.HostExpertCopyBytesForTests > 0);
            }
            List<float[]> rows = Script(model);
            return (rows, model.ExpertHostCountsForTests.HostRoutes);
        }
        var mapped = Run(packed: false);
        var copies = Run(packed: true);
        _output.WriteLine($"host routes: {copies.HostRoutes} from the packed copies, {mapped.HostRoutes} from the mapping");
        Assert.True(copies.HostRoutes > 0, "no route was the host's");
        Assert.Equal(mapped.HostRoutes, copies.HostRoutes);
        for (int i = 0; i < mapped.Rows.Count; i++)
            Assert.True(copies.Rows[i].AsSpan().SequenceEqual(mapped.Rows[i]), $"row {i}: {RelL2(copies.Rows[i], mapped.Rows[i]):E2} from the mapping's sums");
    }

    /// <summary>The host's kernels compute host experts from the packed copies (the ids index the copies' places) with the
    /// sums they compute from the model file's mapping, bit for bit - the same bytes in the same order - and the device stages
    /// a prompt's host experts from either alike: no miss slots, so every miss of a decode row is the host's.</summary>
    [CudaFact]
    public void HostExpertsFromPackedCopiesMatchTheMappingOnDirectCuda()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F16, true);
        using var cpuMoe = MoeCpuOffloadConfig.AllLayersForCurrentFlowInTests();
        string path = Fixture("moe-quantized");
        (List<float[]> Rows, long HostRoutes, long Packed) Run(bool packed)
        {
            using var model = Create(path, backend: BackendType.Cuda);
            model.RouteTailSkipForTests = (0, 100);
            model.WaitHostWarmForTests();
            var executor = (TensorSharp.Cuda.CudaGraphExecutor)model.GraphExecutorForTests;
            executor.GpuMissesForTests = 0;
            executor.UploadsTierForTests = true;
            executor.TierTradesForTests = TensorSharp.Cuda.CudaGraphExecutor.TierTrades.Off;
            int experts = model.ExpertCountForTests;
            foreach (int l in Enumerable.Range(0, model.Config.NumLayers).Where(l => model.ExpertHostForTests(l) != null))
                model.SetExpertTierForTests(l, Enumerable.Range(0, experts / 2).ToArray());
            if (packed) Assert.True(model.RepackHostExpertsForTests(), "the host experts were not packed");
            else model.DropHostExpertCopiesForTests();
            List<float[]> rows = Script(model);
            return (rows, model.ExpertHostCountsForTests.HostRoutes, model.HostExpertCopyBytesForTests);
        }
        long pinnedBefore = TensorSharp.Cuda.CudaHostPins.PinnedBytes;
        var mapped = Run(packed: false);
        // The file ranges that model locked went with it: a lock outliving its mapping pins the old pages, and the next
        // mapping at those addresses uploaded them in place of its own (later tests' first models computed garbage).
        Assert.Equal(pinnedBefore, TensorSharp.Cuda.CudaHostPins.PinnedBytes);
        var copies = Run(packed: true);
        _output.WriteLine($"host routes: {copies.HostRoutes} from the packed copies ({copies.Packed} bytes), {mapped.HostRoutes} from the mapping");
        Assert.Equal(0, mapped.Packed);
        Assert.True(copies.Packed > 0 && copies.HostRoutes > 0, $"packed {copies.Packed} bytes, host routes {copies.HostRoutes}");
        Assert.Equal(mapped.HostRoutes, copies.HostRoutes);
        for (int i = 0; i < mapped.Rows.Count; i++)
            Assert.True(copies.Rows[i].AsSpan().SequenceEqual(mapped.Rows[i]), $"row {i}: {RelL2(copies.Rows[i], mapped.Rows[i]):E2} from the mapping's sums");
    }

    /// <summary>A short chunk of up to 16 rows (a continuation's few new tokens) with host experts rings the doorbell on direct
    /// CUDA, as a decode row does, rather than reading every layer's routes back for the host to compute while the device
    /// waits (a 13-row continuation of Qwen3.8 Flash Next took a second that way); a longer chunk streams.</summary>
    [CudaFact]
    public void AShortChunkOfHostExpertsRingsTheDoorbellOnDirectCuda()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        using var cpuMoe = MoeCpuOffloadConfig.AllLayersForCurrentFlowInTests();
        using var model = Create(Fixture("moe-quantized"), backend: BackendType.Cuda);
        model.WaitHostWarmForTests();
        var executor = (TensorSharp.Cuda.CudaGraphExecutor)model.GraphExecutorForTests;
        var unified = (IUnifiedBatchModel)model;
        unified.EnsureSequenceSlot("a", 128);
        var logits = new float[1][];
        int[] prompt = Enumerable.Range(0, 24).Select(i => 40 + i % 7).ToArray();
        Assert.True(unified.TryForwardUnified(new[] { "a" }, new[] { prompt }, new[] { 0 }, logits));
        long rung = executor.DoorbellCalls, fallen = executor.HostExpertFallbacks;
        int[] chunk = Enumerable.Range(0, 13).Select(i => 50 + i % 5).ToArray();
        Assert.True(unified.TryForwardUnified(new[] { "a" }, new[] { chunk }, new[] { prompt.Length }, logits));
        Assert.True(executor.DoorbellCalls > rung, "the 13-row chunk's host experts did not ring the doorbell");
        Assert.Equal(fallen, executor.HostExpertFallbacks);
        Assert.All(logits[0], v => Assert.True(float.IsFinite(v)));
    }

    /// <summary>
    /// A prompt chunk wider than the doorbell takes and shorter than the staged-ahead path's keeps its routes on the device
    /// on direct CUDA (CudaGraphExecutor.PromptStage: the host experts it selects planned into staging slots and copied
    /// there by kernels from their mapped pages, the grouped MoE kernels over the layer's tables, the tables restored):
    /// every MoE layer of the chunk served so and none read back, with the logits the host-planned path gives (the routes
    /// read back, the same kernels over the same bytes at other staging addresses) bit for bit, the prompt's and the
    /// decode row's after it. The adaptive tier learns the prompt's routes as it did from the host's ids: counted on the
    /// device and handed over at the next step's start, they leave the same recent counts, each route at the chunk's weight.
    /// With the shorter chunks' arena smaller than a layer's host experts (<paramref name="shortArenaSlots"/>: 2 of the 8,
    /// as Qwen3.8 Flash Next's arena, sized from its first layer's experts, was for most of its layers on an A40), every
    /// layer stages through the long chunks' arena halves instead, still none read back, the same logits.
    /// </summary>
    [CudaTheory]
    [InlineData(17, 0)]
    [InlineData(40, 0)]
    [InlineData(100, 0)]
    [InlineData(17, 2)]
    [InlineData(100, 2)]
    public void PromptChunksStagedOnTheDeviceMatchTheHostPlannedPathOnDirectCuda(int rows, int shortArenaSlots)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        using var cpuMoe = MoeCpuOffloadConfig.AllLayersForCurrentFlowInTests();
        string path = Fixture("moe-quantized");
        int[] prompt = Enumerable.Range(0, rows).Select(i => 40 + (i * 11) % 150).ToArray();
        // The reference: the routes read back, through a short-chunk arena that holds a layer's host experts (the grouped kernels).
        var planned = RunPromptStage(path, prompt, hostPlanned: true);
        var device = RunPromptStage(path, prompt, hostPlanned: false, shortArenaSlots);
        _output.WriteLine($"{rows} rows, short-chunk arena slots {(shortArenaSlots > 0 ? shortArenaSlots.ToString() : "sized")}: {device.Device} layers "
            + $"staged on the device ({device.Halves} through the halves), {device.HostPlanned} host-planned; the host-planned run: "
            + $"{planned.HostPlanned} layers");
        Assert.True(device.Device > 0, "no prompt layer kept its routes on the device");
        Assert.Equal(0, device.HostPlanned);
        Assert.Equal(0, device.Fallbacks);
        Assert.Equal(shortArenaSlots > 0 ? device.Device : 0, device.Halves);
        Assert.True(planned.HostPlanned > 0 && planned.Device == 0, $"host-planned run: {planned.HostPlanned} host-planned, {planned.Device} device layers");
        Assert.Equal(planned.HostPlanned, device.Device);
        Assert.True(device.Prompt.AsSpan().SequenceEqual(planned.Prompt), $"prompt row: {RelL2(device.Prompt, planned.Prompt):E2} from the host-planned path");
        Assert.True(device.Next.AsSpan().SequenceEqual(planned.Next), $"decode row: {RelL2(device.Next, planned.Next):E2} from the host-planned path");
        for (int i = 0; i < device.Recent.Length; i++)
        {
            Assert.NotNull(planned.Recent[i]);
            Assert.NotNull(device.Recent[i]);
            Assert.True(device.Recent[i].Sum() > 0, $"MoE layer {i}: no prompt route was recorded");
            Assert.Equal(planned.Recent[i], device.Recent[i]);
        }
    }

    /// <summary>
    /// Where neither arena holds a layer's host experts (the shorter chunks' arena 2 slots of the 8, the long chunks' never
    /// allocated) a prompt chunk's layers read their routes back and stage the selected experts a batch at a time, as before
    /// the device path: every layer host-planned, none on the device, finite logits near the device path's (the batches run
    /// each expert on its own kernels, not the grouped ones, so the bits differ: bound as the fixture's other kernel
    /// comparisons are).
    /// </summary>
    [CudaFact]
    public void PromptChunksFallBackToTheHostPlannedPathWithoutStagingRoomOnDirectCuda()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        using var cpuMoe = MoeCpuOffloadConfig.AllLayersForCurrentFlowInTests();
        string path = Fixture("moe-quantized");
        int[] prompt = Enumerable.Range(0, 40).Select(i => 40 + (i * 11) % 150).ToArray();
        var device = RunPromptStage(path, prompt, hostPlanned: false);
        var cramped = RunPromptStage(path, prompt, hostPlanned: false, shortArenaSlots: 2, noHalves: true);
        double rel = RelL2(cramped.Prompt, device.Prompt);
        _output.WriteLine($"without staging room: {cramped.HostPlanned} host-planned layers, {cramped.Device} on the device; "
            + $"prompt row {rel:E2} from the device path");
        Assert.True(cramped.HostPlanned > 0, "no layer fell back to the host-planned path");
        Assert.Equal(0, cramped.Device);
        Assert.Equal(device.Device, cramped.HostPlanned);
        Assert.All(cramped.Prompt, v => Assert.True(float.IsFinite(v)));
        Assert.All(cramped.Next, v => Assert.True(float.IsFinite(v)));
        Assert.True(rel <= 5e-2, $"prompt row: relL2 {rel:E2} from the device path");
    }

    /// <summary>A prompt and a decode step on direct CUDA, every MoE layer offloaded with half its experts in the tier, and
    /// what the executor's prompt staging did: the rows, its counters and each MoE layer's recent route counts (read after
    /// the decode step's start handed the prompt's device-counted routes to the expert hosts). No adaptive round runs: a
    /// round decays the counts and moves the tier.</summary>
    private static (float[] Prompt, float[] Next, long Device, long Halves, long HostPlanned, long Fallbacks, float[][] Recent) RunPromptStage(
        string path, int[] prompt, bool hostPlanned, int shortArenaSlots = 0, bool noHalves = false)
    {
        using var model = Create(path, backend: BackendType.Cuda);
        model.RouteTailSkipForTests = (0, 100);
        model.WaitHostWarmForTests();
        var executor = (TensorSharp.Cuda.CudaGraphExecutor)model.GraphExecutorForTests;
        executor.HostPlannedStagingForTests = hostPlanned;
        executor.ShortChunkArenaSlotsForTests = shortArenaSlots;
        executor.PrefetchArenaRefusedForTests = noHalves;
        executor.UploadsTierForTests = true;
        executor.TierTradesForTests = TensorSharp.Cuda.CudaGraphExecutor.TierTrades.Off;
        int experts = model.ExpertCountForTests;
        int[] layers = Enumerable.Range(0, model.Config.NumLayers).Where(l => model.ExpertHostForTests(l) != null).ToArray();
        Assert.True(layers.Length > 0, "the fixture has no offloaded MoE layer");
        foreach (int l in layers) model.SetExpertTierForTests(l, Enumerable.Range(0, experts / 2).ToArray());
        float[] first = (float[])model.ForwardRefill(prompt).Clone();
        float[] next = (float[])model.Forward(new[] { 71 }).Clone();
        float[][] recent = layers.Select(l => model.ExpertRecentRoutesForTests(l)).ToArray();
        return (first, next, executor.DeviceStagedLayers, executor.DeviceStagedInHalves, executor.HostPlannedLayers, executor.HostExpertFallbacks,
            recent);
    }

    /// <summary>Quantized experts (IQ3_S / IQ4_XS): the prompt chunk's staged experts (on CUDA the grouped MoE kernels, F16
    /// operands and F32 accumulation) against the device's own kernels. Only the prompt row is judged: the decode rows' host
    /// routes round the activations to Q8_K as ggml's CPU backend does, against the device's unrounded ones, and on this
    /// random model a flipped top-k then moves a later row by 0.1-0.4 (MLX and CUDA alike), past any tolerance that would
    /// still catch a fault.</summary>
    [CudaTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void QuantizedHostExpertsOnDirectCudaMatchDeviceExperts(bool tier) => AssertHostExpertsMatchDevice(BackendType.Cuda, tier, "moe-quantized");

    [MlxTheory]
    [InlineData(false)]
    [InlineData(true)]
    public void QuantizedHostExpertsOnMlxMatchDeviceExperts(bool tier) => AssertHostExpertsMatchDevice(BackendType.Mlx, tier, "moe-quantized");

    /// <summary>A shared expert run beside the router and the routed experts on a second stream (a concurrent region) gives
    /// the logits it gives run in place, bit for bit: the same kernels on the same bytes, the streams joined where its output
    /// is read - prompts, decode steps and their recordings.</summary>
    [CudaFact]
    public void SharedExpertBesideTheRoutedExpertsGivesTheSameLogitsOnDirectCuda()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F16, true);
        string path = Fixture("moe-quantized");
        List<float[]> Run(bool inPlace)
        {
            using var model = Create(path, backend: BackendType.Cuda);
            model.ConcurrentRegionsDisabledForTests = inPlace;
            List<float[]> rows = Script(model);
            Assert.True(model.GraphReplays > 0, "the decode steps did not replay");
            return rows;
        }
        List<float[]> beside = Run(false), inPlace = Run(true);
        for (int i = 0; i < inPlace.Count; i++)
            Assert.True(beside[i].AsSpan().SequenceEqual(inPlace[i]), $"row {i}: {RelL2(beside[i], inPlace[i]):E2} from the in-place run");
    }

    /// <summary>A host-served layer's selection on the CUDA executor's keyed warp select (the selection resident layers'
    /// experts take) gives what its softmax / top-k / renormalize ops give, within the fixture's near-tie bound.</summary>
    [CudaFact]
    public void HostServedSelectionOnDirectCudaMatchesItsOps()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("moe-quantized");
        using var cpuMoe = MoeCpuOffloadConfig.AllLayersForCurrentFlowInTests();
        (List<float[]> Rows, long Regions) Run(bool ops)
        {
            using var model = Create(path, backend: BackendType.Cuda);
            model.MoeSelectRegionDisabledForTests = ops;
            List<float[]> rows = Script(model);
            return (rows, ((TensorSharp.Cuda.CudaGraphExecutor)model.GraphExecutorForTests).MoeSelectRegions);
        }
        var kernel = Run(ops: false);
        var ops = Run(ops: true);
        Assert.True(kernel.Regions > 0, "no selection ran as the region");
        Assert.Equal(0, ops.Regions);
        _output.WriteLine("selection region vs ops: " + string.Join(" ", kernel.Rows.Select((r, i) => RelL2(r, ops.Rows[i]).ToString("E2"))));
        // The fixture's random router has near-ties, where the two sums may pick another expert: rows then move by the
        // quantized checkpoint's ~1e-2 (AssertHostExpertsMatchDevice's bound for it).
        for (int i = 0; i < ops.Rows.Count; i++)
        {
            double rel = RelL2(kernel.Rows[i], ops.Rows[i]);
            Assert.True(rel <= 5e-2, $"row {i}: {rel:E2} from the ops' selection");
        }
    }

    /// <summary>A greedy unified step on the CUDA executor (each sequence's last row's top-1 picked on the device and read
    /// back as an id) picks what the logits step's host argmax picks: two sequences over a prompt and four decode steps.</summary>
    [CudaFact]
    public void GreedyUnifiedStepsPickTheLogitsArgmaxOnDirectCuda()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F16, true);
        string path = Fixture("moe-quantized");
        int[] Run(bool greedy)
        {
            using var model = Create(path, backend: BackendType.Cuda);
            var unified = Assert.IsAssignableFrom<TensorSharp.Runtime.Scheduling.IUnifiedBatchModel>(model);
            Assert.True(unified.SupportsUnifiedGreedyIds);
            string[] ids = { "a", "b" };
            foreach (string id in ids) unified.EnsureSequenceSlot(id, 64);
            int[][] tokens = { Enumerable.Range(0, 9).Select(i => 40 + i * 7).ToArray(), Enumerable.Range(0, 13).Select(i => 50 + i * 5).ToArray() };
            int[] starts = { 0, 0 };
            var picked = new List<int>();
            for (int step = 0; step < 5; step++)
            {
                var next = new int[2];
                if (greedy)
                    Assert.True(unified.TryForwardUnifiedGreedy(ids, tokens, starts, next));
                else
                {
                    var logits = new float[2][];
                    Assert.True(unified.TryForwardUnified(ids, tokens, starts, logits));
                    for (int s = 0; s < 2; s++)
                    {
                        int best = 0;
                        for (int v = 1; v < logits[s].Length; v++) if (logits[s][v] > logits[s][best]) best = v;
                        next[s] = best;
                    }
                }
                picked.AddRange(next);
                for (int s = 0; s < 2; s++) starts[s] += tokens[s].Length;
                tokens = next.Select(t => new[] { t }).ToArray();
            }
            return picked.ToArray();
        }
        Assert.Equal(Run(greedy: false), Run(greedy: true));
    }

    /// <summary>The staged-ahead path of a long prompt chunk (CudaGraphExecutor.PromptPrefetch: every host expert of the
    /// next MoE layer copied into the other arena half while a layer runs, the routes kept on the device), driven with
    /// the fixture's short prompt by lowering this executor's row threshold: the same answers as the device's experts.</summary>
    [CudaFact]
    public void PrefetchedHostExpertsOnDirectCudaMatchDeviceExperts()
        => AssertHostExpertsMatchDevice(BackendType.Cuda, true, "moe-quantized", prefetch: true);

    /// <summary>An adaptive round pairs a layer's hottest experts outside its tier with its coldest members, hottest with
    /// coldest, while the outside one was routed more than twice as often (and at least twice): with every member cold the
    /// two hot experts take the two lowest-numbered idle members' places; with the members warm only the expert routed past
    /// twice theirs trades; the budget keeps the largest gains; nothing trades on a layer whose outside experts are cold.</summary>
    [Fact]
    public void AdaptiveRoundsPairTheHottestOutsideWithTheColdestMembers()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        using var cpuMoe = MoeCpuOffloadConfig.AllLayersForCurrentFlowInTests();
        using var model = Create(Fixture("moe"), backend: BackendType.Cpu);
        int experts = model.ExpertCountForTests, half = experts / 2;
        int[] moe = Enumerable.Range(0, model.Config.NumLayers).Where(l => model.ExpertHostForTests(l) != null).Take(3).ToArray();
        Assert.True(moe.Length >= 2, "the fixture has too few offloaded MoE layers");
        foreach (int l in moe) model.SetExpertTierForTests(l, Enumerable.Range(0, half).ToArray());
        TensorSharp.Graph.IGraphExpertHost host = model.ExpertHostForTests(moe[0]);
        int[] Routes(params (int Expert, int Count)[] counts) => counts.SelectMany(c => Enumerable.Repeat(c.Expert, c.Count)).ToArray();
        // Layer A: two hot outside experts, every member cold.
        host.Record(moe[0], Routes((half + 1, 9), (half + 3, 5)));
        // Layer B: every member routed 3 times; one outside expert at 10 (> 2 * 3 + 1), one at 7 (not).
        host.Record(moe[1], Routes(Enumerable.Range(0, half).Select(e => (e, 3)).Concat(new[] { (half + 2, 10), (half + 4, 7) }).ToArray()));
        var swaps = host.ProposeSwaps(8);
        var a = swaps.Where(s => s.Layer == moe[0]).ToArray();
        var b = swaps.Where(s => s.Layer == moe[1]).ToArray();
        Assert.Equal(new[] { (moe[0], 0, half + 1), (moe[0], 1, half + 3) }, a.Select(s => (s.Layer, s.Out, s.In)).ToArray());
        Assert.Equal(new[] { (moe[1], 0, half + 2) }, b.Select(s => (s.Layer, s.Out, s.In)).ToArray());
        if (moe.Length > 2) Assert.DoesNotContain(swaps, s => s.Layer == moe[2]);
        // The budget keeps the largest gains: layer B's 10 - 3 before layer A's 5 - 0.
        host.Record(moe[0], Routes((half + 1, 18), (half + 3, 10)));
        host.Record(moe[1], Routes(Enumerable.Range(0, half).Select(e => (e, 6)).Concat(new[] { (half + 2, 40) }).ToArray()));
        var top = host.ProposeSwaps(1);
        Assert.Single(top);
        Assert.Equal((moe[1], 0, half + 2), (top[0].Layer, top[0].Out, top[0].In));
    }

    private void AssertHostExpertsMatchDevice(BackendType backend, bool tier, string variant = "moe", bool prefetch = false)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(variant);
        List<float[]> reference;
        using (var device = Create(path, backend: backend))
            reference = Script(device);
        using var cpuMoe = MoeCpuOffloadConfig.AllLayersForCurrentFlowInTests();
        using var offloaded = Create(path, backend: backend);
        if (tier)
            for (int l = 0; l < offloaded.Config.NumLayers; l++)
                offloaded.SetExpertTierForTests(l, Enumerable.Range(0, offloaded.ExpertCountForTests / 2).ToArray());
        var cudaExecutor = prefetch ? (TensorSharp.Cuda.CudaGraphExecutor)offloaded.GraphExecutorForTests : null;
        if (cudaExecutor != null)
        {
            cudaExecutor.PrefetchMinRows = 1;
            // The executor learns the MoE layers' order from a first request; the next prompt stages each layer ahead.
            Script(offloaded);
            offloaded.ResetKVCache();
        }
        List<float[]> actual = Script(offloaded);
        if (cudaExecutor != null)
        {
            _output.WriteLine($"staged-ahead layers {cudaExecutor.PrefetchedLayers}, copied while the previous layer ran {cudaExecutor.PrefetchedAhead}");
            Assert.True(cudaExecutor.PrefetchedLayers > 0 && cudaExecutor.PrefetchedAhead > 0,
                $"staged-ahead layers {cudaExecutor.PrefetchedLayers}, ahead {cudaExecutor.PrefetchedAhead}");
        }
        (long hostRoutes, long streamed) = offloaded.ExpertHostCountsForTests;
        Assert.True(offloaded.GraphRuns > 0, "the offloaded model did not run as graphs");
        Assert.True(hostRoutes > 0 && streamed > 0, $"host routes {hostRoutes}, streamed steps {streamed}");
        // The host's kernels round as ggml's CPU backend does (activations to Q8_K / BF16 against the device's).
        _output.WriteLine($"{backend} {variant} tier={tier}: " + string.Join(" ", reference.Select((r, i) => RelL2(actual[i], r).ToString("E2"))));
        for (int i = 0; i < (variant == "moe" ? reference.Count : 1); i++)
        {
            double rel = RelL2(actual[i], reference[i]);
            // Quantized activations upstream move a quantized checkpoint's rows by ~1e-2 in either implementation.
            Assert.True(rel <= (variant == "moe" ? 2.5e-2 : 5e-2), $"{backend} --cpu-moe {variant} tier={tier}, row {i}: relL2 {rel:E2}");
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
    [InlineData("moe-tp", 2, 1)]
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
    /// traced step's, for one request and for two in one step, across a decode window's bucket edge.</summary>
    [Theory]
    [InlineData("dense")]
    [InlineData("moe")]
    public void AdvancedDecodeStepsMatchTracedOnes(string variant)
        => AssertAdvancedDecodeMatchesTraced(variant, path => Create(path, backend: TestGates.PinnedGgmlBackend));

    /// <summary>The same on the managed tensor executor, whose steps stream: a decode step that could advance runs as
    /// the recorded program, so the next one advances instead of emitting every node again.</summary>
    [Theory]
    [InlineData("dense")]
    [InlineData("moe")]
    public void AdvancedDecodeStepsMatchTracedOnesOnTheManagedExecutor(string variant)
        => AssertAdvancedDecodeMatchesTraced(variant, path => Create(path, backend: BackendType.Cpu));

    /// <summary>The same on direct CUDA, where a lone request's advanced steps are recorded as one CUDA graph and
    /// replayed (attention length and cache write row read on the device): every logit is still the traced step's.</summary>
    [CudaTheory]
    [InlineData("dense")]
    [InlineData("moe")]
    public void AdvancedDecodeStepsMatchTracedOnesOnDirectCuda(string variant)
        => AssertAdvancedDecodeMatchesTraced(variant, path => Create(path, backend: BackendType.Cuda));

    /// <summary>A request's recorded decode steps park when it ends, and the pooled slot's next request replays them where
    /// its steps trace to the same plan (each window it reaches): the same prompt answers with the same logits, bit for
    /// bit, without running and recording those steps again.</summary>
    [CudaTheory]
    [InlineData("dense")]
    public void ParkedDecodeRecordingsServeThePooledSlotsNextRequestOnDirectCuda(string variant)
        => AssertParkedRecordingsServeTheNextRequest(variant, retain: false);

    /// <summary>The same when the first request's slot is kept (the prefix cache keeps finished slots): the next request runs
    /// on a new slot, and its steps replay the parked recordings moved to its caches and states.</summary>
    [CudaTheory]
    [InlineData("dense")]
    public void ParkedDecodeRecordingsMoveToANewSlotOnDirectCuda(string variant)
        => AssertParkedRecordingsServeTheNextRequest(variant, retain: true);

    /// <summary>
    /// On direct CUDA, whose recorded steps bind a slot's buffers, the prefix cache's retention of a finished request
    /// (<see cref="TensorSharp.Runtime.Scheduling.PrefixCache.IPrefixCacheModel.TryCaptureDonate"/>) copies a short slot into a
    /// pooled one (a spare made beside its recorded decode steps): the slot returns to the pool and the next request replays
    /// the parked recordings on it, none moved to a new slot, and the kept copy continues as the request did, bit for bit.
    /// </summary>
    [CudaFact]
    public void FinishedRequestsAreRetainedAsCopiesOnDirectCuda()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F16, true);
        using var model = Create(Fixture("dense"), backend: BackendType.Cuda);
        var unified = (IUnifiedBatchModel)model;
        var cache = (TensorSharp.Runtime.Scheduling.PrefixCache.IPrefixCacheModel)model;
        cache.AttachPrefixCache(new PayloadSink());
        int[] prompt = Enumerable.Range(0, 20).Select(i => 40 + i % 7).ToArray();
        var logits = new float[1][];
        (int Next, int Length) Decode(string id, int steps)
        {
            unified.EnsureSequenceSlot(id, 320);
            Assert.True(unified.TryForwardUnified(new[] { id }, new[] { prompt }, new[] { 0 }, logits));
            int next = ArgMax(logits[0]), length = prompt.Length;
            for (int t = 0; t < steps; t++)
            {
                Assert.True(unified.TryForwardUnified(new[] { id }, new[] { new[] { next } }, new[] { length }, logits));
                next = ArgMax(logits[0]);
                length++;
            }
            return (next, length);
        }

        var (token, kept) = Decode("a", 40);
        Assert.True(model.SlotPoolForTests.Pooled > 0, "no spare slot was made beside the recorded decode steps");
        Assert.True(cache.TryCaptureDonate("a", "kept", kept, out _));
        Assert.Equal(1, model.RetainedByCopy);
        Assert.True(unified.TryForwardUnified(new[] { "a" }, new[] { new[] { token } }, new[] { kept }, logits));
        float[] expected = (float[])logits[0].Clone();
        model.OnSequenceReleased("a");

        int adopted = model.RecordingsAdopted, relocated = model.RecordingsRelocated;
        Decode("b", 40);
        Assert.True(model.RecordingsAdopted > adopted, "the next request replayed none of the parked recordings");
        Assert.Equal(relocated, model.RecordingsRelocated);
        model.OnSequenceReleased("b");

        Assert.True(model.TryRebindRetainedCache("kept", "c"));
        Assert.True(unified.TryForwardUnified(new[] { "c" }, new[] { new[] { token } }, new[] { kept }, logits));
        Assert.True(expected.AsSpan().SequenceEqual(logits[0]),
            $"the kept copy continues {RelL2(logits[0], expected):E2} (relative L2) from the request it copied");
    }

    /// <summary>On direct CUDA, where the device has no room for a fresh slot, a new request takes the oldest retained slot that
    /// holds it (its payload reclaimed) rather than allocating, and answers as the finished request did. The device is full as
    /// the model reads it (DeviceRoomForTests, what making room measures; its answer is final) and as the takeover's own check
    /// reads it (SlotDeviceMemoryForTests). Making room gave up the parked recordings first (the cheapest loss), the finished
    /// request's among them, so the new request records its steps again: before, this test's device had room by the model's
    /// own reading and was full only by the takeover's, which took the slot with its recordings, a state no real device
    /// reaches.</summary>
    [CudaFact]
    public void ANewRequestReclaimsARetainedSlotWhenTheDeviceIsFullOnDirectCuda()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F16, true);
        using var model = Create(Fixture("dense"), backend: BackendType.Cuda);
        var unified = (IUnifiedBatchModel)model;
        int[] prompt = Enumerable.Range(0, 20).Select(i => 40 + i % 7).ToArray();
        List<float[]> Request(string id)
        {
            unified.EnsureSequenceSlot(id, 64);
            var logits = new float[1][];
            Assert.True(unified.TryForwardUnified(new[] { id }, new[] { prompt }, new[] { 0 }, logits));
            var trace = new List<float[]> { (float[])logits[0].Clone() };
            int next = ArgMax(logits[0]), length = prompt.Length;
            for (int t = 0; t < 12; t++)
            {
                Assert.True(unified.TryForwardUnified(new[] { id }, new[] { new[] { next } }, new[] { length }, logits));
                trace.Add((float[])logits[0].Clone());
                next = ArgMax(logits[0]);
                length++;
            }
            return trace;
        }
        List<float[]> first = Request("a");
        Assert.True(model.RetainSequenceCacheAs("a", "kept"));
        model.DeviceRoomForTests = () => (0, 1L << 34);
        model.SlotDeviceMemoryForTests = _ => (0, 1L << 34);
        List<float[]> second = Request("b");
        Assert.Equal(1, model.RetainedSlotsReclaimed);
        Assert.False(model.TryRebindRetainedCache("kept", "c"));
        for (int i = 0; i < first.Count; i++)
            Assert.True(first[i].AsSpan().SequenceEqual(second[i]), $"row {i} differs on the reclaimed slot");
    }

    /// <summary>On direct CUDA an idle engine's slices (<see cref="IUnifiedBatchModel.RunIdleSlice"/>) fill the slot pool to its
    /// cap while the device has room, and make none where it has none; a request then runs on a spare as on a fresh slot.</summary>
    [CudaFact]
    public void IdleSlicesFillTheSlotPoolOnDirectCuda()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F16, true);
        using var model = Create(Fixture("dense"), backend: BackendType.Cuda);
        var unified = (IUnifiedBatchModel)model;
        int[] prompt = Enumerable.Range(0, 20).Select(i => 40 + i % 7).ToArray();
        List<float[]> Request(string id)
        {
            unified.EnsureSequenceSlot(id, 64);
            var logits = new float[1][];
            Assert.True(unified.TryForwardUnified(new[] { id }, new[] { prompt }, new[] { 0 }, logits));
            var trace = new List<float[]> { (float[])logits[0].Clone() };
            int next = ArgMax(logits[0]), length = prompt.Length;
            for (int t = 0; t < 6; t++)
            {
                Assert.True(unified.TryForwardUnified(new[] { id }, new[] { new[] { next } }, new[] { length }, logits));
                trace.Add((float[])logits[0].Clone());
                next = ArgMax(logits[0]);
                length++;
            }
            return trace;
        }
        // A first request measures a slot's bytes; its slot goes back to the pool.
        List<float[]> first = Request("a");
        model.OnSequenceReleased("a");
        int made = model.SpareSlotsMade;
        model.SlotDeviceMemoryForTests = _ => (1L << 40, 1L << 40);
        int slices = 0;
        while (unified.RunIdleSlice() && slices < 100_000) slices++;
        var (pooled, cap) = model.SlotPoolForTests;
        Assert.Equal(cap, pooled);
        Assert.Equal(cap - 1, model.SpareSlotsMade - made);
        Assert.False(unified.RunIdleSlice());
        // A request takes a pooled slot; with no room for another, idle slices make none.
        List<float[]> second = Request("b");
        model.SlotDeviceMemoryForTests = _ => (0, 1L << 40);
        Assert.False(unified.RunIdleSlice());
        Assert.Equal(cap - 1, model.SlotPoolForTests.Pooled);
        for (int i = 0; i < first.Count; i++)
            Assert.True(first[i].AsSpan().SequenceEqual(second[i]), $"row {i} differs on the pooled slot");
        // Requests on every pooled slot (spares among them) run as the first did.
        for (int r = 0; r < cap - 1; r++)
        {
            List<float[]> trace = Request($"c{r}");
            for (int i = 0; i < first.Count; i++)
                Assert.True(first[i].AsSpan().SequenceEqual(trace[i]), $"request c{r}: row {i} differs on a spare slot");
        }
    }

    /// <summary>On direct CUDA a slot that grows forgets only its own graphs: another request's recorded decode step keeps
    /// replaying, and the grown slot's request continues as it would have.</summary>
    [CudaFact]
    public void AGrowingSlotKeepsOtherRequestsRecordingsOnDirectCuda()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F16, true);
        using var model = Create(Fixture("dense"), backend: BackendType.Cuda);
        var unified = (IUnifiedBatchModel)model;
        int[] prompt = Enumerable.Range(0, 20).Select(i => 40 + i % 7).ToArray();
        var logits = new float[1][];
        int next = 0, length = 0;
        void Step(string id, int[] tokens, int at)
        {
            Assert.True(unified.TryForwardUnified(new[] { id }, new[] { tokens }, new[] { at }, logits));
            next = ArgMax(logits[0]);
        }
        unified.EnsureSequenceSlot("a", 64);
        Step("a", prompt, 0);
        length = prompt.Length;
        for (int t = 0; t < 8; t++) Step("a", new[] { next }, length++);
        int replays = model.GraphReplays;
        Step("a", new[] { next }, length++);
        Assert.True(model.GraphReplays > replays, "the decode step did not replay before the grow");

        unified.EnsureSequenceSlot("b", 64);
        Step("b", prompt, 0);
        unified.EnsureSequenceSlot("b", 2048);   // b's slot grows

        replays = model.GraphReplays;
        Step("a", new[] { next }, length++);
        Assert.True(model.GraphReplays > replays, "a's recorded decode step traced again after another slot grew");
    }

    /// <summary>A request that decodes in verify windows (speculative decoding) makes spare slots on direct CUDA, as plain
    /// decode steps do, while its recorded windows run (two: a checkpoint's and a retained copy's); a retained copy takes one
    /// without allocating, and the next windows make it again.</summary>
    /// <summary>A slot split over a mesh's devices (here two whole-layer stages) is fitted to each device by its own share:
    /// either device short of its share beside its headroom refuses another slot; the first device's room alone does not
    /// stand for the other's.</summary>
    [Fact]
    public void SlotRoomIsJudgedOnEveryDeviceOfALayerSplit()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F16, true);
        using var model = Create(Fixture("dense"), stages: 2);
        var unified = (IUnifiedBatchModel)model;
        unified.EnsureSequenceSlot("a", 64);
        var shares = model.SlotSharesForTests;
        Assert.Equal(new[] { (0, 0), (1, 0) }, shares.Select(s => (s.Stage, s.Rank)).ToArray());
        Assert.All(shares, s => Assert.True(s.PerToken > 0, "a stage holds no cache"));
        const int capacity = 1024;
        const long total = 1L << 34, headroom = 1L << 30;   // DeviceHeadroom: the larger of 1 GB and 1/32 of the device
        long Need(int device) => shares[device].PerToken * capacity + shares[device].States + (256L << 20) + headroom;
        long Plenty = long.MaxValue / 4;
        model.SlotDeviceMemoryForTests = d => (d == 1 ? Need(1) : Plenty, total);
        Assert.True(model.RoomForSlotsForTests(capacity, 1));
        model.SlotDeviceMemoryForTests = d => (d == 1 ? Need(1) - 1 : Plenty, total);
        Assert.False(model.RoomForSlotsForTests(capacity, 1));
        model.SlotDeviceMemoryForTests = d => (d == 0 ? Need(0) - 1 : Plenty, total);
        Assert.False(model.RoomForSlotsForTests(capacity, 1));
        // Each device is asked for its own share only: both hold exactly theirs.
        model.SlotDeviceMemoryForTests = d => (Need(d), total);
        Assert.True(model.RoomForSlotsForTests(capacity, 1));
        Assert.False(model.RoomForSlotsForTests(capacity, 2));
    }

    /// <summary>A conversation's next turn adopts its retained slot (exactly its rows) and reserves past it: the rows move into
    /// a pooled slot that holds the reservation instead of the slot growing, and the turn continues as it did grown.</summary>
    [Fact]
    public void AnAdoptedSlotTooSmallForItsReservationMovesIntoAPooledSlot() => AssertAdoptedSlotMoves(BackendType.Cpu);

    [CudaFact]
    public void AnAdoptedSlotTooSmallForItsReservationMovesIntoAPooledSlotOnDirectCuda() => AssertAdoptedSlotMoves(BackendType.Cuda);

    private void AssertAdoptedSlotMoves(BackendType backend)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F16, true);
        int[] prompt = Enumerable.Range(0, 200).Select(i => 40 + i % 9).ToArray();
        float[] Continue(bool move, out int moves)
        {
            using var model = Create(Fixture("dense"), backend: backend);
            model.PooledSlotMoveDisabledForTests = !move;
            var unified = (IUnifiedBatchModel)model;
            var logits = new float[1][];
            // The conversation's first turn, retained as it ended.
            unified.EnsureSequenceSlot("turn1", 300);
            Assert.True(unified.TryForwardUnified(new[] { "turn1" }, new[] { prompt }, new[] { 0 }, logits));
            Assert.True(model.RetainSequenceCacheAs("turn1", "conversation"));
            // Another request, run on a large slot, leaves it in the pool.
            unified.EnsureSequenceSlot("big", 3000);
            Assert.True(unified.TryForwardUnified(new[] { "big" }, new[] { new[] { 7 } }, new[] { 0 }, logits));
            model.OnSequenceReleased("big");
            // The conversation's next turn adopts the retained slot.
            Assert.True(model.TryRebindRetainedCache("conversation", "turn2"));
            // Past the adopted slot: 512 cells on the managed executor; on direct CUDA, where slots are sized to the device's
            // room, its capacity class with headroom (2,048 cells).
            Assert.Equal(prompt.Length, unified.EnsureSequenceSlot("turn2", 3000));
            Assert.True(unified.TryForwardUnified(new[] { "turn2" }, new[] { new[] { 11, 12 } }, new[] { prompt.Length }, logits));
            moves = model.PooledSlotMoves;
            return (float[])logits[0].Clone();
        }
        float[] moved = Continue(move: true, out int movedCount);
        float[] grown = Continue(move: false, out int grownCount);
        Assert.Equal(1, movedCount);
        Assert.Equal(0, grownCount);
        Assert.True(moved.AsSpan().SequenceEqual(grown), "the moved slot answers otherwise than the grown one");
    }

    [CudaFact]
    public void VerifyWindowsMakeSpareSlotsBesideRecordedStepsOnDirectCuda()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F16, true);
        using var model = Create(Fixture("dense"), backend: BackendType.Cuda);
        var unified = (IUnifiedBatchModel)model;
        var batched = (TensorSharp.Runtime.Speculative.IBatchedSpeculativeTarget)model;
        var cache = (TensorSharp.Runtime.Scheduling.PrefixCache.IPrefixCacheModel)model;
        cache.AttachPrefixCache(new PayloadSink());
        int vocab = model.Config.VocabSize;
        int[] prompt = Enumerable.Range(0, 20).Select(i => 40 + i % 7).ToArray();
        unified.EnsureSequenceSlot("a", 320);
        var last = new float[1][];
        Assert.True(unified.TryForwardUnified(new[] { "a" }, new[] { prompt }, new[] { 0 }, last));
        Assert.Equal(0, model.SpareSlotsMade);
        int length = prompt.Length, next = ArgMax(last[0]);
        void Window()
        {
            var logits = new float[2 * vocab];
            Assert.True(batched.TryBatchedSpecVerifyLogits(new[] { "a" }, new[] { length }, 2, new[] { next, next }, logits, null));
            batched.CommitBatchedSpecVerify(new[] { "a" }, new[] { length }, new[] { 1 });
            next = ArgMax(logits.AsSpan(0, vocab).ToArray());
            length++;
        }
        int Until(int spares)
        {
            for (int w = 0; w < 16 && model.SpareSlotsMade < spares; w++) Window();
            return model.SpareSlotsMade;
        }
        Assert.Equal(2, Until(2));
        for (int w = 0; w < 4; w++) Window();
        Assert.Equal(2, model.SpareSlotsMade);   // the pool holds both
        Assert.True(cache.TryCaptureDonate("a", "kept", length, out _));
        Assert.Equal(1, model.RetainedByCopy);
        Assert.Equal(3, Until(3));
    }

    /// <summary>A prefix cache attached to the model (spare slots are made for its copies alone); it hears of no payload.</summary>
    private sealed class PayloadSink : TensorSharp.Runtime.Scheduling.PrefixCache.IPrefixPayloadSink
    {
        public void OnPayloadInvalidated(string payloadKey, TensorSharp.Runtime.Scheduling.PrefixCache.InvalidationReason reason) { }
    }

    private void AssertParkedRecordingsServeTheNextRequest(string variant, bool retain)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F16, true);
        using var model = Create(Fixture(variant), backend: BackendType.Cuda);
        var unified = (IUnifiedBatchModel)model;
        int[] prompt = Enumerable.Range(0, 20).Select(i => 40 + i % 7).ToArray();
        List<float[]> Request(string id)
        {
            unified.EnsureSequenceSlot(id, 320);
            var logits = new float[1][];
            Assert.True(unified.TryForwardUnified(new[] { id }, new[] { prompt }, new[] { 0 }, logits));
            var trace = new List<float[]> { (float[])logits[0].Clone() };
            int next = ArgMax(logits[0]), length = prompt.Length;
            // Past the 256-key window edge: one recording over the slot's capacity (two, one a window, where the step attends
            // its 256-key bucket).
            for (int t = 0; t < 250; t++)
            {
                Assert.True(unified.TryForwardUnified(new[] { id }, new[] { new[] { next } }, new[] { length }, logits));
                trace.Add((float[])logits[0].Clone());
                next = ArgMax(logits[0]);
                length++;
            }
            if (retain) Assert.True(model.RetainSequenceCache(id));
            else model.OnSequenceReleased(id);
            return trace;
        }
        List<float[]> first = Request("a");
        int recorded = model.ParkedRecordingCount;
        Assert.True(recorded >= 1, "no recording parked after the first request");
        int adopted = model.RecordingsAdopted, relocated = model.RecordingsRelocated;
        List<float[]> second = Request("b");
        Assert.True(model.RecordingsAdopted - adopted >= recorded,
            $"the second request replayed {model.RecordingsAdopted - adopted} of {recorded} parked recordings");
        Assert.Equal(retain, model.RecordingsRelocated - relocated >= recorded);
        Assert.Equal(first.Count, second.Count);
        for (int i = 0; i < first.Count; i++)
            Assert.True(first[i].AsSpan().SequenceEqual(second[i]), $"row {i}: the second request's logits differ (relL2 {RelL2(second[i], first[i]):E2})");
    }

    /// <summary>The same over per-rank graphs on two ggml_cuda GPUs: every rank's step advances in place and the ranks run
    /// together without a trace.</summary>
    [GgmlTheory(BackendType.GgmlCuda, Devices = 2)]
    [InlineData("dense")]
    [InlineData("moe-tp")]
    public void AdvancedRankedDecodeStepsMatchTracedOnes(string variant)
        => AssertAdvancedDecodeMatchesTraced(variant, path => new Qwen35RuntimeModel(path, BackendType.GgmlCuda, 2, null, 1));

    /// <summary>A verify window kept to advance (and, on CUDA, recorded) is never run again on verify buffers its states
    /// gave up since (a retained or pooled slot frees them): the next window traces anew and answers as a model whose
    /// buffers stayed. A recorded window had written the freed buffers of the pooled slot the next request took (CUDA
    /// error 700 on an RTX PRO 6000).</summary>
    [Theory]
    [InlineData(BackendType.Cpu)]
    public void AdvancedWindowsRetraceAfterTheirVerifyBuffersAreFreed(BackendType backend) => AssertWindowsRetraceAfterRelease(backend);

    [CudaTheory]
    [InlineData(BackendType.Cuda)]
    public void AdvancedWindowsRetraceAfterTheirVerifyBuffersAreFreedOnDirectCuda(BackendType backend) => AssertWindowsRetraceAfterRelease(backend);

    /// <summary>A verify window that keeps a state after every row (the windows of an executor without in-place windows, here
    /// direct CUDA with them off) writes its rows and states straight into the request's verify buffer (the alias's
    /// destination), where each window had copied them (12.7 MB a layer of a 4-row window of Qwen3.8 Flash Next): the same
    /// kernel into other memory, so every window's logits are the same bits.</summary>
    [CudaFact]
    public void VerifyWindowRecurrenceWritesItsVerifyBufferInPlaceOnDirectCuda()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F16, true);
        string path = Fixture("dense");
        (List<float[]> Logits, long InPlace, long Copies) Run(bool inPlace)
        {
            Qwen35RuntimeModel.WindowStatesInPlaceOffForTests = true;
            Qwen35RuntimeModel created;
            try { created = Create(path, backend: BackendType.Cuda); }
            finally { Qwen35RuntimeModel.WindowStatesInPlaceOffForTests = false; }
            using var model = created;
            var executor = (TensorSharp.Graph.TensorGraphExecutor)model.GraphExecutorForTests;
            executor.AliasDestinationsDisabledForTests = !inPlace;
            var spec = (TensorSharp.Runtime.Speculative.ISpeculativeTarget)model;
            int vocab = model.Config.VocabSize;
            spec.SpecForward(Enumerable.Range(0, 20).Select(i => 40 + i % 7).ToArray(), null, new float[vocab], false);
            long inPlaceBefore = executor.AliasesInPlace, copiesBefore = executor.AliasCopies;
            var trace = new List<float[]>();
            for (int w = 0; w < 6; w++)
            {
                int[] window = Enumerable.Range(0, 4).Select(i => 60 + (w * 4 + i) % 30).ToArray();
                var logits = new float[4 * vocab];
                spec.SpecForward(window, null, logits, true);
                spec.SpecOnVerifyAccepted(3, 4);
                trace.Add(logits);
            }
            return (trace, executor.AliasesInPlace - inPlaceBefore, executor.AliasCopies - copiesBefore);
        }
        var placed = Run(inPlace: true);
        var copied = Run(inPlace: false);
        Assert.True(placed.InPlace > copied.InPlace, $"windows' results in place: {placed.InPlace} vs {copied.InPlace} without destinations");
        Assert.True(placed.Copies < copied.Copies, $"alias copies: {placed.Copies} vs {copied.Copies} without destinations");
        for (int w = 0; w < placed.Logits.Count; w++)
            Assert.True(placed.Logits[w].AsSpan().SequenceEqual(copied.Logits[w]), $"window {w} differs with its result written in place");
    }

    /// <summary>A verify window on direct CUDA writes its rows' outputs alone and saves its scan inputs, the live recurrent
    /// state left as it was (each window had kept a state after every row, 2 MB a row and layer on Qwen3.8 Flash Next): the
    /// commit steps the kept rows on the state again, in place, on the window's kernels, and a restored window leaves the
    /// state alone. Windows cut at every row, accepted whole and restored give the logits of windows that kept a state per
    /// row, bit for bit, and so does the decode after them.</summary>
    [CudaTheory]
    [InlineData("dense")]
    [InlineData("dense-gqa")]
    public void InPlaceVerifyWindowsMatchSnapshotWindowsOnDirectCuda(string variant)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F16, true);
        string path = Fixture(variant);
        (List<float[]> Logits, int Rows) Run(bool inPlace)
        {
            Qwen35RuntimeModel.WindowStatesInPlaceOffForTests = !inPlace;
            Qwen35RuntimeModel created;
            try { created = Create(path, backend: BackendType.Cuda); }
            finally { Qwen35RuntimeModel.WindowStatesInPlaceOffForTests = false; }
            using var model = created;
            var spec = (TensorSharp.Runtime.Speculative.ISpeculativeTarget)model;
            int vocab = model.Config.VocabSize;
            spec.SpecForward(Enumerable.Range(0, 20).Select(i => 40 + i % 7).ToArray(), null, new float[vocab], false);
            var trace = new List<float[]>();
            // Accepted rows of each 4-row window (-1: restored to the state before it).
            int[] accepted = { 3, 0, 2, -1, 1, 3, 0, 2 };
            for (int w = 0; w < accepted.Length; w++)
            {
                int[] window = Enumerable.Range(0, 4).Select(i => 60 + (w * 4 + i) % 30).ToArray();
                var logits = new float[4 * vocab];
                spec.SpecForward(window, null, logits, true);
                if (accepted[w] < 0) spec.SpecRestoreRecurrentState();
                else spec.SpecOnVerifyAccepted(accepted[w], 4);
                trace.Add(logits);
            }
            for (int d = 0; d < 3; d++)
            {
                var logits = new float[vocab];
                spec.SpecForward(new[] { 50 + d }, null, logits, false);
                trace.Add(logits);
            }
            return (trace, model.WindowStateRowsForTests);
        }
        var inPlace = Run(inPlace: true);
        var snapshots = Run(inPlace: false);
        Assert.True(inPlace.Rows > 1, $"the windows did not update the states in place ({inPlace.Rows} rows)");
        Assert.Equal(1, snapshots.Rows);
        for (int w = 0; w < inPlace.Logits.Count; w++)
            Assert.True(inPlace.Logits[w].AsSpan().SequenceEqual(snapshots.Logits[w]),
                $"step {w} differs with in-place windows ({RelL2(inPlace.Logits[w], snapshots.Logits[w]):E2} relative L2)");
    }

    /// <summary>A verify window's commit on direct CUDA runs as one region (GraphFusion.DeltaNetCommit: ts_graph_gdn_commit_f32
    /// for every recurrent layer, where the ops were about fifteen graph nodes a layer): windows cut at every row, accepted
    /// whole and restored, and the decode after them, give the logits of commits run as their ops, bit for bit.</summary>
    [CudaTheory]
    [InlineData("dense")]
    [InlineData("dense-gqa")]
    public void CommitRegionOnDirectCudaAnswersAsItsOps(string variant)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F16, true);
        string path = Fixture(variant);
        (List<float[]> Logits, long Regions, int Layers) Run(bool region)
        {
            using var model = Create(path, backend: BackendType.Cuda);
            model.CommitRegionDisabledForTests = !region;
            var executor = Assert.IsType<TensorSharp.Cuda.CudaGraphExecutor>(model.GraphExecutorForTests);
            var spec = (TensorSharp.Runtime.Speculative.ISpeculativeTarget)model;
            int vocab = model.Config.VocabSize;
            spec.SpecForward(Enumerable.Range(0, 20).Select(i => 40 + i % 7).ToArray(), null, new float[vocab], false);
            var trace = new List<float[]>();
            // Accepted rows of each 4-row window (-1: restored to the state before it).
            int[] accepted = { 3, 0, 2, -1, 1, 3, 0, 2 };
            for (int w = 0; w < accepted.Length; w++)
            {
                int[] window = Enumerable.Range(0, 4).Select(i => 60 + (w * 4 + i) % 30).ToArray();
                var logits = new float[4 * vocab];
                spec.SpecForward(window, null, logits, true);
                if (accepted[w] < 0) spec.SpecRestoreRecurrentState();
                else spec.SpecOnVerifyAccepted(accepted[w], 4);
                trace.Add(logits);
            }
            for (int d = 0; d < 3; d++)
            {
                var logits = new float[vocab];
                spec.SpecForward(new[] { 50 + d }, null, logits, false);
                trace.Add(logits);
            }
            Assert.True(model.WindowStateRowsForTests > 1, $"the windows did not update the states in place ({model.WindowStateRowsForTests} rows)");
            return (trace, executor.DeltaNetCommitRegions, model.DeltaNetCommitRegionLayers);
        }
        var fused = Run(region: true);
        var ops = Run(region: false);
        Assert.True(fused.Regions > 0, "no commit ran as a region");
        Assert.True(fused.Layers > 0, "no commit program held a region");
        Assert.Equal(0, ops.Regions);
        Assert.Equal(0, ops.Layers);
        for (int w = 0; w < fused.Logits.Count; w++)
            Assert.True(fused.Logits[w].AsSpan().SequenceEqual(ops.Logits[w]),
                $"step {w} differs with the commit region ({RelL2(fused.Logits[w], ops.Logits[w]):E2} relative L2)");
    }

    /// <summary>
    /// Every step's query/key prep and K/V cache writes on direct CUDA as one kernel (GraphFusion.AttentionPrep: q and k normed and
    /// rotated where the projections put them, the rows written into the caches) against their ops: prompts of 17 rows (their
    /// queries handed to the attention head by head) and 7, decode steps (recorded and replayed), 4-row verify windows accepted in
    /// part or restored, and a 2,100-token prompt split at the 2,048-row chunk boundary with the decode after it, bit for bit.
    /// </summary>
    [CudaTheory]
    [InlineData("dense")]
    [InlineData("moe-quantized")]
    public void AttentionPrepRegionOnDirectCudaAnswersAsItsOps(string variant)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F16, true);
        string path = Fixture(variant);
        (List<float[]> Logits, long Regions, long HeadMajor, int Replays) Run(bool region)
        {
            using var model = Create(path, backend: BackendType.Cuda);
            model.AttentionPrepRegionDisabledForTests = !region;
            var executor = Assert.IsType<TensorSharp.Cuda.CudaGraphExecutor>(model.GraphExecutorForTests);
            List<float[]> trace = Script(model);
            model.ResetKVCache();
            var spec = (TensorSharp.Runtime.Speculative.ISpeculativeTarget)model;
            int vocab = model.Config.VocabSize;
            var prompt = new float[vocab];
            spec.SpecForward(Enumerable.Range(0, 20).Select(i => 40 + i % 7).ToArray(), null, prompt, false);
            trace.Add(prompt);
            // Accepted rows of each 4-row window (-1: restored to the state before it).
            int[] accepted = { 3, 0, -1, 2 };
            for (int w = 0; w < accepted.Length; w++)
            {
                int[] window = Enumerable.Range(0, 4).Select(i => 60 + (w * 4 + i) % 30).ToArray();
                var logits = new float[4 * vocab];
                spec.SpecForward(window, null, logits, true);
                if (accepted[w] < 0) spec.SpecRestoreRecurrentState();
                else spec.SpecOnVerifyAccepted(accepted[w], 4);
                trace.Add(logits);
            }
            model.ResetKVCache();
            trace.Add((float[])model.ForwardRefill(Enumerable.Range(0, 2100).Select(i => 3 + (i * 37) % 250).ToArray()).Clone());
            foreach (int token in new[] { 81, 82, 83 })
                trace.Add((float[])model.Forward(new[] { token }).Clone());
            return (trace, executor.AttentionPrepRegions, executor.HeadMajorQueryPreps, model.GraphReplays);
        }
        var fused = Run(region: true);
        var ops = Run(region: false);
        Assert.True(fused.Regions > 0, "no step ran its attention prep as the region");
        Assert.True(fused.HeadMajor > 0, "no prompt chunk handed its queries to the attention head by head");
        Assert.Equal(0, ops.Regions);
        Assert.True(fused.Replays > 0 && ops.Replays > 0, "the decode steps did not replay");
        Assert.Equal(ops.Logits.Count, fused.Logits.Count);
        for (int i = 0; i < fused.Logits.Count; i++)
            Assert.True(fused.Logits[i].AsSpan().SequenceEqual(ops.Logits[i]),
                $"step {i} differs with the prep region ({RelL2(fused.Logits[i], ops.Logits[i]):E2} relative L2)");
    }

    private void AssertWindowsRetraceAfterRelease(BackendType backend)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense");
        List<float[]> Run(bool release, out int advances)
        {
            using var model = Create(path, backend: backend);
            var spec = (TensorSharp.Runtime.Speculative.ISpeculativeTarget)model;
            int vocab = model.Config.VocabSize;
            spec.SpecForward(Enumerable.Range(0, 20).Select(i => 40 + i % 7).ToArray(), null, new float[vocab], false);
            var trace = new List<float[]>();
            for (int w = 0; w < 7; w++)
            {
                if (w == 5)
                {
                    // The window before keeps all its rows (settled now, as a request's last verify is before its slot is retained).
                    spec.SpecOnVerifyAccepted(3, 4);
                    if (release) model.ReleaseActiveSpecBuffersForTests();
                }
                int[] window = Enumerable.Range(0, 4).Select(i => 60 + (w * 4 + i) % 30).ToArray();
                var logits = new float[4 * vocab];
                spec.SpecForward(window, null, logits, true);
                trace.Add(logits);
            }
            advances = model.WindowAdvances;
            return trace;
        }
        List<float[]> kept = Run(release: false, out int keptAdvances), released = Run(release: true, out _);
        Assert.True(keptAdvances >= 3, $"{backend}: {keptAdvances} windows advanced");
        // CUDA records a kept window, whose attention runs over the whole window with its mask (a retraced one over the
        // written keys): the same sums in another order. A freed buffer read back gives garbage, or CUDA error 700.
        for (int w = 0; w < kept.Count; w++)
        {
            if (backend == BackendType.Cpu)
            {
                Assert.True(kept[w].AsSpan().SequenceEqual(released[w]), $"{backend}: window {w} differs after the verify buffers were freed");
                continue;
            }
            double num = 0, den = 0;
            for (int i = 0; i < kept[w].Length; i++)
            {
                double d = kept[w][i] - released[w][i];
                num += d * d;
                den += kept[w][i] * (double)kept[w][i];
            }
            Assert.True(Math.Sqrt(num / den) < 1e-2, $"{backend}: window {w} is {Math.Sqrt(num / den):E2} away after the verify buffers were freed");
        }
    }

    private void AssertAdvancedDecodeMatchesTraced(string variant, Func<string, Qwen35RuntimeModel> create)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(variant);

        List<float[]> Run(bool advance, out int advances)
        {
            using var model = create(path);
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
        // ggml's GPU products are not row-invariant (SpeculativeRuntimeChecks: a stream parts from plain decoding's only at a
        // near-tie there), within the tie tolerance the other ggml GPU speculation checks take: on two Blackwell MIG slices
        // the tokens' logits were 2.15e-3 of the largest apart, 4e-4 on an A40.
        double nearTie = backend is BackendType.GgmlCuda or BackendType.GgmlVulkan or BackendType.GgmlMetal ? 1e-2 : 0;
        await SpeculativeRuntimeChecks.AssertEngineSpeculationMatchesPlain(() => Create(path, backend: backend), options,
            $"qwen35 {variant} {speculator} {backend}", nearTie: nearTie);
    }

    /// <summary>As <see cref="EngineSpeculationMatchesPlain"/> on the CUDA executor (the verify windows' decode-size
    /// products and the draft head's step run there).</summary>
    [CudaTheory]
    [InlineData("dense", SpeculatorRegistry.NGram)]
    [InlineData("moe", SpeculatorRegistry.NGram)]
    [InlineData("dense-nextn", SpeculatorRegistry.DraftHead)]
    public Task EngineSpeculationOnDirectCudaMatchesPlain(string variant, string speculator)
        => EngineSpeculationMatchesPlain(variant, speculator, BackendType.Cuda);

    /// <summary>The same over a q8_0 cache (an explicit <c>--kv-cache-dtype q8_0</c>: the CUDA executor's _q8 flash kernels and
    /// q8_0 row writes, the verify windows' rows the decode rows' bits): the plain greedy streams. Every model of both runs held
    /// the q8_0 cache, ran graphs, and sent no attention to the generic or tiled path
    /// (<see cref="CudaQ8KvAttentionRoutesTests.AssertFlashKernelsOnly"/>). Skips where the flash kernels have no sm_80 bodies: the
    /// executor then takes no q8_0 cache and the model runs no graphs, so no verify window is batched.</summary>
    [CudaTheory(TensorCoreModule = "tensorsharp_flash_prefill")]
    [InlineData("dense", SpeculatorRegistry.NGram)]
    [InlineData("dense-nextn", SpeculatorRegistry.DraftHead)]
    public async Task EngineSpeculationOnDirectCudaMatchesPlainOverAQ8Cache(string variant, string speculator)
    {
        using IDisposable kv = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.Q8_0);
        string path = Fixture(variant);
        var options = new SpeculationOptions { Enabled = true, SpeculatorName = speculator, MaxDraftTokens = 3, MinDraftProb = 0 };
        var models = new List<CudaQ8KvAttentionRoutesTests.Q8Watch>();
        await SpeculativeRuntimeChecks.AssertEngineSpeculationMatchesPlain(() =>
        {
            Qwen35RuntimeModel model = Create(path, backend: BackendType.Cuda);
            lock (models) models.Add(CudaQ8KvAttentionRoutesTests.Watch(model));
            return model;
        }, options, $"qwen35 {variant} {speculator} Cuda q8_0");
        Assert.NotEmpty(models);
        foreach (CudaQ8KvAttentionRoutesTests.Q8Watch model in models)
            CudaQ8KvAttentionRoutesTests.AssertFlashKernelsOnly(model, $"qwen35 {variant} {speculator}", _output);
    }

    /// <summary>A batched verify whose graph finds no device memory for its compute buffer is retried once after the
    /// cached graphs are freed, and declined (the step decodes plainly) when it fails again: the requests get the plain
    /// greedy streams either way, where the failure used to fail them.</summary>
    [Theory]
    [InlineData(1)]   // the retry runs the verify
    [InlineData(3)]   // a verify declined (both attempts fail), the next one retried
    public async Task VerifyOutOfDeviceMemoryDeclinesInsteadOfFailing(int failures)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense");
        var options = new SpeculationOptions { Enabled = true, SpeculatorName = SpeculatorRegistry.NGram, MaxDraftTokens = 3, MinDraftProb = 0 };
        await SpeculativeRuntimeChecks.AssertEngineSpeculationMatchesPlain(() =>
        {
            var model = Create(path);
            model.InjectedVerifyAllocationFailures = failures;
            return model;
        }, options, $"qwen35 dense, {failures} verify allocation failures");
    }

    /// <summary>A unified step over several requests whose graph finds no device memory for its compute buffer is
    /// retried once after the cached graphs are freed, then run as two halves of its requests: the requests get the
    /// streams of an unhindered run, where the failure used to fail them all.</summary>
    [Theory]
    [InlineData(1)]   // the retry runs the step
    [InlineData(2)]   // both attempts fail: the step runs in halves
    public async Task ForwardOutOfDeviceMemoryRetriesThenSplits(int failures)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense");
        Qwen35RuntimeModel hindered = null;
        await SpeculativeRuntimeChecks.AssertEngineStreamsMatch(() => Create(path), () =>
        {
            hindered = Create(path);
            hindered.InjectedForwardAllocationFailures = failures;
            return hindered;
        }, $"qwen35 dense, {failures} forward allocation failures");
        Assert.Equal(0, hindered.InjectedForwardAllocationFailures);
    }

    /// <summary>The NextN layer's draft chains of several requests run as one draft-layer forward a step over their slots
    /// (each lane's final-normed row chained, where qwen4exp chains its stream): each lane drafts what its own chain does,
    /// on the managed CPU through the test seam and on the CUDA executor.</summary>
    [Fact]
    public void BatchedDraftChainDraftsEachLanesChain() => AssertBatchedDraftChain(BackendType.Cpu);

    [CudaFact]
    public void BatchedDraftChainDraftsEachLanesChainOnDirectCuda() => AssertBatchedDraftChain(BackendType.Cuda);

    private void AssertBatchedDraftChain(BackendType backend)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense-nextn");
        SpeculativeRuntimeChecks.AssertBatchedDraftChainDraftsEachLane(() =>
        {
            Qwen35RuntimeModel model = Create(path, backend: backend);
            model.DraftChainForTests = backend == BackendType.Cpu;
            return model;
        }, $"qwen35 nextn {backend}", backend == BackendType.Cpu ? 1e-5 : 1e-3);
    }

    /// <summary>Three requests speculating with the NextN layer through the engine draft in one pass a step, and emit
    /// what plain decoding and drafting one request at a time emit.</summary>
    [Fact]
    public Task BatchedDraftChainsSpeculateAsEachRequest() => AssertBatchedDraftChains(BackendType.Cpu);

    [CudaFact]
    public Task BatchedDraftChainsSpeculateAsEachRequestOnDirectCuda() => AssertBatchedDraftChains(BackendType.Cuda);

    private async Task AssertBatchedDraftChains(BackendType backend)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense-nextn");
        var options = new SpeculationOptions { Enabled = true, SpeculatorName = SpeculatorRegistry.DraftHead, MaxDraftTokens = 3, MinDraftProb = 0 };
        await SpeculativeRuntimeChecks.AssertBatchedDraftChainsDraftAsEachRequest(() =>
        {
            Qwen35RuntimeModel model = Create(path, backend: backend);
            model.DraftChainForTests = backend == BackendType.Cpu;
            return model;
        }, options, $"qwen35 nextn {backend}");
    }

    /// <summary>The draft layer follows a speculating request's plain prompt (the trunk's own steps hand it their rows):
    /// its first draft is the one the speculator's catch-up over the same rows gives.</summary>
    [Fact]
    public void FollowedPromptDraftsAsItsCatchUp()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense-nextn");
        SpeculativeRuntimeChecks.AssertFollowedPromptDraftsAsItsCatchUp(() => Create(path), "qwen35 nextn cpu", 1e-4);
    }

    /// <summary>The first decode step after a prompt drafts with the draft layer the trunk keeps in step, from the carry
    /// the prompt's plain steps left with it (the slot's draft carry), and the speculating streams are plain decoding's:
    /// through the engine's unified step and through a context armed after a plain prompt.</summary>
    [Fact]
    public Task FirstDecodeStepAfterThePromptDrafts() => FirstDecodeStepDrafts(BackendType.Cpu);

    /// <summary>The same on the direct CUDA backend, where the prompt's steps must stay exactly the plain ones.</summary>
    [CudaFact]
    public Task FirstDecodeStepAfterThePromptDraftsOnDirectCuda() => FirstDecodeStepDrafts(BackendType.Cuda);

    private async Task FirstDecodeStepDrafts(BackendType backend)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense-nextn");
        var options = new SpeculationOptions { Enabled = true, SpeculatorName = SpeculatorRegistry.DraftHead, MaxDraftTokens = 3, MinDraftProb = 0 };
        await SpeculativeRuntimeChecks.AssertFirstDecodeStepDrafts(() => Create(path, backend: backend), options, $"qwen35 nextn {backend}");
        SpeculativeRuntimeChecks.AssertPlainPromptCarriesIntoTheFirstDraft(() => Create(path, backend: backend), $"qwen35 nextn {backend} plain prompt");
    }

    [CudaFact]
    public void FollowedPromptDraftsAsItsCatchUpOnDirectCuda()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense-nextn");
        SpeculativeRuntimeChecks.AssertFollowedPromptDraftsAsItsCatchUp(() => Create(path, backend: BackendType.Cuda), "qwen35 nextn cuda", 2e-3);
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
        List<float[]> legacy = LegacyGolden("draft_dense_nextn_cpu");
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
        List<float[]> legacy = LegacyGolden("draft_dense_nextn_cpu");
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

    /// <summary>The same image span as Qwen35Model ran it on the managed CPU (its golden).</summary>
    [Fact]
    public void ImageSpanMatchesTheFamilyImplementation()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense");
        List<float[]> legacy = LegacyGolden("media_dense_cpu");
        using var runtime = Create(path, backend: TestGates.PinnedGgmlBackend);
        List<float[]> actual = MediaScript(runtime, out _);
        double tolerance = RoundsActivations(TestGates.PinnedGgmlBackend) ? 2.5e-2 : 5e-3;
        Assert.Equal(legacy.Count, actual.Count);
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
    [InlineData("dense-nextn")]   // a draft layer: K/V rows past the trunk's layers, no recurrent state there
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

    /// <summary>A prompt's automatic checkpoint is taken after its last step and keeps that step's last logits
    /// (the recurrent state cannot rewind to a tail), so the same prompt sent again resumes at its very end with
    /// nothing to prefill, and draws the greedy tokens of the run that computed the row and of a cold engine.</summary>
    [Fact]
    public Task AnIdenticalPromptResumesAtItsEndOnTheManagedBackend()
        => AssertIdenticalPromptResumesAtItsEnd(BackendType.Cpu, KvCacheDtype.F32);

    [CudaFact]
    public Task AnIdenticalPromptResumesAtItsEndOnDirectCuda()
        => AssertIdenticalPromptResumesAtItsEnd(BackendType.Cuda, KvCacheDtype.F16);

    private async Task AssertIdenticalPromptResumesAtItsEnd(BackendType backend, KvCacheDtype kv)
    {
        KvCacheDtypeConfig.RestoreForTests(kv, true);
        string path = Fixture("dense");
        const int maxNewTokens = 8;
        int[] prompt = Enumerable.Range(0, 45).Select(i => 65 + (i * 7) % 50).ToArray();
        SchedulerConfig Config(bool prefixCache) => new()
        {
            MaxNumBatchedTokens = 256,
            MaxNumRunningSequences = 4,
            NumBlocks = 32,
            BlockSize = 16,
            EnablePrefixCaching = prefixCache,
            StopRepetition = false,
        };
        static async Task<(int[] Tokens, InferenceCompletion Completion)> Serve(InferenceEngine engine, string id, int[] prompt)
        {
            var handle = engine.SubmitRequest(new SequenceState(id, prompt, maxNewTokens, 16, SamplingConfig.Greedy,
                cacheScope: "conversation"));
            var tokens = new List<int>();
            await foreach (int t in handle.Tokens.ReadAllAsync())
                tokens.Add(t);
            var completion = await handle.Completion;
            await engine.WhenStepSettledAsync();
            return (tokens.ToArray(), completion);
        }

        int[] cold;
        using (var reference = Create(path, backend: backend))
        using (var coldEngine = new InferenceEngine(reference, Config(false), NullLogger.Instance))
            cold = (await Serve(coldEngine, "cold", prompt).WaitAsync(TimeSpan.FromMinutes(3))).Tokens;

        using var model = Create(path, backend: backend);
        Assert.True(model.SupportsPrefixCheckpoints);
        using var engine = new InferenceEngine(model, Config(true), NullLogger.Instance);
        var first = await Serve(engine, "first", prompt).WaitAsync(TimeSpan.FromMinutes(3));
        var again = await Serve(engine, "again", prompt).WaitAsync(TimeSpan.FromMinutes(3));
        _output.WriteLine($"{backend}: cold [{string.Join(",", cold)}] first [{string.Join(",", first.Tokens)}] " +
            $"again [{string.Join(",", again.Tokens)}] reused {again.Completion.PrefixCacheReusedTokens}");
        Assert.True(cold.Length > 0, "the cold run produced nothing");
        Assert.Equal(0, first.Completion.PrefixCacheReusedTokens);
        Assert.Equal(prompt.Length, again.Completion.PrefixCacheReusedTokens);
        Assert.Equal(cold, first.Tokens);
        Assert.Equal(first.Tokens, again.Tokens);
    }

    /// <summary>On one Metal or CUDA device a new request's slot must fit beside the running requests' (else it waits:
    /// <see cref="TensorSharp.Runtime.Scheduling.SequenceSlotUnavailableException"/>), a lone request always gets one, and
    /// the prefix cache is asked to free what one more fresh slot lacks (as Qwen4ExpModel admits its holders).</summary>
    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData(BackendType.GgmlMetal)]
    public void NewSlotsWaitForDeviceMemoryOnMetal(BackendType backend) => AssertSlotsWaitForDeviceMemory(backend);

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(BackendType.GgmlCuda)]
    public void NewSlotsWaitForDeviceMemoryOnCuda(BackendType backend) => AssertSlotsWaitForDeviceMemory(backend);

    private void AssertSlotsWaitForDeviceMemory(BackendType backend)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        using var model = Create(Fixture("dense"), backend: backend);
        const long gb = 1L << 30;
        (long Free, long Total)? memory = (64 * gb, 64 * gb);
        model.DeviceMemoryProbe = () => memory;
        var unified = (IUnifiedBatchModel)model;

        memory = (1, 64 * gb);
        Assert.Equal(0, unified.EnsureSequenceSlot("lone", 64));    // nothing else holds a slot: it always gets one
        Assert.True(model.DeviceSlotDeficitBytes > 0, "a full device asks the prefix cache for a fresh slot's room");
        Assert.Throws<TensorSharp.Runtime.Scheduling.SequenceSlotUnavailableException>(() => unified.EnsureSequenceSlot("second", 64));
        Assert.True(unified.TryForwardUnified(new[] { "lone" }, new[] { new[] { 5, 6, 7 } }, new[] { 0 }, new float[1][]),
            "the running request is untouched by the refusal");

        memory = (64 * gb, 64 * gb);
        Assert.Equal(0, model.DeviceSlotDeficitBytes);
        Assert.Equal(0, unified.EnsureSequenceSlot("second", 64));
    }

    /// <summary>A slot whose rows a graph left on the device grows there (new caches zero-filled on the device, the rows
    /// copied device to device) when a request's reservation passes its capacity, and decoding continues exactly as
    /// without the growth. Through the host it downloaded every row and uploaded them again.</summary>
    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(BackendType.GgmlCuda)]
    public void SlotsGrowOnTheDeviceOnCuda(BackendType backend) => AssertSlotsGrowOnTheDevice(backend);

    [GgmlTheory(BackendType.GgmlVulkan)]
    [InlineData(BackendType.GgmlVulkan)]
    public void SlotsGrowOnTheDeviceOnVulkan(BackendType backend) => AssertSlotsGrowOnTheDevice(backend);

    private void AssertSlotsGrowOnTheDevice(BackendType backend)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense");
        int[] prompt = Enumerable.Range(0, 37).Select(i => 41 + (i * 7) % 90).ToArray();
        int[] tail = { 71, 72, 73, 74 };
        var expected = new List<float[]>();
        using (var reference = Create(path, backend: backend))
        {
            var u = (IUnifiedBatchModel)reference;
            u.EnsureSequenceSlot("a", 2048);
            var logits = new float[1][];
            Assert.True(u.TryForwardUnified(new[] { "a" }, new[] { prompt }, new[] { 0 }, logits));
            for (int i = 0; i < tail.Length; i++)
            {
                Assert.True(u.TryForwardUnified(new[] { "a" }, new[] { new[] { tail[i] } }, new[] { prompt.Length + i }, logits));
                expected.Add((float[])logits[0].Clone());
            }
        }

        using var model = Create(path, backend: backend);
        var unified = (IUnifiedBatchModel)model;
        unified.EnsureSequenceSlot("a", 64);
        var rows = new float[1][];
        Assert.True(unified.TryForwardUnified(new[] { "a" }, new[] { prompt }, new[] { 0 }, rows));
        int before = model.DeviceSlotGrowths;
        Assert.Equal(prompt.Length, unified.EnsureSequenceSlot("a", 2048));   // past the fresh slot's 512 cells
        Assert.True(model.DeviceSlotGrowths > before, "the slot grew through the host");
        for (int i = 0; i < tail.Length; i++)
        {
            Assert.True(unified.TryForwardUnified(new[] { "a" }, new[] { new[] { tail[i] } }, new[] { prompt.Length + i }, rows));
            double rel = RelL2(rows[0], expected[i]);
            Assert.True(rel <= 1e-5, $"row {i} after the growth: relL2 {rel:E2}");
        }
    }

    /// <summary>A device without room for a new request's slot first takes back what the idle slots hold: a finished
    /// request kept for a continuation drops its device copies (its host tensors keep every row and state), and a
    /// continuation cloned from it later decodes as the request did. Kept on the device, finished requests filled a GPU
    /// until new requests' steps found no memory. (Metal admits by its working set and keeps no separate copies; Vulkan
    /// admits no slot by device memory, so only a failed step's retry drops them there.)</summary>
    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(BackendType.GgmlCuda)]
    public void IdleSlotsGiveBackTheirDeviceCopiesOnCuda(BackendType backend) => AssertIdleSlotsGiveBackDeviceCopies(backend);

    private void AssertIdleSlotsGiveBackDeviceCopies(BackendType backend)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("dense");
        int[] prompt = Enumerable.Range(0, 37).Select(i => 41 + (i * 7) % 90).ToArray();
        int[] tail = { 71, 72, 73 };
        var expected = new List<float[]>();
        using (var reference = Create(path, backend: backend))
        {
            var r = (IUnifiedBatchModel)reference;
            r.EnsureSequenceSlot("a", 64);
            var logits = new float[1][];
            Assert.True(r.TryForwardUnified(new[] { "a" }, new[] { prompt }, new[] { 0 }, logits));
            for (int i = 0; i < tail.Length; i++)
            {
                Assert.True(r.TryForwardUnified(new[] { "a" }, new[] { new[] { tail[i] } }, new[] { prompt.Length + i }, logits));
                expected.Add((float[])logits[0].Clone());
            }
        }

        using var model = Create(path, backend: backend);
        const long gb = 1L << 30;
        (long Free, long Total)? memory = (64 * gb, 64 * gb);
        model.DeviceMemoryProbe = () => memory;
        var unified = (IUnifiedBatchModel)model;
        var prefixCache = (TensorSharp.Runtime.Scheduling.PrefixCache.IPrefixCacheModel)model;
        var rows = new float[1][];
        unified.EnsureSequenceSlot("a", 64);
        Assert.True(unified.TryForwardUnified(new[] { "a" }, new[] { prompt }, new[] { 0 }, rows));
        Assert.True(prefixCache.TryCaptureCopy("a", "kept", out _));   // "a" keeps running beside its copy

        long resident = TensorSharp.GGML.GgmlBasicOps.DeviceCopyCacheResidentBytes();
        memory = (1, 64 * gb);
        Assert.Throws<TensorSharp.Runtime.Scheduling.SequenceSlotUnavailableException>(() => unified.EnsureSequenceSlot("b", 64));
        Assert.True(TensorSharp.GGML.GgmlBasicOps.DeviceCopyCacheResidentBytes() < resident, "the kept slot's device copies were dropped");

        memory = (64 * gb, 64 * gb);
        Assert.True(prefixCache.TryMaterialize(new TensorSharp.Runtime.Scheduling.PrefixCache.MaterializeRequest(
            TensorSharp.Runtime.Scheduling.PrefixCache.MaterializeOp.Clone, "kept", "c", prompt.Length, prompt.Length)));
        for (int i = 0; i < tail.Length; i++)
        {
            Assert.True(unified.TryForwardUnified(new[] { "c" }, new[] { new[] { tail[i] } }, new[] { prompt.Length + i }, rows));
            double rel = RelL2(rows[0], expected[i]);
            Assert.True(rel <= 1e-5, $"row {i} of the continuation: relL2 {rel:E2}");
        }
    }

    /// <summary>The unified step times its host parts as the batched decode does
    /// (tensorsharp.engine.step.phase.duration): the decoders' draws ("sample", by decoders) and each forward ("issue",
    /// by the rows it ran: a prompt chunk's many, two decoders' two).</summary>
    [Fact]
    public async Task UnifiedStepsRecordTheirHostPhases()
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string architecture = "qwen35-phases-" + Guid.NewGuid().ToString("N");
        var phases = new System.Collections.Concurrent.ConcurrentQueue<(string Phase, string Rows)>();
        using var meters = new System.Diagnostics.Metrics.MeterListener
        {
            InstrumentPublished = (instrument, listener) =>
            {
                if (instrument.Meter.Name == TensorSharp.Runtime.Telemetry.TensorSharpTelemetry.MeterName
                    && instrument.Name == "tensorsharp.engine.step.phase.duration")
                    listener.EnableMeasurementEvents(instrument);
            },
        };
        meters.SetMeasurementEventCallback<double>((_, seconds, tags, _) =>
        {
            string arch = null, phase = null, rows = null;
            foreach (var tag in tags)
            {
                if (tag.Key == TensorSharp.Runtime.Telemetry.TelemetryAttributes.ModelArchitecture) arch = tag.Value as string;
                else if (tag.Key == TensorSharp.Runtime.Telemetry.TelemetryAttributes.StepPhase) phase = tag.Value as string;
                else if (tag.Key == TensorSharp.Runtime.Telemetry.TelemetryAttributes.RowsBucket) rows = tag.Value as string;
            }
            if (arch == architecture && seconds >= 0)
                phases.Enqueue((phase, rows));
        });
        meters.Start();

        using (var model = Create(Fixture("dense")))
        {
            model.Config.Architecture = architecture;
            var cfg = new SchedulerConfig
            {
                MaxNumBatchedTokens = 256,
                MaxNumRunningSequences = 4,
                MaxPrefillChunkSize = 32,
                NumBlocks = 32,
                BlockSize = 16,
                EnablePrefixCaching = false,
            };
            using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);
            int[][] prompts =
            {
                Enumerable.Range(0, 19).Select(i => 65 + (i * 7) % 50).ToArray(),
                Enumerable.Range(0, 27).Select(i => 97 + (i * 5) % 26).ToArray(),
            };
            var handles = prompts.Select((p, i) => engine.SubmitRequest(
                new SequenceState($"req{i}", p, 6, cfg.BlockSize, SamplingConfig.Greedy))).ToList();
            await Task.WhenAll(handles.Select(async h =>
            {
                await foreach (int _ in h.Tokens.ReadAllAsync()) { }
                await h.Completion;
            })).WaitAsync(TimeSpan.FromMinutes(3));
            Assert.True(model.BatchedForwardCount > 0, "the requests must be served in unified steps");
        }

        Assert.Contains(phases, p => p.Phase == "sample" && p.Rows == "2-4");
        Assert.Contains(phases, p => p.Phase == "issue" && p.Rows == "2-4");
        Assert.Contains(phases, p => p.Phase == "issue" && p.Rows != "1" && p.Rows != "2-4");
        Assert.All(phases, p => Assert.Contains(p.Phase, new[] { "sample", "issue" }));
    }

    private static int ArgMax(float[] logits)
    {
        int best = 0;
        for (int v = 1; v < logits.Length; v++) if (logits[v] > logits[best]) best = v;
        return best;
    }
}
