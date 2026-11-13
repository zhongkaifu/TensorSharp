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

    /// <summary>The factory's route: the shared runtime serves every checkpoint on every backend and placement; only a
    /// dense NVFP4 checkpoint on one direct-CUDA GPU leaves it for the direct engine.</summary>
    [Theory]
    [InlineData("dense", BackendType.GgmlCuda, 1)]
    [InlineData("dense", BackendType.GgmlMetal, 1)]
    [InlineData("dense", BackendType.GgmlVulkan, 1)]
    [InlineData("dense", BackendType.GgmlCuda, 2)]
    [InlineData("dense", BackendType.Cuda, 1)]
    [InlineData("dense", BackendType.Cpu, 1)]
    [InlineData("dense", BackendType.Mlx, 1)]
    [InlineData("moe", BackendType.Cuda, 1)]
    [InlineData("moe", BackendType.Mlx, 1)]
    public void TheRuntimeServesEveryNonNvfp4Load(string variant, BackendType backend, int tp)
    {
        using var probe = new GgufFile(Fixture(variant));
        var context = new TensorSharp.Models.Architecture.ModelCreateContext(probe.FilePaths[0], backend, probe, tp);
        Assert.False(Qwen35CudaModel.ServesLoad(context));
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
    // disagree by 2.5e-3 on this fixture, and ggml vs the managed kernels by 5e-2.
    [InlineData("moe-quantized", 3e-2)]
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
        // 2.52e-2). With the rotations left out the first row is 1.2 away.
        double tolerance = backend is BackendType.GgmlCuda or BackendType.GgmlVulkan ? 3.5e-2 : 2.5e-2;
        for (int i = 0; i < legacy.Count; i++)
        {
            double rel = RelL2(actual[i], legacy[i]);
            Assert.True(rel <= tolerance, $"row {i}: relL2 {rel:E2} from Qwen35Model");
        }
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
            // BF16 experts: ggml-cuda rounds the activations to BF16 (1.4e-2 measured on an RTX PRO 6000).
            double tolerance = variant == "moe" ? 2e-2 : 3e-2;
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

    private void AssertHostExpertsMatchDevice(BackendType backend, bool tier)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture("moe");
        List<float[]> reference;
        using (var device = Create(path, backend: backend))
            reference = Script(device);
        using var cpuMoe = MoeCpuOffloadConfig.AllLayersForCurrentFlowInTests();
        using var offloaded = Create(path, backend: backend);
        if (tier)
            for (int l = 0; l < offloaded.Config.NumLayers; l++)
                offloaded.SetExpertTierForTests(l, Enumerable.Range(0, offloaded.ExpertCountForTests / 2).ToArray());
        List<float[]> actual = Script(offloaded);
        (long hostRoutes, long streamed) = offloaded.ExpertHostCountsForTests;
        Assert.True(offloaded.GraphRuns > 0, "the offloaded model did not run as graphs");
        Assert.True(hostRoutes > 0 && streamed > 0, $"host routes {hostRoutes}, streamed steps {streamed}");
        // The host's kernels round as ggml's CPU backend does (activations to Q8_K / BF16 against the device's).
        for (int i = 0; i < reference.Count; i++)
        {
            double rel = RelL2(actual[i], reference[i]);
            Assert.True(rel <= 2.5e-2, $"{backend} --cpu-moe tier={tier}, row {i}: relL2 {rel:E2}");
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

    /// <summary>The same over per-rank graphs on two ggml_cuda GPUs: every rank's step advances in place and the ranks run
    /// together without a trace.</summary>
    [GgmlTheory(BackendType.GgmlCuda, Devices = 2)]
    [InlineData("dense")]
    [InlineData("moe-tp")]
    public void AdvancedRankedDecodeStepsMatchTracedOnes(string variant)
        => AssertAdvancedDecodeMatchesTraced(variant, path => new Qwen35RuntimeModel(path, BackendType.GgmlCuda, 2, null, 1));

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
        await SpeculativeRuntimeChecks.AssertEngineSpeculationMatchesPlain(() => Create(path, backend: backend), options,
            $"qwen35 {variant} {speculator} {backend}");
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
