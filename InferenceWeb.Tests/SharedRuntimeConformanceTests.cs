// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Proof, per shared-runtime family, of every mechanism the runtime claims for it: tensor
// parallelism, whole-layer stages and their hybrid each match the single-device run through
// prefill, decode, a truncate-and-refill and a reset, and concurrent requests served by the
// engine (continuous batching) match each request served alone, in every placement.
//
// The devices are simulated (SimulatedDeviceGroup: independent host allocators, exact host
// reduction), so this exercises the runtime's placement, sharding, per-rank execution,
// reduction and stage transfers on any machine. It is functional evidence only: it says
// nothing about GPU kernels or multi-GPU performance, which need real devices.
using Microsoft.Extensions.Logging.Abstractions;

using TensorSharp;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

public sealed class SharedRuntimeConformanceTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-shared-runtime-" + Guid.NewGuid().ToString("N"));
    private readonly KvCacheDtype _dtype = KvCacheDtypeConfig.Current;
    private readonly bool _dtypeExplicit = KvCacheDtypeConfig.IsExplicitlySet;

    public SharedRuntimeConformanceTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        KvCacheDtypeConfig.RestoreForTests(_dtype, _dtypeExplicit);
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private const int Layers = 6;

    /// <summary>Every family on the shared runtime, with its synthetic fixture. A family that
    /// declares <c>ServingContract.SharedRuntime</c> must appear here
    /// (<see cref="ServingContractConformanceTests.EverySharedRuntimeFamilyHasAConformanceFixture"/>).</summary>
    public static TheoryData<string> Families => new() { "mistral3", "hunyuan-dense", "qwen3vl" };

    /// <summary>The family ids in <see cref="Families"/>.</summary>
    public static readonly string[] FamilyIds = { "mistral3", "hunyuan-dense", "qwen3vl" };

    /// <summary>Shared-runtime families the script here cannot drive, with the suite that proves the
    /// same mechanisms on the family's own synthetic fixtures. A recurrent stack rewinds only to its
    /// length or to zero, so the script's truncate-and-refill step does not apply to Qwen 3.5.</summary>
    public static readonly IReadOnlyDictionary<string, Type> DedicatedSuites = new Dictionary<string, Type>(StringComparer.Ordinal)
    {
        ["qwen35"] = typeof(Qwen35RuntimeTests),
        // LFM3's short-conv layers rewind only to their length or to zero, as Qwen 3.5's recurrent layers do.
        ["lfm3"] = typeof(Lfm3RuntimeTests),
        // Muse-Glimmer's sliding-window rings bound its rewinds (RingsKeepTruncation); its suites lap the rings on purpose.
        ["muse-glimmer"] = typeof(MuseGlimmerRuntimeConformanceTests),
        // Inkling 2 keeps short-conv state on every layer; it declares its placements and expert offload as gaps.
        ["inkling2"] = typeof(Inkling2RuntimeTests),
        // Raon's RDA layers rewind only to their length or to zero; it declares its placements as gaps.
        ["raon1nano"] = typeof(RaonRuntimeTests),
        // Gemma 4's fixtures are its own (per-layer embeddings, KV sharing, two head widths, fused GeGLU experts).
        ["gemma4"] = typeof(Gemma4RuntimeTests),
        // DiffusionGemma's canvas steps (block rows over the prompt's cache) on the Gemma 4 backbone; it does not decode.
        ["diffusion-gemma"] = typeof(DiffusionGemmaRuntimeTests),
        // Nemotron-H's Mamba-2 layers rewind only to their length or to zero; its fixtures mix Mamba-2, attention and FFN blocks.
        ["nemotron_h"] = typeof(NemotronRuntimeTests),
        // GPT-OSS's fixtures are its own (sinks, biased projections, MXFP4 experts with biases, alternating sliding windows).
        ["gptoss"] = typeof(GptOssRuntimeTests),
        // Qwen3.8-Flash-Next's checkpoint is its own (hyper-connections, the PLE n-gram layer, QSA, Gated DeltaNet, shared-expert MoE).
        ["qwen4exp"] = typeof(Qwen4ExpRuntimeConformanceTests),
    };

    /// <summary>Shared-runtime families that run whole-forward graphs only (StandardDecoderModel.RequiresGraphs): their
    /// dedicated suite has no per-op path to compare the graphs with. A family leaves the list when it gains one.</summary>
    public static readonly IReadOnlySet<string> GraphOnlyFamilies = new HashSet<string>(StringComparer.Ordinal) { "inkling2", "raon1nano", "gemma4", "diffusion-gemma", "nemotron_h", "gptoss", "qwen4exp" };

    /// <summary>The mechanism test of routed experts on the host, which a dedicated suite of a dense family (no routed experts)
    /// does not declare.</summary>
    public const string HostExpertsMechanism = "HostExpertsMatchDeviceExperts";

    /// <summary>The tests a dedicated suite declares: one per mechanism the runtime claims.</summary>
    public static readonly string[] MechanismTests =
    {
        nameof(PlacementMatchesOneDevice),                 // tensor parallelism, whole-layer stages, their hybrid
        nameof(SlotBatchesMatchEachSequenceAlone),         // batched slots in every placement
        "ConcurrentRequestsAreBatched",                    // continuous batching in every placement
        HostExpertsMechanism,                              // routed experts on the host (families with routed experts)
        nameof(CheckpointsContinueExactly),
        nameof(SpeculativeDecodingMatchesPlainGreedy),
        nameof(WholeForwardGraphMatchesThePerOpPath),
    };

    /// <summary>(tensor parallel, stages) placements: TP, local layer split (even and uneven
    /// stage sizes), and the hybrid.</summary>
    public static TheoryData<string, int, int> Placements()
    {
        var data = new TheoryData<string, int, int>();
        foreach (string family in new[] { "mistral3", "hunyuan-dense", "qwen3vl" })
        {
            data.Add(family, 2, 1);
            data.Add(family, 1, 2);
            data.Add(family, 1, 4);
            data.Add(family, 2, 2);
            data.Add(family, 2, 3);
        }
        return data;
    }

    private string Fixture(string family, int keyLength = DenseDecoderSyntheticModelBuilder.HeadDim) => (family switch
    {
        "mistral3" => new DenseDecoderSyntheticModelBuilder { Architecture = "mistral3", LayerCount = Layers, KeyLength = keyLength },
        "hunyuan-dense" => new DenseDecoderSyntheticModelBuilder
        {
            Architecture = "hunyuan-dense", PreTokenizer = "hunyuan-dense", IncludeMistralControlTokens = false,
            IncludeQkNorms = true, RopeBase = 10000f, LayerCount = Layers, KeyLength = keyLength,
        },
        // Qwen3-VL MoE text (its DeepStack and M-RoPE media paths are Qwen3VLRuntimeTests'): routed experts, QK-norm.
        "qwen3vl" => new DenseDecoderSyntheticModelBuilder
        {
            Architecture = "qwen3vlmoe", PreTokenizer = "qwen2", IncludeMistralControlTokens = false,
            IncludeQkNorms = true, RopeBase = 10000f, LayerCount = Layers, KeyLength = keyLength, MoeExperts = 4, MoeTopK = 2,
        },
        _ => throw new ArgumentOutOfRangeException(nameof(family)),
    }).Write(Path.Combine(_dir, $"{family}-{keyLength}.gguf"));

    private static ModelBase Create(string family, string path, int tp, int stages, BackendType backend = BackendType.Cpu)
    {
        ITensorParallelGroup group = tp > 1 ? new SimulatedDeviceGroup(tp * stages) : null;
        return family switch
        {
            "mistral3" => new Mistral3Model(path, backend, tp, group, stages),
            "hunyuan-dense" => new HunyuanDenseModel(path, backend, tp, group, stages),
            "qwen3vl" => new Qwen3VLModel(path, backend, tp, group, stages),
            _ => throw new ArgumentOutOfRangeException(nameof(family)),
        };
    }

    /// <summary>The single-device script on a real compute backend against the managed
    /// reference: the pinned GGML backend (ggml_cpu by default; ggml_metal, ggml_cuda or
    /// ggml_vulkan in their lanes), so the runtime's backend fast paths (fused SwiGLU FFN,
    /// fused softmax, native paged attention) are covered wherever the suite runs.</summary>
    /// <summary>Tensor parallelism over two ggml_cuda GPUs as per-rank graphs run together: the same logits as one device,
    /// decode replays, and speculation.</summary>
    [GgmlTheory(BackendType.GgmlCuda, Devices = 2)]
    [MemberData(nameof(Families))]
    public void RankGraphsOnGgmlMatchOneDevice(string family) => AssertRankGraphsMatchOneDevice(family, BackendType.GgmlCuda);

    /// <summary>The same over two ggml_vulkan GPUs, whose partials sum through host staging (no device collective).</summary>
    [GgmlTheory(BackendType.GgmlVulkan, Devices = 2)]
    [MemberData(nameof(Families))]
    public void RankGraphsOnGgmlVulkanMatchOneDevice(string family) => AssertRankGraphsMatchOneDevice(family, BackendType.GgmlVulkan);

    private void AssertRankGraphsMatchOneDevice(string family, BackendType backend)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(family, keyLength: 64);
        List<float[]> single;
        using (var one = Create(family, path, 1, 1, backend))
            single = Script(one);
        using ModelBase tp = CreateGgmlTensorParallel(family, path, backend);
        var sdm = Assert.IsAssignableFrom<StandardDecoderModel>(tp);
        Assert.Equal(2, sdm.Mesh.TensorParallel);
        List<float[]> actual = Script(tp);
        Assert.True(sdm.GraphRuns > 0 && sdm.GraphReplays > 0, $"{family}: the per-rank graphs did not run and replay");
        for (int i = 0; i < single.Count; i++)
            // ggml's kernels split each row-parallel product differently per rank (1.5e-4 / 2.2e-4 measured on 2x A40).
            Assert.True(RelL2(actual[i], single[i]) <= 1e-3, $"{family} tp2 rank graphs row {i}: relL2 {RelL2(actual[i], single[i]):E2}");
        // Each rank scores its slice of the untied head, and the whole head stays on the host: a verify step's all-row logits
        // bind it on rank 0 when they first run, and speculation on these graphs still emits plain greedy's tokens.
        Assert.True(sdm.HeadSplitByVocabulary && sdm.WholeHeadStaysOnHost, $"{family}: the head is not split by vocabulary with the whole head on the host");
        SpeculativeRuntimeChecks.AssertMatchesPlainGreedy(() => CreateGgmlTensorParallel(family, path, backend), $"{family} tp2 rank graphs on {backend}",
            tieTolerance: 1e-2);
    }

    /// <summary>Speculation on per-rank graphs matches plain greedy decoding: over two simulated devices, and over the
    /// hybrid's two stages of two (each stage's verify windows as per-rank graphs on its own group).</summary>
    [Theory]
    [MemberData(nameof(Families))]
    public void SpeculationOnRankGraphsMatchesPlainGreedy(string family)
    {
        string path = Fixture(family);
        SpeculativeRuntimeChecks.AssertMatchesPlainGreedy(() => Create(family, path, 2, 1), $"{family} tp2");
        SpeculativeRuntimeChecks.AssertMatchesPlainGreedy(() => Create(family, path, 2, 2), $"{family} tp2 x stages2");
    }

    /// <summary>A GGML tensor-parallel group of two (the model builds it: no group is passed).</summary>
    private static ModelBase CreateGgmlTensorParallel(string family, string path, BackendType backend = BackendType.GgmlCuda) => family switch
    {
        "mistral3" => new Mistral3Model(path, backend, 2, null, 1),
        "hunyuan-dense" => new HunyuanDenseModel(path, backend, 2, null, 1),
        "qwen3vl" => new Qwen3VLModel(path, backend, 2, null, 1),
        _ => throw new ArgumentOutOfRangeException(nameof(family)),
    };

    /// <summary>Two whole-layer stages on two GGML devices: one graph per stage, the same logits as one
    /// device, and speculation across the stages.</summary>
    [GgmlTheory(BackendType.GgmlCuda, Devices = 2)]
    [MemberData(nameof(Families))]
    public void StageGraphsMatchOneDevice(string family)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(family);
        List<float[]> single;
        using (var one = Create(family, path, 1, 1, BackendType.GgmlCuda))
            single = Script(one);
        using var staged = Create(family, path, 1, 2, BackendType.GgmlCuda);
        List<float[]> graph = Script(staged);
        var sdm = Assert.IsAssignableFrom<StandardDecoderModel>(staged);
        Assert.True(sdm.GraphRuns > 0 && sdm.GraphReplays > 0, $"{family}: the stages did not run as replayed graphs");
        for (int i = 0; i < single.Count; i++)
            Assert.True(RelL2(graph[i], single[i]) <= 1e-3, $"{family} stage graphs row {i}: relL2 {RelL2(graph[i], single[i]):E2}");
        SpeculativeRuntimeChecks.AssertMatchesPlainGreedy(() => Create(family, path, 1, 2, BackendType.GgmlCuda), $"{family} stages", tieTolerance: 1e-2);
    }

    [Theory]
    [MemberData(nameof(Families))]
    public void PinnedGgmlBackendMatchesTheManagedReference(string family) => CompareBackend(family, TestGates.PinnedGgmlBackend);

    /// <summary>MLX (fused add+norm, MLX decode attention) against the managed reference.</summary>
    [MlxTheory]
    [MemberData(nameof(Families))]
    public void MlxMatchesTheManagedReference(string family) => CompareBackend(family, BackendType.Mlx);

    /// <summary>The same with 64-wide heads, which the backends' flash-attention kernels serve (the
    /// default 16 runs attention as explicit ops on ggml_cuda).</summary>
    [Theory]
    [MemberData(nameof(Families))]
    public void PinnedGgmlBackendFlashAttentionMatchesTheManagedReference(string family)
        => CompareBackend(family, TestGates.PinnedGgmlBackend, keyLength: 64);

    private void CompareBackend(string family, BackendType backend, int keyLength = DenseDecoderSyntheticModelBuilder.HeadDim)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(family, keyLength);
        List<float[]> reference;
        using (var managed = Create(family, path, 1, 1))
            reference = Script(managed);
        using var model = Create(family, path, 1, 1, backend);
        List<float[]> actual = Script(model);
        // Measured on an M5 Pro (2026-11-10) with the pre-runtime per-family code: MLX sits
        // 1.2e-3 (mistral3) / 1.75e-3 (hunyuan-dense) from the managed reference on these
        // fixtures, ggml_cpu below 1e-6. The runtime reproduces those outputs bit for bit
        // (MLX hunyuan-dense within 3e-7: its add+norm now uses MLX's fused kernel). ggml-cuda's
        // flash attention converts F32 K/V to F16 inside the kernel: 1.2e-3 (hunyuan-dense, RTX PRO
        // 6000, 2026-11-10); an F16 cache (the default) stores what it reads.
        double tolerance = backend switch
        {
            BackendType.Mlx => 5e-3,
            BackendType.GgmlCuda => 2.5e-3,
            _ => 1e-3,
        };
        for (int i = 0; i < reference.Count; i++)
        {
            double rel = RelL2(actual[i], reference[i]);
            Assert.True(rel <= tolerance, $"{family} on {backend} row {i}: relL2 {rel:E2}");
        }
    }

    /// <summary>Three requests' slots in batched steps (prefill chunks of different lengths, then
    /// decode, with a lone step on one slot in between) reproduce each request run alone on the
    /// primary cache: the slot path changes which rows share a forward, not what any row computes.</summary>
    [Theory]
    [MemberData(nameof(Placements))]
    public void SlotBatchesMatchEachSequenceAlone(string family, int tp, int stages)
        => CompareSlotBatches(family, tp, stages, BackendType.Cpu);

    [Theory]
    [MemberData(nameof(Families))]
    public void SlotBatchesMatchEachSequenceAloneOnTheManagedBackend(string family)
        => CompareSlotBatches(family, 1, 1, BackendType.Cpu);

    [Theory]
    [MemberData(nameof(Families))]
    public void SlotBatchesMatchEachSequenceAloneOnThePinnedGgmlBackend(string family)
        => CompareSlotBatches(family, 1, 1, TestGates.PinnedGgmlBackend);

    [MlxTheory]
    [MemberData(nameof(Families))]
    public void SlotBatchesMatchEachSequenceAloneOnMlx(string family)
        => CompareSlotBatches(family, 1, 1, BackendType.Mlx);

    private void CompareSlotBatches(string family, int tp, int stages, BackendType backend)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(family);
        // Managed rows are computed row by row: batching must not change them at all. Device
        // kernels may pick a different matmul for one row than for several.
        // ggml_cuda and ggml_vulkan pick different matmul kernels for one row than for several and
        // round their activations differently: measured 1.4e-3 / 2.2e-3 on an RTX PRO 6000 (ggml_cuda,
        // 2026-11-10), 1.25e-3 on an A40 (ggml_vulkan, hunyuan-dense, 2026-11-12).
        double tolerance = backend == BackendType.Cpu ? 1e-6
            : backend is BackendType.Mlx or BackendType.GgmlCuda or BackendType.GgmlVulkan ? 5e-3 : 1e-3;
        SlotParity.AssertBatchedSlotsMatchAlone(() => Create(family, path, tp, stages, backend), tolerance,
            $"{family} tp={tp} stages={stages} {backend}");
    }

    /// <summary>On a GGML backend every step runs as one whole-forward graph, and it computes what
    /// the per-op path computes.</summary>
    [Theory]
    [MemberData(nameof(Families))]
    public void WholeForwardGraphMatchesThePerOpPath(string family)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(family);
        List<float[]> perOp;
        using (var model = Create(family, path, 1, 1, TestGates.PinnedGgmlBackend))
        {
            ((StandardDecoderModel)model).GraphLoweringEnabled = false;
            perOp = Script(model);
            Assert.Equal(0, ((StandardDecoderModel)model).GraphRuns);
        }
        using var graph = Create(family, path, 1, 1, TestGates.PinnedGgmlBackend);
        List<float[]> rows = Script(graph);
        Assert.True(((StandardDecoderModel)graph).GraphRuns >= rows.Count, "every step must run as a whole-forward graph");
        Assert.True(((StandardDecoderModel)graph).GraphReplays > 0, "consecutive decode tokens must replay their graph");
        for (int i = 0; i < perOp.Count; i++)
        {
            double rel = RelL2(rows[i], perOp[i]);
            Assert.True(rel <= 1e-3, $"{family} row {i}: graph vs per-op relL2 {rel:E2}");
        }
    }

    /// <summary>The tensor executor (programs run node by node on a backend's tensors) computes what
    /// the backend's per-op path computes: the managed CPU here, MLX and direct CUDA in their lanes.</summary>
    [Theory]
    [MemberData(nameof(Families))]
    public void ManagedTensorExecutorMatchesThePerOpPath(string family) => CompareTensorExecutor(family, BackendType.Cpu, 1e-5);

    [MlxTheory]
    [MemberData(nameof(Families))]
    public void MlxTensorExecutorMatchesThePerOpPath(string family) => CompareTensorExecutor(family, BackendType.Mlx, 2e-3);

    [CudaTheory]
    [MemberData(nameof(Families))]
    public void CudaTensorExecutorMatchesThePerOpPath(string family) => CompareTensorExecutor(family, BackendType.Cuda, 2e-3);

    private void CompareTensorExecutor(string family, BackendType backend, double tolerance)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(family);
        List<float[]> perOp;
        using (var model = Create(family, path, 1, 1, backend))
        {
            ((StandardDecoderModel)model).GraphLoweringEnabled = false;
            perOp = Script(model);
        }
        using var graph = Create(family, path, 1, 1, backend);
        List<float[]> rows = Script(graph);
        Assert.True(((StandardDecoderModel)graph).GraphRuns >= rows.Count, $"{family} on {backend}: every step must run as a program");
        for (int i = 0; i < perOp.Count; i++)
        {
            double rel = RelL2(rows[i], perOp[i]);
            Assert.True(rel <= tolerance, $"{family} on {backend} row {i}: executor vs per-op relL2 {rel:E2}");
        }
        SpeculativeRuntimeChecks.AssertMatchesPlainGreedy(() => Create(family, path, 1, 1, backend), $"{family} {backend} executor",
            tieTolerance: backend == BackendType.Cpu ? 0 : 1e-2);
    }

    /// <summary>Greedy speculative decoding (n-gram drafts) emits what plain greedy decoding emits.</summary>
    [Theory]
    [MemberData(nameof(Families))]
    public void SpeculativeDecodingMatchesPlainGreedy(string family)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(family);
        // ggml_cuda / ggml_cpu round a verify window's activations differently than one row's.
        SpeculativeRuntimeChecks.AssertMatchesPlainGreedy(() => Create(family, path, 1, 1, TestGates.PinnedGgmlBackend), family,
            tieTolerance: TestGates.PinnedGgmlBackend is BackendType.GgmlCuda or BackendType.GgmlCpu ? 1e-2 : 0);
    }

    /// <summary>A shared-prefix checkpoint, cloned or written to a file and read back, continues exactly.</summary>
    [Theory]
    [MemberData(nameof(Families))]
    public void CheckpointsContinueExactly(string family)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(family);
        CheckpointChecks.AssertCheckpointsContinueExactly(() => Create(family, path, 1, 1), 1e-6, $"{family} managed");
        CheckpointChecks.AssertCheckpointsContinueExactly(() => Create(family, path, 1, 1, TestGates.PinnedGgmlBackend), 1e-5, $"{family} ggml", deviceCopies: true);
    }

    /// <summary>A fixed script over one cache: prefill, decode, truncate + refill, reset, prefill.</summary>
    private static List<float[]> Script(ModelBase model)
    {
        // Logits buffers may be reused by the model's next forward: keep copies.
        var rows = new List<float[]>();
        int[] prompt = Enumerable.Range(0, 17).Select(i => 40 + (i * 11) % 150).ToArray();
        rows.Add((float[])model.ForwardRefill(prompt).Clone());
        foreach (int token in new[] { 71, 72, 73, 74 })
            rows.Add((float[])model.Forward(new[] { token }).Clone());
        Assert.True(model.TryTruncateKVCache(10));
        rows.Add((float[])model.ForwardRefill(new[] { 91, 92, 93, 94, 95 }).Clone());
        rows.Add((float[])model.Forward(new[] { 96 }).Clone());
        model.ResetKVCache();
        rows.Add((float[])model.ForwardRefill(new[] { 33, 34, 35, 36, 37, 38, 39 }).Clone());
        rows.Add((float[])model.Forward(new[] { 40 }).Clone());
        return rows;
    }

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
    [MemberData(nameof(Placements))]
    public void PlacementMatchesOneDevice(string family, int tp, int stages)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(family);
        List<float[]> reference;
        using (var single = Create(family, path, 1, 1))
            reference = Script(single);

        using var placed = Create(family, path, tp, stages);
        var decoder = Assert.IsAssignableFrom<StandardDecoderModel>(placed);
        Assert.Equal(stages, decoder.Mesh.Stages);
        Assert.Equal(tp, decoder.Mesh.TensorParallel);
        AssertPlacement(decoder, tp, stages);

        List<float[]> actual = Script(placed);
        // Every tensor-parallel placement runs as per-rank graphs, a hybrid's on every stage's group: the runtime's one
        // lowering of the family, whatever the mesh.
        if (tp > 1)
            Assert.True(decoder.RunsRankGraphs && decoder.GraphRuns > 0, $"{family} tp={tp} stages={stages}: the per-rank graphs did not run");
        // Whole-layer stages run the very same ops on another allocator: exact. Tensor
        // parallelism splits each row-parallel sum into per-rank partials, which changes
        // only the summation order.
        double tolerance = tp == 1 ? 0 : 1e-4;
        for (int i = 0; i < reference.Count; i++)
        {
            double rel = RelL2(actual[i], reference[i]);
            Assert.True(rel <= tolerance, $"{family} tp={tp} stages={stages} row {i}: relL2 {rel:E2} > {tolerance:E0}");
        }
    }

    /// <summary>The tensor executors' compiled per-rank plans in lockstep (TensorGraphExecutor.RunRanked: direct CUDA's
    /// decode steps that advance) answer as one device, as the streamed ranks do: every step of the script so.</summary>
    [Theory]
    [InlineData("mistral3", 2, 1)]
    [InlineData("qwen3vl", 2, 1)]
    [InlineData("mistral3", 2, 2)]
    public void CompiledRankPlansMatchOneDevice(string family, int tp, int stages)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(family);
        List<float[]> reference;
        using (var single = Create(family, path, 1, 1))
            reference = Script(single);
        using var placed = Create(family, path, tp, stages);
        var decoder = Assert.IsAssignableFrom<StandardDecoderModel>(placed);
        decoder.CompiledRankedStepsForTests = true;
        List<float[]> actual = Script(placed);
        Assert.True(decoder.RunsRankGraphs && decoder.GraphRuns > 0, $"{family} tp={tp} stages={stages}: the per-rank plans did not run");
        for (int i = 0; i < reference.Count; i++)
        {
            double rel = RelL2(actual[i], reference[i]);
            Assert.True(rel <= 1e-4, $"{family} tp={tp} stages={stages} compiled row {i}: relL2 {rel:E2}");
        }
    }

    private static void AssertPlacement(StandardDecoderModel model, int tp, int stages)
    {
        Tensor[][] caches = model.KeyCachesForTests;
        Assert.Equal(Layers, caches.Length);
        var stageAllocators = new List<HashSet<IAllocator>>();
        for (int s = 0; s < stages; s++) stageAllocators.Add(new HashSet<IAllocator>());
        for (int l = 0; l < Layers; l++)
        {
            Assert.Equal(tp, caches[l].Length);
            int stage = model.Mesh.StageOf(l);
            foreach (Tensor cache in caches[l])
            {
                // Each rank holds its share of the KV heads, on its own device.
                Assert.Equal(DenseDecoderSyntheticModelBuilder.NumKvHeads / tp, cache.Sizes[0]);
                stageAllocators[stage].Add(cache.Storage.Allocator);
            }
        }
        // Every stage uses its own tp devices, and no device serves two stages.
        Assert.All(stageAllocators, set => Assert.Equal(tp, set.Count));
        Assert.Equal(tp * stages, stageAllocators.SelectMany(s => s).Distinct().Count());
    }

    [Theory]
    [MemberData(nameof(Placements))]
    public Task ConcurrentRequestsAreBatchedInEveryPlacement(string family, int tp, int stages)
        => RunConcurrentRequests(family, tp, stages);

    [Theory]
    [MemberData(nameof(Families))]
    public Task ConcurrentRequestsAreBatchedOnOneDevice(string family) => RunConcurrentRequests(family, 1, 1);

    private async Task RunConcurrentRequests(string family, int tp, int stages)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(family);
        const int maxNewTokens = 8;
        var prompts = new List<int[]>
        {
            Enumerable.Range(0, 19).Select(i => 65 + (i * 7) % 50).ToArray(),
            Enumerable.Range(0, 27).Select(i => 97 + (i * 5) % 26).ToArray(),
            Enumerable.Range(0, 13).Select(i => 48 + (i * 3) % 10).ToArray(),
        };

        // Reference: greedy decode, one prompt at a time, on one device.
        var expected = new List<int[]>();
        using (var reference = Create(family, path, 1, 1))
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

        using var model = Create(family, path, tp, stages);
        var batched = Assert.IsAssignableFrom<IBatchedPagedModel>(model);
        Assert.True(batched.SupportsPerSequenceFusedForward, "continuous batching must survive every local placement");
        var caps = ExecutionCapabilities.FromModel(model);
        Assert.True(caps.SupportsPerSequenceFusedForward);

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
        })).WaitAsync(TimeSpan.FromMinutes(2));

        Assert.True(((StandardDecoderModel)model).BatchedForwardCount > 0,
            "the engine must serve these requests in batched steps over their slots, not one at a time");
        for (int i = 0; i < prompts.Count; i++)
        {
            int n = Math.Min(outputs[i].Length, expected[i].Length);
            Assert.True(n > 0, $"request {i} produced nothing");
            Assert.Equal(expected[i].Take(n), outputs[i].Take(n));
        }
    }

    [Theory]
    [InlineData("hunyuan-dense", 2, 1)]
    [InlineData("mistral3", 1, 2)]
    [InlineData("mistral3", 2, 2)]
    public void SnapshotsRoundTripAndCarryTheMeshShape(string family, int tp, int stages)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(family);
        using var model = Create(family, path, tp, stages);
        int[] prompt = Enumerable.Range(0, 12).Select(i => 50 + i).ToArray();
        model.ForwardRefill(prompt);
        float[] expectedNext = (float[])model.Forward(new[] { 77 }).Clone();

        model.ResetKVCache();
        model.ForwardRefill(prompt);
        Assert.True(model.SupportsKVStateSnapshot);
        var block = new byte[model.ComputeKVBlockByteSize(prompt.Length)];
        Assert.True(model.TryExtractKVBlock(0, prompt.Length, block));
        model.ResetKVCache();
        Assert.True(model.TryInjectKVBlock(0, prompt.Length, block));
        Assert.Equal(expectedNext, model.Forward(new[] { 77 }));
        Assert.Contains($"|mesh={stages}x{tp}", model.KVStateFingerprint, StringComparison.Ordinal);
    }

    private static int ArgMax(float[] logits)
    {
        int best = 0;
        for (int v = 1; v < logits.Length; v++) if (logits[v] > logits[best]) best = v;
        return best;
    }
}
