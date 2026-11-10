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
        double tolerance = RoundsActivations(backend) ? 2.5e-2 : 5e-3;
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

    /// <summary>A shared-prefix checkpoint (attention rows and Gated DeltaNet state), cloned or written
    /// to a file and read back, continues exactly as a fresh run.</summary>
    [Theory]
    [InlineData("dense")]
    [InlineData("moe")]
    public void CheckpointsContinueExactly(string variant)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(variant);
        CheckpointChecks.AssertCheckpointsContinueExactly(() => Create(path, backend: TestGates.PinnedGgmlBackend), 1e-5, $"qwen35 {variant}");
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
