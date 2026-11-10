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
    [InlineData("moe")]
    public void PinnedGgmlBackendMatchesTheManagedReference(string variant) => CompareBackend(variant, TestGates.PinnedGgmlBackend);

    [MlxTheory]
    [InlineData("dense")]
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
        double tolerance = backend == BackendType.GgmlCpu ? 2.5e-2 : 5e-3;
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
        double tolerance = backend == BackendType.Cpu ? 1e-6 : backend == BackendType.GgmlCpu ? 2.5e-2 : 5e-3;
        SlotParity.AssertBatchedSlotsMatchAlone(() => Create(path, tp, stages, backend), tolerance,
            $"qwen35 {variant} tp={tp} stages={stages} {backend}");
    }

    /// <summary>On a GGML backend every step (attention, Gated DeltaNet with its slot state, MoE)
    /// runs as one whole-forward graph and computes what the per-op path computes.</summary>
    [Theory]
    [InlineData("dense")]
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
        double tolerance = TestGates.PinnedGgmlBackend == BackendType.GgmlCpu ? 2.5e-2 : 5e-3;
        for (int i = 0; i < perOp.Count; i++)
        {
            double rel = RelL2(rows[i], perOp[i]);
            Assert.True(rel <= tolerance, $"{variant} row {i}: graph vs per-op relL2 {rel:E2}");
        }
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
