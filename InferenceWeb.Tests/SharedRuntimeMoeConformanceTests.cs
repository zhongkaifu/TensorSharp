// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Routed experts on the shared decoder runtime, through a minimal test-only family: a family
// that writes nothing but its rotary step gets MoE layers, expert offload (--n-cpu-moe /
// --cpu-moe), expert parallelism under --tp, layer split, the hybrid and batched serving -
// the template every future MoE family inherits. Simulated devices; functional evidence only.
using Microsoft.Extensions.Logging.Abstractions;

using TensorSharp;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

[Collection("MoeCpuOffloadConfig")]
public sealed class SharedRuntimeMoeConformanceTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-shared-moe-" + Guid.NewGuid().ToString("N"));
    private readonly KvCacheDtype _dtype = KvCacheDtypeConfig.Current;
    private readonly bool _dtypeExplicit = KvCacheDtypeConfig.IsExplicitlySet;
    private const int Layers = 6;

    public SharedRuntimeMoeConformanceTests()
    {
        Directory.CreateDirectory(_dir);
        MoeCpuOffloadConfig.Reset();
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
    }

    public void Dispose()
    {
        MoeCpuOffloadConfig.Reset();
        KvCacheDtypeConfig.RestoreForTests(_dtype, _dtypeExplicit);
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    /// <summary>The whole family: configuration, tokenizer and NeoX rotary. Everything else -
    /// attention, routed and shared experts, placement, offload, batching - is the runtime's.</summary>
    private sealed class TinyMoeModel : StandardDecoderModel
    {
        public TinyMoeModel(string path, BackendType backend, int tp = 1, ITensorParallelGroup group = null, int stages = 1)
            : base(path, backend, tp, group, stages)
        {
            Config = new ModelConfig { Architecture = _gguf.GetString("general.architecture") };
            ParseBaseConfig();
            ParseTokenizer();
            InitializeDecoder(Config.HeadDim, Config.HeadDim, "tiny-moe");
        }

        protected override void RotateQueryKey(RankScope rank, int layer, ref Tensor q, ref Tensor k, TokenPositions positions)
        {
            Rotate(q, rank.LocalHeads, positions);
            Rotate(k, rank.LocalKvHeads, positions);
        }

        private void Rotate(Tensor data, int heads, TokenPositions positions)
        {
            using Tensor pos = CreateIntTensorOn(data.Storage.Allocator, positions.PerHead(heads), positions.Count * heads);
            using Tensor view = data.View(1, positions.Count, heads, KeyDim);
            Ops.RoPEEx(view, view, pos, KeyDim, 2, 0, Config.RopeBase, 1.0f, 0f, 1f, 0f, 0f);
        }
    }

    private string Fixture(bool shared = true) => new DenseDecoderSyntheticModelBuilder
    {
        Architecture = "tiny-moe", IncludeMistralControlTokens = false, RopeBase = 10000f,
        LayerCount = Layers, MoeExperts = 8, MoeTopK = 2, MoeSharedExpert = shared,
    }.Write(Path.Combine(_dir, $"tiny-moe-{shared}.gguf"));

    private static TinyMoeModel Create(string path, BackendType backend, int tp, int stages)
        => new(path, backend, tp, tp > 1 ? new SimulatedDeviceGroup(tp * stages) : null, stages);

    private static List<float[]> Script(ModelBase model)
    {
        var rows = new List<float[]>();
        rows.Add((float[])model.ForwardRefill(Enumerable.Range(0, 15).Select(i => 40 + (i * 13) % 150).ToArray()).Clone());
        foreach (int token in new[] { 70, 71, 72 })
            rows.Add((float[])model.Forward(new[] { token }).Clone());
        Assert.True(model.TryTruncateKVCache(9));
        rows.Add((float[])model.ForwardRefill(new[] { 90, 91, 92, 93 }).Clone());
        rows.Add((float[])model.Forward(new[] { 94 }).Clone());
        return rows;
    }

    private static double RelL2(float[] a, float[] b)
    {
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++) { double d = a[i] - b[i]; num += d * d; den += (double)b[i] * b[i]; }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }

    private static void AssertClose(List<float[]> expected, List<float[]> actual, double tolerance, string what)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (int i = 0; i < expected.Count; i++)
        {
            double rel = RelL2(actual[i], expected[i]);
            Assert.True(rel <= tolerance, $"{what} row {i}: relL2 {rel:E2} > {tolerance:E0}");
        }
    }

    private List<float[]> Reference(string path, BackendType backend = BackendType.Cpu)
    {
        MoeCpuOffloadConfig.Reset();
        using var model = Create(path, backend, 1, 1);
        return Script(model);
    }

    /// <summary>(tp, stages, offloaded layers; -1 = all) - every placement, with and without
    /// experts in host memory.</summary>
    public static TheoryData<int, int, int> Placements => new()
    {
        { 1, 1, -1 }, { 1, 1, 3 },
        { 2, 1, 0 }, { 2, 1, -1 }, { 2, 1, 2 },
        { 1, 2, 0 }, { 1, 3, -1 },
        { 2, 2, 0 }, { 2, 2, 4 },
    };

    [Theory]
    [MemberData(nameof(Placements))]
    public void ExpertsMatchOneDeviceInEveryPlacementWithAndWithoutOffload(int tp, int stages, int offloaded)
    {
        string path = Fixture();
        List<float[]> reference = Reference(path);

        if (offloaded < 0) MoeCpuOffloadConfig.SetAllLayers();
        else if (offloaded > 0) MoeCpuOffloadConfig.SetLayers(offloaded);
        using var model = Create(path, BackendType.Cpu, tp, stages);
        Assert.Equal(stages, model.Mesh.Stages);
        Assert.Equal(tp, model.Mesh.TensorParallel);
        // Host experts run the managed kernels directly, device experts through the backend's
        // matmul (the same managed kernels here), and expert parallelism reorders the sum.
        AssertClose(reference, Script(model), 1e-4, $"tp={tp} stages={stages} offloaded={offloaded}");
    }

    [Fact]
    public void AnIndivisibleDegreeIsRefusedNamingWhatDoesNotDivide()
    {
        var error = Assert.Throws<NotSupportedException>(() => Create(Fixture(), BackendType.Cpu, 4, 1));
        Assert.Contains("2 KV heads must each divide by 4", error.Message, StringComparison.Ordinal);
    }

    [Fact]
    public void ARoutedLayerWithoutASharedExpertWorksToo()
    {
        string path = Fixture(shared: false);
        List<float[]> reference = Reference(path);
        MoeCpuOffloadConfig.SetLayers(2);
        using var model = Create(path, BackendType.Cpu, 2, 2);
        AssertClose(reference, Script(model), 1e-4, "no shared expert, hybrid 2x2, 2 layers offloaded");
    }

    [Fact]
    public void GgmlCpuStackedExpertsMatchWithAndWithoutHostOffload()
    {
        if (TestGates.PinnedGgmlBackend != BackendType.GgmlCpu) return; // one GGML backend per process
        string path = Fixture();
        List<float[]> device = Reference(path, BackendType.GgmlCpu);
        MoeCpuOffloadConfig.SetAllLayers();
        using var model = Create(path, BackendType.GgmlCpu, 1, 1);
        // The same ggml mul_mat_id kernel, bound to ggml's CPU backend instead of the device.
        AssertClose(device, Script(model), 1e-5, "ggml_cpu, every layer's experts on the host");
        // And the stacked kernel agrees with the managed reference on this fixture.
        AssertClose(Reference(path), device, 2e-2, "ggml_cpu stacked experts vs managed");
    }

    [Theory]
    [InlineData(1, 1, 0)]
    [InlineData(2, 1, -1)]
    [InlineData(2, 2, 3)]
    public async Task ConcurrentRequestsAreBatchedWithExperts(int tp, int stages, int offloaded)
    {
        string path = Fixture();
        const int maxNewTokens = 6;
        var prompts = new List<int[]>
        {
            Enumerable.Range(0, 17).Select(i => 65 + (i * 7) % 50).ToArray(),
            Enumerable.Range(0, 23).Select(i => 97 + (i * 5) % 26).ToArray(),
        };
        var expected = new List<int[]>();
        MoeCpuOffloadConfig.Reset();
        using (var reference = Create(path, BackendType.Cpu, 1, 1))
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

        if (offloaded < 0) MoeCpuOffloadConfig.SetAllLayers();
        else if (offloaded > 0) MoeCpuOffloadConfig.SetLayers(offloaded);
        using var model = Create(path, BackendType.Cpu, tp, stages);
        var cfg = new SchedulerConfig
        {
            MaxNumBatchedTokens = 256, MaxNumRunningSequences = 4, MaxPrefillChunkSize = 8,
            NumBlocks = 32, BlockSize = 16, EnablePrefixCaching = true, DecodeQuantumTokens = 2,
        };
        using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);
        var handles = prompts.Select((p, i) => engine.SubmitRequest(
            new SequenceState($"moe{i}", p, maxNewTokens, cfg.BlockSize, SamplingConfig.Greedy))).ToList();
        var outputs = await Task.WhenAll(handles.Select(async h =>
        {
            var tokens = new List<int>();
            await foreach (int t in h.Tokens.ReadAllAsync()) tokens.Add(t);
            await h.Completion;
            return tokens.ToArray();
        })).WaitAsync(TimeSpan.FromMinutes(2));

        Assert.True(model.BatchedForwardCount > 0);
        for (int i = 0; i < prompts.Count; i++)
        {
            int n = Math.Min(outputs[i].Length, expected[i].Length);
            Assert.True(n > 0);
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
