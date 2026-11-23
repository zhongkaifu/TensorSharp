// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Qwen3-VL on the shared decoder runtime (Qwen3VLModel), on a tiny synthetic qwen3vlmoe / qwen3vl GGUF: both converter
// labels load the family; a media span's DeepStack taps join the residual stream (all-zero taps change nothing, real taps
// do) and never outlive their sequence; a step that cannot run as a graph while taps wait refuses instead of dropping
// them; the interleaved M-RoPE table rotates the same on the graph and per-op paths; routed experts on the host answer
// as the experts on the device. The placements, batching, checkpoints and speculation are the shared-runtime script's
// (SharedRuntimeConformanceTests, family "qwen3vl").
using System;
using System.IO;
using System.Linq;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Models.Architecture;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests;

public class Qwen3VLRuntimeTests : IDisposable
{
    private const int Hidden = DenseDecoderSyntheticModelBuilder.Hidden;
    private const int Taps = 3;
    private const int MediaAt = 6, MediaRows = 8;
    private static readonly int[] Prompt = Enumerable.Range(0, 24).Select(i => 40 + (i * 13) % 170).ToArray();

    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-qwen3vl-" + Guid.NewGuid().ToString("N"));

    public Qwen3VLRuntimeTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private string Fixture(string arch = "qwen3vlmoe") => new DenseDecoderSyntheticModelBuilder
    {
        Architecture = arch, PreTokenizer = "qwen2", IncludeMistralControlTokens = false, IncludeQkNorms = true,
        MoeExperts = arch == "qwen3vlmoe" ? 4 : 0, MoeTopK = 2, RopeBase = 10000f, LayerCount = 4,
        RopeSections = new[] { 4, 2, 2, 0 }, DeepStackLayers = Taps,
    }.Write(Path.Combine(_dir, arch + ".gguf"));

    /// <summary>Managed CPU in the cpu lane; a GPU lane's pinned GGML backend elsewhere.</summary>
    private static BackendType Backend =>
        TestGates.PinnedGgmlBackend is BackendType.GgmlCpu ? BackendType.Cpu : TestGates.PinnedGgmlBackend;

    private static float[] Random(int count, int seed, float scale = 0.5f)
    {
        var random = new Random(seed);
        return Enumerable.Range(0, count).Select(_ => (float)(random.NextDouble() - 0.5) * scale).ToArray();
    }

    /// <summary>A span's rows as the tower hands them over: [rows, hidden] without taps, [rows, hidden * 4] with them.</summary>
    private static Tensor Rows(Qwen3VLModel model, float[] main, float[] taps)
    {
        int width = taps == null ? Hidden : Hidden * (1 + Taps);
        var values = new float[MediaRows * width];
        for (int r = 0; r < MediaRows; r++)
        {
            Array.Copy(main, r * Hidden, values, r * width, Hidden);
            if (taps != null)
                for (int t = 0; t < Taps; t++)
                    Array.Copy(taps, (t * MediaRows + r) * Hidden, values, r * width + (1 + t) * Hidden, Hidden);
        }
        var tensor = new Tensor(model.DirectAllocatorForTelemetry, DType.Float32, MediaRows, width);
        tensor.SetElementsAsFloat(values);
        return tensor;
    }

    private float[] RunMedia(string path, float[] taps, bool graphs = true, int[] table = null)
    {
        using var model = Assert.IsType<Qwen3VLModel>(ModelBase.Create(path, Backend));
        model.GraphLoweringEnabled = graphs;
        if (table != null) ((IMRoPEPositionSink)model).SetMRoPEPositions(table);
        model.SetVisionEmbeddings(Rows(model, Random(MediaRows * Hidden, 1), taps), MediaAt);
        return (float[])model.Forward(Prompt).Clone();
    }

    private static double RelL2(float[] a, float[] b)
    {
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++) { double d = a[i] - b[i]; num += d * d; den += (double)b[i] * b[i]; }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }

    [Theory]
    [InlineData("qwen3vlmoe")]
    [InlineData("qwen3vl")]
    public void BothConverterLabelsLoadTheFamily(string arch)
    {
        using var model = ModelBase.Create(Fixture(arch), Backend);
        Assert.IsType<Qwen3VLModel>(model);
        Assert.Contains("|qwen3vl|rope=16|deepstack=3", model.KVStateFingerprint, StringComparison.Ordinal);
        Assert.True(model.SupportsKVStateSnapshot);
    }

    [Fact]
    public void AllZeroTapsMatchMediaWithoutTaps()
    {
        using var kv = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        string path = Fixture();
        float[] plain = RunMedia(path, taps: null);
        float[] zero = RunMedia(path, taps: new float[Taps * MediaRows * Hidden]);
        Assert.True(RelL2(zero, plain) <= 1e-6, $"all-zero taps moved the logits: relL2 {RelL2(zero, plain):E2}");
    }

    [Fact]
    public void TapsJoinTheResidualStream()
    {
        using var kv = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        string path = Fixture();
        float[] zero = RunMedia(path, taps: new float[Taps * MediaRows * Hidden]);
        float[] tapped = RunMedia(path, taps: Random(Taps * MediaRows * Hidden, 2, scale: 2f));
        Assert.True(RelL2(tapped, zero) > 1e-3, $"the taps did not reach the stack: relL2 {RelL2(tapped, zero):E2}");
    }

    [Fact]
    public void TapsDoNotOutliveTheirSequence()
    {
        using var kv = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        string path = Fixture();
        int[] text = Enumerable.Range(0, 20).Select(i => 50 + (i * 7) % 150).ToArray();
        float[] fresh;
        using (var model = ModelBase.Create(path, Backend))
            fresh = (float[])model.Forward(text).Clone();
        using var used = Assert.IsType<Qwen3VLModel>(ModelBase.Create(path, Backend));
        used.SetVisionEmbeddings(Rows(used, Random(MediaRows * Hidden, 1), Random(Taps * MediaRows * Hidden, 2, scale: 2f)), MediaAt);
        used.Forward(Prompt);
        used.ResetKVCache();
        float[] after = (float[])used.Forward(text).Clone();
        Assert.True(RelL2(after, fresh) <= 1e-6, $"a media sequence's taps reached the next one: relL2 {RelL2(after, fresh):E2}");
    }

    [Fact]
    public void APerOpStepRefusesWhileTapsWait()
    {
        string path = Fixture();
        using var model = Assert.IsType<Qwen3VLModel>(ModelBase.Create(path, Backend));
        model.GraphLoweringEnabled = false;
        model.SetVisionEmbeddings(Rows(model, Random(MediaRows * Hidden, 1), Random(Taps * MediaRows * Hidden, 2)), MediaAt);
        Assert.Throws<NotSupportedException>(() => model.Forward(Prompt));
    }

    /// <summary>A position table like the injector's for a 2 x 4 image at <see cref="MediaAt"/>: text rows at t = h = w, the
    /// media rows on their (t, h, w) grid, text after it continuing past the grid's largest coordinate.</summary>
    private static int[] Table()
    {
        var table = new int[3 * Prompt.Length];
        int p = 0;
        for (int i = 0; i < Prompt.Length; i++)
        {
            if (i >= MediaAt && i < MediaAt + MediaRows)
            {
                int k = i - MediaAt;
                table[3 * i] = MediaAt;
                table[3 * i + 1] = MediaAt + k / 4;
                table[3 * i + 2] = MediaAt + k % 4;
                p = MediaAt + 4;
                continue;
            }
            int t = i < MediaAt ? i : p++;
            table[3 * i] = table[3 * i + 1] = table[3 * i + 2] = t;
        }
        return table;
    }

    [Fact]
    public void MRoPETableRotatesTheSameOnGraphsAndPerOp()
    {
        using var kv = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        string path = Fixture();
        float[] graphs = RunMedia(path, taps: null, graphs: true, table: Table());
        float[] perOp = RunMedia(path, taps: null, graphs: false, table: Table());
        // The managed reference agrees to 1e-6; a GPU lane's kernels reorder the sums (1.6e-4 measured on ggml_cuda).
        double bound = Backend == BackendType.Cpu ? 1e-4 : 1e-3;
        Assert.True(RelL2(perOp, graphs) <= bound, $"per-op vs graphs with an M-RoPE table: relL2 {RelL2(perOp, graphs):E2}");
        float[] scalar = RunMedia(path, taps: null, graphs: true);
        Assert.True(RelL2(scalar, graphs) > 1e-4, "the table did not change the rotation of the media rows");
    }

    /// <summary>Routed experts on the host (--cpu-moe), on one device and on whole-layer stages, answer as the experts on
    /// the device.</summary>
    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    public void HostExpertsMatchDeviceExperts(int stages)
    {
        using var kv = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F32);
        string path = Fixture();
        float[] device;
        using (var model = new Qwen3VLModel(path, BackendType.Cpu))
            device = (float[])model.Forward(Prompt).Clone();
        using var cpuMoe = MoeCpuOffloadConfig.AllLayersForCurrentFlowInTests();
        using var offloaded = new Qwen3VLModel(path, BackendType.Cpu, 1, null, stages);
        float[] host = (float[])offloaded.Forward(Prompt).Clone();
        Assert.True(RelL2(host, device) <= 1e-4, $"--cpu-moe, {stages} stage(s): relL2 {RelL2(host, device):E2}");
    }
}
