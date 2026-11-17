// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Diagnostics;
using System.Linq;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.GGML;
using TensorSharp.Models.Graph;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// NetworkGraph on a real mmproj (TS_LFM3_MMPROJ: an LFM3-Omni mmproj GGUF): its vision tower's weights loaded by name and
/// run as one graph on a GGML backend and on the managed CPU, which must agree (the tower itself is the vision port's; this
/// checks the foundation carries real shapes and formats: F16/Q8_0 products, F32 conv kernels, 16/8-head GQA with 2-D rotary,
/// the 2x2 merge and the GELU projector).
/// </summary>
public sealed class NetworkGraphRealWeightsTests
{
    private const float Eps = 9.999999747378752e-06f;
    private readonly ITestOutputHelper _output;

    public NetworkGraphRealWeightsTests(ITestOutputHelper output) => _output = output;

    [ModelFact("TS_LFM3_MMPROJ", GgmlBackend = BackendType.GgmlMetal)]
    public void MetalVisionTowerMatchesManagedCpu() => Compare(GgmlBackendType.Metal, BackendType.GgmlMetal);

    [ModelFact("TS_LFM3_MMPROJ", GgmlBackend = BackendType.GgmlCpu)]
    public void GgmlCpuVisionTowerMatchesManagedCpu() => Compare(GgmlBackendType.Cpu, BackendType.GgmlCpu);

    /// <summary>The tower's speed on Metal at a 512 x 512 and a 1024 x 1024 image's patch grids (first run, then the median of
    /// five replays), against which llama.cpp's mtmd encode of the same image sizes is the reference.</summary>
    [ModelFact("TS_LFM3_MMPROJ", GgmlBackend = BackendType.GgmlMetal)]
    public void MetalVisionTowerSpeed()
    {
        string path = Environment.GetEnvironmentVariable("TS_LFM3_MMPROJ");
        using var gguf = new GgufFile(path);
        var allocator = new GgmlAllocator(new GgmlContext([0], GgmlBackendType.Metal), 0);
        try
        {
            using var weights = Load(gguf, allocator, BackendType.GgmlMetal);
            using var graph = new NetworkGraph(weights, "lfm3 vision tower");
            foreach (int grid in new[] { 32, 64 })
            {
                float[] patches = NetworkGraphTests.Rand(new Random(grid), 768 * grid * grid);
                var sw = Stopwatch.StartNew();
                graph.Run(b => b.Output(Tower(b, weights, patches, grid, grid)));
                double first = sw.Elapsed.TotalMilliseconds;
                var ms = new double[5];
                for (int i = 0; i < ms.Length; i++)
                {
                    sw.Restart();
                    graph.Run(b => b.Output(Tower(b, weights, patches, grid, grid)));
                    ms[i] = sw.Elapsed.TotalMilliseconds;
                }
                Array.Sort(ms);
                string line = $"lfm3 vision tower ({System.IO.Path.GetFileName(path)}) Metal {grid}x{grid} patches ({grid * grid / 4} tokens): " +
                    $"first run {first:F1} ms, replay median {ms[2]:F1} ms (min {ms[0]:F1})";
                _output.WriteLine(line);
                Console.WriteLine(line);
            }
        }
        finally
        {
            GgmlBasicOps.ReleaseReuseComputeBuffers();
            GgmlBasicOps.ClearHostBufferCache();
        }
    }

    internal static NetworkWeights Load(GgufFile gguf, IAllocator allocator, BackendType backend)
    {
        var weights = new NetworkWeights(allocator, backend);
        weights.LoadGguf(gguf, n => n.StartsWith("v.", StringComparison.Ordinal) || n.StartsWith("mm.", StringComparison.Ordinal) && !n.StartsWith("mm.a.", StringComparison.Ordinal));
        // A still image's patch: both temporal kernels see the same frame, so they sum (one product).
        float[] w0 = weights.ReadF32("v.patch_embd.weight"), w1 = weights.ReadF32("v.patch_embd.weight.1");
        weights.AddF32("v.patch_embd.sum", w0.Select((v, i) => v + w1[i]).ToArray(), 768, 1024);
        return weights;
    }

    /// <summary>The tower on <paramref name="backend"/> and on the managed CPU (each over the file's formats) against the
    /// managed CPU over F32 copies of the same weights (exact F32 products): ggml-metal within 1% RMS; the CPUs' block
    /// products (managed and ggml-cpu) quantize their activations to 8 bits, which 14 layers accumulate to a few percent.</summary>
    private void Compare(GgmlBackendType ggml, BackendType backend)
    {
        const int Nx = 16, Ny = 12;
        string path = Environment.GetEnvironmentVariable("TS_LFM3_MMPROJ");
        using var gguf = new GgufFile(path);
        float[] patches = NetworkGraphTests.Rand(new Random(7), 768 * Nx * Ny);
        var cpu = new CpuAllocator(BlasEnum.DotNet);
        var ggmlAllocator = new GgmlAllocator(new GgmlContext([0], ggml), 0);
        try
        {
            float[] reference = RunTower(gguf, cpu, BackendType.Cpu, patches, Nx, Ny, dense: true, out _, out _);
            float[] device = RunTower(gguf, ggmlAllocator, backend, patches, Nx, Ny, dense: false, out double deviceMs, out string deviceWeights);
            float[] managed = RunTower(gguf, cpu, BackendType.Cpu, patches, Nx, Ny, dense: false, out double managedMs, out string managedWeights);
            Assert.Equal(2048 * Nx * Ny / 4, device.Length);
            (double deviceMax, double deviceRms) = Errors(reference, device);
            (double managedMax, double managedRms) = Errors(reference, managed);
            string line = $"lfm3 vision tower ({System.IO.Path.GetFileName(path)}) over {Nx}x{Ny} patches vs F32-weight reference: " +
                $"{backend} max {deviceMax:G3} rms {deviceRms:G3} ({deviceMs:F1} ms) [{deviceWeights}]; managed CPU max {managedMax:G3} " +
                $"rms {managedRms:G3} ({managedMs:F1} ms) [{managedWeights}]";
            _output.WriteLine(line);
            Console.WriteLine(line);
            // ggml-cpu, as the managed CPU, quantizes the activations of a block-format product to 8 bits.
            Assert.True(backend == BackendType.GgmlCpu ? deviceRms <= 6e-2 : deviceRms <= 1e-2 && deviceMax <= 3e-2, line);
            Assert.True(managedRms <= 6e-2, line);
        }
        finally
        {
            GgmlBasicOps.ReleaseReuseComputeBuffers();
            GgmlBasicOps.ClearHostBufferCache();
        }
    }

    /// <summary>Max |a - r| over max |r|, and the RMS of a - r over the RMS of r.</summary>
    private static (double Max, double Rms) Errors(float[] reference, float[] actual)
    {
        double scale = 0, worst = 0, ss = 0, dd = 0;
        for (int i = 0; i < reference.Length; i++)
        {
            double d = actual[i] - reference[i];
            scale = Math.Max(scale, Math.Abs(reference[i]));
            worst = Math.Max(worst, Math.Abs(d));
            ss += (double)reference[i] * reference[i];
            dd += d * d;
        }
        return (worst / scale, Math.Sqrt(dd / ss));
    }

    private static float[] RunTower(GgufFile gguf, IAllocator allocator, BackendType backend, float[] patches, int nx, int ny,
        bool dense, out double ms, out string loaded)
    {
        using var file = Load(gguf, allocator, backend);
        using var copies = dense ? new NetworkWeights(allocator, backend) : null;
        if (dense)
            foreach (string name in file.Names)
                copies.AddF32(name, file.ReadF32(name), file.Get(name).Ne);
        NetworkWeights weights = copies ?? file;
        loaded = $"{weights.Count} weights, {weights.QuantizedBytes >> 20} MB blocks + {weights.DenseBytes >> 20} MB F32";
        using var graph = new NetworkGraph(weights, "lfm3 vision tower");
        float[] Run() => (float[])graph.Run(b => b.Output(Tower(b, weights, patches, nx, ny)))[0].Clone();
        float[] result = Run();
        var sw = Stopwatch.StartNew();
        Run();
        ms = sw.Elapsed.TotalMilliseconds;
        return result;
    }

    internal static NetValue Tower(NetworkBuilder b, NetworkWeights weights, float[] patches, int nx, int ny)
    {
        const int C = 1024, Heads = 16, KvHeads = 8, HeadDim = 64;
        int n = nx * ny;
        NetValue x = b.RmsNorm(b.Linear(b.Input(patches, 768, n), "v.patch_embd.sum"), "v.embedding_norm.weight", Eps);
        NetValue table = b.RasterPositions(nx, ny);
        for (int l = 0; l < 14; l++)
        {
            string p = $"v.blk.{l}.";
            NetValue h = b.RmsNorm(x, p + "ln1.weight", Eps);
            NetValue mixed;
            if (weights.Contains(p + "conv2d_in_proj.weight"))
            {
                NetValue proj = b.Linear(h, p + "conv2d_in_proj.weight");
                NetValue gated = b.Reshape(b.Mul(b.Columns(proj, 0, C), b.Columns(proj, 2 * C, C)), C, nx, ny);
                NetValue conv = b.DepthwiseConv2d(gated, b.Weight(p + "conv2d_spatial.weight"), 1, 1, 1, 1);
                mixed = b.Linear(b.Mul(b.Columns(proj, C, C), b.Reshape(conv, C, n)), p + "conv2d_out_proj.weight");
            }
            else
            {
                mixed = b.SelfAttention(h, Heads, KvHeads, HeadDim, p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight",
                    p + "attn_out.weight", p + "attn_q_norm.weight", p + "attn_k_norm.weight", Eps, v => b.Rope2D(v, table, 1e6f),
                    1f / MathF.Sqrt(HeadDim), default, default, default, out _, out _);
            }
            x = b.Add(x, mixed);
            x = b.Add(x, b.FeedForward(b.RmsNorm(x, p + "ln2.weight", Eps), p + "ffn_gate.weight", p + "ffn_up.weight", p + "ffn_down.weight"));
        }
        NetValue e = b.LayerNorm(x, "mm.input_norm.weight", "mm.input_norm.bias", 1e-6f);
        NetValue merged = b.Reshape(b.Permute(b.Reshape(e, 2 * C, nx / 2, 2, ny / 2), 0, 2, 1, 3), 4 * C, n / 4);
        return b.GeluMlp(merged, "mm.0.weight", "mm.0.bias", "mm.2.weight", "mm.2.bias");
    }
}
