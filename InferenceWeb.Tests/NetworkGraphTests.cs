// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.GGML;
using TensorSharp.Graph;
using TensorSharp.MLX;
using TensorSharp.Models;
using TensorSharp.Models.Graph;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The non-decoder network helpers (TensorSharp.Models.Graph.NetworkBuilder) against straightforward host references on
/// random data, on every executor this machine runs: the managed CPU, the tensor executor's own reference decompositions
/// (what direct CUDA runs for the ops it has no kernel of), GGML cpu, metal, cuda and vulkan (one per process:
/// TS_TEST_GGML_BACKEND), MLX and direct CUDA.
/// </summary>
public sealed class NetworkGraphTests
{
    public enum Lane { Managed, Reference, GgmlCpu, GgmlMetal, Mlx, GgmlCuda, GgmlVulkan, Cuda }

    /// <summary>The GGML backend of a GGML lane.</summary>
    public static GgmlBackendType GgmlBackendOf(Lane lane) => lane switch
    {
        Lane.GgmlMetal => GgmlBackendType.Metal,
        Lane.GgmlCuda => GgmlBackendType.Cuda,
        Lane.GgmlVulkan => GgmlBackendType.Vulkan,
        _ => GgmlBackendType.Cpu,
    };

    /// <summary>The model-level backend of a lane.</summary>
    public static BackendType BackendOf(Lane lane) => lane switch
    {
        Lane.GgmlCpu => BackendType.GgmlCpu,
        Lane.GgmlMetal => BackendType.GgmlMetal,
        Lane.GgmlCuda => BackendType.GgmlCuda,
        Lane.GgmlVulkan => BackendType.GgmlVulkan,
        Lane.Mlx => BackendType.Mlx,
        Lane.Cuda => BackendType.Cuda,
        _ => BackendType.Cpu,
    };

    /// <summary>The lane of this process's pinned GGML backend (TS_TEST_GGML_BACKEND).</summary>
    public static Lane PinnedGgmlLane => TestGates.PinnedGgmlBackend switch
    {
        BackendType.GgmlMetal => Lane.GgmlMetal,
        BackendType.GgmlCuda => Lane.GgmlCuda,
        BackendType.GgmlVulkan => Lane.GgmlVulkan,
        _ => Lane.GgmlCpu,
    };

    /// <summary>GPU lanes whose products and attention take F16 (or F16-rounded) operands: tolerances as ggml-metal's.</summary>
    public static bool HalfOperandLane(Lane lane) => lane is Lane.GgmlMetal or Lane.GgmlCuda or Lane.GgmlVulkan;

    /// <summary>The tensor executor with none of a backend's kernels (CPU ops): the reference decompositions direct CUDA
    /// takes for the network ops (TensorGraphExecutor.Network.cs).</summary>
    private sealed class ReferenceExecutor : TensorGraphExecutor
    {
        public ReferenceExecutor(IAllocator allocator) : base(allocator) { }

        protected override Tensor QuantizedMatMul(IGraphWeight weight, Tensor x)
        {
            int rows = (int)x.Sizes[0];
            float[] host = x.GetElementsAsFloat((int)x.ElementCount());
            var y = new float[rows * weight.Ne1];
            HostQuantizedProduct(weight, host, rows, y);
            var result = new Tensor(Allocator, DType.Float32, rows, weight.Ne1);
            result.SetElementsAsFloat(y);
            return result;
        }
    }

    /// <summary>One lane's backend: its allocator, weights and network graph.</summary>
    internal sealed class LaneContext : IDisposable
    {
        private readonly IDisposable _allocatorOwner;
        private readonly bool _ggml;

        public LaneContext(Lane lane)
        {
            Lane = lane;
            switch (lane)
            {
                case Lane.Managed:
                case Lane.Reference:
                    Allocator = new CpuAllocator(BlasEnum.DotNet);
                    Backend = BackendType.Cpu;
                    break;
                case Lane.GgmlCpu:
                case Lane.GgmlMetal:
                case Lane.GgmlCuda:
                case Lane.GgmlVulkan:
                    Allocator = new GgmlAllocator(new GgmlContext([0], GgmlBackendOf(lane)), 0);
                    Backend = BackendOf(lane);
                    _ggml = true;
                    break;
                case Lane.Cuda:
                    var cuda = new TensorSharp.Cuda.CudaAllocator();
                    Allocator = cuda;
                    _allocatorOwner = cuda;
                    Backend = BackendType.Cuda;
                    break;
                case Lane.Mlx:
                    var mlx = new MlxAllocator();
                    Allocator = mlx;
                    _allocatorOwner = mlx;
                    Backend = BackendType.Mlx;
                    break;
            }
            Weights = new NetworkWeights(Allocator, Backend);
            Graph = new NetworkGraph(Weights, lane == Lane.Reference ? new ReferenceExecutor(Allocator) : null, $"test-{lane}");
            // The lane runs on the executor plugin it names.
            switch (lane)
            {
                case Lane.Managed: Assert.IsType<ManagedGraphExecutor>(Graph.Executor); break;
                case Lane.Reference: Assert.IsType<ReferenceExecutor>(Graph.Executor); break;
                case Lane.Mlx: Assert.IsType<MlxGraphExecutor>(Graph.Executor); break;
                case Lane.Cuda: Assert.IsType<TensorSharp.Cuda.CudaGraphExecutor>(Graph.Executor); break;
                default:
                    Assert.Equal(GgmlBackendOf(lane), Assert.IsType<GgmlGraphExecutor>(Graph.Executor).Backend);
                    Assert.True(Graph.AttentionHalfKv);
                    break;
            }
        }

        public Lane Lane { get; }
        public IAllocator Allocator { get; }
        public BackendType Backend { get; }
        public NetworkWeights Weights { get; }
        public NetworkGraph Graph { get; }

        /// <summary>Run one program; its first output, copied.</summary>
        public float[] Run(Func<NetworkBuilder, NetValue> emit) => (float[])Graph.Run(b => b.Output(emit(b)))[0].Clone();

        public void Dispose()
        {
            Graph.Dispose();
            Weights.Dispose();
            _allocatorOwner?.Dispose();
            if (_ggml)
            {
                GgmlBasicOps.ReleaseReuseComputeBuffers();
                GgmlBasicOps.ClearHostBufferCache();
            }
        }
    }

    private static readonly Dictionary<string, Action<LaneContext>> s_cases = new()
    {
        ["linear_f32_bias"] = c => Linear(c, GraphFormat.F32, 64, 48, bias: true, 2e-3),
        ["linear_f16"] = c => Linear(c, GraphFormat.F16, 64, 40, bias: false, 6e-3),
        ["linear_bf16"] = c => Linear(c, GraphFormat.BF16, 64, 40, bias: true, 1e-2),
        ["linear_q8_0"] = c => Linear(c, GraphFormat.Q8_0, 96, 40, bias: true, 2e-2),
        ["linear_q4_0"] = c => Linear(c, GraphFormat.Q4_0, 96, 40, bias: false, 3e-2),
        ["linear_q4_k"] = c => Linear(c, (int)GgmlTensorType.Q4_K, 256, 24, bias: true, 3e-2),
        ["rmsnorm"] = RmsNorm,
        ["layernorm"] = LayerNorm,
        ["gelu_erf"] = GeluErf,
        ["silu_sigmoid_swiglu"] = SiluSigmoidSwiglu,
        ["unary_squared"] = UnarySquared,
        ["residual_scale_broadcast"] = ResidualScaleBroadcast,
        ["causal_conv1d"] = c => CausalConv1d(c, withState: false),
        ["causal_conv1d_state"] = c => CausalConv1d(c, withState: true),
        ["depthwise_conv2d_s1"] = c => DepthwiseConv2d(c, 16, 7, 5, 1, 1, 1),
        ["depthwise_conv2d_s2_batch"] = c => DepthwiseConv2d(c, 24, 9, 8, 2, 1, 2),
        ["depthwise_conv2d_one_channel"] = c => DepthwiseConv2d(c, 1, 9, 6, 1, 1, 1),
        ["depthwise_conv2d_one_column"] = c => DepthwiseConv2d(c, 8, 1, 6, 1, 1, 1),
        ["conv2d_im2col_s2"] = c => Conv2d(c, 1, 8, 9, 8, 2, 1),
        ["conv2d_im2col_rgb"] = c => Conv2d(c, 3, 5, 6, 4, 1, 1),
        ["upsample_nearest"] = UpsampleNearest,
        ["attention_bidirectional"] = c => Attention(c, "none"),
        ["attention_causal"] = c => Attention(c, "causal"),
        ["attention_sliding"] = c => Attention(c, "sliding"),
        ["attention_segments"] = c => Attention(c, "segments"),
        ["attention_state_prefix"] = c => Attention(c, "prefix"),
        ["rope_neox"] = c => RopeNeox(c, 32),
        ["rope_neox_partial"] = c => RopeNeox(c, 16),
        ["rope_2d"] = Rope2D,
        ["shapes_permute_concat"] = Shapes,
        ["head_gate"] = HeadGate,
        ["get_rows"] = GetRows,
        ["slice_tail_sum_rows"] = SliceTailSumRows,
        ["mini_network"] = MiniNetwork,
    };

    public static IEnumerable<object[]> Cases => s_cases.Keys.Select(k => new object[] { k });

    [Theory]
    [MemberData(nameof(Cases))]
    public void ManagedCpu(string name) => Check(Lane.Managed, name);

    [Theory]
    [MemberData(nameof(Cases))]
    public void ReferenceDecompositions(string name) => Check(Lane.Reference, name);

    [GgmlTheory(BackendType.GgmlCpu)]
    [MemberData(nameof(Cases))]
    public void GgmlCpu(string name) => Check(Lane.GgmlCpu, name);

    [GgmlTheory(BackendType.GgmlMetal)]
    [MemberData(nameof(Cases))]
    public void GgmlMetal(string name) => Check(Lane.GgmlMetal, name);

    [MlxTheory]
    [MemberData(nameof(Cases))]
    public void Mlx(string name) => Check(Lane.Mlx, name);

    [GgmlTheory(BackendType.GgmlCuda)]
    [MemberData(nameof(Cases))]
    public void GgmlCuda(string name) => Check(Lane.GgmlCuda, name);

    [GgmlTheory(BackendType.GgmlVulkan)]
    [MemberData(nameof(Cases))]
    public void GgmlVulkan(string name) => Check(Lane.GgmlVulkan, name);

    [CudaTheory]
    [MemberData(nameof(Cases))]
    public void DirectCuda(string name) => Check(Lane.Cuda, name);

    private static void Check(Lane lane, string name)
    {
        using var context = new LaneContext(lane);
        s_cases[name](context);
    }

    // ================================================================ cases

    private static void Linear(LaneContext c, int format, int inWidth, int outRows, bool bias, double tol)
    {
        var rng = new Random(11 + format);
        int rows = 7;
        string w = "w";
        MakeProductWeight(c.Weights, w, format, inWidth, outRows, rng);
        float[] wv = c.Weights.ReadF32(w);
        float[] b = bias ? Rand(rng, outRows) : null;
        if (bias) c.Weights.AddF32("b", b, outRows);
        float[] x = Rand(rng, inWidth * rows);
        float[] actual = c.Run(n => n.Linear(n.Input(x, inWidth, rows), w, bias ? "b" : null));
        float[] expected = MatMulRef(wv, inWidth, outRows, x, rows);
        if (bias) for (int r = 0; r < rows; r++) for (int o = 0; o < outRows; o++) expected[r * outRows + o] += b[o];
        AssertClose(expected, actual, tol, $"linear {format}");
    }

    private static void RmsNorm(LaneContext c)
    {
        var rng = new Random(2);
        int n = 64, rows = 9;
        float[] x = Rand(rng, n * rows), w = Rand(rng, n);
        c.Weights.AddF32("w", w, n);
        float[] actual = c.Run(b => b.RmsNorm(b.Input(x, n, rows), "w", 1e-5f));
        var expected = new float[x.Length];
        for (int r = 0; r < rows; r++)
        {
            double ss = 0;
            for (int i = 0; i < n; i++) ss += (double)x[r * n + i] * x[r * n + i];
            double s = 1 / Math.Sqrt(ss / n + 1e-5);
            for (int i = 0; i < n; i++) expected[r * n + i] = (float)(x[r * n + i] * s * w[i]);
        }
        AssertClose(expected, actual, 1e-4, "rmsnorm");
    }

    private static void LayerNorm(LaneContext c)
    {
        var rng = new Random(3);
        int n = 96, rows = 6;
        float[] x = Rand(rng, n * rows).Select(v => v * 3 + 1.5f).ToArray(), w = Rand(rng, n), bias = Rand(rng, n);
        c.Weights.AddF32("w", w, n);
        c.Weights.AddF32("b", bias, 1, 1, n);
        float[] actual = c.Run(b => b.LayerNorm(b.Input(x, n, rows), "w", "b", 1e-6f));
        var expected = new float[x.Length];
        for (int r = 0; r < rows; r++)
        {
            double mean = 0, var = 0;
            for (int i = 0; i < n; i++) mean += x[r * n + i];
            mean /= n;
            for (int i = 0; i < n; i++) var += (x[r * n + i] - mean) * (x[r * n + i] - mean);
            var /= n;
            double s = 1 / Math.Sqrt(var + 1e-6);
            for (int i = 0; i < n; i++) expected[r * n + i] = (float)((x[r * n + i] - mean) * s * w[i] + bias[i]);
        }
        AssertClose(expected, actual, 2e-4, "layernorm");
    }

    private static void GeluErf(LaneContext c)
    {
        var rng = new Random(4);
        float[] x = Rand(rng, 40 * 9).Select(v => v * 4f).Concat(new[] { 0f, -0f, 1e-4f, -1e-4f, 7f, -7f, 12f, -12f, 0.5f }).ToArray();
        float[] actual = c.Run(b => b.GeluErf(b.Input(x, x.Length)));
        float[] expected = x.Select(v => (float)(0.5 * v * (1 + Erf(v / Math.Sqrt(2))))).ToArray();
        // Abramowitz and Stegun's erf (ggml-metal, the managed kernel) is within 1.5e-7 of erf: |error| <= 0.5 |x| 1.5e-7.
        for (int i = 0; i < x.Length; i++)
            Assert.True(Math.Abs(actual[i] - expected[i]) <= 2e-6 + 1e-6 * Math.Abs(expected[i]),
                $"{c.Lane} gelu_erf({x[i]}) = {actual[i]}, expected {expected[i]}");
    }

    private static void SiluSigmoidSwiglu(LaneContext c)
    {
        var rng = new Random(5);
        int n = 40, rows = 5;
        float[] g = Rand(rng, n * rows).Select(v => v * 3).ToArray(), u = Rand(rng, n * rows);
        float[][] outputs = c.Graph.Run(b =>
        {
            NetValue gate = b.Input(g, n, rows), up = b.Input(u, n, rows);
            b.Output(b.SwiGlu(gate, up));
            b.Output(b.Sigmoid(gate));
            b.Output(b.Silu(up));
        });
        static double Sig(double v) => 1 / (1 + Math.Exp(-v));
        AssertClose(g.Select((v, i) => (float)(v * Sig(v) * u[i])).ToArray(), outputs[0], 1e-5, "swiglu");
        AssertClose(g.Select(v => (float)Sig(v)).ToArray(), outputs[1], 1e-5, "sigmoid");
        AssertClose(u.Select(v => (float)(v * Sig(v))).ToArray(), outputs[2], 1e-5, "silu");
    }

    /// <summary>A unary op's result times itself (silu(x)^2, sigmoid(x)^2), then another op: ggml-vulkan fuses a UNARY followed
    /// by a MUL reading the other operand from memory, which for a square is the unary's own result, never written (silu(x)^2
    /// came out 0.3 off; ggml d9fe7bf). The graph builder emits a square as SQR on Vulkan.</summary>
    private static void UnarySquared(LaneContext c)
    {
        var rng = new Random(15);
        int n = 24, rows = 3;
        float[] x = Rand(rng, n * rows).Select(v => v * 2).ToArray(), o = Rand(rng, n * rows);
        float[][] outputs = c.Graph.Run(b =>
        {
            NetValue t = b.Silu(b.Scale(b.Add(b.Input(x, n, rows), b.Input(o, n, rows)), 0.5f));
            b.Output(b.Sub(b.Mul(t, t), b.Input(o, n, rows)));
            NetValue g = b.Sigmoid(b.Input(x, n, rows));
            b.Output(b.Mul(g, g));
        });
        static double Sig(double v) => 1 / (1 + Math.Exp(-v));
        AssertClose(x.Select((v, i) =>
        {
            double s = 0.5 * (v + o[i]), t = s * Sig(s);
            return (float)(t * t - o[i]);
        }).ToArray(), outputs[0], 1e-5, $"{c.Lane} silu(x)^2 - o");
        AssertClose(x.Select(v => (float)(Sig(v) * Sig(v))).ToArray(), outputs[1], 1e-5, $"{c.Lane} sigmoid(x)^2");
    }

    private static void ResidualScaleBroadcast(LaneContext c)
    {
        var rng = new Random(6);
        int n = 32, rows = 4;
        float[] x = Rand(rng, n * rows), y = Rand(rng, n * rows), v = Rand(rng, n);
        float[] actual = c.Run(b =>
        {
            NetValue a = b.Input(x, n, rows);
            return b.Mul(b.Add(b.Scale(a, 0.5f), b.Input(y, n, rows)), b.Input(v, n));
        });
        float[] expected = x.Select((xv, i) => (0.5f * xv + y[i]) * v[i % n]).ToArray();
        AssertClose(expected, actual, 1e-6, "residual");
    }

    private static void CausalConv1d(LaneContext c, bool withState)
    {
        var rng = new Random(withState ? 8 : 7);
        int channels = 24, t = 10, k = 3;
        float[] x = Rand(rng, channels * t), kernel = Rand(rng, k * channels), state = withState ? Rand(rng, (k - 1) * channels) : new float[(k - 1) * channels];
        c.Weights.AddF32("conv", kernel, k, channels);
        // The carried columns live in a state tensor the program reads and rewrites ([K - 1, C], time innermost).
        using var stateTensor = new Tensor(c.Allocator, DType.Float32, channels, k - 1);
        stateTensor.SetElementsAsFloat(state);
        float[] actual = c.Run(b =>
        {
            NetValue s = withState ? b.State(stateTensor) : default;
            NetValue y = b.CausalConv1d(b.Input(x, channels, t), b.Weight("conv"), s, out NetValue window);
            if (withState) b.Write(b.ConvTail(window, k - 1), s);
            return y;
        });
        var expected = new float[channels * t];
        float At(int col, int ch) => col < k - 1 ? state[ch * (k - 1) + col] : x[(col - (k - 1)) * channels + ch];
        for (int tt = 0; tt < t; tt++)
            for (int ch = 0; ch < channels; ch++)
            {
                double sum = 0;
                for (int j = 0; j < k; j++) sum += At(tt + j, ch) * kernel[ch * k + j];
                expected[tt * channels + ch] = (float)sum;
            }
        AssertClose(expected, actual, 1e-5, "causal conv");
        if (!withState) return;
        if (c.Graph.Executor.Capabilities.BindsHostPointers)
            c.Graph.Executor.SyncToHost(TensorComputePrimitives.GetStoragePointer(stateTensor), stateTensor.ElementCount() * 4);
        float[] carried = stateTensor.GetElementsAsFloat((int)stateTensor.ElementCount());
        var expectedState = new float[(k - 1) * channels];
        for (int ch = 0; ch < channels; ch++)
            for (int j = 0; j < k - 1; j++) expectedState[ch * (k - 1) + j] = x[(t - (k - 1) + j) * channels + ch];
        AssertClose(expectedState, carried, 0, "conv state");
    }

    private static void DepthwiseConv2d(LaneContext c, int channels, int w, int h, int stride, int pad, int batch)
    {
        var rng = new Random(9 + channels + w);
        float[] x = Rand(rng, channels * w * h * batch), kernel = Rand(rng, 9 * channels);
        c.Weights.AddF32("dw", kernel, 3, 3, 1, channels);
        float[] actual = c.Run(b => b.DepthwiseConv2d(b.Input(x, channels, w, h, batch), b.Weight("dw"), stride, stride, pad, pad));
        int ow = (w + 2 * pad - 3) / stride + 1, oh = (h + 2 * pad - 3) / stride + 1;
        var expected = new float[channels * ow * oh * batch];
        for (int n = 0; n < batch; n++)
            for (int oy = 0; oy < oh; oy++)
                for (int ox = 0; ox < ow; ox++)
                    for (int ch = 0; ch < channels; ch++)
                    {
                        double sum = 0;
                        for (int ky = 0; ky < 3; ky++)
                            for (int kx = 0; kx < 3; kx++)
                            {
                                int ix = ox * stride + kx - pad, iy = oy * stride + ky - pad;
                                if (ix < 0 || ix >= w || iy < 0 || iy >= h) continue;
                                sum += x[((n * h + iy) * w + ix) * channels + ch] * kernel[ch * 9 + ky * 3 + kx];
                            }
                        expected[((n * oh + oy) * ow + ox) * channels + ch] = (float)sum;
                    }
        AssertClose(expected, actual, 1e-5, $"depthwise conv {channels}x{w}x{h} s{stride}");
    }

    private static void Conv2d(LaneContext c, int cin, int cout, int w, int h, int stride, int pad)
    {
        var rng = new Random(10 + cin);
        float[] x = Rand(rng, cin * w * h), weight = Rand(rng, 9 * cin * cout), bias = Rand(rng, cout);
        c.Weights.AddF32("conv", weight, 3, 3, cin, cout);
        c.Weights.AddF32("bias", bias, 1, 1, cout);
        float[] actual = c.Run(b => b.Conv2d(b.Input(x, cin, w, h), "conv", "bias", stride, stride, pad, pad));
        int ow = (w + 2 * pad - 3) / stride + 1, oh = (h + 2 * pad - 3) / stride + 1;
        var expected = new float[cout * ow * oh];
        for (int oy = 0; oy < oh; oy++)
            for (int ox = 0; ox < ow; ox++)
                for (int co = 0; co < cout; co++)
                {
                    double sum = bias[co];
                    for (int ci = 0; ci < cin; ci++)
                        for (int ky = 0; ky < 3; ky++)
                            for (int kx = 0; kx < 3; kx++)
                            {
                                int ix = ox * stride + kx - pad, iy = oy * stride + ky - pad;
                                if (ix < 0 || ix >= w || iy < 0 || iy >= h) continue;
                                sum += x[(iy * w + ix) * cin + ci] * weight[kx + 3 * (ky + 3 * (ci + cin * co))];
                            }
                    expected[(oy * ow + ox) * cout + co] = (float)sum;
                }
        AssertClose(expected, actual, 2e-3, $"conv2d {cin}->{cout} s{stride}");
    }

    private static void UpsampleNearest(LaneContext c)
    {
        var rng = new Random(12);
        int n = 5, t = 4, f = 6;
        float[] x = Rand(rng, n * t);
        float[] actual = c.Run(b => b.UpsampleNearest(b.Input(x, n, t), f));
        var expected = new float[n * t * f];
        for (int r = 0; r < t * f; r++) Array.Copy(x, r / f * n, expected, r * n, n);
        AssertClose(expected, actual, 0, "upsample");
    }

    private static void Attention(LaneContext c, string kind)
    {
        var rng = new Random(13 + kind.Length);
        int d = 32, heads = 4, kvHeads = 2, rows = kind == "prefix" ? 6 : kind == "segments" ? 9 : 9;
        int prefix = kind == "prefix" ? 5 : 0, keys = rows + prefix, queryStart = kind == "prefix" ? 3 : 0;
        float[] q = Rand(rng, d * heads * rows), k = Rand(rng, d * kvHeads * keys), v = Rand(rng, d * kvHeads * keys);
        float scale = 1 / MathF.Sqrt(d);
        ushort[] mask = kind switch
        {
            "none" => null,
            "causal" => NetworkMasks.Causal(rows, keys, 0),
            "sliding" => NetworkMasks.SlidingWindow(rows, keys, 4, 0),
            "segments" => NetworkMasks.Segments(3, 4, 2),
            "prefix" => NetworkMasks.SlidingWindow(rows, keys, 4, queryStart, queryStart - prefix),
            _ => throw new ArgumentException(kind),
        };
        float[] actual = c.Run(b => b.Attention(b.Input(q, d, heads, rows), b.Input(k, d, kvHeads, keys), b.Input(v, d, kvHeads, keys),
            scale, mask == null ? default : b.Mask(mask, keys, rows)));
        var expected = new float[d * heads * rows];
        for (int r = 0; r < rows; r++)
            for (int hh = 0; hh < heads; hh++)
            {
                int kh = hh / (heads / kvHeads);
                var scores = new double[keys];
                double max = double.NegativeInfinity;
                for (int j = 0; j < keys; j++)
                {
                    if (mask != null && mask[r * keys + j] != 0) { scores[j] = double.NegativeInfinity; continue; }
                    double s = 0;
                    for (int i = 0; i < d; i++) s += (double)q[(r * heads + hh) * d + i] * k[(j * kvHeads + kh) * d + i];
                    scores[j] = s * scale;
                    max = Math.Max(max, scores[j]);
                }
                double total = 0;
                for (int j = 0; j < keys; j++) total += scores[j] = double.IsNegativeInfinity(scores[j]) ? 0 : Math.Exp(scores[j] - max);
                for (int i = 0; i < d; i++)
                {
                    double o = 0;
                    for (int j = 0; j < keys; j++) o += scores[j] / total * v[(j * kvHeads + kh) * d + i];
                    expected[(r * heads + hh) * d + i] = (float)o;
                }
            }
        AssertClose(expected, actual, 4e-3, $"attention {kind}");
    }

    private static void RopeNeox(LaneContext c, int dims)
    {
        var rng = new Random(14 + dims);
        int d = 32, heads = 3, t = 6, start = 5;
        float[] x = Rand(rng, d * heads * t);
        float[] actual = c.Run(b => b.RopeNeox(b.Input(x, d, heads, t), b.Positions(start, t), dims, 10000f));
        var expected = (float[])x.Clone();
        for (int tt = 0; tt < t; tt++)
            for (int hh = 0; hh < heads; hh++)
                for (int i = 0; i < dims / 2; i++)
                {
                    double theta = (start + tt) * Math.Pow(10000.0, -2.0 * i / dims);
                    int at = (tt * heads + hh) * d;
                    double x0 = x[at + i], x1 = x[at + i + dims / 2];
                    expected[at + i] = (float)(x0 * Math.Cos(theta) - x1 * Math.Sin(theta));
                    expected[at + i + dims / 2] = (float)(x0 * Math.Sin(theta) + x1 * Math.Cos(theta));
                }
        AssertClose(expected, actual, 1e-4, $"rope neox {dims}");
    }

    private static void Rope2D(LaneContext c)
    {
        var rng = new Random(15);
        int d = 64, heads = 2, nx = 4, ny = 3, t = nx * ny;
        float[] x = Rand(rng, d * heads * t);
        float[] actual = c.Run(b => b.Rope2D(b.Input(x, d, heads, t), b.RasterPositions(nx, ny), 100f));
        float[] expected = Rope2DRef(x, d, heads, nx, ny, 100.0);
        AssertClose(expected, actual, 1e-4, "rope 2d");
    }

    private static float[] Rope2DRef(float[] x, int d, int heads, int nx, int ny, double theta)
    {
        var expected = (float[])x.Clone();
        int quarter = d / 4, half = d / 2;
        for (int tt = 0; tt < nx * ny; tt++)
        {
            int py = tt / nx, px = tt % nx;
            for (int hh = 0; hh < heads; hh++)
                for (int i = 0; i < half; i++)
                {
                    double angle = i < quarter ? py * Math.Pow(theta, -2.0 * i / half) : px * Math.Pow(theta, -2.0 * (i - quarter) / half);
                    int at = (tt * heads + hh) * d;
                    double x0 = x[at + i], x1 = x[at + i + half];
                    expected[at + i] = (float)(x0 * Math.Cos(angle) - x1 * Math.Sin(angle));
                    expected[at + i + half] = (float)(x0 * Math.Sin(angle) + x1 * Math.Cos(angle));
                }
        }
        return expected;
    }

    private static void Shapes(LaneContext c)
    {
        var rng = new Random(16);
        float[] x = Rand(rng, 4 * 3 * 2), y = Rand(rng, 12);
        float[] actual = c.Run(b =>
        {
            NetValue p = b.Permute(b.Input(x, 4, 3, 2), 1, 0, 2, 3);   // [3, 4, 2]
            NetValue r = b.Reshape(p, 12, 2);
            return b.Concat(r, b.Input(y, 12, 1), 1);                   // [12, 3]
        });
        var expected = new float[36];
        for (int k = 0; k < 2; k++)
            for (int j = 0; j < 4; j++)
                for (int i = 0; i < 3; i++)
                    expected[k * 12 + j * 3 + i] = x[k * 12 + i * 4 + j];
        Array.Copy(y, 0, expected, 24, 12);
        AssertClose(expected, actual, 0, "permute/reshape/concat");
    }

    private static void HeadGate(LaneContext c)
    {
        var rng = new Random(17);
        int d = 8, heads = 3, rows = 5;
        float[] a = Rand(rng, d * heads * rows), g = Rand(rng, heads * rows);
        float[] actual = c.Run(b => b.HeadGate(b.Input(a, d * heads, rows), b.Input(g, heads, rows), d));
        float[] expected = a.Select((v, i) => (float)(v / (1 + Math.Exp(-g[i / (d * heads) * heads + i % (d * heads) / d])))).ToArray();
        AssertClose(expected, actual, 1e-5, "head gate");
    }

    private static void GetRows(LaneContext c)
    {
        var rng = new Random(18);
        int width = 64, rows = 10;
        MakeProductWeight(c.Weights, "table", GraphFormat.Q8_0, width, rows, rng);
        float[] dense = Rand(rng, width * rows);
        c.Weights.AddF32("dense", dense, width, rows);
        int[] ids = { 3, 0, 7, 7, 9 };
        float[][] outputs = c.Graph.Run(b =>
        {
            NetValue idx = b.InputI32(ids);
            b.Output(b.GetRows(b.Weight("table"), idx));
            b.Output(b.GetRows(b.Weight("dense"), idx));
        });
        float[] table = c.Weights.ReadF32("table");
        AssertClose(ids.SelectMany(i => table.Skip(i * width).Take(width)).ToArray(), outputs[0], 1e-6, "get_rows q8_0");
        AssertClose(ids.SelectMany(i => dense.Skip(i * width).Take(width)).ToArray(), outputs[1], 0, "get_rows f32");
    }

    private static void SliceTailSumRows(LaneContext c)
    {
        var rng = new Random(22);
        int d = 8, heads = 3, keys = 7;
        float[] x = Rand(rng, d * heads * keys);
        float[][] outputs = c.Graph.Run(b =>
        {
            NetValue v = b.Input(x, d, heads, keys);
            b.Output(b.Tail(v, 2, 3));            // the last 3 keys (packed rows)
            b.Output(b.Slice(v, 1, 1, 2));        // heads 1..2 of every key (strided)
            b.Output(b.SumRows(b.Slice(v, 0, 2, 4)));
        });
        AssertClose(x.Skip(d * heads * (keys - 3)).ToArray(), outputs[0], 0, "tail");
        var heads12 = new List<float>();
        var sums = new List<float>();
        for (int k = 0; k < keys; k++)
            for (int h = 0; h < heads; h++)
            {
                if (h >= 1) heads12.AddRange(x.Skip((k * heads + h) * d).Take(d));
                sums.Add(x.Skip((k * heads + h) * d + 2).Take(4).Sum());
            }
        AssertClose(heads12.ToArray(), outputs[1], 0, "slice");
        AssertClose(sums.ToArray(), outputs[2], 1e-6, "sum rows");
    }

    // ================================================================ an end-to-end mini tower

    /// <summary>A two-layer LFM3-vision-shaped tower: patch product, embedding norm, a gated depthwise-conv layer and a
    /// GQA attention layer (per-head q/k norms, 2-D rotary), each with a SwiGLU FFN, then LayerNorm, the 2x2 merge and a GELU
    /// projector; against the same network in host loops.</summary>
    private static void MiniNetwork(LaneContext c)
    {
        const int Patch = 48, C = 64, Ff = 96, Heads = 4, KvHeads = 2, Hd = 16, Nx = 4, Ny = 2, N = Nx * Ny, Out = 32;
        const float Eps = 1e-5f;
        var rng = new Random(19);
        NetworkWeights w = c.Weights;
        MakeProductWeight(w, "patch", GraphFormat.F16, Patch, C, rng);
        w.AddF32("embed_norm", Rand(rng, C, 0.5f, 1.5f), C);
        foreach (int l in new[] { 0, 1 })
        {
            w.AddF32($"l{l}.ln1", Rand(rng, C, 0.5f, 1.5f), C);
            w.AddF32($"l{l}.ln2", Rand(rng, C, 0.5f, 1.5f), C);
            MakeProductWeight(w, $"l{l}.gate", GraphFormat.Q8_0, C, Ff, rng);
            MakeProductWeight(w, $"l{l}.up", GraphFormat.Q8_0, C, Ff, rng);
            MakeProductWeight(w, $"l{l}.down", GraphFormat.F32, Ff, C, rng);
        }
        MakeProductWeight(w, "l0.in_proj", GraphFormat.Q8_0, C, 3 * C, rng);
        w.AddF32("l0.dw", Rand(rng, 9 * C), 3, 3, 1, C);
        MakeProductWeight(w, "l0.out_proj", GraphFormat.F16, C, C, rng);
        MakeProductWeight(w, "l1.q", GraphFormat.F16, C, Heads * Hd, rng);
        MakeProductWeight(w, "l1.k", GraphFormat.F16, C, KvHeads * Hd, rng);
        MakeProductWeight(w, "l1.v", GraphFormat.F16, C, KvHeads * Hd, rng);
        MakeProductWeight(w, "l1.o", GraphFormat.F16, Heads * Hd, C, rng);
        w.AddF32("l1.qn", Rand(rng, Hd, 0.5f, 1.5f), Hd);
        w.AddF32("l1.kn", Rand(rng, Hd, 0.5f, 1.5f), Hd);
        w.AddF32("ln_w", Rand(rng, C, 0.5f, 1.5f), C);
        w.AddF32("ln_b", Rand(rng, C), C);
        MakeProductWeight(w, "mm0", GraphFormat.F32, 4 * C, 4 * C, rng);
        w.AddF32("mm0.b", Rand(rng, 4 * C), 4 * C);
        MakeProductWeight(w, "mm2", GraphFormat.F32, 4 * C, Out, rng);
        w.AddF32("mm2.b", Rand(rng, Out), Out);
        float[] patches = Rand(rng, Patch * N);

        float[] actual = c.Run(b =>
        {
            NetValue x = b.RmsNorm(b.Linear(b.Input(patches, Patch, N), "patch"), "embed_norm", Eps);
            // Conv layer.
            NetValue h = b.RmsNorm(x, "l0.ln1", Eps);
            NetValue p = b.Linear(h, "l0.in_proj");
            NetValue gated = b.Mul(b.Columns(p, 0, C), b.Columns(p, 2 * C, C));
            NetValue conv = b.DepthwiseConv2d(b.Reshape(gated, C, Nx, Ny), b.Weight("l0.dw"), 1, 1, 1, 1);
            x = b.Add(x, b.Linear(b.Mul(b.Columns(p, C, C), b.Reshape(conv, C, N)), "l0.out_proj"));
            x = b.Add(x, b.FeedForward(b.RmsNorm(x, "l0.ln2", Eps), "l0.gate", "l0.up", "l0.down"));
            // Attention layer.
            h = b.RmsNorm(x, "l1.ln1", Eps);
            NetValue table = b.RasterPositions(Nx, Ny);
            NetValue attended = b.SelfAttention(h, Heads, KvHeads, Hd, "l1.q", "l1.k", "l1.v", "l1.o", "l1.qn", "l1.kn", Eps,
                v => b.Rope2D(v, table, 1e6f), 1f / MathF.Sqrt(Hd), default, default, default, out _, out _);
            x = b.Add(x, attended);
            x = b.Add(x, b.FeedForward(b.RmsNorm(x, "l1.ln2", Eps), "l1.gate", "l1.up", "l1.down"));
            // Projector: LayerNorm, 2x2 merge ((0,0), (0,1), (1,0), (1,1) a merged cell), GELU MLP.
            NetValue e = b.LayerNorm(x, "ln_w", "ln_b", 1e-6f);
            NetValue merged = b.Reshape(b.Permute(b.Reshape(e, 2 * C, Nx / 2, 2, Ny / 2), 0, 2, 1, 3), 4 * C, N / 4);
            return b.GeluMlp(merged, "mm0", "mm0.b", "mm2", "mm2.b");
        });

        // ---- host reference
        float[] Lin(float[] xs, int rows, string name, string bias = null)
        {
            NetworkWeight nw = w.Get(name);
            float[] y = MatMulRef(w.ReadF32(name), (int)nw.RowWidth, (int)nw.Rows, xs, rows);
            if (bias != null)
            {
                float[] bv = w.ReadF32(bias);
                for (int i = 0; i < y.Length; i++) y[i] += bv[i % bv.Length];
            }
            return y;
        }
        float[] Rms(float[] xs, int width, string weight)
        {
            float[] g = w.ReadF32(weight);
            var y = new float[xs.Length];
            for (int r = 0; r < xs.Length / width; r++)
            {
                double ss = 0;
                for (int i = 0; i < width; i++) ss += (double)xs[r * width + i] * xs[r * width + i];
                double s = 1 / Math.Sqrt(ss / width + Eps);
                for (int i = 0; i < width; i++) y[r * width + i] = (float)(xs[r * width + i] * s * g[i]);
            }
            return y;
        }
        float[] Ffn(float[] xs, int l)
        {
            float[] n = Rms(xs, C, $"l{l}.ln2");
            float[] g = Lin(n, N, $"l{l}.gate"), u = Lin(n, N, $"l{l}.up");
            float[] a = g.Select((gv, i) => (float)(gv / (1 + Math.Exp(-gv)) * u[i])).ToArray();
            return Lin(a, N, $"l{l}.down");
        }
        float[] X = Rms(Lin(patches, N, "patch"), C, "embed_norm");
        {
            float[] hh = Rms(X, C, "l0.ln1"), pp = Lin(hh, N, "l0.in_proj");
            float[] dw = w.ReadF32("l0.dw");
            var g = new float[C * N];
            for (int t = 0; t < N; t++) for (int i = 0; i < C; i++) g[t * C + i] = pp[t * 3 * C + i] * pp[t * 3 * C + 2 * C + i];
            var cv = new float[C * N];
            for (int oy = 0; oy < Ny; oy++)
                for (int ox = 0; ox < Nx; ox++)
                    for (int ch = 0; ch < C; ch++)
                    {
                        double s = 0;
                        for (int ky = 0; ky < 3; ky++)
                            for (int kx = 0; kx < 3; kx++)
                            {
                                int ix = ox + kx - 1, iy = oy + ky - 1;
                                if (ix < 0 || ix >= Nx || iy < 0 || iy >= Ny) continue;
                                s += g[(iy * Nx + ix) * C + ch] * dw[ch * 9 + ky * 3 + kx];
                            }
                        cv[(oy * Nx + ox) * C + ch] = (float)s;
                    }
            var m = new float[C * N];
            for (int t = 0; t < N; t++) for (int i = 0; i < C; i++) m[t * C + i] = pp[t * 3 * C + C + i] * cv[t * C + i];
            float[] o = Lin(m, N, "l0.out_proj");
            for (int i = 0; i < X.Length; i++) X[i] += o[i];
            float[] f = Ffn(X, 0);
            for (int i = 0; i < X.Length; i++) X[i] += f[i];
        }
        {
            float[] hh = Rms(X, C, "l1.ln1");
            float[] q = Lin(hh, N, "l1.q"), k = Lin(hh, N, "l1.k"), v = Lin(hh, N, "l1.v");
            q = Rms(q, Hd, "l1.qn");
            k = Rms(k, Hd, "l1.kn");
            q = Rope2DRef(q, Hd, Heads, Nx, Ny, 1e6);
            k = Rope2DRef(k, Hd, KvHeads, Nx, Ny, 1e6);
            var att = new float[Heads * Hd * N];
            for (int r = 0; r < N; r++)
                for (int hd = 0; hd < Heads; hd++)
                {
                    int kh = hd / (Heads / KvHeads);
                    var sc = new double[N];
                    double max = double.NegativeInfinity;
                    for (int j = 0; j < N; j++)
                    {
                        double s = 0;
                        for (int i = 0; i < Hd; i++) s += (double)q[(r * Heads + hd) * Hd + i] * k[(j * KvHeads + kh) * Hd + i];
                        sc[j] = s / Math.Sqrt(Hd);
                        max = Math.Max(max, sc[j]);
                    }
                    double tot = 0;
                    for (int j = 0; j < N; j++) tot += sc[j] = Math.Exp(sc[j] - max);
                    for (int i = 0; i < Hd; i++)
                    {
                        double o = 0;
                        for (int j = 0; j < N; j++) o += sc[j] / tot * v[(j * KvHeads + kh) * Hd + i];
                        att[(r * Heads + hd) * Hd + i] = (float)o;
                    }
                }
            float[] ao = Lin(att, N, "l1.o");
            for (int i = 0; i < X.Length; i++) X[i] += ao[i];
            float[] f = Ffn(X, 1);
            for (int i = 0; i < X.Length; i++) X[i] += f[i];
        }
        float[] lw = w.ReadF32("ln_w"), lb = w.ReadF32("ln_b");
        var E = new float[C * N];
        for (int t = 0; t < N; t++)
        {
            double mean = 0, var = 0;
            for (int i = 0; i < C; i++) mean += X[t * C + i];
            mean /= C;
            for (int i = 0; i < C; i++) var += (X[t * C + i] - mean) * (X[t * C + i] - mean);
            double s = 1 / Math.Sqrt(var / C + 1e-6);
            for (int i = 0; i < C; i++) E[t * C + i] = (float)((X[t * C + i] - mean) * s * lw[i] + lb[i]);
        }
        var M = new float[4 * C * N / 4];
        for (int by = 0; by < Ny / 2; by++)
            for (int bx = 0; bx < Nx / 2; bx++)
            {
                int cell = by * (Nx / 2) + bx, slot = 0;
                foreach ((int dy, int dx) in new[] { (0, 0), (0, 1), (1, 0), (1, 1) })
                    Array.Copy(E, ((2 * by + dy) * Nx + 2 * bx + dx) * C, M, cell * 4 * C + C * slot++, C);
            }
        float[] hidden = Lin(M, N / 4, "mm0", "mm0.b").Select(v => (float)(0.5 * v * (1 + Erf(v / Math.Sqrt(2))))).ToArray();
        float[] expected = Lin(hidden, N / 4, "mm2", "mm2.b");
        AssertClose(expected, actual, HalfOperandLane(c.Lane) || c.Lane == Lane.GgmlCpu ? 3e-2 : 2e-2, "mini network");
    }

    // ================================================================ program cache, drift

    [Fact]
    public void ManagedProgramCacheReplaysByShape() => ProgramCache(Lane.Managed);

    [GgmlFact(BackendType.GgmlCpu)]
    public void GgmlCpuProgramCacheReplaysByShape() => ProgramCache(Lane.GgmlCpu);

    [GgmlFact(BackendType.GgmlMetal)]
    public void GgmlMetalProgramCacheReplaysByShape() => ProgramCache(Lane.GgmlMetal);

    [MlxFact]
    public void MlxProgramCacheReplaysByShape() => ProgramCache(Lane.Mlx);

    [GgmlFact(BackendType.GgmlCuda)]
    public void GgmlCudaProgramCacheReplaysByShape() => ProgramCache(Lane.GgmlCuda);

    [GgmlFact(BackendType.GgmlVulkan)]
    public void GgmlVulkanProgramCacheReplaysByShape() => ProgramCache(Lane.GgmlVulkan);

    [CudaFact]
    public void DirectCudaProgramCacheReplaysByShape() => ProgramCache(Lane.Cuda);

    private static void ProgramCache(Lane lane)
    {
        using var c = new LaneContext(lane);
        var rng = new Random(20);
        const int In = 64, OutRows = 32;
        MakeProductWeight(c.Weights, "w", GraphFormat.F16, In, OutRows, rng);
        float[] wv = c.Weights.ReadF32("w");
        void RunRows(int rows)
        {
            float[] x = Rand(rng, In * rows);
            float[] y = c.Run(b => b.GeluErf(b.Linear(b.Input(x, In, rows), "w")));
            float[] expected = MatMulRef(wv, In, OutRows, x, rows).Select(v => (float)(0.5 * v * (1 + Erf(v / Math.Sqrt(2))))).ToArray();
            AssertClose(expected, y, 6e-3, $"{lane} rows {rows}");
        }
        bool replays = c.Graph.Executor.Capabilities.Replays;
        RunRows(5);
        RunRows(5);   // the same shape, new values
        Assert.Equal(replays ? 1 : 2, c.Graph.Builds);
        Assert.Equal(replays ? 1 : 0, c.Graph.Replays);
        RunRows(9);   // another shape: another program
        Assert.Equal(replays ? 2 : 3, c.Graph.Builds);
        Assert.Equal(replays ? 2 : 0, c.Graph.CachedPlans);
        c.Graph.MaxCachedPlans = 1;
        RunRows(5);   // compiled again (the one-program cache keeps rows 5 now)
        RunRows(5);
        Assert.Equal(replays ? 3 : 5, c.Graph.Builds);
        Assert.Equal(replays ? 2 : 0, c.Graph.Replays);
        Assert.Equal(replays ? 1 : 0, c.Graph.CachedPlans);
        Assert.Equal(5, c.Graph.Runs);
    }

    [Fact]
    public void ProgramWithoutOutputIsRefused()
    {
        using var c = new LaneContext(Lane.Managed);
        var ex = Assert.Throws<InvalidOperationException>(() => c.Graph.Run(b => b.Input(new float[4], 4)));
        Assert.Contains("no output", ex.Message);
    }

    [Fact]
    public void GgmlOpCodesMirrorTheGraphOps()
    {
        foreach (GraphOp op in Enum.GetValues<GraphOp>())
        {
            Assert.True(Enum.TryParse(op.ToString(), out GgmlGraphOp mirrored), $"GgmlGraphOp has no {op}");
            Assert.Equal((int)op, (int)mirrored);
        }
    }

    [Fact]
    public void WeightsKeepProductWeightsInBlocksAndReadThemBack()
    {
        var allocator = new CpuAllocator(BlasEnum.DotNet);
        using var weights = new NetworkWeights(allocator, BackendType.Cpu);
        var rng = new Random(21);
        MakeProductWeight(weights, "q", GraphFormat.Q8_0, 64, 6, rng);
        NetworkWeight q = weights.Get("q");
        Assert.True(q.IsQuantized);
        Assert.Equal(GraphFormat.Q8_0, q.Format);
        // A row block is its own weight over the same bytes; a re-read width keeps whole blocks.
        NetworkWeight block = weights.RowBlock("q", 2, 3);
        Assert.Equal(new long[] { 64, 3 }, block.Ne);
        Assert.Equal(weights.ReadF32("q").Skip(128).Take(192), weights.ReadF32(block.Name));
        Assert.Equal(new long[] { 128, 3 }, weights.Reshaped("q", 128).Ne);
        Assert.Throws<ArgumentException>(() => weights.Reshaped("q", 48));
        Assert.Throws<KeyNotFoundException>(() => weights.Get("missing"));
    }

    // ================================================================ helpers

    internal static float[] Rand(Random rng, int n, float lo = -1f, float hi = 1f)
    {
        var v = new float[n];
        for (int i = 0; i < n; i++) v[i] = lo + (hi - lo) * (float)rng.NextDouble();
        return v;
    }

    /// <summary>A product weight [inWidth, outRows] in <paramref name="format"/>: F32 and F16/BF16 from random values,
    /// Q8_0 and Q4_0 quantized from them, Q4_K as random blocks with fixed scales (every executor's product is checked
    /// against the host dequantization of the same bytes).</summary>
    internal static void MakeProductWeight(NetworkWeights weights, string name, int format, int inWidth, int outRows, Random rng)
    {
        float[] values = Rand(rng, inWidth * outRows, -0.25f, 0.25f);
        switch (format)
        {
            case GraphFormat.F32:
                weights.AddF32(name, values, inWidth, outRows);
                return;
            case GraphFormat.F16:
            {
                var raw = new byte[values.Length * 2];
                for (int i = 0; i < values.Length; i++) BitConverter.TryWriteBytes(raw.AsSpan(i * 2), BitConverter.HalfToUInt16Bits((System.Half)values[i]));
                weights.AddRaw(name, format, raw, inWidth, outRows);
                return;
            }
            case GraphFormat.BF16:
            {
                var raw = new byte[values.Length * 2];
                for (int i = 0; i < values.Length; i++)
                    BitConverter.TryWriteBytes(raw.AsSpan(i * 2), (ushort)(BitConverter.SingleToUInt32Bits(values[i]) >> 16));
                weights.AddRaw(name, format, raw, inWidth, outRows);
                return;
            }
            case GraphFormat.Q8_0:
            case GraphFormat.Q4_0:
            {
                long rowBytes = inWidth / 32 * (format == GraphFormat.Q8_0 ? 34 : 18);
                var raw = new byte[rowBytes * outRows];
                for (int r = 0; r < outRows; r++)
                    ManagedQuantizedOps.QuantizeRowFromFloat32(format, values, r * inWidth, raw, (int)(r * rowBytes), inWidth);
                weights.AddRaw(name, format, raw, inWidth, outRows);
                return;
            }
            case (int)GgmlTensorType.Q4_K:
            {
                // block_q4_K: d, dmin (F16), 12 bytes of 6-bit scales and mins, 128 bytes of nibbles: 144 bytes a 256 values.
                int blocks = inWidth / 256 * outRows;
                var raw = new byte[blocks * 144];
                rng.NextBytes(raw);
                for (int b = 0; b < blocks; b++)
                {
                    BitConverter.TryWriteBytes(raw.AsSpan(b * 144), BitConverter.HalfToUInt16Bits((System.Half)0.004f));
                    BitConverter.TryWriteBytes(raw.AsSpan(b * 144 + 2), BitConverter.HalfToUInt16Bits((System.Half)0.002f));
                }
                weights.AddRaw(name, format, raw, inWidth, outRows);
                return;
            }
            default:
                throw new ArgumentException($"format {format}");
        }
    }

    /// <summary>y [rows, out] = x [rows, in] · wᵀ, w in ggml order ([in, out]: output row o's weights at o * in).</summary>
    internal static float[] MatMulRef(float[] w, int inWidth, int outRows, float[] x, int rows)
    {
        var y = new float[rows * outRows];
        for (int r = 0; r < rows; r++)
            for (int o = 0; o < outRows; o++)
            {
                double s = 0;
                for (int i = 0; i < inWidth; i++) s += (double)w[o * inWidth + i] * x[r * inWidth + i];
                y[r * outRows + o] = (float)s;
            }
        return y;
    }

    /// <summary>erf in double: its series below 2.5, the continued fraction of erfc above.</summary>
    internal static double Erf(double x)
    {
        double ax = Math.Abs(x), r;
        if (ax < 2.5)
        {
            double sum = 0, term = ax;
            for (int n = 0; n < 200; n++)
            {
                double t = term / (2 * n + 1);
                sum += t;
                if (Math.Abs(t) < 1e-18 * Math.Abs(sum) + 1e-300) break;
                term *= -ax * ax / (n + 1);
            }
            r = 2 / Math.Sqrt(Math.PI) * sum;
        }
        else
        {
            double f = ax;
            for (int k = 80; k >= 1; k--) f = ax + k / 2.0 / f;
            r = 1 - Math.Exp(-ax * ax) / Math.Sqrt(Math.PI) / f;
        }
        return x < 0 ? -r : r;
    }

    /// <summary>max |actual - expected| over max |expected| (the outputs' scale) within <paramref name="tol"/>.</summary>
    internal static void AssertClose(float[] expected, float[] actual, double tol, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        double scale = 1e-12, worst = 0;
        int at = -1;
        for (int i = 0; i < expected.Length; i++) scale = Math.Max(scale, Math.Abs(expected[i]));
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.False(float.IsNaN(actual[i]), $"{what}: NaN at {i}");
            double e = Math.Abs(actual[i] - expected[i]);
            if (e > worst) { worst = e; at = i; }
        }
        Assert.True(worst <= tol * scale, $"{what}: max |error| {worst:G4} at {at} (expected {(at >= 0 ? expected[at] : 0)}, got {(at >= 0 ? actual[at] : 0)}) over scale {scale:G4}, tolerance {tol}");
    }
}
