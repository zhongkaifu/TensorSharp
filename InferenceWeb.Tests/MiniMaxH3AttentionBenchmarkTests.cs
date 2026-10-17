// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Diagnostics;
using TensorSharp;
using TensorSharp.MLX;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests
{
    /// <summary>MLX's fused attention at a MiniMax-H3 DiT shape, for comparison with
    /// ggml-metal's flash attention (eng/validation/metal-flash-kv-pad.cpp times that one
    /// at the same shape). Opt in with <c>TS_H3_ATTN_BENCH=1</c>;
    /// <c>TS_H3_ATTN_BENCH_TOKENS</c> sets the sequence (default 34,991, H3 FL2VA at
    /// 1280x768x107). Measures the attention alone: no projections, no model.</summary>
    public sealed class MiniMaxH3AttentionBenchmarkTests
    {
        private readonly ITestOutputHelper _output;

        public MiniMaxH3AttentionBenchmarkTests(ITestOutputHelper output) => _output = output;

        [Fact]
        public void MlxFusedAttentionAtTheDitShape()
        {
            if (Environment.GetEnvironmentVariable("TS_H3_ATTN_BENCH") != "1")
            {
                _output.WriteLine("[attn-bench] set TS_H3_ATTN_BENCH=1 to run; skipping");
                return;
            }
            Assert.True(MlxNative.IsAvailable(), "MLX is not available on this machine.");
            int tokens = int.TryParse(Environment.GetEnvironmentVariable("TS_H3_ATTN_BENCH_TOKENS"), out int t) ? t : 34991;
            const int heads = 56, headDim = 128, repeats = 2;
            float scale = 1f / MathF.Sqrt(headDim);
            double flop = 4.0 * tokens * (double)tokens * headDim * heads;
            _output.WriteLine($"[attn-bench] MLX fused SDPA, [1, {heads}, {tokens}, {headDim}], " +
                              $"MLX_ENABLE_TF32={Environment.GetEnvironmentVariable("MLX_ENABLE_TF32") ?? "(unset)"}");

            MlxNative.MlxArray q = Random(tokens, heads, headDim, 1);
            MlxNative.MlxArray k = Random(tokens, heads, headDim, 2);
            MlxNative.MlxArray v = Random(tokens, heads, headDim, 3);
            foreach (DType type in new[] { DType.Float32, DType.Float16 })
            {
                MlxNative.MlxArray qt = type == DType.Float32 ? q : MlxNative.Astype(q, type);
                MlxNative.MlxArray kt = type == DType.Float32 ? k : MlxNative.Astype(k, type);
                MlxNative.MlxArray vt = type == DType.Float32 ? v : MlxNative.Astype(v, type);
                MlxNative.Eval(qt, kt, vt);
                double best = double.MaxValue;
                for (int r = 0; r <= repeats; r++)
                {
                    var clock = Stopwatch.StartNew();
                    MlxNative.MlxArray o = MlxNative.FastScaledDotProductAttention(qt, kt, vt, scale, string.Empty, default);
                    MlxNative.Eval(o);
                    double seconds = clock.Elapsed.TotalSeconds;
                    MlxNative.FreeArray(o);
                    if (r > 0) best = Math.Min(best, seconds);   // r == 0 compiles the kernel
                }
                _output.WriteLine($"[attn-bench]   {type,-8} best of {repeats}: {best:F3} s  {flop / best / 1e12:F2} TFLOP/s");
                if (type != DType.Float32) { MlxNative.FreeArray(qt); MlxNative.FreeArray(kt); MlxNative.FreeArray(vt); }
            }
            MlxNative.FreeArray(q); MlxNative.FreeArray(k); MlxNative.FreeArray(v);
        }

        // [1, heads, tokens, headDim] F32, N(0, 1.5^2) like the ggml benchmark's inputs.
        private static unsafe MlxNative.MlxArray Random(int tokens, int heads, int headDim, int seed)
        {
            var host = new float[(long)tokens * heads * headDim];
            var rng = new Random(seed);
            for (long i = 0; i < host.Length; i += 2)
            {
                double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
                double r = Math.Sqrt(-2.0 * Math.Log(u1)) * 1.5;
                host[i] = (float)(r * Math.Cos(2 * Math.PI * u2));
                if (i + 1 < host.Length) host[i + 1] = (float)(r * Math.Sin(2 * Math.PI * u2));
            }
            fixed (float* data = host)
            {
                MlxNative.MlxArray array = MlxNative.NewArrayFromHost((IntPtr)data, new[] { 1, heads, tokens, headDim }, DType.Float32);
                MlxNative.Eval(array);
                return array;
            }
        }
    }
}
