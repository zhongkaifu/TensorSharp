// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Pure-C# (BackendType.Cpu) fast path of the Qwen-Image-2.1 VAE: every convolution is an
// implicit-im2col packed SGEMM (CpuPackedGemm) against weights packed once per layer, the
// mid-block attention is two GEMMs per query block with a streaming softmax in between, and
// the channel norm, SiLU, residual add and resampling passes are vectorized and spread over
// a worker pool. The scalar loops in VaeReferenceMath stay the oracle the tests compare the
// fast path against (VaeReferenceMath.UseScalarCpu): the original convolution, attention and
// SiLU loops, i.e. the original numerics bit for bit; the norm, add and resampling passes are
// shared by both modes (they are bit-identical to the loops they replaced, only faster).
//
// Numerics: the GEMM accumulates each output with one FMA per k in k order (~1e-6 relative
// to the scalar loop per layer, 130+ dB PSNR for a whole decode), and that order does not
// depend on the kernel width or thread count, so AVX-512, AVX2 and any pool size give the
// same bits. Norm, add and resampling are bit-identical to the scalar code. SiLU uses a
// vectorized exp (within an ulp or two of MathF.Exp) only on this path; the GGML per-conv
// path keeps the exact scalar SiLU.
using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace TensorSharp.Models.QwenImage
{
    internal static unsafe partial class VaeReferenceMath
    {
        // Runs the original scalar convolution / attention / SiLU loops on the managed path:
        // the oracle tests and QwenImageStagesBench compare the fast path against (never set in
        // production; see the file header).
        internal static bool UseScalarCpu;

        /// <summary>The managed fast path is active: no device convolution, not forced scalar.
        /// Tracing keeps the unfused per-op sequence so every traced tensor still exists.</summary>
        private static bool FastCpu => !UseGpuConv && !UseScalarCpu;

        // The managed VAE runs nothing but these kernels, so it takes the wide pool (see
        // CpuPackedGemm); next to device convolutions the host passes keep the shared pool.
        internal static CpuWorkers CpuPool => FastCpu ? CpuPackedGemm.WidePool : CpuWorkers.Shared;

        // Weights that reach Conv2d as raw arrays (tests, callers without a VaeWeights) are
        // packed once per array identity; VaeWeights-owned layers use VaeWeights.PackedKernel.
        private static readonly ConditionalWeakTable<float[], PackedPanels> s_packedWeights = new();

        internal static PackedPanels PackConvWeight(float[] weight, int oc, int k)
        {
            // The pack reads the array through a raw pointer: check its length here, where the
            // scalar loop's bounds checks used to catch a mismatched weight.
            ArgumentNullException.ThrowIfNull(weight);
            if (weight.LongLength < (long)oc * k)
                throw new ArgumentException($"conv weight has {weight.LongLength} values, expected {oc} x {k}.", nameof(weight));
            CpuGemmIsa isa = CpuPackedGemm.Isa;
            if (s_packedWeights.TryGetValue(weight, out var packed) && packed.Isa == isa && packed.Rows == oc && packed.K == k)
                return packed;
            fixed (float* w = weight) packed = CpuPackedGemm.PackA(w, oc, k, k, 1, isa, CpuPool);
            s_packedWeights.AddOrUpdate(weight, packed);
            return packed;
        }

        /// <summary>
        /// Convolution on the packed GEMM: out[oc, p] = bias[oc] + W[oc, :] . im2col(x)[:, p].
        /// <paramref name="upsample2x"/> reads x through a nearest 2x upsample (the decoder's
        /// resample), so the 4x larger upsampled map is never materialized.
        /// </summary>
        internal static Feature Conv2dCpu(Feature x, PackedPanels weight, float[] bias, int OC, int KH, int KW,
            int strideH, int strideW, int padT, int padB, int padL, int padR, bool upsample2x = false)
        {
            int H = upsample2x ? 2 * x.H : x.H, W = upsample2x ? 2 * x.W : x.W;
            if (weight.Rows != OC || weight.K != x.C * KH * KW)
                throw new ArgumentException($"packed conv weight [{weight.Rows}, {weight.K}] does not match OC {OC} x {x.C}*{KH}*{KW}");
            // The GEMM reads x and the bias through raw pointers (bias[0..OC), x[0..C*H*W)).
            if (bias != null && bias.Length < OC)
                throw new ArgumentException($"conv bias has {bias.Length} values, expected {OC}.", nameof(bias));
            if (x.D.LongLength < (long)x.C * x.H * x.W)
                throw new ArgumentException($"feature holds {x.D.LongLength} values, expected {x.C} x {x.H} x {x.W}.", nameof(x));
            int Ho = (H + padT + padB - KH) / strideH + 1, Wo = (W + padL + padR - KW) / strideW + 1;
            var outp = Feature.Uninitialized(OC, Ho, Wo);
            fixed (float* xp = x.D, op = outp.D, bp = bias)
            {
                var source = new Im2colPanelSource(xp, x.H, x.W, KH, KW, strideH, strideW, padT, padL, Ho, Wo, upsample2x);
                CpuPackedGemm.Gemm(weight, source, Ho * Wo, op, (long)Ho * Wo, biasM: bp, pool: CpuPool);
            }
            return outp;
        }

        /// <summary>A named layer's convolution: packed/cached by <see cref="VaeWeights"/> on the
        /// managed fast path, the raw-array <see cref="Conv2d"/> otherwise.</summary>
        private static Feature ConvLayer(VaeWeights w, string prefix, Feature x, int OC, int IC, int KH, int KW,
            int strideH, int strideW, int padT, int padB, int padL, int padR)
        {
            if (FastCpu)
                return Conv2dCpu(x, w.PackedKernel(prefix + ".weight", OC, IC, 1, KH * KW), w.Get(prefix + ".bias"),
                    OC, KH, KW, strideH, strideW, padT, padB, padL, padR);
            return Conv2d(x, w.Get(prefix + ".weight"), OC, IC, KH, KW, w.Get(prefix + ".bias"),
                strideH, strideW, padT, padB, padL, padR);
        }

        // ---- elementwise ------------------------------------------------------------

        // Pixels per norm task: 1024 floats = one 4 KB page per channel row, so a tile walks
        // C pages instead of revisiting each page from four 256-pixel tiles.
        private const int NormTile = 1024;

        /// <summary>
        /// Vectorized channel RMS norm, bit-identical to the scalar definition: the squares are
        /// exact in double (24-bit x 24-bit mantissas), sums are added in channel order, sqrt and
        /// division are correctly rounded, and the output keeps the ((x*inv)*scale)*gamma order.
        /// With <paramref name="silu"/> the SiLU is applied in the same pass (vectorized exp).
        /// With <paramref name="inPlace"/> the result overwrites (and is) <paramref name="x"/>: a
        /// tile reads all of its pixels' channels for the sums before it writes any of them, and
        /// each element is read before it is written, so the values are the same bits.
        /// </summary>
        internal static Feature RmsNormChannelFast(Feature x, float[] gamma, bool silu, bool inPlace = false)
        {
            int C = x.C, hw = x.H * x.W;
            var outp = inPlace ? x : Feature.Uninitialized(C, x.H, x.W);
            float scale = MathF.Sqrt(C);
            int tiles = (hw + NormTile - 1) / NormTile;
            fixed (float* xp0 = x.D, op0 = outp.D, gp0 = gamma)
            {
                nint xL = (nint)xp0, oL = (nint)op0, gL = (nint)gp0;
                CpuPackedGemm.ForEach(tiles, (long)C * hw >= 1 << 16, tile =>
                {
                    float* xp = (float*)xL, op = (float*)oL, gp = (float*)gL;
                    int start = tile * NormTile, count = Math.Min(NormTile, hw - start);
                    double* sums = stackalloc double[NormTile];
                    float* inv = stackalloc float[NormTile];
                    new Span<double>(sums, count).Clear();
                    for (int c = 0; c < C; c++)
                    {
                        float* s = xp + (long)c * hw + start;
                        int p = 0;
                        for (; p + Vector256<float>.Count <= count; p += Vector256<float>.Count)
                        {
                            var (lo, hi) = Vector256.Widen(Vector256.Load(s + p));
                            Vector256.Store(Vector256.Load(sums + p) + lo * lo, sums + p);
                            Vector256.Store(Vector256.Load(sums + p + 4) + hi * hi, sums + p + 4);
                        }
                        for (; p < count; p++) { double v = s[p]; sums[p] += v * v; }
                    }
                    {
                        int p = 0;
                        var eps = Vector256.Create(1e-12);
                        for (; p + 8 <= count; p += 8)
                        {
                            var a = Vector256.Create(1.0) / Vector256.Sqrt(Vector256.Load(sums + p) + eps);
                            var b = Vector256.Create(1.0) / Vector256.Sqrt(Vector256.Load(sums + p + 4) + eps);
                            Vector256.Store(Vector256.Narrow(a, b), inv + p);
                        }
                        for (; p < count; p++) inv[p] = (float)(1.0 / Math.Sqrt(sums[p] + 1e-12));
                    }
                    var vScale = Vector256.Create(scale);
                    for (int c = 0; c < C; c++)
                    {
                        float* s = xp + (long)c * hw + start;
                        float* d = op + (long)c * hw + start;
                        float g = gp[c];
                        var vg = Vector256.Create(g);
                        int p = 0;
                        for (; p + 8 <= count; p += 8)
                        {
                            var y = Vector256.Load(s + p) * Vector256.Load(inv + p) * vScale * vg;
                            if (silu) y = SiluVector(y);
                            Vector256.Store(y, d + p);
                        }
                        for (; p < count; p++)
                        {
                            float y = s[p] * inv[p] * scale * g;
                            d[p] = silu ? y / (1f + MathF.Exp(-y)) : y;
                        }
                    }
                }, CpuPool);
            }
            return outp;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<float> SiluVector(Vector256<float> y) => y / (Vector256<float>.One + Vector256.Exp(-y));

        /// <summary>Vectorized SiLU (managed fast path only; see the file header on exactness).</summary>
        internal static void SiluInPlaceFast(float[] d)
        {
            const int chunk = 64 * 1024;
            int chunks = (d.Length + chunk - 1) / chunk;
            fixed (float* p0 = d)
            {
                nint pL = (nint)p0;
                int length = d.Length;
                CpuPackedGemm.ForEach(chunks, chunks > 1, i =>
                {
                    float* p = (float*)pL + (long)i * chunk;
                    int count = Math.Min(chunk, length - i * chunk), j = 0;
                    for (; j + 8 <= count; j += 8) Vector256.Store(SiluVector(Vector256.Load(p + j)), p + j);
                    for (; j < count; j++) { float v = p[j]; p[j] = v / (1f + MathF.Exp(-v)); }
                }, CpuPool);
            }
        }

        /// <summary>a += b, vectorized and chunked over the pool (bit-identical).</summary>
        private static Feature AddInPlace(Feature a, Feature b)
        {
            if (a.D.Length != b.D.Length) throw new ArgumentException("feature sizes differ");
            const int chunk = 64 * 1024;
            int chunks = (a.D.Length + chunk - 1) / chunk;
            fixed (float* ap0 = a.D, bp0 = b.D)
            {
                nint aL = (nint)ap0, bL = (nint)bp0;
                int length = a.D.Length;
                CpuPackedGemm.ForEach(chunks, chunks > 1, i =>
                {
                    float* ap = (float*)aL + (long)i * chunk, bp = (float*)bL + (long)i * chunk;
                    int count = Math.Min(chunk, length - i * chunk), j = 0;
                    for (; j + 8 <= count; j += 8) Vector256.Store(Vector256.Load(ap + j) + Vector256.Load(bp + j), ap + j);
                    for (; j < count; j++) ap[j] += bp[j];
                }, CpuPool);
            }
            return a;
        }

        private static Feature NearestUpsample2x(Feature x)
        {
            int C = x.C, H = x.H, W = x.W, Wo = W * 2;
            var outp = Feature.Uninitialized(C, H * 2, Wo);
            fixed (float* xp0 = x.D, op0 = outp.D)
            {
                nint xL = (nint)xp0, oL = (nint)op0;
                CpuPackedGemm.ForEach(C * H, (long)C * H * W >= 1 << 14, row =>
                {
                    float* s = (float*)xL + (long)row * W;
                    float* d = (float*)oL + (long)row * 2 * Wo;   // output rows 2*y and 2*y+1 of channel c
                    for (int i = 0; i < W; i++) { float v = s[i]; d[2 * i] = v; d[2 * i + 1] = v; }
                    Buffer.MemoryCopy(d, d + Wo, Wo * sizeof(float), Wo * sizeof(float));
                }, CpuPool);
            }
            return outp;
        }

        // ---- attention -------------------------------------------------------------

        /// <summary>
        /// Single-head spatial attention over planar q, k, v ([C, hw] each, stacked in
        /// <paramref name="qkv"/>): out[c, i] = sum_j softmax_j(q_i . k_j * scale) v[c, j].
        /// Queries are processed in blocks, so the score matrix is bounded to one
        /// [block, hw] slab instead of hw x hw (1 GB at a 2048 px decode).
        /// </summary>
        internal static Feature AttentionCpu(float[] qkv, int C, int H, int W)
        {
            int hw = H * W;
            if (qkv.LongLength < 3L * C * hw)
                throw new ArgumentException($"qkv holds {qkv.LongLength} values, expected 3 x {C} x {hw}.", nameof(qkv));
            float scale = 1f / MathF.Sqrt(C);
            var outp = Feature.Uninitialized(C, H, W);
            CpuGemmIsa isa = CpuPackedGemm.Isa;
            int mr = CpuPackedGemm.Mr(isa);
            // ~16M floats (64 MB) of scores at most, in whole A panels.
            int block = (int)Math.Clamp(16L * 1024 * 1024 / hw / mr * mr, mr, ((hw + mr - 1) / mr) * mr);
            float[] scores = GC.AllocateUninitializedArray<float>(checked(block * hw));
            fixed (float* qp = qkv, op = outp.D, sp = scores)
            {
                float* kp = qp + (long)C * hw, vp = qp + 2L * C * hw;
                PackedPanels qT = CpuPackedGemm.PackA(qp, hw, C, 1, hw, isa, CpuPool);   // [hw, C] = q^T
                PackedPanels v = CpuPackedGemm.PackA(vp, C, hw, hw, 1, isa, CpuPool);   // [C, hw]
                nint sL = (nint)sp;
                for (int i0 = 0; i0 < hw; i0 += block)
                {
                    int rows = Math.Min(block, hw - i0);
                    // scores[i, j] = q_i . k_j  (B = k as [C, hw], row-major)
                    CpuPackedGemm.Gemm(qT, new StridedPanelSource(kp, hw, 1, hw), hw, sp, hw,
                        panel0: i0 / mr, panelCount: (rows + mr - 1) / mr, pool: CpuPool);
                    CpuPackedGemm.ForEach(rows, (long)rows * hw >= 1 << 15, i => SoftmaxRow((float*)sL + (long)i * hw, hw, scale), CpuPool);
                    // out[c, i0 + i] = v[c, :] . p[i, :]  (B = p^T, read column-major from the slab)
                    CpuPackedGemm.Gemm(v, new StridedPanelSource(sp, 1, hw, rows), rows, op + i0, hw, pool: CpuPool);
                }
            }
            return outp;
        }

        private static void SoftmaxRow(float* s, int n, float scale)
        {
            // s = dot * scale, as the scalar loop rounds it, then exp(s - max) / sum.
            var vScale = Vector256.Create(scale);
            var vMax = Vector256.Create(float.NegativeInfinity);
            int j = 0;
            for (; j + 8 <= n; j += 8)
            {
                var t = Vector256.Load(s + j) * vScale;
                Vector256.Store(t, s + j);
                vMax = Vector256.Max(vMax, t);
            }
            float mx = float.NegativeInfinity;
            for (int l = 0; l < 8; l++) mx = MathF.Max(mx, vMax.GetElement(l));
            for (; j < n; j++) { s[j] *= scale; mx = MathF.Max(mx, s[j]); }

            var vm = Vector256.Create(mx);
            var vSum = Vector256<float>.Zero;
            j = 0;
            for (; j + 8 <= n; j += 8)
            {
                var e = Vector256.Exp(Vector256.Load(s + j) - vm);
                Vector256.Store(e, s + j);
                vSum += e;
            }
            float sum = Vector256.Sum(vSum);
            for (; j < n; j++) { float e = MathF.Exp(s[j] - mx); s[j] = e; sum += e; }
            var inv = Vector256.Create(1f / sum);
            j = 0;
            for (; j + 8 <= n; j += 8) Vector256.Store(Vector256.Load(s + j) * inv, s + j);
            float invSum = 1f / sum;
            for (; j < n; j++) s[j] *= invSum;
        }
    }
}
