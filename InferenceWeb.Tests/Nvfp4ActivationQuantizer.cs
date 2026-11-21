// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace InferenceWeb.Tests;

/// <summary>
/// ggml-cuda's NVFP4 activation quantizer (ggml/src/ggml-cuda/quantize.cu, quantize_mmq_nvfp4 at ggml d9fe7bff), written on
/// the host from its source: the W4A4 products of NVFP4 weights on Blackwell quantize their activation rows this way.
/// <list type="bullet">
/// <item>A row's scale is its largest magnitude over 6 x 448: E2M1's largest value times E4M3's.</item>
/// <item>Each 16 values take a UE4M3 scale: the code of their largest magnitude (over the row scale) over 6, rounded to
/// nearest even and saturated, or of the four codes beside it (-1, +1, -2, +2, in that order) when one gives a smaller
/// squared error; the first of equal errors stays.</item>
/// <item>Each value is coded as E2M1 at that scale: round to nearest even, saturated at 6, the sign kept (also of zero).</item>
/// </list>
/// The planes are TensorSharp's (<c>CudaFp4Kernels.Quantize</c>): codes [rows][k/2], where within 16 values byte j holds value
/// j in its low nibble and value j + 8 in its high one; scale codes [rows][k/16]; row scales [rows].
/// </summary>
internal static class Nvfp4ActivationQuantizer
{
    private static readonly float[] E2M1Magnitudes = { 0f, 0.5f, 1f, 1.5f, 2f, 3f, 4f, 6f };
    private static readonly float[] E4M3 = BuildE4M3();

    private static float[] BuildE4M3()
    {
        var v = new float[0x7F];   // codes 0..0x7E; 0x7F is NaN
        for (int c = 0; c < v.Length; c++)
        {
            int e = (c >> 3) & 15, m = c & 7;
            v[c] = e == 0 ? m / 512f : (1f + m / 8f) * MathF.Pow(2f, e - 7);
        }
        return v;
    }

    /// <summary>ggml's ggml_cuda_ue4m3_to_fp32: the E4M3 value halved (its E2M1 table is doubled).</summary>
    public static float Ue4m3(int code) => code == 0 || code == 0x7F ? 0f : 0.5f * E4M3[code];

    /// <summary>__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3) for x &gt; 0 (0 otherwise, as ggml_cuda_fp32_to_ue4m3).</summary>
    public static int E4m3Code(float x)
    {
        if (!(x > 0f)) return 0;
        if (x >= E4M3[0x7E]) return 0x7E;
        int lo = 0;
        while (E4M3[lo + 1] <= x) lo++;
        if (E4M3[lo] == x) return lo;
        // Halfway points are exact in double: the codes' values are short binary fractions.
        double mid = 0.5 * ((double)E4M3[lo] + E4M3[lo + 1]);
        return x < mid ? lo : x > mid ? lo + 1 : (lo & 1) == 0 ? lo : lo + 1;
    }

    /// <summary>__nv_fp4x4_e2m1's conversion of one value: the 4-bit code (sign in bit 3).</summary>
    public static int E2m1Code(float x)
    {
        int sign = float.IsNegative(x) ? 8 : 0;
        float a = MathF.Abs(x);
        if (a >= 6f) return sign | 7;
        int lo = 0;
        while (lo < 7 && E2M1Magnitudes[lo + 1] <= a) lo++;
        if (E2M1Magnitudes[lo] == a) return sign | lo;
        double mid = 0.5 * ((double)E2M1Magnitudes[lo] + E2M1Magnitudes[lo + 1]);
        return sign | (a < mid ? lo : a > mid ? lo + 1 : (lo & 1) == 0 ? lo : lo + 1);
    }

    public static float E2m1Value(int code) => (code & 8) != 0 ? -E2M1Magnitudes[code & 7] : E2M1Magnitudes[code & 7];

    /// <summary>The squared error of 16 values coded at <paramref name="scale"/>, as quantize.cu's nvfp4_native_scale_error
    /// sums it. <paramref name="contracted"/>: |v| - 2 |d| scale as one fused multiply-add, the form nvcc's default
    /// --fmad=true gives that expression.</summary>
    private static float Error(ReadOnlySpan<float> v, float invRow, float invScale, float scale, bool contracted)
    {
        float err = 0f;
        for (int i = 0; i < 16; i++)
        {
            float x = MathF.Abs(v[i] * invRow);
            float d = MathF.Abs(E2m1Value(E2m1Code(v[i] * invRow * invScale)));
            float e = contracted ? MathF.FusedMultiplyAdd(-(d * 2f), scale, x) : x - d * 2f * scale;
            err = MathF.FusedMultiplyAdd(e, e, err);
        }
        return err;
    }

    /// <summary>Quantizes <paramref name="x"/> [rows, k] (k a multiple of 16) into the planes.</summary>
    public static void Quantize(float[] x, int rows, int k, byte[] codes, byte[] scales, float[] rowScale, bool contracted = true)
    {
        Span<int> offsets = stackalloc[] { -1, 1, -2, 2 };
        for (int r = 0; r < rows; r++)
        {
            ReadOnlySpan<float> row = x.AsSpan(r * k, k);
            float amax = 0f;
            foreach (float v in row) amax = MathF.Max(amax, MathF.Abs(v));
            float rs = amax / (6f * 448f);
            float invRow = rs > 0f ? 1f / rs : 0f;
            rowScale[r] = rs;
            for (int sb = 0; sb < k / 16; sb++)
            {
                ReadOnlySpan<float> v = row.Slice(sb * 16, 16);
                float amaxSub = 0f;
                for (int i = 0; i < 16; i++) amaxSub = MathF.Max(amaxSub, MathF.Abs(v[i] * invRow));
                int first = E4m3Code(amaxSub / 6f), code = first;
                float scale = Ue4m3(code);
                float best = Error(v, invRow, scale > 0f ? 0.5f / scale : 0f, scale, contracted);
                foreach (int off in offsets)
                {
                    int c = first + off;
                    if (c < 0 || c > 0x7E) continue;
                    float sc = Ue4m3(c);
                    float err = Error(v, invRow, sc > 0f ? 0.5f / sc : 0f, sc, contracted);
                    if (err < best) { best = err; code = c; scale = sc; }
                }
                float s = invRow * (scale > 0f ? 0.5f / scale : 0f);
                scales[(long)r * (k / 16) + sb] = (byte)code;
                for (int j = 0; j < 8; j++)
                    codes[(long)r * (k / 2) + sb * 8 + j] = (byte)(E2m1Code(v[j] * s) | (E2m1Code(v[j + 8] * s) << 4));
            }
        }
    }

    /// <summary>Rows [rows, k] replaced by the values their coding stands for (the W4A4 products' activations).</summary>
    public static void RoundRows(float[] x, int rows, int k)
    {
        var codes = new byte[(long)rows * k / 2];
        var scales = new byte[(long)rows * k / 16];
        var rowScale = new float[rows];
        Quantize(x, rows, k, codes, scales, rowScale);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < k; c++)
                x[(long)r * k + c] = Dequantize(codes, scales, rowScale, k, r, c);
    }

    /// <summary>The value a coded activation stands for: E2M1 x UE4M3 (undoubled, unhalved) x the row scale.</summary>
    public static float Dequantize(byte[] codes, byte[] scales, float[] rowScale, int k, int row, int col)
    {
        int b = codes[(long)row * (k / 2) + col / 16 * 8 + col % 8];
        int nibble = col % 16 < 8 ? b & 15 : b >> 4;
        return E2m1Value(nibble) * (2f * Ue4m3(scales[(long)row * (k / 16) + col / 16])) * rowScale[row];
    }
}
