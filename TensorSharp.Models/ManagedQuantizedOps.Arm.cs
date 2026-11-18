// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;

namespace TensorSharp.Models
{
    // ------------------------------------------------------------------
    // ARM64 (AdvSimd + DotProd) bodies of the per-row quantized dots.
    //
    // Until these, ARM64 ran the Q4_K, Q6_K, Q4_0 and NVFP4 dots as scalar C#
    // loops (Q8_0 alone had an SDOT body): ~90 % of an LFM3 prompt's thread time
    // sat in VecDotQ4_KQ8_KScalar / VecDotQ6_KQ8_KScalar.
    //
    // Every kernel here returns the SAME BITS as the scalar loop it replaces:
    // the integer part of each block or sub-block (the only part SDOT computes)
    // is exact whatever the summation order, and the float part runs the scalar
    // loop's operations in the scalar loop's order - the per-sub-block products
    // vectorized across sub-blocks (or blocks), which IEEE makes lane-for-lane
    // identical to the scalar multiplies, then the running sum added one lane at a
    // time in the scalar loop's order. .NET does not contract a * b + c into a
    // fused multiply-add, so the scalar loops round every product and sum, and so
    // do these. InferenceWeb.Tests/ManagedQuantArmTests.cs checks the bits.
    //
    // Being bit-identical, they change no output of any model on ARM64; they only
    // replace the scalar loops' time.
    // ------------------------------------------------------------------
    internal static partial class ManagedQuantizedOps
    {
        /// <summary>The ARM64 dot-product kernels can run (AdvSimd and the DotProd extension).</summary>
        internal static bool ArmDotSupported => Dp.IsSupported && AdvSimd.Arm64.IsSupported;

        /// <summary>2^112: an fp16 magnitude's bits shifted into a float's place, times this, are its value.</summary>
        private const int TwoPow112Bits = (127 + 112) << 23;

        /// <summary>
        /// Four fp16 bit patterns (the low 16 bits of each lane) to float, exactly as
        /// <c>(float)BitConverter.UInt16BitsToHalf</c> converts them: normal and subnormal values shifted into a
        /// float's exponent range by one exact multiply, infinities kept, NaN kept NaN.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static Vector128<float> HalfBitsToSingle(Vector128<uint> h)
        {
            Vector128<uint> em = h & Vector128.Create(0x7fffu);
            Vector128<uint> sign = (h & Vector128.Create(0x8000u)) << 16;
            Vector128<uint> shifted = em << 13;
            Vector128<float> magnitude = shifted.AsSingle() * Vector128.Create(BitConverter.Int32BitsToSingle(TwoPow112Bits));
            Vector128<uint> special = Vector128.GreaterThanOrEqual(em, Vector128.Create(0x7c00u));
            Vector128<uint> bits = Vector128.ConditionalSelect(special, shifted | Vector128.Create(0x7f800000u), magnitude.AsUInt32());
            return (bits | sign).AsSingle();
        }

        /// <summary>Four fp16 values read from four addresses.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector128<float> LoadHalf4(byte* a, byte* b, byte* c, byte* d)
            => HalfBitsToSingle(Vector128.Create((uint)ReadUInt16(a), ReadUInt16(b), ReadUInt16(c), ReadUInt16(d)));

        /// <summary>Four int32 vectors to one vector of their lane sums: [sum a, sum b, sum c, sum d].</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<int> ReduceLanes4(Vector128<int> a, Vector128<int> b, Vector128<int> c, Vector128<int> d)
            => AdvSimd.Arm64.AddPairwise(AdvSimd.Arm64.AddPairwise(a, b), AdvSimd.Arm64.AddPairwise(c, d));

        /// <summary>The running sum plus each lane in order: what a scalar loop adding four terms one by one computes.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float AddLanesInOrder(float sum, Vector128<float> t)
        {
            sum += t.ToScalar();
            sum += t.GetElement(1);
            sum += t.GetElement(2);
            sum += t.GetElement(3);
            return sum;
        }

        /// <summary>
        /// <see cref="QuantizeInt8Groups16Scalar"/> on AdvSimd, bit for bit: NaN lanes zeroed and the clamp done in
        /// float before the round (as the AVX variants do; rounding is monotonic, so the clamp commutes with it for
        /// every finite value), FRINTN being the scalar loop's round-half-to-even.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe void QuantizeInt8Groups16Neon(float* x, float invScale, sbyte* q, int n, int* groupSums)
        {
            Vector128<float> inv = Vector128.Create(invScale);
            Vector128<float> lo = Vector128.Create(-127f), hi = Vector128.Create(127f);
            for (int i = 0, g = 0; i < n; i += 16, g++)
            {
                Vector128<int> r0 = QuantizeLanes(AdvSimd.LoadVector128(x + i) * inv, lo, hi);
                Vector128<int> r1 = QuantizeLanes(AdvSimd.LoadVector128(x + i + 4) * inv, lo, hi);
                Vector128<int> r2 = QuantizeLanes(AdvSimd.LoadVector128(x + i + 8) * inv, lo, hi);
                Vector128<int> r3 = QuantizeLanes(AdvSimd.LoadVector128(x + i + 12) * inv, lo, hi);
                Vector128<short> s01 = AdvSimd.ExtractNarrowingSaturateUpper(AdvSimd.ExtractNarrowingSaturateLower(r0), r1);
                Vector128<short> s23 = AdvSimd.ExtractNarrowingSaturateUpper(AdvSimd.ExtractNarrowingSaturateLower(r2), r3);
                Vector128<sbyte> b = AdvSimd.ExtractNarrowingSaturateUpper(AdvSimd.ExtractNarrowingSaturateLower(s01), s23);
                AdvSimd.Store(q + i, b);
                groupSums[g] = AdvSimd.Arm64.AddAcrossWidening(b).ToScalar();
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<int> QuantizeLanes(Vector128<float> v, Vector128<float> lo, Vector128<float> hi)
        {
            v &= Vector128.Equals(v, v);                                    // NaN -> +0
            v = AdvSimd.Min(AdvSimd.Max(v, lo), hi);
            return AdvSimd.ConvertToInt32RoundToZero(AdvSimd.RoundToNearest(v));
        }

        // ------------------------------------------------------------------
        // Q4_K x Q8_K. Scalar order per sub-block j (8 of 32 values):
        //   sum += d8 * (d4 * sc[j] * prodSum[j] - dmin * mn[j] * q8Sum[j])
        // ------------------------------------------------------------------

        /// <summary>The 6-bit scales (bytes 0-7) and mins (bytes 8-15) of a Q4_K / Q5_K super-block's 12 packed
        /// bytes, as <see cref="GetScaleMinK4"/> unpacks them one at a time (ggml's kmask form).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector128<byte> UnpackKScalesMins(byte* packed)
        {
            uint u0 = ReadUInt32(packed), u1 = ReadUInt32(packed + 4), u2 = ReadUInt32(packed + 8);
            uint sc0 = u0 & 0x3f3f3f3fu;
            uint sc1 = (u2 & 0x0f0f0f0fu) | (((u0 >> 6) & 0x03030303u) << 4);
            uint mn0 = u1 & 0x3f3f3f3fu;
            uint mn1 = ((u2 >> 4) & 0x0f0f0f0fu) | (((u1 >> 6) & 0x03030303u) << 4);
            return Vector128.Create(sc0, sc1, mn0, mn1).AsByte();
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void WidenBytesToSingle(Vector64<byte> v, out Vector128<float> lo, out Vector128<float> hi)
        {
            Vector128<ushort> w = AdvSimd.ZeroExtendWideningLower(v);
            lo = AdvSimd.ConvertToSingle(AdvSimd.ZeroExtendWideningLower(w.GetLower()));
            hi = AdvSimd.ConvertToSingle(AdvSimd.ZeroExtendWideningUpper(w));
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe float VecDotQ4_KQ8_KNeon(byte* q4k, byte* q8k, int superBlockCount)
        {
            float sum = 0.0f;
            Vector128<byte> m4 = Vector128.Create((byte)0x0F);
            for (int block = 0; block < superBlockCount; block++)
            {
                float d4 = HalfToSingle(ReadUInt16(q4k));
                float dmin = HalfToSingle(ReadUInt16(q4k + 2));
                float d8 = ReadSingle(q8k);
                byte* qs = q4k + 16;
                sbyte* q8 = (sbyte*)(q8k + 4);
                short* bsums = (short*)(q8k + 4 + QK_K);

                Vector128<int> p0, p1;
                {
                    Vector128<byte> b0 = AdvSimd.LoadVector128(qs), b1 = AdvSimd.LoadVector128(qs + 16);
                    Vector128<byte> b2 = AdvSimd.LoadVector128(qs + 32), b3 = AdvSimd.LoadVector128(qs + 48);
                    Vector128<int> a0 = SubBlockDot32(AdvSimd.And(b0, m4), AdvSimd.And(b1, m4), q8);
                    Vector128<int> a1 = SubBlockDot32(AdvSimd.ShiftRightLogical(b0, 4), AdvSimd.ShiftRightLogical(b1, 4), q8 + 32);
                    Vector128<int> a2 = SubBlockDot32(AdvSimd.And(b2, m4), AdvSimd.And(b3, m4), q8 + 64);
                    Vector128<int> a3 = SubBlockDot32(AdvSimd.ShiftRightLogical(b2, 4), AdvSimd.ShiftRightLogical(b3, 4), q8 + 96);
                    p0 = ReduceLanes4(a0, a1, a2, a3);
                }
                {
                    Vector128<byte> b0 = AdvSimd.LoadVector128(qs + 64), b1 = AdvSimd.LoadVector128(qs + 80);
                    Vector128<byte> b2 = AdvSimd.LoadVector128(qs + 96), b3 = AdvSimd.LoadVector128(qs + 112);
                    Vector128<int> a0 = SubBlockDot32(AdvSimd.And(b0, m4), AdvSimd.And(b1, m4), q8 + 128);
                    Vector128<int> a1 = SubBlockDot32(AdvSimd.ShiftRightLogical(b0, 4), AdvSimd.ShiftRightLogical(b1, 4), q8 + 160);
                    Vector128<int> a2 = SubBlockDot32(AdvSimd.And(b2, m4), AdvSimd.And(b3, m4), q8 + 192);
                    Vector128<int> a3 = SubBlockDot32(AdvSimd.ShiftRightLogical(b2, 4), AdvSimd.ShiftRightLogical(b3, 4), q8 + 224);
                    p1 = ReduceLanes4(a0, a1, a2, a3);
                }

                Vector128<byte> scmn = UnpackKScalesMins(q4k + 4);
                WidenBytesToSingle(scmn.GetLower(), out Vector128<float> sc0, out Vector128<float> sc1);
                WidenBytesToSingle(scmn.GetUpper(), out Vector128<float> mn0, out Vector128<float> mn1);
                Vector128<int> q8s0 = AdvSimd.AddPairwiseWidening(AdvSimd.LoadVector128(bsums));
                Vector128<int> q8s1 = AdvSimd.AddPairwiseWidening(AdvSimd.LoadVector128(bsums + 8));

                Vector128<float> vd4 = Vector128.Create(d4), vdmin = Vector128.Create(dmin), vd8 = Vector128.Create(d8);
                Vector128<float> t0 = vd8 * ((vd4 * sc0) * AdvSimd.ConvertToSingle(p0) - (vdmin * mn0) * AdvSimd.ConvertToSingle(q8s0));
                Vector128<float> t1 = vd8 * ((vd4 * sc1) * AdvSimd.ConvertToSingle(p1) - (vdmin * mn1) * AdvSimd.ConvertToSingle(q8s1));
                sum = AddLanesInOrder(sum, t0);
                sum = AddLanesInOrder(sum, t1);

                q4k += Q4_KBlockBytes;
                q8k += Q8_KBlockBytes;
            }
            return sum;
        }

        /// <summary>The 32-value dot of two 16-byte weight halves with 32 activation bytes (four int32 lanes).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector128<int> SubBlockDot32(Vector128<byte> w0, Vector128<byte> w1, sbyte* x)
            => Dp.DotProduct(Dp.DotProduct(Vector128<int>.Zero, w0.AsSByte(), AdvSimd.LoadVector128(x)),
                w1.AsSByte(), AdvSimd.LoadVector128(x + 16));

        // ------------------------------------------------------------------
        // Q6_K x Q8_K. Scalar order per sub-block s (16 of 16 values):
        //   sum += (d6 * d8 * scales[s]) * isum[s]
        // ------------------------------------------------------------------

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe float VecDotQ6_KQ8_KNeon(byte* q6k, byte* q8k, int superBlockCount)
        {
            float sum = 0.0f;
            Vector128<byte> m4 = Vector128.Create((byte)0x0F);
            Vector128<byte> m30 = Vector128.Create((byte)0x30);
            Vector128<sbyte> off32 = Vector128.Create((sbyte)32);
            for (int block = 0; block < superBlockCount; block++)
            {
                byte* ql = q6k;
                byte* qh = q6k + QK_K / 2;
                sbyte* scales = (sbyte*)(q6k + QK_K / 2 + QK_K / 4);
                float d6 = HalfToSingle(ReadUInt16((byte*)(scales + QK_K / 16)));
                float d8 = ReadSingle(q8k);
                sbyte* q8 = (sbyte*)(q8k + 4);
                float scaleBase = d6 * d8;

                Vector128<int> i0, i1, i2, i3;
                Q6KHalfDots(ql, qh, q8, m4, m30, off32, out i0, out i1);
                Q6KHalfDots(ql + 64, qh + 32, q8 + 128, m4, m30, off32, out i2, out i3);

                Vector128<sbyte> sc = AdvSimd.LoadVector128(scales);
                Vector128<short> sc16lo = AdvSimd.SignExtendWideningLower(sc.GetLower());
                Vector128<short> sc16hi = AdvSimd.SignExtendWideningUpper(sc);
                Vector128<float> vbase = Vector128.Create(scaleBase);
                Vector128<float> s0 = vbase * AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningLower(sc16lo.GetLower()));
                Vector128<float> s1 = vbase * AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningUpper(sc16lo));
                Vector128<float> s2 = vbase * AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningLower(sc16hi.GetLower()));
                Vector128<float> s3 = vbase * AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningUpper(sc16hi));

                sum = AddLanesInOrder(sum, s0 * AdvSimd.ConvertToSingle(i0));
                sum = AddLanesInOrder(sum, s1 * AdvSimd.ConvertToSingle(i1));
                sum = AddLanesInOrder(sum, s2 * AdvSimd.ConvertToSingle(i2));
                sum = AddLanesInOrder(sum, s3 * AdvSimd.ConvertToSingle(i3));

                q6k += Q6_KBlockBytes;
                q8k += Q8_KBlockBytes;
            }
            return sum;
        }

        /// <summary>One Q6_K half (8 sub-blocks of 16): the signed values (low 4 bits from ql, high 2 from qh, minus 32)
        /// dotted with the activations, as two vectors of four sub-block sums in sub-block order.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void Q6KHalfDots(byte* ql, byte* qh, sbyte* q8, Vector128<byte> m4, Vector128<byte> m30,
            Vector128<sbyte> off32, out Vector128<int> first, out Vector128<int> second)
        {
            Vector128<byte> l0 = AdvSimd.LoadVector128(ql), l1 = AdvSimd.LoadVector128(ql + 16);
            Vector128<byte> l2 = AdvSimd.LoadVector128(ql + 32), l3 = AdvSimd.LoadVector128(ql + 48);
            Vector128<byte> h0 = AdvSimd.LoadVector128(qh), h1 = AdvSimd.LoadVector128(qh + 16);

            Vector128<sbyte> w0 = AdvSimd.Subtract((AdvSimd.And(l0, m4) | AdvSimd.And(AdvSimd.ShiftLeftLogical(h0, 4), m30)).AsSByte(), off32);
            Vector128<sbyte> w1 = AdvSimd.Subtract((AdvSimd.And(l1, m4) | AdvSimd.And(AdvSimd.ShiftLeftLogical(h1, 4), m30)).AsSByte(), off32);
            Vector128<sbyte> w2 = AdvSimd.Subtract((AdvSimd.And(l2, m4) | AdvSimd.And(AdvSimd.ShiftLeftLogical(h0, 2), m30)).AsSByte(), off32);
            Vector128<sbyte> w3 = AdvSimd.Subtract((AdvSimd.And(l3, m4) | AdvSimd.And(AdvSimd.ShiftLeftLogical(h1, 2), m30)).AsSByte(), off32);
            first = ReduceLanes4(
                Dp.DotProduct(Vector128<int>.Zero, w0, AdvSimd.LoadVector128(q8)),
                Dp.DotProduct(Vector128<int>.Zero, w1, AdvSimd.LoadVector128(q8 + 16)),
                Dp.DotProduct(Vector128<int>.Zero, w2, AdvSimd.LoadVector128(q8 + 32)),
                Dp.DotProduct(Vector128<int>.Zero, w3, AdvSimd.LoadVector128(q8 + 48)));

            Vector128<sbyte> w4 = AdvSimd.Subtract((AdvSimd.ShiftRightLogical(l0, 4) | AdvSimd.And(h0, m30)).AsSByte(), off32);
            Vector128<sbyte> w5 = AdvSimd.Subtract((AdvSimd.ShiftRightLogical(l1, 4) | AdvSimd.And(h1, m30)).AsSByte(), off32);
            Vector128<sbyte> w6 = AdvSimd.Subtract((AdvSimd.ShiftRightLogical(l2, 4) | AdvSimd.And(AdvSimd.ShiftRightLogical(h0, 2), m30)).AsSByte(), off32);
            Vector128<sbyte> w7 = AdvSimd.Subtract((AdvSimd.ShiftRightLogical(l3, 4) | AdvSimd.And(AdvSimd.ShiftRightLogical(h1, 2), m30)).AsSByte(), off32);
            second = ReduceLanes4(
                Dp.DotProduct(Vector128<int>.Zero, w4, AdvSimd.LoadVector128(q8 + 64)),
                Dp.DotProduct(Vector128<int>.Zero, w5, AdvSimd.LoadVector128(q8 + 80)),
                Dp.DotProduct(Vector128<int>.Zero, w6, AdvSimd.LoadVector128(q8 + 96)),
                Dp.DotProduct(Vector128<int>.Zero, w7, AdvSimd.LoadVector128(q8 + 112)));
        }

        // ------------------------------------------------------------------
        // Q4_0 / Q8_0 x Q8_0. Scalar order per block b: sum += (dw * dx) * isum[b].
        // Four blocks a step: their integer sums in one vector, their scale products in another.
        // ------------------------------------------------------------------

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector128<int> Q40BlockDot(byte* wb, byte* xb, Vector128<byte> m4, Vector128<sbyte> off8)
        {
            Vector128<byte> qs = AdvSimd.LoadVector128(wb + 2);
            Vector128<sbyte> lo = AdvSimd.Subtract(AdvSimd.And(qs, m4).AsSByte(), off8);
            Vector128<sbyte> hi = AdvSimd.Subtract(AdvSimd.ShiftRightLogical(qs, 4).AsSByte(), off8);
            return Dp.DotProduct(Dp.DotProduct(Vector128<int>.Zero, lo, AdvSimd.LoadVector128((sbyte*)(xb + 2))),
                hi, AdvSimd.LoadVector128((sbyte*)(xb + 18)));
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe float VecDotQ4_0Q8_0Neon(byte* q4, byte* q8, int blockCount)
        {
            float sum = 0.0f;
            Vector128<byte> m4 = Vector128.Create((byte)0x0F);
            Vector128<sbyte> off8 = Vector128.Create((sbyte)8);
            int block = 0;
            for (; block + 4 <= blockCount; block += 4)
            {
                byte* w = q4 + block * Q4_0BlockBytes;
                byte* x = q8 + block * Q8_0BlockBytes;
                Vector128<int> isum = ReduceLanes4(
                    Q40BlockDot(w, x, m4, off8),
                    Q40BlockDot(w + Q4_0BlockBytes, x + Q8_0BlockBytes, m4, off8),
                    Q40BlockDot(w + 2 * Q4_0BlockBytes, x + 2 * Q8_0BlockBytes, m4, off8),
                    Q40BlockDot(w + 3 * Q4_0BlockBytes, x + 3 * Q8_0BlockBytes, m4, off8));
                Vector128<float> dw = LoadHalf4(w, w + Q4_0BlockBytes, w + 2 * Q4_0BlockBytes, w + 3 * Q4_0BlockBytes);
                Vector128<float> dx = LoadHalf4(x, x + Q8_0BlockBytes, x + 2 * Q8_0BlockBytes, x + 3 * Q8_0BlockBytes);
                sum = AddLanesInOrder(sum, (dw * dx) * AdvSimd.ConvertToSingle(isum));
            }
            for (; block < blockCount; block++)
            {
                byte* w = q4 + block * Q4_0BlockBytes;
                byte* x = q8 + block * Q8_0BlockBytes;
                int isum = AdvSimd.Arm64.AddAcross(Q40BlockDot(w, x, m4, off8)).ToScalar();
                sum += HalfToSingle(ReadUInt16(w)) * HalfToSingle(ReadUInt16(x)) * isum;
            }
            return sum;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector128<int> Q80BlockDot(byte* wb, byte* xb)
            => Dp.DotProduct(Dp.DotProduct(Vector128<int>.Zero,
                    AdvSimd.LoadVector128((sbyte*)(wb + 2)), AdvSimd.LoadVector128((sbyte*)(xb + 2))),
                AdvSimd.LoadVector128((sbyte*)(wb + 18)), AdvSimd.LoadVector128((sbyte*)(xb + 18)));

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe float VecDotQ8_0Q8_0Neon(byte* q8w, byte* q8x, int blockCount)
        {
            float sum = 0.0f;
            int block = 0;
            for (; block + 4 <= blockCount; block += 4)
            {
                byte* w = q8w + block * Q8_0BlockBytes;
                byte* x = q8x + block * Q8_0BlockBytes;
                Vector128<int> isum = ReduceLanes4(
                    Q80BlockDot(w, x),
                    Q80BlockDot(w + Q8_0BlockBytes, x + Q8_0BlockBytes),
                    Q80BlockDot(w + 2 * Q8_0BlockBytes, x + 2 * Q8_0BlockBytes),
                    Q80BlockDot(w + 3 * Q8_0BlockBytes, x + 3 * Q8_0BlockBytes));
                Vector128<float> dw = LoadHalf4(w, w + Q8_0BlockBytes, w + 2 * Q8_0BlockBytes, w + 3 * Q8_0BlockBytes);
                Vector128<float> dx = LoadHalf4(x, x + Q8_0BlockBytes, x + 2 * Q8_0BlockBytes, x + 3 * Q8_0BlockBytes);
                sum = AddLanesInOrder(sum, (dw * dx) * AdvSimd.ConvertToSingle(isum));
            }
            for (; block < blockCount; block++)
            {
                byte* w = q8w + block * Q8_0BlockBytes;
                byte* x = q8x + block * Q8_0BlockBytes;
                int isum = AdvSimd.Arm64.AddAcross(Q80BlockDot(w, x)).ToScalar();
                sum += HalfToSingle(ReadUInt16(w)) * HalfToSingle(ReadUInt16(x)) * isum;
            }
            return sum;
        }

        // ------------------------------------------------------------------
        // NVFP4 x Q8_0. Scalar order per 16-value sub-block s of a 64-value block:
        //   sum += (dy[s / 2] * ue4m3(scale[s])) * (sumiLo + sumiHi)
        // The E2M1 codebook lookup is one TBL per 16 nibbles.
        // ------------------------------------------------------------------

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe float VecDotNvfp4Q8_0Neon(byte* nv, byte* q8, int blockCount)
        {
            Vector128<byte> table;
            fixed (sbyte* tbl = Mxfp4Values)
                table = AdvSimd.LoadVector128((byte*)tbl);
            Vector128<byte> m4 = Vector128.Create((byte)0x0F);
            float sum = 0.0f;
            fixed (float* ue = Ue4m3Table)
            {
                for (int block = 0; block < blockCount; block++)
                {
                    byte* xb = nv + block * Nvfp4BlockBytes;
                    byte* y0 = q8 + block * 2 * Q8_0BlockBytes;
                    byte* y1 = y0 + Q8_0BlockBytes;
                    // qs bytes 16h..16h+15 hold sub-blocks 2h (bytes 0-7) and 2h+1 (bytes 8-15); their low nibbles are
                    // each sub-block's values 0-7, the high nibbles 8-15. Activation block h is the pair's 32 values.
                    Vector128<int> p01 = Nvfp4PairDot(AdvSimd.LoadVector128(xb + 4), y0, table, m4);
                    Vector128<int> p23 = Nvfp4PairDot(AdvSimd.LoadVector128(xb + 20), y1, table, m4);
                    // [s0a, s0b, s1a, s1b] + [s2a, s2b, s3a, s3b] -> [s0, s1, s2, s3]
                    Vector128<int> isum = AdvSimd.Arm64.AddPairwise(p01, p23);
                    Vector128<float> dy = HalfBitsToSingle(Vector128.Create((uint)ReadUInt16(y0), ReadUInt16(y0), ReadUInt16(y1), ReadUInt16(y1)));
                    Vector128<float> sx = Vector128.Create(ue[xb[0]], ue[xb[1]], ue[xb[2]], ue[xb[3]]);
                    sum = AddLanesInOrder(sum, (dy * sx) * AdvSimd.ConvertToSingle(isum));
                }
            }
            return sum;
        }

        /// <summary>Two NVFP4 sub-blocks (16 packed bytes) against their 32-value Q8_0 activation block: lanes 0-1 hold the
        /// first sub-block's partial sums, lanes 2-3 the second's.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector128<int> Nvfp4PairDot(Vector128<byte> qs, byte* yb, Vector128<byte> table, Vector128<byte> m4)
        {
            Vector128<sbyte> wLo = AdvSimd.Arm64.VectorTableLookup(table, AdvSimd.And(qs, m4)).AsSByte();
            Vector128<sbyte> wHi = AdvSimd.Arm64.VectorTableLookup(table, AdvSimd.ShiftRightLogical(qs, 4)).AsSByte();
            Vector128<ulong> a0 = AdvSimd.LoadVector128(yb + 2).AsUInt64();
            Vector128<ulong> a1 = AdvSimd.LoadVector128(yb + 18).AsUInt64();
            // wLo pairs with [a0 values 0-7 | a1 values 0-7], wHi with [a0 8-15 | a1 8-15].
            Vector128<sbyte> aLo = AdvSimd.Arm64.ZipLow(a0, a1).AsSByte();
            Vector128<sbyte> aHi = AdvSimd.Arm64.ZipHigh(a0, a1).AsSByte();
            return Dp.DotProduct(Dp.DotProduct(Vector128<int>.Zero, wLo, aLo), wHi, aHi);
        }

        // ------------------------------------------------------------------
        // One activation row against four weight rows (a decode step's matvec): each column's sum is the per-row dot's,
        // operation for operation, with the four columns in vector lanes - the per-sub-block float terms one vector op for
        // all four, the running sums advanced together in the scalar order, the activation loads shared.
        // ------------------------------------------------------------------

        /// <summary>The bytes of four 16-byte vectors regrouped by index: result m holds bytes 4m..4m+3 of each source, as
        /// [b0[4m], b1[4m], b2[4m], b3[4m], b0[4m+1], ...] - so each 4-byte group is one index's value across the four.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void TransposeBytes4x16(Vector128<byte> b0, Vector128<byte> b1, Vector128<byte> b2, Vector128<byte> b3,
            out Vector128<byte> m0, out Vector128<byte> m1, out Vector128<byte> m2, out Vector128<byte> m3)
        {
            Vector128<byte> z01l = AdvSimd.Arm64.ZipLow(b0, b1), z01h = AdvSimd.Arm64.ZipHigh(b0, b1);     // [b0 i, b1 i] pairs
            Vector128<byte> z23l = AdvSimd.Arm64.ZipLow(b2, b3), z23h = AdvSimd.Arm64.ZipHigh(b2, b3);
            m0 = AdvSimd.Arm64.ZipLow(z01l.AsUInt16(), z23l.AsUInt16()).AsByte();                          // indices 0-3
            m1 = AdvSimd.Arm64.ZipHigh(z01l.AsUInt16(), z23l.AsUInt16()).AsByte();                         // 4-7
            m2 = AdvSimd.Arm64.ZipLow(z01h.AsUInt16(), z23h.AsUInt16()).AsByte();                          // 8-11
            m3 = AdvSimd.Arm64.ZipHigh(z01h.AsUInt16(), z23h.AsUInt16()).AsByte();                         // 12-15
        }

        /// <summary>Group g (0-3) of a regrouped vector - one index's byte across the four sources - as four floats
        /// (unsigned).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<float> GroupU8(Vector128<ushort> lo, Vector128<ushort> hi, int g)
            => AdvSimd.ConvertToSingle(g switch
            {
                0 => AdvSimd.ZeroExtendWideningLower(lo.GetLower()),
                1 => AdvSimd.ZeroExtendWideningUpper(lo),
                2 => AdvSimd.ZeroExtendWideningLower(hi.GetLower()),
                _ => AdvSimd.ZeroExtendWideningUpper(hi),
            });

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<float> GroupS8(Vector128<short> lo, Vector128<short> hi, int g)
            => AdvSimd.ConvertToSingle(g switch
            {
                0 => AdvSimd.SignExtendWideningLower(lo.GetLower()),
                1 => AdvSimd.SignExtendWideningUpper(lo),
                2 => AdvSimd.SignExtendWideningLower(hi.GetLower()),
                _ => AdvSimd.SignExtendWideningUpper(hi),
            });

        /// <summary>Q4_K: <see cref="VecDotQ4_KQ8_KNeon"/> for four weight rows at once (results in lanes 0-3).</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe Vector128<float> VecDotQ4_KQ8_KNeon4(byte* w0, byte* w1, byte* w2, byte* w3, byte* q8k, int superBlockCount)
        {
            Vector128<float> sum = Vector128<float>.Zero;
            Vector128<byte> m4 = Vector128.Create((byte)0x0F);
            for (int block = 0; block < superBlockCount; block++)
            {
                Vector128<float> d4 = LoadHalf4(w0, w1, w2, w3);
                Vector128<float> dmin = LoadHalf4(w0 + 2, w1 + 2, w2 + 2, w3 + 2);
                // scales (bytes 0-7) and mins (8-15) of each row, regrouped: one index across the four rows a 4-byte group
                TransposeBytes4x16(UnpackKScalesMins(w0 + 4), UnpackKScalesMins(w1 + 4), UnpackKScalesMins(w2 + 4), UnpackKScalesMins(w3 + 4),
                    out Vector128<byte> g0, out Vector128<byte> g1, out Vector128<byte> g2, out Vector128<byte> g3);
                Vector128<ushort> sc03l = AdvSimd.ZeroExtendWideningLower(g0.GetLower()), sc03h = AdvSimd.ZeroExtendWideningUpper(g0);
                Vector128<ushort> sc47l = AdvSimd.ZeroExtendWideningLower(g1.GetLower()), sc47h = AdvSimd.ZeroExtendWideningUpper(g1);
                Vector128<ushort> mn03l = AdvSimd.ZeroExtendWideningLower(g2.GetLower()), mn03h = AdvSimd.ZeroExtendWideningUpper(g2);
                Vector128<ushort> mn47l = AdvSimd.ZeroExtendWideningLower(g3.GetLower()), mn47h = AdvSimd.ZeroExtendWideningUpper(g3);
                float d8 = ReadSingle(q8k);
                Vector128<float> vd8 = Vector128.Create(d8);
                sbyte* q8 = (sbyte*)(q8k + 4);
                short* bsums = (short*)(q8k + 4 + QK_K);
                byte* qs0 = w0 + 16, qs1 = w1 + 16, qs2 = w2 + 16, qs3 = w3 + 16;
                for (int p = 0; p < 4; p++)
                {
                    sbyte* x = q8 + 64 * p;
                    Vector128<sbyte> xa = AdvSimd.LoadVector128(x), xb = AdvSimd.LoadVector128(x + 16);
                    Vector128<sbyte> xc = AdvSimd.LoadVector128(x + 32), xd = AdvSimd.LoadVector128(x + 48);
                    int at = 32 * p;
                    Vector128<byte> a0 = AdvSimd.LoadVector128(qs0 + at), b0 = AdvSimd.LoadVector128(qs0 + at + 16);
                    Vector128<byte> a1 = AdvSimd.LoadVector128(qs1 + at), b1 = AdvSimd.LoadVector128(qs1 + at + 16);
                    Vector128<byte> a2 = AdvSimd.LoadVector128(qs2 + at), b2 = AdvSimd.LoadVector128(qs2 + at + 16);
                    Vector128<byte> a3 = AdvSimd.LoadVector128(qs3 + at), b3 = AdvSimd.LoadVector128(qs3 + at + 16);
                    // sub-block 2p (low nibbles), then 2p + 1 (high nibbles)
                    Vector128<int> pLo = ReduceLanes4(
                        Sdot2(AdvSimd.And(a0, m4), AdvSimd.And(b0, m4), xa, xb), Sdot2(AdvSimd.And(a1, m4), AdvSimd.And(b1, m4), xa, xb),
                        Sdot2(AdvSimd.And(a2, m4), AdvSimd.And(b2, m4), xa, xb), Sdot2(AdvSimd.And(a3, m4), AdvSimd.And(b3, m4), xa, xb));
                    Vector128<int> pHi = ReduceLanes4(
                        Sdot2(AdvSimd.ShiftRightLogical(a0, 4), AdvSimd.ShiftRightLogical(b0, 4), xc, xd),
                        Sdot2(AdvSimd.ShiftRightLogical(a1, 4), AdvSimd.ShiftRightLogical(b1, 4), xc, xd),
                        Sdot2(AdvSimd.ShiftRightLogical(a2, 4), AdvSimd.ShiftRightLogical(b2, 4), xc, xd),
                        Sdot2(AdvSimd.ShiftRightLogical(a3, 4), AdvSimd.ShiftRightLogical(b3, 4), xc, xd));
                    int j0 = 2 * p, j1 = 2 * p + 1;
                    float q0 = bsums[2 * j0] + bsums[2 * j0 + 1], q1 = bsums[2 * j1] + bsums[2 * j1 + 1];
                    Vector128<float> sc0 = j0 < 4 ? GroupU8(sc03l, sc03h, j0) : GroupU8(sc47l, sc47h, j0 - 4);
                    Vector128<float> sc1 = j1 < 4 ? GroupU8(sc03l, sc03h, j1) : GroupU8(sc47l, sc47h, j1 - 4);
                    Vector128<float> mn0 = j0 < 4 ? GroupU8(mn03l, mn03h, j0) : GroupU8(mn47l, mn47h, j0 - 4);
                    Vector128<float> mn1 = j1 < 4 ? GroupU8(mn03l, mn03h, j1) : GroupU8(mn47l, mn47h, j1 - 4);
                    sum += vd8 * ((d4 * sc0) * AdvSimd.ConvertToSingle(pLo) - (dmin * mn0) * Vector128.Create(q0));
                    sum += vd8 * ((d4 * sc1) * AdvSimd.ConvertToSingle(pHi) - (dmin * mn1) * Vector128.Create(q1));
                }
                w0 += Q4_KBlockBytes; w1 += Q4_KBlockBytes; w2 += Q4_KBlockBytes; w3 += Q4_KBlockBytes;
                q8k += Q8_KBlockBytes;
            }
            return sum;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<int> Sdot2(Vector128<byte> w0, Vector128<byte> w1, Vector128<sbyte> x0, Vector128<sbyte> x1)
            => Dp.DotProduct(Dp.DotProduct(Vector128<int>.Zero, w0.AsSByte(), x0), w1.AsSByte(), x1);

        /// <summary>Q6_K: <see cref="VecDotQ6_KQ8_KNeon"/> for four weight rows at once (results in lanes 0-3).</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe Vector128<float> VecDotQ6_KQ8_KNeon4(byte* w0, byte* w1, byte* w2, byte* w3, byte* q8k, int superBlockCount)
        {
            Vector128<float> sum = Vector128<float>.Zero;
            Vector128<byte> m4 = Vector128.Create((byte)0x0F);
            Vector128<byte> m30 = Vector128.Create((byte)0x30);
            Vector128<sbyte> off32 = Vector128.Create((sbyte)32);
            for (int block = 0; block < superBlockCount; block++)
            {
                const int scalesAt = QK_K / 2 + QK_K / 4;
                Vector128<float> d6 = LoadHalf4(w0 + scalesAt + QK_K / 16, w1 + scalesAt + QK_K / 16, w2 + scalesAt + QK_K / 16, w3 + scalesAt + QK_K / 16);
                float d8 = ReadSingle(q8k);
                Vector128<float> vbase = d6 * Vector128.Create(d8);
                TransposeBytes4x16(AdvSimd.LoadVector128(w0 + scalesAt), AdvSimd.LoadVector128(w1 + scalesAt),
                    AdvSimd.LoadVector128(w2 + scalesAt), AdvSimd.LoadVector128(w3 + scalesAt),
                    out Vector128<byte> g0, out Vector128<byte> g1, out Vector128<byte> g2, out Vector128<byte> g3);
                sbyte* q8 = (sbyte*)(q8k + 4);
                for (int h = 0; h < 2; h++)
                {
                    Vector128<byte> gA = h == 0 ? g0 : g2, gB = h == 0 ? g1 : g3;   // sub-blocks 8h..8h+3 and 8h+4..8h+7
                    Vector128<short> sAl = AdvSimd.SignExtendWideningLower(gA.AsSByte().GetLower()), sAh = AdvSimd.SignExtendWideningUpper(gA.AsSByte());
                    Vector128<short> sBl = AdvSimd.SignExtendWideningLower(gB.AsSByte().GetLower()), sBh = AdvSimd.SignExtendWideningUpper(gB.AsSByte());
                    Q6KHalf4(w0, w1, w2, w3, h, q8 + 128 * h, m4, m30, off32,
                        out Vector128<int> i0, out Vector128<int> i1, out Vector128<int> i2, out Vector128<int> i3,
                        out Vector128<int> i4, out Vector128<int> i5, out Vector128<int> i6, out Vector128<int> i7);
                    sum += (vbase * GroupS8(sAl, sAh, 0)) * AdvSimd.ConvertToSingle(i0);
                    sum += (vbase * GroupS8(sAl, sAh, 1)) * AdvSimd.ConvertToSingle(i1);
                    sum += (vbase * GroupS8(sAl, sAh, 2)) * AdvSimd.ConvertToSingle(i2);
                    sum += (vbase * GroupS8(sAl, sAh, 3)) * AdvSimd.ConvertToSingle(i3);
                    sum += (vbase * GroupS8(sBl, sBh, 0)) * AdvSimd.ConvertToSingle(i4);
                    sum += (vbase * GroupS8(sBl, sBh, 1)) * AdvSimd.ConvertToSingle(i5);
                    sum += (vbase * GroupS8(sBl, sBh, 2)) * AdvSimd.ConvertToSingle(i6);
                    sum += (vbase * GroupS8(sBl, sBh, 3)) * AdvSimd.ConvertToSingle(i7);
                }
                w0 += Q6_KBlockBytes; w1 += Q6_KBlockBytes; w2 += Q6_KBlockBytes; w3 += Q6_KBlockBytes;
                q8k += Q8_KBlockBytes;
            }
            return sum;
        }

        /// <summary>One Q6_K half (sub-blocks 8h..8h+7) of four rows against the activations: each sub-block's four row sums
        /// in one vector.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void Q6KHalf4(byte* w0, byte* w1, byte* w2, byte* w3, int h, sbyte* q8, Vector128<byte> m4,
            Vector128<byte> m30, Vector128<sbyte> off32,
            out Vector128<int> i0, out Vector128<int> i1, out Vector128<int> i2, out Vector128<int> i3,
            out Vector128<int> i4, out Vector128<int> i5, out Vector128<int> i6, out Vector128<int> i7)
        {
            Q6KHalfRaw(w0 + 64 * h, w0 + QK_K / 2 + 32 * h, q8, m4, m30, off32, out var a0, out var a1, out var a2, out var a3, out var a4, out var a5, out var a6, out var a7);
            Q6KHalfRaw(w1 + 64 * h, w1 + QK_K / 2 + 32 * h, q8, m4, m30, off32, out var b0, out var b1, out var b2, out var b3, out var b4, out var b5, out var b6, out var b7);
            Q6KHalfRaw(w2 + 64 * h, w2 + QK_K / 2 + 32 * h, q8, m4, m30, off32, out var c0, out var c1, out var c2, out var c3, out var c4, out var c5, out var c6, out var c7);
            Q6KHalfRaw(w3 + 64 * h, w3 + QK_K / 2 + 32 * h, q8, m4, m30, off32, out var e0, out var e1, out var e2, out var e3, out var e4, out var e5, out var e6, out var e7);
            i0 = ReduceLanes4(a0, b0, c0, e0); i1 = ReduceLanes4(a1, b1, c1, e1);
            i2 = ReduceLanes4(a2, b2, c2, e2); i3 = ReduceLanes4(a3, b3, c3, e3);
            i4 = ReduceLanes4(a4, b4, c4, e4); i5 = ReduceLanes4(a5, b5, c5, e5);
            i6 = ReduceLanes4(a6, b6, c6, e6); i7 = ReduceLanes4(a7, b7, c7, e7);
        }

        /// <summary>One Q6_K half of one row: each sub-block's SDOT lanes (unreduced), in sub-block order.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void Q6KHalfRaw(byte* ql, byte* qh, sbyte* q8, Vector128<byte> m4, Vector128<byte> m30, Vector128<sbyte> off32,
            out Vector128<int> s0, out Vector128<int> s1, out Vector128<int> s2, out Vector128<int> s3,
            out Vector128<int> s4, out Vector128<int> s5, out Vector128<int> s6, out Vector128<int> s7)
        {
            Vector128<byte> l0 = AdvSimd.LoadVector128(ql), l1 = AdvSimd.LoadVector128(ql + 16);
            Vector128<byte> l2 = AdvSimd.LoadVector128(ql + 32), l3 = AdvSimd.LoadVector128(ql + 48);
            Vector128<byte> h0 = AdvSimd.LoadVector128(qh), h1 = AdvSimd.LoadVector128(qh + 16);
            s0 = Dp.DotProduct(Vector128<int>.Zero, AdvSimd.Subtract((AdvSimd.And(l0, m4) | AdvSimd.And(AdvSimd.ShiftLeftLogical(h0, 4), m30)).AsSByte(), off32), AdvSimd.LoadVector128(q8));
            s1 = Dp.DotProduct(Vector128<int>.Zero, AdvSimd.Subtract((AdvSimd.And(l1, m4) | AdvSimd.And(AdvSimd.ShiftLeftLogical(h1, 4), m30)).AsSByte(), off32), AdvSimd.LoadVector128(q8 + 16));
            s2 = Dp.DotProduct(Vector128<int>.Zero, AdvSimd.Subtract((AdvSimd.And(l2, m4) | AdvSimd.And(AdvSimd.ShiftLeftLogical(h0, 2), m30)).AsSByte(), off32), AdvSimd.LoadVector128(q8 + 32));
            s3 = Dp.DotProduct(Vector128<int>.Zero, AdvSimd.Subtract((AdvSimd.And(l3, m4) | AdvSimd.And(AdvSimd.ShiftLeftLogical(h1, 2), m30)).AsSByte(), off32), AdvSimd.LoadVector128(q8 + 48));
            s4 = Dp.DotProduct(Vector128<int>.Zero, AdvSimd.Subtract((AdvSimd.ShiftRightLogical(l0, 4) | AdvSimd.And(h0, m30)).AsSByte(), off32), AdvSimd.LoadVector128(q8 + 64));
            s5 = Dp.DotProduct(Vector128<int>.Zero, AdvSimd.Subtract((AdvSimd.ShiftRightLogical(l1, 4) | AdvSimd.And(h1, m30)).AsSByte(), off32), AdvSimd.LoadVector128(q8 + 80));
            s6 = Dp.DotProduct(Vector128<int>.Zero, AdvSimd.Subtract((AdvSimd.ShiftRightLogical(l2, 4) | AdvSimd.And(AdvSimd.ShiftRightLogical(h0, 2), m30)).AsSByte(), off32), AdvSimd.LoadVector128(q8 + 96));
            s7 = Dp.DotProduct(Vector128<int>.Zero, AdvSimd.Subtract((AdvSimd.ShiftRightLogical(l3, 4) | AdvSimd.And(AdvSimd.ShiftRightLogical(h1, 2), m30)).AsSByte(), off32), AdvSimd.LoadVector128(q8 + 112));
        }

        /// <summary>Q4_0 / Q8_0: <see cref="VecDotQ4_0Q8_0Neon"/> / <see cref="VecDotQ8_0Q8_0Neon"/> for four weight rows at once.
        /// Per block: (dw * dx) * isum added to each row's sum in block order, as the per-row loop (and its four-block steps,
        /// whose lanes it adds one by one) does.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe Vector128<float> VecDotQ0Neon4(bool q4, byte* w0, byte* w1, byte* w2, byte* w3, byte* q8, int blockCount)
        {
            Vector128<float> sum = Vector128<float>.Zero;
            Vector128<byte> m4 = Vector128.Create((byte)0x0F);
            Vector128<sbyte> off8 = Vector128.Create((sbyte)8);
            int wb = q4 ? Q4_0BlockBytes : Q8_0BlockBytes;
            for (int block = 0; block < blockCount; block++)
            {
                byte* x = q8 + block * Q8_0BlockBytes;
                long o = (long)block * wb;
                Vector128<int> isum = q4
                    ? ReduceLanes4(Q40BlockDot(w0 + o, x, m4, off8), Q40BlockDot(w1 + o, x, m4, off8),
                        Q40BlockDot(w2 + o, x, m4, off8), Q40BlockDot(w3 + o, x, m4, off8))
                    : ReduceLanes4(Q80BlockDot(w0 + o, x), Q80BlockDot(w1 + o, x), Q80BlockDot(w2 + o, x), Q80BlockDot(w3 + o, x));
                Vector128<float> dw = LoadHalf4(w0 + o, w1 + o, w2 + o, w3 + o);
                float dx = HalfToSingle(ReadUInt16(x));
                sum += (dw * Vector128.Create(dx)) * AdvSimd.ConvertToSingle(isum);
            }
            return sum;
        }

        /// <summary>NVFP4: <see cref="VecDotNvfp4Q8_0Neon"/> for four weight rows at once.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe Vector128<float> VecDotNvfp4Neon4(byte* w0, byte* w1, byte* w2, byte* w3, byte* q8, int blockCount)
        {
            Vector128<byte> table;
            fixed (sbyte* tbl = Mxfp4Values)
                table = AdvSimd.LoadVector128((byte*)tbl);
            Vector128<byte> m4 = Vector128.Create((byte)0x0F);
            Vector128<float> sum = Vector128<float>.Zero;
            fixed (float* ue = Ue4m3Table)
            {
                for (int block = 0; block < blockCount; block++)
                {
                    long o = (long)block * Nvfp4BlockBytes;
                    byte* x0 = w0 + o, x1 = w1 + o, x2 = w2 + o, x3 = w3 + o;
                    byte* y0 = q8 + block * 2 * Q8_0BlockBytes;
                    byte* y1 = y0 + Q8_0BlockBytes;
                    // per row: [s0a, s0b, s1a, s1b] + [s2a, s2b, s3a, s3b] -> [s0, s1, s2, s3]
                    Vector128<int> r0 = AdvSimd.Arm64.AddPairwise(Nvfp4PairDot(AdvSimd.LoadVector128(x0 + 4), y0, table, m4), Nvfp4PairDot(AdvSimd.LoadVector128(x0 + 20), y1, table, m4));
                    Vector128<int> r1 = AdvSimd.Arm64.AddPairwise(Nvfp4PairDot(AdvSimd.LoadVector128(x1 + 4), y0, table, m4), Nvfp4PairDot(AdvSimd.LoadVector128(x1 + 20), y1, table, m4));
                    Vector128<int> r2 = AdvSimd.Arm64.AddPairwise(Nvfp4PairDot(AdvSimd.LoadVector128(x2 + 4), y0, table, m4), Nvfp4PairDot(AdvSimd.LoadVector128(x2 + 20), y1, table, m4));
                    Vector128<int> r3 = AdvSimd.Arm64.AddPairwise(Nvfp4PairDot(AdvSimd.LoadVector128(x3 + 4), y0, table, m4), Nvfp4PairDot(AdvSimd.LoadVector128(x3 + 20), y1, table, m4));
                    // sub-block s of the four rows: [r0[s], r1[s], r2[s], r3[s]]
                    Vector128<int> t01l = AdvSimd.Arm64.TransposeEven(r0, r1), t01h = AdvSimd.Arm64.TransposeOdd(r0, r1);
                    Vector128<int> t23l = AdvSimd.Arm64.TransposeEven(r2, r3), t23h = AdvSimd.Arm64.TransposeOdd(r2, r3);
                    Vector128<int> i0 = AdvSimd.Arm64.TransposeEven(t01l.AsUInt64(), t23l.AsUInt64()).AsInt32();
                    Vector128<int> i1 = AdvSimd.Arm64.TransposeEven(t01h.AsUInt64(), t23h.AsUInt64()).AsInt32();
                    Vector128<int> i2 = AdvSimd.Arm64.TransposeOdd(t01l.AsUInt64(), t23l.AsUInt64()).AsInt32();
                    Vector128<int> i3 = AdvSimd.Arm64.TransposeOdd(t01h.AsUInt64(), t23h.AsUInt64()).AsInt32();
                    Vector128<float> dy0 = Vector128.Create(HalfToSingle(ReadUInt16(y0))), dy1 = Vector128.Create(HalfToSingle(ReadUInt16(y1)));
                    sum += (dy0 * Vector128.Create(ue[x0[0]], ue[x1[0]], ue[x2[0]], ue[x3[0]])) * AdvSimd.ConvertToSingle(i0);
                    sum += (dy0 * Vector128.Create(ue[x0[1]], ue[x1[1]], ue[x2[1]], ue[x3[1]])) * AdvSimd.ConvertToSingle(i1);
                    sum += (dy1 * Vector128.Create(ue[x0[2]], ue[x1[2]], ue[x2[2]], ue[x3[2]])) * AdvSimd.ConvertToSingle(i2);
                    sum += (dy1 * Vector128.Create(ue[x0[3]], ue[x1[3]], ue[x2[3]], ue[x3[3]])) * AdvSimd.ConvertToSingle(i3);
                }
            }
            return sum;
        }

        /// <summary>Weight-row bytes up to which a decode matvec takes the four-row kernels (a K = 2048 row of Q4_K, Q6_K, Q4_0
        /// or Q8_0); NVFP4 takes them at any length.</summary>
        internal const int Dot4RowsMaxRowBytes = 2304;

        /// <summary>The four-row matvec kernel for <paramref name="type"/> on this host, or false.</summary>
        internal static bool HasDot4Rows(GgmlTensorType type) => ArmDotSupported
            && type is GgmlTensorType.Q4_K or GgmlTensorType.Q6_K or GgmlTensorType.Q4_0 or GgmlTensorType.Q8_0 or GgmlTensorType.NVFP4;

        /// <summary>Four weight rows against one activation row: each lane the per-row dot's bits.</summary>
        internal static unsafe Vector128<float> DotQuantized4Rows(GgmlTensorType type, byte* w0, byte* w1, byte* w2, byte* w3,
            byte* activationRow, int elementCount)
            => type switch
            {
                GgmlTensorType.Q4_K => VecDotQ4_KQ8_KNeon4(w0, w1, w2, w3, activationRow, elementCount / QK_K),
                GgmlTensorType.Q6_K => VecDotQ6_KQ8_KNeon4(w0, w1, w2, w3, activationRow, elementCount / QK_K),
                GgmlTensorType.Q4_0 => VecDotQ0Neon4(true, w0, w1, w2, w3, activationRow, elementCount / QK4_0),
                GgmlTensorType.Q8_0 => VecDotQ0Neon4(false, w0, w1, w2, w3, activationRow, elementCount / QK8_0),
                _ => VecDotNvfp4Neon4(w0, w1, w2, w3, activationRow, elementCount / QK_NVFP4),
            };

        // ---- test hooks ----------------------------------------------------

        /// <summary>The scalar loop of a per-row dot, for tests that check the ARM64 bodies against it bit for bit.</summary>
        internal static unsafe float DotQuantizedRowScalar(GgmlTensorType type, byte* weightRow, byte* activationRow, int elementCount)
        {
            return type switch
            {
                GgmlTensorType.Q4_K => VecDotQ4_KQ8_KScalar(weightRow, activationRow, elementCount / QK_K),
                GgmlTensorType.Q6_K => VecDotQ6_KQ8_KScalar(weightRow, activationRow, elementCount / QK_K),
                GgmlTensorType.Q4_0 => VecDotQ4_0Q8_0Scalar(weightRow, activationRow, elementCount / QK4_0),
                GgmlTensorType.Q8_0 => VecDotQ8_0Q8_0Scalar(weightRow, activationRow, elementCount / QK8_0),
                GgmlTensorType.NVFP4 => VecDotNvfp4Q8_0Scalar(weightRow, activationRow, elementCount / QK_NVFP4),
                _ => throw new NotSupportedException($"No scalar reference for {type}."),
            };
        }
    }
}
