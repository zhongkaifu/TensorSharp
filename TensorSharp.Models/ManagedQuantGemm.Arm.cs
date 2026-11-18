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
using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Threading.Tasks;

namespace TensorSharp.Models
{
    // ------------------------------------------------------------------
    // Multi-row quantized GEMM for ARM64 (AdvSimd + DotProd).
    //
    // The per-row dots (ManagedQuantizedOps.Arm.cs) decode every weight block
    // again for every activation row and reduce every sub-block's SDOT lanes
    // horizontally. Here a group of four weight rows (four output columns) is
    // decoded ONCE per row block into an L1-resident scratch laid out for
    // SDOT-by-element: each 16-byte vector holds four consecutive int8 values of
    // each of the four columns, so one SDOT (by element) adds a k-quad of one
    // activation row into four column lanes - 16 multiply-adds, no horizontal
    // reduction. A tile of four activation rows runs against the scratch with
    // the column sums in vector lanes.
    //
    // Every output is the SAME BITS as the per-row path (which is the scalar
    // loop's bits): each block's or sub-block's integer sum is exact, and the
    // float part is the scalar loop's per-block expression evaluated lane-wise
    // (columns in lanes, IEEE multiplies and adds are per lane) with the running
    // sum advanced block by block in the scalar order. So the row count of a call
    // decides only the speed, never a result: decode, prompt chunks and batches
    // agree bit for bit (what QGemmMinRows guarantees on x86), and the routing
    // threshold is picked by timing alone. ManagedQuantArmTests checks the bits.
    // ------------------------------------------------------------------
    internal static partial class ManagedQuantizedOps
    {
        private enum ArmGemmKind
        {
            None,
            /// <summary>Q4_K: 32-value sub-blocks, 6-bit scale and min (min term off the activation's bsum pairs).</summary>
            Q4K,
            /// <summary>Q6_K: 16-value sub-blocks with int8 scales, values minus 32.</summary>
            Q6K,
            /// <summary>Q4_0 / Q8_0: 32-value blocks with one fp16 scale.</summary>
            Q0,
            /// <summary>NVFP4: 16-value sub-blocks with a UE4M3 scale, against Q8_0 activations.</summary>
            Nvfp4,
        }

        /// <summary>
        /// Smallest row count that takes the ARM GEMM; fewer rows run the per-row dots. Both give the same bits, so this
        /// is a speed choice only.
        /// </summary>
        private const int ArmGemmMinRows = 2;

        private static ArmGemmKind GetArmGemmKind(GgmlTensorType type, int inDim)
        {
            switch (type)
            {
                case GgmlTensorType.Q4_K:
                    return inDim % QK_K == 0 ? ArmGemmKind.Q4K : ArmGemmKind.None;
                case GgmlTensorType.Q6_K:
                    return inDim % QK_K == 0 ? ArmGemmKind.Q6K : ArmGemmKind.None;
                case GgmlTensorType.Q4_0:
                case GgmlTensorType.Q8_0:
                    return inDim % QK8_0 == 0 ? ArmGemmKind.Q0 : ArmGemmKind.None;
                case GgmlTensorType.NVFP4:
                    return inDim % QK_NVFP4 == 0 ? ArmGemmKind.Nvfp4 : ArmGemmKind.None;
                default:
                    return ArmGemmKind.None;
            }
        }

        // Activation row: int8 values [inDim] | float side data, padded to 64 bytes.
        //   Q4K  : d8[nsb] | q8Sum[nsb * 8] (each sub-block's bsum pair, exact in float)
        //   Q6K  : d8[nsb]
        //   Q0, Nvfp4 : dx[inDim / 32] (the fp16-rounded block scale the per-row dot reads back)
        private static int ArmActSideFloats(ArmGemmKind kind, int inDim) => kind switch
        {
            ArmGemmKind.Q4K => inDim / QK_K * 9,
            ArmGemmKind.Q6K => inDim / QK_K,
            _ => inDim / QK8_0,
        };

        private static int ArmActRowBytes(ArmGemmKind kind, int inDim)
            => checked((int)((inDim + (long)ArmActSideFloats(kind, inDim) * 4 + 63) & ~63L));

        // Decoded column group: int8 [inDim / 16][4 k-quads][4 columns][4 values] | float side vectors (4 columns each):
        //   Q4K  : per sub-block j of each super-block: A = d4 * sc[j], C = dmin * mn[j]      (16 vectors a super-block)
        //   Q6K  : per super-block: d6, then per sub-block s: (float)scales[s]               (17 vectors a super-block)
        //   Q0   : per block: dw                                                              (1 vector a block)
        //   Nvfp4: per sub-block: ue4m3(scale)                                                (1 vector a sub-block)
        private static int ArmSideVectors(ArmGemmKind kind, int inDim) => kind switch
        {
            ArmGemmKind.Q4K => inDim / QK_K * 16,
            ArmGemmKind.Q6K => inDim / QK_K * 17,
            ArmGemmKind.Q0 => inDim / QK8_0,
            _ => inDim / 16,
        };

        private static int ArmScratchBytes(ArmGemmKind kind, int inDim) => inDim * 4 + ArmSideVectors(kind, inDim) * 16;

        [ThreadStatic] private static byte[] _armGemmScratch;

        private static unsafe byte* ArmGemmScratch(int bytes)
        {
            byte[] buf = _armGemmScratch;
            if (buf == null || buf.Length < bytes + 64)
            {
                buf = GC.AllocateUninitializedArray<byte>(Math.Max(bytes + 64, 64 * 1024), pinned: true);
                _armGemmScratch = buf;
            }
            byte* p = (byte*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(buf));
            return (byte*)(((nint)p + 63) & ~(nint)63);
        }

        /// <summary>Activation bytes a row block may hold: every decoded column group re-reads the block, so it stays in L2.</summary>
        private const long ArmRowBlockBytes = 1L << 20;

        /// <summary>
        /// The ARM GEMM: <c>output[r, c] = dot(weightRow(c), input[r])</c>, bit-identical to the per-row dots. Returns false
        /// (writing nothing) when the host, type or shape is not covered.
        /// </summary>
        internal static unsafe bool TryQGemmArm(
            int ggmlType, IntPtr weights, int inDim, int outDim, float* input, int inputRowStride, int rowCount,
            float* output, int outputRowStride, ParallelOptions options)
        {
            if (!ArmDotSupported || rowCount <= 0 || outDim <= 0)
                return false;
            var type = (GgmlTensorType)ggmlType;
            ArmGemmKind kind = GetArmGemmKind(type, inDim);
            if (kind == ArmGemmKind.None)
                return false;

            int actStride = ArmActRowBytes(kind, inDim);
            long actBytes = (long)rowCount * actStride + 64;
            if (actBytes > int.MaxValue)
                return false;

            int dop = ResolveDop(options);
            byte[] rented = ArrayPool<byte>.Shared.Rent((int)actBytes);
            try
            {
                fixed (byte* rentedBase = rented)
                {
                    byte* act = (byte*)(((nint)rentedBase + 63) & ~(nint)63);
                    QuantizeRowsArmGemm(kind, input, inputRowStride, rowCount, inDim, act, actStride, options, dop);

                    int rowBytes = (int)RowSize(ggmlType, inDim);
                    int groups = (outDim + 3) / 4;
                    int rowsPerBlock = rowCount;
                    long blockRows = ArmRowBlockBytes / actStride;
                    if (blockRows < rowCount)
                        rowsPerBlock = (int)Math.Max(4, blockRows & ~3L);
                    int rowBlocks = (rowCount + rowsPerBlock - 1) / rowsPerBlock;
                    long wantTasks = QGemmWantTasks((long)rowCount * outDim * inDim, (long)rowBytes * outDim, rowCount, dop);
                    long colBlocks = Math.Clamp((wantTasks + rowBlocks - 1) / rowBlocks, 1, groups);
                    int groupsPerBlock = (int)((groups + colBlocks - 1) / colBlocks);
                    int colBlockCount = (groups + groupsPerBlock - 1) / groupsPerBlock;

                    nint w = weights, a = (nint)act, o = (nint)output;
                    int outStride = outputRowStride;
                    int scratchBytes = ArmScratchBytes(kind, inDim);
                    void RunTask(int t)
                    {
                        int cb = t % colBlockCount, rb = t / colBlockCount;
                        int g0 = cb * groupsPerBlock, g1 = Math.Min(groups, g0 + groupsPerBlock);
                        int r0 = rb * rowsPerBlock, r1 = Math.Min(rowCount, r0 + rowsPerBlock);
                        byte* scratch = ArmGemmScratch(scratchBytes);
                        for (int g = g0; g < g1; g++)
                        {
                            int c0 = g * 4, cols = Math.Min(4, outDim - c0);
                            DecodeArmGroup(type, kind, (byte*)w + (long)c0 * rowBytes, rowBytes, cols, inDim, scratch);
                            for (int r = r0; r < r1; r += 4)
                                RunArmTile(kind, scratch, inDim, (byte*)a, actStride, r, Math.Min(4, r1 - r),
                                    (float*)o + c0, outStride, cols);
                        }
                    }

                    int tasks = rowBlocks * colBlockCount;
                    if (tasks > 1 && dop > 1)
                        RunParallelBlocks(tasks, options, RunTask);
                    else
                        for (int t = 0; t < tasks; t++) RunTask(t);
                }
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(rented);
            }
            return true;
        }

        // ------------------------------------------------------------------
        // Activations
        // ------------------------------------------------------------------

        private static unsafe void QuantizeRowsArmGemm(ArmGemmKind kind, float* input, int inputRowStride, int rowCount,
            int inDim, byte* act, int actStride, ParallelOptions options, int dop)
        {
            long elements = (long)rowCount * inDim;
            if (dop > 1 && rowCount >= 8 && elements >= 64 * 1024)
            {
                int rowsPerTask = (int)Math.Max(4, Math.Min(rowCount, (32 * 1024 + inDim - 1) / inDim));
                int tasks = Math.Min((rowCount + rowsPerTask - 1) / rowsPerTask, dop * TasksPerWorker);
                rowsPerTask = (rowCount + tasks - 1) / tasks;
                nint src = (nint)input, dst = (nint)act;
                RunParallelBlocks(tasks, options, t =>
                {
                    int r0 = t * rowsPerTask, r1 = Math.Min(rowCount, r0 + rowsPerTask);
                    for (int r = r0; r < r1; r++)
                        QuantizeRowArmGemm(kind, (float*)src + (long)r * inputRowStride, (byte*)dst + (long)r * actStride, inDim);
                });
                return;
            }
            for (int r = 0; r < rowCount; r++)
                QuantizeRowArmGemm(kind, input + (long)r * inputRowStride, act + (long)r * actStride, inDim);
        }

        /// <summary>One activation row into the ARM GEMM layout: the int8 values and scales the per-row quantizers
        /// (QuantizeF32ToQ8_K / QuantizeF32ToQ8_0) produce, with the side data as the kernels read it.</summary>
        private static unsafe void QuantizeRowArmGemm(ArmGemmKind kind, float* src, byte* dst, int inDim)
        {
            sbyte* qs = (sbyte*)dst;
            float* side = (float*)(dst + inDim);
            int* groupSums = stackalloc int[QK_K / 16];
            if (kind == ArmGemmKind.Q4K || kind == ArmGemmKind.Q6K)
            {
                int nsb = inDim / QK_K;
                float* q8s = side + nsb;
                for (int sb = 0; sb < nsb; sb++)
                {
                    float* x = src + sb * QK_K;
                    float scale = MaxAbs(x, QK_K) / 127.0f;
                    side[sb] = scale;
                    sbyte* q = qs + sb * QK_K;
                    if (scale == 0.0f)
                    {
                        Unsafe.InitBlockUnaligned(q, 0, QK_K);
                        if (kind == ArmGemmKind.Q4K)
                            for (int j = 0; j < 8; j++) q8s[sb * 8 + j] = 0f;
                        continue;
                    }
                    QuantizeInt8Groups16(x, 1.0f / scale, q, QK_K, groupSums);
                    if (kind == ArmGemmKind.Q4K)
                        for (int j = 0; j < 8; j++)
                            q8s[sb * 8 + j] = (short)groupSums[2 * j] + (short)groupSums[2 * j + 1];
                }
                return;
            }

            int nb = inDim / QK8_0;
            for (int b = 0; b < nb; b++)
            {
                float* x = src + b * QK8_0;
                float scale = MaxAbs(x, QK8_0) / 127.0f;
                side[b] = (float)(System.Half)scale;
                sbyte* q = qs + b * QK8_0;
                if (scale == 0.0f)
                {
                    Unsafe.InitBlockUnaligned(q, 0, QK8_0);
                    continue;
                }
                QuantizeInt8Groups16(x, 1.0f / scale, q, QK8_0, groupSums);
            }
        }

        // ------------------------------------------------------------------
        // Weight decode: four columns into the SDOT-by-element layout
        // ------------------------------------------------------------------

        /// <summary>Four columns' 16-value chunks (each vector: four k-quads as 32-bit lanes) into four k-quad vectors
        /// (each: the four columns' values of one k-quad), stored at <paramref name="dst"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void StoreTransposed(Vector128<byte> v0, Vector128<byte> v1, Vector128<byte> v2, Vector128<byte> v3, byte* dst)
        {
            Vector128<uint> t0 = AdvSimd.Arm64.TransposeEven(v0.AsUInt32(), v1.AsUInt32());
            Vector128<uint> t1 = AdvSimd.Arm64.TransposeOdd(v0.AsUInt32(), v1.AsUInt32());
            Vector128<uint> t2 = AdvSimd.Arm64.TransposeEven(v2.AsUInt32(), v3.AsUInt32());
            Vector128<uint> t3 = AdvSimd.Arm64.TransposeOdd(v2.AsUInt32(), v3.AsUInt32());
            AdvSimd.Store(dst, AdvSimd.Arm64.TransposeEven(t0.AsUInt64(), t2.AsUInt64()).AsByte());
            AdvSimd.Store(dst + 16, AdvSimd.Arm64.TransposeEven(t1.AsUInt64(), t3.AsUInt64()).AsByte());
            AdvSimd.Store(dst + 32, AdvSimd.Arm64.TransposeOdd(t0.AsUInt64(), t2.AsUInt64()).AsByte());
            AdvSimd.Store(dst + 48, AdvSimd.Arm64.TransposeOdd(t1.AsUInt64(), t3.AsUInt64()).AsByte());
        }

        /// <summary>Four float vectors (one per column, four values each) transposed: vector i of the result holds value
        /// i of each column, stored at <paramref name="dst"/> + 16 * i * <paramref name="step"/>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void StoreTransposedF(Vector128<float> v0, Vector128<float> v1, Vector128<float> v2, Vector128<float> v3,
            float* dst, int step)
        {
            Vector128<float> t0 = AdvSimd.Arm64.TransposeEven(v0, v1);
            Vector128<float> t1 = AdvSimd.Arm64.TransposeOdd(v0, v1);
            Vector128<float> t2 = AdvSimd.Arm64.TransposeEven(v2, v3);
            Vector128<float> t3 = AdvSimd.Arm64.TransposeOdd(v2, v3);
            AdvSimd.Store(dst, AdvSimd.Arm64.TransposeEven(t0.AsUInt64(), t2.AsUInt64()).AsSingle());
            AdvSimd.Store(dst + 4 * step, AdvSimd.Arm64.TransposeEven(t1.AsUInt64(), t3.AsUInt64()).AsSingle());
            AdvSimd.Store(dst + 8 * step, AdvSimd.Arm64.TransposeOdd(t0.AsUInt64(), t2.AsUInt64()).AsSingle());
            AdvSimd.Store(dst + 12 * step, AdvSimd.Arm64.TransposeOdd(t1.AsUInt64(), t3.AsUInt64()).AsSingle());
        }

        /// <summary>Decode columns [0, cols) of a group (weight rows <paramref name="rowBytes"/> apart) into the scratch;
        /// missing columns repeat column 0 (their lanes are never stored).</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void DecodeArmGroup(GgmlTensorType type, ArmGemmKind kind, byte* w0, int rowBytes, int cols, int inDim, byte* scratch)
        {
            byte* c0 = w0;
            byte* c1 = cols > 1 ? w0 + rowBytes : w0;
            byte* c2 = cols > 2 ? w0 + 2 * rowBytes : w0;
            byte* c3 = cols > 3 ? w0 + 3 * rowBytes : w0;
            float* side = (float*)(scratch + inDim * 4);
            switch (kind)
            {
                case ArmGemmKind.Q4K:
                    DecodeArmQ4K(c0, c1, c2, c3, inDim / QK_K, scratch, side);
                    break;
                case ArmGemmKind.Q6K:
                    DecodeArmQ6K(c0, c1, c2, c3, inDim / QK_K, scratch, side);
                    break;
                case ArmGemmKind.Q0:
                    if (type == GgmlTensorType.Q4_0)
                        DecodeArmQ40(c0, c1, c2, c3, inDim / QK4_0, scratch, side);
                    else
                        DecodeArmQ80(c0, c1, c2, c3, inDim / QK8_0, scratch, side);
                    break;
                case ArmGemmKind.Nvfp4:
                    DecodeArmNvfp4(c0, c1, c2, c3, inDim / QK_NVFP4, scratch, side);
                    break;
            }
        }

        private static unsafe void DecodeArmQ4K(byte* c0, byte* c1, byte* c2, byte* c3, int nsb, byte* wq, float* side)
        {
            Vector128<byte> m4 = Vector128.Create((byte)0x0F);
            for (int sb = 0; sb < nsb; sb++)
            {
                long off = (long)sb * Q4_KBlockBytes;
                byte* b0 = c0 + off, b1 = c1 + off, b2 = c2 + off, b3 = c3 + off;
                byte* dst = wq + sb * QK_K * 4;
                for (int p = 0; p < 4; p++)
                {
                    for (int half = 0; half < 2; half++)
                    {
                        int at = 16 + 32 * p + 16 * half;
                        Vector128<byte> v0 = AdvSimd.LoadVector128(b0 + at), v1 = AdvSimd.LoadVector128(b1 + at);
                        Vector128<byte> v2 = AdvSimd.LoadVector128(b2 + at), v3 = AdvSimd.LoadVector128(b3 + at);
                        // sub-block 2p (low nibbles) chunk `half`, then sub-block 2p + 1 (high nibbles)
                        StoreTransposed(AdvSimd.And(v0, m4), AdvSimd.And(v1, m4), AdvSimd.And(v2, m4), AdvSimd.And(v3, m4),
                            dst + (2 * p * 32 + 16 * half) * 4);
                        StoreTransposed(AdvSimd.ShiftRightLogical(v0, 4), AdvSimd.ShiftRightLogical(v1, 4),
                            AdvSimd.ShiftRightLogical(v2, 4), AdvSimd.ShiftRightLogical(v3, 4),
                            dst + ((2 * p + 1) * 32 + 16 * half) * 4);
                    }
                }
                // A[j] = d4 * sc[j], C[j] = dmin * mn[j], each a vector over the columns: per sub-block j the pair
                // sits at side + (sb * 8 + j) * 8.
                float* s = side + sb * 64;
                Q4KScaleVectors(b0, out Vector128<float> a00, out Vector128<float> a01, out Vector128<float> m00, out Vector128<float> m01);
                Q4KScaleVectors(b1, out Vector128<float> a10, out Vector128<float> a11, out Vector128<float> m10, out Vector128<float> m11);
                Q4KScaleVectors(b2, out Vector128<float> a20, out Vector128<float> a21, out Vector128<float> m20, out Vector128<float> m21);
                Q4KScaleVectors(b3, out Vector128<float> a30, out Vector128<float> a31, out Vector128<float> m30, out Vector128<float> m31);
                StoreTransposedF(a00, a10, a20, a30, s, 2);
                StoreTransposedF(m00, m10, m20, m30, s + 4, 2);
                StoreTransposedF(a01, a11, a21, a31, s + 32, 2);
                StoreTransposedF(m01, m11, m21, m31, s + 36, 2);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void Q4KScaleVectors(byte* block, out Vector128<float> a0, out Vector128<float> a1,
            out Vector128<float> m0, out Vector128<float> m1)
        {
            float d4 = HalfToSingle(ReadUInt16(block));
            float dmin = HalfToSingle(ReadUInt16(block + 2));
            Vector128<byte> scmn = UnpackKScalesMins(block + 4);
            WidenBytesToSingle(scmn.GetLower(), out Vector128<float> sc0, out Vector128<float> sc1);
            WidenBytesToSingle(scmn.GetUpper(), out Vector128<float> mn0, out Vector128<float> mn1);
            Vector128<float> vd4 = Vector128.Create(d4), vdmin = Vector128.Create(dmin);
            a0 = vd4 * sc0;
            a1 = vd4 * sc1;
            m0 = vdmin * mn0;
            m1 = vdmin * mn1;
        }

        private static unsafe void DecodeArmQ6K(byte* c0, byte* c1, byte* c2, byte* c3, int nsb, byte* wq, float* side)
        {
            Vector128<byte> m4 = Vector128.Create((byte)0x0F);
            Vector128<byte> m30 = Vector128.Create((byte)0x30);
            Vector128<byte> off32 = Vector128.Create((byte)32);
            for (int sb = 0; sb < nsb; sb++)
            {
                long off = (long)sb * Q6_KBlockBytes;
                byte* b0 = c0 + off, b1 = c1 + off, b2 = c2 + off, b3 = c3 + off;
                byte* dst = wq + sb * QK_K * 4;
                for (int h = 0; h < 2; h++)
                {
                    for (int i = 0; i < 4; i++)
                    {
                        // ql chunk i of this half carries sub-blocks 8h + i (low nibbles) and 8h + i + 4 (high);
                        // its qh bits: chunk i % 2, shifted by 2 * (i / 2) for the low sub-block, +4 for the high one.
                        int qlAt = 64 * h + 16 * i, qhAt = QK_K / 2 + 32 * h + 16 * (i % 2);
                        int shLo = 4 - 2 * (i / 2);   // (hi2 << 4) from qh >> (2 * (i / 2)): a left shift by 4 - 2 * (i / 2)
                        Vector128<byte> l0 = AdvSimd.LoadVector128(b0 + qlAt), l1 = AdvSimd.LoadVector128(b1 + qlAt);
                        Vector128<byte> l2 = AdvSimd.LoadVector128(b2 + qlAt), l3 = AdvSimd.LoadVector128(b3 + qlAt);
                        Vector128<byte> h0 = AdvSimd.LoadVector128(b0 + qhAt), h1 = AdvSimd.LoadVector128(b1 + qhAt);
                        Vector128<byte> h2 = AdvSimd.LoadVector128(b2 + qhAt), h3 = AdvSimd.LoadVector128(b3 + qhAt);
                        StoreTransposed(
                            Q6Lo(l0, h0, shLo, m4, m30, off32), Q6Lo(l1, h1, shLo, m4, m30, off32),
                            Q6Lo(l2, h2, shLo, m4, m30, off32), Q6Lo(l3, h3, shLo, m4, m30, off32),
                            dst + (8 * h + i) * 16 * 4);
                        StoreTransposed(
                            Q6Hi(l0, h0, i / 2, m30, off32), Q6Hi(l1, h1, i / 2, m30, off32),
                            Q6Hi(l2, h2, i / 2, m30, off32), Q6Hi(l3, h3, i / 2, m30, off32),
                            dst + (8 * h + i + 4) * 16 * 4);
                    }
                }
                float* s = side + sb * 68;
                Vector128<float> d6 = LoadHalf4(b0 + 208, b1 + 208, b2 + 208, b3 + 208);
                AdvSimd.Store(s, d6);
                Q6KScaleVectors(b0 + 192, out Vector128<float> s00, out Vector128<float> s01, out Vector128<float> s02, out Vector128<float> s03);
                Q6KScaleVectors(b1 + 192, out Vector128<float> s10, out Vector128<float> s11, out Vector128<float> s12, out Vector128<float> s13);
                Q6KScaleVectors(b2 + 192, out Vector128<float> s20, out Vector128<float> s21, out Vector128<float> s22, out Vector128<float> s23);
                Q6KScaleVectors(b3 + 192, out Vector128<float> s30, out Vector128<float> s31, out Vector128<float> s32, out Vector128<float> s33);
                StoreTransposedF(s00, s10, s20, s30, s + 4, 1);
                StoreTransposedF(s01, s11, s21, s31, s + 20, 1);
                StoreTransposedF(s02, s12, s22, s32, s + 36, 1);
                StoreTransposedF(s03, s13, s23, s33, s + 52, 1);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<byte> Q6Lo(Vector128<byte> l, Vector128<byte> h, int shift, Vector128<byte> m4, Vector128<byte> m30, Vector128<byte> off32)
        {
            Vector128<byte> hi = shift == 4 ? AdvSimd.ShiftLeftLogical(h, 4) : shift == 2 ? AdvSimd.ShiftLeftLogical(h, 2) : h;
            return AdvSimd.Subtract(AdvSimd.And(l, m4) | AdvSimd.And(hi, m30), off32);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<byte> Q6Hi(Vector128<byte> l, Vector128<byte> h, int pair, Vector128<byte> m30, Vector128<byte> off32)
        {
            // pair 0: qh bits 4-5 (h & 0x30); pair 1: bits 6-7 ((h >> 2) & 0x30)
            Vector128<byte> hi = pair == 0 ? h : AdvSimd.ShiftRightLogical(h, 2);
            return AdvSimd.Subtract(AdvSimd.ShiftRightLogical(l, 4) | AdvSimd.And(hi, m30), off32);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void Q6KScaleVectors(byte* scales, out Vector128<float> s0, out Vector128<float> s1,
            out Vector128<float> s2, out Vector128<float> s3)
        {
            Vector128<sbyte> sc = AdvSimd.LoadVector128((sbyte*)scales);
            Vector128<short> lo = AdvSimd.SignExtendWideningLower(sc.GetLower());
            Vector128<short> hi = AdvSimd.SignExtendWideningUpper(sc);
            s0 = AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningLower(lo.GetLower()));
            s1 = AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningUpper(lo));
            s2 = AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningLower(hi.GetLower()));
            s3 = AdvSimd.ConvertToSingle(AdvSimd.SignExtendWideningUpper(hi));
        }

        private static unsafe void DecodeArmQ40(byte* c0, byte* c1, byte* c2, byte* c3, int nb, byte* wq, float* side)
        {
            Vector128<byte> m4 = Vector128.Create((byte)0x0F);
            Vector128<byte> off8 = Vector128.Create((byte)8);
            for (int b = 0; b < nb; b++)
            {
                long off = (long)b * Q4_0BlockBytes;
                Vector128<byte> v0 = AdvSimd.LoadVector128(c0 + off + 2), v1 = AdvSimd.LoadVector128(c1 + off + 2);
                Vector128<byte> v2 = AdvSimd.LoadVector128(c2 + off + 2), v3 = AdvSimd.LoadVector128(c3 + off + 2);
                byte* dst = wq + b * QK4_0 * 4;
                StoreTransposed(AdvSimd.Subtract(AdvSimd.And(v0, m4), off8), AdvSimd.Subtract(AdvSimd.And(v1, m4), off8),
                    AdvSimd.Subtract(AdvSimd.And(v2, m4), off8), AdvSimd.Subtract(AdvSimd.And(v3, m4), off8), dst);
                StoreTransposed(AdvSimd.Subtract(AdvSimd.ShiftRightLogical(v0, 4), off8), AdvSimd.Subtract(AdvSimd.ShiftRightLogical(v1, 4), off8),
                    AdvSimd.Subtract(AdvSimd.ShiftRightLogical(v2, 4), off8), AdvSimd.Subtract(AdvSimd.ShiftRightLogical(v3, 4), off8), dst + 64);
                AdvSimd.Store(side + 4 * b, LoadHalf4(c0 + off, c1 + off, c2 + off, c3 + off));
            }
        }

        private static unsafe void DecodeArmQ80(byte* c0, byte* c1, byte* c2, byte* c3, int nb, byte* wq, float* side)
        {
            for (int b = 0; b < nb; b++)
            {
                long off = (long)b * Q8_0BlockBytes;
                byte* dst = wq + b * QK8_0 * 4;
                StoreTransposed(AdvSimd.LoadVector128(c0 + off + 2), AdvSimd.LoadVector128(c1 + off + 2),
                    AdvSimd.LoadVector128(c2 + off + 2), AdvSimd.LoadVector128(c3 + off + 2), dst);
                StoreTransposed(AdvSimd.LoadVector128(c0 + off + 18), AdvSimd.LoadVector128(c1 + off + 18),
                    AdvSimd.LoadVector128(c2 + off + 18), AdvSimd.LoadVector128(c3 + off + 18), dst + 64);
                AdvSimd.Store(side + 4 * b, LoadHalf4(c0 + off, c1 + off, c2 + off, c3 + off));
            }
        }

        private static unsafe void DecodeArmNvfp4(byte* c0, byte* c1, byte* c2, byte* c3, int nb, byte* wq, float* side)
        {
            Vector128<byte> table;
            fixed (sbyte* tbl = Mxfp4Values)
                table = AdvSimd.LoadVector128((byte*)tbl);
            Vector128<byte> m4 = Vector128.Create((byte)0x0F);
            fixed (float* ue = Ue4m3Table)
            {
                for (int b = 0; b < nb; b++)
                {
                    long off = (long)b * Nvfp4BlockBytes;
                    byte* x0 = c0 + off, x1 = c1 + off, x2 = c2 + off, x3 = c3 + off;
                    byte* dst = wq + b * QK_NVFP4 * 4;
                    for (int half = 0; half < 2; half++)
                    {
                        // qs bytes 16h..16h+15: sub-block 2h's values 0-7 (low nibbles of bytes 0-7) and 8-15 (high
                        // nibbles of bytes 0-7), then sub-block 2h + 1 from bytes 8-15.
                        Nvfp4Chunks(AdvSimd.LoadVector128(x0 + 4 + 16 * half), table, m4, out Vector128<byte> a0, out Vector128<byte> b0);
                        Nvfp4Chunks(AdvSimd.LoadVector128(x1 + 4 + 16 * half), table, m4, out Vector128<byte> a1, out Vector128<byte> b1);
                        Nvfp4Chunks(AdvSimd.LoadVector128(x2 + 4 + 16 * half), table, m4, out Vector128<byte> a2, out Vector128<byte> b2);
                        Nvfp4Chunks(AdvSimd.LoadVector128(x3 + 4 + 16 * half), table, m4, out Vector128<byte> a3, out Vector128<byte> b3);
                        StoreTransposed(a0, a1, a2, a3, dst + (2 * half) * 64);
                        StoreTransposed(b0, b1, b2, b3, dst + (2 * half + 1) * 64);
                    }
                    for (int s = 0; s < 4; s++)
                        AdvSimd.Store(side + (4 * b + s) * 4, Vector128.Create(ue[x0[s]], ue[x1[s]], ue[x2[s]], ue[x3[s]]));
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void Nvfp4Chunks(Vector128<byte> qs, Vector128<byte> table, Vector128<byte> m4, out Vector128<byte> first, out Vector128<byte> second)
        {
            Vector128<ulong> lo = AdvSimd.Arm64.VectorTableLookup(table, AdvSimd.And(qs, m4)).AsUInt64();
            Vector128<ulong> hi = AdvSimd.Arm64.VectorTableLookup(table, AdvSimd.ShiftRightLogical(qs, 4)).AsUInt64();
            first = AdvSimd.Arm64.ZipLow(lo, hi).AsByte();
            second = AdvSimd.Arm64.ZipHigh(lo, hi).AsByte();
        }

        // ------------------------------------------------------------------
        // Row tiles: up to four activation rows against a decoded column group
        // ------------------------------------------------------------------

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void StoreTile(Vector128<float> v, float* dst, int cols)
        {
            if (cols == 4)
            {
                AdvSimd.Store(dst, v);
                return;
            }
            dst[0] = v.ToScalar();
            if (cols > 1) dst[1] = v.GetElement(1);
            if (cols > 2) dst[2] = v.GetElement(2);
        }

        private static unsafe void RunArmTile(ArmGemmKind kind, byte* scratch, int inDim, byte* act, int actStride, int row0, int rows,
            float* output, int outStride, int cols)
        {
            // Missing rows repeat the tile's first row; their results are never stored.
            byte* x0 = act + (long)row0 * actStride;
            byte* x1 = rows > 1 ? x0 + actStride : x0;
            byte* x2 = rows > 2 ? x0 + 2L * actStride : x0;
            byte* x3 = rows > 3 ? x0 + 3L * actStride : x0;
            Vector128<float> s0, s1, s2, s3;
            switch (kind)
            {
                case ArmGemmKind.Q4K:
                    ArmTileQ4K(scratch, inDim, x0, x1, x2, x3, out s0, out s1, out s2, out s3);
                    break;
                case ArmGemmKind.Q6K:
                    ArmTileQ6K(scratch, inDim, x0, x1, x2, x3, out s0, out s1, out s2, out s3);
                    break;
                case ArmGemmKind.Q0:
                    ArmTileQ0(scratch, inDim, x0, x1, x2, x3, out s0, out s1, out s2, out s3);
                    break;
                default:
                    ArmTileNvfp4(scratch, inDim, x0, x1, x2, x3, out s0, out s1, out s2, out s3);
                    break;
            }
            float* o = output + (long)row0 * outStride;
            StoreTile(s0, o, cols);
            if (rows > 1) StoreTile(s1, o + outStride, cols);
            if (rows > 2) StoreTile(s2, o + 2L * outStride, cols);
            if (rows > 3) StoreTile(s3, o + 3L * outStride, cols);
        }

        /// <summary>16 activation values of four rows against four decoded k-quad vectors (64 bytes at
        /// <paramref name="w"/>): each row's four column lanes advance by 64 multiply-adds.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void Sdot16x4(byte* w, sbyte* x0, sbyte* x1, sbyte* x2, sbyte* x3,
            ref Vector128<int> a0, ref Vector128<int> a1, ref Vector128<int> a2, ref Vector128<int> a3)
        {
            Vector128<sbyte> r0 = AdvSimd.LoadVector128(x0), r1 = AdvSimd.LoadVector128(x1);
            Vector128<sbyte> r2 = AdvSimd.LoadVector128(x2), r3 = AdvSimd.LoadVector128(x3);
            Vector128<sbyte> q = AdvSimd.LoadVector128((sbyte*)w);
            a0 = Dp.DotProductBySelectedQuadruplet(a0, q, r0, 0);
            a1 = Dp.DotProductBySelectedQuadruplet(a1, q, r1, 0);
            a2 = Dp.DotProductBySelectedQuadruplet(a2, q, r2, 0);
            a3 = Dp.DotProductBySelectedQuadruplet(a3, q, r3, 0);
            q = AdvSimd.LoadVector128((sbyte*)w + 16);
            a0 = Dp.DotProductBySelectedQuadruplet(a0, q, r0, 1);
            a1 = Dp.DotProductBySelectedQuadruplet(a1, q, r1, 1);
            a2 = Dp.DotProductBySelectedQuadruplet(a2, q, r2, 1);
            a3 = Dp.DotProductBySelectedQuadruplet(a3, q, r3, 1);
            q = AdvSimd.LoadVector128((sbyte*)w + 32);
            a0 = Dp.DotProductBySelectedQuadruplet(a0, q, r0, 2);
            a1 = Dp.DotProductBySelectedQuadruplet(a1, q, r1, 2);
            a2 = Dp.DotProductBySelectedQuadruplet(a2, q, r2, 2);
            a3 = Dp.DotProductBySelectedQuadruplet(a3, q, r3, 2);
            q = AdvSimd.LoadVector128((sbyte*)w + 48);
            a0 = Dp.DotProductBySelectedQuadruplet(a0, q, r0, 3);
            a1 = Dp.DotProductBySelectedQuadruplet(a1, q, r1, 3);
            a2 = Dp.DotProductBySelectedQuadruplet(a2, q, r2, 3);
            a3 = Dp.DotProductBySelectedQuadruplet(a3, q, r3, 3);
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void ArmTileQ4K(byte* scratch, int inDim, byte* x0, byte* x1, byte* x2, byte* x3,
            out Vector128<float> s0, out Vector128<float> s1, out Vector128<float> s2, out Vector128<float> s3)
        {
            int nsb = inDim / QK_K;
            float* side = (float*)(scratch + inDim * 4);
            float* f0 = (float*)(x0 + inDim), f1 = (float*)(x1 + inDim), f2 = (float*)(x2 + inDim), f3 = (float*)(x3 + inDim);
            Vector128<float> t0 = default, t1 = default, t2 = default, t3 = default;   // locals: out parameters would live in memory
            for (int sb = 0; sb < nsb; sb++)
            {
                Vector128<float> d0 = Vector128.Create(f0[sb]), d1 = Vector128.Create(f1[sb]);
                Vector128<float> d2 = Vector128.Create(f2[sb]), d3 = Vector128.Create(f3[sb]);
                float* q0 = f0 + nsb + sb * 8, q1 = f1 + nsb + sb * 8, q2 = f2 + nsb + sb * 8, q3 = f3 + nsb + sb * 8;
                for (int j = 0; j < 8; j++)
                {
                    int k = sb * QK_K + j * 32;
                    byte* w = scratch + k * 4;
                    Vector128<int> a0 = Vector128<int>.Zero, a1 = Vector128<int>.Zero, a2 = Vector128<int>.Zero, a3 = Vector128<int>.Zero;
                    Sdot16x4(w, (sbyte*)x0 + k, (sbyte*)x1 + k, (sbyte*)x2 + k, (sbyte*)x3 + k, ref a0, ref a1, ref a2, ref a3);
                    Sdot16x4(w + 64, (sbyte*)x0 + k + 16, (sbyte*)x1 + k + 16, (sbyte*)x2 + k + 16, (sbyte*)x3 + k + 16, ref a0, ref a1, ref a2, ref a3);
                    Vector128<float> A = AdvSimd.LoadVector128(side + (sb * 8 + j) * 8);
                    Vector128<float> C = AdvSimd.LoadVector128(side + (sb * 8 + j) * 8 + 4);
                    t0 += d0 * (A * AdvSimd.ConvertToSingle(a0) - C * Vector128.Create(q0[j]));
                    t1 += d1 * (A * AdvSimd.ConvertToSingle(a1) - C * Vector128.Create(q1[j]));
                    t2 += d2 * (A * AdvSimd.ConvertToSingle(a2) - C * Vector128.Create(q2[j]));
                    t3 += d3 * (A * AdvSimd.ConvertToSingle(a3) - C * Vector128.Create(q3[j]));
                }
            }
            s0 = t0; s1 = t1; s2 = t2; s3 = t3;
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void ArmTileQ6K(byte* scratch, int inDim, byte* x0, byte* x1, byte* x2, byte* x3,
            out Vector128<float> s0, out Vector128<float> s1, out Vector128<float> s2, out Vector128<float> s3)
        {
            int nsb = inDim / QK_K;
            float* side = (float*)(scratch + inDim * 4);
            float* f0 = (float*)(x0 + inDim), f1 = (float*)(x1 + inDim), f2 = (float*)(x2 + inDim), f3 = (float*)(x3 + inDim);
            Vector128<float> t0 = default, t1 = default, t2 = default, t3 = default;   // locals: out parameters would live in memory
            for (int sb = 0; sb < nsb; sb++)
            {
                float* sv = side + sb * 68;
                Vector128<float> d6 = AdvSimd.LoadVector128(sv);
                // scaleBase = d6 * d8, per column and row, as the scalar loop forms it once a super-block
                Vector128<float> b0 = d6 * Vector128.Create(f0[sb]), b1 = d6 * Vector128.Create(f1[sb]);
                Vector128<float> b2 = d6 * Vector128.Create(f2[sb]), b3 = d6 * Vector128.Create(f3[sb]);
                for (int s = 0; s < 16; s++)
                {
                    int k = sb * QK_K + s * 16;
                    Vector128<int> a0 = Vector128<int>.Zero, a1 = Vector128<int>.Zero, a2 = Vector128<int>.Zero, a3 = Vector128<int>.Zero;
                    Sdot16x4(scratch + k * 4, (sbyte*)x0 + k, (sbyte*)x1 + k, (sbyte*)x2 + k, (sbyte*)x3 + k, ref a0, ref a1, ref a2, ref a3);
                    Vector128<float> sc = AdvSimd.LoadVector128(sv + 4 + 4 * s);
                    t0 += (b0 * sc) * AdvSimd.ConvertToSingle(a0);
                    t1 += (b1 * sc) * AdvSimd.ConvertToSingle(a1);
                    t2 += (b2 * sc) * AdvSimd.ConvertToSingle(a2);
                    t3 += (b3 * sc) * AdvSimd.ConvertToSingle(a3);
                }
            }
            s0 = t0; s1 = t1; s2 = t2; s3 = t3;
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void ArmTileQ0(byte* scratch, int inDim, byte* x0, byte* x1, byte* x2, byte* x3,
            out Vector128<float> s0, out Vector128<float> s1, out Vector128<float> s2, out Vector128<float> s3)
        {
            int nb = inDim / QK8_0;
            float* side = (float*)(scratch + inDim * 4);
            float* f0 = (float*)(x0 + inDim), f1 = (float*)(x1 + inDim), f2 = (float*)(x2 + inDim), f3 = (float*)(x3 + inDim);
            Vector128<float> t0 = default, t1 = default, t2 = default, t3 = default;   // locals: out parameters would live in memory
            for (int b = 0; b < nb; b++)
            {
                int k = b * QK8_0;
                byte* w = scratch + k * 4;
                Vector128<int> a0 = Vector128<int>.Zero, a1 = Vector128<int>.Zero, a2 = Vector128<int>.Zero, a3 = Vector128<int>.Zero;
                Sdot16x4(w, (sbyte*)x0 + k, (sbyte*)x1 + k, (sbyte*)x2 + k, (sbyte*)x3 + k, ref a0, ref a1, ref a2, ref a3);
                Sdot16x4(w + 64, (sbyte*)x0 + k + 16, (sbyte*)x1 + k + 16, (sbyte*)x2 + k + 16, (sbyte*)x3 + k + 16, ref a0, ref a1, ref a2, ref a3);
                Vector128<float> dw = AdvSimd.LoadVector128(side + 4 * b);
                t0 += (dw * Vector128.Create(f0[b])) * AdvSimd.ConvertToSingle(a0);
                t1 += (dw * Vector128.Create(f1[b])) * AdvSimd.ConvertToSingle(a1);
                t2 += (dw * Vector128.Create(f2[b])) * AdvSimd.ConvertToSingle(a2);
                t3 += (dw * Vector128.Create(f3[b])) * AdvSimd.ConvertToSingle(a3);
            }
            s0 = t0; s1 = t1; s2 = t2; s3 = t3;
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void ArmTileNvfp4(byte* scratch, int inDim, byte* x0, byte* x1, byte* x2, byte* x3,
            out Vector128<float> s0, out Vector128<float> s1, out Vector128<float> s2, out Vector128<float> s3)
        {
            int nsub = inDim / 16;
            float* side = (float*)(scratch + inDim * 4);
            float* f0 = (float*)(x0 + inDim), f1 = (float*)(x1 + inDim), f2 = (float*)(x2 + inDim), f3 = (float*)(x3 + inDim);
            Vector128<float> t0 = default, t1 = default, t2 = default, t3 = default;   // locals: out parameters would live in memory
            for (int s = 0; s < nsub; s++)
            {
                int k = s * 16;
                Vector128<int> a0 = Vector128<int>.Zero, a1 = Vector128<int>.Zero, a2 = Vector128<int>.Zero, a3 = Vector128<int>.Zero;
                Sdot16x4(scratch + k * 4, (sbyte*)x0 + k, (sbyte*)x1 + k, (sbyte*)x2 + k, (sbyte*)x3 + k, ref a0, ref a1, ref a2, ref a3);
                Vector128<float> ue = AdvSimd.LoadVector128(side + 4 * s);
                int b = s >> 1;
                t0 += (Vector128.Create(f0[b]) * ue) * AdvSimd.ConvertToSingle(a0);
                t1 += (Vector128.Create(f1[b]) * ue) * AdvSimd.ConvertToSingle(a1);
                t2 += (Vector128.Create(f2[b]) * ue) * AdvSimd.ConvertToSingle(a2);
                t3 += (Vector128.Create(f3[b]) * ue) * AdvSimd.ConvertToSingle(a3);
            }
            s0 = t0; s1 = t1; s2 = t2; s3 = t3;
        }
    }
}
