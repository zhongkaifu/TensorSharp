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
using System.Runtime.Intrinsics;
using TensorSharp.Cpu;

namespace TensorSharp.Models
{
    // ------------------------------------------------------------------
    // The F16-cache decode attention spread over every worker, bit for bit.
    //
    // AttentionDecodeGroupedF16 runs one block per KV head: four blocks on a model
    // with four KV heads (LFM3, many GQA models), however many cores there are, and
    // each block walks every cached key - so at 16K keys a decode step's attention
    // took a few milliseconds a layer on four cores while the rest idled.
    //
    // This runs the same computation in four phases whose work splits further
    // WITHOUT changing a single operation of any output:
    //   1. scores, per (KV head, key chunk of 64-512): the same Dot4F32F16 / DotF32F16 per key,
    //      times the scale; each chunk's running max (the max is exact, and chunks
    //      combined in order keep the first of equal values, as the one-pass loop does);
    //   2. exp(score - max), per (head, key chunk): VecExpRow's per-lane exp on the
    //      same 4-value groups (chunks start on multiples of four; the last partial
    //      group padded with -inf as VecExpRow pads it);
    //   3. per (KV head, head-dim slice): each head's exp sum accumulated in VecExpRow's
    //      lane order (one pass, the padded group's zeros included), and the weighted V
    //      sum r[d] = r[d] + v[t][d] * (e[t] * invSum) over the keys in order - per
    //      element the operations of VecScale and ScaleAdd4F16 / ScaleAddF16.
    // Head-dim slices are independent elements, so the V sum splits across workers
    // too. It is used where VecExpRow runs its 128-bit path (ARM64 and hosts without
    // 256-bit vectors); elsewhere, and below SplitDecodeKeys keys, the one-pass kernel
    // runs. ManagedQuantArmTests checks the bits against it.
    // ------------------------------------------------------------------
    public abstract partial class ModelBase
    {
        /// <summary>Keys from which the split kernel replaces the per-KV-head one (a speed choice: the bits agree).</summary>
        internal const int SplitDecodeKeys = 128;

        /// <summary>Key chunks: about two per worker over the KV heads, 64 to 512 keys (multiples of 64, so every chunk
        /// starts on a 4-value group).</summary>
        private static int SplitKeyChunk(int keys, int kvHeads)
        {
            long perWorker = ((long)keys * kvHeads + 2L * CpuParallel.DegreeOfParallelism - 1) / (2L * CpuParallel.DegreeOfParallelism);
            return (int)Math.Clamp((perWorker + 63) / 64 * 64, 64, 512);
        }

        private const int SplitHeadDimSlice = 16;

        /// <summary>The split kernel's results equal the one-pass kernel's on this host (VecExpRow's 128-bit path).</summary>
        internal static bool SplitDecodeAttentionSupported =>
            !CpuKernels.Use512 && !Vector256.IsHardwareAccelerated && Vector128.IsHardwareAccelerated;

        /// <summary><see cref="AttentionDecodeGroupedF16"/>'s result with the work split over keys and head-dim slices (see
        /// above). Callers check <see cref="SplitDecodeAttentionSupported"/>.</summary>
        internal static unsafe void AttentionDecodeGroupedF16Split(float* qPtr, ushort* kPtr, ushort* vPtr, float* rPtr,
            int numHeads, int numKVHeads, int headDim, int maxSeqLen, int totalSeqLen, float scale)
        {
            int groupSize = numHeads / numKVHeads;
            int n = totalSeqLen;
            int chunk = SplitKeyChunk(n, numKVHeads);
            int chunks = (n + chunk - 1) / chunk;
            float[] scoresArr = ArrayPool<float>.Shared.Rent(checked(numHeads * n));
            float[] maxArr = ArrayPool<float>.Shared.Rent(numHeads * chunks);
            try
            {
                fixed (float* scoresP = scoresArr)
                fixed (float* maxP = maxArr)
                {
                    long q = (long)qPtr, k = (long)kPtr, v = (long)vPtr, r = (long)rPtr, s = (long)scoresP, m = (long)maxP;

                    // 1. scores and per-chunk maxima
                    CpuParallel.For(numKVHeads * chunks, task =>
                    {
                        int kvHead = task / chunks, c = task % chunks;
                        int t0 = c * chunk, t1 = Math.Min(n, t0 + chunk);
                        int hStart = kvHead * groupSize;
                        ushort* kHead = (ushort*)k + (long)kvHead * maxSeqLen * headDim;
                        float* scores = (float*)s;
                        float* chunkMax = (float*)m;
                        if (groupSize == 4)
                        {
                            float* q0 = (float*)q + (long)hStart * headDim, q1 = q0 + headDim, q2 = q1 + headDim, q3 = q2 + headDim;
                            float* row0 = scores + (long)hStart * n, row1 = row0 + n, row2 = row1 + n, row3 = row2 + n;
                            float m0 = float.NegativeInfinity, m1 = float.NegativeInfinity, m2 = float.NegativeInfinity, m3 = float.NegativeInfinity;
                            for (int t = t0; t < t1; t++)
                            {
                                TensorComputePrimitives.Dot4F32F16(q0, q1, q2, q3, kHead + (long)t * headDim, headDim,
                                    out float s0, out float s1, out float s2, out float s3);
                                s0 *= scale; s1 *= scale; s2 *= scale; s3 *= scale;
                                row0[t] = s0; row1[t] = s1; row2[t] = s2; row3[t] = s3;
                                if (s0 > m0) m0 = s0;
                                if (s1 > m1) m1 = s1;
                                if (s2 > m2) m2 = s2;
                                if (s3 > m3) m3 = s3;
                            }
                            chunkMax[(hStart + 0) * chunks + c] = m0;
                            chunkMax[(hStart + 1) * chunks + c] = m1;
                            chunkMax[(hStart + 2) * chunks + c] = m2;
                            chunkMax[(hStart + 3) * chunks + c] = m3;
                        }
                        else
                        {
                            for (int g = 0; g < groupSize; g++)
                            {
                                float* qH = (float*)q + (long)(hStart + g) * headDim;
                                float* row = scores + (long)(hStart + g) * n;
                                float mx = float.NegativeInfinity;
                                for (int t = t0; t < t1; t++)
                                {
                                    float sc = TensorComputePrimitives.DotF32F16(qH, kHead + (long)t * headDim, headDim) * scale;
                                    row[t] = sc;
                                    if (sc > mx) mx = sc;
                                }
                                chunkMax[(hStart + g) * chunks + c] = mx;
                            }
                        }
                    });

                    // 2. exp(score - max) in place, per (head, chunk)
                    CpuParallel.For(numHeads * chunks, task =>
                    {
                        int h = task / chunks, c = task % chunks;
                        float* chunkMax = (float*)m + h * chunks;
                        float max = float.NegativeInfinity;
                        for (int i = 0; i < chunks; i++)
                            if (chunkMax[i] > max) max = chunkMax[i];
                        int t0 = c * chunk, t1 = Math.Min(n, t0 + chunk);
                        ExpGroups128((float*)s + (long)h * n, max, t0, t1);
                    });

                    // 3. per (KV head, head-dim slice): exp sums, then the weighted V sum
                    int slices = (headDim + SplitHeadDimSlice - 1) / SplitHeadDimSlice;
                    CpuParallel.For(numKVHeads * slices, task =>
                    {
                        int kvHead = task / slices, sl = task % slices;
                        int d0 = sl * SplitHeadDimSlice, d1 = Math.Min(headDim, d0 + SplitHeadDimSlice);
                        int hStart = kvHead * groupSize;
                        ushort* vHead = (ushort*)v + (long)kvHead * maxSeqLen * headDim;
                        Span<float> inv = stackalloc float[groupSize];
                        for (int g = 0; g < groupSize; g++)
                            inv[g] = 1.0f / SumGroups128((float*)s + (long)(hStart + g) * n, n);
                        for (int g = 0; g < groupSize; g++)
                        {
                            float* rH = (float*)r + (long)(hStart + g) * headDim;
                            float* row = (float*)s + (long)(hStart + g) * n;
                            WeightedVSlice(rH, row, inv[g], vHead, headDim, n, d0, d1);
                        }
                    });
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(scoresArr);
                ArrayPool<float>.Shared.Return(maxArr);
            }
        }

        /// <summary>VecExpRow's 128-bit exp on the four-value groups of [t0, t1) (t0 a multiple of four), written in place;
        /// a last partial group is padded with -inf as VecExpRow pads it.</summary>
        private static unsafe void ExpGroups128(float* row, float max, int t0, int t1)
        {
            Vector128<float> vm = Vector128.Create(max), floor = Vector128.Create(DecodeSoftmaxFlushBelow);
            int i = t0;
            for (; i + 4 <= t1; i += 4)
                Vector128.Store(FlushedExp(Vector128.Load(row + i) - vm, floor), row + i);
            if (i < t1)
            {
                float* t = stackalloc float[4];
                int rem = t1 - i;
                for (int j = 0; j < 4; j++) t[j] = j < rem ? row[i + j] : float.NegativeInfinity;
                Vector128.Store(FlushedExp(Vector128.Load(t) - vm, floor), t);
                for (int j = 0; j < rem; j++) row[i + j] = t[j];
            }
        }

        /// <summary>VecExpRow's sum of an exp'ed row: four lanes accumulated over the groups in order (the padded last
        /// group's lanes are exact zeros), then Vector128.Sum.</summary>
        private static unsafe float SumGroups128(float* row, int n)
        {
            Vector128<float> acc = Vector128<float>.Zero;
            int i = 0;
            for (; i + 4 <= n; i += 4)
                acc += Vector128.Load(row + i);
            if (i < n)
            {
                float* t = stackalloc float[4];
                int rem = n - i;
                for (int j = 0; j < 4; j++) t[j] = j < rem ? row[i + j] : 0f;
                acc += Vector128.Load(t);
            }
            return Vector128.Sum(acc);
        }

        /// <summary>r[d] for d in [d0, d1): zero, then r[d] + v[t][d] * (e[t] * inv) over the keys in order - per element
        /// what VecZero, VecScale and ScaleAdd4F16 / ScaleAddF16 compute.</summary>
        private static unsafe void WeightedVSlice(float* r, float* e, float inv, ushort* vHead, int headDim, int n, int d0, int d1)
        {
            int width = d1 - d0;
            if (width == 16)
            {
                Vector128<float> a0 = Vector128<float>.Zero, a1 = a0, a2 = a0, a3 = a0;
                ushort* vt = vHead + d0;
                for (int t = 0; t < n; t++, vt += headDim)
                {
                    Vector128<float> w = Vector128.Create(e[t] * inv);
                    a0 += ManagedQuantizedOps.HalfBitsToSingle(Vector128.WidenLower(Vector128.Load(vt))) * w;
                    a1 += ManagedQuantizedOps.HalfBitsToSingle(Vector128.WidenUpper(Vector128.Load(vt))) * w;
                    a2 += ManagedQuantizedOps.HalfBitsToSingle(Vector128.WidenLower(Vector128.Load(vt + 8))) * w;
                    a3 += ManagedQuantizedOps.HalfBitsToSingle(Vector128.WidenUpper(Vector128.Load(vt + 8))) * w;
                }
                Vector128.Store(a0, r + d0);
                Vector128.Store(a1, r + d0 + 4);
                Vector128.Store(a2, r + d0 + 8);
                Vector128.Store(a3, r + d0 + 12);
                return;
            }
            for (int d = d0; d < d1; d++) r[d] = 0f;
            for (int t = 0; t < n; t++)
            {
                float w = e[t] * inv;
                ushort* vt = vHead + (long)t * headDim;
                for (int d = d0; d < d1; d++)
                    r[d] += (float)BitConverter.UInt16BitsToHalf(vt[d]) * w;
            }
        }
    }
}
