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
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Models.Architecture;
using TensorSharp.Cuda;
using TensorSharp.GGML;
using TensorSharp.MLX;

namespace TensorSharp.Models
{

    // The pure-C# decode attention kernels and the SIMD primitives they run on.
    // This is the host fallback: every accelerated backend brings its own fused
    // attention, and this path serves CPU runs and the cases a backend declines.
    public abstract partial class ModelBase
    {
        #region SIMD Helpers

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector<float> LdVec(float* p) =>
            TensorComputePrimitives.LoadVector(p);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void StVec(float* p, Vector<float> v) =>
            TensorComputePrimitives.StoreVector(p, v);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected static unsafe float VecDot(float* a, float* b, int n) =>
            TensorComputePrimitives.Dot(a, b, n);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected static unsafe float VecSumSq(float* a, int n) =>
            TensorComputePrimitives.SumSquares(a, n);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected static unsafe void VecScale(float* data, float scale, int n) =>
            TensorComputePrimitives.Scale(data, scale, n);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected static unsafe void VecScaleAdd(float* dst, float* src, float w, int n) =>
            TensorComputePrimitives.ScaleAdd(dst, src, w, n);

        /// <summary>
        /// Batched dot product: simultaneously compute four independent dot products
        /// against the same source vector <paramref name="b"/>. Lets the compiler keep
        /// the vector loads of b in registers and reuse them across the four accumulators,
        /// effectively cutting the load bandwidth on b by 4x compared to four sequential
        /// VecDot calls. Used in GQA decode attention where four query heads share a K/V head.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected static unsafe void VecDot4(float* a0, float* a1, float* a2, float* a3,
            float* b, int n,
            out float r0, out float r1, out float r2, out float r3) =>
            TensorComputePrimitives.Dot4(a0, a1, a2, a3, b, n, out r0, out r1, out r2, out r3);

        /// <summary>
        /// Batched scale-add: simultaneously update four destination vectors with the
        /// same source <paramref name="src"/> scaled by four independent weights. The
        /// hot loop loads each src element exactly once into a register and broadcasts
        /// it to four FMA-style updates, which is the V-aggregation analog of VecDot4.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected static unsafe void VecScaleAdd4(float* d0, float* d1, float* d2, float* d3,
            float* src, float w0, float w1, float w2, float w3, int n) =>
            TensorComputePrimitives.ScaleAdd4(d0, d1, d2, d3, src, w0, w1, w2, w3, n);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected static unsafe void VecSubScale(float* dst, float* a, float* b, float scale, int n) =>
            TensorComputePrimitives.SubScale(dst, a, b, scale, n);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected static unsafe void VecZero(float* data, int n) =>
            TensorComputePrimitives.Zero(data, n);

        /// <summary>
        /// Softmax terms whose exponent argument (score minus the row max) is below this
        /// are flushed to exactly zero. exp(-50) is 1.9e-22: next to the max term's 1, no
        /// F32 sum over fewer than 2^40 positions can register it, so the attention output
        /// does not change. Left in, the tail of a sharply peaked row lands in the SUBNORMAL
        /// range (exp of anything under -87.3), and every multiply that touches a subnormal
        /// operand - the normalisation and the w*V accumulation, per cached position per
        /// head - takes a microcode assist on x86. Measured on an i7-11800H over 16 heads x
        /// 640 positions x headDim 128 (one tensor-parallel rank's decode attention for
        /// Nemotron 3.5): 0.18 ms with a normal tail, 2.1 ms with half of each row
        /// subnormal, 3.9 ms at 90%; with the flush, 0.14 ms at every one of those. That
        /// made host decode attention (every --tp attention layer on the GGML backends,
        /// and the cpu backend) up to 20x slower for content whose attention is sharply
        /// peaked, so one build's decode speed moved with the text it happened to generate.
        /// </summary>
        internal const float DecodeSoftmaxFlushBelow = -50f;

        // exp(x) per lane, with lanes below `floor` (and the -inf tail padding) clamped
        // before the exp, so it never computes a subnormal either, and zeroed after it.
        // NaN lanes compare false and still propagate.
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector512<float> FlushedExp(Vector512<float> x, Vector512<float> floor)
        {
            Vector512<float> below = Vector512.LessThan(x, floor);
            return Vector512.AndNot(Vector512.Exp(Vector512.ConditionalSelect(below, floor, x)), below);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector256<float> FlushedExp(Vector256<float> x, Vector256<float> floor)
        {
            Vector256<float> below = Vector256.LessThan(x, floor);
            return Vector256.AndNot(Vector256.Exp(Vector256.ConditionalSelect(below, floor, x)), below);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<float> FlushedExp(Vector128<float> x, Vector128<float> floor)
        {
            Vector128<float> below = Vector128.LessThan(x, floor);
            return Vector128.AndNot(Vector128.Exp(Vector128.ConditionalSelect(below, floor, x)), below);
        }

        /// <summary>
        /// Vectorized exp(row[i] - max): writes e back into <paramref name="row"/> and
        /// returns the sum. Same shape as the exp phase of
        /// <see cref="CpuKernels.SoftmaxRow"/> (Vector512/256/128.Exp with
        /// -inf-padded tails, scalar tail), but leaves the row un-normalized so the
        /// k-split path can keep chunk-partial sums and the grouped paths keep their
        /// running-max reuse plus <see cref="TensorComputePrimitives.Scale"/> step.
        /// Up to a few ULP from scalar MathF.Exp (BCL vector-exp approximation).
        /// Terms below <see cref="DecodeSoftmaxFlushBelow"/> come out exactly zero.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        internal static unsafe float VecExpRow(float* row, float max, int n)
        {
            float sum = 0f;
            int i = 0;
            if (CpuKernels.Use512)
            {
                Vector512<float> vm = Vector512.Create(max), floor = Vector512.Create(DecodeSoftmaxFlushBelow);
                Vector512<float> acc = Vector512<float>.Zero;
                for (; i + 16 <= n; i += 16)
                {
                    Vector512<float> ex = FlushedExp(Vector512.Load(row + i) - vm, floor);
                    Vector512.Store(ex, row + i);
                    acc += ex;
                }
                if (i < n)
                {
                    float* t = stackalloc float[16];
                    int rem = n - i;
                    for (int j = 0; j < 16; j++) t[j] = j < rem ? row[i + j] : float.NegativeInfinity;
                    Vector512<float> ex = FlushedExp(Vector512.Load(t) - vm, floor);
                    Vector512.Store(ex, t);
                    acc += ex;
                    for (int j = 0; j < rem; j++) row[i + j] = t[j];
                    i = n;
                }
                sum = Vector512.Sum(acc);
            }
            else if (Vector256.IsHardwareAccelerated)
            {
                Vector256<float> vm = Vector256.Create(max), floor = Vector256.Create(DecodeSoftmaxFlushBelow);
                Vector256<float> acc = Vector256<float>.Zero;
                for (; i + 8 <= n; i += 8)
                {
                    Vector256<float> ex = FlushedExp(Vector256.Load(row + i) - vm, floor);
                    Vector256.Store(ex, row + i);
                    acc += ex;
                }
                if (i < n)
                {
                    float* t = stackalloc float[8];
                    int rem = n - i;
                    for (int j = 0; j < 8; j++) t[j] = j < rem ? row[i + j] : float.NegativeInfinity;
                    Vector256<float> ex = FlushedExp(Vector256.Load(t) - vm, floor);
                    Vector256.Store(ex, t);
                    acc += ex;
                    for (int j = 0; j < rem; j++) row[i + j] = t[j];
                    i = n;
                }
                sum = Vector256.Sum(acc);
            }
            else if (Vector128.IsHardwareAccelerated)
            {
                Vector128<float> vm = Vector128.Create(max), floor = Vector128.Create(DecodeSoftmaxFlushBelow);
                Vector128<float> acc = Vector128<float>.Zero;
                for (; i + 4 <= n; i += 4)
                {
                    Vector128<float> ex = FlushedExp(Vector128.Load(row + i) - vm, floor);
                    Vector128.Store(ex, row + i);
                    acc += ex;
                }
                if (i < n)
                {
                    float* t = stackalloc float[4];
                    int rem = n - i;
                    for (int j = 0; j < 4; j++) t[j] = j < rem ? row[i + j] : float.NegativeInfinity;
                    Vector128<float> ex = FlushedExp(Vector128.Load(t) - vm, floor);
                    Vector128.Store(ex, t);
                    acc += ex;
                    for (int j = 0; j < rem; j++) row[i + j] = t[j];
                    i = n;
                }
                sum = Vector128.Sum(acc);
            }
            for (; i < n; i++)
            {
                float d = row[i] - max;
                float e = d < DecodeSoftmaxFlushBelow ? 0f : MathF.Exp(d);
                row[i] = e;
                sum += e;
            }
            return sum;
        }

        #endregion


        protected unsafe void AttentionDecodePureCS(Tensor q, Tensor kCache, Tensor vCache,
            Tensor result, int numHeads, int numKVHeads, int headDim, int totalSeqLen, float scale)
        {
            if (kCache.ElementType == DType.Float16 && vCache.ElementType == DType.Float16)
            {
                AttentionDecodePureCSF16(q, kCache, vCache, result,
                    numHeads, numKVHeads, headDim, totalSeqLen, scale);
                return;
            }

            if (IsBlockQuantCacheDType(kCache.ElementType) && vCache.ElementType == kCache.ElementType)
            {
                // Block-quantized (Q4_0 / Q8_0) caches cannot be walked as flat float
                // buffers. Dequantize the active [0, totalSeqLen) window into compact
                // F32 tensors (no GQA broadcast; the grouped kernel below reads per
                // KV head) and re-enter on the F32 path — the compact copy's
                // Sizes[1] == totalSeqLen doubles as its row stride. This is the
                // deep-fallback path (fused native attention handles quantized
                // caches on-device), so correctness beats the extra dequant cost.
                using (Tensor kF32 = ExpandKVHeadsBlockQuant(kCache, 1, totalSeqLen))
                using (Tensor vF32 = ExpandKVHeadsBlockQuant(vCache, 1, totalSeqLen))
                {
                    AttentionDecodePureCS(q, kF32, vF32, result,
                        numHeads, numKVHeads, headDim, totalSeqLen, scale);
                }
                return;
            }

            AttentionDecodeGrouped(GetFloatPtr(q), GetFloatPtrForRead(kCache), GetFloatPtrForRead(vCache), GetFloatPtr(result),
                numHeads, numKVHeads, headDim, (int)kCache.Sizes[1], totalSeqLen, scale,
                Environment.ProcessorCount);
        }

        /// <summary>
        /// The F32-cache kernel behind <see cref="AttentionDecodePureCS"/>, on raw pointers:
        /// <paramref name="qPtr"/> and <paramref name="rPtr"/> are (numHeads, headDim), the caches
        /// (numKVHeads, maxSeqLen, headDim) with keys [0, totalSeqLen) live.
        /// <paramref name="processorCount"/> sizes the K-split; the model entry point passes
        /// <see cref="Environment.ProcessorCount"/>, tests pin it to reach each schedule.
        /// The (groupSize, keys) score scratch is stack memory only up to
        /// <see cref="CpuScratch.StackFloatLimit"/> floats and rented above it: these blocks run on
        /// <see cref="CpuWorkerPool"/> workers, and a group of 16 over 13,844 keys (865 KiB)
        /// overflowed a worker's 512 KiB stack, which aborts the process.
        /// </summary>
        internal static unsafe void AttentionDecodeGrouped(float* qPtr, float* kPtr, float* vPtr, float* rPtr,
            int numHeads, int numKVHeads, int headDim, int maxSeqLen, int totalSeqLen, float scale,
            int processorCount)
        {
            int groupSize = numHeads / numKVHeads;
            // Read on the entering thread and captured by the blocks (see CpuScratch).
            int stackLimit = CpuScratch.StackFloatLimit;

            // GQA-aware decode attention. For each KV head we compute attention for the
            // groupSize query heads that share it, reading K/V from the cache exactly once
            // per KV head per token instead of groupSize times. On models with GQA this
            // cuts the per-token K/V cache traffic by groupSize (4x for Qwen3.5), which
            // is the dominant cost for long-context decode.
            //
            // To keep multi-core utilization high we split each KV head into kSplit chunks
            // along the sequence dimension and merge partial softmax results using the
            // standard online (log-sum-exp) update. Total parallel tasks = numKVHeads * kSplit.

            // Aim for enough parallel tasks to keep cores busy, but keep per-task work
            // big enough to amortize fork/join dispatch overhead on the shared
            // CpuParallel pool. Each task handles one
            // (KV head, K-chunk) pair. Empirically, ~512 K-positions per task is the sweet
            // spot on Apple M-series: smaller chunks lose to scheduler overhead, larger
            // chunks under-utilize cores at long contexts.
            int procCount = processorCount;
            int kSplit = 1;
            if (numKVHeads < procCount && totalSeqLen >= 1024)
            {
                int target = (procCount + numKVHeads - 1) / numKVHeads;
                int maxSplit = Math.Max(1, totalSeqLen / 512);
                kSplit = Math.Min(target, maxSplit);
            }
            int totalTasks = numKVHeads * kSplit;
            bool useParallel = totalTasks > 1 && (long)numHeads * totalSeqLen >= 4096;

            if (useParallel)
            {
                long qPtrL = (long)qPtr;
                long kPtrL = (long)kPtr;
                long vPtrL = (long)vPtr;
                long rPtrL = (long)rPtr;
                int totalSeqLenLocal = totalSeqLen;
                int headDimLocal = headDim;
                int maxSeqLenLocal = maxSeqLen;
                int groupSizeLocal = groupSize;
                int numKVHeadsLocal = numKVHeads;
                int kSplitLocal = kSplit;
                float scaleLocal = scale;

                if (kSplitLocal == 1)
                {
                    CpuParallel.For(numKVHeadsLocal, kvHead =>
                    {
                        float* qP = (float*)qPtrL;
                        float* kP = (float*)kPtrL;
                        float* vP = (float*)vPtrL;
                        float* rP = (float*)rPtrL;
                        int scoreFloats = groupSizeLocal * totalSeqLenLocal;
                        float[] rented = null;
                        Span<float> scratch = scoreFloats <= stackLimit
                            ? stackalloc float[scoreFloats]
                            : (rented = CpuScratch.RentFloats(scoreFloats)).AsSpan(0, scoreFloats);
                        try
                        {
                            fixed (float* scoresBuf = scratch)
                            {
                                AttentionDecodeKVHeadGrouped(kvHead, qP, kP, vP, rP, scoresBuf,
                                    headDimLocal, maxSeqLenLocal, groupSizeLocal,
                                    totalSeqLenLocal, scaleLocal);
                            }
                        }
                        finally
                        {
                            CpuScratch.Return(rented);
                        }
                    });
                }
                else
                {
                    // Two-pass: partial chunks then merge per KV head. First we compute
                    // running max and (un-normalized) weighted sum for each chunk, then we
                    // merge the chunk results into the final per-query-head output.
                    int chunkSize = (totalSeqLenLocal + kSplitLocal - 1) / kSplitLocal;

                    // Per-chunk partial state: max, sumExp, weighted-V (groupSize * headDim) for each (kvHead, chunk).
                    int partialFloatsPerChunk = groupSizeLocal * (2 + headDimLocal);
                    int partialFloatsTotal = numKVHeadsLocal * kSplitLocal * partialFloatsPerChunk;

                    var partialBuf = ArrayPool<float>.Shared.Rent(partialFloatsTotal);
                    try
                    {
                        fixed (float* partialPtr = partialBuf)
                        {
                            long partialPtrL = (long)partialPtr;

                            CpuParallel.For(numKVHeadsLocal * kSplitLocal, taskIdx =>
                            {
                                int kvHead = taskIdx / kSplitLocal;
                                int chunkIdx = taskIdx % kSplitLocal;
                                int kStart = chunkIdx * chunkSize;
                                int kEnd = Math.Min(kStart + chunkSize, totalSeqLenLocal);
                                int kLen = kEnd - kStart;
                                if (kLen <= 0) return;

                                float* qP = (float*)qPtrL;
                                float* kP = (float*)kPtrL;
                                float* vP = (float*)vPtrL;
                                float* part = (float*)partialPtrL +
                                    (long)taskIdx * partialFloatsPerChunk;

                                // A chunk is totalSeqLen / kSplit keys and kSplit stops at
                                // ceil(cores / KV heads) - 2 on a 4-core host with 2 KV heads -
                                // so this grows with the context too.
                                int scoreFloats = groupSizeLocal * kLen;
                                float[] rented = null;
                                Span<float> scratch = scoreFloats <= stackLimit
                                    ? stackalloc float[scoreFloats]
                                    : (rented = CpuScratch.RentFloats(scoreFloats)).AsSpan(0, scoreFloats);
                                try
                                {
                                    fixed (float* scoresLocal = scratch)
                                    {
                                        AttentionDecodeChunkPartial(kvHead, kStart, kLen, qP, kP, vP,
                                            part, scoresLocal,
                                            headDimLocal, maxSeqLenLocal, groupSizeLocal, scaleLocal);
                                    }
                                }
                                finally
                                {
                                    CpuScratch.Return(rented);
                                }
                            });

                            CpuParallel.For(numKVHeadsLocal, kvHead =>
                            {
                                float* rP = (float*)rPtrL;
                                float* part = (float*)partialPtrL +
                                    (long)kvHead * kSplitLocal * partialFloatsPerChunk;

                                MergeChunkResults(kvHead, rP, part,
                                    headDimLocal, groupSizeLocal, kSplitLocal);
                            });
                        }
                    }
                    finally
                    {
                        ArrayPool<float>.Shared.Return(partialBuf);
                    }
                }
            }
            else
            {
                // On the caller's own thread, whatever its stack: same limit.
                int scoreFloats = groupSize * totalSeqLen;
                float[] rented = null;
                Span<float> scratch = scoreFloats <= stackLimit
                    ? stackalloc float[scoreFloats]
                    : (rented = CpuScratch.RentFloats(scoreFloats)).AsSpan(0, scoreFloats);
                try
                {
                    fixed (float* scores = scratch)
                    {
                        for (int kvHead = 0; kvHead < numKVHeads; kvHead++)
                        {
                            AttentionDecodeKVHeadGrouped(kvHead, qPtr, kPtr, vPtr, rPtr, scores,
                                headDim, maxSeqLen, groupSize, totalSeqLen, scale);
                        }
                    }
                }
                finally
                {
                    CpuScratch.Return(rented);
                }
            }
        }

        /// <summary>
        /// Compute attention for one KV head against all <paramref name="groupSize"/> query heads
        /// sharing it. Reads K and V from the cache exactly once per timestep, regardless of
        /// groupSize. On Qwen3.5-style GQA models this cuts KV-cache memory bandwidth by 4x.
        /// </summary>
        private static unsafe void AttentionDecodeKVHeadGrouped(int kvHead,
            float* qPtr, float* kPtr, float* vPtr, float* rPtr, float* scores,
            int headDim, int maxSeqLen, int groupSize, int totalSeqLen, float scale)
        {
            int hStart = kvHead * groupSize;
            float* kHead = kPtr + (long)kvHead * maxSeqLen * headDim;
            float* vHead = vPtr + (long)kvHead * maxSeqLen * headDim;

            // Per-group running max for online numerical stability. We compute scores
            // per (group, t) into a [groupSize, totalSeqLen] row-major matrix so the
            // later softmax/normalize steps stay vectorizable.
            float maxG0 = float.NegativeInfinity;
            float maxG1 = float.NegativeInfinity;
            float maxG2 = float.NegativeInfinity;
            float maxG3 = float.NegativeInfinity;

            // Score generation: K[t] is read once and dot-producted against groupSize Q heads.
            // Specialize the common groupSize=4 case to keep inner-loop arithmetic tight.
            if (groupSize == 4)
            {
                float* qH0 = qPtr + (long)(hStart + 0) * headDim;
                float* qH1 = qPtr + (long)(hStart + 1) * headDim;
                float* qH2 = qPtr + (long)(hStart + 2) * headDim;
                float* qH3 = qPtr + (long)(hStart + 3) * headDim;
                float* row0 = scores + 0L * totalSeqLen;
                float* row1 = scores + 1L * totalSeqLen;
                float* row2 = scores + 2L * totalSeqLen;
                float* row3 = scores + 3L * totalSeqLen;

                for (int t = 0; t < totalSeqLen; t++)
                {
                    float* kT = kHead + (long)t * headDim;
                    float s0, s1, s2, s3;
                    VecDot4(qH0, qH1, qH2, qH3, kT, headDim, out s0, out s1, out s2, out s3);
                    s0 *= scale; s1 *= scale; s2 *= scale; s3 *= scale;
                    row0[t] = s0; row1[t] = s1; row2[t] = s2; row3[t] = s3;
                    if (s0 > maxG0) maxG0 = s0;
                    if (s1 > maxG1) maxG1 = s1;
                    if (s2 > maxG2) maxG2 = s2;
                    if (s3 > maxG3) maxG3 = s3;
                }
            }
            else
            {
                Span<float> maxScoresSpan = stackalloc float[groupSize];
                for (int g = 0; g < groupSize; g++) maxScoresSpan[g] = float.NegativeInfinity;

                for (int t = 0; t < totalSeqLen; t++)
                {
                    float* kT = kHead + (long)t * headDim;
                    for (int g = 0; g < groupSize; g++)
                    {
                        float* qH = qPtr + (long)(hStart + g) * headDim;
                        float s = VecDot(qH, kT, headDim) * scale;
                        scores[g * totalSeqLen + t] = s;
                        if (s > maxScoresSpan[g]) maxScoresSpan[g] = s;
                    }
                }

                if (groupSize >= 1) maxG0 = maxScoresSpan[0];
                if (groupSize >= 2) maxG1 = maxScoresSpan[1];
                if (groupSize >= 3) maxG2 = maxScoresSpan[2];
                if (groupSize >= 4) maxG3 = maxScoresSpan[3];
            }

            // Softmax (per-group)
            Span<float> invSums = stackalloc float[groupSize];
            for (int g = 0; g < groupSize; g++)
            {
                float maxS;
                if (g == 0) maxS = maxG0;
                else if (g == 1) maxS = maxG1;
                else if (g == 2) maxS = maxG2;
                else if (g == 3) maxS = maxG3;
                else
                {
                    maxS = float.NegativeInfinity;
                    float* rowG0 = scores + (long)g * totalSeqLen;
                    for (int t = 0; t < totalSeqLen; t++)
                        if (rowG0[t] > maxS) maxS = rowG0[t];
                }

                float* rowG = scores + (long)g * totalSeqLen;
                invSums[g] = 1.0f / VecExpRow(rowG, maxS, totalSeqLen);
            }
            for (int g = 0; g < groupSize; g++)
            {
                float invSum = invSums[g];
                float* rowG = scores + (long)g * totalSeqLen;
                VecScale(rowG, invSum, totalSeqLen);
            }

            // Aggregate V: read V[t] once per t, scatter into all groupSize result heads.
            for (int g = 0; g < groupSize; g++)
                VecZero(rPtr + (long)(hStart + g) * headDim, headDim);

            if (groupSize == 4)
            {
                float* r0 = rPtr + (long)(hStart + 0) * headDim;
                float* r1 = rPtr + (long)(hStart + 1) * headDim;
                float* r2 = rPtr + (long)(hStart + 2) * headDim;
                float* r3 = rPtr + (long)(hStart + 3) * headDim;
                float* row0 = scores + 0L * totalSeqLen;
                float* row1 = scores + 1L * totalSeqLen;
                float* row2 = scores + 2L * totalSeqLen;
                float* row3 = scores + 3L * totalSeqLen;

                for (int t = 0; t < totalSeqLen; t++)
                {
                    float* vT = vHead + (long)t * headDim;
                    VecScaleAdd4(r0, r1, r2, r3, vT,
                        row0[t], row1[t], row2[t], row3[t], headDim);
                }
            }
            else
            {
                for (int t = 0; t < totalSeqLen; t++)
                {
                    float* vT = vHead + (long)t * headDim;
                    for (int g = 0; g < groupSize; g++)
                    {
                        float w = scores[g * totalSeqLen + t];
                        float* rH = rPtr + (long)(hStart + g) * headDim;
                        VecScaleAdd(rH, vT, w, headDim);
                    }
                }
            }
        }

        /// <summary>
        /// Compute partial attention for one (KV head, K-chunk) pair. Writes per-group
        /// running max, un-normalized exp sum, and un-normalized weighted-V into the
        /// supplied <paramref name="partial"/> buffer for later cross-chunk merging.
        ///
        /// Layout of <paramref name="partial"/> (length = groupSize * (2 + headDim)):
        ///   [g * (2 + headDim) + 0]            = max for group g
        ///   [g * (2 + headDim) + 1]            = sumExp for group g
        ///   [g * (2 + headDim) + 2 .. + headDim+1] = un-normalized weighted V for group g
        /// </summary>
        private static unsafe void AttentionDecodeChunkPartial(int kvHead,
            int kStart, int kLen,
            float* qPtr, float* kPtr, float* vPtr,
            float* partial, float* scores,
            int headDim, int maxSeqLen, int groupSize, float scale)
        {
            int hStart = kvHead * groupSize;
            float* kHead = kPtr + (long)kvHead * maxSeqLen * headDim;
            float* vHead = vPtr + (long)kvHead * maxSeqLen * headDim;
            int strideG = 2 + headDim;

            for (int g = 0; g < groupSize; g++)
                partial[g * strideG] = float.NegativeInfinity;

            float maxG0 = float.NegativeInfinity;
            float maxG1 = float.NegativeInfinity;
            float maxG2 = float.NegativeInfinity;
            float maxG3 = float.NegativeInfinity;

            if (groupSize == 4)
            {
                float* qH0 = qPtr + (long)(hStart + 0) * headDim;
                float* qH1 = qPtr + (long)(hStart + 1) * headDim;
                float* qH2 = qPtr + (long)(hStart + 2) * headDim;
                float* qH3 = qPtr + (long)(hStart + 3) * headDim;
                float* row0 = scores + 0L * kLen;
                float* row1 = scores + 1L * kLen;
                float* row2 = scores + 2L * kLen;
                float* row3 = scores + 3L * kLen;

                for (int t = 0; t < kLen; t++)
                {
                    float* kT = kHead + (long)(kStart + t) * headDim;
                    float s0, s1, s2, s3;
                    VecDot4(qH0, qH1, qH2, qH3, kT, headDim, out s0, out s1, out s2, out s3);
                    s0 *= scale; s1 *= scale; s2 *= scale; s3 *= scale;
                    row0[t] = s0; row1[t] = s1; row2[t] = s2; row3[t] = s3;
                    if (s0 > maxG0) maxG0 = s0;
                    if (s1 > maxG1) maxG1 = s1;
                    if (s2 > maxG2) maxG2 = s2;
                    if (s3 > maxG3) maxG3 = s3;
                }
            }
            else
            {
                for (int g = 0; g < groupSize; g++)
                    partial[g * strideG] = float.NegativeInfinity;

                for (int t = 0; t < kLen; t++)
                {
                    float* kT = kHead + (long)(kStart + t) * headDim;
                    for (int g = 0; g < groupSize; g++)
                    {
                        float* qH = qPtr + (long)(hStart + g) * headDim;
                        float s = VecDot(qH, kT, headDim) * scale;
                        scores[g * kLen + t] = s;
                        if (s > partial[g * strideG]) partial[g * strideG] = s;
                    }
                }
            }

            if (groupSize == 4)
            {
                partial[0 * strideG] = maxG0;
                partial[1 * strideG] = maxG1;
                partial[2 * strideG] = maxG2;
                partial[3 * strideG] = maxG3;
            }

            // Softmax per group (un-normalized) and partial weighted V
            for (int g = 0; g < groupSize; g++)
            {
                float maxS = partial[g * strideG];
                float* rowG = scores + (long)g * kLen;
                partial[g * strideG + 1] = VecExpRow(rowG, maxS, kLen);
            }

            // Compute weighted V for this chunk
            for (int g = 0; g < groupSize; g++)
                VecZero(partial + g * strideG + 2, headDim);

            if (groupSize == 4)
            {
                float* w0 = partial + 0 * strideG + 2;
                float* w1 = partial + 1 * strideG + 2;
                float* w2 = partial + 2 * strideG + 2;
                float* w3 = partial + 3 * strideG + 2;
                float* row0 = scores + 0L * kLen;
                float* row1 = scores + 1L * kLen;
                float* row2 = scores + 2L * kLen;
                float* row3 = scores + 3L * kLen;

                for (int t = 0; t < kLen; t++)
                {
                    float* vT = vHead + (long)(kStart + t) * headDim;
                    VecScaleAdd4(w0, w1, w2, w3, vT,
                        row0[t], row1[t], row2[t], row3[t], headDim);
                }
            }
            else
            {
                for (int t = 0; t < kLen; t++)
                {
                    float* vT = vHead + (long)(kStart + t) * headDim;
                    for (int g = 0; g < groupSize; g++)
                    {
                        float w = scores[g * kLen + t];
                        VecScaleAdd(partial + g * strideG + 2, vT, w, headDim);
                    }
                }
            }
        }

        /// <summary>
        /// Combine the per-chunk partial sums into the final attention output for one KV head.
        /// Uses the standard online softmax merge: M = max(M_a, M_b),
        ///   sum_new = sum_a*exp(M_a - M) + sum_b*exp(M_b - M),
        ///   acc_new = acc_a*exp(M_a - M) + acc_b*exp(M_b - M),
        /// then divide acc_new by sum_new at the end.
        /// </summary>
        private static unsafe void MergeChunkResults(int kvHead, float* rPtr, float* partial,
            int headDim, int groupSize, int kSplit)
        {
            int strideG = 2 + headDim;
            int strideChunk = groupSize * strideG;
            int hStart = kvHead * groupSize;

            for (int g = 0; g < groupSize; g++)
            {
                float globalMax = float.NegativeInfinity;
                for (int c = 0; c < kSplit; c++)
                {
                    float m = partial[c * strideChunk + g * strideG];
                    if (m > globalMax) globalMax = m;
                }

                float globalSum = 0;
                float* rOut = rPtr + (long)(hStart + g) * headDim;
                VecZero(rOut, headDim);

                for (int c = 0; c < kSplit; c++)
                {
                    float* p = partial + c * strideChunk + g * strideG;
                    float chunkMax = p[0];
                    float chunkSum = p[1];
                    if (chunkSum <= 0) continue;
                    float* chunkAcc = p + 2;

                    // A chunk this far below the global max adds nothing an F32 sum can
                    // hold, and its scale would be subnormal (see VecExpRow).
                    float logScale = chunkMax - globalMax;
                    if (logScale < DecodeSoftmaxFlushBelow) continue;
                    float scale = MathF.Exp(logScale);
                    globalSum += chunkSum * scale;
                    VecScaleAdd(rOut, chunkAcc, scale, headDim);
                }

                if (globalSum > 0)
                    VecScale(rOut, 1.0f / globalSum, headDim);
            }
        }

        /// <summary>
        /// Single-token GQA decode attention specialized for an F16 KV cache.
        /// Reads K/V values as ushort, converts to F32 inside the dot/scale-add
        /// hot loops via <see cref="TensorComputePrimitives"/>. The cache layout
        /// is identical to the F32 variant - <c>(num_kv_heads, max_seq_len, head_dim)</c> -
        /// so callers don't need to special-case anything but the storage dtype.
        ///
        /// This is the C# fallback path when an architecture's native fused decode
        /// kernel is unavailable. Native GPU paths handle F16 K/V directly via
        /// <c>ggml_flash_attn_ext</c>, which is much faster.
        /// </summary>
        protected unsafe void AttentionDecodePureCSF16(Tensor q, Tensor kCache, Tensor vCache,
            Tensor result, int numHeads, int numKVHeads, int headDim, int totalSeqLen, float scale)
        {
            AttentionDecodeGroupedF16(GetFloatPtr(q),
                GetHalfPtrForRead(kCache), GetHalfPtrForRead(vCache),
                GetFloatPtr(result), numHeads, numKVHeads, headDim, (int)kCache.Sizes[1], totalSeqLen, scale);
        }

        /// <summary>
        /// The F16-cache kernel behind <see cref="AttentionDecodePureCSF16"/>, on raw pointers
        /// (layouts as <see cref="AttentionDecodeGrouped"/>). One block per KV head, each with a
        /// (groupSize, totalSeqLen) score scratch that is stack memory only up to
        /// <see cref="CpuScratch.StackFloatLimit"/> floats: this is the block that overflowed a
        /// <see cref="CpuWorkerPool"/> worker for Nemotron-3.5 Lightning at 13,844 keys.
        /// </summary>
        internal static unsafe void AttentionDecodeGroupedF16(float* qPtr, ushort* kPtr, ushort* vPtr, float* rPtr,
            int numHeads, int numKVHeads, int headDim, int maxSeqLen, int totalSeqLen, float scale)
        {
            int groupSize = numHeads / numKVHeads;
            // Read on the entering thread and captured by the blocks (see CpuScratch).
            int stackLimit = CpuScratch.StackFloatLimit;

            bool useParallel = numKVHeads > 1 && (long)numHeads * totalSeqLen >= 4096;

            if (useParallel)
            {
                long qPtrL = (long)qPtr;
                long kPtrL = (long)kPtr;
                long vPtrL = (long)vPtr;
                long rPtrL = (long)rPtr;
                int totalSeqLenLocal = totalSeqLen;
                int headDimLocal = headDim;
                int maxSeqLenLocal = maxSeqLen;
                int groupSizeLocal = groupSize;
                int numKVHeadsLocal = numKVHeads;
                float scaleLocal = scale;

                CpuParallel.For(numKVHeadsLocal, kvHead =>
                {
                    float* qP = (float*)qPtrL;
                    ushort* kP = (ushort*)kPtrL;
                    ushort* vP = (ushort*)vPtrL;
                    float* rP = (float*)rPtrL;
                    int scoreFloats = groupSizeLocal * totalSeqLenLocal;
                    float[] rented = null;
                    Span<float> scratch = scoreFloats <= stackLimit
                        ? stackalloc float[scoreFloats]
                        : (rented = CpuScratch.RentFloats(scoreFloats)).AsSpan(0, scoreFloats);
                    try
                    {
                        fixed (float* scoresBuf = scratch)
                        {
                            AttentionDecodeKVHeadGroupedF16(kvHead, qP, kP, vP, rP, scoresBuf,
                                headDimLocal, maxSeqLenLocal, groupSizeLocal,
                                totalSeqLenLocal, scaleLocal);
                        }
                    }
                    finally
                    {
                        CpuScratch.Return(rented);
                    }
                });
            }
            else
            {
                int scoreFloats = groupSize * totalSeqLen;
                float[] rented = null;
                Span<float> scratch = scoreFloats <= stackLimit
                    ? stackalloc float[scoreFloats]
                    : (rented = CpuScratch.RentFloats(scoreFloats)).AsSpan(0, scoreFloats);
                try
                {
                    fixed (float* scores = scratch)
                    {
                        for (int kvHead = 0; kvHead < numKVHeads; kvHead++)
                        {
                            AttentionDecodeKVHeadGroupedF16(kvHead, qPtr, kPtr, vPtr, rPtr, scores,
                                headDim, maxSeqLen, groupSize, totalSeqLen, scale);
                        }
                    }
                }
                finally
                {
                    CpuScratch.Return(rented);
                }
            }
        }

        private static unsafe void AttentionDecodeKVHeadGroupedF16(int kvHead,
            float* qPtr, ushort* kPtr, ushort* vPtr, float* rPtr, float* scores,
            int headDim, int maxSeqLen, int groupSize, int totalSeqLen, float scale)
        {
            int hStart = kvHead * groupSize;
            ushort* kHead = kPtr + (long)kvHead * maxSeqLen * headDim;
            ushort* vHead = vPtr + (long)kvHead * maxSeqLen * headDim;

            float maxG0 = float.NegativeInfinity;
            float maxG1 = float.NegativeInfinity;
            float maxG2 = float.NegativeInfinity;
            float maxG3 = float.NegativeInfinity;

            if (groupSize == 4)
            {
                float* qH0 = qPtr + (long)(hStart + 0) * headDim;
                float* qH1 = qPtr + (long)(hStart + 1) * headDim;
                float* qH2 = qPtr + (long)(hStart + 2) * headDim;
                float* qH3 = qPtr + (long)(hStart + 3) * headDim;
                float* row0 = scores + 0L * totalSeqLen;
                float* row1 = scores + 1L * totalSeqLen;
                float* row2 = scores + 2L * totalSeqLen;
                float* row3 = scores + 3L * totalSeqLen;

                for (int t = 0; t < totalSeqLen; t++)
                {
                    ushort* kT = kHead + (long)t * headDim;
                    float s0, s1, s2, s3;
                    TensorComputePrimitives.Dot4F32F16(qH0, qH1, qH2, qH3, kT, headDim,
                        out s0, out s1, out s2, out s3);
                    s0 *= scale; s1 *= scale; s2 *= scale; s3 *= scale;
                    row0[t] = s0; row1[t] = s1; row2[t] = s2; row3[t] = s3;
                    if (s0 > maxG0) maxG0 = s0;
                    if (s1 > maxG1) maxG1 = s1;
                    if (s2 > maxG2) maxG2 = s2;
                    if (s3 > maxG3) maxG3 = s3;
                }
            }
            else
            {
                Span<float> maxScoresSpan = stackalloc float[groupSize];
                for (int g = 0; g < groupSize; g++) maxScoresSpan[g] = float.NegativeInfinity;

                for (int t = 0; t < totalSeqLen; t++)
                {
                    ushort* kT = kHead + (long)t * headDim;
                    for (int g = 0; g < groupSize; g++)
                    {
                        float* qH = qPtr + (long)(hStart + g) * headDim;
                        float s = TensorComputePrimitives.DotF32F16(qH, kT, headDim) * scale;
                        scores[g * totalSeqLen + t] = s;
                        if (s > maxScoresSpan[g]) maxScoresSpan[g] = s;
                    }
                }

                if (groupSize >= 1) maxG0 = maxScoresSpan[0];
                if (groupSize >= 2) maxG1 = maxScoresSpan[1];
                if (groupSize >= 3) maxG2 = maxScoresSpan[2];
                if (groupSize >= 4) maxG3 = maxScoresSpan[3];
            }

            // Softmax (per-group)
            Span<float> invSums = stackalloc float[groupSize];
            for (int g = 0; g < groupSize; g++)
            {
                float maxS;
                if (g == 0) maxS = maxG0;
                else if (g == 1) maxS = maxG1;
                else if (g == 2) maxS = maxG2;
                else if (g == 3) maxS = maxG3;
                else
                {
                    maxS = float.NegativeInfinity;
                    float* rowG0 = scores + (long)g * totalSeqLen;
                    for (int t = 0; t < totalSeqLen; t++)
                        if (rowG0[t] > maxS) maxS = rowG0[t];
                }

                float* rowG = scores + (long)g * totalSeqLen;
                invSums[g] = 1.0f / VecExpRow(rowG, maxS, totalSeqLen);
            }
            for (int g = 0; g < groupSize; g++)
            {
                float invSum = invSums[g];
                float* rowG = scores + (long)g * totalSeqLen;
                VecScale(rowG, invSum, totalSeqLen);
            }

            // Aggregate V (F16): read V[t] once per t, scatter into all groupSize result heads.
            for (int g = 0; g < groupSize; g++)
                VecZero(rPtr + (long)(hStart + g) * headDim, headDim);

            if (groupSize == 4)
            {
                float* r0 = rPtr + (long)(hStart + 0) * headDim;
                float* r1 = rPtr + (long)(hStart + 1) * headDim;
                float* r2 = rPtr + (long)(hStart + 2) * headDim;
                float* r3 = rPtr + (long)(hStart + 3) * headDim;
                float* row0 = scores + 0L * totalSeqLen;
                float* row1 = scores + 1L * totalSeqLen;
                float* row2 = scores + 2L * totalSeqLen;
                float* row3 = scores + 3L * totalSeqLen;

                for (int t = 0; t < totalSeqLen; t++)
                {
                    ushort* vT = vHead + (long)t * headDim;
                    TensorComputePrimitives.ScaleAdd4F16(r0, r1, r2, r3, vT,
                        row0[t], row1[t], row2[t], row3[t], headDim);
                }
            }
            else
            {
                for (int t = 0; t < totalSeqLen; t++)
                {
                    ushort* vT = vHead + (long)t * headDim;
                    for (int g = 0; g < groupSize; g++)
                    {
                        float w = scores[g * totalSeqLen + t];
                        float* rH = rPtr + (long)(hStart + g) * headDim;
                        TensorComputePrimitives.ScaleAddF16(rH, vT, w, headDim);
                    }
                }
            }
        }

    }
}
