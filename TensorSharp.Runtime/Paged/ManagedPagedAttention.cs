// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

namespace TensorSharp.Runtime.Paged
{
    /// <summary>
    /// Managed (pure C#) implementation of batched paged attention. Mirrors the
    /// shape and semantics of vLLM's <c>flash_attn_varlen_func</c>: a single
    /// kernel call handles many sequences of varying length, gathering K and V
    /// from a paged block pool via a per-sequence block table and a per-token
    /// slot mapping. Causal-only masking, no ALiBi, no sliding window (yet) -
    /// just the core building block.
    ///
    /// This is the **reference / correctness** implementation. Per-token
    /// throughput is far lower than a fused GGML/CUDA kernel; the value here
    /// is that one call can drive an arbitrary mix of prefill chunks and
    /// decode tokens with one shared layer of weights, which is the
    /// continuous-batching win. A native paged attention kernel (the
    /// long-tail item in the architecture doc) drops into this same
    /// interface.
    ///
    /// Tensor layout convention
    /// ------------------------
    ///   q                : [numTokens, numHeads, headDim]     (row-major)
    ///   kBlocks / vBlocks: [numBlocks, blockSize, numKvHeads, headDim]
    ///   out              : [numTokens, numHeads, headDim]     (row-major)
    ///
    /// Per-sequence metadata
    /// ---------------------
    ///   queryStartLoc[s] : index of first token of sequence s in q
    ///                       (length numSeqs+1, last entry = numTokens)
    ///   seqLens[s]       : total context length (prompt + generated so far)
    ///   blockTables[s]   : list of block ids covering positions [0..seqLens[s])
    ///   positions[t]     : absolute position of token t within its sequence
    ///                       (used for causal masking, NOT for K/V write
    ///                       location)
    ///   slotMapping[t]   : physical slot in the K/V pool where token t's
    ///                       K and V are stored
    ///                       (slot = blockId * blockSize + offsetInBlock)
    /// </summary>
    public static class ManagedPagedAttention
    {
        /// <summary>Multi-query / grouped-query attention. The K/V have a
        /// (potentially smaller) <paramref name="numKvHeads"/>; each Q head is
        /// mapped to a KV head via <paramref name="numHeads"/> / <paramref name="numKvHeads"/>.
        /// Backed by <c>float[]</c> arrays so we can parallelise across
        /// (sequence, head) without ref-struct capture restrictions.</summary>
        public static void Forward(
            float[] q,
            float[] kBlocks,
            float[] vBlocks,
            float[] output,
            int numTokens,
            int numHeads,
            int numKvHeads,
            int headDim,
            int blockSize,
            int[] queryStartLoc,
            int[] seqLens,
            int[] positions,
            int[][] blockTables,
            int numSeqs,
            float scale,
            bool causal = true)
        {
            if (numHeads % numKvHeads != 0)
                throw new ArgumentException("numHeads must be divisible by numKvHeads.");
            int groupSize = numHeads / numKvHeads;

            // Parallelise across (seq, head). Each output token's attention
            // computes independently so this is embarrassingly parallel.
            Parallel.For(0, numSeqs * numHeads, work =>
            {
                int seqIdx = work / numHeads;
                int headIdx = work % numHeads;
                int kvHead = headIdx / groupSize;

                int qStart = queryStartLoc[seqIdx];
                int qEnd = queryStartLoc[seqIdx + 1];
                int seqLen = seqLens[seqIdx];
                int[] table = blockTables[seqIdx];

                for (int t = qStart; t < qEnd; t++)
                {
                    int pos = positions[t];
                    ComputeSingleQueryAttention(
                        q, kBlocks, vBlocks, output,
                        tokenIdx: t,
                        headIdx: headIdx,
                        kvHead: kvHead,
                        numHeads: numHeads,
                        numKvHeads: numKvHeads,
                        headDim: headDim,
                        blockSize: blockSize,
                        contextEndExclusive: causal ? pos + 1 : seqLen,
                        contextStart: 0,
                        blockTable: table,
                        scale: scale);
                }
            });
        }

        /// <summary>
        /// Online-softmax single-query paged attention. Walks the sequence's
        /// blocks, accumulating <c>numerator</c> = Σ exp(s_i - m) v_i and
        /// <c>denominator</c> = Σ exp(s_i - m), tracking the running max <c>m</c>
        /// to keep the exponents from overflowing. Final output is
        /// <c>numerator / denominator</c>. This is the same pattern as
        /// FlashAttention's recurrence, just unfused and unsimd'd.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ComputeSingleQueryAttention(
            float[] q,
            float[] kBlocks,
            float[] vBlocks,
            float[] output,
            int tokenIdx,
            int headIdx,
            int kvHead,
            int numHeads,
            int numKvHeads,
            int headDim,
            int blockSize,
            int contextEndExclusive,
            int contextStart,
            int[] blockTable,
            float scale)
        {
            // q[token, headIdx, :]
            int qOffset = (tokenIdx * numHeads + headIdx) * headDim;

            // Online softmax accumulators. We can't use stackalloc here because
            // this method is called from a Parallel.For lambda and headDim is
            // potentially large (256+). Heap-allocate one buffer per (seq,head)
            // pair - the JIT escape-analyses it away in the common path.
            float maxScore = float.NegativeInfinity;
            float denom = 0f;
            var numerator = new float[headDim];

            for (int contextPos = contextStart; contextPos < contextEndExclusive; contextPos++)
            {
                int blockIdx = contextPos / blockSize;
                if (blockIdx >= blockTable.Length) break; // ran out of allocated blocks
                int slotInBlock = contextPos % blockSize;
                int blockId = blockTable[blockIdx];

                // K and V at (blockId, slotInBlock, kvHead, :)
                int kvBaseOffset =
                    blockId * blockSize * numKvHeads * headDim
                  + slotInBlock * numKvHeads * headDim
                  + kvHead * headDim;

                // Compute Q . K
                float score = 0f;
                for (int d = 0; d < headDim; d++)
                    score += q[qOffset + d] * kBlocks[kvBaseOffset + d];
                score *= scale;

                if (score > maxScore)
                {
                    // Renormalize old terms by the new max.
                    float renorm = MathF.Exp(maxScore - score);
                    denom *= renorm;
                    for (int d = 0; d < headDim; d++)
                        numerator[d] *= renorm;
                    maxScore = score;
                }

                float e = MathF.Exp(score - maxScore);
                denom += e;
                for (int d = 0; d < headDim; d++)
                    numerator[d] += e * vBlocks[kvBaseOffset + d];
            }

            int outOffset = (tokenIdx * numHeads + headIdx) * headDim;
            if (denom == 0f)
            {
                // Empty context: write zeros.
                for (int d = 0; d < headDim; d++)
                    output[outOffset + d] = 0f;
                return;
            }
            float invDenom = 1f / denom;
            for (int d = 0; d < headDim; d++)
                output[outOffset + d] = numerator[d] * invDenom;
        }

        /// <summary>
        /// Paged attention with per-head attention sinks (gpt-oss style) and
        /// per-layer sliding window. Sinks are an extra "virtual" position per
        /// head: a learned scalar logit that participates in the softmax
        /// denominator but contributes zero to the output (sink V == 0). The
        /// effect is to bleed off attention mass when no real position is
        /// strongly relevant, stabilising long contexts.
        ///
        /// Math is identical to <see cref="Forward"/>'s online softmax, with
        /// two changes:
        ///   * <paramref name="sinks"/>[h] seeds the running max as a virtual
        ///     position so the first real score correctly normalises against
        ///     it. The denominator picks up the sink's exp(sink_logit - max);
        ///     the numerator does NOT (sink position has zero V).
        ///   * <paramref name="slidingWindow"/> &gt; 0 truncates context to the
        ///     last <c>slidingWindow</c> tokens (matches the per-token-position
        ///     mask the gpt-oss attention layers alternate ON for even layers).
        /// </summary>
        public static void ForwardWithSinks(
            float[] q,
            float[] kBlocks,
            float[] vBlocks,
            float[] output,
            int numTokens,
            int numHeads,
            int numKvHeads,
            int headDim,
            int blockSize,
            int[] queryStartLoc,
            int[] seqLens,
            int[] positions,
            int[][] blockTables,
            int numSeqs,
            float scale,
            float[] sinks,           // [numHeads] or null
            int slidingWindow)        // 0 = no SWA
        {
            if (numHeads % numKvHeads != 0)
                throw new ArgumentException("numHeads must be divisible by numKvHeads.");
            int groupSize = numHeads / numKvHeads;

            Parallel.For(0, numSeqs * numHeads, work =>
            {
                int seqIdx = work / numHeads;
                int headIdx = work % numHeads;
                int kvHead = headIdx / groupSize;

                int qStart = queryStartLoc[seqIdx];
                int qEnd = queryStartLoc[seqIdx + 1];
                int[] table = blockTables[seqIdx];
                float sinkLogit = sinks != null ? sinks[headIdx] : float.NegativeInfinity;

                for (int t = qStart; t < qEnd; t++)
                {
                    int pos = positions[t];
                    int contextEndExclusive = pos + 1; // causal
                    int contextStart = slidingWindow > 0
                        ? Math.Max(0, contextEndExclusive - slidingWindow)
                        : 0;

                    ComputeSingleQueryAttentionWithSinks(
                        q, kBlocks, vBlocks, output,
                        tokenIdx: t,
                        headIdx: headIdx,
                        kvHead: kvHead,
                        numHeads: numHeads,
                        numKvHeads: numKvHeads,
                        headDim: headDim,
                        blockSize: blockSize,
                        contextEndExclusive: contextEndExclusive,
                        contextStart: contextStart,
                        blockTable: table,
                        scale: scale,
                        sinkLogit: sinkLogit);
                }
            });
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void ComputeSingleQueryAttentionWithSinks(
            float[] q,
            float[] kBlocks,
            float[] vBlocks,
            float[] output,
            int tokenIdx,
            int headIdx,
            int kvHead,
            int numHeads,
            int numKvHeads,
            int headDim,
            int blockSize,
            int contextEndExclusive,
            int contextStart,
            int[] blockTable,
            float scale,
            float sinkLogit)
        {
            int qOffset = (tokenIdx * numHeads + headIdx) * headDim;

            // Seed the online softmax with the sink as a virtual position:
            // maxScore = sinkLogit, denom = exp(sinkLogit - maxScore) = 1.
            // numerator stays zero because the sink has zero V contribution.
            bool hasSink = !float.IsNegativeInfinity(sinkLogit);
            float maxScore = hasSink ? sinkLogit : float.NegativeInfinity;
            float denom = hasSink ? 1f : 0f;
            var numerator = new float[headDim];

            for (int contextPos = contextStart; contextPos < contextEndExclusive; contextPos++)
            {
                int blockIdx = contextPos / blockSize;
                if (blockIdx >= blockTable.Length) break;
                int slotInBlock = contextPos % blockSize;
                int blockId = blockTable[blockIdx];

                int kvBaseOffset =
                    blockId * blockSize * numKvHeads * headDim
                  + slotInBlock * numKvHeads * headDim
                  + kvHead * headDim;

                float score = 0f;
                for (int d = 0; d < headDim; d++)
                    score += q[qOffset + d] * kBlocks[kvBaseOffset + d];
                score *= scale;

                if (score > maxScore)
                {
                    float renorm = MathF.Exp(maxScore - score);
                    denom *= renorm;
                    for (int d = 0; d < headDim; d++)
                        numerator[d] *= renorm;
                    maxScore = score;
                }

                float e = MathF.Exp(score - maxScore);
                denom += e;
                for (int d = 0; d < headDim; d++)
                    numerator[d] += e * vBlocks[kvBaseOffset + d];
            }

            int outOffset = (tokenIdx * numHeads + headIdx) * headDim;
            if (denom == 0f)
            {
                for (int d = 0; d < headDim; d++) output[outOffset + d] = 0f;
                return;
            }
            float invDenom = 1f / denom;
            for (int d = 0; d < headDim; d++)
                output[outOffset + d] = numerator[d] * invDenom;
        }

        /// <summary>
        /// Scatter the new tokens' K and V into the paged pool at the slots
        /// given by <paramref name="slotMapping"/>. K and V come in as
        /// <c>[numTokens, numKvHeads, headDim]</c>. The paged buffers are
        /// laid out as <c>[numBlocks, blockSize, numKvHeads, headDim]</c>.
        /// </summary>
        public static void WriteKvToPagedPool(
            float[] k,
            float[] v,
            float[] kBlocks,
            float[] vBlocks,
            int[] slotMapping,
            int numTokens,
            int numKvHeads,
            int headDim,
            int blockSize)
        {
            int perTokenStride = numKvHeads * headDim;
            for (int t = 0; t < numTokens; t++)
            {
                int slot = slotMapping[t];
                int blockId = slot / blockSize;
                int slotInBlock = slot % blockSize;
                int dstOffset =
                    blockId * blockSize * perTokenStride
                  + slotInBlock * perTokenStride;
                int srcOffset = t * perTokenStride;
                for (int i = 0; i < perTokenStride; i++)
                {
                    kBlocks[dstOffset + i] = k[srcOffset + i];
                    vBlocks[dstOffset + i] = v[srcOffset + i];
                }
            }
        }
    }
}
