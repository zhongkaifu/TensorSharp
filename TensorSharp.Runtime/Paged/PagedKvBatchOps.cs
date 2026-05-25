// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.Runtime.Paged
{
    /// <summary>
    /// Helpers for the batched paged-attention forward path that don't fit
    /// inside <see cref="ManagedPagedAttention"/> proper - mostly format
    /// conversions between the Tensor world (used by model linear/MLP
    /// layers) and the flat-array world used by the attention kernel.
    ///
    /// All of these are pure C# right now. They can be replaced with native
    /// SIMD/Metal/CUDA kernels later without touching the model code that
    /// calls them.
    /// </summary>
    public static class PagedKvBatchOps
    {
        /// <summary>
        /// Per-layer paged K or V buffer. Shape:
        /// <c>[numBlocks, blockSize, numKvHeads, headDim]</c> as a flat
        /// <c>float[]</c>.
        /// </summary>
        public static float[] AllocateLayerBuffer(int numBlocks, int blockSize, int numKvHeads, int headDim)
        {
            return new float[(long)numBlocks * blockSize * numKvHeads * headDim];
        }

        /// <summary>
        /// Build a flat <c>float[numTokens, numHeads, headDim]</c> view by
        /// copying out the contents of a <c>[numTokens, numHeads * headDim]</c>
        /// flat array. Output element order is <c>(token, head, dim)</c>.
        /// (No-op when the input is already row-major in that layout, which
        /// is always the case for what the model's linear-projection
        /// produces.)
        /// </summary>
        public static float[] FlattenForAttention(float[] src, int numTokens, int numHeads, int headDim)
        {
            // src is already [numTokens, numHeads * headDim] row-major, which
            // is the same flat layout as [numTokens, numHeads, headDim].
            // Just return src to avoid the copy.
            if (src.Length != numTokens * numHeads * headDim)
                throw new ArgumentException(
                    $"Q/K/V buffer must have {numTokens * numHeads * headDim} elements, got {src.Length}.");
            return src;
        }

        /// <summary>
        /// Scatter new tokens' K and V into the per-layer paged buffer at
        /// the slots given by <paramref name="slotMapping"/>. K and V come
        /// in as flat row-major <c>[numTokens, numKvHeads * headDim]</c>.
        /// </summary>
        public static void ScatterKv(
            float[] k, float[] v,
            float[] kBuffer, float[] vBuffer,
            int[] slotMapping,
            int numTokens, int numKvHeads, int headDim, int blockSize)
        {
            int stride = numKvHeads * headDim;
            for (int t = 0; t < numTokens; t++)
            {
                int slot = slotMapping[t];
                int dstOffset = slot * stride;
                int srcOffset = t * stride;
                Buffer.BlockCopy(k, srcOffset * sizeof(float), kBuffer, dstOffset * sizeof(float), stride * sizeof(float));
                Buffer.BlockCopy(v, srcOffset * sizeof(float), vBuffer, dstOffset * sizeof(float), stride * sizeof(float));
            }
        }

        /// <summary>
        /// Compute slot mappings: <c>slot = blockId * blockSize + offsetInBlock</c>
        /// for each token's absolute position within its sequence.
        /// </summary>
        public static int[] ComputeSlotMapping(
            int[] positions, int[] sequenceForToken,
            int[][] blockTables, int blockSize)
        {
            int n = positions.Length;
            var slots = new int[n];
            for (int t = 0; t < n; t++)
            {
                int pos = positions[t];
                int seq = sequenceForToken[t];
                int blockIdx = pos / blockSize;
                int offset = pos % blockSize;
                slots[t] = blockTables[seq][blockIdx] * blockSize + offset;
            }
            return slots;
        }

        /// <summary>
        /// Build the per-sequence list of seq lengths from a flat
        /// queryStartLoc + per-sequence absolute lengths.
        /// </summary>
        public static int[] BuildSeqLens(int[] startPositions, int[] numScheduledTokens)
        {
            int n = startPositions.Length;
            var lens = new int[n];
            for (int s = 0; s < n; s++)
                lens[s] = startPositions[s] + numScheduledTokens[s];
            return lens;
        }

        /// <summary>
        /// Gather the LAST token of each sequence from a packed
        /// <c>[numTokens, hidden]</c> buffer. Result is
        /// <c>[numSeqs, hidden]</c>. Used right before the LM head.
        /// </summary>
        public static float[] GatherLastTokenPerSeq(
            float[] packed, int hidden,
            int[] queryStartLoc, int numSeqs)
        {
            var result = new float[(long)numSeqs * hidden];
            for (int s = 0; s < numSeqs; s++)
            {
                int lastTokenIdx = queryStartLoc[s + 1] - 1;
                int srcOffset = lastTokenIdx * hidden;
                int dstOffset = s * hidden;
                Buffer.BlockCopy(packed, srcOffset * sizeof(float), result, dstOffset * sizeof(float), hidden * sizeof(float));
            }
            return result;
        }
    }
}
