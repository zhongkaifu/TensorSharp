// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// GPU-backed (Tensor-based) batched paged-attention kernel. Drop-in
// replacement for <see cref="TensorSharp.Runtime.Paged.ManagedPagedAttention"/>
// with the same signature, but instead of running a scalar online softmax
// in C# it materialises each sequence's K/V into a contiguous Tensor
// (gather from paged blocks) and then drives the existing GPU/Metal/CUDA
// attention path: <c>Ops.AddmmBatch</c> for Q@K^T and softmax@V, with
// <see cref="GgmlBasicOps.AttentionSoftmaxWithSinks"/> (GGML backends) or
// <c>Ops.AddCausalMask</c> + <c>Ops.Softmax</c> (other backends) for the
// causal masked softmax in between.
//
// Trade-off: paid a gather cost (copy K/V from non-contiguous block
// storage into a contiguous Tensor) per sequence per layer, but we
// recoup it many times over by using the optimised attention kernel.
// On a real 14B model this becomes ~10× faster than ManagedPagedAttention.
//
// This is still NOT a "true" native paged-attention kernel - it doesn't
// fuse the gather into the attention compute the way vLLM's
// flash_attn_varlen does. It's the practical middle step: GPU
// acceleration without writing a new CUDA/Metal kernel.
using System;
using TensorSharp;
using TensorSharp.GGML;

namespace TensorSharp.Models.Paged
{
    public static class TensorPagedAttention
    {
        /// <summary>
        /// Multi-query / grouped-query batched paged attention via the
        /// existing TensorSharp Tensor ops. See the file header for the
        /// trade-off vs the pure-managed kernel.
        /// </summary>
        /// <param name="allocator">The model's allocator (so Tensors live on
        /// the right backend).</param>
        /// <param name="isGgmlBackend">When true, use the fused GGML
        /// causal-mask+softmax kernel; otherwise use generic ops.</param>
        public static void Forward(
            IAllocator allocator,
            bool isGgmlBackend,
            float[] q,           // [numTokens, numHeads * headDim]
            float[] kBlocks,     // [numBlocks, blockSize, numKvHeads, headDim] flat
            float[] vBlocks,     // [numBlocks, blockSize, numKvHeads, headDim] flat
            float[] output,      // [numTokens, numHeads * headDim] (writes back)
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
            int qDim = numHeads * headDim;
            int kvDim = numKvHeads * headDim;

            for (int s = 0; s < numSeqs; s++)
            {
                int qStart = queryStartLoc[s];
                int qEnd = queryStartLoc[s + 1];
                int numQ = qEnd - qStart;
                if (numQ <= 0) continue;
                int seqLen = seqLens[s];
                int[] table = blockTables[s];
                int firstQueryPos = positions[qStart];

                // Gather K, V from the paged buffer into contiguous float[].
                float[] kFlat = GatherFromPagedBlocks(kBlocks, table, seqLen, blockSize, kvDim);
                float[] vFlat = GatherFromPagedBlocks(vBlocks, table, seqLen, blockSize, kvDim);

                // Make Tensors. Each is on the model's allocator (Metal /
                // CUDA / CPU) so subsequent matmuls dispatch on that
                // backend.
                using var qTensorFlat = MakeTensor(allocator, q, qStart * qDim, numQ * qDim, numQ, qDim);
                using var kTensorFlat = MakeTensor(allocator, kFlat, 0, kFlat.Length, seqLen, kvDim);
                using var vTensorFlat = MakeTensor(allocator, vFlat, 0, vFlat.Length, seqLen, kvDim);

                // Reshape Q to head-first: [numHeads, numQ, headDim].
                Tensor qHeads = ReshapeToHeadFirst(qTensorFlat, numHeads, numQ, headDim);

                // Reshape K, V to head-first: [numKvHeads, seqLen, headDim].
                Tensor kHeads = ReshapeToHeadFirst(kTensorFlat, numKvHeads, seqLen, headDim);
                Tensor vHeads = ReshapeToHeadFirst(vTensorFlat, numKvHeads, seqLen, headDim);

                // Expand KV heads for grouped-query attention (a no-op when
                // numHeads == numKvHeads).
                Tensor kExp, vExp;
                if (groupSize > 1)
                {
                    kExp = Ops.RepeatInterleave(null, kHeads, groupSize, 0);
                    vExp = Ops.RepeatInterleave(null, vHeads, groupSize, 0);
                    kHeads.Dispose();
                    vHeads.Dispose();
                }
                else
                {
                    kExp = kHeads;
                    vExp = vHeads;
                }

                // Scores = scale * Q @ K^T.
                Tensor scores;
                using (var kT = kExp.Transpose(1, 2))
                {
                    scores = new Tensor(allocator, DType.Float32, numHeads, numQ, seqLen);
                    Ops.AddmmBatch(scores, 0, scores, scale, qHeads, kT);
                }
                qHeads.Dispose();
                kExp.Dispose();

                // Causal mask + softmax. GGML fuses both into one op; non-GGML
                // does them separately.
                if (causal)
                {
                    if (isGgmlBackend)
                    {
                        GgmlBasicOps.AttentionSoftmaxWithSinks(
                            scores, sinks: null,
                            numHeads: numHeads, seqLen: numQ, kvLen: seqLen,
                            maskStartPos: firstQueryPos, slidingWindow: 0, scale: 1.0f);
                    }
                    else
                    {
                        Ops.AddCausalMask(scores, numQ, firstQueryPos, float.NegativeInfinity);
                        Ops.Softmax(scores, scores);
                    }
                }
                else
                {
                    Ops.Softmax(scores, scores);
                }

                // Attention output = scores @ V.
                using var attnOutHeads = new Tensor(allocator, DType.Float32, numHeads, numQ, headDim);
                Ops.AddmmBatch(attnOutHeads, 0, attnOutHeads, 1.0f, scores, vExp);
                scores.Dispose();
                vExp.Dispose();

                // Reshape back to [numQ, qDim] and copy into the output
                // buffer at the right slice.
                using var transposed = attnOutHeads.Transpose(0, 1); // [numQ, numHeads, headDim]
                using var attnPerm = Ops.NewContiguous(transposed);
                using var flat = attnPerm.View(numQ, qDim);
                float[] outSlice = flat.GetElementsAsFloat(numQ * qDim);
                Buffer.BlockCopy(outSlice, 0, output,
                    qStart * qDim * sizeof(float), outSlice.Length * sizeof(float));
            }
        }

        /// <summary>
        /// Copy <paramref name="seqLen"/> tokens' K (or V) from the paged
        /// buffer (laid out as <c>[numBlocks, blockSize, perTokenStride]</c>)
        /// into a contiguous <c>[seqLen, perTokenStride]</c> array. The last
        /// block may be partial; we only copy <paramref name="seqLen"/> tokens'
        /// worth of slots.
        /// </summary>
        private static float[] GatherFromPagedBlocks(
            float[] paged, int[] blockTable, int seqLen, int blockSize, int perTokenStride)
        {
            var result = new float[(long)seqLen * perTokenStride];
            int numBlocks = (seqLen + blockSize - 1) / blockSize;
            int floatSize = sizeof(float);
            for (int blk = 0; blk < numBlocks; blk++)
            {
                int blockId = blockTable[blk];
                int tokensInBlock = Math.Min(blockSize, seqLen - blk * blockSize);
                int srcStart = blockId * blockSize * perTokenStride;
                int dstStart = blk * blockSize * perTokenStride;
                int bytesToCopy = tokensInBlock * perTokenStride * floatSize;
                Buffer.BlockCopy(paged, srcStart * floatSize, result, dstStart * floatSize, bytesToCopy);
            }
            return result;
        }

        /// <summary>Create a 2D Tensor from a slice of a float[].</summary>
        private static Tensor MakeTensor(IAllocator allocator, float[] src, int srcOffset, int length, int dim0, int dim1)
        {
            var t = new Tensor(allocator, DType.Float32, dim0, dim1);
            if (srcOffset == 0 && length == src.Length)
            {
                t.SetElementsAsFloat(src);
            }
            else
            {
                // Tensor.SetElementsAsFloat takes a full float[]; copy the
                // slice into a fresh array.
                var slice = new float[length];
                Buffer.BlockCopy(src, srcOffset * sizeof(float), slice, 0, length * sizeof(float));
                t.SetElementsAsFloat(slice);
            }
            return t;
        }

        /// <summary>Reshape a [length, numHeads * headDim] flat tensor to a
        /// [numHeads, length, headDim] head-first tensor via View + Transpose
        /// + Contiguous.</summary>
        private static Tensor ReshapeToHeadFirst(Tensor flat, int numHeads, int length, int headDim)
        {
            using var view = flat.View(length, numHeads, headDim);
            using var transposed = view.Transpose(0, 1);
            return Ops.NewContiguous(transposed);
        }
    }
}
