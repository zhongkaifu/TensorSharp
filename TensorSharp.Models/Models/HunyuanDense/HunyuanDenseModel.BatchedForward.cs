// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Batched paged-attention forward for hunyuan-dense, following the Mistral 3
// reference implementation of IBatchedPagedModel: one forward packs every
// scheduled token of every sequence, scatters K/V into layer-private paged
// buffers by slot mapping, and runs per-sequence causal attention through the
// native paged-attention kernel on GGML backends (managed kernel elsewhere).
//
// Hunyuan-specific details, identical to the single-sequence path in
// HunyuanDenseModel.cs:
//   * NeoX RoPE (mode 2) FIRST, then per-head Q/K RMSNorm. Qwen 3 normalises
//     before rotating; copying its order gives fluent but wrong output.
//   * Q/K/V may be fused (attn_qkv) or separate, per layer.
//   * FFN gate/up may be fused (ffn_gate_up) or separate, per layer.
using System;
using System.Collections.Generic;
using TensorSharp;
using TensorSharp.Models.Paged;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models
{
    public sealed partial class HunyuanDenseModel : IBatchedPagedModel
    {
        private float[][] _pagedKBuf;
        private float[][] _pagedVBuf;
        private int _pagedNumBlocks;
        private int _pagedBlockSize;

        /// <summary>
        /// The paged buffers are F32 and the engine migrates nothing into them from a
        /// block-quantized linear cache, so a q8_0/q4_0 KV cache keeps the snapshot path
        /// (which serves those dtypes byte-exactly) instead. TS_HUNYUAN_BATCHED=0 forces
        /// the snapshot path for A/B comparison.
        /// </summary>
        public bool BatchedForwardAvailable =>
            !_kvCacheDtype.IsBlockQuantized() &&
            !string.Equals(Environment.GetEnvironmentVariable("TS_HUNYUAN_BATCHED"), "0", StringComparison.Ordinal);

        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx)
        {
            if (ctx == null) throw new ArgumentNullException(nameof(ctx));

            int numSeqs = ctx.Sequences.Count;
            if (numSeqs == 0) return Array.Empty<float[]>();

            int hidden = Config.HiddenSize;
            int numHeads = Config.NumHeads;
            int numKvHeads = Config.NumKVHeads;
            int headDim = _attnKeyLen;
            int qDim = numHeads * headDim;
            int kDim = numKvHeads * headDim;
            float scale = 1.0f / MathF.Sqrt(headDim);

            int blockSize = ctx.Sequences[0].BlockTable.BlockSize;
            int maxBlockId = 0;
            for (int s = 0; s < numSeqs; s++)
            {
                var bt = ctx.BlockTables[s];
                for (int b = 0; b < bt.Length; b++)
                    if (bt[b] > maxBlockId) maxBlockId = bt[b];
            }
            EnsurePagedBuffersAllocated(maxBlockId + 1, blockSize, numKvHeads, headDim);

            int numTokens = 0;
            for (int s = 0; s < numSeqs; s++) numTokens += ctx.NumScheduledTokens[s];

            int[] positions = ctx.Positions.ToArray();
            int[] queryStartLoc = ctx.QueryStartLoc.ToArray();
            int[] slotMapping = ctx.SlotMapping.ToArray();
            int[] seqLens = new int[numSeqs];
            for (int s = 0; s < numSeqs; s++)
                seqLens[s] = ctx.Sequences[s].NumComputedTokens + ctx.NumScheduledTokens[s];

            // Decode steps forward the sampled-but-not-yet-committed token, which is
            // absent from the sequence's token list, so the executor passes it here.
            int[] flatTokens;
            if (ctx.OverrideFlatTokens != null)
            {
                if (ctx.OverrideFlatTokens.Length != numTokens)
                    throw new ArgumentException(
                        $"OverrideFlatTokens length {ctx.OverrideFlatTokens.Length} != numTokens {numTokens}.");
                flatTokens = ctx.OverrideFlatTokens;
            }
            else
            {
                flatTokens = new int[numTokens];
                int cursor = 0;
                for (int s = 0; s < numSeqs; s++)
                {
                    var seq = ctx.Sequences[s];
                    int startTok = seq.NumComputedTokens;
                    int take = ctx.NumScheduledTokens[s];
                    for (int i = 0; i < take; i++)
                        flatTokens[cursor++] = seq.TokenAt(startTok + i);
                }
            }

            Tensor hiddenStates = Embedding(flatTokens);

            using Tensor positionsTensorQ = BuildRoPEPositionsTensor(positions, numHeads);
            using Tensor positionsTensorK = BuildRoPEPositionsTensor(positions, numKvHeads);
            PagedAttentionKernel kernel = ResolvePagedAttentionKernel();
            (int[] blockTableFlat, int[] blockTableOffsets) = kernel == PagedAttentionKernel.Native && IsGgmlBackend
                ? FlattenBlockTables(ctx.BlockTables)
                : (null, null);

            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                string p = $"blk.{layer}.";

                Tensor residual = Ops.NewContiguous(hiddenStates);
                Tensor normed = RMSNormOp(hiddenStates, p + "attn_norm.weight");
                hiddenStates.Dispose();

                Tensor q, k, v;
                if (_layerQkvFused[layer])
                {
                    Tensor qkv = LinearForward(normed, p + "attn_qkv.weight");
                    using (Tensor qView = qkv.Narrow(1, 0, qDim)) q = Ops.NewContiguous(qView);
                    using (Tensor kView = qkv.Narrow(1, qDim, kDim)) k = Ops.NewContiguous(kView);
                    using (Tensor vView = qkv.Narrow(1, qDim + kDim, kDim)) v = Ops.NewContiguous(vView);
                    qkv.Dispose();
                }
                else
                {
                    q = LinearForward(normed, p + "attn_q.weight");
                    k = LinearForward(normed, p + "attn_k.weight");
                    v = LinearForward(normed, p + "attn_v.weight");
                }
                normed.Dispose();

                // NeoX RoPE, then per-head Q/K RMSNorm (Hunyuan's order).
                q = ApplyBatchedNeoXRoPE(q, positionsTensorQ, numTokens, numHeads, headDim);
                k = ApplyBatchedNeoXRoPE(k, positionsTensorK, numTokens, numKvHeads, headDim);
                q = ApplyBatchedQKNorm(q, _weights[p + "attn_q_norm.weight"], numTokens, numHeads, headDim);
                k = ApplyBatchedQKNorm(k, _weights[p + "attn_k_norm.weight"], numTokens, numKvHeads, headDim);

                float[] kFlat = k.GetElementsAsFloat(numTokens * kDim);
                float[] vFlat = v.GetElementsAsFloat(numTokens * kDim);
                PagedKvBatchOps.ScatterKv(
                    kFlat, vFlat, _pagedKBuf[layer], _pagedVBuf[layer],
                    slotMapping, numTokens, numKvHeads, headDim, _pagedBlockSize);
                k.Dispose();
                v.Dispose();

                float[] qFlat = q.GetElementsAsFloat(numTokens * qDim);
                q.Dispose();
                float[] attnFlat = new float[numTokens * qDim];
                if (blockTableFlat != null)
                {
                    TensorSharp.GGML.GgmlBasicOps.PagedAttentionForward(
                        qFlat, _pagedKBuf[layer], _pagedVBuf[layer], attnFlat,
                        queryStartLoc, seqLens, positions,
                        blockTableFlat, blockTableOffsets,
                        numSeqs, numTokens, numHeads, numKvHeads, headDim,
                        _pagedBlockSize, scale);
                }
                else if (kernel == PagedAttentionKernel.Tensor)
                {
                    TensorPagedAttention.Forward(
                        _allocator, IsGgmlBackend,
                        qFlat, _pagedKBuf[layer], _pagedVBuf[layer], attnFlat,
                        numTokens, numHeads, numKvHeads, headDim, _pagedBlockSize,
                        queryStartLoc, seqLens, positions, ctx.BlockTables, numSeqs,
                        scale, causal: true);
                }
                else
                {
                    ManagedPagedAttention.Forward(
                        qFlat, _pagedKBuf[layer], _pagedVBuf[layer], attnFlat,
                        numTokens, numHeads, numKvHeads, headDim, _pagedBlockSize,
                        queryStartLoc, seqLens, positions, ctx.BlockTables, numSeqs,
                        scale, causal: true);
                }

                Tensor attnOut = CreateFloatTensor(attnFlat, numTokens, qDim);
                Tensor attnProj = LinearForward(attnOut, p + "attn_output.weight");
                attnOut.Dispose();
                Ops.Add(residual, residual, attnProj);
                attnProj.Dispose();

                if (!(_layerGateUpFused[layer] &&
                      TryFusedDenseSwiGLUFFNInto(residual, p + "ffn_norm.weight", p + "ffn_gate_up.weight", p + "ffn_down.weight")))
                {
                    Tensor ffnNormed = RMSNormOp(residual, p + "ffn_norm.weight");
                    Tensor ffnOut = FFNLayer(ffnNormed, layer, numTokens);
                    ffnNormed.Dispose();
                    Ops.Add(residual, residual, ffnOut);
                    ffnOut.Dispose();
                }
                hiddenStates = residual;
            }

            Tensor finalNormed = RMSNormOp(hiddenStates, "output_norm.weight");
            hiddenStates.Dispose();
            float[] finalFlat = finalNormed.GetElementsAsFloat(numTokens * hidden);
            finalNormed.Dispose();
            float[] lastTokensPacked = PagedKvBatchOps.GatherLastTokenPerSeq(finalFlat, hidden, queryStartLoc, numSeqs);
            Tensor lastHidden = CreateFloatTensor(lastTokensPacked, numSeqs, hidden);

            Tensor logitsTensor = LinearForward(lastHidden, "output.weight")
                ?? LinearForward(lastHidden, "token_embd.weight");
            lastHidden.Dispose();
            float[] allLogits = logitsTensor.GetElementsAsFloat(numSeqs * Config.VocabSize);            logitsTensor.Dispose();

            var perSeq = new float[numSeqs][];
            for (int s = 0; s < numSeqs; s++)
            {
                var slice = new float[Config.VocabSize];
                Buffer.BlockCopy(allLogits, s * Config.VocabSize * sizeof(float),
                                 slice, 0, Config.VocabSize * sizeof(float));
                perSeq[s] = slice;
            }
            return perSeq;
        }

        /// <summary>NeoX RoPE with one position per (token, head) row - the same call the
        /// single-sequence path makes, with per-token rather than contiguous positions.</summary>
        private Tensor ApplyBatchedNeoXRoPE(Tensor data, Tensor positionsTensor, int numTokens, int numHeads, int headDim)
        {
            using Tensor reshaped = data.View(1, numTokens, numHeads, headDim);
            Tensor result = Ops.RoPEEx(
                null, reshaped, positionsTensor, _ropeDim, 2, 0,
                Config.RopeBase, 1.0f / Config.RopeScale,
                0.0f, 1.0f, 0.0f, 0.0f);
            data.Dispose();
            Tensor flat = result.View(numTokens, numHeads * headDim);
            result.Dispose();
            return flat;
        }

        private Tensor ApplyBatchedQKNorm(Tensor data, Tensor alpha, int numTokens, int numHeads, int headDim)
        {
            using Tensor reshaped = data.View(numTokens * numHeads, headDim);
            Tensor normed = Ops.RMSNorm(null, reshaped, alpha, null, Config.Eps);
            data.Dispose();
            Tensor flat = normed.View(numTokens, numHeads * headDim);
            normed.Dispose();
            return flat;
        }

        private enum PagedAttentionKernel { Native, Tensor, Managed }

        /// <summary>Same switch and spellings as Mistral 3: TS_PAGED_ATTN_KERNEL.</summary>
        private static PagedAttentionKernel ResolvePagedAttentionKernel()
        {
            string raw = Environment.GetEnvironmentVariable("TS_PAGED_ATTN_KERNEL");
            if (string.IsNullOrEmpty(raw)) return PagedAttentionKernel.Native;
            return raw.Trim().ToLowerInvariant() switch
            {
                "tensor" or "gpu" or "addmm" => PagedAttentionKernel.Tensor,
                "managed" or "scalar" or "0" or "false" => PagedAttentionKernel.Managed,
                _ => PagedAttentionKernel.Native,
            };
        }

        private static (int[] flat, int[] offsets) FlattenBlockTables(int[][] tables)
        {
            int total = 0;
            var offsets = new int[tables.Length];
            for (int s = 0; s < tables.Length; s++)
            {
                offsets[s] = total;
                total += tables[s].Length;
            }
            var flat = new int[total];
            for (int s = 0; s < tables.Length; s++)
                Array.Copy(tables[s], 0, flat, offsets[s], tables[s].Length);
            return (flat, offsets);
        }

        private Tensor BuildRoPEPositionsTensor(int[] tokenPositions, int numHeads)
        {
            int total = tokenPositions.Length * numHeads;
            int[] expanded = new int[total];
            for (int t = 0; t < tokenPositions.Length; t++)
                for (int h = 0; h < numHeads; h++)
                    expanded[t * numHeads + h] = tokenPositions[t];
            return CreateIntTensor(expanded, total);
        }

        /// <summary>Grow the per-layer paged buffers, preserving the K/V already written:
        /// a later step can add sequences whose decode still reads an earlier sequence's
        /// prefill rows.</summary>
        private void EnsurePagedBuffersAllocated(int numBlocks, int blockSize, int numKvHeads, int headDim)
        {
            if (_pagedKBuf != null && _pagedNumBlocks >= numBlocks && _pagedBlockSize == blockSize)
                return;
            int targetBlocks = Math.Max(numBlocks, _pagedNumBlocks * 2);
            float[][] oldK = _pagedKBuf;
            float[][] oldV = _pagedVBuf;
            int oldNumBlocks = _pagedNumBlocks;
            int oldBlockSize = _pagedBlockSize;

            _pagedKBuf = new float[Config.NumLayers][];
            _pagedVBuf = new float[Config.NumLayers][];
            for (int l = 0; l < Config.NumLayers; l++)
            {
                _pagedKBuf[l] = PagedKvBatchOps.AllocateLayerBuffer(targetBlocks, blockSize, numKvHeads, headDim);
                _pagedVBuf[l] = PagedKvBatchOps.AllocateLayerBuffer(targetBlocks, blockSize, numKvHeads, headDim);
                if (oldK != null && oldK[l] != null && oldBlockSize == blockSize)
                {
                    long copyLen = Math.Min((long)oldNumBlocks * blockSize * numKvHeads * headDim, oldK[l].LongLength);
                    Array.Copy(oldK[l], _pagedKBuf[l], copyLen);
                    Array.Copy(oldV[l], _pagedVBuf[l], copyLen);
                }
            }
            _pagedNumBlocks = targetBlocks;
            _pagedBlockSize = blockSize;
        }
    }
}
