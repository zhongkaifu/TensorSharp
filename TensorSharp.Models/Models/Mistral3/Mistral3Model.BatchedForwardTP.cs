// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Tensor-parallel batched paged-attention forward path for Mistral 3.
// Combines the Megatron-LM column/row-parallel pattern with continuous
// batching. Architecture-specific details:
//   * No Q-norm / K-norm.
//   * YaRN RoPE scaling parameters.
//   * Position-dependent Q scaling.
//   * Per-layer fused vs separate QKV.
//   * Vision embedding injection before broadcast.
using System;
using System.Collections.Generic;
using TensorSharp;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models
{
    public partial class Mistral3Model
    {
        // Per-rank paged K/V buffers: [rank][layer] -> float[].
        private float[][][] _tpPagedKBuf;
        private float[][][] _tpPagedVBuf;
        private int _tpPagedNumBlocks;
        private int _tpPagedBlockSize;

        private IReadOnlyList<float[]> ForwardBatchTP(BatchedForwardContext ctx)
        {
            int numSeqs = ctx.Sequences.Count;
            if (numSeqs == 0) return Array.Empty<float[]>();

            int tp = TpDegree;
            int hidden = Config.HiddenSize;
            int numHeads = Config.NumHeads;
            int numKvHeads = Config.NumKVHeads;
            int headDim = _attnKeyLen;
            int numHeadsPerGpu = numHeads / tp;
            int numKvHeadsPerGpu = numKvHeads / tp;
            int qDimPerGpu = numHeadsPerGpu * headDim;
            int kDimPerGpu = numKvHeadsPerGpu * headDim;
            float scale = 1.0f / MathF.Sqrt(headDim);

            // Resolve paged-buffer dimensions.
            int blockSize = ctx.Sequences[0].BlockTable.BlockSize;
            int maxBlockId = 0;
            for (int s = 0; s < numSeqs; s++)
            {
                var bt = ctx.BlockTables[s];
                for (int b = 0; b < bt.Length; b++)
                    if (bt[b] > maxBlockId) maxBlockId = bt[b];
            }
            EnsureTpPagedBuffersAllocated(maxBlockId + 1, blockSize, numKvHeadsPerGpu, headDim, tp);

            int numTokens = 0;
            for (int s = 0; s < numSeqs; s++) numTokens += ctx.NumScheduledTokens[s];

            int[] positions = ctx.Positions.ToArray();
            int[] queryStartLoc = ctx.QueryStartLoc.ToArray();
            int[] slotMapping = ctx.SlotMapping.ToArray();
            int[] seqLens = new int[numSeqs];
            for (int s = 0; s < numSeqs; s++)
                seqLens[s] = ctx.Sequences[s].NumComputedTokens + ctx.NumScheduledTokens[s];

            // Concatenate input tokens. The executor passes the batch's input
            // tokens explicitly: decode steps forward the sampled-but-not-yet-
            // committed token, which is absent from the sequence's token list
            // (TokenAt would throw).
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

            // Embed + vision injection on rank 0, then broadcast.
            Tensor hidden0 = Embedding(flatTokens);
            if (_pendingVisionEmbeddingsList.Count > 0)
            {
                int visionStartPos = numSeqs > 0 ? ctx.Sequences[0].NumComputedTokens : 0;
                foreach (var (embeddings, position) in _pendingVisionEmbeddingsList)
                {
                    InjectVisionEmbeddings(hidden0, embeddings, position, visionStartPos);
                    embeddings.Dispose();
                }
                _pendingVisionEmbeddingsList.Clear();
            }
            Tensor[] hiddenStates = BroadcastTensorToAllRanks(hidden0);

            // Per-(token,head) positions for RoPE (reduced head counts).
            using var positionsTensorQ = BuildRoPEPositionsTensor(positions, numHeadsPerGpu);
            using var positionsTensorK = BuildRoPEPositionsTensor(positions, numKvHeadsPerGpu);

            // Per-layer transformer.
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                string[] wn = _layerWeightNames[layer];
                bool fused = _layerQkvFused[layer];
                int outIdx = fused ? 2 : 4;
                int ffnNormIdx = fused ? 3 : 5;
                int gateUpIdx = fused ? 4 : 6;
                int downIdx = fused ? 5 : 7;

                // 1. Attention norm (replicated).
                Tensor[] normed = TpRMSNorm(hiddenStates, wn[0]);

                // 2. Column-parallel QKV + per-rank attention.
                Tensor[] attnOut;
                if (fused)
                {
                    Tensor[] qkvFused = TpColumnParallelLinear(normed[0], wn[1]);
                    for (int r = 0; r < tp; r++)
                        normed[r].Dispose();

                    attnOut = BatchedAttentionTPFused(
                        qkvFused, layer, numTokens, numHeadsPerGpu, numKvHeadsPerGpu,
                        headDim, qDimPerGpu, kDimPerGpu, scale,
                        positionsTensorQ, positionsTensorK, positions,
                        queryStartLoc, seqLens, slotMapping, ctx.BlockTables, numSeqs);
                }
                else
                {
                    Tensor[] qProj = TpColumnParallelLinear(normed[0], wn[1]);
                    Tensor[] kProj = TpColumnParallelLinear(normed[0], wn[2]);
                    Tensor[] vProj = TpColumnParallelLinear(normed[0], wn[3]);
                    for (int r = 0; r < tp; r++)
                        normed[r].Dispose();

                    attnOut = BatchedAttentionTPSeparate(
                        qProj, kProj, vProj, layer, numTokens,
                        numHeadsPerGpu, numKvHeadsPerGpu, headDim,
                        qDimPerGpu, kDimPerGpu, scale,
                        positionsTensorQ, positionsTensorK, positions,
                        queryStartLoc, seqLens, slotMapping, ctx.BlockTables, numSeqs);
                }

                // 3. Row-parallel output projection + AllReduce.
                Tensor reducedAttn = TpRowParallelLinear(attnOut, wn[outIdx]);
                for (int r = 0; r < tp; r++)
                    attnOut[r].Dispose();

                // 4. Residual add.
                Tensor[] attnReplicated = BroadcastTensorToAllRanks(reducedAttn);
                TpResidualAdd(hiddenStates, attnReplicated);
                for (int r = 1; r < tp; r++)
                    attnReplicated[r].Dispose();
                reducedAttn.Dispose();

                // 5. FFN norm (replicated).
                Tensor[] normed2 = TpRMSNorm(hiddenStates, wn[ffnNormIdx]);

                // 6. Column-parallel gate/up.
                Tensor[] gateUp = TpColumnParallelLinear(normed2[0], wn[gateUpIdx]);
                for (int r = 0; r < tp; r++)
                    normed2[r].Dispose();

                // 7. Per-GPU SiLU·mul.
                int halfDim = (int)(gateUp[0].Sizes[1] / 2);
                Tensor[] gateResults = new Tensor[tp];
                for (int r = 0; r < tp; r++)
                {
                    using var gView = gateUp[r].Narrow(1, 0, halfDim);
                    Tensor gate = Ops.NewContiguous(gView);
                    using var uView = gateUp[r].Narrow(1, halfDim, halfDim);
                    Tensor up = Ops.NewContiguous(uView);
                    gateUp[r].Dispose();

                    Ops.SiLUMul(gate, gate, up);
                    up.Dispose();
                    gateResults[r] = gate;
                }

                // 8. Row-parallel down + AllReduce.
                Tensor ffnOut = TpRowParallelLinear(gateResults, wn[downIdx]);
                for (int r = 0; r < tp; r++)
                    gateResults[r].Dispose();

                // 9. Residual add.
                Tensor[] ffnReplicated = BroadcastTensorToAllRanks(ffnOut);
                TpResidualAdd(hiddenStates, ffnReplicated);
                for (int r = 1; r < tp; r++)
                    ffnReplicated[r].Dispose();
                ffnOut.Dispose();
            }

            // Final norm + LM head on rank 0.
            Tensor finalNormed = RMSNormOp(hiddenStates[0], "output_norm.weight");
            for (int r = 0; r < tp; r++)
                hiddenStates[r].Dispose();

            float[] finalFlat = finalNormed.GetElementsAsFloat(numTokens * hidden);
            finalNormed.Dispose();
            float[] lastTokensPacked = PagedKvBatchOps.GatherLastTokenPerSeq(
                finalFlat, hidden, queryStartLoc, numSeqs);
            Tensor lastHidden = CreateFloatTensor(lastTokensPacked, numSeqs, hidden);

            Tensor logitsTensor = LinearForward(lastHidden, "output.weight");
            if (logitsTensor == null)
                logitsTensor = LinearForward(lastHidden, "token_embd.weight");
            lastHidden.Dispose();

            float[] allLogits = logitsTensor.GetElementsAsFloat(numSeqs * Config.VocabSize);
            logitsTensor.Dispose();

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

        private Tensor[] BatchedAttentionTPFused(
            Tensor[] qkvFused, int layer, int numTokens,
            int numHeadsPerGpu, int numKvHeadsPerGpu, int headDim,
            int qDimPerGpu, int kDimPerGpu, float scale,
            Tensor positionsTensorQ, Tensor positionsTensorK, int[] positions,
            int[] queryStartLoc, int[] seqLens, int[] slotMapping,
            int[][] blockTables, int numSeqs)
        {
            int tp = TpDegree;
            var results = new Tensor[tp];

            for (int r = 0; r < tp; r++)
            {
                var alloc = _tpGroup.GetAllocator(r);

                using var qView = qkvFused[r].Narrow(1, 0, qDimPerGpu);
                Tensor q = Ops.NewContiguous(qView);
                using var kView = qkvFused[r].Narrow(1, qDimPerGpu, kDimPerGpu);
                Tensor k = Ops.NewContiguous(kView);
                using var vView = qkvFused[r].Narrow(1, qDimPerGpu + kDimPerGpu, kDimPerGpu);
                Tensor v = Ops.NewContiguous(vView);
                qkvFused[r].Dispose();

                results[r] = RunBatchedAttentionPerGpu(
                    q, k, v, alloc, r, layer, numTokens,
                    numHeadsPerGpu, numKvHeadsPerGpu, headDim,
                    qDimPerGpu, kDimPerGpu, scale,
                    positionsTensorQ, positionsTensorK, positions,
                    queryStartLoc, seqLens, slotMapping, blockTables, numSeqs);
            }

            return results;
        }

        private Tensor[] BatchedAttentionTPSeparate(
            Tensor[] qProj, Tensor[] kProj, Tensor[] vProj, int layer,
            int numTokens, int numHeadsPerGpu, int numKvHeadsPerGpu,
            int headDim, int qDimPerGpu, int kDimPerGpu, float scale,
            Tensor positionsTensorQ, Tensor positionsTensorK, int[] positions,
            int[] queryStartLoc, int[] seqLens, int[] slotMapping,
            int[][] blockTables, int numSeqs)
        {
            int tp = TpDegree;
            var results = new Tensor[tp];

            for (int r = 0; r < tp; r++)
            {
                var alloc = _tpGroup.GetAllocator(r);
                results[r] = RunBatchedAttentionPerGpu(
                    qProj[r], kProj[r], vProj[r], alloc, r, layer, numTokens,
                    numHeadsPerGpu, numKvHeadsPerGpu, headDim,
                    qDimPerGpu, kDimPerGpu, scale,
                    positionsTensorQ, positionsTensorK, positions,
                    queryStartLoc, seqLens, slotMapping, blockTables, numSeqs);
            }

            return results;
        }

        private Tensor RunBatchedAttentionPerGpu(
            Tensor q, Tensor k, Tensor v, IAllocator alloc, int rank,
            int layer, int numTokens, int numHeadsPerGpu, int numKvHeadsPerGpu,
            int headDim, int qDimPerGpu, int kDimPerGpu, float scale,
            Tensor positionsTensorQ, Tensor positionsTensorK, int[] positions,
            int[] queryStartLoc, int[] seqLens, int[] slotMapping,
            int[][] blockTables, int numSeqs)
        {
            // RoPE with YaRN parameters.
            q = ApplyBatchedRoPETP(q, positionsTensorQ, numTokens, numHeadsPerGpu, headDim, rank);
            k = ApplyBatchedRoPETP(k, positionsTensorK, numTokens, numKvHeadsPerGpu, headDim, rank);

            // YaRN position-dependent Q scaling.
            if (_ropeOrigCtx > 0)
                ApplyBatchedPositionScale(q, positions, numTokens, qDimPerGpu);

            // Scatter K/V into this rank's paged buffer.
            float[] kFlat = k.GetElementsAsFloat(numTokens * kDimPerGpu);
            float[] vFlat = v.GetElementsAsFloat(numTokens * kDimPerGpu);
            k.Dispose();
            v.Dispose();

            PagedKvBatchOps.ScatterKv(
                kFlat, vFlat, _tpPagedKBuf[rank][layer], _tpPagedVBuf[rank][layer],
                slotMapping, numTokens, numKvHeadsPerGpu, headDim, _tpPagedBlockSize);

            // Per-sequence paged attention on this rank.
            float[] qFlat = q.GetElementsAsFloat(numTokens * qDimPerGpu);
            q.Dispose();
            float[] attnFlat = new float[numTokens * qDimPerGpu];
            ManagedPagedAttention.Forward(
                qFlat, _tpPagedKBuf[rank][layer], _tpPagedVBuf[rank][layer], attnFlat,
                numTokens, numHeadsPerGpu, numKvHeadsPerGpu, headDim,
                _tpPagedBlockSize,
                queryStartLoc, seqLens, positions, blockTables, numSeqs,
                scale, causal: true);

            var result = new Tensor(alloc, DType.Float32, numTokens, qDimPerGpu);
            result.SetElementsAsFloat(attnFlat);
            return result;
        }

        private Tensor ApplyBatchedRoPETP(Tensor data, Tensor positionsTensor, int numTokens, int numHeads, int headDim, int rank)
        {
            Tensor posLocal = ReplicateTensorToRank(positionsTensor, rank);

            using var reshaped = data.View(1, numTokens, numHeads, headDim);
            Tensor result = Ops.RoPEEx(
                null, reshaped, posLocal, _ropeDim, 0, _ropeOrigCtx,
                Config.RopeBase, 1.0f / Config.RopeScale,
                _ropeType == "yarn" ? _ropeExtFactor : 0f,
                ComputeAttnFactor(),
                _ropeType == "yarn" ? _ropeBetaFast : 0f,
                _ropeType == "yarn" ? _ropeBetaSlow : 0f);
            data.Dispose();
            if (!ReferenceEquals(posLocal, positionsTensor)) posLocal.Dispose();

            Tensor flat = result.View(numTokens, numHeads * headDim);
            result.Dispose();
            return flat;
        }

        private void EnsureTpPagedBuffersAllocated(int numBlocks, int blockSize, int numKvHeadsPerGpu, int headDim, int tp)
        {
            if (_tpPagedKBuf != null && _tpPagedNumBlocks >= numBlocks && _tpPagedBlockSize == blockSize)
                return;

            int targetBlocks = Math.Max(numBlocks, _tpPagedNumBlocks * 2);
            int numLayers = Config.NumLayers;

            float[][][] oldK = _tpPagedKBuf;
            float[][][] oldV = _tpPagedVBuf;
            int oldNumBlocks = _tpPagedNumBlocks;
            int oldBlockSize = _tpPagedBlockSize;

            _tpPagedKBuf = new float[tp][][];
            _tpPagedVBuf = new float[tp][][];
            for (int r = 0; r < tp; r++)
            {
                _tpPagedKBuf[r] = new float[numLayers][];
                _tpPagedVBuf[r] = new float[numLayers][];
                for (int l = 0; l < numLayers; l++)
                {
                    _tpPagedKBuf[r][l] = PagedKvBatchOps.AllocateLayerBuffer(targetBlocks, blockSize, numKvHeadsPerGpu, headDim);
                    _tpPagedVBuf[r][l] = PagedKvBatchOps.AllocateLayerBuffer(targetBlocks, blockSize, numKvHeadsPerGpu, headDim);
                    if (oldK != null && oldK[r][l] != null && oldBlockSize == blockSize)
                    {
                        long copyLen = Math.Min(
                            (long)oldNumBlocks * blockSize * numKvHeadsPerGpu * headDim,
                            oldK[r][l].LongLength);
                        Array.Copy(oldK[r][l], _tpPagedKBuf[r][l], copyLen);
                        Array.Copy(oldV[r][l], _tpPagedVBuf[r][l], copyLen);
                    }
                }
            }
            _tpPagedNumBlocks = targetBlocks;
            _tpPagedBlockSize = blockSize;
        }
    }
}
