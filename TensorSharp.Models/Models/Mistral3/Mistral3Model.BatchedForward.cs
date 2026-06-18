// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Batched paged-attention forward path for Mistral 3. Second reference
// implementation of <see cref="IBatchedPagedModel"/>, after Qwen3.
//
// Differences from the Qwen3 template:
//   * Mistral 3 has no Q-norm / K-norm.
//   * RoPE uses YaRN scaling - extFactor / betaFast / betaSlow / attnFactor
//     are passed through to Ops.RoPEEx.
//   * Position-dependent Q scaling for YaRN: when _ropeOrigCtx > 0 each
//     query is scaled by (1 + beta * log(1 + floor(pos / orig_ctx))).
//   * Layers can be fused-QKV or unfused-QKV depending on the GGUF; the
//     batched path branches off the existing _layerQkvFused[] flag.
//   * Vision embeddings (Pixtral) are injected into the embedded hidden
//     state by position, before the per-layer loop - identical to the
//     legacy forward.
using System;
using System.Collections.Generic;
using TensorSharp;
using TensorSharp.Models.Paged;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models
{
    public partial class Mistral3Model : IBatchedPagedModel
    {
        // Per-layer paged K/V floats. Same shape as for Qwen3 - lazily
        // allocated once the engine's block-pool dimensions are known.
        private float[][] _pagedKBuf;
        private float[][] _pagedVBuf;
        private int _pagedNumBlocks;
        private int _pagedBlockSize;

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

            // Resolve paged-buffer dimensions from the per-sequence block
            // tables; allocate (or grow) the layer-private buffers.
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
            {
                var seq = ctx.Sequences[s];
                seqLens[s] = seq.NumComputedTokens + ctx.NumScheduledTokens[s];
            }

            // Concatenate input tokens.
            int[] flatTokens = new int[numTokens];
            int cursor = 0;
            for (int s = 0; s < numSeqs; s++)
            {
                var seq = ctx.Sequences[s];
                int startTok = seq.NumComputedTokens;
                int take = ctx.NumScheduledTokens[s];
                for (int i = 0; i < take; i++)
                    flatTokens[cursor++] = seq.TokenAt(startTok + i);
            }

            // Embed.
            Tensor hiddenStates = Embedding(flatTokens);
            // Vision injection: identical to the legacy path; multimodal
            // requests must be prepared serially upstream.
            if (_pendingVisionEmbeddingsList.Count > 0)
            {
                // Multimodal requests are prepared serially upstream, so the batch
                // holds a single sequence; its already-computed token count is the
                // chunk's absolute base used to report the absolute injection position.
                int visionStartPos = numSeqs > 0 ? ctx.Sequences[0].NumComputedTokens : 0;
                foreach (var (embeddings, position) in _pendingVisionEmbeddingsList)
                {
                    InjectVisionEmbeddings(hiddenStates, embeddings, position, visionStartPos);
                    embeddings.Dispose();
                }
                _pendingVisionEmbeddingsList.Clear();
            }

            // Per-(token,head) positions for RoPE.
            using var positionsTensorQ = BuildRoPEPositionsTensor(positions, numHeads);
            using var positionsTensorK = BuildRoPEPositionsTensor(positions, numKvHeads);

            // Per-layer transformer.
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                string[] wn = _layerWeightNames[layer];
                bool fused = _layerQkvFused[layer];
                int normIdx = 0;
                int outIdx = fused ? 2 : 4;
                int ffnNormIdx = fused ? 3 : 5;
                int gateUpIdx = fused ? 4 : 6;
                int downIdx = fused ? 5 : 7;

                Tensor residual = Ops.NewContiguous(hiddenStates);
                Tensor normed = RMSNormOp(hiddenStates, wn[normIdx]);
                hiddenStates.Dispose();

                // Q, K, V.
                Tensor q, k, v;
                if (fused)
                {
                    Tensor qkv = LinearForward(normed, wn[1]);
                    normed.Dispose();
                    using (var qView = qkv.Narrow(1, 0, qDim))
                        q = Ops.NewContiguous(qView);
                    using (var kView = qkv.Narrow(1, qDim, kDim))
                        k = Ops.NewContiguous(kView);
                    using (var vView = qkv.Narrow(1, qDim + kDim, kDim))
                        v = Ops.NewContiguous(vView);
                    qkv.Dispose();
                }
                else
                {
                    q = LinearForward(normed, wn[1]);
                    k = LinearForward(normed, wn[2]);
                    v = LinearForward(normed, wn[3]);
                    normed.Dispose();
                }

                // RoPE with per-token positions + Mistral's YaRN params.
                q = ApplyBatchedRoPE(q, positionsTensorQ, numTokens, numHeads, headDim);
                k = ApplyBatchedRoPE(k, positionsTensorK, numTokens, numKvHeads, headDim);

                // YaRN position-dependent Q scaling (one factor per token,
                // shared across heads, scales the WHOLE row by a scalar).
                if (_ropeOrigCtx > 0)
                    ApplyBatchedPositionScale(q, positions, numTokens, qDim);

                // Scatter new K, V into the paged buffer.
                float[] kFlat = k.GetElementsAsFloat(numTokens * kDim);
                float[] vFlat = v.GetElementsAsFloat(numTokens * kDim);
                PagedKvBatchOps.ScatterKv(
                    kFlat, vFlat, _pagedKBuf[layer], _pagedVBuf[layer],
                    slotMapping, numTokens, numKvHeads, headDim, _pagedBlockSize);
                k.Dispose();
                v.Dispose();

                // Per-sequence paged attention. Three kernels available:
                //   - native (default on GGML backends): C++ gather + ggml_flash_attn_ext
                //   - tensor: C# tensor-based gather + Ops.AddmmBatch + AttentionSoftmaxWithSinks
                //   - managed: pure-C# online-softmax scalar fallback
                //
                // Choose via TS_PAGED_ATTN_KERNEL={native|tensor|managed}.
                float[] qFlat = q.GetElementsAsFloat(numTokens * qDim);
                q.Dispose();
                float[] attnFlat = new float[numTokens * qDim];
                var kernel = ResolvePagedAttentionKernel();
                if (kernel == PagedAttentionKernel.Native && IsGgmlBackend)
                {
                    // Build the concatenated block-table layout the native
                    // entry point expects (the engine gives us int[][]).
                    var (blockTableFlat, blockTableOffsets) = FlattenBlockTables(ctx.BlockTables);
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
                Tensor attnProj = LinearForward(attnOut, wn[outIdx]);
                attnOut.Dispose();

                Ops.Add(residual, residual, attnProj);
                attnProj.Dispose();

                // Fused dense SwiGLU FFN (norm + gate/up + SiLU·mul + down + residual)
                // in one GGML graph, keeping the large intermediate on-device. Falls
                // back to the unfused chain when the backend/weights don't qualify.
                if (!TryFusedDenseSwiGLUFFNInto(residual, wn[ffnNormIdx], wn[gateUpIdx], wn[downIdx]))
                {
                    Tensor ffnNormed = RMSNormOp(residual, wn[ffnNormIdx]);
                    Tensor ffnOut = FFN(ffnNormed, wn[gateUpIdx], wn[downIdx], numTokens);
                    ffnNormed.Dispose();

                    Ops.Add(residual, residual, ffnOut);
                    ffnOut.Dispose();
                }
                hiddenStates = residual;
            }

            // Final norm + per-sequence LM head.
            Tensor finalNormed = RMSNormOp(hiddenStates, "output_norm.weight");
            hiddenStates.Dispose();

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

        /// <summary>
        /// Per-(token,head) RoPE through <see cref="Ops.RoPEEx"/>, passing the
        /// YaRN scaling parameters that Mistral 3 GGUFs carry.
        /// </summary>
        private Tensor ApplyBatchedRoPE(Tensor data, Tensor positionsTensor, int numTokens, int numHeads, int headDim)
        {
            using var reshaped = data.View(1, numTokens, numHeads, headDim);
            Tensor result = Ops.RoPEEx(
                null, reshaped, positionsTensor, _ropeDim, 0, _ropeOrigCtx,
                Config.RopeBase, 1.0f / Config.RopeScale,
                _ropeType == "yarn" ? _ropeExtFactor : 0f,
                ComputeAttnFactor(),
                _ropeType == "yarn" ? _ropeBetaFast : 0f,
                _ropeType == "yarn" ? _ropeBetaSlow : 0f);
            data.Dispose();
            Tensor flat = result.View(numTokens, numHeads * headDim);
            result.Dispose();
            return flat;
        }

        /// <summary>
        /// Per-token YaRN position-dependent Q scaling. The legacy single-
        /// sequence path scales each row using <c>position = startPos + s</c>;
        /// in the batched path each token carries its own absolute position
        /// via <paramref name="positions"/>.
        /// </summary>
        private unsafe void ApplyBatchedPositionScale(Tensor qTensor, int[] positions, int numTokens, int qDim)
        {
            float* ptr = GetFloatPtr(qTensor);
            for (int t = 0; t < numTokens; t++)
            {
                int pos = positions[t];
                float interval = MathF.Floor((float)pos / _ropeOrigCtx);
                float posScale = 1.0f + _ropeScalingBeta * MathF.Log(1.0f + interval);
                if (MathF.Abs(posScale - 1.0f) < 1e-7f) continue;
                VecScale(ptr + (long)t * qDim, posScale, qDim);
            }
        }

        private enum PagedAttentionKernel { Native, Tensor, Managed }

        private static PagedAttentionKernel ResolvePagedAttentionKernel()
        {
            string raw = Environment.GetEnvironmentVariable("TS_PAGED_ATTN_KERNEL");
            if (string.IsNullOrEmpty(raw)) return PagedAttentionKernel.Native; // default
            switch (raw.Trim().ToLowerInvariant())
            {
                case "native":
                case "ggml":
                case "flash":
                    return PagedAttentionKernel.Native;
                case "tensor":
                case "gpu":
                case "addmm":
                    return PagedAttentionKernel.Tensor;
                case "managed":
                case "scalar":
                case "0":
                case "false":
                    return PagedAttentionKernel.Managed;
                default:
                    return PagedAttentionKernel.Native;
            }
        }

        // Native paged-attention entry point expects per-seq block tables in
        // one concatenated int[] with per-seq offsets. Keep the engine's
        // int[][] abstraction unchanged; flatten right at the call site.
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

        private void EnsurePagedBuffersAllocated(int numBlocks, int blockSize, int numKvHeads, int headDim)
        {
            if (_pagedKBuf != null && _pagedNumBlocks >= numBlocks && _pagedBlockSize == blockSize)
                return;
            int targetBlocks = Math.Max(numBlocks, _pagedNumBlocks * 2);
            int numLayers = Config.NumLayers;
            // CRITICAL: preserve existing K/V data when growing. The engine
            // submits sequences over multiple scheduler steps; the first
            // step may schedule one sequence (allocating a small buffer
            // here), and a later step may add more sequences (requiring
            // more blocks). The new arrivals' decode tokens still depend
            // on the FIRST sequence's prefill K/V being in the buffer.
            // Rebuilding without copy zeroes that K/V and the first
            // sequence's continuation generates garbage tokens.
            float[][] oldK = _pagedKBuf;
            float[][] oldV = _pagedVBuf;
            int oldNumBlocks = _pagedNumBlocks;
            int oldBlockSize = _pagedBlockSize;

            _pagedKBuf = new float[numLayers][];
            _pagedVBuf = new float[numLayers][];
            for (int l = 0; l < numLayers; l++)
            {
                _pagedKBuf[l] = PagedKvBatchOps.AllocateLayerBuffer(targetBlocks, blockSize, numKvHeads, headDim);
                _pagedVBuf[l] = PagedKvBatchOps.AllocateLayerBuffer(targetBlocks, blockSize, numKvHeads, headDim);
                if (oldK != null && oldK[l] != null && oldBlockSize == blockSize)
                {
                    long copyLen = Math.Min(
                        (long)oldNumBlocks * blockSize * numKvHeads * headDim,
                        oldK[l].LongLength);
                    Array.Copy(oldK[l], _pagedKBuf[l], copyLen);
                    Array.Copy(oldV[l], _pagedVBuf[l], copyLen);
                }
            }
            _pagedNumBlocks = targetBlocks;
            _pagedBlockSize = blockSize;
        }
    }
}
