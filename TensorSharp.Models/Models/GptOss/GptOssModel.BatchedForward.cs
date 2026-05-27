// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// GptOss batched paged-attention forward — vLLM-style continuous batching.
// Mirrors the Mistral 3 / Nemotron 3 batched templates with GptOss-specific
// pieces:
//   - Fused QKV (or three split projections) WITH bias (Q/K/V/O all carry bias)
//   - NeoX-style RoPE with YaRN scaling
//   - GQA (numKVHeads < numHeads)
//   - Per-layer SWA: layer % 2 == 0 → sliding window, else full causal
//   - Per-head attention sinks (a learned scalar logit per head per layer;
//     participates in softmax denominator but contributes zero V — bleeds
//     attention mass when no real position is strongly relevant)
//   - MoE FFN with SwiGLU-clamped activation (handled by MoEForward; the
//     existing batched-MoE path naturally takes [numTokens, hidden] input)
//
// Coverage in Phase 1 of this port:
//   - IBatchedPagedModel
//   - Per-layer paged K/V buffers (single kvDim — GptOss doesn't vary per layer)
//   - Batched attention with sinks via ManagedPagedAttention.ForwardWithSinks
//   - Batched MoE via MoEForward(numTokens) (matches Nemotron Phase 7 pattern)
//   - Opt-in via TS_GPTOSS_BATCHED env var (default OFF)
using System;
using System.Collections.Generic;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models.Paged;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models
{
    public partial class GptOssModel : IBatchedPagedModel
    {
        // Default ON. The batched paged-attention path is the only way two
        // concurrent requests can be served truly in parallel on this model
        // (the per-sequence fallback forwards at most one sequence per step,
        // so a second request stalls until the first releases the executor).
        // Correctness was previously validated against the legacy path by
        // GptOssBatchedCorrectnessTests with TS_GPTOSS_BATCHED=1. Set
        // TS_GPTOSS_BATCHED=0 (or "false") to force the legacy fallback for
        // A/B comparison or to investigate a regression.
        //
        // Method getter (not static readonly) so tests can toggle after class
        // load — a static readonly would capture the env var at class-init
        // time, which is before tests get a chance to set it (same gotcha
        // that bit Nemotron). Mirrors Qwen 3.5's pattern.
        private static bool GptOssBatchedOptIn()
        {
            string raw = Environment.GetEnvironmentVariable("TS_GPTOSS_BATCHED");
            if (string.IsNullOrEmpty(raw)) return true;
            return raw != "0" && !string.Equals(raw, "false", StringComparison.OrdinalIgnoreCase);
        }

        // GptOss is text-only (no vision/audio), so SupportsBatchedMultimodal
        // is a non-issue — set to match the opt-in for symmetry with the other
        // batched models.
        public bool SupportsBatchedMultimodal => GptOssBatchedOptIn();

        // Per-layer paged K/V buffers (vLLM block layout:
        // [numBlocks * blockSize * numKvHeads * headDim] per layer).
        private float[][] _gptOssPagedK;
        private float[][] _gptOssPagedV;
        private int _gptOssPagedNumBlocks;
        private int _gptOssPagedBlockSize;

        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx)
        {
            if (ctx == null) throw new ArgumentNullException(nameof(ctx));
            int numSeqs = ctx.Sequences.Count;
            if (numSeqs == 0) return Array.Empty<float[]>();

            if (!GptOssBatchedOptIn())
                throw new NotSupportedException(
                    "GptOss batched: disabled (TS_GPTOSS_BATCHED=0 set, falling back to per-seq path).");

            int hidden = Config.HiddenSize;
            int numHeads = Config.NumHeads;
            int numKvHeads = Config.NumKVHeads;
            int headDim = Config.HeadDim;
            int qDim = _qDim;
            int kvDim = _kDim;
            float scale = 1.0f / MathF.Sqrt(headDim);

            // Resolve paged-buffer dimensions + allocate.
            int blockSize = ctx.Sequences[0].BlockTable.BlockSize;
            int maxBlockId = 0;
            for (int s = 0; s < numSeqs; s++)
            {
                var bt = ctx.BlockTables[s];
                for (int b = 0; b < bt.Length; b++)
                    if (bt[b] > maxBlockId) maxBlockId = bt[b];
            }
            EnsureGptOssPagedBuffers(maxBlockId + 1, blockSize);

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

            // Flatten tokens + run shared embedding.
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
            Tensor hiddenStates = Embedding(flatTokens);

            // Per-(token, head) position tables for RoPE (NeoX, with YaRN).
            using var positionsTensorQ = BuildBatchedRoPEPositions(positions, numHeads);
            using var positionsTensorK = BuildBatchedRoPEPositions(positions, numKvHeads);

            int numLayers = Config.NumLayers;
            for (int layer = 0; layer < numLayers; layer++)
            {
                hiddenStates = RunBatchedAttentionLayer(
                    hiddenStates, layer, numTokens, numSeqs,
                    queryStartLoc, slotMapping, seqLens, positions,
                    ctx.BlockTables, numHeads, numKvHeads, headDim,
                    qDim, kvDim, scale, positionsTensorQ, positionsTensorK);

                // Post-attention norm + MoE FFN + residual. MoEForward already
                // accepts a batched [seqLen, hidden] tensor — the per-token
                // loop inside it picks experts row-by-row, so passing
                // seqLen=numTokens treats every token independently (whether
                // they come from one sequence or many is irrelevant for MoE).
                Tensor residual = Ops.NewContiguous(hiddenStates);
                Tensor normed = RMSNormOp(hiddenStates, _layerNames[layer][5]);
                hiddenStates.Dispose();

                Tensor moeOut = MoEForward(normed, layer, numTokens);
                normed.Dispose();

                Ops.Add(residual, residual, moeOut);
                moeOut.Dispose();
                hiddenStates = residual;
            }

            // Final norm + per-sequence LM head.
            Tensor finalNormed = RMSNormOp(hiddenStates, "output_norm.weight");
            hiddenStates.Dispose();

            float[] finalFlat = finalNormed.GetElementsAsFloat(numTokens * hidden);
            finalNormed.Dispose();
            float[] lastTokensPacked = PagedKvBatchOps.GatherLastTokenPerSeq(
                finalFlat, hidden, queryStartLoc, numSeqs);
            using Tensor lastHidden = CreateFloatTensor(lastTokensPacked, numSeqs, hidden);

            Tensor logitsTensor = LinearForward(lastHidden, "output.weight");
            if (logitsTensor == null)
                logitsTensor = LinearForward(lastHidden, "token_embd.weight");
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

        private Tensor RunBatchedAttentionLayer(
            Tensor hiddenStates, int layer, int numTokens, int numSeqs,
            int[] queryStartLoc, int[] slotMapping, int[] seqLens, int[] positions,
            int[][] blockTables, int numHeads, int numKvHeads, int headDim,
            int qDim, int kvDim, float scale,
            Tensor positionsTensorQ, Tensor positionsTensorK)
        {
            string[] wn = _layerNames[layer];
            bool isSWA = (layer % 2 == 0);
            int slidingWindow = isSWA ? _slidingWindow : 0;
            float[] sinks = _layerSinks[layer];

            // Pre-attention norm + residual snapshot.
            Tensor residual = Ops.NewContiguous(hiddenStates);
            Tensor normed = RMSNormOp(hiddenStates, wn[0]);
            hiddenStates.Dispose();

            // Q/K/V projection with bias.
            Tensor qTensor, kTensor, vTensor;
            if (_isQkvFused)
            {
                Tensor qkvFused = LinearForwardWithBias(normed, wn[1], wn[2]);
                using (var qView = qkvFused.Narrow(1, 0, qDim))
                    qTensor = Ops.NewContiguous(qView);
                using (var kView = qkvFused.Narrow(1, qDim, kvDim))
                    kTensor = Ops.NewContiguous(kView);
                using (var vView = qkvFused.Narrow(1, qDim + kvDim, kvDim))
                    vTensor = Ops.NewContiguous(vView);
                qkvFused.Dispose();
            }
            else
            {
                qTensor = LinearForwardWithBias(normed, wn[1], wn[2]);
                kTensor = LinearForwardWithBias(normed, wn[8], wn[9]);
                vTensor = LinearForwardWithBias(normed, wn[10], wn[11]);
            }
            normed.Dispose();

            // NeoX RoPE + YaRN with per-token positions (passes the same YaRN
            // params the legacy ApplyRoPEInPlace uses).
            qTensor = ApplyBatchedRoPE(qTensor, positionsTensorQ, numTokens, numHeads, headDim);
            kTensor = ApplyBatchedRoPE(kTensor, positionsTensorK, numTokens, numKvHeads, headDim);

            // Scatter K/V into paged buffers via slotMapping.
            float[] kFlat = kTensor.GetElementsAsFloat(numTokens * kvDim);
            float[] vFlat = vTensor.GetElementsAsFloat(numTokens * kvDim);
            PagedKvBatchOps.ScatterKv(
                kFlat, vFlat, _gptOssPagedK[layer], _gptOssPagedV[layer],
                slotMapping, numTokens, numKvHeads, headDim, _gptOssPagedBlockSize);
            kTensor.Dispose();
            vTensor.Dispose();

            // Paged attention with sinks. Phase 9: prefer the native kernel
            // (TSGgml_PagedAttentionForwardWithSinks → ggml_flash_attn_ext +
            // ggml_flash_attn_ext_add_sinks on Metal/CUDA) when the GGML
            // backend is active. Falls back to managed-C# online softmax with
            // sinks on non-GGML backends (or by setting
            // TS_GPTOSS_PAGED_ATTN_MANAGED=1 for A/B testing).
            float[] qFlat = qTensor.GetElementsAsFloat(numTokens * qDim);
            qTensor.Dispose();
            float[] attnFlat = new float[numTokens * qDim];
            bool useNative = IsGgmlBackend
                && !string.Equals(
                    Environment.GetEnvironmentVariable("TS_GPTOSS_PAGED_ATTN_MANAGED"),
                    "1", StringComparison.Ordinal);
            if (useNative)
            {
                var (blockTableFlat, blockTableOffsets) = FlattenBlockTables(blockTables);
                TensorSharp.GGML.GgmlBasicOps.PagedAttentionForwardWithSinks(
                    qFlat, _gptOssPagedK[layer], _gptOssPagedV[layer], attnFlat,
                    queryStartLoc, seqLens, positions,
                    blockTableFlat, blockTableOffsets,
                    numSeqs, numTokens, numHeads, numKvHeads, headDim,
                    _gptOssPagedBlockSize, scale, slidingWindow, sinks);
            }
            else
            {
                ManagedPagedAttention.ForwardWithSinks(
                    qFlat, _gptOssPagedK[layer], _gptOssPagedV[layer], attnFlat,
                    numTokens, numHeads, numKvHeads, headDim, _gptOssPagedBlockSize,
                    queryStartLoc, seqLens, positions, blockTables, numSeqs,
                    scale, sinks, slidingWindow);
            }

            // Output projection with bias + residual add.
            using Tensor attnOut = CreateFloatTensor(attnFlat, numTokens, qDim);
            Tensor attnProj = LinearForwardWithBias(attnOut, wn[3], wn[4]);
            Ops.Add(residual, residual, attnProj);
            attnProj.Dispose();
            return residual;
        }

        private Tensor ApplyBatchedRoPE(Tensor data, Tensor positionsTensor,
            int numTokens, int numHeads, int headDim)
        {
            using var reshaped = data.View(1, numTokens, numHeads, headDim);
            // YaRN params match the legacy ApplyRoPEInPlace: nDims=headDim,
            // mode=2 (NeoX), origCtx=Config.OriginalContextLength, scale=1/RopeScale,
            // extFactor=1, attnFactor=1, betaFast=32, betaSlow=1.
            Tensor result = Ops.RoPEEx(
                null, reshaped, positionsTensor, headDim, 2,
                Config.OriginalContextLength,
                Config.RopeBase, 1.0f / Config.RopeScale,
                1.0f, 1.0f, 32.0f, 1.0f);
            data.Dispose();
            Tensor flat = result.View(numTokens, numHeads * headDim);
            result.Dispose();
            return flat;
        }

        private Tensor BuildBatchedRoPEPositions(int[] tokenPositions, int numHeads)
        {
            int total = tokenPositions.Length * numHeads;
            int[] expanded = new int[total];
            for (int t = 0; t < tokenPositions.Length; t++)
                for (int h = 0; h < numHeads; h++)
                    expanded[t * numHeads + h] = tokenPositions[t];
            return CreateIntTensor(expanded, total);
        }

        // Flatten per-seq block tables into concatenated int[] + offsets[]
        // for the native paged-attention entry point.
        private static (int[] flat, int[] offsets) FlattenBlockTables(int[][] perSeq)
        {
            int total = 0;
            int[] offsets = new int[perSeq.Length + 1];
            for (int s = 0; s < perSeq.Length; s++)
            {
                offsets[s] = total;
                total += perSeq[s].Length;
            }
            offsets[perSeq.Length] = total;
            int[] flat = new int[total];
            for (int s = 0; s < perSeq.Length; s++)
                Array.Copy(perSeq[s], 0, flat, offsets[s], perSeq[s].Length);
            return (flat, offsets);
        }

        /// <summary>True when the model can migrate a sequence's K/V history
        /// from the legacy linear cache (where <c>Forward()</c> writes it)
        /// into paged storage (where <c>ForwardBatch</c> reads from). Required
        /// so the N=1 fast path can hand off to the batched path when a
        /// second concurrent sequence arrives — without migration the batched
        /// kernel would read zeros for the first sequence's prior positions.
        ///
        /// Block-quantised caches (Q8_0) aren't supported by the migration
        /// code (only F32/F16 dequant is implemented); the batched path
        /// rejects them anyway.</summary>
        public bool SupportsLinearKVMigration =>
            _kvCacheK != null && _kvCacheV != null
            && !_kvCacheDtype.IsBlockQuantized();

        /// <summary>Copy <paramref name="owner"/>'s K/V history out of the
        /// linear per-layer cache <c>_kvCacheK</c>/<c>_kvCacheV</c> (layout
        /// <c>[numKVHeads, capacity, headDim]</c>) and into the paged buffer
        /// <c>_gptOssPagedK</c>/<c>_gptOssPagedV</c> at slot positions
        /// derived from <c>owner.BlockTable</c>. GptOss has no SWA wrap (SWA
        /// is applied as an attention mask rather than via a circular cache),
        /// so every position from 0..<c>_cacheSeqLen</c> is recoverable.</summary>
        public bool TryMigrateLinearKVToPaged(SequenceState owner, int blockSize)
        {
            if (owner == null) return false;
            if (!SupportsLinearKVMigration) return false;
            int ownerTokens = _cacheSeqLen;
            if (ownerTokens <= 0) return true;
            if (owner.BlockTable.NumBlocks <= 0) return false;

            // Flush any device-side cache writes to host so the float reads
            // below see the freshest K/V. GGML Metal kernels may keep the
            // cache hot in the Metal buffer while the host shadow is stale.
            if (IsGgmlBackend)
            {
                var seen = new HashSet<Storage>();
                for (int l = 0; l < Config.NumLayers; l++)
                {
                    if (_kvCacheK[l] != null && seen.Add(_kvCacheK[l].Storage))
                        SyncTensorHostCache(_kvCacheK[l]);
                    if (_kvCacheV[l] != null && seen.Add(_kvCacheV[l].Storage))
                        SyncTensorHostCache(_kvCacheV[l]);
                }
            }

            // Make sure paged buffers cover every block id we'll write into.
            int maxBlockId = 0;
            int numBlocks = owner.BlockTable.NumBlocks;
            for (int b = 0; b < numBlocks; b++)
            {
                int id = owner.BlockTable.Blocks[b].Id;
                if (id > maxBlockId) maxBlockId = id;
            }
            EnsureGptOssPagedBuffers(maxBlockId + 1, blockSize);

            int kvHeads = Config.NumKVHeads;
            int headDim = Config.HeadDim;
            int stridePaged = kvHeads * headDim;
            int numLayers = Config.NumLayers;

            for (int layer = 0; layer < numLayers; layer++)
            {
                int cacheLen = (int)_kvCacheK[layer].Sizes[1];
                int totalElems = kvHeads * cacheLen * headDim;
                if (!TryReadCacheAsF32(_kvCacheK[layer], totalElems, out float[] kFlat) ||
                    !TryReadCacheAsF32(_kvCacheV[layer], totalElems, out float[] vFlat))
                {
                    return false;
                }

                float[] kPaged = _gptOssPagedK[layer];
                float[] vPaged = _gptOssPagedV[layer];

                for (int p = 0; p < ownerTokens; p++)
                {
                    int blockIdx = p / blockSize;
                    int offsetInBlock = p % blockSize;
                    int physBlockId = owner.BlockTable.Blocks[blockIdx].Id;
                    int slot = physBlockId * blockSize + offsetInBlock;
                    int slotOffset = slot * stridePaged;

                    for (int h = 0; h < kvHeads; h++)
                    {
                        int srcOffset = (h * cacheLen + p) * headDim;
                        int dstOffset = slotOffset + h * headDim;
                        Buffer.BlockCopy(
                            kFlat, srcOffset * sizeof(float),
                            kPaged, dstOffset * sizeof(float),
                            headDim * sizeof(float));
                        Buffer.BlockCopy(
                            vFlat, srcOffset * sizeof(float),
                            vPaged, dstOffset * sizeof(float),
                            headDim * sizeof(float));
                    }
                }
            }

            return true;
        }

        private static unsafe bool TryReadCacheAsF32(Tensor cache, int totalElems, out float[] flat)
        {
            if (cache.ElementType == DType.Float32)
            {
                flat = cache.GetElementsAsFloat(totalElems);
                return true;
            }
            if (cache.ElementType == DType.Float16)
            {
                flat = new float[totalElems];
                ushort* src = TensorComputePrimitives.GetHalfPointer(cache);
                fixed (float* dst = flat)
                {
                    TensorComputePrimitives.F16ToF32(dst, src, totalElems);
                }
                return true;
            }
            flat = null;
            return false;
        }

        private void EnsureGptOssPagedBuffers(int numBlocks, int blockSize)
        {
            bool needRebuild = _gptOssPagedK == null
                || _gptOssPagedNumBlocks < numBlocks
                || _gptOssPagedBlockSize != blockSize;
            if (!needRebuild) return;

            int targetBlocks = Math.Max(numBlocks, _gptOssPagedNumBlocks * 2);
            int numLayers = Config.NumLayers;
            int numKvHeads = Config.NumKVHeads;
            int headDim = Config.HeadDim;
            int perTokenStride = numKvHeads * headDim;
            long bufSize = (long)targetBlocks * blockSize * perTokenStride;

            float[][] oldK = _gptOssPagedK;
            float[][] oldV = _gptOssPagedV;
            int oldNumBlocks = _gptOssPagedNumBlocks;
            int oldBlockSize = _gptOssPagedBlockSize;

            _gptOssPagedK = new float[numLayers][];
            _gptOssPagedV = new float[numLayers][];
            for (int l = 0; l < numLayers; l++)
            {
                _gptOssPagedK[l] = new float[bufSize];
                _gptOssPagedV[l] = new float[bufSize];
                if (oldK != null && oldNumBlocks > 0 && oldBlockSize == blockSize
                    && oldK[l] != null)
                {
                    long copyCount = (long)oldNumBlocks * blockSize * perTokenStride;
                    Array.Copy(oldK[l], 0, _gptOssPagedK[l], 0, copyCount);
                    Array.Copy(oldV[l], 0, _gptOssPagedV[l], 0, copyCount);
                }
            }

            _gptOssPagedNumBlocks = targetBlocks;
            _gptOssPagedBlockSize = blockSize;
        }
    }
}
