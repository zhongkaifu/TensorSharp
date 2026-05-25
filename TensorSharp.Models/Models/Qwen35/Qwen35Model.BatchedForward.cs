// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Qwen 3.5 batched paged-attention forward — vLLM-style continuous batching.
// Phase 1b: attention-only path. Mirrors Gemma 4's batched template adapted for
// Qwen 3.5-specific pieces (fused Q+gate projection, sigmoid-gated attention
// output, full attention without SWA, no KV donor mapping).
//
// Opt-in: TS_QWEN35_BATCHED=1. Default OFF until Phase 2 (GDN per-block state)
// lands and Qwen 3.5 actually runs end-to-end through this path. With the
// opt-in unset (current state), ForwardBatch throws NotSupportedException and
// BatchExecutor falls through to ExecuteStepPerSequence — identical behaviour
// to before this file existed.
//
// Coverage in this phase:
//   - Per-layer paged K/V buffers (one block-table-flat slot per token)
//   - Fused norm+QKV (gated Q + K + V) with Qwen 3.5's Q+gate interleaved layout
//   - QK-norm via the existing ApplyQKNormCached helper (already per-token)
//   - NeoX RoPE per layer over per-token positions (scalar positions; MRoPE
//     for vision goes through the per-seq path until Phase 4)
//   - PagedAttentionForward via ScatterKv + GgmlBasicOps.PagedAttentionForward
//   - Sigmoid-gated attention output via existing ApplySigmoidGate / SigmoidMul
//   - Attention output projection + residual add
//   - Dense SwiGLU FFN (FFNCached) — works per-token without modification
//   - Per-seq LM head + tied/untied output weight
//
// NotSupported (throws → per-seq fallback handles them):
//   - Any GDN (recurrent) layer → Phase 2 lands per-block conv/ssm state
//   - Any MoE layer → Phase 3
//   - Multimodal MRoPE positions pending → Phase 4
//   - Block-quantised KV cache dtype → won't reach without those fused kernels
//
// vLLM reference: vllm/model_executor/models/qwen3_5.py (Qwen3_5DecoderLayer
// dispatch on layer_types[i]).
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models.Paged;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models
{
    public partial class Qwen35Model : IBatchedPagedModel
    {
        // Per-layer paged K/V flat float buffers. Allocated lazily on the
        // first ForwardBatch call (and re-grown when the block table needs
        // more blocks than we've allocated). Layout is identical to Gemma 4's
        // _g4PagedK: [num_blocks * block_size * num_kv_heads * head_dim].
        // Only attention layers populate these; GDN layers (Phase 2) will own
        // their own per-block conv_state + ssm_state buffers.
        private float[][] _q35PagedK;
        private float[][] _q35PagedV;
        private int _q35PagedNumBlocks;
        private int _q35PagedBlockSize;

        // Phase 5c: per-slot GDN state, stored as native per-slot objects so
        // each per-seq GatedDeltaNet call swaps a REFERENCE into the
        // model-level _convState/_deltaStateTensor fields instead of copying
        // the slot's bytes in and out. Saves ~2 MB SSM + ~tens-of-KB conv
        // memcpy per GDN layer per seq (Phase 2 paid that cost twice — once
        // in, once out — and it dominated the per-seq overhead for batches
        // with many sequences).
        //
        // Each sequence's "slot" is still its primary block id
        // (seq.BlockTable.Blocks[0].Id), matching vLLM's state_indices_tensor
        // convention from gdn_linear_attn.py.
        //
        //   _q35GdnSlotConvBuf[layer][slot]    : float[(convKernel-1) * qkvDim]
        //   _q35GdnSlotConvWriteIdx[layer][slot]: int (per-slot ring write head)
        //   _q35GdnSlotSsmTensor[layer][slot]  : Tensor[numVHeads, headVDim, headKDim]
        //   _q35GdnSlotInit[layer][slot]       : true once a slot has been used
        //
        // Per-slot buffers and tensors are allocated lazily on first access
        // so workloads that only use a few slots don't pre-allocate the full
        // numSlots × ssmStateLen × numLayers footprint upfront.
        private float[][][] _q35GdnSlotConvBuf;
        private int[][]     _q35GdnSlotConvWriteIdx;
        private Tensor[][]  _q35GdnSlotSsmTensor;
        private bool[][]    _q35GdnSlotInit;

        // Phase 4: yes, ForwardBatch handles multimodal sequences directly
        // (vision embedding inject + per-batch MRoPE position table). The
        // batched path requires the same opt-in env var as the rest of the
        // Qwen3.5 batched plumbing — declining batched multimodal when the
        // opt-in is off so BatchExecutor falls back to the per-seq split.
        public bool SupportsBatchedMultimodal
        {
            get
            {
                string optIn = Environment.GetEnvironmentVariable("TS_QWEN35_BATCHED");
                return string.Equals(optIn, "1", StringComparison.Ordinal)
                    || string.Equals(optIn, "true", StringComparison.OrdinalIgnoreCase);
            }
        }

        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx)
        {
            if (ctx == null) throw new ArgumentNullException(nameof(ctx));
            int numSeqs = ctx.Sequences.Count;
            if (numSeqs == 0) return Array.Empty<float[]>();

            // Opt-in. Default off until Phase 2 lands. See file header.
            string optIn = Environment.GetEnvironmentVariable("TS_QWEN35_BATCHED");
            bool batchedEnabled = string.Equals(optIn, "1", StringComparison.Ordinal)
                || string.Equals(optIn, "true", StringComparison.OrdinalIgnoreCase);
            if (!batchedEnabled)
            {
                throw new NotSupportedException(
                    "Qwen 3.5 batched: not enabled. Set TS_QWEN35_BATCHED=1 to opt in (in progress).");
            }

            // Phase 4: multimodal sequences are now handled here. The injector
            // hands us per-seq MRoPE positions and vision embedding spans; we
            // build per-batch tables below (combined MRoPE table for the
            // attention RoPE step + globally-offset vision embeddings for the
            // post-embedding inject). BatchExecutor consults
            // SupportsBatchedMultimodal=true to keep multimodal seqs in this
            // path instead of peeling them off into the per-seq fallback.
            // (Clear any leftover MRoPE positions from a previous per-seq Forward
            // call so they don't bleed into the per-batch table we build below.)
            _pendingMRoPEPositions = null;

            int numLayers = Config.NumLayers;
            // Phase 3: MoE layers now dispatch through MoEForward in the FFN
            // block below. No early bail required.

            // ----- Flatten the per-seq scheduler metadata into batch arrays -----
            int blockSize = ctx.Sequences[0].BlockTable.BlockSize;
            int maxBlockId = 0;
            for (int s = 0; s < numSeqs; s++)
            {
                var bt = ctx.BlockTables[s];
                for (int b = 0; b < bt.Length; b++)
                    if (bt[b] > maxBlockId) maxBlockId = bt[b];
            }
            EnsureQwen35PagedBuffers(maxBlockId + 1, blockSize, numLayers);

            int numTokens = 0;
            for (int s = 0; s < numSeqs; s++) numTokens += ctx.NumScheduledTokens[s];

            int[] positions = ctx.Positions.ToArray();
            int[] queryStartLoc = ctx.QueryStartLoc.ToArray();
            int[] slotMapping = ctx.SlotMapping.ToArray();
            int[] seqLens = new int[numSeqs];
            for (int s = 0; s < numSeqs; s++)
                seqLens[s] = ctx.Sequences[s].NumComputedTokens + ctx.NumScheduledTokens[s];

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
            var (blockTableFlat, blockTableOffsets) = FlattenBlockTablesQ35(ctx.BlockTables);

            // ----- Embed once for the whole batch -----
            Tensor hiddenStates = Embedding(flatTokens);

            // ----- Phase 4: per-batch multimodal staging -----
            // For each sequence in the batch, fetch the matching slice of its
            // prepared vision embeddings + MRoPE positions from the injector
            // and stage them with positions adjusted to the global batched
            // hidden-tensor offset. Text-only sequences contribute nothing to
            // the vision list and get collapsed (k,k,k) MRoPE rows.
            int[] batchedMRoPE = null;
            bool anyMultimodal = false;
            var mmInj = MultimodalInjector as ModelMultimodalInjector;
            if (mmInj != null)
            {
                _visionEmbeddingsList.Clear();
                batchedMRoPE = new int[3 * numTokens];
                for (int s = 0; s < numSeqs; s++)
                {
                    var seq = ctx.Sequences[s];
                    int seqStart = queryStartLoc[s];
                    int seqLen = ctx.NumScheduledTokens[s];
                    int promptStartToken = seq.NumComputedTokens;

                    int[] mropeSlice = mmInj.TryGetMRoPEPositionsForSlice(seq.RequestId, promptStartToken, seqLen);
                    if (mropeSlice != null)
                    {
                        anyMultimodal = true;
                        Array.Copy(mropeSlice, 0, batchedMRoPE, 3 * seqStart, mropeSlice.Length);
                    }
                    else
                    {
                        // Text-only seq: (k,k,k) per token, where k is the
                        // absolute prompt token position. Collapsing the three
                        // axes to the same scalar makes MRoPE produce the same
                        // rotation as standard RoPE.
                        for (int t = 0; t < seqLen; t++)
                        {
                            int abs = promptStartToken + t;
                            batchedMRoPE[3 * (seqStart + t) + 0] = abs;
                            batchedMRoPE[3 * (seqStart + t) + 1] = abs;
                            batchedMRoPE[3 * (seqStart + t) + 2] = abs;
                        }
                    }

                    // Vision embeddings: ask the injector to push this seq's
                    // spans into _visionEmbeddingsList, then bump positions to
                    // the global batched offset. We snapshot the list length
                    // around the call to know which entries belong to this seq.
                    int countBefore = _visionEmbeddingsList.Count;
                    mmInj.QueuePromptEmbeddingsForSlice(promptStartToken, seqLen, seq.RequestId);
                    // QueuePromptEmbeddingsForSlice also calls SetMRoPEPositions
                    // for the matching slice. We've already pulled the MRoPE
                    // slice into batchedMRoPE above, so discard whatever it
                    // staged on the model.
                    _pendingMRoPEPositions = null;
                    for (int i = countBefore; i < _visionEmbeddingsList.Count; i++)
                    {
                        var (emb, pos) = _visionEmbeddingsList[i];
                        _visionEmbeddingsList[i] = (emb, pos + seqStart);
                        anyMultimodal = true;
                    }
                }

                if (_visionEmbeddingsList.Count > 0)
                    InjectVisionEmbeddings(hiddenStates, numTokens);
                if (!anyMultimodal) batchedMRoPE = null; // pure text — skip MRoPE work.
            }

            int hidden = Config.HiddenSize;
            int numHeads = Config.NumHeads;
            int numKVHeads = Config.NumKVHeads;
            int headDim = Config.HeadDim;
            int qFullDim = numHeads * headDim * 2; // Q + gate interleaved
            int kvDim = numKVHeads * headDim;
            int qDim = numHeads * headDim;
            float attentionScale = 1.0f / MathF.Sqrt(headDim);

            int ropeDim = _ropeDimCount > 0 ? _ropeDimCount : headDim;
            float ropeBase = Config.RopeBase;
            float ropeFreqScale = 1.0f / Config.RopeScale;

            // ----- Per-layer loop -----
            for (int layer = 0; layer < numLayers; layer++)
            {
                Tensor residual = Ops.NewContiguous(hiddenStates);

                bool isGdn = _isRecurrent != null && _isRecurrent[layer];
                Tensor attnProj;
                if (isGdn)
                {
                    // -------- GDN block (Phase 2: per-block state via swap) --------
                    // Each sequence's GDN state lives in _q35PagedConvState /
                    // _q35PagedSsmState indexed by its primary block id. We swap
                    // the active slot into the model-level _convState /
                    // _deltaStateTensor fields, invoke the existing
                    // GatedDeltaNet math (which expects single-seq state), then
                    // persist back. Output projection runs once over the
                    // assembled batched output.
                    attnProj = RunBatchedGdnLayer(hiddenStates, ctx, layer, numTokens, numSeqs, queryStartLoc);
                    hiddenStates.Dispose();
                }
                else
                {
                    // -------- Attention block --------
                    Tensor normed = RMSNormOpQ35(hiddenStates, _attnNormW[layer]);
                    hiddenStates.Dispose();

                    // Fused QKV projection. The fused weight emits
                    // [numTokens, qFullDim + 2*kvDim] where the first qFullDim cols
                    // are Q+gate interleaved per head.
                    Tensor qFull, kTensor, vTensor;
                    if (_attnQkvQW[layer] != null)
                    {
                        using Tensor fusedQkv = LinearForwardCached(normed, _attnQkvQW[layer], _attnQkvF32[layer]);
                        normed.Dispose();
                        using (var qView = fusedQkv.Narrow(1, 0, qFullDim))
                            qFull = Ops.NewContiguous(qView);
                        using (var kView = fusedQkv.Narrow(1, qFullDim, kvDim))
                            kTensor = Ops.NewContiguous(kView);
                        using (var vView = fusedQkv.Narrow(1, qFullDim + kvDim, kvDim))
                            vTensor = Ops.NewContiguous(vView);
                    }
                    else
                    {
                        qFull = LinearForwardCached(normed, _attnQQW[layer], _attnQF32[layer]);
                        kTensor = LinearForwardCached(normed, _attnKQW[layer], _attnKF32[layer]);
                        vTensor = LinearForwardCached(normed, _attnVQW[layer], _attnVF32[layer]);
                        normed.Dispose();
                    }

                    // Deinterleave Q from gate. Works as-is over numTokens.
                    DeinterleaveQGate(qFull, numTokens, numHeads, headDim, out Tensor qTensor, out Tensor gateTensor, out bool ownsQGateBuffers);
                    qFull.Dispose();

                    // QK-norm (pre-RoPE). Existing helper is per-token.
                    qTensor = ApplyQKNormCached(qTensor, _attnQNormW[layer], numHeads, numTokens);
                    kTensor = ApplyQKNormCached(kTensor, _attnKNormW[layer], numKVHeads, numTokens);

                    // RoPE: when at least one seq in the batch has per-axis
                    // MRoPE positions (vision/audio), route everything through
                    // ApplyMRoPEPrefill with the combined batched table —
                    // collapsed (k,k,k) rows make it identical to standard RoPE
                    // for text tokens. Otherwise the cheaper one-shot NeoX
                    // RoPE call wins.
                    if (batchedMRoPE != null)
                    {
                        qTensor = ApplyMRoPEPrefill(qTensor, numHeads, numTokens, batchedMRoPE);
                        kTensor = ApplyMRoPEPrefill(kTensor, numKVHeads, numTokens, batchedMRoPE);
                    }
                    else
                    {
                        using (var posTensorQ = BuildRoPEPositionsTensorQ35(positions, numHeads))
                        {
                            qTensor = ApplyBatchedRoPENeoXQ35(qTensor, posTensorQ, numTokens, numHeads, headDim, ropeDim, ropeBase, ropeFreqScale);
                        }
                        using (var posTensorK = BuildRoPEPositionsTensorQ35(positions, numKVHeads))
                        {
                            kTensor = ApplyBatchedRoPENeoXQ35(kTensor, posTensorK, numTokens, numKVHeads, headDim, ropeDim, ropeBase, ropeFreqScale);
                        }
                    }

                    // Scatter K, V into the paged buffer.
                    float[] kFlat = kTensor.GetElementsAsFloat(numTokens * kvDim);
                    float[] vFlat = vTensor.GetElementsAsFloat(numTokens * kvDim);
                    PagedKvBatchOps.ScatterKv(
                        kFlat, vFlat, _q35PagedK[layer], _q35PagedV[layer],
                        slotMapping, numTokens, numKVHeads, headDim, _q35PagedBlockSize);
                    kTensor.Dispose();
                    vTensor.Dispose();

                    // Native paged attention. Full attention (no SWA).
                    float[] qFlat = qTensor.GetElementsAsFloat(numTokens * qDim);
                    qTensor.Dispose();
                    float[] attnFlat = new float[numTokens * qDim];
                    GgmlBasicOps.PagedAttentionForward(
                        qFlat, _q35PagedK[layer], _q35PagedV[layer], attnFlat,
                        queryStartLoc, seqLens, positions,
                        blockTableFlat, blockTableOffsets,
                        numSeqs, numTokens, numHeads, numKVHeads, headDim,
                        _q35PagedBlockSize, attentionScale, /*slidingWindow*/ 0);

                    Tensor attnOut = CreateFloatTensor(attnFlat, numTokens, qDim);

                    // Sigmoid-gated attention output: attn = attn * sigmoid(gate).
                    Ops.SigmoidMul(attnOut, attnOut, gateTensor);
                    if (ownsQGateBuffers)
                        gateTensor.Dispose();

                    // Output projection.
                    attnProj = LinearForwardCached(attnOut, _attnOutputQW[layer], _attnOutputF32[layer]);
                    attnOut.Dispose();
                }

                // Residual add (same shape for both branches).
                Ops.Add(residual, residual, attnProj);
                attnProj.Dispose();

                // -------- FFN block --------
                // post_attention_norm in this port (HF naming: post-attn /
                // pre-FFN layernorm). Phase 3: branch on _isMoeLayer to
                // dispatch the right FFN — MoE expert routing for Qwen3.5-MoE
                // GGUFs (35B-A3B); dense SwiGLU otherwise. Both are per-token
                // (no cross-token state), so calling with numTokens just works
                // for the batched layout.
                Tensor ffnNormed = RMSNormOpQ35(residual, _postAttnNormW[layer]);
                Tensor ffnOut = (_isMoeLayer != null && _isMoeLayer[layer])
                    ? MoEForward(ffnNormed, layer, numTokens)
                    : FFNCached(ffnNormed, layer, numTokens);
                ffnNormed.Dispose();
                Ops.Add(residual, residual, ffnOut);
                ffnOut.Dispose();

                hiddenStates = residual;
            }

            // ----- Final norm + per-sequence LM head -----
            Tensor finalNormed = RMSNormOpQ35(hiddenStates, _finalNormW);
            hiddenStates.Dispose();

            float[] finalFlat = finalNormed.GetElementsAsFloat(numTokens * hidden);
            finalNormed.Dispose();
            float[] lastTokensPacked = PagedKvBatchOps.GatherLastTokenPerSeq(
                finalFlat, hidden, queryStartLoc, numSeqs);
            Tensor lastHidden = CreateFloatTensor(lastTokensPacked, numSeqs, hidden);

            // LM head. _lmHeadQW / _lmHeadF32 already resolve tied vs untied
            // at load time (Qwen35Model.cs:834-837 — falls back to
            // token_embd.weight when output.weight is missing).
            Tensor logitsTensor = LinearForwardCached(lastHidden, _lmHeadQW, _lmHeadF32);
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

        // ----- Per-Qwen3.5 batched helpers (no Gemma4 cross-dependency) -----

        private Tensor RMSNormOpQ35(Tensor input, Tensor alpha)
            => Ops.RMSNorm(null, input, alpha, null, Config.Eps);

        private Tensor BuildRoPEPositionsTensorQ35(int[] tokenPositions, int numHeads)
        {
            // The legacy ApplyRoPEPrefill replicates the position per-head so
            // Ops.RoPEEx (which takes one position per logical row) sees the
            // same position for every head of a token. Reuse that layout for
            // the batched path.
            int total = tokenPositions.Length * numHeads;
            int[] expanded = new int[total];
            for (int t = 0; t < tokenPositions.Length; t++)
                for (int h = 0; h < numHeads; h++)
                    expanded[t * numHeads + h] = tokenPositions[t];
            return CreateIntTensor(expanded, total);
        }

        private Tensor ApplyBatchedRoPENeoXQ35(
            Tensor data, Tensor positionsTensor,
            int numTokens, int numHeads, int headDim, int ropeDim, float ropeBase, float ropeFreqScale)
        {
            using var reshaped = data.View(1, numTokens, numHeads, headDim);
            // NeoX mode = 2, matches the legacy ApplyRoPEPrefill call site
            // (Qwen35Model.cs:2994). No freq_factors / yaRN scaling for
            // Qwen 3.5's stock GGUFs.
            Ops.RoPEEx(reshaped, reshaped, positionsTensor, ropeDim, 2, 0,
                       ropeBase, ropeFreqScale, 0.0f, 1.0f, 0.0f, 0.0f);
            return data;
        }

        private static (int[] flat, int[] offsets) FlattenBlockTablesQ35(int[][] tables)
        {
            int[] offsets = new int[tables.Length + 1];
            int total = 0;
            for (int s = 0; s < tables.Length; s++)
            {
                offsets[s] = total;
                total += tables[s].Length;
            }
            offsets[tables.Length] = total;
            int[] flat = new int[total];
            for (int s = 0; s < tables.Length; s++)
                Array.Copy(tables[s], 0, flat, offsets[s], tables[s].Length);
            return (flat, offsets);
        }

        private void EnsureQwen35PagedBuffers(int numBlocks, int blockSize, int numLayers)
        {
            bool needRebuild = _q35PagedK == null
                || _q35PagedNumBlocks < numBlocks
                || _q35PagedBlockSize != blockSize;
            if (!needRebuild) return;

            // Grow with 2x slack so we don't reallocate every step. CRITICAL:
            // preserve existing K/V (and GDN state) on grow — a fresh sequence's
            // prefill K/V depends on the previously allocated buffer's contents.
            int targetBlocks = Math.Max(numBlocks, _q35PagedNumBlocks * 2);

            float[][] oldPagedK = _q35PagedK;
            float[][] oldPagedV = _q35PagedV;
            float[][][] oldConvBuf = _q35GdnSlotConvBuf;
            int[][] oldConvIdx = _q35GdnSlotConvWriteIdx;
            Tensor[][] oldSsm = _q35GdnSlotSsmTensor;
            bool[][] oldInit = _q35GdnSlotInit;
            int oldNumBlocks = _q35PagedNumBlocks;
            int oldBlockSize = _q35PagedBlockSize;

            int numKVHeads = Config.NumKVHeads;
            int headDim = Config.HeadDim;
            int perTokenStride = numKVHeads * headDim;

            _q35PagedK = new float[numLayers][];
            _q35PagedV = new float[numLayers][];
            _q35GdnSlotConvBuf = new float[numLayers][][];
            _q35GdnSlotConvWriteIdx = new int[numLayers][];
            _q35GdnSlotSsmTensor = new Tensor[numLayers][];
            _q35GdnSlotInit = new bool[numLayers][];

            for (int l = 0; l < numLayers; l++)
            {
                if (_isRecurrent != null && _isRecurrent[l])
                {
                    // GDN layer: per-slot conv ring buffer + ssm tensor are
                    // allocated LAZILY in EnsureGdnSlotAllocated when first
                    // touched. Pre-allocate only the slot-indexed outer
                    // arrays here.
                    _q35GdnSlotConvBuf[l] = new float[targetBlocks][];
                    _q35GdnSlotConvWriteIdx[l] = new int[targetBlocks];
                    _q35GdnSlotSsmTensor[l] = new Tensor[targetBlocks];
                    _q35GdnSlotInit[l] = new bool[targetBlocks];
                    if (oldConvBuf != null && oldConvBuf[l] != null && oldNumBlocks > 0)
                    {
                        // Preserve previously-allocated slots — sequences in
                        // flight depend on their slot's persistent recurrent
                        // state. Element-wise copy of the reference arrays;
                        // each slot's float[] / Tensor reference is moved as
                        // a single pointer.
                        Array.Copy(oldConvBuf[l], 0, _q35GdnSlotConvBuf[l], 0, oldNumBlocks);
                        Array.Copy(oldConvIdx[l], 0, _q35GdnSlotConvWriteIdx[l], 0, oldNumBlocks);
                        Array.Copy(oldSsm[l], 0, _q35GdnSlotSsmTensor[l], 0, oldNumBlocks);
                        Array.Copy(oldInit[l], 0, _q35GdnSlotInit[l], 0, oldNumBlocks);
                    }
                }
                else
                {
                    // Attention layer: paged K/V buffers.
                    long bufSize = (long)targetBlocks * blockSize * perTokenStride;
                    _q35PagedK[l] = new float[bufSize];
                    _q35PagedV[l] = new float[bufSize];
                    if (oldPagedK != null && oldNumBlocks > 0 && oldBlockSize == blockSize
                        && oldPagedK[l] != null)
                    {
                        long copyCount = (long)oldNumBlocks * blockSize * perTokenStride;
                        Array.Copy(oldPagedK[l], 0, _q35PagedK[l], 0, copyCount);
                        Array.Copy(oldPagedV[l], 0, _q35PagedV[l], 0, copyCount);
                    }
                }
            }

            _q35PagedNumBlocks = targetBlocks;
            _q35PagedBlockSize = blockSize;
        }

        /// <summary>Lazily allocate slot's conv ring buffer + SSM tensor on
        /// first access. The state is zero-initialised; subsequent calls
        /// reuse the same references and accumulate state across forward
        /// calls for that slot.</summary>
        private void EnsureGdnSlotAllocated(int layer, int slot)
        {
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;
            int convStateLen = Math.Max(_convKernel - 1, 0) * qkvDim;
            if (_q35GdnSlotConvBuf[layer][slot] == null)
            {
                _q35GdnSlotConvBuf[layer][slot] = new float[convStateLen];
                _q35GdnSlotConvWriteIdx[layer][slot] = 0;
            }
            if (_q35GdnSlotSsmTensor[layer][slot] == null)
            {
                _q35GdnSlotSsmTensor[layer][slot] = new Tensor(
                    _allocator, DType.Float32, _numVHeads, _headVDim, _headKDim);
                Ops.Fill(_q35GdnSlotSsmTensor[layer][slot], 0);
            }
            else if (!_q35GdnSlotInit[layer][slot])
            {
                // Slot's references survived a buffer regrow but the init
                // flag was wiped — re-zero before use.
                Array.Clear(_q35GdnSlotConvBuf[layer][slot], 0,
                            _q35GdnSlotConvBuf[layer][slot].Length);
                _q35GdnSlotConvWriteIdx[layer][slot] = 0;
                Ops.Fill(_q35GdnSlotSsmTensor[layer][slot], 0);
            }
            _q35GdnSlotInit[layer][slot] = true;
        }

        // ----- GDN per-layer batched dispatch (Phase 5c: reference-swap) -----
        //
        // For each sequence in the batch:
        //   1. Ensure that seq's per-slot state objects exist (lazy alloc).
        //   2. Point the model-level _convState[layer] / _deltaStateTensor[layer]
        //      / _convStateWriteIdx[layer] at THIS slot's storage. No bytes
        //      are copied — GatedDeltaNet's element-wise writes mutate the
        //      slot's float[] / Tensor in place.
        //   3. Slice this seq's input from the batched hidden tensor.
        //   4. Set _cacheSeqLen to seq.NumComputedTokens so GatedDeltaNet's
        //      "where am I in the recurrent state" tracking is correct.
        //   5. Call GatedDeltaNet(skipOutputProj: true) for this seq.
        //   6. Read back the scalar writeIdx (it's an int, not a reference).
        //   7. Copy the per-seq gated output into the assembled batched buffer.
        // After the loop, restore the model-level state references to the
        // original "scratch" instances so legacy Forward paths that share the
        // model see fresh storage, not slot N's persistent state. Output
        // projection runs once over the assembled batched output.
        //
        // Saves ~2 MB SSM + ~tens-of-KB conv memcpy per GDN layer per seq
        // compared to Phase 2's byte-copy LoadGdnStateForSlot/SaveGdnStateForSlot.
        // Sequences still run sequentially through the GDN math itself — a
        // true gather/scatter kernel à la vLLM's state_indices_tensor would
        // be the next perf jump after this.
        // Phase 7 native batched GDN op (see ggml_ops_gated_delta_net.cpp:
        // TSGgml_GatedDeltaNetBatchedStepF32). Single C dispatch covers every
        // (seq, token) pair using per-slot state indexing — vLLM's
        // fused_sigmoid_gating_delta_rule_update API surface. Opt-in until
        // perf vs the verified Phase 5c per-seq path is validated; in this
        // session the model-load + run cycle didn't complete within the
        // available wall time, so we keep the verified Phase 5c path as
        // default and expose the native path via env var.
        // Method getter so tests can toggle the native path after class load.
        // A `static readonly` here would capture TS_QWEN35_BATCHED_GDN_NATIVE
        // at class-init time, which is before tests get a chance to set it —
        // the same gotcha that bit Nemotron's Phase 9 verification.
        private static bool UseNativeBatchedGdn() =>
            string.Equals(Environment.GetEnvironmentVariable("TS_QWEN35_BATCHED_GDN_NATIVE"),
                          "1", StringComparison.Ordinal);

        private Tensor RunBatchedGdnLayer(
            Tensor hiddenStates, BatchedForwardContext ctx, int layer,
            int numTokens, int numSeqs, int[] queryStartLoc)
        {
            return UseNativeBatchedGdn()
                ? RunBatchedGdnLayerNative(hiddenStates, ctx, layer, numTokens, numSeqs, queryStartLoc)
                : RunBatchedGdnLayerPerSeq(hiddenStates, ctx, layer, numTokens, numSeqs, queryStartLoc);
        }

        // Phase 5c per-slot reference-swap (verified). Each seq's slice runs
        // through GatedDeltaNet against its own _convState/_deltaStateTensor
        // (swapped in via the array). Reference assignment, not memcpy.
        private Tensor RunBatchedGdnLayerPerSeq(
            Tensor hiddenStates, BatchedForwardContext ctx, int layer,
            int numTokens, int numSeqs, int[] queryStartLoc)
        {
            int ssmDInner = _ssmDInner;

            float[] origConvBuf      = _convState[layer];
            Tensor  origSsmTensor    = _deltaStateTensor[layer];
            int     origConvWriteIdx = _convStateWriteIdx[layer];
            int     savedCacheSeqLen = _cacheSeqLen;

            float[] batchedGatedFlat = new float[numTokens * ssmDInner];
            try
            {
                for (int s = 0; s < numSeqs; s++)
                {
                    var seq = ctx.Sequences[s];
                    int slot = seq.BlockTable.Blocks[0].Id;
                    int seqStart = queryStartLoc[s];
                    int seqLen = ctx.NumScheduledTokens[s];
                    if (seqLen <= 0) continue;

                    if (seq.NumComputedTokens == 0)
                        _q35GdnSlotInit[layer][slot] = false;
                    EnsureGdnSlotAllocated(layer, slot);

                    _convState[layer]         = _q35GdnSlotConvBuf[layer][slot];
                    _deltaStateTensor[layer]  = _q35GdnSlotSsmTensor[layer][slot];
                    _convStateWriteIdx[layer] = _q35GdnSlotConvWriteIdx[layer][slot];
                    _cacheSeqLen = seq.NumComputedTokens;

                    using Tensor seqHidden = Ops.NewContiguous(hiddenStates.Narrow(0, seqStart, seqLen));
                    using Tensor gated = GatedDeltaNet(
                        seqHidden, _attnNormW[layer], layer, seqLen,
                        residual: null, skipOutputProj: true);
                    if (gated == null)
                        throw new InvalidOperationException(
                            $"GatedDeltaNet returned null for layer {layer}, seq {seq.RequestId}.");

                    _q35GdnSlotConvWriteIdx[layer][slot] = _convStateWriteIdx[layer];

                    float[] gatedFlat = gated.GetElementsAsFloat(seqLen * ssmDInner);
                    Buffer.BlockCopy(
                        gatedFlat, 0,
                        batchedGatedFlat, seqStart * ssmDInner * sizeof(float),
                        seqLen * ssmDInner * sizeof(float));
                }
            }
            finally
            {
                _convState[layer]         = origConvBuf;
                _deltaStateTensor[layer]  = origSsmTensor;
                _convStateWriteIdx[layer] = origConvWriteIdx;
                _cacheSeqLen              = savedCacheSeqLen;
            }

            using Tensor batchedGated = CreateFloatTensor(batchedGatedFlat, numTokens, ssmDInner);
            return LinearForwardCached(batchedGated, _ssmOutQW[layer], _ssmOutF32[layer]);
        }

        // Phase 7 native batched path. Per-seq input projection (FusedNormLinear)
        // still serializes the heavy matmul; the win is replacing N RunPerTokenLoop
        // C# calls with one native dispatch that walks every (seq, token) pair.
        // Opt-in (TS_QWEN35_BATCHED_GDN_NATIVE=1) — see s_useNativeBatchedGdn note.
        private unsafe Tensor RunBatchedGdnLayerNative(
            Tensor hiddenStates, BatchedForwardContext ctx, int layer,
            int numTokens, int numSeqs, int[] queryStartLoc)
        {
            int ssmDInner = _ssmDInner;
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;
            int qkDim = _headKDim * _numKHeads;
            int vDim = _headVDim * _numVHeads;
            int zDim = vDim;
            int packedDim = qkvDim + zDim + _numVHeads * 2;

            float[] packedBatched = new float[numTokens * packedDim];
            for (int s = 0; s < numSeqs; s++)
            {
                int seqStart = queryStartLoc[s];
                int seqLen = ctx.NumScheduledTokens[s];
                if (seqLen <= 0) continue;

                using Tensor seqHidden = Ops.NewContiguous(hiddenStates.Narrow(0, seqStart, seqLen));
                Tensor seqPacked;
                if (_attnNormW[layer] != null && _ssmInProjQW[layer] != null && IsGgmlBackend)
                {
                    seqPacked = FusedNormLinear(seqHidden, _attnNormW[layer],
                        _ssmInProjQW[layer], _ssmInProjF32[layer]);
                }
                else
                {
                    using Tensor seqNormed = _attnNormW[layer] != null
                        ? RMSNormOpCached(seqHidden, _attnNormW[layer])
                        : seqHidden.CopyRef();
                    seqPacked = LinearForwardCached(seqNormed, _ssmInProjQW[layer], _ssmInProjF32[layer]);
                }
                try
                {
                    float[] packedFlat = seqPacked.GetElementsAsFloat(seqLen * packedDim);
                    Buffer.BlockCopy(packedFlat, 0,
                        packedBatched, seqStart * packedDim * sizeof(float),
                        seqLen * packedDim * sizeof(float));
                }
                finally { seqPacked.Dispose(); }
            }

            var descs = new GdnBatchedSeqDesc[numSeqs];
            var convHandles = new GCHandle[numSeqs];
            float[] batchedGatedFlat = new float[numTokens * ssmDInner];
            try
            {
                for (int s = 0; s < numSeqs; s++)
                {
                    var seq = ctx.Sequences[s];
                    int slot = seq.BlockTable.Blocks[0].Id;
                    int seqStart = queryStartLoc[s];
                    int seqLen = ctx.NumScheduledTokens[s];

                    if (seq.NumComputedTokens == 0)
                        _q35GdnSlotInit[layer][slot] = false;
                    EnsureGdnSlotAllocated(layer, slot);

                    float[] convBuf = _q35GdnSlotConvBuf[layer][slot];
                    Tensor   ssmTen = _q35GdnSlotSsmTensor[layer][slot];
                    convHandles[s] = GCHandle.Alloc(convBuf, GCHandleType.Pinned);

                    descs[s].SeqStart     = seqStart;
                    descs[s].SeqLen       = seqLen;
                    descs[s].ConvWriteIdx = _q35GdnSlotConvWriteIdx[layer][slot];
                    descs[s].Pad          = 0;
                    descs[s].ConvState    = convHandles[s].AddrOfPinnedObject();
                    descs[s].SsmState     = (IntPtr)GetFloatPtr(ssmTen);
                }

                float* dtBiasPtr  = GetFloatPtr(_ssmDtBiasW[layer]);
                float* aPtr       = GetFloatPtr(_ssmAW[layer]);
                float* ssmNormPtr = GetFloatPtr(_ssmNormW[layer]);
                float[] convWT    = _gdnConvWT[layer];

                fixed (float* convWTPin = convWT,
                               packedPin = packedBatched,
                               gatedPin = batchedGatedFlat)
                {
                    GgmlBasicOps.GatedDeltaNetBatchedStep(
                        descs, numTokens,
                        (IntPtr)packedPin, packedDim, qkvDim, qkDim, vDim, zDim,
                        _numKHeads, _numVHeads, _headKDim, _headVDim,
                        _convKernel, ssmDInner,
                        (IntPtr)convWTPin, (IntPtr)dtBiasPtr, (IntPtr)aPtr, (IntPtr)ssmNormPtr,
                        Config.Eps, (IntPtr)gatedPin);
                }

                for (int s = 0; s < numSeqs; s++)
                {
                    var seq = ctx.Sequences[s];
                    int slot = seq.BlockTable.Blocks[0].Id;
                    _q35GdnSlotConvWriteIdx[layer][slot] = descs[s].ConvWriteIdx;
                }
            }
            finally
            {
                for (int s = 0; s < numSeqs; s++)
                    if (convHandles[s].IsAllocated) convHandles[s].Free();
            }

            using Tensor batchedGated = CreateFloatTensor(batchedGatedFlat, numTokens, ssmDInner);
            return LinearForwardCached(batchedGated, _ssmOutQW[layer], _ssmOutF32[layer]);
        }
    }
}
