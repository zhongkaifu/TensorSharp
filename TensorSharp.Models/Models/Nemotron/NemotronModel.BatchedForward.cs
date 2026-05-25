// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Nemotron 3 (Nemotron-H hybrid) batched paged-attention forward — vLLM-style
// continuous batching. Nemotron-H mixes three layer types:
//   - Attention layers: standard causal attention with paged K/V (vLLM-style).
//   - Mamba2 SSM layers: recurrent state per sequence (conv1d ring + SSM state).
//     vLLM batches these via state_indices_tensor flat indexing into a slot
//     pool; we mirror that with a per-slot state pool (Phase 2) and call the
//     existing single-seq native Mamba2 kernels per-sequence with state-swap
//     (Phase 5; same pattern as Qwen 3.5 Phase 5c reference-swap).
//   - FFN layers: plain SwiGLU or MoE — token-parallel for SwiGLU; MoE routing
//     is per-token so it naturally batches.
//
// Coverage in Phase 1 (this iteration):
//   - IBatchedPagedModel interface
//   - TS_NEMOTRON_BATCHED opt-in env var (default OFF — falls through to the
//     existing per-seq KV-swap path until later phases land)
//   - SupportsBatchedMultimodal = false (multimodal stays on per-seq path)
//   - ForwardBatch throws NotSupportedException so BatchExecutor catches and
//     falls through to ExecuteStepPerSequence; behaviour unchanged vs today.
//
// Subsequent phases (see phase plan in commit history):
//   Phase 2: per-slot Mamba2 conv + SSM state pool
//   Phase 3: attention layer batched compute (paged attention)
//   Phase 4: FFN / MoE batched compute
//   Phase 5: Mamba2 batched via per-seq state-swap on the existing native kernel
//   Phase 6: correctness vs legacy + perf bench
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
    public partial class NemotronModel : IBatchedPagedModel
    {
        // Opt-in env var, default OFF. Re-read each call so tests can toggle
        // between paths after the model has already loaded — Qwen 3.5's
        // batched partial uses the same pattern (vs a `static readonly` which
        // would capture the env var at class-load time, before tests get a
        // chance to set it).
        private static bool NemoBatchedOptIn() =>
            string.Equals(Environment.GetEnvironmentVariable("TS_NEMOTRON_BATCHED"),
                          "1", StringComparison.Ordinal);

        // Phase 8: ForwardBatch handles multimodal injection directly (same
        // row-wise InjectMultimodalEmbeddings the legacy path uses; works on
        // batched [numTokens, hidden] unchanged). Gate behind the same opt-in
        // env var as the rest of the batched plumbing so BatchExecutor only
        // keeps multimodal seqs in this path when the opt-in is on; otherwise
        // it peels them off to the per-seq fallback.
        //
        // NOTE: the pending-embeddings list is model-level and doesn't track
        // which sequence each entry belongs to. We trust the upstream engine
        // to serialize multimodal requests (only one sequence at a time has
        // pending embeddings), matching the Mistral 3 batched stance.
        public bool SupportsBatchedMultimodal => NemoBatchedOptIn();

        // ----- Phase 2: per-layer paged K/V + per-slot Mamba2 state pool -----
        //
        // Nemotron-H is hybrid; attention layers and Mamba2 layers each need a
        // different paged storage shape:
        //
        //   Attention layers: vLLM-style block-paged K/V keyed by block id +
        //     slot offset. Sized [numBlocks * blockSize * numKVHeads * headDim]
        //     per layer. Multiple sequences index different blocks/slots via the
        //     block table.
        //
        //   Mamba2 layers: per-sequence recurrent state. Mirrors vLLM's
        //     state_indices_tensor scheme — a slot pool keyed by sequence's
        //     primary block id. Each slot owns its own conv ring buffer + SSM
        //     state float[]; per-seq forward swaps the model-level pointers
        //     to point at this slot's storage (same reference-swap pattern as
        //     Qwen 3.5 Phase 5c — proven to give ~1.83× tps at n=3 there).
        private float[][] _nemoPagedK;            // [layer][numBlocks * blockSize * kvDim]
        private float[][] _nemoPagedV;            // [layer][numBlocks * blockSize * kvDim]
        private int _nemoPagedNumBlocks;
        private int _nemoPagedBlockSize;
        private int[] _nemoPagedKvDimPerLayer;    // numKVHeads_l * headDim, varies per layer

        private float[][][] _nemoSlotConvBuf;     // [layer][slot] — conv ring buffer (size = (dConv-1) * convChannels)
        private float[][][] _nemoSlotSsmState;    // [layer][slot] — SSM state    (size = dState * headDim * nHead)
        private bool[][]    _nemoSlotInit;        // [layer][slot] — true once first-touch zero-init complete

        // Per-slot scratch tensors for the existing single-seq native Mamba2
        // decode kernel (mirrors _mamba2NativeDecodeProjected / Hidden, but
        // per-slot so concurrent decodes on different slots don't trample
        // each other's GPU-side staging tensors).
        private Tensor[][] _nemoSlotMamba2NativeDecodeProjected;  // [layer][slot]
        private Tensor[][] _nemoSlotMamba2NativeDecodeHidden;     // [layer][slot]
        private bool[][]   _nemoSlotMamba2NativeDecodeStateInitialized; // [layer][slot]

        /// <summary>Ensure per-layer paged K/V buffers exist for the given
        /// block-pool shape. Grows with 2× slack; preserves previously-written
        /// K/V on resize (a fresh sequence's prefill K/V depends on its
        /// block's contents from the moment the block was first allocated).
        /// Slot pool outer arrays are also extended here; per-slot inner
        /// allocations are lazy in <see cref="EnsureNemoSlotAllocated"/>.</summary>
        private void EnsureNemoPagedBuffers(int numBlocks, int blockSize, int numLayers)
        {
            bool needRebuild = _nemoPagedK == null
                || _nemoPagedNumBlocks < numBlocks
                || _nemoPagedBlockSize != blockSize;
            if (!needRebuild) return;

            int targetBlocks = Math.Max(numBlocks, _nemoPagedNumBlocks * 2);

            float[][] oldPagedK = _nemoPagedK;
            float[][] oldPagedV = _nemoPagedV;
            float[][][] oldConvBuf = _nemoSlotConvBuf;
            float[][][] oldSsm = _nemoSlotSsmState;
            bool[][] oldInit = _nemoSlotInit;
            Tensor[][] oldNdProj = _nemoSlotMamba2NativeDecodeProjected;
            Tensor[][] oldNdHidden = _nemoSlotMamba2NativeDecodeHidden;
            bool[][] oldNdInit = _nemoSlotMamba2NativeDecodeStateInitialized;
            int oldNumBlocks = _nemoPagedNumBlocks;
            int oldBlockSize = _nemoPagedBlockSize;

            _nemoPagedK = new float[numLayers][];
            _nemoPagedV = new float[numLayers][];
            _nemoSlotConvBuf = new float[numLayers][][];
            _nemoSlotSsmState = new float[numLayers][][];
            _nemoSlotInit = new bool[numLayers][];
            _nemoSlotMamba2NativeDecodeProjected = new Tensor[numLayers][];
            _nemoSlotMamba2NativeDecodeHidden = new Tensor[numLayers][];
            _nemoSlotMamba2NativeDecodeStateInitialized = new bool[numLayers][];

            if (_nemoPagedKvDimPerLayer == null || _nemoPagedKvDimPerLayer.Length != numLayers)
            {
                _nemoPagedKvDimPerLayer = new int[numLayers];
                for (int l = 0; l < numLayers; l++)
                {
                    if (_layerTypes[l] == LayerType.Attention)
                        _nemoPagedKvDimPerLayer[l] = _layerNumKVHeads[l] * Config.HeadDim;
                    else
                        _nemoPagedKvDimPerLayer[l] = 0;
                }
            }

            for (int l = 0; l < numLayers; l++)
            {
                if (_layerTypes[l] == LayerType.Mamba2)
                {
                    _nemoSlotConvBuf[l] = new float[targetBlocks][];
                    _nemoSlotSsmState[l] = new float[targetBlocks][];
                    _nemoSlotInit[l] = new bool[targetBlocks];
                    _nemoSlotMamba2NativeDecodeProjected[l] = new Tensor[targetBlocks];
                    _nemoSlotMamba2NativeDecodeHidden[l] = new Tensor[targetBlocks];
                    _nemoSlotMamba2NativeDecodeStateInitialized[l] = new bool[targetBlocks];
                    if (oldConvBuf != null && oldConvBuf[l] != null && oldNumBlocks > 0)
                    {
                        // Preserve previously-allocated slot references — sequences
                        // in flight rely on their slot's persistent recurrent state.
                        Array.Copy(oldConvBuf[l], 0, _nemoSlotConvBuf[l], 0, oldNumBlocks);
                        Array.Copy(oldSsm[l],     0, _nemoSlotSsmState[l], 0, oldNumBlocks);
                        Array.Copy(oldInit[l],    0, _nemoSlotInit[l],     0, oldNumBlocks);
                        Array.Copy(oldNdProj[l],   0, _nemoSlotMamba2NativeDecodeProjected[l], 0, oldNumBlocks);
                        Array.Copy(oldNdHidden[l], 0, _nemoSlotMamba2NativeDecodeHidden[l],    0, oldNumBlocks);
                        Array.Copy(oldNdInit[l],   0, _nemoSlotMamba2NativeDecodeStateInitialized[l], 0, oldNumBlocks);
                    }
                }
                else if (_layerTypes[l] == LayerType.Attention)
                {
                    int perTokenStride = _nemoPagedKvDimPerLayer[l];
                    long bufSize = (long)targetBlocks * blockSize * perTokenStride;
                    _nemoPagedK[l] = new float[bufSize];
                    _nemoPagedV[l] = new float[bufSize];
                    if (oldPagedK != null && oldNumBlocks > 0 && oldBlockSize == blockSize
                        && oldPagedK[l] != null)
                    {
                        long copyCount = (long)oldNumBlocks * blockSize * perTokenStride;
                        Array.Copy(oldPagedK[l], 0, _nemoPagedK[l], 0, copyCount);
                        Array.Copy(oldPagedV[l], 0, _nemoPagedV[l], 0, copyCount);
                    }
                }
                // FFN layers have no per-layer paged state; nothing to allocate.
            }

            _nemoPagedNumBlocks = targetBlocks;
            _nemoPagedBlockSize = blockSize;
        }

        /// <summary>Lazily allocate this slot's Mamba2 conv ring + SSM state on
        /// first touch. State is zero-initialised; subsequent calls reuse the
        /// same float[] references and accumulate state across forward calls
        /// for that slot. Re-zeroes when a slot is reused for a fresh sequence
        /// (NumComputedTokens == 0 in the caller).</summary>
        private void EnsureNemoSlotAllocated(int layer, int slot)
        {
            int convDim = Math.Max(0, _ssmDConv - 1);
            int convChannels = _ssmDInner + 2 * _ssmNGroup * _ssmDState;
            int convStateLen = convDim * convChannels;
            int ssmStateLen = _ssmDState * _ssmHeadDim * _ssmNHead;

            if (_nemoSlotConvBuf[layer][slot] == null)
                _nemoSlotConvBuf[layer][slot] = new float[convStateLen];
            if (_nemoSlotSsmState[layer][slot] == null)
                _nemoSlotSsmState[layer][slot] = new float[ssmStateLen];

            if (!_nemoSlotInit[layer][slot])
            {
                Array.Clear(_nemoSlotConvBuf[layer][slot], 0, _nemoSlotConvBuf[layer][slot].Length);
                Array.Clear(_nemoSlotSsmState[layer][slot], 0, _nemoSlotSsmState[layer][slot].Length);
                _nemoSlotMamba2NativeDecodeStateInitialized[layer][slot] = false;
                _nemoSlotInit[layer][slot] = true;
            }

            // Lazy allocate the per-slot decode-staging tensors only when the
            // GGML backend is active — those tensors live in GPU memory and
            // would otherwise be a wasted ~few KB per (layer, slot) on CPU.
            if (IsGgmlBackend && _nemoSlotMamba2NativeDecodeProjected[layer][slot] == null)
            {
                int dInProjTotal = 2 * _ssmDInner + 2 * _ssmNGroup * _ssmDState + _ssmNHead;
                _nemoSlotMamba2NativeDecodeProjected[layer][slot] =
                    new Tensor(_allocator, DType.Float32, 1, dInProjTotal);
                _nemoSlotMamba2NativeDecodeHidden[layer][slot] =
                    new Tensor(_allocator, DType.Float32, 1, _ssmDInner);
            }
        }

        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx)
        {
            if (ctx == null) throw new ArgumentNullException(nameof(ctx));
            int numSeqs = ctx.Sequences.Count;
            if (numSeqs == 0) return Array.Empty<float[]>();

            // Opt-in gate (Phase 1).
            if (!NemoBatchedOptIn())
                throw new NotSupportedException(
                    "Nemotron batched: disabled (set TS_NEMOTRON_BATCHED=1 to opt in).");

            int numLayers = Config.NumLayers;

            // ----- Resolve batch shape + paged-buffer sizing -----
            int blockSize = ctx.Sequences[0].BlockTable.BlockSize;
            int maxBlockId = 0;
            for (int s = 0; s < numSeqs; s++)
            {
                var bt = ctx.BlockTables[s];
                for (int b = 0; b < bt.Length; b++)
                    if (bt[b] > maxBlockId) maxBlockId = bt[b];
            }
            EnsureNemoPagedBuffers(maxBlockId + 1, blockSize, numLayers);

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

            // ----- Concatenate flat tokens + run shared embedding -----
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

            // Phase 8: multimodal embedding injection. The pending lists carry
            // (embedding tensor, row-position) tuples; for a serialized-by-engine
            // multimodal request the row-position lines up with the row in the
            // batched [numTokens, hidden] tensor. InjectMultimodalEmbeddings is
            // a row-wise Narrow+Copy so it works on the batched tensor the same
            // way it works on a single-seq tensor in the legacy Forward path.
            if (_pendingVisionEmbeddings.Count > 0 || _pendingAudioEmbeddings.Count > 0)
            {
                foreach (var (emb, pos) in _pendingVisionEmbeddings)
                {
                    InjectMultimodalEmbeddings(hiddenStates, emb, pos);
                    emb.Dispose();
                }
                _pendingVisionEmbeddings.Clear();
                foreach (var (emb, pos) in _pendingAudioEmbeddings)
                {
                    InjectMultimodalEmbeddings(hiddenStates, emb, pos);
                    emb.Dispose();
                }
                _pendingAudioEmbeddings.Clear();
            }

            // ----- Per-layer transformer with hybrid dispatch -----
            int headDim = Config.HeadDim;
            for (int layer = 0; layer < numLayers; layer++)
            {
                switch (_layerTypes[layer])
                {
                    case LayerType.Attention:
                        hiddenStates = RunBatchedAttentionLayer(
                            hiddenStates, layer, numTokens, numSeqs,
                            queryStartLoc, slotMapping, seqLens, positions,
                            ctx.BlockTables, headDim);
                        break;

                    case LayerType.Mamba2:
                        hiddenStates = RunBatchedMamba2Layer(
                            hiddenStates, ctx, layer, numTokens, numSeqs, queryStartLoc);
                        break;

                    case LayerType.FFN:
                        // FFN is stateless. Dense FFN (ReLU²) is naturally
                        // token-parallel on the batched [numTokens, hidden]
                        // tensor. MoE FFN is per-token (router → top-k expert
                        // gather → weighted sum) — the existing MoEForward
                        // already iterates tokens internally, so we hand it
                        // the full batched tensor with seqLen=numTokens.
                        hiddenStates = _numExperts > 0
                            ? RunBatchedMoELayer(hiddenStates, layer, numTokens)
                            : RunBatchedFFNLayer(hiddenStates, layer);
                        break;
                }
            }

            // ----- Final norm + per-sequence LM head -----
            Tensor finalNormed = RMSNormOp(hiddenStates, "output_norm.weight");
            hiddenStates.Dispose();

            int hidden = Config.HiddenSize;
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

        // ----- Phase 3: Attention layer batched compute (paged K/V) -----
        //
        // Nemotron-H attention layers have NO RoPE — the model relies on
        // Mamba2 for positional information, so this is simpler than Mistral 3
        // (which carries YaRN scaling + per-layer ropeBase). Q/K/V projection
        // over the batched input, scatter K/V into paged buffers, paged-attention
        // gather, output projection + residual add.
        private Tensor RunBatchedAttentionLayer(
            Tensor hiddenStates, int layer, int numTokens, int numSeqs,
            int[] queryStartLoc, int[] slotMapping, int[] seqLens, int[] positions,
            int[][] blockTables, int headDim)
        {
            int numHeads = _layerNumHeads[layer];
            int numKVHeads = _layerNumKVHeads[layer];
            int qDim = numHeads * headDim;
            int kvDim = numKVHeads * headDim;
            float scale = _attentionScale != 0 ? (float)_attentionScale : 1.0f / MathF.Sqrt(headDim);
            string prefix = _layerPrefixes[layer];

            // Snapshot the input as residual (the layer block adds its
            // attention output back into the residual stream).
            Tensor residual = Ops.NewContiguous(hiddenStates);
            Tensor normed = RMSNormOp(hiddenStates, prefix + "attn_norm.weight");
            hiddenStates.Dispose();

            // Q/K/V projection — fused if the weight exists, else three separate linears.
            Tensor qTensor, kTensor, vTensor;
            Tensor qkvFused = LinearForward(normed, prefix + "attn_qkv.weight");
            if (qkvFused != null)
            {
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
                qTensor = LinearForward(normed, prefix + "attn_q.weight");
                kTensor = LinearForward(normed, prefix + "attn_k.weight");
                vTensor = LinearForward(normed, prefix + "attn_v.weight");
            }
            normed.Dispose();

            // Scatter new K/V into the paged buffers via slotMapping.
            float[] kFlat = kTensor.GetElementsAsFloat(numTokens * kvDim);
            float[] vFlat = vTensor.GetElementsAsFloat(numTokens * kvDim);
            PagedKvBatchOps.ScatterKv(
                kFlat, vFlat, _nemoPagedK[layer], _nemoPagedV[layer],
                slotMapping, numTokens, numKVHeads, headDim, _nemoPagedBlockSize);
            kTensor.Dispose();
            vTensor.Dispose();

            // Paged attention forward — native GGML kernel covers the gather +
            // flash-attn step in one dispatch. Same entry point Mistral 3 uses.
            float[] qFlat = qTensor.GetElementsAsFloat(numTokens * qDim);
            qTensor.Dispose();
            float[] attnFlat = new float[numTokens * qDim];

            if (IsGgmlBackend)
            {
                var (blockTableFlat, blockTableOffsets) = FlattenBlockTables(blockTables);
                TensorSharp.GGML.GgmlBasicOps.PagedAttentionForward(
                    qFlat, _nemoPagedK[layer], _nemoPagedV[layer], attnFlat,
                    queryStartLoc, seqLens, positions,
                    blockTableFlat, blockTableOffsets,
                    numSeqs, numTokens, numHeads, numKVHeads, headDim,
                    _nemoPagedBlockSize, scale);
            }
            else
            {
                // Non-GGML backend fallback: pure C# managed paged attention.
                ManagedPagedAttention.Forward(
                    qFlat, _nemoPagedK[layer], _nemoPagedV[layer], attnFlat,
                    numTokens, numHeads, numKVHeads, headDim, _nemoPagedBlockSize,
                    queryStartLoc, seqLens, positions, blockTables, numSeqs,
                    scale, causal: true);
            }

            using Tensor attnOut = CreateFloatTensor(attnFlat, numTokens, qDim);
            Tensor attnProj = LinearForward(attnOut, prefix + "attn_output.weight");
            Ops.Add(residual, residual, attnProj);
            attnProj.Dispose();
            return residual;
        }

        // ----- Phase 4: FFN layer batched compute (token-parallel ReLU²) -----
        // Nemotron's dense FFN is token-parallel: LinearForward(up) + ReluSquared
        // + LinearForward(down) all work on batched [numTokens, hidden] input
        // unchanged.
        private Tensor RunBatchedFFNLayer(Tensor hiddenStates, int layer)
        {
            string prefix = _layerPrefixes[layer];
            Tensor residual = Ops.NewContiguous(hiddenStates);
            Tensor normed = RMSNormOp(hiddenStates, prefix + "attn_norm.weight");
            hiddenStates.Dispose();

            Tensor up = LinearForward(normed, prefix + "ffn_up.weight");
            normed.Dispose();
            ReluSquaredInPlace(up);
            Tensor down = LinearForward(up, prefix + "ffn_down.weight");
            up.Dispose();
            Ops.Add(residual, residual, down);
            down.Dispose();
            return residual;
        }

        // ----- Phase 7: MoE FFN layer batched compute -----
        //
        // Nemotron's MoEForward is internally a per-token loop (router →
        // sigmoid + bias → top-k → weighted expert sum → optional latent
        // in/out + shared experts). All cross-token state is scratch the
        // function resets per iteration, so passing it the full batched
        // [numTokens, hidden] tensor with seqLen=numTokens treats every
        // token independently — which is exactly the semantics we need
        // for continuous batching (each token's experts are picked from
        // its own router logits row).
        //
        // The legacy prefill optimisations (TryMoEPrefillFusedReluSquared,
        // TryMoEPrefillBatchedByExpert) trigger automatically when seqLen>1,
        // so the batched-decode case (n>1, each seq 1 token) gets the
        // batched-by-expert routed-input path for free instead of the
        // per-step single-token routing.
        private Tensor RunBatchedMoELayer(Tensor hiddenStates, int layer, int numTokens)
        {
            string prefix = _layerPrefixes[layer];
            Tensor residual = Ops.NewContiguous(hiddenStates);
            Tensor normed = RMSNormOp(hiddenStates, prefix + "attn_norm.weight");
            hiddenStates.Dispose();

            // isDecode=false even when numTokens==1 from the batched path —
            // the prefill optimisations cleanly handle seqLen==1 too, and
            // the decode-specific fast paths (CanMoEDecodeResidualAdd) assume
            // residual==input which isn't the case here.
            Tensor moeOut = MoEForward(normed, layer, prefix, numTokens, isDecode: false, residual: null);
            normed.Dispose();

            if (moeOut != null)
            {
                Ops.Add(residual, residual, moeOut);
                moeOut.Dispose();
            }
            return residual;
        }

        // ----- Phase 5: Mamba2 layer batched via per-seq state-swap -----
        //
        // Mamba2's recurrence (causal conv1d + selective state update) is
        // sequential within a sequence, so we walk sequences one at a time
        // and swap in each slot's persistent conv + SSM state via reference
        // assignment on the model-level pointers. Mirrors Qwen 3.5 Phase 5c
        // ("reference-swap, not memcpy") — same proven pattern.
        //
        // Each seq's slice runs through Mamba2Block which reads/writes
        // _convState[layer], _ssmState[layer], and the native-decode state
        // tensors. Restored after the loop so any legacy Forward path
        // sharing this model sees the original scratch instances.
        //
        // True kernel-level batched Mamba2 (vLLM's mamba_chunk_scan_combined_varlen
        // with cu_seqlens + state_indices) is a follow-up requiring either a
        // new native op (mirrors Phase 7 of the Qwen 3.5 work) or wrapping
        // the existing single-seq native kernel with a state-indices dispatch.
        // Opt-in for the native batched Mamba2 kernel (Phase 9). Default OFF;
        // set TS_NEMOTRON_MAMBA2_BATCHED_NATIVE=1 to route through the new path.
        // Property (not static readonly) so tests can toggle after class load —
        // same reason NemoBatchedOptIn() is a method.
        private static bool NemoMamba2BatchedNative() =>
            string.Equals(Environment.GetEnvironmentVariable("TS_NEMOTRON_MAMBA2_BATCHED_NATIVE"),
                          "1", StringComparison.Ordinal);

        private Tensor RunBatchedMamba2Layer(
            Tensor hiddenStates, BatchedForwardContext ctx, int layer,
            int numTokens, int numSeqs, int[] queryStartLoc)
        {
            if (NemoMamba2BatchedNative() && IsGgmlBackend)
                return RunBatchedMamba2LayerNative(
                    hiddenStates, ctx, layer, numTokens, numSeqs, queryStartLoc);
            return RunBatchedMamba2LayerPerSeq(
                hiddenStates, ctx, layer, numTokens, numSeqs, queryStartLoc);
        }

        private Tensor RunBatchedMamba2LayerPerSeq(
            Tensor hiddenStates, BatchedForwardContext ctx, int layer,
            int numTokens, int numSeqs, int[] queryStartLoc)
        {
            int hidden = Config.HiddenSize;

            float[] origConvState  = _convState[layer];
            float[] origSsmState   = _ssmState[layer];
            Tensor  origNdProj     = _mamba2NativeDecodeProjected[layer];
            Tensor  origNdHidden   = _mamba2NativeDecodeHidden[layer];
            bool    origNdInit     = _mamba2NativeDecodeStateInitialized[layer];
            int     savedCacheLen  = _cacheSeqLen;

            float[] batchedOut = new float[numTokens * hidden];
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
                        _nemoSlotInit[layer][slot] = false;
                    EnsureNemoSlotAllocated(layer, slot);

                    _convState[layer]                       = _nemoSlotConvBuf[layer][slot];
                    _ssmState[layer]                        = _nemoSlotSsmState[layer][slot];
                    _mamba2NativeDecodeProjected[layer]     = _nemoSlotMamba2NativeDecodeProjected[layer][slot];
                    _mamba2NativeDecodeHidden[layer]        = _nemoSlotMamba2NativeDecodeHidden[layer][slot];
                    _mamba2NativeDecodeStateInitialized[layer] =
                        _nemoSlotMamba2NativeDecodeStateInitialized[layer][slot];
                    _cacheSeqLen = seq.NumComputedTokens;

                    using Tensor seqHidden = Ops.NewContiguous(hiddenStates.Narrow(0, seqStart, seqLen));
                    bool isDecode = seqLen == 1;
                    using Tensor seqOut = Mamba2Block(seqHidden, layer, seqLen, isDecode);

                    // Mamba2Block mutates state through the swapped references;
                    // read back the native-decode init flag (the only by-value
                    // field) into the per-slot store.
                    _nemoSlotMamba2NativeDecodeStateInitialized[layer][slot] =
                        _mamba2NativeDecodeStateInitialized[layer];

                    float[] seqFlat = seqOut.GetElementsAsFloat(seqLen * hidden);
                    Buffer.BlockCopy(seqFlat, 0,
                        batchedOut, seqStart * hidden * sizeof(float),
                        seqLen * hidden * sizeof(float));
                }
            }
            finally
            {
                _convState[layer]                              = origConvState;
                _ssmState[layer]                               = origSsmState;
                _mamba2NativeDecodeProjected[layer]            = origNdProj;
                _mamba2NativeDecodeHidden[layer]               = origNdHidden;
                _mamba2NativeDecodeStateInitialized[layer]     = origNdInit;
                _cacheSeqLen                                   = savedCacheLen;
            }

            hiddenStates.Dispose();
            return CreateFloatTensor(batchedOut, numTokens, hidden);
        }

        // Phase 9 (Nemotron native Mamba2): replace the N per-seq calls into
        // Mamba2Block with one native batched-step call that walks every
        // (seq, token) pair using the per-slot conv FIFO + SSM state pointers.
        // Per-seq path:
        //   1. RMSNorm + ssm_in matmul (input projection)
        //   2. Concatenate into batched [numTokens, dInProjTotal] CPU buffer
        // Native:
        //   3. One call into TSGgml_NemotronMamba2BatchedStepF32 — conv1d +
        //      SiLU + SSM scan (heads in GCD parallel) + SwiGLU + group RMSNorm
        // Batched:
        //   4. Output projection ssm_out matmul + residual add
        //
        // Math mirrors Mamba2SSMStepSIMD + Mamba2Conv1dStep faithfully. The
        // win is replacing N C# → C transitions with one, plus opportunity
        // for cross-head parallelism that C# Parallel.For doesn't always hit
        // for small head counts.
        private unsafe Tensor RunBatchedMamba2LayerNative(
            Tensor hiddenStates, BatchedForwardContext ctx, int layer,
            int numTokens, int numSeqs, int[] queryStartLoc)
        {
            int dInner = _ssmDInner;
            int dState = _ssmDState;
            int nHead = _ssmNHead;
            int headDim = _ssmHeadDim;
            int nGroup = _ssmNGroup;
            int dConv = _ssmDConv;
            int xbcSize = dInner + 2 * nGroup * dState;
            int dInProjTotal = 2 * dInner + 2 * nGroup * dState + nHead;
            int hidden = Config.HiddenSize;
            string prefix = _layerPrefixes[layer];

            // Residual is the input. Norm + ssm_in matmul produce the per-seq
            // projected packed input; we accumulate into one batched buffer.
            Tensor residual = Ops.NewContiguous(hiddenStates);
            Tensor normed = RMSNormOp(hiddenStates, prefix + "attn_norm.weight");
            hiddenStates.Dispose();

            // Project the entire batched [numTokens, hidden] in one matmul —
            // ssm_in is the heaviest matmul in the Mamba2 layer.
            Tensor projected = LinearForward(normed, prefix + "ssm_in.weight");
            normed.Dispose();
            float[] packedBatched = projected.GetElementsAsFloat(numTokens * dInProjTotal);
            projected.Dispose();

            // Build per-seq descriptors. Pin per-slot conv + ssm state float[]
            // for the duration of the native call.
            var descs = new NemoMamba2BatchedSeqDesc[numSeqs];
            var convHandles = new GCHandle[numSeqs];
            var ssmHandles  = new GCHandle[numSeqs];
            float[] outBatched = new float[numTokens * dInner];
            try
            {
                for (int s = 0; s < numSeqs; s++)
                {
                    var seq = ctx.Sequences[s];
                    int slot = seq.BlockTable.Blocks[0].Id;
                    int seqStart = queryStartLoc[s];
                    int seqLen = ctx.NumScheduledTokens[s];

                    if (seq.NumComputedTokens == 0)
                        _nemoSlotInit[layer][slot] = false;
                    EnsureNemoSlotAllocated(layer, slot);

                    float[] convBuf = _nemoSlotConvBuf[layer][slot];
                    float[] ssmBuf  = _nemoSlotSsmState[layer][slot];
                    convHandles[s] = GCHandle.Alloc(convBuf, GCHandleType.Pinned);
                    ssmHandles[s]  = GCHandle.Alloc(ssmBuf,  GCHandleType.Pinned);

                    descs[s].SeqStart  = seqStart;
                    descs[s].SeqLen    = seqLen;
                    descs[s].Pad0      = 0;
                    descs[s].Pad1      = 0;
                    descs[s].ConvState = convHandles[s].AddrOfPinnedObject();
                    descs[s].SsmState  = ssmHandles[s].AddrOfPinnedObject();
                }

                // Weight pointers. _mamba2ConvWT[layer] is the transposed conv
                // weight [dConv, xbcSize] (matches what Mamba2Conv1dStepVectorized
                // uses); fall back to the raw tensor when the transpose hasn't
                // been built (legacy fallback).
                float[] convWT = _mamba2ConvWT?[layer];
                float* convBiasPtr = _weights.TryGetValue(prefix + "ssm_conv1d.bias", out var cb)
                    ? GetFloatPtr(cb) : null;
                float* dtBiasPtr = GetFloatPtr(_weights[prefix + "ssm_dt.bias"]);
                float* aPtr      = GetFloatPtr(_weights[prefix + "ssm_a"]);
                float* dPtr      = _weights.TryGetValue(prefix + "ssm_d", out var dT) ? GetFloatPtr(dT) : null;
                float* ssmNormPtr = _weights.TryGetValue(prefix + "ssm_norm.weight", out var normW)
                    ? GetFloatPtr(normW) : null;

                if (convWT == null)
                {
                    // Bail to per-seq C# path if the transposed conv weight
                    // hasn't been built — the native kernel's conv math
                    // assumes that layout.
                    for (int s = 0; s < numSeqs; s++)
                    {
                        if (convHandles[s].IsAllocated) convHandles[s].Free();
                        if (ssmHandles[s].IsAllocated) ssmHandles[s].Free();
                    }
                    return RunBatchedMamba2LayerPerSeq(
                        residual, ctx, layer, numTokens, numSeqs, queryStartLoc);
                }

                fixed (float* convWtPin = convWT,
                               packedPin = packedBatched,
                               outPin = outBatched)
                {
                    GgmlBasicOps.NemotronMamba2BatchedStep(
                        descs, numTokens,
                        (IntPtr)packedPin, dInProjTotal,
                        dInner, dState, nHead, headDim, nGroup, dConv,
                        (IntPtr)convWtPin, (IntPtr)convBiasPtr,
                        (IntPtr)dtBiasPtr, (IntPtr)aPtr,
                        (IntPtr)dPtr, (IntPtr)ssmNormPtr,
                        Config.Eps, (IntPtr)outPin);
                }
            }
            finally
            {
                for (int s = 0; s < numSeqs; s++)
                {
                    if (convHandles[s].IsAllocated) convHandles[s].Free();
                    if (ssmHandles[s].IsAllocated) ssmHandles[s].Free();
                }
            }

            // Output projection + residual.
            using Tensor outTensor = CreateFloatTensor(outBatched, numTokens, dInner);
            Tensor ssmOut = LinearForward(outTensor, prefix + "ssm_out.weight");
            Ops.Add(residual, residual, ssmOut);
            ssmOut.Dispose();
            return residual;
        }

        // Flatten per-seq block tables into a single int[] + offsets[] for the
        // native paged-attention entry point. Same packing Mistral 3 uses.
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
    }
}
