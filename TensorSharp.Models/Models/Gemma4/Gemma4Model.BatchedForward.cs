// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Gemma 4 batched paged-attention forward. Mirrors vLLM's per-layer
// pattern (gemma4.py): each layer carries its own attention metadata
// (sliding_window, head_dim, num_kv_heads, RoPE base/dim), the forward
// loop dispatches per-layer using those values. Per-Layer Embeddings
// (PLE) are computed once and injected after each layer's residual.
//
// Coverage in this iteration:
//   - Per-layer SWA dispatch via sliding_window param to native kernel
//     (kernel accepts a per-call slidingWindow argument)
//   - Per-layer head dim + num_kv_heads (per-layer paged buffers)
//   - NeoX-style RoPE per layer with local vs global ropeBase + ropeDims
//   - Q/K/V projection with Q-norm, K-norm, V-norm (V-norm unweighted)
//   - Dense FFN with Gemma-style pre/post FFN norms
//   - PLE injection (inp_gate -> GELUMul -> proj -> post_norm)
//   - Per-layer scalar multiplication
//   - Embedding scaling (Gemma's hidden_size^0.5)
//   - Final logit softcap
//
// NotSupported (throws -> per-seq fallback handles them):
//   - MoE layers (HasMoE(l)) - the legacy fused-MoE path is highly
//     optimised; replicating it for batched would itself be multi-day
//   - KV donor layers (isShared) - K/V sharing across layers requires
//     refcount-based block aliasing in the BlockPool, separate piece
//   - Multimodal embedding queued - per-sequence injector state not
//     batched-safe
//   - Q8_0 KV cache (Q8_0 needs the fused native kernels)
using System;
using System.Collections.Generic;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models.Paged;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models
{
    public partial class Gemma4Model : IBatchedPagedModel
    {
        // Per-layer paged K/V floats. Size depends on the layer's head
        // dimension and num_kv_heads (which CAN vary across layers in
        // Gemma 4).
        private float[][] _g4PagedK;
        private float[][] _g4PagedV;
        private int _g4PagedNumBlocks;
        private int _g4PagedBlockSize;
        private int[] _g4PagedKvDimPerLayer; // num_kv_heads * head_dim per layer

        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx)
        {
            if (ctx == null) throw new ArgumentNullException(nameof(ctx));
            int numSeqs = ctx.Sequences.Count;
            if (numSeqs == 0) return Array.Empty<float[]>();

            // Same rationale as Forward(): on MLX, collapse the entire batch
            // forward into one worker round-trip so the hundreds of internal
            // MlxWorker.Invoke calls run inline. Saves ~25µs × ~600 ops per
            // sequence × N sequences of pure queue hand-off overhead.
            if (_backend == BackendType.Mlx && !TensorSharp.MLX.MlxWorker.Shared.IsOnWorkerThread)
                return TensorSharp.MLX.MlxWorker.Shared.Invoke(() => ForwardBatchCore(ctx));
            return ForwardBatchCore(ctx);
        }

        private IReadOnlyList<float[]> ForwardBatchCore(BatchedForwardContext ctx)
        {
            int numSeqs = ctx.Sequences.Count;

            // The batched/paged graph (prefill on the server, or multi-seq decode)
            // grows ggml-cuda's compute scratch pool, which can move the addresses
            // baked into the persistent CUDA-graph-captured single-token decode
            // graph. Drop it so the next per-seq fused Forward rebuilds + recaptures
            // against the current pool state (see Gemma4ResetDecodeCache).
            if (_backend == BackendType.GgmlCuda || _backend == BackendType.GgmlVulkan)
            {
                GgmlBasicOps.Gemma4ResetDecodeCache();
                GgmlBasicOps.Gemma4ResetBatchedDecodeCache();
                GgmlBasicOps.Gemma4MoEResetDecodeCache();
                GgmlBasicOps.Gemma4ResetMoEBatchedDecodeCache();
            }

            // Disable gate. The batched path is now the DEFAULT for Gemma 4
            // (correctness verified against legacy through 42 layers; see
            // Gemma4BatchedForwardTests). Set TS_GEMMA4_BATCHED=0 to opt OUT
            // and force the per-seq KV-swap fallback - useful only for
            // bisecting performance or debugging the batched kernel itself.
            // The fallback can't serve sequences past the SWA window (512
            // tokens) because the per-sequence KV-swap requires snapshot,
            // and snapshot is only well-defined inside the linear window,
            // so the default needs to be the batched path that has no
            // such limit.
            string optOut = Environment.GetEnvironmentVariable("TS_GEMMA4_BATCHED");
            if (string.Equals(optOut, "0", StringComparison.Ordinal) ||
                string.Equals(optOut, "false", StringComparison.OrdinalIgnoreCase))
                throw new NotSupportedException(
                    "Gemma 4 batched: disabled via TS_GEMMA4_BATCHED=0.");

            if (_pendingVisionEmbeddingsList.Count > 0 || _pendingAudioEmbeddingsList.Count > 0)
                throw new NotSupportedException(
                    "Gemma 4 batched: multimodal embeddings pending; per-seq fallback.");

            int numLayers = Config.NumLayers;
            for (int l = 0; l < numLayers; l++)
            {
                if (HasMoE(l))
                    throw new NotSupportedException(
                        $"Gemma 4 batched: layer {l} is MoE; per-seq fallback.");
            }
            // KV donor map: handled by aliasing the receiver layer's
            // paged buffer to the donor's. EnsureGemma4PagedBuffers
            // populates the aliases below.
            if (_kvCacheDtype.IsBlockQuantized())
                throw new NotSupportedException(
                    "Gemma 4 batched: Q8_0 KV cache requires the fused native " +
                    "kernels; per-seq fallback.");

            int hidden = Config.HiddenSize;
            int numHeads = Config.NumHeads;

            int blockSize = ctx.Sequences[0].BlockTable.BlockSize;
            int maxBlockId = 0;
            for (int s = 0; s < numSeqs; s++)
            {
                var bt = ctx.BlockTables[s];
                for (int b = 0; b < bt.Length; b++)
                    if (bt[b] > maxBlockId) maxBlockId = bt[b];
            }
            EnsureGemma4PagedBuffers(maxBlockId + 1, blockSize, numLayers);

            int numTokens = 0;
            for (int s = 0; s < numSeqs; s++) numTokens += ctx.NumScheduledTokens[s];

            int[] positions = ctx.Positions.ToArray();
            int[] queryStartLoc = ctx.QueryStartLoc.ToArray();
            int[] slotMappingShared = ctx.SlotMapping.ToArray();
            int[] seqLens = new int[numSeqs];
            for (int s = 0; s < numSeqs; s++)
                seqLens[s] = ctx.Sequences[s].NumComputedTokens + ctx.NumScheduledTokens[s];

            // Speculative verify batches forward drafted tokens that are not (yet)
            // in the sequence's token list, so the caller supplies them directly.
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
            var (blockTableFlat, blockTableOffsets) = FlattenBlockTables(ctx.BlockTables);

            Tensor hiddenStates = Embedding(flatTokens);
            ScaleEmbedding(hiddenStates);

            bool dumpDiag = Environment.GetEnvironmentVariable("TS_GEMMA4_DIAG") == "1";
            if (dumpDiag) Console.WriteLine($"[g4-batched] after-embed: {TensorChecksum(hiddenStates, "embed")}");

            // PLE: compute once, slice per layer.
            Tensor perLayerInputs = null;
            if (_pleDim > 0)
                perLayerInputs = ComputePLE(flatTokens, hiddenStates, numTokens);

            for (int layer = 0; layer < numLayers; layer++)
            {
                string prefix = $"blk.{layer}";
                bool isLocal = IsLocalLayer(layer);
                bool isShared = _kvDonorMap != null && _kvDonorMap.ContainsKey(layer);
                int hd = HeadDimForLayer(layer);
                int kvHeads = KVHeadsForLayer(layer);
                int qDim = numHeads * hd;
                int kDim = kvHeads * hd;
                int slidingWindow = isLocal ? _slidingWindow : 0;
                // Gemma 4 folds the 1/sqrt(head_dim) factor into the
                // Q-norm weights, so attention compute uses scale=1.
                // Matches the legacy FusedPrefillAttention call which
                // passes 1.0f as the scale.
                float scale = 1.0f;
                var (ropeBase, ropeDims) = RopeForLayer(layer);

                Tensor residual = Ops.NewContiguous(hiddenStates);
                Tensor normed = RMSNormOp(hiddenStates, $"{prefix}.attn_norm.weight");
                hiddenStates.Dispose();

                // QKV projection. Shared (donor receiver) layers do NOT
                // compute K or V - they read from the donor layer's
                // paged buffer (which we aliased above) via the same
                // _g4PagedK[layer] pointer.
                Tensor q, k = null, v = null;
                string qkvName = $"{prefix}.attn_qkv.weight";
                bool useFusedQKV = !isShared &&
                    (_quantWeights.ContainsKey(qkvName) || _weights.ContainsKey(qkvName));
                if (useFusedQKV)
                {
                    Tensor qkv = LinearForward(normed, qkvName);
                    int vDim = (int)qkv.Sizes[1] - qDim - kDim;
                    using (var qView = qkv.Narrow(1, 0, qDim))
                        q = Ops.NewContiguous(qView);
                    using (var kView = qkv.Narrow(1, qDim, kDim))
                        k = Ops.NewContiguous(kView);
                    using (var vView = qkv.Narrow(1, qDim + kDim, vDim))
                        v = Ops.NewContiguous(vView);
                    qkv.Dispose();
                }
                else
                {
                    q = LinearForward(normed, $"{prefix}.attn_q.weight");
                    if (!isShared)
                    {
                        k = LinearForward(normed, $"{prefix}.attn_k.weight");
                        if (_weights.ContainsKey($"{prefix}.attn_v.weight") ||
                            _quantWeights.ContainsKey($"{prefix}.attn_v.weight"))
                        {
                            v = LinearForward(normed, $"{prefix}.attn_v.weight");
                        }
                        else
                        {
                            v = new Tensor(_allocator, DType.Float32, k.Sizes);
                            Ops.Copy(v, k);
                        }
                    }
                }
                normed.Dispose();

                // Q-norm always; K-norm + V-norm only when this layer has
                // its own K/V (i.e. not a donor receiver).
                q = ApplyHeadwiseRMSNorm(q, $"{prefix}.attn_q_norm.weight", numHeads, numTokens, hd);
                if (!isShared)
                {
                    k = ApplyHeadwiseRMSNorm(k, $"{prefix}.attn_k_norm.weight", kvHeads, numTokens, hd);
                    ApplyUnweightedRMSNorm(v, kvHeads, hd, numTokens);
                }

                // NeoX RoPE: always on Q, on K only if this layer computed K.
                // Global layers also apply per-dim freq_factors scaling
                // ("rope_freqs.weight") to match the legacy Forward path.
                Tensor ropeFreqFactors =
                    !isLocal && _weights.TryGetValue("rope_freqs.weight", out var rff) ? rff : null;
                using (var posTensorQ = BuildRoPEPositionsTensor(positions, numHeads))
                {
                    q = ApplyBatchedRoPENeoX(q, posTensorQ, numTokens, numHeads, hd, ropeDims, ropeBase, ropeFreqFactors);
                }
                if (!isShared)
                {
                    using (var posTensorK = BuildRoPEPositionsTensor(positions, kvHeads))
                    {
                        k = ApplyBatchedRoPENeoX(k, posTensorK, numTokens, kvHeads, hd, ropeDims, ropeBase, ropeFreqFactors);
                    }

                    // Scatter K, V into the (donor or own) paged buffer.
                    float[] kFlat = k.GetElementsAsFloat(numTokens * kDim);
                    float[] vFlat = v.GetElementsAsFloat(numTokens * kDim);
                    PagedKvBatchOps.ScatterKv(
                        kFlat, vFlat, _g4PagedK[layer], _g4PagedV[layer],
                        slotMappingShared, numTokens, kvHeads, hd, _g4PagedBlockSize);
                    k.Dispose();
                    v.Dispose();
                }

                // Paged attention with this layer's sliding_window. Native
                // paged attention is only valid when a GGML backend owns the
                // model; direct CUDA/MLX/CPU runs fall back to managed
                // attention to avoid the native bridge's default backend.
                Tensor attnOut;
                if (IsGgmlBackend)
                {
                    // GPU-resident path: hand the kernel Q's tensor storage
                    // pointer and a pre-allocated output tensor's storage
                    // pointer so it can zero-copy bind them. This eliminates
                    // the per-layer GetElementsAsFloat -> ggml_backend_synchronize
                    // drain (which was waiting for the layer's projections /
                    // RoPE / scatter) and the matching CreateFloatTensor
                    // upload on the way out — both gone. Result stays on
                    // GPU, the next LinearForward consumes it directly and
                    // serialises behind us on the Metal/CUDA command queue.
                    attnOut = new Tensor(_allocator, DType.Float32, numTokens, qDim);
                    GgmlBasicOps.PagedAttentionForwardDevice(
                        q.Storage.PtrAtElement(q.StorageOffset),
                        _g4PagedK[layer], _g4PagedV[layer],
                        attnOut.Storage.PtrAtElement(attnOut.StorageOffset),
                        queryStartLoc, seqLens, positions,
                        blockTableFlat, blockTableOffsets,
                        numSeqs, numTokens, numHeads, kvHeads, hd,
                        _g4PagedBlockSize, scale, slidingWindow);
                    q.Dispose();
                }
                else
                {
                    // Non-GGML backends: managed scalar kernel walking host
                    // float[] gather of paged K/V. Slow but portable; only
                    // hit on the CUDA / MLX / pure-CPU paths.
                    float[] qFlat = q.GetElementsAsFloat(numTokens * qDim);
                    q.Dispose();
                    float[] attnFlat = new float[numTokens * qDim];
                    ManagedPagedAttention.Forward(
                        qFlat, _g4PagedK[layer], _g4PagedV[layer], attnFlat,
                        numTokens, numHeads, kvHeads, hd, _g4PagedBlockSize,
                        queryStartLoc, seqLens, positions, ctx.BlockTables, numSeqs,
                        scale, causal: true, slidingWindow: slidingWindow);
                    attnOut = CreateFloatTensor(attnFlat, numTokens, qDim);
                }
                Tensor attnProj = LinearForward(attnOut, $"{prefix}.attn_output.weight");
                attnOut.Dispose();

                // Post-attention norm + residual.
                Ops.RMSNorm(attnProj, attnProj,
                    _weights[$"{prefix}.post_attention_norm.weight"], null, Config.Eps);
                Ops.Add(residual, residual, attnProj);
                attnProj.Dispose();

                // Dense FFN: Gemma 4 uses GELU-gated (not SiLU). Use the
                // model's FFNGeluWithOptionalNorm helper which fuses
                // pre-norm + gate_up linear + GELUMul split + down linear.
                Tensor ffnOut = FFNGeluWithOptionalNorm(
                    residual,
                    $"{prefix}.ffn_norm.weight",
                    $"{prefix}.ffn_gate_up.weight",
                    $"{prefix}.ffn_down.weight",
                    numTokens);
                string postFfnNormKey = _weights.ContainsKey($"{prefix}.post_ffw_norm.weight")
                    ? $"{prefix}.post_ffw_norm.weight"
                    : $"{prefix}.ffn_post_norm.weight";
                Ops.RMSNorm(ffnOut, ffnOut, _weights[postFfnNormKey], null, Config.Eps);
                Ops.Add(residual, residual, ffnOut);
                ffnOut.Dispose();

                // PLE injection.
                if (perLayerInputs != null &&
                    (_weights.ContainsKey($"{prefix}.inp_gate.weight") || _quantWeights.ContainsKey($"{prefix}.inp_gate.weight")))
                {
                    using var perLayerSlice = ExtractPerLayerSlice(perLayerInputs, layer, numTokens);
                    using var gate = LinearForward(residual, $"{prefix}.inp_gate.weight");
                    if (gate != null && perLayerSlice != null)
                    {
                        Ops.GELUMul(gate, gate, perLayerSlice);
                        using var pleProj = LinearForward(gate, $"{prefix}.proj.weight");
                        if (pleProj != null)
                        {
                            Ops.RMSNorm(pleProj, pleProj,
                                _weights[$"{prefix}.post_norm.weight"], null, Config.Eps);
                            Ops.Add(residual, residual, pleProj);
                        }
                    }
                }

                if (_layerScalars != null && _layerScalars[layer] != 1f)
                    Ops.Mul(residual, residual, _layerScalars[layer]);

                hiddenStates = residual;
                if (dumpDiag) Console.WriteLine($"[g4-batched] after-layer-{layer}: {TensorChecksum(hiddenStates, $"L{layer}")}");
            }

            // Final norm + LM head.
            Tensor finalNormed = RMSNormOp(hiddenStates, "output_norm.weight");
            hiddenStates.Dispose();

            string outputWeight = _hasTiedOutput ? "token_embd.weight" : "output.weight";
            int vocab = Config.VocabSize;

            // MTP speculative trunk: capture every row's post-final-norm hidden
            // (h_nextn) and/or every row's logits, instead of just the last token
            // per sequence. The verify batch is one sequence with K+1 rows.
            if (ctx.CaptureHiddenAll != null)
            {
                float[] hAll = finalNormed.GetElementsAsFloat(numTokens * hidden);
                Array.Copy(hAll, 0, ctx.CaptureHiddenAll, 0,
                    Math.Min(hAll.Length, ctx.CaptureHiddenAll.Length));
            }
            if (ctx.CaptureLogitsAll != null)
            {
                Tensor allLogitsT = LinearForward(finalNormed, outputWeight);
                finalNormed.Dispose();
                if (_finalLogitSoftcap > 0f)
                    ApplyLogitSoftcap(allLogitsT);
                float[] allRows = allLogitsT.GetElementsAsFloat(numTokens * vocab);
                allLogitsT.Dispose();
                Array.Copy(allRows, 0, ctx.CaptureLogitsAll, 0,
                    Math.Min(allRows.Length, ctx.CaptureLogitsAll.Length));

                var perSeqAll = new float[numSeqs][];
                for (int s = 0; s < numSeqs; s++)
                {
                    int lastRow = queryStartLoc[s + 1] - 1;
                    var slice = new float[vocab];
                    Buffer.BlockCopy(allRows, lastRow * vocab * sizeof(float),
                                     slice, 0, vocab * sizeof(float));
                    perSeqAll[s] = slice;
                }
                return perSeqAll;
            }

            float[] finalFlat = finalNormed.GetElementsAsFloat(numTokens * hidden);
            finalNormed.Dispose();
            float[] lastTokensPacked = PagedKvBatchOps.GatherLastTokenPerSeq(
                finalFlat, hidden, queryStartLoc, numSeqs);
            Tensor lastHidden = CreateFloatTensor(lastTokensPacked, numSeqs, hidden);

            Tensor logitsTensor = LinearForward(lastHidden, outputWeight);
            lastHidden.Dispose();

            if (_finalLogitSoftcap > 0f)
                ApplyLogitSoftcap(logitsTensor);

            float[] allLogits = logitsTensor.GetElementsAsFloat(numSeqs * vocab);
            logitsTensor.Dispose();

            var perSeq = new float[numSeqs][];
            for (int s = 0; s < numSeqs; s++)
            {
                var slice = new float[vocab];
                Buffer.BlockCopy(allLogits, s * vocab * sizeof(float),
                                 slice, 0, vocab * sizeof(float));
                perSeq[s] = slice;
            }
            return perSeq;
        }

        private Tensor ApplyHeadwiseRMSNorm(Tensor data, string weightName, int numHeads, int numTokens, int hd)
        {
            using var reshaped = data.View(numTokens * numHeads, hd);
            var alpha = _weights[weightName];
            Tensor normed = Ops.RMSNorm(null, reshaped, alpha, null, Config.Eps);
            data.Dispose();
            Tensor result = normed.View(numTokens, numHeads * hd);
            normed.Dispose();
            return result;
        }

        private Tensor ApplyBatchedRoPENeoX(
            Tensor data, Tensor positionsTensor,
            int numTokens, int numHeads, int hd, int ropeDims, float ropeBase,
            Tensor freqFactors)
        {
            using var reshaped = data.View(1, numTokens, numHeads, hd);
            // NeoX mode = 2 in Gemma 4's ApplyRoPEPrefill. `ropeDims` is
            // the partial-rotary dim count (= hd for local layers, =
            // _partialRotaryDims for global layers). Global layers
            // additionally scale frequencies by `rope_freqs.weight` -
            // but that scaling path is currently only implemented in the
            // GGML backend's RoPEExWithFreqFactors helper. On non-GGML
            // backends (e.g. MLX) we fall through to Ops.RoPEEx (no
            // scaling); the resulting K mismatches the legacy Forward
            // path's K only on global layers, and the visible quality
            // hit is small for short prompts. Plumbing freq_factors
            // through Ops.RoPEEx for other backends is a follow-up.
            bool useGgmlFreqFactors = freqFactors != null && reshaped.Storage is TensorSharp.GGML.GgmlStorage;
            if (useGgmlFreqFactors)
            {
                GgmlBasicOps.RoPEExWithFreqFactors(
                    reshaped, reshaped, positionsTensor, freqFactors,
                    ropeDims, 2, 0, ropeBase, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            }
            else
            {
                Ops.RoPEEx(reshaped, reshaped, positionsTensor, ropeDims, 2, 0,
                           ropeBase, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            }
            return data;
        }

        private static string TensorChecksum(Tensor t, string label)
            => DiagChecksum(t, label);

        private Tensor BuildRoPEPositionsTensor(int[] tokenPositions, int numHeads)
        {
            int total = tokenPositions.Length * numHeads;
            int[] expanded = new int[total];
            for (int t = 0; t < tokenPositions.Length; t++)
                for (int h = 0; h < numHeads; h++)
                    expanded[t * numHeads + h] = tokenPositions[t];
            return CreateIntTensor(expanded, total);
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

        /// <summary>
        /// True when this model can be asked to copy a sequence's K/V history
        /// from the legacy linear cache (where <c>Forward()</c> writes it)
        /// into the paged storage (where <c>ForwardBatch</c> reads from). The
        /// batched path already rejects Q8_0 KV cache anyway, so we don't
        /// claim migration support in that case.
        /// </summary>
        public bool SupportsLinearKVMigration =>
            _kvCacheK != null && _kvCacheV != null && !_kvCacheDtype.IsBlockQuantized();

        /// <summary>
        /// Copy <paramref name="owner"/>'s K/V history out of the linear
        /// per-layer cache <c>_kvCacheK</c>/<c>_kvCacheV</c> and into the
        /// paged buffer <c>_g4PagedK</c>/<c>_g4PagedV</c> at slot positions
        /// derived from <c>owner.BlockTable</c>.
        ///
        /// Used by <see cref="BatchExecutor"/> when a sequence that has been
        /// running through the N=1 fast path (which writes only to the
        /// linear cache) is about to be dispatched through
        /// <see cref="ForwardBatch"/> alongside a second sequence. Without
        /// this migration the batched paged-attention kernel would read
        /// zeros for the owner's prior positions and the sequence would
        /// degenerate into a token-repeat loop.
        ///
        /// Layout difference handled here:
        ///   Linear:  [kvHeads, cacheLen, headDim] per layer Tensor
        ///            (post-RoPE K, raw V; F16 by default, F32 also supported).
        ///   Paged:   float[numBlocks * blockSize * kvHeads * headDim] per
        ///            layer; slot s lives at offset s * kvHeads * headDim.
        ///
        /// SWA-local layers store positions circularly at index
        /// (p % slidingWindow); only the last slidingWindow positions are
        /// recoverable. Older positions are masked out by paged attention
        /// anyway, so leaving their paged slots untouched (zero-initialised)
        /// is correct.
        /// </summary>
        public bool TryMigrateLinearKVToPaged(SequenceState owner, int blockSize)
        {
            if (owner == null) return false;
            if (!SupportsLinearKVMigration) return false;
            int ownerTokens = _cacheSeqLen;
            if (ownerTokens <= 0) return true;
            if (owner.BlockTable.NumBlocks <= 0) return false;

            // Make sure any pending device→host syncs of the linear cache
            // are flushed before we read it as floats.
            EnsureKvCacheHostSynchronized();

            // Ensure the paged buffers cover every block id we need to
            // write into. This also preserves any K/V already in paged
            // storage (it's a copy-into-larger-buffer, not a wipe).
            int maxBlockId = 0;
            int numBlocks = owner.BlockTable.NumBlocks;
            for (int b = 0; b < numBlocks; b++)
            {
                int id = owner.BlockTable.Blocks[b].Id;
                if (id > maxBlockId) maxBlockId = id;
            }
            EnsureGemma4PagedBuffers(maxBlockId + 1, blockSize, Config.NumLayers);

            int swa = _slidingWindow;

            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                // Receivers alias the donor's _kvCacheK[layer] and
                // _g4PagedK[layer] pointers, so migrating the donor
                // already covered them.
                if (_kvDonorMap != null && _kvDonorMap.ContainsKey(layer)) continue;

                int kvHeads = KVHeadsForLayer(layer);
                int hd = HeadDimForLayer(layer);
                int cacheLen = _kvCacheSize[layer];
                int stridePaged = kvHeads * hd;
                bool isLocal = IsLocalLayer(layer);

                // Dequantise the layer's K/V into F32 host buffers. F32
                // caches use the existing Tensor.GetElementsAsFloat fast
                // path; F16 caches go through F16ToF32 (the default for
                // Q8_0-weight models — Tensor.GetElementsAsFloat throws
                // "Element type Float16 not supported", so we read the
                // raw ushort* and dequant ourselves, matching the
                // pattern in ExpandKVHeadsF16).
                int totalElems = kvHeads * cacheLen * hd;
                if (!TryReadCacheAsF32(_kvCacheK[layer], totalElems, out float[] kFlat) ||
                    !TryReadCacheAsF32(_kvCacheV[layer], totalElems, out float[] vFlat))
                {
                    return false;
                }
                float[] kPaged = _g4PagedK[layer];
                float[] vPaged = _g4PagedV[layer];

                // For SWA layers older positions have been overwritten in
                // the circular cache. For global layers all positions are
                // available.
                int firstMigratable = (isLocal && swa > 0)
                    ? Math.Max(0, ownerTokens - swa)
                    : 0;

                for (int p = firstMigratable; p < ownerTokens; p++)
                {
                    int srcPos = (isLocal && swa > 0) ? (p % swa) : p;

                    int blockIdx = p / blockSize;
                    int offsetInBlock = p % blockSize;
                    int physBlockId = owner.BlockTable.Blocks[blockIdx].Id;
                    int slot = physBlockId * blockSize + offsetInBlock;
                    int slotOffset = slot * stridePaged;

                    for (int h = 0; h < kvHeads; h++)
                    {
                        int srcOffset = (h * cacheLen + srcPos) * hd;
                        int dstOffset = slotOffset + h * hd;
                        Buffer.BlockCopy(
                            kFlat, srcOffset * sizeof(float),
                            kPaged, dstOffset * sizeof(float),
                            hd * sizeof(float));
                        Buffer.BlockCopy(
                            vFlat, srcOffset * sizeof(float),
                            vPaged, dstOffset * sizeof(float),
                            hd * sizeof(float));
                    }
                }
            }

            return true;
        }

        /// <summary>Read a linear KV cache tensor into a fresh F32 float[].
        /// Handles both F32 (zero-copy via Tensor.GetElementsAsFloat) and
        /// F16 (dequant via F16ToF32). Returns false for any other dtype —
        /// quantised caches (Q8_0) are filtered out earlier by
        /// SupportsLinearKVMigration so this path doesn't see them.</summary>
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

        private void EnsureGemma4PagedBuffers(int numBlocks, int blockSize, int numLayers)
        {
            bool needRebuild = _g4PagedK == null ||
                _g4PagedNumBlocks < numBlocks ||
                _g4PagedBlockSize != blockSize ||
                _g4PagedKvDimPerLayer == null;
            if (!needRebuild) return;

            int targetBlocks = Math.Max(numBlocks, _g4PagedNumBlocks * 2);

            // CRITICAL: preserve existing K/V data when growing. The engine
            // submits sequences over multiple scheduler steps - the first
            // step may schedule one sequence (allocating a single-block
            // buffer here), and the second step may add more sequences
            // (requiring more blocks). The new arrivals' decode tokens
            // depend on the FIRST sequence's prefill K/V still being in
            // the buffer. A naive `_g4PagedK = new float[][]` rebuild
            // would zero the buffer and produce garbage tokens (the
            // "first-sequence-in-a-multi-seq-batch loops on a single
            // token" failure mode). Allocate fresh storage, then copy
            // the old contents in.
            float[][] oldPagedK = _g4PagedK;
            float[][] oldPagedV = _g4PagedV;
            int oldNumBlocks = _g4PagedNumBlocks;
            int oldBlockSize = _g4PagedBlockSize;

            _g4PagedK = new float[numLayers][];
            _g4PagedV = new float[numLayers][];
            _g4PagedKvDimPerLayer = new int[numLayers];

            // First pass: allocate ONLY for non-donor layers. Donor
            // receivers alias their target's buffer in the second pass.
            for (int l = 0; l < numLayers; l++)
            {
                bool isReceiver = _kvDonorMap != null && _kvDonorMap.ContainsKey(l);
                if (isReceiver) continue;

                int hd = HeadDimForLayer(l);
                int kvHeads = KVHeadsForLayer(l);
                int perTokenStride = kvHeads * hd;
                _g4PagedKvDimPerLayer[l] = perTokenStride;
                long bufBytes = (long)targetBlocks * blockSize * perTokenStride;
                _g4PagedK[l] = new float[bufBytes];
                _g4PagedV[l] = new float[bufBytes];

                // If we're growing an existing buffer (and the block size
                // didn't change - that would invalidate the layout), copy
                // the previously-written blocks across. Block IDs index
                // directly into the buffer (slot = block_id * block_size
                // + offset), so the existing data stays at the same
                // offsets in the larger buffer.
                if (oldPagedK != null && oldPagedK[l] != null && oldBlockSize == blockSize)
                {
                    long oldLen = (long)oldNumBlocks * blockSize * perTokenStride;
                    long copyLen = Math.Min(oldLen, oldPagedK[l].LongLength);
                    Array.Copy(oldPagedK[l], _g4PagedK[l], copyLen);
                    Array.Copy(oldPagedV[l], _g4PagedV[l], copyLen);
                }
            }
            // Second pass: alias receivers to their donors.
            if (_kvDonorMap != null)
            {
                foreach (var kvp in _kvDonorMap)
                {
                    int receiver = kvp.Key;
                    int donor = kvp.Value;
                    _g4PagedK[receiver] = _g4PagedK[donor];
                    _g4PagedV[receiver] = _g4PagedV[donor];
                    _g4PagedKvDimPerLayer[receiver] = _g4PagedKvDimPerLayer[donor];
                }
            }
            _g4PagedNumBlocks = targetBlocks;
            _g4PagedBlockSize = blockSize;
        }
    }
}
