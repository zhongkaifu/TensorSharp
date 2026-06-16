// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
//
// Gemma 4 MTP (multi-token prediction) draft head — the "gemma4-assistant"
// architecture (llama.cpp src/models/gemma4-assistant.cpp + common/speculative.cpp
// draft-mtp). Unlike Qwen3.6 NextN (one extra block fused into the trunk GGUF),
// the Gemma 4 draft head ships as a SEPARATE small GGUF and is an EAGLE-style
// recurrent drafter:
//
//   x      = target.tok_embd[token] * sqrt(n_embd_backbone)   // backbone embedding (3840)
//   xh     = concat(x, h_prev)                                // [2*backbone]
//   cur    = nextn.pre_projection @ xh                        // -> draft hidden (1024)
//   for il in 0..n_nextn-1:                                   // 4 Gemma-style decoder blocks
//       Q  = rope(q_norm(wq @ attn_norm(cur)))               // draft computes ONLY Q
//       a  = attn(Q, target_KV)                               // attends the TARGET's last
//                                                            //   local (layer N-2) / global
//                                                            //   (layer N-1) K/V cache
//       cur = block_residuals(a, ffn, out_scale)
//   cur    = output_norm(cur)
//   logits = draft.tok_embd @ cur                             // draft's own LM head (no softcap)
//   h_next = nextn.post_projection @ cur                      // -> backbone (3840), chains the
//                                                            //   next recurrent draft step
//
// The draft holds NO K/V of its own (no wk/wv tensors): every draft layer queries
// into the target model's existing per-layer KV cache. llama.cpp maps all SWA draft
// layers to the target's last local layer (n_layer-2) and the global draft layer to
// the target's last global layer (n_layer-1), and — with shared KV — uses the SAME
// position for every drafted token (the recurrence flows purely through h). That
// makes MtpCatchUp / the recurrent-state snapshot no-ops here: the draft is stateless
// given (token, h), so the shared MtpSpeculativeExecution core's draft/verify/rollback
// loop drives it with only an attention-KV position rewind on rejection.
using System;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.GGML;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models
{
    public partial class Gemma4Model : IMtpBatchedSpeculativeModel
    {
        // Draft-head hyper-parameters (read from the assistant GGUF).
        private int _mtpNumLayers;
        private int _mtpHidden;          // draft internal hidden (e.g. 1024)
        private int _mtpBackboneDim;     // = Config.HiddenSize (e.g. 3840)
        private int _mtpDraftHeads;      // draft query-head count (may be < target's, e.g. E4B: 4 vs 8)
        private bool[] _mtpSwaPattern;   // per draft layer: true = SWA/local
        private float[] _mtpLayerScale;  // per draft layer layer_output_scale
        private int _mtpLocalDonor;      // target layer whose KV the SWA draft layers read
        private int _mtpGlobalDonor;     // target layer whose KV the global draft layer reads

        // Batched-trunk speculative mode: when the verify runs through the
        // batched paged path (IMtpBatchedSpeculativeModel), the draft must read
        // the sequence's PAGED donor KV (_g4PagedK) via its block table instead
        // of the model's single linear cache. SpecForwardBatched sets these; the
        // linear SpecForward clears the mode.
        private bool _mtpBatchedMode;
        private SequenceState _mtpBatchedSeq;
        private float[] _mtpDraftScores;  // reusable softmax scratch for the paged draft attention

        /// <summary>True when a usable Gemma 4 assistant draft head is loaded.</summary>
        public bool HasMtp { get; private set; }

        /// <summary>
        /// Speculation profitability per backend:
        /// <list type="bullet">
        /// <item>ggml backends run the fused single-graph MTP kernels (the multi-token
        ///   verify <see cref="NativeGemma4ModelVerify"/> / <see cref="TryFusedMoEModelVerify"/>
        ///   and the fused draft <see cref="NativeGemma4DraftStep"/>) — a clear win.</item>
        /// <item>The pure-C# CUDA backend has no fused kernels, but its per-op verify and
        ///   draft are now fully GPU-resident: the draft attends the donor cache on-device
        ///   (<see cref="TryDraftDecodeAttentionCuda"/>), global verify attention runs the
        ///   GQA decode kernel per row against the live cache
        ///   (<see cref="TryGlobalDecodeLoopAttentionCuda"/>), and the global RoPE uses a
        ///   GPU kernel — so the verify layer loop issues ZERO host-sync stalls (was ~10/
        ///   step). That makes spec a win on the prose/chat workloads (≈1.1–1.2x, higher
        ///   acceptance) and ~break-even on low-acceptance greedy. Enabled.</item>
        /// </list>
        /// CPU / MLX (no fused kernels AND no GPU-resident per-op path) stay off — there
        /// the verify can't keep up and the engine serves the fast standard decode.
        /// </summary>
        public bool MtpSpeculationProfitable => HasMtp && (IsGgmlBackend || _backend == BackendType.Cuda);

        /// <summary>
        /// Load the Gemma 4 assistant (MTP draft) GGUF and attach it to this target
        /// model. The draft's weights are stored under an <c>mtp.</c> name prefix in
        /// the shared weight dictionaries so the existing matmul/norm machinery serves
        /// them; the draft re-uses the target's token embedding, RoPE tables, KV cache
        /// and decode-attention kernels.
        /// </summary>
        public void LoadMtpDraftWeights(string draftGgufPath)
        {
            if (string.IsNullOrEmpty(draftGgufPath) || !System.IO.File.Exists(draftGgufPath))
                throw new System.IO.FileNotFoundException("Gemma 4 MTP draft GGUF not found.", draftGgufPath);

            using var draft = new GgufFile(draftGgufPath);
            string arch = draft.GetString("general.architecture", "");
            // Accept both spellings of the draft architecture id: the original
            // converter emitted 'gemma4-assistant' (hyphen; 12B/26B drafts) while
            // newer converters emit 'gemma4_assistant' (underscore; the E4B draft).
            // They are the same architecture.
            if (arch != "gemma4-assistant" && arch != "gemma4_assistant")
                throw new InvalidOperationException(
                    $"Expected a 'gemma4-assistant' GGUF for the Gemma 4 MTP draft head but got '{arch}'.");

            _mtpNumLayers = (int)draft.GetUint32($"{arch}.block_count");
            _mtpHidden = (int)draft.GetUint32($"{arch}.embedding_length");
            _mtpBackboneDim = (int)draft.GetUint32($"{arch}.embedding_length_out", (uint)Config.HiddenSize);
            _mtpDraftHeads = (int)draft.GetUint32($"{arch}.attention.head_count");
            int draftGlobalHd = (int)draft.GetUint32($"{arch}.attention.key_length", (uint)_globalHeadDim);
            int draftLocalHd = (int)draft.GetUint32($"{arch}.attention.key_length_swa", (uint)_localHeadDim);
            _mtpSwaPattern = draft.GetBoolArray($"{arch}.attention.sliding_window_pattern");

            if (_mtpBackboneDim != Config.HiddenSize)
                throw new InvalidOperationException(
                    $"MTP draft backbone dim {_mtpBackboneDim} != target hidden size {Config.HiddenSize}.");
            // The draft may run FEWER query heads than the target (the E4B draft
            // has 4 vs the target's 8). It still attends the target's KV via GQA,
            // so the query head count is the draft's own — only the per-head DIM
            // must match the target, and the count must group cleanly onto the
            // donor's KV heads (local and global).
            if (_mtpDraftHeads <= 0
                || Config.NumKVHeads == 0 || _mtpDraftHeads % Config.NumKVHeads != 0
                || _numGlobalKVHeads == 0 || _mtpDraftHeads % _numGlobalKVHeads != 0)
                throw new InvalidOperationException(
                    $"MTP draft head count {_mtpDraftHeads} does not group onto the target KV heads " +
                    $"(local {Config.NumKVHeads}, global {_numGlobalKVHeads}).");
            if (draftGlobalHd != _globalHeadDim || draftLocalHd != _localHeadDim)
                throw new InvalidOperationException(
                    $"MTP draft head dims (local {draftLocalHd}, global {draftGlobalHd}) do not match the target " +
                    $"(local {_localHeadDim}, global {_globalHeadDim}).");
            if (_mtpSwaPattern == null || _mtpSwaPattern.Length != _mtpNumLayers)
                throw new InvalidOperationException("MTP draft sliding-window pattern is missing or the wrong length.");

            // llama.cpp share() mapping: all SWA draft layers read the target's last
            // local layer's KV, the global draft layer reads the last global layer's KV.
            _mtpGlobalDonor = Config.NumLayers - 1;
            _mtpLocalDonor = Config.NumLayers - 2;
            if (!IsLocalLayer(_mtpLocalDonor) || IsLocalLayer(_mtpGlobalDonor))
                throw new InvalidOperationException(
                    $"MTP donor layers don't have the expected types (local donor {_mtpLocalDonor} should be SWA, " +
                    $"global donor {_mtpGlobalDonor} should be full-attention).");
            // When those tail layers SHARE their K/V with an earlier donor (Gemma 4
            // E-series with shared_kv_layers > 0 — e.g. E4B shares the last 18
            // layers), the physical K/V lives in that donor; the tail layers have
            // no cache of their own. Follow the share map to the layer that actually
            // holds the cache the draft must read.
            if (_kvDonorMap != null)
            {
                if (_kvDonorMap.TryGetValue(_mtpLocalDonor, out int sharedLocalDonor))
                    _mtpLocalDonor = sharedLocalDonor;
                if (_kvDonorMap.TryGetValue(_mtpGlobalDonor, out int sharedGlobalDonor))
                    _mtpGlobalDonor = sharedGlobalDonor;
            }

            LoadMtpDraftTensors(draft);

            _mtpLayerScale = new float[_mtpNumLayers];
            for (int il = 0; il < _mtpNumLayers; il++)
            {
                _mtpLayerScale[il] = _weights.TryGetValue($"mtp.blk.{il}.layer_output_scale.weight", out var s)
                    ? s.GetElementAsFloat(0) : 1f;
            }

            // Sanity-check the tensors the draft step relies on are present.
            HasMtp = HasLinearWeight("mtp.nextn.pre_projection.weight")
                && HasLinearWeight("mtp.nextn.post_projection.weight")
                && HasLinearWeight("mtp.token_embd.weight")
                && _weights.ContainsKey("mtp.output_norm.weight");
            for (int il = 0; il < _mtpNumLayers && HasMtp; il++)
            {
                string p = $"mtp.blk.{il}";
                HasMtp = HasLinearWeight($"{p}.attn_q.weight") && HasLinearWeight($"{p}.attn_output.weight")
                    && HasLinearWeight($"{p}.ffn_gate.weight") && HasLinearWeight($"{p}.ffn_up.weight")
                    && HasLinearWeight($"{p}.ffn_down.weight")
                    && _weights.ContainsKey($"{p}.attn_norm.weight") && _weights.ContainsKey($"{p}.attn_q_norm.weight")
                    && _weights.ContainsKey($"{p}.post_attention_norm.weight") && _weights.ContainsKey($"{p}.ffn_norm.weight")
                    && _weights.ContainsKey($"{p}.post_ffw_norm.weight");
            }

            // Per-layer-embedding (PLE) targets (Gemma 4 E-series, e.g. E4B) are
            // supported: the spec trunk computes PLE and runs the per-op forward
            // (the fused trunk kernels don't thread PLE). Earlier this threw.

            // The fused single-graph draft kernel is parameterised by the draft's
            // own query-head count (passed as num_heads), so it serves a smaller-
            // head draft (E4B: 4 vs the target's 8) too — it reads the donor KV via
            // GQA grouping onto the donor's KV heads.
            if (HasMtp && IsGgmlBackend)
                BuildMtpDraftArrays();

            Console.WriteLine(HasMtp
                ? $"  Gemma 4 MTP draft head ready ({_mtpNumLayers} layers, hidden {_mtpHidden}, " +
                  $"draftHeads={_mtpDraftHeads}, ple={_pleDim}, donors local={_mtpLocalDonor}/global={_mtpGlobalDonor}, " +
                  $"fusedDraft={(_mtpDraftArrays != null ? "yes" : "no")})."
                : "  Gemma 4 MTP draft GGUF loaded but incomplete; MTP drafting disabled.");
        }

        private unsafe void LoadMtpDraftTensors(GgufFile draft)
        {
            foreach (var kv in draft.Tensors)
            {
                var info = kv.Value;

                // Normalize across converter conventions. The newer converter
                // prefixes the draft's own tensors with "mtp." (e.g.
                // mtp.pre_projection.weight) and drops the "nextn." namespace on
                // the recurrent projections; the original converter used no "mtp."
                // prefix and named them "nextn.pre_projection.weight". Map both onto
                // the internal "mtp.nextn.*" keys the draft step looks up.
                string core = info.Name;
                if (core.StartsWith("mtp.", StringComparison.Ordinal))
                    core = core.Substring(4);
                // rope_freqs is reused from the target (verified identical).
                // centroids / token_ordering are an optional clustered draft LM head
                // (newer format); we decode through the full token_embd head, so skip.
                if (core == "rope_freqs.weight" || core == "centroids.weight" || core == "token_ordering.weight")
                    continue;
                if (core == "pre_projection.weight") core = "nextn.pre_projection.weight";
                else if (core == "post_projection.weight") core = "nextn.post_projection.weight";

                string name = "mtp." + core;
                long byteCount = draft.GetTensorByteCount(info);

                if (IsQuantizedLinearWeight(info))
                {
                    if (IsGgmlBackend)
                        EnsureQuantBackendAvailable();
                    IntPtr ptr = QuantizedWeight.AllocateBuffer(byteCount);
                    draft.ReadTensorDataToNative(info, ptr, byteCount);
                    _quantWeights[name] = new QuantizedWeight(ptr, byteCount, (int)info.Type, (long)info.Shape[0], (long)info.Shape[1]);
                }
                else
                {
                    long numElements = info.NumElements;
                    long[] tsShape = new long[info.Shape.Length];
                    for (int i = 0; i < info.Shape.Length; i++)
                        tsShape[i] = (long)info.Shape[info.Shape.Length - 1 - i];

                    var tensor = new Tensor(_allocator, DType.Float32, tsShape);
                    IntPtr destPtr = TensorComputePrimitives.GetStoragePointer(tensor);
                    if (info.Type == GgmlTensorType.F32)
                    {
                        draft.ReadTensorDataToFloat32Native(info, destPtr, numElements);
                    }
                    else
                    {
                        IntPtr tempPtr = QuantizedWeight.AllocateBuffer(byteCount);
                        try
                        {
                            draft.ReadTensorDataToNative(info, tempPtr, byteCount);
                            NativeDequant.DequantizeToFloat32Native((int)info.Type, tempPtr, destPtr, numElements);
                        }
                        finally
                        {
                            QuantizedWeight.FreeBuffer(tempPtr);
                        }
                    }
                    _weights[name] = tensor;
                }
            }
        }

        // ====================================================================
        // IMtpSpeculativeModel
        // ====================================================================

        /// <summary>
        /// Trunk (target) forward for speculative decoding. Same math as Forward()
        /// but captures the post-output-norm hidden state of every row into
        /// <paramref name="hAllOut"/> (n*hidden floats; llama.cpp's h_nextn — the
        /// recurrent input the draft head consumes) and, when
        /// <paramref name="allLogitsRows"/>, per-row logits into
        /// <paramref name="logitsOut"/> (else only the last row). Advances the KV
        /// caches and _cacheSeqLen exactly like Forward(). Runs the per-op layer
        /// path (no model-wide fused decode) so every row's hidden state is available.
        /// </summary>
        public unsafe void SpecForward(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            if (!HasMtp)
                throw new InvalidOperationException("Model has no Gemma 4 MTP draft head.");

            _mtpBatchedMode = false;   // this is the linear-cache trunk
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            int hidden = Config.HiddenSize;
            EnsureCacheCapacity(startPos + seqLen);

            Tensor h = Embedding(tokens);
            ScaleEmbedding(h);

            // Per-layer embeddings (PLE) for Gemma 4 E-series targets (e.g. E4B).
            // Threaded into BOTH the fused trunk (the dense decode/verify kernels
            // accept PLE) and the per-op fallback below.
            Tensor perLayerInputs = _pleDim > 0 ? ComputePLE(tokens, h, seqLen) : null;

            // Fused single-graph trunk: a verify batch (seqLen>1) runs through the
            // multi-token kernel (NativeGemma4ModelVerify), a plain step (seqLen==1)
            // through the single-token decode kernel — ONE GGML graph for all
            // layers instead of ~800 per-op dispatches, so the verify finally
            // amortises and beats K+1 single-token decodes. Works at ANY context
            // length: the kernel windows the SWA layers' circular cache (wrap-aware
            // write + windowed read) so there is no per-window-crossing cliff.
            // The DENSE kernels now also thread PLE and shared-KV (KV donors), so the
            // Gemma 4 E-series (E4B) takes this fast path too. The MoE kernels do
            // not, so they keep the no-PLE/no-donor gate. Keeping plain+verify both
            // on the fused (device-cache) path avoids mixing device and host cache
            // writers within one spec session.
            bool fusedCommon = _decodeArrays != null
                && !_kvCacheDtype.IsBlockQuantized()
                && Environment.GetEnvironmentVariable("TS_GMTP_NO_FUSED") != "1";
            // Dense models take the dense fused trunk; all-MoE models (e.g.
            // gemma-4-26B-A4B, where _canUseFusedFullModelDecode is false) take the
            // fused MoE trunk: a single graph for the whole transformer for both the
            // plain step (seqLen==1, the MoE decode kernel) and the verify batch
            // (seqLen>1, the MoE verify kernel). Without the MoE branch the verify
            // (and even plain steps) fell to the per-op path, making spec net-negative.
            bool fusedDenseOk = fusedCommon && _canUseFusedFullModelDecode;
            bool fusedMoeOk = fusedCommon && !_canUseFusedFullModelDecode && _numExperts > 0
                && _pleDim == 0 && _kvDonorMap.Count == 0;
            bool usedFused = false;
            if (fusedDenseOk)
            {
                RefreshDecodeArraysKvCache();
                if (seqLen == 1)
                {
                    NativeGemma4ModelDecode(h, startPos, perLayerInputs);
                    usedFused = true;
                }
                else
                {
                    usedFused = NativeGemma4ModelVerify(h, startPos, seqLen, perLayerInputs);
                }
            }
            else if (fusedMoeOk)
            {
                RefreshDecodeArraysKvCache();
                usedFused = seqLen == 1
                    ? TryFusedMoEModelDecode(h, startPos)
                    : TryFusedMoEModelVerify(h, startPos, seqLen);
            }
            if (usedFused)
            {
                _kvCacheHostDirty = true;   // device write; draft re-syncs donor layers
                // The kernel wrote h's host storage via download; drop any
                // stale cached device buffer so the final norm re-reads it.
                InvalidateTensorDeviceCache(h);
            }

            if (!usedFused)
            {
                // Per-op fallback (past the SWA window, an unsupported model shape,
                // or a PLE target). Processes the batch exactly like a normal
                // prefill chunk so a verify batch is numerically a prefill of the
                // same length: PLE per-layer inputs are threaded in, shared-KV
                // layers follow their donor, SWA layers get cross-chunk prev
                // windows, and donor layers publish fresh K/V via _prefillSWAKV.
                // Capturing each row's hidden just needs the all-row hidden after
                // the loop.
                EnsureKvCacheHostSynchronized();
                if (_swaKVDonorLayers != null && _swaKVDonorLayers.Count > 0 && seqLen > 1)
                    _prefillSWAKV = new System.Collections.Generic.Dictionary<int, (Tensor k, Tensor v)>();
                if (seqLen > 1)
                    PrepareSwaPrevWindowsForChunk(startPos, seqLen);

                for (int l = 0; l < Config.NumLayers; l++)
                {
                    Tensor perLayerInput = perLayerInputs != null
                        ? ExtractPerLayerSlice(perLayerInputs, l, seqLen) : null;
                    bool isShared = _kvDonorMap.ContainsKey(l);
                    h = TransformerBlock(h, l, seqLen, startPos, isShared, perLayerInput, null);
                    perLayerInput?.Dispose();
                }

                if (_prefillSWAKV != null)
                {
                    foreach (var entry in _prefillSWAKV.Values)
                    {
                        entry.k.Dispose();
                        entry.v.Dispose();
                    }
                    _prefillSWAKV = null;
                }
                if (seqLen > 1)
                    DisposeSwaPrevWindows();
            }
            perLayerInputs?.Dispose();

            // Post-output-norm hidden state for every row (h_nextn).
            Ops.RMSNorm(h, h, _weights["output_norm.weight"], null, Config.Eps);

            if (hAllOut != null)
            {
                float* src = GetFloatPtr(h);
                fixed (float* dst = hAllOut)
                    Buffer.MemoryCopy(src, dst, (long)hAllOut.Length * 4, (long)seqLen * hidden * 4);
            }

            string outputWeight = _hasTiedOutput ? "token_embd.weight" : "output.weight";
            if (allLogitsRows)
            {
                Tensor logitsT = LinearForward(h, outputWeight);
                h.Dispose();
                if (_finalLogitSoftcap > 0f)
                    ApplyLogitSoftcap(logitsT);
                float* src = GetFloatPtr(logitsT);
                fixed (float* dst = logitsOut)
                    Buffer.MemoryCopy(src, dst, (long)logitsOut.Length * 4, (long)seqLen * Config.VocabSize * 4);
                logitsT.Dispose();
            }
            else
            {
                Tensor lastRow;
                if (seqLen > 1)
                {
                    using var narrowed = h.Narrow(0, seqLen - 1, 1);
                    lastRow = Ops.NewContiguous(narrowed);
                    h.Dispose();
                }
                else
                {
                    lastRow = h;
                }
                Tensor logitsT = LinearForward(lastRow, outputWeight);
                lastRow.Dispose();
                if (_finalLogitSoftcap > 0f)
                    ApplyLogitSoftcap(logitsT);
                float* src = GetFloatPtr(logitsT);
                fixed (float* dst = logitsOut)
                    Buffer.MemoryCopy(src, dst, (long)logitsOut.Length * 4, (long)Config.VocabSize * 4);
                logitsT.Dispose();
            }

            _cacheSeqLen += seqLen;
            _forwardCount++;
        }

        /// <summary>
        /// One Gemma 4 assistant draft step: consume (token, h_prev) and fill
        /// next-token logits + the chained recurrent hidden state. With shared KV
        /// every draft step queries at the trunk's current cache position (the
        /// recurrence flows through h, not through new K/V), so the incoming
        /// <paramref name="pos"/> argument is intentionally ignored.
        /// </summary>
        public unsafe void MtpDraftStep(int token, float[] hPrev, int pos, float[] logitsOut, float[] hOut)
        {
            if (!HasMtp)
                throw new InvalidOperationException("Model has no Gemma 4 MTP draft head.");

            // Trunk position; the donor KV holds [0, fixedPos). The linear trunk
            // tracks it in _cacheSeqLen; the batched trunk in seq.NumComputedTokens
            // (the executor advances the sequence only after the step).
            int fixedPos;
            if (_mtpBatchedMode && _mtpBatchedSeq != null)
            {
                // Paged donor KV (_g4PagedK) is a host float[] kept current by the
                // verify's ScatterKv — no device KV sync needed.
                fixedPos = _mtpBatchedSeq.NumComputedTokens;
            }
            else
            {
                fixedPos = _cacheSeqLen;
                // Linear trunk: the fused draft kernel reads the donor KV on-device
                // (no host sync), running the whole head as one graph. Falls back to
                // the per-op draft (host attention, needs the donor sync) past the
                // SWA window or on a non-fused shape.
                if (_mtpDraftArrays != null && !_kvCacheDtype.IsBlockQuantized()
                    && Environment.GetEnvironmentVariable("TS_GMTP_NO_FUSED") != "1"
                    && NativeGemma4DraftStep(token, hPrev, fixedPos, logitsOut, hOut))
                    return;
                SyncDonorKvToHost();
            }
            int backbone = Config.HiddenSize;

            // x = target.tok_embd[token] * sqrt(backbone)
            Tensor x = Embedding(new[] { token });
            ScaleEmbedding(x);

            // xh = concat(x, h_prev)  ->  [1, 2*backbone]
            var xh = new Tensor(_allocator, DType.Float32, 1, 2L * backbone);
            using (var dstX = xh.Narrow(1, 0, backbone))
                Ops.Copy(dstX, x);
            x.Dispose();
            {
                float* dst = GetFloatPtr(xh) + backbone;
                fixed (float* src = hPrev)
                    Buffer.MemoryCopy(src, dst, (long)backbone * 4, (long)backbone * 4);
            }
            InvalidateTensorDeviceCache(xh);

            Tensor cur = LinearForward(xh, "mtp.nextn.pre_projection.weight");   // [1, mtpHidden]
            xh.Dispose();

            for (int il = 0; il < _mtpNumLayers; il++)
                cur = MtpDraftLayer(cur, il, fixedPos);

            Tensor normed = RMSNormOp(cur, "mtp.output_norm.weight");
            cur.Dispose();

            // Draft LM head: the assistant's own token embedding (no logit softcap).
            Tensor logitsT = LinearForward(normed, "mtp.token_embd.weight");
            {
                float* src = GetFloatPtr(logitsT);
                fixed (float* dst = logitsOut)
                    Buffer.MemoryCopy(src, dst, (long)logitsOut.Length * 4, (long)Config.VocabSize * 4);
            }
            logitsT.Dispose();

            // Recurrent hidden state for the next draft step (backbone dim).
            Tensor hNext = LinearForward(normed, "mtp.nextn.post_projection.weight");
            normed.Dispose();
            {
                float* src = GetFloatPtr(hNext);
                fixed (float* dst = hOut)
                    Buffer.MemoryCopy(src, dst, (long)hOut.Length * 4, (long)backbone * 4);
            }
            hNext.Dispose();
        }

        // The draft reads only the two donor layers' KV via host pointers. The
        // trunk verify writes the whole cache device-side (sets _kvCacheHostDirty);
        // syncing only those two layers — instead of all N via
        // EnsureKvCacheHostSynchronized — keeps the per-step DtoH cost tiny without
        // disturbing the dirty flag the next trunk forward relies on.
        private void SyncDonorKvToHost()
        {
            if (!_kvCacheHostDirty || !IsGgmlBackend || _kvCacheK == null)
                return;
            foreach (int l in stackalloc[] { _mtpLocalDonor, _mtpGlobalDonor })
            {
                if (_kvCacheK[l] != null) SyncTensorHostCache(_kvCacheK[l]);
                if (_kvCacheV[l] != null) SyncTensorHostCache(_kvCacheV[l]);
            }
        }

        // One Gemma-style decoder block of the draft head. Computes only Q and
        // attends into the target's donor KV cache; consumes and disposes inpL.
        private Tensor MtpDraftLayer(Tensor inpL, int il, int fixedPos)
        {
            string p = $"mtp.blk.{il}";
            bool isLocal = _mtpSwaPattern[il];
            int hd = isLocal ? _localHeadDim : _globalHeadDim;
            int donor = isLocal ? _mtpLocalDonor : _mtpGlobalDonor;
            int kvHeads = KVHeadsForLayer(donor);
            float[] freqs = isLocal ? _ropeFreqsLocal : _ropeFreqsGlobal;

            using var attnNormed = RMSNormOp(inpL, $"{p}.attn_norm.weight");
            Tensor q = LinearForward(attnNormed, $"{p}.attn_q.weight");   // [1, draftHeads*hd]
            RMSNormInPlace(q, _weights[$"{p}.attn_q_norm.weight"], _mtpDraftHeads, hd, Config.Eps);
            ApplyNeoXRoPEDecode(q, _mtpDraftHeads, hd, fixedPos, freqs);

            var attn = new Tensor(_allocator, DType.Float32, 1, _mtpDraftHeads * hd);
            // Gemma 4 attention scale is 1.0 (no 1/sqrt(d)); see f_attention_scale.
            // The draft uses its OWN query-head count (_mtpDraftHeads), grouped onto
            // the donor's KV heads.
            if (_mtpBatchedMode && _mtpBatchedSeq != null)
            {
                // Batched trunk: attend the sequence's PAGED donor KV.
                MtpDraftPagedAttention(q, donor, kvHeads, hd, fixedPos, isLocal, _mtpBatchedSeq, attn);
                InvalidateTensorDeviceCache(attn);
            }
            else if (TryDraftDecodeAttentionCuda(q, donor, kvHeads, hd, fixedPos, isLocal, attn))
            {
                // On-device GQA decode attention (CUDA): reads the donor cache in
                // place, so the draft head stops DtoH-ing the whole 4 MB donor cache
                // to the host per layer — the dominant draft-phase sync stall on the
                // pure-C# CUDA backend. Mirrors the main decode's attention path.
            }
            else if (isLocal)
            {
                int cacheLen = _kvCacheSize[donor];
                int attendLen = Math.Min(fixedPos, _slidingWindow);
                AttentionDecodeCircular(q, _kvCacheK[donor], _kvCacheV[donor], attn,
                    _mtpDraftHeads, kvHeads, hd, hd, fixedPos - 1, attendLen, cacheLen, 1f);
            }
            else
            {
                AttentionDecodeWithWindow(q, _kvCacheK[donor], _kvCacheV[donor], attn,
                    _mtpDraftHeads, kvHeads, hd, hd, 0, fixedPos, 1f);
            }
            q.Dispose();

            // wo -> post_attention_norm -> residual
            Tensor attnOut = LinearForward(attn, $"{p}.attn_output.weight");
            attn.Dispose();
            Ops.RMSNorm(attnOut, attnOut, _weights[$"{p}.post_attention_norm.weight"], null, Config.Eps);
            Ops.Add(attnOut, attnOut, inpL);   // attn_out = post_attn_norm(wo(attn)) + inpL
            inpL.Dispose();

            // ffn_norm -> GELU FFN -> post_ffw_norm -> residual
            using (var ffnIn = RMSNormOp(attnOut, $"{p}.ffn_norm.weight"))
            {
                Tensor ffnOut = FFNGeluSeparate(ffnIn, $"{p}.ffn_gate.weight", $"{p}.ffn_up.weight", $"{p}.ffn_down.weight");
                Ops.RMSNorm(ffnOut, ffnOut, _weights[$"{p}.post_ffw_norm.weight"], null, Config.Eps);
                Ops.Add(attnOut, attnOut, ffnOut);
                ffnOut.Dispose();
            }

            float scale = _mtpLayerScale[il];
            if (scale != 1f)
                Ops.Mul(attnOut, attnOut, scale);
            return attnOut;
        }

        // GPU GQA decode attention for one draft layer (CUDA backend only). The
        // draft query is at the trunk position <paramref name="fixedPos"/> and
        // attends the donor cache [0, fixedPos) — windowed for the local donor.
        // Mirrors the main decode's CudaFusedOps path (query position startPos with
        // startPos+1 keys ⇒ here startPos+1 == fixedPos) so it is numerically the
        // same as the AttentionDecodeCircular/WithWindow CPU helpers, but reads the
        // donor cache ON-DEVICE — no per-layer 4 MB DtoH of the cache to the host.
        // Returns false on any other backend (caller uses the CPU helpers).
        private bool TryDraftDecodeAttentionCuda(
            Tensor q, int donor, int kvHeads, int hd, int fixedPos, bool isLocal, Tensor attn)
        {
            if (_backend != BackendType.Cuda || fixedPos <= 0)
                return false;
            int cacheLen = _kvCacheSize[donor];
            if (isLocal)
            {
                int attendLen = Math.Min(fixedPos, _slidingWindow);
                int attendStart = Math.Max(0, fixedPos - attendLen);
                return CudaFusedOps.TryGqaDecodeAttention(attn, q, _kvCacheK[donor], _kvCacheV[donor],
                    _mtpDraftHeads, kvHeads, hd, attendStart, fixedPos - attendStart, cacheLen, true, 1f);
            }
            return CudaFusedOps.TryGqaDecodeAttention(attn, q, _kvCacheK[donor], _kvCacheV[donor],
                _mtpDraftHeads, kvHeads, hd, 0, fixedPos, cacheLen, false, 1f);
        }

        /// <summary>Gemma 4's draft head holds no KV of its own (it reads the
        /// target's), so there is nothing to replay — the no-op here is correct.</summary>
        public void MtpCatchUp(int[] tokens, float[] hRows, int startPos) { }

        /// <summary>Pre-grow the trunk KV caches. Safe at any time for Gemma 4: the
        /// draft writes no MTP rows into the cache (it only reads the target's).
        /// In batched-trunk mode the K/V lives in paged blocks the scheduler owns,
        /// so growing the (unused) linear cache would just waste memory.</summary>
        public void MtpEnsureCapacity(int requiredSeqLen)
        {
            if (_mtpBatchedMode) return;
            EnsureCacheCapacity(requiredSeqLen);
        }

        /// <summary>No recurrent (GDN/SSM) state in Gemma 4 — drafting is stateless
        /// given (token, h), so verify rollback needs only an attention-KV rewind.</summary>
        public void MtpSnapshotRecurrentState() { }

        /// <summary>See <see cref="MtpSnapshotRecurrentState"/>.</summary>
        public void MtpRestoreRecurrentState() { }

        /// <summary>
        /// Rewind the trunk KV position counter after rejected speculative tokens.
        /// Rows past <paramref name="length"/> are overwritten by the kept-prefix
        /// re-forward and the causal mask never reads past the live position, so no
        /// data movement is needed (the SWA circular cache re-writes the same slots).
        /// </summary>
        public void MtpRewindCache(int length)
        {
            if (length < 0 || length > _cacheSeqLen)
                throw new ArgumentOutOfRangeException(nameof(length),
                    $"Rewind length {length} outside [0, {_cacheSeqLen}].");
            _cacheSeqLen = length;
        }

        // Escape hatch: TS_GMTP_NO_FAST_ROLLBACK=1 restores the kept-prefix
        // re-forward (slower, but refreshes committed-token KV through the decode
        // kernel so spec output tracks the all-decode no-spec path more closely).
        private static readonly bool s_noFastRollback =
            Environment.GetEnvironmentVariable("TS_GMTP_NO_FAST_ROLLBACK") == "1";

        /// <summary>
        /// Gemma 4's verify (fused MoE/dense or per-op) writes attention KV for every
        /// token in the batch at its true position, and the model has no recurrent
        /// state. So on partial acceptance the kept prefix's KV is already correct in
        /// the live cache — the executor can skip the redundant re-forward and just
        /// rewind the position. This is the dominant rollback cost on long contexts
        /// for the MoE model (manual-attention verify), so it is enabled for MoE
        /// Gemma 4 (e.g. 26B-A4B). It is left OFF for the dense model: there the
        /// re-forward is cheap and refreshing the committed token's KV through the
        /// decode kernel keeps spec output byte-identical to the no-spec path (the
        /// dense exact-match validation). Honoured only on the linear trunk.
        /// </summary>
        public bool MtpVerifyPersistsAcceptedKv => (_numExperts > 0 || _pleDim > 0) && !_mtpBatchedMode && !s_noFastRollback;

        // ====================================================================
        // IMtpBatchedSpeculativeModel — speculative trunk on the batched paged
        // path. The verify runs through ForwardBatch (one sequence, K+1 tokens):
        // its matmuls are batched GEMMs that read the 12B weights ONCE for all
        // K+1 rows, so a verify amortises to ~one batched decode step (unlike the
        // single-sequence Forward path, where the fused single-token decode kernel
        // has no multi-token equivalent and the verify can't keep up). The draft
        // head reads the sequence's paged donor KV (host float[], no device sync).
        // Gemma 4 has no recurrent state, so the per-slot snapshot/restore the
        // interface requires are no-ops.
        // ====================================================================

        /// <summary>
        /// Batched-trunk speculation is implemented (verify through ForwardBatch +
        /// paged-KV draft) but DISABLED by default: it runs the per-op batched path
        /// (~0.56x), while the linear trunk now drives the fused single-graph verify
        /// (<see cref="NativeGemma4ModelVerify"/>) and draft
        /// (<see cref="NativeGemma4DraftStep"/>) kernels for ~2x. Routing solo
        /// speculative sequences to the linear trunk (this returning false) gives
        /// the fast path. Opt back into the batched trunk with TS_GMTP_BATCHED_TRUNK=1
        /// (e.g. to compose with a paged-fused verify once that lands).
        /// </summary>
        public bool SupportsBatchedSpecTrunk =>
            HasMtp && IsGgmlBackend && CanUseBatchedSpecPath()
            && Environment.GetEnvironmentVariable("TS_GMTP_BATCHED_TRUNK") == "1";

        private bool CanUseBatchedSpecPath()
        {
            if (_pleDim > 0) return false;
            if (_kvCacheDtype.IsBlockQuantized()) return false;
            for (int l = 0; l < Config.NumLayers; l++)
                if (HasMoE(l)) return false;
            return true;
        }

        public unsafe void SpecForwardBatched(SequenceState seq, int[] tokens, int startPos,
            float[] hAllOut, float[] logitsOut, bool allLogitsRows)
        {
            ArgumentNullException.ThrowIfNull(seq);
            if (tokens == null || tokens.Length == 0)
                throw new ArgumentException("Tokens must not be empty.", nameof(tokens));
            if (startPos != seq.NumComputedTokens)
                throw new InvalidOperationException(
                    $"SpecForwardBatched at position {startPos} but sequence has {seq.NumComputedTokens} computed tokens.");

            int n = tokens.Length;
            var bt = seq.BlockTable;
            if (bt.CapacityTokens < startPos + n)
                throw new InvalidOperationException(
                    $"Block table covers {bt.CapacityTokens} tokens but the spec pass needs {startPos + n}.");

            var positions = new System.Collections.Generic.List<int>(n);
            var slotMapping = new System.Collections.Generic.List<int>(n);
            for (int i = 0; i < n; i++)
            {
                int pos = startPos + i;
                positions.Add(pos);
                int blockIdx = pos / bt.BlockSize;
                slotMapping.Add(bt.Blocks[blockIdx].Id * bt.BlockSize + pos % bt.BlockSize);
            }
            var table = new int[bt.NumBlocks];
            for (int b = 0; b < bt.NumBlocks; b++)
                table[b] = bt.Blocks[b].Id;

            var ctx = new BatchedForwardContext
            {
                Sequences = new System.Collections.Generic.List<SequenceState> { seq },
                NumScheduledTokens = new System.Collections.Generic.List<int> { n },
                QueryStartLoc = new System.Collections.Generic.List<int> { 0, n },
                Positions = positions,
                SlotMapping = slotMapping,
                BlockTables = new[] { table },
                MaxQueryLen = n,
                MaxSeqLen = startPos + n,
                OverrideFlatTokens = tokens,
                CaptureHiddenAll = hAllOut,
                CaptureLogitsAll = allLogitsRows ? logitsOut : null,
            };

            // Arm the paged-draft path for the subsequent draft steps.
            _mtpBatchedMode = true;
            _mtpBatchedSeq = seq;

            var perSeq = ForwardBatch(ctx);
            if (!allLogitsRows && logitsOut != null)
                Array.Copy(perSeq[0], logitsOut, Config.VocabSize);
        }

        // ====================================================================
        // Fused draft-step kernel (TSGgml_Gemma4DraftStep): runs the whole draft
        // head as ONE GGML graph reading the target's donor KV on-device, so the
        // 4-layer head stops costing a full decode in device↔host ping-pong.
        // Linear-trunk only (the donor KV is the live linear cache); the batched
        // trunk's donor KV is paged and keeps the per-op paged draft.
        // ====================================================================
        private sealed class MtpDraftArrays
        {
            public IntPtr TgtTokEmbd, NextnPre, NextnPost, DraftTokEmbd, OutputNorm;
            public int TteType, NpreType, NpostType, DteType;
            public long TteNe0, TteNe1, TteBytes, NpreNe0, NpreNe1, NpreBytes;
            public long NpostNe0, NpostNe1, NpostBytes, DteNe0, DteNe1, DteBytes;
            public IntPtr[] AttnNorm, Wq, QNorm, Wo, PostAttnNorm, FfnNorm, Gate, Up, Down, PostFfwNorm;
            public int[] WqType, WoType, GateType, UpType, DownType;
            public long[] WqNe0, WqNe1, WqBytes, WoNe0, WoNe1, WoBytes;
            public long[] GateNe0, GateNe1, GateBytes, UpNe0, UpNe1, UpBytes, DownNe0, DownNe1, DownBytes;
            public float[] OutScale, RopeBase;
            public int[] Hd, KvHeads, IsLocal, RopeDims;
            public IntPtr[] DonorK, DonorV;       // refreshed per call (cache may grow)
            public int[] DonorCacheSize;
            public int[] DonorLayer;
        }
        private MtpDraftArrays _mtpDraftArrays;

        private void BuildMtpDraftArrays()
        {
            if (!IsGgmlBackend) return;
            int n = _mtpNumLayers;
            var a = new MtpDraftArrays
            {
                AttnNorm = new IntPtr[n], Wq = new IntPtr[n], QNorm = new IntPtr[n], Wo = new IntPtr[n],
                PostAttnNorm = new IntPtr[n], FfnNorm = new IntPtr[n], Gate = new IntPtr[n], Up = new IntPtr[n],
                Down = new IntPtr[n], PostFfwNorm = new IntPtr[n],
                WqType = new int[n], WoType = new int[n], GateType = new int[n], UpType = new int[n], DownType = new int[n],
                WqNe0 = new long[n], WqNe1 = new long[n], WqBytes = new long[n],
                WoNe0 = new long[n], WoNe1 = new long[n], WoBytes = new long[n],
                GateNe0 = new long[n], GateNe1 = new long[n], GateBytes = new long[n],
                UpNe0 = new long[n], UpNe1 = new long[n], UpBytes = new long[n],
                DownNe0 = new long[n], DownNe1 = new long[n], DownBytes = new long[n],
                OutScale = new float[n], RopeBase = new float[n],
                Hd = new int[n], KvHeads = new int[n], IsLocal = new int[n], RopeDims = new int[n],
                DonorK = new IntPtr[n], DonorV = new IntPtr[n], DonorCacheSize = new int[n], DonorLayer = new int[n],
            };

            IntPtr NormPtr(string name) => TensorComputePrimitives.GetStoragePointer(_weights[name]);
            void Q(string name, out IntPtr data, out int type, out long ne0, out long ne1, out long bytes)
            {
                var qw = _quantWeights[name];
                data = qw.Data; type = qw.GgmlType; ne0 = qw.Ne0; ne1 = qw.Ne1; bytes = qw.RawBytes;
            }

            Q("token_embd.weight", out a.TgtTokEmbd, out a.TteType, out a.TteNe0, out a.TteNe1, out a.TteBytes);
            Q("mtp.nextn.pre_projection.weight", out a.NextnPre, out a.NpreType, out a.NpreNe0, out a.NpreNe1, out a.NpreBytes);
            Q("mtp.nextn.post_projection.weight", out a.NextnPost, out a.NpostType, out a.NpostNe0, out a.NpostNe1, out a.NpostBytes);
            Q("mtp.token_embd.weight", out a.DraftTokEmbd, out a.DteType, out a.DteNe0, out a.DteNe1, out a.DteBytes);
            a.OutputNorm = NormPtr("mtp.output_norm.weight");

            for (int il = 0; il < n; il++)
            {
                string p = $"mtp.blk.{il}";
                bool isLocal = _mtpSwaPattern[il];
                a.AttnNorm[il] = NormPtr($"{p}.attn_norm.weight");
                a.QNorm[il] = NormPtr($"{p}.attn_q_norm.weight");
                a.PostAttnNorm[il] = NormPtr($"{p}.post_attention_norm.weight");
                a.FfnNorm[il] = NormPtr($"{p}.ffn_norm.weight");
                a.PostFfwNorm[il] = NormPtr($"{p}.post_ffw_norm.weight");
                Q($"{p}.attn_q.weight", out a.Wq[il], out a.WqType[il], out a.WqNe0[il], out a.WqNe1[il], out a.WqBytes[il]);
                Q($"{p}.attn_output.weight", out a.Wo[il], out a.WoType[il], out a.WoNe0[il], out a.WoNe1[il], out a.WoBytes[il]);
                Q($"{p}.ffn_gate.weight", out a.Gate[il], out a.GateType[il], out a.GateNe0[il], out a.GateNe1[il], out a.GateBytes[il]);
                Q($"{p}.ffn_up.weight", out a.Up[il], out a.UpType[il], out a.UpNe0[il], out a.UpNe1[il], out a.UpBytes[il]);
                Q($"{p}.ffn_down.weight", out a.Down[il], out a.DownType[il], out a.DownNe0[il], out a.DownNe1[il], out a.DownBytes[il]);
                a.OutScale[il] = _mtpLayerScale[il];
                int hd = isLocal ? _localHeadDim : _globalHeadDim;
                int donor = isLocal ? _mtpLocalDonor : _mtpGlobalDonor;
                a.Hd[il] = hd;
                a.KvHeads[il] = KVHeadsForLayer(donor);
                a.IsLocal[il] = isLocal ? 1 : 0;
                a.RopeBase[il] = isLocal ? _ropeLocalBase : _ropeGlobalBase;
                a.RopeDims[il] = hd;   // full rotary (matches ApplyNeoXRoPEDecode)
                a.DonorLayer[il] = donor;
            }
            _mtpDraftArrays = a;
        }

        private void RefreshMtpDraftDonorCaches()
        {
            var a = _mtpDraftArrays;
            for (int il = 0; il < _mtpNumLayers; il++)
            {
                int donor = a.DonorLayer[il];
                a.DonorK[il] = TensorComputePrimitives.GetStoragePointer(_kvCacheK[donor]);
                a.DonorV[il] = TensorComputePrimitives.GetStoragePointer(_kvCacheV[donor]);
                a.DonorCacheSize[il] = _kvCacheSize[donor];
            }
        }

        // Fused draft step: returns false (caller falls back to the per-op draft)
        // when the kernel declines (e.g. fixed_pos past the donor SWA window).
        private unsafe bool NativeGemma4DraftStep(int token, float[] hPrev, int fixedPos, float[] logitsOut, float[] hOut)
        {
            if (_mtpDraftArrays == null || fixedPos <= 0) return false;
            var a = _mtpDraftArrays;
            RefreshMtpDraftDonorCaches();
            // SWA donors are windowed in-kernel; a global donor's linear cache must
            // cover fixedPos (the trunk's forward grew it) — else fall back.
            for (int il = 0; il < _mtpNumLayers; il++)
                if (a.IsLocal[il] == 0 && fixedPos > a.DonorCacheSize[il]) return false;

            IntPtr freqPtr = IntPtr.Zero;
            int freqLen = 0;
            if (_weights.TryGetValue("rope_freqs.weight", out var ft))
            {
                freqPtr = (IntPtr)GetFloatPtr(ft);
                freqLen = (int)ft.ElementCount();
            }

            fixed (float* hp = hPrev, lo = logitsOut, ho = hOut)
            {
                return GgmlBasicOps.Gemma4DraftStep(
                    token, (IntPtr)hp, fixedPos,
                    Config.HiddenSize, _mtpHidden, _mtpNumLayers, _mtpDraftHeads, Config.VocabSize,
                    Config.Eps, _kvCacheDtype.GgmlType(),
                    freqPtr, freqLen,
                    a.TgtTokEmbd, a.TteType, a.TteNe0, a.TteNe1, a.TteBytes,
                    a.NextnPre, a.NpreType, a.NpreNe0, a.NpreNe1, a.NpreBytes,
                    a.NextnPost, a.NpostType, a.NpostNe0, a.NpostNe1, a.NpostBytes,
                    a.DraftTokEmbd, a.DteType, a.DteNe0, a.DteNe1, a.DteBytes,
                    a.OutputNorm,
                    a.AttnNorm, a.Wq, a.WqType, a.WqNe0, a.WqNe1, a.WqBytes,
                    a.QNorm, a.Wo, a.WoType, a.WoNe0, a.WoNe1, a.WoBytes,
                    a.PostAttnNorm, a.FfnNorm,
                    a.Gate, a.GateType, a.GateNe0, a.GateNe1, a.GateBytes,
                    a.Up, a.UpType, a.UpNe0, a.UpNe1, a.UpBytes,
                    a.Down, a.DownType, a.DownNe0, a.DownNe1, a.DownBytes,
                    a.PostFfwNorm, a.OutScale,
                    a.Hd, a.KvHeads, a.IsLocal, a.RopeBase, a.RopeDims,
                    a.DonorK, a.DonorV, a.DonorCacheSize,
                    (IntPtr)lo, (IntPtr)ho);
            }
        }

        /// <summary>No recurrent state in Gemma 4 — nothing to snapshot per slot.</summary>
        public void MtpSnapshotRecurrentStateSlots(SequenceState seq) { }

        /// <summary>No recurrent state in Gemma 4 — nothing to restore. Paged
        /// attention needs no KV rewind: each pass passes its own sequence length,
        /// and rejected slots are overwritten by the kept-prefix re-forward.</summary>
        public void MtpRestoreRecurrentStateSlots(SequenceState seq) { }

        // Single-query attention of the draft's Q against the sequence's PAGED
        // donor K/V (_g4PagedK[donor], a host float[] indexed by slot). Mirrors the
        // linear AttentionDecodeWithWindow semantics: attend logical positions
        // [attendStart, fixedPos) of the sequence, windowed for SWA donors. Reads
        // q (host, post-q-norm/RoPE) and writes result in place — same host-pointer
        // style as the linear decode-attention kernels.
        private unsafe void MtpDraftPagedAttention(
            Tensor q, int donor, int kvHeads, int hd, int fixedPos, bool isLocal,
            SequenceState seq, Tensor result)
        {
            int numHeads = _mtpDraftHeads;
            int groupSize = numHeads / kvHeads;
            int stride = kvHeads * hd;                 // per-slot K/V stride
            int blockSize = _g4PagedBlockSize;
            int attendStart = isLocal ? Math.Max(0, fixedPos - _slidingWindow) : 0;
            int attendLen = fixedPos - attendStart;

            float* qPtr = GetFloatPtr(q);
            float* rPtr = GetFloatPtr(result);
            if (attendLen <= 0)
            {
                VecZero(rPtr, numHeads * hd);
                return;
            }

            if (_mtpDraftScores == null || _mtpDraftScores.Length < attendLen)
                _mtpDraftScores = new float[attendLen];

            var blocks = seq.BlockTable.Blocks;
            float[] pagedK = _g4PagedK[donor];
            float[] pagedV = _g4PagedV[donor];

            fixed (float* kBase = pagedK, vBase = pagedV)
            fixed (float* scores = _mtpDraftScores)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    float* qHead = qPtr + h * hd;
                    int kvHead = h / groupSize;

                    float maxScore = float.NegativeInfinity;
                    for (int t = 0; t < attendLen; t++)
                    {
                        int lp = attendStart + t;
                        long slot = (long)blocks[lp / blockSize].Id * blockSize + (lp % blockSize);
                        float s = VecDot(qHead, kBase + slot * stride + kvHead * hd, hd);
                        scores[t] = s;
                        if (s > maxScore) maxScore = s;
                    }

                    float sumExp = 0;
                    for (int t = 0; t < attendLen; t++)
                    {
                        float e = MathF.Exp(scores[t] - maxScore);
                        scores[t] = e;
                        sumExp += e;
                    }
                    float invSum = 1f / sumExp;

                    float* rHead = rPtr + h * hd;
                    VecZero(rHead, hd);
                    for (int t = 0; t < attendLen; t++)
                    {
                        int lp = attendStart + t;
                        long slot = (long)blocks[lp / blockSize].Id * blockSize + (lp % blockSize);
                        VecScaleAdd(rHead, vBase + slot * stride + kvHead * hd, scores[t] * invSum, hd);
                    }
                }
            }
        }
    }
}
