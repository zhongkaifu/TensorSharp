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
// ---------------------------------------------------------------------------
// The DFlash / DFlash2 block drafter, for ANY target model.
//
// DFlash is a BLOCK drafter: one forward pass proposes the whole speculative
// window, so it plugs into the shared draft/verify/rollback core through
// IDraftHead.DraftBlock instead of the per-token DraftStep, and it consumes a
// WIDE hidden row - the concatenated per-layer input residuals of the target
// layers its encoder was trained on (dflash.target_layers).
//
// Nothing here is model-specific. A target model gets DFlash by doing three
// things and nothing else:
//
//   1. call LoadDFlashDraftWeights(path) once the trunk weights are loaded,
//   2. capture the residual entering each dflash.target_layers layer during
//      its SpecForward (DFlashCaptureFeature does the row packing), and
//   3. forward the IDraftHead members to DFlashPropose / DFlashCommit.
//
// Three passes, transcribed from llama.cpp src/models/dflash.cpp and, for the
// DFlash2 extensions, from sglang python/sglang/srt/models/dflash.py:
//
//   PASS A -- encoder
//       feat = concat(target input residual of layers dflash.target_layers)
//       g    = rmsnorm(fc @ feat, enc.output_norm, eps)      [1 row per position]
//
//   PASS B -- KV injection
//       K = rope_neox(headnorm(attn_k @ g, attn_k_norm), target position)
//       V = attn_v @ g                                       (no norm, no rope)
//       ring[pos % ringRows] <- K, V                         per draft layer
//       No Q, no attention, no FFN -- and, in DFlash2, no convolution either:
//       the encoder output is the trained KV source for context positions.
//
//   PASS C -- block draft
//       ids  = [anchor, MASK x (block_size-1)] at positions p .. p+B-1
//       inpL = target token_embd[ids]                        (no embedding scale)
//       per draft layer:
//           h = rmsnorm(inpL, attn_norm)
//           [DFlash2] h, kOut = conv.prepare(h)
//           attention over [ring window | this block's own B keys] --
//               NON-CAUSAL inside the block, SWA-masked against the ring
//           [DFlash2] attn = conv.finish(attn, kOut)
//           inpL += attn ; f = rmsnorm(inpL, ffn_norm)
//           [DFlash2] f, kOut = conv.prepare(f)
//           f = SwiGLU(f) ; [DFlash2] f = conv.finish(f, kOut)
//           inpL += f
//       cur = rmsnorm(inpL, output_norm)
//       DFlash : logits = target_lm_head @ cur; each row's argmax is its draft
//       DFlash2: the candidate selector picks the block as a walk (below)
//
// The drafter's logits get NEITHER the target's logit_scale NOR its tanh
// softcap: llama.cpp's dflash graph ends at build_lora_mm(output, cur). argmax
// is invariant to both, but the softmax CONFIDENCE is not, so the per-position
// acceptance probabilities handed to the executor are the softmax of the RAW
// drafter logits.
//
// ---------------------------------------------------------------------------
// DFlash2, in one paragraph each.
//
// GROUPED DYNAMIC CONVOLUTION. Every attention and every FFN sublayer is
// wrapped: one projection of the sublayer's INPUT produces both the kernel that
// convolves that input and the kernel that convolves the sublayer's output.
// The kernel is a K-tap depthwise filter whose static part is per CHANNEL
// (blk.N.*_conv_base, initialised to the identity) and whose per-token delta is
// per GROUP of conv_group_size channels, so tap t of channel c at block
// position r is (base[t][c] + delta[r][t][c / group_size]) and multiplies
// x[r-t][c] - masked to zero for r < t, i.e. the filter never reaches across a
// block boundary. It is the piece that gives a block-diffusion draft a local
// left-to-right signal without a second forward pass.
//
// CANDIDATE SELECTOR. Plain DFlash takes each block position's argmax over the
// target vocabulary INDEPENDENTLY, which is exactly the weakness of block
// diffusion: position i+1 is chosen without knowing what position i chose. The
// selector keeps the top selector_top_k candidates per position and scores every
// (predecessor, candidate) pair through two low-rank [vocab, r] codebooks:
//
//     score[e][p][c] = unary[e][c]
//                    + < A[pred[e][p]] * (P h_e) , B[cand[e][c]] >
//
// where A / B are selector_predecessor / selector_successor, P is
// selector_hidden, pred[0] is the verified anchor token and pred[e] is
// cand[e-1]. The block is then read off as a greedy walk through that lattice,
// which is one extra small matmul per position and no extra draft forward.
// ---------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    public abstract partial class ModelBase
    {
        // Per-layer weight-name slots (index into _dflashLayerNames[il]).
        private protected const int DfAttnNorm = 0;
        private protected const int DfAttnQ = 1;
        private protected const int DfAttnK = 2;
        private protected const int DfAttnV = 3;
        private protected const int DfAttnQNorm = 4;
        private protected const int DfAttnKNorm = 5;
        private protected const int DfAttnOutput = 6;
        private protected const int DfFfnNorm = 7;
        private protected const int DfFfnGate = 8;
        private protected const int DfFfnUp = 9;
        private protected const int DfFfnDown = 10;
        // DFlash2 only; absent (null) in a first-generation drafter.
        private protected const int DfAttnConvBase = 11;
        private protected const int DfAttnConvProj = 12;
        private protected const int DfFfnConvBase = 13;
        private protected const int DfFfnConvProj = 14;
        // DSpark only (the Nemotron-3.5 attention-sink module); absent in a
        // plain DFlash / DFlash2 file.
        private protected const int DfAttnSinks = 15;
        private protected const int DfLayerNameCount = 16;

        /// <summary>
        /// Default prompt-prefill chunk the speculative executor should use, and
        /// the reason it is not the model's own.
        ///
        /// This value drives the TRUNK forward, not only the drafter, so it decides
        /// how many full target forwards a prompt costs. It used to be 128, chosen
        /// purely to bound the host-side capture buffer (one Muse-Glimmer feature
        /// row is 33280 floats = 130 KB), on the assumption that "the trunk's
        /// per-chunk fixed overhead is small next to a 128-row forward". Measured
        /// against llama.cpp it is not: the extra cost per chunk is a FLAT ~60 ms
        /// from 2K to 128K of context (a whole-trunk graph rebuild plus a DFlash
        /// host round trip), against ~13 ms of useful work in a 128-row chunk. At a
        /// 124K prompt that is 980 trunk forwards instead of 61 - 58 s added to a
        /// 112 s prefill, which was the whole of the 0.69x-versus-llama.cpp DFlash
        /// prefill ratio.
        ///
        /// 1024 keeps one 130 MB host buffer (PrefillStep shifts the pairing in
        /// place instead of keeping a second one) and removes 87% of the extra
        /// chunks. Override with TS_DFLASH_PREFILL_CHUNK.
        /// </summary>
        private const int DFlashPrefillChunkDefault = 1024;

        private int _dflashPrefillChunk;

        /// <summary>
        /// Trunk-imposed ceiling on one speculative prefill chunk. A model whose
        /// own forward refuses a batch wider than some window (Muse-Glimmer's SWA
        /// ring) narrows the chunk here; the default is "no trunk limit".
        /// </summary>
        private protected virtual int DFlashTrunkPrefillChunkCap => int.MaxValue;

        private int ResolveDFlashPrefillChunk()
        {
            int chunk = DFlashPrefillChunkDefault;
            string raw = Environment.GetEnvironmentVariable("TS_DFLASH_PREFILL_CHUNK");
            if (!string.IsNullOrWhiteSpace(raw) && int.TryParse(raw, out int parsed) && parsed > 0)
                chunk = parsed;

            // Never exceed what either ring can absorb in one forward: the drafter
            // would alias two live positions onto one ring slot, and a trunk with a
            // window limit throws outright.
            int draftCap = _dflash != null ? _dflash.RingRows - _dflash.BlockSize - 1 : chunk;
            chunk = Math.Min(chunk, Math.Max(1, draftCap));
            chunk = Math.Min(chunk, Math.Max(1, DFlashTrunkPrefillChunkCap));
            return Math.Max(1, chunk);
        }

        /// <summary>
        /// The prefill chunk a DFlash-drafted model reports through
        /// ISpeculativeTarget.SpecPrefillChunkSize. Resolved lazily and once: the
        /// caps come off the drafter's ring and the trunk's own window, so
        /// answering before the drafter exists would cache a wrong value.
        /// </summary>
        private protected int DFlashPrefillChunkSize
        {
            get
            {
                if (_dflash == null)
                    return 0;
                if (_dflashPrefillChunk <= 0)
                    _dflashPrefillChunk = ResolveDFlashPrefillChunk();
                return _dflashPrefillChunk;
            }
        }

        private protected DFlashConfig _dflash;
        private bool _hasDFlash;

        /// <summary>target layer index -> its column block in a feature row, or -1.</summary>
        private protected int[] _dflashCaptureSlot;

        /// <summary>Per draft layer, the weight names of that block.</summary>
        private string[][] _dflashLayerNames;

        /// <summary>The drafter's own KV ring, [numKVHeads, ringRows, headDim] per
        /// draft layer, indexed by (absolute position % ringRows).</summary>
        private Tensor[] _dflashRingK;
        private Tensor[] _dflashRingV;
        private int _dflashRingRows;

        /// <summary>Reused [vocab] scratch for the DSpark Markov chain (one block
        /// row at a time); kept across calls because a draft step allocates none
        /// of the hot buffers.</summary>
        private float[] _dflashMarkovScratch;

        /// <summary>True when a usable DFlash drafter is attached to this model.</summary>
        public bool HasDFlash => _hasDFlash;

        /// <summary>The attached drafter's hyper-parameters, or null.</summary>
        public DFlashConfig DFlashSettings => _dflash;

        // ====================================================================
        // construction / loading
        // ====================================================================

        /// <summary>
        /// Loads the DFlash drafter GGUF and attaches it to this target model. Its
        /// tensors are merged into the shared weight dictionaries under the
        /// "dflash." prefix so the existing matmul/norm machinery serves them (the
        /// same trick Gemma4Model.LoadMtpDraftTensors uses with "mtp."); the drafter
        /// borrows the TARGET's token_embd.weight and output.weight, which the file
        /// does not carry.
        /// </summary>
        public void LoadDFlashDraftWeights(string ggufPath)
        {
            if (string.IsNullOrEmpty(ggufPath) || !System.IO.File.Exists(ggufPath))
                throw new System.IO.FileNotFoundException("DFlash drafter GGUF not found.", ggufPath);

            using var draft = new GgufFile(ggufPath);
            var cfg = DFlashConfig.FromGguf(draft);

            if (cfg.HiddenSize != Config.HiddenSize)
            {
                throw new InvalidOperationException(
                    $"DFlash embedding_length {cfg.HiddenSize} != target hidden size {Config.HiddenSize}.");
            }
            foreach (int lid in cfg.TargetLayerIds)
            {
                if (lid < 0 || lid >= Config.NumLayers)
                {
                    throw new InvalidOperationException(
                        $"DFlash target layer {lid} is outside the target's {Config.NumLayers} layers.");
                }
            }
            if (cfg.BlockSize > cfg.RingRows)
                throw new InvalidOperationException($"DFlash block_size {cfg.BlockSize} exceeds the ring ({cfg.RingRows} rows).");

            LoadDFlashDraftTensors(draft);
            AttachDFlashSidecarScales();

            _dflash = cfg;
            _dflashLayerNames = new string[cfg.NumLayers][];
            for (int il = 0; il < cfg.NumLayers; il++)
            {
                string p = $"{DFlashConfig.WeightPrefix}blk.{il}.";
                var names = new string[DfLayerNameCount];
                names[DfAttnNorm] = p + "attn_norm.weight";
                names[DfAttnQ] = p + "attn_q.weight";
                names[DfAttnK] = p + "attn_k.weight";
                names[DfAttnV] = p + "attn_v.weight";
                names[DfAttnQNorm] = p + "attn_q_norm.weight";
                names[DfAttnKNorm] = p + "attn_k_norm.weight";
                names[DfAttnOutput] = p + "attn_output.weight";
                names[DfFfnNorm] = p + "ffn_norm.weight";
                names[DfFfnGate] = p + "ffn_gate.weight";
                names[DfFfnUp] = p + "ffn_up.weight";
                names[DfFfnDown] = p + "ffn_down.weight";
                if (cfg.HasConv)
                {
                    names[DfAttnConvBase] = p + "attn_conv_base";
                    names[DfAttnConvProj] = p + "attn_conv_proj.weight";
                    names[DfFfnConvBase] = p + "ffn_conv_base";
                    names[DfFfnConvProj] = p + "ffn_conv_proj.weight";
                }
                names[DfAttnSinks] = p + "attn_sinks";
                _dflashLayerNames[il] = names;
            }

            if (!VerifyDFlashTensors(out string missing))
            {
                Console.WriteLine($"  DFlash drafter GGUF loaded but '{missing}' is missing; DFlash drafting disabled.");
                _dflash = null;
                _dflashLayerNames = null;
                return;
            }

            _dflashCaptureSlot = new int[Config.NumLayers];
            for (int l = 0; l < Config.NumLayers; l++)
                _dflashCaptureSlot[l] = -1;
            for (int i = 0; i < cfg.TargetLayerIds.Length; i++)
                _dflashCaptureSlot[cfg.TargetLayerIds[i]] = i;

            _dflashRingRows = cfg.RingRows;
            _dflashRingK = new Tensor[cfg.NumLayers];
            _dflashRingV = new Tensor[cfg.NumLayers];
            for (int il = 0; il < cfg.NumLayers; il++)
            {
                _dflashRingK[il] = new Tensor(_allocator, DType.Float32, cfg.NumKVHeads, _dflashRingRows, cfg.HeadDim);
                _dflashRingV[il] = new Tensor(_allocator, DType.Float32, cfg.NumKVHeads, _dflashRingRows, cfg.HeadDim);
                // Unconditional zero fill (not InitializeCacheTensor, which skips
                // GgmlCuda): the ring is only ever read over positions that have
                // been written, but a finite ring keeps a mis-sized window from
                // silently producing NaNs instead of failing loudly.
                Ops.Fill(_dflashRingK[il], 0f);
                Ops.Fill(_dflashRingV[il], 0f);
            }

            _hasDFlash = true;

            long ringBytes = 2L * cfg.NumLayers * cfg.NumKVHeads * _dflashRingRows * cfg.HeadDim * sizeof(float);
            Console.WriteLine($"  DFlash drafter ready: {cfg}");
            Console.WriteLine($"  DFlash KV ring: {_dflashRingRows} rows x {cfg.NumLayers} layers ({ringBytes / (1024 * 1024)} MB F32)");
        }

        /// <summary>
        /// Merges every tensor of the drafter GGUF into the shared weight
        /// dictionaries under the "dflash." prefix. Byte-for-byte the same shape as
        /// Gemma4Model.LoadMtpDraftTensors (which uses "mtp."), minus the
        /// converter-spelling normalization DFlash does not need: the drafter's
        /// tensor names are already the final ones.
        /// </summary>
        private unsafe void LoadDFlashDraftTensors(GgufFile draft)
        {
            foreach (var kv in draft.Tensors)
            {
                var info = kv.Value;
                string name = DFlashConfig.WeightPrefix + info.Name;
                long byteCount = draft.GetTensorByteCount(info);

                // F16/BF16 linears are NOT "quantized" for this path: the GGML
                // AddmmQuant family only implements block-quant types, and
                // silently returns a zeroed output for anything else - the
                // drafter's BF16 attention weights ran "fine" while contributing
                // nothing at all. Dequantizing to F32 here keeps every drafter
                // weight on a matmul path that is actually implemented.
                bool isQuant = IsQuantizedLinearWeight(info)
                    && info.Type != GgmlTensorType.F16
                    && info.Type != GgmlTensorType.BF16;

                if (isQuant)
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
                        // Dequant with the MANAGED converter, not the GGML-native
                        // one: the drafter's BF16 linears are common (the
                        // Nemotron DSpark export keeps its attention path BF16,
                        // and this converter's own BF16 output routes here too),
                        // and the native dequant silently skips BF16, leaving the
                        // F32 tensor all zeros.
                        IntPtr tempPtr = QuantizedWeight.AllocateBuffer(byteCount);
                        try
                        {
                            draft.ReadTensorDataToNative(info, tempPtr, byteCount);
                            TensorSharp.Models.ManagedQuantizedOps.DequantizeToFloat32Native((int)info.Type, tempPtr, destPtr, numElements);
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

        /// <summary>
        /// Attaches the drafter's own NVFP4 (scale2) sidecars to its merged
        /// QuantizedWeights, mirroring the trunk's
        /// <see cref="AttachSidecarWeightScales"/> for the "dflash." prefix. The
        /// Nemotron-3.5 DSpark drafter stores fc, its FFN blocks and the Markov w2
        /// in NVFP4 with a 1-element "&lt;base&gt;.scale" sidecar each; leaving them
        /// unattached would run the drafter with unscaled weights.
        /// </summary>
        private void AttachDFlashSidecarScales()
        {
            int attached = 0;
            foreach (var kv in _quantWeights)
            {
                if (!kv.Key.StartsWith(DFlashConfig.WeightPrefix, StringComparison.Ordinal)
                    || !kv.Key.EndsWith(".weight", StringComparison.Ordinal))
                {
                    continue;
                }
                string scaleKey = kv.Key.Substring(0, kv.Key.Length - ".weight".Length) + ".scale";
                if (_weights.TryGetValue(scaleKey, out var st) && st.ElementCount() == 1)
                {
                    float v = st.GetElementsAsFloat(1)[0];
                    if (v != 1.0f)
                    {
                        kv.Value.Scale = v;
                        attached++;
                    }
                }
            }
            if (attached > 0)
                Console.WriteLine($"  DFlash drafter: {attached} NVFP4 scale2 sidecars attached.");
        }

        /// <summary>True when <paramref name="name"/> resolves to something
        /// <see cref="LinearForward"/> can multiply by.</summary>
        private bool HasDFlashLinear(string name)
            => _quantWeights.ContainsKey(name) || _weights.ContainsKey(name);

        private bool VerifyDFlashTensors(out string missing)
        {
            string[] globals =
            {
                DFlashConfig.WeightPrefix + "fc.weight",
                DFlashConfig.WeightPrefix + "enc.output_norm.weight",
                DFlashConfig.WeightPrefix + "output_norm.weight",
            };
            foreach (string g in globals)
            {
                bool ok = g.EndsWith("norm.weight", StringComparison.Ordinal)
                    ? _weights.ContainsKey(g)
                    : HasDFlashLinear(g);
                if (!ok) { missing = g; return false; }
            }

            for (int il = 0; il < _dflash.NumLayers; il++)
            {
                string[] n = _dflashLayerNames[il];
                foreach (int slot in new[] { DfAttnNorm, DfAttnQNorm, DfAttnKNorm, DfFfnNorm })
                {
                    if (!_weights.ContainsKey(n[slot])) { missing = n[slot]; return false; }
                }
                foreach (int slot in new[] { DfAttnQ, DfAttnK, DfAttnV, DfAttnOutput, DfFfnGate, DfFfnUp, DfFfnDown })
                {
                    if (!HasDFlashLinear(n[slot])) { missing = n[slot]; return false; }
                }
                if (_dflash.HasConv)
                {
                    // The conv base kernels are small F32 tensors, the projections
                    // ordinary linears - both are required together, because a
                    // half-built convolution silently changes the model.
                    foreach (int slot in new[] { DfAttnConvBase, DfFfnConvBase })
                    {
                        if (!_weights.ContainsKey(n[slot])) { missing = n[slot]; return false; }
                        long need = 2L * _dflash.ConvKernelSize * _dflash.HiddenSize;
                        if (_weights[n[slot]].ElementCount() != need)
                        {
                            missing = $"{n[slot]} (expected {need} elements for "
                                    + $"2 x taps {_dflash.ConvKernelSize} x hidden {_dflash.HiddenSize})";
                            return false;
                        }
                    }
                    foreach (int slot in new[] { DfAttnConvProj, DfFfnConvProj })
                    {
                        if (!HasDFlashLinear(n[slot])) { missing = n[slot]; return false; }
                    }
                }
            }

            if (_dflash.HasSelector)
            {
                foreach (string s in new[]
                {
                    DFlashConfig.WeightPrefix + "selector_hidden.weight",
                    DFlashConfig.WeightPrefix + "selector_predecessor.weight",
                    DFlashConfig.WeightPrefix + "selector_successor.weight",
                })
                {
                    if (!HasDFlashLinear(s)) { missing = s; return false; }
                }
            }

            if (_dflash.MarkovRank > 0)
            {
                // The DSpark Markov head: w1 embeds the previous draft token,
                // w2 maps it to a full-vocab logit bias. Exactly one generation is
                // valid on one drafter - a Markov file with a selector (or vice
                // versa) describes an export no runtime can execute as trained.
                if (_dflash.HasSelector)
                {
                    missing = "both a Markov head and a DFlash2 selector";
                    return false;
                }
                if (!HasDFlashLinear(DFlashConfig.WeightPrefix + "markov_w1.weight"))
                {
                    missing = DFlashConfig.WeightPrefix + "markov_w1.weight";
                    return false;
                }
                if (!HasDFlashLinear(DFlashConfig.WeightPrefix + "markov_w2.weight"))
                {
                    missing = DFlashConfig.WeightPrefix + "markov_w2.weight";
                    return false;
                }
                if (_dflash.HasAttentionSinks)
                {
                    // Attention sinks are all-or-nothing per drafter: a file with
                    // the layer-0 sink but no others would silently drop the bias
                    // from the remaining layers.
                    for (int il = 0; il < _dflash.NumLayers; il++)
                    {
                        string sink = _dflashLayerNames[il][DfAttnSinks];
                        if (!_weights.TryGetValue(sink, out var sinkW)
                            || sinkW.ElementCount() != _dflash.NumHeads)
                        {
                            missing = $"{sink} (expected one value per drafter head, {_dflash.NumHeads})";
                            return false;
                        }
                    }
                }
            }

            // The drafter has no LM head of its own: it borrows the target's.
            if (!HasDFlashLinear(DFlashTargetOutputWeightName)) { missing = DFlashTargetOutputWeightName; return false; }
            if (!HasDFlashLinear("token_embd.weight")) { missing = "token_embd.weight"; return false; }

            missing = null;
            return true;
        }

        /// <summary>The target's LM head, which the drafter borrows. A model that
        /// keeps its head somewhere other than "output.weight"/"token_embd.weight"
        /// overrides this.</summary>
        private protected virtual string DFlashTargetOutputWeightName
            => HasDFlashLinear("output.weight") ? "output.weight" : "token_embd.weight";

        /// <summary>Called from the owning model's Dispose. The drafter's weights
        /// live in the shared dictionaries and are released by
        /// <see cref="Dispose"/>; only the rings are ours.</summary>
        private protected void DisposeDFlash()
        {
            if (_dflashRingK != null)
                foreach (var t in _dflashRingK) t?.Dispose();
            if (_dflashRingV != null)
                foreach (var t in _dflashRingV) t?.Dispose();
            _dflashRingK = null;
            _dflashRingV = null;
            _hasDFlash = false;
        }

        // ====================================================================
        // IDraftHead surface (models forward to these)
        // ====================================================================

        /// <summary>Copies one target layer's input residual into its column block
        /// of the caller's feature rows. <paramref name="slot"/> comes from
        /// <see cref="_dflashCaptureSlot"/>.</summary>
        private protected unsafe void DFlashCaptureFeature(Tensor hidden, int slot, int seqLen, float[] hAllOut, bool lastRowOnly)
        {
            int hs = Config.HiddenSize;
            int feat = _dflash.FeatureSize;
            long rowBytes = (long)hs * sizeof(float);
            float* src = GetFloatPtr(hidden);
            fixed (float* dst0 = hAllOut)
            {
                if (lastRowOnly)
                {
                    Buffer.MemoryCopy(src + (long)(seqLen - 1) * hs, dst0 + (long)slot * hs, rowBytes, rowBytes);
                    return;
                }
                for (int r = 0; r < seqLen; r++)
                    Buffer.MemoryCopy(src + (long)r * hs, dst0 + (long)r * feat + (long)slot * hs, rowBytes, rowBytes);
            }
        }

        /// <summary>
        /// Replays committed trunk positions through the drafter so its KV ring
        /// tracks the real context. Row k of <paramref name="hRows"/> is the feature
        /// row of the token PRECEDING tokens[k] -- i.e. of absolute position
        /// <paramref name="startPos"/> + k - 1, which is exactly the position whose
        /// drafter key it writes (hence the -1, as in DeepSeek4Model.DraftCatchUp).
        /// </summary>
        private protected void DFlashCommit(int[] tokens, float[] hRows, int startPos)
        {
            RequireDFlash();
            if (tokens == null || tokens.Length == 0 || hRows == null)
                return;
            DFlashCatchUp(hRows, tokens.Length, startPos - 1);
        }

        /// <summary>Encodes <paramref name="rows"/> feature rows whose first row is
        /// at absolute position <paramref name="firstPos"/> and writes the resulting
        /// keys/values into the ring. Rows before position 0 (the zeroed "hidden
        /// state of the token before the prompt") are skipped, and rows older than
        /// the ring modulus are dropped -- dropping them is what keeps the ring
        /// writes collision-free.</summary>
        private void DFlashCatchUp(float[] hRows, int rows, int firstPos)
        {
            if (rows <= 0)
                return;
            int skip = firstPos < 0 ? Math.Min(-firstPos, rows) : 0;
            rows -= skip;
            firstPos += skip;
            if (rows <= 0)
                return;

            int keep = Math.Min(rows, _dflashRingRows);
            int drop = rows - keep;

            DFlashEncodeAndInject(hRows, skip + drop, keep, firstPos + drop);
        }

        /// <summary>Encode + ring injection, fused into one GGML graph when the backend
        /// supports it, otherwise the per-op pair.</summary>
        private void DFlashEncodeAndInject(float[] hRows, int rowOffset, int n, int startPos)
        {
            if (TryFusedDFlashInject(hRows, rowOffset, n, startPos))
                return;

            EnsureDFlashRingHostSynchronized();
            using Tensor g = DFlashEncode(hRows, rowOffset, n);
            DFlashInjectKv(g, n, startPos);
        }

        /// <summary>
        /// Drafts one block. <paramref name="hPrev"/> holds the target features of
        /// the last FORWARDED position (<paramref name="position"/> - 1) and
        /// <paramref name="lastToken"/> the token at <paramref name="position"/>
        /// (drawn but not yet forwarded). Returns the number of tokens written to
        /// <paramref name="draftOut"/>; <paramref name="confOut"/> receives their
        /// per-position acceptance estimates.
        /// </summary>
        private protected int DFlashPropose(int lastToken, float[] hPrev, int position, int[] draftOut, float[] confOut)
        {
            RequireDFlash();
            if (position <= 0 || draftOut == null || draftOut.Length == 0)
                return 0;

            // The drafter's own key for the last committed position, exactly like
            // the reference's ring[start_pos % win] = kv(main_x).
            DFlashEncodeAndInject(hPrev, 0, 1, position - 1);

            // A plain DFlash block needs one extra row for the anchor (row 0 is the
            // anchor's own prediction and is discarded). A Markov drafter is the
            // opposite: the anchor's row IS its first draft, so the block width
            // equals the number of drafts.
            int b = _dflash.MarkovRank > 0
                ? Math.Min(_dflash.BlockSize, draftOut.Length)
                : Math.Min(_dflash.BlockSize, draftOut.Length + 1);
            return DFlashDraftBlockCore(lastToken, position, b, draftOut, confOut);
        }

        private void RequireDFlash()
        {
            if (!_hasDFlash)
                throw new InvalidOperationException("No DFlash drafter is loaded for this model.");
        }

        // ====================================================================
        // PASS A -- encoder
        // ====================================================================

        /// <summary>
        /// g = rmsnorm(fc @ feat, enc.output_norm). <paramref name="rowOffset"/> is
        /// the first feature row of <paramref name="hRows"/> to consume and
        /// <paramref name="n"/> how many. Returns [n, hidden].
        /// </summary>
        private unsafe Tensor DFlashEncode(float[] hRows, int rowOffset, int n)
        {
            int feat = _dflash.FeatureSize;
            long need = (long)(rowOffset + n) * feat;
            if (hRows == null || hRows.LongLength < need)
            {
                throw new ArgumentException(
                    $"DFlash encoder needs {need} feature floats but got {hRows?.LongLength ?? 0}.", nameof(hRows));
            }

            var featTensor = new Tensor(_allocator, DType.Float32, n, feat);
            long bytes = (long)n * feat * sizeof(float);
            float* dst = GetFloatPtr(featTensor);
            fixed (float* src = &hRows[(long)rowOffset * feat])
                Buffer.MemoryCopy(src, dst, bytes, bytes);
            InvalidateTensorDeviceCache(featTensor);

            Tensor g = LinearForward(featTensor, DFlashConfig.WeightPrefix + "fc.weight");
            featTensor.Dispose();

            Ops.RMSNorm(g, g, _weights[DFlashConfig.WeightPrefix + "enc.output_norm.weight"], null, _dflash.Eps);
            return g;
        }

        // ====================================================================
        // PASS B -- KV injection
        // ====================================================================

        /// <summary>
        /// Writes the drafter's per-position keys/values for <paramref name="n"/>
        /// encoder rows starting at absolute position <paramref name="startPos"/>.
        /// K is head-normed and NeoX-RoPE'd at the TARGET position; V gets neither.
        /// The DFlash2 convolution does NOT run here: context positions enter the
        /// draft KV straight off the encoder, which is how the drafter was trained
        /// (sglang's DFlashAttention.kv_proj_only takes the projected target hidden
        /// with neither the input layernorm nor the conv applied).
        /// </summary>
        private void DFlashInjectKv(Tensor g, int n, int startPos)
        {
            var cfg = _dflash;
            int kvHeads = cfg.NumKVHeads, hd = cfg.HeadDim;

            int[] positions = new int[n];
            for (int i = 0; i < n; i++)
                positions[i] = startPos + i;
            // ASSIGNMENT, not max: this is the drafter's write FRONTIER, and it
            // has to be able to move backwards. A new sequence (a chat turn after
            // a cache reset, a fresh request) starts injecting at position 0 again,
            // and a frontier still parked at the previous sequence's length would
            // label every live ring slot with a position from that sequence - which
            // the draft mask then reads as "in the future" and drops, leaving the
            // drafter with no context at all. Injection is always contiguous and
            // forward WITHIN a sequence, so the frontier is exactly startPos + n.
            _dflashRingFilled = startPos + n;

            for (int il = 0; il < cfg.NumLayers; il++)
            {
                string[] names = _dflashLayerNames[il];

                Tensor k = LinearForward(g, names[DfAttnK]);
                Tensor v = LinearForward(g, names[DfAttnV]);

                k = DFlashHeadNorm(k, _weights[names[DfAttnKNorm]], kvHeads, n, hd);
                k = DFlashRoPE(k, kvHeads, n, hd, positions);

                using (Tensor kHeads = ReshapeToHeads(k, kvHeads, n, hd))
                    DFlashRingWrite(_dflashRingK[il], kHeads, startPos, n);
                using (Tensor vHeads = ReshapeToHeads(v, kvHeads, n, hd))
                    DFlashRingWrite(_dflashRingV[il], vHeads, startPos, n);

                k.Dispose();
                v.Dispose();
            }
        }

        /// <summary>Scatters a head-first [kvHeads, n, headDim] tensor into the ring
        /// at (startPos + i) % ringRows, splitting the write at the wrap.</summary>
        private void DFlashRingWrite(Tensor ring, Tensor headFirst, int startPos, int n)
        {
            int rows = _dflashRingRows;
            int keep = Math.Min(n, rows);
            int first = n - keep;                 // older rows would be overwritten anyway
            int done = 0;
            while (done < keep)
            {
                int slot = (startPos + first + done) % rows;
                int len = Math.Min(keep - done, rows - slot);
                using var src = headFirst.Narrow(1, first + done, len);
                CopyToCache(ring, src, slot, len);
                done += len;
            }
        }

        // ====================================================================
        // PASS C -- block draft
        // ====================================================================

        /// <summary>
        /// Runs [anchor, MASK x (b-1)] at positions p..p+b-1 through the drafter and
        /// fills <paramref name="draftOut"/> / <paramref name="confOut"/> from rows
        /// 1..b-1. Returns b-1.
        /// </summary>
        private unsafe int DFlashDraftBlockCore(int anchorToken, int position, int b, int[] draftOut, float[] confOut)
        {
            int fused = TryFusedDFlashDraftBlock(anchorToken, position, b, draftOut, confOut);
            if (fused >= 0)
                return fused;

            EnsureDFlashRingHostSynchronized();
            var cfg = _dflash;
            int heads = cfg.NumHeads, hd = cfg.HeadDim, kvHeads = cfg.NumKVHeads;
            float eps = cfg.Eps;

            int[] ids = new int[b];
            int[] positions = new int[b];
            for (int i = 0; i < b; i++)
            {
                ids[i] = i == 0 ? anchorToken : cfg.MaskTokenId;
                positions[i] = position + i;
            }

            // llama.cpp's dflash graph feeds build_inp_embd straight in: no
            // embedding scale, and (unlike some trunks) no weightless RMSNorm over
            // the embeddings.
            Tensor inpL = Embedding(ids);

            for (int il = 0; il < cfg.NumLayers; il++)
            {
                string[] names = _dflashLayerNames[il];

                Tensor h = DFlashRmsNorm(inpL, names[DfAttnNorm], eps);

                // DFlash2: one projection of the sublayer input yields both this
                // sublayer's input filter and the filter its OUTPUT is convolved
                // with, so the coefficients are computed once and held across the
                // attention.
                float[] attnFinishDelta = null;
                if (cfg.HasConv)
                    h = DFlashConvPrepare(h, b, names[DfAttnConvProj], names[DfAttnConvBase], out attnFinishDelta);

                Tensor q = LinearForward(h, names[DfAttnQ]);
                Tensor k = LinearForward(h, names[DfAttnK]);
                Tensor v = LinearForward(h, names[DfAttnV]);
                h.Dispose();

                q = DFlashHeadNorm(q, _weights[names[DfAttnQNorm]], heads, b, hd);
                k = DFlashHeadNorm(k, _weights[names[DfAttnKNorm]], kvHeads, b, hd);
                q = DFlashRoPE(q, heads, b, hd, positions);
                k = DFlashRoPE(k, kvHeads, b, hd, positions);
                Ops.Mul(q, q, 1f / MathF.Sqrt(hd));

                Tensor attn = DFlashBlockAttention(il, q, k, v, position, b);
                q.Dispose();
                k.Dispose();
                v.Dispose();

                Tensor attnOut = LinearForward(attn, names[DfAttnOutput]);
                attn.Dispose();

                if (attnFinishDelta != null)
                    attnOut = DFlashConvFinish(attnOut, b, attnFinishDelta, names[DfAttnConvBase]);

                Ops.Add(attnOut, attnOut, inpL);          // ffn_inp = attn + inpL
                inpL.Dispose();

                Tensor ffnIn = DFlashRmsNorm(attnOut, names[DfFfnNorm], eps);
                float[] ffnFinishDelta = null;
                if (cfg.HasConv)
                    ffnIn = DFlashConvPrepare(ffnIn, b, names[DfFfnConvProj], names[DfFfnConvBase], out ffnFinishDelta);

                Tensor ffnOut = DFlashSwiGLU(ffnIn, names[DfFfnGate], names[DfFfnUp], names[DfFfnDown]);
                ffnIn.Dispose();
                if (ffnFinishDelta != null)
                    ffnOut = DFlashConvFinish(ffnOut, b, ffnFinishDelta, names[DfFfnConvBase]);

                Ops.Add(attnOut, attnOut, ffnOut);        // inpL = ffn + ffn_inp
                ffnOut.Dispose();
                inpL = attnOut;
            }

            Tensor cur = DFlashRmsNorm(inpL, DFlashConfig.WeightPrefix + "output_norm.weight", eps);
            inpL.Dispose();

            if (cfg.HasSelector)
            {
                int produced = DFlashSelectorBlock(cur, b, anchorToken, draftOut, confOut);
                cur.Dispose();
                return produced;
            }

            // The TARGET's LM head, with NEITHER logit_scale NOR the tanh softcap:
            // llama.cpp's dflash graph ends at build_lora_mm(output, cur).
            Tensor logits = LinearForward(cur, DFlashTargetOutputWeightName);
            cur.Dispose();

            // DSpark: every row's logits get the Markov bias chained from the
            // previous draft (row 0 seeds from the anchor) and each row becomes a
            // draft; no row is discarded. The chain makes each position's token
            // depend on the one before it, which is the whole point of the head.
            if (cfg.MarkovRank > 0)
                return DFlashMarkovBlock(logits, b, anchorToken, draftOut, confOut);

            // Softmax on the backend, then a max scan per row: argmax is invariant
            // under softmax, and the winning probability IS the confidence the
            // executor multiplies cumulatively (a zero there drafts nothing).
            Ops.Softmax(logits, logits);

            int vocab = Config.VocabSize;
            int n = b - 1;
            float* lp = GetFloatPtr(logits);
            for (int i = 0; i < n; i++)
            {
                // Row 0 is the anchor's own prediction; plain DFlash discards it.
                float* row = lp + (long)(i + 1) * vocab;
                int best = DFlashArgmaxRow(row, vocab, out float prob);
                draftOut[i] = best;
                if (confOut != null && i < confOut.Length)
                    confOut[i] = prob;
            }
            logits.Dispose();
            return n;
        }

        /// <summary>
        /// DSpark block selection: turns the block's raw LM-head rows into one
        /// draft per position with the Markov head (llama.cpp's
        /// build_dspark_markov_head, sglang's VanillaMarkov):
        ///
        ///     bias_i = w2 @ w1[prev_i]      prev_0 = anchor, prev_{i+1} = argmax(col_i)
        ///     col_i  = base_i + bias_i      (or base_i unbiased when the anchor row
        ///                                    is a sample_from_anchor=false "bonus")
        ///     draft_i = argmax(col_i)
        ///
        /// The acceptance estimate of position i is the softmax probability of its
        /// argmax over col_i (the file's confidence head is not exported; the
        /// executor's cumulative gate consumes these exactly like plain DFlash's).
        ///
        /// Runs with the backend for the rank-wide matmul (w2 is NVFP4 in the
        /// shipped drafter; per-token row adds and the argmax/softmax scan stay on
        /// the host, same as the plain DFlash row scan).
        /// </summary>
        private unsafe int DFlashMarkovBlock(Tensor logits, int b, int anchorToken, int[] draftOut, float[] confOut)
        {
            var cfg = _dflash;
            int vocab = Config.VocabSize;
            int rank = cfg.MarkovRank;
            string w1Key = DFlashConfig.WeightPrefix + "markov_w1.weight";
            string w2Key = DFlashConfig.WeightPrefix + "markov_w2.weight";

            // w1 doubles as the Markov embedding: the chain reads w1[prev] as a
            // host row every step. The loader routes 2D "*.weight" tensors into
            // the QuantizedWeight table regardless of dtype (BF16 here), which is
            // right for matmul consumers but useless for row lookup, so the first
            // use dequantizes it into a plain F32 host tensor. F32-stored files
            // land in _weights directly and skip the dequant.
            Tensor w1 = null;
            if (!_weights.TryGetValue(w1Key, out w1) && _quantWeights.TryGetValue(w1Key, out var qw1))
            {
                long n = qw1.Ne0 * qw1.Ne1;
                var t = new Tensor(_allocator, DType.Float32, qw1.Ne1, qw1.Ne0);
                IntPtr destPtr = TensorComputePrimitives.GetStoragePointer(t);
                NativeDequant.DequantizeToFloat32Native((int)qw1.GgmlType, qw1.Data, destPtr, n);
                _weights[w1Key] = w1 = t;
            }
            if (w1 == null)
            {
                throw new InvalidOperationException(
                    $"DFlash Markov head: neither _weights nor _quantWeights has '{w1Key}'.");
            }
            if (w1.ElementCount() != (long)vocab * rank)
            {
                throw new InvalidOperationException(
                    $"DFlash markov_w1 is {w1.ElementCount()} elements, expected vocab {vocab} x rank {rank}.");
            }
            float* w1p = GetFloatPtr(w1);
            float* lp = GetFloatPtr(logits);
            if (_dflashMarkovScratch == null || _dflashMarkovScratch.Length < vocab)
                _dflashMarkovScratch = new float[vocab];
            float[] col = _dflashMarkovScratch;

            int prev = anchorToken < 0 ? 0 : anchorToken;
            var prevEmb = new Tensor(_allocator, DType.Float32, 1, rank);
            float* pep = GetFloatPtr(prevEmb);
            try
            {
                for (int i = 0; i < b; i++)
                {
                    float* row = lp + (long)i * vocab;
                    bool biased = cfg.SampleFromAnchor || i > 0;
                    if (biased)
                    {
                        // prevEmb = w1[prev] (the last draft token's embedding).
                        Buffer.MemoryCopy(w1p + (long)prev * rank, pep, (long)rank * 4, (long)rank * 4);
                        InvalidateTensorDeviceCache(prevEmb);

                        using (Tensor biasT = LinearForward(prevEmb, w2Key))
                        {
                            // bias = w2 @ w1[prev] + then col = base row + bias.
                            float* bias = GetFloatPtr(biasT);
                            for (int c = 0; c < vocab; c++)
                                col[c] = row[c] + bias[c];
                        }
                    }
                    else
                    {
                        // sample_from_anchor=false: row 0 is the bonus anchor, its
                        // own (unbiased) prediction, and the chain starts from the
                        // anchor token again for row 1 (llama.cpp leaves prev alone
                        // for the bonus row).
                        fixed (float* colp = col)
                            Buffer.MemoryCopy(row, colp, (long)vocab * 4, (long)vocab * 4);
                    }

                    int best = 0;
                    float bestVal = col[0];
                    for (int c = 1; c < vocab; c++)
                    {
                        if (col[c] > bestVal)
                        {
                            bestVal = col[c];
                            best = c;
                        }
                    }

                    // Softmax probability of the winner: exp(col[best] - max) / sum.
                    double sum = 0.0;
                    for (int c = 0; c < vocab; c++)
                        sum += Math.Exp(col[c] - bestVal);

                    draftOut[i] = best;
                    if (confOut != null && i < confOut.Length)
                        confOut[i] = (float)(1.0 / sum);

                    if (biased && i + 1 < b)
                        prev = best;
                }
            }
            finally
            {
                prevEmb.Dispose();
                logits.Dispose();
            }
            return b;
        }

        private static unsafe int DFlashArgmaxRow(float* row, int n, out float best)
        {
            int bestIdx = 0;
            float bestVal = row[0];
            for (int i = 1; i < n; i++)
            {
                float v = row[i];
                if (v > bestVal)
                {
                    bestVal = v;
                    bestIdx = i;
                }
            }
            best = bestVal;
            return bestIdx;
        }

        /// <summary>
        /// One draft layer's attention: the b block queries attend
        /// [ring window | this block's own b keys], NON-CAUSALLY inside the block
        /// (llama_set_causal_attn(ctx_dft, false)) and sliding-window masked against
        /// the ring (a cached key at p0 is masked from a query at p1 when
        /// p1 - p0 &gt;= n_swa).
        /// </summary>
        private unsafe Tensor DFlashBlockAttention(int il, Tensor q, Tensor k, Tensor v, int position, int b)
        {
            var cfg = _dflash;
            int kvHeads = cfg.NumKVHeads, hd = cfg.HeadDim, heads = cfg.NumHeads;
            int groupSize = heads / kvHeads;
            int rings = _dflashRingRows;

            // The FIRST query (position p) sees cached keys down to p - (n_swa - 1);
            // later queries in the block see a strict subset, masked below.
            int winStart = Math.Max(0, position - (cfg.SlidingWindow - 1));
            int w = position - winStart;
            int total = w + b;

            var gk = new Tensor(_allocator, DType.Float32, kvHeads, total, hd);
            var gv = new Tensor(_allocator, DType.Float32, kvHeads, total, hd);

            int done = 0;
            while (done < w)
            {
                int slot = (winStart + done) % rings;
                int len = Math.Min(w - done, rings - slot);
                using (var srcK = _dflashRingK[il].Narrow(1, slot, len))
                using (var dstK = gk.Narrow(1, done, len))
                    Ops.Copy(dstK, srcK);
                using (var srcV = _dflashRingV[il].Narrow(1, slot, len))
                using (var dstV = gv.Narrow(1, done, len))
                    Ops.Copy(dstV, srcV);
                done += len;
            }

            using (Tensor kHeads = ReshapeToHeads(k, kvHeads, b, hd))
            using (var dstBlockK = gk.Narrow(1, w, b))
                Ops.Copy(dstBlockK, kHeads);
            using (Tensor vHeads = ReshapeToHeads(v, kvHeads, b, hd))
            using (var dstBlockV = gv.Narrow(1, w, b))
                Ops.Copy(dstBlockV, vHeads);

            // GQA without materializing the expanded K/V: a contiguous head-first
            // [heads, b, hd] query tensor reinterprets exactly as
            // [kvHeads, groupSize*b, hd] (heads g*gs..g*gs+gs-1 are adjacent blocks
            // of b*hd), so one batched GEMM per kv head serves its whole query
            // group. ExpandKVHeads would instead allocate heads*total*hd floats per
            // layer per draft purely to repeat rows.
            Tensor qHeads = ReshapeToHeads(q, heads, b, hd);
            Tensor scores;
            using (Tensor qGrouped = qHeads.View(kvHeads, (long)groupSize * b, hd))
            using (Tensor kT = gk.Transpose(1, 2))
            {
                scores = new Tensor(_allocator, DType.Float32, kvHeads, (long)groupSize * b, total);
                Ops.AddmmBatch(scores, 0, scores, 1f, qGrouped, kT);
            }
            qHeads.Dispose();
            gk.Dispose();

            DFlashApplyWindowMask(scores, b, groupSize, kvHeads, w, total, position, winStart);
            Ops.Softmax(scores, scores);
            if (cfg.HasAttentionSinks)
                DFlashApplyAttentionSinks(scores, il, b, groupSize, kvHeads, total);

            var attnGrouped = new Tensor(_allocator, DType.Float32, kvHeads, (long)groupSize * b, hd);
            Ops.AddmmBatch(attnGrouped, 0, attnGrouped, 1f, scores, gv);
            scores.Dispose();
            gv.Dispose();

            Tensor attn;
            using (Tensor attnHeads = attnGrouped.View(heads, b, hd))
                attn = ReshapeFromHeads(attnHeads, heads, b, hd);
            attnGrouped.Dispose();
            return attn;
        }

        /// <summary>
        /// Masks the cached (ring) columns a query cannot see. Query row j of kv
        /// group g belongs to block slot s = j % b (see the grouping comment in
        /// <see cref="DFlashBlockAttention"/>), i.e. absolute position
        /// <paramref name="position"/> + s; cached column c holds position
        /// <paramref name="winStart"/> + c and is masked when
        /// (position + s) - (winStart + c) &gt;= n_swa. The block's own b columns
        /// are never masked -- attention inside the block is non-causal.
        /// </summary>
        private unsafe void DFlashApplyWindowMask(Tensor scores, int b, int groupSize, int kvHeads,
            int w, int total, int position, int winStart)
        {
            if (w <= 0)
                return;

            int swa = _dflash.SlidingWindow;
            Span<int> widths = stackalloc int[b];
            bool any = false;
            for (int s = 0; s < b; s++)
            {
                int width = position + s - swa - winStart + 1;
                if (width < 0) width = 0;
                if (width > w) width = w;
                widths[s] = width;
                any |= width > 0;
            }
            if (!any)
                return;

            float* sp = GetFloatPtr(scores);
            int rowsPerGroup = groupSize * b;
            for (int g = 0; g < kvHeads; g++)
            {
                float* groupScores = sp + (long)g * rowsPerGroup * total;
                for (int j = 0; j < rowsPerGroup; j++)
                {
                    int width = widths[j % b];
                    if (width > 0)
                        new Span<float>(groupScores + (long)j * total, width).Fill(float.NegativeInfinity);
                }
            }
            InvalidateTensorDeviceCache(scores);
        }

        /// <summary>
        /// Adds the drafter's per-head attention-sink bias to the normalized
        /// attention weights, mirroring llama.cpp's build_attn (the sink lands on
        /// the POST-softmax scores, a fixed attention-mass floor per head). The
        /// sink is keyed on the QUERY head there, so query row (g, j) of the
        /// grouped scores -- head g*groupSize + j/b, block slot j % b -- gets
        /// sink[head] added to every key column.
        /// </summary>
        private unsafe void DFlashApplyAttentionSinks(Tensor scores, int il, int b, int groupSize, int kvHeads, int total)
        {
            Tensor sinkW = _weights[_dflashLayerNames[il][DfAttnSinks]];
            if (sinkW.ElementCount() != (long)kvHeads * groupSize)
            {
                throw new InvalidOperationException(
                    $"DFlash attn_sinks of draft layer {il} has {sinkW.ElementCount()} values "
                    + $"for {kvHeads * groupSize} heads; refusing to run a half-wired sink.");
            }

            float* sp = GetFloatPtr(scores);
            float* sink = GetFloatPtr(sinkW);
            int rowsPerGroup = groupSize * b;
            for (int g = 0; g < kvHeads; g++)
            {
                float* groupScores = sp + (long)g * rowsPerGroup * total;
                for (int j = 0; j < rowsPerGroup; j++)
                {
                    float s = sink[g * groupSize + j / b];
                    if (s == 0f)
                        continue;
                    float* row = groupScores + (long)j * total;
                    for (int c = 0; c < total; c++)
                        row[c] += s;
                }
            }
            InvalidateTensorDeviceCache(scores);
        }

        // ====================================================================
        // DFlash2 -- grouped dynamic convolution
        // ====================================================================

        /// <summary>
        /// Convolves a sublayer's input and returns the coefficients its OUTPUT is
        /// convolved with. One projection produces both halves, which is why they
        /// cannot be separate calls: the output filter is keyed on the INPUT, not
        /// on whatever the sublayer produced.
        ///
        /// Consumes <paramref name="x"/> and returns a new [b, hidden] tensor.
        /// </summary>
        private unsafe Tensor DFlashConvPrepare(Tensor x, int b, string projName, string baseName,
            out float[] finishDelta)
        {
            var cfg = _dflash;
            int taps = cfg.ConvKernelSize, groups = cfg.ConvNumGroups;
            int half = taps * groups;

            using Tensor coef = LinearForward(x, projName);   // [b, 2 * taps * groups]
            float* cp = GetFloatPtr(coef);
            int stride = cfg.ConvProjOutSize;

            // The output half is copied out because the sublayer between prepare
            // and finish reuses (and may free) every device buffer in flight.
            finishDelta = new float[(long)b * half];
            for (int r = 0; r < b; r++)
                for (int i = 0; i < half; i++)
                    finishDelta[(long)r * half + i] = cp[(long)r * stride + half + i];

            Tensor result = DFlashConvApply(x, b, cp, stride, /*deltaOffset=*/0, baseName, /*side=*/0);
            x.Dispose();
            return result;
        }

        /// <summary>Applies the output-side filter produced by
        /// <see cref="DFlashConvPrepare"/>. Consumes <paramref name="y"/>.</summary>
        private unsafe Tensor DFlashConvFinish(Tensor y, int b, float[] finishDelta, string baseName)
        {
            int half = _dflash.ConvKernelSize * _dflash.ConvNumGroups;
            fixed (float* dp = finishDelta)
            {
                Tensor result = DFlashConvApply(y, b, dp, half, /*deltaOffset=*/0, baseName, /*side=*/1);
                y.Dispose();
                return result;
            }
        }

        /// <summary>
        /// out[r][c] = sum over taps t of (base[side][t][c] + delta[r][t][c / group])
        ///             * x[r-t][c], with the t-th tap zeroed for the first t rows of
        /// the block (the filter never reaches across a block boundary).
        ///
        /// Runs on the host: this is the per-op fallback, b is one block (8-16 rows)
        /// and the whole thing is b*hidden*taps multiply-adds - two orders of
        /// magnitude below one of the projections around it. The fused kernel does
        /// it in the graph.
        /// </summary>
        private unsafe Tensor DFlashConvApply(Tensor x, int b, float* delta, int deltaStride, int deltaOffset,
            string baseName, int side)
        {
            var cfg = _dflash;
            var outT = new Tensor(_allocator, DType.Float32, b, cfg.HiddenSize);
            DFlashGroupedConvolve(
                GetFloatPtr(x), GetFloatPtr(outT), GetFloatPtr(_weights[baseName]), delta,
                b, cfg.HiddenSize, cfg.ConvKernelSize, cfg.ConvGroupSize, cfg.ConvNumGroups,
                deltaStride, deltaOffset, side);
            InvalidateTensorDeviceCache(outT);
            return outT;
        }

        /// <summary>
        /// The convolution arithmetic itself, on raw rows, so the fused graph has
        /// something independent to be checked against:
        ///
        ///   out[r][c] = sum over taps t of
        ///               (base[side][t][c] + delta[r][t][c / groupSize]) * x[r-t][c]
        ///
        /// with tap t contributing nothing for r &lt; t. baseKernel is the
        /// checkpoint's [side, tap, hidden] block; delta is row-major with
        /// deltaStride floats per row, its (tap, group) pairs starting at
        /// deltaOffset.
        /// </summary>
        internal static unsafe void DFlashGroupedConvolve(
            float* x, float* dst0, float* baseKernel, float* delta,
            int rows, int hidden, int taps, int groupSize, int groups,
            int deltaStride, int deltaOffset, int side)
        {
            for (int r = 0; r < rows; r++)
            {
                float* dst = dst0 + (long)r * hidden;
                for (int tap = 0; tap < taps; tap++)
                {
                    if (tap > r)
                    {
                        // Masked tap: the filter never reaches across the block
                        // boundary. Tap 0 is never masked, so row 0 is still written.
                        continue;
                    }
                    float* src = x + (long)(r - tap) * hidden;
                    float* bt = baseKernel + ((long)side * taps + tap) * hidden;
                    float* dt = delta + (long)r * deltaStride + deltaOffset + (long)tap * groups;
                    if (tap == 0)
                    {
                        for (int g = 0; g < groups; g++)
                        {
                            float d = dt[g];
                            int c0 = g * groupSize;
                            for (int c = c0; c < c0 + groupSize; c++)
                                dst[c] = (bt[c] + d) * src[c];
                        }
                    }
                    else
                    {
                        for (int g = 0; g < groups; g++)
                        {
                            float d = dt[g];
                            int c0 = g * groupSize;
                            for (int c = c0; c < c0 + groupSize; c++)
                                dst[c] += (bt[c] + d) * src[c];
                        }
                    }
                }
            }
        }

        /// <summary>Managed-array entry point for the convolution, for tests and for
        /// callers that do not already hold pinned rows.</summary>
        internal static unsafe float[] DFlashGroupedConvolve(
            float[] x, float[] baseKernel, float[] delta,
            int rows, int hidden, int taps, int groupSize,
            int deltaStride, int deltaOffset, int side)
        {
            var result = new float[(long)rows * hidden];
            fixed (float* xp = x)
            fixed (float* bp = baseKernel)
            fixed (float* dp = delta)
            fixed (float* op = result)
            {
                DFlashGroupedConvolve(xp, op, bp, dp, rows, hidden, taps, groupSize,
                    hidden / groupSize, deltaStride, deltaOffset, side);
            }
            return result;
        }

        // ====================================================================
        // DFlash2 -- candidate selector
        // ====================================================================

        /// <summary>
        /// Turns the block's post-norm hidden states into a token per position by
        /// walking the transition lattice instead of taking b-1 independent argmaxes.
        ///
        /// Row 0 of <paramref name="cur"/> is the anchor's own hidden state and is
        /// not a proposal; the gamma = b-1 rows after it are, and each contributes
        /// selector_top_k candidates. Returns gamma.
        /// </summary>
        private unsafe int DFlashSelectorBlock(Tensor cur, int b, int anchorToken, int[] draftOut, float[] confOut)
        {
            var cfg = _dflash;
            int gamma = b - 1;
            int k = cfg.SelectorTopK;
            int rank = cfg.SelectorRank;
            int vocab = Config.VocabSize;

            Tensor predRows;
            using (var view = cur.Narrow(0, 1, gamma))
                predRows = Ops.NewContiguous(view);

            // Unary term: the ordinary DFlash head, kept only at its top-k.
            int[] candIds = new int[gamma * k];
            float[] unary = new float[gamma * k];
            using (Tensor logits = LinearForward(predRows, DFlashTargetOutputWeightName))
            {
                float* lp = GetFloatPtr(logits);
                for (int e = 0; e < gamma; e++)
                    DFlashTopK(lp + (long)e * vocab, vocab, k, candIds, unary, e * k);
            }
            // The target's own logit transform, applied AFTER the top-k because both
            // halves of it are monotonic (so the candidate set is unchanged) and doing
            // it here costs gamma*k operations instead of gamma*vocab.
            DFlashTransformUnary(unary);

            // P h, once per position.
            float[] projected = new float[gamma * rank];
            using (Tensor ph = LinearForward(predRows, DFlashConfig.WeightPrefix + "selector_hidden.weight"))
            {
                float* pp = GetFloatPtr(ph);
                for (int i = 0; i < gamma * rank; i++)
                    projected[i] = pp[i];
            }
            predRows.Dispose();

            // B[cand] for every candidate, and A[pred] for every predecessor: the
            // anchor for position 0, and position e-1's candidates for position e.
            float[] succ = DFlashGatherCodebook(DFlashConfig.WeightPrefix + "selector_successor.weight",
                candIds, gamma * k, rank);

            int[] predIds = new int[gamma * k];
            for (int p = 0; p < k; p++)
                predIds[p] = anchorToken;
            Array.Copy(candIds, 0, predIds, k, (gamma - 1) * k);
            float[] pred = DFlashGatherCodebook(DFlashConfig.WeightPrefix + "selector_predecessor.weight",
                predIds, gamma * k, rank);

            // Greedy walk. Position 0's predecessor row is the anchor's, replicated
            // over p, so only p = 0 is scored there.
            float[] row = new float[k];
            float[] m = new float[rank];
            int chosen = 0;
            for (int e = 0; e < gamma; e++)
            {
                int pRow = e == 0 ? 0 : chosen;
                long predBase = ((long)e * k + pRow) * rank;
                long projBase = (long)e * rank;
                for (int r = 0; r < rank; r++)
                    m[r] = pred[predBase + r] * projected[projBase + r];

                for (int c = 0; c < k; c++)
                {
                    long keyBase = ((long)e * k + c) * rank;
                    float dot = 0f;
                    for (int r = 0; r < rank; r++)
                        dot += m[r] * succ[keyBase + r];
                    row[c] = unary[(long)e * k + c] + dot;
                }

                chosen = 0;
                for (int c = 1; c < k; c++)
                    if (row[c] > row[chosen]) chosen = c;

                if (DFlashSelectorDebug && _dflashSelectorDebugBlocks < 3)
                {
                    // Attribution for the lattice: if the transition term never moves
                    // the choice off the unary argmax, the selector is doing nothing
                    // and its cost is pure loss.
                    int unaryBest = 0;
                    float loMin = float.MaxValue, loMax = float.MinValue;
                    for (int c = 0; c < k; c++)
                    {
                        float u = unary[(long)e * k + c];
                        if (u > unary[(long)e * k + unaryBest]) unaryBest = c;
                        float d = row[c] - u;
                        if (d < loMin) loMin = d;
                        if (d > loMax) loMax = d;
                    }
                    Console.WriteLine(
                        $"  [dflash-sel] slot {e}: unary[{unaryBest}]={unary[(long)e * k + unaryBest]:F3} " +
                        $"range=[{MinOf(unary, e * k, k):F3},{MaxOf(unary, e * k, k):F3}] " +
                        $"transition=[{loMin:F3},{loMax:F3}] chose={chosen}{(chosen == unaryBest ? " (= unary argmax)" : " (MOVED)")}");
                }

                draftOut[e] = candIds[(long)e * k + chosen];
                if (confOut != null && e < confOut.Length)
                    confOut[e] = DFlashSoftmaxAt(row, chosen);
            }

            if (DFlashSelectorDebug)
                _dflashSelectorDebugBlocks++;
            return gamma;
        }

        /// <summary>TS_DFLASH_SELECTOR_DEBUG=1 prints the first few blocks' lattice
        /// attribution: the unary spread, the transition spread, and whether the walk
        /// left the unary argmax. Managed (per-op) drafter only.</summary>
        private static readonly bool DFlashSelectorDebug =
            string.Equals(Environment.GetEnvironmentVariable("TS_DFLASH_SELECTOR_DEBUG"), "1", StringComparison.Ordinal);

        private int _dflashSelectorDebugBlocks;

        private static float MinOf(float[] a, long off, int n)
        {
            float v = a[off];
            for (int i = 1; i < n; i++) if (a[off + i] < v) v = a[off + i];
            return v;
        }

        private static float MaxOf(float[] a, long off, int n)
        {
            float v = a[off];
            for (int i = 1; i < n; i++) if (a[off + i] > v) v = a[off + i];
            return v;
        }

        /// <summary>
        /// scale, then tanh-softcap: the target's LM-head transform, which the
        /// selector's unary term has to carry because the lattice ADDS it to a
        /// transition score rather than taking an argmax over it. A no-op on a
        /// checkpoint whose target has neither (Qwen 3.8).
        /// </summary>
        private void DFlashTransformUnary(float[] unary)
        {
            var cfg = _dflash;
            if (!cfg.HasUnaryLogitTransform)
                return;
            float scale = cfg.LogitScale;
            float cap = cfg.FinalLogitSoftcap;
            for (int i = 0; i < unary.Length; i++)
            {
                float v = unary[i] * scale;
                unary[i] = cap > 0f ? MathF.Tanh(v / cap) * cap : v;
            }
        }

        /// <summary>Top <paramref name="k"/> of one logits row, unsorted, written at
        /// <paramref name="outOffset"/> of the id/value arrays. A k-element
        /// insertion scan: k is 16 and the row is the vocabulary, so anything that
        /// touches each element once is the right shape.</summary>
        internal static unsafe void DFlashTopK(float* rowPtr, int n, int k, int[] idsOut, float[] valsOut, int outOffset)
        {
            // Seed with the first k entries and track the weakest of them.
            int worst = 0;
            for (int i = 0; i < k; i++)
            {
                idsOut[outOffset + i] = i;
                valsOut[outOffset + i] = rowPtr[i];
                if (rowPtr[i] < valsOut[outOffset + worst]) worst = i;
            }
            float cutoff = valsOut[outOffset + worst];
            for (int i = k; i < n; i++)
            {
                float v = rowPtr[i];
                if (v <= cutoff)
                    continue;
                idsOut[outOffset + worst] = i;
                valsOut[outOffset + worst] = v;
                worst = 0;
                for (int j = 1; j < k; j++)
                    if (valsOut[outOffset + j] < valsOut[outOffset + worst]) worst = j;
                cutoff = valsOut[outOffset + worst];
            }
        }

        /// <summary>Managed-array entry point for the top-k selection.</summary>
        internal static unsafe void DFlashTopK(float[] row, int k, int[] idsOut, float[] valsOut, int outOffset)
        {
            fixed (float* rp = row)
                DFlashTopK(rp, row.Length, k, idsOut, valsOut, outOffset);
        }

        /// <summary>softmax(row)[index], computed with the usual max shift.</summary>
        internal static float DFlashSoftmaxAt(float[] row, int index)
        {
            float max = row[0];
            for (int i = 1; i < row.Length; i++)
                if (row[i] > max) max = row[i];
            double sum = 0;
            for (int i = 0; i < row.Length; i++)
                sum += Math.Exp(row[i] - max);
            return sum > 0 ? (float)(Math.Exp(row[index] - max) / sum) : 0f;
        }

        /// <summary>Gathers <paramref name="count"/> rows of a [vocab, rank]
        /// selector codebook into a flat host array. The codebooks are ordinary
        /// quantized tensors, so this is the same row-gather the token embedding
        /// uses.</summary>
        private unsafe float[] DFlashGatherCodebook(string weightName, int[] ids, int count, int rank)
        {
            var result = new float[(long)count * rank];
            int[] rows = ids;
            if (ids.Length != count)
            {
                rows = new int[count];
                Array.Copy(ids, rows, count);
            }

            if (_quantWeights.TryGetValue(weightName, out var qw))
            {
                using var gathered = new Tensor(_allocator, DType.Float32, count, rank);
                PopulateQuantizedRows(gathered, qw, rows);
                float* gp = GetFloatPtr(gathered);
                for (long i = 0; i < (long)count * rank; i++)
                    result[i] = gp[i];
                return result;
            }

            Tensor w = _weights[weightName];
            float* wp = GetFloatPtr(w);
            for (int i = 0; i < count; i++)
            {
                long src = (long)rows[i] * rank;
                for (int r = 0; r < rank; r++)
                    result[(long)i * rank + r] = wp[src + r];
            }
            return result;
        }

        // ====================================================================
        // small shared pieces
        // ====================================================================

        /// <summary>RMSNorm against a named drafter weight with the drafter's own
        /// epsilon. The trunk helpers hardcode Config.Eps, which is not the
        /// drafter's.</summary>
        private Tensor DFlashRmsNorm(Tensor input, string weightName, float eps)
        {
            var alpha = _weights[weightName];
            int rows = (int)input.Sizes[0];
            int dim = (int)(input.ElementCount() / rows);
            Tensor input2d = input.Sizes.Length != 2 ? input.View(rows, dim) : null;
            Tensor src = input2d ?? input;
            Tensor result = Ops.RMSNorm(null, src, alpha, null, eps);
            input2d?.Dispose();
            return result;
        }

        /// <summary>Per-head RMSNorm over a [rows, numHeads*headDim] tensor, with
        /// the drafter's own epsilon. Consumes <paramref name="data"/>.</summary>
        private Tensor DFlashHeadNorm(Tensor data, Tensor alpha, int numHeads, int rows, int headDim)
        {
            using var reshaped = data.View((long)rows * numHeads, headDim);
            Tensor normed = Ops.RMSNorm(null, reshaped, alpha, null, _dflash.Eps);
            data.Dispose();
            Tensor flat = normed.View(rows, (long)numHeads * headDim);
            normed.Dispose();
            return flat;
        }

        /// <summary>
        /// NeoX-flavour RoPE (split halves) over a [rows, numHeads*headDim] tensor
        /// with an explicit per-row position. llama.cpp maps LLM_ARCH_DFLASH to
        /// LLAMA_ROPE_TYPE_NEOX, so this is mode 2 -- NOT the interleaved-pair
        /// (mode 0) RoPE some trunks use. Consumes <paramref name="data"/>.
        /// </summary>
        private Tensor DFlashRoPE(Tensor data, int numHeads, int rows, int headDim, int[] positions)
        {
            int totalRows = rows * numHeads;
            int[] rowPositions = new int[totalRows];
            for (int s = 0; s < rows; s++)
                for (int h = 0; h < numHeads; h++)
                    rowPositions[s * numHeads + h] = positions[s];
            using var posTensor = CreateIntTensorOn(data.Storage.Allocator, rowPositions, totalRows);

            using var reshaped = data.View(1, rows, numHeads, headDim);
            Tensor result = Ops.RoPEEx(
                null, reshaped, posTensor, headDim, DFlashConfig.RopeTypeNeoX, 0,
                _dflash.RopeBase, 1.0f,
                0.0f, 1.0f, 0.0f, 0.0f);

            data.Dispose();
            Tensor flat = result.View(rows, (long)numHeads * headDim);
            result.Dispose();
            return flat;
        }

        /// <summary>silu(gate(x)) * up(x) -> down. The drafter ships gate and up as
        /// SEPARATE tensors (FuseGateUpWeights only fuses the target's "blk.{l}."
        /// names), so this cannot use ModelBase.FFN.</summary>
        private Tensor DFlashSwiGLU(Tensor input, string gateName, string upName, string downName)
        {
            Tensor gate = LinearForward(input, gateName);
            Tensor up = LinearForward(input, upName);
            Ops.SiLUMul(gate, gate, up);
            up.Dispose();
            Tensor down = LinearForward(gate, downName);
            gate.Dispose();
            return down;
        }
    }
}
