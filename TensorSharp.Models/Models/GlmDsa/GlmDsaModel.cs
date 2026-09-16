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
// GLM (glm-dsa) — Zhipu/Z.ai GLM-5.x with DeepSeek Sparse Attention.
//
// The block is a DeepSeek-V3.2 derivative:
//   * Multi-head Latent Attention (MLA) with the "absorption" optimisation, so
//     the KV cache holds ONE compressed row per token
//     (kv_lora_rank + n_rot = 576 floats) instead of per-head K and V.
//   * A "lightning indexer" (DSA) that scores every cached token with a cheap
//     32-head/128-dim attention and keeps only the top-k (2048) for the real
//     attention. GLM differs from DeepSeek 3.2 in that only some layers compute
//     a fresh top-k; the rest reuse the previous full layer's selection
//     (glm-dsa.attention.indexer.types, default pattern 1,1,1,0,0,0,1,0,0,0...).
//   * Sigmoid-gated MoE with a routing bias, weight renormalisation, a x2.5
//     routed scale, and one always-on shared expert. The first
//     leading_dense_block_count layers are dense SwiGLU instead.
//   * One trailing NextN/MTP block (block_count includes it; the trunk is
//     block_count - nextn_predict_layers layers).
//
// Reference: llama.cpp src/models/glm-dsa.cpp (graph), src/llama-graph.cpp
// (build_attn for DSA + build_moe_ffn), src/llama-kv-cache-dsa.cpp (two caches).
using System;
using System.Collections.Generic;
using System.Diagnostics;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models
{
    public partial class GlmDsaModel : ModelBase, IBatchedPagedModel
    {
        // ---- architecture ----------------------------------------------------

        /// <summary>Trunk decoder blocks (block_count minus the NextN/MTP blocks).</summary>
        private int _numTrunkLayers;
        /// <summary>Every block in the file, including the NextN/MTP block(s).</summary>
        private int _numAllLayers;
        /// <summary>NextN/MTP blocks at the end of the file (1 for GLM-5.2, 0 when absent).</summary>
        private int _numNextnLayers;
        /// <summary>
        /// Index of the NextN/MTP draft block, or -1 when the checkpoint declares
        /// <c>nextn_predict_layers</c> but ships no MTP tensors (a trunk-only
        /// re-quantization, which llama.cpp also accepts and runs without a draft
        /// head). Set by <see cref="ResolveMtpLayer"/> once the weights are in.
        /// </summary>
        private int _mtpLayer = -1;
        /// <summary>Layers that own an MLA cache: the trunk, plus the MTP block when it loaded.</summary>
        private int NumCacheLayers => _mtpLayer >= 0 ? _numTrunkLayers + 1 : _numTrunkLayers;
        /// <summary>Leading dense (non-MoE) blocks.</summary>
        private int _numDenseLead;

        private int _qLoraRank;      // 2048
        private int _kvLoraRank;     // 512
        private int _headDimK;       // n_embd_head_k_mla, 256
        private int _headDimV;       // n_embd_head_v_mla, 256
        private int _ropeDim;        // n_rot, 64
        private int _headDimNope;    // _headDimK - _ropeDim, 192
        private int _kvRowDim;       // _kvLoraRank + _ropeDim, 576 (one cache row)

        private int _numExperts;         // 256
        private int _numExpertsUsed;     // 8
        private int _numSharedExperts;   // 1
        private int _expertFfnLength;    // 2048
        private float _expertWeightsScale;
        private bool _expertWeightsNorm;
        private int _expertGatingFunc;   // 1 = softmax, 2 = sigmoid

        private int _indexerHeads;       // 32
        private int _indexerHeadDim;     // 128
        private int _indexerTopK;        // 2048
        private bool[] _indexerFull;     // per trunk layer: computes a fresh top-k

        /// <summary>Softmax scale of the MLA attention: 1/sqrt(n_embd_head_k_mla).</summary>
        private float _kqScale;

        // ---- caches ----------------------------------------------------------

        /// <summary>
        /// Per-layer MLA cache, split so both halves stay row-contiguous:
        /// <c>_kvCache[l]</c> is [capacity, kv_lora_rank] (the compressed KV, which is
        /// also the attention's V) and <c>_kPeCache[l]</c> is [capacity, n_rot] (the
        /// RoPE'd key tail). llama.cpp keeps them concatenated in one 576-wide row;
        /// splitting them lets the score product run as two contiguous GEMMs and the
        /// context product read V without materialising a strided copy.
        /// </summary>
        private Tensor[] _kvCache;
        private Tensor[] _kPeCache;
        /// <summary>Per-layer lightning-indexer key cache: [capacity, indexer_head_dim]. Null on layers that reuse a previous top-k.</summary>
        private Tensor[] _indexerCache;
        private int _kvCacheCapacity;

        // ---- weights ---------------------------------------------------------

        /// <summary>Per-layer wk_b as [n_head, kv_lora_rank, n_embd_head_qk_nope] (GGUF order reversed).</summary>
        private Tensor[] _wkB;
        /// <summary>Per-layer wv_b as [n_head, n_embd_head_v_mla, kv_lora_rank] (GGUF order reversed).</summary>
        private Tensor[] _wvB;

        private string[][] _layerNames;
        private StackedExpertWeights[] _stackedGate;
        private StackedExpertWeights[] _stackedUp;
        private StackedExpertWeights[] _stackedDown;
        private QuantizedWeight[][] _expertGateQW;
        private QuantizedWeight[][] _expertUpQW;
        private QuantizedWeight[][] _expertDownQW;

        // ---- scratch ---------------------------------------------------------

        private float[] _routerScratch;
        private int[] _selectedExperts;
        private float[] _routingWeights;
        /// <summary>Top-k token indices from the last full-indexer layer, reused by the "shared" layers.</summary>
        private int[] _sharedTopK;
        private int _sharedTopKCount;

        // Names of the tensors that make up one decoder block. Index constants
        // keep the per-layer forward free of string formatting.
        private const int WN_ATTN_NORM = 0;
        private const int WN_Q_A = 1;
        private const int WN_Q_A_NORM = 2;
        private const int WN_Q_B = 3;
        private const int WN_KV_A_MQA = 4;
        private const int WN_KV_A_NORM = 5;
        private const int WN_ATTN_OUT = 6;
        private const int WN_FFN_NORM = 7;
        private const int WN_FFN_GATE = 8;
        private const int WN_FFN_UP = 9;
        private const int WN_FFN_DOWN = 10;
        private const int WN_GATE_INP = 11;
        private const int WN_EXP_PROBS_B = 12;
        private const int WN_SH_GATE = 13;
        private const int WN_SH_UP = 14;
        private const int WN_SH_DOWN = 15;
        private const int WN_IDX_Q_B = 16;
        private const int WN_IDX_K = 17;
        private const int WN_IDX_K_NORM_W = 18;
        private const int WN_IDX_K_NORM_B = 19;
        private const int WN_IDX_PROJ = 20;
        // NextN/MTP wiring. Only the trailing MTP block(s) carry these; on a
        // trunk layer every one of them resolves to a missing tensor.
        private const int WN_NEXTN_EH_PROJ = 21;
        private const int WN_NEXTN_ENORM = 22;
        private const int WN_NEXTN_HNORM = 23;
        private const int WN_NEXTN_SHARED_HEAD_NORM = 24;
        private const int WN_NEXTN_EMBED_TOKENS = 25;
        private const int WN_NEXTN_SHARED_HEAD_HEAD = 26;
        private const int WN_COUNT = 27;

        public GlmDsaModel(string ggufPath, BackendType backend, int tpDegree = 1, ITensorParallelGroup tpGroup = null)
            : base(ggufPath, NormalizeBackend(backend), tpDegree, tpGroup)
        {
            string arch = _gguf.GetString("general.architecture") ?? "glm-dsa";
            Config = new ModelConfig { Architecture = arch };
            ParseBaseConfig();
            ParseGlmDsaConfig(arch);
            ParseTokenizer();

            Console.WriteLine($"Model: {arch}, Layers={_numTrunkLayers}" +
                (_numNextnLayers > 0 ? $" (+{_numNextnLayers} NextN/MTP block in the file)" : "") +
                $", Hidden={Config.HiddenSize}, " +
                $"Heads={Config.NumHeads}, MLA(q_lora={_qLoraRank}, kv_lora={_kvLoraRank}, head_k={_headDimK}, head_v={_headDimV}, rope={_ropeDim}), " +
                $"Vocab={Config.VocabSize}");
            Console.WriteLine($"  MoE: {_numExperts} experts, top-{_numExpertsUsed}, ffn={_expertFfnLength}, shared={_numSharedExperts}, " +
                $"scale={_expertWeightsScale}, norm={_expertWeightsNorm}, gating={(_expertGatingFunc == 2 ? "sigmoid" : "softmax")}, dense_lead={_numDenseLead}");
            Console.WriteLine($"  DSA indexer: {_indexerHeads} heads x {_indexerHeadDim}, top-k={_indexerTopK}, " +
                $"full layers={CountIndexerFull()}/{_numTrunkLayers}");
            Console.WriteLine($"RoPE base={Config.RopeBase}, scale={Config.RopeScale}, eps={Config.Eps}");

            int maxContextLength = ResolveConfiguredContextLength();

            if (NativeRequested(backend))
            {
                // The whole model lives in the native executor: no managed weight
                // tensors, no managed caches.
                InitNativeExecutor(ggufPath, backend, tpDegree, maxContextLength);
                return;
            }

            // glm5next now has a managed per-op path (GlmDsaModel.Glm5Next.cs): KDA
            // linear attention and Sinkhorn hyper-connections both run in C#, so the
            // pure-C# `cpu` backend serves this family too.
            ParseGlm5NextConfig(arch, _numTrunkLayers);

            LoadWeights();
            BuildLayerNames();
            CacheMoeWeightHandles();
            ResolveMtpLayer();

            if (IsTensorParallel)
            {
                ShardGlmDsaWeightsForTP();
                PrepareCudaQuantizedWeightsForInferenceTP();
            }
            else
            {
                PrepareCudaQuantizedWeightsForInference();
            }

            int initialCacheLength = ResolveInitialCacheAllocationLength(maxContextLength);
            if (initialCacheLength < maxContextLength)
                Console.WriteLine($"Initial {_backend} KV cache allocation: {initialCacheLength} tokens (grows on demand up to {maxContextLength}).");

            InitCaches(initialCacheLength, maxContextLength);
            AllocateScratch();
        }

        /// <summary>
        /// glm5next's KDA recurrence cannot be rewound to an earlier position, so a
        /// cached prefix is only reusable when the new prompt EXTENDS it exactly
        /// (same contract as the retained Qwen hybrid GDN models). glm-dsa proper has no
        /// recurrent state and keeps the base behaviour.
        /// </summary>
        public override bool SupportsKVCacheTruncation => Config.Architecture != "glm5next";

        private int CountIndexerFull()
        {
            int n = 0;
            for (int i = 0; i < _indexerFull.Length; i++) if (_indexerFull[i]) n++;
            return n;
        }

        /// <summary>
        /// glm-dsa hyper-parameters. Mirrors
        /// <c>llama_model_glm_dsa::load_arch_hparams</c> including its defaults, so a
        /// GGUF written by any converter that llama.cpp accepts loads identically here.
        /// </summary>
        private void ParseGlmDsaConfig(string arch)
        {
            _numAllLayers = Config.NumLayers;
            _numNextnLayers = (int)_gguf.GetUint32($"{arch}.nextn_predict_layers", 0);
            if (_numNextnLayers >= _numAllLayers)
                throw new InvalidDataException_GlmDsa($"nextn_predict_layers ({_numNextnLayers}) must be < block_count ({_numAllLayers}).");
            _numTrunkLayers = _numAllLayers - _numNextnLayers;

            // ModelBase's generic helpers (cache sizing, per-layer loops) all mean
            // "trunk layers"; the MTP block is driven separately.
            Config.NumLayers = _numTrunkLayers;

            _numDenseLead = (int)_gguf.GetUint32($"{arch}.leading_dense_block_count", 0);

            _qLoraRank = (int)_gguf.GetUint32($"{arch}.attention.q_lora_rank");
            _kvLoraRank = (int)_gguf.GetUint32($"{arch}.attention.kv_lora_rank");
            _ropeDim = (int)_gguf.GetUint32($"{arch}.rope.dimension_count");

            // n_embd_head_k_mla / n_embd_head_v_mla are the per-head sizes AFTER the
            // wk_b / wv_b decompression. They default to key_length / value_length,
            // which for a non-MLA file are the same thing.
            _headDimK = (int)_gguf.GetUint32($"{arch}.attention.key_length_mla", (uint)Config.KeyLength);
            _headDimV = (int)_gguf.GetUint32($"{arch}.attention.value_length_mla", (uint)Config.ValueLength);
            _headDimNope = _headDimK - _ropeDim;
            _kvRowDim = _kvLoraRank + _ropeDim;

            if (_headDimNope <= 0)
                throw new InvalidDataException_GlmDsa($"key_length_mla ({_headDimK}) must exceed rope.dimension_count ({_ropeDim}).");
            if (Config.KeyLength != _kvRowDim)
                Console.WriteLine($"  note: attention.key_length={Config.KeyLength} but kv_lora_rank+n_rot={_kvRowDim}; using {_kvRowDim} for the MLA cache row.");

            _numExperts = (int)_gguf.GetUint32($"{arch}.expert_count");
            _numExpertsUsed = (int)_gguf.GetUint32($"{arch}.expert_used_count");
            _numSharedExperts = (int)_gguf.GetUint32($"{arch}.expert_shared_count", 0);
            _expertFfnLength = (int)_gguf.GetUint32($"{arch}.expert_feed_forward_length");
            _expertWeightsScale = _gguf.GetFloat32($"{arch}.expert_weights_scale", 1.0f);
            _expertWeightsNorm = _gguf.GetBool($"{arch}.expert_weights_norm", false);
            // 1 = softmax, 2 = sigmoid; glm-dsa defaults to sigmoid when unset.
            _expertGatingFunc = (int)_gguf.GetUint32($"{arch}.expert_gating_func", 2);
            if (_expertGatingFunc == 0) _expertGatingFunc = 2;

            Config.NumExperts = _numExperts;
            Config.NumExpertsUsed = _numExpertsUsed;

            int expertGroups = (int)_gguf.GetUint32($"{arch}.expert_group_count", 1);
            int expertGroupsUsed = (int)_gguf.GetUint32($"{arch}.expert_group_used_count", 1);
            if (expertGroups > 1 && expertGroupsUsed != expertGroups)
                throw new NotSupportedException(
                    $"glm-dsa with {expertGroups} expert groups (top-{expertGroupsUsed}) is not supported yet; " +
                    "GLM-5.x ships a single group.");

            _indexerHeads = (int)_gguf.GetUint32($"{arch}.attention.indexer.head_count");
            _indexerHeadDim = (int)_gguf.GetUint32($"{arch}.attention.indexer.key_length");
            _indexerTopK = (int)_gguf.GetUint32($"{arch}.attention.indexer.top_k");
            _indexerFull = ResolveIndexerTypes(arch);

            _kqScale = 1.0f / MathF.Sqrt(_headDimK);

            Config.VocabSize = (int)_gguf.GetUint32($"{arch}.vocab_size", 0);
            if (Config.VocabSize == 0)
            {
                var toks = _gguf.GetStringArray("tokenizer.ggml.tokens");
                Config.VocabSize = toks?.Length ?? 0;
            }

            Config.OriginalContextLength = (int)_gguf.GetUint32($"{arch}.rope.scaling.original_context_length", 0);
        }

        /// <summary>
        /// Which trunk layers run a full lightning indexer (compute a fresh top-k) and
        /// which reuse the previous full layer's selection.
        ///
        /// <para>GLM-5.0/5.1 gave every layer a full indexer; 5.2 keeps one full layer
        /// every four (plus the first three) and shares its selection with the rest.
        /// The GGUFs published so far do not carry the array, so — exactly as
        /// llama.cpp does — the pattern is inferred: a 5.2-class file (1M training
        /// context) gets the published 1,1,1,0,0,0,1,0,0,0,... pattern, anything
        /// shorter gets all-full.</para>
        /// </summary>
        private bool[] ResolveIndexerTypes(string arch)
        {
            var full = new bool[_numTrunkLayers];

            if (arch == "glm5next")
            {
                // GLM-5.3-Flash: attention.head_count_kv is a per-layer array,
                // 0 on KDA (linear-attention) layers and 1 on MLA+DSA layers.
                // Every MLA layer carries a full pooled indexer; KDA layers have
                // none and never share a selection.
                var kvh = _gguf.GetInt32Array($"{arch}.attention.head_count_kv")
                          ?? ToInt32(_gguf.GetUint32Array($"{arch}.attention.head_count_kv"));
                if (kvh != null)
                {
                    for (int i = 0; i < full.Length && i < kvh.Length; i++)
                        full[i] = kvh[i] != 0;
                    return full;
                }
            }

            int trainCtx = (int)_gguf.GetUint32($"{arch}.context_length", 0);
            bool pre52 = trainCtx > 0 && trainCtx < 1048576;

            if (pre52)
            {
                for (int i = 0; i < full.Length; i++) full[i] = true;
            }
            else
            {
                // https://huggingface.co/zai-org/GLM-5.2/blob/main/config.json — layers
                // 0,1,2 then every 4th layer from 6 onwards are "full".
                for (int i = 0; i < full.Length; i++)
                    full[i] = i < 3 || ((i - 2) % 4 == 0);
            }

            var explicitTypes = _gguf.GetInt32Array($"{arch}.attention.indexer.types")
                                ?? ToInt32(_gguf.GetUint32Array($"{arch}.attention.indexer.types"));
            if (explicitTypes != null)
            {
                for (int i = 0; i < full.Length && i < explicitTypes.Length; i++)
                    full[i] = explicitTypes[i] != 0;
            }

            if (full.Length > 0 && !full[0])
                throw new InvalidDataException_GlmDsa("the first layer must have a full DSA indexer (nothing precedes it to share a top-k with).");
            return full;
        }

        private static int[] ToInt32(uint[] src)
        {
            if (src == null) return null;
            var dst = new int[src.Length];
            for (int i = 0; i < src.Length; i++) dst[i] = (int)src[i];
            return dst;
        }

        private sealed class InvalidDataException_GlmDsa : InvalidOperationException
        {
            public InvalidDataException_GlmDsa(string message) : base("glm-dsa: " + message) { }
        }

        private void BuildLayerNames()
        {
            _layerNames = new string[_numAllLayers][];
            for (int l = 0; l < _numAllLayers; l++)
            {
                string p = $"blk.{l}.";
                var n = new string[WN_COUNT];
                n[WN_ATTN_NORM] = p + "attn_norm.weight";
                n[WN_Q_A] = p + "attn_q_a.weight";
                n[WN_Q_A_NORM] = p + "attn_q_a_norm.weight";
                n[WN_Q_B] = p + "attn_q_b.weight";
                n[WN_KV_A_MQA] = p + "attn_kv_a_mqa.weight";
                n[WN_KV_A_NORM] = p + "attn_kv_a_norm.weight";
                n[WN_ATTN_OUT] = p + "attn_output.weight";
                n[WN_FFN_NORM] = p + "ffn_norm.weight";
                n[WN_FFN_GATE] = p + "ffn_gate.weight";
                n[WN_FFN_UP] = p + "ffn_up.weight";
                n[WN_FFN_DOWN] = p + "ffn_down.weight";
                n[WN_GATE_INP] = p + "ffn_gate_inp.weight";
                n[WN_EXP_PROBS_B] = p + "exp_probs_b.bias";
                n[WN_SH_GATE] = p + "ffn_gate_shexp.weight";
                n[WN_SH_UP] = p + "ffn_up_shexp.weight";
                n[WN_SH_DOWN] = p + "ffn_down_shexp.weight";
                n[WN_IDX_Q_B] = p + "indexer.attn_q_b.weight";
                n[WN_IDX_K] = p + "indexer.attn_k.weight";
                n[WN_IDX_K_NORM_W] = p + "indexer.k_norm.weight";
                n[WN_IDX_K_NORM_B] = p + "indexer.k_norm.bias";
                n[WN_IDX_PROJ] = p + "indexer.proj.weight";
                n[WN_NEXTN_EH_PROJ] = p + "nextn.eh_proj.weight";
                n[WN_NEXTN_ENORM] = p + "nextn.enorm.weight";
                n[WN_NEXTN_HNORM] = p + "nextn.hnorm.weight";
                n[WN_NEXTN_SHARED_HEAD_NORM] = p + "nextn.shared_head_norm.weight";
                n[WN_NEXTN_EMBED_TOKENS] = p + "nextn.embed_tokens.weight";
                n[WN_NEXTN_SHARED_HEAD_HEAD] = p + "nextn.shared_head_head.weight";
                _layerNames[l] = n;
            }

            _wkB = new Tensor[_numAllLayers];
            _wvB = new Tensor[_numAllLayers];
            for (int l = 0; l < _numAllLayers; l++)
            {
                _weights.TryGetValue($"blk.{l}.attn_k_b.weight", out _wkB[l]);
                _weights.TryGetValue($"blk.{l}.attn_v_b.weight", out _wvB[l]);
            }

            for (int l = 0; l < _numTrunkLayers; l++)
            {
                // glm5next is KDA-recurrent on most of its trunk and only its
                // full-attention layers carry MLA at all, so the absorbed tensors
                // exist on roughly a quarter of them (GLM-5.3-Flash: 12 of 45,
                // starting at layer 3). Demanding them everywhere is right for
                // GLM-5.2, where every layer is MLA, but rejects a perfectly good
                // 5.3 checkpoint at layer 0. _layerIsRecurrent is null for the
                // non-glm5next families, which keeps their behaviour unchanged.
                if (_layerIsRecurrent != null && l < _layerIsRecurrent.Length && _layerIsRecurrent[l])
                    continue;

                if (_wkB[l] == null || _wvB[l] == null)
                    throw new InvalidDataException_GlmDsa(
                        $"layer {l} is missing attn_k_b / attn_v_b. This build requires the MLA-absorbed " +
                        "checkpoint layout (llama.cpp refuses glm-dsa without MLA as well).");
                ValidateAbsorbShape(_wkB[l], $"blk.{l}.attn_k_b.weight", Config.NumHeads, _kvLoraRank, _headDimNope);
                ValidateAbsorbShape(_wvB[l], $"blk.{l}.attn_v_b.weight", Config.NumHeads, _headDimV, _kvLoraRank);
            }
        }

        private static void ValidateAbsorbShape(Tensor t, string name, long d0, long d1, long d2)
        {
            if (t.Sizes.Length != 3 || t.Sizes[0] != d0 || t.Sizes[1] != d1 || t.Sizes[2] != d2)
                throw new InvalidOperationException(
                    $"glm-dsa: {name} has shape [{string.Join(',', t.Sizes.ToArray())}], expected [{d0},{d1},{d2}].");
        }

        private void CacheMoeWeightHandles()
        {
            _stackedGate = new StackedExpertWeights[_numAllLayers];
            _stackedUp = new StackedExpertWeights[_numAllLayers];
            _stackedDown = new StackedExpertWeights[_numAllLayers];
            _expertGateQW = new QuantizedWeight[_numAllLayers][];
            _expertUpQW = new QuantizedWeight[_numAllLayers][];
            _expertDownQW = new QuantizedWeight[_numAllLayers][];

            for (int l = 0; l < _numAllLayers; l++)
            {
                _stackedExpertWeights.TryGetValue($"blk.{l}.ffn_gate_exps.weight", out _stackedGate[l]);
                _stackedExpertWeights.TryGetValue($"blk.{l}.ffn_up_exps.weight", out _stackedUp[l]);
                _stackedExpertWeights.TryGetValue($"blk.{l}.ffn_down_exps.weight", out _stackedDown[l]);

                if (_stackedGate[l] == null)
                    continue;

                _expertGateQW[l] = new QuantizedWeight[_numExperts];
                _expertUpQW[l] = new QuantizedWeight[_numExperts];
                _expertDownQW[l] = new QuantizedWeight[_numExperts];
                for (int e = 0; e < _numExperts; e++)
                {
                    _quantWeights.TryGetValue($"blk.{l}.ffn_gate_exps.{e}.weight", out _expertGateQW[l][e]);
                    _quantWeights.TryGetValue($"blk.{l}.ffn_up_exps.{e}.weight", out _expertUpQW[l][e]);
                    _quantWeights.TryGetValue($"blk.{l}.ffn_down_exps.{e}.weight", out _expertDownQW[l][e]);
                }
            }
        }

        /// <summary>Routed experts stay in host RAM on layers the operator offloaded (--n-cpu-moe).</summary>
        protected override bool ShouldPreloadCudaQuantWeightToDevice(string weightName)
            => !MoeCpuOffloadConfig.IsOffloadedExpertWeightName(weightName);

        private void AllocateScratch()
        {
            _routerScratch = new float[_numExperts];
            _selectedExperts = new int[_numExpertsUsed];
            _routingWeights = new float[_numExpertsUsed];
        }

        private void InitCaches(int initialSeqLen, int maxSeqLen)
        {
            _maxContextLength = maxSeqLen;
            _kvCacheCapacity = initialSeqLen;
            // The MTP block owns one more MLA cache. It needs no indexer cache:
            // the draft head runs DENSE attention (llama.cpp's graph_mtp does the
            // same), so its lightning-indexer weights are never read.
            int nCache = NumCacheLayers;
            _kvCache = new Tensor[nCache];
            _kPeCache = new Tensor[nCache];
            _indexerCache = new Tensor[nCache];
            for (int l = 0; l < nCache; l++)
            {
                _kvCache[l] = new Tensor(_allocator, DType.Float32, initialSeqLen, _kvLoraRank);
                InitializeCacheTensor(_kvCache[l]);
                // NoPE MLA (glm5next) has no rope half, so there is no pe row to
                // cache -- allocating it would be a zero-width tensor.
                if (_ropeDim > 0)
                {
                    _kPeCache[l] = new Tensor(_allocator, DType.Float32, initialSeqLen, _ropeDim);
                    InitializeCacheTensor(_kPeCache[l]);
                }
                if (l < _numTrunkLayers && _indexerFull[l])
                {
                    _indexerCache[l] = new Tensor(_allocator, DType.Float32, initialSeqLen, _indexerHeadDim);
                    InitializeCacheTensor(_indexerCache[l]);
                }
            }
            _cacheSeqLen = 0;
        }

        private void EnsureCacheCapacity(int requiredSeqLen)
        {
            if (requiredSeqLen <= _kvCacheCapacity)
                return;
            if (requiredSeqLen > _maxContextLength)
                throw new InvalidOperationException(
                    $"Requested sequence length {requiredSeqLen} exceeds configured max context {_maxContextLength}.");

            int newCapacity = Math.Max(_kvCacheCapacity, 1);
            while (newCapacity < requiredSeqLen)
                newCapacity = Math.Min(_maxContextLength, newCapacity * 2);

            for (int l = 0; l < NumCacheLayers; l++)
            {
                GrowCache(ref _kvCache[l], newCapacity, _kvLoraRank);
                if (_kPeCache[l] != null)
                    GrowCache(ref _kPeCache[l], newCapacity, _ropeDim);
                if (_indexerCache[l] != null)
                    GrowCache(ref _indexerCache[l], newCapacity, _indexerHeadDim);
            }
            _kvCacheCapacity = newCapacity;
            Console.WriteLine($"Expanded glm-dsa caches to {newCapacity} tokens.");
        }

        private void GrowCache(ref Tensor cache, int newCapacity, int rowDim)
        {
            var grown = new Tensor(_allocator, DType.Float32, newCapacity, rowDim);
            InitializeCacheTensor(grown);
            if (_cacheSeqLen > 0)
            {
                using var src = cache.Narrow(0, 0, _cacheSeqLen);
                using var dst = grown.Narrow(0, 0, _cacheSeqLen);
                Ops.Copy(dst, src);
            }
            cache.Dispose();
            cache = grown;
        }

        private void DisposeCaches()
        {
            if (_kvCache != null)
            {
                for (int l = 0; l < _kvCache.Length; l++) _kvCache[l]?.Dispose();
                _kvCache = null;
            }
            if (_kPeCache != null)
            {
                for (int l = 0; l < _kPeCache.Length; l++) _kPeCache[l]?.Dispose();
                _kPeCache = null;
            }
            if (_indexerCache != null)
            {
                for (int l = 0; l < _indexerCache.Length; l++) _indexerCache[l]?.Dispose();
                _indexerCache = null;
            }
            _cachedPosQ?.Dispose(); _cachedPosQ = null;
            _cachedPosK?.Dispose(); _cachedPosK = null;
            _cachedPosIdx?.Dispose(); _cachedPosIdx = null;
            _cachedPosSeqLen = -1;
            _cachedPosStartPos = -1;
        }

        protected override void ResetKVCacheCore()
        {
            if (UsesNativeExecutor)
            {
                ResetNative();
                Glm5NextInvalidateSnapshot();
                return;
            }

            // glm5next carries recurrent KDA state alongside the attention cache;
            // leaving it behind would silently condition the next sequence on this
            // one, which no amount of KV clearing would show up as.
            ResetKdaState();
            Glm5NextInvalidateSnapshot();

            _cacheSeqLen = 0;
            _sharedTopKCount = 0;
            _linearTicks = _attnTicks = _normTicks = _embTicks = _lmHeadTicks = _logitsCopyTicks = 0;
            _forwardCount = 0;
            _forwardSw.Reset();
            _mtpCacheSeqLen = 0;
            if (_kvCache == null) return;
            for (int l = 0; l < NumCacheLayers; l++)
            {
                ResetCacheTensor(_kvCache[l]);
                ResetCacheTensor(_kPeCache[l]);
                if (_indexerCache[l] != null)
                    ResetCacheTensor(_indexerCache[l]);
            }
        }

        // ---- forward ---------------------------------------------------------

        protected override float[] ForwardCore(int[] tokens)
        {
            if (UsesNativeExecutor)
                return ForwardNative(tokens);
            if (IsGlm5Next)
                return ForwardCoreGlm5Next(tokens);

            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            EnsureCacheCapacity(startPos + seqLen);

            long t1 = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t1;
            Trace("inp_embd", -1, hidden);

            _sharedTopKCount = 0;
            for (int layer = 0; layer < _numTrunkLayers; layer++)
                hidden = DecoderBlock(hidden, layer, seqLen, startPos);

            Tensor normed = RMSNormOp(hidden, "output_norm.weight");
            Trace("result_norm", -1, normed);
            hidden.Dispose();

            Tensor lastHidden;
            if (seqLen > 1)
            {
                using var narrowed = normed.Narrow(0, seqLen - 1, 1);
                lastHidden = Ops.NewContiguous(narrowed);
            }
            else
            {
                lastHidden = normed.CopyRef();
            }
            normed.Dispose();

            long t2 = Stopwatch.GetTimestamp();
            Tensor logitsTensor = LinearForward(lastHidden, "output.weight")
                                  ?? LinearForward(lastHidden, "token_embd.weight");
            _lmHeadTicks += Stopwatch.GetTimestamp() - t2;
            lastHidden.Dispose();

            long t3 = Stopwatch.GetTimestamp();
            _logitsBuffer = TensorToFloatArray(logitsTensor);
            _logitsCopyTicks += Stopwatch.GetTimestamp() - t3;
            logitsTensor.Dispose();

            _cacheSeqLen += seqLen;
            _forwardCount++;
            _forwardSw.Stop();
            return _logitsBuffer;
        }

        /// <summary>
        /// Chunked prefill. The per-op attention materialises a
        /// [n_head, seqLen, kvLen] score tensor, so a long prompt is fed in
        /// bounded chunks; only the final token needs logits.
        /// </summary>
        protected override float[] ForwardRefillCore(int[] tokens)
        {
            if (tokens == null || tokens.Length <= 1)
                return ForwardCore(tokens);

            // The native executor chunks the prompt into ubatches itself.
            if (UsesNativeExecutor)
                return ForwardNative(tokens);

            int chunkSize = ResolvePrefillChunkSize();
            int lastIdx = tokens.Length - 1;
            if (tokens.Length <= chunkSize)
                return ForwardCore(tokens);

            for (int pos = 0; pos < lastIdx; pos += chunkSize)
            {
                int chunkLen = Math.Min(chunkSize, lastIdx - pos);
                var chunk = new int[chunkLen];
                Array.Copy(tokens, pos, chunk, 0, chunkLen);
                PrefillWithoutLogits(chunk);
            }
            return ForwardCore(new[] { tokens[lastIdx] });
        }

        private static int ResolvePrefillChunkSize()
        {
            string env = Environment.GetEnvironmentVariable("TS_PREFILL_CHUNK");
            if (!string.IsNullOrEmpty(env) && int.TryParse(env, out int v) && v > 0)
                return v;
            return 512;
        }

        private void PrefillWithoutLogits(int[] tokens)
        {
            if (tokens == null || tokens.Length == 0)
                return;

            if (IsGlm5Next)
            {
                // The glm5next block is hyper-connected and its attention may be
                // KDA; DecoderBlock below is GLM-5.2's plain residual block.
                _forwardSw.Start();
                SpecForwardGlm5Next(tokens, null, null, allLogitsRows: false);
                _forwardSw.Stop();
                return;
            }

            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            EnsureCacheCapacity(startPos + seqLen);

            Tensor hidden = Embedding(tokens);
            _sharedTopKCount = 0;
            for (int layer = 0; layer < _numTrunkLayers; layer++)
                hidden = DecoderBlock(hidden, layer, seqLen, startPos);
            hidden.Dispose();

            _cacheSeqLen += seqLen;
            _forwardSw.Stop();
        }

        private Tensor DecoderBlock(Tensor hidden, int layer, int seqLen, int startPos)
        {
            string[] wn = _layerNames[layer];

            Tensor normed = RMSNormOp(hidden, wn[WN_ATTN_NORM]);
            Trace("attn_norm", layer, normed);
            Tensor attnOut = Attention(normed, layer, wn, seqLen, startPos);
            normed.Dispose();
            Trace("attn_out", layer, attnOut);
            Ops.Add(hidden, hidden, attnOut);
            attnOut.Dispose();
            Trace("ffn_inp", layer, hidden);

            Tensor normed2 = RMSNormOp(hidden, wn[WN_FFN_NORM]);
            Trace("ffn_norm", layer, normed2);
            Tensor ffnOut = layer < _numDenseLead
                ? DenseFFN(normed2, wn, seqLen)
                : MoEBlock(normed2, layer, wn, seqLen);
            normed2.Dispose();
            Trace("ffn_out", layer, ffnOut);

            Ops.Add(hidden, hidden, ffnOut);
            ffnOut.Dispose();
            Trace("l_out", layer, hidden);
            return hidden;
        }

        private Tensor DenseFFN(Tensor input, string[] wn, int seqLen)
        {
            Tensor gate = LinearForward(input, wn[WN_FFN_GATE]);
            Tensor up = LinearForward(input, wn[WN_FFN_UP]);
            Ops.SiLUMul(gate, gate, up);
            up.Dispose();
            Tensor down = LinearForward(gate, wn[WN_FFN_DOWN]);
            gate.Dispose();
            return down;
        }
    }
}
