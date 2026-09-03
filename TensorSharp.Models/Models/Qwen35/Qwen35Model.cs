// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.GGML;
using TensorSharp.MLX;

namespace TensorSharp.Models
{
    /// <summary>
    /// Qwen3.5 hybrid model: alternates GatedDeltaNet (recurrent) and FullAttention layers.
    /// Every Nth layer is full attention (determined by full_attention_interval), the rest are recurrent.
    /// Full attention layers: gated Q (Q+gate interleaved), QK-norm, sigmoid-gated output.
    /// Recurrent layers: SSM conv1d, gated delta net with recurrent state.
    /// Both layer types use post_attention_norm before FFN. FFN is either a dense SwiGLU
    /// (gate+up + down) or a Mixture-of-Experts SwiGLU (router + top-K SwiGLU experts +
    /// optional shared SwiGLU expert gated by sigmoid), depending on which weights are present.
    /// </summary>
    public partial class Qwen35Model : ModelBase
    {
        private bool[] _isRecurrent;
        private int _fullAttentionInterval;

        // NextN/MTP (Qwen3.5/3.6): number of extra draft decoder blocks appended
        // after the main stack in the GGUF (`{arch}.nextn_predict_layers`).
        // Config.NumLayers is reduced to the trunk-only count at load; the MTP
        // block lives at layer index _mtpLayerIdx (== trunk count) and is only
        // executed by the speculative-decoding paths in Qwen35Model.Mtp.cs.
        private int _numNextnLayers;
        private int _mtpLayerIdx = -1;

        // Per-layer arrays must cover the MTP block (it reuses AttentionBlock
        // and the standard KV-cache machinery) while the main forward loops
        // iterate only Config.NumLayers trunk layers.
        private int TotalLayerCount => Config.NumLayers + _numNextnLayers;

        // MoE configuration (qwen35moe / qwen3next variants)
        private int _numExperts;
        private int _numExpertsUsed;
        private int _expertFfnLength;
        private int _sharedExpertFfnLength;
        // Cached ForwardRefill chunk width; see ResolvePrefillChunkSize.
        private int _prefillChunkSize;
        private bool _normTopKProb;
        private bool[] _isMoeLayer;
        private bool[] _hasSharedExperts;
        private bool[] _hasSharedExpertGate;

        // Per-layer cached expert weight key strings (avoids string interpolation in the hot loop)
        private string[][] _expertGateKeys;
        private string[][] _expertUpKeys;
        private string[][] _expertDownKeys;

        // Per-layer pre-cached QuantizedWeight references (eliminates dictionary lookups in hot loops)
        private QuantizedWeight[][] _expertGateQW;

        // Per-layer cached references for MoE router and shared experts. These avoid
        // dictionary lookups on the hot decode path that happen once per token per layer.
        private QuantizedWeight[] _ffnGateInpQW;
        private Tensor[] _ffnGateInpF32;
        private QuantizedWeight[] _ffnGateShexpQW;
        private Tensor[] _ffnGateShexpF32;
        private QuantizedWeight[] _ffnUpShexpQW;
        private Tensor[] _ffnUpShexpF32;
        private QuantizedWeight[] _ffnDownShexpQW;
        private Tensor[] _ffnDownShexpF32;
        private Tensor[] _ffnGateInpShexpVec;
        private QuantizedWeight[][] _expertUpQW;
        private QuantizedWeight[][] _expertDownQW;
        private Tensor[][] _expertGateF32;
        private Tensor[][] _expertUpF32;
        private Tensor[][] _expertDownF32;

        // Per-layer stacked-along-experts weight handles for the fused MoE
        // prefill kernel. Each entry references the original 3D GGUF tensor
        // (zero-cost view for mmap'd models). Null when stacked weights are
        // unavailable for that layer (e.g. non-mmap models with mismatched
        // layouts), in which case we fall back to the per-expert batched path.
        private StackedExpertWeights[] _layerStackedGate;
        private StackedExpertWeights[] _layerStackedUp;
        private StackedExpertWeights[] _layerStackedDown;

        // Pre-allocated MoE work tensors (reused per token)
        private Tensor _moeTokenInput;     // [1, hiddenSize]
        private Tensor _moeGateBuf;        // [1, expertFfnLength]
        private Tensor _moeUpBuf;          // [1, expertFfnLength]
        private Tensor _moeDownBuf;        // [1, hiddenSize]
        private Tensor _moeBatchedResult;  // [1, hiddenSize] - accumulated batched MoE output

        // Pre-allocated MoE routing buffers
        private float[] _moeProbs;
        private int[] _moeTopExperts;
        private float[] _moeRouteW;
        // Pre-allocated MoE pointer arrays for the batched SwiGLU GGML kernel
        private IntPtr[] _moeGatePtrs;
        private IntPtr[] _moeUpPtrs;
        private IntPtr[] _moeDownPtrs;

        // Pre-allocated FullAttention decode work tensors (decode hot path; shape is fixed by config).
        // Pre-allocating these once eliminates ~3 small tensor allocations per FullAttention layer
        // per decode token.
        private Tensor _attnDecodeQBuf;    // [1, numHeads * headDim]   (deinterleaved Q)
        private Tensor _attnDecodeGBuf;    // [1, numHeads * headDim]   (deinterleaved gate)
        private Tensor _attnDecodeOutBuf;  // [1, numHeads * headDim]   (attention output)
        // [1, qFullDim + 2*kvDim] - reused fused QKV output for FullAttention decode.
        private Tensor _attnDecodeQkvBuf;
        // [1, 2 * intermediateSize] - reused fused gate_up output for dense FFN decode.
        private Tensor _ffnDecodeGateUpBuf;
        // Minimum total sequence length at which the standalone GPU flash attention decode
        // kernel becomes worthwhile. Below this, per-call Metal command buffer setup dominates
        // the savings vs. the CPU SIMD path. Tuned empirically on Apple M-series. Once the
        // fully-fused attention layer kernel is available we prefer it instead (see
        // <see cref="TryFusedAttnLayerDecode"/>) which avoids the standalone path entirely.
        private const int FlashAttnDecodeMinSeqLen = 2048;

        // Minimum total sequence length at which the fully-fused per-layer attention decode
        // kernel beats the existing FusedRmsNormMatMulQuant + CPU-SIMD attention + FusedMatMulQuantAdd
        // path. The fused kernel folds 6 small CPU-side ops into a single Metal graph dispatch,
        // but flash_attn_ext on Metal has a fixed setup cost that only amortises when the cached
        // sequence is long enough. Below this threshold we keep the existing decode path.
        // Set FUSED_ATTN_LAYER_MIN_SEQ_LEN=N to override at runtime for benchmarking.
        private static readonly int FusedAttnLayerDecodeMinSeqLen = ResolveFusedAttnLayerMinSeqLen();
        private static readonly int MlxFlashAttnDecodeMinSeqLen = ResolveMlxFlashAttnDecodeMinSeqLen();
        private static readonly int MlxEvalEveryNLayers = ResolveMlxEvalEveryNLayers();
        private static readonly bool MlxEvalDecodeLayerBoundaries =
            string.Equals(Environment.GetEnvironmentVariable("TS_MLX_EVAL_DECODE_LAYER_BOUNDARIES"), "1", StringComparison.Ordinal);
        private static readonly bool MlxEvalFinalLogits =
            string.Equals(Environment.GetEnvironmentVariable("TS_MLX_EVAL_FINAL_LOGITS"), "1", StringComparison.Ordinal);
        // Pipelined greedy decode state. _pipelineNextInputDevice holds the
        // input embedding for the NEXT decode step, computed on-device from
        // the previous step's argmax. When set, SubmitNextGreedyDecodeStep
        // uses it as the forward's input instead of calling Embedding(host
        // int). This lets the inference loop queue step N+1's forward
        // BEFORE syncing step N's token, so the LM-head + sync wait at the
        // end of step N overlaps with step N+1's first kernels.
        private Tensor _pipelineNextInputDevice;

        // Batched-MoE decode scratch buffers. Hold one row per active
        // expert (K = numExpertsUsed) so all K experts' matmuls can be
        // dispatched in a single Metal kernel.
        // _moeBatchedGate / _moeBatchedUp: [K, intermediate]
        // _moeBatchedDown: [K, hidden]
        // _moeBatchedExpertIndices: [K] int32 (device-side topK)
        // _moeBatchedRouteWeights: [1, K] float32 (device-side routing
        //     weights ÔÇö used as the LHS of the final matmul that turns
        //     [K, hidden] expert outputs into a single [1, hidden]).
        // All allocated lazily on first use; reused across all calls.
        private Tensor _moeBatchedGate;
        private Tensor _moeBatchedUp;
        private Tensor _moeBatchedDown;
        private Tensor _moeBatchedExpertIndices;
        private Tensor _moeBatchedRouteWeights;
        // Cache keys for the per-layer stacked weight uploads. Use the
        // StackedExpertWeights.Data pointer as the unique identifier.
        //
        // Batched MoE decode: issue 1 Metal dispatch per (gate/up/down)
        // instead of K per-expert dispatches. Default on; set
        // TS_MLX_BATCHED_MOE_DECODE=0 to revert to the legacy per-expert
        // sequence. Uploads the full stacked expert weights as a SEPARATE
        // MLX array per (layer, kind), roughly doubling MLX-tracked
        // weight memory (per-expert arrays are still retained for the
        // prefill path); only set =0 on memory-constrained machines.
        private static readonly bool MlxBatchedMoeDecode =
            !string.Equals(Environment.GetEnvironmentVariable("TS_MLX_BATCHED_MOE_DECODE"), "0", StringComparison.Ordinal);

        // Fused (gate matmul + up matmul + SiLUMul) Metal kernel for the
        // batched MoE decode path. On by default; set
        // TS_MLX_MOE_FUSED_GATE_UP_SILU=0 to fall back to the legacy
        // 3-dispatch sequence for A/B comparison. Saves ~2 MLX dispatches
        // per MoE layer per decode token plus the [K, intermediate]
        // gate/up materialization round-trip.
        private static readonly bool MlxMoeFusedGateUpSiluDisabled =
            string.Equals(Environment.GetEnvironmentVariable("TS_MLX_MOE_FUSED_GATE_UP_SILU"), "0", StringComparison.Ordinal);

        // Device-router for MoE decode: compute top-K + softmax on the
        // device, skipping the per-MoE-layer host sync on routerLogits.
        // Eliminates ~60 GetFloatPtr round trips per decode token on
        // Qwen3.6-35B-A3B (60 MoE layers). On by default; set
        // TS_MLX_DEVICE_ROUTER=0 to disable. Falls back to host routing
        // automatically when prerequisites are missing (non-greedy
        // router, no batched MoE decode, or a shared-expert gate
        // requiring a host-side VecDot).
        private static readonly bool MlxDeviceRouter =
            !string.Equals(Environment.GetEnvironmentVariable("TS_MLX_DEVICE_ROUTER"), "0", StringComparison.Ordinal);

        private static int ResolveFusedAttnLayerMinSeqLen()
        {
            string env = Environment.GetEnvironmentVariable("FUSED_ATTN_LAYER_MIN_SEQ_LEN");
            if (!string.IsNullOrWhiteSpace(env) && int.TryParse(env, out int v) && v > 0)
                return v;
            // The fused per-layer attention decode kernel folds 6 small
            // CPU-side ops into a single Metal graph dispatch. omlx/vllm
            // both use the equivalent (mx.fast.scaled_dot_product_attention)
            // unconditionally ÔÇö the per-call setup cost is amortised even
            // at small KV lengths once the rest of the model is on-device.
            return 1;
        }

        private static int ResolveMlxFlashAttnDecodeMinSeqLen()
        {
            string env = Environment.GetEnvironmentVariable("TS_MLX_FLASH_ATTN_DECODE_MIN_SEQ_LEN");
            if (!string.IsNullOrWhiteSpace(env) && int.TryParse(env, out int v) && v > 0)
                return v;
            // Always prefer the Metal flash-attention kernel for MLX
            // decode ÔÇö see omlx/mlx-lm which dispatches
            // mx.fast.scaled_dot_product_attention for every step.
            return 1;
        }

        private static int ResolveMlxEvalEveryNLayers()
        {
            string env = Environment.GetEnvironmentVariable("TS_MLX_EVAL_EVERY_N_LAYERS");
            if (!string.IsNullOrWhiteSpace(env) && int.TryParse(env, out int v) && v >= 0)
                return v;
            // Measured 11.6 tok/s @ N=8 vs 11.8 tok/s @ N=16 on Qwen3.6-35B-
            // A3B IQ2_XXS decode (100-token median of 3 with 3 warmups). The
            // larger interval gives MLX more graph to schedule between syncs;
            // smaller intervals (4) regress to 10.8 tok/s as sync count grows
            // and larger (32) regress to 11.1 tok/s as the graph becomes
            // unwieldy.
            return 16;
        }

        // Pre-cached layer prefix and weight name strings (avoids string interpolation in hot loops)
        private string[] _layerPrefix;
        private string[] _attnNormKey;
        private string[] _postAttnNormKey;
        private string[] _attnQkvKey;
        private string[] _attnQKey;
        private string[] _attnKKey;
        private string[] _attnVKey;
        private string[] _attnQNormKey;
        private string[] _attnKNormKey;
        private string[] _attnOutputKey;
        private string[] _ffnGateUpKey;
        private string[] _ffnDownKey;
        private string[] _ffnGateInpKey;
        private string[] _ffnGateShexpKey;
        private string[] _ffnUpShexpKey;
        private string[] _ffnDownShexpKey;
        private string[] _ffnGateInpShexpKey;

        // (GDN-only key arrays and weight tensors live in Qwen35Model.GatedDeltaNet.cs)

        // Pre-resolved weight references for FullAttention linear projections.
        // QK-norm weights are also cached here as F32 tensors (always F32 in GGUF).
        private QuantizedWeight[] _attnQkvQW;
        private Tensor[] _attnQkvF32;
        private QuantizedWeight[] _attnQQW;
        private Tensor[] _attnQF32;
        private QuantizedWeight[] _attnKQW;
        private Tensor[] _attnKF32;
        private QuantizedWeight[] _attnVQW;
        private Tensor[] _attnVF32;
        private QuantizedWeight[] _attnOutputQW;
        private Tensor[] _attnOutputF32;
        private Tensor[] _attnQNormW;
        private Tensor[] _attnKNormW;

        // Pre-resolved attention norm weights (always F32).
        private Tensor[] _attnNormW;
        private Tensor[] _postAttnNormW;

        // Pre-resolved final norm and LM head weights (called once per forward step).
        private Tensor _finalNormW;
        private QuantizedWeight _lmHeadQW;
        private Tensor _lmHeadF32;

        // Pre-resolved weight references for non-MoE FFN paths.
        private QuantizedWeight[] _ffnGateUpQW;
        // Mixed-quant "UD"/dynamic GGUFs can store ffn_gate and ffn_up in different
        // GGML types (IQ2_XS vs IQ2_S, IQ1_S vs IQ2_XXS, ...). One fused tensor
        // cannot represent that, and both of those types need an importance matrix
        // to requantize, so ModelBase.FuseGateUpWeights leaves such a layer alone.
        // These hold the unfused pair for exactly those layers; the FFN then runs
        // two matmuls instead of one, with the weights untouched. Half the layers
        // of an unsloth Qwen3.8 UD quant land here, so this is the normal case for
        // that family, not a corner.
        private QuantizedWeight[] _ffnGateSplitQW;
        private Tensor[] _ffnGateSplitF32;
        private QuantizedWeight[] _ffnUpSplitQW;
        private Tensor[] _ffnUpSplitF32;
        private Tensor[] _ffnGateUpF32;
        private QuantizedWeight[] _ffnDownQW;
        private Tensor[] _ffnDownF32;

        // Full attention KV cache (only for attention layers)
        private Tensor[] _kvCacheK;
        private Tensor[] _kvCacheV;
        // The fused GGML verify/decode graphs update attention K/V in the
        // cacheable device mirror.  The host allocation is therefore stale until
        // it is explicitly downloaded.  Keep that fact separate from
        // _cacheSeqLen so cache growth and paged snapshots never memcpy stale
        // host bytes (most visibly when a long decode crosses 32K -> 64K).
        private bool _kvCacheHostDirty;
        private MlxFusedOps.AttentionKvCache[] _mlxAttentionCache;
        private int _kvCacheCapacity;
        // Initial KV capacity captured at InitCaches; fresh per-request fused-cache
        // holders (see Qwen35Model.PerSeqCache) allocate at this size and grow.
        private int _initialKvCacheCapacity;

        // GatedDeltaNet recurrent state, dimensions and projection weights live in
        // Qwen35Model.GatedDeltaNet.cs (also a partial of this class).
        private int _ropeDimCount;

        // MRoPE sections
        private int[] _mropeSections;

        // Pre-computed RoPE frequency table
        private float[] _ropeFreqs;

        // Cached RoPE position tensors (reused across attention layers in one forward pass)
        private Tensor _cachedRoPEPosQ, _cachedRoPEPosK;
        private int _cachedRoPEPosSeqLen, _cachedRoPEPosStartPos = -1;

        // Per-axis (T,H,W) positions for the next prefill chunk, packed flat as
        // [T0,H0,W0, T1,H1,W1, ...]. Populated by the multimodal injector just
        // before Forward() (via SetMRoPEPositions) when the request is a
        // Qwen3.5 vision prompt. Forward() consumes and clears the field at
        // the start of each call; missing/null means use plain scalar RoPE.
        // Per-pair modality assignment (which (T,H,W) axis drives which rotary
        // pair) is precomputed once in PrecomputeMRoPEInterleavedIds from
        // _mropeSections ÔÇö vLLM mrope_interleaved.py's get_mrope_interleaved_id_list.
        private int[] _pendingMRoPEPositions;
        private int[] _mropeInterleavedIds; // length rotary_dim/2; values Ôêê {0=T,1=H,2=W}

        // (GDN scratch buffers, chunked-prefill staging, and timing counters live in
        // Qwen35Model.GatedDeltaNet.cs.)

        // Vision encoder
        public Qwen35VisionEncoder VisionEncoder { get; private set; }
        private List<(Tensor embeddings, int position)> _visionEmbeddingsList = new();

        // Set QWEN35_DISABLE_FUSED_FFN=1 to disable the fully fused dense FFN graph
        // dispatch in FFNCachedFused (useful for A/B benchmarking against the legacy
        // 3-dispatch path: FusedRmsNormMatMul + SiLUMul + FusedMatMulQuantAdd).
        private static readonly bool _useFusedFfnPrefill =
            !string.Equals(Environment.GetEnvironmentVariable("QWEN35_DISABLE_FUSED_FFN"), "1", StringComparison.Ordinal);

        // Latched so a VRAM-starved prefill reports the degrade once instead of
        // once per layer per chunk.
        private bool _fusedFfnDeclineLogged;

        private void WarnFusedFfnDeclinedOnce(string reason)
        {
            if (_fusedFfnDeclineLogged) return;
            _fusedFfnDeclineLogged = true;
            Console.Error.WriteLine(
                $"[qwen35] fused dense FFN declined, falling back to the unfused chain: {reason}");
        }

        // FusedOutProjFFN threw once on the attention-layer callsite: latched so the
        // call is not retried — and re-failed — for every layer of every token. The
        // latch also serves as the one-shot warning. Per-callsite on purpose: the
        // GDN callsite in RecurrentBlock latches independently, so a failure here
        // cannot switch that path off (and vice versa).
        private bool _fusedOutProjFfnFailed;

        // The direct-CUDA fused GQA prefill attention threw: once per exception
        // TYPE (not per layer/chunk), because the legacy expanded-KV fallback it
        // degrades to cannot serve long contexts at all (see FullAttention).
        private readonly HashSet<string> _cudaGqaPrefillExWarned = new HashSet<string>(StringComparer.Ordinal);

        private long _mlxEvalBoundaryTicks;
        private long _mlxCacheEvalTicks;

        /// <param name="draftModelPath">Optional DFlash / DFlash2 drafter GGUF
        /// (general.architecture = "dflash"). When present it replaces the trunk's
        /// own NextN/MTP block as the drafter.</param>
        public Qwen35Model(string ggufPath, BackendType backend, int tpDegree = 1, ITensorParallelGroup tpGroup = null,
            string draftModelPath = null)
            : base(ggufPath, backend, tpDegree, tpGroup)
        {
            _useMetalGdnInplaceState = ShouldUseMetalGdnInplaceState(
                backend,
                IsTensorParallel,
                Environment.GetEnvironmentVariable("TS_QWEN35_METAL_GDN_INPLACE_STATE"));

            string arch = _gguf.GetString("general.architecture") ?? "qwen35";
            Config = new ModelConfig { Architecture = arch };
            ParseBaseConfig();

            // Qwen3.5 uses per-layer KV head counts; get the first non-zero value
            if (_gguf.Metadata.TryGetValue($"{arch}.attention.head_count_kv", out var hckvVal))
                Config.NumKVHeads = Convert.ToInt32(hckvVal);
            else
                Config.NumKVHeads = Config.NumHeads;

            // SSM config (GDN-specific dimensions are parsed inside the GDN partial)
            ParseGdnConfig(arch);
            _fullAttentionInterval = (int)_gguf.GetUint32($"{arch}.full_attention_interval", 4);

            // NextN/MTP draft blocks: the GGUF block_count includes them, but they
            // are not part of the main decoder stack. Split them off so the main
            // forward loops never execute the MTP block (llama.cpp does the same:
            // hparams.n_layer() = n_layer_all - n_layer_nextn).
            _numNextnLayers = (int)_gguf.GetUint32($"{arch}.nextn_predict_layers", 0);
            if (_numNextnLayers > 0)
            {
                Config.NumLayers -= _numNextnLayers;
                _mtpLayerIdx = Config.NumLayers;
            }

            // MRoPE sections
            var sections = _gguf.GetInt32Array($"{arch}.rope.dimension_sections");
            _mropeSections = sections;
            _ropeDimCount = (int)_gguf.GetUint32($"{arch}.rope.dimension_count", 0);

            // MoE configuration: present for qwen35moe / qwen3next variants
            _numExperts = (int)_gguf.GetUint32($"{arch}.expert_count", 0);
            _numExpertsUsed = (int)_gguf.GetUint32($"{arch}.expert_used_count", 0);
            _expertFfnLength = (int)_gguf.GetUint32($"{arch}.expert_feed_forward_length", 0);
            _sharedExpertFfnLength = (int)_gguf.GetUint32($"{arch}.expert_shared_feed_forward_length", (uint)_expertFfnLength);
            // This MoE family renormalizes the selected top-K probabilities by default.
            _normTopKProb = true;
            Config.NumExperts = _numExperts;
            Config.NumExpertsUsed = _numExpertsUsed;

            // Determine which layers are recurrent (GatedDeltaNet) vs full-attention.
            // Prefer an explicit GGUF `layer_types` array if present (vLLM reads
            // `config.layer_types[i]` Ôêê {"linear_attention","full_attention"} at
            // qwen3_5.py:230; future fine-tunes can ship a non-default pattern).
            // Fall back to the period-`full_attention_interval` pattern that stock
            // 27B/35B-A3B GGUFs use (no layer_types array shipped).
            // Sized to TotalLayerCount: the MTP block (if any) is always a
            // full-attention layer, which the default `false` already encodes.
            _isRecurrent = new bool[TotalLayerCount];
            var layerTypes = _gguf.GetStringArray($"{arch}.layer_types");
            if (layerTypes != null && layerTypes.Length == Config.NumLayers)
            {
                for (int i = 0; i < Config.NumLayers; i++)
                    _isRecurrent[i] = string.Equals(layerTypes[i], "linear_attention", StringComparison.OrdinalIgnoreCase);
            }
            else
            {
                for (int i = 0; i < Config.NumLayers; i++)
                    _isRecurrent[i] = (i + 1) % _fullAttentionInterval != 0;
            }

            ParseTokenizer();

            Console.WriteLine($"Model: {arch}, Layers={Config.NumLayers}, Hidden={Config.HiddenSize}, " +
                $"Heads={Config.NumHeads}, KVHeads={Config.NumKVHeads}, HeadDim={Config.HeadDim}, Vocab={Config.VocabSize}");
            Console.WriteLine($"RoPE base={Config.RopeBase}, scale={Config.RopeScale}, eps={Config.Eps}");
            Console.WriteLine($"SSM: dInner={_ssmDInner}, dState={_ssmDState}, nGroup={_ssmNGroup}, " +
                $"dtRank={_ssmDtRank}, convKernel={_convKernel}, fullAttnInterval={_fullAttentionInterval}");
            Console.WriteLine($"MRoPE sections: [{string.Join(", ", _mropeSections ?? Array.Empty<int>())}]");

            int attnCount = 0, recCount = 0;
            for (int i = 0; i < Config.NumLayers; i++)
            {
                if (_isRecurrent[i]) recCount++; else attnCount++;
            }
            Console.WriteLine($"Layer types: {attnCount} full attention, {recCount} recurrent (GatedDeltaNet)");

            if (_numNextnLayers > 0)
                Console.WriteLine($"NextN/MTP: {_numNextnLayers} draft block(s) at layer {_mtpLayerIdx} (excluded from main stack)");

            if (_numExperts > 0)
            {
                Console.WriteLine($"MoE: experts={_numExperts}, used={_numExpertsUsed}, " +
                    $"expertFFN={_expertFfnLength}, sharedFFN={_sharedExpertFfnLength}");
                // The host-MoE seam hangs off the GGML MoE FFN kernel; the
                // pure-C# CUDA path serves experts from its own device-resident
                // stacked buffer and has no offload, so the flag would silently
                // do nothing there.
                if (!IsGgmlBackend)
                    MoeCpuOffloadConfig.WarnUnsupportedBackend("qwen35moe", _backend.ToString());
            }

            LoadWeights();
            FuseAttentionProjectionWeights();
            FuseRecurrentInputWeights();
            FuseGateUpWeights(TotalLayerCount);
            DetectMoeLayers();
            BuildLayerKeys();
            BuildProjScaleTable();
            InitMoeBuffers();

            if (IsTensorParallel)
            {
                ValidateTpConstraints();
                ShardQwen35WeightsForTP();
                PrepareCudaQuantizedWeightsForInferenceTP();
            }
            else
            {
                PrepareCudaQuantizedWeightsForInference();
            }

            int maxContextLength = ResolveConfiguredContextLength();
            int initialCacheLength = ResolveInitialCacheAllocationLength(maxContextLength);
            if (initialCacheLength < maxContextLength)
                Console.WriteLine($"Initial {_backend} KV cache allocation: {initialCacheLength} tokens (grows on demand up to {maxContextLength}).");

            if (IsTensorParallel)
                InitTpCaches(initialCacheLength, maxContextLength);
            else
                InitCaches(initialCacheLength, maxContextLength);

            PrecomputeRoPE();
            InitGDNBuffers();
            CacheRecurrentWeights();
            CacheMtpWeights();

            // --draft-model / TS_QWEN35_DFLASH. Last, because the drafter's tensors
            // are merged into the trunk's weight dictionaries and its KV ring comes
            // off the same allocator.
            TryLoadQwen35DFlash(draftModelPath);
        }

        private unsafe void FuseAttentionProjectionWeights()
        {
            int fused = 0;
            int keptSeparate = 0;
            for (int layer = 0; layer < TotalLayerCount; layer++)
            {
                if (_isRecurrent[layer])
                    continue;

                string prefix = $"blk.{layer}.";
                string[] sources =
                {
                    prefix + "attn_q.weight",
                    prefix + "attn_k.weight",
                    prefix + "attn_v.weight",
                };
                if (TryFuseWeights(prefix + "attn_qkv.weight", keepSources: false, sources))
                {
                    fused++;
                }
                else
                {
                    // Importance-matrix UD quants commonly choose different
                    // types for Q, K and V. Repacking those mixed sources as F32
                    // expanded Qwen3.6-27B's 409 MiB of source weights to 4.65 GiB
                    // and streamed that expansion on every layer pass. llama.cpp
                    // keeps the original tensors and issues three quantized
                    // matmuls; our managed and native full-model paths already
                    // support the same SeparateQkv contract, so retain them too.
                    keptSeparate++;
                }
            }

            if (fused > 0)
                Console.WriteLine($"  Fused projections: {fused} Q+K+V");
            if (keptSeparate > 0)
                Console.WriteLine($"  Separate projections: {keptSeparate} mixed-quant Q/K/V sets preserved");
        }

        private unsafe void FuseRecurrentInputWeights()
        {
            int fused = 0;
            int fusedF32 = 0;
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                if (!_isRecurrent[layer])
                    continue;

                string prefix = $"blk.{layer}.";
                string[] sources =
                {
                    prefix + "attn_qkv.weight",
                    prefix + "attn_gate.weight",
                    prefix + "ssm_beta.weight",
                    prefix + "ssm_alpha.weight",
                };
                if (TryFuseWeights(prefix + "ssm_in_proj.weight", keepSources: true, sources))
                {
                    fused++;
                }
                else if (IsTensorParallel &&
                         TryFuseWeightsToFloat32(prefix + "ssm_in_proj.weight", sources))
                {
                    // The same-type quant fusion can't run when the sources have
                    // mismatched ggml types — e.g. an aggressively quantized model
                    // (IQ2_XXS/…) whose ssm_beta/ssm_alpha stay F32 while attn_qkv
                    // is 2-bit. The fused ssm_in_proj is only consumed by the TP
                    // recurrent path (non-TP uses the separate source weights), so
                    // build an F32 pack there instead of leaving it missing (which
                    // crashed TP with "ssm_in_proj not found in sharded weights").
                    fusedF32++;
                }
            }

            if (fused > 0)
                Console.WriteLine($"  Fused projections: {fused} recurrent input packs");
            if (fusedF32 > 0)
                Console.WriteLine($"  Fused projections: {fusedF32} recurrent input packs (dequantized to F32 for TP; mixed source quant types)");
        }

        /// <summary>
        /// Build a fused projection as one F32 tensor by dequantizing (quantized
        /// sources) or copying (F32 sources) each in order along the output
        /// dimension. Fallback for <see cref="TryFuseWeights"/> when the sources
        /// have mismatched ggml types and cannot be packed in-place. Sources are
        /// left in place. Returns false if any source is absent or the input
        /// dimensions disagree. The result is F32, so this trades memory for
        /// compatibility — used only where the fused pack is required (TP).
        /// </summary>
        private unsafe bool TryFuseWeightsToFloat32(string fusedName, params string[] weightNames)
        {
            if (weightNames == null || weightNames.Length < 2)
                return false;

            int inDim = -1;
            long totalRows = 0;
            var quant = new QuantizedWeight[weightNames.Length];
            var f32 = new Tensor[weightNames.Length];
            for (int i = 0; i < weightNames.Length; i++)
            {
                if (_quantWeights.TryGetValue(weightNames[i], out quant[i]))
                {
                    int d = (int)quant[i].Ne0;
                    if (inDim < 0) inDim = d; else if (inDim != d) return false;
                    totalRows += quant[i].Ne1;
                }
                else if (_weights.TryGetValue(weightNames[i], out f32[i]))
                {
                    int d = (int)f32[i].Sizes[1];
                    if (inDim < 0) inDim = d; else if (inDim != d) return false;
                    totalRows += f32[i].Sizes[0];
                }
                else
                {
                    return false; // missing source
                }
            }

            var fused = new Tensor(_allocator, DType.Float32, totalRows, inDim);
            float* dstBase = GetFloatPtr(fused);
            long rowOffset = 0;
            for (int i = 0; i < weightNames.Length; i++)
            {
                if (quant[i] != null)
                {
                    var qw = quant[i];
                    long rowBytes = NativeDequant.RowSize(qw.GgmlType, qw.Ne0);
                    byte* srcBase = (byte*)qw.Data.ToPointer();
                    for (long row = 0; row < qw.Ne1; row++)
                        NativeDequant.DequantizeToFloat32Native(
                            qw.GgmlType,
                            (IntPtr)(srcBase + row * rowBytes),
                            (IntPtr)(dstBase + (rowOffset + row) * inDim),
                            inDim);
                    if (qw.Scale != 1.0f)
                    {
                        // Bake the sidecar per-tensor scale into the F32 rows so the
                        // fused pack needs no scalar at inference.
                        long scaledCount = qw.Ne1 * inDim;
                        float ws = qw.Scale;
                        float* wsPtr = dstBase + rowOffset * inDim;
                        for (long e = 0; e < scaledCount; e++) wsPtr[e] *= ws;
                    }
                    rowOffset += qw.Ne1;
                }
                else
                {
                    var w = f32[i];
                    long rows = w.Sizes[0];
                    float* srcPtr = GetFloatPtr(w);
                    long bytes = rows * inDim * sizeof(float);
                    Buffer.MemoryCopy(srcPtr, dstBase + rowOffset * inDim, bytes, bytes);
                    rowOffset += rows;
                }
            }

            InvalidateTensorDeviceCache(fused);
            _weights[fusedName] = fused;
            return true;
        }

        private unsafe bool TryFuseWeights(string fusedName, bool keepSources, params string[] weightNames)
        {
            if (weightNames == null || weightNames.Length < 2)
                return false;

            var quantWeights = new QuantizedWeight[weightNames.Length];
            bool allQuant = true;
            for (int i = 0; i < weightNames.Length; i++)
            {
                if (!_quantWeights.TryGetValue(weightNames[i], out quantWeights[i]))
                {
                    allQuant = false;
                    break;
                }
            }

            if (allQuant)
            {
                var first = quantWeights[0];
                long totalBytes = 0;
                long totalNe1 = 0;
                for (int i = 0; i < quantWeights.Length; i++)
                {
                    var qw = quantWeights[i];
                    if (qw.GgmlType != first.GgmlType || qw.Ne0 != first.Ne0 || qw.Scale != first.Scale)
                        return false;

                    totalBytes += qw.RawBytes;
                    totalNe1 += qw.Ne1;
                }

                if (!TryCreateFusedQuantizedWeight(out QuantizedWeight fusedWeight, quantWeights))
                    return false;

                fusedWeight.Scale = quantWeights[0].Scale;
                _quantWeights[fusedName] = fusedWeight;
                if (!keepSources)
                {
                    for (int i = 0; i < weightNames.Length; i++)
                    {
                        var name = weightNames[i];
                        var qw = quantWeights[i];
                        _quantWeights.Remove(name);
                        qw.Dispose();
                    }
                }
                return true;
            }

            var floatWeights = new Tensor[weightNames.Length];
            for (int i = 0; i < weightNames.Length; i++)
            {
                if (!_weights.TryGetValue(weightNames[i], out floatWeights[i]))
                    return false;
            }

            int inDim = (int)floatWeights[0].Sizes[1];
            int totalOutDim = 0;
            for (int i = 0; i < floatWeights.Length; i++)
            {
                var w = floatWeights[i];
                if ((int)w.Sizes[1] != inDim)
                    return false;

                totalOutDim += (int)w.Sizes[0];
            }

            var fusedTensor = new Tensor(_allocator, DType.Float32, totalOutDim, inDim);
            int outOffset = 0;
            for (int i = 0; i < floatWeights.Length; i++)
            {
                var w = floatWeights[i];
                int outDim = (int)w.Sizes[0];
                using var slice = fusedTensor.Narrow(0, outOffset, outDim);
                Ops.Copy(slice, w);
                outOffset += outDim;
            }

            _weights[fusedName] = fusedTensor;
            for (int i = 0; i < weightNames.Length; i++)
            {
                var name = weightNames[i];
                var w = floatWeights[i];
                _weights.Remove(name);
                w.Dispose();
            }

            return true;
        }

        private void DetectMoeLayers()
        {
            int numLayers = TotalLayerCount;
            _isMoeLayer = new bool[numLayers];
            _hasSharedExperts = new bool[numLayers];
            _hasSharedExpertGate = new bool[numLayers];

            int moeCount = 0, sharedCount = 0, sharedGateCount = 0;
            for (int l = 0; l < numLayers; l++)
            {
                string prefix = $"blk.{l}.";
                _isMoeLayer[l] = WeightExists(prefix + "ffn_gate_inp.weight");
                if (_isMoeLayer[l])
                {
                    moeCount++;
                    _hasSharedExperts[l] = WeightExists(prefix + "ffn_up_shexp.weight") &&
                                           WeightExists(prefix + "ffn_down_shexp.weight");
                    if (_hasSharedExperts[l])
                        sharedCount++;
                    _hasSharedExpertGate[l] = _weights.ContainsKey(prefix + "ffn_gate_inp_shexp.weight");
                    if (_hasSharedExpertGate[l])
                        sharedGateCount++;
                }
            }

            if (moeCount > 0)
                Console.WriteLine($"  MoE layers: {moeCount}/{numLayers} (shared experts: {sharedCount}, gated shared: {sharedGateCount})");
        }

        private bool WeightExists(string name) =>
            _quantWeights.ContainsKey(name) || _weights.ContainsKey(name);

        private void InitMoeBuffers()
        {
            if (_numExperts <= 0)
                return;

            _moeProbs = new float[_numExperts];
            _moeTopExperts = new int[_numExpertsUsed];
            _moeRouteW = new float[_numExpertsUsed];

            int numLayers = TotalLayerCount;
            _expertGateKeys = new string[numLayers][];
            _expertUpKeys = new string[numLayers][];
            _expertDownKeys = new string[numLayers][];
            _expertGateQW = new QuantizedWeight[numLayers][];
            _expertUpQW = new QuantizedWeight[numLayers][];
            _expertDownQW = new QuantizedWeight[numLayers][];
            _expertGateF32 = new Tensor[numLayers][];
            _expertUpF32 = new Tensor[numLayers][];
            _expertDownF32 = new Tensor[numLayers][];

            _ffnGateInpQW = new QuantizedWeight[numLayers];
            _ffnGateInpF32 = new Tensor[numLayers];
            _ffnGateShexpQW = new QuantizedWeight[numLayers];
            _ffnGateShexpF32 = new Tensor[numLayers];
            _ffnUpShexpQW = new QuantizedWeight[numLayers];
            _ffnUpShexpF32 = new Tensor[numLayers];
            _ffnDownShexpQW = new QuantizedWeight[numLayers];
            _ffnDownShexpF32 = new Tensor[numLayers];
            _ffnGateInpShexpVec = new Tensor[numLayers];
            _layerStackedGate = new StackedExpertWeights[numLayers];
            _layerStackedUp = new StackedExpertWeights[numLayers];
            _layerStackedDown = new StackedExpertWeights[numLayers];

            for (int l = 0; l < numLayers; l++)
            {
                if (!_isMoeLayer[l])
                    continue;
                _expertGateKeys[l] = new string[_numExperts];
                _expertUpKeys[l] = new string[_numExperts];
                _expertDownKeys[l] = new string[_numExperts];
                _expertGateQW[l] = new QuantizedWeight[_numExperts];
                _expertUpQW[l] = new QuantizedWeight[_numExperts];
                _expertDownQW[l] = new QuantizedWeight[_numExperts];
                _expertGateF32[l] = new Tensor[_numExperts];
                _expertUpF32[l] = new Tensor[_numExperts];
                _expertDownF32[l] = new Tensor[_numExperts];
                string p = _layerPrefix[l];
                for (int e = 0; e < _numExperts; e++)
                {
                    _expertGateKeys[l][e] = p + "ffn_gate_exps." + e + ".weight";
                    _expertUpKeys[l][e] = p + "ffn_up_exps." + e + ".weight";
                    _expertDownKeys[l][e] = p + "ffn_down_exps." + e + ".weight";

                    _quantWeights.TryGetValue(_expertGateKeys[l][e], out _expertGateQW[l][e]);
                    _quantWeights.TryGetValue(_expertUpKeys[l][e], out _expertUpQW[l][e]);
                    _quantWeights.TryGetValue(_expertDownKeys[l][e], out _expertDownQW[l][e]);
                    _weights.TryGetValue(_expertGateKeys[l][e], out _expertGateF32[l][e]);
                    _weights.TryGetValue(_expertUpKeys[l][e], out _expertUpF32[l][e]);
                    _weights.TryGetValue(_expertDownKeys[l][e], out _expertDownF32[l][e]);
                }

                // Look up the original 3D stacked-along-experts views populated
                // by ModelBase.LoadWeights. Used by the batched MoE decode path
                // (MlxBatchedMoeDecode) to upload all experts as one MLX array
                // per (layer, kind) and dispatch a single matmul instead of K
                // per-expert matmuls.
                _stackedExpertWeights.TryGetValue(p + "ffn_gate_exps.weight", out _layerStackedGate[l]);
                _stackedExpertWeights.TryGetValue(p + "ffn_up_exps.weight", out _layerStackedUp[l]);
                _stackedExpertWeights.TryGetValue(p + "ffn_down_exps.weight", out _layerStackedDown[l]);

                // Cache router and shared expert weights for this layer.
                _quantWeights.TryGetValue(_ffnGateInpKey[l], out _ffnGateInpQW[l]);
                _weights.TryGetValue(_ffnGateInpKey[l], out _ffnGateInpF32[l]);

                if (_hasSharedExperts != null && _hasSharedExperts[l])
                {
                    _quantWeights.TryGetValue(_ffnGateShexpKey[l], out _ffnGateShexpQW[l]);
                    _weights.TryGetValue(_ffnGateShexpKey[l], out _ffnGateShexpF32[l]);
                    _quantWeights.TryGetValue(_ffnUpShexpKey[l], out _ffnUpShexpQW[l]);
                    _weights.TryGetValue(_ffnUpShexpKey[l], out _ffnUpShexpF32[l]);
                    _quantWeights.TryGetValue(_ffnDownShexpKey[l], out _ffnDownShexpQW[l]);
                    _weights.TryGetValue(_ffnDownShexpKey[l], out _ffnDownShexpF32[l]);

                    if (_hasSharedExpertGate != null && _hasSharedExpertGate[l])
                        _weights.TryGetValue(_ffnGateInpShexpKey[l], out _ffnGateInpShexpVec[l]);
                }
            }

            // Pre-allocate scratch tensors reused across MoE expert calls. This avoids the
            // hot-loop allocation of [1, expertFFN] / [1, hidden] tensors per token per expert.
            if (_expertFfnLength > 0)
            {
                _moeTokenInput = new Tensor(_allocator, DType.Float32, 1, Config.HiddenSize);
                _moeGateBuf = new Tensor(_allocator, DType.Float32, 1, _expertFfnLength);
                _moeUpBuf = new Tensor(_allocator, DType.Float32, 1, _expertFfnLength);
                _moeDownBuf = new Tensor(_allocator, DType.Float32, 1, Config.HiddenSize);
                _moeBatchedResult = new Tensor(_allocator, DType.Float32, 1, Config.HiddenSize);
                _moeGatePtrs = new IntPtr[_numExpertsUsed];
                _moeUpPtrs = new IntPtr[_numExpertsUsed];
                _moeDownPtrs = new IntPtr[_numExpertsUsed];
            }

            // Pre-allocate decode-only FullAttention work tensors. Used only when seqLen == 1.
            int qHeadDim = Config.NumHeads * Config.HeadDim;
            if (qHeadDim > 0)
            {
                _attnDecodeQBuf = new Tensor(_allocator, DType.Float32, 1, qHeadDim);
                _attnDecodeGBuf = new Tensor(_allocator, DType.Float32, 1, qHeadDim);
                _attnDecodeOutBuf = new Tensor(_allocator, DType.Float32, 1, qHeadDim);

                // Pre-allocate fused QKV (Q+gate interleaved + K + V) and FFN gate_up buffers.
                // Eliminates 2 tensor allocations per attention layer per decode token.
                int qFullDim = Config.NumHeads * Config.HeadDim * 2;
                int kvDim = Config.NumKVHeads * Config.HeadDim;
                if (qFullDim > 0 && kvDim > 0)
                    _attnDecodeQkvBuf = new Tensor(_allocator, DType.Float32, 1, qFullDim + 2 * kvDim);
                if (Config.IntermediateSize > 0)
                    _ffnDecodeGateUpBuf = new Tensor(_allocator, DType.Float32, 1, 2 * Config.IntermediateSize);
            }
        }

        /// <summary>
        /// Pre-cache layer-prefix and weight name strings for every layer so the hot Forward()
        /// path never executes string interpolation. This eliminates dozens of allocations per
        /// layer per forward step and removes hundreds of dictionary string-hash lookups.
        /// </summary>
        private void BuildLayerKeys()
        {
            int n = TotalLayerCount;
            _layerPrefix = new string[n];
            _attnNormKey = new string[n];
            _postAttnNormKey = new string[n];
            _attnQkvKey = new string[n];
            _attnQKey = new string[n];
            _attnKKey = new string[n];
            _attnVKey = new string[n];
            _attnQNormKey = new string[n];
            _attnKNormKey = new string[n];
            _attnOutputKey = new string[n];
            _ffnGateUpKey = new string[n];
            _ffnDownKey = new string[n];
            _ffnGateInpKey = new string[n];
            _ffnGateShexpKey = new string[n];
            _ffnUpShexpKey = new string[n];
            _ffnDownShexpKey = new string[n];
            _ffnGateInpShexpKey = new string[n];
            InitGdnLayerKeyArrays(n);

            for (int l = 0; l < n; l++)
            {
                string p = $"blk.{l}.";
                _layerPrefix[l] = p;
                _attnNormKey[l] = p + "attn_norm.weight";
                _postAttnNormKey[l] = p + "post_attention_norm.weight";
                _attnQkvKey[l] = p + "attn_qkv.weight";
                _attnQKey[l] = p + "attn_q.weight";
                _attnKKey[l] = p + "attn_k.weight";
                _attnVKey[l] = p + "attn_v.weight";
                _attnQNormKey[l] = p + "attn_q_norm.weight";
                _attnKNormKey[l] = p + "attn_k_norm.weight";
                _attnOutputKey[l] = p + "attn_output.weight";
                _ffnGateUpKey[l] = p + "ffn_gate_up.weight";
                _ffnDownKey[l] = p + "ffn_down.weight";
                _ffnGateInpKey[l] = p + "ffn_gate_inp.weight";
                _ffnGateShexpKey[l] = p + "ffn_gate_shexp.weight";
                _ffnUpShexpKey[l] = p + "ffn_up_shexp.weight";
                _ffnDownShexpKey[l] = p + "ffn_down_shexp.weight";
                _ffnGateInpShexpKey[l] = p + "ffn_gate_inp_shexp.weight";
                SetGdnLayerKeys(l, p);
            }
        }

        /// <summary>
        /// Pre-resolve recurrent layer constant tensors and pre-compute a transposed conv1d
        /// weight layout that is friendly for SIMD vectorization across channels.
        /// </summary>
        private unsafe void CacheRecurrentWeights()
        {
            int n = TotalLayerCount;
            InitGdnWeightArrays(n);
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;

            _attnQkvQW = new QuantizedWeight[n];
            _attnQkvF32 = new Tensor[n];
            _attnQQW = new QuantizedWeight[n];
            _attnQF32 = new Tensor[n];
            _attnKQW = new QuantizedWeight[n];
            _attnKF32 = new Tensor[n];
            _attnVQW = new QuantizedWeight[n];
            _attnVF32 = new Tensor[n];
            _attnOutputQW = new QuantizedWeight[n];
            _attnOutputF32 = new Tensor[n];
            _attnQNormW = new Tensor[n];
            _attnKNormW = new Tensor[n];
            _attnNormW = new Tensor[n];
            _postAttnNormW = new Tensor[n];

            _ffnGateUpQW = new QuantizedWeight[n];
            _ffnGateUpF32 = new Tensor[n];
            _ffnGateSplitQW = new QuantizedWeight[n];
            _ffnGateSplitF32 = new Tensor[n];
            _ffnUpSplitQW = new QuantizedWeight[n];
            _ffnUpSplitF32 = new Tensor[n];
            _ffnDownQW = new QuantizedWeight[n];
            _ffnDownF32 = new Tensor[n];

            // Final layer norm + LM head (vocab projection). Tied embedding weights fall
            // back to token_embd.weight when output.weight is not present.
            _weights.TryGetValue("output_norm.weight", out _finalNormW);
            if (!_quantWeights.TryGetValue("output.weight", out _lmHeadQW))
                _quantWeights.TryGetValue("token_embd.weight", out _lmHeadQW);
            if (!_weights.TryGetValue("output.weight", out _lmHeadF32))
                _weights.TryGetValue("token_embd.weight", out _lmHeadF32);

            for (int l = 0; l < n; l++)
            {
                // Norms / FFN are present for every layer (recurrent and attention).
                _weights.TryGetValue(_attnNormKey[l], out _attnNormW[l]);
                _weights.TryGetValue(_postAttnNormKey[l], out _postAttnNormW[l]);
                _quantWeights.TryGetValue(_ffnGateUpKey[l], out _ffnGateUpQW[l]);
                _weights.TryGetValue(_ffnGateUpKey[l], out _ffnGateUpF32[l]);
                if (_ffnGateUpQW[l] == null && _ffnGateUpF32[l] == null)
                {
                    // Unfused mixed-quant layer: keep the pair as it was quantized.
                    string p = $"blk.{l}.";
                    _quantWeights.TryGetValue(p + "ffn_gate.weight", out _ffnGateSplitQW[l]);
                    _weights.TryGetValue(p + "ffn_gate.weight", out _ffnGateSplitF32[l]);
                    _quantWeights.TryGetValue(p + "ffn_up.weight", out _ffnUpSplitQW[l]);
                    _weights.TryGetValue(p + "ffn_up.weight", out _ffnUpSplitF32[l]);
                }
                _quantWeights.TryGetValue(_ffnDownKey[l], out _ffnDownQW[l]);
                _weights.TryGetValue(_ffnDownKey[l], out _ffnDownF32[l]);

                if (_isRecurrent[l])
                {
                    CacheRecurrentLayerWeights(l, qkvDim);
                }
                else
                {
                    _quantWeights.TryGetValue(_attnQkvKey[l], out _attnQkvQW[l]);
                    _weights.TryGetValue(_attnQkvKey[l], out _attnQkvF32[l]);
                    _quantWeights.TryGetValue(_attnQKey[l], out _attnQQW[l]);
                    _weights.TryGetValue(_attnQKey[l], out _attnQF32[l]);
                    _quantWeights.TryGetValue(_attnKKey[l], out _attnKQW[l]);
                    _weights.TryGetValue(_attnKKey[l], out _attnKF32[l]);
                    _quantWeights.TryGetValue(_attnVKey[l], out _attnVQW[l]);
                    _weights.TryGetValue(_attnVKey[l], out _attnVF32[l]);
                    _quantWeights.TryGetValue(_attnOutputKey[l], out _attnOutputQW[l]);
                    _weights.TryGetValue(_attnOutputKey[l], out _attnOutputF32[l]);
                    _weights.TryGetValue(_attnQNormKey[l], out _attnQNormW[l]);
                    _weights.TryGetValue(_attnKNormKey[l], out _attnKNormW[l]);
                }
            }
        }

        private void InitCaches(int initialSeqLen, int maxSeqLen)
        {
            _maxContextLength = maxSeqLen;
            _kvCacheCapacity = initialSeqLen;
            _initialKvCacheCapacity = initialSeqLen;
            int numLayers = TotalLayerCount;
            _kvCacheK = new Tensor[numLayers];
            _kvCacheV = new Tensor[numLayers];
            _mlxAttentionCache = _backend == BackendType.Mlx
                ? new MlxFusedOps.AttentionKvCache[numLayers]
                : null;

            InitGdnCacheArrays(numLayers);
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;

            ApplyModelAlignedKvCacheDefault(_quantWeights);
            DType kvDtype = _kvCacheDtype.ToDType();
            for (int l = 0; l < numLayers; l++)
            {
                if (!_isRecurrent[l])
                {
                    _kvCacheK[l] = new Tensor(_allocator, kvDtype, Config.NumKVHeads, initialSeqLen, Config.HeadDim);
                    _kvCacheV[l] = new Tensor(_allocator, kvDtype, Config.NumKVHeads, initialSeqLen, Config.HeadDim);
                    InitializeCacheTensor(_kvCacheK[l]);
                    InitializeCacheTensor(_kvCacheV[l]);
                    if (_mlxAttentionCache != null)
                        _mlxAttentionCache[l] = new MlxFusedOps.AttentionKvCache();
                }
                else
                {
                    InitGdnLayerCache(l, qkvDim);
                }
            }
            _cacheSeqLen = 0;
        }

        /// <summary>Reserve the complete request context once, before its first
        /// prefill. This avoids a 65,536-to-131,072 power-of-two expansion in
        /// the middle of a long decode when only a few thousand additional
        /// slots are required.</summary>
        public override void PrepareForPrefill(int requiredContextTokens)
        {
            if (requiredContextTokens <= 0)
                return;
            // Tensor-parallel runs keep no host-side attention cache — each rank
            // owns a device-resident shard that the TP forward path grows itself —
            // so there is nothing to pre-reserve here.
            if (_kvCacheK == null || _kvCacheV == null)
                return;
            // The request declares prompt + its whole generation budget, which on a
            // server started with a large --max-tokens is the entire context window:
            // 260,864 tokens here is 16 GiB of K/V on top of the weights. Reserve
            // only what the device can actually hold — EnsureCacheCapacity still
            // grows the cache on demand if the generation really runs that long.
            requiredContextTokens = ResolvePrefillReservationLength(
                requiredContextTokens, _kvCacheCapacity, granularity: CacheCapacityAlignment);
            EnsureCacheCapacity(requiredContextTokens, geometricGrowth: false);
        }

        /// <summary>Native Qwen decode windows are 256-token aligned, so every
        /// non-geometric cache reservation snaps to that boundary.</summary>
        private const int CacheCapacityAlignment = 256;

        private void EnsureCacheCapacity(int requiredSeqLen, bool geometricGrowth = true)
        {
            // Growth discards and reallocates the active caches; if this holder
            // occupies an arena batched-decode slot, its newest rows/state exist
            // only there — flush them to host first or growth strands them.
            FlushArenaSlotForActiveHolder();
            if (requiredSeqLen <= _kvCacheCapacity)
                return;
            if (requiredSeqLen > _maxContextLength)
                throw new InvalidOperationException($"Requested sequence length {requiredSeqLen} exceeds configured max context {_maxContextLength}.");

            // Ops.Copy below is a host-side copy.  Whole-model GGML prefill and
            // decode leave attention K/V and recurrent state device-resident, so
            // drain both before replacing the buffers / invalidating their graphs.
            EnsureKvCacheHostSynchronized();
            EnsureFusedDecodeStateHostSynchronized();
            // Native graphs retain bindings to the current cache/state buffers.
            // Drop them while those pointers are still alive, before any tensor is
            // replaced or its device-copy cache entry is evicted.
            InvalidateFullDecodeState(hardBindings: true);
            InvalidateVerifyCache();

            int newCapacity;
            if (geometricGrowth)
            {
                newCapacity = Math.Max(_kvCacheCapacity, 1);
                while (newCapacity < requiredSeqLen)
                    newCapacity = Math.Min(_maxContextLength, newCapacity * 2);
            }
            else
            {
                // Native Qwen decode windows are 256-token aligned. Reserve
                // exactly the request budget rounded to that boundary rather
                // than doubling a multi-gigabyte cache.
                const int alignment = CacheCapacityAlignment;
                long rounded = ((long)requiredSeqLen + alignment - 1) / alignment * alignment;
                newCapacity = (int)Math.Min(_maxContextLength, rounded);
            }

            DType kvDtype = _kvCacheDtype.ToDType();
            for (int l = 0; l < TotalLayerCount; l++)
            {
                if (_isRecurrent[l])
                    continue;

                var newK = new Tensor(_allocator, kvDtype, Config.NumKVHeads, newCapacity, Config.HeadDim);
                var newV = new Tensor(_allocator, kvDtype, Config.NumKVHeads, newCapacity, Config.HeadDim);
                InitializeCacheTensor(newK);
                InitializeCacheTensor(newV);

                if (_cacheSeqLen > 0)
                {
                    using var srcK = _kvCacheK[l].Narrow(1, 0, _cacheSeqLen);
                    using var dstK = newK.Narrow(1, 0, _cacheSeqLen);
                    Ops.Copy(dstK, srcK);

                    using var srcV = _kvCacheV[l].Narrow(1, 0, _cacheSeqLen);
                    using var dstV = newV.Narrow(1, 0, _cacheSeqLen);
                    Ops.Copy(dstV, srcV);
                }

                // Device-copy cache entries are keyed by the old host pointers.
                // Evict them while those pointers are still valid; otherwise a
                // grow leaks the old slabs and a captured graph can retain them.
                InvalidateTensorDeviceCache(_kvCacheK[l]);
                InvalidateTensorDeviceCache(_kvCacheV[l]);
                _kvCacheK[l].Dispose();
                _kvCacheV[l].Dispose();
                _kvCacheK[l] = newK;
                _kvCacheV[l] = newV;
            }

            _kvCacheCapacity = newCapacity;
            _kvCacheHostDirty = false;
            Console.WriteLine($"Expanded Qwen3.5 attention cache to {newCapacity} tokens.");
        }

        private void PrecomputeRoPE()
        {
            int headDim = Config.HeadDim;
            int ropeDim = _ropeDimCount > 0 ? _ropeDimCount : headDim;
            int halfDim = ropeDim / 2;
            float freqScale = 1.0f / Config.RopeScale;
            _ropeFreqs = new float[halfDim];
            for (int i = 0; i < halfDim; i++)
                _ropeFreqs[i] = freqScale / MathF.Pow(Config.RopeBase, (2.0f * i) / ropeDim);
        }

        /// <summary>
        /// On <c>ggml_cuda</c> the MoE decode/prefill read the experts exclusively
        /// through the per-layer STACKED expert device buffer (one device copy per
        /// <c>*_exps</c> tensor, cached on first use). Preloading each per-expert
        /// split view as well would put a SECOND full copy of every expert in VRAM ÔÇö
        /// for the 35B-A3B that is an extra ~10 GB, which overflows the 16 GB card
        /// into shared GPU memory (WDDM paging) and tanks decode speed. Skip the
        /// per-expert members; their host views stay mapped so the rare per-op
        /// sequential fallback can still stream the bytes on demand.
        /// </summary>
        protected override bool ShouldPreloadCudaQuantWeightToDevice(string weightName)
            => !_stackedExpertMemberNames.Contains(weightName)
               && base.ShouldPreloadCudaQuantWeightToDevice(weightName);

        /// <summary>
        /// Under tensor parallelism a recurrent layer's GatedDeltaNet input is read
        /// exclusively through the sharded <c>ssm_in_proj.weight</c> pack. Its four
        /// sources (<c>attn_qkv</c>, <c>attn_gate</c>, <c>ssm_beta</c>,
        /// <c>ssm_alpha</c>) are deliberately kept alive by the fusion
        /// (<c>keepSources: true</c>, and the F32 fallback leaves them in place), so
        /// they were still in <c>_quantWeights</c> at preload time and rank 0 ended up
        /// holding a complete unsharded copy of the whole GDN trunk on top of its
        /// shards — 4 tensors x 48 recurrent layers on Qwen3.8-27B.
        ///
        /// Skipping the PRELOAD only drops a device-side cache entry: the host mapping
        /// is untouched, so any path that still reads a source streams it on demand
        /// exactly as an MoE-offloaded expert does. Numerics are unaffected.
        /// </summary>
        private HashSet<string> _tpSupersededSources;

        protected override bool IsSupersededByTpShard(string weightName)
        {
            if (!IsTensorParallel || _isRecurrent == null || _ssmInProjKey == null)
                return false;

            if (_tpSupersededSources == null)
            {
                var set = new HashSet<string>(StringComparer.Ordinal);
                for (int l = 0; l < TotalLayerCount; l++)
                {
                    if (!_isRecurrent[l] || _ssmInProjKey[l] == null)
                        continue;
                    if (!_tpQuantWeights.ContainsKey(_ssmInProjKey[l])
                        && !_tpWeights.ContainsKey(_ssmInProjKey[l]))
                        continue;
                    set.Add(_attnQkvRecKey[l]);
                    set.Add(_attnGateRecKey[l]);
                    set.Add(_ssmBetaKey[l]);
                    set.Add(_ssmAlphaKey[l]);
                }
                _tpSupersededSources = set;
                if (set.Count > 0)
                    Console.WriteLine($"  TP: {set.Count} GDN source weight(s) superseded by the sharded " +
                        "ssm_in_proj pack; not preloaded to rank 0.");
            }
            return _tpSupersededSources.Contains(weightName);
        }

        protected override void ResetKVCacheCore()
        {
            if (IsTensorParallel)
            {
                ResetTpKVCache();
                _cacheSeqLen = 0;
                _linearTicks = _attnTicks = _normTicks = _embTicks = _lmHeadTicks = _logitsCopyTicks = 0;
                _forwardCount = 0;
                _forwardSw.Reset();
                return;
            }

            // Reset intentionally discards the current device-resident state. Clear
            // the dirty latches first so dropping the graphs does not copy it back,
            // and release all native bindings before touching their host tensors.
            _kvCacheHostDirty = false;
            _gdnStateHostDirty = false;
            InvalidateFullDecodeState();
            InvalidateVerifyCache();

            for (int l = 0; l < TotalLayerCount; l++)
            {
                if (!_isRecurrent[l])
                {
                    ResetCacheTensor(_kvCacheK[l]);
                    ResetCacheTensor(_kvCacheV[l]);
                    _mlxAttentionCache?[l]?.Reset();
                }
                else
                {
                    ResetGdnLayerCache(l);
                }
            }
            _cacheSeqLen = 0;
            _fdSpecSessionActive = false;
            _linearTicks = _attnTicks = _normTicks = _embTicks = _lmHeadTicks = _logitsCopyTicks = 0;
            _mlxEvalBoundaryTicks = 0;
            _mlxCacheEvalTicks = 0;
            _forwardCount = 0;
            ResetGdnTimingCounters();
            _forwardSw.Reset();

        }

        public override bool SupportsKVCacheTruncation => false;

        /// <summary>Qwen 3.5/3.8 implement the split FFN: BuildLayerKeys populates
        /// _ffnGateSplitQW/_ffnUpSplitQW when the fused tensor is absent,
        /// FFNCachedSplitGateUp runs the managed path, and all three fused native
        /// graphs branch on <c>gu_w == nullptr</c> to emit two matmuls. Every
        /// unsloth "UD" quant of this family needs it - about half their layers
        /// store ffn_gate and ffn_up in different IQ types.</summary>
        protected override bool SupportsSplitGateUpFfn => true;

        /// <summary>
        /// The fused per-rank TP graphs carry the per-projection scale table
        /// (see <c>TryBuildTpFdLayerDescs</c>), and the native graphs apply each
        /// scale on the correct side of the AllReduce cut. The per-op TP
        /// fallback linears do NOT, so a scaled GGUF that cannot take the fused
        /// path still has to be refused - <c>ValidateTpConstraints</c> does that.
        /// </summary>
        protected override bool SupportsTensorParallelWeightScales => true;

        // Per-block snapshot for Qwen 3.5 (mix of attention layers and GDN
        // recurrent layers). Each block bundles:
        //   * For every attention layer L: K bytes for [start,start+B), V bytes
        //     for [start,start+B). Uses the same byte layout as the simpler
        //     models so a single helper can capture it.
        //   * For every GDN layer L: a snapshot of the layer's running state at
        //     the END of this block (convState ring buffer + writeIdx +
        //     deltaState tensor bytes). The state is "as of after the block's
        //     final token" because Capture runs after each prefill chunk lands.
        // Recurrent state isn't decomposable into per-position slices, so this
        // model relies on the per-chunk capture path
        // (<see cref="RequiresPerBlockCapture"/>).
        public override bool RequiresPerBlockCapture => true;

        public override bool SupportsKVStateSnapshot => _kvCacheK != null && _kvCacheV != null;

        // A byte snapshot is retained as a diagnostic/legacy capability, but it
        // is not a viable cross-request cache for this hybrid architecture: each
        // 256-token block repeats the complete GDN recurrent state (about 50 MiB
        // for the 9B model), and interleaving those snapshots has historically
        // corrupted sequence isolation. Continuous batching uses the model's
        // device-resident per-request KV+GDN holders instead; fallback paths
        // re-prefill cleanly.
        public override bool SupportsCrossSequenceKvReuse => false;

        public override string KVStateFingerprint =>
            $"qwen35|arch={Config.Architecture}|L={Config.NumLayers}|H={Config.NumHeads}|KV={Config.NumKVHeads}|D={Config.HeadDim}|gdnK={_headKDim}|gdnV={_headVDim}|nKHead={_numKHeads}|nVHead={_numVHeads}|convKern={_convKernel}|dtype={_kvCacheDtype.ToShortString()}";

        public override long ComputeKVBlockByteSize(int tokenCount)
        {
            if (tokenCount <= 0 || _kvCacheK == null || _isRecurrent == null) return 0;
            long total = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!_isRecurrent[l])
                {
                    if (_kvCacheK[l] == null || _kvCacheV[l] == null) return 0;
                    total += AttentionLayerBlockBytes(_kvCacheK[l], tokenCount);
                    total += AttentionLayerBlockBytes(_kvCacheV[l], tokenCount);
                }
                else
                {
                    total += GdnLayerStateBytes(l);
                }
            }
            return total;
        }

        public override bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        {
            if (!SupportsKVStateSnapshot) return false;
            long expected = ComputeKVBlockByteSize(tokenCount);
            if (destination.Length != expected) return false;

            // CopyAttentionOut / CopyGdnStateOut read raw host pointers.  A fused
            // GGML call may have advanced only their device mirrors.
            EnsureKvCacheHostSynchronized();
            EnsureFusedDecodeStateHostSynchronized();

            int offset = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!_isRecurrent[l])
                {
                    if (startToken + tokenCount > _cacheSeqLen) return false;
                    if (!CopyAttentionOut(_kvCacheK[l], startToken, tokenCount, destination[offset..], out int wK))
                        return false;
                    offset += wK;
                    if (!CopyAttentionOut(_kvCacheV[l], startToken, tokenCount, destination[offset..], out int wV))
                        return false;
                    offset += wV;
                }
                else
                {
                    if (!CopyGdnStateOut(l, destination[offset..], out int wG))
                        return false;
                    offset += wG;
                }
            }
            return offset == destination.Length;
        }

        public override bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source)
        {
            if (!SupportsKVStateSnapshot) return false;
            if (destToken != _cacheSeqLen) return false;
            long expected = ComputeKVBlockByteSize(tokenCount);
            if (source.Length != expected) return false;

            EnsureCacheCapacity(destToken + tokenCount);
            EnsureKvCacheHostSynchronized();
            EnsureFusedDecodeStateHostSynchronized();
            int offset = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!_isRecurrent[l])
                {
                    if (!CopyAttentionIn(_kvCacheK[l], destToken, tokenCount, source[offset..], out int rK))
                        return false;
                    offset += rK;
                    if (!CopyAttentionIn(_kvCacheV[l], destToken, tokenCount, source[offset..], out int rV))
                        return false;
                    offset += rV;
                }
                else
                {
                    if (!CopyGdnStateIn(l, source[offset..], out int rG))
                        return false;
                    offset += rG;
                }
            }
            _cacheSeqLen = destToken + tokenCount;

            // Invalidate any device-cached views so the next forward refills them
            // from the freshly-written host buffers. Drop native graphs first:
            // their bindings still point at those device-cache entries.
            InvalidateFullDecodeState(hardBindings: true);
            InvalidateVerifyCache();
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_isRecurrent[l]) continue;
                InvalidateTensorDeviceCache(_kvCacheK[l]);
                InvalidateTensorDeviceCache(_kvCacheV[l]);
            }
            _kvCacheHostDirty = false;
            _gdnStateHostDirty = false;
            return true;
        }

        /// <summary>
        /// Download attention caches written through GGML's cacheable device-copy
        /// path before any code reads or copies their host allocations.
        /// </summary>
        private void EnsureKvCacheHostSynchronized()
        {
            if (!IsGgmlBackend || _kvCacheK == null)
                return;
            // If the active holder occupies an arena batched-decode slot, its
            // newest KV rows/state exist only there: flush-and-retire the slot
            // first so the resident-copy download below sees current bytes.
            FlushArenaSlotForActiveHolder();
            if (!_kvCacheHostDirty)
                return;

            var seen = new HashSet<Storage>();
            for (int l = 0; l < TotalLayerCount; l++)
            {
                if (_kvCacheK[l] != null && seen.Add(_kvCacheK[l].Storage))
                    SyncTensorHostCache(_kvCacheK[l]);
                if (_kvCacheV[l] != null && seen.Add(_kvCacheV[l].Storage))
                    SyncTensorHostCache(_kvCacheV[l]);
            }

            _kvCacheHostDirty = false;
        }

        private static long AttentionLayerBlockBytes(Tensor cacheTensor, int tokenCount)
        {
            long numKVHeads = cacheTensor.Sizes[0];
            long capacity = cacheTensor.Sizes[1];
            long rowBytes = cacheTensor.Storage.ByteLength / (numKVHeads * capacity);
            return numKVHeads * tokenCount * rowBytes;
        }

        private long GdnLayerStateBytes(int layer)
        {
            // convState bytes + writeIdx (4 bytes) + deltaState tensor bytes.
            long convBytes = (long)_convState[layer].Length * sizeof(float);
            long deltaBytes = GdnDeltaStateBytes(_deltaStateTensor[layer]);
            return convBytes + sizeof(int) + deltaBytes;
        }

        private static bool CopyAttentionOut(Tensor cacheTensor, int startToken, int tokenCount, Span<byte> destination, out int written)
        {
            cacheTensor.Storage.EnsureHostReadable();
            long numKVHeads = cacheTensor.Sizes[0];
            long capacity = cacheTensor.Sizes[1];
            long rowBytes = cacheTensor.Storage.ByteLength / (numKVHeads * capacity);
            long blockBytes = numKVHeads * tokenCount * rowBytes;
            if (destination.Length < blockBytes) { written = 0; return false; }
            IntPtr basePtr = cacheTensor.Storage.PtrAtElement(0);
            unsafe
            {
                byte* src = (byte*)basePtr;
                fixed (byte* dst = destination)
                {
                    long perHead = tokenCount * rowBytes;
                    for (long h = 0; h < numKVHeads; h++)
                    {
                        long s = (h * capacity + startToken) * rowBytes;
                        long d = h * perHead;
                        Buffer.MemoryCopy(src + s, dst + d, destination.Length - d, perHead);
                    }
                }
            }
            written = (int)blockBytes;
            return true;
        }

        private static bool CopyAttentionIn(Tensor cacheTensor, int destToken, int tokenCount, ReadOnlySpan<byte> source, out int read)
        {
            cacheTensor.Storage.EnsureHostReadable();
            long numKVHeads = cacheTensor.Sizes[0];
            long capacity = cacheTensor.Sizes[1];
            if (destToken + tokenCount > capacity) { read = 0; return false; }
            long rowBytes = cacheTensor.Storage.ByteLength / (numKVHeads * capacity);
            long blockBytes = numKVHeads * tokenCount * rowBytes;
            if (source.Length < blockBytes) { read = 0; return false; }
            IntPtr basePtr = cacheTensor.Storage.PtrAtElement(0);
            unsafe
            {
                byte* dst = (byte*)basePtr;
                fixed (byte* srcBase = source)
                {
                    long perHead = tokenCount * rowBytes;
                    for (long h = 0; h < numKVHeads; h++)
                    {
                        long d = (h * capacity + destToken) * rowBytes;
                        long s = h * perHead;
                        Buffer.MemoryCopy(srcBase + s, dst + d, cacheTensor.Storage.ByteLength - d, perHead);
                    }
                }
            }
            read = (int)blockBytes;
            return true;
        }

        private bool CopyGdnStateOut(int layer, Span<byte> destination, out int written)
        {
            // Reads the host mirrors; see RecurrentBlock.
            DrainDeviceRecurrentState();

            written = 0;
            SyncCudaGdnConvStateToHost(layer);
            float[] conv = _convState[layer];
            int convBytes = conv.Length * sizeof(float);
            Tensor delta = _deltaStateTensor[layer];
            long deltaBytes = GdnDeltaStateBytes(delta);
            long total = (long)convBytes + sizeof(int) + deltaBytes;
            if (destination.Length < total) return false;

            // convState as raw float bytes.
            MemoryMarshal.AsBytes(conv.AsSpan()).CopyTo(destination[..convBytes]);
            // writeIdx as int.
            BitConverter.TryWriteBytes(destination.Slice(convBytes, sizeof(int)), _convStateWriteIdx[layer]);
            // deltaState bytes via storage pointer (host-resident on every backend).
            delta.Storage.EnsureHostReadable();
            IntPtr deltaBase = GdnDeltaStatePointer(delta);
            unsafe
            {
                fixed (byte* dst = destination[(convBytes + sizeof(int))..])
                {
                    Buffer.MemoryCopy((void*)deltaBase, dst, deltaBytes, deltaBytes);
                }
            }
            written = (int)total;
            return true;
        }

        private bool CopyGdnStateIn(int layer, ReadOnlySpan<byte> source, out int read)
        {
            // Writes the host mirrors, so whatever the device holds is superseded.
            DrainDeviceRecurrentState();

            read = 0;
            float[] conv = _convState[layer];
            int convBytes = conv.Length * sizeof(float);
            Tensor delta = _deltaStateTensor[layer];
            long deltaBytes = GdnDeltaStateBytes(delta);
            long total = (long)convBytes + sizeof(int) + deltaBytes;
            if (source.Length < total) return false;

            source[..convBytes].CopyTo(MemoryMarshal.AsBytes(conv.AsSpan()));
            _convStateWriteIdx[layer] = BitConverter.ToInt32(source.Slice(convBytes, sizeof(int)));
            SyncCudaGdnConvStateFromHost(layer);

            delta.Storage.EnsureHostReadable();
            IntPtr deltaBase = GdnDeltaStatePointer(delta);
            unsafe
            {
                fixed (byte* src = source[(convBytes + sizeof(int))..])
                {
                    Buffer.MemoryCopy(src, (void*)deltaBase, deltaBytes, deltaBytes);
                }
            }
            // MLX cache reflects the host-side bytes lazily on next use; resetting
            // its scratch indices is enough for correctness.
            _mlxGdnCache?[layer]?.Reset();
            read = (int)total;
            return true;
        }

        // Chunk size for ForwardRefill: long prompts are processed in this-many-token
        // chunks so the per-layer attention-score allocation stays bounded. Because a
        // chunk runs as ONE fused-verify graph, the chunk size is also that graph's
        // width, which bounds the MoE activation peak. Override with TS_PREFILL_CHUNK.
        /// <inheritdoc/>
        /// <remarks>
        /// Only the full-attention layers hold a KV cache — the GatedDeltaNet
        /// layers carry a fixed-size recurrent state instead, which does not grow
        /// with the context and so plays no part in this per-token figure.
        /// </remarks>
        protected override long KvCacheBytesPerToken
        {
            get
            {
                if (_isRecurrent == null || Config == null)
                    return 0;
                int attnLayers = 0;
                for (int l = 0; l < TotalLayerCount; l++)
                    if (!_isRecurrent[l]) attnLayers++;
                if (attnLayers <= 0)
                    return 0;
                long elems = 2L * attnLayers * Config.NumKVHeads * Config.HeadDim;   // K + V
                return _kvCacheDtype.ToDType().ByteLengthFor(elems);
            }
        }

        /// <inheritdoc/>
        protected override int GgmlPrefillChunkWarmupLength =>
            IsGgmlBackend ? ResolvePrefillChunkSize() : 0;

        private int ResolvePrefillChunkSize()
        {
            // Resolved once and cached: the chunk width IS the fused-verify graph's
            // shape, and a width that drifts with whatever happens to be free right
            // now would make every prompt a fresh shape (persist-cache churn on
            // CUDA, pipeline rebuilds on Vulkan) instead of reusing the one the
            // startup warmup already built.
            if (_prefillChunkSize > 0)
                return _prefillChunkSize;
            _prefillChunkSize = ComputePrefillChunkSize();
            return _prefillChunkSize;
        }

        private int ComputePrefillChunkSize()
        {
            string env = Environment.GetEnvironmentVariable("TS_PREFILL_CHUNK");
            if (!string.IsNullOrEmpty(env) && int.TryParse(env, out int v) && v > 0)
                return v;
            // Vulkan VRAM cliff: a fused-verify graph wider than ~768 tokens pushes the
            // MoE activation peak past a 16 GB card's VRAM, and WDDM then PERMANENTLY
            // demotes the reuse-gallocr to shared system memory -> pp2048 collapses from
            // ~360 to ~18 tok/s (measured). 768 keeps the peak device-resident and holds
            // across context lengths (the KV cache grows separately). A FIXED 768 also
            // keeps every full chunk on ONE verify graph shape, so the ggml-vulkan
            // pipelines the startup warmup compiles are REUSED across all prompts/scenarios
            // (a variable/larger chunk makes each prompt length a fresh shape whose pipeline
            // + graph is built cold inside the measured TTFT — measured to REGRESS
            // multi-prompt Vulkan runs). CUDA (cuBLAS/MMQ, no WDDM demotion) keeps 2048.
            if (_backend == BackendType.GgmlVulkan)
                return 768;

            // CUDA used to keep a flat 2048 on the reasoning that only Vulkan
            // suffers the WDDM demotion described above. It does not: the same
            // card demotes a CUDA allocation just as happily once the card is
            // full, and on a model whose weights nearly fill VRAM the 2048-wide
            // graph's ~500 MB of scratch is what fills it. Cap the width against
            // the VRAM actually left after the weights are resident.
            //
            // This is not purely a memory trade. Measured on a 16 GB RTX 3080
            // Laptop, Qwen3.8-27B-UD-IQ4_XS, 5000-token chunked prefill:
            //   2048 -> 521 tok/s, 505 MB of graph scratch
            //   1024 -> 533 tok/s, 253 MB
            //    512 -> 500 tok/s, 128 MB
            // so the narrower graph is also the faster one until 512, where the
            // per-chunk fixed cost finally shows. 1024 is therefore the floor
            // rather than the smallest width that fits: it is both faster than the
            // 2048 it replaces and less than half the memory, so there is nothing
            // to buy by going narrower. A card with room is unaffected — it still
            // gets the full 2048.
            const int desired = 2048;
            if (!GpuMemoryBudget.TryGetSpareBytes(_backend, out long spare))
                return desired;
            // Peak live activation of one fused-verify chunk, per token: the
            // gate+up pair dominates (2 x intermediate), plus the hidden-sized
            // residual/attention intermediates the allocator keeps alive
            // alongside it. Deliberately a slight over-estimate — erring wide
            // costs a little prefill throughput, erring narrow costs eviction.
            long bytesPerToken = 4L * (2L * Config.IntermediateSize + 8L * Config.HiddenSize);
            return GpuMemoryBudget.FitTokens(spare, bytesPerToken, desired, minTokens: 1024, granularity: 512);
        }

        // Gates the whole-model fused prefill path (TSGgml_Qwen35ModelVerify, one
        // GGML graph for all layers). Default on; TS_QWEN35_PREFILL_VERIFY=0 forces
        // the per-op layer loop for A/B comparison.
        private static readonly bool _prefillVerifyEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_QWEN35_PREFILL_VERIFY"), "0", StringComparison.Ordinal);

        /// <summary>
        /// Whether a dense (text-only) prefill chunk can run through the fused
        /// whole-model verify kernel. The kernel computes sequential RoPE positions
        /// (start_pos..+N) and a pure causal mask, so it is correct only when there
        /// are no multimodal spans / custom MRoPE positions. Capacity (start_pos +
        /// seqLen &lt;= cacheSize) and weight/shape support are re-checked inside
        /// <see cref="TryFullModelVerify"/>, which bails (ÔåÆ per-op fallback) otherwise.
        /// </summary>
        private bool CanUsePrefillVerify(int startPos, int seqLen)
        {
            if (!_prefillVerifyEnabled) return false;
            if (_fvUnsupported) return false;
            // All GGML GPU backends run the whole-model prefill graph. CUDA and
            // Vulkan use dynamic set_rows writes; Metal uses contiguous cpy views
            // whose offset is baked into this one-shot prefill graph. The latter
            // mirrors llama.cpp's linear KV-store path without relying on Metal's
            // problematic multi-dimensional set_rows shape.
            if (_backend != BackendType.GgmlCuda && _backend != BackendType.GgmlVulkan
                    && _backend != BackendType.GgmlMetal)
                return FvBail($"backend {_backend} has no whole-model prefill graph");
            if (seqLen <= 1) return false;   // decode; TryFullModelDecode's job
            // Vision embeddings must already be spliced into `hidden` before the
            // graph runs; ForwardCore injects them, so a non-empty queue here means
            // this call is upstream of that.
            if (_visionEmbeddingsList.Count > 0)
                return FvBail("vision embeddings are still queued for injection");
            // NOTE: pending MRoPE positions used to bail here too, which sent every
            // image-bearing prompt down the op-by-op layer loop for the whole
            // prompt. The native graph implements MRoPE end to end (ggml_rope_multi
            // with GGML_ROPE_TYPE_IMROPE for both Q and K) and TryFullModelVerify
            // already packs `_pendingMRoPEPositions`/`_mropeSections` into its
            // mropePos/mropeSecs arguments, so the guard was blocking a finished
            // path. TS_QWEN35_MROPE_VERIFY=0 restores the old behaviour.
            if (_pendingMRoPEPositions != null)
            {
                if (!_mropeVerifyEnabled)
                    return FvBail("multimodal prompt (disabled via TS_QWEN35_MROPE_VERIFY=0)");
                if (_mropeSections == null || _mropeSections.Length < 4)
                    return FvBail("multimodal prompt without MRoPE section metadata");
            }
            if (_headKDim != _headVDim)
                return FvBail($"asymmetric GDN state widths (headKDim={_headKDim}, headVDim={_headVDim})");
            return true;
        }

        /// <summary>Set TS_QWEN35_MROPE_VERIFY=0 to keep image prompts on the
        /// op-by-op prefill loop (A/B against the fused graph).</summary>
        private static readonly bool _mropeVerifyEnabled =
            Environment.GetEnvironmentVariable("TS_QWEN35_MROPE_VERIFY") != "0";

        /// <summary>
        /// Report, once, why the whole-model fused PREFILL graph declined. Its
        /// absence costs an order of magnitude (the op-by-op fallback re-allocates,
        /// uploads and synchronises per operation, ~15 ops x 65 layers per chunk),
        /// and every one of these returns used to be silent — the decode path got
        /// this diagnostic; prefill never did.
        /// </summary>
        private bool _fvDeclineLogged;

        private bool FvBail(string reason)
        {
            if (!_fvDeclineLogged)
            {
                _fvDeclineLogged = true;
                Console.Error.WriteLine(
                    $"[qwen35] fused whole-model prefill NOT engaged ({reason}); " +
                    "falling back to the per-op prefill loop.");
            }
            return false;
        }

        // Cached CUDA graphs of the per-op prefill layer loop (direct cuda backend
        // only). One instantiated graph per (seqLen, startPos, KV-buffer identity,
        // conv-ring phase); replay collapses the loop's ~900 kernel dispatches into
        // a single cuGraphLaunch. See CudaPrefillGraphCache for the capture rules.
        private CudaPrefillGraphCache _cudaPrefillGraphs;
        private CudaDecodeDynParams _cudaDecodeDynParams;

        private Tensor RunPerOpLayerLoop(Tensor hidden, int seqLen, int startPos)
        {
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                if (_isRecurrent[layer])
                    hidden = RecurrentBlock(hidden, layer, seqLen, startPos);
                else
                    hidden = AttentionBlock(hidden, layer, seqLen, startPos);
                TryEvaluateMlxLayerBoundary(hidden, layer, seqLen);
                TraceLayer(hidden, layer, "");
            }
            _layerTraceForwards++;
            return hidden;
        }

        /// <summary>
        /// Runs the per-op transformer layer loop, using a cached CUDA graph when
        /// one exists for this exact shape/position/cache identity. Ownership
        /// contract matches the plain loop: consumes <paramref name="hidden"/> and
        /// returns the tensor holding the final hidden states (caller disposes).
        /// Text-only multi-token prefill on the direct cuda backend only; every
        /// other case runs the loop directly.
        /// </summary>
        // Prefill graph capture's value is amortizing per-op kernel-launch
        // overhead, which matters when per-kernel compute is small (short
        // chunks); its cost is the whole per-op loop's peak activation set,
        // stolen from the pool for the entry's lifetime. Above this length the
        // trade flips AND the cost compounds badly on a large hybrid model: a
        // server/benchmark that visits many different (seqLen, startPos) shapes
        // (real chunked prefill always does -- MaxPrefillChunkSize splits long
        // prompts, and every KV-cache resize changes the key's kvPtr) keeps
        // capturing and evicting large entries, each capture itself costing 2x
        // (one plain warm-up run, one captured run) plus teardown. Measured on
        // Qwen3.6-27B-UD-IQ2_XXS (RTX 3080 16GB): a 5-length x N-iter prefill
        // sweep through the engine/scheduler path took >20 minutes and pushed
        // the process to ~15.3GB dedicated + ~3.1GB shared (WDDM spillover,
        // PCIe-speed) VRAM with capture on; the identical sweep with capture
        // off completed in under a minute at 200-320 tok/s and no VRAM
        // spillover. TS_CUDA_PREFILL_GRAPH_MAX_SEQLEN overrides; 0 = unlimited
        // (restores the old always-try-to-capture behavior).
        private static readonly int CudaPrefillGraphMaxSeqLen = ReadCudaPrefillGraphMaxSeqLen();

        private static int ReadCudaPrefillGraphMaxSeqLen()
        {
            string s = Environment.GetEnvironmentVariable("TS_CUDA_PREFILL_GRAPH_MAX_SEQLEN");
            if (!string.IsNullOrEmpty(s) && int.TryParse(s, out int v) && v >= 0)
                return v;
            return 512;
        }

        private Tensor RunCudaPrefillLayerLoop(Tensor hidden, int seqLen, int startPos)
        {
            bool graphable = _backend == BackendType.Cuda
                && _pendingMRoPEPositions == null && _visionEmbeddingsList.Count == 0
                && _kvCacheK != null && _isRecurrent != null;
            if (graphable && seqLen == 1 && CudaPrefillGraphCache.DecodeEnabled)
                return RunCudaDecodeLayerLoop(hidden, startPos);
            if (CudaPrefillGraphMaxSeqLen > 0 && seqLen > CudaPrefillGraphMaxSeqLen)
                graphable = false;
            if (!graphable || seqLen <= 1 || !CudaPrefillGraphCache.Enabled)
            {
                return RunPerOpLayerLoop(hidden, seqLen, startPos);
            }

            _cudaPrefillGraphs ??= new CudaPrefillGraphCache(_allocator);
            CudaPrefillGraphCache graphs = _cudaPrefillGraphs;
            if (!graphs.IsUsable)
                return RunPerOpLayerLoop(hidden, seqLen, startPos);

            // Everything a captured kernel bakes in must be part of the key:
            // shape, positions (RoPE + attention windows), the KV/state buffer
            // identity (a grown cache means new pointers) and the GDN conv ring
            // phase (a kernel parameter).
            int firstAttn = -1, firstRec = -1;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_isRecurrent[l]) { if (firstRec < 0) firstRec = l; }
                else if (firstAttn < 0) firstAttn = l;
            }
            long kvPtr = firstAttn >= 0 && _kvCacheK[firstAttn] != null
                ? CudaFusedOps.GetDevicePointer(_kvCacheK[firstAttn]).ToInt64() : 0;
            long statePtr = firstRec >= 0 && _deltaStateTensor?[firstRec] != null
                ? CudaFusedOps.GetDevicePointer(_deltaStateTensor[firstRec]).ToInt64() : 0;
            int convPhase = firstRec >= 0 && _convStateWriteIdx != null ? _convStateWriteIdx[firstRec] : 0;
            string key = $"{seqLen}|{startPos}|{kvPtr:x}|{statePtr:x}|{convPhase}";

            if (graphs.TryGetReplayInput(key, out Tensor pinned))
            {
                Ops.Copy(pinned, hidden);
                hidden.Dispose();
                graphs.Replay(key);
                MarkCudaGraphStateModified();
                return pinned;
            }

            if (!graphs.ShouldCapture(key))
                return RunPerOpLayerLoop(hidden, seqLen, startPos);

            // Pre-warm the RoPE position tensors so their host->device upload
            // happens before capture (a captured upload would bake the host
            // pointer into the graph; these tensors are pinned by the entry).
            Tensor posQ = EnsureRoPEPositions(startPos, seqLen, Config.NumHeads);
            Tensor posK = EnsureRoPEPositions(startPos, seqLen, Config.NumKVHeads);
            CudaFusedOps.TryEnsureDeviceResident(posQ);
            CudaFusedOps.TryEnsureDeviceResident(posK);

            // The captured loop must run on a stable input buffer the graph owns.
            Tensor pinnedIn = new Tensor(_allocator, DType.Float32, hidden.Sizes);
            Ops.Copy(pinnedIn, hidden);
            hidden.Dispose();

            if (!graphs.BeginCapture(key))
                return RunPerOpLayerLoop(pinnedIn, seqLen, startPos);

            bool captured = false;
            try
            {
                Tensor result = RunPerOpLayerLoop(pinnedIn, seqLen, startPos);
                if (ReferenceEquals(result, pinnedIn))
                    captured = graphs.EndCaptureAndLaunch(key, pinnedIn, new[] { posQ, posK });
                else
                    graphs.AbortCapture(key);
            }
            catch (CudaGraphCaptureAbortedException ex)
            {
                graphs.AbortCapture(key, ex.Message);
            }
            catch (TensorSharp.Cuda.Interop.CudaException ex)
            {
                graphs.AbortCapture(key, ex.ToString());
            }
            catch
            {
                graphs.AbortCapture(key);
                throw;
            }

            if (!captured)
            {
                // Nothing executed during the failed capture; run the loop for real.
                return RunPerOpLayerLoop(pinnedIn, seqLen, startPos);
            }
            return pinnedIn;
        }

        /// <summary>
        /// CUDA-graph capture/replay of the per-op DECODE step (seqLen == 1).
        /// Unlike prefill, the position-dependent values (attention length, KV
        /// write position, GDN conv ring index, RoPE position) change every
        /// token, so they are NOT part of the key: the captured kernels re-read
        /// them from a small device block whose refresh (pinned host -> device
        /// memcpy) is the graph's leading node. A replay is then: copy the
        /// embedding row into the pinned input, rewrite 4 pinned ints, one
        /// cuGraphLaunch. The key carries only buffer identities: a grown KV
        /// cache (new pointers, new baked grids) captures a fresh graph.
        /// </summary>
        private Tensor RunCudaDecodeLayerLoop(Tensor hidden, int startPos)
        {
            _cudaPrefillGraphs ??= new CudaPrefillGraphCache(_allocator);
            CudaPrefillGraphCache graphs = _cudaPrefillGraphs;
            if (!graphs.IsUsable)
                return RunPerOpLayerLoop(hidden, 1, startPos);

            _cudaDecodeDynParams ??= new CudaDecodeDynParams(_allocator);
            CudaDecodeDynParams dyn = _cudaDecodeDynParams;
            if (!dyn.IsValid)
                return RunPerOpLayerLoop(hidden, 1, startPos);

            int firstAttn = -1, firstRec = -1;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_isRecurrent[l]) { if (firstRec < 0) firstRec = l; }
                else if (firstAttn < 0) firstAttn = l;
            }
            long kvPtr = firstAttn >= 0 && _kvCacheK[firstAttn] != null
                ? CudaFusedOps.GetDevicePointer(_kvCacheK[firstAttn]).ToInt64() : 0;
            long statePtr = firstRec >= 0 && _deltaStateTensor?[firstRec] != null
                ? CudaFusedOps.GetDevicePointer(_deltaStateTensor[firstRec]).ToInt64() : 0;
            long convPtr = firstRec >= 0 && _cudaGdnConvStateTensor?[firstRec] != null
                ? CudaFusedOps.GetDevicePointer(_cudaGdnConvStateTensor[firstRec]).ToInt64() : 0;
            string key = $"dec|{kvPtr:x}|{statePtr:x}|{convPtr:x}|{_kvCacheCapacity}";

            int attendLen = startPos + 1;
            int convDim = _convKernel - 1;
            // All recurrent layers advance their ring index together (single
            // stream), so the first one represents them all; the kernels read
            // the live value from the dyn block on replay.
            int convPhase = firstRec >= 0 && _convStateWriteIdx != null ? _convStateWriteIdx[firstRec] : 0;

            if (graphs.TryGetReplayInput(key, attendLen, out Tensor pinned))
            {
                // The graph reads the KV/state buffers in place; re-upload
                // anything a host-side path dirtied since the last device pass.
                EnsureDecodeGraphInputsResident();
                Ops.Copy(pinned, hidden);
                hidden.Dispose();
                dyn.Write(attendLen, startPos, convPhase, startPos);
                graphs.Replay(key);
                // Mirror the C# bookkeeping the plain loop would have done.
                AdvanceGdnConvPhases(convDim);
                MarkCudaGraphStateModified();
                _cachedRoPEPosStartPos = -1; // device content now diverges from the host cache key
                return pinned;
            }

            if (!graphs.ShouldCapture(key))
                return RunPerOpLayerLoop(hidden, 1, startPos);

            // Pre-warm the RoPE position tensors so their host->device upload
            // happens before capture; the in-graph fill kernel refreshes their
            // CONTENT from the dyn block on every replay.
            Tensor posQ = EnsureRoPEPositions(startPos, 1, Config.NumHeads);
            Tensor posK = EnsureRoPEPositions(startPos, 1, Config.NumKVHeads);
            CudaFusedOps.TryEnsureDeviceResident(posQ);
            CudaFusedOps.TryEnsureDeviceResident(posK);

            Tensor pinnedIn = new Tensor(_allocator, DType.Float32, hidden.Sizes);
            Ops.Copy(pinnedIn, hidden);
            hidden.Dispose();

            dyn.Write(attendLen, startPos, convPhase, startPos);
            if (!graphs.BeginCapture(key))
                return RunPerOpLayerLoop(pinnedIn, 1, startPos);

            bool captured = false;
            try
            {
                dyn.EnqueueUpload();
                dyn.Activate();
                if (!CudaFusedOps.TryFillRopePositions(posQ, posK, dyn))
                    throw new CudaGraphCaptureAbortedException("RoPE position fill kernel unavailable.");
                Tensor result = RunPerOpLayerLoop(pinnedIn, 1, startPos);
                CudaDecodeDynParams.Deactivate();
                if (ReferenceEquals(result, pinnedIn))
                    captured = graphs.EndCaptureAndLaunch(key, pinnedIn, new[] { posQ, posK },
                        CudaDecodeDynParams.CaptureMaxAttendLen);
                else
                    graphs.AbortCapture(key);
            }
            catch (CudaGraphCaptureAbortedException ex)
            {
                graphs.AbortCapture(key, ex.Message);
            }
            catch (TensorSharp.Cuda.Interop.CudaException ex)
            {
                graphs.AbortCapture(key, ex.ToString());
            }
            catch
            {
                graphs.AbortCapture(key);
                throw;
            }
            finally
            {
                CudaDecodeDynParams.Deactivate();
            }

            if (!captured)
            {
                // Nothing executed during the failed capture; run the loop for real.
                return RunPerOpLayerLoop(pinnedIn, 1, startPos);
            }
            _cachedRoPEPosStartPos = -1;
            return pinnedIn;
        }

        /// <summary>Mirrors the per-layer conv ring advance the plain loop's
        /// TryRunCudaNativeGdnPacked does, for decode-graph replays (the C#
        /// loop does not run, but the captured kernel advanced the device ring).</summary>
        private void AdvanceGdnConvPhases(int convDim)
        {
            if (convDim <= 0 || _convStateWriteIdx == null || _isRecurrent == null)
                return;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_isRecurrent[l])
                    _convStateWriteIdx[l] = (_convStateWriteIdx[l] + 1) % convDim;
            }
        }

        /// <summary>Re-uploads any decode-graph input buffer (KV caches, GDN
        /// conv/delta state) whose authoritative copy a host-side path dirtied;
        /// a replayed graph reads the device buffers directly and would
        /// otherwise see stale state. No-ops when everything is device-clean.</summary>
        private void EnsureDecodeGraphInputsResident()
        {
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_isRecurrent[l])
                {
                    if (_cudaGdnConvStateTensor?[l] != null)
                        CudaFusedOps.TryEnsureDeviceResident(_cudaGdnConvStateTensor[l]);
                    if (_deltaStateTensor?[l] != null)
                        CudaFusedOps.TryEnsureDeviceResident(_deltaStateTensor[l]);
                }
                else
                {
                    if (_kvCacheK?[l] != null)
                        CudaFusedOps.TryEnsureDeviceResident(_kvCacheK[l]);
                    if (_kvCacheV?[l] != null)
                        CudaFusedOps.TryEnsureDeviceResident(_kvCacheV[l]);
                }
            }
        }

        /// <summary>The inverse bookkeeping after a graph REPLAY: the graph
        /// rewrote the KV caches and recurrent state on device without going
        /// through the C# launchers, so their host mirrors are stale. Flag it,
        /// or a later host read (state serialization, CPU fallback) would
        /// return pre-replay data.</summary>
        private void MarkCudaGraphStateModified()
        {
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_isRecurrent[l])
                {
                    if (_cudaGdnConvStateTensor?[l] != null)
                        CudaFusedOps.TryMarkDeviceModified(_cudaGdnConvStateTensor[l]);
                    if (_deltaStateTensor?[l] != null)
                        CudaFusedOps.TryMarkDeviceModified(_deltaStateTensor[l]);
                }
                else
                {
                    if (_kvCacheK?[l] != null)
                        CudaFusedOps.TryMarkDeviceModified(_kvCacheK[l]);
                    if (_kvCacheV?[l] != null)
                        CudaFusedOps.TryMarkDeviceModified(_kvCacheV[l]);
                }
            }
        }

        protected override float[] ForwardRefillCore(int[] tokens)
        {
            // Prefill runs the recurrent state on the host; re-seed the fused
            // decode's device-resident GDN state afterwards.
            InvalidateFullDecodeState();
            if (tokens == null || tokens.Length <= 1)
                return ForwardCore(tokens);

            // The chunked prefill path (PrefillWithoutLogits) uses the non-TP
            // layer loop and non-sharded weights, which are unavailable under
            // tensor parallelism. Route through ForwardCore → ForwardTP instead.
            if (IsTensorParallel)
                return ForwardCore(tokens);

            // Multimodal embeddings carry positions relative to the current
            // Forward call's hidden tensor; chunked prefill would need to
            // remap them per-chunk. Fall back to single Forward when any are pending.
            bool hasMultimodal = _visionEmbeddingsList.Count > 0;
            int chunkSize = ResolvePrefillChunkSize();
            int lastIdx = tokens.Length - 1;

            if (hasMultimodal || tokens.Length <= chunkSize)
                return ForwardCore(tokens);

            // The FINAL chunk carries the last prompt token and produces the
            // logits, so every prompt position - including the one the first
            // sampled token is drawn from - is evaluated by the same prefill
            // graph. Running that last token alone through the 1-token decode
            // graph (as this did) drew the first logits from a structurally
            // different kernel than the rest of the prompt, and than the
            // corresponding llama.cpp ubatch.
            for (int pos = 0; pos < tokens.Length; pos += chunkSize)
            {
                int chunkLen = Math.Min(chunkSize, tokens.Length - pos);
                var chunk = new int[chunkLen];
                Array.Copy(tokens, pos, chunk, 0, chunkLen);
                if (pos + chunkLen >= tokens.Length)
                    return ForwardCore(chunk);
                PrefillWithoutLogits(chunk);
            }

            throw new InvalidOperationException("Chunked prefill produced no logits.");
        }

        private void PrefillWithoutLogits(int[] tokens)
        {
            if (tokens == null || tokens.Length == 0)
                return;

            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            EnsureCacheCapacity(startPos + seqLen);

            long t1 = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);
            long embEnd = Stopwatch.GetTimestamp();
            _embTicks += embEnd - t1;

            // Whole-model fused prefill chunk (same path as Forward, but the logits
            // are discarded ÔÇö this is an interior chunk). KV + GDN state are written
            // on-device; the next chunk / decode reads the committed state. Carries
            // across chunks exactly like the per-op loop (the kernel attends over the
            // full [0, startPos+seqLen) cache and reads the committed GDN ring).
            if (CanUsePrefillVerify(startPos, seqLen))
            {
                if (_logitsBuffer == null || _logitsBuffer.Length != Config.VocabSize)
                    _logitsBuffer = new float[Config.VocabSize];
                if (TryFullModelVerify(hidden, startPos, seqLen, normedOut: null, logitsOut: _logitsBuffer, nLogitRows: 1))
                {
                    hidden.Dispose();
                    _cacheSeqLen += seqLen;
                    InvalidateFullDecodeState();
                    _forwardSw.Stop();
                    return;
                }
            }

            EnsureKvCacheHostSynchronized();
            EnsureFusedDecodeStateHostSynchronized();
            // The per-op recurrent kernels below invalidate and replace their
            // host-keyed delta-state device buffers. A retained Metal decode
            // graph still binds those buffers, so drop that graph before the
            // fallback mutates the cache entries.
            InvalidateFullDecodeState(hardBindings: true);
            hidden = RunCudaPrefillLayerLoop(hidden, seqLen, startPos);

            hidden.Dispose();
            _cacheSeqLen += seqLen;
            _forwardSw.Stop();
        }

        internal static bool CanTryDirectTokenDecode(
            BackendType backend,
            bool isTensorParallel,
            int sequenceLength,
            bool hasVisionEmbeddings,
            bool hasMRoPEPositions,
            int tokenId,
            int vocabSize)
        {
            return backend == BackendType.GgmlMetal
                && !isTensorParallel
                && sequenceLength == 1
                && !hasVisionEmbeddings
                && !hasMRoPEPositions
                && tokenId >= 0
                && tokenId < vocabSize;
        }

        protected override float[] ForwardCore(int[] tokens)
        {
            if (IsTensorParallel)
                return ForwardTP(tokens);

            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            EnsureCacheCapacity(startPos + seqLen);

            if (_logitsBuffer == null || _logitsBuffer.Length != Config.VocabSize)
                _logitsBuffer = new float[Config.VocabSize];

            // llama.cpp feeds a token id into its decode graph and performs the
            // quantized embedding lookup on-device. Do the same for ordinary
            // text-only Metal decode, avoiding a per-token CPU dequantization,
            // Tensor allocation, and hidden-vector upload.
            int directTokenId = seqLen == 1 ? tokens[0] : -1;
            if (CanTryDirectTokenDecode(
                    _backend,
                    IsTensorParallel,
                    seqLen,
                    _visionEmbeddingsList.Count != 0,
                    _pendingMRoPEPositions != null,
                    directTokenId,
                    Config.VocabSize) &&
                TryFullModelDecodeToken(directTokenId, startPos, _logitsBuffer))
            {
                _cacheSeqLen++;
                _forwardCount++;
                _forwardSw.Stop();
                return _logitsBuffer;
            }

            long t1 = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);
            long embEnd = Stopwatch.GetTimestamp();
            _embTicks += embEnd - t1;

            // Inject any vision embeddings queued for this Forward call. The
            // queued positions are already expressed relative to the current
            // tokens slice (see ModelMultimodalInjector.QueuePreparedVisionEmbeddings),
            // so this works for both a fresh prefill (startPos == 0) and a
            // multi-turn refill that re-uses the prior KV cache (startPos > 0).
            // The latter is the path taken when a user uploads a second image
            // in the same chat session - gating on startPos == 0 here used to
            // silently drop the second image's vision embeddings and turn its
            // <|image_pad|> tokens into garbage.
            if (_visionEmbeddingsList.Count > 0)
                InjectVisionEmbeddings(hidden, seqLen);


            // Fast path: run the whole hybrid transformer (incl. final-norm + lm_head)
            // as one fused, CUDA-graph-captured GGML graph that outputs LOGITS
            // directly. Collapses ~120-400 per-op dispatches/token + the separate
            // lm_head graph_compute into a single captured replay.
            if (seqLen == 1 && TryFullModelDecode(hidden, startPos, _logitsBuffer))
            {
                // _logitsBuffer holds the final logits; skip the per-op loop + lm head.
                hidden.Dispose();
                _cacheSeqLen += seqLen;
                _forwardCount++;
                _pendingMRoPEPositions = null;
                _forwardSw.Stop();
                return _logitsBuffer;
            }

            // Prefill fast path: run the WHOLE hybrid transformer (attention + GDN +
            // MoE + final-norm + last-token lm_head) for all N prompt tokens as ONE
            // fused GGML graph (TSGgml_Qwen35ModelVerify with nLogitRows=1), writing
            // KV + GDN state on-device. This replaces the per-layer dispatch loop
            // whose host round-trip per op keeps the GPU mostly idle - the dominant
            // CUDA prefill cost. Mirrors Gemma4's whole-model prefill-verify routing.
            // Image prompts take this path too: the graph applies MRoPE itself
            // (ggml_rope_multi / IMROPE) from the mropePos+mropeSecs arguments
            // TryFullModelVerify packs. Long prompts chunk via ForwardRefill.
            if (seqLen > 1 && CanUsePrefillVerify(startPos, seqLen)
                && TryFullModelVerify(hidden, startPos, seqLen, normedOut: null, logitsOut: _logitsBuffer, nLogitRows: 1))
            {
                // _logitsBuffer holds the LAST token's logits; KV + GDN state were
                // committed (host ring) by the kernel. Re-seed the device-resident
                // fused-decode GDN state next decode (the per-op convention).
                hidden.Dispose();
                _cacheSeqLen += seqLen;
                _forwardCount++;
                _pendingMRoPEPositions = null;
                InvalidateFullDecodeState();
                _forwardSw.Stop();
                return _logitsBuffer;
            }
            {
            // Per-op path runs the recurrent state on the host, so the fused
            // decode's device-resident GDN state must be re-seeded next time.
            EnsureKvCacheHostSynchronized();
            EnsureFusedDecodeStateHostSynchronized();
            // RecurrentBlock may invalidate the host-keyed delta-state device
            // buffers. Retaining a Metal graph across that transition would
            // leave it holding freed buffer bindings.
            InvalidateFullDecodeState(hardBindings: true);
            hidden = RunCudaPrefillLayerLoop(hidden, seqLen, startPos);
            }

            // Pick out the last token's hidden state BEFORE the final norm so we can
            // fuse final-norm + LM-head matmul into a single GGML kernel for decode.
            Tensor lastHiddenRaw;
            if (seqLen > 1)
            {
                using var narrowed = hidden.Narrow(0, seqLen - 1, 1);
                lastHiddenRaw = Ops.NewContiguous(narrowed);
            }
            else
            {
                lastHiddenRaw = hidden.CopyRef();
            }
            hidden.Dispose();

            long t2 = Stopwatch.GetTimestamp();
            Tensor logitsTensor;
            if (IsGgmlBackend && _lmHeadQW != null && _finalNormW != null
                && lastHiddenRaw.DimensionCount == 2)
            {
                logitsTensor = FusedNormLinear(lastHiddenRaw, _finalNormW, _lmHeadQW, _lmHeadF32);
            }
            else
            {
                Tensor lastNormed = RMSNormOpCached(lastHiddenRaw, _finalNormW);
                logitsTensor = LinearForwardCached(lastNormed, _lmHeadQW, _lmHeadF32);
                lastNormed.Dispose();
            }
            lastHiddenRaw.Dispose();
            if (_backend == BackendType.Mlx && MlxEvalFinalLogits)
                MlxFusedOps.TryEvaluate(logitsTensor);
            long lmHeadEnd = Stopwatch.GetTimestamp();
            _lmHeadTicks += lmHeadEnd - t2;

            long t3 = Stopwatch.GetTimestamp();
            if (_logitsBuffer == null || _logitsBuffer.Length != Config.VocabSize)
                _logitsBuffer = new float[Config.VocabSize];
            unsafe
            {
                float* src = GetFloatPtr(logitsTensor);
                fixed (float* dst = _logitsBuffer)
                    Buffer.MemoryCopy(src, dst, Config.VocabSize * 4, Config.VocabSize * 4);
            }
            _logitsCopyTicks += Stopwatch.GetTimestamp() - t3;
            logitsTensor.Dispose();

            _cacheSeqLen += seqLen;
            _forwardCount++;
            // Drop the MRoPE positions staged by the injector so the next
            // Forward (e.g. a decode token, or a different sequence under
            // the engine's per-seq KV swap) starts from a clean slate.
            _pendingMRoPEPositions = null;
            _forwardSw.Stop();
            return _logitsBuffer;
        }

        // Pipelined greedy decode is implemented only on the MLX path
        // (uses MLX argmax + device-side embedding lookup); other backends
        // fall through to the standard Forward + host-side sample loop.
        public override bool SupportsPipelinedGreedy =>
            _backend == BackendType.Mlx
            && _lmHeadQW != null
            && _finalNormW != null
            && _quantWeights.ContainsKey("token_embd.weight");

        // ===== Pipelined greedy decode =====
        //
        // Standard Forward(int[] tokens) issues all 60 layers + the LM head
        // and then host-syncs the 200K-float logits tensor before returning.
        // The sync drains all queued MLX kernels, costing ~8 ms/token on MLX
        // (Qwen3.6-35B-A3B IQ2_XXS) ÔÇö pure GPU-idle wait from the host's
        // perspective.
        //
        // The pipelined API lets the CLI inference loop kick off forward N+1
        // BEFORE syncing forward N's token. The trick: compute argmax on
        // device, look up the next embedding on device, and return the
        // resulting [1] int32 device tensor as a deferred handle. The
        // caller queues the next step and only THEN host-reads the previous
        // step's token.
        //
        // Greedy only: top-K / temperature sampling still needs the full
        // logits on host. Gated by TS_MLX_PIPELINED_DECODE=1 (off by default).

        // Run one pipelined decode step. Returns a [1] int32 device tensor
        // holding the predicted next token. The caller is responsible for
        // syncing it to host (via Tensor.GetElementsAsInt) and disposing.
        // If firstTokenForBegin is non-null this is the first call after
        // prefill: we look up the embedding for that host int. Otherwise
        // we use the cached _pipelineNextInputDevice from the previous
        // call. Each call queues the next embedding lookup (via on-device
        // argmax) so the next call can run without any host work besides
        // building the layer graph.
        public override Tensor SubmitGreedyDecodeStep(int? firstTokenForBegin)
        {
            _forwardSw.Start();
            int seqLen = 1;
            int startPos = _cacheSeqLen;
            EnsureCacheCapacity(startPos + seqLen);

            Tensor inputHidden;
            if (firstTokenForBegin.HasValue)
            {
                // Begin path: upload host int and look up embedding on host.
                // Subsequent calls go through the device lookup path.
                if (_pipelineNextInputDevice != null)
                {
                    _pipelineNextInputDevice.Dispose();
                    _pipelineNextInputDevice = null;
                }
                long embT0 = Stopwatch.GetTimestamp();
                inputHidden = Embedding(new[] { firstTokenForBegin.Value });
                _embTicks += Stopwatch.GetTimestamp() - embT0;
            }
            else if (_pipelineNextInputDevice != null)
            {
                inputHidden = _pipelineNextInputDevice;
                _pipelineNextInputDevice = null;
            }
            else
            {
                throw new InvalidOperationException(
                    "SubmitGreedyDecodeStep: no cached input embedding and no firstTokenForBegin provided.");
            }

            // Run the layer stack. Same path as Forward.
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                if (_isRecurrent[layer])
                    inputHidden = RecurrentBlock(inputHidden, layer, seqLen, startPos);
                else
                    inputHidden = AttentionBlock(inputHidden, layer, seqLen, startPos);
                TryEvaluateMlxLayerBoundary(inputHidden, layer, seqLen);
            }

            // Final norm + LM head.
            long lmT0 = Stopwatch.GetTimestamp();
            Tensor lastNormed = RMSNormOpCached(inputHidden, _finalNormW);
            inputHidden.Dispose();
            Tensor logitsTensor = LinearForwardCached(lastNormed, _lmHeadQW, _lmHeadF32);
            lastNormed.Dispose();
            _lmHeadTicks += Stopwatch.GetTimestamp() - lmT0;

            // Device argmax ÔåÆ [1] int32. Falls back to host argmax if MLX
            // path fails (e.g. non-MLX backend or unsupported dtype).
            var deviceToken = new Tensor(_allocator, DType.Int32, 1);
            if (!MlxFusedOps.TryArgMaxLastAxis(deviceToken, logitsTensor))
            {
                // Host fallback. This forces a sync but only fires when MLX
                // argmax isn't available; the loop will still work, just
                // without the pipelining benefit.
                unsafe
                {
                    float* src = GetFloatPtr(logitsTensor);
                    int maxIdx = 0;
                    float maxVal = src[0];
                    int vocab = (int)logitsTensor.ElementCount();
                    for (int i = 1; i < vocab; i++)
                    {
                        float v = src[i];
                        if (v > maxVal) { maxVal = v; maxIdx = i; }
                    }
                    deviceToken.SetElementsAsInt(new[] { maxIdx });
                }
            }
            logitsTensor.Dispose();

            // Pre-compute the input embedding for the NEXT call so the
            // caller can submit the next forward without any host work
            // related to the predicted token.
            _pipelineNextInputDevice = new Tensor(_allocator, DType.Float32, 1, Config.HiddenSize);
            if (!TryComputeNextInputEmbedding(_pipelineNextInputDevice, deviceToken))
            {
                // If device embedding lookup failed, sync the token and use
                // the host path. This is the fallback for backends without
                // device get_rows or unsupported quant types.
                int hostTok = deviceToken.GetElementsAsInt(1)[0];
                _pipelineNextInputDevice.Dispose();
                _pipelineNextInputDevice = Embedding(new[] { hostTok });
            }
            else if (_backend == BackendType.Mlx)
            {
                // Kick the queued kernels (argmax + embedding lookup) so
                // they start executing on Metal before the host issues the
                // next forward step. The host sync of `deviceToken` will
                // also drain this implicitly, but the explicit async-eval
                // gives MLX a chance to schedule the work earlier.
                MlxFusedOps.TryAsyncEvaluate(_pipelineNextInputDevice);
            }

            _cacheSeqLen += seqLen;
            _forwardCount++;
            _pendingMRoPEPositions = null;
            _forwardSw.Stop();

            return deviceToken;
        }

        // Look up token_embd row for a [1] int32 device token, writing the
        // [1, hidden] result into outEmb. Reuses the existing MLX quantized
        // get_rows path (MlxQuantizedOps.TryGetRowsQuantizedToFloat32) for
        // quantized embedding tables; otherwise returns false so the caller
        // falls back to the host path.
        private bool TryComputeNextInputEmbedding(Tensor outEmb, Tensor deviceTokenInt)
        {
            if (_backend != BackendType.Mlx)
                return false;

            if (_quantWeights.TryGetValue("token_embd.weight", out var qw))
            {
                return MlxQuantizedOps.TryGetRowsQuantizedToFloat32(
                    outEmb,
                    qw.CacheKey,
                    qw.Data,
                    qw.GgmlType,
                    qw.Ne0,
                    qw.Ne1,
                    qw.RawBytes,
                    deviceTokenInt);
            }

            // F32 token_embd path: use Ops.IndexSelect if supported on MLX.
            // Most quantized models won't hit this ÔÇö left as a TODO so we
            // fall back to host path until/unless someone tests it.
            return false;
        }

        // Release any pending pipelined-decode state. Call at end of a
        // generation run so resources don't linger.
        public override void ResetPipelinedGreedyState()
        {
            _pipelineNextInputDevice?.Dispose();
            _pipelineNextInputDevice = null;
        }

        private void TryEvaluateMlxLayerBoundary(Tensor hidden, int layer, int seqLen)
        {
            if (_backend != BackendType.Mlx || MlxEvalEveryNLayers <= 0)
            {
                return;
            }

            if (seqLen == 1 && !MlxEvalDecodeLayerBoundaries)
            {
                if ((layer + 1) % MlxEvalEveryNLayers == 0 || layer + 1 == Config.NumLayers)
                {
                    int firstDecodeLayer = Math.Max(0, layer - MlxEvalEveryNLayers + 1);
                    TryEvaluateMlxLayerCacheState(firstDecodeLayer, layer);
                }
                return;
            }

            if (hidden == null
                || ((layer + 1) % MlxEvalEveryNLayers != 0 && layer + 1 != Config.NumLayers))
            {
                return;
            }

            long start = Stopwatch.GetTimestamp();
            // Async at intermediate boundaries; the LM head/RMSNorm at the end
            // of forward will drain the graph (or the sampler's host read does).
            // Last layer keeps a sync eval as a safety net.
            bool isLastLayer = (layer + 1) == Config.NumLayers;
            bool evaluated = isLastLayer
                ? MlxFusedOps.TryEvaluate(hidden)
                : MlxFusedOps.TryAsyncEvaluate(hidden);
            if (evaluated)
                _mlxEvalBoundaryTicks += Stopwatch.GetTimestamp() - start;

            int firstLayer = Math.Max(0, layer - MlxEvalEveryNLayers + 1);
            TryEvaluateMlxLayerCacheState(firstLayer, layer);
        }

        private void TryEvaluateMlxLayerCacheState(int firstLayer, int lastLayer)
        {
            if (_backend != BackendType.Mlx)
                return;

            long start = Stopwatch.GetTimestamp();
            bool evaluated = false;
            for (int l = firstLayer; l <= lastLayer && l < Config.NumLayers; l++)
            {
                if (l < 0)
                    continue;

                if (_isRecurrent[l])
                    evaluated |= _mlxGdnCache?[l]?.TryEvaluateState() == true;
                else
                    evaluated |= _mlxAttentionCache?[l]?.TryEvaluateState() == true;
            }

            if (evaluated)
                _mlxCacheEvalTicks += Stopwatch.GetTimestamp() - start;
        }

        #region Full Attention Block

        /// <summary>
        /// Full attention with gated Q, QK-norm, sigmoid-gated output, and post-attention norm.
        /// Q projection outputs 2x: [Q, gate] interleaved per head.
        /// </summary>
        private Tensor AttentionBlock(Tensor hidden, int layer, int seqLen, int startPos)
        {
            // Decode fast path (long context): fold the entire attention block (norm + QKV +
            // QK-norm + RoPE + KV cache append + flash attention + sigmoid-gated mix + output
            // projection + residual add) into one fused GGML graph dispatch. This pays off once
            // the per-layer CPU attention cost (which scales with the cached sequence length)
            // exceeds the cost of building the fused graph + the GPU flash-attn dispatch. For
            // short contexts the optimised CPU SIMD attention + 2 small GGML dispatches is
            // already faster, so we keep the existing path.
            int totalSeqLen = startPos + seqLen;
            bool fusedDecodeApplied = false;
            if (seqLen == 1 && totalSeqLen >= FusedAttnLayerDecodeMinSeqLen
                && TryFusedAttnLayerDecode(hidden, layer, startPos))
            {
                fusedDecodeApplied = true;
            }

            // Fuse:
            //   hidden = hidden + attn_out_proj(attention(rms_norm(hidden)))
            // into the FullAttention call: input norm + QKV is fused inside FullAttention, and
            // when the GGML backend is available the output projection is also fused with the
            // residual add (FusedMatMulQuantAdd) so the residual lives entirely on the GPU.
            // Fused outproj+FFN for attention layers: when the fused attention layer
            // decode is NOT used and the layer is dense FFN (not MoE), fuse the attention
            // output projection + residual + FFN into one GPU dispatch.
            bool canFuseAttnOutFFN = !fusedDecodeApplied && IsGgmlBackend
                && !(_isMoeLayer != null && _isMoeLayer[layer])
                && _attnOutputQW[layer] != null
                && _postAttnNormW[layer] != null
                && _ffnGateUpQW[layer] != null
                && _ffnDownQW[layer] != null
                && Unscaled(_attnOutputQW[layer], _ffnGateUpQW[layer], _ffnDownQW[layer]);

            Tensor attnOut;
            if (canFuseAttnOutFFN)
                attnOut = FullAttention(hidden, _attnNormW[layer], layer, seqLen, startPos,
                    residual: null, skipOutputProj: true);
            else if (fusedDecodeApplied)
                attnOut = null;
            else
                attnOut = FullAttention(hidden, _attnNormW[layer], layer, seqLen, startPos, residual: hidden);


            if (canFuseAttnOutFFN && attnOut != null)
            {
                int intermSize = Config.IntermediateSize;
                int halfDim = intermSize > 0 ? intermSize : (int)(_ffnGateUpQW[layer].Ne1 / 2);

                if (halfDim > 0 && !_fusedOutProjFfnFailed
                    && hidden.DimensionCount == 2 && attnOut.DimensionCount == 2
                    && hidden.Sizes[0] == attnOut.Sizes[0])
                {
                    try
                    {
                        long t0 = Stopwatch.GetTimestamp();
                        GgmlBasicOps.FusedOutProjFFN(
                            hidden, attnOut,
                            _attnOutputQW[layer].CacheKey, _attnOutputQW[layer].GgmlType,
                            _attnOutputQW[layer].Ne0, _attnOutputQW[layer].Ne1, _attnOutputQW[layer].RawBytes,
                            _postAttnNormW[layer], Config.Eps,
                            _ffnGateUpQW[layer].CacheKey, _ffnGateUpQW[layer].GgmlType,
                            _ffnGateUpQW[layer].Ne0, _ffnGateUpQW[layer].Ne1, _ffnGateUpQW[layer].RawBytes,
                            _ffnDownQW[layer].CacheKey, _ffnDownQW[layer].GgmlType,
                            _ffnDownQW[layer].Ne0, _ffnDownQW[layer].Ne1, _ffnDownQW[layer].RawBytes,
                            halfDim);
                        _linearTicks += Stopwatch.GetTimestamp() - t0;
                        attnOut.Dispose();
                        return hidden;
                    }
                    catch (Exception e)
                    {
                        _fusedOutProjFfnFailed = true;
                        Console.Error.WriteLine(
                            $"[qwen35] FusedOutProjFFN failed on the attention path ({e.Message}); " +
                            "using the separate out-proj + FFN dispatches instead (slower). " +
                            "Not retried. Reported once.");
                        // fall through
                    }
                }

                // Fallback: separate output proj + residual.
                if (TryLinearAddInto(hidden, attnOut, _attnOutputQW[layer]))
                    attnOut.Dispose();
                else
                {
                    var o = LinearForwardCached(attnOut, _attnOutputQW[layer], _attnOutputF32[layer]);
                    attnOut.Dispose();
                    Ops.Add(hidden, hidden, o);
                    o.Dispose();
                }
                attnOut = null;
            }

            if (attnOut != null)
            {
                Ops.Add(hidden, hidden, attnOut);
                attnOut.Dispose();
            }

            Tensor ffnOut;
            if (_isMoeLayer != null && _isMoeLayer[layer])
            {
                Tensor normed2;
                bool ownsNormed2 = true;
                if (seqLen == 1 && IsGgmlBackend && _moeTokenInput != null
                    && _moeTokenInput.Sizes[1] == Config.HiddenSize
                    && _postAttnNormW[layer] != null)
                {
                    RMSNormToBufferCpu(_moeTokenInput, hidden, _postAttnNormW[layer], Config.HiddenSize, Config.Eps);
                    normed2 = _moeTokenInput;
                    ownsNormed2 = false;
                }
                else
                    normed2 = RMSNormOpCached(hidden, _postAttnNormW[layer]);

                if (seqLen == 1 && TryMoEResidualDecode(hidden, normed2, layer))
                    ffnOut = null;
                else
                    ffnOut = MoEForward(normed2, layer, seqLen);
                if (ownsNormed2)
                    normed2.Dispose();
            }
            else
            {
                ffnOut = FFNCachedFused(hidden, _postAttnNormW[layer], layer, seqLen);
            }

            if (ffnOut != null)
            {
                Ops.Add(hidden, hidden, ffnOut);
                ffnOut.Dispose();
            }


            return hidden;
        }

        private unsafe Tensor FullAttention(Tensor input, Tensor inputNormW, int layer, int seqLen, int startPos,
            Tensor residual = null, bool skipOutputProj = false)
        {
            int numHeads = Config.NumHeads;
            int numKVHeads = Config.NumKVHeads;
            int headDim = Config.HeadDim;
            int qFullDim = numHeads * headDim * 2;
            int kvDim = numKVHeads * headDim;
            int totalSeqLen = startPos + seqLen;

            // Fused norm + QKV when the fused-QKV weight is available; otherwise we have to
            // produce the normalized input separately for the three independent projections.
            Tensor qFull;
            Tensor kTensor;
            Tensor vTensor;
            Tensor normedInput = null;
            Tensor fusedQkv;
            bool ownsFusedQkv = true;
            if (inputNormW != null && _attnQkvQW[layer] != null)
            {
                // Decode hot path: write fused norm+QKV into the pre-allocated buffer to
                // avoid a tensor allocation per layer per token. Falls back to the regular
                // allocating path for prefill (multi-row) or when the buffer doesn't fit.
                if (seqLen == 1 && _attnDecodeQkvBuf != null
                    && _attnDecodeQkvBuf.Sizes[1] == qFullDim + 2 * kvDim)
                {
                    fusedQkv = TryFusedNormLinearInto(_attnDecodeQkvBuf, input, inputNormW, _attnQkvQW[layer]);
                    if (fusedQkv != null)
                        ownsFusedQkv = false;
                    else
                        fusedQkv = FusedNormLinear(input, inputNormW, _attnQkvQW[layer], _attnQkvF32[layer]);
                }
                else
                {
                    fusedQkv = FusedNormLinear(input, inputNormW, _attnQkvQW[layer], _attnQkvF32[layer]);
                }
            }
            else
            {
                normedInput = inputNormW != null ? RMSNormOpCached(input, inputNormW) : input.CopyRef();
                fusedQkv = LinearForwardCached(normedInput, _attnQkvQW[layer], _attnQkvF32[layer]);
            }
            if (fusedQkv != null)
            {
                if (seqLen == 1)
                {
                    qFull = fusedQkv.Narrow(1, 0, qFullDim);
                    kTensor = fusedQkv.Narrow(1, qFullDim, kvDim);
                    vTensor = fusedQkv.Narrow(1, qFullDim + kvDim, kvDim);
                }
                else
                {
                    using (var qView = fusedQkv.Narrow(1, 0, qFullDim))
                        qFull = Ops.NewContiguous(qView);
                    using (var kView = fusedQkv.Narrow(1, qFullDim, kvDim))
                        kTensor = Ops.NewContiguous(kView);
                    using (var vView = fusedQkv.Narrow(1, qFullDim + kvDim, kvDim))
                        vTensor = Ops.NewContiguous(vView);
                }
                if (ownsFusedQkv)
                    fusedQkv.Dispose();
            }
            else
            {
                if (normedInput == null)
                    normedInput = inputNormW != null ? RMSNormOpCached(input, inputNormW) : input.CopyRef();

                // Q projection outputs [seqLen, numHeads * headDim * 2] (Q + gate interleaved per head).
                qFull = LinearForwardCached(normedInput, _attnQQW[layer], _attnQF32[layer]);
                kTensor = LinearForwardCached(normedInput, _attnKQW[layer], _attnKF32[layer]);
                vTensor = LinearForwardCached(normedInput, _attnVQW[layer], _attnVF32[layer]);
            }
            normedInput?.Dispose();

            Tensor qTensor, gateTensor;
            bool ownsQGateBuffers;
            DeinterleaveQGate(qFull, seqLen, numHeads, headDim, out qTensor, out gateTensor, out ownsQGateBuffers);
            qFull.Dispose();

            // Fused QK-RMSNorm + NeoX RoPE path: on CUDA backend without MRoPE,
            // fuse the per-head RMSNorm and RoPE into a single kernel per Q/K tensor.
            // This eliminates 2-4 separate kernel launches (2x RMSNorm + 2x RoPE)
            // and the intermediate global memory writes of the normalized tensors.
            bool useMRoPE = _pendingMRoPEPositions != null && _pendingMRoPEPositions.Length >= 3 * seqLen;
            bool fusedNormRopeApplied = false;
            // TS_FUSED_QKNORM_ROPE=0 disables the fused path. Default: ON (lookup-table optimized).
            bool fusedEnabled = !string.Equals(Environment.GetEnvironmentVariable("TS_FUSED_QKNORM_ROPE"), "0", StringComparison.Ordinal);
            if (fusedEnabled && _backend == BackendType.Cuda && !useMRoPE)
            {
                int ropeDim = _ropeDimCount > 0 ? _ropeDimCount : headDim;
                int ropeHalf = ropeDim / 2;
                int qRows = seqLen * numHeads;
                int kRows = seqLen * numKVHeads;

                Tensor qPosTensor = EnsureRoPEPositions(startPos, seqLen, numHeads);
                Tensor kPosTensor = numHeads == numKVHeads
                    ? qPosTensor
                    : EnsureRoPEPositions(startPos, seqLen, numKVHeads);

                bool qOk = CudaFusedOps.TryQKNormRopeNeox(
                    qTensor, _attnQNormW[layer], qPosTensor,
                    qRows, headDim, ropeHalf, Config.Eps,
                    Config.RopeBase, 1.0f / Config.RopeScale);

                bool kOk = qOk && CudaFusedOps.TryQKNormRopeNeox(
                    kTensor, _attnKNormW[layer], kPosTensor,
                    kRows, headDim, ropeHalf, Config.Eps,
                    Config.RopeBase, 1.0f / Config.RopeScale);

                fusedNormRopeApplied = qOk && kOk;
            }

            if (!fusedNormRopeApplied)
            {
                // Separate RMSNorm + RoPE path (non-CUDA or fused fallback).
                qTensor = ApplyQKNormCached(qTensor, _attnQNormW[layer], numHeads, seqLen);
                kTensor = ApplyQKNormCached(kTensor, _attnKNormW[layer], numKVHeads, seqLen);

                if (useMRoPE)
                {
                    qTensor = ApplyMRoPEPrefill(qTensor, numHeads, seqLen, _pendingMRoPEPositions);
                    kTensor = ApplyMRoPEPrefill(kTensor, numKVHeads, seqLen, _pendingMRoPEPositions);
                }
                else if (seqLen == 1)
                {
                    if (_backend == BackendType.Mlx || _backend == BackendType.Cuda)
                    {
                        qTensor = ApplyRoPEPrefill(qTensor, numHeads, seqLen, startPos);
                        kTensor = ApplyRoPEPrefill(kTensor, numKVHeads, seqLen, startPos);
                    }
                    else
                    {
                        ApplyRoPEDecodeQKInPlace(qTensor, kTensor, numHeads, numKVHeads, startPos);
                    }
                }
                else
                {
                    qTensor = ApplyRoPEPrefill(qTensor, numHeads, seqLen, startPos);
                    kTensor = ApplyRoPEPrefill(kTensor, numKVHeads, seqLen, startPos);
                }
            }

            long t0 = Stopwatch.GetTimestamp();

            // Attention computation
            float attentionScale = 1.0f / MathF.Sqrt(headDim);
            Tensor attnOutput = null;

            if (seqLen == 1)
            {
                // Reuse the pre-allocated decode output buffer to avoid a per-token allocation.
                attnOutput = _attnDecodeOutBuf != null
                    ? _attnDecodeOutBuf
                    : new Tensor(_allocator, DType.Float32, 1, numHeads * headDim);

                int maxSeqLen = (int)_kvCacheK[layer].Sizes[1];

                // Fast path: fuse KV cache append + flash attention into a single device graph.
                // Only worth the GPU dispatch overhead once the cache is large enough that
                // CPU SIMD attention starts to dominate. Below the threshold the per-layer
                // per-token Metal command buffer setup costs more than the saved compute.
                bool flashOk = false;
                bool cacheCopied = false;
                if (_backend == BackendType.Mlx
                    && headDim == 256
                    && _mlxAttentionCache?[layer] != null
                    && _mlxAttentionCache[layer].Length == startPos)
                {
                    using Tensor qHeads = qTensor.View(numHeads, 1, headDim);
                    using Tensor kHeads = kTensor.View(numKVHeads, 1, headDim);
                    using Tensor vHeads = vTensor.View(numKVHeads, 1, headDim);
                    flashOk = _mlxAttentionCache[layer].TryAttentionHeadDim256(
                        attnOutput,
                        qHeads,
                        kHeads,
                        vHeads,
                        numHeads,
                        numKVHeads,
                        1,
                        startPos,
                        causal: true);
                    cacheCopied = flashOk;
                }
                else if (IsGgmlBackend && totalSeqLen >= FlashAttnDecodeMinSeqLen)
                {
                    flashOk = TryFlashAttnDecode(qTensor, kTensor, vTensor,
                        _kvCacheK[layer], _kvCacheV[layer], attnOutput,
                        numHeads, numKVHeads, headDim, maxSeqLen, startPos, attentionScale);
                }
                else if (_backend == BackendType.Mlx && totalSeqLen >= MlxFlashAttnDecodeMinSeqLen)
                {
                    CopyToCacheDecode(_kvCacheK[layer], kTensor, _kvCacheV[layer], vTensor,
                        numKVHeads, headDim, startPos);
                    cacheCopied = true;
                    flashOk = MlxFusedOps.TryDecodeAttention(
                        attnOutput, qTensor, _kvCacheK[layer], _kvCacheV[layer],
                        numHeads, numKVHeads, headDim,
                        0, totalSeqLen, maxSeqLen, false, attentionScale);
                }

                if (!flashOk)
                {
                    if (!cacheCopied)
                        CopyToCacheDecode(_kvCacheK[layer], kTensor, _kvCacheV[layer], vTensor,
                            numKVHeads, headDim, startPos);

                    // Direct-CUDA GQA decode attention runs entirely on the GPU, reading
                    // the (head-first) KV cache in place. The legacy AttentionDecodePureCS
                    // path copies the whole allocated cache to the host every layer (a 4 MB
                    // DtoH that drains the pipeline 16x per token) and computes attention on
                    // the CPU ÔÇö by far the dominant per-token stall on the cuda backend.
                    bool cudaAttn = _backend == BackendType.Cuda &&
                        CudaFusedOps.TryGqaDecodeAttention(
                            attnOutput, qTensor, _kvCacheK[layer], _kvCacheV[layer],
                            numHeads, numKVHeads, headDim,
                            0, totalSeqLen, maxSeqLen, false, attentionScale);

                    if (!cudaAttn)
                        AttentionDecodePureCS(qTensor, _kvCacheK[layer], _kvCacheV[layer],
                            attnOutput, numHeads, numKVHeads, headDim, totalSeqLen, attentionScale);
                }

                kTensor.Dispose();
                vTensor.Dispose();
                if (ownsQGateBuffers)
                    qTensor.Dispose();
            }
            else
            {

                Tensor kHeads = ReshapeToHeads(kTensor, numKVHeads, seqLen, headDim);
                Tensor vHeads = ReshapeToHeads(vTensor, numKVHeads, seqLen, headDim);


                // Fused GPU attention: Q*K^T ÔåÆ causal mask ÔåÆ softmax ÔåÆ *V in one
                // GGML graph dispatch, eliminating ExpandKVHeads + 5 separate ops.
                //
                // FusedPrefillAttention is F32-only on the native side. The continuation
                // branch (startPos > 0) reads K/V directly from the persistent KV cache,
                // which may be Float16 when the caller picked a quantized cache dtype
                // (e.g. on IQ2_XXS / Q4_K models where ApplyModelAlignedKvCacheDefault
                // selects F16 to halve cache memory). Reading F16 bytes as F32 silently
                // produces garbage that propagates through softmax+V into the residual,
                // ultimately surfacing as a NaN router probability in the next MoE layer
                // and an out-of-range expert index that crashes inside the native
                // ggml_mul_mat_id dispatch. Restrict the continuation fast path to F32
                // caches and fall back to ExpandKVHeads (which handles F16) otherwise.
                bool usedFusedAttn = false;
                bool usedMlxAttentionCache = false;
                bool kvCacheIsF32 = _kvCacheK[layer].ElementType == DType.Float32
                    && _kvCacheV[layer].ElementType == DType.Float32;
                bool canFuseContinuation = kvCacheIsF32 || startPos == 0;
                if (_backend == BackendType.Mlx
                    && headDim == 256
                    && _mlxAttentionCache?[layer] != null
                    && _mlxAttentionCache[layer].Length == startPos)
                {
                    Tensor qHeadsForAttn = null;
                    try
                    {
                        attnOutput = new Tensor(_allocator, DType.Float32, seqLen, numHeads * headDim);
                        qHeadsForAttn = ReshapeToHeads(qTensor, numHeads, seqLen, headDim);
                        if (_mlxAttentionCache[layer].TryAttentionHeadDim256(
                            attnOutput,
                            qHeadsForAttn,
                            kHeads,
                            vHeads,
                            numHeads,
                            numKVHeads,
                            seqLen,
                            startPos,
                            causal: true))
                        {
                            usedFusedAttn = true;
                            usedMlxAttentionCache = true;
                            qTensor.Dispose();
                            kTensor.Dispose();
                            vTensor.Dispose();
                        }
                        else
                        {
                            attnOutput.Dispose();
                            attnOutput = null;
                        }
                    }
                    finally
                    {
                        qHeadsForAttn?.Dispose();
                    }
                }

                bool tryMlxPrefillAttention = _backend == BackendType.Mlx
                    && !usedFusedAttn
                    && (headDim <= 128
                        || string.Equals(Environment.GetEnvironmentVariable("TS_MLX_CHUNKED_VECTOR_PREFILL"), "1", StringComparison.Ordinal));
                if (tryMlxPrefillAttention)
                {
                    Tensor qHeadsForAttn = null;
                    try
                    {
                        attnOutput = new Tensor(_allocator, DType.Float32, seqLen, numHeads * headDim);
                        qHeadsForAttn = ReshapeToHeads(qTensor, numHeads, seqLen, headDim);
                        if (MlxFusedOps.TryPrefillAttention(
                            attnOutput, qHeadsForAttn, kHeads, vHeads,
                            numHeads, numKVHeads, headDim,
                            seqLen, totalSeqLen,
                            startPos, 0, attentionScale))
                        {
                            usedFusedAttn = true;
                            qTensor.Dispose();
                            kTensor.Dispose();
                            vTensor.Dispose();
                        }
                        else
                        {
                            attnOutput.Dispose();
                            attnOutput = null;
                        }
                    }
                    finally
                    {
                        qHeadsForAttn?.Dispose();
                    }
                }

                // Write to KV cache after the MLX prefill-attention attempt. The
                // default MLX cache copy uses host pointers for short prompts; doing
                // that before attention dirties K/V and forces a re-upload.
                if (!usedMlxAttentionCache)
                {
                    CopyToCache(_kvCacheK[layer], kHeads, startPos, seqLen);
                    CopyToCache(_kvCacheV[layer], vHeads, startPos, seqLen);
                }

                if (!usedFusedAttn && IsGgmlBackend && canFuseContinuation)
                {
                    try
                    {
                        attnOutput = new Tensor(_allocator, DType.Float32, seqLen, numHeads * headDim);

                        if (startPos == 0)
                        {
                            // Initial prefill: pass flat Q/K/V directly (inputFormat=1).
                            // The GPU kernel does reshape+permute as free graph ops,
                            // eliminating the ReshapeToHeads copies for Q.
                            GgmlBasicOps.FusedPrefillAttention(
                                qTensor, kTensor, vTensor, attnOutput,
                                numHeads, numKVHeads, headDim,
                                seqLen, seqLen,
                                0, 0, attentionScale, inputFormat: 1);
                        }
                        else
                        {
                            // Continuation: KV cache is head-first format (inputFormat=0),
                            // but Q is flat (inputFormat=1). Use head-first path with cache.
                            Tensor qHeadsForAttn = ReshapeToHeads(qTensor, numHeads, seqLen, headDim);
                            GgmlBasicOps.FusedPrefillAttention(
                                qHeadsForAttn, _kvCacheK[layer], _kvCacheV[layer], attnOutput,
                                numHeads, numKVHeads, headDim,
                                seqLen, totalSeqLen,
                                startPos, 0, attentionScale, inputFormat: 0);
                            qHeadsForAttn.Dispose();
                        }
                        usedFusedAttn = true;
                        qTensor.Dispose();
                        kTensor.Dispose();
                        vTensor.Dispose();
                    }
                    catch
                    {
                        attnOutput?.Dispose();
                        attnOutput = null;
                    }
                }
                // Direct-CUDA fused prefill attention for the speculative verify
                // window (and any multi-row continuation). The legacy fallback
                // below materializes the whole expanded KV cache and runs
                // QK^T / mask / softmax / *V as separate dispatches -- the
                // dominant attention cost during MTP verify. The fused kernel
                // reads the head-first KV cache in place (F16 or F32). Gated to
                // text continuations (no per-axis MRoPE staged); on any miss it
                // returns false and we fall through to the legacy path.
                //
                // There is deliberately no kvLen cap here: the flash-tiled
                // kernel handles arbitrary kvLen, and the legacy fallback CANNOT
                // handle long contexts at all -- its [heads, seqLen, kvLen]
                // score tensor overflows the int element count past ~2G entries
                // (a 15K-token prompt is ~7G), which silently drops to the CPU
                // path and faults. TryGqaPrefillAttentionWithSinks declines the
                // shapes its flash path cannot serve.
                if (!usedFusedAttn && _backend == BackendType.Cuda && !useMRoPE)
                {
                    int cacheSize = (int)_kvCacheK[layer].Sizes[1];
                    var cudaPrefill = new Tensor(_allocator, DType.Float32, seqLen, numHeads * headDim);
                    Tensor qHeadsCuda = ReshapeToHeads(qTensor, numHeads, seqLen, headDim);
                    bool okCuda;
                    try
                    {
                        okCuda = CudaFusedOps.TryGqaPrefillAttentionWithSinks(
                            cudaPrefill, qHeadsCuda, _kvCacheK[layer], _kvCacheV[layer],
                            sinks: null,
                            numQHeads: numHeads, numKVHeads: numKVHeads, headDim: headDim,
                            seqLen: seqLen, kvLen: totalSeqLen, cacheSize: cacheSize,
                            maskStart: startPos, windowSize: 0, scale: attentionScale);
                    }
                    catch (Exception e)
                    {
                        okCuda = false;
                        if (_cudaGqaPrefillExWarned.Add(e.GetType().Name))
                            Console.Error.WriteLine(
                                $"[qwen35] CUDA fused GQA prefill attention threw {e.GetType().Name} ({e.Message}); " +
                                "using the legacy expanded-KV attention instead (slower, and it cannot serve " +
                                "long contexts). Reported once per exception type.");
                    }
                    qHeadsCuda.Dispose();
                    if (okCuda)
                    {
                        attnOutput = cudaPrefill;
                        usedFusedAttn = true;
                        qTensor.Dispose();
                        kTensor.Dispose();
                        vTensor.Dispose();
                    }
                    else
                    {
                        cudaPrefill.Dispose();
                    }
                }
                kHeads.Dispose();
                vHeads.Dispose();

                if (!usedFusedAttn)
                {
                    Tensor qHeads = ReshapeToHeads(qTensor, numHeads, seqLen, headDim);
                    qTensor.Dispose();
                    kTensor.Dispose();
                    vTensor.Dispose();

                    int groupSize = numHeads / numKVHeads;
                    Tensor kExpanded = ExpandKVHeads(_kvCacheK[layer], groupSize, totalSeqLen);
                    Tensor vExpanded = ExpandKVHeads(_kvCacheV[layer], groupSize, totalSeqLen);

                    using var kT = kExpanded.Transpose(1, 2);

                    // The score matrix is [numHeads, seqLen, totalSeqLen]. For a
                    // long prompt that overflows what the elementwise ops can
                    // address (they cap at int.MaxValue ELEMENTS and otherwise
                    // silently fall back to a CPU path that then faults), so
                    // split the QUERY rows into chunks small enough to stay
                    // inside the cap. Each chunk attends over the same full KV,
                    // with its causal mask shifted by the chunk offset, so the
                    // result is identical to the unchunked form; short prompts
                    // take a single chunk and behave exactly as before.
                    const long MaxScoreElements = 256L * 1024 * 1024;
                    long scoreElems = (long)numHeads * seqLen * totalSeqLen;
                    int qChunk = seqLen;
                    if (scoreElems > MaxScoreElements)
                    {
                        long perRow = (long)numHeads * totalSeqLen;
                        qChunk = (int)Math.Max(1, Math.Min(seqLen, MaxScoreElements / Math.Max(1, perRow)));
                    }

                    var attnOut = new Tensor(_allocator, DType.Float32, numHeads, seqLen, headDim);
                    bool chunked = qChunk < seqLen;
                    for (int q0 = 0; q0 < seqLen; q0 += qChunk)
                    {
                        int qc = Math.Min(qChunk, seqLen - q0);
                        // AddmmBatch needs contiguous operands, and a narrowed
                        // view of the [heads, seq, dim] tensors is strided, so
                        // chunks are staged through compact buffers. The
                        // unchunked case (every prompt short enough to fit the
                        // element cap) uses the original tensors directly.
                        Tensor qChunkT = qHeads;
                        if (chunked)
                        {
                            qChunkT = new Tensor(_allocator, DType.Float32, numHeads, qc, headDim);
                            using Tensor qView = qHeads.Narrow(1, q0, qc);
                            Ops.Copy(qChunkT, qView);
                        }
                        var scores = new Tensor(_allocator, DType.Float32, numHeads, qc, totalSeqLen);
                        Ops.AddmmBatch(scores, 0, scores, attentionScale, qChunkT, kT);
                        if (chunked)
                            qChunkT.Dispose();

                        // Fused causal-mask + softmax on GPU. Replaces AddCausalMask + Softmax
                        // (two separate ops) with one Metal kernel. No sinks for Qwen3.5
                        // dense attention.
                        if (IsGgmlBackend)
                        {
                            GgmlBasicOps.AttentionSoftmaxWithSinks(
                                scores, sinks: null,
                                numHeads: numHeads, seqLen: qc, kvLen: totalSeqLen,
                                maskStartPos: startPos + q0, slidingWindow: 0, scale: 1.0f);
                        }
                        else
                        {
                            Ops.AddCausalMask(scores, qc, startPos + q0, float.NegativeInfinity);
                            Ops.Softmax(scores, scores);
                        }

                        if (chunked)
                        {
                            var outChunk = new Tensor(_allocator, DType.Float32, numHeads, qc, headDim);
                            Ops.AddmmBatch(outChunk, 0, outChunk, 1.0f, scores, vExpanded);
                            using (Tensor outView = attnOut.Narrow(1, q0, qc))
                                Ops.Copy(outView, outChunk);
                            outChunk.Dispose();
                        }
                        else
                        {
                            Ops.AddmmBatch(attnOut, 0, attnOut, 1.0f, scores, vExpanded);
                        }
                        scores.Dispose();
                    }

                    qHeads?.Dispose();
                    kExpanded.Dispose();
                    vExpanded.Dispose();

                    attnOutput = ReshapeFromHeads(attnOut, numHeads, seqLen, headDim);
                    attnOut.Dispose();
                }
            }

            // Decode hot path: do the sigmoid-gated mix on CPU. The data is tiny
            // (single row, numHeads * headDim floats) so the GPU dispatch overhead
            // dominates. Eliminates one Metal command buffer per attention layer.
            if (seqLen == 1 && _backend != BackendType.Mlx && _backend != BackendType.Cuda && attnOutput != null && gateTensor != null
                && attnOutput.ElementType == DType.Float32 && gateTensor.ElementType == DType.Float32)
                ApplySigmoidGateCpu(attnOutput, gateTensor);
            else
                ApplySigmoidGate(attnOutput, gateTensor);
            if (ownsQGateBuffers)
                gateTensor.Dispose();

            _attnTicks += Stopwatch.GetTimestamp() - t0;

            bool ownsAttnOutput = !(seqLen == 1 && _attnDecodeOutBuf != null && ReferenceEquals(attnOutput, _attnDecodeOutBuf));

            // When skipOutputProj is set, return the raw attention output (post sigmoid gate)
            // so the caller can fuse output_proj + FFN into one dispatch.
            if (skipOutputProj)
            {
                Tensor rawOut = ownsAttnOutput ? attnOutput : Ops.NewContiguous(attnOutput);
                _attnTicks += Stopwatch.GetTimestamp() - t0;
                return rawOut;
            }

            // Fast path: fuse output projection with the residual add (eliminates the
            // intermediate output tensor and one GPU sync). Only valid for matching shapes.
            if (residual != null
                && _attnOutputQW[layer] != null
                && residual.DimensionCount == 2
                && attnOutput.DimensionCount == 2
                && residual.Sizes[0] == attnOutput.Sizes[0]
                && TryLinearAddInto(residual, attnOutput, _attnOutputQW[layer]))
            {
                if (ownsAttnOutput)
                    attnOutput.Dispose();
                return null;
            }

            Tensor output = LinearForwardCached(attnOutput, _attnOutputQW[layer], _attnOutputF32[layer]);
            if (ownsAttnOutput)
                attnOutput.Dispose();
            return output;
        }

        private void ApplySigmoidGate(Tensor attn, Tensor gate)
        {
            Ops.SigmoidMul(attn, attn, gate);
        }

        /// <summary>
        /// CPU SIMD sigmoid-gated mix for the single-row decode path:
        /// <c>attn[i] = attn[i] * sigmoid(gate[i])</c>. The tensors are small enough that
        /// going through the GPU just incurs Metal dispatch overhead with no compute win.
        /// </summary>
        private unsafe void ApplySigmoidGateCpu(Tensor attn, Tensor gate)
        {
            float* aPtr = GetFloatPtr(attn);
            float* gPtr = GetFloatPtr(gate);
            int n = (int)attn.ElementCount();

            int vLen = Vector<float>.Count;
            int i = 0;
            // Sigmoid via 1/(1+exp(-x)) is not directly vectorizable in System.Numerics
            // (no element-wise exp). Process scalar but keep the multiply vectorized.
            // Going scalar here is still ~5x faster than a GPU dispatch for ~4KB of data.
            for (; i < n; i++)
            {
                float g = gPtr[i];
                float sig = 1.0f / (1.0f + MathF.Exp(-g));
                aPtr[i] = aPtr[i] * sig;
            }

            InvalidateTensorDeviceCache(attn);
        }

        /// <summary>
        /// Single-pass deinterleave of the gated Q projection into separate Q and gate tensors.
        /// The fused Q projection produces [seqLen, numHeads, 2*headDim] where each head holds
        /// Q followed by gate. The original implementation called Narrow + NewContiguous twice,
        /// each of which allocates and copies. Doing the split with explicit Buffer.MemoryCopy
        /// avoids the intermediate strided views and pays a single contiguous copy per slice.
        /// </summary>
        private unsafe void DeinterleaveQGate(Tensor qFull, int seqLen, int numHeads, int headDim,
            out Tensor qTensor, out Tensor gateTensor, out bool ownsBuffers)
        {
            int dimPerHead = headDim * 2;
            int totalPerToken = numHeads * headDim;

            // GPU deinterleave (Strided View + Contiguous) keeps everything on
            // device. The CPU path below is faster per-call for tiny tensors
            // but forces 3 GetFloatPtr syncs per attention layer; on MLX decode
            // that's 180 round trips per token. Always prefer the GPU path for
            // MLX seqLen==1; for prefill keep the historical headDim==256 gate
            // (parallel CPU SIMD pays off for the larger per-token work).
            bool mlxGpuDeinterleave = _backend == BackendType.Mlx
                && qFull.Storage is MlxStorage
                && (seqLen == 1
                    || headDim == 256
                    || string.Equals(Environment.GetEnvironmentVariable("TS_MLX_QWEN_GPU_DEINTERLEAVE"), "1", StringComparison.Ordinal));
            if (mlxGpuDeinterleave)
            {
                using Tensor qGate = qFull.View(seqLen, numHeads, 2, headDim);
                using Tensor qView = qGate.Select(2, 0);
                using Tensor gateView = qGate.Select(2, 1);
                Tensor qContiguous = Ops.NewContiguous(qView);
                Tensor gateContiguous = Ops.NewContiguous(gateView);
                qTensor = qContiguous.View(seqLen, totalPerToken);
                gateTensor = gateContiguous.View(seqLen, totalPerToken);
                qContiguous.Dispose();
                gateContiguous.Dispose();
                ownsBuffers = true;
                return;
            }

            // Decode hot path: re-use the pre-allocated decode buffers'
            // STORAGE but hand back a fresh Tensor object that AddRef'd it.
            // Returning the bare _attnDecodeQBuf reference was a footgun:
            // any unguarded qTensor.Dispose() in the caller would drop the
            // storage's refcount to 0 and free the model's persistent
            // decode buffer, then the next attention layer's
            // DeinterleaveQGate would crash with a NullReferenceException
            // when GetFloatPtr returned the now-null buffer pointer (the
            // Qwen35 batched ForwardBatch hit exactly this on every
            // decode token). The CopyRef'd handle owns its own refcount
            // increment; caller can freely Dispose it and the underlying
            // storage stays alive as long as _attnDecodeQBuf does.
            if (seqLen == 1 && _attnDecodeQBuf != null && _attnDecodeGBuf != null
                && _attnDecodeQBuf.Sizes[1] == totalPerToken)
            {
                qTensor = _attnDecodeQBuf.CopyRef();
                gateTensor = _attnDecodeGBuf.CopyRef();
                ownsBuffers = true;
            }
            else
            {
                qTensor = new Tensor(_allocator, DType.Float32, seqLen, totalPerToken);
                gateTensor = new Tensor(_allocator, DType.Float32, seqLen, totalPerToken);
                ownsBuffers = true;
            }

            // Direct-CUDA: deinterleave on the device. The host path below forces a
            // full DtoH sync of the fused QKV buffer per attention layer (draining
            // the async pipeline) and re-uploads q/gate afterwards; measured as the
            // dominant per-layer stall for both prefill chunks and decode tokens.
            if (_backend == BackendType.Cuda
                && CudaFusedOps.TryDeinterleaveQGate(qTensor, gateTensor, qFull, seqLen, numHeads, headDim))
            {
                return;
            }

            float* src = GetFloatPtr(qFull);
            float* qDst = GetFloatPtr(qTensor);
            float* gDst = GetFloatPtr(gateTensor);
            if (src == null || qDst == null || gDst == null)
            {
                throw new InvalidOperationException(
                    $"DeinterleaveQGate: null storage pointer "
                    + $"(src={(IntPtr)src:x}, qDst={(IntPtr)qDst:x}, gDst={(IntPtr)gDst:x}). "
                    + $"seqLen={seqLen}, numHeads={numHeads}, headDim={headDim}. "
                    + $"qFull.Storage={qFull.Storage?.GetType().Name}, "
                    + $"qTensor.Storage={qTensor.Storage?.GetType().Name}, "
                    + $"gateTensor.Storage={gateTensor.Storage?.GetType().Name}.");
            }
            int headBytes = headDim * sizeof(float);

            if (seqLen > 1)
            {
                // Prefill path: parallelize across tokens. Each token's deinterleave
                // is independent, so this scales linearly with CPU core count.
                long srcL = (long)src, qDstL = (long)qDst, gDstL = (long)gDst;
                int capturedNumHeads = numHeads;
                int capturedDimPerHead = dimPerHead;
                int capturedTotalPerToken = totalPerToken;
                int capturedHeadDim = headDim;
                int capturedHeadBytes = headBytes;

                Parallel.For(0, seqLen, s =>
                {
                    float* srcRow = (float*)srcL + (long)s * capturedNumHeads * capturedDimPerHead;
                    float* qRow = (float*)qDstL + (long)s * capturedTotalPerToken;
                    float* gRow = (float*)gDstL + (long)s * capturedTotalPerToken;
                    for (int h = 0; h < capturedNumHeads; h++)
                    {
                        float* srcHead = srcRow + h * capturedDimPerHead;
                        Buffer.MemoryCopy(srcHead, qRow + h * capturedHeadDim, capturedHeadBytes, capturedHeadBytes);
                        Buffer.MemoryCopy(srcHead + capturedHeadDim, gRow + h * capturedHeadDim, capturedHeadBytes, capturedHeadBytes);
                    }
                });
            }
            else
            {
                float* srcRow = src;
                float* qRow = qDst;
                float* gRow = gDst;
                for (int h = 0; h < numHeads; h++)
                {
                    float* srcHead = srcRow + h * dimPerHead;
                    Buffer.MemoryCopy(srcHead, qRow + h * headDim, headBytes, headBytes);
                    Buffer.MemoryCopy(srcHead + headDim, gRow + h * headDim, headBytes, headBytes);
                }
            }

            InvalidateTensorDeviceCache(qTensor);
            InvalidateTensorDeviceCache(gateTensor);
        }

        /// <summary>
        /// FFN with pre-resolved weight references, mirroring <see cref="ModelBase.FFN"/>
        /// but skipping the dictionary lookup. SwiGLU on a fused gate+up projection.
        /// </summary>
        /// <summary>True when none of the given weights carries a sidecar
        /// per-tensor scale (fused mega-dispatches have no scale hook and must
        /// decline to the scale-aware per-op primitives).</summary>
        private static bool Unscaled(params QuantizedWeight[] ws)
        {
            foreach (var w in ws)
                if (w != null && w.Scale != 1.0f)
                    return false;
            return true;
        }

        // Per-layer per-projection sidecar scale table handed to the fused native
        // graphs (slot order = the native TSQ35_SC_* constants; 16 floats/layer).
        private float[] _projScaleTable;
        private GCHandle _projScaleHandle;

        private unsafe IntPtr ProjScalesPtr(int layer)
            => _projScaleTable == null
                ? IntPtr.Zero
                : (IntPtr)((float*)_projScaleHandle.AddrOfPinnedObject() + layer * 16);

        private void BuildProjScaleTable()
        {
            if (!HasSidecarWeightScales)
                return;
            int n = TotalLayerCount;
            var tbl = new float[n * 16];
            for (int i = 0; i < tbl.Length; i++) tbl[i] = 1.0f;
            float S(string key) => _quantWeights.TryGetValue(key, out var qw) ? qw.Scale : 1.0f;
            for (int l = 0; l < n; l++)
            {
                string p = $"blk.{l}.";
                int b = l * 16;
                tbl[b + 0] = _quantWeights.ContainsKey(p + "attn_qkv.weight") ? S(p + "attn_qkv.weight") : S(p + "attn_q.weight"); // QKV (or Q when separate)
                tbl[b + 1] = S(p + "attn_k.weight");        // K (separate only)
                tbl[b + 2] = S(p + "attn_v.weight");        // V (separate only)
                tbl[b + 3] = S(p + "attn_output.weight");   // O
                tbl[b + 4] = S(p + "attn_qkv.weight");      // GDN in-proj (same GGUF name)
                tbl[b + 5] = S(p + "attn_gate.weight");     // GDN z gate
                tbl[b + 6] = S(p + "ssm_beta.weight");
                tbl[b + 7] = S(p + "ssm_alpha.weight");
                tbl[b + 8] = S(p + "ssm_out.weight");
                tbl[b + 9] = S(p + "ffn_gate_up.weight");   // fused gate_up (gate==up scale)
                tbl[b + 10] = S(p + "ffn_gate.weight");     // split fallback
                tbl[b + 11] = S(p + "ffn_up.weight");
                tbl[b + 12] = S(p + "ffn_down.weight");
            }
            _projScaleTable = tbl;
            _projScaleHandle = GCHandle.Alloc(tbl, GCHandleType.Pinned);
        }

        private Tensor FFNCached(Tensor input, int layer, int seqLen)
        {
            int intermSize = Config.IntermediateSize;
            if (_ffnGateUpQW[layer] == null && _ffnGateUpF32[layer] == null)
                return FFNCachedSplitGateUp(input, layer, seqLen);
            Tensor gateUp = LinearForwardCached(input, _ffnGateUpQW[layer], _ffnGateUpF32[layer]);
            int halfDim = intermSize > 0 ? intermSize : (int)(gateUp.Sizes[1] / 2);

            Tensor gate, up;
            if (seqLen == 1)
            {
                gate = gateUp.Narrow(1, 0, halfDim);
                up = gateUp.Narrow(1, halfDim, halfDim);
            }
            else
            {
                using (var gView = gateUp.Narrow(1, 0, halfDim))
                    gate = Ops.NewContiguous(gView);
                using (var uView = gateUp.Narrow(1, halfDim, halfDim))
                    up = Ops.NewContiguous(uView);
            }
            gateUp.Dispose();

            Ops.SiLUMul(gate, gate, up);
            up.Dispose();

            Tensor down = LinearForwardCached(gate, _ffnDownQW[layer], _ffnDownF32[layer]);
            gate.Dispose();
            return down;
        }

        /// <summary>
        /// Dense FFN with fused post-attn-norm + gate_up projection AND fused down + residual add.
        /// Reduces 4 GPU dispatches (norm, gateUp, down, add) to 2, plus the SiLU*Mul activation.
        /// Returns null when the down+residual fusion succeeded (residual already updated in-place);
        /// otherwise returns the down output for the caller to add manually.
        /// </summary>
        private Tensor FFNCachedFused(Tensor residual, Tensor postNormW, int layer, int seqLen)
        {
            int intermSize = Config.IntermediateSize;

            // Prefill fast path: collapse the entire dense SwiGLU FFN
            //   residual += down_W^T @ ( silu(gate_part) * up_part )
            // into a single GGML graph dispatch. Replaces 3 separate dispatches
            // (FusedRmsNormMatMulQuant + SiLUMulSplit + FusedMatMulQuantAdd)
            // with one, removing the matched graph builds, allocations and
            // host<->backend syncs. This is the dominant prefill cost on the
            // hybrid GatedDeltaNet+Attention Qwen35 architecture. Set
            // QWEN35_DISABLE_FUSED_FFN=1 to bypass the fast path for A/B
            // comparison or debugging.
            if (seqLen > 1
                && IsGgmlBackend
                && _useFusedFfnPrefill
                && postNormW != null
                && _ffnGateUpQW[layer] != null
                && _ffnDownQW[layer] != null
                && _ffnGateUpQW[layer].Scale == 1.0f
                && _ffnDownQW[layer].Scale == 1.0f   // sidecar scales: use the scale-aware unfused chain
                && residual.DimensionCount == 2
                && residual.Sizes[0] == seqLen
                && residual.Sizes[1] == _ffnGateUpQW[layer].Ne0)
            {
                int halfDimFused = intermSize > 0 ? intermSize : (int)(_ffnGateUpQW[layer].Ne1 / 2);
                if (halfDimFused > 0 && _ffnGateUpQW[layer].Ne1 == 2L * halfDimFused
                    && _ffnDownQW[layer].Ne0 == halfDimFused
                    && _ffnDownQW[layer].Ne1 == residual.Sizes[1])
                {
                    long t0 = Stopwatch.GetTimestamp();
                    try
                    {
                        GgmlBasicOps.FusedFFNSwiGLUQuant(residual, residual, postNormW, Config.Eps,
                            _ffnGateUpQW[layer].CacheKey, _ffnGateUpQW[layer].GgmlType,
                            _ffnGateUpQW[layer].Ne0, _ffnGateUpQW[layer].Ne1, _ffnGateUpQW[layer].RawBytes,
                            _ffnDownQW[layer].CacheKey, _ffnDownQW[layer].GgmlType,
                            _ffnDownQW[layer].Ne0, _ffnDownQW[layer].Ne1, _ffnDownQW[layer].RawBytes,
                            halfDimFused);
                        _linearTicks += Stopwatch.GetTimestamp() - t0;
                        return null;
                    }
                    catch (InvalidOperationException ex)
                    {
                        // The native op already row-slabs and retries on device OOM, so a
                        // throw here means even a single row would not fit. Degrade to the
                        // unfused chain below (three smaller dispatches, ~2.5x lower peak)
                        // instead of killing the turn - the sibling FusedOutProjFFN call in
                        // RecurrentBlock has always had this guard, and its absence here is
                        // what turned a tight-VRAM multimodal prefill into a failed request.
                        _linearTicks += Stopwatch.GetTimestamp() - t0;
                        WarnFusedFfnDeclinedOnce(ex.Message);
                    }
                }
            }

            // Fused norm + gate_up projection. Decode reuses a pre-allocated [1, 2*intermSize]
            // buffer to avoid one tensor allocation per layer per token.
            Tensor gateUp = null;
            if (_ffnGateUpQW[layer] == null && _ffnGateUpF32[layer] == null)
            {
                // Unfused mixed-quant gate/up: norm once, then two matmuls.
                Tensor splitNormed = RMSNormOpCached(residual, postNormW);
                Tensor splitGate = FFNCachedSplitGateUp(splitNormed, layer, seqLen);
                splitNormed.Dispose();
                return splitGate;
            }

            bool ownsGateUp = true;
            if (postNormW != null && _ffnGateUpQW[layer] != null && IsGgmlBackend)
            {
                if (seqLen == 1 && _ffnDecodeGateUpBuf != null
                    && _ffnDecodeGateUpBuf.Sizes[1] == _ffnGateUpQW[layer].Ne1)
                {
                    gateUp = TryFusedNormLinearInto(_ffnDecodeGateUpBuf, residual, postNormW, _ffnGateUpQW[layer]);
                    if (gateUp != null)
                        ownsGateUp = false;
                }
                if (gateUp == null)
                    gateUp = FusedNormLinear(residual, postNormW, _ffnGateUpQW[layer], _ffnGateUpF32[layer]);
            }

            if (gateUp == null)
            {
                // Fallback: explicit norm then gate_up.
                Tensor normed = RMSNormOpCached(residual, postNormW);
                gateUp = LinearForwardCached(normed, _ffnGateUpQW[layer], _ffnGateUpF32[layer]);
                normed.Dispose();
            }

            int halfDim = intermSize > 0 ? intermSize : (int)(gateUp.Sizes[1] / 2);

            Tensor gate;
            if (seqLen == 1)
            {
                Tensor gView = gateUp.Narrow(1, 0, halfDim);
                Tensor uView = gateUp.Narrow(1, halfDim, halfDim);
                if (IsGgmlBackend)
                    SiLUMulInPlaceCpu(gView, uView);
                else
                    Ops.SiLUMul(gView, gView, uView);
                uView.Dispose();
                gate = gView;
            }
            else if ((IsGgmlBackend || _backend == BackendType.Mlx || _backend == BackendType.Cuda) && ownsGateUp)
            {
                // Fused split path: silu(gate_up[:, :H]) * gate_up[:, H:] without the
                // two large contiguous half-copies used by the legacy split path.
                gate = new Tensor(gateUp.Allocator, DType.Float32, gateUp.Sizes[0], halfDim);
                Ops.SiLUMulSplit(gate, gateUp, halfDim);
            }
            else
            {
                using (var gView = gateUp.Narrow(1, 0, halfDim))
                    gate = Ops.NewContiguous(gView);
                using (var uView = gateUp.Narrow(1, halfDim, halfDim))
                {
                    Ops.SiLUMul(gate, gate, uView);
                }
            }
            if (ownsGateUp)
                gateUp.Dispose();

            // Fused down + residual add.
            if (residual.DimensionCount == 2
                && gate.DimensionCount == 2
                && residual.Sizes[0] == gate.Sizes[0]
                && _ffnDownQW[layer] != null
                && TryLinearAddInto(residual, gate, _ffnDownQW[layer]))
            {
                gate.Dispose();
                return null;
            }

            Tensor down = LinearForwardCached(gate, _ffnDownQW[layer], _ffnDownF32[layer]);
            gate.Dispose();
            return down;
        }

        /// <summary>
        /// silu(gate(x)) * up(x) -> down for a layer whose gate and up could not be
        /// fused (different GGML types, neither requantizable without an importance
        /// matrix). Same arithmetic as the fused path, one extra matmul.
        /// </summary>
        private Tensor FFNCachedSplitGateUp(Tensor input, int layer, int seqLen)
        {
            Tensor gate = LinearForwardCached(input, _ffnGateSplitQW[layer], _ffnGateSplitF32[layer]);
            Tensor up = LinearForwardCached(input, _ffnUpSplitQW[layer], _ffnUpSplitF32[layer]);
            Ops.SiLUMul(gate, gate, up);
            up.Dispose();
            Tensor down = LinearForwardCached(gate, _ffnDownQW[layer], _ffnDownF32[layer]);
            gate.Dispose();
            return down;
        }

        /// <summary>True when layer <paramref name="l"/> has a usable dense FFN,
        /// fused or split.</summary>
        private bool HasDenseFfnWeights(int l)
            => (_ffnGateUpQW[l] != null || _ffnGateUpF32[l] != null)
               || ((_ffnGateSplitQW[l] != null || _ffnGateSplitF32[l] != null)
                   && (_ffnUpSplitQW[l] != null || _ffnUpSplitF32[l] != null));

        /// <summary>
        /// RMSNorm with a pre-resolved alpha tensor, avoiding the dictionary lookup that
        /// <see cref="ModelBase.RMSNormOp"/> performs per call. The arithmetic is identical.
        /// </summary>
        private Tensor RMSNormOpCached(Tensor input, Tensor alpha)
        {
            long t0 = Stopwatch.GetTimestamp();
            int rows = (int)input.Sizes[0];
            int dim = (int)(input.ElementCount() / rows);

            Tensor input2d = input.Sizes.Length != 2 ? input.View(rows, dim) : null;
            Tensor src = input2d ?? input;

            Tensor result = Ops.RMSNorm(null, src, alpha, null, Config.Eps);

            input2d?.Dispose();
            _normTicks += Stopwatch.GetTimestamp() - t0;
            return result;
        }

        /// <summary>
        /// Fused RMSNorm + quantized matmul in a single GGML kernel dispatch.
        /// Equivalent to: matmul(rms_norm(input, normW, eps), qW), but reduces 2 dispatches to 1
        /// and skips materialising the intermediate normalized tensor on the GPU.
        /// Falls back to the unfused path when the GGML backend or quant weight is unavailable.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Tensor FusedNormLinear(Tensor input, Tensor normW, QuantizedWeight qw, Tensor wF32)
        {
            // Fused path: needs GGML backend, a quantized weight, and a 2D input view.
            if (IsGgmlBackend && qw != null && normW != null && input.DimensionCount == 2)
            {
                long t0 = Stopwatch.GetTimestamp();
                int seqLen = (int)input.Sizes[0];
                int outDim = (int)qw.Ne1;
                Tensor result = new Tensor(_allocator, DType.Float32, seqLen, outDim);
                GgmlBasicOps.FusedRmsNormMatMulQuant(result, input, normW, Config.Eps,
                    qw.CacheKey, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes);
                if (qw.Scale != 1.0f)
                    Ops.Mul(result, result, qw.Scale); // sidecar per-tensor scale2
                _linearTicks += Stopwatch.GetTimestamp() - t0;
                return result;
            }

            if (_backend == BackendType.Mlx && qw != null && normW != null && input.DimensionCount == 2 && qw.Scale == 1.0f)
            {
                long t0 = Stopwatch.GetTimestamp();
                int seqLen = (int)input.Sizes[0];
                int outDim = (int)qw.Ne1;
                Tensor result = new Tensor(_allocator, DType.Float32, seqLen, outDim);
                if (MlxQuantizedOps.TryRmsNormAddmmQuantizedToFloat32(
                    result,
                    input,
                    normW,
                    Config.Eps,
                    qw.EnsureDeviceCacheKey(),
                    qw.Data,
                    qw.GgmlType,
                    qw.Ne0,
                    qw.Ne1,
                    qw.RawBytes))
                {
                    _linearTicks += Stopwatch.GetTimestamp() - t0;
                    return result;
                }

                result.Dispose();
            }

            // Fallback: explicit norm + linear.
            Tensor normed = RMSNormOpCached(input, normW);
            Tensor projected = LinearForwardCached(normed, qw, wF32);
            normed.Dispose();
            return projected;
        }

        /// <summary>
        /// Variant of <see cref="FusedNormLinear"/> that writes into a caller-supplied
        /// pre-allocated output buffer instead of allocating a new tensor each call.
        /// Returns the result buffer if the fused fast path executed, or null when the
        /// fallback path (explicit norm + linear) had to be used; in that case the
        /// caller will see the standard <see cref="FusedNormLinear"/> result.
        /// </summary>
        private Tensor TryFusedNormLinearInto(Tensor output, Tensor input, Tensor normW, QuantizedWeight qw)
        {
            if (qw == null || normW == null
                || input.DimensionCount != 2 || output == null
                || output.DimensionCount != 2 || output.Sizes[1] != qw.Ne1
                || output.Sizes[0] != input.Sizes[0])
                return null;

            if (qw.Scale != 1.0f)
                return null; // scale-aware fallback (FusedNormLinear) applies it

            if (_backend == BackendType.Mlx)
            {
                long tMlx = Stopwatch.GetTimestamp();
                if (MlxQuantizedOps.TryRmsNormAddmmQuantizedToFloat32(
                    output,
                    input,
                    normW,
                    Config.Eps,
                    qw.EnsureDeviceCacheKey(),
                    qw.Data,
                    qw.GgmlType,
                    qw.Ne0,
                    qw.Ne1,
                    qw.RawBytes))
                {
                    _linearTicks += Stopwatch.GetTimestamp() - tMlx;
                    return output;
                }

                return null;
            }

            if (!IsGgmlBackend)
                return null;

            long t0 = Stopwatch.GetTimestamp();
            GgmlBasicOps.FusedRmsNormMatMulQuant(output, input, normW, Config.Eps,
                qw.CacheKey, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes);
            _linearTicks += Stopwatch.GetTimestamp() - t0;
            return output;
        }

        /// <summary>
        /// Fused output projection + residual add: residual += matmul(input, qW) in one dispatch.
        /// Equivalent to LinearForwardCached + Ops.Add but avoids the intermediate output tensor
        /// and one GPU sync. Returns true if the fused path executed; the caller must do its own
        /// add otherwise.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool TryLinearAddInto(Tensor residual, Tensor input, QuantizedWeight qw)
        {
            if (qw == null || input.DimensionCount != 2 || residual.DimensionCount != 2)
                return false;

            if (qw.Scale != 1.0f)
                return false; // the scale must apply before the residual add

            if (_backend == BackendType.Mlx)
            {
                long tMlx = Stopwatch.GetTimestamp();
                bool ok = MlxQuantizedOps.TryAddmmQuantizedAddToFloat32(
                    residual,
                    input,
                    qw.EnsureDeviceCacheKey(),
                    qw.Data,
                    qw.GgmlType,
                    qw.Ne0,
                    qw.Ne1,
                    qw.RawBytes);
                if (ok)
                    _linearTicks += Stopwatch.GetTimestamp() - tMlx;
                return ok;
            }

            if (!IsGgmlBackend)
                return false;

            long t0 = Stopwatch.GetTimestamp();
            GgmlBasicOps.FusedMatMulQuantAdd(residual, input,
                qw.CacheKey, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes);
            _linearTicks += Stopwatch.GetTimestamp() - t0;
            return true;
        }

        /// <summary>
        /// Single-token flash attention decode. Combines the KV cache append and the
        /// scaled-dot-product attention into a single GGML graph that runs on the device,
        /// replacing the host-side <see cref="ModelBase.AttentionDecodePureCS"/> path. The
        /// kernel writes the new K/V vectors directly into the persistent KV cache (zero-copy
        /// host-pointer binding) and produces the attention output for the new query.
        ///
        /// Returns true if the device kernel was used; false to indicate the caller should
        /// fall back to the CPU SIMD attention path. Only safe for the GGML backend.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool TryFlashAttnDecode(Tensor q, Tensor k, Tensor v,
            Tensor kCache, Tensor vCache, Tensor output,
            int numHeads, int numKVHeads, int headDim,
            int maxSeqLen, int position, float scale)
        {
            if (!IsGgmlBackend || q == null || k == null || v == null || kCache == null || vCache == null || output == null)
                return false;
            if (q.ElementType != DType.Float32 || k.ElementType != DType.Float32 || v.ElementType != DType.Float32 ||
                output.ElementType != DType.Float32)
                return false;
            if (!IsFusedGraphKvCacheDType(kCache.ElementType))
                return false;
            if (vCache.ElementType != kCache.ElementType)
                return false;

            try
            {
                GgmlBasicOps.FlashAttnDecode(q, k, v, kCache, vCache, output,
                    numHeads, numKVHeads, headDim, maxSeqLen, position, scale);

                // The kernel writes the output through the host pointer (unified memory on
                // Apple Silicon), but downstream GGML ops still need to know the buffer is
                // host-fresh so any cached device mirror is reloaded.
                InvalidateTensorDeviceCache(output);
                _kvCacheHostDirty = true;
                return true;
            }
            catch (Exception)
            {
                return false;
            }
        }

        /// <summary>
        /// Single-token Qwen3.5 attention layer decode in one fused GGML graph. Performs the
        /// entire FullAttention block (input RMSNorm, fused QKV, deinterleave Q/gate, per-head
        /// QK norm, RoPE, KV cache append, flash attention, sigmoid-gated mix, output projection,
        /// residual add) in a single dispatch. Eliminates ~2 standalone GGML calls plus the
        /// CPU/GPU sync overhead between the QKV and output kernels.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        /// <summary>
        /// Fused per-layer Qwen3.5 attention prefill. Folds the entire attention
        /// block (input RMSNorm + fused QKV + Q/K norm + RoPE + KV-cache append
        /// + causal-masked softmax + attention + sigmoid-gated mix + output
        /// projection + residual add) into ONE ggml_cgraph dispatch per layer,
        /// writing the residual back into the caller's `hidden` buffer in place.
        ///
        /// Returns false (and does NOT touch `hidden`) when:
        ///  - any required quantized weight isn't loaded for this layer
        ///    (the kernel currently requires the QKV / O quantized CacheKey),
        ///  - the layer is a gated-delta-net layer (not handled here),
        ///  - the KV cache dtype isn't F32 / F16, or
        ///  - the layer is shared (KV donor mapping isn't handled here yet).
        /// Caller falls back to the legacy multi-dispatch FullAttention path.
        /// </summary>
        private unsafe bool TryFusedAttnLayerPrefill(Tensor hidden, int layer, int seqLen, int startPos)
        {
            if (!IsGgmlBackend) return false;
            if (hidden == null || hidden.DimensionCount != 2 || hidden.ElementType != DType.Float32)
                return false;

            // Skip gated-delta-net (recurrent) layers ÔÇö they go through their
            // own kernel path entirely.
            if (_isRecurrent != null && _isRecurrent[layer]) return false;

            // Fused kernels bake scalar RoPE into the graph. Per-axis MRoPE
            // angles can't be expressed without a kernel update, so when
            // multimodal positions are pending fall back to the legacy
            // multi-dispatch path which routes through ApplyMRoPEPrefill.
            if (_pendingMRoPEPositions != null) return false;

            QuantizedWeight qkv = _attnQkvQW[layer];
            QuantizedWeight oOut = _attnOutputQW[layer];
            if ((qkv?.Scale ?? 1.0f) != 1.0f || (oOut?.Scale ?? 1.0f) != 1.0f)
                return false; // sidecar scale2 not wired into this per-layer graph
            Tensor attnNorm = _attnNormW[layer];
            Tensor qNorm = _attnQNormW[layer];
            Tensor kNorm = _attnKNormW[layer];
            Tensor kCache = _kvCacheK[layer];
            Tensor vCache = _kvCacheV[layer];

            if (qkv == null || oOut == null || attnNorm == null || qNorm == null || kNorm == null
                || kCache == null || vCache == null)
                return false;
            if (!IsFusedGraphKvCacheDType(kCache.ElementType))
                return false;
            if (vCache.ElementType != kCache.ElementType)
                return false;

            int headDim = Config.HeadDim;
            int numHeads = Config.NumHeads;
            int numKVHeads = Config.NumKVHeads;
            int maxSeqLen = (int)kCache.Sizes[1];
            int ropeDims = _ropeDimCount > 0 ? _ropeDimCount : headDim;
            const int ropeMode = 2; // NeoX, matches the legacy ApplyRoPEPrefill
            float ropeFreqScale = 1.0f / Config.RopeScale;

            int kvCacheTypeId = FusedGraphKvCacheTypeId(kCache.ElementType);

            try
            {
                long t0 = Stopwatch.GetTimestamp();
                GgmlBasicOps.Qwen35AttentionLayerPrefill(
                    (IntPtr)GetFloatPtr(hidden), Config.HiddenSize, seqLen,
                    (IntPtr)GetFloatPtr(attnNorm),
                    qkv.CacheKey, qkv.GgmlType, qkv.Ne0, qkv.Ne1, qkv.RawBytes,
                    (IntPtr)GetFloatPtr(qNorm), (IntPtr)GetFloatPtr(kNorm),
                    oOut.CacheKey, oOut.GgmlType, oOut.Ne0, oOut.Ne1, oOut.RawBytes,
                    TensorComputePrimitives.GetStoragePointer(kCache),
                    TensorComputePrimitives.GetStoragePointer(vCache),
                    numHeads, numKVHeads, headDim,
                    maxSeqLen, startPos,
                    Config.RopeBase, ropeFreqScale, ropeDims, ropeMode,
                    kvCacheTypeId, Config.Eps);
                _attnTicks += Stopwatch.GetTimestamp() - t0;

                InvalidateTensorDeviceCache(hidden);
                InvalidateTensorDeviceCache(kCache);
                InvalidateTensorDeviceCache(vCache);
                return true;
            }
            catch
            {
                return false;
            }
        }

        private bool TryFusedAttnLayerDecode(Tensor residual, int layer, int position)
        {
            if (!IsGgmlBackend)
                return false;
            if (residual == null || residual.DimensionCount != 2 || residual.ElementType != DType.Float32)
                return false;
            if (residual.Sizes[0] != 1)
                return false;

            QuantizedWeight qkv = _attnQkvQW[layer];
            QuantizedWeight oOut = _attnOutputQW[layer];
            if ((qkv?.Scale ?? 1.0f) != 1.0f || (oOut?.Scale ?? 1.0f) != 1.0f)
                return false; // sidecar scale2 not wired into this per-layer graph
            Tensor attnNorm = _attnNormW[layer];
            Tensor qNorm = _attnQNormW[layer];
            Tensor kNorm = _attnKNormW[layer];
            Tensor kCache = _kvCacheK[layer];
            Tensor vCache = _kvCacheV[layer];

            if (qkv == null || oOut == null || attnNorm == null || qNorm == null || kNorm == null
                || kCache == null || vCache == null)
                return false;
            if (!IsFusedGraphKvCacheDType(kCache.ElementType))
                return false;
            if (vCache.ElementType != kCache.ElementType)
                return false;

            int headDim = Config.HeadDim;
            int numHeads = Config.NumHeads;
            int numKVHeads = Config.NumKVHeads;
            int maxSeqLen = (int)kCache.Sizes[1];

            // The fused kernel uses NeoX RoPE (rope_mode = 2), matching the standalone path.
            // Partial rotary (rope.dimension_count, e.g. 64 of the 256-dim head).
            // Must match the whole-model decode graph AND the KV rows prefill
            // wrote; roping all 256 dims here rotates a different subspace.
            int ropeDims = _ropeDimCount > 0 ? _ropeDimCount : headDim;
            const int ropeMode = 2;
            float ropeFreqScale = 1.0f / Config.RopeScale;

            try
            {
                long t0 = Stopwatch.GetTimestamp();
                GgmlBasicOps.Qwen35AttentionLayerDecode(
                    residual,
                    attnNorm,
                    qkv.CacheKey, qkv.GgmlType, qkv.Ne0, qkv.Ne1, qkv.RawBytes,
                    qNorm, kNorm, headDim,
                    oOut.CacheKey, oOut.GgmlType, oOut.Ne0, oOut.Ne1, oOut.RawBytes,
                    kCache, vCache,
                    numHeads, numKVHeads,
                    maxSeqLen, position,
                    Config.Eps, Config.RopeBase, ropeFreqScale, ropeDims, ropeMode);
                _attnTicks += Stopwatch.GetTimestamp() - t0;

                // The residual is downloaded to its host buffer, but on CUDA the
                // appended K/V remains authoritative in the cacheable device copy.
                // Keep those device slabs alive and lazily download them before a
                // host snapshot/grow instead of invalidating (and losing) the write.
                InvalidateTensorDeviceCache(residual);
                _kvCacheHostDirty = true;
                return true;
            }
            catch (Exception)
            {
                return false;
            }
        }

        private Tensor ApplyQKNormCached(Tensor data, Tensor alpha, int numHeads, int seqLen)
        {
            int headDim = Config.HeadDim;

            // Decode (seqLen==1): the CPU path is faster per call for tiny
            // tensors, but on MLX every GetFloatPtr forces an mlx_eval +
            // deviceÔåÆhost copy. With 4 syncs/attn-layer ├ù 60 layers that's a
            // lot of round trips, so stay on device for MLX. Other backends
            // keep the CPU SIMD fast path which saves 2 GPU dispatches.
            // CPU SIMD norm avoids a GPU dispatch for tiny decode tensors, but on
            // backends where reading the data to host forces a pipeline-draining
            // sync (MLX mlx_eval, direct CUDA cuMemcpyDtoH) the round trip costs far
            // more than the dispatch. Keep those on the device.
            if (seqLen == 1 && _backend != BackendType.Mlx && _backend != BackendType.Cuda)
            {
                RMSNormInPlaceCpu(data, alpha, numHeads, headDim, Config.Eps);
                return data;
            }

            // In-place RMSNorm: row-independent normalization works regardless
            // of row order, avoiding one tensor allocation+copy per Q/K per layer.
            using var reshaped = data.View(seqLen * numHeads, headDim);
            Ops.RMSNorm(reshaped, reshaped, alpha, null, Config.Eps);
            return data;
        }

        /// <summary>
        /// Fused decode-path RoPE for Q and K at the same position. Computes the
        /// cos/sin table once (since both share <paramref name="position"/>) and applies
        /// the rotation in place to both tensors. The inner loop is SIMD-vectorized
        /// when <c>halfDim</c> is divisible by the hardware vector width, with a scalar
        /// tail for the remainder.
        /// </summary>
        private unsafe void ApplyRoPEDecodeQKInPlace(Tensor qData, Tensor kData,
            int numQHeads, int numKHeads, int position)
        {
            int headDim = Config.HeadDim;
            int ropeDim = _ropeDimCount > 0 ? _ropeDimCount : headDim;
            int halfDim = ropeDim / 2;

            float* cosTable = stackalloc float[halfDim];
            float* sinTable = stackalloc float[halfDim];
            for (int i = 0; i < halfDim; i++)
            {
                float theta = position * _ropeFreqs[i];
                cosTable[i] = MathF.Cos(theta);
                sinTable[i] = MathF.Sin(theta);
            }

            float* qPtr = GetFloatPtr(qData);
            float* kPtr = GetFloatPtr(kData);
            ApplyRoPERotationInPlace(qPtr, numQHeads, headDim, halfDim, cosTable, sinTable);
            ApplyRoPERotationInPlace(kPtr, numKHeads, headDim, halfDim, cosTable, sinTable);

            InvalidateTensorDeviceCache(qData);
            InvalidateTensorDeviceCache(kData);
        }

        private static unsafe void ApplyRoPERotationInPlace(float* ptr, int numHeads,
            int headDim, int halfDim, float* cosTable, float* sinTable)
        {
            int vecSz = Vector<float>.Count;
            int vecEnd = halfDim - (halfDim % vecSz);

            for (int h = 0; h < numHeads; h++)
            {
                float* head = ptr + h * headDim;
                float* hi = head + halfDim;

                int i = 0;
                for (; i < vecEnd; i += vecSz)
                {
                    var x0 = LdVecLocal(head + i);
                    var x1 = LdVecLocal(hi + i);
                    var c = LdVecLocal(cosTable + i);
                    var s = LdVecLocal(sinTable + i);
                    StVecLocal(head + i, x0 * c - x1 * s);
                    StVecLocal(hi + i, x0 * s + x1 * c);
                }
                for (; i < halfDim; i++)
                {
                    float x0 = head[i];
                    float x1 = hi[i];
                    head[i] = x0 * cosTable[i] - x1 * sinTable[i];
                    hi[i] = x0 * sinTable[i] + x1 * cosTable[i];
                }
            }
        }

        private Tensor ApplyRoPEPrefill(Tensor data, int numHeads, int seqLen, int startPos)
        {
            int headDim = Config.HeadDim;
            int ropeDim = _ropeDimCount > 0 ? _ropeDimCount : headDim;

            // Cache position tensors across attention layers in the same forward pass.
            // All attention layers share (seqLen, startPos); only numHeads differs (Q vs K).
            // Under TP the position tensor is a kernel input, so it must live on
            // DATA's GPU. Only the default-allocator (rank 0 / single-GPU) copy is
            // cached; other ranks build a fresh one on their own GPU per call.
            Tensor posTensor;
            bool isQ = (numHeads == Config.NumHeads);
            var dataAlloc = data.Storage.Allocator;
            bool onDefaultAlloc = ReferenceEquals(dataAlloc, _allocator);

            if (onDefaultAlloc && _cachedRoPEPosSeqLen == seqLen && _cachedRoPEPosStartPos == startPos)
            {
                posTensor = isQ ? _cachedRoPEPosQ : _cachedRoPEPosK;
                if (posTensor == null || (int)posTensor.Sizes[0] != seqLen * numHeads)
                    posTensor = null;
            }
            else if (onDefaultAlloc)
            {
                _cachedRoPEPosQ?.Dispose();
                _cachedRoPEPosK?.Dispose();
                _cachedRoPEPosQ = null;
                _cachedRoPEPosK = null;
                _cachedRoPEPosSeqLen = seqLen;
                _cachedRoPEPosStartPos = startPos;
                posTensor = null;
            }
            else
            {
                posTensor = null; // per-rank GPU: never cached
            }

            bool freshForRank = false;
            if (posTensor == null)
            {
                int totalRows = seqLen * numHeads;
                int[] positions = new int[totalRows];
                for (int s = 0; s < seqLen; s++)
                    for (int h = 0; h < numHeads; h++)
                        positions[s * numHeads + h] = startPos + s;
                posTensor = CreateIntTensorOn(dataAlloc, positions, totalRows);
                if (onDefaultAlloc)
                {
                    if (isQ) _cachedRoPEPosQ = posTensor;
                    else _cachedRoPEPosK = posTensor;
                }
                else
                {
                    freshForRank = true;
                }
            }

            // In-place RoPE: avoids allocating a new tensor per call
            using var reshaped = data.View(1, seqLen, numHeads, headDim);
            Ops.RoPEEx(reshaped, reshaped, posTensor, ropeDim, 2, 0,
                Config.RopeBase, 1.0f / Config.RopeScale,
                0.0f, 1.0f, 0.0f, 0.0f);
            if (freshForRank)
                posTensor.Dispose();
            return data;
        }

        /// <summary>
        /// Returns a cached int32 position tensor [seqLen * numHeads] for the CUDA
        /// fused QK-RMSNorm + RoPE kernel.  Positions are laid out as
        /// [startPos, startPos, ..., startPos+1, startPos+1, ...] ÔÇö each position
        /// repeated numHeads times so row r maps to position startPos + (r / numHeads).
        /// </summary>
        private Tensor EnsureRoPEPositions(int startPos, int seqLen, int numHeads)
        {
            int totalRows = seqLen * numHeads;
            bool isQ = numHeads == Config.NumHeads;
            if (_cachedRoPEPosSeqLen == seqLen && _cachedRoPEPosStartPos == startPos)
            {
                Tensor cached = isQ ? _cachedRoPEPosQ : _cachedRoPEPosK;
                if (cached != null && (int)cached.Sizes[0] == totalRows)
                    return cached;
            }
            else
            {
                _cachedRoPEPosQ?.Dispose();
                _cachedRoPEPosK?.Dispose();
                _cachedRoPEPosQ = null;
                _cachedRoPEPosK = null;
                _cachedRoPEPosSeqLen = seqLen;
                _cachedRoPEPosStartPos = startPos;
            }
            int[] positions = new int[totalRows];
            for (int s = 0; s < seqLen; s++)
                for (int h = 0; h < numHeads; h++)
                    positions[s * numHeads + h] = startPos + s;
            // Cache the tensor (it used to be recreated per attention layer): the
            // CUDA-graph prefill capture pre-warms these two tensors so their
            // host->device upload never lands inside a captured graph.
            Tensor created = CreateIntTensor(positions, totalRows);
            if (isQ) _cachedRoPEPosQ = created;
            else _cachedRoPEPosK = created;
            return created;
        }

        /// <summary>Set the per-axis (T,H,W) positions for the next prefill
        /// call. Length must equal 3 * seqLen of the upcoming Forward(). Called
        /// by ModelMultimodalInjector.QueuePromptEmbeddingsForSlice when an
        /// image is in the prompt slice.</summary>
        public void SetMRoPEPositions(int[] flatThw)
        {
            _pendingMRoPEPositions = flatThw;
        }

        /// <summary>Build the per-pair modality assignment from `_mropeSections`
        /// using vLLM's get_mrope_interleaved_id_list algorithm (see
        /// mrope_interleaved.py:138-185). Result: int[rotary_dim/2] where
        /// each entry Ôêê {0=T, 1=H, 2=W}. Called lazily on first MRoPE-prefill.</summary>
        private void PrecomputeMRoPEInterleavedIds()
        {
            int ropeDim = _ropeDimCount > 0 ? _ropeDimCount : Config.HeadDim;
            int pairs = ropeDim / 2;
            int[] sec = _mropeSections;
            // Qwen3.5 GGUFs ship [11,11,10,0]; the trailing 0 is video-time
            // padding. Collapse to length-3 (T,H,W) and apply force_last like
            // vLLM does for len-3 sections.
            int a = sec != null && sec.Length > 0 ? sec[0] : pairs / 3;
            int b = sec != null && sec.Length > 1 ? sec[1] : pairs / 3;
            int c = sec != null && sec.Length > 2 ? sec[2] : pairs - 2 * (pairs / 3);
            if (a + b + c != pairs)
            {
                Console.WriteLine($"[qwen35-mrope] section sum {a + b + c} != rotary_dim/2 {pairs}; " +
                                  $"falling back to T-only (axis=0) - mrope_section may be wrong");
                _mropeInterleavedIds = new int[pairs]; // all zeros = all-T = standard RoPE
                return;
            }
            // force_last=true: last pair must be T (axis 0). Python's
            // mrope_interleaved.py decrements `a` BEFORE building the counts
            // dict, so the placement-fraction score uses `a-1` as the T
            // denominator (not `a`). We track that as `aTotal` below.
            int aTotal = a - 1, bTotal = b, cTotal = c;
            int aRem = aTotal, bRem = bTotal, cRem = cTotal;
            int[] ids = new int[pairs];
            int last = -1;
            for (int i = 0; i < pairs - 1; i++)
            {
                // Candidates: remaining > 0 and != last; if all candidates == last, relax.
                int bestK = -1;
                double bestScore = double.MaxValue;
                for (int k = 0; k < 3; k++)
                {
                    int rem = k == 0 ? aRem : k == 1 ? bRem : cRem;
                    if (rem <= 0) continue;
                    if (k == last) continue;
                    int total = k == 0 ? aTotal : k == 1 ? bTotal : cTotal;
                    int placed = total - rem;
                    double score = (double)placed / total;
                    if (score < bestScore || (score == bestScore && k < bestK))
                    {
                        bestScore = score;
                        bestK = k;
                    }
                }
                if (bestK < 0)
                {
                    // All candidates blocked by "!= last" - relax.
                    for (int k = 0; k < 3; k++)
                    {
                        int rem = k == 0 ? aRem : k == 1 ? bRem : cRem;
                        if (rem <= 0) continue;
                        int total = k == 0 ? aTotal : k == 1 ? bTotal : cTotal;
                        int placed = total - rem;
                        double score = (double)placed / total;
                        if (score < bestScore || (score == bestScore && k < bestK))
                        {
                            bestScore = score;
                            bestK = k;
                        }
                    }
                }
                ids[i] = bestK;
                if (bestK == 0) aRem--;
                else if (bestK == 1) bRem--;
                else cRem--;
                last = bestK;
            }
            ids[pairs - 1] = 0; // force_last
            _mropeInterleavedIds = ids;
        }

        // TS_QWEN35_MROPE_NATIVE=0 forces the managed MRoPE rotation loop on the
        // GGML backends instead of the native ggml_rope_multi kernel. A/B switch
        // for isolating rope-kernel issues from the rest of the multimodal path.
        private static readonly bool _mropeNativeEnabled =
            Environment.GetEnvironmentVariable("TS_QWEN35_MROPE_NATIVE") != "0";

        /// <summary>Per-axis (interleaved MRoPE) variant of ApplyRoPEPrefill.
        /// Q/K go in, get rotated in place by NeoX-style pair rotation where
        /// pair i uses _pendingMRoPEPositions[3*token + _mropeInterleavedIds[i]]
        /// as the position instead of a single token position. Returns the
        /// original tensor (modified). Falls back silently if positions or ids
        /// aren't set; the caller should have checked _pendingMRoPEPositions.
        ///
        /// Costs one host download + one host upload per call (Q and K each).
        /// Acceptable for prefill since it's already the slow path; not used in
        /// decode.</summary>
        private Tensor ApplyMRoPEPrefill(Tensor data, int numHeads, int seqLen, int[] mropePositions)
        {
            int headDim = Config.HeadDim;
            int ropeDim = _ropeDimCount > 0 ? _ropeDimCount : headDim;
            int pairs = ropeDim / 2;
            if (_mropeInterleavedIds == null || _mropeInterleavedIds.Length != pairs)
                PrecomputeMRoPEInterleavedIds();
            int[] modality = _mropeInterleavedIds;

            // On the GGML backend, route through the native ggml_rope_multi
            // kernel which runs entirely on-device (no per-layer host download
            // / upload that the managed path below incurs). Requires shape
            // [headDim, numHeads, seqLen, 1] in GGML order (= C# View(1, seqLen,
            // numHeads, headDim)) and positions laid out per-axis-concatenated
            // as [T0..T_n, H0..H_n, W0..W_n, 0..0] of length 4*seqLen.
            // Mode 40 = GGML_ROPE_TYPE_IMROPE; sections come straight from the
            // GGUF (Qwen3.5 ships [11,11,10,0]).
            if (data.Storage is TensorSharp.GGML.GgmlStorage && _mropeSections != null && _mropeSections.Length >= 4
                && _mropeNativeEnabled)
            {
                int[] flatThw = new int[4 * seqLen];
                for (int t = 0; t < seqLen; t++)
                {
                    flatThw[t]            = mropePositions[3 * t + 0]; // T axis
                    flatThw[seqLen + t]   = mropePositions[3 * t + 1]; // H axis
                    flatThw[2*seqLen + t] = mropePositions[3 * t + 2]; // W axis
                    // 4th axis: 0 (no video / fourth section is 0 for Qwen3.5).
                }
                using var positionsTensor = CreateIntTensor(flatThw, flatThw.Length);
                using var reshaped = data.View(1, seqLen, numHeads, headDim);
                const int GGML_ROPE_TYPE_IMROPE = 40;
                int[] sec = new int[4] {
                    _mropeSections[0], _mropeSections[1], _mropeSections[2], _mropeSections[3]
                };
                TensorSharp.GGML.GgmlBasicOps.RoPEMRoPE(
                    reshaped, reshaped, positionsTensor, sec,
                    ropeDim, GGML_ROPE_TYPE_IMROPE,
                    /*originalContextLength*/ 0,
                    Config.RopeBase, 1.0f / Config.RopeScale);
                return data;
            }

            int total = seqLen * numHeads * headDim;
            float[] buf = data.GetElementsAsFloat(total);

            // NeoX rotation: dim pair (i, i + pairs) rotates by theta.
            // Per-pair angle = positions[token, modality[i]] * freq[i].
            // freq[i] = (1/RopeScale) / RopeBase^(2i/ropeDim) (matches _ropeFreqs).
            float[] freqs = _ropeFreqs;
            int strideToken = numHeads * headDim;
            int strideHead = headDim;
            for (int t = 0; t < seqLen; t++)
            {
                int p0 = mropePositions[3 * t + 0];
                int p1 = mropePositions[3 * t + 1];
                int p2 = mropePositions[3 * t + 2];
                for (int h = 0; h < numHeads; h++)
                {
                    int baseOff = t * strideToken + h * strideHead;
                    for (int i = 0; i < pairs; i++)
                    {
                        int mod = modality[i];
                        int pos = mod == 0 ? p0 : mod == 1 ? p1 : p2;
                        float theta = pos * freqs[i];
                        float c = MathF.Cos(theta);
                        float s = MathF.Sin(theta);
                        float x0 = buf[baseOff + i];
                        float x1 = buf[baseOff + i + pairs];
                        buf[baseOff + i]         = x0 * c - x1 * s;
                        buf[baseOff + i + pairs] = x0 * s + x1 * c;
                    }
                }
            }

            // Upload back. Use the same trick as elsewhere - create a contiguous
            // fresh tensor and copy via the helper that backends understand.
            long[] shape = data.Sizes.ToArray();
            using var refreshed = CreateFloatTensor(buf, shape);
            Ops.Copy(data, refreshed);
            return data;
        }

        #endregion


        #region Mixture-of-Experts FFN

        /// <summary>
        /// Mixture-of-Experts SwiGLU FFN as used by qwen35moe / qwen3next.
        /// 1. router_logits = input @ ffn_gate_inp ; probs = softmax(router_logits)
        /// 2. select top-K experts per token, optionally renormalize selected probs
        /// 3. expert SwiGLU: SiLU(input @ gate_e) * (input @ up_e) @ down_e
        /// 4. weighted sum across selected experts
        /// 5. optional shared expert SwiGLU gated by sigmoid(input . ffn_gate_inp_shexp)
        ///
        /// Optimizations vs the original implementation:
        /// - Pre-cached `QuantizedWeight` / F32 expert tensors avoid per-token dictionary lookups.
        /// - Reuses pre-allocated decode-path tensors (`_moeTokenInput`, `_moeGateBuf`, `_moeUpBuf`,
        ///   `_moeDownBuf`) so the seqLen=1 hot path performs no per-expert tensor allocation.
        /// - Falls back to the previous per-token allocation strategy for prefill (seqLen > 1)
        ///   where reuse would constrain parallelism between token slices.
        /// </summary>
        /// <summary>
        /// Single-token MoE that fuses (a) routed expert SwiGLU, (b) optional shared expert
        /// SwiGLU, and (c) the residual add into one GGML graph dispatch via the
        /// MoEExpertsSwiGLUResidual kernel. Returns true if the fully fused path executed and
        /// `residual` already contains the updated value; false to fall back to MoEForward.
        /// </summary>
        private unsafe bool TryMoEResidualDecode(Tensor residual, Tensor input, int layer)
        {
            if (!IsGgmlBackend
                || _moeBatchedResult == null
                || _moeGatePtrs == null
                || _expertGateQW == null || _expertUpQW == null || _expertDownQW == null
                || _expertGateQW[layer] == null || _expertUpQW[layer] == null || _expertDownQW[layer] == null)
                return false;

            int hiddenSize = Config.HiddenSize;

            Tensor routerLogits = LinearForwardCached(input, _ffnGateInpQW[layer], _ffnGateInpF32[layer]);
            if (routerLogits == null)
                return false;

            // Routing: skip the full-vocab softmax dispatch and operate on the raw logits.
            // Softmax is monotonic so top-K of softmax equals top-K of logits; for the
            // normalized-top-K case (the common path) the top-K weights are softmax(top-K
            // logits), which we compute on CPU. Eliminates one Metal dispatch per MoE layer.
            float* logits = GetFloatPtr(routerLogits);
            int[] topExperts = _moeTopExperts;
            float[] routeW = _moeRouteW;
            SelectTopKInPlace(logits, _numExperts, _numExpertsUsed, topExperts);

            if (_normTopKProb)
            {
                float maxLogit = float.NegativeInfinity;
                for (int k = 0; k < _numExpertsUsed; k++)
                {
                    float v = logits[topExperts[k]];
                    if (v > maxLogit) maxLogit = v;
                }
                float wSum = 0f;
                for (int k = 0; k < _numExpertsUsed; k++)
                {
                    float w = MathF.Exp(logits[topExperts[k]] - maxLogit);
                    routeW[k] = w;
                    wSum += w;
                }
                if (wSum > 0f)
                {
                    float inv = 1.0f / wSum;
                    for (int k = 0; k < _numExpertsUsed; k++)
                        routeW[k] *= inv;
                }
            }
            else
            {
                // Unnormalised case: weights must be the absolute softmax probabilities,
                // which need the full denominator (sum over all experts). Compute it on CPU.
                float maxLogit = float.NegativeInfinity;
                for (int i = 0; i < _numExperts; i++)
                    if (logits[i] > maxLogit) maxLogit = logits[i];
                float denom = 0f;
                for (int i = 0; i < _numExperts; i++)
                    denom += MathF.Exp(logits[i] - maxLogit);
                float invDenom = denom > 0f ? 1.0f / denom : 0f;
                for (int k = 0; k < _numExpertsUsed; k++)
                    routeW[k] = MathF.Exp(logits[topExperts[k]] - maxLogit) * invDenom;
            }
            routerLogits.Dispose();

            QuantizedWeight gQW0 = null, uQW0 = null, dQW0 = null;
            for (int k = 0; k < _numExpertsUsed; k++)
            {
                int e = topExperts[k];
                var g = _expertGateQW[layer][e];
                var u = _expertUpQW[layer][e];
                var d = _expertDownQW[layer][e];
                if (g == null || u == null || d == null)
                    return false;
                if (gQW0 == null) { gQW0 = g; uQW0 = u; dQW0 = d; }

                if (g.GgmlType != gQW0.GgmlType || g.Ne0 != gQW0.Ne0 || g.Ne1 != gQW0.Ne1 ||
                    u.GgmlType != uQW0.GgmlType || u.Ne0 != uQW0.Ne0 || u.Ne1 != uQW0.Ne1 ||
                    d.GgmlType != dQW0.GgmlType || d.Ne0 != dQW0.Ne0 || d.Ne1 != dQW0.Ne1)
                    return false;

                _moeGatePtrs[k] = g.CacheKey;
                _moeUpPtrs[k] = u.CacheKey;
                _moeDownPtrs[k] = d.CacheKey;
            }

            // Optional shared expert weights & gate scalar.
            bool useShared = false;
            IntPtr sgPtr = IntPtr.Zero, suPtr = IntPtr.Zero, sdPtr = IntPtr.Zero;
            int sgType = 0, suType = 0, sdType = 0;
            long sgNe0 = 0, sgNe1 = 0, sgBytes = 0;
            long suNe0 = 0, suNe1 = 0, suBytes = 0;
            long sdNe0 = 0, sdNe1 = 0, sdBytes = 0;
            float sharedScalar = 0f;

            if (_hasSharedExperts != null && _hasSharedExperts[layer]
                && _ffnGateShexpQW[layer] != null
                && _ffnUpShexpQW[layer] != null
                && _ffnDownShexpQW[layer] != null)
            {
                var sg = _ffnGateShexpQW[layer];
                var su = _ffnUpShexpQW[layer];
                var sd = _ffnDownShexpQW[layer];
                sgPtr = sg.CacheKey; suPtr = su.CacheKey; sdPtr = sd.CacheKey;
                sgType = sg.GgmlType; suType = su.GgmlType; sdType = sd.GgmlType;
                sgNe0 = sg.Ne0; sgNe1 = sg.Ne1; sgBytes = sg.RawBytes;
                suNe0 = su.Ne0; suNe1 = su.Ne1; suBytes = su.RawBytes;
                sdNe0 = sd.Ne0; sdNe1 = sd.Ne1; sdBytes = sd.RawBytes;

                sharedScalar = 1.0f;
                if (_hasSharedExpertGate != null && _hasSharedExpertGate[layer])
                {
                    var gateInpVec = _ffnGateInpShexpVec[layer];
                    if (gateInpVec != null)
                    {
                        float* tokenRow = GetFloatPtr(input);
                        float* gateInpPtr = GetFloatPtr(gateInpVec);
                        int n = Math.Min((int)gateInpVec.ElementCount(), hiddenSize);
                        sharedScalar = SigmoidScalar(VecDot(tokenRow, gateInpPtr, n));
                    }
                }
                useShared = true;
            }

            long t0exp = Stopwatch.GetTimestamp();
            GgmlBasicOps.MoEExpertsSwiGLUResidual(
                residual, input,
                _numExpertsUsed,
                _moeGatePtrs, _moeUpPtrs, _moeDownPtrs,
                gQW0.GgmlType, gQW0.Ne0, gQW0.Ne1, gQW0.RawBytes,
                uQW0.GgmlType, uQW0.Ne0, uQW0.Ne1, uQW0.RawBytes,
                dQW0.GgmlType, dQW0.Ne0, dQW0.Ne1, dQW0.RawBytes,
                routeW,
                useShared,
                sgPtr, suPtr, sdPtr,
                sgType, sgNe0, sgNe1, sgBytes,
                suType, suNe0, suNe1, suBytes,
                sdType, sdNe0, sdNe1, sdBytes,
                sharedScalar);
            _linearTicks += Stopwatch.GetTimestamp() - t0exp;

            InvalidateTensorDeviceCache(residual);
            return true;
        }

        /// <summary>
        /// MoE decode with pre-computed router logits (from the fused outproj+norm+router kernel).
        /// Skips the router projection dispatch since logits are already available.
        /// </summary>
        private unsafe bool TryMoEResidualDecodeWithRouter(Tensor residual, Tensor input, Tensor routerLogits, int layer)
        {
            if (!IsGgmlBackend || _moeBatchedResult == null || _moeGatePtrs == null
                || _expertGateQW == null || _expertUpQW == null || _expertDownQW == null
                || _expertGateQW[layer] == null || _expertUpQW[layer] == null || _expertDownQW[layer] == null)
                return false;

            int hiddenSize = Config.HiddenSize;
            float* logits = GetFloatPtr(routerLogits);
            int[] topExperts = _moeTopExperts;
            float[] routeW = _moeRouteW;
            SelectTopKInPlace(logits, _numExperts, _numExpertsUsed, topExperts);

            if (_normTopKProb)
            {
                float maxLogit = float.NegativeInfinity;
                for (int k = 0; k < _numExpertsUsed; k++)
                    if (logits[topExperts[k]] > maxLogit) maxLogit = logits[topExperts[k]];
                float wSum = 0f;
                for (int k = 0; k < _numExpertsUsed; k++)
                {
                    float w = MathF.Exp(logits[topExperts[k]] - maxLogit);
                    routeW[k] = w; wSum += w;
                }
                if (wSum > 0f) { float inv = 1.0f / wSum; for (int k = 0; k < _numExpertsUsed; k++) routeW[k] *= inv; }
            }
            else
            {
                float maxLogit = float.NegativeInfinity;
                for (int i = 0; i < _numExperts; i++) if (logits[i] > maxLogit) maxLogit = logits[i];
                float denom = 0f;
                for (int i = 0; i < _numExperts; i++) denom += MathF.Exp(logits[i] - maxLogit);
                float invDenom = denom > 0f ? 1.0f / denom : 0f;
                for (int k = 0; k < _numExpertsUsed; k++) routeW[k] = MathF.Exp(logits[topExperts[k]] - maxLogit) * invDenom;
            }

            QuantizedWeight gQW0 = null, uQW0 = null, dQW0 = null;
            for (int k = 0; k < _numExpertsUsed; k++)
            {
                int e = topExperts[k];
                var g = _expertGateQW[layer][e]; var u = _expertUpQW[layer][e]; var d = _expertDownQW[layer][e];
                if (g == null || u == null || d == null) return false;
                if (gQW0 == null) { gQW0 = g; uQW0 = u; dQW0 = d; }
                if (g.GgmlType != gQW0.GgmlType || g.Ne0 != gQW0.Ne0 || g.Ne1 != gQW0.Ne1 ||
                    u.GgmlType != uQW0.GgmlType || u.Ne0 != uQW0.Ne0 || u.Ne1 != uQW0.Ne1 ||
                    d.GgmlType != dQW0.GgmlType || d.Ne0 != dQW0.Ne0 || d.Ne1 != dQW0.Ne1) return false;
                _moeGatePtrs[k] = g.CacheKey; _moeUpPtrs[k] = u.CacheKey; _moeDownPtrs[k] = d.CacheKey;
            }

            bool useShared = false;
            IntPtr sgPtr = IntPtr.Zero, suPtr = IntPtr.Zero, sdPtr = IntPtr.Zero;
            int sgType = 0, suType = 0, sdType = 0;
            long sgNe0 = 0, sgNe1 = 0, sgBytes = 0, suNe0 = 0, suNe1 = 0, suBytes = 0, sdNe0 = 0, sdNe1 = 0, sdBytes = 0;
            float sharedScalar = 0f;

            if (_hasSharedExperts != null && _hasSharedExperts[layer]
                && _ffnGateShexpQW[layer] != null && _ffnUpShexpQW[layer] != null && _ffnDownShexpQW[layer] != null)
            {
                var sg = _ffnGateShexpQW[layer]; var su = _ffnUpShexpQW[layer]; var sd = _ffnDownShexpQW[layer];
                sgPtr = sg.CacheKey; suPtr = su.CacheKey; sdPtr = sd.CacheKey;
                sgType = sg.GgmlType; suType = su.GgmlType; sdType = sd.GgmlType;
                sgNe0 = sg.Ne0; sgNe1 = sg.Ne1; sgBytes = sg.RawBytes;
                suNe0 = su.Ne0; suNe1 = su.Ne1; suBytes = su.RawBytes;
                sdNe0 = sd.Ne0; sdNe1 = sd.Ne1; sdBytes = sd.RawBytes;
                sharedScalar = 1.0f;
                if (_hasSharedExpertGate != null && _hasSharedExpertGate[layer])
                {
                    var gateInpVec = _ffnGateInpShexpVec[layer];
                    if (gateInpVec != null)
                    {
                        float* tokenRow = GetFloatPtr(input);
                        float* gateInpPtr = GetFloatPtr(gateInpVec);
                        int n = Math.Min((int)gateInpVec.ElementCount(), hiddenSize);
                        sharedScalar = SigmoidScalar(VecDot(tokenRow, gateInpPtr, n));
                    }
                }
                useShared = true;
            }

            long t0exp = Stopwatch.GetTimestamp();
            GgmlBasicOps.MoEExpertsSwiGLUResidual(residual, input,
                _numExpertsUsed, _moeGatePtrs, _moeUpPtrs, _moeDownPtrs,
                gQW0.GgmlType, gQW0.Ne0, gQW0.Ne1, gQW0.RawBytes,
                uQW0.GgmlType, uQW0.Ne0, uQW0.Ne1, uQW0.RawBytes,
                dQW0.GgmlType, dQW0.Ne0, dQW0.Ne1, dQW0.RawBytes,
                routeW, useShared, sgPtr, suPtr, sdPtr,
                sgType, sgNe0, sgNe1, sgBytes, suType, suNe0, suNe1, suBytes,
                sdType, sdNe0, sdNe1, sdBytes, sharedScalar);
            _linearTicks += Stopwatch.GetTimestamp() - t0exp;
            InvalidateTensorDeviceCache(residual);
            return true;
        }

        private unsafe bool TryMoEPrefillBatchedMlx(
            Tensor input,
            Tensor output,
            float* routePtr,
            bool routeRowsAreLogits,
            Tensor sharedDownAll,
            Tensor sharedGateInpVec,
            int layer,
            int seqLen,
            int hiddenSize)
        {
            if (_backend != BackendType.Mlx
                || seqLen <= 1
                || _numExperts <= 0
                || _numExpertsUsed <= 0
                || _expertGateQW == null
                || _expertUpQW == null
                || _expertDownQW == null
                || _expertGateQW[layer] == null
                || _expertUpQW[layer] == null
                || _expertDownQW[layer] == null)
            {
                return false;
            }

            int nUsed = _numExpertsUsed;
            int totalRoutes = checked(seqLen * nUsed);
            int[] selectedExperts = new int[totalRoutes];
            float[] selectedWeights = new float[totalRoutes];
            int[] expertCounts = new int[_numExperts];
            int[] tokTop = new int[nUsed];
            float[] tokW = new float[nUsed];

            for (int s = 0; s < seqLen; s++)
            {
                float* routeRow = routePtr + (long)s * _numExperts;
                SelectTopKRouteWeights(routeRow, routeRowsAreLogits, tokTop, tokW);

                int routeOffset = s * nUsed;
                for (int k = 0; k < nUsed; k++)
                {
                    int expert = tokTop[k];
                    selectedExperts[routeOffset + k] = expert;
                    selectedWeights[routeOffset + k] = tokW[k];
                    expertCounts[expert]++;
                }
            }

            int[] expertOffsets = new int[_numExperts + 1];
            for (int e = 0; e < _numExperts; e++)
                expertOffsets[e + 1] = expertOffsets[e] + expertCounts[e];

            int[] cursors = new int[_numExperts];
            Array.Copy(expertOffsets, cursors, _numExperts);
            int[] routedTokenRows = new int[totalRoutes];
            float[] routedWeights = new float[totalRoutes];
            for (int s = 0; s < seqLen; s++)
            {
                int routeOffset = s * nUsed;
                for (int k = 0; k < nUsed; k++)
                {
                    int expert = selectedExperts[routeOffset + k];
                    int dst = cursors[expert]++;
                    routedTokenRows[dst] = s;
                    routedWeights[dst] = selectedWeights[routeOffset + k];
                }
            }

            // Batched per-layer expert dispatch. Uploads routedTokenRows /
            // routedWeights once, gathers all routes into a single
            // [totalRoutes, hidden] tensor, then narrow-views that buffer
            // per expert. Per-expert work reduces to: narrow ÔåÆ matmul
            // (gate/up) ÔåÆ SiLUMul ÔåÆ matmul (down) ÔåÆ scatter-add with
            // narrow-views of the layer-wide indices/weights tensors.
            // Each narrow is a free view (shares storage, just adjusts
            // offset/sizes).
            Tensor rowIndicesAll = null;
            Tensor routeWeightsAll = null;
            Tensor batchInputAll = null;
            try
            {
                rowIndicesAll = CreateIntTensor(routedTokenRows, totalRoutes);
                routeWeightsAll = CreateFloatTensor(routedWeights, totalRoutes);
                batchInputAll = new Tensor(_allocator, DType.Float32, totalRoutes, hiddenSize);
                if (!MlxFusedOps.TryGatherRows(batchInputAll, input, rowIndicesAll))
                    return false;

                for (int e = 0; e < _numExperts; e++)
                {
                    int start = expertOffsets[e];
                    int batchSize = expertOffsets[e + 1] - start;
                    if (batchSize == 0)
                        continue;

                    Tensor batchInput = null;
                    Tensor rowIndices = null;
                    Tensor routeWeights = null;
                    Tensor gate = null;
                    Tensor up = null;
                    Tensor down = null;
                    try
                    {
                        batchInput = batchInputAll.Narrow(0, start, batchSize);
                        rowIndices = rowIndicesAll.Narrow(0, start, batchSize);
                        routeWeights = routeWeightsAll.Narrow(0, start, batchSize);

                        gate = ExpertLinearForwardAlloc(batchInput, layer, e, kind: 0);
                        up = ExpertLinearForwardAlloc(batchInput, layer, e, kind: 1);
                        if (gate == null || up == null)
                            return false;

                        Ops.SiLUMul(gate, gate, up);

                        down = ExpertLinearForwardAlloc(gate, layer, e, kind: 2);
                        if (down == null)
                            return false;

                        if (!MlxFusedOps.TryScatterAddWeightedRows(output, down, rowIndices, routeWeights))
                            return false;
                    }
                    finally
                    {
                        down?.Dispose();
                        up?.Dispose();
                        gate?.Dispose();
                        // Narrow views: dispose to release their tensor
                        // wrappers; the underlying storage stays alive via
                        // batchInputAll / rowIndicesAll / routeWeightsAll.
                        batchInput?.Dispose();
                        rowIndices?.Dispose();
                        routeWeights?.Dispose();
                    }
                }

                if (!TryAddSharedExpertMlx(output, input, sharedDownAll, sharedGateInpVec, seqLen, hiddenSize))
                    return false;

                return true;
            }
            catch (NotSupportedException)
            {
                return false;
            }
            catch (InvalidOperationException)
            {
                return false;
            }
            finally
            {
                batchInputAll?.Dispose();
                routeWeightsAll?.Dispose();
                rowIndicesAll?.Dispose();
            }
        }

        private bool TryAddSharedExpertMlx(Tensor output, Tensor input, Tensor sharedDownAll, Tensor sharedGateInpVec, int seqLen, int hiddenSize)
        {
            if (sharedDownAll == null)
                return true;

            if (sharedGateInpVec == null)
            {
                Ops.Add(output, output, sharedDownAll);
                return true;
            }

            if (sharedGateInpVec.ElementType != DType.Float32
                || sharedGateInpVec.ElementCount() != hiddenSize
                || sharedGateInpVec.Storage is not MlxStorage)
            {
                return false;
            }

            using var gateMatrix = sharedGateInpVec.View(hiddenSize, 1);
            using var gateLogits = new Tensor(_allocator, DType.Float32, seqLen, 1);
            Ops.Addmm(gateLogits, 0, gateLogits, 1.0f, input, gateMatrix);
            Ops.Sigmoid(gateLogits, gateLogits);

            using var gatedShared = new Tensor(_allocator, DType.Float32, seqLen, hiddenSize);
            Ops.Mul(gatedShared, sharedDownAll, gateLogits);
            Ops.Add(output, output, gatedShared);
            return true;
        }

        private unsafe Tensor MoEForward(Tensor input, int layer, int seqLen)
        {
            int hiddenSize = Config.HiddenSize;

            Tensor routerLogits = LinearForwardCached(input, _ffnGateInpQW[layer], _ffnGateInpF32[layer]);
            if (routerLogits == null)
                throw new InvalidOperationException($"Missing MoE router weight for layer {layer}: {_ffnGateInpKey[layer]}");

            bool routeRowsAreLogits = _normTopKProb;

            // Direct-CUDA on-device MoE decode: router + expert FFN + shared expert
            // entirely on the GPU (no per-expert host readback), so the decode layer
            // loop stays CUDA-graph capturable. Requires the normalized top-k softmax
            // routing (== the on-device ts_moe_router_f32); other routings fall through
            // to the host path below. routerLogits stays alive on a fall-through.
            if (_backend == BackendType.Cuda && routeRowsAreLogits && CanUseQwenCudaMoEOnDevice(layer)
                && (seqLen == 1 || s_qwenCudaMoePrefillOnDevice))
            {
                Tensor onDevice = seqLen == 1
                    ? TryCudaMoEForwardOnDevice(input, routerLogits, layer)
                    : TryCudaMoEForwardPrefillOnDevice(input, routerLogits, layer, seqLen);
                if (onDevice != null)
                    return onDevice;
            }

            // Device-routing fast path: skip the per-MoE-layer routerData
            // host sync entirely. Compute top-K + softmax on the device and
            // feed the resulting [K] int32 + [1, K] float32 device tensors
            // directly into the batched MoE matmul.
            //
            // Requires: greedy router, MLX decode-on-device + batched MoE
            // available, no shared-expert gate (the gate scalar still uses
            // host VecDot today). Falls through to the host path otherwise.
            bool hasSharedGate = _hasSharedExperts != null && _hasSharedExperts[layer]
                && _hasSharedExpertGate != null && _hasSharedExpertGate[layer]
                && _ffnGateInpShexpVec?[layer] != null;
            if (MlxDeviceRouter
                && _backend == BackendType.Mlx
                && seqLen == 1
                && routeRowsAreLogits
                && MlxBatchedMoeDecode
                && !hasSharedGate
                && _layerStackedGate != null && _layerStackedGate[layer] != null
                && _moeGateBuf != null)
            {
                Tensor? maybe = TryMoEForwardDeviceRouter(input, routerLogits, layer);
                if (maybe != null)
                {
                    return maybe;
                }
                // Fell through: continue with the original host-routing path.
                // routerLogits is still alive.
            }

            Tensor routerData;
            if (routeRowsAreLogits)
            {
                routerData = routerLogits;
            }
            else
            {
                routerData = Ops.Softmax(null, routerLogits);
                routerLogits.Dispose();
            }

            float* routePtr = GetFloatPtr(routerData);

            // Direct CUDA prefill: retain host top-k grouping, but gather input
            // rows and scatter weighted expert outputs on device. This preserves
            // the legacy expert-batched math while removing its per-expert
            // activation round-trips.
            if (_backend == BackendType.Cuda && seqLen > 1)
            {
                Tensor groupedCuda = TryCudaMoEForwardPrefillGrouped(
                    input, routePtr, routeRowsAreLogits, layer, seqLen);
                if (groupedCuda != null)
                {
                    routerData.Dispose();
                    return groupedCuda;
                }
            }

            // GGML stacked-MoE fast path (CUDA / CPU / Metal). Route top-K on
            // host, then run EVERY routed expert for ALL tokens through ONE
            // ggml_mul_mat_id dispatch per projection against the DEVICE-RESIDENT
            // stacked expert weights (GgmlBasicOps.MoEFFNPrefill), instead of the
            // per-expert ExpertLinearForwardAlloc loop below. That loop re-streams
            // each selected expert's quantized weight from host every step ÔÇö and
            // the stacked-MoE VRAM fix (ShouldPreloadCudaQuantWeightToDevice)
            // deliberately makes per-expert weights NON-resident, so on ggml_cuda
            // it is catastrophically slow (the N>=2 batched decode "deadlock":
            // ~K*3*numLayers host->device weight uploads per token). The stacked
            // weights ARE device-resident, so this dispatch amortizes the weight
            // read across all tokens ÔÇö the vLLM-style batched MoE. Handles decode
            // (seqLen==1/N) and prefill chunks uniformly.
            if (IsGgmlBackend
                && _layerStackedGate != null && _layerStackedGate[layer] != null
                && _layerStackedUp != null && _layerStackedUp[layer] != null
                && _layerStackedDown != null && _layerStackedDown[layer] != null
                && !IsQwen35GgmlStackedMoeDisabled())
            {
                Tensor stackedOut = TryMoEForwardGgmlStacked(input, routePtr, routeRowsAreLogits, layer, seqLen);
                if (stackedOut != null)
                {
                    routerData.Dispose();
                    InvalidateTensorDeviceCache(stackedOut);
                    return stackedOut;
                }
            }

            var output = new Tensor(_allocator, DType.Float32, seqLen, hiddenSize);
            float* outputPtr = null;
            // MLX decode (seqLen == 1) keeps expert accumulation on-device
            // (one Ops.Fill then per-expert weighted-add via
            // TryAddScaledInPlace) and syncs once at the layer boundary
            // instead of forcing a host round-trip per expert.
            bool mlxDecodeOnDevice = _backend == BackendType.Mlx
                && seqLen == 1;
            if (_backend == BackendType.Mlx && (seqLen > 1 || mlxDecodeOnDevice))
            {
                Ops.Fill(output, 0f);
            }
            else
            {
                outputPtr = GetFloatPtr(output);
                VecZero(outputPtr, seqLen * hiddenSize);
            }

            // For prefill, batch the shared expert over all tokens up-front.
            Tensor sharedDownAll = null;
            float* sharedGateInpPtr = null;
            int sharedGateInpDim = hiddenSize;

            if (_hasSharedExperts != null && _hasSharedExperts[layer])
            {
                Tensor sharedGate = LinearForwardCached(input, _ffnGateShexpQW[layer], _ffnGateShexpF32[layer]);
                Tensor sharedUp = LinearForwardCached(input, _ffnUpShexpQW[layer], _ffnUpShexpF32[layer]);
                if (sharedGate != null && sharedUp != null)
                {
                    Ops.SiLUMul(sharedGate, sharedGate, sharedUp);
                    sharedDownAll = LinearForwardCached(sharedGate, _ffnDownShexpQW[layer], _ffnDownShexpF32[layer]);
                }
                sharedUp?.Dispose();
                sharedGate?.Dispose();

                if (_hasSharedExpertGate != null && _hasSharedExpertGate[layer])
                {
                    var gateInpVec = _ffnGateInpShexpVec[layer];
                    if (gateInpVec != null)
                    {
                        sharedGateInpPtr = GetFloatPtr(gateInpVec);
                        sharedGateInpDim = (int)gateInpVec.ElementCount();
                    }
                }
            }

            if (_backend == BackendType.Mlx
                && TryMoEPrefillBatchedMlx(input, output, routePtr, routeRowsAreLogits, sharedDownAll, _ffnGateInpShexpVec?[layer], layer, seqLen, hiddenSize))
            {
                sharedDownAll?.Dispose();
                routerData.Dispose();
                return output;
            }

            // For the mlxDecodeOnDevice path we already zeroed `output` on
            // device above; re-syncing it to host here would force a
            // deviceÔåÆhost copy and discard that work. Likewise `input` only
            // needs a host pointer if a downstream code path (shared expert
            // gate scalar / legacy outRow accumulate) actually reads it.
            float* inputPtr = null;
            if (!mlxDecodeOnDevice)
            {
                outputPtr = GetFloatPtr(output);
                VecZero(outputPtr, seqLen * hiddenSize);
                inputPtr = GetFloatPtr(input);
            }
            else if (sharedDownAll != null && sharedGateInpPtr != null)
            {
                // Shared expert with a gate vector needs the input row on
                // host for the dot product. Only sync here, never above.
                inputPtr = GetFloatPtr(input);
            }
            float[] routeW = _moeRouteW;
            int[] topExperts = _moeTopExperts;

            // Reuse scratch buffers for both decode and prefill. For prefill (seqLen > 1),
            // instead of Narrow + NewContiguous per token (which allocates and copies), we
            // reuse a single [1, hiddenSize] tensor and copy the row data into it with
            // Buffer.MemoryCopy. This eliminates O(seqLen * numMoELayers) allocations.
            bool useReusedBuffers = _moeGateBuf != null;

            // For prefill: reuse a single row tensor that we populate via memcpy per token.
            Tensor prefillRowBuf = null;
            if (seqLen > 1 && useReusedBuffers && _moeTokenInput != null
                && _moeTokenInput.Sizes[1] == hiddenSize)
            {
                prefillRowBuf = _moeTokenInput;
            }

            // Prefill batched-by-expert path: group tokens by expert assignment
            // and run batched matmuls instead of per-token dispatches.
            // Converts seqLen*numExpertsUsed individual matmuls ÔåÆ numExperts batched matmuls.
            if (seqLen > 1 && _expertGateQW != null && _expertGateQW[layer] != null)
            {
                // 1. Route all tokens and group by expert
                var expertBatches = new List<(int tokenIdx, float weight)>[_numExperts];
                for (int i = 0; i < _numExperts; i++)
                    expertBatches[i] = new List<(int, float)>();

                for (int s = 0; s < seqLen; s++)
                {
                    float* routeRow = routePtr + (long)s * _numExperts;
                    int[] tokTop = new int[_numExpertsUsed];
                    float[] tokW = new float[_numExpertsUsed];
                    SelectTopKRouteWeights(routeRow, routeRowsAreLogits, tokTop, tokW);
                    for (int k = 0; k < _numExpertsUsed; k++)
                        expertBatches[tokTop[k]].Add((s, tokW[k]));
                }

                // 2. Process each expert with batched tokens
                int rowBytes = hiddenSize * sizeof(float);
                for (int e = 0; e < _numExperts; e++)
                {
                    var batch = expertBatches[e];
                    if (batch.Count == 0) continue;
                    int batchSize = batch.Count;

                    // Gather input rows
                    var batchInput = new Tensor(_allocator, DType.Float32, batchSize, hiddenSize);
                    float* batchPtr = GetFloatPtr(batchInput);
                    for (int b = 0; b < batchSize; b++)
                        Buffer.MemoryCopy(inputPtr + (long)batch[b].tokenIdx * hiddenSize,
                            batchPtr + (long)b * hiddenSize, rowBytes, rowBytes);

                    // Batched expert FFN: gate ÔåÆ up ÔåÆ SiLU*up ÔåÆ down
                    Tensor gate = ExpertLinearForwardAlloc(batchInput, layer, e, kind: 0);
                    Tensor up = ExpertLinearForwardAlloc(batchInput, layer, e, kind: 1);
                    batchInput.Dispose();
                    if (gate == null || up == null) { gate?.Dispose(); up?.Dispose(); continue; }

                    Ops.SiLUMul(gate, gate, up);
                    up.Dispose();

                    Tensor down = ExpertLinearForwardAlloc(gate, layer, e, kind: 2);
                    gate.Dispose();
                    if (down == null) continue;

                    // Scatter-accumulate with routing weights
                    float* downPtr = GetFloatPtr(down);
                    for (int b = 0; b < batchSize; b++)
                    {
                        float w = batch[b].weight;
                        float* src = downPtr + (long)b * hiddenSize;
                        float* dst = outputPtr + (long)batch[b].tokenIdx * hiddenSize;
                        VecScaleAdd(dst, src, w, hiddenSize);
                    }
                    down.Dispose();
                }

                // 3. Add shared expert contribution (already computed in batch)
                if (sharedDownAll != null)
                {
                    for (int s = 0; s < seqLen; s++)
                    {
                        float gateScalar = 1.0f;
                        if (sharedGateInpPtr != null)
                        {
                            float* tokenRow = inputPtr + (long)s * hiddenSize;
                            int n = Math.Min(sharedGateInpDim, hiddenSize);
                            gateScalar = SigmoidScalar(VecDot(tokenRow, sharedGateInpPtr, n));
                        }
                        float* sharedPtr = GetFloatPtr(sharedDownAll) + (long)s * hiddenSize;
                        VecScaleAdd(outputPtr + (long)s * hiddenSize, sharedPtr, gateScalar, hiddenSize);
                    }
                }
            }
            else
            {
            // Original token-by-token path (decode and fallback)
            for (int s = 0; s < seqLen; s++)
            {
                float* routeRow = routePtr + (long)s * _numExperts;
                SelectTopKRouteWeights(routeRow, routeRowsAreLogits, topExperts, routeW);

                Tensor tokenInput;
                bool disposeTokenInput;
                if (seqLen == 1)
                {
                    tokenInput = input;
                    disposeTokenInput = false;
                }
                else if (prefillRowBuf != null)
                {
                    float* srcRow = inputPtr + (long)s * hiddenSize;
                    float* dstRow = GetFloatPtr(prefillRowBuf);
                    long bytes = (long)hiddenSize * sizeof(float);
                    Buffer.MemoryCopy(srcRow, dstRow, bytes, bytes);
                    InvalidateTensorDeviceCache(prefillRowBuf);
                    tokenInput = prefillRowBuf;
                    disposeTokenInput = false;
                }
                else
                {
                    using var rowView = input.Narrow(0, s, 1);
                    tokenInput = Ops.NewContiguous(rowView);
                    disposeTokenInput = true;
                }

                // For MLX decode (seqLen=1), keep accumulation on device so
                // the kernels for all 8 active experts queue up without a
                // host sync between them. The legacy host-side accumulation
                // is still used for other backends and for fallback.
                if (mlxDecodeOnDevice && useReusedBuffers)
                {
                    RunMoEExpertsReusedMlxOnDevice(output, tokenInput, layer, topExperts, routeW);

                    if (sharedDownAll != null)
                    {
                        float gateScalar = 1.0f;
                        if (sharedGateInpPtr != null && inputPtr != null)
                        {
                            // seqLen==1 in this branch, so the input row is
                            // just inputPtr (no s*hiddenSize offset).
                            int n = Math.Min(sharedGateInpDim, hiddenSize);
                            gateScalar = SigmoidScalar(VecDot(inputPtr, sharedGateInpPtr, n));
                        }
                        AddScaledTensorMlx(output, sharedDownAll, gateScalar);
                    }
                }
                else
                {
                    float* outRow = outputPtr + (long)s * hiddenSize;

                    if (useReusedBuffers)
                    {
                        RunMoEExpertsReused(tokenInput, layer, topExperts, routeW, outRow, hiddenSize);
                    }
                    else
                    {
                        RunMoEExpertsAllocating(tokenInput, layer, topExperts, routeW, outRow, hiddenSize);
                    }

                    if (sharedDownAll != null)
                    {
                        float gateScalar = 1.0f;
                        if (sharedGateInpPtr != null)
                        {
                            float* tokenRow = inputPtr + (long)s * hiddenSize;
                            int n = Math.Min(sharedGateInpDim, hiddenSize);
                            gateScalar = SigmoidScalar(VecDot(tokenRow, sharedGateInpPtr, n));
                        }
                        float* sharedPtr = GetFloatPtr(sharedDownAll) + (long)s * hiddenSize;
                        VecScaleAdd(outRow, sharedPtr, gateScalar, hiddenSize);
                    }
                }

                if (disposeTokenInput)
                    tokenInput.Dispose();
            }
            } // end of token-by-token else

            sharedDownAll?.Dispose();
            routerData.Dispose();

            InvalidateTensorDeviceCache(output);
            return output;
        }

        private static bool IsQwen35GgmlStackedMoeDisabled()
            => string.Equals(Environment.GetEnvironmentVariable("TS_QWEN35_STACKED_MOE"),
                             "0", StringComparison.Ordinal);

        // GGML stacked-MoE forward over N tokens (decode batch or prefill chunk).
        // Routes top-K on host, runs the routed experts via ONE
        // GgmlBasicOps.MoEFFNPrefill (a single ggml_mul_mat_id per projection
        // over the device-resident stacked weights), then adds the gated shared
        // expert. Returns null on any precondition miss so MoEForward falls back
        // to the per-expert path. This is the path that makes N>=2 batched MoE
        // decode fast on ggml_cuda (see the call-site comment in MoEForward).
        private unsafe Tensor TryMoEForwardGgmlStacked(
            Tensor input, float* routePtr, bool routeRowsAreLogits, int layer, int seqLen)
        {
            var gateW = _layerStackedGate[layer];
            var upW = _layerStackedUp[layer];
            var downW = _layerStackedDown[layer];

            int hiddenSize = Config.HiddenSize;
            int intermediate = (int)gateW.PerExpertNe1;
            int K = _numExpertsUsed;

            // Stacked dims must line up with hidden/intermediate. The three
            // projections do NOT need to share a quant type: MoEFFNPrefill builds
            // gate_w/up_w/down_w as independent ggml tensors (each with its own
            // gateType/upType/downType) and runs a separate ggml_mul_mat_id per
            // projection, so mixed-type "UD"/dynamic quants (e.g. Qwen3.6 UD-IQ2_XXS,
            // where gate/up are IQ2_XXS but down is IQ2_S) are fully supported. The
            // old all-types-must-match guard forced every such model onto the
            // per-expert ExpertLinearForwardAlloc loop, which re-streams every routed
            // expert's NON-resident quantized weight from host each forward ÔÇö the
            // dominant (~18 s/forward, seqLen-independent) prefill cost on ggml_cuda.
            if (gateW.PerExpertNe0 != hiddenSize || upW.PerExpertNe0 != hiddenSize
                || upW.PerExpertNe1 != intermediate
                || downW.PerExpertNe0 != intermediate || downW.PerExpertNe1 != hiddenSize)
                return null;

            // Host top-K routing for every token: ids[t*K+u], weights[t*K+u].
            int[] selExperts = new int[seqLen * K];
            float[] routeWts = new float[seqLen * K];
            int[] tokTop = new int[K];
            float[] tokW = new float[K];
            for (int s = 0; s < seqLen; s++)
            {
                float* routeRow = routePtr + (long)s * _numExperts;
                SelectTopKRouteWeights(routeRow, routeRowsAreLogits, tokTop, tokW);
                for (int k = 0; k < K; k++)
                {
                    selExperts[s * K + k] = tokTop[k];
                    routeWts[s * K + k] = tokW[k];
                }
            }

            var output = new Tensor(_allocator, DType.Float32, seqLen, hiddenSize);
            try
            {
                GgmlBasicOps.MoEFFNPrefill(
                    input, output, seqLen, hiddenSize, intermediate,
                    gateW.NumExperts, K, selExperts, routeWts,
                    gateW.Data, gateW.GgmlType, gateW.PerExpertNe0, gateW.PerExpertNe1, gateW.TotalRawBytes,
                    upW.Data, upW.GgmlType, upW.PerExpertNe0, upW.PerExpertNe1, upW.TotalRawBytes,
                    downW.Data, downW.GgmlType, downW.PerExpertNe0, downW.PerExpertNe1, downW.TotalRawBytes,
                    gateBias: null, upBias: null, downBias: null,
                    activation: GgmlBasicOps.MoEActivation.SwiGLUSplit,
                    runOnCpu: MoeCpuOffloadConfig.IsLayerOnCpu(layer));
            }
            catch (Exception)
            {
                output.Dispose();
                return null;
            }

            // Gated shared expert, batched over all tokens:
            //   shared = down(silu(gate(x)) * up(x))
            //   out[t] += sigmoid(x[t] . gate_inp_shexp) * shared[t]
            if (_hasSharedExperts != null && _hasSharedExperts[layer])
            {
                Tensor sharedGate = LinearForwardCached(input, _ffnGateShexpQW[layer], _ffnGateShexpF32[layer]);
                Tensor sharedUp = LinearForwardCached(input, _ffnUpShexpQW[layer], _ffnUpShexpF32[layer]);
                Tensor sharedDown = null;
                if (sharedGate != null && sharedUp != null)
                {
                    Ops.SiLUMul(sharedGate, sharedGate, sharedUp);
                    sharedDown = LinearForwardCached(sharedGate, _ffnDownShexpQW[layer], _ffnDownShexpF32[layer]);
                }
                sharedUp?.Dispose();
                sharedGate?.Dispose();

                if (sharedDown != null)
                {
                    float* sharedGateInpPtr = null;
                    int sharedGateInpDim = hiddenSize;
                    if (_hasSharedExpertGate != null && _hasSharedExpertGate[layer]
                        && _ffnGateInpShexpVec?[layer] != null)
                    {
                        sharedGateInpPtr = GetFloatPtr(_ffnGateInpShexpVec[layer]);
                        sharedGateInpDim = (int)_ffnGateInpShexpVec[layer].ElementCount();
                    }

                    float* outputPtr = GetFloatPtr(output);
                    float* inputPtr = GetFloatPtr(input);
                    float* sharedPtr = GetFloatPtr(sharedDown);
                    for (int s = 0; s < seqLen; s++)
                    {
                        float gateScalar = 1.0f;
                        if (sharedGateInpPtr != null)
                        {
                            int n = Math.Min(sharedGateInpDim, hiddenSize);
                            gateScalar = SigmoidScalar(VecDot(inputPtr + (long)s * hiddenSize, sharedGateInpPtr, n));
                        }
                        VecScaleAdd(outputPtr + (long)s * hiddenSize,
                                    sharedPtr + (long)s * hiddenSize, gateScalar, hiddenSize);
                    }
                    sharedDown.Dispose();
                }
            }

            return output;
        }

        // MLX-only decode helper: keep all expert accumulation on the GPU.
        // The legacy RunMoEExpertsReused path issues 3 matmuls + SiLUMul per
        // expert and then calls GetFloatPtr+VecScaleAdd, which forces an
        // mlx_eval+deviceÔåÆhost sync after every expert. With 8 active
        // experts ├ù 60 MoE layers that's ~480 round trips per decode token.
        //
        // Two on-device variants:
        //  - Batched: a single custom Metal kernel processes all K active
        //    experts' gate (then up, then down) matmuls in one Metal
        //    dispatch each. Requires a per-quant-type batched MoE kernel
        //    (currently only IQ2_XXS). Saves K-1 dispatches per matmul:
        //    on Qwen3.5 with K=8 that's ~21 dispatches saved per MoE
        //    layer (was 24 individual matmuls + 8 SiLUMul + 8 AddScaled ÔåÆ
        //    now 3 batched matmuls + 1 SiLUMul + 1 routeW@down matmul).
        //  - Sequential (fallback): per-expert matmul loop with the
        //    fused AddScaled accumulator. Used when no batched kernel
        //    is available for the layer's quant type.
        private void RunMoEExpertsReusedMlxOnDevice(Tensor output, Tensor tokenInput, int layer,
            int[] topExperts, float[] routeW)
        {
            if (MlxBatchedMoeDecode && TryRunMoEExpertsBatchedMlx(output, tokenInput, layer, topExperts, routeW))
                return;

            for (int k = 0; k < _numExpertsUsed; k++)
            {
                int e = topExperts[k];
                if (!ExpertLinearForwardInto(_moeGateBuf, tokenInput, layer, e, kind: 0))
                    continue;
                if (!ExpertLinearForwardInto(_moeUpBuf, tokenInput, layer, e, kind: 1))
                    continue;

                Ops.SiLUMul(_moeGateBuf, _moeGateBuf, _moeUpBuf);

                if (!ExpertLinearForwardInto(_moeDownBuf, _moeGateBuf, layer, e, kind: 2))
                    continue;

                AddScaledTensorMlx(output, _moeDownBuf, routeW[k]);
            }
        }

        // Decode-only MoEForward that does the routing on-device. Skips
        // the per-MoE-layer host sync on routerLogits/routerData by:
        //   1. Running the top-K + softmax of routerLogits on device ÔåÆ
        //      _moeBatchedExpertIndices [K] int32, _moeBatchedRouteWeights
        //      [1, K] float32.
        //   2. Running the batched MoE matmul using those device tensors.
        //   3. Adding the shared expert contribution if applicable
        //      (no host-side gate scalar ÔÇö only used when there's no
        //      shared-expert gate, asserted by the caller).
        // Returns null on precondition failure so the caller can fall
        // through to the host-routing path.
        private Tensor? TryMoEForwardDeviceRouter(Tensor input, Tensor routerLogits, int layer)
        {
            int hiddenSize = Config.HiddenSize;
            int K = _numExpertsUsed;

            // Lazy-allocate the batched scratch buffers. Same buffers as
            // TryRunMoEExpertsBatchedMlx so reuse the existing shapes /
            // disposal hooks. We need the indices + routeWeights tensors
            // before the batched MoE runs; the gate/up/down buffers are
            // also needed and have the same lifecycle.
            var gateW = _layerStackedGate[layer];
            if (gateW == null) return null;
            int intermediate = (int)gateW.PerExpertNe1;

            if (_moeBatchedGate == null
                || _moeBatchedGate.Sizes[0] != K || _moeBatchedGate.Sizes[1] != intermediate)
            {
                _moeBatchedGate?.Dispose();
                _moeBatchedUp?.Dispose();
                _moeBatchedDown?.Dispose();
                _moeBatchedExpertIndices?.Dispose();
                _moeBatchedRouteWeights?.Dispose();
                _moeBatchedGate = new Tensor(_allocator, DType.Float32, K, intermediate);
                _moeBatchedUp = new Tensor(_allocator, DType.Float32, K, intermediate);
                _moeBatchedDown = new Tensor(_allocator, DType.Float32, K, hiddenSize);
                _moeBatchedExpertIndices = new Tensor(_allocator, DType.Int32, K);
                _moeBatchedRouteWeights = new Tensor(_allocator, DType.Float32, 1, K);
            }

            // 1. Device-side top-K + softmax on routerLogits.
            if (!MlxFusedOps.TryMoeRouterTopKSoftmax(
                    routerLogits, _moeBatchedExpertIndices, _moeBatchedRouteWeights))
            {
                return null;
            }

            // 2. Allocate output, zero on device.
            var output = new Tensor(_allocator, DType.Float32, 1, hiddenSize);
            Ops.Fill(output, 0f);

            // 3. Pre-compute shared expert (no shared gate path here).
            Tensor sharedDownAll = null;
            if (_hasSharedExperts != null && _hasSharedExperts[layer])
            {
                Tensor sharedGate = LinearForwardCached(input, _ffnGateShexpQW[layer], _ffnGateShexpF32[layer]);
                Tensor sharedUp = LinearForwardCached(input, _ffnUpShexpQW[layer], _ffnUpShexpF32[layer]);
                if (sharedGate != null && sharedUp != null)
                {
                    Ops.SiLUMul(sharedGate, sharedGate, sharedUp);
                    sharedDownAll = LinearForwardCached(sharedGate, _ffnDownShexpQW[layer], _ffnDownShexpF32[layer]);
                }
                sharedUp?.Dispose();
                sharedGate?.Dispose();
            }

            try
            {
                // 4. Batched MoE matmul using the device tensors we just
                //    computed. Note we bypass TryRunMoEExpertsBatchedMlx's
                //    SetElementsAsInt/SetElementsAsFloat (since the tensors
                //    are already populated) by inlining a smaller variant.
                if (!RunBatchedMoeMatmulFromDevice(output, input, layer))
                {
                    output.Dispose();
                    sharedDownAll?.Dispose();
                    return null;
                }

                // 5. Shared expert add (no host scalar ÔÇö the gate-less
                //    path just adds the full shared down).
                if (sharedDownAll != null)
                {
                    Ops.Add(output, output, sharedDownAll);
                }
                routerLogits.Dispose();
                return output;
            }
            finally
            {
                sharedDownAll?.Dispose();
            }
        }

        // Inner helper: runs the 3 batched matmuls + SiLUMul + routeW@down
        // chain assuming _moeBatchedExpertIndices and _moeBatchedRouteWeights
        // are already populated (typically by the device-router path).
        private bool RunBatchedMoeMatmulFromDevice(Tensor output, Tensor tokenInput, int layer)
        {
            var gateW = _layerStackedGate[layer];
            var upW = _layerStackedUp[layer];
            var downW = _layerStackedDown[layer];
            if (gateW == null || upW == null || downW == null) return false;
            if (!MlxQuantizedOps.SupportsBatchedMoeMatmul(gateW.GgmlType)) return false;
            try
            {
                if (!MlxQuantizedOps.TryMoeMatmulBatched(
                        _moeBatchedGate, tokenInput, _moeBatchedExpertIndices,
                        gateW.Data, gateW.Data, gateW.GgmlType,
                        gateW.PerExpertNe0, gateW.PerExpertNe1, gateW.NumExperts, gateW.TotalRawBytes,
                        sharedInput: true))
                    return false;
                if (!MlxQuantizedOps.TryMoeMatmulBatched(
                        _moeBatchedUp, tokenInput, _moeBatchedExpertIndices,
                        upW.Data, upW.Data, upW.GgmlType,
                        upW.PerExpertNe0, upW.PerExpertNe1, upW.NumExperts, upW.TotalRawBytes,
                        sharedInput: true))
                    return false;
                Ops.SiLUMul(_moeBatchedGate, _moeBatchedGate, _moeBatchedUp);
                if (!MlxQuantizedOps.TryMoeMatmulBatched(
                        _moeBatchedDown, _moeBatchedGate, _moeBatchedExpertIndices,
                        downW.Data, downW.Data, downW.GgmlType,
                        downW.PerExpertNe0, downW.PerExpertNe1, downW.NumExperts, downW.TotalRawBytes,
                        sharedInput: false))
                    return false;
                Ops.Addmm(output, 1.0f, output, 1.0f, _moeBatchedRouteWeights, _moeBatchedDown);
                return true;
            }
            catch
            {
                return false;
            }
        }

        // Batched-MoE decode. Requires:
        //  - Stacked expert weights for gate/up/down on this layer
        //  - A batched-MoE matmul kernel for the weights' quant type
        //  - All experts share the same (quant type, dimensions)
        // Returns false if any precondition fails so the caller falls back
        // to the sequential path.
        private bool TryRunMoEExpertsBatchedMlx(Tensor output, Tensor tokenInput, int layer,
            int[] topExperts, float[] routeW)
        {
            // Need stacked weights for all three projections.
            if (_layerStackedGate == null || _layerStackedUp == null || _layerStackedDown == null)
                return false;
            var gateW = _layerStackedGate[layer];
            var upW = _layerStackedUp[layer];
            var downW = _layerStackedDown[layer];
            if (gateW == null || upW == null || downW == null)
                return false;
            if (!MlxQuantizedOps.SupportsBatchedMoeMatmul(gateW.GgmlType)) return false;
            if (gateW.GgmlType != upW.GgmlType || upW.GgmlType != downW.GgmlType) return false;
            // gate/up share input dim (hidden) and output dim (intermediate)
            if (gateW.PerExpertNe0 != upW.PerExpertNe0 || gateW.PerExpertNe1 != upW.PerExpertNe1) return false;
            // down: input dim = intermediate, output dim = hidden
            if (downW.PerExpertNe0 != gateW.PerExpertNe1 || downW.PerExpertNe1 != gateW.PerExpertNe0) return false;

            int K = _numExpertsUsed;
            int hiddenSize = (int)gateW.PerExpertNe0;
            int intermediate = (int)gateW.PerExpertNe1;
            if (tokenInput.DimensionCount != 2 || tokenInput.Sizes[0] != 1 || tokenInput.Sizes[1] != hiddenSize)
                return false;
            if (output.DimensionCount != 2 || output.Sizes[0] != 1 || output.Sizes[1] != hiddenSize)
                return false;

            // Lazy-init the batched scratch buffers.
            if (_moeBatchedGate == null
                || _moeBatchedGate.Sizes[0] != K || _moeBatchedGate.Sizes[1] != intermediate)
            {
                _moeBatchedGate?.Dispose();
                _moeBatchedUp?.Dispose();
                _moeBatchedDown?.Dispose();
                _moeBatchedExpertIndices?.Dispose();
                _moeBatchedRouteWeights?.Dispose();
                _moeBatchedGate = new Tensor(_allocator, DType.Float32, K, intermediate);
                _moeBatchedUp = new Tensor(_allocator, DType.Float32, K, intermediate);
                _moeBatchedDown = new Tensor(_allocator, DType.Float32, K, hiddenSize);
                _moeBatchedExpertIndices = new Tensor(_allocator, DType.Int32, K);
                _moeBatchedRouteWeights = new Tensor(_allocator, DType.Float32, 1, K);
            }

            // Upload topK expert indices and routing weights to device.
            // These are tiny (K ints + K floats) so the upload cost is
            // negligible compared to the per-MoE-layer GPU work we save.
            _moeBatchedExpertIndices.SetElementsAsInt(topExperts);
            _moeBatchedRouteWeights.SetElementsAsFloat(routeW);

            try
            {
                // 1+2+3 fused: produce _moeBatchedGate = silu(gate @ x) * (up @ x)
                // in ONE Metal kernel. Falls back to the 3-dispatch sequence
                // if the fused kernel isn't available (e.g. quant type
                // doesn't support it yet) or fails. Gated by
                // TS_MLX_MOE_FUSED_GATE_UP_SILU=0 to disable for A/B.
                bool fusedOk = false;
                if (!MlxMoeFusedGateUpSiluDisabled)
                {
                    fusedOk = MlxQuantizedOps.TryMoeFusedGateUpSilu(
                        _moeBatchedGate, tokenInput, _moeBatchedExpertIndices,
                        gateW.Data, gateW.Data, gateW.TotalRawBytes,
                        upW.Data, upW.Data, upW.TotalRawBytes,
                        gateW.GgmlType,
                        gateW.PerExpertNe0, gateW.PerExpertNe1, gateW.NumExperts);
                }
                if (!fusedOk)
                {
                    // 1. Batched gate matmul: [1, hidden] @ stackedGate ÔåÆ [K, intermediate]
                    if (!MlxQuantizedOps.TryMoeMatmulBatched(
                            _moeBatchedGate, tokenInput, _moeBatchedExpertIndices,
                            gateW.Data, gateW.Data, gateW.GgmlType,
                            gateW.PerExpertNe0, gateW.PerExpertNe1, gateW.NumExperts, gateW.TotalRawBytes,
                            sharedInput: true))
                        return false;

                    // 2. Batched up matmul: same input ÔåÆ [K, intermediate]
                    if (!MlxQuantizedOps.TryMoeMatmulBatched(
                            _moeBatchedUp, tokenInput, _moeBatchedExpertIndices,
                            upW.Data, upW.Data, upW.GgmlType,
                            upW.PerExpertNe0, upW.PerExpertNe1, upW.NumExperts, upW.TotalRawBytes,
                            sharedInput: true))
                        return false;

                    // 3. SiLUMul element-wise on [K, intermediate]: one kernel.
                    Ops.SiLUMul(_moeBatchedGate, _moeBatchedGate, _moeBatchedUp);
                }

                // 4. Batched down matmul: [K, intermediate] @ stackedDown ÔåÆ [K, hidden]
                if (!MlxQuantizedOps.TryMoeMatmulBatched(
                        _moeBatchedDown, _moeBatchedGate, _moeBatchedExpertIndices,
                        downW.Data, downW.Data, downW.GgmlType,
                        downW.PerExpertNe0, downW.PerExpertNe1, downW.NumExperts, downW.TotalRawBytes,
                        sharedInput: false))
                    return false;

                // 5. Weighted sum: routeW[1, K] @ down[K, hidden] = [1, hidden]
                //    Then accumulate into output. We do this as one Addmm:
                //    output = 1.0 * output + 1.0 * (routeW @ down)
                Ops.Addmm(output, 1.0f, output, 1.0f, _moeBatchedRouteWeights, _moeBatchedDown);
                return true;
            }
            catch
            {
                return false;
            }
        }

        // output += scalar * src, all on device, no host round trip. With the
        // mlx_compile path enabled this is ONE fused Metal kernel via
        // MlxFusedOps.TryAddScaledInPlace; otherwise we fall back to
        // mulv + addt (2 kernels). The fallback variant is destructive on
        // `src` ÔÇö _moeDownBuf is rewritten next iteration by the next
        // expert's down matmul anyway, so this is safe in the decode loop.
        private void AddScaledTensorMlx(Tensor output, Tensor src, float scalar)
        {
            if (scalar == 0f)
                return;

            if (MlxFusedOps.TryAddScaledInPlace(output, src, scalar))
                return;

            // Eager fallback: 2 MLX kernels (mulv + addt).
            Ops.Mul(src, src, scalar);
            Ops.Add(output, output, src);
        }

        private unsafe void RunMoEExpertsReused(Tensor tokenInput, int layer,
            int[] topExperts, float[] routeW, float* outRow, int hiddenSize)
        {
            // Fast path: batched SwiGLU MoE in a single GGML graph (decode only).
            // Combines 4*N = 32 dispatches (gate/up/silu*mul/down for 8 experts) into one
            // graph submission, dramatically reducing GPU dispatch overhead on Metal/CUDA.
            if (IsGgmlBackend
                && _moeBatchedResult != null
                && _moeGatePtrs != null
                && _expertGateQW != null && _expertUpQW != null && _expertDownQW != null
                && _expertGateQW[layer] != null && _expertUpQW[layer] != null && _expertDownQW[layer] != null)
            {
                bool allQuantized = true;
                QuantizedWeight gQW0 = null, uQW0 = null, dQW0 = null;
                for (int k = 0; k < _numExpertsUsed; k++)
                {
                    int e = topExperts[k];
                    var g = _expertGateQW[layer][e];
                    var u = _expertUpQW[layer][e];
                    var d = _expertDownQW[layer][e];
                    if (g == null || u == null || d == null)
                    {
                        allQuantized = false;
                        break;
                    }
                    if (gQW0 == null) { gQW0 = g; uQW0 = u; dQW0 = d; }

                    // All experts in a layer must share dtype + shape so the batched kernel can
                    // bind them as identically-shaped tensors. The GGUF expert layout guarantees
                    // this, but we keep a defensive check for forward compatibility.
                    if (g.GgmlType != gQW0.GgmlType || g.Ne0 != gQW0.Ne0 || g.Ne1 != gQW0.Ne1 ||
                        u.GgmlType != uQW0.GgmlType || u.Ne0 != uQW0.Ne0 || u.Ne1 != uQW0.Ne1 ||
                        d.GgmlType != dQW0.GgmlType || d.Ne0 != dQW0.Ne0 || d.Ne1 != dQW0.Ne1)
                    {
                        allQuantized = false;
                        break;
                    }

                    _moeGatePtrs[k] = g.CacheKey;
                    _moeUpPtrs[k] = u.CacheKey;
                    _moeDownPtrs[k] = d.CacheKey;
                }

                if (allQuantized)
                {
                    long t0exp = Stopwatch.GetTimestamp();
                    GgmlBasicOps.MoEExpertsSwiGLUForward(
                        _moeBatchedResult, tokenInput,
                        _numExpertsUsed,
                        _moeGatePtrs, _moeUpPtrs, _moeDownPtrs,
                        gQW0.GgmlType, gQW0.Ne0, gQW0.Ne1, gQW0.RawBytes,
                        uQW0.GgmlType, uQW0.Ne0, uQW0.Ne1, uQW0.RawBytes,
                        dQW0.GgmlType, dQW0.Ne0, dQW0.Ne1, dQW0.RawBytes,
                        routeW);
                    _linearTicks += Stopwatch.GetTimestamp() - t0exp;

                    InvalidateTensorDeviceCache(_moeBatchedResult);
                    float* batchedPtr = GetFloatPtr(_moeBatchedResult);
                    VecScaleAdd(outRow, batchedPtr, 1.0f, hiddenSize);
                    return;
                }
            }

            // Fallback: per-expert dispatch using reusable scratch tensors.
            for (int k = 0; k < _numExpertsUsed; k++)
            {
                int e = topExperts[k];
                if (!ExpertLinearForwardInto(_moeGateBuf, tokenInput, layer, e, kind: 0))
                    continue;
                if (!ExpertLinearForwardInto(_moeUpBuf, tokenInput, layer, e, kind: 1))
                    continue;

                Ops.SiLUMul(_moeGateBuf, _moeGateBuf, _moeUpBuf);

                if (!ExpertLinearForwardInto(_moeDownBuf, _moeGateBuf, layer, e, kind: 2))
                    continue;

                float w = routeW[k];
                float* downPtr = GetFloatPtr(_moeDownBuf);
                VecScaleAdd(outRow, downPtr, w, hiddenSize);
            }
        }

        private unsafe void RunMoEExpertsAllocating(Tensor tokenInput, int layer,
            int[] topExperts, float[] routeW, float* outRow, int hiddenSize)
        {
            for (int k = 0; k < _numExpertsUsed; k++)
            {
                int e = topExperts[k];
                Tensor gate = ExpertLinearForwardAlloc(tokenInput, layer, e, kind: 0);
                Tensor up = ExpertLinearForwardAlloc(tokenInput, layer, e, kind: 1);
                if (gate == null || up == null)
                {
                    gate?.Dispose();
                    up?.Dispose();
                    continue;
                }

                Ops.SiLUMul(gate, gate, up);
                up.Dispose();

                Tensor down = ExpertLinearForwardAlloc(gate, layer, e, kind: 2);
                gate.Dispose();
                if (down == null)
                    continue;

                float w = routeW[k];
                float* downPtr = GetFloatPtr(down);
                VecScaleAdd(outRow, downPtr, w, hiddenSize);
                down.Dispose();
            }
        }

        /// <summary>
        /// Linear forward that allocates the result tensor, using cached expert weight
        /// references (kind: 0=gate, 1=up, 2=down). Used by prefill where the output rows
        /// vary per token. Returns null if the weight is missing.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private Tensor ExpertLinearForwardAlloc(Tensor input, int layer, int expert, int kind)
        {
            long t0 = Stopwatch.GetTimestamp();
            QuantizedWeight qw = kind == 0 ? _expertGateQW[layer][expert]
                                : kind == 1 ? _expertUpQW[layer][expert]
                                            : _expertDownQW[layer][expert];

            if (qw != null)
            {
                Tensor result = new Tensor(_allocator, DType.Float32, input.Sizes[0], qw.Ne1);
                if (IsGgmlBackend)
                    GgmlBasicOps.AddmmQuant(result, input, qw.CacheKey, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes);
                else
                    AddmmQuantManaged(result, input, qw);
                if (qw.Scale != 1.0f)
                    Ops.Mul(result, result, qw.Scale); // NVFP4 scale2 sidecar
                _linearTicks += Stopwatch.GetTimestamp() - t0;
                return result;
            }

            Tensor w = kind == 0 ? _expertGateF32[layer][expert]
                     : kind == 1 ? _expertUpF32[layer][expert]
                                 : _expertDownF32[layer][expert];
            if (w != null)
            {
                Tensor result = new Tensor(_allocator, DType.Float32, input.Sizes[0], w.Sizes[0]);
                using var wT = w.Transpose();
                Ops.Addmm(result, 0, result, 1.0f, input, wT);
                _linearTicks += Stopwatch.GetTimestamp() - t0;
                return result;
            }

            return null;
        }

        /// <summary>
        /// LinearForward variant that takes pre-resolved weight references, eliminating the
        /// dictionary lookup that dominates the hot decode path for layer-shared weights
        /// such as MoE routers and shared expert projections.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        /// <summary>Debug-only: time a representative quantized matmul at a given
        /// row count to isolate the matmul's batch scaling (independent of the
        /// recurrent/attention layers). Forces a device sync per measured batch.</summary>
        public double DebugTimeQuantMatmul(string which, int rows, int reps, out int ggmlType, out long inDim, out long outDim)
        {
            QuantizedWeight qw = which switch
            {
                "down" => _ffnDownQW[0],
                "lmhead" => _lmHeadQW,
                _ => _ffnGateUpQW[0],
            };
            ggmlType = qw.GgmlType;
            inDim = qw.Ne0;
            outDim = qw.Ne1;
            using var input = new Tensor(_allocator, DType.Float32, rows, (int)qw.Ne0);
            Ops.Fill(input, 0.02f);
            using var result = new Tensor(_allocator, DType.Float32, rows, (int)qw.Ne1);
            AddmmQuantManaged(result, input, qw);
            result.GetElementsAsFloat(1); // drain warmup
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < reps; i++)
                AddmmQuantManaged(result, input, qw);
            result.GetElementsAsFloat(1); // single drain captures all queued reps
            sw.Stop();
            return sw.Elapsed.TotalMilliseconds / reps;
        }

        private Tensor LinearForwardCached(Tensor input, QuantizedWeight qw, Tensor wF32)
        {
            long t0 = Stopwatch.GetTimestamp();
            Tensor result;

            if (qw != null)
            {
                int seqLen = (int)input.Sizes[0];
                int outDim = (int)qw.Ne1;
                result = new Tensor(_allocator, DType.Float32, seqLen, outDim);
                if (IsGgmlBackend)
                    GgmlBasicOps.AddmmQuant(result, input, qw.CacheKey, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes);
                else
                    AddmmQuantManaged(result, input, qw);
                if (qw.Scale != 1.0f)
                    Ops.Mul(result, result, qw.Scale); // sidecar per-tensor scale2
            }
            else if (wF32 != null)
            {
                int outDimF32 = (int)wF32.Sizes[0];
                int seqLenF32 = (int)input.Sizes[0];
                using var wT = wF32.Transpose();
                result = new Tensor(_allocator, DType.Float32, seqLenF32, outDimF32);
                Ops.Addmm(result, 0, result, 1.0f, input, wT);
            }
            else
            {
                return null;
            }

            _linearTicks += Stopwatch.GetTimestamp() - t0;
            return result;
        }

        /// <summary>
        /// Linear forward into a pre-allocated result tensor using cached expert weight
        /// references (kind: 0=gate, 1=up, 2=down). Returns false if the weight is missing.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool ExpertLinearForwardInto(Tensor result, Tensor input, int layer, int expert, int kind)
        {
            long t0 = Stopwatch.GetTimestamp();
            QuantizedWeight qw = kind == 0 ? _expertGateQW[layer][expert]
                                : kind == 1 ? _expertUpQW[layer][expert]
                                            : _expertDownQW[layer][expert];

            if (qw != null)
            {
                if (IsGgmlBackend)
                    GgmlBasicOps.AddmmQuant(result, input, qw.CacheKey, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes);
                else
                    AddmmQuantManaged(result, input, qw);
                _linearTicks += Stopwatch.GetTimestamp() - t0;
                return true;
            }

            Tensor w = kind == 0 ? _expertGateF32[layer][expert]
                     : kind == 1 ? _expertUpF32[layer][expert]
                                 : _expertDownF32[layer][expert];
            if (w != null)
            {
                using var wT = w.Transpose();
                Ops.Addmm(result, 0, result, 1.0f, input, wT);
                _linearTicks += Stopwatch.GetTimestamp() - t0;
                return true;
            }

            return false;
        }

        private static unsafe void SelectTopKInPlace(float* values, int n, int k, int[] indices) =>
            TensorComputePrimitives.SelectTopKInPlace(values, n, k, indices);

        private unsafe void SelectTopKRouteWeights(float* routeRow, bool routeRowIsLogits, int[] topExperts, float[] routeWeights)
        {
            SelectTopKInPlace(routeRow, _numExperts, _numExpertsUsed, topExperts);

            if (routeRowIsLogits)
            {
                if (_normTopKProb)
                {
                    float maxLogit = float.NegativeInfinity;
                    for (int k = 0; k < _numExpertsUsed; k++)
                    {
                        float v = routeRow[topExperts[k]];
                        if (v > maxLogit)
                            maxLogit = v;
                    }

                    float sum = 0f;
                    for (int k = 0; k < _numExpertsUsed; k++)
                    {
                        float w = MathF.Exp(routeRow[topExperts[k]] - maxLogit);
                        routeWeights[k] = w;
                        sum += w;
                    }

                    if (sum > 0f)
                    {
                        float inv = 1.0f / sum;
                        for (int k = 0; k < _numExpertsUsed; k++)
                            routeWeights[k] *= inv;
                    }
                    return;
                }

                float fullMax = float.NegativeInfinity;
                for (int i = 0; i < _numExperts; i++)
                    if (routeRow[i] > fullMax)
                        fullMax = routeRow[i];

                float denom = 0f;
                for (int i = 0; i < _numExperts; i++)
                    denom += MathF.Exp(routeRow[i] - fullMax);

                float invDenom = denom > 0f ? 1.0f / denom : 0f;
                for (int k = 0; k < _numExpertsUsed; k++)
                    routeWeights[k] = MathF.Exp(routeRow[topExperts[k]] - fullMax) * invDenom;
                return;
            }

            float wSum = 0f;
            for (int k = 0; k < _numExpertsUsed; k++)
            {
                routeWeights[k] = routeRow[topExperts[k]];
                wSum += routeWeights[k];
            }

            if (_normTopKProb && wSum > 0f)
            {
                float inv = 1.0f / wSum;
                for (int k = 0; k < _numExpertsUsed; k++)
                    routeWeights[k] *= inv;
            }
        }

        #endregion

        #region Vision Support

        public void LoadVisionEncoder(string mmProjPath)
        {
            VisionEncoder = new Qwen35VisionEncoder(mmProjPath, _allocator);
            VisionEncoder.SetHostModel(this);
        }

        public void SetVisionEmbeddings(Tensor visionEmbeddings, int startPosition)
        {
            _visionEmbeddingsList.Add((visionEmbeddings, startPosition));
        }

        /// <summary>
        /// Inject vision embeddings into text embeddings at the image_pad token positions.
        /// </summary>
        private unsafe void InjectVisionEmbeddings(Tensor textEmbeddings, int seqLen)
        {
            if (_visionEmbeddingsList.Count == 0)
                return;

            float* textPtr = GetFloatPtr(textEmbeddings);
            int dim = Config.HiddenSize;
            foreach (var (visionEmbeddings, startPos) in _visionEmbeddingsList)
            {
                if (visionEmbeddings == null || startPos < 0)
                    continue;

                int numVisionTokens = (int)visionEmbeddings.Sizes[0];
                int projDim = (int)visionEmbeddings.Sizes[1];

                if (projDim != dim)
                {
                    Console.WriteLine($"Warning: Vision embedding dim ({projDim}) != text hidden dim ({dim}). Skipping injection.");
                    visionEmbeddings.Dispose();
                    continue;
                }

                if (startPos + numVisionTokens > seqLen)
                {
                    Console.WriteLine($"Warning: Vision tokens ({numVisionTokens}) exceed sequence at position {startPos}. Skipping.");
                    visionEmbeddings.Dispose();
                    continue;
                }

                float* visPtr = GetFloatPtr(visionEmbeddings);
                int bytes = numVisionTokens * dim * sizeof(float);
                Buffer.MemoryCopy(visPtr, textPtr + startPos * dim, bytes, bytes);

                Console.WriteLine($"  Injected {numVisionTokens} vision embeddings at position {startPos}");
                visionEmbeddings.Dispose();
            }

            _visionEmbeddingsList.Clear();
        }

        #endregion

        public override void PrintTimingStats()
        {
            if (_backend == BackendType.Mlx && _forwardCount > 0)
            {
                double totalMs = _forwardSw.Elapsed.TotalMilliseconds;
                double msPerTick = 1000.0 / Stopwatch.Frequency;
                double linearMs = _linearTicks * msPerTick;
                double attnMs = _attnTicks * msPerTick;
                double normMs = _normTicks * msPerTick;
                double embMs = _embTicks * msPerTick;
                double lmHeadMs = _lmHeadTicks * msPerTick;
                double logitsCopyMs = _logitsCopyTicks * msPerTick;
                double evalMs = _mlxEvalBoundaryTicks * msPerTick;
                double cacheEvalMs = _mlxCacheEvalTicks * msPerTick;
                double otherMs = totalMs - linearMs - attnMs - normMs - evalMs - cacheEvalMs;

                Console.WriteLine($"Timing ({_forwardCount} forward calls, {totalMs:F0} ms total, {totalMs / _forwardCount:F0} ms/token):");
                Console.WriteLine($"  Linear graph build: {linearMs:F0} ms ({100 * linearMs / totalMs:F1}%)");
                Console.WriteLine($"  Attention CPU/ops: {attnMs:F0} ms ({100 * attnMs / totalMs:F1}%)");
                Console.WriteLine($"  Norm:              {normMs:F0} ms ({100 * normMs / totalMs:F1}%)");
                Console.WriteLine($"  MLX graph eval:    {evalMs:F0} ms ({100 * evalMs / totalMs:F1}%, interval={MlxEvalEveryNLayers})");
                Console.WriteLine($"  MLX cache eval:    {cacheEvalMs:F0} ms ({100 * cacheEvalMs / totalMs:F1}%)");
                Console.WriteLine($"  (LM head:          {lmHeadMs:F0} ms, included in Linear/eval)");
                Console.WriteLine($"  (Embedding:        {embMs:F0} ms, in Other)");
                Console.WriteLine($"  (Logits copy:      {logitsCopyMs:F0} ms, in Other/eval)");
                Console.WriteLine($"  Other:             {otherMs:F0} ms ({100 * otherMs / totalMs:F1}%)");
            }
            else
            {
                base.PrintTimingStats();
            }

            if (_backend != BackendType.Mlx && _mlxEvalBoundaryTicks != 0)
            {
                double ms = _mlxEvalBoundaryTicks * 1000.0 / Stopwatch.Frequency;
                Console.WriteLine($"  (MLX layer eval: {ms:F0} ms, interval={MlxEvalEveryNLayers})");
            }
            if (_backend != BackendType.Mlx && _mlxCacheEvalTicks != 0)
            {
                double ms = _mlxCacheEvalTicks * 1000.0 / Stopwatch.Frequency;
                Console.WriteLine($"  (MLX cache eval: {ms:F0} ms, interval={MlxEvalEveryNLayers})");
            }
            PrintGdnTimingStats();
            PrintTpTimingStats();
        }

        protected override void OnBeforeReleaseGgmlDeviceResidency()
        {
            // Arena slots hold the only current KV/GDN state; flush them to
            // host before the residency wipe frees the resident copies.
            GgmlBasicOps.Qwen35ArenaResetBatchedDecodeCache();
            // Whole-model Qwen graphs pin weight buffers. Preserve mutable
            // device-authoritative state, then drop every graph family before
            // ModelBase evicts the corresponding weight bindings.
            EnsureKvCacheHostSynchronized();
            EnsureFusedDecodeStateHostSynchronized();
            InvalidateFullDecodeState(hardBindings: true);
            InvalidateVerifyCache();
            GgmlBasicOps.Qwen35ResetBatchedDecodeCache();
            GgmlBasicOps.Qwen35ReleaseVerifyTpGraphs();
            GgmlBasicOps.Qwen35ReleaseAttentionTpGraphs();
            GgmlBasicOps.Qwen35GdnDropTpGraphs();
        }

        public override void Dispose()
        {
            // Native whole-model graphs retain backend buffers and weight/cache
            // bindings. Release them while the model tensors and Metal backend are
            // still alive; otherwise Metal residency sets retain allocations until
            // process teardown.
            if (IsGgmlBackend)
            {
                GgmlBasicOps.Qwen35ResetDecodeCache();
                // The arena batched-decode pool survives everything else by
                // design; drop it (with a dirty-slot flush) at teardown or its
                // per-entry buffers keep their VRAM until process exit.
                GgmlBasicOps.Qwen35ArenaResetBatchedDecodeCache();
                GgmlBasicOps.Qwen35ResetVerifyCache();
                GgmlBasicOps.Qwen35ResetBatchedDecodeCache();
                GgmlBasicOps.Qwen35ReleaseVerifyTpGraphs();
                GgmlBasicOps.Qwen35ReleaseAttentionTpGraphs();
                GgmlBasicOps.Qwen35GdnDropTpGraphs();
            }

            // Cached CUDA prefill graphs own pool blocks + pinned tensors; free
            // them before the tensors/caches they reference are torn down.
            _cudaPrefillGraphs?.Dispose();
            _cudaPrefillGraphs = null;
            _cudaDecodeDynParams?.Dispose();
            _cudaDecodeDynParams = null;

            // Free the on-device MoE decode pointer tables (device u64 buffers).
            FreeQwenCudaMoETables();

            DisposeDFlash();

            VisionEncoder?.Dispose();
            foreach (var (visionEmbeddings, _) in _visionEmbeddingsList)
                visionEmbeddings?.Dispose();
            _visionEmbeddingsList.Clear();

            // Free per-request fused-decode holders (concurrent N>=2 decode); the
            // active holder shares _kvCacheK / _deltaStateTensor / _fdConvScratch,
            // freed by the teardown below.
            DisposeAllFusedHolders();

            if (_kvCacheK != null)
                foreach (var t in _kvCacheK) t?.Dispose();
            if (_kvCacheV != null)
                foreach (var t in _kvCacheV) t?.Dispose();
            if (_mlxAttentionCache != null)
                foreach (var cache in _mlxAttentionCache) cache?.Dispose();

            DisposeGdnState();
            DisposeTpState();

            _moeTokenInput?.Dispose();
            _moeGateBuf?.Dispose();
            _moeUpBuf?.Dispose();
            _moeDownBuf?.Dispose();
            _moeBatchedResult?.Dispose();
            _moeBatchedGate?.Dispose();
            _moeBatchedUp?.Dispose();
            _moeBatchedDown?.Dispose();
            _moeBatchedExpertIndices?.Dispose();
            _moeBatchedRouteWeights?.Dispose();

            _attnDecodeQBuf?.Dispose();
            _attnDecodeGBuf?.Dispose();
            _attnDecodeOutBuf?.Dispose();
            _attnDecodeQkvBuf?.Dispose();
            _ffnDecodeGateUpBuf?.Dispose();

            base.Dispose();
        }
    }
}
