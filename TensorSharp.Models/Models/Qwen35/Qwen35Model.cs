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
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.GGML;

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
    public class Qwen35Model : ModelBase
    {
        private bool[] _isRecurrent;
        private int _fullAttentionInterval;

        // MoE configuration (qwen35moe / qwen3next variants)
        private int _numExperts;
        private int _numExpertsUsed;
        private int _expertFfnLength;
        private int _sharedExpertFfnLength;
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
        // [1, packedDim] - reused fused norm + input proj output for GatedDeltaNet decode.
        private Tensor _gdnDecodePackedBuf;

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

        private static int ResolveFusedAttnLayerMinSeqLen()
        {
            string env = Environment.GetEnvironmentVariable("FUSED_ATTN_LAYER_MIN_SEQ_LEN");
            if (!string.IsNullOrWhiteSpace(env) && int.TryParse(env, out int v) && v > 0)
                return v;
            return 4096;
        }

        // Minimum prefill length at which the fused chunked GatedDeltaNet GGML kernel
        // beats the per-token CPU loop. The chunked kernel pads to a multiple of
        // GdnChunkSize (64) and dispatches once per layer; for very short prefills the
        // padding overhead and the host->device copies dominate. Set
        // GDN_CHUNK_PREFILL_MIN_SEQ_LEN=N to override (e.g. =1 to always use it,
        // =1000000 to disable).
        private static readonly int GdnChunkedPrefillMinSeqLenEnv = ResolveGdnChunkedPrefillMinSeqLen();

        private static int ResolveGdnChunkedPrefillMinSeqLen()
        {
            string env = Environment.GetEnvironmentVariable("GDN_CHUNK_PREFILL_MIN_SEQ_LEN");
            if (!string.IsNullOrWhiteSpace(env) && int.TryParse(env, out int v) && v > 0)
                return v;
            return GdnChunkSize;
        }

        private static readonly bool GdnChunkedPrefillDisabledEnv =
            string.Equals(Environment.GetEnvironmentVariable("GDN_DISABLE_CHUNKED_PREFILL"), "1",
                StringComparison.Ordinal);

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
        private string[] _ssmInProjKey;
        private string[] _attnQkvRecKey;
        private string[] _attnGateRecKey;
        private string[] _ssmBetaKey;
        private string[] _ssmAlphaKey;
        private string[] _ssmConv1dKey;
        private string[] _ssmDtBiasKey;
        private string[] _ssmAKey;
        private string[] _ssmNormKey;
        private string[] _ssmOutKey;

        // Pre-cached pointers/tensors for recurrent layers
        private Tensor[] _ssmConv1dW;
        private Tensor[] _ssmDtBiasW;
        private Tensor[] _ssmAW;
        private Tensor[] _ssmNormW;

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
        private Tensor[] _ffnGateUpF32;
        private QuantizedWeight[] _ffnDownQW;
        private Tensor[] _ffnDownF32;

        // Pre-resolved weight references for GatedDeltaNet linear projections.
        // These eliminate dictionary lookups in the per-layer hot path during decode.
        private QuantizedWeight[] _ssmInProjQW;
        private Tensor[] _ssmInProjF32;
        private QuantizedWeight[] _attnQkvRecQW;
        private Tensor[] _attnQkvRecF32;
        private QuantizedWeight[] _attnGateRecQW;
        private Tensor[] _attnGateRecF32;
        private QuantizedWeight[] _ssmBetaQW;
        private Tensor[] _ssmBetaF32;
        private QuantizedWeight[] _ssmAlphaQW;
        private Tensor[] _ssmAlphaF32;
        private QuantizedWeight[] _ssmOutQW;
        private Tensor[] _ssmOutF32;

        // Full attention KV cache (only for attention layers)
        private Tensor[] _kvCacheK;
        private Tensor[] _kvCacheV;
        private int _kvCacheCapacity;

        // GatedDeltaNet state (only for recurrent layers).
        // _convState uses a circular buffer indexed by _convStateWriteIdx[layer] to avoid
        // O(convDim*qkvDim) Array.Copy per token in the recurrent step.
        private float[][] _convState;  // [layer][convChannels * (convKernelSize-1)]
        private int[] _convStateWriteIdx;
        private Tensor[] _deltaStateTensor; // [layer]: Tensor[numVHeads, headVDim, headKDim]

        // SSM dimensions
        private int _ssmDInner;   // headVDim * numVHeads
        private int _ssmDState;   // headKDim
        private int _ssmNGroup;   // numKHeads
        private int _ssmDtRank;   // numVHeads
        private int _convKernel;
        private int _headVDim;
        private int _headKDim;
        private int _numVHeads;
        private int _numKHeads;
        private int _ropeDimCount;

        // MRoPE sections
        private int[] _mropeSections;

        // Pre-computed RoPE frequency table
        private float[] _ropeFreqs;

        // Pre-allocated work buffers for GatedDeltaNet
        private float[] _gdnQ, _gdnK, _gdnV;
        private float[] _gdnQExp, _gdnKExp;
        private float[] _gdnDelta, _gdnCore;

        // Transposed convolution weights laid out [kernelSize, qkvDim] for cache-friendly SIMD
        // access along the channel dimension while iterating over kernel taps.
        private float[][] _gdnConvWT;
        // Whether to use parallel per-head update in GatedDeltaNetStep.
        private bool _gdnParallelHeads;

        // Pre-allocated tensor work buffers for GatedDeltaNet state update.
        // _gdnConvOutBuf is a managed scratch array (no GGML allocation needed).
        private float[] _gdnConvOutBuf; // [qkvDim]
        private Tensor _gdnGatedOutT;   // [1, ssmDInner] (passed to LinearForward, must be a Tensor)

        // Chunked GatedDeltaNet (prefill) acceleration. The chunked path runs the entire
        // recurrent block as a single fused GGML graph dispatch per layer (via
        // GgmlBasicOps.GatedDeltaNetChunked) which moves L2Norm / mul_mat / triangular solve
        // / RMSNorm onto the GPU backend. CPU-side conv1d runs upstream and writes Q/K/V into
        // these reusable [seqLen, H, D] staging buffers.
        //
        // The path is enabled only when:
        //   * the runtime backend is GGML (Metal/CUDA/CPU)
        //   * headKDim == headVDim (chunked kernel pre-condition)
        //   * seqLen >= _gdnChunkPrefillThreshold
        //   * the kernel has not previously failed for this run (kill switch)
        private const int GdnChunkSize = 64;
        private int _gdnChunkPrefillThreshold = GdnChunkedPrefillMinSeqLenEnv;
        private bool _gdnDisableChunkedPrefill = GdnChunkedPrefillDisabledEnv;
        private long _gdnChunkedTicks;       // Total time spent in the chunked path
        private long _gdnPerTokenTicks;      // Total time spent in the per-token path (prefill only)
        private int _gdnChunkedCalls;        // Number of times the chunked path has been used

        // Reusable staging buffers for the chunked prefill path. These are sized for the
        // largest seqLen we have seen so far; subsequent calls reuse the same memory and
        // build a transient sub-view at the actual seqLen, avoiding (numLayers) allocator
        // round-trips per forward pass.
        private Tensor _gdnChunkedQBuf;
        private Tensor _gdnChunkedKBuf;
        private Tensor _gdnChunkedVBuf;
        private Tensor _gdnChunkedZBuf;
        private Tensor _gdnChunkedAlphaBuf;
        private Tensor _gdnChunkedBetaBuf;
        private int _gdnChunkedBufCapacity; // SeqLen capacity covered by the staging buffers
        private int _gdnPerTokenCalls;       // Number of times the per-token path has been used (excluding decode)

        // Vision encoder
        public Qwen35VisionEncoder VisionEncoder { get; private set; }
        private List<(Tensor embeddings, int position)> _visionEmbeddingsList = new();

        public Qwen35Model(string ggufPath, BackendType backend)
            : base(ggufPath, backend)
        {
            string arch = _gguf.GetString("general.architecture") ?? "qwen35";
            Config = new ModelConfig { Architecture = arch };
            ParseBaseConfig();

            // Qwen3.5 uses per-layer KV head counts; get the first non-zero value
            if (_gguf.Metadata.TryGetValue($"{arch}.attention.head_count_kv", out var hckvVal))
                Config.NumKVHeads = Convert.ToInt32(hckvVal);
            else
                Config.NumKVHeads = Config.NumHeads;

            // SSM config
            _ssmDInner = (int)_gguf.GetUint32($"{arch}.ssm.inner_size");
            _ssmDState = (int)_gguf.GetUint32($"{arch}.ssm.state_size");
            _ssmNGroup = (int)_gguf.GetUint32($"{arch}.ssm.group_count");
            _ssmDtRank = (int)_gguf.GetUint32($"{arch}.ssm.time_step_rank");
            _convKernel = (int)_gguf.GetUint32($"{arch}.ssm.conv_kernel");
            _fullAttentionInterval = (int)_gguf.GetUint32($"{arch}.full_attention_interval", 4);

            _numVHeads = _ssmDtRank;
            _numKHeads = _ssmNGroup;
            _headVDim = _ssmDInner / _numVHeads;
            _headKDim = _ssmDState;

            // MRoPE sections
            var sections = _gguf.GetInt32Array($"{arch}.rope.dimension_sections");
            _mropeSections = sections;
            _ropeDimCount = (int)_gguf.GetUint32($"{arch}.rope.dimension_count", 0);

            // MoE configuration: present for qwen35moe / qwen3next variants
            _numExperts = (int)_gguf.GetUint32($"{arch}.expert_count", 0);
            _numExpertsUsed = (int)_gguf.GetUint32($"{arch}.expert_used_count", 0);
            _expertFfnLength = (int)_gguf.GetUint32($"{arch}.expert_feed_forward_length", 0);
            _sharedExpertFfnLength = (int)_gguf.GetUint32($"{arch}.expert_shared_feed_forward_length", (uint)_expertFfnLength);
            // qwen3 MoE renormalizes the selected top-K probabilities by default.
            _normTopKProb = true;
            Config.NumExperts = _numExperts;
            Config.NumExpertsUsed = _numExpertsUsed;

            // Determine which layers are recurrent
            _isRecurrent = new bool[Config.NumLayers];
            for (int i = 0; i < Config.NumLayers; i++)
                _isRecurrent[i] = (i + 1) % _fullAttentionInterval != 0;

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

            if (_numExperts > 0)
                Console.WriteLine($"MoE: experts={_numExperts}, used={_numExpertsUsed}, " +
                    $"expertFFN={_expertFfnLength}, sharedFFN={_sharedExpertFfnLength}");

            LoadWeights();
            FuseAttentionProjectionWeights();
            FuseRecurrentInputWeights();
            FuseGateUpWeights();
            DetectMoeLayers();
            BuildLayerKeys();
            InitMoeBuffers();
            PrepareCudaQuantizedWeightsForInference();
            int maxContextLength = ResolveConfiguredContextLength();
            int initialCacheLength = ResolveInitialCacheAllocationLength(maxContextLength);
            if (initialCacheLength < maxContextLength)
                Console.WriteLine($"Initial CUDA cache allocation: {initialCacheLength} tokens (grows on demand up to {maxContextLength}).");
            InitCaches(initialCacheLength, maxContextLength);
            PrecomputeRoPE();
            InitGDNBuffers();
            CacheRecurrentWeights();

            // Heuristic: only parallelize per-head GDN work for models with many V-heads
            // (where the per-head work amortizes the parallel dispatch overhead).
            _gdnParallelHeads = _numVHeads >= 16 && Environment.ProcessorCount > 1;
        }

        private unsafe void FuseAttentionProjectionWeights()
        {
            int fused = 0;
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                if (_isRecurrent[layer])
                    continue;

                string prefix = $"blk.{layer}.";
                if (TryFuseWeights(prefix + "attn_qkv.weight",
                    prefix + "attn_q.weight",
                    prefix + "attn_k.weight",
                    prefix + "attn_v.weight"))
                {
                    fused++;
                }
            }

            if (fused > 0)
                Console.WriteLine($"  Fused projections: {fused} Q+K+V");
        }

        private unsafe void FuseRecurrentInputWeights()
        {
            int fused = 0;
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                if (!_isRecurrent[layer])
                    continue;

                string prefix = $"blk.{layer}.";
                if (TryFuseWeights(prefix + "ssm_in_proj.weight",
                    prefix + "attn_qkv.weight",
                    prefix + "attn_gate.weight",
                    prefix + "ssm_beta.weight",
                    prefix + "ssm_alpha.weight"))
                {
                    fused++;
                }
            }

            if (fused > 0)
                Console.WriteLine($"  Fused projections: {fused} recurrent input packs");
        }

        private unsafe bool TryFuseWeights(string fusedName, params string[] weightNames)
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
                    if (qw.GgmlType != first.GgmlType || qw.Ne0 != first.Ne0)
                        return false;

                    totalBytes += qw.RawBytes;
                    totalNe1 += qw.Ne1;
                }

                _quantWeights[fusedName] = QuantizedWeight.ConcatOrCreateCopy(quantWeights);
                for (int i = 0; i < weightNames.Length; i++)
                {
                    var name = weightNames[i];
                    var qw = quantWeights[i];
                    _quantWeights.Remove(name);
                    qw.Dispose();
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
            int numLayers = Config.NumLayers;
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

            int numLayers = Config.NumLayers;
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

        private void InitGDNBuffers()
        {
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;
            int qkDim = _headKDim * _numKHeads;
            int vDim = _headVDim * _numVHeads;
            _gdnQ = new float[qkDim];
            _gdnK = new float[qkDim];
            _gdnV = new float[vDim];
            _gdnQExp = new float[_headKDim * _numVHeads];
            _gdnKExp = new float[_headKDim * _numVHeads];
            _gdnDelta = new float[vDim];
            _gdnCore = new float[vDim];

            _gdnConvOutBuf = new float[qkvDim];
            _gdnGatedOutT = new Tensor(_allocator, DType.Float32, 1, _ssmDInner);

            // Pre-allocated fused norm + input projection output for GatedDeltaNet decode.
            // Shape matches the packed projection used in the recurrent block hot path.
            int packedDim = qkvDim + (_headVDim * _numVHeads) + _numVHeads * 2;
            if (packedDim > 0)
                _gdnDecodePackedBuf = new Tensor(_allocator, DType.Float32, 1, packedDim);
        }

        /// <summary>
        /// Pre-cache layer-prefix and weight name strings for every layer so the hot Forward()
        /// path never executes string interpolation. This eliminates dozens of allocations per
        /// layer per forward step and removes hundreds of dictionary string-hash lookups.
        /// </summary>
        private void BuildLayerKeys()
        {
            int n = Config.NumLayers;
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
            _ssmInProjKey = new string[n];
            _attnQkvRecKey = new string[n];
            _attnGateRecKey = new string[n];
            _ssmBetaKey = new string[n];
            _ssmAlphaKey = new string[n];
            _ssmConv1dKey = new string[n];
            _ssmDtBiasKey = new string[n];
            _ssmAKey = new string[n];
            _ssmNormKey = new string[n];
            _ssmOutKey = new string[n];

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
                _ssmInProjKey[l] = p + "ssm_in_proj.weight";
                _attnQkvRecKey[l] = p + "attn_qkv.weight";
                _attnGateRecKey[l] = p + "attn_gate.weight";
                _ssmBetaKey[l] = p + "ssm_beta.weight";
                _ssmAlphaKey[l] = p + "ssm_alpha.weight";
                _ssmConv1dKey[l] = p + "ssm_conv1d.weight";
                _ssmDtBiasKey[l] = p + "ssm_dt.bias";
                _ssmAKey[l] = p + "ssm_a";
                _ssmNormKey[l] = p + "ssm_norm.weight";
                _ssmOutKey[l] = p + "ssm_out.weight";
            }
        }

        /// <summary>
        /// Pre-resolve recurrent layer constant tensors and pre-compute a transposed conv1d
        /// weight layout that is friendly for SIMD vectorization across channels.
        /// </summary>
        private unsafe void CacheRecurrentWeights()
        {
            int n = Config.NumLayers;
            _ssmConv1dW = new Tensor[n];
            _ssmDtBiasW = new Tensor[n];
            _ssmAW = new Tensor[n];
            _ssmNormW = new Tensor[n];
            _gdnConvWT = new float[n][];

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
            _ffnDownQW = new QuantizedWeight[n];
            _ffnDownF32 = new Tensor[n];

            _ssmInProjQW = new QuantizedWeight[n];
            _ssmInProjF32 = new Tensor[n];
            _attnQkvRecQW = new QuantizedWeight[n];
            _attnQkvRecF32 = new Tensor[n];
            _attnGateRecQW = new QuantizedWeight[n];
            _attnGateRecF32 = new Tensor[n];
            _ssmBetaQW = new QuantizedWeight[n];
            _ssmBetaF32 = new Tensor[n];
            _ssmAlphaQW = new QuantizedWeight[n];
            _ssmAlphaF32 = new Tensor[n];
            _ssmOutQW = new QuantizedWeight[n];
            _ssmOutF32 = new Tensor[n];

            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;

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
                _quantWeights.TryGetValue(_ffnDownKey[l], out _ffnDownQW[l]);
                _weights.TryGetValue(_ffnDownKey[l], out _ffnDownF32[l]);

                if (_isRecurrent[l])
                {
                    _weights.TryGetValue(_ssmConv1dKey[l], out _ssmConv1dW[l]);
                    _weights.TryGetValue(_ssmDtBiasKey[l], out _ssmDtBiasW[l]);
                    _weights.TryGetValue(_ssmAKey[l], out _ssmAW[l]);
                    _weights.TryGetValue(_ssmNormKey[l], out _ssmNormW[l]);

                    _quantWeights.TryGetValue(_ssmInProjKey[l], out _ssmInProjQW[l]);
                    _weights.TryGetValue(_ssmInProjKey[l], out _ssmInProjF32[l]);
                    _quantWeights.TryGetValue(_attnQkvRecKey[l], out _attnQkvRecQW[l]);
                    _weights.TryGetValue(_attnQkvRecKey[l], out _attnQkvRecF32[l]);
                    _quantWeights.TryGetValue(_attnGateRecKey[l], out _attnGateRecQW[l]);
                    _weights.TryGetValue(_attnGateRecKey[l], out _attnGateRecF32[l]);
                    _quantWeights.TryGetValue(_ssmBetaKey[l], out _ssmBetaQW[l]);
                    _weights.TryGetValue(_ssmBetaKey[l], out _ssmBetaF32[l]);
                    _quantWeights.TryGetValue(_ssmAlphaKey[l], out _ssmAlphaQW[l]);
                    _weights.TryGetValue(_ssmAlphaKey[l], out _ssmAlphaF32[l]);
                    _quantWeights.TryGetValue(_ssmOutKey[l], out _ssmOutQW[l]);
                    _weights.TryGetValue(_ssmOutKey[l], out _ssmOutF32[l]);

                    if (_ssmConv1dW[l] != null)
                    {
                        // Stored as [qkvDim, kernelSize] (each row = filter for one channel).
                        // Transpose to [kernelSize, qkvDim] so that for a fixed kernel tap ki we
                        // access a contiguous block of channel weights, enabling SIMD across ch.
                        float* src = GetFloatPtr(_ssmConv1dW[l]);
                        var wt = new float[_convKernel * qkvDim];
                        for (int ch = 0; ch < qkvDim; ch++)
                            for (int ki = 0; ki < _convKernel; ki++)
                                wt[ki * qkvDim + ch] = src[ch * _convKernel + ki];
                        _gdnConvWT[l] = wt;
                    }
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
            int numLayers = Config.NumLayers;
            _kvCacheK = new Tensor[numLayers];
            _kvCacheV = new Tensor[numLayers];
            _convState = new float[numLayers][];
            _convStateWriteIdx = new int[numLayers];
            _deltaStateTensor = new Tensor[numLayers];

            int convDim = _convKernel - 1;
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;

            for (int l = 0; l < numLayers; l++)
            {
                if (!_isRecurrent[l])
                {
                    _kvCacheK[l] = new Tensor(_allocator, DType.Float32, Config.NumKVHeads, initialSeqLen, Config.HeadDim);
                    _kvCacheV[l] = new Tensor(_allocator, DType.Float32, Config.NumKVHeads, initialSeqLen, Config.HeadDim);
                    InitializeCacheTensor(_kvCacheK[l]);
                    InitializeCacheTensor(_kvCacheV[l]);
                }
                else
                {
                    _convState[l] = new float[convDim * qkvDim];
                    _convStateWriteIdx[l] = 0;
                    _deltaStateTensor[l] = new Tensor(_allocator, DType.Float32, _numVHeads, _headVDim, _headKDim);
                    Ops.Fill(_deltaStateTensor[l], 0);
                }
            }
            _cacheSeqLen = 0;
        }

        private void EnsureCacheCapacity(int requiredSeqLen)
        {
            if (requiredSeqLen <= _kvCacheCapacity)
                return;
            if (requiredSeqLen > _maxContextLength)
                throw new InvalidOperationException($"Requested sequence length {requiredSeqLen} exceeds configured max context {_maxContextLength}.");

            int newCapacity = Math.Max(_kvCacheCapacity, 1);
            while (newCapacity < requiredSeqLen)
                newCapacity = Math.Min(_maxContextLength, newCapacity * 2);

            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_isRecurrent[l])
                    continue;

                var newK = new Tensor(_allocator, DType.Float32, Config.NumKVHeads, newCapacity, Config.HeadDim);
                var newV = new Tensor(_allocator, DType.Float32, Config.NumKVHeads, newCapacity, Config.HeadDim);
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

                _kvCacheK[l].Dispose();
                _kvCacheV[l].Dispose();
                _kvCacheK[l] = newK;
                _kvCacheV[l] = newV;
            }

            _kvCacheCapacity = newCapacity;
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

        public override void ResetKVCache()
        {
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!_isRecurrent[l])
                {
                    ResetCacheTensor(_kvCacheK[l]);
                    ResetCacheTensor(_kvCacheV[l]);
                }
                else
                {
                    Array.Clear(_convState[l]);
                    _convStateWriteIdx[l] = 0;
                    Ops.Fill(_deltaStateTensor[l], 0);
                }
            }
            _cacheSeqLen = 0;
            _linearTicks = _attnTicks = _normTicks = _embTicks = _lmHeadTicks = _logitsCopyTicks = 0;
            _forwardCount = 0;
            _forwardSw.Reset();
        }

        public override bool SupportsKVCacheTruncation => false;

        public override float[] Forward(int[] tokens)
        {
            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            EnsureCacheCapacity(startPos + seqLen);

            long t1 = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t1;

            if (_visionEmbeddingsList.Count > 0 && startPos == 0)
                InjectVisionEmbeddings(hidden, seqLen);

            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                if (_isRecurrent[layer])
                    hidden = RecurrentBlock(hidden, layer, seqLen, startPos);
                else
                    hidden = AttentionBlock(hidden, layer, seqLen, startPos);
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
            _lmHeadTicks += Stopwatch.GetTimestamp() - t2;
            lastHiddenRaw.Dispose();

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
            _forwardSw.Stop();
            return _logitsBuffer;
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
            Tensor attnOut = fusedDecodeApplied
                ? null
                : FullAttention(hidden, _attnNormW[layer], layer, seqLen, startPos, residual: hidden);

            if (attnOut != null)
            {
                Ops.Add(hidden, hidden, attnOut);
                attnOut.Dispose();
            }

            // Fuse post-attn norm + first FFN/MoE projection. For MoE this fusion is not yet
            // applicable to the batched MoE kernel so we keep the explicit norm; for dense FFN
            // we route through FFNCachedFused which fuses norm + gate.
            Tensor ffnOut;
            if (_isMoeLayer != null && _isMoeLayer[layer])
            {
                // Decode hot path: do RMSNorm on CPU into the pre-allocated input buffer
                // (saves a Metal command buffer + a tensor allocation per MoE layer per token).
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
                {
                    normed2 = RMSNormOpCached(hidden, _postAttnNormW[layer]);
                }

                // Decode hot path: full MoE + shared expert + residual fused into one GGML graph.
                if (seqLen == 1 && TryMoEResidualDecode(hidden, normed2, layer))
                {
                    ffnOut = null;
                }
                else
                {
                    ffnOut = MoEForward(normed2, layer, seqLen);
                }

                if (ownsNormed2)
                    normed2.Dispose();
            }
            else
            {
                ffnOut = FFNCachedFused(hidden, _postAttnNormW[layer], layer, seqLen);
            }

            // For dense FFN we may have already accumulated into hidden via FusedMatMulQuantAdd;
            // in that case ffnOut is null.
            if (ffnOut != null)
            {
                Ops.Add(hidden, hidden, ffnOut);
                ffnOut.Dispose();
            }

            return hidden;
        }

        private unsafe Tensor FullAttention(Tensor input, Tensor inputNormW, int layer, int seqLen, int startPos,
            Tensor residual = null)
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

            qTensor = ApplyQKNormCached(qTensor, _attnQNormW[layer], numHeads, seqLen);
            kTensor = ApplyQKNormCached(kTensor, _attnKNormW[layer], numKVHeads, seqLen);

            // RoPE - decode path applies Q and K together so the cos/sin table is computed once.
            if (seqLen == 1)
            {
                ApplyRoPEDecodeQKInPlace(qTensor, kTensor, numHeads, numKVHeads, startPos);
            }
            else
            {
                qTensor = ApplyRoPEPrefill(qTensor, numHeads, seqLen, startPos);
                kTensor = ApplyRoPEPrefill(kTensor, numKVHeads, seqLen, startPos);
            }

            long t0 = Stopwatch.GetTimestamp();

            // Attention computation
            float attentionScale = 1.0f / MathF.Sqrt(headDim);
            Tensor attnOutput;

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
                if (IsGgmlBackend && totalSeqLen >= FlashAttnDecodeMinSeqLen)
                {
                    flashOk = TryFlashAttnDecode(qTensor, kTensor, vTensor,
                        _kvCacheK[layer], _kvCacheV[layer], attnOutput,
                        numHeads, numKVHeads, headDim, maxSeqLen, startPos, attentionScale);
                }

                if (!flashOk)
                {
                    CopyToCacheDecode(_kvCacheK[layer], kTensor, _kvCacheV[layer], vTensor,
                        numKVHeads, headDim, startPos);
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
                Tensor qHeads = ReshapeToHeads(qTensor, numHeads, seqLen, headDim);
                qTensor.Dispose();
                Tensor kHeads = ReshapeToHeads(kTensor, numKVHeads, seqLen, headDim);
                kTensor.Dispose();
                Tensor vHeads = ReshapeToHeads(vTensor, numKVHeads, seqLen, headDim);
                vTensor.Dispose();

                CopyToCache(_kvCacheK[layer], kHeads, startPos, seqLen);
                CopyToCache(_kvCacheV[layer], vHeads, startPos, seqLen);
                kHeads.Dispose();
                vHeads.Dispose();

                int groupSize = numHeads / numKVHeads;
                Tensor kExpanded = ExpandKVHeads(_kvCacheK[layer], groupSize, totalSeqLen);
                Tensor vExpanded = ExpandKVHeads(_kvCacheV[layer], groupSize, totalSeqLen);

                using var kT = kExpanded.Transpose(1, 2);
                var scores = new Tensor(_allocator, DType.Float32, numHeads, seqLen, totalSeqLen);
                Ops.AddmmBatch(scores, 0, scores, attentionScale, qHeads, kT);
                qHeads.Dispose();
                kExpanded.Dispose();

                Ops.AddCausalMask(scores, seqLen, startPos, float.NegativeInfinity);
                Ops.Softmax(scores, scores);

                var attnOut = new Tensor(_allocator, DType.Float32, numHeads, seqLen, headDim);
                Ops.AddmmBatch(attnOut, 0, attnOut, 1.0f, scores, vExpanded);
                scores.Dispose();
                vExpanded.Dispose();

                attnOutput = ReshapeFromHeads(attnOut, numHeads, seqLen, headDim);
                attnOut.Dispose();
            }

            // Decode hot path: do the sigmoid-gated mix on CPU. The data is tiny
            // (single row, numHeads * headDim floats) so the GPU dispatch overhead
            // dominates. Eliminates one Metal command buffer per attention layer.
            if (seqLen == 1 && attnOutput != null && gateTensor != null
                && attnOutput.ElementType == DType.Float32 && gateTensor.ElementType == DType.Float32)
                ApplySigmoidGateCpu(attnOutput, gateTensor);
            else
                ApplySigmoidGate(attnOutput, gateTensor);
            if (ownsQGateBuffers)
                gateTensor.Dispose();

            _attnTicks += Stopwatch.GetTimestamp() - t0;

            bool ownsAttnOutput = !(seqLen == 1 && _attnDecodeOutBuf != null && ReferenceEquals(attnOutput, _attnDecodeOutBuf));

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

            // Decode hot path: reuse pre-allocated buffers; caller must NOT dispose them.
            if (seqLen == 1 && _attnDecodeQBuf != null && _attnDecodeGBuf != null
                && _attnDecodeQBuf.Sizes[1] == totalPerToken)
            {
                qTensor = _attnDecodeQBuf;
                gateTensor = _attnDecodeGBuf;
                ownsBuffers = false;
            }
            else
            {
                qTensor = new Tensor(_allocator, DType.Float32, seqLen, totalPerToken);
                gateTensor = new Tensor(_allocator, DType.Float32, seqLen, totalPerToken);
                ownsBuffers = true;
            }

            float* src = GetFloatPtr(qFull);
            float* qDst = GetFloatPtr(qTensor);
            float* gDst = GetFloatPtr(gateTensor);
            int headBytes = headDim * sizeof(float);

            for (int s = 0; s < seqLen; s++)
            {
                float* srcRow = src + (long)s * numHeads * dimPerHead;
                float* qRow = qDst + (long)s * totalPerToken;
                float* gRow = gDst + (long)s * totalPerToken;
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
        private Tensor FFNCached(Tensor input, int layer, int seqLen)
        {
            int intermSize = Config.IntermediateSize;
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

            // Fused norm + gate_up projection. Decode reuses a pre-allocated [1, 2*intermSize]
            // buffer to avoid one tensor allocation per layer per token.
            Tensor gateUp = null;
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
            if (ownsGateUp)
                gateUp.Dispose();

            // Decode hot path: do SiLU * up on CPU. The data is small enough that a
            // single GPU dispatch costs more than the compute; eliminating one Metal
            // command buffer per FFN layer per token saves a few percent of decode time.
            if (seqLen == 1 && IsGgmlBackend)
                SiLUMulInPlaceCpu(gate, up);
            else
                Ops.SiLUMul(gate, gate, up);
            up.Dispose();

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
                _linearTicks += Stopwatch.GetTimestamp() - t0;
                return result;
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
            if (!IsGgmlBackend || qw == null || normW == null
                || input.DimensionCount != 2 || output == null
                || output.DimensionCount != 2 || output.Sizes[1] != qw.Ne1
                || output.Sizes[0] != input.Sizes[0])
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
            if (!IsGgmlBackend || qw == null || input.DimensionCount != 2 || residual.DimensionCount != 2)
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
                kCache.ElementType != DType.Float32 || vCache.ElementType != DType.Float32 || output.ElementType != DType.Float32)
                return false;

            try
            {
                GgmlBasicOps.FlashAttnDecode(q, k, v, kCache, vCache, output,
                    numHeads, numKVHeads, headDim, maxSeqLen, position, scale);

                // The kernel writes the output through the host pointer (unified memory on
                // Apple Silicon), but downstream GGML ops still need to know the buffer is
                // host-fresh so any cached device mirror is reloaded.
                InvalidateTensorDeviceCache(output);
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
            Tensor attnNorm = _attnNormW[layer];
            Tensor qNorm = _attnQNormW[layer];
            Tensor kNorm = _attnKNormW[layer];
            Tensor kCache = _kvCacheK[layer];
            Tensor vCache = _kvCacheV[layer];

            if (qkv == null || oOut == null || attnNorm == null || qNorm == null || kNorm == null
                || kCache == null || vCache == null)
                return false;

            int headDim = Config.HeadDim;
            int numHeads = Config.NumHeads;
            int numKVHeads = Config.NumKVHeads;
            int maxSeqLen = (int)kCache.Sizes[1];

            // The fused kernel uses NeoX RoPE (rope_mode = 2), matching the standalone path.
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
                    Config.Eps, Config.RopeBase, ropeFreqScale, ropeMode);
                _attnTicks += Stopwatch.GetTimestamp() - t0;

                // Output is written through the host pointer (unified memory). Downstream
                // GGML ops need a fresh device mirror so invalidate the cache for the residual
                // and the KV cache slabs that we just appended into.
                InvalidateTensorDeviceCache(residual);
                InvalidateTensorDeviceCache(kCache);
                InvalidateTensorDeviceCache(vCache);
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

            if (seqLen == 1)
            {
                // Decode hot path: normalize on CPU. The data and alpha tensors are tiny
                // (e.g. 16x256 + 256 floats) so the GPU dispatch overhead dominates the
                // compute. Going through SIMD eliminates ~2 dispatches per layer per token.
                RMSNormInPlaceCpu(data, alpha, numHeads, headDim, Config.Eps);
                return data;
            }

            using var reshaped = data.View(seqLen * numHeads, headDim);
            Tensor normed = Ops.RMSNorm(null, reshaped, alpha, null, Config.Eps);
            data.Dispose();

            Tensor result = normed.View(seqLen, numHeads * headDim);
            normed.Dispose();
            return result;
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
            int totalRows = seqLen * numHeads;
            int[] positions = new int[totalRows];
            for (int s = 0; s < seqLen; s++)
                for (int h = 0; h < numHeads; h++)
                    positions[s * numHeads + h] = startPos + s;
            using var posTensor = CreateIntTensor(positions, totalRows);

            using var reshaped = data.View(1, seqLen, numHeads, headDim);
            Tensor result = Ops.RoPEEx(
                null, reshaped, posTensor, ropeDim, 2, 0,
                Config.RopeBase, 1.0f / Config.RopeScale,
                0.0f, 1.0f, 0.0f, 0.0f);

            data.Dispose();
            Tensor flat = result.View(seqLen, numHeads * headDim);
            result.Dispose();
            return flat;
        }

        #endregion

        #region Recurrent (GatedDeltaNet) Block

        /// <summary>
        /// GatedDeltaNet recurrent block: SSM conv1d -> gated delta net -> norm + gate -> output.
        /// Both decode and prefill use the same recurrent core. Prefill batches the large
        /// input/output projections across the whole chunk, then walks the recurrent state
        /// token-by-token in CPU memory.
        /// </summary>
        private Tensor RecurrentBlock(Tensor hidden, int layer, int seqLen, int startPos)
        {
            // Fused pre-norm + GatedDeltaNet input projection inside GatedDeltaNet, and fused
            // output projection + residual via TryLinearAddInto when possible.
            Tensor attnOut = GatedDeltaNet(hidden, _attnNormW[layer], layer, seqLen, residual: hidden);
            if (attnOut != null)
            {
                Ops.Add(hidden, hidden, attnOut);
                attnOut.Dispose();
            }

            Tensor ffnOut;
            if (_isMoeLayer != null && _isMoeLayer[layer])
            {
                // Decode hot path: do RMSNorm on CPU into the pre-allocated input buffer.
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
                {
                    normed2 = RMSNormOpCached(hidden, _postAttnNormW[layer]);
                }

                if (seqLen == 1 && TryMoEResidualDecode(hidden, normed2, layer))
                {
                    ffnOut = null;
                }
                else
                {
                    ffnOut = MoEForward(normed2, layer, seqLen);
                }
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

        /// <summary>
        /// GatedDeltaNet recurrent step with batched input/output projections.
        /// Prefill projects the whole chunk once, then walks the recurrent state token-by-token.
        /// Decode follows the same path with seqLen=1, avoiding several tiny GGML dispatches.
        /// </summary>
        private unsafe Tensor GatedDeltaNet(Tensor input, Tensor inputNormW, int layer, int seqLen,
            Tensor residual = null)
        {
            long t0 = Stopwatch.GetTimestamp();
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;
            int qkDim = _headKDim * _numKHeads;
            int vDim = _headVDim * _numVHeads;
            int zDim = _headVDim * _numVHeads;
            int packedDim = qkvDim + zDim + _numVHeads * 2;

            // Fused input norm + packed input projection (single GGML kernel). Decode reuses
            // a pre-allocated [1, packedDim] buffer to avoid one tensor allocation per layer
            // per token.
            Tensor packedInput = null;
            bool ownsPackedInput = true;
            if (inputNormW != null && _ssmInProjQW[layer] != null && IsGgmlBackend)
            {
                if (seqLen == 1 && _gdnDecodePackedBuf != null
                    && _gdnDecodePackedBuf.Sizes[1] == _ssmInProjQW[layer].Ne1)
                {
                    packedInput = TryFusedNormLinearInto(_gdnDecodePackedBuf, input, inputNormW, _ssmInProjQW[layer]);
                    if (packedInput != null)
                        ownsPackedInput = false;
                }
                if (packedInput == null)
                    packedInput = FusedNormLinear(input, inputNormW, _ssmInProjQW[layer], _ssmInProjF32[layer]);
            }

            Tensor normedInput = null;
            if (packedInput == null)
            {
                normedInput = inputNormW != null ? RMSNormOpCached(input, inputNormW) : input.CopyRef();
                packedInput = LinearForwardCached(normedInput, _ssmInProjQW[layer], _ssmInProjF32[layer]);
            }

            Tensor qkvRaw = null;
            Tensor zRaw = null;
            Tensor betaRaw = null;
            Tensor alphaRaw = null;

            float* packedPtr = null;
            float* qkvBase = null;
            float* zBase = null;
            float* betaBase = null;
            float* alphaBase = null;

            if (packedInput != null)
            {
                packedPtr = GetFloatPtr(packedInput);
            }
            else
            {
                if (normedInput == null)
                    normedInput = inputNormW != null ? RMSNormOpCached(input, inputNormW) : input.CopyRef();

                qkvRaw = LinearForwardCached(normedInput, _attnQkvRecQW[layer], _attnQkvRecF32[layer]);
                zRaw = LinearForwardCached(normedInput, _attnGateRecQW[layer], _attnGateRecF32[layer]);
                betaRaw = LinearForwardCached(normedInput, _ssmBetaQW[layer], _ssmBetaF32[layer]);
                alphaRaw = LinearForwardCached(normedInput, _ssmAlphaQW[layer], _ssmAlphaF32[layer]);

                qkvBase = GetFloatPtr(qkvRaw);
                zBase = GetFloatPtr(zRaw);
                betaBase = GetFloatPtr(betaRaw);
                alphaBase = GetFloatPtr(alphaRaw);
            }

            // Pre-resolved layer constants (cached at construction; no dictionary lookup here).
            float* dtBiasPtr = GetFloatPtr(_ssmDtBiasW[layer]);
            float* aPtr = GetFloatPtr(_ssmAW[layer]);
            float* ssmNormPtr = GetFloatPtr(_ssmNormW[layer]);

            Tensor gated = seqLen == 1 ? _gdnGatedOutT : new Tensor(_allocator, DType.Float32, seqLen, _ssmDInner);
            float* gatedBase = GetFloatPtr(gated);

            float[] convWT = _gdnConvWT[layer];

            // Prefer the fused chunked GGML kernel when running on a GGML backend with
            // sufficient sequence length and matching K/V head dims. The chunked kernel
            // packs the entire delta-net block (L2Norm, mul_mat, triangular solve, RMSNorm,
            // gating) into a single GPU dispatch per layer, drastically reducing CPU work
            // during prefill.
            bool useChunked = !_gdnDisableChunkedPrefill
                && IsGgmlBackend
                && seqLen >= _gdnChunkPrefillThreshold
                && _headKDim == _headVDim
                && _ssmDtBiasW[layer] != null
                && _ssmAW[layer] != null
                && _ssmNormW[layer] != null;

            if (useChunked)
            {
                long tChunk = Stopwatch.GetTimestamp();
                bool chunkedOk = false;
                try
                {
                    GatedDeltaNetChunkedPrefill(
                        packedPtr, qkvBase, zBase, betaBase, alphaBase,
                        layer, seqLen, qkvDim, qkDim, zDim, packedDim, gated);
                    chunkedOk = true;
                }
                catch (Exception ex)
                {
                    // First failure trips the kill switch so subsequent layers / forwards
                    // do not pay the same overhead twice. Fall back to the per-token loop.
                    _gdnDisableChunkedPrefill = true;
                    Console.WriteLine($"[Qwen35] GatedDeltaNetChunked disabled (layer {layer}, seqLen {seqLen}): {ex.Message}");
                }

                if (chunkedOk)
                {
                    _gdnChunkedTicks += Stopwatch.GetTimestamp() - tChunk;
                    _gdnChunkedCalls++;
                }
                else
                {
                    // Fall back: clean state was already mutated only inside the helper. Run
                    // the per-token path on the same gated buffer.
                    long tFallback = Stopwatch.GetTimestamp();
                    RunPerTokenLoop(packedPtr, qkvBase, zBase, betaBase, alphaBase,
                        layer, seqLen, qkvDim, qkDim, vDim, zDim, packedDim,
                        convWT, dtBiasPtr, aPtr, ssmNormPtr, gatedBase);
                    _gdnPerTokenTicks += Stopwatch.GetTimestamp() - tFallback;
                    if (seqLen > 1) _gdnPerTokenCalls++;
                }
            }
            else
            {
                long tLoop = Stopwatch.GetTimestamp();
                RunPerTokenLoop(packedPtr, qkvBase, zBase, betaBase, alphaBase,
                    layer, seqLen, qkvDim, qkDim, vDim, zDim, packedDim,
                    convWT, dtBiasPtr, aPtr, ssmNormPtr, gatedBase);
                _gdnPerTokenTicks += Stopwatch.GetTimestamp() - tLoop;
                if (seqLen > 1) _gdnPerTokenCalls++;
            }

            InvalidateTensorDeviceCache(gated);

            // Fast path: fuse SSM output projection with the residual add.
            Tensor output;
            bool fusedAdd = false;
            if (residual != null
                && _ssmOutQW[layer] != null
                && residual.DimensionCount == 2
                && gated.DimensionCount == 2
                && residual.Sizes[0] == gated.Sizes[0]
                && TryLinearAddInto(residual, gated, _ssmOutQW[layer]))
            {
                output = null;
                fusedAdd = true;
            }
            else
            {
                output = LinearForwardCached(gated, _ssmOutQW[layer], _ssmOutF32[layer]);
            }

            if (seqLen > 1)
                gated.Dispose();

            normedInput?.Dispose();
            if (ownsPackedInput) packedInput?.Dispose();
            qkvRaw?.Dispose();
            zRaw?.Dispose();
            betaRaw?.Dispose();
            alphaRaw?.Dispose();

            _attnTicks += Stopwatch.GetTimestamp() - t0;
            return fusedAdd ? null : output;
        }

        /// <summary>
        /// Per-token recurrent loop that walks the chunk one input at a time. Used both
        /// for decode (seqLen=1) and as the chunked-path fallback for prefill.
        /// </summary>
        private unsafe void RunPerTokenLoop(
            float* packedPtr, float* qkvBase, float* zBase, float* betaBase, float* alphaBase,
            int layer, int seqLen, int qkvDim, int qkDim, int vDim, int zDim, int packedDim,
            float[] convWT, float* dtBiasPtr, float* aPtr, float* ssmNormPtr, float* gatedBase)
        {
            for (int s = 0; s < seqLen; s++)
            {
                float* qkvPtr;
                float* zPtr;
                float* betaPtr;
                float* alphaPtr;

                if (packedPtr != null)
                {
                    float* row = packedPtr + (long)s * packedDim;
                    qkvPtr = row;
                    zPtr = row + qkvDim;
                    betaPtr = zPtr + zDim;
                    alphaPtr = betaPtr + _numVHeads;
                }
                else
                {
                    qkvPtr = qkvBase + (long)s * qkvDim;
                    zPtr = zBase + (long)s * zDim;
                    betaPtr = betaBase + (long)s * _numVHeads;
                    alphaPtr = alphaBase + (long)s * _numVHeads;
                }

                GatedDeltaNetStep(qkvPtr, zPtr, betaPtr, alphaPtr,
                    layer, qkvDim, qkDim, vDim,
                    convWT, dtBiasPtr, aPtr, ssmNormPtr,
                    gatedBase + (long)s * _ssmDInner);
            }
        }

        /// <summary>
        /// Chunked GatedDeltaNet prefill path.
        ///
        /// Runs the conv1d step token-by-token on CPU (so the recurrent ring state is
        /// updated correctly), packs the per-token Q/K/V/Z/alpha/beta into row-major
        /// [seqLen, H, D] (or [seqLen, H]) tensors, and dispatches a single fused GGML
        /// graph that does L2Norm, Q-scale, sigmoid(beta), softplus(alpha), per-chunk
        /// (k @ q) attention with triangular solve, RMSNorm, and silu(z) gating - all on
        /// the GGML backend (Metal / CUDA when available).
        ///
        /// The recurrent state tensor (_deltaStateTensor[layer], shape [H, D, D]) is
        /// passed in as input/output. The kernel updates it in place on the device and
        /// downloads the new value back to the host buffer.
        /// </summary>
        private unsafe void GatedDeltaNetChunkedPrefill(
            float* packedPtr, float* qkvBase, float* zBase, float* betaBase, float* alphaBase,
            int layer, int seqLen, int qkvDim, int qkDim, int zDim, int packedDim,
            Tensor gated)
        {
            int H = _numVHeads;
            int Dk = _headKDim;
            int Dv = _headVDim;
            int hKDim = H * Dk;
            int hVDim = H * Dv;

            EnsureChunkedStagingBuffers(seqLen, H, Dk, Dv);

            // The staging tensors are sized for the largest seqLen seen so far. We work
            // on sub-views at the actual seqLen so the native kernel sees the correct shape.
            Tensor qBuf = _gdnChunkedQBuf.Narrow(0, 0, seqLen);
            Tensor kBuf = _gdnChunkedKBuf.Narrow(0, 0, seqLen);
            Tensor vBuf = _gdnChunkedVBuf.Narrow(0, 0, seqLen);
            Tensor zBuf = _gdnChunkedZBuf.Narrow(0, 0, seqLen);
            Tensor alphaBuf = _gdnChunkedAlphaBuf.Narrow(0, 0, seqLen);
            Tensor betaBuf = _gdnChunkedBetaBuf.Narrow(0, 0, seqLen);

            try
            {
                float* qPtr = GetFloatPtr(qBuf);
                float* kPtr = GetFloatPtr(kBuf);
                float* vPtr = GetFloatPtr(vBuf);
                float* zPtr = GetFloatPtr(zBuf);
                float* alphaPtr = GetFloatPtr(alphaBuf);
                float* betaPtr = GetFloatPtr(betaBuf);

                int convDim = _convKernel - 1;
                float[] convState = _convState[layer];
                float[] convWT = _gdnConvWT[layer];
                int writeIdx = _convStateWriteIdx[layer];

                fixed (float* convOutPtr = _gdnConvOutBuf)
                fixed (float* statePtr = convState)
                {
                    for (int s = 0; s < seqLen; s++)
                    {
                        float* qkvSrc;
                        float* zSrc;
                        float* betaSrc;
                        float* alphaSrc;

                        if (packedPtr != null)
                        {
                            float* row = packedPtr + (long)s * packedDim;
                            qkvSrc = row;
                            zSrc = row + qkvDim;
                            betaSrc = zSrc + zDim;
                            alphaSrc = betaSrc + _numVHeads;
                        }
                        else
                        {
                            qkvSrc = qkvBase + (long)s * qkvDim;
                            zSrc = zBase + (long)s * zDim;
                            betaSrc = betaBase + (long)s * _numVHeads;
                            alphaSrc = alphaBase + (long)s * _numVHeads;
                        }

                        ComputeConv1DStep(qkvSrc, qkvDim, convDim, writeIdx, convState, convWT, convOutPtr);

                        if (convDim > 0)
                        {
                            Buffer.MemoryCopy(qkvSrc, statePtr + writeIdx * qkvDim,
                                qkvDim * sizeof(float), qkvDim * sizeof(float));
                            writeIdx = (writeIdx + 1) % convDim;
                        }

                        long vBytes = (long)hVDim * sizeof(float);
                        Buffer.MemoryCopy(convOutPtr + 2 * qkDim, vPtr + (long)s * hVDim, vBytes, vBytes);
                        Buffer.MemoryCopy(zSrc,                   zPtr + (long)s * hVDim, vBytes, vBytes);

                        long aBytes = (long)H * sizeof(float);
                        Buffer.MemoryCopy(alphaSrc, alphaPtr + (long)s * H, aBytes, aBytes);
                        Buffer.MemoryCopy(betaSrc,  betaPtr  + (long)s * H, aBytes, aBytes);

                        if (_numKHeads == _numVHeads)
                        {
                            long kBytes = (long)hKDim * sizeof(float);
                            Buffer.MemoryCopy(convOutPtr,         qPtr + (long)s * hKDim, kBytes, kBytes);
                            Buffer.MemoryCopy(convOutPtr + qkDim, kPtr + (long)s * hKDim, kBytes, kBytes);
                        }
                        else
                        {
                            float* qDst = qPtr + (long)s * hKDim;
                            float* kDst = kPtr + (long)s * hKDim;
                            long perHeadBytes = (long)Dk * sizeof(float);
                            for (int h = 0; h < H; h++)
                            {
                                int srcHead = h % _numKHeads;
                                Buffer.MemoryCopy(convOutPtr + srcHead * Dk, qDst + h * Dk, perHeadBytes, perHeadBytes);
                                Buffer.MemoryCopy(convOutPtr + qkDim + srcHead * Dk, kDst + h * Dk, perHeadBytes, perHeadBytes);
                            }
                        }
                    }
                }
                _convStateWriteIdx[layer] = writeIdx;

                // Tell the GGML host-buffer cache that the staging buffers were written
                // on host. The chunked kernel uses backend_tensor_set internally so this
                // is a no-op safety belt for any other consumer that may read these.
                InvalidateTensorDeviceCache(qBuf);
                InvalidateTensorDeviceCache(kBuf);
                InvalidateTensorDeviceCache(vBuf);
                InvalidateTensorDeviceCache(zBuf);
                InvalidateTensorDeviceCache(alphaBuf);
                InvalidateTensorDeviceCache(betaBuf);

                Tensor state = _deltaStateTensor[layer];
                InvalidateTensorDeviceCache(state);

                // The GGML kernel writes [T, H, D] - same memory as gated[T, H*D].
                Tensor gated3D = gated.View(seqLen, H, Dv);
                try
                {
                    GgmlBasicOps.GatedDeltaNetChunked(
                        qBuf, kBuf, vBuf, zBuf,
                        alphaBuf, betaBuf,
                        state, gated3D,
                        new IntPtr(GetFloatPtr(_ssmDtBiasW[layer])),
                        new IntPtr(GetFloatPtr(_ssmAW[layer])),
                        new IntPtr(GetFloatPtr(_ssmNormW[layer])),
                        chunkSize: GdnChunkSize, eps: Config.Eps);
                }
                finally
                {
                    gated3D.Dispose();
                }

                // The kernel downloaded fresh state and gated bytes back to the host
                // buffer; downstream GGML kernels need to re-upload those bytes.
                InvalidateTensorDeviceCache(state);
                InvalidateTensorDeviceCache(gated);
            }
            finally
            {
                // Sub-views increment the underlying storage refcount; dispose them so
                // we don't leak. The persistent backing tensors (_gdnChunked*Buf) are
                // released in Dispose().
                qBuf.Dispose();
                kBuf.Dispose();
                vBuf.Dispose();
                zBuf.Dispose();
                alphaBuf.Dispose();
                betaBuf.Dispose();
            }
        }

        /// <summary>
        /// Lazily (re)allocate the chunked-prefill staging tensors so they cover at least
        /// <paramref name="seqLen"/> rows. The buffers are sized to the largest seqLen we
        /// have seen and reused for every layer in the same forward pass and every
        /// subsequent forward pass with seqLen <= capacity.
        /// </summary>
        private void EnsureChunkedStagingBuffers(int seqLen, int H, int Dk, int Dv)
        {
            if (_gdnChunkedQBuf != null && _gdnChunkedBufCapacity >= seqLen)
                return;

            // Free old buffers (if any) before resizing upward.
            _gdnChunkedQBuf?.Dispose();
            _gdnChunkedKBuf?.Dispose();
            _gdnChunkedVBuf?.Dispose();
            _gdnChunkedZBuf?.Dispose();
            _gdnChunkedAlphaBuf?.Dispose();
            _gdnChunkedBetaBuf?.Dispose();

            _gdnChunkedQBuf     = new Tensor(_allocator, DType.Float32, seqLen, H, Dk);
            _gdnChunkedKBuf     = new Tensor(_allocator, DType.Float32, seqLen, H, Dk);
            _gdnChunkedVBuf     = new Tensor(_allocator, DType.Float32, seqLen, H, Dv);
            _gdnChunkedZBuf     = new Tensor(_allocator, DType.Float32, seqLen, H, Dv);
            _gdnChunkedAlphaBuf = new Tensor(_allocator, DType.Float32, seqLen, H);
            _gdnChunkedBetaBuf  = new Tensor(_allocator, DType.Float32, seqLen, H);
            _gdnChunkedBufCapacity = seqLen;
        }

        /// <summary>
        /// Single-token GatedDeltaNet step.
        ///
        /// Optimizations vs the original implementation:
        /// 1. Convolution uses a circular buffer keyed by `_convStateWriteIdx[layer]`. This
        ///    eliminates the O(convDim * qkvDim) `Array.Copy` shift that ran every token; the
        ///    new step writes the latest input row into one ring slot and reads previous
        ///    slots in modular order. For convKernel=4 and qkvDim ~ 16k this saves ~50k float
        ///    copies per token per recurrent layer.
        /// 2. The conv1d weight is pre-transposed to `[kernelSize, qkvDim]` so the inner
        ///    SIMD loop accumulates `state_tap * weight_tap` over a whole channel block at
        ///    once via System.Numerics vectors instead of scalar accumulation per channel.
        /// 3. SiLU on the conv output uses the SIMD vector helper from VecApplySiLU.
        /// 4. Per-head state updates can run on a thread pool when `_numVHeads >= 16`, which
        ///    matches MoE/qwen3-next configurations that use 32 V-heads.
        /// </summary>
        private unsafe void GatedDeltaNetStep(float* qkvPtr, float* zPtr, float* betaPtr, float* alphaPtr,
            int layer, int qkvDim, int qkDim, int vDim,
            float[] convWT, float* dtBiasPtr, float* aPtr, float* ssmNormPtr,
            float* gatedOutPtr)
        {
            int convDim = _convKernel - 1;
            float[] convState = _convState[layer];
            int writeIdx = _convStateWriteIdx[layer];

            fixed (float* convOutPtr = _gdnConvOutBuf)
            fixed (float* qBase = _gdnQ, kBase = _gdnK, vBase = _gdnV)
            {
                ComputeConv1DStep(qkvPtr, qkvDim, convDim, writeIdx, convState, convWT, convOutPtr);

                if (convDim > 0)
                {
                    fixed (float* statePtr = convState)
                    {
                        Buffer.MemoryCopy(qkvPtr, statePtr + writeIdx * qkvDim,
                            qkvDim * sizeof(float), qkvDim * sizeof(float));
                    }
                    _convStateWriteIdx[layer] = (writeIdx + 1) % convDim;
                }

                Buffer.MemoryCopy(convOutPtr, qBase, qkDim * sizeof(float), qkDim * sizeof(float));
                Buffer.MemoryCopy(convOutPtr + qkDim, kBase, qkDim * sizeof(float), qkDim * sizeof(float));
                Buffer.MemoryCopy(convOutPtr + 2 * qkDim, vBase, vDim * sizeof(float), vDim * sizeof(float));
            }

            float[] qActive = _gdnQ;
            float[] kActive = _gdnK;
            if (_numKHeads != _numVHeads)
            {
                qActive = _gdnQExp;
                kActive = _gdnKExp;
                for (int h = 0; h < _numVHeads; h++)
                {
                    int srcHead = h % _numKHeads;
                    Array.Copy(_gdnQ, srcHead * _headKDim, qActive, h * _headKDim, _headKDim);
                    Array.Copy(_gdnK, srcHead * _headKDim, kActive, h * _headKDim, _headKDim);
                }
            }

            L2NormalizePerHead(qActive, _numVHeads, _headKDim);
            L2NormalizePerHead(kActive, _numVHeads, _headKDim);

            float qScale = 1.0f / MathF.Sqrt(_headVDim);
            int totalQK = _numVHeads * _headKDim;
            fixed (float* qPtr = qActive)
                VecScale(qPtr, qScale, totalQK);

            // Capture pointers/values for parallel head update
            Tensor state = _deltaStateTensor[layer];
            float* statePtrBase = GetFloatPtr(state);
            int statePerHead = _headVDim * _headKDim;
            int headKDim = _headKDim;
            int headVDim = _headVDim;
            float eps = Config.Eps;

            fixed (float* qPin = qActive, kPin = kActive, vPin = _gdnV,
                          deltaPin = _gdnDelta, corePin = _gdnCore)
            {
                float* qPtr = qPin;
                float* kPtr = kPin;
                float* vPtr = vPin;
                float* deltaPtr = deltaPin;
                float* corePtr = corePin;

                if (_gdnParallelHeads)
                {
                    // Local copies because pointers cannot be captured by reference in lambdas.
                    float* qPtrLocal = qPtr;
                    float* kPtrLocal = kPtr;
                    float* vPtrLocal = vPtr;
                    float* deltaPtrLocal = deltaPtr;
                    float* corePtrLocal = corePtr;
                    float* statePtrLocal = statePtrBase;
                    float* zPtrLocal = zPtr;
                    float* dtBiasPtrLocal = dtBiasPtr;
                    float* aPtrLocal = aPtr;
                    float* alphaPtrLocal = alphaPtr;
                    float* betaPtrLocal = betaPtr;
                    float* ssmNormPtrLocal = ssmNormPtr;
                    float* gatedOutPtrLocal = gatedOutPtr;

                    Parallel.For(0, _numVHeads, h =>
                    {
                        ProcessHead(h, qPtrLocal, kPtrLocal, vPtrLocal,
                            deltaPtrLocal, corePtrLocal, statePtrLocal,
                            zPtrLocal, dtBiasPtrLocal, aPtrLocal,
                            alphaPtrLocal, betaPtrLocal, ssmNormPtrLocal,
                            gatedOutPtrLocal,
                            headKDim, headVDim, statePerHead, eps);
                    });
                }
                else
                {
                    for (int h = 0; h < _numVHeads; h++)
                    {
                        ProcessHead(h, qPtr, kPtr, vPtr, deltaPtr, corePtr, statePtrBase,
                            zPtr, dtBiasPtr, aPtr, alphaPtr, betaPtr, ssmNormPtr, gatedOutPtr,
                            headKDim, headVDim, statePerHead, eps);
                    }
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ProcessHead(int h,
            float* qPtr, float* kPtr, float* vPtr,
            float* deltaPtr, float* corePtr, float* statePtrBase,
            float* zPtr, float* dtBiasPtr, float* aPtr,
            float* alphaPtr, float* betaPtr, float* ssmNormPtr,
            float* gatedOutPtr,
            int headKDim, int headVDim, int statePerHead, float eps)
        {
            float* stateHead = statePtrBase + h * statePerHead;
            float* qHead = qPtr + h * headKDim;
            float* kHead = kPtr + h * headKDim;
            float* vHead = vPtr + h * headVDim;
            float* deltaHead = deltaPtr + h * headVDim;
            float* coreHead = corePtr + h * headVDim;
            float* zHead = zPtr + h * headVDim;
            float* gatedHead = gatedOutPtr + h * headVDim;

            float alphaBiased = alphaPtr[h] + dtBiasPtr[h];
            float gateH = SoftplusScalar(alphaBiased) * aPtr[h];
            VecScale(stateHead, MathF.Exp(gateH), statePerHead);

            float betaH = SigmoidScalar(betaPtr[h]);

            // delta_row = (v_row - dot(state_row, k)) * beta
            for (int row = 0; row < headVDim; row++)
            {
                float kvMem = VecDot(stateHead + row * headKDim, kHead, headKDim);
                deltaHead[row] = (vHead[row] - kvMem) * betaH;
            }

            // state_row += k * delta_row;  core_row = dot(state_row, q)
            for (int row = 0; row < headVDim; row++)
            {
                float* stateRow = stateHead + row * headKDim;
                VecScaleAdd(stateRow, kHead, deltaHead[row], headKDim);
                coreHead[row] = VecDot(stateRow, qHead, headKDim);
            }

            float rmsInv = 1.0f / MathF.Sqrt((VecSumSq(coreHead, headVDim) / headVDim) + eps);
            for (int i = 0; i < headVDim; i++)
                gatedHead[i] = coreHead[i] * rmsInv * ssmNormPtr[i] * SiLUScalar(zHead[i]);
        }

        /// <summary>
        /// Vectorized 1D convolution step using a circular state buffer and a transposed
        /// weight layout. For each channel ch and kernel tap ki, we read the state slot
        /// from a logical index that wraps around the ring, and multiply by the contiguous
        /// weight tap row. SIMD vectorization runs along the channel dimension. After the
        /// reduction, SiLU is applied in-place over the channel vector.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ComputeConv1DStep(float* qkvPtr, int qkvDim, int convDim,
            int writeIdx, float[] convState, float[] convWT, float* convOutPtr)
        {
            int vLen = Vector<float>.Count;

            VecZero(convOutPtr, qkvDim);

            fixed (float* statePtr = convState, wtPtr = convWT)
            {
                // Kernel taps 0..convDim-1 sample from the previous (ring buffered) state.
                // The element written most recently is at slot ((writeIdx + convDim - 1) % convDim)
                // and corresponds to the newest historical input (kernel position convDim-1).
                for (int ki = 0; ki < convDim; ki++)
                {
                    int slot = (writeIdx + ki) % convDim;
                    float* statePos = statePtr + slot * qkvDim;
                    float* wtPos = wtPtr + ki * qkvDim;

                    int ch = 0;
                    for (; ch <= qkvDim - vLen; ch += vLen)
                    {
                        var acc = LdVecLocal(convOutPtr + ch);
                        var sv = LdVecLocal(statePos + ch);
                        var wv = LdVecLocal(wtPos + ch);
                        StVecLocal(convOutPtr + ch, acc + sv * wv);
                    }
                    for (; ch < qkvDim; ch++)
                        convOutPtr[ch] += statePos[ch] * wtPos[ch];
                }

                // Final tap reads the new input qkvPtr.
                {
                    float* wtPos = wtPtr + convDim * qkvDim;
                    int ch = 0;
                    for (; ch <= qkvDim - vLen; ch += vLen)
                    {
                        var acc = LdVecLocal(convOutPtr + ch);
                        var iv = LdVecLocal(qkvPtr + ch);
                        var wv = LdVecLocal(wtPos + ch);
                        StVecLocal(convOutPtr + ch, acc + iv * wv);
                    }
                    for (; ch < qkvDim; ch++)
                        convOutPtr[ch] += qkvPtr[ch] * wtPos[ch];
                }
            }

            VecApplySiLU(convOutPtr, qkvDim);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector<float> LdVecLocal(float* p) =>
            Unsafe.ReadUnaligned<Vector<float>>(ref *(byte*)p);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void StVecLocal(float* p, Vector<float> v) =>
            Unsafe.WriteUnaligned(ref *(byte*)p, v);

        /// <summary>
        /// SIMD-vectorized SiLU(x) = x / (1 + exp(-x)).
        /// Falls back to scalar when the |x| range trips the saturation guards used by
        /// SiLUScalar; vector path uses MathF.Exp for the sigmoid since System.Numerics
        /// does not expose Vector exp.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void VecApplySiLU(float* data, int n)
        {
            for (int i = 0; i < n; i++)
                data[i] = SiLUScalar(data[i]);
        }

        private unsafe void L2NormalizePerHead(float[] data, int numHeads, int headDim)
        {
            fixed (float* ptr = data)
            {
                for (int h = 0; h < numHeads; h++)
                {
                    float* head = ptr + h * headDim;
                    float inv = 1.0f / MathF.Sqrt(VecSumSq(head, headDim) + Config.Eps);
                    VecScale(head, inv, headDim);
                }
            }
        }

        private static float SigmoidScalar(float x)
        {
            if (x >= 0)
            {
                float e = MathF.Exp(-x);
                return 1.0f / (1.0f + e);
            }

            float en = MathF.Exp(x);
            return en / (1.0f + en);
        }

        private static float SiLUScalar(float x) => x * SigmoidScalar(x);

        private static float SoftplusScalar(float x)
        {
            if (x > 20f)
                return x;
            if (x < -20f)
                return MathF.Exp(x);
            return MathF.Log(1.0f + MathF.Exp(x));
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

        private unsafe Tensor MoEForward(Tensor input, int layer, int seqLen)
        {
            int hiddenSize = Config.HiddenSize;

            Tensor routerLogits = LinearForwardCached(input, _ffnGateInpQW[layer], _ffnGateInpF32[layer]);
            if (routerLogits == null)
                throw new InvalidOperationException($"Missing MoE router weight for layer {layer}: {_ffnGateInpKey[layer]}");

            Tensor routerProbs = Ops.Softmax(null, routerLogits);
            routerLogits.Dispose();

            float* probsPtr = GetFloatPtr(routerProbs);

            var output = new Tensor(_allocator, DType.Float32, seqLen, hiddenSize);
            float* outputPtr = GetFloatPtr(output);
            VecZero(outputPtr, seqLen * hiddenSize);

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

            float* inputPtr = GetFloatPtr(input);
            float[] routeW = _moeRouteW;
            int[] topExperts = _moeTopExperts;

            // Hot path: for the common decode case (seqLen == 1) we reuse a single set of
            // scratch tensors across all selected experts and skip the per-expert tensor
            // allocations. For prefill (seqLen > 1) we keep the per-token scratch path which
            // matches the original allocation discipline; this keeps the code paths simple
            // while removing the dominant per-token cost in decode.
            bool useReusedBuffers = seqLen == 1 && _moeGateBuf != null;

            for (int s = 0; s < seqLen; s++)
            {
                float* probsRow = probsPtr + (long)s * _numExperts;
                SelectTopKInPlace(probsRow, _numExperts, _numExpertsUsed, topExperts);

                float wSum = 0f;
                for (int k = 0; k < _numExpertsUsed; k++)
                {
                    routeW[k] = probsRow[topExperts[k]];
                    wSum += routeW[k];
                }
                if (_normTopKProb && wSum > 0f)
                {
                    float inv = 1.0f / wSum;
                    for (int k = 0; k < _numExpertsUsed; k++)
                        routeW[k] *= inv;
                }

                Tensor tokenInput;
                bool disposeTokenInput;
                if (seqLen > 1)
                {
                    using var rowView = input.Narrow(0, s, 1);
                    tokenInput = Ops.NewContiguous(rowView);
                    disposeTokenInput = true;
                }
                else
                {
                    tokenInput = input;
                    disposeTokenInput = false;
                }

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

                if (disposeTokenInput)
                    tokenInput.Dispose();
            }

            sharedDownAll?.Dispose();
            routerProbs.Dispose();

            InvalidateTensorDeviceCache(output);
            return output;
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
                    // bind them as identically-shaped tensors. The Qwen3 GGUF format guarantees
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

        private static unsafe void SelectTopKInPlace(float* values, int n, int k, int[] indices)
        {
            Span<float> topVals = stackalloc float[k];
            for (int i = 0; i < k; i++)
            {
                topVals[i] = float.NegativeInfinity;
                indices[i] = -1;
            }

            for (int i = 0; i < n; i++)
            {
                int minIdx = 0;
                for (int j = 1; j < k; j++)
                {
                    if (topVals[j] < topVals[minIdx])
                        minIdx = j;
                }
                if (values[i] > topVals[minIdx])
                {
                    topVals[minIdx] = values[i];
                    indices[minIdx] = i;
                }
            }
        }

        #endregion

        #region Vision Support

        public void LoadVisionEncoder(string mmProjPath)
        {
            VisionEncoder = new Qwen35VisionEncoder(mmProjPath, _allocator);
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
            base.PrintTimingStats();

            double msPerTick = 1000.0 / Stopwatch.Frequency;
            double chunkedMs = _gdnChunkedTicks * msPerTick;
            double perTokenMs = _gdnPerTokenTicks * msPerTick;

            if (_gdnChunkedCalls == 0 && _gdnPerTokenCalls == 0)
                return;

            Console.WriteLine($"  GatedDeltaNet:");
            Console.WriteLine($"    chunked path:   {_gdnChunkedCalls} calls, {chunkedMs:F0} ms total" +
                (_gdnChunkedCalls > 0 ? $", {chunkedMs / _gdnChunkedCalls:F2} ms/call" : ""));
            Console.WriteLine($"    per-token path: {_gdnPerTokenCalls} prefill calls, {perTokenMs:F0} ms total" +
                (_gdnPerTokenCalls > 0 ? $", {perTokenMs / _gdnPerTokenCalls:F2} ms/call" : ""));
            if (_gdnDisableChunkedPrefill)
                Console.WriteLine($"    (chunked path disabled at runtime)");
            else
                Console.WriteLine($"    (chunked threshold: seqLen >= {_gdnChunkPrefillThreshold}, chunkSize {GdnChunkSize})");
        }

        public override void Dispose()
        {
            VisionEncoder?.Dispose();
            foreach (var (visionEmbeddings, _) in _visionEmbeddingsList)
                visionEmbeddings?.Dispose();
            _visionEmbeddingsList.Clear();

            if (_kvCacheK != null)
                foreach (var t in _kvCacheK) t?.Dispose();
            if (_kvCacheV != null)
                foreach (var t in _kvCacheV) t?.Dispose();
            if (_deltaStateTensor != null)
                foreach (var t in _deltaStateTensor) t?.Dispose();

            _gdnGatedOutT?.Dispose();
            _gdnChunkedQBuf?.Dispose();
            _gdnChunkedKBuf?.Dispose();
            _gdnChunkedVBuf?.Dispose();
            _gdnChunkedZBuf?.Dispose();
            _gdnChunkedAlphaBuf?.Dispose();
            _gdnChunkedBetaBuf?.Dispose();

            _moeTokenInput?.Dispose();
            _moeGateBuf?.Dispose();
            _moeUpBuf?.Dispose();
            _moeDownBuf?.Dispose();
            _moeBatchedResult?.Dispose();

            _attnDecodeQBuf?.Dispose();
            _attnDecodeGBuf?.Dispose();
            _attnDecodeOutBuf?.Dispose();
            _attnDecodeQkvBuf?.Dispose();
            _ffnDecodeGateUpBuf?.Dispose();
            _gdnDecodePackedBuf?.Dispose();

            base.Dispose();
        }
    }
}
