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
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.GGML;
using TensorSharp.MLX;

namespace TensorSharp.Models
{
    /// <summary>
    /// Gemma 4 model architecture.
    /// Matches Ollama's gemma4 implementation for identical inference results.
    /// Key features:
    /// - Sliding window pattern (SWA local vs full-attention global layers)
    /// - Different head dims for local (SWA) vs global layers
    /// - KV sharing: last N layers reuse KV cache from earlier donor layers
    /// - Per-Layer Embedding (PLE)
    /// - Proportional RoPE with freq_factors on global layers
    /// - Partial rotary dimensions for global layers
    /// - V norm: unweighted RMSNorm on V projections
    /// - Per-layer output scaling
    /// - Optional MoE (Mixture of Experts) layers
    /// </summary>
    public partial class Gemma4Model : ModelBase
    {
        private bool[] _slidingWindowPattern;
        private int _slidingWindow;
        private float _finalLogitSoftcap;

        private int _localHeadDim;
        private int _globalHeadDim;
        private int _numGlobalKVHeads;
        private int _partialRotaryDims;

        private float _ropeLocalBase, _ropeGlobalBase;
        private float[] _ropeFreqsLocal;
        private float[] _ropeFreqsGlobal;

        private int _sharedKVLayers;
        private Dictionary<int, int> _kvDonorMap;
        private HashSet<int> _swaKVDonorLayers;
        private Dictionary<int, (Tensor k, Tensor v)> _prefillSWAKV;

        // Per-chunk cache of the "previous window" K/V for SWA layers.
        // Holds positions [startPos - prevWindowLen, startPos - 1] in chronological
        // order, gathered from the rolling SWA cache *before* the current chunk
        // overwrites it. This is the critical missing piece that makes chunked
        // prefill correct for sliding-window attention layers: without it, queries
        // near the start of a non-first chunk would see only their own chunk and
        // miss the (W-1) preceding tokens that fall inside their window. Keyed by
        // the KV-cache-owning layer index (donor for KV-shared layers) so all
        // sharing layers reuse the same gather.
        private Dictionary<int, (Tensor k, Tensor v)> _swaPrevWindow;
        private int _swaPrevWindowStartPos = -1;
        private int _swaPrevWindowLen;

        private int _pleDim;

        private Tensor[] _kvCacheK;
        private Tensor[] _kvCacheV;
        private int[] _kvCacheSize; // per-layer cache capacity (slidingWindow for SWA, maxSeqLen for global)

        private float[] _layerScalars;
        private bool _hasTiedOutput;

        private Tensor _onesForVNorm;

        // Per-forward-pass SWA mask cache: all SWA layers share the same
        // mask parameters (queryLen, startPos, windowSize), so we precompute
        // per-row fill widths once and reuse across layers. This eliminates
        // redundant mask rebuilds that were a ~28% regression in ollama.
        private int[] _cachedSWAMaskWidths;
        private int _cachedSWAMaskQueryLen;
        private int _cachedSWAMaskStartPos = -1;

        // NeoX RoPE cos/sin lookup table cached across global layers.
        // The table depends only on (seqLen, startPos, freqs) which are
        // identical for all global layers in a forward pass, eliminating
        // ~35M MathF.Cos/Sin calls per chunk (~700ms → ~5ms).
        private float[] _neoXRopeCos, _neoXRopeSin;
        private int _neoXRopeCacheSeqLen, _neoXRopeCacheStartPos = -1;
        private float[] _neoXRopeCacheFreqs;
        private Tensor _neoXRopeCosTensor, _neoXRopeSinTensor;
        private int _neoXRopeDeviceCacheSeqLen, _neoXRopeDeviceCacheStartPos = -1;
        private float[] _neoXRopeDeviceCacheFreqs;

        // Cached RoPE position tensors for local layers.
        // All local layers in a forward pass share the same (seqLen, startPos),
        // so we precompute once and reuse.
        private Tensor _cachedRoPEPosQ, _cachedRoPEPosK;
        private int _cachedRoPEPosSeqLen, _cachedRoPEPosStartPos = -1;

        private int _numExperts;
        private int _numExpertsUsed;

        // Pooled per-call scratch for MoE routing. Reallocated lazily when the
        // current request needs a larger seqLen (prefill). Decode reuses the
        // same buffers across all layers/steps, eliminating the per-call
        // float[]/int[] allocation in MoERoute().
        private float[] _moeRoutingWeightsScratch;
        private int[] _moeSelectedExpertsScratch;
        private int[] _moeTopKScratch;

        // Per-layer caches for the fused expert-batched MoE FFN kernel
        // (GgmlBasicOps.MoEFFNPrefill with GEGLU activation). These are
        // zero-cost views into the original 3D <c>ffn_{gate,up,down}_exps.weight</c>
        // blocks populated by <see cref="ModelBase.LoadWeights"/>. When a layer
        // has them, MoEForward collapses O(num_active_experts * seqLen)
        // per-expert / per-token matmuls into one ggml_mul_mat_id dispatch
        // per projection, mirroring llama.cpp's build_moe_ffn. _layerPerExpertScale
        // folds the per-expert post-down-projection scale (Gemma 4's
        // ffn_down_exps.scale / ffn_gate_inp.per_expert_scale) into the
        // routing weights so the native kernel stays activation-agnostic.
        // Null entries mean the layer is either non-MoE or its expert weights
        // aren't in a stacked layout (e.g. F32 fallback) and the legacy
        // batched-by-expert C# path is used instead.
        private StackedExpertWeights[] _layerStackedGate;
        private StackedExpertWeights[] _layerStackedUp;
        private StackedExpertWeights[] _layerStackedDown;
        private float[][] _layerPerExpertScale;

        private bool _canUseFusedDecode;
        // Gates the model-wide single-graph decode kernel (NativeGemma4ModelDecode).
        // When any layer is MoE this stays false because the model-wide kernel has
        // no MoE branch — but per-layer fused prefill/decode (gated by
        // _canUseFusedDecode + HasMoE(l) at the call site) is still allowed for
        // the dense layers, recovering the speedup on the dense majority of an
        // MoE Gemma 4 model. Kept as a separate flag so a future MoE-aware
        // model-wide kernel can flip this on independently.
        private bool _canUseFusedFullModelDecode;
        // Gates the fused single-graph MoE-layer decode kernel
        // (TSGgml_Gemma4MoELayerDecode). Disable via TS_GGML_MOE_FUSED_DECODE=0
        // for A/B comparison against the per-op TransformerBlock path. Flipped
        // off at runtime if the kernel ever throws (graceful degradation).
        private bool _moeFusedDecodeEnabled =
            Environment.GetEnvironmentVariable("TS_GGML_MOE_FUSED_DECODE") != "0";
        private bool _kvCacheHostDirty;
        private Gemma4DecodeArrays _decodeArrays;

        private Gemma4VisionEncoder _visionEncoder;
        private Gemma4AudioEncoder _audioEncoder;
        private List<(Tensor embeddings, int position)> _pendingVisionEmbeddingsList = new();
        private List<(Tensor embeddings, int position)> _pendingAudioEmbeddingsList = new();

        // Pipelined greedy decode state. _pipelineNextInputHidden holds the
        // scaled input embedding for the NEXT decode step, computed on-device
        // from the previous step's argmax. _pipelineNextPLE holds the
        // [1, _pleDim * numLayers] per-layer-input tensor for the same step
        // (computed via the same device get_rows on per_layer_token_embd).
        // When set, SubmitGreedyDecodeStep uses them instead of looking up
        // host ints — so the inference loop can queue step N+1's forward
        // BEFORE syncing step N's predicted token to host. See
        // SubmitGreedyDecodeStep below for details.
        private Tensor _pipelineNextInputHidden;
        private Tensor _pipelineNextPLE;

        // Set by Attention() when the fused MLX QKV preprocess kernel ran
        // (Q/K/V split + RMSNorm + Q/K NeoX RoPE in one Metal dispatch). The
        // RoPE block further down then skips its own Q/K RoPE since the
        // kernel already applied it. Reset at the start of every Attention
        // call so it doesn't leak across layers.
        private bool _attnFusedDecodePreprocessApplied;
        private static readonly int MlxEvalEveryNLayers = ResolveMlxEvalEveryNLayers();
        private static readonly int MlxLocalKvMaterializeInterval = ResolveMlxLocalKvMaterializeInterval();
        private static readonly bool MlxEvalDecodeLayerBoundaries =
            !string.Equals(Environment.GetEnvironmentVariable("TS_MLX_GEMMA4_EVAL_DECODE_LAYER_BOUNDARIES"), "0", StringComparison.Ordinal);
        private static readonly bool MlxBaselineSyncLayerEval =
            string.Equals(Environment.GetEnvironmentVariable("TS_MLX_BASELINE_ASYNC_LAYER"), "1", StringComparison.Ordinal);

        public Gemma4VisionEncoder VisionEncoder => _visionEncoder;
        public Gemma4AudioEncoder AudioEncoder => _audioEncoder;

        private static int ResolveMlxEvalEveryNLayers()
        {
            string env = Environment.GetEnvironmentVariable("TS_MLX_GEMMA4_EVAL_EVERY_N_LAYERS")
                         ?? Environment.GetEnvironmentVariable("TS_MLX_EVAL_EVERY_N_LAYERS");
            if (!string.IsNullOrWhiteSpace(env) && int.TryParse(env, out int v) && v >= 0)
                return v;
            // Phase 7: every 4 layers, kick MLX's accumulated graph via
            // mlx_async_eval. Without this MLX builds the whole 42-layer
            // dependency chain in lazy form and the GPU sits idle until the
            // host-side host sync at LM-head triggers evaluation — i.e.
            // GPU/CPU work is serialized rather than overlapped. Periodic
            // async_eval lets the GPU start the front half of the layer
            // stack while the host is still queueing the back half.
            //
            // Sweep on gemma-4-E4B Q8_0 (42 layers, 256 prefill / 128 decode):
            //   N=0  (disabled)  → 41.87 ms/tok
            //   N=3              → 40.03 ms/tok
            //   N=4              → 39.19 ms/tok  ← best
            //   N=5              → 41.21 ms/tok
            //   N=7              → 40.10 ms/tok
            //   N=14             → 40.52 ms/tok
            //   N=42 (once/tok)  → 42.82 ms/tok
            //
            // Best win comes from N≈4 (~11 evals per token). Smaller N
            // wastes dispatches; larger N leaves more of the stack
            // un-pipelined. The pre-Phase-6 default was 8, then 0 because
            // the kernels' tree reductions hid the pipelining gain; now
            // with simdgroup-fast kernels the bottleneck shifts to CPU/GPU
            // overlap and async_eval becomes a real win.
            return 4;
        }

        private static int ResolveMlxLocalKvMaterializeInterval()
        {
            string env = Environment.GetEnvironmentVariable("TS_MLX_GEMMA4_LOCAL_KV_MATERIALIZE_INTERVAL")
                         ?? Environment.GetEnvironmentVariable("TS_MLX_LOCAL_KV_MATERIALIZE_INTERVAL");
            if (!string.IsNullOrWhiteSpace(env) && int.TryParse(env, out int v) && v >= 0)
                return v;
            // The materialize check fires whenever (startPos + 1 + layer) is a
            // multiple of this interval. Disabled by default for the standard
            // <c>Forward</c> path used by the server engine — the per-token
            // LM-head host sync (<c>GetFloatPtr</c> on the logits tensor)
            // already materializes the K/V slice_update chain through the
            // attention dependency, so an additional periodic materialize
            // is pure overhead. A sweep at interval ∈ {0, 8, 32, 64, 128}
            // showed interval=8 (previous default) costing ~1.5 ms/token
            // vs. interval=0 with no observable benefit even at long
            // decodes (256–512 tokens). Re-enable via the env var when
            // running the pipelined greedy decode path (<c>SubmitGreedyDecodeStep</c>),
            // which only host-syncs a 1-int argmax per step and therefore
            // does need a periodic graph drain to keep chain depth bounded
            // — a value of 128 is a reasonable starting point there.
            return 0;
        }

        public void LoadVisionEncoder(string mmProjPath)
        {
            // The direct CUDA backend currently diverges numerically in the Gemma4
            // vision stack; keep projector embeddings on the stable CPU path and
            // copy the final embeddings into the CUDA language model.
            IAllocator visionAllocator = _backend == BackendType.Cuda
                ? new CpuAllocator(BlasEnum.DotNet)
                : _allocator;
            _visionEncoder = new Gemma4VisionEncoder(mmProjPath, visionAllocator);
            // Give the encoder a reference back to us so its per-block loop
            // can yield ModelBase.GpuComputeLock between blocks (keeps the
            // engine worker responsive during long image encodes).
            _visionEncoder.SetHostModel(this);
        }

        public void LoadAudioEncoder(string mmProjPath)
        {
            _audioEncoder = new Gemma4AudioEncoder(mmProjPath, _allocator);
            _audioEncoder.SetHostModel(this);
        }

        public void SetAudioEmbeddings(Tensor embeddings, int insertPosition)
        {
            _pendingAudioEmbeddingsList.Add((embeddings, insertPosition));
        }

        public Gemma4Model(string ggufPath, BackendType backend) : base(ggufPath, backend)
        {
            Config = new ModelConfig { Architecture = _gguf.GetString("general.architecture") };
            ParseBaseConfig();

            string arch = Config.Architecture;

            _slidingWindowPattern = _gguf.GetBoolArray($"{arch}.attention.sliding_window_pattern");
            _slidingWindow = (int)_gguf.GetUint32($"{arch}.attention.sliding_window", 512);
            Config.SlidingWindow = _slidingWindow;
            Config.UsesCircularKvCache = _slidingWindowPattern != null && Array.Exists(_slidingWindowPattern, isLocal => isLocal);

            // Head dimensions: key_length is global head dim, key_length_swa is local head dim
            // Ollama uses a single headDim for Q/K/V per layer type
            _globalHeadDim = (int)_gguf.GetUint32($"{arch}.attention.key_length", 512);
            _localHeadDim = (int)_gguf.GetUint32($"{arch}.attention.key_length_swa", 256);

            // RoPE dimensions for global layers with proportional RoPE
            _partialRotaryDims = (int)_gguf.GetUint32($"{arch}.rope.dimension_count", 0);
            if (_partialRotaryDims == 0)
            {
                float partialFactor = _gguf.GetFloat32($"{arch}.rope.partial_rotary_factor", 1.0f);
                _partialRotaryDims = (int)(_globalHeadDim * partialFactor);
            }

            // KV heads: try per-layer array first, then fall back to scalar
            _numGlobalKVHeads = (int)_gguf.GetUint32($"{arch}.attention.global_head_count_kv", 0);
            var kvHeadsArray = _gguf.GetInt32Array($"{arch}.attention.head_count_kv");
            if (kvHeadsArray != null && kvHeadsArray.Length > 0)
            {
                Config.NumKVHeads = kvHeadsArray[0];
                if (_numGlobalKVHeads == 0 && _slidingWindowPattern != null)
                {
                    for (int i = 0; i < _slidingWindowPattern.Length && i < kvHeadsArray.Length; i++)
                    {
                        if (!_slidingWindowPattern[i])
                        {
                            _numGlobalKVHeads = kvHeadsArray[i];
                            break;
                        }
                    }
                }
            }
            if (_numGlobalKVHeads == 0) _numGlobalKVHeads = Config.NumKVHeads;

            _ropeLocalBase = _gguf.GetFloat32($"{arch}.rope.freq_base_swa", 0);
            if (_ropeLocalBase == 0) _ropeLocalBase = _gguf.GetFloat32($"{arch}.rope.local.freq_base", 10000f);
            _ropeGlobalBase = Config.RopeBase;

            _finalLogitSoftcap = _gguf.GetFloat32($"{arch}.final_logit_softcapping", 0f);
            _pleDim = (int)_gguf.GetUint32($"{arch}.embedding_length_per_layer_input", 0);

            _sharedKVLayers = (int)_gguf.GetUint32($"{arch}.attention.shared_kv_layers", 0);
            BuildKVDonorMap();

            _numExperts = (int)_gguf.GetUint32($"{arch}.expert_count", 0);
            _numExpertsUsed = (int)_gguf.GetUint32($"{arch}.expert_used_count", 0);
            if (_numExpertsUsed > 0)
                _moeTopKScratch = new int[_numExpertsUsed];

            Console.WriteLine($"Model: {arch}, Layers={Config.NumLayers}, " +
                $"Hidden={Config.HiddenSize}, Heads={Config.NumHeads}, KVHeads={Config.NumKVHeads}, " +
                $"GlobalKVHeads={_numGlobalKVHeads}, Vocab={Config.VocabSize}");
            Console.WriteLine($"Head dims: local={_localHeadDim}, global={_globalHeadDim}");
            Console.WriteLine($"RoPE global={_ropeGlobalBase} local={_ropeLocalBase}");
            Console.WriteLine($"Partial rotary dims={_partialRotaryDims}");
            Console.WriteLine($"Sliding window={_slidingWindow}, Softcap={_finalLogitSoftcap}");
            Console.WriteLine($"PLE dim={_pleDim}, SharedKVLayers={_sharedKVLayers}");
            if (_numExperts > 0)
                Console.WriteLine($"MoE: {_numExperts} experts, {_numExpertsUsed} used per token");

            int localCount = 0, globalCount = 0;
            for (int i = 0; i < Config.NumLayers; i++)
            {
                if (IsLocalLayer(i)) localCount++;
                else globalCount++;
            }
            Console.WriteLine($"Layer types: {globalCount} global (causal), {localCount} local (SWA)");
            if (_kvDonorMap.Count > 0)
            {
                int firstShared = Config.NumLayers - _sharedKVLayers;
                Console.WriteLine($"KV sharing: layers {firstShared}-{Config.NumLayers - 1} share with donors");
            }

            ParseTokenizer();
            LoadWeights();

            _hasTiedOutput = !_weights.ContainsKey("output.weight") && !_quantWeights.ContainsKey("output.weight");
            if (_hasTiedOutput)
                Console.WriteLine("  Output tied to token_embd.weight");

            DetectHeadDimsFromWeights();
            LoadLayerScalars();
            FuseQKVWeights();
            FuseGateUpWeights();
            FuseExpertGateUpWeights();
            CacheMoEStackedWeights();
            PrepareCudaQuantizedWeightsForInference();
            PrecomputeRoPE();
            int maxContextLength = ResolveConfiguredContextLength();
            int initialCacheLength = ResolveInitialCacheAllocationLength(maxContextLength);
            if (initialCacheLength < maxContextLength)
                Console.WriteLine($"Initial {_backend} KV cache allocation: {initialCacheLength} tokens for global layers (grows on demand up to {maxContextLength}).");
            InitKVCache(initialCacheLength, maxContextLength);
            BuildGemma4DecodeArrays();
        }

        private bool IsLocalLayer(int layer) =>
            _slidingWindowPattern != null && layer < _slidingWindowPattern.Length && _slidingWindowPattern[layer];

        private int HeadDimForLayer(int layer) => IsLocalLayer(layer) ? _localHeadDim : _globalHeadDim;
        private int KVHeadsForLayer(int layer) => IsLocalLayer(layer) ? Config.NumKVHeads : _numGlobalKVHeads;

        private (float ropeBase, int ropeDims) RopeForLayer(int layer)
        {
            if (IsLocalLayer(layer))
                return (_ropeLocalBase, _localHeadDim);
            return (_ropeGlobalBase, _partialRotaryDims);
        }

        private void BuildKVDonorMap()
        {
            _kvDonorMap = new Dictionary<int, int>();
            _swaKVDonorLayers = new HashSet<int>();
            if (_sharedKVLayers <= 0 || _slidingWindowPattern == null) return;

            int firstShared = Config.NumLayers - _sharedKVLayers;
            for (int i = firstShared; i < Config.NumLayers; i++)
            {
                bool isLocal = IsLocalLayer(i);
                for (int j = firstShared - 1; j >= 0; j--)
                {
                    if (IsLocalLayer(j) == isLocal)
                    {
                        _kvDonorMap[i] = j;
                        // Track every donor (both SWA and global) so the
                        // fused-prefill kernel can publish their post-norm /
                        // post-RoPE K/V to downstream shared layers via
                        // _prefillSWAKV. Without this, global shared layers
                        // bail out to the C# managed path - which would crash
                        // when the cache is block-quantized (Q8_0).
                        _swaKVDonorLayers.Add(j);
                        break;
                    }
                }
            }
        }

        private void LoadLayerScalars()
        {
            _layerScalars = new float[Config.NumLayers];
            for (int l = 0; l < Config.NumLayers; l++)
            {
                string key = $"blk.{l}.layer_output_scale.weight";
                if (_weights.TryGetValue(key, out var t))
                {
                    _layerScalars[l] = t.GetElementAsFloat(0);
                }
                else
                {
                    _layerScalars[l] = 1f;
                }
            }
        }

        private int GetWeightOutputDim(string weightName)
        {
            if (_quantWeights.TryGetValue(weightName, out var qw))
                return (int)qw.Ne1;
            if (_weights.TryGetValue(weightName, out var w))
                return (int)w.Sizes[0];
            return -1;
        }

        private void DetectHeadDimsFromWeights()
        {
            bool localDone = false, globalDone = false;
            for (int l = 0; l < Config.NumLayers && (!localDone || !globalDone); l++)
            {
                bool isLocal = IsLocalLayer(l);
                if ((isLocal && localDone) || (!isLocal && globalDone)) continue;

                int kvHeads = KVHeadsForLayer(l);
                if (kvHeads <= 0) continue;

                int kOutDim = GetWeightOutputDim($"blk.{l}.attn_k.weight");
                if (kOutDim <= 0) continue;

                int actualHeadDim = kOutDim / kvHeads;
                if (isLocal)
                {
                    if (actualHeadDim != _localHeadDim)
                    {
                        Console.WriteLine($"  Adjusted local head dim: {_localHeadDim} -> {actualHeadDim}");
                        _localHeadDim = actualHeadDim;
                    }
                    localDone = true;
                }
                else
                {
                    if (actualHeadDim != _globalHeadDim)
                    {
                        Console.WriteLine($"  Adjusted global head dim: {_globalHeadDim} -> {actualHeadDim}");
                        _globalHeadDim = actualHeadDim;
                        if (_partialRotaryDims > _globalHeadDim)
                            _partialRotaryDims = _globalHeadDim;
                    }
                    globalDone = true;
                }
            }
        }

        private void PrecomputeRoPE()
        {
            int localHalfDim = _localHeadDim / 2;
            _ropeFreqsLocal = new float[localHalfDim];
            for (int i = 0; i < localHalfDim; i++)
                _ropeFreqsLocal[i] = (float)(1.0 / Math.Pow(_ropeLocalBase, 2.0 * i / _localHeadDim));

            // Global RoPE uses partialRotaryDims for the freq computation
            int globalHalfDim = _partialRotaryDims / 2;
            _ropeFreqsGlobal = new float[globalHalfDim];

            float[] freqFactors = null;
            if (_weights.TryGetValue("rope_freqs.weight", out var freqTensor))
            {
                freqFactors = TensorToFloatArray(freqTensor);
            }

            for (int i = 0; i < globalHalfDim; i++)
            {
                double freq = 1.0 / Math.Pow(_ropeGlobalBase, 2.0 * i / _partialRotaryDims);
                if (freqFactors != null && i < freqFactors.Length)
                    freq /= freqFactors[i];
                _ropeFreqsGlobal[i] = (float)freq;
            }
        }

        // Current capacity of global-attention layers (SWA layers stay at
        // _slidingWindow and never grow).
        private int _kvCacheGlobalCapacity;

        private void InitKVCache(int initialGlobalSeqLen, int maxSeqLen)
        {
            _maxContextLength = maxSeqLen;
            _kvCacheGlobalCapacity = initialGlobalSeqLen;
            _kvCacheK = new Tensor[Config.NumLayers];
            _kvCacheV = new Tensor[Config.NumLayers];
            _kvCacheSize = new int[Config.NumLayers];

            // Pick a model-aligned default cache dtype (F16 for any non-F32
            // weights) when the user hasn't explicitly chosen one. This gives
            // quantized models the bandwidth/memory benefits of F16 cache by
            // default while staying byte-identical to F32 outputs.
            ApplyModelAlignedKvCacheDefault(_quantWeights);
            DType kvDtype = _kvCacheDtype.ToDType();

            long totalCacheBytes = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_kvDonorMap.ContainsKey(l)) continue;

                int kvHeads = KVHeadsForLayer(l);
                int hd = HeadDimForLayer(l);
                int cacheLen = IsLocalLayer(l) ? _slidingWindow : initialGlobalSeqLen;
                _kvCacheSize[l] = cacheLen;
                _kvCacheK[l] = new Tensor(_allocator, kvDtype, kvHeads, cacheLen, hd);
                _kvCacheV[l] = new Tensor(_allocator, kvDtype, kvHeads, cacheLen, hd);
                InitializeCacheTensor(_kvCacheK[l]);
                InitializeCacheTensor(_kvCacheV[l]);
                // Q8_0 has fractional bytes/elem (1.0625) - go through ByteLengthFor
                // so block-quantized layouts are accounted for correctly.
                long perLayerElems = (long)kvHeads * cacheLen * hd;
                totalCacheBytes += _kvCacheDtype.ByteLengthFor(perLayerElems) * 2;
            }

            foreach (var kv in _kvDonorMap)
            {
                _kvCacheK[kv.Key] = _kvCacheK[kv.Value];
                _kvCacheV[kv.Key] = _kvCacheV[kv.Value];
                _kvCacheSize[kv.Key] = _kvCacheSize[kv.Value];
            }

            Console.WriteLine($"  KV cache: {totalCacheBytes / 1024 / 1024} MB " +
                $"(dtype: {_kvCacheDtype.ToShortString()}, global layers: {initialGlobalSeqLen} seq, SWA layers: {_slidingWindow} seq)");
        }

        // Grow the global-attention layers' KV cache to fit requiredSeqLen
        // (doubling, capped at _maxContextLength). SWA layers are left alone
        // because they wrap circularly within _slidingWindow and never need
        // more storage. Donor-shared layers track their donor — we only
        // resize each underlying cache once and update the alias entries.
        private void EnsureCacheCapacity(int requiredSeqLen)
        {
            if (requiredSeqLen <= _kvCacheGlobalCapacity)
                return;
            if (requiredSeqLen > _maxContextLength)
                throw new InvalidOperationException($"Requested sequence length {requiredSeqLen} exceeds configured max context {_maxContextLength}.");

            // The resize loop below uses Ops.Copy to move existing K/V into the
            // larger tensors, and Ops.Copy is a CPU memcpy. Under device-copy
            // mode on the Metal backend the freshest cache writes may still
            // live in the Metal buffer; sync them back to the host buffer
            // first so the memcpy reads the right bytes.
            EnsureKvCacheHostSynchronized();

            int newCapacity = Math.Max(_kvCacheGlobalCapacity, 1);
            while (newCapacity < requiredSeqLen)
                newCapacity = Math.Min(_maxContextLength, newCapacity * 2);

            DType kvDtype = _kvCacheDtype.ToDType();
            var resized = new HashSet<int>();
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_kvDonorMap.ContainsKey(l)) continue;
                if (IsLocalLayer(l)) continue;          // SWA — never grows
                if (!resized.Add(l)) continue;

                int kvHeads = KVHeadsForLayer(l);
                int hd = HeadDimForLayer(l);
                var newK = new Tensor(_allocator, kvDtype, kvHeads, newCapacity, hd);
                var newV = new Tensor(_allocator, kvDtype, kvHeads, newCapacity, hd);
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
                _kvCacheSize[l] = newCapacity;
            }

            // Re-point donor aliases to the resized underlying caches.
            foreach (var kv in _kvDonorMap)
            {
                if (IsLocalLayer(kv.Value)) continue;
                _kvCacheK[kv.Key] = _kvCacheK[kv.Value];
                _kvCacheV[kv.Key] = _kvCacheV[kv.Value];
                _kvCacheSize[kv.Key] = _kvCacheSize[kv.Value];
            }

            // BuildGemma4DecodeArrays cached the K/V host pointers and capacity
            // for the fused per-layer / full-model decode kernels at model load
            // time. Those pointers refer to the storage we just disposed, so
            // refresh them to the new tensors — otherwise the next layer kernel
            // hands GGML a freed pointer and Metal reports "buffer is nil" for
            // every K/V-derived intermediate.
            RefreshDecodeArraysKvCache();

            _kvCacheGlobalCapacity = newCapacity;
            Console.WriteLine($"Expanded Gemma4 global attention cache to {newCapacity} tokens.");
        }

        private void RefreshDecodeArraysKvCache()
        {
            if (_decodeArrays == null) return;
            var a = _decodeArrays;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                int kvSource = _kvDonorMap.TryGetValue(l, out int donor) ? donor : l;
                if (_kvCacheK[kvSource] != null)
                    a.KCache[l] = TensorComputePrimitives.GetStoragePointer(_kvCacheK[kvSource]);
                if (_kvCacheV[kvSource] != null)
                    a.VCache[l] = TensorComputePrimitives.GetStoragePointer(_kvCacheV[kvSource]);
                a.CacheSize[l] = _kvCacheSize[kvSource];
            }
        }

        public override void ResetKVCache()
        {
            _cacheSeqLen = 0;
            _kvCacheHostDirty = false;
            DisposeSwaPrevWindows();
            if (_kvCacheK == null) return;
            var cleared = new HashSet<int>();
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (cleared.Contains(l)) continue;
                if (_kvDonorMap.ContainsKey(l)) continue;
                ResetCacheTensor(_kvCacheK[l]);
                ResetCacheTensor(_kvCacheV[l]);
                cleared.Add(l);
            }
        }

        public override void TruncateKVCache(int tokenCount)
        {
            DisposeSwaPrevWindows();
            EnsureKvCacheHostSynchronized();
            base.TruncateKVCache(tokenCount);
            _kvCacheHostDirty = false;
            if (_kvCacheK == null) return;
            var invalidated = new HashSet<int>();
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (invalidated.Contains(l)) continue;
                if (_kvDonorMap.ContainsKey(l)) continue;
                InvalidateTensorDeviceCache(_kvCacheK[l]);
                InvalidateTensorDeviceCache(_kvCacheV[l]);
                invalidated.Add(l);
            }
        }

        // Gemma 4 hosts its local-attention layers in a circular slidingWindow-sized
        // cache (see CopyToCacheDecode line ~2972: cachePos = startPos % cacheSize).
        // The byte-level snapshot path below applies per-layer modular addressing so
        // a position p is read/written at SWA slot (p % _slidingWindow) for local
        // layers while global layers use linear addressing. This keeps the snapshot
        // well-defined even after the SWA cache has wrapped, which is what lets
        // multi-turn conversations achieve full prefix-cache reuse instead of
        // capping at one sliding window.
        public override bool SupportsKVStateSnapshot
            => _kvCacheK != null && _kvCacheV != null;

        // Gemma 4 CAN reuse a K/V snapshot across sequences, but the POOLED snapshot
        // path only up to one sliding window. Its local-attention layers (40 of 48)
        // live in a circular cache that physically retains just the last
        // _slidingWindow positions; the global layers keep everything. Re-injecting a
        // pooled snapshot longer than the window forces the local layers to
        // reconstruct wrapped state that no longer exists, which corrupts output
        // (verified: the model loses the conversation and replies as if the context
        // were random text). Within the window the pooled snapshot is correct.
        //
        // MaxReusablePrefixTokens caps the POOLED path at the window. Reuse BEYOND the
        // window is still achieved - correctly - via live-cache continuation
        // (BatchExecutor.ComputeLiveContinuationLcp), which keeps the model's actual
        // live cache between same-session turns instead of reconstructing it.
        public override bool SupportsCrossSequenceKvReuse => true;

        public override int MaxReusablePrefixTokens =>
            Config != null && Config.UsesCircularKvCache && _slidingWindow > 0
                ? _slidingWindow
                : int.MaxValue;

        public override string KVStateFingerprint =>
            $"gemma4|arch={Config.Architecture}|L={Config.NumLayers}|H={Config.NumHeads}|KV={Config.NumKVHeads}|gKV={_numGlobalKVHeads}|gD={_globalHeadDim}|lD={_localHeadDim}|swa={_slidingWindow}|dtype={_kvCacheDtype.ToShortString()}";

        public override long ComputeKVBlockByteSize(int tokenCount)
            => KvBlockTransfer.ComputeBlockByteSize(_kvCacheK, _kvCacheV, tokenCount);

        public override bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        {
            if (_kvCacheK == null || _kvCacheV == null) return false;
            if (startToken < 0 || tokenCount <= 0) return false;
            if (startToken + tokenCount > _cacheSeqLen) return false;
            // For SWA local layers, only positions in the current circular window are
            // recoverable. Once a block's positions have been overwritten by later
            // wrap-around writes we can't reconstruct them. Per-seq CaptureNewlyFullBlocks
            // extracts blocks the moment they become full, which keeps them within
            // [_cacheSeqLen - _slidingWindow, _cacheSeqLen) and avoids that edge case.
            if (_slidingWindow > 0
                && _cacheSeqLen > _slidingWindow
                && startToken < _cacheSeqLen - _slidingWindow)
            {
                return false;
            }
            long expected = ComputeKVBlockByteSize(tokenCount);
            if (destination.Length != expected) return false;
            EnsureKvCacheHostSynchronized();
            return CopyKVBlockPerLayer(startToken, tokenCount, destination, ReadOnlySpan<byte>.Empty, copyOut: true);
        }

        public override bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source)
        {
            if (_kvCacheK == null || _kvCacheV == null) return false;
            if (destToken < 0 || tokenCount <= 0) return false;
            // Inject is called by BatchExecutor.InjectAllBlocks block-by-block in
            // ascending order starting at destToken==0 after a ResetKVCache. The
            // modular SWA writes wrap as the in-order injection progresses; the
            // final cache state after N blocks is byte-identical to the state the
            // original sequence would have left after forwarding tokens [0, N*BS)
            // sequentially, which is what the new sequence's attention expects.
            if (destToken != _cacheSeqLen) return false;
            EnsureCacheCapacity(destToken + tokenCount);
            EnsureKvCacheHostSynchronized();
            long expected = ComputeKVBlockByteSize(tokenCount);
            if (source.Length != expected) return false;
            if (!CopyKVBlockPerLayer(destToken, tokenCount, Span<byte>.Empty, source, copyOut: false))
                return false;
            _cacheSeqLen = destToken + tokenCount;
            // Match TruncateKVCache: invalidate each unique storage exactly once.
            var invalidated = new HashSet<int>();
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (invalidated.Contains(l)) continue;
                if (_kvDonorMap.ContainsKey(l)) continue;
                InvalidateTensorDeviceCache(_kvCacheK[l]);
                InvalidateTensorDeviceCache(_kvCacheV[l]);
                invalidated.Add(l);
            }
            _kvCacheHostDirty = false;
            return true;
        }

        /// <summary>
        /// Walks layer K/V tensors in the same unique-storage order
        /// <see cref="KvBlockTransfer"/> uses (so the destination byte layout matches
        /// what <see cref="ComputeKVBlockByteSize"/> reports), copying each per-layer
        /// payload using a per-layer offset that is modular for SWA local layers and
        /// linear for global layers. Pass <paramref name="copyOut"/>=true to extract
        /// from cache into <paramref name="destination"/>, or false to inject from
        /// <paramref name="source"/> into cache.
        /// </summary>
        private unsafe bool CopyKVBlockPerLayer(
            int positionStart,
            int tokenCount,
            Span<byte> destination,
            ReadOnlySpan<byte> source,
            bool copyOut)
        {
            int totalOffset = 0;
            var seenStorages = new HashSet<Storage>(ReferenceEqualityComparer.Instance);
            for (int l = 0; l < Config.NumLayers; l++)
            {
                bool isLocal = IsLocalLayer(l);
                int layerPos = (isLocal && _slidingWindow > 0)
                    ? (positionStart % _slidingWindow)
                    : positionStart;
                if (layerPos + tokenCount > (isLocal && _slidingWindow > 0 ? _slidingWindow : int.MaxValue))
                    return false; // block straddles a wrap boundary - blockSize must divide slidingWindow

                var k = _kvCacheK[l];
                if (k != null && seenStorages.Add(k.Storage))
                {
                    if (!CopyTensorBlock(k, layerPos, tokenCount, destination, source, totalOffset, copyOut, out int wrote))
                        return false;
                    totalOffset += wrote;
                }
                var v = _kvCacheV[l];
                if (v != null && seenStorages.Add(v.Storage))
                {
                    if (!CopyTensorBlock(v, layerPos, tokenCount, destination, source, totalOffset, copyOut, out int wrote))
                        return false;
                    totalOffset += wrote;
                }
            }
            int expectedLen = copyOut ? destination.Length : source.Length;
            return totalOffset == expectedLen;
        }

        private static unsafe bool CopyTensorBlock(
            Tensor t,
            int startToken,
            int tokenCount,
            Span<byte> destination,
            ReadOnlySpan<byte> source,
            int bufferOffset,
            bool copyOut,
            out int bytesMoved)
        {
            t.Storage.EnsureHostReadable();
            long numHeads = t.Sizes[0];
            long capacity = t.Sizes[1];
            long rowBytes = t.Storage.ByteLength / (numHeads * capacity);
            long blockBytes = numHeads * tokenCount * rowBytes;
            int bufLen = copyOut ? destination.Length : source.Length;
            if (bufferOffset + blockBytes > bufLen) { bytesMoved = 0; return false; }
            IntPtr storageBase = t.Storage.PtrAtElement(0);
            if (storageBase == IntPtr.Zero) { bytesMoved = 0; return false; }
            byte* storagePtr = (byte*)storageBase;
            long perHead = tokenCount * rowBytes;
            if (copyOut)
            {
                fixed (byte* dstBase = destination)
                {
                    for (long h = 0; h < numHeads; h++)
                    {
                        long srcOff = (h * capacity + startToken) * rowBytes;
                        long dstOff = bufferOffset + h * perHead;
                        Buffer.MemoryCopy(storagePtr + srcOff, dstBase + dstOff, bufLen - dstOff, perHead);
                    }
                }
            }
            else
            {
                fixed (byte* srcBase = source)
                {
                    for (long h = 0; h < numHeads; h++)
                    {
                        long dstOff = (h * capacity + startToken) * rowBytes;
                        long srcOff = bufferOffset + h * perHead;
                        Buffer.MemoryCopy(srcBase + srcOff, storagePtr + dstOff, t.Storage.ByteLength - dstOff, perHead);
                    }
                }
            }
            bytesMoved = (int)blockBytes;
            return true;
        }

        public void SetVisionEmbeddings(Tensor embeddings, int insertPosition)
        {
            _pendingVisionEmbeddingsList.Add((embeddings, insertPosition));
        }

        public override float[] Forward(int[] tokens)
        {
            // On MLX, every public op routes through MlxWorker.Shared.Invoke
            // which costs ~25µs per round-trip. Decode dispatches ~600 MLX
            // calls per token across 42 layers, so the per-call hand-off
            // dwarfs actual compute. Wrapping the whole forward in one Invoke
            // moves ALL nested ops onto the worker thread — they detect
            // IsOnWorkerThread and run inline, eliminating ~15 ms / token of
            // queue overhead. Other backends are unaffected.
            if (_backend == BackendType.Mlx && !MlxWorker.Shared.IsOnWorkerThread)
                return MlxWorker.Shared.Invoke(() => ForwardCore(tokens));
            return ForwardCore(tokens);
        }

        // Per-section decode profiler: gated by TS_MLX_DECODE_PROFILE=1.
        // Each section is followed by a forced MLX eval so the wall-clock
        // tick range reflects GPU work, not just dispatch overhead. The
        // forced eval breaks MLX's normal pipelining, so this is a profiling
        // mode only — turn off for steady-state perf measurement.
        private static readonly bool s_DecodeProfileEnabled =
            string.Equals(Environment.GetEnvironmentVariable("TS_MLX_DECODE_PROFILE"), "1", StringComparison.Ordinal);
        private long _profEmbedTicks, _profPleTicks, _profLayerTicks, _profFinalNormTicks, _profLmHeadTicks, _profSoftcapTicks, _profLogitsCopyTicks;
        private long _profAttnSdpaTicks, _profAttnOutTicks, _profFfnTicks, _profPleInjectTicks;
        private int _profDecodeCalls;

        private void ProfMark(ref long counter, long since, Tensor syncTensor)
        {
            if (!s_DecodeProfileEnabled || _backend != BackendType.Mlx) return;
            if (syncTensor != null) MlxFusedOps.TryEvaluate(syncTensor);
            counter += Stopwatch.GetTimestamp() - since;
        }

        public override void PrintTimingStats()
        {
            base.PrintTimingStats();
            DumpDecodeProfile();
        }

        public void DumpDecodeProfile()
        {
            if (!s_DecodeProfileEnabled || _profDecodeCalls == 0) return;
            double inv = 1000.0 / Stopwatch.Frequency / _profDecodeCalls;
            Console.WriteLine($"=== MLX decode profile ({_profDecodeCalls} decode calls) ===");
            Console.WriteLine($"  embed       : {_profEmbedTicks * inv:F3} ms/tok");
            Console.WriteLine($"  PLE compute : {_profPleTicks * inv:F3} ms/tok");
            Console.WriteLine($"  per-layer   : {_profLayerTicks * inv:F3} ms/tok");
            Console.WriteLine($"    attn SDPA : {_profAttnSdpaTicks * inv:F3} ms/tok");
            Console.WriteLine($"    attn out  : {_profAttnOutTicks * inv:F3} ms/tok  (output proj + post-attn-norm+add)");
            Console.WriteLine($"    FFN       : {_profFfnTicks * inv:F3} ms/tok  (gate_up+gelu+down + post-ffn-norm+add)");
            Console.WriteLine($"    PLE inject: {_profPleInjectTicks * inv:F3} ms/tok");
            Console.WriteLine($"  final norm  : {_profFinalNormTicks * inv:F3} ms/tok");
            Console.WriteLine($"  LM head     : {_profLmHeadTicks * inv:F3} ms/tok");
            Console.WriteLine($"  softcap     : {_profSoftcapTicks * inv:F3} ms/tok");
            Console.WriteLine($"  logits copy : {_profLogitsCopyTicks * inv:F3} ms/tok  (forced sync to host)");
        }

        private float[] ForwardCore(int[] tokens)
        {
            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            // The model-wide single-graph decode kernel is only enabled when
            // every layer is dense; an MoE layer in the model forces the
            // per-layer dispatch loop below (which can still hit the fused
            // per-layer kernel for the dense majority).
            bool useFusedDecode = seqLen == 1 && _canUseFusedFullModelDecode;

            EnsureCacheCapacity(startPos + seqLen);

            // Optional tensor-level diag. Forces the non-fused per-op layer
            // path so legacy and batched share the same code shape.
            bool _g4DumpDiag = System.Environment.GetEnvironmentVariable("TS_GEMMA4_DIAG") == "1";
            // The tests' batched-vs-legacy comparison sets TS_GEMMA4_FORCE_UNFUSED=1
            // to make the legacy path fully deterministic (no fused layer prefill,
            // no fused decode) without enabling the verbose per-layer checksum
            // prints from TS_GEMMA4_DIAG.
            bool _g4ForceUnfused = _g4DumpDiag ||
                System.Environment.GetEnvironmentVariable("TS_GEMMA4_FORCE_UNFUSED") == "1";
            if (_g4ForceUnfused)
            {
                System.Environment.SetEnvironmentVariable("TS_FUSED_LAYER_PREFILL", "0");
                useFusedDecode = false;
            }

            long t0 = Stopwatch.GetTimestamp();
            long pStart = s_DecodeProfileEnabled && seqLen == 1 ? Stopwatch.GetTimestamp() : 0;
            Tensor hidden = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t0;

            ScaleEmbedding(hidden);
            if (s_DecodeProfileEnabled && seqLen == 1) ProfMark(ref _profEmbedTicks, pStart, hidden);

            if (_g4DumpDiag) System.Console.WriteLine($"[g4-legacy ] after-embed: {DiagChecksum(hidden, "embed")}");

            HashSet<int> exceptPositions = null;

            if (_pendingVisionEmbeddingsList.Count > 0)
            {
                exceptPositions = new HashSet<int>();
                foreach (var (emb, pos) in _pendingVisionEmbeddingsList)
                {
                    int numTokens = (int)emb.Sizes[0];
                    for (int i = 0; i < numTokens; i++)
                        exceptPositions.Add(startPos + pos + i);
                    InjectVisionEmbeddings(hidden, emb, pos, startPos);
                    emb.Dispose();
                }
                _pendingVisionEmbeddingsList.Clear();
            }

            if (_pendingAudioEmbeddingsList.Count > 0)
            {
                exceptPositions ??= new HashSet<int>();
                foreach (var (emb, pos) in _pendingAudioEmbeddingsList)
                {
                    int numTokens = (int)emb.Sizes[0];
                    for (int i = 0; i < numTokens; i++)
                        exceptPositions.Add(startPos + pos + i);
                    InjectVisionEmbeddings(hidden, emb, pos, startPos);
                    emb.Dispose();
                }
                _pendingAudioEmbeddingsList.Clear();
            }

            Tensor perLayerInputs = null;
            long pPle = s_DecodeProfileEnabled && seqLen == 1 ? Stopwatch.GetTimestamp() : 0;
            if (_pleDim > 0)
                perLayerInputs = ComputePLE(tokens, hidden, seqLen);
            if (s_DecodeProfileEnabled && seqLen == 1) ProfMark(ref _profPleTicks, pPle, perLayerInputs);

            if (!useFusedDecode)
                EnsureKvCacheHostSynchronized();

            long pLayers = s_DecodeProfileEnabled && seqLen == 1 ? Stopwatch.GetTimestamp() : 0;

            if (useFusedDecode)
            {
                long tFused = Stopwatch.GetTimestamp();
                NativeGemma4ModelDecode(hidden, startPos, perLayerInputs);
                _linearTicks += Stopwatch.GetTimestamp() - tFused;
                _kvCacheHostDirty = true;
            }
            else
            {
                // Save donor SWA layer's freshly-computed K/V so KV-shared SWA
                // layers can attend to the *full* chunk's K/V instead of the
                // (incomplete) rolling cache. This matters whenever any chunk
                // has seqLen > slidingWindow because by then the cache no
                // longer holds the early positions of the chunk - a chunk-1
                // shared layer reading the cache would see the chunk's last
                // W positions only, breaking attention for queries near the
                // start of the chunk. Pre-allocating the dict here means the
                // donor's TransformerBlock will populate it on its way out.
                bool useFusedLayerPrefill = Environment.GetEnvironmentVariable("TS_FUSED_LAYER_PREFILL") != "0";
                bool useFusedLayerGraph = CanUseFusedLayerGraph(seqLen, exceptPositions, useFusedLayerPrefill);
                if (_g4DumpDiag || System.Environment.GetEnvironmentVariable("TS_GEMMA4_PATH_TRACE") == "1")
                    System.Console.WriteLine($"[g4-legacy ] path: useFusedLayerPrefill={useFusedLayerPrefill} useFusedLayerGraph={useFusedLayerGraph} TS_FUSED_LAYER_PREFILL={Environment.GetEnvironmentVariable("TS_FUSED_LAYER_PREFILL")}");

                if (_swaKVDonorLayers.Count > 0 && (seqLen > 1 || useFusedLayerGraph))
                    _prefillSWAKV = new Dictionary<int, (Tensor, Tensor)>();

                // Capture the live SWA "previous window" from the rolling cache
                // *before* any layer in this chunk overwrites it. This is what
                // makes chunked SWA prefill produce the same logits as non-chunked.
                if (seqLen > 1 || useFusedLayerGraph)
                    PrepareSwaPrevWindowsForChunk(startPos, seqLen);

                // The fused per-layer prefill kernel runs the entire transformer
                // block (attn + MLP + PLE) as a single GGML graph. It accepts the
                // SWA prev-window for cross-chunk attention and publishes the
                // donor's freshly-computed K/V to a host buffer so KV-shared
                // layers see the full chunk's K/V rather than a partial rolling
                // cache. Real-world prompts get a 45-50% speedup (chunked) over
                // the per-op C# path. Set TS_FUSED_LAYER_PREFILL=0 to disable.
                int _fusedHits = 0, _nonFusedHits = 0;
                for (int l = 0; l < Config.NumLayers; l++)
                {
                    Tensor perLayerInput = null;
                    if (perLayerInputs != null)
                        perLayerInput = ExtractPerLayerSlice(perLayerInputs, l, seqLen);

                    bool isShared = _kvDonorMap.ContainsKey(l);

                    bool canUseLayerGraph = useFusedLayerGraph && !HasMoE(l);
                    // Shared global layers must attend the donor's full prefix.
                    // The fused layer graph currently accepts only donor K/V for
                    // the current token/chunk, so keep that case on the C# path.
                    if (canUseLayerGraph && seqLen == 1 && isShared && !IsLocalLayer(l) && startPos > 0)
                        canUseLayerGraph = false;

                    if (canUseLayerGraph)
                    {
                        if (TryFusedLayerPrefill(hidden, l, seqLen, startPos, perLayerInput))
                        {
                            _fusedHits++;
                            perLayerInput?.Dispose();
                            continue;
                        }
                    }

                    // Fused single-graph MoE layer decode (attention + dense FFN
                    // + in-graph-routed experts) on GGML. Collapses ~18-20 per-op
                    // dispatches per MoE layer (each allocating/freeing a Metal
                    // buffer and synchronising) into one GPU graph, which is the
                    // dominant decode cost for MoE Gemma 4 (e.g. 26B-A4B). Only
                    // for the plain decode shape (no PLE / vision injection).
                    if (seqLen == 1 && HasMoE(l) && perLayerInput == null
                        && exceptPositions == null && _moeFusedDecodeEnabled
                        && TryFusedMoELayerDecode(hidden, l, startPos))
                    {
                        _fusedHits++;
                        if (_g4DumpDiag) System.Console.WriteLine($"[g4-legacy ] after-layer-{l}: {DiagChecksum(hidden, $"L{l}")}");
                        continue;
                    }

                    _nonFusedHits++;
                    hidden = TransformerBlock(hidden, l, seqLen, startPos, isShared, perLayerInput, exceptPositions);
                    TryEvaluateMlxLayerBoundary(hidden, l, seqLen);
                    perLayerInput?.Dispose();
                    if (_g4DumpDiag) System.Console.WriteLine($"[g4-legacy ] after-layer-{l}: {DiagChecksum(hidden, $"L{l}")}");
                    else if (_g4ForceUnfused)
                    {
                        // Force a CPU read between layers when the test-only
                        // FORCE_UNFUSED mode is on. Metal queues ops async by
                        // default; without flushing, parallel reductions in
                        // RMSNorm/softmax can give bit-different results
                        // between runs and the test loses comparability with
                        // the (eagerly-synced) batched path. The pinged byte
                        // is discarded; only the implicit download-barrier
                        // matters.
                        _ = hidden.GetElementsAsFloat(1);
                    }
                }
                if (_g4DumpDiag || System.Environment.GetEnvironmentVariable("TS_GEMMA4_PATH_TRACE") == "1")
                    System.Console.WriteLine($"[g4-legacy ] _fusedHits={_fusedHits} _nonFusedHits={_nonFusedHits}");

                if (_prefillSWAKV != null)
                {
                    foreach (var kv in _prefillSWAKV.Values)
                    {
                        kv.k.Dispose();
                        kv.v.Dispose();
                    }
                    _prefillSWAKV = null;
                }
                DisposeSwaPrevWindows();
            }

            if (s_DecodeProfileEnabled && seqLen == 1) ProfMark(ref _profLayerTicks, pLayers, hidden);

            perLayerInputs?.Dispose();

            long pFinalNorm = s_DecodeProfileEnabled && seqLen == 1 ? Stopwatch.GetTimestamp() : 0;

            Tensor lastHidden;
            if (seqLen > 1)
            {
                using var lastRow = hidden.Narrow(0, seqLen - 1, 1);
                lastHidden = Ops.NewContiguous(lastRow);
                hidden.Dispose();
                Ops.RMSNorm(lastHidden, lastHidden, _weights["output_norm.weight"], null, Config.Eps);
            }
            else
            {
                Tensor normed = RMSNormOp(hidden, "output_norm.weight");
                hidden.Dispose();
                lastHidden = normed;
            }
            if (s_DecodeProfileEnabled && seqLen == 1) ProfMark(ref _profFinalNormTicks, pFinalNorm, lastHidden);

            t0 = Stopwatch.GetTimestamp();
            long pLmHead = s_DecodeProfileEnabled && seqLen == 1 ? Stopwatch.GetTimestamp() : 0;
            string outputWeight = _hasTiedOutput ? "token_embd.weight" : "output.weight";
            Tensor logitsTensor = LinearForward(lastHidden, outputWeight);
            _lmHeadTicks += Stopwatch.GetTimestamp() - t0;
            lastHidden.Dispose();
            if (s_DecodeProfileEnabled && seqLen == 1) ProfMark(ref _profLmHeadTicks, pLmHead, logitsTensor);

            long pSoftcap = s_DecodeProfileEnabled && seqLen == 1 ? Stopwatch.GetTimestamp() : 0;
            if (_finalLogitSoftcap > 0f)
                ApplyLogitSoftcap(logitsTensor);
            if (s_DecodeProfileEnabled && seqLen == 1) ProfMark(ref _profSoftcapTicks, pSoftcap, logitsTensor);

            t0 = Stopwatch.GetTimestamp();
            long pCopy = s_DecodeProfileEnabled && seqLen == 1 ? Stopwatch.GetTimestamp() : 0;
            if (_logitsBuffer == null || _logitsBuffer.Length != Config.VocabSize)
                _logitsBuffer = new float[Config.VocabSize];

            unsafe
            {
                float* ptr = GetFloatPtr(logitsTensor);
                fixed (float* dst = _logitsBuffer)
                    Buffer.MemoryCopy(ptr, dst, Config.VocabSize * 4, Config.VocabSize * 4);
            }
            logitsTensor.Dispose();
            _logitsCopyTicks += Stopwatch.GetTimestamp() - t0;
            if (s_DecodeProfileEnabled && seqLen == 1)
            {
                _profLogitsCopyTicks += Stopwatch.GetTimestamp() - pCopy;
                _profDecodeCalls++;
            }

            _cacheSeqLen += seqLen;
            _forwardCount++;
            _forwardSw.Stop();
            return _logitsBuffer;
        }

        // ===== Pipelined greedy decode =====
        //
        // Standard Forward(int[]) builds the layer graph, then host-syncs the
        // [1, vocab] logits tensor (256K floats for gemma-4) at the LM head
        // before returning. The sync drains all queued MLX kernels — pure
        // GPU-idle wait from the host's perspective, ~30+ ms per token on
        // gemma-4-E4B Q8_0.
        //
        // SubmitGreedyDecodeStep replaces that pattern with a deferred
        // device-side token tensor: argmax runs on GPU, the predicted token
        // is a [1] int32 device array, AND the next step's input embedding
        // (+ per-layer-embedding) is pre-computed by a device get_rows on
        // that same int tensor. The caller (CLI inference loop) can submit
        // step N+1 BEFORE host-syncing step N's predicted int, so the LM
        // head + sync wait at the end of step N overlaps with step N+1's
        // first kernels — the textbook pipelined decode pattern that
        // ollama's mlxrunner uses.
        //
        // Greedy only: top-K / temperature sampling still needs the full
        // logits on host. Gated by model.SupportsPipelinedGreedy on the
        // call site (returns true only when MLX backend + tied/quantized
        // LM head + quantized token_embd are all available).
        public override bool SupportsPipelinedGreedy =>
            _backend == BackendType.Mlx
            && _quantWeights.ContainsKey("token_embd.weight");

        public override Tensor SubmitGreedyDecodeStep(int? firstTokenForBegin)
        {
            // Same wrapping rationale as Forward(): collapse all nested MLX
            // worker round-trips into one big inline run on the worker thread.
            if (_backend == BackendType.Mlx && !MlxWorker.Shared.IsOnWorkerThread)
                return MlxWorker.Shared.Invoke(() => SubmitGreedyDecodeStepCore(firstTokenForBegin));
            return SubmitGreedyDecodeStepCore(firstTokenForBegin);
        }

        private Tensor SubmitGreedyDecodeStepCore(int? firstTokenForBegin)
        {
            _forwardSw.Start();
            int seqLen = 1;
            int startPos = _cacheSeqLen;
            EnsureCacheCapacity(startPos + seqLen);

            Tensor hidden;
            Tensor perLayerInputs;

            if (firstTokenForBegin.HasValue)
            {
                // Begin path: starting from a host int (sampled from prefill
                // logits via the CPU sampler). Drop any cached pipeline state
                // — we're re-seeding the chain.
                if (_pipelineNextInputHidden != null) { _pipelineNextInputHidden.Dispose(); _pipelineNextInputHidden = null; }
                if (_pipelineNextPLE != null) { _pipelineNextPLE.Dispose(); _pipelineNextPLE = null; }

                int[] beginTokens = new[] { firstTokenForBegin.Value };
                long embT0 = Stopwatch.GetTimestamp();
                hidden = Embedding(beginTokens);
                ScaleEmbedding(hidden);
                _embTicks += Stopwatch.GetTimestamp() - embT0;

                perLayerInputs = _pleDim > 0 ? ComputePLE(beginTokens, hidden, seqLen) : null;
            }
            else if (_pipelineNextInputHidden != null)
            {
                // Steady state: the previous step pre-computed both the input
                // embedding AND (if PLE applies) the per-layer-embedding via
                // device get_rows on the previous argmax. Use them as-is.
                hidden = _pipelineNextInputHidden;
                _pipelineNextInputHidden = null;
                perLayerInputs = _pipelineNextPLE;
                _pipelineNextPLE = null;
            }
            else
            {
                throw new InvalidOperationException(
                    "SubmitGreedyDecodeStep: no cached input embedding and no firstTokenForBegin provided. " +
                    "Call with firstTokenForBegin set after prefill before any null-token continuations.");
            }

            // Run the layer stack. Same shape as the decode path in Forward —
            // no chunked prefill, no fused-layer-prefill (which only fires for
            // GGML anyway), no SWA prev-window gather (decode reads the rolling
            // cache directly).
            EnsureKvCacheHostSynchronized();
            for (int l = 0; l < Config.NumLayers; l++)
            {
                Tensor perLayerInput = null;
                if (perLayerInputs != null)
                    perLayerInput = ExtractPerLayerSlice(perLayerInputs, l, seqLen);

                bool isShared = _kvDonorMap.ContainsKey(l);
                hidden = TransformerBlock(hidden, l, seqLen, startPos, isShared, perLayerInput, null);
                TryEvaluateMlxLayerBoundary(hidden, l, seqLen);
                perLayerInput?.Dispose();
            }
            perLayerInputs?.Dispose();

            // Final norm + LM head + softcap, same as Forward().
            long lmT0 = Stopwatch.GetTimestamp();
            Tensor lastNormed = RMSNormOp(hidden, "output_norm.weight");
            hidden.Dispose();
            string outputWeight = _hasTiedOutput ? "token_embd.weight" : "output.weight";
            Tensor logitsTensor = LinearForward(lastNormed, outputWeight);
            lastNormed.Dispose();
            _lmHeadTicks += Stopwatch.GetTimestamp() - lmT0;

            if (_finalLogitSoftcap > 0f)
                ApplyLogitSoftcap(logitsTensor);

            // Device argmax → [1] int32. Stays on device — this is the handle
            // we return; the caller defers .GetElementsAsInt() until AFTER
            // submitting the next step.
            var deviceToken = new Tensor(_allocator, DType.Int32, 1);
            if (!MlxFusedOps.TryArgMaxLastAxis(deviceToken, logitsTensor))
            {
                // Host fallback. Forces a sync (defeating the pipeline) but
                // keeps the output correct on backends or shapes the MLX
                // argmax kernel doesn't cover.
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

            // Pre-compute the NEXT step's input embedding (+ PLE if applicable)
            // via device-side get_rows on `deviceToken`. After this returns the
            // next forward can run without any host int round-trip.
            var nextHidden = new Tensor(_allocator, DType.Float32, 1, Config.HiddenSize);
            bool deviceEmbOk = TryComputeNextInputEmbeddingDevice(nextHidden, deviceToken);
            if (!deviceEmbOk)
            {
                // Device path unsupported (no MLX, or non-quantized embedding
                // table). Sync the token to host and rebuild via the standard
                // path. This eats the pipeline win for THIS step but keeps
                // correctness — and is rare in practice for our supported
                // gemma-4 GGUFs (token_embd is always quantized).
                int hostTok = deviceToken.GetElementsAsInt(1)[0];
                nextHidden.Dispose();
                nextHidden = Embedding(new[] { hostTok });
                ScaleEmbedding(nextHidden);
                _pipelineNextInputHidden = nextHidden;
                _pipelineNextPLE = _pleDim > 0 ? ComputePLE(new[] { hostTok }, nextHidden, seqLen) : null;
            }
            else
            {
                // Apply the post-lookup scale on device.
                ScaleEmbedding(nextHidden);
                _pipelineNextInputHidden = nextHidden;
                _pipelineNextPLE = _pleDim > 0
                    ? ComputeNextPLEDevice(deviceToken, nextHidden)
                    : null;

                // Kick the queued kernels (argmax + embedding + PLE lookups)
                // so they start executing on Metal NOW. The host sync of
                // deviceToken in the caller's loop would drain this implicitly,
                // but an explicit async-eval gives MLX a chance to schedule
                // and start the work before the host issues the next
                // SubmitGreedyDecodeStep — that's the pipeline overlap.
                MlxFusedOps.TryAsyncEvaluate(_pipelineNextInputHidden);
                if (_pipelineNextPLE != null)
                    MlxFusedOps.TryAsyncEvaluate(_pipelineNextPLE);
            }

            _cacheSeqLen += seqLen;
            _forwardCount++;
            _forwardSw.Stop();
            return deviceToken;
        }

        // Device-side equivalent of Embedding(int[]) for a single [1] int32
        // device tensor. Reuses MlxQuantizedOps.TryGetRowsQuantizedToFloat32
        // — same path Embedding(int[]) takes for MLX-quantized token_embd.
        // Returns false (and leaves outHidden zero-initialized) if the
        // backend or weight type doesn't support a device get_rows.
        private bool TryComputeNextInputEmbeddingDevice(Tensor outHidden, Tensor deviceTokenInt)
        {
            if (_backend != BackendType.Mlx)
                return false;

            if (_quantWeights.TryGetValue("token_embd.weight", out var qw))
            {
                return MlxQuantizedOps.TryGetRowsQuantizedToFloat32(
                    outHidden,
                    qw.EnsureDeviceCacheKey(),
                    qw.Data,
                    qw.GgmlType,
                    qw.Ne0,
                    qw.Ne1,
                    qw.RawBytes,
                    deviceTokenInt);
            }

            // F32 token_embd via Ops.IndexSelect would also work but isn't
            // exercised by our supported gemma-4 GGUFs (Q8_0/Q4_K_M/etc all
            // quantize token_embd). Leave as a future TODO if needed.
            return false;
        }

        // Device-side PLE computation for the next decode step. Mirrors
        // ComputePLE(int[], hidden, seqLen) but takes a [1] int32 device
        // token tensor for the per_layer_token_embd lookup and uses the
        // already-scaled hidden tensor for the projection. Returns the
        // combined [1, _pleDim * numLayers] tensor or null if PLE is
        // disabled or all paths fall back to host. Note: any failure
        // here is fine — the caller will fall back to host PLE on the
        // next call.
        private Tensor ComputeNextPLEDevice(Tensor deviceTokenInt, Tensor hidden)
        {
            int totalPleDim = _pleDim * Config.NumLayers;
            int seqLen = 1;

            Tensor pleTokenEmb = null;
            if (_quantWeights.TryGetValue("per_layer_token_embd.weight", out var pleQw))
            {
                pleTokenEmb = new Tensor(_allocator, DType.Float32, seqLen, totalPleDim);
                if (!MlxQuantizedOps.TryGetRowsQuantizedToFloat32(
                        pleTokenEmb,
                        pleQw.EnsureDeviceCacheKey(),
                        pleQw.Data,
                        pleQw.GgmlType,
                        pleQw.Ne0,
                        pleQw.Ne1,
                        pleQw.RawBytes,
                        deviceTokenInt))
                {
                    pleTokenEmb.Dispose();
                    pleTokenEmb = null;
                }
                else
                {
                    float pleScale = MathF.Sqrt(_pleDim);
                    Ops.Mul(pleTokenEmb, pleTokenEmb, pleScale);
                }
            }

            // Projection branch — same shape as ComputePLE.
            Tensor pleProj = LinearForward(hidden, "per_layer_model_proj.weight");
            if (pleProj != null)
            {
                float projScale = 1f / MathF.Sqrt(Config.HiddenSize);
                Ops.Mul(pleProj, pleProj, projScale);

                int totalRows = seqLen * Config.NumLayers;
                using var reshaped = pleProj.View(totalRows, _pleDim);
                var normWeight = _weights["per_layer_proj_norm.weight"];
                Ops.RMSNorm(reshaped, reshaped, normWeight, null, Config.Eps);
            }

            if (pleTokenEmb != null && pleProj != null)
            {
                Ops.Add(pleProj, pleProj, pleTokenEmb);
                float combineScale = 1f / MathF.Sqrt(2f);
                Ops.Mul(pleProj, pleProj, combineScale);
                pleTokenEmb.Dispose();
                return pleProj;
            }
            if (pleProj != null) return pleProj;
            if (pleTokenEmb != null) return pleTokenEmb;
            return null;
        }

        public override void ResetPipelinedGreedyState()
        {
            _pipelineNextInputHidden?.Dispose();
            _pipelineNextInputHidden = null;
            _pipelineNextPLE?.Dispose();
            _pipelineNextPLE = null;
        }

        public override float[] ForwardRefill(int[] tokens)
        {
            if (tokens == null || tokens.Length <= 1 || !_canUseFusedDecode)
                return Forward(tokens);

            int lastIdx = tokens.Length - 1;

            // Chunked prefill: process the prefix in bounded chunks to keep
            // attention score tensors for full-attention layers small on long
            // prompts. Skip chunking when multimodal embeddings are queued
            // because their insertion positions are relative to the full
            // sequence.
            //
            // Chunk size choice: bigger chunks amortize per-chunk overhead
            // (RoPE table rebuild, mask construction, tensor allocations,
            // KV-cache prev-window gather) but raise peak memory for the
            // full-attention layers (their score tensor is
            // ~numHeads × chunkLen × totalKvLen × 4B). Past 2048 the score
            // tensor for the longest-context decode reaches hundreds of MB
            // and starts thrashing the memory pool. Override via
            // TS_PREFILL_CHUNK env var when tuning for unusual contexts.
            int prefillChunkSize = Math.Min(2048, Math.Max(_slidingWindow * 2, 1024));
            string chunkOverride = Environment.GetEnvironmentVariable("TS_PREFILL_CHUNK");
            if (!string.IsNullOrEmpty(chunkOverride) && int.TryParse(chunkOverride, out int chunkOv) && chunkOv > 0)
                prefillChunkSize = chunkOv;
            bool hasMultimodal = _pendingVisionEmbeddingsList.Count > 0
                              || _pendingAudioEmbeddingsList.Count > 0;

            if (!hasMultimodal && lastIdx > prefillChunkSize)
            {
                for (int pos = 0; pos < lastIdx; pos += prefillChunkSize)
                {
                    int chunkLen = Math.Min(prefillChunkSize, lastIdx - pos);
                    var chunk = new int[chunkLen];
                    Array.Copy(tokens, pos, chunk, 0, chunkLen);
                    PrefillWithoutLogits(chunk);
                }
                return Forward(new[] { tokens[lastIdx] });
            }

            var prefixTokens = new int[lastIdx];
            Array.Copy(tokens, prefixTokens, lastIdx);
            PrefillWithoutLogits(prefixTokens);
            return Forward(new[] { tokens[lastIdx] });
        }

        private void PrefillWithoutLogits(int[] tokens)
        {
            if (tokens == null || tokens.Length == 0)
                return;

            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            bool useFusedDecode = seqLen == 1 && _canUseFusedFullModelDecode;
            bool _g4DumpDiag = false; // PrefillWithoutLogits doesn't dump diag

            EnsureCacheCapacity(startPos + seqLen);

            long t0 = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t0;

            ScaleEmbedding(hidden);

            HashSet<int> exceptPositions = null;

            if (_pendingVisionEmbeddingsList.Count > 0)
            {
                exceptPositions = new HashSet<int>();
                foreach (var (emb, pos) in _pendingVisionEmbeddingsList)
                {
                    int numTokens = (int)emb.Sizes[0];
                    for (int i = 0; i < numTokens; i++)
                        exceptPositions.Add(startPos + pos + i);
                    InjectVisionEmbeddings(hidden, emb, pos, startPos);
                    emb.Dispose();
                }
                _pendingVisionEmbeddingsList.Clear();
            }

            if (_pendingAudioEmbeddingsList.Count > 0)
            {
                exceptPositions ??= new HashSet<int>();
                foreach (var (emb, pos) in _pendingAudioEmbeddingsList)
                {
                    int numTokens = (int)emb.Sizes[0];
                    for (int i = 0; i < numTokens; i++)
                        exceptPositions.Add(startPos + pos + i);
                    InjectVisionEmbeddings(hidden, emb, pos, startPos);
                    emb.Dispose();
                }
                _pendingAudioEmbeddingsList.Clear();
            }

            Tensor perLayerInputs = null;
            if (_pleDim > 0)
                perLayerInputs = ComputePLE(tokens, hidden, seqLen);

            if (!useFusedDecode)
                EnsureKvCacheHostSynchronized();

            if (useFusedDecode)
            {
                long tFused = Stopwatch.GetTimestamp();
                NativeGemma4ModelDecode(hidden, startPos, perLayerInputs);
                _linearTicks += Stopwatch.GetTimestamp() - tFused;
                _kvCacheHostDirty = true;
            }
            else
            {
                bool useFusedLayerPrefill = Environment.GetEnvironmentVariable("TS_FUSED_LAYER_PREFILL") != "0";
                bool useFusedLayerGraph = CanUseFusedLayerGraph(seqLen, exceptPositions, useFusedLayerPrefill);

                if (_swaKVDonorLayers.Count > 0 && (seqLen > 1 || useFusedLayerGraph))
                    _prefillSWAKV = new Dictionary<int, (Tensor, Tensor)>();

                if (seqLen > 1 || useFusedLayerGraph)
                    PrepareSwaPrevWindowsForChunk(startPos, seqLen);

                for (int l = 0; l < Config.NumLayers; l++)
                {
                    Tensor perLayerInput = null;
                    if (perLayerInputs != null)
                        perLayerInput = ExtractPerLayerSlice(perLayerInputs, l, seqLen);

                    bool isShared = _kvDonorMap.ContainsKey(l);

                    bool canUseLayerGraph = useFusedLayerGraph && !HasMoE(l);
                    // Shared global layers must attend the donor's full prefix.
                    // The fused layer graph currently accepts only donor K/V for
                    // the current token/chunk, so keep that case on the C# path.
                    if (canUseLayerGraph && seqLen == 1 && isShared && !IsLocalLayer(l) && startPos > 0)
                        canUseLayerGraph = false;

                    if (canUseLayerGraph)
                    {
                        if (TryFusedLayerPrefill(hidden, l, seqLen, startPos, perLayerInput))
                        {
                            perLayerInput?.Dispose();
                            continue;
                        }
                    }

                    hidden = TransformerBlock(hidden, l, seqLen, startPos, isShared, perLayerInput, exceptPositions);
                    TryEvaluateMlxLayerBoundary(hidden, l, seqLen);
                    perLayerInput?.Dispose();
                    if (_g4DumpDiag) System.Console.WriteLine($"[g4-legacy ] after-layer-{l}: {DiagChecksum(hidden, $"L{l}")}");
                }

                if (_prefillSWAKV != null)
                {
                    foreach (var kv in _prefillSWAKV.Values)
                    {
                        kv.k.Dispose();
                        kv.v.Dispose();
                    }
                    _prefillSWAKV = null;
                }
                DisposeSwaPrevWindows();
            }

            perLayerInputs?.Dispose();
            hidden.Dispose();

            _cacheSeqLen += seqLen;
            _forwardSw.Stop();
        }

        private void ScaleEmbedding(Tensor hidden)
        {
            float scale = MathF.Sqrt(Config.HiddenSize);
            Ops.Mul(hidden, hidden, scale);
        }

        // Diagnostic-only: per-tensor checksum for tensor-level diff between
        // legacy and batched forward paths. Enable via TS_GEMMA4_DIAG=1.
        // Sample the LAST token's state since the LM head reads from there.
        internal static string DiagChecksum(Tensor t, string label)
        {
            int total = (int)t.ElementCount();
            int hidden = t.DimensionCount >= 2 ? (int)t.Sizes[t.DimensionCount - 1] : total;
            int numRows = total / hidden;
            // Sample last row + first row to catch divergence anywhere.
            float[] data = t.GetElementsAsFloat(total);
            double sumFirst = 0, sumLast = 0, absFirst = 0, absLast = 0;
            for (int i = 0; i < hidden; i++) {
                sumFirst += data[i];
                absFirst += Math.Abs(data[i]);
                int lastIdx = (numRows - 1) * hidden + i;
                sumLast += data[lastIdx];
                absLast += Math.Abs(data[lastIdx]);
            }
            int lastTokOff = (numRows - 1) * hidden;
            return $"{label}: rows={numRows} hidden={hidden} | tok0 sum={sumFirst:F3} abs={absFirst:F3} first3=[{data[0]:F3},{data[1]:F3},{data[2]:F3}] | tokN sum={sumLast:F3} abs={absLast:F3} first3=[{data[lastTokOff]:F3},{data[lastTokOff+1]:F3},{data[lastTokOff+2]:F3}]";
        }

        private void ApplyLogitSoftcap(Tensor logits)
        {
            float cap = _finalLogitSoftcap;
            Ops.Mul(logits, logits, 1f / cap);
            Ops.Tanh(logits, logits);
            Ops.Mul(logits, logits, cap);
        }

        private void InjectVisionEmbeddings(Tensor hidden, Tensor visionEmbeddings, int insertPos, int startPos)
        {
            int numVisionTokens = (int)visionEmbeddings.Sizes[0];
            using var target = hidden.Narrow(0, insertPos, numVisionTokens);
            if (ReferenceEquals(target.Allocator, visionEmbeddings.Allocator))
            {
                Ops.Copy(target, visionEmbeddings);
            }
            else
            {
                float[] hostEmbeddings = visionEmbeddings.GetElementsAsFloat((int)visionEmbeddings.ElementCount());
                using var deviceEmbeddings = new Tensor(hidden.Allocator, visionEmbeddings.ElementType, visionEmbeddings.Sizes);
                deviceEmbeddings.SetElementsAsFloat(hostEmbeddings);
                Ops.Copy(target, deviceEmbeddings);
            }
            // insertPos is the offset within the current prefill chunk; startPos
            // (the cached sequence length) makes the absolute position explicit so
            // the log reads monotonically across chunked multimodal prefill.
            Console.WriteLine($"Injected {numVisionTokens} vision tokens at chunk-offset {insertPos} (absolute position {startPos + insertPos})");
        }

        private bool CanUseFusedLayerGraph(int seqLen, HashSet<int> exceptPositions, bool enabled)
        {
            return enabled
                && IsGgmlBackend
                && seqLen > 0
                && exceptPositions == null
                && _canUseFusedDecode;
        }

        #region Fused Decode

        private class Gemma4DecodeArrays
        {
            public IntPtr[] AttnNorm, Qkv, QNorm, KNorm, O, PostAttnNorm;
            public IntPtr[] FfnNorm, Gu, Down, PostFfnNorm;
            public IntPtr[] KCache, VCache;
            public int[] HeadDim, KvHeads, CacheSize, IsLocal, KvSource, RopeNDims;
            public float[] RopeBase, LayerScalar;
            public int[] QkvType; public long[] QkvNe0, QkvNe1, QkvBytes;
            // Separate K/V projection weights for mixed-quant layers (null
            // pointer entry => layer uses the fused Qkv weight instead).
            public IntPtr[] K, V;
            public int[] KType; public long[] KNe0, KNe1, KBytes;
            public int[] VType; public long[] VNe0, VNe1, VBytes;
            public int[] OType; public long[] ONe0, ONe1, OBytes;
            public int[] GuType; public long[] GuNe0, GuNe1, GuBytes;
            public int[] DownType; public long[] DownNe0, DownNe1, DownBytes;
            // PLE
            public IntPtr[] PleGate, PleProj, PlePostNorm;
            public int[] PleGateType, PleProjType;
            public long[] PleGateNe0, PleGateNe1, PleGateBytes;
            public long[] PleProjNe0, PleProjNe1, PleProjBytes;
        }

        private unsafe void BuildGemma4DecodeArrays()
        {
            if (!IsGgmlBackend) return;

            bool anyMoE = false;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (HasMoE(l)) { anyMoE = true; break; }
            }

            if (anyMoE)
            {
                // MoE Gemma 4 (e.g. gemma-4-26B-A4B) cannot use the model-wide
                // single-graph decode kernel (NativeGemma4ModelDecode) because
                // that kernel has no MoE branch — it would need to embed a
                // ggml_mul_mat_id-based MoE FFN inside its giant graph. Until
                // that lands, leave _canUseFusedFullModelDecode = false so the
                // forward path falls back to the per-layer dispatch loop.
                //
                // We deliberately still build _decodeArrays and set
                // _canUseFusedDecode = true (further down) so the per-layer
                // fused prefill / decode kernel (TSGgml_Gemma4LayerPrefill) is
                // available to the dense layers in the model. The per-layer
                // call site is guarded by !HasMoE(l), so MoE layers fall
                // through to the C# TransformerBlock — which now itself runs
                // the post-MoE-norm + residual add through the fused
                // Gemma4MoEGEGLUResidual kernel rather than two extra device
                // dispatches. Net effect: the dense majority of layers gets
                // the same fused-graph speedup as a non-MoE Gemma 4 model,
                // and MoE layers shed two dispatches each.
                _canUseFusedFullModelDecode = false;

                // The C# MoE / managed-attention fallback path mixes CPU loads
                // / stores with device kernels — for example the legacy
                // per-expert FFNGelu loop in MoEForward builds `batchInput`
                // via Buffer.MemoryCopy on the host pointer and then
                // immediately dispatches FFNGelu on the device. The Metal
                // lazy-sync optimisation that GgmlContext enables by default
                // (see <see cref="GgmlBasicOps.SetAsyncCompute"/>) is only
                // safe for pure CPU-after-GPU read patterns: GetFloatPtr /
                // EnsureHostReadable drains pending work before returning the
                // host pointer, but there is no host_write_barrier counterpart
                // to make CPU writes visible to the next GPU op. Under async
                // compute the next kernel can dispatch against a stale cached
                // device buffer for the same host pointer, leaving a portion
                // of the residual stream effectively zeroed on every layer.
                // The user-visible symptom on Apple Silicon Metal is
                // repetitive / off-topic output for any non-trivial prompt
                // (the model still emits coherent token-by-token text but
                // loses its conditional content). Disable async compute for
                // the lifetime of this model so each per-op kernel blocks on
                // `ggml_backend_synchronize` before returning.
                if (GgmlBasicOps.GetAsyncCompute())
                    GgmlBasicOps.SetAsyncCompute(false);
            }
            else
            {
                _canUseFusedFullModelDecode = true;
            }

            // The async / lazy-sync hazard above is NOT specific to MoE. Every
            // Gemma 4 variant's seqLen>1 prefill mixes CPU writes with device
            // kernels: the global-attention fast path
            // (SplitQKVToHeadFirst + ApplyNeoXRoPEHeadFirst) and the flat
            // ApplyNeoXRoPEPrefill populate Q/K/V on the CPU via Parallel.For,
            // and the following RMSNorm / attention op reads them on the GPU.
            // Under Metal async compute there is no host-write barrier, so that
            // GPU op can run against a stale device buffer and silently drop the
            // prompt's contribution to the residual stream — the model then emits
            // coherent but off-topic text (e.g. defining a word lifted from the
            // prompt instead of answering it). This bites dense, non-KV-shared
            // builds such as gemma-4-12b-it whose global layers take the CPU-write
            // fast path. Decode runs as a single fused graph, so forcing the
            // per-op synchronize falls almost entirely on one-time prefill.
            if (GgmlBasicOps.GetAsyncCompute())
                GgmlBasicOps.SetAsyncCompute(false);

            int n = Config.NumLayers;

            // Verify all *non-MoE* layers expose the quantized attention
            // projection weights the fused kernel needs. MoE layers are
            // dispatched via the C# TransformerBlock so they don't need the
            // precomputed arrays — the loop below leaves their Qkv slot at
            // IntPtr.Zero and the per-layer call site bails on HasMoE(l)
            // before touching it.
            //
            // Two valid shapes per non-shared layer:
            //   * fused  : attn_qkv.weight  (Q/K/V share one ggml type)
            //   * separate: attn_q + attn_k + attn_v.weight  (mixed-quant
            //     models such as UD-IQ2_M where Q/K/V carry different types
            //     and cannot be fused — the kernel runs three matmuls)
            // Shared (KV-donor-following) layers only project Q. If a dense
            // layer is missing even the separate set, disable fused dispatch
            // entirely to keep the dispatch decision simple.
            // Tracks whether EVERY non-shared dense layer carries a fused
            // attn_qkv weight. The per-layer fused kernel
            // (TSGgml_Gemma4LayerPrefill) only understands a fused QKV weight,
            // so it stays gated on this. The full-model decode kernel handles
            // separate Q/K/V, so it relaxes to "fused OR separate" below.
            bool allLayersFusedQkv = true;
            for (int l = 0; l < n; l++)
            {
                if (HasMoE(l)) continue;
                string prefix = $"blk.{l}";
                bool isShared = _kvDonorMap.ContainsKey(l);
                bool ok;
                if (isShared)
                {
                    ok = _quantWeights.ContainsKey($"{prefix}.attn_q.weight");
                }
                else
                {
                    bool fused = _quantWeights.ContainsKey($"{prefix}.attn_qkv.weight");
                    bool separate = _quantWeights.ContainsKey($"{prefix}.attn_q.weight")
                        && _quantWeights.ContainsKey($"{prefix}.attn_k.weight")
                        && _quantWeights.ContainsKey($"{prefix}.attn_v.weight");
                    ok = fused || separate;
                    if (!fused)
                        allLayersFusedQkv = false;
                }
                if (!ok)
                {
                    _canUseFusedDecode = false;
                    _canUseFusedFullModelDecode = false;
                    return;
                }
            }

            var a = new Gemma4DecodeArrays();
            a.AttnNorm = new IntPtr[n]; a.Qkv = new IntPtr[n]; a.QNorm = new IntPtr[n]; a.KNorm = new IntPtr[n];
            a.O = new IntPtr[n]; a.PostAttnNorm = new IntPtr[n];
            a.FfnNorm = new IntPtr[n]; a.Gu = new IntPtr[n]; a.Down = new IntPtr[n]; a.PostFfnNorm = new IntPtr[n];
            a.KCache = new IntPtr[n]; a.VCache = new IntPtr[n];
            a.HeadDim = new int[n]; a.KvHeads = new int[n]; a.CacheSize = new int[n]; a.IsLocal = new int[n];
            a.KvSource = new int[n]; a.RopeNDims = new int[n];
            a.RopeBase = new float[n]; a.LayerScalar = new float[n];
            a.QkvType = new int[n]; a.QkvNe0 = new long[n]; a.QkvNe1 = new long[n]; a.QkvBytes = new long[n];
            a.K = new IntPtr[n]; a.KType = new int[n]; a.KNe0 = new long[n]; a.KNe1 = new long[n]; a.KBytes = new long[n];
            a.V = new IntPtr[n]; a.VType = new int[n]; a.VNe0 = new long[n]; a.VNe1 = new long[n]; a.VBytes = new long[n];
            a.OType = new int[n]; a.ONe0 = new long[n]; a.ONe1 = new long[n]; a.OBytes = new long[n];
            a.GuType = new int[n]; a.GuNe0 = new long[n]; a.GuNe1 = new long[n]; a.GuBytes = new long[n];
            a.DownType = new int[n]; a.DownNe0 = new long[n]; a.DownNe1 = new long[n]; a.DownBytes = new long[n];
            a.PleGate = new IntPtr[n]; a.PleProj = new IntPtr[n]; a.PlePostNorm = new IntPtr[n];
            a.PleGateType = new int[n]; a.PleProjType = new int[n];
            a.PleGateNe0 = new long[n]; a.PleGateNe1 = new long[n]; a.PleGateBytes = new long[n];
            a.PleProjNe0 = new long[n]; a.PleProjNe1 = new long[n]; a.PleProjBytes = new long[n];

            for (int l = 0; l < n; l++)
            {
                string prefix = $"blk.{l}";
                bool isShared = _kvDonorMap.ContainsKey(l);
                int kvSource = _kvDonorMap.TryGetValue(l, out int donor) ? donor : l;
                bool isLocal = IsLocalLayer(kvSource);
                int hd = HeadDimForLayer(l);
                int kvH = KVHeadsForLayer(l);

                a.HeadDim[l] = hd;
                a.KvHeads[l] = kvH;
                a.CacheSize[l] = _kvCacheSize[kvSource];
                a.IsLocal[l] = isLocal ? 1 : 0;
                a.KvSource[l] = kvSource;
                a.RopeBase[l] = IsLocalLayer(l) ? _ropeLocalBase : _ropeGlobalBase;
                a.RopeNDims[l] = IsLocalLayer(l) ? _localHeadDim : _partialRotaryDims;
                a.LayerScalar[l] = _layerScalars[l];

                a.AttnNorm[l] = (IntPtr)GetFloatPtr(_weights[$"{prefix}.attn_norm.weight"]);
                a.QNorm[l] = (IntPtr)GetFloatPtr(_weights[$"{prefix}.attn_q_norm.weight"]);
                a.KCache[l] = TensorComputePrimitives.GetStoragePointer(_kvCacheK[kvSource]);
                a.VCache[l] = TensorComputePrimitives.GetStoragePointer(_kvCacheV[kvSource]);

                if (!isShared)
                    a.KNorm[l] = (IntPtr)GetFloatPtr(_weights[$"{prefix}.attn_k_norm.weight"]);

                // Post-attention norm
                string postAttnKey = _weights.ContainsKey($"{prefix}.post_attention_norm.weight")
                    ? $"{prefix}.post_attention_norm.weight" : $"{prefix}.attn_post_norm.weight";
                a.PostAttnNorm[l] = (IntPtr)GetFloatPtr(_weights[postAttnKey]);

                // FFN norm
                a.FfnNorm[l] = (IntPtr)GetFloatPtr(_weights[$"{prefix}.ffn_norm.weight"]);

                // Post-FFN norm
                string postFfnKey = _weights.ContainsKey($"{prefix}.post_ffw_norm.weight")
                    ? $"{prefix}.post_ffw_norm.weight" : $"{prefix}.ffn_post_norm.weight";
                a.PostFfnNorm[l] = (IntPtr)GetFloatPtr(_weights[postFfnKey]);

                // For shared layers, use Q-only weight; for non-shared, use fused QKV
                if (isShared)
                {
                    string qName = $"{prefix}.attn_q.weight";
                    if (_quantWeights.TryGetValue(qName, out var qW))
                    {
                        a.Qkv[l] = qW.CacheKey;
                        a.QkvType[l] = qW.GgmlType;
                        a.QkvNe0[l] = qW.Ne0;
                        a.QkvNe1[l] = qW.Ne1;
                        a.QkvBytes[l] = qW.RawBytes;
                    }
                }
                else
                {
                    string qkvName = $"{prefix}.attn_qkv.weight";
                    if (_quantWeights.TryGetValue(qkvName, out var qkvW))
                    {
                        a.Qkv[l] = qkvW.CacheKey;
                        a.QkvType[l] = qkvW.GgmlType;
                        a.QkvNe0[l] = qkvW.Ne0;
                        a.QkvNe1[l] = qkvW.Ne1;
                        a.QkvBytes[l] = qkvW.RawBytes;
                    }
                    else if (_quantWeights.TryGetValue($"{prefix}.attn_q.weight", out var qW)
                        && _quantWeights.TryGetValue($"{prefix}.attn_k.weight", out var kW)
                        && _quantWeights.TryGetValue($"{prefix}.attn_v.weight", out var vW))
                    {
                        // Mixed-quant layer: Q/K/V carry different ggml types,
                        // so the kernel runs three separate matmuls. Qkv slot
                        // holds Q; K/V go in their own arrays (a non-null K
                        // pointer is the native kernel's "separate" signal).
                        a.Qkv[l] = qW.CacheKey;
                        a.QkvType[l] = qW.GgmlType;
                        a.QkvNe0[l] = qW.Ne0;
                        a.QkvNe1[l] = qW.Ne1;
                        a.QkvBytes[l] = qW.RawBytes;

                        a.K[l] = kW.CacheKey;
                        a.KType[l] = kW.GgmlType;
                        a.KNe0[l] = kW.Ne0;
                        a.KNe1[l] = kW.Ne1;
                        a.KBytes[l] = kW.RawBytes;

                        a.V[l] = vW.CacheKey;
                        a.VType[l] = vW.GgmlType;
                        a.VNe0[l] = vW.Ne0;
                        a.VNe1[l] = vW.Ne1;
                        a.VBytes[l] = vW.RawBytes;
                    }
                }

                string oName = $"{prefix}.attn_output.weight";
                if (_quantWeights.TryGetValue(oName, out var oW))
                {
                    a.O[l] = oW.CacheKey;
                    a.OType[l] = oW.GgmlType;
                    a.ONe0[l] = oW.Ne0;
                    a.ONe1[l] = oW.Ne1;
                    a.OBytes[l] = oW.RawBytes;
                }

                string guName = $"{prefix}.ffn_gate_up.weight";
                if (_quantWeights.TryGetValue(guName, out var guW))
                {
                    a.Gu[l] = guW.CacheKey;
                    a.GuType[l] = guW.GgmlType;
                    a.GuNe0[l] = guW.Ne0;
                    a.GuNe1[l] = guW.Ne1;
                    a.GuBytes[l] = guW.RawBytes;
                }

                string downName = $"{prefix}.ffn_down.weight";
                if (_quantWeights.TryGetValue(downName, out var downW))
                {
                    a.Down[l] = downW.CacheKey;
                    a.DownType[l] = downW.GgmlType;
                    a.DownNe0[l] = downW.Ne0;
                    a.DownNe1[l] = downW.Ne1;
                    a.DownBytes[l] = downW.RawBytes;
                }

                // PLE weights (optional) - check both quantized and F32 dictionaries
                string pleGateName = $"{prefix}.inp_gate.weight";
                bool hasPleGate = false;
                if (_quantWeights.TryGetValue(pleGateName, out var pleGW))
                {
                    a.PleGate[l] = pleGW.CacheKey;
                    a.PleGateType[l] = pleGW.GgmlType;
                    a.PleGateNe0[l] = pleGW.Ne0;
                    a.PleGateNe1[l] = pleGW.Ne1;
                    a.PleGateBytes[l] = pleGW.RawBytes;
                    hasPleGate = true;
                }
                else if (_weights.TryGetValue(pleGateName, out var pleGateF32))
                {
                    a.PleGate[l] = (IntPtr)GetFloatPtr(pleGateF32);
                    a.PleGateType[l] = 0; // GGML_TYPE_F32
                    a.PleGateNe0[l] = pleGateF32.Sizes[1];
                    a.PleGateNe1[l] = pleGateF32.Sizes[0];
                    a.PleGateBytes[l] = pleGateF32.ElementCount() * 4;
                    hasPleGate = true;
                }

                if (hasPleGate)
                {
                    string pleProjName = $"{prefix}.proj.weight";
                    if (_quantWeights.TryGetValue(pleProjName, out var plePW))
                    {
                        a.PleProj[l] = plePW.CacheKey;
                        a.PleProjType[l] = plePW.GgmlType;
                        a.PleProjNe0[l] = plePW.Ne0;
                        a.PleProjNe1[l] = plePW.Ne1;
                        a.PleProjBytes[l] = plePW.RawBytes;
                    }
                    else if (_weights.TryGetValue(pleProjName, out var pleProjF32))
                    {
                        a.PleProj[l] = (IntPtr)GetFloatPtr(pleProjF32);
                        a.PleProjType[l] = 0; // GGML_TYPE_F32
                        a.PleProjNe0[l] = pleProjF32.Sizes[1];
                        a.PleProjNe1[l] = pleProjF32.Sizes[0];
                        a.PleProjBytes[l] = pleProjF32.ElementCount() * 4;
                    }

                    string plePostNormName = $"{prefix}.post_norm.weight";
                    if (_weights.ContainsKey(plePostNormName))
                        a.PlePostNorm[l] = (IntPtr)GetFloatPtr(_weights[plePostNormName]);
                }
            }

            _decodeArrays = a;
            // The per-layer fused kernel only supports fused QKV; mixed-quant
            // (separate Q/K/V) models rely solely on the full-model decode
            // kernel, which handles them. Keeping this false for mixed-quant
            // preserves the existing MoE / chunked-prefill fallbacks.
            _canUseFusedDecode = allLayersFusedQkv;
            Console.WriteLine(_canUseFusedFullModelDecode
                ? (allLayersFusedQkv
                    ? "  Gemma4 fused model decode enabled"
                    : "  Gemma4 fused model decode enabled (separate Q/K/V for mixed-quant layers)")
                : "  Gemma4 fused dense-layer graph enabled (MoE layers use hybrid path)");
        }

        private unsafe void NativeGemma4ModelDecode(Tensor hidden, int startPos, Tensor perLayerInputs)
        {
            float* hiddenPtr = GetFloatPtr(hidden);
            var a = _decodeArrays;

            IntPtr pleDataPtr = IntPtr.Zero;
            if (perLayerInputs != null)
                pleDataPtr = (IntPtr)GetFloatPtr(perLayerInputs);

            IntPtr freqFactorsPtr = IntPtr.Zero;
            int freqFactorsLen = 0;
            if (_weights.TryGetValue("rope_freqs.weight", out var freqTensor))
            {
                freqFactorsPtr = (IntPtr)GetFloatPtr(freqTensor);
                freqFactorsLen = (int)freqTensor.ElementCount();
            }

            GgmlBasicOps.Gemma4ModelDecode(
                (IntPtr)hiddenPtr, Config.HiddenSize, Config.NumLayers,
                a.AttnNorm, a.Qkv, a.QNorm, a.KNorm,
                a.O, a.PostAttnNorm,
                a.FfnNorm, a.Gu, a.Down, a.PostFfnNorm,
                a.KCache, a.VCache,
                a.HeadDim, a.KvHeads, a.CacheSize, a.IsLocal,
                a.KvSource,
                a.RopeBase, a.LayerScalar,
                a.QkvType, a.QkvNe0, a.QkvNe1, a.QkvBytes,
                a.OType, a.ONe0, a.ONe1, a.OBytes,
                a.GuType, a.GuNe0, a.GuNe1, a.GuBytes,
                a.DownType, a.DownNe0, a.DownNe1, a.DownBytes,
                Config.NumHeads, startPos,
                Config.Eps, _slidingWindow,
                freqFactorsPtr, freqFactorsLen,
                a.RopeNDims,
                pleDataPtr, _pleDim,
                a.PleGate, a.PleGateType, a.PleGateNe0, a.PleGateNe1, a.PleGateBytes,
                a.PleProj, a.PleProjType, a.PleProjNe0, a.PleProjNe1, a.PleProjBytes,
                a.PlePostNorm,
                _kvCacheDtype.GgmlType(),
                a.K, a.KType, a.KNe0, a.KNe1, a.KBytes,
                a.V, a.VType, a.VNe0, a.VNe1, a.VBytes);
        }

        /// <summary>
        /// Decode (seqLen == 1) one entire Gemma 4 MoE transformer block as a
        /// single fused GGML graph: attention (norm → QKV → QK/V-norm → RoPE →
        /// KV-cache write → flash attention → O-proj → post-attn-norm → residual)
        /// + dense shared FFN + in-graph-routed experts + post norms/residual.
        ///
        /// Replaces the ~18-20 per-op dispatches that the C# TransformerBlock
        /// issues per MoE layer — each of which allocates+frees a Metal buffer
        /// and forces a queue synchronise — with ONE device graph. This is the
        /// dominant decode bottleneck for MoE Gemma 4 (attention runs on the CPU
        /// in the per-op path and grows with KV length). Returns false (caller
        /// falls back to TransformerBlock) when any required weight is missing or
        /// the layout is unsupported.
        ///
        /// The in-graph router mirrors <see cref="MoERoute"/> + the per-expert
        /// scale fold in <see cref="TryMoEForwardResidual"/> exactly, so output
        /// is numerically equivalent to the per-op path.
        /// </summary>
        private unsafe bool TryFusedMoELayerDecode(Tensor hidden, int layer, int startPos)
        {
            if (!IsGgmlBackend || _decodeArrays == null) return false;
            // Requires the stacked (fused-gate_up) expert weights; the separate
            // gate/up expert layout isn't handled by this kernel.
            if (_layerStackedGate == null || _layerStackedGate[layer] == null
                || _layerStackedDown == null || _layerStackedDown[layer] == null)
                return false;
            if (_layerStackedUp != null && _layerStackedUp[layer] != null) return false;

            // Block-quantized KV cache is written via ggml_cpy(F32->cacheType)
            // by the kernel; only F32/F16 caches are wired here.
            if (_kvCacheDtype.IsBlockQuantized()) return false;

            var a = _decodeArrays;
            // Attention weights must be present in the precomputed arrays.
            if (a.Qkv[layer] == IntPtr.Zero || a.O[layer] == IntPtr.Zero
                || a.Gu[layer] == IntPtr.Zero || a.Down[layer] == IntPtr.Zero)
                return false;

            string prefix = $"blk.{layer}";

            if (!_weights.TryGetValue($"{prefix}.ffn_gate_inp.weight", out var routerW))
                return false;
            string preNorm2Key = _weights.ContainsKey($"{prefix}.pre_ffw_norm_2.weight")
                ? $"{prefix}.pre_ffw_norm_2.weight" : $"{prefix}.ffn_pre_norm_2.weight";
            if (!_weights.TryGetValue(preNorm2Key, out var preNorm2W)) return false;
            string postNorm1Key = _weights.ContainsKey($"{prefix}.post_ffw_norm_1.weight")
                ? $"{prefix}.post_ffw_norm_1.weight" : $"{prefix}.ffn_post_norm_1.weight";
            if (!_weights.TryGetValue(postNorm1Key, out var postNorm1W)) return false;
            string postNorm2Key = _weights.ContainsKey($"{prefix}.post_ffw_norm_2.weight")
                ? $"{prefix}.post_ffw_norm_2.weight" : $"{prefix}.ffn_post_norm_2.weight";
            if (!_weights.TryGetValue(postNorm2Key, out var postNorm2W)) return false;

            bool isLocal = IsLocalLayer(layer);
            bool isShared = _kvDonorMap.ContainsKey(layer);
            int H = Config.HiddenSize;

            // Proportional RoPE freq factors apply to global layers only.
            IntPtr freqPtr = IntPtr.Zero;
            int freqLen = 0;
            if (!isLocal && _weights.TryGetValue("rope_freqs.weight", out var freqTensor))
            {
                freqPtr = (IntPtr)GetFloatPtr(freqTensor);
                freqLen = (int)freqTensor.ElementCount();
            }

            // Optional routing-input per-dim scale and per-expert post-down scale.
            IntPtr gateInpScalePtr = IntPtr.Zero;
            if (_weights.TryGetValue($"{prefix}.ffn_gate_inp.scale", out var giScale))
                gateInpScalePtr = (IntPtr)GetFloatPtr(giScale);
            IntPtr downScalePtr = IntPtr.Zero;
            string downScaleKey = _weights.ContainsKey($"{prefix}.ffn_down_exps.scale")
                ? $"{prefix}.ffn_down_exps.scale" : $"{prefix}.ffn_gate_inp.per_expert_scale";
            if (_weights.TryGetValue(downScaleKey, out var downScaleT))
                downScalePtr = (IntPtr)GetFloatPtr(downScaleT);

            var gateW = _layerStackedGate[layer];
            var downW = _layerStackedDown[layer];

            var args = new Gemma4MoELayerDecodeArgs
            {
                Hidden = (IntPtr)GetFloatPtr(hidden),
                AttnNormW = a.AttnNorm[layer],
                QkvW = a.Qkv[layer], QkvType = a.QkvType[layer], QkvNe0 = a.QkvNe0[layer], QkvNe1 = a.QkvNe1[layer], QkvBytes = a.QkvBytes[layer],
                KW = a.K[layer], KType = a.KType[layer], KNe0 = a.KNe0[layer], KNe1 = a.KNe1[layer], KBytes = a.KBytes[layer],
                VW = a.V[layer], VType = a.VType[layer], VNe0 = a.VNe0[layer], VNe1 = a.VNe1[layer], VBytes = a.VBytes[layer],
                QNormW = a.QNorm[layer],
                KNormW = isShared ? IntPtr.Zero : a.KNorm[layer],
                OW = a.O[layer], OType = a.OType[layer], ONe0 = a.ONe0[layer], ONe1 = a.ONe1[layer], OBytes = a.OBytes[layer],
                PostAttnNormW = a.PostAttnNorm[layer],
                KCache = a.KCache[layer], VCache = a.VCache[layer],
                FreqFactors = freqPtr, FreqFactorsLen = freqLen,
                FfnNormW = a.FfnNorm[layer],
                GuW = a.Gu[layer], GuType = a.GuType[layer], GuNe0 = a.GuNe0[layer], GuNe1 = a.GuNe1[layer], GuBytes = a.GuBytes[layer],
                DownW = a.Down[layer], DownType = a.DownType[layer], DownNe0 = a.DownNe0[layer], DownNe1 = a.DownNe1[layer], DownBytes = a.DownBytes[layer],
                PostFfwNorm1W = (IntPtr)GetFloatPtr(postNorm1W),
                GateInpW = (IntPtr)GetFloatPtr(routerW),
                GateInpScale = gateInpScalePtr,
                PreFfwNorm2W = (IntPtr)GetFloatPtr(preNorm2W),
                GateUpExps = gateW.Data, GueType = gateW.GgmlType, GueNe0 = gateW.PerExpertNe0, GueNe1 = gateW.PerExpertNe1, GueBytes = gateW.TotalRawBytes,
                DownExps = downW.Data, DeType = downW.GgmlType, DeNe0 = downW.PerExpertNe0, DeNe1 = downW.PerExpertNe1, DeBytes = downW.TotalRawBytes,
                DownExpsScale = downScalePtr,
                PostFfwNorm2W = (IntPtr)GetFloatPtr(postNorm2W),
                PostFfwNormW = a.PostFfnNorm[layer],

                StructBytes = Marshal.SizeOf<Gemma4MoELayerDecodeArgs>(),
                HiddenSize = H,
                NumHeads = Config.NumHeads,
                NumKvHeads = a.KvHeads[layer],
                HeadDim = a.HeadDim[layer],
                CacheSize = a.CacheSize[layer],
                IsLocal = a.IsLocal[layer],
                IsShared = isShared ? 1 : 0,
                SlidingWindow = _slidingWindow,
                Position = startPos,
                RopeNDims = a.RopeNDims[layer],
                KvCacheType = _kvCacheDtype.GgmlType(),
                NumExperts = _numExperts,
                NumExpertsUsed = _numExpertsUsed,
                SeparateQkv = a.K[layer] != IntPtr.Zero ? 1 : 0,

                Eps = Config.Eps,
                RopeBase = a.RopeBase[layer],
                InvSqrtHidden = 1f / MathF.Sqrt(H),
                LayerOutputScale = a.LayerScalar[layer],
            };

            try
            {
                GgmlBasicOps.Gemma4MoELayerDecode(in args);
            }
            catch (Exception ex)
            {
                // Disable for the rest of this model's lifetime and fall back to
                // the per-op path so a kernel issue degrades to (slow) correctness
                // rather than failing generation.
                System.Console.WriteLine($"[gemma4] fused MoE layer decode disabled after error: {ex.Message}");
                _moeFusedDecodeEnabled = false;
                return false;
            }
            // The kernel writes K/V into the device-local cached buffer (bound
            // with USAGE_COMPUTE), so the host copy is now stale — mark dirty so
            // any later host read of the cache syncs first (mirrors the fused
            // full-model decode path).
            _kvCacheHostDirty = true;
            return true;
        }

        /// <summary>
        /// Run a single Gemma4 transformer layer as one fused GGML graph.
        ///
        /// Replaces ~10 separate dispatches (LinearForward QKV/Norm/RoPE/CopyToCache/
        /// FusedAttn/LinearO + FFN + post-norms + residual + PLE) with a single GPU
        /// graph compute. This is the critical path for prefill speed because the
        /// per-dispatch overhead on Metal/CUDA dominates short-context layers.
        ///
        /// Supports:
        /// - PLE injection (mandatory for E4B - the previous fused path skipped any
        ///   layer with PLE which meant E4B never used the fused path).
        /// - Chunked prefill: SWA layers in chunks 2+ get the previous-window K/V
        ///   gathered from the rolling cache; full-attention layers get a 0-copy
        ///   view of the cache prefix. Both are concatenated with fresh K/V inside
        ///   the graph so attention is correct across chunk boundaries.
        /// </summary>
        private unsafe bool TryFusedLayerPrefill(Tensor hidden, int layer, int seqLen, int startPos,
            Tensor perLayerInput)
        {
            if (_decodeArrays == null) return false;
            // The fused prefill kernel binds the K/V cache as F32 or F16 based on
            // _kvCacheDtype. Fresh K/V is always F32 inside the kernel; the cache
            // write goes through ggml_cpy(F32->cacheType) and the global-prev path
            // materializes the F16 cache view as F32 before concatenating with fresh.
            var a = _decodeArrays;

            string prefix = $"blk.{layer}";
            string postAttnKey = $"{prefix}.post_attention_norm.weight";
            string postFfnKey = _weights.ContainsKey($"{prefix}.post_ffw_norm.weight")
                ? $"{prefix}.post_ffw_norm.weight" : $"{prefix}.ffn_post_norm.weight";

            if (!_weights.ContainsKey(postAttnKey) || !_weights.ContainsKey(postFfnKey))
                return false;

            bool isLocal = IsLocalLayer(layer);
            bool isShared = _kvDonorMap.ContainsKey(layer);

            // Correctness guard: the fused per-layer prefill kernel
            // (TSGgml_Gemma4LayerPrefill) miscomputes sliding-window attention for
            // SWA (local) layers once the sequence reaches past the window — query
            // positions p >= slidingWindow must have their early keys masked out,
            // and the fused kernel's windowed path corrupts the hidden state, so
            // the model emits garbage for any prompt longer than the window.
            //
            // The corruption cannot be contained by falling back only for the SWA
            // layers or only for the chunk that crosses the window. Prefill is
            // processed chunk-by-chunk and the fused vs. per-op paths are not
            // interchangeable mid-sequence: fused chunks leave K/V on the device
            // (and write the circular SWA cache differently than the per-op path),
            // so mixing fused and per-op work within a prefill corrupts the later
            // (windowed) chunks. Empirically the corruption appears for any prompt
            // longer than the sliding window and worsens with length.
            //
            // Keep multi-token prefill (seqLen > 1) entirely on the per-op path
            // (TransformerBlock -> FusedPrefillAttention, which masks the window
            // correctly and is verified correct at every length). Single-token
            // decode (seqLen == 1) is unaffected — it is served by the fused
            // full-model decode kernel (NativeGemma4ModelDecode) over the circular
            // cache, never this per-layer prefill path — so generation speed is
            // unchanged; only multi-token prefill loses the per-layer graph fusion.
            if (seqLen > 1)
                return false;

            int donorLayer = isShared ? _kvDonorMap[layer] : layer;
            var (ropeBase, ropeDims) = RopeForLayer(layer);

            IntPtr freqFactorsPtr = IntPtr.Zero;
            int freqFactorsLen = 0;
            if (!isLocal && _weights.TryGetValue("rope_freqs.weight", out var freqTensor))
            {
                freqFactorsPtr = (IntPtr)GetFloatPtr(freqTensor);
                freqFactorsLen = (int)freqTensor.ElementCount();
            }

            // SWA prev-window data (from PrepareSwaPrevWindowsForChunk). Only set when
            // we're in chunk 2+ of a prefill run on a sliding-window layer that
            // computes its own K/V (shared layers reuse the donor's K/V which
            // already includes the prev window).
            IntPtr swaPrevKPtr = IntPtr.Zero;
            IntPtr swaPrevVPtr = IntPtr.Zero;
            int prevWindowLen = 0;
            if (isLocal && !isShared && startPos > 0 && _swaPrevWindow != null
                && _swaPrevWindow.TryGetValue(layer, out var prevKv) && _swaPrevWindowLen > 0)
            {
                swaPrevKPtr = (IntPtr)GetFloatPtr(prevKv.k);
                swaPrevVPtr = (IntPtr)GetFloatPtr(prevKv.v);
                prevWindowLen = _swaPrevWindowLen;
            }

            // Shared layer: reuse donor's already-computed (and SWA-prev-extended)
            // K/V. The donor must have already run in this same chunk and saved
            // its fresh K/V into _prefillSWAKV. For SWA chunks 2+ we additionally
            // prepend the gathered previous window so the kernel sees the same
            // attention context the C# fallback would see for shared SWA layers.
            IntPtr donorKPtr = IntPtr.Zero;
            IntPtr donorVPtr = IntPtr.Zero;
            int donorKvLen = 0;
            Tensor sharedDonorK = null, sharedDonorV = null;
            bool ownsSharedDonor = false;
            if (isShared)
            {
                if (_prefillSWAKV == null || !_prefillSWAKV.TryGetValue(donorLayer, out var donorKv))
                {
                    // Donor hasn't published K/V (e.g. kernel was disabled for
                    // this run, or this layer's donor lookup happened before
                    // the donor's TryFusedLayerPrefill ran). Fall back to C#.
                    return false;
                }

                if (isLocal && startPos > 0 && _swaPrevWindow != null
                    && _swaPrevWindow.TryGetValue(donorLayer, out var sharedPrev)
                    && _swaPrevWindowLen > 0)
                {
                    sharedDonorK = ConcatHeadFirstKV(sharedPrev.k, donorKv.k);
                    sharedDonorV = ConcatHeadFirstKV(sharedPrev.v, donorKv.v);
                    donorKvLen = _swaPrevWindowLen + seqLen;
                    ownsSharedDonor = true;
                    donorKPtr = (IntPtr)GetFloatPtr(sharedDonorK);
                    donorVPtr = (IntPtr)GetFloatPtr(sharedDonorV);
                }
                else
                {
                    donorKPtr = (IntPtr)GetFloatPtr(donorKv.k);
                    donorVPtr = (IntPtr)GetFloatPtr(donorKv.v);
                    donorKvLen = (int)donorKv.k.Sizes[1];
                }
            }

            // PLE inputs (per-layer slice of the precomputed PLE state).
            IntPtr pleInputPtr = IntPtr.Zero;
            IntPtr pleGateW = IntPtr.Zero, pleProjW = IntPtr.Zero, plePostNormW = IntPtr.Zero;
            int pleGateType = 0, pleProjType = 0;
            long pleGateNe0 = 0, pleGateNe1 = 0, pleGateBytes = 0;
            long pleProjNe0 = 0, pleProjNe1 = 0, pleProjBytes = 0;
            if (perLayerInput != null && a.PleGate[layer] != IntPtr.Zero && a.PleProj[layer] != IntPtr.Zero
                && a.PlePostNorm[layer] != IntPtr.Zero)
            {
                pleInputPtr = (IntPtr)GetFloatPtr(perLayerInput);
                pleGateW = a.PleGate[layer];
                pleProjW = a.PleProj[layer];
                plePostNormW = a.PlePostNorm[layer];
                pleGateType = a.PleGateType[layer];
                pleGateNe0 = a.PleGateNe0[layer];
                pleGateNe1 = a.PleGateNe1[layer];
                pleGateBytes = a.PleGateBytes[layer];
                pleProjType = a.PleProjType[layer];
                pleProjNe0 = a.PleProjNe0[layer];
                pleProjNe1 = a.PleProjNe1[layer];
                pleProjBytes = a.PleProjBytes[layer];
            }
            else if (perLayerInput != null)
            {
                // Caller asked for PLE but the precomputed weights aren't all present
                // (e.g. a layer-specific config we don't yet handle). Fall back to the
                // C# path which understands every Gemma4 PLE variant.
                return false;
            }

            // Donor publish: non-shared SWA donors that downstream KV-shared
            // layers depend on get host buffers the kernel writes fresh K/V
            // into. The C# attention path (and the kernel's shared-layer mode
            // below) hand them to shared layers via _prefillSWAKV instead of
            // forcing reads from the rolling cache (which loses early positions
            // when seqLen > slidingWindow).
            Tensor freshKBuffer = null;
            Tensor freshVBuffer = null;
            IntPtr freshKOutPtr = IntPtr.Zero;
            IntPtr freshVOutPtr = IntPtr.Zero;
            int kvHeadsLayer = a.KvHeads[layer];
            int hdLayer = a.HeadDim[layer];
            // Donor publish: layers that downstream KV-shared layers depend on
            // get host buffers the kernel writes fresh post-norm/RoPE K/V into.
            // Both SWA (local) and global donors must publish, otherwise their
            // shared layers bail out to the C# managed path (which can't read
            // a Q8_0 cache). The donor list `_swaKVDonorLayers` already tracks
            // both kinds; the local-vs-global distinction matters only inside
            // the kernel itself.
            if (!isShared && _swaKVDonorLayers != null && _swaKVDonorLayers.Contains(layer)
                && _prefillSWAKV != null)
            {
                freshKBuffer = new Tensor(_allocator, DType.Float32, kvHeadsLayer, seqLen, hdLayer);
                freshVBuffer = new Tensor(_allocator, DType.Float32, kvHeadsLayer, seqLen, hdLayer);
                freshKOutPtr = (IntPtr)GetFloatPtr(freshKBuffer);
                freshVOutPtr = (IntPtr)GetFloatPtr(freshVBuffer);
            }

            try
            {
                GgmlBasicOps.Gemma4LayerPrefill(
                    (IntPtr)GetFloatPtr(hidden), Config.HiddenSize, seqLen,
                    (IntPtr)GetFloatPtr(_weights[$"{prefix}.attn_norm.weight"]),
                    a.Qkv[layer], a.QkvType[layer], a.QkvNe0[layer], a.QkvNe1[layer], a.QkvBytes[layer],
                    a.QNorm[layer],
                    a.KNorm[layer],
                    a.O[layer], a.OType[layer], a.ONe0[layer], a.ONe1[layer], a.OBytes[layer],
                    (IntPtr)GetFloatPtr(_weights[postAttnKey]),
                    (IntPtr)GetFloatPtr(_weights[$"{prefix}.ffn_norm.weight"]),
                    a.Gu[layer], a.GuType[layer], a.GuNe0[layer], a.GuNe1[layer], a.GuBytes[layer],
                    a.Down[layer], a.DownType[layer], a.DownNe0[layer], a.DownNe1[layer], a.DownBytes[layer],
                    (IntPtr)GetFloatPtr(_weights[postFfnKey]),
                    a.KCache[layer], a.VCache[layer],
                    Config.NumHeads, a.KvHeads[layer], a.HeadDim[layer],
                    a.CacheSize[layer], startPos,
                    a.IsLocal[layer], _slidingWindow,
                    ropeBase, ropeDims,
                    freqFactorsPtr, freqFactorsLen,
                    a.LayerScalar[layer], Config.Eps,
                    swaPrevKPtr, swaPrevVPtr, prevWindowLen,
                    pleInputPtr, _pleDim,
                    pleGateW, pleGateType, pleGateNe0, pleGateNe1, pleGateBytes,
                    pleProjW, pleProjType, pleProjNe0, pleProjNe1, pleProjBytes,
                    plePostNormW,
                    freshKOutPtr, freshVOutPtr,
                    isShared ? 1 : 0,
                    donorKPtr, donorVPtr, donorKvLen,
                    _kvCacheDtype.GgmlType());

                if (freshKBuffer != null)
                {
                    // Hand the kernel-published fresh K/V to downstream shared
                    // layers via _prefillSWAKV. The dictionary owns the tensors
                    // and disposes them at end of chunk.
                    _prefillSWAKV[layer] = (freshKBuffer, freshVBuffer);
                }
                if (ownsSharedDonor)
                {
                    sharedDonorK?.Dispose();
                    sharedDonorV?.Dispose();
                }
                return true;
            }
            catch
            {
                freshKBuffer?.Dispose();
                freshVBuffer?.Dispose();
                if (ownsSharedDonor)
                {
                    sharedDonorK?.Dispose();
                    sharedDonorV?.Dispose();
                }
                return false;
            }
        }

        private void EnsureKvCacheHostSynchronized()
        {
            if (!_kvCacheHostDirty || !IsGgmlBackend || _kvCacheK == null)
                return;

            var seen = new HashSet<Storage>();
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_kvDonorMap.ContainsKey(l))
                    continue;

                if (_kvCacheK[l] != null && seen.Add(_kvCacheK[l].Storage))
                    SyncTensorHostCache(_kvCacheK[l]);
                if (_kvCacheV[l] != null && seen.Add(_kvCacheV[l].Storage))
                    SyncTensorHostCache(_kvCacheV[l]);
            }

            _kvCacheHostDirty = false;
        }

        #endregion

        #region PLE (Per-Layer Embedding)

        private unsafe Tensor ComputePLE(int[] tokens, Tensor hiddenState, int seqLen)
        {
            int totalPleDim = _pleDim * Config.NumLayers;

            Tensor pleTokenEmb = null;
            if (_quantWeights.TryGetValue("per_layer_token_embd.weight", out var pleQw))
            {
                pleTokenEmb = new Tensor(_allocator, DType.Float32, seqLen, totalPleDim);
                using var pleIdx = CreateIntTensor(tokens, seqLen);
                if (IsGgmlBackend)
                {
                    bool canUseGgmlLookup = CanUseGgmlQuantizedGetRows(pleQw.GgmlType);
                    if ((!canUseGgmlLookup || seqLen == 1) && pleQw.HasHostData)
                    {
                        PopulateQuantizedRows(pleTokenEmb, pleQw, tokens);
                    }
                    else
                    {
                        if (!canUseGgmlLookup)
                            throw new InvalidOperationException($"CUDA get_rows does not support GGML tensor type {(GgmlTensorType)pleQw.GgmlType}, and no host copy is available for CPU fallback.");

                        GgmlBasicOps.GetRowsQuant(pleTokenEmb, pleQw.CacheKey, pleQw.GgmlType, pleQw.Ne0, pleQw.Ne1, pleQw.RawBytes, pleIdx);
                    }
                }
                else
                {
                    if (_backend == BackendType.Cuda &&
                        CudaQuantizedOps.TryGetRowsQuantizedToFloat32(
                            pleTokenEmb,
                            pleQw.EnsureDeviceCacheKey(),
                            pleQw.Data,
                            pleQw.GgmlType,
                            pleQw.Ne0,
                            pleQw.Ne1,
                            pleQw.RawBytes,
                            pleIdx))
                    {
                        // PLE rows are now resident on the CUDA device; no host dequant needed.
                    }
                    else if (_backend == BackendType.Mlx &&
                        MlxQuantizedOps.TryGetRowsQuantizedToFloat32(
                            pleTokenEmb,
                            pleQw.EnsureDeviceCacheKey(),
                            pleQw.Data,
                            pleQw.GgmlType,
                            pleQw.Ne0,
                            pleQw.Ne1,
                            pleQw.RawBytes,
                            pleIdx))
                    {
                        // PLE rows are now resident on the MLX device; no host dequant needed.
                    }
                    else
                    {
                        if (!pleQw.HasHostData)
                            throw new InvalidOperationException("Native PLE quantized row lookup failed and host quantized data has been released.");

                        float* pleDst = GetFloatPtr(pleTokenEmb);
                        byte* pleBase = (byte*)pleQw.Data.ToPointer();
                        long rowBytes = NativeDequant.RowSize(pleQw.GgmlType, pleQw.Ne0);
                        for (int i = 0; i < seqLen; i++)
                        {
                            int token = tokens[i];
                            byte* rowPtr = pleBase + (long)token * rowBytes;
                            ManagedQuantizedOps.DequantizeRowToFloat32(pleQw.GgmlType, (IntPtr)rowPtr, pleDst + (long)i * totalPleDim, totalPleDim);
                        }
                    }
                }

                float pleScale = MathF.Sqrt(_pleDim);
                Ops.Mul(pleTokenEmb, pleTokenEmb, pleScale);
            }
            else if (_weights.TryGetValue("per_layer_token_embd.weight", out var embWeight))
            {
                if (embWeight.IsContiguous())
                {
                    pleTokenEmb = new Tensor(_allocator, DType.Float32, seqLen, totalPleDim);
                    float* embPtr = GetFloatPtr(embWeight);
                    float* dstPtr = GetFloatPtr(pleTokenEmb);
                    long rowBytes = totalPleDim * sizeof(float);
                    for (int i = 0; i < seqLen; i++)
                        Buffer.MemoryCopy(embPtr + (long)tokens[i] * totalPleDim,
                            dstPtr + (long)i * totalPleDim, rowBytes, rowBytes);
                }
                else
                {
                    using var indices = CreateIntTensor(tokens, seqLen);
                    pleTokenEmb = Ops.IndexSelect(null, embWeight, indices);
                }

                float pleScale = MathF.Sqrt(_pleDim);
                Ops.Mul(pleTokenEmb, pleTokenEmb, pleScale);
            }

            Tensor pleProj = LinearForward(hiddenState, "per_layer_model_proj.weight");
            if (pleProj != null)
            {
                float projScale = 1f / MathF.Sqrt(Config.HiddenSize);
                Ops.Mul(pleProj, pleProj, projScale);

                int totalRows = seqLen * Config.NumLayers;
                using var reshaped = pleProj.View(totalRows, _pleDim);
                var normWeight = _weights["per_layer_proj_norm.weight"];
                Ops.RMSNorm(reshaped, reshaped, normWeight, null, Config.Eps);
            }

            Tensor combined;
            if (pleTokenEmb != null && pleProj != null)
            {
                Ops.Add(pleProj, pleProj, pleTokenEmb);
                float combineScale = 1f / MathF.Sqrt(2f);
                Ops.Mul(pleProj, pleProj, combineScale);
                pleTokenEmb.Dispose();
                combined = pleProj;
            }
            else if (pleProj != null)
            {
                combined = pleProj;
            }
            else if (pleTokenEmb != null)
            {
                combined = pleTokenEmb;
            }
            else
            {
                return null;
            }

            return combined;
        }

        private Tensor ExtractPerLayerSlice(Tensor perLayerInputs, int layer, int seqLen)
        {
            int offset = layer * _pleDim;
            if (seqLen > 1)
                return SliceColumnsContiguous(perLayerInputs, offset, _pleDim);

            var slice = perLayerInputs.Narrow(1, offset, _pleDim);
            return slice;
        }

        #endregion

        private bool HasMoE(int layer)
        {
            if (_numExperts == 0) return false;
            string routerKey = $"blk.{layer}.ffn_gate_inp.weight";
            if (!_weights.ContainsKey(routerKey) && !_quantWeights.ContainsKey(routerKey))
                return false;
            // Check for expert weights (could be original 3D tensor or split per-expert)
            string downKey3D = $"blk.{layer}.ffn_down_exps.weight";
            string downKey0 = $"blk.{layer}.ffn_down_exps.0.weight";
            return _weights.ContainsKey(downKey3D) || _quantWeights.ContainsKey(downKey3D) ||
                   _weights.ContainsKey(downKey0) || _quantWeights.ContainsKey(downKey0);
        }

        private Tensor TransformerBlock(Tensor hidden, int layer, int seqLen, int startPos,
            bool isShared, Tensor perLayerInput, HashSet<int> exceptPositions = null)
        {
            // The C# managed prefill / decode path reads and writes the cache as a
            // flat F32 (or F16) buffer. Block-quantized layouts (Q8_0) cannot be
            // walked with raw pointer arithmetic, so we surface a clear error
            // here rather than letting downstream pointer math silently corrupt
            // the cache. Users should pick --kv-cache-dtype f16 for multimodal
            // prompts or any setup that disables the native fused kernels.
            if (_kvCacheDtype.IsBlockQuantized())
                throw new InvalidOperationException(
                    $"Q8_0 KV cache requires the fused native attention kernels. " +
                    $"This call path (multimodal injection / fused-prefill bailout / non-fused decode) " +
                    $"falls back to the C# managed attention helpers which only support F32/F16. " +
                    $"Use --kv-cache-dtype f16 for this configuration.");

            string prefix = $"blk.{layer}";

            bool prof = s_DecodeProfileEnabled && seqLen == 1;
            long pAttnPre = prof ? Stopwatch.GetTimestamp() : 0;

            using var attnNormed = RMSNormOp(hidden, $"{prefix}.attn_norm.weight");

            var attnOut = Attention(attnNormed, layer, prefix, seqLen, startPos, isShared, exceptPositions);

            // When the test-only TS_GEMMA4_DIAG mode is on, force a CPU read
            // here. Metal queues ops async; without flushing, parallel
            // reductions in RMSNorm/softmax can give bit-different results
            // run-to-run, breaking the batched-vs-legacy comparison. The
            // returned byte is discarded; only the implicit download barrier
            // matters.
            bool _diagSync = seqLen > 1 &&
                System.Environment.GetEnvironmentVariable("TS_GEMMA4_DIAG") == "1";
            if (_diagSync) _ = attnOut.GetElementsAsFloat(1);
            if (prof) ProfMark(ref _profAttnSdpaTicks, pAttnPre, attnOut);

            // Fold post-attention RMSNorm + residual add into one MLX Metal
            // kernel. This mirrors the GGML fused graph shape and avoids one
            // intermediate tensor / graph node chain per layer.
            long pAttnOut = prof ? Stopwatch.GetTimestamp() : 0;
            if (TryRmsNormAddInPlaceMlx(hidden, attnOut, $"{prefix}.post_attention_norm.weight"))
            {
                attnOut.Dispose();
                attnOut = hidden;
            }
            else
            {
                Ops.RMSNorm(attnOut, attnOut, _weights[$"{prefix}.post_attention_norm.weight"], null, Config.Eps);
                if (_diagSync) _ = attnOut.GetElementsAsFloat(1);
                Ops.Add(attnOut, attnOut, hidden);
                hidden.Dispose();
            }
            if (_diagSync) _ = attnOut.GetElementsAsFloat(1);
            if (prof) ProfMark(ref _profAttnOutTicks, pAttnOut, attnOut);

            Tensor result;

            if (HasMoE(layer))
            {
                var mlpOut = FFNGeluWithOptionalNorm(attnOut, $"{prefix}.ffn_norm.weight",
                    $"{prefix}.ffn_gate_up.weight", $"{prefix}.ffn_down.weight", seqLen);

                string postMlpNorm1Key = $"{prefix}.post_ffw_norm_1.weight";
                if (!_weights.ContainsKey(postMlpNorm1Key))
                    postMlpNorm1Key = $"{prefix}.ffn_post_norm_1.weight";
                Ops.RMSNorm(mlpOut, mlpOut, _weights[postMlpNorm1Key], null, Config.Eps);

                string postMoeNormKey = $"{prefix}.post_ffw_norm_2.weight";
                if (!_weights.ContainsKey(postMoeNormKey))
                    postMoeNormKey = $"{prefix}.ffn_post_norm_2.weight";

                // Try the residual-fused MoE GEGLU kernel: one dispatch performs
                // moe_ffn(...) → rms_norm(post_norm_2) → add into mlpOut. This
                // collapses three device dispatches (MoEForward + RMSNorm +
                // Add) into one, mirroring Qwen 3.5's MoEExpertsSwiGLUResidual
                // pattern. Falls back to the legacy split path when the
                // stacked expert weights aren't built (e.g. F32-only model)
                // or when the kernel rejects the layout.
                if (!TryMoEForwardResidual(attnOut, mlpOut, layer, prefix, seqLen, postMoeNormKey))
                {
                    using var moeOut = MoEForward(attnOut, layer, prefix, seqLen);
                    if (!TryRmsNormAddInPlaceMlx(mlpOut, moeOut, postMoeNormKey))
                    {
                        using var postMoeNormed = RMSNormOp(moeOut, postMoeNormKey);
                        Ops.Add(mlpOut, mlpOut, postMoeNormed);
                    }
                }

                string postFfnNormKey = $"{prefix}.post_ffw_norm.weight";
                if (!_weights.ContainsKey(postFfnNormKey))
                    postFfnNormKey = $"{prefix}.ffn_post_norm.weight";

                if (!TryRmsNormAddInPlaceMlx(attnOut, mlpOut, postFfnNormKey))
                {
                    Ops.RMSNorm(mlpOut, mlpOut, _weights[postFfnNormKey], null, Config.Eps);
                    Ops.Add(attnOut, attnOut, mlpOut);
                }
                mlpOut.Dispose();
                result = attnOut;
            }
            else
            {
                string postFfnNormKey = $"{prefix}.post_ffw_norm.weight";
                if (!_weights.ContainsKey(postFfnNormKey))
                    postFfnNormKey = $"{prefix}.ffn_post_norm.weight";

                long pFfn = prof ? Stopwatch.GetTimestamp() : 0;

                // Single-closure FFN: pre-norm + gate_up matmul + GeluMulSplit
                // + down matmul + post-norm + residual add all in one
                // mlx_closure_apply via mlx_compile. Implemented to test the
                // hypothesis that bundling 4 separate MLX ops into one
                // compiled closure would skip per-op graph-build overhead
                // and improve decode throughput. **Result: neutral-to-slightly
                // negative.** On Gemma 4 E4B Q8_0, decode best went from
                // 52.76 ms/tok (per-op path) to 53.33 ms/tok (fused
                // closure) — mlx_compile fuses adjacent element-wise ops
                // but can't fuse matmuls, so the dominant per-op cost
                // (graph-build for each quantized_matmul kernel) is paid
                // either way; the closure's own apply-time bookkeeping
                // ate the small theoretical saving.
                //
                // Kept committed (gated off by default) for future
                // benchmarking on larger models or non-Q8_0 quant paths
                // where the trade may be different. Set TS_MLX_FUSED_FFN=1
                // to engage.
                bool fusedFFNDone = false;
                if (seqLen == 1
                    && _backend == BackendType.Mlx
                    && string.Equals(Environment.GetEnvironmentVariable("TS_MLX_FUSED_FFN"), "1", StringComparison.Ordinal)
                    && TryFusedDenseFFNDecode(attnOut, prefix, postFfnNormKey))
                {
                    // attnOut now holds (residual + post_norm(ffn_out)). It
                    // is the layer's running residual stream and matches the
                    // shape of the per-op result below.
                    fusedFFNDone = true;
                }

                if (!fusedFFNDone)
                {
                    var ffnOut = FFNGeluWithOptionalNorm(attnOut, $"{prefix}.ffn_norm.weight",
                        $"{prefix}.ffn_gate_up.weight", $"{prefix}.ffn_down.weight", seqLen);
                    if (_diagSync) _ = ffnOut.GetElementsAsFloat(1);

                    if (!TryRmsNormAddInPlaceMlx(attnOut, ffnOut, postFfnNormKey))
                    {
                        Ops.RMSNorm(ffnOut, ffnOut, _weights[postFfnNormKey], null, Config.Eps);
                        Ops.Add(attnOut, attnOut, ffnOut);
                    }
                    ffnOut.Dispose();
                }
                result = attnOut;
                if (_diagSync) _ = result.GetElementsAsFloat(1);
                if (prof) ProfMark(ref _profFfnTicks, pFfn, result);
            }

            long pPleInject = prof ? Stopwatch.GetTimestamp() : 0;

            // PLE injection
            if (perLayerInput != null &&
                (_weights.ContainsKey($"{prefix}.inp_gate.weight") || _quantWeights.ContainsKey($"{prefix}.inp_gate.weight")))
            {
                Tensor gate = null;

                // Phase 6h: fused Q8 matmul + GeluMul kernel. Saves one MLX
                // op per layer × 42 layers / token on Gemma 4 E4B Q8_0.
                if (seqLen == 1
                    && _backend == BackendType.Mlx
                    && _quantWeights.TryGetValue($"{prefix}.inp_gate.weight", out var inpGateQw)
                    && perLayerInput.ElementType == DType.Float32
                    && perLayerInput.IsContiguous()
                    && perLayerInput.ElementCount() == inpGateQw.Ne1)
                {
                    var fusedGate = new Tensor(_allocator, DType.Float32, 1, (int)inpGateQw.Ne1);
                    if (MlxQuantizedOps.TryFusedQ8MatmulGeluMul(
                            fusedGate, result, perLayerInput,
                            inpGateQw.EnsureDeviceCacheKey(), inpGateQw.Data,
                            inpGateQw.GgmlType, inpGateQw.Ne0, inpGateQw.Ne1, inpGateQw.RawBytes))
                    {
                        gate = fusedGate;
                    }
                    else
                    {
                        fusedGate.Dispose();
                    }
                }

                if (gate == null)
                {
                    gate = LinearForward(result, $"{prefix}.inp_gate.weight");
                    if (gate != null)
                    {
                        if (_diagSync) _ = gate.GetElementsAsFloat(1);
                        Ops.GELUMul(gate, gate, perLayerInput);
                    }
                }

                if (gate != null)
                {
                    if (_diagSync) _ = gate.GetElementsAsFloat(1);
                    using var pleProj = LinearForward(gate, $"{prefix}.proj.weight");
                    gate.Dispose();
                    if (pleProj != null)
                    {
                        string postPleNormKey = $"{prefix}.post_norm.weight";
                        if (!TryRmsNormAddInPlaceMlx(result, pleProj, postPleNormKey))
                        {
                            using var pleNormed = RMSNormOp(pleProj, postPleNormKey);
                            Ops.Add(result, result, pleNormed);
                        }
                    }
                }
            }
            if (prof) ProfMark(ref _profPleInjectTicks, pPleInject, result);

            float scalar = _layerScalars[layer];
            if (scalar != 1f)
                Ops.Mul(result, result, scalar);

            return result;
        }

        #region MoE

        private unsafe Tensor MoEForward(Tensor hiddenState, int layer, string prefix, int seqLen)
        {
            var (routingWeights, selectedExperts) = MoERoute(hiddenState, prefix, seqLen);

            string moeNormKey = $"{prefix}.pre_ffw_norm_2.weight";
            if (!_weights.ContainsKey(moeNormKey))
                moeNormKey = $"{prefix}.ffn_pre_norm_2.weight";
            using var moeInput = RMSNormOp(hiddenState, moeNormKey);

            int hiddenDim = (int)moeInput.Sizes[1];
            var output = new Tensor(_allocator, DType.Float32, seqLen, hiddenDim);

            if (_backend == BackendType.Mlx &&
                TryMoEFusedGEGLUMlx(moeInput, output, selectedExperts, routingWeights,
                                    layer, prefix, seqLen, hiddenDim))
            {
                return output;
            }

            // Expert-batched fused path: one ggml_mul_mat_id-backed dispatch
            // replaces the per-expert batched matmul loop below. This is the
            // dominant MoE decode bottleneck (sequential per-token / per-
            // selected-expert FFNs). Gated by IsGgmlBackend + availability of
            // stacked expert weights (built at load time for quantized MoE
            // tensors; F32-only models fall through to the legacy path).
            // _layerStackedUp is null when the GGUF ships a pre-fused
            // ffn_gate_up_exps.weight; in that case the kernel handles the
            // split internally.
            if (IsGgmlBackend
                && _layerStackedGate != null
                && _layerStackedGate[layer] != null
                && _layerStackedDown[layer] != null)
            {
                if (TryMoEFusedGEGLU(moeInput, output, selectedExperts, routingWeights,
                                     layer, seqLen, hiddenDim))
                {
                    return output;
                }
            }

            // Legacy batched-by-expert fallback (still better than per-token
            // per-expert matmuls: at most numExperts batched matmuls per
            // MoE layer).
            Ops.Fill(output, 0f);

            float* inputPtr = GetFloatPtr(moeInput);
            float* outputPtr = GetFloatPtr(output);

            // Group tokens by expert for batched processing.
            // Instead of seqLen*numExpertsUsed individual single-row matmuls,
            // we run at most numExperts batched matmuls with much better GEMM efficiency.
            var expertBatches = new List<(int tokenIdx, float weight)>[_numExperts];
            for (int i = 0; i < _numExperts; i++)
                expertBatches[i] = new List<(int, float)>();

            for (int s = 0; s < seqLen; s++)
                for (int e = 0; e < _numExpertsUsed; e++)
                {
                    int expertIdx = selectedExperts[s * _numExpertsUsed + e];
                    float weight = routingWeights[s * _numExpertsUsed + e];
                    expertBatches[expertIdx].Add((s, weight));
                }

            // Look up scale key once per layer
            string scaleKey = $"{prefix}.ffn_down_exps.scale";
            if (!_weights.ContainsKey(scaleKey))
                scaleKey = $"{prefix}.ffn_gate_inp.per_expert_scale";
            _weights.TryGetValue(scaleKey, out var perExpertScale);

            for (int expertIdx = 0; expertIdx < _numExperts; expertIdx++)
            {
                var batch = expertBatches[expertIdx];
                if (batch.Count == 0) continue;

                int batchSize = batch.Count;

                // Gather input rows assigned to this expert
                var batchInput = new Tensor(_allocator, DType.Float32, batchSize, hiddenDim);
                float* batchPtr = GetFloatPtr(batchInput);
                int rowBytes = hiddenDim * sizeof(float);
                for (int b = 0; b < batchSize; b++)
                    Buffer.MemoryCopy(inputPtr + (long)batch[b].tokenIdx * hiddenDim,
                        batchPtr + (long)b * hiddenDim, rowBytes, rowBytes);

                // Run batched expert FFN
                string fusedKey = $"{prefix}.ffn_gate_up_exps.{expertIdx}.weight";
                string downKey = $"{prefix}.ffn_down_exps.{expertIdx}.weight";
                Tensor expertOut;

                if (_weights.ContainsKey(fusedKey) || _quantWeights.ContainsKey(fusedKey))
                {
                    expertOut = FFNGelu(batchInput, fusedKey, downKey, batchSize);
                }
                else
                {
                    string gateKey = $"{prefix}.ffn_gate_exps.{expertIdx}.weight";
                    string upKey = $"{prefix}.ffn_up_exps.{expertIdx}.weight";
                    if (!(_weights.ContainsKey(gateKey) || _quantWeights.ContainsKey(gateKey)) ||
                        !(_weights.ContainsKey(downKey) || _quantWeights.ContainsKey(downKey)))
                    {
                        batchInput.Dispose();
                        continue;
                    }
                    using var gateOut = LinearForward(batchInput, gateKey);
                    using var upOut = LinearForward(batchInput, upKey);
                    Ops.GELUMul(gateOut, gateOut, upOut);
                    expertOut = LinearForward(gateOut, downKey);
                }
                batchInput.Dispose();

                // Apply per-expert scale
                if (perExpertScale != null)
                {
                    float s = perExpertScale.GetElementAsFloat(expertIdx);
                    if (s != 1f) Ops.Mul(expertOut, expertOut, s);
                }

                // Scatter-accumulate back with routing weights
                float* expPtr = GetFloatPtr(expertOut);
                for (int b = 0; b < batchSize; b++)
                {
                    int tokenIdx = batch[b].tokenIdx;
                    float w = batch[b].weight;
                    float* src = expPtr + (long)b * hiddenDim;
                    float* dst = outputPtr + (long)tokenIdx * hiddenDim;
                    for (int d = 0; d < hiddenDim; d++)
                        dst[d] += w * src[d];
                }
                expertOut.Dispose();
            }

            return output;
        }

        /// <summary>
        /// Fused MoE FFN via <see cref="GgmlBasicOps.MoEFFNPrefill"/> with
        /// GEGLU activation. Collapses the per-active-expert loop into a
        /// single graph dispatch per MoE layer (3 <c>ggml_mul_mat_id</c> ops
        /// plus the GEGLU fused-op + expert aggregation) regardless of token
        /// count, which is especially valuable on the decode path where
        /// <paramref name="seqLen"/> = 1 and the legacy batched path
        /// degenerates into <see cref="_numExpertsUsed"/> single-row matmuls.
        /// </summary>
        /// <remarks>
        /// Per-expert post-down-projection scales (Gemma 4's
        /// <c>ffn_down_exps.scale</c> / <c>ffn_gate_inp.per_expert_scale</c>)
        /// are folded into <paramref name="routingWeights"/> before the
        /// dispatch so the kernel itself stays activation-agnostic.
        /// </remarks>
        private unsafe bool TryMoEFusedGEGLU(
            Tensor moeInput,
            Tensor output,
            int[] selectedExperts,
            float[] routingWeights,
            int layer,
            int seqLen,
            int hiddenDim)
        {
            var gateW = _layerStackedGate[layer];
            var upW = _layerStackedUp[layer];
            var downW = _layerStackedDown[layer];
            // When upW is null, gateW is actually the pre-fused gate_up_exps
            // weight with ne1 = 2*n_ff. The native kernel detects this via
            // upData == IntPtr.Zero and splits internally.
            bool fusedGateUp = upW == null;
            int nFf = fusedGateUp ? (int)(gateW.PerExpertNe1 / 2) : (int)gateW.PerExpertNe1;
            int nUsed = _numExpertsUsed;

            // Fold the per-expert scalar into routingWeights so a single
            // multiply inside the kernel covers both contributions. Source
            // arrays belong to the caller so we copy before mutating.
            float[] kernelWeights = routingWeights;
            float[] perExpertScale = _layerPerExpertScale?[layer];
            if (perExpertScale != null)
            {
                int totalRoutes = seqLen * nUsed;
                kernelWeights = new float[totalRoutes];
                for (int s = 0; s < seqLen; s++)
                {
                    for (int k = 0; k < nUsed; k++)
                    {
                        int idx = s * nUsed + k;
                        int expertIdx = selectedExperts[idx];
                        kernelWeights[idx] = routingWeights[idx] * perExpertScale[expertIdx];
                    }
                }
            }

            // moeInput is the output of RMSNormOp(hiddenState, moeNormKey) and
            // is guaranteed Float32 + contiguous. If a future refactor ever
            // changes that invariant, the NotSupportedException catch below
            // falls the call back to the legacy batched path.
            // For the pre-fused gate_up layout pass IntPtr.Zero + zeros for the
            // up weight metadata; the native kernel uses that as the signal
            // to consume gateW as a 2*n_ff block and skip the second matmul.
            IntPtr upData = fusedGateUp ? IntPtr.Zero : upW.Data;
            int upType = fusedGateUp ? 0 : upW.GgmlType;
            long upNe0 = fusedGateUp ? 0L : upW.PerExpertNe0;
            long upNe1 = fusedGateUp ? 0L : upW.PerExpertNe1;
            long upBytes = fusedGateUp ? 0L : upW.TotalRawBytes;

            try
            {
                GgmlBasicOps.MoEFFNPrefill(
                    moeInput, output,
                    seqLen, hiddenDim, nFf, _numExperts, nUsed,
                    selectedExperts, kernelWeights,
                    gateW.Data, gateW.GgmlType, gateW.PerExpertNe0, gateW.PerExpertNe1, gateW.TotalRawBytes,
                    upData,     upType,        upNe0,              upNe1,              upBytes,
                    downW.Data, downW.GgmlType, downW.PerExpertNe0, downW.PerExpertNe1, downW.TotalRawBytes,
                    gateBias: null, upBias: null, downBias: null,
                    activation: GgmlBasicOps.MoEActivation.GEGLUSplit);
                InvalidateTensorDeviceCache(output);
                return true;
            }
            catch (NotSupportedException)
            {
                return false;
            }
        }

        private bool TryMoEFusedGEGLUMlx(
            Tensor moeInput,
            Tensor output,
            int[] selectedExperts,
            float[] routingWeights,
            int layer,
            string prefix,
            int seqLen,
            int hiddenDim)
        {
            if (_backend != BackendType.Mlx
                || selectedExperts == null
                || routingWeights == null
                || _numExperts <= 0
                || _numExpertsUsed <= 0
                || selectedExperts.Length < seqLen * _numExpertsUsed
                || routingWeights.Length < seqLen * _numExpertsUsed)
            {
                return false;
            }

            int totalRoutes = checked(seqLen * _numExpertsUsed);
            int[] expertCounts = new int[_numExperts];
            for (int i = 0; i < totalRoutes; i++)
            {
                int expert = selectedExperts[i];
                if ((uint)expert >= (uint)_numExperts)
                    return false;
                expertCounts[expert]++;
            }

            int[] expertOffsets = new int[_numExperts + 1];
            for (int e = 0; e < _numExperts; e++)
                expertOffsets[e + 1] = expertOffsets[e] + expertCounts[e];

            int[] cursors = new int[_numExperts];
            Array.Copy(expertOffsets, cursors, _numExperts);
            int[] routedTokenRows = new int[totalRoutes];
            float[] routedWeights = new float[totalRoutes];
            float[] perExpertScale = GetMoEPerExpertScale(layer, prefix);

            for (int s = 0; s < seqLen; s++)
            {
                int routeOffset = s * _numExpertsUsed;
                for (int k = 0; k < _numExpertsUsed; k++)
                {
                    int src = routeOffset + k;
                    int expert = selectedExperts[src];
                    int dst = cursors[expert]++;
                    float weight = routingWeights[src];
                    if (perExpertScale != null)
                        weight *= perExpertScale[expert];
                    routedTokenRows[dst] = s;
                    routedWeights[dst] = weight;
                }
            }

            try
            {
                Ops.Fill(output, 0f);

                for (int expertIdx = 0; expertIdx < _numExperts; expertIdx++)
                {
                    int start = expertOffsets[expertIdx];
                    int batchSize = expertOffsets[expertIdx + 1] - start;
                    if (batchSize == 0)
                        continue;

                    int[] rowBatch = new int[batchSize];
                    float[] weightBatch = new float[batchSize];
                    Array.Copy(routedTokenRows, start, rowBatch, 0, batchSize);
                    Array.Copy(routedWeights, start, weightBatch, 0, batchSize);

                    using var rowIndices = CreateIntTensor(rowBatch, batchSize);
                    using var routeWeights = CreateFloatTensor(weightBatch, batchSize);
                    using var batchInput = new Tensor(_allocator, DType.Float32, batchSize, hiddenDim);
                    if (!MlxFusedOps.TryGatherRows(batchInput, moeInput, rowIndices))
                        return false;

                    Tensor expertOut = null;
                    try
                    {
                        string fusedKey = $"{prefix}.ffn_gate_up_exps.{expertIdx}.weight";
                        string downKey = $"{prefix}.ffn_down_exps.{expertIdx}.weight";
                        expertOut = FFNGelu(batchInput, fusedKey, downKey, batchSize);
                        if (expertOut == null)
                            return false;

                        if (!MlxFusedOps.TryScatterAddWeightedRows(output, expertOut, rowIndices, routeWeights))
                            return false;
                    }
                    finally
                    {
                        expertOut?.Dispose();
                    }
                }

                return true;
            }
            catch (InvalidOperationException)
            {
                return false;
            }
            catch (NotSupportedException)
            {
                return false;
            }
        }

        private float[] GetMoEPerExpertScale(int layer, string prefix)
        {
            if (_layerPerExpertScale != null
                && layer >= 0
                && layer < _layerPerExpertScale.Length
                && _layerPerExpertScale[layer] != null)
            {
                return _layerPerExpertScale[layer];
            }

            string scaleKey = $"{prefix}.ffn_down_exps.scale";
            if (!_weights.ContainsKey(scaleKey))
                scaleKey = $"{prefix}.ffn_gate_inp.per_expert_scale";
            if (!_weights.TryGetValue(scaleKey, out var perExpertScale))
                return null;

            var scales = new float[_numExperts];
            for (int expertIdx = 0; expertIdx < _numExperts; expertIdx++)
                scales[expertIdx] = perExpertScale.GetElementAsFloat(expertIdx);
            return scales;
        }

        /// <summary>
        /// Single-dispatch MoE GEGLU + post_ffw_norm_2 + add-to-residual for
        /// Gemma 4 MoE layers. Folds three device dispatches (MoEForward +
        /// RMSNorm + Add) into one ggml graph submission via the native
        /// <see cref="GgmlBasicOps.MoEFFNGEGLUResidualGemma4"/> kernel.
        ///
        /// On success <paramref name="residual"/> already contains
        /// <c>residual + rms_norm(moe_ffn(hiddenState), eps) * post_norm_w</c>
        /// and the caller proceeds directly to the post-FFN norm. On failure
        /// the caller falls back to the legacy 3-dispatch sequence.
        /// </summary>
        /// <param name="hiddenState">Attention residual (input to the MoE block).</param>
        /// <param name="residual">Dense FFN output (post_ffw_norm_1) — written in place.</param>
        /// <param name="layer">Layer index for weight lookup.</param>
        /// <param name="prefix">"blk.{layer}" prefix for tensor lookup.</param>
        /// <param name="seqLen">Number of tokens in the chunk (1 for decode).</param>
        /// <param name="postMoeNormKey">Resolved key for post_ffw_norm_2.weight.</param>
        private unsafe bool TryMoEForwardResidual(
            Tensor hiddenState,
            Tensor residual,
            int layer,
            string prefix,
            int seqLen,
            string postMoeNormKey)
        {
            if (_backend == BackendType.Mlx)
                return TryMoEForwardResidualMlx(hiddenState, residual, layer, prefix, seqLen, postMoeNormKey);

            // Same gating as TryMoEFusedGEGLU: requires the GGML backend and
            // the stacked-experts cache populated by CacheMoEStackedWeights.
            // Falls back to the legacy split path on any miss so quantization
            // / weight-layout edge cases don't degrade correctness.
            if (!IsGgmlBackend
                || _layerStackedGate == null
                || _layerStackedGate[layer] == null
                || _layerStackedDown[layer] == null)
                return false;

            if (!_weights.TryGetValue(postMoeNormKey, out var postNormW))
                return false;

            // Routing must run before the kernel: the C# loop builds the
            // selected_experts / routing_weights arrays from the router
            // logits (CPU softmax + top-K + per-expert-scale fold). The
            // kernel itself is purely the GEGLU FFN + post-norm + add.
            var (routingWeights, selectedExperts) = MoERoute(hiddenState, prefix, seqLen);

            string moeNormKey = $"{prefix}.pre_ffw_norm_2.weight";
            if (!_weights.ContainsKey(moeNormKey))
                moeNormKey = $"{prefix}.ffn_pre_norm_2.weight";
            using var moeInput = RMSNormOp(hiddenState, moeNormKey);

            int hiddenDim = (int)moeInput.Sizes[1];
            var gateW = _layerStackedGate[layer];
            var upW = _layerStackedUp[layer];
            var downW = _layerStackedDown[layer];

            bool fusedGateUp = upW == null;
            int nFf = fusedGateUp ? (int)(gateW.PerExpertNe1 / 2) : (int)gateW.PerExpertNe1;
            int nUsed = _numExpertsUsed;

            // Fold per-expert post-down scale (Gemma 4's
            // ffn_down_exps.scale / ffn_gate_inp.per_expert_scale) into the
            // routing weights so the kernel sees a single scalar per (token,
            // active-expert) slot. Must copy to a fresh array because the
            // caller's routingWeights may be reused.
            float[] kernelWeights = routingWeights;
            float[] perExpertScale = _layerPerExpertScale?[layer];
            if (perExpertScale != null)
            {
                int totalRoutes = seqLen * nUsed;
                kernelWeights = new float[totalRoutes];
                for (int s = 0; s < seqLen; s++)
                {
                    for (int k = 0; k < nUsed; k++)
                    {
                        int idx = s * nUsed + k;
                        int expertIdx = selectedExperts[idx];
                        kernelWeights[idx] = routingWeights[idx] * perExpertScale[expertIdx];
                    }
                }
            }

            // Up-weight metadata is signalled to the kernel via IntPtr.Zero +
            // zeros for the fused-gate_up layout (mirrors MoEFFNPrefill /
            // TryMoEFusedGEGLU).
            IntPtr upData = fusedGateUp ? IntPtr.Zero : upW.Data;
            int upType = fusedGateUp ? 0 : upW.GgmlType;
            long upNe0 = fusedGateUp ? 0L : upW.PerExpertNe0;
            long upNe1 = fusedGateUp ? 0L : upW.PerExpertNe1;
            long upBytes = fusedGateUp ? 0L : upW.TotalRawBytes;

            try
            {
                GgmlBasicOps.MoEFFNGEGLUResidualGemma4(
                    moeInput, residual, postNormW, Config.Eps,
                    seqLen, hiddenDim, nFf, _numExperts, nUsed,
                    selectedExperts, kernelWeights,
                    gateW.Data, gateW.GgmlType, gateW.PerExpertNe0, gateW.PerExpertNe1, gateW.TotalRawBytes,
                    upData,     upType,        upNe0,              upNe1,              upBytes,
                    downW.Data, downW.GgmlType, downW.PerExpertNe0, downW.PerExpertNe1, downW.TotalRawBytes,
                    gateBias: null, upBias: null, downBias: null,
                    activation: GgmlBasicOps.MoEActivation.GEGLUSplit);
                InvalidateTensorDeviceCache(residual);
                return true;
            }
            catch (NotSupportedException)
            {
                return false;
            }
        }

        private bool TryMoEForwardResidualMlx(
            Tensor hiddenState,
            Tensor residual,
            int layer,
            string prefix,
            int seqLen,
            string postMoeNormKey)
        {
            if (!_weights.TryGetValue(postMoeNormKey, out var postNormW))
                return false;

            var (routingWeights, selectedExperts) = MoERoute(hiddenState, prefix, seqLen);

            string moeNormKey = $"{prefix}.pre_ffw_norm_2.weight";
            if (!_weights.ContainsKey(moeNormKey))
                moeNormKey = $"{prefix}.ffn_pre_norm_2.weight";

            using var moeInput = RMSNormOp(hiddenState, moeNormKey);
            int hiddenDim = (int)moeInput.Sizes[1];
            using var moeOut = new Tensor(_allocator, DType.Float32, seqLen, hiddenDim);
            if (!TryMoEFusedGEGLUMlx(moeInput, moeOut, selectedExperts, routingWeights,
                                     layer, prefix, seqLen, hiddenDim))
            {
                return false;
            }

            if (MlxFusedOps.TryRmsNormAddInPlace(residual, moeOut, postNormW, Config.Eps))
                return true;

            Ops.RMSNorm(moeOut, moeOut, postNormW, null, Config.Eps);
            Ops.Add(residual, residual, moeOut);
            return true;
        }

        private unsafe (float[] routingWeights, int[] selectedExperts) MoERoute(
            Tensor input, string prefix, int seqLen)
        {
            int hiddenDim = (int)input.Sizes[1];

            // Unweighted RMSNorm on a copy (input is used elsewhere)
            using var normed = Ops.NewContiguous(input);
            ApplyUnweightedRMSNorm(normed, 1, hiddenDim, seqLen);

            // Scale by 1/sqrt(hidden_size)
            float invSqrtHidden = 1f / MathF.Sqrt(hiddenDim);
            Ops.Mul(normed, normed, invSqrtHidden);

            // Multiply by learned scale parameter (broadcast per-dim scale across tokens)
            string scaleKey = $"{prefix}.ffn_gate_inp.scale";
            if (_weights.TryGetValue(scaleKey, out var scaleTensor))
                Ops.Mul(normed, normed, scaleTensor);

            // Project to expert logits
            using var expertScores = LinearForward(normed, $"{prefix}.ffn_gate_inp.weight");

            float* scoresPtr = GetFloatPtr(expertScores);
            int numExperts = (int)expertScores.Sizes[1];
            int nUsed = _numExpertsUsed;
            int needed = seqLen * nUsed;

            float[] routingWeights = _moeRoutingWeightsScratch;
            int[] selectedExperts = _moeSelectedExpertsScratch;
            if (routingWeights == null || routingWeights.Length < needed)
                routingWeights = _moeRoutingWeightsScratch = new float[needed];
            if (selectedExperts == null || selectedExperts.Length < needed)
                selectedExperts = _moeSelectedExpertsScratch = new int[needed];
            int[] topK = _moeTopKScratch;

            for (int s = 0; s < seqLen; s++)
            {
                float* row = scoresPtr + s * numExperts;
                int rowOff = s * nUsed;

                // Softmax over all experts (in place).
                float maxVal = float.NegativeInfinity;
                for (int i = 0; i < numExperts; i++)
                    if (row[i] > maxVal) maxVal = row[i];
                float sumExp = 0f;
                for (int i = 0; i < numExperts; i++)
                {
                    float ex = MathF.Exp(row[i] - maxVal);
                    row[i] = ex;
                    sumExp += ex;
                }
                float invSum = sumExp > 0f ? 1f / sumExp : 0f;
                for (int i = 0; i < numExperts; i++)
                    row[i] *= invSum;

                // O(n*k) top-k selection on the post-softmax probabilities.
                TensorComputePrimitives.SelectTopKInPlace(row, numExperts, nUsed, topK);

                // Gather selected probabilities + renormalize over selected.
                float selectedSum = 0f;
                for (int k = 0; k < nUsed; k++)
                {
                    int idx = topK[k];
                    float v = row[idx];
                    selectedExperts[rowOff + k] = idx;
                    routingWeights[rowOff + k] = v;
                    selectedSum += v;
                }
                if (selectedSum > 0f)
                {
                    float invSel = 1f / selectedSum;
                    for (int k = 0; k < nUsed; k++)
                        routingWeights[rowOff + k] *= invSel;
                }
            }

            return (routingWeights, selectedExperts);
        }

        private unsafe void FuseQKVWeights()
        {
            int fused = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                bool isShared = _kvDonorMap.ContainsKey(l);
                if (isShared) continue;

                string prefix = $"blk.{l}";
                string qName = $"{prefix}.attn_q.weight";
                string kName = $"{prefix}.attn_k.weight";
                string vName = $"{prefix}.attn_v.weight";
                string qkvName = $"{prefix}.attn_qkv.weight";

                if (_quantWeights.TryGetValue(qName, out var qw) &&
                    _quantWeights.TryGetValue(kName, out var kw))
                {
                    bool hasQuantV = _quantWeights.TryGetValue(vName, out QuantizedWeight vw);

                    if (qw.GgmlType == kw.GgmlType && qw.Ne0 == kw.Ne0 &&
                        (!hasQuantV || (vw.GgmlType == kw.GgmlType && vw.Ne0 == kw.Ne0)))
                    {
                        QuantizedWeight fusedWeight;
                        bool fusedOk = hasQuantV
                            ? TryCreateFusedQuantizedWeight(out fusedWeight, qw, kw, vw)
                            : TryCreateFusedQuantizedWeight(out fusedWeight, qw, kw, kw);
                        if (!fusedOk)
                            continue;

                        _quantWeights[qkvName] = fusedWeight;
                        _quantWeights.Remove(qName); qw.Dispose();
                        _quantWeights.Remove(kName); kw.Dispose();
                        if (hasQuantV) { _quantWeights.Remove(vName); vw.Dispose(); }
                        fused++;
                    }
                }
                else if (_weights.TryGetValue(qName, out var qf) &&
                         _weights.TryGetValue(kName, out var kf))
                {
                    bool hasF32V = _weights.TryGetValue(vName, out Tensor vf);

                    int qDim = (int)qf.Sizes[0], kDim = (int)kf.Sizes[0];
                    int vDim = hasF32V ? (int)vf.Sizes[0] : kDim;
                    int inDim = (int)qf.Sizes[1];
                    var fusedTensor = new Tensor(_allocator, DType.Float32, qDim + kDim + vDim, inDim);
                    using (var s0 = fusedTensor.Narrow(0, 0, qDim)) Ops.Copy(s0, qf);
                    using (var s1 = fusedTensor.Narrow(0, qDim, kDim)) Ops.Copy(s1, kf);
                    if (hasF32V)
                    {
                        using (var s2 = fusedTensor.Narrow(0, qDim + kDim, vDim)) Ops.Copy(s2, vf);
                    }
                    else
                    {
                        using (var s2 = fusedTensor.Narrow(0, qDim + kDim, vDim)) Ops.Copy(s2, kf);
                    }
                    _weights[qkvName] = fusedTensor;
                    _weights.Remove(qName); qf.Dispose();
                    _weights.Remove(kName); kf.Dispose();
                    if (hasF32V) { _weights.Remove(vName); vf.Dispose(); }
                    fused++;
                }
            }
            if (fused > 0)
                Console.WriteLine($"  Fused QKV projections: {fused}");
        }

        private unsafe void FuseExpertGateUpWeights()
        {
            if (_numExperts == 0) return;
            int fused = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                for (int e = 0; e < _numExperts; e++)
                {
                    string gateName = $"blk.{l}.ffn_gate_exps.{e}.weight";
                    string upName = $"blk.{l}.ffn_up_exps.{e}.weight";
                    string fusedName = $"blk.{l}.ffn_gate_up_exps.{e}.weight";

                    if (_quantWeights.TryGetValue(gateName, out var gw) &&
                        _quantWeights.TryGetValue(upName, out var uw) &&
                        gw.GgmlType == uw.GgmlType && gw.Ne0 == uw.Ne0)
                    {
                        if (!TryCreateFusedQuantizedWeight(out QuantizedWeight fusedWeight, gw, uw))
                            continue;

                        _quantWeights[fusedName] = fusedWeight;
                        _quantWeights.Remove(gateName); gw.Dispose();
                        _quantWeights.Remove(upName); uw.Dispose();
                        fused++;
                    }
                    else if (_weights.TryGetValue(gateName, out var gf) &&
                             _weights.TryGetValue(upName, out var uf))
                    {
                        int gateDim = (int)gf.Sizes[0], upDim = (int)uf.Sizes[0];
                        int inDim = (int)gf.Sizes[1];
                        var fusedTensor = new Tensor(_allocator, DType.Float32, gateDim + upDim, inDim);
                        using (var s0 = fusedTensor.Narrow(0, 0, gateDim)) Ops.Copy(s0, gf);
                        using (var s1 = fusedTensor.Narrow(0, gateDim, upDim)) Ops.Copy(s1, uf);
                        _weights[fusedName] = fusedTensor;
                        _weights.Remove(gateName); gf.Dispose();
                        _weights.Remove(upName); uf.Dispose();
                        fused++;
                    }
                }
            }
            if (fused > 0)
                Console.WriteLine($"  Fused expert projections: {fused} Gate+Up");
        }

        /// <summary>
        /// Snapshot per-layer stacked MoE expert weights and per-expert scales
        /// so <see cref="MoEForward"/> can dispatch the fused GEGLU kernel
        /// without re-scanning dictionaries every call. Runs once at model
        /// load time after <see cref="FuseExpertGateUpWeights"/>; the stacked
        /// views survive that fusion because it only mutates the per-expert
        /// entries in <c>_quantWeights</c> / <c>_weights</c>, never the
        /// <c>_stackedExpertWeights</c> dictionary.
        ///
        /// Two GGUF layouts are handled transparently:
        ///   * Pre-fused <c>ffn_gate_up_exps.weight</c> of shape
        ///     <c>[hidden, 2*n_ff, num_experts]</c> (used by the first-party
        ///     Gemma 4 26B-A4B GGUFs). In this case <c>_layerStackedUp</c>
        ///     stays <c>null</c> and the native kernel receives the gate
        ///     pointer with <c>upData = IntPtr.Zero</c>, mirroring
        ///     llama.cpp's <c>gate_up_exps</c> path.
        ///   * Separate <c>ffn_gate_exps.weight</c> + <c>ffn_up_exps.weight</c>
        ///     (used by some third-party converts). Both pointers are
        ///     populated and the kernel issues two <c>ggml_mul_mat_id</c>
        ///     projections.
        /// </summary>
        private void CacheMoEStackedWeights()
        {
            if (_numExperts == 0) return;

            int numLayers = Config.NumLayers;
            _layerStackedGate = new StackedExpertWeights[numLayers];
            _layerStackedUp = new StackedExpertWeights[numLayers];
            _layerStackedDown = new StackedExpertWeights[numLayers];
            _layerPerExpertScale = new float[numLayers][];

            int fusedCapable = 0;
            int fusedGateUpLayers = 0;
            for (int l = 0; l < numLayers; l++)
            {
                string prefix = $"blk.{l}";

                // Prefer the pre-fused gate_up layout when the GGUF provides it:
                // skips a ggml_mul_mat_id and saves one quant dequant path inside
                // the kernel. Up stays null to signal the fused layout.
                if (_stackedExpertWeights.TryGetValue($"{prefix}.ffn_gate_up_exps.weight", out var gateUp))
                {
                    _layerStackedGate[l] = gateUp;
                    _layerStackedUp[l] = null;
                    fusedGateUpLayers++;
                }
                else
                {
                    _stackedExpertWeights.TryGetValue($"{prefix}.ffn_gate_exps.weight", out _layerStackedGate[l]);
                    _stackedExpertWeights.TryGetValue($"{prefix}.ffn_up_exps.weight",   out _layerStackedUp[l]);
                }
                _stackedExpertWeights.TryGetValue($"{prefix}.ffn_down_exps.weight", out _layerStackedDown[l]);

                string scaleKey = $"{prefix}.ffn_down_exps.scale";
                if (!_weights.ContainsKey(scaleKey))
                    scaleKey = $"{prefix}.ffn_gate_inp.per_expert_scale";
                if (_weights.TryGetValue(scaleKey, out var scaleT))
                {
                    var scales = new float[_numExperts];
                    for (int e = 0; e < _numExperts; e++)
                        scales[e] = scaleT.GetElementAsFloat(e);
                    _layerPerExpertScale[l] = scales;
                }

                // Re-evaluate per layer: gate must exist; either up exists, or the
                // gate is the pre-fused gate_up (ne1 = 2 * n_ff).
                bool gateIsFused = _layerStackedGate[l] != null
                    && _layerStackedUp[l] == null
                    && _stackedExpertWeights.ContainsKey($"{prefix}.ffn_gate_up_exps.weight");
                bool ok = _layerStackedGate[l] != null
                          && _layerStackedDown[l] != null
                          && (_layerStackedUp[l] != null || gateIsFused);
                if (ok) fusedCapable++;
            }

            if (fusedCapable > 0)
            {
                string layout = fusedGateUpLayers == fusedCapable
                    ? "fused gate_up"
                    : (fusedGateUpLayers > 0 ? "mixed" : "separate gate/up");
                Console.WriteLine($"  Expert-batched MoE FFN kernel available on {fusedCapable}/{numLayers} layers ({layout})");
            }
        }

        #endregion

        private Tensor FFNGelu(Tensor input, string gateUpWeightName, string downWeightName, int seqLen)
        {
            Tensor gateUp = LinearForward(input, gateUpWeightName);
            if (gateUp == null)
            {
                if (TryResolveSeparateGateUpWeights(gateUpWeightName, out string gateWeightName, out string upWeightName) &&
                    HasLinearWeight(gateWeightName) &&
                    HasLinearWeight(upWeightName))
                {
                    return FFNGeluSeparate(input, gateWeightName, upWeightName, downWeightName);
                }

                throw new InvalidOperationException(
                    $"Missing FFN gate/up projection weight '{gateUpWeightName}' and no separate gate/up fallback was found.");
            }

            return FFNGeluProjected(gateUp, downWeightName, seqLen);
        }

        private Tensor FFNGeluWithOptionalNorm(
            Tensor input,
            string normWeightName,
            string gateUpWeightName,
            string downWeightName,
            int seqLen)
        {
            if (TryFFNGeluWithNormMlx(input, normWeightName, gateUpWeightName, downWeightName, seqLen, out var fused))
                return fused;

            using var normed = RMSNormOp(input, normWeightName);
            return FFNGelu(normed, gateUpWeightName, downWeightName, seqLen);
        }

        private bool TryFFNGeluWithNormMlx(
            Tensor input,
            string normWeightName,
            string gateUpWeightName,
            string downWeightName,
            int seqLen,
            out Tensor result)
        {
            result = null;
            if (_backend != BackendType.Mlx
                || input.DimensionCount != 2
                || !_weights.TryGetValue(normWeightName, out var normW)
                || !_quantWeights.TryGetValue(gateUpWeightName, out var gateUpQw))
            {
                return false;
            }

            var gateUp = new Tensor(_allocator, DType.Float32, input.Sizes[0], gateUpQw.Ne1);
            long t0 = Stopwatch.GetTimestamp();
            if (!MlxQuantizedOps.TryRmsNormAddmmQuantizedToFloat32(
                    gateUp,
                    input,
                    normW,
                    Config.Eps,
                    gateUpQw.EnsureDeviceCacheKey(),
                    gateUpQw.Data,
                    gateUpQw.GgmlType,
                    gateUpQw.Ne0,
                    gateUpQw.Ne1,
                    gateUpQw.RawBytes))
            {
                gateUp.Dispose();
                return false;
            }

            _linearTicks += Stopwatch.GetTimestamp() - t0;
            result = FFNGeluProjected(gateUp, downWeightName, seqLen);
            return true;
        }

        private bool TryRmsNormAddInPlaceMlx(Tensor residual, Tensor input, string normWeightName)
        {
            return _backend == BackendType.Mlx
                && _weights.TryGetValue(normWeightName, out var normW)
                && MlxFusedOps.TryRmsNormAddInPlace(residual, input, normW, Config.Eps);
        }

        private void TryEvaluateMlxLayerBoundary(Tensor hidden, int layer, int seqLen)
        {
            if (_backend != BackendType.Mlx || MlxEvalEveryNLayers <= 0)
                return;
            if (seqLen == 1 && !MlxEvalDecodeLayerBoundaries)
                return;
            if ((layer + 1) % MlxEvalEveryNLayers != 0 && layer + 1 != Config.NumLayers)
                return;

            // Async at intermediate boundaries lets Metal keep issuing the next
            // layer's commands while earlier graphs are still completing.
            // The final layer must sync because the LM head reads hidden on host.
            if (MlxBaselineSyncLayerEval || layer + 1 == Config.NumLayers)
                MlxFusedOps.TryEvaluate(hidden);
            else
                MlxFusedOps.TryAsyncEvaluate(hidden);
        }

        /// <summary>
        /// Consumes the already-projected [tokens, 2*intermediate] gate_up tensor.
        /// </summary>
        private Tensor FFNGeluProjected(Tensor gateUp, string downWeightName, int seqLen)
        {
            int halfDim = (int)(gateUp.Sizes[1] / 2);

            if (_backend == BackendType.Mlx)
            {
                var activated = new Tensor(_allocator, DType.Float32, gateUp.Sizes[0], halfDim);
                if (MlxFusedOps.TryGeluMulSplit(activated, gateUp, halfDim))
                {
                    gateUp.Dispose();
                    Tensor downMlx = null;
                    try
                    {
                        downMlx = LinearForward(activated, downWeightName);
                    }
                    finally
                    {
                        activated.Dispose();
                    }
                    if (downMlx == null)
                        throw new InvalidOperationException($"Missing FFN down projection weight '{downWeightName}'.");
                    return downMlx;
                }
                activated.Dispose();
            }

            if (_backend == BackendType.Cuda && seqLen > 1)
            {
                var activated = new Tensor(_allocator, DType.Float32, gateUp.Sizes[0], halfDim);
                if (CudaFusedOps.TryGELUMulSplit(activated, gateUp, halfDim))
                {
                    gateUp.Dispose();
                    Tensor downFast = LinearForward(activated, downWeightName);
                    activated.Dispose();
                    return downFast;
                }
                activated.Dispose();
            }

            Tensor gate, up;
            if (seqLen == 1)
            {
                gate = gateUp.Narrow(1, 0, halfDim);
                up = gateUp.Narrow(1, halfDim, halfDim);
            }
            else
            {
                // gate needs a contiguous copy for the downstream LinearForward.
                // up is only read by GELUMul, so a non-contiguous narrow view
                // suffices (GELUMul's Apply3 path handles strides).
                using (var gView = gateUp.Narrow(1, 0, halfDim))
                    gate = Ops.NewContiguous(gView);
                up = gateUp.Narrow(1, halfDim, halfDim);
            }

            Ops.GELUMul(gate, gate, up);
            up.Dispose();
            gateUp.Dispose();

            Tensor down = null;
            try
            {
                down = LinearForward(gate, downWeightName);
            }
            finally
            {
                gate.Dispose();
            }
            if (down == null)
                throw new InvalidOperationException($"Missing FFN down projection weight '{downWeightName}'.");
            return down;
        }

        private Tensor FFNGeluSeparate(Tensor input, string gateWeightName, string upWeightName, string downWeightName)
        {
            Tensor gate = LinearForward(input, gateWeightName);
            if (gate == null)
                throw new InvalidOperationException($"Missing FFN gate projection weight '{gateWeightName}'.");

            Tensor up = null;
            try
            {
                up = LinearForward(input, upWeightName);
                if (up == null)
                    throw new InvalidOperationException($"Missing FFN up projection weight '{upWeightName}'.");

                Ops.GELUMul(gate, gate, up);
            }
            finally
            {
                up?.Dispose();
            }

            Tensor down = null;
            try
            {
                down = LinearForward(gate, downWeightName);
            }
            finally
            {
                gate.Dispose();
            }
            if (down == null)
                throw new InvalidOperationException($"Missing FFN down projection weight '{downWeightName}'.");
            return down;
        }

        private bool HasLinearWeight(string weightName)
        {
            return _quantWeights.ContainsKey(weightName) || _weights.ContainsKey(weightName);
        }

        private static bool TryResolveSeparateGateUpWeights(string gateUpWeightName, out string gateWeightName, out string upWeightName)
        {
            gateWeightName = null;
            upWeightName = null;

            const string fusedDense = ".ffn_gate_up.weight";
            int denseIndex = gateUpWeightName.IndexOf(fusedDense, StringComparison.Ordinal);
            if (denseIndex >= 0)
            {
                string prefix = gateUpWeightName.Substring(0, denseIndex);
                gateWeightName = prefix + ".ffn_gate.weight";
                upWeightName = prefix + ".ffn_up.weight";
                return true;
            }

            const string fusedExpert = ".ffn_gate_up_exps.";
            int expertIndex = gateUpWeightName.IndexOf(fusedExpert, StringComparison.Ordinal);
            if (expertIndex >= 0 && gateUpWeightName.EndsWith(".weight", StringComparison.Ordinal))
            {
                string prefix = gateUpWeightName.Substring(0, expertIndex);
                string suffix = gateUpWeightName.Substring(expertIndex + fusedExpert.Length);
                gateWeightName = prefix + ".ffn_gate_exps." + suffix;
                upWeightName = prefix + ".ffn_up_exps." + suffix;
                return true;
            }

            return false;
        }

        #region Attention

        private Tensor Attention(Tensor input, int layer, string prefix, int seqLen, int startPos, bool isShared, HashSet<int> exceptPositions = null)
        {
            long t0 = Stopwatch.GetTimestamp();
            _attnFusedDecodePreprocessApplied = false;
            bool isLocal = IsLocalLayer(layer);
            int hd = HeadDimForLayer(layer);
            int kvHeads = KVHeadsForLayer(layer);

            int qDim = Config.NumHeads * hd;
            int kDim = kvHeads * hd;

            Tensor q, k = null, v = null;
            string qkvName = $"{prefix}.attn_qkv.weight";
            bool useFusedQKV = !isShared && (_quantWeights.ContainsKey(qkvName) || _weights.ContainsKey(qkvName));

            // For global (non-SWA) prefill layers with fused QKV, use a fast path
            // that copies directly from QKV to head-first layout, skipping the
            // intermediate flat copies and the separate ReshapeToHeads step.
            bool _useGlobalFastPath = useFusedQKV && seqLen > 1 && !isLocal && !isShared;
            Tensor _globalQHeads = null, _globalKHeads = null, _globalVHeads = null;

            if (_useGlobalFastPath)
            {
                Tensor qkv = LinearForward(input, qkvName);

                _globalQHeads = SplitQKVToHeadFirst(qkv, 0, Config.NumHeads, seqLen, hd);
                _globalKHeads = SplitQKVToHeadFirst(qkv, qDim, kvHeads, seqLen, hd);
                _globalVHeads = SplitQKVToHeadFirst(qkv, qDim + kDim, kvHeads, seqLen, hd);
                qkv.Dispose();

                // RMSNorm on head-first layout (row-independent, order doesn't matter)
                using (var qR = _globalQHeads.View(Config.NumHeads * seqLen, hd))
                    Ops.RMSNorm(qR, qR, _weights[$"{prefix}.attn_q_norm.weight"], null, Config.Eps);
                using (var kR = _globalKHeads.View(kvHeads * seqLen, hd))
                    Ops.RMSNorm(kR, kR, _weights[$"{prefix}.attn_k_norm.weight"], null, Config.Eps);
                ApplyUnweightedRMSNorm(_globalVHeads, kvHeads, hd, seqLen);

                // NeoX RoPE on head-first layout
                float[] globalFreqs = _ropeFreqsGlobal;
                ApplyNeoXRoPEHeadFirst(_globalQHeads, Config.NumHeads, hd, seqLen, startPos, globalFreqs);
                ApplyNeoXRoPEHeadFirst(_globalKHeads, kvHeads, hd, seqLen, startPos, globalFreqs);

                q = null; k = null; v = null;
            }
            else if (useFusedQKV)
            {
                Tensor qkv = LinearForward(input, qkvName);
                int vDim = (int)qkv.Sizes[1] - qDim - kDim;

                // Fused QKV preprocess kernel (TryFusedDecodeQkvPreprocess
                // → TryGemma4QkvPreprocessDecode). Phase 6f rewrote it with
                // simdgroup reductions but still perf-neutral / slightly
                // negative on Gemma 4 E4B — the kernel runs only
                // (NumHeads + 2*NumKVHeads) = 12 threadgroups, under-
                // saturating the GPU vs the per-op path's ~22 threadgroups
                // across 5 separate launches. Kept gated off as opt-in for
                // models with more heads. Set TS_MLX_FUSED_QKV_PREP=1 to
                // engage.
                bool fusedDone = false;
                if (seqLen == 1
                    && _backend == BackendType.Mlx
                    && vDim == kDim
                    && string.Equals(Environment.GetEnvironmentVariable("TS_MLX_FUSED_QKV_PREP"), "1", StringComparison.Ordinal)
                    && TryFusedDecodeQkvPreprocess(qkv, prefix, isLocal, hd, kvHeads, qDim, startPos, out q, out k, out v))
                {
                    qkv.Dispose();
                    fusedDone = true;
                }
                else
                {
                    if (seqLen == 1)
                    {
                        q = qkv.Narrow(1, 0, qDim);
                        k = qkv.Narrow(1, qDim, kDim);
                        v = qkv.Narrow(1, qDim + kDim, vDim);
                    }
                    else
                    {
                        q = SliceColumnsContiguous(qkv, 0, qDim);
                        k = SliceColumnsContiguous(qkv, qDim, kDim);
                        v = SliceColumnsContiguous(qkv, qDim + kDim, vDim);
                    }
                    qkv.Dispose();

                    if (seqLen == 1)
                    {
                        RMSNormInPlace(q, _weights[$"{prefix}.attn_q_norm.weight"], Config.NumHeads, hd, Config.Eps);
                        RMSNormInPlace(k, _weights[$"{prefix}.attn_k_norm.weight"], kvHeads, hd, Config.Eps);
                    }
                    else
                    {
                        q = ApplyBatchRMSNorm(q, $"{prefix}.attn_q_norm.weight", Config.NumHeads, seqLen, hd);
                        k = ApplyBatchRMSNorm(k, $"{prefix}.attn_k_norm.weight", kvHeads, seqLen, hd);
                    }
                    ApplyUnweightedRMSNorm(v, kvHeads, hd, seqLen);
                }
                _attnFusedDecodePreprocessApplied = fusedDone;
            }
            else
            {
                q = LinearForward(input, $"{prefix}.attn_q.weight");

                if (seqLen == 1)
                    RMSNormInPlace(q, _weights[$"{prefix}.attn_q_norm.weight"], Config.NumHeads, hd, Config.Eps);
                else
                    q = ApplyBatchRMSNorm(q, $"{prefix}.attn_q_norm.weight", Config.NumHeads, seqLen, hd);

                if (!isShared)
                {
                    k = LinearForward(input, $"{prefix}.attn_k.weight");

                    bool hasVWeight = _weights.ContainsKey($"{prefix}.attn_v.weight") ||
                                      _quantWeights.ContainsKey($"{prefix}.attn_v.weight");
                    if (hasVWeight)
                        v = LinearForward(input, $"{prefix}.attn_v.weight");
                    else
                    {
                        v = new Tensor(_allocator, DType.Float32, k.Sizes);
                        Ops.Copy(v, k);
                    }

                    if (seqLen == 1)
                        RMSNormInPlace(k, _weights[$"{prefix}.attn_k_norm.weight"], kvHeads, hd, Config.Eps);
                    else
                        k = ApplyBatchRMSNorm(k, $"{prefix}.attn_k_norm.weight", kvHeads, seqLen, hd);

                    ApplyUnweightedRMSNorm(v, kvHeads, hd, seqLen);
                }
            }

            // Apply NeoX-style RoPE (skipped for global fast path - already
            // applied; also skipped for fused MLX decode preprocess, which
            // bakes RoPE into the same kernel as the Q/K/V norms).
            if (!_useGlobalFastPath && !_attnFusedDecodePreprocessApplied)
            {
                float[] freqs = isLocal ? _ropeFreqsLocal : _ropeFreqsGlobal;
                if (seqLen == 1)
                {
                    ApplyNeoXRoPEDecode(q, Config.NumHeads, hd, startPos, freqs);
                    if (k != null)
                        ApplyNeoXRoPEDecode(k, kvHeads, hd, startPos, freqs);
                }
                else if (isLocal)
                {
                    q = ApplyRoPEPrefill(q, Config.NumHeads, hd, seqLen, startPos, _ropeLocalBase);
                    if (k != null)
                        k = ApplyRoPEPrefill(k, kvHeads, hd, seqLen, startPos, _ropeLocalBase);
                }
                else
                {
                    q = ApplyNeoXRoPEPrefill(q, Config.NumHeads, hd, seqLen, startPos, freqs);
                    if (k != null)
                        k = ApplyNeoXRoPEPrefill(k, kvHeads, hd, seqLen, startPos, freqs);
                }
            }

            int totalSeqLen = startPos + seqLen;
            Tensor result;

            if (seqLen == 1)
            {
                if (!isShared)
                {
                    int cachePos = isLocal ? (startPos % _kvCacheSize[layer]) : startPos;
                    CopyToCacheDecode(_kvCacheK[layer], k, _kvCacheV[layer], v,
                        kvHeads, hd, cachePos);
                    // Periodically materialize the KV cache to break the
                    // unbounded slice_update chain that MLX's lazy graph
                    // builds up one node per decode step. Without this,
                    // each subsequent attention read has to walk the chain
                    // back to the original tensor — fine for the first few
                    // hundred tokens, catastrophic after ~500 (chain depth
                    // grows linearly in decode step count). Originally this
                    // fired only for SWA (local) layers because they were
                    // the chronic-slowdown case; global layers also chain
                    // and need the same treatment for long generations.
                    if (_backend == BackendType.Mlx
                        && MlxLocalKvMaterializeInterval > 0
                        && ((startPos + 1 + layer) % MlxLocalKvMaterializeInterval) == 0)
                    {
                        MlxFusedOps.TryMaterialize(_kvCacheK[layer]);
                        MlxFusedOps.TryMaterialize(_kvCacheV[layer]);
                    }
                }

                int kvCacheLayer = _kvDonorMap.TryGetValue(layer, out int donor) ? donor : layer;
                int cacheLen = _kvCacheSize[kvCacheLayer];

                if (isLocal)
                {
                    int attendLen = Math.Min(totalSeqLen, _slidingWindow);
                    result = new Tensor(_allocator, DType.Float32, 1, Config.NumHeads * hd);
                    int attendStart = Math.Max(0, startPos + 1 - attendLen);
                    bool usedMlx = _backend == BackendType.Mlx &&
                        MlxFusedOps.TryDecodeAttention(result, q, _kvCacheK[kvCacheLayer], _kvCacheV[kvCacheLayer],
                            Config.NumHeads, kvHeads, hd, attendStart, startPos + 1 - attendStart, cacheLen, true, 1f);
                    if (!usedMlx && !CudaFusedOps.TryGqaDecodeAttention(result, q, _kvCacheK[kvCacheLayer], _kvCacheV[kvCacheLayer],
                            Config.NumHeads, kvHeads, hd, attendStart, startPos + 1 - attendStart, cacheLen, true, 1f))
                    {
                        AttentionDecodeCircular(q, _kvCacheK[kvCacheLayer], _kvCacheV[kvCacheLayer], result,
                            Config.NumHeads, kvHeads, hd, hd,
                            startPos, attendLen, cacheLen, 1f);
                    }
                }
                else
                {
                    result = new Tensor(_allocator, DType.Float32, 1, Config.NumHeads * hd);
                    bool usedMlx = _backend == BackendType.Mlx &&
                        MlxFusedOps.TryDecodeAttention(result, q, _kvCacheK[kvCacheLayer], _kvCacheV[kvCacheLayer],
                            Config.NumHeads, kvHeads, hd, 0, totalSeqLen, cacheLen, false, 1f);
                    if (!usedMlx && !CudaFusedOps.TryGqaDecodeAttention(result, q, _kvCacheK[kvCacheLayer], _kvCacheV[kvCacheLayer],
                            Config.NumHeads, kvHeads, hd, 0, totalSeqLen, cacheLen, false, 1f))
                    {
                        AttentionDecodeWithWindow(q, _kvCacheK[kvCacheLayer], _kvCacheV[kvCacheLayer], result,
                            Config.NumHeads, kvHeads, hd, hd,
                            0, totalSeqLen, 1f);
                    }
                }
            }
            else
            {
                Tensor kHeadsForAttn = null, vHeadsForAttn = null;
                if (_useGlobalFastPath)
                {
                    // Global fast path: Q/K/V already in head-first format
                    kHeadsForAttn = _globalKHeads;
                    vHeadsForAttn = _globalVHeads;
                    CopyToCache(_kvCacheK[layer], kHeadsForAttn, startPos, seqLen);
                    CopyToCache(_kvCacheV[layer], vHeadsForAttn, startPos, seqLen);
                }
                else if (!isShared)
                {
                    Tensor kHeads = ReshapeToHeads(k, kvHeads, seqLen, hd);
                    Tensor vHeads = ReshapeToHeads(v, kvHeads, seqLen, hd);
                    if (isLocal)
                    {
                        CopyToCacheCircular(_kvCacheK[layer], kHeads, startPos, seqLen, _kvCacheSize[layer]);
                        CopyToCacheCircular(_kvCacheV[layer], vHeads, startPos, seqLen, _kvCacheSize[layer]);
                    }
                    else
                    {
                        CopyToCache(_kvCacheK[layer], kHeads, startPos, seqLen);
                        CopyToCache(_kvCacheV[layer], vHeads, startPos, seqLen);
                    }
                    kHeadsForAttn = kHeads;
                    vHeadsForAttn = vHeads;
                }

                Tensor qHeads = _useGlobalFastPath
                    ? _globalQHeads
                    : ReshapeToHeads(q, Config.NumHeads, seqLen, hd);

                int kvCacheLayer = _kvDonorMap.TryGetValue(layer, out int donor2) ? donor2 : layer;
                int groupSize = Config.NumHeads / kvHeads;

                // Identify K/V source for attention (un-expanded heads).
                //
                // For SWA layers in chunked prefill, the freshly computed K/V alone
                // is not enough whenever any chunk has seqLen > slidingWindow:
                //   * Chunk 1 (startPos == 0): non-shared layers compute fresh K/V
                //     for the full chunk and use it directly. Shared (donor-following)
                //     layers must use the donor's saved fresh K/V instead of the
                //     rolling cache, because the rolling cache only holds the last
                //     slidingWindow tokens of the donor and queries near the start
                //     of the chunk would otherwise see no in-window keys at all.
                //   * Chunks 2+ (startPos > 0): each query also legitimately attends
                //     to up to (W-1) tokens from previous chunks. Those tokens are
                //     gathered from the rolling cache *before* this chunk overwrote
                //     it (see PrepareSwaPrevWindowsForChunk) and prepended to the
                //     attention K/V source via ConcatHeadFirstKV. The mask uses
                //     maskStart = prevLen so logical position alignment holds.
                //
                // For full-attention (global) layers the persistent cache is linear
                // and contains every prior position, so the original cache-read path
                // is correct.
                Tensor kvSrcK, kvSrcV;
                int kvLen;
                bool kvIsSeqHeads = false;
                bool ownsKvSrc = false; // true if kvSrcK/V are concat tensors we allocated here
                if (kHeadsForAttn != null && (startPos == 0 || isLocal))
                {
                    if (isLocal && startPos > 0 && _swaPrevWindow != null
                        && _swaPrevWindow.TryGetValue(kvCacheLayer, out var prevKv)
                        && _swaPrevWindowLen > 0)
                    {
                        kvSrcK = ConcatHeadFirstKV(prevKv.k, kHeadsForAttn);
                        kvSrcV = ConcatHeadFirstKV(prevKv.v, vHeadsForAttn);
                        kvLen = _swaPrevWindowLen + seqLen;
                        ownsKvSrc = true;
                    }
                    else
                    {
                        kvSrcK = kHeadsForAttn;
                        kvSrcV = vHeadsForAttn;
                        kvLen = seqLen;
                    }
                    kvIsSeqHeads = true;
                }
                else if (isLocal && _prefillSWAKV != null
                         && _prefillSWAKV.TryGetValue(kvCacheLayer, out var donorKV))
                {
                    if (_swaPrevWindow != null && _swaPrevWindow.TryGetValue(kvCacheLayer, out var prevKv2)
                        && _swaPrevWindowLen > 0)
                    {
                        kvSrcK = ConcatHeadFirstKV(prevKv2.k, donorKV.k);
                        kvSrcV = ConcatHeadFirstKV(prevKv2.v, donorKV.v);
                        kvLen = _swaPrevWindowLen + seqLen;
                        ownsKvSrc = true;
                    }
                    else
                    {
                        kvSrcK = donorKV.k;
                        kvSrcV = donorKV.v;
                        kvLen = seqLen;
                    }
                    kvIsSeqHeads = true;
                }
                else
                {
                    int cacheLen = _kvCacheSize[kvCacheLayer];
                    kvLen = Math.Min(totalSeqLen, cacheLen);
                    kvSrcK = _kvCacheK[kvCacheLayer];
                    kvSrcV = _kvCacheV[kvCacheLayer];
                }

                // Block-wise windowed attention for SWA layers with very large
                // sequences (e.g., when chunking is disabled for multimodal).
                // Avoids O(n²) score tensor that can exhaust memory.
                // Disabled when multimodal soft tokens are present: that path
                // applies a plain causal+window mask and cannot honour the
                // bidirectional image span, so fall back to the full O(n²)
                // ApplyCausalMask path which does.
                bool useWindowedAttn = isLocal && kvIsSeqHeads && seqLen > _slidingWindow * 4
                    && exceptPositions == null;

                // Fused prefill attention: run Q*K^T → mask → softmax → *V as a
                // a fused backend kernel where available, eliminating several dispatches
                // when there are no special mask exceptions (multimodal tokens).
                bool canUseFusedPrefillAttn = !useWindowedAttn && kvIsSeqHeads && exceptPositions == null;
                result = null;

                if (IsGgmlBackend && canUseFusedPrefillAttn)
                {
                    int windowSize = isLocal ? _slidingWindow : 0;
                    int maskStart = kvLen - seqLen;
                    // Native kernel outputs directly in flat [seqLen, numHeads*hd]
                    // via on-device permute+cont, skipping ReshapeFromHeadsEx copy.
                    result = new Tensor(_allocator, DType.Float32, seqLen, Config.NumHeads * hd);
                    GgmlBasicOps.FusedPrefillAttention(
                        qHeads, kvSrcK, kvSrcV, result,
                        Config.NumHeads, kvHeads, hd,
                        seqLen, kvLen,
                        maskStart, windowSize, 1.0f);
                    qHeads.Dispose();
                }
                else if (_backend == BackendType.Cuda && canUseFusedPrefillAttn)
                {
                    int windowSize = isLocal ? _slidingWindow : 0;
                    int maskStart = kvLen - seqLen;
                    var fusedResult = new Tensor(_allocator, DType.Float32, seqLen, Config.NumHeads * hd);
                    if (CudaFusedOps.TryGqaPrefillAttention(
                        fusedResult, qHeads, kvSrcK, kvSrcV,
                        Config.NumHeads, kvHeads, hd,
                        seqLen, kvLen,
                        maskStart, windowSize, 1.0f))
                    {
                        result = fusedResult;
                        qHeads.Dispose();
                    }
                    else
                    {
                        fusedResult.Dispose();
                    }
                }
                else if (_backend == BackendType.Mlx && canUseFusedPrefillAttn)
                {
                    int windowSize = isLocal ? _slidingWindow : 0;
                    int maskStart = kvLen - seqLen;
                    var fusedResult = new Tensor(_allocator, DType.Float32, seqLen, Config.NumHeads * hd);
                    if (MlxFusedOps.TryPrefillAttention(
                        fusedResult, qHeads, kvSrcK, kvSrcV,
                        Config.NumHeads, kvHeads, hd,
                        seqLen, kvLen,
                        maskStart, windowSize, 1.0f))
                    {
                        result = fusedResult;
                        qHeads.Dispose();
                    }
                    else
                    {
                        fusedResult.Dispose();
                    }
                }

                if (result == null && useWindowedAttn)
                {
                    result = WindowedPrefillAttention(qHeads, kvSrcK, kvSrcV,
                        Config.NumHeads, kvHeads, seqLen, hd);
                    qHeads.Dispose();
                }
                else if (result == null)
                {
                    Tensor kExpanded = ExpandKVHeads(kvSrcK, groupSize, kvLen);
                    Tensor vExpanded = ExpandKVHeads(kvSrcV, groupSize, kvLen);

                    using var kT = kExpanded.Transpose(1, 2);
                    var scores = new Tensor(_allocator, DType.Float32, Config.NumHeads, seqLen, kvLen);
                    Ops.AddmmBatch(scores, 0, scores, 1f, qHeads, kT);
                    qHeads.Dispose();
                    kExpanded.Dispose();

                    int windowSize = isLocal ? _slidingWindow : 0;
                    // Bidirectional (non-causal) attention over multimodal
                    // soft-token spans must apply on EVERY layer - including
                    // the local/SWA layers (Gemma 4 is ~5:1 SWA:global, so
                    // restricting the exception to global layers left the
                    // image tokens causally masked on ~83% of layers and
                    // badly degraded image understanding). llama.cpp encodes
                    // the image batch with set_causal_attn(false) on all
                    // layers; the soft-token span fits inside the 1024 SWA
                    // window so the window bound never clips it.
                    ApplyCausalMask(scores, seqLen, kvLen, windowSize, exceptPositions);
                    Ops.Softmax(scores, scores);

                    var attnOut = new Tensor(_allocator, DType.Float32, Config.NumHeads, seqLen, hd);
                    Ops.AddmmBatch(attnOut, 0, attnOut, 1.0f, scores, vExpanded);
                    scores.Dispose();
                    vExpanded.Dispose();

                    result = ReshapeFromHeadsEx(attnOut, Config.NumHeads, seqLen, hd);
                    attnOut.Dispose();
                }

                // Save the non-shared SWA donor's freshly computed K/V so shared
                // layers downstream in this chunk can attend to it directly. The
                // alternative (reading the rolling cache) is incorrect whenever
                // seqLen > slidingWindow, because the cache is overwritten with
                // the last slidingWindow positions and queries near the start of
                // the chunk would lose all of their in-window keys.
                if (isLocal && kHeadsForAttn != null
                    && _swaKVDonorLayers.Contains(layer) && _prefillSWAKV != null)
                {
                    _prefillSWAKV[layer] = (kHeadsForAttn, vHeadsForAttn);
                }
                else
                {
                    kHeadsForAttn?.Dispose();
                    vHeadsForAttn?.Dispose();
                }

                if (ownsKvSrc)
                {
                    kvSrcK.Dispose();
                    kvSrcV.Dispose();
                }
            }

            q?.Dispose();
            k?.Dispose();
            v?.Dispose();

            _attnTicks += Stopwatch.GetTimestamp() - t0;

            using (result)
            {
                return LinearForward(result, $"{prefix}.attn_output.weight");
            }
        }

        /// <summary>
        /// Block-wise windowed attention for SWA prefill. Instead of computing
        /// the full [numHeads, seqLen, seqLen] score matrix (O(n²)), splits queries
        /// into blocks of slidingWindow size and computes attention only against
        /// keys within the window, reducing peak memory for very large sequences.
        /// </summary>
        private Tensor WindowedPrefillAttention(
            Tensor qHeads, Tensor kHeads, Tensor vHeads,
            int numQHeads, int numKVHeads, int seqLen, int headDim)
        {
            int W = _slidingWindow;
            int groupSize = numQHeads / numKVHeads;
            var output = new Tensor(_allocator, DType.Float32, numQHeads, seqLen, headDim);

            for (int bStart = 0; bStart < seqLen; bStart += W)
            {
                int bLen = Math.Min(W, seqLen - bStart);
                int kStart = Math.Max(0, bStart - W + 1);
                int kEnd = bStart + bLen;
                int kLen = kEnd - kStart;

                Tensor qBlock;
                using (var qNarrow = qHeads.Narrow(1, bStart, bLen))
                    qBlock = Ops.NewContiguous(qNarrow);

                Tensor kBlock;
                using (var kNarrow = kHeads.Narrow(1, kStart, kLen))
                {
                    var kContig = Ops.NewContiguous(kNarrow);
                    if (groupSize <= 1) { kBlock = kContig; }
                    else { kBlock = Ops.RepeatInterleave(null, kContig, groupSize, 0); kContig.Dispose(); }
                }

                Tensor vBlock;
                using (var vNarrow = vHeads.Narrow(1, kStart, kLen))
                {
                    var vContig = Ops.NewContiguous(vNarrow);
                    if (groupSize <= 1) { vBlock = vContig; }
                    else { vBlock = Ops.RepeatInterleave(null, vContig, groupSize, 0); vContig.Dispose(); }
                }

                using var kT = kBlock.Transpose(1, 2);
                var scores = new Tensor(_allocator, DType.Float32, numQHeads, bLen, kLen);
                Ops.AddmmBatch(scores, 0, scores, 1f, qBlock, kT);
                qBlock.Dispose();
                kBlock.Dispose();

                ApplyCausalMask(scores, bLen, kLen, _slidingWindow, null);
                Ops.Softmax(scores, scores);

                var attnBlock = new Tensor(_allocator, DType.Float32, numQHeads, bLen, headDim);
                Ops.AddmmBatch(attnBlock, 0, attnBlock, 1f, scores, vBlock);
                scores.Dispose();
                vBlock.Dispose();

                using (var outSlice = output.Narrow(1, bStart, bLen))
                    Ops.Copy(outSlice, attnBlock);
                attnBlock.Dispose();
            }

            Tensor flatResult = ReshapeFromHeadsEx(output, numQHeads, seqLen, headDim);
            output.Dispose();
            return flatResult;
        }

        /// <summary>
        /// Fused QKV split + transpose to head-first layout in a single strided
        /// copy. Combines the Narrow→NewContiguous and View→Transpose→NewContiguous
        /// steps, eliminating one full tensor copy per projection.
        /// </summary>
        private Tensor SliceColumnsContiguous(Tensor src, int colOffset, int width)
        {
            var result = new Tensor(_allocator, DType.Float32, src.Sizes[0], width);
            if (CudaFusedOps.TrySliceColumns(result, src, colOffset, width))
                return result;
            result.Dispose();

            using var view = src.Narrow(1, colOffset, width);
            return Ops.NewContiguous(view);
        }

        private unsafe Tensor SplitQKVToHeadFirst(Tensor qkv, int colOffset, int numHeads, int seqLen, int headDim)
        {
            var result = new Tensor(_allocator, DType.Float32, numHeads, seqLen, headDim);
            if (CudaFusedOps.TrySplitQkvToHeadFirst(result, qkv, colOffset, numHeads, seqLen, headDim))
                return result;
            if (MlxFusedOps.TryFlatToHeadFirst(result, qkv, numHeads, seqLen, headDim, colOffset))
                return result;

            float* src = GetFloatPtr(qkv);
            float* dst = GetFloatPtr(result);
            int qkvStride = (int)qkv.Sizes[1];
            int headBytes = headDim * sizeof(float);

            if (numHeads * seqLen >= 64)
            {
                int totalWork = numHeads * seqLen;
                System.Threading.Tasks.Parallel.For(0, totalWork, idx =>
                {
                    int h = idx / seqLen;
                    int s = idx % seqLen;
                    float* srcRow = src + (long)s * qkvStride + colOffset + h * headDim;
                    float* dstRow = dst + ((long)h * seqLen + s) * headDim;
                    Buffer.MemoryCopy(srcRow, dstRow, headBytes, headBytes);
                });
            }
            else
            {
                for (int h = 0; h < numHeads; h++)
                {
                    float* dstHead = dst + (long)h * seqLen * headDim;
                    for (int s = 0; s < seqLen; s++)
                    {
                        float* srcRow = src + (long)s * qkvStride + colOffset + h * headDim;
                        Buffer.MemoryCopy(srcRow, dstHead + (long)s * headDim, headBytes, headBytes);
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// NeoX RoPE for head-first layout [numHeads, seqLen, headDim].
        /// Uses the same cached cos/sin table as the flat-layout version.
        /// </summary>
        private unsafe void ApplyNeoXRoPEHeadFirst(Tensor data, int numHeads, int headDim,
            int seqLen, int startPos, float[] freqs)
        {
            int ropeHalf = freqs.Length;
            bool rebuiltTables = false;

            if (_neoXRopeCos == null || _neoXRopeCacheSeqLen != seqLen ||
                _neoXRopeCacheStartPos != startPos || _neoXRopeCacheFreqs != freqs)
            {
                int tableSize = seqLen * ropeHalf;
                if (_neoXRopeCos == null || _neoXRopeCos.Length != tableSize)
                {
                    _neoXRopeCos = new float[tableSize];
                    _neoXRopeSin = new float[tableSize];
                }
                for (int s = 0; s < seqLen; s++)
                {
                    int pos = startPos + s;
                    int off = s * ropeHalf;
                    for (int j = 0; j < ropeHalf; j++)
                    {
                        float angle = pos * freqs[j];
                        _neoXRopeCos[off + j] = MathF.Cos(angle);
                        _neoXRopeSin[off + j] = MathF.Sin(angle);
                    }
                }
                _neoXRopeCacheSeqLen = seqLen;
                _neoXRopeCacheStartPos = startPos;
                _neoXRopeCacheFreqs = freqs;
                rebuiltTables = true;
            }

            if (EnsureNeoXRopeDeviceTables(seqLen, startPos, freqs, rebuiltTables))
            {
                if (CudaFusedOps.TryNeoXRoPEHeadFirst(data, _neoXRopeCosTensor, _neoXRopeSinTensor,
                        numHeads, seqLen, headDim, ropeHalf) ||
                    MlxFusedOps.TryNeoXRoPEHeadFirstInPlace(data, _neoXRopeCosTensor, _neoXRopeSinTensor,
                        numHeads, seqLen, headDim, ropeHalf))
                {
                    return;
                }
            }

            float* ptr = GetFloatPtr(data);
            if (numHeads * seqLen >= 64)
            {
                int totalWork = numHeads * seqLen;
                var cosTab = _neoXRopeCos;
                var sinTab = _neoXRopeSin;
                System.Threading.Tasks.Parallel.For(0, totalWork, idx =>
                {
                    int h = idx / seqLen;
                    int s = idx % seqLen;
                    int tableOff = s * ropeHalf;
                    float* head = ptr + ((long)h * seqLen + s) * headDim;
                    for (int j = 0; j < ropeHalf; j++)
                    {
                        float cos = cosTab[tableOff + j];
                        float sin = sinTab[tableOff + j];
                        float x0 = head[j];
                        float x1 = head[j + ropeHalf];
                        head[j] = x0 * cos - x1 * sin;
                        head[j + ropeHalf] = x0 * sin + x1 * cos;
                    }
                });
            }
            else
            {
                for (int h = 0; h < numHeads; h++)
                {
                    for (int s = 0; s < seqLen; s++)
                    {
                        int tableOff = s * ropeHalf;
                        float* head = ptr + ((long)h * seqLen + s) * headDim;
                        for (int j = 0; j < ropeHalf; j++)
                        {
                            float cos = _neoXRopeCos[tableOff + j];
                            float sin = _neoXRopeSin[tableOff + j];
                            float x0 = head[j];
                            float x1 = head[j + ropeHalf];
                            head[j] = x0 * cos - x1 * sin;
                            head[j + ropeHalf] = x0 * sin + x1 * cos;
                        }
                    }
                }
            }
        }

        private bool EnsureNeoXRopeDeviceTables(int seqLen, int startPos, float[] freqs, bool rebuiltTables)
        {
            if ((_backend != BackendType.Cuda && _backend != BackendType.Mlx) || _neoXRopeCos == null || _neoXRopeSin == null)
                return false;

            int tableSize = seqLen * freqs.Length;
            bool needsUpload = rebuiltTables ||
                _neoXRopeCosTensor == null ||
                (int)_neoXRopeCosTensor.Sizes[0] != tableSize ||
                _neoXRopeDeviceCacheSeqLen != seqLen ||
                _neoXRopeDeviceCacheStartPos != startPos ||
                _neoXRopeDeviceCacheFreqs != freqs;

            if (!needsUpload)
                return true;

            if (_neoXRopeCosTensor == null || (int)_neoXRopeCosTensor.Sizes[0] != tableSize)
            {
                _neoXRopeCosTensor?.Dispose();
                _neoXRopeSinTensor?.Dispose();
                _neoXRopeCosTensor = new Tensor(_allocator, DType.Float32, tableSize);
                _neoXRopeSinTensor = new Tensor(_allocator, DType.Float32, tableSize);
            }

            _neoXRopeCosTensor.SetElementsAsFloat(_neoXRopeCos);
            _neoXRopeSinTensor.SetElementsAsFloat(_neoXRopeSin);
            _neoXRopeDeviceCacheSeqLen = seqLen;
            _neoXRopeDeviceCacheStartPos = startPos;
            _neoXRopeDeviceCacheFreqs = freqs;
            return true;
        }

        private Tensor ApplyBatchRMSNorm(Tensor data, string weightName, int numHeads, int seqLen, int headDim)
        {
            var alpha = _weights[weightName];
            using var reshaped = data.View(seqLen * numHeads, headDim);
            Ops.RMSNorm(reshaped, reshaped, alpha, null, Config.Eps);
            return data;
        }

        private void ApplyUnweightedRMSNorm(Tensor data, int numVectors, int dim, int seqLen)
        {
            int total = seqLen * numVectors;
            if (_onesForVNorm == null || (int)_onesForVNorm.Sizes[0] != dim)
            {
                _onesForVNorm?.Dispose();
                _onesForVNorm = new Tensor(_allocator, DType.Float32, dim);
                Ops.Fill(_onesForVNorm, 1f);
            }
            using var reshaped = data.View(total, dim);
            Ops.RMSNorm(reshaped, reshaped, _onesForVNorm, null, Config.Eps);
        }

        private Tensor ApplyRoPEPrefill(Tensor data, int numHeads, int headDim,
            int seqLen, int startPos, float ropeBase)
        {
            // Cache the position tensor: all local layers in a forward pass
            // share the same (seqLen, startPos), only numHeads differs (Q vs K).
            Tensor posTensor;
            bool isQ = (numHeads == Config.NumHeads);

            if (_cachedRoPEPosSeqLen == seqLen && _cachedRoPEPosStartPos == startPos)
            {
                posTensor = isQ ? _cachedRoPEPosQ : _cachedRoPEPosK;
                if (posTensor == null || (int)posTensor.Sizes[0] != seqLen * numHeads)
                    posTensor = null;
            }
            else
            {
                _cachedRoPEPosQ?.Dispose();
                _cachedRoPEPosK?.Dispose();
                _cachedRoPEPosQ = null;
                _cachedRoPEPosK = null;
                _cachedRoPEPosSeqLen = seqLen;
                _cachedRoPEPosStartPos = startPos;
                posTensor = null;
            }

            if (posTensor == null)
            {
                int totalRows = seqLen * numHeads;
                int[] positions = new int[totalRows];
                for (int s = 0; s < seqLen; s++)
                    for (int h = 0; h < numHeads; h++)
                        positions[s * numHeads + h] = startPos + s;
                posTensor = CreateIntTensor(positions, totalRows);
                if (isQ) _cachedRoPEPosQ = posTensor;
                else _cachedRoPEPosK = posTensor;
            }

            using var reshaped = data.View(1, seqLen, numHeads, headDim);
            Ops.RoPEEx(reshaped, reshaped, posTensor, headDim, 2, 0,
                ropeBase, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
            return data;
        }

        /// <summary>
        /// Decode-step fast path: a single mlx_compile closure does the
        /// dense FFN block (pre-norm + gate_up matmul + GeluMulSplit + down
        /// matmul + post-norm + residual add) in one
        /// <c>mlx_closure_apply</c>. Returns false (and the caller falls
        /// back to the per-op path) when the layer's quant types / shape
        /// aren't covered. Requires the Q8_0 gate_up/down weight pair the
        /// dense Gemma 4 decode path uses.
        /// </summary>
        private bool TryFusedDenseFFNDecode(Tensor residual, string prefix, string postFfnNormKey)
        {
            if (!_quantWeights.TryGetValue($"{prefix}.ffn_gate_up.weight", out var gateUpQw)) return false;
            if (!_quantWeights.TryGetValue($"{prefix}.ffn_down.weight", out var downQw)) return false;
            if (!_weights.TryGetValue($"{prefix}.ffn_norm.weight", out var preNormW)) return false;
            if (!_weights.TryGetValue(postFfnNormKey, out var postNormW)) return false;

            int halfDim = (int)(gateUpQw.Ne1 / 2);
            return MlxQuantizedOps.TryFusedGemma4DenseFFNDecode(
                residual,
                residual,
                preNormW, Config.Eps,
                gateUpQw.EnsureDeviceCacheKey(), gateUpQw.Data, gateUpQw.GgmlType, gateUpQw.Ne0, gateUpQw.Ne1, gateUpQw.RawBytes,
                downQw.EnsureDeviceCacheKey(), downQw.Data, downQw.GgmlType, downQw.Ne0, downQw.Ne1, downQw.RawBytes,
                postNormW,
                halfDim);
        }

        /// <summary>
        /// Decode-step fast path: one Metal kernel does Q/K/V split +
        /// RMSNorm + Q/K NeoX RoPE. On success returns
        /// <c>q</c> [1, NumHeads*headDim] flat,
        /// <c>k</c> [kvHeads, 1, headDim] head-first (cache-write ready),
        /// <c>v</c> [kvHeads, 1, headDim] head-first.
        /// Returns false when the kernel declines (e.g. unsupported headDim,
        /// missing norm weights) — the caller falls back to the per-op path.
        /// </summary>
        private bool TryFusedDecodeQkvPreprocess(
            Tensor qkv, string prefix, bool isLocal,
            int headDim, int kvHeads, int qDim, int startPos,
            out Tensor q, out Tensor k, out Tensor v)
        {
            q = null; k = null; v = null;

            // Norm weights must exist and be on the device.
            if (!_weights.TryGetValue($"{prefix}.attn_q_norm.weight", out var qNormW)) return false;
            if (!_weights.TryGetValue($"{prefix}.attn_k_norm.weight", out var kNormW)) return false;

            float[] freqs = isLocal ? _ropeFreqsLocal : _ropeFreqsGlobal;
            int rotHalf = freqs.Length;
            if (rotHalf <= 0 || rotHalf * 2 > headDim) return false;

            // Build cos/sin tables for the scalar decode position.
            int tableSize = rotHalf;
            if (_neoXRopeCos == null || _neoXRopeCos.Length != tableSize)
            {
                _neoXRopeCos = new float[tableSize];
                _neoXRopeSin = new float[tableSize];
            }
            bool rebuiltTables = false;
            if (_neoXRopeCacheSeqLen != 1 || _neoXRopeCacheStartPos != startPos || _neoXRopeCacheFreqs != freqs)
            {
                for (int j = 0; j < rotHalf; j++)
                {
                    float angle = startPos * freqs[j];
                    _neoXRopeCos[j] = MathF.Cos(angle);
                    _neoXRopeSin[j] = MathF.Sin(angle);
                }
                _neoXRopeCacheSeqLen = 1;
                _neoXRopeCacheStartPos = startPos;
                _neoXRopeCacheFreqs = freqs;
                rebuiltTables = true;
            }
            if (!EnsureNeoXRopeDeviceTables(1, startPos, freqs, rebuiltTables)) return false;

            var qResult = new Tensor(_allocator, DType.Float32, 1, qDim);
            var kResult = new Tensor(_allocator, DType.Float32, kvHeads, 1, headDim);
            var vResult = new Tensor(_allocator, DType.Float32, kvHeads, 1, headDim);

            if (!MlxFusedOps.TryGemma4QkvPreprocessDecode(
                    qResult, kResult, vResult,
                    qkv, qNormW, kNormW,
                    _neoXRopeCosTensor, _neoXRopeSinTensor,
                    Config.NumHeads, kvHeads, headDim, rotHalf, Config.Eps))
            {
                qResult.Dispose();
                kResult.Dispose();
                vResult.Dispose();
                return false;
            }

            q = qResult;
            k = kResult;
            v = vResult;
            return true;
        }

        private unsafe void ApplyNeoXRoPEDecode(Tensor data, int numHeads, int headDim, int position, float[] freqs)
        {
            int ropeHalf = freqs.Length;
            bool rebuiltTables = false;
            if (_neoXRopeCos == null || _neoXRopeCacheSeqLen != 1 ||
                _neoXRopeCacheStartPos != position || _neoXRopeCacheFreqs != freqs)
            {
                if (_neoXRopeCos == null || _neoXRopeCos.Length != ropeHalf)
                {
                    _neoXRopeCos = new float[ropeHalf];
                    _neoXRopeSin = new float[ropeHalf];
                }

                for (int j = 0; j < ropeHalf; j++)
                {
                    float angle = position * freqs[j];
                    _neoXRopeCos[j] = MathF.Cos(angle);
                    _neoXRopeSin[j] = MathF.Sin(angle);
                }

                _neoXRopeCacheSeqLen = 1;
                _neoXRopeCacheStartPos = position;
                _neoXRopeCacheFreqs = freqs;
                rebuiltTables = true;
            }

            if (EnsureNeoXRopeDeviceTables(1, position, freqs, rebuiltTables) &&
                MlxFusedOps.TryNeoXRoPEFlatInPlace(data, _neoXRopeCosTensor, _neoXRopeSinTensor,
                    numHeads, 1, headDim, ropeHalf))
            {
                return;
            }

            float* ptr = GetFloatPtr(data);

            for (int h = 0; h < numHeads; h++)
            {
                float* head = ptr + h * headDim;
                for (int j = 0; j < ropeHalf; j++)
                {
                    float angle = position * freqs[j];
                    float cos = MathF.Cos(angle);
                    float sin = MathF.Sin(angle);
                    float x0 = head[j];
                    float x1 = head[j + ropeHalf];
                    head[j] = x0 * cos - x1 * sin;
                    head[j + ropeHalf] = x0 * sin + x1 * cos;
                }
            }
        }

        private unsafe Tensor ApplyNeoXRoPEPrefill(Tensor data, int numHeads, int headDim,
            int seqLen, int startPos, float[] freqs)
        {
            int ropeHalf = freqs.Length;
            bool rebuiltTables = false;

            // Precompute cos/sin table once, reused across all global layers
            // in the same forward pass (same seqLen, startPos, freqs).
            if (_neoXRopeCos == null || _neoXRopeCacheSeqLen != seqLen ||
                _neoXRopeCacheStartPos != startPos || _neoXRopeCacheFreqs != freqs)
            {
                int tableSize = seqLen * ropeHalf;
                if (_neoXRopeCos == null || _neoXRopeCos.Length != tableSize)
                {
                    _neoXRopeCos = new float[tableSize];
                    _neoXRopeSin = new float[tableSize];
                }
                for (int s = 0; s < seqLen; s++)
                {
                    int pos = startPos + s;
                    int off = s * ropeHalf;
                    for (int j = 0; j < ropeHalf; j++)
                    {
                        float angle = pos * freqs[j];
                        _neoXRopeCos[off + j] = MathF.Cos(angle);
                        _neoXRopeSin[off + j] = MathF.Sin(angle);
                    }
                }
                _neoXRopeCacheSeqLen = seqLen;
                _neoXRopeCacheStartPos = startPos;
                _neoXRopeCacheFreqs = freqs;
                rebuiltTables = true;
            }

            if (EnsureNeoXRopeDeviceTables(seqLen, startPos, freqs, rebuiltTables) &&
                MlxFusedOps.TryNeoXRoPEFlatInPlace(data, _neoXRopeCosTensor, _neoXRopeSinTensor,
                    numHeads, seqLen, headDim, ropeHalf))
            {
                return data;
            }

            float* ptr = GetFloatPtr(data);

            // Parallel over sequence positions: each position's heads are independent
            if (seqLen >= 64)
            {
                var cosTab = _neoXRopeCos;
                var sinTab = _neoXRopeSin;
                System.Threading.Tasks.Parallel.For(0, seqLen, s =>
                {
                    int tableOff = s * ropeHalf;
                    for (int h = 0; h < numHeads; h++)
                    {
                        float* head = ptr + ((long)s * numHeads + h) * headDim;
                        for (int j = 0; j < ropeHalf; j++)
                        {
                            float cos = cosTab[tableOff + j];
                            float sin = sinTab[tableOff + j];
                            float x0 = head[j];
                            float x1 = head[j + ropeHalf];
                            head[j] = x0 * cos - x1 * sin;
                            head[j + ropeHalf] = x0 * sin + x1 * cos;
                        }
                    }
                });
            }
            else
            {
                for (int s = 0; s < seqLen; s++)
                {
                    int tableOff = s * ropeHalf;
                    for (int h = 0; h < numHeads; h++)
                    {
                        float* head = ptr + ((long)s * numHeads + h) * headDim;
                        for (int j = 0; j < ropeHalf; j++)
                        {
                            float cos = _neoXRopeCos[tableOff + j];
                            float sin = _neoXRopeSin[tableOff + j];
                            float x0 = head[j];
                            float x1 = head[j + ropeHalf];
                            head[j] = x0 * cos - x1 * sin;
                            head[j + ropeHalf] = x0 * sin + x1 * cos;
                        }
                    }
                }
            }
            return data;
        }

        private Tensor ReshapeFromHeadsEx(Tensor data, int numHeads, int seqLen, int headDim)
        {
            if (seqLen == 1)
                return data.View(1, numHeads * headDim);

            using var transposed = data.Transpose(0, 1);
            using var contiguous = Ops.NewContiguous(transposed);
            return contiguous.View(seqLen, numHeads * headDim);
        }

        private unsafe void AttentionDecodeWithWindow(Tensor q, Tensor kCache, Tensor vCache,
            Tensor result, int numHeads, int numKVHeads, int keyDim, int valDim,
            int attendStart, int totalSeqLen, float scale)
        {
            if (kCache.ElementType == DType.Float16 && vCache.ElementType == DType.Float16)
            {
                AttentionDecodeWithWindowF16(q, kCache, vCache, result,
                    numHeads, numKVHeads, keyDim, valDim, attendStart, totalSeqLen, scale);
                return;
            }

            float* qPtr = GetFloatPtr(q);
            float* kPtr = GetFloatPtr(kCache);
            float* vPtr = GetFloatPtr(vCache);
            float* rPtr = GetFloatPtr(result);
            int maxSeqLen = (int)kCache.Sizes[1];
            int groupSize = numHeads / numKVHeads;
            int attendLen = totalSeqLen - attendStart;

            float* scores = stackalloc float[attendLen];

            for (int h = 0; h < numHeads; h++)
            {
                float* qHead = qPtr + h * keyDim;
                int kvHead = h / groupSize;
                float* kHead = kPtr + kvHead * maxSeqLen * keyDim;
                float* vHead = vPtr + kvHead * maxSeqLen * valDim;

                float maxScore = float.NegativeInfinity;
                for (int t = 0; t < attendLen; t++)
                {
                    float s = VecDot(qHead, kHead + (attendStart + t) * keyDim, keyDim) * scale;
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
                for (int t = 0; t < attendLen; t++)
                    scores[t] *= invSum;

                float* rHead = rPtr + h * valDim;
                VecZero(rHead, valDim);
                for (int t = 0; t < attendLen; t++)
                    VecScaleAdd(rHead, vHead + (attendStart + t) * valDim, scores[t], valDim);
            }
        }

        private unsafe void AttentionDecodeWithWindowF16(Tensor q, Tensor kCache, Tensor vCache,
            Tensor result, int numHeads, int numKVHeads, int keyDim, int valDim,
            int attendStart, int totalSeqLen, float scale)
        {
            float* qPtr = GetFloatPtr(q);
            ushort* kPtr = TensorComputePrimitives.GetHalfPointer(kCache);
            ushort* vPtr = TensorComputePrimitives.GetHalfPointer(vCache);
            float* rPtr = GetFloatPtr(result);
            int maxSeqLen = (int)kCache.Sizes[1];
            int groupSize = numHeads / numKVHeads;
            int attendLen = totalSeqLen - attendStart;

            float* scores = stackalloc float[attendLen];

            for (int h = 0; h < numHeads; h++)
            {
                float* qHead = qPtr + h * keyDim;
                int kvHead = h / groupSize;
                ushort* kHead = kPtr + kvHead * maxSeqLen * keyDim;
                ushort* vHead = vPtr + kvHead * maxSeqLen * valDim;

                float maxScore = float.NegativeInfinity;
                for (int t = 0; t < attendLen; t++)
                {
                    float s = TensorComputePrimitives.DotF32F16(qHead, kHead + (attendStart + t) * keyDim, keyDim) * scale;
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
                for (int t = 0; t < attendLen; t++)
                    scores[t] *= invSum;

                float* rHead = rPtr + h * valDim;
                VecZero(rHead, valDim);
                for (int t = 0; t < attendLen; t++)
                    TensorComputePrimitives.ScaleAddF16(rHead, vHead + (attendStart + t) * valDim, scores[t], valDim);
            }
        }

        private unsafe void AttentionDecodeCircular(Tensor q, Tensor kCache, Tensor vCache,
            Tensor result, int numHeads, int numKVHeads, int keyDim, int valDim,
            int currentPos, int attendLen, int cacheSize, float scale)
        {
            if (kCache.ElementType == DType.Float16 && vCache.ElementType == DType.Float16)
            {
                AttentionDecodeCircularF16(q, kCache, vCache, result,
                    numHeads, numKVHeads, keyDim, valDim, currentPos, attendLen, cacheSize, scale);
                return;
            }

            float* qPtr = GetFloatPtr(q);
            float* kPtr = GetFloatPtr(kCache);
            float* vPtr = GetFloatPtr(vCache);
            float* rPtr = GetFloatPtr(result);
            int groupSize = numHeads / numKVHeads;

            float* scores = stackalloc float[attendLen];

            int startLogicalPos = currentPos + 1 - attendLen;
            if (startLogicalPos < 0) startLogicalPos = 0;
            int actualAttendLen = currentPos + 1 - startLogicalPos;

            for (int h = 0; h < numHeads; h++)
            {
                float* qHead = qPtr + h * keyDim;
                int kvHead = h / groupSize;
                float* kHead = kPtr + kvHead * cacheSize * keyDim;
                float* vHead = vPtr + kvHead * cacheSize * valDim;

                float maxScore = float.NegativeInfinity;
                for (int t = 0; t < actualAttendLen; t++)
                {
                    int logicalPos = startLogicalPos + t;
                    int cacheIdx = logicalPos % cacheSize;
                    float s = VecDot(qHead, kHead + cacheIdx * keyDim, keyDim) * scale;
                    scores[t] = s;
                    if (s > maxScore) maxScore = s;
                }

                float sumExp = 0;
                for (int t = 0; t < actualAttendLen; t++)
                {
                    float e = MathF.Exp(scores[t] - maxScore);
                    scores[t] = e;
                    sumExp += e;
                }
                float invSum = 1f / sumExp;
                for (int t = 0; t < actualAttendLen; t++)
                    scores[t] *= invSum;

                float* rHead = rPtr + h * valDim;
                VecZero(rHead, valDim);
                for (int t = 0; t < actualAttendLen; t++)
                {
                    int logicalPos = startLogicalPos + t;
                    int cacheIdx = logicalPos % cacheSize;
                    VecScaleAdd(rHead, vHead + cacheIdx * valDim, scores[t], valDim);
                }
            }
        }

        private unsafe void AttentionDecodeCircularF16(Tensor q, Tensor kCache, Tensor vCache,
            Tensor result, int numHeads, int numKVHeads, int keyDim, int valDim,
            int currentPos, int attendLen, int cacheSize, float scale)
        {
            float* qPtr = GetFloatPtr(q);
            ushort* kPtr = TensorComputePrimitives.GetHalfPointer(kCache);
            ushort* vPtr = TensorComputePrimitives.GetHalfPointer(vCache);
            float* rPtr = GetFloatPtr(result);
            int groupSize = numHeads / numKVHeads;

            float* scores = stackalloc float[attendLen];

            int startLogicalPos = currentPos + 1 - attendLen;
            if (startLogicalPos < 0) startLogicalPos = 0;
            int actualAttendLen = currentPos + 1 - startLogicalPos;

            for (int h = 0; h < numHeads; h++)
            {
                float* qHead = qPtr + h * keyDim;
                int kvHead = h / groupSize;
                ushort* kHead = kPtr + kvHead * cacheSize * keyDim;
                ushort* vHead = vPtr + kvHead * cacheSize * valDim;

                float maxScore = float.NegativeInfinity;
                for (int t = 0; t < actualAttendLen; t++)
                {
                    int logicalPos = startLogicalPos + t;
                    int cacheIdx = logicalPos % cacheSize;
                    float s = TensorComputePrimitives.DotF32F16(qHead, kHead + cacheIdx * keyDim, keyDim) * scale;
                    scores[t] = s;
                    if (s > maxScore) maxScore = s;
                }

                float sumExp = 0;
                for (int t = 0; t < actualAttendLen; t++)
                {
                    float e = MathF.Exp(scores[t] - maxScore);
                    scores[t] = e;
                    sumExp += e;
                }
                float invSum = 1f / sumExp;
                for (int t = 0; t < actualAttendLen; t++)
                    scores[t] *= invSum;

                float* rHead = rPtr + h * valDim;
                VecZero(rHead, valDim);
                for (int t = 0; t < actualAttendLen; t++)
                {
                    int logicalPos = startLogicalPos + t;
                    int cacheIdx = logicalPos % cacheSize;
                    TensorComputePrimitives.ScaleAddF16(rHead, vHead + cacheIdx * valDim, scores[t], valDim);
                }
            }
        }

        /// <summary>
        /// Gather the live "previous window" K (or V) of an SWA layer's rolling
        /// cache into a contiguous tensor of shape [kvHeads, prevWindowLen, hd].
        ///
        /// SWA correctness in chunked prefill: queries near the start of a non-first
        /// chunk legitimately attend to up to `slidingWindow - 1` tokens from the
        /// previous chunk(s). Those tokens are still resident in the rolling cache
        /// at the moment we begin processing this chunk, so we copy them out *before*
        /// the current chunk overwrites them, and concatenate them with the freshly
        /// computed K/V to form the attention K/V source.
        ///
        /// Logical positions [startPos - prevWindowLen, startPos - 1] are written in
        /// chronological order. The source cache is circular with size = cacheSize
        /// (typically equal to slidingWindow): logical position p lives in slot
        /// `p % cacheSize`. This gather collapses to one or two contiguous memcpys
        /// per head depending on whether the live range wraps around the cache.
        /// </summary>
        private unsafe Tensor BuildSwaPrevWindow(Tensor cache, int startPos, int prevWindowLen,
            int kvHeads, int headDim, int cacheSize)
        {
            if (prevWindowLen <= 0) return null;
            var result = new Tensor(_allocator, DType.Float32, kvHeads, prevWindowLen, headDim);
            int prevStart = startPos - prevWindowLen;
            if (CudaFusedOps.TryGatherCircularHeadFirst(result, cache, prevStart, prevWindowLen, cacheSize))
                return result;

            int firstSlot = ((prevStart % cacheSize) + cacheSize) % cacheSize;

            if (cache.ElementType == DType.Float16)
            {
                ushort* cachePtrH = TensorComputePrimitives.GetHalfPointer(cache);
                float* dstPtrH = GetFloatPtr(result);
                int doParallelH = kvHeads >= 4 ? 1 : 0;
                int firstSlotLocal = firstSlot;
                int prevWindowLenLocal = prevWindowLen;
                int cacheSizeLocal = cacheSize;
                int headDimLocal = headDim;
                long cachePtrHAddr = (long)cachePtrH;
                long dstPtrHAddr = (long)dstPtrH;
                void CopyOneHeadH(int h)
                {
                    ushort* cacheHead = (ushort*)cachePtrHAddr + (long)h * cacheSizeLocal * headDimLocal;
                    float* dstHead = (float*)dstPtrHAddr + (long)h * prevWindowLenLocal * headDimLocal;
                    if (firstSlotLocal + prevWindowLenLocal <= cacheSizeLocal)
                    {
                        TensorComputePrimitives.F16ToF32(dstHead,
                            cacheHead + (long)firstSlotLocal * headDimLocal,
                            prevWindowLenLocal * headDimLocal);
                    }
                    else
                    {
                        int tailLen = cacheSizeLocal - firstSlotLocal;
                        TensorComputePrimitives.F16ToF32(dstHead,
                            cacheHead + (long)firstSlotLocal * headDimLocal,
                            tailLen * headDimLocal);
                        int headLen = prevWindowLenLocal - tailLen;
                        TensorComputePrimitives.F16ToF32(dstHead + (long)tailLen * headDimLocal,
                            cacheHead, headLen * headDimLocal);
                    }
                }
                if (doParallelH != 0)
                    System.Threading.Tasks.Parallel.For(0, kvHeads, CopyOneHeadH);
                else
                    for (int h = 0; h < kvHeads; h++) CopyOneHeadH(h);
                return result;
            }

            float* cachePtr = GetFloatPtr(cache);
            float* dstPtr = GetFloatPtr(result);
            long headBytes = (long)headDim * sizeof(float);

            int doParallel = kvHeads >= 4 ? 1 : 0;
            void CopyOneHead(int h)
            {
                float* cacheHead = cachePtr + (long)h * cacheSize * headDim;
                float* dstHead = dstPtr + (long)h * prevWindowLen * headDim;
                if (firstSlot + prevWindowLen <= cacheSize)
                {
                    long bytes = (long)prevWindowLen * headBytes;
                    Buffer.MemoryCopy(cacheHead + (long)firstSlot * headDim, dstHead, bytes, bytes);
                }
                else
                {
                    int tailLen = cacheSize - firstSlot;
                    long tailBytes = (long)tailLen * headBytes;
                    Buffer.MemoryCopy(cacheHead + (long)firstSlot * headDim, dstHead, tailBytes, tailBytes);
                    int headLen = prevWindowLen - tailLen;
                    long headRangeBytes = (long)headLen * headBytes;
                    Buffer.MemoryCopy(cacheHead, dstHead + (long)tailLen * headDim, headRangeBytes, headRangeBytes);
                }
            }
            if (doParallel != 0)
                System.Threading.Tasks.Parallel.For(0, kvHeads, CopyOneHead);
            else
                for (int h = 0; h < kvHeads; h++) CopyOneHead(h);
            return result;
        }

        /// <summary>
        /// Concatenate two head-first K-or-V tensors along the sequence axis.
        /// a: [kvHeads, lenA, hd], b: [kvHeads, lenB, hd], result: [kvHeads, lenA+lenB, hd].
        /// Used to prepend the SWA "previous window" gathered from the rolling cache to
        /// the current chunk's freshly computed K/V before running attention.
        /// </summary>
        private unsafe Tensor ConcatHeadFirstKV(Tensor a, Tensor b)
        {
            int kvHeads = (int)a.Sizes[0];
            int lenA = (int)a.Sizes[1];
            int lenB = (int)b.Sizes[1];
            int hd = (int)a.Sizes[2];
            int totalLen = lenA + lenB;
            var result = new Tensor(_allocator, DType.Float32, kvHeads, totalLen, hd);
            if (CudaFusedOps.TryConcatHeadFirst(result, a, b))
                return result;

            float* aPtr = GetFloatPtr(a);
            float* bPtr = GetFloatPtr(b);
            float* dstPtr = GetFloatPtr(result);
            long aHeadBytes = (long)lenA * hd * sizeof(float);
            long bHeadBytes = (long)lenB * hd * sizeof(float);

            void CopyOneHead(int h)
            {
                float* dstHead = dstPtr + (long)h * totalLen * hd;
                Buffer.MemoryCopy(aPtr + (long)h * lenA * hd, dstHead, aHeadBytes, aHeadBytes);
                Buffer.MemoryCopy(bPtr + (long)h * lenB * hd, dstHead + lenA * hd, bHeadBytes, bHeadBytes);
            }
            if (kvHeads >= 4)
                System.Threading.Tasks.Parallel.For(0, kvHeads, CopyOneHead);
            else
                for (int h = 0; h < kvHeads; h++) CopyOneHead(h);
            return result;
        }

        /// <summary>
        /// At the start of each prefill chunk or fused decode step, gather the
        /// SWA "previous window" K/V from the rolling cache for every distinct
        /// cache-owning SWA layer. Done once per call so KV-shared SWA layers
        /// can reuse the same gather as their donor. Must be called before any
        /// layer writes fresh K/V to the circular cache.
        /// </summary>
        private void PrepareSwaPrevWindowsForChunk(int startPos, int seqLen)
        {
            DisposeSwaPrevWindows();
            if (seqLen <= 0 || startPos <= 0 || _slidingWindow <= 0 || _kvCacheK == null) return;
            int W = _slidingWindow;
            int prevWindowLen = Math.Min(startPos, W - 1);
            if (prevWindowLen <= 0) return;

            _swaPrevWindow = new Dictionary<int, (Tensor, Tensor)>();
            _swaPrevWindowStartPos = startPos;
            _swaPrevWindowLen = prevWindowLen;

            var seenLayers = new HashSet<int>();
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!IsLocalLayer(l)) continue;
                int srcLayer = _kvDonorMap.TryGetValue(l, out int donor) ? donor : l;
                if (!seenLayers.Add(srcLayer)) continue;

                int kvHeads = KVHeadsForLayer(srcLayer);
                int hd = HeadDimForLayer(srcLayer);
                int cacheSize = _kvCacheSize[srcLayer];
                Tensor kPrev = BuildSwaPrevWindow(_kvCacheK[srcLayer], startPos, prevWindowLen, kvHeads, hd, cacheSize);
                Tensor vPrev = BuildSwaPrevWindow(_kvCacheV[srcLayer], startPos, prevWindowLen, kvHeads, hd, cacheSize);
                _swaPrevWindow[srcLayer] = (kPrev, vPrev);
            }
        }

        private void DisposeSwaPrevWindows()
        {
            if (_swaPrevWindow == null) return;
            foreach (var kv in _swaPrevWindow.Values)
            {
                kv.k?.Dispose();
                kv.v?.Dispose();
            }
            _swaPrevWindow = null;
            _swaPrevWindowStartPos = -1;
            _swaPrevWindowLen = 0;
        }

        private unsafe void CopyToCacheCircular(Tensor cache, Tensor src, int startPos, int seqLen, int cacheSize)
        {
            if (CudaFusedOps.TryCopyHeadFirstToCache(cache, src, startPos, seqLen, cacheSize, true))
                return;
            if (TryCopyHeadFirstToCacheMlx(cache, src, startPos, seqLen, circular: true))
                return;

            int numHeads = (int)cache.Sizes[0];
            int headDim = (int)cache.Sizes[2];

            if (cache.ElementType == DType.Float16)
            {
                float* srcPtrF16 = GetFloatPtr(src);
                ushort* cachePtrF16 = TensorComputePrimitives.GetHalfPointer(cache);
                int totalWorkF16 = seqLen * numHeads;
                if (totalWorkF16 >= 64)
                {
                    long srcAddrF16 = (long)srcPtrF16;
                    long dstAddrF16 = (long)cachePtrF16;
                    int seqLenLocal = seqLen;
                    int cacheSizeLocal = cacheSize;
                    int headDimLocal = headDim;
                    int startPosLocal = startPos;
                    int numHeadsLocal = numHeads;
                    System.Threading.Tasks.Parallel.For(0, totalWorkF16, idx =>
                    {
                        int s = idx / numHeadsLocal;
                        int h = idx % numHeadsLocal;
                        int cacheIdx = (startPosLocal + s) % cacheSizeLocal;
                        float* srcRow = (float*)srcAddrF16 + (long)h * seqLenLocal * headDimLocal + (long)s * headDimLocal;
                        ushort* dstRow = (ushort*)dstAddrF16 + (long)h * cacheSizeLocal * headDimLocal + (long)cacheIdx * headDimLocal;
                        TensorComputePrimitives.F32ToF16(dstRow, srcRow, headDimLocal);
                    });
                }
                else
                {
                    for (int s = 0; s < seqLen; s++)
                    {
                        int cacheIdx = (startPos + s) % cacheSize;
                        for (int h = 0; h < numHeads; h++)
                        {
                            float* srcRow = srcPtrF16 + (long)h * seqLen * headDim + (long)s * headDim;
                            ushort* dstRow = cachePtrF16 + (long)h * cacheSize * headDim + (long)cacheIdx * headDim;
                            TensorComputePrimitives.F32ToF16(dstRow, srcRow, headDim);
                        }
                    }
                }
                InvalidateTensorDeviceCache(cache);
                return;
            }

            float* srcPtr = GetFloatPtr(src);
            float* cachePtr = GetFloatPtr(cache);
            int headBytes = headDim * sizeof(float);

            int totalWork = seqLen * numHeads;
            if (totalWork >= 64)
            {
                System.Threading.Tasks.Parallel.For(0, totalWork, idx =>
                {
                    int s = idx / numHeads;
                    int h = idx % numHeads;
                    int cacheIdx = (startPos + s) % cacheSize;
                    float* srcRow = srcPtr + (long)h * seqLen * headDim + (long)s * headDim;
                    float* dstRow = cachePtr + (long)h * cacheSize * headDim + (long)cacheIdx * headDim;
                    Buffer.MemoryCopy(srcRow, dstRow, headBytes, headBytes);
                });
            }
            else
            {
                for (int s = 0; s < seqLen; s++)
                {
                    int cacheIdx = (startPos + s) % cacheSize;
                    for (int h = 0; h < numHeads; h++)
                    {
                        float* srcRow = srcPtr + (long)h * seqLen * headDim + (long)s * headDim;
                        float* dstRow = cachePtr + (long)h * cacheSize * headDim + (long)cacheIdx * headDim;
                        Buffer.MemoryCopy(srcRow, dstRow, headBytes, headBytes);
                    }
                }
            }

            InvalidateTensorDeviceCache(cache);
        }

        private unsafe void ApplyCausalMask(Tensor scores, int queryLen, int totalKVLen, int windowSize,
            HashSet<int> exceptPositions = null)
        {
            int startPos = totalKVLen - queryLen;

            if (exceptPositions != null && exceptPositions.Count > 0)
            {
                // Combined causal + sliding-window + bidirectional-image mask.
                // A query at absolute position qa attends to key kv when:
                //   * kv is within the (optional) sliding window lower bound, AND
                //   * kv <= qa (causal)  OR  both qa and kv are multimodal
                //     soft tokens (bidirectional within the image/audio span).
                // This matches llama.cpp's set_causal_attn(false) image batch:
                // soft tokens see each other in both directions on every layer,
                // but never attend forward to ordinary (text) tokens, and text
                // tokens never attend forward to the image. Window + bidi are
                // handled together here so the separate window pass below is
                // skipped (early return).
                float* sPtr = GetFloatPtr(scores);
                int numHeads = (int)scores.Sizes[0];
                int rowStride = queryLen * totalKVLen;

                for (int h = 0; h < numHeads; h++)
                {
                    float* headScores = sPtr + h * rowStride;
                    for (int q = 0; q < queryLen; q++)
                    {
                        int queryAbsPos = startPos + q;
                        bool queryIsExcept = exceptPositions.Contains(queryAbsPos);
                        int lowerBound = windowSize > 0 ? queryAbsPos - windowSize + 1 : 0;
                        float* row = headScores + q * totalKVLen;
                        for (int kv = 0; kv < totalKVLen; kv++)
                        {
                            bool allowed;
                            if (kv <= queryAbsPos)
                                allowed = windowSize <= 0 || kv >= lowerBound;
                            else
                                allowed = queryIsExcept && exceptPositions.Contains(kv);
                            if (!allowed)
                                row[kv] = float.NegativeInfinity;
                        }
                    }
                }
                InvalidateTensorDeviceCache(scores);
                return;
            }
            else
            {
                Ops.AddCausalMask(scores, queryLen, startPos, float.NegativeInfinity);
            }

            if (windowSize > 0)
            {
                // Cache per-row fill widths: all SWA layers in a forward pass share
                // the same (queryLen, startPos, windowSize), so the widths are
                // identical across layers. Recompute only when parameters change.
                if (_cachedSWAMaskWidths == null ||
                    _cachedSWAMaskQueryLen != queryLen ||
                    _cachedSWAMaskStartPos != startPos)
                {
                    _cachedSWAMaskWidths = new int[queryLen];
                    _cachedSWAMaskQueryLen = queryLen;
                    _cachedSWAMaskStartPos = startPos;
                    for (int q = 0; q < queryLen; q++)
                        _cachedSWAMaskWidths[q] = Math.Max(0, startPos + q - windowSize + 1);
                }

                float* sPtr = GetFloatPtr(scores);
                int numHeads = (int)scores.Sizes[0];
                int rowStride = queryLen * totalKVLen;

                for (int h = 0; h < numHeads; h++)
                {
                    float* headScores = sPtr + h * rowStride;
                    for (int q = 0; q < queryLen; q++)
                    {
                        int width = _cachedSWAMaskWidths[q];
                        if (width > 0)
                            new Span<float>(headScores + q * totalKVLen, width).Fill(float.NegativeInfinity);
                    }
                }
                InvalidateTensorDeviceCache(scores);
            }
        }

        #endregion

        public override void Dispose()
        {
            _onesForVNorm?.Dispose();
            _neoXRopeCosTensor?.Dispose();
            _neoXRopeSinTensor?.Dispose();
            _visionEncoder?.Dispose();
            _audioEncoder?.Dispose();
            foreach (var (emb, _) in _pendingVisionEmbeddingsList)
                emb?.Dispose();
            _pendingVisionEmbeddingsList.Clear();
            foreach (var (emb, _) in _pendingAudioEmbeddingsList)
                emb?.Dispose();
            _pendingAudioEmbeddingsList.Clear();
            DisposeSwaPrevWindows();
            if (_kvCacheK != null)
            {
                var disposed = new HashSet<int>();
                for (int l = 0; l < Config.NumLayers; l++)
                {
                    if (_kvDonorMap.ContainsKey(l)) continue;
                    if (disposed.Contains(l)) continue;
                    _kvCacheK[l]?.Dispose();
                    _kvCacheV[l]?.Dispose();
                    disposed.Add(l);
                }
            }
            base.Dispose();
        }
    }
}
