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
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.GGML;
using TensorSharp.MLX;

namespace TensorSharp.Models
{
    /// <summary>
    /// Nemotron-H hybrid model: mixes Mamba2 SSM layers, attention-only layers, and FFN-only layers.
    /// Per-layer type is determined by GGUF metadata arrays (head_count_kv and feed_forward_length).
    /// - Mamba2 layer: head_count_kv == 0 AND feed_forward_length == 0
    /// - Attention-only layer: head_count_kv > 0 AND feed_forward_length == 0
    /// - FFN-only layer: feed_forward_length > 0
    /// Attention uses no RoPE. FFN uses ReLU-squared activation. MoE is supported on FFN layers.
    ///
    /// Performance optimization: for decode (seqLen=1), small operations (RMSNorm, residual add,
    /// small matmuls like expert and router) are executed on CPU to avoid Metal GPU dispatch overhead
    /// (~1ms+ per dispatch). Large matmuls (SSM in/out, attention QKV/output, LM head) remain on GPU.
    /// </summary>
    public partial class NemotronModelEvalConfig
    {
        public static readonly int MlxEvalEveryNLayers = Resolve();
        private static int Resolve()
        {
            string env = Environment.GetEnvironmentVariable("TS_MLX_EVAL_EVERY_N_LAYERS");
            if (!string.IsNullOrWhiteSpace(env) && int.TryParse(env, out int v) && v > 0)
                return v;
            return 16;
        }
    }

    public partial class NemotronModel : ModelBase
    {
        // LayerType is internal to the model but the batched forward partial
        // (NemotronModel.BatchedForward.cs) needs to read _layerTypes per layer.
        internal enum LayerType { Mamba2, Attention, FFN }

        private LayerType[] _layerTypes;
        private int[] _layerNFF;
        private int[] _layerNumHeads;
        private int[] _layerNumKVHeads;

        // Attention KV cache (only for attention layers)
        private Tensor[] _kvCacheK;
        private Tensor[] _kvCacheV;

        // Mamba2 SSM state
        private float[][] _convState;       // [layer][convChannels * (convKernelSize-1)]
        private float[][] _ssmState;         // [layer][dState * headDim * nHead]
        private float[][] _mamba2ConvWT;     // [layer][convKernelSize * convChannels], transposed for SIMD
        private Tensor[] _mamba2NativeDecodeProjected;
        private Tensor[] _mamba2NativeDecodeHidden;
        private bool[] _mamba2NativeDecodeStateInitialized;

        // SSM config
        private int _ssmDConv;
        private int _ssmDInner;
        private int _ssmDState;
        private int _ssmNHead;
        private int _ssmNGroup;
        private int _ssmHeadDim;

        // Attention config
        private double _attentionScale;

        // MoE config
        private int _numExperts;
        private int _numExpertsUsed;
        private bool _expertWeightsNorm;
        private float _expertWeightsScale;

        // Pre-allocated decode buffers for Mamba2
        private float[] _mamba2ConvOutBuf;
        private float[] _mamba2YBuf;
        private float[] _mamba2SiluTmpBuf;

        // Pre-cached per-layer data
        private string[] _layerPrefixes;

        private struct MoELayerInfo
        {
            public bool HasLatentIn;
            public bool HasSharedExperts;
            public int LatentDim;
        }
        private MoELayerInfo[] _moeLayerInfo;

        // CPU matmul threshold: disabled (0) by default on Apple Silicon unified memory
        // where GPU dispatch overhead is low. Enable for discrete GPU systems.
        private const long CPU_MATMUL_THRESHOLD = 0L;
        private const int DefaultBatchedMoEPrefillMinTokens = 8;
        private const int DefaultNativeMamba2PrefillMinTokens = 8;
        private const int DefaultMultimodalWarmupTileCount = 3;
        private static long s_nextNativeMamba2DecodeModelId;
        private readonly ulong _nativeMamba2DecodeModelId =
            (ulong)Interlocked.Increment(ref s_nextNativeMamba2DecodeModelId);
        private static readonly bool DisableBatchedMoEPrefill =
            string.Equals(Environment.GetEnvironmentVariable("TS_NEMOTRON_MOE_PREFILL_BATCHED"), "0", StringComparison.Ordinal);
        private static readonly bool EnableFusedMoEPrefill =
            string.Equals(Environment.GetEnvironmentVariable("TS_NEMOTRON_MOE_PREFILL_FUSED"), "1", StringComparison.Ordinal);
        private static readonly int BatchedMoEPrefillMinTokens =
            ParsePositiveIntEnv("TS_NEMOTRON_MOE_PREFILL_BATCHED_MIN_TOKENS", DefaultBatchedMoEPrefillMinTokens);
        private static readonly bool DisableFusedLinearResidual =
            string.Equals(Environment.GetEnvironmentVariable("TS_NEMOTRON_LINEAR_RESIDUAL_FUSED"), "0", StringComparison.Ordinal);
        private static readonly bool EnableFusedLinearResidualPrefill =
            string.Equals(Environment.GetEnvironmentVariable("TS_NEMOTRON_LINEAR_RESIDUAL_FUSED_PREFILL"), "1", StringComparison.Ordinal);
        private static readonly bool DisableNativeMamba2Prefill =
            string.Equals(Environment.GetEnvironmentVariable("TS_NEMOTRON_MAMBA2_NATIVE_PREFILL"), "0", StringComparison.Ordinal);
        private static readonly bool DisableNativeMamba2Decode =
            string.Equals(Environment.GetEnvironmentVariable("TS_NEMOTRON_MAMBA2_NATIVE_DECODE"), "0", StringComparison.Ordinal);
        private static readonly bool DisableMultimodalWarmup =
            string.Equals(Environment.GetEnvironmentVariable("TS_NEMOTRON_MULTIMODAL_WARMUP"), "0", StringComparison.Ordinal);
        private static readonly int NativeMamba2PrefillMinTokens =
            ParsePositiveIntEnv("TS_NEMOTRON_MAMBA2_NATIVE_PREFILL_MIN_TOKENS", DefaultNativeMamba2PrefillMinTokens);
        private static readonly int MultimodalWarmupTileCount =
            ParsePositiveIntEnvAllowOne("TS_NEMOTRON_MULTIMODAL_WARMUP_TILES", DefaultMultimodalWarmupTileCount);

        private static int ParsePositiveIntEnv(string name, int defaultValue)
        {
            string value = Environment.GetEnvironmentVariable(name);
            return int.TryParse(value, out int parsed) && parsed > 0
                ? Math.Max(2, parsed)
                : defaultValue;
        }

        private static int ParsePositiveIntEnvAllowOne(string name, int defaultValue)
        {
            string value = Environment.GetEnvironmentVariable(name);
            return int.TryParse(value, out int parsed) && parsed > 0
                ? parsed
                : defaultValue;
        }

        // Pre-allocated MoE buffers
        private float[] _moeProbs;
        private float[] _moeSelectionProbs;
        private int[] _moeTopExperts;
        private float[] _moeRouteW;
        private float[] _moeLatentAccum;
        private int[] _moePrefillSelectedExperts;
        private float[] _moePrefillRoutingWeights;
        private int[] _moePrefillExpertCounts;
        private int[] _moePrefillExpertOffsets;
        private int[] _moePrefillExpertCursors;
        private int[] _moePrefillRoutedRows;
        private float[] _moePrefillRoutedWeights;

        // Pre-cached expert weight key strings to avoid string interpolation in hot loop
        private string[][] _expertUpKeys;   // [layer][expertIdx]
        private string[][] _expertDownKeys; // [layer][expertIdx]

        // Pre-cached expert QuantizedWeight refs to avoid dictionary lookups in hot loop
        private QuantizedWeight[][] _expertUpQW;   // [layer][expertIdx]
        private QuantizedWeight[][] _expertDownQW; // [layer][expertIdx]

        // Stacked expert tensors used by the multi-token GGML MoE prefill kernel.
        private StackedExpertWeights[] _layerStackedUp;
        private StackedExpertWeights[] _layerStackedDown;

        // Pre-allocated buffers for batched MoE P/Invoke call
        private IntPtr[] _moeUpPtrs;
        private IntPtr[] _moeDownPtrs;

        // Pre-allocated tensors for expert matmul reuse (avoids per-expert allocation)
        private Tensor _expertUpResult;
        private Tensor _expertDownResult;
        private int _expertUpDim;
        private int _expertDownDim;

        // Pre-allocated tensor for latent_out input (avoids per-token allocation)
        private Tensor _latentAccumTensor;
        private Tensor _latentOutResult;

        // Scratch tensors for the MLX batched-MoE decode path. Replaces the
        // K-per-layer × {up matmul, ReLU², down matmul, scaled add} sequence
        // — 18+ MLX kernel dispatches per layer — with a 4-dispatch batched
        // chain (batched up · ReLU² · batched down · weighted sum). On
        // Nemotron-H 30B Q2 with 23 MoE layers and K=6 this drops ~322 MLX
        // kernel launches per generated token, which is where the per-token
        // ms budget was going (matmul = 76% of token time, per the timer
        // breakdown). All five tensors are sized once per layer's
        // (intermediate, hidden) on first use and held until model
        // teardown; the cost is ~one extra K×hidden + K×intermediate +
        // small-control-tensor allocation, far less than the kernel-launch
        // savings.
        private Tensor _moeBatchedUp;          // [K, intermediate] F32 MLX
        private Tensor _moeBatchedDown;        // [K, hidden]       F32 MLX
        private Tensor _moeBatchedExpertIndices;  // [K]             I32 MLX
        private Tensor _moeBatchedRouteWeights;   // [1, K]          F32 MLX
        private Tensor _moeBatchedResult;      // [1, hidden]       F32 MLX
        private int _moeBatchedIntermediate;   // last (intermediate, hidden) used so we
        private int _moeBatchedHidden;         //  can detect a shape change and rebuild

        // Multimodal: pending injections to apply at the next Forward() call.
        private NemotronVisionEncoder _visionEncoder;
        private readonly List<(Tensor embeddings, int position)> _pendingVisionEmbeddings = new();
        private readonly List<(Tensor embeddings, int position)> _pendingAudioEmbeddings = new();

        public NemotronVisionEncoder VisionEncoder => _visionEncoder;

        public void LoadVisionEncoder(string mmProjPath)
        {
            // The mmproj uses the GGML allocator path so we get fast Metal/CUDA matmul
            // for the vision encoder's BF16 weights. Falls back to CPU when the LM is on CPU.
            IAllocator visionAllocator = _backend == BackendType.Cuda
                ? new CpuAllocator(BlasEnum.DotNet)
                : _allocator;
            _visionEncoder = new NemotronVisionEncoder(mmProjPath, visionAllocator);
            _visionEncoder.SetHostModel(this);
        }

        public NemotronImageProcessor ImageProcessor =>
            _visionEncoder?.ImageProcessor
                ?? throw new InvalidOperationException("Vision encoder must be loaded before accessing the image processor.");

        public void SetVisionEmbeddings(Tensor embeddings, int insertPosition)
        {
            _pendingVisionEmbeddings.Add((embeddings, insertPosition));
        }

        public void SetAudioEmbeddings(Tensor embeddings, int insertPosition)
        {
            _pendingAudioEmbeddings.Add((embeddings, insertPosition));
        }

        public override void WarmUpMultimodalKernels()
        {
            if (_visionEncoder == null || DisableMultimodalWarmup)
                return;

            Tensor tileEmbeddings = null;
            Tensor warmupEmbeddings = null;
            try
            {
                int imageSize = _visionEncoder.ImageSize;
                int channels = Math.Max(1, _visionEncoder.NumChannels);
                float[] pixels = new float[channels * imageSize * imageSize];

                tileEmbeddings = _visionEncoder.Encode(pixels, imageSize, imageSize);
                int tileTokens = (int)tileEmbeddings.Sizes[0];
                int warmupTiles = Math.Max(1, Math.Min(MultimodalWarmupTileCount, ImageProcessor.MaxTiles));
                int warmupImageTokens = Math.Max(tileTokens, tileTokens * warmupTiles);

                warmupEmbeddings = CreateRepeatedEmbeddingRows(tileEmbeddings, warmupImageTokens);
                if (!ReferenceEquals(warmupEmbeddings, tileEmbeddings))
                {
                    tileEmbeddings.Dispose();
                    tileEmbeddings = null;
                }

                int safeToken = Config?.VocabSize > 1 ? 1 : 0;
                int imageTokenId = Tokenizer.LookupToken("<image>");
                int imageStartId = Tokenizer.LookupToken("<img>");
                int imageEndId = Tokenizer.LookupToken("</img>");
                if (imageTokenId < 0) imageTokenId = safeToken;
                if (imageStartId < 0) imageStartId = safeToken;
                if (imageEndId < 0) imageEndId = safeToken;

                var tokens = new int[warmupImageTokens + 2];
                tokens[0] = imageStartId;
                Array.Fill(tokens, imageTokenId, 1, warmupImageTokens);
                tokens[tokens.Length - 1] = imageEndId;

                bool handedOffTileEmbedding = ReferenceEquals(warmupEmbeddings, tileEmbeddings);
                SetVisionEmbeddings(warmupEmbeddings, 1);
                if (handedOffTileEmbedding)
                    tileEmbeddings = null;
                warmupEmbeddings = null;
                ForwardRefill(tokens);
                Console.WriteLine($"  Nemotron multimodal warmup: 1 vision tile + {warmupImageTokens} image-token prefill");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"  Nemotron multimodal warmup skipped: {ex.Message}");
            }
            finally
            {
                tileEmbeddings?.Dispose();
                warmupEmbeddings?.Dispose();
                ClearPendingMultimodalEmbeddings();
                ResetKVCache();
                ResetForwardTiming();
            }
        }

        public NemotronModel(string ggufPath, BackendType backend)
            : base(ggufPath, backend)
        {
            string arch = _gguf.GetString("general.architecture") ?? "nemotron_h";
            Config = new ModelConfig { Architecture = arch };
            ParseBaseConfig();

            // SSM config
            _ssmDConv = (int)_gguf.GetUint32($"{arch}.ssm.conv_kernel");
            _ssmDInner = (int)_gguf.GetUint32($"{arch}.ssm.inner_size");
            _ssmDState = (int)_gguf.GetUint32($"{arch}.ssm.state_size");
            _ssmNHead = (int)_gguf.GetUint32($"{arch}.ssm.time_step_rank");
            _ssmNGroup = (int)_gguf.GetUint32($"{arch}.ssm.group_count");
            _ssmHeadDim = _ssmNHead > 0 ? _ssmDInner / _ssmNHead : 0;

            // Attention scale
            _attentionScale = _gguf.GetFloat32($"{arch}.attention.scale", 0f);

            // MoE config
            _numExperts = (int)_gguf.GetUint32($"{arch}.expert_count", 0);
            _numExpertsUsed = (int)_gguf.GetUint32($"{arch}.expert_used_count", 0);
            Config.NumExperts = _numExperts;
            Config.NumExpertsUsed = _numExpertsUsed;
            _expertWeightsNorm = _gguf.GetBool($"{arch}.expert_weights_norm", false);
            _expertWeightsScale = _gguf.GetFloat32($"{arch}.expert_weights_scale", 1.0f);

            // Per-layer config from GGUF arrays
            var headCountKV = _gguf.GetUint32Array($"{arch}.attention.head_count_kv");
            var ffnLength = _gguf.GetUint32Array($"{arch}.feed_forward_length");
            var headCount = _gguf.GetUint32Array($"{arch}.attention.head_count");

            int numLayers = Config.NumLayers;
            _layerTypes = new LayerType[numLayers];
            _layerNFF = new int[numLayers];
            _layerNumHeads = new int[numLayers];
            _layerNumKVHeads = new int[numLayers];

            int attnCount = 0, mamba2Count = 0, ffnCount = 0;
            for (int i = 0; i < numLayers; i++)
            {
                uint kvHeads = (headCountKV != null && i < headCountKV.Length) ? headCountKV[i] : 1;
                uint ff = (ffnLength != null && i < ffnLength.Length) ? ffnLength[i] : 0;
                _layerNFF[i] = (int)ff;

                if (kvHeads == 0 && ff == 0)
                {
                    _layerTypes[i] = LayerType.Mamba2;
                    mamba2Count++;
                }
                else if (ff == 0)
                {
                    _layerTypes[i] = LayerType.Attention;
                    attnCount++;
                }
                else
                {
                    _layerTypes[i] = LayerType.FFN;
                    ffnCount++;
                }

                _layerNumKVHeads[i] = (int)kvHeads;
                uint hc = (headCount != null && i < headCount.Length) ? headCount[i] : (uint)Config.NumHeads;
                _layerNumHeads[i] = (int)hc;
            }

            if (Config.NumHeads <= 1 || Config.NumKVHeads <= 0)
            {
                for (int i = 0; i < numLayers; i++)
                {
                    if (_layerTypes[i] == LayerType.Attention && _layerNumHeads[i] > 0 && _layerNumKVHeads[i] > 0)
                    {
                        if (Config.NumHeads <= 1) Config.NumHeads = _layerNumHeads[i];
                        if (Config.NumKVHeads <= 0) Config.NumKVHeads = _layerNumKVHeads[i];
                        break;
                    }
                }
            }

            ParseTokenizer();

            Console.WriteLine($"Model: {arch}, Layers={numLayers}, Hidden={Config.HiddenSize}, " +
                $"Heads={Config.NumHeads}, KVHeads={Config.NumKVHeads}, HeadDim={Config.HeadDim}, Vocab={Config.VocabSize}");
            Console.WriteLine($"SSM: dConv={_ssmDConv}, dInner={_ssmDInner}, dState={_ssmDState}, " +
                $"nHead={_ssmNHead}, nGroup={_ssmNGroup}, headDim={_ssmHeadDim}");
            Console.WriteLine($"Layer types: {attnCount} attention, {mamba2Count} Mamba2, {ffnCount} FFN" +
                (_numExperts > 0 ? $" (MoE: {_numExperts} experts, top-{_numExpertsUsed})" : " (dense)"));
            if (_attentionScale != 0)
                Console.WriteLine($"Attention scale: {_attentionScale}");

            LoadWeights();

            if (_numExperts == 0)
                FuseFFNWeights();
            FuseQKVWeights();
            PrepareCudaQuantizedWeightsForInference();

            int maxContextLength = ResolveConfiguredContextLength();
            int initialCacheLength = ResolveInitialCacheAllocationLength(maxContextLength);
            if (initialCacheLength < maxContextLength)
                Console.WriteLine($"Initial {_backend} KV cache allocation: {initialCacheLength} tokens (grows on demand up to {maxContextLength}).");
            InitCaches(initialCacheLength, maxContextLength);
            InitMamba2Buffers();
            InitLayerInfo();
            CacheMamba2ConvWeights();
            InitMoEBuffers();
        }

        #region Initialization

        private unsafe void FuseQKVWeights()
        {
            int fused = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_layerTypes[l] != LayerType.Attention)
                    continue;

                string qName = $"blk.{l}.attn_q.weight";
                string kName = $"blk.{l}.attn_k.weight";
                string vName = $"blk.{l}.attn_v.weight";
                string qkvName = $"blk.{l}.attn_qkv.weight";

                if (_quantWeights.TryGetValue(qName, out var qw) &&
                    _quantWeights.TryGetValue(kName, out var kw) &&
                    _quantWeights.TryGetValue(vName, out var vw) &&
                    qw.GgmlType == kw.GgmlType && kw.GgmlType == vw.GgmlType &&
                    qw.Ne0 == kw.Ne0 && kw.Ne0 == vw.Ne0)
                {
                    if (!TryCreateFusedQuantizedWeight(out QuantizedWeight fusedWeight, qw, kw, vw))
                        continue;

                    _quantWeights[qkvName] = fusedWeight;
                    _quantWeights.Remove(qName); qw.Dispose();
                    _quantWeights.Remove(kName); kw.Dispose();
                    _quantWeights.Remove(vName); vw.Dispose();
                    fused++;
                }
                else if (_weights.TryGetValue(qName, out var qf) &&
                         _weights.TryGetValue(kName, out var kf) &&
                         _weights.TryGetValue(vName, out var vf))
                {
                    int qDim = (int)qf.Sizes[0], kDim = (int)kf.Sizes[0], vDim = (int)vf.Sizes[0];
                    int inDim = (int)qf.Sizes[1];
                    var fusedTensor = new Tensor(_allocator, DType.Float32, qDim + kDim + vDim, inDim);
                    using (var s0 = fusedTensor.Narrow(0, 0, qDim)) Ops.Copy(s0, qf);
                    using (var s1 = fusedTensor.Narrow(0, qDim, kDim)) Ops.Copy(s1, kf);
                    using (var s2 = fusedTensor.Narrow(0, qDim + kDim, vDim)) Ops.Copy(s2, vf);
                    _weights[qkvName] = fusedTensor;
                    _weights.Remove(qName); qf.Dispose();
                    _weights.Remove(kName); kf.Dispose();
                    _weights.Remove(vName); vf.Dispose();
                    fused++;
                }
            }
            if (fused > 0)
                Console.WriteLine($"  Fused projections: {fused} QKV");
        }

        private unsafe void FuseFFNWeights()
        {
            // Placeholder for dense FFN weight fusion
        }

        private void InitMamba2Buffers()
        {
            int xBCSize = _ssmDInner + 2 * _ssmNGroup * _ssmDState;
            _mamba2ConvOutBuf = new float[xBCSize];
            _mamba2YBuf = new float[_ssmDInner];
            _mamba2SiluTmpBuf = new float[xBCSize];
        }

        private unsafe void CacheMamba2ConvWeights()
        {
            int numLayers = Config.NumLayers;
            int convChannels = _ssmDInner + 2 * _ssmNGroup * _ssmDState;
            _mamba2ConvWT = new float[numLayers][];

            for (int l = 0; l < numLayers; l++)
            {
                if (_layerTypes[l] != LayerType.Mamba2)
                    continue;

                string name = _layerPrefixes[l] + "ssm_conv1d.weight";
                if (!_weights.TryGetValue(name, out Tensor weight))
                    continue;

                float* src = GetFloatPtr(weight);
                var transposed = new float[_ssmDConv * convChannels];
                for (int ch = 0; ch < convChannels; ch++)
                {
                    for (int k = 0; k < _ssmDConv; k++)
                        transposed[k * convChannels + ch] = src[ch * _ssmDConv + k];
                }

                _mamba2ConvWT[l] = transposed;
            }
        }

        private int _kvCacheCapacity;

        private void InitCaches(int initialSeqLen, int maxSeqLen)
        {
            _maxContextLength = maxSeqLen;
            _kvCacheCapacity = initialSeqLen;
            int numLayers = Config.NumLayers;
            _kvCacheK = new Tensor[numLayers];
            _kvCacheV = new Tensor[numLayers];
            _convState = new float[numLayers][];
            _ssmState = new float[numLayers][];
            if (IsGgmlBackend)
            {
                _mamba2NativeDecodeProjected = new Tensor[numLayers];
                _mamba2NativeDecodeHidden = new Tensor[numLayers];
                _mamba2NativeDecodeStateInitialized = new bool[numLayers];
            }
            ApplyModelAlignedKvCacheDefault(_quantWeights);

            int convDim = Math.Max(0, _ssmDConv - 1);
            int convChannels = _ssmDInner + 2 * _ssmNGroup * _ssmDState;
            int ssmStateSize = _ssmDState * _ssmHeadDim * _ssmNHead;
            int dInProjTotal = 2 * _ssmDInner + 2 * _ssmNGroup * _ssmDState + _ssmNHead;

            DType kvDtype = _kvCacheDtype.ToDType();
            for (int l = 0; l < numLayers; l++)
            {
                switch (_layerTypes[l])
                {
                    case LayerType.Attention:
                        int numKVH = _layerNumKVHeads[l];
                        int headDim = Config.HeadDim;
                        _kvCacheK[l] = new Tensor(_allocator, kvDtype, numKVH, initialSeqLen, headDim);
                        _kvCacheV[l] = new Tensor(_allocator, kvDtype, numKVH, initialSeqLen, headDim);
                        InitializeCacheTensor(_kvCacheK[l]);
                        InitializeCacheTensor(_kvCacheV[l]);
                        break;
                    case LayerType.Mamba2:
                        _convState[l] = new float[convDim * convChannels];
                        _ssmState[l] = new float[ssmStateSize];
                        if (IsGgmlBackend)
                        {
                            _mamba2NativeDecodeProjected[l] = new Tensor(_allocator, DType.Float32, 1, dInProjTotal);
                            _mamba2NativeDecodeHidden[l] = new Tensor(_allocator, DType.Float32, 1, _ssmDInner);
                        }
                        break;
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

            DType kvDtype = _kvCacheDtype.ToDType();
            int headDim = Config.HeadDim;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_layerTypes[l] != LayerType.Attention)
                    continue;

                int numKVH = _layerNumKVHeads[l];
                var newK = new Tensor(_allocator, kvDtype, numKVH, newCapacity, headDim);
                var newV = new Tensor(_allocator, kvDtype, numKVH, newCapacity, headDim);
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
            Console.WriteLine($"Expanded Nemotron attention cache to {newCapacity} tokens.");
        }

        private unsafe void InitLayerInfo()
        {
            int numLayers = Config.NumLayers;
            _layerPrefixes = new string[numLayers];
            _moeLayerInfo = new MoELayerInfo[numLayers];

            for (int l = 0; l < numLayers; l++)
            {
                _layerPrefixes[l] = $"blk.{l}.";

                if (_layerTypes[l] == LayerType.FFN && _numExperts > 0)
                {
                    string prefix = _layerPrefixes[l];
                    ref var info = ref _moeLayerInfo[l];
                    info.HasLatentIn = _quantWeights.ContainsKey(prefix + "ffn_latent_in.weight") ||
                                       _weights.ContainsKey(prefix + "ffn_latent_in.weight");
                    info.HasSharedExperts = _quantWeights.ContainsKey(prefix + "ffn_up_shexp.weight") ||
                                            _weights.ContainsKey(prefix + "ffn_up_shexp.weight");

                    if (info.HasLatentIn)
                    {
                        string key = prefix + "ffn_latent_in.weight";
                        if (_quantWeights.TryGetValue(key, out var qw))
                            info.LatentDim = (int)qw.Ne1;
                        else if (_weights.TryGetValue(key, out var fw))
                            info.LatentDim = (int)fw.Sizes[0];
                    }
                }
            }
        }

        private void InitMoEBuffers()
        {
            if (_numExperts <= 0) return;
            _moeProbs = new float[_numExperts];
            _moeSelectionProbs = new float[_numExperts];
            _moeTopExperts = new int[_numExpertsUsed];
            _moeRouteW = new float[_numExpertsUsed];

            int maxLatentDim = Config.HiddenSize;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_layerTypes[l] == LayerType.FFN && _moeLayerInfo[l].LatentDim > 0)
                    maxLatentDim = Math.Max(maxLatentDim, _moeLayerInfo[l].LatentDim);
            }
            _moeLatentAccum = new float[maxLatentDim];

            int maxUpDim = 0, maxDownDim = 0;
            _expertUpKeys = new string[Config.NumLayers][];
            _expertDownKeys = new string[Config.NumLayers][];
            _expertUpQW = new QuantizedWeight[Config.NumLayers][];
            _expertDownQW = new QuantizedWeight[Config.NumLayers][];
            _layerStackedUp = new StackedExpertWeights[Config.NumLayers];
            _layerStackedDown = new StackedExpertWeights[Config.NumLayers];
            int stackedCapable = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_layerTypes[l] != LayerType.FFN) continue;
                _expertUpKeys[l] = new string[_numExperts];
                _expertDownKeys[l] = new string[_numExperts];
                _expertUpQW[l] = new QuantizedWeight[_numExperts];
                _expertDownQW[l] = new QuantizedWeight[_numExperts];
                for (int e = 0; e < _numExperts; e++)
                {
                    _expertUpKeys[l][e] = $"blk.{l}.ffn_up_exps.{e}.weight";
                    _expertDownKeys[l][e] = $"blk.{l}.ffn_down_exps.{e}.weight";
                    _quantWeights.TryGetValue(_expertUpKeys[l][e], out _expertUpQW[l][e]);
                    _quantWeights.TryGetValue(_expertDownKeys[l][e], out _expertDownQW[l][e]);
                }

                if (_expertUpQW[l][0] != null)
                    maxUpDim = Math.Max(maxUpDim, (int)_expertUpQW[l][0].Ne1);
                if (_expertDownQW[l][0] != null)
                    maxDownDim = Math.Max(maxDownDim, (int)_expertDownQW[l][0].Ne1);

                string prefix = _layerPrefixes[l].TrimEnd('.');
                _stackedExpertWeights.TryGetValue(prefix + ".ffn_up_exps.weight", out _layerStackedUp[l]);
                _stackedExpertWeights.TryGetValue(prefix + ".ffn_down_exps.weight", out _layerStackedDown[l]);
                if (_layerStackedUp[l] != null && _layerStackedDown[l] != null)
                    stackedCapable++;
            }

            if (stackedCapable > 0)
                Console.WriteLine($"  Nemotron batched MoE prefill kernel available on {stackedCapable}/{Config.NumLayers} layers (min tokens: {BatchedMoEPrefillMinTokens})");

            // Diagnostic: also log Mamba2 ssm_in / ssm_out quant types.
            // These matmuls run on every Mamba2 layer (23 layers in Nemotron-H
            // Nano-Omni 30B); if they're IQ4_NL they benefit from the
            // multi-row IQ4_NL kernel during prefill. If they use a different
            // quant, the Rows kernel doesn't help prefill matmul perf.
            if (_quantWeights != null)
            {
                var mamba2Types = new System.Collections.Generic.Dictionary<int, int>();
                long mamba2Bytes = 0;
                foreach (var kv in _quantWeights)
                {
                    if (kv.Key.Contains("ssm_in.weight") || kv.Key.Contains("ssm_out.weight"))
                    {
                        mamba2Types.TryGetValue(kv.Value.GgmlType, out int c);
                        mamba2Types[kv.Value.GgmlType] = c + 1;
                        mamba2Bytes += kv.Value.RawBytes;
                    }
                }
                if (mamba2Types.Count > 0)
                {
                    var parts = new System.Collections.Generic.List<string>();
                    foreach (var kv in mamba2Types) parts.Add($"ggml_type={kv.Key} ×{kv.Value}");
                    Console.WriteLine($"  Nemotron Mamba2 ssm_in/ssm_out quant: {string.Join(", ", parts)} ({mamba2Bytes / (1024.0 * 1024.0):F0} MB total)");
                }
            }

            // Diagnostic: log the quant types of MoE expert weights. Helps
            // diagnose why the MLX batched-MoE decode kernel might not fire
            // (it currently only supports IQ2_XXS). One-shot, only when
            // experts exist on at least one layer (Nemotron-H's first
            // layer is usually attention or Mamba2, not MoE).
            if (_numExperts > 0 && _expertUpQW != null)
            {
                var upTypes = new System.Collections.Generic.Dictionary<int, int>();
                var downTypes = new System.Collections.Generic.Dictionary<int, int>();
                long upBytes = 0, downBytes = 0;
                int upPerExpertNe0 = 0, upPerExpertNe1 = 0;
                int downPerExpertNe0 = 0, downPerExpertNe1 = 0;
                for (int l = 0; l < Config.NumLayers; l++)
                {
                    if (_expertUpQW[l] == null) continue;
                    for (int e = 0; e < _numExperts; e++)
                    {
                        var up = _expertUpQW[l][e];
                        var dn = _expertDownQW[l][e];
                        if (up != null)
                        {
                            upTypes.TryGetValue(up.GgmlType, out int c1);
                            upTypes[up.GgmlType] = c1 + 1;
                            upBytes += up.RawBytes;
                            upPerExpertNe0 = (int)up.Ne0;
                            upPerExpertNe1 = (int)up.Ne1;
                        }
                        if (dn != null)
                        {
                            downTypes.TryGetValue(dn.GgmlType, out int c2);
                            downTypes[dn.GgmlType] = c2 + 1;
                            downBytes += dn.RawBytes;
                            downPerExpertNe0 = (int)dn.Ne0;
                            downPerExpertNe1 = (int)dn.Ne1;
                        }
                    }
                }
                static string FormatTypes(System.Collections.Generic.Dictionary<int, int> d)
                {
                    if (d.Count == 0) return "(none)";
                    var parts = new System.Collections.Generic.List<string>();
                    foreach (var kv in d)
                        parts.Add($"ggml_type={kv.Key} ×{kv.Value}");
                    return string.Join(", ", parts);
                }
                Console.WriteLine($"  Nemotron MoE expert quant: up={FormatTypes(upTypes)} (Ne0={upPerExpertNe0}, Ne1={upPerExpertNe1}, {upBytes / (1024.0 * 1024.0):F0} MB total)");
                Console.WriteLine($"  Nemotron MoE expert quant: down={FormatTypes(downTypes)} (Ne0={downPerExpertNe0}, Ne1={downPerExpertNe1}, {downBytes / (1024.0 * 1024.0):F0} MB total)");
            }

            _expertUpDim = maxUpDim;
            _expertDownDim = maxDownDim;
            if (maxUpDim > 0)
                _expertUpResult = new Tensor(_allocator, DType.Float32, 1, maxUpDim);
            if (maxDownDim > 0)
                _expertDownResult = new Tensor(_allocator, DType.Float32, 1, maxDownDim);

            _moeUpPtrs = new IntPtr[_numExpertsUsed];
            _moeDownPtrs = new IntPtr[_numExpertsUsed];

            int maxLatent = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_layerTypes[l] == LayerType.FFN && _moeLayerInfo[l].HasLatentIn)
                    maxLatent = Math.Max(maxLatent, _moeLayerInfo[l].LatentDim);
            }
            if (maxLatent > 0)
            {
                _latentAccumTensor = new Tensor(_allocator, DType.Float32, 1, maxLatent);
                _latentOutResult = new Tensor(_allocator, DType.Float32, 1, Config.HiddenSize);
            }
        }

        private void EnsureMoEPrefillRouteBuffers(int totalRoutes)
        {
            if (_moePrefillSelectedExperts == null || _moePrefillSelectedExperts.Length != totalRoutes)
                _moePrefillSelectedExperts = new int[totalRoutes];
            if (_moePrefillRoutingWeights == null || _moePrefillRoutingWeights.Length != totalRoutes)
                _moePrefillRoutingWeights = new float[totalRoutes];
            if (_moePrefillRoutedRows == null || _moePrefillRoutedRows.Length != totalRoutes)
                _moePrefillRoutedRows = new int[totalRoutes];
            if (_moePrefillRoutedWeights == null || _moePrefillRoutedWeights.Length != totalRoutes)
                _moePrefillRoutedWeights = new float[totalRoutes];
            if (_moePrefillExpertCounts == null || _moePrefillExpertCounts.Length != _numExperts)
                _moePrefillExpertCounts = new int[_numExperts];
            if (_moePrefillExpertOffsets == null || _moePrefillExpertOffsets.Length != _numExperts + 1)
                _moePrefillExpertOffsets = new int[_numExperts + 1];
            if (_moePrefillExpertCursors == null || _moePrefillExpertCursors.Length != _numExperts)
                _moePrefillExpertCursors = new int[_numExperts];
        }

        #endregion

        public override void ResetKVCache()
        {
            for (int l = 0; l < Config.NumLayers; l++)
            {
                switch (_layerTypes[l])
                {
                    case LayerType.Attention:
                        ResetCacheTensor(_kvCacheK[l]);
                        ResetCacheTensor(_kvCacheV[l]);
                        break;
                    case LayerType.Mamba2:
                        Array.Clear(_convState[l]);
                        Array.Clear(_ssmState[l]);
                        break;
                }
            }
            if (_mamba2NativeDecodeStateInitialized != null)
            {
                Array.Clear(_mamba2NativeDecodeStateInitialized);
                if (IsGgmlBackend)
                    GgmlBasicOps.NemotronMamba2DecodeClear(_nativeMamba2DecodeModelId);
            }
            _cacheSeqLen = 0;
            _linearTicks = _attnTicks = _normTicks = _embTicks = _lmHeadTicks = _logitsCopyTicks = 0;
            _forwardCount = 0;
            _forwardSw.Reset();
        }

        public override bool SupportsKVCacheTruncation => false;

        // Per-block snapshot for Nemotron-H (mix of attention, Mamba2 SSM, and
        // FFN-only layers). Each block bundles:
        //   * For every attention layer L: K bytes for [start,start+B), V bytes
        //     for [start,start+B).
        //   * For every Mamba2 layer L: a snapshot of convState (ring buffer)
        //     and ssmState (recurrent state) AT THE END of this block. Mamba2
        //     state is a function of all preceding tokens, so capture happens
        //     after each prefill chunk lands (RequiresPerBlockCapture=true).
        //   * FFN layers contribute zero bytes (stateless).
        public override bool RequiresPerBlockCapture => true;

        public override bool SupportsKVStateSnapshot => _kvCacheK != null && _kvCacheV != null;

        public override string KVStateFingerprint
        {
            get
            {
                int attn = 0, mamba = 0, ffn = 0;
                if (_layerTypes != null)
                    for (int l = 0; l < _layerTypes.Length; l++)
                    {
                        if (_layerTypes[l] == LayerType.Attention) attn++;
                        else if (_layerTypes[l] == LayerType.Mamba2) mamba++;
                        else ffn++;
                    }
                return $"nemotron|arch={Config.Architecture}|L={Config.NumLayers}|attn={attn}|mamba2={mamba}|ffn={ffn}|H={Config.NumHeads}|D={Config.HeadDim}|ssmInner={_ssmDInner}|ssmState={_ssmDState}|ssmHead={_ssmHeadDim}|ssmNHead={_ssmNHead}|conv={_ssmDConv}|dtype={_kvCacheDtype.ToShortString()}";
            }
        }

        public override long ComputeKVBlockByteSize(int tokenCount)
        {
            if (tokenCount <= 0 || _kvCacheK == null || _layerTypes == null) return 0;
            long total = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                switch (_layerTypes[l])
                {
                    case LayerType.Attention:
                        if (_kvCacheK[l] == null || _kvCacheV[l] == null) return 0;
                        total += NemotronLayerBlockBytes(_kvCacheK[l], tokenCount);
                        total += NemotronLayerBlockBytes(_kvCacheV[l], tokenCount);
                        break;
                    case LayerType.Mamba2:
                        total += (long)_convState[l].Length * sizeof(float);
                        total += (long)_ssmState[l].Length * sizeof(float);
                        break;
                    // FFN layers: stateless, contribute nothing.
                }
            }
            return total;
        }

        public override bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        {
            if (!SupportsKVStateSnapshot) return false;
            long expected = ComputeKVBlockByteSize(tokenCount);
            if (destination.Length != expected) return false;

            int offset = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                switch (_layerTypes[l])
                {
                    case LayerType.Attention:
                        if (startToken + tokenCount > _cacheSeqLen) return false;
                        if (!CopyAttentionOut(_kvCacheK[l], startToken, tokenCount, destination[offset..], out int wK))
                            return false;
                        offset += wK;
                        if (!CopyAttentionOut(_kvCacheV[l], startToken, tokenCount, destination[offset..], out int wV))
                            return false;
                        offset += wV;
                        break;
                    case LayerType.Mamba2:
                        if (!CopyMamba2StateOut(l, destination[offset..], out int wM))
                            return false;
                        offset += wM;
                        break;
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

            int offset = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                switch (_layerTypes[l])
                {
                    case LayerType.Attention:
                        if (!CopyAttentionIn(_kvCacheK[l], destToken, tokenCount, source[offset..], out int rK))
                            return false;
                        offset += rK;
                        if (!CopyAttentionIn(_kvCacheV[l], destToken, tokenCount, source[offset..], out int rV))
                            return false;
                        offset += rV;
                        break;
                    case LayerType.Mamba2:
                        if (!CopyMamba2StateIn(l, source[offset..], out int rM))
                            return false;
                        offset += rM;
                        break;
                }
            }
            _cacheSeqLen = destToken + tokenCount;

            // The native Mamba2 decode shadow state mirrors _convState / _ssmState
            // lazily. Force a refresh on the next decode step so it picks up the
            // host arrays we just rewrote.
            if (_mamba2NativeDecodeStateInitialized != null)
            {
                Array.Clear(_mamba2NativeDecodeStateInitialized);
                if (IsGgmlBackend)
                    GgmlBasicOps.NemotronMamba2DecodeClear(_nativeMamba2DecodeModelId);
            }
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_layerTypes[l] != LayerType.Attention) continue;
                InvalidateTensorDeviceCache(_kvCacheK[l]);
                InvalidateTensorDeviceCache(_kvCacheV[l]);
            }
            return true;
        }

        private static long NemotronLayerBlockBytes(Tensor cacheTensor, int tokenCount)
        {
            long numKVHeads = cacheTensor.Sizes[0];
            long capacity = cacheTensor.Sizes[1];
            long rowBytes = cacheTensor.Storage.ByteLength / (numKVHeads * capacity);
            return numKVHeads * tokenCount * rowBytes;
        }

        private static bool CopyAttentionOut(Tensor t, int startToken, int tokenCount, Span<byte> dest, out int written)
        {
            t.Storage.EnsureHostReadable();
            long numKVHeads = t.Sizes[0];
            long capacity = t.Sizes[1];
            long rowBytes = t.Storage.ByteLength / (numKVHeads * capacity);
            long blockBytes = numKVHeads * tokenCount * rowBytes;
            if (dest.Length < blockBytes) { written = 0; return false; }
            IntPtr basePtr = t.Storage.PtrAtElement(0);
            unsafe
            {
                byte* src = (byte*)basePtr;
                fixed (byte* dst = dest)
                {
                    long perHead = tokenCount * rowBytes;
                    for (long h = 0; h < numKVHeads; h++)
                    {
                        long s = (h * capacity + startToken) * rowBytes;
                        long d = h * perHead;
                        Buffer.MemoryCopy(src + s, dst + d, dest.Length - d, perHead);
                    }
                }
            }
            written = (int)blockBytes;
            return true;
        }

        private static bool CopyAttentionIn(Tensor t, int destToken, int tokenCount, ReadOnlySpan<byte> source, out int read)
        {
            t.Storage.EnsureHostReadable();
            long numKVHeads = t.Sizes[0];
            long capacity = t.Sizes[1];
            if (destToken + tokenCount > capacity) { read = 0; return false; }
            long rowBytes = t.Storage.ByteLength / (numKVHeads * capacity);
            long blockBytes = numKVHeads * tokenCount * rowBytes;
            if (source.Length < blockBytes) { read = 0; return false; }
            IntPtr basePtr = t.Storage.PtrAtElement(0);
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
                        Buffer.MemoryCopy(srcBase + s, dst + d, t.Storage.ByteLength - d, perHead);
                    }
                }
            }
            read = (int)blockBytes;
            return true;
        }

        private bool CopyMamba2StateOut(int layer, Span<byte> destination, out int written)
        {
            written = 0;
            float[] conv = _convState[layer];
            float[] ssm = _ssmState[layer];
            int convBytes = conv.Length * sizeof(float);
            int ssmBytes = ssm.Length * sizeof(float);
            long total = (long)convBytes + ssmBytes;
            if (destination.Length < total) return false;

            MemoryMarshal.AsBytes(conv.AsSpan()).CopyTo(destination[..convBytes]);
            MemoryMarshal.AsBytes(ssm.AsSpan()).CopyTo(destination.Slice(convBytes, ssmBytes));
            written = (int)total;
            return true;
        }

        private bool CopyMamba2StateIn(int layer, ReadOnlySpan<byte> source, out int read)
        {
            read = 0;
            float[] conv = _convState[layer];
            float[] ssm = _ssmState[layer];
            int convBytes = conv.Length * sizeof(float);
            int ssmBytes = ssm.Length * sizeof(float);
            long total = (long)convBytes + ssmBytes;
            if (source.Length < total) return false;

            source[..convBytes].CopyTo(MemoryMarshal.AsBytes(conv.AsSpan()));
            source.Slice(convBytes, ssmBytes).CopyTo(MemoryMarshal.AsBytes(ssm.AsSpan()));
            read = (int)total;
            return true;
        }

        private static void DisposeTensorArray(Tensor[] tensors)
        {
            if (tensors == null)
                return;
            for (int i = 0; i < tensors.Length; i++)
            {
                tensors[i]?.Dispose();
                tensors[i] = null;
            }
        }

        #region CPU-optimized helpers for decode path

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector<float> LdV(float* p) =>
            TensorComputePrimitives.LoadVector(p);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void StV(float* p, Vector<float> v) =>
            TensorComputePrimitives.StoreVector(p, v);

        private unsafe Tensor RMSNormCPU(Tensor input, string weightName)
        {
            long t0 = Stopwatch.GetTimestamp();
            var alpha = _weights[weightName];
            int rows = (int)input.Sizes[0];
            int dim = (int)(input.ElementCount() / rows);

            var result = new Tensor(_allocator, DType.Float32, rows, dim);
            float* srcPtr = GetFloatPtr(input);
            float* dstPtr = GetFloatPtr(result);
            float* alphaPtr = GetFloatPtr(alpha);
            float eps = Config.Eps;

            for (int r = 0; r < rows; r++)
            {
                float* src = srcPtr + (long)r * dim;
                float* dst = dstPtr + (long)r * dim;
                float sumSq = VecSumSq(src, dim);
                float rmsInv = 1.0f / MathF.Sqrt(sumSq / dim + eps);

                int vLen = Vector<float>.Count;
                var vInv = new Vector<float>(rmsInv);
                int i = 0;
                for (; i <= dim - vLen * 2; i += vLen * 2)
                {
                    StV(dst + i, LdV(src + i) * LdV(alphaPtr + i) * vInv);
                    StV(dst + i + vLen, LdV(src + i + vLen) * LdV(alphaPtr + i + vLen) * vInv);
                }
                for (; i <= dim - vLen; i += vLen)
                    StV(dst + i, LdV(src + i) * LdV(alphaPtr + i) * vInv);
                for (; i < dim; i++)
                    dst[i] = src[i] * alphaPtr[i] * rmsInv;
            }

            _normTicks += Stopwatch.GetTimestamp() - t0;
            return result;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private unsafe void AddResidualCPU(Tensor target, Tensor residual)
        {
            float* tPtr = GetFloatPtr(target);
            float* rPtr = GetFloatPtr(residual);
            int n = (int)target.ElementCount();
            VecScaleAdd(tPtr, rPtr, 1.0f, n);
            InvalidateTensorDeviceCache(target);
        }

        /// <summary>
        /// Smart LinearForward: routes to CPU for small matmuls during decode, GPU for large ones.
        /// This avoids the ~1ms+ Metal GPU dispatch overhead for operations where CPU is faster.
        /// </summary>
        private unsafe Tensor LinearForwardAuto(Tensor input, string weightName)
        {
            if (!_quantWeights.TryGetValue(weightName, out var qw))
                return LinearForward(input, weightName);

            int seqLen = (int)input.Sizes[0];
            long compute = seqLen * qw.Ne0 * qw.Ne1;

            if (seqLen <= 1 && compute < CPU_MATMUL_THRESHOLD && IsGgmlBackend)
            {
                long t0 = Stopwatch.GetTimestamp();
                int outDim = (int)qw.Ne1;
                var result = new Tensor(_allocator, DType.Float32, seqLen, outDim);
                AddmmQuantManaged(result, input, qw);
                _linearTicks += Stopwatch.GetTimestamp() - t0;
                return result;
            }

            return LinearForward(input, weightName);
        }

        /// <summary>
        /// Force CPU path for a known QuantizedWeight. Avoids dictionary lookup overhead.
        /// </summary>
        private unsafe Tensor LinearForwardCPUDirect(Tensor input, QuantizedWeight qw)
        {
            long t0 = Stopwatch.GetTimestamp();
            int seqLen = (int)input.Sizes[0];
            int outDim = (int)qw.Ne1;
            var result = new Tensor(_allocator, DType.Float32, seqLen, outDim);
            AddmmQuantManaged(result, input, qw);
            _linearTicks += Stopwatch.GetTimestamp() - t0;
            return result;
        }

        /// <summary>
        /// Performs a linear (matmul) operation writing into an existing pre-allocated result tensor.
        /// Avoids per-call tensor allocation overhead in hot loops.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private void LinearForwardInto(Tensor result, Tensor input, string weightName)
        {
            long t0 = Stopwatch.GetTimestamp();
            if (_quantWeights.TryGetValue(weightName, out var qw))
            {
                if (IsGgmlBackend)
                    GgmlBasicOps.AddmmQuant(result, input, qw.CacheKey, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes);
                else
                    AddmmQuantManaged(result, input, qw);
            }
            else if (_weights.TryGetValue(weightName, out var w))
            {
                using var wT = w.Transpose();
                Ops.Addmm(result, 0, result, 1.0f, input, wT);
            }
            _linearTicks += Stopwatch.GetTimestamp() - t0;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool CanLinearAddInto(Tensor residual, Tensor input, string weightName)
        {
            return IsGgmlBackend
                && !DisableFusedLinearResidual
                && residual != null
                && input != null
                && residual.DimensionCount == 2
                && input.DimensionCount == 2
                && residual.Sizes[0] == input.Sizes[0]
                && (input.Sizes[0] == 1 || EnableFusedLinearResidualPrefill)
                && _quantWeights.TryGetValue(weightName, out var qw)
                && input.Sizes[1] == qw.Ne0
                && residual.Sizes[1] == qw.Ne1;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private bool TryLinearAddInto(Tensor residual, Tensor input, string weightName)
        {
            if (!CanLinearAddInto(residual, input, weightName))
                return false;

            var qw = _quantWeights[weightName];
            long t0 = Stopwatch.GetTimestamp();
            GgmlBasicOps.FusedMatMulQuantAdd(residual, input,
                qw.CacheKey, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes);
            _linearTicks += Stopwatch.GetTimestamp() - t0;
            return true;
        }

        #endregion

        // Chunk size for ForwardRefill: long prompts are processed in this-many-token
        // chunks so the per-layer attention-score allocation stays bounded.
        // Override with TS_PREFILL_CHUNK when tuning.
        private static int ResolvePrefillChunkSize()
        {
            string env = Environment.GetEnvironmentVariable("TS_PREFILL_CHUNK");
            if (!string.IsNullOrEmpty(env) && int.TryParse(env, out int v) && v > 0)
                return v;
            return 2048;
        }

        public override float[] ForwardRefill(int[] tokens)
        {
            if (tokens == null || tokens.Length <= 1)
                return Forward(tokens);

            // Multimodal embeddings carry absolute insert positions within the
            // current Forward call's hidden tensor, so chunked prefill would
            // need to remap them per-chunk. Skip chunking when any are pending
            // and let the single-call path handle injection.
            bool hasMultimodal = _pendingVisionEmbeddings.Count > 0 || _pendingAudioEmbeddings.Count > 0;
            int chunkSize = ResolvePrefillChunkSize();
            int lastIdx = tokens.Length - 1;

            if (hasMultimodal || tokens.Length <= chunkSize)
                return Forward(tokens);

            for (int pos = 0; pos < lastIdx; pos += chunkSize)
            {
                int chunkLen = Math.Min(chunkSize, lastIdx - pos);
                var chunk = new int[chunkLen];
                Array.Copy(tokens, pos, chunk, 0, chunkLen);
                PrefillWithoutLogits(chunk);
            }
            return Forward(new[] { tokens[lastIdx] });
        }

        private void PrefillWithoutLogits(int[] tokens)
        {
            if (tokens == null || tokens.Length == 0)
                return;

            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            bool isDecode = false;

            EnsureCacheCapacity(startPos + seqLen);

            long t1 = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t1;

            int evalEveryN = NemotronModelEvalConfig.MlxEvalEveryNLayers;
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                switch (_layerTypes[layer])
                {
                    case LayerType.Mamba2:
                        hidden = Mamba2Block(hidden, layer, seqLen, isDecode);
                        break;
                    case LayerType.Attention:
                        hidden = AttentionBlock(hidden, layer, seqLen, startPos, isDecode);
                        break;
                    case LayerType.FFN:
                        hidden = FFNBlock(hidden, layer, seqLen, isDecode);
                        break;
                }
                if (_backend == BackendType.Mlx && (layer + 1) % evalEveryN == 0
                    && layer + 1 != Config.NumLayers && hidden != null)
                {
                    MlxFusedOps.TryAsyncEvaluate(hidden);
                }
            }

            hidden.Dispose();
            _cacheSeqLen += seqLen;
            _forwardSw.Stop();
        }

        public override float[] Forward(int[] tokens)
        {
            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            bool isDecode = seqLen == 1;

            EnsureCacheCapacity(startPos + seqLen);

            long t1 = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t1;

            if (_pendingVisionEmbeddings.Count > 0 || _pendingAudioEmbeddings.Count > 0)
            {
                foreach (var (emb, pos) in _pendingVisionEmbeddings)
                {
                    InjectMultimodalEmbeddings(hidden, emb, pos);
                    emb.Dispose();
                }
                _pendingVisionEmbeddings.Clear();
                foreach (var (emb, pos) in _pendingAudioEmbeddings)
                {
                    InjectMultimodalEmbeddings(hidden, emb, pos);
                    emb.Dispose();
                }
                _pendingAudioEmbeddings.Clear();
            }

            int evalEveryN = NemotronModelEvalConfig.MlxEvalEveryNLayers;
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                switch (_layerTypes[layer])
                {
                    case LayerType.Mamba2:
                        hidden = Mamba2Block(hidden, layer, seqLen, isDecode);
                        break;
                    case LayerType.Attention:
                        hidden = AttentionBlock(hidden, layer, seqLen, startPos, isDecode);
                        break;
                    case LayerType.FFN:
                        hidden = FFNBlock(hidden, layer, seqLen, isDecode);
                        break;
                }
                if (_backend == BackendType.Mlx && (layer + 1) % evalEveryN == 0
                    && layer + 1 != Config.NumLayers && hidden != null)
                {
                    MlxFusedOps.TryAsyncEvaluate(hidden);
                }
            }

            Tensor normed = RMSNormOp(hidden, "output_norm.weight");
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
            Tensor logitsTensor = LinearForward(lastHidden, "output.weight");
            if (logitsTensor == null)
                logitsTensor = LinearForward(lastHidden, "token_embd.weight");
            _lmHeadTicks += Stopwatch.GetTimestamp() - t2;
            lastHidden.Dispose();

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

        #region Attention Block (no RoPE)

        private Tensor AttentionBlock(Tensor hidden, int layer, int seqLen, int startPos, bool isDecode)
        {
            string prefix = _layerPrefixes[layer];

            Tensor normed = RMSNormOp(hidden, prefix + "attn_norm.weight");
            Tensor attnOut = AttentionForward(normed, layer, prefix, seqLen, startPos, hidden);
            normed.Dispose();

            if (attnOut != null)
            {
                Ops.Add(hidden, hidden, attnOut);
                attnOut.Dispose();
            }
            return hidden;
        }

        private unsafe Tensor AttentionForward(Tensor input, int layer, string prefix, int seqLen, int startPos, Tensor residual = null)
        {
            long t0 = Stopwatch.GetTimestamp();

            int headDim = Config.HeadDim;
            int numHeads = _layerNumHeads[layer];
            int numKVHeads = _layerNumKVHeads[layer];
            int qDim = numHeads * headDim;
            int kvDim = numKVHeads * headDim;
            int totalSeqLen = startPos + seqLen;

            float scale = _attentionScale != 0 ? (float)_attentionScale : 1.0f / MathF.Sqrt(headDim);

            Tensor qTensor, kTensor, vTensor;
            Tensor qkvFused = LinearForward(input, prefix + "attn_qkv.weight");
            if (qkvFused != null)
            {
                if (seqLen == 1)
                {
                    qTensor = qkvFused.Narrow(1, 0, qDim);
                    kTensor = qkvFused.Narrow(1, qDim, kvDim);
                    vTensor = qkvFused.Narrow(1, qDim + kvDim, kvDim);
                    qkvFused.Dispose();
                }
                else
                {
                    using (var qView = qkvFused.Narrow(1, 0, qDim))
                        qTensor = Ops.NewContiguous(qView);
                    using (var kView = qkvFused.Narrow(1, qDim, kvDim))
                        kTensor = Ops.NewContiguous(kView);
                    using (var vView = qkvFused.Narrow(1, qDim + kvDim, kvDim))
                        vTensor = Ops.NewContiguous(vView);
                    qkvFused.Dispose();
                }
            }
            else
            {
                qTensor = LinearForward(input, prefix + "attn_q.weight");
                kTensor = LinearForward(input, prefix + "attn_k.weight");
                vTensor = LinearForward(input, prefix + "attn_v.weight");
            }

            if (seqLen == 1)
            {
                CopyToCacheDecode(_kvCacheK[layer], kTensor, _kvCacheV[layer], vTensor,
                    numKVHeads, headDim, startPos);
                kTensor.Dispose();
                vTensor.Dispose();

                var attnResult = new Tensor(_allocator, DType.Float32, 1, numHeads * headDim);

                // MLX path: device-side attention via mlx_fast_sdpa. Avoids the
                // per-layer device→host KV cache download that
                // AttentionDecodePureCS triggers.
                bool attnOk = false;
                if (_backend == BackendType.Mlx)
                {
                    attnOk = MlxFusedOps.TryDecodeAttention(
                        attnResult, qTensor, _kvCacheK[layer], _kvCacheV[layer],
                        numHeads, numKVHeads, headDim,
                        0, totalSeqLen, _kvCacheCapacity, false, scale);
                }
                if (!attnOk)
                {
                    AttentionDecodePureCS(qTensor, _kvCacheK[layer], _kvCacheV[layer],
                        attnResult, numHeads, numKVHeads, headDim, totalSeqLen, scale);
                }
                qTensor.Dispose();

                _attnTicks += Stopwatch.GetTimestamp() - t0;

                if (TryLinearAddInto(residual, attnResult, prefix + "attn_output.weight"))
                {
                    attnResult.Dispose();
                    return null;
                }

                Tensor decodeOut = LinearForward(attnResult, prefix + "attn_output.weight");
                attnResult.Dispose();
                return decodeOut;
            }

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
            Ops.AddmmBatch(scores, 0, scores, scale, qHeads, kT);
            qHeads.Dispose();
            kExpanded.Dispose();

            // Fused causal-mask + softmax on GPU. Replaces AddCausalMask + Softmax
            // (two separate ops) with one Metal kernel.
            if (IsGgmlBackend)
            {
                GgmlBasicOps.AttentionSoftmaxWithSinks(
                    scores, sinks: null,
                    numHeads: numHeads, seqLen: seqLen, kvLen: totalSeqLen,
                    maskStartPos: startPos, slidingWindow: 0, scale: 1.0f);
            }
            else
            {
                Ops.AddCausalMask(scores, seqLen, startPos, float.NegativeInfinity);
                Ops.Softmax(scores, scores);
            }

            var attnOut = new Tensor(_allocator, DType.Float32, numHeads, seqLen, headDim);
            Ops.AddmmBatch(attnOut, 0, attnOut, 1.0f, scores, vExpanded);
            scores.Dispose();
            vExpanded.Dispose();

            Tensor flatOutput = ReshapeFromHeads(attnOut, numHeads, seqLen, headDim);
            attnOut.Dispose();

            _attnTicks += Stopwatch.GetTimestamp() - t0;

            if (TryLinearAddInto(residual, flatOutput, prefix + "attn_output.weight"))
            {
                flatOutput.Dispose();
                return null;
            }

            Tensor output = LinearForward(flatOutput, prefix + "attn_output.weight");
            flatOutput.Dispose();
            return output;
        }

        #endregion

        #region FFN Block (ReLU-squared)

        private Tensor FFNBlock(Tensor hidden, int layer, int seqLen, bool isDecode)
        {
            string prefix = _layerPrefixes[layer];

            Tensor normed = RMSNormOp(hidden, prefix + "attn_norm.weight");

            Tensor ffnOut;
            if (_numExperts > 0)
                ffnOut = MoEForward(normed, layer, prefix, seqLen, isDecode, hidden);
            else
                ffnOut = DenseFFNForward(normed, prefix, seqLen, hidden);
            normed.Dispose();

            if (ffnOut != null)
            {
                Ops.Add(hidden, hidden, ffnOut);
                ffnOut.Dispose();
            }
            return hidden;
        }

        private Tensor DenseFFNForward(Tensor input, string prefix, int seqLen, Tensor residual = null)
        {
            Tensor up = LinearForward(input, prefix + "ffn_up.weight");
            ReluSquaredInPlace(up);
            if (TryLinearAddInto(residual, up, prefix + "ffn_down.weight"))
            {
                up.Dispose();
                return null;
            }

            Tensor down = LinearForward(up, prefix + "ffn_down.weight");
            up.Dispose();
            return down;
        }

        private unsafe Tensor MoEForward(Tensor input, int layer, string prefix, int seqLen, bool isDecode, Tensor residual = null)
        {
            int hiddenSize = Config.HiddenSize;
            ref var moeInfo = ref _moeLayerInfo[layer];
            bool directDecodeResidual = CanMoEDecodeResidualAdd(input, residual, prefix, ref moeInfo, isDecode);

            Tensor routerLogits = LinearForward(input, prefix + "ffn_gate_inp.weight");
            float* routerPtr = GetFloatPtr(routerLogits);

            Tensor output = directDecodeResidual ? null : new Tensor(_allocator, DType.Float32, seqLen, hiddenSize);
            float* outputPtr = directDecodeResidual ? null : GetFloatPtr(output);
            if (!directDecodeResidual)
                VecZero(outputPtr, seqLen * hiddenSize);

            float* biasPtr = null;
            string biasName = prefix + "exp_probs_b.bias";
            if (!_weights.TryGetValue(biasName, out var biasTensor))
                _weights.TryGetValue(prefix + "exp_probs_b", out biasTensor);
            if (biasTensor != null)
                biasPtr = GetFloatPtr(biasTensor);

            // For prefill, batch shared operations across all tokens
            Tensor latentAllTokens = null;
            Tensor sharedDownAll = null;

            if (!isDecode && seqLen > 1)
            {
                if (moeInfo.HasLatentIn)
                    latentAllTokens = LinearForward(input, prefix + "ffn_latent_in.weight");

                if (moeInfo.HasSharedExperts)
                {
                    Tensor sharedUpAll = LinearForward(input, prefix + "ffn_up_shexp.weight");
                    ReluSquaredInPlace(sharedUpAll);
                    sharedDownAll = LinearForward(sharedUpAll, prefix + "ffn_down_shexp.weight");
                    sharedUpAll.Dispose();
                }
            }

            // Pre-allocate latent accumulator for batched latent_out during prefill
            int latentDim = moeInfo.LatentDim;
            Tensor latentOutAllTokens = null;
            float* latentOutAllPtr = null;
            if (!isDecode && seqLen > 1 && moeInfo.HasLatentIn && latentDim > 0)
            {
                latentOutAllTokens = new Tensor(_allocator, DType.Float32, seqLen, latentDim);
                latentOutAllPtr = GetFloatPtr(latentOutAllTokens);
                VecZero(latentOutAllPtr, seqLen * latentDim);
            }

            float[] probs = _moeProbs;
            float[] selectionProbs = _moeSelectionProbs;
            int[] topExperts = _moeTopExperts;
            float[] routeW = _moeRouteW;

            if (!isDecode && seqLen > 1)
            {
                Tensor fusedInput = moeInfo.HasLatentIn ? latentAllTokens : input;
                Tensor fusedOutput = moeInfo.HasLatentIn ? latentOutAllTokens : output;
                int fusedDim = moeInfo.HasLatentIn ? latentDim : hiddenSize;

                bool usedBatchedPrefill = false;
                if (EnableFusedMoEPrefill && fusedInput != null && fusedOutput != null)
                    usedBatchedPrefill = TryMoEPrefillFusedReluSquared(
                        fusedInput, fusedOutput, routerPtr, biasPtr, layer, seqLen, fusedDim);

                if (!usedBatchedPrefill && fusedInput != null && fusedOutput != null)
                    usedBatchedPrefill =
                        TryMoEPrefillBatchedByExpert(fusedInput, fusedOutput, routerPtr, biasPtr, layer, seqLen, fusedDim);

                if (usedBatchedPrefill)
                {
                    if (moeInfo.HasLatentIn)
                    {
                        Tensor projected = LinearForward(latentOutAllTokens, prefix + "ffn_latent_out.weight");
                        float* projPtr = GetFloatPtr(projected);
                        outputPtr = GetFloatPtr(output);
                        for (int s = 0; s < seqLen; s++)
                            VecScaleAdd(outputPtr + (long)s * hiddenSize, projPtr + (long)s * hiddenSize, 1.0f, hiddenSize);
                        projected.Dispose();
                    }

                    if (sharedDownAll != null)
                    {
                        outputPtr = GetFloatPtr(output);
                        float* sharedAllPtr = GetFloatPtr(sharedDownAll);
                        for (int s = 0; s < seqLen; s++)
                            VecScaleAdd(outputPtr + (long)s * hiddenSize, sharedAllPtr + (long)s * hiddenSize, 1.0f, hiddenSize);
                    }

                    latentOutAllTokens?.Dispose();
                    latentAllTokens?.Dispose();
                    sharedDownAll?.Dispose();
                    routerLogits.Dispose();

                    InvalidateTensorDeviceCache(output);
                    return output;
                }

                outputPtr = GetFloatPtr(output);
                VecZero(outputPtr, seqLen * hiddenSize);
                if (latentOutAllPtr != null)
                    VecZero(latentOutAllPtr, seqLen * latentDim);
            }

            for (int s = 0; s < seqLen; s++)
            {
                float* logitsRow = routerPtr + s * _numExperts;

                for (int e = 0; e < _numExperts; e++)
                    probs[e] = SigmoidScalar(logitsRow[e]);

                if (biasPtr != null)
                {
                    for (int e = 0; e < _numExperts; e++)
                        selectionProbs[e] = probs[e] + biasPtr[e];
                }
                else
                {
                    Array.Copy(probs, 0, selectionProbs, 0, _numExperts);
                }

                SelectTopKInPlace(selectionProbs, _numExperts, _numExpertsUsed, topExperts);

                for (int k = 0; k < _numExpertsUsed; k++)
                    routeW[k] = probs[topExperts[k]];

                if (_expertWeightsNorm)
                {
                    float wSum = 0;
                    for (int k = 0; k < _numExpertsUsed; k++) wSum += routeW[k];
                    if (wSum < 6.103515625e-5f) wSum = 6.103515625e-5f;
                    for (int k = 0; k < _numExpertsUsed; k++) routeW[k] /= wSum;
                }

                if (_expertWeightsScale != 1.0f)
                {
                    for (int k = 0; k < _numExpertsUsed; k++)
                        routeW[k] *= _expertWeightsScale;
                }

                // Get the input for expert computation
                Tensor routedInput;
                bool disposeRouted;

                if (moeInfo.HasLatentIn)
                {
                    if (!isDecode && latentAllTokens != null)
                    {
                        // Prefill: extract pre-computed latent row
                        using var rowView = latentAllTokens.Narrow(0, s, 1);
                        routedInput = Ops.NewContiguous(rowView);
                    }
                    else
                    {
                        routedInput = LinearForward(input, prefix + "ffn_latent_in.weight");
                    }
                    disposeRouted = true;
                }
                else
                {
                    if (seqLen > 1)
                    {
                        using var rowView = input.Narrow(0, s, 1);
                        routedInput = Ops.NewContiguous(rowView);
                        disposeRouted = true;
                    }
                    else
                    {
                        routedInput = input;
                        disposeRouted = false;
                    }
                }

                int outDim = moeInfo.HasLatentIn ? latentDim : hiddenSize;
                float* outRow = directDecodeResidual ? null : outputPtr + (long)s * hiddenSize;

                // Accumulate expert outputs in latent space using pre-allocated buffer
                fixed (float* latentAccum = _moeLatentAccum)
                {
                    VecZero(latentAccum, outDim);

                    // Batched GPU path: all experts in a single GGML graph
                    bool usedBatchedMoE = false;
                    if (isDecode && IsGgmlBackend && _expertUpQW[layer] != null && _expertDownQW[layer] != null)
                    {
                        var upQw0 = _expertUpQW[layer][topExperts[0]];
                        var dnQw0 = _expertDownQW[layer][topExperts[0]];
                        if (upQw0 != null && dnQw0 != null)
                        {
                            for (int k = 0; k < _numExpertsUsed; k++)
                            {
                                int ei = topExperts[k];
                                _moeUpPtrs[k] = _expertUpQW[layer][ei].CacheKey;
                                _moeDownPtrs[k] = _expertDownQW[layer][ei].CacheKey;
                            }

                            long t0exp = Stopwatch.GetTimestamp();

                            // Use _expertDownResult as the output accumulator
                            GgmlBasicOps.MoEExpertsForward(
                                _expertDownResult, routedInput,
                                _numExpertsUsed, _moeUpPtrs, _moeDownPtrs,
                                upQw0.GgmlType, upQw0.Ne0, upQw0.Ne1, upQw0.RawBytes,
                                dnQw0.GgmlType, dnQw0.Ne0, dnQw0.Ne1, dnQw0.RawBytes,
                                routeW);

                            _linearTicks += Stopwatch.GetTimestamp() - t0exp;

                            float* expPtr = GetFloatPtr(_expertDownResult);
                            Buffer.MemoryCopy(expPtr, latentAccum, outDim * 4, outDim * 4);
                            usedBatchedMoE = true;
                        }
                    }

                    // Batched GPU path on MLX: 4 dispatches per layer instead of
                    // K × 3 + K syncs through the per-expert fallback. Profiler
                    // shows MoE matmul is ~76% of MLX decode token time on
                    // Nemotron-H 30B with K=6 / 23 MoE layers, so collapsing
                    // these per-expert dispatches into 4 batched ones is the
                    // single biggest decode-perf win on MLX.
                    if (!usedBatchedMoE && isDecode && _backend == BackendType.Mlx)
                    {
                        long t0exp = Stopwatch.GetTimestamp();
                        usedBatchedMoE = TryRunMoEExpertsBatchedMlx(
                            latentAccum, routedInput, layer, topExperts, routeW, outDim);
                        _linearTicks += Stopwatch.GetTimestamp() - t0exp;
                    }

                    if (!usedBatchedMoE)
                    {
                        for (int k = 0; k < _numExpertsUsed; k++)
                        {
                            int expertIdx = topExperts[k];

                            Tensor up = LinearForward(routedInput, _expertUpKeys[layer][expertIdx]);
                            ReluSquaredInPlace(up);
                            Tensor expertOut = LinearForward(up, _expertDownKeys[layer][expertIdx]);
                            up.Dispose();

                            float w = routeW[k];
                            float* expPtr = GetFloatPtr(expertOut);
                            VecScaleAdd(latentAccum, expPtr, w, outDim);
                            expertOut.Dispose();
                        }
                    }

                    if (moeInfo.HasLatentIn)
                    {
                        if (!isDecode && latentOutAllPtr != null)
                        {
                            Buffer.MemoryCopy(latentAccum, latentOutAllPtr + (long)s * latentDim,
                                latentDim * 4, latentDim * 4);
                        }
                        else if (directDecodeResidual && _latentAccumTensor != null)
                        {
                            float* ltPtr = GetFloatPtr(_latentAccumTensor);
                            Buffer.MemoryCopy(latentAccum, ltPtr, latentDim * 4, latentDim * 4);
                            InvalidateTensorDeviceCache(_latentAccumTensor);
                            if (!TryLinearAddInto(residual, _latentAccumTensor, prefix + "ffn_latent_out.weight"))
                                throw new InvalidOperationException("Nemotron MoE decode residual fast path failed for latent_out projection.");
                        }
                        else if (_latentAccumTensor != null && _latentOutResult != null)
                        {
                            float* ltPtr = GetFloatPtr(_latentAccumTensor);
                            Buffer.MemoryCopy(latentAccum, ltPtr, latentDim * 4, latentDim * 4);
                            InvalidateTensorDeviceCache(_latentAccumTensor);
                            LinearForwardInto(_latentOutResult, _latentAccumTensor, prefix + "ffn_latent_out.weight");

                            float* projPtr = GetFloatPtr(_latentOutResult);
                            VecScaleAdd(outRow, projPtr, 1.0f, hiddenSize);
                        }
                        else
                        {
                            var latentTensor = new Tensor(_allocator, DType.Float32, 1, latentDim);
                            float* ltPtr = GetFloatPtr(latentTensor);
                            Buffer.MemoryCopy(latentAccum, ltPtr, latentDim * 4, latentDim * 4);
                            InvalidateTensorDeviceCache(latentTensor);
                            Tensor projected = LinearForward(latentTensor, prefix + "ffn_latent_out.weight");
                            latentTensor.Dispose();

                            float* projPtr = GetFloatPtr(projected);
                            VecScaleAdd(outRow, projPtr, 1.0f, hiddenSize);
                            projected.Dispose();
                        }
                    }
                    else
                    {
                        if (directDecodeResidual)
                        {
                            float* residualPtr = GetFloatPtr(residual);
                            VecScaleAdd(residualPtr, latentAccum, 1.0f, hiddenSize);
                            InvalidateTensorDeviceCache(residual);
                        }
                        else
                        {
                            for (int i = 0; i < hiddenSize; i++)
                                outRow[i] += latentAccum[i];
                        }
                    }
                }

                if (disposeRouted)
                    routedInput.Dispose();

                // Shared experts
                if (moeInfo.HasSharedExperts)
                {
                    if (!isDecode && sharedDownAll != null)
                    {
                        float* sharedAllPtr = GetFloatPtr(sharedDownAll);
                        VecScaleAdd(outRow, sharedAllPtr + (long)s * hiddenSize, 1.0f, hiddenSize);
                    }
                    else
                    {
                        Tensor sharedUp = LinearForward(input, prefix + "ffn_up_shexp.weight");
                        ReluSquaredInPlace(sharedUp);
                        if (directDecodeResidual)
                        {
                            if (!TryLinearAddInto(residual, sharedUp, prefix + "ffn_down_shexp.weight"))
                                throw new InvalidOperationException("Nemotron MoE decode residual fast path failed for shared expert down projection.");
                            sharedUp.Dispose();
                            continue;
                        }

                        Tensor sharedDown = LinearForward(sharedUp, prefix + "ffn_down_shexp.weight");
                        sharedUp.Dispose();

                        float* sharedPtr = GetFloatPtr(sharedDown);
                        VecScaleAdd(outRow, sharedPtr, 1.0f, hiddenSize);
                        sharedDown.Dispose();
                    }
                }
            }

            // Prefill: batch latent_out projection
            if (!isDecode && latentOutAllTokens != null)
            {
                InvalidateTensorDeviceCache(latentOutAllTokens);
                Tensor projected = LinearForward(latentOutAllTokens, prefix + "ffn_latent_out.weight");
                latentOutAllTokens.Dispose();

                float* projPtr = GetFloatPtr(projected);
                for (int s = 0; s < seqLen; s++)
                    VecScaleAdd(outputPtr + (long)s * hiddenSize, projPtr + (long)s * hiddenSize, 1.0f, hiddenSize);
                projected.Dispose();
            }
            else
            {
                latentOutAllTokens?.Dispose();
            }

            latentAllTokens?.Dispose();
            sharedDownAll?.Dispose();
            routerLogits.Dispose();

            if (directDecodeResidual)
                return null;

            InvalidateTensorDeviceCache(output);
            return output;
        }

        private bool CanMoEDecodeResidualAdd(
            Tensor input,
            Tensor residual,
            string prefix,
            ref MoELayerInfo moeInfo,
            bool isDecode)
        {
            if (!isDecode
                || !IsGgmlBackend
                || DisableFusedLinearResidual
                || input == null
                || residual == null
                || input.DimensionCount != 2
                || residual.DimensionCount != 2
                || input.Sizes[0] != 1
                || residual.Sizes[0] != 1
                || residual.Sizes[1] != Config.HiddenSize)
            {
                return false;
            }

            if (moeInfo.HasLatentIn)
            {
                if (_latentAccumTensor == null
                    || !_quantWeights.TryGetValue(prefix + "ffn_latent_out.weight", out var latentOutQw)
                    || _latentAccumTensor.DimensionCount != 2
                    || _latentAccumTensor.Sizes[0] != 1
                    || _latentAccumTensor.Sizes[1] != latentOutQw.Ne0
                    || latentOutQw.Ne1 != Config.HiddenSize)
                {
                    return false;
                }
            }

            if (moeInfo.HasSharedExperts)
            {
                if (!_quantWeights.TryGetValue(prefix + "ffn_up_shexp.weight", out var sharedUpQw)
                    || !_quantWeights.TryGetValue(prefix + "ffn_down_shexp.weight", out var sharedDownQw)
                    || sharedUpQw.Ne1 != sharedDownQw.Ne0
                    || sharedDownQw.Ne1 != Config.HiddenSize)
                {
                    return false;
                }
            }

            return true;
        }

        private unsafe bool TryMoEPrefillBatchedByExpert(
            Tensor routedInput,
            Tensor moeOut,
            float* routerPtr,
            float* biasPtr,
            int layer,
            int seqLen,
            int routedDim)
        {
            if (DisableBatchedMoEPrefill
                || seqLen <= 1
                || seqLen < BatchedMoEPrefillMinTokens
                || routedInput == null
                || moeOut == null
                || _expertUpKeys == null
                || _expertDownKeys == null
                || layer < 0
                || layer >= _expertUpKeys.Length
                || _expertUpKeys[layer] == null
                || _expertDownKeys[layer] == null
                || routedInput.DimensionCount != 2
                || moeOut.DimensionCount != 2
                || routedInput.Sizes[0] != seqLen
                || routedInput.Sizes[1] != routedDim
                || moeOut.Sizes[0] != seqLen
                || moeOut.Sizes[1] != routedDim)
            {
                return false;
            }

            int nUsed = _numExpertsUsed;
            int totalRoutes = checked(seqLen * nUsed);
            EnsureMoEPrefillRouteBuffers(totalRoutes);
            int[] selectedExperts = _moePrefillSelectedExperts;
            float[] routingWeights = _moePrefillRoutingWeights;
            int[] expertCounts = _moePrefillExpertCounts;
            int[] tokenTopExperts = _moeTopExperts;
            float[] probs = _moeProbs;
            float[] selectionProbs = _moeSelectionProbs;
            Array.Clear(expertCounts, 0, _numExperts);

            for (int s = 0; s < seqLen; s++)
            {
                float* logitsRow = routerPtr + (long)s * _numExperts;

                for (int e = 0; e < _numExperts; e++)
                    probs[e] = SigmoidScalar(logitsRow[e]);

                if (biasPtr != null)
                {
                    for (int e = 0; e < _numExperts; e++)
                        selectionProbs[e] = probs[e] + biasPtr[e];
                }
                else
                {
                    Array.Copy(probs, 0, selectionProbs, 0, _numExperts);
                }

                SelectTopKInPlace(selectionProbs, _numExperts, nUsed, tokenTopExperts);

                int routeOffset = s * nUsed;
                for (int k = 0; k < nUsed; k++)
                {
                    int expert = tokenTopExperts[k];
                    selectedExperts[routeOffset + k] = expert;
                    routingWeights[routeOffset + k] = probs[expert];
                    expertCounts[expert]++;
                }

                if (_expertWeightsNorm)
                {
                    float wSum = 0;
                    for (int k = 0; k < nUsed; k++)
                        wSum += routingWeights[routeOffset + k];
                    if (wSum < 6.103515625e-5f)
                        wSum = 6.103515625e-5f;
                    float inv = 1.0f / wSum;
                    for (int k = 0; k < nUsed; k++)
                        routingWeights[routeOffset + k] *= inv;
                }

                if (_expertWeightsScale != 1.0f)
                {
                    for (int k = 0; k < nUsed; k++)
                        routingWeights[routeOffset + k] *= _expertWeightsScale;
                }
            }

            int[] expertOffsets = _moePrefillExpertOffsets;
            expertOffsets[0] = 0;
            for (int e = 0; e < _numExperts; e++)
                expertOffsets[e + 1] = expertOffsets[e] + expertCounts[e];

            int[] cursors = _moePrefillExpertCursors;
            Array.Copy(expertOffsets, cursors, _numExperts);
            int[] routedRows = _moePrefillRoutedRows;
            float[] routedWeights = _moePrefillRoutedWeights;
            for (int s = 0; s < seqLen; s++)
            {
                int routeOffset = s * nUsed;
                for (int k = 0; k < nUsed; k++)
                {
                    int expert = selectedExperts[routeOffset + k];
                    int dst = cursors[expert]++;
                    routedRows[dst] = s;
                    routedWeights[dst] = routingWeights[routeOffset + k];
                }
            }

            float* inputPtr = GetFloatPtr(routedInput);
            float* outputPtr = GetFloatPtr(moeOut);
            int rowBytes = checked(routedDim * sizeof(float));

            for (int expert = 0; expert < _numExperts; expert++)
            {
                int start = expertOffsets[expert];
                int batchSize = expertOffsets[expert + 1] - start;
                if (batchSize == 0)
                    continue;

                Tensor batchInput = null;
                Tensor up = null;
                Tensor down = null;
                try
                {
                    batchInput = new Tensor(_allocator, DType.Float32, batchSize, routedDim);
                    float* batchPtr = GetFloatPtr(batchInput);
                    for (int b = 0; b < batchSize; b++)
                    {
                        int token = routedRows[start + b];
                        Buffer.MemoryCopy(
                            inputPtr + (long)token * routedDim,
                            batchPtr + (long)b * routedDim,
                            rowBytes,
                            rowBytes);
                    }
                    InvalidateTensorDeviceCache(batchInput);

                    up = LinearForward(batchInput, _expertUpKeys[layer][expert]);
                    if (up == null)
                        return false;

                    ReluSquaredInPlace(up);
                    InvalidateTensorDeviceCache(up);

                    down = LinearForward(up, _expertDownKeys[layer][expert]);
                    if (down == null)
                        return false;

                    float* downPtr = GetFloatPtr(down);
                    for (int b = 0; b < batchSize; b++)
                    {
                        int token = routedRows[start + b];
                        float weight = routedWeights[start + b];
                        VecScaleAdd(
                            outputPtr + (long)token * routedDim,
                            downPtr + (long)b * routedDim,
                            weight,
                            routedDim);
                    }
                }
                finally
                {
                    down?.Dispose();
                    up?.Dispose();
                    batchInput?.Dispose();
                }
            }

            InvalidateTensorDeviceCache(moeOut);
            return true;
        }

        // ----- MLX batched-MoE decode path -----
        //
        // Mirrors the Qwen 3.5 MLX MoE decode pattern (TryRunMoEExpertsBatchedMlx)
        // but with Nemotron-H's ReLU² activation (no gate projection / SiLU).
        // The serial per-expert path through this layer issues, per K-active
        // expert:
        //   (a) up matmul              — 1 MLX dispatch
        //   (b) ReluSquaredInPlace     — pulls the up tensor to host, runs
        //                                 SIMD on CPU, marks device-stale
        //                                 — forces a sync per expert
        //   (c) down matmul            — 1 MLX dispatch
        //   (d) VecScaleAdd into the CPU accumulator
        // K=6 experts × 3 GPU+sync touches × 23 MoE layers = ~414 round
        // trips per token. The batched path collapses all of that into 4
        // dispatches per layer (batched up · device-side ReLU² · batched
        // down · weighted sum) plus a single host download of the final
        // [1, outDim] result. Eligibility requires IQ2_XXS stacked
        // weights (the only quant type with a batched-MoE Metal kernel
        // today) and a homogeneous (intermediate, hidden) shape across
        // experts.
        //
        // `latentAccum` is the CPU float* the caller uses to accumulate
        // expert outputs (sized `outDim` = hidden for non-latent layers,
        // or latentDim for latent layers). We compute the weighted sum
        // on the GPU into _moeBatchedResult and copy it into latentAccum
        // — one sync per layer instead of K syncs.
        private unsafe bool TryRunMoEExpertsBatchedMlx(
            float* latentAccum,
            Tensor routedInput,
            int layer,
            int[] topExperts,
            float[] routeW,
            int outDim)
        {
            if (_backend != BackendType.Mlx
                || _layerStackedUp == null
                || _layerStackedDown == null
                || layer < 0
                || layer >= _layerStackedUp.Length)
            {
                return false;
            }

            var upW = _layerStackedUp[layer];
            var downW = _layerStackedDown[layer];
            if (upW == null
                || downW == null
                || upW.NumExperts != _numExperts
                || downW.NumExperts != _numExperts
                || !MLX.MlxQuantizedOps.SupportsBatchedMoeMatmul(upW.GgmlType)
                || upW.GgmlType != downW.GgmlType)
            {
                return false;
            }
            // up: [hidden, intermediate] per expert (PerExpertNe0=hidden, PerExpertNe1=intermediate).
            // down: [intermediate, hidden] per expert. They must form a
            // consistent (hidden ↔ intermediate) pair.
            int hidden = (int)upW.PerExpertNe0;
            int intermediate = (int)upW.PerExpertNe1;
            if (downW.PerExpertNe0 != intermediate || downW.PerExpertNe1 != hidden)
                return false;
            if (outDim != hidden)
                return false;
            if (routedInput == null
                || routedInput.DimensionCount != 2
                || routedInput.Sizes[0] != 1
                || routedInput.Sizes[1] != hidden)
            {
                return false;
            }

            int K = _numExpertsUsed;

            // Lazy-init / shape-change rebuild of scratch buffers.
            if (_moeBatchedUp == null
                || _moeBatchedIntermediate != intermediate
                || _moeBatchedHidden != hidden
                || _moeBatchedUp.Sizes[0] != K)
            {
                _moeBatchedUp?.Dispose();
                _moeBatchedDown?.Dispose();
                _moeBatchedExpertIndices?.Dispose();
                _moeBatchedRouteWeights?.Dispose();
                _moeBatchedResult?.Dispose();
                _moeBatchedUp = new Tensor(_allocator, DType.Float32, K, intermediate);
                _moeBatchedDown = new Tensor(_allocator, DType.Float32, K, hidden);
                _moeBatchedExpertIndices = new Tensor(_allocator, DType.Int32, K);
                _moeBatchedRouteWeights = new Tensor(_allocator, DType.Float32, 1, K);
                _moeBatchedResult = new Tensor(_allocator, DType.Float32, 1, hidden);
                _moeBatchedIntermediate = intermediate;
                _moeBatchedHidden = hidden;
            }

            // Upload top-K expert indices + routing weights to device. Tiny
            // (K ints + K floats), negligible vs the kernel-launch budget we
            // are saving.
            _moeBatchedExpertIndices.SetElementsAsInt(topExperts);
            _moeBatchedRouteWeights.SetElementsAsFloat(routeW);

            try
            {
                // 1. Batched up matmul: [1, hidden] × stacked_up → [K, intermediate].
                //    sharedInput=true because every expert applies its own
                //    weights to the SAME single decode token row.
                if (!MLX.MlxQuantizedOps.TryMoeMatmulBatched(
                        _moeBatchedUp,
                        routedInput,
                        _moeBatchedExpertIndices,
                        upW.Data, upW.Data, upW.GgmlType,
                        upW.PerExpertNe0, upW.PerExpertNe1, upW.NumExperts, upW.TotalRawBytes,
                        sharedInput: true))
                {
                    return false;
                }

                // 2. ReLU² on the [K, intermediate] activations, ON DEVICE.
                //    ReluSquared(x) = max(x, 0)² — squaring alone would
                //    flip the sign of negative inputs, so we relu first and
                //    then square. Two MLX dispatches (relu + mul) replace
                //    the K host-side SIMD loops in the per-expert path.
                Ops.Relu(_moeBatchedUp, _moeBatchedUp);
                Ops.Mul(_moeBatchedUp, _moeBatchedUp, _moeBatchedUp);

                // 3. Batched down matmul: [K, intermediate] × stacked_down → [K, hidden].
                //    sharedInput=false because each row k of the input is
                //    expert-k's post-activation output.
                if (!MLX.MlxQuantizedOps.TryMoeMatmulBatched(
                        _moeBatchedDown,
                        _moeBatchedUp,
                        _moeBatchedExpertIndices,
                        downW.Data, downW.Data, downW.GgmlType,
                        downW.PerExpertNe0, downW.PerExpertNe1, downW.NumExperts, downW.TotalRawBytes,
                        sharedInput: false))
                {
                    return false;
                }

                // 4. Weighted sum: routeW[1, K] @ down[K, hidden] → [1, hidden],
                //    written into _moeBatchedResult. We use Addmm with the
                //    output's beta=0 so the destination's prior contents are
                //    ignored (it's a scratch buffer; we own it).
                Ops.Addmm(_moeBatchedResult, 0.0f, _moeBatchedResult,
                          1.0f, _moeBatchedRouteWeights, _moeBatchedDown);

                // 5. Single host download of the [1, hidden] result into the
                //    caller's CPU accumulator. With the per-expert path this
                //    same data movement happened K times — and was
                //    interleaved with the actual GPU work.
                float* resultPtr = GetFloatPtr(_moeBatchedResult);
                Buffer.MemoryCopy(resultPtr, latentAccum, outDim * 4, outDim * 4);
                return true;
            }
            catch
            {
                return false;
            }
        }

        private unsafe bool TryMoEPrefillFusedReluSquared(
            Tensor routedInput,
            Tensor moeOut,
            float* routerPtr,
            float* biasPtr,
            int layer,
            int seqLen,
            int routedDim)
        {
            if (!IsGgmlBackend
                || !EnableFusedMoEPrefill
                || seqLen <= 1
                || routedInput == null
                || moeOut == null
                || _layerStackedUp == null
                || _layerStackedDown == null
                || layer < 0
                || layer >= _layerStackedUp.Length)
            {
                return false;
            }

            var upW = _layerStackedUp[layer];
            var downW = _layerStackedDown[layer];
            if (upW == null
                || downW == null
                || upW.NumExperts != _numExperts
                || downW.NumExperts != _numExperts
                || upW.PerExpertNe0 != routedDim
                || downW.PerExpertNe1 != routedDim
                || upW.PerExpertNe1 != downW.PerExpertNe0
                || routedInput.DimensionCount != 2
                || moeOut.DimensionCount != 2
                || routedInput.Sizes[0] != seqLen
                || routedInput.Sizes[1] != routedDim
                || moeOut.Sizes[0] != seqLen
                || moeOut.Sizes[1] != routedDim)
            {
                return false;
            }

            int nUsed = _numExpertsUsed;
            int nFf = checked((int)upW.PerExpertNe1);
            int totalRoutes = checked(seqLen * nUsed);
            EnsureMoEPrefillRouteBuffers(totalRoutes);
            int[] selectedExperts = _moePrefillSelectedExperts;
            float[] routingWeights = _moePrefillRoutingWeights;
            int[] tokenTopExperts = _moeTopExperts;
            float[] probs = _moeProbs;
            float[] selectionProbs = _moeSelectionProbs;

            for (int s = 0; s < seqLen; s++)
            {
                float* logitsRow = routerPtr + (long)s * _numExperts;

                for (int e = 0; e < _numExperts; e++)
                    probs[e] = SigmoidScalar(logitsRow[e]);

                if (biasPtr != null)
                {
                    for (int e = 0; e < _numExperts; e++)
                        selectionProbs[e] = probs[e] + biasPtr[e];
                }
                else
                {
                    Array.Copy(probs, 0, selectionProbs, 0, _numExperts);
                }

                SelectTopKInPlace(selectionProbs, _numExperts, nUsed, tokenTopExperts);

                int routeOffset = s * nUsed;
                for (int k = 0; k < nUsed; k++)
                {
                    int expert = tokenTopExperts[k];
                    selectedExperts[routeOffset + k] = expert;
                    routingWeights[routeOffset + k] = probs[expert];
                }

                if (_expertWeightsNorm)
                {
                    float wSum = 0;
                    for (int k = 0; k < nUsed; k++)
                        wSum += routingWeights[routeOffset + k];
                    if (wSum < 6.103515625e-5f)
                        wSum = 6.103515625e-5f;
                    float inv = 1.0f / wSum;
                    for (int k = 0; k < nUsed; k++)
                        routingWeights[routeOffset + k] *= inv;
                }

                if (_expertWeightsScale != 1.0f)
                {
                    for (int k = 0; k < nUsed; k++)
                        routingWeights[routeOffset + k] *= _expertWeightsScale;
                }
            }

            try
            {
                long t0exp = Stopwatch.GetTimestamp();
                GgmlBasicOps.MoEFFNPrefill(
                    routedInput,
                    moeOut,
                    seqLen,
                    routedDim,
                    nFf,
                    _numExperts,
                    nUsed,
                    selectedExperts,
                    routingWeights,
                    upW.Data,
                    upW.GgmlType,
                    upW.PerExpertNe0,
                    upW.PerExpertNe1,
                    upW.TotalRawBytes,
                    IntPtr.Zero,
                    0,
                    0,
                    0,
                    0,
                    downW.Data,
                    downW.GgmlType,
                    downW.PerExpertNe0,
                    downW.PerExpertNe1,
                    downW.TotalRawBytes,
                    gateBias: null,
                    upBias: null,
                    downBias: null,
                    activation: GgmlBasicOps.MoEActivation.ReluSquared);
                _linearTicks += Stopwatch.GetTimestamp() - t0exp;
                InvalidateTensorDeviceCache(moeOut);
                return true;
            }
            catch (ArgumentException)
            {
                if (_mamba2NativeDecodeStateInitialized[layer])
                    throw;
                return false;
            }
            catch (InvalidOperationException)
            {
                if (_mamba2NativeDecodeStateInitialized[layer])
                    throw;
                return false;
            }
            catch (NotSupportedException)
            {
                if (_mamba2NativeDecodeStateInitialized[layer])
                    throw;
                return false;
            }
        }

        private static void SelectTopKInPlace(float[] values, int n, int k, int[] indices) =>
            TensorComputePrimitives.SelectTopKInPlace(values, n, k, indices);

        private static unsafe void ReluSquaredInPlace(Tensor t) =>
            TensorComputePrimitives.ReluSquaredInPlace(t);

        #endregion

        #region Mamba2 Block

        private Tensor Mamba2Block(Tensor hidden, int layer, int seqLen, bool isDecode, int slot = 0)
        {
            string prefix = _layerPrefixes[layer];

            Tensor normed = RMSNormOp(hidden, prefix + "attn_norm.weight");
            Tensor mamba2Out = Mamba2Forward(normed, layer, prefix, seqLen, hidden, slot);
            normed.Dispose();

            if (mamba2Out != null)
            {
                Ops.Add(hidden, hidden, mamba2Out);
                mamba2Out.Dispose();
            }
            return hidden;
        }

        /// <summary>
        /// Mamba2 SSM forward pass with SIMD-optimized inner loops.
        /// </summary>
        /// <param name="slot">Per-active-sequence Mamba2 slot index used to key
        /// the persistent GPU decode-state cache so concurrent sequences in
        /// the batched path don't share GPU state via cache-key collision.
        /// Pass 0 for the legacy single-sequence Forward path.</param>
        private unsafe Tensor Mamba2Forward(Tensor input, int layer, string prefix, int seqLen, Tensor residual = null, int slot = 0)
        {
            long t0 = Stopwatch.GetTimestamp();

            int dConv = _ssmDConv;
            int dInner = _ssmDInner;
            int dState = _ssmDState;
            int nHead = _ssmNHead;
            int headDim = _ssmHeadDim;
            int nGroup = _ssmNGroup;
            int convDim = Math.Max(0, dConv - 1);

            int xBCSize = dInner + 2 * nGroup * dState;
            int dInProjTotal = 2 * dInner + 2 * nGroup * dState + nHead;

            if (TryMamba2NativeDecode(input, residual, layer, prefix, seqLen,
                    dInner, dState, nHead, headDim, nGroup, dConv,
                    out Tensor nativeDecodeOut, slot))
            {
                _attnTicks += Stopwatch.GetTimestamp() - t0;
                return nativeDecodeOut;
            }

            Tensor projected = LinearForward(input, prefix + "ssm_in.weight");
            Tensor result = new Tensor(_allocator, DType.Float32, seqLen, dInner);

            float[] convState = _convState[layer];
            float[] ssmState = _ssmState[layer];

            if (TryMamba2NativePrefill(projected, result, layer, prefix, seqLen,
                    dInner, dState, nHead, headDim, nGroup, dConv))
            {
                projected.Dispose();
                InvalidateTensorDeviceCache(result);

                if (TryLinearAddInto(residual, result, prefix + "ssm_out.weight"))
                {
                    result.Dispose();
                    _attnTicks += Stopwatch.GetTimestamp() - t0;
                    return null;
                }

                Tensor nativeOutProj = LinearForward(result, prefix + "ssm_out.weight");
                result.Dispose();

                _attnTicks += Stopwatch.GetTimestamp() - t0;
                return nativeOutProj;
            }

            float* projPtr = GetFloatPtr(projected);
            float* resultPtr = GetFloatPtr(result);

            float[] convWT = _mamba2ConvWT?[layer];
            float* convWPtr = convWT == null ? GetFloatPtr(_weights[prefix + "ssm_conv1d.weight"]) : null;
            float* convBiasPtr = _weights.TryGetValue(prefix + "ssm_conv1d.bias", out var cb) ? GetFloatPtr(cb) : null;
            float* dtBiasPtr = GetFloatPtr(_weights[prefix + "ssm_dt.bias"]);
            float* aPtr = GetFloatPtr(_weights[prefix + "ssm_a"]);
            float* dPtr = _weights.TryGetValue(prefix + "ssm_d", out var dTensor) ? GetFloatPtr(dTensor) : null;
            float* ssmNormPtr = _weights.TryGetValue(prefix + "ssm_norm.weight", out var normW) ? GetFloatPtr(normW) : null;

            float[] convOutBuf = _mamba2ConvOutBuf;
            float[] yBuf = _mamba2YBuf;
            float* dtBuf = stackalloc float[nHead];

            int vLen = Vector<float>.Count;

            for (int s = 0; s < seqLen; s++)
            {
                float* row = projPtr + (long)s * dInProjTotal;

                float* zPtr = row;
                float* xBCPtr = row + dInner;
                float* dtPtr = row + 2 * dInner + 2 * nGroup * dState;

                Mamba2Conv1dStep(xBCPtr, xBCSize, convState, convDim, convWT, convWPtr, convBiasPtr, convOutBuf);

                // SiLU on conv output - SIMD optimized
                TensorComputePrimitives.ApplySiLUInPlace(
                    convOutBuf.AsSpan(0, xBCSize),
                    _mamba2SiluTmpBuf.AsSpan(0, xBCSize));

                for (int h = 0; h < nHead; h++)
                    dtBuf[h] = dtPtr[h] + dtBiasPtr[h];

                // SSM scan step - SIMD optimized
                Mamba2SSMStepSIMD(convOutBuf, dtBuf, aPtr, dPtr, ssmState,
                    dInner, dState, nHead, headDim, nGroup, yBuf);

                // Swiglu: y = silu(z) * y - SIMD optimized
                fixed (float* yPtr = yBuf)
                {
                    int i = 0;
                    for (; i < dInner; i++)
                        yPtr[i] = SiLUScalar(zPtr[i]) * yPtr[i];
                }

                // Group RMSNorm
                if (ssmNormPtr != null)
                {
                    int innerPerGroup = dInner / nGroup;
                    fixed (float* yPtr = yBuf)
                    {
                        for (int g = 0; g < nGroup; g++)
                        {
                            int offset = g * innerPerGroup;
                            float sumSq = VecSumSq(yPtr + offset, innerPerGroup);
                            float rmsInv = 1.0f / MathF.Sqrt(sumSq / innerPerGroup + Config.Eps);

                            var vInv = new Vector<float>(rmsInv);
                            int i = 0;
                            for (; i <= innerPerGroup - vLen; i += vLen)
                            {
                                var vy = LdV(yPtr + offset + i);
                                var vn = LdV(ssmNormPtr + offset + i);
                                StV(yPtr + offset + i, vy * vn * vInv);
                            }
                            for (; i < innerPerGroup; i++)
                                yPtr[offset + i] = yPtr[offset + i] * rmsInv * ssmNormPtr[offset + i];
                        }
                    }
                }

                fixed (float* yPtr = yBuf)
                    Buffer.MemoryCopy(yPtr, resultPtr + (long)s * dInner, dInner * 4, dInner * 4);
            }

            projected.Dispose();

            InvalidateTensorDeviceCache(result);

            if (TryLinearAddInto(residual, result, prefix + "ssm_out.weight"))
            {
                result.Dispose();
                _attnTicks += Stopwatch.GetTimestamp() - t0;
                return null;
            }

            Tensor outProj = LinearForward(result, prefix + "ssm_out.weight");
            result.Dispose();

            _attnTicks += Stopwatch.GetTimestamp() - t0;
            return outProj;
        }

        private unsafe bool TryMamba2NativePrefill(
            Tensor projected,
            Tensor result,
            int layer,
            string prefix,
            int seqLen,
            int dInner,
            int dState,
            int nHead,
            int headDim,
            int nGroup,
            int dConv)
        {
            if (DisableNativeMamba2Prefill || !IsGgmlBackend || seqLen < NativeMamba2PrefillMinTokens)
                return false;
            if (projected.DimensionCount != 2 || result.DimensionCount != 2)
                return false;
            if (projected.ElementType != DType.Float32 || result.ElementType != DType.Float32)
                return false;
            if (!_weights.TryGetValue(prefix + "ssm_conv1d.weight", out Tensor convW)
                || !_weights.TryGetValue(prefix + "ssm_dt.bias", out Tensor dtBias)
                || !_weights.TryGetValue(prefix + "ssm_a", out Tensor a))
            {
                return false;
            }

            _weights.TryGetValue(prefix + "ssm_conv1d.bias", out Tensor convBias);
            _weights.TryGetValue(prefix + "ssm_d", out Tensor d);
            _weights.TryGetValue(prefix + "ssm_norm.weight", out Tensor normW);

            try
            {
                GgmlBasicOps.NemotronMamba2Prefill(
                    projected,
                    result,
                    _convState[layer],
                    _ssmState[layer],
                    TensorComputePrimitives.GetStoragePointer(convW),
                    convBias == null ? IntPtr.Zero : TensorComputePrimitives.GetStoragePointer(convBias),
                    TensorComputePrimitives.GetStoragePointer(dtBias),
                    TensorComputePrimitives.GetStoragePointer(a),
                    d == null ? IntPtr.Zero : TensorComputePrimitives.GetStoragePointer(d),
                    normW == null ? IntPtr.Zero : TensorComputePrimitives.GetStoragePointer(normW),
                    dInner,
                    dState,
                    nHead,
                    headDim,
                    nGroup,
                    dConv,
                    Config.Eps);
                return true;
            }
            catch (ArgumentException)
            {
                return false;
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

        private unsafe bool TryMamba2NativeDecode(
            Tensor input,
            Tensor residual,
            int layer,
            string prefix,
            int seqLen,
            int dInner,
            int dState,
            int nHead,
            int headDim,
            int nGroup,
            int dConv,
            out Tensor output,
            int slot = 0)
        {
            output = null;
            if (DisableNativeMamba2Decode
                || _backend != BackendType.GgmlMetal
                || seqLen != 1
                || _mamba2NativeDecodeProjected == null
                || _mamba2NativeDecodeHidden == null
                || _mamba2NativeDecodeStateInitialized == null)
            {
                return false;
            }

            Tensor projected = _mamba2NativeDecodeProjected[layer];
            Tensor result = _mamba2NativeDecodeHidden[layer];
            if (projected == null || result == null)
                return false;
            if (!_weights.TryGetValue(prefix + "ssm_conv1d.weight", out Tensor convW)
                || !_weights.TryGetValue(prefix + "ssm_dt.bias", out Tensor dtBias)
                || !_weights.TryGetValue(prefix + "ssm_a", out Tensor a))
            {
                return false;
            }

            _weights.TryGetValue(prefix + "ssm_conv1d.bias", out Tensor convBias);
            _weights.TryGetValue(prefix + "ssm_d", out Tensor d);
            _weights.TryGetValue(prefix + "ssm_norm.weight", out Tensor normW);

            try
            {
                LinearForwardInto(projected, input, prefix + "ssm_in.weight");

                GgmlBasicOps.NemotronMamba2Decode(
                    NativeMamba2DecodeStateKey(layer, slot),
                    projected,
                    result,
                    _convState[layer],
                    _ssmState[layer],
                    !_mamba2NativeDecodeStateInitialized[layer],
                    downloadState: false,
                    TensorComputePrimitives.GetStoragePointer(convW),
                    convBias == null ? IntPtr.Zero : TensorComputePrimitives.GetStoragePointer(convBias),
                    TensorComputePrimitives.GetStoragePointer(dtBias),
                    TensorComputePrimitives.GetStoragePointer(a),
                    d == null ? IntPtr.Zero : TensorComputePrimitives.GetStoragePointer(d),
                    normW == null ? IntPtr.Zero : TensorComputePrimitives.GetStoragePointer(normW),
                    dInner,
                    dState,
                    nHead,
                    headDim,
                    nGroup,
                    dConv,
                    Config.Eps);

                _mamba2NativeDecodeStateInitialized[layer] = true;
                if (TryLinearAddInto(residual, result, prefix + "ssm_out.weight"))
                {
                    output = null;
                    return true;
                }

                output = LinearForward(result, prefix + "ssm_out.weight");
                return true;
            }
            catch (ArgumentException)
            {
                return false;
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

        // Per-(layer, slot) state key for the persistent GPU decode-state cache
        // (g_mamba2_decode_cache in ggml_ops_mamba2.cpp). Encoding:
        //   bits 63..32 : model instance id
        //   bits 31..16 : layer index
        //   bits 15..0  : slot index
        // The cache-clear API (TSGgml_NemotronMamba2DecodeClear) matches entries
        // by `state_key >> 32 == model_key`, so the model id occupies the same
        // upper-32-bit slot as before and the existing clear logic continues to
        // work unchanged.
        //
        // CRITICAL: the slot MUST be part of the key. The C++ cache also keys
        // on the host projected/hidden_out pointers, but only when zero-copy
        // succeeds (Metal + 4KB+ tensor + aligned host ptr). When zero-copy
        // fails the pointers are 0u in the cache key, and without slot here
        // every concurrent batched sequence would collapse to the same cache
        // entry and trample each other's GPU-side conv/SSM state across
        // decode steps, producing garbled output for all participants.
        private ulong NativeMamba2DecodeStateKey(int layer, int slot = 0) =>
            (_nativeMamba2DecodeModelId << 32)
            | ((ulong)(uint)(layer & 0xFFFF) << 16)
            | (uint)(slot & 0xFFFF);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private unsafe void Mamba2Conv1dStep(float* xBCPtr, int xBCSize,
            float[] convState, int convDim, float[] convWT, float* convWPtr, float* convBiasPtr,
            float[] convOutBuf)
        {
            int dConv = convDim + 1;

            fixed (float* statePtr = convState, outPtr = convOutBuf)
            {
                if (convWT != null)
                {
                    fixed (float* wtPtr = convWT)
                    {
                        Mamba2Conv1dStepVectorized(xBCPtr, xBCSize, statePtr, convDim, wtPtr, convBiasPtr, outPtr);
                    }
                }
                else
                {
                    for (int ch = 0; ch < xBCSize; ch++)
                    {
                        float sum = 0;
                        float* cW = convWPtr + ch * dConv;
                        for (int ki = 0; ki < convDim; ki++)
                            sum += statePtr[ki * xBCSize + ch] * cW[ki];
                        sum += xBCPtr[ch] * cW[convDim];

                        if (convBiasPtr != null)
                            sum += convBiasPtr[ch];

                        outPtr[ch] = sum;
                    }
                }
            }

            if (convDim > 1)
                Array.Copy(convState, xBCSize, convState, 0, (convDim - 1) * xBCSize);
            if (convDim > 0)
            {
                fixed (float* statePtr = convState)
                    Buffer.MemoryCopy(xBCPtr, statePtr + (convDim - 1) * xBCSize, xBCSize * 4, xBCSize * 4);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void Mamba2Conv1dStepVectorized(
            float* xBCPtr,
            int xBCSize,
            float* statePtr,
            int convDim,
            float* wtPtr,
            float* convBiasPtr,
            float* outPtr)
        {
            int vLen = Vector<float>.Count;
            int ch = 0;

            if (convDim > 0)
            {
                float* wt0 = wtPtr;
                for (; ch <= xBCSize - vLen; ch += vLen)
                    StV(outPtr + ch, LdV(statePtr + ch) * LdV(wt0 + ch));
                for (; ch < xBCSize; ch++)
                    outPtr[ch] = statePtr[ch] * wt0[ch];

                for (int ki = 1; ki < convDim; ki++)
                {
                    float* stateRow = statePtr + (long)ki * xBCSize;
                    float* wtRow = wtPtr + (long)ki * xBCSize;
                    ch = 0;
                    for (; ch <= xBCSize - vLen; ch += vLen)
                        StV(outPtr + ch, LdV(outPtr + ch) + LdV(stateRow + ch) * LdV(wtRow + ch));
                    for (; ch < xBCSize; ch++)
                        outPtr[ch] += stateRow[ch] * wtRow[ch];
                }
            }

            float* inputWt = wtPtr + (long)convDim * xBCSize;
            ch = 0;
            if (convDim > 0)
            {
                for (; ch <= xBCSize - vLen; ch += vLen)
                    StV(outPtr + ch, LdV(outPtr + ch) + LdV(xBCPtr + ch) * LdV(inputWt + ch));
                for (; ch < xBCSize; ch++)
                    outPtr[ch] += xBCPtr[ch] * inputWt[ch];
            }
            else
            {
                for (; ch <= xBCSize - vLen; ch += vLen)
                    StV(outPtr + ch, LdV(xBCPtr + ch) * LdV(inputWt + ch));
                for (; ch < xBCSize; ch++)
                    outPtr[ch] = xBCPtr[ch] * inputWt[ch];
            }

            if (convBiasPtr != null)
            {
                ch = 0;
                for (; ch <= xBCSize - vLen; ch += vLen)
                    StV(outPtr + ch, LdV(outPtr + ch) + LdV(convBiasPtr + ch));
                for (; ch < xBCSize; ch++)
                    outPtr[ch] += convBiasPtr[ch];
            }
        }

        internal static unsafe void Mamba2Conv1dStepVectorizedForTest(
            float[] xBC,
            float[] convState,
            int convDim,
            float[] convWT,
            float[] convBias,
            float[] output)
        {
            if (xBC == null || convState == null || convWT == null || output == null)
                throw new ArgumentNullException();
            int xBCSize = xBC.Length;
            fixed (float* xPtr = xBC)
            fixed (float* statePtr = convState)
            fixed (float* wtPtr = convWT)
            fixed (float* biasPtr = convBias)
            fixed (float* outPtr = output)
            {
                Mamba2Conv1dStepVectorized(xPtr, xBCSize, statePtr, convDim, wtPtr, biasPtr, outPtr);
            }
        }

        /// <summary>
        /// SIMD-optimized Mamba2 SSM scan step. The inner dState loop is vectorized
        /// using System.Numerics.Vector for 4-8x throughput on NEON/AVX2.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private unsafe void Mamba2SSMStepSIMD(float[] convOut, float* dt, float* A, float* D,
            float[] ssmState, int dInner, int dState, int nHead, int headDim, int nGroup,
            float[] yBuf)
        {
            int headsPerGroup = nHead / nGroup;
            int statePerHead = dState * headDim;
            int vLen = Vector<float>.Count;

            fixed (float* xBase = convOut, stateBase = ssmState, yBase = yBuf)
            {
                float* bBase = xBase + dInner;
                float* cBase = bBase + nGroup * dState;

                for (int h = 0; h < nHead; h++)
                {
                    float dtSoftplus = SoftplusScalar(dt[h]);
                    float dA = MathF.Exp(dtSoftplus * A[h]);
                    int g = h / headsPerGroup;

                    float* stateH = stateBase + h * statePerHead;
                    float* xH = xBase + h * headDim;
                    float* yH = yBase + h * headDim;
                    float* bG = bBase + g * dState;
                    float* cG = cBase + g * dState;

                    var vDA = new Vector<float>(dA);

                    for (int d = 0; d < headDim; d++)
                    {
                        float xDt = xH[d] * dtSoftplus;
                        var vXDt = new Vector<float>(xDt);

                        float* stateCol = stateH + d * dState;

                        // SIMD-vectorized inner loop over dState
                        var vSum = Vector<float>.Zero;
                        int si = 0;
                        for (; si <= dState - vLen * 2; si += vLen * 2)
                        {
                            var vs0 = LdV(stateCol + si);
                            var vb0 = LdV(bG + si);
                            var vc0 = LdV(cG + si);
                            vs0 = vs0 * vDA + vb0 * vXDt;
                            StV(stateCol + si, vs0);
                            vSum += vs0 * vc0;

                            var vs1 = LdV(stateCol + si + vLen);
                            var vb1 = LdV(bG + si + vLen);
                            var vc1 = LdV(cG + si + vLen);
                            vs1 = vs1 * vDA + vb1 * vXDt;
                            StV(stateCol + si + vLen, vs1);
                            vSum += vs1 * vc1;
                        }
                        for (; si <= dState - vLen; si += vLen)
                        {
                            var vs = LdV(stateCol + si);
                            var vb = LdV(bG + si);
                            var vc = LdV(cG + si);
                            vs = vs * vDA + vb * vXDt;
                            StV(stateCol + si, vs);
                            vSum += vs * vc;
                        }
                        float sumf = Vector.Sum(vSum);
                        for (; si < dState; si++)
                        {
                            stateCol[si] = stateCol[si] * dA + bG[si] * xDt;
                            sumf += stateCol[si] * cG[si];
                        }

                        yH[d] = sumf;
                    }

                    if (D != null)
                    {
                        float dH = D[h];
                        for (int d = 0; d < headDim; d++)
                            yH[d] += dH * xH[d];
                    }
                }
            }
        }

        #endregion

        #region Helper functions

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float SigmoidScalar(float x) => TensorComputePrimitives.Sigmoid(x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float SiLUScalar(float x) => TensorComputePrimitives.SiLU(x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float SoftplusScalar(float x) => TensorComputePrimitives.Softplus(x);

        #endregion

        /// <summary>
        /// Copy <paramref name="multimodalEmbeddings"/> (shape [N, hiddenSize]) into the
        /// rows <paramref name="insertPos"/>..<paramref name="insertPos"/>+N-1 of
        /// <paramref name="hidden"/>. Handles cross-allocator copies for multimodal
        /// encoders that live on a different backend than the LM (eg. CPU vision
        /// encoder feeding a CUDA LM).
        /// </summary>
        private void InjectMultimodalEmbeddings(Tensor hidden, Tensor multimodalEmbeddings, int insertPos)
        {
            int numTokens = (int)multimodalEmbeddings.Sizes[0];
            using var target = hidden.Narrow(0, insertPos, numTokens);
            if (ReferenceEquals(target.Allocator, multimodalEmbeddings.Allocator))
            {
                Ops.Copy(target, multimodalEmbeddings);
            }
            else
            {
                float[] hostEmbeddings = multimodalEmbeddings.GetElementsAsFloat((int)multimodalEmbeddings.ElementCount());
                using var deviceEmb = new Tensor(hidden.Allocator, multimodalEmbeddings.ElementType, multimodalEmbeddings.Sizes);
                deviceEmb.SetElementsAsFloat(hostEmbeddings);
                Ops.Copy(target, deviceEmb);
            }
        }

        private static Tensor CreateRepeatedEmbeddingRows(Tensor source, int targetRows)
        {
            int sourceRows = (int)source.Sizes[0];
            if (targetRows <= sourceRows)
            {
                if (targetRows == sourceRows)
                    return source;

                using var rows = source.Narrow(0, 0, targetRows);
                var clone = new Tensor(source.Allocator, source.ElementType, rows.Sizes);
                Ops.Copy(clone, rows);
                return clone;
            }

            int hidden = (int)source.Sizes[1];
            var repeated = new Tensor(source.Allocator, source.ElementType, targetRows, hidden);
            int offset = 0;
            while (offset < targetRows)
            {
                int rows = Math.Min(sourceRows, targetRows - offset);
                using var src = source.Narrow(0, 0, rows);
                using var dst = repeated.Narrow(0, offset, rows);
                Ops.Copy(dst, src);
                offset += rows;
            }

            return repeated;
        }

        private void ClearPendingMultimodalEmbeddings()
        {
            foreach (var (emb, _) in _pendingVisionEmbeddings) emb?.Dispose();
            _pendingVisionEmbeddings.Clear();
            foreach (var (emb, _) in _pendingAudioEmbeddings) emb?.Dispose();
            _pendingAudioEmbeddings.Clear();
        }

        public override void Dispose()
        {
            if (_kvCacheK != null)
                foreach (var t in _kvCacheK) t?.Dispose();
            if (_kvCacheV != null)
                foreach (var t in _kvCacheV) t?.Dispose();
            DisposeTensorArray(_mamba2NativeDecodeProjected);
            DisposeTensorArray(_mamba2NativeDecodeHidden);
            if (IsGgmlBackend)
                GgmlBasicOps.NemotronMamba2DecodeClear(_nativeMamba2DecodeModelId);
            _expertUpResult?.Dispose();
            _expertDownResult?.Dispose();
            _latentAccumTensor?.Dispose();
            _latentOutResult?.Dispose();

            ClearPendingMultimodalEmbeddings();

            _visionEncoder?.Dispose();
            _visionEncoder = null;

            base.Dispose();
        }
    }
}
