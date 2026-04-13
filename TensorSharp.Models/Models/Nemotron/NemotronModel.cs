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
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.GGML;

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
    public class NemotronModel : ModelBase
    {
        private enum LayerType { Mamba2, Attention, FFN }

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

        // Pre-allocated MoE buffers
        private float[] _moeProbs;
        private float[] _moeSelectionProbs;
        private int[] _moeTopExperts;
        private float[] _moeRouteW;
        private float[] _moeLatentAccum;

        // Pre-cached expert weight key strings to avoid string interpolation in hot loop
        private string[][] _expertUpKeys;   // [layer][expertIdx]
        private string[][] _expertDownKeys; // [layer][expertIdx]

        // Pre-cached expert QuantizedWeight refs to avoid dictionary lookups in hot loop
        private QuantizedWeight[][] _expertUpQW;   // [layer][expertIdx]
        private QuantizedWeight[][] _expertDownQW; // [layer][expertIdx]

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

            InitCaches(4096);
            InitMamba2Buffers();
            InitLayerInfo();
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
                    long totalBytes = qw.RawBytes + kw.RawBytes + vw.RawBytes;
                    IntPtr fusedPtr = QuantizedWeight.AllocateBuffer(totalBytes);
                    Buffer.MemoryCopy(qw.Data.ToPointer(), fusedPtr.ToPointer(), totalBytes, qw.RawBytes);
                    Buffer.MemoryCopy(kw.Data.ToPointer(), (fusedPtr + (int)qw.RawBytes).ToPointer(), totalBytes - qw.RawBytes, kw.RawBytes);
                    Buffer.MemoryCopy(vw.Data.ToPointer(), (fusedPtr + (int)(qw.RawBytes + kw.RawBytes)).ToPointer(), totalBytes - qw.RawBytes - kw.RawBytes, vw.RawBytes);
                    _quantWeights[qkvName] = new QuantizedWeight(fusedPtr, totalBytes, qw.GgmlType, qw.Ne0, qw.Ne1 + kw.Ne1 + vw.Ne1);
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
        }

        private void InitCaches(int maxSeqLen)
        {
            _maxContextLength = maxSeqLen;
            int numLayers = Config.NumLayers;
            _kvCacheK = new Tensor[numLayers];
            _kvCacheV = new Tensor[numLayers];
            _convState = new float[numLayers][];
            _ssmState = new float[numLayers][];

            int convDim = Math.Max(0, _ssmDConv - 1);
            int convChannels = _ssmDInner + 2 * _ssmNGroup * _ssmDState;
            int ssmStateSize = _ssmDState * _ssmHeadDim * _ssmNHead;

            for (int l = 0; l < numLayers; l++)
            {
                switch (_layerTypes[l])
                {
                    case LayerType.Attention:
                        int numKVH = _layerNumKVHeads[l];
                        int headDim = Config.HeadDim;
                        _kvCacheK[l] = new Tensor(_allocator, DType.Float32, numKVH, maxSeqLen, headDim);
                        _kvCacheV[l] = new Tensor(_allocator, DType.Float32, numKVH, maxSeqLen, headDim);
                        Ops.Fill(_kvCacheK[l], 0);
                        Ops.Fill(_kvCacheV[l], 0);
                        break;
                    case LayerType.Mamba2:
                        _convState[l] = new float[convDim * convChannels];
                        _ssmState[l] = new float[ssmStateSize];
                        break;
                }
            }
            _cacheSeqLen = 0;
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

        #endregion

        public override void ResetKVCache()
        {
            for (int l = 0; l < Config.NumLayers; l++)
            {
                switch (_layerTypes[l])
                {
                    case LayerType.Attention:
                        Ops.Fill(_kvCacheK[l], 0);
                        Ops.Fill(_kvCacheV[l], 0);
                        InvalidateTensorDeviceCache(_kvCacheK[l]);
                        InvalidateTensorDeviceCache(_kvCacheV[l]);
                        break;
                    case LayerType.Mamba2:
                        Array.Clear(_convState[l]);
                        Array.Clear(_ssmState[l]);
                        break;
                }
            }
            _cacheSeqLen = 0;
            _linearTicks = _attnTicks = _normTicks = _embTicks = _lmHeadTicks = _logitsCopyTicks = 0;
            _forwardCount = 0;
            _forwardSw.Reset();
        }

        public override bool SupportsKVCacheTruncation => false;

        #region CPU-optimized helpers for decode path

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector<float> LdV(float* p) =>
            Unsafe.ReadUnaligned<Vector<float>>(ref *(byte*)p);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void StV(float* p, Vector<float> v) =>
            Unsafe.WriteUnaligned(ref *(byte*)p, v);

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
                    GgmlBasicOps.AddmmQuant(result, input, qw.Data, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes);
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

        #endregion

        public override float[] Forward(int[] tokens)
        {
            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            bool isDecode = seqLen == 1;

            long t1 = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t1;

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
            Tensor attnOut = AttentionForward(normed, layer, prefix, seqLen, startPos);
            normed.Dispose();

            Ops.Add(hidden, hidden, attnOut);
            attnOut.Dispose();
            return hidden;
        }

        private unsafe Tensor AttentionForward(Tensor input, int layer, string prefix, int seqLen, int startPos)
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
                AttentionDecodePureCS(qTensor, _kvCacheK[layer], _kvCacheV[layer],
                    attnResult, numHeads, numKVHeads, headDim, totalSeqLen, scale);
                qTensor.Dispose();

                _attnTicks += Stopwatch.GetTimestamp() - t0;

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

            Ops.AddCausalMask(scores, seqLen, startPos, float.NegativeInfinity);
            Ops.Softmax(scores, scores);

            var attnOut = new Tensor(_allocator, DType.Float32, numHeads, seqLen, headDim);
            Ops.AddmmBatch(attnOut, 0, attnOut, 1.0f, scores, vExpanded);
            scores.Dispose();
            vExpanded.Dispose();

            Tensor flatOutput = ReshapeFromHeads(attnOut, numHeads, seqLen, headDim);
            attnOut.Dispose();

            _attnTicks += Stopwatch.GetTimestamp() - t0;

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
                ffnOut = MoEForward(normed, layer, prefix, seqLen, isDecode);
            else
                ffnOut = DenseFFNForward(normed, prefix, seqLen);
            normed.Dispose();

            Ops.Add(hidden, hidden, ffnOut);
            ffnOut.Dispose();
            return hidden;
        }

        private Tensor DenseFFNForward(Tensor input, string prefix, int seqLen)
        {
            Tensor up = LinearForward(input, prefix + "ffn_up.weight");
            ReluSquaredInPlace(up);
            Tensor down = LinearForward(up, prefix + "ffn_down.weight");
            up.Dispose();
            return down;
        }

        private unsafe Tensor MoEForward(Tensor input, int layer, string prefix, int seqLen, bool isDecode)
        {
            int hiddenSize = Config.HiddenSize;
            ref var moeInfo = ref _moeLayerInfo[layer];

            Tensor routerLogits = LinearForward(input, prefix + "ffn_gate_inp.weight");
            float* routerPtr = GetFloatPtr(routerLogits);

            var output = new Tensor(_allocator, DType.Float32, seqLen, hiddenSize);
            float* outputPtr = GetFloatPtr(output);
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
                float* outRow = outputPtr + (long)s * hiddenSize;

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
                                _moeUpPtrs[k] = _expertUpQW[layer][ei].Data;
                                _moeDownPtrs[k] = _expertDownQW[layer][ei].Data;
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
                        for (int i = 0; i < hiddenSize; i++)
                            outRow[i] += latentAccum[i];
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

            InvalidateTensorDeviceCache(output);
            return output;
        }

        private static void SelectTopKInPlace(float[] values, int n, int k, int[] indices)
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

        private unsafe void ReluSquaredInPlace(Tensor t)
        {
            float* ptr = GetFloatPtr(t);
            long count = t.ElementCount();
            int vLen = Vector<float>.Count;
            var zero = Vector<float>.Zero;
            long i = 0;
            for (; i <= count - vLen; i += vLen)
            {
                var v = LdV(ptr + i);
                var mask = Vector.GreaterThan(v, zero);
                StV(ptr + i, Vector.ConditionalSelect(mask, v * v, zero));
            }
            for (; i < count; i++)
            {
                float v = ptr[i];
                ptr[i] = v > 0f ? v * v : 0f;
            }
        }

        #endregion

        #region Mamba2 Block

        private Tensor Mamba2Block(Tensor hidden, int layer, int seqLen, bool isDecode)
        {
            string prefix = _layerPrefixes[layer];

            Tensor normed = RMSNormOp(hidden, prefix + "attn_norm.weight");
            Tensor mamba2Out = Mamba2Forward(normed, layer, prefix, seqLen);
            normed.Dispose();

            Ops.Add(hidden, hidden, mamba2Out);
            mamba2Out.Dispose();
            return hidden;
        }

        /// <summary>
        /// Mamba2 SSM forward pass with SIMD-optimized inner loops.
        /// </summary>
        private unsafe Tensor Mamba2Forward(Tensor input, int layer, string prefix, int seqLen)
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

            Tensor projected = LinearForward(input, prefix + "ssm_in.weight");
            float* projPtr = GetFloatPtr(projected);

            Tensor result = new Tensor(_allocator, DType.Float32, seqLen, dInner);
            float* resultPtr = GetFloatPtr(result);

            float[] convState = _convState[layer];
            float[] ssmState = _ssmState[layer];

            float* convWPtr = GetFloatPtr(_weights[prefix + "ssm_conv1d.weight"]);
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

                Mamba2Conv1dStep(xBCPtr, xBCSize, convState, convDim, convWPtr, convBiasPtr, convOutBuf);

                // SiLU on conv output - SIMD optimized
                fixed (float* coBuf = convOutBuf)
                {
                    for (int i = 0; i < xBCSize; i++)
                        coBuf[i] = SiLUScalar(coBuf[i]);
                }

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

            Tensor outProj = LinearForward(result, prefix + "ssm_out.weight");
            result.Dispose();

            _attnTicks += Stopwatch.GetTimestamp() - t0;
            return outProj;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private unsafe void Mamba2Conv1dStep(float* xBCPtr, int xBCSize,
            float[] convState, int convDim, float* convWPtr, float* convBiasPtr,
            float[] convOutBuf)
        {
            int dConv = convDim + 1;

            fixed (float* statePtr = convState, outPtr = convOutBuf)
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

            if (convDim > 1)
                Array.Copy(convState, xBCSize, convState, 0, (convDim - 1) * xBCSize);
            if (convDim > 0)
            {
                fixed (float* statePtr = convState)
                    Buffer.MemoryCopy(xBCPtr, statePtr + (convDim - 1) * xBCSize, xBCSize * 4, xBCSize * 4);
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

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float SiLUScalar(float x) => x * SigmoidScalar(x);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static float SoftplusScalar(float x)
        {
            if (x > 20f) return x;
            if (x < -20f) return MathF.Exp(x);
            return MathF.Log(1.0f + MathF.Exp(x));
        }

        #endregion

        public override void Dispose()
        {
            if (_kvCacheK != null)
                foreach (var t in _kvCacheK) t?.Dispose();
            if (_kvCacheV != null)
                foreach (var t in _kvCacheV) t?.Dispose();
            _expertUpResult?.Dispose();
            _expertDownResult?.Dispose();
            _latentAccumTensor?.Dispose();
            _latentOutResult?.Dispose();

            base.Dispose();
        }
    }
}

