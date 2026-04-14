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
using TensorSharp.GGML;

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
    public class Gemma4Model : ModelBase
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

        private int _pleDim;

        private Tensor[] _kvCacheK;
        private Tensor[] _kvCacheV;
        private int[] _kvCacheSize; // per-layer cache capacity (slidingWindow for SWA, maxSeqLen for global)

        private float[] _layerScalars;
        private bool _hasTiedOutput;

        private Tensor _onesForVNorm;

        private int _numExperts;
        private int _numExpertsUsed;

        private bool _canUseFusedDecode;
        private Gemma4DecodeArrays _decodeArrays;

        private Gemma4VisionEncoder _visionEncoder;
        private Gemma4AudioEncoder _audioEncoder;
        private List<(Tensor embeddings, int position)> _pendingVisionEmbeddingsList = new();
        private List<(Tensor embeddings, int position)> _pendingAudioEmbeddingsList = new();

        public Gemma4VisionEncoder VisionEncoder => _visionEncoder;
        public Gemma4AudioEncoder AudioEncoder => _audioEncoder;

        public void LoadVisionEncoder(string mmProjPath)
        {
            _visionEncoder = new Gemma4VisionEncoder(mmProjPath, _allocator);
        }

        public void LoadAudioEncoder(string mmProjPath)
        {
            _audioEncoder = new Gemma4AudioEncoder(mmProjPath, _allocator);
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
            PrecomputeRoPE();
            InitKVCache(ResolveConfiguredContextLength());
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
                        if (isLocal)
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

        private void InitKVCache(int maxSeqLen)
        {
            _maxContextLength = maxSeqLen;
            _kvCacheK = new Tensor[Config.NumLayers];
            _kvCacheV = new Tensor[Config.NumLayers];
            _kvCacheSize = new int[Config.NumLayers];

            long totalCacheBytes = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_kvDonorMap.ContainsKey(l)) continue;

                int kvHeads = KVHeadsForLayer(l);
                int hd = HeadDimForLayer(l);
                int cacheLen = IsLocalLayer(l) ? _slidingWindow : maxSeqLen;
                _kvCacheSize[l] = cacheLen;
                _kvCacheK[l] = new Tensor(_allocator, DType.Float32, kvHeads, cacheLen, hd);
                _kvCacheV[l] = new Tensor(_allocator, DType.Float32, kvHeads, cacheLen, hd);
                Ops.Fill(_kvCacheK[l], 0f);
                Ops.Fill(_kvCacheV[l], 0f);
                totalCacheBytes += (long)kvHeads * cacheLen * hd * 4 * 2;
            }

            foreach (var kv in _kvDonorMap)
            {
                _kvCacheK[kv.Key] = _kvCacheK[kv.Value];
                _kvCacheV[kv.Key] = _kvCacheV[kv.Value];
                _kvCacheSize[kv.Key] = _kvCacheSize[kv.Value];
            }

            Console.WriteLine($"  KV cache: {totalCacheBytes / 1024 / 1024} MB " +
                $"(global layers: {maxSeqLen} seq, SWA layers: {_slidingWindow} seq)");
        }

        public override void ResetKVCache()
        {
            _cacheSeqLen = 0;
            if (_kvCacheK == null) return;
            var cleared = new HashSet<int>();
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (cleared.Contains(l)) continue;
                if (_kvDonorMap.ContainsKey(l)) continue;
                Ops.Fill(_kvCacheK[l], 0f);
                Ops.Fill(_kvCacheV[l], 0f);
                InvalidateTensorDeviceCache(_kvCacheK[l]);
                InvalidateTensorDeviceCache(_kvCacheV[l]);
                cleared.Add(l);
            }
        }

        public override void TruncateKVCache(int tokenCount)
        {
            base.TruncateKVCache(tokenCount);
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

        public void SetVisionEmbeddings(Tensor embeddings, int insertPosition)
        {
            _pendingVisionEmbeddingsList.Add((embeddings, insertPosition));
        }

        public override float[] Forward(int[] tokens)
        {
            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;

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
                        exceptPositions.Add(pos + i);
                    InjectVisionEmbeddings(hidden, emb, pos);
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
                        exceptPositions.Add(pos + i);
                    InjectVisionEmbeddings(hidden, emb, pos);
                    emb.Dispose();
                }
                _pendingAudioEmbeddingsList.Clear();
            }

            Tensor perLayerInputs = null;
            if (_pleDim > 0)
                perLayerInputs = ComputePLE(tokens, hidden, seqLen);

            if (seqLen == 1 && _canUseFusedDecode)
            {
                long tFused = Stopwatch.GetTimestamp();
                NativeGemma4ModelDecode(hidden, startPos, perLayerInputs);
                _linearTicks += Stopwatch.GetTimestamp() - tFused;
            }
            else
            {
                if (seqLen > 1 && startPos > 0 && _swaKVDonorLayers.Count > 0)
                    _prefillSWAKV = new Dictionary<int, (Tensor, Tensor)>();

                for (int l = 0; l < Config.NumLayers; l++)
                {
                    Tensor perLayerInput = null;
                    if (perLayerInputs != null)
                        perLayerInput = ExtractPerLayerSlice(perLayerInputs, l, seqLen);

                    bool isShared = _kvDonorMap.ContainsKey(l);
                    hidden = TransformerBlock(hidden, l, seqLen, startPos, isShared, perLayerInput, exceptPositions);

                    perLayerInput?.Dispose();
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
            }

            perLayerInputs?.Dispose();

            Tensor normed = RMSNormOp(hidden, "output_norm.weight");
            hidden.Dispose();

            Tensor lastHidden;
            if (seqLen > 1)
            {
                // Prefill only needs next-token logits, so keep the LM head on the final token.
                using var lastRow = normed.Narrow(0, seqLen - 1, 1);
                lastHidden = Ops.NewContiguous(lastRow);
            }
            else
            {
                lastHidden = normed.CopyRef();
            }
            normed.Dispose();

            t0 = Stopwatch.GetTimestamp();
            string outputWeight = _hasTiedOutput ? "token_embd.weight" : "output.weight";
            Tensor logitsTensor = LinearForward(lastHidden, outputWeight);
            _lmHeadTicks += Stopwatch.GetTimestamp() - t0;
            lastHidden.Dispose();

            if (_finalLogitSoftcap > 0f)
                ApplyLogitSoftcap(logitsTensor);

            t0 = Stopwatch.GetTimestamp();
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

            _cacheSeqLen += seqLen;
            _forwardCount++;
            _forwardSw.Stop();
            return _logitsBuffer;
        }

        private void ScaleEmbedding(Tensor hidden)
        {
            float scale = MathF.Sqrt(Config.HiddenSize);
            Ops.Mul(hidden, hidden, scale);
        }

        private void ApplyLogitSoftcap(Tensor logits)
        {
            float cap = _finalLogitSoftcap;
            Ops.Mul(logits, logits, 1f / cap);
            Ops.Tanh(logits, logits);
            Ops.Mul(logits, logits, cap);
        }

        private void InjectVisionEmbeddings(Tensor hidden, Tensor visionEmbeddings, int insertPos)
        {
            int numVisionTokens = (int)visionEmbeddings.Sizes[0];
            using var target = hidden.Narrow(0, insertPos, numVisionTokens);
            Ops.Copy(target, visionEmbeddings);
            Console.WriteLine($"Injected {numVisionTokens} vision tokens at position {insertPos}");
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
                _canUseFusedDecode = false;
                return;
            }

            int n = Config.NumLayers;

            // Verify all layers have the needed quantized weight
            for (int l = 0; l < n; l++)
            {
                string prefix = $"blk.{l}";
                bool isShared = _kvDonorMap.ContainsKey(l);
                string qkvKey = isShared ? $"{prefix}.attn_q.weight" : $"{prefix}.attn_qkv.weight";
                if (!_quantWeights.ContainsKey(qkvKey))
                {
                    _canUseFusedDecode = false;
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
                a.KCache[l] = (IntPtr)GetFloatPtr(_kvCacheK[kvSource]);
                a.VCache[l] = (IntPtr)GetFloatPtr(_kvCacheV[kvSource]);

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
                        a.Qkv[l] = qW.Data;
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
                        a.Qkv[l] = qkvW.Data;
                        a.QkvType[l] = qkvW.GgmlType;
                        a.QkvNe0[l] = qkvW.Ne0;
                        a.QkvNe1[l] = qkvW.Ne1;
                        a.QkvBytes[l] = qkvW.RawBytes;
                    }
                }

                string oName = $"{prefix}.attn_output.weight";
                if (_quantWeights.TryGetValue(oName, out var oW))
                {
                    a.O[l] = oW.Data;
                    a.OType[l] = oW.GgmlType;
                    a.ONe0[l] = oW.Ne0;
                    a.ONe1[l] = oW.Ne1;
                    a.OBytes[l] = oW.RawBytes;
                }

                string guName = $"{prefix}.ffn_gate_up.weight";
                if (_quantWeights.TryGetValue(guName, out var guW))
                {
                    a.Gu[l] = guW.Data;
                    a.GuType[l] = guW.GgmlType;
                    a.GuNe0[l] = guW.Ne0;
                    a.GuNe1[l] = guW.Ne1;
                    a.GuBytes[l] = guW.RawBytes;
                }

                string downName = $"{prefix}.ffn_down.weight";
                if (_quantWeights.TryGetValue(downName, out var downW))
                {
                    a.Down[l] = downW.Data;
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
                    a.PleGate[l] = pleGW.Data;
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
                        a.PleProj[l] = plePW.Data;
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
            _canUseFusedDecode = true;
            Console.WriteLine("  Gemma4 fused model decode enabled");
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
                a.PlePostNorm);
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
                    GgmlBasicOps.GetRowsQuant(pleTokenEmb, pleQw.Data, pleQw.GgmlType, pleQw.Ne0, pleQw.Ne1, pleQw.RawBytes, pleIdx);
                }
                else
                {
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
            var slice = perLayerInputs.Narrow(1, offset, _pleDim);
            if (seqLen == 1)
                return slice;
            var contiguous = Ops.NewContiguous(slice);
            slice.Dispose();
            return contiguous;
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
            string prefix = $"blk.{layer}";

            using var attnNormed = RMSNormOp(hidden, $"{prefix}.attn_norm.weight");

            using var attnOut = Attention(attnNormed, layer, prefix, seqLen, startPos, isShared, exceptPositions);

            using var postAttnNormed = RMSNormOp(attnOut, $"{prefix}.post_attention_norm.weight");

            Ops.Add(postAttnNormed, postAttnNormed, hidden);
            hidden.Dispose();

            Tensor result;

            if (HasMoE(layer))
            {
                // MoE: run dense MLP and MoE in parallel, then combine
                using var mlpNormed = RMSNormOp(postAttnNormed, $"{prefix}.ffn_norm.weight");
                using var mlpOut = FFNGelu(mlpNormed, $"{prefix}.ffn_gate_up.weight",
                    $"{prefix}.ffn_down.weight", seqLen);

                string postMlpNorm1Key = $"{prefix}.post_ffw_norm_1.weight";
                if (!_weights.ContainsKey(postMlpNorm1Key))
                    postMlpNorm1Key = $"{prefix}.ffn_post_norm_1.weight";
                using var postMlpNormed1 = RMSNormOp(mlpOut, postMlpNorm1Key);

                using var moeOut = MoEForward(postAttnNormed, layer, prefix, seqLen);

                string postMoeNormKey = $"{prefix}.post_ffw_norm_2.weight";
                if (!_weights.ContainsKey(postMoeNormKey))
                    postMoeNormKey = $"{prefix}.ffn_post_norm_2.weight";
                using var postMoeNormed = RMSNormOp(moeOut, postMoeNormKey);

                // Combine MLP + MoE
                var combined = new Tensor(_allocator, DType.Float32, postMlpNormed1.Sizes);
                Ops.Add(combined, postMlpNormed1, postMoeNormed);

                string postFfnNormKey = $"{prefix}.post_ffw_norm.weight";
                if (!_weights.ContainsKey(postFfnNormKey))
                    postFfnNormKey = $"{prefix}.ffn_post_norm.weight";
                using var postFfnNormed = RMSNormOp(combined, postFfnNormKey);
                combined.Dispose();

                result = new Tensor(_allocator, DType.Float32, postAttnNormed.Sizes);
                Ops.Add(result, postAttnNormed, postFfnNormed);
            }
            else
            {
                // Dense layers: MLP only
                using var ffnNormed = RMSNormOp(postAttnNormed, $"{prefix}.ffn_norm.weight");
                using var ffnOut = FFNGelu(ffnNormed, $"{prefix}.ffn_gate_up.weight",
                    $"{prefix}.ffn_down.weight", seqLen);

                string postFfnNormKey = $"{prefix}.post_ffw_norm.weight";
                if (!_weights.ContainsKey(postFfnNormKey))
                    postFfnNormKey = $"{prefix}.ffn_post_norm.weight";
                using var postFfnNormed = RMSNormOp(ffnOut, postFfnNormKey);

                result = new Tensor(_allocator, DType.Float32, postAttnNormed.Sizes);
                Ops.Add(result, postAttnNormed, postFfnNormed);
            }

            // PLE injection
            if (perLayerInput != null &&
                (_weights.ContainsKey($"{prefix}.inp_gate.weight") || _quantWeights.ContainsKey($"{prefix}.inp_gate.weight")))
            {
                using var gate = LinearForward(result, $"{prefix}.inp_gate.weight");
                if (gate != null)
                {
                    Ops.GELUMul(gate, gate, perLayerInput);
                    using var pleProj = LinearForward(gate, $"{prefix}.proj.weight");
                    if (pleProj != null)
                    {
                        string postPleNormKey = $"{prefix}.post_norm.weight";
                        using var pleNormed = RMSNormOp(pleProj, postPleNormKey);
                        Ops.Add(result, result, pleNormed);
                    }
                }
            }

            float scalar = _layerScalars[layer];
            if (scalar != 1f)
                Ops.Mul(result, result, scalar);

            return result;
        }

        #region MoE

        private unsafe Tensor MoEForward(Tensor hiddenState, int layer, string prefix, int seqLen)
        {
            // Router receives UN-normed hiddenState (Router does its own unweighted RMSNorm)
            var (routingWeights, selectedExperts) = MoERoute(hiddenState, prefix, seqLen);

            // Pre-norm for expert input (separate from router norm)
            string moeNormKey = $"{prefix}.pre_ffw_norm_2.weight";
            if (!_weights.ContainsKey(moeNormKey))
                moeNormKey = $"{prefix}.ffn_pre_norm_2.weight";
            using var moeInput = RMSNormOp(hiddenState, moeNormKey);

            // Expert computation: for each token, run selected experts
            int hiddenDim = (int)moeInput.Sizes[1];
            var output = new Tensor(_allocator, DType.Float32, seqLen, hiddenDim);
            Ops.Fill(output, 0f);

            float* inputPtr = GetFloatPtr(moeInput);
            float* outputPtr = GetFloatPtr(output);

            var tokenInput = new Tensor(_allocator, DType.Float32, 1, hiddenDim);
            float* tokenPtr = GetFloatPtr(tokenInput);

            for (int s = 0; s < seqLen; s++)
            {
                for (int e = 0; e < _numExpertsUsed; e++)
                {
                    int expertIdx = selectedExperts[s * _numExpertsUsed + e];
                    float weight = routingWeights[s * _numExpertsUsed + e];

                    Buffer.MemoryCopy(inputPtr + (long)s * hiddenDim, tokenPtr,
                        hiddenDim * sizeof(float), hiddenDim * sizeof(float));

                    // Run expert FFN: GELU(gate) * up, then down
                    string gateKey = $"{prefix}.ffn_gate_exps.{expertIdx}.weight";
                    string upKey = $"{prefix}.ffn_up_exps.{expertIdx}.weight";
                    string downKey = $"{prefix}.ffn_down_exps.{expertIdx}.weight";

                    // Try fused gate_up first
                    string fusedKey = $"{prefix}.ffn_gate_up_exps.{expertIdx}.weight";
                    Tensor expertOut;
                    if (_weights.ContainsKey(fusedKey) || _quantWeights.ContainsKey(fusedKey))
                    {
                        expertOut = FFNGelu(tokenInput, fusedKey, downKey, 1);
                    }
                    else if ((_weights.ContainsKey(gateKey) || _quantWeights.ContainsKey(gateKey)) &&
                             (_weights.ContainsKey(downKey) || _quantWeights.ContainsKey(downKey)))
                    {
                        using var gateOut = LinearForward(tokenInput, gateKey);
                        using var upOut = LinearForward(tokenInput, upKey);
                        Ops.GELUMul(gateOut, gateOut, upOut);
                        expertOut = LinearForward(gateOut, downKey);
                    }
                    else
                    {
                        continue;
                    }

                    // Apply per-expert down scale if present
                    string scaleKey = $"{prefix}.ffn_down_exps.scale";
                    if (!_weights.ContainsKey(scaleKey))
                        scaleKey = $"{prefix}.ffn_gate_inp.per_expert_scale";
                    if (_weights.TryGetValue(scaleKey, out var scaleTensor))
                    {
                        float expertScale = scaleTensor.GetElementAsFloat(expertIdx);
                        Ops.Mul(expertOut, expertOut, expertScale);
                    }

                    // Accumulate weighted expert output
                    float* expertPtr = GetFloatPtr(expertOut);
                    for (int d = 0; d < hiddenDim; d++)
                        outputPtr[s * hiddenDim + d] += weight * expertPtr[d];
                    expertOut.Dispose();
                }
            }

            tokenInput.Dispose();
            return output;
        }

        private unsafe (float[] routingWeights, int[] selectedExperts) MoERoute(
            Tensor input, string prefix, int seqLen)
        {
            int hiddenDim = (int)input.Sizes[1];

            // Unweighted RMSNorm
            using var normed = new Tensor(_allocator, DType.Float32, input.Sizes);
            Ops.Copy(normed, input);
            ApplyUnweightedRMSNorm(normed, 1, hiddenDim, seqLen);

            // Scale by 1/sqrt(hidden_size)
            float invSqrtHidden = 1f / MathF.Sqrt(hiddenDim);
            Ops.Mul(normed, normed, invSqrtHidden);

            // Multiply by learned scale parameter
            string scaleKey = $"{prefix}.ffn_gate_inp.scale";
            if (_weights.TryGetValue(scaleKey, out var scaleTensor))
            {
                // Element-wise multiply per hidden dim
                float* nPtr = GetFloatPtr(normed);
                float* sPtr = GetFloatPtr(scaleTensor);
                int scaleDim = (int)scaleTensor.ElementCount();
                for (int s = 0; s < seqLen; s++)
                    for (int d = 0; d < Math.Min(hiddenDim, scaleDim); d++)
                        nPtr[s * hiddenDim + d] *= sPtr[d];
            }

            // Project to expert logits
            using var expertScores = LinearForward(normed, $"{prefix}.ffn_gate_inp.weight");

            // Softmax over experts + TopK selection
            float* scoresPtr = GetFloatPtr(expertScores);
            int numExperts = (int)expertScores.Sizes[1];

            float[] routingWeights = new float[seqLen * _numExpertsUsed];
            int[] selectedExperts = new int[seqLen * _numExpertsUsed];

            for (int s = 0; s < seqLen; s++)
            {
                float* row = scoresPtr + s * numExperts;

                // Softmax
                float maxVal = float.NegativeInfinity;
                for (int i = 0; i < numExperts; i++)
                    if (row[i] > maxVal) maxVal = row[i];
                float sumExp = 0;
                for (int i = 0; i < numExperts; i++)
                {
                    row[i] = MathF.Exp(row[i] - maxVal);
                    sumExp += row[i];
                }
                for (int i = 0; i < numExperts; i++)
                    row[i] /= sumExp;

                // TopK
                for (int k = 0; k < _numExpertsUsed; k++)
                {
                    int bestIdx = 0;
                    float bestVal = float.NegativeInfinity;
                    for (int i = 0; i < numExperts; i++)
                    {
                        bool alreadySelected = false;
                        for (int j = 0; j < k; j++)
                            if (selectedExperts[s * _numExpertsUsed + j] == i) { alreadySelected = true; break; }
                        if (!alreadySelected && row[i] > bestVal)
                        {
                            bestVal = row[i];
                            bestIdx = i;
                        }
                    }
                    selectedExperts[s * _numExpertsUsed + k] = bestIdx;
                    routingWeights[s * _numExpertsUsed + k] = row[bestIdx];
                }

                // Renormalize selected routing weights
                float selectedSum = 0;
                for (int k = 0; k < _numExpertsUsed; k++)
                    selectedSum += routingWeights[s * _numExpertsUsed + k];
                if (selectedSum > 0)
                    for (int k = 0; k < _numExpertsUsed; k++)
                        routingWeights[s * _numExpertsUsed + k] /= selectedSum;
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

                bool hasV = _quantWeights.ContainsKey(vName) || _weights.ContainsKey(vName);

                if (_quantWeights.TryGetValue(qName, out var qw) &&
                    _quantWeights.TryGetValue(kName, out var kw))
                {
                    QuantizedWeight vw = null;
                    bool hasQuantV = _quantWeights.TryGetValue(vName, out vw);

                    if (qw.GgmlType == kw.GgmlType && qw.Ne0 == kw.Ne0 &&
                        (!hasQuantV || (vw.GgmlType == kw.GgmlType && vw.Ne0 == kw.Ne0)))
                    {
                        long vBytes = hasQuantV ? vw.RawBytes : kw.RawBytes;
                        long vNe1 = hasQuantV ? vw.Ne1 : kw.Ne1;
                        long totalBytes = qw.RawBytes + kw.RawBytes + vBytes;
                        IntPtr fusedPtr = QuantizedWeight.AllocateBuffer(totalBytes);
                        Buffer.MemoryCopy(qw.Data.ToPointer(), fusedPtr.ToPointer(), totalBytes, qw.RawBytes);
                        Buffer.MemoryCopy(kw.Data.ToPointer(), (fusedPtr + (int)qw.RawBytes).ToPointer(), totalBytes - qw.RawBytes, kw.RawBytes);
                        if (hasQuantV)
                            Buffer.MemoryCopy(vw.Data.ToPointer(), (fusedPtr + (int)(qw.RawBytes + kw.RawBytes)).ToPointer(), vBytes, vBytes);
                        else
                            Buffer.MemoryCopy(kw.Data.ToPointer(), (fusedPtr + (int)(qw.RawBytes + kw.RawBytes)).ToPointer(), vBytes, kw.RawBytes);
                        _quantWeights[qkvName] = new QuantizedWeight(fusedPtr, totalBytes, qw.GgmlType, qw.Ne0, qw.Ne1 + kw.Ne1 + vNe1);
                        _quantWeights.Remove(qName); qw.Dispose();
                        _quantWeights.Remove(kName); kw.Dispose();
                        if (hasQuantV) { _quantWeights.Remove(vName); vw.Dispose(); }
                        fused++;
                    }
                }
                else if (_weights.TryGetValue(qName, out var qf) &&
                         _weights.TryGetValue(kName, out var kf))
                {
                    Tensor vf = null;
                    bool hasF32V = _weights.TryGetValue(vName, out vf);

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
                        long totalBytes = gw.RawBytes + uw.RawBytes;
                        IntPtr fusedPtr = QuantizedWeight.AllocateBuffer(totalBytes);
                        Buffer.MemoryCopy(gw.Data.ToPointer(), fusedPtr.ToPointer(), totalBytes, gw.RawBytes);
                        Buffer.MemoryCopy(uw.Data.ToPointer(), (fusedPtr + (int)gw.RawBytes).ToPointer(), totalBytes - gw.RawBytes, uw.RawBytes);
                        _quantWeights[fusedName] = new QuantizedWeight(fusedPtr, totalBytes, gw.GgmlType, gw.Ne0, gw.Ne1 + uw.Ne1);
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

        #endregion

        private Tensor FFNGelu(Tensor input, string gateUpWeightName, string downWeightName, int seqLen)
        {
            Tensor gateUp = LinearForward(input, gateUpWeightName);
            int halfDim = (int)(gateUp.Sizes[1] / 2);

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

            Ops.GELUMul(gate, gate, up);
            up.Dispose();

            Tensor down = LinearForward(gate, downWeightName);
            gate.Dispose();
            return down;
        }

        #region Attention

        private Tensor Attention(Tensor input, int layer, string prefix, int seqLen, int startPos, bool isShared, HashSet<int> exceptPositions = null)
        {
            long t0 = Stopwatch.GetTimestamp();
            bool isLocal = IsLocalLayer(layer);
            int hd = HeadDimForLayer(layer);
            int kvHeads = KVHeadsForLayer(layer);

            int qDim = Config.NumHeads * hd;
            int kDim = kvHeads * hd;

            Tensor q, k = null, v = null;
            string qkvName = $"{prefix}.attn_qkv.weight";
            bool useFusedQKV = !isShared && (_quantWeights.ContainsKey(qkvName) || _weights.ContainsKey(qkvName));

            if (useFusedQKV)
            {
                Tensor qkv = LinearForward(input, qkvName);
                int vDim = (int)qkv.Sizes[1] - qDim - kDim;
                if (seqLen == 1)
                {
                    q = qkv.Narrow(1, 0, qDim);
                    k = qkv.Narrow(1, qDim, kDim);
                    v = qkv.Narrow(1, qDim + kDim, vDim);
                }
                else
                {
                    using (var qView = qkv.Narrow(1, 0, qDim)) q = Ops.NewContiguous(qView);
                    using (var kView = qkv.Narrow(1, qDim, kDim)) k = Ops.NewContiguous(kView);
                    using (var vView = qkv.Narrow(1, qDim + kDim, vDim)) v = Ops.NewContiguous(vView);
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

            // Apply NeoX-style RoPE
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

            int totalSeqLen = startPos + seqLen;
            Tensor result;

            if (seqLen == 1)
            {
                if (!isShared)
                {
                    int cachePos = isLocal ? (startPos % _kvCacheSize[layer]) : startPos;
                    CopyToCacheDecode(_kvCacheK[layer], k, _kvCacheV[layer], v,
                        kvHeads, hd, cachePos);
                }

                int kvCacheLayer = _kvDonorMap.TryGetValue(layer, out int donor) ? donor : layer;
                int cacheLen = _kvCacheSize[kvCacheLayer];

                if (isLocal)
                {
                    int attendLen = Math.Min(totalSeqLen, _slidingWindow);
                    result = new Tensor(_allocator, DType.Float32, 1, Config.NumHeads * hd);
                    AttentionDecodeCircular(q, _kvCacheK[kvCacheLayer], _kvCacheV[kvCacheLayer], result,
                        Config.NumHeads, kvHeads, hd, hd,
                        startPos, attendLen, cacheLen, 1f);
                }
                else
                {
                    result = new Tensor(_allocator, DType.Float32, 1, Config.NumHeads * hd);
                    AttentionDecodeWithWindow(q, _kvCacheK[kvCacheLayer], _kvCacheV[kvCacheLayer], result,
                        Config.NumHeads, kvHeads, hd, hd,
                        0, totalSeqLen, 1f);
                }
            }
            else
            {
                Tensor kHeadsForAttn = null, vHeadsForAttn = null;
                if (!isShared)
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

                Tensor qHeads = ReshapeToHeads(q, Config.NumHeads, seqLen, hd);

                int kvCacheLayer = _kvDonorMap.TryGetValue(layer, out int donor2) ? donor2 : layer;
                int groupSize = Config.NumHeads / kvHeads;

                int kvLen;
                Tensor kExpanded, vExpanded;
                if (kHeadsForAttn != null && (startPos == 0 || isLocal))
                {
                    // Non-shared SWA layers bypass the circular cache for attention;
                    // use freshly computed K/V in correct sequential order.
                    kvLen = seqLen;
                    kExpanded = ExpandKVHeads(kHeadsForAttn, groupSize, seqLen);
                    vExpanded = ExpandKVHeads(vHeadsForAttn, groupSize, seqLen);
                }
                else if (isLocal && startPos > 0 && _prefillSWAKV != null
                         && _prefillSWAKV.TryGetValue(kvCacheLayer, out var donorKV))
                {
                    // Shared SWA layer: reuse donor's saved K/V in sequential order
                    kvLen = seqLen;
                    kExpanded = ExpandKVHeads(donorKV.k, groupSize, seqLen);
                    vExpanded = ExpandKVHeads(donorKV.v, groupSize, seqLen);
                }
                else
                {
                    int cacheLen = _kvCacheSize[kvCacheLayer];
                    kvLen = Math.Min(totalSeqLen, cacheLen);
                    kExpanded = ExpandKVHeads(_kvCacheK[kvCacheLayer], groupSize, kvLen);
                    vExpanded = ExpandKVHeads(_kvCacheV[kvCacheLayer], groupSize, kvLen);
                }

                // Save non-shared SWA K/V for shared layers that use this as donor
                if (isLocal && startPos > 0 && kHeadsForAttn != null
                    && _swaKVDonorLayers.Contains(layer) && _prefillSWAKV != null)
                {
                    _prefillSWAKV[layer] = (kHeadsForAttn, vHeadsForAttn);
                }
                else
                {
                    kHeadsForAttn?.Dispose();
                    vHeadsForAttn?.Dispose();
                }

                using var kT = kExpanded.Transpose(1, 2);
                var scores = new Tensor(_allocator, DType.Float32, Config.NumHeads, seqLen, kvLen);
                Ops.AddmmBatch(scores, 0, scores, 1f, qHeads, kT);
                qHeads.Dispose();
                kExpanded.Dispose();

                int windowSize = isLocal ? _slidingWindow : 0;
                HashSet<int> maskExcept = isLocal ? null : exceptPositions;
                ApplyCausalMask(scores, seqLen, kvLen, windowSize, maskExcept);
                Ops.Softmax(scores, scores);

                var attnOut = new Tensor(_allocator, DType.Float32, Config.NumHeads, seqLen, hd);
                Ops.AddmmBatch(attnOut, 0, attnOut, 1.0f, scores, vExpanded);
                scores.Dispose();
                vExpanded.Dispose();

                result = ReshapeFromHeadsEx(attnOut, Config.NumHeads, seqLen, hd);
                attnOut.Dispose();
            }

            q.Dispose();
            k?.Dispose();
            v?.Dispose();

            _attnTicks += Stopwatch.GetTimestamp() - t0;

            using (result)
            {
                return LinearForward(result, $"{prefix}.attn_output.weight");
            }
        }

        private Tensor ApplyBatchRMSNorm(Tensor data, string weightName, int numHeads, int seqLen, int headDim)
        {
            var alpha = _weights[weightName];
            using var reshaped = data.View(seqLen * numHeads, headDim);
            Tensor normed = Ops.RMSNorm(null, reshaped, alpha, null, Config.Eps);
            data.Dispose();
            Tensor flat = normed.View(seqLen, numHeads * headDim);
            normed.Dispose();
            return flat;
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
            int totalRows = seqLen * numHeads;
            int[] positions = new int[totalRows];
            for (int s = 0; s < seqLen; s++)
                for (int h = 0; h < numHeads; h++)
                    positions[s * numHeads + h] = startPos + s;
            using var posTensor = CreateIntTensor(positions, totalRows);

            using var reshaped = data.View(1, seqLen, numHeads, headDim);
            Tensor result = Ops.RoPEEx(
                null, reshaped, posTensor, headDim, 2, 0,
                ropeBase, 1.0f,
                0.0f, 1.0f, 0.0f, 0.0f);

            data.Dispose();
            Tensor flat = result.View(seqLen, numHeads * headDim);
            result.Dispose();
            return flat;
        }

        private unsafe void ApplyNeoXRoPEDecode(Tensor data, int numHeads, int headDim, int position, float[] freqs)
        {
            float* ptr = GetFloatPtr(data);
            int ropeHalf = freqs.Length;

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
            float* ptr = GetFloatPtr(data);
            int ropeHalf = freqs.Length;

            for (int s = 0; s < seqLen; s++)
            {
                int position = startPos + s;
                for (int h = 0; h < numHeads; h++)
                {
                    float* head = ptr + ((long)s * numHeads + h) * headDim;
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

        private unsafe void AttentionDecodeCircular(Tensor q, Tensor kCache, Tensor vCache,
            Tensor result, int numHeads, int numKVHeads, int keyDim, int valDim,
            int currentPos, int attendLen, int cacheSize, float scale)
        {
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

        private unsafe void CopyToCacheCircular(Tensor cache, Tensor src, int startPos, int seqLen, int cacheSize)
        {
            float* srcPtr = GetFloatPtr(src);
            float* cachePtr = GetFloatPtr(cache);
            int numHeads = (int)cache.Sizes[0];
            int headDim = (int)cache.Sizes[2];
            int headBytes = headDim * sizeof(float);

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

            InvalidateTensorDeviceCache(cache);
        }

        private unsafe void ApplyCausalMask(Tensor scores, int queryLen, int totalKVLen, int windowSize,
            HashSet<int> exceptPositions = null)
        {
            int startPos = totalKVLen - queryLen;

            if (exceptPositions != null && exceptPositions.Count > 0)
            {
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
                        float* row = headScores + q * totalKVLen;
                        for (int kv = queryAbsPos + 1; kv < totalKVLen; kv++)
                        {
                            if (!queryIsExcept && !exceptPositions.Contains(kv))
                                row[kv] = float.NegativeInfinity;
                        }
                    }
                }
                InvalidateTensorDeviceCache(scores);
            }
            else
            {
                Ops.AddCausalMask(scores, queryLen, startPos, float.NegativeInfinity);
            }

            if (windowSize > 0)
            {
                float* sPtr = GetFloatPtr(scores);
                int numHeads = (int)scores.Sizes[0];
                int rowStride = queryLen * totalKVLen;

                for (int h = 0; h < numHeads; h++)
                {
                    float* headScores = sPtr + h * rowStride;
                    for (int q = 0; q < queryLen; q++)
                    {
                        int queryPos = startPos + q;
                        int windowStart = queryPos - windowSize + 1;
                        if (windowStart > 0)
                        {
                            float* row = headScores + q * totalKVLen;
                            for (int kv = 0; kv < windowStart; kv++)
                                row[kv] = float.NegativeInfinity;
                        }
                    }
                }
                InvalidateTensorDeviceCache(scores);
            }
        }

        #endregion

        public override void Dispose()
        {
            _onesForVNorm?.Dispose();
            _visionEncoder?.Dispose();
            _audioEncoder?.Dispose();
            foreach (var (emb, _) in _pendingVisionEmbeddingsList)
                emb?.Dispose();
            _pendingVisionEmbeddingsList.Clear();
            foreach (var (emb, _) in _pendingAudioEmbeddingsList)
                emb?.Dispose();
            _pendingAudioEmbeddingsList.Clear();
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
