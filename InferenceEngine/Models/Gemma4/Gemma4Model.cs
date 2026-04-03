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

namespace InferenceEngine
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

        private int _pleDim;

        private Tensor[] _kvCacheK;
        private Tensor[] _kvCacheV;

        private float[] _layerScalars;
        private bool _hasTiedOutput;

        private Tensor _onesForVNorm;

        private int _numExperts;
        private int _numExpertsUsed;

        private Gemma4VisionEncoder _visionEncoder;
        private Gemma4AudioEncoder _audioEncoder;
        private List<(Tensor embeddings, int position)> _pendingVisionEmbeddingsList = new();
        private Tensor _pendingAudioEmbeddings;
        private int _audioInsertPosition = -1;

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
            _pendingAudioEmbeddings?.Dispose();
            _pendingAudioEmbeddings = embeddings;
            _audioInsertPosition = insertPosition;
        }

        public Gemma4Model(string ggufPath, BackendType backend) : base(ggufPath, backend)
        {
            Config = new ModelConfig { Architecture = _gguf.GetString("general.architecture") };
            ParseBaseConfig();

            string arch = Config.Architecture;

            _slidingWindowPattern = _gguf.GetBoolArray($"{arch}.attention.sliding_window_pattern");
            _slidingWindow = (int)_gguf.GetUint32($"{arch}.attention.sliding_window", 512);

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
            FuseGateUpWeights();
            PrecomputeRoPE();
            InitKVCache(8192);
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
            _kvCacheK = new Tensor[Config.NumLayers];
            _kvCacheV = new Tensor[Config.NumLayers];

            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_kvDonorMap.ContainsKey(l)) continue;

                int kvHeads = KVHeadsForLayer(l);
                int hd = HeadDimForLayer(l);
                _kvCacheK[l] = new Tensor(_allocator, DType.Float32, kvHeads, maxSeqLen, hd);
                _kvCacheV[l] = new Tensor(_allocator, DType.Float32, kvHeads, maxSeqLen, hd);
                Ops.Fill(_kvCacheK[l], 0f);
                Ops.Fill(_kvCacheV[l], 0f);
            }

            foreach (var kv in _kvDonorMap)
            {
                _kvCacheK[kv.Key] = _kvCacheK[kv.Value];
                _kvCacheV[kv.Key] = _kvCacheV[kv.Value];
            }
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
                cleared.Add(l);
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

            if (_pendingVisionEmbeddingsList.Count > 0)
            {
                foreach (var (emb, pos) in _pendingVisionEmbeddingsList)
                {
                    InjectVisionEmbeddings(hidden, emb, pos);
                    emb.Dispose();
                }
                _pendingVisionEmbeddingsList.Clear();
            }

            if (_pendingAudioEmbeddings != null && _audioInsertPosition >= 0)
            {
                InjectVisionEmbeddings(hidden, _pendingAudioEmbeddings, _audioInsertPosition);
                _pendingAudioEmbeddings.Dispose();
                _pendingAudioEmbeddings = null;
                _audioInsertPosition = -1;
            }

            Tensor perLayerInputs = null;
            if (_pleDim > 0)
                perLayerInputs = ComputePLE(tokens, hidden, seqLen);

            for (int l = 0; l < Config.NumLayers; l++)
            {
                Tensor perLayerInput = null;
                if (perLayerInputs != null)
                    perLayerInput = ExtractPerLayerSlice(perLayerInputs, l, seqLen);

                bool isShared = _kvDonorMap.ContainsKey(l);
                hidden = TransformerBlock(hidden, l, seqLen, startPos, isShared, perLayerInput);

                perLayerInput?.Dispose();
            }

            perLayerInputs?.Dispose();

            using var normed = RMSNormOp(hidden, "output_norm.weight");
            hidden.Dispose();

            t0 = Stopwatch.GetTimestamp();
            string outputWeight = _hasTiedOutput ? "token_embd.weight" : "output.weight";
            Tensor logitsTensor = LinearForward(normed, outputWeight);
            _lmHeadTicks += Stopwatch.GetTimestamp() - t0;

            if (_finalLogitSoftcap > 0f)
                ApplyLogitSoftcap(logitsTensor);

            t0 = Stopwatch.GetTimestamp();
            int lastTokenIdx = seqLen - 1;
            if (_logitsBuffer == null || _logitsBuffer.Length != Config.VocabSize)
                _logitsBuffer = new float[Config.VocabSize];

            if (seqLen == 1)
            {
                unsafe
                {
                    float* ptr = GetFloatPtr(logitsTensor);
                    fixed (float* dst = _logitsBuffer)
                        Buffer.MemoryCopy(ptr, dst, Config.VocabSize * 4, Config.VocabSize * 4);
                }
            }
            else
            {
                using var lastRow = logitsTensor.Narrow(0, lastTokenIdx, 1);
                using var contiguous = Ops.NewContiguous(lastRow);
                unsafe
                {
                    float* ptr = GetFloatPtr(contiguous);
                    fixed (float* dst = _logitsBuffer)
                        Buffer.MemoryCopy(ptr, dst, Config.VocabSize * 4, Config.VocabSize * 4);
                }
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

        #region PLE (Per-Layer Embedding)

        private unsafe Tensor ComputePLE(int[] tokens, Tensor hiddenState, int seqLen)
        {
            int totalPleDim = _pleDim * Config.NumLayers;

            Tensor pleTokenEmb = null;
            if (_quantWeights.TryGetValue("per_layer_token_embd.weight", out var pleQw))
            {
                pleTokenEmb = new Tensor(_allocator, DType.Float32, seqLen, totalPleDim);
                float* dstPtr = GetFloatPtr(pleTokenEmb);
                long rowBytes = NativeDequant.RowSize(pleQw.GgmlType, pleQw.Ne0);
                float[] rowBuf = new float[totalPleDim];

                for (int i = 0; i < seqLen; i++)
                {
                    long srcOffset = (long)tokens[i] * rowBytes;
                    IntPtr rowPtr = (IntPtr)((byte*)pleQw.Data.ToPointer() + srcOffset);
                    NativeDequant.DequantizeToFloat32(pleQw.GgmlType, rowPtr, rowBuf, 0, pleQw.Ne0);
                    fixed (float* src = rowBuf)
                        Buffer.MemoryCopy(src, dstPtr + (long)i * totalPleDim,
                            totalPleDim * sizeof(float), totalPleDim * sizeof(float));
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
            string downKey = $"blk.{layer}.ffn_down_exps.weight";
            return (_weights.ContainsKey(routerKey) || _quantWeights.ContainsKey(routerKey)) &&
                   (_weights.ContainsKey(downKey) || _quantWeights.ContainsKey(downKey));
        }

        private Tensor TransformerBlock(Tensor hidden, int layer, int seqLen, int startPos,
            bool isShared, Tensor perLayerInput)
        {
            string prefix = $"blk.{layer}";

            using var attnNormed = RMSNormOp(hidden, $"{prefix}.attn_norm.weight");

            using var attnOut = Attention(attnNormed, layer, prefix, seqLen, startPos, isShared);

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
            // Pre-norm for MoE input
            string moeNormKey = $"{prefix}.pre_ffw_norm_2.weight";
            if (!_weights.ContainsKey(moeNormKey))
                moeNormKey = $"{prefix}.ffn_pre_norm_2.weight";
            using var moeInput = RMSNormOp(hiddenState, moeNormKey);

            // Router: RMSNorm(no weight) -> scale -> project -> softmax -> topk
            var (routingWeights, selectedExperts) = MoERoute(moeInput, prefix, seqLen);

            // Expert computation: for each token, run selected experts
            int hiddenDim = (int)moeInput.Sizes[1];
            var output = new Tensor(_allocator, DType.Float32, seqLen, hiddenDim);
            Ops.Fill(output, 0f);

            float* inputPtr = GetFloatPtr(moeInput);
            float* outputPtr = GetFloatPtr(output);

            for (int s = 0; s < seqLen; s++)
            {
                for (int e = 0; e < _numExpertsUsed; e++)
                {
                    int expertIdx = selectedExperts[s * _numExpertsUsed + e];
                    float weight = routingWeights[s * _numExpertsUsed + e];

                    // Create single-token input for this expert
                    using var tokenInput = new Tensor(_allocator, DType.Float32, 1, hiddenDim);
                    float* tokenPtr = GetFloatPtr(tokenInput);
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

        #endregion

        private Tensor FFNGelu(Tensor input, string gateUpWeightName, string downWeightName, int seqLen)
        {
            Tensor gateUp = LinearForward(input, gateUpWeightName);
            int intermSize = Config.IntermediateSize;
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

            Ops.GELUMul(gate, gate, up);
            up.Dispose();

            Tensor down = LinearForward(gate, downWeightName);
            gate.Dispose();
            return down;
        }

        #region Attention

        private Tensor Attention(Tensor input, int layer, string prefix, int seqLen, int startPos, bool isShared)
        {
            long t0 = Stopwatch.GetTimestamp();
            bool isLocal = IsLocalLayer(layer);
            int hd = HeadDimForLayer(layer);
            int kvHeads = KVHeadsForLayer(layer);

            Tensor q = LinearForward(input, $"{prefix}.attn_q.weight");

            // QK norm: reshape to [seqLen*numHeads, hd], normalize, reshape back
            if (seqLen == 1)
            {
                RMSNormInPlace(q, _weights[$"{prefix}.attn_q_norm.weight"], Config.NumHeads, hd, Config.Eps);
            }
            else
            {
                q = ApplyBatchRMSNorm(q, $"{prefix}.attn_q_norm.weight", Config.NumHeads, seqLen, hd);
            }

            Tensor k = null, v = null;
            if (!isShared)
            {
                k = LinearForward(input, $"{prefix}.attn_k.weight");

                // V: use attn_v if present, otherwise K=V (raw K before K norm)
                bool hasVWeight = _weights.ContainsKey($"{prefix}.attn_v.weight") ||
                                  _quantWeights.ContainsKey($"{prefix}.attn_v.weight");
                if (hasVWeight)
                {
                    v = LinearForward(input, $"{prefix}.attn_v.weight");
                }
                else
                {
                    // K=V: use raw K projection as V (before K norm)
                    v = new Tensor(_allocator, DType.Float32, k.Sizes);
                    Ops.Copy(v, k);
                }

                // K norm
                if (seqLen == 1)
                {
                    RMSNormInPlace(k, _weights[$"{prefix}.attn_k_norm.weight"], kvHeads, hd, Config.Eps);
                }
                else
                {
                    k = ApplyBatchRMSNorm(k, $"{prefix}.attn_k_norm.weight", kvHeads, seqLen, hd);
                }

                // V norm (unweighted RMSNorm)
                ApplyUnweightedRMSNorm(v, kvHeads, hd, seqLen);
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
                    CopyToCacheDecode(_kvCacheK[layer], k, _kvCacheV[layer], v,
                        kvHeads, hd, startPos);
                }

                int kvCacheLayer = _kvDonorMap.TryGetValue(layer, out int donor) ? donor : layer;
                int attendLen = isLocal ? Math.Min(totalSeqLen, _slidingWindow) : totalSeqLen;
                int attendStart = totalSeqLen - attendLen;

                result = new Tensor(_allocator, DType.Float32, 1, Config.NumHeads * hd);
                AttentionDecodeWithWindow(q, _kvCacheK[kvCacheLayer], _kvCacheV[kvCacheLayer], result,
                    Config.NumHeads, kvHeads, hd, hd,
                    attendStart, totalSeqLen, 1f);
            }
            else
            {
                if (!isShared)
                {
                    Tensor kHeads = ReshapeToHeads(k, kvHeads, seqLen, hd);
                    Tensor vHeads = ReshapeToHeads(v, kvHeads, seqLen, hd);
                    CopyToCache(_kvCacheK[layer], kHeads, startPos, seqLen);
                    CopyToCache(_kvCacheV[layer], vHeads, startPos, seqLen);
                    kHeads.Dispose();
                    vHeads.Dispose();
                }

                Tensor qHeads = ReshapeToHeads(q, Config.NumHeads, seqLen, hd);

                int kvCacheLayer = _kvDonorMap.TryGetValue(layer, out int donor2) ? donor2 : layer;
                int groupSize = Config.NumHeads / kvHeads;
                Tensor kExpanded = ExpandKVHeads(_kvCacheK[kvCacheLayer], groupSize, totalSeqLen);
                Tensor vExpanded = ExpandKVHeads(_kvCacheV[kvCacheLayer], groupSize, totalSeqLen);

                using var kT = kExpanded.Transpose(1, 2);
                var scores = new Tensor(_allocator, DType.Float32, Config.NumHeads, seqLen, totalSeqLen);
                Ops.AddmmBatch(scores, 0, scores, 1f, qHeads, kT);
                qHeads.Dispose();
                kExpanded.Dispose();

                int windowSize = isLocal ? _slidingWindow : 0;
                ApplyCausalMask(scores, seqLen, totalSeqLen, windowSize);
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
            int halfDim = headDim / 2;

            for (int h = 0; h < numHeads; h++)
            {
                float* head = ptr + h * headDim;
                int limit = Math.Min(halfDim, freqs.Length);
                for (int j = 0; j < limit; j++)
                {
                    float angle = position * freqs[j];
                    float cos = MathF.Cos(angle);
                    float sin = MathF.Sin(angle);
                    float x0 = head[j];
                    float x1 = head[j + halfDim];
                    head[j] = x0 * cos - x1 * sin;
                    head[j + halfDim] = x0 * sin + x1 * cos;
                }
            }
        }

        private unsafe Tensor ApplyNeoXRoPEPrefill(Tensor data, int numHeads, int headDim,
            int seqLen, int startPos, float[] freqs)
        {
            float* ptr = GetFloatPtr(data);
            int halfDim = headDim / 2;
            int limit = Math.Min(halfDim, freqs.Length);

            for (int s = 0; s < seqLen; s++)
            {
                int position = startPos + s;
                for (int h = 0; h < numHeads; h++)
                {
                    float* head = ptr + ((long)s * numHeads + h) * headDim;
                    for (int j = 0; j < limit; j++)
                    {
                        float angle = position * freqs[j];
                        float cos = MathF.Cos(angle);
                        float sin = MathF.Sin(angle);
                        float x0 = head[j];
                        float x1 = head[j + halfDim];
                        head[j] = x0 * cos - x1 * sin;
                        head[j + halfDim] = x0 * sin + x1 * cos;
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

        private unsafe void ApplyCausalMask(Tensor scores, int queryLen, int totalKVLen, int windowSize)
        {
            int startPos = totalKVLen - queryLen;
            Ops.AddCausalMask(scores, queryLen, startPos, float.NegativeInfinity);

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
            _pendingAudioEmbeddings?.Dispose();
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
