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
using TensorSharp;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    /// <summary>
    /// GPT OSS (Mixture-of-Experts) transformer model.
    /// Key features:
    ///   - MoE FFN with TopK routing + softmax on selected experts
    ///   - Alternating SWA (even layers) / full causal (odd layers) attention
    ///   - Attention sinks for SWA layers
    ///   - SiLU with alpha scaling and clamping (SiLUAlphaLimit)
    ///   - RoPE NeoX with yarn scaling
    ///   - Bias on all attention and FFN projections
    /// Optimizations:
    ///   - Fused QKV projection (3 matmuls -> 1)
    ///   - Expert batching in MoE (N*K matmuls -> up to numExperts batched matmuls)
    ///   - Pre-computed weight name strings (zero allocation per forward)
    ///   - Cached attention sinks arrays
    ///   - SIMD-vectorized bias addition and activation
    /// </summary>
    public class GptOssModel : ModelBase
    {
        private Tensor[] _kvCacheK;
        private Tensor[] _kvCacheV;
        private int _numExperts;
        private int _numExpertsUsed;
        private int _slidingWindow;
        private int _expertFfnLength;

        private const float SiluAlpha = 1.702f;
        private const float SiluLimit = 7.0f;

        private string[][] _layerNames;
        private string[][][] _expertNames;
        private float[][] _layerSinks;
        private int _qDim, _kDim;
        private bool _isQkvFused;

        private int[] _moeExpertCounts;
        private int[] _moeExpertOffsets;
        private int[] _moeTokenMap;
        private float[] _moeWeightMap;

        public GptOssModel(string ggufPath, BackendType backend)
            : base(ggufPath, backend)
        {
            string arch = _gguf.GetString("general.architecture") ?? "gpt-oss";
            Config = new ModelConfig { Architecture = arch };
            ParseBaseConfig();

            _numExperts = (int)_gguf.GetUint32($"{arch}.expert_count", 0);
            _numExpertsUsed = (int)_gguf.GetUint32($"{arch}.expert_used_count", 0);
            _slidingWindow = (int)_gguf.GetUint32($"{arch}.attention.sliding_window", 128);
            _expertFfnLength = (int)_gguf.GetUint32($"{arch}.expert_feed_forward_length", 0);

            Config.NumExperts = _numExperts;
            Config.NumExpertsUsed = _numExpertsUsed;
            Config.SlidingWindow = _slidingWindow;
            Config.OriginalContextLength = (int)_gguf.GetUint32($"{arch}.rope.scaling.original_context_length", 4096);

            ParseTokenizer();

            Console.WriteLine($"Model: {arch}, Layers={Config.NumLayers}, Hidden={Config.HiddenSize}, " +
                $"Heads={Config.NumHeads}, KVHeads={Config.NumKVHeads}, HeadDim={Config.HeadDim}, Vocab={Config.VocabSize}");
            Console.WriteLine($"RoPE base={Config.RopeBase}, scale={Config.RopeScale}, eps={Config.Eps}");
            Console.WriteLine($"MoE: {_numExperts} experts, {_numExpertsUsed} used, " +
                $"SlidingWindow={_slidingWindow}, ExpertFFN={_expertFfnLength}");

            LoadWeights();
            SplitExpertBiases();
            FuseExpertGateUpWeights();
            FuseQKVWeights();
            PrepareCudaQuantizedWeightsForInference();
            InitKVCache(ResolveConfiguredContextLength());
            PrecomputeConstants();
        }

        #region Weight Fusion and Pre-computation

        private void SplitExpertBiases()
        {
            int split = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                foreach (string kind in new[] { "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps" })
                {
                    string biasName = $"blk.{l}.{kind}.bias";
                    if (!_weights.TryGetValue(biasName, out var biasTensor))
                        continue;

                    int numExp = (int)biasTensor.Sizes[0];
                    int biasDim = (int)biasTensor.Sizes[1];
                    float[] biasData = TensorToFloatArray(biasTensor);

                    for (int e = 0; e < numExp; e++)
                    {
                        float[] expertBias = new float[biasDim];
                        for (int d = 0; d < biasDim; d++)
                            expertBias[d] = biasData[e * biasDim + d];
                        _weights[$"blk.{l}.{kind}.{e}.bias"] = CreateFloatTensor(expertBias, 1, biasDim);
                    }
                    _weights.Remove(biasName);
                    biasTensor.Dispose();
                    split++;
                }
            }
            if (split > 0)
                Console.WriteLine($"  Split expert biases: {split} tensors");
        }

        private unsafe void FuseExpertGateUpWeights()
        {
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
                        _quantWeights[fusedName] = QuantizedWeight.ConcatOrCreateCopy(gw, uw);
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

                    string gateBias = $"blk.{l}.ffn_gate_exps.{e}.bias";
                    string upBias = $"blk.{l}.ffn_up_exps.{e}.bias";
                    string fusedBias = $"blk.{l}.ffn_gate_up_exps.{e}.bias";
                    if (_weights.TryGetValue(gateBias, out var gb) &&
                        _weights.TryGetValue(upBias, out var ub))
                    {
                        int gbDim = (int)gb.Sizes[1], ubDim = (int)ub.Sizes[1];
                        float[] gbData = TensorToFloatArray(gb);
                        float[] ubData = TensorToFloatArray(ub);
                        float[] fusedData = new float[gbDim + ubDim];
                        Array.Copy(gbData, 0, fusedData, 0, gbDim);
                        Array.Copy(ubData, 0, fusedData, gbDim, ubDim);
                        _weights[fusedBias] = CreateFloatTensor(fusedData, 1, gbDim + ubDim);
                        _weights.Remove(gateBias); gb.Dispose();
                        _weights.Remove(upBias); ub.Dispose();
                    }
                }
            }
            if (fused > 0)
                Console.WriteLine($"  Fused expert Gate+Up projections: {fused}");
        }

        private unsafe void FuseQKVWeights()
        {
            int fused = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
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
                    _quantWeights[qkvName] = QuantizedWeight.ConcatOrCreateCopy(qw, kw, vw);
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

                string qBias = $"blk.{l}.attn_q.bias";
                string kBias = $"blk.{l}.attn_k.bias";
                string vBias = $"blk.{l}.attn_v.bias";
                string qkvBias = $"blk.{l}.attn_qkv.bias";
                if (_weights.TryGetValue(qBias, out var qb) &&
                    _weights.TryGetValue(kBias, out var kb) &&
                    _weights.TryGetValue(vBias, out var vb))
                {
                    int qbDim = (int)qb.ElementCount();
                    int kbDim = (int)kb.ElementCount();
                    int vbDim = (int)vb.ElementCount();
                    float[] qbData = TensorToFloatArray(qb);
                    float[] kbData = TensorToFloatArray(kb);
                    float[] vbData = TensorToFloatArray(vb);
                    float[] fusedData = new float[qbDim + kbDim + vbDim];
                    Array.Copy(qbData, 0, fusedData, 0, qbDim);
                    Array.Copy(kbData, 0, fusedData, qbDim, kbDim);
                    Array.Copy(vbData, 0, fusedData, qbDim + kbDim, vbDim);
                    _weights[qkvBias] = CreateFloatTensor(fusedData, 1, qbDim + kbDim + vbDim);
                    _weights.Remove(qBias); qb.Dispose();
                    _weights.Remove(kBias); kb.Dispose();
                    _weights.Remove(vBias); vb.Dispose();
                }
            }
            if (fused > 0)
                Console.WriteLine($"  Fused projections: {fused} QKV");
        }

        private void PrecomputeConstants()
        {
            int numLayers = Config.NumLayers;
            _qDim = Config.NumHeads * Config.HeadDim;
            _kDim = Config.NumKVHeads * Config.HeadDim;

            _isQkvFused = _quantWeights.ContainsKey("blk.0.attn_qkv.weight") ||
                           _weights.ContainsKey("blk.0.attn_qkv.weight");

            _layerNames = new string[numLayers][];
            for (int l = 0; l < numLayers; l++)
            {
                string p = $"blk.{l}.";
                if (_isQkvFused)
                {
                    _layerNames[l] = new[]
                    {
                        p + "attn_norm.weight",           // 0
                        p + "attn_qkv.weight",            // 1
                        p + "attn_qkv.bias",              // 2
                        p + "attn_output.weight",          // 3
                        p + "attn_output.bias",            // 4
                        p + "post_attention_norm.weight",  // 5
                        p + "ffn_gate_inp.weight",         // 6
                        p + "ffn_gate_inp.bias",           // 7
                    };
                }
                else
                {
                    _layerNames[l] = new[]
                    {
                        p + "attn_norm.weight",            // 0
                        p + "attn_q.weight",               // 1
                        p + "attn_q.bias",                 // 2
                        p + "attn_output.weight",          // 3
                        p + "attn_output.bias",            // 4
                        p + "post_attention_norm.weight",  // 5
                        p + "ffn_gate_inp.weight",         // 6
                        p + "ffn_gate_inp.bias",           // 7
                        p + "attn_k.weight",               // 8
                        p + "attn_k.bias",                 // 9
                        p + "attn_v.weight",               // 10
                        p + "attn_v.bias",                 // 11
                    };
                }
            }

            _expertNames = new string[numLayers][][];
            for (int l = 0; l < numLayers; l++)
            {
                _expertNames[l] = new string[_numExperts][];
                string p = $"blk.{l}.";
                for (int e = 0; e < _numExperts; e++)
                {
                    _expertNames[l][e] = new[]
                    {
                        p + $"ffn_gate_up_exps.{e}.weight",  // 0
                        p + $"ffn_gate_up_exps.{e}.bias",    // 1
                        p + $"ffn_down_exps.{e}.weight",     // 2
                        p + $"ffn_down_exps.{e}.bias",       // 3
                    };
                }
            }

            _layerSinks = new float[numLayers][];
            for (int l = 0; l < numLayers; l++)
            {
                string sinksKey = $"blk.{l}.attn_sinks.weight";
                if (_weights.TryGetValue(sinksKey, out var sinksTensor))
                    _layerSinks[l] = TensorToFloatArray(sinksTensor);
            }

            int maxBatchTokens = 4096 * _numExpertsUsed;
            _moeExpertCounts = new int[_numExperts];
            _moeExpertOffsets = new int[_numExperts];
            _moeTokenMap = new int[maxBatchTokens];
            _moeWeightMap = new float[maxBatchTokens];
        }

        #endregion

        private void InitKVCache(int maxSeqLen)
        {
            _maxContextLength = maxSeqLen;
            int numKVHeads = Config.NumKVHeads;
            int headDim = Config.HeadDim;
            _kvCacheK = new Tensor[Config.NumLayers];
            _kvCacheV = new Tensor[Config.NumLayers];
            for (int l = 0; l < Config.NumLayers; l++)
            {
                _kvCacheK[l] = new Tensor(_allocator, DType.Float32, numKVHeads, maxSeqLen, headDim);
                _kvCacheV[l] = new Tensor(_allocator, DType.Float32, numKVHeads, maxSeqLen, headDim);
                InitializeCacheTensor(_kvCacheK[l]);
                InitializeCacheTensor(_kvCacheV[l]);
            }
            _cacheSeqLen = 0;
        }

        public override void ResetKVCache()
        {
            for (int l = 0; l < Config.NumLayers; l++)
            {
                ResetCacheTensor(_kvCacheK[l]);
                ResetCacheTensor(_kvCacheV[l]);
            }
            _cacheSeqLen = 0;
            _linearTicks = _attnTicks = _normTicks = _embTicks = _lmHeadTicks = _logitsCopyTicks = 0;
            _forwardCount = 0;
            _forwardSw.Reset();
        }

        public override void TruncateKVCache(int tokenCount)
        {
            base.TruncateKVCache(tokenCount);
            for (int l = 0; l < Config.NumLayers; l++)
            {
                InvalidateTensorDeviceCache(_kvCacheK[l]);
                InvalidateTensorDeviceCache(_kvCacheV[l]);
            }
        }

        public override float[] Forward(int[] tokens)
        {
            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;

            long t1 = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t1;

            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                bool isLastLayer = (layer == Config.NumLayers - 1);
                hidden = TransformerBlock(hidden, layer, seqLen, startPos, isLastLayer);
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
            _logitsBuffer = TensorToFloatArray(logitsTensor);
            _logitsCopyTicks += Stopwatch.GetTimestamp() - t3;
            logitsTensor.Dispose();

            _cacheSeqLen += seqLen;
            _forwardCount++;
            _forwardSw.Stop();
            return _logitsBuffer;
        }

        private Tensor TransformerBlock(Tensor hidden, int layer, int seqLen, int startPos, bool isLastLayer)
        {
            string[] wn = _layerNames[layer];

            Tensor normed = RMSNormOp(hidden, wn[0]);
            Tensor attnOut = Attention(normed, layer, wn, seqLen, startPos);
            normed.Dispose();
            Ops.Add(hidden, hidden, attnOut);
            attnOut.Dispose();

            int moeSeqLen = seqLen;
            Tensor moeInput = hidden;
            if (isLastLayer && seqLen > 1)
            {
                using var lastRow = hidden.Narrow(0, seqLen - 1, 1);
                moeInput = Ops.NewContiguous(lastRow);
                moeSeqLen = 1;
            }

            Tensor normed2 = RMSNormOp(moeInput, wn[5]);
            Tensor moeOut = MoEForward(normed2, layer, moeSeqLen);
            normed2.Dispose();

            if (isLastLayer && seqLen > 1)
            {
                unsafe
                {
                    float* hidPtr = GetFloatPtr(hidden);
                    float* moePtr = GetFloatPtr(moeOut);
                    int dim = Config.HiddenSize;
                    long offset = (long)(seqLen - 1) * dim;
                    for (int d = 0; d < dim; d++)
                        hidPtr[offset + d] += moePtr[d];
                }
                moeInput.Dispose();
            }
            else
            {
                Ops.Add(hidden, hidden, moeOut);
            }
            moeOut.Dispose();

            return hidden;
        }

        #region Attention

        private Tensor Attention(Tensor input, int layer, string[] wn, int seqLen, int startPos)
        {
            int numHeads = Config.NumHeads;
            int numKVHeads = Config.NumKVHeads;
            int headDim = Config.HeadDim;
            int totalSeqLen = startPos + seqLen;
            float scale = 1.0f / MathF.Sqrt(headDim);
            bool isSWA = (layer % 2 == 0);

            Tensor qTensor, kTensor, vTensor;

            if (_isQkvFused)
            {
                Tensor qkvFused = LinearForwardWithBias(input, wn[1], wn[2]);

                if (seqLen == 1)
                {
                    qTensor = qkvFused.Narrow(1, 0, _qDim);
                    kTensor = qkvFused.Narrow(1, _qDim, _kDim);
                    vTensor = qkvFused.Narrow(1, _qDim + _kDim, _kDim);
                    qkvFused.Dispose();
                }
                else
                {
                    using (var qView = qkvFused.Narrow(1, 0, _qDim))
                        qTensor = Ops.NewContiguous(qView);
                    using (var kView = qkvFused.Narrow(1, _qDim, _kDim))
                        kTensor = Ops.NewContiguous(kView);
                    using (var vView = qkvFused.Narrow(1, _qDim + _kDim, _kDim))
                        vTensor = Ops.NewContiguous(vView);
                    qkvFused.Dispose();
                }
            }
            else
            {
                qTensor = LinearForwardWithBias(input, wn[1], wn[2]);
                kTensor = LinearForwardWithBias(input, wn[8], wn[9]);
                vTensor = LinearForwardWithBias(input, wn[10], wn[11]);
            }

            qTensor = ApplyRoPEInPlace(qTensor, numHeads, headDim, seqLen, startPos);
            kTensor = ApplyRoPEInPlace(kTensor, numKVHeads, headDim, seqLen, startPos);

            float[] sinks = _layerSinks[layer];

            long t0 = Stopwatch.GetTimestamp();

            if (seqLen == 1)
            {
                CopyToCacheDecode(_kvCacheK[layer], kTensor, _kvCacheV[layer], vTensor,
                    numKVHeads, headDim, startPos);
                kTensor.Dispose();
                vTensor.Dispose();

                var attnResult = new Tensor(_allocator, DType.Float32, 1, numHeads * headDim);
                AttentionDecodeWithSinks(qTensor, _kvCacheK[layer], _kvCacheV[layer],
                    attnResult, numHeads, numKVHeads, headDim, totalSeqLen, scale, sinks, isSWA);
                qTensor.Dispose();

                _attnTicks += Stopwatch.GetTimestamp() - t0;

                Tensor decodeOut = LinearForwardWithBias(attnResult, wn[3], wn[4]);
                attnResult.Dispose();
                return decodeOut;
            }

            // Prefill path
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
            if (isSWA)
                ApplySWAMask(scores, numHeads, seqLen, totalSeqLen, startPos);
            ApplySoftmaxWithSinks(scores, numHeads, seqLen, totalSeqLen, sinks);

            var attnOut = new Tensor(_allocator, DType.Float32, numHeads, seqLen, headDim);
            Ops.AddmmBatch(attnOut, 0, attnOut, 1.0f, scores, vExpanded);
            scores.Dispose();
            vExpanded.Dispose();

            Tensor flatOutput = ReshapeFromHeads(attnOut, numHeads, seqLen, headDim);
            attnOut.Dispose();

            _attnTicks += Stopwatch.GetTimestamp() - t0;

            Tensor output = LinearForwardWithBias(flatOutput, wn[3], wn[4]);
            flatOutput.Dispose();
            return output;
        }

        private unsafe void ApplySWAMask(Tensor scores, int numHeads, int seqLen, int totalSeqLen, int startPos)
        {
            float* ptr = GetFloatPtr(scores);
            for (int h = 0; h < numHeads; h++)
            {
                for (int q = 0; q < seqLen; q++)
                {
                    int qPos = startPos + q;
                    float* row = ptr + ((long)h * seqLen + q) * totalSeqLen;
                    int limit = qPos - _slidingWindow + 1;
                    for (int k = 0; k < totalSeqLen && k < limit; k++)
                        row[k] = float.NegativeInfinity;
                }
            }
        }

        private unsafe void ApplySoftmaxWithSinks(Tensor scores, int numHeads, int seqLen, int totalSeqLen, float[] sinks)
        {
            if (sinks == null)
            {
                Ops.Softmax(scores, scores);
                return;
            }

            float* ptr = GetFloatPtr(scores);
            for (int h = 0; h < numHeads; h++)
            {
                float sinkVal = sinks[h];
                for (int s = 0; s < seqLen; s++)
                {
                    float* row = ptr + ((long)h * seqLen + s) * totalSeqLen;

                    float maxVal = sinkVal;
                    for (int t = 0; t < totalSeqLen; t++)
                        if (row[t] > maxVal) maxVal = row[t];

                    float sumExp = MathF.Exp(sinkVal - maxVal);
                    for (int t = 0; t < totalSeqLen; t++)
                    {
                        row[t] = MathF.Exp(row[t] - maxVal);
                        sumExp += row[t];
                    }

                    float invSum = 1.0f / sumExp;
                    for (int t = 0; t < totalSeqLen; t++)
                        row[t] *= invSum;
                }
            }
        }

        private unsafe void AttentionDecodeWithSinks(Tensor q, Tensor kCache, Tensor vCache,
            Tensor result, int numHeads, int numKVHeads, int headDim, int totalSeqLen, float scale, float[] sinks, bool isSWA)
        {
            float* qPtr = GetFloatPtr(q);
            float* kPtr = GetFloatPtr(kCache);
            float* vPtr = GetFloatPtr(vCache);
            float* rPtr = GetFloatPtr(result);
            int maxSeqLen = (int)kCache.Sizes[1];
            int groupSize = numHeads / numKVHeads;

            int startT = isSWA ? Math.Max(0, totalSeqLen - _slidingWindow) : 0;
            int numScores = totalSeqLen - startT;
            float* scores = stackalloc float[numScores];

            for (int h = 0; h < numHeads; h++)
            {
                float* qHead = qPtr + h * headDim;
                int kvHead = h / groupSize;
                float* kHead = kPtr + kvHead * maxSeqLen * headDim;
                float* vHead = vPtr + kvHead * maxSeqLen * headDim;

                float maxScore = (sinks != null) ? sinks[h] : float.NegativeInfinity;
                for (int i = 0; i < numScores; i++)
                {
                    int t = startT + i;
                    float s = VecDot(qHead, kHead + t * headDim, headDim) * scale;
                    scores[i] = s;
                    if (s > maxScore) maxScore = s;
                }

                float sumExp = (sinks != null) ? MathF.Exp(sinks[h] - maxScore) : 0f;
                for (int i = 0; i < numScores; i++)
                {
                    float e = MathF.Exp(scores[i] - maxScore);
                    scores[i] = e;
                    sumExp += e;
                }
                float invSum = 1.0f / sumExp;
                for (int i = 0; i < numScores; i++)
                    scores[i] *= invSum;

                float* rHead = rPtr + h * headDim;
                VecZero(rHead, headDim);
                for (int i = 0; i < numScores; i++)
                    VecScaleAdd(rHead, vHead + (startT + i) * headDim, scores[i], headDim);
            }
        }

        private Tensor ApplyRoPEInPlace(Tensor data, int numHeads, int headDim, int seqLen, int startPos)
        {
            int totalRows = seqLen * numHeads;
            int[] positions = new int[totalRows];
            for (int s = 0; s < seqLen; s++)
                for (int h = 0; h < numHeads; h++)
                    positions[s * numHeads + h] = startPos + s;
            using var posTensor = CreateIntTensor(positions, totalRows);

            using var reshaped = data.View(1, seqLen, numHeads, headDim);
            Tensor result = Ops.RoPEEx(
                null, reshaped, posTensor, headDim, 2,
                Config.OriginalContextLength,
                Config.RopeBase, 1.0f / Config.RopeScale,
                1.0f, 1.0f, 32.0f, 1.0f);

            data.Dispose();
            Tensor flat = result.View(seqLen, numHeads * headDim);
            result.Dispose();
            return flat;
        }

        #endregion

        #region MoE

        private unsafe Tensor MoEForward(Tensor hiddenState, int layer, int seqLen)
        {
            string[] wn = _layerNames[layer];
            var (routingWeights, selectedExperts) = MoERoute(hiddenState, wn[6], wn[7], seqLen);

            int hiddenDim = (int)hiddenState.Sizes[1];
            var output = new Tensor(_allocator, DType.Float32, seqLen, hiddenDim);
            Ops.Fill(output, 0f);

            if (seqLen == 1)
            {
                MoEForwardSingleToken(hiddenState, output, routingWeights, selectedExperts, layer, hiddenDim);
                return output;
            }

            MoEForwardBatched(hiddenState, output, routingWeights, selectedExperts, layer, seqLen, hiddenDim);
            return output;
        }

        private unsafe void MoEForwardSingleToken(Tensor hiddenState, Tensor output,
            float[] routingWeights, int[] selectedExperts, int layer, int hiddenDim)
        {
            float* inputPtr = GetFloatPtr(hiddenState);
            float* outputPtr = GetFloatPtr(output);

            for (int e = 0; e < _numExpertsUsed; e++)
            {
                int expertIdx = selectedExperts[e];
                float weight = routingWeights[e];
                string[] en = _expertNames[layer][expertIdx];

                Tensor expertOut = ExpertFFN(hiddenState, en[0], en[1], en[2], en[3], 1);
                float* expertPtr = GetFloatPtr(expertOut);
                VecScaleAdd(outputPtr, expertPtr, weight, hiddenDim);
                expertOut.Dispose();
            }
        }

        private unsafe void MoEForwardBatched(Tensor hiddenState, Tensor output,
            float[] routingWeights, int[] selectedExperts, int layer, int seqLen, int hiddenDim)
        {
            float* inputPtr = GetFloatPtr(hiddenState);
            float* outputPtr = GetFloatPtr(output);

            int totalAssignments = seqLen * _numExpertsUsed;
            int[] expertCounts = _moeExpertCounts;
            int[] expertOffsets = _moeExpertOffsets;
            int[] tokenMap = _moeTokenMap;
            float[] weightMap = _moeWeightMap;

            if (totalAssignments > tokenMap.Length)
            {
                tokenMap = _moeTokenMap = new int[totalAssignments];
                weightMap = _moeWeightMap = new float[totalAssignments];
            }

            Array.Clear(expertCounts, 0, _numExperts);

            for (int s = 0; s < seqLen; s++)
                for (int k = 0; k < _numExpertsUsed; k++)
                    expertCounts[selectedExperts[s * _numExpertsUsed + k]]++;

            expertOffsets[0] = 0;
            for (int e = 1; e < _numExperts; e++)
                expertOffsets[e] = expertOffsets[e - 1] + expertCounts[e - 1];

            int[] fillPos = _moeExpertCounts;
            Array.Copy(expertOffsets, fillPos, _numExperts);

            for (int s = 0; s < seqLen; s++)
            {
                for (int k = 0; k < _numExpertsUsed; k++)
                {
                    int expertIdx = selectedExperts[s * _numExpertsUsed + k];
                    int pos = fillPos[expertIdx]++;
                    tokenMap[pos] = s;
                    weightMap[pos] = routingWeights[s * _numExpertsUsed + k];
                }
            }

            for (int e = 0; e < _numExperts; e++)
            {
                int count = (e < _numExperts - 1) ? expertOffsets[e + 1] - expertOffsets[e]
                                                   : totalAssignments - expertOffsets[e];
                if (count == 0) continue;

                int offset = expertOffsets[e];
                string[] en = _expertNames[layer][e];

                var batchInput = new Tensor(_allocator, DType.Float32, count, hiddenDim);
                float* batchPtr = GetFloatPtr(batchInput);

                long rowBytes = hiddenDim * sizeof(float);
                for (int i = 0; i < count; i++)
                {
                    int tokenIdx = tokenMap[offset + i];
                    Buffer.MemoryCopy(inputPtr + (long)tokenIdx * hiddenDim,
                        batchPtr + (long)i * hiddenDim, rowBytes, rowBytes);
                }

                Tensor expertOut = ExpertFFN(batchInput, en[0], en[1], en[2], en[3], count);
                batchInput.Dispose();

                float* expertOutPtr = GetFloatPtr(expertOut);
                for (int i = 0; i < count; i++)
                {
                    int tokenIdx = tokenMap[offset + i];
                    float weight = weightMap[offset + i];
                    VecScaleAdd(outputPtr + (long)tokenIdx * hiddenDim,
                        expertOutPtr + (long)i * hiddenDim, weight, hiddenDim);
                }
                expertOut.Dispose();
            }
        }

        private unsafe (float[] routingWeights, int[] selectedExperts) MoERoute(
            Tensor input, string routerWeightName, string routerBiasName, int seqLen)
        {
            using var routerScores = LinearForwardWithBias(input, routerWeightName, routerBiasName);

            float* scoresPtr = GetFloatPtr(routerScores);
            int numExperts = (int)routerScores.Sizes[1];

            float[] routingWeights = new float[seqLen * _numExpertsUsed];
            int[] selectedExperts = new int[seqLen * _numExpertsUsed];

            for (int s = 0; s < seqLen; s++)
            {
                float* row = scoresPtr + s * numExperts;

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

                float maxVal = float.NegativeInfinity;
                for (int k = 0; k < _numExpertsUsed; k++)
                    if (routingWeights[s * _numExpertsUsed + k] > maxVal)
                        maxVal = routingWeights[s * _numExpertsUsed + k];
                float sumExp = 0;
                for (int k = 0; k < _numExpertsUsed; k++)
                {
                    float ex = MathF.Exp(routingWeights[s * _numExpertsUsed + k] - maxVal);
                    routingWeights[s * _numExpertsUsed + k] = ex;
                    sumExp += ex;
                }
                if (sumExp > 0)
                    for (int k = 0; k < _numExpertsUsed; k++)
                        routingWeights[s * _numExpertsUsed + k] /= sumExp;
            }

            return (routingWeights, selectedExperts);
        }

        private unsafe Tensor ExpertFFN(Tensor input, string gateUpWeightName, string gateUpBiasName,
            string downWeightName, string downBiasName, int seqLen)
        {
            Tensor gateUp = LinearForwardWithBias(input, gateUpWeightName, gateUpBiasName);
            int halfDim = (int)(gateUp.Sizes[1] / 2);

            float* guPtr = GetFloatPtr(gateUp);

            for (int s = 0; s < seqLen; s++)
            {
                float* gatePtr = guPtr + (long)s * halfDim * 2;
                float* upPtr = gatePtr + halfDim;
                ApplySwiGluOaiInPlace(gatePtr, upPtr, halfDim);
            }

            Tensor activated;
            if (seqLen == 1)
            {
                activated = gateUp.Narrow(1, 0, halfDim);
                gateUp.Dispose();
            }
            else
            {
                using var gView = gateUp.Narrow(1, 0, halfDim);
                activated = Ops.NewContiguous(gView);
                gateUp.Dispose();
            }

            Tensor down = LinearForwardWithBias(activated, downWeightName, downBiasName);
            activated.Dispose();
            return down;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void ApplySwiGluOaiInPlace(float* gate, float* up, int n)
        {
            int vLen = Vector<float>.Count;
            var vAlpha = new Vector<float>(SiluAlpha);
            var vNegAlpha = new Vector<float>(-SiluAlpha);
            var vLimit = new Vector<float>(SiluLimit);
            var vNegLimit = new Vector<float>(-SiluLimit);
            var vOne = Vector<float>.One;

            int i = 0;
            for (; i <= n - vLen; i += vLen)
            {
                var gRaw = Unsafe.ReadUnaligned<Vector<float>>(ref *(byte*)(gate + i));
                var uRaw = Unsafe.ReadUnaligned<Vector<float>>(ref *(byte*)(up + i));

                var x = Vector.Min(gRaw, vLimit);
                var y = Vector.Max(Vector.Min(uRaw, vLimit), vNegLimit);

                var negAx = x * vNegAlpha;
                var expNegAx = VecExpApprox(negAx);
                var sigmoid = vOne / (vOne + expNegAx);
                var result = x * sigmoid * (y + vOne);

                Unsafe.WriteUnaligned(ref *(byte*)(gate + i), result);
            }

            for (; i < n; i++)
            {
                float x = MathF.Min(gate[i], SiluLimit);
                float y = Math.Clamp(up[i], -SiluLimit, SiluLimit);
                float outGlu = x / (1.0f + MathF.Exp(SiluAlpha * (-x)));
                gate[i] = outGlu * (y + 1.0f);
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector<float> VecExpApprox(Vector<float> x)
        {
            var clampLo = new Vector<float>(-88.0f);
            var clampHi = new Vector<float>(88.0f);
            x = Vector.Max(x, clampLo);
            x = Vector.Min(x, clampHi);

            var ln2inv = new Vector<float>(1.4426950409f);
            var n = x * ln2inv;

            var half = new Vector<float>(0.5f);
            var nFloor = Vector.Floor(n + half);

            var ln2 = new Vector<float>(0.6931471806f);
            var r = x - nFloor * ln2;

            var c0 = Vector<float>.One;
            var c1 = Vector<float>.One;
            var c2 = new Vector<float>(0.5f);
            var c3 = new Vector<float>(0.16666667f);
            var c4 = new Vector<float>(0.04166667f);
            var c5 = new Vector<float>(0.00833333f);

            var poly = c0 + r * (c1 + r * (c2 + r * (c3 + r * (c4 + r * c5))));

            var intN = Vector.ConvertToInt32(nFloor);
            var bias = new Vector<int>(127);
            var shift = intN + bias;
            var pow2 = Vector.AsVectorSingle(shift << 23);

            return poly * pow2;
        }

        #endregion

        #region Linear with Bias

        private unsafe Tensor LinearForwardWithBias(Tensor input, string weightName, string biasName)
        {
            Tensor result = LinearForward(input, weightName);
            if (result == null)
                return null;

            if (_weights.TryGetValue(biasName, out var bias))
            {
                int seqLen = (int)result.Sizes[0];
                int outDim = (int)result.Sizes[1];
                float* rPtr = GetFloatPtr(result);
                float* bPtr = GetFloatPtr(bias);
                int biasDim = (int)bias.ElementCount();
                int dim = Math.Min(outDim, biasDim);

                for (int s = 0; s < seqLen; s++)
                    VecScaleAdd(rPtr + (long)s * outDim, bPtr, 1.0f, dim);
            }

            return result;
        }

        #endregion

        public override void Dispose()
        {
            if (_kvCacheK != null)
                foreach (var t in _kvCacheK) t?.Dispose();
            if (_kvCacheV != null)
                foreach (var t in _kvCacheV) t?.Dispose();
            base.Dispose();
        }
    }
}
