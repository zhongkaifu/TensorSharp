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
using TensorSharp;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    /// <summary>
    /// Mistral 3 model architecture.
    /// Key features:
    /// - Standard LLaMA-like transformer with SiLU-gated MLP (SwiGLU)
    /// - GPT-J (norm) style RoPE with YaRN scaling for extended context
    /// - Position-dependent Q scaling: q *= (1 + beta * log(1 + floor(pos / orig_ctx)))
    /// - No QK-norm (unlike Qwen3/Gemma3)
    /// - Supports multimodal (vision) via separate Pixtral vision encoder
    /// </summary>
    public class Mistral3Model : ModelBase
    {
        private Tensor[] _kvCacheK;
        private Tensor[] _kvCacheV;

        private string[][] _layerWeightNames;
        private float[] _ropeFreqs;
        private int _ropeDim;
        private int _attnKeyLen;
        private int _attnValLen;

        // YaRN scaling parameters
        private float _ropeScalingBeta;
        private int _ropeOrigCtx;
        private float _ropeExtFactor;
        private float _ropeBetaFast;
        private float _ropeBetaSlow;
        private float _ropeMscale;
        private float _ropeMscaleAllDim;
        private string _ropeType;

        // Vision support
        private Mistral3VisionEncoder _visionEncoder;
        private List<(Tensor embeddings, int position)> _pendingVisionEmbeddingsList = new();

        public Mistral3Model(string ggufPath, BackendType backend)
            : base(ggufPath, backend)
        {
            string arch = _gguf.GetString("general.architecture") ?? "mistral3";
            Config = new ModelConfig { Architecture = arch };
            ParseBaseConfig();

            _attnKeyLen = Config.KeyLength > 0 ? Config.KeyLength : Config.HeadDim;
            _attnValLen = Config.ValueLength > 0 ? Config.ValueLength : _attnKeyLen;
            _ropeDim = (int)_gguf.GetUint32($"{arch}.rope.dimension_count", (uint)_attnKeyLen);

            // YaRN parameters
            _ropeType = _gguf.GetString($"{arch}.rope.scaling.type", "");
            _ropeScalingBeta = _gguf.GetFloat32($"{arch}.attention.temperature_scale",
                               _gguf.GetFloat32($"{arch}.rope.scaling_beta", 0.1f));
            _ropeOrigCtx = (int)_gguf.GetUint32($"{arch}.rope.scaling.original_context_length", 0);
            Config.OriginalContextLength = _ropeOrigCtx;
            _ropeExtFactor = _gguf.GetFloat32($"{arch}.rope.scaling.extrapolation_factor", 1.0f);
            _ropeBetaFast = _gguf.GetFloat32($"{arch}.rope.scaling.yarn_beta_fast",
                            _gguf.GetFloat32($"{arch}.rope.scaling.beta_fast", 32.0f));
            _ropeBetaSlow = _gguf.GetFloat32($"{arch}.rope.scaling.yarn_beta_slow",
                            _gguf.GetFloat32($"{arch}.rope.scaling.beta_slow", 1.0f));
            _ropeMscale = _gguf.GetFloat32($"{arch}.rope.scaling.mscale", 0f);
            _ropeMscaleAllDim = _gguf.GetFloat32($"{arch}.rope.scaling.mscale_all_dim", 0f);

            Console.WriteLine($"Model: {arch}, Layers={Config.NumLayers}, Hidden={Config.HiddenSize}, " +
                $"Heads={Config.NumHeads}, KVHeads={Config.NumKVHeads}, KeyLen={_attnKeyLen}, " +
                $"ValLen={_attnValLen}, Vocab={Config.VocabSize}");
            Console.WriteLine($"RoPE base={Config.RopeBase}, scale={Config.RopeScale}, type={_ropeType}, " +
                $"dim={_ropeDim}, origCtx={_ropeOrigCtx}");
            if (_ropeType == "yarn")
                Console.WriteLine($"YaRN beta={_ropeScalingBeta}, betaFast={_ropeBetaFast}, " +
                    $"betaSlow={_ropeBetaSlow}, extFactor={_ropeExtFactor}");

            ParseTokenizer();
            LoadWeights();
            FuseQKVWeights();
            FuseGateUpWeights();
            PrepareCudaQuantizedWeightsForInference();

            InitKVCache(ResolveConfiguredContextLength());
            PrecomputeConstants();
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
            }
            if (fused > 0)
                Console.WriteLine($"  Fused projections: {fused} QKV");
        }

        private bool[] _layerQkvFused;

        private void PrecomputeConstants()
        {
            int numLayers = Config.NumLayers;
            _layerQkvFused = new bool[numLayers];

            _layerWeightNames = new string[numLayers][];
            for (int l = 0; l < numLayers; l++)
            {
                string p = $"blk.{l}.";
                bool fused = _quantWeights.ContainsKey(p + "attn_qkv.weight") ||
                             _weights.ContainsKey(p + "attn_qkv.weight");
                _layerQkvFused[l] = fused;

                if (fused)
                {
                    _layerWeightNames[l] = new[]
                    {
                        p + "attn_norm.weight",      // 0
                        p + "attn_qkv.weight",        // 1
                        p + "attn_output.weight",     // 2
                        p + "ffn_norm.weight",         // 3
                        p + "ffn_gate_up.weight",      // 4
                        p + "ffn_down.weight",         // 5
                    };
                }
                else
                {
                    _layerWeightNames[l] = new[]
                    {
                        p + "attn_norm.weight",      // 0
                        p + "attn_q.weight",          // 1
                        p + "attn_k.weight",          // 2
                        p + "attn_v.weight",          // 3
                        p + "attn_output.weight",     // 4
                        p + "ffn_norm.weight",         // 5
                        p + "ffn_gate_up.weight",      // 6
                        p + "ffn_down.weight",         // 7
                    };
                }
            }

            int halfDim = _ropeDim / 2;
            float freqScale = 1.0f / Config.RopeScale;
            _ropeFreqs = new float[halfDim];
            for (int i = 0; i < halfDim; i++)
                _ropeFreqs[i] = freqScale / MathF.Pow(Config.RopeBase, (2.0f * i) / _ropeDim);

            if (_ropeType == "yarn" && _ropeOrigCtx > 0)
                ApplyYarnFreqCorrection(_ropeFreqs, halfDim);
        }

        /// <summary>
        /// Apply YaRN frequency correction to precomputed RoPE frequencies for decode path.
        /// Interpolates between extrapolated and interpolated frequencies based on
        /// whether each frequency band is within the "slow" or "fast" rotation range.
        /// </summary>
        private void ApplyYarnFreqCorrection(float[] freqs, int halfDim)
        {
            float lowFreqWavelen = (float)(_ropeOrigCtx / _ropeBetaSlow);
            float highFreqWavelen = (float)(_ropeOrigCtx / _ropeBetaFast);

            for (int i = 0; i < halfDim; i++)
            {
                float origFreq = 1.0f / MathF.Pow(Config.RopeBase, (2.0f * i) / _ropeDim);
                float wavelen = 2.0f * MathF.PI / origFreq;

                if (wavelen < highFreqWavelen)
                {
                    // High frequency: use original frequency (extrapolation)
                    freqs[i] = origFreq;
                }
                else if (wavelen > lowFreqWavelen)
                {
                    // Low frequency: use interpolated frequency
                    freqs[i] = origFreq / Config.RopeScale;
                }
                else
                {
                    // Intermediate: smooth blend between interpolated and extrapolated
                    float smooth = (lowFreqWavelen / wavelen - 1.0f) /
                                   (lowFreqWavelen / highFreqWavelen - 1.0f);
                    float interpFreq = origFreq / Config.RopeScale;
                    freqs[i] = (1.0f - smooth) * interpFreq + smooth * origFreq;
                }
            }
        }

        private void InitKVCache(int maxSeqLen)
        {
            _maxContextLength = maxSeqLen;
            int numKVHeads = Config.NumKVHeads;
            _kvCacheK = new Tensor[Config.NumLayers];
            _kvCacheV = new Tensor[Config.NumLayers];
            for (int l = 0; l < Config.NumLayers; l++)
            {
                _kvCacheK[l] = new Tensor(_allocator, DType.Float32, numKVHeads, maxSeqLen, _attnKeyLen);
                _kvCacheV[l] = new Tensor(_allocator, DType.Float32, numKVHeads, maxSeqLen, _attnValLen);
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

        // Vision support
        public void LoadVisionEncoder(string mmProjPath)
        {
            _visionEncoder = new Mistral3VisionEncoder(mmProjPath, _allocator);
        }

        public void SetVisionEmbeddings(Tensor embeddings, int insertPosition)
        {
            _pendingVisionEmbeddingsList.Add((embeddings, insertPosition));
        }

        public Mistral3VisionEncoder VisionEncoder => _visionEncoder;

        public override float[] Forward(int[] tokens)
        {
            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;

            long t1 = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t1;

            if (_pendingVisionEmbeddingsList.Count > 0)
            {
                foreach (var (embeddings, position) in _pendingVisionEmbeddingsList)
                {
                    InjectVisionEmbeddings(hidden, embeddings, position);
                    embeddings.Dispose();
                }
                _pendingVisionEmbeddingsList.Clear();
            }

            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                hidden = TransformerBlock(hidden, layer, seqLen, startPos);
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

        private unsafe void InjectVisionEmbeddings(Tensor hidden, Tensor visionEmbeddings, int insertPos)
        {
            int numVisionTokens = (int)visionEmbeddings.Sizes[0];
            int dim = Config.HiddenSize;
            float* hPtr = GetFloatPtr(hidden);
            float* vPtr = GetFloatPtr(visionEmbeddings);

            for (int t = 0; t < numVisionTokens; t++)
            {
                float* dst = hPtr + (long)(insertPos + t) * dim;
                float* src = vPtr + (long)t * dim;
                Buffer.MemoryCopy(src, dst, dim * sizeof(float), dim * sizeof(float));
            }

            Console.WriteLine($"Injected {numVisionTokens} vision tokens at position {insertPos}");
        }

        private Tensor TransformerBlock(Tensor hidden, int layer, int seqLen, int startPos)
        {
            string[] wn = _layerWeightNames[layer];

            bool fused = _layerQkvFused[layer];
            int normIdx = 0;
            int ffnNormIdx = fused ? 3 : 5;
            int gateUpIdx = fused ? 4 : 6;
            int downIdx = fused ? 5 : 7;

            Tensor normed = RMSNormOp(hidden, wn[normIdx]);
            Tensor attnOut = Attention(normed, layer, wn, seqLen, startPos);
            normed.Dispose();

            Ops.Add(hidden, hidden, attnOut);
            attnOut.Dispose();

            Tensor normed2 = RMSNormOp(hidden, wn[ffnNormIdx]);
            Tensor ffnOut = FFN(normed2, wn[gateUpIdx], wn[downIdx], seqLen);
            normed2.Dispose();

            Ops.Add(hidden, hidden, ffnOut);
            ffnOut.Dispose();

            return hidden;
        }

        private Tensor Attention(Tensor input, int layer, string[] wn, int seqLen, int startPos)
        {
            int numHeads = Config.NumHeads;
            int numKVHeads = Config.NumKVHeads;
            int headDim = _attnKeyLen;
            int qDim = numHeads * headDim;
            int kDim = numKVHeads * headDim;
            int totalSeqLen = startPos + seqLen;
            float scale = 1.0f / MathF.Sqrt(headDim);

            Tensor qTensor, kTensor, vTensor;

            bool layerFused = _layerQkvFused[layer];
            if (layerFused)
            {
                Tensor qkvFused = LinearForward(input, wn[1]);
                if (seqLen == 1)
                {
                    qTensor = qkvFused.Narrow(1, 0, qDim);
                    kTensor = qkvFused.Narrow(1, qDim, kDim);
                    vTensor = qkvFused.Narrow(1, qDim + kDim, kDim);
                    qkvFused.Dispose();
                }
                else
                {
                    using (var qView = qkvFused.Narrow(1, 0, qDim))
                        qTensor = Ops.NewContiguous(qView);
                    using (var kView = qkvFused.Narrow(1, qDim, kDim))
                        kTensor = Ops.NewContiguous(kView);
                    using (var vView = qkvFused.Narrow(1, qDim + kDim, kDim))
                        vTensor = Ops.NewContiguous(vView);
                    qkvFused.Dispose();
                }
            }
            else
            {
                qTensor = LinearForward(input, wn[1]);  // attn_q
                kTensor = LinearForward(input, wn[2]);  // attn_k
                vTensor = LinearForward(input, wn[3]);  // attn_v
            }

            if (seqLen == 1)
            {
                ApplyRoPEDecode(qTensor, numHeads, headDim, startPos);
                ApplyRoPEDecode(kTensor, numKVHeads, headDim, startPos);

                // Position-dependent Q scaling for YaRN
                if (_ropeOrigCtx > 0)
                    ApplyPositionScale(qTensor, numHeads * headDim, startPos);
            }
            else
            {
                qTensor = ApplyRoPEPrefill(qTensor, numHeads, headDim, seqLen, startPos);
                kTensor = ApplyRoPEPrefill(kTensor, numKVHeads, headDim, seqLen, startPos);

                // Position-dependent Q scaling for YaRN
                if (_ropeOrigCtx > 0)
                    ApplyPositionScalePrefill(qTensor, numHeads, headDim, seqLen, startPos);
            }

            long t0 = Stopwatch.GetTimestamp();

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

                int outputIdx = layerFused ? 2 : 4;
                Tensor decodeOut = LinearForward(attnResult, wn[outputIdx]);
                attnResult.Dispose();
                return decodeOut;
            }

            Tensor qHeads = ReshapeToHeads(qTensor, numHeads, seqLen, headDim);
            qTensor.Dispose();
            Tensor kHeads = ReshapeToHeads(kTensor, numKVHeads, seqLen, headDim);
            kTensor.Dispose();
            Tensor vHeads = ReshapeToHeads(vTensor, numKVHeads, seqLen, _attnValLen);
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

            var attnOut = new Tensor(_allocator, DType.Float32, numHeads, seqLen, _attnValLen);
            Ops.AddmmBatch(attnOut, 0, attnOut, 1.0f, scores, vExpanded);
            scores.Dispose();
            vExpanded.Dispose();

            Tensor flatOutput = ReshapeFromHeads(attnOut, numHeads, seqLen, _attnValLen);
            attnOut.Dispose();

            _attnTicks += Stopwatch.GetTimestamp() - t0;

            int outIdx = layerFused ? 2 : 4;
            Tensor output = LinearForward(flatOutput, wn[outIdx]);
            flatOutput.Dispose();

            return output;
        }

        /// <summary>
        /// GPT-J (norm) style RoPE: pairs adjacent elements (x[2i], x[2i+1]).
        /// Uses precomputed YaRN-corrected frequencies for decode.
        /// </summary>
        private unsafe void ApplyRoPEDecode(Tensor data, int numHeads, int headDim, int position)
        {
            int halfDim = _ropeDim / 2;
            float* ptr = GetFloatPtr(data);

            float* cosTable = stackalloc float[halfDim];
            float* sinTable = stackalloc float[halfDim];
            for (int i = 0; i < halfDim; i++)
            {
                float theta = position * _ropeFreqs[i];
                cosTable[i] = MathF.Cos(theta);
                sinTable[i] = MathF.Sin(theta);
            }

            for (int h = 0; h < numHeads; h++)
            {
                float* head = ptr + h * headDim;
                for (int i = 0; i < halfDim; i++)
                {
                    float x0 = head[2 * i];
                    float x1 = head[2 * i + 1];
                    head[2 * i] = x0 * cosTable[i] - x1 * sinTable[i];
                    head[2 * i + 1] = x0 * sinTable[i] + x1 * cosTable[i];
                }
            }
        }

        private Tensor ApplyRoPEPrefill(Tensor data, int numHeads, int headDim, int seqLen, int startPos)
        {
            int totalRows = seqLen * numHeads;
            int[] positions = new int[totalRows];
            for (int s = 0; s < seqLen; s++)
                for (int h = 0; h < numHeads; h++)
                    positions[s * numHeads + h] = startPos + s;
            using var posTensor = CreateIntTensor(positions, totalRows);

            using var reshaped = data.View(1, seqLen, numHeads, headDim);
            Tensor result = Ops.RoPEEx(
                null, reshaped, posTensor, _ropeDim, 0, _ropeOrigCtx,
                Config.RopeBase, 1.0f / Config.RopeScale,
                _ropeType == "yarn" ? _ropeExtFactor : 0f,
                ComputeAttnFactor(),
                _ropeType == "yarn" ? _ropeBetaFast : 0f,
                _ropeType == "yarn" ? _ropeBetaSlow : 0f);

            data.Dispose();

            Tensor flat = result.View(seqLen, numHeads * headDim);
            result.Dispose();
            return flat;
        }

        private float ComputeAttnFactor()
        {
            if (_ropeMscale != 0 && _ropeMscaleAllDim != 0)
                return 1.0f / (0.1f * MathF.Log(Config.RopeScale) + 1.0f);
            return 1.0f;
        }

        /// <summary>
        /// Position-dependent Q scaling for YaRN:
        /// q *= (1 + beta * log(1 + floor(pos / orig_ctx)))
        /// </summary>
        private unsafe void ApplyPositionScale(Tensor qTensor, int totalQDim, int position)
        {
            float interval = MathF.Floor((float)position / _ropeOrigCtx);
            float posScale = 1.0f + _ropeScalingBeta * MathF.Log(1.0f + interval);
            if (MathF.Abs(posScale - 1.0f) < 1e-7f)
                return;

            float* ptr = GetFloatPtr(qTensor);
            VecScale(ptr, posScale, totalQDim);
        }

        private unsafe void ApplyPositionScalePrefill(Tensor qTensor, int numHeads, int headDim,
            int seqLen, int startPos)
        {
            float* ptr = GetFloatPtr(qTensor);
            int stride = numHeads * headDim;

            for (int s = 0; s < seqLen; s++)
            {
                int pos = startPos + s;
                float interval = MathF.Floor((float)pos / _ropeOrigCtx);
                float posScale = 1.0f + _ropeScalingBeta * MathF.Log(1.0f + interval);
                if (MathF.Abs(posScale - 1.0f) < 1e-7f)
                    continue;
                VecScale(ptr + (long)s * stride, posScale, stride);
            }
        }

        // Native batch decode is not used for Mistral 3 because YaRN applies
        // per-dimension frequency correction that the generic TransformerLayerDecode
        // API cannot express. The C# decode path uses GGML-backed matmul/attention
        // and only adds a lightweight C# RoPE kernel.

        public override void Dispose()
        {
            _visionEncoder?.Dispose();
            foreach (var (embeddings, _) in _pendingVisionEmbeddingsList)
                embeddings?.Dispose();
            _pendingVisionEmbeddingsList.Clear();

            if (_kvCacheK != null)
                foreach (var t in _kvCacheK) t?.Dispose();
            if (_kvCacheV != null)
                foreach (var t in _kvCacheV) t?.Dispose();

            base.Dispose();
        }
    }
}
