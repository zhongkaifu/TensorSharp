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

namespace InferenceEngine
{
    /// <summary>
    /// Qwen3.5 hybrid model: alternates GatedDeltaNet (recurrent) and FullAttention layers.
    /// Every Nth layer is full attention (determined by full_attention_interval), the rest are recurrent.
    /// Full attention layers: gated Q (Q+gate interleaved), QK-norm, sigmoid-gated output.
    /// Recurrent layers: SSM conv1d, gated delta net with recurrent state.
    /// Both layer types: post_attention_norm before FFN, dense FFN (gate+up SiLU + down).
    /// </summary>
    public class Qwen35Model : ModelBase
    {
        private bool[] _isRecurrent;
        private int _fullAttentionInterval;

        // Full attention KV cache (only for attention layers)
        private Tensor[] _kvCacheK;
        private Tensor[] _kvCacheV;

        // GatedDeltaNet state (only for recurrent layers)
        private float[][] _convState;  // [layer][convChannels * (convKernelSize-1)]
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

        // Pre-allocated tensor work buffers for GatedDeltaNet state update
        private Tensor _gdnConvOutT;   // [1, qkvDim] for conv1d output + SiLU
        private Tensor _gdnKBuf;       // [numVHeads, headKDim]
        private Tensor _gdnQBuf;       // [numVHeads, headKDim]
        private Tensor _gdnVBuf;       // [numVHeads, headVDim]
        private Tensor _gdnKvMemBuf;   // [numVHeads, headVDim, 1]
        private Tensor _gdnCoreOutBuf; // [numVHeads, headVDim, 1]
        private Tensor _gdnGatedOutT;  // [1, ssmDInner]

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

            LoadWeights();
            FuseAttentionProjectionWeights();
            FuseRecurrentInputWeights();
            FuseGateUpWeights();
            InitCaches(4096);
            PrecomputeRoPE();
            InitGDNBuffers();
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

                IntPtr fusedPtr = GgmlBasicOps.AlignedAlloc(totalBytes);
                byte* fusedDst = (byte*)fusedPtr.ToPointer();
                long offset = 0;
                for (int i = 0; i < quantWeights.Length; i++)
                {
                    var qw = quantWeights[i];
                    Buffer.MemoryCopy(qw.Data.ToPointer(), fusedDst + offset, totalBytes - offset, qw.RawBytes);
                    offset += qw.RawBytes;
                }

                _quantWeights[fusedName] = new QuantizedWeight(fusedPtr, totalBytes, first.GgmlType, first.Ne0, totalNe1);
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

            _gdnConvOutT = new Tensor(_allocator, DType.Float32, 1, qkvDim);
            _gdnKBuf = new Tensor(_allocator, DType.Float32, _numVHeads, _headKDim);
            _gdnQBuf = new Tensor(_allocator, DType.Float32, _numVHeads, _headKDim);
            _gdnVBuf = new Tensor(_allocator, DType.Float32, _numVHeads, _headVDim);
            _gdnKvMemBuf = new Tensor(_allocator, DType.Float32, _numVHeads, _headVDim, 1);
            _gdnCoreOutBuf = new Tensor(_allocator, DType.Float32, _numVHeads, _headVDim, 1);
            _gdnGatedOutT = new Tensor(_allocator, DType.Float32, 1, _ssmDInner);
        }

        private void InitCaches(int maxSeqLen)
        {
            int numLayers = Config.NumLayers;
            _kvCacheK = new Tensor[numLayers];
            _kvCacheV = new Tensor[numLayers];
            _convState = new float[numLayers][];
            _deltaStateTensor = new Tensor[numLayers];

            int convDim = _convKernel - 1;
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;

            for (int l = 0; l < numLayers; l++)
            {
                if (!_isRecurrent[l])
                {
                    _kvCacheK[l] = new Tensor(_allocator, DType.Float32, Config.NumKVHeads, maxSeqLen, Config.HeadDim);
                    _kvCacheV[l] = new Tensor(_allocator, DType.Float32, Config.NumKVHeads, maxSeqLen, Config.HeadDim);
                    Ops.Fill(_kvCacheK[l], 0);
                    Ops.Fill(_kvCacheV[l], 0);
                }
                else
                {
                    _convState[l] = new float[convDim * qkvDim];
                    _deltaStateTensor[l] = new Tensor(_allocator, DType.Float32, _numVHeads, _headVDim, _headKDim);
                    Ops.Fill(_deltaStateTensor[l], 0);
                }
            }
            _cacheSeqLen = 0;
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
                    Ops.Fill(_kvCacheK[l], 0);
                    Ops.Fill(_kvCacheV[l], 0);
                    InvalidateTensorDeviceCache(_kvCacheK[l]);
                    InvalidateTensorDeviceCache(_kvCacheV[l]);
                }
                else
                {
                    Array.Clear(_convState[l]);
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

        #region Full Attention Block

        /// <summary>
        /// Full attention with gated Q, QK-norm, sigmoid-gated output, and post-attention norm.
        /// Q projection outputs 2x: [Q, gate] interleaved per head.
        /// </summary>
        private Tensor AttentionBlock(Tensor hidden, int layer, int seqLen, int startPos)
        {
            string prefix = $"blk.{layer}.";
            Tensor residual = hidden;

            Tensor normed = RMSNormOp(hidden, prefix + "attn_norm.weight");

            Tensor attnOut = FullAttention(normed, layer, prefix, seqLen, startPos);
            normed.Dispose();

            // First residual
            Ops.Add(hidden, hidden, attnOut);
            attnOut.Dispose();

            // Post-attention norm before FFN
            Tensor normed2 = RMSNormOp(hidden, prefix + "post_attention_norm.weight");
            Tensor ffnOut = FFN(normed2, prefix + "ffn_gate_up.weight", prefix + "ffn_down.weight", seqLen);
            normed2.Dispose();

            // Second residual
            Ops.Add(hidden, hidden, ffnOut);
            ffnOut.Dispose();

            return hidden;
        }

        private unsafe Tensor FullAttention(Tensor input, int layer, string prefix, int seqLen, int startPos)
        {
            int numHeads = Config.NumHeads;
            int numKVHeads = Config.NumKVHeads;
            int headDim = Config.HeadDim;
            int qFullDim = numHeads * headDim * 2;
            int kvDim = numKVHeads * headDim;
            int totalSeqLen = startPos + seqLen;

            Tensor qFull;
            Tensor kTensor;
            Tensor vTensor;
            Tensor fusedQkv = LinearForward(input, prefix + "attn_qkv.weight");
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
                fusedQkv.Dispose();
            }
            else
            {
                // Q projection: outputs [seqLen, numHeads * headDim * 2] (Q + gate interleaved per head)
                qFull = LinearForward(input, prefix + "attn_q.weight");
                kTensor = LinearForward(input, prefix + "attn_k.weight");
                vTensor = LinearForward(input, prefix + "attn_v.weight");
            }

            // Deinterleave Q and gate using tensor ops
            Tensor qTensor, gateTensor;
            {
                var reshaped = qFull.View(seqLen, numHeads, headDim * 2);
                qFull.Dispose();

                var qSlice = reshaped.Narrow(2, 0, headDim);
                var qContig = Ops.NewContiguous(qSlice);
                qTensor = qContig.View(seqLen, numHeads * headDim);
                qContig.Dispose();
                qSlice.Dispose();

                var gSlice = reshaped.Narrow(2, headDim, headDim);
                var gContig = Ops.NewContiguous(gSlice);
                gateTensor = gContig.View(seqLen, numHeads * headDim);
                gContig.Dispose();
                gSlice.Dispose();

                reshaped.Dispose();
            }

            // QK norm
            qTensor = ApplyQKNorm(qTensor, prefix + "attn_q_norm.weight", numHeads, seqLen);
            kTensor = ApplyQKNorm(kTensor, prefix + "attn_k_norm.weight", numKVHeads, seqLen);

            // RoPE
            if (seqLen == 1)
            {
                ApplyRoPEDecodeInPlace(qTensor, numHeads, startPos);
                ApplyRoPEDecodeInPlace(kTensor, numKVHeads, startPos);
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
                CopyToCacheDecode(_kvCacheK[layer], kTensor, _kvCacheV[layer], vTensor,
                    numKVHeads, headDim, startPos);
                kTensor.Dispose();
                vTensor.Dispose();

                attnOutput = new Tensor(_allocator, DType.Float32, 1, numHeads * headDim);
                AttentionDecodePureCS(qTensor, _kvCacheK[layer], _kvCacheV[layer],
                    attnOutput, numHeads, numKVHeads, headDim, totalSeqLen, attentionScale);
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

            // Apply sigmoid gate: output = attention * sigmoid(gate)
            ApplySigmoidGate(attnOutput, gateTensor);
            gateTensor.Dispose();

            _attnTicks += Stopwatch.GetTimestamp() - t0;

            Tensor output = LinearForward(attnOutput, prefix + "attn_output.weight");
            attnOutput.Dispose();
            return output;
        }

        private void ApplySigmoidGate(Tensor attn, Tensor gate)
        {
            Ops.SigmoidMul(attn, attn, gate);
        }

        private Tensor ApplyQKNorm(Tensor data, string weightName, int numHeads, int seqLen)
        {
            int headDim = Config.HeadDim;
            var alpha = _weights[weightName];

            if (seqLen == 1)
            {
                RMSNormInPlace(data, alpha, numHeads, headDim, Config.Eps);
                return data;
            }

            using var reshaped = data.View(seqLen * numHeads, headDim);
            Tensor normed = Ops.RMSNorm(null, reshaped, alpha, null, Config.Eps);
            data.Dispose();

            Tensor result = normed.View(seqLen, numHeads * headDim);
            normed.Dispose();
            return result;
        }

        private unsafe void ApplyRoPEDecodeInPlace(Tensor data, int numHeads, int position)
        {
            int headDim = Config.HeadDim;
            int ropeDim = _ropeDimCount > 0 ? _ropeDimCount : headDim;
            int halfDim = ropeDim / 2;
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
                    float x0 = head[i];
                    float x1 = head[i + halfDim];
                    head[i] = x0 * cosTable[i] - x1 * sinTable[i];
                    head[i + halfDim] = x0 * sinTable[i] + x1 * cosTable[i];
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
            string prefix = $"blk.{layer}.";
            Tensor residual = hidden;

            Tensor normed = RMSNormOp(hidden, prefix + "attn_norm.weight");

            Tensor attnOut = GatedDeltaNet(normed, layer, prefix, seqLen);
            normed.Dispose();

            // First residual
            Ops.Add(hidden, hidden, attnOut);
            attnOut.Dispose();

            // Post-attention norm before FFN
            Tensor normed2 = RMSNormOp(hidden, prefix + "post_attention_norm.weight");
            Tensor ffnOut = FFN(normed2, prefix + "ffn_gate_up.weight", prefix + "ffn_down.weight", seqLen);
            normed2.Dispose();

            // Second residual
            Ops.Add(hidden, hidden, ffnOut);
            ffnOut.Dispose();

            return hidden;
        }

        /// <summary>
        /// GatedDeltaNet recurrent step with batched input/output projections.
        /// Prefill projects the whole chunk once, then walks the recurrent state token-by-token.
        /// Decode follows the same path with seqLen=1, avoiding several tiny GGML dispatches.
        /// </summary>
        private unsafe Tensor GatedDeltaNet(Tensor input, int layer, string prefix, int seqLen)
        {
            long t0 = Stopwatch.GetTimestamp();
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;
            int qkDim = _headKDim * _numKHeads;
            int vDim = _headVDim * _numVHeads;
            int zDim = _headVDim * _numVHeads;
            int packedDim = qkvDim + zDim + _numVHeads * 2;

            Tensor packedInput = LinearForward(input, prefix + "ssm_in_proj.weight");
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
                qkvRaw = LinearForward(input, prefix + "attn_qkv.weight");
                zRaw = LinearForward(input, prefix + "attn_gate.weight");
                betaRaw = LinearForward(input, prefix + "ssm_beta.weight");
                alphaRaw = LinearForward(input, prefix + "ssm_alpha.weight");

                qkvBase = GetFloatPtr(qkvRaw);
                zBase = GetFloatPtr(zRaw);
                betaBase = GetFloatPtr(betaRaw);
                alphaBase = GetFloatPtr(alphaRaw);
            }

            float* convWPtr = GetFloatPtr(_weights[prefix + "ssm_conv1d.weight"]);
            float* dtBiasPtr = GetFloatPtr(_weights[prefix + "ssm_dt.bias"]);
            float* aPtr = GetFloatPtr(_weights[prefix + "ssm_a"]);
            float* ssmNormPtr = GetFloatPtr(_weights[prefix + "ssm_norm.weight"]);

            Tensor gated = seqLen == 1 ? _gdnGatedOutT : new Tensor(_allocator, DType.Float32, seqLen, _ssmDInner);
            float* gatedBase = GetFloatPtr(gated);

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
                    convWPtr, dtBiasPtr, aPtr, ssmNormPtr,
                    gatedBase + (long)s * _ssmDInner);
            }

            InvalidateTensorDeviceCache(gated);
            Tensor output = LinearForward(gated, prefix + "ssm_out.weight");

            if (seqLen > 1)
                gated.Dispose();

            packedInput?.Dispose();
            qkvRaw?.Dispose();
            zRaw?.Dispose();
            betaRaw?.Dispose();
            alphaRaw?.Dispose();

            _attnTicks += Stopwatch.GetTimestamp() - t0;
            return output;
        }

        private unsafe void GatedDeltaNetStep(float* qkvPtr, float* zPtr, float* betaPtr, float* alphaPtr,
            int layer, int qkvDim, int qkDim, int vDim,
            float* convWPtr, float* dtBiasPtr, float* aPtr, float* ssmNormPtr,
            float* gatedOutPtr)
        {
            int convDim = _convKernel - 1;
            float[] convState = _convState[layer];
            float* convOutPtr = GetFloatPtr(_gdnConvOutT);

            for (int ch = 0; ch < qkvDim; ch++)
            {
                float sum = 0;
                for (int ki = 0; ki < _convKernel; ki++)
                {
                    float stateVal = ki < convDim ? convState[ki * qkvDim + ch] : qkvPtr[ch];
                    sum += stateVal * convWPtr[ch * _convKernel + ki];
                }
                convOutPtr[ch] = SiLUScalar(sum);
            }

            for (int c = 0; c < convDim - 1; c++)
                Array.Copy(convState, (c + 1) * qkvDim, convState, c * qkvDim, qkvDim);
            for (int d = 0; d < qkvDim; d++)
                convState[(convDim - 1) * qkvDim + d] = qkvPtr[d];

            fixed (float* qBase = _gdnQ, kBase = _gdnK, vBase = _gdnV)
            {
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

            fixed (float* qPtr = qActive, kPtr = kActive, vPtr = _gdnV, deltaPtr = _gdnDelta, corePtr = _gdnCore)
            {
                VecScale(qPtr, 1.0f / MathF.Sqrt(_headVDim), _numVHeads * _headKDim);

                Tensor state = _deltaStateTensor[layer];
                float* statePtr = GetFloatPtr(state);
                int statePerHead = _headVDim * _headKDim;
                for (int h = 0; h < _numVHeads; h++)
                {
                    float* stateHead = statePtr + h * statePerHead;
                    float* qHead = qPtr + h * _headKDim;
                    float* kHead = kPtr + h * _headKDim;
                    float* vHead = vPtr + h * _headVDim;
                    float* deltaHead = deltaPtr + h * _headVDim;
                    float* coreHead = corePtr + h * _headVDim;
                    float* zHead = zPtr + h * _headVDim;
                    float* gatedHead = gatedOutPtr + h * _headVDim;

                    float alphaBiased = alphaPtr[h] + dtBiasPtr[h];
                    float gateH = SoftplusScalar(alphaBiased) * aPtr[h];
                    VecScale(stateHead, MathF.Exp(gateH), statePerHead);

                    float betaH = SigmoidScalar(betaPtr[h]);
                    for (int row = 0; row < _headVDim; row++)
                    {
                        float* stateRow = stateHead + row * _headKDim;
                        float kvMem = VecDot(stateRow, kHead, _headKDim);
                        deltaHead[row] = (vHead[row] - kvMem) * betaH;
                    }

                    for (int row = 0; row < _headVDim; row++)
                    {
                        float* stateRow = stateHead + row * _headKDim;
                        VecScaleAdd(stateRow, kHead, deltaHead[row], _headKDim);
                        coreHead[row] = VecDot(stateRow, qHead, _headKDim);
                    }

                    float rmsInv = 1.0f / MathF.Sqrt((VecSumSq(coreHead, _headVDim) / _headVDim) + Config.Eps);
                    for (int i = 0; i < _headVDim; i++)
                        gatedHead[i] = coreHead[i] * rmsInv * ssmNormPtr[i] * SiLUScalar(zHead[i]);
                }
            }
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

            _gdnConvOutT?.Dispose();
            _gdnKBuf?.Dispose();
            _gdnQBuf?.Dispose();
            _gdnVBuf?.Dispose();
            _gdnKvMemBuf?.Dispose();
            _gdnCoreOutBuf?.Dispose();
            _gdnGatedOutT?.Dispose();

            base.Dispose();
        }
    }
}
