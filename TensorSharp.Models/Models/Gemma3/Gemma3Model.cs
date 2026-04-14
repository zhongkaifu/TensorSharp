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
    /// Gemma 3 model architecture.
    /// Key features:
    /// - Alternating sliding-window (local) and full causal (global) attention every 6 layers
    /// - NeoX-style RoPE with different bases for local/global layers
    /// - GELU activation in MLP (GeGLU: GELU(Gate) * Up)
    /// - 4 RMSNorms per layer: attn_norm, post_attention_norm, ffn_norm, post_ffw_norm
    /// - QK-norm (per-head RMSNorm on Q and K)
    /// - Embedding scaling by sqrt(hidden_size)
    /// - Tied token embedding / output weights
    /// - Final logit softcapping (model-dependent)
    /// </summary>
    public class Gemma3Model : ModelBase
    {
        private const int GlobalCacheInterval = 6;

        private Tensor[] _kvCacheK;
        private Tensor[] _kvCacheV;
        private float[] _ropeFreqsLocal;
        private float[] _ropeFreqsGlobal;
        private int _slidingWindow;
        private float _ropeLocalBase;
        private float _ropeGlobalBase;
        private float _ropeScale;
        private float _finalLogitSoftcap;
        private int _attnKeyLen;
        private int _attnValLen;
        private bool _hasTiedOutput;

        private Gemma3VisionEncoder _visionEncoder;
        private List<(Tensor embeddings, int position)> _pendingVisionEmbeddingsList = new();

        public Gemma3Model(string ggufPath, BackendType backend) : base(ggufPath, backend)
        {
            Config = new ModelConfig { Architecture = _gguf.GetString("general.architecture") };
            ParseBaseConfig();

            _attnKeyLen = Config.KeyLength > 0 ? Config.KeyLength : 256;
            _attnValLen = Config.ValueLength > 0 ? Config.ValueLength : 256;
            _slidingWindow = (int)_gguf.GetUint32($"{Config.Architecture}.attention.sliding_window", 1024);
            Config.SlidingWindow = _slidingWindow;
            _ropeLocalBase = _gguf.GetFloat32($"{Config.Architecture}.rope.local.freq_base", 10000f);
            _ropeGlobalBase = Config.RopeBase;
            _ropeScale = Config.RopeScale;
            _finalLogitSoftcap = _gguf.GetFloat32($"{Config.Architecture}.final_logit_softcapping", 0f);

            if (Config.NumLayers == 34)
                _ropeScale = 8.0f;

            Console.WriteLine($"Model: {Config.Architecture}, Layers={Config.NumLayers}, " +
                $"Hidden={Config.HiddenSize}, Heads={Config.NumHeads}, KVHeads={Config.NumKVHeads}, " +
                $"KeyLen={_attnKeyLen}, ValLen={_attnValLen}, Vocab={Config.VocabSize}");
            Console.WriteLine($"RoPE global={_ropeGlobalBase} local={_ropeLocalBase} scale={_ropeScale}");
            Console.WriteLine($"Sliding window={_slidingWindow}, Softcap={_finalLogitSoftcap}");

            int globalCount = 0;
            for (int i = 0; i < Config.NumLayers; i++)
                if (IsGlobalLayer(i)) globalCount++;
            Console.WriteLine($"Layer types: {globalCount} global (causal), {Config.NumLayers - globalCount} local (SWA)");

            ParseTokenizer();
            LoadWeights();

            _hasTiedOutput = !_weights.ContainsKey("output.weight") && !_quantWeights.ContainsKey("output.weight");
            if (_hasTiedOutput)
                Console.WriteLine("  Output tied to token_embd.weight");

            FuseGateUpWeights();
            PrecomputeRoPE();
            InitKVCache(ResolveConfiguredContextLength());
        }

        private bool IsGlobalLayer(int layer) => (layer + 1) % GlobalCacheInterval == 0;

        private void PrecomputeRoPE()
        {
            int halfDim = _attnKeyLen / 2;
            _ropeFreqsLocal = new float[halfDim];
            _ropeFreqsGlobal = new float[halfDim];

            for (int i = 0; i < halfDim; i++)
            {
                double freqLocal = 1.0 / Math.Pow(_ropeLocalBase, 2.0 * i / _attnKeyLen);
                _ropeFreqsLocal[i] = (float)freqLocal;

                double freqGlobal = 1.0 / Math.Pow(_ropeGlobalBase, 2.0 * i / _attnKeyLen);
                _ropeFreqsGlobal[i] = (float)(freqGlobal / _ropeScale);
            }
        }

        private void InitKVCache(int maxSeqLen)
        {
            _maxContextLength = maxSeqLen;
            _kvCacheK = new Tensor[Config.NumLayers];
            _kvCacheV = new Tensor[Config.NumLayers];

            for (int l = 0; l < Config.NumLayers; l++)
            {
                _kvCacheK[l] = new Tensor(_allocator, DType.Float32, Config.NumKVHeads, maxSeqLen, _attnKeyLen);
                _kvCacheV[l] = new Tensor(_allocator, DType.Float32, Config.NumKVHeads, maxSeqLen, _attnValLen);
                Ops.Fill(_kvCacheK[l], 0f);
                Ops.Fill(_kvCacheV[l], 0f);
            }
        }

        public override void ResetKVCache()
        {
            _cacheSeqLen = 0;
            if (_kvCacheK != null)
            {
                foreach (var k in _kvCacheK)
                {
                    Ops.Fill(k, 0f);
                    InvalidateTensorDeviceCache(k);
                }
                foreach (var v in _kvCacheV)
                {
                    Ops.Fill(v, 0f);
                    InvalidateTensorDeviceCache(v);
                }
            }
        }

        public override void TruncateKVCache(int tokenCount)
        {
            base.TruncateKVCache(tokenCount);
            if (_kvCacheK != null)
            {
                foreach (var k in _kvCacheK)
                    InvalidateTensorDeviceCache(k);
                foreach (var v in _kvCacheV)
                    InvalidateTensorDeviceCache(v);
            }
        }

        public void LoadVisionEncoder(string mmProjPath)
        {
            _visionEncoder = new Gemma3VisionEncoder(mmProjPath, _allocator);
        }

        public void SetVisionEmbeddings(Tensor embeddings, int insertPosition)
        {
            _pendingVisionEmbeddingsList.Add((embeddings, insertPosition));
        }

        public Gemma3VisionEncoder VisionEncoder => _visionEncoder;

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
                foreach (var (embeddings, position) in _pendingVisionEmbeddingsList)
                {
                    InjectVisionEmbeddings(hidden, embeddings, position);
                    embeddings.Dispose();
                }
                _pendingVisionEmbeddingsList.Clear();
            }

            bool dumpLayers = Environment.GetEnvironmentVariable("DUMP_LAYERS") == "1";
            if (seqLen > 1 && Environment.GetEnvironmentVariable("TEST_MATMUL") == "1")
            {
                TestMatmulPrecision(hidden, seqLen);
            }

            for (int l = 0; l < Config.NumLayers; l++)
            {
                hidden = TransformerBlock(hidden, l, seqLen, startPos);
                if (dumpLayers)
                    DumpHiddenState(hidden, seqLen, l);
            }

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

        /// <summary>
        /// Replace token embeddings at image placeholder positions with vision encoder output.
        /// The vision embeddings tensor has shape [numTokens, projDim] where projDim == hiddenSize.
        /// </summary>
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

        private Tensor TransformerBlock(Tensor hidden, int layer, int seqLen, int startPos)
        {
            string prefix = $"blk.{layer}";

            // Pre-attention norm
            using var attnNormed = RMSNormOp(hidden, $"{prefix}.attn_norm.weight");

            // Self-attention
            using var attnOut = Attention(attnNormed, layer, prefix, seqLen, startPos);

            // Post-attention norm
            using var postAttnNormed = RMSNormOp(attnOut, $"{prefix}.post_attention_norm.weight");

            // Residual connection
            Ops.Add(postAttnNormed, postAttnNormed, hidden);
            hidden.Dispose();

            // Pre-FFN norm
            using var ffnNormed = RMSNormOp(postAttnNormed, $"{prefix}.ffn_norm.weight");

            // FFN (GeGLU with GELU activation)
            using var ffnOut = FFNGelu(ffnNormed, $"{prefix}.ffn_gate_up.weight",
                $"{prefix}.ffn_down.weight", seqLen);

            // Post-FFN norm
            using var postFfnNormed = RMSNormOp(ffnOut, $"{prefix}.post_ffw_norm.weight");

            // Residual connection
            var result = new Tensor(_allocator, DType.Float32, postAttnNormed.Sizes);
            Ops.Add(result, postAttnNormed, postFfnNormed);
            return result;
        }

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

        private Tensor Attention(Tensor input, int layer, string prefix, int seqLen, int startPos)
        {
            long t0 = Stopwatch.GetTimestamp();
            bool isGlobal = IsGlobalLayer(layer);

            Tensor q = LinearForward(input, $"{prefix}.attn_q.weight");
            Tensor k = LinearForward(input, $"{prefix}.attn_k.weight");
            Tensor v = LinearForward(input, $"{prefix}.attn_v.weight");

            // QK norm
            if (seqLen == 1)
            {
                RMSNormInPlace(q, _weights[$"{prefix}.attn_q_norm.weight"], Config.NumHeads, _attnKeyLen, Config.Eps);
                RMSNormInPlace(k, _weights[$"{prefix}.attn_k_norm.weight"], Config.NumKVHeads, _attnKeyLen, Config.Eps);
            }
            else
            {
                q = ApplyBatchRMSNorm(q, $"{prefix}.attn_q_norm.weight", Config.NumHeads, seqLen, _attnKeyLen);
                k = ApplyBatchRMSNorm(k, $"{prefix}.attn_k_norm.weight", Config.NumKVHeads, seqLen, _attnKeyLen);
            }

            // Apply NeoX-style RoPE
            if (seqLen == 1)
            {
                float[] freqs = isGlobal ? _ropeFreqsGlobal : _ropeFreqsLocal;
                ApplyNeoXRoPEDecode(q, Config.NumHeads, _attnKeyLen, startPos, freqs);
                ApplyNeoXRoPEDecode(k, Config.NumKVHeads, _attnKeyLen, startPos, freqs);
            }
            else
            {
                float ropeBase = isGlobal ? _ropeGlobalBase : _ropeLocalBase;
                float freqScale = isGlobal ? (1.0f / _ropeScale) : 1.0f;
                q = ApplyRoPEPrefill(q, Config.NumHeads, _attnKeyLen, seqLen, startPos, ropeBase, freqScale);
                k = ApplyRoPEPrefill(k, Config.NumKVHeads, _attnKeyLen, seqLen, startPos, ropeBase, freqScale);
            }

            // Q scaling
            float qScale = 1f / MathF.Sqrt(_attnKeyLen);
            ScaleTensor(q, qScale);

            int totalSeqLen = startPos + seqLen;
            Tensor result;

            if (seqLen == 1)
            {
                CopyToCacheDecode(_kvCacheK[layer], k, _kvCacheV[layer], v,
                    Config.NumKVHeads, _attnKeyLen, startPos);

                int attendLen = isGlobal ? totalSeqLen : Math.Min(totalSeqLen, _slidingWindow);
                int attendStart = totalSeqLen - attendLen;

                result = new Tensor(_allocator, DType.Float32, 1, Config.NumHeads * _attnValLen);
                AttentionDecodeWithWindow(q, _kvCacheK[layer], _kvCacheV[layer], result,
                    Config.NumHeads, Config.NumKVHeads, _attnKeyLen, _attnValLen,
                    attendStart, totalSeqLen, 1f);
            }
            else
            {
                Tensor qHeads = ReshapeToHeads(q, Config.NumHeads, seqLen, _attnKeyLen);
                Tensor kHeads = ReshapeToHeads(k, Config.NumKVHeads, seqLen, _attnKeyLen);
                Tensor vHeads = ReshapeToHeads(v, Config.NumKVHeads, seqLen, _attnValLen);

                CopyToCache(_kvCacheK[layer], kHeads, startPos, seqLen);
                CopyToCache(_kvCacheV[layer], vHeads, startPos, seqLen);
                kHeads.Dispose();
                vHeads.Dispose();

                int groupSize = Config.NumHeads / Config.NumKVHeads;
                Tensor kExpanded = ExpandKVHeads(_kvCacheK[layer], groupSize, totalSeqLen);
                Tensor vExpanded = ExpandKVHeads(_kvCacheV[layer], groupSize, totalSeqLen);

                using var kT = kExpanded.Transpose(1, 2);
                var scores = new Tensor(_allocator, DType.Float32, Config.NumHeads, seqLen, totalSeqLen);
                Ops.AddmmBatch(scores, 0, scores, 1f, qHeads, kT);
                qHeads.Dispose();
                kExpanded.Dispose();

                int windowSize = isGlobal ? 0 : _slidingWindow;
                ApplyCausalMask(scores, seqLen, totalSeqLen, windowSize);
                Ops.Softmax(scores, scores);

                var attnOut = new Tensor(_allocator, DType.Float32, Config.NumHeads, seqLen, _attnValLen);
                Ops.AddmmBatch(attnOut, 0, attnOut, 1.0f, scores, vExpanded);
                scores.Dispose();
                vExpanded.Dispose();

                result = ReshapeFromHeads(attnOut, Config.NumHeads, seqLen, _attnValLen);
                attnOut.Dispose();
            }

            q.Dispose();
            k.Dispose();
            v.Dispose();

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

        /// <summary>
        /// NeoX-style RoPE: pairs (x[j], x[j + d/2]) — first half with second half.
        /// GGML GGML_ROPE_TYPE_NEOX uses n_offset = n_dims/2.
        /// </summary>
        private unsafe void ApplyNeoXRoPEDecode(Tensor data, int numHeads, int headDim, int position, float[] freqs)
        {
            float* ptr = GetFloatPtr(data);
            int halfDim = headDim / 2;

            for (int h = 0; h < numHeads; h++)
            {
                float* head = ptr + h * headDim;
                for (int j = 0; j < halfDim; j++)
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

        private Tensor ApplyRoPEPrefill(Tensor data, int numHeads, int headDim,
            int seqLen, int startPos, float ropeBase, float freqScale)
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
                ropeBase, freqScale,
                0.0f, 1.0f, 0.0f, 0.0f);

            data.Dispose();
            Tensor flat = result.View(seqLen, numHeads * headDim);
            result.Dispose();
            return flat;
        }

        private void ScaleTensor(Tensor t, float scale)
        {
            Ops.Mul(t, t, scale);
        }

        /// <summary>
        /// Attention decode with optional sliding window.
        /// attendStart..totalSeqLen-1 is the window of positions to attend to.
        /// </summary>
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

        /// <summary>
        /// Apply causal mask to attention scores with optional sliding window.
        /// For sliding window, only positions within windowSize of the query are attended to.
        /// </summary>
        private unsafe void ApplyCausalMask(Tensor scores, int queryLen, int totalKVLen, int windowSize)
        {
            float* sPtr = GetFloatPtr(scores);
            int numHeads = (int)scores.Sizes[0];
            int rowStride = queryLen * totalKVLen;

            for (int h = 0; h < numHeads; h++)
            {
                float* headScores = sPtr + h * rowStride;
                for (int q = 0; q < queryLen; q++)
                {
                    int queryPos = totalKVLen - queryLen + q;
                    float* row = headScores + q * totalKVLen;

                    // Mask future positions (kv > queryPos)
                    for (int kv = queryPos + 1; kv < totalKVLen; kv++)
                        row[kv] = float.NegativeInfinity;

                    // Mask positions outside sliding window
                    if (windowSize > 0)
                    {
                        int windowStart = queryPos - windowSize + 1;
                        if (windowStart > 0)
                        {
                            for (int kv = 0; kv < windowStart; kv++)
                                row[kv] = float.NegativeInfinity;
                        }
                    }
                }
            }
        }

        private unsafe void TestMatmulPrecision(Tensor hidden, int seqLen)
        {
            int dim = Config.HiddenSize;
            Console.WriteLine($"=== Precision test: seqLen={seqLen}, dim={dim} ===");

            // Test RMSNorm: batch vs single row
            string normWeight = "blk.0.attn_norm.weight";
            using var batchNorm = RMSNormOp(hidden, normWeight);

            using var lastRow = hidden.Narrow(0, seqLen - 1, 1);
            using var lastRowContig = Ops.NewContiguous(lastRow);
            using var singleNorm = RMSNormOp(lastRowContig, normWeight);

            CompareRows(batchNorm, singleNorm, seqLen - 1, dim, "RMSNorm");

            // Test Linear on RMSNorm output
            string qWeight = "blk.0.attn_q.weight";
            using var batchQ = LinearForward(batchNorm, qWeight);
            using var singleQ = LinearForward(singleNorm, qWeight);
            int qDim = (int)batchQ.Sizes[1];
            CompareRows(batchQ, singleQ, seqLen - 1, qDim, "Linear(Q)");
        }

        private unsafe void CompareRows(Tensor batch, Tensor single, int rowIdx, int dim, string label)
        {
            float* batchPtr = GetFloatPtr(batch);
            float* singlePtr = GetFloatPtr(single);
            float* batchRow = batchPtr + rowIdx * dim;

            float maxDiff = 0;
            double sumDiff = 0;
            int diffCount = 0;
            for (int d = 0; d < dim; d++)
            {
                float diff = MathF.Abs(batchRow[d] - singlePtr[d]);
                if (diff > 0) { diffCount++; sumDiff += diff; }
                if (diff > maxDiff) maxDiff = diff;
            }
            Console.WriteLine($"  {label} row[{rowIdx}]: maxDiff={maxDiff:E6}, avgDiff={sumDiff / dim:E6}, nonZero={diffCount}/{dim}");
            Console.Write($"    batch: ");
            for (int i = 0; i < 5; i++) Console.Write($"{batchRow[i]:F8} ");
            Console.Write($"\n    single: ");
            for (int i = 0; i < 5; i++) Console.Write($"{singlePtr[i]:F8} ");
            Console.WriteLine();
        }

        private unsafe void DumpHiddenState(Tensor hidden, int seqLen, int layer)
        {
            float* ptr = GetFloatPtr(hidden);
            int lastPos = seqLen - 1;
            int dim = Config.HiddenSize;
            float* row = ptr + lastPos * dim;
            Console.Write($"  L{layer:D2} pos{lastPos}: ");
            for (int i = 0; i < 5; i++)
                Console.Write($"{row[i]:F6} ");
            float sum = 0;
            for (int i = 0; i < dim; i++)
                sum += row[i] * row[i];
            Console.WriteLine($" norm={MathF.Sqrt(sum):F6}");
        }

        public override void Dispose()
        {
            _visionEncoder?.Dispose();
            foreach (var (embeddings, _) in _pendingVisionEmbeddingsList)
                embeddings?.Dispose();
            _pendingVisionEmbeddingsList.Clear();
            if (_kvCacheK != null)
                foreach (var k in _kvCacheK) k?.Dispose();
            if (_kvCacheV != null)
                foreach (var v in _kvCacheV) v?.Dispose();
            base.Dispose();
        }
    }
}
