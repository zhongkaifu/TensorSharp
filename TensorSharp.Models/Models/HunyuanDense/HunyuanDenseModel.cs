// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Diagnostics;
using TensorSharp;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    /// <summary>
    /// Hunyuan dense transformer (GGUF <c>hunyuan-dense</c>).
    /// Matches llama.cpp's hunyuan-vl text graph: RMSNorm → QKV → NeoX RoPE →
    /// per-head Q/K RMSNorm → GQA attention → SwiGLU. QK-norm is AFTER RoPE,
    /// the opposite of Qwen 3.5.
    /// </summary>
    public sealed partial class HunyuanDenseModel : ModelBase
    {
        private Tensor[] _kvCacheK;
        private Tensor[] _kvCacheV;
        private bool[] _layerQkvFused;
        private bool[] _layerGateUpFused;
        private int _attnKeyLen;
        private int _attnValLen;
        private int _ropeDim;
        private int _kvCacheCapacity;

        public HunyuanDenseModel(string ggufPath, BackendType backend, int tpDegree = 1, ITensorParallelGroup tpGroup = null)
            : base(ggufPath, backend, tpDegree, tpGroup)
        {
            string arch = _gguf.GetString("general.architecture") ?? "hunyuan-dense";
            Config = new ModelConfig { Architecture = arch };
            ParseBaseConfig();

            _attnKeyLen = Config.KeyLength > 0 ? Config.KeyLength : Config.HeadDim;
            _attnValLen = Config.ValueLength > 0 ? Config.ValueLength : _attnKeyLen;
            if (_attnKeyLen != _attnValLen)
            {
                throw new NotSupportedException(
                    $"hunyuan-dense expects equal key/value head dims, got key={_attnKeyLen} value={_attnValLen}.");
            }

            _ropeDim = (int)_gguf.GetUint32($"{arch}.rope.dimension_count", (uint)_attnKeyLen);
            ApplyHunyuanRopeBase(arch);
            ParseTokenizer();

            Console.WriteLine($"Model: {arch}, Layers={Config.NumLayers}, Hidden={Config.HiddenSize}, " +
                $"Heads={Config.NumHeads}, KVHeads={Config.NumKVHeads}, KeyLen={_attnKeyLen}, " +
                $"ValLen={_attnValLen}, Vocab={Config.VocabSize}");
            Console.WriteLine($"RoPE base={Config.RopeBase}, scale={Config.RopeScale}, dim={_ropeDim} (NeoX, then QK-norm)");

            LoadWeights();
            FuseQKVWeights();
            FuseGateUpWeights();
            PrepareCudaQuantizedWeightsForInference();
            PrecomputeLayerFlags();

            int maxContextLength = ResolveConfiguredContextLength();
            int initialCacheLength = ResolveInitialCacheAllocationLength(maxContextLength);
            if (initialCacheLength < maxContextLength)
            {
                Console.WriteLine(
                    $"Initial {_backend} KV cache allocation: {initialCacheLength} tokens (grows on demand up to {maxContextLength}).");
            }

            InitKVCache(initialCacheLength, maxContextLength);
        }

        protected override bool SupportsSplitGateUpFfn => true;

        /// <summary>A plain GQA linear cache: layer count, head geometry and dtype are the
        /// whole identity of what a snapshot of it holds.</summary>
        public override string KVStateFingerprint =>
            $"hunyuan-dense|arch={Config.Architecture}|L={Config.NumLayers}|H={Config.NumHeads}|KV={Config.NumKVHeads}" +
            $"|kL={_attnKeyLen}|vL={_attnValLen}|rope={_ropeDim}|dtype={_kvCacheDtype.ToShortString()}";

        public override void PrepareForPrefill(int requiredContextTokens)
            => EnsureCacheCapacity(requiredContextTokens);

        /// <summary>
        /// llama.cpp hunyuan-vl: <c>base = rope_theta * alpha^(dim / (dim - 2))</c>
        /// when XDRoPE / NTK alpha is present. Hy-MT2 Q4 ships <c>scaling.type=none</c>.
        /// </summary>
        private void ApplyHunyuanRopeBase(string arch)
        {
            float alpha = _gguf.GetFloat32($"{arch}.rope.scaling.alpha", 0f);
            if (alpha <= 0f || _attnKeyLen <= 2)
                return;

            Config.RopeBase *= MathF.Pow(alpha, (float)_attnKeyLen / (float)(_attnKeyLen - 2));
            Console.WriteLine($"  Applied Hunyuan NTK RoPE alpha={alpha}, effective base={Config.RopeBase}");
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

                if (_quantWeights.TryGetValue(qName, out QuantizedWeight qw) &&
                    _quantWeights.TryGetValue(kName, out QuantizedWeight kw) &&
                    _quantWeights.TryGetValue(vName, out QuantizedWeight vw) &&
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
                else if (_weights.TryGetValue(qName, out Tensor qf) &&
                         _weights.TryGetValue(kName, out Tensor kf) &&
                         _weights.TryGetValue(vName, out Tensor vf))
                {
                    int qDim = (int)qf.Sizes[0];
                    int kDim = (int)kf.Sizes[0];
                    int vDim = (int)vf.Sizes[0];
                    int inDim = (int)qf.Sizes[1];
                    Tensor fusedTensor = new Tensor(_allocator, DType.Float32, qDim + kDim + vDim, inDim);
                    using (Tensor s0 = fusedTensor.Narrow(0, 0, qDim)) Ops.Copy(s0, qf);
                    using (Tensor s1 = fusedTensor.Narrow(0, qDim, kDim)) Ops.Copy(s1, kf);
                    using (Tensor s2 = fusedTensor.Narrow(0, qDim + kDim, vDim)) Ops.Copy(s2, vf);
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

        private void PrecomputeLayerFlags()
        {
            int numLayers = Config.NumLayers;
            _layerQkvFused = new bool[numLayers];
            _layerGateUpFused = new bool[numLayers];
            for (int l = 0; l < numLayers; l++)
            {
                string qkvName = $"blk.{l}.attn_qkv.weight";
                _layerQkvFused[l] = _quantWeights.ContainsKey(qkvName) || _weights.ContainsKey(qkvName);

                string gateUpName = $"blk.{l}.ffn_gate_up.weight";
                _layerGateUpFused[l] = _quantWeights.ContainsKey(gateUpName) || _weights.ContainsKey(gateUpName);

                if (!_weights.ContainsKey($"blk.{l}.attn_q_norm.weight") ||
                    !_weights.ContainsKey($"blk.{l}.attn_k_norm.weight"))
                {
                    throw new InvalidOperationException(
                        $"hunyuan-dense layer {l} is missing attn_q_norm / attn_k_norm weights.");
                }
            }
        }

        private void InitKVCache(int initialSeqLen, int maxSeqLen)
        {
            _maxContextLength = maxSeqLen;
            _kvCacheCapacity = initialSeqLen;
            ApplyModelAlignedKvCacheDefault(_quantWeights);
            DType kvDtype = _kvCacheDtype.ToDType();
            _kvCacheK = new Tensor[Config.NumLayers];
            _kvCacheV = new Tensor[Config.NumLayers];
            for (int l = 0; l < Config.NumLayers; l++)
            {
                _kvCacheK[l] = new Tensor(_allocator, kvDtype, Config.NumKVHeads, initialSeqLen, _attnKeyLen);
                _kvCacheV[l] = new Tensor(_allocator, kvDtype, Config.NumKVHeads, initialSeqLen, _attnValLen);
                InitializeCacheTensor(_kvCacheK[l]);
                InitializeCacheTensor(_kvCacheV[l]);
            }
            _cacheSeqLen = 0;
        }

        private void EnsureCacheCapacity(int requiredSeqLen)
        {
            if (requiredSeqLen <= _kvCacheCapacity)
                return;
            if (requiredSeqLen > _maxContextLength)
            {
                throw new InvalidOperationException(
                    $"Requested sequence length {requiredSeqLen} exceeds configured max context {_maxContextLength}.");
            }

            int newCapacity = Math.Max(_kvCacheCapacity, 1);
            while (newCapacity < requiredSeqLen)
                newCapacity = Math.Min(_maxContextLength, newCapacity * 2);

            DType kvDtype = _kvCacheDtype.ToDType();
            for (int l = 0; l < Config.NumLayers; l++)
            {
                Tensor newK = new Tensor(_allocator, kvDtype, Config.NumKVHeads, newCapacity, _attnKeyLen);
                Tensor newV = new Tensor(_allocator, kvDtype, Config.NumKVHeads, newCapacity, _attnValLen);
                InitializeCacheTensor(newK);
                InitializeCacheTensor(newV);

                if (_cacheSeqLen > 0)
                {
                    using Tensor srcK = _kvCacheK[l].Narrow(1, 0, _cacheSeqLen);
                    using Tensor dstK = newK.Narrow(1, 0, _cacheSeqLen);
                    Ops.Copy(dstK, srcK);

                    using Tensor srcV = _kvCacheV[l].Narrow(1, 0, _cacheSeqLen);
                    using Tensor dstV = newV.Narrow(1, 0, _cacheSeqLen);
                    Ops.Copy(dstV, srcV);
                }

                _kvCacheK[l].Dispose();
                _kvCacheV[l].Dispose();
                _kvCacheK[l] = newK;
                _kvCacheV[l] = newV;
            }

            _kvCacheCapacity = newCapacity;
            Console.WriteLine($"Expanded hunyuan-dense attention cache to {newCapacity} tokens.");
        }

        protected override void ResetKVCacheCore()
        {
            _cacheSeqLen = 0;
            _linearTicks = _attnTicks = _normTicks = _embTicks = _lmHeadTicks = _logitsCopyTicks = 0;
            _forwardCount = 0;
            _forwardSw.Reset();
            if (_kvCacheK == null)
                return;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                ResetCacheTensor(_kvCacheK[l]);
                ResetCacheTensor(_kvCacheV[l]);
            }
        }

        protected override void TruncateKVCacheCore(int tokenCount)
        {
            base.TruncateKVCacheCore(tokenCount);
            if (_kvCacheK == null)
                return;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                InvalidateTensorDeviceCache(_kvCacheK[l]);
                InvalidateTensorDeviceCache(_kvCacheV[l]);
            }
        }

        // ---- K/V state snapshot contract --------------------------------------------
        //
        // The server's continuous-batching engine needs one of two capabilities: a
        // batched paged forward, or a byte-exact extract/inject of a sequence's K/V rows.
        // Without either, InferenceEngineHost.TryGetEngine returned null and every chat
        // request was a 500. Every layer here is full causal attention over a LINEAR
        // cache (row == absolute position, no sliding window, no recurrent state), so a
        // snapshot restores exactly what a fresh prefill would write: concurrent requests
        // swap ownership of the single cache, and shared prompt prefixes are reused
        // across requests. Same contract, same helper and same device-cache invalidation
        // as the per-op Mistral 3 path this model's attention mirrors.

        public override bool SupportsKVStateSnapshot => _kvCacheK != null && _kvCacheV != null;

        // KVStateFingerprint is declared with the model's other overrides above (it also
        // carries the architecture and RoPE dimension).

        public override long ComputeKVBlockByteSize(int tokenCount)
            => KvBlockTransfer.ComputeBlockByteSize(_kvCacheK, _kvCacheV, tokenCount);

        public override bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        {
            if (!SupportsKVStateSnapshot)
                return false;
            return KvBlockTransfer.Extract(
                _allocator, _kvCacheK, _kvCacheV, _cacheSeqLen,
                startToken, tokenCount, destination);
        }

        public override bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source)
        {
            if (!SupportsKVStateSnapshot)
                return false;
            EnsureCacheCapacity(destToken + tokenCount);
            if (!KvBlockTransfer.Inject(
                    _allocator, _kvCacheK, _kvCacheV, _cacheSeqLen,
                    destToken, tokenCount, source))
            {
                return false;
            }
            _cacheSeqLen = destToken + tokenCount;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                InvalidateTensorDeviceCache(_kvCacheK[l]);
                InvalidateTensorDeviceCache(_kvCacheV[l]);
            }
            return true;
        }

        protected override float[] ForwardCore(int[] tokens)
        {
            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            EnsureCacheCapacity(startPos + seqLen);

            long t1 = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t1;

            for (int layer = 0; layer < Config.NumLayers; layer++)
                hidden = TransformerBlock(hidden, layer, seqLen, startPos);

            Tensor normed = RMSNormOp(hidden, "output_norm.weight");
            hidden.Dispose();

            Tensor lastHidden;
            if (seqLen > 1)
            {
                using Tensor narrowed = normed.Narrow(0, seqLen - 1, 1);
                lastHidden = Ops.NewContiguous(narrowed);
            }
            else
            {
                lastHidden = normed.CopyRef();
            }
            normed.Dispose();

            long t2 = Stopwatch.GetTimestamp();
            Tensor logitsTensor = LinearForward(lastHidden, "output.weight")
                ?? LinearForward(lastHidden, "token_embd.weight");
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

        private Tensor TransformerBlock(Tensor hidden, int layer, int seqLen, int startPos)
        {
            string p = $"blk.{layer}.";
            Tensor normed = RMSNormOp(hidden, p + "attn_norm.weight");
            Tensor attnOut = Attention(normed, layer, seqLen, startPos);
            normed.Dispose();

            Ops.Add(hidden, hidden, attnOut);
            attnOut.Dispose();

            if (_layerGateUpFused[layer] &&
                TryFusedDenseSwiGLUFFNInto(hidden, p + "ffn_norm.weight", p + "ffn_gate_up.weight", p + "ffn_down.weight"))
            {
                return hidden;
            }

            Tensor ffnNormed = RMSNormOp(hidden, p + "ffn_norm.weight");
            Tensor ffnOut = FFNLayer(ffnNormed, layer, seqLen);
            ffnNormed.Dispose();
            Ops.Add(hidden, hidden, ffnOut);
            ffnOut.Dispose();
            return hidden;
        }

        private Tensor FFNLayer(Tensor input, int layer, int seqLen)
        {
            string p = $"blk.{layer}.";
            if (_layerGateUpFused[layer])
                return FFN(input, p + "ffn_gate_up.weight", p + "ffn_down.weight", seqLen);

            Tensor gate = LinearForward(input, p + "ffn_gate.weight");
            Tensor up = LinearForward(input, p + "ffn_up.weight");
            Ops.SiLUMul(gate, gate, up);
            up.Dispose();
            Tensor down = LinearForward(gate, p + "ffn_down.weight");
            gate.Dispose();
            return down;
        }

        private Tensor Attention(Tensor input, int layer, int seqLen, int startPos)
        {
            int numHeads = Config.NumHeads;
            int numKVHeads = Config.NumKVHeads;
            int headDim = _attnKeyLen;
            int qDim = numHeads * headDim;
            int kDim = numKVHeads * headDim;
            int totalSeqLen = startPos + seqLen;
            float scale = 1.0f / MathF.Sqrt(headDim);
            string p = $"blk.{layer}.";

            Tensor qTensor;
            Tensor kTensor;
            Tensor vTensor;
            if (_layerQkvFused[layer])
            {
                Tensor qkvFused = LinearForward(input, p + "attn_qkv.weight");
                if (seqLen == 1)
                {
                    qTensor = qkvFused.Narrow(1, 0, qDim);
                    kTensor = qkvFused.Narrow(1, qDim, kDim);
                    vTensor = qkvFused.Narrow(1, qDim + kDim, kDim);
                    qkvFused.Dispose();
                }
                else
                {
                    using (Tensor qView = qkvFused.Narrow(1, 0, qDim))
                        qTensor = Ops.NewContiguous(qView);
                    using (Tensor kView = qkvFused.Narrow(1, qDim, kDim))
                        kTensor = Ops.NewContiguous(kView);
                    using (Tensor vView = qkvFused.Narrow(1, qDim + kDim, kDim))
                        vTensor = Ops.NewContiguous(vView);
                    qkvFused.Dispose();
                }
            }
            else
            {
                qTensor = LinearForward(input, p + "attn_q.weight");
                kTensor = LinearForward(input, p + "attn_k.weight");
                vTensor = LinearForward(input, p + "attn_v.weight");
            }

            // NeoX RoPE first, then per-head Q/K RMSNorm. Do not use Qwen35's
            // fused QKNorm+RoPE kernel — that applies the norm before rotation.
            ApplyNeoXRoPE(qTensor, numHeads, seqLen, startPos);
            ApplyNeoXRoPE(kTensor, numKVHeads, seqLen, startPos);
            ApplyQKNorm(qTensor, _weights[p + "attn_q_norm.weight"], numHeads, seqLen);
            ApplyQKNorm(kTensor, _weights[p + "attn_k_norm.weight"], numKVHeads, seqLen);

            long t0 = Stopwatch.GetTimestamp();
            if (seqLen == 1)
            {
                CopyToCacheDecode(_kvCacheK[layer], kTensor, _kvCacheV[layer], vTensor,
                    numKVHeads, headDim, startPos);
                kTensor.Dispose();
                vTensor.Dispose();

                Tensor attnResult = new Tensor(_allocator, DType.Float32, 1, numHeads * headDim);
                AttentionDecodePureCS(qTensor, _kvCacheK[layer], _kvCacheV[layer],
                    attnResult, numHeads, numKVHeads, headDim, totalSeqLen, scale);
                qTensor.Dispose();
                _attnTicks += Stopwatch.GetTimestamp() - t0;

                Tensor decodeOut = LinearForward(attnResult, p + "attn_output.weight");
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

            using Tensor kT = kExpanded.Transpose(1, 2);
            Tensor scores = new Tensor(_allocator, DType.Float32, numHeads, seqLen, totalSeqLen);
            Ops.AddmmBatch(scores, 0, scores, scale, qHeads, kT);
            qHeads.Dispose();
            kExpanded.Dispose();

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

            Tensor attnOut = new Tensor(_allocator, DType.Float32, numHeads, seqLen, _attnValLen);
            Ops.AddmmBatch(attnOut, 0, attnOut, 1.0f, scores, vExpanded);
            scores.Dispose();
            vExpanded.Dispose();

            Tensor flatOutput = ReshapeFromHeads(attnOut, numHeads, seqLen, _attnValLen);
            attnOut.Dispose();
            _attnTicks += Stopwatch.GetTimestamp() - t0;

            Tensor output = LinearForward(flatOutput, p + "attn_output.weight");
            flatOutput.Dispose();
            return output;
        }

        private void ApplyNeoXRoPE(Tensor data, int numHeads, int seqLen, int startPos)
        {
            int headDim = _attnKeyLen;
            int totalRows = seqLen * numHeads;
            int[] positions = new int[totalRows];
            for (int s = 0; s < seqLen; s++)
            {
                for (int h = 0; h < numHeads; h++)
                    positions[s * numHeads + h] = startPos + s;
            }

            using Tensor posTensor = CreateIntTensorOn(data.Storage.Allocator, positions, totalRows);
            using Tensor reshaped = data.View(1, seqLen, numHeads, headDim);
            Ops.RoPEEx(reshaped, reshaped, posTensor, _ropeDim, 2, 0,
                Config.RopeBase, 1.0f / Config.RopeScale,
                0.0f, 1.0f, 0.0f, 0.0f);
        }

        private void ApplyQKNorm(Tensor data, Tensor alpha, int numHeads, int seqLen)
        {
            int headDim = _attnKeyLen;
            if (seqLen == 1 && _backend != BackendType.Mlx && _backend != BackendType.Cuda)
            {
                RMSNormInPlaceCpu(data, alpha, numHeads, headDim, Config.Eps);
                return;
            }

            using Tensor reshaped = data.View(seqLen * numHeads, headDim);
            Ops.RMSNorm(reshaped, reshaped, alpha, null, Config.Eps);
        }

        public override void Dispose()
        {
            if (_kvCacheK != null)
            {
                foreach (Tensor t in _kvCacheK)
                    t?.Dispose();
            }
            if (_kvCacheV != null)
            {
                foreach (Tensor t in _kvCacheV)
                    t?.Dispose();
            }
            base.Dispose();
        }
    }
}
