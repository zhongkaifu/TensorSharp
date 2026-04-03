using System;
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
        private Tensor _visionEmbeddings;
        private int _visionEmbedStart = -1;

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
            FuseGateUpWeights();
            InitCaches(4096);
            PrecomputeRoPE();
            InitGDNBuffers();
        }

        private void InitGDNBuffers()
        {
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;
            int qkDim = _headKDim * _numKHeads;
            _gdnQ = new float[qkDim];
            _gdnK = new float[qkDim];
            _gdnV = new float[_headVDim * _numVHeads];
            _gdnQExp = new float[_headKDim * _numVHeads];
            _gdnKExp = new float[_headKDim * _numVHeads];

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

        public override float[] Forward(int[] tokens)
        {
            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;

            long t1 = Stopwatch.GetTimestamp();
            Tensor hidden = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t1;

            if (_visionEmbeddings != null && startPos == 0)
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
            int totalSeqLen = startPos + seqLen;

            // Q projection: outputs [seqLen, numHeads * headDim * 2] (Q + gate interleaved per head)
            Tensor qFull = LinearForward(input, prefix + "attn_q.weight");

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

            // K, V projections
            Tensor kTensor = LinearForward(input, prefix + "attn_k.weight");
            Tensor vTensor = LinearForward(input, prefix + "attn_v.weight");

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
        /// GatedDeltaNet recurrent block: SSM conv1d → gated delta net → norm + gate → output.
        /// Only supports seqLen=1 autoregressive mode for now (chunked prefill is complex).
        /// Falls back to sequential token-by-token processing for prefill.
        /// </summary>
        private Tensor RecurrentBlock(Tensor hidden, int layer, int seqLen, int startPos)
        {
            string prefix = $"blk.{layer}.";
            Tensor residual = hidden;

            Tensor normed = RMSNormOp(hidden, prefix + "attn_norm.weight");

            Tensor attnOut;
            if (seqLen == 1)
            {
                attnOut = GatedDeltaNetDecode(normed, layer, prefix);
            }
            else
            {
                int dim = Config.HiddenSize;
                attnOut = new Tensor(_allocator, DType.Float32, seqLen, dim);
                unsafe
                {
                    float* outBase = GetFloatPtr(attnOut);
                    for (int s = 0; s < seqLen; s++)
                    {
                        using var tokenInput = normed.Narrow(0, s, 1);
                        using var tokenOut = GatedDeltaNetDecode(tokenInput, layer, prefix);
                        Buffer.MemoryCopy(GetFloatPtr(tokenOut), outBase + (long)s * dim,
                            dim * sizeof(float), dim * sizeof(float));
                    }
                }
            }
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
        /// Single-token GatedDeltaNet decode step.
        /// Uses batched tensor matmul (AddmmBatch) for the recurrent state update
        /// and tensor ops (RMSNorm, SiLU, Mul) for gated normalization.
        /// </summary>
        private unsafe Tensor GatedDeltaNetDecode(Tensor input, int layer, string prefix)
        {
            long t0 = Stopwatch.GetTimestamp();
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;
            int qkDim = _headKDim * _numKHeads;
            int vDim = _headVDim * _numVHeads;

            Tensor qkvRaw = LinearForward(input, prefix + "attn_qkv.weight");
            Tensor zRaw = LinearForward(input, prefix + "attn_gate.weight");
            Tensor betaRaw = LinearForward(input, prefix + "ssm_beta.weight");
            Tensor alphaRaw = LinearForward(input, prefix + "ssm_alpha.weight");

            float* qkvPtr = GetFloatPtr(qkvRaw);
            float* betaPtr = GetFloatPtr(betaRaw);
            float* alphaPtr = GetFloatPtr(alphaRaw);

            // Conv1d → write directly to tensor buffer
            int convDim = _convKernel - 1;
            float[] convState = _convState[layer];
            float* convWPtr = GetFloatPtr(_weights[prefix + "ssm_conv1d.weight"]);
            float* convOutPtr = GetFloatPtr(_gdnConvOutT);

            for (int ch = 0; ch < qkvDim; ch++)
            {
                float sum = 0;
                for (int ki = 0; ki < _convKernel; ki++)
                {
                    float stateVal = ki < convDim ? convState[ki * qkvDim + ch] : qkvPtr[ch];
                    sum += stateVal * convWPtr[ch * _convKernel + ki];
                }
                convOutPtr[ch] = sum;
            }

            for (int c = 0; c < convDim - 1; c++)
                Array.Copy(convState, (c + 1) * qkvDim, convState, c * qkvDim, qkvDim);
            for (int d = 0; d < qkvDim; d++)
                convState[(convDim - 1) * qkvDim + d] = qkvPtr[d];

            // SiLU activation via tensor op
            Ops.SiLU(_gdnConvOutT, _gdnConvOutT);

            // Split Q, K, V from tensor buffer
            float* siluPtr = GetFloatPtr(_gdnConvOutT);
            fixed (float* qDst = _gdnQ, kDst = _gdnK, vDst = _gdnV)
            {
                Buffer.MemoryCopy(siluPtr, qDst, qkDim * sizeof(float), qkDim * sizeof(float));
                Buffer.MemoryCopy(siluPtr + qkDim, kDst, qkDim * sizeof(float), qkDim * sizeof(float));
                Buffer.MemoryCopy(siluPtr + 2 * qkDim, vDst, vDim * sizeof(float), vDim * sizeof(float));
            }

            // Expand Q, K heads if needed
            float[] qExp, kExp;
            if (_numKHeads != _numVHeads)
            {
                qExp = _gdnQExp;
                kExp = _gdnKExp;
                for (int h = 0; h < _numVHeads; h++)
                {
                    int srcHead = h % _numKHeads;
                    Array.Copy(_gdnQ, srcHead * _headKDim, qExp, h * _headKDim, _headKDim);
                    Array.Copy(_gdnK, srcHead * _headKDim, kExp, h * _headKDim, _headKDim);
                }
            }
            else
            {
                qExp = _gdnQ;
                kExp = _gdnK;
            }

            L2NormalizePerHead(qExp, _numVHeads, _headKDim);
            L2NormalizePerHead(kExp, _numVHeads, _headKDim);

            // Copy Q, K, V into pre-allocated tensor buffers
            float* kBufPtr = GetFloatPtr(_gdnKBuf);
            float* qBufPtr = GetFloatPtr(_gdnQBuf);
            float* vBufPtr = GetFloatPtr(_gdnVBuf);
            int kqBytes = _headKDim * _numVHeads * sizeof(float);
            int vBytes = _headVDim * _numVHeads * sizeof(float);
            fixed (float* kSrc = kExp, qSrc = qExp, vSrc = _gdnV)
            {
                Buffer.MemoryCopy(kSrc, kBufPtr, kqBytes, kqBytes);
                Buffer.MemoryCopy(qSrc, qBufPtr, kqBytes, kqBytes);
                Buffer.MemoryCopy(vSrc, vBufPtr, vBytes, vBytes);
            }

            // Scale Q via tensor op
            Ops.Mul(_gdnQBuf, _gdnQBuf, 1.0f / MathF.Sqrt(_headVDim));

            // Create views for batched matmul
            using var k_col = _gdnKBuf.View(_numVHeads, _headKDim, 1);
            using var k_row = k_col.Transpose(1, 2);  // [numVHeads, 1, headKDim]
            using var q_col = _gdnQBuf.View(_numVHeads, _headKDim, 1);

            // Recurrent state update using batched tensor operations
            Tensor state = _deltaStateTensor[layer];
            float* dtBiasPtr = GetFloatPtr(_weights[prefix + "ssm_dt.bias"]);
            float* aPtr = GetFloatPtr(_weights[prefix + "ssm_a"]);

            float* statePtr = GetFloatPtr(state);
            int statePerHead = _headVDim * _headKDim;
            for (int h = 0; h < _numVHeads; h++)
            {
                float alphaBiased = alphaPtr[h] + dtBiasPtr[h];
                float gateH = MathF.Log(1.0f + MathF.Exp(alphaBiased)) * aPtr[h];
                VecScale(statePtr + h * statePerHead, MathF.Exp(gateH), statePerHead);
            }

            // kvMem = state @ k_col → [numVHeads, headVDim, 1]
            Ops.AddmmBatch(_gdnKvMemBuf, 0, _gdnKvMemBuf, 1.0f, state, k_col);

            // delta = (v - kvMem) * sigmoid(beta), stored in kvMem buffer
            float* kvMemPtr = GetFloatPtr(_gdnKvMemBuf);
            for (int h = 0; h < _numVHeads; h++)
            {
                float betaH = 1.0f / (1.0f + MathF.Exp(-betaPtr[h]));
                VecSubScale(kvMemPtr + h * _headVDim, vBufPtr + h * _headVDim,
                    kvMemPtr + h * _headVDim, betaH, _headVDim);
            }

            // state += delta @ k^T (batched rank-1 update)
            Ops.AddmmBatch(state, 1.0f, state, 1.0f, _gdnKvMemBuf, k_row);

            // coreOut = state @ q_col → [numVHeads, headVDim, 1]
            Ops.AddmmBatch(_gdnCoreOutBuf, 0, _gdnCoreOutBuf, 1.0f, state, q_col);

            // Gated normalization: result = SiLU(z) * RMSNorm(coreOut)
            using var coreOut2d = _gdnCoreOutBuf.View(_numVHeads, _headVDim);
            using var gatedOut2d = _gdnGatedOutT.View(_numVHeads, _headVDim);
            var ssmNormW = _weights[prefix + "ssm_norm.weight"];
            Ops.RMSNorm(gatedOut2d, coreOut2d, ssmNormW, null, Config.Eps);

            using var zView = zRaw.View(_numVHeads, _headVDim);
            Ops.SiLUMul(gatedOut2d, zView, gatedOut2d);

            qkvRaw.Dispose();
            zRaw.Dispose();
            betaRaw.Dispose();
            alphaRaw.Dispose();

            Tensor output = LinearForward(_gdnGatedOutT, prefix + "ssm_out.weight");

            _attnTicks += Stopwatch.GetTimestamp() - t0;
            return output;
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

        #endregion

        #region Vision Support

        public void LoadVisionEncoder(string mmProjPath)
        {
            VisionEncoder = new Qwen35VisionEncoder(mmProjPath, _allocator);
        }

        public void SetVisionEmbeddings(Tensor visionEmbeddings, int startPosition)
        {
            _visionEmbeddings?.Dispose();
            _visionEmbeddings = visionEmbeddings;
            _visionEmbedStart = startPosition;
        }

        /// <summary>
        /// Inject vision embeddings into text embeddings at the image_pad token positions.
        /// </summary>
        private unsafe void InjectVisionEmbeddings(Tensor textEmbeddings, int seqLen)
        {
            if (_visionEmbeddings == null || _visionEmbedStart < 0)
                return;

            int startPos = _visionEmbedStart;
            int numVisionTokens = (int)_visionEmbeddings.Sizes[0];
            int dim = Config.HiddenSize;
            int projDim = (int)_visionEmbeddings.Sizes[1];

            if (projDim != dim)
            {
                Console.WriteLine($"Warning: Vision embedding dim ({projDim}) != text hidden dim ({dim}). Skipping injection.");
                return;
            }

            if (startPos + numVisionTokens > seqLen)
            {
                Console.WriteLine($"Warning: Vision tokens ({numVisionTokens}) exceed sequence at position {startPos}. Skipping.");
                return;
            }

            float* textPtr = GetFloatPtr(textEmbeddings);
            float* visPtr = GetFloatPtr(_visionEmbeddings);
            int bytes = numVisionTokens * dim * sizeof(float);
            Buffer.MemoryCopy(visPtr, textPtr + startPos * dim, bytes, bytes);

            Console.WriteLine($"  Injected {numVisionTokens} vision embeddings at position {startPos}");

            _visionEmbeddings.Dispose();
            _visionEmbeddings = null;
            _visionEmbedStart = -1;
        }

        #endregion

        public override void Dispose()
        {
            VisionEncoder?.Dispose();
            _visionEmbeddings?.Dispose();

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
