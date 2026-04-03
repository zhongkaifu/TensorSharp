using System;
using System.Diagnostics;
using TensorSharp;
using TensorSharp.GGML;

namespace InferenceEngine
{
    public class Qwen3Model : ModelBase
    {
        private Tensor[] _kvCacheK;
        private Tensor[] _kvCacheV;

        private string[][] _layerWeightNames;
        private int[] _decodeQPositions;
        private int[] _decodeKPositions;
        private float[] _ropeFreqs;

        private ModelDecodeArrays _modelDecodeArrays;

        public Qwen3Model(string ggufPath, BackendType backend)
            : base(ggufPath, backend)
        {
            string arch = _gguf.GetString("general.architecture") ?? "qwen3";
            Config = new ModelConfig { Architecture = arch };
            ParseBaseConfig();

            Config.NumKVHeads = (int)_gguf.GetUint32($"{arch}.attention.head_count_kv");

            ParseTokenizer();

            Console.WriteLine($"Model: {arch}, Layers={Config.NumLayers}, Hidden={Config.HiddenSize}, " +
                $"Heads={Config.NumHeads}, KVHeads={Config.NumKVHeads}, HeadDim={Config.HeadDim}, Vocab={Config.VocabSize}");
            Console.WriteLine($"RoPE base={Config.RopeBase}, scale={Config.RopeScale}, eps={Config.Eps}");

            LoadWeights();
            FuseQKVWeights();
            FuseGateUpWeights();
            InitKVCache(4096);
            PrecomputeConstants();
            BuildModelDecodeArrays();
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
                    long totalBytes = qw.RawBytes + kw.RawBytes + vw.RawBytes;
                    IntPtr fusedPtr = GgmlBasicOps.AlignedAlloc(totalBytes);
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

        private void PrecomputeConstants()
        {
            int numLayers = Config.NumLayers;
            int headDim = Config.HeadDim;

            _layerWeightNames = new string[numLayers][];
            for (int l = 0; l < numLayers; l++)
            {
                string p = $"blk.{l}.";
                _layerWeightNames[l] = new[]
                {
                    p + "attn_norm.weight",
                    p + "attn_qkv.weight",
                    p + "attn_q_norm.weight",
                    p + "attn_k_norm.weight",
                    p + "attn_output.weight",
                    p + "ffn_norm.weight",
                    p + "ffn_gate_up.weight",
                    p + "ffn_down.weight",
                };
            }

            _decodeQPositions = new int[Config.NumHeads];
            _decodeKPositions = new int[Config.NumKVHeads];

            int halfDim = headDim / 2;
            float freqScale = 1.0f / Config.RopeScale;
            _ropeFreqs = new float[halfDim];
            for (int i = 0; i < halfDim; i++)
                _ropeFreqs[i] = freqScale / MathF.Pow(Config.RopeBase, (2.0f * i) / headDim);
        }

        private void InitKVCache(int maxSeqLen)
        {
            int numKVHeads = Config.NumKVHeads;
            int headDim = Config.HeadDim;
            _kvCacheK = new Tensor[Config.NumLayers];
            _kvCacheV = new Tensor[Config.NumLayers];
            for (int l = 0; l < Config.NumLayers; l++)
            {
                _kvCacheK[l] = new Tensor(_allocator, DType.Float32, numKVHeads, maxSeqLen, headDim);
                _kvCacheV[l] = new Tensor(_allocator, DType.Float32, numKVHeads, maxSeqLen, headDim);
                Ops.Fill(_kvCacheK[l], 0);
                Ops.Fill(_kvCacheV[l], 0);
            }
            _cacheSeqLen = 0;
        }

        public override void ResetKVCache()
        {
            for (int l = 0; l < Config.NumLayers; l++)
            {
                Ops.Fill(_kvCacheK[l], 0);
                Ops.Fill(_kvCacheV[l], 0);
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

            if (seqLen == 1 && IsGgmlBackend && _modelDecodeArrays != null)
            {
                long t0 = Stopwatch.GetTimestamp();
                NativeTransformerModelDecode(hidden, startPos);
                _linearTicks += Stopwatch.GetTimestamp() - t0;
            }
            else
            {
                for (int layer = 0; layer < Config.NumLayers; layer++)
                {
                    hidden = TransformerBlock(hidden, layer, seqLen, startPos);
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
            string[] wn = _layerWeightNames[layer];

            if (seqLen == 1 && IsGgmlBackend && _quantWeights.ContainsKey(wn[1]))
            {
                long t0 = Stopwatch.GetTimestamp();
                NativeTransformerLayerDecode(hidden, layer, wn, startPos);
                _linearTicks += Stopwatch.GetTimestamp() - t0;
                return hidden;
            }

            Tensor normed = RMSNormOp(hidden, wn[0]);
            Tensor attnOut = Attention(normed, layer, wn, seqLen, startPos);
            normed.Dispose();

            Ops.Add(hidden, hidden, attnOut);
            attnOut.Dispose();

            Tensor normed2 = RMSNormOp(hidden, wn[5]);
            Tensor ffnOut = FFN(normed2, wn[6], wn[7], seqLen);
            normed2.Dispose();

            Ops.Add(hidden, hidden, ffnOut);
            ffnOut.Dispose();

            return hidden;
        }

        private Tensor Attention(Tensor input, int layer, string[] wn, int seqLen, int startPos)
        {
            int numHeads = Config.NumHeads;
            int numKVHeads = Config.NumKVHeads;
            int headDim = Config.HeadDim;
            int qDim = numHeads * headDim;
            int kDim = numKVHeads * headDim;
            int totalSeqLen = startPos + seqLen;
            float scale = 1.0f / MathF.Sqrt(headDim);

            Tensor qkvFused = LinearForward(input, wn[1]);
            Tensor qTensor, kTensor, vTensor;
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

            qTensor = ApplyQKNormInPlace(qTensor, wn[2], numHeads, seqLen);
            kTensor = ApplyQKNormInPlace(kTensor, wn[3], numKVHeads, seqLen);

            if (seqLen == 1)
            {
                ApplyRoPEDecodeInPlace(qTensor, numHeads, headDim, startPos);
                ApplyRoPEDecodeInPlace(kTensor, numKVHeads, headDim, startPos);
            }
            else
            {
                qTensor = ApplyRoPEInPlace(qTensor, numHeads, headDim, seqLen, startPos);
                kTensor = ApplyRoPEInPlace(kTensor, numKVHeads, headDim, seqLen, startPos);
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

                Tensor decodeOut = LinearForward(attnResult, wn[4]);
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

            Tensor output = LinearForward(flatOutput, wn[4]);
            flatOutput.Dispose();

            return output;
        }

        private Tensor ApplyQKNormInPlace(Tensor data, string weightName, int numHeads, int seqLen)
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

        private unsafe void ApplyRoPEDecodeInPlace(Tensor data, int numHeads, int headDim, int position)
        {
            int halfDim = headDim / 2;
            float[] freqs = _ropeFreqs;
            float* ptr = GetFloatPtr(data);

            float* cosTable = stackalloc float[halfDim];
            float* sinTable = stackalloc float[halfDim];
            for (int i = 0; i < halfDim; i++)
            {
                float theta = position * freqs[i];
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
                null, reshaped, posTensor, headDim, 2, 0,
                Config.RopeBase, 1.0f / Config.RopeScale,
                0.0f, 1.0f, 0.0f, 0.0f);

            data.Dispose();

            Tensor flat = result.View(seqLen, numHeads * headDim);
            result.Dispose();
            return flat;
        }

        #region Native decode paths

        private unsafe void NativeTransformerLayerDecode(Tensor hidden, int layer, string[] wn, int startPos)
        {
            float* hiddenPtr = GetFloatPtr(hidden);
            int hiddenSize = Config.HiddenSize;

            var attnNormW = _weights[wn[0]];
            var qkvW = _quantWeights[wn[1]];
            var qNormW = _weights[wn[2]];
            var kNormW = _weights[wn[3]];
            var oW = _quantWeights[wn[4]];
            var ffnNormW = _weights[wn[5]];
            var guW = _quantWeights[wn[6]];
            var downW = _quantWeights[wn[7]];

            int maxSeqLen = (int)_kvCacheK[layer].Sizes[1];

            GgmlBasicOps.TransformerLayerDecode(
                (IntPtr)hiddenPtr, hiddenSize,
                (IntPtr)GetFloatPtr(attnNormW),
                qkvW.Data, qkvW.GgmlType, qkvW.Ne0, qkvW.Ne1, qkvW.RawBytes,
                (IntPtr)GetFloatPtr(qNormW), (IntPtr)GetFloatPtr(kNormW), Config.HeadDim,
                oW.Data, oW.GgmlType, oW.Ne0, oW.Ne1, oW.RawBytes,
                (IntPtr)GetFloatPtr(ffnNormW),
                guW.Data, guW.GgmlType, guW.Ne0, guW.Ne1, guW.RawBytes,
                downW.Data, downW.GgmlType, downW.Ne0, downW.Ne1, downW.RawBytes,
                (IntPtr)GetFloatPtr(_kvCacheK[layer]), (IntPtr)GetFloatPtr(_kvCacheV[layer]),
                Config.NumHeads, Config.NumKVHeads,
                maxSeqLen, startPos,
                Config.Eps, Config.RopeBase, 1.0f / Config.RopeScale,
                Config.IntermediateSize, 2);
        }

        private class ModelDecodeArrays
        {
            public IntPtr[] AttnNorm, Qkv, QNorm, KNorm, O, FfnNorm, Gu, Down, KCache, VCache;
            public int QkvType, OType, GuType, DownType;
            public long QkvNe0, QkvNe1, QkvBytes;
            public long ONe0, ONe1, OBytes;
            public long GuNe0, GuNe1, GuBytes;
            public long DownNe0, DownNe1, DownBytes;
        }

        private unsafe void BuildModelDecodeArrays()
        {
            int numLayers = Config.NumLayers;
            if (!IsGgmlBackend) return;

            string[] wn0 = _layerWeightNames[0];
            if (!_quantWeights.ContainsKey(wn0[1])) return;

            var arr = new ModelDecodeArrays();
            arr.AttnNorm = new IntPtr[numLayers];
            arr.Qkv = new IntPtr[numLayers];
            arr.QNorm = new IntPtr[numLayers];
            arr.KNorm = new IntPtr[numLayers];
            arr.O = new IntPtr[numLayers];
            arr.FfnNorm = new IntPtr[numLayers];
            arr.Gu = new IntPtr[numLayers];
            arr.Down = new IntPtr[numLayers];
            arr.KCache = new IntPtr[numLayers];
            arr.VCache = new IntPtr[numLayers];

            var qkv0 = _quantWeights[wn0[1]];
            arr.QkvType = qkv0.GgmlType; arr.QkvNe0 = qkv0.Ne0; arr.QkvNe1 = qkv0.Ne1; arr.QkvBytes = qkv0.RawBytes;
            var o0 = _quantWeights[wn0[4]];
            arr.OType = o0.GgmlType; arr.ONe0 = o0.Ne0; arr.ONe1 = o0.Ne1; arr.OBytes = o0.RawBytes;
            var gu0 = _quantWeights[wn0[6]];
            arr.GuType = gu0.GgmlType; arr.GuNe0 = gu0.Ne0; arr.GuNe1 = gu0.Ne1; arr.GuBytes = gu0.RawBytes;
            var down0 = _quantWeights[wn0[7]];
            arr.DownType = down0.GgmlType; arr.DownNe0 = down0.Ne0; arr.DownNe1 = down0.Ne1; arr.DownBytes = down0.RawBytes;

            for (int l = 0; l < numLayers; l++)
            {
                string[] wn = _layerWeightNames[l];
                arr.AttnNorm[l] = (IntPtr)GetFloatPtr(_weights[wn[0]]);
                arr.Qkv[l] = _quantWeights[wn[1]].Data;
                arr.QNorm[l] = (IntPtr)GetFloatPtr(_weights[wn[2]]);
                arr.KNorm[l] = (IntPtr)GetFloatPtr(_weights[wn[3]]);
                arr.O[l] = _quantWeights[wn[4]].Data;
                arr.FfnNorm[l] = (IntPtr)GetFloatPtr(_weights[wn[5]]);
                arr.Gu[l] = _quantWeights[wn[6]].Data;
                arr.Down[l] = _quantWeights[wn[7]].Data;
                arr.KCache[l] = (IntPtr)GetFloatPtr(_kvCacheK[l]);
                arr.VCache[l] = (IntPtr)GetFloatPtr(_kvCacheV[l]);
            }

            _modelDecodeArrays = arr;
        }

        private unsafe void NativeTransformerModelDecode(Tensor hidden, int startPos)
        {
            float* hiddenPtr = GetFloatPtr(hidden);
            int maxSeqLen = (int)_kvCacheK[0].Sizes[1];
            var a = _modelDecodeArrays;

            GgmlBasicOps.TransformerModelDecode(
                (IntPtr)hiddenPtr, Config.HiddenSize, Config.NumLayers,
                a.AttnNorm, a.Qkv, a.QNorm, a.KNorm,
                a.O, a.FfnNorm, a.Gu, a.Down,
                a.KCache, a.VCache,
                a.QkvType, a.QkvNe0, a.QkvNe1, a.QkvBytes,
                a.OType, a.ONe0, a.ONe1, a.OBytes,
                a.GuType, a.GuNe0, a.GuNe1, a.GuBytes,
                a.DownType, a.DownNe0, a.DownNe1, a.DownBytes,
                Config.HeadDim, Config.NumHeads, Config.NumKVHeads,
                maxSeqLen, startPos,
                Config.Eps, Config.RopeBase, 1.0f / Config.RopeScale,
                Config.IntermediateSize, 2);
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
