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
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.GGML;

namespace InferenceEngine
{
    public enum BackendType
    {
        Cpu,
        GgmlCpu,
        GgmlMetal,
        GgmlCuda,
    }

    public class ModelConfig
    {
        public string Architecture { get; set; }
        public int HiddenSize { get; set; }
        public int NumHeads { get; set; }
        public int NumKVHeads { get; set; }
        public int KeyLength { get; set; }
        public int ValueLength { get; set; }
        public float Eps { get; set; }
        public float RopeBase { get; set; }
        public float RopeScale { get; set; } = 1f;
        public int NumLayers { get; set; }
        public int VocabSize { get; set; }
        public int IntermediateSize { get; set; }
        public string ChatTemplate { get; set; }

        public int HeadDim => KeyLength > 0 ? KeyLength : (ValueLength > 0 ? ValueLength : HiddenSize / NumHeads);
    }

    public class QuantizedWeight : IDisposable
    {
        public IntPtr Data { get; }
        public int GgmlType { get; }
        public long Ne0 { get; }
        public long Ne1 { get; }
        public long RawBytes { get; }
        private readonly bool _aligned;

        public QuantizedWeight(byte[] raw, int ggmlType, long ne0, long ne1)
        {
            GgmlType = ggmlType;
            Ne0 = ne0;
            Ne1 = ne1;
            RawBytes = raw.Length;
            Data = GgmlBasicOps.AlignedAlloc(raw.Length);
            _aligned = true;
            Marshal.Copy(raw, 0, Data, raw.Length);
        }

        public QuantizedWeight(IntPtr data, long rawBytes, int ggmlType, long ne0, long ne1)
        {
            Data = data;
            RawBytes = rawBytes;
            GgmlType = ggmlType;
            Ne0 = ne0;
            Ne1 = ne1;
            _aligned = true;
        }

        public void Dispose()
        {
            if (Data != IntPtr.Zero)
            {
                if (_aligned)
                    GgmlBasicOps.AlignedFree(Data);
                else
                    Marshal.FreeHGlobal(Data);
            }
        }
    }

    public abstract class ModelBase : IDisposable
    {
        public ModelConfig Config { get; protected set; }
        public ITokenizer Tokenizer { get; protected set; }

        protected readonly GgufFile _gguf;
        private readonly GgmlContext _ggmlContext;
        protected readonly IAllocator _allocator;
        protected readonly BackendType _backend;

        protected readonly Dictionary<string, Tensor> _weights = new();
        protected readonly Dictionary<string, QuantizedWeight> _quantWeights = new();

        protected int _cacheSeqLen;
        protected float[] _logitsBuffer;

        // Timing
        protected long _linearTicks;
        protected long _attnTicks;
        protected long _normTicks;
        protected long _embTicks, _lmHeadTicks, _logitsCopyTicks;
        protected int _forwardCount;
        protected Stopwatch _forwardSw = new Stopwatch();

        protected ModelBase(string ggufPath, BackendType backend)
        {
            _backend = backend;
            switch (backend)
            {
                case BackendType.GgmlCpu:
                    _ggmlContext = new GgmlContext(new[] { 0 }, GgmlBackendType.Cpu);
                    _allocator = new GgmlAllocator(_ggmlContext, 0);
                    break;
                case BackendType.GgmlMetal:
                    _ggmlContext = new GgmlContext(new[] { 0 }, GgmlBackendType.Metal);
                    _allocator = new GgmlAllocator(_ggmlContext, 0);
                    break;
                case BackendType.GgmlCuda:
                    _ggmlContext = new GgmlContext(new[] { 0 }, GgmlBackendType.Cuda);
                    _allocator = new GgmlAllocator(_ggmlContext, 0);
                    break;
                case BackendType.Cpu:
                    _allocator = new CpuAllocator(BlasEnum.DotNet);
                    break;
                default:
                    throw new ArgumentException($"Unsupported backend: {backend}");
            }
            Console.WriteLine($"Backend: {backend}");

            _gguf = new GgufFile(ggufPath);
        }

        protected bool IsGgmlBackend => _backend == BackendType.GgmlCpu ||
                                        _backend == BackendType.GgmlMetal ||
                                        _backend == BackendType.GgmlCuda;

        protected void ParseBaseConfig()
        {
            string arch = Config.Architecture;
            Config.NumLayers = (int)_gguf.GetUint32($"{arch}.block_count");
            Config.HiddenSize = (int)_gguf.GetUint32($"{arch}.embedding_length");
            Config.NumHeads = (int)_gguf.GetUint32($"{arch}.attention.head_count");
            Config.NumKVHeads = (int)_gguf.GetUint32($"{arch}.attention.head_count_kv", (uint)Config.NumHeads);
            Config.Eps = _gguf.GetFloat32($"{arch}.attention.layer_norm_rms_epsilon");
            Config.RopeBase = _gguf.GetFloat32($"{arch}.rope.freq_base");
            Config.RopeScale = _gguf.GetFloat32($"{arch}.rope.scaling.factor", 1f);
            Config.ChatTemplate = _gguf.GetString("tokenizer.chat_template");

            Config.KeyLength = (int)_gguf.GetUint32($"{arch}.attention.key_length", 0);
            Config.ValueLength = (int)_gguf.GetUint32($"{arch}.attention.value_length", 0);
            Config.IntermediateSize = (int)_gguf.GetUint32($"{arch}.feed_forward_length", 0);
        }

        protected void ParseTokenizer()
        {
            var vocabTokens = _gguf.GetStringArray("tokenizer.ggml.tokens");
            Config.VocabSize = vocabTokens.Length;

            var tokenTypes = _gguf.GetInt32Array("tokenizer.ggml.token_type");
            int bosId = (int)_gguf.GetUint32("tokenizer.ggml.bos_token_id");
            int eosId = (int)_gguf.GetUint32("tokenizer.ggml.eos_token_id");
            bool addBos = _gguf.GetBool("tokenizer.ggml.add_bos_token", false);
            bool addEos = _gguf.GetBool("tokenizer.ggml.add_eos_token", false);

            var eosIds = new List<int> { eosId };
            var extraEos = _gguf.GetInt32Array("tokenizer.ggml.eos_token_ids");
            if (extraEos != null)
                eosIds.AddRange(extraEos);

            string tokenizerModel = _gguf.GetString("tokenizer.ggml.model", "gpt2");

            if (tokenizerModel == "llama" || tokenizerModel == "t5" || tokenizerModel == "gemma4")
            {
                var scores = _gguf.GetFloatArray("tokenizer.ggml.scores");

                int eotId = (int)_gguf.GetUint32("tokenizer.ggml.eot_token_id", 106);
                if (!eosIds.Contains(eotId))
                    eosIds.Add(eotId);

                Tokenizer = new SentencePieceTokenizer(vocabTokens, tokenTypes, scores,
                    bosId, eosIds.ToArray(), addBos, addEos);
            }
            else
            {
                var merges = _gguf.GetStringArray("tokenizer.ggml.merges");
                Tokenizer = new BpeTokenizer(vocabTokens, tokenTypes, merges,
                    bosId, eosIds.ToArray(), addBos, addEos);
            }
        }

        protected virtual bool IsQuantizedLinearWeight(GgufTensorInfo info)
        {
            if (info.Type == GgmlTensorType.F32)
                return false;
            if (!IsGgmlBackend && info.Type == GgmlTensorType.F16)
                return false;
            if (info.Shape.Length == 2)
                return true;
            // 3D tensors: MoE expert weights (ffn_gate_exps, ffn_up_exps, ffn_down_exps)
            if (info.Shape.Length == 3 && info.Name.Contains("_exps."))
                return true;
            return false;
        }

        protected void LoadWeights()
        {
            Console.Write("Loading model weights...");
            int countF32 = 0;
            int countQuant = 0;
            long totalQuantBytes = 0;
            long totalF32Bytes = 0;
            foreach (var kv in _gguf.Tensors)
            {
                var info = kv.Value;
                long byteCount = _gguf.GetTensorByteCount(info);

                if (IsGgmlBackend && IsQuantizedLinearWeight(info))
                {
                    long ne0 = (long)info.Shape[0];
                    long ne1 = (long)info.Shape[1];

                    if (info.Shape.Length == 3 && info.Name.Contains("_exps."))
                    {
                        // 3D MoE expert tensor: split into per-expert 2D quantized weights
                        int numExperts = (int)info.Shape[2];
                        long perExpertBytes = byteCount / numExperts;
                        IntPtr bulkPtr = GgmlBasicOps.AlignedAlloc(byteCount);
                        _gguf.ReadTensorDataToNative(info, bulkPtr, byteCount);

                        string baseName = info.Name;
                        if (baseName.EndsWith(".weight"))
                            baseName = baseName.Substring(0, baseName.Length - 7);

                        for (int e = 0; e < numExperts; e++)
                        {
                            IntPtr expertPtr = GgmlBasicOps.AlignedAlloc(perExpertBytes);
                            unsafe
                            {
                                Buffer.MemoryCopy(
                                    ((byte*)bulkPtr.ToPointer()) + e * perExpertBytes,
                                    expertPtr.ToPointer(),
                                    perExpertBytes, perExpertBytes);
                            }
                            _quantWeights[$"{baseName}.{e}.weight"] = new QuantizedWeight(expertPtr, perExpertBytes, (int)info.Type, ne0, ne1);
                        }

                        GgmlBasicOps.AlignedFree(bulkPtr);
                        countQuant += numExperts;
                        totalQuantBytes += byteCount;
                    }
                    else
                    {
                        IntPtr ptr = GgmlBasicOps.AlignedAlloc(byteCount);
                        _gguf.ReadTensorDataToNative(info, ptr, byteCount);
                        _quantWeights[info.Name] = new QuantizedWeight(ptr, byteCount, (int)info.Type, ne0, ne1);
                        countQuant++;
                        totalQuantBytes += byteCount;
                    }
                }
                else
                {
                    long numElements = info.NumElements;

                    long[] ggufShape = new long[info.Shape.Length];
                    for (int i = 0; i < info.Shape.Length; i++)
                        ggufShape[i] = (long)info.Shape[i];

                    long[] tsShape = new long[ggufShape.Length];
                    for (int i = 0; i < ggufShape.Length; i++)
                        tsShape[i] = ggufShape[ggufShape.Length - 1 - i];

                    var tensor = new Tensor(_allocator, DType.Float32, tsShape);
                    IntPtr destPtr = GetStoragePtr(tensor);

                    if (info.Type == GgmlTensorType.F32)
                    {
                        _gguf.ReadTensorDataToFloat32Native(info, destPtr, numElements);
                    }
                    else
                    {
                        IntPtr tempPtr = GgmlBasicOps.AlignedAlloc(byteCount);
                        try
                        {
                            _gguf.ReadTensorDataToNative(info, tempPtr, byteCount);
                            NativeDequant.DequantizeToFloat32Native((int)info.Type, tempPtr, destPtr, numElements);
                        }
                        finally { GgmlBasicOps.AlignedFree(tempPtr); }
                    }

                    _weights[info.Name] = tensor;

                    countF32++;
                    totalF32Bytes += numElements * 4;
                }
            }
            Console.WriteLine($" done ({countF32} F32 tensors, {countQuant} quantized tensors)");
            if (countQuant > 0)
                Console.WriteLine($"  Quantized: {totalQuantBytes / 1024 / 1024} MB, F32: {totalF32Bytes / 1024 / 1024} MB");
        }

        protected unsafe void FuseGateUpWeights()
        {
            int fused = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                string gateName = $"blk.{l}.ffn_gate.weight";
                string upName = $"blk.{l}.ffn_up.weight";
                string guName = $"blk.{l}.ffn_gate_up.weight";

                if (_quantWeights.TryGetValue(gateName, out var gw) &&
                    _quantWeights.TryGetValue(upName, out var uw) &&
                    gw.GgmlType == uw.GgmlType && gw.Ne0 == uw.Ne0)
                {
                    long totalBytes = gw.RawBytes + uw.RawBytes;
                    IntPtr fusedPtr = GgmlBasicOps.AlignedAlloc(totalBytes);
                    Buffer.MemoryCopy(gw.Data.ToPointer(), fusedPtr.ToPointer(), totalBytes, gw.RawBytes);
                    Buffer.MemoryCopy(uw.Data.ToPointer(), (fusedPtr + (int)gw.RawBytes).ToPointer(), totalBytes - gw.RawBytes, uw.RawBytes);
                    _quantWeights[guName] = new QuantizedWeight(fusedPtr, totalBytes, gw.GgmlType, gw.Ne0, gw.Ne1 + uw.Ne1);
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
                    _weights[guName] = fusedTensor;
                    _weights.Remove(gateName); gf.Dispose();
                    _weights.Remove(upName); uf.Dispose();
                    fused++;
                }
            }
            if (fused > 0)
                Console.WriteLine($"  Fused projections: {fused} Gate+Up");
        }

        protected Tensor CreateFloatTensor(float[] data, params long[] sizes)
        {
            var tensor = new Tensor(_allocator, DType.Float32, sizes);
            tensor.SetElementsAsFloat(data);
            return tensor;
        }

        protected Tensor CreateIntTensor(int[] data, params long[] sizes)
        {
            var tensor = new Tensor(_allocator, DType.Int32, sizes);
            tensor.SetElementsAsInt(data);
            return tensor;
        }

        protected float[] TensorToFloatArray(Tensor t)
        {
            if (t.IsContiguous())
                return t.GetElementsAsFloat((int)t.ElementCount());
            using var contiguous = Ops.NewContiguous(t);
            return contiguous.GetElementsAsFloat((int)contiguous.ElementCount());
        }

        protected unsafe Tensor Embedding(int[] tokens)
        {
            int dim = Config.HiddenSize;

            if (_quantWeights.TryGetValue("token_embd.weight", out var qw))
            {
                var result = new Tensor(_allocator, DType.Float32, tokens.Length, dim);
                using var idxTensor = CreateIntTensor(tokens, tokens.Length);
                GgmlBasicOps.GetRowsQuant(result, qw.Data, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes, idxTensor);
                return result;
            }

            var embWeight = _weights["token_embd.weight"];

            if (embWeight.IsContiguous())
            {
                var result = new Tensor(_allocator, DType.Float32, tokens.Length, dim);
                float* embPtr = GetFloatPtr(embWeight);
                float* dstPtr = GetFloatPtr(result);
                long rowBytes = dim * sizeof(float);
                for (int i = 0; i < tokens.Length; i++)
                    Buffer.MemoryCopy(embPtr + (long)tokens[i] * dim, dstPtr + (long)i * dim, rowBytes, rowBytes);
                return result;
            }

            using var indices = CreateIntTensor(tokens, tokens.Length);
            return Ops.IndexSelect(null, embWeight, indices);
        }

        protected Tensor LinearForward(Tensor input, string weightName)
        {
            long t0 = Stopwatch.GetTimestamp();

            Tensor result;
            if (_quantWeights.TryGetValue(weightName, out var qw))
            {
                int seqLen = (int)input.Sizes[0];
                int outDim = (int)qw.Ne1;
                result = new Tensor(_allocator, DType.Float32, seqLen, outDim);
                GgmlBasicOps.AddmmQuant(result, input, qw.Data, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes);
            }
            else if (_weights.TryGetValue(weightName, out var w))
            {
                int outDimF32 = (int)w.Sizes[0];
                int seqLenF32 = (int)input.Sizes[0];
                using var wT = w.Transpose();
                result = new Tensor(_allocator, DType.Float32, seqLenF32, outDimF32);
                Ops.Addmm(result, 0, result, 1.0f, input, wT);
            }
            else
            {
                return null;
            }

            _linearTicks += Stopwatch.GetTimestamp() - t0;
            return result;
        }

        #region SIMD Helpers

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector<float> LdVec(float* p) =>
            Unsafe.ReadUnaligned<Vector<float>>(ref *(byte*)p);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void StVec(float* p, Vector<float> v) =>
            Unsafe.WriteUnaligned(ref *(byte*)p, v);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected static unsafe float VecDot(float* a, float* b, int n)
        {
            int vLen = Vector<float>.Count;
            var acc0 = Vector<float>.Zero;
            var acc1 = Vector<float>.Zero;
            int i = 0;
            for (; i <= n - vLen * 2; i += vLen * 2)
            {
                acc0 += LdVec(a + i) * LdVec(b + i);
                acc1 += LdVec(a + i + vLen) * LdVec(b + i + vLen);
            }
            var acc = acc0 + acc1;
            for (; i <= n - vLen; i += vLen)
                acc += LdVec(a + i) * LdVec(b + i);
            float sum = Vector.Sum(acc);
            for (; i < n; i++)
                sum += a[i] * b[i];
            return sum;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected static unsafe float VecSumSq(float* a, int n)
        {
            int vLen = Vector<float>.Count;
            var acc0 = Vector<float>.Zero;
            var acc1 = Vector<float>.Zero;
            int i = 0;
            for (; i <= n - vLen * 2; i += vLen * 2)
            {
                var v0 = LdVec(a + i);
                var v1 = LdVec(a + i + vLen);
                acc0 += v0 * v0;
                acc1 += v1 * v1;
            }
            var acc = acc0 + acc1;
            for (; i <= n - vLen; i += vLen)
            {
                var v = LdVec(a + i);
                acc += v * v;
            }
            float sum = Vector.Sum(acc);
            for (; i < n; i++)
                sum += a[i] * a[i];
            return sum;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected static unsafe void VecScale(float* data, float scale, int n)
        {
            int vLen = Vector<float>.Count;
            var vs = new Vector<float>(scale);
            int i = 0;
            for (; i <= n - vLen; i += vLen)
                StVec(data + i, LdVec(data + i) * vs);
            for (; i < n; i++)
                data[i] *= scale;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected static unsafe void VecScaleAdd(float* dst, float* src, float w, int n)
        {
            int vLen = Vector<float>.Count;
            var vw = new Vector<float>(w);
            int i = 0;
            for (; i <= n - vLen; i += vLen)
                StVec(dst + i, LdVec(dst + i) + LdVec(src + i) * vw);
            for (; i < n; i++)
                dst[i] += w * src[i];
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected static unsafe void VecSubScale(float* dst, float* a, float* b, float scale, int n)
        {
            int vLen = Vector<float>.Count;
            var vs = new Vector<float>(scale);
            int i = 0;
            for (; i <= n - vLen; i += vLen)
                StVec(dst + i, (LdVec(a + i) - LdVec(b + i)) * vs);
            for (; i < n; i++)
                dst[i] = (a[i] - b[i]) * scale;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected static unsafe void VecZero(float* data, int n)
        {
            int vLen = Vector<float>.Count;
            int i = 0;
            for (; i <= n - vLen; i += vLen)
                StVec(data + i, Vector<float>.Zero);
            for (; i < n; i++)
                data[i] = 0;
        }

        #endregion

        protected Tensor RMSNormOp(Tensor input, string weightName)
        {
            long t0 = Stopwatch.GetTimestamp();
            var alpha = _weights[weightName];

            int rows = (int)input.Sizes[0];
            int dim = (int)(input.ElementCount() / rows);

            Tensor input2d = input.Sizes.Length != 2 ? input.View(rows, dim) : null;
            Tensor src = input2d ?? input;

            Tensor result = Ops.RMSNorm(null, src, alpha, null, Config.Eps);

            input2d?.Dispose();
            _normTicks += Stopwatch.GetTimestamp() - t0;
            return result;
        }

        protected Tensor FFN(Tensor input, string gateUpWeightName, string downWeightName, int seqLen)
        {
            int intermSize = Config.IntermediateSize;
            Tensor gateUp = LinearForward(input, gateUpWeightName);
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

            Ops.SiLUMul(gate, gate, up);
            up.Dispose();

            Tensor down = LinearForward(gate, downWeightName);
            gate.Dispose();
            return down;
        }


        protected void RMSNormInPlace(Tensor data, Tensor alpha, int numHeads, int headDim, float eps)
        {
            using var reshaped = data.View(numHeads, headDim);
            Ops.RMSNorm(reshaped, reshaped, alpha, null, eps);
        }

        protected Tensor ReshapeToHeads(Tensor data, int numHeads, int seqLen, int headDim)
        {
            if (seqLen == 1)
                return data.View(numHeads, 1, headDim);

            using var reshaped = data.View(seqLen, numHeads, headDim);
            using var transposed = reshaped.Transpose(0, 1);
            return Ops.NewContiguous(transposed);
        }

        protected Tensor ReshapeFromHeads(Tensor data, int numHeads, int seqLen, int headDim)
        {
            if (seqLen == 1)
                return data.View(1, numHeads * headDim);

            using var transposed = data.Transpose(0, 1);
            using var contiguous = Ops.NewContiguous(transposed);
            return contiguous.View(seqLen, numHeads * headDim);
        }

        protected void CopyToCache(Tensor cache, Tensor src, int startPos, int seqLen)
        {
            using var cacheSlice = cache.Narrow(1, startPos, seqLen);
            Ops.Copy(cacheSlice, src);
        }

        protected Tensor ExpandKVHeads(Tensor cache, int groupSize, int totalSeqLen)
        {
            using var active = cache.Narrow(1, 0, totalSeqLen);
            if (groupSize == 1)
                return Ops.NewContiguous(active);
            return Ops.RepeatInterleave(null, active, groupSize, 0);
        }

        protected unsafe void CopyToCacheDecode(Tensor kCache, Tensor kTensor,
            Tensor vCache, Tensor vTensor, int numKVHeads, int headDim, int startPos)
        {
            float* kSrc = GetFloatPtr(kTensor);
            float* vSrc = GetFloatPtr(vTensor);
            float* kCachePtr = GetFloatPtr(kCache);
            float* vCachePtr = GetFloatPtr(vCache);
            int maxSeqLen = (int)kCache.Sizes[1];
            int headBytes = headDim * sizeof(float);

            for (int h = 0; h < numKVHeads; h++)
            {
                int cacheOffset = h * maxSeqLen * headDim + startPos * headDim;
                int srcOffset = h * headDim;
                Buffer.MemoryCopy(kSrc + srcOffset, kCachePtr + cacheOffset, headBytes, headBytes);
                Buffer.MemoryCopy(vSrc + srcOffset, vCachePtr + cacheOffset, headBytes, headBytes);
            }
        }

        protected unsafe void AttentionDecodePureCS(Tensor q, Tensor kCache, Tensor vCache,
            Tensor result, int numHeads, int numKVHeads, int headDim, int totalSeqLen, float scale)
        {
            float* qPtr = GetFloatPtr(q);
            float* kPtr = GetFloatPtr(kCache);
            float* vPtr = GetFloatPtr(vCache);
            float* rPtr = GetFloatPtr(result);
            int maxSeqLen = (int)kCache.Sizes[1];
            int groupSize = numHeads / numKVHeads;

            float* scores = stackalloc float[totalSeqLen];

            for (int h = 0; h < numHeads; h++)
            {
                float* qHead = qPtr + h * headDim;
                int kvHead = h / groupSize;
                float* kHead = kPtr + kvHead * maxSeqLen * headDim;
                float* vHead = vPtr + kvHead * maxSeqLen * headDim;

                float maxScore = float.NegativeInfinity;
                for (int t = 0; t < totalSeqLen; t++)
                {
                    float s = VecDot(qHead, kHead + t * headDim, headDim) * scale;
                    scores[t] = s;
                    if (s > maxScore) maxScore = s;
                }

                float sumExp = 0;
                for (int t = 0; t < totalSeqLen; t++)
                {
                    float e = MathF.Exp(scores[t] - maxScore);
                    scores[t] = e;
                    sumExp += e;
                }
                float invSum = 1.0f / sumExp;
                for (int t = 0; t < totalSeqLen; t++)
                    scores[t] *= invSum;

                float* rHead = rPtr + h * headDim;
                VecZero(rHead, headDim);
                for (int t = 0; t < totalSeqLen; t++)
                    VecScaleAdd(rHead, vHead + t * headDim, scores[t], headDim);
            }
        }

        protected static unsafe float* GetFloatPtr(Tensor t)
        {
            if (t.Storage is GgmlStorage gs)
                return (float*)gs.PtrAtElement(t.StorageOffset);
            if (t.Storage is CpuStorage cs)
                return (float*)cs.PtrAtElement(t.StorageOffset);
            throw new NotSupportedException("Requires GgmlStorage or CpuStorage");
        }

        private static IntPtr GetStoragePtr(Tensor t)
        {
            if (t.Storage is GgmlStorage gs)
                return gs.PtrAtElement(t.StorageOffset);
            if (t.Storage is CpuStorage cs)
                return cs.PtrAtElement(t.StorageOffset);
            throw new NotSupportedException("Requires GgmlStorage or CpuStorage");
        }

        public abstract float[] Forward(int[] tokens);
        public abstract void ResetKVCache();

        /// <summary>
        /// Check if this model has vision encoder weights (v.* prefix tensors).
        /// </summary>
        public bool HasVisionEncoder()
        {
            foreach (var name in _weights.Keys)
                if (name.StartsWith("v.")) return true;
            foreach (var name in _quantWeights.Keys)
                if (name.StartsWith("v.")) return true;
            return false;
        }

        public void PrintTimingStats()
        {
            if (_forwardCount == 0) return;
            double totalMs = _forwardSw.Elapsed.TotalMilliseconds;
            double msPerTick = 1000.0 / Stopwatch.Frequency;
            double linearMs = _linearTicks * msPerTick;
            double attnMs = _attnTicks * msPerTick;
            double normMs = _normTicks * msPerTick;
            double embMs = _embTicks * msPerTick;
            double lmHeadMs = _lmHeadTicks * msPerTick;
            double logitsCopyMs = _logitsCopyTicks * msPerTick;
            double otherMs = totalMs - linearMs - attnMs - normMs;
            Console.WriteLine($"Timing ({_forwardCount} forward calls, {totalMs:F0} ms total, {totalMs / _forwardCount:F0} ms/token):");
            Console.WriteLine($"  Linear (matmul): {linearMs:F0} ms ({100 * linearMs / totalMs:F1}%)");
            Console.WriteLine($"  Attention:       {attnMs:F0} ms ({100 * attnMs / totalMs:F1}%)");
            Console.WriteLine($"  Norm:            {normMs:F0} ms ({100 * normMs / totalMs:F1}%)");
            Console.WriteLine($"  (LM head:        {lmHeadMs:F0} ms, included in Linear)");
            Console.WriteLine($"  (Embedding:      {embMs:F0} ms, in Other)");
            Console.WriteLine($"  (Logits copy:    {logitsCopyMs:F0} ms, in Other)");
            Console.WriteLine($"  Other:           {otherMs:F0} ms ({100 * otherMs / totalMs:F1}%)");
        }

        public int SampleGreedy(float[] logits)
        {
            int maxIdx = 0;
            float maxVal = logits[0];
            for (int i = 1; i < logits.Length; i++)
            {
                if (logits[i] > maxVal)
                {
                    maxVal = logits[i];
                    maxIdx = i;
                }
            }
            return maxIdx;
        }

        /// <summary>
        /// Sample a token using the given sampling configuration.
        /// Creates a one-shot sampler; for repeated calls in a generation loop,
        /// prefer creating a <see cref="TokenSampler"/> once and calling it directly.
        /// </summary>
        public int Sample(float[] logits, SamplingConfig config, IList<int> generatedTokenIds = null)
        {
            if (config == null || config.IsGreedy)
                return SampleGreedy(logits);
            var sampler = new TokenSampler(config);
            return sampler.Sample(logits, generatedTokenIds);
        }

        public virtual void Dispose()
        {
            foreach (var w in _weights.Values)
                w.Dispose();
            _weights.Clear();

            GgmlBasicOps.ClearHostBufferCache();

            foreach (var qw in _quantWeights.Values)
                qw.Dispose();
            _quantWeights.Clear();

            _gguf?.Dispose();
        }

        public static ModelBase Create(string ggufPath, BackendType backend)
        {
            using var probe = new GgufFile(ggufPath);
            string arch = probe.GetString("general.architecture") ?? "qwen3";

            return arch switch
            {
                "qwen3" => new Qwen3Model(ggufPath, backend),
                "qwen35" or "qwen35moe" or "qwen3next" => new Qwen35Model(ggufPath, backend),
                "gemma3" => new Gemma3Model(ggufPath, backend),
                "gemma4" => new Gemma4Model(ggufPath, backend),
                _ => throw new NotSupportedException($"Unsupported architecture: {arch}"),
            };
        }
    }
}
