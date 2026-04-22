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
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    public class QuantizedWeight : IDisposable
    {
        private IntPtr _data;
        private GCHandle _cacheKeyHandle;

        public IntPtr Data => _data;
        public IntPtr CacheKey { get; private set; }
        public int GgmlType { get; }
        public long Ne0 { get; }
        public long Ne1 { get; }
        public long RawBytes { get; }
        private bool _ownsBuffer;
        private bool _ownsCacheKeyHandle;
        private object _ownerToken;
        public bool HasHostData => _data != IntPtr.Zero;
        public bool HasExternalHostView => _data != IntPtr.Zero && !_ownsBuffer && _ownerToken != null;

        public QuantizedWeight(byte[] raw, int ggmlType, long ne0, long ne1)
        {
            GgmlType = ggmlType;
            Ne0 = ne0;
            Ne1 = ne1;
            RawBytes = raw.Length;
            _data = AllocateBuffer(raw.Length);
            CacheKey = _data;
            _ownsBuffer = true;
            Marshal.Copy(raw, 0, _data, raw.Length);
        }

        public QuantizedWeight(IntPtr data, long rawBytes, int ggmlType, long ne0, long ne1)
            : this(data, rawBytes, ggmlType, ne0, ne1, true, null)
        {
        }

        private QuantizedWeight(IntPtr data, long rawBytes, int ggmlType, long ne0, long ne1, bool ownsBuffer, object ownerToken)
        {
            _data = data;
            CacheKey = data;
            RawBytes = rawBytes;
            GgmlType = ggmlType;
            Ne0 = ne0;
            Ne1 = ne1;
            _ownsBuffer = ownsBuffer;
            _ownerToken = ownerToken;
        }

        public void Dispose()
        {
            ReleaseHostData();

            if (_ownsCacheKeyHandle)
            {
                _cacheKeyHandle.Free();
                _ownsCacheKeyHandle = false;
                CacheKey = IntPtr.Zero;
            }
        }

        public static QuantizedWeight CreateExternalView(IntPtr data, long rawBytes, int ggmlType, long ne0, long ne1, object ownerToken)
        {
            if (data == IntPtr.Zero)
                throw new ArgumentException("External quantized weight view requires a non-zero data pointer.", nameof(data));
            if (ownerToken == null)
                throw new ArgumentNullException(nameof(ownerToken));

            return new QuantizedWeight(data, rawBytes, ggmlType, ne0, ne1, false, ownerToken);
        }

        public static bool TryCreateConcatenatedView(out QuantizedWeight fused, params QuantizedWeight[] weights)
        {
            fused = null;
            if (weights == null || weights.Length < 2 || weights[0] == null)
                return false;

            QuantizedWeight first = weights[0];
            if (!first.HasHostData || first._ownsBuffer || first._ownerToken == null)
                return false;

            long totalBytes = 0;
            long totalNe1 = 0;
            long expectedAddress = first.Data.ToInt64();

            for (int i = 0; i < weights.Length; i++)
            {
                QuantizedWeight weight = weights[i];
                if (weight == null ||
                    weight._ownsBuffer ||
                    !ReferenceEquals(weight._ownerToken, first._ownerToken) ||
                    weight.GgmlType != first.GgmlType ||
                    weight.Ne0 != first.Ne0 ||
                    weight.Data.ToInt64() != expectedAddress)
                {
                    return false;
                }

                totalBytes += weight.RawBytes;
                totalNe1 += weight.Ne1;
                expectedAddress += weight.RawBytes;
            }

            fused = new QuantizedWeight(first.Data, totalBytes, first.GgmlType, first.Ne0, totalNe1, false, first._ownerToken);
            return true;
        }

        public static unsafe QuantizedWeight ConcatOrCreateCopy(params QuantizedWeight[] weights)
        {
            if (weights == null || weights.Length == 0 || weights[0] == null)
                throw new ArgumentException("At least one quantized weight is required.", nameof(weights));

            if (TryCreateConcatenatedView(out QuantizedWeight fused, weights))
                return fused;

            QuantizedWeight first = weights[0];
            long totalBytes = 0;
            long totalNe1 = 0;
            for (int i = 0; i < weights.Length; i++)
            {
                QuantizedWeight weight = weights[i] ?? throw new ArgumentException("Quantized weight list cannot contain null entries.", nameof(weights));
                if (!weight.HasHostData)
                    throw new InvalidOperationException("Cannot concatenate quantized weights after their host storage has been released.");
                totalBytes += weight.RawBytes;
                totalNe1 += weight.Ne1;
            }

            IntPtr fusedPtr = AllocateBuffer(totalBytes);
            byte* fusedDst = (byte*)fusedPtr.ToPointer();
            long offset = 0;
            for (int i = 0; i < weights.Length; i++)
            {
                QuantizedWeight weight = weights[i];
                Buffer.MemoryCopy(weight.Data.ToPointer(), fusedDst + offset, totalBytes - offset, weight.RawBytes);
                offset += weight.RawBytes;
            }

            return new QuantizedWeight(fusedPtr, totalBytes, first.GgmlType, first.Ne0, totalNe1);
        }

        public IntPtr EnsureDeviceCacheKey()
        {
            if (_ownsCacheKeyHandle)
                return CacheKey;

            _cacheKeyHandle = GCHandle.Alloc(this, GCHandleType.Normal);
            CacheKey = GCHandle.ToIntPtr(_cacheKeyHandle);
            _ownsCacheKeyHandle = true;
            return CacheKey;
        }

        public void ReleaseHostData()
        {
            if (_data == IntPtr.Zero)
                return;

            IntPtr currentData = _data;
            if (_ownsBuffer)
                FreeBuffer(currentData);

            if (CacheKey == currentData)
                CacheKey = IntPtr.Zero;

            _data = IntPtr.Zero;
            _ownsBuffer = false;
            _ownerToken = null;
        }

        public static unsafe IntPtr AllocateBuffer(long size)
        {
            void* ptr = NativeMemory.AlignedAlloc((nuint)size, 64);
            if (ptr == null)
                throw new OutOfMemoryException($"Unable to allocate {size} bytes for quantized weight storage.");
            return (IntPtr)ptr;
        }

        public static unsafe void FreeBuffer(IntPtr ptr)
        {
            if (ptr != IntPtr.Zero)
                NativeMemory.AlignedFree(ptr.ToPointer());
        }
    }

    public abstract class ModelBase : IModelArchitecture
    {
        public ModelConfig Config { get; protected set; }
        public ITokenizer Tokenizer { get; protected set; }
        public IMultimodalInjector MultimodalInjector { get; }
        public IBackendExecutionPlan ExecutionPlan { get; }

        protected readonly GgufFile _gguf;
        private readonly GgmlContext _ggmlContext;
        protected readonly IAllocator _allocator;
        protected readonly BackendType _backend;

        protected readonly Dictionary<string, Tensor> _weights = new();
        protected readonly Dictionary<string, QuantizedWeight> _quantWeights = new();
        private bool _quantBackendReady;
        private bool _cudaQuantWeightsPrepared;

        protected int _cacheSeqLen;
        protected int _maxContextLength;
        protected float[] _logitsBuffer;

        public int MaxContextLength => _maxContextLength;
        public int CacheSeqLen => _cacheSeqLen;

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
            ExecutionPlan = new BackendExecutionPlan(backend);
            MultimodalInjector = new ModelMultimodalInjector(this);
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

        protected bool IsGgmlBackend => ExecutionPlan.UsesGgmlBackend;

        protected void EnsureQuantBackendAvailable()
        {
            if (_quantBackendReady || !IsGgmlBackend)
                return;

            GgmlBackendType backendType = _backend switch
            {
                BackendType.GgmlCpu => GgmlBackendType.Cpu,
                BackendType.GgmlMetal => GgmlBackendType.Metal,
                BackendType.GgmlCuda => GgmlBackendType.Cuda,
                _ => throw new InvalidOperationException($"No GGML backend is associated with {_backend}."),
            };
            GgmlBasicOps.EnsureBackendAvailable(backendType);

            _quantBackendReady = true;
        }

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

        protected int ResolveConfiguredContextLength(int fallback = 4096)
        {
            int? explicitOverride = null;
            string source;
            string ctxEnv = Environment.GetEnvironmentVariable("MAX_CONTEXT");
            if (!string.IsNullOrWhiteSpace(ctxEnv) && int.TryParse(ctxEnv, out int envCtx) && envCtx > 0)
                explicitOverride = envCtx;

            int resolved = ResolveConfiguredContextLength(
                Config?.Architecture ?? _gguf.GetString("general.architecture") ?? string.Empty,
                _gguf.Metadata,
                fallback,
                explicitOverride,
                out source);

            if (explicitOverride.HasValue)
                Console.WriteLine($"Context length: using MAX_CONTEXT={resolved}.");
            else if (source == "fallback")
                Console.WriteLine($"Context length: metadata missing, falling back to {resolved} tokens.");
            else
                Console.WriteLine($"Context length: using GGUF metadata {source}={resolved}.");

            return resolved;
        }

        protected int ResolveInitialCacheAllocationLength(int requestedContextLength, int gpuDefault = 8192)
        {
            // GPU backends (CUDA, Metal) can be sensitive to allocating a multi-gigabyte KV
            // cache up-front when the model advertises a 256K+ context window. Cap the initial
            // allocation and let the cache grow on demand when actual prompts approach the
            // limit. CPU backends have no such constraint and use the full requested length.
            bool isGpuBackend = _backend == BackendType.GgmlCuda || _backend == BackendType.GgmlMetal;
            if (isGpuBackend &&
                string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("MAX_CONTEXT")))
            {
                return Math.Min(requestedContextLength, gpuDefault);
            }

            return requestedContextLength;
        }

        protected bool ShouldZeroFillCacheTensors => _backend != BackendType.GgmlCuda;

        protected void InitializeCacheTensor(Tensor tensor)
        {
            if (tensor != null && ShouldZeroFillCacheTensors)
                Ops.Fill(tensor, 0f);
        }

        protected void ResetCacheTensor(Tensor tensor)
        {
            if (tensor == null)
                return;

            if (ShouldZeroFillCacheTensors)
                Ops.Fill(tensor, 0f);

            InvalidateTensorDeviceCache(tensor);
        }

        internal static int ResolveConfiguredContextLength(
            string architecture,
            IReadOnlyDictionary<string, object> metadata,
            int fallback,
            int? explicitOverride,
            out string source)
        {
            if (explicitOverride.HasValue && explicitOverride.Value > 0)
            {
                source = "MAX_CONTEXT";
                return explicitOverride.Value;
            }

            foreach (string key in GetContextLengthMetadataKeys(architecture))
            {
                if (TryGetPositiveInt(metadata, key, out int contextLength))
                {
                    source = key;
                    return contextLength;
                }
            }

            source = "fallback";
            return fallback;
        }

        private static IEnumerable<string> GetContextLengthMetadataKeys(string architecture)
        {
            if (!string.IsNullOrWhiteSpace(architecture))
            {
                yield return $"{architecture}.context_length";
                yield return $"{architecture}.attention.context_length";
                yield return $"{architecture}.max_position_embeddings";
                yield return $"{architecture}.max_sequence_length";
                yield return $"{architecture}.sequence_length";
                yield return $"{architecture}.seq_length";
                yield return $"{architecture}.n_ctx";
                yield return $"{architecture}.rope.scaling.original_context_length";
            }

            yield return "context_length";
            yield return "max_position_embeddings";
            yield return "max_sequence_length";
            yield return "sequence_length";
            yield return "seq_length";
            yield return "n_ctx";
        }

        private static bool TryGetPositiveInt(IReadOnlyDictionary<string, object> metadata, string key, out int value)
        {
            value = 0;
            if (metadata == null || string.IsNullOrWhiteSpace(key) || !metadata.TryGetValue(key, out var raw) || raw == null)
                return false;

            try
            {
                switch (raw)
                {
                    case int i when i > 0:
                        value = i;
                        return true;
                    case uint ui when ui > 0:
                        value = (int)ui;
                        return true;
                    case long l when l > 0 && l <= int.MaxValue:
                        value = (int)l;
                        return true;
                    case ulong ul when ul > 0 && ul <= int.MaxValue:
                        value = (int)ul;
                        return true;
                    case int[] ia when ia.Length > 0 && ia[0] > 0:
                        value = ia[0];
                        return true;
                    case uint[] ua when ua.Length > 0 && ua[0] > 0 && ua[0] <= int.MaxValue:
                        value = (int)ua[0];
                        return true;
                    case long[] la when la.Length > 0 && la[0] > 0 && la[0] <= int.MaxValue:
                        value = (int)la[0];
                        return true;
                    case ulong[] ula when ula.Length > 0 && ula[0] > 0 && ula[0] <= int.MaxValue:
                        value = (int)ula[0];
                        return true;
                    default:
                        value = Convert.ToInt32(raw);
                        return value > 0;
                }
            }
            catch
            {
                value = 0;
                return false;
            }
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
                string preType = _gguf.GetString("tokenizer.ggml.pre", null);
                Tokenizer = new BpeTokenizer(vocabTokens, tokenTypes, merges,
                    bosId, eosIds.ToArray(), addBos, addEos, preType);
            }
        }

        protected virtual bool IsQuantizedLinearWeight(GgufTensorInfo info)
        {
            return ExecutionPlan.ShouldStoreWeightQuantized(info);
        }

        internal static bool ShouldStoreWeightQuantized(BackendType backend, GgufTensorInfo info)
        {
            if (info.Type == GgmlTensorType.F32)
                return false;

            if (backend == BackendType.Cpu && !ManagedQuantizedOps.SupportsCpuQuantizedStorage(info.Type))
                return false;

            if (info.Shape.Length == 2)
                return true;

            return info.Shape.Length == 3 && info.Name.Contains("_exps.");
        }

        /// <summary>
        /// Whether quantized weights for this backend can be backed directly by the GGUF file
        /// via memory mapping instead of being copied into freshly-allocated host buffers.
        ///
        /// On Apple Silicon (Metal, integrated GPU, unified memory) and on the GGML CPU backend
        /// the on-disk layout matches what the kernels consume verbatim, so we can skip the
        /// per-tensor copy and let the OS page in / out of the file as needed. This roughly
        /// halves the resident set for large quantized models (e.g. ~10 GB GGUF files no longer
        /// need a second 10 GB native heap copy).
        ///
        /// On discrete CUDA GPUs the kernels still want device-local memory, but the original
        /// host pointer is needed once at preload time so the device copy is performed via
        /// <see cref="PrepareCudaQuantizedWeightsForInference"/> from the file-backed view.
        /// </summary>
        protected bool CanUseFileMappedQuantizedWeights
            => _backend == BackendType.GgmlCuda
            || _backend == BackendType.GgmlMetal
            || _backend == BackendType.GgmlCpu;

        protected void LoadWeights()
        {
            Console.Write("Loading model weights...");
            int countF32 = 0;
            int countQuant = 0;
            long totalQuantBytes = 0;
            long totalF32Bytes = 0;
            long mappedQuantBytes = 0;
            bool tryMmap = CanUseFileMappedQuantizedWeights;
            foreach (var kv in _gguf.Tensors)
            {
                var info = kv.Value;
                long byteCount = _gguf.GetTensorByteCount(info);

                if (IsQuantizedLinearWeight(info))
                {
                    if (IsGgmlBackend)
                        EnsureQuantBackendAvailable();

                    long ne0 = (long)info.Shape[0];
                    long ne1 = (long)info.Shape[1];

                    if (info.Shape.Length == 3 && info.Name.Contains("_exps."))
                    {
                        // 3D MoE expert tensor: split into per-expert 2D quantized weights
                        int numExperts = (int)info.Shape[2];
                        long perExpertBytes = byteCount / numExperts;
                        string baseName = info.Name;
                        if (baseName.EndsWith(".weight"))
                            baseName = baseName.Substring(0, baseName.Length - 7);

                        if (tryMmap && _gguf.TryGetTensorDataPointer(info, out IntPtr mappedTensorPtr))
                        {
                            for (int e = 0; e < numExperts; e++)
                            {
                                IntPtr expertPtr = new IntPtr(mappedTensorPtr.ToInt64() + e * perExpertBytes);
                                _quantWeights[$"{baseName}.{e}.weight"] = QuantizedWeight.CreateExternalView(
                                    expertPtr, perExpertBytes, (int)info.Type, ne0, ne1, _gguf);
                            }
                            mappedQuantBytes += byteCount;
                        }
                        else
                        {
                            IntPtr bulkPtr = QuantizedWeight.AllocateBuffer(byteCount);
                            _gguf.ReadTensorDataToNative(info, bulkPtr, byteCount);

                            for (int e = 0; e < numExperts; e++)
                            {
                                IntPtr expertPtr = QuantizedWeight.AllocateBuffer(perExpertBytes);
                                unsafe
                                {
                                    Buffer.MemoryCopy(
                                        ((byte*)bulkPtr.ToPointer()) + e * perExpertBytes,
                                        expertPtr.ToPointer(),
                                        perExpertBytes, perExpertBytes);
                                }
                                _quantWeights[$"{baseName}.{e}.weight"] = new QuantizedWeight(expertPtr, perExpertBytes, (int)info.Type, ne0, ne1);
                            }

                            QuantizedWeight.FreeBuffer(bulkPtr);
                        }
                        countQuant += numExperts;
                        totalQuantBytes += byteCount;
                    }
                    else
                    {
                        if (tryMmap && _gguf.TryGetTensorDataPointer(info, out IntPtr mappedTensorPtr))
                        {
                            _quantWeights[info.Name] = QuantizedWeight.CreateExternalView(
                                mappedTensorPtr, byteCount, (int)info.Type, ne0, ne1, _gguf);
                            mappedQuantBytes += byteCount;
                        }
                        else
                        {
                            IntPtr ptr = QuantizedWeight.AllocateBuffer(byteCount);
                            _gguf.ReadTensorDataToNative(info, ptr, byteCount);
                            _quantWeights[info.Name] = new QuantizedWeight(ptr, byteCount, (int)info.Type, ne0, ne1);
                        }
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
                        IntPtr tempPtr = QuantizedWeight.AllocateBuffer(byteCount);
                        try
                        {
                            _gguf.ReadTensorDataToNative(info, tempPtr, byteCount);
                            NativeDequant.DequantizeToFloat32Native((int)info.Type, tempPtr, destPtr, numElements);
                        }
                        finally { QuantizedWeight.FreeBuffer(tempPtr); }
                    }

                    _weights[info.Name] = tensor;

                    countF32++;
                    totalF32Bytes += numElements * 4;
                }
            }
            Console.WriteLine($" done ({countF32} F32 tensors, {countQuant} quantized tensors)");
            if (countQuant > 0)
            {
                if (mappedQuantBytes > 0)
                    Console.WriteLine($"  Quantized: {totalQuantBytes / 1024 / 1024} MB ({mappedQuantBytes / 1024 / 1024} MB file-backed), F32: {totalF32Bytes / 1024 / 1024} MB");
                else
                    Console.WriteLine($"  Quantized: {totalQuantBytes / 1024 / 1024} MB, F32: {totalF32Bytes / 1024 / 1024} MB");
            }
        }

        protected void PrepareCudaQuantizedWeightsForInference()
        {
            if (_backend != BackendType.GgmlCuda || _cudaQuantWeightsPrepared || _quantWeights.Count == 0)
                return;

            EnsureQuantBackendAvailable();

            long preloadedBytes = 0;
            int preloadedCount = 0;
            int mappedHostViews = 0;

            foreach (QuantizedWeight qw in _quantWeights.Values)
            {
                if (qw.HasExternalHostView)
                    mappedHostViews++;
            }

            foreach (var kv in _quantWeights)
            {
                string weightName = kv.Key;
                QuantizedWeight qw = kv.Value;

                if (!qw.HasHostData)
                    continue;

                IntPtr cacheKey = qw.EnsureDeviceCacheKey();
                GgmlBasicOps.PreloadQuantizedWeight(cacheKey, qw.Data, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes);
                preloadedBytes += qw.RawBytes;
                preloadedCount++;

                if (!ShouldRetainCudaHostQuantWeight(weightName))
                {
                    bool wasMappedView = qw.HasExternalHostView;
                    qw.ReleaseHostData();

                    if (wasMappedView)
                        mappedHostViews--;
                }
            }

            if (mappedHostViews == 0)
                _gguf?.Dispose();
            _cudaQuantWeightsPrepared = true;

            if (preloadedCount > 0)
                Console.WriteLine($"  CUDA resident quantized weights: {preloadedBytes / 1024 / 1024} MB across {preloadedCount} tensors");
        }

        private static bool ShouldRetainCudaHostQuantWeight(string weightName)
        {
            return string.Equals(weightName, "token_embd.weight", StringComparison.Ordinal) ||
                string.Equals(weightName, "per_layer_token_embd.weight", StringComparison.Ordinal);
        }

        protected bool CanUseGgmlQuantizedGetRows(int ggmlType)
        {
            if (!IsGgmlBackend)
                return false;

            if (_backend != BackendType.GgmlCuda)
                return true;

            return ((GgmlTensorType)ggmlType) switch
            {
                GgmlTensorType.Q4_0 => true,
                GgmlTensorType.Q4_1 => true,
                GgmlTensorType.Q5_0 => true,
                GgmlTensorType.Q5_1 => true,
                GgmlTensorType.Q8_0 => true,
                GgmlTensorType.Q6_K => true,
                _ => false,
            };
        }

        protected unsafe void PopulateQuantizedRows(Tensor result, QuantizedWeight weight, int[] rowIndices)
        {
            if (result == null)
                throw new ArgumentNullException(nameof(result));
            if (weight == null)
                throw new ArgumentNullException(nameof(weight));
            if (rowIndices == null)
                throw new ArgumentNullException(nameof(rowIndices));
            if (!weight.HasHostData)
                throw new InvalidOperationException("Quantized row lookup requires host-side weight data.");

            int dim = (int)weight.Ne0;
            if (result.DimensionCount != 2 || result.ElementType != DType.Float32 ||
                result.Sizes[0] != rowIndices.Length || result.Sizes[1] != dim)
            {
                throw new ArgumentException("Result tensor shape must be [rowIndices.Length, weight.Ne0].", nameof(result));
            }

            long rowBytes = NativeDequant.RowSize(weight.GgmlType, weight.Ne0);
            byte* basePtr = (byte*)weight.Data.ToPointer();
            float* dst = GetFloatPtr(result);
            for (int i = 0; i < rowIndices.Length; i++)
            {
                byte* rowPtr = basePtr + (long)rowIndices[i] * rowBytes;
                NativeDequant.DequantizeToFloat32Native(
                    weight.GgmlType,
                    (IntPtr)rowPtr,
                    (IntPtr)(dst + (long)i * dim),
                    dim);
            }

            InvalidateTensorDeviceCache(result);
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
                    _quantWeights[guName] = QuantizedWeight.ConcatOrCreateCopy(gw, uw);
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
                if (IsGgmlBackend)
                {
                    bool canUseGgmlLookup = CanUseGgmlQuantizedGetRows(qw.GgmlType);

                    // A direct host dequant is faster for single-token decode, and it is
                    // also the compatibility path for CUDA quant types whose get_rows
                    // kernel is not implemented upstream.
                    if ((tokens.Length == 1 || !canUseGgmlLookup) && qw.HasHostData)
                    {
                        var result = new Tensor(_allocator, DType.Float32, tokens.Length, dim);
                        PopulateQuantizedRows(result, qw, tokens);
                        return result;
                    }

                    if (!canUseGgmlLookup)
                        throw new InvalidOperationException($"CUDA get_rows does not support GGML tensor type {(GgmlTensorType)qw.GgmlType}, and no host copy is available for CPU fallback.");

                    var resultMulti = new Tensor(_allocator, DType.Float32, tokens.Length, dim);
                    using var idxTensor = CreateIntTensor(tokens, tokens.Length);
                    GgmlBasicOps.GetRowsQuant(resultMulti, qw.CacheKey, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes, idxTensor);
                    return resultMulti;
                }

                return EmbeddingManagedQuantized(tokens, qw);
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
                if (IsGgmlBackend)
                    GgmlBasicOps.AddmmQuant(result, input, qw.CacheKey, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes);
                else
                    AddmmQuantManaged(result, input, qw);
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

        private unsafe Tensor EmbeddingManagedQuantized(int[] tokens, QuantizedWeight weight)
        {
            int dim = (int)weight.Ne0;
            long rowBytes = NativeDequant.RowSize(weight.GgmlType, weight.Ne0);
            var result = new Tensor(_allocator, DType.Float32, tokens.Length, dim);
            float* dst = GetFloatPtr(result);
            byte* basePtr = (byte*)weight.Data.ToPointer();

            for (int i = 0; i < tokens.Length; i++)
            {
                byte* rowPtr = basePtr + (long)tokens[i] * rowBytes;
                ManagedQuantizedOps.DequantizeRowToFloat32(weight.GgmlType, (IntPtr)rowPtr, dst + (long)i * dim, dim);
            }

            return result;
        }

        protected unsafe void AddmmQuantManaged(Tensor result, Tensor input, QuantizedWeight weight)
        {
            if (!input.IsContiguous() || !result.IsContiguous())
                throw new NotSupportedException("Managed quantized matmul requires contiguous input and output tensors.");

            int seqLen = (int)input.Sizes[0];
            int inDim = (int)weight.Ne0;
            int outDim = (int)weight.Ne1;
            if ((int)input.Sizes[1] != inDim)
                throw new ArgumentException($"Input dim {input.Sizes[1]} does not match quantized weight width {inDim}.");

            long rowBytes = NativeDequant.RowSize(weight.GgmlType, weight.Ne0);
            float* inputPtr = GetFloatPtr(input);
            float* resultPtr = GetFloatPtr(result);
            byte* weightBase = (byte*)weight.Data.ToPointer();

            void RunRange(int start, int end, float* sums)
            {
                for (int col = start; col < end; col++)
                {
                    byte* rowPtr = weightBase + (long)col * rowBytes;
                    ManagedQuantizedOps.DotRowBatchToFloat32(
                        weight.GgmlType,
                        (IntPtr)rowPtr,
                        inputPtr,
                        inDim,
                        seqLen,
                        inDim,
                        sums);

                    for (int row = 0; row < seqLen; row++)
                    {
                        resultPtr[(long)row * outDim + col] = sums[row];
                    }
                }
            }

            bool useParallel = outDim >= 128 && seqLen * outDim >= 512 && Environment.ProcessorCount > 1;
            if (!useParallel)
            {
                float[] sumsArr = ArrayPool<float>.Shared.Rent(seqLen);
                try
                {
                    fixed (float* sums = sumsArr)
                    {
                        RunRange(0, outDim, sums);
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(sumsArr);
                }

                return;
            }

            Parallel.For(0, outDim,
                () => ArrayPool<float>.Shared.Rent(seqLen),
                (col, _, sumsArr) =>
                {
                    fixed (float* sums = sumsArr)
                    {
                        RunRange(col, col + 1, sums);
                    }
                    return sumsArr;
                },
                sumsArr => ArrayPool<float>.Shared.Return(sumsArr));
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

        /// <summary>
        /// Batched dot product: simultaneously compute four independent dot products
        /// against the same source vector <paramref name="b"/>. Lets the compiler keep
        /// the vector loads of b in registers and reuse them across the four accumulators,
        /// effectively cutting the load bandwidth on b by 4x compared to four sequential
        /// VecDot calls. Used in GQA decode attention where four query heads share a K/V head.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected static unsafe void VecDot4(float* a0, float* a1, float* a2, float* a3,
            float* b, int n,
            out float r0, out float r1, out float r2, out float r3)
        {
            int vLen = Vector<float>.Count;
            var acc0 = Vector<float>.Zero;
            var acc1 = Vector<float>.Zero;
            var acc2 = Vector<float>.Zero;
            var acc3 = Vector<float>.Zero;
            int i = 0;
            for (; i <= n - vLen; i += vLen)
            {
                var vb = LdVec(b + i);
                acc0 += LdVec(a0 + i) * vb;
                acc1 += LdVec(a1 + i) * vb;
                acc2 += LdVec(a2 + i) * vb;
                acc3 += LdVec(a3 + i) * vb;
            }
            float s0 = Vector.Sum(acc0);
            float s1 = Vector.Sum(acc1);
            float s2 = Vector.Sum(acc2);
            float s3 = Vector.Sum(acc3);
            for (; i < n; i++)
            {
                float bi = b[i];
                s0 += a0[i] * bi;
                s1 += a1[i] * bi;
                s2 += a2[i] * bi;
                s3 += a3[i] * bi;
            }
            r0 = s0; r1 = s1; r2 = s2; r3 = s3;
        }

        /// <summary>
        /// Batched scale-add: simultaneously update four destination vectors with the
        /// same source <paramref name="src"/> scaled by four independent weights. The
        /// hot loop loads each src element exactly once into a register and broadcasts
        /// it to four FMA-style updates, which is the V-aggregation analog of VecDot4.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected static unsafe void VecScaleAdd4(float* d0, float* d1, float* d2, float* d3,
            float* src, float w0, float w1, float w2, float w3, int n)
        {
            int vLen = Vector<float>.Count;
            var vw0 = new Vector<float>(w0);
            var vw1 = new Vector<float>(w1);
            var vw2 = new Vector<float>(w2);
            var vw3 = new Vector<float>(w3);
            int i = 0;
            for (; i <= n - vLen; i += vLen)
            {
                var vs = LdVec(src + i);
                StVec(d0 + i, LdVec(d0 + i) + vs * vw0);
                StVec(d1 + i, LdVec(d1 + i) + vs * vw1);
                StVec(d2 + i, LdVec(d2 + i) + vs * vw2);
                StVec(d3 + i, LdVec(d3 + i) + vs * vw3);
            }
            for (; i < n; i++)
            {
                float s = src[i];
                d0[i] += w0 * s;
                d1[i] += w1 * s;
                d2[i] += w2 * s;
                d3[i] += w3 * s;
            }
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

        /// <summary>
        /// CPU SIMD in-place RMSNorm for the single-row decode hot path. Avoids the GPU
        /// dispatch overhead of <see cref="RMSNormInPlace"/> for a tiny tensor (e.g. QK
        /// norm: 16x256 floats). Each "row" (head) is normalized independently using its
        /// own scale factor and the shared <paramref name="alpha"/> per-element weight.
        /// Safe only when <paramref name="data"/> and <paramref name="alpha"/> are
        /// host-accessible (CpuStorage or GGML host-mapped) which is true on Metal/CUDA
        /// for these intermediate decode tensors.
        /// </summary>
        protected unsafe void RMSNormInPlaceCpu(Tensor data, Tensor alpha, int numHeads, int headDim, float eps)
        {
            float* dataPtr = GetFloatPtr(data);
            float* alphaPtr = GetFloatPtr(alpha);
            float invHeadDim = 1.0f / headDim;
            int vLen = Vector<float>.Count;

            for (int h = 0; h < numHeads; h++)
            {
                float* row = dataPtr + (long)h * headDim;
                float ssq = VecSumSq(row, headDim);
                float invRms = 1.0f / MathF.Sqrt(ssq * invHeadDim + eps);
                var vScale = new Vector<float>(invRms);

                int i = 0;
                for (; i <= headDim - vLen; i += vLen)
                {
                    var x = LdVec(row + i);
                    var a = LdVec(alphaPtr + i);
                    StVec(row + i, x * vScale * a);
                }
                for (; i < headDim; i++)
                    row[i] = row[i] * invRms * alphaPtr[i];
            }

            InvalidateTensorDeviceCache(data);
        }

        /// <summary>
        /// SiLU(gate) * up in place: <c>gate[i] = gate[i] / (1 + exp(-gate[i])) * up[i]</c>.
        /// For the single-row FFN decode path the GPU dispatch overhead is comparable to
        /// the actual compute, so doing it on CPU and saving one Metal command buffer
        /// per FFN layer per token is a net win on Apple unified memory. The inner loop
        /// is dominated by MathF.Exp which has no vectorized intrinsic, so we keep it
        /// scalar but allow the JIT to unroll it.
        /// </summary>
        protected unsafe void SiLUMulInPlaceCpu(Tensor gate, Tensor up)
        {
            float* gPtr = GetFloatPtr(gate);
            float* uPtr = GetFloatPtr(up);
            int n = (int)gate.ElementCount();

            for (int i = 0; i < n; i++)
            {
                float g = gPtr[i];
                float silu = g / (1.0f + MathF.Exp(-g));
                gPtr[i] = silu * uPtr[i];
            }

            InvalidateTensorDeviceCache(gate);
        }

        /// <summary>
        /// CPU SIMD RMSNorm that writes to a separate output tensor (does not modify the
        /// input). Used for the MoE post-attention norm in the decode hot path where the
        /// residual must be preserved for the later residual add. Treats <paramref name="input"/>
        /// as a single row of length <paramref name="dim"/> and applies the per-element
        /// alpha weight to the normalized output.
        /// </summary>
        protected unsafe void RMSNormToBufferCpu(Tensor output, Tensor input, Tensor alpha, int dim, float eps)
        {
            float* outPtr = GetFloatPtr(output);
            float* inPtr = GetFloatPtr(input);
            float* alphaPtr = GetFloatPtr(alpha);
            int vLen = Vector<float>.Count;

            float ssq = VecSumSq(inPtr, dim);
            float invRms = 1.0f / MathF.Sqrt(ssq / dim + eps);
            var vScale = new Vector<float>(invRms);

            int i = 0;
            for (; i <= dim - vLen; i += vLen)
            {
                var x = LdVec(inPtr + i);
                var a = LdVec(alphaPtr + i);
                StVec(outPtr + i, x * vScale * a);
            }
            for (; i < dim; i++)
                outPtr[i] = inPtr[i] * invRms * alphaPtr[i];

            InvalidateTensorDeviceCache(output);
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
            InvalidateTensorDeviceCache(cache);
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

            InvalidateTensorDeviceCache(kCache);
            InvalidateTensorDeviceCache(vCache);
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

            // GQA-aware decode attention. For each KV head we compute attention for the
            // groupSize query heads that share it, reading K/V from the cache exactly once
            // per KV head per token instead of groupSize times. On models with GQA this
            // cuts the per-token K/V cache traffic by groupSize (4x for Qwen3.5), which
            // is the dominant cost for long-context decode.
            //
            // To keep multi-core utilization high we split each KV head into kSplit chunks
            // along the sequence dimension and merge partial softmax results using the
            // standard online (log-sum-exp) update. Total parallel tasks = numKVHeads * kSplit.

            // Aim for enough parallel tasks to keep cores busy, but keep per-task work
            // big enough to amortize Parallel.For dispatch overhead. Each task handles one
            // (KV head, K-chunk) pair. Empirically, ~512 K-positions per task is the sweet
            // spot on Apple M-series: smaller chunks lose to scheduler overhead, larger
            // chunks under-utilize cores at long contexts.
            int procCount = Environment.ProcessorCount;
            int kSplit = 1;
            if (numKVHeads < procCount && totalSeqLen >= 1024)
            {
                int target = (procCount + numKVHeads - 1) / numKVHeads;
                int maxSplit = Math.Max(1, totalSeqLen / 512);
                kSplit = Math.Min(target, maxSplit);
            }
            int totalTasks = numKVHeads * kSplit;
            bool useParallel = totalTasks > 1 && (long)numHeads * totalSeqLen >= 4096;

            if (useParallel)
            {
                long qPtrL = (long)qPtr;
                long kPtrL = (long)kPtr;
                long vPtrL = (long)vPtr;
                long rPtrL = (long)rPtr;
                int totalSeqLenLocal = totalSeqLen;
                int headDimLocal = headDim;
                int maxSeqLenLocal = maxSeqLen;
                int groupSizeLocal = groupSize;
                int numKVHeadsLocal = numKVHeads;
                int kSplitLocal = kSplit;
                float scaleLocal = scale;

                if (kSplitLocal == 1)
                {
                    Parallel.For(0, numKVHeadsLocal, kvHead =>
                    {
                        float* qP = (float*)qPtrL;
                        float* kP = (float*)kPtrL;
                        float* vP = (float*)vPtrL;
                        float* rP = (float*)rPtrL;
                        float* scoresBuf = stackalloc float[groupSizeLocal * totalSeqLenLocal];
                        AttentionDecodeKVHeadGrouped(kvHead, qP, kP, vP, rP, scoresBuf,
                            headDimLocal, maxSeqLenLocal, groupSizeLocal,
                            totalSeqLenLocal, scaleLocal);
                    });
                }
                else
                {
                    // Two-pass: partial chunks then merge per KV head. First we compute
                    // running max and (un-normalized) weighted sum for each chunk, then we
                    // merge the chunk results into the final per-query-head output.
                    int chunkSize = (totalSeqLenLocal + kSplitLocal - 1) / kSplitLocal;

                    // Per-chunk partial state: max, sumExp, weighted-V (groupSize * headDim) for each (kvHead, chunk).
                    int partialFloatsPerChunk = groupSizeLocal * (2 + headDimLocal);
                    int partialFloatsTotal = numKVHeadsLocal * kSplitLocal * partialFloatsPerChunk;

                    var partialBuf = ArrayPool<float>.Shared.Rent(partialFloatsTotal);
                    try
                    {
                        fixed (float* partialPtr = partialBuf)
                        {
                            long partialPtrL = (long)partialPtr;

                            Parallel.For(0, numKVHeadsLocal * kSplitLocal, taskIdx =>
                            {
                                int kvHead = taskIdx / kSplitLocal;
                                int chunkIdx = taskIdx % kSplitLocal;
                                int kStart = chunkIdx * chunkSize;
                                int kEnd = Math.Min(kStart + chunkSize, totalSeqLenLocal);
                                int kLen = kEnd - kStart;
                                if (kLen <= 0) return;

                                float* qP = (float*)qPtrL;
                                float* kP = (float*)kPtrL;
                                float* vP = (float*)vPtrL;
                                float* part = (float*)partialPtrL +
                                    (long)taskIdx * partialFloatsPerChunk;

                                float* scoresLocal = stackalloc float[groupSizeLocal * kLen];
                                AttentionDecodeChunkPartial(kvHead, kStart, kLen, qP, kP, vP,
                                    part, scoresLocal,
                                    headDimLocal, maxSeqLenLocal, groupSizeLocal, scaleLocal);
                            });

                            Parallel.For(0, numKVHeadsLocal, kvHead =>
                            {
                                float* rP = (float*)rPtrL;
                                float* part = (float*)partialPtrL +
                                    (long)kvHead * kSplitLocal * partialFloatsPerChunk;

                                MergeChunkResults(kvHead, rP, part,
                                    headDimLocal, groupSizeLocal, kSplitLocal);
                            });
                        }
                    }
                    finally
                    {
                        ArrayPool<float>.Shared.Return(partialBuf);
                    }
                }
            }
            else
            {
                float* scores = stackalloc float[groupSize * totalSeqLen];
                for (int kvHead = 0; kvHead < numKVHeads; kvHead++)
                {
                    AttentionDecodeKVHeadGrouped(kvHead, qPtr, kPtr, vPtr, rPtr, scores,
                        headDim, maxSeqLen, groupSize, totalSeqLen, scale);
                }
            }
        }

        /// <summary>
        /// Compute attention for one KV head against all <paramref name="groupSize"/> query heads
        /// sharing it. Reads K and V from the cache exactly once per timestep, regardless of
        /// groupSize. On Qwen3.5-style GQA models this cuts KV-cache memory bandwidth by 4x.
        /// </summary>
        private static unsafe void AttentionDecodeKVHeadGrouped(int kvHead,
            float* qPtr, float* kPtr, float* vPtr, float* rPtr, float* scores,
            int headDim, int maxSeqLen, int groupSize, int totalSeqLen, float scale)
        {
            int hStart = kvHead * groupSize;
            float* kHead = kPtr + (long)kvHead * maxSeqLen * headDim;
            float* vHead = vPtr + (long)kvHead * maxSeqLen * headDim;

            // Per-group running max for online numerical stability. We compute scores
            // per (group, t) into a [groupSize, totalSeqLen] row-major matrix so the
            // later softmax/normalize steps stay vectorizable.
            float maxG0 = float.NegativeInfinity;
            float maxG1 = float.NegativeInfinity;
            float maxG2 = float.NegativeInfinity;
            float maxG3 = float.NegativeInfinity;

            // Score generation: K[t] is read once and dot-producted against groupSize Q heads.
            // Specialize the common groupSize=4 case to keep inner-loop arithmetic tight.
            if (groupSize == 4)
            {
                float* qH0 = qPtr + (long)(hStart + 0) * headDim;
                float* qH1 = qPtr + (long)(hStart + 1) * headDim;
                float* qH2 = qPtr + (long)(hStart + 2) * headDim;
                float* qH3 = qPtr + (long)(hStart + 3) * headDim;
                float* row0 = scores + 0L * totalSeqLen;
                float* row1 = scores + 1L * totalSeqLen;
                float* row2 = scores + 2L * totalSeqLen;
                float* row3 = scores + 3L * totalSeqLen;

                for (int t = 0; t < totalSeqLen; t++)
                {
                    float* kT = kHead + (long)t * headDim;
                    float s0, s1, s2, s3;
                    VecDot4(qH0, qH1, qH2, qH3, kT, headDim, out s0, out s1, out s2, out s3);
                    s0 *= scale; s1 *= scale; s2 *= scale; s3 *= scale;
                    row0[t] = s0; row1[t] = s1; row2[t] = s2; row3[t] = s3;
                    if (s0 > maxG0) maxG0 = s0;
                    if (s1 > maxG1) maxG1 = s1;
                    if (s2 > maxG2) maxG2 = s2;
                    if (s3 > maxG3) maxG3 = s3;
                }
            }
            else
            {
                Span<float> maxScoresSpan = stackalloc float[groupSize];
                for (int g = 0; g < groupSize; g++) maxScoresSpan[g] = float.NegativeInfinity;

                for (int t = 0; t < totalSeqLen; t++)
                {
                    float* kT = kHead + (long)t * headDim;
                    for (int g = 0; g < groupSize; g++)
                    {
                        float* qH = qPtr + (long)(hStart + g) * headDim;
                        float s = VecDot(qH, kT, headDim) * scale;
                        scores[g * totalSeqLen + t] = s;
                        if (s > maxScoresSpan[g]) maxScoresSpan[g] = s;
                    }
                }

                if (groupSize >= 1) maxG0 = maxScoresSpan[0];
                if (groupSize >= 2) maxG1 = maxScoresSpan[1];
                if (groupSize >= 3) maxG2 = maxScoresSpan[2];
                if (groupSize >= 4) maxG3 = maxScoresSpan[3];
            }

            // Softmax (per-group)
            Span<float> invSums = stackalloc float[groupSize];
            for (int g = 0; g < groupSize; g++)
            {
                float maxS;
                if (g == 0) maxS = maxG0;
                else if (g == 1) maxS = maxG1;
                else if (g == 2) maxS = maxG2;
                else if (g == 3) maxS = maxG3;
                else
                {
                    maxS = float.NegativeInfinity;
                    float* rowG0 = scores + (long)g * totalSeqLen;
                    for (int t = 0; t < totalSeqLen; t++)
                        if (rowG0[t] > maxS) maxS = rowG0[t];
                }

                float sum = 0;
                float* rowG = scores + (long)g * totalSeqLen;
                for (int t = 0; t < totalSeqLen; t++)
                {
                    float e = MathF.Exp(rowG[t] - maxS);
                    rowG[t] = e;
                    sum += e;
                }
                invSums[g] = 1.0f / sum;
            }
            for (int g = 0; g < groupSize; g++)
            {
                float invSum = invSums[g];
                float* rowG = scores + (long)g * totalSeqLen;
                VecScale(rowG, invSum, totalSeqLen);
            }

            // Aggregate V: read V[t] once per t, scatter into all groupSize result heads.
            for (int g = 0; g < groupSize; g++)
                VecZero(rPtr + (long)(hStart + g) * headDim, headDim);

            if (groupSize == 4)
            {
                float* r0 = rPtr + (long)(hStart + 0) * headDim;
                float* r1 = rPtr + (long)(hStart + 1) * headDim;
                float* r2 = rPtr + (long)(hStart + 2) * headDim;
                float* r3 = rPtr + (long)(hStart + 3) * headDim;
                float* row0 = scores + 0L * totalSeqLen;
                float* row1 = scores + 1L * totalSeqLen;
                float* row2 = scores + 2L * totalSeqLen;
                float* row3 = scores + 3L * totalSeqLen;

                for (int t = 0; t < totalSeqLen; t++)
                {
                    float* vT = vHead + (long)t * headDim;
                    VecScaleAdd4(r0, r1, r2, r3, vT,
                        row0[t], row1[t], row2[t], row3[t], headDim);
                }
            }
            else
            {
                for (int t = 0; t < totalSeqLen; t++)
                {
                    float* vT = vHead + (long)t * headDim;
                    for (int g = 0; g < groupSize; g++)
                    {
                        float w = scores[g * totalSeqLen + t];
                        float* rH = rPtr + (long)(hStart + g) * headDim;
                        VecScaleAdd(rH, vT, w, headDim);
                    }
                }
            }
        }

        /// <summary>
        /// Compute partial attention for one (KV head, K-chunk) pair. Writes per-group
        /// running max, un-normalized exp sum, and un-normalized weighted-V into the
        /// supplied <paramref name="partial"/> buffer for later cross-chunk merging.
        ///
        /// Layout of <paramref name="partial"/> (length = groupSize * (2 + headDim)):
        ///   [g * (2 + headDim) + 0]            = max for group g
        ///   [g * (2 + headDim) + 1]            = sumExp for group g
        ///   [g * (2 + headDim) + 2 .. + headDim+1] = un-normalized weighted V for group g
        /// </summary>
        private static unsafe void AttentionDecodeChunkPartial(int kvHead,
            int kStart, int kLen,
            float* qPtr, float* kPtr, float* vPtr,
            float* partial, float* scores,
            int headDim, int maxSeqLen, int groupSize, float scale)
        {
            int hStart = kvHead * groupSize;
            float* kHead = kPtr + (long)kvHead * maxSeqLen * headDim;
            float* vHead = vPtr + (long)kvHead * maxSeqLen * headDim;
            int strideG = 2 + headDim;

            for (int g = 0; g < groupSize; g++)
                partial[g * strideG] = float.NegativeInfinity;

            float maxG0 = float.NegativeInfinity;
            float maxG1 = float.NegativeInfinity;
            float maxG2 = float.NegativeInfinity;
            float maxG3 = float.NegativeInfinity;

            if (groupSize == 4)
            {
                float* qH0 = qPtr + (long)(hStart + 0) * headDim;
                float* qH1 = qPtr + (long)(hStart + 1) * headDim;
                float* qH2 = qPtr + (long)(hStart + 2) * headDim;
                float* qH3 = qPtr + (long)(hStart + 3) * headDim;
                float* row0 = scores + 0L * kLen;
                float* row1 = scores + 1L * kLen;
                float* row2 = scores + 2L * kLen;
                float* row3 = scores + 3L * kLen;

                for (int t = 0; t < kLen; t++)
                {
                    float* kT = kHead + (long)(kStart + t) * headDim;
                    float s0, s1, s2, s3;
                    VecDot4(qH0, qH1, qH2, qH3, kT, headDim, out s0, out s1, out s2, out s3);
                    s0 *= scale; s1 *= scale; s2 *= scale; s3 *= scale;
                    row0[t] = s0; row1[t] = s1; row2[t] = s2; row3[t] = s3;
                    if (s0 > maxG0) maxG0 = s0;
                    if (s1 > maxG1) maxG1 = s1;
                    if (s2 > maxG2) maxG2 = s2;
                    if (s3 > maxG3) maxG3 = s3;
                }
            }
            else
            {
                for (int g = 0; g < groupSize; g++)
                    partial[g * strideG] = float.NegativeInfinity;

                for (int t = 0; t < kLen; t++)
                {
                    float* kT = kHead + (long)(kStart + t) * headDim;
                    for (int g = 0; g < groupSize; g++)
                    {
                        float* qH = qPtr + (long)(hStart + g) * headDim;
                        float s = VecDot(qH, kT, headDim) * scale;
                        scores[g * kLen + t] = s;
                        if (s > partial[g * strideG]) partial[g * strideG] = s;
                    }
                }
            }

            if (groupSize == 4)
            {
                partial[0 * strideG] = maxG0;
                partial[1 * strideG] = maxG1;
                partial[2 * strideG] = maxG2;
                partial[3 * strideG] = maxG3;
            }

            // Softmax per group (un-normalized) and partial weighted V
            for (int g = 0; g < groupSize; g++)
            {
                float maxS = partial[g * strideG];
                float sum = 0;
                float* rowG = scores + (long)g * kLen;
                for (int t = 0; t < kLen; t++)
                {
                    float e = MathF.Exp(rowG[t] - maxS);
                    rowG[t] = e;
                    sum += e;
                }
                partial[g * strideG + 1] = sum;
            }

            // Compute weighted V for this chunk
            for (int g = 0; g < groupSize; g++)
                VecZero(partial + g * strideG + 2, headDim);

            if (groupSize == 4)
            {
                float* w0 = partial + 0 * strideG + 2;
                float* w1 = partial + 1 * strideG + 2;
                float* w2 = partial + 2 * strideG + 2;
                float* w3 = partial + 3 * strideG + 2;
                float* row0 = scores + 0L * kLen;
                float* row1 = scores + 1L * kLen;
                float* row2 = scores + 2L * kLen;
                float* row3 = scores + 3L * kLen;

                for (int t = 0; t < kLen; t++)
                {
                    float* vT = vHead + (long)(kStart + t) * headDim;
                    VecScaleAdd4(w0, w1, w2, w3, vT,
                        row0[t], row1[t], row2[t], row3[t], headDim);
                }
            }
            else
            {
                for (int t = 0; t < kLen; t++)
                {
                    float* vT = vHead + (long)(kStart + t) * headDim;
                    for (int g = 0; g < groupSize; g++)
                    {
                        float w = scores[g * kLen + t];
                        VecScaleAdd(partial + g * strideG + 2, vT, w, headDim);
                    }
                }
            }
        }

        /// <summary>
        /// Combine the per-chunk partial sums into the final attention output for one KV head.
        /// Uses the standard online softmax merge: M = max(M_a, M_b),
        ///   sum_new = sum_a*exp(M_a - M) + sum_b*exp(M_b - M),
        ///   acc_new = acc_a*exp(M_a - M) + acc_b*exp(M_b - M),
        /// then divide acc_new by sum_new at the end.
        /// </summary>
        private static unsafe void MergeChunkResults(int kvHead, float* rPtr, float* partial,
            int headDim, int groupSize, int kSplit)
        {
            int strideG = 2 + headDim;
            int strideChunk = groupSize * strideG;
            int hStart = kvHead * groupSize;

            for (int g = 0; g < groupSize; g++)
            {
                float globalMax = float.NegativeInfinity;
                for (int c = 0; c < kSplit; c++)
                {
                    float m = partial[c * strideChunk + g * strideG];
                    if (m > globalMax) globalMax = m;
                }

                float globalSum = 0;
                float* rOut = rPtr + (long)(hStart + g) * headDim;
                VecZero(rOut, headDim);

                for (int c = 0; c < kSplit; c++)
                {
                    float* p = partial + c * strideChunk + g * strideG;
                    float chunkMax = p[0];
                    float chunkSum = p[1];
                    if (chunkSum <= 0) continue;
                    float* chunkAcc = p + 2;

                    float scale = MathF.Exp(chunkMax - globalMax);
                    globalSum += chunkSum * scale;
                    VecScaleAdd(rOut, chunkAcc, scale, headDim);
                }

                if (globalSum > 0)
                    VecScale(rOut, 1.0f / globalSum, headDim);
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

        private static IntPtr GetStorageBasePtr(Tensor t)
        {
            if (t.Storage is GgmlStorage gs)
                return gs.PtrAtElement(0);
            if (t.Storage is CpuStorage cs)
                return cs.PtrAtElement(0);
            throw new NotSupportedException("Requires GgmlStorage or CpuStorage");
        }

        protected void InvalidateTensorDeviceCache(Tensor tensor)
        {
            if (!IsGgmlBackend || tensor == null)
                return;

            GgmlBasicOps.InvalidateHostBuffer(GetStoragePtr(tensor));
        }

        protected void SyncTensorHostCache(Tensor tensor)
        {
            if (!IsGgmlBackend || tensor == null)
                return;
                
            GgmlBasicOps.SyncHostBuffer(GetStorageBasePtr(tensor), tensor.Storage.ByteLength);
        }

        public abstract float[] Forward(int[] tokens);
        public virtual float[] ForwardRefill(int[] tokens) => Forward(tokens);
        public abstract void ResetKVCache();

        /// <summary>
        /// Whether this model supports partial KV cache truncation.
        /// Models with recurrent layers (e.g. Qwen3.5) cannot truncate because
        /// the running recurrent state cannot be rewound to an earlier position.
        /// </summary>
        public virtual bool SupportsKVCacheTruncation => true;

        /// <summary>
        /// Truncate KV cache to keep only the first <paramref name="tokenCount"/> positions.
        /// Subsequent Forward calls will append starting at this position.
        /// Subclasses MUST override to invalidate device (GPU/Metal) caches.
        /// </summary>
        public virtual void TruncateKVCache(int tokenCount)
        {
            Console.WriteLine($"[KV cache] Truncating from {_cacheSeqLen} to {tokenCount}");
            _cacheSeqLen = tokenCount;
        }

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

        public virtual void PrintTimingStats()
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
            if (MultimodalInjector is IDisposable multimodalInjector)
                multimodalInjector.Dispose();

            foreach (var w in _weights.Values)
                w.Dispose();
            _weights.Clear();

            if (IsGgmlBackend)
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
                "gptoss" or "gpt-oss" => new GptOssModel(ggufPath, backend),
                "nemotron_h" or "nemotron_h_moe" => new NemotronModel(ggufPath, backend),
                "mistral3" => new Mistral3Model(ggufPath, backend),
                _ => throw new NotSupportedException($"Unsupported architecture: {arch}"),
            };
        }
    }
}
