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
using System.Linq;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp;
using TensorSharp.Cpu;
using TensorSharp.Cuda;
using TensorSharp.GGML;
using TensorSharp.MLX;

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
        internal bool IsExternalHostViewOwnedBy(object owner)
            => HasExternalHostView && ReferenceEquals(_ownerToken, owner);

        /// <summary>
        /// True when the active device could not hold this weight in a single
        /// backend buffer (e.g. ggml-vulkan rejects any buffer above the driver's
        /// maxBufferSize; WSL's dzn layer caps it under 3 GB), so the device
        /// preload was skipped and the host copy retained. Consumers must serve
        /// this weight through their host-gather/dequant fallback instead of
        /// device-side lookups keyed by <see cref="CacheKey"/>.
        /// </summary>
        public bool DevicePreloadTooLarge { get; private set; }

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

            // Once flagged too-large the cache key must stay the host data
            // pointer: no device-resident entry exists for this weight, and a
            // native cache miss on an opaque GCHandle key would dereference it
            // as if it were weight bytes.
            if (DevicePreloadTooLarge)
                return CacheKey;

            _cacheKeyHandle = GCHandle.Alloc(this, GCHandleType.Normal);
            CacheKey = GCHandle.ToIntPtr(_cacheKeyHandle);
            _ownsCacheKeyHandle = true;
            return CacheKey;
        }

        /// <summary>
        /// Record that the device preload was skipped because this weight exceeds
        /// the device's single-buffer size limit. Frees any GCHandle-based device
        /// cache key and restores <see cref="CacheKey"/> to the host data pointer,
        /// so a native call that still receives the key resolves through the
        /// host-pointer path instead of dereferencing an opaque GCHandle.
        /// </summary>
        public void MarkDevicePreloadTooLarge()
        {
            DevicePreloadTooLarge = true;
            if (_ownsCacheKeyHandle)
            {
                _cacheKeyHandle.Free();
                _ownsCacheKeyHandle = false;
            }
            CacheKey = _data;
        }

        public void ReleaseHostData()
        {
            if (_data == IntPtr.Zero)
                return;

            IntPtr currentData = _data;
            // Page-out advice is only valid for views into the GgufFile mmap,
            // where MADV_DONTNEED just drops clean file-backed pages. On views
            // into another weight's heap buffer (per-expert views of a stacked
            // 3D tensor, TP shard views) the same call ZEROES the anonymous
            // pages in place — including malloc metadata of whatever else
            // shares them — and the next free() aborts the process.
            bool wasFileBackedView = !_ownsBuffer && _ownerToken is GgufFile;
            if (_ownsBuffer)
                FreeBuffer(currentData);
            else if (wasFileBackedView)
                AdviseExternalViewCanBePagedOut(currentData, RawBytes);

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

        private static unsafe void AdviseExternalViewCanBePagedOut(IntPtr data, long byteCount)
        {
            if (data == IntPtr.Zero || byteCount <= 0)
                return;
            if (!OperatingSystem.IsMacOS() && !OperatingSystem.IsLinux())
                return;

            long pageSize = Environment.SystemPageSize;
            long address = data.ToInt64();
            long pageMask = ~(pageSize - 1);
            long alignedAddress = address & pageMask;
            long prefixBytes = address - alignedAddress;
            ulong length = checked((ulong)(byteCount + prefixBytes));
            ulong roundedLength = (length + (ulong)pageSize - 1) & ~((ulong)pageSize - 1);

            try
            {
                _ = madvise((void*)alignedAddress, (nuint)roundedLength, MadvDontNeed);
            }
            catch (DllNotFoundException)
            {
            }
            catch (EntryPointNotFoundException)
            {
            }
        }

        private const int MadvDontNeed = 4;

        [DllImport("libc", SetLastError = true, EntryPoint = "madvise")]
        private static extern unsafe int madvise(void* addr, nuint len, int advice);
    }

    /// <summary>
    /// A view of a per-layer 3D MoE expert weight tensor as stored on disk
    /// (<c>[ne0, ne1, num_experts]</c> contiguous). Built when the per-expert
    /// quantized weights are split out of the original 3D GGUF tensor in
    /// <see cref="ModelBase.LoadWeights"/>, so it costs nothing on top of the
    /// per-expert weights for mmap'd models — the base pointer is the start
    /// of the original 3D block and the bytes are the same bytes the per-expert
    /// views point into.
    ///
    /// The <see cref="MoEFFNPrefillSwiGLU"/> kernel consumes this directly to
    /// run an entire MoE layer's gate/up/down via three <c>ggml_mul_mat_id</c>
    /// dispatches (mirroring llama.cpp's <c>build_moe_ffn</c>) instead of the
    /// previous per-active-expert loop that issued thousands of dispatches per
    /// pp2048 forward.
    /// </summary>
    public sealed class StackedExpertWeights
    {
        public IntPtr Data { get; }
        public int GgmlType { get; }
        public long PerExpertNe0 { get; }
        public long PerExpertNe1 { get; }
        public int NumExperts { get; }
        public long TotalRawBytes { get; }
        public long PerExpertRawBytes => TotalRawBytes / NumExperts;
        public bool IsExternalView { get; }

        // Strong reference held to keep the underlying memory alive when this
        // is an external view (e.g. into a GgufFile mmap or a sibling owning
        // QuantizedWeight buffer). For owned buffers this is null.
        private readonly object _ownerToken;

        // For the non-mmap fallback path we own a pinned native buffer and
        // free it on disposal of the parent ModelBase. Tracked so the buffer
        // doesn't leak when ModelBase exits.
        public IntPtr OwnedBuffer { get; }

        public StackedExpertWeights(
            IntPtr data,
            int ggmlType,
            long perExpertNe0,
            long perExpertNe1,
            int numExperts,
            long totalRawBytes,
            bool isExternalView,
            object ownerToken,
            IntPtr ownedBuffer)
        {
            Data = data;
            GgmlType = ggmlType;
            PerExpertNe0 = perExpertNe0;
            PerExpertNe1 = perExpertNe1;
            NumExperts = numExperts;
            TotalRawBytes = totalRawBytes;
            IsExternalView = isExternalView;
            _ownerToken = ownerToken;
            OwnedBuffer = ownedBuffer;
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

        // ---- Tensor Parallelism ----
        protected readonly ITensorParallelGroup _tpGroup;
        protected int TpDegree => _tpGroup?.Degree ?? 1;
        protected bool IsTensorParallel => _tpGroup != null && _tpGroup.IsActive;

        /// <summary>Total GPUs across all nodes (for weight shard sizing).</summary>
        protected int GlobalTpDegree => _tpGroup?.GlobalDegree ?? 1;

        /// <summary>First global rank on this node (for selecting local weight shards).</summary>
        protected int TpRankOffset => _tpGroup?.GlobalRankOffset ?? 0;

        /// <summary>Number of nodes in the distributed group (1 for local-only).</summary>
        protected int TpNodeCount => _tpGroup?.NodeCount ?? 1;

        /// <summary>Per-GPU sharded quantized weights, keyed by weight name.</summary>
        protected readonly Dictionary<string, QuantizedWeight[]> _tpQuantWeights = new();

        /// <summary>Per-GPU sharded F32 weights, keyed by weight name.</summary>
        protected readonly Dictionary<string, Tensor[]> _tpWeights = new();

        /// <summary>
        /// Stacked-along-experts views of MoE expert weight tensors keyed by
        /// the original GGUF tensor name (e.g. <c>"blk.0.ffn_gate_exps.weight"</c>).
        /// Populated in <see cref="LoadWeights"/> for any 3D <c>_exps.</c>
        /// tensor. Used by <see cref="GgmlBasicOps.MoEFFNPrefillSwiGLU"/> to
        /// dispatch the entire MoE FFN as a few <c>ggml_mul_mat_id</c> calls
        /// per layer instead of per-active-expert. May be null/empty when the
        /// model doesn't expose stacked views (e.g. some non-mmap paths).
        /// </summary>
        protected readonly Dictionary<string, StackedExpertWeights> _stackedExpertWeights = new();

        /// <summary>
        /// Names of the per-expert split views in <see cref="_quantWeights"/> that
        /// were carved out of a 3D <c>_exps.</c> tensor and are also covered by a
        /// <see cref="_stackedExpertWeights"/> entry (same underlying bytes). A
        /// model whose CUDA path serves MoE experts exclusively through the
        /// stacked-expert device buffer can consult this set (via
        /// <see cref="ShouldPreloadCudaQuantWeightToDevice"/>) to skip giving each
        /// per-expert view its own device copy, which would otherwise duplicate
        /// every expert byte a second time in VRAM on top of the stacked copy.
        /// </summary>
        protected readonly HashSet<string> _stackedExpertMemberNames = new();
        private bool _quantBackendReady;
        private bool _cudaQuantWeightsPrepared;
        private bool _mlxQuantWeightsPrepared;

        protected int _cacheSeqLen;
        protected int _maxContextLength;
        protected float[] _logitsBuffer;

        /// <summary>
        /// Storage dtype for the per-layer K/V cache tensors. Captured at model
        /// construction time from <see cref="KvCacheDtypeConfig.Current"/> so the
        /// rest of the per-model code (cache allocation, write-on-decode,
        /// attention reads, native-layer-decode bindings) can specialize without
        /// repeatedly polling the global config.
        /// </summary>
        protected KvCacheDtype _kvCacheDtype = KvCacheDtypeConfig.Current;

        /// <summary>
        /// Pick a model-aligned default KV-cache dtype based on the dominant
        /// weight quantization tier seen in <paramref name="quantWeights"/>.
        /// Mirrors <see cref="KvCacheDtypeConfig.ApplyModelDtypeDefault"/> but
        /// is callable from inside a model constructor (after LoadWeights, before
        /// InitKVCache) so each model picks its own default without forcing the
        /// CLI front-end to inspect every GGUF file. Honors any explicit user
        /// choice (env var or <c>--kv-cache-dtype</c> flag) - we only step in
        /// when the user has left the dtype unset.
        /// </summary>
        protected void ApplyModelAlignedKvCacheDefault(IDictionary<string, QuantizedWeight> quantWeights)
        {
            if (KvCacheDtypeConfig.IsExplicitlySet) return;

            int dominant = 0; // GGML_TYPE_F32
            if (quantWeights != null && quantWeights.Count > 0)
            {
                Dictionary<int, long> typeBytes = new Dictionary<int, long>();
                foreach (var qw in quantWeights.Values)
                {
                    if (qw == null) continue;
                    if (!typeBytes.TryGetValue(qw.GgmlType, out long bytes)) bytes = 0;
                    typeBytes[qw.GgmlType] = bytes + qw.RawBytes;
                }
                long bestBytes = 0;
                foreach (var kv in typeBytes)
                {
                    if (kv.Value > bestBytes) { bestBytes = kv.Value; dominant = kv.Key; }
                }
            }

            KvCacheDtypeConfig.ApplyModelDtypeDefault(dominant);
            _kvCacheDtype = KvCacheDtypeConfig.Current;
        }

        public KvCacheDtype KvCacheDtype => _kvCacheDtype;

        /// <summary>
        /// Map the model's KV-cache storage dtype to the codec element type
        /// the paged tier's optional TurboQuant codec uses to interpret the
        /// raw block bytes. Block-quantized caches (Q8_0, Q4_0) bypass the codec
        /// entirely (the bytes are already quantized with their own per-block
        /// scale, so re-quantizing would compound error for no real shrink).
        /// Q4_0 maps onto the same passthrough handling as Q8_0 - the codec's
        /// passthrough branch and <c>FromEnvironment</c> skip are keyed on the
        /// Q8_0 element type, which means "already block-quantized; leave the
        /// bytes untouched" regardless of the underlying 4- vs 8-bit width.
        /// </summary>
        public virtual KvCodecElementType KVStateElementType => _kvCacheDtype switch
        {
            KvCacheDtype.F32 => KvCodecElementType.Float32,
            KvCacheDtype.F16 => KvCodecElementType.Float16,
            KvCacheDtype.Q8_0 => KvCodecElementType.Q8_0,
            KvCacheDtype.Q4_0 => KvCodecElementType.Q8_0,
            _ => KvCodecElementType.Float32,
        };

        public int MaxContextLength => _maxContextLength;
        public int CacheSeqLen => _cacheSeqLen;

        /// <summary>Prefill-length hint (see <see cref="IModelArchitecture.PrepareForPrefill"/>).
        /// Default no-op; models with a grow-on-demand KV cache override to pre-size it.</summary>
        public virtual void PrepareForPrefill(int requiredContextTokens) { }

        // Timing
        protected long _linearTicks;
        protected long _attnTicks;
        protected long _normTicks;
        protected long _embTicks, _lmHeadTicks, _logitsCopyTicks;
        protected int _forwardCount;
        protected Stopwatch _forwardSw = new Stopwatch();

        protected ModelBase(string ggufPath, BackendType backend, int tpDegree = 1, ITensorParallelGroup tpGroup = null)
        {
            _backend = backend;
            // The pure-C# CPU backend must never touch native (ggml P/Invoke) dequant — route
            // every dequant/row-size through the managed implementation (bit-exact vs native,
            // verified). Other backends keep native dequant (faster load; their runtime quant
            // ops go through GgmlBasicOps, not NativeDequant). One model/backend at a time.
            NativeDequant.PreferManaged = backend == BackendType.Cpu;
            ExecutionPlan = new BackendExecutionPlan(backend);
            MultimodalInjector = new ModelMultimodalInjector(this);

            if (tpGroup != null)
                _tpGroup = tpGroup;
            else if (tpDegree > 1 && backend == BackendType.Cuda)
                _tpGroup = new Cuda.TensorParallelGroup(tpDegree);

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
                case BackendType.GgmlVulkan:
                {
                    var ggmlType = backend == BackendType.GgmlCuda ? GgmlBackendType.Cuda : GgmlBackendType.Vulkan;
                    // A caller-supplied group (multi-node) already owns the
                    // multi-GPU context; reuse it rather than initializing the
                    // devices a second time.
                    _ggmlContext = FindGgmlContext(_tpGroup) ?? CreateGgmlContext(ggmlType, tpDegree);
                    _tpGroup ??= CreateGgmlTpGroup(_ggmlContext);
                    _allocator = _tpGroup != null ? _tpGroup.GetAllocator(0) : new GgmlAllocator(_ggmlContext, 0);
                    break;
                }
                case BackendType.Cuda:
                    _allocator = _tpGroup != null ? _tpGroup.GetAllocator(0) : new CudaAllocator(0);
                    break;
                case BackendType.Mlx:
                    MlxBackend.Register();
                    _allocator = new MlxAllocator(0);
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

        /// <summary>
        /// Build the GGML context, spanning several GPUs when tensor parallelism
        /// is requested. Device ordinals are 0..degree-1 by default; set
        /// TENSORSHARP_TP_DEVICES (e.g. "0,2") to pick specific GPUs, which is how
        /// you avoid a display-attached or otherwise busy card.
        /// </summary>
        private static GgmlContext CreateGgmlContext(GgmlBackendType backendType, int tpDegree)
        {
            if (tpDegree <= 1)
                return new GgmlContext(new[] { 0 }, backendType);

            int[] devices = ParseTpDevices(tpDegree);
            int available = GgmlBasicOps.GetGpuDeviceCount(backendType);
            if (available < tpDegree)
            {
                throw new InvalidOperationException(
                    $"Requested tensor-parallel degree {tpDegree} but the GGML {backendType} backend sees only {available} GPU(s).");
            }
            return new GgmlContext(devices, backendType);
        }

        private static int[] ParseTpDevices(int tpDegree)
        {
            string raw = Environment.GetEnvironmentVariable("TENSORSHARP_TP_DEVICES");
            if (string.IsNullOrWhiteSpace(raw))
            {
                var seq = new int[tpDegree];
                for (int i = 0; i < tpDegree; i++) seq[i] = i;
                return seq;
            }

            string[] parts = raw.Split(new[] { ',', ';', ' ' }, StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length != tpDegree)
            {
                throw new ArgumentException(
                    $"TENSORSHARP_TP_DEVICES lists {parts.Length} device(s) but the tensor-parallel degree is {tpDegree}.");
            }
            var devices = new int[tpDegree];
            for (int i = 0; i < tpDegree; i++)
            {
                if (!int.TryParse(parts[i].Trim(), out devices[i]) || devices[i] < 0)
                    throw new ArgumentException($"Invalid device index '{parts[i]}' in TENSORSHARP_TP_DEVICES.");
            }
            return devices;
        }

        private static ITensorParallelGroup CreateGgmlTpGroup(GgmlContext context)
            => context.Degree > 1 ? new GgmlTensorParallelGroup(context) : null;

        private static GgmlContext FindGgmlContext(ITensorParallelGroup group)
        {
            if (group is INestedTensorParallelGroup nested)
                group = nested.LocalGroup;
            return (group as GgmlTensorParallelGroup)?.Context;
        }

        /// <summary>
        /// Build the GGML multi-GPU group that a distributed (multi-node) run
        /// wraps. The cross-node layer needs the on-node group up front, before a
        /// model exists, so this is exposed for the CLI and the server to call.
        /// </summary>
        public static GgmlTensorParallelGroup CreateGgmlLocalTpGroup(BackendType backend, int localDegree)
        {
            GgmlBackendType ggmlType = backend switch
            {
                BackendType.GgmlCuda => GgmlBackendType.Cuda,
                BackendType.GgmlVulkan => GgmlBackendType.Vulkan,
                _ => throw new ArgumentException($"{backend} is not a multi-device GGML backend.", nameof(backend)),
            };
            return new GgmlTensorParallelGroup(CreateGgmlContext(ggmlType, Math.Max(1, localDegree)));
        }

        /// <summary>
        /// True when the model carries per-expert MoE weights. Detected from the
        /// loaded weight names so it works before any model-specific expert
        /// bookkeeping is built.
        /// </summary>
        protected bool HasMoEExpertWeights
        {
            get
            {
                if (_hasMoEExpertWeights.HasValue)
                    return _hasMoEExpertWeights.Value;
                bool found = false;
                foreach (var name in _quantWeights.Keys)
                {
                    if (name.Contains("_exps", StringComparison.Ordinal)) { found = true; break; }
                }
                if (!found)
                {
                    foreach (var name in _tpQuantWeights.Keys)
                    {
                        if (name.Contains("_exps", StringComparison.Ordinal)) { found = true; break; }
                    }
                }
                _hasMoEExpertWeights = found;
                return found;
            }
        }
        private bool? _hasMoEExpertWeights;

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
                BackendType.GgmlVulkan => GgmlBackendType.Vulkan,
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

            // Sampling parameters the model author shipped in the file. llama.cpp
            // layers these over its own defaults for every field the operator did
            // not pin; TensorSharp read none of them, so a GGUF that says
            // "sample me with top_k=20, top_p=0.95" was ignored.
            Config.RecommendedSampling = RecommendedSampling.FromGgufMetadata(_gguf.Metadata);
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

        /// <summary>
        /// Initial direct-CUDA KV allocation used when the GGUF advertises a much
        /// larger context and MAX_CONTEXT was not supplied. Models may raise this
        /// when their cache is compact enough that avoiding an in-request resize is
        /// a better trade than the extra resident memory.
        /// </summary>
        protected virtual int NativeCudaInitialCacheAllocationLength => 2048;

        /// <summary>
        /// Multi-token direct-CUDA startup warmup. Most architectures stay at a
        /// lightweight shape; models with persistent prefill scratch may override.
        /// TS_PREFILL_WARMUP_LEN remains the explicit operator override.
        /// </summary>
        protected virtual int NativeCudaPrefillWarmupLength => 32;

        /// <summary>
        /// Additional input tokens a model needs beyond its target prefill rows.
        /// For example, a refill implementation may reserve one final token for
        /// the logits-producing decode pass. Applied after the startup-cost cap;
        /// an explicit TS_PREFILL_WARMUP_LEN receives no model-specific overhead.
        /// </summary>
        protected virtual int NativeCudaPrefillWarmupTokenOverhead => 0;

        /// <summary>
        /// Some direct-CUDA models cache different decode graphs on opposite
        /// sides of an attention launch crossover. Revisit the one-token shape
        /// after a long prefill warmup so startup leaves the short graph captured.
        /// </summary>
        protected virtual bool NativeCudaPrimeShortDecodeGraphAfterPrefill => false;

        protected int ResolveInitialCacheAllocationLength(int requestedContextLength, int gpuDefault = 8192)
        {
            return ResolveInitialCacheAllocationLength(
                _backend,
                requestedContextLength,
                gpuDefault,
                NativeCudaInitialCacheAllocationLength);
        }

        internal static int ResolveInitialCacheAllocationLength(
            BackendType backend,
            int requestedContextLength,
            int gpuDefault = 8192,
            int nativeCudaDefault = 2048)
        {
            // GPU backends can be sensitive to allocating a multi-gigabyte KV
            // cache up-front when the model advertises a 256K+ context window. Cap the initial
            // allocation and let the cache grow on demand when actual prompts approach the
            // limit. CPU backends have no such constraint and use the full requested length.
            bool isGpuBackend =
                backend == BackendType.Cuda ||
                backend == BackendType.Mlx ||
                backend == BackendType.GgmlCuda ||
                backend == BackendType.GgmlVulkan ||
                backend == BackendType.GgmlMetal;
            string maxContextOverride = Environment.GetEnvironmentVariable("MAX_CONTEXT");
            bool hasValidExplicitContext =
                !string.IsNullOrWhiteSpace(maxContextOverride) &&
                int.TryParse(maxContextOverride, out int explicitContext) &&
                explicitContext > 0;
            if (isGpuBackend && !hasValidExplicitContext)
            {
                // Direct GPU backends benefit from a smaller initial KV allocation so
                // huge advertised contexts (for example 262k) do not reserve the entire
                // GPU budget before the dynamic CPU/KV cache compressor activates.
                // The cache grows on demand for longer sessions; users with persistent
                // long contexts can override via MAX_CONTEXT to allocate the full window
                // up-front.
                int effectiveDefault = backend switch
                {
                    BackendType.Mlx => Math.Min(gpuDefault, 2048),
                    BackendType.Cuda => Math.Min(gpuDefault, Math.Max(1, nativeCudaDefault)),
                    // GgmlMetal: cap the up-front KV allocation too. On Apple
                    // Silicon the GPU working set is small (e.g. ~19 GB) and a
                    // big model (gpt-oss 20B Q8_0 ≈ 12 GB) plus a full 8192-token
                    // KV cache (≈6.4 GB) sits right at the limit. Once memory is
                    // that tight the OS continually purges wired buffers, so the
                    // residency-set keep-alive thread must constantly re-request
                    // residency for every buffer — holding the device residency
                    // lock that every per-op buffer alloc/free contends on, which
                    // collapses long-context decode to seconds/token (GPU idle).
                    // A smaller initial allocation (the cache still grows on
                    // demand) keeps headroom so residency stays cheap.
                    BackendType.GgmlMetal => Math.Min(gpuDefault, 2048),
                    _ => gpuDefault,
                };
                return Math.Min(requestedContextLength, effectiveDefault);
            }

            return requestedContextLength;
        }

        internal static int ResolvePrefillWarmupInputLength(
            int targetLength,
            int maxContextLength,
            int tokenOverhead,
            bool explicitLength)
        {
            int length = Math.Max(2, targetLength);
            if (maxContextLength > 0)
                length = Math.Min(length, Math.Max(2, maxContextLength / 4));

            if (!explicitLength && tokenOverhead > 0)
            {
                length += tokenOverhead;
                if (maxContextLength > 0)
                    length = Math.Min(length, maxContextLength);
            }
            return length;
        }

        internal static bool UsesLightweightPrefillWarmupByDefault(BackendType backend)
        {
            // Metal is always unified-memory on supported Apple hardware. Unlike
            // CUDA/Vulkan, several Metal model paths cannot use the persistent
            // whole-model prefill graph, so a 2048-token startup pass only creates
            // a large transient working set. On near-capacity models that can pin
            // ggml-metal's residency-set worker in requestResidency while per-op
            // buffer destruction waits on the same lock, making startup appear
            // permanently hung. A 32-token pass still compiles the multi-token
            // kernels without manufacturing long-prompt memory pressure.
            return backend == BackendType.Mlx ||
                   backend == BackendType.Cpu ||
                   backend == BackendType.GgmlMetal;
        }

        internal static int ResolvePrefillWarmupTargetLength(
            BackendType backend,
            bool integratedGpu,
            bool mostlyHostBacked,
            bool moeUnderTp,
            int nativeCudaLength,
            int? explicitLength)
        {
            if (explicitLength is >= 2)
                return explicitLength.Value;

            bool conservative = UsesLightweightPrefillWarmupByDefault(backend)
                || integratedGpu || mostlyHostBacked || moeUnderTp;
            if (conservative)
                return 32;
            return backend == BackendType.Cuda
                ? Math.Max(2, nativeCudaLength)
                : 2048;
        }

        // GgmlVulkan and GgmlMetal follow GgmlCuda here: the fused prefill/decode
        // paths mask or overwrite every cache position they read, so zero-filling
        // 100s of MB of host KV arrays on every request reset is pure waste.
        protected bool ShouldZeroFillCacheTensors =>
            _backend != BackendType.GgmlCuda && _backend != BackendType.Mlx &&
            _backend != BackendType.GgmlVulkan && _backend != BackendType.GgmlMetal;

        protected void InitializeCacheTensor(Tensor tensor)
        {
            // First allocation still zero-fills on every backend that keeps a host
            // copy (including Vulkan/Metal): the fused kernels' flash-padding may
            // read never-written cache rows, which must be finite.
            if (tensor != null && (ShouldZeroFillCacheTensors ||
                _backend == BackendType.GgmlVulkan || _backend == BackendType.GgmlMetal))
                Ops.Fill(tensor, 0f);
        }

        protected void ResetCacheTensor(Tensor tensor)
        {
            if (tensor == null)
                return;

            if (ShouldZeroFillCacheTensors)
                Ops.Fill(tensor, 0f);

            // On GgmlVulkan/Metal keep the resident device copy VALID across
            // logical resets. The host copy was not touched above, so both retain
            // the previous request's bytes; _cacheSeqLen plus the attention mask
            // makes those rows semantically empty until overwritten. Besides
            // avoiding a full KV re-upload, stable Metal bindings let a warmed
            // persistent decode graph survive from startup into the first request.
            if (_backend != BackendType.GgmlVulkan && _backend != BackendType.GgmlMetal)
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

        /// <summary>
        /// Decide whether the tokenizer should prepend a BOS token when encoding a
        /// prompt with <c>addSpecial=true</c>.
        ///
        /// Normally this mirrors the GGUF's <c>tokenizer.ggml.add_bos_token</c> flag.
        /// However, some GGUF conversions (notably several Gemma 4 builds, e.g.
        /// gemma-4-31B IQ2_M) set <c>add_bos_token=false</c> and instead rely on the
        /// chat template's leading <c>{{ bos_token }}</c> to emit the
        /// beginning-of-sequence marker. TensorSharp always renders <c>bos_token</c> as
        /// an empty string (and its hardcoded chat renderers deliberately omit a literal
        /// BOS to avoid a double BOS when the tokenizer owns it), so for such models the
        /// rendered prompt would otherwise carry NO BOS at all. A Gemma-family model
        /// with a missing BOS degenerates into repetition / off-topic output. When the
        /// template declares a leading BOS but the tokenizer is configured not to add
        /// one, let the tokenizer own it so the prompt still begins with exactly one BOS
        /// (the empty-rendered <c>bos_token</c> guarantees we never double it).
        /// </summary>
        public static bool ResolveAddBosToken(
            bool addBosFromMetadata,
            int bosTokenId,
            string? chatTemplate,
            string? tokenizerModel = null)
        {
            if (addBosFromMetadata)
                return true;
            if (bosTokenId < 0)
                return false;
            // Gemma 4's tokenizer contract requires BOS even when older GGUF
            // conversions recorded add_bos_token=false.  This is independent
            // of whether a particular chat template mentions bos_token.
            if (string.Equals(tokenizerModel, "gemma4", StringComparison.Ordinal))
                return true;
            return !string.IsNullOrEmpty(chatTemplate)
                && chatTemplate.Contains("bos_token", StringComparison.Ordinal);
        }

        // llama.cpp's vocabulary loader treats these control-token spellings as
        // end-of-generation even when a GGUF converter only records one of them
        // in tokenizer.ggml.eos_token_id.  Qwen3.5 is a concrete example: its
        // metadata names <|im_end|>, while <|endoftext|> is also a valid EOG.
        private static readonly HashSet<string> TextualEogTokens = new(StringComparer.Ordinal)
        {
            "<|eot_id|>",
            "<|im_end|>",
            "<|end|>",
            "<|return|>",
            "<|call|>",
            "<|flush|>",
            "<|calls|>",
            "<end_of_turn>",
            "<|endoftext|>",
            "</s>",
            "<|eom_id|>",
            "<EOT>",
            "_<EOT>",
            "[EOT]",
            "[EOS]",
            "<|end_of_text|>",
            "<end_of_utterance>",
            "<eos>",
            "<turn|>",
            "<|tool_response>",
            "<｜end▁of▁sentence｜>",
        };

        /// <summary>
        /// Augment the GGUF EOS list with llama-compatible, text-discovered EOG
        /// controls. Public for tokenizer regression tests.
        /// </summary>
        public static int[] ResolveEogTokenIds(
            IReadOnlyList<string> vocabTokens,
            int eosId,
            IEnumerable<int>? extraEosIds = null)
        {
            var ids = new HashSet<int>();
            if (eosId >= 0 && eosId < vocabTokens.Count)
                ids.Add(eosId);
            if (extraEosIds != null)
            {
                foreach (int id in extraEosIds)
                    if (id >= 0 && id < vocabTokens.Count)
                        ids.Add(id);
            }

            for (int id = 0; id < vocabTokens.Count; id++)
            {
                if (TextualEogTokens.Contains(vocabTokens[id]))
                    ids.Add(id);
            }

            // Match llama.cpp's tokenizer-specific EOG workarounds. Harmony
            // and Solar use <|end|> as a structural marker rather than a stop;
            // Gemma4/PaddleOCR similarly use </s> as ordinary vocabulary when
            // the tool-response control token is present.
            int endId = -1;
            int slashSId = -1;
            bool hasReturn = false;
            bool hasCall = false;
            bool hasFlush = false;
            bool hasToolResponse = false;
            foreach (int id in ids)
            {
                switch (vocabTokens[id])
                {
                    case "<|return|>": hasReturn = true; break;
                    case "<|call|>":
                    case "<|calls|>": hasCall = true; break;
                    case "<|flush|>": hasFlush = true; break;
                    case "<|end|>": endId = id; break;
                    case "<|tool_response>": hasToolResponse = true; break;
                    case "</s>": slashSId = id; break;
                }
            }
            if (endId >= 0 && ((hasReturn && hasCall) || (hasCall && hasFlush)))
                ids.Remove(endId);
            if (slashSId >= 0 && hasToolResponse)
                ids.Remove(slashSId);

            var result = new int[ids.Count];
            ids.CopyTo(result);
            Array.Sort(result);
            return result;
        }

        protected void ParseTokenizer()
        {
            Tokenizer = CreateTokenizerFromGguf(_gguf);
            Config.VocabSize = Tokenizer.VocabSize;
        }

        internal static bool UsesSentencePieceTokenizer(string tokenizerModel)
        {
            return string.Equals(tokenizerModel, "llama", StringComparison.Ordinal)
                || string.Equals(tokenizerModel, "t5", StringComparison.Ordinal);
        }

        /// <summary>
        /// Build the tokenizer declared by a GGUF without loading model weights.
        /// Kept internal so tokenizer-oracle tests exercise the exact production
        /// dispatch instead of duplicating the metadata branching.
        /// </summary>
        /// <summary>
        /// Build just the tokenizer described by a GGUF's metadata, without
        /// loading any weights. Useful for vocabulary-only work such as
        /// compiling a grammar's token masks.
        /// </summary>
        public static ITokenizer CreateTokenizerFromGguf(GgufFile gguf)
        {
            if (gguf == null)
                throw new ArgumentNullException(nameof(gguf));

            var vocabTokens = gguf.GetStringArray("tokenizer.ggml.tokens");

            var tokenTypes = gguf.GetInt32Array("tokenizer.ggml.token_type");
            int bosId = (int)gguf.GetUint32("tokenizer.ggml.bos_token_id");
            int eosId = (int)gguf.GetUint32("tokenizer.ggml.eos_token_id");
            bool addBosMetadata = gguf.GetBool("tokenizer.ggml.add_bos_token", false);
            bool addEos = gguf.GetBool("tokenizer.ggml.add_eos_token", false);
            string tokenizerModel = gguf.GetString("tokenizer.ggml.model", "gpt2");

            bool addBos = ResolveAddBosToken(
                addBosMetadata,
                bosId,
                gguf.GetString("tokenizer.chat_template"),
                tokenizerModel);
            if (addBos && !addBosMetadata)
            {
                Console.WriteLine(
                    "  Tokenizer: add_bos_token=false but the model requires BOS; " +
                    "enabling BOS so the prompt starts with exactly one BOS.");
            }

            var extraEos = gguf.GetInt32Array("tokenizer.ggml.eos_token_ids");
            var eosIds = new List<int>(ResolveEogTokenIds(vocabTokens, eosId, extraEos));

            // llama.cpp folds the declared end-of-turn control into the EOG set
            // for EVERY tokenizer type (llama_vocab::impl::load inserts
            // special_eot_id into special_eog_ids). This used to run only on the
            // SentencePiece branch, so a BPE model whose turn ends on a token
            // other than tokenizer.ggml.eos_token_id never stopped: Muse-Glimmer
            // declares eos_token_id = <|end_of_text|> but ends every assistant
            // turn with <|eot|> (tokenizer.ggml.eot_token_id = 200008), so it ran
            // past its answer and re-answered until max_tokens.
            bool isSentencePiece = UsesSentencePieceTokenizer(tokenizerModel);
            // 106 (<end_of_turn>) is the SentencePiece/Gemma fallback that
            // predates the metadata key; BPE vocabularies get no fallback, only
            // the key when the converter wrote one.
            if (gguf.Metadata.ContainsKey("tokenizer.ggml.eot_token_id") || isSentencePiece)
            {
                int eotId = (int)gguf.GetUint32("tokenizer.ggml.eot_token_id", 106);
                if (eotId >= 0 && eotId < vocabTokens.Length && !eosIds.Contains(eotId))
                    eosIds.Add(eotId);
            }

            if (isSentencePiece)
            {
                var scores = gguf.GetFloatArray("tokenizer.ggml.scores");
                return new SentencePieceTokenizer(vocabTokens, tokenTypes, scores,
                    bosId, eosIds.ToArray(), addBos, addEos);
            }

            var merges = gguf.GetStringArray("tokenizer.ggml.merges");
            // tokenizer.ggml.model=gemma4 is an SPM-style BPE vocabulary,
            // not a unigram SentencePiece vocabulary.  It does not need a
            // tokenizer.ggml.pre entry: the model name selects its raw
            // UTF-8/newline-only pre-tokenizer in llama.cpp.
            string preType = tokenizerModel == "gemma4"
                ? "gemma4"
                : gguf.GetString("tokenizer.ggml.pre", null);
            return new BpeTokenizer(vocabTokens, tokenTypes, merges,
                bosId, eosIds.ToArray(), addBos, addEos, preType);
        }

        protected virtual bool IsQuantizedLinearWeight(GgufTensorInfo info)
        {
            return ExecutionPlan.ShouldStoreWeightQuantized(info);
        }

        internal static bool ShouldStoreWeightQuantized(BackendType backend, GgufTensorInfo info)
        {
            if (info.Type == GgmlTensorType.F32)
                return false;

            if (backend == BackendType.Cuda && !CanStoreDirectCudaCompressedWeight(info.Type))
                return false;

            if (backend == BackendType.Cpu && !ManagedQuantizedOps.SupportsCpuQuantizedStorage(info.Type))
                return false;

            if (backend == BackendType.Mlx && !MlxQuantizedOps.SupportsQuantizedType(info.Type))
                return false;

            if (info.Shape.Length == 2)
                return true;

            return info.Shape.Length == 3 && info.Name.Contains("_exps.");
        }

        private static bool CanStoreDirectCudaCompressedWeight(GgmlTensorType type)
        {
            return type switch
            {
                GgmlTensorType.F16 or
                GgmlTensorType.BF16 or
                GgmlTensorType.Q4_0 or
                GgmlTensorType.Q4_1 or
                GgmlTensorType.Q5_0 or
                GgmlTensorType.Q5_1 or
                GgmlTensorType.Q8_0 or
                GgmlTensorType.Q8_1 or
                GgmlTensorType.Q2_K or
                GgmlTensorType.Q3_K or
                GgmlTensorType.Q4_K or
                GgmlTensorType.Q5_K or
                GgmlTensorType.Q6_K or
                GgmlTensorType.Q8_K or
                GgmlTensorType.IQ2_XXS or
                GgmlTensorType.IQ2_XS or
                GgmlTensorType.IQ3_XXS or
                GgmlTensorType.IQ1_S or
                GgmlTensorType.IQ4_NL or
                GgmlTensorType.IQ3_S or
                GgmlTensorType.IQ2_S or
                GgmlTensorType.IQ4_XS or
                GgmlTensorType.IQ1_M or
                GgmlTensorType.TQ1_0 or
                GgmlTensorType.TQ2_0 or
                GgmlTensorType.MXFP4 => true,
                _ => false,
            };
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
            || _backend == BackendType.GgmlVulkan
            || _backend == BackendType.Cuda
            || _backend == BackendType.Mlx
            || _backend == BackendType.GgmlMetal
            || _backend == BackendType.GgmlCpu;

        protected void LoadWeights()
        {
            // Parallel page-cache warm-up first: everything below (serial
            // F32/dequant reads, mmap faults from the sharding/upload threads)
            // otherwise reads the file at one-or-two-stream speed, which is the
            // whole cold-load time on network-backed model storage.
            _gguf.PrefaultFileCache();
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
                        // 3D MoE expert tensor: split into per-expert 2D quantized weights.
                        // Also build a single stacked-along-experts view that the fused
                        // MoE prefill kernel can hand to ggml_mul_mat_id directly.
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
                                string expertName = $"{baseName}.{e}.weight";
                                _quantWeights[expertName] = QuantizedWeight.CreateExternalView(
                                    expertPtr, perExpertBytes, (int)info.Type, ne0, ne1, _gguf);
                                _stackedExpertMemberNames.Add(expertName);
                            }
                            // Free zero-cost stacked view: same bytes the per-expert
                            // views point into, owner is the GgufFile mmap.
                            _stackedExpertWeights[info.Name] = new StackedExpertWeights(
                                mappedTensorPtr, (int)info.Type, ne0, ne1, numExperts,
                                byteCount, isExternalView: true, ownerToken: _gguf,
                                ownedBuffer: IntPtr.Zero);
                            mappedQuantBytes += byteCount;
                        }
                        else
                        {
                            // Non-mmap path: keep the bulk buffer alive as the
                            // owning storage, and make per-expert views into it
                            // instead of memcpy'ing into per-expert buffers. This
                            // lets us expose a stacked-experts view for free at
                            // the cost of an extra strong reference held by the
                            // stacked weight (no memory duplication).
                            IntPtr bulkPtr = QuantizedWeight.AllocateBuffer(byteCount);
                            _gguf.ReadTensorDataToNative(info, bulkPtr, byteCount);

                            var stacked = new StackedExpertWeights(
                                bulkPtr, (int)info.Type, ne0, ne1, numExperts,
                                byteCount, isExternalView: false, ownerToken: null,
                                ownedBuffer: bulkPtr);
                            _stackedExpertWeights[info.Name] = stacked;

                            for (int e = 0; e < numExperts; e++)
                            {
                                IntPtr expertPtr = new IntPtr(bulkPtr.ToInt64() + e * perExpertBytes);
                                string expertName = $"{baseName}.{e}.weight";
                                _quantWeights[expertName] = QuantizedWeight.CreateExternalView(
                                    expertPtr, perExpertBytes, (int)info.Type, ne0, ne1, stacked);
                                _stackedExpertMemberNames.Add(expertName);
                            }
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
            if (_backend == BackendType.Mlx)
            {
                PrepareMlxQuantizedWeightsForInference();
                return;
            }

            if (_backend == BackendType.Cuda)
            {
                PrepareDirectCudaQuantizedWeightsForInference();
                return;
            }

            if (_backend == BackendType.GgmlMetal)
            {
                PrepareGgmlMetalQuantizedWeightsForInference();
                return;
            }

            // GgmlCuda and GgmlVulkan share this path: the preload below goes through
            // the backend-agnostic GGML device buffer API (TSGgml_PreloadQuantizedWeight),
            // which gives both discrete-GPU backends device-resident weights.
            if ((_backend != BackendType.GgmlCuda && _backend != BackendType.GgmlVulkan) ||
                _cudaQuantWeightsPrepared || _quantWeights.Count == 0)
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

                // Skip weights the model serves device-resident by another route
                // (e.g. MoE per-expert split views that are covered by the stacked
                // expert device buffer). Preloading them here would put a second,
                // redundant copy of every expert byte in VRAM. The host view is
                // left intact so the stacked path / any per-op fallback can still
                // reach the bytes (and lazily upload on demand if ever needed).
                if (!ShouldPreloadCudaQuantWeightToDevice(weightName))
                    continue;

                // llama.cpp keeps token_embd on the host (its CPU_Mapped model
                // buffer): embedding lookup is a row gather, and when the quant
                // type has no device get_rows kernel Embedding() always serves it
                // from the retained host copy, so a device copy would be pure
                // VRAM waste (521 MB for Qwen3.6-27B's 248320x5120 Q3_K table).
                // Tied-output models matmul against token_embd through its device
                // cache key, so the skip requires a separate output.weight.
                if (string.Equals(weightName, "token_embd.weight", StringComparison.Ordinal)
                    && !CanUseGgmlQuantizedGetRows(qw.GgmlType)
                    && (_quantWeights.ContainsKey("output.weight") || _weights.ContainsKey("output.weight")))
                    continue;

                IntPtr cacheKey = qw.EnsureDeviceCacheKey();
                if (!GgmlBasicOps.PreloadQuantizedWeight(cacheKey, qw.Data, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes))
                {
                    // The device cannot hold this weight in a single backend buffer
                    // (e.g. ggml-vulkan's per-buffer maxBufferSize cap; WSL's dzn
                    // Vulkan layer caps it under 3 GB, below Gemma E4B's ~2.9 GB
                    // Q8_0 per_layer_token_embd). Keep the host copy and let the
                    // model's host-gather fallbacks serve it.
                    qw.MarkDevicePreloadTooLarge();
                    Console.WriteLine(
                        $"  {weightName}: {qw.RawBytes / 1024 / 1024} MB exceeds the {_backend} device's single-buffer limit; keeping host copy (device lookups fall back to host).");
                    continue;
                }
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
                Console.WriteLine($"  Device-resident quantized weights: {preloadedBytes / 1024 / 1024} MB across {preloadedCount} tensors");
        }

        private void PrepareMlxQuantizedWeightsForInference()
        {
            if (_mlxQuantWeightsPrepared || _quantWeights.Count == 0)
                return;

            if (_allocator is not MlxAllocator mlxAllocator)
                return;

            long fallbackBytes = MlxHostFallbackQuantizedBytes();
            long nativeBytes = MlxNativePreloadableQuantizedBytes();
            if (fallbackBytes > 0)
            {
                Console.WriteLine(
                    $"  MLX eager quantized preload: {nativeBytes / 1024 / 1024} MB native-capable weights will be device-resident; " +
                    $"{fallbackBytes / 1024 / 1024} MB fallback quantized weights remain file-backed.");
            }

            bool offloadEnabled = MoeExpertOffload.IsEnabled;
            long preloadedBytes = 0;
            int preloadedCount = 0;
            long deferredBytes = 0;
            int deferredCount = 0;
            long zeroCopyExpertBytes = 0;
            int zeroCopyExpertCount = 0;
            long fallbackExpertBytes = 0;
            int fallbackExpertCount = 0;
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

                // Skip weights the model serves device-resident by another route
                // (e.g. MoE per-expert views covered by a stacked-experts MLX
                // weight built for mlx_gather_qmm). Preloading them here would
                // put a second, redundant copy of every expert byte in unified
                // memory. The host view is left intact so the per-expert
                // fallback can still lazily upload on first use if the batched
                // path ever refuses at runtime.
                if (!ShouldPreloadMlxQuantWeightToDevice(weightName, qw))
                    continue;

                bool isExpert = offloadEnabled && MoeExpertOffload.IsExpertWeightName(weightName);
                bool canPreload = MlxQuantizedOps.CanPreloadQuantizedType(qw.GgmlType);
                bool preloadCopies = canPreload && MlxQuantizedOps.PreloadDuplicatesHostMemory(qw.GgmlType);

                if (isExpert && !canPreload)
                {
                    // Host-fallback expert (e.g. IQ1_S / IQ2_XS / IQ1_M in
                    // Nemotron's UD-IQ2_XXS): matmul runs the host-side
                    // dequant path and never enters the MLX cache. Track for
                    // accounting only.
                    IntPtr cacheKey = qw.EnsureDeviceCacheKey();
                    MoeExpertOffload.RegisterOffloadable(cacheKey);
                    if (qw.HasExternalHostView)
                        MoeExpertOffload.AdvisePagesNotNeeded(qw.Data, qw.RawBytes);
                    fallbackExpertBytes += qw.RawBytes;
                    fallbackExpertCount++;
                    continue;
                }

                if (isExpert && canPreload && preloadCopies)
                {
                    // Repack-kernel expert (Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 /
                    // MXFP4, or Q5_K with TS_MLX_Q5K_RAW=0). The MLX preload
                    // would allocate fresh MLX-managed memory and double the
                    // residency cost; offload bypasses that by deferring the
                    // upload to first use and bounding total residency via the
                    // LRU. This is where the offload mechanism produces the
                    // largest measured memory savings.
                    IntPtr cacheKey = qw.EnsureDeviceCacheKey();
                    MoeExpertOffload.RegisterOffloadable(cacheKey);
                    if (qw.HasExternalHostView)
                        MoeExpertOffload.AdvisePagesNotNeeded(qw.Data, qw.RawBytes);
                    deferredBytes += qw.RawBytes;
                    deferredCount++;
                    continue;
                }

                if (isExpert && canPreload && !preloadCopies)
                {
                    // Raw-wrap kernel expert (Q4_K / Q6_K, IQ2_XXS / IQ2_S /
                    // IQ3_S / IQ4_XS, or Q5_K when raw mode is enabled). The
                    // MLX preload does NOT allocate fresh memory — it just
                    // wraps the GGUF mmap pointer as an MLX array. The
                    // baseline preload path's qw.ReleaseHostData() call after
                    // upload already issues madvise(DONTNEED) on the mmap
                    // region, letting the OS evict page-cache pages between
                    // accesses. Routing these experts through the offload LRU
                    // instead would just churn MlxArray wrappers without any
                    // memory-residency win, and on Apple Silicon makes
                    // measured RSS WORSE because lazy wrappers prevent the
                    // OS from settling its page-cache eviction policy.
                    //
                    // → Fall through to the baseline-preload path below.
                    zeroCopyExpertBytes += qw.RawBytes;
                    zeroCopyExpertCount++;
                }

                if (!canPreload)
                    continue;

                IntPtr preloadKey = qw.EnsureDeviceCacheKey();
                MlxQuantizedOps.PreloadQuantizedWeight(
                    mlxAllocator,
                    preloadKey,
                    qw.Data,
                    qw.GgmlType,
                    qw.Ne0,
                    qw.Ne1,
                    qw.RawBytes);

                preloadedBytes += qw.RawBytes;
                preloadedCount++;

                // Repack quants (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/MXFP4/Q5_K-repack)
                // were materialised into a fresh MLX-allocator MTLBuffer in
                // the preload above. The original GGUF/host bytes are now
                // redundant — releasing them frees the source view and
                // (when external) lets the OS reclaim those mmap pages.
                //
                // Raw-wrap quants (Q4_K, Q6_K, IQ2_XXS, IQ2_S, IQ3_S,
                // IQ4_XS, IQ4_NL, Q5_K-raw) are wrapped zero-copy via
                // mlx_array_new_data_managed → MTLBuffer-with-bytes-no-copy
                // pointing at the GGUF mmap. They MUST keep that mmap
                // alive — calling ReleaseHostData here would (a) lose the
                // host pointer that MLX is reading from, (b) invoke
                // madvise(MADV_DONTNEED) on still-active model pages,
                // forcing the kernel to re-read them from disk on every
                // forward pass.
                bool wasMappedView = qw.HasExternalHostView;
                if (preloadCopies)
                {
                    qw.ReleaseHostData();
                    if (wasMappedView)
                        mappedHostViews--;
                }
            }

            // Stacked-experts views are lazily uploaded by the batched-MoE matmul
            // path (no explicit preload). Register them as offloadable so any
            // repack-kernel batched-MoE uploads are governed by the LRU. For
            // raw-wrap kernel stacked views (the common case — IQ2_XXS, Q4_K
            // etc.) the LRU does no harm because no MLX-allocator memory is
            // duplicated, and the registration is essentially a no-op there.
            if (offloadEnabled)
            {
                foreach (var stacked in _stackedExpertWeights.Values)
                    MoeExpertOffload.RegisterOffloadable(stacked.Data);
            }

            _mlxQuantWeightsPrepared = true;
            // Keep the GGUF mmap alive whenever any quantized weight still has a
            // file-backed view — both the existing fallback path (unpreloadable
            // types) AND the offload path (expert weights with retained host
            // pointers) need it to remain mapped.
            if (mappedHostViews == 0 && preloadedCount > 0)
                _gguf?.Dispose();
            else if (_gguf != null && string.Equals(
                Environment.GetEnvironmentVariable("TS_MLX_MLOCK_GGUF") ?? "1", "1", StringComparison.Ordinal))
            {
                // Pin the GGUF mmap region in physical RAM. Without this,
                // macOS treats file-backed pages as evictable and the kernel
                // throws model weights into the page cache between forward
                // passes — every subsequent layer page-faults them back from
                // disk and inference collapses to ~0.3 tok/s.
                //
                // mlx_set_wired_limit only governs MLX-allocator MTLBuffer
                // residency, not arbitrary mmap'd pages, so MTLBuffer-backed
                // zero-copy wrappers (CreateIq4XsRawWeight etc.) need this
                // explicit mlock too. Opt out via TS_MLX_MLOCK_GGUF=0.
                bool locked = _gguf.TryLockMappedRegion();
                if (locked)
                {
                    Console.WriteLine(
                        "  GGUF mmap pinned via mlock (model weights stay resident; set TS_MLX_MLOCK_GGUF=0 to disable).");
                }
                else
                {
                    Console.WriteLine(
                        $"  GGUF mlock failed (errno={_gguf.LastLockError}); inference may swap under memory pressure. " +
                        "Set TS_MLX_MLOCK_GGUF=0 to suppress this message.");
                }
            }

            if (preloadedCount > 0 || deferredCount > 0 || zeroCopyExpertCount > 0 || fallbackExpertCount > 0)
            {
                var snapshot = mlxAllocator.GetMemorySnapshot();
                Console.WriteLine(
                    $"  MLX resident quantized weights: {preloadedBytes / 1024 / 1024} MB across {preloadedCount} tensors " +
                    $"(active {snapshot.ActiveBytes / 1024 / 1024} MB, cache {snapshot.CacheBytes / 1024 / 1024} MB, peak {snapshot.PeakBytes / 1024 / 1024} MB)");
                if (deferredCount > 0 || zeroCopyExpertCount > 0 || fallbackExpertCount > 0)
                {
                    long capMb = MoeExpertOffload.MaxCacheBytes / 1024 / 1024;
                    long totalExpertMb = (deferredBytes + zeroCopyExpertBytes + fallbackExpertBytes) / 1024 / 1024;
                    int totalExpertCount = deferredCount + zeroCopyExpertCount + fallbackExpertCount;
                    Console.WriteLine(
                        $"  MoE expert weights detected: {totalExpertMb} MB across {totalExpertCount} tensors " +
                        $"(TS_MLX_EXPERT_OFFLOAD_MB={(offloadEnabled ? capMb.ToString() : "0")})");
                    if (deferredCount > 0)
                    {
                        Console.WriteLine(
                            $"    Offload-LRU: {deferredBytes / 1024 / 1024} MB / {deferredCount} tensors are " +
                            $"repack-kernel quants (LRU bounds MLX-allocator residency to ~{capMb} MB).");
                    }
                    if (zeroCopyExpertCount > 0)
                    {
                        Console.WriteLine(
                            $"    Zero-copy preload: {zeroCopyExpertBytes / 1024 / 1024} MB / {zeroCopyExpertCount} tensors are " +
                            $"raw-wrap kernel quants (no MLX allocator copy; baseline madvise upfront, OS page-cache evicts cold pages).");
                    }
                    if (fallbackExpertCount > 0)
                    {
                        Console.WriteLine(
                            $"    Host fallback: {fallbackExpertBytes / 1024 / 1024} MB / {fallbackExpertCount} tensors use " +
                            $"unpreloadable quant types (matmul runs via host-side dequant; OS page cache governs residency).");
                    }
                }
                MlxBackend.ClearCache();
            }
        }

        // GGML_METAL doesn't perform an eager device upload — weights are
        // wrapped as MTLBuffer pointers around the GGUF mmap via
        // ggml_backend_dev_buffer_from_host_ptr, so they already live in
        // unified memory at zero extra bytes. The wrapper itself, cached
        // in the native g_host_buffer_cache, can still keep Metal's claim
        // on those pages and prevent the OS from paging them out. When
        // TS_MLX_EXPERT_OFFLOAD_MB is set, we register expert host pointers
        // with the native cache so it LRU-bounds their MTLBuffer wrappers
        // and frees the oldest ones when the budget is exceeded.
        private void PrepareGgmlMetalQuantizedWeightsForInference()
        {
            if (_quantWeights.Count == 0)
                return;

            EnsureQuantBackendAvailable();

            if (!MoeExpertOffload.IsEnabled)
                return;

            long offloadedBytes = 0;
            int offloadedCount = 0;
            foreach (var kv in _quantWeights)
            {
                QuantizedWeight qw = kv.Value;
                if (!qw.HasHostData)
                    continue;
                if (!MoeExpertOffload.IsExpertWeightName(kv.Key))
                    continue;
                GgmlBasicOps.RegisterOffloadable(qw.Data);
                offloadedBytes += qw.RawBytes;
                offloadedCount++;
            }

            // The native MoE FFN kernels look up each expert weight via
            // try_get_cacheable_tensor_buffer keyed by `data` — the GGUF
            // mmap pointer. The stacked-experts view points at the SAME
            // bytes (its Data is the start of the 3D GGUF tensor, which is
            // also the first per-expert tile's address), so the per-expert
            // RegisterOffloadable above already covers it. We do not
            // register stacked.Data separately because doing so would
            // double-count the resident bytes.

            if (offloadedCount > 0)
            {
                GgmlBasicOps.SetOffloadableBudget(MoeExpertOffload.MaxCacheBytes);
                long capMb = MoeExpertOffload.MaxCacheBytes / 1024 / 1024;
                Console.WriteLine(
                    $"  GGML_METAL MoE expert offload: {offloadedBytes / 1024 / 1024} MB across {offloadedCount} tensors registered " +
                    $"(LRU cap {capMb} MB, set TS_MLX_EXPERT_OFFLOAD_MB=0 to disable)");
            }
        }

        /// <summary>Diagnostic (TS_CUDA_LOG_VRAM=1): logs dedicated-VRAM free/used at
        /// <paramref name="label"/> when the active allocator is the direct-CUDA one.</summary>
        internal static void LogCudaVram(IAllocator allocator, string label)
        {
            if (allocator is CudaAllocator cuda)
                cuda.LogVram(label);
        }

        /// <summary>Diagnostic (TS_CUDA_LOG_VRAM=1): logs the model allocator's
        /// dedicated-VRAM free/used at <paramref name="label"/>.</summary>
        public void LogVramSnapshot(string label) => LogCudaVram(_allocator, label);

        private void PrepareDirectCudaQuantizedWeightsForInference()
        {
            if (_cudaQuantWeightsPrepared || _quantWeights.Count == 0)
                return;

            if (_allocator is not CudaAllocator cudaAllocator)
                return;

            cudaAllocator.LogVram("before direct-CUDA quant weight preload");

            // When CUDA kernels are unavailable (PTX load failed), device-side
            // quantized matmul/embedding will fail and every op falls back to
            // the CPU dequant path.  Keep all host data alive in that case.
            bool kernelsAvailable = CudaQuantizedOps.AreKernelsAvailable(cudaAllocator);

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
                var qw = kv.Value;
                if (!qw.HasHostData || !CudaQuantizedOps.SupportsQuantizedType(qw.GgmlType))
                    continue;

                IntPtr cacheKey = qw.EnsureDeviceCacheKey();
                CudaQuantizedOps.PreloadQuantizedWeight(
                    cudaAllocator,
                    cacheKey,
                    qw.Data,
                    qw.GgmlType,
                    qw.Ne0,
                    qw.Ne1,
                    qw.RawBytes);
                preloadedBytes += qw.RawBytes;
                preloadedCount++;

                bool wasMappedView = qw.HasExternalHostView;
                if (kernelsAvailable && !ShouldRetainCudaHostQuantWeight(kv.Key))
                {
                    qw.ReleaseHostData();
                    if (wasMappedView)
                        mappedHostViews--;
                }
            }

            _cudaQuantWeightsPrepared = true;
            if (mappedHostViews == 0)
                _gguf?.Dispose();

            if (preloadedCount > 0)
                Console.WriteLine($"  Direct CUDA resident quantized weights: {preloadedBytes / 1024 / 1024} MB across {preloadedCount} tensors (host copies released)");

            cudaAllocator.LogVram("after direct-CUDA quant weight preload");
        }

        // TS_GGML_RETAIN_HOST_WEIGHTS=1 keeps every quantized weight's host copy
        // alive after the device preload instead of releasing it. Costs the model's
        // full host footprint in RAM; diagnostic/workaround knob for any native
        // path that still reads weight bytes through the original host pointer
        // after preload (symptom: memcpy access violation on first forward).
        private static readonly bool s_retainAllHostQuantWeights =
            Environment.GetEnvironmentVariable("TS_GGML_RETAIN_HOST_WEIGHTS") == "1";

        private static bool ShouldRetainCudaHostQuantWeight(string weightName)
        {
            return s_retainAllHostQuantWeights ||
                string.Equals(weightName, "token_embd.weight", StringComparison.Ordinal) ||
                string.Equals(weightName, "per_layer_token_embd.weight", StringComparison.Ordinal);
        }

        /// <summary>
        /// Whether <paramref name="weightName"/> should get its own device-resident
        /// copy during <see cref="PrepareCudaQuantizedWeightsForInference"/> (the
        /// <c>ggml_cuda</c> backend). Defaults to true. Models whose CUDA decode and
        /// prefill paths serve MoE experts exclusively through the stacked-expert
        /// device buffer override this to return false for the per-expert split
        /// views, avoiding a second full copy of the experts in VRAM.
        ///
        /// Overrides MUST keep the <see cref="MoeCpuOffloadConfig"/> term: a routed
        /// expert belonging to a <c>--n-cpu-moe</c> layer is multiplied on the host
        /// and uploading it would spend exactly the VRAM the flag exists to save.
        /// </summary>
        protected virtual bool ShouldPreloadCudaQuantWeightToDevice(string weightName)
            => !MoeCpuOffloadConfig.IsOffloadedExpertWeightName(weightName);

        /// <summary>
        /// Per-weight veto for the eager MLX quantized preload
        /// (<see cref="PrepareMlxQuantizedWeightsForInference"/>). Models whose
        /// MLX forward serves a weight device-resident by another route (e.g.
        /// routed experts through a stacked-experts <c>mlx_gather_qmm</c>
        /// weight) override this to return false for those names, avoiding a
        /// second full copy of the bytes in unified memory. Skipped weights
        /// keep their host data, so any per-op fallback still lazily uploads
        /// them on first use.
        /// </summary>
        protected virtual bool ShouldPreloadMlxQuantWeightToDevice(string weightName, QuantizedWeight weight)
            => true;

        protected bool CanUseGgmlQuantizedGetRows(int ggmlType)
        {
            if (!IsGgmlBackend)
                return false;

            if (_backend != BackendType.GgmlCuda)
                return true;

            // ggml-cuda's get_rows kernel only implements the legacy round-number
            // quant types (see ExternalProjects/ggml/src/ggml-cuda/getrows.cu:
            // ggml_cuda_get_rows_switch_src0_type). k-quants such as Q6_K are NOT
            // supported and abort at runtime, so they must fall back to the host
            // dequant path (PopulateQuantizedRows). Keep this list in sync with the
            // upstream kernel's supported src0 types.
            return ((GgmlTensorType)ggmlType) switch
            {
                GgmlTensorType.Q4_0 => true,
                GgmlTensorType.Q4_1 => true,
                GgmlTensorType.Q5_0 => true,
                GgmlTensorType.Q5_1 => true,
                GgmlTensorType.Q8_0 => true,
                _ => false,
            };
        }

        protected bool TryCreateFusedQuantizedWeight(out QuantizedWeight fused, params QuantizedWeight[] weights)
        {
            if (CanUseFileMappedQuantizedWeights && QuantizedWeight.TryCreateConcatenatedView(out fused, weights))
                return true;

            fused = QuantizedWeight.ConcatOrCreateCopy(weights);
            return true;
        }

        protected bool HasMlxHostFallbackQuantizedWeights()
        {
            if (_backend != BackendType.Mlx)
                return false;

            foreach (QuantizedWeight weight in _quantWeights.Values)
            {
                if (!MlxQuantizedOps.CanPreloadQuantizedType(weight.GgmlType))
                    return true;
            }

            return false;
        }

        protected long MlxHostFallbackQuantizedBytes()
        {
            if (_backend != BackendType.Mlx)
                return 0;

            long bytes = 0;
            foreach (QuantizedWeight weight in _quantWeights.Values)
            {
                if (!MlxQuantizedOps.CanPreloadQuantizedType(weight.GgmlType))
                    bytes += weight.RawBytes;
            }

            return bytes;
        }

        protected long MlxNativePreloadableQuantizedBytes()
        {
            if (_backend != BackendType.Mlx)
                return 0;

            long bytes = 0;
            foreach (QuantizedWeight weight in _quantWeights.Values)
            {
                if (MlxQuantizedOps.CanPreloadQuantizedType(weight.GgmlType))
                    bytes += weight.RawBytes;
            }

            return bytes;
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

        protected unsafe void FuseGateUpWeights(int numLayers = 0)
        {
            if (numLayers <= 0)
                numLayers = Config.NumLayers;
            int fused = 0;
            int requantized = 0;
            for (int l = 0; l < numLayers; l++)
            {
                string gateName = $"blk.{l}.ffn_gate.weight";
                string upName = $"blk.{l}.ffn_up.weight";
                string guName = $"blk.{l}.ffn_gate_up.weight";

                if (_quantWeights.TryGetValue(gateName, out var gw) &&
                    _quantWeights.TryGetValue(upName, out var uw) &&
                    gw.Ne0 == uw.Ne0)
                {
                    // Mixed-quant "UD"/dynamic GGUFs (e.g. Qwen3.8 UD quants, where
                    // ffn_gate is IQ4_XS but ffn_up is Q5_K) store gate and up in
                    // different types, which a single fused tensor can't represent.
                    // Requantize the lower-fidelity side into the higher-fidelity
                    // type first, then fuse as usual.
                    QuantizedWeight gateSrc = gw, upSrc = uw, requant = null;
                    if (gw.GgmlType != uw.GgmlType)
                    {
                        requant = TryRequantizeForFusion(gw, uw, out bool requantIsGate);
                        if (requant == null)
                        {
                            Console.WriteLine(
                                $"  WARNING: layer {l} ffn_gate ({(Runtime.GgmlTensorType)(uint)gw.GgmlType}) and ffn_up " +
                                $"({(Runtime.GgmlTensorType)(uint)uw.GgmlType}) quant types differ and requantization is " +
                                "unavailable; gate/up left unfused.");
                            continue;
                        }
                        if (requantIsGate) gateSrc = requant; else upSrc = requant;
                        requantized++;
                    }

                    // Gate-up fusion must always succeed: model FFN code expects
                    // a single fused tensor at guName. If MLX view-fusion fails
                    // (gate/up not contiguous in the GGUF file), fall back to a
                    // copy. Cost is bounded — 2 tensors × per-layer, host memory
                    // released after the MLX device upload.
                    if (!TryCreateFusedQuantizedWeight(out QuantizedWeight fusedWeight, gateSrc, upSrc))
                        fusedWeight = QuantizedWeight.ConcatOrCreateCopy(gateSrc, upSrc);

                    _quantWeights[guName] = fusedWeight;
                    _quantWeights.Remove(gateName); gw.Dispose();
                    _quantWeights.Remove(upName); uw.Dispose();
                    if (requant != null && !ReferenceEquals(requant, fusedWeight))
                        requant.Dispose();
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
                Console.WriteLine(requantized > 0
                    ? $"  Fused projections: {fused} Gate+Up ({requantized} mixed-quant layers requantized to a common type)"
                    : $"  Fused projections: {fused} Gate+Up");
        }

        /// <summary>
        /// Produce a copy of the lower-fidelity side of a mixed-type gate/up pair,
        /// requantized to the other side's type so the pair can be fused. Prefers
        /// upcasting (smaller row size → larger); tries the opposite direction when
        /// the preferred target can't be produced without an importance matrix.
        /// Returns null when neither direction is possible.
        /// </summary>
        private unsafe QuantizedWeight TryRequantizeForFusion(QuantizedWeight gw, QuantizedWeight uw, out bool requantIsGate)
        {
            requantIsGate = false;
            if (!gw.HasHostData || !uw.HasHostData || gw.Ne0 != uw.Ne0)
                return null;

            long gRow = NativeDequant.RowSize(gw.GgmlType, gw.Ne0);
            long uRow = NativeDequant.RowSize(uw.GgmlType, uw.Ne0);
            QuantizedWeight lower = gRow <= uRow ? gw : uw;
            QuantizedWeight higher = gRow <= uRow ? uw : gw;

            QuantizedWeight result = TryRequantizeWeight(lower, higher.GgmlType);
            if (result != null)
            {
                requantIsGate = ReferenceEquals(lower, gw);
                return result;
            }

            result = TryRequantizeWeight(higher, lower.GgmlType);
            if (result != null)
            {
                requantIsGate = ReferenceEquals(higher, gw);
                return result;
            }

            return null;
        }

        /// <summary>
        /// Dequantize a weight row-chunk-wise to FP32 and requantize it into
        /// <paramref name="targetType"/>. Returns null when the conversion is not
        /// possible (imatrix-only target, or no native quantize available).
        /// </summary>
        private unsafe QuantizedWeight TryRequantizeWeight(QuantizedWeight src, int targetType)
        {
            try
            {
                long ne0 = src.Ne0, ne1 = src.Ne1;
                long srcRow = NativeDequant.RowSize(src.GgmlType, ne0);
                long dstRow = NativeDequant.RowSize(targetType, ne0);
                byte[] dstBuf = new byte[checked(dstRow * ne1)];
                const int ChunkRows = 512;
                int numChunks = (int)((ne1 + ChunkRows - 1) / ChunkRows);
                IntPtr srcBase = src.Data;
                bool failed = false;
                GCHandle hDst = GCHandle.Alloc(dstBuf, GCHandleType.Pinned);
                try
                {
                    IntPtr dstBase = hDst.AddrOfPinnedObject();
                    Parallel.For(0, numChunks,
                        () => new float[(long)ChunkRows * ne0],
                        (ci, state, f32) =>
                        {
                            if (Volatile.Read(ref failed))
                            {
                                state.Stop();
                                return f32;
                            }
                            long r = (long)ci * ChunkRows;
                            long rows = Math.Min(ChunkRows, ne1 - r);
                            fixed (float* pF32 = f32)
                            {
                                NativeDequant.DequantizeToFloat32Native(src.GgmlType,
                                    (IntPtr)((byte*)srcBase.ToPointer() + r * srcRow), (IntPtr)pF32, rows * ne0);
                                long written = GgmlGgufTensorDequant.QuantizeFloat32RowsOrZero(targetType,
                                    (IntPtr)pF32, (IntPtr)((byte*)dstBase.ToPointer() + r * dstRow), rows, ne0);
                                if (written != rows * dstRow)
                                {
                                    Volatile.Write(ref failed, true);
                                    state.Stop();
                                }
                            }
                            return f32;
                        },
                        _ => { });
                }
                finally
                {
                    hDst.Free();
                }

                if (failed)
                    return null;

                return new QuantizedWeight(dstBuf, targetType, ne0, ne1);
            }
            catch (Exception ex) when (IsRequantizeUnavailable(ex))
            {
                return null;
            }
        }

        private static bool IsRequantizeUnavailable(Exception ex)
        {
            if (ex is AggregateException agg)
                return agg.InnerExceptions.Count > 0 && agg.InnerExceptions.All(IsRequantizeUnavailable);
            return ex is DllNotFoundException or EntryPointNotFoundException or NotSupportedException;
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

        /// <summary>
        /// Create an int tensor on a specific GPU. Use this (with the consuming
        /// tensor's allocator) for anything read by a per-rank TP kernel — e.g.
        /// RoPE position tensors — so a rank-r kernel doesn't read a GPU-0 tensor
        /// across GPUs (illegal access without peer / wrong data with peer).
        /// </summary>
        protected static Tensor CreateIntTensorOn(IAllocator allocator, int[] data, params long[] sizes)
        {
            var tensor = new Tensor(allocator, DType.Int32, sizes);
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
                    // kernel is not implemented upstream, and for a table too large to
                    // be device-resident (DevicePreloadTooLarge).
                    if ((tokens.Length == 1 || !canUseGgmlLookup || qw.DevicePreloadTooLarge) && qw.HasHostData)
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
                result = new Tensor(_allocator, DType.Float32, seqLenF32, outDimF32);
                if (!TryGgmlF32LinearResident(result, input, w))
                {
                    using var wT = w.Transpose();
                    Ops.Addmm(result, 0, result, 1.0f, input, wT);
                }
            }
            else
            {
                return null;
            }

            _linearTicks += Stopwatch.GetTimestamp() - t0;
            return result;
        }

        /// <summary>
        /// GGML linear against an F32 weight, routed through the quantized entry
        /// point so the weight goes through the per-rank cacheable-buffer cache
        /// and becomes DEVICE-RESIDENT.
        ///
        /// The generic <see cref="Ops.Addmm"/> path has no weight cache: it
        /// re-binds and re-uploads the whole weight on every call. For a matmul
        /// that runs once per layer per token — an MoE router, say — that is the
        /// dominant cost of the layer, not the arithmetic. Measured on
        /// Qwen3.5-35B-A3B under --tp 2: the 40 F32 routers were 12.8 s of a
        /// 20.2 s decode, ~4 ms each to push 2 MB over PCIe for a
        /// [1,2048]x[2048,256] product.
        ///
        /// Requires a row-contiguous 2D weight, which is the GGUF layout
        /// ([outDim][inDim] row-major == ggml ne0=inDim, ne1=outDim). Returns
        /// false for anything else so the caller keeps the generic path.
        /// </summary>
        private static readonly bool GgmlF32ResidentLinearEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_GGML_F32_RESIDENT"), "0", StringComparison.Ordinal);

        protected unsafe bool TryGgmlF32LinearResident(Tensor result, Tensor input, Tensor w)
        {
            if (!GgmlF32ResidentLinearEnabled)
                return false;
            if (!IsGgmlBackend || w == null || w.DimensionCount != 2 || !w.IsContiguous()
                || w.ElementType != DType.Float32 || input.ElementType != DType.Float32)
                return false;

            long inDim = w.Sizes[1];
            long outDim = w.Sizes[0];
            if (input.DimensionCount != 2 || input.Sizes[1] != inDim)
                return false;

            IntPtr data = (IntPtr)GetFloatPtr(w);
            if (data == IntPtr.Zero)
                return false;

            try
            {
                GgmlBasicOps.AddmmQuant(result, input, data,
                    0 /* GGML_TYPE_F32 */, inDim, outDim, inDim * outDim * sizeof(float));
                return true;
            }
            catch (NotSupportedException)
            {
                return false;
            }
            catch (ArgumentException)
            {
                return false;
            }
        }

        private unsafe Tensor EmbeddingManagedQuantized(int[] tokens, QuantizedWeight weight)
        {
            int dim = (int)weight.Ne0;
            if (_backend == BackendType.Cuda)
            {
                var resultCuda = new Tensor(_allocator, DType.Float32, tokens.Length, dim);
                using var indicesCuda = CreateIntTensor(tokens, tokens.Length);
                if (CudaQuantizedOps.TryGetRowsQuantizedToFloat32(
                    resultCuda,
                    weight.EnsureDeviceCacheKey(),
                    weight.Data,
                    weight.GgmlType,
                    weight.Ne0,
                    weight.Ne1,
                    weight.RawBytes,
                    indicesCuda))
                {
                    return resultCuda;
                }

                resultCuda.Dispose();
            }

            if (_backend == BackendType.Mlx)
            {
                var resultMlx = new Tensor(_allocator, DType.Float32, tokens.Length, dim);
                using var indicesMlx = CreateIntTensor(tokens, tokens.Length);
                if (MlxQuantizedOps.TryGetRowsQuantizedToFloat32(
                    resultMlx,
                    weight.EnsureDeviceCacheKey(),
                    weight.Data,
                    weight.GgmlType,
                    weight.Ne0,
                    weight.Ne1,
                    weight.RawBytes,
                    indicesMlx))
                {
                    return resultMlx;
                }

                resultMlx.Dispose();
            }

            if (!weight.HasHostData)
                throw new InvalidOperationException($"Quantized embedding weight type {(GgmlTensorType)weight.GgmlType} is not available on the selected device and its host copy has been released.");

            long rowBytes = NativeDequant.RowSize(weight.GgmlType, weight.Ne0);
            var result = new Tensor(_allocator, DType.Float32, tokens.Length, dim);
            float* dst = GetFloatPtr(result);
            byte* basePtr = (byte*)weight.Data.ToPointer();

            for (int i = 0; i < tokens.Length; i++)
            {
                byte* rowPtr = basePtr + (long)tokens[i] * rowBytes;
                NativeDequant.DequantizeToFloat32Native(
                    weight.GgmlType,
                    (IntPtr)rowPtr,
                    (IntPtr)(dst + (long)i * dim),
                    dim);
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

            if (_backend == BackendType.Cuda &&
                CudaQuantizedOps.TryAddmmQuantizedToFloat32(
                    result,
                    input,
                    weight.EnsureDeviceCacheKey(),
                    weight.Data,
                    weight.GgmlType,
                    weight.Ne0,
                    weight.Ne1,
                    weight.RawBytes))
            {
                return;
            }

            if (_backend == BackendType.Mlx &&
                MlxQuantizedOps.TryAddmmQuantizedToFloat32(
                    result,
                    input,
                    weight.EnsureDeviceCacheKey(),
                    weight.Data,
                    weight.GgmlType,
                    weight.Ne0,
                    weight.Ne1,
                    weight.RawBytes))
            {
                return;
            }

            // GGML backends: dispatch to the device. Without this the tensor-
            // parallel path — which reaches this helper for every sharded
            // projection — would drop to the managed dequant loop below and run
            // the whole model on the CPU.
            if (IsGgmlBackend && weight.HasHostData)
            {
                // CacheKey, not Data: the GGML device cache is keyed by it, and
                // after a preload it is an opaque handle rather than the host
                // pointer. Passing Data would miss the resident copy and re-upload
                // the weight on every call.
                GgmlBasicOps.AddmmQuant(result, input, weight.CacheKey, weight.GgmlType, weight.Ne0, weight.Ne1, weight.RawBytes);
                return;
            }

            if (!weight.HasHostData)
                throw new InvalidOperationException($"Quantized linear weight type {(GgmlTensorType)weight.GgmlType} is not available on the selected device and its host copy has been released.");

            // One managed implementation, in ManagedQuantizedOps: integer dot
            // kernels where the weight type has them, dequant-once + register-
            // blocked float dots otherwise.
            ManagedQuantizedOps.AddmmQuantizedToFloat32(
                weight.GgmlType,
                weight.Data,
                weight.Ne0,
                weight.Ne1,
                GetFloatPtr(input),
                inDim,
                seqLen,
                GetFloatPtr(result),
                outDim);
            InvalidateTensorDeviceCache(result);
        }

        #region SIMD Helpers

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector<float> LdVec(float* p) =>
            TensorComputePrimitives.LoadVector(p);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void StVec(float* p, Vector<float> v) =>
            TensorComputePrimitives.StoreVector(p, v);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected static unsafe float VecDot(float* a, float* b, int n) =>
            TensorComputePrimitives.Dot(a, b, n);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected static unsafe float VecSumSq(float* a, int n) =>
            TensorComputePrimitives.SumSquares(a, n);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected static unsafe void VecScale(float* data, float scale, int n) =>
            TensorComputePrimitives.Scale(data, scale, n);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected static unsafe void VecScaleAdd(float* dst, float* src, float w, int n) =>
            TensorComputePrimitives.ScaleAdd(dst, src, w, n);

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
            out float r0, out float r1, out float r2, out float r3) =>
            TensorComputePrimitives.Dot4(a0, a1, a2, a3, b, n, out r0, out r1, out r2, out r3);

        /// <summary>
        /// Batched scale-add: simultaneously update four destination vectors with the
        /// same source <paramref name="src"/> scaled by four independent weights. The
        /// hot loop loads each src element exactly once into a register and broadcasts
        /// it to four FMA-style updates, which is the V-aggregation analog of VecDot4.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected static unsafe void VecScaleAdd4(float* d0, float* d1, float* d2, float* d3,
            float* src, float w0, float w1, float w2, float w3, int n) =>
            TensorComputePrimitives.ScaleAdd4(d0, d1, d2, d3, src, w0, w1, w2, w3, n);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected static unsafe void VecSubScale(float* dst, float* a, float* b, float scale, int n) =>
            TensorComputePrimitives.SubScale(dst, a, b, scale, n);

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        protected static unsafe void VecZero(float* data, int n) =>
            TensorComputePrimitives.Zero(data, n);

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

        /// <summary>
        /// Dense SwiGLU FFN block (pre-norm + gate/up + SiLU·mul + down + residual add)
        /// collapsed into a single GGML graph dispatch via <c>FusedFFNSwiGLUQuant</c>:
        ///   residual += down_W^T @ ( silu(gate) * up ),  [gate|up] = gateUp_W^T @ rmsnorm(residual, normW)
        ///
        /// On the GGML CUDA backend each op keeps its tensors in host memory and
        /// uploads inputs / downloads outputs across PCIe per dispatch, so the
        /// unfused chain ping-pongs the large [tokens, 2·intermediate] activation
        /// (e.g. 114 MB for a 1024-token chunk) host↔device three times per layer.
        /// Fusing keeps that intermediate resident on the device; only the small
        /// [tokens, hidden] residual crosses the bus. This is the dominant prefill
        /// cost in the batched paths (matches the legacy per-sequence fast path in
        /// Qwen35Model.FFNCachedFused).
        ///
        /// Returns false — leaving <paramref name="residual"/> untouched — when the
        /// backend, weight quantization, or layout does not qualify; callers must
        /// then run the unfused norm+FFN+add chain.
        /// </summary>
        // A/B switch: TS_DISABLE_FUSED_DENSE_FFN=1 forces the unfused norm+FFN+add
        // chain so the fused vs unfused paths can be compared on the same build.
        private static readonly bool _disableFusedDenseFFN =
            Environment.GetEnvironmentVariable("TS_DISABLE_FUSED_DENSE_FFN") is string s
            && (s == "1" || string.Equals(s, "true", StringComparison.OrdinalIgnoreCase));

        protected bool TryFusedDenseSwiGLUFFNInto(
            Tensor residual, string normWeightName, string gateUpWeightName, string downWeightName)
        {
            if (_disableFusedDenseFFN)
                return false;
            if (!IsGgmlBackend || residual == null || residual.DimensionCount != 2)
                return false;
            if (!_quantWeights.TryGetValue(gateUpWeightName, out var gateUpQW) || gateUpQW == null)
                return false;
            if (!_quantWeights.TryGetValue(downWeightName, out var downQW) || downQW == null)
                return false;
            if (!_weights.TryGetValue(normWeightName, out var normW) || normW == null)
                return false;

            int intermSize = Config.IntermediateSize;
            int halfDim = intermSize > 0 ? intermSize : (int)(gateUpQW.Ne1 / 2);
            long hidden = residual.Sizes[1];
            if (halfDim <= 0
                || gateUpQW.Ne1 != 2L * halfDim
                || gateUpQW.Ne0 != hidden
                || downQW.Ne0 != halfDim
                || downQW.Ne1 != hidden)
                return false;

            long t0 = Stopwatch.GetTimestamp();
            GgmlBasicOps.FusedFFNSwiGLUQuant(residual, residual, normW, Config.Eps,
                gateUpQW.CacheKey, gateUpQW.GgmlType, gateUpQW.Ne0, gateUpQW.Ne1, gateUpQW.RawBytes,
                downQW.CacheKey, downQW.GgmlType, downQW.Ne0, downQW.Ne1, downQW.RawBytes,
                halfDim);
            _linearTicks += Stopwatch.GetTimestamp() - t0;
            return true;
        }

        /// <summary>
        /// Fused dense FFN <em>projection</em> (pre-norm + gate/up + activation·mul + down)
        /// in one GGML graph, returning the FFN output instead of folding it into a
        /// residual. For models that apply a post-FFN norm to the output before the
        /// residual add (Gemma 4's <c>post_ffw_norm</c>), the caller runs that norm + add
        /// on the small returned tensor while the large [tokens, 2·intermediate] gate_up
        /// intermediate stays resident on the device — the dominant batched/legacy prefill
        /// cost on GGML CUDA. <paramref name="actType"/>: 0 = SiLU (SwiGLU), 1 = GELU tanh
        /// (GeGLU). The fused rms_norm uses the same loaded weight as <see cref="RMSNormOp"/>,
        /// so the result is numerically identical to the unfused chain.
        ///
        /// Returns the FFN output, or null (caller must run the unfused path) when the
        /// backend, quantization, or layout does not qualify.
        /// </summary>
        protected Tensor TryFusedDenseFFNProject(
            Tensor input, string normWeightName, string gateUpWeightName, string downWeightName, int actType)
        {
            if (_disableFusedDenseFFN)
                return null;
            if (!IsGgmlBackend || input == null || input.DimensionCount != 2)
                return null;
            if (!_quantWeights.TryGetValue(gateUpWeightName, out var gateUpQW) || gateUpQW == null)
                return null;
            if (!_quantWeights.TryGetValue(downWeightName, out var downQW) || downQW == null)
                return null;
            if (!_weights.TryGetValue(normWeightName, out var normW) || normW == null)
                return null;

            int intermSize = Config.IntermediateSize;
            int halfDim = intermSize > 0 ? intermSize : (int)(gateUpQW.Ne1 / 2);
            long hidden = input.Sizes[1];
            if (halfDim <= 0
                || gateUpQW.Ne1 != 2L * halfDim
                || gateUpQW.Ne0 != hidden
                || downQW.Ne0 != halfDim
                || downQW.Ne1 != hidden)
                return null;

            long t0 = Stopwatch.GetTimestamp();
            var output = new Tensor(_allocator, DType.Float32, input.Sizes[0], hidden);
            GgmlBasicOps.FusedFFNActProjectQuant(output, input, normW, Config.Eps,
                gateUpQW.CacheKey, gateUpQW.GgmlType, gateUpQW.Ne0, gateUpQW.Ne1, gateUpQW.RawBytes,
                downQW.CacheKey, downQW.GgmlType, downQW.Ne0, downQW.Ne1, downQW.RawBytes,
                halfDim, actType);
            _linearTicks += Stopwatch.GetTimestamp() - t0;
            return output;
        }

        #region Tensor Parallelism

        /// <summary>
        /// Shard linear weights across GPUs for tensor parallelism. Call after
        /// <see cref="LoadWeights"/> and any weight fusion (e.g. FuseQKVWeights).
        ///
        /// Column-parallel weights (QKV, gate_up) are split along the output
        /// dimension (ne1 / Sizes[0]): each GPU gets outDim/tp consecutive rows.
        /// Row-parallel weights (output, down) are split along the input
        /// dimension (ne0 / Sizes[1]): each GPU gets inDim/tp columns.
        ///
        /// Replicated weights (norms, embeddings) are not sharded — every GPU
        /// keeps the full copy via the existing <see cref="_weights"/> dictionary.
        /// </summary>
        protected void ShardWeightsForTensorParallelism(
            IEnumerable<string> columnParallelPatterns,
            IEnumerable<string> rowParallelPatterns)
        {
            if (!IsTensorParallel) return;

            int tp = TpDegree;
            int globalTp = GlobalTpDegree;
            int rankOffset = TpRankOffset;
            var colPatterns = new List<string>(columnParallelPatterns);
            var rowPatterns = new List<string>(rowParallelPatterns);

            if (tp > 1 || globalTp > 1)
                Console.WriteLine($"  ShardWeightsForTP: tp={tp} globalTp={globalTp} rankOffset={rankOffset} colPatterns=[{string.Join(",", colPatterns)}] rowPatterns=[{string.Join(",", rowPatterns)}]");

            // Shard quantized weights.
            var quantToRemove = new List<string>();
            var colParallelOwners = new List<QuantizedWeight>();
            foreach (var kv in _quantWeights)
            {
                string name = kv.Key;
                bool isCol = colPatterns.Any(p => name.Contains(p));
                bool isRow = !isCol && rowPatterns.Any(p => name.Contains(p));
                if (!isCol && !isRow) continue;

                var qw = kv.Value;
                var shards = new QuantizedWeight[tp];

                if (isCol)
                {
                    // Split along ne1 (output dim): consecutive rows.
                    long rowsPerShard = qw.Ne1 / globalTp;
                    long rowBytes = NativeDequant.RowSize(qw.GgmlType, qw.Ne0);
                    long bytesPerShard = rowsPerShard * rowBytes;

                    for (int lr = 0; lr < tp; lr++)
                    {
                        int globalRank = rankOffset + lr;
                        IntPtr shardPtr = new IntPtr((long)qw.Data + globalRank * bytesPerShard);
                        shards[lr] = QuantizedWeight.CreateExternalView(
                            shardPtr, bytesPerShard, qw.GgmlType, qw.Ne0, rowsPerShard, qw);
                    }

                    // Keep the original owner alive: column-parallel shards
                    // are external views into its buffer. Disposing it here
                    // would leave the shards with dangling pointers.
                    colParallelOwners.Add(qw);
                }
                else
                {
                    // Split along ne0 (input dim): extract block-aligned columns per row.
                    var type = (GgmlTensorType)qw.GgmlType;
                    long blockSize = GgufFile.GetBlockSize(type);
                    long typeSize = GgufFile.GetTypeSize(type);
                    long blocksPerRow = qw.Ne0 / blockSize;
                    long blocksPerShard = blocksPerRow / globalTp;
                    long ne0PerShard = blocksPerShard * blockSize;
                    long srcRowBytes = NativeDequant.RowSize(qw.GgmlType, qw.Ne0);
                    long dstRowBytes = (ne0PerShard / blockSize) * typeSize;
                    long totalBytesPerShard = qw.Ne1 * dstRowBytes;
                    long blockBytesPerShard = blocksPerShard * typeSize;

                    if (quantToRemove.Count < 2)
                        Console.WriteLine($"    Row-parallel quant debug: {name} origNe0={qw.Ne0} origNe1={qw.Ne1} globalTp={globalTp} blockSize={blockSize} blocksPerRow={blocksPerRow} blocksPerShard={blocksPerShard} ne0PerShard={ne0PerShard} totalBytesPerShard={totalBytesPerShard}");


                    // Extract every rank's slice concurrently: each rank writes
                    // its own buffer, and when the source is the memory-mapped
                    // GGUF the parallel ranks fault in disjoint file regions at
                    // once — on slow or network-backed storage the page-fault
                    // reads, not the memcpy, are what this loop actually costs.
                    Parallel.For(0, tp, lr =>
                    {
                        int globalRank = rankOffset + lr;
                        IntPtr shardPtr = QuantizedWeight.AllocateBuffer(totalBytesPerShard);
                        unsafe
                        {
                            byte* src = (byte*)qw.Data.ToPointer();
                            byte* dst = (byte*)shardPtr.ToPointer();
                            long srcBlockOffset = globalRank * blocksPerShard * typeSize;
                            for (long row = 0; row < qw.Ne1; row++)
                            {
                                Buffer.MemoryCopy(
                                    src + row * srcRowBytes + srcBlockOffset,
                                    dst + row * dstRowBytes,
                                    dstRowBytes,
                                    blockBytesPerShard);
                            }
                        }
                        shards[lr] = new QuantizedWeight(shardPtr, totalBytesPerShard,
                            qw.GgmlType, ne0PerShard, qw.Ne1);
                    });
                }

                _tpQuantWeights[name] = shards;
                quantToRemove.Add(name);
            }

            // Remove original unsharded quantized weights that have been sharded.
            // Column-parallel owners are kept alive because their shards are
            // external views into the original buffer; only row-parallel
            // originals (whose shards own independent copies) are disposed.
            foreach (var name in quantToRemove)
            {
                var qw = _quantWeights[name];
                if (!colParallelOwners.Contains(qw))
                    qw.Dispose();
                _quantWeights.Remove(name);
            }

            // Shard F32 weights.
            var f32ToRemove = new List<string>();
            foreach (var kv in _weights)
            {
                string name = kv.Key;
                bool isCol = colPatterns.Any(p => name.Contains(p));
                bool isRow = !isCol && rowPatterns.Any(p => name.Contains(p));
                if (!isCol && !isRow) continue;

                var w = kv.Value;
                var shards = new Tensor[tp];

                if (isCol)
                {
                    // Split along dim 0 (output dim).
                    long shardSize = w.Sizes[0] / globalTp;
                    for (int lr = 0; lr < tp; lr++)
                    {
                        int globalRank = rankOffset + lr;
                        var view = w.Narrow(0, globalRank * shardSize, shardSize);
                        shards[lr] = Ops.NewContiguous(view);
                        view.Dispose();
                    }
                }
                else
                {
                    // Split along dim 1 (input dim).
                    long shardSize = w.Sizes[1] / globalTp;
                    for (int lr = 0; lr < tp; lr++)
                    {
                        int globalRank = rankOffset + lr;
                        var view = w.Narrow(1, globalRank * shardSize, shardSize);
                        shards[lr] = Ops.NewContiguous(view);
                        view.Dispose();
                    }
                }

                _tpWeights[name] = shards;
                f32ToRemove.Add(name);
            }

            foreach (var name in f32ToRemove)
            {
                _weights[name].Dispose();
                _weights.Remove(name);
            }

            Console.WriteLine($"  TP sharded: {quantToRemove.Count} quantized + {f32ToRemove.Count} F32 weights across {tp} GPUs.");
        }

        /// <summary>
        /// Output dimension (row count) of a possibly-quantized weight, or 0 if
        /// the weight is not present in either weight dictionary.
        /// </summary>
        protected int GetFusedOutputDim(string weightName)
        {
            if (_quantWeights.TryGetValue(weightName, out var qw))
                return (int)qw.Ne1;
            if (_weights.TryGetValue(weightName, out var w))
                return (int)w.Sizes[0];
            return 0;
        }

        /// <summary>
        /// Shard a column-parallel weight whose output dimension is the
        /// concatenation of several logical segments — e.g. a fused QKV
        /// projection [Q | K | V] or a fused FFN [gate | up].
        ///
        /// The generic <see cref="ShardWeightsForTensorParallelism"/> column
        /// split assigns each rank one CONTIGUOUS block of output rows. For a
        /// concatenated weight that mixes whole segments across ranks (rank 0
        /// gets all of Q/gate, rank 1 gets all of K+V/up), which is wrong: the
        /// forward pass re-splits each rank's shard expecting it to contain that
        /// rank's slice of EVERY segment, i.e. [seg0_r | seg1_r | …].
        ///
        /// This method instead gathers, for each rank, its 1/tp slice of every
        /// segment in order, producing the per-rank layout the forward pass
        /// expects. Every segment dim must be divisible by the global TP degree
        /// (enforced by the per-model ValidateTpConstraints for head counts and
        /// intermediate sizes). No-op if the weight is not present.
        /// </summary>
        protected void ShardConcatenatedColumnParallel(string weightName, params int[] segmentDims)
        {
            if (!IsTensorParallel) return;

            int tp = TpDegree;
            int globalTp = GlobalTpDegree;
            int rankOffset = TpRankOffset;

            // Output-row indices for a global rank: its 1/tp slice of each segment.
            int[] BuildRows(int globalRank)
            {
                var idx = new List<int>();
                int baseOff = 0;
                foreach (int segDim in segmentDims)
                {
                    int perRank = segDim / globalTp;
                    int start = baseOff + globalRank * perRank;
                    for (int i = 0; i < perRank; i++)
                        idx.Add(start + i);
                    baseOff += segDim;
                }
                return idx.ToArray();
            }

            if (_quantWeights.TryGetValue(weightName, out var qw))
            {
                long rowBytes = NativeDequant.RowSize(qw.GgmlType, qw.Ne0);
                var shards = new QuantizedWeight[tp];
                for (int r = 0; r < tp; r++)
                {
                    int[] rows = BuildRows(rankOffset + r);
                    long totalBytes = rows.Length * rowBytes;
                    IntPtr shardPtr = QuantizedWeight.AllocateBuffer(totalBytes);
                    unsafe
                    {
                        byte* src = (byte*)qw.Data.ToPointer();
                        byte* dst = (byte*)shardPtr.ToPointer();
                        for (int row = 0; row < rows.Length; row++)
                            Buffer.MemoryCopy(
                                src + (long)rows[row] * rowBytes,
                                dst + (long)row * rowBytes,
                                rowBytes, rowBytes);
                    }
                    shards[r] = new QuantizedWeight(shardPtr, totalBytes,
                        qw.GgmlType, qw.Ne0, rows.Length);
                }

                _tpQuantWeights[weightName] = shards;
                _quantWeights.Remove(weightName);
                qw.Dispose();
            }
            else if (_weights.TryGetValue(weightName, out var w))
            {
                int inDim = (int)w.Sizes[1];
                var shards = new Tensor[tp];
                for (int r = 0; r < tp; r++)
                {
                    int[] rows = BuildRows(rankOffset + r);
                    var shard = new Tensor(_tpGroup.GetAllocator(r), DType.Float32, rows.Length, inDim);
                    unsafe
                    {
                        float* srcPtr = GetFloatPtr(w);
                        float* dstPtr = GetFloatPtr(shard);
                        for (int row = 0; row < rows.Length; row++)
                            Buffer.MemoryCopy(
                                srcPtr + (long)rows[row] * inDim,
                                dstPtr + (long)row * inDim,
                                (long)inDim * 4, (long)inDim * 4);
                    }
                    shards[r] = shard;
                }

                _tpWeights[weightName] = shards;
                _weights.Remove(weightName);
                w.Dispose();
            }
        }

        /// <summary>
        /// Shard a fused [gate | up] FFN projection as two equal column-parallel
        /// segments, deriving the half size from the weight itself. No-op if the
        /// weight is not present (e.g. a MoE layer with no dense gate_up).
        /// </summary>
        protected void ShardFusedGateUpColumnParallel(string weightName)
        {
            int fullDim = GetFusedOutputDim(weightName);
            if (fullDim <= 0) return;
            int half = fullDim / 2;
            ShardConcatenatedColumnParallel(weightName, half, half);
        }

        /// <summary>
        /// Column-parallel shard from SEPARATE source weights that were never
        /// fused (e.g. Q/K/V with mismatched quant types that prevented
        /// <see cref="ShardConcatenatedColumnParallel"/>). Each source is
        /// dequantized (if quantized) and sliced per-rank on the fly, so no
        /// full-model F32 intermediate is ever materialised. The per-rank
        /// results are stored under <paramref name="fusedName"/> - as Q8_0 in
        /// <see cref="_tpQuantWeights"/> when the input dim allows it, otherwise
        /// as F32 in <see cref="_tpWeights"/>; source weights are removed and
        /// disposed.
        /// </summary>
        protected void ShardSeparateColumnParallel(string fusedName, string[] sourceNames, int[] segmentDims)
        {
            if (!IsTensorParallel) return;
            if (sourceNames.Length != segmentDims.Length)
                throw new ArgumentException("sourceNames and segmentDims must have the same length.");

            int tp = TpDegree;
            int globalTp = GlobalTpDegree;
            int rankOffset = TpRankOffset;

            // Resolve sources and common input dim.
            int inDim = -1;
            var quants = new QuantizedWeight[sourceNames.Length];
            var tensors = new Tensor[sourceNames.Length];
            for (int s = 0; s < sourceNames.Length; s++)
            {
                if (_quantWeights.TryGetValue(sourceNames[s], out quants[s]))
                {
                    int d = (int)quants[s].Ne0;
                    if (inDim < 0) inDim = d; else if (inDim != d) return;
                }
                else if (_weights.TryGetValue(sourceNames[s], out tensors[s]))
                {
                    int d = (int)tensors[s].Sizes[1];
                    if (inDim < 0) inDim = d; else if (inDim != d) return;
                }
                else
                {
                    return; // source missing — nothing to shard
                }
            }

            // This path exists because the sources had MIXED quant types, so they
            // could not be packed in place. Materializing the shard in F32 as well
            // costs ~8x the source bytes AND drops the layer off the cached
            // quantized matmul onto the generic Ops.Addmm path, which re-uploads
            // the whole weight per call - on a mixed-quant Qwen3.8-27B that alone
            // was several GB per rank and erased the throughput tensor parallelism
            // was supposed to buy. Re-encode each gathered row as Q8_0 instead: the
            // gather is row-wise, so a row's values are untouched, and an 8-bit
            // round-trip of an already <=6-bit source is lossless in practice. Q8_0
            // blocks are 32 elements along the INPUT dim, so this needs inDim % 32.
            const int q8Type = (int)GgmlTensorType.Q8_0;
            bool requantize = (inDim % 32) == 0;

            var shards = requantize ? null : new Tensor[tp];
            var quantShards = requantize ? new QuantizedWeight[tp] : null;
            long q8RowBytes = requantize ? NativeDequant.RowSize(q8Type, inDim) : 0;

            for (int r = 0; r < tp; r++)
            {
                int globalRank = rankOffset + r;
                int totalRows = 0;
                foreach (int segDim in segmentDims)
                    totalRows += segDim / globalTp;

                Tensor shard = requantize
                    ? null
                    : new Tensor(_tpGroup.GetAllocator(r), DType.Float32, totalRows, inDim);
                IntPtr quantPtr = requantize
                    ? QuantizedWeight.AllocateBuffer((long)totalRows * q8RowBytes)
                    : IntPtr.Zero;
                // One reusable row buffer when re-quantizing: the shard is written
                // row at a time, so no full-size F32 intermediate is ever allocated.
                float[] rowScratch = requantize ? new float[inDim] : null;

                unsafe
                {
                    float* dstBase = requantize ? null : GetFloatPtr(shard);
                    byte* quantBase = requantize ? (byte*)quantPtr.ToPointer() : null;
                    long dstRow = 0;

                    fixed (float* scratch = rowScratch)
                    {
                        for (int s = 0; s < sourceNames.Length; s++)
                        {
                            int perRank = segmentDims[s] / globalTp;
                            int srcStart = globalRank * perRank;

                            for (int row = 0; row < perRank; row++)
                            {
                                float* dstRowPtr = requantize
                                    ? scratch
                                    : dstBase + (dstRow + row) * inDim;

                                if (quants[s] != null)
                                {
                                    var qw = quants[s];
                                    long rowBytes = NativeDequant.RowSize(qw.GgmlType, qw.Ne0);
                                    byte* srcBase = (byte*)qw.Data.ToPointer();
                                    NativeDequant.DequantizeToFloat32Native(
                                        qw.GgmlType,
                                        (IntPtr)(srcBase + (long)(srcStart + row) * rowBytes),
                                        (IntPtr)dstRowPtr,
                                        inDim);
                                }
                                else
                                {
                                    float* srcPtr = GetFloatPtr(tensors[s]);
                                    long bytes = (long)inDim * sizeof(float);
                                    Buffer.MemoryCopy(
                                        srcPtr + (long)(srcStart + row) * inDim,
                                        dstRowPtr, bytes, bytes);
                                }

                                if (requantize)
                                    ManagedQuantizedOps.QuantizeRowFromFloat32(
                                        q8Type, dstRowPtr,
                                        (IntPtr)(quantBase + (dstRow + row) * q8RowBytes),
                                        inDim);
                            }
                            dstRow += perRank;
                        }
                    }
                }

                if (requantize)
                    quantShards[r] = new QuantizedWeight(quantPtr, (long)totalRows * q8RowBytes,
                        q8Type, inDim, totalRows);
                else
                    shards[r] = shard;
            }

            if (requantize)
                _tpQuantWeights[fusedName] = quantShards;
            else
                _tpWeights[fusedName] = shards;

            for (int s = 0; s < sourceNames.Length; s++)
            {
                if (quants[s] != null)
                {
                    _quantWeights.Remove(sourceNames[s]);
                    quants[s].Dispose();
                }
                else
                {
                    _weights.Remove(sourceNames[s]);
                    tensors[s].Dispose();
                }
            }
        }

        /// <summary>
        /// Column-parallel shard of a 1-D bias whose length is the concatenation
        /// of several logical segments (fused QKV bias [Q|K|V], fused gate_up
        /// bias [gate|up], …). The bias must be regrouped exactly like the
        /// weight it accompanies (<see cref="ShardConcatenatedColumnParallel"/>),
        /// otherwise each rank would add the wrong bias slice to its outputs.
        /// No-op if the bias is not present.
        /// </summary>
        protected void ShardConcatenatedBiasColumnParallel(string biasName, params int[] segmentDims)
        {
            if (!IsTensorParallel) return;
            if (!_weights.TryGetValue(biasName, out var bias)) return;

            int tp = TpDegree;
            int globalTp = GlobalTpDegree;
            int rankOffset = TpRankOffset;

            int[] BuildRows(int globalRank)
            {
                var idx = new List<int>();
                int baseOff = 0;
                foreach (int segDim in segmentDims)
                {
                    int perRank = segDim / globalTp;
                    int start = baseOff + globalRank * perRank;
                    for (int i = 0; i < perRank; i++)
                        idx.Add(start + i);
                    baseOff += segDim;
                }
                return idx.ToArray();
            }

            var shards = new Tensor[tp];
            for (int r = 0; r < tp; r++)
            {
                int[] rows = BuildRows(rankOffset + r);
                var shard = new Tensor(_tpGroup.GetAllocator(r), DType.Float32, rows.Length);
                unsafe
                {
                    float* srcPtr = GetFloatPtr(bias);
                    float* dstPtr = GetFloatPtr(shard);
                    for (int i = 0; i < rows.Length; i++)
                        dstPtr[i] = srcPtr[rows[i]];
                }
                shards[r] = shard;
            }

            _tpWeights[biasName] = shards;
            _weights.Remove(biasName);
            bias.Dispose();
        }

        /// <summary>
        /// Column-parallel shard of a logical fused bias whose SOURCE tensors
        /// were never fused (e.g. Q/K/V biases next to mixed-quant-type Q/K/V
        /// weights that declined load-time fusion). Gathers each rank's slice
        /// of every source bias, in order, into one per-rank tensor registered
        /// under <paramref name="fusedBiasName"/> — the bias counterpart of
        /// <see cref="ShardSeparateColumnParallel"/>, producing exactly the
        /// per-rank [seg0_r | seg1_r | …] layout the forward's fused lookup
        /// expects. No-op if any source bias is absent.
        /// </summary>
        protected void ShardSeparateBiasColumnParallel(string fusedBiasName, string[] sourceNames, int[] segmentDims)
        {
            if (!IsTensorParallel) return;
            if (sourceNames.Length != segmentDims.Length)
                throw new ArgumentException("sourceNames and segmentDims must have the same length.");

            var sources = new Tensor[sourceNames.Length];
            for (int s = 0; s < sourceNames.Length; s++)
            {
                if (!_weights.TryGetValue(sourceNames[s], out sources[s]))
                    return; // biases are optional — mirror the fused-bias no-op
            }

            int tp = TpDegree;
            int globalTp = GlobalTpDegree;
            int rankOffset = TpRankOffset;

            int totalLen = 0;
            foreach (int segDim in segmentDims)
                totalLen += segDim / globalTp;

            var shards = new Tensor[tp];
            for (int r = 0; r < tp; r++)
            {
                int globalRank = rankOffset + r;
                var shard = new Tensor(_tpGroup.GetAllocator(r), DType.Float32, totalLen);
                unsafe
                {
                    float* dst = GetFloatPtr(shard);
                    int dstOff = 0;
                    for (int s = 0; s < sourceNames.Length; s++)
                    {
                        int perRank = segmentDims[s] / globalTp;
                        float* src = GetFloatPtr(sources[s]);
                        int start = globalRank * perRank;
                        for (int i = 0; i < perRank; i++)
                            dst[dstOff + i] = src[start + i];
                        dstOff += perRank;
                    }
                }
                shards[r] = shard;
            }

            _tpWeights[fusedBiasName] = shards;
            for (int s = 0; s < sourceNames.Length; s++)
            {
                _weights.Remove(sourceNames[s]);
                sources[s].Dispose();
            }
        }

        /// <summary>
        /// Column-parallel shard of a fused [gate | up] bias as two equal
        /// segments, deriving the half length from the bias itself.
        /// </summary>
        protected void ShardFusedGateUpBiasColumnParallel(string biasName)
        {
            if (!_weights.TryGetValue(biasName, out var bias)) return;
            int total = (int)bias.ElementCount();
            int half = total / 2;
            ShardConcatenatedBiasColumnParallel(biasName, half, half);
        }

        /// <summary>
        /// Column-parallel linear forward: each GPU computes its output slice
        /// using its weight shard. Input is replicated across all GPUs.
        /// Returns one output tensor per GPU, each with outDim/tp columns.
        /// </summary>
        protected Tensor[] TpColumnParallelLinear(Tensor input, string weightName)
        {
            int tp = TpDegree;
            var results = new Tensor[tp];
            long t0 = Stopwatch.GetTimestamp();

            if (_tpQuantWeights.TryGetValue(weightName, out var qShards))
            {
                int seqLen = (int)input.Sizes[0];
                // Column-parallel: every rank reads the same replicated
                // activation straight out of host memory, so one shared input
                // tensor covers all of them.
                var sharedInputs = new Tensor[tp];
                for (int r = 0; r < tp; r++) sharedInputs[r] = input;
                if (TryTpFusedQuantLinear(sharedInputs, qShards, results, seqLen, allReduce: false))
                {
                    _linearTicks += Stopwatch.GetTimestamp() - t0;
                    return results;
                }

                _tpGroup.RunPerRank(r =>
                {
                    var qw = qShards[r];
                    int outDim = (int)qw.Ne1;
                    var alloc = _tpGroup.GetAllocator(r);
                    results[r] = new Tensor(alloc, DType.Float32, seqLen, outDim);
                    var localInput = ReplicateTensorToRank(input, r);
                    AddmmQuantManaged(results[r], localInput, qw);
                    // ReplicateTensorToRank allocates a fresh cross-GPU copy for
                    // every rank > 0; without this it leaked one [seqLen, inDim]
                    // tensor per rank per call, on every layer of every token.
                    if (!ReferenceEquals(localInput, input)) localInput.Dispose();
                });
            }
            else if (_tpWeights.TryGetValue(weightName, out var wShards))
            {
                int seqLen = (int)input.Sizes[0];
                _tpGroup.RunPerRank(r =>
                {
                    var w = wShards[r];
                    int outDim = (int)w.Sizes[0];
                    var alloc = _tpGroup.GetAllocator(r);
                    using var wT = w.Transpose();
                    results[r] = new Tensor(alloc, DType.Float32, seqLen, outDim);
                    var localInput = ReplicateTensorToRank(input, r);
                    Ops.Addmm(results[r], 0, results[r], 1.0f, localInput, wT);
                    if (!ReferenceEquals(localInput, input)) localInput.Dispose();
                });
            }
            else
            {
                throw new KeyNotFoundException($"TP column-parallel weight '{weightName}' not found in sharded weights.");
            }

            _linearTicks += Stopwatch.GetTimestamp() - t0;
            return results;
        }

        /// <summary>
        /// GGML fast path for a tensor-parallel quantized linear: hand all ranks
        /// to the native bridge in one call so the per-rank matmuls are submitted
        /// as concurrent device graphs (and, for a row-parallel layer, reduced
        /// on-device before they come back). Returns false on other backends, or
        /// when the shapes don't fit the fused entry point, so the caller can use
        /// the generic per-rank loop.
        /// </summary>
        /// <summary>
        /// The fused multi-rank linear is OFF by default; set
        /// TS_GGML_TP_FUSED_MATMUL=1 to use it.
        ///
        /// It submits both ranks' graphs from one thread without a hand-off,
        /// which sounds like the faster shape, but it has to allocate a fresh
        /// backend buffer per rank per call: its graphs run asynchronously, so it
        /// cannot use the shared per-rank compute buffer the way the ordinary
        /// per-op path does. On CUDA that is a cudaMalloc/cudaFree pair per rank
        /// per linear, and on a nearly-full card the cost dwarfs the hand-off it
        /// saves. Measured on Qwen3.5-35B-A3B, --tp 2, ggml_cuda: decode 8.7 →
        /// 20.1 tok/s and the LM head 44 → 1.6 ms/token with it disabled.
        ///
        /// The generic path is not serial either — <c>RunPerRank</c> fans the
        /// ranks out across worker threads — so what is given up is the in-call
        /// device AllReduce, which <c>ITensorParallelGroup.AllReduce</c> does
        /// anyway.
        /// </summary>
        private static readonly bool TpFusedMatmulEnabled =
            string.Equals(Environment.GetEnvironmentVariable("TS_GGML_TP_FUSED_MATMUL"), "1", StringComparison.Ordinal);

        private bool TryTpFusedQuantLinear(Tensor[] inputs, QuantizedWeight[] shards, Tensor[] results, int seqLen, bool allReduce)
        {
            // The fused entry point exists to overlap several local GPUs. With a
            // single local rank — the multi-node layout, one GPU per node — there
            // is nothing to overlap, and it is not the validated configuration:
            // it was measured to produce wrong results there, while the generic
            // per-rank path is correct. Require a genuine multi-GPU group.
            if (!TpFusedMatmulEnabled || !IsGgmlBackend || _ggmlContext == null ||
                TpDegree < 2 || _ggmlContext.Degree != TpDegree)
                return false;

            int tp = TpDegree;
            for (int r = 0; r < tp; r++)
            {
                if (!shards[r].HasHostData)
                    return false;
            }

            var data = new IntPtr[tp];
            var types = new int[tp];
            var ne0 = new long[tp];
            var ne1 = new long[tp];
            var raw = new long[tp];

            for (int r = 0; r < tp; r++)
            {
                var qw = shards[r];
                results[r] = new Tensor(_tpGroup.GetAllocator(r), DType.Float32, seqLen, (int)qw.Ne1);
                // CacheKey identifies the rank-resident device copy created by
                // PrepareGgmlQuantizedWeightsForInferenceTP; Data would miss it.
                data[r] = qw.CacheKey;
                types[r] = qw.GgmlType;
                ne0[r] = qw.Ne0;
                ne1[r] = qw.Ne1;
                raw[r] = qw.RawBytes;
            }

            try
            {
                GgmlBasicOps.TensorParallelMatmul(results, inputs, data, types, ne0, ne1, raw, allReduce);
                return true;
            }
            catch (NotSupportedException)
            {
                for (int r = 0; r < tp; r++)
                {
                    results[r]?.Dispose();
                    results[r] = null;
                }
                return false;
            }
        }

        /// <summary>
        /// Row-parallel linear forward: each GPU computes a partial result using
        /// its weight shard, then AllReduce sums the partials. Returns the
        /// reduced result (replicated on all GPUs); the caller typically uses
        /// the rank-0 tensor and disposes the rest.
        /// </summary>
        protected Tensor TpRowParallelLinear(Tensor[] inputs, string weightName)
        {
            int tp = TpDegree;
            var partials = new Tensor[tp];
            long t0 = Stopwatch.GetTimestamp();

            if (_tpQuantWeights.TryGetValue(weightName, out var qShards))
            {
                int seqLen = (int)inputs[0].Sizes[0];
                if (TryTpFusedQuantLinear(inputs, qShards, partials, seqLen, allReduce: true))
                {
                    _linearTicks += Stopwatch.GetTimestamp() - t0;
                    for (int r = 1; r < tp; r++)
                        partials[r].Dispose();
                    return partials[0];
                }

                _tpGroup.RunPerRank(r =>
                {
                    var qw = qShards[r];
                    int rows = (int)inputs[r].Sizes[0];
                    int outDim = (int)qw.Ne1;
                    var alloc = _tpGroup.GetAllocator(r);
                    partials[r] = new Tensor(alloc, DType.Float32, rows, outDim);
                    AddmmQuantManaged(partials[r], inputs[r], qw);
                });
            }
            else if (_tpWeights.TryGetValue(weightName, out var wShards))
            {
                _tpGroup.RunPerRank(r =>
                {
                    var w = wShards[r];
                    int seqLen = (int)inputs[r].Sizes[0];
                    int outDim = (int)w.Sizes[0];
                    var alloc = _tpGroup.GetAllocator(r);
                    using var wT = w.Transpose();
                    partials[r] = new Tensor(alloc, DType.Float32, seqLen, outDim);
                    Ops.Addmm(partials[r], 0, partials[r], 1.0f, inputs[r], wT);
                });
            }
            else
            {
                throw new KeyNotFoundException($"TP row-parallel weight '{weightName}' not found in sharded weights.");
            }

            _linearTicks += Stopwatch.GetTimestamp() - t0;

            _tpGroup.AllReduce(partials);

            // Dispose non-zero-rank tensors; caller uses rank-0 result.
            for (int r = 1; r < tp; r++)
                partials[r].Dispose();

            return partials[0];
        }

        /// <summary>
        /// Row-parallel linear that keeps every rank's copy of the reduced result.
        /// AllReduce already leaves the sum on all GPUs, so callers that need the
        /// value replicated (a residual add on each rank, say) should use this
        /// instead of <see cref="TpRowParallelLinear"/> followed by
        /// <see cref="BroadcastTensorToAllRanks"/> — that pair throws away the
        /// reduced copies and then re-sends rank 0's, costing an extra full
        /// cross-GPU broadcast per call (and on a box without working P2P, an
        /// extra host round trip).
        /// </summary>
        protected Tensor[] TpRowParallelLinearAllRanks(Tensor[] inputs, string weightName)
        {
            int tp = TpDegree;
            var partials = new Tensor[tp];
            long t0 = Stopwatch.GetTimestamp();

            if (_tpQuantWeights.TryGetValue(weightName, out var qShards))
            {
                int seqLen = (int)inputs[0].Sizes[0];
                if (TryTpFusedQuantLinear(inputs, qShards, partials, seqLen, allReduce: true))
                {
                    _linearTicks += Stopwatch.GetTimestamp() - t0;
                    return partials;
                }

                _tpGroup.RunPerRank(r =>
                {
                    var qw = qShards[r];
                    int rows = (int)inputs[r].Sizes[0];
                    var alloc = _tpGroup.GetAllocator(r);
                    partials[r] = new Tensor(alloc, DType.Float32, rows, (int)qw.Ne1);
                    AddmmQuantManaged(partials[r], inputs[r], qw);
                });
            }
            else if (_tpWeights.TryGetValue(weightName, out var wShards))
            {
                _tpGroup.RunPerRank(r =>
                {
                    var w = wShards[r];
                    int seqLen = (int)inputs[r].Sizes[0];
                    var alloc = _tpGroup.GetAllocator(r);
                    using var wT = w.Transpose();
                    partials[r] = new Tensor(alloc, DType.Float32, seqLen, (int)w.Sizes[0]);
                    Ops.Addmm(partials[r], 0, partials[r], 1.0f, inputs[r], wT);
                });
            }
            else
            {
                throw new KeyNotFoundException($"TP row-parallel weight '{weightName}' not found in sharded weights.");
            }

            _linearTicks += Stopwatch.GetTimestamp() - t0;

            _tpGroup.AllReduce(partials);
            return partials;
        }

        /// <summary>
        /// Ensure a tensor is available on the given rank's GPU. If the tensor
        /// is already on that GPU (rank 0 typically), returns it as-is.
        /// Otherwise copies it to the target GPU.
        /// </summary>
        protected Tensor ReplicateTensorToRank(Tensor tensor, int rank)
        {
            if (rank == 0) return tensor;

            // GGML tensors live in host memory that every rank's backend can read
            // directly, so "replicating" is just handing over the same buffer.
            // Copying would add a full activation-sized memcpy per rank per
            // linear for no benefit.
            if (IsGgmlBackend) return tensor;

            var alloc = _tpGroup.GetAllocator(rank);
            var copy = new Tensor(alloc, tensor.ElementType, tensor.Sizes);
            Ops.Copy(copy, tensor);
            return copy;
        }

        /// <summary>
        /// Broadcast a rank-0 tensor to all other GPUs. Returns an array where
        /// element 0 is the original tensor and elements 1..tp-1 are copies.
        /// </summary>
        protected Tensor[] BroadcastTensorToAllRanks(Tensor tensor)
        {
            int tp = TpDegree;
            var result = new Tensor[tp];

            // Note: do NOT collapse the GGML case to one shared buffer, even
            // though every rank's backend can read the same host memory. Callers
            // accumulate into these per rank (TpResidualAdd is
            // `hidden[r] += residual[r]` under RunPerRank), so aliasing them
            // would apply the same residual tp times — concurrently, on one
            // buffer. The copy below is what keeps that correct.

            // Clone for rank 0 too — the caller may dispose the original
            // tensor after broadcasting, and sharing the storage would leave
            // rank 0 with a dangling reference.
            var alloc0 = _tpGroup.GetAllocator(0);
            result[0] = new Tensor(alloc0, tensor.ElementType, tensor.Sizes);
            Ops.Copy(result[0], tensor);
            for (int r = 1; r < tp; r++)
                result[r] = ReplicateTensorToRank(tensor, r);
            return result;
        }

        // Per-rank copies of REPLICATED weights (norm alphas, gate vectors) that a
        // rank-r kernel reads directly. The source lives on rank 0, so a rank-r
        // kernel reading it would cross GPUs — fault without peer access, wrong
        // data with it. Cached because these are re-read on every layer of every
        // token: re-copying per call cost one cross-GPU transfer per norm per
        // rank per token, which on a host-staged (no working P2P) box dominated
        // the forward pass.
        private Dictionary<(Tensor, int), Tensor> _tpWeightReplicaCache;

        /// <summary>
        /// A replicated weight resident on rank <paramref name="rank"/>'s GPU.
        /// Rank 0 aliases the original; other ranks get a lazily-cached copy.
        /// </summary>
        protected Tensor TpReplicatedWeight(Tensor weight, int rank)
        {
            if (rank == 0 || weight == null)
                return weight;

            _tpWeightReplicaCache ??= new Dictionary<(Tensor, int), Tensor>();
            var key = (weight, rank);
            if (!_tpWeightReplicaCache.TryGetValue(key, out var copy))
            {
                copy = ReplicateTensorToRank(weight, rank);
                _tpWeightReplicaCache[key] = copy;
            }
            return copy;
        }

        protected void DisposeTpWeightReplicaCache()
        {
            if (_tpWeightReplicaCache == null)
                return;
            // Every entry is a rank >= 1 copy (rank 0 aliases the original and is
            // never stored), so all of them are ours to free.
            foreach (var copy in _tpWeightReplicaCache.Values)
                copy?.Dispose();
            _tpWeightReplicaCache.Clear();
            _tpWeightReplicaCache = null;
        }

        /// <summary>
        /// TP-aware RMSNorm: runs independently on each GPU (replicated weights).
        /// </summary>
        protected Tensor[] TpRMSNorm(Tensor[] inputs, string weightName)
        {
            int tp = TpDegree;
            var results = new Tensor[tp];
            var alpha = _weights[weightName];

            // Materialize every rank's replica up front: TpReplicatedWeight
            // populates a shared dictionary, which is not safe to fill from the
            // concurrent rank workers below.
            var alphaLocal = new Tensor[tp];
            for (int r = 0; r < tp; r++)
                alphaLocal[r] = TpReplicatedWeight(alpha, r);

            _tpGroup.RunPerRank(r =>
            {
                int rows = (int)inputs[r].Sizes[0];
                int dim = (int)(inputs[r].ElementCount() / rows);
                Tensor input2d = inputs[r].Sizes.Length != 2 ? inputs[r].View(rows, dim) : null;
                Tensor src = input2d ?? inputs[r];

                results[r] = Ops.RMSNorm(null, src, alphaLocal[r], null, Config.Eps);

                input2d?.Dispose();
            });

            return results;
        }

        /// <summary>
        /// TP-aware residual add: hidden[r] += residual[r] on each GPU.
        /// </summary>
        protected void TpResidualAdd(Tensor[] hidden, Tensor[] residual)
        {
            _tpGroup.RunPerRank(r => Ops.Add(hidden[r], hidden[r], residual[r]));
        }

        /// <summary>
        /// GGML counterpart of <see cref="PrepareCudaQuantizedWeightsForInferenceTP"/>:
        /// upload rank r's shard of every sharded weight to rank r's GPU, and the
        /// replicated (unsharded) weights to all of them.
        ///
        /// This is what actually splits the model across the GPUs. The shards are
        /// keyed in the native cache by host pointer *per rank*, so rank r's
        /// matmul finds its slice already resident on its own device and no GPU
        /// holds more than its share.
        ///
        /// Host copies are kept: unlike direct CUDA, the GGML path reads
        /// activations and any non-resident weight straight out of host memory,
        /// and the mmap'd GGUF pages cost no extra RAM.
        /// </summary>
        private void PrepareGgmlQuantizedWeightsForInferenceTP()
        {
            if (_cudaQuantWeightsPrepared || _tpGroup == null)
                return;

            EnsureQuantBackendAvailable();

            int tp = TpDegree;
            long[] bytesPerRank = new long[tp];
            int[] countPerRank = new int[tp];
            int tooLarge = 0;

            // Upload every rank concurrently: each rank's worker pushes only its
            // own shards (rank 0 also takes the replicated weights), so the
            // host reads — page faults against the memory-mapped GGUF included —
            // and the PCIe transfers to different GPUs overlap instead of
            // running one after another. RunPerRank pins each worker to its
            // rank, and the native per-rank buffer caches are mutex-protected,
            // so the concurrency here matches what inference already does.
            _tpGroup.RunPerRank(r =>
            {
                foreach (var kv in _tpQuantWeights)
                {
                    var qw = kv.Value[r];
                    // Same residency policy as the single-GPU preload: a routed
                    // expert belonging to a --n-cpu-moe layer is multiplied on the
                    // host, and uploading its shard would spend exactly the VRAM
                    // the flag exists to save.
                    if (!qw.HasHostData || !ShouldPreloadCudaQuantWeightToDevice(kv.Key))
                        continue;

                    // A zero-sized (or otherwise degenerate) shard is a producer
                    // bug in the model's TP sharding code. Say WHICH weight it is:
                    // the native preload can only report "invalid cache key, host
                    // data, and size", which is undiagnosable across the ~2400
                    // tensors of a real model.
                    if (qw.RawBytes <= 0 || qw.Ne0 <= 0 || qw.Ne1 <= 0)
                        throw new InvalidOperationException(
                            $"TP shard for weight '{kv.Key}' (rank {r}) is degenerate: " +
                            $"type={(GgmlTensorType)qw.GgmlType}, ne0={qw.Ne0}, ne1={qw.Ne1}, rawBytes={qw.RawBytes}. " +
                            "This is a bug in the model's TP weight sharding.");

                    IntPtr cacheKey = qw.EnsureDeviceCacheKey();
                    if (!GgmlBasicOps.PreloadQuantizedWeight(cacheKey, qw.Data, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes))
                    {
                        qw.MarkDevicePreloadTooLarge();
                        Interlocked.Increment(ref tooLarge);
                        continue;
                    }
                    bytesPerRank[r] += qw.RawBytes;
                    countPerRank[r]++;
                }

                // Replicated weights (embeddings, the LM head, anything the
                // model did not shard) stay on rank 0: they are read by the
                // pre/post-transformer stages, which run there. Duplicating
                // them on every GPU would undo a chunk of the memory saving TP
                // just bought.
                if (r != 0)
                    return;
                foreach (var kv in _quantWeights)
                {
                    QuantizedWeight qw = kv.Value;
                    if (!qw.HasHostData || !ShouldPreloadCudaQuantWeightToDevice(kv.Key))
                        continue;
                    // A source weight that was folded into a TP shard has no reader
                    // left under tensor parallelism, but it is still sitting in
                    // _quantWeights. Uploading it puts a full unsharded copy of that
                    // tensor on rank 0 on top of the shards - which is exactly the
                    // "TP still loads the whole model on each GPU" complaint.
                    if (IsSupersededByTpShard(kv.Key))
                        continue;
                    if (string.Equals(kv.Key, "token_embd.weight", StringComparison.Ordinal)
                        && !CanUseGgmlQuantizedGetRows(qw.GgmlType)
                        && (_quantWeights.ContainsKey("output.weight") || _weights.ContainsKey("output.weight")))
                        continue;

                    IntPtr cacheKey = qw.EnsureDeviceCacheKey();
                    if (!GgmlBasicOps.PreloadQuantizedWeight(cacheKey, qw.Data, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes))
                    {
                        qw.MarkDevicePreloadTooLarge();
                        Interlocked.Increment(ref tooLarge);
                        continue;
                    }
                    bytesPerRank[0] += qw.RawBytes;
                    countPerRank[0]++;
                }
            });

            // Architecture-specific per-rank weights that are not plain
            // QuantizedWeight shards (Gemma 4's stacked expert slices). A second
            // fan-out (rather than a tail call in the first) keeps the hook free
            // to assume every generic shard is already resident.
            _tpGroup.RunPerRank(r => PreloadGgmlTpAuxiliaryWeightsForRank(r, bytesPerRank, countPerRank));

            for (int r = 0; r < tp; r++)
            {
                Console.WriteLine($"  TP rank {r}: {countPerRank[r]} weight(s), {bytesPerRank[r] / 1024 / 1024} MB resident on GPU {_ggmlContext.DeviceIds[r]}");
            }
            if (tooLarge > 0)
                Console.WriteLine($"  {tooLarge} weight(s) exceeded the device single-buffer limit and stream from host memory.");

            // The banner above counts only quantized shards. Two other categories
            // decide whether tensor parallelism actually shrank per-GPU memory, and
            // neither was ever reported -- so "each GPU still loads the whole model"
            // could not be confirmed or refuted from a normal load log:
            //   * _tpWeights: shards that had to be DEQUANTIZED to F32 (mixed-quant
            //     Q/K/V, the packed GDN in-projection). These are ~8x the quantized
            //     bytes AND are served by the uncached generic matmul.
            //   * _quantWeights: whatever was never sharded at all, which stays
            //     replicated on rank 0 on top of its shards.
            long tpF32Bytes = 0;
            int tpF32Count = 0;
            foreach (var kv in _tpWeights)
            {
                var shards = kv.Value;
                if (shards == null) continue;
                tpF32Count++;
                foreach (var t in shards)
                    if (t != null) tpF32Bytes += t.ElementCount() * t.ElementType.Size();
            }
            if (tpF32Count > 0)
            {
                Console.WriteLine($"  TP F32 shards: {tpF32Count} weight(s), {tpF32Bytes / 1024 / 1024} MB across all ranks " +
                    "(dequantized because the source could not be split in its quant type; " +
                    "these also bypass the cached quantized matmul).");
            }
            long replicatedBytes = 0, supersededBytes = 0;
            int replicatedCount = 0, supersededCount = 0;
            foreach (var kv in _quantWeights)
            {
                if (kv.Value == null) continue;
                if (IsSupersededByTpShard(kv.Key))
                {
                    supersededCount++;
                    supersededBytes += kv.Value.RawBytes;
                }
                else
                {
                    replicatedCount++;
                    replicatedBytes += kv.Value.RawBytes;
                }
            }
            if (replicatedCount > 0)
            {
                Console.WriteLine($"  TP replicated on rank 0: {replicatedCount} unsharded quantized weight(s), " +
                    $"{replicatedBytes / 1024 / 1024} MB.");
            }
            if (supersededCount > 0)
            {
                Console.WriteLine($"  TP not resident: {supersededCount} fusion-source weight(s), " +
                    $"{supersededBytes / 1024 / 1024} MB (superseded by a shard; host mapping kept).");
            }

            GgmlBasicOps.SetActiveRank(0);
            _cudaQuantWeightsPrepared = true;
        }

        /// <summary>
        /// Hook for per-rank weights that do not live in
        /// <see cref="_tpQuantWeights"/> — Gemma 4's stacked expert slices, for
        /// instance. Called once per rank from a <c>RunPerRank</c> fan-out, so
        /// every rank uploads concurrently: the calling thread is already pinned
        /// to <paramref name="rank"/>'s GPU (do NOT call
        /// <c>GgmlBasicOps.SetActiveRank</c>), the implementation must touch only
        /// that rank's shards, and it must add what it uploaded to that rank's
        /// slot in the running totals so the load report stays accurate.
        /// </summary>
        protected virtual void PreloadGgmlTpAuxiliaryWeightsForRank(int rank, long[] bytesPerRank, int[] countPerRank) { }

        /// <summary>
        /// True when <paramref name="weightName"/> is still present in
        /// <see cref="_quantWeights"/> only because a fusion kept its sources alive,
        /// and tensor parallelism reads the fused/sharded tensor instead. Such a
        /// weight must not be uploaded to rank 0: it would duplicate, unsharded, a
        /// tensor the TP path already holds in shards. The host mapping stays, so a
        /// rare non-TP reader can still stream it.
        /// </summary>
        protected virtual bool IsSupersededByTpShard(string weightName) => false;

        /// <summary>
        /// True when the MoE layers under TP fall back to the per-token,
        /// per-expert dispatch loop, which is slow enough that startup needs the
        /// lightweight warmup. Architectures with a batched per-rank MoE path
        /// override this to false.
        /// </summary>
        protected virtual bool MoEUnderTpIsSlow => IsTensorParallel && HasMoEExpertWeights;

        /// <summary>
        /// Preload TP-sharded quantized weights onto their respective GPUs.
        /// Each shard is uploaded to the GPU that will use it, then the host
        /// copy is released.
        /// </summary>
        protected void PrepareCudaQuantizedWeightsForInferenceTP()
        {
            if (_backend == BackendType.GgmlCuda || _backend == BackendType.GgmlVulkan)
            {
                PrepareGgmlQuantizedWeightsForInferenceTP();
                return;
            }

            if (_backend != BackendType.Cuda || _tpQuantWeights.Count == 0)
                return;

            // When CUDA kernels are unavailable (PTX load failed), device-side
            // quantized matmul will fail and every op falls back to the CPU
            // dequant path.  In that case we MUST keep the host data alive
            // regardless of ShouldRetainCudaHostQuantWeight.
            bool kernelsAvailable = _allocator is CudaAllocator primaryAlloc
                && CudaQuantizedOps.AreKernelsAvailable(primaryAlloc);

            long preloadedBytes = 0;
            int preloadedCount = 0;

            long tpTotalBytes = 0;
            int tpTotalCount = 0;
            int tpDebugPrinted = 0;
            foreach (var kv in _tpQuantWeights)
            {
                foreach (var s in kv.Value)
                {
                    tpTotalBytes += s.RawBytes;
                    tpTotalCount++;
                }
                if (tpDebugPrinted < 3)
                {
                    var firstShard = kv.Value[0];
                    Console.WriteLine($"    TP debug: {kv.Key} shards={kv.Value.Length} shardRawBytes={firstShard.RawBytes} ne0={firstShard.Ne0} ne1={firstShard.Ne1} type={firstShard.GgmlType}");
                    tpDebugPrinted++;
                }
            }
            Console.WriteLine($"  TP sharded weights to preload: {tpTotalCount} shards, {tpTotalBytes / 1024 / 1024} MB total");

            int skippedExpertShards = 0;
            var oomShardsPerRank = new int[TpDegree];
            var oomBytesPerRank = new long[TpDegree];
            foreach (var kv in _tpQuantWeights)
            {
                var shards = kv.Value;
                bool retain = !kernelsAvailable || ShouldRetainCudaHostQuantWeight(kv.Key);
                // MoE expert shards are numerous (experts × layers × ranks) but they
                // are also the overwhelming majority of the model's bytes — on
                // Qwen3.5-35B-A3B they are ~15.6 GB of a 17 GB checkpoint. Keeping
                // them on the host (as this path used to) meant TP loaded almost
                // nothing into VRAM and every routed expert ran the CPU dequant
                // fallback in AddmmQuantManaged, which is what made TP prefill
                // ~100x slower than single-GPU. Each rank only holds a 1/tp slice,
                // so the sharded experts fit comfortably; the OOM catch below still
                // degrades gracefully to the host path if a GPU really is too small.
                for (int r = 0; r < shards.Length; r++)
                {
                    var qw = shards[r];
                    if (!qw.HasHostData || !CudaQuantizedOps.SupportsQuantizedType(qw.GgmlType)
                        // A routed expert on a --n-cpu-moe layer stays in system
                        // RAM; uploading its shard would spend exactly the VRAM
                        // the flag exists to save.
                        || !ShouldPreloadCudaQuantWeightToDevice(kv.Key))
                    {
                        if (kv.Key.Contains("_exps."))
                            skippedExpertShards++;
                        continue;
                    }

                    var alloc = (CudaAllocator)_tpGroup.GetAllocator(r);
                    IntPtr cacheKey = qw.EnsureDeviceCacheKey();
                    try
                    {
                        CudaQuantizedOps.PreloadQuantizedWeight(
                            alloc, cacheKey, qw.Data, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes);
                    }
                    catch (Exception ex) when (ex.Message.Contains("out of memory"))
                    {
                        // With experts now preloaded this can fire thousands of
                        // times on an undersized GPU, so report the first per rank
                        // and tally the rest.
                        if (oomShardsPerRank[r]++ == 0)
                        {
                            Console.WriteLine($"  Skipping TP device preload for {kv.Key} rank={r} ({qw.RawBytes / 1024 / 1024} MB) — GPU {r} out of memory, keeping host copy for CPU fallback.");
                        }
                        oomBytesPerRank[r] += qw.RawBytes;
                        continue;
                    }
                    preloadedBytes += qw.RawBytes;
                    preloadedCount++;
                    if (!retain)
                        qw.ReleaseHostData();
                }
            }

            // Also preload any remaining non-sharded quantized weights (e.g. embedding).
            if (_allocator is CudaAllocator cudaAllocator)
            {
                long remainingTotalBytes = 0;
                int remainingCount = 0;
                foreach (var kv in _quantWeights)
                {
                    remainingTotalBytes += kv.Value.RawBytes;
                    remainingCount++;
                }
                if (remainingCount > 0)
                    Console.WriteLine($"  TP non-sharded quantized weights remaining: {remainingCount} tensors, {remainingTotalBytes / 1024 / 1024} MB");

                foreach (var kv in _quantWeights)
                {
                    var qw = kv.Value;
                    if (!qw.HasHostData || !CudaQuantizedOps.SupportsQuantizedType(qw.GgmlType))
                        continue;

                    IntPtr cacheKey = qw.EnsureDeviceCacheKey();
                    try
                    {
                        CudaQuantizedOps.PreloadQuantizedWeight(
                            cudaAllocator, cacheKey, qw.Data, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes);
                    }
                    catch (Exception ex) when (ex.Message.Contains("out of memory"))
                    {
                        // GPU is full — keep the host copy so the CPU fallback
                        // path (EmbeddingManagedQuantized / AddmmQuantManaged)
                        // can serve this weight from host memory.
                        Console.WriteLine($"  Skipping device preload for {kv.Key} ({qw.RawBytes / 1024 / 1024} MB) — GPU out of memory, keeping host copy for CPU fallback.");
                        continue;
                    }
                    preloadedBytes += qw.RawBytes;
                    preloadedCount++;
                    if (kernelsAvailable && !ShouldRetainCudaHostQuantWeight(kv.Key))
                        qw.ReleaseHostData();
                }
            }

            _cudaQuantWeightsPrepared = true;

            // Only dispose the GGUF file when no external host views remain
            // (column-parallel shards may still reference mmap'd data) and
            // no TP shards were kept on host (expert CPU fallback).
            bool anyHostViewsRemain = false;
            foreach (var kv in _tpQuantWeights)
            {
                foreach (var qw in kv.Value)
                {
                    if (qw.HasHostData)
                    {
                        anyHostViewsRemain = true;
                        break;
                    }
                }
                if (anyHostViewsRemain) break;
            }
            if (!anyHostViewsRemain)
            {
                foreach (QuantizedWeight qw in _quantWeights.Values)
                {
                    if (qw.HasHostData)
                    {
                        anyHostViewsRemain = true;
                        break;
                    }
                }
            }
            if (!anyHostViewsRemain)
                _gguf?.Dispose();

            if (preloadedCount > 0)
                Console.WriteLine($"  TP CUDA resident quantized weights: {preloadedBytes / 1024 / 1024} MB across {preloadedCount} shards (host copies released)");

            for (int r = 0; r < oomShardsPerRank.Length; r++)
            {
                if (oomShardsPerRank[r] > 0)
                {
                    Console.WriteLine($"  TP GPU {r}: {oomShardsPerRank[r]} shards ({oomBytesPerRank[r] / 1024 / 1024} MB) did not fit in VRAM and stay on the host (CPU dequant fallback — expect much slower decode).");
                }
            }
            if (skippedExpertShards > 0)
            {
                Console.WriteLine($"  TP: {skippedExpertShards} expert shards have no CUDA quantized kernel for their type and stay on the host.");
            }

            for (int r = 0; r < TpDegree; r++)
            {
                if (_tpGroup.GetAllocator(r) is CudaAllocator rankAlloc)
                    rankAlloc.LogVram($"after TP quant weight preload (rank {r})");
            }
        }

        #endregion


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

            // Allocate the head-first result on DATA's GPU, not _allocator (GPU 0).
            // Under TP, `data` lives on rank r's GPU; a GPU-0 result would make the
            // reshape kernel read across GPUs (fault without peer, wrong data with).
            var result = new Tensor(data.Storage.Allocator, data.ElementType, numHeads, seqLen, headDim);
            if (CudaFusedOps.TryFlatToHeadFirst(result, data, numHeads, seqLen, headDim))
                return result;
            if (MlxFusedOps.TryFlatToHeadFirst(result, data, numHeads, seqLen, headDim))
                return result;
            result.Dispose();

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
            if (TryCopyHeadFirstToCacheMlx(cache, src, startPos, seqLen))
                return;

            if (CudaFusedOps.TryCopyHeadFirstToCache(cache, src, startPos, seqLen, (int)cache.Sizes[1], false))
                return;

            if (cache.ElementType == DType.Float16)
            {
                CopyToCacheF16(cache, src, startPos, seqLen);
                return;
            }
            if (IsBlockQuantCacheDType(cache.ElementType))
            {
                CopyToCacheBlockQuant(cache, src, startPos, seqLen);
                return;
            }

            using var cacheSlice = cache.Narrow(1, startPos, seqLen);
            Ops.Copy(cacheSlice, src);
            InvalidateTensorDeviceCache(cache);
        }

        /// <summary>
        /// Append <paramref name="seqLen"/> rows of an F32 (numKVHeads, seqLen, headDim)
        /// tensor to a block-quantized (Q4_0 / Q8_0) linear cache starting at
        /// <paramref name="startPos"/>, quantizing each appended position into the cache's
        /// block layout. The bytes written are dequantized identically by ggml's native
        /// kernels on the subsequent fused-decode read. Block-quant analogue of
        /// CopyToCacheF16.
        /// </summary>
        private unsafe void CopyToCacheBlockQuant(Tensor cache, Tensor src, int startPos, int seqLen)
        {
            int numKVHeads = (int)cache.Sizes[0];
            int maxSeqLen = (int)cache.Sizes[1];
            int headDim = (int)cache.Sizes[2];
            int ggmlType = GgmlTypeForCacheDType(cache.ElementType);
            long rowBytes = ManagedQuantizedOps.RowSize(ggmlType, headDim);

            cache.Storage.EnsureHostReadable();
            byte* dstBase = (byte*)TensorComputePrimitives.GetStoragePointer(cache);
            float* srcBase = GetFloatPtr(src);

            // Linear (global) cache: positions [startPos, startPos+seqLen) are appended
            // contiguously, so each head's seqLen rows form one contiguous block-aligned
            // run that quantizes in a single pass.
            for (int h = 0; h < numKVHeads; h++)
            {
                byte* dstHead = dstBase + (long)h * maxSeqLen * rowBytes + (long)startPos * rowBytes;
                float* srcHead = srcBase + (long)h * seqLen * headDim;
                ManagedQuantizedOps.QuantizeRowFromFloat32(ggmlType, srcHead, (IntPtr)dstHead,
                    (long)seqLen * headDim);
            }

            InvalidateTensorDeviceCache(cache);
        }

        /// <summary>
        /// Append <paramref name="seqLen"/> rows of an F32 (numKVHeads, seqLen, headDim) tensor
        /// to a Float16 cache of layout (numKVHeads, maxSeqLen, headDim) starting at
        /// <paramref name="startPos"/>. Performs a per-element F32-&gt;F16 conversion.
        /// </summary>
        private unsafe void CopyToCacheF16(Tensor cache, Tensor src, int startPos, int seqLen)
        {
            int numKVHeads = (int)cache.Sizes[0];
            int maxSeqLen = (int)cache.Sizes[1];
            int headDim = (int)cache.Sizes[2];

            ushort* dstBase = TensorComputePrimitives.GetHalfPointer(cache);
            float* srcBase = GetFloatPtr(src);

            // Source layout (head-first contiguous after ReshapeToHeads): (numKVHeads, seqLen, headDim).
            for (int h = 0; h < numKVHeads; h++)
            {
                ushort* dstHead = dstBase + (long)h * maxSeqLen * headDim + (long)startPos * headDim;
                float* srcHead = srcBase + (long)h * seqLen * headDim;
                TensorComputePrimitives.F32ToF16(dstHead, srcHead, seqLen * headDim);
            }

            InvalidateTensorDeviceCache(cache);
        }

        /// <summary>
        /// Return a contiguous F32 view of the active (0..totalSeqLen) region of the K
        /// or V cache, broadcasting along the head axis when GQA group_size &gt; 1.
        /// For Float16 caches the active region is dequantized into a freshly-allocated
        /// F32 tensor before broadcasting; for Float32 caches the existing fast path is used.
        /// </summary>
        protected unsafe Tensor ExpandKVHeads(Tensor cache, int groupSize, int totalSeqLen)
        {
            if (cache.ElementType == DType.Float16 || cache.ElementType == DType.Float32)
            {
                int numKVHeads = (int)cache.Sizes[0];
                int headDim = (int)cache.Sizes[2];
                var expanded = new Tensor(
                    _allocator, DType.Float32,
                    numKVHeads * groupSize, totalSeqLen, headDim);
                if (CudaFusedOps.TryExpandKvHeads(expanded, cache, groupSize, totalSeqLen))
                    return expanded;
                expanded.Dispose();
            }

            if (cache.ElementType == DType.Float16)
                return ExpandKVHeadsF16(cache, groupSize, totalSeqLen);
            if (IsBlockQuantCacheDType(cache.ElementType))
                return ExpandKVHeadsBlockQuant(cache, groupSize, totalSeqLen);

            using var active = cache.Narrow(1, 0, totalSeqLen);
            if (groupSize == 1)
                return Ops.NewContiguous(active);
            return Ops.RepeatInterleave(null, active, groupSize, 0);
        }

        // Block-quantized (Q4_0 / Q8_0) caches cannot be walked as a flat float
        // buffer, so the per-op prefill attention reads them by dequantizing the
        // active [0, totalSeqLen) window into a fresh F32 tensor (GQA-broadcast
        // along the head axis when group_size > 1) — the block-quant analogue of
        // ExpandKVHeadsF16. mul_mat then accumulates in F32, identical to having
        // dequantized in the native kernel.
        protected static bool IsBlockQuantCacheDType(DType dt) =>
            dt == DType.Q4_0 || dt == DType.Q8_0;

        // ggml type id for a block-quantized KV-cache dtype (must match ggml.h /
        // KvCacheDtypeExtensions.GgmlType): Q4_0 -> 2, Q8_0 -> 8.
        protected static int GgmlTypeForCacheDType(DType dt) => dt switch
        {
            DType.Q4_0 => 2,
            DType.Q8_0 => 8,
            _ => throw new NotSupportedException($"Not a block-quantized KV-cache dtype: {dt}"),
        };

        private unsafe Tensor ExpandKVHeadsBlockQuant(Tensor cache, int groupSize, int totalSeqLen)
        {
            int numKVHeads = (int)cache.Sizes[0];
            int maxSeqLen = (int)cache.Sizes[1];
            int headDim = (int)cache.Sizes[2];
            int outHeads = numKVHeads * groupSize;
            int ggmlType = GgmlTypeForCacheDType(cache.ElementType);
            long rowBytes = ManagedQuantizedOps.RowSize(ggmlType, headDim);

            cache.Storage.EnsureHostReadable();
            var f32 = new Tensor(cache.Storage.Allocator, DType.Float32, outHeads, totalSeqLen, headDim);
            float* dstBase = GetFloatPtr(f32);
            byte* srcBase = (byte*)TensorComputePrimitives.GetStoragePointer(cache);

            for (int h = 0; h < numKVHeads; h++)
            {
                // Active window is the first totalSeqLen slots of this head, which are
                // contiguous (each slot = headDim elements = a whole number of blocks).
                byte* srcHead = srcBase + (long)h * maxSeqLen * rowBytes;
                for (int g = 0; g < groupSize; g++)
                {
                    float* dstHead = dstBase + (long)(h * groupSize + g) * totalSeqLen * headDim;
                    ManagedQuantizedOps.DequantizeRowToFloat32(ggmlType, (IntPtr)srcHead,
                        dstHead, (long)totalSeqLen * headDim);
                }
            }

            InvalidateTensorDeviceCache(f32);
            return f32;
        }

        private unsafe Tensor ExpandKVHeadsF16(Tensor cache, int groupSize, int totalSeqLen)
        {
            int numKVHeads = (int)cache.Sizes[0];
            int maxSeqLen = (int)cache.Sizes[1];
            int headDim = (int)cache.Sizes[2];
            int outHeads = numKVHeads * groupSize;

            var f32 = new Tensor(cache.Storage.Allocator, DType.Float32, outHeads, totalSeqLen, headDim);
            float* dstBase = GetFloatPtr(f32);
            ushort* srcBase = TensorComputePrimitives.GetHalfPointer(cache);

            for (int h = 0; h < numKVHeads; h++)
            {
                ushort* srcHead = srcBase + (long)h * maxSeqLen * headDim;
                for (int g = 0; g < groupSize; g++)
                {
                    float* dstHead = dstBase + (long)(h * groupSize + g) * totalSeqLen * headDim;
                    TensorComputePrimitives.F16ToF32(dstHead, srcHead, totalSeqLen * headDim);
                }
            }

            InvalidateTensorDeviceCache(f32);
            return f32;
        }

        protected unsafe void CopyToCacheDecode(Tensor kCache, Tensor kTensor,
            Tensor vCache, Tensor vTensor, int numKVHeads, int headDim, int startPos)
        {
            using (var kHeads = kTensor.View(numKVHeads, 1, headDim))
            using (var vHeads = vTensor.View(numKVHeads, 1, headDim))
            {
                if (TryCopyHeadFirstToCacheMlx(kCache, kHeads, startPos, 1) &&
                    TryCopyHeadFirstToCacheMlx(vCache, vHeads, startPos, 1))
                {
                    return;
                }

                int cacheSize = (int)kCache.Sizes[1];
                if (CudaFusedOps.TryCopyHeadFirstToCache(kCache, kHeads, startPos, 1, cacheSize, false) &&
                    CudaFusedOps.TryCopyHeadFirstToCache(vCache, vHeads, startPos, 1, cacheSize, false))
                {
                    return;
                }
            }

            if (kCache.ElementType == DType.Float16 && vCache.ElementType == DType.Float16)
            {
                CopyToCacheDecodeF16(kCache, kTensor, vCache, vTensor, numKVHeads, headDim, startPos);
                return;
            }

            if (IsBlockQuantCacheDType(kCache.ElementType) && vCache.ElementType == kCache.ElementType)
            {
                CopyToCacheDecodeBlockQuant(kCache, kTensor, vCache, vTensor, numKVHeads, headDim, startPos);
                return;
            }

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

        protected bool TryCopyHeadFirstToCacheMlx(Tensor cache, Tensor src, int startPos, int seqLen, bool circular = false)
        {
            if (string.Equals(Environment.GetEnvironmentVariable("TS_MLX_DEVICE_KV_COPY"), "0", StringComparison.Ordinal))
                return false;

            if (circular)
                return TryCopyHeadFirstToCacheCircularMlx(cache, src, startPos, seqLen);

            if (_backend != BackendType.Mlx
                || cache == null
                || src == null
                || cache.Storage is not MlxStorage
                || src.Storage is not MlxStorage
                || cache.DimensionCount != 3
                || src.DimensionCount != 3
                || cache.Sizes[0] != src.Sizes[0]
                || src.Sizes[1] != seqLen
                || cache.Sizes[2] != src.Sizes[2]
                || startPos < 0
                || startPos + seqLen > cache.Sizes[1])
            {
                return false;
            }

            // Single multi-dim slice_update beats the per-head loop below by
            // ~8× MLX dispatches per cache write — for decode (kvHeads=2, K+V
            // per layer × 42 layers) that's ~600 MLX op dispatches/token
            // collapsed into ~80. Falls back to the per-head loop if the
            // fused path declines (e.g. dtype mismatch, sub-view storage).
            // Disable via TS_MLX_FUSED_KV_WRITE=0 to A/B against the per-head
            // path (helpful when investigating slice_update perf regressions).
            if (!string.Equals(Environment.GetEnvironmentVariable("TS_MLX_FUSED_KV_WRITE"), "0", StringComparison.Ordinal)
                && MlxFusedOps.TryWriteKvCacheBlock(cache, src, startPos, seqLen))
                return true;

            int heads = (int)cache.Sizes[0];
            for (int h = 0; h < heads; h++)
            {
                using Tensor cacheHead = cache.Select(0, h);
                using Tensor cacheSlice = cacheHead.Narrow(0, startPos, seqLen);
                using Tensor srcHead = src.Select(0, h);
                Ops.Copy(cacheSlice, srcHead);
            }

            return true;
        }

        private bool TryCopyHeadFirstToCacheCircularMlx(Tensor cache, Tensor src, int startPos, int seqLen)
        {
            if (_backend != BackendType.Mlx
                || cache == null
                || src == null
                || cache.Storage is not MlxStorage
                || src.Storage is not MlxStorage
                || cache.DimensionCount != 3
                || src.DimensionCount != 3
                || cache.Sizes[0] != src.Sizes[0]
                || src.Sizes[1] != seqLen
                || cache.Sizes[2] != src.Sizes[2]
                || startPos < 0
                || seqLen <= 0
                || cache.Sizes[1] <= 0)
            {
                return false;
            }

            int cacheSize = checked((int)cache.Sizes[1]);
            int srcOffset = 0;
            int remaining = seqLen;
            int logicalStart = startPos;
            if (remaining > cacheSize)
            {
                srcOffset = remaining - cacheSize;
                logicalStart += srcOffset;
                remaining = cacheSize;
            }

            while (remaining > 0)
            {
                int dstOffset = logicalStart % cacheSize;
                int chunk = Math.Min(remaining, cacheSize - dstOffset);
                if (!TryCopyHeadFirstRangeToCacheMlx(cache, src, srcOffset, dstOffset, chunk))
                    return false;

                srcOffset += chunk;
                logicalStart += chunk;
                remaining -= chunk;
            }

            return true;
        }

        private bool TryCopyHeadFirstRangeToCacheMlx(Tensor cache, Tensor src, int srcOffset, int dstOffset, int length)
        {
            // Fast path: single multi-dim slice_update for the full
            // [heads, length, headDim] block when src happens to start at
            // offset 0. For the wrap-around case we fall back to the per-head
            // loop with a manually narrowed src.
            if (srcOffset == 0 && length == src.Sizes[1])
            {
                if (MlxFusedOps.TryWriteKvCacheBlock(cache, src, dstOffset, length))
                    return true;
            }

            int heads = checked((int)cache.Sizes[0]);
            for (int h = 0; h < heads; h++)
            {
                using Tensor cacheHead = cache.Select(0, h);
                using Tensor cacheSlice = cacheHead.Narrow(0, dstOffset, length);
                using Tensor srcHead = src.Select(0, h);
                using Tensor srcSlice = srcHead.Narrow(0, srcOffset, length);
                Ops.Copy(cacheSlice, srcSlice);
            }

            return true;
        }

        /// <summary>
        /// Single-position decode append into a block-quantized (Q4_0 / Q8_0) linear
        /// cache: each head's new K/V row (headDim elements, a whole number of
        /// 32-element blocks) is quantized in place at its row offset. Decode analogue
        /// of <see cref="CopyToCacheBlockQuant"/>; bytes match ggml's block layout so
        /// the fused native kernels dequantize them identically on later reads.
        /// </summary>
        private unsafe void CopyToCacheDecodeBlockQuant(Tensor kCache, Tensor kTensor,
            Tensor vCache, Tensor vTensor, int numKVHeads, int headDim, int startPos)
        {
            int ggmlType = GgmlTypeForCacheDType(kCache.ElementType);
            long rowBytes = ManagedQuantizedOps.RowSize(ggmlType, headDim);
            int maxSeqLen = (int)kCache.Sizes[1];

            kCache.Storage.EnsureHostReadable();
            vCache.Storage.EnsureHostReadable();
            float* kSrc = GetFloatPtr(kTensor);
            float* vSrc = GetFloatPtr(vTensor);
            byte* kBase = (byte*)TensorComputePrimitives.GetStoragePointer(kCache);
            byte* vBase = (byte*)TensorComputePrimitives.GetStoragePointer(vCache);

            for (int h = 0; h < numKVHeads; h++)
            {
                long cacheOffset = (long)h * maxSeqLen * rowBytes + (long)startPos * rowBytes;
                int srcOffset = h * headDim;
                ManagedQuantizedOps.QuantizeRowFromFloat32(ggmlType, kSrc + srcOffset,
                    (IntPtr)(kBase + cacheOffset), headDim);
                ManagedQuantizedOps.QuantizeRowFromFloat32(ggmlType, vSrc + srcOffset,
                    (IntPtr)(vBase + cacheOffset), headDim);
            }

            InvalidateTensorDeviceCache(kCache);
            InvalidateTensorDeviceCache(vCache);
        }

        private unsafe void CopyToCacheDecodeF16(Tensor kCache, Tensor kTensor,
            Tensor vCache, Tensor vTensor, int numKVHeads, int headDim, int startPos)
        {
            float* kSrc = GetFloatPtr(kTensor);
            float* vSrc = GetFloatPtr(vTensor);
            ushort* kDst = TensorComputePrimitives.GetHalfPointer(kCache);
            ushort* vDst = TensorComputePrimitives.GetHalfPointer(vCache);
            int maxSeqLen = (int)kCache.Sizes[1];

            for (int h = 0; h < numKVHeads; h++)
            {
                long cacheOffset = (long)h * maxSeqLen * headDim + (long)startPos * headDim;
                int srcOffset = h * headDim;
                TensorComputePrimitives.F32ToF16(kDst + cacheOffset, kSrc + srcOffset, headDim);
                TensorComputePrimitives.F32ToF16(vDst + cacheOffset, vSrc + srcOffset, headDim);
            }

            InvalidateTensorDeviceCache(kCache);
            InvalidateTensorDeviceCache(vCache);
        }

        protected unsafe void AttentionDecodePureCS(Tensor q, Tensor kCache, Tensor vCache,
            Tensor result, int numHeads, int numKVHeads, int headDim, int totalSeqLen, float scale)
        {
            if (kCache.ElementType == DType.Float16 && vCache.ElementType == DType.Float16)
            {
                AttentionDecodePureCSF16(q, kCache, vCache, result,
                    numHeads, numKVHeads, headDim, totalSeqLen, scale);
                return;
            }

            if (IsBlockQuantCacheDType(kCache.ElementType) && vCache.ElementType == kCache.ElementType)
            {
                // Block-quantized (Q4_0 / Q8_0) caches cannot be walked as flat float
                // buffers. Dequantize the active [0, totalSeqLen) window into compact
                // F32 tensors (no GQA broadcast; the grouped kernel below reads per
                // KV head) and re-enter on the F32 path — the compact copy's
                // Sizes[1] == totalSeqLen doubles as its row stride. This is the
                // deep-fallback path (fused native attention handles quantized
                // caches on-device), so correctness beats the extra dequant cost.
                using (Tensor kF32 = ExpandKVHeadsBlockQuant(kCache, 1, totalSeqLen))
                using (Tensor vF32 = ExpandKVHeadsBlockQuant(vCache, 1, totalSeqLen))
                {
                    AttentionDecodePureCS(q, kF32, vF32, result,
                        numHeads, numKVHeads, headDim, totalSeqLen, scale);
                }
                return;
            }

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

        /// <summary>
        /// Single-token GQA decode attention specialized for an F16 KV cache.
        /// Reads K/V values as ushort, converts to F32 inside the dot/scale-add
        /// hot loops via <see cref="TensorComputePrimitives"/>. The cache layout
        /// is identical to the F32 variant - <c>(num_kv_heads, max_seq_len, head_dim)</c> -
        /// so callers don't need to special-case anything but the storage dtype.
        ///
        /// This is the C# fallback path when the native fused decode kernel is
        /// unavailable. On Apple Silicon Metal / CUDA the native path
        /// (<c>TransformerLayerDecode</c> / <c>TransformerModelDecode</c>) handles
        /// F16 K/V directly via <c>ggml_flash_attn_ext</c>, which is much faster.
        /// </summary>
        protected unsafe void AttentionDecodePureCSF16(Tensor q, Tensor kCache, Tensor vCache,
            Tensor result, int numHeads, int numKVHeads, int headDim, int totalSeqLen, float scale)
        {
            float* qPtr = GetFloatPtr(q);
            ushort* kPtr = TensorComputePrimitives.GetHalfPointer(kCache);
            ushort* vPtr = TensorComputePrimitives.GetHalfPointer(vCache);
            float* rPtr = GetFloatPtr(result);
            int maxSeqLen = (int)kCache.Sizes[1];
            int groupSize = numHeads / numKVHeads;

            int procCount = Environment.ProcessorCount;
            bool useParallel = numKVHeads > 1 && (long)numHeads * totalSeqLen >= 4096;

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
                float scaleLocal = scale;

                Parallel.For(0, numKVHeadsLocal, kvHead =>
                {
                    float* qP = (float*)qPtrL;
                    ushort* kP = (ushort*)kPtrL;
                    ushort* vP = (ushort*)vPtrL;
                    float* rP = (float*)rPtrL;
                    float* scoresBuf = stackalloc float[groupSizeLocal * totalSeqLenLocal];
                    AttentionDecodeKVHeadGroupedF16(kvHead, qP, kP, vP, rP, scoresBuf,
                        headDimLocal, maxSeqLenLocal, groupSizeLocal,
                        totalSeqLenLocal, scaleLocal);
                });
            }
            else
            {
                float* scores = stackalloc float[groupSize * totalSeqLen];
                for (int kvHead = 0; kvHead < numKVHeads; kvHead++)
                {
                    AttentionDecodeKVHeadGroupedF16(kvHead, qPtr, kPtr, vPtr, rPtr, scores,
                        headDim, maxSeqLen, groupSize, totalSeqLen, scale);
                }
            }
        }

        private static unsafe void AttentionDecodeKVHeadGroupedF16(int kvHead,
            float* qPtr, ushort* kPtr, ushort* vPtr, float* rPtr, float* scores,
            int headDim, int maxSeqLen, int groupSize, int totalSeqLen, float scale)
        {
            int hStart = kvHead * groupSize;
            ushort* kHead = kPtr + (long)kvHead * maxSeqLen * headDim;
            ushort* vHead = vPtr + (long)kvHead * maxSeqLen * headDim;

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
                float* row0 = scores + 0L * totalSeqLen;
                float* row1 = scores + 1L * totalSeqLen;
                float* row2 = scores + 2L * totalSeqLen;
                float* row3 = scores + 3L * totalSeqLen;

                for (int t = 0; t < totalSeqLen; t++)
                {
                    ushort* kT = kHead + (long)t * headDim;
                    float s0, s1, s2, s3;
                    TensorComputePrimitives.Dot4F32F16(qH0, qH1, qH2, qH3, kT, headDim,
                        out s0, out s1, out s2, out s3);
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
                    ushort* kT = kHead + (long)t * headDim;
                    for (int g = 0; g < groupSize; g++)
                    {
                        float* qH = qPtr + (long)(hStart + g) * headDim;
                        float s = TensorComputePrimitives.DotF32F16(qH, kT, headDim) * scale;
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

            // Aggregate V (F16): read V[t] once per t, scatter into all groupSize result heads.
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
                    ushort* vT = vHead + (long)t * headDim;
                    TensorComputePrimitives.ScaleAdd4F16(r0, r1, r2, r3, vT,
                        row0[t], row1[t], row2[t], row3[t], headDim);
                }
            }
            else
            {
                for (int t = 0; t < totalSeqLen; t++)
                {
                    ushort* vT = vHead + (long)t * headDim;
                    for (int g = 0; g < groupSize; g++)
                    {
                        float w = scores[g * totalSeqLen + t];
                        float* rH = rPtr + (long)(hStart + g) * headDim;
                        TensorComputePrimitives.ScaleAddF16(rH, vT, w, headDim);
                    }
                }
            }
        }

        protected static unsafe float* GetFloatPtr(Tensor t) =>
            TensorComputePrimitives.GetFloatPointer(t);

        private static IntPtr GetStoragePtr(Tensor t) =>
            TensorComputePrimitives.GetStoragePointer(t);

        private static IntPtr GetStorageBasePtr(Tensor t) =>
            TensorComputePrimitives.GetStorageBasePointer(t);

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

        /// <summary>
        /// Drop THIS model's device-resident weight copies while keeping the model itself
        /// usable: the GGUF mmap, the parsed weight table and the tokenizer all stay, so the
        /// next forward re-uploads from host memory instead of re-reading and re-parsing the
        /// file. The device cache is keyed by host pointer, so releasing entry-by-entry touches
        /// only this model — unlike <see cref="GgmlBasicOps.ClearHostBufferCache"/> (which
        /// <see cref="Dispose"/> calls), a process-global wipe that would also evict every
        /// OTHER live model's weights.
        ///
        /// Use this to hand VRAM back between phases of a multi-model pipeline. Disposing the
        /// model is the wrong tool there: it frees ~nothing extra on a file-backed model (the
        /// weights are mmap pages, not owned buffers) and costs a full reload next time.
        /// </summary>
        public void ReleaseGgmlDeviceResidency()
        {
            if (!IsGgmlBackend)
                return;

            OnBeforeReleaseGgmlDeviceResidency();

            foreach (Tensor t in _weights.Values)
            {
                if (t != null)
                    GgmlBasicOps.InvalidateHostBuffer(GetStoragePtr(t));
            }
            foreach (QuantizedWeight qw in _quantWeights.Values)
            {
                if (qw != null && qw.Data != IntPtr.Zero)
                    GgmlBasicOps.InvalidateHostBuffer(qw.Data);
            }
            foreach (StackedExpertWeights stacked in _stackedExpertWeights.Values)
            {
                if (stacked != null && stacked.Data != IntPtr.Zero)
                    GgmlBasicOps.InvalidateHostBuffer(stacked.Data);
            }
        }

        /// <summary>
        /// Architecture hook for releasing persistent native graphs before their
        /// weight bindings are evicted by <see cref="ReleaseGgmlDeviceResidency"/>.
        /// Implementations must preserve any device-authoritative mutable state
        /// needed to continue inference after the release.
        /// </summary>
        protected virtual void OnBeforeReleaseGgmlDeviceResidency()
        {
        }

        // ====================================================================
        // Forward / cache entry points.
        //
        // These are template methods: models implement the *Core variants; the
        // public methods add the multi-node tensor-parallel driver hook. When
        // this process is the distributed DRIVER (node 0 of a >1-node group,
        // after BeginDistributedDriver), each call first broadcasts its op +
        // tokens to the worker nodes so they run the identical forward pass in
        // lockstep (their per-layer AllReduces line up with the driver's).
        // On single-node and worker processes the hook is inert.
        // ====================================================================
        private const int TpControlForward       = 1;
        private const int TpControlForwardRefill = 2;
        private const int TpControlReset         = 3;
        private const int TpControlShutdown      = 4;
        private const int TpControlTruncate      = 5;

        private bool _distributedDriver;

        /// <summary>True when this process is a worker node in a multi-node TP group.</summary>
        public bool IsDistributedWorker =>
            _tpGroup != null && _tpGroup.NodeCount > 1 && _tpGroup.GlobalRankOffset > 0;

        /// <summary>
        /// Start acting as the distributed driver. After this call every
        /// Forward/ForwardRefill/ResetKVCache broadcasts to the worker nodes.
        /// Call once, on the driver node (global rank offset 0), AFTER
        /// <see cref="WarmUpKernels"/> (warmup runs symmetrically on every node
        /// and must not broadcast).
        /// </summary>
        public void BeginDistributedDriver()
        {
            if (_tpGroup != null && _tpGroup.NodeCount > 1 && _tpGroup.GlobalRankOffset == 0)
                _distributedDriver = true;
        }

        public float[] Forward(int[] tokens)
        {
            if (_distributedDriver) _tpGroup.BroadcastControl(TpControlForward, tokens);
            return ForwardCore(tokens);
        }

        public float[] ForwardRefill(int[] tokens)
        {
            if (_distributedDriver) _tpGroup.BroadcastControl(TpControlForwardRefill, tokens);
            return ForwardRefillCore(tokens);
        }

        public void ResetKVCache()
        {
            if (_distributedDriver) _tpGroup.BroadcastControl(TpControlReset, Array.Empty<int>());
            ResetKVCacheCore();
        }

        protected abstract float[] ForwardCore(int[] tokens);
        protected virtual float[] ForwardRefillCore(int[] tokens) => ForwardCore(tokens);
        protected abstract void ResetKVCacheCore();

        /// <summary>
        /// Worker-node event loop for multi-node tensor parallelism. Blocks,
        /// mirroring the driver's op stream (forward / refill / reset) so this
        /// node contributes its weight shards to every AllReduce, until the
        /// driver broadcasts shutdown (or the connection drops). Returns
        /// immediately on single-node groups.
        /// </summary>
        public void RunDistributedWorkerLoop()
        {
            if (_tpGroup == null || _tpGroup.NodeCount <= 1)
                return;

            Console.WriteLine(
                $"[TP worker] ready — mirroring driver forward passes (global rank offset {_tpGroup.GlobalRankOffset}).");

            try
            {
                while (true)
                {
                    var (op, payload) = _tpGroup.ReceiveControl();
                    switch (op)
                    {
                        case TpControlForward:       ForwardCore(payload); break;
                        case TpControlForwardRefill: ForwardRefillCore(payload); break;
                        case TpControlReset:         ResetKVCacheCore(); break;
                        case TpControlTruncate:      TruncateKVCacheCore(payload.Length > 0 ? payload[0] : 0); break;
                        case TpControlShutdown:
                            Console.WriteLine("[TP worker] shutdown received; exiting worker loop.");
                            return;
                        default:
                            throw new InvalidOperationException($"Unknown TP control op {op}.");
                    }
                }
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[TP worker] loop ended: {ex.Message}");
            }
        }

        /// <summary>
        /// Driver-side shutdown: tell worker nodes to leave their loops. Safe to
        /// call when not the driver (no-op). Invoked from Dispose so every exit
        /// path releases the workers.
        /// </summary>
        private void SignalDistributedWorkersShutdown()
        {
            if (!_distributedDriver) return;
            try { _tpGroup.BroadcastControl(TpControlShutdown, Array.Empty<int>()); }
            catch { /* workers may already be gone; ignore */ }
            _distributedDriver = false;
        }

        // Pipelined greedy decode (overridden by models that support it,
        // e.g. Qwen35Model on MLX). When SupportsPipelinedGreedy is true,
        // the inference loop can call SubmitGreedyDecodeStep to issue a
        // decode forward that returns its predicted token as a [1] int32
        // device tensor (host-readable via Tensor.GetElementsAsInt). This
        // lets the loop queue the next step before host-syncing the
        // current one — overlapping the LM-head sync wait with the next
        // forward's first kernels.
        public virtual bool SupportsPipelinedGreedy => false;
        public virtual Tensor SubmitGreedyDecodeStep(int? firstTokenForBegin)
        {
            throw new NotSupportedException(
                $"{GetType().Name} does not implement SubmitGreedyDecodeStep.");
        }
        public virtual void ResetPipelinedGreedyState() { }

        /// <summary>
        /// Run a tiny forward pass to force lazy kernel compilation (Metal pipelines,
        /// CUDA JIT, memory pool warm-up, etc.) so the first real inference request
        /// doesn't pay the compilation cost.  Resets KV cache and timing counters
        /// afterwards so the warmup is invisible to callers.
        /// </summary>
        public virtual void WarmUpKernels()
        {
            if (_backend == BackendType.Mlx && !IsMlxKernelWarmupEnabled())
            {
                long nativeBytes = MlxNativePreloadableQuantizedBytes();
                Console.WriteLine(
                    $"  Skipping MLX kernel warmup by default ({nativeBytes / 1024 / 1024} MB of resident quantized weights). Set TS_MLX_KERNEL_WARMUP=1 to force it.");
                ResetForwardTiming();
                return;
            }

            if (HasMlxHostFallbackQuantizedWeights())
            {
                long fallbackBytes = MlxHostFallbackQuantizedBytes();
                Console.WriteLine(
                    $"  Skipping MLX kernel warmup: {fallbackBytes / 1024 / 1024} MB of quantized weights use GGUF row-dequant fallback.");
                ResetForwardTiming();
                return;
            }

            int safeToken = (Config?.VocabSize ?? 0) > 1 ? 1 : 0;

            if (_tpGroup != null && _tpGroup.NodeCount > 1)
            {
                Console.WriteLine("  Waiting for all nodes to finish loading...");
                _tpGroup.Barrier();
            }

            Console.WriteLine("  Warming up kernels (decode + prefill)...");

            long decodeStart = Stopwatch.GetTimestamp();
            Forward(new[] { safeToken });
            ResetKVCache();
            double decodeMs = (Stopwatch.GetTimestamp() - decodeStart) * 1000.0 / Stopwatch.Frequency;
            Console.WriteLine($"    Decode warmup: {decodeMs:F1} ms");

            // Prime the MULTI-TOKEN prefill path. The 1-token Forward above only
            // warms the decode-shaped graph; on CUDA/GGML a real prompt takes the
            // fused whole-model prefill ("verify") graph + flash-attention kernels,
            // whose first build/capture and gallocr reservation otherwise lands on
            // the first real request and inflates its TTFT by ~50-110 ms (measured
            // on Gemma 4 E4B: first long prompt 906 ms cold -> 796 ms warm). Short
            // prompts ("Hello", the typical discarded warmup) skip the fused path,
            // so prime it with a longer dummy prompt here. Previously this ran for
            // MLX only. Guarded so a model that dislikes a dummy refill can never
            // block startup; disable entirely via TS_PREFILL_WARMUP=0.
            if (!string.Equals(Environment.GetEnvironmentVariable("TS_PREFILL_WARMUP"), "0", StringComparison.Ordinal))
            {
                // MLX, Metal, and the managed CPU backend stay conservative (short
                // prompt); discrete CUDA/Vulkan GGML paths use a longer one to reach
                // the fused-verify prefill path that short prompts bypass. The long
                // warmup only pays off where a first real prompt builds/captures a
                // fused whole-model prefill graph
                // and reserves a gallocr (CUDA/GGML) -- the managed CPU backend has
                // none of that, so a 1024-token prefill there is pure cost: tens of
                // seconds of GEMM plus large O(N^2) activation/KV first-touch paging
                // that makes `--backend cpu` look hung at startup for no benefit.
                // TS_PREFILL_WARMUP_LEN overrides (e.g. to pre-size the prefill
                // gallocr on roomy GPUs); a larger value does NOT reliably help a
                // near-full GPU because the legacy ForwardRefill warmup graph sizes
                // the reused gallocr differently than the engine prefill graph.
                // Integrated GPUs (unified-memory iGPUs: Intel UHD / AMD APU via
                // ggml-vulkan, Tegra via ggml-cuda) are memory-bandwidth bound and
                // run the fused multi-token prefill an order of magnitude slower than
                // a discrete GPU. A 2048-token verify-prefill warmup there takes
                // MINUTES (measured ~6+ min on Intel UHD with Qwen3.6-27B), during
                // which the server prints "Startup model loaded" then appears hung
                // before it ever starts listening. They also get no CUDA-graph
                // capture benefit from the long warmup. Treat them like MLX/CPU: a
                // short warmup that still primes the fused-verify graph once, cheaply.
                bool integratedGpu = IsIntegratedGgmlGpu();
                if (integratedGpu)
                {
                    Console.WriteLine(BuildIntegratedGpuWarning());
                }
                // Native CUDA models whose weights are mostly a CUDA-unsupported quant
                // type never become GPU-resident: their matmuls dequantize on the CPU,
                // and a 2048-token prefill warmup then pegs every core for MINUTES while
                // the server prints "Startup model loaded" and looks hung (observed on a
                // Qwen3.5-9B IQ4_XS build before IQ4_XS residency was added). Detect that
                // up front and use the short warmup instead so startup stays responsive;
                // the inference itself is still slow, so point the operator at a
                // supported quant / backend.
                double hostBackedFrac = HostBackedQuantWeightFraction();
                bool mostlyHostBacked = hostBackedFrac >= 0.5;
                if (mostlyHostBacked)
                {
                    Console.WriteLine(
                        $"  {hostBackedFrac * 100:F0}% of quantized weights use a CUDA-unsupported quant type and stay host-backed (CPU matmul); using a lightweight startup warmup. Inference will be slow — prefer a quant the direct CUDA backend supports (Q4_0/Q4_K/Q5_K/Q6_K/Q8_0/IQ2/IQ3/IQ4_XS) or run with --backend ggml_cuda.");
                }
                // GGML CUDA/Vulkan build/capture a fused whole-model prefill graph and
                // pre-grow its gallocr. Native CUDA has no captured verify-prefill
                // graph, so its architecture default remains a lightweight 32-token
                // shape. A model may opt into a larger direct-CUDA warmup when it has
                // persistent activation/QMM scratch that is expensive to grow during
                // the first real request. Hybrid GatedDeltaNet models intentionally
                // retain the lightweight base default because a 2K warmup can take minutes.
                // Multi-token prefill of an MoE model under tensor parallelism is
                // slow enough that the full 2048-token warmup dominates startup —
                // measured at ~61 s on Gemma 4 26B A4B with --tp 2, during which
                // the process looks hung. Same treatment as the other known-slow
                // prefill configurations: warm the path cheaply and say why.
                bool moeUnderTp = MoEUnderTpIsSlow;
                if (moeUnderTp)
                {
                    Console.WriteLine(
                        "  MoE under tensor parallelism: using a lightweight startup warmup (the full one costs ~a minute here). " +
                        "Long-prompt prefill stays slower than a single GPU; TP is for models that do not fit on one.");
                }

                bool conservativeWarmup = UsesLightweightPrefillWarmupByDefault(_backend)
                    || integratedGpu || mostlyHostBacked || moeUnderTp;
                // 2048 matches ComputePrefillChunkSize, so the warmup runs ONE
                // fused verify chunk at the largest legacy-chunk shape: the shared
                // reuse-gallocr is pre-grown (and its device memory first-touched)
                // for every prompt up to 2048 tokens, which covers typical chat
                // prompts. Measured on gemma4-12B/Vulkan: first ~2k-token request
                // was ~300-500 ms slower than warm (gallocr growth + residency)
                // with the old 1024 warmup; with 2048 it starts warm.
                int? explicitWarmupLengthValue = null;
                {
                    string wl = Environment.GetEnvironmentVariable("TS_PREFILL_WARMUP_LEN");
                    if (!string.IsNullOrEmpty(wl) && int.TryParse(wl, out int wlv) && wlv >= 2)
                        explicitWarmupLengthValue = wlv;
                }
                int warmupLength = ResolvePrefillWarmupTargetLength(
                    _backend,
                    integratedGpu,
                    mostlyHostBacked,
                    moeUnderTp,
                    NativeCudaPrefillWarmupLength,
                    explicitWarmupLengthValue);
                bool explicitWarmupLength = explicitWarmupLengthValue.HasValue;
                int warmupTokenOverhead =
                    _backend == BackendType.Cuda && !conservativeWarmup
                        ? NativeCudaPrefillWarmupTokenOverhead
                        : 0;
                warmupLength = ResolvePrefillWarmupInputLength(
                    warmupLength,
                    MaxContextLength,
                    warmupTokenOverhead,
                    explicitWarmupLength);

                int[] warmupPrompt = new int[warmupLength];
                Array.Fill(warmupPrompt, safeToken);

                try
                {
                    Console.WriteLine($"    Prefill warmup ({warmupLength} tokens): starting...");
                    long prefillStart = Stopwatch.GetTimestamp();
                    ForwardRefill(warmupPrompt);
                    Forward(new[] { safeToken });
                    double prefillMs = (Stopwatch.GetTimestamp() - prefillStart) * 1000.0 / Stopwatch.Frequency;
                    Console.WriteLine($"    Prefill warmup ({warmupLength} tokens): completed in {prefillMs:F1} ms");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"  Prefill warmup skipped: {ex.GetType().Name}: {ex.Message}");
                }
                finally
                {
                    ResetKVCache();
                }

                if (_backend == BackendType.Cuda &&
                    NativeCudaPrimeShortDecodeGraphAfterPrefill &&
                    CudaPrefillGraphCache.Enabled &&
                    CudaPrefillGraphCache.DecodeEnabled)
                {
                    try
                    {
                        // The one-token pass at the beginning of WarmUpKernels
                        // registered this graph key once. If the long prefill used
                        // a different attention tier, this second sighting captures
                        // the short route instead of charging the first request.
                        Forward(new[] { safeToken });
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"  Short decode-graph warmup skipped: {ex.GetType().Name}: {ex.Message}");
                    }
                    finally
                    {
                        ResetKVCache();
                    }
                }
            }

            WarmUpMultimodalKernels();
            ResetForwardTiming();
        }

        public virtual void WarmUpMultimodalKernels()
        {
        }

        /// <summary>
        /// True when the active GGML device (ggml_cuda / ggml_vulkan) is an
        /// integrated, unified-memory GPU (Intel UHD / AMD APU / Tegra). Such
        /// devices are memory-bandwidth bound and cannot afford the heavy
        /// multi-token prefill warmup; startup warmup falls back to the short
        /// conservative path for them. Only ggml backends are queried; every other
        /// backend (and any query failure) reports false.
        /// </summary>
        private bool IsIntegratedGgmlGpu()
        {
            if (_backend != BackendType.GgmlCuda && _backend != BackendType.GgmlVulkan)
                return false;
            try { return GgmlBasicOps.IsActiveDeviceIntegrated(); }
            catch { return false; }
        }

        /// <summary>
        /// Builds the prominent startup banner shown when inference is running on an
        /// integrated, unified-memory GPU. A 27B model on an Intel UHD iGPU decodes at
        /// ~0.7 tok/s (measured, and identical to llama.cpp on the same device) versus
        /// ~15 tok/s on a discrete RTX 3080 — a ~20x gap that is purely hardware, not a
        /// software regression. The single-line notice this replaces was easy to miss in
        /// the startup log, so operators kept running large models on the iGPU by mistake.
        /// On ggml_vulkan (multiple adapters are enumerable) the banner names the selected
        /// adapter and lists every other device with the exact <c>--gpu-device N</c> flag to
        /// switch to it. Enumeration failures / single-device hosts (e.g. Tegra via
        /// ggml_cuda) fall back to the generic guidance.
        /// </summary>
        private string BuildIntegratedGpuWarning()
        {
            const string sep = "  ==============================================================================";
            var sb = new System.Text.StringBuilder();
            sb.AppendLine(sep);
            sb.AppendLine("  PERFORMANCE WARNING: inference is running on an INTEGRATED GPU.");
            sb.AppendLine("  Integrated (unified-memory) GPUs are memory-bandwidth bound: large models run");
            sb.AppendLine("  roughly an order of magnitude slower here than on a discrete GPU, and often");
            sb.AppendLine("  slower than the CPU backend. This is a hardware limit, not a TensorSharp issue.");

            bool namedAlternative = false;
            if (_backend == BackendType.GgmlVulkan)
            {
                try
                {
                    int selected = 0;
                    string sel = Environment.GetEnvironmentVariable(GgmlBasicOps.VulkanDeviceEnvVar);
                    if (!string.IsNullOrEmpty(sel) && int.TryParse(sel, out int s) && s >= 0)
                        selected = s;

                    int count = GgmlBasicOps.GetVulkanDeviceCount();
                    if (count > 0 && selected < count)
                        sb.AppendLine($"  Selected:  --gpu-device {selected}   {GgmlBasicOps.GetVulkanDeviceDescription(selected)}");

                    var others = new System.Collections.Generic.List<string>();
                    for (int i = 0; i < count; i++)
                    {
                        if (i == selected) continue;
                        others.Add($"    --gpu-device {i}   {GgmlBasicOps.GetVulkanDeviceDescription(i)}");
                    }
                    if (others.Count > 0)
                    {
                        sb.AppendLine("  For full performance re-run against a discrete GPU (see --list-gpus):");
                        foreach (var o in others) sb.AppendLine(o);
                        namedAlternative = true;
                    }
                }
                catch { /* enumeration unavailable — fall through to generic guidance */ }
            }
            if (!namedAlternative)
                sb.AppendLine("  For full performance use a discrete GPU (--gpu-device <index>, see --list-gpus), or --backend ggml_cpu.");
            sb.Append(sep);
            return sb.ToString();
        }

        /// <summary>
        /// Fraction (0..1) of quantized weight bytes whose quant type the direct
        /// CUDA backend cannot make GPU-resident (<see cref="CudaQuantizedOps.SupportsQuantizedType"/>).
        /// Such weights stay host-backed and their matmuls dequantize on the CPU, so a
        /// large fraction means a multi-token prefill (including the startup warmup)
        /// runs CPU-bound. Only the native <see cref="BackendType.Cuda"/> backend is
        /// evaluated (GGML backends upload weights to the device themselves); every
        /// other backend reports 0. Computed from the quant TYPE, not the live
        /// host-data flag, so it is unaffected by TS_GGML_RETAIN_HOST_WEIGHTS.
        /// </summary>
        private double HostBackedQuantWeightFraction()
        {
            if (_backend != BackendType.Cuda || _quantWeights.Count == 0)
                return 0.0;
            long total = 0, hostBacked = 0;
            foreach (QuantizedWeight qw in _quantWeights.Values)
            {
                total += qw.RawBytes;
                if (!CudaQuantizedOps.SupportsQuantizedType(qw.GgmlType))
                    hostBacked += qw.RawBytes;
            }
            return total > 0 ? (double)hostBacked / total : 0.0;
        }

        private static bool IsMlxKernelWarmupEnabled()
        {
            return string.Equals(Environment.GetEnvironmentVariable("TS_MLX_KERNEL_WARMUP"), "1", StringComparison.Ordinal);
        }

        /// <summary>
        /// Reset the cumulative forward-pass timing counters used by
        /// <see cref="PrintTimingStats"/>. Useful when a benchmark driver wants
        /// to discard the cost of one or more warm-up inference passes (Metal
        /// pipeline JIT for new batch sizes, allocator pool growth, etc.) so
        /// only the timed run contributes to reported numbers.
        /// </summary>
        public void ResetForwardTiming()
        {
            _linearTicks = 0;
            _attnTicks = 0;
            _normTicks = 0;
            _embTicks = 0;
            _lmHeadTicks = 0;
            _logitsCopyTicks = 0;
            _forwardCount = 0;
            _forwardSw.Reset();
        }

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
        public void TruncateKVCache(int tokenCount)
        {
            if (_distributedDriver) _tpGroup.BroadcastControl(TpControlTruncate, new[] { tokenCount });
            TruncateKVCacheCore(tokenCount);
        }

        protected virtual void TruncateKVCacheCore(int tokenCount)
        {
            Console.WriteLine($"[KV cache] Truncating from {_cacheSeqLen} to {tokenCount}");
            _cacheSeqLen = tokenCount;
        }

        /// <summary>
        /// Process-wide GPU-compute serialisation lock. The GGML Metal backend
        /// is *not* thread-safe (concurrent <c>ggml_backend_graph_compute</c>
        /// from two threads will silently corrupt the command queue and
        /// eventually <c>ggml_abort</c> on a command-buffer status=1/2).
        /// Every callsite that drives the backend through this model must
        /// take this lock for the duration of the GPU work.
        ///
        /// Today that's two callers: the InferenceEngine's worker thread
        /// (around its per-step ForwardBatch / Forward) and
        /// ChatGenerationPipeline (around the multimodal vision/audio
        /// encoder it invokes at prompt-prep time, which runs many GGML ops
        /// of its own). Without the lock, a parallel image-bearing request
        /// arriving while the engine is mid-batch races the engine's
        /// command buffer and aborts the process.
        /// </summary>
        public object GpuComputeLock { get; } = new object();

        /// <summary>
        /// Release <see cref="GpuComputeLock"/>, briefly yield, then re-acquire.
        /// Designed to be called from inside a <c>lock(model.GpuComputeLock)</c>
        /// scope by long-running GPU operations (vision / audio / video
        /// encoders' per-block loops) so the engine worker's
        /// <see cref="BatchExecutor.ExecuteStep"/> can grab the lock and run
        /// one inference step in between. Without this, an encoder forward
        /// holds the lock for its full duration (100ms–several seconds)
        /// and all in-flight decode requests appear frozen.
        ///
        /// The <see cref="Monitor"/> wait queue is FIFO-ish on .NET, so a
        /// waiting engine-worker thread typically wins the re-acquisition
        /// race; the caller then blocks at <see cref="Monitor.Enter(object)"/>
        /// until the engine's step completes. Net effect: encoder layer
        /// → engine step → encoder layer → engine step → … instead of
        /// encoder-blocks-everyone.
        ///
        /// Costs the encoder some wall-clock time per yield (one engine
        /// step ≈ 50–200ms), but in exchange the engine never stalls. If
        /// the engine has no work pending, the yield is a cheap no-op
        /// (Monitor.Exit, Sleep(0), Monitor.Enter).
        /// </summary>
        public void YieldGpuComputeLock()
        {
            // Allow disabling via env var for A/B testing or troubleshooting.
            if (string.Equals(Environment.GetEnvironmentVariable("TS_ENCODER_YIELD"), "0", StringComparison.Ordinal))
                return;
            try { Monitor.Exit(GpuComputeLock); }
            catch (SynchronizationLockException) { return; } // not held — nothing to yield
            try
            {
                System.Threading.Thread.Sleep(0);
            }
            finally
            {
                Monitor.Enter(GpuComputeLock);
            }
        }

        /// <summary>
        /// Whether this architecture exposes block-level snapshot / restore for use
        /// by the paged KV cache. Default: not supported. Pure-attention models
        /// opt in by overriding alongside the four members below.
        /// </summary>
        public virtual bool SupportsKVStateSnapshot => false;

        /// <summary>
        /// Whether a K/V snapshot taken by one sequence can be re-injected into another
        /// sequence's fresh cache (cross-request prefix reuse + executor ownership swap).
        /// Defaults to <see cref="SupportsKVStateSnapshot"/>; models whose snapshot
        /// restore does not faithfully reproduce a fresh prefill (e.g. sliding-window /
        /// circular caches) override this to false to force a correct re-prefill.
        /// </summary>
        public virtual bool SupportsCrossSequenceKvReuse => SupportsKVStateSnapshot;

        /// <summary>
        /// Maximum leading-prompt-token count whose K/V snapshot can be faithfully
        /// restored into another sequence. Defaults to unbounded; sliding-window models
        /// (e.g. Gemma 4) override this with their window size.
        /// </summary>
        public virtual int MaxReusablePrefixTokens => int.MaxValue;

        /// <summary>
        /// Stable identifier tying snapshots to a specific (model, layer count,
        /// head counts, head dim, KV dtype) tuple. The paged cache stores blocks
        /// keyed by SHA-256 chain over this fingerprint, so changing it
        /// effectively invalidates the cache for the previous model variant.
        /// </summary>
        public virtual string KVStateFingerprint => string.Empty;

        /// <summary>
        /// Bytes occupied by a block of <paramref name="tokenCount"/> tokens worth
        /// of K/V state across all layers, or 0 when snapshotting is unsupported.
        /// </summary>
        public virtual long ComputeKVBlockByteSize(int tokenCount) => 0;

        /// <summary>
        /// Whether this architecture must capture state at every block boundary
        /// during prefill (true for models with recurrent / SSM layers whose state
        /// is a function of all preceding tokens). See <see cref="IModelArchitecture"/>.
        /// </summary>
        public virtual bool RequiresPerBlockCapture => false;

        /// <summary>
        /// Copy bytes for token positions <c>[startToken, startToken+tokenCount)</c>
        /// into <paramref name="destination"/>. Returns false if the range is not
        /// valid or the model does not support snapshots. See <see cref="IModelArchitecture"/>.
        /// </summary>
        public virtual bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination) => false;

        /// <summary>
        /// Write a block of K/V bytes at token position <paramref name="destToken"/>.
        /// After a successful call the model behaves as if <paramref name="tokenCount"/>
        /// tokens had been forwarded into the cache at that position. See
        /// <see cref="IModelArchitecture"/>.
        /// </summary>
        public virtual bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source) => false;

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
            {
                // The greedy shortcut skips TokenSampler, so the grammar mask has
                // to be applied here too. Structured output is very often run at
                // temperature 0, which is exactly this branch -- leaving it out
                // would make constrained decoding silently do nothing in the case
                // that needs it most.
                var g = config?.Grammar;
                if (g != null && !g.IsDead)
                    g.ApplyMask(logits, allowEos: g.IsComplete);
                return SampleGreedy(logits);
            }
            var sampler = new TokenSampler(config);
            return sampler.Sample(logits, generatedTokenIds);
        }

        public virtual void Dispose()
        {
            // Release any distributed worker nodes before tearing down the TP
            // group, so every driver exit path (normal or exception) lets the
            // workers leave their loops cleanly.
            SignalDistributedWorkersShutdown();

            if (MultimodalInjector is IDisposable multimodalInjector)
                multimodalInjector.Dispose();

            foreach (var w in _weights.Values)
                w.Dispose();
            _weights.Clear();

            if (IsGgmlBackend)
            {
                // Clear offloadable registrations FIRST so they don't outlive
                // the host pointers (which become invalid once the GgufFile
                // mmap below is disposed). ClearHostBufferCache then frees the
                // MTLBuffer wrappers; the LRU state goes with it.
                GgmlBasicOps.ClearOffloadableState();
                GgmlBasicOps.ClearHostBufferCache();
            }

            if (_backend == BackendType.Cuda && _allocator is CudaAllocator cudaAllocator)
            {
                foreach (var qw in _quantWeights.Values)
                    CudaQuantizedOps.ReleaseQuantizedWeight(cudaAllocator, qw.CacheKey);
                // Weights are suballocated from arena slabs; free the slabs now
                // that every resident weight has been released (individual
                // releases only drop cache entries — the slabs are freed here).
                CudaQuantizedOps.ReleaseArena(cudaAllocator);
            }

            if (_backend == BackendType.Mlx && _allocator is MlxAllocator mlxAllocator)
            {
                foreach (var qw in _quantWeights.Values)
                    MlxQuantizedOps.ReleaseQuantizedWeight(mlxAllocator, qw.CacheKey);
            }

            foreach (var qw in _quantWeights.Values)
                qw.Dispose();
            _quantWeights.Clear();

            // Free any owned bulk buffers backing stacked-experts views (only
            // populated by the non-mmap path in LoadWeights). External-view
            // entries that point into the GgufFile mmap have OwnedBuffer == 0
            // and are released when the GgufFile itself is disposed below.
            foreach (var stacked in _stackedExpertWeights.Values)
            {
                if (stacked.OwnedBuffer != IntPtr.Zero)
                    QuantizedWeight.FreeBuffer(stacked.OwnedBuffer);
            }
            _stackedExpertWeights.Clear();

            _gguf?.Dispose();

            // Dispose TP sharded weights.
            foreach (var shards in _tpQuantWeights.Values)
                foreach (var qw in shards) qw?.Dispose();
            _tpQuantWeights.Clear();

            foreach (var shards in _tpWeights.Values)
                foreach (var w in shards) w?.Dispose();
            _tpWeights.Clear();

            _tpGroup?.Dispose();

            if (_allocator is IDisposable allocatorDisposable)
                allocatorDisposable.Dispose();
        }

        /// <summary>
        /// Native tunables that must be set BEFORE the compute backend spins up,
        /// because they are read once when the device is probed. Called from
        /// <see cref="Create"/> with the architecture already known from the GGUF
        /// header but no backend initialised yet.
        /// </summary>
        private static void ApplyArchitectureNativeTunables(string arch, BackendType backend, GgufFile probe)
        {
            if (backend != BackendType.GgmlMetal)
                return;
            if (arch is not ("wan" or "wan2.1" or "wan2.2"))
                return;

            // Wan renders NaN - uniformly black frames - on Metal whenever ggml
            // routes mul_mm through the Metal 4 tensor API. Diagnosis (2026-08-13):
            // the corruption is confined to the VAE's conv GEMMs. The tensor-API
            // mul_mm intermittently misreads operand columns there on the FIRST
            // pass over a graph (M5, macOS 26.6) - buffer-layout dependent (32x32
            // latents failed while 33x33 packed differently and passed),
            // first-run-only (recomputes read back the previous pass's bytes and
            // look clean), and immune to every kill switch (fusion / concurrency /
            // graph-optimize off, even serialized 64-node slices with full syncs).
            // The same GEMMs are bit-correct in isolation and LLM/DiT-style GEMMs
            // never corrupt - an upstream ggml-metal/driver defect, and has_tensor
            // is a device-level property fixed at init, so the kernel choice
            // cannot be scoped per-op.
            //
            // Re-verified 2026-08-17 that this is NOT yet fixed, and how easy it is
            // to conclude otherwise: an isolated WanVideoBench decode of synthetic
            // latents at five shapes - including the 32x32 all-NaN repro - matched
            // the non-tensor decode to 91-93 dB PSNR with no NaN. The next full
            // 1088x832x121f generation with the tensor API on produced 121 BLACK
            // frames. Only an end-to-end video decides this.
            //
            // With the tensor API on, the VAE stays CORRECT because
            // ggml_ops_wan.cpp routes its convs through ggml_conv_2d_direct on
            // tensor-API devices (wan_vae_gemm_budget) - slower than im2col+GEMM,
            // but a FIXED per-video cost, while the tensor API's DiT speedup
            // scales with step count and model size. Measured at 480x480x9f,
            // 6 steps, M5 Pro:
            //   A14B I2V: 17.1 vs 30.2 s/step (1.77x); VAE enc+dec 135s vs 19s
            //             -> break-even ~9 steps; the 40-step recipe is ~33% faster
            //             with the tensor API (~13.7 vs ~20.5 min).
            //   TI2V-5B:  1.6 vs 2.9 s/step; VAE decode 179s vs 13s
            //             -> break-even ~128 steps; never wins.
            // So the default follows the DiT class: enabled for A14B/14B-class
            // models (patch_embedding oc >= 5120), disabled for smaller ones.
            // TS_WAN_METAL_TENSOR_API=1/0 forces either way. When upstream fixes
            // the tensor-API mul_mm, enable it everywhere and drop the direct-conv
            // carve-out - and confirm with a full video, not a decode probe.
            long dim = 0;
            if (probe.Tensors.TryGetValue("patch_embedding.weight", out var patch) && patch.Shape.Length >= 5)
                dim = (long)patch.Shape[4];

            string env = Environment.GetEnvironmentVariable("TS_WAN_METAL_TENSOR_API");
            bool enable = string.Equals(env, "1", StringComparison.Ordinal) ||
                          (!string.Equals(env, "0", StringComparison.Ordinal) && dim >= 5120);
            if (enable)
            {
                Console.WriteLine("  [wan] Metal 4 tensor API enabled (~1.8x faster DiT steps; the VAE runs " +
                                  "on the slower-but-correct direct-conv path; TS_WAN_METAL_TENSOR_API=0 opts out).");
                return;
            }

            try
            {
                if (TensorSharp.GGML.GgmlBasicOps.SetNativeEnvironmentVariable(
                        "GGML_METAL_TENSOR_DISABLE", "1", overwrite: false))
                {
                    Console.WriteLine("  [wan] Metal 4 tensor API disabled for this process (fastest correct " +
                                      "config for this model class; TS_WAN_METAL_TENSOR_API=1 opts in: faster " +
                                      "DiT steps but a slower direct-conv VAE).");
                }
            }
            catch (DllNotFoundException) { /* managed-only host; nothing to configure */ }
            catch (EntryPointNotFoundException) { /* older GgmlOps without the setter */ }
        }

        /// <param name="draftModelPath">Optional speculative-decoding draft
        /// model (DeepSeek V4's DSpark support GGUF); ignored by architectures
        /// that have no drafter.</param>
        public static ModelBase Create(string ggufPath, BackendType backend, int tpDegree = 1, ITensorParallelGroup tpGroup = null,
            string draftModelPath = null)
        {
            if (tpGroup == null && tpDegree <= 1)
            {
                string envTp = Environment.GetEnvironmentVariable("TENSORSHARP_TP_DEGREE");
                if (int.TryParse(envTp, out int envTpDegree) && envTpDegree > 1)
                    tpDegree = envTpDegree;
            }

            using var probe = new GgufFile(ggufPath);
            string arch = probe.GetString("general.architecture") ?? "qwen3";

            ApplyArchitectureNativeTunables(arch, backend, probe);

            return arch switch
            {
                // qwen2vl is Qwen2/Qwen2.5-VL. Its language model is Qwen3's block
                // with a QKV bias and no QK norm, both of which Qwen3Model detects
                // from the weights. Text-only chat: M-RoPE degenerates to standard
                // RoPE when the t/h/w position components are equal, which they are
                // for text tokens, so the vision tower (mmproj) is not required.
                "qwen3" or "qwen2" or "qwen2vl" or "qwen2_vl" => new Qwen3Model(ggufPath, backend, tpDegree, tpGroup),
                "qwen35" or "qwen35moe" or "qwen3next" => new Qwen35Model(ggufPath, backend, tpDegree, tpGroup),
                "gemma3" => new Gemma3Model(ggufPath, backend, tpDegree, tpGroup),
                "gemma4" => new Gemma4Model(ggufPath, backend, tpDegree, tpGroup),
                "diffusion-gemma" or "diffusion_gemma" => new DiffusionGemmaModel(ggufPath, backend),
                "qwen_image" or "qwen-image" => new QwenImage.QwenImageModel(ggufPath, backend),
                "wan" or "wan2.1" or "wan2.2" => new WanVideo.WanVideoModel(ggufPath, backend),
                "gptoss" or "gpt-oss" => new GptOssModel(ggufPath, backend, tpDegree, tpGroup),
                "nemotron_h" or "nemotron_h_moe" => new NemotronModel(ggufPath, backend, tpDegree, tpGroup),
                "mistral3" => new Mistral3Model(ggufPath, backend, tpDegree, tpGroup),
                "muse-glimmer" or "muse_glimmer" => new MuseGlimmerModel(ggufPath, backend, tpDegree, tpGroup, draftModelPath),
                "deepseek4" => new DeepSeek4Model(ggufPath, backend, tpDegree, tpGroup, draftModelPath),
                // GLM-5.x with DeepSeek Sparse Attention (MLA + lightning indexer + sigmoid MoE).
                "glm-dsa" or "glm_dsa" => new GlmDsaModel(ggufPath, backend, tpDegree, tpGroup),
                _ => throw new NotSupportedException($"Unsupported architecture: {arch}"),
            };
        }
    }
}
