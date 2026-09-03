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
using TensorSharp.Models.Architecture;
using TensorSharp.Cuda;
using TensorSharp.GGML;
using TensorSharp.MLX;

namespace TensorSharp.Models
{

    public abstract partial class ModelBase : IModelArchitecture
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

        /// <summary>Per-tensor weight scale of a SHARDED weight, keyed by the
        /// name the TP linears look it up by. Recorded when the shards are cut,
        /// because the shard itself may be a plain F32 <see cref="Tensor"/> with
        /// nowhere to carry it. The scalar is shard-invariant, so one value
        /// covers every rank (see the TP linears for why).</summary>
        protected readonly Dictionary<string, float> _tpWeightScales = new();

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
            RefuseUnsupportedBlockQuantizedKvCache();
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

        /// <summary>
        /// Whether every path this architecture can take can READ a
        /// block-quantized (Q8_0 / Q4_0) K/V cache.
        ///
        /// A block-quantized cache is only ever readable by kernels that know the
        /// block layout. An architecture whose fused kernels decline the type AND
        /// whose managed fallback walks the cache as a flat float buffer cannot
        /// honour <c>--kv-cache-dtype q8_0</c> on any path, and must say so here:
        /// otherwise the request survives model construction and dies much later
        /// as an unhandled "Requires a Float32 tensor" deep inside the first
        /// forward pass - which, because kernel warm-up is the first forward,
        /// means the process aborts before it has generated a single token.
        /// Default true; override to false for a family that cannot.
        /// </summary>
        protected virtual bool SupportsBlockQuantizedKvCache => true;

        /// <summary>
        /// Downgrade an explicitly requested block-quantized KV cache to F16 on
        /// architectures that cannot read one (see
        /// <see cref="SupportsBlockQuantizedKvCache"/>), with a message on stderr
        /// naming the substitution. F16 is the right substitute rather than F32:
        /// it is what <see cref="KvCacheDtypeConfig.ApplyModelDtypeDefault"/>
        /// would have chosen for any quantized model, so the operator gets the
        /// cache the model would have used anyway instead of an abort.
        /// </summary>
        private void RefuseUnsupportedBlockQuantizedKvCache()
        {
            if (SupportsBlockQuantizedKvCache) return;
            KvCacheDtype requested = KvCacheDtypeConfig.Current;
            if (requested != KvCacheDtype.Q8_0 && requested != KvCacheDtype.Q4_0) return;

            Console.Error.WriteLine(
                $"[kv-cache] {GetType().Name} cannot read a {requested.ToShortString()} K/V cache "
                + "on any of its attention paths; using f16 instead.");
            KvCacheDtypeConfig.Set(KvCacheDtype.F16);
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

        /// <summary>
        /// Number of GPUs this model spreads its LAYERS across (1 = single GPU).
        ///
        /// This is llama.cpp's <c>--split-mode layer</c>, not tensor parallelism:
        /// each GPU owns a contiguous run of whole layers, nothing is sharded and
        /// no collective is ever issued - only the residual crosses a device
        /// boundary, and it does so through host memory. It is a CAPACITY feature
        /// (measured on 2xA100 with llama.cpp: +10% prefill, +0.5% decode when the
        /// model already fits on one GPU), so the win is running a model, context
        /// or resident-weight set that one GPU cannot hold.
        /// </summary>
        protected int LayerSplitDegree { get; }

        protected ModelBase(string ggufPath, BackendType backend, int tpDegree = 1,
            ITensorParallelGroup tpGroup = null, int layerSplitDegree = 1)
        {
            LayerSplitDegree = Math.Max(1, layerSplitDegree);
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
                    if (LayerSplitDegree > 1)
                    {
                        // LAYER SPLIT: one backend per GPU, NO tensor-parallel group.
                        // _tpGroup must stay null - IsTensorParallel gates the weight
                        // sharding and AllReduce machinery, none of which applies here,
                        // and leaving it set would also make the startup banner claim a
                        // transport that is never used.
                        _ggmlContext = CreateGgmlContext(ggmlType, LayerSplitDegree, enableCollectives: false);
                        _allocator = new GgmlAllocator(_ggmlContext, 0);
                    }
                    else
                    {
                        // A caller-supplied group (multi-node) already owns the
                        // multi-GPU context; reuse it rather than initializing the
                        // devices a second time.
                        _ggmlContext = FindGgmlContext(_tpGroup) ?? CreateGgmlContext(ggmlType, tpDegree);
                        _tpGroup ??= CreateGgmlTpGroup(_ggmlContext);
                        _allocator = _tpGroup != null ? _tpGroup.GetAllocator(0) : new GgmlAllocator(_ggmlContext, 0);
                    }
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

            // Tell the kernels about the whole cluster, not just this process.
            // Expert parallelism is sharded by the GLOBAL degree, so a kernel
            // that sized its expert stack from the local one declared more
            // experts than it had bytes bound for and let the router address the
            // difference. Harmless to set on a single node: it publishes the
            // local degree and offset 0, which is what the kernels assume anyway.
            if (backend is BackendType.GgmlCuda or BackendType.GgmlVulkan
                        or BackendType.GgmlCpu or BackendType.GgmlMetal)
            {
                // Published unconditionally, including the (0, 0) reset for a
                // non-TP model: the value is process-global and sticky, so a
                // plain model loaded after a tensor-parallel one would inherit
                // the old degree and take the plan-mode branch it must not.
                GgmlBasicOps.TensorParallelSetGlobalGeometry(
                    _tpGroup?.GlobalDegree ?? 0, _tpGroup?.GlobalRankOffset ?? 0);
            }

            _gguf = new GgufFile(ggufPath);
        }

        /// <summary>
        /// Build the GGML context, spanning several GPUs when tensor parallelism
        /// is requested. Device ordinals are 0..degree-1 by default; set
        /// TENSORSHARP_TP_DEVICES (e.g. "0,2") to pick specific GPUs, which is how
        /// you avoid a display-attached or otherwise busy card.
        /// </summary>
        private static GgmlContext CreateGgmlContext(GgmlBackendType backendType, int tpDegree,
            bool enableCollectives = true)
        {
            if (tpDegree <= 1)
                return new GgmlContext(new[] { 0 }, backendType);

            int[] devices = ParseTpDevices(tpDegree);
            int available = GgmlBasicOps.GetGpuDeviceCount(backendType);
            if (available < tpDegree)
            {
                throw new InvalidOperationException(
                    $"Requested {tpDegree} GPU(s) but the GGML {backendType} backend sees only {available}.");
            }
            return new GgmlContext(devices, backendType, enableCollectives);
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
        /// Width of one ForwardRefill chunk on the GGML GPU backends, for models
        /// that size it themselves (see Qwen35Model.ResolvePrefillChunkSize).
        /// The startup prefill warmup uses it so the shared reuse-gallocr is grown
        /// to exactly the shape real requests build and no wider. 0 = the model has
        /// no opinion and the architecture default stands.
        /// </summary>
        protected virtual int GgmlPrefillChunkWarmupLength => 0;

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
                NativeCudaInitialCacheAllocationLength,
                KvCacheBytesPerToken);
        }

        /// <summary>
        /// Device bytes one token of KV cache costs across every attention layer
        /// (K and V together). Lets the initial allocation be capped against free
        /// VRAM instead of a fixed token count. 0 = unknown, which keeps the fixed
        /// default and today's behaviour exactly.
        /// </summary>
        protected virtual long KvCacheBytesPerToken => 0;

        /// <summary>Last reservation this model actually trimmed to, so a clamp is
        /// reported when it changes rather than once per request.</summary>
        private int _loggedReservationClamp;

        /// <summary>
        /// Cap a <see cref="PrepareForPrefill"/> reservation against the device
        /// memory that is actually free right now.
        ///
        /// The reservation is only a hint — the KV cache still grows on demand — so
        /// trimming it costs at most one later grow. Honouring it blind, on the
        /// other hand, sizes a multi-gigabyte buffer from the request's declared
        /// GENERATION budget rather than from anything the machine has:
        /// <c>BatchExecutor.BuildPrefillChunk</c> passes prompt + MaxNewTokens, so a
        /// server started with a large --max-tokens reserves the whole window on the
        /// very first request. On Metal that is fatal rather than merely slow (see
        /// <see cref="GpuMemoryBudget.AppliesToReservations"/>).
        ///
        /// Only ever trims: returns <paramref name="requiredContextTokens"/>
        /// unchanged when the backend has no queryable budget, the model reports no
        /// per-token KV cost, or the reservation already fits.
        /// </summary>
        protected int ResolvePrefillReservationLength(
            int requiredContextTokens, int currentCapacityTokens, int granularity = 256)
        {
            long bytesPerToken = KvCacheBytesPerToken;
            if (bytesPerToken <= 0 || requiredContextTokens <= currentCapacityTokens)
                return requiredContextTokens;
            if (!GpuMemoryBudget.TryGetReservationSpareBytes(_backend, out long spare))
                return requiredContextTokens;

            int fitted = ResolvePrefillReservationLength(
                spare, bytesPerToken, requiredContextTokens, currentCapacityTokens, granularity);
            if (fitted >= requiredContextTokens)
                return requiredContextTokens;

            if (_loggedReservationClamp != fitted)
            {
                _loggedReservationClamp = fitted;
                Console.WriteLine(
                    $"[KV cache] Reserving {fitted} of the {requiredContextTokens} tokens this request " +
                    $"declared: {GibiBytes((long)requiredContextTokens * bytesPerToken)} of KV does not fit " +
                    $"the {GibiBytes(spare)} the {_backend} device has spare. The cache still grows on demand.");
            }
            return fitted;
        }

        /// <summary>The arithmetic of <see cref="ResolvePrefillReservationLength(int, int, int)"/>,
        /// separated from the device query so it can be exercised directly.</summary>
        internal static int ResolvePrefillReservationLength(
            long spareBytes,
            long kvBytesPerToken,
            int requiredContextTokens,
            int currentCapacityTokens,
            int granularity = 256)
        {
            if (kvBytesPerToken <= 0 || requiredContextTokens <= currentCapacityTokens)
                return requiredContextTokens;
            // Half the spare, the same split ResolveInitialCacheAllocationLength
            // uses: the other half has to cover the prefill and decode graph
            // scratch, which is sized separately and is live at the same time.
            return GpuMemoryBudget.FitTokens(
                spareBytes / 2, kvBytesPerToken, requiredContextTokens,
                minTokens: Math.Max(currentCapacityTokens, 1), granularity: granularity);
        }

        private static string GibiBytes(long bytes) => $"{bytes / 1024.0 / 1024.0 / 1024.0:F1} GiB";

        internal static int ResolveInitialCacheAllocationLength(
            BackendType backend,
            int requestedContextLength,
            int gpuDefault = 8192,
            int nativeCudaDefault = 2048,
            long kvBytesPerToken = 0)
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
                // Same reasoning as the GgmlMetal cap above, for the same reason —
                // it is just that on WDDM the driver pages the excess out to host
                // RAM instead of purging wired buffers. A fixed 8192 is fine on a
                // card with room and ruinous on one where the weights already fill
                // it, so cap it against what is actually free once the model is
                // resident (this runs after weight upload). The cache still grows on
                // demand, and MAX_CONTEXT still reserves the full window up front.
                if (kvBytesPerToken > 0 &&
                    GpuMemoryBudget.TryGetSpareBytes(backend, out long spare))
                {
                    // At most half the spare goes to KV: the rest has to cover the
                    // prefill and decode graph scratch, which is sized separately.
                    effectiveDefault = GpuMemoryBudget.FitTokens(
                        spare / 2, kvBytesPerToken, effectiveDefault, minTokens: 2048, granularity: 1024);
                }
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
            // ALLOCATION-time zero, on every backend that can do it. The fused
            // decode kernels read a flash/attention window padded past the rows any
            // token has written; a masked-off column contributes nothing only if
            // its K row is FINITE, and an Inf in never-written memory plus the
            // -inf mask is NaN - which takes the whole softmax row, then every
            // logit, then argmax, which returns token 0 forever.
            //
            // GgmlCuda used to be excluded here (via ShouldZeroFillCacheTensors)
            // and got recycled, uncleared pool blocks instead. That is the same
            // defect in ten model families at once: every KV grow past the initial
            // capacity, and every freshly-allocated per-request cache, could come
            // up non-finite. The perf argument for skipping the fill belongs to
            // ResetCacheTensor (per REQUEST, potentially multi-GB), not here -
            // this runs once per cache allocation, next to the allocation itself.
            //
            // Mlx stays excluded: its Fill goes through MlxNative.Full, whose
            // behaviour for block-quantized KV dtypes is unverified on this
            // machine, and no Mlx-specific failure of this kind has been observed.
            if (tensor != null && _backend != BackendType.Mlx)
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
            IEnumerable<int>? extraEosIds = null,
            int? declaredEotId = null)
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
            if (declaredEotId is int eotId && eotId >= 0 && eotId < vocabTokens.Count)
                ids.Add(eotId);

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

            // A GGUF with no tokenizer vocabulary is a pipeline COMPONENT — the
            // MiniMax-H3 towers ship with zero KV metadata, for example — not a chat
            // model. Loading one via --model used to die on an unhandled null
            // reference deep in EOG resolution; say what is actually wrong instead.
            if (vocabTokens == null || vocabTokens.Length == 0)
            {
                throw new InvalidOperationException(
                    "this GGUF carries no tokenizer vocabulary (tokenizer.ggml.tokens), so it cannot be " +
                    "served as a chat model. Component GGUFs — a pipeline's encoder or tower — are loaded " +
                    "by the pipeline that owns them, not via --model.");
            }

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
            int? declaredEotId = gguf.Metadata.ContainsKey("tokenizer.ggml.eot_token_id")
                ? (int)gguf.GetUint32("tokenizer.ggml.eot_token_id")
                : null;
            var eosIds = new List<int>(ResolveEogTokenIds(
                vocabTokens, eosId, extraEos, declaredEotId));

            // llama.cpp folds the declared end-of-turn control into the EOG set
            // for EVERY tokenizer type (llama_vocab::impl::load inserts
            // special_eot_id into special_eog_ids). This used to run only on the
            // SentencePiece branch, so a BPE model whose turn ends on a token
            // other than tokenizer.ggml.eos_token_id never stopped: Muse-Glimmer
            // declares eos_token_id = <|end_of_text|> but ends every assistant
            // turn with <|eot|> (tokenizer.ggml.eot_token_id = 200008), so it ran
            // past its answer and re-answered until max_tokens.
            bool isSentencePiece = UsesSentencePieceTokenizer(tokenizerModel);

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
                GgmlTensorType.MXFP4 or
                GgmlTensorType.NVFP4 => true,
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
            || _backend == BackendType.GgmlCpu
            // The pure-C# backend was the only one missing here, so it alone
            // copied every quantized tensor into fresh anonymous memory at load.
            // ManagedQuantizedOps reads a weight through a raw pointer and never
            // writes to it, exactly as the GgmlCpu path does, so the file-backed
            // view is safe and saves both the copy and the duplicate resident set
            // (measured on GLM-5.3-Flash UD-Q2_K_XL: ~101 GB of copying avoided).
            || _backend == BackendType.Cpu;


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
                // NVFP4 scale2 sidecar ("<base>.scale"): the true weight is
                // (quantized blocks) x Scale, so the projection output is scaled
                // here, once, for every consumer that runs through the generic
                // linear path (the DFlash drafter's NVFP4 weights included).
                if (qw.Scale != 1.0f)
                    Ops.Mul(result, result, qw.Scale);
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

        // Once per process: the resident path failing means every F32 linear falls
        // back the same way, and this runs once per layer per token.
        private static int _f32ResidentFallbackWarned;

        private static void WarnGgmlF32ResidentFallback(Exception ex)
        {
            if (Interlocked.Exchange(ref _f32ResidentFallbackWarned, 1) != 0)
                return;
            Console.Error.WriteLine(
                $"WARNING: device-resident GGML F32 linear rejected ({ex.Message}); using the generic " +
                "Addmm path instead, which re-uploads the weight on every call and can dominate decode " +
                "time. Reported once.");
        }

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
            catch (NotSupportedException ex)
            {
                WarnGgmlF32ResidentFallback(ex);
                return false;
            }
            catch (ArgumentException ex)
            {
                WarnGgmlF32ResidentFallback(ex);
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

        // Once per (backend, quant type): a missing device kernel for a quant type
        // affects every layer that uses that type, on every token.
        private static readonly HashSet<(BackendType, int)> _managedQuantFallbackWarned = new();

        private void WarnAddmmQuantManagedFallback(int ggmlType)
        {
            // On the pure-C# CPU backend the managed dequant IS the kernel, not a fallback.
            if (_backend == BackendType.Cpu)
                return;
            lock (_managedQuantFallbackWarned)
            {
                if (!_managedQuantFallbackWarned.Add((_backend, ggmlType)))
                    return;
            }
            Console.Error.WriteLine(
                $"WARNING: the {_backend} backend has no device kernel for quant type " +
                $"{(GgmlTensorType)ggmlType}; every projection of that type runs on the CPU " +
                "managed-dequant path instead, which is much slower. Reported once per backend and type.");
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

            WarnAddmmQuantManagedFallback(weight.GgmlType);

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
            float[] logits = ForwardCore(tokens);
            ThrowIfBackendFailed();
            return DumpLogitsIfRequested(logits);
        }

        /// <summary>
        /// Stop as soon as the GPU backend has died, rather than at whichever op
        /// happens to fail next.
        ///
        /// A command buffer that fails on the GPU (Metal reports
        /// kIOGPUCommandBufferCallbackErrorOutOfMemory) is discovered inside
        /// ggml_backend_synchronize, which returns void: the op that drained it
        /// returns SUCCESS over undefined results, and only a LATER graph fails —
        /// so the exception used to name an innocent bystander (an embedding
        /// get_rows, typically, since that is the first op of the next forward) and
        /// every forward in between produced quietly wrong logits.
        ///
        /// One P/Invoke reading one atomic per forward, and only on the GGML
        /// backends — nothing measurable next to the forward itself.
        /// </summary>
        private void ThrowIfBackendFailed()
        {
            if (!IsGgmlBackend || !GgmlBasicOps.HasBackendFailure())
                return;
            string detail = GgmlBasicOps.BackendFailureText();
            throw new InvalidOperationException(
                $"The GGML {_backend} backend failed during GPU execution and cannot recover in this " +
                "process — the results of this and any preceding forward are undefined. Restart the host. " +
                (string.IsNullOrWhiteSpace(detail) ? string.Empty : $"ggml reported: {detail}"));
        }

        /// <summary>
        /// Write the FIRST forward's logits to TS_DUMP_LOGITS when set, then stop.
        /// Comparing two executors by their generated TEXT is a poor test: greedy
        /// decoding turns a near-tie into a visibly different sentence, and a
        /// genuinely broken implementation and a 2-bit expert-pick flip look the
        /// same from the outside. The logit vector distinguishes them - a correct
        /// port stays highly correlated even when the argmax moves.
        /// </summary>
        private float[] DumpLogitsIfRequested(float[] logits)
        {
            // WarmUpKernels runs its own Forward calls (a dummy decode token and a
            // dummy prefill) BEFORE the real prompt. Dumping those compares two
            // executors on throwaway input and says nothing about the model.
            if (_logitsDumped || _inWarmup || logits == null) return logits;
            string path = Environment.GetEnvironmentVariable("TS_DUMP_LOGITS");
            if (string.IsNullOrEmpty(path)) return logits;
            _logitsDumped = true;
            using (var fs = new System.IO.FileStream(path, System.IO.FileMode.Create, System.IO.FileAccess.Write))
            using (var bw = new System.IO.BinaryWriter(fs))
                foreach (float v in logits) bw.Write(v);
            return logits;
        }

        private bool _logitsDumped;
        /// <summary>Set while WarmUpKernels drives its throwaway forwards.</summary>
        protected bool _inWarmup;

        public float[] ForwardRefill(int[] tokens)
        {
            if (_distributedDriver) _tpGroup.BroadcastControl(TpControlForwardRefill, tokens);
            float[] logits = ForwardRefillCore(tokens);
            ThrowIfBackendFailed();
            return logits;
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
        /// <summary>Whether this model can actually run a vision tower right now.</summary>
        public bool HasVisionEncoder()
        {
            // A tower loaded from a separate mmproj lives on the model, not in the
            // main GGUF's weight tables, so ask the model first.
            if (this is Architecture.IVisionCapableModel vision && vision.IsVisionEncoderLoaded)
                return true;

            // Families whose tower is baked into the model GGUF under "v." names.
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
        /// <param name="draftModelPath">Optional speculative-decoding draft
        /// model (DeepSeek V4's DSpark support GGUF, Muse-Glimmer's DFlash block);
        /// ignored by architectures that have no drafter.</param>
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
            var architecture = ModelArchitectureRegistry.Resolve(probe.GetString("general.architecture"), probe);

            var context = new ModelCreateContext(ggufPath, backend, probe, tpDegree, tpGroup, draftModelPath);
            architecture.ApplyNativeTunables?.Invoke(context);

            tpDegree = ResolveTensorParallelSupport(architecture, backend, tpDegree, ref tpGroup, out int layerSplit);

            ModelBase model = architecture.Factory(context.With(tpDegree, tpGroup, layerSplit));
            model.WarnIfTensorParallelShardedNothing(architecture.Id);
            return model;
        }

        /// <summary>
        /// Decide whether the requested tensor-parallel degree can actually be
        /// honoured for <paramref name="architecture"/>. A single-node request degrades
        /// (to a layer split where the architecture supports one, otherwise to a single
        /// GPU) with a loud explanation, so an existing <c>--tp N</c> script keeps
        /// working, just honestly; a DISTRIBUTED group throws, because one node quietly
        /// dropping to a single rank desynchronises the collective.
        ///
        /// The mode and its explanation come from the architecture's own descriptor, so
        /// the next family that lands without tensor parallelism declares that fact
        /// beside its model instead of in a table here. Silence plus a banner asserting
        /// the opposite is the worst possible outcome; be explicit instead.
        /// </summary>
        internal static int ResolveTensorParallelSupport(ModelArchitectureDescriptor architecture,
            BackendType backend, int tpDegree, ref ITensorParallelGroup tpGroup, out int layerSplitDegree)
        {
            ArgumentNullException.ThrowIfNull(architecture);

            layerSplitDegree = 1;
            bool wantsTp = tpDegree > 1 || tpGroup != null;
            if (!wantsTp || architecture.MultiGpu == MultiGpuMode.TensorParallel)
                return tpDegree;

            string why = architecture.MultiGpuLimitation;

            if (tpGroup != null)
            {
                throw new NotSupportedException(
                    why + " A distributed tensor-parallel group cannot be downgraded on one node without " +
                    "desynchronising the others, so this run is refused. Start the node without --tp-node-id/--tp-peers.");
            }

            // No sharding, but the architecture can still spread its LAYERS across the
            // GPUs. That is what an operator asking for N GPUs wants, and it is the same
            // mode llama.cpp uses for these models, so honour --tp N as a layer split
            // rather than throwing the second GPU away.
            if (architecture.MultiGpu == MultiGpuMode.LayerSplit
                && ModelArchitectureDescriptor.BackendHasSeveralDevices(backend))
            {
                layerSplitDegree = tpDegree;
                Console.WriteLine(
                    $"  Multi-GPU: {tpDegree} GPUs by LAYER SPLIT (each GPU holds a contiguous run of whole " +
                    "layers), not tensor parallelism - this architecture shards no weights. Same mode " +
                    "llama.cpp uses for it. This raises capacity; it is not expected to raise decode speed.");
                return 1;
            }

            Console.Error.WriteLine(
                $"WARNING: --tp {tpDegree} ignored. {why} Running on ONE GPU; the extra GPUs would have been " +
                "given a CUDA context and NCCL buffers and then left idle. To choose WHICH GPU, set " +
                "CUDA_VISIBLE_DEVICES (e.g. CUDA_VISIBLE_DEVICES=1).");
            return 1;
        }

        /// <summary>
        /// Backstop for the next architecture that lands without TP: tensor
        /// parallelism was requested and the group is live, yet the model sharded
        /// no weights at all, so every rank but 0 is idle. Costs one dictionary
        /// count and only ever runs on a TP load.
        /// </summary>
        private void WarnIfTensorParallelShardedNothing(string arch)
        {
            if (!IsTensorParallel)
                return;
            if (_tpQuantWeights.Count > 0 || _tpWeights.Count > 0)
                return;
            Console.Error.WriteLine(
                $"WARNING: tensor parallelism is active ({_tpGroup.Degree} ranks) but architecture '{arch}' " +
                "sharded 0 weights - the whole model is resident on rank 0 and the other GPUs are idle. " +
                "This architecture has no tensor-parallel implementation; run without --tp.");
        }
    }
}
