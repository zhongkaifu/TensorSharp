// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// DeepSeek V4 (Flash) — driver for the native whole-model executor.
//
// The architecture (4-stream hyper-connections, shared single-KV-head
// attention with sinks and inverse-RoPE outputs, per-layer raw/CSA/HCA
// compressed attention with a lightning indexer, sqrt-softplus MoE with hash
// routing) is implemented natively in
// TensorSharp.GGML.Native/ggml_ops_deepseek4.cpp as one ggml graph per
// ubatch, scheduled across every visible GPU (layer split) so models larger
// than a single GPU's VRAM are hosted across all of them.
//
// This class parses metadata/tokenizer from the (first) GGUF shard, feeds
// token batches to the native executor, and returns last-token logits. KV
// truncation is V4.1 on the native and direct-CUDA executors
// (SupportsKVCacheTruncation): it rewinds to the matching prefix of the next
// turn, which a thinking turn needs because its render drops the previous
// answer's reasoning. Plain V4 and the pure-C# executor re-prefill.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models
{
    public partial class DeepSeek4Model : ModelBase, IBatchedPagedModel
    {
        private IntPtr _handle;
        private DeepSeek4CpuExecutor _cpuExec;
        private DeepSeek4CudaExecutor _cudaExec;
        // The executor's sequence slots: the native executor's or the direct-CUDA engine's.
        // Null on the pure-C# executor, which serves one sequence.
        private IDsv4SlotExecutor _slotExecutor;
        private readonly object _sync = new object();
        // The multiple a KV truncation target must be, or 0 when this load cannot truncate
        // at all (plain V4, or the pure-C# executor). Resolved once: it is a property of the
        // checkpoint's compression ratios, and SupportsKVCacheTruncation is read on hot
        // scheduler paths.
        private readonly int _truncateAlign;
        private GgmlDeepSeek4Native.CrossNodeAllReduce _distributedTpReduce;
        private GgmlDeepSeek4Native.LayerTransfer _distributedLayerTransfer;
        private float[] _distributedTpBuffer = Array.Empty<float>();
        // Native stage transfers address peers by pipeline stage; see LayerSplitPlan.
        private bool _layerDriverRunsLastStage;
        private bool _distributedTpAborted;
        protected bool IsDistributedNativeTensorParallel => _tpGroup?.NodeCount > 1;
        protected bool IsDistributedNativeLayerSplit => LayerSplitGroup?.NodeCount > 1;
        private ITensorParallelGroup NativeDistributedGroup => (ITensorParallelGroup)LayerSplitGroup ?? _tpGroup;
        protected override bool ShardsWeightsInComponents => true;
        protected internal IntPtr NativeHandle => _handle;
        protected object NativeSync => _sync;

        public DeepSeek4Model(string ggufPath, BackendType backend, int tpDegree = 1, ITensorParallelGroup tpGroup = null,
            string draftModelPath = null, int layerSplitDegree = 1)
            : base(ggufPath, ValidateParallelism(backend, tpDegree, tpGroup, layerSplitDegree), 1, tpGroup)
        {
            string arch = _gguf.GetString("general.architecture") ?? "deepseek4";
            bool isV41 = string.Equals(arch, "deepseek41", StringComparison.Ordinal);
            if (isV41)
            {
                // A refusal here would otherwise leak the GGUF mapping the base
                // constructor opened: nobody disposes an object whose constructor threw.
                try
                {
                    DeepSeek41Architecture.ValidateLoad(ggufPath, backend, ResolveDsparkPath(draftModelPath),
                        tpGroup, tpDegree, _gguf);
                }
                catch
                {
                    base.Dispose();
                    throw;
                }
                // Once per load, and only here: ValidateLoad also runs as the
                // descriptor's ApplyNativeTunables, so warning from inside it
                // would print the same line twice.
                if (DeepSeek41Architecture.DescribeCpuBackendChoice(backend) is string cpuNote)
                    Console.Error.WriteLine(cpuNote);
                if (DeepSeek41Architecture.DescribeYocoAbsence(backend) is string yocoNote)
                    Console.Error.WriteLine(yocoNote);
            }
            else if (DescribeCpuBackendChoice(backend) is string v4CpuNote)
            {
                Console.Error.WriteLine(v4CpuNote);
            }
            Config = new ModelConfig { Architecture = arch };
            // V4.1 was refused in ValidateLoad above (and once more before the
            // factory ran); plain V4 has no pre-load hook, so refuse here.
            if (!isV41)
            {
                try
                {
                    if (tpDegree > 1 || tpGroup != null)
                        throw new NotSupportedException("DeepSeek V4 does not implement tensor parallelism; use --layer-split N. --tp is supported by DeepSeek V4.1.");
                    DeepSeek4Architecture.RefuseBlockQuantizedKvCache("DeepSeek V4 (Flash)", v41: false);
                }
                catch
                {
                    base.Dispose();
                    throw;
                }
            }
            // Every executor of this family keeps F16 caches and ignores the
            // process-wide dtype, so report what is actually allocated rather
            // than whatever KV_CACHE_DTYPE happened to say.
            _kvCacheDtype = DeepSeek4Architecture.ExecutorKvCacheDtype;
            if (KvCacheDtypeConfig.IsExplicitlySet && KvCacheDtypeConfig.Current == KvCacheDtype.F32)
                Console.Error.WriteLine(DeepSeek4Architecture.DescribeF32KvCacheRequest(
                    isV41 ? "DeepSeek V4.1 Flash" : "DeepSeek V4 (Flash)"));
            ParseBaseConfig();
            if (isV41)
            {
                Config.NumExperts = (int)_gguf.GetUint32($"{arch}.expert_count");
                Config.NumExpertsUsed = (int)_gguf.GetUint32($"{arch}.expert_used_count");
                Config.IntermediateSize = (int)_gguf.GetUint32($"{arch}.expert_feed_forward_length");
                Config.SlidingWindow = (int)_gguf.GetUint32($"{arch}.attention.sliding_window");
                Config.OriginalContextLength = (int)_gguf.GetUint32($"{arch}.rope.scaling.original_context_length");
            }
            ParseTokenizer();

            int requestedGpuCount = Math.Max(tpDegree, layerSplitDegree);
            if (requestedGpuCount > 1)
            {
                int available = backend == BackendType.Cuda ? Cuda.CudaDevice.GetDeviceCount() :
                    GgmlBasicOps.GetGpuDeviceCount(backend == BackendType.GgmlCuda
                        ? GgmlBackendType.Cuda : GgmlBackendType.Vulkan);
                if (available < requestedGpuCount)
                    throw new InvalidOperationException($"Requested {requestedGpuCount} GPU(s) but {backend} sees only {available}.");
            }

            int maxContext = ResolveConfiguredContextLength();
            // The GGUF advertises 1M context; cache rows scale with n_ctx, so keep a
            // practical default unless the operator asks for more via MAX_CONTEXT.
            string ctxEnv = Environment.GetEnvironmentVariable("MAX_CONTEXT");
            if (string.IsNullOrWhiteSpace(ctxEnv))
                maxContext = Math.Min(maxContext, 65536);
            _maxContextLength = maxContext;

            int nUbatch = ResolveNativeUbatch(isV41, backend, ExecutionVariable("TS_DSV4_UBATCH"),
                out string ubatchWarning);
            if (ubatchWarning != null)
                Console.Error.WriteLine(ubatchWarning);

            if (backend == BackendType.Cuda)
            {
                // Direct-CUDA whole-model executor: quantized weights resident in
                // device memory, layer-split across the visible GPUs, driver-API
                // kernels only (no ggml).
                int nGpu = ResolveNativeGpuCount(Math.Max(tpDegree, layerSplitDegree));
                string dspark = ResolveDsparkPath(draftModelPath);
                Console.WriteLine($"Model: {arch} (direct-CUDA whole-model executor), Layers={Config.NumLayers}, " +
                    $"Hidden={Config.HiddenSize}, Heads={Config.NumHeads}, HeadDim={Config.KeyLength}, Vocab={Config.VocabSize}" +
                    (dspark != null ? ", DSpark drafter" : string.Empty));
                if (IsDistributedNativeLayerSplit)
                {
                    // One stage of a cross-node layer split. Like the ggml stages, it serves the one mirrored
                    // sequence: no slots, truncation or prefix reuse (the scheduler resets and re-prefills).
                    try
                    {
                        _cudaExec = LoadDistributedCudaStage(ggufPath, arch, isV41, maxContext, nUbatch, dspark);
                    }
                    catch
                    {
                        // As for the refusals above: nobody disposes an object whose constructor threw, and
                        // the GGUF mapping and the group (so the peers stop waiting) must not outlive it.
                        base.Dispose();
                        throw;
                    }
                    _slotExecutor = null;
                    _truncateAlign = 0;
                }
                else
                {
                    _cudaExec = new DeepSeek4CudaExecutor(ggufPath, maxContext, nUbatch, nGpu, dspark, ResolveCpuMoeLayers());
                    _slotExecutor = _cudaExec;
                    _truncateAlign = _cudaExec.TruncateAlign;
                }
            }
            else if (_backend == BackendType.Cpu)
            {
                // Pure C# whole-model executor: quantized weights served straight
                // from the memory-mapped GGUF shards, managed SIMD kernels only.
                WarnDsparkUnavailable(draftModelPath, backend);
                int nThreads = ParseEnvInt("TS_DSV4_THREADS", Environment.ProcessorCount);
                Console.WriteLine($"Model: {arch} (pure C# CPU executor), Layers={Config.NumLayers}, " +
                    $"Hidden={Config.HiddenSize}, Heads={Config.NumHeads}, HeadDim={Config.KeyLength}, Vocab={Config.VocabSize}");
                _cpuExec = new DeepSeek4CpuExecutor(ggufPath, maxContext, nUbatch, nThreads, _allocator);
            }
            else
            {
                int nThreads = ParseEnvInt("TS_DSV4_THREADS", Math.Min(Environment.ProcessorCount, 32));
                int nGpu = ResolveNativeGpuCount(Math.Max(tpDegree, layerSplitDegree));
                string dspark = backend == BackendType.GgmlCuda || (isV41 && backend == BackendType.GgmlCpu)
                    ? ResolveDsparkPath(draftModelPath) : null;
                if (dspark == null)
                    WarnDsparkUnavailable(draftModelPath, backend);

                Console.WriteLine($"Model: {arch} (native whole-model executor), Layers={Config.NumLayers}, " +
                    $"Hidden={Config.HiddenSize}, Heads={Config.NumHeads}, HeadDim={Config.KeyLength}, Vocab={Config.VocabSize}" +
                    (dspark != null ? ", DSpark drafter" : string.Empty));

                int nCpuMoe = ResolveCpuMoeLayers();
                // Use the operator's original backend: native execution owns
                // its compute queues independently of the managed context.
                string backendName = BackendRegistryName(backend);
                int tensorParallelRanks = isV41
                    ? (IsDistributedNativeTensorParallel ? tpDegree : DeepSeek41Architecture.ResolveTensorParallelRanks(tpDegree)) : 0;
                if (tensorParallelRanks > 0)
                    Console.WriteLine(DeepSeek41Architecture.DescribeTensorParallelPlacement(
                        IsDistributedNativeTensorParallel ? _tpGroup.GlobalDegree : tensorParallelRanks));
                if (IsDistributedNativeLayerSplit)
                {
                    if (dspark != null)
                        throw new NotSupportedException("Distributed layer split does not mirror DSpark operations; omit --draft-model.");
                    string headerIdentity = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(
                        ReadModelHeaderForIdentity(ggufPath)));
                    _layerDriverRunsLastStage = LayerSplitPlan.DriverRunsLastStageFromEnvironment();
                    DeepSeek41Architecture.ValidateDistributedExecutionConfig(LayerSplitGroup,
                        $"layer|{arch}|header={headerIdentity}|layers={Config.NumLayers}|hidden={Config.HiddenSize}" +
                        $"|nodes={LayerSplitGroup.NodeCount}|local={LayerSplitGroup.Degree}|ctx={maxContext}" +
                        $"|driverLast={_layerDriverRunsLastStage}" +
                        $"|ubatch={(nUbatch > 0 ? nUbatch : 256)}|cpu_moe={nCpuMoe}" +
                        DeepSeek41Architecture.DistributedNumericalOptions());
                    _distributedLayerTransfer = TransferLayerState;
                    // With the default rotated order node 0 runs the last stage and
                    // node 1 the first, so the full-vocabulary logits stay local.
                    _handle = GgmlDeepSeek4Native.LoadModelLayerDistributed(ggufPath, LayerSplitGroup.Degree,
                        LayerSplitPlan.StageOf(LayerSplitGroup.NodeId, LayerSplitGroup.NodeCount, _layerDriverRunsLastStage),
                        LayerSplitGroup.NodeCount, maxContext, nUbatch, nThreads,
                        nCpuMoe, backendName, _distributedLayerTransfer);
                    if (_handle != IntPtr.Zero)
                        GgmlDeepSeek4Native.SetPipelineDriver(_handle,
                            LayerSplitPlan.StageOf(0, LayerSplitGroup.NodeCount, _layerDriverRunsLastStage));
                }
                else if (IsDistributedNativeTensorParallel)
                {
                    var reducer = (INestedTensorParallelGroup)tpGroup;
                    string yoco = Environment.GetEnvironmentVariable(DeepSeek41Architecture.YocoKeepVariable)
                        ?? Math.Max(140, Config.SlidingWindow + 8).ToString(System.Globalization.CultureInfo.InvariantCulture);
                    string headerIdentity = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(
                        ReadModelHeaderForIdentity(ggufPath)));
                    DeepSeek41Architecture.ValidateDistributedExecutionConfig(tpGroup,
                        $"{arch}|header={headerIdentity}|layers={Config.NumLayers}|hidden={Config.HiddenSize}" +
                        $"|experts={Config.NumExperts}|used={Config.NumExpertsUsed}|ff={Config.IntermediateSize}" +
                        $"|local={tpDegree}|global={tpGroup.GlobalDegree}|ctx={maxContext}|ubatch={(nUbatch > 0 ? nUbatch : 256)}" +
                        $"|cpu_moe={Math.Min(nCpuMoe, Config.NumLayers)}|yoco={yoco}" +
                        DeepSeek41Architecture.DistributedNumericalOptions());
                    _distributedTpReduce = (user, data, count) =>
                    {
                        if (_distributedTpAborted) return false;
                        if (count < 0) { AbortDistributedTensorParallel(); return false; }
                        try
                        {
                            if (count <= 0) return true;
                            if (_distributedTpBuffer.Length < count) _distributedTpBuffer = new float[count];
                            Marshal.Copy(data, _distributedTpBuffer, 0, count);
                            reducer.CrossNodeAllReduce(_distributedTpBuffer, count);
                            Marshal.Copy(_distributedTpBuffer, 0, data, count);
                            return true;
                        }
                        catch (Exception error)
                        {
                            Console.Error.WriteLine($"[dsv41-tp] cross-node gather failed: {error.Message}");
                            AbortDistributedTensorParallel(error);
                            return false;
                        }
                    };
                    Console.WriteLine($"  Distributed routed-MoE TP: {_tpGroup.NodeCount} nodes, {tpDegree} local / {_tpGroup.GlobalDegree} global ranks, offset {_tpGroup.GlobalRankOffset}.");
                    _handle = GgmlDeepSeek4Native.LoadModelDistributed(ggufPath, tpDegree, _tpGroup.GlobalDegree,
                        _tpGroup.GlobalRankOffset, maxContext, nUbatch, nThreads, nCpuMoe, backendName, _distributedTpReduce);
                }
                else
                    _handle = GgmlDeepSeek4Native.LoadModel(ggufPath, nGpu, maxContext, nUbatch, nThreads,
                        dspark, nCpuMoe, backendName, tensorParallelRanks);
                _nativeDsparkBlock = _handle != IntPtr.Zero && dspark != null
                    ? GgmlDeepSeek4Native.DsparkBlockSize(_handle) : 0;
                if (_handle == IntPtr.Zero)
                    throw NativeLoadRefused("dsv4", ggufPath);
                // The width the loader actually runs: the request, or its own
                // choice for UBatchAuto. Speculative prefill chunks to it.
                _nativeUBatch = GgmlDeepSeek4Native.UBatch(_handle);
                // Zero for plain V4: its compressor overlaps blocks, so a rewind reads state
                // rows an aligned target does not protect, and the native side declines.
                // Existing distributed controls mirror ordinary forwards and
                // resets. Slot selection, refusable truncation and retention
                // have no mirrored control, so the scheduler uses its serial
                // per-sequence fallback with reset/re-prefill between requests.
                _truncateAlign = NativeDistributedGroup?.NodeCount > 1 ? 0 : GgmlDeepSeek4Native.TruncateAlign(_handle);
                _slotExecutor = NativeDistributedGroup?.NodeCount > 1 ? null : new NativeDsv4Slots(_handle);
            }
        }

        /// <summary>The direct-CUDA executor (tests read its engine's boundary counters).</summary>
        internal DeepSeek4CudaExecutor CudaExecutor => _cudaExec;

        /// <summary>
        /// DeepSeek V4.1's Engram tables: llama.cpp's combined <c>per_layer_token_embd.weight</c>, or TensorSharp's
        /// per-layer <c>blk.N.engram_embd.weight</c>. Every executor reads them a few scattered rows per token from
        /// the model file unless it placed them in VRAM (TS_DSV41_ENGRAM_DEVICE), so their page cache is what makes a
        /// request cold or warm. Taken from the checkpoint, not from this node's layers: every node of a distributed
        /// group evicts the same ranges. Plain V4 has none.
        /// </summary>
        public override IReadOnlyList<TensorSharp.Runtime.HostFileRange> HostMappedTables
        {
            get
            {
                var tables = new List<TensorSharp.Runtime.HostFileRange>();
                void Add(string name)
                {
                    if (_gguf.TryGetTensorFileRange(name, out string path, out long offset, out long bytes))
                        tables.Add(new TensorSharp.Runtime.HostFileRange(name, path, offset, bytes));
                }
                Add(Dsv41EngramData.CombinedTableName);
                for (int layer = 0; layer < Config.NumLayers; layer++)
                    Add(Dsv41EngramData.TableName(layer));
                return tables;
            }
        }

        /// <summary>The direct-CUDA executor's own reads of the Engram tables (its read-ahead and a background warm)
        /// stop before a page-cache eviction drops them.</summary>
        protected override void QuiesceHostTableReaders() => _cudaExec?.QuiesceEngramReads();

        /// <summary>
        /// Tests only: one node's value of an execution variable, for an in-process multi-node test where every
        /// node shares the process environment. Null (always, outside tests) reads the environment.
        /// </summary>
        internal static Func<int, string, string> DistributedEnvironmentForTests { get; set; }

        /// <summary>An execution variable as this node sees it.</summary>
        private string ExecutionVariable(string name)
            => LayerSplitGroup is { NodeCount: > 1 } group && DistributedEnvironmentForTests is { } read
                ? read(group.NodeId, name)
                : Environment.GetEnvironmentVariable(name);

        /// <summary>
        /// This node's stage of a cross-node layer split on the direct-CUDA engine. Every node digests the
        /// checkpoint header and every setting that changes what the engine computes (TS_DSV4_UBATCH, the dense
        /// matmul routing, where the Engram tables live, ...) and refuses to load unless all agree; then the
        /// placement, which each node computes from the shards' bytes, and the kernel module's decoders. The
        /// driver (node 0) runs the last stage, so the head is local to it and logits never cross the network.
        /// </summary>
        private DeepSeek4CudaExecutor LoadDistributedCudaStage(string ggufPath, string arch, bool isV41, int maxContext,
            int nUbatch, string dspark)
        {
            if (!isV41)
                throw new NotSupportedException("--backend cuda distributed layer split runs DeepSeek V4.1 only.");
            if (dspark != null)
                throw new NotSupportedException("Distributed layer split does not mirror DSpark operations; omit --draft-model.");
            if (ResolveCpuMoeLayers() != 0)
                throw new NotSupportedException("--backend cuda distributed layer split keeps every routed expert in VRAM; " +
                    "remove --n-cpu-moe / --cpu-moe (or use --backend ggml_cuda).");
            bool driverLast = LayerSplitPlan.DriverRunsLastStageFromEnvironment();
            if (!driverLast)
                throw new NotSupportedException($"--backend cuda distributed layer split runs the driver (node 0) as the LAST " +
                    $"stage, so the full-vocabulary logits never cross the network; {LayerSplitPlan.DriverStageVariable}=first " +
                    "is not supported on cuda. Unset it, or use --backend ggml_cuda.");
            ILayerSplitGroup group = LayerSplitGroup;
            int nodes = group.NodeCount, local = group.Degree;
            int stage = LayerSplitPlan.StageOf(group.NodeId, nodes, driverLast);
            string headerIdentity = Convert.ToHexString(System.Security.Cryptography.SHA256.HashData(
                ReadModelHeaderForIdentity(ggufPath)));
            string configuration =
                $"layer|exec=cuda|{arch}|header={headerIdentity}|layers={Config.NumLayers}|hidden={Config.HiddenSize}" +
                $"|vocab={Config.VocabSize}|nodes={nodes}|local={local}|ctx={maxContext}|driverLast={driverLast}" +
                $"|ubatch={(nUbatch > 0 ? nUbatch : 1024)}|cpu_moe=0" +
                DeepSeek41Architecture.DirectCudaNumericalOptions(ExecutionVariable);
            DeepSeek41Architecture.ValidateDistributedExecutionConfig(group, configuration);
            Console.WriteLine($"  Distributed direct-CUDA layer split: node {group.NodeId} runs stage {stage} of {nodes} " +
                $"on {local} GPU(s); the driver runs the last stage and the head.");
            var link = new LayerSplitStageLink(group,
                stage > 0 ? LayerSplitPlan.NodeOfStage(stage - 1, nodes, driverLast) : -1,
                stage < nodes - 1 ? LayerSplitPlan.NodeOfStage(stage + 1, nodes, driverLast) : -1,
                other => other >= 0 && other < nodes && other != stage ? LayerSplitPlan.NodeOfStage(other, nodes, driverLast) : -1);
            return new DeepSeek4CudaExecutor(ggufPath, maxContext, nUbatch, new DeepSeek4CudaExecutor.DistributedStage
            {
                Nodes = nodes,
                NodeId = group.NodeId,
                LocalDevices = local,
                DriverRunsLastStage = driverLast,
                Link = link,
                Agree = identity => DeepSeek41Architecture.ValidateDistributedExecutionConfig(group, configuration + "|" + identity),
                Build = DirectCudaBuildIdentity,
            });
        }

        /// <summary>The build a direct-CUDA stage runs, for the module agreement: this assembly's informational version
        /// (the release and the source revision it was built from). The CUDA backend assembly carries no version of its
        /// own, and the two always ship together.</summary>
        internal static string DirectCudaBuildIdentity { get; } =
            System.Reflection.CustomAttributeExtensions.GetCustomAttribute<System.Reflection.AssemblyInformationalVersionAttribute>(
                typeof(DeepSeek4Model).Assembly)?.InformationalVersion
            ?? typeof(DeepSeek4Model).Assembly.GetName().Version?.ToString() ?? "unknown";

        /// <summary>
        /// The direct-CUDA engine's stage link over the node pipeline: the engine's pinned payload moves as
        /// lossless 32-bit words (it pads every payload to 4 bytes). A transport fault throws out of the engine's
        /// forward, which fails the request, and the model then aborts the mesh so no peer waits on it.
        /// Each direction has its own word buffer: a middle stage (3 or more nodes) both receives and sends, and
        /// the engine's deferred send of one ubatch runs after the receive of the next, so a shared buffer would
        /// only be safe for as long as both stay on one thread.
        /// </summary>
        private sealed unsafe class LayerSplitStageLink(ILayerSplitGroup group, int previousNode, int nextNode,
            Func<int, int> nodeOfStage) : TensorSharp.Cuda.IDsv4StageLink
        {
            private float[] _sendWords = Array.Empty<float>(), _receiveWords = Array.Empty<float>();

            public void SendNext(IntPtr data, int bytes) => Move(nextNode, data, bytes, ref _sendWords, send: true);

            public void ReceivePrevious(IntPtr data, int bytes) => Move(previousNode, data, bytes, ref _receiveWords, send: false);

            // The Engram rows a node keeps for another node's layer (Dsv4StagePlan.EngramOwnerStage). The engine moves
            // them on the forward's own thread, before any boundary of the forward and after the previous forward's
            // deferred send has drained, through the same per-direction buffers, so they never overlap a boundary.
            public void SendToStage(int stage, IntPtr data, int bytes) => Move(nodeOfStage(stage), data, bytes, ref _sendWords, send: true);

            public void ReceiveFromStage(int stage, IntPtr data, int bytes) => Move(nodeOfStage(stage), data, bytes, ref _receiveWords, send: false);

            private void Move(int peer, IntPtr data, int bytes, ref float[] buffer, bool send)
            {
                if (peer < 0)
                    throw new InvalidOperationException("This pipeline stage has no neighbour in that direction.");
                if (bytes < 0 || (bytes & 3) != 0)
                    throw new ArgumentException($"A stage payload must be a whole number of 32-bit words, got {bytes} bytes.");
                int words = bytes / 4;
                if (buffer.Length < words)
                    buffer = new float[words];   // grows to the largest payload once
                fixed (float* w = buffer)
                {
                    if (send)
                    {
                        Buffer.MemoryCopy((void*)data, w, (long)buffer.Length * 4, bytes);
                        group.Send(buffer, words, peer);
                    }
                    else
                    {
                        group.Receive(buffer, words, peer);
                        Buffer.MemoryCopy(w, (void*)data, bytes, bytes);
                    }
                }
            }
        }

        private static byte[] ReadModelHeaderForIdentity(string path)
        {
            // A bounded header identity catches accidental checkpoint/config
            // differences without reading hundreds of GiB during model load.
            // Validation records a separate immutable full-weight identity.
            using var stream = System.IO.File.OpenRead(path);
            var header = new byte[checked((int)Math.Min(stream.Length, 64 * 1024))];
            stream.ReadExactly(header);
            return header;
        }

        private void AbortDistributedTensorParallel(Exception failure = null)
        {
            if (NativeDistributedGroup?.NodeCount is not > 1 || _distributedTpAborted) return;
            _distributedTpAborted = true;
            // Native rank queues belong to the executor, independently of the
            // managed group's allocators. Closing its sockets unblocks peers
            // even when they entered a collective before this node failed.
            try
            {
                if (NativeDistributedGroup is IDistributedExecutionGroup execution)
                    execution.Abort(failure ?? new InvalidOperationException("DeepSeek distributed execution failed."));
                else NativeDistributedGroup.Dispose();
            }
            catch (Exception error) { Console.Error.WriteLine($"[dsv41-tp] transport abort: {error.Message}"); }
        }

        private unsafe bool TransferLayerState(IntPtr user, int peer, int send, IntPtr data, int bytes)
        {
            if (_distributedTpAborted || bytes < 0 || data == IntPtr.Zero) return false;
            try
            {
                peer = LayerSplitPlan.NodeOfStage(peer, LayerSplitGroup.NodeCount, _layerDriverRunsLastStage);
                int words = checked((bytes + 3) / 4);
                if (_distributedTpBuffer.Length < words) _distributedTpBuffer = new float[words];
                fixed (float* buffer = _distributedTpBuffer)
                {
                    if (send != 0)
                    {
                        if (words != 0) _distributedTpBuffer[words - 1] = 0;
                        Buffer.MemoryCopy(data.ToPointer(), buffer, (long) words * 4, bytes);
                        LayerSplitGroup.Send(_distributedTpBuffer, words, peer);
                    }
                    else
                    {
                        LayerSplitGroup.Receive(_distributedTpBuffer, words, peer);
                        Buffer.MemoryCopy(buffer, data.ToPointer(), bytes, bytes);
                    }
                }
                return true;
            }
            catch (Exception error)
            {
                Console.Error.WriteLine($"[dsv4-layer] state transfer failed: {error.Message}");
                AbortDistributedTensorParallel(error);
                return false;
            }
        }

        /// <summary>
        /// The one line a DeepSeek V4 load on the ggml CPU backend prints, null
        /// for every other backend.
        ///
        /// <para>Until <see cref="BackendRegistryName"/> learned "CPU",
        /// <c>ggml_cpu</c> reached the native loader as "any GPU" and a
        /// CUDA-capable host ran the GPUs. It now runs where it says, which is
        /// orders of magnitude slower for the same launch line — and this is the
        /// backend the server picks when <c>--backend</c> is omitted off macOS,
        /// so a V4 script that never named one changes behavior. The device list
        /// that follows says CPU but not that it used to say GPU, so say it
        /// here. V4.1 prints its own, longer note instead (see
        /// DeepSeek41Architecture.DescribeCpuBackendChoice).</para>
        /// </summary>
        internal static string DescribeCpuBackendChoice(BackendType backend)
            => backend != BackendType.GgmlCpu ? null
                : "[dsv4] --backend ggml_cpu: DeepSeek V4 will run on ONE CPU device, not on any GPU this host " +
                  "has. This is also the backend chosen when --backend is omitted off macOS. Pass --backend " +
                  "ggml_cuda or --backend cuda for a GPU executor.";

        /// <summary>
        /// Translate the process-wide <see cref="MoeCpuOffloadConfig"/> into the
        /// native loader's routed-expert offload policy.
        ///
        /// <para>Offload is OFF unless the operator asks for it, exactly as it is
        /// for every other architecture. DeepSeek V4 used to default to an
        /// automatic spill because a Q8_K_XL checkpoint (151 GiB) outweighs most
        /// hosts, but choosing that silently is the wrong trade on a host that
        /// *does* have the VRAM: it moves ~29 GiB of experts to the CPU and costs
        /// most of the decode throughput for no reason. A host that genuinely
        /// cannot fit the model now gets a load error naming the fewest layers
        /// that would (see the "[dsv4] does not fit" message), which is a better
        /// answer than a silent slowdown or an out-of-memory abort.</para>
        /// </summary>
        private static int ResolveCpuMoeLayers()
        {
            if (!MoeCpuOffloadConfig.IsExplicitlySet)
                return 0;
            return MoeCpuOffloadConfig.AllLayers ? int.MaxValue : MoeCpuOffloadConfig.CpuMoeLayers;
        }

        /// <summary>
        /// ggml backend registry name whose devices the native executor should
        /// run on. GgmlOps links every backend it was built with and the loader
        /// enumerates devices in registration order, so without this a
        /// CUDA-capable box runs <c>--backend ggml_vulkan</c> on its CUDA
        /// devices and the Vulkan path is never exercised. Null = any GPU.
        /// </summary>
        private protected static string BackendRegistryName(BackendType backend) => backend switch
        {
            // GGML_CUDA_NAME is "CUDA" / "ROCm" / "MUSA" depending on how
            // ggml-cuda was built; the native side treats them as one family.
            BackendType.GgmlCuda => "CUDA",
            BackendType.GgmlVulkan => "Vulkan",
            BackendType.GgmlMetal => "Metal",
            // "CPU" is the one name that reaches the loader's cpu_only branch,
            // where it builds a single CPU compute device instead of
            // enumerating accelerators. Without it --backend ggml_cpu fell into
            // the null case, which means "any GPU": on a CUDA box the operator
            // asked for the CPU and silently got the GPUs, and on a host with
            // no GPU at all the load failed with "refusing CPU-only run".
            BackendType.GgmlCpu => "CPU",
            _ => null,
        };

        /// <summary>Draft block size of the native (ggml) executor's drafter,
        /// 0 when none is loaded.</summary>
        private int _nativeDsparkBlock;

        /// <summary>Prefill micro-batch of the native executor.</summary>
        private int _nativeUBatch = 512;

        /// <summary>
        /// The DSpark drafter is implemented twice: inside the direct-CUDA engine
        /// (Dsv4CudaEngine.Dspark.cs, on that engine's own kernels and cache
        /// rings), and inside the native ggml executor
        /// (<c>GgmlDeepSeek4Native.LoadModel</c>'s drafter path), which this constructor hands it on
        /// ggml_cuda and, for V4.1, on ggml_cpu. Every other executor/backend
        /// pairing serves plain decode, so say so instead of silently ignoring
        /// the drafter the operator asked for.
        /// </summary>
        private static void WarnDsparkUnavailable(string draftModelPath, BackendType backend)
        {
            if (ResolveDsparkPath(draftModelPath) == null)
                return;
            Console.Error.WriteLine(
                $"[dsv4] a DSpark drafter was configured but the {backend} backend has no speculative " +
                "path for DeepSeek V4 (it is implemented in the direct-CUDA and ggml_cuda engines); " +
                "decoding without it.");
        }

        private static BackendType ValidateParallelism(BackendType backend, int tpDegree,
            ITensorParallelGroup tpGroup, int layerSplitDegree)
        {
            if (tpDegree < 1) throw new ArgumentOutOfRangeException(nameof(tpDegree));
            if (layerSplitDegree < 1) throw new ArgumentOutOfRangeException(nameof(layerSplitDegree));
            DeepSeek41Architecture.ValidateTensorParallelGroup(backend, tpDegree, tpGroup);
            if ((tpDegree > 1 || (tpGroup != null && tpGroup is not ILayerSplitGroup)) && layerSplitDegree > 1)
                throw new ArgumentException("--layer-split and --tp cannot be combined.");
            if (layerSplitDegree > 1 && backend is not (BackendType.Cuda or BackendType.GgmlCuda or BackendType.GgmlVulkan))
                throw new NotSupportedException($"DeepSeek --layer-split is unavailable on {backend}.");
            if (tpDegree > 1)
            {
                if (backend != BackendType.GgmlCuda)
                    throw new NotSupportedException("--tp requires DeepSeek V4.1 routed-MoE tensor parallelism on ggml_cuda. Use --layer-split N for whole-layer placement.");
                DeepSeek41Architecture.ResolveTensorParallelRanks(tpDegree);
            }
            // The pure C# executor does not use native GPU placement settings.
            // Keep their validation on the GGML and direct-CUDA paths only.
            if (backend != BackendType.Cpu)
                ResolveNativeGpuCount(Math.Max(tpDegree, layerSplitDegree));
            return NormalizeBackend(backend);
        }

        internal static int ResolveNativeGpuCount(int requestedDegree)
        {
            // Match the native DeepSeek executor's eight-rank capacity at the
            // common model entry point instead of silently capping requests.
            if (requestedDegree > 8)
                throw new NotSupportedException("The DeepSeek executor supports at most 8 GPUs; reduce --tp or --layer-split.");
            return requestedDegree;
        }

        private static BackendType NormalizeBackend(BackendType backend)
        {
            // BackendType.Cpu runs the pure C# executor. BackendType.Cuda runs
            // the direct-CUDA executor, which owns its own per-device contexts —
            // the base class only needs a lightweight CPU allocator, so it is
            // coerced to Cpu (the ctor dispatches on the ORIGINAL backend). The
            // GGML CUDA/Vulkan preserve their managed context for distributed
            // transport and backend-failure tracking. Other native backends
            // need only the lightweight GGML CPU context on the managed side.
            return backend switch
            {
                BackendType.Cpu => BackendType.Cpu,
                BackendType.Cuda => BackendType.Cpu,
                BackendType.GgmlCuda => BackendType.GgmlCuda,
                BackendType.GgmlVulkan => BackendType.GgmlVulkan,
                _ => BackendType.GgmlCpu,
            };
        }

        /// <summary>
        /// The prefill micro-batch width handed to the executor.
        ///
        /// <para>A positive <c>TS_DSV4_UBATCH</c> is used verbatim by every executor.
        /// Unset, DeepSeek V4.1 on a ggml GPU backend passes
        /// <see cref="GgmlDeepSeek4Native.UBatchAuto"/>: the native loader then
        /// evaluates 1024, 512 and 256 against the visible VRAM and keeps the widest
        /// one that needs no more routed-expert CPU offload than 256 would. A resident
        /// routed-expert layer costs about the same per chunk at any width (one
        /// Q4_K_M-shaped layer on an A40: 35.3-35.7 / 36.9-37.2 / 38.9-39.1 ms at
        /// 256 / 512 / 1024 tokens), so 1024 is ~3.6x cheaper per prefill token, while
        /// an extra host layer would slow every decoded token.
        /// V4.1's CPU and direct-CUDA executors keep 256; plain V4 keeps 512 on the
        /// pure C# executor and 1024 elsewhere (GPU chunks amortize its expert-GEMM
        /// tile padding, measured ~11-21% faster than 512).</para>
        ///
        /// <para>A set value that is not a positive integer is reported and ignored
        /// rather than silently replaced by the default.</para>
        /// </summary>
        internal static int ResolveNativeUbatch(bool isV41, BackendType backend, string configured, out string warning)
        {
            warning = null;
            if (!string.IsNullOrWhiteSpace(configured))
            {
                if (int.TryParse(configured, out int requested) && requested > 0)
                    return requested;
                warning = $"[dsv4] TS_DSV4_UBATCH='{configured}' is not a positive integer; ignoring it and using " +
                          "the default prefill width.";
            }
            if (!isV41)
                return backend == BackendType.Cpu ? 512 : 1024;
            return backend is BackendType.GgmlCuda or BackendType.GgmlVulkan or BackendType.GgmlMetal
                ? GgmlDeepSeek4Native.UBatchAuto
                : 256;
        }

        private static int ParseEnvInt(string name, int fallback)
        {
            string raw = Environment.GetEnvironmentVariable(name);
            return int.TryParse(raw, out int v) && v > 0 ? v : fallback;
        }

        /// <summary>
        /// Partial KV reuse, for V4.1 on the native and direct-CUDA executors.
        ///
        /// <para>Why it matters here: V4.1's chat protocol re-renders a past assistant
        /// turn with its reasoning removed (ordinary chat drops it, per DeepSeek's
        /// reference encoder), so from the second turn on the rendered prompt diverges
        /// from the cache exactly one token after the previous turn's
        /// <c>&lt;|Assistant|&gt;</c> - and everything before that point is the WHOLE of
        /// the previous prompt. Without truncation the planner throws all of it away and
        /// re-prefills, which turns per-turn prefill into a function of the entire
        /// conversation instead of the newest answer.</para>
        ///
        /// <para>Why only V4.1: both executors rewind past the raw sliding-window ring because
        /// every slot carries a checkpoint of the modular rings taken at the last prompt
        /// boundary (TSGgml_Dsv4Truncate, Dsv4CudaEngine.Truncate). Plain V4 compresses
        /// OVERLAPPING blocks, so a boundary still reads the previous block's state rows and
        /// aligning the head to the ratio is not sufficient. Plain V4 and the pure-C# executor,
        /// which has no checkpoint, keep re-prefilling, which is correct, just not cheap.</para>
        /// </summary>
        public override bool SupportsKVCacheTruncation => _truncateAlign > 0;

        /// <summary>The compression-block alignment the native truncate requires (the lcm
        /// of the per-layer compress ratios: 2 for the released checkpoint).</summary>
        public override int KVCacheTruncationGranularity => Math.Max(1, _truncateAlign);

        /// <summary>
        /// What a slot of this load holds: the MLA latent ring, compressed and indexer
        /// rows, all sized by the layer count, head geometry and sliding window, plus which
        /// executor owns them (the native, direct-CUDA and pure-C# executors keep different
        /// state and rewind differently) and the dtype actually allocated (always F16, see
        /// <see cref="DeepSeek4Architecture.ExecutorKvCacheDtype"/>). All construction-time.
        /// </summary>
        public override string KVStateFingerprint =>
            $"deepseek4|arch={Config.Architecture}|L={Config.NumLayers}|H={Config.NumHeads}|D={Config.KeyLength}" +
            $"|hidden={Config.HiddenSize}|experts={Config.NumExperts}x{Config.NumExpertsUsed}|swa={Config.SlidingWindow}" +
            $"|exec={(_cudaExec != null ? "cuda" : _cpuExec != null ? "cpu" : "native")}|align={_truncateAlign}" +
            $"|dspark={DraftBlockSize}|dtype={_kvCacheDtype.ToShortString()}";

        /// <summary>
        /// Refusable truncation. A refusal is normal - it means the target is further back
        /// than the slot's checkpoint can reach - and the caller resets and re-prefills.
        /// </summary>
        protected override bool TryTruncateKVCacheCore(int tokenCount)
        {
            lock (_sync)
            {
                if (!SupportsKVCacheTruncation) return false;
                return _slotExecutor.Truncate(tokenCount);
            }
        }

        /// <summary>
        /// The non-refusable form, for callers with no re-prefill fallback. It throws
        /// rather than returning with the head where it was: continuing from a stale head
        /// answers from positions the caller believes it dropped, and nothing downstream
        /// could tell.
        /// </summary>
        protected override void TruncateKVCacheCore(int tokenCount)
        {
            if (!TryTruncateKVCacheCore(tokenCount))
            {
                throw new InvalidOperationException(
                    $"DeepSeek {Config.Architecture} cannot truncate its KV cache to {tokenCount} tokens " +
                    $"(head at {CacheSeqLen}). Use TryTruncateKVCache and re-prefill when it declines.");
            }
        }

        protected override float[] ForwardCore(int[] tokens)
        {
            lock (_sync)
            {
                if (_distributedTpAborted)
                    throw new InvalidOperationException("DeepSeek distributed execution was aborted after a node failure; restart every node before generating again.");
                try
                {
                    var logits = new float[Config.VocabSize];
                    if (_cudaExec != null)
                    {
                        _cudaExec.Forward(tokens, logits);
                    }
                    else if (_cpuExec != null)
                    {
                        _cpuExec.Forward(tokens, logits);
                    }
                    else if (!GgmlDeepSeek4Native.Forward(_handle, tokens, logits))
                    {
                        throw new InvalidOperationException("DeepSeek V4 native forward failed (see stderr).");
                    }
                    return logits;
                }
                catch (Exception error)
                {
                    // Covers failures before the native TP hook was reached
                    // (graph construction/allocation, for example) as well.
                    AbortDistributedTensorParallel(error);
                    throw;
                }
            }
        }

        protected override void ResetKVCacheCore()
        {
            lock (_sync)
            {
                if (_distributedTpAborted)
                    throw new InvalidOperationException("Distributed DeepSeek execution was aborted; restart every node before resetting.");
                try
                {
                    if (_cudaExec != null)
                        _cudaExec.Reset();
                    else if (_cpuExec != null)
                        _cpuExec.Reset();
                    else if (_handle != IntPtr.Zero)
                    {
                        if (!GgmlDeepSeek4Native.ResetChecked(_handle))
                            throw new InvalidOperationException("DSV4 native cache reset failed; cache remains unusable.");
                    }
                }
                catch (Exception error)
                {
                    AbortDistributedTensorParallel(error);
                    throw;
                }
            }
        }

        public override void WarmUpKernels()
        {
            // One tiny forward allocates the scheduler's compute buffers so the
            // first user request does not pay the allocation cost.
            if (NativeDistributedGroup?.NodeCount > 1) NativeDistributedGroup.Barrier();
            try
            {
                int bos = Tokenizer?.BosTokenId ?? 0;
                ForwardCore(new[] { bos });
            }
            catch (Exception ex)
            {
                Console.Error.WriteLine($"[dsv4] warmup forward failed: {ex.Message}");
                if (NativeDistributedGroup?.NodeCount > 1) throw;
            }
            finally
            {
                if (!_distributedTpAborted) ResetKVCacheCore();
            }
        }

        public override void Dispose()
        {
            lock (_sync)
            {
                _slotExecutor = null;
                if (_cudaExec != null)
                {
                    _cudaExec.Dispose();
                    _cudaExec = null;
                }
                if (_cpuExec != null)
                {
                    _cpuExec.Dispose();
                    _cpuExec = null;
                }
                if (_handle != IntPtr.Zero)
                {
                    GgmlDeepSeek4Native.Free(_handle);
                    _handle = IntPtr.Zero;
                }
                GC.KeepAlive(_distributedTpReduce);
                GC.KeepAlive(_distributedLayerTransfer);
            }
            base.Dispose();
        }
    }
}
