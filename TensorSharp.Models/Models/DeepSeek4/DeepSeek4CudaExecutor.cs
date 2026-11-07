// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// DeepSeek V4 (Flash) executor for the direct-CUDA backend.
//
// The model-side half of the DSV4 direct-CUDA path: opens the split-GGUF
// shards, parses the deepseek4 hyper-parameters, dequantizes the small
// tensors (norms, gates, sinks, APE tables, the router) to F32 host arrays,
// precomputes the raw/compress RoPE cos-sin tables (same YaRN math as
// DeepSeek4CpuExecutor.RopeCacheInit), and hands everything to
// TensorSharp.Cuda.Dsv4CudaEngine, which runs the forward pass with
// driver-API kernels — fully independent of ggml.
//
// The bulk weights are never staged in host RAM: each one is described to the
// engine as (shard, file offset, length) and the engine's loader pool streams
// it through pinned chunks straight into VRAM (layer-split across the visible
// GPUs). A 150 GiB model therefore loads with a ~2 GB host footprint.
// ---------------------------------------------------------------------------
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Cuda;

namespace TensorSharp.Models
{
    internal sealed unsafe class DeepSeek4CudaExecutor : IDisposable, IDsv41EngramSource, IDsv4SlotExecutor
    {
        private const int CsaRatio = 4;
        private const int HcaRatio = 128;
        // Hyper-connection residual streams; ParseHparams refuses any other count.
        private const int HcStreams = 4;

        // The checkpoint's shards (and a DSpark drafter's GGUF beside them).
        private GgufShardSet _gguf;

        // hparams
        private int _nLayer, _nEmbd, _nHead, _nVocab, _headDim, _nRot, _qLoraRank, _oGroups, _oLoraRank, _nSwa;
        private float _rmsEps;
        private int _nExpert, _nExpertUsed, _nFfExp, _hashLayerCount;
        private float _expertWeightsScale;
        private bool _expertWeightsNorm;
        private float[] _swigluClampExp = Array.Empty<float>();
        private float[] _swigluClampShexp = Array.Empty<float>();
        private int _idxNHead, _idxHeadSize, _idxTopK;
        private int[] _compressRatios = Array.Empty<int>();
        private float _compressRopeBase = 10000f, _ropeFreqBase = 10000f;
        private float _yarnFreqScale = 1f, _yarnExtFactor;
        private float _yarnBetaFast = 32f, _yarnBetaSlow = 1f;
        private int _nCtxOrig;
        private int _hcSinkhornIters = 20;
        private float _hcEps = 1e-6f;
        private float _compCorr0, _compCorr1;

        private Dsv4CudaEngine _engine;

        public int VocabSize => _nVocab;
        public int NPast => _engine.NPast;

        /// <summary>The engine this executor drives (tests read its route counts).</summary>
        internal Dsv4CudaEngine Engine => _engine;


        /// <summary>
        /// The engine's host-side quantized matmul for <c>--n-cpu-moe</c>
        /// layers. It lives here because the managed quantized kernels are in
        /// this assembly, which TensorSharp.Backends.Cuda cannot reference (the
        /// dependency runs the other way) — same inversion as
        /// <c>ICudaWeightSource</c>.
        /// </summary>
        internal sealed class HostMatMul : IDsv4HostMatMul
        {
            public static readonly HostMatMul Instance = new HostMatMul();

            /// <summary>Exactly the types TryAddmmQuantizedBatch has a direct kernel for; it
            /// returns false for every other one.</summary>
            public bool SupportsType(int ggmlType, int inDim)
                => ManagedQuantizedOps.TryGetActivationPlan((GgmlTensorType)ggmlType, inDim, out _);

            public bool TryMatMulBatch(int ggmlType, int inDim, int inputRowStride,
                ReadOnlySpan<IDsv4HostMatMul.Job> jobs)
            {
                if (jobs.Length == 0)
                    return true;
                var mapped = new ManagedQuantizedOps.QuantMatMulJob[jobs.Length];
                for (int i = 0; i < jobs.Length; i++)
                    mapped[i] = new ManagedQuantizedOps.QuantMatMulJob(
                        jobs[i].Weights, jobs[i].Input, jobs[i].Output,
                        jobs[i].OutDim, jobs[i].RowCount, jobs[i].OutputRowStride);
                return ManagedQuantizedOps.TryAddmmQuantizedBatch(ggmlType, inDim, inputRowStride, mapped);
            }
        }

        /// <param name="nCpuMoe">Routed-expert CPU offload policy: 0 none (the
        /// default — offload is opt-in, and a model that does not fit is refused
        /// with the number of layers that would make it fit), N the first N
        /// layers, <see cref="int.MaxValue"/> every layer, -1 auto (the fewest
        /// leading layers that make the model fit; opt-in only).</param>
        public DeepSeek4CudaExecutor(string ggufPath, int maxContext, int nUbatch, int nGpu, string dsparkPath = null,
            int nCpuMoe = 0)
        {
            var sw = Stopwatch.StartNew();
            bool stats = ParseEnvInt("TS_DSV4_LOAD_STATS", 0) != 0;
            void Mark(string phase)
            {
                if (stats)
                    Console.Error.WriteLine($"[dsv4-cuda]   +{sw.Elapsed.TotalSeconds,6:F1}s {phase}");
            }

            _gguf = new GgufShardSet(ggufPath, "dsv4-cuda", dsparkPath);
            try
            {
                ParseHparams();
                if (_isV41)
                    LoadEngramMetadata();
                Mark("shards opened / hparams parsed");

                // Large weights are read exactly once, on their way to VRAM, so the
                // engine streams them straight from the shards through pinned
                // chunks instead of staging the whole (hundreds of GB) model in
                // host RAM first. TS_DSV4_MMAP=1 opts back into the mmap path.
                bool stream = ParseEnvInt("TS_DSV4_MMAP", 0) == 0;
                if (stream)
                {
                    _gguf.DropPagesAfterRead = DecideLoadDropCache(null, null);
                    _gguf.BeginStreaming();
                    PrefetchedTensorNames = _gguf.PrefetchedNames;
                    Mark("small tensors prefetched");
                }

                int nCtx = maxContext > 0 ? maxContext : 16384;
                int ubatch = nUbatch > 0 ? nUbatch : 1024;

                var desc = BuildModelDesc(nCtx, ubatch);
                Mark("model desc built");
                _engine = new Dsv4CudaEngine(desc, nGpu, nCpuMoe);
                Mark("engine ready");

                // Everything lives in VRAM now; drop the host-side scraps.
                _gguf.EndLoad();

                Console.Error.WriteLine($"[dsv4-cuda] model ready in {sw.Elapsed.TotalSeconds:F1}s");
                StartEngramWarm();
            }
            catch
            {
                ReleaseRefusedLoad();
                throw;
            }
        }

        /// <summary>
        /// A constructor that throws is never disposed by its caller, and a refused load (a peer that does not fit, a
        /// module or placement the nodes disagree on, too little VRAM, an upload fault) would otherwise keep the
        /// shards open (locking the files on Windows), the prefetched small tensors (about 474 MB on the 1017) and
        /// any Engram mapping for the life of the process.
        /// </summary>
        private void ReleaseRefusedLoad()
        {
            try { Dispose(); }
            catch { /* the load's own error is the one to report */ }
        }

        /// <summary>The small tensors this load read ahead (names only), for tests.</summary>
        internal IReadOnlyList<string> PrefetchedTensorNames { get; private set; } = Array.Empty<string>();

        /// <summary>A node's part of a cross-node layer split, built by the model layer from its
        /// <see cref="ILayerSplitGroup"/>.</summary>
        internal sealed class DistributedStage
        {
            public int Nodes, NodeId, LocalDevices;
            public bool DriverRunsLastStage;
            public IDsv4StageLink Link;
            /// <summary>Cross-node agreement on an identity string (the model's
            /// ValidateDistributedExecutionConfig): the placement before any weight is read, and the
            /// engine's kernel module before any is uploaded.</summary>
            public Action<string> Agree;
            /// <summary>The build this node runs, which the stages agree on with the kernel module
            /// (<see cref="Dsv4StagePlan.Build"/>).</summary>
            public string Build;
        }

        /// <summary>
        /// Tests only: the global device boundaries a distributed load uses instead of the byte-balanced cut
        /// (entry g is the first layer of global device g, the last entry the layer count), so a test can put
        /// the cut where it wants the state to cross. Every node of the test sees the same value.
        /// </summary>
        internal static int[] ForceStageBoundaries { get; set; }

        /// <summary>The global device of every layer this load placed, null on a single-process load.</summary>
        internal int[] GlobalDeviceOfLayer { get; private set; }

        /// <summary>
        /// One node of a cross-node layer split (DeepSeek V4.1 only): the placement is computed from the
        /// shards' headers (<see cref="Dsv4StagePlacement.Plan"/>), so every node derives the same one; this
        /// node then describes, maps and uploads only its own layers. Routed experts stay in VRAM and Engram
        /// tables are host mappings unless TS_DSV41_ENGRAM_DEVICE=1.
        /// </summary>
        public DeepSeek4CudaExecutor(string ggufPath, int maxContext, int nUbatch, DistributedStage distributed)
        {
            ArgumentNullException.ThrowIfNull(distributed);
            var sw = Stopwatch.StartNew();
            _gguf = new GgufShardSet(ggufPath, "dsv4-cuda");
            try
            {
                ParseHparams();
                if (!_isV41)
                    throw new NotSupportedException("[dsv4-cuda] cross-node layer split is implemented for DeepSeek V4.1 only.");
                LoadEngramMetadata();

                int nCtx = maxContext > 0 ? maxContext : 16384;
                int ubatch = nUbatch > 0 ? nUbatch : 1024;
                bool engramOnDevice = EngramPlacement() == true;
                Dsv4Topology topology = BuildTopology(nCtx, ubatch);
                var (layerBytes, embeddingBytes, headBytes) = HeaderBytes(engramOnDevice);
                int devices = checked(distributed.Nodes * distributed.LocalDevices);
                int[] globalDevice = ForceStageBoundaries is { } forced
                    ? Dsv4StagePlacement.FromBoundaries(forced, _nLayer)
                    : Dsv4StagePlacement.Plan(topology, layerBytes, embeddingBytes, headBytes,
                        distributed.Nodes, distributed.LocalDevices, distributed.DriverRunsLastStage);
                if (globalDevice[^1] != devices - 1)
                    throw new InvalidOperationException($"[dsv4-cuda] the layer split names {globalDevice[^1] + 1} devices; the run has {devices}.");
                GlobalDeviceOfLayer = globalDevice;
                int stage = LayerSplitPlan.StageOf(distributed.NodeId, distributed.Nodes, distributed.DriverRunsLastStage);
                int[] boundaries = Dsv4StagePlacement.Boundaries(globalDevice, devices);

                // Which node keeps each Engram table (host-mapped tables only; TS_DSV41_ENGRAM_OWNER): a pure function
                // of the placement and the shard headers, so every node computes the same owners.
                _stageIndex = stage;
                _engramOwner = Dsv4StagePlacement.EngramOwners(topology, globalDevice, distributed.LocalDevices,
                    distributed.Nodes, EngramTableBytes(), engramOnDevice ? "local" : EngramOwnerMode());

                // Before any weight is read: every node must have computed the same placement from the same bytes.
                ulong bytesHash = Fnv1a(layerBytes, embeddingBytes, headBytes);
                distributed.Agree?.Invoke($"dsv4-cuda-placement|devices={string.Join(",", boundaries)}" +
                    $"|assignment={Dsv4StagePlacement.AssignmentHash(globalDevice):X16}|bytes={bytesHash:X16}" +
                    $"|ctx={nCtx}|ubatch={ubatch}|engram_device={engramOnDevice}|engram_owners={string.Join(",", _engramOwner)}");
                for (int t = 0; t < _engramOwner.Length; t++)
                    Console.Error.WriteLine($"[dsv4-cuda] Engram table {t} (layer {_engram.Layers[t].Id}): kept by stage {_engramOwner[t]}" +
                        (_engramOwner[t] == stage ? " (this node)" : string.Empty));
                Console.Error.WriteLine($"[dsv4-cuda] cross-node layer split: global device boundaries {string.Join(",", boundaries)}; " +
                    $"node {distributed.NodeId} runs stage {stage} of {distributed.Nodes}");
                // Every node computes the same verdict from the same topology, so all refuse here together, before
                // anything is read or allocated, when a payload would not fit one transport frame.
                Dsv4StagePlacement.ValidateWireLimits(topology, globalDevice, distributed.LocalDevices);

                int first = stage * distributed.LocalDevices, end = first + distributed.LocalDevices;
                bool Owns(int il) => il < _nLayer && globalDevice[il] >= first && globalDevice[il] < end;
                if (ParseEnvInt("TS_DSV4_MMAP", 0) == 0)
                {
                    // Only this stage's layers' small tensors (and the non-layer ones): another stage's norms,
                    // gates and routers are never read here.
                    _gguf.DropPagesAfterRead = DecideLoadDropCache(Owns, engramOnDevice ? null : KeepsEngramTable);
                    _gguf.BeginStreaming(name => !LayerSplitPlan.TryGetLayer(name, out int il) || Owns(il));
                    PrefetchedTensorNames = _gguf.PrefetchedNames;
                }
                var desc = BuildModelDesc(nCtx, ubatch, Owns, engramOnDevice);
                // A table this node keeps for another node's layer: mapped here, though its layer is not.
                for (int t = 0; t < _engramOwner.Length; t++)
                {
                    if (_engramOwner[t] == stage && _engramTables[t].Base == null)
                        _engramTables[t] = MapEngramTable($"blk.{_engram.Layers[t].Id}.engram_embd.weight");
                }
                _engine = new Dsv4CudaEngine(desc, new Dsv4StagePlan
                {
                    GlobalDeviceOfLayer = globalDevice,
                    Stage = stage,
                    StageCount = distributed.Nodes,
                    LocalDevices = distributed.LocalDevices,
                    Link = distributed.Link,
                    Agree = distributed.Agree,
                    Build = distributed.Build,
                    EngramOwnerStage = _engramOwner,
                });
                _gguf.EndLoad();
                Console.Error.WriteLine($"[dsv4-cuda] stage {stage} ready in {sw.Elapsed.TotalSeconds:F1}s");
                StartEngramWarm();
            }
            catch
            {
                ReleaseRefusedLoad();
                throw;
            }
        }

        /// <summary>The topology the cross-node placement is computed from: the engine derives the same one
        /// from the descriptors this executor hands it.</summary>
        private Dsv4Topology BuildTopology(int nCtx, int nUbatch)
        {
            var t = new Dsv4Topology
            {
                NLayer = _nLayer, NEmbd = _nEmbd, HeadDim = _headDim, IdxHeadSize = _idxHeadSize, IdxTopK = _idxTopK,
                NUbatch = nUbatch, NCtx = nCtx, NSwa = _nSwa, V41 = _isV41,
                NVocab = (int)(_gguf.InfoOf("token_embd.weight")?.Shape[1] ?? 0),
                CandidateSource = _isV41 ? _engram.CandidateSourceLayerId : -1,
                Ratio = new int[_nLayer], KvSource = new int[_nLayer], IndexSource = new int[_nLayer], EngramIndex = new int[_nLayer],
            };
            for (int il = 0; il < _nLayer; il++)
            {
                t.Ratio[il] = _compressRatios[il];
                t.KvSource[il] = _isV41 ? _v41KvSource[il] : -1;
                t.IndexSource[il] = _isV41 ? _v41IndexSource[il] : -1;
                t.EngramIndex[il] = EngramIndexOf(il);
            }
            return t;
        }

        /// <summary>The Engram table of layer <paramref name="il"/>, -1 for none.</summary>
        private int EngramIndexOf(int il)
        {
            if (!_isV41)
                return -1;
            for (int t = 0; t < _engram.Layers.Length; t++)
            {
                if (_engram.Layers[t].Id == il)
                    return t;
            }
            return -1;
        }

        /// <summary>
        /// What each layer, the embedding and the head occupy on a GPU, from the shards' headers alone: every
        /// tensor of the layer as the engine places it (256-byte aligned), the Engram table only when it goes
        /// to VRAM. The same on every node, which free VRAM is not.
        /// </summary>
        private (long[] Layers, long Embedding, long Head) HeaderBytes(bool engramOnDevice)
        {
            var layers = new long[_nLayer];
            long embedding = 0, head = 0;
            static long Align(long v) => (v + 255) & ~255L;
            foreach (var (name, _, bytes) in _gguf.TensorBytes())
            {
                if (LayerSplitPlan.TryGetLayer(name, out int il))
                {
                    if (il >= _nLayer || (!engramOnDevice && name.EndsWith(".engram_embd.weight", StringComparison.Ordinal)))
                        continue;
                    layers[il] = checked(layers[il] + Align(bytes));
                }
                else if (name == "token_embd.weight")
                    embedding = checked(embedding + Align(bytes));
                else if (name is "output.weight" or "output_norm.weight")
                    head = checked(head + Align(bytes));
            }
            return (layers, embedding, head);
        }

        private static ulong Fnv1a(long[] values, params long[] more)
        {
            ulong h = 1469598103934665603UL;
            void Mix(long v)
            {
                for (int i = 0; i < 8; i++)
                {
                    h ^= (ulong)(v >> (8 * i)) & 0xff;
                    h *= 1099511628211UL;
                }
            }
            foreach (long v in values)
                Mix(v);
            foreach (long v in more)
                Mix(v);
            return h;
        }

        private static int ParseEnvInt(string name, int fallback)
        {
            string raw = Environment.GetEnvironmentVariable(name);
            return int.TryParse(raw, out int v) ? v : fallback;
        }

        public void Forward(int[] tokens, float[] logitsOut) => _engine.Forward(tokens, logitsOut);

        public void Reset() => _engine.Reset();

        // ---- DSpark speculative decoding (no-ops without a drafter) ----

        public bool HasDspark => _engine.DsparkBlockSize > 0;

        public int DsparkBlockSize => _engine.DsparkBlockSize;

        /// <summary>Row width of the target features the drafter consumes.</summary>
        public int DsparkFeatureSize => _engine.DsparkFeatureSize;

        public int UBatch => _engine.UBatch;

        public void ForwardSpec(int[] tokens, float[] hAllOut, float[] logitsOut, bool allLogitsRows)
            => _engine.ForwardSpec(tokens, hAllOut, logitsOut, allLogitsRows);

        public void DsparkCatchUp(float[] hRows, int rows, int firstPos)
            => _engine.DsparkCatchUp(hRows, rows, firstPos);

        public int DsparkDraft(int anchorToken, float[] hPrev, int position, int[] draftOut, float[] confOut)
            => _engine.DsparkDraft(anchorToken, hPrev, position, draftOut, confOut);

        public void Rewind(int nPast) => _engine.Rewind(nPast);

        // ---- sequence slots (the engine's; see Dsv4CudaEngine.Slots.cs) ----

        /// <summary>The multiple a truncation target must be, 0 when this model cannot truncate.</summary>
        public int TruncateAlign => _engine.TruncateAlign;

        /// <summary>The slot Forward, Reset and Truncate act on.</summary>
        public int ActiveSlot => _engine.ActiveSlot;

        public int SlotAlloc() => _engine.SlotAlloc();

        public bool SlotStatus(int slot, out int head, out int checkpoint, out bool healthy)
            => _engine.SlotStatus(slot, out head, out checkpoint, out healthy);

        public bool SlotCanReuse(int slot, int cachedHead, int target) => _engine.SlotCanReuse(slot, cachedHead, target);

        public bool SlotCanRetain(int slot, int retainedCount, ulong budgetPerDevice)
            => _engine.SlotCanRetain(slot, retainedCount, budgetPerDevice);

        public bool SlotCanAlloc() => _engine.SlotCanAlloc();

        /// <summary>The engine captures no graphs: a slot is its caches.</summary>
        public bool SlotReleaseGraphs(int slot) => _engine.SlotStatus(slot, out _, out _, out _);

        public bool SetActiveSlot(int slot) => _engine.SetActiveSlot(slot);

        public bool SlotFree(int slot) => _engine.SlotFree(slot);

        public bool ResetChecked() => _engine.ResetChecked();

        public bool Truncate(int nPast) => _engine.Truncate(nPast);

        public bool ForwardBatchedDecode(int[] slots, int[] tokens, int[] positions, float[] logits)
            => _engine.ForwardBatchedDecode(slots, tokens, positions, logits);

        // -------------------------------------------------------------------
        // Loading (mirrors DeepSeek4CpuExecutor's split-shard resolver)
        // -------------------------------------------------------------------

        private void ParseHparams()
        {
            GgufFile g = _gguf.First;
            // V4 and V4.1 are separate architectures with separate key prefixes.
            string arch = g.GetString("general.architecture", "deepseek4");
            if (arch != "deepseek4" && arch != "deepseek41")
                throw new NotSupportedException(
                    $"The DeepSeek CUDA executor requires the deepseek4 or deepseek41 architecture, got '{arch}'.");
            _isV41 = arch == "deepseek41";
            string a = arch;
            _nLayer = (int)g.GetUint32($"{a}.block_count");
            _nEmbd = (int)g.GetUint32($"{a}.embedding_length");
            _nHead = (int)g.GetUint32($"{a}.attention.head_count");
            _headDim = (int)g.GetUint32($"{a}.attention.key_length");
            _nRot = (int)g.GetUint32($"{a}.rope.dimension_count");
            _qLoraRank = (int)g.GetUint32($"{a}.attention.q_lora_rank");
            _oGroups = (int)g.GetUint32($"{a}.attention.output_group_count");
            _oLoraRank = (int)g.GetUint32($"{a}.attention.output_lora_rank");
            _nSwa = (int)g.GetUint32($"{a}.attention.sliding_window");
            _rmsEps = g.GetFloat32($"{a}.attention.layer_norm_rms_epsilon", 1e-6f);
            _nExpert = (int)g.GetUint32($"{a}.expert_count");
            _nExpertUsed = (int)g.GetUint32($"{a}.expert_used_count");
            _nFfExp = (int)g.GetUint32($"{a}.expert_feed_forward_length");
            _expertWeightsScale = g.GetFloat32($"{a}.expert_weights_scale", 1.0f);
            _expertWeightsNorm = g.GetBool($"{a}.expert_weights_norm");
            _hashLayerCount = (int)g.GetUint32($"{a}.hash_layer_count");
            _idxNHead = (int)g.GetUint32($"{a}.attention.indexer.head_count");
            _idxHeadSize = (int)g.GetUint32($"{a}.attention.indexer.key_length");
            _idxTopK = (int)g.GetUint32($"{a}.attention.indexer.top_k");
            _compressRatios = g.GetInt32Array($"{a}.attention.compress_ratios") ?? Array.Empty<int>();
            _compressRopeBase = g.GetFloat32($"{a}.attention.compress_rope_freq_base", 10000f);
            _hcSinkhornIters = (int)g.GetUint32($"{a}.hyper_connection.sinkhorn_iterations", 20);
            _hcEps = g.GetFloat32($"{a}.hyper_connection.epsilon", 1e-6f);
            _swigluClampExp = g.GetFloatArray($"{a}.swiglu_clamp_exp") ?? Array.Empty<float>();
            _swigluClampShexp = g.GetFloatArray($"{a}.swiglu_clamp_shexp") ?? _swigluClampExp;
            _ropeFreqBase = g.GetFloat32($"{a}.rope.freq_base", 10000f);

            float yarnFactor = g.GetFloat32($"{a}.rope.scaling.factor", 0f);
            if (yarnFactor > 0f)
            {
                _yarnFreqScale = 1.0f / yarnFactor;
                _yarnExtFactor = 1.0f;
            }
            _nCtxOrig = (int)g.GetUint32($"{a}.rope.scaling.original_context_length");
            _yarnBetaFast = g.GetFloat32($"{a}.rope.scaling.yarn_beta_fast", 32f);
            _yarnBetaSlow = g.GetFloat32($"{a}.rope.scaling.yarn_beta_slow", 1f);

            int hc = (int)g.GetUint32($"{a}.hyper_connection.count", 4);
            if (hc != 4)
                throw new NotSupportedException($"DeepSeek4 CUDA executor supports hyper_connection.count == 4, got {hc}.");
            if (_nLayer <= 0 || _compressRatios.Length < _nLayer)
                throw new InvalidOperationException("Missing or invalid deepseek4 GGUF metadata.");

            _compCorr0 = MathF.Max(0f, MathF.Floor(YarnCorrDim(_nRot, _nCtxOrig, _yarnBetaFast, _compressRopeBase)));
            _compCorr1 = MathF.Min(_nRot - 1, MathF.Ceiling(YarnCorrDim(_nRot, _nCtxOrig, _yarnBetaSlow, _compressRopeBase)));
        }

        /// <summary>deepseek41 rather than deepseek4.</summary>
        private bool _isV41;
        private Dsv41EngramData _engram;
        private int[] _v41KvSource, _v41IndexSource;
        private int[] _engramHashes;
        private int _engramUbatchTokens;

        /// <summary>Reads embedded Engram metadata and derives cache-sharing
        /// topology from the checkpoint's tensor ownership across all shards.</summary>
        private void LoadEngramMetadata()
        {
            _engram = Dsv41EngramData.Load(_gguf.First, _gguf.InfoOf);
            // llama.cpp's layout: each Engram layer's table is a row range of one combined
            // tensor. Name those ranges as TensorSharp's layout names its tables, as views of
            // the same shard bytes, and drop the combined entry so nothing streams or stages it
            // whole. From here on (placement, mapping, warming) both layouts read alike, which
            // is what the native loader does too.
            if (_engram.Slices != null)
                _gguf.ReplaceWithRowViews(Dsv41EngramData.CombinedTableName, _engram.TableViews());

            if (_engram.Layers[^1].Id >= _nLayer ||
                _engram.KvSourceLayerIds[^1] >= _nLayer || _engram.IndexSourceLayerIds[^1] >= _nLayer)
                throw new InvalidOperationException("DeepSeek V4.1 GGUF metadata names a layer beyond the layer count");

            _v41KvSource = new int[_nLayer];
            _v41IndexSource = new int[_nLayer];
            int kv = -1, index = -1;
            for (int il = 0; il < _nLayer; il++)
            {
                if (Array.IndexOf(_engram.KvSourceLayerIds, il) >= 0) kv = il;
                if (Array.IndexOf(_engram.IndexSourceLayerIds, il) >= 0) index = il;
                int ratio = _compressRatios[il];
                if (ratio < 0 || ratio > 2)
                    throw new NotSupportedException($"Invalid DeepSeek V4.1 compression ratio {ratio} on layer {il}.");
                if (ratio != 0 && (kv < 0 || index < 0 ||
                    _compressRatios[kv] != ratio || _compressRatios[index] != ratio))
                    throw new InvalidOperationException(
                        "DeepSeek V4.1 cache-sharing topology does not match the compression ratios");
                _v41KvSource[il] = ratio != 0 ? kv : -1;
                _v41IndexSource[il] = ratio != 0 ? index : -1;
            }
            _engramTables = new EngramTable[_engram.Layers.Length];
            _engramRemote = new RemoteRows[_engram.Layers.Length];
            _engramReader = Dsv41EngramReader.FromEnvironment();
            Console.Error.WriteLine($"[dsv4-cuda] Engram host reads: {_engramReader.Describe()}");
        }

        // -------------------------------------------------------------------
        // IDsv41EngramSource: the host side of the Engram lookup
        //
        // An Engram table is hundreds of millions of rows and tens to hundreds
        // of GiB. It stays a host mapping and only the rows a token actually
        // selects are dequantized and handed to the engine, which uploads them.
        // -------------------------------------------------------------------

        private struct EngramTable
        {
            public byte* Base;
            public GgmlTensorType Type;
            public long RowBytes;
            public long Rows;

            public Dsv41EngramReader.Table View => new Dsv41EngramReader.Table(Base, (int)Type, RowBytes, Rows);
        }

        /// <summary>The rows another node sent for one table this forward (<see cref="ImportEngramRows"/>).</summary>
        private sealed class RemoteRows
        {
            public int StartPos, Count;
            public int[] Ids = Array.Empty<int>();
            public byte[] Raw = Array.Empty<byte>();
        }

        // Per table: the mapping (Base null when another node keeps it), and the rows that node sent this forward.
        private EngramTable[] _engramTables;
        private RemoteRows[] _engramRemote;
        private Dsv41EngramReader _engramReader;
        // A distributed stage: its index and each table's keeper (null on a single-process load: all local).
        private int _stageIndex = -1;
        private int[] _engramOwner;
        private int _engramUbatchStart;
        // Each host-mapped table's bytes in its shard and its mapping, for the warm pass.
        private readonly List<MappedTableWarm.Range> _engramRanges = new List<MappedTableWarm.Range>();
        private Task _engramWarm;
        private readonly CancellationTokenSource _engramWarmCancel = new CancellationTokenSource();

        /// <summary>The Engram reader's settings (tests).</summary>
        internal Dsv41EngramReader EngramReader => _engramReader;

        /// <summary>Each Engram table's keeping stage on a distributed load (null on a single-process one).</summary>
        internal int[] EngramOwners => _engramOwner;

        /// <summary>Whether this node maps table <paramref name="engramIndex"/> (tests).</summary>
        internal bool MapsEngramTable(int engramIndex) => _engramTables != null && _engramTables[engramIndex].Base != null;

        /// <summary>Whether this node keeps (maps and gathers from) Engram table <paramref name="t"/>.</summary>
        private bool KeepsEngramTable(int t) => _engramOwner == null || _engramOwner[t] == _stageIndex;

        /// <summary>The bytes of received Engram rows a table keeps between forwards: a forward that needs at most half of
        /// a larger buffer gets a buffer of its own size, so one long CLI prompt does not hold its rows for the life of
        /// the process (tests lower it).</summary>
        internal static long RemoteRowsKeepBytes = 32L << 20;

        /// <summary>The bytes the received rows of every table hold now (tests).</summary>
        internal long RemoteRowsBytes
        {
            get
            {
                long bytes = 0;
                foreach (RemoteRows r in _engramRemote ?? Array.Empty<RemoteRows>())
                    bytes += r == null ? 0 : r.Raw.LongLength + 4L * r.Ids.Length;
                return bytes;
            }
        }

        /// <summary>TS_DSV41_ENGRAM_OWNER: <c>local</c> (unset) or <c>balance</c>; see Dsv4StagePlacement.EngramOwners.</summary>
        internal static string EngramOwnerMode() => Environment.GetEnvironmentVariable("TS_DSV41_ENGRAM_OWNER")?.Trim();

        /// <summary>Each Engram table's bytes, from the shard headers (the same on every node).</summary>
        private long[] EngramTableBytes()
        {
            var bytes = new long[_engram.Layers.Length];
            for (int t = 0; t < bytes.Length; t++)
            {
                GgufTensorInfo info = _gguf.InfoOf($"blk.{_engram.Layers[t].Id}.engram_embd.weight")
                    ?? throw new InvalidOperationException($"[dsv4-cuda] missing tensor: blk.{_engram.Layers[t].Id}.engram_embd.weight");
                bytes[t] = ManagedQuantizedOps.RowSize((int)info.Type, (int)info.Shape[0]) * (long)(info.Shape.Length > 1 ? info.Shape[1] : 1);
            }
            return bytes;
        }

        /// <summary>
        /// TS_DSV4_LOAD_DROP_CACHE (LoadPageCachePolicy, the native loader's rule): whether the loader drops each weight
        /// chunk's page cache once it has read it (Linux; shards on tmpfs are left alone). Automatic: when the bytes this
        /// node uploads plus the Engram tables it keeps host-mapped plus 8 GiB exceed its cgroup limit. On the 1017's
        /// 100 GB VMs a node uploads ~76 GiB and keeps a 63 GiB slice, so it drops, and the next process's weight read no
        /// longer pushes the warm slice out.
        /// </summary>
        /// <param name="owns">The layers this node loads (null: all).</param>
        /// <param name="keepsTable">The Engram tables this distributed stage keeps host-mapped (null: none). A
        /// single-process load keeps every table host-mapped unless TS_DSV41_ENGRAM_DEVICE=1.</param>
        private bool DecideLoadDropCache(Func<int, bool> owns, Func<int, bool> keepsTable)
        {
            long upload = 0, mapped = 0;
            var tables = new Dictionary<string, int>(StringComparer.Ordinal);
            if (_isV41)
                for (int t = 0; t < _engram.Layers.Length; t++)
                    tables[$"blk.{_engram.Layers[t].Id}.engram_embd.weight"] = t;
            bool singleHostTables = owns == null && _isV41 && EngramPlacement() != true;
            foreach (var (name, _, bytes) in _gguf.TensorBytes())
            {
                bool layer = LayerSplitPlan.TryGetLayer(name, out int il);
                if (tables.TryGetValue(name, out int t))
                {
                    if (owns == null ? singleHostTables : keepsTable != null && keepsTable(t))
                        mapped += bytes;
                    else if (owns == null || (layer && owns(il)))
                        upload += bytes;
                    continue;
                }
                if (owns != null && layer && !owns(il))
                    continue;
                upload += bytes;
            }
            long allowance = LoadPageCachePolicy.Allowance();
            var decision = LoadPageCachePolicy.Decide(Environment.GetEnvironmentVariable(LoadPageCachePolicy.Variable),
                upload, mapped, allowance);
            if (OperatingSystem.IsLinux())
                Console.Error.WriteLine("[dsv4-cuda] load page cache: " + LoadPageCachePolicy.Describe(decision, upload, mapped, allowance));
            return decision.Drop;
        }

        /// <summary>
        /// A host-mapped Engram table is read a few scattered rows per token, and a row whose page is not
        /// cached is a storage round trip: about a millisecond on a network filesystem, so a cold table
        /// halves decode. Read the tables once after the model is ready, as the native loader does
        /// (dsv4_warm_engram), when this process's cgroup (and the host) has room to keep them cached; only the
        /// tables this node keeps, and only the pages that are not already resident (MappedTableWarm).
        /// TS_DSV41_ENGRAM_WARM is read as the native loader reads it (atoi): unset warms in the background,
        /// <c>0</c> never warms, and a positive value warms before the model is handed out. Ignoring it read
        /// the whole 63 GiB table of the 1017 behind every request of a run that asked for no warm, and filled
        /// a 100 GB cgroup that MemAvailable (host-wide) said had room. TS_DSV41_ENGRAM_WARM_THREADS (1-64,
        /// default 16) sets how many threads read, each on its own file handles.
        /// </summary>
        private void StartEngramWarm()
        {
            if (_engramRanges.Count == 0 || _engine.EngramResident)
                return;
            int mode = EngramWarmMode();
            if (mode == 0)
            {
                Console.Error.WriteLine("[dsv4-cuda] Engram tables are not warmed (TS_DSV41_ENGRAM_WARM=0)");
                return;
            }
            var options = new MappedTableWarm.Options { Threads = EngramWarmThreads() };
            _engramWarm = MappedTableWarm.Start(_engramRanges, "dsv4-cuda", "Engram tables", _engramWarmCancel.Token, options);
            if (mode > 0)
                _engramWarm?.Wait();
        }

        /// <summary>TS_DSV41_ENGRAM_WARM as the native loader's atoi reads it: -1 when unset (background),
        /// 0 for off (and for anything that is not a number, as atoi), the value otherwise (synchronous).</summary>
        internal static int EngramWarmMode()
        {
            string option = Environment.GetEnvironmentVariable("TS_DSV41_ENGRAM_WARM");
            if (option == null)
                return -1;
            return int.TryParse(option.Trim(), out int v) ? v : 0;
        }

        /// <summary>TS_DSV41_ENGRAM_WARM_THREADS: 1-64, default 16.</summary>
        internal static int EngramWarmThreads()
        {
            string option = Environment.GetEnvironmentVariable("TS_DSV41_ENGRAM_WARM_THREADS");
            if (option == null)
                return MappedTableWarm.DefaultThreads;
            if (!int.TryParse(option.Trim(), out int v) || v < 1 || v > 64)
                throw new NotSupportedException($"TS_DSV41_ENGRAM_WARM_THREADS must be in [1, 64], got '{option}'.");
            return v;
        }

        public int HashColumns => (int)_engram.HashColumns;

        public int HeadDim => (int)_engram.HeadDim;

        public void BeginEngramUbatch(Dsv41EngramHistory history, ReadOnlySpan<int> tokens, int startPos)
        {
            _engramHashes = _engram.HashTokens(tokens, startPos, ref history.Tokens, ref history.Length);
            _engramUbatchTokens = tokens.Length;
            _engramUbatchStart = startPos;
        }

        public void BeginEngramRows(Dsv41EngramHistory[] histories, ReadOnlySpan<int> tokens, ReadOnlySpan<int> positions)
        {
            // Hash each row against its own sequence, then lay the rows out as one ubatch:
            // [table][row][column], the order GatherEngramRows reads.
            int n = tokens.Length, columns = HashColumns, tables = _engram.Layers.Length;
            var hashes = new int[tables * n * columns];
            for (int i = 0; i < n; i++)
            {
                Dsv41EngramHistory history = histories[i];
                int[] row = _engram.HashTokens(tokens.Slice(i, 1), positions[i], ref history.Tokens, ref history.Length);
                for (int t = 0; t < tables; t++)
                    Array.Copy(row, t * columns, hashes, ((long)t * n + i) * columns, columns);
            }
            _engramHashes = hashes;
            _engramUbatchTokens = n;
            _engramUbatchStart = -1;   // rows of several sequences: never served from another node's rows
        }

        public void CopyEngramRowIndices(int engramIndex, int count, int* dst)
        {
            if (_engramHashes == null || count > _engramUbatchTokens)
                throw new InvalidOperationException("[dsv4-cuda] Engram rows requested before the ubatch was hashed");
            int columns = HashColumns;
            long first = (long)engramIndex * _engramUbatchTokens * columns;
            fixed (int* hashes = _engramHashes)
                Buffer.MemoryCopy(hashes + first, dst, (long)count * columns * 4, (long)count * columns * 4);
        }

        private long _engramReadTicks;

        public long EngramReadTicks => _engramReadTicks;

        public void GatherEngramRows(int engramIndex, int count, float* dst)
        {
            long start = Stopwatch.GetTimestamp();
            try
            {
                GatherEngramRowsCore(engramIndex, count, dst);
            }
            finally
            {
                _engramReadTicks += Stopwatch.GetTimestamp() - start;
            }
        }

        private void GatherEngramRowsCore(int engramIndex, int count, float* dst)
        {
            if (_engramHashes == null || count > _engramUbatchTokens)
                throw new InvalidOperationException("[dsv4-cuda] Engram rows requested before the ubatch was hashed");

            EngramTable table = _engramTables[engramIndex];
            int columns = HashColumns, headDim = HeadDim;
            fixed (int* hashes = _engramHashes)
            {
                int* mine = hashes + (long)engramIndex * _engramUbatchTokens * columns;
                if (!KeepsEngramTable(engramIndex))
                {
                    GatherReceivedRows(engramIndex, table, mine, count, columns, headDim, dst);
                    return;
                }
                if (table.Base == null)
                    throw new InvalidOperationException($"[dsv4-cuda] Engram table {engramIndex} is not mapped on this node");
                // Each (token, column) is an independent random row: the one place
                // this executor touches the big table at all.
                _engramReader.Gather(table.View, mine, count, columns, headDim, dst);
            }
        }

        /// <summary>
        /// A table another node keeps: this ubatch's rows are in what it sent at the start of the forward. Their ids must be
        /// this node's own hashes of the same positions (both nodes hash the same tokens with the same history), which
        /// catches nodes that fell out of step before a wrong row is used; then each row is dequantized by the same function
        /// a local gather calls, so the floats are the same.
        /// </summary>
        private void GatherReceivedRows(int engramIndex, EngramTable table, int* mine, int count, int columns, int headDim, float* dst)
        {
            RemoteRows received = _engramRemote[engramIndex];
            int offset = _engramUbatchStart - (received?.StartPos ?? 0);
            if (received == null || _engramUbatchStart < 0 || offset < 0 || offset + count > received.Count)
                throw new InvalidOperationException(
                    $"[dsv4-cuda] Engram table {engramIndex} is kept by stage {_engramOwner[engramIndex]}, which sent no rows for " +
                    $"positions {_engramUbatchStart}..{_engramUbatchStart + count - 1}");
            int items = count * columns;
            fixed (int* ids = received.Ids)
            fixed (byte* raw = received.Raw)
            {
                int* theirs = ids + (long)offset * columns;
                for (int i = 0; i < items; i++)
                {
                    if (theirs[i] != mine[i])
                        throw new InvalidOperationException(
                            $"[dsv4-cuda] Engram table {engramIndex}: stage {_engramOwner[engramIndex]} sent row {theirs[i]} for position " +
                            $"{_engramUbatchStart + i / columns}, column {i % columns}, where this node hashes {mine[i]}. The nodes' token " +
                            "histories differ; restart every node.");
                }
                // The received rows, as a table whose row k is the k-th row of this ubatch.
                var view = new Dsv41EngramReader.Table(raw + (long)offset * columns * table.RowBytes, (int)table.Type, table.RowBytes, items);
                var order = new int[items];
                for (int i = 0; i < items; i++)
                    order[i] = i;
                fixed (int* rows = order)
                    _engramReader.Gather(view, rows, count, columns, headDim, dst);
            }
        }

        public void PrefetchEngramRows(Dsv41EngramHistory history, ReadOnlySpan<int> tokens, int startPos, int ubatch)
        {
            if (_engramReader == null || !_engramReader.Prefetch || tokens.Length == 0 || ubatch <= 0)
                return;
            int tables = _engram.Layers.Length, columns = HashColumns, n = tokens.Length;
            bool any = false;
            for (int t = 0; t < tables; t++)
                any |= KeepsEngramTable(t) && _engramTables[t].Base != null;
            if (!any)
                return;
            int[] hashes;
            try
            {
                hashes = _engram.PreviewHashes(tokens, startPos, history.Tokens, history.Length);
            }
            catch (Exception ex) when (ex is InvalidOperationException or ArgumentOutOfRangeException)
            {
                return;   // the forward's own hashing reports it
            }
            // The order the layers ask: every table of ubatch 0 (in layer order, which is table order), then ubatch 1, ...
            var segments = new List<Dsv41EngramReader.Segment>();
            for (int u = 0; u < n; u += ubatch)
            {
                int nt = Math.Min(ubatch, n - u);
                for (int t = 0; t < tables; t++)
                {
                    if (KeepsEngramTable(t) && _engramTables[t].Base != null)
                        segments.Add(new Dsv41EngramReader.Segment(_engramTables[t].View, hashes, (t * n + u) * columns, nt * columns));
                }
            }
            _engramReader.PrefetchRows(segments);
        }

        public void CancelEngramPrefetch() => _engramReader?.CancelPrefetch();

        /// <summary>
        /// Before a page-cache eviction of the Engram tables (ModelBase.EvictHostTablePageCache): stop every read this
        /// executor would make of them on its own, so nothing reads the dropped pages straight back. The read-ahead of
        /// the last forward is abandoned and waited for (each thread finishes the row it is on), and a background warm
        /// that is still running is cancelled and waited for (it stops after its current 64 MiB blocks): a cold row
        /// asks for cold tables. The forward's own gathers run only inside a forward, never during an eviction.
        /// </summary>
        internal void QuiesceEngramReads()
        {
            _engramReader?.CancelPrefetch(wait: true);
            if (_engramWarm is { IsCompleted: false } warm)
            {
                Console.Error.WriteLine("[dsv4-cuda] a page-cache eviction stops the background Engram warm");
                _engramWarmCancel.Cancel();
                try { warm.Wait(); }
                catch (AggregateException) { }
            }
        }

        public long EngramTransferBytes(int engramIndex, int count)
        {
            long items = (long)count * HashColumns;
            long bytes = items * 4 + items * _engramTables[engramIndex].RowBytes;
            return (bytes + 3) & ~3L;
        }

        public void ExportEngramRows(int engramIndex, int count, byte* dst)
        {
            if (_engramHashes == null || count != _engramUbatchTokens)
                throw new InvalidOperationException("[dsv4-cuda] Engram rows exported before the forward was hashed");
            EngramTable table = _engramTables[engramIndex];
            if (!KeepsEngramTable(engramIndex) || table.Base == null)
                throw new InvalidOperationException($"[dsv4-cuda] Engram table {engramIndex} is not kept by this node");
            int items = count * HashColumns;
            long start = Stopwatch.GetTimestamp();
            fixed (int* hashes = _engramHashes)
            {
                int* mine = hashes + (long)engramIndex * count * HashColumns;
                Buffer.MemoryCopy(mine, dst, (long)items * 4, (long)items * 4);
                _engramReader.CopyRaw(table.View, mine, items, dst + (long)items * 4);
            }
            _engramReadTicks += Stopwatch.GetTimestamp() - start;
            long used = (long)items * 4 + items * table.RowBytes;
            long padded = EngramTransferBytes(engramIndex, count);
            for (long b = used; b < padded; b++)
                dst[b] = 0;
        }

        public void ImportEngramRows(int engramIndex, int startPos, int count, byte* src)
        {
            if (KeepsEngramTable(engramIndex))
                throw new InvalidOperationException($"[dsv4-cuda] Engram table {engramIndex} is kept by this node; it takes no rows");
            int items = count * HashColumns;
            long rawBytes = items * _engramTables[engramIndex].RowBytes;
            RemoteRows r = _engramRemote[engramIndex] ??= new RemoteRows();
            // Grow to this forward; shrink back when a large buffer (one long prompt's rows) is mostly unused.
            static bool Resize(long have, long need) => have < need || (have > RemoteRowsKeepBytes && need <= have / 2);
            if (Resize(4L * r.Ids.Length, 4L * items))
                r.Ids = new int[items];
            if (Resize(r.Raw.LongLength, rawBytes))
                r.Raw = new byte[rawBytes];
            fixed (int* ids = r.Ids)
                Buffer.MemoryCopy(src, ids, (long)r.Ids.Length * 4, (long)items * 4);
            fixed (byte* raw = r.Raw)
                Buffer.MemoryCopy(src + (long)items * 4, raw, r.Raw.LongLength, rawBytes);
            r.StartPos = startPos;
            r.Count = count;
        }

        /// <summary>Maps one Engram table read-only (with MADV_RANDOM unless the reader is told otherwise). Falls back to
        /// a full staged copy only when the platform cannot map, which for a table this size would mean RAM the box does
        /// not have -- so say so rather than silently allocating.</summary>
        private EngramTable MapEngramTable(string name)
        {
            GgufTensorInfo info = _gguf.InfoOf(name);
            var (path, offset, bytes, mapped) = _gguf.MapTensor(name);
            if (_engramReader.RandomAdvice)
            {
                var status = PageCacheHints.AdviseRandom(mapped, bytes, out int error);
                if (status == PageCacheHints.AdviceStatus.Failed)
                    Console.Error.WriteLine($"[dsv4-cuda] note: MADV_RANDOM on {name} failed (error {error}); faults read ahead as before");
                else if (status == PageCacheHints.AdviceStatus.Applied)
                    Console.Error.WriteLine($"[dsv4-cuda] {name}: {bytes / (double)(1L << 30):F2} GiB mapped for random reads (MADV_RANDOM)");
            }
            _engramRanges.Add(new MappedTableWarm.Range(path, offset, bytes, mapped));
            return new EngramTable
            {
                Base = (byte*)mapped,
                Type = info.Type,
                RowBytes = ManagedQuantizedOps.RowSize((int)info.Type, (int)info.Shape[0]),
                Rows = info.Shape.Length > 1 ? (long)info.Shape[1] : 1,
            };
        }

        /// <summary>A table another node keeps: its row layout only, for the rows that node sends.</summary>
        private EngramTable DescribeRemoteEngramTable(string name)
        {
            GgufTensorInfo info = _gguf.InfoOf(name) ?? throw new InvalidOperationException($"[dsv4-cuda] missing tensor: {name}");
            return new EngramTable
            {
                Base = null,
                Type = info.Type,
                RowBytes = ManagedQuantizedOps.RowSize((int)info.Type, (int)info.Shape[0]),
                Rows = info.Shape.Length > 1 ? (long)info.Shape[1] : 1,
            };
        }

        /// <summary>
        /// TS_DSV41_ENGRAM_DEVICE, read as the native loader reads it: unset places the Engram tables in
        /// VRAM when the split holds them without offloading more experts, <c>0</c> keeps them host
        /// mappings (the path a host that cannot hold them takes, and the one a bit-exact comparison with
        /// a host oracle needs), <c>1</c> requires VRAM. Anything else is refused rather than ignored.
        /// </summary>
        internal static bool? EngramPlacement()
        {
            string option = Environment.GetEnvironmentVariable("TS_DSV41_ENGRAM_DEVICE");
            return option switch
            {
                null => null,
                "0" => false,
                "1" => true,
                _ => throw new NotSupportedException("TS_DSV41_ENGRAM_DEVICE must be 0 or 1 (unset selects automatically)."),
            };
        }

        private static float YarnCorrDim(int nDims, int nCtxOrig, float nRot, float freqBase)
        {
            return nDims * MathF.Log(nCtxOrig / (nRot * 2f * MathF.PI)) / (2f * MathF.Log(freqBase));
        }

        /// <summary>
        /// Materializes a tensor in host memory. Only used for the small
        /// tensors the host actually has to look at (norms, gates, sinks, APE
        /// tables, the router, tid2eid); the bulk weights never come here —
        /// they stream from the shard directly into VRAM.
        /// </summary>
        private (IntPtr Ptr, GgufTensorInfo Info) GetRaw(string name, bool required = true) => _gguf.Raw(name, required);

        private CudaWeightDesc GetQW(string name, bool required = true) => _gguf.Weight(name, required);

        private float[] GetF32(string name, bool required = true) => _gguf.Floats(name, required);

        private int[] GetI32(string name) => _gguf.Ints(name);

        // -------------------------------------------------------------------
        // Descriptor construction
        // -------------------------------------------------------------------

        /// <param name="owns">The layers this node describes in full (null: every layer). Another node's layer
        /// keeps its topology only (ratio, sources, Engram table), which is what the boundaries are planned from;
        /// none of its tensors is described, read or mapped.</param>
        /// <param name="engramOnDevice">Overrides TS_DSV41_ENGRAM_DEVICE's automatic choice (a distributed
        /// load decides it the same way on every node).</param>
        private Dsv4CudaEngine.ModelDesc BuildModelDesc(int nCtx, int nUbatch, Func<int, bool> owns = null,
            bool? engramOnDevice = null)
        {
            var tokEmbd = GetQW("token_embd.weight");
            _nVocab = tokEmbd.Ne1;
            int tokType = tokEmbd.GgmlType;
            // ts_dsv4_embed_f32 decodes these row layouts directly: Q8_0, the K-quants (Q2_K..Q6_K), F16, BF16, F32.
            if (tokType != 8 && !(tokType >= 10 && tokType <= 14) && tokType != 1 && tokType != 0 && tokType != 30)
                throw new NotSupportedException($"[dsv4-cuda] token_embd type {tokType} unsupported (Q8_0/Q2_K..Q6_K/F16/BF16/F32).");

            var m = new Dsv4CudaEngine.ModelDesc
            {
                NLayer = _nLayer,
                NEmbd = _nEmbd,
                NHead = _nHead,
                HeadDim = _headDim,
                NRot = _nRot,
                QLoraRank = _qLoraRank,
                OGroups = _oGroups,
                OLoraRank = _oLoraRank,
                NSwa = _nSwa,
                NVocab = _nVocab,
                NExpert = _nExpert,
                NExpertUsed = _nExpertUsed,
                NFfExp = _nFfExp,
                HashLayerCount = _hashLayerCount,
                IdxNHead = _idxNHead,
                IdxHeadSize = _idxHeadSize,
                IdxTopK = _idxTopK,
                HcSinkhornIters = _hcSinkhornIters,
                RmsEps = _rmsEps,
                HcEps = _hcEps,
                ExpertWeightsScale = _expertWeightsScale,
                ExpertWeightsNorm = _expertWeightsNorm,
                NCtx = nCtx,
                NUbatch = nUbatch,
                TokEmbd = tokEmbd,
                Output = GetQW("output.weight"),
                OutputNorm = GetF32("output_norm.weight"),
                // V4.1 collapses the streams for the head with the LAST layer's
                // FFN gates, so it ships no output_hc_* tensors at all.
                HcHeadFn = GetF32("output_hc_fn.weight", required: !_isV41),
                HcHeadScale = GetF32("output_hc_scale.weight", required: !_isV41),
                HcHeadBase = GetF32("output_hc_base.weight", required: !_isV41),
                V41 = _isV41,
                CandidateSource = _isV41 ? _engram.CandidateSourceLayerId : -1,
                CandidateTopk = _isV41 ? (int)_engram.CandidateTopkBlocks : 0,
                CandidateBlock = _isV41 ? (int)_engram.CandidateBlockSize : 0,
                Engram = _isV41 ? this : null,
                EngramOnDevice = _isV41 ? engramOnDevice ?? EngramPlacement() : null,
                RopeRawTable = BuildRopeTable(nCtx, comp: false),
                RopeCompTable = BuildRopeTable(nCtx, comp: true),
                Layers = new Dsv4CudaEngine.LayerDesc[_nLayer],
                Dspark = BuildDsparkDesc(),
                HostMatMul = HostMatMul.Instance,
            };

            for (int il = 0; il < _nLayer; il++)
            {
                if (owns != null && !owns(il))
                {
                    m.Layers[il] = new Dsv4CudaEngine.LayerDesc
                    {
                        Ratio = _compressRatios[il],
                        KvSource = _isV41 ? _v41KvSource[il] : -1,
                        IndexSource = _isV41 ? _v41IndexSource[il] : -1,
                        EngramIndex = EngramIndexOf(il),
                    };
                    continue;
                }
                string p = $"blk.{il}.";
                var L = new Dsv4CudaEngine.LayerDesc
                {
                    Ratio = _compressRatios[il],
                    ClampExp = il < _swigluClampExp.Length ? _swigluClampExp[il] : 0f,
                    ClampShexp = il < _swigluClampShexp.Length ? _swigluClampShexp[il] : 0f,
                    AttnNorm = GetF32(p + "attn_norm.weight"),
                    Sinks = GetF32(p + "attn_sinks.weight"),
                    WqA = GetQW(p + "attn_q_a.weight"),
                    QANorm = GetF32(p + "attn_q_a_norm.weight"),
                    WqB = GetQW(p + "attn_q_b.weight"),
                    Wkv = GetQW(p + "attn_kv.weight"),
                    KvNorm = GetF32(p + "attn_kv_a_norm.weight"),
                    WoA = GetQW(p + "attn_output_a.weight"),
                    WoB = GetQW(p + "attn_output_b.weight"),
                    HcAttnFn = GetF32(p + "hc_attn_fn.weight"),
                    HcAttnScale = GetF32(p + "hc_attn_scale.weight"),
                    HcAttnBase = GetF32(p + "hc_attn_base.weight"),
                    HcFfnFn = GetF32(p + "hc_ffn_fn.weight"),
                    HcFfnScale = GetF32(p + "hc_ffn_scale.weight"),
                    HcFfnBase = GetF32(p + "hc_ffn_base.weight"),
                    GateInp = GetF32(p + "ffn_gate_inp.weight"),
                    FfnNorm = GetF32(p + "ffn_norm.weight"),
                    GateExps = GetQW(p + "ffn_gate_exps.weight"),
                    DownExps = GetQW(p + "ffn_down_exps.weight"),
                    UpExps = GetQW(p + "ffn_up_exps.weight"),
                    GateShexp = GetQW(p + "ffn_gate_shexp.weight"),
                    DownShexp = GetQW(p + "ffn_down_shexp.weight"),
                    UpShexp = GetQW(p + "ffn_up_shexp.weight"),
                };

                if (_isV41)
                {
                    // Only the per-ratio source layers carry compressor and
                    // indexer-query tensors; the rest of their group reads the
                    // caches those layers build.
                    L.KvSource = _v41KvSource[il];
                    L.IndexSource = _v41IndexSource[il];
                    if (L.KvSource == il)
                    {
                        L.CompWkv = GetQW(p + "attn_compressor_kv.weight");
                        L.CompNorm = GetF32(p + "attn_compressor_norm.weight");
                        if (L.Ratio > 1)
                            L.CompWgate = GetQW(p + "attn_compressor_gate.weight");
                        L.IndexerK = GetQW(p + "indexer.attn_k.weight");
                        L.IndexerKNorm = GetF32(p + "indexer.k_norm.weight");
                    }
                    if (L.IndexSource == il)
                    {
                        L.IdxProj = GetQW(p + "indexer.proj.weight");
                        L.IdxQB = GetQW(p + "indexer.attn_q_b.weight");
                    }
                    for (int t = 0; t < _engram.Layers.Length; t++)
                    {
                        if (_engram.Layers[t].Id != il)
                            continue;
                        L.EngramIndex = t;
                        L.EngramTable = GetQW(p + "engram_embd.weight");
                        // llama.cpp's layout: engram.py only ever uses q_weight * k_weight, so
                        // the file stores that product once, as engram_gate, beside engram_kv
                        // (engram_wkv by another name). The gate kernel scales the normalized
                        // query by EngramQ and the normalized key by EngramK, and reads both;
                        // the product goes on the query and the key gets a gain of exactly one,
                        // which is the native graph's arithmetic (rms(x) * gate against a plain
                        // rms(key)) with no second kernel.
                        bool fused = _gguf.Has(p + "engram_gate.weight");
                        string q = p + (fused ? "engram_gate.weight" : "engram_q.weight");
                        L.EngramWkv = GetQW(p + (fused ? "engram_kv.weight" : "engram_wkv.weight"));
                        L.EngramQ = GetF32(q);
                        if (fused)
                        {
                            L.EngramK = new float[HcStreams * _nEmbd];
                            Array.Fill(L.EngramK, 1f);
                        }
                        else
                        {
                            L.EngramK = GetF32(p + "engram_k.weight");
                        }
                        ulong[] qShape = _gguf.InfoOf(q).Shape;
                        int lookup = HashColumns * HeadDim;
                        if (L.EngramWkv.Ne0 != lookup || L.EngramWkv.Ne1 != (HcStreams + 1) * _nEmbd ||
                            qShape.Length < 2 || qShape[0] != (ulong)_nEmbd || qShape[1] != HcStreams ||
                            L.EngramK.Length != HcStreams * _nEmbd)
                            throw new InvalidOperationException(
                                $"[dsv4-cuda] DeepSeek V4.1 Engram weights at layer {il} do not match {lookup} lookup " +
                                $"values, {HcStreams} streams and n_embd {_nEmbd}");
                        _engramTables[t] = KeepsEngramTable(t)
                            ? MapEngramTable(p + "engram_embd.weight")
                            : DescribeRemoteEngramTable(p + "engram_embd.weight");
                        break;
                    }
                }
                else if (L.Ratio != 0)
                {
                    L.CompWkv = GetQW(p + "attn_compressor_kv.weight");
                    L.CompWgate = GetQW(p + "attn_compressor_gate.weight");
                    L.CompApe = GetF32(p + "attn_compressor_ape.weight");
                    L.CompNorm = GetF32(p + "attn_compressor_norm.weight");
                    if (L.Ratio == CsaRatio)
                    {
                        L.IdxProj = GetQW(p + "indexer.proj.weight");
                        L.IdxQB = GetQW(p + "indexer.attn_q_b.weight");
                        L.IdxCompWkv = GetQW(p + "indexer_compressor_kv.weight");
                        L.IdxCompWgate = GetQW(p + "indexer_compressor_gate.weight");
                        L.IdxCompApe = GetF32(p + "indexer_compressor_ape.weight");
                        L.IdxCompNorm = GetF32(p + "indexer_compressor_norm.weight");
                    }
                }

                if (il < _hashLayerCount)
                    L.Tid2Eid = GetI32(p + "ffn_gate_tid2eid.weight");
                else
                    L.ExpProbsBias = GetF32(p + "exp_probs_b.bias");

                m.Layers[il] = L;
            }

            return m;
        }

        /// <summary>
        /// Describes the DSpark drafter (a separate GGUF: three DSV4 blocks with
        /// compress_ratio 0, plus the Markov and confidence heads) for the
        /// engine. Returns null when no drafter was supplied.
        /// </summary>
        private Dsv4CudaEngine.DsparkDesc BuildDsparkDesc()
        {
            if (_gguf.Extra == null)
                return null;

            // Published DSpark drafters carry the same weights under three
            // naming schemes (the ds4 builder's `mtp.*`, and two `dspark.*`
            // variants that differ in the metadata prefix), so resolve both the
            // keys and the tensor names by trying each spelling.
            string arch = _gguf.Extra.GetString("general.architecture") ?? string.Empty;
            if (arch != "deepseek4-dspark" && arch != "deepseek_v4_flash_dspark_draft")
            {
                throw new InvalidOperationException(
                    $"[dsv4-cuda] draft model architecture '{arch}' is not a DeepSeek V4 DSpark drafter " +
                    "(expected deepseek4-dspark or deepseek_v4_flash_dspark_draft). DSpark drafters for other " +
                    "architectures such as Gemma 4 use a different drafter design and are not supported.");
            }

            int DsUint(params string[] keys)
            {
                foreach (string k in keys)
                {
                    uint v = _gguf.Extra.GetUint32(k, 0);
                    if (v != 0)
                        return (int)v;
                }
                return 0;
            }

            int nStages = DsUint("dspark.n_layers", "dspark.stage_count", "dspark.layer_count",
                                 "deepseek4.dspark.n_layers", "deepseek4.dspark.layer_count");
            int blockSize = DsUint("dspark.block_size", "deepseek4.dspark.block_size");
            int markovRank = DsUint("dspark.markov_rank", "deepseek4.dspark.markov_rank");
            int noiseToken = DsUint("dspark.noise_token_id", "deepseek4.dspark.noise_token_id");
            int[] targetLayers = _gguf.Extra.GetInt32Array("dspark.target_layer_ids")
                ?? _gguf.Extra.GetInt32Array("dspark.target_layers")
                ?? _gguf.Extra.GetInt32Array("deepseek4.dspark.target_layer_ids")
                ?? _gguf.Extra.GetInt32Array("deepseek4.dspark.target_layers")
                ?? Array.Empty<int>();
            if (nStages <= 0 || blockSize <= 0 || markovRank <= 0 || targetLayers.Length == 0)
                throw new InvalidOperationException("[dsv4-cuda] draft model is missing dspark.* metadata");

            // The drafter has no per-layer swiglu clamp of its own; the module is
            // trained with the target's single swiglu_limit.
            float clamp = _swigluClampExp.Length > 0 ? _swigluClampExp[_swigluClampExp.Length - 1] : 0f;
            float clampSh = _swigluClampShexp.Length > 0 ? _swigluClampShexp[_swigluClampShexp.Length - 1] : clamp;

            // First spelling that exists in the drafter file wins.
            string Pick(params string[] names)
            {
                foreach (string n in names)
                    if (_gguf.Has(n))
                        return n;
                return names[0];
            }

            var stages = new Dsv4CudaEngine.LayerDesc[nStages];
            for (int s = 0; s < nStages; s++)
            {
                string p = _gguf.Has($"mtp.{s}.attn_norm.weight") ? $"mtp.{s}." : $"dspark.{s}.";
                stages[s] = new Dsv4CudaEngine.LayerDesc
                {
                    Ratio = 0,
                    ClampExp = clamp,
                    ClampShexp = clampSh,
                    AttnNorm = GetF32(p + "attn_norm.weight"),
                    Sinks = GetF32(p + "attn_sinks.weight"),
                    WqA = GetQW(p + "attn_q_a.weight"),
                    QANorm = GetF32(p + "attn_q_a_norm.weight"),
                    WqB = GetQW(p + "attn_q_b.weight"),
                    Wkv = GetQW(p + "attn_kv.weight"),
                    KvNorm = GetF32(p + "attn_kv_a_norm.weight"),
                    WoA = GetQW(p + "attn_output_a.weight"),
                    WoB = GetQW(p + "attn_output_b.weight"),
                    HcAttnFn = GetF32(p + "hc_attn_fn.weight"),
                    HcAttnScale = GetF32(p + "hc_attn_scale.weight"),
                    HcAttnBase = GetF32(p + "hc_attn_base.weight"),
                    HcFfnFn = GetF32(p + "hc_ffn_fn.weight"),
                    HcFfnScale = GetF32(p + "hc_ffn_scale.weight"),
                    HcFfnBase = GetF32(p + "hc_ffn_base.weight"),
                    GateInp = GetF32(p + "ffn_gate_inp.weight"),
                    ExpProbsBias = GetF32(p + "exp_probs_b.bias", required: false),
                    FfnNorm = GetF32(p + "ffn_norm.weight"),
                    GateExps = GetQW(p + "ffn_gate_exps.weight"),
                    DownExps = GetQW(p + "ffn_down_exps.weight"),
                    UpExps = GetQW(p + "ffn_up_exps.weight"),
                    GateShexp = GetQW(p + "ffn_gate_shexp.weight"),
                    DownShexp = GetQW(p + "ffn_down_shexp.weight"),
                    UpShexp = GetQW(p + "ffn_up_shexp.weight"),
                };
            }

            string first = "mtp.0.";
            string lastS = $"mtp.{nStages - 1}.";
            return new Dsv4CudaEngine.DsparkDesc
            {
                BlockSize = blockSize,
                NoiseTokenId = noiseToken,
                MarkovRank = markovRank,
                TargetLayerIds = targetLayers,
                Stages = stages,
                MainProj = GetQW(Pick(first + "main_proj.weight", "dspark.main_proj.weight")),
                MainNorm = GetF32(Pick(first + "main_norm.weight", "dspark.main_norm.weight")),
                Norm = GetF32(Pick(lastS + "norm.weight", "dspark.norm.weight")),
                HcHeadFn = GetF32(Pick(lastS + "hc_head_fn.weight", "dspark.hc_head_fn.weight")),
                HcHeadScale = GetF32(Pick(lastS + "hc_head_scale.weight", "dspark.hc_head_scale.weight")),
                HcHeadBase = GetF32(Pick(lastS + "hc_head_base.weight", "dspark.hc_head_base.weight")),
                MarkovW1 = GetF32(Pick(lastS + "markov_head.markov_w1.weight", "dspark.markov_w1.weight")),
                MarkovW2 = GetQW(Pick(lastS + "markov_head.markov_w2.weight", "dspark.markov_w2.weight")),
                ConfProj = GetF32(Pick(lastS + "confidence_head.proj.weight",
                                       "dspark.conf_proj.weight", "dspark.confidence_head.weight")),
            };
        }

        /// <summary>
        /// Interleaved (cos, sin) per position with the YaRN attention factor
        /// folded in — the table-driven form of DeepSeek4CpuExecutor.RopeCacheInit.
        /// </summary>
        private float[] BuildRopeTable(int nCtx, bool comp)
        {
            int nDims = _nRot;
            float freqBase = comp ? _compressRopeBase : _ropeFreqBase;
            float freqScale = comp ? _yarnFreqScale : 1f;
            float extFactor = comp ? _yarnExtFactor : 0f;
            float attnFactor = extFactor == 0f ? 1f : 1f / (1f + 0.1f * MathF.Log(1f / freqScale));
            float thetaScale = MathF.Pow(freqBase, -2f / nDims);

            var table = new float[(long)nCtx * nDims];
            Parallel.For(0, nCtx, pos =>
            {
                float theta = pos;
                long baseIdx = (long)pos * nDims;
                for (int i0 = 0; i0 < nDims; i0 += 2)
                {
                    float thetaInterp = freqScale * theta;
                    float th = thetaInterp;
                    float mscale = attnFactor;
                    if (extFactor != 0f)
                    {
                        float y = (i0 / 2 - _compCorr0) / MathF.Max(0.001f, _compCorr1 - _compCorr0);
                        float ramp = (1f - MathF.Min(1f, MathF.Max(0f, y))) * extFactor;
                        th = thetaInterp * (1f - ramp) + theta * ramp;
                        mscale *= 1f + 0.1f * MathF.Log(1f / freqScale);
                    }
                    table[baseIdx + i0 + 0] = MathF.Cos(th) * mscale;
                    table[baseIdx + i0 + 1] = MathF.Sin(th) * mscale;
                    theta *= thetaScale;
                }
            });
            return table;
        }

        public void Dispose()
        {
            // The prefetch threads read the mappings: they leave before anything is unmapped.
            _engramReader?.Dispose();
            _engramWarmCancel.Cancel();
            try { _engramWarm?.Wait(); }
            catch (AggregateException) { }
            _engramWarmCancel.Dispose();
            _engine?.Dispose();
            _engine = null;
            _gguf?.Dispose();
            _gguf = null;
        }
    }
}
