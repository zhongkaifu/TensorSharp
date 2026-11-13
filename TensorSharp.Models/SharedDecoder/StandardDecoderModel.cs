// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Linq;
using System.Threading;

using TensorSharp.GGML;
using TensorSharp.MLX;
using TensorSharp.Models.Placement;

namespace TensorSharp.Models
{
    /// <summary>
    /// The shared decoder runtime for pre-norm transformer decoders (attention + SwiGLU FFN
    /// per layer): it owns everything that is not the family's arithmetic, so a family built
    /// on it gets every serving mechanism with no code of its own.
    ///
    /// <list type="bullet">
    /// <item><b>Continuous batching</b>: one paged batched forward (<see cref="ForwardBatch"/>);
    /// the single-sequence forward and KV snapshots are implemented once.</item>
    /// <item><b>Tensor parallelism</b> (<c>--tp T</c>): weights sharded by role, each branch
    /// body run once per rank, the runtime reducing the row-parallel partials.</item>
    /// <item><b>Layer split</b> (<c>--layer-split P</c>, local or multi-node): whole layers per
    /// device, activations moved at stage boundaries.</item>
    /// <item><b>Hybrid</b> (<c>--tp T --layer-split P</c>): P stages, each its own T-device group.</item>
    /// </list>
    ///
    /// A family supplies its configuration, tokenizer and the few things that really differ
    /// between decoders: rotary embedding and Q/K treatment (<see cref="RotateQueryKey"/>),
    /// and optionally media injection. Everything else - including the backend fast paths the
    /// per-family forwards used (fused SwiGLU FFN, MLX fused add+norm and decode attention,
    /// half-operand attention GEMMs on direct CUDA) - lives here, so it is shared too.
    /// See docs/design/unified-execution.md.
    /// </summary>
    public abstract partial class StandardDecoderModel : ModelBase
    {
        // MLX: hand the lazy graph to the GPU every N layers (async), so it runs while later layers
        // are built. Measured on an M5 Pro (2026-11-10), real generation: Qwen3.5-0.8B 191 tok/s at
        // N=4 vs 183 (8) and 168 (16); Qwen3.8-27B IQ3_XXS 19.3-20.0 at 4 vs 18.5-19.5 at 16.
        private static readonly int MlxEvalEveryNLayers = ResolvePositiveEnv("TS_MLX_EVAL_EVERY_N_LAYERS", 4);
        private static readonly int PrefillChunkTokens = ResolvePositiveEnv("TS_PREFILL_CHUNK", 2048);

        private readonly int _requestedTensorParallel;
        private readonly int _requestedStages;

        private DeviceMesh _mesh;
        private Tensor[][] _cacheK;   // [layer][rank]; null rows for layers another node owns
        private Tensor[][] _cacheV;
        private int _cacheCapacity;
        private bool[] _qkvFused;
        private bool[] _gateUpFused;
        private int _keyDim;
        private int _valueDim;
        private int _localHeads;
        private int _localKvHeads;
        private readonly ConcurrentDictionary<(Tensor Weight, int Device), Tensor> _replicas = new();

        /// <param name="tpDegree">T, the tensor-parallel degree of every stage.</param>
        /// <param name="tpGroup">A caller-supplied device group: a distributed TP group, a
        /// multi-node layer-split group, or (tests) a simulated group of T x P devices.</param>
        /// <param name="layerSplitDegree">P, the number of whole-layer stages.</param>
        protected StandardDecoderModel(string ggufPath, BackendType backend, int tpDegree, ITensorParallelGroup tpGroup,
            int layerSplitDegree)
            : base(ggufPath, backend,
                BaseTensorParallel(tpDegree, tpGroup, layerSplitDegree),
                tpGroup,
                tpGroup is ILayerSplitGroup ? 1 : IsHybrid(tpDegree, tpGroup, layerSplitDegree) ? 1 : layerSplitDegree)
        {
            _requestedStages = tpGroup is ILayerSplitGroup ? 1 : Math.Max(1, layerSplitDegree);
            _requestedTensorParallel = tpGroup is ILayerSplitGroup layers
                ? layers.StageTensorParallel
                : IsHybrid(tpDegree, tpGroup, layerSplitDegree)
                    ? tpDegree
                    : tpGroup?.Degree ?? Math.Max(1, tpDegree);
        }

        /// <summary>A node of a multi-node pipeline may run its own number of stages, each a tensor-parallel group of
        /// its own size: the plan follows the nodes' agreed topology and each node runs its layers on its local mesh.</summary>
        protected override bool RunsMixedNodeShapes => true;

        private static bool IsHybrid(int tp, ITensorParallelGroup group, int stages)
            => group is not ILayerSplitGroup && stages > 1 && (tp > 1 || group != null);

        /// <summary>What the ModelBase constructor provisions: the full device group (T x P devices)
        /// for any tensor-parallel mesh, nothing for a pure layer split (ModelBase's own
        /// whole-layer placement creates those devices).</summary>
        private static int BaseTensorParallel(int tp, ITensorParallelGroup group, int stages)
        {
            if (group is ILayerSplitGroup) return 1;
            if (IsHybrid(tp, group, stages))
            {
                if (tp < 2)
                    throw new ArgumentException("Hybrid placement needs a tensor-parallel degree of at least 2 per stage.");
                int devices = checked(tp * stages);
                if (group != null && group.Degree != devices)
                    throw new ArgumentException($"A hybrid placement of {stages} stages x tp {tp} needs a group of {devices} devices, not {group.Degree}.");
                return devices;
            }
            return tp;
        }

        // ---------------------------------------------------------------- family hooks

        /// <summary>
        /// Rotary embedding and every per-token Q/K treatment of the family, for the rank-local
        /// heads in <paramref name="rank"/>: <paramref name="q"/> is [tokens, LocalHeads*KeyDim],
        /// <paramref name="k"/> is [tokens, LocalKvHeads*KeyDim]. Either may be replaced (dispose
        /// the old one). <paramref name="positions"/> gives every token's absolute position.
        /// </summary>
        protected abstract void RotateQueryKey(RankScope rank, int layer, ref Tensor q, ref Tensor k, TokenPositions positions);

        /// <summary>The FFN pre-norm weight of <paramref name="layer"/>. Default
        /// <c>blk.N.ffn_norm.weight</c>; Qwen 3.5 calls it <c>post_attention_norm</c>.</summary>
        protected virtual string FfnNormWeight(int layer) => LayerName(layer, "ffn_norm.weight");

        /// <summary>NextN / MTP draft layers after the decoder stack (<c>blk.{NumLayers + i}</c>): one decoder
        /// layer each, run by the draft head, never by the trunk. 0 when the family has none here.</summary>
        protected virtual int DraftLayerCount => 0;

        /// <summary>A drafter on this load reads the trunk's token embedding (a draft layer without its own, or a DFlash
        /// drafter): a multi-node pipeline's last stage, where the driver drafts, then loads the embedding too. Set by a
        /// family before its weights load.</summary>
        private protected bool DrafterReadsTrunkEmbedding { get; set; }

        /// <summary>Decoder layers plus draft layers: what weights, fusion, routing and caches cover.</summary>
        private int TotalLayers => Config.NumLayers + DraftLayerCount;

        // ---------------------------------------------------------------- extra key caches

        /// <summary>A family's extra per-cell key cache for <paramref name="layer"/>: one head of this many values per
        /// cell, in the K/V cache's type (Qwen 3.8's QSA indexer keys); 0 for none. A slot keeps it after the layers'
        /// caches (entry <see cref="TotalLayers"/> + j of its K array, no V), so whatever moves a slot's rows moves it.</summary>
        protected virtual int ExtraKeyCacheWidth(int layer) => 0;

        private int[] _extraCacheLayer = Array.Empty<int>();   // extra cache j -> its layer

        /// <summary>Entries of a slot's K array: the layers' caches, then the extra key caches.</summary>
        private int CacheEntries => TotalLayers + _extraCacheLayer.Length;

        /// <summary>The stage an entry of a slot's caches lives on.</summary>
        private int CacheStage(int entry) => StageOfLayer(entry < TotalLayers ? entry : _extraCacheLayer[entry - TotalLayers]);

        /// <summary>Entry <paramref name="entry"/>'s cache tensor for one rank at <paramref name="capacity"/> cells (null where
        /// it has none: a recurrent layer, another node's layer, an extra cache's value side).</summary>
        private Tensor NewCacheTensor(int entry, int rank, int capacity, bool value)
        {
            DType dtype = _kvCacheDtype.ToDType();
            Tensor t;
            if (entry >= TotalLayers)
            {
                int layer = _extraCacheLayer[entry - TotalLayers];
                if (value || !OwnsLayer(layer)) return null;
                t = new Tensor(_mesh.Allocator(CacheStage(entry), rank), dtype, 1, capacity, ExtraKeyCacheWidth(layer));
            }
            else
            {
                if (!OwnsLayer(entry) || _recurrentLayer[entry]) return null;
                // A draft layer is not sharded: under tensor parallelism it runs whole on rank 0, with every KV head.
                bool draft = entry >= Config.NumLayers && _mesh.TensorParallel > 1;
                if (draft && rank > 0) return null;
                IAllocator alloc = _mesh.Allocator(CacheStage(entry), rank);
                int kvHeads = draft ? Config.NumKVHeads : _localKvHeads;
                t = value ? new Tensor(alloc, dtype, kvHeads, capacity, _valueDim)
                          : new Tensor(alloc, dtype, kvHeads, capacity, _keyDim);
            }
            InitializeCacheTensor(t);
            return t;
        }

        /// <summary>The extra key cache of <paramref name="layer"/> in <paramref name="k"/> (a slot's K array), or null.</summary>
        private protected Tensor ExtraKeyCache(Tensor[][] k, int layer, int rank = 0)
        {
            int j = Array.IndexOf(_extraCacheLayer, layer);
            return j < 0 || k == null || k.Length <= TotalLayers + j ? null : k[TotalLayers + j]?[rank];
        }

        private void InitExtraCaches()
        {
            var layers = new System.Collections.Generic.List<int>();
            for (int l = 0; l < Config.NumLayers; l++)
                if (ExtraKeyCacheWidth(l) > 0) layers.Add(l);
            _extraCacheLayer = layers.ToArray();
        }

        /// <summary>The stage a layer runs on; draft layers run with the head, on the last stage.</summary>
        private int StageOfLayer(int layer) => layer >= Config.NumLayers ? _mesh.Stages - 1 : _mesh.StageOf(layer);

        /// <summary>
        /// Attention whose query projection also emits a per-head output gate, interleaved per
        /// head as <c>[q_h | gate_h]</c> (Qwen 3.5 / Qwen3-Next): the runtime splits it off and
        /// multiplies the attention output by <c>sigmoid(gate)</c> before the output projection.
        /// </summary>
        protected virtual bool GatedAttentionOutput => false;

        /// <summary>Called once after loading and fusing, before sharding and device residency:
        /// the place to repack or derive weights from their host copies. Default: nothing.</summary>
        protected virtual void PrepareLoadedWeights() { }

        /// <summary>Constraints beyond heads and FFN width that a tensor-parallel degree must meet
        /// (recurrent head counts, ...); each entry names what does not divide. Default: none.</summary>
        protected virtual System.Collections.Generic.IEnumerable<string> ShardingProblems(int degree)
            => Array.Empty<string>();

        /// <summary>Extra replicated weights the per-rank code reads on every rank (Q/K norms):
        /// pre-copied to each rank's device at load. Default: none.</summary>
        protected virtual System.Collections.Generic.IEnumerable<string> ReplicatedLayerWeights(int layer)
            => Array.Empty<string>();

        // ---------------------------------------------------------------- geometry

        protected int KeyDim => _keyDim;
        protected int ValueDim => _valueDim;

        private int _shardDegree = 1;
        private int _shardRankOffset;

        /// <summary>The degree weights are sharded by: the stage's tensor-parallel degree, or a
        /// distributed group's global degree (this process then holds ranks
        /// [<see cref="ShardRankOffset"/>, +TensorParallel) of it).</summary>
        protected int ShardDegree => _shardDegree;

        /// <summary>The global shard index of this process's first local rank.</summary>
        protected int ShardRankOffset => _shardRankOffset;

        /// <summary>The device mesh (stages x tensor-parallel ranks) this model runs on.</summary>
        internal DeviceMesh Mesh => _mesh;

        /// <summary>The K caches as [layer][rank], for tests that check placement.</summary>
        internal Tensor[][] KeyCachesForTests => _cacheK;

        /// <summary>Placement summary, e.g. "2 whole-layer stages x tensor parallel 2 = 4 devices".</summary>
        public string PlacementDescription => _mesh?.Describe() ?? "not initialized";

        /// <summary>One rank of one stage, as the per-rank code sees it.</summary>
        protected sealed class RankScope
        {
            internal RankScope(int stage, int rank, int degree, IAllocator allocator, int localHeads, int localKvHeads)
            {
                Stage = stage;
                Rank = rank;
                Degree = degree;
                Allocator = allocator;
                LocalHeads = localHeads;
                LocalKvHeads = localKvHeads;
            }

            public int Stage { get; }
            public int Rank { get; }
            /// <summary>Tensor-parallel degree of the stage (1 = the whole layer on one device).</summary>
            public int Degree { get; }
            public IAllocator Allocator { get; }
            public int LocalHeads { get; }
            public int LocalKvHeads { get; }
        }

        /// <summary>Absolute positions of the tokens of one forward.</summary>
        protected readonly struct TokenPositions
        {
            private readonly int[] _explicit;

            private TokenPositions(int start, int count, int[] explicitPositions, int ropeDelta = 0, int[] table = null)
            {
                Start = start;
                Count = count;
                _explicit = explicitPositions;
                RopeDelta = ropeDelta;
                Table = table;
            }

            public static TokenPositions Contiguous(int start, int count) => new(start, count, null);
            /// <summary>Per-row rotary positions (a batched step: each sequence's KV index plus its delta).</summary>
            public static TokenPositions Explicit(int[] positions) => new(0, positions.Length, positions);

            /// <summary>The same rows rotated at KV index + <paramref name="delta"/>, or by the (T,H,W)
            /// rows of <paramref name="table"/> (multi-axis rotary).</summary>
            public TokenPositions WithRotary(int delta, int[] table) => new(Start, Count, _explicit, delta, table);

            /// <summary>KV index of the first row of a contiguous run (one sequence).</summary>
            public int Start { get; }
            public int Count { get; }
            public bool IsContiguous => _explicit == null;
            /// <summary>Rotary position minus KV index for a contiguous run (0 without a position table).</summary>
            public int RopeDelta { get; }
            /// <summary>Flat (T,H,W) rotary positions of these rows, or null (every axis at this[t]).</summary>
            public int[] Table { get; }
            /// <summary>The rotary position of a row.</summary>
            public int this[int token] => _explicit?[token] ?? Start + token + RopeDelta;

            /// <summary>Positions repeated for each of <paramref name="heads"/> heads, token-major,
            /// as Ops.RoPEEx reads them.</summary>
            public int[] PerHead(int heads)
            {
                var result = new int[Count * heads];
                for (int t = 0; t < Count; t++)
                {
                    int p = this[t];
                    for (int h = 0; h < heads; h++)
                        result[t * heads + h] = p;
                }
                return result;
            }
        }

        private static int ResolvePositiveEnv(string name, int fallback)
        {
            string raw = Environment.GetEnvironmentVariable(name);
            return !string.IsNullOrWhiteSpace(raw) && int.TryParse(raw, out int v) && v > 0 ? v : fallback;
        }

        // ---------------------------------------------------------------- initialization

        /// <summary>
        /// Build the device mesh, load, fuse, shard and place the weights and allocate the KV
        /// caches. Call once from the family constructor, after <see cref="ModelBase.Config"/>
        /// (heads, KV heads, layers) and the tokenizer are set.
        /// </summary>
        /// <param name="executionIdentity">Everything that must agree across the nodes of a
        /// multi-node layer split (hyperparameters, KV dtype); hashed by ModelBase.</param>
        protected void InitializeDecoder(int keyDim, int valueDim, string executionIdentity)
        {
            if (keyDim != valueDim)
                throw new NotSupportedException($"{Config.Architecture}: the shared decoder runtime needs equal key/value head dims, got {keyDim}/{valueDim}.");
            _keyDim = keyDim;
            _valueDim = valueDim;
            // The runtime's per-op paths interleave host writes (positions, decode rotary and
            // norms, paged-attention results, stage transfers, host experts) with device
            // kernels. Metal's lazy-sync drains device work before host READS but has no
            // host-WRITE barrier, so a later kernel can run against a stale device copy:
            // measured on an M5 Pro, the per-family Mistral 3 forward this replaced was
            // non-deterministic on ggml_metal (two identical runs 6.5e-2 apart, 8.9e-2 from the
            // managed reference) and exact with it off (9e-4 from the reference, like
            // hunyuan-dense). Gemma 4 and DiffusionGemma make the same call for the same
            // reason. Process-wide; the next model's GgmlContext re-enables it.
            if (_backend == BackendType.GgmlMetal && GgmlBasicOps.GetAsyncCompute())
                GgmlBasicOps.SetAsyncCompute(false);
            try
            {
                DetectRecurrentLayers();
                bool multiDevice = _requestedStages > 1 || _requestedTensorParallel > 1 || LayerSplitGroup != null;
                // The model-aligned KV default must not depend on placement: a multi-device
                // mesh resolves it from the whole checkpoint's metadata (a stage loads only
                // its own layers), a single device from its loaded weights, as before.
                if (multiDevice)
                    ApplyModelAlignedKvCacheDefaultFromMetadata();
                BuildMesh(executionIdentity);
                LoadWeights();
                FuseProjections();
                DetectMoeLayers();
                // Families reshape loaded weights here, while every host copy still exists:
                // residency below may release them (direct and GGML CUDA preloads).
                PrepareLoadedWeights();
                if (!multiDevice)
                    ApplyModelAlignedKvCacheDefault(_quantWeights);
                if (_mesh.TensorParallel > 1)
                    ShardForMesh();
                PrepareResidency();

                int maxContext = ResolveConfiguredContextLength();
                int initial = ResolveInitialCacheAllocationLength(maxContext);
                if (initial < maxContext)
                    Console.WriteLine($"Initial {_backend} KV cache allocation: {initial} tokens (grows on demand up to {maxContext}).");
                InitCaches(initial, maxContext);
                InitRecurrentStates();
                PlanExpertPlacement();
                StartHostWarm();
                PreReplicateNormWeights();
                Console.WriteLine($"  Placement: {_mesh.Describe()} (shared decoder runtime).");
            }
            catch
            {
                Dispose();
                throw;
            }
        }

        /// <summary>The tokens of the step being embedded (every sequence's rows, in row order), for a family whose
        /// layers read more than the embedding (Flash Next's per-layer n-gram rows).</summary>
        protected int[] StepTokens { get; private set; }

        private Tensor EmbedStep(int[] tokens)
        {
            StepTokens = tokens;
            return Embedding(tokens);
        }

        // A layer without ffn_gate_up runs ffn_gate and ffn_up as two products on every path (graph
        // SwigluSplit, per-op, tensor-parallel shards), so a mixed-quant pair stays in its own types
        // where fusing it would requantize: IQ2_XXS/IQ2_XS/IQ1_S pairs (no importance matrix), and on
        // MLX every mixed pair (a requantized side reads more bits a weight and rounds twice).
        protected override bool SupportsSplitGateUpFfn => true;
        protected override bool SplitsMixedGateUpOnMlx => true;

        private void FuseProjections()
        {
            int fusedQkv = 0;
            for (int l = 0; l < TotalLayers; l++)
            {
                if (OwnsLayer(l) && TryFuseQkvWeights(l))
                    fusedQkv++;
            }
            if (fusedQkv > 0)
                Console.WriteLine($"  Fused projections: {fusedQkv} QKV");
            FuseGateUpWeights();

            _qkvFused = new bool[TotalLayers];
            _gateUpFused = new bool[TotalLayers];
            for (int l = 0; l < TotalLayers; l++)
            {
                _qkvFused[l] = HasWeight($"blk.{l}.attn_qkv.weight");
                _gateUpFused[l] = HasWeight($"blk.{l}.ffn_gate_up.weight");
            }
        }

        protected bool HasWeight(string name) =>
            _quantWeights.ContainsKey(name) || _weights.ContainsKey(name) ||
            _tpQuantWeights.ContainsKey(name) || _tpWeights.ContainsKey(name);

        protected bool OwnsLayer(int layer) => LayerPlacement == null || LayerPlacement.Owns(PlacementLayer(layer));

        // ---------------------------------------------------------------- KV caches

        private void InitCaches(int initial, int maxContext)
        {
            _maxContextLength = maxContext;
            _cacheCapacity = initial;
            InitExtraCaches();
            (_cacheK, _cacheV) = AllocateCaches(initial);
            _cacheSeqLen = 0;
        }

        private void EnsureCacheCapacity(int required)
        {
            if (required <= _cacheCapacity)
                return;
            if (required > _maxContextLength)
                throw new InvalidOperationException($"Requested sequence length {required} exceeds configured max context {_maxContextLength}.");
            SyncActiveToHost();
            ClearGraphCache();
            int capacity = Math.Max(_cacheCapacity, 1);
            while (capacity < required)
                capacity = Math.Min(_maxContextLength, capacity * 2);
            for (int l = 0; l < _cacheK.Length; l++)
            {
                if (_cacheK[l] == null) continue;
                for (int r = 0; r < _mesh.TensorParallel; r++)
                {
                    if (_cacheK[l][r] == null) continue;   // a rank that does not run this entry (a draft layer off rank 0)
                    _cacheK[l][r] = Grow(_cacheK[l][r], NewCacheTensor(l, r, capacity, value: false));
                    if (_cacheV[l]?[r] != null) _cacheV[l][r] = Grow(_cacheV[l][r], NewCacheTensor(l, r, capacity, value: true));
                }
            }
            _cacheCapacity = capacity;
            Console.WriteLine($"Expanded {Config.Architecture} attention cache to {capacity} tokens.");
        }

        private Tensor Grow(Tensor old, Tensor grown)
        {
            if (_cacheSeqLen > 0)
            {
                using Tensor src = old.Narrow(1, 0, _cacheSeqLen);
                using Tensor dst = grown.Narrow(1, 0, _cacheSeqLen);
                Ops.Copy(dst, src);
            }
            InvalidateTensorDeviceCache(old);
            old.Dispose();
            return grown;
        }

        private System.Collections.Generic.IEnumerable<Tensor> AllCacheTensors(Tensor[][] caches)
        {
            if (caches == null) yield break;
            foreach (Tensor[] layer in caches)
                if (layer != null)
                    foreach (Tensor t in layer)
                        if (t != null) yield return t;
        }

        public override void PrepareForPrefill(int requiredContextTokens) => EnsureCacheCapacity(requiredContextTokens);

        protected override void ResetKVCacheCore()
        {
            _verifyRows = 0;   // the state is reset; nothing to settle
            _draftStart = 0;
            _ropeDelta = 0;
            _pendingPositionTable = null;
            _cacheSeqLen = 0;
            _linearTicks = _attnTicks = _normTicks = _embTicks = _lmHeadTicks = _logitsCopyTicks = 0;
            _forwardCount = 0;
            _forwardSw.Reset();
            foreach (Tensor t in AllCacheTensors(_cacheK)) ResetCacheTensor(t);
            foreach (Tensor t in AllCacheTensors(_cacheV)) ResetCacheTensor(t);
            ResetRecurrentPrimary();
            // Rows past the length are dead and the states were rewritten on the host.
            _cacheDeviceDirty = false;
            ClearGraphCache();
        }

        protected override void TruncateKVCacheCore(int tokenCount)
        {
            SyncActiveToHost();
            ClearGraphCache();
            if (_anyRecurrent && tokenCount != _cacheSeqLen)
            {
                if (tokenCount != 0)
                    throw new NotSupportedException(
                        $"{Config.Architecture} has recurrent layers: its cache truncates only to its length or to zero, not {tokenCount}.");
                ResetRecurrentPrimary();
            }
            base.TruncateKVCacheCore(tokenCount);
            foreach (Tensor t in AllCacheTensors(_cacheK)) InvalidateTensorDeviceCache(t);
            foreach (Tensor t in AllCacheTensors(_cacheV)) InvalidateTensorDeviceCache(t);
        }

        // ---- K/V snapshots: every layer's, every rank's rows, in (layer, rank) order. A plain
        // linear GQA cache (row == absolute position, no window, no recurrent state), so a
        // snapshot restores exactly what a fresh prefill writes. The fingerprint carries the
        // mesh shape: a TP=2 snapshot holds half-head rows and must never reach a TP=1 cache.

        public override bool SupportsKVStateSnapshot => LayerSplitGroup == null && _cacheK != null && !_anyRecurrent;

        public override string KVStateFingerprint =>
            $"{Config.Architecture}|decoder|L={Config.NumLayers}|H={Config.NumHeads}|KV={Config.NumKVHeads}" +
            $"|k={_keyDim}|v={_valueDim}|dtype={_kvCacheDtype.ToShortString()}|mesh={_mesh?.Stages}x{_mesh?.TensorParallel}" +
            FingerprintExtras;

        /// <summary>Family terms the cache identity also depends on (rotary layout, ...).</summary>
        protected virtual string FingerprintExtras => "";

        /// <summary>The decoder layers' caches in transfer order (draft layers' rows are not pages).</summary>
        private Tensor[] FlatCaches(Tensor[][] caches)
            => AllCacheTensors(caches?.Take(Config.NumLayers).ToArray()).ToArray();

        public override long ComputeKVBlockByteSize(int tokenCount)
            => KvBlockTransfer.ComputeBlockByteSize(FlatCaches(_cacheK), FlatCaches(_cacheV), tokenCount);

        public override bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        {
            if (!SupportsKVStateSnapshot) return false;
            SyncActiveToHost();
            return KvBlockTransfer.Extract(
                _allocator, FlatCaches(_cacheK), FlatCaches(_cacheV), _cacheSeqLen, startToken, tokenCount, destination);
        }

        public override bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source)
        {
            if (!SupportsKVStateSnapshot)
                return false;
            SyncActiveToHost();
            ClearGraphCache();
            EnsureCacheCapacity(destToken + tokenCount);
            if (!KvBlockTransfer.Inject(_allocator, FlatCaches(_cacheK), FlatCaches(_cacheV), _cacheSeqLen,
                    destToken, tokenCount, source))
                return false;
            _cacheSeqLen = destToken + tokenCount;
            foreach (Tensor t in AllCacheTensors(_cacheK)) InvalidateTensorDeviceCache(t);
            foreach (Tensor t in AllCacheTensors(_cacheV)) InvalidateTensorDeviceCache(t);
            return true;
        }

        // ---------------------------------------------------------------- forward

        protected override float[] ForwardRefillCore(int[] tokens)
        {
            // Long prompts run in chunks so the per-layer score tensor stays bounded
            // (~heads x chunk x context). Media carry positions within one forward, and a
            // multi-node pipeline follows the serialized protocol, so neither is chunked.
            if (tokens == null || tokens.Length <= PrefillChunkTokens || HasPendingMedia || LayerSplitGroup?.NodeCount > 1)
                return ForwardCore(tokens);
            int last = tokens.Length - 1;
            for (int pos = 0; pos < last; pos += PrefillChunkTokens)
            {
                int len = Math.Min(PrefillChunkTokens, last - pos);
                RunSequence(tokens.AsSpan(pos, len).ToArray(), wantLogits: false);
            }
            return ForwardCore(new[] { tokens[last] });
        }

        protected override float[] ForwardCore(int[] tokens)
        {
            ArgumentNullException.ThrowIfNull(tokens);
            if (tokens.Length == 0) throw new ArgumentException("A forward needs at least one token.", nameof(tokens));
            if (LayerSplitGroup?.NodeCount > 1)
                return ForwardDistributedPipeline(tokens);
            _logitsBuffer = RunSequence(tokens, wantLogits: true);
            _forwardCount++;
            return _logitsBuffer;
        }

        private float[] RunSequence(int[] tokens, bool wantLogits)
        {
            SettlePendingVerify();
            int seqLen = tokens.Length;
            int start = _cacheSeqLen;
            EnsureCacheCapacity(checked(start + seqLen));
            _forwardSw.Start();
            try
            {
                var positions = BeginRotaryPositions(start, seqLen);
                EnterStage(0);
                long t0 = Stopwatch.GetTimestamp();
                Tensor hidden = EmbedStep(tokens);
                _embTicks += Stopwatch.GetTimestamp() - t0;
                InjectMedia(hidden, start);

                if (TryRunGraph(hidden, new[] { SnapshotActive() }, new[] { 0 }, new[] { seqLen }, new[] { start },
                        wantLogits, out float[][] graphLogits, positions.Table))
                {
                    hidden.Dispose();
                    _cacheDeviceDirty = true;
                    _cacheSeqLen += seqLen;
                    return graphLogits?[0];
                }
                SyncActiveToHost();
                ClearGraphCache();

                hidden = RunLayers(hidden, positions, batch: null);
                float[] logits = null;
                if (wantLogits)
                    logits = ProjectRows(hidden, new[] { seqLen - 1 });
                hidden.Dispose();
                _cacheSeqLen += seqLen;
                return logits;
            }
            finally
            {
                EnterStage(0);
                _forwardSw.Stop();
            }
        }

        /// <summary>A multi-node layer split: the serialized stage protocol (ModelBase).</summary>
        private float[] ForwardDistributedPipeline(int[] tokens)
        {
            SettlePendingVerify();   // every node: a verify the driver did not commit keeps its whole window
            int seqLen = tokens.Length;
            int start = _cacheSeqLen;
            EnsureCacheCapacity(checked(start + seqLen));
            var positions = BeginRotaryPositions(start, seqLen);
            _forwardSw.Start();
            try
            {
                Tensor Embed()
                {
                    Tensor hidden = EmbedStep(tokens);
                    InjectMedia(hidden, start);
                    return hidden;
                }
                float[] Project(Tensor hidden) => ProjectRows(hidden, new[] { seqLen - 1 });
                // Single-device stages run layer by layer; tensor-parallel stages run on this node's mesh.
                _logitsBuffer = _mesh.TensorParallel == 1
                    ? ForwardLayerPipeline(tokens, Embed, runLayer: (hidden, layer) => LayerOnOneDevice(hidden, layer, positions, null), Project)
                    : ForwardLayerPipeline(tokens, Embed, runLocalLayers: hidden => RunLayers(hidden, positions, batch: null), Project);
                _cacheSeqLen += seqLen;
                _forwardCount++;
                return _logitsBuffer;
            }
            finally { _forwardSw.Stop(); }
        }

        /// <summary>Run every local layer, moving the activation between stages.</summary>
        private Tensor RunLayers(Tensor hidden, TokenPositions positions, SlotBatch batch, Action<int, Tensor> beforeLayer = null)
        {
            int stage = 0;
            Tensor[] ranks = null;
            if (_mesh.TensorParallel > 1)
                ranks = Replicate(hidden, 0);
            // A node of a multi-node pipeline runs its own layers, starting on its first local stage.
            int first = LayerPlacement?.FirstLayer ?? 0, end = LayerPlacement?.EndLayer ?? Config.NumLayers;
            for (int layer = first; layer < end; layer++)
            {
                int s = _mesh.StageOf(layer);
                if (s != stage)
                {
                    if (ranks != null)
                    {
                        hidden = ranks[0];
                        for (int r = 1; r < ranks.Length; r++) ranks[r].Dispose();
                    }
                    EnterStage(s);
                    hidden = MoveTo(hidden, _mesh.Allocator(s, 0));
                    stage = s;
                    if (ranks != null)
                        ranks = Replicate(hidden, s);
                }

                if (ranks == null)
                {
                    beforeLayer?.Invoke(layer, hidden);
                    hidden = LayerOnOneDevice(hidden, layer, positions, batch);
                    if (_backend == BackendType.Mlx && (layer + 1) % MlxEvalEveryNLayers == 0 && layer + 1 != Config.NumLayers)
                        MlxFusedOps.TryAsyncEvaluate(hidden);
                }
                else
                {
                    beforeLayer?.Invoke(layer, ranks[0]);   // every rank holds the whole residual stream
                    LayerAcrossRanks(ranks, layer, s, positions, batch);
                }
            }
            if (ranks != null)
            {
                hidden = ranks[0];
                for (int r = 1; r < ranks.Length; r++) ranks[r].Dispose();
            }
            // The head runs on the last stage's first device.
            int last = _mesh.Stages - 1;
            if (stage != last)
            {
                EnterStage(last);
                hidden = MoveTo(hidden, _mesh.Allocator(last, 0));
            }
            return hidden;
        }

        /// <summary>Final norm and LM head for <paramref name="rows"/> of <paramref name="hidden"/>.</summary>
        private float[] ProjectRows(Tensor hidden, int[] rows)
        {
            Tensor normed = RMSNormOp(hidden, "output_norm.weight");
            Tensor selected;
            if (rows.Length == hidden.Sizes[0])
            {
                selected = normed;
            }
            else if (rows.Length == 1)
            {
                using Tensor row = normed.Narrow(0, rows[0], 1);
                selected = Ops.NewContiguous(row);
                normed.Dispose();
            }
            else
            {
                int dim = Config.HiddenSize;
                float[] all = normed.GetElementsAsFloat((int)normed.ElementCount());
                normed.Dispose();
                var packed = new float[rows.Length * dim];
                for (int i = 0; i < rows.Length; i++)
                    Array.Copy(all, (long)rows[i] * dim, packed, (long)i * dim, dim);
                selected = CreateFloatTensor(packed, rows.Length, dim);
            }

            long t0 = Stopwatch.GetTimestamp();
            Tensor logits = LinearForward(selected, "output.weight") ?? LinearForward(selected, "token_embd.weight");
            _lmHeadTicks += Stopwatch.GetTimestamp() - t0;
            selected.Dispose();
            long t1 = Stopwatch.GetTimestamp();
            float[] result = TensorToFloatArray(logits);
            _logitsCopyTicks += Stopwatch.GetTimestamp() - t1;
            logits.Dispose();
            return result;
        }

        // ---------------------------------------------------------------- one layer

        /// <summary>A layer whose stage is a single device: the backend fast paths apply.</summary>
        private Tensor LayerOnOneDevice(Tensor hidden, int layer, TokenPositions positions, SlotBatch batch)
        {
            string p = $"blk.{layer}.";
            // A draft layer is not sharded: under tensor parallelism it runs whole on the last stage's rank 0.
            RankScope scope = layer >= Config.NumLayers && _mesh.TensorParallel > 1
                ? new RankScope(StageOfLayer(layer), 0, 1, _mesh.Allocator(StageOfLayer(layer), 0), Config.NumHeads, Config.NumKVHeads)
                : Scope(StageOfLayer(layer), 0);

            Tensor normed = _recurrentLayer[layer] && RecurrentMixerTakesRawInput
                ? hidden.CopyRef()
                : RMSNormOp(hidden, p + "attn_norm.weight");
            Tensor attn = Mixer(scope, normed, layer, positions, batch);
            normed.Dispose();

            // MLX: (hidden += attn; normed2 = rmsnorm(hidden)) in one dispatch.
            Tensor normed2 = null;
            string ffnNormName = FfnNormWeight(layer);
            if (_backend == BackendType.Mlx && _weights.TryGetValue(ffnNormName, out var ffnNorm))
            {
                normed2 = new Tensor(_allocator, DType.Float32, hidden.Sizes[0], hidden.Sizes[1]);
                if (!MlxFusedOps.TryAddRmsNorm(hidden, attn, ffnNorm, Config.Eps, normed2))
                {
                    normed2.Dispose();
                    normed2 = null;
                }
            }
            if (normed2 == null)
            {
                Ops.Add(hidden, hidden, attn);
                attn.Dispose();
                // GGML: norm + gate/up + SiLU*mul + down + residual in one graph.
                if (_gateUpFused[layer] &&
                    TryFusedDenseSwiGLUFFNInto(hidden, ffnNormName, p + "ffn_gate_up.weight", p + "ffn_down.weight"))
                    return hidden;
                normed2 = RMSNormOp(hidden, ffnNormName);
            }
            else
            {
                attn.Dispose();
            }

            Tensor ffn = Ffn(scope, normed2, layer);
            normed2.Dispose();
            Ops.Add(hidden, hidden, ffn);
            ffn.Dispose();
            return hidden;
        }

        /// <summary>A layer whose stage is a tensor-parallel group: each branch body runs once per
        /// rank on that rank's shards and replicated activation, and the partial outputs are
        /// summed before the residual add.</summary>
        private void LayerAcrossRanks(Tensor[] hidden, int layer, int stage, TokenPositions positions, SlotBatch batch)
        {
            string p = $"blk.{layer}.";
            int degree = _mesh.TensorParallel;
            var partial = new Tensor[degree];

            Tensor attnNorm = _weights[p + "attn_norm.weight"];
            _mesh.RunPerRank(stage, r =>
            {
                RankScope scope = Scope(stage, r);
                using Tensor normed = _recurrentLayer[layer] && RecurrentMixerTakesRawInput
                    ? hidden[r].CopyRef()
                    : Ops.RMSNorm(null, hidden[r], Replica(attnNorm, stage, r), null, Config.Eps);
                partial[r] = Mixer(scope, normed, layer, positions, batch);
            });
            _mesh.AllReduce(stage, partial);
            _mesh.RunPerRank(stage, r =>
            {
                Ops.Add(hidden[r], hidden[r], partial[r]);
                partial[r].Dispose();
                partial[r] = null;
            });

            Tensor ffnNorm = _weights[FfnNormWeight(layer)];
            if (_moeLayer[layer])
            {
                // Routing is identical on every rank (replicated input and router): decide it
                // once, then each rank runs its own experts.
                var normed = new Tensor[degree];
                _mesh.RunPerRank(stage, r => normed[r] = Ops.RMSNorm(null, hidden[r], Replica(ffnNorm, stage, r), null, Config.Eps));
                Routing routing = Route(Scope(stage, 0), normed[0], layer);
                _mesh.RunPerRank(stage, r =>
                {
                    partial[r] = MoeFfn(Scope(stage, r), normed[r], layer, routing);
                    normed[r].Dispose();
                });
            }
            else
            {
                _mesh.RunPerRank(stage, r =>
                {
                    RankScope scope = Scope(stage, r);
                    using Tensor normed = Ops.RMSNorm(null, hidden[r], Replica(ffnNorm, stage, r), null, Config.Eps);
                    partial[r] = Ffn(scope, normed, layer);
                });
            }
            _mesh.AllReduce(stage, partial);
            _mesh.RunPerRank(stage, r =>
            {
                Ops.Add(hidden[r], hidden[r], partial[r]);
                partial[r].Dispose();
            });
        }

        /// <summary>
        /// Rank <paramref name="rank"/> of <paramref name="stage"/>. Its allocator is that device's,
        /// not the current stage's: state built outside the stage (recurrent states made for every
        /// layer at once) must land on the device whose layers use it.
        /// </summary>
        private RankScope Scope(int stage, int rank)
            => new(stage, rank, _mesh.TensorParallel,
                _mesh.IsSingleDevice ? _allocator : _mesh.Allocator(stage, rank),
                _localHeads, _localKvHeads);

        // ---------------------------------------------------------------- branch bodies

        /// <summary>
        /// Attention branch body for one rank: Q/K/V projections, the family's rotary step, the
        /// cache write and attention over the rank's heads, and the output projection. Under
        /// tensor parallelism the result is that rank's partial sum (a row-parallel projection).
        /// </summary>
        private Tensor Attention(RankScope scope, Tensor input, int layer, TokenPositions positions, SlotBatch batch)
        {
            string p = $"blk.{layer}.";
            int heads = scope.LocalHeads, kvHeads = scope.LocalKvHeads;
            int qDim = heads * _keyDim, kDim = kvHeads * _keyDim, vDim = kvHeads * _valueDim;
            int qWidth = GatedAttentionOutput ? 2 * qDim : qDim;
            int tokens = positions.Count;
            // A floored step (a draft layer that resumed after a gap) masks its keys as a chunk does.
            bool decode = batch == null && tokens == 1 && _attentionFloor == 0;

            Tensor q, k, v;
            if (_qkvFused[layer])
            {
                Tensor qkv = Linear(scope, input, p + "attn_qkv.weight");
                if (decode && !GatedAttentionOutput)
                {
                    q = qkv.Narrow(1, 0, qWidth);
                    k = qkv.Narrow(1, qWidth, kDim);
                    v = qkv.Narrow(1, qWidth + kDim, vDim);
                }
                else
                {
                    using (Tensor view = qkv.Narrow(1, 0, qWidth)) q = Ops.NewContiguous(view);
                    using (Tensor view = qkv.Narrow(1, qWidth, kDim)) k = Ops.NewContiguous(view);
                    using (Tensor view = qkv.Narrow(1, qWidth + kDim, vDim)) v = Ops.NewContiguous(view);
                }
                qkv.Dispose();
            }
            else
            {
                q = Linear(scope, input, p + "attn_q.weight");
                k = Linear(scope, input, p + "attn_k.weight");
                v = Linear(scope, input, p + "attn_v.weight");
            }

            Tensor gate = GatedAttentionOutput ? SplitGatedQuery(ref q, tokens, heads) : null;
            RotateQueryKey(scope, layer, ref q, ref k, positions);

            long t0 = Stopwatch.GetTimestamp();
            Tensor cacheK = _cacheK[layer][scope.Rank], cacheV = _cacheV[layer][scope.Rank];
            Tensor attn = batch != null
                ? SlotAttention(scope, layer, q, k, v, batch)
                : decode
                    ? DecodeAttention(scope, cacheK, cacheV, _cacheCapacity, q, k, v, positions.Start)
                    : AttendsWindowOnHost(cacheK, tokens)
                        ? WindowAttention(scope, cacheK, cacheV, q, k, v, positions.Start, tokens)
                        : PrefillAttention(scope, cacheK, cacheV, q, k, v, positions.Start, tokens);
            _attnTicks += Stopwatch.GetTimestamp() - t0;
            if (gate != null)
            {
                Ops.SigmoidMul(attn, attn, gate);
                gate.Dispose();
            }

            Tensor output = Linear(scope, attn, p + "attn_output.weight");
            attn.Dispose();
            return output;
        }

        /// <summary>Split a gated query [tokens, heads * 2 * keyDim], laid out [q_h | gate_h] per
        /// head, into the query (replacing <paramref name="q"/>) and the gate.</summary>
        private Tensor SplitGatedQuery(ref Tensor q, int tokens, int heads)
        {
            using Tensor grouped = q.View(tokens, heads, 2, _keyDim);
            using Tensor qView = grouped.Select(2, 0);
            using Tensor gateView = grouped.Select(2, 1);
            Tensor query = Ops.NewContiguous(qView);
            Tensor gate = Ops.NewContiguous(gateView);
            q.Dispose();
            q = query.View(tokens, heads * _keyDim);
            query.Dispose();
            Tensor flatGate = gate.View(tokens, heads * _keyDim);
            gate.Dispose();
            return flatGate;
        }

        /// <summary>MLX's fused SDPA over the cache rows [0, <paramref name="kvLen"/>) for the last
        /// <paramref name="rows"/> query rows (causal), reading the cache in place.</summary>
        private bool MlxCachedAttention(RankScope scope, Tensor result, Tensor q, Tensor cacheK, Tensor cacheV, int rows, int kvLen, float scale)
            => _backend == BackendType.Mlx && scope.Degree == 1 && _keyDim == _valueDim && q.IsContiguous()
               && MlxFusedOps.TryCachedAttention(result, q, cacheK, cacheV, scope.LocalHeads, scope.LocalKvHeads, _keyDim, rows, kvLen, scale,
                   kvFloor: _attentionFloor);

        /// <summary>Cache rows below this are not attended by the per-op path (a draft layer that resumed
        /// after a gap); 0 for the trunk.</summary>
        private int _attentionFloor;

        private Tensor DecodeAttention(RankScope scope, Tensor cacheK, Tensor cacheV, int capacity, Tensor q, Tensor k, Tensor v, int start)
        {
            int heads = scope.LocalHeads, kvHeads = scope.LocalKvHeads;
            float scale = 1.0f / MathF.Sqrt(_keyDim);
            CopyToCacheDecode(cacheK, k, cacheV, v, kvHeads, _keyDim, start);
            k.Dispose();
            v.Dispose();

            var result = new Tensor(scope.Allocator, DType.Float32, 1, heads * _valueDim);
            // MLX keeps K/V on the device and runs its fused SDPA over a strided view of the
            // cache's live rows (mlx-lm's update_and_fetch + scaled_dot_product_attention); the
            // host path would copy the whole cache back every token.
            bool done = MlxCachedAttention(scope, result, q, cacheK, cacheV, 1, start + 1, scale)
                || (_backend == BackendType.Mlx && scope.Degree == 1 && MlxFusedOps.TryDecodeAttention(
                    result, q, cacheK, cacheV, heads, kvHeads, _keyDim, 0, start + 1, capacity, false, scale));
            // Direct CUDA: GQA decode attention on the device over the head-first cache in place.
            // The host path below copies the whole cache to the host every layer.
            if (!done && _backend == BackendType.Cuda)
                done = TensorSharp.Cuda.CudaFusedOps.TryGqaDecodeAttention(
                    result, q, cacheK, cacheV, heads, kvHeads, _keyDim, 0, start + 1, (int)cacheK.Sizes[1], false, scale);
            if (!done)
                AttentionDecodePureCS(q, cacheK, cacheV, result, heads, kvHeads, _keyDim, start + 1, scale);
            q.Dispose();
            return result;
        }

        /// <summary>Rows a verify window (or a draft layer's floored step) attends with in place on the host.</summary>
        private const int HostWindowAttentionRows = 16;

        /// <summary>
        /// A few rows on a backend whose decode attention runs on the host (GGML, the managed CPU): each row attends over
        /// the cache in place, as a decode row does (<see cref="WindowAttention"/>), instead of the prompt path's copy of
        /// the whole cache expanded per query head. Qwen3.8-27B's 4-row verify windows at 3.6K keys took 1.1 s that way on
        /// a 30-layer stage of a multi-node pipeline that decoded a row in 85 ms, and its draft layer copied 3.6K rows to
        /// attend to ten.
        /// </summary>
        private bool AttendsWindowOnHost(Tensor cacheK, int tokens)
            => tokens <= HostWindowAttentionRows && _backend != BackendType.Mlx && _backend != BackendType.Cuda
               && _keyDim == _valueDim && (cacheK.ElementType == DType.Float32 || cacheK.ElementType == DType.Float16);

        /// <summary>The rows' K/V into the cache, then each row's attention over keys [floor, start + row + 1) in place,
        /// by the decode kernels (the floor offsets the cache's base; rows keep the cache's stride).</summary>
        private unsafe Tensor WindowAttention(RankScope scope, Tensor cacheK, Tensor cacheV, Tensor q, Tensor k, Tensor v, int start, int tokens)
        {
            int heads = scope.LocalHeads, kvHeads = scope.LocalKvHeads;
            float scale = 1.0f / MathF.Sqrt(_keyDim);
            using (Tensor kHeads = ReshapeToHeads(k, kvHeads, tokens, _keyDim)) CopyToCache(cacheK, kHeads, start, tokens);
            using (Tensor vHeads = ReshapeToHeads(v, kvHeads, tokens, _valueDim)) CopyToCache(cacheV, vHeads, start, tokens);
            k.Dispose();
            v.Dispose();
            Tensor query = q.IsContiguous() ? q : Ops.NewContiguous(q);
            if (!ReferenceEquals(query, q)) q.Dispose();
            var result = new Tensor(scope.Allocator, DType.Float32, tokens, heads * _valueDim);
            int floor = Math.Min(_attentionFloor, start);
            int capacity = (int)cacheK.Sizes[1];
            long qRow = (long)heads * _keyDim, rRow = (long)heads * _valueDim, skip = (long)floor * _keyDim;
            float* qp = GetFloatPtr(query), rp = GetFloatPtr(result);
            for (int r = 0; r < tokens; r++)
            {
                int keys = start + r + 1 - floor;
                if (cacheK.ElementType == DType.Float16)
                    AttentionDecodeGroupedF16(qp + r * qRow, GetHalfPtrForRead(cacheK) + skip, GetHalfPtrForRead(cacheV) + skip, rp + r * rRow,
                        heads, kvHeads, _keyDim, capacity, keys, scale);
                else
                    AttentionDecodeGrouped(qp + r * qRow, GetFloatPtrForRead(cacheK) + skip, GetFloatPtrForRead(cacheV) + skip, rp + r * rRow,
                        heads, kvHeads, _keyDim, capacity, keys, scale, Environment.ProcessorCount);
            }
            query.Dispose();
            return result;
        }

        private Tensor PrefillAttention(RankScope scope, Tensor cacheK, Tensor cacheV, Tensor q, Tensor k, Tensor v, int start, int tokens)
        {
            int heads = scope.LocalHeads, kvHeads = scope.LocalKvHeads;
            int total = start + tokens;
            float scale = 1.0f / MathF.Sqrt(_keyDim);

            Tensor kHeads = ReshapeToHeads(k, kvHeads, tokens, _keyDim);
            k.Dispose();
            Tensor vHeads = ReshapeToHeads(v, kvHeads, tokens, _valueDim);
            v.Dispose();
            CopyToCache(cacheK, kHeads, start, tokens);
            CopyToCache(cacheV, vHeads, start, tokens);
            kHeads.Dispose();
            vHeads.Dispose();

            // MLX: the chunk's rows are in the cache; one fused SDPA over its live rows.
            var fused = new Tensor(scope.Allocator, DType.Float32, tokens, heads * _valueDim);
            if (MlxCachedAttention(scope, fused, q, cacheK, cacheV, tokens, total, scale))
            {
                q.Dispose();
                return fused;
            }
            fused.Dispose();

            Tensor qHeads = ReshapeToHeads(q, heads, tokens, _keyDim);
            q.Dispose();

            int group = heads / kvHeads;
            Tensor kExpanded = ExpandKVHeads(cacheK, group, total);
            Tensor vExpanded = ExpandKVHeads(cacheV, group, total);
            using Tensor kT = kExpanded.Transpose(1, 2);
            var scores = new Tensor(scope.Allocator, DType.Float32, heads, tokens, total);
            // Half operands by name on direct CUDA: K/V come from the F16 cache, Q and P are
            // rounded as ggml's flash attention rounds them.
            HalfOperandGemm.AddmmBatch(scores, 0, scores, scale, qHeads, kT);
            qHeads.Dispose();
            kExpanded.Dispose();

            if (_attentionFloor > 0)
            {
                using Tensor below = scores.Narrow(2, 0, Math.Min(_attentionFloor, total));
                Ops.Fill(below, float.NegativeInfinity);
            }
            if (IsGgmlBackend)
            {
                GgmlBasicOps.AttentionSoftmaxWithSinks(scores, sinks: null, numHeads: heads, seqLen: tokens,
                    kvLen: total, maskStartPos: start, slidingWindow: 0, scale: 1.0f);
            }
            else
            {
                Ops.AddCausalMask(scores, tokens, start, float.NegativeInfinity);
                Ops.Softmax(scores, scores);
            }

            var attn = new Tensor(scope.Allocator, DType.Float32, heads, tokens, _valueDim);
            HalfOperandGemm.AddmmBatch(attn, 0, attn, 1.0f, scores, vExpanded);
            scores.Dispose();
            vExpanded.Dispose();
            Tensor flat = ReshapeFromHeads(attn, heads, tokens, _valueDim);
            attn.Dispose();
            return flat;
        }

        /// <summary>SwiGLU FFN branch body for one rank (fused gate|up or separate projections).</summary>
        private Tensor Ffn(RankScope scope, Tensor input, int layer)
        {
            if (_moeLayer[layer])
                return MoeFfn(scope, input, layer, routing: null);
            string p = $"blk.{layer}.";
            int tokens = (int)input.Sizes[0];
            Tensor gate, up;
            if (_gateUpFused[layer])
            {
                if (scope.Degree == 1)
                    return FFN(input, p + "ffn_gate_up.weight", p + "ffn_down.weight", tokens);
                Tensor gateUp = Linear(scope, input, p + "ffn_gate_up.weight");
                int half = (int)(gateUp.Sizes[1] / 2);
                if (tokens == 1)
                {
                    gate = gateUp.Narrow(1, 0, half);
                    up = gateUp.Narrow(1, half, half);
                }
                else
                {
                    using (Tensor view = gateUp.Narrow(1, 0, half)) gate = Ops.NewContiguous(view);
                    using (Tensor view = gateUp.Narrow(1, half, half)) up = Ops.NewContiguous(view);
                }
                gateUp.Dispose();
            }
            else
            {
                gate = Linear(scope, input, p + "ffn_gate.weight");
                up = Linear(scope, input, p + "ffn_up.weight");
            }
            Ops.SiLUMul(gate, gate, up);
            up.Dispose();
            Tensor down = Linear(scope, gate, p + "ffn_down.weight");
            gate.Dispose();
            return down;
        }

        /// <summary>A projection on <paramref name="scope"/>'s device: the whole weight on a single
        /// device (LinearForward, with LoRA and every backend fast path), the rank's shard under
        /// tensor parallelism.</summary>
        protected Tensor Linear(RankScope scope, Tensor input, string weightName)
        {
            if (scope.Degree == 1)
                return LinearForward(input, weightName)
                    ?? throw new InvalidOperationException($"{Config.Architecture}: weight '{weightName}' is missing.");

            long t0 = Stopwatch.GetTimestamp();
            int rows = (int)input.Sizes[0];
            Tensor result;
            if (_tpQuantWeights.TryGetValue(weightName, out QuantizedWeight[] quant))
            {
                QuantizedWeight shard = quant[scope.Rank];
                result = new Tensor(scope.Allocator, DType.Float32, rows, (int)shard.Ne1);
                AddmmQuantManaged(result, input, shard);
            }
            else if (_tpWeights.TryGetValue(weightName, out Tensor[] dense))
            {
                Tensor shard = dense[scope.Rank];
                using Tensor transposed = shard.Transpose();
                result = new Tensor(scope.Allocator, DType.Float32, rows, (int)shard.Sizes[0]);
                Ops.Addmm(result, 0, result, 1.0f, input, transposed);
            }
            else
            {
                throw new InvalidOperationException($"{Config.Architecture}: weight '{weightName}' has no tensor-parallel shard.");
            }
            if (_tpWeightScales.TryGetValue(weightName, out float scale) && scale != 1.0f)
                Ops.Mul(result, result, scale);
            Interlocked.Add(ref _linearTicks, Stopwatch.GetTimestamp() - t0);
            return result;
        }

        // ---------------------------------------------------------------- devices

        /// <summary>Make <paramref name="stage"/>'s first device the one the single-device helpers
        /// (Embedding, LinearForward, RMSNormOp, ...) allocate on and run on.</summary>
        /// <summary>Make <paramref name="stage"/>'s first device the active one (a family's own graphs).</summary>
        protected void EnterGraphStage(int stage) => EnterStage(stage);

        private void EnterStage(int stage)
        {
            if (_mesh == null || _mesh.IsSingleDevice) return;
            _allocator = _mesh.Allocator(stage, 0);
            if (IsGgmlBackend) GgmlBasicOps.SetActiveRank(_mesh.GlobalDevice(stage, 0));
        }

        /// <summary>
        /// Mark a tensor just written through a host pointer, so no later device kernel reads a
        /// stale device copy of it. Metal's lazy-sync drains device work before host READS
        /// (GetFloatPtr, GetElementsAsFloat) but has no host-WRITE barrier. Call it right after
        /// the write, and only for tensors actually written on the host: a tensor whose newest
        /// value is still a pending device result must not be invalidated.
        /// </summary>
        protected void HostWritten(Tensor tensor)
        {
            if (IsGgmlBackend && tensor != null)
                GgmlBasicOps.InvalidateTensorHostBuffer(tensor);
        }

        /// <summary>The activation on <paramref name="target"/>; copied through host memory when it
        /// lives elsewhere (stage boundaries carry one [tokens, hidden] tensor).</summary>
        private Tensor MoveTo(Tensor tensor, IAllocator target)
        {
            if (ReferenceEquals(tensor.Storage.Allocator, target))
                return tensor;
            float[] values = TensorToFloatArray(tensor);
            var moved = new Tensor(target, DType.Float32, tensor.Sizes);
            moved.SetElementsAsFloat(values);
            HostWritten(moved);
            tensor.Dispose();
            return moved;
        }

        /// <summary>One copy of <paramref name="hidden"/> per rank of <paramref name="stage"/>;
        /// takes ownership of <paramref name="hidden"/> (it becomes rank 0's copy).</summary>
        private Tensor[] Replicate(Tensor hidden, int stage)
        {
            var copies = new Tensor[_mesh.TensorParallel];
            copies[0] = MoveTo(hidden, _mesh.Allocator(stage, 0));
            float[] values = null;
            for (int r = 1; r < copies.Length; r++)
            {
                IAllocator alloc = _mesh.Allocator(stage, r);
                copies[r] = new Tensor(alloc, DType.Float32, copies[0].Sizes);
                if (IsGgmlBackend || alloc is TensorSharp.Cpu.CpuAllocator)
                {
                    Ops.Copy(copies[r], copies[0]);
                }
                else
                {
                    values ??= TensorToFloatArray(copies[0]);
                    copies[r].SetElementsAsFloat(values);
                    HostWritten(copies[r]);
                }
            }
            return copies;
        }

        /// <summary>A replicated (unsharded) weight resident on (stage, rank)'s device, copied once.</summary>
        protected Tensor Replica(Tensor weight, int stage, int rank)
        {
            IAllocator target = _mesh.Allocator(stage, rank);
            if (ReferenceEquals(weight.Storage.Allocator, target) || IsGgmlBackend)
                return weight;
            return _replicas.GetOrAdd((weight, _mesh.GlobalDevice(stage, rank)), _ =>
            {
                var copy = new Tensor(target, weight.ElementType, weight.Sizes);
                if (target is TensorSharp.Cpu.CpuAllocator)
                    Ops.Copy(copy, weight);
                else
                    copy.SetElementsAsFloat(weight.GetElementsAsFloat((int)weight.ElementCount()));
                return copy;
            });
        }

        private void PreReplicateNormWeights()
        {
            if (_mesh.TensorParallel == 1) return;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!OwnsLayer(l)) continue;
                int stage = _mesh.StageOf(l);
                var names = new[] { $"blk.{l}.attn_norm.weight", FfnNormWeight(l) }.Concat(ReplicatedLayerWeights(l));
                foreach (string name in names)
                {
                    if (!_weights.TryGetValue(name, out Tensor w)) continue;
                    for (int r = 0; r < _mesh.TensorParallel; r++)
                        Replica(w, stage, r);
                }
            }
        }

        public override void Dispose()
        {
            StopHostWarm();
            foreach (RankExpertStack stack in _rankStacks.Values) stack.Stack.Dispose();
            _rankStacks.Clear();
            foreach (Tensor t in AllCacheTensors(_cacheK)) t.Dispose();
            foreach (Tensor t in AllCacheTensors(_cacheV)) t.Dispose();
            _cacheK = _cacheV = null;
            foreach (Tensor copy in _replicas.Values) copy.Dispose();
            _replicas.Clear();
            DisposeSlots();
            ClearGraphCache();
            DisposeRecurrentStates();
            DropPendingMedia();
            base.Dispose();
        }
    }
}
