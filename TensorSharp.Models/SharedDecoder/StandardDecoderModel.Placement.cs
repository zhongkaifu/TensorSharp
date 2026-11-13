// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;

using TensorSharp.Cuda;
using TensorSharp.GGML;
using TensorSharp.Graph;
using TensorSharp.Models.Placement;

namespace TensorSharp.Models
{
    /// <summary>
    /// Device placement for the shared decoder runtime: which device each layer and weight
    /// lives on, how tensor-parallel weights are sharded per stage, and where quantized
    /// weights are made resident. Nothing here is family specific.
    /// </summary>
    public abstract partial class StandardDecoderModel
    {
        private void BuildMesh(string executionIdentity)
        {
            int stages = _requestedStages, tp = _requestedTensorParallel;
            if (tp > 1 && stages == 1 && LayerSplitGroup == null && _tpGroup is GgmlTensorParallelGroup && GraphTensorParallelism)
            {
                // Tensor parallelism as whole-forward graphs on ggml's meta device: the model is laid
                // out as on one device (global heads, unsharded weights), and each graph's weight and
                // state leaves are spread over the ranks by their split (GraphWeightSplit).
                ValidateShardable(tp);
                _graphTpRanks = Enumerable.Range(0, tp).ToArray();
                _shardDegree = 1;
                _shardRankOffset = 0;
                _localHeads = Config.NumHeads;
                _localKvHeads = Config.NumKVHeads;
                IAllocator rank0 = _tpGroup.GetAllocator(0);
                _mesh = new DeviceMesh(1, 1, _ => rank0, null, null);
                Console.WriteLine($"  Tensor parallelism: whole-forward graphs over {tp} GPUs (ggml's meta device).");
                return;
            }
            // A distributed TP group shards by its GLOBAL degree; a stage group (local or a pipeline node's) by its own.
            bool distributedTp = tp > 1 && stages == 1 && LayerSplitGroup == null;
            int shardDegree = tp == 1 ? 1 : distributedTp ? GlobalTpDegree : tp;
            _shardDegree = shardDegree;
            _shardRankOffset = distributedTp ? TpRankOffset : 0;
            ValidateShardable(shardDegree);
            _localHeads = Config.NumHeads / shardDegree;
            _localKvHeads = Config.NumKVHeads / shardDegree;

            if (LayerSplitGroup != null && tp > 1)
            {
                // A pipeline node whose stages are tensor-parallel groups: the node's own layers (the agreed plan) over
                // its P local stages of T devices, each stage reducing within its devices.
                ConfigureLayerPlacement(executionIdentity + "|" + _kvCacheDtype, balanceWeightBytes: true);
                ITensorParallelGroup local = ((INestedTensorParallelGroup)LayerSplitGroup).LocalGroup;
                _mesh = new DeviceMesh(LayerSplitDegree, tp, d => local.GetAllocator(d), local, LayerPlacement);
                if (IsGgmlBackend)
                    GgmlBasicOps.TensorParallelSetGlobalGeometry(tp, 0);
            }
            else if (LayerSplitGroup != null || (stages > 1 && tp == 1))
            {
                // ModelBase's whole-layer placement: local devices, or this node's share of a
                // multi-node pipeline (whose nodes must agree on everything in the identity).
                ConfigureLayerPlacement(executionIdentity + "|" + _kvCacheDtype, balanceWeightBytes: true);
                _mesh = new DeviceMesh(LayerSplitDegree, 1, LayerDeviceAllocator, null, LayerPlacement);
            }
            else if (stages > 1)
            {
                if (_tpGroup == null || _tpGroup.Degree != stages * tp)
                    throw new InvalidOperationException($"Hybrid placement needs {stages * tp} devices.");
                _mesh = new DeviceMesh(stages, tp, d => _tpGroup.GetAllocator(d), _tpGroup, WeightedPlan(stages));
                // ModelBase published the whole group (T x P) as the native kernels' global TP
                // geometry; every collective here spans one stage's T devices.
                if (IsGgmlBackend)
                    GgmlBasicOps.TensorParallelSetGlobalGeometry(tp, 0);
            }
            else if (tp > 1)
            {
                if (_tpGroup == null || _tpGroup.Degree != tp)
                    throw new InvalidOperationException($"Tensor parallelism needs a group of {tp} devices.");
                _mesh = new DeviceMesh(1, tp, d => _tpGroup.GetAllocator(d), _tpGroup, null);
                _rankGraphs = RankGraphsServe(tp);
                if (_rankGraphs)
                {
                    // A streaming executor sums each partial as the last rank hands it in; a compiling one (GGML) runs
                    // the ranks' plans together, pausing at every all-reduce (IRankedGraphExecutor).
                    if (ExecutorFor(_mesh.Allocator(0, 0)) is TensorGraphExecutor)
                    {
                        var collective = new MeshCollective(_mesh, 0);
                        for (int r = 0; r < tp; r++)
                            ((TensorGraphExecutor)ExecutorFor(_mesh.Allocator(0, r))).JoinCollective(collective, r);
                    }
                    Console.WriteLine($"  Tensor parallelism: each of {tp} devices runs its shard of every step as a graph, in lockstep.");
                }
            }
            else
            {
                _mesh = new DeviceMesh(1, 1, _ => _allocator, null, null);
            }
        }

        /// <summary>Ranks of tensor-parallel graphs (ggml's meta device), or null.</summary>
        private int[] _graphTpRanks;

        /// <summary>
        /// Tensor parallelism as each rank's own program (direct CUDA, and the managed executor of a simulated group):
        /// every rank traces its shard of the step on its device's executor, the ranks' nodes interleaved block by block
        /// (TryRunRankedGraphStep); each block's partial is summed by <see cref="GraphOp.AllReduce"/> before it joins the
        /// rank's copy of the residual stream. Routed experts are sharded like the dense feed-forwards (each expert's
        /// width), not spread whole over the ranks as the per-op path does, so such a model runs only as these graphs.
        /// </summary>
        private bool _rankGraphs;

        /// <summary>Whether this model's tensor-parallel steps run as per-rank graphs (see <see cref="_rankGraphs"/>).</summary>
        protected bool RankGraphs => _rankGraphs;

        /// <summary>Whether the family's lowering traces one rank's shard (its local heads, sharded weights and per-rank
        /// state); default: families without recurrent layers, whose blocks the runtime emits.</summary>
        protected virtual bool RanksLower => !_anyRecurrent;

        private bool RankGraphsServe(int tp)
            => GraphLoweringEnabled && GraphRotary != null && RanksLower && !MoeCpuOffloadConfig.IsEnabled
               && StepAdapter == null && GlobalTpDegree == tp && TpRankOffset == 0
               && ExecutorFor(_mesh.Allocator(0, 0)) switch
               {
                   TensorGraphExecutor => true,
                   IRankedGraphExecutor ranked => ranked.RunsRanks(tp),
                   _ => false,
               };

        /// <summary>A stage group's sum for per-rank graphs: the last rank's partial runs the group's all-reduce.</summary>
        private sealed class MeshCollective : IGraphCollective
        {
            private readonly DeviceMesh _mesh;
            private readonly int _stage;
            private readonly Tensor[] _parts;
            private int _arrived;

            public MeshCollective(DeviceMesh mesh, int stage)
            {
                _mesh = mesh;
                _stage = stage;
                _parts = new Tensor[mesh.TensorParallel];
            }

            public int Degree => _parts.Length;

            public void Contribute(int rank, Tensor partial)
            {
                if (_parts[rank] != null)
                {
                    partial.Dispose();
                    throw new InvalidOperationException($"Tensor-parallel rank {rank} reduced twice before every rank reduced once.");
                }
                _parts[rank] = partial;
                if (++_arrived < _parts.Length) return;
                try
                {
                    _mesh.AllReduce(_stage, _parts);
                }
                finally
                {
                    for (int r = 0; r < _parts.Length; r++)
                    {
                        _parts[r].Dispose();
                        _parts[r] = null;
                    }
                    _arrived = 0;
                }
            }
        }

        /// <summary>
        /// Tensor parallelism runs as whole-forward graphs on ggml's meta device: the family lowers
        /// (rotary, mixers) and no layer keeps its experts on the host; recurrent stacks wait for
        /// their state splits. Opt-in (TS_GRAPH_TENSOR_PARALLEL=1) until validated on GPUs with peer
        /// access: on MIG slices without it (the 2026-11-10 lab VM) the meta device's internal
        /// all-reduce faults in flash attention or hangs, and --tp keeps the per-op path.
        /// </summary>
        private bool GraphTensorParallelism
            => GraphRotary != null && !_anyRecurrent && !MoeCpuOffloadConfig.IsEnabled
               && Environment.GetEnvironmentVariable("TS_GRAPH_TENSOR_PARALLEL") == "1";

        private void ValidateShardable(int degree)
        {
            if (degree <= 1) return;
            var problems = new List<string>();
            if (Config.NumHeads % degree != 0) problems.Add($"{Config.NumHeads} attention heads");
            if (Config.NumKVHeads % degree != 0) problems.Add($"{Config.NumKVHeads} KV heads");
            if (Config.IntermediateSize > 0 && Config.IntermediateSize % degree != 0) problems.Add($"FFN width {Config.IntermediateSize}");
            problems.AddRange(ShardingProblems(degree));
            if (problems.Count > 0)
                throw new NotSupportedException(
                    $"{Config.Architecture} cannot be split over {degree} tensor-parallel devices: {string.Join(", ", problems)} " +
                    $"must each divide by {degree}. Choose a --tp that divides them, or use --layer-split instead.");
        }

        /// <summary>Whole-layer stages balanced by resident weight bytes (ModelBase's balancer).</summary>
        private LayerSplitPlan WeightedPlan(int stages)
        {
            var bytes = new long[Config.NumLayers];
            long embedding = 0, head = 0;
            foreach (GgufTensorInfo info in _gguf.Tensors.Values)
            {
                long resident = IsQuantizedLinearWeight(info) ? _gguf.GetTensorByteCount(info) : checked(info.NumElements * 4);
                if (LayerSplitPlan.TryGetLayer(info.Name, out int layer) && layer < bytes.Length)
                    bytes[layer] = checked(bytes[layer] + resident);
                else if (info.Name == "token_embd.weight") embedding = checked(embedding + resident);
                else head = checked(head + resident);
            }
            if (!_gguf.Tensors.ContainsKey("output.weight")) head = checked(head + embedding);
            return LayerSplitPlan.CreateWeighted(bytes, stages, 0, 1, embedding, head, false);
        }

        // ---------------------------------------------------------------- weight ownership

        /// <summary>The layer whose placement a layer follows: a draft layer after the stack sits with
        /// the last decoder layer (its node, its stage).</summary>
        private int PlacementLayer(int layer) => layer >= Config.NumLayers ? Config.NumLayers - 1 : layer;

        private bool IsWholeLayerSplit => _mesh != null && _mesh.TensorParallel == 1 && LayerPlacement != null;

        private bool IsHeadWeight(string name) =>
            name.StartsWith("output", StringComparison.Ordinal) ||
            (name == "token_embd.weight" && LayerPlacement != null && !LayerPlacement.IsFirst);

        /// <summary>A multi-node pipeline loads only this node's layers (plus the embedding on
        /// the first stage and the head on the last).</summary>
        protected override bool ShouldLoadWeight(GgufTensorInfo info)
        {
            if (LayerPlacement == null) return true;
            if (LayerSplitPlan.TryGetLayer(info.Name, out int layer)) return LayerPlacement.Owns(PlacementLayer(layer));
            if (info.Name == "token_embd.weight")
                return LayerPlacement.IsFirst || (LayerPlacement.IsLast && (!_gguf.Tensors.ContainsKey("output.weight") || DrafterReadsTrunkEmbedding));
            if (info.Name.StartsWith("output", StringComparison.Ordinal)) return LayerPlacement.IsLast;
            return true;
        }

        protected override int PreloadRankForWeight(string name)
        {
            if (!IsWholeLayerSplit) return 0;
            if (LayerSplitPlan.TryGetLayer(name, out int layer)) return LayerPlacement.LocalRank(PlacementLayer(layer));
            return IsHeadWeight(name) ? LayerSplitDegree - 1 : 0;
        }

        protected override IAllocator AllocatorForWeight(string name)
        {
            if (_mesh == null || _mesh.IsSingleDevice) return base.AllocatorForWeight(name);
            if (IsWholeLayerSplit)
            {
                if (LayerSplitPlan.TryGetLayer(name, out int owned)) return LayerAllocator(PlacementLayer(owned));
                return LayerAllocator(IsHeadWeight(name) ? LayerPlacement.EndLayer - 1 : LayerPlacement.FirstLayer);
            }
            return _mesh.Allocator(StageOfWeight(name), 0);
        }

        protected override bool ShouldRetainCudaHostQuantWeight(string name)
            => (IsWholeLayerSplit && !LayerSplitPlan.TryGetLayer(name, out _)) || base.ShouldRetainCudaHostQuantWeight(name);

        /// <summary>The stage holding <paramref name="name"/> on a local mesh.</summary>
        private int StageOfWeight(string name)
        {
            if (LayerSplitPlan.TryGetLayer(name, out int layer))
                return layer < Config.NumLayers ? _mesh.StageOf(layer) : _mesh.Stages - 1;
            return name.StartsWith("output", StringComparison.Ordinal) ? _mesh.Stages - 1 : 0;
        }

        // ---------------------------------------------------------------- sharding

        /// <summary>
        /// Shard every layer's projections for its stage's tensor-parallel group, with the
        /// existing (tested) sharding helpers bound to that group in turn: column-parallel
        /// Q/K/V and gate/up (segment-aware when fused), row-parallel attention output and down.
        /// </summary>
        private void ShardForMesh()
        {
            for (int stage = 0; stage < _mesh.Stages; stage++)
            {
                using IDisposable bound = BindTensorParallelGroup(_mesh.Group(stage));
                var columns = new List<string>();
                var rows = new List<string>();
                for (int l = 0; l < Config.NumLayers; l++)
                {
                    if (!OwnsLayer(l) || _mesh.StageOf(l) != stage) continue;
                    string p = $"blk.{l}.";
                    int qRows = Config.NumHeads * _keyDim * (GatedAttentionOutput ? 2 : 1);
                    if (_recurrentLayer[l])
                        ShardRecurrentLayer(l);
                    else if (_qkvFused[l])
                        ShardConcatenatedColumnParallel(p + "attn_qkv.weight",
                            qRows, Config.NumKVHeads * _keyDim, Config.NumKVHeads * _valueDim);
                    else
                        columns.AddRange(new[] { p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight" });
                    if (_gateUpFused[l])
                        ShardFusedGateUpColumnParallel(p + "ffn_gate_up.weight");
                    else
                        columns.AddRange(new[] { p + "ffn_gate.weight", p + "ffn_up.weight" });
                    if (!_recurrentLayer[l])
                        rows.Add(p + "attn_output.weight");
                    rows.Add(p + "ffn_down.weight");
                    if (_rankGraphs && _moeLayer[l])
                    {
                        // Each routed expert and the shared expert split by their width, like a dense feed-forward:
                        // every rank runs every selected expert's slice. GGML's mul_mat_id reads one stacked tensor, so
                        // there each rank gets its own stack; other executors take the members one at a time.
                        if (IsGgmlBackend)
                        {
                            ShardStackedExperts(p + "ffn_gate_exps.weight", columns: true, stage);
                            ShardStackedExperts(p + "ffn_up_exps.weight", columns: true, stage);
                            ShardStackedExperts(p + "ffn_down_exps.weight", columns: false, stage);
                        }
                        else
                        {
                            columns.AddRange(new[] { p + "ffn_gate_exps.", p + "ffn_up_exps." });
                            rows.Add(p + "ffn_down_exps.");
                        }
                        columns.AddRange(new[] { p + "ffn_gate_shexp.weight", p + "ffn_up_shexp.weight" });
                        rows.Add(p + "ffn_down_shexp.weight");
                    }
                }
                ShardWeightsForTensorParallelism(columns, rows);
                if (_mesh.Stages == 1) ShardHeadByVocab();
            }
        }

        /// <summary>The per-rank vocabulary slices of the LM head on per-rank graphs (each rank's rows, in rank order).</summary>
        private const string VocabHeadName = "output.weight@vocab";

        /// <summary>The head rows each rank holds under <see cref="VocabHeadName"/>, or null: rank 0 runs the whole head.</summary>
        private int[] _vocabHeadRows;

        /// <summary>Each rank scores its slice of the vocabulary (tests).</summary>
        internal bool HeadSplitByVocabulary => _vocabHeadRows != null;

        /// <summary>
        /// Per-rank graphs: each rank keeps a contiguous slice of the untied head's vocabulary rows, so every rank scores
        /// its share of the vocabulary at once and the host joins the slices. Rank 0 ran the whole head: on Qwen3.5-9B
        /// `--tp 2` that was ~1.9 ms of a ~11 ms decode step (2x A40), the main gap to Qwen35Model, which split it the
        /// same way. The whole head stays for the draft heads and the per-op path; a tied head (the embedding) is not
        /// split.
        /// </summary>
        private unsafe void ShardHeadByVocab()
        {
            const string head = "output.weight";
            int ranks = _mesh.TensorParallel;
            if (!_rankGraphs || ranks < 2) return;
            long vocabRows;
            if (_quantWeights.TryGetValue(head, out QuantizedWeight q)) vocabRows = q.Ne1;
            else if (_weights.TryGetValue(head, out Tensor w)) vocabRows = w.Sizes[0];
            else return;
            if (vocabRows != Config.VocabSize || vocabRows < ranks) return;
            var counts = new int[ranks];
            for (int r = 0; r < ranks; r++) counts[r] = (int)(vocabRows / ranks + (r < vocabRows % ranks ? 1 : 0));
            if (q != null)
            {
                long rowBytes = NativeDequant.RowSize(q.GgmlType, q.Ne0);
                var shards = new QuantizedWeight[ranks];
                long first = 0;
                for (int r = 0; r < ranks; r++)
                {
                    long bytes = counts[r] * rowBytes;
                    IntPtr data = QuantizedWeight.AllocateBuffer(bytes);
                    Buffer.MemoryCopy((byte*)q.Data + first * rowBytes, (byte*)data, bytes, bytes);
                    shards[r] = new QuantizedWeight(data, bytes, q.GgmlType, q.Ne0, counts[r]) { Scale = q.Scale };
                    first += counts[r];
                }
                _tpQuantWeights[VocabHeadName] = shards;
            }
            else
            {
                Tensor full = _weights[head];
                long cols = full.Sizes[1];
                var shards = new Tensor[ranks];
                long first = 0;
                for (int r = 0; r < ranks; r++)
                {
                    shards[r] = new Tensor(_tpGroup.GetAllocator(r), DType.Float32, counts[r], cols);
                    Buffer.MemoryCopy(GetFloatPtr(full) + first * cols, GetFloatPtr(shards[r]), counts[r] * cols * 4L, counts[r] * cols * 4L);
                    first += counts[r];
                }
                _tpWeights[VocabHeadName] = shards;
            }
            _vocabHeadRows = counts;
            Console.WriteLine($"  {Config.Architecture}: the LM head is split by vocabulary over {ranks} ranks ({string.Join(" + ", counts)} rows).");
        }

        /// <summary>Each rank's slice of a stacked expert weight, as its own contiguous stack (GGML per-rank graphs).</summary>
        private readonly Dictionary<(string Name, int Rank), RankExpertStack> _rankStacks = new();

        /// <summary>
        /// Cut <paramref name="name"/>'s stacked experts [ne0, ne1, E] into one stack per rank of
        /// <paramref name="stage"/>: each expert's output rows [r·ne1/T, (r+1)·ne1/T) (<paramref name="columns"/>, gate and
        /// up), or each row's input blocks [r·ne0/T, ...) (down, split on quantization-block boundaries).
        /// </summary>
        private unsafe void ShardStackedExperts(string name, bool columns, int stage)
        {
            if (!_stackedExpertWeights.TryGetValue(name, out StackedExpertWeights w))
                throw new NotSupportedException($"{Config.Architecture}: per-rank graphs on GGML need '{name}' as one stacked tensor.");
            int degree = _mesh.TensorParallel, experts = w.NumExperts;
            long ne0 = w.PerExpertNe0, ne1 = w.PerExpertNe1;
            long rowBytes = NativeDequant.RowSize(w.GgmlType, ne0), perExpert = rowBytes * ne1;
            var type = (GgmlTensorType)w.GgmlType;
            long blockSize = GgufFile.GetBlockSize(type), typeSize = GgufFile.GetTypeSize(type);
            if (columns ? ne1 % degree != 0 : (ne0 / blockSize) % degree != 0)
                throw new NotSupportedException($"{Config.Architecture}: '{name}' ({ne0} x {ne1}) does not split over {degree} ranks.");
            long shardNe0 = columns ? ne0 : ne0 / degree, shardNe1 = columns ? ne1 / degree : ne1;
            long shardRowBytes = columns ? rowBytes : (ne0 / blockSize / degree) * typeSize;
            long shardBytes = checked(shardRowBytes * shardNe1 * experts);
            byte* src = (byte*)w.Data;
            for (int r = 0; r < degree; r++)
            {
                IntPtr buffer = QuantizedWeight.AllocateBuffer(shardBytes);
                byte* dst = (byte*)buffer;
                int rank = r;
                System.Threading.Tasks.Parallel.For(0, experts, e =>
                {
                    byte* from = src + e * perExpert;
                    byte* to = dst + e * shardRowBytes * shardNe1;
                    if (columns)
                        Buffer.MemoryCopy(from + rank * shardNe1 * rowBytes, to, shardRowBytes * shardNe1, shardRowBytes * shardNe1);
                    else
                        for (long j = 0; j < ne1; j++)
                            Buffer.MemoryCopy(from + j * rowBytes + rank * shardRowBytes, to + j * shardRowBytes, shardRowBytes, shardRowBytes);
                });
                var stack = new QuantizedWeight(buffer, shardBytes, w.GgmlType, shardNe0, shardNe1 * experts);
                _rankStacks[(name, _mesh.GlobalDevice(stage, r))] = new RankExpertStack(stack, shardNe0, shardNe1, experts);
            }
        }

        /// <summary>One rank's stacked expert slices [ne0, ne1, E] (contiguous), read whole by mul_mat_id.</summary>
        private sealed class RankExpertStack : IGraphExpertWeights
        {
            public RankExpertStack(QuantizedWeight stack, long ne0, long ne1, int experts)
            {
                Stack = stack;
                Ne0 = ne0;
                Ne1 = ne1;
                Ne2 = experts;
                CacheKey = stack.EnsureDeviceCacheKey();
            }

            public QuantizedWeight Stack { get; }
            public int Format => Stack.GgmlType;
            public long Ne0 { get; }
            public long Ne1 { get; }
            public long Ne2 { get; }
            public IntPtr Data => Stack.Data;
            public IntPtr CacheKey { get; }
            public long Bytes => Stack.RawBytes;
            public IGraphWeight Expert(int index)
                => throw new NotSupportedException("a rank's stacked expert slices are read whole (mul_mat_id)");
        }

        // ---------------------------------------------------------------- residency

        private void PrepareResidency()
        {
            // Tensor-parallel graphs upload each weight's slices to the meta device on first use.
            if (_graphTpRanks != null) return;
            if (_mesh.TensorParallel == 1)
            {
                // One device, or whole layers: the single-device preload places each weight on
                // its layer's device (PreloadRankForWeight / AllocatorForWeight).
                PrepareCudaQuantizedWeightsForInference();
                return;
            }
            if (_mesh.Stages == 1 && LayerSplitGroup == null)
            {
                if (_rankGraphs) PrepareRankGraphResidency();
                else PrepareCudaQuantizedWeightsForInferenceTP();
                return;
            }
            PrepareHybridResidency();
        }

        /// <summary>
        /// Hybrid placement: every stage's shards onto that stage's ranks (concurrently, each
        /// worker pinned to its device), and every unsharded weight onto the first device of the
        /// stage that reads it. Host copies are kept: a hybrid run is new, and keeping them lets
        /// any op that misses the device cache still read the weight.
        /// </summary>
        private void PrepareHybridResidency()
        {
            bool ggml = _backend is BackendType.GgmlCuda or BackendType.GgmlVulkan;
            if (!ggml && _backend != BackendType.Cuda)
                return; // host-memory devices (simulated meshes): nothing to upload
            if (ggml) EnsureQuantBackendAvailable();

            long[] bytes = new long[_mesh.DeviceCount];
            for (int stage = 0; stage < _mesh.Stages; stage++)
            {
                int s = stage;
                var shards = _tpQuantWeights.Where(kv => StageOfWeight(kv.Key) == s).ToArray();
                _mesh.RunPerRank(s, r =>
                {
                    foreach (var kv in shards)
                    {
                        QuantizedWeight shard = kv.Value[r];
                        if (!shard.HasHostData || !ShouldPreloadCudaQuantWeightToDevice(kv.Key)) continue;
                        if (Preload(shard, _mesh.Allocator(s, r), ggml))
                            bytes[_mesh.GlobalDevice(s, r)] += shard.RawBytes;
                    }
                });
            }
            foreach (var kv in _quantWeights)
            {
                QuantizedWeight weight = kv.Value;
                if (!weight.HasHostData || !ShouldPreloadCudaQuantWeightToDevice(kv.Key)) continue;
                if (ggml && kv.Key == "token_embd.weight" && !CanUseGgmlQuantizedGetRows(weight.GgmlType) &&
                    (_quantWeights.ContainsKey("output.weight") || _weights.ContainsKey("output.weight")))
                    continue; // a host row gather serves the embedding
                int device = _mesh.GlobalDevice(StageOfWeight(kv.Key), 0);
                if (ggml) GgmlBasicOps.SetActiveRank(device);
                if (Preload(weight, _mesh.Allocator(StageOfWeight(kv.Key), 0), ggml))
                    bytes[device] += weight.RawBytes;
            }
            if (ggml) GgmlBasicOps.SetActiveRank(0);
            for (int d = 0; d < bytes.Length; d++)
                Console.WriteLine($"  Hybrid device {d} (stage {d / _mesh.TensorParallel}, rank {d % _mesh.TensorParallel}): " +
                    $"{bytes[d] / 1024 / 1024} MB of quantized weights resident.");
        }

        /// <summary>
        /// Per-rank graphs: each shard onto its rank's device, and every replicated weight onto every rank (each rank's
        /// executor finds a weight under its own allocator); the head stays on rank 0, which runs it. Host copies of the
        /// replicated weights are kept (the per-op embedding and sampling read them); the shards' are released.
        /// </summary>
        private void PrepareRankGraphResidency()
        {
            bool ggml = _backend is BackendType.GgmlCuda or BackendType.GgmlVulkan;
            if (_backend != BackendType.Cuda && !ggml) return;   // host-memory devices (simulated groups): nothing to upload
            if (ggml) EnsureQuantBackendAvailable();
            int ranks = _mesh.TensorParallel;
            long[] bytes = new long[ranks];
            var resident = new HashSet<QuantizedWeight>[ranks];
            for (int r = 0; r < ranks; r++) resident[r] = new HashSet<QuantizedWeight>(ReferenceEqualityComparer.Instance);
            void PreloadShards(int r)
            {
                foreach (var kv in _tpQuantWeights)
                {
                    QuantizedWeight shard = kv.Value[r];
                    if (!shard.HasHostData || !ShouldPreloadCudaQuantWeightToDevice(kv.Key)) continue;
                    if (Preload(shard, _mesh.Allocator(0, r), ggml))
                    {
                        bytes[r] += shard.RawBytes;
                        resident[r].Add(shard);
                    }
                }
            }
            // A ggml preload lands on the calling thread's active rank: one thread, rank by rank.
            if (ggml)
                for (int r = 0; r < ranks; r++)
                {
                    GgmlBasicOps.SetActiveRank(_mesh.GlobalDevice(0, r));
                    PreloadShards(r);
                    // The rank's expert stacks (each its own copy, read whole by its graph's mul_mat_id).
                    foreach (var ((_, device), stack) in _rankStacks)
                        if (device == _mesh.GlobalDevice(0, r) && stack.Stack.HasHostData && Preload(stack.Stack, null, ggml: true))
                        {
                            bytes[r] += stack.Bytes;
                            stack.Stack.ReleaseHostData();
                        }
                }
            else
                _mesh.RunPerRank(0, PreloadShards);
            foreach (var kv in _quantWeights)
            {
                QuantizedWeight weight = kv.Value;
                if (!weight.HasHostData || !ShouldPreloadCudaQuantWeightToDevice(kv.Key) || kv.Key == "token_embd.weight") continue;
                int holders = IsHeadWeight(kv.Key) ? 1 : ranks;
                for (int r = 0; r < holders; r++)
                {
                    if (ggml) GgmlBasicOps.SetActiveRank(_mesh.GlobalDevice(0, r));
                    if (Preload(weight, _mesh.Allocator(0, r), ggml)) bytes[r] += weight.RawBytes;
                }
            }
            if (ggml) GgmlBasicOps.SetActiveRank(0);
            // Only a shard its rank holds gives up its host copy (another is read from the host).
            foreach (var kv in _tpQuantWeights)
                for (int r = 0; r < kv.Value.Length; r++)
                    if (resident[r].Contains(kv.Value[r]) && !ShouldRetainCudaHostQuantWeight(kv.Key)) kv.Value[r].ReleaseHostData();
            for (int r = 0; r < bytes.Length; r++)
                Console.WriteLine($"  Rank {r}: {bytes[r] / 1024 / 1024} MB of quantized weights resident.");
        }

        private static bool Preload(QuantizedWeight weight, IAllocator allocator, bool ggml)
        {
            IntPtr key = weight.EnsureDeviceCacheKey();
            if (ggml)
            {
                if (GgmlBasicOps.PreloadQuantizedWeight(key, weight.Data, weight.GgmlType, weight.Ne0, weight.Ne1, weight.RawBytes))
                    return true;
                weight.MarkDevicePreloadTooLarge();
                return false;
            }
            if (allocator is not CudaAllocator cuda || !CudaQuantizedOps.SupportsQuantizedType(weight.GgmlType))
                return false;
            CudaQuantizedOps.PreloadQuantizedWeight(cuda, key, weight.Data, weight.GgmlType, weight.Ne0, weight.Ne1, weight.RawBytes);
            return true;
        }
    }
}
