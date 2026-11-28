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
                EnableRankGraphs(tp);
            }
            else if (tp > 1)
            {
                if (_tpGroup == null || _tpGroup.Degree != tp)
                    throw new InvalidOperationException($"Tensor parallelism needs a group of {tp} devices.");
                _mesh = new DeviceMesh(1, tp, d => _tpGroup.GetAllocator(d), _tpGroup, null);
                EnableRankGraphs(tp);
            }
            else
            {
                _mesh = new DeviceMesh(1, 1, _ => _allocator, null, null);
            }
        }

        /// <summary>
        /// Per-rank graphs for a local mesh of <paramref name="tp"/> ranks a stage (<see cref="_rankGraphs"/>): one stage's
        /// group (tensor parallelism), or every stage's of a hybrid placement. GGML's executor runs the stage's plans together,
        /// pausing at every all-reduce (IRankedGraphExecutor); the tensor executors run each rank's own plan in lockstep, the
        /// stage's last rank to reach an all-reduce summing every partial (a collective a stage, TensorGraphExecutor.RunRanked).
        /// </summary>
        private void EnableRankGraphs(int tp)
        {
            _rankGraphs = RankGraphsServe(tp);
            if (!_rankGraphs) return;
            IGraphExecutor lead = ExecutorFor(_mesh.Allocator(0, 0));
            _rankSumsCrossHost = lead is IRankedGraphExecutor ranked && ranked.RanksSumOnHost(tp, _mesh.Stages);
            if (lead is TensorGraphExecutor)
                for (int s = 0; s < _mesh.Stages; s++)
                {
                    var collective = new MeshCollective(_mesh, s);
                    for (int r = 0; r < tp; r++)
                        ((TensorGraphExecutor)ExecutorFor(_mesh.Allocator(s, r))).JoinCollective(collective, r);
                }
            Console.WriteLine(_mesh.Stages == 1
                ? $"  Tensor parallelism: each of {tp} devices runs its shard of every step as a graph, in lockstep."
                : $"  Hybrid parallelism: each of {_mesh.Stages} whole-layer stages runs its layers as per-rank graphs over its {tp} devices.");
        }

        /// <summary>
        /// Tensor parallelism as each rank's own program (every executor that lowers the shared decoder: GGML's, direct CUDA's
        /// and the managed executor of a simulated group, all as compiled plans that replay and advance a decode step): every
        /// rank traces its shard of the step for its device, the ranks' nodes interleaved block by block, and the ranks' plans
        /// run together from one all-reduce to the next (TryRunRankedGraphStep, RankRunner); each block's partial is summed by
        /// <see cref="GraphOp.AllReduce"/> before it joins the rank's copy of the residual stream. A hybrid placement runs each
        /// whole-layer stage so over its own group, the stream passing from stage to stage (TryRunStagedGraphStep). Routed
        /// experts are sharded like the dense feed-forwards (each expert's width), not spread whole over the ranks as the
        /// per-op path does, so such a model runs only as these graphs.
        /// </summary>
        private bool _rankGraphs;

        /// <summary>Whether this model's tensor-parallel steps run as per-rank graphs (see <see cref="_rankGraphs"/>).</summary>
        protected bool RankGraphs => _rankGraphs;

        /// <summary>Tests: whether the placement's steps run as per-rank graphs (tensor parallelism, or every stage of a hybrid).</summary>
        internal bool RunsRankGraphs => _rankGraphs;

        /// <summary>Per-rank graphs whose all-reduces go through host memory (<see cref="IRankedGraphExecutor.RanksSumOnHost"/>):
        /// every step pays a host round trip a reduction, two a layer.</summary>
        private bool _rankSumsCrossHost;

        /// <summary>Whether this placement's per-rank sums cross the host (see <see cref="_rankSumsCrossHost"/>).</summary>
        internal bool RankSumsCrossHost => _rankSumsCrossHost;

        /// <summary>Whether the family's lowering traces one rank's shard (its local heads, sharded weights and per-rank
        /// state); default: families without recurrent layers, whose blocks the runtime emits.</summary>
        protected virtual bool RanksLower => !_anyRecurrent;

        /// <summary>Whether the mesh's steps run as per-rank graphs: the family lowers a rank's shard, no layer's experts run on
        /// the host, and the group is this process's own devices (a distributed group's ranks of other nodes run the per-op
        /// path's collectives); the executor runs ranks of <paramref name="tp"/> devices over the mesh's stages.</summary>
        private bool RankGraphsServe(int tp)
            => GraphLoweringEnabled && AttentionLowers && RanksLower && !MoeCpuOffloadConfig.IsEnabled
               && StepAdapter == null && GlobalTpDegree == _mesh.DeviceCount && TpRankOffset == 0
               && ExecutorFor(_mesh.Allocator(0, 0)) switch
               {
                   TensorGraphExecutor => true,
                   IRankedGraphExecutor ranked => ranked.RunsRanks(tp, _mesh.Stages),
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

        private void ValidateShardable(int degree)
        {
            if (degree <= 1) return;
            var problems = new List<string>();
            if (Config.NumHeads % degree != 0) problems.Add($"{Config.NumHeads} attention heads");
            if (Config.NumKVHeads % degree != 0) problems.Add($"{Config.NumKVHeads} KV heads");
            // Layers of their own KV head count (Gemma 4's global layers) split them too.
            foreach (int kv in _layerKvHeads.Distinct())
                if (kv != Config.NumKVHeads && kv % degree != 0) problems.Add($"{kv} KV heads of some layers");
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
            return LayerSplitPlan.CreateWeighted(bytes, stages, 0, 1, embedding, head, false, LayerCutAllowed);
        }

        /// <summary>A KV-shared layer attends its source's operands of the same step: no stage starts between them.</summary>
        protected override bool LayerCutAllowed(int layer)
        {
            for (int l = layer; l < Config.NumLayers; l++)
                if (SharesKv(l) && KvSourceOf(l) < layer) return false;
            return true;
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
                    int qRows = Config.NumHeads * KeyDimOf(l) * (GatedAttentionOutput ? 2 : 1);
                    if (!HasMixer(l)) { }   // a feed-forward layer (LayerHasMixer)
                    else if (_recurrentLayer[l])
                        ShardRecurrentLayer(l);
                    else if (_qkvFused[l])
                    {
                        ShardConcatenatedColumnParallel(p + "attn_qkv.weight",
                            qRows, KvHeadsOf(l) * KeyDimOf(l), KvHeadsOf(l) * ValueDimOf(l));
                        // The fused [q | k | v] bias (FuseProjectionBiases): each rank's blocks of the three, as its product's rows.
                        if (AttentionProjectionsHaveBiases)
                            ShardVectorSegments(p + "attn_qkv.bias", qRows, KvHeadsOf(l) * KeyDimOf(l), KvHeadsOf(l) * ValueDimOf(l));
                    }
                    else if (SharesKv(l))
                    {
                        columns.Add(p + "attn_q.weight");   // a KV-shared layer projects its queries only
                        if (AttentionProjectionsHaveBiases) ShardVectorSegments(p + "attn_q.bias", qRows);
                    }
                    else
                    {
                        columns.AddRange(new[] { p + "attn_q.weight", p + "attn_k.weight", p + "attn_v.weight" });
                        if (AttentionProjectionsHaveBiases)
                        {
                            ShardVectorSegments(p + "attn_q.bias", qRows);
                            ShardVectorSegments(p + "attn_k.bias", KvHeadsOf(l) * KeyDimOf(l));
                            ShardVectorSegments(p + "attn_v.bias", KvHeadsOf(l) * ValueDimOf(l));
                        }
                    }
                    if (IsAttentionLayer(l))
                        columns.AddRange(HeadParallelLayerWeights(l));   // per-head rows, in the query's head blocks
                    // Per-head attention sinks: a rank's own heads, as its queries.
                    if (IsAttentionLayer(l) && AttentionSinksWeight(l) is { } sinks)
                        ShardVectorSegments(sinks, Config.NumHeads);
                    if (IsAttentionLayer(l))
                        rows.Add(p + "attn_output.weight");
                    if (!HasFeedForward(l)) continue;   // a mixer layer (LayerHasFeedForward)
                    if (_gateUpFused[l])
                        ShardFusedGateUpColumnParallel(p + "ffn_gate_up.weight");
                    else
                        columns.AddRange(new[] { p + "ffn_gate.weight", p + "ffn_up.weight" });
                    rows.Add(p + "ffn_down.weight");
                    if (_rankGraphs && _moeLayer[l])
                    {
                        // Each routed expert and the shared expert split by their width, like a dense feed-forward:
                        // every rank runs every selected expert's slice. GGML's mul_mat_id reads one stacked tensor, so
                        // there each rank gets its own stack; other executors take the members one at a time.
                        // A fused [gate | up] stack (Gemma 4's ffn_gate_up_exps) splits each half by the ranks, so a rank's
                        // slice of an expert is [gate_r | up_r].
                        bool fusedExperts = _stackedExpertWeights.ContainsKey(p + "ffn_gate_up_exps.weight");
                        if (IsGgmlBackend)
                        {
                            if (fusedExperts)
                                ShardStackedExperts(p + "ffn_gate_up_exps.weight", columns: true, stage, halves: true);
                            else
                            {
                                // Experts without a gate (Nemotron-H's ReLU^2 experts) split their up stack alone.
                                if (_stackedExpertWeights.ContainsKey(p + "ffn_gate_exps.weight"))
                                    ShardStackedExperts(p + "ffn_gate_exps.weight", columns: true, stage);
                                ShardStackedExperts(p + "ffn_up_exps.weight", columns: true, stage);
                            }
                            ShardStackedExperts(p + "ffn_down_exps.weight", columns: false, stage);
                        }
                        else
                        {
                            if (fusedExperts)
                            {
                                foreach (string member in _stackedExpertMemberNames.Where(n => n.StartsWith(p + "ffn_gate_up_exps.", StringComparison.Ordinal)).ToArray())
                                    ShardFusedGateUpColumnParallel(member);
                                // The members by name: the prefix would take the per-expert output scales (ffn_down_exps.scale), read whole.
                                rows.AddRange(_stackedExpertMemberNames.Where(n => n.StartsWith(p + "ffn_down_exps.", StringComparison.Ordinal)));
                            }
                            else
                            {
                                // The members by name: the prefixes would take the per-expert scales (ffn_up_exps.scale) too.
                                columns.AddRange(_stackedExpertMemberNames.Where(n => n.StartsWith(p + "ffn_gate_exps.", StringComparison.Ordinal)
                                    || n.StartsWith(p + "ffn_up_exps.", StringComparison.Ordinal)));
                                rows.AddRange(_stackedExpertMemberNames.Where(n => n.StartsWith(p + "ffn_down_exps.", StringComparison.Ordinal)));
                            }
                        }
                        columns.AddRange(new[] { p + "ffn_gate_shexp.weight", p + "ffn_up_shexp.weight" });
                        rows.Add(p + "ffn_down_shexp.weight");
                        // Per-expert biases of the gate and up rows (GPT-OSS's ffn_{gate,up}_exps.bias, [E, ff]): each rank's slice
                        // of every expert's width, as its stacks'. The down bias stays whole: a family adds it on rank 0 only.
                        ShardExpertBiasColumns(p + "ffn_gate_exps.bias");
                        ShardExpertBiasColumns(p + "ffn_up_exps.bias");
                    }
                }
                ShardWeightsForTensorParallelism(columns, rows);
                if (stage == _mesh.Stages - 1) ShardHeadByVocab();
            }
        }

        /// <summary>
        /// A one-dimensional F32 weight whose values follow a column-parallel projection's output rows (a bias, the per-head
        /// attention sinks), the concatenation of <paramref name="segments"/>: rank r keeps its slice of each segment, in order,
        /// on its own device (no-op where the weight is absent or the group is not tensor-parallel).
        /// </summary>
        private void ShardVectorSegments(string name, params int[] segments)
        {
            if (!IsTensorParallel || !_weights.TryGetValue(name, out Tensor w)) return;
            int ranks = TpDegree, degree = GlobalTpDegree, offset = TpRankOffset;
            long total = 0;
            foreach (int segment in segments)
            {
                if (segment % degree != 0)
                    throw new NotSupportedException($"{Config.Architecture}: '{name}' ({segment} values) does not split over {degree} ranks.");
                total += segment;
            }
            if (w.ElementCount() != total)
                throw new InvalidOperationException($"{Config.Architecture}: '{name}' holds {w.ElementCount()} values, {total} expected.");
            float[] all = w.GetElementsAsFloat((int)total);
            var shards = new Tensor[ranks];
            for (int r = 0; r < ranks; r++)
            {
                var part = new float[total / degree];
                int at = 0, written = 0;
                foreach (int segment in segments)
                {
                    int width = segment / degree;
                    Array.Copy(all, at + (offset + r) * width, part, written, width);
                    written += width;
                    at += segment;
                }
                shards[r] = new Tensor(_tpGroup.GetAllocator(r), DType.Float32, part.Length);
                shards[r].SetElementsAsFloat(part);
                HostWritten(shards[r]);
            }
            _tpWeights[name] = shards;
            _weights.Remove(name);
            w.Dispose();
        }

        /// <summary>Per-expert F32 biases [E, ff] (ggml [ff, E]) of a column-parallel expert stack: rank r keeps columns
        /// [r ff / T, (r + 1) ff / T) of every expert's row, as its slice of the stack's output rows (no-op where absent).</summary>
        private void ShardExpertBiasColumns(string name)
        {
            if (!IsTensorParallel || !_weights.TryGetValue(name, out Tensor w) || w.DimensionCount != 2) return;
            int ranks = TpDegree, degree = GlobalTpDegree, offset = TpRankOffset;
            int experts = (int)w.Sizes[0], width = (int)w.Sizes[1];
            if (width % degree != 0)
                throw new NotSupportedException($"{Config.Architecture}: '{name}' ({width} values an expert) does not split over {degree} ranks.");
            float[] all = w.GetElementsAsFloat(experts * width);
            int slice = width / degree;
            var shards = new Tensor[ranks];
            for (int r = 0; r < ranks; r++)
            {
                var part = new float[experts * slice];
                for (int e = 0; e < experts; e++)
                    Array.Copy(all, e * width + (offset + r) * slice, part, e * slice, slice);
                shards[r] = new Tensor(_tpGroup.GetAllocator(r), DType.Float32, experts, slice);
                shards[r].SetElementsAsFloat(part);
                HostWritten(shards[r]);
            }
            _tpWeights[name] = shards;
            _weights.Remove(name);
            w.Dispose();
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
        /// same way. The whole head stays for the draft heads, the all-row logits of verify steps and the per-op path (on
        /// the host until one of them runs, <see cref="PrepareRankGraphResidency"/>); a tied head (the embedding) is not
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
        /// <param name="halves">A fused [gate | up] stack: each expert's two halves split apart, rank r's slice its gate rows
        /// then its up rows (output rows [r·h/T, (r+1)·h/T) of each half of h rows).</param>
        private unsafe void ShardStackedExperts(string name, bool columns, int stage, bool halves = false)
        {
            if (!_stackedExpertWeights.TryGetValue(name, out StackedExpertWeights w))
                throw new NotSupportedException($"{Config.Architecture}: per-rank graphs on GGML need '{name}' as one stacked tensor.");
            int degree = _mesh.TensorParallel, experts = w.NumExperts;
            long ne0 = w.PerExpertNe0, ne1 = w.PerExpertNe1;
            long rowBytes = NativeDequant.RowSize(w.GgmlType, ne0), perExpert = rowBytes * ne1;
            var type = (GgmlTensorType)w.GgmlType;
            long blockSize = GgufFile.GetBlockSize(type), typeSize = GgufFile.GetTypeSize(type);
            if (columns ? ne1 % (halves ? 2 * degree : degree) != 0 : (ne0 / blockSize) % degree != 0)
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
                    if (columns && halves)
                    {
                        long half = ne1 / 2, slice = shardNe1 / 2;
                        Buffer.MemoryCopy(from + rank * slice * rowBytes, to, slice * rowBytes, slice * rowBytes);
                        Buffer.MemoryCopy(from + (half + rank * slice) * rowBytes, to + slice * rowBytes, slice * rowBytes, slice * rowBytes);
                    }
                    else if (columns)
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
            if (_mesh.TensorParallel == 1)
            {
                // One device, or whole layers: the single-device preload places each weight on
                // its layer's device (PreloadRankForWeight / AllocatorForWeight).
                PrepareCudaQuantizedWeightsForInference();
                return;
            }
            if (_rankGraphs)
                PrepareRankGraphResidency();
            else if (_mesh.Stages == 1 && LayerSplitGroup == null)
                PrepareCudaQuantizedWeightsForInferenceTP();
            else
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
        /// Per-rank graphs: each shard onto its rank's device, and every replicated weight onto every rank of the stage that
        /// reads it (each rank's executor finds a weight under its own device); the head stays on the last stage's rank 0,
        /// which runs it. Host copies of the replicated weights are kept (the per-op embedding and sampling read them); the
        /// shards' are released.
        /// </summary>
        /// <remarks>
        /// Where every rank holds its vocabulary slice of the head (<see cref="ShardHeadByVocab"/>), the whole head is not
        /// preloaded: the slices score every step's last rows. Its other readers (the all-row logits of a speculative
        /// verify step, a draft model's head, the per-op path) bind it on rank 0 from the kept host copy when they first
        /// run, so they pay its device copy only when they are used. A draft layer in the model reads it every draft step,
        /// so with one the whole head stays resident.
        /// </remarks>
        private void PrepareRankGraphResidency()
        {
            bool ggml = _backend is BackendType.GgmlCuda or BackendType.GgmlVulkan;
            if (_backend != BackendType.Cuda && !ggml) return;   // host-memory devices (simulated groups): nothing to upload
            if (ggml) EnsureQuantBackendAvailable();
            int ranks = _mesh.TensorParallel, devices = _mesh.DeviceCount;
            long[] bytes = new long[devices];
            var resident = new HashSet<QuantizedWeight>[devices];
            for (int d = 0; d < devices; d++) resident[d] = new HashSet<QuantizedWeight>(ReferenceEqualityComparer.Instance);
            void PreloadShards(int stage, int r)
            {
                int device = _mesh.GlobalDevice(stage, r);
                foreach (var kv in _tpQuantWeights)
                {
                    QuantizedWeight shard = kv.Value[r];
                    if (StageOfWeight(kv.Key) != stage || !shard.HasHostData || !ShouldPreloadCudaQuantWeightToDevice(kv.Key)) continue;
                    if (Preload(shard, _mesh.Allocator(stage, r), ggml))
                    {
                        bytes[device] += shard.RawBytes;
                        resident[device].Add(shard);
                    }
                }
            }
            for (int stage = 0; stage < _mesh.Stages; stage++)
            {
                int s = stage;
                // A ggml preload lands on the calling thread's active rank: one thread, device by device.
                if (ggml)
                    for (int r = 0; r < ranks; r++)
                    {
                        int device = _mesh.GlobalDevice(s, r);
                        GgmlBasicOps.SetActiveRank(device);
                        PreloadShards(s, r);
                        // The rank's expert stacks (each its own copy, read whole by its graph's mul_mat_id).
                        foreach (var ((_, holder), stack) in _rankStacks)
                            if (holder == device && stack.Stack.HasHostData && Preload(stack.Stack, null, ggml: true))
                            {
                                bytes[device] += stack.Bytes;
                                stack.Stack.ReleaseHostData();
                            }
                    }
                else
                    _mesh.RunPerRank(s, r => PreloadShards(s, r));
            }
            bool wholeHeadOnHost = WholeHeadStaysOnHost;
            foreach (var kv in _quantWeights)
            {
                QuantizedWeight weight = kv.Value;
                if (!weight.HasHostData || !ShouldPreloadCudaQuantWeightToDevice(kv.Key) || kv.Key == "token_embd.weight") continue;
                if (wholeHeadOnHost && kv.Key == "output.weight") continue;
                int stage = StageOfWeight(kv.Key), holders = IsHeadWeight(kv.Key) ? 1 : ranks;
                for (int r = 0; r < holders; r++)
                {
                    int device = _mesh.GlobalDevice(stage, r);
                    if (ggml) GgmlBasicOps.SetActiveRank(device);
                    if (Preload(weight, _mesh.Allocator(stage, r), ggml)) bytes[device] += weight.RawBytes;
                }
            }
            if (ggml) GgmlBasicOps.SetActiveRank(0);
            // Only a shard its rank holds gives up its host copy (another is read from the host).
            foreach (var kv in _tpQuantWeights)
                for (int r = 0; r < kv.Value.Length; r++)
                    if (resident[_mesh.GlobalDevice(StageOfWeight(kv.Key), r)].Contains(kv.Value[r]) && !ShouldRetainCudaHostQuantWeight(kv.Key))
                        kv.Value[r].ReleaseHostData();
            for (int d = 0; d < bytes.Length; d++)
                Console.WriteLine(_mesh.Stages == 1
                    ? $"  Rank {d}: {bytes[d] / 1024 / 1024} MB of quantized weights resident."
                    : $"  Device {d} (stage {d / ranks}, rank {d % ranks}): {bytes[d] / 1024 / 1024} MB of quantized weights resident.");
        }

        /// <summary>Per-rank graphs leave the whole untied head on the host (see <see cref="PrepareRankGraphResidency"/>).</summary>
        internal bool WholeHeadStaysOnHost => _rankGraphs && _vocabHeadRows != null && DraftLayerCount <= 0;

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
