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
            // A distributed TP group shards by its GLOBAL degree; a stage group by its own.
            int shardDegree = tp == 1 ? 1 : stages == 1 ? GlobalTpDegree : tp;
            _shardDegree = shardDegree;
            _shardRankOffset = tp > 1 && stages == 1 ? TpRankOffset : 0;
            ValidateShardable(shardDegree);
            _localHeads = Config.NumHeads / shardDegree;
            _localKvHeads = Config.NumKVHeads / shardDegree;

            if (LayerSplitGroup != null || (stages > 1 && tp == 1))
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
            }
            else
            {
                _mesh = new DeviceMesh(1, 1, _ => _allocator, null, null);
            }
        }

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

        private bool IsWholeLayerSplit => _mesh != null && _mesh.TensorParallel == 1 && LayerPlacement != null;

        private bool IsHeadWeight(string name) =>
            name.StartsWith("output", StringComparison.Ordinal) ||
            (name == "token_embd.weight" && LayerPlacement != null && !LayerPlacement.IsFirst);

        /// <summary>A multi-node pipeline loads only this node's layers (plus the embedding on
        /// the first stage and the head on the last).</summary>
        protected override bool ShouldLoadWeight(GgufTensorInfo info)
        {
            if (LayerPlacement == null) return true;
            if (LayerSplitPlan.TryGetLayer(info.Name, out int layer)) return LayerPlacement.Owns(layer);
            if (info.Name == "token_embd.weight")
                return LayerPlacement.IsFirst || (LayerPlacement.IsLast && !_gguf.Tensors.ContainsKey("output.weight"));
            if (info.Name.StartsWith("output", StringComparison.Ordinal)) return LayerPlacement.IsLast;
            return true;
        }

        protected override int PreloadRankForWeight(string name)
        {
            if (!IsWholeLayerSplit) return 0;
            if (LayerSplitPlan.TryGetLayer(name, out int layer)) return LayerPlacement.LocalRank(layer);
            return IsHeadWeight(name) ? LayerSplitDegree - 1 : 0;
        }

        protected override IAllocator AllocatorForWeight(string name)
        {
            if (_mesh == null || _mesh.IsSingleDevice) return base.AllocatorForWeight(name);
            if (IsWholeLayerSplit)
            {
                if (LayerSplitPlan.TryGetLayer(name, out int owned)) return LayerAllocator(owned);
                return LayerAllocator(IsHeadWeight(name) ? LayerPlacement.EndLayer - 1 : LayerPlacement.FirstLayer);
            }
            return _mesh.Allocator(StageOfWeight(name), 0);
        }

        protected override bool ShouldRetainCudaHostQuantWeight(string name)
            => (IsWholeLayerSplit && !LayerSplitPlan.TryGetLayer(name, out _)) || base.ShouldRetainCudaHostQuantWeight(name);

        /// <summary>The stage holding <paramref name="name"/> on a local mesh.</summary>
        private int StageOfWeight(string name)
        {
            if (LayerSplitPlan.TryGetLayer(name, out int layer) && layer < Config.NumLayers) return _mesh.StageOf(layer);
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
                    if (_mesh.StageOf(l) != stage) continue;
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
                }
                ShardWeightsForTensorParallelism(columns, rows);
            }
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
            if (_mesh.Stages == 1)
            {
                PrepareCudaQuantizedWeightsForInferenceTP();
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
