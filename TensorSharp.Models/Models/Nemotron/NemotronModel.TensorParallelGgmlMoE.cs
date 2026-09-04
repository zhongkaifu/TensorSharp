// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
//
// ============================================================================
// NemotronModel.TensorParallelGgmlMoE.cs
//
// Expert-parallel MoE for the Nemotron tensor-parallel path on the GGML backends.
//
// The Megatron split this replaces slices INSIDE every expert (up column-parallel,
// down row-parallel). That leaves each rank holding a piece of all 128 experts, so
// a layer can no longer be one batched dispatch and degenerates into a per-expert
// loop: at 24 MoE layers x 2 ranks x up-to-128 active experts x 2 projections that
// is ~12k tiny matmuls per prefill chunk, each wrapped in a host gather and a host
// scatter-add of every routed row.
//
// Whole experts partition cleanly instead. The expert index is the stacked tensor's
// OUTER dimension, so rank r's share is a contiguous byte range - a zero-copy view -
// and each rank runs the same batched ggml_mul_mat_id kernel the single-GPU path
// uses. Each rank sums only the experts it owns, so the AllReduce the block already
// performs is exactly the right recombination.
//
// Nemotron's routed experts are ReLU-squared single-projection (up -> relu(x)^2 ->
// down, no gate), which GgmlBasicOps.MoEFFNPrefill expresses by passing the single
// projection as gateData and leaving upData null.
//
// The shared expert is deliberately NOT touched here: it stays replicated on rank 0
// and is added by NemotronMoEBlockTP after the AllReduce, exactly as before.
// ============================================================================
using System;
using System.Diagnostics;
using TensorSharp;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    public partial class NemotronModel
    {
        // Per-rank slices of the stacked expert weights, [layer][local rank].
        private StackedExpertWeights[][] _tpStackedUp;
        private StackedExpertWeights[][] _tpStackedDown;
        private int _tpExpertsPerRank;

        /// <summary>
        /// True when the routed experts are partitioned by whole expert rather than
        /// sliced inside each expert.
        /// </summary>
        private bool UsesExpertParallelMoE => _tpStackedUp != null;

        /// <summary>
        /// Whether this model/backend combination can take the expert-parallel path.
        /// Deliberately shape-based rather than reading <c>_moeLayerInfo</c>: this
        /// runs during sharding, which happens BEFORE InitLayerInfo populates that.
        /// </summary>
        private bool CanUseNemotronExpertParallelMoE()
        {
            int tp = GlobalTpDegree;
            if (!IsGgmlBackend || _numExperts <= 0 || _numExpertsUsed <= 0 || tp <= 1)
                return false;
            if ((_numExperts % tp) != 0)
                return false;
            // A route owned by another rank is neutralised by pointing it at an
            // unused LOCAL expert id, which requires there to be a spare one.
            if (_numExperts / tp < _numExpertsUsed)
                return false;
            if (_layerTypes == null || _stackedExpertWeights == null)
                return false;

            int hiddenSize = Config.HiddenSize;
            bool sawLayer = false;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_layerTypes[l] != LayerType.FFN)
                    continue;
                if (!_stackedExpertWeights.TryGetValue($"blk.{l}.ffn_up_exps.weight", out var up) || up == null)
                    return false;
                if (!_stackedExpertWeights.TryGetValue($"blk.{l}.ffn_down_exps.weight", out var down) || down == null)
                    return false;
                if (up.NumExperts != _numExperts || down.NumExperts != _numExperts)
                    return false;
                // A latent-space MoE feeds the experts a projected vector that
                // NemotronMoEBlockTP computes ONCE on rank 0, which the per-rank kernel
                // below could not consume. Exclude it by NAME rather than inferring it
                // from the expert width, so a checkpoint whose latentDim happens to
                // equal hiddenSize cannot slip through.
                if (_quantWeights.ContainsKey($"blk.{l}.ffn_latent_in.weight")
                    || _weights.ContainsKey($"blk.{l}.ffn_latent_in.weight"))
                    return false;
                if (up.PerExpertNe0 != hiddenSize || down.PerExpertNe1 != hiddenSize)
                    return false;
                if (up.PerExpertNe1 != down.PerExpertNe0)
                    return false;
                sawLayer = true;
            }
            return sawLayer;
        }

        /// <summary>
        /// Build the per-rank whole-expert slices. Returns false when the model did
        /// not expose usable stacked expert weights, leaving the caller on the
        /// per-expert sharding path.
        ///
        /// All-or-nothing on purpose: a per-layer bail-out would leave some layers
        /// with no per-expert shards AND no stacked shard, and the per-expert
        /// fallback would then throw looking up weights that were never created.
        /// </summary>
        private bool BuildNemotronExpertParallelShards()
        {
            if (!CanUseNemotronExpertParallelMoE())
                return false;

            int globalTp = GlobalTpDegree;
            int localTp = TpDegree;
            int rankOffset = TpRankOffset;
            int perRank = _numExperts / globalTp;
            int n = Config.NumLayers;

            var up = new StackedExpertWeights[n][];
            var down = new StackedExpertWeights[n][];

            for (int layer = 0; layer < n; layer++)
            {
                if (_layerTypes[layer] != LayerType.FFN)
                    continue;

                var srcUp = _stackedExpertWeights[$"blk.{layer}.ffn_up_exps.weight"];
                var srcDown = _stackedExpertWeights[$"blk.{layer}.ffn_down_exps.weight"];

                up[layer] = new StackedExpertWeights[localTp];
                down[layer] = new StackedExpertWeights[localTp];
                for (int lr = 0; lr < localTp; lr++)
                {
                    int firstExpert = (rankOffset + lr) * perRank;
                    up[layer][lr] = SliceNemotronStackedExperts(srcUp, firstExpert, perRank);
                    down[layer][lr] = SliceNemotronStackedExperts(srcDown, firstExpert, perRank);
                }
            }

            _tpStackedUp = up;
            _tpStackedDown = down;
            _tpExpertsPerRank = perRank;

            Console.WriteLine(
                $"  Nemotron MoE: expert-parallel across {globalTp} GPU(s), {perRank} of {_numExperts} experts per GPU "
                + "(one batched ggml_mul_mat_id dispatch per projection per layer).");
            return true;
        }

        /// <summary>
        /// Zero-copy view of <paramref name="count"/> consecutive experts. The expert
        /// index is the stacked tensor's outer dimension, so this is a byte offset -
        /// no copy, and each rank's device cache holds only its own slice.
        /// </summary>
        private static StackedExpertWeights SliceNemotronStackedExperts(
            StackedExpertWeights src, int firstExpert, int count)
        {
            long perExpertBytes = src.PerExpertRawBytes;
            return new StackedExpertWeights(
                new IntPtr(src.Data.ToInt64() + firstExpert * perExpertBytes),
                src.GgmlType,
                src.PerExpertNe0,
                src.PerExpertNe1,
                count,
                perExpertBytes * count,
                isExternalView: true,
                ownerToken: src,
                ownedBuffer: IntPtr.Zero);
        }

        /// <summary>
        /// Make each rank's expert slice device-resident at load time. Without this
        /// the first forward pays the whole upload, which looks like a hang.
        /// </summary>
        protected override void PreloadGgmlTpAuxiliaryWeightsForRank(int rank, long[] bytesPerRank, int[] countPerRank)
        {
            // Runs inside the per-rank preload fan-out: the calling thread is already
            // pinned to this rank's GPU and every rank uploads concurrently.
            if (!UsesExpertParallelMoE)
                return;

            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                if (_layerTypes[layer] != LayerType.FFN || _tpStackedUp[layer] == null)
                    continue;
                // --n-cpu-moe: this layer's experts are multiplied on the host out of
                // the GGUF mmap. Not uploading them IS the VRAM saving.
                if (MoeCpuOffloadConfig.IsLayerOnCpu(layer))
                    continue;
                PreloadNemotronStackedShard(_tpStackedUp[layer][rank], bytesPerRank, countPerRank, rank);
                PreloadNemotronStackedShard(_tpStackedDown[layer][rank], bytesPerRank, countPerRank, rank);
            }
        }

        private static void PreloadNemotronStackedShard(
            StackedExpertWeights w, long[] bytesPerRank, int[] countPerRank, int rank)
        {
            if (w == null) return;
            // The MoE kernel looks the buffer up by its data pointer, so the cache
            // key must be that same pointer.
            if (GgmlBasicOps.PreloadQuantizedWeight(
                    w.Data, w.Data, w.GgmlType, w.PerExpertNe0, w.PerExpertNe1 * w.NumExperts, w.TotalRawBytes))
            {
                bytesPerRank[rank] += w.TotalRawBytes;
                countPerRank[rank]++;
            }
        }

        /// <summary>
        /// Under expert parallelism the per-expert weights are only ever read through
        /// the stacked per-rank slices above. They are still present in
        /// <c>_quantWeights</c> (sharding skipped them), and the replicated-weight
        /// preload would otherwise push a full unsharded copy of every expert - the
        /// whole ~30 GB of them - onto rank 0.
        /// </summary>
        protected override bool ShouldPreloadCudaQuantWeightToDevice(string weightName)
            => !(UsesExpertParallelMoE && _stackedExpertMemberNames.Contains(weightName))
               && base.ShouldPreloadCudaQuantWeightToDevice(weightName);

        /// <summary>
        /// The same weights, but for the load-time ACCOUNTING rather than the upload.
        /// Without this the per-expert entries left in <c>_quantWeights</c> are tallied
        /// as "replicated on rank 0" - ~30 GB of tensors that are not resident there at
        /// all - and startup then warns that TP sharded almost nothing and the run is
        /// probably slower than one GPU. Both statements would be false, and that
        /// warning is worth keeping honest so it still means something when a real
        /// regression puts it back.
        /// </summary>
        protected override bool IsSupersededByTpShard(string weightName)
            => (UsesExpertParallelMoE && _stackedExpertMemberNames.Contains(weightName))
               || base.IsSupersededByTpShard(weightName);

        /// <summary>
        /// Expert parallelism gives every rank ONE batched dispatch per projection per
        /// layer, so the MoE block is no longer the slow per-expert loop that the
        /// lightweight startup warmup exists to dodge.
        /// </summary>
        protected override bool MoEUnderTpIsSlow => IsTensorParallel && !UsesExpertParallelMoE;

        /// <summary>
        /// The routed experts are the bulk of this model and they are split by whole
        /// expert, not through <c>_tpQuantWeights</c>. Report those bytes so the
        /// "TP sharded almost nothing" warning reflects what was actually split.
        /// </summary>
        protected override long AdditionalTpShardedBytes
        {
            get
            {
                if (!UsesExpertParallelMoE)
                    return 0;
                long total = 0;
                for (int layer = 0; layer < Config.NumLayers; layer++)
                {
                    if (_tpStackedUp[layer] == null)
                        continue;
                    for (int r = 0; r < TpDegree; r++)
                    {
                        total += _tpStackedUp[layer][r]?.TotalRawBytes ?? 0;
                        total += _tpStackedDown[layer][r]?.TotalRawBytes ?? 0;
                    }
                }
                return total;
            }
        }

        /// <summary>
        /// Run one MoE layer expert-parallel: each rank dispatches the batched
        /// <c>ggml_mul_mat_id</c> kernel over the experts it owns. Every rank's result
        /// is a PARTIAL that the caller's AllReduce completes. Returns null when this
        /// model is not on the expert-parallel path.
        /// </summary>
        private Tensor[] TryNemotronMoEExpertParallel(
            int layer, int seqLen, int expertOutDim, Tensor[] normed,
            int[] selectedExperts, float[] routeWeightsAll)
        {
            if (!UsesExpertParallelMoE)
                return null;
            var upShards = _tpStackedUp[layer];
            var downShards = _tpStackedDown[layer];
            if (upShards == null || downShards == null)
                return null;

            int tp = TpDegree;
            int rankOffset = TpRankOffset;
            int hiddenSize = Config.HiddenSize;
            int nUsed = _numExpertsUsed;
            int perRank = _tpExpertsPerRank;
            int intermediate = (int)upShards[0].PerExpertNe1;

            // BuildNemotronExpertParallelShards already proved these at load time, and
            // it removed the per-expert shards the fallback would need. So a mismatch
            // here is a bug, not a shape this model can serve another way - say so
            // instead of failing later with a confusing missing-weight lookup.
            if (expertOutDim != hiddenSize)
            {
                throw new InvalidOperationException(
                    $"Nemotron expert-parallel MoE expects the routed output to be hidden-sized "
                    + $"({hiddenSize}), got {expertOutDim}. The per-expert shards were not built, "
                    + "so there is no fallback.");
            }

            // Per-rank dense route table, built on the calling thread: the rank
            // workers below must not race on shared scratch.
            int totalRoutes = seqLen * nUsed;
            var localExperts = new int[tp][];
            var localWeights = new float[tp][];
            for (int r = 0; r < tp; r++)
            {
                int first = (rankOffset + r) * perRank;
                int last = first + perRank;
                var ids = new int[totalRoutes];
                var wts = new float[totalRoutes];
                var taken = new bool[perRank];

                for (int s = 0; s < seqLen; s++)
                {
                    int baseIdx = s * nUsed;
                    Array.Clear(taken, 0, perRank);

                    // Own routes first, so the filler pass can see which local experts
                    // this token already uses.
                    for (int k = 0; k < nUsed; k++)
                    {
                        int i = baseIdx + k;
                        int e = selectedExperts[i];
                        if (e >= first && e < last)
                        {
                            int local = e - first;
                            ids[i] = local;
                            taken[local] = true;
                            wts[i] = routeWeightsAll[i];
                        }
                        else
                        {
                            ids[i] = -1;
                            wts[i] = 0f;
                        }
                    }

                    // A route owned by another rank still occupies a slot in the dense
                    // table; it is neutralised with weight 0. It must still get a
                    // DISTINCT local expert id: real top-k routing never repeats an
                    // expert within a token and the batched kernel's per-expert
                    // gather/scatter relies on that: the id list is collapsed per token
                    // while the slot count is not, so a duplicate MISPLACES rows rather
                    // than being harmlessly ignored. perRank >= nUsed (checked at load)
                    // guarantees a free id exists.
                    int probe = 0;
                    for (int k = 0; k < nUsed; k++)
                    {
                        int i = baseIdx + k;
                        if (ids[i] >= 0) continue;
                        while (probe < perRank && taken[probe]) probe++;
                        if (probe < perRank) { taken[probe] = true; ids[i] = probe; }
                        else ids[i] = 0;
                    }
                }

                localExperts[r] = ids;
                localWeights[r] = wts;
            }

            var results = new Tensor[tp];
            Exception failure = null;

            long t0 = Stopwatch.GetTimestamp();
            _tpGroup.RunPerRank(r =>
            {
                var output = new Tensor(_tpGroup.GetAllocator(r), DType.Float32, seqLen, hiddenSize);
                var u = upShards[r];
                var d = downShards[r];
                try
                {
                    GgmlBasicOps.MoEFFNPrefill(
                        normed[r], output,
                        seqLen, hiddenSize, intermediate, perRank, nUsed,
                        localExperts[r], localWeights[r],
                        // ReLU^2 is a SINGLE projection: it goes in the gate slot and
                        // the up slot stays null.
                        u.Data, u.GgmlType, u.PerExpertNe0, u.PerExpertNe1, u.TotalRawBytes,
                        IntPtr.Zero, 0, 0, 0, 0,
                        d.Data, d.GgmlType, d.PerExpertNe0, d.PerExpertNe1, d.TotalRawBytes,
                        gateBias: null, upBias: null, downBias: null,
                        activation: GgmlBasicOps.MoEActivation.ReluSquared,
                        runOnCpu: MoeCpuOffloadConfig.IsLayerOnCpu(layer));
                    InvalidateTensorDeviceCache(output);
                    results[r] = output;
                }
                catch (Exception ex)
                {
                    output.Dispose();
                    failure ??= ex;
                }
            });
            _linearTicks += Stopwatch.GetTimestamp() - t0;

            if (failure != null)
            {
                for (int r = 0; r < tp; r++)
                {
                    results[r]?.Dispose();
                    results[r] = null;
                }
                throw new InvalidOperationException(
                    "Nemotron expert-parallel MoE dispatch failed and the per-expert shards "
                    + "were not built, so there is no fallback: " + failure.Message, failure);
            }

            return results;
        }
    }
}
