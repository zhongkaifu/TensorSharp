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
// Gemma4Model.TensorParallel.cs
//
// Tensor-parallel forward pass for Gemma 4, using Megatron-LM column/row
// parallelism with:
//   - Fused QKV projection (attn_qkv.weight)
//   - Per-layer head dimensions (local SWA vs global)
//   - Per-layer KV head counts
//   - MoE layers: tensor-parallel experts (1/tp slice of every expert)
//   - Dense + MoE FFN in the same MoE layer
//   - Shared KV layers (KV donor map)
// ============================================================================
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    public partial class Gemma4Model
    {
        // Per-GPU KV caches: [layer][rank]. Only populated when TP is active.
        private Tensor[][] _tpKvCacheK;
        private Tensor[][] _tpKvCacheV;
        private int _tpKvCacheCapacity;

        // ====================================================================
        // TP constraint validation
        // ====================================================================

        private void ValidateGemma4TpConstraints()
        {
            int tp = GlobalTpDegree;
            var errors = new List<string>();

            if (Config.NumHeads % tp != 0)
                errors.Add($"Attention heads ({Config.NumHeads}) not divisible by global TP degree ({tp})");
            if (Config.NumKVHeads % tp != 0)
                errors.Add($"Local KV heads ({Config.NumKVHeads}) not divisible by global TP degree ({tp})");
            if (_numGlobalKVHeads % tp != 0)
                errors.Add($"Global KV heads ({_numGlobalKVHeads}) not divisible by global TP degree ({tp})");

            // Check per-layer head dims are divisible
            for (int l = 0; l < Config.NumLayers; l++)
            {
                int kvHeads = KVHeadsForLayer(l);
                if (kvHeads % tp != 0)
                {
                    errors.Add($"Layer {l} KV heads ({kvHeads}) not divisible by global TP degree ({tp})");
                    break;
                }
            }

            // Expert-parallel MoE (GGML) partitions whole experts, so it needs the
            // expert *count* to divide evenly and places no constraint on the
            // expert FFN width. Only the per-expert slicing path needs that.
            bool expertParallel = IsGgmlBackend && _numExperts > 0 && (_numExperts % tp) == 0
                && _layerStackedGate != null && _layerStackedDown != null;

            if (_numExperts > 0 && !expertParallel)
            {
                // Check expert FFN dims from weight shapes
                for (int l = 0; l < Config.NumLayers; l++)
                {
                    if (!HasMoE(l)) continue;
                    string gateKey = $"blk.{l}.ffn_gate_exps.0.weight";
                    if (_quantWeights.TryGetValue(gateKey, out var qw) && qw.Ne1 % tp != 0)
                    {
                        errors.Add($"Layer {l} expert FFN ({qw.Ne1}) not divisible by global TP degree ({tp})");
                        break;
                    }
                    if (_weights.TryGetValue(gateKey, out var w) && w.Sizes[0] % tp != 0)
                    {
                        errors.Add($"Layer {l} expert FFN ({w.Sizes[0]}) not divisible by global TP degree ({tp})");
                        break;
                    }
                }
            }

            // Tensor parallelism runs on the multi-GPU backends: direct CUDA and
            // the GGML CUDA/Vulkan backends (one ggml backend per GPU).
            if (_backend is not (BackendType.Cuda or BackendType.GgmlCuda or BackendType.GgmlVulkan))
                errors.Add($"TP requires a multi-GPU backend (cuda, ggml-cuda, ggml-vulkan), got {_backend}");

            if (errors.Count > 0)
                throw new InvalidOperationException(
                    $"Gemma4 TP validation failed:\n  " + string.Join("\n  ", errors));

            Console.WriteLine($"  TP constraints validated: globalTp={tp}, localTp={TpDegree}, " +
                $"Heads={Config.NumHeads}, KVHeads local={Config.NumKVHeads}/global={_numGlobalKVHeads}");
        }

        // ====================================================================
        // Weight sharding
        // ====================================================================

        private void ShardGemma4WeightsForTP()
        {
            // Attention + FFN row-parallel weights. attn_q.weight (KV-sharing
            // layers, which only project Q) is a single column segment, so the
            // generic contiguous column split is correct for it.
            // attn_k/attn_v are also column-parallel: when the fused attn_qkv
            // cannot be created (Q/K/V have different output dims, e.g. E4B
            // Q=2048 vs K/V=512), the TP forward runs three separate column-
            // parallel projections and concatenates the results.
            // The fused attn_qkv ([Q|K|V]) and ffn_gate_up ([gate|up]) are
            // handled below with segment-aware sharding — a contiguous split
            // would mix whole segments across ranks and corrupt the per-rank
            // [Q_r|K_r|V_r] / [gate_r|up_r] layout the forward pass expects.
            ShardWeightsForTensorParallelism(
                columnParallelPatterns: new[] { "attn_q.weight", "attn_k.weight", "attn_v.weight" },
                rowParallelPatterns: new[] { "attn_output.weight", "ffn_down.weight" });

            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                int hd = HeadDimForLayer(layer);
                int kvHeads = KVHeadsForLayer(layer);
                // Non-shared layers carry a fused [Q|K|V]; shared layers use
                // attn_q (handled above) and have no attn_qkv (no-op here).
                ShardConcatenatedColumnParallel($"blk.{layer}.attn_qkv.weight",
                    Config.NumHeads * hd,  // Q
                    kvHeads * hd,          // K
                    kvHeads * hd);         // V

                ShardFusedGateUpColumnParallel($"blk.{layer}.ffn_gate_up.weight");
            }

            // MoE expert weights: tensor-parallel experts
            if (_numExperts > 0)
                ShardGemma4MoeWeightsForTP();

            Console.WriteLine($"  Gemma4 TP weight sharding complete ({TpDegree} GPUs).");
        }

        // Per-rank slices of the stacked expert weights, [layer][rank]. Populated
        // when the expert-parallel path is in use (see BuildGemma4ExpertParallelShards).
        private StackedExpertWeights[][] _tpStackedGate;
        private StackedExpertWeights[][] _tpStackedUp;
        private StackedExpertWeights[][] _tpStackedDown;
        private int _tpExpertsPerRank;
        private bool _loggedExpertParallelShapes;

        /// <summary>
        /// True when the MoE layers are split by whole expert rather than by
        /// slicing inside each expert.
        /// </summary>
        private bool UsesExpertParallelMoE => _tpStackedGate != null;

        /// <summary>
        /// Make each rank's stacked expert slice device-resident at load time.
        /// Without this the first forward pays the whole upload — measured at
        /// ~51 s on Gemma-4-26B — and it looks like a hang.
        /// </summary>
        protected override void PreloadGgmlTpAuxiliaryWeightsForRank(int rank, long[] bytesPerRank, int[] countPerRank)
        {
            if (UsesTensorSlicedMoE)
            {
                PreloadGemma4TensorSlicedExpertsForRank(rank, bytesPerRank, countPerRank);
                return;
            }
            if (!UsesExpertParallelMoE)
                return;

            // The calling thread is pinned to this rank's GPU by the RunPerRank
            // fan-out, so each rank streams its own expert slices concurrently.
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                if (!HasMoE(layer)) continue;
                // --n-cpu-moe: this layer's experts are multiplied on the host
                // out of the GGUF mmap. Not uploading them IS the VRAM saving.
                if (MoeCpuOffloadConfig.IsLayerOnCpu(layer)) continue;
                PreloadStackedShard(_tpStackedGate[layer]?[rank], bytesPerRank, countPerRank, rank);
                PreloadStackedShard(_tpStackedUp?[layer]?[rank], bytesPerRank, countPerRank, rank);
                PreloadStackedShard(_tpStackedDown[layer]?[rank], bytesPerRank, countPerRank, rank);
            }
        }

        private static void PreloadStackedShard(StackedExpertWeights w, long[] bytesPerRank, int[] countPerRank, int rank)
        {
            if (w == null) return;
            // The MoE kernel looks the buffer up by the data pointer, so the
            // cache key must be that same pointer.
            if (GgmlBasicOps.PreloadQuantizedWeight(
                    w.Data, w.Data, w.GgmlType, w.PerExpertNe0, w.PerExpertNe1 * w.NumExperts, w.TotalRawBytes))
            {
                bytesPerRank[rank] += w.TotalRawBytes;
                countPerRank[rank]++;
            }
        }

        /// <summary>
        /// Split the MoE layers by *expert* instead of by tensor.
        ///
        /// Slicing inside each expert (column-parallel gate/up, row-parallel
        /// down) forces the per-token expert loop: every rank holds a piece of
        /// every expert, so the layer cannot be expressed as one
        /// <c>ggml_mul_mat_id</c> dispatch. On a 2048-token prefill of
        /// Gemma-4-26B that is ~3 million tiny matmuls.
        ///
        /// Whole experts partition cleanly instead. The stacked expert tensor
        /// has the expert index as its outer dimension, so rank r's share is a
        /// contiguous byte range — a zero-copy view — and each rank can run the
        /// same batched kernel the single-GPU path uses, one dispatch per layer.
        /// Each rank sums only the experts it owns, so the existing AllReduce
        /// over the layer output is exactly the right recombination.
        ///
        /// Returns false when the model did not expose stacked expert weights,
        /// leaving the caller on the per-expert sharding path.
        /// </summary>
        private bool BuildGemma4ExpertParallelShards()
        {
            // The batched dispatch is GgmlBasicOps.MoEFFNPrefill, so this path is
            // for the GGML backends only; direct CUDA keeps its per-expert
            // sharding and its own on-device MoE kernels.
            if (!IsGgmlBackend)
                return false;

            int tp = GlobalTpDegree;
            if (_numExperts <= 0 || tp <= 1 || (_numExperts % tp) != 0)
                return false;
            if (_layerStackedGate == null || _layerStackedDown == null)
                return false;

            int localTp = TpDegree;
            int rankOffset = TpRankOffset;
            int perRank = _numExperts / tp;
            int n = Config.NumLayers;

            var gate = new StackedExpertWeights[n][];
            var up = new StackedExpertWeights[n][];
            var down = new StackedExpertWeights[n][];

            for (int layer = 0; layer < n; layer++)
            {
                if (!HasMoE(layer))
                    continue;
                var g = _layerStackedGate[layer];
                var d = _layerStackedDown[layer];
                if (g == null || d == null)
                    return false;
                var u = _layerStackedUp?[layer];

                gate[layer] = new StackedExpertWeights[localTp];
                down[layer] = new StackedExpertWeights[localTp];
                if (u != null) up[layer] = new StackedExpertWeights[localTp];

                for (int lr = 0; lr < localTp; lr++)
                {
                    int firstExpert = (rankOffset + lr) * perRank;
                    gate[layer][lr] = SliceExperts(g, firstExpert, perRank);
                    down[layer][lr] = SliceExperts(d, firstExpert, perRank);
                    if (u != null) up[layer][lr] = SliceExperts(u, firstExpert, perRank);
                }
            }

            _tpStackedGate = gate;
            _tpStackedUp = _layerStackedUp != null ? up : null;
            _tpStackedDown = down;
            _tpExpertsPerRank = perRank;

            Console.WriteLine(
                $"  Gemma4 MoE: expert-parallel across {tp} GPU(s), {perRank} of {_numExperts} experts per GPU " +
                "(batched ggml_mul_mat_id dispatch per layer).");
            return true;
        }

        /// <summary>
        /// Zero-copy view of <paramref name="count"/> consecutive experts. The
        /// expert index is the stacked tensor's outer dimension, so this is a
        /// byte offset — no copy, and each rank's device cache holds only its
        /// own slice.
        /// </summary>
        private static StackedExpertWeights SliceExperts(StackedExpertWeights src, int firstExpert, int count)
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

        private void ShardGemma4MoeWeightsForTP()
        {
            // First choice: slice INSIDE each expert (Megatron), which keeps every
            // expert on every rank and so keeps the in-graph router's global ids
            // valid — the one layout the fused whole-model MoE trunk can run under
            // tensor parallelism. Costs a one-time materialization at load; buys
            // the whole model as one graph per rank instead of ~90 op dispatches
            // per layer. See Gemma4Model.TensorParallelMoEFused.cs.
            if (BuildGemma4TensorSlicedExpertShards())
                return;

            // Otherwise whole-expert partitioning: zero-copy, and it keeps the
            // batched per-layer MoE kernel usable.
            if (BuildGemma4ExpertParallelShards())
                return;

            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                if (!HasMoE(layer))
                    continue;

                string prefix = $"blk.{layer}.";

                // Router weight: replicated (stays in _weights)

                // Expert weights: column-parallel for gate/up, row-parallel for down
                for (int e = 0; e < _numExperts; e++)
                {
                    // Check for fused gate_up first, then separate. A fused
                    // [gate|up] expert weight needs segment-aware sharding (like
                    // the dense gate_up); separate gate/up are single segments
                    // and shard correctly with a contiguous split.
                    string fusedGateUpKey = prefix + $"ffn_gate_up_exps.{e}.weight";
                    if (_weights.ContainsKey(fusedGateUpKey) || _quantWeights.ContainsKey(fusedGateUpKey))
                    {
                        ShardFusedGateUpColumnParallel(fusedGateUpKey);
                    }
                    else
                    {
                        ShardExpertColumnParallel(prefix + $"ffn_gate_exps.{e}.weight");
                        ShardExpertColumnParallel(prefix + $"ffn_up_exps.{e}.weight");
                    }
                    ShardExpertRowParallel(prefix + $"ffn_down_exps.{e}.weight");
                }
            }
        }

        private void ShardExpertColumnParallel(string weightName)
        {
            int tp = TpDegree;
            int globalTp = GlobalTpDegree;

            if (_quantWeights.TryGetValue(weightName, out var qw))
            {
                long rowsPerShard = qw.Ne1 / globalTp;
                long rowBytes = NativeDequant.RowSize(qw.GgmlType, qw.Ne0);
                long bytesPerShard = rowsPerShard * rowBytes;

                var shards = new QuantizedWeight[tp];
                for (int r = 0; r < tp; r++)
                {
                    IntPtr shardPtr = IntPtr.Add(qw.Data, (int)((TpRankOffset + r) * bytesPerShard));
                    shards[r] = QuantizedWeight.CreateExternalView(
                        shardPtr, bytesPerShard, qw.GgmlType, qw.Ne0, rowsPerShard, qw);
                }

                _tpQuantWeights[weightName] = shards;
                _quantWeights.Remove(weightName);
                // Keep qw alive: shards are external views into its buffer.
                // Disposing here would leave dangling pointers (the expert
                // weights are NOT preloaded to GPU — they're served from host).
            }
            else if (_weights.TryGetValue(weightName, out var w))
            {
                long shardSize = w.Sizes[0] / globalTp;
                var shards = new Tensor[tp];
                for (int r = 0; r < tp; r++)
                {
                    var view = w.Narrow(0, (TpRankOffset + r) * shardSize, shardSize);
                    shards[r] = Ops.NewContiguous(view);
                    view.Dispose();
                }

                _tpWeights[weightName] = shards;
                _weights.Remove(weightName);
                w.Dispose();
            }
        }

        private void ShardExpertRowParallel(string weightName)
        {
            int tp = TpDegree;
            int globalTp = GlobalTpDegree;

            if (_quantWeights.TryGetValue(weightName, out var qw))
            {
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

                var shards = new QuantizedWeight[tp];
                for (int r = 0; r < tp; r++)
                {
                    IntPtr shardPtr = QuantizedWeight.AllocateBuffer(totalBytesPerShard);
                    unsafe
                    {
                        byte* src = (byte*)qw.Data.ToPointer();
                        byte* dst = (byte*)shardPtr.ToPointer();
                        long srcBlockOffset = (TpRankOffset + r) * blocksPerShard * typeSize;
                        for (long row = 0; row < qw.Ne1; row++)
                        {
                            Buffer.MemoryCopy(
                                src + row * srcRowBytes + srcBlockOffset,
                                dst + row * dstRowBytes,
                                dstRowBytes, blockBytesPerShard);
                        }
                    }
                    shards[r] = new QuantizedWeight(shardPtr, totalBytesPerShard,
                        qw.GgmlType, ne0PerShard, qw.Ne1);
                }

                _tpQuantWeights[weightName] = shards;
                _quantWeights.Remove(weightName);
                qw.Dispose();
            }
            else if (_weights.TryGetValue(weightName, out var w))
            {
                long shardSize = w.Sizes[1] / globalTp;
                var shards = new Tensor[tp];
                for (int r = 0; r < tp; r++)
                {
                    var view = w.Narrow(1, (TpRankOffset + r) * shardSize, shardSize);
                    shards[r] = Ops.NewContiguous(view);
                    view.Dispose();
                }

                _tpWeights[weightName] = shards;
                _weights.Remove(weightName);
                w.Dispose();
            }
        }

        // ====================================================================
        // TP KV cache initialization
        // ====================================================================

        private void InitGemma4TpKVCache(int initialSeqLen, int maxSeqLen)
        {
            int tp = TpDegree;              // local GPUs on this node (loop / allocator index)
            int globalTp = GlobalTpDegree; // total ranks across the cluster (shard/head split)
            DType kvDtype = _kvCacheDtype.ToDType();

            _maxContextLength = maxSeqLen;
            _tpKvCacheCapacity = initialSeqLen;
            _tpKvCacheK = new Tensor[Config.NumLayers][];
            _tpKvCacheV = new Tensor[Config.NumLayers][];

            for (int l = 0; l < Config.NumLayers; l++)
            {
                int kvHeads = KVHeadsForLayer(l);
                // Each rank owns 1/globalTp of the KV heads (weights are sharded
                // by the GLOBAL degree). For multi-node runs globalTp > tp.
                int kvHeadsPerGpu = kvHeads / globalTp;
                int headDim = HeadDimForLayer(l);
                int cacheLen = IsLocalLayer(l) ? Math.Min(_slidingWindow, initialSeqLen) : initialSeqLen;

                _tpKvCacheK[l] = new Tensor[tp];
                _tpKvCacheV[l] = new Tensor[tp];
                for (int r = 0; r < tp; r++)
                {
                    var alloc = _tpGroup.GetAllocator(r);
                    _tpKvCacheK[l][r] = new Tensor(alloc, kvDtype, kvHeadsPerGpu, cacheLen, headDim);
                    _tpKvCacheV[l][r] = new Tensor(alloc, kvDtype, kvHeadsPerGpu, cacheLen, headDim);
                    // Same finite-padding requirement as the single-GPU cache.
                    InitGemma4CacheTensor(_tpKvCacheK[l][r]);
                    InitGemma4CacheTensor(_tpKvCacheV[l][r]);
                }
            }

            Console.WriteLine($"  Gemma4 TP KV cache initialized: {tp} local GPU(s)/{globalTp} total, " +
                $"local KV/GPU={Config.NumKVHeads / globalTp}, global KV/GPU={_numGlobalKVHeads / globalTp}");
        }

        private void EnsureGemma4TpCacheCapacity(int requiredSeqLen)
        {
            if (requiredSeqLen <= _tpKvCacheCapacity)
                return;
            if (requiredSeqLen > _maxContextLength)
                throw new InvalidOperationException(
                    $"Requested sequence length {requiredSeqLen} exceeds configured max context {_maxContextLength}.");

            int newCapacity = Math.Max(_tpKvCacheCapacity, 1);
            while (newCapacity < requiredSeqLen)
                newCapacity = Math.Min(_maxContextLength, newCapacity * 2);

            int tp = TpDegree;
            int globalTp = GlobalTpDegree;
            DType kvDtype = _kvCacheDtype.ToDType();

            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (IsLocalLayer(l))
                    continue; // SWA layers never grow

                int kvHeads = KVHeadsForLayer(l);
                int kvHeadsPerGpu = kvHeads / globalTp; // 1/globalTp of KV heads per rank
                int headDim = HeadDimForLayer(l);

                for (int r = 0; r < tp; r++)
                {
                    var alloc = _tpGroup.GetAllocator(r);
                    var newK = new Tensor(alloc, kvDtype, kvHeadsPerGpu, newCapacity, headDim);
                    var newV = new Tensor(alloc, kvDtype, kvHeadsPerGpu, newCapacity, headDim);
                    InitGemma4CacheTensor(newK);
                    InitGemma4CacheTensor(newV);

                    if (_cacheSeqLen > 0)
                    {
                        using var srcK = _tpKvCacheK[l][r].Narrow(1, 0, _cacheSeqLen);
                        using var dstK = newK.Narrow(1, 0, _cacheSeqLen);
                        Ops.Copy(dstK, srcK);

                        using var srcV = _tpKvCacheV[l][r].Narrow(1, 0, _cacheSeqLen);
                        using var dstV = newV.Narrow(1, 0, _cacheSeqLen);
                        Ops.Copy(dstV, srcV);
                    }

                    _tpKvCacheK[l][r].Dispose();
                    _tpKvCacheV[l][r].Dispose();
                    _tpKvCacheK[l][r] = newK;
                    _tpKvCacheV[l][r] = newV;
                }
            }

            _tpKvCacheCapacity = newCapacity;
            // The fused TP graphs bake in the KV buffer addresses and the cache
            // extents, both of which just moved.
            if (_tpFusedDecodeReady || _tpMoeFusedReady)
            {
                GgmlBasicOps.Gemma4ResetDecodeCache();
                GgmlBasicOps.Gemma4MoEResetDecodeCache();
                CountDecodeGraphReset();
                BuildGemma4TpDecodeArrays();
            }
            Console.WriteLine($"Expanded Gemma4 TP cache to {newCapacity} tokens ({tp} GPUs).");
        }

        // ====================================================================
        // TP forward pass
        // ====================================================================

        private float[] ForwardTP(int[] tokens)
        {
            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            int tp = TpDegree;
            EnsureGemma4TpCacheCapacity(startPos + seqLen);

            long t1 = Stopwatch.GetTimestamp();
            Tensor hidden0 = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t1;

            // Gemma4 scales embeddings by sqrt(hidden_size).
            ScaleEmbedding(hidden0);

            // Inject pending vision/audio embeddings on rank 0 before
            // broadcasting (mirrors the non-TP ForwardComputeCore). Without
            // this, multimodal prompts silently drop the image/audio under TP.
            HashSet<int> exceptPositions = null;
            if (_pendingVisionEmbeddingsList.Count > 0)
            {
                exceptPositions = new HashSet<int>();
                foreach (var (emb, pos) in _pendingVisionEmbeddingsList)
                {
                    int numTokens = (int)emb.Sizes[0];
                    for (int i = 0; i < numTokens; i++)
                        exceptPositions.Add(startPos + pos + i);
                    InjectVisionEmbeddings(hidden0, emb, pos, startPos);
                    emb.Dispose();
                }
                _pendingVisionEmbeddingsList.Clear();
            }
            if (_pendingAudioEmbeddingsList.Count > 0)
            {
                exceptPositions ??= new HashSet<int>();
                foreach (var (emb, pos) in _pendingAudioEmbeddingsList)
                {
                    int numTokens = (int)emb.Sizes[0];
                    for (int i = 0; i < numTokens; i++)
                        exceptPositions.Add(startPos + pos + i);
                    InjectVisionEmbeddings(hidden0, emb, pos, startPos);
                    emb.Dispose();
                }
                _pendingAudioEmbeddingsList.Clear();
            }

            // Per-Layer Embedding (PLE): a per-layer input combining a token
            // embedding and a projection of the (initial) hidden state, injected
            // at the end of every block. Computed once from the embedding — the
            // non-TP path does the same. Without it the per-layer token-identity
            // signal is never reinjected and generation collapses to the same
            // input-independent output. Replicated input => identical on every
            // node, so it needs no AllReduce.
            Tensor perLayerInputs = _pleDim > 0 ? ComputePLE(tokens, hidden0, seqLen) : null;

            // Single-token step: run the whole trunk as one fused graph per rank,
            // cut at the two AllReduce points per layer, instead of ~2000 per-op
            // host round trips. Falls through when the native side declines.
            if (seqLen == 1 && exceptPositions == null && _tpFusedDecodeReady)
            {
                float[] fused = EnsureFoldLogitsBuffer();
                if (TryGemma4FusedModelDecodeTP(hidden0, startPos, perLayerInputs, fused))
                {
                    perLayerInputs?.Dispose();
                    _cacheSeqLen += seqLen;
                    _forwardCount++;
                    _forwardSw.Stop();
                    return fused;
                }
            }

            // MoE trunk: same treatment through its own whole-model kernel, with a
            // third AllReduce per layer for the expert down projection. The verify
            // kernel takes the multimodal bidirectional-span mask directly (at
            // startPos == 0), so image chunks stay on the fused path too.
            if (perLayerInputs == null && _tpMoeFusedReady)
            {
                if (seqLen == 1 && exceptPositions == null)
                {
                    float[] fused = EnsureFoldLogitsBuffer();
                    if (TryGemma4FusedMoEModelDecodeTP(hidden0, startPos, fused))
                    {
                        _cacheSeqLen += seqLen;
                        _forwardCount++;
                        _forwardSw.Stop();
                        return fused;
                    }
                }
                else if (seqLen > 1 && TryGemma4FusedMoEModelVerifyTP(hidden0, startPos, seqLen, exceptPositions))
                {
                    float[] result = Gemma4TpFinalNormAndLmHead(hidden0, seqLen);
                    _cacheSeqLen += seqLen;
                    _forwardCount++;
                    _forwardSw.Stop();
                    return result;
                }
            }

            // Multi-token step (prefill / chunk): same fused-graph treatment, over
            // N tokens. The trunk output lands back in hidden0 and the final norm
            // + LM head run here on rank 0, mirroring the single-GPU verify path.
            if (seqLen > 1 && _tpFusedDecodeReady &&
                TryGemma4FusedModelVerifyTP(hidden0, startPos, seqLen, perLayerInputs, exceptPositions))
            {
                perLayerInputs?.Dispose();
                float[] result = Gemma4TpFinalNormAndLmHead(hidden0, seqLen);
                _cacheSeqLen += seqLen;
                _forwardCount++;
                _forwardSw.Stop();
                return result;
            }

            // The per-op path below reads the KV caches from host memory; a fused
            // decode leaves its writes on the devices only.
            SyncGemma4TpKvCacheToHost();

            // Broadcast embedding to all GPUs.
            Tensor[] hidden = BroadcastTensorToAllRanks(hidden0);

            bool tpDebug = Environment.GetEnvironmentVariable("TS_TP_DEBUG") == "1";
            if (tpDebug)
                DumpTpTensorStats(hidden[0], $"embed seqLen={seqLen} startPos={startPos}");

            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                Tensor perLayerInput = perLayerInputs != null
                    ? ExtractPerLayerSlice(perLayerInputs, layer, seqLen)
                    : null;
                hidden = Gemma4TransformerBlockTP(hidden, layer, seqLen, startPos, perLayerInput, exceptPositions);
                perLayerInput?.Dispose();
                if (tpDebug)
                {
                    DumpTpTensorStats(hidden[0], $"layer {layer} rank0");
                    if (TpDegree > 1)
                        DumpTpTensorStats(hidden[1], $"layer {layer} rank1");
                }
            }

            perLayerInputs?.Dispose();

            // Final norm + LM head on GPU 0 only.
            Tensor normed = RMSNormOp(hidden[0], "output_norm.weight");
            for (int r = 0; r < tp; r++)
                hidden[r].Dispose();

            Tensor lastHidden;
            if (seqLen > 1)
            {
                using var narrowed = normed.Narrow(0, seqLen - 1, 1);
                lastHidden = Ops.NewContiguous(narrowed);
            }
            else
            {
                lastHidden = normed.CopyRef();
            }
            normed.Dispose();

            long t2 = Stopwatch.GetTimestamp();
            string outputWeight = _hasTiedOutput ? "token_embd.weight" : "output.weight";
            Tensor logitsTensor = LinearForward(lastHidden, outputWeight);
            _lmHeadTicks += Stopwatch.GetTimestamp() - t2;
            lastHidden.Dispose();

            if (_finalLogitSoftcap > 0f)
                ApplyLogitSoftcap(logitsTensor);

            long t3 = Stopwatch.GetTimestamp();
            if (_logitsBuffer == null || _logitsBuffer.Length != Config.VocabSize)
                _logitsBuffer = new float[Config.VocabSize];
            _logitsBuffer = TensorToFloatArray(logitsTensor);
            _logitsCopyTicks += Stopwatch.GetTimestamp() - t3;
            logitsTensor.Dispose();

            _cacheSeqLen += seqLen;
            _forwardCount++;
            _forwardSw.Stop();
            return _logitsBuffer;
        }

        private Tensor[] Gemma4TransformerBlockTP(Tensor[] hidden, int layer, int seqLen, int startPos,
            Tensor perLayerInput, HashSet<int> exceptPositions = null)
        {
            string prefix = $"blk.{layer}";
            int tp = TpDegree;
            bool isLocal = IsLocalLayer(layer);
            bool isShared = _kvDonorMap != null && _kvDonorMap.ContainsKey(layer);
            int headDim = HeadDimForLayer(layer);
            int numKVHeads = KVHeadsForLayer(layer);
            // Per-rank head counts use the GLOBAL degree: weights/caches are
            // sharded across all ranks in the cluster, not just this node's GPUs.
            // (Single-node: GlobalTpDegree == TpDegree, so this is unchanged.)
            int numHeadsPerGpu = Config.NumHeads / GlobalTpDegree;
            int numKVHeadsPerGpu = numKVHeads / GlobalTpDegree;

            // 1. Pre-attention norm (replicated).
            Tensor[] normed = TpRMSNorm(hidden, $"{prefix}.attn_norm.weight");

            // 2. Column-parallel projection. KV-sharing layers only project Q
            // (they read K/V from their donor layer's cache); all other layers
            // use the fused QKV projection when available, or fall back to
            // three separate column-parallel projections (E4B: Q/K/V have
            // different output dims so FuseQKVWeights cannot merge them).
            Tensor[] qkvFused;
            if (isShared)
            {
                qkvFused = TpColumnParallelLinear(normed[0], $"{prefix}.attn_q.weight");
            }
            else if (_tpWeights.ContainsKey($"{prefix}.attn_qkv.weight") ||
                     _tpQuantWeights.ContainsKey($"{prefix}.attn_qkv.weight"))
            {
                qkvFused = TpColumnParallelLinear(normed[0], $"{prefix}.attn_qkv.weight");
            }
            else
            {
                Tensor[] qPart = TpColumnParallelLinear(normed[0], $"{prefix}.attn_q.weight");
                Tensor[] kPart = TpColumnParallelLinear(normed[0], $"{prefix}.attn_k.weight");
                Tensor[] vPart = TpColumnParallelLinear(normed[0], $"{prefix}.attn_v.weight");
                qkvFused = new Tensor[tp];
                for (int r = 0; r < tp; r++)
                {
                    var alloc = _tpGroup.GetAllocator(r);
                    long totalDim = qPart[r].Sizes[1] + kPart[r].Sizes[1] + vPart[r].Sizes[1];
                    qkvFused[r] = new Tensor(alloc, qPart[r].ElementType, seqLen, totalDim);
                    Ops.Concat(qkvFused[r], 1, qPart[r], kPart[r], vPart[r]);
                    qPart[r].Dispose();
                    kPart[r].Dispose();
                    vPart[r].Dispose();
                }
            }
            for (int r = 0; r < tp; r++)
                normed[r].Dispose();

            if (TpDebugLevel >= 2)
                DumpTpTensorStats(qkvFused[0], $"layer {layer} qkv rank0");

            // 3. Per-GPU attention.
            Tensor[] attnOut = Gemma4AttentionTP(qkvFused, layer, seqLen, startPos,
                isLocal, isShared, headDim, numHeadsPerGpu, numKVHeadsPerGpu, exceptPositions);

            if (TpDebugLevel >= 2)
                DumpTpTensorStats(attnOut[0], $"layer {layer} attnout rank0 local={isLocal} shared={isShared}");

            // 4. Row-parallel output projection + AllReduce.
            Tensor reducedAttn = TpRowParallelLinear(attnOut, $"{prefix}.attn_output.weight");
            for (int r = 0; r < tp; r++)
                attnOut[r].Dispose();

            if (Environment.GetEnvironmentVariable("TS_TP_DEBUG") == "1")
                DumpTpTensorStats(reducedAttn, $"layer {layer} attn-reduced");

            // 5. Broadcast + post-attention norm + residual.
            Tensor[] attnReplicated = BroadcastTensorToAllRanks(reducedAttn);
            Tensor[] postAttnNormed = TpRMSNorm(attnReplicated, $"{prefix}.post_attention_norm.weight");
            for (int r = 1; r < tp; r++)
                attnReplicated[r].Dispose();
            reducedAttn.Dispose();

            TpResidualAdd(postAttnNormed, hidden);
            for (int r = 0; r < tp; r++)
                hidden[r].Dispose();

            // 6. FFN (dense or MoE).
            Tensor[] ffnOut = HasMoE(layer)
                ? Gemma4MoEBlockTP(postAttnNormed, layer, seqLen, prefix)
                : Gemma4DenseFFNBlockTP(postAttnNormed, layer, seqLen, prefix);

            // 7. PLE injection + per-layer output scalar (replicated, rank 0).
            ApplyGemma4PleAndScaleTP(ffnOut, perLayerInput, layer, prefix, seqLen);

            return ffnOut;
        }

        /// <summary>
        /// Gemma4's per-layer PLE block, run on the replicated rank-0 activation:
        /// a gated bottleneck MLP (inp_gate -> GELU·perLayerInput -> proj ->
        /// post_norm) added to the residual, followed by the per-layer output
        /// scalar. Mirrors the non-TP TransformerBlock's PLE injection. The result
        /// is then re-broadcast to the other local ranks so the next layer's
        /// column-parallel projection sees identical input on every GPU.
        /// </summary>
        private void ApplyGemma4PleAndScaleTP(Tensor[] hidden, Tensor perLayerInput, int layer, string prefix, int seqLen)
        {
            int tp = TpDegree;

            if (perLayerInput != null &&
                (_weights.ContainsKey($"{prefix}.inp_gate.weight") || _quantWeights.ContainsKey($"{prefix}.inp_gate.weight")))
            {
                Tensor gate = LinearForward(hidden[0], $"{prefix}.inp_gate.weight");
                if (gate != null)
                {
                    Ops.GELUMul(gate, gate, perLayerInput);
                    using var pleProj = LinearForward(gate, $"{prefix}.proj.weight");
                    gate.Dispose();
                    if (pleProj != null)
                    {
                        string postPleNormKey = $"{prefix}.post_norm.weight";
                        using var pleNormed = RMSNormOp(pleProj, postPleNormKey);
                        Ops.Add(hidden[0], hidden[0], pleNormed);
                    }
                }
            }

            float scalar = _layerScalars[layer];
            if (scalar != 1f)
                Ops.Mul(hidden[0], hidden[0], scalar);

            for (int r = 1; r < tp; r++)
            {
                hidden[r].Dispose();
                hidden[r] = ReplicateTensorToRank(hidden[0], r);
            }
        }

        private Tensor[] Gemma4AttentionTP(Tensor[] qkvFused, int layer, int seqLen, int startPos,
            bool isLocal, bool isShared, int headDim, int numHeadsPerGpu, int numKVHeadsPerGpu,
            HashSet<int> exceptPositions = null)
        {
            int tp = TpDegree;
            int qDimPerGpu = numHeadsPerGpu * headDim;
            int kDimPerGpu = numKVHeadsPerGpu * headDim;
            int totalSeqLen = startPos + seqLen;
            string prefix = $"blk.{layer}";
            float ropeBase = isLocal ? _ropeLocalBase : _ropeGlobalBase;
            // KV-sharing layers attend the donor layer's cache instead of their own.
            int kvCacheLayer = isShared ? _kvDonorMap[layer] : layer;

            var results = new Tensor[tp];

            for (int r = 0; r < tp; r++)
            {
                var alloc = _tpGroup.GetAllocator(r);

                // Split Q, K, V from fused QKV output. KV-sharing layers only
                // projected Q; K/V come from the donor's cache.
                Tensor qTensor, kTensor = null, vTensor = null;
                if (isShared)
                {
                    // The projection output is Q-only; use it directly.
                    qTensor = qkvFused[r];
                }
                else if (seqLen == 1)
                {
                    qTensor = qkvFused[r].Narrow(1, 0, qDimPerGpu);
                    kTensor = qkvFused[r].Narrow(1, qDimPerGpu, kDimPerGpu);
                    vTensor = qkvFused[r].Narrow(1, qDimPerGpu + kDimPerGpu, kDimPerGpu);
                    qkvFused[r].Dispose();
                }
                else
                {
                    using (var qView = qkvFused[r].Narrow(1, 0, qDimPerGpu))
                        qTensor = Ops.NewContiguous(qView);
                    using (var kView = qkvFused[r].Narrow(1, qDimPerGpu, kDimPerGpu))
                        kTensor = Ops.NewContiguous(kView);
                    using (var vView = qkvFused[r].Narrow(1, qDimPerGpu + kDimPerGpu, kDimPerGpu))
                        vTensor = Ops.NewContiguous(vView);
                    qkvFused[r].Dispose();
                }

                // QK norm (per-GPU, replicated weights). K is skipped for
                // KV-sharing layers (they reuse the donor's cached K/V).
                qTensor = ApplyGemma4QKNormTP(qTensor, $"{prefix}.attn_q_norm.weight", numHeadsPerGpu, headDim, seqLen, r);
                if (!isShared)
                {
                    kTensor = ApplyGemma4QKNormTP(kTensor, $"{prefix}.attn_k_norm.weight", numKVHeadsPerGpu, headDim, seqLen, r);
                    // Gemma4 applies an UNWEIGHTED RMS norm to V before attention
                    // (the non-TP path's ApplyUnweightedRMSNorm). An earlier TP path
                    // omitted it, so every layer attended to un-normalised
                    // values and the output collapsed into repetitive gibberish.
                    vTensor = ApplyGemma4VNormTP(vTensor, numKVHeadsPerGpu, headDim, seqLen, r);
                }

                if (TpDebugLevel >= 2 && r == 0)
                {
                    DumpTpTensorStats(qTensor, $"  L{layer} q-postnorm");
                    if (kTensor != null) DumpTpTensorStats(kTensor, $"  L{layer} k-postnorm");
                    if (vTensor != null) DumpTpTensorStats(vTensor, $"  L{layer} v-postnorm");
                }

                // RoPE.
                qTensor = ApplyGemma4RoPETP(qTensor, numHeadsPerGpu, headDim, seqLen, startPos, ropeBase);
                if (!isShared)
                    kTensor = ApplyGemma4RoPETP(kTensor, numKVHeadsPerGpu, headDim, seqLen, startPos, ropeBase);

                if (TpDebugLevel >= 2 && r == 0)
                {
                    DumpTpTensorStats(qTensor, $"  L{layer} q-postrope");
                    if (kTensor != null) DumpTpTensorStats(kTensor, $"  L{layer} k-postrope");
                }

                // NOTE: no 1/sqrt(headDim) Q scaling here. Gemma4 attention runs at
                // unit scale — the Q/K RMS norms provide the normalisation and the
                // non-TP path passes scale=1.0 to every attention kernel with no Q
                // pre-scaling. The previous 1/sqrt(headDim) pre-scale
                // shrank the logits ~16-22x, flattening attention to near-uniform and
                // driving the repeated-token output under TP.

                if (seqLen == 1)
                {
                    // Decode: copy K/V to per-GPU cache (own layers only), run
                    // attention against the own or donor cache.
                    if (!isShared)
                    {
                        // SWA caches are circular: wrap the write position.
                        int cachePos = isLocal
                            ? startPos % (int)_tpKvCacheK[layer][r].Sizes[1]
                            : startPos;
                        CopyToCacheDecode(_tpKvCacheK[layer][r], kTensor, _tpKvCacheV[layer][r], vTensor,
                            numKVHeadsPerGpu, headDim, cachePos);
                        kTensor.Dispose();
                        vTensor.Dispose();
                    }

                    var attnResult = new Tensor(alloc, DType.Float32, 1, numHeadsPerGpu * headDim);
                    Tensor kCache = _tpKvCacheK[kvCacheLayer][r];
                    Tensor vCache = _tpKvCacheV[kvCacheLayer][r];
                    int cacheLen = (int)kCache.Sizes[1];

                    // Prefer the fused on-device GQA decode kernel, exactly as the
                    // single-GPU path does. The AttentionDecode* fallbacks below run
                    // on the CPU: they take GetFloatPtr of the KV cache, which drags
                    // the whole per-rank cache back to host memory once per layer per
                    // token. That is what made a TP decode step cost seconds rather
                    // than milliseconds -- the kernels were simply never offered the
                    // work under TP. The kernel picks its device from the result
                    // tensor's allocator, so it lands on rank r's GPU.
                    if (isLocal)
                    {
                        int attendLen = Math.Min(totalSeqLen, _slidingWindow);
                        int attendStart = Math.Max(0, startPos + 1 - attendLen);
                        if (!CudaFusedOps.TryGqaDecodeAttention(
                                attnResult, qTensor, kCache, vCache,
                                numHeadsPerGpu, numKVHeadsPerGpu, headDim,
                                attendStart, startPos + 1 - attendStart, cacheLen, true, 1f))
                        {
                            // SWA: circular decode attention.
                            AttentionDecodeCircular(qTensor, kCache, vCache, attnResult,
                                numHeadsPerGpu, numKVHeadsPerGpu, headDim, headDim,
                                startPos, attendLen, cacheLen, 1f);
                        }
                    }
                    else
                    {
                        if (!CudaFusedOps.TryGqaDecodeAttention(
                                attnResult, qTensor, kCache, vCache,
                                numHeadsPerGpu, numKVHeadsPerGpu, headDim,
                                0, totalSeqLen, cacheLen, false, 1f))
                        {
                            // Global: linear decode attention.
                            AttentionDecodeWithWindow(qTensor, kCache, vCache, attnResult,
                                numHeadsPerGpu, numKVHeadsPerGpu, headDim, headDim,
                                0, totalSeqLen, 1f);
                        }
                    }
                    qTensor.Dispose();

                    results[r] = attnResult;
                }
                else
                {
                    // Prefill path.
                    Tensor qHeads = ReshapeToHeads(qTensor, numHeadsPerGpu, seqLen, headDim);
                    qTensor.Dispose();

                    // Own layers project + cache their K/V; KV-sharing layers
                    // reuse the donor's already-cached K/V.
                    Tensor freshKHeads = null, freshVHeads = null;
                    if (!isShared)
                    {
                        Tensor kHeads = ReshapeToHeads(kTensor, numKVHeadsPerGpu, seqLen, headDim);
                        kTensor.Dispose();
                        Tensor vHeads = ReshapeToHeads(vTensor, numKVHeadsPerGpu, seqLen, headDim);
                        vTensor.Dispose();

                        if (isLocal)
                        {
                            int cacheLen = (int)_tpKvCacheK[layer][r].Sizes[1];
                            CopyToCacheCircular(_tpKvCacheK[layer][r], kHeads, startPos, seqLen, cacheLen);
                            CopyToCacheCircular(_tpKvCacheV[layer][r], vHeads, startPos, seqLen, cacheLen);
                            if (startPos > 0 && seqLen < cacheLen)
                            {
                                // Chunk 2+: queries near the chunk start attend the
                                // previous window, which fresh-only K/V can't serve.
                                // The circular cache (just written) holds the last
                                // min(totalSeqLen, cacheLen) positions — fall through
                                // to the gather path below (same as the donor case).
                                // Attending only the fresh chunk here dropped that
                                // history and, with an image, fed the bidi mask
                                // chunk-relative query positions against absolute
                                // exceptPositions.
                                kHeads.Dispose();
                                vHeads.Dispose();
                            }
                            else
                            {
                                // Chunk 1 (or a chunk that overflows the window, where
                                // the cache has already overwritten itself): attend the
                                // fresh K/V directly with the sliding-window mask.
                                freshKHeads = kHeads;
                                freshVHeads = vHeads;
                            }
                        }
                        else
                        {
                            CopyToCache(_tpKvCacheK[layer][r], kHeads, startPos, seqLen);
                            CopyToCache(_tpKvCacheV[layer][r], vHeads, startPos, seqLen);
                            kHeads.Dispose();
                            vHeads.Dispose();
                        }
                    }

                    int groupSize = numHeadsPerGpu / numKVHeadsPerGpu;

                    // Resolve the attention K/V source WITHOUT expanding the KV
                    // heads first. ExpandKVHeads (RepeatInterleave) has no direct
                    // CUDA kernel and falls back to the CPU, so materialising the
                    // grouped copy is itself a host round trip; the fused prefill
                    // kernel below reads the un-expanded heads in place.
                    Tensor kvSrcK, kvSrcV;
                    int kvAttendLen;
                    int kvStride;          // -1 => compact [heads, kvLen, dim]
                    bool ownsKvSrc;        // true => we allocated kvSrcK/V here

                    if (freshKHeads != null)
                    {
                        // SWA non-shared: attend the freshly computed K/V directly.
                        // The causal+window mask restricts each query to its window.
                        kvSrcK = freshKHeads;
                        kvSrcV = freshVHeads;
                        kvAttendLen = seqLen;
                        kvStride = -1;
                        ownsKvSrc = true;
                    }
                    else if (isLocal)
                    {
                        // SWA shared: gather the attend window from the donor's
                        // circular cache into a linear buffer.
                        int cacheLen = (int)_tpKvCacheK[kvCacheLayer][r].Sizes[1];
                        int attendLen = Math.Min(totalSeqLen, _slidingWindow);
                        int attendStart = totalSeqLen - attendLen;
                        var linK = new Tensor(alloc, DType.Float32, numKVHeadsPerGpu, attendLen, headDim);
                        var linV = new Tensor(alloc, DType.Float32, numKVHeadsPerGpu, attendLen, headDim);
                        GatherCircularToLinear(_tpKvCacheK[kvCacheLayer][r], linK, attendStart, attendLen, cacheLen, numKVHeadsPerGpu, headDim);
                        GatherCircularToLinear(_tpKvCacheV[kvCacheLayer][r], linV, attendStart, attendLen, cacheLen, numKVHeadsPerGpu, headDim);
                        kvSrcK = linK;
                        kvSrcV = linV;
                        kvAttendLen = attendLen;
                        kvStride = -1;
                        ownsKvSrc = true;
                    }
                    else
                    {
                        // Global: linear cache, full history, read in place.
                        kvSrcK = _tpKvCacheK[kvCacheLayer][r];
                        kvSrcV = _tpKvCacheV[kvCacheLayer][r];
                        kvAttendLen = totalSeqLen;
                        kvStride = (int)kvSrcK.Sizes[1];
                        ownsKvSrc = false;
                    }

                    int prefillWindow = isLocal ? _slidingWindow : 0;

                    // Fused flash-style prefill attention on the rank's GPU. Skipped
                    // when multimodal soft tokens need the bidirectional mask, which
                    // only the generic ApplyCausalMask path understands.
                    if (exceptPositions == null)
                    {
                        var fusedResult = new Tensor(alloc, DType.Float32, seqLen, numHeadsPerGpu * headDim);
                        if (CudaFusedOps.TryGqaPrefillAttention(
                                fusedResult, qHeads, kvSrcK, kvSrcV,
                                numHeadsPerGpu, numKVHeadsPerGpu, headDim,
                                seqLen, kvAttendLen,
                                kvAttendLen - seqLen, prefillWindow, 1.0f, kvStride))
                        {
                            qHeads.Dispose();
                            if (ownsKvSrc) { kvSrcK.Dispose(); kvSrcV.Dispose(); }
                            results[r] = fusedResult;
                            continue;
                        }
                        fusedResult.Dispose();
                    }

                    // Generic path: expand the grouped KV heads and run the
                    // score/mask/softmax/value chain op by op.
                    Tensor kExpanded = ExpandKVHeads(kvSrcK, groupSize, kvAttendLen);
                    Tensor vExpanded = ExpandKVHeads(kvSrcV, groupSize, kvAttendLen);
                    if (ownsKvSrc)
                    {
                        kvSrcK.Dispose();
                        kvSrcV.Dispose();
                    }

                    using var kT = kExpanded.Transpose(1, 2);
                    var scores = new Tensor(alloc, DType.Float32, numHeadsPerGpu, seqLen, kvAttendLen);
                    Ops.AddmmBatch(scores, 0, scores, 1f, qHeads, kT);
                    qHeads.Dispose();
                    kExpanded.Dispose();

                    // ApplyCausalMask works in key-index coordinates: key 0 is
                    // absolute position (totalSeqLen - kvAttendLen), and it derives
                    // query positions from that same origin. exceptPositions holds
                    // ABSOLUTE positions, so shift them into the mask's frame (the
                    // global branch has shift 0; the fresh-chunk and gathered-window
                    // branches start later in the sequence).
                    HashSet<int> exceptForMask = exceptPositions;
                    int maskShift = totalSeqLen - kvAttendLen;
                    if (exceptPositions != null && maskShift != 0)
                    {
                        exceptForMask = new HashSet<int>();
                        foreach (int p in exceptPositions)
                            exceptForMask.Add(p - maskShift);
                    }
                    ApplyCausalMask(scores, seqLen, kvAttendLen, prefillWindow, exceptForMask);
                    Ops.Softmax(scores, scores);

                    var attnOutTensor = new Tensor(alloc, DType.Float32, numHeadsPerGpu, seqLen, headDim);
                    Ops.AddmmBatch(attnOutTensor, 0, attnOutTensor, 1.0f, scores, vExpanded);
                    scores.Dispose();
                    vExpanded.Dispose();

                    Tensor flatOutput = ReshapeFromHeads(attnOutTensor, numHeadsPerGpu, seqLen, headDim);
                    attnOutTensor.Dispose();

                    results[r] = flatOutput;
                }
            }

            return results;
        }

        /// <summary>
        /// Gather a contiguous logical range [start, start+length) from a
        /// circular head-first cache [numHeads, cacheSize, headDim] into a
        /// linear buffer [numHeads, length, headDim]. Handles the wrap-around.
        /// </summary>
        private unsafe void GatherCircularToLinear(Tensor cache, Tensor result,
            int start, int length, int cacheSize, int numHeads, int headDim)
        {
            if (CudaFusedOps.TryGatherCircularHeadFirst(result, cache, start, length, cacheSize))
                return;

            float* src = GetFloatPtr(cache);
            float* dst = GetFloatPtr(result);
            for (int h = 0; h < numHeads; h++)
            {
                float* srcHead = src + (long)h * cacheSize * headDim;
                float* dstHead = dst + (long)h * length * headDim;
                int firstSlot = ((start % cacheSize) + cacheSize) % cacheSize;

                if (firstSlot + length <= cacheSize)
                {
                    Buffer.MemoryCopy(
                        srcHead + (long)firstSlot * headDim,
                        dstHead,
                        (long)length * headDim * 4,
                        (long)length * headDim * 4);
                }
                else
                {
                    int tailLen = cacheSize - firstSlot;
                    Buffer.MemoryCopy(
                        srcHead + (long)firstSlot * headDim,
                        dstHead,
                        (long)tailLen * headDim * 4,
                        (long)tailLen * headDim * 4);
                    int headLen = length - tailLen;
                    Buffer.MemoryCopy(
                        srcHead,
                        dstHead + (long)tailLen * headDim,
                        (long)headLen * headDim * 4,
                        (long)headLen * headDim * 4);
                }
            }
        }

        private Tensor ApplyGemma4QKNormTP(Tensor data, string weightName, int numHeads, int headDim, int seqLen, int rank)
        {
            var alpha = _weights[weightName];
            Tensor alphaLocal = ReplicateTensorToRank(alpha, rank);

            if (seqLen == 1)
            {
                RMSNormInPlace(data, alphaLocal, numHeads, headDim, Config.Eps);
                if (!ReferenceEquals(alphaLocal, alpha)) alphaLocal.Dispose();
                return data;
            }

            using var reshaped = data.View(seqLen * numHeads, headDim);
            Tensor normed = Ops.RMSNorm(null, reshaped, alphaLocal, null, Config.Eps);
            data.Dispose();
            if (!ReferenceEquals(alphaLocal, alpha)) alphaLocal.Dispose();

            Tensor result = normed.View(seqLen, numHeads * headDim);
            normed.Dispose();
            return result;
        }

        // Per-rank all-ones weight for the unweighted V RMS norm, recreated when
        // the head dim changes (local SWA layers use 256, global layers 512).
        private Tensor[] _tpVNormOnes;
        private int _tpVNormOnesDim;

        /// <summary>
        /// TP-aware unweighted V RMS norm — the counterpart of the non-TP path's
        /// ApplyUnweightedRMSNorm. Normalises each head's headDim-vector to unit
        /// RMS with no learned weight (an all-ones weight on the rank's allocator).
        /// </summary>
        private Tensor ApplyGemma4VNormTP(Tensor data, int numHeads, int headDim, int seqLen, int rank)
        {
            int tp = TpDegree;
            if (_tpVNormOnes == null || _tpVNormOnesDim != headDim)
            {
                if (_tpVNormOnes != null)
                    for (int r = 0; r < _tpVNormOnes.Length; r++)
                        _tpVNormOnes[r]?.Dispose();
                _tpVNormOnes = new Tensor[tp];
                for (int r = 0; r < tp; r++)
                {
                    _tpVNormOnes[r] = new Tensor(_tpGroup.GetAllocator(r), DType.Float32, headDim);
                    Ops.Fill(_tpVNormOnes[r], 1f);
                }
                _tpVNormOnesDim = headDim;
            }

            RMSNormInPlace(data, _tpVNormOnes[rank], numHeads * seqLen, headDim, Config.Eps);
            return data;
        }

        private Tensor ApplyGemma4RoPETP(Tensor data, int numHeads, int headDim, int seqLen, int startPos,
            float ropeBase)
        {
            return ApplyRoPEPrefill(data, numHeads, headDim, seqLen, startPos, ropeBase);
        }

        // ====================================================================
        // Dense FFN block under TP
        // ====================================================================

        private Tensor[] Gemma4DenseFFNBlockTP(Tensor[] hidden, int layer, int seqLen, string prefix)
        {
            int tp = TpDegree;

            // 1. Pre-FFN norm (replicated).
            Tensor[] ffnNormed = TpRMSNorm(hidden, $"{prefix}.ffn_norm.weight");

            // 2. Column-parallel gate/up projection.
            Tensor[] gateUp = TpColumnParallelLinear(ffnNormed[0], $"{prefix}.ffn_gate_up.weight");
            for (int r = 0; r < tp; r++)
                ffnNormed[r].Dispose();

            // 3. Per-GPU GELU·mul (GeGLU activation).
            int halfDim = (int)(gateUp[0].Sizes[1] / 2);
            Tensor[] gateResults = new Tensor[tp];
            for (int r = 0; r < tp; r++)
            {
                Tensor gate, up;
                if (seqLen == 1)
                {
                    gate = gateUp[r].Narrow(1, 0, halfDim);
                    up = gateUp[r].Narrow(1, halfDim, halfDim);
                }
                else
                {
                    using var gView = gateUp[r].Narrow(1, 0, halfDim);
                    gate = Ops.NewContiguous(gView);
                    using var uView = gateUp[r].Narrow(1, halfDim, halfDim);
                    up = Ops.NewContiguous(uView);
                }
                gateUp[r].Dispose();

                Ops.GELUMul(gate, gate, up);
                up.Dispose();
                gateResults[r] = gate;
            }

            // 4. Row-parallel down projection + AllReduce.
            Tensor ffnOut = TpRowParallelLinear(gateResults, $"{prefix}.ffn_down.weight");
            for (int r = 0; r < tp; r++)
                gateResults[r].Dispose();

            // 5. Broadcast + post-FFN norm + residual.
            Tensor[] ffnReplicated = BroadcastTensorToAllRanks(ffnOut);
            string postFfnNormKey = $"{prefix}.post_ffw_norm.weight";
            if (!_weights.ContainsKey(postFfnNormKey))
                postFfnNormKey = $"{prefix}.ffn_post_norm.weight";
            Tensor[] postFfnNormed = TpRMSNorm(ffnReplicated, postFfnNormKey);
            for (int r = 1; r < tp; r++)
                ffnReplicated[r].Dispose();
            ffnOut.Dispose();

            TpResidualAdd(hidden, postFfnNormed);
            for (int r = 0; r < tp; r++)
                postFfnNormed[r].Dispose();

            return hidden;
        }

        // ====================================================================
        // MoE block under TP
        // ====================================================================

        private Tensor[] Gemma4MoEBlockTP(Tensor[] hidden, int layer, int seqLen, string prefix)
        {
            int tp = TpDegree;
            int hiddenSize = Config.HiddenSize;

            // Gemma4 MoE layers have BOTH a dense FFN and MoE FFN.
            // Step 1: Dense FFN (gate_up + GELU + down) with post_ffw_norm_1
            Tensor[] ffnNormed = TpRMSNorm(hidden, $"{prefix}.ffn_norm.weight");
            Tensor[] gateUp = TpColumnParallelLinear(ffnNormed[0], $"{prefix}.ffn_gate_up.weight");
            for (int r = 0; r < tp; r++)
                ffnNormed[r].Dispose();

            int halfDim = (int)(gateUp[0].Sizes[1] / 2);
            Tensor[] denseResults = new Tensor[tp];
            for (int r = 0; r < tp; r++)
            {
                Tensor gate, up;
                if (seqLen == 1)
                {
                    gate = gateUp[r].Narrow(1, 0, halfDim);
                    up = gateUp[r].Narrow(1, halfDim, halfDim);
                }
                else
                {
                    using var gView = gateUp[r].Narrow(1, 0, halfDim);
                    gate = Ops.NewContiguous(gView);
                    using var uView = gateUp[r].Narrow(1, halfDim, halfDim);
                    up = Ops.NewContiguous(uView);
                }
                gateUp[r].Dispose();
                Ops.GELUMul(gate, gate, up);
                up.Dispose();
                denseResults[r] = gate;
            }

            Tensor denseFFNOut = TpRowParallelLinear(denseResults, $"{prefix}.ffn_down.weight");
            for (int r = 0; r < tp; r++)
                denseResults[r].Dispose();

            // Apply post_ffw_norm_1 to dense FFN output
            Tensor[] denseReplicated = BroadcastTensorToAllRanks(denseFFNOut);
            string postNorm1Key = $"{prefix}.post_ffw_norm_1.weight";
            if (!_weights.ContainsKey(postNorm1Key))
                postNorm1Key = $"{prefix}.ffn_post_norm_1.weight";
            Tensor[] denseNormed = TpRMSNorm(denseReplicated, postNorm1Key);
            for (int r = 1; r < tp; r++)
                denseReplicated[r].Dispose();
            denseFFNOut.Dispose();

            // Step 2: MoE FFN (tensor-parallel experts)
            // Dense FFN and MoE operate in PARALLEL on the same input (hidden),
            // mirroring the non-TP path where both MoEForward(attnOut, ...) and
            // FFNGelu(attnOut, ...) receive the same pre-FFN hidden state.
            // Router is replicated — compute routing on rank 0.
            var (routingWeightsFlat, selectedExpertsFlat) = MoERoute(hidden[0], prefix, seqLen);

            // Apply pre_ffw_norm_2 to get the expert FFN input (the non-TP path
            // normalizes hiddenState with this weight before feeding experts).
            string moeNormKey = $"{prefix}.pre_ffw_norm_2.weight";
            if (!_weights.ContainsKey(moeNormKey))
                moeNormKey = $"{prefix}.ffn_pre_norm_2.weight";
            Tensor[] moeInput = TpRMSNorm(hidden, moeNormKey);

            var moeResults = new Tensor[tp];

            // Batched fast path: one dispatch per rank per layer. Runs from the
            // whole-expert stacks (expert-parallel) or, when the fused trunk's
            // Megatron-sliced stacks are the only shards (multimodal prompts
            // fall off the fused trunk onto this per-op path), from those.
            if ((UsesExpertParallelMoE || UsesTensorSlicedMoE) &&
                TryGemma4MoEExpertParallel(moeInput, moeResults, selectedExpertsFlat, routingWeightsFlat,
                    layer, seqLen, hiddenSize))
            {
                for (int r = 0; r < tp; r++)
                    moeInput[r].Dispose();
                _tpGroup.AllReduce(moeResults);
                return Gemma4MoEBlockTPFinish(hidden, denseNormed, moeResults, prefix);
            }

            // Every rank walks the same token/expert schedule over its own slice
            // of the expert weights, so the ranks are independent right up to the
            // AllReduce below — run them concurrently.
            _tpGroup.RunPerRank(r =>
            {
                var alloc = _tpGroup.GetAllocator(r);

                // Accumulate expert outputs.
                var output = new Tensor(alloc, DType.Float32, seqLen, hiddenSize);
                Ops.Fill(output, 0f);

                // Per-token expert routing: each token may select different
                // experts, so process one token at a time (matches the non-TP
                // MoEForward which also iterates per token).
                for (int s = 0; s < seqLen; s++)
                {
                    Tensor tokenInput;
                    if (seqLen == 1)
                    {
                        tokenInput = moeInput[r];
                    }
                    else
                    {
                        using var slice = moeInput[r].Narrow(0, s, 1);
                        tokenInput = Ops.NewContiguous(slice);
                    }

                    for (int k = 0; k < _numExpertsUsed; k++)
                    {
                        int flatIdx = s * _numExpertsUsed + k;
                        int expertIdx = selectedExpertsFlat[flatIdx];
                        float weight = routingWeightsFlat[flatIdx];

                        // Check for fused gate_up first
                        string fusedGateUpKey = prefix + $".ffn_gate_up_exps.{expertIdx}.weight";
                        Tensor gateOut;
                        if (_tpQuantWeights.ContainsKey(fusedGateUpKey) || _tpWeights.ContainsKey(fusedGateUpKey))
                        {
                            Tensor fusedOut = TpExpertLinear(tokenInput, fusedGateUpKey, r, 1);
                            int expertHalf = (int)(fusedOut.Sizes[1] / 2);
                            using var gView = fusedOut.Narrow(1, 0, expertHalf);
                            Tensor gatePart = Ops.NewContiguous(gView);
                            using var uView = fusedOut.Narrow(1, expertHalf, expertHalf);
                            Tensor upPart = Ops.NewContiguous(uView);
                            fusedOut.Dispose();
                            Ops.GELUMul(gatePart, gatePart, upPart);
                            upPart.Dispose();
                            gateOut = gatePart;
                        }
                        else
                        {
                            string gateKey = prefix + $".ffn_gate_exps.{expertIdx}.weight";
                            string upKey = prefix + $".ffn_up_exps.{expertIdx}.weight";
                            Tensor g = TpExpertLinear(tokenInput, gateKey, r, 1);
                            Tensor u = TpExpertLinear(tokenInput, upKey, r, 1);
                            Ops.GELUMul(g, g, u);
                            u.Dispose();
                            gateOut = g;
                        }

                        string downKey = prefix + $".ffn_down_exps.{expertIdx}.weight";
                        Tensor downOut = TpExpertLinear(gateOut, downKey, r, 1);
                        gateOut.Dispose();

                        // Apply per-expert scale if present
                        float expertScale = GetGemma4ExpertScale(layer, expertIdx);
                        Ops.Mul(downOut, downOut, weight * expertScale);

                        // Accumulate into the correct token row.
                        if (seqLen == 1)
                        {
                            Ops.Add(output, output, downOut);
                        }
                        else
                        {
                            using var outRow = output.Narrow(0, s, 1);
                            Ops.Add(outRow, outRow, downOut);
                        }
                        downOut.Dispose();
                    }

                    if (seqLen > 1)
                        tokenInput.Dispose();
                }

                moeResults[r] = output;
            });

            for (int r = 0; r < tp; r++)
                moeInput[r].Dispose();

            // AllReduce MoE results.
            _tpGroup.AllReduce(moeResults);

            return Gemma4MoEBlockTPFinish(hidden, denseNormed, moeResults, prefix);
        }

        /// <summary>
        /// Tail of the MoE block, shared by the expert-parallel and per-expert
        /// paths: post-norm the reduced MoE output, add it to the dense FFN
        /// branch, then final-norm and add back into the residual stream.
        /// </summary>
        private Tensor[] Gemma4MoEBlockTPFinish(Tensor[] hidden, Tensor[] denseNormed, Tensor[] moeResults, string prefix)
        {
            int tp = TpDegree;

            if (Environment.GetEnvironmentVariable("TS_TP_DEBUG") == "1")
            {
                DumpTpTensorStats(denseNormed[0], $"{prefix} dense-ffn");
                DumpTpTensorStats(moeResults[0], $"{prefix} moe-reduced");
            }

            // Apply post_ffw_norm_2 to MoE output
            string postNorm2Key = $"{prefix}.post_ffw_norm_2.weight";
            if (!_weights.ContainsKey(postNorm2Key))
                postNorm2Key = $"{prefix}.ffn_post_norm_2.weight";
            Tensor[] moeNormed = TpRMSNorm(moeResults, postNorm2Key);
            for (int r = 1; r < tp; r++)
                moeResults[r].Dispose();

            // Add dense FFN (post-norm-1) + MoE (post-norm-2)
            TpResidualAdd(denseNormed, moeNormed);
            for (int r = 0; r < tp; r++)
                moeNormed[r].Dispose();

            // Apply final post_ffw_norm + residual to original hidden
            string postFfnNormKey = $"{prefix}.post_ffw_norm.weight";
            if (!_weights.ContainsKey(postFfnNormKey))
                postFfnNormKey = $"{prefix}.ffn_post_norm.weight";

            // denseNormed now holds (dense + moe), apply final norm
            Tensor[] finalNormed = TpRMSNorm(denseNormed, postFfnNormKey);
            TpResidualAdd(hidden, finalNormed);
            for (int r = 0; r < tp; r++)
            {
                finalNormed[r].Dispose();
                denseNormed[r].Dispose();
            }

            return hidden;
        }

        /// <summary>
        /// Run one MoE layer expert-parallel: each rank dispatches the batched
        /// <c>ggml_mul_mat_id</c> kernel over the experts it owns.
        ///
        /// The kernel wants a dense [seqLen][nUsed] route table, so routes that
        /// belong to another rank are neutralised rather than removed — expert 0
        /// with weight 0, which contributes nothing. That costs each rank a full
        /// nUsed-wide pass instead of its ~nUsed/tp share, but it is still three
        /// dispatches per layer against the hundreds of thousands the per-token
        /// loop issued, and it keeps the route table shape the kernel expects.
        ///
        /// Returns false if any rank's shapes don't suit the kernel, leaving the
        /// caller on the per-expert path.
        /// </summary>
        private bool TryGemma4MoEExpertParallel(
            Tensor[] moeInput,
            Tensor[] moeResults,
            int[] selectedExpertsFlat,
            float[] routingWeightsFlat,
            int layer,
            int seqLen,
            int hiddenSize)
        {
            var gateShards = _tpStackedGate?[layer];
            var downShards = _tpStackedDown?[layer];
            StackedExpertWeights[] upShards = null;
            // Two shard layouts and one host layout, in the order they are
            // checked. Megatron-sliced: every rank holds ALL experts at 1/tp of
            // the FFN width (fused [gate|up]), so the dispatch runs with global
            // expert ids and the row-parallel down projection's partial sums meet
            // in the caller's AllReduce — the same reduction the whole-expert
            // split needs.
            //
            // A layer offloaded by --n-cpu-moe has no per-rank shard at all: its
            // experts stay in system RAM, unsharded. Reaching here means the fused
            // trunk declined this shape (a multimodal chunk past position 0, say),
            // so serve the layer the way that trunk would have — once, on the host,
            // over the whole expert stack — and give the result to rank 0 alone so
            // the caller's AllReduce reproduces the single-GPU value exactly.
            bool layerOnCpu = MoeCpuOffloadConfig.IsLayerOnCpu(layer);
            bool sliced;
            if (layerOnCpu)
            {
                if (_layerStackedGate?[layer] == null || _layerStackedDown?[layer] == null)
                    return false;
                sliced = true;              // global expert ids, no per-rank filtering
                gateShards = null;          // resolved per rank below
                downShards = null;
            }
            else
            {
                sliced = gateShards == null || downShards == null;
                if (sliced)
                {
                    gateShards = _tpSlicedGateUp?[layer];
                    downShards = _tpSlicedDown?[layer];
                    if (gateShards == null || downShards == null)
                        return false;
                }
                else
                {
                    upShards = _tpStackedUp?[layer];
                }
            }

            int tp = TpDegree;
            int rankOffset = TpRankOffset;
            int nUsed = _numExpertsUsed;
            int perRank = sliced ? _numExperts : _tpExpertsPerRank;
            // An offloaded layer reads the unsharded stacks; every other mode
            // reads its rank's shard (identical shapes across ranks).
            StackedExpertWeights gateShape = layerOnCpu ? _layerStackedGate[layer] : gateShards[0];
            bool fusedGateUp = layerOnCpu ? (_layerStackedUp?[layer] == null) : (upShards == null);
            int nFf = fusedGateUp
                ? (int)(gateShape.PerExpertNe1 / 2)
                : (int)gateShape.PerExpertNe1;

            float[] perExpertScale = _layerPerExpertScale?[layer];
            int totalRoutes = seqLen * nUsed;

            // Per-rank route tables, built once on the calling thread: the rank
            // workers must not race on shared scratch.
            var localExperts = new int[tp][];
            var localWeights = new float[tp][];
            for (int r = 0; r < tp; r++)
            {
                // Sliced mode: all experts are rank-local, ids pass through
                // globally and the foreign-route filler below never runs.
                int first = sliced ? 0 : (rankOffset + r) * perRank;
                int last = first + perRank;
                var ids = new int[totalRoutes];
                var wts = new float[totalRoutes];
                var taken = new bool[perRank];
                for (int s = 0; s < seqLen; s++)
                {
                    int baseIdx = s * nUsed;
                    Array.Clear(taken, 0, perRank);

                    // Own routes first, so the filler pass below can see which
                    // local experts this token already uses.
                    for (int k = 0; k < nUsed; k++)
                    {
                        int i = baseIdx + k;
                        int e = selectedExpertsFlat[i];
                        if (e >= first && e < last)
                        {
                            int local = e - first;
                            ids[i] = local;
                            taken[local] = true;
                            float w = routingWeightsFlat[i];
                            if (perExpertScale != null) w *= perExpertScale[e];
                            wts[i] = w;
                        }
                        else
                        {
                            ids[i] = -1;
                            wts[i] = 0f;
                        }
                    }

                    // Foreign routes contribute nothing, but they still occupy a
                    // slot in the dense route table. Give each a *distinct*
                    // local expert: real top-k routing never repeats an expert
                    // within a token, and the batched kernel's per-expert
                    // gather/scatter relies on that — duplicates make two
                    // destination slots claim one source row. perRank (>= nUsed)
                    // guarantees a free id exists.
                    int probe = 0;
                    for (int k = 0; k < nUsed; k++)
                    {
                        int i = baseIdx + k;
                        if (ids[i] >= 0) continue;
                        while (probe < perRank && taken[probe]) probe++;
                        int filler = probe < perRank ? probe : 0;
                        if (probe < perRank) taken[probe] = true;
                        ids[i] = filler;
                    }
                }
                localExperts[r] = ids;
                localWeights[r] = wts;
            }

            if (!_loggedExpertParallelShapes &&
                string.Equals(Environment.GetEnvironmentVariable("TS_GGML_LOG_VRAM"), "1", StringComparison.Ordinal))
            {
                _loggedExpertParallelShapes = true;
                var g0 = gateShape;
                var d0 = layerOnCpu ? _layerStackedDown[layer] : downShards[0];
                Console.WriteLine($"  [MoE-EP] seqLen={seqLen} hidden={hiddenSize} nFf={nFf} perRank={perRank} nUsed={nUsed} " +
                    $"fusedGateUp={fusedGateUp} gate=[{g0.PerExpertNe0}x{g0.PerExpertNe1}x{g0.NumExperts}] type={g0.GgmlType} bytes={g0.TotalRawBytes} " +
                    $"down=[{d0.PerExpertNe0}x{d0.PerExpertNe1}x{d0.NumExperts}] type={d0.GgmlType} bytes={d0.TotalRawBytes}");
            }

            bool ok = true;
            _tpGroup.RunPerRank(r =>
            {
                var alloc = _tpGroup.GetAllocator(r);
                var output = new Tensor(alloc, DType.Float32, seqLen, hiddenSize);
                // Offloaded: only rank 0 evaluates the (unsharded) experts on the
                // host; the others contribute zeros, so the AllReduce that
                // follows yields exactly the single-GPU value.
                if (layerOnCpu && r != 0)
                {
                    Ops.Fill(output, 0f);
                    moeResults[r] = output;
                    return;
                }
                var g = layerOnCpu ? _layerStackedGate[layer] : gateShards[r];
                var d = layerOnCpu ? _layerStackedDown[layer] : downShards[r];
                var u = layerOnCpu ? _layerStackedUp?[layer] : upShards?[r];

                IntPtr upData = u == null ? IntPtr.Zero : u.Data;
                int upType = u == null ? 0 : u.GgmlType;
                long upNe0 = u == null ? 0L : u.PerExpertNe0;
                long upNe1 = u == null ? 0L : u.PerExpertNe1;
                long upBytes = u == null ? 0L : u.TotalRawBytes;

                try
                {
                    GgmlBasicOps.MoEFFNPrefill(
                        moeInput[r], output,
                        seqLen, hiddenSize, nFf, perRank, nUsed,
                        localExperts[r], localWeights[r],
                        g.Data, g.GgmlType, g.PerExpertNe0, g.PerExpertNe1, g.TotalRawBytes,
                        upData, upType, upNe0, upNe1, upBytes,
                        d.Data, d.GgmlType, d.PerExpertNe0, d.PerExpertNe1, d.TotalRawBytes,
                        gateBias: null, upBias: null, downBias: null,
                        activation: GgmlBasicOps.MoEActivation.GEGLUSplit,
                        runOnCpu: MoeCpuOffloadConfig.IsLayerOnCpu(layer));
                    InvalidateTensorDeviceCache(output);
                    moeResults[r] = output;
                }
                catch (NotSupportedException)
                {
                    output.Dispose();
                    ok = false;
                }
            });

            if (!ok)
            {
                for (int r = 0; r < tp; r++)
                {
                    moeResults[r]?.Dispose();
                    moeResults[r] = null;
                }
            }
            return ok;
        }

        /// <summary>TS_TP_DEBUG: 1 = per-layer activation stats, 2 = per-op stats
        /// inside the attention block, for hunting the first divergent op in the
        /// tensor-parallel per-op path.</summary>
        private static readonly int TpDebugLevel =
            int.TryParse(Environment.GetEnvironmentVariable("TS_TP_DEBUG"), out int _tpDbgLvl) ? _tpDbgLvl : 0;

        private static unsafe void DumpTpTensorStats(Tensor t, string label)
        {
            long n = t.ElementCount();
            float* p = GetFloatPtr(t);
            double sumSq = 0; float mn = float.MaxValue, mx = float.MinValue;
            long nanCount = 0;
            for (long i = 0; i < n; i++)
            {
                float v = p[i];
                if (float.IsNaN(v) || float.IsInfinity(v)) { nanCount++; continue; }
                sumSq += (double)v * v;
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            Console.WriteLine($"  [tp-debug] {label}: rms={Math.Sqrt(sumSq / Math.Max(1, n - nanCount)):G6} min={mn:G6} max={mx:G6} nan/inf={nanCount}/{n}");
        }

        private Tensor TpExpertLinear(Tensor input, string weightName, int rank, int seqLen)
        {
            var alloc = _tpGroup.GetAllocator(rank);

            if (_tpQuantWeights.TryGetValue(weightName, out var qShards))
            {
                var qw = qShards[rank];
                int outDim = (int)qw.Ne1;
                var result = new Tensor(alloc, DType.Float32, seqLen, outDim);
                AddmmQuantManaged(result, ReplicateTensorToRank(input, rank), qw);
                return result;
            }
            else if (_tpWeights.TryGetValue(weightName, out var wShards))
            {
                var w = wShards[rank];
                int outDim = (int)w.Sizes[0];
                var result = new Tensor(alloc, DType.Float32, seqLen, outDim);
                using var wT = w.Transpose();
                var localInput = ReplicateTensorToRank(input, rank);
                Ops.Addmm(result, 0, result, 1.0f, localInput, wT);
                if (!ReferenceEquals(localInput, input)) localInput.Dispose();
                return result;
            }

            throw new KeyNotFoundException($"TP expert weight '{weightName}' not found.");
        }

        private (int[] experts, float[] weights) SelectGemma4TopKExperts(float[] routerLogits)
        {
            int numExperts = routerLogits.Length;
            var indices = new int[numExperts];
            for (int i = 0; i < numExperts; i++) indices[i] = i;
            Array.Sort(indices, (a, b) => routerLogits[b].CompareTo(routerLogits[a]));

            var topExperts = new int[_numExpertsUsed];
            var topWeights = new float[_numExpertsUsed];
            float sum = 0;
            for (int k = 0; k < _numExpertsUsed; k++)
            {
                topExperts[k] = indices[k];
                topWeights[k] = routerLogits[indices[k]];
                sum += topWeights[k];
            }

            if (sum > 0)
                for (int k = 0; k < _numExpertsUsed; k++)
                    topWeights[k] /= sum;

            return (topExperts, topWeights);
        }

        private float GetGemma4ExpertScale(int layer, int expertIdx)
        {
            if (_layerPerExpertScale != null && _layerPerExpertScale[layer] != null
                && expertIdx < _layerPerExpertScale[layer].Length)
                return _layerPerExpertScale[layer][expertIdx];
            return 1.0f;
        }

        // ====================================================================
        // TP-aware Dispose
        // ====================================================================

        private void DisposeGemma4TpState()
        {
            if (_tpVNormOnes != null)
            {
                for (int r = 0; r < _tpVNormOnes.Length; r++)
                    _tpVNormOnes[r]?.Dispose();
                _tpVNormOnes = null;
                _tpVNormOnesDim = 0;
            }

            if (_tpKvCacheK != null)
            {
                for (int l = 0; l < _tpKvCacheK.Length; l++)
                {
                    if (_tpKvCacheK[l] == null) continue;
                    for (int r = 0; r < _tpKvCacheK[l].Length; r++)
                    {
                        _tpKvCacheK[l][r]?.Dispose();
                        _tpKvCacheV[l][r]?.Dispose();
                    }
                }
            }
        }
    }
}
