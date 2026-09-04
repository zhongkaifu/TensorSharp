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
// NemotronModel.TensorParallel.cs
//
// Tensor-parallel forward pass for Nemotron (Mamba2 + Attention + FFN/MoE).
//
// Layer-type strategy:
//   - Attention layers: full Megatron column/row parallelism (fused QKV + output)
//   - FFN layers: column/row for dense, tensor-parallel experts for MoE
//   - Mamba2 layers: REPLICATED on rank 0 (host-side SSM state prevents
//     head-parallel sharding in this initial implementation). The result is
//     broadcast to all ranks after each Mamba2 layer.
//
// The Mamba2 replication is the honest trade-off: the SSM state lives in
// host float[] arrays and the managed per-token loop is the only CUDA path.
// Sharding it would require a device-resident per-rank SSM kernel. The
// attention + FFN/MoE layers hold the bulk of the model weights, so TP
// still delivers most of the memory savings.
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
    public partial class NemotronModel
    {
        // Per-GPU KV caches for attention layers: [layer][rank]
        // Per-rank expert shard tables for the batched decode kernel on the PER-EXPERT
        // (Megatron) MoE path, indexed [layer][expert][rank]. Unused when expert
        // parallelism takes: that path partitions whole experts and already covers
        // decode with the same one-dispatch-per-rank kernel.
        //
        // Built from _tpQuantWeights, not _quantWeights, because the per-expert
        // sharding REMOVES the expert weights from _quantWeights - which is why the
        // non-TP _expertUpQW/_expertDownQW tables come up empty on that path.
        private QuantizedWeight[][][] _tpExpertUpQW;
        private QuantizedWeight[][][] _tpExpertDownQW;
        private IntPtr[][] _tpMoeUpPtrs;
        private IntPtr[][] _tpMoeDownPtrs;
        private bool _tpMoeBatchedDecodeReady;

        private Tensor[][] _tpKvCacheK;
        private Tensor[][] _tpKvCacheV;
        private int _tpKvCacheCapacity;

        // ====================================================================
        // TP constraint validation
        // ====================================================================

        private void ValidateNemotronTpConstraints()
        {
            int tp = GlobalTpDegree;
            var errors = new List<string>();

            // Check per-layer attention head divisibility
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_layerTypes[l] != LayerType.Attention)
                    continue;

                if (_layerNumHeads[l] % tp != 0)
                {
                    errors.Add($"Layer {l} attention heads ({_layerNumHeads[l]}) not divisible by TP degree ({tp})");
                    break;
                }
                if (_layerNumKVHeads[l] % tp != 0)
                {
                    errors.Add($"Layer {l} KV heads ({_layerNumKVHeads[l]}) not divisible by TP degree ({tp})");
                    break;
                }
            }

            // Check FFN divisibility
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_layerTypes[l] != LayerType.FFN)
                    continue;

                if (_layerNFF[l] > 0 && _layerNFF[l] % tp != 0)
                {
                    errors.Add($"Layer {l} FFN length ({_layerNFF[l]}) not divisible by TP degree ({tp})");
                    break;
                }
            }

            // Tensor parallelism runs on the multi-GPU backends: direct CUDA and
            // the GGML CUDA/Vulkan backends (one ggml backend per GPU).
            if (_backend is not (BackendType.Cuda or BackendType.GgmlCuda or BackendType.GgmlVulkan))
                errors.Add($"TP requires a multi-GPU backend (cuda, ggml-cuda, ggml-vulkan), got {_backend}");

            if (errors.Count > 0)
                throw new InvalidOperationException(
                    $"Nemotron TP validation failed:\n  " + string.Join("\n  ", errors));

            int attnCount = 0, mambaCount = 0, ffnCount = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                switch (_layerTypes[l])
                {
                    case LayerType.Attention: attnCount++; break;
                    case LayerType.Mamba2: mambaCount++; break;
                    case LayerType.FFN: ffnCount++; break;
                }
            }

            Console.WriteLine($"  TP constraints validated: tp={tp} (local={TpDegree}), " +
                $"{attnCount} attention (TP), {mambaCount} Mamba2 (replicated), {ffnCount} FFN (TP)");
        }

        // ====================================================================
        // Weight sharding
        // ====================================================================

        private void ShardNemotronWeightsForTP()
        {
            // Attention output (row) + dense FFN down (row). The fused attn_qkv
            // ([Q|K|V]) is column-parallel but needs segment-aware sharding
            // (below). The dense FFN up projection (ffn_up.weight) is a single
            // column segment, so the generic contiguous column split is correct.
            var columnPatterns = _numExperts == 0
                ? new[] { "ffn_up.weight" }
                : Array.Empty<string>();
            var rowPatterns = _numExperts == 0
                ? new[] { "attn_output.weight", "ffn_down.weight" }
                : new[] { "attn_output.weight" };
            ShardWeightsForTensorParallelism(
                columnParallelPatterns: columnPatterns,
                rowParallelPatterns: rowPatterns);

            int headDim = Config.HeadDim;
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                // attn_qkv exists only on attention layers; the call no-ops
                // otherwise. Mamba2 layers stay replicated (not sharded).
                // NB: _layerPrefixes isn't populated until InitLayerInfo (which
                // runs after sharding), so use the literal prefix here.
                ShardConcatenatedColumnParallel($"blk.{layer}.attn_qkv.weight",
                    _layerNumHeads[layer] * headDim,     // Q
                    _layerNumKVHeads[layer] * headDim,   // K
                    _layerNumKVHeads[layer] * headDim);  // V
            }

            // MoE expert weights: tensor-parallel experts
            if (_numExperts > 0)
                ShardNemotronMoeWeightsForTP();

            // Mamba2 weights: NOT sharded (replicated on rank 0).
            // ssm_in.weight, ssm_out.weight, ssm_conv1d.weight, etc. stay in _weights.

            Console.WriteLine($"  Nemotron TP weight sharding complete ({GlobalTpDegree} GPUs, {TpDegree} local).");
        }

        private void ShardNemotronMoeWeightsForTP()
        {
            // Whole-expert partitioning first. It keeps each layer to ONE batched
            // dispatch per projection per rank; the per-expert Megatron slicing below
            // leaves every rank holding a piece of all 128 experts, which forces a
            // per-expert loop. All-or-nothing: if it takes, the per-expert shards must
            // NOT also be built, or every expert byte ends up resident twice.
            if (BuildNemotronExpertParallelShards())
                return;

            int tp = TpDegree;

            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                if (_layerTypes[layer] != LayerType.FFN || _numExperts == 0)
                    continue;

                string prefix = $"blk.{layer}.";

                // Router weight: replicated (stays in _weights)

                for (int e = 0; e < _numExperts; e++)
                {
                    ShardNemotronExpertColumnParallel(prefix + $"ffn_up_exps.{e}.weight");
                    ShardNemotronExpertRowParallel(prefix + $"ffn_down_exps.{e}.weight");
                }
            }
        }

        private void ShardNemotronExpertColumnParallel(string weightName)
        {
            int tp = TpDegree;
            int globalTp = GlobalTpDegree;
            int rankOffset = TpRankOffset;

            if (_quantWeights.TryGetValue(weightName, out var qw))
            {
                long rowsPerShard = qw.Ne1 / globalTp;
                long rowBytes = NativeDequant.RowSize(qw.GgmlType, qw.Ne0);
                long bytesPerShard = rowsPerShard * rowBytes;

                var shards = new QuantizedWeight[tp];
                for (int r = 0; r < tp; r++)
                {
                    int globalRank = rankOffset + r;
                    IntPtr shardPtr = IntPtr.Add(qw.Data, (int)(globalRank * bytesPerShard));
                    shards[r] = QuantizedWeight.CreateExternalView(
                        shardPtr, bytesPerShard, qw.GgmlType, qw.Ne0, rowsPerShard, qw);
                }

                _tpQuantWeights[weightName] = shards;
                _quantWeights.Remove(weightName);
                qw.Dispose();
            }
            else if (_weights.TryGetValue(weightName, out var w))
            {
                long shardSize = w.Sizes[0] / globalTp;
                var shards = new Tensor[tp];
                for (int r = 0; r < tp; r++)
                {
                    int globalRank = rankOffset + r;
                    var view = w.Narrow(0, globalRank * shardSize, shardSize);
                    shards[r] = Ops.NewContiguous(view);
                    view.Dispose();
                }

                _tpWeights[weightName] = shards;
                _weights.Remove(weightName);
                w.Dispose();
            }
        }

        private void ShardNemotronExpertRowParallel(string weightName)
        {
            int tp = TpDegree;
            int globalTp = GlobalTpDegree;
            int rankOffset = TpRankOffset;

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
                    int globalRank = rankOffset + r;
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
                    int globalRank = rankOffset + r;
                    var view = w.Narrow(1, globalRank * shardSize, shardSize);
                    shards[r] = Ops.NewContiguous(view);
                    view.Dispose();
                }

                _tpWeights[weightName] = shards;
                _weights.Remove(weightName);
                w.Dispose();
            }
        }

        // ====================================================================
        // TP cache initialization
        // ====================================================================

        private void InitNemotronTpCaches(int initialSeqLen, int maxSeqLen)
        {
            int tp = TpDegree;
            DType kvDtype = _kvCacheDtype.ToDType();

            _maxContextLength = maxSeqLen;
            _tpKvCacheCapacity = initialSeqLen;
            _tpKvCacheK = new Tensor[Config.NumLayers][];
            _tpKvCacheV = new Tensor[Config.NumLayers][];

            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_layerTypes[l] != LayerType.Attention)
                {
                    _tpKvCacheK[l] = null;
                    _tpKvCacheV[l] = null;
                    continue;
                }

                int kvHeadsPerGpu = _layerNumKVHeads[l] / GlobalTpDegree;
                int headDim = Config.HeadDim;

                _tpKvCacheK[l] = new Tensor[tp];
                _tpKvCacheV[l] = new Tensor[tp];
                for (int r = 0; r < tp; r++)
                {
                    var alloc = _tpGroup.GetAllocator(r);
                    _tpKvCacheK[l][r] = new Tensor(alloc, kvDtype, kvHeadsPerGpu, initialSeqLen, headDim);
                    _tpKvCacheV[l][r] = new Tensor(alloc, kvDtype, kvHeadsPerGpu, initialSeqLen, headDim);
                    InitializeCacheTensor(_tpKvCacheK[l][r]);
                    InitializeCacheTensor(_tpKvCacheV[l][r]);
                }
            }

            // Mamba2 state: replicated (weights aren't sharded under TP), so the
            // host-side recurrent state arrays are the same full size as the non-TP
            // path. InitCaches (which normally allocates these) is skipped under TP,
            // so allocate them here for every Mamba2 layer.
            //
            // The native decode scratch tensors are deliberately NOT allocated here,
            // which is what keeps TryMamba2NativeDecode off under TP. That kernel
            // holds the conv/SSM state on the device between decode steps and never
            // drains it, so the next multi-token forward - a chat turn that reuses the
            // KV prefix, or a speculative verify batch - resumes from whatever the
            // host arrays held before this turn generated anything. Keeping the state
            // coherent instead (downloadState: true) costs ~46 MB/token and measured
            // SLOWER than the managed path under TP: 49.0 vs 57.5 tok/s. So TP keeps
            // the recurrent state host-side, which is both correct and faster here.
            int convDim = Math.Max(0, _ssmDConv - 1);
            int convChannels = _ssmDInner + 2 * _ssmNGroup * _ssmDState;
            int ssmStateSize = _ssmDState * _ssmHeadDim * _ssmNHead;
            _convState = new float[Config.NumLayers][];
            _ssmState = new float[Config.NumLayers][];
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_layerTypes[l] != LayerType.Mamba2) continue;
                _convState[l] = new float[convDim * convChannels];
                _ssmState[l] = new float[ssmStateSize];
            }

            Console.WriteLine($"  Nemotron TP caches initialized: {tp} GPUs");

            // InitCaches does this for the non-TP path; without it the DFlash/DSpark
            // drafter is never attached under TP, so speculation arms against a trunk
            // that has no drafter.
            TryLoadNemotronDFlash(_draftModelPath);
        }

        private void EnsureNemotronTpCacheCapacity(int requiredSeqLen)
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
            DType kvDtype = _kvCacheDtype.ToDType();

            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_layerTypes[l] != LayerType.Attention || _tpKvCacheK[l] == null)
                    continue;

                int kvHeadsPerGpu = _layerNumKVHeads[l] / GlobalTpDegree;
                int headDim = Config.HeadDim;

                for (int r = 0; r < tp; r++)
                {
                    var alloc = _tpGroup.GetAllocator(r);
                    var newK = new Tensor(alloc, kvDtype, kvHeadsPerGpu, newCapacity, headDim);
                    var newV = new Tensor(alloc, kvDtype, kvHeadsPerGpu, newCapacity, headDim);
                    InitializeCacheTensor(newK);
                    InitializeCacheTensor(newV);

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
            Console.WriteLine($"Expanded Nemotron TP cache to {newCapacity} tokens ({tp} GPUs).");
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
            bool isDecode = seqLen == 1;
            EnsureNemotronTpCacheCapacity(startPos + seqLen);

            long t1 = Stopwatch.GetTimestamp();
            Tensor hidden0 = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t1;

            // Inject pending multimodal (vision/audio) embeddings on rank 0 before
            // broadcasting (mirrors the non-TP ForwardCore). Without this, image/
            // audio input is silently dropped under TP.
            if (_pendingVisionEmbeddings.Count > 0 || _pendingAudioEmbeddings.Count > 0)
            {
                foreach (var (emb, pos) in _pendingVisionEmbeddings)
                {
                    InjectMultimodalEmbeddings(hidden0, emb, pos);
                    emb.Dispose();
                }
                _pendingVisionEmbeddings.Clear();
                foreach (var (emb, pos) in _pendingAudioEmbeddings)
                {
                    InjectMultimodalEmbeddings(hidden0, emb, pos);
                    emb.Dispose();
                }
                _pendingAudioEmbeddings.Clear();
            }

            // Broadcast embedding to all GPUs.
            Tensor[] hidden = BroadcastTensorToAllRanks(hidden0);

            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                switch (_layerTypes[layer])
                {
                    case LayerType.Mamba2:
                        // Replicated on rank 0: gather, run, broadcast.
                        hidden = Mamba2BlockTP(hidden, layer, seqLen, isDecode);
                        break;
                    case LayerType.Attention:
                        hidden = NemotronAttentionBlockTP(hidden, layer, seqLen, startPos, isDecode);
                        break;
                    case LayerType.FFN:
                        hidden = NemotronFFNBlockTP(hidden, layer, seqLen, isDecode);
                        break;
                }
            }

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
            Tensor logitsTensor = LinearForward(lastHidden, "output.weight");
            if (logitsTensor == null)
                logitsTensor = LinearForward(lastHidden, "token_embd.weight");
            _lmHeadTicks += Stopwatch.GetTimestamp() - t2;
            lastHidden.Dispose();

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

        // ====================================================================
        // Mamba2 block under TP (replicated on rank 0)
        // ====================================================================

        private Tensor[] Mamba2BlockTP(Tensor[] hidden, int layer, int seqLen, bool isDecode)
        {
            int tp = TpDegree;

            // Run Mamba2 on rank 0 only (host-side SSM state).
            // hidden[0] is the rank-0 tensor, which is all we need.
            Tensor result = Mamba2Block(hidden[0], layer, seqLen, isDecode);

            // Dispose non-rank-0 copies (they're stale after the Mamba2 update).
            for (int r = 1; r < tp; r++)
                hidden[r].Dispose();

            // Broadcast the updated rank-0 result to all ranks.
            return BroadcastTensorToAllRanks(result);
        }

        // ====================================================================
        // Attention block under TP
        // ====================================================================

        private Tensor[] NemotronAttentionBlockTP(Tensor[] hidden, int layer, int seqLen, int startPos, bool isDecode)
        {
            int tp = TpDegree;
            string prefix = _layerPrefixes[layer];
            int numHeads = _layerNumHeads[layer];
            int numKVHeads = _layerNumKVHeads[layer];
            int numHeadsPerGpu = numHeads / GlobalTpDegree;
            int numKVHeadsPerGpu = numKVHeads / GlobalTpDegree;
            int headDim = Config.HeadDim;
            int qDimPerGpu = numHeadsPerGpu * headDim;
            int kDimPerGpu = numKVHeadsPerGpu * headDim;
            int totalSeqLen = startPos + seqLen;
            float scale = _attentionScale != 0 ? (float)_attentionScale : (1.0f / MathF.Sqrt(headDim));

            // 1. Attention norm (replicated).
            Tensor[] normed = TpRMSNorm(hidden, prefix + "attn_norm.weight");

            // 2. Column-parallel fused QKV projection.
            Tensor[] qkvFused = TpColumnParallelLinear(normed[0], prefix + "attn_qkv.weight");
            for (int r = 0; r < tp; r++)
                normed[r].Dispose();

            // 3. Per-GPU attention.
            var attnResults = new Tensor[tp];
            // Fan the ranks out instead of walking them: GGML ops submit AND
            // synchronize inside one call, so a sequential rank loop runs GPU 0 to
            // completion before GPU 1 starts and the wall clock is the SUM of the
            // ranks rather than the MAX. RunPerRank pins each worker to its own
            // backend, which is also what lets the direct native calls below skip
            // their own rank selection.
            _tpGroup.RunPerRank(r =>
            {
                var alloc = _tpGroup.GetAllocator(r);

                Tensor qTensor, kTensor, vTensor;
                if (seqLen == 1)
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

                // RoPE.
                qTensor = ApplyNemotronRoPETP(qTensor, numHeadsPerGpu, headDim, seqLen, startPos);
                kTensor = ApplyNemotronRoPETP(kTensor, numKVHeadsPerGpu, headDim, seqLen, startPos);

                if (seqLen == 1)
                {
                    CopyToCacheDecode(_tpKvCacheK[layer][r], kTensor, _tpKvCacheV[layer][r], vTensor,
                        numKVHeadsPerGpu, headDim, startPos);
                    kTensor.Dispose();
                    vTensor.Dispose();

                    var attnResult = new Tensor(alloc, DType.Float32, 1, numHeadsPerGpu * headDim);
                    AttentionDecodePureCS(qTensor, _tpKvCacheK[layer][r], _tpKvCacheV[layer][r],
                        attnResult, numHeadsPerGpu, numKVHeadsPerGpu, headDim, totalSeqLen, scale);
                    qTensor.Dispose();
                    attnResults[r] = attnResult;
                }
                else
                {
                    Tensor qHeads = ReshapeToHeads(qTensor, numHeadsPerGpu, seqLen, headDim);
                    qTensor.Dispose();
                    Tensor kHeads = ReshapeToHeads(kTensor, numKVHeadsPerGpu, seqLen, headDim);
                    kTensor.Dispose();
                    Tensor vHeads = ReshapeToHeads(vTensor, numKVHeadsPerGpu, seqLen, headDim);
                    vTensor.Dispose();

                    CopyToCache(_tpKvCacheK[layer][r], kHeads, startPos, seqLen);
                    CopyToCache(_tpKvCacheV[layer][r], vHeads, startPos, seqLen);
                    kHeads.Dispose();
                    vHeads.Dispose();

                    int groupSize = numHeadsPerGpu / numKVHeadsPerGpu;
                    Tensor kExpanded = ExpandKVHeads(_tpKvCacheK[layer][r], groupSize, totalSeqLen);
                    Tensor vExpanded = ExpandKVHeads(_tpKvCacheV[layer][r], groupSize, totalSeqLen);

                    using var kT = kExpanded.Transpose(1, 2);
                    var scores = new Tensor(alloc, DType.Float32, numHeadsPerGpu, seqLen, totalSeqLen);
                    Ops.AddmmBatch(scores, 0, scores, scale, qHeads, kT);
                    qHeads.Dispose();
                    kExpanded.Dispose();

                    Ops.AddCausalMask(scores, seqLen, startPos, float.NegativeInfinity);
                    Ops.Softmax(scores, scores);

                    var attnOut = new Tensor(alloc, DType.Float32, numHeadsPerGpu, seqLen, headDim);
                    Ops.AddmmBatch(attnOut, 0, attnOut, 1.0f, scores, vExpanded);
                    scores.Dispose();
                    vExpanded.Dispose();

                    Tensor flatOutput = ReshapeFromHeads(attnOut, numHeadsPerGpu, seqLen, headDim);
                    attnOut.Dispose();
                    attnResults[r] = flatOutput;
                }
            });

            // 4. Row-parallel output projection + AllReduce.
            Tensor reducedAttn = TpRowParallelLinear(attnResults, prefix + "attn_output.weight");
            for (int r = 0; r < tp; r++)
                attnResults[r].Dispose();

            // 5. Residual add.
            Tensor[] attnReplicated = BroadcastTensorToAllRanks(reducedAttn);
            TpResidualAdd(hidden, attnReplicated);
            for (int r = 1; r < tp; r++)
                attnReplicated[r].Dispose();
            reducedAttn.Dispose();

            return hidden;
        }

        private Tensor ApplyNemotronRoPETP(Tensor data, int numHeads, int headDim, int seqLen, int startPos)
        {
            // Nemotron attention uses no RoPE — return data unchanged.
            return data;
        }

        // ====================================================================
        // FFN block under TP (dense or MoE)
        // ====================================================================

        private Tensor[] NemotronFFNBlockTP(Tensor[] hidden, int layer, int seqLen, bool isDecode)
        {
            string prefix = _layerPrefixes[layer];

            if (_numExperts > 0)
                return NemotronMoEBlockTP(hidden, layer, seqLen, prefix);

            // Dense FFN: norm -> up (column) -> ReLU^2 -> down (row) -> residual.
            // Nemotron uses attn_norm.weight for the FFN input norm and a single
            // ffn_up projection with ReLU^2 (not a gated SwiGLU).
            int tp = TpDegree;

            Tensor[] normed = TpRMSNorm(hidden, prefix + "attn_norm.weight");

            Tensor[] upResults = TpColumnParallelLinear(normed[0], prefix + "ffn_up.weight");
            for (int r = 0; r < tp; r++)
                normed[r].Dispose();

            for (int r = 0; r < tp; r++)
                ReluSquaredInPlace(upResults[r]);

            Tensor ffnOut = TpRowParallelLinear(upResults, prefix + "ffn_down.weight");
            for (int r = 0; r < tp; r++)
                upResults[r].Dispose();

            Tensor[] ffnReplicated = BroadcastTensorToAllRanks(ffnOut);
            TpResidualAdd(hidden, ffnReplicated);
            for (int r = 1; r < tp; r++)
                ffnReplicated[r].Dispose();
            ffnOut.Dispose();

            return hidden;
        }

        // ====================================================================
        // MoE block under TP
        // ====================================================================

        private Tensor[] NemotronMoEBlockTP(Tensor[] hidden, int layer, int seqLen, string prefix)
        {
            int tp = TpDegree;
            int hiddenSize = Config.HiddenSize;
            ref var moeInfo = ref _moeLayerInfo[layer];
            bool hasLatent = moeInfo.HasLatentIn && moeInfo.LatentDim > 0;
            int expertOutDim = hasLatent ? moeInfo.LatentDim : hiddenSize;

            // 1. FFN input norm. Nemotron uses attn_norm.weight for every block
            //    type (the TP path previously read ffn_norm.weight, which the
            //    GGUF does not contain).
            Tensor[] normed = TpRMSNorm(hidden, prefix + "attn_norm.weight");

            // Router bias (replicated), if the model ships one.
            float[] routerBias = null;
            if (_weights.TryGetValue(prefix + "exp_probs_b.bias", out var biasTensor) ||
                _weights.TryGetValue(prefix + "exp_probs_b", out biasTensor))
                routerBias = TensorToFloatArray(biasTensor);

            // Latent-in projection (replicated): when present, experts operate in
            // latent space and the accumulated result is projected back to hidden
            // size via ffn_latent_out after the AllReduce.
            Tensor latentIn = hasLatent ? LinearForward(normed[0], prefix + "ffn_latent_in.weight") : null;

            // 2. Router (replicated — identical routing on every rank): computed
            //    ONCE on rank 0 where the unsharded router weight lives, then each
            //    token routes independently. SelectNemotronTopKExperts treats its
            //    whole input as ONE token's logits, so it must be fed exactly one
            //    numExperts-length row — the previous code passed the flattened
            //    [seqLen, numExperts] batch, which indexed the router bias out of
            //    bounds for seqLen > 1 (TP prefill crashed) and applied a single
            //    expert set to the whole chunk.
            Tensor routerLogitsT = LinearForward(normed[0], prefix + "ffn_gate_inp.weight");
            float[] routePtr = TensorToFloatArray(routerLogitsT);
            routerLogitsT.Dispose();

            int nUsed = _numExpertsUsed;
            int totalAssignments = seqLen * nUsed;
            var selectedExperts = new int[totalAssignments];
            var routeWeightsAll = new float[totalAssignments];
            var tokenLogits = new float[_numExperts];
            for (int s = 0; s < seqLen; s++)
            {
                Array.Copy(routePtr, s * _numExperts, tokenLogits, 0, _numExperts);
                var (te, tw) = SelectNemotronTopKExperts(tokenLogits, routerBias);
                for (int k = 0; k < nUsed; k++)
                {
                    selectedExperts[s * nUsed + k] = te[k];
                    routeWeightsAll[s * nUsed + k] = tw[k];
                }
            }

            // 3. Decode fast path: ONE GGML dispatch per rank covering every routed
            //    expert, reusing the kernel the non-TP decode path already uses. The
            //    per-expert fallback below costs two device dispatches plus a host
            //    gather/scatter for EACH expert on EACH rank - at 24 MoE layers x
            //    top-6 x 2 ranks that is 288 synchronous device round trips per token,
            //    which is what pinned both GPUs at 10-20% utilisation during decode.
            Tensor[] results =
                TryNemotronMoEExpertParallel(layer, seqLen, expertOutDim, normed,
                    selectedExperts, routeWeightsAll)
                ?? TryTpMoeBatchedExpertDecode(
                    layer, seqLen, expertOutDim, hasLatent, latentIn, normed,
                    selectedExperts, routeWeightsAll);

            if (results == null)
            {
                // Bucket the assignments by expert so each expert runs ONE batched
                // matmul over its tokens instead of a per-token dispatch (the gpt-oss
                // TP MoE shape; mirrors the non-TP batched path).
                var expertCounts = new int[_numExperts];
                for (int a = 0; a < totalAssignments; a++)
                    expertCounts[selectedExperts[a]]++;
                var expertOffsets = new int[_numExperts];
                for (int e = 1; e < _numExperts; e++)
                    expertOffsets[e] = expertOffsets[e - 1] + expertCounts[e - 1];
                var tokenMap = new int[totalAssignments];
                var weightMap = new float[totalAssignments];
                var fillPos = (int[])expertOffsets.Clone();
                for (int s = 0; s < seqLen; s++)
                    for (int k = 0; k < nUsed; k++)
                    {
                        int e = selectedExperts[s * nUsed + k];
                        int pos = fillPos[e]++;
                        tokenMap[pos] = s;
                        weightMap[pos] = routeWeightsAll[s * nUsed + k];
                    }

                // 3. Per-rank expert accumulation (column-parallel up + row-parallel down).
                results = new Tensor[tp];
                // Fan the ranks out instead of walking them: GGML ops submit AND
                // synchronize inside one call, so a sequential rank loop runs GPU 0 to
                // completion before GPU 1 starts and the wall clock is the SUM of the
                // ranks rather than the MAX. RunPerRank pins each worker to its own
                // backend, which is also what lets the direct native calls below skip
                // their own rank selection.
                _tpGroup.RunPerRank(r =>
                {
                    var alloc = _tpGroup.GetAllocator(r);
                    Tensor expertInput = hasLatent ? latentIn : normed[r];
                    int expertInDim = (int)expertInput.Sizes[1];
                    long inRowBytes = (long)expertInDim * sizeof(float);

                    var output = new Tensor(alloc, DType.Float32, seqLen, expertOutDim);

                    unsafe
                    {
                        float* inPtr = GetFloatPtr(expertInput);
                        float* outPtr = GetFloatPtr(output);
                        for (long z = 0; z < (long)seqLen * expertOutDim; z++)
                            outPtr[z] = 0f;

                        for (int e = 0; e < _numExperts; e++)
                        {
                            int count = expertCounts[e];
                            if (count == 0) continue;
                            int offset = expertOffsets[e];

                            string upKey = prefix + $"ffn_up_exps.{e}.weight";
                            string downKey = prefix + $"ffn_down_exps.{e}.weight";

                            // Gather this expert's tokens into a [count, dim] batch.
                            var batchInput = new Tensor(alloc, DType.Float32, count, expertInDim);
                            float* bPtr = GetFloatPtr(batchInput);
                            for (int i = 0; i < count; i++)
                            {
                                int tokenIdx = tokenMap[offset + i];
                                Buffer.MemoryCopy(inPtr + (long)tokenIdx * expertInDim,
                                    bPtr + (long)i * expertInDim, inRowBytes, inRowBytes);
                            }

                            Tensor upOut = TpNemotronExpertLinear(batchInput, upKey, r, count);
                            batchInput.Dispose();
                            ReluSquaredInPlace(upOut);   // Nemotron MoE uses ReLU^2, not SiLU.
                            Tensor downOut = TpNemotronExpertLinear(upOut, downKey, r, count);
                            upOut.Dispose();

                            // Weighted scatter-add back into each token's output row.
                            float* dPtr = GetFloatPtr(downOut);
                            for (int i = 0; i < count; i++)
                            {
                                int tokenIdx = tokenMap[offset + i];
                                float w = weightMap[offset + i];
                                float* src = dPtr + (long)i * expertOutDim;
                                float* dst = outPtr + (long)tokenIdx * expertOutDim;
                                for (int d = 0; d < expertOutDim; d++)
                                    dst[d] += w * src[d];
                            }
                            downOut.Dispose();
                        }
                    }

                    // Host buffer is authoritative — push to device for the AllReduce.
                    output.EnsureDeviceCurrent();
                    results[r] = output;
                });
            }

            latentIn?.Dispose();

            // 3. AllReduce the expert accumulation across ranks.
            _tpGroup.AllReduce(results);

            // 4. Project back from latent space (replicated) to hidden size.
            Tensor contribution;
            if (hasLatent)
            {
                contribution = LinearForward(results[0], prefix + "ffn_latent_out.weight");
                for (int r = 0; r < tp; r++) results[r].Dispose();
            }
            else
            {
                contribution = results[0];
                for (int r = 1; r < tp; r++) results[r].Dispose();
            }

            // 5. Shared experts (replicated): normed -> up_shexp -> ReLU^2 ->
            //    down_shexp, added to the contribution. Computed on rank 0 from the
            //    replicated normed input.
            if (moeInfo.HasSharedExperts)
            {
                Tensor sharedUp = LinearForward(normed[0], prefix + "ffn_up_shexp.weight");
                ReluSquaredInPlace(sharedUp);
                using var sharedDown = LinearForward(sharedUp, prefix + "ffn_down_shexp.weight");
                sharedUp.Dispose();
                Ops.Add(contribution, contribution, sharedDown);
            }

            for (int r = 0; r < tp; r++) normed[r].Dispose();

            // 6. Broadcast the contribution to all ranks and add the residual.
            Tensor[] contribReplicated = BroadcastTensorToAllRanks(contribution);
            TpResidualAdd(hidden, contribReplicated);
            for (int r = 1; r < tp; r++) contribReplicated[r].Dispose();
            contribution.Dispose();

            return hidden;
        }

        private Tensor TpNemotronExpertLinear(Tensor input, string weightName, int rank, int seqLen)
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

        private (int[] experts, float[] weights) SelectNemotronTopKExperts(float[] routerLogits, float[] bias)
        {
            int numExperts = routerLogits.Length;

            // Router logits -> sigmoid probabilities; top-K selection uses the
            // bias-adjusted probabilities (matches the non-TP MoEForward).
            var probs = new float[numExperts];
            var selectionProbs = new float[numExperts];
            for (int e = 0; e < numExperts; e++)
            {
                probs[e] = SigmoidScalar(routerLogits[e]);
                selectionProbs[e] = bias != null ? probs[e] + bias[e] : probs[e];
            }

            var indices = new int[numExperts];
            for (int i = 0; i < numExperts; i++) indices[i] = i;
            Array.Sort(indices, (a, b) => selectionProbs[b].CompareTo(selectionProbs[a]));

            var topExperts = new int[_numExpertsUsed];
            var topWeights = new float[_numExpertsUsed];
            float sum = 0;
            for (int k = 0; k < _numExpertsUsed; k++)
            {
                topExperts[k] = indices[k];
                // Routing weight is the SIGMOID probability (not the raw logit and
                // not the bias-adjusted selection probability).
                topWeights[k] = probs[topExperts[k]];
                sum += topWeights[k];
            }

            // Nemotron normalizes expert weights.
            if (_expertWeightsNorm)
            {
                if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
                for (int k = 0; k < _numExpertsUsed; k++)
                    topWeights[k] /= sum;
            }

            // Apply global expert scale.
            if (_expertWeightsScale != 1.0f)
                for (int k = 0; k < _numExpertsUsed; k++)
                    topWeights[k] *= _expertWeightsScale;

            return (topExperts, topWeights);
        }

        // ====================================================================
        // Batched MoE decode under TP
        // ====================================================================

        /// <summary>
        /// Build the per-rank expert shard tables the batched MoE decode kernel
        /// needs. Runs after ShardNemotronMoeWeightsForTP has moved every expert
        /// into _tpQuantWeights.
        /// </summary>
        private void InitNemotronTpMoeBatchedDecode()
        {
            if (!IsTensorParallel || !IsGgmlBackend || _numExperts <= 0 || _numExpertsUsed <= 0)
                return;

            int tp = TpDegree;
            _tpExpertUpQW = new QuantizedWeight[Config.NumLayers][][];
            _tpExpertDownQW = new QuantizedWeight[Config.NumLayers][][];

            int ready = 0;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_layerTypes[l] != LayerType.FFN)
                    continue;

                var up = new QuantizedWeight[_numExperts][];
                var down = new QuantizedWeight[_numExperts][];
                bool ok = true;
                for (int e = 0; e < _numExperts; e++)
                {
                    if (!_tpQuantWeights.TryGetValue($"blk.{l}.ffn_up_exps.{e}.weight", out up[e])
                        || !_tpQuantWeights.TryGetValue($"blk.{l}.ffn_down_exps.{e}.weight", out down[e])
                        || up[e] == null || down[e] == null
                        || up[e].Length < tp || down[e].Length < tp)
                    {
                        ok = false;
                        break;
                    }
                }
                if (!ok)
                    continue;

                _tpExpertUpQW[l] = up;
                _tpExpertDownQW[l] = down;
                ready++;
            }

            if (ready == 0)
                return;

            _tpMoeUpPtrs = new IntPtr[tp][];
            _tpMoeDownPtrs = new IntPtr[tp][];
            for (int r = 0; r < tp; r++)
            {
                _tpMoeUpPtrs[r] = new IntPtr[_numExpertsUsed];
                _tpMoeDownPtrs[r] = new IntPtr[_numExpertsUsed];
            }

            _tpMoeBatchedDecodeReady = true;
            Console.WriteLine($"  Nemotron TP batched MoE decode: {ready} layer(s), "
                + $"1 dispatch per rank instead of {2 * _numExpertsUsed}.");
        }

        /// <summary>
        /// Decode (seqLen == 1) fast path for the routed experts: ONE GGML graph
        /// per rank covering all top-K experts - up, ReLU^2, down, route-weighted
        /// accumulate - over that rank's column/row shards. Returns one partial
        /// [1, expertOutDim] tensor per rank for the caller to AllReduce, or null
        /// when this layer is not eligible and the caller must fall back to the
        /// per-expert host loop.
        ///
        /// This is the same kernel the non-TP decode path uses. The fallback costs
        /// two device dispatches plus a host gather/scatter PER EXPERT PER RANK.
        /// </summary>
        private Tensor[] TryTpMoeBatchedExpertDecode(
            int layer, int seqLen, int expertOutDim, bool hasLatent,
            Tensor latentIn, Tensor[] normed, int[] selectedExperts, float[] routeWeightsAll)
        {
            if (!_tpMoeBatchedDecodeReady || seqLen != 1)
                return null;

            var upShards = _tpExpertUpQW[layer];
            var downShards = _tpExpertDownQW[layer];
            if (upShards == null || downShards == null)
                return null;

            int tp = TpDegree;
            int nUsed = _numExpertsUsed;

            // Validate and collect every shard pointer BEFORE allocating anything,
            // so a bail-out cannot leak a partially built result set.
            for (int r = 0; r < tp; r++)
            {
                for (int k = 0; k < nUsed; k++)
                {
                    int e = selectedExperts[k];
                    if ((uint)e >= (uint)_numExperts)
                        return null;
                    QuantizedWeight uq = upShards[e]?[r];
                    QuantizedWeight dq = downShards[e]?[r];
                    if (uq == null || dq == null || dq.Ne1 != expertOutDim)
                        return null;
                    _tpMoeUpPtrs[r][k] = uq.CacheKey;
                    _tpMoeDownPtrs[r][k] = dq.CacheKey;
                }
            }

            // Shapes are shard-uniform, so the first selected expert describes them all.
            var routeWeights = new float[nUsed];
            Array.Copy(routeWeightsAll, routeWeights, nUsed);

            var results = new Tensor[tp];
            // Fan the ranks out instead of walking them: GGML ops submit AND
            // synchronize inside one call, so a sequential rank loop runs GPU 0 to
            // completion before GPU 1 starts and the wall clock is the SUM of the
            // ranks rather than the MAX. RunPerRank pins each worker to its own
            // backend, which is also what lets the direct native calls below skip
            // their own rank selection.
            _tpGroup.RunPerRank(r =>
            {
                QuantizedWeight up0 = upShards[selectedExperts[0]][r];
                QuantizedWeight dn0 = downShards[selectedExperts[0]][r];

                var output = new Tensor(_tpGroup.GetAllocator(r), DType.Float32, 1, expertOutDim);
                GgmlBasicOps.MoEExpertsForward(
                    output, hasLatent ? latentIn : normed[r],
                    nUsed, _tpMoeUpPtrs[r], _tpMoeDownPtrs[r],
                    up0.GgmlType, up0.Ne0, up0.Ne1, up0.RawBytes,
                    dn0.GgmlType, dn0.Ne0, dn0.Ne1, dn0.RawBytes,
                    routeWeights);
                results[r] = output;
            });

            return results;
        }

        // ====================================================================
        // TP-aware Dispose
        // ====================================================================

        private void DisposeNemotronTpState()
        {
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
