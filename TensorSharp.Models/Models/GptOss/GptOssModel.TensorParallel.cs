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
// GptOssModel.TensorParallel.cs
//
// Tensor-parallel forward pass for GPT-OSS (pure MoE transformer).
//
// GptOss specifics handled here:
//   - Bias on ALL linear projections (QKV, output, expert gate/up/down, router)
//   - Attention sinks (per-head learned softmax bias, sliced per rank)
//   - Clamped SiLU GLU activation (alpha=1.702, limit=7.0, (up+1) variant)
//   - Alternating SWA (even layers) / full causal (odd layers)
//   - TopK-then-softmax routing
//   - Fused QKV with bias
//   - Every layer is MoE (no dense FFN)
// ============================================================================
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cuda;

namespace TensorSharp.Models
{
    public partial class GptOssModel
    {
        // Per-GPU KV caches: [layer][rank]
        private Tensor[][] _tpKvCacheK;
        private Tensor[][] _tpKvCacheV;
        private int _tpKvCacheCapacity;

        // Per-rank attention sinks: [layer][rank] — sliced from _layerSinks
        private float[][] _tpSinks;

        // ====================================================================
        // TP constraint validation
        // ====================================================================

        private void ValidateGptOssTpConstraints()
        {
            int tp = GlobalTpDegree;
            var errors = new List<string>();

            if (Config.NumHeads % tp != 0)
                errors.Add($"Attention heads ({Config.NumHeads}) not divisible by global TP degree ({tp})");
            if (Config.NumKVHeads % tp != 0)
                errors.Add($"KV heads ({Config.NumKVHeads}) not divisible by global TP degree ({tp})");
            // Expert parallelism partitions whole experts, so it constrains the
            // expert COUNT and places nothing on the per-expert FFN width. Only
            // the per-expert slicing path needs that.
            if (_expertFfnLength > 0 && _expertFfnLength % tp != 0 && !CanUseGgmlExpertParallelMoE())
                errors.Add($"Expert FFN length ({_expertFfnLength}) not divisible by global TP degree ({tp})");
            // Tensor parallelism runs on the multi-GPU backends: direct CUDA and
            // the GGML CUDA/Vulkan backends (one ggml backend per GPU).
            if (_backend is not (BackendType.Cuda or BackendType.GgmlCuda or BackendType.GgmlVulkan))
                errors.Add($"TP requires a multi-GPU backend (cuda, ggml-cuda, ggml-vulkan), got {_backend}");

            if (errors.Count > 0)
                throw new InvalidOperationException(
                    $"GptOss TP validation failed:\n  " + string.Join("\n  ", errors));

            Console.WriteLine($"  TP constraints validated: globalTp={tp}, localTp={TpDegree}, " +
                $"Heads={Config.NumHeads}, KVHeads={Config.NumKVHeads}, " +
                $"Experts={_numExperts}, ExpertFFN={_expertFfnLength}");
        }

        // ====================================================================
        // Weight sharding
        // ====================================================================

        private void ShardGptOssWeightsForTP()
        {
            // Attention output is row-parallel. The fused attn_qkv ([Q|K|V]) is
            // column-parallel but needs segment-aware sharding — a contiguous
            // split would give each rank whole segments instead of its
            // [Q_r|K_r|V_r] slice, corrupting the forward re-split.
            ShardWeightsForTensorParallelism(
                columnParallelPatterns: Array.Empty<string>(),
                rowParallelPatterns: new[] { "attn_output.weight" });

            int headDim = Config.HeadDim;
            int qDim = Config.NumHeads * headDim;
            int kDim = Config.NumKVHeads * headDim;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                string qkvName = $"blk.{l}.attn_qkv.weight";
                if (_quantWeights.ContainsKey(qkvName) || _weights.ContainsKey(qkvName))
                {
                    ShardConcatenatedColumnParallel(qkvName,
                        qDim,   // Q
                        kDim,   // K
                        kDim);  // V
                }
                else
                {
                    // Community/UD requants quantize Q, K and V with different
                    // types (unsloth's gpt-oss Q2_K ships IQ4_NL/IQ4_NL/Q5_0),
                    // which declines load-time QKV fusion — there is no
                    // attn_qkv tensor to shard, and the forward's fused lookup
                    // used to die with "TP column-parallel weight
                    // 'blk.0.attn_q.weight' not found". Mirror Qwen3.5's
                    // ShardFusedQkvForTP fallback: build each rank's
                    // [Q_r|K_r|V_r] shard straight from the separate weights
                    // (dequantised slice by slice) and register it under the
                    // fused name, so the TP forward keeps its single
                    // column-parallel QKV projection per layer.
                    string prefix = $"blk.{l}.";
                    ShardSeparateColumnParallel(qkvName,
                        new[] { prefix + "attn_q.weight", prefix + "attn_k.weight", prefix + "attn_v.weight" },
                        new[] { qDim, kDim, kDim });
                }
            }

            // MoE expert weights: whole-expert partitioning when the batched
            // dispatch can take it, per-expert slicing otherwise. FIRST, because
            // the bias sharding below asks which layout won.
            ShardGptOssMoeWeightsForTP();

            // Shard QKV bias (column-parallel: split along output dim)
            ShardGptOssBiasesForTP();

            Console.WriteLine($"  GptOss TP weight sharding complete ({TpDegree} GPUs).");
        }

        /// <summary>
        /// Shard bias tensors for column-parallel projections.
        /// Column-parallel biases split along the output dim (same as weights).
        /// Row-parallel biases (attn_output.bias, ffn_down_exps.E.bias) stay
        /// replicated because they're added AFTER the AllReduce.
        /// </summary>
        private void ShardGptOssBiasesForTP()
        {
            int tp = TpDegree;

            int headDim = Config.HeadDim;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                string prefix = $"blk.{l}.";

                // QKV bias: segment-aware column-parallel [Q|K|V] — must match
                // the attn_qkv weight regrouping. When load-time fusion was
                // declined (mixed quant types), the biases are still separate
                // attn_q/k/v.bias tensors: gather each rank's fused bias slice
                // from them, mirroring the weight fallback above.
                string qkvBias = prefix + "attn_qkv.bias";
                if (_weights.ContainsKey(qkvBias))
                {
                    ShardConcatenatedBiasColumnParallel(qkvBias,
                        Config.NumHeads * headDim,     // Q
                        Config.NumKVHeads * headDim,   // K
                        Config.NumKVHeads * headDim);  // V
                }
                else
                {
                    ShardSeparateBiasColumnParallel(qkvBias,
                        new[] { prefix + "attn_q.bias", prefix + "attn_k.bias", prefix + "attn_v.bias" },
                        new[] { Config.NumHeads * headDim, Config.NumKVHeads * headDim, Config.NumKVHeads * headDim });
                }

                // Expert gate_up biases: segment-aware column-parallel [gate|up].
                // Expert-parallel ranks own whole experts, so they take the
                // bias whole (sliced by expert at load, not by width).
                if (UsesExpertParallelMoE) continue;
                for (int e = 0; e < _numExperts; e++)
                    ShardFusedGateUpBiasColumnParallel(prefix + $"ffn_gate_up_exps.{e}.bias");
            }
        }

        private void ShardGptOssMoeWeightsForTP()
        {
            int tp = TpDegree;

            // First choice: partition WHOLE experts across ranks. That is a
            // zero-copy view and it keeps the batched mul_mat_id dispatch the
            // single-GPU path uses; slicing inside each expert forces the
            // per-(token, expert) host loop below, which measured 34x slower
            // than one GPU. See GptOssModel.TensorParallelGgmlMoE.cs.
            if (BuildGptOssExpertParallelShards())
                return;

            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                string prefix = $"blk.{layer}.";

                // Router: replicated (stays in _weights)

                for (int e = 0; e < _numExperts; e++)
                {
                    // Fused gate_up: segment-aware column-parallel [gate|up].
                    // A contiguous split would give rank 0 all of gate and
                    // rank 1 all of up, corrupting the SwiGLU inputs.
                    ShardFusedGateUpColumnParallel(prefix + $"ffn_gate_up_exps.{e}.weight");
                    // Down: row-parallel
                    ShardGptOssExpertRowParallel(prefix + $"ffn_down_exps.{e}.weight");
                }
            }
        }

        private void ShardGptOssExpertRowParallel(string weightName)
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
        // TP KV cache + sinks initialization
        // ====================================================================

        private void InitGptOssTpKVCache(int initialSeqLen, int maxSeqLen)
        {
            int tp = TpDegree;
            int numKVHeadsPerGpu = Config.NumKVHeads / GlobalTpDegree;
            int headDim = Config.HeadDim;
            DType kvDtype = _kvCacheDtype.ToDType();

            _maxContextLength = maxSeqLen;
            _tpKvCacheCapacity = initialSeqLen;
            _tpKvCacheK = new Tensor[Config.NumLayers][];
            _tpKvCacheV = new Tensor[Config.NumLayers][];

            for (int l = 0; l < Config.NumLayers; l++)
            {
                _tpKvCacheK[l] = new Tensor[tp];
                _tpKvCacheV[l] = new Tensor[tp];
                for (int r = 0; r < tp; r++)
                {
                    var alloc = _tpGroup.GetAllocator(r);
                    _tpKvCacheK[l][r] = new Tensor(alloc, kvDtype, numKVHeadsPerGpu, initialSeqLen, headDim);
                    _tpKvCacheV[l][r] = new Tensor(alloc, kvDtype, numKVHeadsPerGpu, initialSeqLen, headDim);
                    InitializeCacheTensor(_tpKvCacheK[l][r]);
                    InitializeCacheTensor(_tpKvCacheV[l][r]);
                }
            }

            // Slice attention sinks per rank.
            int numHeadsPerGpu = Config.NumHeads / GlobalTpDegree;
            _tpSinks = new float[Config.NumLayers][];
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (_layerSinks?[l] == null)
                    continue;

                // Each rank owns heads [rank*numHeadsPerGpu, (rank+1)*numHeadsPerGpu).
                // Sinks are per-head, so we store the full array and slice at use time.
                // Actually, store per-rank slices for efficiency.
                // We'll store as [tp][numHeadsPerGpu] flattened.
                _tpSinks[l] = _layerSinks[l]; // Keep full array, slice in attention
            }

            Console.WriteLine($"  GptOss TP KV cache initialized: {tp} GPUs, " +
                $"KV heads/GPU={numKVHeadsPerGpu}");
        }

        private void EnsureGptOssTpCacheCapacity(int requiredSeqLen)
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
            int numKVHeadsPerGpu = Config.NumKVHeads / GlobalTpDegree;
            int headDim = Config.HeadDim;
            DType kvDtype = _kvCacheDtype.ToDType();

            for (int l = 0; l < Config.NumLayers; l++)
            {
                for (int r = 0; r < tp; r++)
                {
                    var alloc = _tpGroup.GetAllocator(r);
                    var newK = new Tensor(alloc, kvDtype, numKVHeadsPerGpu, newCapacity, headDim);
                    var newV = new Tensor(alloc, kvDtype, numKVHeadsPerGpu, newCapacity, headDim);
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
            Console.WriteLine($"Expanded GptOss TP cache to {newCapacity} tokens ({tp} GPUs).");
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
            EnsureGptOssTpCacheCapacity(startPos + seqLen);

            long t1 = Stopwatch.GetTimestamp();
            Tensor hidden0 = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t1;

            // Fused whole-model tensor-parallel decode: one graph per rank per
            // token, summed at the two cut points every layer has. The per-op
            // chain below is the fallback, and it is ~12x slower - it is the
            // pre-fused-graph path.
            if (seqLen > 1 && TryGptOssFusedModelPrefillTP(hidden0, startPos, seqLen))
            {
                hidden0.Dispose();
                _cacheSeqLen += seqLen;
                _forwardCount++;
                _forwardSw.Stop();
                return _foldLogitsBuffer;
            }

            if (seqLen == 1 && TryGptOssFusedModelDecodeTP(hidden0, startPos))
            {
                hidden0.Dispose();
                _cacheSeqLen += seqLen;
                _forwardCount++;
                _forwardSw.Stop();
                return _foldLogitsBuffer;
            }

            Tensor[] hidden = BroadcastTensorToAllRanks(hidden0);

            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                hidden = GptOssTransformerBlockTP(hidden, layer, seqLen, startPos);
            }

            // Final norm + LM head on GPU 0.
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

        private Tensor[] GptOssTransformerBlockTP(Tensor[] hidden, int layer, int seqLen, int startPos)
        {
            string[] wn = _layerNames[layer];
            int tp = TpDegree;

            // 1. Attention norm (replicated).
            Tensor[] normed = TpRMSNorm(hidden, wn[0]);

            // 2. Column-parallel QKV + bias.
            Tensor[] qkvFused = TpColumnParallelLinearWithBias(normed[0], wn[1], wn[2]);
            for (int r = 0; r < tp; r++)
                normed[r].Dispose();

            // 3. Per-GPU attention (with sinks + SWA).
            bool isSWA = layer % 2 == 0;
            Tensor[] attnOut = GptOssAttentionTP(qkvFused, layer, seqLen, startPos, isSWA);

            // 4. Row-parallel output + bias + AllReduce.
            // Output bias is replicated (added after AllReduce).
            Tensor reducedAttn = TpRowParallelLinear(attnOut, wn[3]);
            for (int r = 0; r < tp; r++)
                attnOut[r].Dispose();

            // Add output bias (replicated).
            AddBiasToTensor(reducedAttn, wn[4]);

            // 5. Residual add.
            Tensor[] attnReplicated = BroadcastTensorToAllRanks(reducedAttn);
            TpResidualAdd(hidden, attnReplicated);
            for (int r = 1; r < tp; r++)
                attnReplicated[r].Dispose();
            reducedAttn.Dispose();

            // 6. Post-attention norm (replicated).
            Tensor[] postAttnNormed = TpRMSNorm(hidden, wn[5]);

            // 7. MoE FFN (tensor-parallel experts).
            Tensor[] moeOut = GptOssMoEBlockTP(postAttnNormed, layer, seqLen, wn);
            for (int r = 0; r < tp; r++)
                postAttnNormed[r].Dispose();

            // 8. Residual add.
            TpResidualAdd(hidden, moeOut);
            for (int r = 1; r < tp; r++)
                moeOut[r].Dispose();

            return hidden;
        }

        // ====================================================================
        // Column-parallel linear with bias
        // ====================================================================

        private Tensor[] TpColumnParallelLinearWithBias(Tensor input, string weightName, string biasName)
        {
            int tp = TpDegree;
            Tensor[] results = TpColumnParallelLinear(input, weightName);

            // Add per-rank bias shard.
            if (_tpWeights.TryGetValue(biasName, out var biasShards))
            {
                for (int r = 0; r < tp; r++)
                {
                    unsafe
                    {
                        float* rPtr = GetFloatPtr(results[r]);
                        float* bPtr = GetFloatPtr(biasShards[r]);
                        int seqLen = (int)results[r].Sizes[0];
                        int outDim = (int)results[r].Sizes[1];
                        int biasDim = (int)biasShards[r].ElementCount();
                        int dim = Math.Min(outDim, biasDim);
                        for (int s = 0; s < seqLen; s++)
                        {
                            float* row = rPtr + s * outDim;
                            for (int i = 0; i < dim; i++)
                                row[i] += bPtr[i];
                        }
                    }
                }
            }

            return results;
        }

        private void AddBiasToTensor(Tensor tensor, string biasName)
        {
            if (!_weights.TryGetValue(biasName, out var bias))
                return;

            unsafe
            {
                float* rPtr = GetFloatPtr(tensor);
                float* bPtr = GetFloatPtr(bias);
                int seqLen = (int)tensor.Sizes[0];
                int outDim = (int)tensor.Sizes[1];
                int biasDim = (int)bias.ElementCount();
                int dim = Math.Min(outDim, biasDim);
                for (int s = 0; s < seqLen; s++)
                {
                    float* row = rPtr + s * outDim;
                    for (int i = 0; i < dim; i++)
                        row[i] += bPtr[i];
                }
            }
        }

        // ====================================================================
        // Attention under TP (with sinks + SWA)
        // ====================================================================

        private Tensor[] GptOssAttentionTP(Tensor[] qkvFused, int layer, int seqLen, int startPos, bool isSWA)
        {
            int tp = TpDegree;
            int numHeadsPerGpu = Config.NumHeads / GlobalTpDegree;
            int numKVHeadsPerGpu = Config.NumKVHeads / GlobalTpDegree;
            int headDim = Config.HeadDim;
            int qDimPerGpu = numHeadsPerGpu * headDim;
            int kDimPerGpu = numKVHeadsPerGpu * headDim;
            int totalSeqLen = startPos + seqLen;
            float scale = 1.0f / MathF.Sqrt(headDim);

            // Use _layerSinks directly — _tpSinks was populated during
            // InitGptOssTpKVCache which runs BEFORE _layerSinks is loaded,
            // so the copied references are all null.
            float[] fullSinks = _layerSinks?[layer];

            var results = new Tensor[tp];

            for (int r = 0; r < tp; r++)
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

                // RoPE (NeoX with YaRN scaling).
                if (seqLen == 1)
                {
                    qTensor = ApplyGptOssRoPEDecode(qTensor, numHeadsPerGpu, headDim, startPos);
                    kTensor = ApplyGptOssRoPEDecode(kTensor, numKVHeadsPerGpu, headDim, startPos);
                }
                else
                {
                    qTensor = ApplyGptOssRoPEPrefill(qTensor, numHeadsPerGpu, headDim, seqLen, startPos);
                    kTensor = ApplyGptOssRoPEPrefill(kTensor, numKVHeadsPerGpu, headDim, seqLen, startPos);
                }

                // Slice sinks for this rank's heads. Local rank r maps to global
                // rank (TpRankOffset + r), which owns heads
                // [globalRank*numHeadsPerGpu, +numHeadsPerGpu) of the full array.
                float[] rankSinks = null;
                if (fullSinks != null)
                {
                    int globalRank = TpRankOffset + r;
                    rankSinks = new float[numHeadsPerGpu];
                    Array.Copy(fullSinks, globalRank * numHeadsPerGpu, rankSinks, 0, numHeadsPerGpu);
                }

                if (seqLen == 1)
                {
                    CopyToCacheDecode(_tpKvCacheK[layer][r], kTensor, _tpKvCacheV[layer][r], vTensor,
                        numKVHeadsPerGpu, headDim, startPos);
                    kTensor.Dispose();
                    vTensor.Dispose();

                    int attendLen = isSWA ? Math.Min(totalSeqLen, _slidingWindow) : totalSeqLen;
                    int attendStart = totalSeqLen - attendLen;

                    var attnResult = new Tensor(alloc, DType.Float32, 1, numHeadsPerGpu * headDim);

                    if (rankSinks != null)
                    {
                        AttentionDecodeWithSinksTP(qTensor, _tpKvCacheK[layer][r], _tpKvCacheV[layer][r],
                            attnResult, numHeadsPerGpu, numKVHeadsPerGpu, headDim,
                            attendStart, totalSeqLen, scale, rankSinks);
                    }
                    else
                    {
                        AttentionDecodePureCS(qTensor, _tpKvCacheK[layer][r], _tpKvCacheV[layer][r],
                            attnResult, numHeadsPerGpu, numKVHeadsPerGpu, headDim, totalSeqLen, scale);
                    }
                    qTensor.Dispose();
                    results[r] = attnResult;
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

                    // Apply causal mask + SWA.
                    int windowSize = isSWA ? _slidingWindow : 0;
                    ApplyGptOssCausalMask(scores, seqLen, totalSeqLen, windowSize);

                    // Softmax with sinks.
                    if (rankSinks != null)
                        ApplySoftmaxWithSinksTP(scores, numHeadsPerGpu, seqLen, totalSeqLen, rankSinks);
                    else
                        Ops.Softmax(scores, scores);

                    var attnOut = new Tensor(alloc, DType.Float32, numHeadsPerGpu, seqLen, headDim);
                    Ops.AddmmBatch(attnOut, 0, attnOut, 1.0f, scores, vExpanded);
                    scores.Dispose();
                    vExpanded.Dispose();

                    Tensor flatOutput = ReshapeFromHeads(attnOut, numHeadsPerGpu, seqLen, headDim);
                    attnOut.Dispose();
                    results[r] = flatOutput;
                }
            }

            return results;
        }

        private Tensor ApplyGptOssRoPEDecode(Tensor data, int numHeads, int headDim, int position)
        {
            // Delegate to the shared non-TP RoPE so the full YaRN extension
            // (per-frequency ramp blend + mscale amplitude) is applied — the
            // previous hand-rolled loop did only flat position-interpolation,
            // which corrupts attention whenever RopeScale != 1.
            return ApplyRoPEInPlace(data, numHeads, headDim, 1, position);
        }

        private Tensor ApplyGptOssRoPEPrefill(Tensor data, int numHeads, int headDim, int seqLen, int startPos)
        {
            return ApplyRoPEInPlace(data, numHeads, headDim, seqLen, startPos);
        }

        private void ApplyGptOssCausalMask(Tensor scores, int seqLen, int totalSeqLen, int windowSize)
        {
            if (windowSize > 0)
            {
                // SWA: mask positions outside the window
                unsafe
                {
                    float* ptr = GetFloatPtr(scores);
                    int numHeads = (int)scores.Sizes[0];
                    for (int h = 0; h < numHeads; h++)
                    {
                        for (int q = 0; q < seqLen; q++)
                        {
                            int qPos = totalSeqLen - seqLen + q;
                            float* row = ptr + (h * seqLen + q) * totalSeqLen;
                            for (int k = 0; k < totalSeqLen; k++)
                            {
                                if (k > qPos || (windowSize > 0 && k < qPos - windowSize + 1))
                                    row[k] = float.NegativeInfinity;
                            }
                        }
                    }
                }
                InvalidateTensorDeviceCache(scores);
            }
            else
            {
                Ops.AddCausalMask(scores, seqLen, totalSeqLen - seqLen, float.NegativeInfinity);
            }
        }

        /// <summary>
        /// Softmax with attention sinks for TP. Each rank handles its own head subset.
        /// </summary>
        private unsafe void ApplySoftmaxWithSinksTP(Tensor scores, int numHeads, int seqLen, int totalSeqLen, float[] sinks)
        {
            float* ptr = GetFloatPtr(scores);
            for (int h = 0; h < numHeads; h++)
            {
                float sink = sinks[h];
                for (int q = 0; q < seqLen; q++)
                {
                    float* row = ptr + (h * seqLen + q) * totalSeqLen;

                    // Find max (including sink)
                    float max = sink;
                    for (int k = 0; k < totalSeqLen; k++)
                        if (row[k] > max) max = row[k];

                    // Exp and sum (including sink)
                    float sinkExp = MathF.Exp(sink - max);
                    float sum = sinkExp;
                    for (int k = 0; k < totalSeqLen; k++)
                    {
                        row[k] = MathF.Exp(row[k] - max);
                        sum += row[k];
                    }

                    // Normalize
                    float invSum = 1.0f / sum;
                    for (int k = 0; k < totalSeqLen; k++)
                        row[k] *= invSum;
                }
            }
            InvalidateTensorDeviceCache(scores);
        }

        /// <summary>
        /// Decode attention with sinks for TP (per-rank head subset).
        /// </summary>
        private void AttentionDecodeWithSinksTP(
            Tensor q, Tensor kCache, Tensor vCache, Tensor result,
            int numHeads, int numKVHeads, int headDim,
            int attendStart, int attendEnd, float scale, float[] sinks)
        {
            AttentionDecodeWithSinksTPHost(q, kCache, vCache, result,
                numHeads, numKVHeads, headDim, attendStart, attendEnd, scale, sinks);
            InvalidateTensorDeviceCache(result);
        }

        /// <summary>
        /// Host body of <see cref="AttentionDecodeWithSinksTP"/> (no model state). The score
        /// row spans the attended range - the whole context on a full-attention layer - so it
        /// is stack memory only up to <see cref="CpuScratch.StackFloatLimit"/> keys.
        /// </summary>
        internal static unsafe void AttentionDecodeWithSinksTPHost(
            Tensor q, Tensor kCache, Tensor vCache, Tensor result,
            int numHeads, int numKVHeads, int headDim,
            int attendStart, int attendEnd, float scale, float[] sinks)
        {
            float* qPtr = GetFloatPtr(q);
            float* kPtr = GetFloatPtrForRead(kCache);
            float* vPtr = GetFloatPtrForRead(vCache);
            float* rPtr = GetFloatPtr(result);

            int groupSize = numHeads / numKVHeads;
            int kvSeqLen = (int)kCache.Sizes[1];
            // Every head overwrites all scores, so reuse one buffer across heads.
            int numScores = attendEnd - attendStart;
            float[] rented = null;
            Span<float> scores = numScores <= CpuScratch.StackFloatLimit
                ? stackalloc float[numScores]
                : (rented = CpuScratch.RentFloats(numScores)).AsSpan(0, numScores);
            try
            {
                for (int h = 0; h < numHeads; h++)
                {
                    int kvH = h / groupSize;
                    float sink = sinks[h];
                    float* qHead = qPtr + h * headDim;
                    float* rHead = rPtr + h * headDim;

                    // Compute scores
                    float maxScore = sink;
                    for (int k = attendStart; k < attendEnd; k++)
                    {
                        float* kHead = kPtr + (kvH * kvSeqLen + k) * headDim;
                        float dot = 0;
                        for (int d = 0; d < headDim; d++)
                            dot += qHead[d] * kHead[d];
                        dot *= scale;
                        scores[k - attendStart] = dot;
                        if (dot > maxScore) maxScore = dot;
                    }

                    // Softmax with sink
                    float sinkExp = MathF.Exp(sink - maxScore);
                    float sum = sinkExp;
                    for (int k = 0; k < scores.Length; k++)
                    {
                        scores[k] = MathF.Exp(scores[k] - maxScore);
                        sum += scores[k];
                    }
                    float invSum = 1.0f / sum;

                    // Weighted sum of V
                    for (int d = 0; d < headDim; d++)
                        rHead[d] = 0;

                    for (int k = attendStart; k < attendEnd; k++)
                    {
                        float w = scores[k - attendStart] * invSum;
                        float* vHead = vPtr + (kvH * kvSeqLen + k) * headDim;
                        for (int d = 0; d < headDim; d++)
                            rHead[d] += w * vHead[d];
                    }
                }
            }
            finally
            {
                CpuScratch.Return(rented);
            }
        }

        // ====================================================================
        // MoE block under TP (clamped SiLU GLU)
        // ====================================================================

        private Tensor[] GptOssMoEBlockTP(Tensor[] normed, int layer, int seqLen, string[] wn)
        {
            int tp = TpDegree;
            int hiddenSize = Config.HiddenSize;
            string prefix = $"blk.{layer}.";

            int numExperts = _numExperts;
            int nUsed = _numExpertsUsed;
            float invG = 1f / GlobalTpDegree;
            var results = new Tensor[tp];

            // Router (replicated — identical routing on all ranks). Compute it ONCE
            // on rank 0's GPU, where the (unsharded) router weight lives. normed[*]
            // are replicas of the same hidden state, so the routing decision is
            // identical on every rank. Computing it per-rank (matmul of rank-r input
            // against the GPU-0 router weight) is a cross-GPU access → CUDA error 700
            // on hardware without peer access.
            Tensor routerLogitsT = LinearForward(normed[0], wn[6]);
            AddBiasToTensor(routerLogitsT, wn[7]);
            // Row-major [seqLen, numExperts]: each token routes independently.
            float[] routePtr = TensorToFloatArray(routerLogitsT);
            routerLogitsT.Dispose();

            // Per-token top-k selection, then group assignments by expert (identical
            // on every rank). SelectGptOssTopKExperts treats its whole input as ONE
            // token's logits, so feed it exactly one token's numExperts-length row —
            // feeding the flattened batch made it return out-of-range expert indices.
            int totalAssignments = seqLen * nUsed;
            var selectedExperts = new int[totalAssignments];
            var routeWeights = new float[totalAssignments];
            var tokenLogits = new float[numExperts];
            for (int s = 0; s < seqLen; s++)
            {
                Array.Copy(routePtr, s * numExperts, tokenLogits, 0, numExperts);
                var (te, tw) = SelectGptOssTopKExperts(tokenLogits);
                for (int k = 0; k < nUsed; k++)
                {
                    selectedExperts[s * nUsed + k] = te[k];
                    routeWeights[s * nUsed + k] = tw[k];
                }
            }

            // Bucket token assignments by expert so each expert runs a single fat
            // matmul over its assigned tokens (mirrors the non-TP MoEForwardBatched).
            var expertCounts = new int[numExperts];
            for (int a = 0; a < totalAssignments; a++)
                expertCounts[selectedExperts[a]]++;
            var expertOffsets = new int[numExperts];
            for (int e = 1; e < numExperts; e++)
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
                    weightMap[pos] = routeWeights[s * nUsed + k];
                }

            // Expert-parallel: each rank runs ONE batched dispatch over the
            // experts it owns, instead of walking all 32 through host staging.
            if (UsesExpertParallelMoE)
            {
                if (TryGptOssMoEExpertParallel(normed, results, selectedExperts, routeWeights, layer, seqLen))
                {
                    _tpGroup.AllReduce(results);
                    return results;
                }
                // The per-expert loop below cannot stand in: whole-expert
                // partitioning means there are no per-(layer, expert) shards for
                // it to read. Say that, rather than fault several frames deeper
                // on a missing weight.
                throw new InvalidOperationException(
                    $"GPT-OSS layer {layer}: the batched expert-parallel MoE dispatch failed and the per-expert " +
                    "tensor-parallel path has no shards to fall back to. Re-run with " +
                    "TS_GPTOSS_TP_EXPERT_PARALLEL=0 to use per-expert slicing instead.");
            }

            long rowBytes = (long)hiddenSize * sizeof(float);

            for (int r = 0; r < tp; r++)
            {
                var alloc = _tpGroup.GetAllocator(r);
                var localInput = normed[r];

                // Partial per-rank output (row-parallel down is AllReduced below).
                // Gather/scatter run host-side (explicit token offsets), so the row
                // accumulation is unambiguous; EnsureDeviceCurrent pushes the result
                // to the device before the P2P AllReduce (which reads device ptrs).
                var output = new Tensor(alloc, DType.Float32, seqLen, hiddenSize);

                unsafe
                {
                    float* inPtr = GetFloatPtr(localInput);   // rank-r hidden state (host)
                    float* outPtr = GetFloatPtr(output);      // host accumulator
                    for (long z = 0; z < (long)seqLen * hiddenSize; z++)
                        outPtr[z] = 0f;

                    for (int e = 0; e < numExperts; e++)
                    {
                        int count = expertCounts[e];
                        if (count == 0) continue;
                        int offset = expertOffsets[e];

                        string gateUpKey = prefix + $"ffn_gate_up_exps.{e}.weight";
                        string gateUpBiasKey = prefix + $"ffn_gate_up_exps.{e}.bias";
                        string downKey = prefix + $"ffn_down_exps.{e}.weight";
                        string downBiasKey = prefix + $"ffn_down_exps.{e}.bias";

                        // Gather this expert's tokens into a [count, hidden] batch.
                        var batchInput = new Tensor(alloc, DType.Float32, count, hiddenSize);
                        float* bPtr = GetFloatPtr(batchInput);
                        for (int i = 0; i < count; i++)
                        {
                            int tokenIdx = tokenMap[offset + i];
                            Buffer.MemoryCopy(inPtr + (long)tokenIdx * hiddenSize,
                                bPtr + (long)i * hiddenSize, rowBytes, rowBytes);
                        }

                        // Column-parallel gate_up + per-rank bias shard.
                        Tensor gateUp = TpExpertLinear(batchInput, gateUpKey, r, count);
                        batchInput.Dispose();
                        AddTpBiasToTensor(gateUp, gateUpBiasKey, r);

                        // Clamped SiLU GLU activation.
                        int halfDim = (int)(gateUp.Sizes[1] / 2);
                        using (var gView = gateUp.Narrow(1, 0, halfDim))
                        using (var uView = gateUp.Narrow(1, halfDim, halfDim))
                        {
                            Tensor gate = Ops.NewContiguous(gView);
                            Tensor up = Ops.NewContiguous(uView);
                            gateUp.Dispose();

                            ApplyClampedSiLUGlu(gate, up);
                            up.Dispose();

                            // Row-parallel down (partial result, AllReduced later).
                            Tensor downOut = TpExpertLinear(gate, downKey, r, count);
                            gate.Dispose();

                            // Scatter weighted rows back into their tokens' output rows.
                            // The down bias is replicated (full hidden): add weight*bias
                            // scaled by 1/GlobalTpDegree so the upcoming AllReduce sum
                            // over all ranks reproduces weight*bias exactly once.
                            float* dPtr = GetFloatPtr(downOut);
                            float* dbPtr = _weights.TryGetValue(downBiasKey, out var downBiasT)
                                ? GetFloatPtr(downBiasT) : null;
                            for (int i = 0; i < count; i++)
                            {
                                int tokenIdx = tokenMap[offset + i];
                                float w = weightMap[offset + i];
                                float* dst = outPtr + (long)tokenIdx * hiddenSize;
                                float* srcRow = dPtr + (long)i * hiddenSize;
                                if (dbPtr != null)
                                {
                                    float wb = w * invG;
                                    for (int d = 0; d < hiddenSize; d++)
                                        dst[d] += w * srcRow[d] + wb * dbPtr[d];
                                }
                                else
                                {
                                    for (int d = 0; d < hiddenSize; d++)
                                        dst[d] += w * srcRow[d];
                                }
                            }
                            downOut.Dispose();
                        }
                    }
                }

                // Host buffer is authoritative — push to device for the P2P AllReduce.
                output.EnsureDeviceCurrent();
                results[r] = output;
            }

            // AllReduce across ranks (sums the row-parallel down partials + biases).
            _tpGroup.AllReduce(results);

            return results;
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

        private void AddTpBiasToTensor(Tensor tensor, string biasName, int rank)
        {
            if (!_tpWeights.TryGetValue(biasName, out var biasShards))
                return;

            var bias = biasShards[rank];
            unsafe
            {
                float* rPtr = GetFloatPtr(tensor);
                float* bPtr = GetFloatPtr(bias);
                int seqLen = (int)tensor.Sizes[0];
                int outDim = (int)tensor.Sizes[1];
                int biasDim = (int)bias.ElementCount();
                int dim = Math.Min(outDim, biasDim);
                for (int s = 0; s < seqLen; s++)
                {
                    float* row = rPtr + s * outDim;
                    for (int i = 0; i < dim; i++)
                        row[i] += bPtr[i];
                }
            }
        }

        /// <summary>
        /// Add a replicated (unsharded) bias scaled by <paramref name="scale"/> to
        /// a [seqLen, outDim] tensor, broadcasting the [outDim] bias across rows.
        /// Used for the MoE expert down bias, which is replicated and must be added
        /// once across the row-parallel AllReduce: callers pass weight/GlobalTpDegree
        /// so the AllReduce's sum over all ranks reproduces weight*bias exactly once.
        /// </summary>
        private unsafe void AddReplicatedBiasScaled(Tensor tensor, string biasName, float scale)
        {
            if (!_weights.TryGetValue(biasName, out var bias))
                return;

            float* rPtr = GetFloatPtr(tensor);
            float* bPtr = GetFloatPtr(bias);
            int seqLen = (int)tensor.Sizes[0];
            int outDim = (int)tensor.Sizes[1];
            int biasDim = (int)bias.ElementCount();
            int dim = Math.Min(outDim, biasDim);
            for (int s = 0; s < seqLen; s++)
            {
                float* row = rPtr + s * outDim;
                for (int i = 0; i < dim; i++)
                    row[i] += scale * bPtr[i];
            }
        }

        /// <summary>
        /// Clamped SiLU GLU activation (GptOss "SwiGLU OAI" variant):
        ///   gate = clamp(gate, -inf, 7.0)
        ///   up = clamp(up, -7.0, 7.0)
        ///   out = gate * sigmoid(1.702 * gate) * (up + 1)
        /// Result is stored in-place in the gate tensor.
        /// </summary>
        private unsafe void ApplyClampedSiLUGlu(Tensor gate, Tensor up)
        {
            float* gPtr = GetFloatPtr(gate);
            float* uPtr = GetFloatPtr(up);
            int n = (int)gate.ElementCount();

            // Reuse the existing SIMD-optimized implementation.
            ApplySwiGluOaiInPlace(gPtr, uPtr, n);
            InvalidateTensorDeviceCache(gate);
        }

        private (int[] experts, float[] weights) SelectGptOssTopKExperts(float[] routerLogits)
        {
            int numExperts = routerLogits.Length;
            var indices = new int[numExperts];
            for (int i = 0; i < numExperts; i++) indices[i] = i;
            Array.Sort(indices, (a, b) => routerLogits[b].CompareTo(routerLogits[a]));

            var topExperts = new int[_numExpertsUsed];
            var topWeights = new float[_numExpertsUsed];

            // TopK-then-softmax: select top-K, then softmax over selected
            float maxLogit = float.NegativeInfinity;
            for (int k = 0; k < _numExpertsUsed; k++)
            {
                topExperts[k] = indices[k];
                topWeights[k] = routerLogits[indices[k]];
                if (topWeights[k] > maxLogit) maxLogit = topWeights[k];
            }

            float sum = 0;
            for (int k = 0; k < _numExpertsUsed; k++)
            {
                topWeights[k] = MathF.Exp(topWeights[k] - maxLogit);
                sum += topWeights[k];
            }

            if (sum > 0)
                for (int k = 0; k < _numExpertsUsed; k++)
                    topWeights[k] /= sum;

            return (topExperts, topWeights);
        }

        // ====================================================================
        // TP-aware Dispose
        // ====================================================================

        private void DisposeGptOssTpState()
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
