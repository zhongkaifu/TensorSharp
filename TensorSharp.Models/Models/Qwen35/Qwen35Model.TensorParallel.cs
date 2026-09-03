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
// Qwen35Model.TensorParallel.cs
//
// Tensor-parallel forward pass for the Qwen3.5 hybrid model. Splits:
//   * FullAttention layers: Q/K/V heads + output projection (Megatron pattern)
//   * GatedDeltaNet layers: V heads (block-cyclic on K mapping) + ssm_out
//   * Dense FFN layers: gate_up (column) + down (row)
//   * MoE layers: tensor-parallel experts (1/tp slice of every expert)
//
// The GDN recurrent state (delta state + conv state) is per-rank and never
// communicated — each rank owns the state for its own V heads. The CUDA-native
// GDN kernel is the only supported execution path under TP.
// ============================================================================
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    public partial class Qwen35Model
    {
        // ====================================================================
        // Per-rank KV caches for full-attention layers: [layer][rank]
        // ====================================================================
        private Tensor[][] _tpKvCacheK;
        private Tensor[][] _tpKvCacheV;
        private int _tpKvCacheCapacity;

        // ====================================================================
        // Per-rank GDN recurrent state: [layer][rank]
        // ====================================================================
        private Tensor[][] _tpDeltaState;    // [layer][rank]: [nV/tp, headVDim, headKDim]
        private Tensor[][] _tpConvState;     // [layer][rank]: [convKernel-1, qkvDim/tp]
        private int[] _tpConvWriteIdx;       // [layer] — identical on all ranks

        // ====================================================================
        // Block-cyclic head mapping
        // ====================================================================

        /// <summary>
        /// Compute the block-cyclic V-head assignment for a given rank.
        /// Rank r owns K heads [r*nK/tp, (r+1)*nK/tp) and the V heads that
        /// map to them: { h : (h % nK) ∈ that range }.
        /// Returns the sorted list of global V-head indices owned by this rank.
        /// </summary>
        internal static int[] ComputeBlockCyclicVHeads(int rank, int tp, int numVHeads, int numKHeads)
        {
            int kBlockWidth = numKHeads / tp;
            int kStart = rank * kBlockWidth;
            int kEnd = kStart + kBlockWidth;

            var heads = new List<int>();
            for (int h = 0; h < numVHeads; h++)
            {
                int kHead = h % numKHeads;
                if (kHead >= kStart && kHead < kEnd)
                    heads.Add(h);
            }
            return heads.ToArray();
        }

        /// <summary>
        /// Compute the K-head indices owned by a given rank (contiguous block).
        /// </summary>
        internal static int[] ComputeKHeadsForRank(int rank, int tp, int numKHeads)
        {
            int blockWidth = numKHeads / tp;
            int start = rank * blockWidth;
            var heads = new int[blockWidth];
            for (int i = 0; i < blockWidth; i++)
                heads[i] = start + i;
            return heads;
        }

        /// <summary>
        /// Build the permutation that maps local V-head order (sorted by global
        /// head index) to the block-cyclic order expected by the ssm_out weight.
        /// The ssm_out weight's input dimension is indexed by (v_head, v_dim),
        /// so its column ordering must match the V-head shard.
        /// </summary>
        internal static int[] BuildVHeadPermutation(int rank, int tp, int numVHeads, int numKHeads, int headVDim)
        {
            int[] globalHeads = ComputeBlockCyclicVHeads(rank, tp, numVHeads, numKHeads);
            int localVHeads = globalHeads.Length;
            var perm = new int[localVHeads * headVDim];
            for (int lh = 0; lh < localVHeads; lh++)
            {
                int gh = globalHeads[lh];
                for (int d = 0; d < headVDim; d++)
                    perm[lh * headVDim + d] = gh * headVDim + d;
            }
            return perm;
        }

        // ====================================================================
        // TP phase accounting
        // ====================================================================
        //
        // The generic Linear/Attention/Norm buckets lump the whole TP forward
        // into "Linear", which is useless for deciding what to fuse next. These
        // split it by block so the dispatch-bound parts are visible.
        private long _tpGdnTicks, _tpSsmOutTicks, _tpAttnBlockTicks;
        private long _tpMoeDispatchTicks, _tpSharedExpertTicks, _tpRouterTicks;
        private long _tpAllReduceTicks, _tpResidualTicks;

        private void PrintTpTimingStats()
        {
            if (!IsTensorParallel || _forwardCount == 0)
                return;
            double ms = 1000.0 / Stopwatch.Frequency;
            Console.WriteLine("  Tensor-parallel phases (per rank, wall clock):");
            Console.WriteLine($"    GDN block:       {_tpGdnTicks * ms:F0} ms");
            Console.WriteLine($"    ssm_out+AR:      {_tpSsmOutTicks * ms:F0} ms");
            Console.WriteLine($"    attention block: {_tpAttnBlockTicks * ms:F0} ms");
            Console.WriteLine($"    MoE router:      {_tpRouterTicks * ms:F0} ms");
            Console.WriteLine($"    MoE experts:     {_tpMoeDispatchTicks * ms:F0} ms");
            Console.WriteLine($"    shared expert:   {_tpSharedExpertTicks * ms:F0} ms");
            Console.WriteLine($"    AllReduce:       {_tpAllReduceTicks * ms:F0} ms");
            Console.WriteLine($"    residual adds:   {_tpResidualTicks * ms:F0} ms");
        }

        // ====================================================================
        // Layer trace (TS_QWEN35_LAYER_TRACE=1)
        // ====================================================================
        //
        // Prints a per-layer summary of the residual stream for the FIRST forward
        // only, from both the single-GPU and the tensor-parallel loops. Diffing
        // the two runs is what localizes a TP divergence to a layer — without it
        // the only observable is the sampled text, which says nothing about where
        // the two paths parted.
        private static readonly bool LayerTraceEnabled =
            string.Equals(Environment.GetEnvironmentVariable("TS_QWEN35_LAYER_TRACE"), "1", StringComparison.Ordinal);
        private int _layerTraceForwards;

        private unsafe void TraceLayer(Tensor hidden, int layer, string tag)
        {
            if (!LayerTraceEnabled || _layerTraceForwards > 0 || hidden == null)
                return;

            long n = hidden.ElementCount();
            float* p = GetFloatPtr(hidden);
            double sum = 0, absSum = 0;
            float max = float.NegativeInfinity;
            for (long i = 0; i < n; i++)
            {
                double v = p[i];
                sum += v;
                absSum += Math.Abs(v);
                if (p[i] > max) max = p[i];
            }
            Console.WriteLine($"[TRACE{tag}] layer={layer} kind={(_isRecurrent[layer] ? "gdn" : "attn")} " +
                $"n={n} sum={sum:F3} abs={absSum:F3} max={max:F5}");
        }

        // ====================================================================
        // TP constraint validation
        // ====================================================================

        private void ValidateTpConstraints()
        {
            int tp = GlobalTpDegree;
            var errors = new List<string>();

            if (_numKHeads % tp != 0)
                errors.Add($"GDN K heads ({_numKHeads}) not divisible by TP degree ({tp})");
            if (_numVHeads % tp != 0)
                errors.Add($"GDN V heads ({_numVHeads}) not divisible by TP degree ({tp})");
            if (Config.NumHeads % tp != 0)
                errors.Add($"Attention heads ({Config.NumHeads}) not divisible by TP degree ({tp})");
            if (Config.NumKVHeads % tp != 0)
                errors.Add($"Attention KV heads ({Config.NumKVHeads}) not divisible by TP degree ({tp})");
            if (Config.IntermediateSize > 0 && Config.IntermediateSize % tp != 0)
                errors.Add($"Intermediate size ({Config.IntermediateSize}) not divisible by TP degree ({tp})");
            // Expert parallelism partitions whole experts, so it constrains the
            // expert COUNT and places none on the per-expert FFN width. Only the
            // per-expert slicing path needs that.
            bool expertParallel = CanUseGgmlExpertParallelMoE();
            if (_numExperts > 0 && !expertParallel && _expertFfnLength % tp != 0)
                errors.Add($"Expert FFN length ({_expertFfnLength}) not divisible by TP degree ({tp})");
            if (_numExperts > 0 && _sharedExpertFfnLength > 0 && _sharedExpertFfnLength % tp != 0)
                errors.Add($"Shared expert FFN length ({_sharedExpertFfnLength}) not divisible by TP degree ({tp})");
            if (_numVHeads % _numKHeads != 0)
                errors.Add($"Model invariant violated: V heads ({_numVHeads}) not divisible by K heads ({_numKHeads})");

            // The per-rank GatedDeltaNet needs a packed-input kernel that keeps the
            // conv window and the delta state on the device: direct CUDA has
            // ts_qwen35_gdn_*, GGML has TSGgml_Qwen35GdnLayerTP. No other backend
            // has one, and the per-token host loop is not a viable substitute
            // (it round-trips the whole delta state per layer per token).
            if (_backend != BackendType.Cuda && !IsGgmlBackend)
                errors.Add($"Qwen3.5/3.6 tensor parallelism requires the direct CUDA or a GGML backend, got {_backend}. " +
                           "Its per-rank GatedDeltaNet needs a device-resident packed-GDN kernel, which only those " +
                           "backends provide. Run multi-GPU on --backend cuda / --backend ggml_cuda, or single-GPU here.");

            // The GGML packed GDN builds one ggml_gated_delta_net node, whose
            // packed state layout requires a square per-head state.
            if (IsGgmlBackend && _headKDim != _headVDim)
                errors.Add($"GGML tensor parallelism requires headKDim == headVDim for the GatedDeltaNet " +
                           $"({_headKDim} vs {_headVDim}); use --backend cuda --tp {tp}.");

            if (errors.Count > 0)
                throw new InvalidOperationException(
                    $"Qwen3.5 TP validation failed:\n  " + string.Join("\n  ", errors));

            Console.WriteLine($"  TP constraints validated: tp={tp} (local={TpDegree}), " +
                $"GDN heads V={_numVHeads}/K={_numKHeads}, " +
                $"Attn heads Q={Config.NumHeads}/KV={Config.NumKVHeads}");
        }

        // ====================================================================
        // Weight sharding
        // ====================================================================

        private void ShardQwen35WeightsForTP()
        {
            int tp = TpDegree;
            int globalTp = GlobalTpDegree;

            // --- Full-attention layers: column/row parallel ---
            // attn_output.weight is row-parallel (split input dim).
            // attn_qkv.weight is column-parallel, but the fused weight is a plain
            // [Q(+gate) | K | V] concatenation along the output dim. A contiguous
            // split would hand rank 0 mostly Q rows and no K/V, so it is sharded
            // separately with head-aware regrouping (see ShardFusedQkvForTP).
            ShardWeightsForTensorParallelism(
                columnParallelPatterns: Array.Empty<string>(),
                rowParallelPatterns: new[] { "attn_output.weight" });

            for (int layer = 0; layer < TotalLayerCount; layer++)
            {
                if (_isRecurrent[layer])
                    continue; // recurrent layers pack Q/K into ssm_in_proj instead
                ShardFusedQkvForTP($"blk.{layer}.attn_qkv.weight");
            }

            // --- GDN layers: segmented sharding ---
            // Run BEFORE the dense FFN sharding so the large F32 recurrent
            // input packs (dequantized during fusion for mixed-quant models)
            // are sharded and their full-size sources freed early, reducing
            // peak host memory during the subsequent gate_up sharding.
            ShardGdnWeightsForTP();

            // --- Dense FFN layers: column/row parallel ---
            // ffn_down.weight is row-parallel. ffn_gate_up.weight is a plain
            // [gate | up] concatenation, so it needs the same segment-aware
            // regrouping as QKV: a contiguous split would give rank 0 all of
            // gate and rank 1 all of up.
            ShardWeightsForTensorParallelism(
                columnParallelPatterns: Array.Empty<string>(),
                rowParallelPatterns: new[] { "ffn_down.weight" });

            for (int layer = 0; layer < TotalLayerCount; layer++)
            {
                if (layer == 0 || (layer + 1) % 16 == 0 || layer + 1 == TotalLayerCount)
                    Console.WriteLine($"    Gate+Up sharding: layer {layer} ({layer + 1}/{TotalLayerCount})");
                ShardFusedGateUpColumnParallel($"blk.{layer}.ffn_gate_up.weight");
            }

            // --- MoE layers: tensor-parallel experts ---
            if (_numExperts > 0)
                ShardMoeWeightsForTP();

            // --- LM head: column-parallel over the vocabulary ---
            ShardLmHeadForTP();

            Console.WriteLine($"  Qwen3.5 TP weight sharding complete ({globalTp} GPUs, {tp} local).");
        }

        /// <summary>
        /// Name of the column-parallel LM head weight, or null when the head is
        /// replicated on rank 0 (the direct-CUDA layout, and the tied-embedding
        /// case where the same tensor is still needed whole for the embedding
        /// lookup).
        /// </summary>
        private string _tpLmHeadKey;

        /// <summary>
        /// Split the LM head across ranks by vocabulary, so each GPU produces its
        /// own slice of the logits from its own slice of the weight.
        ///
        /// The head is the single largest tensor left after the layers are
        /// sharded (398 MB Q6_K on Qwen3.5-35B, 1.1 GB Q8_0 on the 9B) and it is
        /// read in full for every decoded token, so leaving it replicated on rank
        /// 0 wastes both that GPU's bandwidth and its VRAM while the other card
        /// sits idle. The vocabulary is the output dimension, so rank r's share is
        /// a contiguous row range — a zero-copy view — and the "gather" afterwards
        /// is just two writes into disjoint halves of the logits buffer, with no
        /// collective at all.
        /// </summary>
        private void ShardLmHeadForTP()
        {
            // Every early return here also disables the whole-model fused TP decode
            // (which folds the column-parallel head into its graph), so each one says
            // why. Silence here read as "TP just isn't any faster".
            //
            // Direct CUDA keeps its replicated head: its TP forward reads the
            // logits through the CUDA-resident weight path, not by name.
            if (!IsGgmlBackend)
                return;
            // Tied embeddings: token_embd.weight doubles as the embedding table,
            // which Embedding() gathers rows from on rank 0. Sharding it would
            // leave half the table missing there.
            if (!_quantWeights.TryGetValue("output.weight", out var qw) || qw == null)
            {
                Console.WriteLine("  Qwen3.5 LM head: kept replicated on rank 0 " +
                    "(no separate output.weight - tied embeddings).");
                return;
            }
            // Split across THIS NODE's ranks, not the cluster's. The head is
            // column-parallel and nothing reduces its output: each rank writes a
            // disjoint slice of the logits and the driver reads the buffer
            // directly. Splitting it globally therefore left the driver holding
            // only its own node's slice of the vocabulary, with the rest of the
            // buffer never written - and because Qwen's EOS id sits in the upper
            // half, EOS could never win the argmax. Generation ran to max_tokens
            // and repeated itself, on output that otherwise read as fluent.
            // Every node computing the whole vocabulary is the same arrangement
            // gpt-oss uses, and on one node this is bit-identical to before.
            if (qw.Ne1 % TpDegree != 0 || qw.Ne1 != Config.VocabSize)
            {
                Console.WriteLine($"  Qwen3.5 LM head: kept replicated on rank 0 " +
                    $"(vocab rows {qw.Ne1} vs config vocab {Config.VocabSize}, local TP degree {TpDegree}).");
                return;
            }

            ShardExpertColumnParallel("output.weight", perNodeOnly: true);
            if (!_tpQuantWeights.ContainsKey("output.weight"))
            {
                Console.WriteLine("  Qwen3.5 LM head: kept replicated on rank 0 " +
                    "(column-parallel split declined).");
                return;
            }

            _tpLmHeadKey = "output.weight";
            Console.WriteLine($"  Qwen3.5 LM head: column-parallel across {TpDegree} local GPU(s), " +
                $"{qw.Ne1 / TpDegree} vocab rows each" +
                (GlobalTpDegree != TpDegree ? " (replicated per node: nothing reduces the head)." : "."));
        }

        /// <summary>
        /// Shard the fused full-attention QKV weight for TP with head-aware
        /// regrouping. The fused weight is a plain output-dim concatenation
        ///   [ Q+gate (2*numHeads*headDim) | K (numKVHeads*headDim) | V (numKVHeads*headDim) ]
        /// so a generic contiguous split would give rank 0 mostly Q rows and no
        /// K/V. See <see cref="ShardConcatenatedColumnParallel"/>.
        /// When the fused weight was never created (mixed quant types prevented
        /// fusion), falls back to <see cref="ShardSeparateColumnParallel"/> which
        /// reads the individual Q/K/V weights directly, avoiding a full F32
        /// intermediate that can OOM on memory-constrained hosts.
        /// </summary>
        private void ShardFusedQkvForTP(string weightName)
        {
            int headDim = Config.HeadDim;
            int qDim = 2 * Config.NumHeads * headDim;
            int kDim = Config.NumKVHeads * headDim;
            int vDim = Config.NumKVHeads * headDim;

            if (_quantWeights.ContainsKey(weightName) || _weights.ContainsKey(weightName))
            {
                ShardConcatenatedColumnParallel(weightName, qDim, kDim, vDim);
            }
            else
            {
                // Derive the layer prefix from the fused name (blk.N.attn_qkv.weight → blk.N.).
                string prefix = weightName.Substring(0, weightName.IndexOf("attn_qkv", StringComparison.Ordinal));
                ShardSeparateColumnParallel(weightName,
                    new[] { prefix + "attn_q.weight", prefix + "attn_k.weight", prefix + "attn_v.weight" },
                    new[] { qDim, kDim, vDim });
            }
        }

        /// <summary>
        /// Shard the GDN-specific weights using block-cyclic head assignment.
        /// The packed ssm_in_proj has layout: [Q | K | V | Z | beta | alpha]
        /// where Q/K are contiguous by K-head and V/Z/beta/alpha are strided by V-head.
        /// </summary>
        private void ShardGdnWeightsForTP()
        {
            int tp = TpDegree;
            int qkDim = _headKDim * _numKHeads;
            int vDim = _headVDim * _numVHeads;
            int qkvDim = 2 * qkDim + vDim;
            int packedDim = qkvDim + vDim + 2 * _numVHeads; // Q+K+V + Z + beta + alpha

            int recurrentTotal = 0, recurrentDone = 0;
            for (int l = 0; l < Config.NumLayers; l++)
                if (_isRecurrent[l]) recurrentTotal++;

            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                if (!_isRecurrent[layer])
                    continue;

                recurrentDone++;
                if (recurrentDone == 1 || recurrentDone % 12 == 0 || recurrentDone == recurrentTotal)
                    Console.WriteLine($"    GDN sharding: layer {layer} ({recurrentDone}/{recurrentTotal})");

                string prefix = $"blk.{layer}.";

                // --- ssm_in_proj.weight: segmented column-parallel ---
                ShardPackedSsmInProj(prefix + "ssm_in_proj.weight",
                    qkDim, vDim, qkvDim, packedDim);

                // --- ssm_conv1d.weight: shard dim 0 with Q|K|V segmentation ---
                ShardConv1dWeight(prefix + "ssm_conv1d.weight", qkDim, vDim, qkvDim);

                // --- ssm_dt_bias (ssm_dt.bias): block-cyclic per V head ---
                ShardPerVHeadWeight(prefix + "ssm_dt.bias");

                // --- ssm_a: block-cyclic per V head ---
                ShardPerVHeadWeight(prefix + "ssm_a");

                // --- ssm_norm.weight: replicated (shared across heads) ---
                // Already in _weights, no sharding needed.

                // --- ssm_out.weight: row-parallel with V-head permutation ---
                ShardSsmOutWeight(prefix + "ssm_out.weight");
            }
        }

        /// <summary>
        /// Shard the packed ssm_in_proj weight. Layout:
        /// [Q(nK*dK) | K(nK*dK) | V(nV*dV) | Z(nV*dV) | beta(nV) | alpha(nV)]
        /// Q/K: contiguous split by K-head blocks.
        /// V/Z/beta/alpha: block-cyclic gather by V-head.
        /// </summary>
        private void ShardPackedSsmInProj(string weightName, int qkDim, int vDim, int qkvDim, int packedDim)
        {
            int tp = TpDegree;
            int globalTp = GlobalTpDegree;
            int rankOffset = TpRankOffset;
            int hiddenSize = Config.HiddenSize;
            int kBlockWidth = _numKHeads / globalTp;
            int localKHeads = kBlockWidth;
            int localQkDim = _headKDim * localKHeads;
            int localVHeads = _numVHeads / globalTp;
            int localVDim = _headVDim * localVHeads;
            int localQkvDim = 2 * localQkDim + localVDim;
            int localPackedDim = localQkvDim + localVDim + 2 * localVHeads;

            if (_quantWeights.TryGetValue(weightName, out var qw))
            {
                // Quantized path: extract per-rank shards
                var shards = new QuantizedWeight[tp];
                var type = (GgmlTensorType)qw.GgmlType;
                long blockSize = GgufFile.GetBlockSize(type);
                long typeSize = GgufFile.GetTypeSize(type);
                long srcRowBytes = NativeDequant.RowSize(qw.GgmlType, qw.Ne0);

                for (int r = 0; r < tp; r++)
                {
                    int globalRank = rankOffset + r;
                    int[] vHeads = ComputeBlockCyclicVHeads(globalRank, globalTp, _numVHeads, _numKHeads);
                    int kStart = globalRank * kBlockWidth;

                    // Build the row indices for this rank's shard
                    var rowIndices = BuildSsmInProjRowIndices(globalRank, globalTp, qkDim, vDim, qkvDim, packedDim);
                    long dstNe0 = qw.Ne0; // input dim unchanged
                    long dstNe1 = rowIndices.Length;
                    long dstRowBytes = NativeDequant.RowSize(qw.GgmlType, dstNe0);
                    long totalBytes = dstNe1 * dstRowBytes;

                    IntPtr shardPtr = QuantizedWeight.AllocateBuffer(totalBytes);
                    unsafe
                    {
                        byte* src = (byte*)qw.Data.ToPointer();
                        byte* dst = (byte*)shardPtr.ToPointer();
                        for (long row = 0; row < dstNe1; row++)
                        {
                            long srcRow = rowIndices[row];
                            Buffer.MemoryCopy(
                                src + srcRow * srcRowBytes,
                                dst + row * dstRowBytes,
                                dstRowBytes, dstRowBytes);
                        }
                    }
                    shards[r] = new QuantizedWeight(shardPtr, totalBytes,
                        qw.GgmlType, dstNe0, dstNe1);
                    // A per-tensor scale is shard-invariant: it does not depend on
                    // the output row, and it distributes over the row-parallel
                    // AllReduce, so every shard carries the parent's value.
                    shards[r].Scale = qw.Scale;
                }

                _tpQuantWeights[weightName] = shards;
                RecordTpWeightScale(weightName, qw);
                _quantWeights.Remove(weightName);
                qw.Dispose();
            }
            else if (_weights.TryGetValue(weightName, out var w))
            {
                // The source pack is F32 because the four GDN input weights had
                // mismatched quant types (every importance-matrix "UD" build does
                // this), so TryFuseWeights fell back to TryFuseWeightsToFloat32.
                //
                // Keeping the SHARD in F32 as well was the single biggest cost of
                // tensor parallelism on such a model: on Qwen3.8-27B it is 337 MB
                // per recurrent layer x 48 layers = 16 GB of F32 shards, i.e. MORE
                // device memory than the whole quantized model, and _tpWeights is
                // served by the generic Ops.Addmm path which has no weight cache and
                // re-uploads the entire weight on every layer, every token, every
                // rank. That is why tp=2 measured no faster than tp=1.
                //
                // Re-encode each rank's gathered rows as Q8_0 instead. The gather is
                // along ROWS, so a row's contents are untouched and the only loss is
                // one 8-bit round-trip of a weight that was already <=5-bit at the
                // source - the same argument ShardSsmOutWeightRequantized makes.
                // Q8_0's 32-element blocks divide the input dim (a power of two), so
                // the shard stays on the cached quantized matmul.
                const int q8Type = (int)GgmlTensorType.Q8_0;
                bool canRequant = (hiddenSize % 32) == 0;
                if (canRequant)
                {
                    var qShards = new QuantizedWeight[tp];
                    long dstRowBytes = NativeDequant.RowSize(q8Type, hiddenSize);
                    for (int r = 0; r < tp; r++)
                    {
                        int globalRank = rankOffset + r;
                        int[] rowIndices = BuildSsmInProjRowIndices(globalRank, globalTp, qkDim, vDim, qkvDim, packedDim);
                        long totalBytes = (long)rowIndices.Length * dstRowBytes;
                        IntPtr shardPtr = QuantizedWeight.AllocateBuffer(totalBytes);
                        unsafe
                        {
                            float* srcPtr = GetFloatPtr(w);
                            byte* dst = (byte*)shardPtr.ToPointer();
                            for (int row = 0; row < rowIndices.Length; row++)
                            {
                                ManagedQuantizedOps.QuantizeRowFromFloat32(
                                    q8Type,
                                    srcPtr + (long)rowIndices[row] * hiddenSize,
                                    (IntPtr)(dst + (long)row * dstRowBytes),
                                    hiddenSize);
                            }
                        }
                        qShards[r] = new QuantizedWeight(shardPtr, totalBytes,
                            q8Type, hiddenSize, rowIndices.Length);
                        // The shard keeps the default scale of 1.0. There is no parent
                        // QuantizedWeight here - this branch runs on the F32 pack that
                        // TryFuseWeightsToFloat32 built, and that already MULTIPLIED each
                        // source's sidecar per-tensor scale into the rows it dequantized.
                        // Carrying a scale again (as the quantized branch above rightly
                        // does from its own parent) would apply it twice.
                    }
                    _tpQuantWeights[weightName] = qShards;
                    RecordTpWeightScale(weightName, qw);
                }
                else
                {
                    var shards = new Tensor[tp];
                    for (int r = 0; r < tp; r++)
                    {
                        int globalRank = rankOffset + r;
                        int[] rowIndices = BuildSsmInProjRowIndices(globalRank, globalTp, qkDim, vDim, qkvDim, packedDim);
                        var shard = new Tensor(_tpGroup.GetAllocator(r), DType.Float32, rowIndices.Length, hiddenSize);
                        unsafe
                        {
                            float* srcPtr = GetFloatPtr(w);
                            float* dstPtr = GetFloatPtr(shard);
                            for (int row = 0; row < rowIndices.Length; row++)
                            {
                                Buffer.MemoryCopy(
                                    srcPtr + (long)rowIndices[row] * hiddenSize,
                                    dstPtr + (long)row * hiddenSize,
                                    hiddenSize * 4, hiddenSize * 4);
                            }
                        }
                        shards[r] = shard;
                    }
                    _tpWeights[weightName] = shards;
                    RecordTpWeightScale(weightName, qw);
                }

                _weights.Remove(weightName);
                w.Dispose();
            }
        }

        /// <summary>
        /// Build the output-row indices for rank r's ssm_in_proj shard.
        /// The packed layout is: [Q | K | V | Z | beta | alpha]
        /// Q rows: kStart*dK .. (kStart+kBlockWidth)*dK - 1
        /// K rows: qkDim + kStart*dK .. qkDim + (kStart+kBlockWidth)*dK - 1
        /// V rows: 2*qkDim + vHead*dV .. 2*qkDim + (vHead+1)*dV - 1 for each owned V head
        /// Z rows: qkvDim + vHead*dV .. qkvDim + (vHead+1)*dV - 1
        /// beta rows: qkvDim + vDim + vHead
        /// alpha rows: qkvDim + vDim + nV + vHead
        /// </summary>
        private int[] BuildSsmInProjRowIndices(int rank, int tp, int qkDim, int vDim, int qkvDim, int packedDim)
        {
            int kBlockWidth = _numKHeads / tp;
            int kStart = rank * kBlockWidth;
            int[] vHeads = ComputeBlockCyclicVHeads(rank, tp, _numVHeads, _numKHeads);

            var indices = new List<int>();

            // Q rows: contiguous block of kBlockWidth K-heads
            for (int kh = kStart; kh < kStart + kBlockWidth; kh++)
                for (int d = 0; d < _headKDim; d++)
                    indices.Add(kh * _headKDim + d);

            // K rows: same block, offset by qkDim
            for (int kh = kStart; kh < kStart + kBlockWidth; kh++)
                for (int d = 0; d < _headKDim; d++)
                    indices.Add(qkDim + kh * _headKDim + d);

            // V rows: block-cyclic V heads, offset by 2*qkDim
            foreach (int vh in vHeads)
                for (int d = 0; d < _headVDim; d++)
                    indices.Add(2 * qkDim + vh * _headVDim + d);

            // Z rows: same V heads, offset by qkvDim
            foreach (int vh in vHeads)
                for (int d = 0; d < _headVDim; d++)
                    indices.Add(qkvDim + vh * _headVDim + d);

            // beta rows: one per V head
            foreach (int vh in vHeads)
                indices.Add(qkvDim + vDim + vh);

            // alpha rows: one per V head
            foreach (int vh in vHeads)
                indices.Add(qkvDim + vDim + _numVHeads + vh);

            return indices.ToArray();
        }

        /// <summary>
        /// Shard the depthwise conv1d weight [qkvDim, convKernel] along dim 0
        /// using the same Q|K|V segmentation as ssm_in_proj.
        /// </summary>
        private void ShardConv1dWeight(string weightName, int qkDim, int vDim, int qkvDim)
        {
            int tp = TpDegree;
            int globalTp = GlobalTpDegree;
            int rankOffset = TpRankOffset;

            if (!_weights.TryGetValue(weightName, out var w))
                return;

            int convKernel = (int)w.Sizes[1];
            var shards = new Tensor[tp];

            for (int r = 0; r < tp; r++)
            {
                int globalRank = rankOffset + r;
                int[] rowIndices = BuildConv1dRowIndices(globalRank, globalTp, qkDim, vDim, qkvDim);
                var shard = new Tensor(_tpGroup.GetAllocator(r), DType.Float32, rowIndices.Length, convKernel);
                unsafe
                {
                    float* srcPtr = GetFloatPtr(w);
                    float* dstPtr = GetFloatPtr(shard);
                    for (int row = 0; row < rowIndices.Length; row++)
                    {
                        Buffer.MemoryCopy(
                            srcPtr + (long)rowIndices[row] * convKernel,
                            dstPtr + (long)row * convKernel,
                            convKernel * 4, convKernel * 4);
                    }
                }
                shards[r] = shard;
            }

            _tpWeights[weightName] = shards;
            _weights.Remove(weightName);
            w.Dispose();
        }

        private int[] BuildConv1dRowIndices(int rank, int tp, int qkDim, int vDim, int qkvDim)
        {
            int kBlockWidth = _numKHeads / tp;
            int kStart = rank * kBlockWidth;
            int[] vHeads = ComputeBlockCyclicVHeads(rank, tp, _numVHeads, _numKHeads);

            var indices = new List<int>();

            // Q channels
            for (int kh = kStart; kh < kStart + kBlockWidth; kh++)
                for (int d = 0; d < _headKDim; d++)
                    indices.Add(kh * _headKDim + d);

            // K channels
            for (int kh = kStart; kh < kStart + kBlockWidth; kh++)
                for (int d = 0; d < _headKDim; d++)
                    indices.Add(qkDim + kh * _headKDim + d);

            // V channels
            foreach (int vh in vHeads)
                for (int d = 0; d < _headVDim; d++)
                    indices.Add(2 * qkDim + vh * _headVDim + d);

            return indices.ToArray();
        }

        /// <summary>
        /// Shard a 1D weight indexed per V-head (dt_bias, a) using block-cyclic assignment.
        /// </summary>
        private void ShardPerVHeadWeight(string weightName)
        {
            int tp = TpDegree;
            int globalTp = GlobalTpDegree;
            int rankOffset = TpRankOffset;

            if (!_weights.TryGetValue(weightName, out var w))
                return;

            var shards = new Tensor[tp];
            for (int r = 0; r < tp; r++)
            {
                int globalRank = rankOffset + r;
                int[] vHeads = ComputeBlockCyclicVHeads(globalRank, globalTp, _numVHeads, _numKHeads);
                var shard = new Tensor(_tpGroup.GetAllocator(r), DType.Float32, vHeads.Length);
                unsafe
                {
                    float* srcPtr = GetFloatPtr(w);
                    float* dstPtr = GetFloatPtr(shard);
                    for (int i = 0; i < vHeads.Length; i++)
                        dstPtr[i] = srcPtr[vHeads[i]];
                }
                shards[r] = shard;
            }

            _tpWeights[weightName] = shards;
            _weights.Remove(weightName);
            w.Dispose();
        }

        /// <summary>
        /// Shard ssm_out.weight [hidden, v_dim] as row-parallel.
        /// The input columns are gathered in block-cyclic V-head order so
        /// each rank's shard aligns with the GDN kernel output (which emits
        /// V heads in the order returned by <see cref="ComputeBlockCyclicVHeads"/>).
        /// A plain contiguous split would pair the wrong V-head weights with
        /// the wrong GDN outputs and corrupt every recurrent layer.
        /// </summary>
        private void ShardSsmOutWeight(string weightName)
        {
            int tp = TpDegree;
            int globalTp = GlobalTpDegree;
            int rankOffset = TpRankOffset;

            if (_quantWeights.TryGetValue(weightName, out var qw))
            {
                // Quantized row-parallel: gather block-aligned columns per V head.
                var type = (GgmlTensorType)qw.GgmlType;
                long blockSize = GgufFile.GetBlockSize(type);

                // The per-V-head gather below moves whole quant blocks, so it
                // requires the per-head slice (headVDim columns) to be a whole
                // number of blocks. Super-block types (Q4_K/Q6_K/IQ*: 256
                // elements per block) fail that for headVDim=128 — the old math
                // silently produced blocksPerVHead = 0 and a ZERO-BYTE shard
                // that crashed the TP preload ("PreloadQuantizedWeight requires
                // valid cache key, host data, and size"). UD-quant mixes hit
                // this (ssm_out is Q4_K in Qwen3.5-9B-UD-IQ2_XXS; the Q8_0
                // reference model divides fine at 32 elements/block).
                // Re-encode instead of crashing: dequantise each source row,
                // gather the per-V-head slices in F32, requantise to Q8_0.
                if (SelectSsmOutShardEncoding(type, _headVDim) != SsmOutShardEncoding.SourceBlocks)
                {
                    ShardSsmOutWeightRequantized(weightName, qw);
                    return;
                }

                long typeSize = GgufFile.GetTypeSize(type);
                long srcRowBytes = NativeDequant.RowSize(qw.GgmlType, qw.Ne0);
                int blocksPerVHead = _headVDim / (int)blockSize;
                long vHeadBytes = (long)blocksPerVHead * typeSize;
                int localVHeads = _numVHeads / globalTp;
                long ne0PerShard = (long)localVHeads * _headVDim;
                long dstRowBytes = (long)localVHeads * blocksPerVHead * typeSize;
                long totalBytesPerShard = qw.Ne1 * dstRowBytes;

                var shards = new QuantizedWeight[tp];
                for (int r = 0; r < tp; r++)
                {
                    int globalRank = rankOffset + r;
                    int[] vHeads = ComputeBlockCyclicVHeads(globalRank, globalTp, _numVHeads, _numKHeads);
                    IntPtr shardPtr = QuantizedWeight.AllocateBuffer(totalBytesPerShard);
                    unsafe
                    {
                        byte* src = (byte*)qw.Data.ToPointer();
                        byte* dst = (byte*)shardPtr.ToPointer();
                        for (long row = 0; row < qw.Ne1; row++)
                        {
                            long dstOffset = 0;
                            for (int vhIdx = 0; vhIdx < vHeads.Length; vhIdx++)
                            {
                                long srcVhOffset = (long)vHeads[vhIdx] * blocksPerVHead * typeSize;
                                Buffer.MemoryCopy(
                                    src + row * srcRowBytes + srcVhOffset,
                                    dst + row * dstRowBytes + dstOffset,
                                    vHeadBytes, vHeadBytes);
                                dstOffset += vHeadBytes;
                            }
                        }
                    }
                    shards[r] = new QuantizedWeight(shardPtr, totalBytesPerShard,
                        qw.GgmlType, ne0PerShard, qw.Ne1);
                    // A per-tensor scale is shard-invariant: it does not depend on
                    // the output row, and it distributes over the row-parallel
                    // AllReduce, so every shard carries the parent's value.
                    shards[r].Scale = qw.Scale;
                }

                _tpQuantWeights[weightName] = shards;
                RecordTpWeightScale(weightName, qw);
                _quantWeights.Remove(weightName);
                qw.Dispose();
            }
            else if (_weights.TryGetValue(weightName, out var w))
            {
                // F32 row-parallel: gather columns per V head in block-cyclic order.
                int totalVDim = (int)w.Sizes[1];
                int localVHeads = _numVHeads / globalTp;
                int vDimPerShard = localVHeads * _headVDim;
                int hiddenDim = (int)w.Sizes[0];
                var shards = new Tensor[tp];
                for (int r = 0; r < tp; r++)
                {
                    int globalRank = rankOffset + r;
                    int[] vHeads = ComputeBlockCyclicVHeads(globalRank, globalTp, _numVHeads, _numKHeads);
                    var shard = new Tensor(_tpGroup.GetAllocator(r), DType.Float32, hiddenDim, vDimPerShard);
                    unsafe
                    {
                        float* srcPtr = GetFloatPtr(w);
                        float* dstPtr = GetFloatPtr(shard);
                        for (int row = 0; row < hiddenDim; row++)
                        {
                            long dstColOffset = 0;
                            for (int vhIdx = 0; vhIdx < vHeads.Length; vhIdx++)
                            {
                                long srcCol = (long)vHeads[vhIdx] * _headVDim;
                                Buffer.MemoryCopy(
                                    srcPtr + (long)row * totalVDim + srcCol,
                                    dstPtr + (long)row * vDimPerShard + dstColOffset,
                                    (long)_headVDim * 4, (long)_headVDim * 4);
                                dstColOffset += _headVDim;
                            }
                        }
                    }
                    shards[r] = shard;
                }

                _tpWeights[weightName] = shards;
                RecordTpWeightScale(weightName, qw);
                _weights.Remove(weightName);
                w.Dispose();
            }
        }

        /// <summary>
        /// How a quantized ssm_out.weight can be sharded per V head. The
        /// block-cyclic gather moves whole quant blocks, so it needs
        /// headVDim % blockSize == 0; otherwise the shard must be re-encoded
        /// (Q8_0 when its 32-element block divides headVDim, F32 as the last
        /// resort). Extracted as a pure decision so the UD-quant fallback is
        /// unit-testable without a model file.
        /// </summary>
        internal enum SsmOutShardEncoding { SourceBlocks, RequantQ8, Float32 }

        internal static SsmOutShardEncoding SelectSsmOutShardEncoding(GgmlTensorType type, int headVDim)
        {
            long blockSize = GgufFile.GetBlockSize(type);
            if (blockSize > 0 && headVDim % blockSize == 0)
                return SsmOutShardEncoding.SourceBlocks;
            return headVDim % 32 == 0 ? SsmOutShardEncoding.RequantQ8 : SsmOutShardEncoding.Float32;
        }

        /// <summary>
        /// Fallback for <see cref="ShardSsmOutWeight"/> when the source quant
        /// type's block (e.g. Q4_K's 256 elements) does not divide the per-V-head
        /// column width, so the block-cyclic gather cannot move source blocks.
        /// Each source row is dequantised, its V-head slices gathered in F32 in
        /// block-cyclic order, and the permuted row re-encoded as Q8_0 (32-element
        /// blocks divide any power-of-two head dim; int8-with-per-32-scale
        /// re-encoding of an already ≤6-bit weight is lossless in practice), so
        /// the shard stays on the quantized fast path. Falls back to plain F32
        /// shards in the (unexpected) case that even 32 does not divide headVDim.
        /// </summary>
        private void ShardSsmOutWeightRequantized(string weightName, QuantizedWeight qw)
        {
            int tp = TpDegree;
            int globalTp = GlobalTpDegree;
            int rankOffset = TpRankOffset;
            int localVHeads = _numVHeads / globalTp;
            long ne0PerShard = (long)localVHeads * _headVDim;
            long srcRowBytes = NativeDequant.RowSize(qw.GgmlType, qw.Ne0);
            const int q8Type = (int)GgmlTensorType.Q8_0;
            bool useQ8 = SelectSsmOutShardEncoding((GgmlTensorType)qw.GgmlType, _headVDim)
                == SsmOutShardEncoding.RequantQ8;

            Console.WriteLine($"  {weightName}: {(GgmlTensorType)qw.GgmlType} block " +
                $"({GgufFile.GetBlockSize((GgmlTensorType)qw.GgmlType)} elems) does not divide headVDim={_headVDim}; " +
                $"re-encoding TP shards as {(useQ8 ? "Q8_0" : "F32")}.");

            if (useQ8)
            {
                long dstRowBytes = NativeDequant.RowSize(q8Type, ne0PerShard);
                long totalBytesPerShard = qw.Ne1 * dstRowBytes;
                var shards = new QuantizedWeight[tp];
                for (int r = 0; r < tp; r++)
                {
                    int globalRank = rankOffset + r;
                    int[] vHeads = ComputeBlockCyclicVHeads(globalRank, globalTp, _numVHeads, _numKHeads);
                    IntPtr shardPtr = QuantizedWeight.AllocateBuffer(totalBytesPerShard);
                    unsafe
                    {
                        byte* src = (byte*)qw.Data.ToPointer();
                        byte* dst = (byte*)shardPtr.ToPointer();
                        var srcRowArr = new float[qw.Ne0];
                        var dstRowArr = new float[ne0PerShard];
                        fixed (float* srcF = srcRowArr)
                        fixed (float* dstF = dstRowArr)
                        {
                            for (long row = 0; row < qw.Ne1; row++)
                            {
                                NativeDequant.DequantizeToFloat32Native(
                                    qw.GgmlType, (IntPtr)(src + row * srcRowBytes), (IntPtr)srcF, qw.Ne0);
                                for (int vhIdx = 0; vhIdx < vHeads.Length; vhIdx++)
                                {
                                    Buffer.MemoryCopy(
                                        srcF + (long)vHeads[vhIdx] * _headVDim,
                                        dstF + (long)vhIdx * _headVDim,
                                        (long)_headVDim * sizeof(float),
                                        (long)_headVDim * sizeof(float));
                                }
                                ManagedQuantizedOps.QuantizeRowFromFloat32(
                                    q8Type, dstF, (IntPtr)(dst + row * dstRowBytes), ne0PerShard);
                            }
                        }
                    }
                    shards[r] = new QuantizedWeight(shardPtr, totalBytesPerShard,
                        q8Type, ne0PerShard, qw.Ne1);
                    // A per-tensor scale is shard-invariant: it does not depend on
                    // the output row, and it distributes over the row-parallel
                    // AllReduce, so every shard carries the parent's value.
                    shards[r].Scale = qw.Scale;
                }
                _tpQuantWeights[weightName] = shards;
                RecordTpWeightScale(weightName, qw);
            }
            else
            {
                var shards = new Tensor[tp];
                for (int r = 0; r < tp; r++)
                {
                    int globalRank = rankOffset + r;
                    int[] vHeads = ComputeBlockCyclicVHeads(globalRank, globalTp, _numVHeads, _numKHeads);
                    var shard = new Tensor(_tpGroup.GetAllocator(r), DType.Float32, qw.Ne1, ne0PerShard);
                    unsafe
                    {
                        byte* src = (byte*)qw.Data.ToPointer();
                        float* dstBase = GetFloatPtr(shard);
                        var srcRowArr = new float[qw.Ne0];
                        fixed (float* srcF = srcRowArr)
                        {
                            for (long row = 0; row < qw.Ne1; row++)
                            {
                                NativeDequant.DequantizeToFloat32Native(
                                    qw.GgmlType, (IntPtr)(src + row * srcRowBytes), (IntPtr)srcF, qw.Ne0);
                                float* dstRow = dstBase + row * ne0PerShard;
                                for (int vhIdx = 0; vhIdx < vHeads.Length; vhIdx++)
                                {
                                    Buffer.MemoryCopy(
                                        srcF + (long)vHeads[vhIdx] * _headVDim,
                                        dstRow + (long)vhIdx * _headVDim,
                                        (long)_headVDim * sizeof(float),
                                        (long)_headVDim * sizeof(float));
                                }
                            }
                        }
                    }
                    shards[r] = shard;
                }
                _tpWeights[weightName] = shards;
                RecordTpWeightScale(weightName, qw);
            }

            _quantWeights.Remove(weightName);
            qw.Dispose();
        }

        /// <summary>
        /// Shard MoE expert weights for tensor-parallel experts.
        /// Each rank holds 1/tp of every expert's FFN width.
        /// Router weights (ffn_gate_inp) are replicated.
        /// </summary>
        private void ShardMoeWeightsForTP()
        {
            int tp = TpDegree;

            // Prefer whole-expert partitioning on GGML: keeping the batched
            // ggml_mul_mat_id dispatch usable is worth far more than the
            // marginally finer memory split per-expert slicing would give.
            bool expertParallel = BuildQwen35ExpertParallelShards();

            for (int layer = 0; layer < TotalLayerCount; layer++)
            {
                if (_isMoeLayer == null || !_isMoeLayer[layer])
                    continue;

                string prefix = $"blk.{layer}.";

                // Router weight: replicated (no sharding needed, stays in _weights)

                // Expert gate/up weights: column-parallel (split expertFfnLength)
                for (int e = 0; e < _numExperts && !expertParallel; e++)
                {
                    string gateKey = prefix + $"ffn_gate_exps.{e}.weight";
                    string upKey = prefix + $"ffn_up_exps.{e}.weight";
                    string downKey = prefix + $"ffn_down_exps.{e}.weight";

                    ShardExpertColumnParallel(gateKey);
                    ShardExpertColumnParallel(upKey);
                    ShardExpertRowParallel(downKey);
                }

                // Shared expert weights: same column/row split
                if (_hasSharedExperts != null && _hasSharedExperts[layer])
                {
                    ShardExpertColumnParallel(prefix + "ffn_gate_shexp.weight");
                    ShardExpertColumnParallel(prefix + "ffn_up_shexp.weight");
                    ShardExpertRowParallel(prefix + "ffn_down_shexp.weight");
                    // ffn_gate_inp_shexp: replicated (stays in _weights)
                }
            }
        }

        /// <param name="perNodeOnly">
        /// Split across this node's ranks only, giving every node the whole
        /// tensor. Correct exactly for a column-parallel output that nothing
        /// reduces afterwards - the LM head. Everything else here feeds a
        /// row-parallel projection whose AllReduce spans the cluster, so it must
        /// keep splitting by the GLOBAL degree.
        /// </param>
        private void ShardExpertColumnParallel(string weightName, bool perNodeOnly = false)
        {
            int tp = TpDegree;
            int globalTp = perNodeOnly ? TpDegree : GlobalTpDegree;
            int rankOffset = perNodeOnly ? 0 : TpRankOffset;

            if (_quantWeights.TryGetValue(weightName, out var qw))
            {
                long rowsPerShard = qw.Ne1 / globalTp;
                long rowBytes = NativeDequant.RowSize(qw.GgmlType, qw.Ne0);
                long bytesPerShard = rowsPerShard * rowBytes;

                var shards = new QuantizedWeight[tp];
                for (int r = 0; r < tp; r++)
                {
                    int globalRank = rankOffset + r;
                    // 64-bit offset arithmetic: `globalRank * bytesPerShard` is a long,
                    // and a per-rank shard of the LM head or a shared expert can exceed
                    // 2 GiB. The old (int) cast wrapped silently there, handing out a
                    // wrong (possibly negative) base pointer with no error. Matches the
                    // long-safe form the generic sharder already uses in ModelBase.
                    IntPtr shardPtr = new IntPtr((long)qw.Data + globalRank * bytesPerShard);
                    shards[r] = QuantizedWeight.CreateExternalView(
                        shardPtr, bytesPerShard, qw.GgmlType, qw.Ne0, rowsPerShard, qw);
                }

                _tpQuantWeights[weightName] = shards;
                RecordTpWeightScale(weightName, qw);
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
                RecordTpWeightScale(weightName, qw);
                _weights.Remove(weightName);
                w.Dispose();
            }
        }

        private void ShardExpertRowParallel(string weightName)
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
                // A row-parallel split can only cut on quant-block boundaries. If the
                // blocks per row do not divide evenly the shards would silently drop
                // the remainder (and at tp > blocksPerRow they would be empty),
                // producing a model that loads fine and generates garbage.
                if (blocksPerShard <= 0 || blocksPerRow % globalTp != 0)
                {
                    throw new InvalidOperationException(
                        $"TP degree {globalTp} cannot row-split '{weightName}': {blocksPerRow} " +
                        $"{(GgmlTensorType)qw.GgmlType} blocks per row (block size {blockSize}, ne0 {qw.Ne0}) " +
                        $"is not divisible by {globalTp}. Use a TP degree that divides the block count.");
                }
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
                    // A per-tensor scale is shard-invariant: it does not depend on
                    // the output row, and it distributes over the row-parallel
                    // AllReduce, so every shard carries the parent's value.
                    shards[r].Scale = qw.Scale;
                }

                _tpQuantWeights[weightName] = shards;
                RecordTpWeightScale(weightName, qw);
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
                RecordTpWeightScale(weightName, qw);
                _weights.Remove(weightName);
                w.Dispose();
            }
        }

        // ====================================================================
        // TP cache initialization
        // ====================================================================

        private void InitTpCaches(int initialSeqLen, int maxSeqLen)
        {
            _maxContextLength = maxSeqLen;
            _initialKvCacheCapacity = initialSeqLen;
            int tp = TpDegree;

            // --- Full-attention KV caches ---
            int numKVHeadsPerGpu = Config.NumKVHeads / GlobalTpDegree;
            int headDim = Config.HeadDim;
            DType kvDtype = _kvCacheDtype.ToDType();

            _tpKvCacheCapacity = initialSeqLen;
            _tpKvCacheK = new Tensor[TotalLayerCount][];
            _tpKvCacheV = new Tensor[TotalLayerCount][];

            for (int l = 0; l < TotalLayerCount; l++)
            {
                if (_isRecurrent[l])
                {
                    _tpKvCacheK[l] = null;
                    _tpKvCacheV[l] = null;
                    continue;
                }

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

            // --- GDN recurrent state (sharded by the GLOBAL degree) ---
            int convDim = _convKernel - 1;
            int localVHeads = _numVHeads / GlobalTpDegree;
            int localKHeads = _numKHeads / GlobalTpDegree;
            int localQkvDim = 2 * (_headKDim * localKHeads) + (_headVDim * localVHeads);

            _tpDeltaState = new Tensor[Config.NumLayers][];
            _tpConvState = new Tensor[Config.NumLayers][];
            _tpConvWriteIdx = new int[Config.NumLayers];

            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!_isRecurrent[l])
                {
                    _tpDeltaState[l] = null;
                    _tpConvState[l] = null;
                    continue;
                }

                _tpDeltaState[l] = new Tensor[tp];
                _tpConvState[l] = new Tensor[tp];
                _tpConvWriteIdx[l] = 0;

                for (int r = 0; r < tp; r++)
                {
                    var alloc = _tpGroup.GetAllocator(r);
                    _tpDeltaState[l][r] = new Tensor(alloc, DType.Float32, localVHeads, _headVDim, _headKDim);
                    Ops.Fill(_tpDeltaState[l][r], 0);

                    if (convDim > 0)
                    {
                        // Direct CUDA drives a ring buffer indexed by
                        // _tpConvWriteIdx, so time is the slow axis. The GGML
                        // kernel feeds ggml_ssm_conv, which wants an ordered
                        // window with time contiguous per channel — the shard is
                        // transposed and the write index stays 0.
                        _tpConvState[l][r] = IsGgmlBackend
                            ? new Tensor(alloc, DType.Float32, localQkvDim, convDim)
                            : new Tensor(alloc, DType.Float32, convDim, localQkvDim);
                        Ops.Fill(_tpConvState[l][r], 0);
                    }
                }
            }

            Console.WriteLine($"  TP caches initialized: {tp} GPUs, " +
                $"KV heads/GPU={numKVHeadsPerGpu}, GDN V heads/GPU={localVHeads}");
        }

        private void EnsureTpCacheCapacity(int requiredSeqLen)
        {
            if (requiredSeqLen <= _tpKvCacheCapacity)
                return;
            if (requiredSeqLen > _maxContextLength)
                throw new InvalidOperationException(
                    $"Requested sequence length {requiredSeqLen} exceeds configured max context {_maxContextLength}.");

            int newCapacity = Math.Max(_tpKvCacheCapacity, 1);
            while (newCapacity < requiredSeqLen)
                newCapacity = Math.Min(_maxContextLength, newCapacity * 2);

            // Ops.Copy below is a HOST-side copy, but the fused TP paths (both the
            // per-rank attention block and the whole-model decode) leave K/V
            // device-resident and only mark _tpAttnKvDeviceOnly. Without this drain
            // the copy reads a host mirror that is stale by however many tokens have
            // been decoded since the last sync, and every attention layer silently
            // inherits garbage history from the grow onward — fluent output that
            // degenerates into repetition, on tp>=2 only, once the prompt+generation
            // crosses the initial capacity. The non-TP EnsureCacheCapacity has done
            // this from the start; the TP twin never did.
            SyncQwen35TpAttentionKvToHost();
            // The persistent TP decode graphs bake the old cache tensors'
            // device buffers; drop them before those tensors are disposed.
            // (SyncQwen35TpAttentionKvToHost also drops them, but only when it had
            // something to drain.)
            DropTpFusedDecodeGraphs();

            int tp = TpDegree;
            int numKVHeadsPerGpu = Config.NumKVHeads / GlobalTpDegree;
            int headDim = Config.HeadDim;
            DType kvDtype = _kvCacheDtype.ToDType();
            int previousRank = IsGgmlBackend ? GgmlBasicOps.GetActiveRank() : 0;

            for (int l = 0; l < TotalLayerCount; l++)
            {
                if (_isRecurrent[l] || _tpKvCacheK[l] == null)
                    continue;

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

                    // Evict the device-copy entries keyed by the OLD host pointers
                    // while those pointers are still valid: the allocator may hand
                    // the same address back for a later tensor, which would then
                    // bind a stale device buffer. The cache is per rank, hence the
                    // rank switch (same pattern as ResetTpKVCache).
                    if (IsGgmlBackend)
                    {
                        GgmlBasicOps.SetActiveRank(r);
                        InvalidateTensorDeviceCache(_tpKvCacheK[l][r]);
                        InvalidateTensorDeviceCache(_tpKvCacheV[l][r]);
                    }

                    _tpKvCacheK[l][r].Dispose();
                    _tpKvCacheV[l][r].Dispose();
                    _tpKvCacheK[l][r] = newK;
                    _tpKvCacheV[l][r] = newV;
                }
            }

            if (IsGgmlBackend)
                GgmlBasicOps.SetActiveRank(previousRank);

            _tpKvCacheCapacity = newCapacity;
            Console.WriteLine($"Expanded Qwen3.5 TP attention cache to {newCapacity} tokens ({tp} GPUs).");
        }

        // ====================================================================
        // TP forward pass
        // ====================================================================

        private unsafe float[] ForwardTP(int[] tokens)
        {
            _forwardSw.Start();
            int seqLen = tokens.Length;
            int startPos = _cacheSeqLen;
            int tp = TpDegree;
            EnsureTpCacheCapacity(startPos + seqLen);

            long t1 = Stopwatch.GetTimestamp();
            Tensor hidden0 = Embedding(tokens);
            _embTicks += Stopwatch.GetTimestamp() - t1;

            // Inject any queued vision embeddings on rank 0 before broadcasting
            // (mirrors the non-TP ForwardCore). The matching MRoPE positions are
            // staged via SetMRoPEPositions and consumed by the attention block.
            if (_visionEmbeddingsList.Count > 0)
                InjectVisionEmbeddings(hidden0, seqLen);

            // Whole-model fused decode: one persistent segmented graph per rank
            // (KV append, GDN recurrence, MoE routing and the column-parallel LM
            // head all in-graph). Falls through to the per-op layer loop below
            // when unavailable. MRoPE positions only ever accompany a multimodal
            // prefill, so the scalar-position decode graph applies whenever none
            // are staged — the same condition the per-op attention path uses.
            if (seqLen == 1 && _pendingMRoPEPositions == null)
            {
                if (_logitsBuffer == null || _logitsBuffer.Length != Config.VocabSize)
                    _logitsBuffer = new float[Config.VocabSize];
                if (TryQwen35FusedModelDecodeTP(hidden0, startPos, _logitsBuffer))
                {
                    hidden0.Dispose();
                    _cacheSeqLen += seqLen;
                    _forwardCount++;
                    _forwardSw.Stop();
                    return _logitsBuffer;
                }
            }

            // Whole-model fused prefill: the N-token sibling of the fused decode
            // (verify-kernel tp_mode). One segmented graph per rank keeps the
            // [N, hidden] activations in VRAM instead of round-tripping them
            // through the host between every per-layer fused block.
            if (seqLen > 1)
            {
                if (_logitsBuffer == null || _logitsBuffer.Length != Config.VocabSize)
                    _logitsBuffer = new float[Config.VocabSize];
                if (TryQwen35FusedModelPrefillTP(hidden0, seqLen, startPos, _logitsBuffer))
                {
                    hidden0.Dispose();
                    _cacheSeqLen += seqLen;
                    _forwardCount++;
                    _pendingMRoPEPositions = null;
                    _forwardSw.Stop();
                    return _logitsBuffer;
                }
            }

            // Broadcast embedding to all GPUs.
            Tensor[] hidden = BroadcastTensorToAllRanks(hidden0);

            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                if (_isRecurrent[layer])
                    hidden = RecurrentBlockTP(hidden, layer, seqLen, startPos);
                else
                    hidden = AttentionBlockTP(hidden, layer, seqLen, startPos);
                TraceLayer(hidden[0], layer, "-tp");
            }
            _layerTraceForwards++;

            // Final norm + LM head on GPU 0 only (hidden is replicated after AllReduce).
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
            if (LayerTraceEnabled && _layerTraceForwards <= 1)
                Console.WriteLine($"[TRACE-lm] columnParallel={_tpLmHeadKey != null} " +
                    $"activeRank={(IsGgmlBackend ? GgmlBasicOps.GetActiveRank() : -1)}");
            if (_tpLmHeadKey != null)
            {
                // Column-parallel head: each rank produces its own vocabulary
                // slice, and the slices are disjoint, so the "gather" is a pair of
                // copies into halves of the logits buffer rather than a collective.
                Tensor[] logitParts = TpColumnParallelLinear(lastHidden, _tpLmHeadKey);
                _lmHeadTicks += Stopwatch.GetTimestamp() - t2;
                lastHidden.Dispose();

                long tGather = Stopwatch.GetTimestamp();
                if (_logitsBuffer == null || _logitsBuffer.Length != Config.VocabSize)
                    _logitsBuffer = new float[Config.VocabSize];
                int offset = 0;
                for (int r = 0; r < tp; r++)
                {
                    int count = (int)logitParts[r].ElementCount();
                    unsafe
                    {
                        float* src = GetFloatPtr(logitParts[r]);
                        fixed (float* dst = &_logitsBuffer[offset])
                            Buffer.MemoryCopy(src, dst, (long)count * sizeof(float), (long)count * sizeof(float));
                    }
                    offset += count;
                    logitParts[r].Dispose();
                }
                _logitsCopyTicks += Stopwatch.GetTimestamp() - tGather;
            }
            else
            {
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
            }

            _cacheSeqLen += seqLen;
            _forwardCount++;
            // Drop the MRoPE positions staged for this (multimodal) forward so the
            // next call defaults to scalar positions, matching the non-TP path.
            _pendingMRoPEPositions = null;
            _forwardSw.Stop();
            return _logitsBuffer;
        }

        // ====================================================================
        // Full-attention block under TP
        // ====================================================================

        private Tensor[] AttentionBlockTP(Tensor[] hidden, int layer, int seqLen, int startPos)
        {
            int tp = TpDegree;
            string prefix = $"blk.{layer}.";

            // Fast path: the whole block (norm through residual add) as one
            // segmented graph per rank, cut at the row-parallel output
            // projection. Falls through to the per-op chain below when the fused
            // kernel declines.
            if (TryQwen35FusedAttentionBlockTP(hidden, layer, seqLen, startPos))
                return FFNBlockTP(hidden, layer, seqLen);

            // Per-op attention reads the KV caches from host memory; a fused
            // block leaves its appends in the device copies only.
            SyncQwen35TpAttentionKvToHost();

            long tAttnBlock = Stopwatch.GetTimestamp();

            // 1. Attention norm (replicated).
            Tensor[] normed = TpRMSNorm(hidden, _attnNormKey[layer]);

            // 2. Column-parallel QKV projection.
            // Qwen3.5 QKV output: [Q+gate (2*numHeads*headDim) | K (numKVHeads*headDim) | V (numKVHeads*headDim)]
            Tensor[] qkvFused = TpColumnParallelLinear(normed[0], _attnQkvKey[layer]);
            for (int r = 0; r < tp; r++)
                normed[r].Dispose();

            // 3. Per-GPU attention.
            Tensor[] attnOut = FullAttentionTP(qkvFused, layer, seqLen, startPos);

            // 4. Row-parallel output projection + AllReduce. AllReduce leaves the
            // sum on every rank, so the residual can be added in place without a
            // second broadcast from rank 0.
            Tensor[] attnReduced = TpRowParallelLinearAllRanks(attnOut, _attnOutputKey[layer]);
            for (int r = 0; r < tp; r++)
                attnOut[r].Dispose();

            // 5. Residual add (hidden stays replicated: identical inputs + identical addend).
            TpResidualAdd(hidden, attnReduced);
            for (int r = 0; r < tp; r++)
                attnReduced[r].Dispose();
            _tpAttnBlockTicks += Stopwatch.GetTimestamp() - tAttnBlock;

            // 6. FFN (dense or MoE).
            hidden = FFNBlockTP(hidden, layer, seqLen);

            return hidden;
        }

        private Tensor[] FullAttentionTP(Tensor[] qkvFused, int layer, int seqLen, int startPos)
        {
            int tp = TpDegree;
            int numHeadsPerGpu = Config.NumHeads / GlobalTpDegree;
            int numKVHeadsPerGpu = Config.NumKVHeads / GlobalTpDegree;
            int headDim = Config.HeadDim;
            // Qwen3.5: Q output includes gate (2x), so Q dim per GPU = 2 * numHeadsPerGpu * headDim
            int qFullDimPerGpu = 2 * numHeadsPerGpu * headDim;
            int kDimPerGpu = numKVHeadsPerGpu * headDim;
            int totalSeqLen = startPos + seqLen;
            float scale = 1.0f / MathF.Sqrt(headDim);

            var results = new Tensor[tp];

            for (int r = 0; r < tp; r++)
            {
                var alloc = _tpGroup.GetAllocator(r);

                // Split Q+gate, K, V from the fused QKV output.
                Tensor qFull, kTensor, vTensor;
                if (seqLen == 1)
                {
                    qFull = qkvFused[r].Narrow(1, 0, qFullDimPerGpu);
                    kTensor = qkvFused[r].Narrow(1, qFullDimPerGpu, kDimPerGpu);
                    vTensor = qkvFused[r].Narrow(1, qFullDimPerGpu + kDimPerGpu, kDimPerGpu);
                    qkvFused[r].Dispose();
                }
                else
                {
                    using (var qView = qkvFused[r].Narrow(1, 0, qFullDimPerGpu))
                        qFull = Ops.NewContiguous(qView);
                    using (var kView = qkvFused[r].Narrow(1, qFullDimPerGpu, kDimPerGpu))
                        kTensor = Ops.NewContiguous(kView);
                    using (var vView = qkvFused[r].Narrow(1, qFullDimPerGpu + kDimPerGpu, kDimPerGpu))
                        vTensor = Ops.NewContiguous(vView);
                    qkvFused[r].Dispose();
                }

                // Deinterleave Q and gate: Q is [numHeadsPerGpu, headDim], gate is [numHeadsPerGpu, headDim]
                // interleaved per head: [Q0, gate0, Q1, gate1, ...]
                int qDimPerGpu = numHeadsPerGpu * headDim;
                Tensor qTensor, gateTensor;
                DeinterleaveQGate(qFull, out qTensor, out gateTensor, numHeadsPerGpu, headDim, seqLen, alloc);
                qFull.Dispose();

                // QK norm (per-GPU). The RMSNorm weight is replicated on GPU 0;
                // use a rank-local copy so the kernel doesn't read it cross-GPU.
                qTensor = ApplyQKNormCached(qTensor, ReplicaOnRank(_attnQNormW[layer], r), numHeadsPerGpu, seqLen);
                kTensor = ApplyQKNormCached(kTensor, ReplicaOnRank(_attnKNormW[layer], r), numKVHeadsPerGpu, seqLen);

                // RoPE: per-axis MRoPE when multimodal positions are staged for
                // this forward, otherwise the scalar position RoPE. MRoPE positions
                // are per-token, so they apply identically to this rank's head slice.
                bool useMRoPE = _pendingMRoPEPositions != null && _pendingMRoPEPositions.Length >= 3 * seqLen;
                if (useMRoPE)
                {
                    qTensor = ApplyMRoPEPrefill(qTensor, numHeadsPerGpu, seqLen, _pendingMRoPEPositions);
                    kTensor = ApplyMRoPEPrefill(kTensor, numKVHeadsPerGpu, seqLen, _pendingMRoPEPositions);
                }
                else
                {
                    qTensor = ApplyRoPEPrefill(qTensor, numHeadsPerGpu, seqLen, startPos);
                    kTensor = ApplyRoPEPrefill(kTensor, numKVHeadsPerGpu, seqLen, startPos);
                }

                if (seqLen == 1)
                {
                    // Decode: copy K/V to per-GPU cache, run attention.
                    CopyToCacheDecode(_tpKvCacheK[layer][r], kTensor, _tpKvCacheV[layer][r], vTensor,
                        numKVHeadsPerGpu, headDim, startPos);
                    kTensor.Dispose();
                    vTensor.Dispose();

                    var attnResult = new Tensor(alloc, DType.Float32, 1, numHeadsPerGpu * headDim);
                    AttentionDecodePureCS(qTensor, _tpKvCacheK[layer][r], _tpKvCacheV[layer][r],
                        attnResult, numHeadsPerGpu, numKVHeadsPerGpu, headDim, totalSeqLen, scale);
                    qTensor.Dispose();

                    // Apply sigmoid gate: output = attn * sigmoid(gate)
                    Ops.SigmoidMul(attnResult, attnResult, gateTensor);
                    gateTensor.Dispose();

                    results[r] = attnResult;
                }
                else
                {
                    // Prefill path.
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

                    // Apply sigmoid gate.
                    Ops.SigmoidMul(flatOutput, flatOutput, gateTensor);
                    gateTensor.Dispose();

                    results[r] = flatOutput;
                }
            }

            return results;
        }

        /// <summary>
        /// Deinterleave Q and gate from the fused Q+gate tensor.
        /// Input layout: [Q0, gate0, Q1, gate1, ...] per head.
        /// Output: separate Q [seqLen, numHeads*headDim] and gate [seqLen, numHeads*headDim].
        /// </summary>
        private void DeinterleaveQGate(Tensor qFull, out Tensor q, out Tensor gate,
            int numHeads, int headDim, int seqLen, IAllocator alloc)
        {
            int totalDim = numHeads * headDim;
            q = new Tensor(alloc, DType.Float32, seqLen, totalDim);
            gate = new Tensor(alloc, DType.Float32, seqLen, totalDim);

            unsafe
            {
                float* srcPtr = GetFloatPtr(qFull);
                float* qPtr = GetFloatPtr(q);
                float* gPtr = GetFloatPtr(gate);

                for (int s = 0; s < seqLen; s++)
                {
                    float* srcRow = srcPtr + s * 2 * totalDim;
                    float* qRow = qPtr + s * totalDim;
                    float* gRow = gPtr + s * totalDim;

                    for (int h = 0; h < numHeads; h++)
                    {
                        int srcBase = h * 2 * headDim;
                        int dstBase = h * headDim;
                        Buffer.MemoryCopy(srcRow + srcBase, qRow + dstBase, headDim * 4, headDim * 4);
                        Buffer.MemoryCopy(srcRow + srcBase + headDim, gRow + dstBase, headDim * 4, headDim * 4);
                    }
                }
            }
        }

        // ====================================================================
        // GDN recurrent block under TP
        // ====================================================================

        private Tensor[] RecurrentBlockTP(Tensor[] hidden, int layer, int seqLen, int startPos)
        {
            int tp = TpDegree;
            string prefix = $"blk.{layer}.";

            Tensor[] gatedOut;
            if (IsGgmlBackend)
            {
                // GGML fuses steps 1-3 (norm, packed in-projection, conv, scan,
                // gated norm) into one graph per rank — see GatedDeltaNetTpGgml.
                gatedOut = GatedDeltaNetTpGgml(hidden, layer, seqLen);
            }
            else
            {
                // 1. Input norm (replicated).
                Tensor[] normed = TpRMSNorm(hidden, _attnNormKey[layer]);

                // 2. Column-parallel packed input projection (segmented).
                Tensor[] packedInput = TpColumnParallelLinear(normed[0], _ssmInProjKey[layer]);
                for (int r = 0; r < tp; r++)
                    normed[r].Dispose();

                // 3. Per-rank GDN: conv1d → L2norm → delta-rule scan → gated RMSNorm.
                gatedOut = GatedDeltaNetTP(packedInput, layer, seqLen);
                for (int r = 0; r < tp; r++)
                    packedInput[r].Dispose();
            }

            // 4-5. Row-parallel ssm_out + AllReduce + residual add. One segmented
            // graph per rank when the fused path is available, otherwise the
            // three-dispatch chain.
            long tOut = Stopwatch.GetTimestamp();
            _tpQuantWeights.TryGetValue(_ssmOutKey[layer], out var ssmOutShards);
            if (TryQwen35FusedRowParallelResidualTP(hidden, gatedOut, ssmOutShards))
            {
                for (int r = 0; r < tp; r++)
                    gatedOut[r].Dispose();
                _tpSsmOutTicks += Stopwatch.GetTimestamp() - tOut;
            }
            else
            {
                Tensor[] gdnReduced = TpRowParallelLinearAllRanks(gatedOut, _ssmOutKey[layer]);
                for (int r = 0; r < tp; r++)
                    gatedOut[r].Dispose();
                _tpSsmOutTicks += Stopwatch.GetTimestamp() - tOut;

                long tRes = Stopwatch.GetTimestamp();
                TpResidualAdd(hidden, gdnReduced);
                for (int r = 0; r < tp; r++)
                    gdnReduced[r].Dispose();
                _tpResidualTicks += Stopwatch.GetTimestamp() - tRes;
            }

            // 6. FFN (dense or MoE).
            hidden = FFNBlockTP(hidden, layer, seqLen);

            return hidden;
        }

        /// <summary>
        /// Run the GDN mixer on each rank's shard. Uses the CUDA-native kernel
        /// with per-rank dimensions (localVHeads, localKHeads).
        /// </summary>
        private Tensor[] GatedDeltaNetTP(Tensor[] packedInput, int layer, int seqLen)
        {
            int tp = TpDegree;
            int localKHeads = _numKHeads / GlobalTpDegree;
            int localVHeads = _numVHeads / GlobalTpDegree;
            int localQkDim = _headKDim * localKHeads;
            int localVDim = _headVDim * localVHeads;
            int localQkvDim = 2 * localQkDim + localVDim;
            int localPackedDim = localQkvDim + localVDim + 2 * localVHeads;

            var results = new Tensor[tp];

            for (int r = 0; r < tp; r++)
            {
                var alloc = _tpGroup.GetAllocator(r);
                Tensor gated = new Tensor(alloc, DType.Float32, seqLen, localVDim);

                // The packed input is already in the correct per-rank layout
                // (Q|K|V|Z|beta|alpha for this rank's heads).
                // Run the CUDA-native GDN kernel with local dimensions.
                bool ok = CudaFusedOps.TryQwen35GatedDeltaNetPacked(
                    gated,
                    packedInput[r],
                    _tpConvState[layer][r],
                    _tpDeltaState[layer][r],
                    GetTpShardTensor(_ssmConv1dKey[layer], r),
                    GetTpShardTensor(_ssmDtBiasKey[layer], r),
                    GetTpShardTensor(_ssmAKey[layer], r),
                    ReplicaOnRank(_ssmNormW[layer], r),  // replicated, but resident on rank r's GPU
                    seqLen,
                    localPackedDim,
                    localQkvDim,
                    localQkDim,
                    localVDim,
                    localKHeads,
                    localVHeads,
                    _headKDim,
                    _headVDim,
                    _convKernel,
                    _tpConvWriteIdx[layer],
                    Config.Eps);

                if (!ok)
                    throw new InvalidOperationException(
                        $"CUDA-native GDN kernel failed under TP (layer {layer}, rank {r}). " +
                        "TP requires the CUDA-native GDN path (TS_CUDA_QWEN35_GDN_NATIVE must not be 0).");

                results[r] = gated;
            }

            // Advance conv write index (identical on all ranks).
            int convDim = _convKernel - 1;
            if (convDim > 0)
                _tpConvWriteIdx[layer] = (_tpConvWriteIdx[layer] + seqLen) % convDim;

            return results;
        }

        /// <summary>
        /// Per-rank GatedDeltaNet on the GGML backends. Unlike the direct-CUDA
        /// path — which takes an already-projected packed input — the GGML kernel
        /// owns the whole front of the block: input RMSNorm, the packed
        /// column-parallel in-projection, ssm_conv, the delta-rule scan and the
        /// gated RMSNorm, in ONE graph per rank. Folding the projection in is
        /// what makes it worth doing: it removes a separate multi-rank matmul
        /// dispatch and an activation round-trip per recurrent layer, and there
        /// are 30 of them per token on Qwen3.5-35B.
        ///
        /// The conv window and delta state never leave the device; the native
        /// side keys them on these host pointers (see Qwen35GdnLayerTP).
        /// </summary>
        private unsafe Tensor[] GatedDeltaNetTpGgml(Tensor[] hidden, int layer, int seqLen)
        {
            int tp = TpDegree;
            int localKHeads = _numKHeads / GlobalTpDegree;
            int localVHeads = _numVHeads / GlobalTpDegree;
            int localQkDim = _headKDim * localKHeads;
            int localVDim = _headVDim * localVHeads;
            int localQkvDim = 2 * localQkDim + localVDim;
            int localPackedDim = localQkvDim + localVDim + 2 * localVHeads;

            // Resolve every pointer on the calling thread: the replicated-weight
            // cache behind GetTpShardTensor is a plain dictionary and the rank
            // workers below run concurrently.
            IntPtr attnNormPtr = (IntPtr)GetFloatPtr(_weights[_attnNormKey[layer]]);
            IntPtr ssmNormPtr = (IntPtr)GetFloatPtr(_weights[_ssmNormKey[layer]]);

            string inprojKey = _ssmInProjKey[layer];
            _tpQuantWeights.TryGetValue(inprojKey, out var inprojQuant);
            if (inprojQuant == null && !_tpWeights.ContainsKey(inprojKey))
                throw new KeyNotFoundException($"TP in-projection weight '{inprojKey}' not found in sharded weights.");
            var inprojF32 = inprojQuant == null ? _tpWeights[inprojKey] : null;

            var inprojPtr = new IntPtr[tp];
            var inprojType = new int[tp];
            var inprojNe0 = new long[tp];
            var inprojNe1 = new long[tp];
            var inprojBytes = new long[tp];
            var conv1dPtr = new IntPtr[tp];
            var dtBiasPtr = new IntPtr[tp];
            var aPtr = new IntPtr[tp];
            var convStatePtr = new IntPtr[tp];
            var deltaStatePtr = new IntPtr[tp];

            for (int r = 0; r < tp; r++)
            {
                if (inprojQuant != null)
                {
                    var qw = inprojQuant[r];
                    // CacheKey, not Data: it identifies the rank-resident device
                    // copy made by PrepareGgmlQuantizedWeightsForInferenceTP.
                    inprojPtr[r] = qw.CacheKey;
                    inprojType[r] = qw.GgmlType;
                    inprojNe0[r] = qw.Ne0;
                    inprojNe1[r] = qw.Ne1;
                    inprojBytes[r] = qw.RawBytes;
                }
                else
                {
                    var w = inprojF32[r];
                    inprojPtr[r] = (IntPtr)GetFloatPtr(w);
                    inprojType[r] = 0; // GGML_TYPE_F32
                    inprojNe0[r] = w.Sizes[1];
                    inprojNe1[r] = w.Sizes[0];
                    inprojBytes[r] = w.ElementCount() * sizeof(float);
                }

                conv1dPtr[r] = (IntPtr)GetFloatPtr(GetTpShardTensor(_ssmConv1dKey[layer], r));
                dtBiasPtr[r] = (IntPtr)GetFloatPtr(GetTpShardTensor(_ssmDtBiasKey[layer], r));
                aPtr[r] = (IntPtr)GetFloatPtr(GetTpShardTensor(_ssmAKey[layer], r));
                convStatePtr[r] = (IntPtr)GetFloatPtr(_tpConvState[layer][r]);
                deltaStatePtr[r] = (IntPtr)GetFloatPtr(_tpDeltaState[layer][r]);
            }

            var results = new Tensor[tp];
            long tGdn = Stopwatch.GetTimestamp();
            _tpGroup.RunPerRank(r =>
            {
                var alloc = _tpGroup.GetAllocator(r);
                var gated = new Tensor(alloc, DType.Float32, seqLen, localVDim);
                try
                {
                    GgmlBasicOps.Qwen35GdnLayerTP(
                        (IntPtr)GetFloatPtr(hidden[r]), Config.HiddenSize, seqLen,
                        attnNormPtr,
                        inprojPtr[r], inprojType[r], inprojNe0[r], inprojNe1[r], inprojBytes[r],
                        conv1dPtr[r], dtBiasPtr[r], aPtr[r], ssmNormPtr,
                        convStatePtr[r], deltaStatePtr[r],
                        (IntPtr)GetFloatPtr(gated),
                        localPackedDim, localQkvDim, localQkDim, localVDim,
                        localKHeads, localVHeads, _headKDim, _headVDim,
                        _convKernel, Config.Eps);
                }
                catch
                {
                    gated.Dispose();
                    throw;
                }
                InvalidateTensorDeviceCache(gated);
                results[r] = gated;
            });
            long gdnElapsed = Stopwatch.GetTimestamp() - tGdn;
            _gdnCudaNativeTicks += gdnElapsed;
            _tpGdnTicks += gdnElapsed;
            _gdnCudaNativeCalls++;

            return results;
        }

        /// <summary>
        /// Get a TP-sharded F32 tensor for a given weight name and rank.
        /// </summary>
        private Tensor GetTpShardTensor(string weightName, int rank)
        {
            if (_tpWeights.TryGetValue(weightName, out var shards))
                return shards[rank];
            // Fall back to a replicated weight (e.g. ssm_norm). It lives on rank 0,
            // so hand back a copy resident on THIS rank's GPU — the caller feeds it
            // straight to a rank-r kernel.
            if (_weights.TryGetValue(weightName, out var w))
                return ReplicaOnRank(w, rank);
            throw new KeyNotFoundException($"TP weight '{weightName}' not found.");
        }

        /// <summary>
        /// A replicated weight resident on rank r's GPU, so a per-rank kernel
        /// that reads it directly never crosses GPUs. Rank 0 aliases the
        /// original; other ranks get a lazily-cached copy. Cheap for the small
        /// norm weights this is used for.
        /// </summary>
        private Tensor ReplicaOnRank(Tensor weight, int rank) => TpReplicatedWeight(weight, rank);

        // ====================================================================
        // FFN block under TP (dense or MoE)
        // ====================================================================

        private Tensor[] FFNBlockTP(Tensor[] hidden, int layer, int seqLen)
        {
            bool isMoe = _isMoeLayer != null && _isMoeLayer[layer];

            if (isMoe)
                return MoEBlockTP(hidden, layer, seqLen);

            // Fast path: the whole dense FFN as one segmented graph per rank.
            if (TryQwen35FusedDenseFfnTP(hidden, layer, seqLen))
                return hidden;

            // Dense FFN: column-parallel gate_up → SiLU·mul → row-parallel down + AllReduce.
            int tp = TpDegree;

            // 1. Post-attention norm (replicated).
            Tensor[] normed = TpRMSNorm(hidden, _postAttnNormKey[layer]);

            // 2. Column-parallel gate/up.
            Tensor[] gateUp = TpColumnParallelLinear(normed[0], _ffnGateUpKey[layer]);
            for (int r = 0; r < tp; r++)
                normed[r].Dispose();

            // 3. Per-GPU SiLU·mul.
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

                Ops.SiLUMul(gate, gate, up);
                up.Dispose();
                gateResults[r] = gate;
            }

            // 4. Row-parallel down + AllReduce (result already on every rank).
            Tensor[] ffnReduced = TpRowParallelLinearAllRanks(gateResults, _ffnDownKey[layer]);
            for (int r = 0; r < tp; r++)
                gateResults[r].Dispose();

            // 5. Residual add.
            TpResidualAdd(hidden, ffnReduced);
            for (int r = 0; r < tp; r++)
                ffnReduced[r].Dispose();

            return hidden;
        }

        // ====================================================================
        // MoE block under TP (tensor-parallel experts)
        // ====================================================================

        private unsafe Tensor[] MoEBlockTP(Tensor[] hidden, int layer, int seqLen)
        {
            int tp = TpDegree;
            int hiddenSize = Config.HiddenSize;
            string prefix = $"blk.{layer}.";

            // 1. Post-attention norm (replicated).
            Tensor[] normed = TpRMSNorm(hidden, _postAttnNormKey[layer]);

            // 2. Router. The router weight is NOT sharded — it lives on rank 0 —
            // so the logits are computed ONCE there and replicated. Running
            // LinearForward per rank would mix a rank-r input with the rank-0
            // weight and a rank-0 output in a single kernel launch: a cross-device
            // dereference that faults the context (CUDA 719) or silently returns
            // garbage. Computing once is also the only way to guarantee every rank
            // routes each token to the same experts.
            var results = new Tensor[tp];

            long tRouter = Stopwatch.GetTimestamp();
            Tensor rank0Logits = LinearForward(normed[0], _ffnGateInpKey[layer]);
            _tpRouterTicks += Stopwatch.GetTimestamp() - tRouter;

            // Expert-parallel (GGML): each rank runs the batched kernel over the
            // whole experts it owns, so the layer is three dispatches per rank
            // instead of a per-(token, expert) loop.
            if (UsesExpertParallelMoE)
            {
                bool routeRowsAreLogits = _normTopKProb;
                Tensor epRoutes = routeRowsAreLogits ? rank0Logits : Ops.Softmax(null, rank0Logits);
                bool done = TryQwen35MoEExpertParallel(normed, results, epRoutes, routeRowsAreLogits, layer, seqLen);
                if (!ReferenceEquals(epRoutes, rank0Logits))
                    epRoutes.Dispose();

                if (done)
                {
                    rank0Logits.Dispose();
                    for (int r = 0; r < tp; r++)
                        normed[r].Dispose();
                    return FinishMoEBlockTP(hidden, results, tp);
                }
                // Fall through to the generic paths below with rank0Logits intact.
            }

            Tensor[] routerLogitsPerRank = BroadcastTensorToAllRanks(rank0Logits);
            rank0Logits.Dispose();

            // 2a. Fully on-device fused MoE per rank (device-resident expert shards,
            // no host round-trip). Each rank returns its partial contribution.
            bool allOnDevice = true;
            for (int r = 0; r < tp; r++)
            {
                Tensor devOut = TryMoEForwardTpOnDevice(normed[r], routerLogitsPerRank[r], layer, r, seqLen);
                if (devOut == null)
                {
                    allOnDevice = false;
                    break;
                }
                // TryMoEForwardTpOnDevice consumed (disposed) routerLogitsPerRank[r].
                routerLogitsPerRank[r] = null;
                results[r] = devOut;
            }

            if (allOnDevice)
            {
                for (int r = 0; r < tp; r++)
                    normed[r].Dispose();
                return FinishMoEBlockTP(hidden, results, tp);
            }

            // Fused path unavailable — undo the partial attempt and fall back to the
            // host loop below.
            for (int r = 0; r < tp; r++)
            {
                results[r]?.Dispose();
                results[r] = null;
            }

            // Host fallback: the routing decision is derived once from the rank-0
            // logits and shared by every rank, so all ranks stay in lockstep.
            // If the fused path already consumed rank 0's copy before bailing out on a
            // later rank, recompute it here — and own it, so it is disposed below
            // rather than leaked.
            Tensor routerSource = routerLogitsPerRank[0];
            if (routerSource == null)
            {
                routerSource = BroadcastRouterLogitsForFallback(normed[0], layer);
                routerLogitsPerRank[0] = routerSource;
            }
            Tensor routerData = _normTopKProb ? routerSource : Ops.Softmax(null, routerSource);
            float[] routePtr = TensorToFloatArray(routerData);
            if (!ReferenceEquals(routerData, routerSource)) routerData.Dispose();
            for (int r = 0; r < tp; r++)
                routerLogitsPerRank[r]?.Dispose();

            for (int r = 0; r < tp; r++)
            {
                var alloc = _tpGroup.GetAllocator(r);
                var localInput = normed[r];

                // Accumulate expert outputs.
                var output = new Tensor(alloc, DType.Float32, seqLen, hiddenSize);
                Ops.Fill(output, 0f);

                // Per-token routing: each token selects its own top-K experts.
                for (int t = 0; t < seqLen; t++)
                {
                    // Extract this token's router logits (numExperts elements).
                    float[] tokenLogits = new float[_numExperts];
                    Array.Copy(routePtr, t * _numExperts, tokenLogits, 0, _numExperts);

                    var (topExperts, routeWeights) = SelectTopKExperts(tokenLogits, _numExpertsUsed);

                    // Extract this token's input row [1, hiddenSize].
                    using var tokenInput = localInput.Narrow(0, t, 1);

                    for (int k = 0; k < _numExpertsUsed; k++)
                    {
                        int expertIdx = topExperts[k];
                        float weight = routeWeights[k];

                        string gateKey = prefix + $"ffn_gate_exps.{expertIdx}.weight";
                        string upKey = prefix + $"ffn_up_exps.{expertIdx}.weight";
                        string downKey = prefix + $"ffn_down_exps.{expertIdx}.weight";

                        // Column-parallel gate/up (per-rank shard).
                        Tensor gateOut = TpExpertLinear(tokenInput, gateKey, r, 1);
                        Tensor upOut = TpExpertLinear(tokenInput, upKey, r, 1);

                        // SiLU·mul.
                        Ops.SiLUMul(gateOut, gateOut, upOut);
                        upOut.Dispose();

                        // Row-parallel down (per-rank shard, partial result).
                        Tensor downOut = TpExpertLinear(gateOut, downKey, r, 1);
                        gateOut.Dispose();

                        // Weighted accumulate into the token's output row.
                        Ops.Mul(downOut, downOut, weight);
                        using var outputRow = output.Narrow(0, t, 1);
                        Ops.Add(outputRow, outputRow, downOut);
                        downOut.Dispose();
                    }
                }

                // Shared experts (if present) — apply to ALL tokens.
                if (_hasSharedExperts != null && _hasSharedExperts[layer])
                {
                    Tensor sharedGate = TpExpertLinear(localInput, prefix + "ffn_gate_shexp.weight", r, seqLen);
                    Tensor sharedUp = TpExpertLinear(localInput, prefix + "ffn_up_shexp.weight", r, seqLen);
                    Ops.SiLUMul(sharedGate, sharedGate, sharedUp);
                    sharedUp.Dispose();
                    Tensor sharedDown = TpExpertLinear(sharedGate, prefix + "ffn_down_shexp.weight", r, seqLen);
                    sharedGate.Dispose();

                    // Shared expert gate (sigmoid scalar). The gate vector is
                    // replicated on rank 0, so the dot product is evaluated against
                    // the rank-0 input — identical on every rank — and the resulting
                    // scalar scales this rank's partial. Reading it with a rank-r
                    // input would pair a rank-r pointer with a rank-0 pointer.
                    if (_hasSharedExpertGate != null && _hasSharedExpertGate[layer])
                    {
                        var gateVec = _ffnGateInpShexpVec?[layer];
                        if (gateVec != null)
                        {
                            float gateVal = ComputeSharedExpertGate(normed[0], gateVec);
                            Ops.Mul(sharedDown, sharedDown, gateVal);
                        }
                        Ops.Add(output, output, sharedDown);
                    }
                    else
                    {
                        Ops.Add(output, output, sharedDown);
                    }
                    sharedDown.Dispose();
                }

                results[r] = output;
            }

            for (int r = 0; r < tp; r++)
                normed[r].Dispose();

            return FinishMoEBlockTP(hidden, results, tp);
        }

        /// <summary>
        /// Shared tail of the TP MoE block: AllReduce the per-rank partial expert
        /// outputs, add the residual and re-replicate the hidden state for the next
        /// layer.
        /// </summary>
        private Tensor[] FinishMoEBlockTP(Tensor[] hidden, Tensor[] results, int tp)
        {
            // AllReduce across ranks (sum partial expert results). This leaves the
            // identical sum on every rank.
            long tAr = Stopwatch.GetTimestamp();
            _tpGroup.AllReduce(results);
            _tpAllReduceTicks += Stopwatch.GetTimestamp() - tAr;

            // Residual add. `hidden` entered replicated and every rank adds the same
            // reduced value, so it stays replicated — the block used to follow this
            // with a full BroadcastTensorToAllRanks(hidden[0]), re-sending a value
            // each rank already held.
            TpResidualAdd(hidden, results);
            for (int r = 0; r < tp; r++)
                results[r].Dispose();

            return hidden;
        }

        /// <summary>
        /// Router logits on rank 0 for the host fallback, used only when the fused
        /// device path consumed the pre-broadcast copies before bailing out.
        /// </summary>
        private Tensor BroadcastRouterLogitsForFallback(Tensor normed0, int layer)
            => LinearForward(normed0, _ffnGateInpKey[layer]);

        /// <summary>
        /// Linear forward using a TP-sharded expert weight for a specific rank.
        /// </summary>
        private Tensor TpExpertLinear(Tensor input, string weightName, int rank, int seqLen)
        {
            var alloc = _tpGroup.GetAllocator(rank);

            if (_tpQuantWeights.TryGetValue(weightName, out var qShards))
            {
                var qw = qShards[rank];
                int outDim = (int)qw.Ne1;
                var result = new Tensor(alloc, DType.Float32, seqLen, outDim);
                var localInput = ReplicateTensorToRank(input, rank);
                AddmmQuantManaged(result, localInput, qw);
                if (!ReferenceEquals(localInput, input)) localInput.Dispose();
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

        private (int[] experts, float[] weights) SelectTopKExperts(float[] routerLogits, int topK)
        {
            int numExperts = routerLogits.Length;
            var indices = new int[numExperts];
            for (int i = 0; i < numExperts; i++) indices[i] = i;
            Array.Sort(indices, (a, b) => routerLogits[b].CompareTo(routerLogits[a]));

            var topExperts = new int[topK];
            var topWeights = new float[topK];
            for (int k = 0; k < topK; k++)
                topExperts[k] = indices[k];

            if (_normTopKProb)
            {
                // routerLogits are RAW logits: softmax over the selected top-K
                // experts (matches the non-TP SelectTopKRouteWeights). The previous
                // raw-logit / raw-sum renormalization produced wrong (even negative)
                // expert weights and corrupted every MoE layer.
                float maxLogit = float.NegativeInfinity;
                for (int k = 0; k < topK; k++)
                {
                    float v = routerLogits[topExperts[k]];
                    if (v > maxLogit) maxLogit = v;
                }
                float sum = 0f;
                for (int k = 0; k < topK; k++)
                {
                    float w = MathF.Exp(routerLogits[topExperts[k]] - maxLogit);
                    topWeights[k] = w;
                    sum += w;
                }
                if (sum > 0f)
                {
                    float inv = 1.0f / sum;
                    for (int k = 0; k < topK; k++)
                        topWeights[k] *= inv;
                }
            }
            else
            {
                // routerLogits are already full-softmax probabilities (the caller
                // pre-softmaxed): use the selected probabilities directly, with no
                // further renormalization.
                for (int k = 0; k < topK; k++)
                    topWeights[k] = routerLogits[topExperts[k]];
            }

            return (topExperts, topWeights);
        }

        private float ComputeSharedExpertGate(Tensor input, Tensor gateVec)
        {
            // dot(input, gateVec) → sigmoid
            unsafe
            {
                float* inputPtr = GetFloatPtr(input);
                float* gatePtr = GetFloatPtr(gateVec);
                int dim = (int)gateVec.ElementCount();
                float dot = 0;
                for (int i = 0; i < dim; i++)
                    dot += inputPtr[i] * gatePtr[i];
                return 1.0f / (1.0f + MathF.Exp(-dot));
            }
        }

        // ====================================================================
        // TP-aware ResetKVCache
        // ====================================================================

        private void ResetTpKVCache()
        {
            int tp = TpDegree;
            int previousRank = IsGgmlBackend ? GgmlBasicOps.GetActiveRank() : 0;

            // The reset below invalidates (frees) the per-rank state device
            // buffers the persistent TP decode graphs reference; drop them first.
            DropTpFusedDecodeGraphs();

            for (int l = 0; l < TotalLayerCount; l++)
            {
                if (!_isRecurrent[l] && _tpKvCacheK[l] != null)
                {
                    for (int r = 0; r < tp; r++)
                    {
                        ResetCacheTensor(_tpKvCacheK[l][r]);
                        ResetCacheTensor(_tpKvCacheV[l][r]);
                    }
                }
                else if (_isRecurrent[l] && _tpDeltaState[l] != null)
                {
                    for (int r = 0; r < tp; r++)
                    {
                        Ops.Fill(_tpDeltaState[l][r], 0);
                        if (_tpConvState[l][r] != null)
                            Ops.Fill(_tpConvState[l][r], 0);

                        // On GGML the recurrent state lives on the device between
                        // calls, keyed by these host pointers. Zeroing the host
                        // copy alone would leave the GPU running on the old state,
                        // so drop the device copy and let the next call re-upload
                        // the zeros. The cache is per rank, hence the rank switch.
                        if (IsGgmlBackend)
                        {
                            GgmlBasicOps.SetActiveRank(r);
                            InvalidateTensorDeviceCache(_tpDeltaState[l][r]);
                            if (_tpConvState[l][r] != null)
                                InvalidateTensorDeviceCache(_tpConvState[l][r]);
                        }
                    }
                    _tpConvWriteIdx[l] = 0;
                }
            }

            if (IsGgmlBackend)
                GgmlBasicOps.SetActiveRank(previousRank);
        }

        // ====================================================================
        // TP-aware Dispose
        // ====================================================================

        private void DisposeTpState()
        {
            // Native TP and verify graphs are released by Qwen35Model.Dispose
            // before reaching this method, while every bound tensor is alive.
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

            if (_tpDeltaState != null)
            {
                for (int l = 0; l < _tpDeltaState.Length; l++)
                {
                    if (_tpDeltaState[l] == null) continue;
                    for (int r = 0; r < _tpDeltaState[l].Length; r++)
                    {
                        _tpDeltaState[l][r]?.Dispose();
                        _tpConvState[l]?[r]?.Dispose();
                    }
                }
            }

            DisposeTpWeightReplicaCache();
            FreeQwenTpMoETables();
        }
    }
}
