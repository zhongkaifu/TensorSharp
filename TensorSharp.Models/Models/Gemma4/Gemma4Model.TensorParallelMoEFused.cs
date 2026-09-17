// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    /// <summary>
    /// Fused tensor-parallel forward for the MoE Gemma 4 variants (26B-A4B).
    ///
    /// The MoE trunk has a whole-model kernel on the single-GPU path
    /// (<c>TSGgml_Gemma4MoEModelDecode</c> / <c>...Verify</c>): attention, dense
    /// FFN, router and stacked experts for all 30 layers in one graph. Tensor
    /// parallelism could not use it, so <c>--tp 2</c> ran the whole trunk
    /// op-at-a-time and landed at 5.1 tok/s decode against 48.5 on one GPU.
    ///
    /// **Why whole-expert sharding could not be fused.** The MoE layers were split
    /// by whole expert — rank r owns experts [r·P, (r+1)·P). That is a zero-copy
    /// view and it works for a per-layer managed dispatch, but the fused graph
    /// computes the router *inside* itself and `ggml_top_k` returns *global*
    /// expert ids. Rank r would have to rewrite ids it does not own to some local
    /// filler, and ggml-cuda's `mul_mat_id` aborts when a token repeats an id: it
    /// binds each (expert, token) pair with a `break` after the first matching
    /// slot and then asserts the binding count equals `n_tokens * n_used`
    /// (ggml-cuda.cu, `ggml_cuda_mul_mat_id`). Real top-k never repeats, so any
    /// filler scheme that can collide is a crash, and with P local slots and P
    /// foreign experts no static map avoids collisions.
    ///
    /// **What this does instead.** Split *inside* each expert, the Megatron way:
    /// every rank keeps all 128 experts and holds 1/tp of each expert's FFN width
    /// (gate/up column-parallel on the intermediate dim, down row-parallel on it).
    /// `sel` stays global, so the graph is byte-for-byte the single-GPU graph with
    /// narrower expert matrices, and the per-rank expert bytes are unchanged —
    /// half the model either way. The down projection then produces a partial sum,
    /// which is the same cut the attention output and the dense FFN already need.
    ///
    /// The cost is that the slices must be materialized at load (the whole-expert
    /// view was free): rank r's share of a stacked tensor is a strided gather, not
    /// a byte range. That is a one-time ~6 GB copy per rank, parallelized over
    /// layers.
    /// </summary>
    public partial class Gemma4Model
    {
        // Megatron-sliced stacked experts, [layer][rank]. Distinct from
        // _tpStackedGate/_tpStackedDown, which are whole-expert views.
        private StackedExpertWeights[][] _tpSlicedGateUp;
        private StackedExpertWeights[][] _tpSlicedDown;
        private bool _tpMoeFusedReady;
        private bool _tpMoeFusedLogged;
        private IntPtr[] _tpMoePlans;
        private Gemma4MoELayerDecodeArgs[][] _tpMoeArgs;   // [rank][layer]

        /// <summary>Disable with TS_GEMMA4_TP_FUSED_MOE=0.</summary>
        private static readonly bool _tpFusedMoeEnabled =
            Environment.GetEnvironmentVariable("TS_GEMMA4_TP_FUSED_MOE") != "0";

        /// <summary>
        /// True once the Megatron-sliced expert shards exist, i.e. the MoE trunk
        /// can run through the fused whole-model kernel per rank.
        /// </summary>
        private bool UsesTensorSlicedMoE => _tpSlicedGateUp != null;

        /// <summary>
        /// Build rank-local slices of every MoE layer's stacked expert tensors.
        /// Returns false (leaving the whole-expert path in place) when the shapes
        /// do not divide, when the model does not expose fused stacked experts, or
        /// when the fused kernel is unavailable.
        /// </summary>
        private bool BuildGemma4TensorSlicedExpertShards()
        {
            if (!_tpFusedMoeEnabled || !IsGgmlBackend || _numExperts <= 0)
                return false;
            // One driving thread submits every rank, so this is single-process.
            if (GlobalTpDegree != TpDegree || TpDegree <= 1)
                return false;
            if (!GgmlBasicOps.TensorParallelFusedAvailable(TpDegree))
                return false;
            // The fused kernel wants the FUSED gate_up stack; a model that kept
            // gate and up apart keeps the whole-expert path.
            if (_layerStackedGate == null || _layerStackedDown == null)
                return false;
            if (_layerStackedUp != null)
            {
                for (int l = 0; l < Config.NumLayers; l++)
                    if (_layerStackedUp[l] != null) return false;
            }
            // The fused MoE trunk kernel has no PLE / KV-donor / block-quant-KV
            // branch, and it needs every layer to be MoE.
            if (_pleDim != 0 || _kvDonorMap.Count != 0 || _kvCacheDtype.IsBlockQuantized())
                return false;
            for (int l = 0; l < Config.NumLayers; l++)
                if (!HasMoE(l)) return false;

            int tp = TpDegree;
            int n = Config.NumLayers;
            var gateUp = new StackedExpertWeights[n][];
            var down = new StackedExpertWeights[n][];

            // Validate divisibility once, on layer 0 — every layer shares the shape.
            {
                var g0 = _layerStackedGate[0];
                var d0 = _layerStackedDown[0];
                if (g0 == null || d0 == null) return false;
                long ffMoe = g0.PerExpertNe1 / 2;          // gate_up is [hidden, 2*ff]
                if (g0.PerExpertNe1 % 2 != 0 || ffMoe % tp != 0) return false;
                if (d0.PerExpertNe0 != ffMoe) return false;
                // The down slice cuts ne0, which for a quantized matrix has to land
                // on a block boundary.
                long downBlock = GgufFile.GetBlockSize((GgmlTensorType)d0.GgmlType);
                if (downBlock <= 0 || (ffMoe / tp) % downBlock != 0) return false;
            }

            var sw = System.Diagnostics.Stopwatch.StartNew();
            long totalBytes = 0;

            for (int layer = 0; layer < n; layer++)
            {
                gateUp[layer] = new StackedExpertWeights[tp];
                down[layer] = new StackedExpertWeights[tp];
            }

            // Parallelize over (layer, rank, projection): the down slice is a
            // per-row gather (2816 rows x 128 experts x 30 layers of ~176-byte
            // copies), so the work is dominated by copy count, not bytes, and
            // spreading it over every core is what keeps this off the critical
            // path. The two ranks and the two projections are independent.
            var work = new (int layer, int rank, bool isDown)[n * tp * 2];
            int w = 0;
            for (int layer = 0; layer < n; layer++)
                for (int r = 0; r < tp; r++)
                {
                    work[w++] = (layer, r, false);
                    work[w++] = (layer, r, true);
                }

            Parallel.ForEach(work, item =>
            {
                // An offloaded layer never runs on the accelerator, so it needs
                // no per-rank slice: the host reads the whole expert stack out of
                // the GGUF mmap. Skipping it is not just a load-time saving —
                // materializing the slices would cost a second full copy of the
                // experts in RAM, which is the opposite of what --cpu-moe is for.
                if (MoeCpuOffloadConfig.IsLayerOnCpu(item.layer))
                    return;
                if (item.isDown)
                    down[item.layer][item.rank] = SliceDownRowParallel(_layerStackedDown[item.layer], item.rank, tp);
                else
                    gateUp[item.layer][item.rank] = SliceGateUpColumnParallel(_layerStackedGate[item.layer], item.rank, tp);
            });

            for (int layer = 0; layer < n; layer++)
                for (int r = 0; r < tp; r++)
                    totalBytes += (gateUp[layer][r]?.TotalRawBytes ?? 0) + (down[layer][r]?.TotalRawBytes ?? 0);

            _tpSlicedGateUp = gateUp;
            _tpSlicedDown = down;
            sw.Stop();

            long ffPerRank = (_layerStackedGate[0].PerExpertNe1 / 2) / tp;
            Console.WriteLine(
                $"  Gemma4 MoE: tensor-sliced experts across {tp} GPU(s), all {_numExperts} experts on each, " +
                $"intermediate {_layerStackedGate[0].PerExpertNe1 / 2} -> {ffPerRank} per GPU " +
                $"({totalBytes / 1024 / 1024} MB sliced in {sw.ElapsedMilliseconds} ms).");
            return true;
        }

        /// <summary>
        /// Column-parallel slice of a fused [hidden, 2·ff] stacked expert tensor:
        /// rank r takes its band of the gate half and the matching band of the up
        /// half, keeping the [gate_r | up_r] layout the kernel's view arithmetic
        /// assumes. Rows are ne1 entries of ne0 elements each, so a band of rows is
        /// a contiguous byte range per expert — two memcpys per expert.
        /// </summary>
        private static unsafe StackedExpertWeights SliceGateUpColumnParallel(StackedExpertWeights src, int rank, int tp)
        {
            long rowBytes = NativeDequant.RowSize(src.GgmlType, src.PerExpertNe0);
            long ffMoe = src.PerExpertNe1 / 2;
            long ffLocal = ffMoe / tp;
            long halfBytes = ffLocal * rowBytes;
            long dstPerExpert = 2 * halfBytes;
            long total = dstPerExpert * src.NumExperts;

            IntPtr buffer = QuantizedWeight.AllocateBuffer(total);
            byte* s = (byte*)src.Data.ToPointer();
            byte* dptr = (byte*)buffer.ToPointer();
            long srcPerExpert = src.PerExpertRawBytes;
            long gateOff = rank * ffLocal * rowBytes;
            long upOff = (ffMoe + rank * ffLocal) * rowBytes;

            for (int e = 0; e < src.NumExperts; e++)
            {
                byte* se = s + e * srcPerExpert;
                byte* de = dptr + e * dstPerExpert;
                Buffer.MemoryCopy(se + gateOff, de, halfBytes, halfBytes);
                Buffer.MemoryCopy(se + upOff, de + halfBytes, halfBytes, halfBytes);
            }

            return new StackedExpertWeights(buffer, src.GgmlType,
                src.PerExpertNe0, 2 * ffLocal, src.NumExperts, total,
                isExternalView: false, ownerToken: null, ownedBuffer: buffer);
        }

        /// <summary>
        /// Row-parallel slice of a [ff, hidden] stacked expert tensor: rank r takes
        /// a band of the ne0 (input) dimension. For a quantized matrix that is a
        /// band of whole blocks out of each of the `hidden` rows, so it is a gather
        /// rather than a byte range.
        /// </summary>
        private static unsafe StackedExpertWeights SliceDownRowParallel(StackedExpertWeights src, int rank, int tp)
        {
            var type = (GgmlTensorType)src.GgmlType;
            long blockSize = GgufFile.GetBlockSize(type);
            long typeSize = GgufFile.GetTypeSize(type);
            long blocksPerRow = src.PerExpertNe0 / blockSize;
            long blocksPerShard = blocksPerRow / tp;
            long ne0Local = blocksPerShard * blockSize;
            long srcRowBytes = NativeDequant.RowSize(src.GgmlType, src.PerExpertNe0);
            long dstRowBytes = blocksPerShard * typeSize;
            long rows = src.PerExpertNe1;
            long dstPerExpert = rows * dstRowBytes;
            long total = dstPerExpert * src.NumExperts;

            IntPtr buffer = QuantizedWeight.AllocateBuffer(total);
            byte* s = (byte*)src.Data.ToPointer();
            byte* dptr = (byte*)buffer.ToPointer();
            long srcPerExpert = src.PerExpertRawBytes;
            long srcBlockOffset = rank * blocksPerShard * typeSize;

            for (int e = 0; e < src.NumExperts; e++)
            {
                byte* se = s + e * srcPerExpert + srcBlockOffset;
                byte* de = dptr + e * dstPerExpert;
                for (long row = 0; row < rows; row++)
                    Buffer.MemoryCopy(se + row * srcRowBytes, de + row * dstRowBytes, dstRowBytes, dstRowBytes);
            }

            return new StackedExpertWeights(buffer, src.GgmlType,
                ne0Local, rows, src.NumExperts, total,
                isExternalView: false, ownerToken: null, ownedBuffer: buffer);
        }

        /// <summary>
        /// Make one rank's sliced expert stack device-resident. Without this the
        /// first forward pays the whole ~6 GB upload per rank. Runs inside the
        /// per-rank preload fan-out: the calling thread is pinned to the rank's
        /// GPU, and every rank uploads its slices concurrently.
        /// </summary>
        private void PreloadGemma4TensorSlicedExpertsForRank(int rank, long[] bytesPerRank, int[] countPerRank)
        {
            for (int layer = 0; layer < Config.NumLayers; layer++)
            {
                // Offloaded layers have no slice at all (see
                // BuildGemma4TensorSlicedExpertShards); PreloadStackedShard
                // no-ops on the nulls, and that omission IS the VRAM saving.
                PreloadStackedShard(_tpSlicedGateUp[layer]?[rank], bytesPerRank, countPerRank, rank);
                PreloadStackedShard(_tpSlicedDown[layer]?[rank], bytesPerRank, countPerRank, rank);
            }
        }

        // ====================================================================
        // Per-rank descriptors + drivers
        // ====================================================================

        /// <summary>
        /// Build the per-rank layer descriptors for the fused MoE trunk. Called
        /// once the sliced experts and the per-rank KV caches exist.
        /// </summary>
        private void BuildGemma4TpMoeArgs()
        {
            _tpMoeFusedReady = false;
            if (!UsesTensorSlicedMoE || _tpDecodeArrays == null)
                return;

            int tp = TpDegree;
            int n = Config.NumLayers;
            _tpMoeArgs = new Gemma4MoELayerDecodeArgs[tp][];
            for (int r = 0; r < tp; r++)
                _tpMoeArgs[r] = new Gemma4MoELayerDecodeArgs[n];
            _tpMoePlans = new IntPtr[tp];
            _tpMoeFusedReady = true;
        }

        /// <summary>
        /// Fill rank <paramref name="rank"/>'s descriptor for one layer. Weight
        /// and KV pointers come from that rank's decode arrays (already sharded);
        /// the norms, router and per-expert scales are replicated.
        /// </summary>
        private unsafe bool TryBuildTpMoeLayerArgs(int rank, int layer, IntPtr hiddenPtr, out Gemma4MoELayerDecodeArgs args)
        {
            args = default;
            var a = _tpDecodeArrays[rank];
            if (a.Qkv[layer] == IntPtr.Zero || a.O[layer] == IntPtr.Zero
                || a.Gu[layer] == IntPtr.Zero || a.Down[layer] == IntPtr.Zero)
                return false;

            string prefix = $"blk.{layer}";
            if (!_weights.TryGetValue($"{prefix}.ffn_gate_inp.weight", out var routerW)) return false;
            string preNorm2Key = _weights.ContainsKey($"{prefix}.pre_ffw_norm_2.weight")
                ? $"{prefix}.pre_ffw_norm_2.weight" : $"{prefix}.ffn_pre_norm_2.weight";
            if (!_weights.TryGetValue(preNorm2Key, out var preNorm2W)) return false;
            string postNorm1Key = _weights.ContainsKey($"{prefix}.post_ffw_norm_1.weight")
                ? $"{prefix}.post_ffw_norm_1.weight" : $"{prefix}.ffn_post_norm_1.weight";
            if (!_weights.TryGetValue(postNorm1Key, out var postNorm1W)) return false;
            string postNorm2Key = _weights.ContainsKey($"{prefix}.post_ffw_norm_2.weight")
                ? $"{prefix}.post_ffw_norm_2.weight" : $"{prefix}.ffn_post_norm_2.weight";
            if (!_weights.TryGetValue(postNorm2Key, out var postNorm2W)) return false;

            bool isLocal = IsLocalLayer(layer);
            IntPtr freqPtr = IntPtr.Zero;
            int freqLen = 0;
            if (!isLocal && _weights.TryGetValue("rope_freqs.weight", out var freqTensor))
            {
                freqPtr = (IntPtr)GetFloatPtr(freqTensor);
                freqLen = (int)freqTensor.ElementCount();
            }

            IntPtr gateInpScalePtr = IntPtr.Zero;
            if (_weights.TryGetValue($"{prefix}.ffn_gate_inp.scale", out var giScale))
                gateInpScalePtr = (IntPtr)GetFloatPtr(giScale);
            IntPtr downScalePtr = IntPtr.Zero;
            string downScaleKey = _weights.ContainsKey($"{prefix}.ffn_down_exps.scale")
                ? $"{prefix}.ffn_down_exps.scale" : $"{prefix}.ffn_gate_inp.per_expert_scale";
            if (_weights.TryGetValue(downScaleKey, out var downScaleT))
                downScalePtr = (IntPtr)GetFloatPtr(downScaleT);

            // MoE CPU offload: an offloaded layer is evaluated once on the host,
            // over the WHOLE expert stack, and every rank takes that one result
            // (nothing reduces it — see tsg::HostMoeSegment::tp_reduced). So it
            // has no per-rank slice and the descriptor points at the unsharded
            // tensors; nothing uploads them.
            bool layerOnCpu = MoeCpuOffloadConfig.IsLayerOnCpu(layer);
            var gu = layerOnCpu ? _layerStackedGate[layer] : _tpSlicedGateUp[layer][rank];
            var dn = layerOnCpu ? _layerStackedDown[layer] : _tpSlicedDown[layer][rank];
            if (gu == null || dn == null)
                return false;

            args = new Gemma4MoELayerDecodeArgs
            {
                Hidden = hiddenPtr,
                AttnNormW = a.AttnNorm[layer],
                QkvW = a.Qkv[layer], QkvType = a.QkvType[layer], QkvNe0 = a.QkvNe0[layer], QkvNe1 = a.QkvNe1[layer], QkvBytes = a.QkvBytes[layer],
                KW = a.K[layer], KType = a.KType[layer], KNe0 = a.KNe0[layer], KNe1 = a.KNe1[layer], KBytes = a.KBytes[layer],
                VW = a.V[layer], VType = a.VType[layer], VNe0 = a.VNe0[layer], VNe1 = a.VNe1[layer], VBytes = a.VBytes[layer],
                QNormW = a.QNorm[layer],
                KNormW = a.KNorm[layer],
                OW = a.O[layer], OType = a.OType[layer], ONe0 = a.ONe0[layer], ONe1 = a.ONe1[layer], OBytes = a.OBytes[layer],
                PostAttnNormW = a.PostAttnNorm[layer],
                KCache = a.KCache[layer], VCache = a.VCache[layer],
                FreqFactors = freqPtr, FreqFactorsLen = freqLen,
                FfnNormW = a.FfnNorm[layer],
                GuW = a.Gu[layer], GuType = a.GuType[layer], GuNe0 = a.GuNe0[layer], GuNe1 = a.GuNe1[layer], GuBytes = a.GuBytes[layer],
                DownW = a.Down[layer], DownType = a.DownType[layer], DownNe0 = a.DownNe0[layer], DownNe1 = a.DownNe1[layer], DownBytes = a.DownBytes[layer],
                PostFfwNorm1W = (IntPtr)GetFloatPtr(postNorm1W),
                GateInpW = (IntPtr)GetFloatPtr(routerW),
                GateInpScale = gateInpScalePtr,
                PreFfwNorm2W = (IntPtr)GetFloatPtr(preNorm2W),
                GateUpExps = gu.Data, GueType = gu.GgmlType, GueNe0 = gu.PerExpertNe0, GueNe1 = gu.PerExpertNe1, GueBytes = gu.TotalRawBytes,
                DownExps = dn.Data, DeType = dn.GgmlType, DeNe0 = dn.PerExpertNe0, DeNe1 = dn.PerExpertNe1, DeBytes = dn.TotalRawBytes,
                DownExpsScale = downScalePtr,
                CpuMoe = layerOnCpu ? 1 : 0,
                PostFfwNorm2W = (IntPtr)GetFloatPtr(postNorm2W),
                PostFfwNormW = a.PostFfnNorm[layer],

                StructBytes = Marshal.SizeOf<Gemma4MoELayerDecodeArgs>(),
                HiddenSize = Config.HiddenSize,
                // Per-rank head counts: the column-parallel QKV shard produces this
                // rank's heads only, and its KV cache holds just those.
                NumHeads = Config.NumHeads / TpDegree,
                NumKvHeads = a.KvHeads[layer],
                HeadDim = a.HeadDim[layer],
                CacheSize = a.CacheSize[layer],
                IsLocal = a.IsLocal[layer],
                IsShared = 0,
                SlidingWindow = _slidingWindow,
                Position = 0,
                RopeNDims = a.RopeNDims[layer],
                KvCacheType = _kvCacheDtype.GgmlType(),
                // Expert COUNT is unchanged: every rank keeps all experts and holds
                // a slice of each one's width, so the in-graph router's global ids
                // index the same experts on every GPU.
                NumExperts = _numExperts,
                NumExpertsUsed = _numExpertsUsed,
                SeparateQkv = a.K[layer] != IntPtr.Zero ? 1 : 0,
                Eps = Config.Eps,
                RopeBase = a.RopeBase[layer],
                InvSqrtHidden = 1.0f / MathF.Sqrt(Config.HiddenSize),
                LayerOutputScale = a.LayerScalar[layer],
            };
            return true;
        }

        /// <summary>
        /// One fused tensor-parallel MoE decode step: the whole 30-layer trunk as
        /// one segmented graph per rank, cut at the three row-parallel projections
        /// per layer. Returns false when the native side declines, leaving the
        /// caller on its per-op forward.
        /// </summary>
        private unsafe bool TryGemma4FusedMoEModelDecodeTP(Tensor hidden, int startPos, float[] logitsOut)
        {
            if (!_tpMoeFusedReady || logitsOut == null)
                return false;
            if (!_fdFoldLmHead
                || !_weights.TryGetValue("output_norm.weight", out var finalNormT)
                || !_quantWeights.TryGetValue(_hasTiedOutput ? "token_embd.weight" : "output.weight", out var lmqw))
                return false;

            int tp = TpDegree;
            int n = Config.NumLayers;
            IntPtr hiddenPtr = (IntPtr)GetFloatPtr(hidden);
            IntPtr finalNormPtr = (IntPtr)GetFloatPtr(finalNormT);

            int previousRank = GgmlBasicOps.GetActiveRank();
            var planSlot = new IntPtr[1];
            try
            {
                fixed (float* logitsPtr = logitsOut)
                {
                    for (int r = 0; r < tp; r++)
                    {
                        for (int l = 0; l < n; l++)
                        {
                            if (!TryBuildTpMoeLayerArgs(r, l, hiddenPtr, out _tpMoeArgs[r][l]))
                            {
                                _tpMoeFusedReady = false;
                                return false;
                            }
                        }

                        GgmlBasicOps.SetActiveRank(r);
                        planSlot[0] = IntPtr.Zero;
                        // Only rank 0 folds the final norm + LM head: it is the
                        // largest tensor left after sharding.
                        bool fold = r == 0;
                        GgmlBasicOps.Gemma4MoEModelDecode(
                            _tpMoeArgs[r], n, hiddenPtr, Config.HiddenSize, startPos,
                            fold ? (IntPtr)logitsPtr : IntPtr.Zero, fold ? Config.VocabSize : 0,
                            fold ? lmqw.CacheKey : IntPtr.Zero, lmqw.GgmlType, lmqw.Ne0, lmqw.Ne1, lmqw.RawBytes,
                            fold ? finalNormPtr : IntPtr.Zero, _finalLogitSoftcap,
                            tpDegree: tp, tpPlanOut: planSlot);

                        if (planSlot[0] == IntPtr.Zero)
                        {
                            WarnTpMoeFusedDeclined("decode", 1);
                            return false;
                        }
                        _tpMoePlans[r] = planSlot[0];
                    }

                    GgmlBasicOps.TensorParallelExecutePlans(_tpMoePlans);
                }
            }
            catch (InvalidOperationException)
            {
                return false;
            }
            finally
            {
                GgmlBasicOps.SetActiveRank(previousRank);
            }

            _tpKvDeviceDirty = true;
            if (!_tpMoeFusedLogged)
            {
                _tpMoeFusedLogged = true;
                Console.WriteLine($"  Gemma4 fused tensor-parallel MoE decode active ({tp} GPUs, " +
                    $"{3 * n} AllReduce points/token).");
            }
            return true;
        }

        /// <summary>
        /// Fused tensor-parallel MoE prefill over N tokens. Writes the trunk
        /// output back into <paramref name="hidden"/>; the caller owns the final
        /// norm and the LM head.
        /// </summary>
        private unsafe bool TryGemma4FusedMoEModelVerifyTP(Tensor hidden, int startPos, int n,
            HashSet<int> exceptPositions = null)
        {
            if (!_tpMoeFusedReady || n <= 1)
                return false;
            // Multimodal spans: one byte per chunk token, honoured by the kernel at
            // any startPos.
            byte[] isExcept = ChunkSoftTokenMask(exceptPositions, startPos, n);

            int tp = TpDegree;
            int layers = Config.NumLayers;
            IntPtr hiddenPtr = (IntPtr)GetFloatPtr(hidden);

            int previousRank = GgmlBasicOps.GetActiveRank();
            var planSlot = new IntPtr[1];
            try
            {
                for (int r = 0; r < tp; r++)
                {
                    for (int l = 0; l < layers; l++)
                    {
                        if (!TryBuildTpMoeLayerArgs(r, l, hiddenPtr, out _tpMoeArgs[r][l]))
                            return false;
                    }

                    GgmlBasicOps.SetActiveRank(r);
                    planSlot[0] = IntPtr.Zero;
                    if (!GgmlBasicOps.Gemma4MoEModelVerify(
                            _tpMoeArgs[r], layers, hiddenPtr, Config.HiddenSize, startPos, n,
                            mmIsExcept: isExcept, tpDegree: tp, tpPlanOut: planSlot)
                        || planSlot[0] == IntPtr.Zero)
                    {
                        // Say why, once. The per-op TP fallback below is orders of
                        // magnitude slower (and, with tensor-sliced experts, cannot
                        // serve this layer at all), so a silent decline reads as a
                        // hang or a crash rather than as a fallback.
                        WarnTpMoeFusedDeclined("prefill", n);
                        return false;
                    }
                    _tpMoePlans[r] = planSlot[0];
                }

                GgmlBasicOps.TensorParallelExecutePlans(_tpMoePlans);
            }
            catch (InvalidOperationException)
            {
                return false;
            }
            finally
            {
                GgmlBasicOps.SetActiveRank(previousRank);
            }

            _tpKvDeviceDirty = true;
            if (!_tpMoeVerifyLogged)
            {
                _tpMoeVerifyLogged = true;
                Console.WriteLine($"  Gemma4 fused tensor-parallel MoE prefill active ({tp} GPUs).");
            }
            return true;
        }

        private bool _tpMoeVerifyLogged;
        private bool _tpMoeDeclineLogged;

        /// <summary>
        /// Report, once, that the fused tensor-parallel MoE trunk declined and
        /// why. Worth a warning rather than a silent fallback: the per-op TP path
        /// it drops to is ~50x slower, and when the experts are tensor-sliced it
        /// has no per-expert weights to run at all.
        /// </summary>
        private void WarnTpMoeFusedDeclined(string phase, int tokens)
        {
            if (_tpMoeDeclineLogged) return;
            _tpMoeDeclineLogged = true;
            Console.Error.WriteLine(
                $"[gemma4-tp] WARNING: the fused tensor-parallel MoE {phase} kernel declined at {tokens} token(s): " +
                GgmlBasicOps.LastNativeError() + " — falling back to the per-op path.");
        }
    }
}
