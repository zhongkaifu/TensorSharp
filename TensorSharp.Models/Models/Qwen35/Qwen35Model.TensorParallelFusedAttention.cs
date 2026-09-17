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
using System.Diagnostics;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    /// <summary>
    /// Fused tensor-parallel attention block for Qwen 3.5 / 3.6 on the GGML
    /// backends.
    ///
    /// The op-at-a-time TP attention block is where this architecture's tensor
    /// parallelism went: on Qwen3.5-9B with <c>--tp 2</c>, 13.0 s of a 13.4 s
    /// forward total. Norm, QKV, head deinterleave, Q/K norm, RoPE, KV append,
    /// attention, the sigmoid gate and the output projection are ~30 native
    /// dispatches per layer per rank, each submitting a graph, synchronizing and
    /// copying its result back to host memory.
    ///
    /// The single-GPU path has had all of that as one graph
    /// (<c>TSGgml_Qwen35AttentionLayerPrefill</c>) for a while; what stopped TP
    /// using it was the residual add folded into the end — a row-parallel output
    /// projection produces a partial sum, and the residual cannot be added until
    /// the ranks have summed their partials. Cutting the graph at that projection
    /// resolves it: segment 0 runs everything up to and including the projection,
    /// the driver AllReduces the partials in VRAM, and segment 1 does the residual
    /// add. Two graph launches per layer per rank instead of ~30 round trips.
    /// </summary>
    public partial class Qwen35Model
    {
        private IntPtr[] _tpAttnPlans;
        private bool _tpFusedAttnChecked;
        private bool _tpFusedAttnReady;
        private bool _tpFusedAttnLogged;

        // Set once the fused block has appended K/V straight into the device
        // copies of the per-rank caches. The op-at-a-time attention reads those
        // tensors out of host memory, so it has to pull them back first.
        private bool _tpAttnKvDeviceOnly;

        /// <summary>
        /// The generic row-parallel tail: <c>hidden[r] += input[r] @ W_r</c>
        /// with the ranks' partial sums reduced in VRAM between the matmul and
        /// the add. Replaces a row-parallel linear, a collective and a residual
        /// add — three host round trips per rank — with two graph launches.
        /// Used for the GatedDeltaNet output projection.
        /// </summary>
        private bool TryQwen35FusedRowParallelResidualTP(Tensor[] hidden, Tensor[] input, QuantizedWeight[] shards)
        {
            if (!TpFusedAttentionAvailable() || shards == null)
                return false;
            // The fused residual matmul has no scale hook; a scaled shard must
            // take the scale-aware per-op path instead.
            if (shards.Length > 0 && shards[0] != null && shards[0].Scale != 1.0f)
                return false;

            int tp = TpDegree;
            int previousRank = GgmlBasicOps.GetActiveRank();
            var planSlot = new IntPtr[1];
            try
            {
                for (int r = 0; r < tp; r++)
                {
                    var w = shards[r];
                    GgmlBasicOps.SetActiveRank(r);
                    planSlot[0] = IntPtr.Zero;
                    GgmlBasicOps.FusedMatMulQuantAdd(
                        hidden[r], input[r],
                        w.CacheKey, w.GgmlType, w.Ne0, w.Ne1, w.RawBytes,
                        tpDegree: tp, tpPlanOut: planSlot);
                    if (planSlot[0] == IntPtr.Zero)
                        return false;
                    _tpAttnPlans[r] = planSlot[0];
                }

                if (TpCrossNodeReducer != null)
                    GgmlBasicOps.TensorParallelExecutePlansDistributed(_tpAttnPlans, TpCrossNodeCallback);
                else
                    GgmlBasicOps.TensorParallelExecutePlans(_tpAttnPlans);
            }
            catch (Exception ex) when (ex is InvalidOperationException || ex is NotSupportedException || ex is ArgumentException)
            {
                return false;
            }
            finally
            {
                GgmlBasicOps.SetActiveRank(previousRank);
            }

            for (int r = 0; r < tp; r++)
                InvalidateTensorDeviceCache(hidden[r]);
            return true;
        }

        /// <summary>
        /// Run one dense FFN block (post-attention norm, column-parallel
        /// gate/up, SiLU·mul, row-parallel down, AllReduce, residual add) as a
        /// single segmented graph per rank. Same shape as the attention block:
        /// ~10 per-op dispatches per rank collapse into two graph launches, with
        /// the reduction happening in VRAM between them.
        /// </summary>
        private unsafe bool TryQwen35FusedDenseFfnTP(Tensor[] hidden, int layer, int seqLen)
        {
            // No scale hook in the fused FFN kernel: decline so the scale-aware
            // per-op TP linears run instead.
            if (TpAnyWeightScaled(_ffnGateUpKey[layer], _ffnDownKey[layer]))
                return TpAttnBail($"layer {layer} FFN carries a per-tensor weight scale");
            if (!TpFusedAttentionAvailable())
                return false;

            Tensor postNorm = _postAttnNormW != null ? _postAttnNormW[layer] : null;
            if (postNorm == null)
                return false;
            if (!_tpQuantWeights.TryGetValue(_ffnGateUpKey[layer], out var gateUpShards) ||
                !_tpQuantWeights.TryGetValue(_ffnDownKey[layer], out var downShards))
                return false;

            int tp = TpDegree;
            int previousRank = GgmlBasicOps.GetActiveRank();
            var planSlot = new IntPtr[1];
            try
            {
                for (int r = 0; r < tp; r++)
                {
                    var gu = gateUpShards[r];
                    var dn = downShards[r];
                    // half_dim is this rank's slice of the intermediate size.
                    int halfDim = (int)(gu.Ne1 / 2);
                    GgmlBasicOps.SetActiveRank(r);
                    planSlot[0] = IntPtr.Zero;
                    GgmlBasicOps.FusedFFNSwiGLUQuant(
                        hidden[r], hidden[r], postNorm, Config.Eps,
                        gu.CacheKey, gu.GgmlType, gu.Ne0, gu.Ne1, gu.RawBytes,
                        dn.CacheKey, dn.GgmlType, dn.Ne0, dn.Ne1, dn.RawBytes,
                        halfDim,
                        tpDegree: tp, tpPlanOut: planSlot);

                    if (planSlot[0] == IntPtr.Zero)
                        return false;
                    _tpAttnPlans[r] = planSlot[0];
                }

                if (TpCrossNodeReducer != null)
                    GgmlBasicOps.TensorParallelExecutePlansDistributed(_tpAttnPlans, TpCrossNodeCallback);
                else
                    GgmlBasicOps.TensorParallelExecutePlans(_tpAttnPlans);
            }
            catch (Exception ex) when (ex is InvalidOperationException || ex is NotSupportedException || ex is ArgumentException)
            {
                return false;
            }
            finally
            {
                GgmlBasicOps.SetActiveRank(previousRank);
            }

            for (int r = 0; r < tp; r++)
                InvalidateTensorDeviceCache(hidden[r]);
            return true;
        }

        /// <summary>
        /// Bring the per-rank attention KV caches back to host memory, for the
        /// per-op attention path after a fused block has run. No-op when the
        /// fused path has not written anything device-only.
        /// </summary>
        private void SyncQwen35TpAttentionKvToHost()
        {
            if (!_tpAttnKvDeviceOnly || _tpKvCacheK == null)
                return;
            _tpAttnKvDeviceOnly = false;

            // Entering the per-op attention path: it reads and rewrites the
            // caches through host memory, after which the device copies the
            // persistent whole-model decode graphs bake would go stale (or be
            // invalidated outright). Drop those graphs; the next fused decode
            // rebuilds against the refreshed state.
            DropTpFusedDecodeGraphs();

            for (int l = 0; l < _tpKvCacheK.Length; l++)
            {
                if (_tpKvCacheK[l] == null) continue;
                for (int r = 0; r < _tpKvCacheK[l].Length; r++)
                {
                    SyncTpCacheTensor(_tpKvCacheK[l][r]);
                    SyncTpCacheTensor(_tpKvCacheV[l][r]);
                }
            }
        }

        private static void SyncTpCacheTensor(Tensor t)
        {
            if (t == null) return;
            IntPtr ptr = TensorComputePrimitives.GetStoragePointer(t);
            if (ptr == IntPtr.Zero) return;
            long bytes = t.ElementCount() * t.ElementType.Size();
            try { GgmlBasicOps.SyncHostBuffer(ptr, bytes); }
            catch (InvalidOperationException) { /* never went device-resident */ }
        }

        /// <summary>Disable with TS_QWEN35_TP_FUSED=0.</summary>
        private static readonly bool _tpFusedBlocksEnabled =
            Environment.GetEnvironmentVariable("TS_QWEN35_TP_FUSED") != "0";

        /// <summary>
        /// Report the first reason the fused per-rank attention block declined, once,
        /// and return false. Without this the fallback to the per-op TP chain — which
        /// the header above measures at 13.0 s of a 13.4 s forward — is completely
        /// silent, so a run where tensor parallelism buys nothing looks identical to
        /// one where it works. (The non-TP whole-model decode already reports its
        /// decline this way; this is the TP twin.)
        /// </summary>
        private bool _tpFusedAttnDeclineLogged;

        private bool TpAttnBail(string reason)
        {
            if (!_tpFusedAttnDeclineLogged)
            {
                _tpFusedAttnDeclineLogged = true;
                Console.Error.WriteLine(
                    $"[qwen35-tp] fused attention block NOT engaged ({reason}); " +
                    "falling back to the per-op tensor-parallel chain for every full-attention layer.");
            }
            return false;
        }

        private bool TpFusedAttentionAvailable()
        {
            if (_tpFusedAttnChecked)
                return _tpFusedAttnReady;
            _tpFusedAttnChecked = true;

            _tpFusedAttnReady =
                _tpFusedBlocksEnabled &&
                IsGgmlBackend &&
                IsTensorParallel &&
                // Across nodes the local ranks still reduce on-device; the
                // cluster half of each boundary goes through the cross-node
                // hook, exactly as the whole-model graph does. What it cannot
                // do is span nodes with no reducer to call.
                (GlobalTpDegree == TpDegree || TpCrossNodeReducer != null) &&
                (TpCrossNodeReducer != null
                    ? GgmlBasicOps.TensorParallelFusedAvailableDistributed(TpDegree)
                    : GgmlBasicOps.TensorParallelFusedAvailable(TpDegree));

            if (_tpFusedAttnReady)
                _tpAttnPlans = new IntPtr[TpDegree];
            else if (IsTensorParallel)
                TpAttnBail(
                    !_tpFusedBlocksEnabled ? "disabled via TS_QWEN35_TP_FUSED=0"
                    : !IsGgmlBackend ? $"backend {_backend} has no fused TP attention kernel"
                    : GlobalTpDegree != TpDegree ? $"multi-node TP (global={GlobalTpDegree}, local={TpDegree}) without a cross-node reducer"
                    : $"the native bridge reports no fused TP support for tp={TpDegree}");
            return _tpFusedAttnReady;
        }

        /// <summary>
        /// Run one full-attention layer's block (norm through residual add) as a
        /// single segmented graph per rank. Returns false when the fused kernel
        /// declines or a weight is missing, in which case the caller keeps its
        /// per-op path. <paramref name="hidden"/> is updated in place — every rank
        /// holds the same replicated activation, so only rank 0 copies it back.
        /// </summary>
        private unsafe bool TryQwen35FusedAttentionBlockTP(Tensor[] hidden, int layer, int seqLen, int startPos)
        {
            // Same for the fused attention block: qkv / o are matmuls with no
            // scale hook in the native graph.
            if (TpAnyWeightScaled(_attnQkvKey[layer], _attnOutputKey[layer],
                    _attnQKey[layer], _attnKKey[layer], _attnVKey[layer]))
                return TpAttnBail($"layer {layer} attention carries a per-tensor weight scale");
            if (!TpFusedAttentionAvailable())
                return false;
            // MRoPE bakes per-axis angles the fused graph cannot express, and the
            // layer graph rotates at its KV index, so a sequence past an image
            // (non-zero M-RoPE delta) keeps the per-op path too.
            if (_pendingMRoPEPositions != null || _ropeDelta != 0)
                return false;

            int tp = TpDegree;
            Tensor attnNorm = _attnNormW[layer];
            Tensor qNorm = _attnQNormW[layer];
            Tensor kNorm = _attnKNormW[layer];
            if (attnNorm == null || qNorm == null || kNorm == null)
                return TpAttnBail($"layer {layer} is missing attn/q/k norm weights");
            if (!_tpQuantWeights.TryGetValue(_attnQkvKey[layer], out var qkvShards) ||
                !_tpQuantWeights.TryGetValue(_attnOutputKey[layer], out var oShards))
            {
                // The usual cause is a mixed-quant GGUF (UD / imatrix builds pick
                // different types for Q, K and V), where the QKV fusion declines and
                // the TP shard is built as a dequantized F32 tensor in _tpWeights
                // instead. Every full-attention layer then silently drops to the
                // per-op TP chain, which is where "tp=2 is no faster than tp=1"
                // comes from.
                return TpAttnBail(
                    $"layer {layer}: no quantized TP shard for " +
                    $"{(_tpQuantWeights.ContainsKey(_attnQkvKey[layer]) ? _attnOutputKey[layer] : _attnQkvKey[layer])}" +
                    (_tpWeights.ContainsKey(_attnQkvKey[layer]) ? " (an F32 shard exists; mixed-quant Q/K/V forced the dequantized path)" : ""));
            }
            if (_tpKvCacheK == null || _tpKvCacheK[layer] == null)
                return TpAttnBail($"layer {layer} has no per-rank TP KV cache");

            Tensor kCache0 = _tpKvCacheK[layer][0];
            if (kCache0.ElementType != DType.Float32 && kCache0.ElementType != DType.Float16)
                return TpAttnBail($"KV cache dtype {kCache0.ElementType} is unsupported by the fused TP attention graph");

            int headDim = Config.HeadDim;
            int numHeadsPerGpu = Config.NumHeads / tp;
            int numKVHeadsPerGpu = Config.NumKVHeads / tp;
            int maxSeqLen = (int)kCache0.Sizes[1];
            int ropeDims = _ropeDimCount > 0 ? _ropeDimCount : headDim;
            const int ropeMode = 2; // NeoX
            float ropeFreqScale = 1.0f / Config.RopeScale;
            int kvCacheTypeId = (kCache0.ElementType == DType.Float16) ? 1 : 0;

            // One activation buffer per rank. They hold identical values (the
            // stream is replicated), but they are SEPARATE buffers that the
            // surrounding per-op blocks accumulate into independently, so each
            // rank's graph must read and write its own — updating only rank 0's
            // leaves the others a layer behind and the model drifts immediately.
            var hiddenPtrs = new IntPtr[tp];
            for (int r = 0; r < tp; r++)
                hiddenPtrs[r] = (IntPtr)GetFloatPtr(hidden[r]);
            IntPtr attnNormPtr = (IntPtr)GetFloatPtr(attnNorm);
            IntPtr qNormPtr = (IntPtr)GetFloatPtr(qNorm);
            IntPtr kNormPtr = (IntPtr)GetFloatPtr(kNorm);

            int previousRank = GgmlBasicOps.GetActiveRank();
            var planSlot = new IntPtr[1];
            long t0 = Stopwatch.GetTimestamp();
            try
            {
                for (int r = 0; r < tp; r++)
                {
                    var qkv = qkvShards[r];
                    var oOut = oShards[r];
                    GgmlBasicOps.SetActiveRank(r);
                    planSlot[0] = IntPtr.Zero;
                    GgmlBasicOps.Qwen35AttentionLayerPrefill(
                        hiddenPtrs[r], Config.HiddenSize, seqLen,
                        attnNormPtr,
                        qkv.CacheKey, qkv.GgmlType, qkv.Ne0, qkv.Ne1, qkv.RawBytes,
                        qNormPtr, kNormPtr,
                        oOut.CacheKey, oOut.GgmlType, oOut.Ne0, oOut.Ne1, oOut.RawBytes,
                        TensorComputePrimitives.GetStoragePointer(_tpKvCacheK[layer][r]),
                        TensorComputePrimitives.GetStoragePointer(_tpKvCacheV[layer][r]),
                        numHeadsPerGpu, numKVHeadsPerGpu, headDim,
                        maxSeqLen, startPos,
                        Config.RopeBase, ropeFreqScale, ropeDims, ropeMode,
                        kvCacheTypeId, Config.Eps,
                        tpDegree: tp, tpPlanOut: planSlot);

                    if (planSlot[0] == IntPtr.Zero)
                        return false;
                    _tpAttnPlans[r] = planSlot[0];
                }

                if (TpCrossNodeReducer != null)
                    GgmlBasicOps.TensorParallelExecutePlansDistributed(_tpAttnPlans, TpCrossNodeCallback);
                else
                    GgmlBasicOps.TensorParallelExecutePlans(_tpAttnPlans);
            }
            catch (InvalidOperationException)
            {
                return false;
            }
            finally
            {
                GgmlBasicOps.SetActiveRank(previousRank);
            }
            _tpAttnBlockTicks += Stopwatch.GetTimestamp() - t0;

            // The block output went back to host memory, so any device copy of it
            // cached under the same pointer is stale and the per-op GDN / FFN
            // blocks that read it next would use the wrong values.
            for (int r = 0; r < tp; r++)
                InvalidateTensorDeviceCache(hidden[r]);

            // The K/V caches are deliberately NOT invalidated: the graph appended
            // to the device-resident copies and every attention layer goes through
            // this same path, so the device copy stays authoritative. Dropping it
            // would re-upload the whole cache (16 MB per layer per rank at an 8K
            // context) on the next layer — and, worse, re-upload the stale host
            // copy. TpKvNeedsHostSync marks it for the fallback path instead.
            _tpAttnKvDeviceOnly = true;

            if (!_tpFusedAttnLogged)
            {
                _tpFusedAttnLogged = true;
                Console.WriteLine($"  Qwen3.5 fused tensor-parallel attention block active ({tp} GPUs).");
            }
            return true;
        }
    }
}
