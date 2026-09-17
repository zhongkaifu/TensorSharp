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
// Qwen35Model.TensorParallelFusedModelDecode.cs
//
// Whole-model fused decode under GGML tensor parallelism: one persistent
// per-rank graph for the entire hybrid transformer (GDN recurrence, attention
// with in-graph KV append, MoE with in-graph top-k routing, folded final norm
// + column-parallel LM head), executed segment-by-segment with the row-parallel
// partials reduced at ~2 cut points per layer. This is the TP sibling of
// TryFullModelDecode and the Qwen3.5 analogue of Gemma4's fused TP MoE decode.
//
// Why it exists: the per-op TP decode dispatches every block through the
// host bridge — dozens of graph submissions and host round trips per layer per
// token. On backends without a device collective (ggml-vulkan) each of those
// costs a queue submit + fence wait, which measured ~800 ms/token on the 35B.
// One persistent graph per rank collapses that to ~2 segment submits + one
// host-staged reduction per layer.
//
// MoE runs expert-parallel (whole experts per rank, matching the sharding that
// TryQwen35MoEExpertParallel preloads): every rank derives the same global
// top-k in-graph from the replicated hidden stream, then confines it to its
// own experts through an I32 id-LUT and an F32 weight mask (see the native
// ep_lut/ep_mask comment). The shared expert stays Megatron-split, so the sum
// of both partials is the layer's single reduction.
// ============================================================================
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    public partial class Qwen35Model
    {
        // Per-rank layer descriptors for the native whole-model decode. Rebuilt
        // when the KV cache is reallocated (its pointers are baked in).
        private Qwen35LayerDecodeArgs[][] _tpFdLayers;
        private IntPtr[] _tpFdPlans;
        private bool _tpFdChecked;
        private bool _tpFdReady;
        private bool _tpFdLogged;
        private bool _tpFdFailed;            // latched: the native side declined once
        private int _tpFdBuiltCapacity = -1;

        /// <summary>Disable with TS_QWEN35_TP_FUSED_DECODE=0.</summary>
        private static readonly bool _tpFdEnabled =
            Environment.GetEnvironmentVariable("TS_QWEN35_TP_FUSED_DECODE") != "0";

        /// <summary>
        /// Report, once, why the whole-model fused tensor-parallel decode declined.
        /// Declining latches this path off for the process and falls back to the
        /// per-op TP loop, which is roughly an order of magnitude slower — that is
        /// the mechanism behind "tp=2 decodes no faster than tp=1", and it used to
        /// leave no trace at all. Mirrors the non-TP FdBail added for the same
        /// failure mode on the single-GPU path.
        /// </summary>
        private readonly HashSet<string> _tpFdDeclineLogged = new(StringComparer.Ordinal);

        /// <summary>The cross-node half of a distributed AllReduce, or null on a
        /// single-node run. Present exactly when the TP group spans nodes.</summary>
        private INestedTensorParallelGroup TpCrossNodeReducer =>
            GlobalTpDegree != TpDegree ? _tpGroup as INestedTensorParallelGroup : null;

        private GgmlBasicOps.CrossNodeAllReduce _tpCrossNodeCallback;
        private float[] _tpCrossNodeBuf = Array.Empty<float>();

        /// <summary>Marshalled once and cached: the native executor calls this at
        /// every AllReduce boundary, so a per-call delegate allocation would land
        /// on the hot path.</summary>
        private GgmlBasicOps.CrossNodeAllReduce TpCrossNodeCallback =>
            _tpCrossNodeCallback ??= (user, data, count) =>
            {
                try
                {
                    if (count <= 0)
                        return true;
                    if (_tpCrossNodeBuf.Length < count)
                        _tpCrossNodeBuf = new float[count];
                    System.Runtime.InteropServices.Marshal.Copy(data, _tpCrossNodeBuf, 0, count);
                    TpCrossNodeReducer.CrossNodeAllReduce(_tpCrossNodeBuf, count);
                    System.Runtime.InteropServices.Marshal.Copy(_tpCrossNodeBuf, 0, data, count);
                    return true;
                }
                catch (Exception ex)
                {
                    Console.Error.WriteLine($"[qwen35-tp] cross-node AllReduce failed: {ex.Message}");
                    return false;
                }
            };

        private bool TpFdBail(string reason)
        {
            // Distinct reasons each print once. A single process-wide latch hid
            // the real cause whenever an outer check declined first.
            if (_tpFdDeclineLogged.Add(reason))
            {
                Console.Error.WriteLine(
                    $"[qwen35-tp] fused whole-model decode NOT engaged ({reason}); " +
                    "falling back to the per-op tensor-parallel decode.");
            }
            return false;
        }

        private bool TpFusedModelDecodeAvailable()
        {
            if (_tpFdFailed)
                return false;
            if (_tpFdChecked)
                return _tpFdReady;
            _tpFdChecked = true;
            _tpFdReady =
                _tpFdEnabled && IsGgmlBackend && IsTensorParallel
                // Multi-node keeps this fused schedule: the executor reduces this
                // node's ranks on-device and then calls back for the cross-node
                // exchange, so the graph is identical and only the reduction is
                // wider. Without such a group there is nobody to do that half.
                && (GlobalTpDegree == TpDegree || TpCrossNodeReducer != null)
                // The graph folds the column-parallel LM head per rank.
                && _tpLmHeadKey != null && _tpQuantWeights.ContainsKey(_tpLmHeadKey)
                && _weights.ContainsKey("output_norm.weight")
                // MoE must be sharded by whole experts (the stacked slices the
                // graph's mul_mat_id dispatches over).
                && (_numExperts <= 0 || UsesExpertParallelMoE)
                && (TpCrossNodeReducer != null
                    ? GgmlBasicOps.TensorParallelFusedAvailableDistributed(TpDegree)
                    : GgmlBasicOps.TensorParallelFusedAvailable(TpDegree));
            if (_tpFdReady)
                _tpFdPlans = new IntPtr[TpDegree];
            else if (IsTensorParallel)
                TpFdBail(
                    !_tpFdEnabled ? "disabled via TS_QWEN35_TP_FUSED_DECODE=0"
                    : !IsGgmlBackend ? $"backend {_backend} has no fused TP decode kernel"
                    : GlobalTpDegree != TpDegree ? $"multi-node TP (global={GlobalTpDegree}, local={TpDegree}) without a cross-node reducer"
                    : _tpLmHeadKey == null ? "the LM head was not sharded column-parallel (tied embeddings, a vocab that does not divide by the TP degree, or a non-GGML backend)"
                    : !_tpQuantWeights.ContainsKey(_tpLmHeadKey) ? $"no quantized TP shard for the LM head ({_tpLmHeadKey})"
                    : !_weights.ContainsKey("output_norm.weight") ? "output_norm.weight is missing"
                    : (_numExperts > 0 && !UsesExpertParallelMoE) ? "MoE is not expert-parallel sharded"
                    : $"the native bridge reports no fused TP support for tp={TpDegree} (distributed={TpCrossNodeReducer != null})");
            return _tpFdReady;
        }

        /// <summary>
        /// Drop the persistent per-rank decode graphs. They bake references to
        /// the per-rank KV / GDN-state device buffers, so anything that frees,
        /// reallocates or re-seeds those must drop the graphs first (KV growth,
        /// KV reset, the per-op path pulling the caches back to host).
        /// </summary>
        private void DropTpFusedDecodeGraphs()
        {
            if (!IsGgmlBackend || !IsTensorParallel)
                return;
            GgmlBasicOps.Qwen35ResetDecodeCache();
            CountDecodeGraphReset();
            _tpFdBuiltCapacity = -1;
        }

        private unsafe bool TryTpFdShard(string key, int rank,
            out (IntPtr ptr, int type, long ne0, long ne1, long bytes) w)
        {
            if (_tpQuantWeights.TryGetValue(key, out var shards) && shards != null && shards[rank] != null)
            {
                var qw = shards[rank];
                // CacheKey, not Data: it identifies the rank-resident device copy.
                w = (qw.CacheKey, qw.GgmlType, qw.Ne0, qw.Ne1, qw.RawBytes);
                return true;
            }
            if (_tpWeights.TryGetValue(key, out var f32) && f32 != null && f32[rank] != null)
            {
                var t = f32[rank];
                w = ((IntPtr)GetFloatPtr(t), 0, t.Sizes[t.DimensionCount - 1], t.Sizes[0], t.ElementCount() * 4L);
                return true;
            }
            w = default;
            return false;
        }

        private unsafe bool TryBuildTpFdLayerDescs()
        {
            int tp = TpDegree;
            int n = Config.NumLayers;
            int structBytes = Marshal.SizeOf<Qwen35LayerDecodeArgs>();
            var perRank = new Qwen35LayerDecodeArgs[tp][];

            for (int r = 0; r < tp; r++)
            {
                var arr = new Qwen35LayerDecodeArgs[n];
                for (int l = 0; l < n; l++)
                {
                    var a = default(Qwen35LayerDecodeArgs);
                    a.StructBytes = structBytes;
                    // Per-tensor sidecar scales. Shard-invariant: column-parallel
                    // splits output rows (the scalar is row-independent) and
                    // row-parallel splits the contraction (the scalar distributes
                    // over the AllReduce), so every rank applies the same value.
                    a.ProjScales = ProjScalesPtr(l);
                    if (!_weights.TryGetValue(_attnNormKey[l], out var attnNorm) || attnNorm == null ||
                        !_weights.TryGetValue(_postAttnNormKey[l], out var postNorm) || postNorm == null)
                        return false;
                    a.AttnNormW = (IntPtr)GetFloatPtr(attnNorm);
                    a.PostAttnNormW = (IntPtr)GetFloatPtr(postNorm);

                    bool isMoe = _isMoeLayer != null && _isMoeLayer[l];
                    a.IsMoe = isMoe ? 1 : 0;
                    if (!isMoe)
                    {
                        if (!TryTpFdShard(_ffnGateUpKey[l], r, out var gu)) return false;
                        if (!TryTpFdShard(_ffnDownKey[l], r, out var dn)) return false;
                        a.GuW = gu.ptr; a.GuType = gu.type; a.GuNe0 = gu.ne0; a.GuNe1 = gu.ne1; a.GuBytes = gu.bytes;
                        a.DownW = dn.ptr; a.DownType = dn.type; a.DownNe0 = dn.ne0; a.DownNe1 = dn.ne1; a.DownBytes = dn.bytes;
                        a.FfDense = (int)(gu.ne1 / 2);
                    }
                    else
                    {
                        // Router: replicated (the per-rank buffer cache gives each
                        // GPU its own device copy from the one host pointer).
                        if (_ffnGateInpQW?[l] == null && _ffnGateInpF32?[l] == null)
                            return false;
                        var gi = ResolveW(_ffnGateInpQW[l], _ffnGateInpF32[l]);
                        a.GateInpW = gi.ptr; a.GateInpType = gi.type; a.GateInpNe0 = gi.ne0; a.GateInpNe1 = gi.ne1; a.GateInpBytes = gi.bytes;

                        // MoE CPU offload: an offloaded layer is evaluated once on
                        // the host, over the WHOLE expert stack, and the driver
                        // gives that single result to rank 0 (the layer's
                        // AllReduce then spreads it). So point the descriptor at
                        // the unsharded tensors — the rank slices exist only for
                        // the layers that stay on the accelerator, and nothing
                        // uploads these. See tsg::HostMoeSegment::tp_reduced.
                        bool layerOnCpu = MoeCpuOffloadConfig.IsLayerOnCpu(l);
                        a.CpuMoe = layerOnCpu ? 1 : 0;
                        var sg = layerOnCpu ? _layerStackedGate?[l] : _tpStackedGate?[l]?[r];
                        var su = layerOnCpu ? _layerStackedUp?[l] : _tpStackedUp?[l]?[r];
                        var sd = layerOnCpu ? _layerStackedDown?[l] : _tpStackedDown?[l]?[r];
                        if (sg == null || su == null || sd == null)
                            return false;
                        a.GateExps = sg.Data; a.GateExpsType = sg.GgmlType; a.GateExpsBytes = sg.TotalRawBytes;
                        a.UpExps = su.Data; a.UpExpsType = su.GgmlType; a.UpExpsBytes = su.TotalRawBytes;
                        a.DownExps = sd.Data; a.DownExpsType = sd.GgmlType; a.DownExpsBytes = sd.TotalRawBytes;

                        string prefix = $"blk.{l}.";
                        // The native decode graph builds the gated shared-expert branch
                        // unconditionally for a MoE layer, so this path genuinely
                        // requires the shexp weights - a variant without them has to
                        // decline rather than leave the descriptor zeroed. Say so out
                        // loud: declining latches the fused TP decode off for the whole
                        // process, and it used to do that in silence.
                        if (_hasSharedExperts == null || !_hasSharedExperts[l])
                            return TpFdBail($"layer {l} is MoE without shared experts, which the fused TP decode graph requires");
                        if (!TryTpFdShard(prefix + "ffn_gate_shexp.weight", r, out var shg))
                            return TpFdBail($"layer {l} rank {r}: no TP shard for {prefix}ffn_gate_shexp.weight");
                        if (!TryTpFdShard(prefix + "ffn_up_shexp.weight", r, out var shu))
                            return TpFdBail($"layer {l} rank {r}: no TP shard for {prefix}ffn_up_shexp.weight");
                        if (!TryTpFdShard(prefix + "ffn_down_shexp.weight", r, out var shd))
                            return TpFdBail($"layer {l} rank {r}: no TP shard for {prefix}ffn_down_shexp.weight");
                        a.ShexpGateW = shg.ptr; a.ShexpGateType = shg.type; a.ShexpGateNe0 = shg.ne0; a.ShexpGateNe1 = shg.ne1; a.ShexpGateBytes = shg.bytes;
                        a.ShexpUpW = shu.ptr; a.ShexpUpType = shu.type; a.ShexpUpNe0 = shu.ne0; a.ShexpUpNe1 = shu.ne1; a.ShexpUpBytes = shu.bytes;
                        a.ShexpDownW = shd.ptr; a.ShexpDownType = shd.type; a.ShexpDownNe0 = shd.ne0; a.ShexpDownNe1 = shd.ne1; a.ShexpDownBytes = shd.bytes;
                        if (_ffnGateInpShexpVec?[l] == null)
                            return TpFdBail($"layer {l}: shared experts present but ffn_gate_inp_shexp is missing");
                        a.ShexpGateInpW = (IntPtr)GetFloatPtr(_ffnGateInpShexpVec[l]);
                    }

                    if (!_isRecurrent[l])
                    {
                        a.IsRecurrent = 0;
                        // The regrouped fused QKV shard ([Q_r|K_r|V_r]); missing
                        // means mixed quant types prevented it — keep per-op.
                        if (!TryTpFdShard(_attnQkvKey[l], r, out var qkv))
                            return TpFdBail($"layer {l} rank {r}: no TP shard for {_attnQkvKey[l]}" +
                                (_tpWeights.ContainsKey(_attnQkvKey[l])
                                    ? " (an F32 shard exists; mixed-quant Q/K/V forced the dequantized path)"
                                    : " (mixed quant types prevented the fused QKV shard)"));
                        if (!TryTpFdShard(_attnOutputKey[l], r, out var o))
                            return TpFdBail($"layer {l} rank {r}: no TP shard for {_attnOutputKey[l]}");
                        a.QkvW = qkv.ptr; a.QkvType = qkv.type; a.QkvNe0 = qkv.ne0; a.QkvNe1 = qkv.ne1; a.QkvBytes = qkv.bytes;
                        a.SeparateQkv = 0;
                        a.OW = o.ptr; a.OType = o.type; a.ONe0 = o.ne0; a.ONe1 = o.ne1; a.OBytes = o.bytes;
                        if (_attnQNormW[l] == null || _attnKNormW[l] == null)
                            return false;
                        a.QNormW = (IntPtr)GetFloatPtr(_attnQNormW[l]);
                        a.KNormW = (IntPtr)GetFloatPtr(_attnKNormW[l]);
                        if (_tpKvCacheK?[l]?[r] == null || _tpKvCacheV?[l]?[r] == null)
                            return false;
                        a.KCache = TensorComputePrimitives.GetStoragePointer(_tpKvCacheK[l][r]);
                        a.VCache = TensorComputePrimitives.GetStoragePointer(_tpKvCacheV[l][r]);
                    }
                    else
                    {
                        a.IsRecurrent = 1;
                        // Packed in-projection shard ([hidden, Q|K|V|Z|beta|alpha]);
                        // GdnGateW stays null, which tells the native graph to slice
                        // the packed matmul instead of running four projections.
                        if (!TryTpFdShard(_ssmInProjKey[l], r, out var pk)) return false;
                        if (!TryTpFdShard(_ssmOutKey[l], r, out var so)) return false;
                        a.GdnQkvW = pk.ptr; a.GdnQkvType = pk.type; a.GdnQkvNe0 = pk.ne0; a.GdnQkvNe1 = pk.ne1; a.GdnQkvBytes = pk.bytes;
                        a.SsmOutW = so.ptr; a.SsmOutType = so.type; a.SsmOutNe0 = so.ne0; a.SsmOutNe1 = so.ne1; a.SsmOutBytes = so.bytes;
                        a.Conv1dW = (IntPtr)GetFloatPtr(GetTpShardTensor(_ssmConv1dKey[l], r));
                        a.SsmDtW = (IntPtr)GetFloatPtr(GetTpShardTensor(_ssmDtBiasKey[l], r));
                        a.SsmAW = (IntPtr)GetFloatPtr(GetTpShardTensor(_ssmAKey[l], r));
                        if (!_weights.TryGetValue(_ssmNormKey[l], out var ssmNorm) || ssmNorm == null)
                            return false;
                        a.SsmNormW = (IntPtr)GetFloatPtr(ssmNorm);
                        if (_tpConvState?[l]?[r] == null || _tpDeltaState?[l]?[r] == null)
                            return false;
                        // The per-rank states are already in the ggml decode layout
                        // ([time, channel] ordered window / [k, v, heads]) and stay
                        // device-resident keyed by these host pointers — the same
                        // binding the per-op GDN kernel uses, so the two paths see
                        // one authoritative state.
                        IntPtr conv = (IntPtr)GetFloatPtr(_tpConvState[l][r]);
                        IntPtr delta = (IntPtr)GetFloatPtr(_tpDeltaState[l][r]);
                        a.ConvStateIn = conv; a.ConvStateOut = conv;
                        a.DeltaStateIn = delta; a.DeltaStateOut = delta;
                    }
                    arr[l] = a;
                }
                perRank[r] = arr;
            }

            _tpFdLayers = perRank;
            _tpFdBuiltCapacity = _tpKvCacheCapacity;
            return true;
        }

        /// <summary>
        /// One fused tensor-parallel decode step: the whole model as one
        /// segmented persistent graph per rank, logits written directly into
        /// <paramref name="logitsOut"/> (each rank downloads its own vocabulary
        /// slice — the LM head is column-parallel, so no collective is needed
        /// there). Returns false when unavailable, leaving the caller on the
        /// per-op TP forward.
        /// </summary>
        private unsafe bool TryQwen35FusedModelDecodeTP(Tensor hidden, int position, float[] logitsOut)
        {
            if (!TpFusedModelDecodeAvailable() || logitsOut == null || logitsOut.Length < Config.VocabSize)
                return TpFdBail("availability/logits buffer");
            if (hidden == null || hidden.DimensionCount != 2 || hidden.Sizes[0] != 1
                || hidden.ElementType != DType.Float32)
                return TpFdBail("hidden shape");

            int tp = TpDegree;
            int n = Config.NumLayers;

            int cacheSize = 0;
            int kvCacheType = 0;
            for (int l = 0; l < n; l++)
            {
                if (_isRecurrent[l] || _tpKvCacheK?[l] == null)
                    continue;
                var k0 = _tpKvCacheK[l][0];
                if (k0.ElementType != DType.Float32 && k0.ElementType != DType.Float16)
                    return TpFdBail("kv cache geometry");
                cacheSize = (int)k0.Sizes[1];
                kvCacheType = k0.ElementType == DType.Float16 ? 1 : 0;
                break;
            }
            if (cacheSize <= 0 || position >= cacheSize)
                return TpFdBail("no attention KV cache");

            if (_tpFdLayers == null || _tpFdBuiltCapacity != _tpKvCacheCapacity)
            {
                // KV pointers changed (or first use): the old graphs bake stale
                // buffer references, so drop them before rebuilding the descs.
                DropTpFusedDecodeGraphs();
                bool built;
                try { built = TryBuildTpFdLayerDescs(); }
                catch (KeyNotFoundException e) { built = TpFdBail($"missing weight {e.Message}"); }
                if (!built)
                {
                    // Latched for the process: report it if TryBuildTpFdLayerDescs
                    // returned false through a path that had nothing to say.
                    TpFdBail("a per-layer descriptor could not be built");
                    _tpFdFailed = true;
                    return TpFdBail("layer descriptor build");
                }
            }

            var lmShards = _tpQuantWeights[_tpLmHeadKey];
            IntPtr finalNormPtr = (IntPtr)GetFloatPtr(_weights["output_norm.weight"]);
            int gTp = GlobalTpDegree;
            int headsPerRank = Config.NumHeads / gTp;
            int kvHeadsPerRank = Config.NumKVHeads / gTp;
            int kHeadsPerRank = _numKHeads / gTp;
            int vHeadsPerRank = _numVHeads / gTp;
            int sharedFfPerRank = _sharedExpertFfnLength > 0 ? _sharedExpertFfnLength / gTp : 0;
            int headDim = Config.HeadDim;

            int previousRank = GgmlBasicOps.GetActiveRank();
            var planSlot = new IntPtr[1];
            long t0 = Stopwatch.GetTimestamp();
            try
            {
                fixed (float* lp = logitsOut)
                {
                    long offset = 0;
                    for (int r = 0; r < tp; r++)
                    {
                        var lm = lmShards[r];
                        GgmlBasicOps.SetActiveRank(r);
                        planSlot[0] = IntPtr.Zero;
                        bool ok = GgmlBasicOps.Qwen35ModelDecode(
                            _tpFdLayers[r], n,
                            false,
                            (IntPtr)GetFloatPtr(hidden), Config.HiddenSize, position,
                            headsPerRank, kvHeadsPerRank, headDim, cacheSize,
                            _ropeDimCount > 0 ? _ropeDimCount : headDim, 2, kvCacheType,
                            _convKernel, _headKDim, _headVDim, kHeadsPerRank, vHeadsPerRank,
                            Config.Eps, Config.RopeBase, 1.0f / Config.RopeScale,
                            _numExperts, _numExpertsUsed, _expertFfnLength, sharedFfPerRank,
                            _normTopKProb ? 1 : 0, 1.0f,
                            (IntPtr)(lp + offset), (int)lm.Ne1,
                            lm.CacheKey, lm.GgmlType, lm.Ne0, lm.Ne1, lm.RawBytes,
                            finalNormPtr,
                            tpDegree: tp, tpPlanOut: planSlot);
                        if (!ok || planSlot[0] == IntPtr.Zero)
                        {
                            _tpFdFailed = true;
                            if (!_tpFdLogged)
                            {
                                _tpFdLogged = true;
                                Console.Error.WriteLine(
                                    "[tp-full-decode] native declined; staying on the per-op TP decode.");
                            }
                            return false;
                        }
                        _tpFdPlans[r] = planSlot[0];
                        offset += lm.Ne1;
                    }

                    if (TpCrossNodeReducer != null)
                        GgmlBasicOps.TensorParallelExecutePlansDistributed(_tpFdPlans, TpCrossNodeCallback);
                    else
                        GgmlBasicOps.TensorParallelExecutePlans(_tpFdPlans);
                }
            }
            catch (InvalidOperationException e)
            {
                _tpFdFailed = true;
                if (!_tpFdLogged)
                {
                    _tpFdLogged = true;
                    Console.Error.WriteLine(
                        $"[tp-full-decode] native execution failed ({e.Message}); " +
                        "staying on the per-op TP decode (roughly 10x slower). Reported once.");
                }
                return false;
            }
            finally
            {
                GgmlBasicOps.SetActiveRank(previousRank);
            }
            _tpAttnBlockTicks += Stopwatch.GetTimestamp() - t0;

            // KV rows and the GDN state advanced in the device-resident copies
            // only; the per-op fallback pulls them back through the existing
            // sync (which also drops these graphs first).
            _tpAttnKvDeviceOnly = true;

            if (!_tpFdLogged)
            {
                _tpFdLogged = true;
                Console.WriteLine($"  Qwen3.5 fused tensor-parallel decode active ({tp} GPUs, " +
                    $"{2 * n} AllReduce points/token).");
            }
            return true;
        }

        // ====================================================================
        // Whole-model fused PREFILL under tensor parallelism
        // ====================================================================
        //
        // The N-token sibling of TryQwen35FusedModelDecodeTP, backed by the
        // verify kernel's tp_mode. The per-layer fused-block prefill this
        // replaces round-trips the [N, hidden] activation through host memory
        // several times per layer (measured 612 vs 1459 tok/s single-GPU on the
        // 9B at N=512); one segmented whole-model graph per rank keeps the
        // activations in VRAM and reduces the row-parallel partials at ~2 cut
        // points per layer, exactly like the decode.

        private bool _tpPfFailed;            // latched: the native side declined once
        private bool _tpPfLogged;

        /// <summary>Disable with TS_QWEN35_TP_FUSED_PREFILL=0.</summary>
        private static readonly bool _tpPfEnabled =
            Environment.GetEnvironmentVariable("TS_QWEN35_TP_FUSED_PREFILL") != "0";

        /// <summary>
        /// Pull the per-rank GDN states back to host memory. The fused decode
        /// (and the per-op GDN kernel) advance the device-resident copies keyed
        /// by these host pointers, so before a host-mode prefill reads the state
        /// the host copy must be refreshed. No-op for state that never went
        /// device-resident (first turn).
        /// </summary>
        private unsafe void SyncQwen35TpGdnStateToHost()
        {
            if (_tpDeltaState == null)
                return;
            int previousRank = GgmlBasicOps.GetActiveRank();
            try
            {
                for (int l = 0; l < Config.NumLayers; l++)
                {
                    if (!_isRecurrent[l] || _tpDeltaState[l] == null)
                        continue;
                    for (int r = 0; r < TpDegree; r++)
                    {
                        GgmlBasicOps.SetActiveRank(r);
                        SyncTpHostTensor(_tpDeltaState[l][r]);
                        SyncTpHostTensor(_tpConvState[l]?[r]);
                    }
                }
            }
            finally
            {
                GgmlBasicOps.SetActiveRank(previousRank);
            }
        }

        private unsafe void SyncTpHostTensor(Tensor t)
        {
            if (t == null) return;
            IntPtr ptr = (IntPtr)GetFloatPtr(t);
            if (ptr == IntPtr.Zero) return;
            long bytes = t.ElementCount() * sizeof(float);
            try { GgmlBasicOps.SyncHostBuffer(ptr, bytes); }
            catch (InvalidOperationException) { /* never went device-resident */ }
        }

        /// <summary>
        /// The prefill advanced the GDN states in HOST memory; any device-resident
        /// copy keyed by those pointers is now stale. Drop the decode graphs that
        /// pin them first, then invalidate so the next decode (fused or per-op)
        /// re-uploads the post-prefill state.
        /// </summary>
        private void InvalidateQwen35TpGdnDeviceState()
        {
            DropTpFusedDecodeGraphs();
            if (_tpDeltaState == null)
                return;
            int previousRank = GgmlBasicOps.GetActiveRank();
            try
            {
                for (int l = 0; l < Config.NumLayers; l++)
                {
                    if (!_isRecurrent[l] || _tpDeltaState[l] == null)
                        continue;
                    for (int r = 0; r < TpDegree; r++)
                    {
                        GgmlBasicOps.SetActiveRank(r);
                        InvalidateTensorDeviceCache(_tpDeltaState[l][r]);
                        if (_tpConvState[l]?[r] != null)
                            InvalidateTensorDeviceCache(_tpConvState[l][r]);
                    }
                }
            }
            finally
            {
                GgmlBasicOps.SetActiveRank(previousRank);
            }
        }

        /// <summary>
        /// One fused tensor-parallel prefill: the whole model over
        /// <paramref name="seqLen"/> tokens as one segmented graph per rank,
        /// the last token's logits written into <paramref name="logitsOut"/>
        /// (column-parallel LM head — each rank downloads its own vocabulary
        /// slice). Returns false when unavailable, leaving the caller on the
        /// per-layer TP prefill.
        /// </summary>
        private unsafe bool TryQwen35FusedModelPrefillTP(Tensor hidden, int seqLen, int startPos, float[] logitsOut)
        {
            if (!_tpPfEnabled || _tpPfFailed || seqLen < 2)
                return false;
            if (!TpFusedModelDecodeAvailable() || logitsOut == null || logitsOut.Length < Config.VocabSize)
                return false;
            if (hidden == null || hidden.DimensionCount != 2 || hidden.Sizes[0] != seqLen
                || hidden.ElementType != DType.Float32)
                return false;
            // gated_delta_net over N tokens needs the square per-head state.
            if (_headKDim != _headVDim)
                return false;

            int tp = TpDegree;
            int n = Config.NumLayers;

            int cacheSize = 0;
            int kvCacheType = 0;
            for (int l = 0; l < n; l++)
            {
                if (_isRecurrent[l] || _tpKvCacheK?[l] == null)
                    continue;
                var k0 = _tpKvCacheK[l][0];
                if (k0.ElementType != DType.Float32 && k0.ElementType != DType.Float16)
                    return false;
                cacheSize = (int)k0.Sizes[1];
                kvCacheType = k0.ElementType == DType.Float16 ? 1 : 0;
                break;
            }
            if (cacheSize <= 0 || startPos + seqLen > cacheSize)
                return false;

            if (_tpFdLayers == null || _tpFdBuiltCapacity != _tpKvCacheCapacity)
            {
                DropTpFusedDecodeGraphs();
                bool built;
                string buildErr = null;
                try { built = TryBuildTpFdLayerDescs(); }
                catch (KeyNotFoundException e) { built = false; buildErr = $"missing weight {e.Message}"; }
                if (!built)
                {
                    _tpPfFailed = true;
                    if (!_tpPfLogged)
                    {
                        _tpPfLogged = true;
                        Console.Error.WriteLine(
                            $"[tp-full-prefill] {buildErr ?? "a per-layer descriptor could not be built"}; " +
                            "staying on the per-layer TP prefill (significantly slower). Reported once.");
                    }
                    return false;
                }
            }

            // The graph reads the GDN state from the HOST pointers; refresh them
            // from the device-resident copies a preceding decode advanced.
            SyncQwen35TpGdnStateToHost();

            // Multimodal MRoPE positions, [3 * seqLen] T/H/W per token staged by
            // SetMRoPEPositions, re-laid out axis-concatenated [4 * seqLen] for
            // ggml_rope_multi (mirrors TryFullModelVerify).
            int[] mropePos = null;
            int[] mropeSecs = null;
            if (_pendingMRoPEPositions != null)
            {
                if (_mropeSections == null || _mropeSections.Length < 4
                    || _pendingMRoPEPositions.Length < 3 * seqLen)
                    return false;
                mropePos = new int[4 * seqLen];
                for (int i = 0; i < seqLen; i++)
                {
                    mropePos[i]              = _pendingMRoPEPositions[3 * i + 0]; // T
                    mropePos[seqLen + i]     = _pendingMRoPEPositions[3 * i + 1]; // H
                    mropePos[2 * seqLen + i] = _pendingMRoPEPositions[3 * i + 2]; // W
                }
                mropeSecs = new int[4] { _mropeSections[0], _mropeSections[1], _mropeSections[2], _mropeSections[3] };
            }

            var lmShards = _tpQuantWeights[_tpLmHeadKey];
            IntPtr finalNormPtr = (IntPtr)GetFloatPtr(_weights["output_norm.weight"]);
            int gTp = GlobalTpDegree;
            int headsPerRank = Config.NumHeads / gTp;
            int kvHeadsPerRank = Config.NumKVHeads / gTp;
            int kHeadsPerRank = _numKHeads / gTp;
            int vHeadsPerRank = _numVHeads / gTp;
            int sharedFfPerRank = _sharedExpertFfnLength > 0 ? _sharedExpertFfnLength / gTp : 0;
            int headDim = Config.HeadDim;

            long t0 = Stopwatch.GetTimestamp();
            // Native TP plans borrow process-global per-rank execution resources
            // between the build calls and TensorParallelExecutePlans. Owner-keyed
            // pending slots prevent accidental replacement, while this lock closes
            // the remaining build-all-ranks -> execute lifetime window across two
            // live Qwen35 model instances.
            lock (_verifyTpPlanLock)
            {
                int previousRank = GgmlBasicOps.GetActiveRank();
                var planSlot = new IntPtr[1];
                try
                {
                    fixed (float* lp = logitsOut)
                    {
                        long offset = 0;
                        for (int r = 0; r < tp; r++)
                        {
                            var lm = lmShards[r];
                            GgmlBasicOps.SetActiveRank(r);
                            planSlot[0] = IntPtr.Zero;
                            bool ok = GgmlBasicOps.Qwen35ModelVerify(
                                _tpFdLayers[r], n,
                                (IntPtr)GetFloatPtr(hidden), Config.HiddenSize, startPos, seqLen,
                                headsPerRank, kvHeadsPerRank, headDim, cacheSize,
                                _ropeDimCount > 0 ? _ropeDimCount : headDim, 2, kvCacheType,
                                _convKernel, _headKDim, _headVDim, kHeadsPerRank, vHeadsPerRank,
                                Config.Eps, Config.RopeBase, 1.0f / Config.RopeScale,
                                _numExperts, _numExpertsUsed, _expertFfnLength, sharedFfPerRank,
                                _normTopKProb ? 1 : 0, 1.0f,
                                (IntPtr)(lp + offset), (int)lm.Ne1,
                                lm.CacheKey, lm.GgmlType, lm.Ne0, lm.Ne1, lm.RawBytes,
                                finalNormPtr, IntPtr.Zero, nLogitRows: 1,
                                mropePos, mropeSecs,
                                tpDegree: tp, tpPlanOut: planSlot,
                                ownerId: _verifyOwnerId);
                            if (!ok || planSlot[0] == IntPtr.Zero)
                            {
                                _tpPfFailed = true;
                                if (!_tpPfLogged)
                                {
                                    _tpPfLogged = true;
                                    Console.Error.WriteLine(
                                        "[tp-full-prefill] native declined; staying on the per-layer TP prefill.");
                                }
                                return false;
                            }
                            _tpFdPlans[r] = planSlot[0];
                            offset += lm.Ne1;
                        }

                        if (TpCrossNodeReducer != null)
                            GgmlBasicOps.TensorParallelExecutePlansDistributed(_tpFdPlans, TpCrossNodeCallback);
                        else
                            GgmlBasicOps.TensorParallelExecutePlans(_tpFdPlans);
                    }
                }
                catch (InvalidOperationException e)
                {
                    _tpPfFailed = true;
                    if (!_tpPfLogged)
                    {
                        _tpPfLogged = true;
                        Console.Error.WriteLine(
                            $"[tp-full-prefill] native execution failed ({e.Message}); " +
                            "staying on the per-layer TP prefill (significantly slower). Reported once.");
                    }
                    return false;
                }
                finally
                {
                    GgmlBasicOps.SetActiveRank(previousRank);
                }
            }
            _tpAttnBlockTicks += Stopwatch.GetTimestamp() - t0;

            // KV rows advanced in the device-resident copies only; the GDN state
            // advanced on the HOST (downloaded by the plan) — flip both caches'
            // coherence accordingly.
            _tpAttnKvDeviceOnly = true;
            InvalidateQwen35TpGdnDeviceState();

            if (!_tpPfLogged)
            {
                _tpPfLogged = true;
                Console.WriteLine($"  Qwen3.5 fused tensor-parallel prefill active ({tp} GPUs, " +
                    $"{2 * n} AllReduce points/pass).");
            }
            return true;
        }
    }
}
