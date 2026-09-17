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
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    /// <summary>
    /// Fused tensor-parallel decode for Gemma 4 on the GGML backends.
    ///
    /// The generic TP forward runs one native op at a time, and every GGML op
    /// submits, synchronizes and copies its result back to host memory. A 42-layer
    /// Gemma 4 decode is ~2000 such round trips per token, doubled across two
    /// ranks, which is why <c>--tp 2</c> measured 5.5 tok/s against 36.8 on a
    /// single GPU.
    ///
    /// This path instead builds the SAME whole-model graph the single-GPU decode
    /// uses, once per rank over that rank's weight shards, and cuts it at the two
    /// places per layer where a row-parallel projection leaves a partial sum: the
    /// attention output projection and the FFN down projection. The native driver
    /// then submits segment k on every GPU asynchronously, AllReduces the partials
    /// in VRAM, and moves on — the schedule llama.cpp's meta backend uses for
    /// <c>--split-mode tensor</c>. Activations never touch the PCIe bus and the
    /// host issues 2*L graph launches per token instead of ~2000 op dispatches.
    ///
    /// Everything outside the two projections is replicated work: each rank runs
    /// the norms, RoPE, PLE injection and layer scalars on the same values and
    /// gets the same answer. Only rank 0 folds the final norm and the LM head,
    /// because the LM head is the largest tensor left after sharding and
    /// duplicating it would give back a big slice of what TP just saved.
    /// </summary>
    public partial class Gemma4Model
    {
        // One decode-array set per rank, holding that rank's weight shards and KV
        // cache pointers. Null when the fused TP decode is unavailable.
        private Gemma4DecodeArrays[] _tpDecodeArrays;
        private IntPtr[] _tpDecodePlans;
        private bool _tpFusedDecodeReady;
        private bool _tpFusedDecodeLogged;
        private bool _tpFusedDecodeDeclineLogged;
        private bool _tpFusedPrefillDeclineLogged;

        // Set once a fused TP decode has written KV rows straight into the
        // device-resident cache copies. The op-at-a-time TP prefill reads the KV
        // tensors out of HOST memory, so the device copies have to be pulled back
        // before it runs or the second turn of a conversation attends over a
        // stale cache.
        private bool _tpKvDeviceDirty;

        /// <summary>Disable with TS_GEMMA4_TP_FUSED_DECODE=0.</summary>
        private static readonly bool _tpFusedDecodeEnabled =
            Environment.GetEnvironmentVariable("TS_GEMMA4_TP_FUSED_DECODE") != "0";

        /// <summary>
        /// Build the per-rank decode arrays. Called after weight sharding and the
        /// per-rank device preload, so every shard already has a device cache key.
        /// </summary>
        private unsafe void BuildGemma4TpDecodeArrays()
        {
            _tpFusedDecodeReady = false;
            _tpDecodeArrays = null;

            if (!_tpFusedDecodeEnabled || !IsGgmlBackend || !IsTensorParallel)
                return;
            // MoE models share these per-rank arrays but run through their own
            // whole-model trunk kernel; only a dense model can use the dense
            // decode/verify graphs, so _tpFusedDecodeReady stays clear for MoE.
            bool anyMoE = false;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (HasMoE(l)) { anyMoE = true; break; }
            }
            // A MoE model only needs the arrays when its sliced-expert layout is
            // in place; otherwise nothing will consume them.
            if (anyMoE && !UsesTensorSlicedMoE)
                return;
            // Needs a device collective; without one every segment boundary would
            // be a host round trip and this would be slower than what it replaces.
            if (!GgmlBasicOps.TensorParallelFusedAvailable(TpDegree))
                return;
            // Multi-node splits a rank across processes; the driver here submits
            // every rank from one thread, so it is a single-process facility.
            if (GlobalTpDegree != TpDegree)
                return;

            int n = Config.NumLayers;
            int tp = TpDegree;
            var perRank = new Gemma4DecodeArrays[tp];

            for (int r = 0; r < tp; r++)
            {
                var a = NewDecodeArrays(n);
                for (int l = 0; l < n; l++)
                {
                    string prefix = $"blk.{l}";
                    bool isShared = _kvDonorMap.ContainsKey(l);
                    int kvSource = _kvDonorMap.TryGetValue(l, out int donor) ? donor : l;

                    a.HeadDim[l] = HeadDimForLayer(l);
                    // Per-rank head counts: the column-parallel QKV shard produces
                    // this rank's heads only, and its KV cache holds just those.
                    a.KvHeads[l] = KVHeadsForLayer(l) / tp;
                    a.CacheSize[l] = (int)_tpKvCacheK[kvSource][r].Sizes[1];
                    a.IsLocal[l] = IsLocalLayer(kvSource) ? 1 : 0;
                    a.KvSource[l] = kvSource;
                    a.RopeBase[l] = IsLocalLayer(l) ? _ropeLocalBase : _ropeGlobalBase;
                    a.RopeNDims[l] = IsLocalLayer(l) ? _localHeadDim : _partialRotaryDims;
                    a.LayerScalar[l] = _layerScalars[l];

                    // Replicated F32 norms: one host pointer, but the native
                    // buffer cache is per rank, so each GPU gets its own copy.
                    a.AttnNorm[l] = (IntPtr)GetFloatPtr(_weights[$"{prefix}.attn_norm.weight"]);
                    a.QNorm[l] = (IntPtr)GetFloatPtr(_weights[$"{prefix}.attn_q_norm.weight"]);
                    if (!isShared)
                        a.KNorm[l] = (IntPtr)GetFloatPtr(_weights[$"{prefix}.attn_k_norm.weight"]);

                    a.KCache[l] = TensorComputePrimitives.GetStoragePointer(_tpKvCacheK[kvSource][r]);
                    a.VCache[l] = TensorComputePrimitives.GetStoragePointer(_tpKvCacheV[kvSource][r]);

                    string postAttnKey = _weights.ContainsKey($"{prefix}.post_attention_norm.weight")
                        ? $"{prefix}.post_attention_norm.weight" : $"{prefix}.attn_post_norm.weight";
                    a.PostAttnNorm[l] = (IntPtr)GetFloatPtr(_weights[postAttnKey]);
                    a.FfnNorm[l] = (IntPtr)GetFloatPtr(_weights[$"{prefix}.ffn_norm.weight"]);
                    string postFfnKey = _weights.ContainsKey($"{prefix}.post_ffw_norm.weight")
                        ? $"{prefix}.post_ffw_norm.weight" : $"{prefix}.ffn_post_norm.weight";
                    a.PostFfnNorm[l] = (IntPtr)GetFloatPtr(_weights[postFfnKey]);

                    // Sharded projections. Shared (KV-follower) layers project Q
                    // only; the rest carry the fused [Q|K|V], already split
                    // segment-wise so each rank holds [Q_r|K_r|V_r].
                    string qkvName = isShared ? $"{prefix}.attn_q.weight" : $"{prefix}.attn_qkv.weight";
                    if (!TryFillShard(a.Qkv, a.QkvType, a.QkvNe0, a.QkvNe1, a.QkvBytes, l, qkvName, r))
                    {
                        if (isShared || !TryFillSeparateQkvShards(a, l, prefix, r))
                            return;
                    }

                    if (!TryFillShard(a.O, a.OType, a.ONe0, a.ONe1, a.OBytes, l, $"{prefix}.attn_output.weight", r))
                        return;
                    if (!TryFillShard(a.Gu, a.GuType, a.GuNe0, a.GuNe1, a.GuBytes, l, $"{prefix}.ffn_gate_up.weight", r))
                        return;
                    if (!TryFillShard(a.Down, a.DownType, a.DownNe0, a.DownNe1, a.DownBytes, l, $"{prefix}.ffn_down.weight", r))
                        return;

                    // PLE injection is replicated: the same weights on every rank
                    // (they are small — a 2560->256 gate and its inverse).
                    FillPleWeights(a, l, prefix);
                }
                perRank[r] = a;
            }

            _tpDecodeArrays = perRank;
            _tpDecodePlans = new IntPtr[tp];
            // The dense whole-model graphs only exist for a dense trunk.
            _tpFusedDecodeReady = !anyMoE;
            if (anyMoE)
                BuildGemma4TpMoeArgs();
        }

        private static Gemma4DecodeArrays NewDecodeArrays(int n)
        {
            var a = new Gemma4DecodeArrays();
            a.AttnNorm = new IntPtr[n]; a.Qkv = new IntPtr[n]; a.QNorm = new IntPtr[n]; a.KNorm = new IntPtr[n];
            a.O = new IntPtr[n]; a.PostAttnNorm = new IntPtr[n];
            a.FfnNorm = new IntPtr[n]; a.Gu = new IntPtr[n]; a.Down = new IntPtr[n]; a.PostFfnNorm = new IntPtr[n];
            a.KCache = new IntPtr[n]; a.VCache = new IntPtr[n];
            a.HeadDim = new int[n]; a.KvHeads = new int[n]; a.CacheSize = new int[n]; a.IsLocal = new int[n];
            a.KvSource = new int[n]; a.RopeNDims = new int[n];
            a.RopeBase = new float[n]; a.LayerScalar = new float[n];
            a.QkvType = new int[n]; a.QkvNe0 = new long[n]; a.QkvNe1 = new long[n]; a.QkvBytes = new long[n];
            a.K = new IntPtr[n]; a.KType = new int[n]; a.KNe0 = new long[n]; a.KNe1 = new long[n]; a.KBytes = new long[n];
            a.V = new IntPtr[n]; a.VType = new int[n]; a.VNe0 = new long[n]; a.VNe1 = new long[n]; a.VBytes = new long[n];
            a.OType = new int[n]; a.ONe0 = new long[n]; a.ONe1 = new long[n]; a.OBytes = new long[n];
            a.GuType = new int[n]; a.GuNe0 = new long[n]; a.GuNe1 = new long[n]; a.GuBytes = new long[n];
            a.DownType = new int[n]; a.DownNe0 = new long[n]; a.DownNe1 = new long[n]; a.DownBytes = new long[n];
            a.PleGate = new IntPtr[n]; a.PleProj = new IntPtr[n]; a.PlePostNorm = new IntPtr[n];
            a.PleGateType = new int[n]; a.PleProjType = new int[n];
            a.PleGateNe0 = new long[n]; a.PleGateNe1 = new long[n]; a.PleGateBytes = new long[n];
            a.PleProjNe0 = new long[n]; a.PleProjNe1 = new long[n]; a.PleProjBytes = new long[n];
            return a;
        }

        private bool TryFillShard(IntPtr[] data, int[] type, long[] ne0, long[] ne1, long[] bytes,
            int layer, string weightName, int rank)
        {
            if (!_tpQuantWeights.TryGetValue(weightName, out var shards))
                return false;
            var qw = shards[rank];
            // CacheKey, not Data: after the per-rank preload the device copy is
            // addressed by an opaque handle and Data would miss it entirely.
            data[layer] = qw.CacheKey;
            type[layer] = qw.GgmlType;
            ne0[layer] = qw.Ne0;
            ne1[layer] = qw.Ne1;
            bytes[layer] = qw.RawBytes;
            return true;
        }

        /// <summary>
        /// Mixed-quantization layers keep Q/K/V apart because they carry different
        /// ggml types. The Qkv slot then holds Q and a non-null K pointer is the
        /// native kernel's signal to run three matmuls.
        /// </summary>
        private bool TryFillSeparateQkvShards(Gemma4DecodeArrays a, int layer, string prefix, int rank)
        {
            return TryFillShard(a.Qkv, a.QkvType, a.QkvNe0, a.QkvNe1, a.QkvBytes, layer, $"{prefix}.attn_q.weight", rank)
                && TryFillShard(a.K, a.KType, a.KNe0, a.KNe1, a.KBytes, layer, $"{prefix}.attn_k.weight", rank)
                && TryFillShard(a.V, a.VType, a.VNe0, a.VNe1, a.VBytes, layer, $"{prefix}.attn_v.weight", rank);
        }

        private unsafe void FillPleWeights(Gemma4DecodeArrays a, int layer, string prefix)
        {
            string pleGateName = $"{prefix}.inp_gate.weight";
            bool hasPleGate = false;
            if (_quantWeights.TryGetValue(pleGateName, out var pleGW))
            {
                a.PleGate[layer] = pleGW.CacheKey;
                a.PleGateType[layer] = pleGW.GgmlType;
                a.PleGateNe0[layer] = pleGW.Ne0;
                a.PleGateNe1[layer] = pleGW.Ne1;
                a.PleGateBytes[layer] = pleGW.RawBytes;
                hasPleGate = true;
            }
            else if (_weights.TryGetValue(pleGateName, out var pleGateF32))
            {
                a.PleGate[layer] = (IntPtr)GetFloatPtr(pleGateF32);
                a.PleGateType[layer] = 0;
                a.PleGateNe0[layer] = pleGateF32.Sizes[1];
                a.PleGateNe1[layer] = pleGateF32.Sizes[0];
                a.PleGateBytes[layer] = pleGateF32.ElementCount() * 4;
                hasPleGate = true;
            }
            if (!hasPleGate)
                return;

            string pleProjName = $"{prefix}.proj.weight";
            if (_quantWeights.TryGetValue(pleProjName, out var plePW))
            {
                a.PleProj[layer] = plePW.CacheKey;
                a.PleProjType[layer] = plePW.GgmlType;
                a.PleProjNe0[layer] = plePW.Ne0;
                a.PleProjNe1[layer] = plePW.Ne1;
                a.PleProjBytes[layer] = plePW.RawBytes;
            }
            else if (_weights.TryGetValue(pleProjName, out var pleProjF32))
            {
                a.PleProj[layer] = (IntPtr)GetFloatPtr(pleProjF32);
                a.PleProjType[layer] = 0;
                a.PleProjNe0[layer] = pleProjF32.Sizes[1];
                a.PleProjNe1[layer] = pleProjF32.Sizes[0];
                a.PleProjBytes[layer] = pleProjF32.ElementCount() * 4;
            }

            string plePostNormName = $"{prefix}.post_norm.weight";
            if (_weights.ContainsKey(plePostNormName))
                a.PlePostNorm[layer] = (IntPtr)GetFloatPtr(_weights[plePostNormName]);
        }

        /// <summary>
        /// One fused tensor-parallel decode step. Returns false when the native
        /// build declines (the persistent graph is unavailable for this shape, an
        /// unsupported KV window) so the caller falls back to the per-op TP
        /// forward. On success <paramref name="logitsOut"/> holds the token's
        /// logits and the KV caches have advanced on-device.
        /// </summary>
        private unsafe bool TryGemma4FusedModelDecodeTP(Tensor hidden, int startPos, Tensor perLayerInputs, float[] logitsOut)
        {
            if (!_tpFusedDecodeReady || logitsOut == null)
                return false;

            // The fold needs a quantized LM head + the F32 final norm. Without
            // them the graph would stop at the hidden state and the caller would
            // have to run the tail itself, which is not wired up here.
            if (!_fdFoldLmHead
                || !_weights.TryGetValue("output_norm.weight", out var finalNormT)
                || !_quantWeights.TryGetValue(_hasTiedOutput ? "token_embd.weight" : "output.weight", out var lmqw))
                return false;

            int tp = TpDegree;
            float* hiddenPtr = GetFloatPtr(hidden);
            IntPtr pleDataPtr = perLayerInputs != null ? (IntPtr)GetFloatPtr(perLayerInputs) : IntPtr.Zero;
            IntPtr finalNormPtr = (IntPtr)GetFloatPtr(finalNormT);

            IntPtr freqFactorsPtr = IntPtr.Zero;
            int freqFactorsLen = 0;
            if (_weights.TryGetValue("rope_freqs.weight", out var freqTensor))
            {
                freqFactorsPtr = (IntPtr)GetFloatPtr(freqTensor);
                freqFactorsLen = (int)freqTensor.ElementCount();
            }

            int previousRank = GgmlBasicOps.GetActiveRank();
            var planSlot = new IntPtr[1];
            try
            {
                fixed (float* logitsPtr = logitsOut)
                {
                    for (int r = 0; r < tp; r++)
                    {
                        var a = _tpDecodeArrays[r];
                        GgmlBasicOps.SetActiveRank(r);

                        // Only rank 0 folds the final norm + LM head: the head is
                        // the biggest tensor left once the layers are sharded, and
                        // replicating it would undo much of TP's memory saving.
                        bool fold = r == 0;
                        planSlot[0] = IntPtr.Zero;
                        GgmlBasicOps.Gemma4ModelDecode(
                            (IntPtr)hiddenPtr, Config.HiddenSize, Config.NumLayers,
                            a.AttnNorm, a.Qkv, a.QNorm, a.KNorm,
                            a.O, a.PostAttnNorm,
                            a.FfnNorm, a.Gu, a.Down, a.PostFfnNorm,
                            a.KCache, a.VCache,
                            a.HeadDim, a.KvHeads, a.CacheSize, a.IsLocal,
                            a.KvSource,
                            a.RopeBase, a.LayerScalar,
                            a.QkvType, a.QkvNe0, a.QkvNe1, a.QkvBytes,
                            a.OType, a.ONe0, a.ONe1, a.OBytes,
                            a.GuType, a.GuNe0, a.GuNe1, a.GuBytes,
                            a.DownType, a.DownNe0, a.DownNe1, a.DownBytes,
                            Config.NumHeads / tp, startPos,
                            Config.Eps, _slidingWindow,
                            freqFactorsPtr, freqFactorsLen,
                            a.RopeNDims,
                            pleDataPtr, _pleDim,
                            a.PleGate, a.PleGateType, a.PleGateNe0, a.PleGateNe1, a.PleGateBytes,
                            a.PleProj, a.PleProjType, a.PleProjNe0, a.PleProjNe1, a.PleProjBytes,
                            a.PlePostNorm,
                            _kvCacheDtype.GgmlType(),
                            a.K, a.KType, a.KNe0, a.KNe1, a.KBytes,
                            a.V, a.VType, a.VNe0, a.VNe1, a.VBytes,
                            fold ? (IntPtr)logitsPtr : IntPtr.Zero, fold ? Config.VocabSize : 0,
                            fold ? lmqw.CacheKey : IntPtr.Zero, lmqw.GgmlType, lmqw.Ne0, lmqw.Ne1, lmqw.RawBytes,
                            fold ? finalNormPtr : IntPtr.Zero, _finalLogitSoftcap,
                            tpDegree: tp, tpPlanOut: planSlot);

                        if (planSlot[0] == IntPtr.Zero)
                            return false;
                        _tpDecodePlans[r] = planSlot[0];
                    }

                    GgmlBasicOps.TensorParallelExecutePlans(_tpDecodePlans);
                }
            }
            catch (InvalidOperationException ex)
            {
                // The native side declined (no persistent graph for this window).
                // Fall back to the per-op TP forward for this token.
                if (!_tpFusedDecodeDeclineLogged)
                {
                    _tpFusedDecodeDeclineLogged = true;
                    Console.WriteLine($"  Gemma4 fused tensor-parallel decode declined: {ex.Message}; " +
                        "falling back to the per-op TP forward (slower). Reported once.");
                }
                return false;
            }
            finally
            {
                GgmlBasicOps.SetActiveRank(previousRank);
            }

            _tpKvDeviceDirty = true;
            if (!_tpFusedDecodeLogged)
            {
                _tpFusedDecodeLogged = true;
                Console.WriteLine($"  Gemma4 fused tensor-parallel decode active ({tp} GPUs, " +
                    $"{2 * Config.NumLayers} AllReduce points/token).");
            }
            return true;
        }

        /// <summary>
        /// Fused tensor-parallel prefill: the whole trunk over N tokens as one
        /// graph per rank, cut at the same two projections per layer. Writes the
        /// per-row hidden state back into <paramref name="hidden"/>; the caller
        /// still owns the final norm and the LM head, exactly as on the
        /// single-GPU verify path. Returns false when the native kernel declines
        /// (an SWA window it cannot express, an unsupported head dim).
        /// </summary>
        private unsafe bool TryGemma4FusedModelVerifyTP(Tensor hidden, int startPos, int n,
            Tensor perLayerInputs, HashSet<int> exceptPositions)
        {
            if (!_tpFusedDecodeReady || n <= 1)
                return false;

            // One byte per chunk token at any startPos. This used to be built only
            // at startPos 0, so a media chunk after a prefix ran here with no
            // soft-token mask at all (plain causal attention over the image).
            byte[] isExcept = ChunkSoftTokenMask(exceptPositions, startPos, n);

            int tp = TpDegree;
            float* hiddenPtr = GetFloatPtr(hidden);
            IntPtr pleDataPtr = perLayerInputs != null ? (IntPtr)GetFloatPtr(perLayerInputs) : IntPtr.Zero;

            IntPtr freqFactorsPtr = IntPtr.Zero;
            int freqFactorsLen = 0;
            if (_weights.TryGetValue("rope_freqs.weight", out var freqTensor))
            {
                freqFactorsPtr = (IntPtr)GetFloatPtr(freqTensor);
                freqFactorsLen = (int)freqTensor.ElementCount();
            }

            int previousRank = GgmlBasicOps.GetActiveRank();
            var planSlot = new IntPtr[1];
            try
            {
                for (int r = 0; r < tp; r++)
                {
                    var a = _tpDecodeArrays[r];
                    GgmlBasicOps.SetActiveRank(r);
                    planSlot[0] = IntPtr.Zero;
                    bool ok = GgmlBasicOps.Gemma4ModelVerify(
                        (IntPtr)hiddenPtr, Config.HiddenSize, Config.NumLayers, n,
                        a.AttnNorm, a.Qkv, a.QNorm, a.KNorm,
                        a.O, a.PostAttnNorm,
                        a.FfnNorm, a.Gu, a.Down, a.PostFfnNorm,
                        a.KCache, a.VCache,
                        a.HeadDim, a.KvHeads, a.CacheSize, a.IsLocal,
                        a.RopeBase, a.LayerScalar,
                        a.QkvType, a.QkvNe0, a.QkvNe1, a.QkvBytes,
                        a.OType, a.ONe0, a.ONe1, a.OBytes,
                        a.GuType, a.GuNe0, a.GuNe1, a.GuBytes,
                        a.DownType, a.DownNe0, a.DownNe1, a.DownBytes,
                        Config.NumHeads / tp, startPos, Config.Eps,
                        freqFactorsPtr, freqFactorsLen, a.RopeNDims,
                        _kvCacheDtype.GgmlType(),
                        a.K, a.KType, a.KNe0, a.KNe1, a.KBytes,
                        a.V, a.VType, a.VNe0, a.VNe1, a.VBytes,
                        a.KvSource,
                        pleDataPtr, _pleDim,
                        a.PleGate, a.PleGateType, a.PleGateNe0, a.PleGateNe1, a.PleGateBytes,
                        a.PleProj, a.PleProjType, a.PleProjNe0, a.PleProjNe1, a.PleProjBytes,
                        a.PlePostNorm,
                        isExcept,
                        tpDegree: tp, tpPlanOut: planSlot);

                    if (!ok || planSlot[0] == IntPtr.Zero)
                    {
                        // Ranks already built have parked graphs; they are recycled
                        // when that rank next builds one, so nothing leaks.
                        return false;
                    }
                    _tpDecodePlans[r] = planSlot[0];
                }

                GgmlBasicOps.TensorParallelExecutePlans(_tpDecodePlans);
            }
            catch (InvalidOperationException ex)
            {
                if (!_tpFusedPrefillDeclineLogged)
                {
                    _tpFusedPrefillDeclineLogged = true;
                    Console.WriteLine($"  Gemma4 fused tensor-parallel prefill declined: {ex.Message}; " +
                        "falling back to the per-op TP forward (slower). Reported once.");
                }
                return false;
            }
            finally
            {
                GgmlBasicOps.SetActiveRank(previousRank);
            }

            _tpKvDeviceDirty = true;
            if (!_tpFusedPrefillLogged)
            {
                _tpFusedPrefillLogged = true;
                Console.WriteLine($"  Gemma4 fused tensor-parallel prefill active ({tp} GPUs).");
            }
            return true;
        }

        private bool _tpFusedPrefillLogged;

        /// <summary>
        /// The tail the fused TP prefill leaves to the caller: output norm, take
        /// the last row, LM head, logit softcap. All replicated work, so it runs
        /// on rank 0 where the (unsharded) LM head lives.
        /// </summary>
        private float[] Gemma4TpFinalNormAndLmHead(Tensor trunkOutput, int seqLen)
        {
            Tensor normed = RMSNormOp(trunkOutput, "output_norm.weight");
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

            string outputWeight = _hasTiedOutput ? "token_embd.weight" : "output.weight";
            Tensor logitsTensor = LinearForward(lastHidden, outputWeight);
            lastHidden.Dispose();

            if (_finalLogitSoftcap > 0f)
                ApplyLogitSoftcap(logitsTensor);

            float[] logits = TensorToFloatArray(logitsTensor);
            logitsTensor.Dispose();
            return logits;
        }

        /// <summary>
        /// Pull the device-resident KV caches back to host memory. The fused TP
        /// decode writes KV rows straight into the cached device copies; the
        /// op-at-a-time TP prefill reads the same tensors out of host memory, so
        /// without this a second turn attends over a cache missing every token the
        /// fused path produced.
        /// </summary>
        private void SyncGemma4TpKvCacheToHost()
        {
            if (!_tpKvDeviceDirty || _tpKvCacheK == null)
                return;
            _tpKvDeviceDirty = false;

            var seen = new HashSet<IntPtr>();
            for (int l = 0; l < _tpKvCacheK.Length; l++)
            {
                if (_tpKvCacheK[l] == null) continue;
                for (int r = 0; r < _tpKvCacheK[l].Length; r++)
                {
                    SyncOneKvTensor(_tpKvCacheK[l][r], seen);
                    SyncOneKvTensor(_tpKvCacheV[l][r], seen);
                }
            }
        }

        private static void SyncOneKvTensor(Tensor t, HashSet<IntPtr> seen)
        {
            if (t == null) return;
            IntPtr ptr = TensorComputePrimitives.GetStoragePointer(t);
            // KV-sharing layers alias their donor's tensors; sync each buffer once.
            if (ptr == IntPtr.Zero || !seen.Add(ptr)) return;
            long bytes = t.ElementCount() * t.ElementType.Size();
            try { GgmlBasicOps.SyncHostBuffer(ptr, bytes); }
            catch (InvalidOperationException) { /* never went device-resident */ }
        }
    }
}
