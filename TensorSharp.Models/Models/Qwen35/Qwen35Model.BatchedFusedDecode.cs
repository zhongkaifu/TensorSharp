// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TRUE token-batched fused decode (vLLM-style) for Qwen3.5/3.6 on ggml_cuda.
// Processes N sequences' decode tokens (one per sequence) through the WHOLE
// hybrid transformer in ONE captured-capable ggml graph (native
// TSGgml_Qwen35ModelDecodeBatched). The heavy per-layer matmuls run BATCHED
// over all N tokens (weights read from VRAM once per step = the throughput win
// over N separate single-sequence decodes); the cheap per-token recurrent /
// attention ops are emitted per-sequence inside the same graph.
//
// KV is PAGED: each attention layer owns a device-resident pool
// [total_slots, num_kv_heads*head_dim]; token t writes its K/V to slot_mapping[t]
// (ggml_set_rows) and each sequence gathers its own history from the pool
// (ggml_get_rows over its block-table slot list). The pool persists across
// decode steps (in-place set_rows append). It is seeded from the op-by-op host
// paged pool (_q35PagedK/V) on the first decode after any op-by-op / prefill
// step, and the freshly written token slots are mirrored back to the host pool
// so the two stay consistent (fallback-safe).
//
// GDN recurrent state lives in the existing per-slot host buffers
// (_q35GdnSlotConvBuf / _q35GdnSlotSsmTensor, keyed by the sequence's primary
// block id, the same convention the op-by-op RunBatchedGdnLayerPerSeq uses). It
// is gathered into a contiguous [.., n_seqs] batch buffer before the call and
// scattered back after, so the fused and op-by-op paths share one source of
// truth for the recurrent state.
//
// Default OFF (TS_QWEN35_BATCHED_FUSED=1 to enable) until validated; when off or
// when the kernel declines, ForwardBatch runs its existing op-by-op layer loop.
using System;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models
{
    public partial class Qwen35Model
    {
        // Per-attention-layer device-resident paged K/V pools [total_slots, kvFlat] F32.
        private Tensor[] _bfdPoolK;
        private Tensor[] _bfdPoolV;
        private int _bfdTotalSlots;
        private bool _bfdPoolSeeded;
        private bool _bfdUnsupported;

        // Reusable unmanaged scratch for the batched GDN conv ring (ggml layout)
        // and delta state, plus the per-call descriptor array.
        private IntPtr _bfdConvScratch;
        private long _bfdConvScratchBytes;
        private IntPtr _bfdDeltaScratch;
        private long _bfdDeltaScratchBytes;
        private Qwen35LayerDecodeArgs[] _bfdLayers;
        private int[] _bfdGdnSlot;        // layer -> gdn index (or -1)

        private static bool IsBatchedFusedEnabled()
        {
            // Default ON: the true token-batched fused decode is the vLLM-style
            // continuous-batching decode path for ggml_cuda. Set
            // TS_QWEN35_BATCHED_FUSED=0 to force the op-by-op batched path.
            string raw = Environment.GetEnvironmentVariable("TS_QWEN35_BATCHED_FUSED");
            if (string.IsNullOrEmpty(raw)) return true;
            return raw != "0" && !string.Equals(raw, "false", StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>Invalidate the batched-fused KV-pool seed. Called whenever the
        /// op-by-op path runs (it writes the host paged pool the device pool is
        /// seeded from) or the KV cache is reset, so the next fused decode re-seeds.</summary>
        private void InvalidateBatchedFusedSeed() => _bfdPoolSeeded = false;

        private void EnsureBatchedKvPools(int totalSlots, int numLayers)
        {
            int kvFlat = Config.NumKVHeads * Config.HeadDim;
            bool needRealloc = _bfdPoolK == null || _bfdTotalSlots < totalSlots;
            if (!needRealloc) return;

            if (_bfdPoolK != null)
                for (int l = 0; l < _bfdPoolK.Length; l++) { _bfdPoolK[l]?.Dispose(); _bfdPoolV[l]?.Dispose(); }

            _bfdPoolK = new Tensor[numLayers];
            _bfdPoolV = new Tensor[numLayers];
            for (int l = 0; l < numLayers; l++)
            {
                if (_isRecurrent != null && _isRecurrent[l]) continue; // attention layers only
                _bfdPoolK[l] = new Tensor(_allocator, DType.Float32, totalSlots, kvFlat);
                _bfdPoolV[l] = new Tensor(_allocator, DType.Float32, totalSlots, kvFlat);
            }
            _bfdTotalSlots = totalSlots;
            _bfdPoolSeeded = false; // fresh pools must be (re)seeded from host
            // The captured graph pins the pool device addresses; a realloc moves
            // them, so drop the cached graph (rebuilds against the new pools).
            GgmlBasicOps.Qwen35ResetBatchedDecodeCache();
        }

        /// <summary>Run the whole hybrid transformer over the batch's decode tokens
        /// (one token per sequence) via the native batched fused kernel. Returns the
        /// final pre-output-norm hidden state [numTokens, hidden], or null to decline
        /// (caller falls back to the op-by-op layer loop). On decline the KV-pool seed
        /// is invalidated so state stays consistent.</summary>
        private unsafe Tensor TryRunBatchedFusedDecode(
            Tensor hiddenStates, BatchedForwardContext ctx,
            int numTokens, int numSeqs, int[] positions, int[] slotMapping,
            int blockSize, int[] seqLens)
        {
            if (_bfdUnsupported) return null;
            if (_backend != BackendType.GgmlCuda) return null;
            if (numTokens != numSeqs) return null;        // V1: pure decode, 1 token/seq
            if (_lmHeadQW == null && _lmHeadF32 == null) return null;
            if (hiddenStates.ElementType != DType.Float32 || hiddenStates.DimensionCount != 2) return null;

            int n = Config.NumLayers;
            int headDim = Config.HeadDim;
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;
            int convDim = _convKernel - 1;
            if (convDim <= 0) return null;

            // --- one-time capability gate (mirrors TryFullModelDecode) ---
            if (_bfdLayers == null)
            {
                static bool HasW(QuantizedWeight q, Tensor f) => q != null || f != null;
                for (int l = 0; l < n; l++)
                {
                    bool isMoeL = _isMoeLayer != null && _isMoeLayer[l];
                    bool ffnOk = isMoeL
                        ? ((_ffnGateInpQW[l] != null || _ffnGateInpF32[l] != null)
                            && _layerStackedGate[l] != null && _layerStackedUp[l] != null && _layerStackedDown[l] != null
                            && HasW(_ffnGateShexpQW[l], _ffnGateShexpF32[l]) && HasW(_ffnUpShexpQW[l], _ffnUpShexpF32[l])
                            && HasW(_ffnDownShexpQW[l], _ffnDownShexpF32[l]) && _ffnGateInpShexpVec[l] != null)
                        : (HasW(_ffnGateUpQW[l], _ffnGateUpF32[l]) && HasW(_ffnDownQW[l], _ffnDownF32[l]));
                    bool ok = _attnNormW[l] != null && _postAttnNormW[l] != null && ffnOk;
                    if (ok && !_isRecurrent[l])
                        ok = (HasW(_attnQkvQW[l], _attnQkvF32[l])
                              || (HasW(_attnQQW[l], _attnQF32[l]) && HasW(_attnKQW[l], _attnKF32[l]) && HasW(_attnVQW[l], _attnVF32[l])))
                            && HasW(_attnOutputQW[l], _attnOutputF32[l])
                            && _attnQNormW[l] != null && _attnKNormW[l] != null;
                    if (ok && _isRecurrent[l])
                        ok = HasW(_attnQkvRecQW[l], _attnQkvRecF32[l]) && HasW(_attnGateRecQW[l], _attnGateRecF32[l])
                            && HasW(_ssmBetaQW[l], _ssmBetaF32[l]) && HasW(_ssmAlphaQW[l], _ssmAlphaF32[l])
                            && _ssmConv1dW[l] != null && _ssmDtBiasW[l] != null && _ssmAW[l] != null
                            && _ssmNormW[l] != null && HasW(_ssmOutQW[l], _ssmOutF32[l]);
                    if (!ok) { _bfdUnsupported = true; return null; }
                }
                _bfdGdnSlot = new int[n];
                int gc = 0;
                for (int l = 0; l < n; l++) _bfdGdnSlot[l] = _isRecurrent[l] ? gc++ : -1;
                _bfdLayers = new Qwen35LayerDecodeArgs[n];
            }

            int gdnCount = 0;
            for (int l = 0; l < n; l++) if (_isRecurrent[l]) gdnCount++;

            // Device pools sized to the op-by-op host pool capacity.
            int totalSlots = _q35PagedNumBlocks * _q35PagedBlockSize;
            if (totalSlots <= 0) return null;
            EnsureBatchedKvPools(totalSlots, n);

            int kvFlat = Config.NumKVHeads * headDim;

            // Seed the device pools from the host paged pool (prefill + prior-step
            // history) when not already seeded for the current host-pool contents.
            if (!_bfdPoolSeeded)
            {
                for (int l = 0; l < n; l++)
                {
                    if (_isRecurrent[l]) continue;
                    if (_bfdPoolK[l] == null || _q35PagedK[l] == null) { _bfdUnsupported = true; return null; }
                    long need = (long)totalSlots * kvFlat;
                    if (_q35PagedK[l].LongLength < need) return null;
                    _bfdPoolK[l].SetElementsAsFloat(_q35PagedK[l]);
                    _bfdPoolV[l].SetElementsAsFloat(_q35PagedV[l]);
                }
                _bfdPoolSeeded = true;
            }

            // GDN state scratch (ggml layout), grown as needed.
            long convScratchBytes = (long)gdnCount * convDim * qkvDim * numSeqs * sizeof(float);
            long deltaScratchBytes = (long)gdnCount * _headKDim * _headVDim * _numVHeads * numSeqs * sizeof(float);
            EnsureBfdScratch(ref _bfdConvScratch, ref _bfdConvScratchBytes, convScratchBytes);
            EnsureBfdScratch(ref _bfdDeltaScratch, ref _bfdDeltaScratchBytes, deltaScratchBytes);
            float* convBase = (float*)_bfdConvScratch;
            float* deltaBase = (float*)_bfdDeltaScratch;
            int convBlockPerSeq = convDim * qkvDim;
            int deltaPerSeq = _headKDim * _headVDim * _numVHeads;

            // Gather per-slot GDN state into the batched [.., n_seqs] scratch.
            for (int l = 0; l < n; l++)
            {
                if (!_isRecurrent[l]) continue;
                int gi = _bfdGdnSlot[l];
                for (int s = 0; s < numSeqs; s++)
                {
                    int slot = ctx.Sequences[s].BlockTable.Blocks[0].Id;
                    EnsureGdnSlotAllocated(l, slot);
                    float[] ring = _q35GdnSlotConvBuf[l][slot];
                    int w = _q35GdnSlotConvWriteIdx[l][slot];
                    float* convDst = convBase + (long)gi * convBlockPerSeq * numSeqs + (long)s * convBlockPerSeq;
                    for (int t = 0; t < convDim; t++)
                    {
                        int srcBase = ((w + t) % convDim) * qkvDim;
                        for (int ch = 0; ch < qkvDim; ch++)
                            convDst[ch * convDim + t] = ring[srcBase + ch];
                    }
                    float* deltaSrc = GetFloatPtr(_q35GdnSlotSsmTensor[l][slot]);
                    float* deltaDst = deltaBase + (long)gi * deltaPerSeq * numSeqs + (long)s * deltaPerSeq;
                    Buffer.MemoryCopy(deltaSrc, deltaDst, (long)deltaPerSeq * sizeof(float), (long)deltaPerSeq * sizeof(float));
                }
            }

            // Build per-layer descriptors.
            int kvCacheType = 0; // F32 pool
            for (int l = 0; l < n; l++)
            {
                var a = default(Qwen35LayerDecodeArgs);
                a.StructBytes = Marshal.SizeOf<Qwen35LayerDecodeArgs>();
                a.AttnNormW = (IntPtr)GetFloatPtr(_attnNormW[l]);
                a.PostAttnNormW = (IntPtr)GetFloatPtr(_postAttnNormW[l]);
                bool isMoe = _isMoeLayer != null && _isMoeLayer[l];
                a.IsMoe = isMoe ? 1 : 0;
                if (!isMoe)
                {
                    var gu = ResolveW(_ffnGateUpQW[l], _ffnGateUpF32[l]);
                    var dn = ResolveW(_ffnDownQW[l], _ffnDownF32[l]);
                    a.GuW = gu.Item1; a.GuType = gu.Item2; a.GuNe0 = gu.Item3; a.GuNe1 = gu.Item4; a.GuBytes = gu.Item5;
                    a.DownW = dn.Item1; a.DownType = dn.Item2; a.DownNe0 = dn.Item3; a.DownNe1 = dn.Item4; a.DownBytes = dn.Item5;
                    a.FfDense = (int)(gu.Item4 / 2);
                }
                else
                {
                    var giw = ResolveW(_ffnGateInpQW[l], _ffnGateInpF32[l]);
                    a.GateInpW = giw.Item1; a.GateInpType = giw.Item2; a.GateInpNe0 = giw.Item3; a.GateInpNe1 = giw.Item4; a.GateInpBytes = giw.Item5;
                    var sg = _layerStackedGate[l]; var su = _layerStackedUp[l]; var sd = _layerStackedDown[l];
                    a.GateExps = sg.Data; a.GateExpsType = sg.GgmlType; a.GateExpsBytes = sg.TotalRawBytes;
                    a.UpExps = su.Data; a.UpExpsType = su.GgmlType; a.UpExpsBytes = su.TotalRawBytes;
                    a.DownExps = sd.Data; a.DownExpsType = sd.GgmlType; a.DownExpsBytes = sd.TotalRawBytes;
                    var shg = ResolveW(_ffnGateShexpQW[l], _ffnGateShexpF32[l]);
                    var shu = ResolveW(_ffnUpShexpQW[l], _ffnUpShexpF32[l]);
                    var shd = ResolveW(_ffnDownShexpQW[l], _ffnDownShexpF32[l]);
                    a.ShexpGateW = shg.Item1; a.ShexpGateType = shg.Item2; a.ShexpGateNe0 = shg.Item3; a.ShexpGateNe1 = shg.Item4; a.ShexpGateBytes = shg.Item5;
                    a.ShexpUpW = shu.Item1; a.ShexpUpType = shu.Item2; a.ShexpUpNe0 = shu.Item3; a.ShexpUpNe1 = shu.Item4; a.ShexpUpBytes = shu.Item5;
                    a.ShexpDownW = shd.Item1; a.ShexpDownType = shd.Item2; a.ShexpDownNe0 = shd.Item3; a.ShexpDownNe1 = shd.Item4; a.ShexpDownBytes = shd.Item5;
                    a.ShexpGateInpW = (IntPtr)GetFloatPtr(_ffnGateInpShexpVec[l]);
                }

                if (!_isRecurrent[l])
                {
                    a.IsRecurrent = 0;
                    var o = ResolveW(_attnOutputQW[l], _attnOutputF32[l]);
                    a.OW = o.Item1; a.OType = o.Item2; a.ONe0 = o.Item3; a.ONe1 = o.Item4; a.OBytes = o.Item5;
                    if (_attnQkvQW[l] != null || _attnQkvF32[l] != null)
                    {
                        var qkv = ResolveW(_attnQkvQW[l], _attnQkvF32[l]);
                        a.QkvW = qkv.Item1; a.QkvType = qkv.Item2; a.QkvNe0 = qkv.Item3; a.QkvNe1 = qkv.Item4; a.QkvBytes = qkv.Item5;
                        a.SeparateQkv = 0;
                    }
                    else
                    {
                        var q = ResolveW(_attnQQW[l], _attnQF32[l]);
                        var k = ResolveW(_attnKQW[l], _attnKF32[l]);
                        var v = ResolveW(_attnVQW[l], _attnVF32[l]);
                        a.QkvW = q.Item1; a.QkvType = q.Item2; a.QkvNe0 = q.Item3; a.QkvNe1 = q.Item4; a.QkvBytes = q.Item5;
                        a.KW = k.Item1; a.KType = k.Item2; a.KNe0 = k.Item3; a.KNe1 = k.Item4; a.KBytes = k.Item5;
                        a.VW = v.Item1; a.VType = v.Item2; a.VNe0 = v.Item3; a.VNe1 = v.Item4; a.VBytes = v.Item5;
                        a.SeparateQkv = 1;
                    }
                    a.QNormW = (IntPtr)GetFloatPtr(_attnQNormW[l]);
                    a.KNormW = (IntPtr)GetFloatPtr(_attnKNormW[l]);
                    a.KCache = TensorComputePrimitives.GetStoragePointer(_bfdPoolK[l]);
                    a.VCache = TensorComputePrimitives.GetStoragePointer(_bfdPoolV[l]);
                }
                else
                {
                    a.IsRecurrent = 1;
                    var gq = ResolveW(_attnQkvRecQW[l], _attnQkvRecF32[l]);
                    var gz = ResolveW(_attnGateRecQW[l], _attnGateRecF32[l]);
                    var sb = ResolveW(_ssmBetaQW[l], _ssmBetaF32[l]);
                    var sa = ResolveW(_ssmAlphaQW[l], _ssmAlphaF32[l]);
                    var so = ResolveW(_ssmOutQW[l], _ssmOutF32[l]);
                    a.GdnQkvW = gq.Item1; a.GdnQkvType = gq.Item2; a.GdnQkvNe0 = gq.Item3; a.GdnQkvNe1 = gq.Item4; a.GdnQkvBytes = gq.Item5;
                    a.GdnGateW = gz.Item1; a.GdnGateType = gz.Item2; a.GdnGateNe0 = gz.Item3; a.GdnGateNe1 = gz.Item4; a.GdnGateBytes = gz.Item5;
                    a.SsmBetaW = sb.Item1; a.SsmBetaType = sb.Item2; a.SsmBetaNe0 = sb.Item3; a.SsmBetaNe1 = sb.Item4; a.SsmBetaBytes = sb.Item5;
                    a.SsmAlphaW = sa.Item1; a.SsmAlphaType = sa.Item2; a.SsmAlphaNe0 = sa.Item3; a.SsmAlphaNe1 = sa.Item4; a.SsmAlphaBytes = sa.Item5;
                    a.SsmOutW = so.Item1; a.SsmOutType = so.Item2; a.SsmOutNe0 = so.Item3; a.SsmOutNe1 = so.Item4; a.SsmOutBytes = so.Item5;
                    a.Conv1dW = (IntPtr)GetFloatPtr(_ssmConv1dW[l]);
                    a.SsmDtW = (IntPtr)GetFloatPtr(_ssmDtBiasW[l]);
                    a.SsmAW = (IntPtr)GetFloatPtr(_ssmAW[l]);
                    a.SsmNormW = (IntPtr)GetFloatPtr(_ssmNormW[l]);
                    int gi = _bfdGdnSlot[l];
                    IntPtr convPtr = (IntPtr)(convBase + (long)gi * convBlockPerSeq * numSeqs);
                    IntPtr deltaPtr = (IntPtr)(deltaBase + (long)gi * deltaPerSeq * numSeqs);
                    a.ConvStateIn = convPtr; a.ConvStateOut = convPtr;
                    a.DeltaStateIn = deltaPtr; a.DeltaStateOut = deltaPtr;
                }
                _bfdLayers[l] = a;
            }

            // Build per-token positions / slot mapping (I64) and PADDED per-seq gather
            // lists. The gather length is fixed at padKv = round_up(maxSeqLen, stride)
            // so the captured graph topology is identical token-to-token (CUDA-graph
            // capture); padded positions point at slot 0 and are masked out by the
            // per-seq attention mask (driven by seqLens).
            const int kStride = 64;
            int maxSeqLen = 0;
            for (int s = 0; s < numSeqs; s++) if (seqLens[s] > maxSeqLen) maxSeqLen = seqLens[s];
            int padKv = ((maxSeqLen + kStride - 1) / kStride) * kStride;
            if (padKv > totalSlots) padKv = totalSlots;
            if (padKv < 1) padKv = kStride;

            long[] slotMap64 = new long[numTokens];
            for (int t = 0; t < numTokens; t++) slotMap64[t] = slotMapping[t];
            int[] gatherIdx = new int[(long)numSeqs * padKv];
            int[] seqLensArr = new int[numSeqs];
            for (int s = 0; s < numSeqs; s++)
            {
                var seq = ctx.Sequences[s];
                seqLensArr[s] = seqLens[s];
                int baseOff = s * padKv;
                int realLen = Math.Min(seqLens[s], padKv);
                for (int p = 0; p < realLen; p++)
                {
                    int blk = seq.BlockTable.Blocks[p / blockSize].Id;
                    gatherIdx[baseOff + p] = blk * blockSize + (p % blockSize);
                }
                // padded positions [realLen, padKv) default to slot 0 (masked out).
            }

            float* hiddenPtr = GetFloatPtr(hiddenStates);
            bool ok2;
            fixed (int* posPtr = positions, gidxPtr = gatherIdx, slPtr = seqLensArr)
            fixed (long* slotPtr = slotMap64)
            {
                ok2 = GgmlBasicOps.Qwen35ModelDecodeBatched(
                    _bfdLayers, n, (IntPtr)hiddenPtr, Config.HiddenSize, numTokens, numSeqs,
                    (IntPtr)posPtr, (IntPtr)slotPtr, (IntPtr)gidxPtr, (IntPtr)slPtr, padKv, totalSlots,
                    Config.NumHeads, Config.NumKVHeads, headDim, headDim, 2, kvCacheType,
                    _convKernel, _headKDim, _headVDim, _numKHeads, _numVHeads,
                    Config.Eps, Config.RopeBase, 1.0f / Config.RopeScale,
                    _numExperts, _numExpertsUsed, _expertFfnLength, _sharedExpertFfnLength,
                    _normTopKProb ? 1 : 0, 1.0f);
            }
            if (!ok2)
            {
                _bfdUnsupported = true;
                Console.Error.WriteLine("[bfd] batched fused decode declined (native 0); using op-by-op.");
                return null;
            }

            // hiddenStates host buffer now holds the batched result [numTokens, H].
            InvalidateTensorDeviceCache(hiddenStates);

            // Scatter updated GDN state back to the per-slot host buffers.
            for (int l = 0; l < n; l++)
            {
                if (!_isRecurrent[l]) continue;
                int gi = _bfdGdnSlot[l];
                for (int s = 0; s < numSeqs; s++)
                {
                    int slot = ctx.Sequences[s].BlockTable.Blocks[0].Id;
                    float[] ring = _q35GdnSlotConvBuf[l][slot];
                    float* convSrc = convBase + (long)gi * convBlockPerSeq * numSeqs + (long)s * convBlockPerSeq;
                    for (int t = 0; t < convDim; t++)
                        for (int ch = 0; ch < qkvDim; ch++)
                            ring[t * qkvDim + ch] = convSrc[ch * convDim + t];
                    _q35GdnSlotConvWriteIdx[l][slot] = 0;
                    float* deltaSrc = deltaBase + (long)gi * deltaPerSeq * numSeqs + (long)s * deltaPerSeq;
                    float* deltaDst = GetFloatPtr(_q35GdnSlotSsmTensor[l][slot]);
                    Buffer.MemoryCopy(deltaSrc, deltaDst, (long)deltaPerSeq * sizeof(float), (long)deltaPerSeq * sizeof(float));
                    InvalidateTensorDeviceCache(_q35GdnSlotSsmTensor[l][slot]);
                }
            }

            // Mirror the freshly written token K/V slots back into the host paged
            // pool so it stays the consistent source of truth (re-seed / op-by-op
            // fallback). Download the touched slots from the device pools. For a
            // pure-decode run (no op-by-op interleaving) the device pool persists
            // via in-place set_rows, so the mirror is only needed for consistency
            // across path transitions — skip it with TS_QWEN35_BFD_NOMIRROR=1.
            if (!string.Equals(Environment.GetEnvironmentVariable("TS_QWEN35_BFD_NOMIRROR"), "1", StringComparison.Ordinal))
                MirrorNewSlotsToHostPool(slotMapping, numTokens, kvFlat);
            return hiddenStates;
        }

        // Drain the device-resident GDN state left by the N=1 fast-path fused decode
        // (TryFullModelDecode keeps conv in _fdConvScratch and delta in
        // _deltaStateTensor on the device) back into the host buffers, so a
        // subsequent linear->paged migration reads the LATEST recurrent state.
        private unsafe void DrainFusedDecodeStateForMigration()
        {
            if (!_fdStateResident || _fdConvScratch == IntPtr.Zero || _fdGdnSlot == null)
                return; // host _convState / _deltaStateTensor are already current

            int convDim = _convKernel - 1;
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;
            int gdnCount = 0;
            for (int l = 0; l < Config.NumLayers; l++) if (_isRecurrent[l]) gdnCount++;

            // conv state: download the cacheable device buffer -> _fdConvScratch (ggml
            // [time, channel]) then un-rotate into each layer's host ring (_convState).
            GgmlBasicOps.SyncHostBuffer(_fdConvScratch, (long)gdnCount * convDim * qkvDim * sizeof(float));
            float* convBase = (float*)_fdConvScratch;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!_isRecurrent[l]) continue;
                int gi = _fdGdnSlot[l];
                float* conv = convBase + (long)gi * convDim * qkvDim;
                float[] ring = _convState[l];
                for (int t = 0; t < convDim; t++)
                    for (int ch = 0; ch < qkvDim; ch++)
                        ring[t * qkvDim + ch] = conv[ch * convDim + t];
                _convStateWriteIdx[l] = 0;
                // delta state: drain the device buffer back to host + invalidate the
                // device cache so the subsequent Ops.Copy re-reads the current state.
                GgmlBasicOps.SyncHostBuffer((IntPtr)GetFloatPtr(_deltaStateTensor[l]), _deltaStateTensor[l].Storage.ByteLength);
                InvalidateTensorDeviceCache(_deltaStateTensor[l]);
            }
            _fdStateResident = false;
        }

        private void EnsureBfdScratch(ref IntPtr buf, ref long curBytes, long needBytes)
        {
            if (needBytes <= curBytes && buf != IntPtr.Zero) return;
            if (buf != IntPtr.Zero) Marshal.FreeHGlobal(buf);
            buf = Marshal.AllocHGlobal((IntPtr)needBytes);
            curBytes = needBytes;
        }

        // Download only the newly-written token slots from the device pools back
        // into the host paged pool (_q35PagedK/V) so the host pool remains the full
        // source of truth for the next re-seed and for any op-by-op fallback. Only
        // the N touched rows per attention layer are downloaded (not the whole pool).
        private unsafe void MirrorNewSlotsToHostPool(int[] slotMapping, int numTokens, int kvFlat)
        {
            int n = Config.NumLayers;
            for (int l = 0; l < n; l++)
            {
                if (_isRecurrent[l] || _bfdPoolK[l] == null) continue;
                for (int t = 0; t < numTokens; t++)
                {
                    int slot = slotMapping[t];
                    using (var rowK = _bfdPoolK[l].Narrow(0, slot, 1))
                        Array.Copy(rowK.GetElementsAsFloat(kvFlat), 0, _q35PagedK[l], (long)slot * kvFlat, kvFlat);
                    using (var rowV = _bfdPoolV[l].Narrow(0, slot, 1))
                        Array.Copy(rowV.GetElementsAsFloat(kvFlat), 0, _q35PagedV[l], (long)slot * kvFlat, kvFlat);
                }
            }
        }
    }
}
