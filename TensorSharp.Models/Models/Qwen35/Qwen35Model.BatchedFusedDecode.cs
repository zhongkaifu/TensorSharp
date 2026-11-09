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
// KV is PAGED: each attention layer owns a device pool [device_slots,
// num_kv_heads*head_dim]; token t writes its K/V to its device slot
// (ggml_set_rows) and each sequence gathers its own history from the pool
// (ggml_get_rows over its block table's device slots). The host paged pool
// (_q35PagedK/V) stays the source of truth: BatchedDecodePool maps the batch's
// blocks into a compact device pool, seeds it from the host pool and downloads
// every row a step writes (see BatchedDecodePool.cs for the order).
//
// GDN recurrent state lives in the existing per-slot host buffers
// (_q35GdnSlotConvBuf / _q35GdnSlotSsmTensor, keyed by the sequence's primary
// block id, the same convention the op-by-op RunBatchedGdnLayerPerSeq uses). It
// is gathered into a contiguous [.., n_seqs] batch buffer before the call and
// scattered back after, so the fused and op-by-op paths share one source of
// truth for the recurrent state.
//
// It serves all-decode batches on ggml_cuda; when the kernel declines,
// ForwardBatch runs its op-by-op layer loop.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Models
{
    public partial class Qwen35Model
    {
        // The device K/V pool, its block map and its transfers (BatchedDecodePool.cs).
        private BatchedDecodePool _bfdPool;
        private GgmlBfdPoolTransfers _bfdTransfers;
        private KvBlock[] _bfdBlockRefs;   // host block id -> the block the scheduler owns (liveness)
        private readonly List<int> _bfdStepBlocks = new();
        private bool _bfdUnsupported;
        private bool _bfdDeferWarned;

        // Reusable unmanaged scratch for the batched GDN conv ring (ggml layout)
        // and delta state, plus the per-call descriptor array.
        private IntPtr _bfdConvScratch;
        private long _bfdConvScratchBytes;
        private IntPtr _bfdDeltaScratch;
        private long _bfdDeltaScratchBytes;
        private Qwen35LayerDecodeArgs[] _bfdLayers;
        private int[] _bfdGdnSlot;        // layer -> gdn index (or -1)

        /// <summary>The host paged pool was written under the device pool: an op-by-op step, a restore or a KV reset.
        /// Rows a fused step still owes the host pool reach it first; the next fused decode reseeds.</summary>
        private void InvalidateBatchedFusedSeed() => _bfdPool?.InvalidateSeed();

        /// <summary>
        /// Whether a native decline of the batched fused decode is a shortage (allocating its graph, context or
        /// device pool) that a later step may not meet, rather than something the kernel does not support. A
        /// shortage defers that step to the op-by-op loop; anything else stops the kernel for the model.
        /// </summary>
        internal static bool BfdDeclineIsDeferrable(string nativeError)
            => !string.IsNullOrEmpty(nativeError)
               && (nativeError.Contains("allocat", StringComparison.OrdinalIgnoreCase)
                   || nativeError.Contains("acquire", StringComparison.OrdinalIgnoreCase)
                   || nativeError.Contains("out of memory", StringComparison.OrdinalIgnoreCase)
                   || nativeError.Contains("budget", StringComparison.OrdinalIgnoreCase));

        private void DeferBatchedFusedDecode(string reason)
        {
            if (_bfdDeferWarned)
                return;
            _bfdDeferWarned = true;
            Console.Error.WriteLine($"[bfd] the batched fused decode is short of device memory ({reason}); such steps " +
                "run op-by-op (slower) until memory frees, and the kernel is tried again on the next step. Further " +
                "deferrals are not reported.");
        }

        private void DeclineBatchedFusedDecode(string reason)
        {
            _bfdUnsupported = true;
            Console.Error.WriteLine($"[bfd] batched fused decode declined ({reason}); every later all-decode batch " +
                "runs op-by-op.");
        }

        /// <summary>Run the whole hybrid transformer over the batch's decode tokens
        /// (one token per sequence) via the native batched fused kernel. Returns the
        /// final pre-output-norm hidden state [numTokens, hidden], or null to decline
        /// (caller falls back to the op-by-op layer loop, which invalidates the
        /// device pool's seed).</summary>
        private unsafe Tensor TryRunBatchedFusedDecode(
            Tensor hiddenStates, BatchedForwardContext ctx,
            int numTokens, int numSeqs, int[] positions, int[] slotMapping,
            int blockSize, int[] seqLens)
        {
            if (_bfdUnsupported) return null;
            if (_backend != BackendType.GgmlCuda) return null;
            // This kernel inlines every layer's expert matmuls on the accelerator,
            // so running it under MoE CPU offload would upload exactly the experts
            // the flag keeps in system RAM.
            if (MoeCpuOffloadConfig.IsEnabled) return null;
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
                        : (HasDenseFfnWeights(l) && HasW(_ffnDownQW[l], _ffnDownF32[l]));
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
                    if (!ok)
                    {
                        DeclineBatchedFusedDecode($"layer {l} lacks a weight the fused kernel binds");
                        return null;
                    }
                }
                _bfdGdnSlot = new int[n];
                int gc = 0;
                for (int l = 0; l < n; l++) _bfdGdnSlot[l] = _isRecurrent[l] ? gc++ : -1;
                _bfdLayers = new Qwen35LayerDecodeArgs[n];
            }

            int gdnCount = 0;
            for (int l = 0; l < n; l++) if (_isRecurrent[l]) gdnCount++;
            if (_q35PagedNumBlocks <= 0 || _q35PagedBlockSize != blockSize) return null;
            for (int l = 0; l < n; l++)
                if (!_isRecurrent[l] && (_q35PagedK[l] == null || _q35PagedV[l] == null)) return null;

            // The blocks this step reads (each sequence's history) and writes (its new token's slot).
            if (_bfdPool == null || _bfdPool.BlockSize != blockSize)
            {
                _bfdTransfers?.Release();
                _bfdTransfers = new GgmlBfdPoolTransfers(this);
                _bfdPool = new BatchedDecodePool(blockSize, _bfdTransfers);
            }
            if (_bfdBlockRefs == null || _bfdBlockRefs.Length < _q35PagedNumBlocks)
                Array.Resize(ref _bfdBlockRefs, _q35PagedNumBlocks);
            _bfdStepBlocks.Clear();
            for (int s = 0; s < numSeqs; s++)
            {
                IReadOnlyList<KvBlock> blocks = ctx.Sequences[s].BlockTable.Blocks;
                int used = Math.Max(1, (seqLens[s] + blockSize - 1) / blockSize);   // seqLens counts the new token
                int written = slotMapping[s] / blockSize;
                bool writesMapped = false;
                for (int b = 0; b < blocks.Count && b < used; b++)
                {
                    _bfdStepBlocks.Add(blocks[b].Id);
                    _bfdBlockRefs[blocks[b].Id] = blocks[b];
                    writesMapped |= blocks[b].Id == written;
                }
                for (int b = used; !writesMapped && b < blocks.Count; b++)
                    if (blocks[b].Id == written)
                    {
                        _bfdStepBlocks.Add(written);
                        _bfdBlockRefs[written] = blocks[b];
                        writesMapped = true;
                    }
                if (!writesMapped)
                    return null;   // the token's slot is in none of the sequence's blocks: not a plain decode step
            }
            KvBlock[] refs = _bfdBlockRefs;
            if (!_bfdPool.TryPrepare(_bfdStepBlocks, id => id < refs.Length && refs[id] is { RefCount: > 0 }, out string deferReason))
            {
                DeferBatchedFusedDecode(deferReason);
                return null;
            }
            int deviceSlots = _bfdPool.DeviceSlots;

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
                a.ProjScales = ProjScalesPtr(l);   // every layer kind, not just dense FFN
                a.StructBytes = Marshal.SizeOf<Qwen35LayerDecodeArgs>();
                a.AttnNormW = (IntPtr)GetFloatPtr(_attnNormW[l]);
                a.PostAttnNormW = (IntPtr)GetFloatPtr(_postAttnNormW[l]);
                bool isMoe = _isMoeLayer != null && _isMoeLayer[l];
                a.IsMoe = isMoe ? 1 : 0;
                if (!isMoe)
                {
                    var dn = ResolveW(_ffnDownQW[l], _ffnDownF32[l]);
                    a.DownW = dn.Item1; a.DownType = dn.Item2; a.DownNe0 = dn.Item3; a.DownNe1 = dn.Item4; a.DownBytes = dn.Item5;
                    FillDenseFfnArgs(ref a, l);
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
                    a.KCache = TensorComputePrimitives.GetStoragePointer(_bfdTransfers.PoolK[l]);
                    a.VCache = TensorComputePrimitives.GetStoragePointer(_bfdTransfers.PoolV[l]);
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

            // Build per-token device slots and PADDED per-seq gather lists of device
            // slots. The gather length is fixed at padKv = round_up(maxSeqLen, stride)
            // so the captured graph topology is identical token-to-token (CUDA-graph
            // capture); padded positions point at slot 0 and are masked out by the
            // per-seq attention mask (driven by seqLens).
            const int kStride = 64;
            int maxSeqLen = 0;
            for (int s = 0; s < numSeqs; s++) if (seqLens[s] > maxSeqLen) maxSeqLen = seqLens[s];
            int padKv = ((maxSeqLen + kStride - 1) / kStride) * kStride;
            if (padKv > deviceSlots) padKv = deviceSlots;
            if (padKv < 1) padKv = kStride;

            long[] slotMap64 = new long[numTokens];
            for (int t = 0; t < numTokens; t++) slotMap64[t] = _bfdPool.DeviceSlotOf(slotMapping[t]);
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
                    gatherIdx[baseOff + p] = checked((int)_bfdPool.DeviceSlotOf((long)blk * blockSize + p % blockSize));
                }
                // padded positions [realLen, padKv) default to slot 0 (masked out).
            }

            // The rows this step writes are owed to the host pool from here on (BatchedDecodePool.BeginStep).
            _bfdPool.BeginStep(slotMapping.AsSpan(0, numTokens));
            float* hiddenPtr = GetFloatPtr(hiddenStates);
            bool ok2;
            fixed (int* posPtr = positions, gidxPtr = gatherIdx, slPtr = seqLensArr)
            fixed (long* slotPtr = slotMap64)
            {
                ok2 = GgmlBasicOps.Qwen35ModelDecodeBatched(
                    _bfdLayers, n, (IntPtr)hiddenPtr, Config.HiddenSize, numTokens, numSeqs,
                    (IntPtr)posPtr, (IntPtr)slotPtr, (IntPtr)gidxPtr, (IntPtr)slPtr, padKv, deviceSlots,
                    // rope_n_dims must be the model's partial-rotary width (rope.dimension_count,
                    // 64 of the 256-dim head), NOT headDim — passing headDim rotates all dims with
                    // the wrong frequencies and mismatches the KV cache, degenerating output. See
                    // TryFullModelDecode.
                    Config.NumHeads, Config.NumKVHeads, headDim, _ropeDimCount > 0 ? _ropeDimCount : headDim, 2, kvCacheType,
                    _convKernel, _headKDim, _headVDim, _numKHeads, _numVHeads,
                    Config.Eps, Config.RopeBase, 1.0f / Config.RopeScale,
                    _numExperts, _numExpertsUsed, _expertFfnLength, _sharedExpertFfnLength,
                    _normTopKProb ? 1 : 0, 1.0f);
            }
            if (!ok2)
            {
                _bfdPool.AbandonStep();
                string error = GgmlBasicOps.LastNativeError();
                if (BfdDeclineIsDeferrable(error))
                    DeferBatchedFusedDecode(error);
                else
                    DeclineBatchedFusedDecode(error);
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

            // The K/V rows the step wrote in the device pool, into the host paged pool.
            _bfdPool.CompleteStep();
            return hiddenStates;
        }

        /// <summary>
        /// The device pool's tensors and its transfers: [device_slots, kvFlat] F32 per attention layer, whose device
        /// copies the native graph binds (keyed by the tensors' host pointers) and appends to. Seeds are uploaded
        /// straight into those device copies and written rows downloaded straight from them; the tensors' own host
        /// bytes are never read.
        /// </summary>
        private sealed class GgmlBfdPoolTransfers : IBatchedDecodePoolTransfers
        {
            private readonly Qwen35Model _model;
            public Tensor[] PoolK { get; private set; }
            public Tensor[] PoolV { get; private set; }
            public int DeviceSlots { get; private set; }
            public string LastError { get; private set; } = "";

            public GgmlBfdPoolTransfers(Qwen35Model model) => _model = model;

            private int KvFlat => _model.Config.NumKVHeads * _model.Config.HeadDim;
            private long RowBytes => (long)KvFlat * sizeof(float);

            public bool Resize(int deviceSlots)
            {
                Release();
                int n = _model.Config.NumLayers;
                try
                {
                    PoolK = new Tensor[n];
                    PoolV = new Tensor[n];
                    for (int l = 0; l < n; l++)
                    {
                        if (_model._isRecurrent[l]) continue;
                        PoolK[l] = new Tensor(_model._allocator, DType.Float32, deviceSlots, KvFlat);
                        PoolV[l] = new Tensor(_model._allocator, DType.Float32, deviceSlots, KvFlat);
                    }
                }
                catch (Exception ex) when (ex is OutOfMemoryException or InvalidOperationException)
                {
                    LastError = ex.Message;
                    Release();
                    return false;
                }
                DeviceSlots = deviceSlots;
                return true;
            }

            /// <summary>Drop the pool's tensors and their device copies (a disposed tensor's device copy would
            /// otherwise stay cached under its host pointer).</summary>
            public void Release()
            {
                foreach (Tensor[] set in new[] { PoolK, PoolV })
                    if (set != null)
                        foreach (Tensor t in set)
                            if (t != null)
                            {
                                _model.InvalidateTensorDeviceCache(t);
                                t.Dispose();
                            }
                PoolK = PoolV = null;
                DeviceSlots = 0;
            }

            public unsafe bool Upload(IReadOnlyList<PoolRowRange> ranges, bool zeroFirst)
            {
                long rowBytes = RowBytes;
                var src = new long[ranges.Count];
                var dst = new long[ranges.Count];
                var len = new long[ranges.Count];
                for (int i = 0; i < ranges.Count; i++)
                {
                    src[i] = ranges[i].HostSlot * rowBytes;
                    dst[i] = ranges[i].DeviceSlot * rowBytes;
                    len[i] = ranges[i].Rows * rowBytes;
                }
                for (int l = 0; l < _model.Config.NumLayers; l++)
                {
                    if (_model._isRecurrent[l]) continue;
                    foreach (var (host, pool) in new[] { (_model._q35PagedK[l], PoolK[l]), (_model._q35PagedV[l], PoolV[l]) })
                        fixed (float* h = host)
                            if (!GgmlBasicOps.UploadToDeviceCopyRanges((IntPtr)h, host.LongLength * sizeof(float),
                                    TensorComputePrimitives.GetStoragePointer(pool), pool.Storage.ByteLength, src, dst, len, zeroFirst))
                            {
                                LastError = GgmlBasicOps.LastNativeError();
                                return false;
                            }
                }
                return true;
            }

            public unsafe bool Download(IReadOnlyList<PoolRowRange> ranges)
            {
                long rowBytes = RowBytes;
                var src = new long[ranges.Count];
                var dst = new long[ranges.Count];
                var len = new long[ranges.Count];
                for (int i = 0; i < ranges.Count; i++)
                {
                    src[i] = ranges[i].DeviceSlot * rowBytes;
                    dst[i] = ranges[i].HostSlot * rowBytes;
                    len[i] = ranges[i].Rows * rowBytes;
                }
                for (int l = 0; l < _model.Config.NumLayers; l++)
                {
                    if (_model._isRecurrent[l]) continue;
                    foreach (var (host, pool) in new[] { (_model._q35PagedK[l], PoolK[l]), (_model._q35PagedV[l], PoolV[l]) })
                        fixed (float* h = host)
                        {
                            int r = GgmlBasicOps.DownloadDeviceCopyRanges(TensorComputePrimitives.GetStoragePointer(pool),
                                (IntPtr)h, src, dst, len);
                            if (r != 1)
                            {
                                LastError = r == 0 ? "the device pool has no resident copy" : GgmlBasicOps.LastNativeError();
                                return false;
                            }
                        }
                }
                return true;
            }

            public bool Resident
            {
                get
                {
                    if (PoolK == null) return false;
                    for (int l = 0; l < PoolK.Length; l++)
                        foreach (Tensor t in new[] { PoolK[l], PoolV[l] })
                            if (t != null && !GgmlBasicOps.HasResidentDeviceCopy(
                                    TensorComputePrimitives.GetStoragePointer(t), t.Storage.ByteLength))
                                return false;
                    return true;
                }
            }
        }

        // Drain the device-resident GDN state left by the N=1 fast-path fused decode
        // (TryFullModelDecode keeps conv in _fdConvScratch and delta in
        // _deltaStateTensor on the device) back into the host buffers, so a
        // subsequent linear->paged migration reads the LATEST recurrent state.
        private unsafe void DrainFusedDecodeStateForMigration()
        {
            if (!_fdStateResident || _fdConvScratch == IntPtr.Zero)
                return; // host _convState / _deltaStateTensor are already current

            EnsureFusedDecodeStateHostSynchronized();

            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!_isRecurrent[l]) continue;
                // Migration switches to the paged host source of truth.  The
                // shared helper already downloaded and converted the state; now
                // evict the per-sequence device mirror before copying it.
                InvalidateTensorDeviceCache(_deltaStateTensor[l]);
            }
            _fdStateResident = false;
            _gdnStateHostDirty = false;
        }

        private void EnsureBfdScratch(ref IntPtr buf, ref long curBytes, long needBytes)
        {
            if (needBytes <= curBytes && buf != IntPtr.Zero) return;
            if (buf != IntPtr.Zero) Marshal.FreeHGlobal(buf);
            buf = Marshal.AllocHGlobal((IntPtr)needBytes);
            curBytes = needBytes;
        }
    }
}
