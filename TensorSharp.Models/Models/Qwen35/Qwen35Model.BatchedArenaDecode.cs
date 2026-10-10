// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

// TRUE token-batched decode for the qwen35 family over the SLOT-STABLE ARENA
// kernel (ggml_ops_qwen35_batched_arena.cpp) — the GPT-OSS arena design ported
// to the hybrid GDN + attention architecture. One fused graph decodes one
// token for every concurrent request per step: weights are read once, the
// attention KV lives in persistent per-layer arenas written by one set_rows
// and read by flash-attention in that graph, and the GDN conv/delta recurrent
// state lives in per-slot device arenas updated in-graph — no per-step host
// round trips. CUDA can capture the graph, while Metal retains and replays the
// same slot-stable graph across request churn.
//
// The engine drives this through the same three IBatchedPagedModel hooks
// GPT-OSS implements: TryForwardBatchedFusedDecode (host logits),
// TryForwardBatchedFusedDecodeSampled (in-graph argmax) and CanBatchDecode
// (per-sequence growth/holder gate). Any decline falls back to the serial
// round-robin path, whose solo kernel entry flushes and retires arena slots
// through the native coherence hooks before touching a holder's caches.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using TensorSharp.GGML;

namespace TensorSharp.Models
{
    public partial class Qwen35Model
    {
        private string _lastArenaDeclineLogged;
        private float[] _arenaLogitsStaging;

        public string BatchedFusedDecodeDeclineReason { get; private set; }

        /// <summary>
        /// How many batched arena decode steps this model has actually served.
        ///
        /// <para>
        /// A decline is loud (it prints a reason) but a fallback is not
        /// distinguishable from success by looking at the answer: the round-robin
        /// path produces correct tokens too. A test that widens the arena's
        /// admission rules therefore cannot prove it changed anything without a
        /// positive signal, so this counter is the signal.
        /// </para>
        /// </summary>
        internal long ArenaBatchedDecodeSteps => _arenaBatchedDecodeSteps;
        private long _arenaBatchedDecodeSteps;

        /// <summary>A sequence can join the arena batch when its holder exists
        /// and needs no growth — one growing/unbound sequence falls to the
        /// round-robin loop alone instead of declining the whole batch.</summary>
        public bool CanBatchDecode(string requestId, int position)
        {
            if (HasStreamingWeights) return false;
            if (_fusedHolders == null || !_fusedHolders.TryGetValue(requestId, out var h) || h.K == null)
                return false;
            return position + 1 <= h.KvCapacity;
        }

        public unsafe bool TryForwardBatchedFusedDecode(
            IReadOnlyList<string> requestIds, int[] tokens, int[] positions, float[][] outLogits)
            => ForwardArenaBatchedDecodeCore(requestIds, tokens, positions, outLogits, null);

        public unsafe bool TryForwardBatchedFusedDecodeSampled(
            IReadOnlyList<string> requestIds, int[] tokens, int[] positions, int[] outNextTokens)
            => ForwardArenaBatchedDecodeCore(requestIds, tokens, positions, null, outNextTokens);

        // Convert a holder's host conv RING (rotating write index) into the
        // ggml [time, channel] layout the fused kernels use, in the holder's
        // own ConvScratch. Identical to the solo decode's reseed transform —
        // after it, the scratch bytes are the state's source of truth and the
        // ring is no longer consulted (FdStateResident = true).
        private unsafe void ConvertHolderRingToScratch(Qwen35KvCacheHolder h)
        {
            int convDim = _convKernel - 1;
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;
            int convBlock = convDim * qkvDim;
            float* convBase = (float*)h.ConvScratch;
            for (int l = 0; l < Config.NumLayers; l++)
            {
                if (!_isRecurrent[l]) continue;
                float* conv = convBase + (long)_fdGdnSlot[l] * convBlock;
                float[] ring = h.ConvState[l];
                int w = h.ConvWriteIdx[l];
                for (int t = 0; t < convDim; t++)
                {
                    int slot = (w + t) % convDim;
                    int srcBase = slot * qkvDim;
                    for (int ch = 0; ch < qkvDim; ch++)
                        conv[ch * convDim + t] = ring[srcBase + ch];
                }
                // The resident device copies (if any) predate this rewrite;
                // the solo reseed invalidates both keys, mirror it exactly.
                GgmlBasicOps.InvalidateHostBuffer((IntPtr)conv);
                GgmlBasicOps.InvalidateHostBuffer(GdnDeltaStatePointer(h.DeltaState[l]));
            }
        }

        /// <summary>Sync a checked-in holder's device-resident recurrent state (the solo fused decode's
        /// conv scratch and delta mirrors, and any verify-owned slices) back to its host bytes and mark it
        /// host-authoritative, the way <see cref="SettleForCopy"/> does for the whole holder.</summary>
        private void SettleSoloDecodeStateForArena(Qwen35KvCacheHolder holder)
        {
            Qwen35KvCacheHolder previous = SnapshotActiveCache();
            LoadCacheHolder(holder);
            try
            {
                EnsureFusedDecodeStateHostSynchronized();
                DrainDeviceRecurrentState();
                if (!_gdnStateHostDirty)
                    _fdStateResident = false;
                holder.GdnHostDirty = _gdnStateHostDirty;
                holder.FdStateResident = _fdStateResident;
                holder.ArenaStateResident = _arenaStateResident;
            }
            finally
            {
                LoadCacheHolder(previous);
            }
        }

        private bool ArenaDecline(string reason)
        {
            BatchedFusedDecodeDeclineReason = reason;
            if (reason != _lastArenaDeclineLogged)
            {
                _lastArenaDeclineLogged = reason;
                Console.Error.WriteLine($"[qwen35 arena-decode] declined: {reason}");
            }
            return false;
        }

        private unsafe bool ForwardArenaBatchedDecodeCore(
            IReadOnlyList<string> requestIds, int[] tokens, int[] positions,
            float[][] outLogits, int[] outNextTokens)
        {
            BatchedFusedDecodeDeclineReason = null;
            if (HasStreamingWeights)
                return ArenaDecline("file-backed weights do not support persistent fused weight graphs");
            // The native graph is shared by CUDA and Metal. Its recurrent-slot
            // aggregation carries explicit dataflow dependencies, so Metal's
            // concurrent graph scheduler cannot race the column producers.
            if (_backend != BackendType.GgmlCuda && _backend != BackendType.GgmlMetal)
                return ArenaDecline($"backend {_backend} has no arena batched decode graph");
            if (IsTensorParallel)
                return ArenaDecline("tensor parallelism uses separate decode graphs");
            if (_fusedHolders == null)
                return ArenaDecline("per-sequence caches have not been initialized");
            if (_fdUnsupported)
                return ArenaDecline("the whole-model fused decode capability check failed");
            ExitSpecSession();   // a batched decode outside the speculative session ends it
            int n = requestIds.Count;
            if (n < 2 || tokens.Length != n || positions.Length != n)
                return ArenaDecline("a batch requires at least two sequences and matching token/position arrays");
            if ((outLogits == null && outNextTokens == null)
                || (outLogits != null && outLogits.Length != n)
                || (outNextTokens != null && outNextTokens.Length != n))
                return ArenaDecline("batch output arrays do not match the sequence count");
            // The slot-invariant weight descriptors are built by the solo fused
            // decode; the first token of the first sequence always warms them
            // through the round-robin path, so declining here costs one step.
            if (_fdLayers == null)
                return ArenaDecline("fused weight descriptors need one serial decode warmup step");
            if (MoeCpuOffloadConfig.IsEnabled)
                return ArenaDecline("MoE CPU offload active");
            // NVFP4 per-tensor weight scales (the `<base>.scale` sidecars, HF
            // weight_scale_2). Every OTHER fused qwen35 graph multiplies each
            // projection output by its slot from the proj_scales table; the arena
            // graph does not read TSGgmlQwen35LayerDesc::proj_scales at all. Running
            // it anyway would not fail - it would quietly compute every QKV/O/GDN/
            // FFN matmul with an effective scale of 1.0, so the same model would
            // answer correctly with one request in flight and produce garbage the
            // moment a second one joined. Decline instead and let the round-robin
            // per-sequence path (which does apply the scales) serve the batch, the
            // same contract the fused per-block TP kernels already follow.
            if (HasSidecarWeightScales)
                return ArenaDecline("per-tensor sidecar weight scales (NVFP4 scale2) are not applied by the arena graph");
            if (!IsGgmlBackend)
                return ArenaDecline("the arena graph runs on the GGML backends only");
            DType kvDt = _kvCacheDtype.ToDType();
            // One predicate for every fused graph, including this one. The arena used
            // to hardcode F32/F16 here while the solo whole-model graph had already
            // been widened to block-quantized K/V on CUDA and Metal. The arena kernel
            // itself was never the obstacle - it builds its arenas with the caller's
            // ggml_type and strides every copy by ggml_row_size - so the gate alone
            // turned a q8_0 KV cache (what the agent configs pin, to halve KV on a
            // Mac where Metal charges it twice) into a permanent decline: every
            // concurrent decode step fell back to round-robin, one full weight sweep
            // per sequence per token, and aggregate throughput stayed at 1x.
            // The native side still has the last word - it checks backend_supports_op
            // on the arena's set_rows and flash_attn shapes and aborts the build if
            // either is missing, which lands back here as exactly today's decline.
            if (!IsFusedGraphKvCacheDType(kvDt))
                return ArenaDecline(
                    $"KV cache dtype {_kvCacheDtype} is not supported by the fused graphs on {_backend}");
            if (_headKDim != _headVDim || _convKernel <= 1)
                return ArenaDecline("unsupported GDN geometry");

            // Token embedding for the in-graph get_rows.
            (IntPtr ptr, int type, long ne0, long ne1, long bytes) emb;
            QuantizedWeight hostEmbedding = null;
            if (_quantWeights.TryGetValue("token_embd.weight", out QuantizedWeight tokenQw))
            {
                if (!CanUseGgmlQuantizedGetRows(tokenQw.GgmlType) || tokenQw.DevicePreloadTooLarge)
                {
                    if (!tokenQw.HasHostData)
                        return ArenaDecline($"token embedding type {tokenQw.GgmlType} requires host rows but no host data is available");
                    hostEmbedding = tokenQw;
                }
                emb = ResolveW(tokenQw, null);
            }
            else if (_weights.TryGetValue("token_embd.weight", out Tensor tokenF32))
            {
                emb = ResolveW(null, tokenF32);
            }
            else
            {
                return ArenaDecline("token embedding weight missing");
            }
            if (emb.ptr == IntPtr.Zero || emb.ne0 != Config.HiddenSize)
                return ArenaDecline("token embedding shape mismatch");

            var lmh = ResolveW(_lmHeadQW, _lmHeadF32);
            if (lmh.ptr == IntPtr.Zero || _finalNormW == null)
                return ArenaDecline("folded lm_head/final norm missing");

            // Check any checked-out holder back in FIRST: the dict entry for a
            // checked-out request is a stale pre-checkout snapshot (Restore
            // replaces it with a current one), and reading FdStateResident or
            // writing CacheSeqLen through a stale object rolls recurrent state
            // back / desyncs positions.
            RestorePrimaryCache();

            // Torn down underneath a step (the engine is disposed before a model is
            // released, but this is the same guard every sibling hook carries): decline
            // rather than dereference a dictionary that no longer exists.
            if (_fusedHolders == null)
                return ArenaDecline("per-sequence caches were released before decode");

            var holders = new Qwen35KvCacheHolder[n];
            for (int i = 0; i < n; i++)
            {
                if (!_fusedHolders.TryGetValue(requestIds[i], out holders[i]) || holders[i].K == null)
                    return ArenaDecline($"sequence {i} has no fused cache holder");
                if (positions[i] < 0 || positions[i] >= holders[i].KvCapacity)
                    return ArenaDecline($"sequence {i} needs cache growth or has an invalid position");
                if (tokens[i] < 0 || tokens[i] >= emb.ne1)
                    return ArenaDecline($"sequence {i} has an out-of-range token id");
            }

            // A holder whose last step was a SOLO fused decode keeps its recurrent state in the solo
            // path's device mirrors, and the arena's slot seeding did not pick that state up: the
            // sequence decoded on 1.4-2.8 max |dlogit| away from decoding alone on Qwen3.5-9B (Metal)
            // for the rest of the request - every request that was decoding alone when a second one
            // arrived. Bring such a holder's recurrent state back to the host first; the seeding below
            // then starts its slot from those bytes (Qwen35ArenaSoloTransitionTests).
            for (int i = 0; i < n; i++)
                if (holders[i].FdStateResident && !holders[i].ArenaStateResident)
                    SettleSoloDecodeStateForArena(holders[i]);

            // Canonical order: ascending first-attention-layer K storage pointer.
            // With the native registry keyed on the same pointers, a stable set
            // of in-flight holders keeps replaying its captured graph no matter
            // which requests occupy them.
            int firstAttn = -1;
            for (int l = 0; l < Config.NumLayers; l++)
                if (!_isRecurrent[l]) { firstAttn = l; break; }
            if (firstAttn < 0)
                return ArenaDecline("no attention layers");

            var order = new int[n];
            for (int i = 0; i < n; i++) order[i] = i;
            var keys = new ulong[n];
            for (int i = 0; i < n; i++)
                keys[i] = (ulong)TensorComputePrimitives.GetStoragePointer(holders[i].K[firstAttn]).ToInt64();
            Array.Sort(keys, order);

            int numLayers = Config.NumLayers;
            int attnLayers = 0, gdnLayers = 0;
            for (int l = 0; l < numLayers; l++)
                if (_isRecurrent[l]) gdnLayers++; else attnLayers++;

            int convDim = _convKernel - 1;
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;
            long convBlockBytes = (long)convDim * qkvDim * sizeof(float);

            var tokSorted = new int[n];
            var posSorted = new int[n];
            var ropeSorted = new int[n];
            var cacheSizes = new int[n];
            var gdnHostAuth = new int[n];
            var kPtrs = new IntPtr[attnLayers * n];
            var vPtrs = new IntPtr[attnLayers * n];
            var convPtrs = new IntPtr[gdnLayers * n];
            var deltaPtrs = new IntPtr[gdnLayers * n];
            for (int i = 0; i < n; i++)
            {
                var h = holders[order[i]];
                tokSorted[i] = tokens[order[i]];
                posSorted[i] = positions[order[i]];
                // RoPE at KV index + the holder's M-RoPE delta (non-zero past an image).
                ropeSorted[i] = checked(positions[order[i]] + h.RopeDelta);
                cacheSizes[i] = h.KvCapacity;
                if (h.ConvScratch == IntPtr.Zero)
                {
                    // A holder adopted from a primary cache that had never decoded
                    // (see TryFullModelDecodeCore): its GDN truth is the host ring.
                    h.ConvScratch = AllocateConvScratch();
                    h.FdStateResident = false;
                }
                if (!h.FdStateResident)
                {
                    // Host ring is the GDN truth: land it in the scratch layout
                    // once. The residency flip is committed only after the
                    // native call succeeds (conversion is idempotent), so a
                    // decline leaves the solo fallback's reseed contract whole.
                    ConvertHolderRingToScratch(h);
                    gdnHostAuth[i] = 1;
                }
                for (int l = 0, al = 0, gl = 0; l < numLayers; l++)
                {
                    if (!_isRecurrent[l])
                    {
                        kPtrs[al * n + i] = TensorComputePrimitives.GetStoragePointer(h.K[l]);
                        vPtrs[al * n + i] = TensorComputePrimitives.GetStoragePointer(h.V[l]);
                        al++;
                    }
                    else
                    {
                        convPtrs[gl * n + i] = h.ConvScratch + (nint)((long)_fdGdnSlot[l] * convBlockBytes);
                        deltaPtrs[gl * n + i] = GdnDeltaStatePointer(h.DeltaState[l]);
                        gl++;
                    }
                }
            }

            int vocab = Config.VocabSize;
            bool wantLogits = outLogits != null;
            long needed = (long)vocab * n;
            if (_arenaLogitsStaging == null || _arenaLogitsStaging.LongLength < needed)
                _arenaLogitsStaging = new float[needed];
            var logitsBuf = _arenaLogitsStaging;
            var sampledSorted = outNextTokens != null ? new int[n] : null;

            // CUDA GET_ROWS does not cover K/IQ embedding formats (the UD-IQ3
            // model uses Q2_K). Gather just these N rows through the same host
            // dequantizer as solo decode; all transformer/LM-head matmuls still
            // execute together in the persistent arena graph. Use sorted order
            // so each row follows its cache holder through native slot mapping.
            using var embeddingRows = hostEmbedding == null ? null
                : new Tensor(_allocator, DType.Float32, n, Config.HiddenSize);
            if (embeddingRows != null)
                PopulateQuantizedRows(embeddingRows, hostEmbedding, tokSorted);

            int arenaStatus;
            fixed (float* lp = logitsBuf)
            fixed (int* sp = sampledSorted)
            {
                arenaStatus = GgmlBasicOps.Qwen35ArenaDecodeBatchedHiddenStatus(
                    _fdLayers, numLayers, n,
                    tokSorted, posSorted, ropeSorted,
                    kPtrs, vPtrs, convPtrs, deltaPtrs,
                    gdnHostAuth, cacheSizes,
                    Config.NumHeads, Config.NumKVHeads, Config.HeadDim,
                    _ropeDimCount > 0 ? _ropeDimCount : Config.HeadDim, 2,
                    FusedGraphKvCacheTypeId(kvDt),
                    _convKernel, _headKDim, _headVDim, _numKHeads, _numVHeads,
                    Config.Eps, Config.RopeBase, 1.0f / Config.RopeScale,
                    _numExperts, _numExpertsUsed, _expertFfnLength, _sharedExpertFfnLength,
                    _normTopKProb ? 1 : 0, 1.0f,
                    (IntPtr)lp, vocab,
                    lmh.ptr, lmh.type, lmh.ne0, lmh.ne1, lmh.bytes,
                    TensorComputePrimitives.GetStoragePointer(_finalNormW),
                    emb.ptr, emb.type, emb.ne0, emb.ne1, emb.bytes,
                    (IntPtr)sp, wantLogits,
                    embeddingRows == null ? IntPtr.Zero : (IntPtr)GetFloatPtr(embeddingRows));
            }
            if (arenaStatus != 1)
            {
                string err = GgmlBasicOps.LastNativeError();
                ThrowIfArenaDecodeStateUnrecoverable(arenaStatus, err);
                return ArenaDecline($"native declined (n={n}): " +
                    (string.IsNullOrWhiteSpace(err) ? "native library returned no diagnostic" : err));
            }

            for (int i = 0; i < n; i++)
            {
                var h = holders[order[i]];
                if (gdnHostAuth[i] != 0)
                    h.FdStateResident = true;   // arena/scratch chain is now the truth
                if (wantLogits)
                {
                    h.Logits ??= new float[vocab];
                    Array.Copy(logitsBuf, (long)i * vocab, h.Logits, 0, vocab);
                    outLogits[order[i]] = h.Logits;
                }
                if (outNextTokens != null)
                    outNextTokens[order[i]] = sampledSorted[i];
                h.CacheSeqLen = posSorted[i] + 1;
                h.KvHostDirty = true;
                h.GdnHostDirty = true;
                h.ArenaStateResident = true;
            }
            _arenaBatchedDecodeSteps++;
            return true;
        }

        /// <summary>Interpret the native arena's tri-state result. A graph that
        /// started execution can leave only a subset of recurrent layers advanced;
        /// there is no correct serial fallback without a pre-step state checkpoint,
        /// so fail the affected requests and let normal error cleanup recycle their
        /// holders. Zero remains an ordinary pre-compute decline.</summary>
        internal static void ThrowIfArenaDecodeStateUnrecoverable(int status, string nativeError)
        {
            if (status >= 0)
                return;

            throw new InvalidOperationException(
                (string.IsNullOrWhiteSpace(nativeError)
                    ? "Qwen3.5 arena batched decode graph execution failed."
                    : nativeError) +
                " The graph may have partially advanced recurrent state; the affected " +
                "sequences were failed because serial fallback would use stale host state.");
        }
    }
}
