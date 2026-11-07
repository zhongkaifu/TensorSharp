// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

// Batched speculative decoding for concurrent requests on the per-request fused
// holders (ggml_ops_qwen35_batched_verify.cpp). Each step the executor drafts a
// window per sequence with that sequence's own DFlash ring, this model verifies
// every window in one graph over the holders' resident copies, and the commit
// advances each holder's recurrent state by exactly its accepted prefix. The
// holders end each step as a solo fused decode would leave them, so a request
// can move between this path, the batched decode and solo speculation freely.
using System;
using System.Collections.Generic;
using TensorSharp.GGML;
using TensorSharp.Runtime.Speculative;

namespace TensorSharp.Models
{
    public partial class Qwen35Model : IBatchedSpeculativeTarget
    {
        private static readonly bool s_batchedSpecEnabled =
            Environment.GetEnvironmentVariable("TS_QWEN35_BATCHED_SPEC") != "0";

        /// <summary>Largest verify window (anchor + drafts) per sequence. The solo
        /// verify keeps a recurrent snapshot per row; here a row costs a column of a
        /// shared sweep and the commit replays only the accepted prefix, so the window
        /// can be the drafter's whole block. The executor narrows it while drafts are
        /// being rejected. TS_QWEN35_BATCHED_SPEC_WINDOW overrides (2..8).</summary>
        private static readonly int s_batchedSpecWindowSetting =
            int.TryParse(Environment.GetEnvironmentVariable("TS_QWEN35_BATCHED_SPEC_WINDOW"), out int w) ? w : 8;

        /// <summary>Draft cap of the single-sequence verify (see
        /// ISpeculativeTarget.SpecSoloMaxDraftTokens). It was 3 while the verify kept a
        /// recurrent state per row; with the replayed commit an 8-row verify costs 1.3x a
        /// 4-row one and window 7 won on every coding turn measured (see
        /// SpecPreferredDraftWindow). Seven keeps the verify within ggml-cuda's matvec
        /// kernel. TS_QWEN35_SOLO_SPEC_DRAFT overrides.</summary>
        private static readonly int s_soloSpecDraftSetting =
            int.TryParse(Environment.GetEnvironmentVariable("TS_QWEN35_SOLO_SPEC_DRAFT"), out int sd) && sd > 0 ? sd : 7;

        public int SpecSoloMaxDraftTokens => HasDFlash && _backend == BackendType.GgmlCuda ? s_soloSpecDraftSetting : int.MaxValue;

        private bool _batchedSpecScope;
        private Qwen35KvCacheHolder[] _batchedSpecHolders;
        private int[] _batchedSpecOrder;
        private int[] _batchedSpecPositions;
        private int _batchedSpecWindowPending;
        private float[] _batchedSpecFeatures;
        private string _lastBatchedSpecDecline;

        /// <summary>Batched verify steps this model has served (a positive signal for
        /// tests; a decline falls back to the batched decode silently).</summary>
        internal long BatchedSpecVerifySteps { get; private set; }

        public int BatchedSpecWindow(int sequences)
        {
            if (!s_batchedSpecEnabled || sequences < 1 || !HasDFlash || _dflash.MarkovRank > 0)
                return 0;
            if (_backend != BackendType.GgmlCuda || IsTensorParallel || _fdUnsupported || _fdLayers == null)
                return 0;
            if (HasMoEExpertWeights || MoeCpuOffloadConfig.IsEnabled || !ArenaUsesHolderBuffers)
                return 0;
            if (_lmHeadQW != null && _lmHeadQW.Scale != 1.0f)
                return 0;
            return Math.Clamp(s_batchedSpecWindowSetting, 2, Math.Min(8, 1 + _dflash.MaxDraftTokens));
        }

        public int FusedCacheSeqLen(string requestId)
        {
            if (_fusedHolders == null || string.IsNullOrEmpty(requestId))
                return -1;
            // A checked-out holder's dictionary entry is a pre-checkout snapshot.
            if (string.Equals(_activeFusedKey, requestId, StringComparison.Ordinal))
                return _cacheSeqLen;
            return _fusedHolders.TryGetValue(requestId, out var h) && h.K != null ? h.CacheSeqLen : -1;
        }

        public int FusedCacheFreeRows(string requestId)
        {
            if (_fusedHolders == null || string.IsNullOrEmpty(requestId))
                return 0;
            if (string.Equals(_activeFusedKey, requestId, StringComparison.Ordinal))
                return Math.Max(0, _kvCacheCapacity - _cacheSeqLen);
            return _fusedHolders.TryGetValue(requestId, out var h) && h.K != null
                ? Math.Max(0, h.KvCapacity - h.CacheSeqLen) : 0;
        }

        public void BindDraftState(string requestId) => BindDFlashRing(requestId);

        public void ReleaseDraftState(string requestId) => ReleaseDFlashRing(requestId);

        public bool TryDraftBatch(IReadOnlyList<string> requestIds, int[] anchors, float[][] carries, int[] positions,
            int[][] tokensOut, float[][] confOut)
            => s_batchedDraftEnabled && HasDFlash && requestIds.Count >= 2
                && TryFusedDFlashDraftBatch(requestIds, anchors, carries, positions, tokensOut, confOut);

        public int BatchDraftBlockWidth => HasDFlash ? DFlashBatchDraftWidth : 0;

        private static readonly bool s_batchedDraftEnabled =
            Environment.GetEnvironmentVariable("TS_QWEN35_BATCHED_DRAFT") != "0";

        public IDisposable BeginBatchedSpeculation()
        {
            _batchedSpecScope = true;
            // Every lane's ring writes of the step (the carried rows before the draft,
            // the accepted rows at commit) go out as one graph each.
            BeginDeferredDFlashInjects();
            return new BatchedSpecScope(this);
        }

        private sealed class BatchedSpecScope : IDisposable
        {
            private Qwen35Model _model;
            public BatchedSpecScope(Qwen35Model model) => _model = model;
            public void Dispose()
            {
                var model = _model;
                _model = null;
                if (model == null)
                    return;
                model._batchedSpecScope = false;
                model.EndDeferredDFlashInjects();
            }
        }

        private bool BatchedSpecDecline(string reason)
        {
            if (reason != _lastBatchedSpecDecline)
            {
                _lastBatchedSpecDecline = reason;
                Console.Error.WriteLine($"[qwen35 batched-spec] declined: {reason}");
            }
            return false;
        }

        public unsafe bool TryBatchedSpecVerify(IReadOnlyList<string> requestIds, int[] positions, int window,
            int[] tokens, int[] drawn, float[] features)
        {
            _batchedSpecHolders = null;
            int n = requestIds.Count;
            if (BatchedSpecWindow(n) < window || window < 2)
                return BatchedSpecDecline($"no batched verify for {n} sequences x {window} tokens");
            if (positions.Length != n || tokens.Length != n * window || drawn.Length < n * window)
                return BatchedSpecDecline("argument arrays do not match the batch");
            int feat = _dflash.FeatureSize;
            if (features == null || features.LongLength < (long)n * window * feat)
                return BatchedSpecDecline("feature buffer too small");
            if (!IsFusedGraphKvCacheDType(_kvCacheDtype.ToDType()))
                return BatchedSpecDecline($"KV cache dtype {_kvCacheDtype} is not supported by the fused graphs");

            var lmh = ResolveW(_lmHeadQW, _lmHeadF32);
            if (lmh.ptr == IntPtr.Zero || _finalNormW == null)
                return BatchedSpecDecline("folded lm_head/final norm missing");
            if (!_quantWeights.TryGetValue("token_embd.weight", out QuantizedWeight tokenQw)
                || tokenQw.Scale != 1.0f || !CanUseGgmlQuantizedGetRows(tokenQw.GgmlType) || tokenQw.DevicePreloadTooLarge)
                return BatchedSpecDecline("token embedding needs host-gathered rows");
            var emb = ResolveW(tokenQw, null);
            if (emb.ptr == IntPtr.Zero || emb.ne0 != Config.HiddenSize)
                return BatchedSpecDecline("token embedding shape mismatch");

            // The single-sequence speculative session leaves recurrent state in the
            // verify family's slices; settle it before reading the holders' copies.
            ExitSpecSession();
            RestorePrimaryCache();
            if (_fusedHolders == null)
                return BatchedSpecDecline("per-sequence caches were released");

            for (int i = 0; i < n; i++)
            {
                if (!_fusedHolders.TryGetValue(requestIds[i], out var h) || h.K == null)
                    return BatchedSpecDecline($"sequence {i} has no fused cache holder");
                if (h.CacheSeqLen != positions[i])
                    return BatchedSpecDecline($"sequence {i} holder is at {h.CacheSeqLen}, not {positions[i]}");
                // Never grown here: the window's padding rows alone would double a clone
                // sized to its prompt and budget (to the whole context on Qwen3.8 - 1 GB
                // per holder). The executor fits the window to every holder's free rows.
                if (positions[i] + window > h.KvCapacity)
                    return BatchedSpecDecline($"sequence {i} has {h.KvCapacity - positions[i]} free cache rows for a {window}-row window");
            }
            for (int i = 0; i < n * window; i++)
                if (tokens[i] < 0 || tokens[i] >= emb.ne1)
                    return BatchedSpecDecline("token id out of range");

            var holders = new Qwen35KvCacheHolder[n];
            for (int i = 0; i < n; i++)
                holders[i] = _fusedHolders[requestIds[i]];

            // Canonical slot order (ascending first-attention K pointer), as in the
            // batched decode: a stable set of holders keeps replaying one graph.
            int firstAttn = -1;
            for (int l = 0; l < Config.NumLayers; l++)
                if (!_isRecurrent[l]) { firstAttn = l; break; }
            var order = new int[n];
            var keys = new ulong[n];
            for (int i = 0; i < n; i++)
            {
                order[i] = i;
                keys[i] = (ulong)TensorComputePrimitives.GetStoragePointer(holders[i].K[firstAttn]).ToInt64();
            }
            Array.Sort(keys, order);

            int numLayers = Config.NumLayers;
            int attnLayers = 0, gdnLayers = 0;
            for (int l = 0; l < numLayers; l++)
                if (_isRecurrent[l]) gdnLayers++; else attnLayers++;
            int convDim = _convKernel - 1;
            int qkvDim = _headKDim * _numKHeads * 2 + _headVDim * _numVHeads;
            long convBlockBytes = (long)convDim * qkvDim * sizeof(float);

            var tokSorted = new int[n * window];
            var ropeSorted = new int[n * window];
            var posSorted = new int[n];
            var cacheSizes = new int[n];
            var kPtrs = new IntPtr[attnLayers * n];
            var vPtrs = new IntPtr[attnLayers * n];
            var convPtrs = new IntPtr[gdnLayers * n];
            var deltaPtrs = new IntPtr[gdnLayers * n];
            var hostAuthoritative = new bool[n];
            for (int i = 0; i < n; i++)
            {
                int src = order[i];
                var h = holders[src];
                posSorted[i] = positions[src];
                cacheSizes[i] = h.KvCapacity;
                for (int t = 0; t < window; t++)
                {
                    tokSorted[i * window + t] = tokens[src * window + t];
                    ropeSorted[i * window + t] = checked(positions[src] + t + h.RopeDelta);
                }
                if (h.ConvScratch == IntPtr.Zero)
                {
                    h.ConvScratch = AllocateConvScratch();
                    h.FdStateResident = false;
                }
                if (!h.FdStateResident)
                {
                    // The host ring is the truth: land it in the scratch layout and drop
                    // the stale device copies, so the bind below uploads these bytes.
                    // The residency flip waits for success (the conversion is
                    // idempotent), as in the batched decode.
                    ConvertHolderRingToScratch(h);
                    hostAuthoritative[i] = true;
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

            long featNeeded = (long)n * window * feat;
            if (_batchedSpecFeatures == null || _batchedSpecFeatures.LongLength < featNeeded)
                _batchedSpecFeatures = new float[featNeeded];
            var drawnSorted = new int[n * window];
            DType kvDt = _kvCacheDtype.ToDType();
            int status = GgmlBasicOps.Qwen35BatchedVerify(
                _fdLayers, numLayers, n, window,
                tokSorted, posSorted, ropeSorted,
                kPtrs, vPtrs, convPtrs, deltaPtrs, cacheSizes,
                Config.NumHeads, Config.NumKVHeads, Config.HeadDim,
                _ropeDimCount > 0 ? _ropeDimCount : Config.HeadDim, 2,
                FusedGraphKvCacheTypeId(kvDt),
                _convKernel, _headKDim, _headVDim, _numKHeads, _numVHeads,
                Config.Eps, Config.RopeBase, 1.0f / Config.RopeScale,
                Config.VocabSize,
                lmh.ptr, lmh.type, lmh.ne0, lmh.ne1, lmh.bytes,
                TensorComputePrimitives.GetStoragePointer(_finalNormW),
                emb.ptr, emb.type, emb.ne0, emb.ne1, emb.bytes,
                _dflash.TargetLayerIds, _batchedSpecFeatures, drawnSorted);
            if (status != 1)
            {
                string err = GgmlBasicOps.LastNativeError();
                if (status < 0)
                    throw new InvalidOperationException(
                        (string.IsNullOrWhiteSpace(err) ? "Qwen3.5 batched verify failed." : err) +
                        " K/V rows past each sequence's position may have been written; the affected requests were failed.");
                return BatchedSpecDecline("native declined: " +
                    (string.IsNullOrWhiteSpace(err) ? "native library returned no diagnostic" : err));
            }

            for (int i = 0; i < n; i++)
                if (hostAuthoritative[i])
                    holders[order[i]].FdStateResident = true;

            // Back to the caller's order: the native rows follow the sorted slots.
            long rowFloats = (long)window * feat;
            for (int i = 0; i < n; i++)
            {
                int dst = order[i];
                Array.Copy(drawnSorted, i * window, drawn, dst * window, window);
                Array.Copy(_batchedSpecFeatures, i * rowFloats, features, dst * rowFloats, rowFloats);
            }
            _batchedSpecHolders = holders;
            _batchedSpecOrder = order;
            _batchedSpecPositions = posSorted;
            _batchedSpecWindowPending = window;
            return true;
        }

        public void CommitBatchedSpecVerify(IReadOnlyList<string> requestIds, int[] positions, int[] kept)
        {
            var holders = _batchedSpecHolders;
            var order = _batchedSpecOrder;
            _batchedSpecHolders = null;
            int n = requestIds.Count;
            if (holders == null || holders.Length != n || kept.Length != n)
                throw new InvalidOperationException("No batched speculative verify is pending for these sequences.");
            var accepted = new int[n];
            for (int i = 0; i < n; i++)
            {
                int k = kept[order[i]];
                if (k < 1 || k > _batchedSpecWindowPending)
                    throw new ArgumentOutOfRangeException(nameof(kept));
                accepted[i] = k - 1;
            }
            int status = GgmlBasicOps.Qwen35BatchedVerifyCommit(accepted);
            if (status != 1)
            {
                string err = GgmlBasicOps.LastNativeError();
                throw new InvalidOperationException(
                    (string.IsNullOrWhiteSpace(err) ? "Qwen3.5 batched verify commit failed." : err) +
                    " Recurrent state may be partially advanced; the affected requests were failed.");
            }
            for (int i = 0; i < n; i++)
            {
                var h = holders[order[i]];
                h.CacheSeqLen = _batchedSpecPositions[i] + accepted[i] + 1;
                h.KvHostDirty = true;
                h.GdnHostDirty = true;
                h.FdStateResident = true;
                h.DeviceMirrored = true;
            }
            BatchedSpecVerifySteps++;
        }
    }
}
