// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    public partial class Qwen4ExpModel
    {
        public string BatchedFusedDecodeDeclineReason { get; private set; }
        internal long ArenaBatchedDecodeSteps { get; private set; }
        private float[] _arenaLogitsStaging;

        public bool CanBatchDecode(string requestId, int position)
        {
            if (_fusedHolders == null || requestId == null || !_fusedHolders.TryGetValue(requestId, out var h))
                return false;
            // The dictionary snapshot of the currently checked-out holder can
            // predate its prefill and growth. Read its live fields until check-in.
            bool active = string.Equals(requestId, _activeFusedKey, StringComparison.Ordinal);
            return position >= 0 && position == (active ? _cacheSeqLen : h.CacheSeqLen)
                && position < (active ? _kvCacheCapacity : h.KvCapacity)
                && !(active ? _specStateFailed : h.SpecStateFailed)
                && (active ? _deviceStateAuthoritative : h.DeviceStateSeeded);
        }

        public bool TryForwardBatchedFusedDecode(IReadOnlyList<string> requestIds,
            int[] tokens, int[] positions, float[][] outLogits)
            => ForwardArenaBatchedDecode(requestIds, tokens, positions, outLogits, null);

        public bool TryForwardBatchedFusedDecodeSampled(IReadOnlyList<string> requestIds,
            int[] tokens, int[] positions, int[] outNextTokens)
            => ForwardArenaBatchedDecode(requestIds, tokens, positions, null, outNextTokens);

        private bool ArenaDecline(string reason)
        {
            BatchedFusedDecodeDeclineReason = reason;
            return false;
        }

        private bool PrepareArenaHolder(Qwen4ExpKvCacheHolder h)
        {
            var previous = SnapshotActiveCache();
            LoadCacheHolder(h);
            try
            {
                if (!EnsureAttnArgs() || !EnsureGdnArgs() || !EnsureQsaArgs()
                    || (_pleHeads > 0 && !EnsurePleArgs())) return false;
                h.AttnArgs = _attnArgs; h.GdnArgs = _gdnArgs; h.PleArgs = _pleArgs; h.QsaArgs = _qsaArgs;
                h.GdnConvStateT = _gdnConvStateT; h.GdnStateT = _gdnStateT;
                return true;
            }
            finally { LoadCacheHolder(previous); }
        }

        private static IntPtr PinnedArrayPointer(Array array)
            => array == null || array.Length == 0 ? IntPtr.Zero : Marshal.UnsafeAddrOfPinnedArrayElement(array, 0);

        private unsafe bool ForwardArenaBatchedDecode(IReadOnlyList<string> requestIds,
            int[] tokens, int[] positions, float[][] outLogits, int[] outNextTokens)
        {
            BatchedFusedDecodeDeclineReason = null;
            if (Environment.GetEnvironmentVariable("TS_Q4E_DISABLE_ARENA_DECODE") == "1")
                return ArenaDecline("disabled via TS_Q4E_DISABLE_ARENA_DECODE=1");
            if (!IsGgmlBackend || _tokenGraphUnsupported)
                return ArenaDecline("the complete GGML token-span path is unavailable");
            if (IsTensorParallel || LayerSplitDegree > 1)
                return ArenaDecline("arena decode currently requires one device");
            if (_specForwardActive || _pendingMRoPEPositions != null)
                return ArenaDecline("speculative or media prefill is active");
            int n = requestIds?.Count ?? 0;
            if (n < 2 || tokens == null || positions == null || tokens.Length != n || positions.Length != n
                || (outLogits == null && outNextTokens == null)
                || (outLogits != null && outLogits.Length != n)
                || (outNextTokens != null && outNextTokens.Length != n))
                return ArenaDecline("at least two sequences and matching input/output arrays are required");
            if (_kvCacheDtype.ToDType() != DType.Float16)
                return ArenaDecline("arena decode currently requires F16 KV caches");
            // Check in the active holder before reading snapshots. In particular
            // a request's first prefill can leave its dictionary count at zero.
            RestorePrimaryCache();
            if (_fusedHolders == null) return ArenaDecline("no request holders have been initialized");
            var holders = new Qwen4ExpKvCacheHolder[n];
            var unique = new HashSet<string>(StringComparer.Ordinal);
            for (int i = 0; i < n; ++i)
            {
                string id = requestIds[i];
                if (id == null || !unique.Add(id) || !CanBatchDecode(id, positions[i]))
                    return ArenaDecline($"sequence {i} is duplicated, unbound, stale, or needs cache growth");
                holders[i] = _fusedHolders[id];
                if (tokens[i] < 0 || tokens[i] >= Config.VocabSize)
                    return ArenaDecline($"sequence {i} has an invalid token");
                if (HasQsa && holders[i].QsaPositionCount != positions[i])
                    return ArenaDecline($"sequence {i} has incomplete QSA coordinate history");
            }
            if (!EnsureFfnArgs() || !EnsureHeadArgs()
                || !TryResolveQuant("token_embd.weight", out IntPtr embedding, out int embeddingType, out long embeddingBytes))
                return ArenaDecline("model weight descriptors are unavailable");
            foreach (var h in holders)
                if (!PrepareArenaHolder(h)) return ArenaDecline("request state descriptors are unavailable");
            _layerKinds ??= GC.AllocateArray<byte>(Config.NumLayers, pinned: true);
            int attentionLayers = 0, recurrentLayers = 0, firstAttention = -1;
            for (int l = 0; l < Config.NumLayers; ++l)
            {
                _layerKinds[l] = _isRecurrent[l] ? (byte)1 : (byte)0;
                if (_isRecurrent[l]) ++recurrentLayers;
                else { ++attentionLayers; if (firstAttention < 0) firstAttention = l; }
            }
            if (firstAttention < 0) return ArenaDecline("no attention layers");
            var order = new int[n]; var keys = new ulong[n];
            for (int i = 0; i < n; ++i)
            {
                order[i] = i;
                keys[i] = unchecked((ulong)TensorComputePrimitives.GetStoragePointer(holders[i].K[firstAttention]).ToInt64());
            }
            Array.Sort(keys, order);
            var sortedTokens = new int[n]; var sortedPositions = new int[n]; var ropePositions = new int[n];
            var capacities = new int[n]; var qsaCounts = new int[n];
            var k = new IntPtr[attentionLayers * n]; var v = new IntPtr[attentionLayers * n];
            var conv = new IntPtr[recurrentLayers * n]; var ssm = new IntPtr[recurrentLayers * n]; var pleState = new IntPtr[n];
            var attnArgs = new IntPtr[n]; var gdnArgs = new IntPtr[n]; var pleArgs = new IntPtr[n];
            var qsaArgs = new IntPtr[n]; var qsaPositions = new IntPtr[n];
            float[] pleEmbedding = _pleHeads > 0 ? new float[checked(n * Config.HiddenSize)] : null;
            for (int i = 0; i < n; ++i)
            {
                int original = order[i]; var h = holders[original];
                sortedTokens[i] = tokens[original]; sortedPositions[i] = positions[original];
                ropePositions[i] = checked((int)Math.Max(0L, (long)positions[original] - h.MropeCacheGap));
                capacities[i] = h.KvCapacity;
                attnArgs[i] = PinnedArrayPointer(h.AttnArgs); gdnArgs[i] = PinnedArrayPointer(h.GdnArgs);
                pleArgs[i] = PinnedArrayPointer(h.PleArgs); qsaArgs[i] = PinnedArrayPointer(h.QsaArgs);
                pleState[i] = PinnedArrayPointer(h.PleConvState);
                for (int l = 0, al = 0, gl = 0; l < Config.NumLayers; ++l)
                {
                    if (_isRecurrent[l])
                    {
                        conv[gl * n + i] = (IntPtr)GetFloatPtr(h.GdnConvStateT[l]);
                        ssm[gl++ * n + i] = (IntPtr)GetFloatPtr(h.GdnStateT[l]);
                    }
                    else
                    {
                        k[al * n + i] = TensorComputePrimitives.GetStoragePointer(h.K[l]);
                        v[al++ * n + i] = TensorComputePrimitives.GetStoragePointer(h.V[l]);
                    }
                }
                if (HasQsa)
                {
                    // Only an uncommitted tail row is written. Solo retry writes
                    // this same coordinate; live history/count stay unchanged.
                    int required = checked(3 * h.KvCapacity);
                    if (h.QsaPositions == null || h.QsaPositions.Length < required)
                    {
                        var grown = GC.AllocateArray<int>(required, pinned: true);
                        if (h.QsaPositions != null) Array.Copy(h.QsaPositions, grown, checked(3 * h.QsaPositionCount));
                        h.QsaPositions = grown;
                    }
                    WriteQsaPositions(h.QsaPositions, positions[original], 1, ropePositions[i], null);
                    qsaPositions[i] = PinnedArrayPointer(h.QsaPositions); qsaCounts[i] = positions[original] + 1;
                }
                if (pleEmbedding != null)
                {
                    int[] rows = ComputePleRows(new[] { tokens[original] }, positions[original], h.PleHistory, h.PleNextPos);
                    fixed (float* pe = pleEmbedding) GatherPleRowsRaw(pe + (long)i * Config.HiddenSize, rows, 1);
                }
            }
            // Unsupported GET_ROWS types and very large embedding tables take
            // the same N-row host dequant path as solo decode.
            Tensor embeddingRows = null;
            if (_quantWeights.TryGetValue("token_embd.weight", out var qw)
                && (!CanUseGgmlQuantizedGetRows(qw.GgmlType) || qw.DevicePreloadTooLarge))
                embeddingRows = Embedding(sortedTokens);
            using (embeddingRows)
            {
                int vocab = Config.VocabSize;
                int needed = checked(n * vocab);
                if (_arenaLogitsStaging == null || _arenaLogitsStaging.Length < needed)
                    _arenaLogitsStaging = new float[needed];
                // Reserve every managed destination before execution advances
                // device state. A checkpoint's short history list may have no
                // spare element even though normal prefill histories do.
                foreach (var h in holders)
                {
                    if (outLogits != null) h.Logits ??= new float[vocab];
                    if (_pleHeads > 0) h.PleHistory.EnsureCapacity(checked(h.PleHistory.Count + 1));
                }
                int[] sampled = outNextTokens != null ? new int[n] : null;
                int status;
                fixed (float* lp = _arenaLogitsStaging)
                fixed (float* pe = pleEmbedding)
                fixed (int* sp = sampled)
                    status = GgmlBasicOps.Qwen4ExpArenaDecodeBatchedStatus(
                        PinnedArrayPointer(_ffnArgs), gdnArgs[0], attnArgs[0], PinnedArrayPointer(_headArgs), pleArgs[0],
                        PinnedArrayPointer(_layerKinds), Config.NumLayers, _pleHeads > 0 ? _pleLayerIndex : -1, n,
                        sortedTokens, sortedPositions, ropePositions, capacities, k, v, conv, ssm, pleState,
                        attnArgs, gdnArgs, pleArgs, HasQsa ? qsaArgs : null,
                        HasQsa ? qsaPositions : null, HasQsa ? qsaCounts : null,
                        Config.HiddenSize, _hc, _hcLowRank, Config.HeadDim, Config.NumHeads, Config.NumKVHeads, _ropeDimCount,
                        Config.RopeBase, 1f / Config.RopeScale, _attnScale,
                        _headKDim, _headVDim, _numKHeads, _numVHeads, _convKernel,
                        _numExperts, _numExpertsUsed, _expertFf, _sharedFf, Config.Eps, 1,
                        embedding, embeddingType, Config.HiddenSize, Config.VocabSize, embeddingBytes,
                        embeddingRows == null ? IntPtr.Zero : (IntPtr)GetFloatPtr(embeddingRows), (IntPtr)pe,
                        (IntPtr)lp, (IntPtr)sp, outLogits != null, DeviceForLayer(0));
                if (status != 1)
                {
                    string error = GgmlBasicOps.LastNativeError();
                    if (status < 0)
                    {
                        foreach (var h in holders) h.SpecStateFailed = true;
                        throw new InvalidOperationException("qwen4exp arena execution failed; affected requests must be reset because recurrent state may have advanced. " + error);
                    }
                    return ArenaDecline("native declined: " + error);
                }
                try
                {
                    for (int i = 0; i < n; ++i)
                    {
                        int original = order[i]; var h = holders[original];
                        if (outLogits != null)
                        {
                            Array.Copy(_arenaLogitsStaging, i * vocab, h.Logits, 0, vocab);
                            outLogits[original] = h.Logits;
                        }
                        if (outNextTokens != null) outNextTokens[original] = sampled[i];
                        if (_pleHeads > 0)
                        {
                            CommitPleHistory(h.PleHistory, new[] { tokens[original] }, positions[original], h.PleNextPos);
                            h.PleNextPos = positions[original] + 1;
                        }
                        h.CacheSeqLen = positions[original] + 1;
                        if (HasQsa) h.QsaPositionCount = h.CacheSeqLen;
                        h.DeviceStateSeeded = true; h.KvHostStale = true; h.DeviceMirrored = KeepsDeviceKvMirrors;
                    }
                }
                catch
                {
                    // Native accepted the whole batch. Partial managed
                    // publication must fence every holder, never permit a
                    // retry at a head its recurrence has already passed.
                    foreach (var h in holders) h.SpecStateFailed = true;
                    throw;
                }
            }
            ++ArenaBatchedDecodeSteps;
            return true;
        }
    }
}
