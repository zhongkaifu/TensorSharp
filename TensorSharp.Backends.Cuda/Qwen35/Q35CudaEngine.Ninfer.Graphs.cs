// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda;

public sealed unsafe partial class Q35CudaEngine
{
    internal const string NinferDecodeGraphsEnvironment = "TS_Q35_NINFER_DECODE_GRAPHS";
    private readonly NinferGraphSetting _ninferDecodeGraphsSetting = SnapshotNinferGraphSetting(Environment.GetEnvironmentVariable(NinferDecodeGraphsEnvironment));
    private bool _ninferDecodeGraphsEnabled;
    private bool _ninferDecodeCapturePrepared;
    internal readonly record struct NinferDecodeGraphKey(int Lanes, int Window, bool Verify, int TapFirst,
        int AttentionChunk, int AttentionSplits, bool Verify8Attention, string ResidualNormIdentity = null,
        string Bf16MmaIdentity = null, string GateUpSwiGluIdentity = null);
    private struct NinferDecodeGraphEntry
    {
        public NinferDecodeGraphKey Key;
        public IntPtr Executable;
        public long LastUse;
    }
    private readonly NinferDecodeGraphEntry[] _ninferDecodeGraphs = new NinferDecodeGraphEntry[MaxVerifyGraphs];
    private readonly NinferDecodeGraphKey[] _ninferDecodeGraphRefusals = new NinferDecodeGraphKey[MaxVerifyGraphs];
    private int _ninferDecodeGraphCount, _ninferDecodeGraphRefusalCount, _ninferDecodeGraphRefusalNext;
    private long _ninferDecodeGraphClock;
    public int CapturedNinferDecodeGraphCount => _ninferDecodeGraphCount;
    public long NinferDecodeGraphCaptureCount { get; private set; }
    public long NinferDecodeGraphReplayCount { get; private set; }
    public long NinferDecodeGraphEvictionCount { get; private set; }
    private string NinferDecodeGraphsNamespaceSuffix => _ninferDecodeGraphsEnabled ? "|ninfer-decode-graphs=v1" : string.Empty;

    internal static bool NinferDecodeGraphsRequested(string value) => value switch
    {
        null or "0" => false,
        "1" => true,
        _ => throw new ArgumentException(NinferDecodeGraphsEnvironment + " accepts only 0 or 1.")
    };

    internal static bool NinferDecodeGraphShapeSupported(int groups, int rowsPerSequence, int lanes,
        bool verify, bool counts, bool embeddings, bool featureTaps, bool drafter)
        => groups == 1 && rowsPerSequence == 1 && lanes > 0 && lanes <= MaxSeqs
            && !verify && !counts && !embeddings && !featureTaps && !drafter;

    internal static bool NinferVerifyGraphShapeSupported(int groups, int window, int lanes, bool counts,
        bool embeddings, bool featureTaps, bool ggufDrafter)
        => groups == 1 && window > 0 && window <= 8 && lanes > 0 && lanes <= MaxSeqs
            && (long)window * lanes <= MaxVerifyRows && !counts && !embeddings && !featureTaps && !ggufDrafter;

    internal static NinferDecodeGraphKey PlanNinferDecodeGraphKey(int lanes, int window, bool verify,
        int tapFirst, int maxKeys, bool useVerify8, string residualNormIdentity = null, string bf16MmaIdentity = null,
        string gateUpSwiGluIdentity = null)
    {
        ValidateNinferBf16MmaGraphIdentity(lanes, window, verify, tapFirst, bf16MmaIdentity);
        ValidateNinferGateUpSwiGluGraphIdentity(lanes, window, verify, tapFirst, gateUpSwiGluIdentity);
        if (residualNormIdentity != null)
        {
            const string prefix = "|ninfer-residual-norm=" + Q35Kernels.NinferResidualNormPolicy + ":";
            if (lanes != 1 || window != 8 || !verify || tapFirst != 0
                || residualNormIdentity.Length != prefix.Length + 64 || !residualNormIdentity.StartsWith(prefix, StringComparison.Ordinal))
                throw new ArgumentException("A residual/norm graph needs its exact T8/B1 shape and loaded policy/image identity.");
            for (int i = prefix.Length; i < residualNormIdentity.Length; i++)
                if (!(residualNormIdentity[i] is >= '0' and <= '9' or >= 'a' and <= 'f'))
                    throw new ArgumentException("The residual/norm graph image identity must be normalized SHA256.");
        }
        if (useVerify8)
        {
            if (!CudaNinferPrefillAttentionModule.SupportsVerify8(window, lanes, verify, true, maxKeys))
                throw new ArgumentException("The captured Verify8 route requires its exact admitted shape/history.");
            // All candidate launch arguments are fixed or dynamic device inputs;
            // legacy split boundaries must neither alias this route nor recapture it.
            return new(lanes, window, verify, tapFirst, 0, 4, true, residualNormIdentity, bf16MmaIdentity, gateUpSwiGluIdentity);
        }
        var plan = Q35Kernels.GetSplitAttentionPlan(maxKeys);
        return new(lanes, window, verify, tapFirst, plan.Chunk, plan.Splits, false, residualNormIdentity, bf16MmaIdentity, gateUpSwiGluIdentity);
    }

    private void PrepareNinferDecodeCapture()
    {
        if (_ninferDecodeCapturePrepared) return;
        int maximumColumns = 1;
        foreach (var layer in _m.Ninfer.Layers)
            foreach (var d in new[] { layer.Input, layer.AlphaBeta, layer.Output, layer.GateUp, layer.Down })
                if (d?.Format == CudaNinferProjectionFormat.BFloat16)
                    maximumColumns = Math.Max(maximumColumns, d.Columns);
        // All future prefill/decode shapes fit this reservation. Captured cuBLAS
        // input addresses must survive a larger prefill after a captured decode.
        _ninfer.ReserveBf16Input(checked((long)RowCap * maximumColumns), Stream);
        if (_ninferAlphaBeta == IntPtr.Zero) _ninferAlphaBeta = Alloc(checked((long)RowCap * 96 * sizeof(float)));
        _ninferDecodeCapturePrepared = true;
    }

    private bool TryRunNinferDecodeGraph(int rows, int tapFirst, RowGroup[] groups, int[] maxKeys)
    {
        if (!_ninferDecodeGraphsEnabled || _m.Ninfer == null || groups.Length != 1) return false;
        var group = groups[0];
        bool eligible = group.Verify
            ? _ninferVerificationEnabled && NinferVerifyGraphShapeSupported(groups.Length, group.RowsPerSeq, group.Slots.Length,
                group.Counts != null, _embedSpans != null, _featureCount != 0, _d != null)
            : NinferDecodeGraphShapeSupported(groups.Length, group.RowsPerSeq, group.Slots.Length,
                false, group.Counts != null, _embedSpans != null, _featureCount != 0, _d != null);
        if (!eligible) return false;
        bool useVerify8 = _ninferAttention.UsesVerify8(group.RowsPerSeq, group.Slots.Length, group.Verify,
            group.Counts == null, _m.NHead, _m.NKvHead, maxKeys[0]);
        var key = PlanNinferDecodeGraphKey(group.Slots.Length, group.RowsPerSeq, group.Verify, tapFirst, maxKeys[0], useVerify8,
            group.PassMode == NinferLayerPass.FusedResidualNorm ? _ninferResidualNormSuffix : null,
            UsesNinferBf16Mma(groups, tapFirst) ? _ninferBf16MmaSuffix : null,
            UsesNinferGateUpSwiGlu(groups, tapFirst) ? _ninferGateUpSwiGluSuffix : null);
        for (int i = 0; i < _ninferDecodeGraphRefusalCount; i++)
            if (_ninferDecodeGraphRefusals[i] == key) return false;
        PrepareNinferDecodeCapture();
        int found = -1, free = -1, oldest = -1;
        for (int i = 0; i < _ninferDecodeGraphs.Length; i++)
        {
            var entry = _ninferDecodeGraphs[i];
            if (entry.Executable == IntPtr.Zero) free = i;
            else
            {
                if (entry.Key == key) found = i;
                if (oldest < 0 || entry.LastUse < _ninferDecodeGraphs[oldest].LastUse) oldest = i;
            }
        }
        if (found < 0)
        {
            found = free;
            if (found < 0)
            {
                found = oldest;
                // RunGroups' metadata synchronization drained all previous launches.
                CudaDriverApi.cuGraphExecDestroy(_ninferDecodeGraphs[found].Executable).ThrowOnError();
                _ninferDecodeGraphs[found] = default;
                _ninferDecodeGraphCount--;
                NinferDecodeGraphEvictionCount++;
            }
            IntPtr executable = IntPtr.Zero;
            try
            {
                try { executable = CudaDecodeCapture.Capture(_alloc, () => RunNinferLayers(rows, tapFirst, groups, maxKeys)); }
                catch (Exception error)
                {
                    _ninferDecodeGraphRefusals[_ninferDecodeGraphRefusalNext] = key;
                    _ninferDecodeGraphRefusalNext = (_ninferDecodeGraphRefusalNext + 1) % MaxVerifyGraphs;
                    _ninferDecodeGraphRefusalCount = Math.Min(MaxVerifyGraphs, _ninferDecodeGraphRefusalCount + 1);
                    Console.Error.WriteLine($"[q35-cuda] NInfer decode graph unavailable for {key}: {error.Message}; using kernel launches");
                    return false; // A failed capture executed none of the layer kernels.
                }
                _ninferDecodeGraphs[found] = new() { Key = key, Executable = executable };
                executable = IntPtr.Zero;
                _ninferDecodeGraphCount++;
                NinferDecodeGraphCaptureCount++;
                if (key.ResidualNormIdentity != null) NinferResidualNormCaptureCount++;
                if (key.Bf16MmaIdentity != null) NinferBf16MmaCaptureCount++;
                if (key.GateUpSwiGluIdentity != null) NinferGateUpSwiGluCaptureCount++;
            }
            finally { if (executable != IntPtr.Zero) CudaDriverApi.cuGraphExecDestroy(executable); }
            Console.Error.WriteLine($"[q35-cuda] NInfer layer graph captured: lanes={key.Lanes}, window={key.Window}, verify={key.Verify}, attention={key.AttentionChunk}x{key.AttentionSplits}, verify8Attention={key.Verify8Attention}, residualNorm={key.ResidualNormIdentity != null}, bf16Mma={key.Bf16MmaIdentity != null}, gateUpSwiGlu={key.GateUpSwiGluIdentity != null}");
        }
        _ninferDecodeGraphs[found].LastUse = ++_ninferDecodeGraphClock;
        // Failed launches are never retried: recurrent/KV writes may have begun.
        CudaDriverApi.cuGraphLaunch(_ninferDecodeGraphs[found].Executable, Stream).ThrowOnError();
        NinferDecodeGraphReplayCount++;
        if (key.ResidualNormIdentity != null) NinferResidualNormReplayCount++;
        if (key.Bf16MmaIdentity != null) NinferBf16MmaReplayCount++;
        if (key.GateUpSwiGluIdentity != null) NinferGateUpSwiGluReplayCount++;
        return true;
    }

    private void InvalidateNinferDecodeGraphs()
    {
        if (_ninferDecodeGraphCount > 0) { MakeCurrent(); Sync(); }
        foreach (var entry in _ninferDecodeGraphs)
            if (entry.Executable != IntPtr.Zero) CudaDriverApi.cuGraphExecDestroy(entry.Executable).ThrowOnError();
        Array.Clear(_ninferDecodeGraphs);
        _ninferDecodeGraphCount = _ninferDecodeGraphRefusalCount = _ninferDecodeGraphRefusalNext = 0;
    }
}
