// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    public sealed unsafe partial class Q35CudaEngine
    {
        // Explicit opt-ins while real-model quality/performance coverage is collected.
        // Caches are allocated once: publishing a captured native handle must not allocate
        // managed dictionary entries or linked-list nodes, or need an OOM rollback.
        private readonly bool _draftGraphsEnabled =
            Environment.GetEnvironmentVariable("TS_Q35_CUDA_GRAPHS") == "1";
        private readonly IntPtr[] _draftGraphs = new IntPtr[MaxSeqs + 1];
        private readonly bool[] _uncapturableDraftLanes = new bool[MaxSeqs + 1];
        private int _draftGraphCount;

        /// <summary>Successfully captured DFlash lane shapes (diagnostics; capture fallback
        /// must not be reported as having exercised CUDA graphs).</summary>
        public int CapturedDraftGraphCount => _draftGraphCount;

        private void RunDraftPass(int lanes)
        {
            if (lanes < 1 || lanes > MaxSeqs)
                throw new ArgumentOutOfRangeException(nameof(lanes));
            if (!_draftGraphsEnabled || _uncapturableDraftLanes[lanes])
            {
                DraftPass(lanes);
                return;
            }
            IntPtr graph = _draftGraphs[lanes];
            if (graph == IntPtr.Zero)
            {
                IntPtr owned = IntPtr.Zero;
                try
                {
                    try
                    {
                        // Slot rings and metadata are indirect device inputs, refreshed before
                        // this point; weights, scratch and table addresses have engine lifetime.
                        owned = CudaDecodeCapture.Capture(_alloc, () => DraftPass(lanes));
                    }
                    catch (Exception ex)
                    {
                        // A failed capture executes none of its kernels. This catch does not
                        // enclose publication, logging after publication, or a graph launch.
                        _uncapturableDraftLanes[lanes] = true;
                        Console.Error.WriteLine($"[q35-cuda] DFlash CUDA graph unavailable for {lanes} lanes: {ex.Message}; using kernel launches");
                        DraftPass(lanes);
                        return;
                    }
                    graph = owned;
                    _draftGraphs[lanes] = graph;
                    owned = IntPtr.Zero; // ownership transferred to the fixed cache slot
                    _draftGraphCount++;
                }
                finally
                {
                    if (owned != IntPtr.Zero)
                        CudaDriverApi.cuGraphExecDestroy(owned);
                }
                Console.Error.WriteLine($"[q35-cuda] DFlash CUDA graph captured: lanes={lanes}");
            }
            // Never retry a failed launch: device execution may already have begun.
            CudaDriverApi.cuGraphLaunch(graph, Stream).ThrowOnError();
        }

        private void FreeDraftGraphs()
        {
            for (int i = 1; i < _draftGraphs.Length; i++)
            {
                if (_draftGraphs[i] != IntPtr.Zero)
                    CudaDriverApi.cuGraphExecDestroy(_draftGraphs[i]);
                _draftGraphs[i] = IntPtr.Zero;
            }
            _draftGraphCount = 0;
        }

        private readonly bool _verifyGraphsEnabled =
            Environment.GetEnvironmentVariable("TS_Q35_VERIFY_GRAPHS") == "1";
        internal const int MaxVerifyGraphs = 32;
        private int _verifyGraphGeneration;
        private readonly record struct VerifyGraphKey(int Lanes, int Window, int AttentionChunk, int AttentionSplits, int Generation);
        private struct VerifyGraphEntry
        {
            public VerifyGraphKey Key;
            public IntPtr Executable;
            public long LastUse;
        }
        private readonly VerifyGraphEntry[] _verifyGraphs = new VerifyGraphEntry[MaxVerifyGraphs];
        private readonly VerifyGraphKey[] _uncapturableVerifyKeys = new VerifyGraphKey[MaxVerifyGraphs];
        private int _verifyGraphCount, _uncapturableVerifyCount, _uncapturableVerifyNext;
        private long _verifyGraphClock;

        /// <summary>Live Verify graph entries; capture fallback does not count as coverage.</summary>
        public int CapturedVerifyGraphCount => _verifyGraphCount;
        public long VerifyGraphCaptureCount { get; private set; }
        public long VerifyGraphReplayCount { get; private set; }
        public long VerifyGraphEvictionCount { get; private set; }

        internal static bool VerifyGraphShapeSupported(int lanes, int window, int heads, int kvHeads) =>
            lanes > 0 && lanes <= MaxSeqs && window > 0 && (long)lanes * window <= MaxVerifyRows &&
            heads > 0 && kvHeads > 0 && heads % kvHeads == 0 &&
            (long)window * (heads / kvHeads) <= Q35Kernels.AttnBlockRows;

        private void RunVerifyLayerPass(int n, int tapFirst, RowGroup[] groups, int[] maxKeys)
        {
            // Capture only one ordinary Verify group. Mixed forwards, prefill, padded groups,
            // embedding replacements and attention shapes on the other kernels keep their path.
            if (!_verifyGraphsEnabled || groups.Length != 1 || !groups[0].Verify ||
                groups[0].Counts != null || _embedSpans != null || tapFirst != 0 ||
                !VerifyGraphShapeSupported(groups[0].Slots.Length, groups[0].RowsPerSeq, _m.NHead, _m.NKvHead))
            {
                RunLayerKernels(n, tapFirst, groups, maxKeys);
                return;
            }
            var attention = Q35Kernels.GetSplitAttentionPlan(maxKeys[0]);
            var key = new VerifyGraphKey(groups[0].Slots.Length, groups[0].RowsPerSeq,
                attention.Chunk, attention.Splits, _verifyGraphGeneration);
            for (int i = 0; i < _uncapturableVerifyCount; i++)
                if (_uncapturableVerifyKeys[i] == key)
                {
                    RunLayerKernels(n, tapFirst, groups, maxKeys);
                    return;
                }
            int slot = -1, free = -1, oldest = -1;
            for (int i = 0; i < _verifyGraphs.Length; i++)
            {
                if (_verifyGraphs[i].Executable == IntPtr.Zero)
                    free = i;
                else
                {
                    if (_verifyGraphs[i].Key == key) slot = i;
                    if (oldest < 0 || _verifyGraphs[i].LastUse < _verifyGraphs[oldest].LastUse) oldest = i;
                }
            }
            if (slot < 0)
            {
                slot = free;
                if (slot < 0)
                {
                    slot = oldest;
                    // Evict BEFORE capture/instantiation, so even the transient peak is <=32
                    // Verify executables. RunGroups' metadata Sync drained every old graph;
                    // the only queued work now is embedding, which uses no graph executable.
                    // A destroy failure leaves the existing entry tracked and fails closed.
                    CudaDriverApi.cuGraphExecDestroy(_verifyGraphs[slot].Executable).ThrowOnError();
                    _verifyGraphs[slot] = default;
                    _verifyGraphCount--;
                    VerifyGraphEvictionCount++;
                }
                IntPtr owned = IntPtr.Zero;
                try
                {
                    try
                    {
                        // All inputs are engine-owned stable addresses or refreshed device
                        // tables. The layer loop contains no host mutation or host transfer.
                        owned = CudaDecodeCapture.Capture(_alloc, () => RunLayerKernels(n, tapFirst, groups, maxKeys));
                    }
                    catch (Exception ex)
                    {
                        _uncapturableVerifyKeys[_uncapturableVerifyNext] = key;
                        _uncapturableVerifyNext = (_uncapturableVerifyNext + 1) % MaxVerifyGraphs;
                        _uncapturableVerifyCount = Math.Min(MaxVerifyGraphs, _uncapturableVerifyCount + 1);
                        Console.Error.WriteLine($"[q35-cuda] Verify CUDA graph unavailable for {key}: {ex.Message}; using kernel launches");
                        RunLayerKernels(n, tapFirst, groups, maxKeys);
                        return;
                    }
                    // No allocation between successful capture and native ownership transfer.
                    _verifyGraphs[slot] = new VerifyGraphEntry { Key = key, Executable = owned };
                    owned = IntPtr.Zero;
                    _verifyGraphCount++;
                    VerifyGraphCaptureCount++;
                }
                finally
                {
                    if (owned != IntPtr.Zero)
                        CudaDriverApi.cuGraphExecDestroy(owned);
                }
                Console.Error.WriteLine($"[q35-cuda] Verify CUDA graph captured: lanes={key.Lanes}, window={key.Window}, attention={key.AttentionChunk}x{key.AttentionSplits}, generation={key.Generation}");
            }
            _verifyGraphs[slot].LastUse = ++_verifyGraphClock;
            // No fallback after a failed launch: some device writes may already have occurred.
            CudaDriverApi.cuGraphLaunch(_verifyGraphs[slot].Executable, Stream).ThrowOnError();
            VerifyGraphReplayCount++;
        }

        private void InvalidateVerifyGraphs()
        {
            ForgetNinferVerification();
            InvalidateNinferDecodeGraphs();
            if (_verifyGraphCount > 0)
            {
                MakeCurrent();
                Sync();
                FreeVerifyGraphs();
            }
            _uncapturableVerifyCount = 0;
            _uncapturableVerifyNext = 0;
            _verifyGraphGeneration++;
        }

        private void FreeVerifyGraphs()
        {
            for (int i = 0; i < _verifyGraphs.Length; i++)
            {
                if (_verifyGraphs[i].Executable != IntPtr.Zero)
                    CudaDriverApi.cuGraphExecDestroy(_verifyGraphs[i].Executable);
                _verifyGraphs[i] = default;
            }
            _verifyGraphCount = 0;
        }
    }
}
