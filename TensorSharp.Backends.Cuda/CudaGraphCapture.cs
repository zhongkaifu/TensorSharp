using System;
using System.Collections.Generic;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// Thrown when an operation that cannot be captured (a stream synchronize,
    /// i.e. a host read of device data) runs while a CUDA graph capture is
    /// active. The capture site catches this, aborts the capture, blacklists
    /// the shape and re-runs the region plainly (no kernel executed during the
    /// failed capture, so a re-run is safe).
    /// </summary>
    public sealed class CudaGraphCaptureAbortedException : Exception
    {
        public CudaGraphCaptureAbortedException(string message) : base(message) { }
    }

    /// <summary>
    /// Process-wide hooks for the single active CUDA-graph stream capture.
    /// While a capture is active on an allocator's stream:
    ///  - device blocks that the pool would cuMemFree are quarantined instead
    ///    (the captured graph references them),
    ///  - every block rented or pooled is tracked so the capture owner can
    ///    remove ("steal") the free ones from the pool when the capture ends,
    ///  - host mirror buffers of storages destroyed mid-capture are donated to
    ///    the capture owner (a captured HtoD copy may reference them),
    ///  - a stream synchronize throws <see cref="CudaGraphCaptureAbortedException"/>.
    /// </summary>
    internal static class CudaGraphCapture
    {
        private static readonly object Sync = new object();
        private static CaptureContext active;

        internal sealed class CaptureContext
        {
            public CudaAllocator Allocator;
            public readonly HashSet<IntPtr> TrackedBlocks = new HashSet<IntPtr>();
            public readonly Dictionary<IntPtr, long> TrackedSizes = new Dictionary<IntPtr, long>();
            public readonly List<(IntPtr ptr, long bytes)> QuarantinedBlocks = new List<(IntPtr, long)>();
            public readonly List<IntPtr> DonatedHostBuffers = new List<IntPtr>();
        }

        internal static CaptureContext Begin(CudaAllocator allocator)
        {
            lock (Sync)
            {
                if (active != null)
                    return null;
                active = new CaptureContext { Allocator = allocator };
                return active;
            }
        }

        internal static void End(CaptureContext context)
        {
            lock (Sync)
            {
                if (ReferenceEquals(active, context))
                    active = null;
            }
        }

        internal static bool IsCapturing(CudaAllocator allocator)
        {
            CaptureContext ctx = active;
            return ctx != null && ReferenceEquals(ctx.Allocator, allocator);
        }

        internal static void OnRent(CudaAllocator allocator, IntPtr ptr, long bytes)
        {
            CaptureContext ctx = active;
            if (ctx == null || !ReferenceEquals(ctx.Allocator, allocator) || ptr == IntPtr.Zero)
                return;
            lock (Sync)
            {
                if (ctx.TrackedBlocks.Add(ptr))
                    ctx.TrackedSizes[ptr] = bytes;
            }
        }

        /// <summary>
        /// Called for every pool return while this allocator is capturing. Returns
        /// true when the block was quarantined (the caller must NOT free or pool
        /// it); false when normal pooling should proceed (the block was tracked so
        /// it can be stolen from the pool at capture end).
        /// </summary>
        internal static bool InterceptReturn(CudaAllocator allocator, IntPtr ptr, long bytes, bool wouldPool)
        {
            CaptureContext ctx = active;
            if (ctx == null || !ReferenceEquals(ctx.Allocator, allocator) || ptr == IntPtr.Zero)
                return false;
            lock (Sync)
            {
                if (wouldPool)
                {
                    if (ctx.TrackedBlocks.Add(ptr))
                        ctx.TrackedSizes[ptr] = bytes;
                    return false;
                }

                // The pool would cuMemFree this block, but the captured graph
                // still references it: quarantine instead.
                ctx.TrackedBlocks.Remove(ptr);
                ctx.TrackedSizes.Remove(ptr);
                ctx.QuarantinedBlocks.Add((ptr, bytes));
                return true;
            }
        }

        internal static void OnHostBufferOrphaned(CudaAllocator allocator, IntPtr hostBuffer, out bool donated)
        {
            donated = false;
            CaptureContext ctx = active;
            if (ctx == null || !ReferenceEquals(ctx.Allocator, allocator) || hostBuffer == IntPtr.Zero)
                return;
            lock (Sync)
            {
                ctx.DonatedHostBuffers.Add(hostBuffer);
                donated = true;
            }
        }

        internal static readonly bool DiagnosticLog =
            string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_PREFILL_GRAPH_LOG"), "1", StringComparison.Ordinal);

        /// <summary>Diagnostic: a pageable host->device upload is about to run on a
        /// capturing stream (it will fail with CUDA error 900). Logs the culprit's
        /// call site so the host-dirtying code path can be fixed.</summary>
        internal static void OnCapturedHostUpload(CudaAllocator allocator, long bytes)
        {
            if (!DiagnosticLog || !IsCapturing(allocator))
                return;
            Console.WriteLine($"[cuda-graph] HtoD upload of {bytes} B inside capture at:");
            Console.WriteLine(Environment.StackTrace);
        }

        internal static void OnStreamSynchronize(IntPtr streamHandle)
        {
            CaptureContext ctx = active;
            if (ctx != null && ctx.Allocator.Stream.Handle == streamHandle)
                throw new CudaGraphCaptureAbortedException(
                    "Stream synchronization (host read of device data) is not permitted while a CUDA graph capture is active."
                    + (DiagnosticLog ? "\n" + Environment.StackTrace : string.Empty));
        }
    }

    /// <summary>
    /// Per-token dynamic parameters for CUDA-graph DECODE replay. A captured
    /// graph bakes every scalar kernel argument, but the decode step's
    /// position-dependent values (attention length, KV write position, GDN conv
    /// ring index, RoPE position) change every token. Kernels on the decode path
    /// therefore re-read those values from a small device block; the graph's
    /// first node is a captured memcpy from this object's PINNED host buffer to
    /// that device block, so a replay only needs the host ints refreshed before
    /// cuGraphLaunch (the memcpy node re-reads pinned memory on every launch).
    /// </summary>
    public sealed class CudaDecodeDynParams : IDisposable
    {
        // Slot layout must match the TS_DYN_* defines in tensorsharp_kernels.cu:
        // [0]=attend_len, [1]=kv write pos, [2]=conv ring write idx, [3]=rope pos.
        private const int SlotCount = 4;
        private const int ByteCount = SlotCount * sizeof(int);

        private readonly CudaAllocator allocator;
        private IntPtr devicePtr;
        private IntPtr hostPtr;

        // Capture launchers are synchronous on one host thread, but unrelated
        // allocators may launch ordinary work concurrently. Keep the ambient
        // source thread-local and retain the owning instance so a launcher can
        // reject a dynamic pointer that belongs to another allocator/stream.
        [ThreadStatic]
        private static CudaDecodeDynParams activeInstance;

        [ThreadStatic]
        private static int captureMaxAttendLen;

        /// <summary>Largest attention length the graph being captured stays
        /// valid for (0 = unlimited). Set by launchers that bake a
        /// length-dependent launch configuration (shared memory, grid).</summary>
        public static int CaptureMaxAttendLen => captureMaxAttendLen;

        /// <summary>Returns the ambient dynamic-parameter pointer only when it
        /// belongs to the allocator issuing this launch. Otherwise the caller
        /// must use its ordinary scalar kernel arguments.</summary>
        internal static IntPtr GetActiveDevicePtr(CudaAllocator launchAllocator)
        {
            CudaDecodeDynParams active = activeInstance;
            return active != null &&
                ReferenceEquals(active.allocator, launchAllocator) &&
                active.devicePtr != IntPtr.Zero
                    ? active.devicePtr
                    : IntPtr.Zero;
        }

        public CudaDecodeDynParams(IAllocator allocator)
        {
            this.allocator = allocator as CudaAllocator;
            if (this.allocator == null)
                return;
            this.allocator.Context.MakeCurrent();
            if (CudaDriverApi.cuMemAlloc(out devicePtr, (UIntPtr)ByteCount) != 0)
            {
                devicePtr = IntPtr.Zero;
                return;
            }
            if (CudaDriverApi.cuMemHostAlloc(out hostPtr, (UIntPtr)ByteCount, 0) != 0)
            {
                hostPtr = IntPtr.Zero;
                CudaDriverApi.cuMemFree(devicePtr);
                devicePtr = IntPtr.Zero;
            }
        }

        public bool IsValid => devicePtr != IntPtr.Zero && hostPtr != IntPtr.Zero;

        internal IntPtr DevicePtr => devicePtr;

        /// <summary>Writes this token's values into the pinned host block. Must
        /// run before Replay (and before EndCaptureAndLaunch when capturing).</summary>
        public unsafe void Write(int attendLen, int kvWritePos, int convWriteIdx, int ropePos)
        {
            int* p = (int*)hostPtr;
            p[0] = attendLen;
            p[1] = kvWritePos;
            p[2] = convWriteIdx;
            p[3] = ropePos;
        }

        /// <summary>Enqueues the pinned-host -> device upload on the allocator
        /// stream. Called once INSIDE the capture region so the copy becomes the
        /// graph's leading node.</summary>
        public void EnqueueUpload()
        {
            allocator.Context.MakeCurrent();
            CudaDriverApi.cuMemcpyHtoDAsync(devicePtr, hostPtr, (UIntPtr)ByteCount, allocator.Stream.Handle)
                .ThrowOnError();
        }

        /// <summary>Makes this block the ambient dyn-parameter source for the
        /// decode-path kernel launchers (capture scope only; clear with
        /// <see cref="Deactivate"/> in a finally).</summary>
        public void Activate()
        {
            activeInstance = this;
            captureMaxAttendLen = 0;
        }

        public static void Deactivate()
        {
            activeInstance = null;
        }

        internal static void LimitCaptureAttendLen(int limit)
        {
            if (activeInstance != null &&
                limit > 0 &&
                (captureMaxAttendLen == 0 || limit < captureMaxAttendLen))
            {
                captureMaxAttendLen = limit;
            }
        }

        public void Dispose()
        {
            if (devicePtr != IntPtr.Zero)
            {
                if (ReferenceEquals(activeInstance, this))
                    activeInstance = null;
                allocator?.Context.MakeCurrent();
                CudaDriverApi.cuMemFree(devicePtr);
                devicePtr = IntPtr.Zero;
            }
            if (hostPtr != IntPtr.Zero)
            {
                CudaDriverApi.cuMemFreeHost(hostPtr);
                hostPtr = IntPtr.Zero;
            }
        }
    }

    /// <summary>
    /// Cache of instantiated CUDA graphs for the direct-CUDA per-op prefill
    /// layer loop. A graph replays the exact kernel sequence with the exact
    /// pointers/parameters that were captured, so entries are keyed on every
    /// value that gets baked into kernel launches (sequence length, start
    /// position, KV-cache buffer identity, recurrent conv ring phase) and the
    /// graph owns the pool blocks its kernels reference. Capture happens on the
    /// SECOND run of a key (the first plain run grows every scratch buffer and
    /// lazily-created cache, so nothing allocates or synchronizes mid-capture);
    /// later runs replay in a single cuGraphLaunch, collapsing the ~900 per-op
    /// dispatches of a prefill chunk.
    /// </summary>
    public sealed class CudaPrefillGraphCache : IDisposable
    {
        /// <summary>TS_CUDA_PREFILL_GRAPH=0 disables capture/replay entirely.</summary>
        public static readonly bool Enabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_PREFILL_GRAPH"), "0", StringComparison.Ordinal);

        /// <summary>TS_CUDA_DECODE_GRAPH=0 disables the per-token decode
        /// capture/replay (the seqLen==1 layer loop with dynamic parameters);
        /// prefill graphs are governed separately by TS_CUDA_PREFILL_GRAPH.</summary>
        public static readonly bool DecodeEnabled =
            !string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_DECODE_GRAPH"), "0", StringComparison.Ordinal);

        private static readonly bool Log =
            string.Equals(Environment.GetEnvironmentVariable("TS_CUDA_PREFILL_GRAPH_LOG"), "1", StringComparison.Ordinal);

        private static readonly int Capacity = ReadCapacity();

        private static int ReadCapacity()
        {
            string s = Environment.GetEnvironmentVariable("TS_CUDA_PREFILL_GRAPH_MAX");
            return !string.IsNullOrEmpty(s) && int.TryParse(s, out int v) && v > 0 ? v : 4;
        }

        private sealed class Entry
        {
            public IntPtr Exec;
            public Tensor InputHidden;
            public List<Tensor> KeepAlive;
            public List<(IntPtr ptr, long bytes)> OwnedBlocks;
            public List<IntPtr> OwnedHostBuffers;
            public long LastUse;
            /// <summary>Largest attention length this graph replays correctly
            /// with (0 = unlimited). Decode graphs that baked a length-bounded
            /// launch configuration expire past it and are re-captured.</summary>
            public int MaxAttendLen;
        }

        private readonly CudaAllocator allocator;
        private readonly Dictionary<string, Entry> entries = new Dictionary<string, Entry>(StringComparer.Ordinal);
        private readonly HashSet<string> blacklist = new HashSet<string>(StringComparer.Ordinal);
        private readonly HashSet<string> seenOnce = new HashSet<string>(StringComparer.Ordinal);
        private CudaGraphCapture.CaptureContext captureContext;
        private string captureKey;
        private long useCounter;
        private bool disposed;

        public long ReplayCount { get; private set; }
        public long CaptureCount { get; private set; }

        public CudaPrefillGraphCache(IAllocator allocator)
        {
            this.allocator = allocator as CudaAllocator;
        }

        public bool IsUsable => Enabled && allocator != null && !disposed;

        /// <summary>
        /// When a cached graph exists for <paramref name="key"/>, returns (a
        /// CopyRef of) the pinned hidden-state tensor the captured loop ran on.
        /// The caller copies the fresh embedding output into it and calls
        /// <see cref="Replay"/>.
        /// </summary>
        public bool TryGetReplayInput(string key, out Tensor pinnedHidden)
            => TryGetReplayInput(key, 0, out pinnedHidden);

        /// <summary>
        /// Validity-checked variant for decode graphs: an entry whose baked
        /// launch configuration only covers attention lengths up to
        /// MaxAttendLen is disposed when <paramref name="attendLen"/> exceeds
        /// it. The key stays in the seen-once set, so the caller's
        /// ShouldCapture immediately re-captures with the longer-length route.
        /// </summary>
        public bool TryGetReplayInput(string key, int attendLen, out Tensor pinnedHidden)
        {
            pinnedHidden = null;
            if (!IsUsable || captureContext != null || !entries.TryGetValue(key, out Entry entry))
                return false;
            if (entry.MaxAttendLen > 0 && attendLen > entry.MaxAttendLen)
            {
                DisposeEntry(key);
                return false;
            }
            entry.LastUse = ++useCounter;
            pinnedHidden = entry.InputHidden.CopyRef();
            return true;
        }

        public void Replay(string key)
        {
            Entry entry = entries[key];
            allocator.Context.MakeCurrent();
            CudaDriverApi.cuGraphLaunch(entry.Exec, allocator.Stream.Handle).ThrowOnError();
            // The graph rewrote the entry's device tensors without going through
            // the C# launchers, so their host mirrors are stale now; flag it or a
            // later host read would trust a clean mirror. (The model does the
            // same for KV caches / recurrent state the graph writes in place.)
            MarkDeviceModified(entry.InputHidden);
            foreach (Tensor t in entry.KeepAlive)
                MarkDeviceModified(t);
            ReplayCount++;
            if (Log)
                Console.WriteLine($"[cuda-graph] replay {key} (total {ReplayCount})");
        }

        private static void MarkDeviceModified(Tensor t)
        {
            if (t?.Storage is CudaStorage storage)
                storage.MarkDeviceModified();
        }

        /// <summary>
        /// Second-use policy: the first sighting of a key runs plainly (growing
        /// scratch buffers and lazy caches), the second captures.
        /// </summary>
        public bool ShouldCapture(string key)
        {
            if (!IsUsable || captureContext != null || blacklist.Contains(key) || entries.ContainsKey(key))
                return false;
            if (seenOnce.Add(key))
                return false;
            return true;
        }

        public bool BeginCapture(string key)
        {
            if (!IsUsable || captureContext != null)
                return false;
            CudaGraphCapture.CaptureContext ctx = CudaGraphCapture.Begin(allocator);
            if (ctx == null)
                return false;
            allocator.Context.MakeCurrent();
            // Relaxed mode so a pool miss inside the captured loop may still
            // cuMemAlloc (the rent is tracked and the block quarantined with the
            // graph); thread-local capture rejects it with CUDA error 900.
            int rc = CudaDriverApi.cuStreamBeginCapture(
                allocator.Stream.Handle, CudaDriverApi.CU_STREAM_CAPTURE_MODE_RELAXED);
            if (rc != 0)
            {
                CudaGraphCapture.End(ctx);
                blacklist.Add(key);
                return false;
            }
            captureContext = ctx;
            captureKey = key;
            return true;
        }

        /// <summary>
        /// Ends the active capture, instantiates and caches the graph, and
        /// launches it once (this forward's execution — nothing ran during
        /// capture). Returns false when instantiation failed; the caller must
        /// then re-run the region plainly.
        /// </summary>
        public bool EndCaptureAndLaunch(string key, Tensor pinnedHidden, IEnumerable<Tensor> keepAlive)
            => EndCaptureAndLaunch(key, pinnedHidden, keepAlive, 0);

        public bool EndCaptureAndLaunch(string key, Tensor pinnedHidden, IEnumerable<Tensor> keepAlive, int maxAttendLen)
        {
            if (captureContext == null || !string.Equals(captureKey, key, StringComparison.Ordinal))
                return false;

            CudaGraphCapture.CaptureContext ctx = captureContext;
            allocator.Context.MakeCurrent();
            int rc = CudaDriverApi.cuStreamEndCapture(allocator.Stream.Handle, out IntPtr graph);
            CudaGraphCapture.End(ctx);
            captureContext = null;
            captureKey = null;

            if (rc != 0 || graph == IntPtr.Zero)
            {
                ReleaseContextBlocks(ctx);
                blacklist.Add(key);
                if (Log)
                    Console.WriteLine($"[cuda-graph] end-capture failed rc={rc} for {key}");
                return false;
            }

            rc = CudaDriverApi.cuGraphInstantiateWithFlags(out IntPtr exec, graph, 0);
            CudaDriverApi.cuGraphDestroy(graph);
            if (rc != 0 || exec == IntPtr.Zero)
            {
                ReleaseContextBlocks(ctx);
                blacklist.Add(key);
                if (Log)
                    Console.WriteLine($"[cuda-graph] instantiate failed rc={rc} for {key}");
                return false;
            }

            // Take ownership of every pool block the captured kernels used: the
            // free ones are stolen out of the pool so nothing else can rent them
            // while this graph can replay.
            var owned = new List<(IntPtr, long)>(ctx.QuarantinedBlocks);
            foreach (IntPtr ptr in ctx.TrackedBlocks)
            {
                long bytes = ctx.TrackedSizes[ptr];
                if (allocator.TryStealPooledBlock(ptr, bytes))
                    owned.Add((ptr, bytes));
            }

            if (entries.Count >= Capacity)
                EvictLru();

            var keep = new List<Tensor>();
            keep.Add(pinnedHidden.CopyRef());
            if (keepAlive != null)
            {
                foreach (Tensor t in keepAlive)
                    if (t != null)
                        keep.Add(t.CopyRef());
            }

            entries[key] = new Entry
            {
                Exec = exec,
                InputHidden = pinnedHidden.CopyRef(),
                KeepAlive = keep,
                OwnedBlocks = owned,
                OwnedHostBuffers = new List<IntPtr>(ctx.DonatedHostBuffers),
                LastUse = ++useCounter,
                MaxAttendLen = maxAttendLen,
            };

            CudaDriverApi.cuGraphLaunch(exec, allocator.Stream.Handle).ThrowOnError();
            CaptureCount++;
            if (Log)
                Console.WriteLine($"[cuda-graph] captured {key} ({owned.Count} owned blocks)");
            return true;
        }

        /// <summary>
        /// Aborts an active capture after a failure (illegal sync, CUDA error).
        /// Pops the stream out of capture mode, releases quarantined blocks and
        /// blacklists the key so the shape never tries again.
        /// </summary>
        public void AbortCapture(string key) => AbortCapture(key, null);

        public void AbortCapture(string key, string reason)
        {
            if (Log && reason != null)
                Console.WriteLine($"[cuda-graph] abort reason: {reason}");
            if (captureContext == null)
                return;
            CudaGraphCapture.CaptureContext ctx = captureContext;
            allocator.Context.MakeCurrent();
            // End the (invalidated) capture to restore normal stream semantics.
            if (CudaDriverApi.cuStreamEndCapture(allocator.Stream.Handle, out IntPtr graph) == 0 && graph != IntPtr.Zero)
                CudaDriverApi.cuGraphDestroy(graph);
            CudaGraphCapture.End(ctx);
            captureContext = null;
            captureKey = null;
            ReleaseContextBlocks(ctx);
            blacklist.Add(key);
            if (Log)
                Console.WriteLine($"[cuda-graph] capture aborted for {key}");
        }

        private void ReleaseContextBlocks(CudaGraphCapture.CaptureContext ctx)
        {
            foreach ((IntPtr ptr, long bytes) in ctx.QuarantinedBlocks)
                allocator.ReturnDeviceMemory(ptr, bytes);
            foreach (IntPtr host in ctx.DonatedHostBuffers)
                CudaStorage.FreeDonatedHostBuffer(host);
            ctx.QuarantinedBlocks.Clear();
            ctx.DonatedHostBuffers.Clear();
        }

        private void EvictLru()
        {
            string lruKey = null;
            long lru = long.MaxValue;
            foreach (KeyValuePair<string, Entry> kv in entries)
            {
                if (kv.Value.LastUse < lru)
                {
                    lru = kv.Value.LastUse;
                    lruKey = kv.Key;
                }
            }
            if (lruKey != null)
                DisposeEntry(lruKey);
        }

        private void DisposeEntry(string key)
        {
            Entry entry = entries[key];
            entries.Remove(key);
            allocator.Context.MakeCurrent();
            // The graph may have a replay in flight; entries are only disposed
            // between forwards (each forward ends in a logits sync), so the
            // stream is idle with respect to this graph here.
            CudaDriverApi.cuGraphExecDestroy(entry.Exec);
            foreach (Tensor t in entry.KeepAlive)
                t.Dispose();
            entry.InputHidden.Dispose();
            foreach ((IntPtr ptr, long bytes) in entry.OwnedBlocks)
                allocator.ReturnDeviceMemory(ptr, bytes);
            foreach (IntPtr host in entry.OwnedHostBuffers)
                CudaStorage.FreeDonatedHostBuffer(host);
            if (Log)
                Console.WriteLine($"[cuda-graph] evicted {key}");
        }

        public void Dispose()
        {
            if (disposed)
                return;
            disposed = true;
            var keys = new List<string>(entries.Keys);
            foreach (string key in keys)
                DisposeEntry(key);
        }
    }
}
