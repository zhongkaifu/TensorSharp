using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    /// <summary>
    /// Multi-GPU collective operations using CUDA peer-to-peer memory access.
    /// Works on any platform with CUDA (Windows, Linux) without requiring NCCL.
    ///
    /// AllReduce where every pair of GPUs reads the other's memory from kernels (the self-test below verifies it): ordered
    /// on the devices, without the host waiting for any of them. Each rank's stream records that its partial is ready and
    /// waits for its peers'; every rank then sums all the partials, read in place over the peer mappings, in rank order
    /// ((p0 + p1) + p2 ...: the same additions on every rank, so every rank holds the same bits, the ones the
    /// reduce-to-zero below makes) into its own scratch; once every rank has read the others' partials (a second event a
    /// rank), each copies its sum over its partial. The host only enqueues: a tensor-parallel step's ops keep flowing to the
    /// devices across its reductions, where the reduce-to-zero waited for every device three times a reduction.
    ///
    /// Otherwise (a pair without verified peer access, no kernels) reduce-to-zero + broadcast:
    ///   1. Each non-zero GPU copies its partial to GPU 0's staging buffer
    ///   2. GPU 0 accumulates: buffer[0] += staging (via elementwise-add kernel)
    ///   3. GPU 0 broadcasts the reduced result to all other GPUs
    /// with the host waiting for every device before and after.
    /// </summary>
    internal sealed class CudaP2PCommunicator : IDisposable
    {
        private readonly CudaAllocator[] _allocators;
        private readonly int _worldSize;
        private readonly bool[] _p2pEnabled;

        // Staging buffer on GPU 0 for accumulation during AllReduce.
        private IntPtr _stagingBuffer;
        private long _stagingBufferBytes;

        // The device-ordered AllReduce: each rank's "partial ready" and "peers read" events and its sum's scratch.
        private readonly bool _deviceOrdered;
        private readonly IntPtr[] _ready, _summed, _scratch;
        private readonly long[] _scratchBytes;

        /// <summary>Tests: reduce through the host-synchronized reduce-to-zero even where peers read each other.</summary>
        internal bool HostSynchronizedForTests { get; set; }

        /// <summary>Whether AllReduce runs ordered on the devices (every pair's kernels read the peer's memory).</summary>
        internal bool DeviceOrdered => _deviceOrdered && !HostSynchronizedForTests;

        public CudaP2PCommunicator(CudaAllocator[] allocators)
        {
            _allocators = allocators ?? throw new ArgumentNullException(nameof(allocators));
            _worldSize = allocators.Length;

            if (_worldSize < 2)
                throw new ArgumentException("P2P communicator requires at least 2 GPUs.", nameof(allocators));

            _p2pEnabled = EnablePeerAccess(allocators);
            _deviceOrdered = Array.TrueForAll(allocators, a => a.Kernels != null);
            for (int i = 0; i < _worldSize; i++)
                for (int j = 0; j < _worldSize; j++)
                    _deviceOrdered &= i == j || _p2pEnabled[i * _worldSize + j];
            _ready = new IntPtr[_worldSize];
            _summed = new IntPtr[_worldSize];
            _scratch = new IntPtr[_worldSize];
            _scratchBytes = new long[_worldSize];
            if (!_deviceOrdered) return;
            const uint disableTiming = 0x2;   // CU_EVENT_DISABLE_TIMING
            for (int r = 0; r < _worldSize; r++)
            {
                allocators[r].Context.MakeCurrent();
                CudaDriverApi.cuEventCreate(out _ready[r], disableTiming).ThrowOnError();
                CudaDriverApi.cuEventCreate(out _summed[r], disableTiming).ThrowOnError();
            }
            Console.WriteLine($"  TP: AllReduce ordered on the {_worldSize} devices (peer reads, no host synchronization).");
        }

        public int WorldSize => _worldSize;

        // Device pointer to a tensor's storage. NOTE: Storage.PtrAtElement returns
        // a HOST pointer and marks the storage host-dirty (it's the "raw host write"
        // checkout), so it must NOT be used for device-to-device / peer copies —
        // doing so passes a host address where CUDA expects a device address
        // (CUDA_ERROR_INVALID_VALUE) and corrupts the tensor's dirty state,
        // clobbering the reduced result on the next EnsureDeviceCurrent.
        private static IntPtr DevicePtr(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(0);

        private static bool[] EnablePeerAccess(CudaAllocator[] allocators)
        {
            int n = allocators.Length;
            var enabled = new bool[n * n];

            // TENSORSHARP_TP_DISABLE_P2P=1 → leave every pair disabled so all
            // cross-GPU transfers stage through host memory, matching the
            // behaviour of no-peer hardware (A16 vGPU, consumer cards) exactly.
            if (CudaStorage.DisableP2P)
            {
                Console.WriteLine("  TP: P2P disabled by TENSORSHARP_TP_DISABLE_P2P; all cross-GPU transfers use host staging.");
                return enabled;
            }

            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (i == j) continue;

                    allocators[i].Context.MakeCurrent();
                    CudaDriverApi.cuDeviceCanAccessPeer(out int canAccess,
                        allocators[i].DeviceId, allocators[j].DeviceId).ThrowOnError();

                    if (canAccess == 1)
                    {
                        // cuCtxEnablePeerAccess returns CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED
                        // (704) if already enabled — treat that as success.
                        int result = CudaDriverApi.cuCtxEnablePeerAccess(allocators[j].Context.Handle, 0);
                        if (result == 0 || result == 704)
                            enabled[i * n + j] = true;
                        else
                            result.ThrowOnError();
                    }
                }
            }

            // Verify that P2P DMA actually round-trips correctly. Some platforms
            // (L4 behind certain PCIe switches, IOMMU-enabled hosts, constrained
            // BAR1) report canAccessPeer=1 and accept cuCtxEnablePeerAccess, yet
            // cuMemcpyPeerAsync silently transfers corrupt data. Write a known
            // pattern on GPU i, peer-copy to GPU j, read back and compare.
            // Failed pairs are demoted to host-staged transfers permanently.
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (i == j || !enabled[i * n + j]) continue;

                    if (!VerifyP2PRoundTrip(allocators, i, j))
                    {
                        Console.WriteLine(
                            $"  TP: P2P DMA self-test FAILED for GPU {i} → GPU {j} " +
                            $"(cuDeviceCanAccessPeer=1 but data is corrupt). " +
                            $"Falling back to host-staged transfers for this pair.");
                        // Demote BOTH directions. A pair whose DMA is corrupt one
                        // way is not trustworthy the other way either, and
                        // CudaStorage.MarkPeerAccessFailed (used by every ordinary
                        // cross-GPU tensor copy) already demotes the pair
                        // symmetrically — leaving this table asymmetric would let a
                        // collective peer-copy over a link that every other code
                        // path has already given up on.
                        enabled[i * n + j] = false;
                        enabled[j * n + i] = false;
                        CudaStorage.MarkPeerAccessFailed(
                            allocators[i].DeviceId, allocators[j].DeviceId);
                        // The contexts are the devices' primary ones, which ggml's CUDA backend shares: left
                        // enabled, its cross-device copies in this process kept using the broken mapping
                        // (MuseSpark's ggml_cuda layer split went wrong once a group had enabled it).
                        allocators[i].Context.MakeCurrent();
                        CudaDriverApi.cuCtxDisablePeerAccess(allocators[j].Context.Handle);
                        allocators[j].Context.MakeCurrent();
                        CudaDriverApi.cuCtxDisablePeerAccess(allocators[i].Context.Handle);
                    }
                }
            }

            return enabled;
        }

        /// <summary>
        /// Allocate a small buffer on GPU <paramref name="src"/>, fill it with a
        /// known pattern, peer-copy it to GPU <paramref name="dst"/>, read the
        /// result back to the host and verify every byte. Returns false when the
        /// round-trip corrupts data — the exact failure mode seen on some L4
        /// PCIe topologies where the driver reports P2P as available but the
        /// platform silently breaks the DMA.
        /// </summary>
        private static unsafe bool VerifyP2PRoundTrip(CudaAllocator[] allocators, int src, int dst)
        {
            const int testBytes = 4096;
            IntPtr srcBuf = IntPtr.Zero;
            IntPtr dstBuf = IntPtr.Zero;

            try
            {
                // Allocate on source GPU and write a recognisable pattern.
                allocators[src].Context.MakeCurrent();
                CudaDriverApi.cuMemAlloc(out srcBuf, new UIntPtr(testBytes)).ThrowOnError();

                var pattern = new byte[testBytes];
                for (int k = 0; k < testBytes; k++)
                    pattern[k] = (byte)((k * 7 + src * 31 + dst * 17) & 0xFF);

                fixed (byte* p = pattern)
                    CudaDriverApi.cuMemcpyHtoD(srcBuf, (IntPtr)p, new UIntPtr(testBytes)).ThrowOnError();

                // Allocate on destination GPU (zero-fill so a failed copy is visible).
                allocators[dst].Context.MakeCurrent();
                CudaDriverApi.cuMemAlloc(out dstBuf, new UIntPtr(testBytes)).ThrowOnError();
                CudaDriverApi.cuMemsetD8(dstBuf, 0, new UIntPtr(testBytes)).ThrowOnError();

                // Peer copy: src GPU → dst GPU, enqueued on dst's stream.
                CudaDriverApi.cuMemcpyPeerAsync(
                    dstBuf, allocators[dst].Context.Handle,
                    srcBuf, allocators[src].Context.Handle,
                    new UIntPtr(testBytes),
                    allocators[dst].Stream.Handle).ThrowOnError();

                allocators[dst].Stream.Synchronize();

                // Read back and compare.
                var readback = new byte[testBytes];
                fixed (byte* r = readback)
                    CudaDriverApi.cuMemcpyDtoH((IntPtr)r, dstBuf, new UIntPtr(testBytes)).ThrowOnError();

                for (int k = 0; k < testBytes; k++)
                {
                    if (readback[k] != pattern[k])
                        return false;
                }

                return KernelReadsPeer(allocators, src, dst, srcBuf, dstBuf, testBytes / sizeof(float));
            }
            catch
            {
                // Any CUDA error during the self-test → treat P2P as broken.
                return false;
            }
            finally
            {
                if (srcBuf != IntPtr.Zero)
                {
                    allocators[src].Context.MakeCurrent();
                    CudaDriverApi.cuMemFree(srcBuf);
                }
                if (dstBuf != IntPtr.Zero)
                {
                    allocators[dst].Context.MakeCurrent();
                    CudaDriverApi.cuMemFree(dstBuf);
                }
            }
        }

        /// <summary>
        /// A kernel on <paramref name="dst"/> reading <paramref name="src"/>'s memory through the peer mapping
        /// (SM loads, not the copy engine) sees what is there: <paramref name="dstBuf"/> + <paramref name="srcBuf"/>
        /// into <paramref name="dstBuf"/>, which holds a copy of <paramref name="srcBuf"/>, must double every value.
        /// A virtualized PCIe switch can grant peer access while dropping peer traffic: on a 2x A40 VM (PXB) these
        /// loads read NaN every time and peer copies lost whole buffers in a third of runs (fresh buffers, first use),
        /// which the copy test above passes. True without the kernels (nothing to run).
        /// </summary>
        private static unsafe bool KernelReadsPeer(CudaAllocator[] allocators, int src, int dst, IntPtr srcBuf, IntPtr dstBuf, int floats)
        {
            CudaKernels kernels = allocators[dst].Kernels;
            if (kernels == null)
                return true;
            var values = new float[floats];
            for (int k = 0; k < floats; k++)
                values[k] = (k % 251) * 0.25f + src + 1;
            allocators[src].Context.MakeCurrent();
            fixed (float* v = values)
                CudaDriverApi.cuMemcpyHtoD(srcBuf, (IntPtr)v, new UIntPtr((ulong)floats * sizeof(float))).ThrowOnError();
            allocators[dst].Context.MakeCurrent();
            fixed (float* v = values)
                CudaDriverApi.cuMemcpyHtoD(dstBuf, (IntPtr)v, new UIntPtr((ulong)floats * sizeof(float))).ThrowOnError();
            kernels.LaunchBinaryF32(dstBuf, srcBuf, dstBuf, floats, 0, allocators[dst].Stream.Handle);   // op 0: add
            allocators[dst].Stream.Synchronize();
            var readback = new float[floats];
            fixed (float* r = readback)
                CudaDriverApi.cuMemcpyDtoH((IntPtr)r, dstBuf, new UIntPtr((ulong)floats * sizeof(float))).ThrowOnError();
            for (int k = 0; k < floats; k++)
                if (readback[k] != 2 * values[k])
                    return false;
            return true;
        }

        /// <summary>
        /// True when a peer DMA moving data FROM GPU <paramref name="from"/> TO GPU
        /// <paramref name="to"/> is known good. Matches the direction convention of
        /// <see cref="VerifyP2PRoundTrip"/> (which writes on <c>src</c> and reads
        /// back on <c>dst</c>) and of <see cref="CudaStorage.CopyDeviceFrom"/>.
        /// Callers must pass the direction the BYTES travel, not the direction of
        /// the context that happens to enqueue the copy.
        /// </summary>
        private bool CanAccessPeer(int from, int to) => _p2pEnabled[from * _worldSize + to];

        private void EnsureStagingBuffer(long requiredBytes)
        {
            if (_stagingBufferBytes >= requiredBytes)
                return;

            if (_stagingBuffer != IntPtr.Zero)
            {
                _allocators[0].Context.MakeCurrent();
                CudaDriverApi.cuMemFree(_stagingBuffer);
            }

            _allocators[0].Context.MakeCurrent();
            CudaDriverApi.cuMemAlloc(out _stagingBuffer, new UIntPtr((ulong)requiredBytes)).ThrowOnError();
            _stagingBufferBytes = requiredBytes;
        }

        /// <summary>
        /// In-place AllReduce (sum) across all GPUs. After this call, every
        /// tensor in <paramref name="tensors"/> holds the element-wise sum of
        /// all input values. tensors[i] must reside on GPU i.
        /// </summary>
        public void AllReduce(Tensor[] tensors)
        {
            if (tensors == null || tensors.Length != _worldSize)
                throw new ArgumentException($"Expected {_worldSize} tensors, got {tensors?.Length ?? 0}.");

            long byteCount = tensors[0].Storage.ByteLength;
            int elementCount = (int)tensors[0].Storage.ElementCount;

            // Flush any pending HOST writes to the device before we read the raw
            // device pointers below. A partial that was accumulated host-side (any
            // GetFloatPtr/PtrAtElement checkout marks the storage host-dirty) still
            // has a STALE device buffer until this runs, and we would reduce the
            // stale contents. Mirrors CudaStorage.CopyDeviceFrom, which does the
            // same before a cross-device copy. The upload is enqueued on the rank's
            // stream, ahead of the reduction on either path.
            for (int i = 0; i < _worldSize; i++)
                ((CudaStorage)tensors[i].Storage).EnsureDeviceCurrent();

            if (DeviceOrdered)
            {
                AllReduceOnDevices(tensors, byteCount, elementCount);
                return;
            }

            // Synchronize all GPUs to ensure partial results are complete
            // (this also flushes the async HtoD uploads issued just above).
            for (int i = 0; i < _worldSize; i++)
                _allocators[i].Synchronize();

            // Phase 1: Reduce to GPU 0.
            _allocators[0].Context.MakeCurrent();
            for (int i = 1; i < _worldSize; i++)
            {
                IntPtr srcPtr = DevicePtr(tensors[i]);
                IntPtr dstPtr = DevicePtr(tensors[0]);

                // The bytes travel FROM GPU i TO GPU 0, so that is the direction to
                // vet. This used to ask CanAccessPeer(0, i) — the opposite link.
                // On a host where only one direction of a pair is corrupt (the
                // self-test above flags exactly that) the reduce then ran its
                // cuMemcpyPeerAsync over the known-bad direction while consulting
                // the healthy one, silently summing garbage into rank 0 on every
                // row-parallel layer.
                if (CanAccessPeer(i, 0) && _allocators[0].Kernels != null)
                {
                    // P2P copy: GPU i → GPU 0 staging, then add.
                    EnsureStagingBuffer(byteCount);
                    CudaDriverApi.cuMemcpyPeerAsync(
                        _stagingBuffer, _allocators[0].Context.Handle,
                        srcPtr, _allocators[i].Context.Handle,
                        new UIntPtr((ulong)byteCount),
                        _allocators[0].Stream.Handle).ThrowOnError();

                    _allocators[0].Kernels.LaunchBinaryF32(
                        dstPtr, _stagingBuffer, dstPtr,
                        elementCount, 0, // op=0 → Add
                        _allocators[0].Stream.Handle);
                }
                else
                {
                    // Fallback: stage through host memory.
                    AllReduceViaHost(tensors[0], tensors[i], elementCount, byteCount);
                }
            }

            _allocators[0].Synchronize();

            // Phase 2: Broadcast from GPU 0 to all other GPUs.
            IntPtr resultPtr = DevicePtr(tensors[0]);
            for (int i = 1; i < _worldSize; i++)
            {
                _allocators[i].Context.MakeCurrent();
                IntPtr dstPtr = DevicePtr(tensors[i]);

                // Bytes travel FROM GPU 0 TO GPU i here (the mirror image of the
                // reduce above), so vet that direction.
                if (CanAccessPeer(0, i))
                {
                    CudaDriverApi.cuMemcpyPeerAsync(
                        dstPtr, _allocators[i].Context.Handle,
                        resultPtr, _allocators[0].Context.Handle,
                        new UIntPtr((ulong)byteCount),
                        _allocators[i].Stream.Handle).ThrowOnError();
                }
                else
                {
                    BroadcastViaHost(tensors[0], tensors[i], byteCount);
                }
            }

            // Final sync so all GPUs have the reduced result.
            for (int i = 0; i < _worldSize; i++)
                _allocators[i].Synchronize();

            // Every tensor's DEVICE buffer was just rewritten through raw pointers,
            // which bypasses the storage's dirty tracking. Without this the flags
            // still claim host and device agree (EnsureDeviceCurrent clears BOTH),
            // so the next SyncHostFromDevice short-circuits on !deviceDirty and
            // hands back the pre-AllReduce host contents — silently discarding the
            // reduced result. Mark device-authoritative so host reads re-fetch.
            for (int i = 0; i < _worldSize; i++)
                ((CudaStorage)tensors[i].Storage).MarkDeviceModified();
        }

        /// <summary>The device-ordered AllReduce (see the class summary): event waits between the ranks' streams, every rank
        /// summing every partial in rank order into its scratch, then over its partial once its peers have read it.</summary>
        private void AllReduceOnDevices(Tensor[] tensors, long byteCount, int elementCount)
        {
            var pointers = new IntPtr[_worldSize];
            for (int r = 0; r < _worldSize; r++)
            {
                if (tensors[r].Storage.ByteLength != byteCount)
                    throw new ArgumentException($"AllReduce partials differ in size: rank {r} has {tensors[r].Storage.ByteLength} bytes, rank 0 {byteCount}.");
                pointers[r] = DevicePtr(tensors[r]);
            }
            // Every partial is ready on its own stream.
            for (int r = 0; r < _worldSize; r++)
            {
                _allocators[r].Context.MakeCurrent();
                CudaDriverApi.cuEventRecord(_ready[r], _allocators[r].Stream.Handle).ThrowOnError();
            }
            // Each rank waits for its peers' partials and sums them all, in rank order, into its scratch.
            for (int r = 0; r < _worldSize; r++)
            {
                CudaAllocator a = _allocators[r];
                a.Context.MakeCurrent();
                IntPtr stream = a.Stream.Handle;
                for (int k = 0; k < _worldSize; k++)
                    if (k != r) CudaDriverApi.cuStreamWaitEvent(stream, _ready[k], 0).ThrowOnError();
                IntPtr scratch = Scratch(r, byteCount);
                a.Kernels.LaunchBinaryF32(pointers[0], pointers[1], scratch, elementCount, 0, stream);   // op 0: add
                for (int k = 2; k < _worldSize; k++)
                    a.Kernels.LaunchBinaryF32(scratch, pointers[k], scratch, elementCount, 0, stream);
                CudaDriverApi.cuEventRecord(_summed[r], stream).ThrowOnError();
            }
            // Once every rank has read the others' partials, each writes its sum over its own.
            for (int r = 0; r < _worldSize; r++)
            {
                CudaAllocator a = _allocators[r];
                a.Context.MakeCurrent();
                IntPtr stream = a.Stream.Handle;
                for (int k = 0; k < _worldSize; k++)
                    if (k != r) CudaDriverApi.cuStreamWaitEvent(stream, _summed[k], 0).ThrowOnError();
                CudaDriverApi.cuMemcpyDtoDAsync(pointers[r], _scratch[r], new UIntPtr((ulong)byteCount), stream).ThrowOnError();
            }
            // The device buffers were written through raw pointers (see the host-synchronized path's note).
            for (int r = 0; r < _worldSize; r++)
                ((CudaStorage)tensors[r].Storage).MarkDeviceModified();
        }

        /// <summary>Rank <paramref name="rank"/>'s scratch of at least <paramref name="bytes"/>. A smaller one is freed once the
        /// rank's stream has finished with it (its last sum may still be writing it).</summary>
        private IntPtr Scratch(int rank, long bytes)
        {
            if (_scratchBytes[rank] >= bytes) return _scratch[rank];
            CudaAllocator a = _allocators[rank];
            if (_scratch[rank] != IntPtr.Zero)
            {
                a.Stream.Synchronize();
                CudaDriverApi.cuMemFree(_scratch[rank]);
            }
            // Rounded up to a MiB so a prompt chunk's growing partials do not reallocate every step.
            long size = (bytes + (1L << 20) - 1) & ~((1L << 20) - 1);
            CudaDriverApi.cuMemAlloc(out _scratch[rank], new UIntPtr((ulong)size)).ThrowOnError();
            _scratchBytes[rank] = size;
            return _scratch[rank];
        }

        private unsafe void AllReduceViaHost(Tensor dst, Tensor src, int elementCount, long byteCount)
        {
            // Copy src (GPU i) → host, copy dst (GPU 0) → host, add on host,
            // then upload result back to GPU 0.  When CUDA kernels are
            // unavailable we cannot use LaunchBinaryF32 on device, so the
            // entire reduce happens on the CPU.
            int srcDevice = ((CudaStorage)src.Storage).DeviceId;
            int dstDevice = ((CudaStorage)dst.Storage).DeviceId;
            var srcBuf = new float[elementCount];
            var dstBuf = new float[elementCount];
            fixed (float* srcPtr = srcBuf)
            fixed (float* dstPtr = dstBuf)
            {
                _allocators[srcDevice].Context.MakeCurrent();
                CudaDriverApi.cuMemcpyDtoH((IntPtr)srcPtr, DevicePtr(src),
                    new UIntPtr((ulong)byteCount)).ThrowOnError();

                _allocators[dstDevice].Context.MakeCurrent();
                CudaDriverApi.cuMemcpyDtoH((IntPtr)dstPtr, DevicePtr(dst),
                    new UIntPtr((ulong)byteCount)).ThrowOnError();

                for (int j = 0; j < elementCount; j++)
                    dstBuf[j] += srcBuf[j];

                _allocators[0].Context.MakeCurrent();
                CudaDriverApi.cuMemcpyHtoD(DevicePtr(dst), (IntPtr)dstPtr,
                    new UIntPtr((ulong)byteCount)).ThrowOnError();
            }
        }

        private unsafe void BroadcastViaHost(Tensor src, Tensor dst, long byteCount)
        {
            int elementCount = (int)src.Storage.ElementCount;
            var hostBuf = new float[elementCount];
            fixed (float* hostPtr = hostBuf)
            {
                _allocators[0].Context.MakeCurrent();
                CudaDriverApi.cuMemcpyDtoH((IntPtr)hostPtr, DevicePtr(src),
                    new UIntPtr((ulong)byteCount)).ThrowOnError();

                int dstDevice = dst.Storage.Allocator.DeviceId;
                _allocators[dstDevice].Context.MakeCurrent();
                CudaDriverApi.cuMemcpyHtoD(DevicePtr(dst), (IntPtr)hostPtr,
                    new UIntPtr((ulong)byteCount)).ThrowOnError();
            }
        }

        public void Dispose()
        {
            if (_stagingBuffer != IntPtr.Zero)
            {
                _allocators[0].Context.MakeCurrent();
                CudaDriverApi.cuMemFree(_stagingBuffer);
                _stagingBuffer = IntPtr.Zero;
                _stagingBufferBytes = 0;
            }
            for (int r = 0; r < _worldSize; r++)
            {
                if (_ready[r] == IntPtr.Zero && _scratch[r] == IntPtr.Zero) continue;
                _allocators[r].Context.MakeCurrent();
                _allocators[r].Stream.Synchronize();
                if (_scratch[r] != IntPtr.Zero) CudaDriverApi.cuMemFree(_scratch[r]);
                if (_ready[r] != IntPtr.Zero) CudaDriverApi.cuEventDestroy(_ready[r]);
                if (_summed[r] != IntPtr.Zero) CudaDriverApi.cuEventDestroy(_summed[r]);
                _scratch[r] = _ready[r] = _summed[r] = IntPtr.Zero;
                _scratchBytes[r] = 0;
            }
        }
    }
}
